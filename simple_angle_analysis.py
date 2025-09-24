#!/usr/bin/env python3
"""
# Simple ISM330DHCX Angle Detection Analysis

A simplified analysis tool that works with basic Python libraries to analyze
angle detection capabilities of the ISM330DHCX sensor.

## Features:
- Load test configurations and raw sensor data
- Apply motion estimator algorithms for angle detection
- Analyze minimal detectable angles based on DPS thresholds
- Generate analysis reports

## Usage:
```python
# Load test data
analyzer = SimpleAngleAnalyzer()
analyzer.load_test_data("test_config_altitude.json", "raw_data_output_altitude.csv")

# Analyze angle detection
results = analyzer.analyze_angle_detection()

# Generate report
report = analyzer.generate_analysis_report()
print(report)
```
"""

import json
import csv
import math
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Import our motion estimator
from simple_motion_estimator import MotionEstimator

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TestSequence:
    """Individual test sequence data"""
    test_id: int
    duration: float
    change_time: float
    description: str
    power_state: str
    theta_change: float = 0.0
    phi_change: float = 0.0
    psi_change: float = 0.0
    expected_detectable: bool = False

@dataclass
class SensorSpecs:
    """Sensor specifications from test config"""
    accel_range: int
    gyro_range: int
    odr: int
    noise_floor: float

@dataclass
class InitialConditions:
    """Initial conditions for the test"""
    imu_offset: float
    theta_start: float
    phi_start: float
    psi_start: float

@dataclass
class AngleDetectionResult:
    """Result of angle detection analysis"""
    test_id: int
    angle_type: str  # 'theta', 'phi', 'psi'
    expected_angle: float
    detected_angle: float
    detection_time: float
    is_detected: bool
    confidence: float
    dps_used: float
    angle_magnitude: float

# ============================================================================
# Main Analysis Class
# ============================================================================

class SimpleAngleAnalyzer:
    """
    Simplified angle detection analysis tool for ISM330DHCX sensor.
    
    This class provides functionality to:
    - Load test configurations and raw sensor data
    - Apply motion estimator algorithms for angle detection
    - Analyze minimal detectable angles based on DPS thresholds
    - Generate analysis reports
    """
    
    def __init__(self):
        """Initialize the analyzer with default settings"""
        self.motion_estimator = MotionEstimator()
        self.test_sequences: List[TestSequence] = []
        self.sensor_specs: Optional[SensorSpecs] = None
        self.initial_conditions: Optional[InitialConditions] = None
        self.raw_data: List[Dict] = []
        self.detection_results: List[AngleDetectionResult] = []
        
        # Detection parameters (can be tuned)
        self.dps_thresholds = {
            'theta': 0.5,  # degrees per second for pitch
            'phi': 0.5,    # degrees per second for yaw
            'psi': 0.5     # degrees per second for roll
        }
        
        # Angle detection thresholds
        self.angle_thresholds = {
            'theta': 1.0,  # degrees
            'phi': 1.0,    # degrees
            'psi': 1.0     # degrees
        }
        
    # ========================================================================
    # Data Loading Functions
    # ========================================================================
    
    def load_test_data(self, config_file: str, data_file: str) -> bool:
        """
        Load test configuration and raw sensor data.
        
        Args:
            config_file: Path to test configuration JSON file
            data_file: Path to raw sensor data CSV file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load test configuration
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Parse sensor specifications
            self.sensor_specs = SensorSpecs(**config_data['sensor_specs'])
            
            # Parse initial conditions
            self.initial_conditions = InitialConditions(**config_data['initial_conditions'])
            
            # Parse test sequence
            self.test_sequences = []
            for test in config_data['test_sequence']:
                test_seq = TestSequence(
                    test_id=test['test_id'],
                    duration=test['duration'],
                    change_time=test['change_time'],
                    description=test['description'],
                    power_state=test['power_state'],
                    theta_change=test.get('theta_change', 0.0),
                    phi_change=test.get('phi_change', 0.0),
                    psi_change=test.get('psi_change', 0.0),
                    expected_detectable=test.get('expected_detectable', False)
                )
                self.test_sequences.append(test_seq)
            
            # Load raw sensor data
            self.raw_data = []
            with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row_num, row in enumerate(reader):
                    try:
                        # Convert string values to float, handling potential encoding issues
                        data_point = {
                            'timestamp': float(row['timestamp'].strip()),
                            'accel_x': float(row['accel_x'].strip()),
                            'accel_y': float(row['accel_y'].strip()),
                            'accel_z': float(row['accel_z'].strip()),
                            'gyro_x': float(row['gyro_x'].strip()),
                            'gyro_y': float(row['gyro_y'].strip()),
                            'gyro_z': float(row['gyro_z'].strip())
                        }
                        self.raw_data.append(data_point)
                    except (ValueError, KeyError) as e:
                        print(f"⚠ Skipping row {row_num + 2}: {e}")
                        continue
            
            # Process timestamps - convert to seconds if needed
            if self.raw_data:
                max_timestamp = max(point['timestamp'] for point in self.raw_data)
                if max_timestamp > 1000000:  # Likely microseconds
                    for point in self.raw_data:
                        point['timestamp'] = point['timestamp'] / 1000000.0
                elif max_timestamp > 1000:  # Likely milliseconds
                    for point in self.raw_data:
                        point['timestamp'] = point['timestamp'] / 1000.0
            
            print(f"✓ Loaded {len(self.test_sequences)} test sequences")
            print(f"✓ Loaded {len(self.raw_data)} sensor data points")
            print(f"✓ Sensor ODR: {self.sensor_specs.odr} Hz")
            if self.raw_data:
                duration = self.raw_data[-1]['timestamp'] - self.raw_data[0]['timestamp']
                print(f"✓ Data duration: {duration:.1f} seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading test data: {e}")
            return False
    
    # ========================================================================
    # Angle Detection Functions
    # ========================================================================
    
    def detect_angle_changes_in_data(self, data: List[Dict], angle_type: str) -> List[Dict]:
        """
        Detect angle changes using motion estimator algorithms.
        
        Args:
            data: List of sensor data dictionaries
            angle_type: Type of angle to detect ('theta', 'phi', 'psi')
            
        Returns:
            List of detected angle changes with timestamps and magnitudes
        """
        detections = []
        
        # Initialize motion estimator state
        self.motion_estimator = MotionEstimator()
        
        # Process data through motion estimator
        angles = []
        timestamps = []
        
        for data_point in data:
            # Prepare sensor data
            accel = [data_point['accel_x'], data_point['accel_y'], data_point['accel_z']]
            gyro = [data_point['gyro_x'], data_point['gyro_y'], data_point['gyro_z']]
            timestamp = data_point['timestamp']
            
            # Update motion estimator
            result = self.motion_estimator.update(accel, gyro, timestamp)
            
            # Extract angle based on type
            if angle_type == 'theta':
                angle = result['pitch_deg']  # Theta corresponds to pitch
            elif angle_type == 'phi':
                angle = result['yaw_deg']    # Phi corresponds to yaw
            elif angle_type == 'psi':
                angle = result['roll_deg']   # Psi corresponds to roll
            else:
                continue
                
            angles.append(angle)
            timestamps.append(timestamp)
        
        # Calculate angular velocity (DPS)
        if len(angles) > 1:
            for i in range(1, len(angles)):
                dt = timestamps[i] - timestamps[i-1]
                if dt > 0:  # Avoid division by zero
                    angular_velocity = abs(angles[i] - angles[i-1]) / dt
                    
                    # Detect significant changes based on DPS threshold
                    threshold = self.dps_thresholds[angle_type]
                    if angular_velocity > threshold:
                        detection = {
                            'timestamp': timestamps[i],
                            'angle': angles[i],
                            'angle_change': angles[i] - angles[i-1],
                            'dps': angular_velocity,
                            'confidence': min(1.0, angular_velocity / (threshold * 2))
                        }
                        detections.append(detection)
        
        return detections
    
    def analyze_angle_detection(self) -> List[AngleDetectionResult]:
        """
        Analyze angle detection capabilities across all test sequences.
        
        Returns:
            List of angle detection results
        """
        if not self.raw_data or len(self.test_sequences) == 0:
            print("✗ No test data loaded")
            return []
        
        results = []
        
        # Calculate total duration
        if self.raw_data:
            total_duration = self.raw_data[-1]['timestamp'] - self.raw_data[0]['timestamp']
            print(f"Analyzing {len(self.test_sequences)} test sequences")
            print(f"Total data duration: {total_duration:.1f} seconds")
        
        # Process each test sequence
        for test_seq in self.test_sequences:
            # Calculate time window for this test
            test_start_time = test_seq.change_time
            test_end_time = test_start_time + test_seq.duration
            
            # Extract data for this test
            test_data = [
                point for point in self.raw_data
                if test_start_time <= point['timestamp'] <= test_end_time
            ]
            
            if len(test_data) == 0:
                print(f"⚠ No data found for test {test_seq.test_id}")
                continue
            
            print(f"Test {test_seq.test_id}: {len(test_data)} data points")
            
            # Reset motion estimator for each test
            self.motion_estimator = MotionEstimator()
            
            # Analyze each angle type
            for angle_type in ['theta', 'phi', 'psi']:
                expected_change = getattr(test_seq, f'{angle_type}_change')
                
                if abs(expected_change) > 0.1:  # Only analyze if there's an expected change
                    print(f"  Analyzing {angle_type}: expected {expected_change:.1f}°")
                    
                    # Detect angle changes
                    detections = self.detect_angle_changes_in_data(test_data, angle_type)
                    
                    # Find the most significant detection
                    if detections:
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        
                        result = AngleDetectionResult(
                            test_id=test_seq.test_id,
                            angle_type=angle_type,
                            expected_angle=expected_change,
                            detected_angle=best_detection['angle_change'],
                            detection_time=best_detection['timestamp'],
                            is_detected=abs(best_detection['angle_change']) >= self.angle_thresholds[angle_type],
                            confidence=best_detection['confidence'],
                            dps_used=best_detection['dps'],
                            angle_magnitude=abs(expected_change)
                        )
                        results.append(result)
                        print(f"    Detected: {best_detection['angle_change']:.2f}° (confidence: {best_detection['confidence']:.3f})")
                    else:
                        # No detection found
                        result = AngleDetectionResult(
                            test_id=test_seq.test_id,
                            angle_type=angle_type,
                            expected_angle=expected_change,
                            detected_angle=0.0,
                            detection_time=0.0,
                            is_detected=False,
                            confidence=0.0,
                            dps_used=0.0,
                            angle_magnitude=abs(expected_change)
                        )
                        results.append(result)
                        print(f"    No detection found")
        
        self.detection_results = results
        print(f"✓ Generated {len(results)} detection results")
        return results
    
    # ========================================================================
    # Analysis and Reporting Functions
    # ========================================================================
    
    def find_minimal_detectable_angles(self) -> Dict[str, float]:
        """
        Find minimal detectable angles for each angle type.
        
        Returns:
            Dictionary with minimal detectable angles for each type
        """
        minimal_angles = {}
        
        for angle_type in ['theta', 'phi', 'psi']:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            detected_results = [r for r in type_results if r.is_detected]
            
            if detected_results:
                # Find the smallest detected angle
                minimal_angle = min(r.angle_magnitude for r in detected_results)
                minimal_angles[angle_type] = minimal_angle
            else:
                minimal_angles[angle_type] = float('inf')
        
        return minimal_angles
    
    def analyze_dps_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze optimal DPS thresholds for each angle type.
        
        Returns:
            Dictionary with DPS analysis for each angle type
        """
        dps_analysis = {}
        
        for angle_type in ['theta', 'phi', 'psi']:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            
            if type_results:
                detected_results = [r for r in type_results if r.is_detected]
                missed_results = [r for r in type_results if not r.is_detected]
                
                analysis = {
                    'current_threshold': self.dps_thresholds[angle_type],
                    'detected_count': len(detected_results),
                    'missed_count': len(missed_results),
                    'accuracy': len(detected_results) / len(type_results) if type_results else 0
                }
                
                if detected_results:
                    dps_values = [r.dps_used for r in detected_results]
                    analysis['avg_dps_detected'] = sum(dps_values) / len(dps_values)
                    analysis['min_dps_detected'] = min(dps_values)
                    analysis['max_dps_detected'] = max(dps_values)
                
                if missed_results:
                    dps_values = [r.dps_used for r in missed_results]
                    analysis['avg_dps_missed'] = sum(dps_values) / len(dps_values)
                
                dps_analysis[angle_type] = analysis
        
        return dps_analysis
    
    def generate_analysis_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.detection_results:
            return "No detection results available for report generation."
        
        report = []
        report.append("=" * 80)
        report.append("ISM330DHCX ANGLE DETECTION ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall statistics
        total_tests = len(self.detection_results)
        detected_tests = sum(1 for r in self.detection_results if r.is_detected)
        overall_accuracy = detected_tests / total_tests if total_tests > 0 else 0
        
        report.append(f"Overall Detection Accuracy: {overall_accuracy:.1%} ({detected_tests}/{total_tests})")
        report.append("")
        
        # Analysis by angle type
        angle_types = ['theta', 'phi', 'psi']
        for angle_type in angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            if type_results:
                detected = sum(1 for r in type_results if r.is_detected)
                total = len(type_results)
                accuracy = detected / total if total > 0 else 0
                
                # Find minimal detectable angle
                detected_results = [r for r in type_results if r.is_detected]
                if detected_results:
                    minimal_angle = min(r.angle_magnitude for r in detected_results)
                    confidences = [r.confidence for r in detected_results]
                    avg_confidence = sum(confidences) / len(confidences)
                    dps_values = [r.dps_used for r in detected_results]
                    avg_dps = sum(dps_values) / len(dps_values)
                else:
                    minimal_angle = float('inf')
                    avg_confidence = 0
                    avg_dps = 0
                
                report.append(f"{angle_type.upper()} Analysis:")
                report.append(f"  Accuracy: {accuracy:.1%} ({detected}/{total})")
                report.append(f"  Minimal Detectable Angle: {minimal_angle:.2f}°" if minimal_angle != float('inf') else "  Minimal Detectable Angle: Not detected")
                report.append(f"  Average Confidence: {avg_confidence:.3f}")
                report.append(f"  Average DPS Used: {avg_dps:.2f} deg/s")
                report.append("")
        
        # DPS threshold analysis
        report.append("DPS Threshold Analysis:")
        for angle_type in angle_types:
            threshold = self.dps_thresholds[angle_type]
            report.append(f"  {angle_type.upper()}: {threshold:.1f} deg/s")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 40)
        
        # Find the most reliable angle type
        best_angle_type = max(angle_types, key=lambda t: 
            sum(1 for r in self.detection_results if r.angle_type == t and r.is_detected) / 
            max(1, sum(1 for r in self.detection_results if r.angle_type == t)))
        
        report.append(f"• Most reliable angle type: {best_angle_type.upper()}")
        
        # Suggest optimal DPS thresholds
        for angle_type in angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            if type_results:
                detected_results = [r for r in type_results if r.is_detected]
                if detected_results:
                    dps_values = [r.dps_used for r in detected_results]
                    avg_dps = sum(dps_values) / len(dps_values)
                    report.append(f"• Optimal {angle_type.upper()} DPS threshold: {avg_dps:.2f} deg/s")
        
        # Minimal detectable angles
        minimal_angles = self.find_minimal_detectable_angles()
        report.append("")
        report.append("Minimal Detectable Angles:")
        for angle_type, angle in minimal_angles.items():
            if angle != float('inf'):
                report.append(f"• {angle_type.upper()}: {angle:.2f}°")
            else:
                report.append(f"• {angle_type.upper()}: Not detected in any test")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "ism330dhcx_analysis_results.json") -> bool:
        """
        Save detection results to JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert results to serializable format
            results_data = []
            for result in self.detection_results:
                results_data.append({
                    'test_id': result.test_id,
                    'angle_type': result.angle_type,
                    'expected_angle': result.expected_angle,
                    'detected_angle': result.detected_angle,
                    'detection_time': result.detection_time,
                    'is_detected': result.is_detected,
                    'confidence': result.confidence,
                    'dps_used': result.dps_used,
                    'angle_magnitude': result.angle_magnitude
                })
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"✓ Results saved to {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving results: {e}")
            return False

# ============================================================================
# Example Usage and Testing Functions
# ============================================================================

def main():
    """Main function for testing the analyzer"""
    print("ISM330DHCX Angle Detection Analysis Tool")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SimpleAngleAnalyzer()
    
    # Load test data (example files)
    config_files = [
        "test_config_altitude.json",
        "test_config_azimuth.json"
    ]
    
    data_files = [
        "raw_data_output_altitude.csv",
        "raw_data_output_azimuth.csv"
    ]
    
    for config_file, data_file in zip(config_files, data_files):
        if os.path.exists(config_file) and os.path.exists(data_file):
            print(f"\nAnalyzing {config_file} with {data_file}")
            print("-" * 40)
            
            # Load data
            if analyzer.load_test_data(config_file, data_file):
                # Analyze angle detection
                results = analyzer.analyze_angle_detection()
                
                # Generate and print report
                report = analyzer.generate_analysis_report()
                print(report)
                
                # Save results
                analyzer.save_results(f"results_{os.path.splitext(config_file)[0]}.json")
            else:
                print(f"✗ Failed to load {config_file} or {data_file}")
        else:
            print(f"✗ Files not found: {config_file} or {data_file}")

if __name__ == "__main__":
    main()