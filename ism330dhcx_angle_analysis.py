#!/usr/bin/env python3
"""
# ISM330DHCX Angle Detection Analysis Suite

This notebook-style analysis tool provides comprehensive functions for analyzing
angle detection capabilities of the ISM330DHCX sensor using the motion estimator
algorithms. The tool processes test configuration files and raw sensor data to
determine minimal detectable angles based on DPS thresholds.

## Key Features:
- Load and parse test_config_*.json and raw_data_output_*.csv files
- Apply motion estimator algorithms for angle detection
- Visualize detected angles with color-coded regions (theta, phi, psi)
- Analyze minimal detectable angles based on DPS thresholds
- Generate comprehensive analysis reports

## Usage:
```python
# Load test data
analyzer = ISM330DHCAnalyzer()
analyzer.load_test_data("test_config_altitude.json", "raw_data_output_altitude.csv")

# Analyze angle detection
results = analyzer.analyze_angle_detection()

# Visualize results
analyzer.create_angle_visualization()

# Generate report
report = analyzer.generate_analysis_report()
print(report)
```
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our motion estimator
from motion_estimator import MotionEstimator

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

class ISM330DHCAnalyzer:
    """
    Comprehensive angle detection analysis tool for ISM330DHCX sensor.
    
    This class provides functionality to:
    - Load test configurations and raw sensor data
    - Apply motion estimator algorithms for angle detection
    - Visualize detected angles with color-coded regions
    - Analyze minimal detectable angles based on DPS thresholds
    - Generate comprehensive analysis reports
    """
    
    def __init__(self):
        """Initialize the analyzer with default settings"""
        self.motion_estimator = MotionEstimator()
        self.test_sequences: List[TestSequence] = []
        self.sensor_specs: Optional[SensorSpecs] = None
        self.initial_conditions: Optional[InitialConditions] = None
        self.raw_data: Optional[pd.DataFrame] = None
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
        
        # Setup plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
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
            self.raw_data = pd.read_csv(data_file)
            
            # Process timestamps - convert to seconds if needed
            if 'timestamp' in self.raw_data.columns:
                # Check if timestamps are in microseconds
                if self.raw_data['timestamp'].max() > 1000000:
                    self.raw_data['timestamp'] = self.raw_data['timestamp'] / 1000000.0
                else:
                    self.raw_data['timestamp'] = self.raw_data['timestamp'] / 1000.0
            
            print(f"✓ Loaded {len(self.test_sequences)} test sequences")
            print(f"✓ Loaded {len(self.raw_data)} sensor data points")
            print(f"✓ Sensor ODR: {self.sensor_specs.odr} Hz")
            print(f"✓ Data duration: {self.raw_data['timestamp'].max() - self.raw_data['timestamp'].min():.1f} seconds")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading test data: {e}")
            return False
    
    # ========================================================================
    # Angle Detection Functions
    # ========================================================================
    
    def detect_angle_changes_in_data(self, data: pd.DataFrame, angle_type: str) -> List[Dict]:
        """
        Detect angle changes using motion estimator algorithms.
        
        Args:
            data: Sensor data DataFrame with columns: timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
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
        
        for idx, row in data.iterrows():
            # Prepare sensor data
            accel = [row['accel_x'], row['accel_y'], row['accel_z']]
            gyro = [row['gyro_x'], row['gyro_y'], row['gyro_z']]
            timestamp = row['timestamp']
            
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
        
        # Convert to numpy arrays for processing
        angles = np.array(angles)
        timestamps = np.array(timestamps)
        
        # Calculate angular velocity (DPS)
        if len(angles) > 1:
            dt = np.diff(timestamps)
            # Avoid division by zero
            dt = np.where(dt == 0, 1e-6, dt)
            angular_velocity = np.abs(np.diff(angles) / dt)
            
            # Detect significant changes based on DPS threshold
            threshold = self.dps_thresholds[angle_type]
            significant_changes = angular_velocity > threshold
            
            # Find change points
            change_indices = np.where(significant_changes)[0]
            
            for idx in change_indices:
                if idx > 0:  # Ensure we have a previous value
                    detection = {
                        'timestamp': timestamps[idx],
                        'angle': angles[idx],
                        'angle_change': angles[idx] - angles[idx-1],
                        'dps': angular_velocity[idx],
                        'confidence': min(1.0, angular_velocity[idx] / (threshold * 2))
                    }
                    detections.append(detection)
        
        return detections
    
    def analyze_angle_detection(self) -> List[AngleDetectionResult]:
        """
        Analyze angle detection capabilities across all test sequences.
        
        Returns:
            List of angle detection results
        """
        if self.raw_data is None or len(self.test_sequences) == 0:
            print("✗ No test data loaded")
            return []
        
        results = []
        
        # Calculate samples per test based on duration and ODR
        total_duration = self.raw_data['timestamp'].max() - self.raw_data['timestamp'].min()
        samples_per_second = self.sensor_specs.odr
        total_expected_samples = int(total_duration * samples_per_second)
        
        print(f"Analyzing {len(self.test_sequences)} test sequences")
        print(f"Total data duration: {total_duration:.1f} seconds")
        print(f"Expected samples: {total_expected_samples}, Actual samples: {len(self.raw_data)}")
        
        # Process each test sequence
        for test_seq in self.test_sequences:
            # Calculate time window for this test
            test_start_time = test_seq.change_time
            test_end_time = test_start_time + test_seq.duration
            
            # Extract data for this test
            test_data = self.raw_data[
                (self.raw_data['timestamp'] >= test_start_time) & 
                (self.raw_data['timestamp'] <= test_end_time)
            ].copy()
            
            if len(test_data) == 0:
                print(f"⚠ No data found for test {test_seq.test_id}")
                continue
            
            # Reset motion estimator for each test
            self.motion_estimator = MotionEstimator()
            
            # Analyze each angle type
            for angle_type in ['theta', 'phi', 'psi']:
                expected_change = getattr(test_seq, f'{angle_type}_change')
                
                if abs(expected_change) > 0.1:  # Only analyze if there's an expected change
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
        
        self.detection_results = results
        print(f"✓ Generated {len(results)} detection results")
        return results
    
    # ========================================================================
    # Visualization Functions
    # ========================================================================
    
    def create_angle_visualization(self, save_plots: bool = True) -> None:
        """
        Create comprehensive visualizations of angle detection results.
        
        Args:
            save_plots: Whether to save plots to files
        """
        if not self.detection_results:
            print("✗ No detection results to visualize")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Detection Summary by Angle Type
        ax1 = plt.subplot(2, 3, 1)
        self._plot_detection_summary(ax1)
        
        # 2. Angle Magnitude vs Detection Success
        ax2 = plt.subplot(2, 3, 2)
        self._plot_angle_magnitude_vs_detection(ax2)
        
        # 3. DPS vs Detection Accuracy
        ax3 = plt.subplot(2, 3, 3)
        self._plot_dps_vs_accuracy(ax3)
        
        # 4. Detection Timeline with Color-coded Regions
        ax4 = plt.subplot(2, 3, (4, 6))
        self._plot_detection_timeline(ax4)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('ism330dhcx_angle_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Saved visualization to ism330dhcx_angle_analysis.png")
        
        plt.show()
    
    def _plot_detection_summary(self, ax):
        """Plot overall detection summary by angle type"""
        if not self.detection_results:
            return
        
        # Count detections by angle type
        angle_types = ['theta', 'phi', 'psi']
        detected_counts = []
        total_counts = []
        
        for angle_type in angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            detected = sum(1 for r in type_results if r.is_detected)
            total = len(type_results)
            
            detected_counts.append(detected)
            total_counts.append(total)
        
        x = np.arange(len(angle_types))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, detected_counts, width, label='Detected', alpha=0.8, color='green')
        bars2 = ax.bar(x + width/2, total_counts, width, label='Total', alpha=0.8, color='lightblue')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(height)}', ha='center', va='bottom')
        
        ax.set_xlabel('Angle Type')
        ax.set_ylabel('Number of Tests')
        ax.set_title('Detection Summary by Angle Type')
        ax.set_xticks(x)
        ax.set_xticklabels(angle_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_angle_magnitude_vs_detection(self, ax):
        """Plot angle magnitude vs detection success"""
        if not self.detection_results:
            return
        
        expected_angles = [abs(r.expected_angle) for r in self.detection_results]
        detected = [r.is_detected for r in self.detection_results]
        confidences = [r.confidence for r in self.detection_results]
        
        # Create scatter plot with confidence as size
        colors = ['red' if not d else 'green' for d in detected]
        sizes = [max(20, c * 100) for c in confidences]  # Scale confidence to size
        
        scatter = ax.scatter(expected_angles, detected, c=colors, s=sizes, alpha=0.7)
        
        # Add trend line
        if len(expected_angles) > 1:
            z = np.polyfit(expected_angles, [float(d) for d in detected], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(expected_angles), max(expected_angles), 100)
            ax.plot(x_trend, p(x_trend), "b--", alpha=0.8, label='Trend')
        
        ax.set_xlabel('Expected Angle Magnitude (deg)')
        ax.set_ylabel('Detection Success')
        ax.set_title('Angle Magnitude vs Detection Success\n(Size = Confidence)')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_dps_vs_accuracy(self, ax):
        """Plot DPS threshold vs detection accuracy"""
        if not self.detection_results:
            return
        
        # Group by angle type and calculate accuracy
        angle_types = ['theta', 'phi', 'psi']
        accuracies = []
        
        for angle_type in angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            if type_results:
                accuracy = sum(1 for r in type_results if r.is_detected) / len(type_results)
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        bars = ax.bar(angle_types, accuracies, alpha=0.8, color=['red', 'green', 'blue'])
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{acc:.1%}', ha='center', va='bottom')
        
        ax.set_xlabel('Angle Type')
        ax.set_ylabel('Detection Accuracy')
        ax.set_title('Detection Accuracy by Angle Type')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_detection_timeline(self, ax):
        """Plot detection timeline with color-coded regions"""
        if not self.detection_results:
            return
        
        # Create timeline data
        test_ids = sorted(set(r.test_id for r in self.detection_results))
        angle_types = ['theta', 'phi', 'psi']
        
        # Create color map for detections
        y_pos = 0
        y_labels = []
        
        for test_id in test_ids:
            test_results = [r for r in self.detection_results if r.test_id == test_id]
            
            for angle_type in angle_types:
                type_result = next((r for r in test_results if r.angle_type == angle_type), None)
                
                if type_result and abs(type_result.expected_angle) > 0.1:
                    # Create rectangle for this detection
                    height = 0.8
                    width = 1.0
                    
                    color = 'green' if type_result.is_detected else 'red'
                    alpha = type_result.confidence if type_result.is_detected else 0.3
                    
                    rect = patches.Rectangle(
                        (test_id - width/2, y_pos - height/2), width, height,
                        linewidth=1, edgecolor='black', facecolor=color, alpha=alpha
                    )
                    ax.add_patch(rect)
                    
                    # Add text label
                    ax.text(test_id, y_pos, f'{angle_type}\n{type_result.expected_angle:.1f}°', 
                           ha='center', va='center', fontsize=8, weight='bold')
                    
                    y_labels.append(f'Test {test_id} - {angle_type}')
                    y_pos += 1
        
        ax.set_xlabel('Test ID')
        ax.set_ylabel('Test Sequence')
        ax.set_title('Detection Timeline\n(Green=Detected, Red=Missed, Size=Confidence)')
        ax.set_xlim(min(test_ids) - 1, max(test_ids) + 1)
        ax.set_ylim(-0.5, y_pos + 0.5)
        ax.set_yticks(range(y_pos))
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.grid(True, alpha=0.3)
    
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
                    analysis['avg_dps_detected'] = np.mean([r.dps_used for r in detected_results])
                    analysis['min_dps_detected'] = np.min([r.dps_used for r in detected_results])
                    analysis['max_dps_detected'] = np.max([r.dps_used for r in detected_results])
                
                if missed_results:
                    analysis['avg_dps_missed'] = np.mean([r.dps_used for r in missed_results])
                
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
                    avg_confidence = np.mean([r.confidence for r in detected_results])
                    avg_dps = np.mean([r.dps_used for r in detected_results])
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
                    avg_dps = np.mean([r.dps_used for r in detected_results])
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
    analyzer = ISM330DHCAnalyzer()
    
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
        if Path(config_file).exists() and Path(data_file).exists():
            print(f"\nAnalyzing {config_file} with {data_file}")
            print("-" * 40)
            
            # Load data
            if analyzer.load_test_data(config_file, data_file):
                # Analyze angle detection
                results = analyzer.analyze_angle_detection()
                
                # Generate and print report
                report = analyzer.generate_analysis_report()
                print(report)
                
                # Create visualizations
                analyzer.create_angle_visualization()
                
                # Save results
                analyzer.save_results(f"results_{Path(config_file).stem}.json")
            else:
                print(f"✗ Failed to load {config_file} or {data_file}")
        else:
            print(f"✗ Files not found: {config_file} or {data_file}")

if __name__ == "__main__":
    main()