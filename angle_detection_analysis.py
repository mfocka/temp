#!/usr/bin/env python3
"""
# Angle Detection Analysis Tool for ISM330DHCX Sensor

This tool analyzes angle detection capabilities of the ISM330DHCX sensor
using the motion estimator algorithms. It processes test configuration files
and raw sensor data to determine minimal detectable angles based on DPS thresholds.

## Features:
- Load test configurations and raw sensor data
- Apply motion estimator algorithms for angle detection
- Visualize detected angles with color-coded regions
- Analyze minimal detectable angles based on DPS thresholds
- Generate comprehensive analysis reports

## Usage:
```python
# Load and analyze test data
analyzer = AngleDetectionAnalyzer()
analyzer.load_test_data("test_config_altitude.json", "raw_data_output_altitude.csv")
results = analyzer.analyze_angle_detection()
analyzer.visualize_results()
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

@dataclass
class TestConfig:
    """Test configuration data structure"""
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
    """Sensor specifications"""
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
    expected_angle: float
    detected_angle: float
    detection_time: float
    is_detected: bool
    confidence: float
    dps_threshold: float
    angle_type: str  # 'theta', 'phi', 'psi'

class AngleDetectionAnalyzer:
    """
    Comprehensive angle detection analysis tool for ISM330DHCX sensor.
    
    This class provides functionality to:
    - Load test configurations and raw sensor data
    - Apply motion estimator algorithms
    - Detect angle changes and analyze detection capabilities
    - Visualize results with color-coded regions
    - Determine minimal detectable angles
    """
    
    def __init__(self):
        """Initialize the analyzer with default settings"""
        self.motion_estimator = MotionEstimator()
        self.test_configs: List[TestConfig] = []
        self.sensor_specs: Optional[SensorSpecs] = None
        self.initial_conditions: Optional[InitialConditions] = None
        self.raw_data: Optional[pd.DataFrame] = None
        self.detection_results: List[AngleDetectionResult] = []
        
        # Detection parameters
        self.dps_thresholds = {
            'theta': 0.5,  # degrees per second
            'phi': 0.5,    # degrees per second  
            'psi': 0.5     # degrees per second
        }
        
        # Angle detection thresholds
        self.angle_thresholds = {
            'theta': 1.0,  # degrees
            'phi': 1.0,    # degrees
            'psi': 1.0     # degrees
        }
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
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
            self.test_configs = []
            for test in config_data['test_sequence']:
                test_config = TestConfig(
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
                self.test_configs.append(test_config)
            
            # Load raw sensor data
            self.raw_data = pd.read_csv(data_file)
            
            # Convert timestamps to seconds if needed
            if 'timestamp' in self.raw_data.columns:
                self.raw_data['timestamp'] = self.raw_data['timestamp'] / 1000.0
            
            print(f"✓ Loaded {len(self.test_configs)} test configurations")
            print(f"✓ Loaded {len(self.raw_data)} sensor data points")
            print(f"✓ Sensor ODR: {self.sensor_specs.odr} Hz")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading test data: {e}")
            return False
    
    def detect_angle_changes(self, data: pd.DataFrame, angle_type: str) -> List[Dict]:
        """
        Detect angle changes using motion estimator algorithms.
        
        Args:
            data: Sensor data DataFrame
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
        Analyze angle detection capabilities across all test configurations.
        
        Returns:
            List of angle detection results
        """
        if self.raw_data is None or len(self.test_configs) == 0:
            print("✗ No test data loaded")
            return []
        
        results = []
        
        # Group data by test periods (assuming sequential tests)
        data_length = len(self.raw_data)
        tests_per_data = len(self.test_configs)
        samples_per_test = data_length // tests_per_data
        
        print(f"Analyzing {tests_per_data} tests with ~{samples_per_test} samples each")
        
        for i, test_config in enumerate(self.test_configs):
            # Extract data for this test
            start_idx = i * samples_per_test
            end_idx = min((i + 1) * samples_per_test, data_length)
            test_data = self.raw_data.iloc[start_idx:end_idx].copy()
            
            # Reset motion estimator for each test
            self.motion_estimator = MotionEstimator()
            
            # Analyze each angle type
            for angle_type in ['theta', 'phi', 'psi']:
                expected_change = getattr(test_config, f'{angle_type}_change')
                
                if abs(expected_change) > 0.1:  # Only analyze if there's an expected change
                    # Detect angle changes
                    detections = self.detect_angle_changes(test_data, angle_type)
                    
                    # Find the most significant detection
                    if detections:
                        best_detection = max(detections, key=lambda x: x['confidence'])
                        
                        result = AngleDetectionResult(
                            test_id=test_config.test_id,
                            expected_angle=expected_change,
                            detected_angle=best_detection['angle_change'],
                            detection_time=best_detection['timestamp'],
                            is_detected=abs(best_detection['angle_change']) >= self.angle_thresholds[angle_type],
                            confidence=best_detection['confidence'],
                            dps_threshold=self.dps_thresholds[angle_type],
                            angle_type=angle_type
                        )
                        results.append(result)
                    else:
                        # No detection found
                        result = AngleDetectionResult(
                            test_id=test_config.test_id,
                            expected_angle=expected_change,
                            detected_angle=0.0,
                            detection_time=0.0,
                            is_detected=False,
                            confidence=0.0,
                            dps_threshold=self.dps_thresholds[angle_type],
                            angle_type=angle_type
                        )
                        results.append(result)
        
        self.detection_results = results
        return results
    
    def visualize_results(self, save_plots: bool = True) -> None:
        """
        Create comprehensive visualizations of angle detection results.
        
        Args:
            save_plots: Whether to save plots to files
        """
        if not self.detection_results:
            print("✗ No detection results to visualize")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Angle Detection Summary
        ax1 = plt.subplot(3, 3, 1)
        self._plot_detection_summary(ax1)
        
        # 2. DPS vs Detection Accuracy
        ax2 = plt.subplot(3, 3, 2)
        self._plot_dps_vs_accuracy(ax2)
        
        # 3. Angle Magnitude vs Detection
        ax3 = plt.subplot(3, 3, 3)
        self._plot_angle_magnitude_vs_detection(ax3)
        
        # 4. Detection Timeline
        ax4 = plt.subplot(3, 3, (4, 6))
        self._plot_detection_timeline(ax4)
        
        # 5. Confidence Distribution
        ax5 = plt.subplot(3, 3, 7)
        self._plot_confidence_distribution(ax5)
        
        # 6. Minimal Detectable Angles
        ax6 = plt.subplot(3, 3, 8)
        self._plot_minimal_detectable_angles(ax6)
        
        # 7. Detection Performance by Angle Type
        ax7 = plt.subplot(3, 3, 9)
        self._plot_performance_by_angle_type(ax7)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('angle_detection_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Saved visualization to angle_detection_analysis.png")
        
        plt.show()
    
    def _plot_detection_summary(self, ax):
        """Plot overall detection summary"""
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
        
        ax.bar(x - width/2, detected_counts, width, label='Detected', alpha=0.8)
        ax.bar(x + width/2, total_counts, width, label='Total', alpha=0.8)
        
        ax.set_xlabel('Angle Type')
        ax.set_ylabel('Number of Tests')
        ax.set_title('Detection Summary by Angle Type')
        ax.set_xticks(x)
        ax.set_xticklabels(angle_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_dps_vs_accuracy(self, ax):
        """Plot DPS threshold vs detection accuracy"""
        if not self.detection_results:
            return
        
        # Group by DPS threshold
        dps_values = sorted(set(r.dps_threshold for r in self.detection_results))
        accuracies = []
        
        for dps in dps_values:
            results_at_dps = [r for r in self.detection_results if r.dps_threshold == dps]
            if results_at_dps:
                accuracy = sum(1 for r in results_at_dps if r.is_detected) / len(results_at_dps)
                accuracies.append(accuracy)
            else:
                accuracies.append(0)
        
        ax.plot(dps_values, accuracies, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('DPS Threshold (deg/s)')
        ax.set_ylabel('Detection Accuracy')
        ax.set_title('DPS Threshold vs Detection Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    def _plot_angle_magnitude_vs_detection(self, ax):
        """Plot angle magnitude vs detection success"""
        if not self.detection_results:
            return
        
        expected_angles = [abs(r.expected_angle) for r in self.detection_results]
        detected = [r.is_detected for r in self.detection_results]
        
        # Create scatter plot
        colors = ['red' if not d else 'green' for d in detected]
        ax.scatter(expected_angles, detected, c=colors, alpha=0.7, s=50)
        
        ax.set_xlabel('Expected Angle Magnitude (deg)')
        ax.set_ylabel('Detection Success')
        ax.set_title('Angle Magnitude vs Detection Success')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)
    
    def _plot_detection_timeline(self, ax):
        """Plot detection timeline with color-coded regions"""
        if not self.detection_results:
            return
        
        # Create timeline data
        test_ids = sorted(set(r.test_id for r in self.detection_results))
        angle_types = ['theta', 'phi', 'psi']
        
        # Create color map for detections
        colors = plt.cm.Set3(np.linspace(0, 1, len(test_ids)))
        
        y_pos = 0
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
                           ha='center', va='center', fontsize=8)
                    
                    y_pos += 1
        
        ax.set_xlabel('Test ID')
        ax.set_ylabel('Angle Type')
        ax.set_title('Detection Timeline (Green=Detected, Red=Missed)')
        ax.set_xlim(min(test_ids) - 1, max(test_ids) + 1)
        ax.set_ylim(-0.5, y_pos + 0.5)
        ax.grid(True, alpha=0.3)
    
    def _plot_confidence_distribution(self, ax):
        """Plot confidence distribution of detections"""
        if not self.detection_results:
            return
        
        detected_results = [r for r in self.detection_results if r.is_detected]
        confidences = [r.confidence for r in detected_results]
        
        if confidences:
            ax.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(confidences), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(confidences):.3f}')
            ax.set_xlabel('Detection Confidence')
            ax.set_ylabel('Frequency')
            ax.set_title('Confidence Distribution of Detected Angles')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No detections found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Confidence Distribution')
    
    def _plot_minimal_detectable_angles(self, ax):
        """Plot minimal detectable angles by angle type"""
        if not self.detection_results:
            return
        
        angle_types = ['theta', 'phi', 'psi']
        minimal_angles = []
        
        for angle_type in angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            detected_results = [r for r in type_results if r.is_detected]
            
            if detected_results:
                # Find the smallest detected angle
                minimal_angle = min(abs(r.expected_angle) for r in detected_results)
                minimal_angles.append(minimal_angle)
            else:
                minimal_angles.append(float('inf'))
        
        # Plot bars
        x = np.arange(len(angle_types))
        bars = ax.bar(x, minimal_angles, alpha=0.7)
        
        # Color bars based on detection success
        for i, (bar, angle) in enumerate(zip(bars, minimal_angles)):
            if angle == float('inf'):
                bar.set_color('red')
                bar.set_alpha(0.3)
            else:
                bar.set_color('green')
        
        ax.set_xlabel('Angle Type')
        ax.set_ylabel('Minimal Detectable Angle (deg)')
        ax.set_title('Minimal Detectable Angles by Type')
        ax.set_xticks(x)
        ax.set_xticklabels(angle_types)
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_by_angle_type(self, ax):
        """Plot detection performance metrics by angle type"""
        if not self.detection_results:
            return
        
        angle_types = ['theta', 'phi', 'psi']
        accuracies = []
        precisions = []
        
        for angle_type in angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            
            if type_results:
                # Calculate accuracy
                detected = sum(1 for r in type_results if r.is_detected)
                total = len(type_results)
                accuracy = detected / total if total > 0 else 0
                accuracies.append(accuracy)
                
                # Calculate precision (for detected cases)
                detected_results = [r for r in type_results if r.is_detected]
                if detected_results:
                    # Precision based on how close detected angle is to expected
                    errors = [abs(r.detected_angle - r.expected_angle) for r in detected_results]
                    precision = 1.0 - (np.mean(errors) / 10.0)  # Normalize by 10 degrees
                    precision = max(0, min(1, precision))
                else:
                    precision = 0
                precisions.append(precision)
            else:
                accuracies.append(0)
                precisions.append(0)
        
        x = np.arange(len(angle_types))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax.bar(x + width/2, precisions, width, label='Precision', alpha=0.8)
        
        ax.set_xlabel('Angle Type')
        ax.set_ylabel('Performance Metric')
        ax.set_title('Detection Performance by Angle Type')
        ax.set_xticks(x)
        ax.set_xticklabels(angle_types)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.detection_results:
            return "No detection results available for report generation."
        
        report = []
        report.append("=" * 80)
        report.append("ANGLE DETECTION ANALYSIS REPORT")
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
                    minimal_angle = min(abs(r.expected_angle) for r in detected_results)
                    avg_confidence = np.mean([r.confidence for r in detected_results])
                else:
                    minimal_angle = float('inf')
                    avg_confidence = 0
                
                report.append(f"{angle_type.upper()} Analysis:")
                report.append(f"  Accuracy: {accuracy:.1%} ({detected}/{total})")
                report.append(f"  Minimal Detectable Angle: {minimal_angle:.2f}°" if minimal_angle != float('inf') else "  Minimal Detectable Angle: Not detected")
                report.append(f"  Average Confidence: {avg_confidence:.3f}")
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
                    avg_dps = np.mean([r.dps_threshold for r in detected_results])
                    report.append(f"• Optimal {angle_type.upper()} DPS threshold: {avg_dps:.2f} deg/s")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "angle_detection_results.json") -> bool:
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
                    'expected_angle': result.expected_angle,
                    'detected_angle': result.detected_angle,
                    'detection_time': result.detection_time,
                    'is_detected': result.is_detected,
                    'confidence': result.confidence,
                    'dps_threshold': result.dps_threshold,
                    'angle_type': result.angle_type
                })
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"✓ Results saved to {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error saving results: {e}")
            return False

# Example usage and testing functions
def main():
    """Main function for testing the analyzer"""
    print("Angle Detection Analysis Tool for ISM330DHCX")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AngleDetectionAnalyzer()
    
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
                report = analyzer.generate_report()
                print(report)
                
                # Create visualizations
                analyzer.visualize_results()
                
                # Save results
                analyzer.save_results(f"results_{Path(config_file).stem}.json")
            else:
                print(f"✗ Failed to load {config_file} or {data_file}")
        else:
            print(f"✗ Files not found: {config_file} or {data_file}")

if __name__ == "__main__":
    main()