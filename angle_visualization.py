#!/usr/bin/env python3
"""
# Angle Detection Visualization Tool

A simple visualization tool that creates charts highlighting detected angles
(theta, phi, psi) using basic Python libraries.

## Features:
- Create ASCII-based charts and plots
- Highlight detected angles with color-coded regions
- Generate HTML-based visualizations
- Export analysis results to various formats

## Usage:
```python
# Load analysis results
visualizer = AngleVisualizer()
visualizer.load_results("results_test_config_altitude.json")

# Create visualizations
visualizer.create_ascii_charts()
visualizer.create_html_visualization()
```
"""

import json
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class DetectionResult:
    """Detection result data structure"""
    test_id: int
    angle_type: str
    expected_angle: float
    detected_angle: float
    detection_time: float
    is_detected: bool
    confidence: float
    dps_used: float
    angle_magnitude: float

class AngleVisualizer:
    """
    Simple visualization tool for angle detection results.
    
    Creates ASCII-based charts and HTML visualizations to highlight
    detected angles with color-coded regions.
    """
    
    def __init__(self):
        """Initialize the visualizer"""
        self.detection_results: List[DetectionResult] = []
        self.angle_types = ['theta', 'phi', 'psi']
        
    def load_results(self, filename: str) -> bool:
        """
        Load detection results from JSON file.
        
        Args:
            filename: Path to results JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.detection_results = []
            for item in data:
                result = DetectionResult(
                    test_id=item['test_id'],
                    angle_type=item['angle_type'],
                    expected_angle=item['expected_angle'],
                    detected_angle=item['detected_angle'],
                    detection_time=item['detection_time'],
                    is_detected=item['is_detected'],
                    confidence=item['confidence'],
                    dps_used=item['dps_used'],
                    angle_magnitude=item['angle_magnitude']
                )
                self.detection_results.append(result)
            
            print(f"✓ Loaded {len(self.detection_results)} detection results from {filename}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading results: {e}")
            return False
    
    def create_ascii_charts(self) -> None:
        """Create ASCII-based charts for terminal display"""
        if not self.detection_results:
            print("No detection results to visualize")
            return
        
        print("\n" + "="*80)
        print("ANGLE DETECTION VISUALIZATION")
        print("="*80)
        
        # 1. Detection Summary Chart
        self._create_detection_summary_chart()
        
        # 2. Angle Magnitude vs Detection Chart
        self._create_angle_magnitude_chart()
        
        # 3. DPS Analysis Chart
        self._create_dps_analysis_chart()
        
        # 4. Timeline Chart
        self._create_timeline_chart()
    
    def _create_detection_summary_chart(self):
        """Create ASCII chart showing detection summary by angle type"""
        print("\nDETECTION SUMMARY BY ANGLE TYPE")
        print("-" * 40)
        
        for angle_type in self.angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            if type_results:
                detected = sum(1 for r in type_results if r.is_detected)
                total = len(type_results)
                accuracy = detected / total if total > 0 else 0
                
                # Create ASCII bar chart
                bar_length = int(accuracy * 20)  # Scale to 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                print(f"{angle_type.upper():>6}: {bar} {accuracy:.1%} ({detected}/{total})")
            else:
                print(f"{angle_type.upper():>6}: {'░' * 20} 0.0% (0/0)")
    
    def _create_angle_magnitude_chart(self):
        """Create ASCII chart showing angle magnitude vs detection"""
        print("\nANGLE MAGNITUDE vs DETECTION")
        print("-" * 40)
        
        # Group results by magnitude ranges
        magnitude_ranges = [
            (0, 2, "0-2°"),
            (2, 5, "2-5°"),
            (5, 10, "5-10°"),
            (10, 15, "10-15°"),
            (15, 20, "15-20°")
        ]
        
        for min_angle, max_angle, label in magnitude_ranges:
            range_results = [
                r for r in self.detection_results
                if min_angle <= r.angle_magnitude < max_angle
            ]
            
            if range_results:
                detected = sum(1 for r in range_results if r.is_detected)
                total = len(range_results)
                accuracy = detected / total if total > 0 else 0
                
                # Create ASCII bar chart
                bar_length = int(accuracy * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                print(f"{label:>8}: {bar} {accuracy:.1%} ({detected}/{total})")
            else:
                print(f"{label:>8}: {'░' * 20} 0.0% (0/0)")
    
    def _create_dps_analysis_chart(self):
        """Create ASCII chart showing DPS analysis"""
        print("\nDPS ANALYSIS")
        print("-" * 40)
        
        for angle_type in self.angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            if type_results:
                detected_results = [r for r in type_results if r.is_detected]
                if detected_results:
                    dps_values = [r.dps_used for r in detected_results]
                    avg_dps = sum(dps_values) / len(dps_values)
                    min_dps = min(dps_values)
                    max_dps = max(dps_values)
                    
                    print(f"{angle_type.upper():>6}: Avg={avg_dps:6.2f} Min={min_dps:6.2f} Max={max_dps:6.2f} deg/s")
                else:
                    print(f"{angle_type.upper():>6}: No detections")
            else:
                print(f"{angle_type.upper():>6}: No data")
    
    def _create_timeline_chart(self):
        """Create ASCII timeline chart"""
        print("\nDETECTION TIMELINE")
        print("-" * 40)
        
        # Group by test ID
        test_ids = sorted(set(r.test_id for r in self.detection_results))
        
        for test_id in test_ids:
            test_results = [r for r in self.detection_results if r.test_id == test_id]
            
            # Create timeline for this test
            timeline = []
            for angle_type in self.angle_types:
                type_result = next((r for r in test_results if r.angle_type == angle_type), None)
                if type_result and abs(type_result.expected_angle) > 0.1:
                    if type_result.is_detected:
                        timeline.append(f"{angle_type.upper()}:✓")
                    else:
                        timeline.append(f"{angle_type.upper()}:✗")
            
            if timeline:
                print(f"Test {test_id:2d}: {' '.join(timeline)}")
    
    def create_html_visualization(self, output_file: str = "angle_detection_visualization.html") -> bool:
        """
        Create HTML-based visualization.
        
        Args:
            output_file: Output HTML filename
            
        Returns:
            True if successful, False otherwise
        """
        if not self.detection_results:
            print("No detection results to visualize")
            return False
        
        try:
            html_content = self._generate_html_content()
            
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            print(f"✓ HTML visualization saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"✗ Error creating HTML visualization: {e}")
            return False
    
    def _generate_html_content(self) -> str:
        """Generate HTML content for visualization"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>ISM330DHCX Angle Detection Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        h2 { color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }
        .chart { margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-radius: 5px; }
        .bar { height: 30px; background: linear-gradient(90deg, #4CAF50, #8BC34A); margin: 5px 0; border-radius: 15px; display: flex; align-items: center; padding: 0 15px; color: white; font-weight: bold; }
        .bar.missed { background: linear-gradient(90deg, #F44336, #FF9800); }
        .timeline { display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0; }
        .test-item { padding: 10px; border-radius: 5px; text-align: center; min-width: 100px; }
        .test-item.detected { background-color: #4CAF50; color: white; }
        .test-item.missed { background-color: #F44336; color: white; }
        .test-item.no-data { background-color: #9E9E9E; color: white; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .stat-card { background-color: #e3f2fd; padding: 15px; border-radius: 5px; text-align: center; }
        .stat-value { font-size: 2em; font-weight: bold; color: #1976d2; }
        .stat-label { color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ISM330DHCX Angle Detection Analysis</h1>
"""
        
        # Add summary statistics
        total_tests = len(self.detection_results)
        detected_tests = sum(1 for r in self.detection_results if r.is_detected)
        overall_accuracy = detected_tests / total_tests if total_tests > 0 else 0
        
        html += f"""
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{total_tests}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{detected_tests}</div>
                <div class="stat-label">Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{overall_accuracy:.1%}</div>
                <div class="stat-label">Overall Accuracy</div>
            </div>
        </div>
"""
        
        # Add detection summary by angle type
        html += """
        <h2>Detection Summary by Angle Type</h2>
        <div class="chart">
"""
        
        for angle_type in self.angle_types:
            type_results = [r for r in self.detection_results if r.angle_type == angle_type]
            if type_results:
                detected = sum(1 for r in type_results if r.is_detected)
                total = len(type_results)
                accuracy = detected / total if total > 0 else 0
                bar_width = int(accuracy * 100)
                
                html += f"""
            <div style="margin: 10px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span><strong>{angle_type.upper()}</strong></span>
                    <span>{accuracy:.1%} ({detected}/{total})</span>
                </div>
                <div style="background-color: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #4CAF50, #8BC34A); height: 100%; width: {bar_width}%; transition: width 0.3s ease;"></div>
                </div>
            </div>
"""
        
        html += "        </div>"
        
        # Add timeline visualization
        html += """
        <h2>Detection Timeline</h2>
        <div class="chart">
            <div class="timeline">
"""
        
        test_ids = sorted(set(r.test_id for r in self.detection_results))
        for test_id in test_ids:
            test_results = [r for r in self.detection_results if r.test_id == test_id]
            
            # Determine test status
            has_detections = any(r.is_detected for r in test_results if abs(r.expected_angle) > 0.1)
            has_expected = any(abs(r.expected_angle) > 0.1 for r in test_results)
            
            if not has_expected:
                status_class = "no-data"
                status_text = "No Expected Change"
            elif has_detections:
                status_class = "detected"
                status_text = "Detected"
            else:
                status_class = "missed"
                status_text = "Missed"
            
            html += f"""
                <div class="test-item {status_class}">
                    <div><strong>Test {test_id}</strong></div>
                    <div style="font-size: 0.8em;">{status_text}</div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        # Add detailed results table
        html += """
        <h2>Detailed Results</h2>
        <div class="chart">
            <table style="width: 100%; border-collapse: collapse;">
                <thead>
                    <tr style="background-color: #f5f5f5;">
                        <th style="padding: 10px; border: 1px solid #ddd;">Test ID</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Angle Type</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Expected</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Detected</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Status</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">Confidence</th>
                        <th style="padding: 10px; border: 1px solid #ddd;">DPS Used</th>
                    </tr>
                </thead>
                <tbody>
"""
        
        for result in self.detection_results:
            status = "✓ Detected" if result.is_detected else "✗ Missed"
            status_color = "#4CAF50" if result.is_detected else "#F44336"
            
            html += f"""
                    <tr>
                        <td style="padding: 10px; border: 1px solid #ddd;">{result.test_id}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{result.angle_type.upper()}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{result.expected_angle:.2f}°</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{result.detected_angle:.2f}°</td>
                        <td style="padding: 10px; border: 1px solid #ddd; color: {status_color}; font-weight: bold;">{status}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{result.confidence:.3f}</td>
                        <td style="padding: 10px; border: 1px solid #ddd;">{result.dps_used:.2f} deg/s</td>
                    </tr>
"""
        
        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def export_to_csv(self, output_file: str = "angle_detection_results.csv") -> bool:
        """
        Export detection results to CSV file.
        
        Args:
            output_file: Output CSV filename
            
        Returns:
            True if successful, False otherwise
        """
        if not self.detection_results:
            print("No detection results to export")
            return False
        
        try:
            import csv
            
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'test_id', 'angle_type', 'expected_angle', 'detected_angle',
                    'detection_time', 'is_detected', 'confidence', 'dps_used', 'angle_magnitude'
                ])
                
                # Write data
                for result in self.detection_results:
                    writer.writerow([
                        result.test_id, result.angle_type, result.expected_angle,
                        result.detected_angle, result.detection_time, result.is_detected,
                        result.confidence, result.dps_used, result.angle_magnitude
                    ])
            
            print(f"✓ Results exported to {output_file}")
            return True
            
        except Exception as e:
            print(f"✗ Error exporting to CSV: {e}")
            return False

def main():
    """Main function for testing the visualizer"""
    print("Angle Detection Visualization Tool")
    print("=" * 40)
    
    # Initialize visualizer
    visualizer = AngleVisualizer()
    
    # Load results from both test files
    result_files = [
        "results_test_config_altitude.json",
        "results_test_config_azimuth.json"
    ]
    
    for result_file in result_files:
        if visualizer.load_results(result_file):
            # Create ASCII charts
            visualizer.create_ascii_charts()
            
            # Create HTML visualization
            output_name = result_file.replace("results_", "").replace(".json", "")
            visualizer.create_html_visualization(f"visualization_{output_name}.html")
            
            # Export to CSV
            visualizer.export_to_csv(f"export_{output_name}.csv")
            
            print(f"\n✓ Completed visualization for {result_file}")
        else:
            print(f"✗ Failed to load {result_file}")

if __name__ == "__main__":
    main()