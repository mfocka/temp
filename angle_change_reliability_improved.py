"""
Improved Angle Change Reliability Analysis
Accurately detects angle changes and calculates DPS (degrees per second)
Based on proper understanding of the motion_estimator behavior
"""

import argparse
import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import uniform_filter1d

from motion_estimator import MotionEstimator, Config

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class AngleChangeDetector:
    """Detects angle changes in IMU data with proper motion profile understanding"""
    
    def __init__(self, fs: float = 104.0):
        self.fs = fs
        self.dt = 1.0 / fs
    
    def detect_change_window(self, angle_signal: np.ndarray, change_time_s: float, 
                            window_before_s: float = 5.0, window_after_s: float = 10.0) -> Dict:
        """
        Detect the actual angle change by analyzing the motion profile.
        
        The change_time is when the motion STARTS, not when it completes.
        We need to find the stable regions before and after the change.
        """
        change_sample = int(change_time_s * self.fs)
        
        # Define analysis windows
        stable_before_start = max(0, change_sample - int(window_before_s * self.fs))
        stable_before_end = max(0, change_sample - int(0.5 * self.fs))  # 0.5s before change
        
        # For the "after" window, we need to find when motion stops
        # This is typically 2-5 seconds after change_time depending on the motion speed
        motion_search_start = change_sample + int(1.0 * self.fs)  # Start looking 1s after
        motion_search_end = min(len(angle_signal), change_sample + int(window_after_s * self.fs))
        
        # Calculate angle derivative to find motion periods
        angle_derivative = np.gradient(angle_signal) * self.fs  # degrees/second
        
        # Find when motion stops (derivative near zero)
        motion_threshold = 0.5  # deg/s threshold for "stationary"
        
        # Find stable region after motion
        stable_after_start = motion_search_start
        for i in range(motion_search_start, motion_search_end):
            window = angle_derivative[i:i+int(0.5*self.fs)]  # Check 0.5s window
            if len(window) > 0 and np.abs(window).mean() < motion_threshold:
                stable_after_start = i
                break
        
        stable_after_end = min(len(angle_signal), stable_after_start + int(2.0 * self.fs))
        
        # Calculate stable values
        angle_before = np.median(angle_signal[stable_before_start:stable_before_end]) if stable_before_end > stable_before_start else 0
        angle_after = np.median(angle_signal[stable_after_start:stable_after_end]) if stable_after_end > stable_after_start else 0
        
        # Calculate the actual change
        angle_change = angle_after - angle_before
        
        # Find peak angular velocity during motion
        motion_window = angle_derivative[change_sample:stable_after_start]
        peak_dps = np.abs(motion_window).max() if len(motion_window) > 0 else 0
        mean_dps = np.abs(motion_window).mean() if len(motion_window) > 0 else 0
        
        # Calculate motion duration
        motion_duration_s = (stable_after_start - change_sample) / self.fs
        
        # Calculate effective DPS (angle change / time taken)
        effective_dps = abs(angle_change) / motion_duration_s if motion_duration_s > 0 else 0
        
        return {
            'angle_before': angle_before,
            'angle_after': angle_after,
            'angle_change': angle_change,
            'peak_dps': peak_dps,
            'mean_dps': mean_dps,
            'effective_dps': effective_dps,
            'motion_duration_s': motion_duration_s,
            'stable_before_window': (stable_before_start, stable_before_end),
            'stable_after_window': (stable_after_start, stable_after_end),
            'motion_window': (change_sample, stable_after_start)
        }
    
    def analyze_angle_stability(self, angle_signal: np.ndarray, window_size_s: float = 1.0) -> np.ndarray:
        """Calculate rolling standard deviation to assess signal stability"""
        window_samples = int(window_size_s * self.fs)
        if window_samples > len(angle_signal):
            window_samples = len(angle_signal)
        
        # Use pandas rolling for efficient computation
        df = pd.DataFrame({'angle': angle_signal})
        stability = df['angle'].rolling(window=window_samples, center=True).std().fillna(0).values
        return stability


def load_config(config_path: Path) -> dict:
    """Load test configuration from JSON file"""
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {"test_sequence": []}


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load and validate IMU data from CSV"""
    df = pd.read_csv(csv_path)
    required = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    
    # Convert to numeric and drop NaN
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    return df


def edn_to_ned(df: pd.DataFrame) -> pd.DataFrame:
    """Convert from EDN to NED coordinate system"""
    df = df.copy()
    # Accelerometer transformation
    df["accel_x"], df["accel_y"], df["accel_z"] = (
        df["accel_z"],
        df["accel_x"],
        -df["accel_y"],
    )
    # Gyroscope transformation
    df["gyro_x"], df["gyro_y"], df["gyro_z"] = (
        df["gyro_z"],
        df["gyro_x"],
        -df["gyro_y"],
    )
    return df


def regularize_timestamps(ts: np.ndarray, fs: float) -> np.ndarray:
    """Regularize timestamps to fixed sampling rate"""
    current = float(ts[0])
    updated = []
    for i in range(1, len(ts)):
        if ts[i] + 10 < ts[i - 1]:  # Handle wraparound
            current = float(ts[i])
        updated.append(current)
        current += 1.0 / fs
    updated.append(current)
    return np.array(updated, dtype=float)


def segment_by_gaps(df: pd.DataFrame, gap_s: float = 5.0) -> list[tuple[int, int]]:
    """Segment data by detecting gaps in timestamps"""
    ts = df["timestamp"].to_numpy()
    gaps = np.where(np.abs(np.diff(ts)) > gap_s)[0]
    bounds = []
    start = 0
    for gi in gaps:
        bounds.append((start, gi + 1))
        start = gi + 1
    bounds.append((start, len(df)))
    return bounds


def map_segments_to_config(df: pd.DataFrame, config: dict) -> list[dict]:
    """Map data segments to test configuration"""
    bounds = segment_by_gaps(df)
    seq = config.get("test_sequence", [])
    segments = []
    for i, (s, e) in enumerate(bounds):
        meta = seq[i] if i < len(seq) else {"test_id": i}
        segments.append({
            "test_id": int(meta.get("test_id", i)),
            "data": df.iloc[s:e].reset_index(drop=True),
            "config": meta,
            "start_idx": s,
            "end_idx": e,
        })
    return segments


def process_segment_with_estimator(segment_data: pd.DataFrame, estimator: MotionEstimator) -> pd.DataFrame:
    """Process a segment through the MotionEstimator"""
    results = []
    
    for i, row in segment_data.iterrows():
        accel_raw = [row["accel_x"], row["accel_y"], row["accel_z"]]  # mg
        gyro_raw = [row["gyro_x"], row["gyro_y"], row["gyro_z"]]      # dps
        
        output = estimator.update(accel_raw, gyro_raw, row["timestamp"])
        output["sample_idx"] = i
        results.append(output)
    
    return pd.DataFrame(results)


def analyze_segment(segment: dict, estimator: MotionEstimator, detector: AngleChangeDetector) -> Optional[Dict]:
    """
    Comprehensive analysis of a single test segment.
    """
    test_config = segment["config"]
    
    # Skip if power is off (no reliable data)
    if test_config.get("power_state", "off") == "off":
        return None
    
    # Process through motion estimator
    processed_df = process_segment_with_estimator(segment["data"], estimator)
    
    # Get change time from config
    change_time_s = test_config.get("change_time", 20)  # Default 20s
    
    result = {
        "test_id": segment["test_id"],
        "description": test_config.get("description", ""),
        "power_state": test_config.get("power_state", "unknown"),
        "processed_data": processed_df
    }
    
    # Analyze theta (altitude/pitch) change if present
    if "theta_change" in test_config:
        theta_analysis = detector.detect_change_window(
            processed_df["pitch_deg"].values, 
            change_time_s
        )
        
        expected_theta = test_config["theta_change"]
        detected_theta = theta_analysis["angle_change"]
        
        result.update({
            "expected_theta": expected_theta,
            "detected_theta": detected_theta,
            "theta_error": abs(detected_theta - expected_theta),
            "theta_before": theta_analysis["angle_before"],
            "theta_after": theta_analysis["angle_after"],
            "theta_peak_dps": theta_analysis["peak_dps"],
            "theta_mean_dps": theta_analysis["mean_dps"],
            "theta_effective_dps": theta_analysis["effective_dps"],
            "theta_motion_duration": theta_analysis["motion_duration_s"],
            "theta_analysis": theta_analysis
        })
    
    # Analyze phi (azimuth/yaw) change if present
    if "phi_change" in test_config:
        phi_analysis = detector.detect_change_window(
            processed_df["yaw_deg"].values, 
            change_time_s
        )
        
        expected_phi = test_config["phi_change"]
        detected_phi = phi_analysis["angle_change"]
        
        result.update({
            "expected_phi": expected_phi,
            "detected_phi": detected_phi,
            "phi_error": abs(detected_phi - expected_phi),
            "phi_before": phi_analysis["angle_before"],
            "phi_after": phi_analysis["angle_after"],
            "phi_peak_dps": phi_analysis["peak_dps"],
            "phi_mean_dps": phi_analysis["mean_dps"],
            "phi_effective_dps": phi_analysis["effective_dps"],
            "phi_motion_duration": phi_analysis["motion_duration_s"],
            "phi_analysis": phi_analysis
        })
    
    # Get max raw gyro DPS from segment
    result["max_raw_gyro_dps"] = segment["data"][["gyro_x", "gyro_y", "gyro_z"]].abs().max().max()
    
    return result


def plot_segment_analysis(segment_result: Dict, save_path: Optional[Path] = None):
    """Create detailed plot of segment analysis"""
    if segment_result is None:
        return
    
    processed_data = segment_result["processed_data"]
    test_id = segment_result["test_id"]
    description = segment_result["description"]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f"Test {test_id}: {description}", fontsize=14, fontweight='bold')
    
    # Time axis
    time = processed_data["timestamp"].values
    
    # Plot angles
    angle_names = ["yaw_deg", "pitch_deg", "roll_deg"]
    angle_labels = ["Yaw (Azimuth)", "Pitch (Altitude)", "Roll"]
    colors = ['blue', 'green', 'red']
    
    for i, (angle_name, label, color) in enumerate(zip(angle_names, angle_labels, colors)):
        ax = axes[i, 0]
        ax.plot(time, processed_data[angle_name], color=color, linewidth=1.5, label='Fused')
        
        # Add simple and complementary filter outputs if available
        if f"simple_{angle_name.split('_')[0]}" in processed_data.columns:
            ax.plot(time, processed_data[f"simple_{angle_name.split('_')[0]}"], 
                   '--', color=color, alpha=0.5, linewidth=0.8, label='Simple')
        if f"comp_{angle_name.split('_')[0]}" in processed_data.columns:
            ax.plot(time, processed_data[f"comp_{angle_name.split('_')[0]}"], 
                   ':', color=color, alpha=0.5, linewidth=0.8, label='Comp')
        
        # Mark analysis windows
        if angle_name == "pitch_deg" and "theta_analysis" in segment_result:
            analysis = segment_result["theta_analysis"]
            ax.axhspan(analysis["angle_before"] - 0.5, analysis["angle_before"] + 0.5, 
                      alpha=0.2, color='blue', label='Before')
            ax.axhspan(analysis["angle_after"] - 0.5, analysis["angle_after"] + 0.5, 
                      alpha=0.2, color='green', label='After')
            
            # Mark motion window
            motion_start_t = time[analysis["motion_window"][0]] if analysis["motion_window"][0] < len(time) else time[-1]
            motion_end_t = time[analysis["motion_window"][1]-1] if analysis["motion_window"][1] <= len(time) else time[-1]
            ax.axvspan(motion_start_t, motion_end_t, alpha=0.1, color='red')
            
            # Add detected change annotation
            ax.text(0.02, 0.98, f"Expected: {segment_result.get('expected_theta', 'N/A'):.1f}°\n"
                              f"Detected: {segment_result.get('detected_theta', 0):.1f}°\n"
                              f"Error: {segment_result.get('theta_error', 0):.1f}°",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        elif angle_name == "yaw_deg" and "phi_analysis" in segment_result:
            analysis = segment_result["phi_analysis"]
            ax.axhspan(analysis["angle_before"] - 0.5, analysis["angle_before"] + 0.5, 
                      alpha=0.2, color='blue')
            ax.axhspan(analysis["angle_after"] - 0.5, analysis["angle_after"] + 0.5, 
                      alpha=0.2, color='green')
            
            # Mark motion window
            motion_start_t = time[analysis["motion_window"][0]] if analysis["motion_window"][0] < len(time) else time[-1]
            motion_end_t = time[analysis["motion_window"][1]-1] if analysis["motion_window"][1] <= len(time) else time[-1]
            ax.axvspan(motion_start_t, motion_end_t, alpha=0.1, color='red')
            
            # Add detected change annotation
            ax.text(0.02, 0.98, f"Expected: {segment_result.get('expected_phi', 'N/A'):.1f}°\n"
                              f"Detected: {segment_result.get('detected_phi', 0):.1f}°\n"
                              f"Error: {segment_result.get('phi_error', 0):.1f}°",
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_ylabel(f'{label} (deg)')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    # Plot angular velocities
    for i, (angle_name, label, color) in enumerate(zip(angle_names, angle_labels, colors)):
        ax = axes[i, 1]
        angle_derivative = np.gradient(processed_data[angle_name].values) * 104  # Convert to deg/s
        ax.plot(time, angle_derivative, color=color, linewidth=1.0)
        
        # Add DPS statistics
        if angle_name == "pitch_deg" and "theta_peak_dps" in segment_result:
            ax.axhline(segment_result["theta_peak_dps"], color='red', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f"Peak: {segment_result['theta_peak_dps']:.1f} dps\n"
                              f"Mean: {segment_result['theta_mean_dps']:.1f} dps\n"
                              f"Effective: {segment_result['theta_effective_dps']:.1f} dps\n"
                              f"Duration: {segment_result['theta_motion_duration']:.1f} s",
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        elif angle_name == "yaw_deg" and "phi_peak_dps" in segment_result:
            ax.axhline(segment_result["phi_peak_dps"], color='red', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f"Peak: {segment_result['phi_peak_dps']:.1f} dps\n"
                              f"Mean: {segment_result['phi_mean_dps']:.1f} dps\n"
                              f"Effective: {segment_result['phi_effective_dps']:.1f} dps\n"
                              f"Duration: {segment_result['phi_motion_duration']:.1f} s",
                   transform=ax.transAxes, verticalalignment='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_ylabel(f'{label} Rate (dps)')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{label} Angular Velocity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_summary_report(results: List[Dict], output_prefix: str):
    """Generate comprehensive summary report"""
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to analyze")
        return
    
    # Convert to DataFrame for easier analysis
    summary_data = []
    for r in valid_results:
        summary = {
            "test_id": r["test_id"],
            "description": r["description"],
            "power_state": r["power_state"]
        }
        
        # Add theta (altitude) data if present
        if "expected_theta" in r:
            summary.update({
                "expected_theta": r["expected_theta"],
                "detected_theta": r["detected_theta"],
                "theta_error": r["theta_error"],
                "theta_peak_dps": r["theta_peak_dps"],
                "theta_effective_dps": r["theta_effective_dps"],
                "theta_duration": r["theta_motion_duration"]
            })
        
        # Add phi (azimuth) data if present
        if "expected_phi" in r:
            summary.update({
                "expected_phi": r["expected_phi"],
                "detected_phi": r["detected_phi"],
                "phi_error": r["phi_error"],
                "phi_peak_dps": r["phi_peak_dps"],
                "phi_effective_dps": r["phi_effective_dps"],
                "phi_duration": r["phi_motion_duration"]
            })
        
        summary["max_raw_gyro_dps"] = r["max_raw_gyro_dps"]
        summary_data.append(summary)
    
    df = pd.DataFrame(summary_data)
    
    # Save to CSV
    df.to_csv(f"{output_prefix}_summary.csv", index=False)
    print(f"\nSummary saved to {output_prefix}_summary.csv")
    
    # Print statistics
    print("\n" + "="*80)
    print("ANGLE CHANGE DETECTION SUMMARY")
    print("="*80)
    
    if "theta_error" in df.columns:
        print("\n--- ALTITUDE (THETA) ANALYSIS ---")
        print(f"Mean Error: {df['theta_error'].mean():.2f}°")
        print(f"Std Error: {df['theta_error'].std():.2f}°")
        print(f"Max Error: {df['theta_error'].max():.2f}°")
        print(f"Mean Peak DPS: {df['theta_peak_dps'].mean():.1f} dps")
        print(f"Mean Effective DPS: {df['theta_effective_dps'].mean():.1f} dps")
        print(f"Mean Motion Duration: {df['theta_duration'].mean():.2f} s")
    
    if "phi_error" in df.columns:
        print("\n--- AZIMUTH (PHI) ANALYSIS ---")
        print(f"Mean Error: {df['phi_error'].mean():.2f}°")
        print(f"Std Error: {df['phi_error'].std():.2f}°")
        print(f"Max Error: {df['phi_error'].max():.2f}°")
        print(f"Mean Peak DPS: {df['phi_peak_dps'].mean():.1f} dps")
        print(f"Mean Effective DPS: {df['phi_effective_dps'].mean():.1f} dps")
        print(f"Mean Motion Duration: {df['phi_duration'].mean():.2f} s")
    
    print("\n--- RELIABILITY RANGES ---")
    
    # Determine reliable operating ranges
    for error_threshold in [1.0, 2.0, 5.0]:
        print(f"\nError < {error_threshold}°:")
        
        if "theta_error" in df.columns:
            reliable_theta = df[df['theta_error'] < error_threshold]
            if not reliable_theta.empty:
                print(f"  Altitude: {reliable_theta['expected_theta'].abs().min():.1f}° - "
                      f"{reliable_theta['expected_theta'].abs().max():.1f}°")
                print(f"    DPS range: {reliable_theta['theta_effective_dps'].min():.1f} - "
                      f"{reliable_theta['theta_effective_dps'].max():.1f} dps")
        
        if "phi_error" in df.columns:
            reliable_phi = df[df['phi_error'] < error_threshold]
            if not reliable_phi.empty:
                print(f"  Azimuth: {reliable_phi['expected_phi'].abs().min():.1f}° - "
                      f"{reliable_phi['expected_phi'].abs().max():.1f}°")
                print(f"    DPS range: {reliable_phi['phi_effective_dps'].min():.1f} - "
                      f"{reliable_phi['phi_effective_dps'].max():.1f} dps")
    
    return df


def main():
    """Main analysis pipeline"""
    parser = argparse.ArgumentParser(description='Improved Angle Change Reliability Analysis')
    parser.add_argument('--config', type=str, help='Test configuration JSON file')
    parser.add_argument('--data', type=str, help='Raw IMU data CSV file')
    parser.add_argument('--output', type=str, default='analysis', help='Output file prefix')
    parser.add_argument('--plot', action='store_true', help='Generate plots for each segment')
    parser.add_argument('--use-edn', action='store_true', help='Convert from EDN to NED coordinates')
    
    args = parser.parse_args()
    
    # Default to altitude test if no config specified
    if not args.config:
        args.config = 'test_config_altitude.json'
    if not args.data:
        args.data = 'raw_data_output_altitude.csv'
    
    config_path = Path(args.config)
    data_path = Path(args.data)
    
    # Load configuration and data
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    print(f"Loading data from {data_path}")
    df = load_dataframe(data_path)
    
    # Apply coordinate transformation if needed
    if args.use_edn:
        print("Converting from EDN to NED coordinates")
        df = edn_to_ned(df)
    
    # Regularize timestamps
    fs = 104.0
    df["timestamp"] = regularize_timestamps(df["timestamp"].to_numpy(), fs)
    
    # Segment data
    segments = map_segments_to_config(df, config)
    print(f"Found {len(segments)} test segments")
    
    # Initialize motion estimator and detector
    estimator_config = Config()
    estimator = MotionEstimator(estimator_config)
    detector = AngleChangeDetector(fs)
    
    # Process each segment
    results = []
    for i, segment in enumerate(segments):
        print(f"\nProcessing segment {i}: Test {segment['test_id']} - {segment['config'].get('description', 'Unknown')}")
        
        result = analyze_segment(segment, estimator, detector)
        results.append(result)
        
        # Generate plot if requested
        if args.plot and result is not None:
            plot_path = Path(f"{args.output}_test_{segment['test_id']}.png")
            plot_segment_analysis(result, plot_path)
    
    # Generate summary report
    summary_df = generate_summary_report(results, args.output)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())