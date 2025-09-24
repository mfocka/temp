#!/usr/bin/env python3
"""
Fixed Angle Change Analysis for Motion Estimator
Properly analyzes angle changes considering the actual motion behavior
"""

import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
from motion_estimator import MotionEstimator, Config


def load_config(config_path: Path) -> dict:
    """Load test configuration"""
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {"test_sequence": []}


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load IMU data"""
    df = pd.read_csv(csv_path)
    required = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required)
    return df


def edn_to_ned(df: pd.DataFrame) -> pd.DataFrame:
    """Convert EDN to NED coordinates"""
    df = df.copy()
    df["accel_x"], df["accel_y"], df["accel_z"] = (
        df["accel_z"], df["accel_x"], -df["accel_y"]
    )
    df["gyro_x"], df["gyro_y"], df["gyro_z"] = (
        df["gyro_z"], df["gyro_x"], -df["gyro_y"]
    )
    return df


def regularize_timestamps(ts: np.ndarray, fs: float) -> np.ndarray:
    """Fix timestamp wraparound issues"""
    current = float(ts[0])
    updated = []
    for i in range(1, len(ts)):
        if ts[i] + 10 < ts[i - 1]:
            current = float(ts[i])
        updated.append(current)
        current += 1.0 / fs
    updated.append(current)
    return np.array(updated, dtype=float)


def segment_by_gaps(df: pd.DataFrame, gap_s: float = 5.0) -> list:
    """Segment data by timestamp gaps"""
    ts = df["timestamp"].to_numpy()
    gaps = np.where(np.abs(np.diff(ts)) > gap_s)[0]
    bounds = []
    start = 0
    for gi in gaps:
        bounds.append((start, gi + 1))
        start = gi + 1
    bounds.append((start, len(df)))
    return bounds


def map_segments_to_config(df: pd.DataFrame, config: dict) -> list:
    """Map segments to test configuration"""
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
    """Process segment through MotionEstimator"""
    results = []
    for i, row in segment_data.iterrows():
        accel = [row["accel_x"], row["accel_y"], row["accel_z"]]
        gyro = [row["gyro_x"], row["gyro_y"], row["gyro_z"]]
        output = estimator.update(accel, gyro, row["timestamp"])
        output["sample_idx"] = i
        results.append(output)
    return pd.DataFrame(results)


def analyze_angle_change(angle_signal: np.ndarray, change_time_s: float, fs: float = 104.0) -> dict:
    """
    Properly analyze angle change considering motion dynamics.
    
    Key insights:
    1. The motion estimator outputs RELATIVE angles from reference
    2. For a stationary reference, angle should be near 0
    3. When motion occurs, angle increases from 0 to the change amount
    4. The change_time is when motion STARTS
    """
    change_sample = int(change_time_s * fs)
    
    # Define windows for analysis
    # Before motion: should be near 0 (relative to reference)
    before_start = max(0, change_sample - int(5 * fs))
    before_end = max(0, change_sample - int(0.5 * fs))
    
    # After motion: look for stable value after change
    # Motion typically takes 2-5 seconds
    after_start = min(len(angle_signal), change_sample + int(3 * fs))
    after_end = min(len(angle_signal), change_sample + int(8 * fs))
    
    # Get stable values
    if before_end > before_start:
        angle_before = np.median(angle_signal[before_start:before_end])
    else:
        angle_before = 0.0
    
    if after_end > after_start:
        # For relative angles, we want the maximum stable value reached
        angle_after = np.percentile(angle_signal[after_start:after_end], 75)
    else:
        angle_after = angle_signal[-1] if len(angle_signal) > 0 else 0.0
    
    # The detected change is the difference
    angle_change = angle_after - angle_before
    
    # Calculate angular velocity during motion
    motion_window = angle_signal[change_sample:after_start]
    if len(motion_window) > 1:
        angular_velocity = np.gradient(motion_window) * fs
        peak_dps = np.abs(angular_velocity).max()
        mean_dps = np.abs(angular_velocity).mean()
    else:
        peak_dps = 0.0
        mean_dps = 0.0
    
    # Motion duration
    motion_duration = (after_start - change_sample) / fs
    
    # Effective DPS
    effective_dps = abs(angle_change) / motion_duration if motion_duration > 0 else 0
    
    return {
        'angle_before': angle_before,
        'angle_after': angle_after,
        'angle_change': angle_change,
        'peak_dps': peak_dps,
        'mean_dps': mean_dps,
        'effective_dps': effective_dps,
        'motion_duration': motion_duration
    }


def plot_segment_analysis(segment: dict, processed_df: pd.DataFrame, save_path: Path = None):
    """Create detailed plot of segment analysis"""
    config = segment['config']
    test_id = segment['test_id']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Test {test_id}: {config.get('description', '')}", fontsize=14)
    
    # Time axis
    time = np.arange(len(processed_df)) / 104.0
    change_time = config.get('change_time', 20)
    
    # Plot Yaw (Azimuth)
    ax = axes[0, 0]
    ax.plot(time, processed_df['yaw_deg'], 'b-', label='Yaw')
    ax.axvline(change_time, color='r', linestyle='--', alpha=0.5, label='Change Time')
    if 'phi_change' in config:
        ax.axhline(abs(config['phi_change']), color='g', linestyle=':', alpha=0.5, 
                  label=f"Expected: {config['phi_change']}°")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Yaw (deg)')
    ax.set_title('Yaw (Azimuth) Angle')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot Pitch (Altitude)
    ax = axes[0, 1]
    ax.plot(time, processed_df['pitch_deg'], 'g-', label='Pitch')
    ax.axvline(change_time, color='r', linestyle='--', alpha=0.5, label='Change Time')
    if 'theta_change' in config:
        ax.axhline(abs(config['theta_change']), color='g', linestyle=':', alpha=0.5,
                  label=f"Expected: {config['theta_change']}°")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (deg)')
    ax.set_title('Pitch (Altitude) Angle')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot Roll
    ax = axes[1, 0]
    ax.plot(time, processed_df['roll_deg'], 'r-', label='Roll')
    ax.axvline(change_time, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Roll (deg)')
    ax.set_title('Roll Angle')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot angular velocities
    ax = axes[1, 1]
    if 'theta_change' in config:
        pitch_rate = np.gradient(processed_df['pitch_deg'].values) * 104
        ax.plot(time, pitch_rate, 'g-', alpha=0.7, label='Pitch rate')
    if 'phi_change' in config:
        yaw_rate = np.gradient(processed_df['yaw_deg'].values) * 104
        ax.plot(time, yaw_rate, 'b-', alpha=0.7, label='Yaw rate')
    ax.axvline(change_time, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Rate (dps)')
    ax.set_title('Angular Velocities')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    """Main analysis with fixed angle detection"""
    
    # Configuration
    config_file = Path('test_config_altitude.json')
    # config_file = Path('test_config_azimuth.json')
    csv_file = Path('raw_data_output_altitude.csv')
    # csv_file = Path('raw_data_output_azimuth.csv')
    
    use_edn = True
    fs = 104.0
    
    print("="*80)
    print("FIXED ANGLE CHANGE ANALYSIS")
    print("="*80)
    
    # Load data
    config = load_config(config_file)
    df = load_dataframe(csv_file)
    
    if use_edn:
        df = edn_to_ned(df)
    
    df["timestamp"] = regularize_timestamps(df["timestamp"].to_numpy(), fs)
    segments = map_segments_to_config(df, config)
    
    print(f"Loaded {len(df)} samples, {len(segments)} segments")
    
    # Initialize estimator
    estimator = MotionEstimator()
    
    # Process segments
    results = []
    processed_data_dict = {}
    
    for segment in segments:
        test_config = segment['config']
        test_id = segment['test_id']
        
        print(f"\nTest {test_id}: {test_config.get('description', '')}")
        
        # Skip power-off tests
        if test_config.get('power_state', 'off') == 'off':
            print("  Skipping (power off)")
            continue
        
        # Process through estimator
        processed_df = process_segment_with_estimator(segment['data'], estimator)
        processed_data_dict[test_id] = processed_df
        
        # Analyze changes
        change_time = test_config.get('change_time', 20)
        
        result = {
            'test_id': test_id,
            'description': test_config.get('description', '')
        }
        
        # Analyze theta (pitch/altitude)
        if 'theta_change' in test_config:
            theta_analysis = analyze_angle_change(
                processed_df['pitch_deg'].values,
                change_time,
                fs
            )
            
            expected = abs(test_config['theta_change'])
            detected = theta_analysis['angle_after']  # For relative angles
            
            result['expected_theta'] = expected
            result['detected_theta'] = detected
            result['theta_error'] = abs(detected - expected)
            result['theta_dps'] = theta_analysis['effective_dps']
            
            print(f"  Theta: Expected={expected:.1f}°, Detected={detected:.1f}°, Error={result['theta_error']:.1f}°")
        
        # Analyze phi (yaw/azimuth)
        if 'phi_change' in test_config:
            phi_analysis = analyze_angle_change(
                processed_df['yaw_deg'].values,
                change_time,
                fs
            )
            
            expected = abs(test_config['phi_change'])
            detected = phi_analysis['angle_after']
            
            result['expected_phi'] = expected
            result['detected_phi'] = detected
            result['phi_error'] = abs(detected - expected)
            result['phi_dps'] = phi_analysis['effective_dps']
            
            print(f"  Phi: Expected={expected:.1f}°, Detected={detected:.1f}°, Error={result['phi_error']:.1f}°")
        
        results.append(result)
        
        # Plot if significant change
        if test_config.get('theta_change', 0) != 0 or test_config.get('phi_change', 0) != 0:
            plot_segment_analysis(segment, processed_df)
    
    # Summary statistics
    if results:
        df_results = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        if 'theta_error' in df_results.columns:
            print("\nAltitude (Theta) Analysis:")
            print(f"  Mean Error: {df_results['theta_error'].mean():.2f}°")
            print(f"  Std Error: {df_results['theta_error'].std():.2f}°")
            print(f"  Max Error: {df_results['theta_error'].max():.2f}°")
            print(f"  Mean DPS: {df_results['theta_dps'].mean():.1f} dps")
        
        if 'phi_error' in df_results.columns:
            print("\nAzimuth (Phi) Analysis:")
            print(f"  Mean Error: {df_results['phi_error'].mean():.2f}°")
            print(f"  Std Error: {df_results['phi_error'].std():.2f}°")
            print(f"  Max Error: {df_results['phi_error'].max():.2f}°")
            print(f"  Mean DPS: {df_results['phi_dps'].mean():.1f} dps")
        
        # Save results
        df_results.to_csv('angle_analysis_results.csv', index=False)
        print(f"\nResults saved to angle_analysis_results.csv")
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    main()