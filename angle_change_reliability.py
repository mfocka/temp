#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from motion_estimator import MotionEstimator

def load_config(config_path: Path | None) -> dict:
	default = Path("test_config_altitude.json")
	if config_path and config_path.exists():
		with open(config_path, "r") as f:
			return json.load(f)
	if default.exists():
		with open(default, "r") as f:
			return json.load(f)
	return {"test_sequence": []}


def load_dataframe(csv_path: Path) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	required = ["timestamp", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]
	missing = [c for c in required if c not in df.columns]
	if missing:
		raise ValueError(f"Missing columns in {csv_path}: {missing}")
	for c in required[1:]:
		df[c] = pd.to_numeric(df[c], errors="coerce")
	df = df.dropna(subset=required)
	return df


def edn_to_ned(df: pd.DataFrame) -> pd.DataFrame:
	# Copy to avoid modifying caller
	df = df.copy()
	# Accelerometer
	df["accel_x"], df["accel_y"], df["accel_z"] = (
		df["accel_z"],
		df["accel_x"],
		-df["accel_y"],
	)
	# Gyroscope
	df["gyro_x"], df["gyro_y"], df["gyro_z"] = (
		df["gyro_z"],
		df["gyro_x"],
		-df["gyro_y"],
	)
	return df


def regularize_timestamps(ts: np.ndarray, fs: float) -> np.ndarray:
	current = float(ts[0])
	updated: list[float] = []
	for i in range(1, len(ts)):
		if ts[i] + 10 < ts[i - 1]:
			current = float(ts[i])
		updated.append(current)
		current += 1.0 / fs
	updated.append(current)
	return np.array(updated, dtype=float)


def segment_by_gaps(df: pd.DataFrame, gap_s: float = 5.0) -> list[tuple[int, int]]:
	ts = df["timestamp"].to_numpy()
	gaps = np.where(np.abs(np.diff(ts)) > gap_s)[0]
	bounds: list[tuple[int, int]] = []
	start = 0
	for gi in gaps:
		bounds.append((start, gi + 1))
		start = gi + 1
	bounds.append((start, len(df)))
	return bounds


def map_segments_to_config(df: pd.DataFrame, config: dict) -> list[dict]:
	bounds = segment_by_gaps(df)
	seq = config.get("test_sequence", [])
	segments: list[dict] = []
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
        # Convert from mg to g for consistency with MotionEstimator
        accel_raw = [row["accel_x"], row["accel_y"], row["accel_z"]]  # Already in mg
        gyro_raw = [row["gyro_x"], row["gyro_y"], row["gyro_z"]]      # Already in dps
        
        output = estimator.update(accel_raw, gyro_raw, row["timestamp"])
        output["sample_idx"] = i
        results.append(output)
    
    return pd.DataFrame(results)

def detect_angle_changes(segment_df: pd.DataFrame, config: dict, fs: float = 104.0) -> dict:
    """Detect angle changes in processed segment data"""
    # Get stable regions (first and last 20% of data)
    n_samples = len(segment_df)
    stable_start = int(n_samples * 0.1)
    stable_end = int(n_samples * 0.9)
    
    # Calculate mean angles in stable regions
    start_angles = segment_df.iloc[:stable_start][["pitch_deg", "yaw_deg", "roll_deg"]].mean()
    end_angles = segment_df.iloc[stable_end:][["pitch_deg", "yaw_deg", "roll_deg"]].mean()
    
    # Calculate changes
    detected_changes = {
        "theta_change": end_angles["pitch_deg"] - start_angles["pitch_deg"],
        "phi_change": end_angles["yaw_deg"] - start_angles["yaw_deg"],
        "psi_change": end_angles["roll_deg"] - start_angles["roll_deg"],
        "max_dps": segment_df[["pitch_deg", "yaw_deg", "roll_deg"]].diff().abs().max() * fs
    }
    
    return detected_changes

def analyze_segment_errors(segment: dict, estimator: MotionEstimator, fs: float = 104) -> dict:
    """Analyze errors for a single segment"""
    test_config = segment["config"]
    
    # Process through estimator
    processed_df = process_segment_with_estimator(segment["data"], estimator)
    
    # Detect actual changes
    detected = detect_angle_changes(processed_df, test_config, fs)
    
    # Get expected changes
    expected_theta = test_config.get("theta_change", 0)
    expected_phi = test_config.get("phi_change", 0)
    
    # Calculate errors
    theta_error = abs(detected["theta_change"] - expected_theta)
    phi_error = abs(detected["phi_change"] - expected_phi)
    
    # Determine if detection was successful (within threshold)
    detection_threshold = 2.0  # degrees
    theta_detected = theta_error < detection_threshold if expected_theta != 0 else theta_error < 0.5
    phi_detected = phi_error < detection_threshold if expected_phi != 0 else phi_error < 0.5
    
    return {
        "test_id": segment["test_id"],
        "description": test_config.get("description", ""),
        "expected_theta": expected_theta,
        "expected_phi": expected_phi,
        "detected_theta": detected["theta_change"],
        "detected_phi": detected["phi_change"],
        "theta_error": theta_error,
        "phi_error": phi_error,
        "max_dps": detected["max_dps"],
        "theta_detected": theta_detected,
        "phi_detected": phi_detected,
        "expected_detectable": test_config.get("expected_detectable", True),
        "processed_data": processed_df
    }
def generate_confusion_matrix(results_df: pd.DataFrame) -> np.ndarray:
    """Generate confusion matrix for angle detection"""
    # For altitude (theta)
    theta_tp = ((results_df["expected_theta"] != 0) & results_df["theta_detected"]).sum()
    theta_fp = ((results_df["expected_theta"] == 0) & ~results_df["theta_detected"]).sum()
    theta_tn = ((results_df["expected_theta"] == 0) & results_df["theta_detected"]).sum()
    theta_fn = ((results_df["expected_theta"] != 0) & ~results_df["theta_detected"]).sum()
    
    # For azimuth (phi)
    phi_tp = ((results_df["expected_phi"] != 0) & results_df["phi_detected"]).sum()
    phi_fp = ((results_df["expected_phi"] == 0) & ~results_df["phi_detected"]).sum()
    phi_tn = ((results_df["expected_phi"] == 0) & results_df["phi_detected"]).sum()
    phi_fn = ((results_df["expected_phi"] != 0) & ~results_df["phi_detected"]).sum()
    
    return {
        "theta": np.array([[theta_tp, theta_fp], [theta_fn, theta_tn]]),
        "phi": np.array([[phi_tp, phi_fp], [phi_fn, phi_tn]])
    }

def create_error_heatmap(results_df: pd.DataFrame) -> tuple:
    """Create error heatmap vs angle magnitude and DPS"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for heatmap
    angle_bins = [0, 1, 2, 3, 5, 10, 15, 20, 30, 45]
    dps_bins = [0, 5, 10, 20, 50, 100, 200]
    
    # Create binned columns
    results_df["angle_bin"] = pd.cut(
        results_df[["expected_theta", "expected_phi"]].abs().max(axis=1),
        bins=angle_bins,
        labels=[f"{angle_bins[i]}-{angle_bins[i+1]}" for i in range(len(angle_bins)-1)]
    )
    
    results_df["dps_bin"] = pd.cut(
        results_df["max_dps"],
        bins=dps_bins,
        labels=[f"{dps_bins[i]}-{dps_bins[i+1]}" for i in range(len(dps_bins)-1)]
    )
    
    # Calculate mean error for each bin combination
    error_matrix = results_df.pivot_table(
        values=["theta_error", "phi_error"],
        index="angle_bin",
        columns="dps_bin",
        aggfunc="mean"
    )
    
    return error_matrix

def plot_analysis_results(results_df: pd.DataFrame, output_prefix: str):
    """Create comprehensive visualization of results"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Error vs Angle Magnitude
    ax = axes[0, 0]
    ax.scatter(results_df["expected_theta"].abs(), results_df["theta_error"], 
               alpha=0.6, label="Altitude")
    ax.scatter(results_df["expected_phi"].abs(), results_df["phi_error"], 
               alpha=0.6, label="Azimuth")
    ax.set_xlabel("Expected Angle (deg)")
    ax.set_ylabel("Error (deg)")
    ax.set_title("Error vs Angle Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Error vs DPS
    ax = axes[0, 1]
    ax.scatter(results_df["max_dps"], results_df["theta_error"], 
               alpha=0.6, label="Altitude")
    ax.scatter(results_df["max_dps"], results_df["phi_error"], 
               alpha=0.6, label="Azimuth")
    ax.set_xlabel("Max DPS")
    ax.set_ylabel("Error (deg)")
    ax.set_title("Error vs Angular Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Detection Success Rate
    ax = axes[0, 2]
    detection_data = pd.DataFrame({
        "Altitude": [results_df["theta_detected"].sum(), (~results_df["theta_detected"]).sum()],
        "Azimuth": [results_df["phi_detected"].sum(), (~results_df["phi_detected"]).sum()]
    }, index=["Detected", "Missed"])
    detection_data.plot(kind="bar", ax=ax)
    ax.set_title("Detection Success Rate")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    # 4. Confusion Matrix - Altitude
    ax = axes[1, 0]
    cm = generate_confusion_matrix(results_df)
    sns.heatmap(cm["theta"], annot=True, fmt="d", ax=ax, cmap="Blues")
    ax.set_title("Altitude Detection Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    # 5. Confusion Matrix - Azimuth
    ax = axes[1, 1]
    sns.heatmap(cm["phi"], annot=True, fmt="d", ax=ax, cmap="Greens")
    ax.set_title("Azimuth Detection Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    
    # 6. Error Distribution
    ax = axes[1, 2]
    ax.hist(results_df["theta_error"], bins=20, alpha=0.5, label="Altitude")
    ax.hist(results_df["phi_error"], bins=20, alpha=0.5, label="Azimuth")
    ax.set_xlabel("Error (deg)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error Distribution")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_analysis.png", dpi=150)
    plt.show()
    
    # Create error heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    error_matrix = create_error_heatmap(results_df)
    if not error_matrix.empty:
        sns.heatmap(error_matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title("Error Heatmap: Angle vs DPS")
        plt.savefig(f"{output_prefix}_heatmap.png", dpi=150)
        plt.show()

def generate_analysis_report(results_df: pd.DataFrame, output_prefix: str):
    """Generate comprehensive analysis report"""
    
    # Save detailed results to CSV
    results_df.to_csv(f"{output_prefix}_results.csv", index=False)
    print(f"\nDetailed results saved to {output_prefix}_results.csv")
    
    # Create summary statistics
    summary = {
        "Total Tests": len(results_df),
        "Altitude Mean Error": results_df["theta_error"].mean(),
        "Altitude Std Error": results_df["theta_error"].std(),
        "Altitude Max Error": results_df["theta_error"].max(),
        "Azimuth Mean Error": results_df["phi_error"].mean(),
        "Azimuth Std Error": results_df["phi_error"].std(),
        "Azimuth Max Error": results_df["phi_error"].max(),
        "Detection Rate": (results_df["theta_detected"] | results_df["phi_detected"]).mean() * 100,
        "False Positive Rate": ((results_df["expected_detectable"] == False) & 
                               (results_df["theta_detected"] | results_df["phi_detected"])).mean() * 100
    }
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    for key, value in summary.items():
        if "Error" in key:
            print(f"{key:<25}: {value:>8.3f} deg")
        elif "Rate" in key:
            print(f"{key:<25}: {value:>8.1f} %")
        else:
            print(f"{key:<25}: {value:>8}")
    
    # Generate reliability ranges
    reliability_ranges = determine_reliability_ranges(results_df)
    
    print("\n" + "="*60)
    print("RELIABILITY RANGES")
    print("="*60)
    for category, ranges in reliability_ranges.items():
        print(f"\n{category}:")
        for key, value in ranges.items():
            print(f"  {key}: {value}")
    
    # Generate visualizations
    plot_analysis_results(results_df, output_prefix)
    
    # Save summary to text file
    with open(f"{output_prefix}_summary.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("MOTION ESTIMATOR ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key:<25}: {value}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("RELIABILITY RANGES\n")
        f.write("="*60 + "\n")
        for category, ranges in reliability_ranges.items():
            f.write(f"\n{category}:\n")
            for key, value in ranges.items():
                f.write(f"  {key}: {value}\n")
    
    print(f"\nSummary saved to {output_prefix}_summary.txt")

def determine_reliability_ranges(results_df: pd.DataFrame) -> dict:
    """Determine reliability ranges based on error analysis"""
    ranges = {}
    
    for error_threshold in [0.5, 1.0, 2.0, 5.0]:
        mask = (results_df["theta_error"] <= error_threshold) | (results_df["phi_error"] <= error_threshold)
        if mask.any():
            reliable_data = results_df[mask]
            ranges[f"Error < {error_threshold}°"] = {
                "Angle Range": f"{reliable_data[['expected_theta', 'expected_phi']].abs().min().min():.1f} - "
                              f"{reliable_data[['expected_theta', 'expected_phi']].abs().max().max():.1f} deg",
                "DPS Range": f"{reliable_data['max_dps'].min():.1f} - {reliable_data['max_dps'].max():.1f} dps",
                "Success Rate": f"{(reliable_data['theta_detected'] | reliable_data['phi_detected']).mean()*100:.1f}%"
            }
    
    return ranges

def main(argv=None) -> int:
    config_file = Path('test_config_altitude.json')
    csv = Path('raw_data_output_altitude.csv')
    use_edn = True
    fs = 104
    
    # Load config and data
    config = load_config(config_file)
    df = load_dataframe(csv)
    
    if use_edn:
        df = edn_to_ned(df)
    
    # Regularize timestamps
    df["timestamp"] = regularize_timestamps(df["timestamp"].to_numpy(), fs)
    
    # Segment data
    segments = map_segments_to_config(df, config)
    print(f"Loaded {len(df)} samples, {len(segments)} segments from {csv}")
    
    # Initialize MotionEstimator
    estimator = MotionEstimator()
    
    # Process all segments
    results = []
    for segment in segments:
        print(f"\nProcessing segment {segment['test_id']}: {segment['config'].get('description', '')}")
        result = analyze_segment_errors(segment, estimator, fs)
        results.append(result)
    
    # Generate analysis tables and visualizations
    results_df = pd.DataFrame(results)
    generate_analysis_report(results_df, config_file.stem)
    
    return 0


if __name__ == "__main__":
	sys.exit(main())
