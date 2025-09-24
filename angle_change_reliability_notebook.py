# %%
import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import pandas as pd

from motion_estimator import MotionEstimator

# %%
config_file = Path('test_config_altitude.json')
csv = Path('raw_data_output_altitude.csv')

# config_file = Path('test_config_azimuth.json')
# csv = Path('raw_data_output_azimuth.csv')
use_edn = True
fs = 104

# %%
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
    change_time = config['change_time'] * 104
    n_samples = len(segment_df)
    time_passed = int(change_time * 0.1)
    
    # Calculate mean angles in stable regions
    start_angles = segment_df.iloc[change_time-time_passed: change_time+time_passed][["pitch_deg", "yaw_deg", "roll_deg"]].abs().max()
    # end_angles = segment_df.iloc[change_time+2*time_passed:stable_end][["pitch_deg", "yaw_deg", "roll_deg"]].mean()
    
    print(segment_df.head())
    # Calculate changes
    detected_changes = {
    	"theta_change": start_angles["pitch_deg"],
    	"phi_change": start_angles["yaw_deg"],
    	"psi_change": start_angles["roll_deg"]
    }

    return detected_changes
def analyze_segment_errors(segment: dict, estimator: MotionEstimator, fs: float = 104) -> dict:
    """Analyze errors for a single segment"""
    test_config = segment["config"]

    processed_df = process_segment_with_estimator(segment["data"], estimator)
    detected = detect_angle_changes(processed_df, test_config, fs)
    detected['max_dps'] = segment["data"][["gyro_x", "gyro_y", "gyro_z"]].max().max()

    detection_threshold = 2.0  # degrees
    result = {
        "test_id": segment["test_id"],
        "description": test_config.get("description", ""),
        "max_dps": detected["max_dps"],
        "expected_detectable": test_config.get("expected_detectable", True),
        "processed_data": processed_df
    }

    if test_config.get("power_state", "off") != "on":
        return None

    # θ (altitude)
    if "theta_change" in test_config:
        expected_theta = test_config["theta_change"]
        theta_error = abs(detected["theta_change"] - expected_theta)
        theta_detected = theta_error < detection_threshold if expected_theta != 0 else theta_error < 0.5
        result.update({
            "expected_theta": expected_theta,
            "detected_theta": detected["theta_change"],
            "theta_error": theta_error,
            "theta_detected": theta_detected,
        })

    # φ (azimuth)
    if "phi_change" in test_config:
        expected_phi = test_config["phi_change"]
        phi_error = abs(detected["phi_change"] - expected_phi)
        phi_detected = phi_error < detection_threshold if expected_phi != 0 else phi_error < 0.5
        result.update({
            "expected_phi": expected_phi,
            "detected_phi": detected["phi_change"],
            "phi_error": phi_error,
            "phi_detected": phi_detected,
        })
    return result


# %% [markdown]
# 

# %%
config = load_config(config_file)
config

# %%
df = load_dataframe(csv)
df.head()

# %%
if use_edn:
    df = edn_to_ned(df)

df.head() 

# %%
df["timestamp"] = regularize_timestamps(df["timestamp"].to_numpy(), fs)
df.head()

# %%
segments = map_segments_to_config(df, config)
print(f"Loaded {len(df)} samples, {len(segments)} segments from {csv}")
print(segments[0].keys())
segments[0]

# %%
estimator = MotionEstimator()

# %%


# %%
processed_data_dict = {}
results = []

for segment in segments:
    print(f"\nProcessing segment {segment['test_id']}: {segment['config'].get('description', '')}")
    
    # Process segment
    result = analyze_segment_errors(segment, estimator)

    # Store processed data separately
    if result is not None:
        processed_data_dict[segment['test_id']] = result.pop('processed_data', None)
        results.append(result)
        if 'expected_theta' in result.keys():
            plt.scatter(result['expected_theta'], result['detected_theta'])
        if 'expected_phi' in result.keys():
            plt.scatter(result['expected_phi'], result['detected_phi'])
        plt.show()

results_df = pd.DataFrame(results)

# %%
    for k,d in processed_data_dict.items():
        if 'phi_change' in segments[k]['config'].keys():
            change_time = segments[k]['config']['change_time']
            change = segments[k]['config']['phi_change'] if 'phi_change' in segments[k]['config'].keys() else segments[k]['config']['theta_change']

            delay = 10

            angle_before = d['yaw_deg'][(change_time - delay)*104:change_time*104].mean()
            angle_after = d['yaw_deg'][change_time*104:(change_time + delay)*104].mean()

            angle_delta = int(abs(angle_after-angle_before))
            plt.plot(d['yaw_deg'], label='yaw')
        if 'theta_change' in segments[k]['config'].keys():
            change_time = segments[k]['config']['change_time']
            change = segments[k]['config']['theta_change'] if 'theta_change' in segments[k]['config'].keys() else segments[k]['config']['theta_change']

            delay = 10

            angle_before = d['pitch_deg'][(change_time - delay)*104:change_time*104].mean()
            angle_after = d['pitch_deg'][change_time*104:(change_time + delay)*104].mean()

            angle_delta = int(abs(angle_after-angle_before))
            plt.plot(d['pitch_deg'], label='pitch')
        plt.axvline(change_time*104, color="r")
        plt.title(f"Test {k}: change at {change_time} with expected {change} degrees with actual {angle_delta} degrees")
        plt.legend()
        plt.show()


