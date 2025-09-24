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
use_wds_mapping = True
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

def wds_to_enu(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wx, wy, wz = df["accel_x"].to_numpy(), df["accel_y"].to_numpy(), df["accel_z"].to_numpy()
    df["accel_x"] = -wx
    df["accel_y"] = -wz
    df["accel_z"] = -wy
    gx, gy, gz = df["gyro_x"].to_numpy(), df["gyro_y"].to_numpy(), df["gyro_z"].to_numpy()
    df["gyro_x"] = -gx
    df["gyro_y"] = -gz
    df["gyro_z"] = -gy
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

def process_segment_with_estimator(segment_data: pd.DataFrame, estimator: MotionEstimator, fs: float = 104.0) -> pd.DataFrame:
    results = []
    estimator.reset_calibration()
    cal_needed = 1040
    cal_count = 0
    for i, row in segment_data.iterrows():
        accel_raw = [row["accel_x"], row["accel_y"], row["accel_z"]]
        gyro_raw = [row["gyro_x"], row["gyro_y"], row["gyro_z"]]
        if not estimator.is_ready() and cal_count < cal_needed:
            estimator.add_calibration_sample(accel_raw, gyro_raw)
            cal_count += 1
            continue
        output = estimator.update(accel_raw, gyro_raw, row["timestamp"])
        output["sample_idx"] = i
        results.append(output)
    return pd.DataFrame(results)

def _compute_change_metrics(signal: np.ndarray, change_time_s: float, fs: float = 104.0) -> dict:
    n = len(signal)
    if n == 0:
        return {"angle_delta": 0.0, "avg_dps": 0.0, "max_dps": 0.0}
    win = max(3, int(0.5 * fs))
    kernel = np.ones(win) / win
    smooth = np.convolve(signal, kernel, mode="same")
    deriv = np.gradient(smooth) * fs
    ct = int(change_time_s * fs)
    w = int(3.0 * fs)
    s0 = max(0, ct - w)
    s1 = min(n, ct + w)
    if s1 - s0 < 5:
        s0, s1 = 0, n
    local = np.abs(deriv[s0:s1])
    if local.size == 0:
        return {"angle_delta": 0.0, "avg_dps": 0.0, "max_dps": 0.0}
    peak_local_idx = int(np.argmax(local))
    peak_idx = s0 + peak_local_idx
    peak_dps = float(local[peak_local_idx])
    thr = max(0.15 * peak_dps, 1.0)
    onset = s0
    for i in range(peak_idx, s0, -1):
        if np.all(np.abs(deriv[max(s0, i-5):i]) < thr):
            onset = i
            break
    offset = s1 - 1
    for i in range(peak_idx, s1 - 5):
        if np.all(np.abs(deriv[i:i+5]) < thr):
            offset = i
            break
    pre_end = max(0, onset - int(0.2 * fs))
    pre_start = max(0, pre_end - int(1.0 * fs))
    post_start = min(n, offset + int(0.2 * fs))
    post_end = min(n, post_start + int(1.0 * fs))
    base = float(np.mean(smooth[pre_start:pre_end])) if pre_end > pre_start else float(smooth[max(0, onset - int(1.0*fs)):onset].mean()) if onset > int(1.0*fs) else float(smooth[:max(1, onset)].mean())
    plat = float(np.mean(smooth[post_start:post_end])) if post_end > post_start else float(smooth[offset:min(n, offset+int(1.0*fs))].mean())
    angle_delta = abs(plat - base)
    move_time_s = max(1.0 / fs, (offset - onset) / fs)
    avg_dps = angle_delta / move_time_s
    max_dps = float(np.max(local))
    return {
        "angle_delta": float(angle_delta),
        "avg_dps": float(avg_dps),
        "max_dps": float(max_dps),
        "onset_idx": int(onset),
        "offset_idx": int(offset),
        "baseline": float(base),
        "plateau": float(plat),
        "peak_idx": int(peak_idx),
        "peak_dps": float(peak_dps),
    }

def detect_angle_changes(segment_df: pd.DataFrame, config: dict, fs: float = 104.0) -> dict:
    change_time_s = float(config['change_time'])
    yaw_metrics = _compute_change_metrics(segment_df["yaw_deg"].to_numpy(), change_time_s, fs) if "yaw_deg" in segment_df.columns else {"angle_delta": 0.0, "avg_dps": 0.0, "max_dps": 0.0}
    pitch_metrics = _compute_change_metrics(segment_df["pitch_deg"].to_numpy(), change_time_s, fs) if "pitch_deg" in segment_df.columns else {"angle_delta": 0.0, "avg_dps": 0.0, "max_dps": 0.0}
    roll_metrics = _compute_change_metrics(segment_df["roll_deg"].to_numpy(), change_time_s, fs) if "roll_deg" in segment_df.columns else {"angle_delta": 0.0, "avg_dps": 0.0, "max_dps": 0.0}
    return {
        "theta_change": pitch_metrics["angle_delta"],
        "phi_change": yaw_metrics["angle_delta"],
        "psi_change": roll_metrics["angle_delta"],
        "theta_dps": pitch_metrics["avg_dps"],
        "phi_dps": yaw_metrics["avg_dps"],
        "theta_max_dps": pitch_metrics["max_dps"],
        "phi_max_dps": yaw_metrics["max_dps"],
        "_pitch_metrics": pitch_metrics,
        "_yaw_metrics": yaw_metrics,
        "_roll_metrics": roll_metrics,
    }
def analyze_segment_errors(segment: dict, estimator: MotionEstimator, fs: float = 104) -> dict:
    """Analyze errors for a single segment"""
    test_config = segment["config"]

    processed_df = process_segment_with_estimator(segment["data"], estimator, fs)
    detected = detect_angle_changes(processed_df, test_config, fs)
    detected['max_dps_raw'] = float(segment["data"][ ["gyro_x", "gyro_y", "gyro_z"] ].abs().max().max())

    detection_threshold = 2.0  # degrees
    result = {
        "test_id": segment["test_id"],
        "description": test_config.get("description", ""),
        "max_dps": detected.get("phi_max_dps", detected.get("theta_max_dps", 0.0)),
        "max_dps_raw": detected["max_dps_raw"],
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
            "detected_theta_dps": detected.get("theta_dps", 0.0),
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
            "detected_phi_dps": detected.get("phi_dps", 0.0),
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
if use_wds_mapping:
    df = wds_to_enu(df)

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
    result = analyze_segment_errors(segment, estimator, fs)

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
# Visual verification: plot and annotate using computed onset/offset
for k, d in processed_data_dict.items():
    cfg = segments[k]['config']
    change_time = cfg['change_time']
    if 'phi_change' in cfg:
        metrics = _compute_change_metrics(d['yaw_deg'].to_numpy(), change_time, fs)
        angle_delta = metrics['angle_delta']
        plt.plot(d['yaw_deg'], label='yaw')
    if 'theta_change' in cfg:
        metrics = _compute_change_metrics(d['pitch_deg'].to_numpy(), change_time, fs)
        angle_delta = metrics['angle_delta']
        plt.plot(d['pitch_deg'], label='pitch')
    plt.axvline(int(change_time*fs), color='r')
    if metrics:
        plt.axvline(metrics['onset_idx'], color='g', linestyle='--', alpha=0.6)
        plt.axvline(metrics['offset_idx'], color='m', linestyle='--', alpha=0.6)
    expected = cfg.get('phi_change', cfg.get('theta_change', 0))
    plt.title(f"Test {k}: expected {expected}°, detected {angle_delta:.2f}°")
    plt.legend()
    plt.show()


