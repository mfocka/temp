#!/usr/bin/env python3
"""
Angle change reliability analysis: errors vs angle magnitude and DPS.

Usage:
  python angle_change_reliability.py --csv raw_data_output_altitude.csv \
      --config test_config_altitude.json --output ./analysis_output --fs 104 --edn

Inputs:
- --csv: path to CSV containing columns: timestamp, accel_x/y/z, gyro_x/y/z
- --config: optional JSON with test_sequence describing expected angles
- --output: directory to write CSVs and PNGs (created if missing)
- --edn/--no-edn: convert EDN→NED axes (default: enabled)
- --fs: sampling rate Hz (default 104)

Outputs in --output directory:
- results.csv: per-segment metrics (expected angle, measured, error, dps)
- error_by_angle.csv: binned error stats vs angle magnitude
- error_by_dps.csv: binned error stats vs max DPS
- error_heatmap.csv/.png: RMSE heatmap by angle_bin × dps_bin
- confusion_matrix.csv/.png: expected-change vs detected-change matrix
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from motion_estimator import MotionEstimator


def parse_args(argv=None):

	parser = argparse.ArgumentParser(
		description="Angle change reliability analysis: errors vs angle magnitude and DPS"
	)
	parser.add_argument("--csv", type=Path, required=True, help="Path to raw_data_output_altitude.csv (or azimuth)")
	parser.add_argument(
		"--config",
		type=Path,
		required=False,
		help="Path to test_config JSON (optional). Defaults to test_config_altitude.json if present.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		required=False,
		help="Output directory for results (CSV/plots). Defaults to ./analysis_output",
	)
	parser.add_argument("--edn", action="store_true", help="Input is EDN; convert to NED (default true)")
	parser.add_argument("--no-edn", action="store_true", help="Disable EDN->NED conversion")
	parser.add_argument("--fs", type=float, default=104.0, help="Sampling rate Hz (default 104)")
	return parser.parse_args(argv)


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


def compute_reference_angles(df: pd.DataFrame, fs: float, window_s: float = 1.0) -> dict:
	# Use early static window to compute reference from accel
	n = max(1, int(window_s * fs))
	ax = (df["accel_x"].iloc[:n].mean()) * 9.80665 / 1000.0
	ay = (df["accel_y"].iloc[:n].mean()) * 9.80665 / 1000.0
	az = (df["accel_z"].iloc[:n].mean()) * 9.80665 / 1000.0
	pitch_ref = float(np.degrees(np.arctan2(ay, az)))
	roll_ref = float(np.degrees(np.arctan2(-ax, np.sqrt(ay * ay + az * az))))
	return {"pitch": pitch_ref, "roll": roll_ref, "yaw": 0.0}


def detect_motion_window(df: pd.DataFrame, axis: str, fs: float, thresh: float = 2.0) -> tuple[int, int]:
	# Axis → gyro column mapping
	gyro_col = {"pitch": "gyro_x", "roll": "gyro_y", "yaw": "gyro_z"}.get(axis, "gyro_z")
	gyro = df[gyro_col].to_numpy()
	# Simple smoothing (moving average)
	win = max(3, int(0.1 * fs))
	if len(gyro) >= win:
		smooth = pd.Series(gyro).rolling(window=win, min_periods=1, center=True).mean().to_numpy()
	else:
		smooth = gyro
	in_motion = np.abs(smooth) > thresh
	changes = np.diff(np.concatenate([[False], in_motion, [False]])).astype(int)
	starts = np.where(changes == 1)[0]
	stops = np.where(changes == -1)[0]
	if len(starts) and len(stops):
		# choose the longest
		durs = stops - starts
		idx = int(np.argmax(durs))
		return int(starts[idx]), int(stops[idx])
	# fallback: whole segment
	return 0, len(df) - 1


def process_segment(seg: dict, fs: float) -> dict:
	est = MotionEstimator()
	df = seg["data"]
	ref = compute_reference_angles(df, fs)
	axis = seg.get("config", {}).get("axis") or seg.get("config", {}).get("angle_axis") or seg.get("config", {}).get("change_axis") or "pitch"
	start_i, stop_i = detect_motion_window(df, axis, fs)
	outputs: list[dict] = []
	for _, row in df.iterrows():
		accel = [row["accel_x"], row["accel_y"], row["accel_z"]]
		gyro = [row["gyro_x"], row["gyro_y"], row["gyro_z"]]
		out = est.update(accel, gyro, float(row["timestamp"]))
		out["pitch_rel"] = out["pitch_deg"] - ref["pitch"]
		out["roll_rel"] = out["roll_deg"] - ref["roll"]
		out["yaw_rel"] = out["yaw_deg"] - ref["yaw"]
		outputs.append(out)
	out_df = pd.DataFrame(outputs)
	axis_key = {"pitch": "pitch_rel", "roll": "roll_rel", "yaw": "yaw_rel"}[axis]
	# Measured angle: stabilized after stop_i + small guard
	guard = min(20, max(0, len(out_df) - 1 - stop_i))
	meas_idx = min(stop_i + guard, len(out_df) - 1)
	measured_angle = float(out_df.iloc[meas_idx][axis_key])
	# Expected angle (if provided)
	expected_angle = float(seg.get("config", {}).get("theta_change") or seg.get("config", {}).get("phi_change") or seg.get("config", {}).get("angle", 0.0))
	# DPS metrics from gyro during motion
	motion_slice = df.iloc[start_i:stop_i]
	gyro_axis = {"pitch": "gyro_x", "roll": "gyro_y", "yaw": "gyro_z"}[axis]
	max_dps = float(motion_slice[gyro_axis].abs().max()) if len(motion_slice) else 0.0
	mean_dps = float(motion_slice[gyro_axis].abs().mean()) if len(motion_slice) else 0.0
	# Errors
	error_deg = abs(measured_angle - expected_angle)
	error_pct = (error_deg / max(abs(expected_angle), 0.1)) * 100.0
	return {
		"test_id": int(seg.get("test_id", -1)),
		"axis": axis,
		"expected_angle": expected_angle,
		"measured_angle": measured_angle,
		"error_deg": error_deg,
		"error_percent": error_pct,
		"max_dps": max_dps,
		"mean_dps": mean_dps,
		"duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if len(df) else 0.0,
	}


def summarize_binned_tables(results: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	angle_bins = [0, 1, 2, 5, 10, 20, 30, 45, 90]
	dps_bins = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
	results = results.copy()
	results["angle_mag"] = results["expected_angle"].abs()
	results["angle_bin"] = pd.cut(results["angle_mag"], bins=angle_bins, include_lowest=True)
	results["dps_bin"] = pd.cut(results["max_dps"], bins=dps_bins, include_lowest=True)
	by_angle = results.groupby("angle_bin")["error_deg"].agg(["count", "mean", "median", "max", "std"]).reset_index()
	by_dps = results.groupby("dps_bin")["error_deg"].agg(["count", "mean", "median", "max", "std"]).reset_index()
	heatmap = results.pivot_table(values="error_deg", index="angle_bin", columns="dps_bin", aggfunc="mean")
	by_angle.to_csv(out_dir / "error_by_angle.csv", index=False)
	by_dps.to_csv(out_dir / "error_by_dps.csv", index=False)
	heatmap.to_csv(out_dir / "error_heatmap.csv")
	return by_angle, by_dps, heatmap


def save_heatmap_png(heatmap_df: pd.DataFrame, out_file: Path):
	plt.figure(figsize=(10, 6))
	sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd")
	plt.title("Mean absolute error (deg) vs Angle and DPS")
	plt.tight_layout()
	plt.savefig(out_file, dpi=150)
	plt.close()


def build_confusion_matrix(results: pd.DataFrame, angle_thresh: float = 0.5) -> Tuple[np.ndarray, List[str], List[str]]:
	# Expected change if |expected_angle| >= angle_thresh
	expected_pos = (results["expected_angle"].abs() >= angle_thresh).to_numpy()
	# Detected change if measured deviates from 0 by threshold (or use error vs expected?)
	detected_pos = (results["measured_angle"].abs() >= angle_thresh).to_numpy()
	TP = int(np.sum(detected_pos & expected_pos))
	FN = int(np.sum((~detected_pos) & expected_pos))
	FP = int(np.sum(detected_pos & (~expected_pos)))
	TN = int(np.sum((~detected_pos) & (~expected_pos)))
	cm = np.array([[TP, FN], [FP, TN]], dtype=int)
	return cm, ["Expected", "Not Expected"], ["Detected", "Not Detected"]


def save_confusion_png(cm: np.ndarray, col_labels: List[str], row_labels: List[str], out_file: Path):
	fig, ax = plt.subplots(figsize=(5, 4))
	cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)
	for (i, j), val in np.ndenumerate(cm):
		ax.text(j, i, str(val), va="center", ha="center", color="black")
	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])
	ax.set_xticklabels(col_labels)
	ax.set_yticklabels(row_labels)
	ax.set_xlabel("Ground Truth")
	ax.set_ylabel("Predicted")
	plt.title("Confusion Matrix")
	fig.colorbar(cax)
	plt.tight_layout()
	plt.savefig(out_file, dpi=150)
	plt.close()


def main(argv=None) -> int:
	args = parse_args(argv)
	out_dir = args.output or Path("analysis_output")
	out_dir.mkdir(parents=True, exist_ok=True)
	config = load_config(args.config)
	df = load_dataframe(args.csv)
	use_edn = not args.no_edn if args.edn or args.no_edn else True
	if use_edn:
		df = edn_to_ned(df)
	df["timestamp"] = regularize_timestamps(df["timestamp"].to_numpy(), args.fs)
	segments = map_segments_to_config(df, config)
	metrics: list[dict] = []
	for seg in segments:
		# Provide fallback axis from config name if available
		cfg = seg.get("config", {})
		if "axis" not in cfg:
			if "theta" in cfg or "theta_change" in cfg:
				cfg["axis"] = "pitch"
			elif "phi" in cfg or "phi_change" in cfg:
				cfg["axis"] = "yaw"
			seg["config"] = cfg
		m = process_segment(seg, args.fs)
		metrics.append(m)
	results = pd.DataFrame(metrics)
	results.to_csv(out_dir / "results.csv", index=False)
	by_angle, by_dps, heatmap = summarize_binned_tables(results, out_dir)
	save_heatmap_png(heatmap, out_dir / "error_heatmap.png")
	cm, cols, rows = build_confusion_matrix(results)
	pd.DataFrame(cm, index=rows, columns=cols).to_csv(out_dir / "confusion_matrix.csv")
	save_confusion_png(cm, cols, rows, out_dir / "confusion_matrix.png")
	print(f"Loaded {len(df)} samples, {len(segments)} segments from {args.csv}")
	print(f"Wrote outputs to {out_dir}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

