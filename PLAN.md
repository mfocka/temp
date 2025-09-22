# Angle Change Reliability - Plan of Attack

Scope: Build a simple, reproducible pipeline to quantify estimation errors vs. angle magnitude and vs. angular speed (DPS) from raw logs, then export CSV summaries and plots (heatmap + confusion matrix) to inform reliability ranges.

Steps

1) CLI script skeleton
   - Input: --csv, --config (optional), --output, --edn/--no-edn, --fs
   - Output: analysis CSVs and PNGs under output dir

2) Data loading and preprocessing
   - Validate required columns
   - Optional EDN->NED conversion (default on)
   - Regularize timestamps to uniform fs
   - Segment by timestamp gaps and map to config if provided

3) Per-segment processing
   - Run MotionEstimator sequentially
   - Compute reference-relative angles (stabilize with first N samples)
   - Detect motion window (start/stop) using gyro magnitude threshold
   - Measure: measured final angle, expected angle (from config if available), error_deg, error_percent
   - Compute max_dps and mean_dps during motion

4) Aggregate metrics
   - Concatenate segment metrics → results.csv
   - Bin by angle magnitude and by DPS ranges → tables (CSV)
   - Build 2D heatmap table (angle_bin × dps_bin) of RMSE/MAE

5) Visualizations
   - Error heatmap (angle vs dps)
   - Confusion matrix: expected change (>0) vs detected change (>|threshold|)

6) Docs
   - Script docstring with usage examples
   - Short README snippet in PLAN.md footer

Outputs

- results.csv: per-segment metrics
- error_by_angle.csv, error_by_dps.csv
- error_heatmap.csv, error_heatmap.png
- confusion_matrix.csv, confusion_matrix.png

Quick usage

```bash
python3 angle_change_reliability.py \
  --csv raw_data_output_altitude.csv \
  --config test_config_altitude.json \
  --output analysis_output \
  --fs 104 --edn
```

Reliability guidance

- Derive suggested operating ranges from heatmap quantiles (e.g., RMSE <= 1°) across bins.

