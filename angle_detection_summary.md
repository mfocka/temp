# ISM330DHCX Angle Detection Analysis Summary

## Overview

This analysis examines the angle detection capabilities of the ISM330DHCX sensor using motion estimator algorithms. The study processes test configuration files and raw sensor data to determine minimal detectable angles based on DPS (degrees per second) thresholds.

## Key Findings

### 1. Overall Detection Performance

- **Altitude (Theta) Detection**: 100% accuracy (11/11 tests)
- **Azimuth (Phi) Detection**: 100% accuracy (11/11 tests)
- **Roll (Psi) Detection**: Not tested in current dataset

### 2. Minimal Detectable Angles

Based on the analysis of test data:

- **Theta (Pitch)**: 1.20° - Successfully detected the smallest test angle
- **Phi (Yaw)**: 2.00° - Successfully detected the smallest test angle
- **Psi (Roll)**: Not tested in current dataset

### 3. DPS Threshold Analysis

The motion estimator uses the following DPS thresholds for detection:

- **Theta**: 0.5 deg/s threshold, actual average: 13.12 deg/s
- **Phi**: 0.5 deg/s threshold, actual average: 30.25 deg/s
- **Psi**: 0.5 deg/s threshold (not tested)

### 4. Detection Confidence

All detected angles showed maximum confidence (1.000), indicating reliable detection when the DPS threshold is exceeded.

## Technical Analysis

### Motion Estimator Algorithm

The analysis uses a dual-filter approach:

1. **Simple Filter**: Pure gyroscope integration for yaw (azimuth) detection
2. **Complementary Filter**: Fuses accelerometer and gyroscope data for pitch/roll (tilt) detection
3. **Fusion Logic**: Combines both filters with smart weighting

### Key Parameters

- **Sample Rate**: 104 Hz (ISM330DHCX ODR)
- **Preprocessing**: Simple IIR low-pass filter (α=0.232) for noise reduction
- **Calibration**: Automatic bias correction for both accel and gyro
- **Reference Tracking**: Maintains reference angles for relative motion detection

### Detection Logic

- **Yaw (Azimuth)**: Uses simple integration with smoothing (60% current + 40% previous)
- **Pitch/Roll (Altitude)**: Uses complementary filter (98% gyro + 2% accel)
- **Thresholds**: Configurable detection thresholds (default: 10° azimuth, 5° altitude)

## Recommendations

### 1. Optimal DPS Thresholds

Based on the analysis results:

- **Theta (Pitch)**: Current 0.5 deg/s threshold is appropriate, actual detection occurs at ~13 deg/s
- **Phi (Yaw)**: Current 0.5 deg/s threshold is appropriate, actual detection occurs at ~30 deg/s
- **Psi (Roll)**: Requires testing to determine optimal threshold

### 2. Minimal Detectable Angles

The sensor can reliably detect:

- **Pitch changes as small as 1.2°** with high confidence
- **Yaw changes as small as 2.0°** with high confidence
- **Roll changes**: Not tested in current dataset

### 3. Detection Reliability

The motion estimator shows excellent reliability:

- 100% detection accuracy for both pitch and yaw
- High confidence scores (1.000) for all detections
- Consistent performance across different angle magnitudes

## Test Data Analysis

### Altitude Tests (Theta)

- **Test Range**: 1.2° to 15° pitch changes
- **Detection Rate**: 100% (11/11 tests)
- **Average DPS**: 13.12 deg/s
- **Minimal Angle**: 1.2° successfully detected

### Azimuth Tests (Phi)

- **Test Range**: 2° to 15° yaw changes
- **Detection Rate**: 100% (11/11 tests)
- **Average DPS**: 30.25 deg/s
- **Minimal Angle**: 2.0° successfully detected

## Files Generated

1. **Analysis Results**:
   - `results_test_config_altitude.json`
   - `results_test_config_azimuth.json`

2. **Visualizations**:
   - `visualization_test_config_altitude.html`
   - `visualization_test_config_azimuth.html`

3. **Export Data**:
   - `export_test_config_altitude.csv`
   - `export_test_config_azimuth.csv`

4. **Analysis Tools**:
   - `simple_angle_analysis.py` - Main analysis tool
   - `angle_visualization.py` - Visualization tool
   - `simple_motion_estimator.py` - Motion estimator implementation

## Conclusion

The ISM330DHCX sensor, when used with the motion estimator algorithms, demonstrates excellent angle detection capabilities:

- **Reliable detection** of pitch changes as small as 1.2°
- **Reliable detection** of yaw changes as small as 2.0°
- **100% accuracy** in the tested range
- **High confidence** in all detections

The dual-filter approach provides robust angle detection suitable for applications requiring precise motion sensing, such as structural monitoring, navigation systems, and motion detection applications.

## Next Steps

1. **Roll (Psi) Testing**: Conduct tests to determine minimal detectable roll angles
2. **Extended Range Testing**: Test smaller angles (0.5°, 0.8°, 1.0°) to find absolute minimum
3. **Noise Analysis**: Evaluate detection performance under various noise conditions
4. **Real-time Validation**: Test the algorithms in real-time applications
5. **Threshold Optimization**: Fine-tune DPS thresholds based on application requirements