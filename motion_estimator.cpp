/*
 ********************************************************************************
 *  ADB Safegate Belgium BV
 *
 *  Motion Estimator - Combined Filter Implementation
 *
 ********************************************************************************
 */

#include "motion_estimator.h"
#include <cstring>
#include <cmath>

// Constants

#define M_PI		3.14159265358979323846	/* pi */
#define M_PI_2		1.57079632679489661923	/* pi/2 */
static constexpr float DEG_TO_RAD = M_PI / 180.0f;
static constexpr float RAD_TO_DEG = 180.0f / M_PI;
static constexpr float G_TO_MS2 = 9.80665f;
static constexpr float MG_TO_MS2 = G_TO_MS2 / 1000.0f;

// // Simple 2nd order Butterworth low-pass filter implementation
// class ButterworthFilter {
// private:
//     float a0, a1, a2, b0, b1, b2;
//     float x1, x2, y1, y2;
    
// public:
//     // Constructor to initialize filter coefficients
//     // fs: sampling frequency, cutoff: cutoff frequency
//     ButterworthFilter(float fs, float cutoff) : x1(0), x2(0), y1(0), y2(0) {
//         // Normalize cutoff frequency
//         float Wn = cutoff / fs;
        
//         // Calculate coefficients for 2nd order Butterworth filter
//         // Using bilinear transform
//         float K = tan(M_PI * Wn);
//         float K2 = K * K;
        
//         // Normalized Butterworth polynomial for 2nd order: s^2 + 1.4142s + 1
//         float Q = 0.7071f; // 1/sqrt(2) for Butterworth
//         float norm = 1.0f + K/Q + K2;
        
//         b0 = K2 / norm;
//         b1 = 2.0f * b0;
//         b2 = b0;
//         a1 = 2.0f * (K2 - 1.0f) / norm;
//         a2 = (1.0f - K/Q + K2) / norm;
//         a0 = 1.0f;
//     }
    
//     // Apply filter to a single sample
//     float filter(float input) {
//         float output = b0*input + b1*x1 + b2*x2 - a1*y1 - a2*y2;
        
//         // Update delay elements
//         x2 = x1;
//         x1 = input;
//         y2 = y1;
//         y1 = output;
        
//         return output;
//     }
    
//     // Reset filter state
//     void reset() {
//         x1 = x2 = y1 = y2 = 0.0f;
//     }
// };

// // Forward-backward filtering (zero-phase) implementation
// void filtfilt(ButterworthFilter& filter, float* data, int length) {
//     // Temporary array for filtered data
//     float* temp = new float[length];
    
//     // Forward filtering
//     filter.reset();
//     for(int i = 0; i < length; i++) {
//         temp[i] = filter.filter(data[i]);
//     }
    
//     // Backward filtering
//     filter.reset();
//     for(int i = length-1; i >= 0; i--) {
//         data[i] = filter.filter(temp[i]);
//     }
    
//     delete[] temp;
// }

// // Main preprocessing function
// void preprocess_imu(float* accel_x, float* accel_y, float* accel_z,
//                    float* gyro_x, float* gyro_y, float* gyro_z,
//                    int data_length, float fs=104.0f, float cutoff=5.0f) {
    
//     // Create Butterworth filter
//     ButterworthFilter filter(fs, cutoff);
    
//     // Apply filtering to each axis
//     filtfilt(filter, accel_x, data_length);
//     filtfilt(filter, accel_y, data_length);
//     filtfilt(filter, accel_z, data_length);
//     filtfilt(filter, gyro_x, data_length);
//     filtfilt(filter, gyro_y, data_length);
//     filtfilt(filter, gyro_z, data_length);
// }
MotionEstimator::MotionEstimator()
    : _dt(0.0f)
    , _prev_yaw_deg(0.0f)
    , _prev_pitch_deg(0.0f)
    , _prev_roll_deg(0.0f)
    , _simple_yaw_deg(0.0f)
    , _simple_pitch_deg(0.0f)
    , _simple_roll_deg(0.0f)
    , _comp_yaw_deg(0.0f)
    , _comp_pitch_deg(0.0f)
    , _comp_roll_deg(0.0f)
    , _ref_yaw_deg(0.0f)
    , _ref_pitch_deg(0.0f)
    , _ref_roll_deg(0.0f)
    , _has_reference(false)
{
    // Initialize calibration data
    std::memset(&_calibration, 0, sizeof(_calibration));
    _initPreprocessingFilter();
}

bool MotionEstimator::initialize(const Config& config)
{
    _config = config;
    _dt = 1.0f / _config.sample_rate_hz;
    
    // Reset all state variables
    _prev_yaw_deg = _prev_pitch_deg = _prev_roll_deg = 0.0f;
    _simple_yaw_deg = _simple_pitch_deg = _simple_roll_deg = 0.0f;
    _comp_yaw_deg = _comp_pitch_deg = _comp_roll_deg = 0.0f;
    _ref_yaw_deg = _ref_pitch_deg = _ref_roll_deg = 0.0f;
    _has_reference = false;
    
    // Reset calibration
    resetCalibration();
    
    return true;
}

bool MotionEstimator::addCalibrationSample(const float accel[3], const float gyro[3])
{
    if (_calibration.is_calibrated) {
        return true; // Already calibrated
    }
    
    // Add to running sums
    for (int i = 0; i < 3; i++) {
        _calibration.acc_bias[i] += accel[i];
        _calibration.gyro_bias[i] += gyro[i];
    }
    _calibration.sample_count++;
    
    // Check if we have enough samples
    if (_calibration.sample_count >= _config.calibration_samples) {
        // Calculate averages
        for (int i = 0; i < 3; i++) {
            _calibration.acc_bias[i] /= _calibration.sample_count;
            _calibration.gyro_bias[i] /= _calibration.sample_count;
        }
        
        // Check if gyro bias is reasonable (not too noisy)
        float gyro_magnitude = std::sqrt(
            _calibration.gyro_bias[0] * _calibration.gyro_bias[0] +
            _calibration.gyro_bias[1] * _calibration.gyro_bias[1] +
            _calibration.gyro_bias[2] * _calibration.gyro_bias[2]
        );
        
        if (gyro_magnitude < _config.gyro_noise_threshold_dps) {
            _calibration.is_calibrated = true;
            return true;
        } else {
            // Reset and try again
            resetCalibration();
            return false;
        }
    }
    
    return false;
}

void MotionEstimator::setDefaultCalibrationData() {
    float default_gyro_bias[3] = {-0.398, 0.587, 0.770};
    float default_acce_bias[3] = {-79.406,204.207,989.005};

    for (size_t i = 0; i < 3; i++)
    {
        _calibration.gyro_bias[i] = default_gyro_bias[i];
        _calibration.acc_bias[i] = default_acce_bias[i];
    }
    _calibration.is_calibrated = true;
    printCalibrationData();
    
}

MotionEstimator::Output MotionEstimator::update(const float accel[3], const float gyro[3], uint64_t timestamp_us)
{
    Output output;
    output.timestamp_us = timestamp_us;
    
    if (!_calibration.is_calibrated) {
        // Return zero angles if not calibrated
        output.yaw_deg = output.pitch_deg = output.roll_deg = 0.0f;
        output.has_large_change = false;
        return output;
    }
    
    // Apply bias correction
    float accel_corrected[3], gyro_corrected[3];
    for (int i = 0; i < 3; i++) {
        accel_corrected[i] = accel[i] - _calibration.acc_bias[i];
        gyro_corrected[i] = gyro[i] - _calibration.gyro_bias[i];
    }
    float accel_filtered[3] = {};
    float gyro_filtered[3] = {};
    _applyPreprocessingFilter(accel_corrected, accel_filtered, _accel_filtered_prev);
    _applyPreprocessingFilter(gyro_corrected, gyro_filtered, _gyro_filtered_prev);
    
    // Update both filters
    _updateSimpleFilter(gyro_filtered);
    _updateComplementaryFilter(accel_filtered, gyro_filtered);
    
    // Combine filters using (?) logic
    _combineFilters(output);
    
    // Set reference angles if not set yet
    if (!_has_reference) {
        _setReferenceAngles();
    }
    
    // Update previous angles
    _prev_yaw_deg = output.yaw_deg;
    _prev_pitch_deg = output.pitch_deg;
    _prev_roll_deg = output.roll_deg;
    
    return output;
}

void MotionEstimator::convertToRelativeAngles(Output& output) {
    if (!_has_reference) {
        _setReferenceAngles();
    }

    output.yaw_deg = fabsf(_ref_yaw_deg - output.yaw_deg);
    output.pitch_deg = fabsf(_ref_pitch_deg - output.pitch_deg);
    output.roll_deg = fabsf(_ref_roll_deg - output.roll_deg);
}

void MotionEstimator::resetCalibration()
{
    std::memset(&_calibration, 0, sizeof(_calibration));
    _calibration.is_calibrated = false;
    _calibration.sample_count = 0;
    resetFilterStates(true);
}
void MotionEstimator::_initPreprocessingFilter() {
    // Calculate alpha based on your desired cutoff frequency
    // alpha = dt / (RC + dt) where RC = 1/(2*pi*cutoff)
    // For fs=104Hz, dt = 1/104 ~~ 0.0096s
    // For cutoff=5Hz, RC = 1/(2*pi*5) ~~ 0.0318
    // alpha ~~ 0.0096 / (0.0318 + 0.0096) ~~ 0.232
    
    _alpha = 0.232f;
    
    // Initialize previous values
    for (int i = 0; i < 3; i++) {
        _accel_filtered_prev[i] = 0.0f;
        _gyro_filtered_prev[i] = 0.0f;
    }
}

void MotionEstimator::_applyPreprocessingFilter(const float input[3], float output[3], float prev[3]) {
    if(_alpha){
        for (int i = 0; i < 3; i++) {
            output[i] = _alpha * input[i] + (1.0f - _alpha) * prev[i];
            prev[i] = output[i];
        }
    }
}
void MotionEstimator::_updateSimpleFilter(const float gyro[3])
{
    // Simple integration filter for all axes
    // Yaw: integrate around Z-axis (gyro[2])
    _simple_yaw_deg = _simple_yaw_deg + gyro[2] * _dt;
    
    // Pitch: integrate around Y-axis (gyro[1])
    // TODO: swapped to gyro[0] due to semantics 
    _simple_pitch_deg = _simple_pitch_deg + gyro[0] * _dt;
    
    // Roll: integrate around X-axis (gyro[0])
    // TODO: swapped to gyro[1] due to semantics 
    _simple_roll_deg = _simple_roll_deg + gyro[1] * _dt;
    
    // Normalize angles to [0, 180] range
    Output simple_output = {};
    simple_output.yaw_deg   = _normalizeAngle(_simple_yaw_deg);
    simple_output.pitch_deg = _normalizeAngle(_simple_pitch_deg);
    simple_output.roll_deg  = _normalizeAngle(_simple_roll_deg);

    convertToRelativeAngles(simple_output);
    _simple_yaw_deg = simple_output.yaw_deg;
    _simple_pitch_deg = simple_output.pitch_deg;
    _simple_roll_deg = simple_output.roll_deg;
}

void MotionEstimator::_updateComplementaryFilter(const float accel[3], const float gyro[3])
{
    // Get tilt angles from accelerometer
    // TODO: swapped to axis 0 due to semantics 
    float acc_pitch_deg = _tiltAngleFromAccel(accel, 0); // Pitch from Y-axis
    // TODO: swapped to axis 1 due to semantics 
    float acc_roll_deg = _tiltAngleFromAccel(accel, 1);  // Roll from X-axis
    
    // Gyro integration

    // TODO: swapped to gyro[0] due to semantics 
    float gyro_pitch_deg = _comp_pitch_deg + gyro[0] * _dt;
    // TODO: swapped to gyro[1] due to semantics 
    float gyro_roll_deg = _comp_roll_deg + gyro[1] * _dt;
    float gyro_yaw_deg = _comp_yaw_deg + gyro[2] * _dt;
    
    // Complementary filter fusion
    _comp_pitch_deg = _config.alpha * gyro_pitch_deg + (1.0f - _config.alpha) * acc_pitch_deg;
    _comp_roll_deg = _config.alpha * gyro_roll_deg + (1.0f - _config.alpha) * acc_roll_deg;
    _comp_yaw_deg = gyro_yaw_deg; // Yaw only from gyro (no accelerometer reference)
    
    // Normalize angles to [0,180]
    Output complementary_output = {};
    complementary_output.yaw_deg   = _normalizeAngle(_comp_yaw_deg);
    complementary_output.pitch_deg = _normalizeAngle(_comp_pitch_deg);
    complementary_output.roll_deg  = _normalizeAngle(_comp_roll_deg);

    convertToRelativeAngles(complementary_output);
    _comp_yaw_deg = complementary_output.yaw_deg;
    _comp_pitch_deg = complementary_output.pitch_deg;
    _comp_roll_deg = complementary_output.roll_deg;
}

void MotionEstimator::_combineFilters(Output& output)
{
    //combination logic based on testing results:
    // - Azimuth (yaw): Simple filter works better but can be noisy
    // - Tilt (pitch/roll): Complementary filter has less drift
    
    // Yaw: Use simple filter as primary, but validate with complementary filter
    output.yaw_deg = _simple_yaw_deg;
    
    // Check if complementary filter shows significant change
    float yaw_diff = fabsf(_comp_yaw_deg - _prev_yaw_deg);
    if (yaw_diff < 10.0f) { // Small change threshold
        // If complementary filter shows small change, trust simple filter
        output.yaw_deg = _simple_yaw_deg;
    } else {
        // If complementary filter shows large change, use weighted average
        output.yaw_deg = 0.6f * _simple_yaw_deg + 0.4f * _comp_yaw_deg;
    }
    
    // Pitch and Roll: Use complementary filter as primary (less drift)
    output.pitch_deg = _comp_pitch_deg;
    output.roll_deg = _comp_roll_deg;
    
    // Normalize final angles to [0,180]
    output.yaw_deg = _normalizeAngle(output.yaw_deg);
    output.pitch_deg = _normalizeAngle(output.pitch_deg);
    output.roll_deg = _normalizeAngle(output.roll_deg);
}
void MotionEstimator::_setReferenceAngles()
{
    _ref_yaw_deg = _prev_yaw_deg;
    _ref_pitch_deg = _prev_pitch_deg;
    _ref_roll_deg = _prev_roll_deg;
    _has_reference = true;
}

float MotionEstimator::_tiltAngleFromAccel(const float accel[3], int axis)
{
    // Convert mg to m/s^2
    float ax = accel[0] * MG_TO_MS2;
    float ay = accel[1] * MG_TO_MS2;
    float az = accel[2] * MG_TO_MS2;
    
    float angle_rad;
    if (axis == 0) { // Roll (around X-axis)
        angle_rad = std::atan2(ay, az);
    } else if (axis == 1) { // Pitch (around Y-axis)
        angle_rad = std::atan2(-ax, std::sqrt(ay * ay + az * az));
    } else { // Invalid axis
        return 0.0f;
    }
    
    return angle_rad * RAD_TO_DEG;
}

float MotionEstimator::_normalizeAngle(float angle_deg)
{
    // Normalize angle to [0, 180] range
    while (angle_deg > 180.0f) {
        angle_deg -= 360.0f;
    }
    while (angle_deg < -180.0f) {
        angle_deg += 360.0f;
    }
    return fabsf(angle_deg);
}
void MotionEstimator::resetFilterStates(bool reset_reference) 
{
    // Reset all filter states to prevent accumulated drift
    _simple_yaw_deg = _simple_pitch_deg = _simple_roll_deg = 0.0f;
    _comp_yaw_deg = _comp_pitch_deg = _comp_roll_deg = 0.0f;
    _prev_yaw_deg = _prev_pitch_deg = _prev_roll_deg = 0.0f;
    
    // Reset preprocessing filter states
    _initPreprocessingFilter();
    
    if (reset_reference) {
        // Reset reference angles - this allows us to ignore slow drift
        _ref_yaw_deg = _ref_pitch_deg = _ref_roll_deg = 0.0f;
        _has_reference = false;
    }
    
    console->printOutput("MotionEstimator: Filter states reset (reference=%s)\n", 
                        reset_reference ? "reset" : "kept");
}
void MotionEstimator::setSynchronizedReference(const float euler_angles[3])
{
    _ref_yaw_deg = 0.7 * euler_angles[0] + 0.3 *_ref_yaw_deg;
    _ref_pitch_deg = 0.7 * euler_angles[1] + 0.3 *_ref_pitch_deg; 
    _ref_roll_deg = 0.7 * euler_angles[2]+ 0.3 *_ref_roll_deg;
    _has_reference = true;
    
    // Also set current filter outputs to match
    _prev_yaw_deg = euler_angles[0];
    _prev_pitch_deg = euler_angles[1];
    _prev_roll_deg = euler_angles[2];
}
void MotionEstimator::setGyroBiasFromExternal(const float gyro_bias[3]) {
    bool bias_changed = false;
    const float BIAS_CHANGE_THRESHOLD = 0.5f; // 0.5 deg/s threshold // TODO: set this up properly in header with tested values
    
    if (_calibration.is_calibrated) {
        for (int i = 0; i < 3; i++) {
            if (fabsf(gyro_bias[i] - _calibration.gyro_bias[i]) > BIAS_CHANGE_THRESHOLD) {
                bias_changed = true;
                break;
            }
        }
    }
    
    // Set gyro bias from external source (e.g., MotionDI)
    for (int i = 0; i < 3; i++) {
        _calibration.gyro_bias[i] = gyro_bias[i];
    }
    
    // Mark as calibrated if we have valid bias values
    if (fabsf(gyro_bias[0]) < 100.0f && fabsf(gyro_bias[1]) < 100.0f && fabsf(gyro_bias[2]) < 100.0f) {
        _calibration.is_calibrated = true;
    }
    
    // Reset filters if bias changed significantly (indicates recalibration)
    if (bias_changed) {
        console->printOutput("MotionEstimator: Significant bias change detected, resetting filters\n");
        resetFilterStates(true);  // Reset reference angles to ignore accumulated drift
    }
    _calibration.is_calibrated = true;
}

void MotionEstimator::updateCompleteEulerAngles(const float other_euler_angles[3], const float trust[3], float resulting_euler_angles[3]) {
    const float yaw = other_euler_angles[0];
    const float pitch = other_euler_angles[1];
    const float roll = other_euler_angles[2];
    
    // Initialize with input angles
    resulting_euler_angles[0] = yaw;
    resulting_euler_angles[1] = pitch; 
    resulting_euler_angles[2] = roll;
    if (!isReady()) {
        // If not ready, just pass through the input angles
        return;
    }
    if (fabsf(yaw - _prev_yaw_deg) < 5.0) {
        // Agreement: use trust-weighted fusion
        resulting_euler_angles[0] = (trust[0] * yaw) + ((1.0f - trust[0]) * _prev_yaw_deg);
    } else {
        // Disagreement: be conservative
        resulting_euler_angles[0] = fminf(_prev_yaw_deg, yaw);
    }
    
    // Pitch (Altitude) - High trust in MotionDI, use MotionEstimator for drift correction
    if (fabsf(pitch - _prev_pitch_deg) < 10.0f) {
        resulting_euler_angles[1] = (trust[1] * pitch) + ((1.0f - trust[1]) * _prev_pitch_deg);
    } else {
        // Cross-check: validate pitch changes
        float pitch_diff = fabsf(pitch - _prev_pitch_deg);
        float estimator_diff = fabsf(_comp_pitch_deg - _prev_pitch_deg);
        
        if (fabsf(pitch_diff - estimator_diff) < 15.0f) {
            // Changes are similar, use trust-based fusion
            resulting_euler_angles[1] = (trust[1] * pitch) + ((1.0f - trust[1]) * _prev_pitch_deg);
        } else {
            // Large disagreement, be conservative
            resulting_euler_angles[1] = fminf(_prev_pitch_deg, pitch);
        }
    }
    
    // Roll - Normal trust in MotionDI, use MotionEstimator for validation
    if (fabsf(roll - _prev_roll_deg) < 5.0f) {
        resulting_euler_angles[2] = (trust[2] * roll) + ((1.0f - trust[2]) * _prev_roll_deg);
    } else {
        // Cross-check: validate roll changes
        float roll_diff = fabsf(roll - _prev_roll_deg);
        float estimator_diff = fabsf(_comp_roll_deg - _prev_roll_deg);
        
        if (fabsf(roll_diff - estimator_diff) < 15.0f) {
            // Changes are similar, use trust-based fusion
            resulting_euler_angles[2] = (trust[2] * roll) + ((1.0f - trust[2]) * _prev_roll_deg);
        } else {
            // Large disagreement, be conservative
            resulting_euler_angles[2] = fminf(_prev_roll_deg, roll);
        }
    }
}

void MotionEstimator::printCalibrationData() {
    console->printOutput("Final gyro bias: [%f, %f, %f] dps\n", 
                    _calibration.gyro_bias[0], _calibration.gyro_bias[1], _calibration.gyro_bias[2]);

    console->printOutput("Final acc bias: [%f, %f, %f] mg\n", 
                    _calibration.acc_bias[0], _calibration.acc_bias[1], _calibration.acc_bias[2]);
}



void MotionEstimator::getDebugInfo(DebugInfo& debug_info) const {
    // Fill debug info structure with current state
    debug_info.simple_filter[0] = _simple_yaw_deg;
    debug_info.simple_filter[1] = _simple_pitch_deg;
    debug_info.simple_filter[2] = _simple_roll_deg;
    
    debug_info.complementary_filter[0] = _comp_yaw_deg;
    debug_info.complementary_filter[1] = _comp_pitch_deg;
    debug_info.complementary_filter[2] = _comp_roll_deg;
    
    debug_info.fused_output[0] = _prev_yaw_deg;
    debug_info.fused_output[1] = _prev_pitch_deg;
    debug_info.fused_output[2] = _prev_roll_deg;
    
    debug_info.previous_angles[0] = _prev_yaw_deg;
    debug_info.previous_angles[1] = _prev_pitch_deg;
    debug_info.previous_angles[2] = _prev_roll_deg;
    
    debug_info.reference_angles[0] = _ref_yaw_deg;
    debug_info.reference_angles[1] = _ref_pitch_deg;
    debug_info.reference_angles[2] = _ref_roll_deg;
    
    debug_info.is_calibrated = _calibration.is_calibrated;
    debug_info.calibration_samples = _calibration.sample_count;
}