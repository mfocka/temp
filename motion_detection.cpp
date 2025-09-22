/*
 *******************************************************************************
 *  ADB Safegate Belgium BV
 *
 *
 *  Motion Detection System using MotionDI + MotionFX Libraries
 *
 *******************************************************************************
 */

#include "motion_detection.h"
#include "motion_estimator.h"
#include "console.h"
#include "globals.h"
#include "PLCMsg.h"


extern Console *console;

bool MotionDetection::PRINT_RAW = false;
bool MotionDetection::PRINT_INFO = false;
bool MotionDetection::PRINT_ANGLES = false;
bool MotionDetection::PRINT_QUAT = false;
bool MotionDetection::PRINT_EVENTS = true;
bool MotionDetection::PRINT_BIAS = false;
bool MotionDetection::PRINT_ESTIMATOR = false;



MotionDetection::MotionDetection()
    : _reference_quat(Quaternion::identity())
    , _current_quat(Quaternion::identity())
    , _has_reference(false)
    , _calibrated(false)
    , _calibration_sample_count(0)
    , _startup_settling_count(0)
    , _altitude_threshold(15.0f)
    , _azimuth_threshold(30.0f)
    , _validation_time_minutes(1)  // 4 hours default
    , _persistent_validation_time_us(0)
    , _was_validating(false)
    , _validation_start_timestamp_us(0)
    , _pending_event_type(MotionEventType::NO_EVENT)
    , _altitude_event_active(false)
    , _azimuth_event_active(false)
    , _combined_event_active(false)
    , _bad_sample_count(0)
    , _state(MotionDetectionState::STARTUP)
    , _altitude_angle(0, ALTITUDE_DEV, "Altitude deviation from set point angle")
    , _azimuth_angle(0, AZIMUTH_DEV, "Azimuth deviation from set point angle")
    // , _high_gyro_noise_detected(false)
    // , _gyro_noise_sample_count(0)
    // , _gyro_history_index(0)
    , _timing()
    , _validation()
    , _calibration_context()
    , _operation_mode(MotionMode::THRESHOLD_ONLY)
    , _prev_altitude(0.0f)
    , _prev_azimuth(0.0f)
    , _motion_estimator(nullptr)
{
    list_of_sensor_ids.push_back(&_altitude_angle);
    list_of_sensor_ids.push_back(&_azimuth_angle);

    memset(&_mdi_knobs, 0, sizeof(_mdi_knobs));
    memset(&_mdi_output, 0, sizeof(_mdi_output));


}

MotionDetection::~MotionDetection()
{
    if (_state != MotionDetectionState::STARTUP) {
    }
    
    // Clean up MotionEstimator
    if (_motion_estimator) {
        delete _motion_estimator;
        _motion_estimator = nullptr;
    }
}

bool MotionDetection::initialize()
{
    console->printOutput("MotionDetection: Initializing system...\n");

    if (!_initializeMotionDI()) {
        _state = MotionDetectionState::ERROR;
        return false;
    }
    if (!_initializeMotionEstimator()){
        _state = MotionDetectionState::ERROR;
        return false;
    }

    // Try to load calibration data from EEPROM
    if (loadCalibrationData()) {
        console->printOutput("MotionDetection: Loaded calibration from EEPROM\n");

        if (_state == MotionDetectionState::VALIDATING) {
            console->printOutput("MotionDetection: Resuming validation state after reboot\n");
        } else {
            _state = MotionDetectionState::STARTUP;
        }

        _calibrated = true;
        _has_reference = true;
    } else {
        console->printOutput("MotionDetection: No valid calibration data, waiting for calibration command\n");
        _state = MotionDetectionState::STARTUP;
    }

    return true;
}

bool MotionDetection::_initializeMotionEstimator() {

    console->printOutput("MotionDetection: Initializing MotionEstimator for attitude estimation...\n");

    // Create and initialize our custom MotionEstimator
    _motion_estimator = new MotionEstimator();
    
    MotionEstimator::Config config;
    config.sample_rate_hz = SAMPLE_RATE_HZ;
    config.alpha = 0.88f;
    config.azimuth_threshold_deg = _azimuth_threshold;
    config.altitude_threshold_deg = _altitude_threshold;
    config.calibration_samples = MINIMUM_CALIBRATION_SAMPLES;
    config.gyro_noise_threshold_dps = 1.8f;
    
    if (!_motion_estimator->initialize(config)) {
        console->printOutput("ERROR: Failed to initialize MotionEstimator\n");
        return false;
    }

    console->printOutput("MotionEstimator initialized with parameters:\n");
    console->printOutput("  Sample Rate: %f hz, alpha: %f, calibration samples: %u\n",
                        config.sample_rate_hz, config.alpha, static_cast<unsigned>(config.calibration_samples));
    
    // Always print initialization success
    console->printOutput("MotionEstimator: Successfully initialized and ready for calibration\n");

    return true;
}

bool MotionDetection::_initializeMotionDI()
{
    console->printOutput("MotionDetection: Initializing MotionDI for static inclinometer...\n");
    
    // Initialize MotionDI library
    float freq = SAMPLE_RATE_HZ;
    MotionDI_Initialize(&freq);
    
    // Get default knobs
    MotionDI_getKnobs(&_mdi_knobs);
    
    // Accelerometer Configuration - More conservative for static applications
    _mdi_knobs.AccKnob.MoveThresh_g = 0.15f;      // Reduced from 0.25f - more sensitive to small movements
    _mdi_knobs.AccKnob.CalType = MDI_CAL_CONTINUOUS; // Changed from ONETIME - continuous bias correction

    // Gyroscope Configuration - Tighter thresholds for better bias estimation
    _mdi_knobs.GyrKnob.AccThr = 0.005f;           // Reduced from 0.01f - stricter stillness detection
    _mdi_knobs.GyrKnob.GyroThr = 0.4f;
    _mdi_knobs.GyrKnob.MaxGyro = 15.0f;
    _mdi_knobs.GyrKnob.CalType = MDI_CAL_CONTINUOUS; // Keep continuous for ongoing bias correction

    // Sensor Fusion - More aggressive filtering for static applications
    _mdi_knobs.SFKnob.ATime = 1.0f;               // Increased from 0.5f - longer averaging time
    _mdi_knobs.SFKnob.FrTime = 4.0f;              // Increased from 2.0f - longer fusion recovery time
    _mdi_knobs.SFKnob.modx = 1;
    _mdi_knobs.SFKnob.output_type = MDI_ENGINE_OUTPUT_ENU;
        
    // strcpy(_mdi_knobs.AccOrientation, "enu");
    // strcpy(_mdi_knobs.GyroOrientation, "enu");

    strcpy(_mdi_knobs.AccOrientation, "enu");
    strcpy(_mdi_knobs.GyroOrientation, "enu");
    
    MotionDI_setKnobs(&_mdi_knobs);
    
    MotionDI_AccCal_reset();
    MotionDI_GyrCal_reset();
    
    console->printOutput("MotionDI initialized with parameters:\n");
    console->printOutput("  AccThr: %f g, GyroThr: %f dps, MaxGyro: %f dps\n",
                        _mdi_knobs.GyrKnob.AccThr, _mdi_knobs.GyrKnob.GyroThr, _mdi_knobs.GyrKnob.MaxGyro);
    console->printOutput("  ATime: %f, FrTime: %f, MoveThresh: %f g\n",
                        _mdi_knobs.SFKnob.ATime, _mdi_knobs.SFKnob.FrTime, _mdi_knobs.AccKnob.MoveThresh_g);
    
    return true;
}

bool MotionDetection::processData(MotionSensorData &data, uint64_t sample_timestamp_us, uint8_t sample_index)
{
    // if (!_validateSensorData(data)) {
    //     _bad_sample_count++;

    //     if (_bad_sample_count > BAD_SAMPLE_THRESHOLD) {
    //         console->printOutput("ERROR: Too many bad samples, entering error state\n");
    //         _state = MotionDetectionState::ERROR;
    //     }

    //     return false;
    // }

    _bad_sample_count = 0;
    _timing.incrementSampleCount();

    data.timestamp_us = sample_timestamp_us;
    
    _updateMotionDI(data);
    _updateMotionEstimator(data);
    
    _processStateMachine();

    if (sample_index == 0) {
        if (PRINT_RAW)       _printRawData(data);
        if (PRINT_ANGLES)    _printAngleData(sample_timestamp_us);
        if (PRINT_QUAT)      _printQuatData(sample_timestamp_us);
        if (PRINT_BIAS)      _printBiasData(sample_timestamp_us);
        if (PRINT_ESTIMATOR) _printEstimatorAngles(sample_timestamp_us);
    }

    return true;
}

void MotionDetection::_convertWDStoENU(const float wds[3], float enu[3])
{
    // Transformation matrix WDS -> ENU
    // WDS: X-West, Y-Down, Z-South
    // ENU: X-East, Y-North, Z-Up
    enu[0] = -wds[0];  // East = -West
    enu[1] = -wds[2];  // North = -South  
    enu[2] = -wds[1];  // Up = -Down
}

void MotionDetection::_updateMotionDI(const MotionSensorData &data)
{
    MDI_input_t mdi_input{};
    
    mdi_input.Timestamp = data.timestamp_us;
    
    // Convert WDS -> ENU and mg -> g
    float accel_enu[3];
    float gyro_enu[3];
    _convertWDStoENU(data.accel_mg, accel_enu);
    _convertWDStoENU(data.gyro_dps, gyro_enu);
    
    mdi_input.Acc[0] = accel_enu[0] / 1000.0f;
    mdi_input.Acc[1] = accel_enu[1] / 1000.0f;
    mdi_input.Acc[2] = accel_enu[2] / 1000.0f;
    
    mdi_input.Gyro[0] = gyro_enu[0];
    mdi_input.Gyro[1] = gyro_enu[1];
    mdi_input.Gyro[2] = gyro_enu[2];
    
    MotionDI_update(&_mdi_output, &mdi_input);

    float qx = _mdi_output.quaternion[0];
    float qy = _mdi_output.quaternion[1];
    float qz = _mdi_output.quaternion[2];
    float qw = _mdi_output.quaternion[3];
    _current_quat = Quaternion(qw, qx, qy, qz);
    
    // Safety check for NaNs in MotionDI output
    if (isnan(_mdi_output.quaternion[0]) || isnan(_mdi_output.quaternion[1]) ||
        isnan(_mdi_output.quaternion[2]) || isnan(_mdi_output.quaternion[3])) {
        console->printOutput("ERROR: NaN in MotionDI quaternion output!\n");
    }
}
void MotionDetection::_updateMotionEstimator(const MotionSensorData &data) {

    if (!_motion_estimator || !_motion_estimator->isReady()) {
        // Add calibration sample if not ready
        if (_motion_estimator) {
            float accel_enu[3], gyro_enu[3];
            _convertWDStoENU(data.accel_mg, accel_enu);
            _convertWDStoENU(data.gyro_dps, gyro_enu);
            bool was_calibrated = _motion_estimator->isReady();
            bool is_calibrated = _motion_estimator->addCalibrationSample(accel_enu, gyro_enu);
            
            // Print calibration completion message
            if (!was_calibrated && is_calibrated) {
                console->printOutput("MotionEstimator: Calibration completed successfully\n");
            }
        }
        return;
    }
    
    // Update motion estimation
    float accel_enu[3], gyro_enu[3];
    _convertWDStoENU(data.accel_mg, accel_enu);
    _convertWDStoENU(data.gyro_dps, gyro_enu);
    
    MotionEstimator::Output output = _motion_estimator->update(
        accel_enu, gyro_enu, data.timestamp_us);
    
    
}
void MotionDetection::_processStateMachine()
{
    switch (_state) {
        case MotionDetectionState::STARTUP:
            // Add settling period after power cycle to avoid initial instability
            _startup_settling_count++;
            if (_startup_settling_count >= 520) {  // ~5 seconds at 104Hz
                if (_calibrated && _has_reference) {
                    _state = MotionDetectionState::MONITORING;
                    console->printOutput("MotionDetection: Settling period complete, transitioning to MONITORING\n");
                } else {
                    _state = MotionDetectionState::IDLE;
                    console->printOutput("MotionDetection: Settling period complete, transitioning to IDLE (no calibration)\n");
                }
            }
            break;

        case MotionDetectionState::IDLE:
            // Waiting for calibration command
            break;

        case MotionDetectionState::CALIBRATING:
        {
            _calibration_sample_count++;
            
            // Check MotionDI calibration status (primary)
            MDI_cal_output_t mdi_gyro_cal;
            MotionDI_GyrCal_getParams(&mdi_gyro_cal);

            bool hasEnoughSamples = _calibration_sample_count >= MINIMUM_CALIBRATION_SAMPLES;
            bool hasTooManySamples = _calibration_sample_count >= CALIBRATION_SAMPLES;
            bool isCalibrated = mdi_gyro_cal.CalQuality >= MDI_CAL_OK;
            bool motionReady = _motion_estimator && _motion_estimator->isReady();
            bool calibrationCanStop = (hasEnoughSamples && isCalibrated &&motionReady) || hasTooManySamples;

            if (hasEnoughSamples && (isCalibrated || motionReady)) {
                // Check if MotionDI has converged
                if (isCalibrated && !_mdi_is_calibrated){
                    console->printOutput("MotionDI calibration complete at sample %u\n", _calibration_sample_count);
                    console->printOutput("Final gyro bias: [%f, %f, %f] dps\n", 
                                    mdi_gyro_cal.Bias[0], mdi_gyro_cal.Bias[1], mdi_gyro_cal.Bias[2]);
                    _mdi_is_calibrated = true;
                }
                
                if (motionReady && !_me_is_calibrated) {
                    console->printOutput("MotionEstimator calibration complete at sample %u\n", _calibration_sample_count);
                    _motion_estimator->printCalibrationData();
                    _me_is_calibrated = true;
                }
            }
                
            if (hasTooManySamples)
            {
                if (!isCalibrated){
                    mdi_gyro_cal.Bias[0] = -0.411;
                    mdi_gyro_cal.Bias[1] = 0.587;
                    mdi_gyro_cal.Bias[2] = 0.774;

                    console->printOutput("MotionDI calibration did not complete at sample %u\n", _calibration_sample_count);
                    console->printOutput("Default gyro bias: [%f, %f, %f] dps\n", 
                        mdi_gyro_cal.Bias[0],
                        mdi_gyro_cal.Bias[1],
                        mdi_gyro_cal.Bias[2]
                    );
                    _mdi_is_calibrated = true;
                }

                if (!motionReady){
                    console->printOutput("MotionEstimator calibration did not complete at sample %u\n", _calibration_sample_count);
                    _motion_estimator->setDefaultCalibrationData();
                    _motion_estimator->printCalibrationData();
                    _me_is_calibrated = true;
                }
            }
                
            
                // Share gyro bias with MotionEstimator
                // if (_motion_estimator) {
                //     _motion_estimator->setGyroBiasFromExternal(mdi_gyro_cal.Bias);
                //     console->printOutput("MotionEstimator: Gyro bias shared from MotionDI and filters reset\n");
                //     // Force filter reset after calibration to start clean
                //     _motion_estimator->resetFilterStates(true);
                // }
            if (calibrationCanStop) {
                if (_storeReferenceQuaternion()) {
                    _startup_settling_count = 0;  // Reset settling counter
                    _state = MotionDetectionState::STARTUP;  // Go through settling first
                    _calibrated = true;
                    _has_reference = true;
                    saveCalibrationData();
                    console->printOutput("MotionDetection: Calibration complete, entering settling period\n");
                } else {
                    _state = MotionDetectionState::ERROR;
                }
            }
            
        }
        break;

        case MotionDetectionState::MONITORING:
        {
            if (!_has_reference) {
                _state = MotionDetectionState::ERROR;
                break;
            }


            _updateAngles();

            
            
            if (_checkMotionThresholdsWithMode()) {
                _state = MotionDetectionState::VALIDATING;
                _validation.reset();
                _validation.setRequiredTime(_validation_time_minutes);
                
                // Print MotionEstimator event detection
                console->printOutput("MotionEstimator: Motion event detected, entering validation state\n");
                
                if (PRINT_EVENTS) {
                    _printEventData(_timing.getCurrentTimeStampUs(), "VALIDATION_START");
                }
            }
        }
        break;

        case MotionDetectionState::VALIDATING:
        {
            _updateAngles();
            
            _validation.update(SAMPLE_PERIOD_US);
            
            _updateValidationState();
        }
        break;

        case MotionDetectionState::ERROR:
            _last_result.is_calibrated = false;
            break;
    }
}

void MotionDetection::_updateAngles(){

            _calculateAnglesToReference(_current_quat, _reference_quat,
                _last_result.azimuth_angle_degrees,
                _last_result.altitude_angle_degrees,
                _last_result.zenith_angle_degrees);

            // TODO: Define the reset behaviour properly
            // if(_timing.total_samples_processed % 62400  == 0){
            //     _motion_estimator->resetFilterStates(true);
            // }
            _checkAndApplyPeriodicReset();
            
            // TODO: Improve set-up of all this for updateCompleteEulerAngles
            float mdi_yaw = 0.0f, mdi_roll = 0.0f, mdi_pitch = 0.0f;
            horizontalToEuler(_last_result.azimuth_angle_degrees, _last_result.altitude_angle_degrees, _last_result.zenith_angle_degrees, mdi_yaw, mdi_pitch, mdi_roll);
            float mdi_euler_angles[3] = {mdi_yaw, mdi_pitch, mdi_roll};
            // Apply MotionEstimator fusion and cross-checking
            float trust_vector[3] = {0.3f, 0.9f, 0.5f}; // Low trust in yaw, high trust in pitch, medium trust in roll
            float fused_euler_angles[3] = {0.0f, 0.0f, 0.0f};
            
            // Apply MotionEstimator fusion and cross-checking
            _motion_estimator->updateCompleteEulerAngles(mdi_euler_angles, trust_vector, fused_euler_angles);

            eulerToHorizontal(fused_euler_angles[0], fused_euler_angles[1], fused_euler_angles[2], _last_result.azimuth_angle_degrees, _last_result.altitude_angle_degrees, _last_result.zenith_angle_degrees);

            data16 altitude, azimuth;
            altitude.asUINT16 = static_cast<uint16_t>(fabsf(_last_result.altitude_angle_degrees) * 10.0f);
            azimuth.asUINT16 = static_cast<uint16_t>(fabsf(_last_result.azimuth_angle_degrees) * 10.0f);
            _altitude_angle.updateData(altitude);
            _azimuth_angle.updateData(azimuth);
            
}

void MotionDetection::horizontalToEuler(float azimuth, float altitude, float zenith, float &yaw, float &pitch, float &roll){
    // Convert horizontal coordinates to Euler angles
    yaw = azimuth;
    pitch = altitude;
    roll = zenith;
}

void MotionDetection::eulerToHorizontal(float yaw, float pitch, float roll, float &azimuth, float &altitude, float &zenith){
    // Convert Euler angles to horizontal coordinates
    azimuth = yaw;
    altitude = pitch;
    zenith = roll;
}

// TODO: do we go back to monitoring after we clear an event/threshold exceeded? If we exceed threshold but then go back, what happens? We should go back to monitoring, but we see that we stay in validation?
void MotionDetection::_updateValidationState()
{
    bool altitude_exceeded = fabsf(_last_result.altitude_angle_degrees) > _altitude_threshold;
    bool azimuth_exceeded = fabsf(_last_result.azimuth_angle_degrees) > _azimuth_threshold;

    if (!altitude_exceeded && !azimuth_exceeded) {
        _state = MotionDetectionState::MONITORING;
        _validation.reset();
        _pending_event_type = MotionEventType::NO_EVENT;

        if (PRINT_EVENTS) {
            _printEventData(_timing.getCurrentTimeStampUs(), "VALIDATION_CANCELLED");
        }
        return;
    }

   if (_validation.isComplete()) {
        if (altitude_exceeded && azimuth_exceeded) {
            _last_result.event_type = MotionEventType::COMBINED_EVENT;
            _combined_event_active = true;
            hw->setAlarm(SENSOR_ALARM_ALTITUDE);
            hw->setAlarm(SENSOR_ALARM_AZIMUTH);
        }
        else if (altitude_exceeded)
        {
            _last_result.event_type = MotionEventType::ALTITUDE_EVENT;
            _altitude_event_active = true;
            hw->setAlarm(SENSOR_ALARM_ALTITUDE);
        }
        else
        {
            _last_result.event_type = MotionEventType::AZIMUTH_EVENT;
            _azimuth_event_active = true;
            hw->setAlarm(SENSOR_ALARM_AZIMUTH);
        }

        _last_result.is_event_active = true;

        if (PRINT_EVENTS) {
            const char* event_name = (_last_result.event_type == MotionEventType::COMBINED_EVENT) ? "COMBINED_EVENT" :
                                    (_last_result.event_type == MotionEventType::ALTITUDE_EVENT) ? "ALTITUDE_EVENT" :
                                    "AZIMUTH_EVENT";
            _printEventData(_timing.getCurrentTimeStampUs(), event_name);
        }

        _state = MotionDetectionState::MONITORING;
        _validation.reset();
        _was_validating = false;
    }
}


bool MotionDetection::calibrate()
{
    if (_state == MotionDetectionState::CALIBRATING) {
        console->printOutput("WARNING: Already calibrating\n");
        return false;
    }

    if (_state == MotionDetectionState::ERROR) {
        console->printOutput("ERROR: Cannot calibrate - system in error state\n");
        return false;
    }
    
    console->printOutput("MotionDetection: Starting calibration for static inclinometer...\n");
    console->printOutput("Keep device PERFECTLY STILL for %d seconds\n", CALIBRATION_SAMPLES / (int)SAMPLE_RATE_HZ);
    
    // Reset all state
    _calibration_context.reset();
    _validation.reset();
    _was_validating = false;
    _persistent_validation_time_us = 0;
    // Reset internal calibration flags
    _mdi_is_calibrated = false;
    _me_is_calibrated = false;
    
    // Clear any existing events
    _altitude_event_active = false;
    _azimuth_event_active = false;
    _combined_event_active = false;
    hw->clearAlarm(SENSOR_ALARM_ALTITUDE);
    hw->clearAlarm(SENSOR_ALARM_AZIMUTH);
    _last_result.event_type = MotionEventType::NO_EVENT;
    _last_result.is_event_active = false;
    
    // Reset MotionDI calibration algorithms
    MotionDI_AccCal_reset();
    MotionDI_GyrCal_reset();

    _motion_estimator->resetCalibration();
    
    _state = MotionDetectionState::CALIBRATING;
    _calibration_sample_count = 0;
    _calibrated = false;
    _has_reference = false;

    return true;
}

bool MotionDetection::_storeReferenceQuaternion()
{
    if (_calibration_sample_count == 0)
    {
        return false;
    }

    _reference_quat = _current_quat;
    _has_reference = true;

    _printReferenceQuaternion(_timing.getCurrentTimeStampUs());
    _printCalibrationComplete(_timing.getCurrentTimeStampUs());
    return true;
}

bool MotionDetection::saveCalibrationData()
{

    // TODO - add gyro bias from motion_estimator here as well as data.appEstGyroBiasX.asFloat = ....
    MDI_cal_output_t mdi_gyro_cal;
    float gyro_bias[3];
    
    ConfigFlash.appConfig.FlashMemory.data.appAltReferenceQuaternionW.asFloat = _reference_quat.w;
    ConfigFlash.appConfig.FlashMemory.data.appAltReferenceQuaternionX.asFloat = _reference_quat.x;
    ConfigFlash.appConfig.FlashMemory.data.appAltReferenceQuaternionY.asFloat = _reference_quat.y;
    ConfigFlash.appConfig.FlashMemory.data.appAltReferenceQuaternionZ.asFloat = _reference_quat.z;

    // Get gyro bias from MotionDI
    MotionDI_GyrCal_getParams(&mdi_gyro_cal);
    gyro_bias[0] = mdi_gyro_cal.Bias[0];
    gyro_bias[1] = mdi_gyro_cal.Bias[1];
    gyro_bias[2] = mdi_gyro_cal.Bias[2];
    ConfigFlash.appConfig.FlashMemory.data.appAltGyroBiasX.asFloat = gyro_bias[0];
    ConfigFlash.appConfig.FlashMemory.data.appAltGyroBiasY.asFloat = gyro_bias[1];
    ConfigFlash.appConfig.FlashMemory.data.appAltGyroBiasZ.asFloat = gyro_bias[2];


    ConfigFlash.setWriteRequest(0, EepromConfigMgr::EEPROM_WRITE_APP_CONFIG);
    osEventFlagsSet(debug_module_signals, DebugModuleThread::SIG_FLASH_SAVE_REQUEST);
    console->debugOutput("Calibration data saved to EEPROM\n");

    return true;
}

bool MotionDetection::hasValidCalibrationData()
{
    return _reference_quat != Quaternion::identity();
}

bool MotionDetection::_validateEEPROMData(float gyro_bias[])
{
    float w, x, y, z;

    w = ConfigFlash.appConfig.FlashMemory.data.appAltReferenceQuaternionW.asFloat;
    x = ConfigFlash.appConfig.FlashMemory.data.appAltReferenceQuaternionX.asFloat;
    y = ConfigFlash.appConfig.FlashMemory.data.appAltReferenceQuaternionY.asFloat;
    z = ConfigFlash.appConfig.FlashMemory.data.appAltReferenceQuaternionZ.asFloat;

    gyro_bias[0] = ConfigFlash.appConfig.FlashMemory.data.appAltGyroBiasX.asFloat;
    gyro_bias[1] = ConfigFlash.appConfig.FlashMemory.data.appAltGyroBiasY.asFloat;
    gyro_bias[2] = ConfigFlash.appConfig.FlashMemory.data.appAltGyroBiasZ.asFloat;

    // Validate data
    if ((Quaternion(w, x, y, z) == Quaternion::identity()) ||
          (gyro_bias[0] == 0.0f && gyro_bias[1] == 0.0f && gyro_bias[2] == 0.0f)) {
        console->printOutput("WARNING: Calibration data corrupted in EEPROM\n");
        return false;
    }

    //Load data
    _reference_quat = Quaternion(w, x, y, z);

    return true;
}

void MotionDetection::_resetStateVariables()
{
    _altitude_event_active = false;
    _azimuth_event_active = false;
    _combined_event_active = false;
    _last_result.is_event_active = false;
    _last_result.altitude_angle_degrees = 0;
    _last_result.azimuth_angle_degrees = 0;
    _last_result.event_type = MotionEventType::NO_EVENT;
    _last_result.is_calibrated = false;
    _last_result.zenith_angle_degrees = 0;
    _pending_event_type = MotionEventType::NO_EVENT;

    _validation_start_timestamp_us = 0;

    _prev_altitude = 0.0f;
    _prev_azimuth = 0.0f;

    _was_validating = false;
    _state = MotionDetectionState::IDLE;

    _current_quat = Quaternion::identity();
}

bool MotionDetection::_initializeMotionDIWithBias(float gyro_bias[])
{
    MDI_cal_output_t mdi_gyro_cal;
    
    // We will use the CALIBRATION_SAMPLES for this with our timing system
    _timing.total_samples_processed = 0;

    // Set gyro bias in MotionDI
    mdi_gyro_cal.Bias[0] = gyro_bias[0];
    mdi_gyro_cal.Bias[1] = gyro_bias[1];
    mdi_gyro_cal.Bias[2] = gyro_bias[2];
    MotionDI_GyrCal_setParams(&mdi_gyro_cal);
    
    return true;
}

bool MotionDetection::loadCalibrationData()
{
    float gyro_bias[3];

    _resetStateVariables();
    if(!_validateEEPROMData(gyro_bias)) {
        console->printOutput("MotionDetection: Failed to validate EEPROM data\n");
        return false;
    }
    if(!_initializeMotionDIWithBias(gyro_bias)) {
        console->printOutput("MotionDetection: Failed to initialize MotionDI with bias\n");
        return false;
    }
    _motion_estimator->setGyroBiasFromExternal(gyro_bias);
    if(!_motion_estimator->isCalibrated()){
        console->printOutput("MotionEstimator: Failed to initialize MotionE with bias\n");
        return false;
    }

    _altitude_threshold = ConfigFlash.appConfig.FlashMemory.data.appAltThresholdAltitude.asFloat;
    _azimuth_threshold = ConfigFlash.appConfig.FlashMemory.data.appAltThresholdAzimuth.asFloat;
    _validation_time_minutes = ConfigFlash.appConfig.FlashMemory.data.appAltValidationTime.asUINT32;

    _calibrated = true;
    _has_reference = true;
    _startup_settling_count = 0;  // Reset settling counter
    _state = MotionDetectionState::STARTUP;  // Stay in STARTUP for settling

    console->printOutput("MotionDetection: Calibration loaded, staying in STARTUP for settling period\n");

    return true;
}

SensorData *MotionDetection::getSensorData(std::uint8_t sensorId)
{
    // Set an iterator to the beginning of the list
    std::list<SensorData*>::iterator itr = list_of_sensor_ids.begin();

    // loop through the list and return the sensor if we have one matching the requested sensor_id
    for (itr = list_of_sensor_ids.begin(); itr != list_of_sensor_ids.end(); itr++)
    {
        if ((*itr)->_sensor_id == sensorId)
        {
            return *itr;
        }
    }

    return NULL;
}
/**
 * @brief Sets debug print mask to control various debug outputs
 * 
 * @param new_mask Bit mask controlling which debug prints are enabled (2^6 - 1 => 0-63).
 *                 so, e.g.: new_mask = 3 -> we enable raw data and info prints
 */
void MotionDetection::setDebugMask(unsigned int new_mask)
{
    PRINT_RAW = (new_mask & DEBUG_PRINT_RAW_ENABLE_MASK) != 0;
    PRINT_INFO = (new_mask & DEBUG_PRINT_INFO_ENABLE_MASK) != 0;
    PRINT_ANGLES = (new_mask & DEBUG_PRINT_ANGLES_ENABLE_MASK) != 0;
    PRINT_QUAT = (new_mask & DEBUG_PRINT_QUAT_ENABLE_MASK) != 0;
    PRINT_EVENTS = (new_mask & DEBUG_PRINT_EVENTS_ENABLE_MASK) != 0;
    PRINT_BIAS = (new_mask & DEBUG_PRINT_BIAS_ENABLE_MASK) != 0;
    PRINT_ESTIMATOR = (new_mask & DEBUG_PRINT_ESTIMATOR_ENABLE_MASK) != 0;
}

/* Returns 
0 if not calibrated
1 if calibrated
2 if calibration on-going
*/
uint32_t MotionDetection::getConfigValue()
{
    if (_state == MotionDetectionState::CALIBRATING) {
        return 2;
    } else {
        return static_cast<uint32_t>(_calibrated);
    }
}

const char *MotionDetection::getStateString() const
{
    switch (_state)
    {
    case MotionDetectionState::STARTUP:
        return "STARTUP";
    case MotionDetectionState::IDLE:
        return "IDLE";
    case MotionDetectionState::CALIBRATING:
        return "CALIBRATING";
    case MotionDetectionState::MONITORING:
        return "MONITORING";
    case MotionDetectionState::VALIDATING:
        return "VALIDATING";
    case MotionDetectionState::ERROR:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

int MotionDetection::setAltitudeThreshold(float threshold_deg)
{
    // Validate ranges
    if (threshold_deg < 0.0f || threshold_deg > 90.0f) {  // 0 deg to 90 deg
        console->printOutput("ERROR: Altitude threshold must be 0-90 (0 deg to 90 deg)\n");
        return -1;
    }
    _altitude_threshold = fmaxf(MIN_ALTITUDE_THRESHOLD_DEG, threshold_deg);
    ConfigFlash.appConfig.FlashMemory.data.appAltThresholdAltitude.asFloat = _altitude_threshold;
    ConfigFlash.setWriteRequest(0, EepromConfigMgr::EEPROM_WRITE_APP_CONFIG);
    osEventFlagsSet(debug_module_signals, DebugModuleThread::SIG_FLASH_SAVE_REQUEST);

    console->debugOutput("Altitude threshold set to %f degrees\n", _altitude_threshold);
    return 0;
}

int MotionDetection::setAzimuthThreshold(float threshold_deg)
{
    // Validate ranges
    if (threshold_deg < 0.0f || threshold_deg > 180.0f) {  // 0 deg to 180 deg
        console->printOutput("ERROR: Azimuth threshold must be 0-180 (0 deg to 180 deg)\n");
        return -1;
    }

    _azimuth_threshold = fmaxf(MIN_AZIMUTH_THRESHOLD_DEG, threshold_deg);
    ConfigFlash.appConfig.FlashMemory.data.appAltThresholdAzimuth.asFloat = _azimuth_threshold;
    ConfigFlash.setWriteRequest(0, EepromConfigMgr::EEPROM_WRITE_APP_CONFIG);
    osEventFlagsSet(debug_module_signals, DebugModuleThread::SIG_FLASH_SAVE_REQUEST);

    console->debugOutput("Azimuth threshold set to %f degrees\n", _azimuth_threshold);
    return 0;
}

int MotionDetection::setValidationTime(uint32_t minutes)
{
    // Validate range (1 minute to 10080 minutes = 1 week)
    if (minutes < 1 || minutes > 10080) {
        console->printOutput("ERROR: Validation time must be 1-10080 minutes (1 min to 1 week)\n");
        return -1;
    }

    _validation_time_minutes = minutes;
    ConfigFlash.appConfig.FlashMemory.data.appAltValidationTime.asUINT32 = minutes;
    ConfigFlash.setWriteRequest(0, EepromConfigMgr::EEPROM_WRITE_APP_CONFIG);
    osEventFlagsSet(debug_module_signals, DebugModuleThread::SIG_FLASH_SAVE_REQUEST);

    console->debugOutput("Validation time set to %u minutes\n", minutes);
    return 0;
}

bool MotionDetection::setOperationMode(MotionMode mode)
{
    _operation_mode = mode;
    _applyModeConfiguration();
    return true;
}

void MotionDetection::_calculateAnglesToReference(const Quaternion &current, const Quaternion &reference,
                                                 float &azimuth, float &altitude, float &zenith)
{
    Quaternion relative = QuaternionMath::relativeRotation(current, reference);
    float roll, pitch, yaw;
    data16 altitude_s, azimuth_s;
    QuaternionMath::toEulerAngles(relative, roll, pitch, yaw);

    // Apply range clamping for motion detection
    // We only care about magnitude, so clamp to positive ranges
    // Yaw: [0, 180] degrees, Pitch: [0, 180] degrees, Roll: [0, 90] degrees
    
    yaw = fabsf(yaw);
    if (yaw > 180.0f) yaw = 180.0f;
    
    pitch = fabsf(pitch);
    if (pitch > 180.0f) pitch = 180.0f;
    
    roll = fabsf(roll);
    if (roll > 90.0f) roll = 90.0f;

    // Use mapping directly for consistent coordinate assignment
    eulerToHorizontal(yaw, pitch, roll, azimuth, altitude, zenith);

    // update sensor data    
    altitude_s.asUINT16 = static_cast<uint16_t>(altitude * 10.0f);
    azimuth_s.asUINT16 = static_cast<uint16_t>(azimuth * 10.0f);
    _altitude_angle.updateData(altitude_s);
    _azimuth_angle.updateData(azimuth_s);
}


void MotionDetection::_applyModeConfiguration()
{
    ModeConfig config = _getModeConfig(_operation_mode);
    
    if(_operation_mode == MotionMode::DEBUG) {
        PRINT_RAW = PRINT_INFO = PRINT_ANGLES = true;
        PRINT_QUAT = PRINT_EVENTS = PRINT_BIAS = true;
    }
}
bool MotionDetection::_checkMotionThresholdsWithMode()
{
    const ModeConfig config = _getModeConfig(_operation_mode);

    if (_altitude_threshold <= 0.0f && _azimuth_threshold <= 0.0f) {
        return false;
    }

    // Clear ACTIVE events once the signal drops below the hysteresis thresholds.
    const float altitude_hyst = _altitude_threshold * HYSTERESIS_FACTOR;
    const float azimuth_hyst  = _azimuth_threshold  * HYSTERESIS_FACTOR;

    if (_altitude_event_active &&
        fabsf(_last_result.altitude_angle_degrees) < altitude_hyst)
    {
        _altitude_event_active = false;
        hw->clearAlarm(SENSOR_ALARM_ALTITUDE);
        if (PRINT_EVENTS) {
            _printEventData(_timing.getCurrentTimeStampUs(), "ALTITUDE_CLEARED");
        }
    }

    if (_azimuth_event_active &&
        fabsf(_last_result.azimuth_angle_degrees) < azimuth_hyst)
    {
        _azimuth_event_active = false;
        hw->clearAlarm(SENSOR_ALARM_AZIMUTH);
        if (PRINT_EVENTS) {
            _printEventData(_timing.getCurrentTimeStampUs(), "AZIMUTH_CLEARED");
        }
    }

    if (_combined_event_active &&
        fabsf(_last_result.altitude_angle_degrees) < altitude_hyst && fabsf(_last_result.azimuth_angle_degrees) < azimuth_hyst)
    {
        _combined_event_active = false;
        hw->clearAlarm(SENSOR_ALARM_ALTITUDE);
        hw->clearAlarm(SENSOR_ALARM_AZIMUTH);
        if (PRINT_EVENTS) {
            _printEventData(_timing.getCurrentTimeStampUs(), "COMBINED_CLEARED");
        }
    }

     // Evaluate if sample EXCEEDS the real thresholds.
    bool altitude_over_threshold =
            (_altitude_threshold > 0.0f) &&
            (fabsf(_last_result.altitude_angle_degrees) > _altitude_threshold);

    bool azimuth_over_threshold  =
            (_azimuth_threshold  > 0.0f) &&
            (fabsf(_last_result.azimuth_angle_degrees)  > _azimuth_threshold);

    // Optional rate gating depending on mode configuration.
    bool altitude_rate_ok = true;
    bool azimuth_rate_ok  = true;

    if (!config.enable_altitude_drift) {
        const float altitude_rate =
            fabsf(_last_result.altitude_angle_degrees - _prev_altitude) *
            SAMPLE_RATE_HZ;                                  // deg/s
        altitude_rate_ok = altitude_rate > config.altitude_drift_threshold_degree_per_second;  // 1 deg/s default
    }

    if (!config.enable_azimuth_drift) {
        const float azimuth_rate =
            fabsf(_last_result.azimuth_angle_degrees - _prev_azimuth) *
            SAMPLE_RATE_HZ;
        azimuth_rate_ok  = azimuth_rate > config.azimuth_drift_threshold_degree_per_second;    // 5 deg/s default
    }

    const bool altitude_exceeded = altitude_over_threshold && altitude_rate_ok;
    const bool azimuth_exceeded  = azimuth_over_threshold  && azimuth_rate_ok;

    // Book-keeping for next call
    _prev_altitude = _last_result.altitude_angle_degrees;
    _prev_azimuth  = _last_result.azimuth_angle_degrees;

    _last_result.is_event_active = _altitude_event_active || _azimuth_event_active;

    // Return “true” only if a **new** event should be raised.

    return (altitude_exceeded || azimuth_exceeded) && !_last_result.is_event_active;
}

// RESETTING
void MotionDetection::_checkAndApplyPeriodicReset()
{
    // Check if it's time for a periodic reset
    uint64_t samples_since_reset = _timing.total_samples_processed - _last_estimator_reset_sample;
    
    if (samples_since_reset >= DRIFT_RESET_INTERVAL_SAMPLES) {
        // Only reset if we're in a "safe" state (small angles, no active events)
        float max_current_angle = fmaxf(
            fabsf(_last_result.azimuth_angle_degrees),
            fmaxf(fabsf(_last_result.altitude_angle_degrees), 
                  fabsf(_last_result.zenith_angle_degrees))
        );
        
        bool safe_to_reset = (max_current_angle < MAX_ANGLE_FOR_RESET_DEG) && 
                            !_last_result.is_event_active &&
                            (_state == MotionDetectionState::MONITORING);
        
        if (safe_to_reset) {
            console->printOutput("MotionEstimator: Applying periodic drift reset (angles < %f deg)\n", 
                               MAX_ANGLE_FOR_RESET_DEG);
            
            // Reset MotionEstimator but synchronize with current MotionDI state
            _synchronizedEstimatorReset();
            _last_estimator_reset_sample = _timing.total_samples_processed;
        } else {
            // Defer reset until conditions are safe
            if (!_estimator_reset_pending) {
                console->printOutput("MotionEstimator: Drift reset deferred - angles too large or event active\n");
                _estimator_reset_pending = true;
            }
            
            // Check periodically if conditions improve
            if (samples_since_reset % (6240) == 0) { // Check every minute
                console->printOutput("MotionEstimator: Still waiting for safe reset conditions (max_angle=%f deg)\n", 
                                   max_current_angle);
            }
        }
    }
}

void MotionDetection::_synchronizedEstimatorReset()
{
    // TODO: Is this safe to do?
    _reference_quat = _current_quat;
    if (!_motion_estimator) return;
    
    // Get current MotionDI euler angles
    float current_euler[3];

    Quaternion orientation_quat = QuaternionMath::relativeRotation(_current_quat, _reference_quat);

    QuaternionMath::toEulerAngles(orientation_quat, current_euler[0], current_euler[1], current_euler[2]); // yaw, pitch, roll

    
    // Reset MotionEstimator filters but set reference to current MotionDI angles
    _motion_estimator->resetFilterStates(true);
    
    // Manually set the reference in MotionEstimator to match current state
    // This prevents the 0,0,0 reference problem
    _motion_estimator->setSynchronizedReference(current_euler);
    
    _estimator_reset_pending = false;
    
    console->printOutput("MotionEstimator: Synchronized reset complete - reference set to [%f, %f, %f] deg\n",
                        current_euler[0], current_euler[1], current_euler[2]);
}

// Logging methods implementation

void MotionDetection::_printInitialParameters()
{
    console->printOutput("=== MotionDetection Parameters (MotionDI Only) ===\n");
    console->printOutput("Sample Rate: %f Hz\n", SAMPLE_RATE_HZ);
    console->printOutput("Orientation: %s (accel), %s (gyro)\n",
                        _mdi_knobs.AccOrientation, _mdi_knobs.GyroOrientation);
    console->printOutput("Output Type: %s\n",
                        _mdi_knobs.SFKnob.output_type == MDI_ENGINE_OUTPUT_ENU ? "ENU" : "NED");
    console->printOutput("ATime: %f, FrTime: %f, MoveThresh: %f g\n",
                        _mdi_knobs.SFKnob.ATime, _mdi_knobs.SFKnob.FrTime, _mdi_knobs.AccKnob.MoveThresh_g);
    console->printOutput("AccThr: %f g, GyroThr: %f dps, MaxGyro: %f dps\n",
                        _mdi_knobs.GyrKnob.AccThr, _mdi_knobs.GyrKnob.GyroThr, _mdi_knobs.GyrKnob.MaxGyro);
    console->printOutput("================================\n");
}

void MotionDetection::_printRawData(const MotionSensorData &data)
{
    console->printOutput("RAW_DATA,%u,%f,%f,%f,%f,%f,%f\n",
                         data.timestamp_us,
                         data.accel_mg[0], data.accel_mg[1], data.accel_mg[2],
                         data.gyro_dps[0], data.gyro_dps[1], data.gyro_dps[2]);
}

void MotionDetection::_printAngleData(uint64_t timestamp_us)
{
    const char *state = getStateString();
    console->printOutput("ANGLES,%u,%f,%f,%f",
                        timestamp_us,
                        _last_result.altitude_angle_degrees,
                        _last_result.azimuth_angle_degrees,
                        _last_result.zenith_angle_degrees
    );
    console->printOutputWoTime(",%s\n", state);
}

void MotionDetection::_printQuatData(uint64_t timestamp_us)
{
    console->printOutput("QUAT,%u,%f,%f,%f,%f\n",
                         timestamp_us,
                         _mdi_output.quaternion[0], _mdi_output.quaternion[1],
                         _mdi_output.quaternion[2], _mdi_output.quaternion[3]);
}

void MotionDetection::_printEventData(uint64_t timestamp_us, const char *event_desc)
{
    console->printOutput("EVENT,%u,", timestamp_us);
    console->printOutputWoTime("%s", event_desc);
    console->printOutputWoTime(",%f,%f\n",
                         _last_result.altitude_angle_degrees,
                         _last_result.azimuth_angle_degrees);
}

void MotionDetection::_printBiasData(uint64_t timestamp_us)
{
    MDI_cal_output_t mdi_gyro_cal;
    MotionDI_GyrCal_getParams(&mdi_gyro_cal);

    console->printOutput("GYRO_BIAS_MDI,%u,%f,%f,%f\n",
                         timestamp_us, mdi_gyro_cal.Bias[0], mdi_gyro_cal.Bias[1], mdi_gyro_cal.Bias[2]);
}

void MotionDetection::_printReferenceQuaternion(uint64_t timestamp_us)
{
    float ref[4];
    _reference_quat.toArray(ref);

    console->printOutput("REFERENCE,%u,%f,%f,%f,%f,%f,%f,%f,%f\n",
                         timestamp_us,
                         ref[0], ref[1], ref[2], ref[3],
                         ref[0], ref[1], ref[2], ref[3]); // Duplicate for compatibility with other visualization tools
}

void MotionDetection::_printCalibrationComplete(uint64_t timestamp_us)
{
    console->printOutput("CALIBRATION_COMPLETE,%u\n", timestamp_us);
}

void MotionDetection::_printEstimatorAngles(uint64_t timestamp_us)
{
    // Print MotionDI euler angles (ANGLES_DI)
    float roll, pitch, yaw;
    Quaternion mdi_rotation_quat = QuaternionMath::relativeRotation(_current_quat, _reference_quat);
    QuaternionMath::toEulerAngles(mdi_rotation_quat, roll, pitch, yaw);
    console->printOutput("ANGLES_DI,%u,%f,%f,%f\n", timestamp_us, pitch, yaw, roll);
    
    // Print MotionEstimator data if available
    if (_motion_estimator && _motion_estimator->isReady()) {
        MotionEstimator::DebugInfo debug_info;
        _motion_estimator->getDebugInfo(debug_info);
        
        // Simple Integration Filter (ANGLES_SI)
        console->printOutput("ANGLES_SI,%u,%f,%f,%f\n", 
                           timestamp_us, 
                           debug_info.simple_filter[1],      // pitch
                           debug_info.simple_filter[0],      // yaw 
                           debug_info.simple_filter[2]);     // roll
        
        // Complementary Filter (ANGLES_CO)
        console->printOutput("ANGLES_CO,%u,%f,%f,%f\n", 
                           timestamp_us, 
                           debug_info.complementary_filter[1],  // pitch
                           debug_info.complementary_filter[0],  // yaw
                           debug_info.complementary_filter[2]); // roll
        
        // Fused Output (ANGLES_FU) - combination of MDI and MotionEstimator
        console->printOutput("ANGLES_FU,%u,%f,%f,%f\n", 
                           timestamp_us, 
                           debug_info.fused_output[1],       // pitch
                           debug_info.fused_output[0],       // yaw
                           debug_info.fused_output[2]);      // roll
    }
}