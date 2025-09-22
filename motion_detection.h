/*
 ********************************************************************************
 *  ADB Safegate Belgium BV
 *
 *  Motion Detection System
 *
 ********************************************************************************
 */

#ifndef MOTION_DETECTION_H
#define MOTION_DETECTION_H

#include <cstdint>
#include <cmath>
#include <cstring>
#include <list>
#include "adb_types.h"
#include "sensor_data.h"
#include "quaternion.h"

#define DEBUG_PRINT_RAW_ENABLE_MASK       (0x01)
#define DEBUG_PRINT_INFO_ENABLE_MASK      (0x02)
#define DEBUG_PRINT_ANGLES_ENABLE_MASK    (0x04)
#define DEBUG_PRINT_QUAT_ENABLE_MASK      (0x08)
#define DEBUG_PRINT_EVENTS_ENABLE_MASK    (0x10)
#define DEBUG_PRINT_BIAS_ENABLE_MASK      (0x20)
#define DEBUG_PRINT_ESTIMATOR_ENABLE_MASK (0x40)

extern "C"
{
#include "motion_di.h"
}

// Forward declaration of our MotionEstimator class
class MotionEstimator;

// Forward declarations
class Console;

// Constants
static constexpr float SAMPLE_RATE_HZ = 104.0f;
static constexpr float SAMPLE_PERIOD_S = 1.0f / SAMPLE_RATE_HZ;
static constexpr uint32_t SAMPLE_PERIOD_US = static_cast<uint32_t>(1000000.0f / SAMPLE_RATE_HZ); // 9615
static constexpr uint32_t CALIBRATION_SAMPLES = static_cast<uint32_t>(SAMPLE_RATE_HZ * 60); // 6240
static constexpr uint32_t MINIMUM_CALIBRATION_SAMPLES = static_cast<uint32_t>(SAMPLE_RATE_HZ * 10); // 1040
static constexpr uint32_t BAD_SAMPLE_THRESHOLD = 104 * 10; // 10 seconds of bad samples
static constexpr float MIN_AZIMUTH_THRESHOLD_DEG = 20; // 20 degrees
static constexpr float MIN_ALTITUDE_THRESHOLD_DEG = 10; // 10 degrees

/**
 * @brief Motion modes
 */
enum class MotionMode : uint8_t
{
    THRESHOLD_ONLY = 0, // Fast motion detection only
    ALTITUDE_DRIFT = 1, // Altitude drift detection, Azimuth threshold only
    FULL_DRIFT = 2,     // Both axis drift detection
    DEBUG = 3           // Debug mode with verbose output
};

/**
 * @brief Motion detection states
 */
enum class MotionDetectionState
{
    STARTUP = 0,
    IDLE,
    CALIBRATING,
    MONITORING,
    VALIDATING,
    ERROR
};

/**
 * @brief Motion event types
 */
enum class MotionEventType
{
    NO_EVENT = 0,
    ALTITUDE_EVENT,
    AZIMUTH_EVENT,
    COMBINED_EVENT
};

/**
 * @brief EEPROM data structure for persistence
 */
struct MotionCalibrationData
{
    uint32_t magic;                   // Magic number for validation (0xDEADBEEF) - we probably don't need this
    float reference_quaternion[4];    // W,X,Y,Z
    float gyro_bias[3];               // X,Y,Z bias values
    float altitude_threshold;         // Degrees
    float azimuth_threshold;          // Degrees
    uint32_t validation_time_minutes; // Time to wait before triggering event
    uint64_t persistent_validation_time_us;
    bool was_validating;
    uint64_t calibration_timestamp; // When calibration was performed
    uint32_t crc32;                 // CRC32 of all above data - we don't need this, I guess?
};

/**
 * @brief Sensor data structure
 */
struct MotionSensorData
{
    float accel_mg[3];
    float gyro_dps[3];
    bool accel_valid;
    bool gyro_valid;
    uint64_t timestamp_us;

    void toAccelG(float g[3]) const
    {
        g[0] = accel_mg[0] / 1000.0f;
        g[1] = accel_mg[1] / 1000.0f;
        g[2] = accel_mg[2] / 1000.0f;
    }
};

/**
 * @brief Motion detection result
 */
struct MotionResult
{
    float altitude_angle_degrees;
    float azimuth_angle_degrees;
    float zenith_angle_degrees; // For logging only
    MotionEventType event_type;
    bool is_event_active;
    bool is_calibrated;
    uint32_t validation_start_time_ms; // When validation period started

    MotionResult() : altitude_angle_degrees(0.0f),
                     azimuth_angle_degrees(0.0f),
                     zenith_angle_degrees(0.0f),
                     event_type(MotionEventType::NO_EVENT),
                     is_event_active(false),
                     is_calibrated(false),
                     validation_start_time_ms(0) {}
};

static constexpr float DRIFT_FILTER_ALPHA = 0.02f;  // Very slow filter (50 sample time constant)
static constexpr uint32_t DRIFT_FILTER_SAMPLES = 104 * 5;  // 5-second window


/**
 * @brief Motion detection class
 *
 */
class MotionDetection
{
public:
    MotionDetection();
    ~MotionDetection();

    bool initialize();
    bool processData(MotionSensorData &data, uint64_t sample_timestamp_us, uint8_t sample_index = 0);
    bool calibrate();
    int setAltitudeThreshold(float threshold_deg);
    int setAzimuthThreshold(float threshold_deg);
    int setValidationTime(uint32_t minutes);
    float getAltitudeThreshold() { return _altitude_threshold; }
    float getAzimuthThreshold() { return _azimuth_threshold; }

    MotionDetectionState getState() const { return _state; }
    const MotionResult &getLastResult() const { return _last_result; }
    bool isCalibrated() const { return _calibrated; }
    uint32_t getConfigValue();
    const char *getStateString() const;

    // EEPROM interface
    bool saveCalibrationData();
    bool loadCalibrationData();
    bool hasValidCalibrationData();

    // Motion mode control
    bool setOperationMode(MotionMode mode);
    static void setDebugMask(unsigned int new_mask);

    // Angle mapping
    void horizontalToEuler(float azimuth, float altitude, float zenith, float &yaw, float &pitch, float &roll);
    void eulerToHorizontal(float yaw, float pitch, float roll, float &azimuth, float &altitude, float &zenith);
    SensorData *getSensorData(std::uint8_t sensorId);
    std::list<SensorData *> list_of_sensor_ids;

    // Resets
    void _checkAndApplyPeriodicReset();
    void _synchronizedEstimatorReset();
    static constexpr uint32_t DRIFT_RESET_INTERVAL_SAMPLES = 104 * 60 * 10; // 10 minutes at 104Hz
    static constexpr float MAX_ANGLE_FOR_RESET_DEG = 10.0f; // Only reset if angles are small
    uint64_t _last_estimator_reset_sample = 0;
    bool _estimator_reset_pending = false;

    // Filters
    // - prefilters
    // MedianFilter median_filter;

    // Logging control
    static bool PRINT_RAW;
    static bool PRINT_INFO;
    static bool PRINT_ANGLES;
    static bool PRINT_QUAT;
    static bool PRINT_EVENTS;
    static bool PRINT_BIAS;
    static bool PRINT_ESTIMATOR;

private:
    // MotionDI structures
    MDI_knobs_t _mdi_knobs;
    MDI_output_t _mdi_output;
    MDI_cal_output_t _mdi_acc_cal;
    MDI_cal_output_t _mdi_gyro_cal;
    
    // Our custom MotionEstimator
    MotionEstimator* _motion_estimator;

    // Calibration
    bool _mdi_is_calibrated = false;
    bool _me_is_calibrated = false;

    // Post-processing
    float _filtered_azimuth{0.0f};
    float _filtered_altitude{0.0f};
    float _filtered_zenith{0.0f};
    // Simple rate tracking - just 3 values per axis
    struct RateTracker {
        float angles[3];        // Last 3 samples
        uint32_t timestamps[3]; // Last 3 timestamps
        uint8_t index = 0;      // Circular buffer index
        bool initialized = false;
    };
    RateTracker _altitude_rate;
    RateTracker _azimuth_rate;
    float _altitude_rate_threshold = 2.5f;  // degrees per second
    float _azimuth_rate_threshold = 5.0f;   // degrees per second

    // Reference and current quaternions
    Quaternion _reference_quat;
    Quaternion _current_quat;
    bool _has_reference;
    bool _calibrated;
    uint32_t _calibration_sample_count;
    uint32_t _startup_settling_count;  // Counter for settling period after power cycle

    // Thresholds and timing
    float _altitude_threshold = 0.0f;
    float _azimuth_threshold = 0.0f;
    float _prev_altitude{0.0f};
    float _prev_azimuth{0.0f};
    uint32_t _validation_time_minutes;
    uint64_t _persistent_validation_time_us; // Total validation time across reboots
    bool _was_validating;                    // Flag to track if we were validating at shutdown
    uint64_t _validation_start_timestamp_us;
    MotionEventType _pending_event_type;

    // Operation mode - moved to fix initialization order
    MotionMode _operation_mode{MotionMode::THRESHOLD_ONLY};

    // Event tracking
    bool _altitude_event_active;
    bool _azimuth_event_active;
    bool _combined_event_active;

    // Error tracking
    uint32_t _bad_sample_count;

    MotionDetectionState _state;
    MotionResult _last_result;

    SensorData _altitude_angle;
    SensorData _azimuth_angle;

    // Gyro noise detection
    // static constexpr float GYRO_NOISE_THRESHOLD_DPS = 50.0f;   // Jet blast detection
    // static constexpr uint32_t GYRO_NOISE_SETTLE_SAMPLES = 520; // 5 seconds
    // bool _high_gyro_noise_detected;
    // uint32_t _gyro_noise_sample_count;
    // float _gyro_magnitude_history[10];
    // uint32_t _gyro_history_index;

    // EEPROM
    bool _validateEEPROMData(float gyro_bias[]);
    void _resetStateVariables();
    bool _initializeMotionDIWithBias(float gyro_bias[]);

    bool _initializeMotionDI();
    bool _initializeMotionEstimator();
    bool _startMonitoring();
    void _processStateMachine();
    // void _preFilters(const MotionSensorData& data);
    void _updateMotionDI(const MotionSensorData &data);
    void _updateMotionEstimator(const MotionSensorData &data);
    void _calculateAnglesToReference(const Quaternion &current, const Quaternion &reference,
                                     float &azimuth, float &altitude, float &zenith);
    bool _checkMotionThresholds();
    bool _validateSensorData(const MotionSensorData &data);
    void _updateValidationState();
    bool _storeReferenceQuaternion();
    /**
     * @brief Unified coordinate conversion from WDS to ENU
     * @param wds Input in WDS coordinates (X-West, Y-Down, Z-South)  
     * @param enu Output in ENU coordinates (X-East, Y-North, Z-Up)
     */
    static void _convertWDStoENU(const float wds[3], float enu[3]);
    
    /**
     * @brief Apply consistent coordinate mapping using HorizontalCoordinatesMapping
     * @param euler_angles Input euler angles [yaw, pitch, roll] in degrees
     * @param horizontal_coords Output [azimuth, altitude, zenith] in degrees
     */
    void _detectAndHandleGyroNoise(const MotionSensorData &data);
    void _updateAngles();

    // Post filter

    void _addRateSample(float altitude, float azimuth, uint32_t timestamp_ms);
    bool _checkRateThresholds();

    void _printInitialParameters();
    void _printRawData(const MotionSensorData &data);
    void _printAngleData(uint64_t timestamp_us);
    void _printQuatData(uint64_t timestamp_us);
    void _printEventData(uint64_t timestamp_us, const char *event_desc);
    void _printBiasData(uint64_t timestamp_us);
    void _printReferenceQuaternion(uint64_t timestamp_us);
    void _printCalibrationComplete(uint64_t timestamp_us);
    void _printEstimatorAngles(uint64_t timestamp_us);

    // Mode configuration
    struct TimingSystem
    {
        static constexpr uint32_t SAMPLE_RATE_HZ = 104;
        static constexpr uint32_t SAMPLE_PERIOD_US = 9615; // 104Hz = 9615 microseconds
        uint64_t total_samples_processed{0};

        uint64_t getCurrentTimeStampUs() const
        {
            return total_samples_processed * SAMPLE_PERIOD_US;
        }
        uint32_t getCurrentTimeStampMs() const
        {
            return static_cast<uint32_t>(getCurrentTimeStampUs() / 1000);
        }
        void incrementSampleCount()
        {
            total_samples_processed++;
        }
    } _timing;
    struct ValidationManager
    {
        uint64_t accumulated_time_us{0};
        uint64_t required_time_us{0};
        uint64_t last_save_time_us{0};
        static constexpr uint64_t SAVE_INTERVAL_US = 600000000ULL;

        void update(uint64_t delta_us)
        {
            accumulated_time_us += delta_us;
            if (accumulated_time_us - last_save_time_us > SAVE_INTERVAL_US)
            {
                last_save_time_us = accumulated_time_us;
            }
        }

        bool isComplete() const
        {
            return accumulated_time_us >= required_time_us;
        }

        void reset()
        {
            accumulated_time_us = 0;
            last_save_time_us = 0;
        }

        void setRequiredTime(uint32_t minutes)
        {
            required_time_us = static_cast<uint64_t>(minutes) * 60ULL * 1000000ULL;
        }
    } _validation;
    struct CalibrationContext
    {
        float gyro_accumulator[3]{0.0f, 0.0f, 0.0f};
        uint32_t accumulated_samples{0};
        uint32_t timeout_samples{CALIBRATION_SAMPLES};

        void reset()
        {
            memset(gyro_accumulator, 0, sizeof(gyro_accumulator));
            accumulated_samples = 0;
            timeout_samples = CALIBRATION_SAMPLES;  // Reset timeout samples
        }

        void accumulate(const float gyro_dps[3])
        {
            for (int i = 0; i < 3; ++i)
            {
                gyro_accumulator[i] += gyro_dps[i];
            }
            accumulated_samples++;
        }

        bool computeBias(float bias_out[3])
        {
            if (accumulated_samples < timeout_samples)
            {
                return false; // Not enough samples
            }
            for (int i = 0; i < 3; ++i)
            {
                bias_out[i] = gyro_accumulator[i] / accumulated_samples;
            }
            return true;
        }
    } _calibration_context;

    struct ModeConfig
    {
        float altitude_drift_threshold_degree_per_second{1.0f};
        float azimuth_drift_threshold_degree_per_second{5.0f};
        bool enable_altitude_drift{false};
        bool enable_azimuth_drift{false};
    };

    ModeConfig _getModeConfig(MotionMode mode) const
    {
        ModeConfig config;
        switch (mode)
        {
        case MotionMode::THRESHOLD_ONLY:
            config.altitude_drift_threshold_degree_per_second = _altitude_threshold;
            config.azimuth_drift_threshold_degree_per_second = _azimuth_threshold;
            break;
        case MotionMode::ALTITUDE_DRIFT:
            config.enable_altitude_drift = true;
            config.azimuth_drift_threshold_degree_per_second = _azimuth_threshold;
            break;
        case MotionMode::FULL_DRIFT:
            config.enable_altitude_drift = true;
            config.enable_azimuth_drift = true;
            break;
        case MotionMode::DEBUG:
            PRINT_RAW = PRINT_INFO = PRINT_ANGLES = true;
            PRINT_QUAT = PRINT_EVENTS = PRINT_BIAS = true;
            break;
        }
        return config;
    }

    void _applyModeConfiguration();
    bool _checkMotionThresholdsWithMode();

    // Constants
    static constexpr float HYSTERESIS_FACTOR = 0.8f;
    static constexpr uint32_t EEPROM_MAGIC = 0xDEADBEEF;
    static constexpr float MAX_VALID_ACCEL_G = 2.0f;    // Maximum expected acceleration
    static constexpr float MAX_VALID_GYRO_DPS = 500.0f; // Maximum expected rotation rate
};

#endif // MOTION_DETECTION_H