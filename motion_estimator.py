"""
Motion Estimator - Python Port from C++ Implementation
Accurate implementation matching the C++ motion_estimator.cpp behavior
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import math

# Constants matching C++ implementation
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi
G_TO_MS2 = 9.80665
MG_TO_MS2 = G_TO_MS2 / 1000.0

@dataclass
class Config:
    """Configuration for MotionEstimator matching C++ Config struct"""
    sample_rate_hz: float = 104.0
    alpha: float = 0.88  # Complementary filter gain (from C++ line 125)
    azimuth_threshold_deg: float = 30.0
    altitude_threshold_deg: float = 15.0
    calibration_samples: int = 1040  # MINIMUM_CALIBRATION_SAMPLES from C++
    gyro_noise_threshold_dps: float = 1.8

@dataclass
class CalibrationData:
    """Calibration data matching C++ CalibrationData struct"""
    gyro_bias: np.ndarray = None
    acc_bias: np.ndarray = None
    is_calibrated: bool = False
    sample_count: int = 0
    
    def __post_init__(self):
        if self.gyro_bias is None:
            self.gyro_bias = np.zeros(3)
        if self.acc_bias is None:
            self.acc_bias = np.zeros(3)

class MotionEstimator:
    """
    Python implementation of the C++ MotionEstimator class.
    Uses combined simple and complementary filters for motion estimation.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the MotionEstimator with given configuration"""
        self.config = config or Config()
        self.dt = 1.0 / self.config.sample_rate_hz
        
        # Calibration
        self.calibration = CalibrationData()
        self.calibration_accumulator_gyro = np.zeros(3)
        self.calibration_accumulator_accel = np.zeros(3)
        
        # Filter states - matching C++ private members
        self.prev_yaw_deg = 0.0
        self.prev_pitch_deg = 0.0
        self.prev_roll_deg = 0.0
        
        # Simple filter state (gyro integration only)
        self.simple_yaw_deg = 0.0
        self.simple_pitch_deg = 0.0
        self.simple_roll_deg = 0.0
        
        # Complementary filter state
        self.comp_yaw_deg = 0.0
        self.comp_pitch_deg = 0.0
        self.comp_roll_deg = 0.0
        
        # Reference angles for relative measurements
        self.ref_yaw_deg = 0.0
        self.ref_pitch_deg = 0.0
        self.ref_roll_deg = 0.0
        self.has_reference = False
        
        # Preprocessing filter (low-pass)
        self.filter_alpha = 0.232  # From C++ line 269
        self.accel_filtered_prev = np.zeros(3)
        self.gyro_filtered_prev = np.zeros(3)
        
        # Set default calibration data (from C++ lines 189-197)
        self.set_default_calibration()
    
    def set_default_calibration(self):
        """Set default calibration values from C++ implementation"""
        # From motion_estimator.cpp lines 189-190
        self.calibration.gyro_bias = np.array([-0.398, 0.587, 0.770])
        self.calibration.acc_bias = np.array([-79.406, 204.207, 989.005])
        self.calibration.is_calibrated = True
    
    def add_calibration_sample(self, accel: np.ndarray, gyro: np.ndarray) -> bool:
        """
        Add calibration sample for bias calculation.
        Returns True when calibration is complete.
        """
        if self.calibration.is_calibrated:
            return True
        
        # Accumulate samples
        self.calibration_accumulator_accel += accel
        self.calibration_accumulator_gyro += gyro
        self.calibration.sample_count += 1
        
        # Check if we have enough samples
        if self.calibration.sample_count >= self.config.calibration_samples:
            # Calculate averages
            self.calibration.acc_bias = self.calibration_accumulator_accel / self.calibration.sample_count
            self.calibration.gyro_bias = self.calibration_accumulator_gyro / self.calibration.sample_count
            
            # Check if gyro bias is reasonable
            gyro_magnitude = np.linalg.norm(self.calibration.gyro_bias)
            
            if gyro_magnitude < self.config.gyro_noise_threshold_dps:
                self.calibration.is_calibrated = True
                return True
            else:
                # Reset and try again
                self.reset_calibration()
                return False
        
        return False
    
    def reset_calibration(self):
        """Reset calibration data"""
        self.calibration = CalibrationData()
        self.calibration_accumulator_gyro = np.zeros(3)
        self.calibration_accumulator_accel = np.zeros(3)
        self.reset_filter_states(reset_reference=True)
    
    def reset_filter_states(self, reset_reference: bool = True):
        """Reset all filter states"""
        # Reset filter states
        self.simple_yaw_deg = 0.0
        self.simple_pitch_deg = 0.0
        self.simple_roll_deg = 0.0
        
        self.comp_yaw_deg = 0.0
        self.comp_pitch_deg = 0.0
        self.comp_roll_deg = 0.0
        
        self.prev_yaw_deg = 0.0
        self.prev_pitch_deg = 0.0
        self.prev_roll_deg = 0.0
        
        # Reset preprocessing filter
        self.accel_filtered_prev = np.zeros(3)
        self.gyro_filtered_prev = np.zeros(3)
        
        if reset_reference:
            self.ref_yaw_deg = 0.0
            self.ref_pitch_deg = 0.0
            self.ref_roll_deg = 0.0
            self.has_reference = False
    
    def apply_preprocessing_filter(self, input_data: np.ndarray, prev_data: np.ndarray) -> np.ndarray:
        """Apply low-pass filter matching C++ _applyPreprocessingFilter"""
        if self.filter_alpha > 0:
            output = self.filter_alpha * input_data + (1.0 - self.filter_alpha) * prev_data
            return output
        return input_data
    
    def update_simple_filter(self, gyro_filtered: np.ndarray):
        """Update simple integration filter (C++ _updateSimpleFilter)"""
        # Simple integration for all axes
        # Note: C++ has TODO comments about swapped axes (lines 293-298)
        # Following the actual C++ implementation:
        
        # Yaw: integrate around Z-axis (gyro[2])
        self.simple_yaw_deg += gyro_filtered[2] * self.dt
        
        # Pitch: integrate around Y-axis but using gyro[0] due to semantics
        self.simple_pitch_deg += gyro_filtered[0] * self.dt
        
        # Roll: integrate around X-axis but using gyro[1] due to semantics
        self.simple_roll_deg += gyro_filtered[1] * self.dt
        
        # Normalize and convert to relative angles
        self.simple_yaw_deg = self.normalize_angle(self.simple_yaw_deg)
        self.simple_pitch_deg = self.normalize_angle(self.simple_pitch_deg)
        self.simple_roll_deg = self.normalize_angle(self.simple_roll_deg)
        
        # Convert to relative angles (C++ convertToRelativeAngles)
        if self.has_reference:
            self.simple_yaw_deg = abs(self.ref_yaw_deg - self.simple_yaw_deg)
            self.simple_pitch_deg = abs(self.ref_pitch_deg - self.simple_pitch_deg)
            self.simple_roll_deg = abs(self.ref_roll_deg - self.simple_roll_deg)
    
    def tilt_angle_from_accel(self, accel: np.ndarray, axis: int) -> float:
        """Calculate tilt angle from accelerometer (C++ _tiltAngleFromAccel)"""
        # Convert mg to m/s^2
        ax = accel[0] * MG_TO_MS2
        ay = accel[1] * MG_TO_MS2
        az = accel[2] * MG_TO_MS2
        
        if axis == 0:  # Roll (around X-axis)
            angle_rad = np.arctan2(ay, az)
        elif axis == 1:  # Pitch (around Y-axis)
            angle_rad = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
        else:
            return 0.0
        
        return angle_rad * RAD_TO_DEG
    
    def update_complementary_filter(self, accel_filtered: np.ndarray, gyro_filtered: np.ndarray):
        """Update complementary filter (C++ _updateComplementaryFilter)"""
        # Get tilt angles from accelerometer
        # Note: C++ has swapped axes (lines 316-318)
        acc_pitch_deg = self.tilt_angle_from_accel(accel_filtered, 0)  # Using axis 0 due to swap
        acc_roll_deg = self.tilt_angle_from_accel(accel_filtered, 1)   # Using axis 1 due to swap
        
        # Gyro integration (with swapped axes from C++ lines 322-326)
        gyro_pitch_deg = self.comp_pitch_deg + gyro_filtered[0] * self.dt
        gyro_roll_deg = self.comp_roll_deg + gyro_filtered[1] * self.dt
        gyro_yaw_deg = self.comp_yaw_deg + gyro_filtered[2] * self.dt
        
        # Complementary filter fusion
        self.comp_pitch_deg = self.config.alpha * gyro_pitch_deg + (1.0 - self.config.alpha) * acc_pitch_deg
        self.comp_roll_deg = self.config.alpha * gyro_roll_deg + (1.0 - self.config.alpha) * acc_roll_deg
        self.comp_yaw_deg = gyro_yaw_deg  # Yaw only from gyro
        
        # Normalize angles
        self.comp_yaw_deg = self.normalize_angle(self.comp_yaw_deg)
        self.comp_pitch_deg = self.normalize_angle(self.comp_pitch_deg)
        self.comp_roll_deg = self.normalize_angle(self.comp_roll_deg)
        
        # Convert to relative angles
        if self.has_reference:
            self.comp_yaw_deg = abs(self.ref_yaw_deg - self.comp_yaw_deg)
            self.comp_pitch_deg = abs(self.ref_pitch_deg - self.comp_pitch_deg)
            self.comp_roll_deg = abs(self.ref_roll_deg - self.comp_roll_deg)
    
    def combine_filters(self) -> Dict[str, float]:
        """Combine filters using logic from C++ _combineFilters"""
        # From C++ lines 345-361
        # Yaw: slow change with previous value smoothing
        yaw_deg = 0.6 * self.simple_yaw_deg + 0.4 * self.prev_yaw_deg
        
        # Pitch and Roll: Use complementary filter (less drift)
        pitch_deg = self.comp_pitch_deg
        roll_deg = self.comp_roll_deg
        
        # Normalize final angles
        yaw_deg = self.normalize_angle(yaw_deg)
        pitch_deg = self.normalize_angle(pitch_deg)
        roll_deg = self.normalize_angle(roll_deg)
        
        return {
            'yaw_deg': yaw_deg,
            'pitch_deg': pitch_deg,
            'roll_deg': roll_deg
        }
    
    def normalize_angle(self, angle_deg: float) -> float:
        """Normalize angle to [0, 180] range (C++ _normalizeAngle)"""
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return abs(angle_deg)
    
    def set_reference_angles(self):
        """Set reference angles from current state (C++ _setReferenceAngles)"""
        self.ref_yaw_deg = self.prev_yaw_deg
        self.ref_pitch_deg = self.prev_pitch_deg
        self.ref_roll_deg = self.prev_roll_deg
        self.has_reference = True
    
    def update(self, accel: list, gyro: list, timestamp: float) -> Dict[str, Any]:
        """
        Main update function matching C++ update method.
        
        Args:
            accel: Accelerometer data in mg [x, y, z]
            gyro: Gyroscope data in dps [x, y, z]
            timestamp: Timestamp in seconds
            
        Returns:
            Dictionary with angle estimates and metadata
        """
        accel = np.array(accel, dtype=np.float32)
        gyro = np.array(gyro, dtype=np.float32)
        
        # Return zero angles if not calibrated
        if not self.calibration.is_calibrated:
            return {
                'yaw_deg': 0.0,
                'pitch_deg': 0.0,
                'roll_deg': 0.0,
                'has_large_change': False,
                'timestamp': timestamp
            }
        
        # Apply bias correction (C++ lines 217-219)
        accel_corrected = accel - self.calibration.acc_bias
        gyro_corrected = gyro - self.calibration.gyro_bias
        
        # Apply preprocessing filter (C++ lines 221-223)
        accel_filtered = self.apply_preprocessing_filter(accel_corrected, self.accel_filtered_prev)
        gyro_filtered = self.apply_preprocessing_filter(gyro_corrected, self.gyro_filtered_prev)
        
        # Update previous filtered values
        self.accel_filtered_prev = accel_filtered.copy()
        self.gyro_filtered_prev = gyro_filtered.copy()
        
        # Update both filters (C++ lines 226-227)
        self.update_simple_filter(gyro_filtered)
        self.update_complementary_filter(accel_filtered, gyro_filtered)
        
        # Combine filters (C++ line 230)
        output = self.combine_filters()
        
        # Set reference angles if not set yet (C++ lines 233-235)
        if not self.has_reference:
            self.set_reference_angles()
        
        # Check for large changes
        yaw_change = abs(output['yaw_deg'] - self.prev_yaw_deg)
        pitch_change = abs(output['pitch_deg'] - self.prev_pitch_deg)
        has_large_change = (yaw_change > self.config.azimuth_threshold_deg or 
                           pitch_change > self.config.altitude_threshold_deg)
        
        # Update previous angles (C++ lines 238-240)
        self.prev_yaw_deg = output['yaw_deg']
        self.prev_pitch_deg = output['pitch_deg']
        self.prev_roll_deg = output['roll_deg']
        
        # Add metadata to output
        output['has_large_change'] = has_large_change
        output['timestamp'] = timestamp
        
        # Add debug information
        output['simple_yaw'] = self.simple_yaw_deg
        output['simple_pitch'] = self.simple_pitch_deg
        output['simple_roll'] = self.simple_roll_deg
        output['comp_yaw'] = self.comp_yaw_deg
        output['comp_pitch'] = self.comp_pitch_deg
        output['comp_roll'] = self.comp_roll_deg
        
        return output
    
    def is_ready(self) -> bool:
        """Check if estimator is ready (calibrated)"""
        return self.calibration.is_calibrated