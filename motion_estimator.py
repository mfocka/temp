import numpy as np
 
M_PI = np.pi
DEG_TO_RAD = M_PI / 180.0
RAD_TO_DEG = 180.0 / M_PI
G_TO_MS2 = 9.80665
MG_TO_MS2 = G_TO_MS2 / 1000.0
 
 
class MotionEstimator:
    """
    Extended complementary filter for ISM330DHCX-style IMU data.
    - ODR is assumed fixed at 104 Hz
    - Preprocessing alpha is fixed at 0.232
    - Complementary filter alpha is fixed at 0.98
    """
 
    def __init__(self):
        self._hz = 104
        self._dt = 1.0 / self._hz
 
        # Calibration biases (from characterization)
        self._calibration_gyro_bias: list[float] = np.array([-0.398, 0.587, 0.770])
        self._calibration_acc_bias: list[float]  = np.array([-79.406, 204.207, 989.005])
 
        # Previous filtered states
        self._accel_filtered_prev = np.zeros(3)
        self._gyro_filtered_prev = np.zeros(3)
 
        # Reference angles (for relative mode)
        self._ref_yaw_deg = 0.0
        self._ref_pitch_deg = 0.0
        self._ref_roll_deg = 0.0
 
        # Previous angles (for differencing / sanity checks)
        self._prev_yaw_deg = 0.0
        self._prev_pitch_deg = 0.0
        self._prev_roll_deg = 0.0
 
        # State angles
        self._simple_yaw_deg = 0.0
        self._simple_pitch_deg = 0.0
        self._simple_roll_deg = 0.0
 
        self._comp_yaw_deg = 0.0
        self._comp_pitch_deg = 0.0
        self._comp_roll_deg = 0.0
 
    # --- Core math helpers ---
    def tiltAngleFromAccel(self, accel: list[float], axis: int) -> float:
        """
        Compute tilt angle from accelerometer.
        axis=0 -> pitch
        axis=1 -> roll
        """
        ax = accel[0] * MG_TO_MS2
        ay = accel[1] * MG_TO_MS2
        az = accel[2] * MG_TO_MS2
 
        if axis == 0:  # pitch
            angle_rad = np.arctan2(ay, az)
        elif axis == 1:  # roll
            angle_rad = np.arctan2(-ax, np.sqrt(ay * ay + az * az))
        else:
            return 0.0
 
        return angle_rad * RAD_TO_DEG
 
    def preprocessing(self, input: list[float], prev: list[float]) -> list[float]:
        """
        Simple one-pole IIR low-pass filter for accel/gyro.
        Alpha fixed at 0.232 for ODR=104 Hz.
        """
        alpha = 0.232
        output = np.zeros(3)
        for i in range(3):
            output[i] = alpha * input[i] + (1 - alpha) * prev[i]
            prev[i] = output[i]
        return output
 
    def normalizeAngle(self, angle: float) -> float:
        """Wrap angle into [-180, 180] range."""
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle
 
    def convertToRelativeAngles(self, output: dict) -> dict:
        """Subtract reference to produce relative angles."""
        output["yaw_deg"] -= self._ref_yaw_deg
        output["pitch_deg"] -= self._ref_pitch_deg
        output["roll_deg"] -= self._ref_roll_deg
        return output
 
    # --- Filters ---
    def updateSimpleFilter(self, gyro_filtered: list[float], makeRelative=True):
        """Pure gyro integration (Euler) filter."""
        self._simple_yaw_deg += gyro_filtered[2] * self._dt
        self._simple_pitch_deg += gyro_filtered[0] * self._dt
        self._simple_roll_deg += gyro_filtered[1] * self._dt
 
        output = {
            "yaw_deg": self.normalizeAngle(self._simple_yaw_deg),
            "pitch_deg": self.normalizeAngle(self._simple_pitch_deg),
            "roll_deg": self.normalizeAngle(self._simple_roll_deg),
        }
 
        if makeRelative:
            output = self.convertToRelativeAngles(output)
 
        self._simple_yaw_deg = output["yaw_deg"]
        self._simple_pitch_deg = output["pitch_deg"]
        self._simple_roll_deg = output["roll_deg"]
 
    def updateComplementaryFilter(
        self, accel_filtered: list[float], gyro_filtered: list[float], makeRelative=True
    ):
        """Complementary filter for pitch/roll (gyro+accel), yaw=gyro only."""
        acc_pitch_deg = self.tiltAngleFromAccel(accel_filtered, 0)
        acc_roll_deg = self.tiltAngleFromAccel(accel_filtered, 1)
 
        gyro_pitch_deg = self._comp_pitch_deg + gyro_filtered[0] * self._dt
        gyro_roll_deg = self._comp_roll_deg + gyro_filtered[1] * self._dt
        gyro_yaw_deg = self._comp_yaw_deg + gyro_filtered[2] * self._dt
 
        alpha = 0.98
        self._comp_pitch_deg = alpha * gyro_pitch_deg + (1.0 - alpha) * acc_pitch_deg
        self._comp_roll_deg = alpha * gyro_roll_deg + (1.0 - alpha) * acc_roll_deg
        self._comp_yaw_deg = gyro_yaw_deg
 
        output = {
            "yaw_deg": self.normalizeAngle(self._comp_yaw_deg),
            "pitch_deg": self.normalizeAngle(self._comp_pitch_deg),
            "roll_deg": self.normalizeAngle(self._comp_roll_deg),
        }
 
        if makeRelative:
            output = self.convertToRelativeAngles(output)
 
        self._comp_yaw_deg = output["yaw_deg"]
        self._comp_pitch_deg = output["pitch_deg"]
        self._comp_roll_deg = output["roll_deg"]
 
    def combineFilters(self) -> dict:
        """
        Fuse simple yaw with complementary yaw (sanity check).
        - Pitch/Roll from complementary
        - Yaw from simple (unless large divergence)
        """
        yaw_diff = abs(self._comp_yaw_deg - self._prev_yaw_deg)
 
        if yaw_diff < 10.0:  # small difference
            yaw = self._simple_yaw_deg
        else:  # weighted mix
            yaw = 0.6 * self._simple_yaw_deg + 0.4 * self._comp_yaw_deg
 
        output = {
            "yaw_deg": self.normalizeAngle(yaw),
            "pitch_deg": self.normalizeAngle(self._comp_pitch_deg),
            "roll_deg": self.normalizeAngle(self._comp_roll_deg),
        }
 
        return output
 
    # --- Main update ---
    def update(self, accel_raw: list[float], gyro_raw: list[float], timestamp: float) -> dict:
        """
        Main update function.
        Input: raw accel (mg), raw gyro (dps), timestamp (s).
        Returns: dict with yaw, pitch, roll in degrees.
        """
        # Apply calibration
        accel_corrected = np.array(accel_raw) - self._calibration_acc_bias
        gyro_corrected = np.array(gyro_raw) - self._calibration_gyro_bias
 
        # Low-pass filter
        accel_filtered = self.preprocessing(accel_corrected, self._accel_filtered_prev)
        gyro_filtered = self.preprocessing(gyro_corrected, self._gyro_filtered_prev)
 
        # Update filters
        self.updateSimpleFilter(gyro_filtered)
        self.updateComplementaryFilter(accel_filtered, gyro_filtered)
 
        # Combine
        output = self.combineFilters()
 
        # Store prev
        self._prev_yaw_deg = output["yaw_deg"]
        self._prev_pitch_deg = output["pitch_deg"]
        self._prev_roll_deg = output["roll_deg"]
 
        output["timestamp"] = timestamp
        return output