import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MotionEstimatorConfig:
	# Matches C++ defaults and MotionDetection initialization
	sample_rate_hz: float = 104.0
	alpha: float = 0.88  # complementary filter gain (C++ sets 0.88 in MotionDetection)
	azimuth_threshold_deg: float = 10.0
	altitude_threshold_deg: float = 5.0
	calibration_samples: int = 1040  # MINIMUM_CALIBRATION_SAMPLES in C++ (10s)
	gyro_noise_threshold_dps: float = 1.8


class MotionEstimator:
	"""Python port of motion_estimator.{h,cpp} focusing on angle estimation.

	- Inputs:
	  accel: [mg] in ENU (X-East, Y-North, Z-Up)
	  gyro:  [dps] in ENU (X-East, Y-North, Z-Up)
	- Output angles are absolute magnitude of relative deviation from a reference
	  (see convertToRelativeAngles in C++), normalized to [0, 180].
	"""

	def __init__(self, config: MotionEstimatorConfig | None = None):
		self._config = config or MotionEstimatorConfig()
		self._dt = 1.0 / self._config.sample_rate_hz

		# Previous/fused angles
		self._prev_yaw_deg = 0.0
		self._prev_pitch_deg = 0.0
		self._prev_roll_deg = 0.0

		# Simple integration filter state (deg)
		self._simple_yaw_deg = 0.0
		self._simple_pitch_deg = 0.0
		self._simple_roll_deg = 0.0

		# Complementary filter state (deg)
		self._comp_yaw_deg = 0.0
		self._comp_pitch_deg = 0.0
		self._comp_roll_deg = 0.0

		# Reference angles for relative magnitude conversion
		self._ref_yaw_deg = 0.0
		self._ref_pitch_deg = 0.0
		self._ref_roll_deg = 0.0
		self._has_reference = False

		# Preprocessing low-pass filter alpha (C++ sets ~0.232 for 5 Hz cutoff at 104 Hz)
		self._pre_alpha = 0.232
		self._accel_filtered_prev = [0.0, 0.0, 0.0]
		self._gyro_filtered_prev = [0.0, 0.0, 0.0]

		# Calibration
		self._cal_is_calibrated = False
		self._cal_sample_count = 0
		self._acc_bias = [0.0, 0.0, 0.0]  # mg
		self._gyro_bias = [0.0, 0.0, 0.0]  # dps

	def is_ready(self) -> bool:
		return self._cal_is_calibrated

	def reset_calibration(self) -> None:
		self._cal_is_calibrated = False
		self._cal_sample_count = 0
		self._acc_bias = [0.0, 0.0, 0.0]
		self._gyro_bias = [0.0, 0.0, 0.0]
		self.reset_filter_states(True)

	def set_default_calibration_data(self) -> None:
		# Ported from C++ setDefaultCalibrationData
		default_gyro_bias = (-0.398, 0.587, 0.770)
		default_acc_bias = (-79.406, 204.207, 989.005)
		self._gyro_bias = list(default_gyro_bias)
		self._acc_bias = list(default_acc_bias)
		self._cal_is_calibrated = True

	def add_calibration_sample(self, accel_mg: List[float], gyro_dps: List[float]) -> bool:
		if self._cal_is_calibrated:
			return True
		for i in range(3):
			self._acc_bias[i] += float(accel_mg[i])
			self._gyro_bias[i] += float(gyro_dps[i])
		self._cal_sample_count += 1
		if self._cal_sample_count >= self._config.calibration_samples:
			for i in range(3):
				self._acc_bias[i] /= self._cal_sample_count
				self._gyro_bias[i] /= self._cal_sample_count
			gyro_mag = math.sqrt(self._gyro_bias[0] ** 2 + self._gyro_bias[1] ** 2 + self._gyro_bias[2] ** 2)
			if gyro_mag < self._config.gyro_noise_threshold_dps:
				self._cal_is_calibrated = True
				return True
			# If too noisy: reset and keep trying
			self.reset_calibration()
			return False
		return False

	def set_gyro_bias_from_external(self, gyro_bias: List[float]) -> None:
		self._gyro_bias = [float(gyro_bias[0]), float(gyro_bias[1]), float(gyro_bias[2])]
		# Sanity
		if all(abs(b) < 100.0 for b in self._gyro_bias):
			self._cal_is_calibrated = True

	def reset_filter_states(self, reset_reference: bool = True) -> None:
		self._simple_yaw_deg = self._simple_pitch_deg = self._simple_roll_deg = 0.0
		self._comp_yaw_deg = self._comp_pitch_deg = self._comp_roll_deg = 0.0
		self._prev_yaw_deg = self._prev_pitch_deg = self._prev_roll_deg = 0.0
		self._accel_filtered_prev = [0.0, 0.0, 0.0]
		self._gyro_filtered_prev = [0.0, 0.0, 0.0]
		if reset_reference:
			self._ref_yaw_deg = self._ref_pitch_deg = self._ref_roll_deg = 0.0
			self._has_reference = False

	def set_synchronized_reference(self, euler_angles: List[float]) -> None:
		# Weighted moving reference as in C++
		self._ref_yaw_deg = 0.7 * euler_angles[0] + 0.3 * self._ref_yaw_deg
		self._ref_pitch_deg = 0.7 * euler_angles[1] + 0.3 * self._ref_pitch_deg
		self._ref_roll_deg = 0.7 * euler_angles[2] + 0.3 * self._ref_roll_deg
		self._has_reference = True
		self._prev_yaw_deg = euler_angles[0]
		self._prev_pitch_deg = euler_angles[1]
		self._prev_roll_deg = euler_angles[2]

	def _apply_preprocessing_filter(self, inp: List[float], prev: List[float]) -> List[float]:
		alpha = self._pre_alpha
		out = [0.0, 0.0, 0.0]
		if alpha:
			for i in range(3):
				out[i] = alpha * float(inp[i]) + (1.0 - alpha) * prev[i]
				prev[i] = out[i]
		return out

	@staticmethod
	def _normalize_angle_deg(angle_deg: float) -> float:
		# Normalize to [-180, 180] then return absolute value (C++)
		while angle_deg > 180.0:
			angle_deg -= 360.0
		while angle_deg < -180.0:
			angle_deg += 360.0
		return abs(angle_deg)

	def _set_reference_angles(self) -> None:
		self._ref_yaw_deg = self._prev_yaw_deg
		self._ref_pitch_deg = self._prev_pitch_deg
		self._ref_roll_deg = self._prev_roll_deg
		self._has_reference = True

	def _convert_to_relative(self, yaw_deg: float, pitch_deg: float, roll_deg: float) -> tuple[float, float, float]:
		if not self._has_reference:
			self._set_reference_angles()
		yaw = abs(self._ref_yaw_deg - yaw_deg)
		pitch = abs(self._ref_pitch_deg - pitch_deg)
		roll = abs(self._ref_roll_deg - roll_deg)
		return yaw, pitch, roll

	@staticmethod
	def _tilt_angle_from_accel(accel_mg: List[float], axis: int) -> float:
		# Convert mg to m/s^2 as in C++ (but only the ratio matters for atan2)
		G_TO_MS2 = 9.80665
		MG_TO_MS2 = G_TO_MS2 / 1000.0
		ax = accel_mg[0] * MG_TO_MS2
		ay = accel_mg[1] * MG_TO_MS2
		az = accel_mg[2] * MG_TO_MS2
		# C++ uses:
		# if axis == 0: atan2(ay, az)
		# if axis == 1: atan2(-ax, sqrt(ay^2 + az^2))
		if axis == 0:
			angle_rad = math.atan2(ay, az)
		elif axis == 1:
			angle_rad = math.atan2(-ax, math.sqrt(ay * ay + az * az))
		else:
			return 0.0
		return math.degrees(angle_rad)

	def update(self, accel_mg: List[float], gyro_dps: List[float], timestamp_s: float | int | None = None) -> Dict[str, float]:
		"""Update estimator and return angles dict.

		If not calibrated yet, this call accumulates calibration samples and returns zeros.
		"""
		if not self._cal_is_calibrated:
			self.add_calibration_sample(accel_mg, gyro_dps)
			return {
				"yaw_deg": 0.0,
				"pitch_deg": 0.0,
				"roll_deg": 0.0,
				"has_large_change": False,
				"timestamp": float(timestamp_s) if timestamp_s is not None else 0.0,
			}

		# Bias correction
		accel_corr = [accel_mg[0] - self._acc_bias[0],
					 accel_mg[1] - self._acc_bias[1],
					 accel_mg[2] - self._acc_bias[2]]
		gyro_corr = [gyro_dps[0] - self._gyro_bias[0],
					 gyro_dps[1] - self._gyro_bias[1],
					 gyro_dps[2] - self._gyro_bias[2]]

		# Pre-filter
		accel_f = self._apply_preprocessing_filter(accel_corr, self._accel_filtered_prev)
		gyro_f = self._apply_preprocessing_filter(gyro_corr, self._gyro_filtered_prev)

		# Simple integration filter (deg):
		# yaw: integrate gyro[2], pitch: integrate gyro[0], roll: integrate gyro[1]
		self._simple_yaw_deg = self._simple_yaw_deg + gyro_f[2] * self._dt
		self._simple_pitch_deg = self._simple_pitch_deg + gyro_f[0] * self._dt
		self._simple_roll_deg = self._simple_roll_deg + gyro_f[1] * self._dt

		# Normalize
		simple_yaw = self._normalize_angle_deg(self._simple_yaw_deg)
		simple_pitch = self._normalize_angle_deg(self._simple_pitch_deg)
		simple_roll = self._normalize_angle_deg(self._simple_roll_deg)
		simple_yaw, simple_pitch, simple_roll = self._convert_to_relative(simple_yaw, simple_pitch, simple_roll)
		self._simple_yaw_deg, self._simple_pitch_deg, self._simple_roll_deg = simple_yaw, simple_pitch, simple_roll

		# Complementary filter for tilt, yaw from gyro only
		acc_pitch = self._tilt_angle_from_accel(accel_f, 0)  # swapped semantics same as C++
		acc_roll = self._tilt_angle_from_accel(accel_f, 1)
		gyro_pitch = self._comp_pitch_deg + gyro_f[0] * self._dt
		gyro_roll = self._comp_roll_deg + gyro_f[1] * self._dt
		gyro_yaw = self._comp_yaw_deg + gyro_f[2] * self._dt

		self._comp_pitch_deg = self._config.alpha * gyro_pitch + (1.0 - self._config.alpha) * acc_pitch
		self._comp_roll_deg = self._config.alpha * gyro_roll + (1.0 - self._config.alpha) * acc_roll
		self._comp_yaw_deg = gyro_yaw

		comp_yaw = self._normalize_angle_deg(self._comp_yaw_deg)
		comp_pitch = self._normalize_angle_deg(self._comp_pitch_deg)
		comp_roll = self._normalize_angle_deg(self._comp_roll_deg)
		comp_yaw, comp_pitch, comp_roll = self._convert_to_relative(comp_yaw, comp_pitch, comp_roll)
		self._comp_yaw_deg, self._comp_pitch_deg, self._comp_roll_deg = comp_yaw, comp_pitch, comp_roll

		# Combine filters
		yaw_out = 0.6 * self._simple_yaw_deg + 0.4 * self._prev_yaw_deg
		pitch_out = self._comp_pitch_deg
		roll_out = self._comp_roll_deg

		yaw_out = self._normalize_angle_deg(yaw_out)
		pitch_out = self._normalize_angle_deg(pitch_out)
		roll_out = self._normalize_angle_deg(roll_out)

		if not self._has_reference:
			self._set_reference_angles()

		self._prev_yaw_deg = yaw_out
		self._prev_pitch_deg = pitch_out
		self._prev_roll_deg = roll_out

		return {
			"yaw_deg": yaw_out,
			"pitch_deg": pitch_out,
			"roll_deg": roll_out,
			"has_large_change": False,
			"timestamp": float(timestamp_s) if timestamp_s is not None else 0.0,
		}

	def get_debug_info(self) -> Dict[str, List[float]]:
		return {
			"simple_filter": [self._simple_yaw_deg, self._simple_pitch_deg, self._simple_roll_deg],
			"complementary_filter": [self._comp_yaw_deg, self._comp_pitch_deg, self._comp_roll_deg],
			"fused_output": [self._prev_yaw_deg, self._prev_pitch_deg, self._prev_roll_deg],
			"reference_angles": [self._ref_yaw_deg, self._ref_pitch_deg, self._ref_roll_deg],
			"calibrated": self._cal_is_calibrated,
			"calibration_samples": self._cal_sample_count,
		}

