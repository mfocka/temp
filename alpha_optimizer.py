import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, signal  # type: ignore[import]
from typing import Tuple, List, Dict, Optional
import json

class AlphaOptimizer:
    """
    Optimizes the complementary filter alpha parameter for IMU fusion
    """
    
    def __init__(self, sample_rate=104.0):
        self.dt = 1.0 / sample_rate
        self.sample_rate = sample_rate
        
    def complementary_filter(self, accel_data: np.ndarray, gyro_data: np.ndarray, 
                            alpha: float, initial_angle: float = 0.0) -> np.ndarray:
        """
        Run complementary filter with given alpha
        
        Args:
            accel_data: Accelerometer data (N samples)
            gyro_data: Gyroscope data (N samples)
            alpha: Filter parameter (0-1)
            initial_angle: Starting angle
            
        Returns:
            Filtered angle estimates
        """
        n_samples = len(accel_data)
        angles = np.zeros(n_samples)
        angles[0] = initial_angle
        
        for i in range(1, n_samples):
            # Gyro integration
            gyro_angle = angles[i-1] + gyro_data[i] * self.dt
            
            # Accelerometer angle (simplified for single axis)
            accel_angle = accel_data[i]
            
            # Complementary filter
            angles[i] = alpha * gyro_angle + (1 - alpha) * accel_angle
            
        return angles
    
    def yaw_alpha_filter(self, gyro_data: np.ndarray, alpha: float, initial_angle: float = 0.0) -> np.ndarray:
        """
        Gyro-only yaw estimation with alpha smoothing between previous angle and
        current integrated angle (no accelerometer contribution).
        
        Args:
            gyro_data: Gyroscope Z-axis rates (deg/s), length N
            alpha: Smoothing parameter (0-1). Higher alpha -> more inertia
            initial_angle: Starting yaw angle in degrees
        
        Returns:
            Filtered yaw angle estimates in degrees
        """
        n_samples = len(gyro_data)
        angles = np.zeros(n_samples)
        if n_samples == 0:
            return angles
        angles[0] = initial_angle
        
        for i in range(1, n_samples):
            # Pure gyro integration from previous filtered angle
            integrated_angle = angles[i-1] + gyro_data[i] * self.dt
            # Alpha smoothing between previous angle and current integrated angle
            angles[i] = alpha * angles[i-1] + (1 - alpha) * integrated_angle
        
        return angles
    
    def calculate_accel_tilt(self, accel_x: np.ndarray, accel_y: np.ndarray, 
                            accel_z: np.ndarray, axis: str = 'pitch') -> np.ndarray:
        """
        Calculate tilt angle from accelerometer data
        
        Args:
            accel_x, accel_y, accel_z: Accelerometer data in mg
            axis: 'pitch' or 'roll'
            
        Returns:
            Tilt angles in degrees
        """
        # Convert mg to m/s^2
        ax = accel_x * 0.00981
        ay = accel_y * 0.00981
        az = accel_z * 0.00981
        
        if axis == 'pitch':
            angles = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        else:  # roll
            angles = np.arctan2(ay, az)
            
        return np.degrees(angles)
    
    def calculate_gyro_tilt(self, gyro_x: np.ndarray, gyro_y: np.ndarray, gyro_z: np.ndarray, axis: str = 'yaw') -> np.ndarray:
        """
        Return gyroscope rates (deg/s) for the specified axis.
        """
        if axis == 'yaw':
            rates = gyro_z
        elif axis == 'pitch':
            rates = gyro_y
        else:
            rates = gyro_x
        return rates
    def grid_search(self, accel_data: Optional[np.ndarray], gyro_data: np.ndarray,
                   reference: Optional[np.ndarray] = None, alpha_range: Tuple[float, float] = (0.1, 0.99),
                   n_steps: int = 20, axis: str = 'pitch') -> Dict:
        """
        Grid search for optimal alpha
        
        Args:
            accel_data: Accelerometer-derived angles (ignored for axis='yaw')
            gyro_data: Gyroscope data (deg/s)
            reference: Ground truth angles (if available)
            alpha_range: Min and max alpha to test
            n_steps: Number of alpha values to test
            axis: Which axis to optimize ('pitch', 'roll', or 'yaw')
            
        Returns:
            Dictionary with results
        """
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_steps)
        errors = []
        
        for alpha in alphas:
            # Run filter depending on axis
            if axis == 'yaw':
                filtered = self.yaw_alpha_filter(gyro_data, alpha)
                # Calculate error
                if reference is not None:
                    error = np.mean((filtered - reference)**2)
                else:
                    error = self._yaw_error_metric(filtered, gyro_data)
            else:
                assert accel_data is not None
                filtered = self.complementary_filter(accel_data, gyro_data, alpha)
                # Calculate error
                if reference is not None:
                    # Use ground truth
                    error = np.mean((filtered - reference)**2)
                else:
                    # Use smoothness and drift metrics
                    error = self._combined_error_metric(filtered, accel_data, gyro_data)
            
            errors.append(error)
        
        # Find best alpha
        best_idx = np.argmin(errors)
        best_alpha = alphas[best_idx]
        
        return {
            'alphas': alphas,
            'errors': errors,
            'best_alpha': best_alpha,
            'best_error': errors[best_idx]
        }
    
    def optimize_alpha(self, accel_data: Optional[np.ndarray], gyro_data: np.ndarray,
                      reference: Optional[np.ndarray] = None, axis: str = 'pitch') -> Dict:
        """
        Optimize alpha using scipy optimization
        
        Args:
            accel_data: Accelerometer-derived angles (ignored for axis='yaw')
            gyro_data: Gyroscope data (deg/s)
            reference: Ground truth angles (if available)
            axis: Which axis to optimize ('pitch', 'roll', or 'yaw')
            
        Returns:
            Optimization results
        """
        def objective(alpha):
            # Ensure alpha is in valid range
            alpha = np.clip(alpha, 0.01, 0.99)
            
            # Run filter
            if axis == 'yaw':
                filtered = self.yaw_alpha_filter(gyro_data, alpha)
                # Calculate error
                if reference is not None:
                    error = np.mean((filtered - reference)**2)
                else:
                    error = self._yaw_error_metric(filtered, gyro_data)
            else:
                assert accel_data is not None
                filtered = self.complementary_filter(accel_data, gyro_data, alpha)
                # Calculate error
                if reference is not None:
                    error = np.mean((filtered - reference)**2)
                else:
                    error = self._combined_error_metric(filtered, accel_data, gyro_data)
            
            return error
        
        # Initial guess
        x0 = 0.9
        
        # Optimize
        result = optimize.minimize_scalar(objective, bounds=(0.01, 0.99), method='bounded')
        
        # Get filter output with optimal alpha
        if axis == 'yaw':
            optimal_filtered = self.yaw_alpha_filter(gyro_data, result.x)
        else:
            assert accel_data is not None
            optimal_filtered = self.complementary_filter(accel_data, gyro_data, result.x)
        
        return {
            'best_alpha': result.x,
            'best_error': result.fun,
            'success': result.success,
            'filtered_signal': optimal_filtered
        }
    
    def _combined_error_metric(self, filtered: np.ndarray, accel: np.ndarray, 
                               gyro: np.ndarray) -> float:
        """
        Combined error metric when no ground truth available
        
        Balances:
        - Smoothness (low noise)
        - Drift resistance (tracks accelerometer long-term)
        - Responsiveness (follows gyro short-term)
        """
        # Smoothness: penalize high-frequency noise
        smoothness = np.mean(np.diff(filtered)**2)
        
        # Drift: long-term deviation from accelerometer
        window = min(len(filtered) // 10, 100)  # Adaptive window
        if window > 1:
            filtered_avg = signal.convolve(filtered, np.ones(window)/window, mode='valid')
            accel_avg = signal.convolve(accel, np.ones(window)/window, mode='valid')
            drift = np.mean((filtered_avg - accel_avg)**2)
        else:
            drift = np.mean((filtered - accel)**2)
        
        # Responsiveness: short-term tracking of changes
        if len(gyro) > 1:
            gyro_changes = np.diff(np.cumsum(gyro) * self.dt)
            filtered_changes = np.diff(filtered)
            responsiveness = np.mean((filtered_changes - gyro_changes)**2)
        else:
            responsiveness = 0
        
        # Weighted combination
        weights = {'smoothness': 0.3, 'drift': 0.5, 'responsiveness': 0.2}
        error = (weights['smoothness'] * smoothness + 
                weights['drift'] * drift + 
                weights['responsiveness'] * responsiveness)
        
        return error
    
    def analyze_filter_characteristics(self, alpha: float) -> Dict:
        """
        Analyze filter characteristics for given alpha
        
        Returns time constants, cutoff frequencies, etc.
        """
        # Time constant
        tau = -self.dt / np.log(alpha)
        
        # Cutoff frequency (approximate)
        fc = 1 / (2 * np.pi * tau)
        
        # Settling time (to 95%)
        settling_time = 3 * tau
        
        return {
            'alpha': alpha,
            'time_constant_s': tau,
            'cutoff_frequency_hz': fc,
            'settling_time_s': settling_time,
            'samples_to_settle': int(settling_time * self.sample_rate)
        }
    
    def visualize_optimization(self, results: Dict, accel_data: Optional[np.ndarray], 
                              gyro_data: np.ndarray, reference: Optional[np.ndarray] = None, 
                              axis: str = 'pitch'):
        """
        Visualize optimization results
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Error vs Alpha
        ax = axes[0, 0]
        if 'alphas' in results:
            ax.plot(results['alphas'], results['errors'], 'b-')
            ax.plot(results['best_alpha'], results['best_error'], 'ro', markersize=10)
            ax.set_xlabel('Alpha')
            ax.set_ylabel('Error')
            ax.set_title(f'Error vs Alpha (Best: {results["best_alpha"]:.3f})')
            ax.grid(True, alpha=0.3)
        
        # Plot 2: Filter comparison
        ax = axes[0, 1]
        n_samples = len(gyro_data) if gyro_data is not None else (len(accel_data) if accel_data is not None else 0)
        time = np.arange(n_samples) * self.dt
        
        # Test different alphas
        test_alphas = [0.5, 0.8, 0.9, 0.95, results['best_alpha']]
        for alpha in test_alphas:
            if axis == 'yaw':
                filtered = self.yaw_alpha_filter(gyro_data, alpha)
            else:
                assert accel_data is not None
                filtered = self.complementary_filter(accel_data, gyro_data, alpha)
            label = f'α={alpha:.2f}'
            if alpha == results['best_alpha']:
                label += ' (optimal)'
                ax.plot(time, filtered, linewidth=2, label=label)
            else:
                ax.plot(time, filtered, alpha=0.5, label=label)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Filter Output Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Input signals
        ax = axes[1, 0]
        if axis != 'yaw' and accel_data is not None:
            ax.plot(time, accel_data, 'g-', alpha=0.7, label='Accel angle')
        ax.plot(time, np.cumsum(gyro_data) * self.dt, 'b-', alpha=0.7, label='Gyro integrated')
        if reference is not None:
            ax.plot(time, reference, 'k--', alpha=0.7, label='Reference')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title('Input Signals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Filter characteristics
        ax = axes[1, 1]
        alphas = np.linspace(0.1, 0.99, 100)
        taus = []
        fcs = []
        for alpha in alphas:
            char = self.analyze_filter_characteristics(alpha)
            taus.append(char['time_constant_s'])
            fcs.append(char['cutoff_frequency_hz'])
        
        ax2 = ax.twinx()
        l1 = ax.plot(alphas, taus, 'b-', label='Time constant')
        l2 = ax2.plot(alphas, fcs, 'r-', label='Cutoff freq')
        
        # Mark optimal
        opt_char = self.analyze_filter_characteristics(results['best_alpha'])
        ax.plot(results['best_alpha'], opt_char['time_constant_s'], 'bo', markersize=8)
        ax2.plot(results['best_alpha'], opt_char['cutoff_frequency_hz'], 'ro', markersize=8)
        
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Time Constant (s)', color='b')
        ax2.set_ylabel('Cutoff Frequency (Hz)', color='r')
        ax.set_title('Filter Characteristics')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Combine legends
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='center right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _yaw_error_metric(self, filtered: np.ndarray, gyro: np.ndarray) -> float:
        """
        Error metric for yaw when no accelerometer reference exists.
        Balances smoothness and responsiveness to gyro changes.
        """
        # Smoothness: penalize rapid changes in the output
        smoothness = np.mean(np.diff(filtered)**2) if len(filtered) > 1 else 0.0
        
        # Responsiveness: filtered derivative should track gyro rate (deg)
        if len(gyro) > 0 and len(filtered) > 1:
            # Expected change per sample from gyro integration
            expected_changes = gyro[1:] * self.dt
            actual_changes = np.diff(filtered)
            # Align lengths (in case of any mismatch)
            k = min(len(expected_changes), len(actual_changes))
            responsiveness = np.mean((actual_changes[:k] - expected_changes[:k])**2)
        else:
            responsiveness = 0.0
        
        # Weighted combination (no drift term available for yaw)
        weights = {'smoothness': 0.5, 'responsiveness': 0.5}
        error = weights['smoothness'] * smoothness + weights['responsiveness'] * responsiveness
        return error
    
    def process_imu_data(self, accel_raw: np.ndarray, gyro_raw: np.ndarray, 
                        gyro_bias: Optional[np.ndarray] = None, accel_bias: Optional[np.ndarray] = None) -> Dict:
        """
        Process raw IMU data for optimization
        
        Args:
            accel_raw: Raw accelerometer data [N x 3] in mg
            gyro_raw: Raw gyroscope data [N x 3] in dps
            gyro_bias: Gyro bias to subtract
            accel_bias: Accel bias to subtract
            
        Returns:
            Dict with keys 'pitch', 'roll', 'yaw' containing angle and gyro arrays
        """
        # Apply bias correction if provided
        if gyro_bias is not None:
            gyro_corrected = gyro_raw - gyro_bias
        else:
            gyro_corrected = gyro_raw
            
        if accel_bias is not None:
            accel_corrected = accel_raw - accel_bias
        else:
            accel_corrected = accel_raw
        
        # Calculate tilt angles from accelerometer
        pitch_accel = self.calculate_accel_tilt(
            accel_corrected[:, 0], 
            accel_corrected[:, 1], 
            accel_corrected[:, 2], 
            'pitch'
        )
        
        roll_accel = self.calculate_accel_tilt(
            accel_corrected[:, 0], 
            accel_corrected[:, 1], 
            accel_corrected[:, 2], 
            'roll'
        )
        
        yaw_gyro = self.calculate_gyro_tilt(
            gyro_corrected[:, 0],
            gyro_corrected[:, 1],
            gyro_corrected[:, 2],
            'yaw'
        )
        
        return {
            'pitch': {'accel': pitch_accel, 'gyro': gyro_corrected[:, 1]},
            'roll': {'accel': roll_accel, 'gyro': gyro_corrected[:, 0]},
            'yaw': {'accel': yaw_gyro, 'gyro': gyro_corrected[:, 2]}
        }


def main_example():
    """Example usage of the alpha optimizer"""
    print("Complementary Filter Alpha Optimization")
    print("="*50)
    
    # Create optimizer
    optimizer = AlphaOptimizer(sample_rate=104.0)
    
    # Generate synthetic test data
    print("\nGenerating test data...")
    n_samples = 1000
    t = np.linspace(0, n_samples/104, n_samples)
    
    # True angle: slow sine wave with step change
    true_angle = 10 * np.sin(2*np.pi*0.1*t)
    true_angle[500:600] += 20  # Step change
    
    # Gyroscope: derivative of true angle + noise + bias
    gyro = np.gradient(true_angle) * 104  # Convert to deg/s
    gyro += np.random.normal(0, 0.5, n_samples)  # Noise
    gyro += 0.5  # Bias
    
    # Accelerometer: true angle + different noise
    accel = true_angle + np.random.normal(0, 2, n_samples)
    
    # Method 1: Grid search
    print("\nPerforming grid search...")
    grid_results = optimizer.grid_search(accel, gyro, reference=true_angle, n_steps=30)
    print(f"Grid search best alpha: {grid_results['best_alpha']:.4f}")
    print(f"Grid search best error: {grid_results['best_error']:.4f}")
    
    # Method 2: Optimization
    print("\nPerforming optimization...")
    opt_results = optimizer.optimize_alpha(accel, gyro, reference=true_angle)
    print(f"Optimization best alpha: {opt_results['best_alpha']:.4f}")
    print(f"Optimization best error: {opt_results['best_error']:.4f}")
    
    # Analyze characteristics
    print("\nFilter characteristics for optimal alpha:")
    char = optimizer.analyze_filter_characteristics(opt_results['best_alpha'])
    for key, value in char.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Visualize results
    optimizer.visualize_optimization(grid_results, accel, gyro, true_angle)
    
    # Save results
    with open('alpha_optimization_results.json', 'w') as f:
        save_data = {
            'best_alpha': float(opt_results['best_alpha']),
            'characteristics': char,
            'grid_search': {
                'alphas': grid_results['alphas'].tolist(),
                'errors': [float(e) for e in grid_results['errors']]
            }
        }
        json.dump(save_data, f, indent=2)
        print("\nResults saved to alpha_optimization_results.json")


def optimize_from_file(filename: str, axis: str = 'pitch'):
    """
    Optimize alpha from logged data file
    
    Args:
        filename: Path to data file with RAW_DATA lines
        axis: Which axis to optimize ('pitch', 'roll', or 'yaw')
    """
    print(f"Optimizing alpha for {axis} from {filename}")
    print("="*50)
    
    # Parse data
    accel_data: List[List[float]] = []
    gyro_data: List[List[float]] = []
    
    with open(filename, 'r') as f:
        for line in f:
            if 'RAW_DATA' in line:
                parts = line.split(',')
                if len(parts) >= 8:
                    accel_data.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    gyro_data.append([float(parts[5]), float(parts[6]), float(parts[7])])
    
    accel_data_np: np.ndarray = np.array(accel_data, dtype=float)
    gyro_data_np: np.ndarray = np.array(gyro_data, dtype=float)
    
    print(f"Loaded {len(accel_data)} samples")
    
    # Create optimizer
    optimizer = AlphaOptimizer(sample_rate=104.0)
    
    # Process data
    # Simple gyro bias removal using first N seconds if timestamps are present
    # Fallback if no timestamps in file format: use first 5 seconds worth of samples
    static_seconds = 5.0
    num_static = int(static_seconds * optimizer.sample_rate)
    if len(gyro_data_np) >= num_static:
        gyro_bias = np.mean(gyro_data_np[:num_static, :], axis=0)
    else:
        gyro_bias = np.mean(gyro_data_np, axis=0)
    gyro_corrected_np = gyro_data_np - gyro_bias
    
    processed = optimizer.process_imu_data(accel_data_np, gyro_corrected_np, gyro_bias=None, accel_bias=None)
    
    # Optimize
    axis_data = processed[axis]
    
    print(f"\nOptimizing for {axis}...")
    if axis == 'yaw':
        grid_results = optimizer.grid_search(
            None,
            axis_data['gyro'],
            n_steps=50,
            axis='yaw'
        )
        opt_results = optimizer.optimize_alpha(
            None,
            axis_data['gyro'],
            axis='yaw'
        )
    else:
        grid_results = optimizer.grid_search(
            axis_data['accel'], 
            axis_data['gyro'], 
            n_steps=50,
            axis=axis
        )
        
        opt_results = optimizer.optimize_alpha(
            axis_data['accel'], 
            axis_data['gyro'],
            axis=axis
        )
    
    print(f"\nResults for {axis}:")
    print(f"  Grid search alpha: {grid_results['best_alpha']:.4f}")
    print(f"  Optimized alpha:   {opt_results['best_alpha']:.4f}")
    
    # Analyze
    char = optimizer.analyze_filter_characteristics(opt_results['best_alpha'])
    print(f"\nCharacteristics:")
    print(f"  Time constant: {char['time_constant_s']:.3f} s")
    print(f"  Cutoff freq:   {char['cutoff_frequency_hz']:.3f} Hz")
    print(f"  Settling time: {char['settling_time_s']:.3f} s")
    
    # Visualize
    if axis == 'yaw':
        optimizer.visualize_optimization(grid_results, None, axis_data['gyro'], axis='yaw')
    else:
        optimizer.visualize_optimization(grid_results, axis_data['accel'], axis_data['gyro'], axis=axis)
    
    return opt_results['best_alpha']


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Optimize from file
        filename = sys.argv[1]
        axis = sys.argv[2] if len(sys.argv) > 2 else 'pitch'
        optimize_from_file(filename, axis)
    else:
        # Run example
        main_example()