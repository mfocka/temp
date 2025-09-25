"""
Utility Functions Module
Common utility functions for the ISM330DHCX tool.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import time
from datetime import datetime, timedelta
import json

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            # Create log directory if it doesn't exist
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    return root_logger

def format_timestamp(timestamp: float, format_str: str = "%H:%M:%S") -> str:
    """
    Format timestamp to string.
    
    Args:
        timestamp: Unix timestamp
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime(format_str)
    except (ValueError, OSError):
        return "Invalid timestamp"

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.1f}s"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert value to int.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Int value or default
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))

def normalize_angle(angle: float, min_angle: float = 0, max_angle: float = 360) -> float:
    """
    Normalize angle to specified range.
    
    Args:
        angle: Angle to normalize
        min_angle: Minimum angle
        max_angle: Maximum angle
        
    Returns:
        Normalized angle
    """
    range_size = max_angle - min_angle
    normalized = ((angle - min_angle) % range_size) + min_angle
    return normalized

def calculate_magnitude(x: float, y: float, z: float) -> float:
    """
    Calculate 3D vector magnitude.
    
    Args:
        x: X component
        y: Y component
        z: Z component
        
    Returns:
        Vector magnitude
    """
    return (x**2 + y**2 + z**2)**0.5

def calculate_angle_between_vectors(v1: List[float], v2: List[float]) -> float:
    """
    Calculate angle between two 3D vectors.
    
    Args:
        v1: First vector [x, y, z]
        v2: Second vector [x, y, z]
        
    Returns:
        Angle in degrees
    """
    import math
    
    if len(v1) != 3 or len(v2) != 3:
        return 0.0
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Calculate magnitudes
    mag1 = calculate_magnitude(v1[0], v1[1], v1[2])
    mag2 = calculate_magnitude(v2[0], v2[1], v2[2])
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    # Calculate angle
    cos_angle = dot_product / (mag1 * mag2)
    cos_angle = clamp(cos_angle, -1.0, 1.0)  # Clamp to avoid numerical errors
    
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg

def quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> tuple:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        qx, qy, qz, qw: Quaternion components
        
    Returns:
        Tuple of (roll, pitch, yaw) in degrees
    """
    import math
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    # Convert to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    return (roll_deg, pitch_deg, yaw_deg)

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple:
    """
    Convert Euler angles to quaternion.
    
    Args:
        roll, pitch, yaw: Euler angles in degrees
        
    Returns:
        Tuple of (qx, qy, qz, qw)
    """
    import math
    
    # Convert to radians
    roll_rad = math.radians(roll)
    pitch_rad = math.radians(pitch)
    yaw_rad = math.radians(yaw)
    
    # Calculate quaternion components
    cy = math.cos(yaw_rad * 0.5)
    sy = math.sin(yaw_rad * 0.5)
    cp = math.cos(pitch_rad * 0.5)
    sp = math.sin(pitch_rad * 0.5)
    cr = math.cos(roll_rad * 0.5)
    sr = math.sin(roll_rad * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return (qx, qy, qz, qw)

def validate_data_range(value: float, min_val: float, max_val: float, 
                       name: str = "value") -> bool:
    """
    Validate that value is within specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages
        
    Returns:
        True if valid, False otherwise
    """
    if not (min_val <= value <= max_val):
        print(f"Warning: {name} ({value}) is outside valid range [{min_val}, {max_val}]")
        return False
    return True

def create_directory(path: Union[str, Path]) -> bool:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        return False

def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes, 0 if file doesn't exist
    """
    try:
        return Path(file_path).stat().st_size
    except (OSError, FileNotFoundError):
        return 0

def get_file_age(file_path: Union[str, Path]) -> float:
    """
    Get file age in seconds.
    
    Args:
        file_path: Path to file
        
    Returns:
        File age in seconds, 0 if file doesn't exist
    """
    try:
        stat = Path(file_path).stat()
        return time.time() - stat.st_mtime
    except (OSError, FileNotFoundError):
        return 0

def cleanup_old_files(directory: Union[str, Path], pattern: str = "*", 
                     max_age_days: int = 7) -> int:
    """
    Clean up old files in directory.
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        max_age_days: Maximum age in days
        
    Returns:
        Number of files removed
    """
    try:
        from pathlib import Path
        import time
        
        directory = Path(directory)
        if not directory.exists():
            return 0
        
        max_age_seconds = max_age_days * 24 * 3600
        current_time = time.time()
        removed_count = 0
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    file_path.unlink()
                    removed_count += 1
        
        return removed_count
        
    except Exception as e:
        print(f"Error cleaning up files: {e}")
        return 0

def load_json_file(file_path: Union[str, Path], default: Any = None) -> Any:
    """
    Load JSON file safely.
    
    Args:
        file_path: Path to JSON file
        default: Default value if file doesn't exist or is invalid
        
    Returns:
        Loaded data or default value
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default

def save_json_file(file_path: Union[str, Path], data: Any) -> bool:
    """
    Save data to JSON file safely.
    
    Args:
        file_path: Path to JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except (OSError, TypeError, ValueError):
        return False

def get_system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    return {
        'platform': platform.platform(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': sys.version,
        'python_executable': sys.executable
    }

def format_error_message(error: Exception, context: str = "") -> str:
    """
    Format error message with context.
    
    Args:
        error: Exception object
        context: Additional context string
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    if context:
        return f"{context}: {error_type}: {error_msg}"
    else:
        return f"{error_type}: {error_msg}"

def retry_on_exception(func, max_retries: int = 3, delay: float = 1.0, 
                      exceptions: tuple = (Exception,)) -> Any:
    """
    Retry function on exception.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Function result or raises last exception
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except exceptions as e:
            if attempt == max_retries:
                raise e
            time.sleep(delay)
    
    return None

def measure_execution_time(func):
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to measure
        
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.3f} seconds")
        return result
    return wrapper

def validate_config_file(config_path: Union[str, Path]) -> List[str]:
    """
    Validate configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        List of validation errors
    """
    errors = []
    
    try:
        config_data = load_json_file(config_path)
        if config_data is None:
            errors.append("Configuration file is empty or invalid JSON")
            return errors
        
        # Validate required sections
        required_sections = ['serial', 'data', 'visualization', 'filter', 'sensor', 'ui']
        for section in required_sections:
            if section not in config_data:
                errors.append(f"Missing required section: {section}")
        
        # Validate data types
        if 'serial' in config_data:
            serial_config = config_data['serial']
            if 'default_baud_rate' in serial_config:
                baud_rate = serial_config['default_baud_rate']
                if not isinstance(baud_rate, int) or baud_rate not in [9600, 38400, 57600, 115200, 230400]:
                    errors.append("Invalid baud rate in serial configuration")
        
        # Add more validation as needed
        
    except Exception as e:
        errors.append(f"Error validating configuration: {e}")
    
    return errors
