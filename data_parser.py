"""
Data Parser Module
Parses incoming data streams from ISM330DHCX sensor.
"""

import re
import time
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RawData:
    """Raw sensor data structure."""
    timestamp: float
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

@dataclass
class AngleData:
    """Angle data structure."""
    timestamp: float
    altitude: float
    azimuth: float
    zenith: float
    state: str

@dataclass
class QuaternionData:
    """Quaternion data structure."""
    timestamp: float
    qx: float
    qy: float
    qz: float
    qw: float

@dataclass
class GyroBiasData:
    """Gyroscope bias data structure."""
    timestamp: float
    bias_x: float
    bias_y: float
    bias_z: float

@dataclass
class FilterData:
    """Filter output data structure."""
    timestamp: float
    pitch: float
    yaw: float
    roll: float

class DataParser:
    """Parses data from ISM330DHCX sensor."""
    
    def __init__(self):
        """Initialize data parser."""
        self.logger = logging.getLogger(__name__)
        
        # Timestamp tracking to ensure monotonic increasing
        self.last_timestamp = 0.0
        self.timestamp_offset = 0.0
        
        # Data storage
        self.data_buffers = {
            'RAW_DATA': [],
            'ANGLES': [],
            'QUAT': [],
            'GYRO_BIAS_MDI': [],
            'ANGLES_DI': [],
            'ANGLES_SI': [],
            'ANGLES_CO': [],
            'ANGLES_FU': []
        }
        
        # Data counters
        self.data_counts = {key: 0 for key in self.data_buffers.keys()}
        self.total_data_count = 0
        
        # Compile regex patterns for better performance
        self.patterns = {
            'timestamp': re.compile(r'\[(\d+\.\d+)s\]'),
            'raw_data': re.compile(r'RAW_DATA,(\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+)'),
            'angles': re.compile(r'ANGLES,(\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),(\w+)'),
            'quat': re.compile(r'QUAT,(\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+)'),
            'gyro_bias': re.compile(r'GYRO_BIAS_MDI,(\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+)'),
            'filter_data': re.compile(r'ANGLES_(DI|SI|CO|FU),(\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+)')
        }
        
        # Maximum buffer sizes
        self.max_buffer_size = 1000
        
    def parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single line of data from the sensor.
        
        Args:
            line: Raw line from serial port
            
        Returns:
            Parsed data dictionary or None if parsing failed
        """
        if not line or not line.strip():
            return None
            
        try:
            # Extract timestamp
            timestamp_match = self.patterns['timestamp'].search(line)
            if not timestamp_match:
                return None
                
            raw_timestamp = float(timestamp_match.group(1))
            
            # Ensure monotonic increasing timestamps
            timestamp = self._ensure_monotonic_timestamp(raw_timestamp)
            
            # Parse different data types
            if 'RAW_DATA' in line:
                return self._parse_raw_data(line, timestamp)
            elif 'ANGLES,' in line and not any(f in line for f in ['DI', 'SI', 'CO', 'FU']):
                return self._parse_angles(line, timestamp)
            elif 'QUAT' in line:
                return self._parse_quaternion(line, timestamp)
            elif 'GYRO_BIAS_MDI' in line:
                return self._parse_gyro_bias(line, timestamp)
            elif any(f in line for f in ['ANGLES_DI', 'ANGLES_SI', 'ANGLES_CO', 'ANGLES_FU']):
                return self._parse_filter_data(line, timestamp)
            else:
                # Handle other data types or info messages
                return self._parse_info_message(line, timestamp)
                
        except Exception as e:
            self.logger.error(f"Error parsing line '{line}': {e}")
            return None
    
    def _parse_raw_data(self, line: str, timestamp: float) -> Optional[Dict[str, Any]]:
        """Parse raw accelerometer and gyroscope data."""
        match = self.patterns['raw_data'].search(line)
        if not match:
            return None
            
        try:
            raw_data = RawData(
                timestamp=timestamp,
                acc_x=float(match.group(2)),
                acc_y=float(match.group(3)),
                acc_z=float(match.group(4)),
                gyro_x=float(match.group(5)),
                gyro_y=float(match.group(6)),
                gyro_z=float(match.group(7))
            )
            
            self._add_to_buffer('RAW_DATA', raw_data)
            return {
                'type': 'RAW_DATA',
                'data': raw_data,
                'timestamp': timestamp
            }
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing raw data: {e}")
            return None
    
    def _parse_angles(self, line: str, timestamp: float) -> Optional[Dict[str, Any]]:
        """Parse angle data."""
        match = self.patterns['angles'].search(line)
        if not match:
            return None
            
        try:
            angle_data = AngleData(
                timestamp=timestamp,
                altitude=float(match.group(2)),
                azimuth=float(match.group(3)),
                zenith=float(match.group(4)),
                state=match.group(5)
            )
            
            self._add_to_buffer('ANGLES', angle_data)
            return {
                'type': 'ANGLES',
                'data': angle_data,
                'timestamp': timestamp
            }
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing angles: {e}")
            return None
    
    def _parse_quaternion(self, line: str, timestamp: float) -> Optional[Dict[str, Any]]:
        """Parse quaternion data."""
        match = self.patterns['quat'].search(line)
        if not match:
            return None
            
        try:
            quat_data = QuaternionData(
                timestamp=timestamp,
                qx=float(match.group(2)),
                qy=float(match.group(3)),
                qz=float(match.group(4)),
                qw=float(match.group(5))
            )
            
            self._add_to_buffer('QUAT', quat_data)
            return {
                'type': 'QUAT',
                'data': quat_data,
                'timestamp': timestamp
            }
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing quaternion: {e}")
            return None
    
    def _parse_gyro_bias(self, line: str, timestamp: float) -> Optional[Dict[str, Any]]:
        """Parse gyroscope bias data."""
        match = self.patterns['gyro_bias'].search(line)
        if not match:
            return None
            
        try:
            bias_data = GyroBiasData(
                timestamp=timestamp,
                bias_x=float(match.group(2)),
                bias_y=float(match.group(3)),
                bias_z=float(match.group(4))
            )
            
            self._add_to_buffer('GYRO_BIAS_MDI', bias_data)
            return {
                'type': 'GYRO_BIAS_MDI',
                'data': bias_data,
                'timestamp': timestamp
            }
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing gyro bias: {e}")
            return None
    
    def _parse_filter_data(self, line: str, timestamp: float) -> Optional[Dict[str, Any]]:
        """Parse filter output data."""
        match = self.patterns['filter_data'].search(line)
        if not match:
            return None
            
        try:
            filter_type = match.group(1)
            filter_data = FilterData(
                timestamp=timestamp,
                pitch=float(match.group(3)),
                yaw=float(match.group(4)),
                roll=float(match.group(5))
            )
            
            buffer_key = f'ANGLES_{filter_type}'
            self._add_to_buffer(buffer_key, filter_data)
            return {
                'type': buffer_key,
                'data': filter_data,
                'timestamp': timestamp
            }
            
        except (ValueError, IndexError) as e:
            self.logger.error(f"Error parsing filter data: {e}")
            return None
    
    def _parse_info_message(self, line: str, timestamp: float) -> Optional[Dict[str, Any]]:
        """Parse info messages and other data types."""
        # Handle various info messages
        if 'INFO' in line or 'VERSION' in line or 'MSTATUS' in line:
            return {
                'type': 'INFO',
                'message': line,
                'timestamp': timestamp
            }
        elif 'EVENTS' in line:
            return {
                'type': 'EVENTS',
                'message': line,
                'timestamp': timestamp
            }
        else:
            # Unknown data type
            self.logger.debug(f"Unknown data type: {line}")
            return {
                'type': 'UNKNOWN',
                'message': line,
                'timestamp': timestamp
            }
    
    def _add_to_buffer(self, data_type: str, data: Any):
        """Add data to appropriate buffer."""
        if data_type in self.data_buffers:
            self.data_buffers[data_type].append(data)
            self.data_counts[data_type] += 1
            self.total_data_count += 1
            
            # Maintain buffer size
            if len(self.data_buffers[data_type]) > self.max_buffer_size:
                self.data_buffers[data_type].pop(0)
    
    def get_data(self, data_type: str, count: Optional[int] = None) -> List[Any]:
        """
        Get data from a specific buffer.
        
        Args:
            data_type: Type of data to retrieve
            count: Number of recent samples to return (None for all)
            
        Returns:
            List of data objects
        """
        if data_type not in self.data_buffers:
            return []
            
        data = self.data_buffers[data_type]
        if count is None:
            return data.copy()
        else:
            return data[-count:] if count > 0 else []
    
    def get_latest_data(self, data_type: str) -> Optional[Any]:
        """
        Get the most recent data of a specific type.
        
        Args:
            data_type: Type of data to retrieve
            
        Returns:
            Most recent data object or None
        """
        data = self.get_data(data_type, 1)
        return data[0] if data else None
    
    def get_data_count(self, data_type: str) -> int:
        """Get count of data points for a specific type."""
        return self.data_counts.get(data_type, 0)
    
    def get_total_data_count(self) -> int:
        """Get total count of all data points."""
        return self.total_data_count
    
    def _ensure_monotonic_timestamp(self, timestamp: float) -> float:
        """
        Ensure timestamp is monotonically increasing.
        
        Args:
            timestamp: Raw timestamp from sensor
            
        Returns:
            Adjusted monotonic timestamp
        """
        # If timestamp goes backwards, it might be a reset or wraparound
        if timestamp < self.last_timestamp:
            # Check if it's a significant jump backwards (likely a reset)
            if self.last_timestamp - timestamp > 1000:  # More than 1000 seconds backwards
                # This is likely a sensor reset, adjust offset
                self.timestamp_offset = self.last_timestamp + 0.001
            else:
                # Small backwards jump, use last timestamp + small increment
                timestamp = self.last_timestamp + 0.001
        
        # Apply offset if needed
        adjusted_timestamp = timestamp + self.timestamp_offset
        self.last_timestamp = adjusted_timestamp
        
        return adjusted_timestamp
    
    def clear_data(self, data_type: Optional[str] = None):
        """
        Clear data buffers.
        
        Args:
            data_type: Specific data type to clear (None for all)
        """
        if data_type is None:
            for key in self.data_buffers:
                self.data_buffers[key].clear()
                self.data_counts[key] = 0
            self.total_data_count = 0
            # Reset timestamp tracking
            self.last_timestamp = 0.0
            self.timestamp_offset = 0.0
        elif data_type in self.data_buffers:
            self.data_buffers[data_type].clear()
            self.data_counts[data_type] = 0
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all data."""
        summary = {
            'total_data_points': self.total_data_count,
            'data_types': {}
        }
        
        for data_type, count in self.data_counts.items():
            summary['data_types'][data_type] = {
                'count': count,
                'buffer_size': len(self.data_buffers[data_type])
            }
            
        return summary
    
    def export_data(self, data_type: str, filename: str) -> bool:
        """
        Export data to CSV file.
        
        Args:
            data_type: Type of data to export
            filename: Output filename
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            import csv
            
            data = self.get_data(data_type)
            if not data:
                return False
                
            with open(filename, 'w', newline='') as csvfile:
                if data_type == 'RAW_DATA':
                    fieldnames = ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in data:
                        writer.writerow({
                            'timestamp': item.timestamp,
                            'acc_x': item.acc_x,
                            'acc_y': item.acc_y,
                            'acc_z': item.acc_z,
                            'gyro_x': item.gyro_x,
                            'gyro_y': item.gyro_y,
                            'gyro_z': item.gyro_z
                        })
                elif data_type == 'ANGLES':
                    fieldnames = ['timestamp', 'altitude', 'azimuth', 'zenith', 'state']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in data:
                        writer.writerow({
                            'timestamp': item.timestamp,
                            'altitude': item.altitude,
                            'azimuth': item.azimuth,
                            'zenith': item.zenith,
                            'state': item.state
                        })
                # Add other data types as needed
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return False
    
    def validate_data(self, data: Any, data_type: str) -> bool:
        """
        Validate parsed data for reasonable values.
        
        Args:
            data: Data object to validate
            data_type: Type of data
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            if data_type == 'RAW_DATA':
                # Check for reasonable accelerometer values (typically -2000 to 2000 mg)
                if not (-2000 <= data.acc_x <= 2000 and -2000 <= data.acc_y <= 2000 and -2000 <= data.acc_z <= 2000):
                    return False
                # Check for reasonable gyroscope values (typically -2000 to 2000 dps)
                if not (-2000 <= data.gyro_x <= 2000 and -2000 <= data.gyro_y <= 2000 and -2000 <= data.gyro_z <= 2000):
                    return False
                    
            elif data_type == 'ANGLES':
                # Check for reasonable angle values
                if not (0 <= data.altitude <= 180 and 0 <= data.azimuth <= 360 and 0 <= data.zenith <= 180):
                    return False
                    
            elif data_type == 'QUAT':
                # Check for valid quaternion (magnitude should be close to 1)
                magnitude = (data.qx**2 + data.qy**2 + data.qz**2 + data.qw**2)**0.5
                if not (0.9 <= magnitude <= 1.1):
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False
