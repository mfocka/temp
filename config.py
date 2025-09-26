"""
Configuration Management Module
Handles application configuration and settings persistence.
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class SerialConfig:
    """Serial communication configuration."""
    default_port: str = ""
    default_baud_rate: int = 115200
    timeout: float = 0.001  # Optimized for 52Hz
    auto_connect: bool = False
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0

@dataclass
class DataConfig:
    """Data processing configuration."""
    max_buffer_size: int = 500  # Optimized for 52Hz
    update_rate_hz: int = 52  # Target 52Hz rate
    enable_logging: bool = True
    log_directory: str = "./logs"
    max_file_size_mb: int = 10
    retention_days: int = 7
    auto_export: bool = False
    export_interval_minutes: int = 60

@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    chart_theme: str = "seaborn-v0_8"
    chart_dpi: int = 100
    chart_size: tuple = (12, 8)
    line_width: float = 1.0
    grid_alpha: float = 0.3
    auto_scale: bool = True
    show_legend: bool = True
    gauge_size: int = 150
    gauge_colors: Optional[List[str]] = None

@dataclass
class FilterConfig:
    """Filter configuration."""
    enable_motion_di: bool = True
    enable_simple_integration: bool = True
    enable_complementary: bool = True
    enable_fused: bool = True
    motion_di_alpha: float = 0.98
    complementary_alpha: float = 0.98
    fusion_weight: float = 0.5

@dataclass
class SensorConfig:
    """Sensor-specific configuration."""
    data_mask: int = 127  # All data types enabled
    calibration_time: int = 60  # seconds
    altitude_threshold: int = 50  # tenths of degrees
    azimuth_threshold: int = 100  # tenths of degrees
    validation_time: int = 240  # minutes
    collection_time: int = 10  # seconds

@dataclass
class UIConfig:
    """User interface configuration."""
    window_width: int = 1400
    window_height: int = 900
    sidebar_width: int = 300
    theme: str = "default"
    font_family: str = "Arial"
    font_size: int = 10
    show_toolbar: bool = True
    show_status_bar: bool = True
    remember_window_state: bool = True

class Config:
    """Main configuration manager."""
    
    def __init__(self, config_file: str = "config.json"):
        """Initialize configuration manager."""
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize default configurations
        self.serial = SerialConfig()
        self.data = DataConfig()
        self.visualization = VisualizationConfig()
        self.filter = FilterConfig()
        self.sensor = SensorConfig()
        self.ui = UIConfig()
        
        # Set default gauge colors
        if self.visualization.gauge_colors is None:
            self.visualization.gauge_colors = ['red', 'green', 'blue']
        
        # Load configuration
        self.load_config()
        
    def load_config(self) -> bool:
        """Load configuration from file."""
        if not self.config_file.exists():
            self.logger.info("Configuration file not found, using defaults")
            return False
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'serial' in config_data:
                self.serial = SerialConfig(**config_data['serial'])
            
            if 'data' in config_data:
                self.data = DataConfig(**config_data['data'])
            
            if 'visualization' in config_data:
                self.visualization = VisualizationConfig(**config_data['visualization'])
            
            if 'filter' in config_data:
                self.filter = FilterConfig(**config_data['filter'])
            
            if 'sensor' in config_data:
                self.sensor = SensorConfig(**config_data['sensor'])
            
            if 'ui' in config_data:
                self.ui = UIConfig(**config_data['ui'])
            
            self.logger.info(f"Configuration loaded from {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            config_data = {
                'serial': asdict(self.serial),
                'data': asdict(self.data),
                'visualization': asdict(self.visualization),
                'filter': asdict(self.filter),
                'sensor': asdict(self.sensor),
                'ui': asdict(self.ui),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset all configurations to defaults."""
        self.serial = SerialConfig()
        self.data = DataConfig()
        self.visualization = VisualizationConfig()
        self.filter = FilterConfig()
        self.sensor = SensorConfig()
        self.ui = UIConfig()
        
        # Set default gauge colors
        self.visualization.gauge_colors = ['red', 'green', 'blue']
        
        self.logger.info("Configuration reset to defaults")
    
    def get_serial_config(self) -> SerialConfig:
        """Get serial configuration."""
        return self.serial
    
    def get_data_config(self) -> DataConfig:
        """Get data configuration."""
        return self.data
    
    def get_visualization_config(self) -> VisualizationConfig:
        """Get visualization configuration."""
        return self.visualization
    
    def get_filter_config(self) -> FilterConfig:
        """Get filter configuration."""
        return self.filter
    
    def get_sensor_config(self) -> SensorConfig:
        """Get sensor configuration."""
        return self.sensor
    
    def get_ui_config(self) -> UIConfig:
        """Get UI configuration."""
        return self.ui
    
    def update_serial_config(self, **kwargs):
        """Update serial configuration."""
        for key, value in kwargs.items():
            if hasattr(self.serial, key):
                setattr(self.serial, key, value)
        self.logger.debug("Serial configuration updated")
    
    def update_data_config(self, **kwargs):
        """Update data configuration."""
        for key, value in kwargs.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)
        self.logger.debug("Data configuration updated")
    
    def update_visualization_config(self, **kwargs):
        """Update visualization configuration."""
        for key, value in kwargs.items():
            if hasattr(self.visualization, key):
                setattr(self.visualization, key, value)
        self.logger.debug("Visualization configuration updated")
    
    def update_filter_config(self, **kwargs):
        """Update filter configuration."""
        for key, value in kwargs.items():
            if hasattr(self.filter, key):
                setattr(self.filter, key, value)
        self.logger.debug("Filter configuration updated")
    
    def update_sensor_config(self, **kwargs):
        """Update sensor configuration."""
        for key, value in kwargs.items():
            if hasattr(self.sensor, key):
                setattr(self.sensor, key, value)
        self.logger.debug("Sensor configuration updated")
    
    def update_ui_config(self, **kwargs):
        """Update UI configuration."""
        for key, value in kwargs.items():
            if hasattr(self.ui, key):
                setattr(self.ui, key, value)
        self.logger.debug("UI configuration updated")
    
    def get_data_mask(self) -> int:
        """Get current data mask based on enabled filters."""
        mask = 0
        
        # Always enable basic data types
        mask |= 0x01  # RAW_DATA
        mask |= 0x02  # INFO
        mask |= 0x04  # ANGLES
        mask |= 0x08  # QUAT
        mask |= 0x10  # EVENTS
        mask |= 0x20  # GYRO_BIAS_MDI
        
        # Enable filters if configured
        if any([self.filter.enable_motion_di, self.filter.enable_simple_integration,
                self.filter.enable_complementary, self.filter.enable_fused]):
            mask |= 0x40  # ANGLES_*
        
        return mask
    
    def get_enabled_data_types(self) -> List[str]:
        """Get list of enabled data types."""
        enabled_types = ['RAW_DATA', 'INFO', 'ANGLES', 'QUAT', 'EVENTS', 'GYRO_BIAS_MDI']
        
        if any([self.filter.enable_motion_di, self.filter.enable_simple_integration,
                self.filter.enable_complementary, self.filter.enable_fused]):
            enabled_types.append('FILTERS')
        
        return enabled_types
    
    def get_available_ports(self) -> List[str]:
        """Get list of available COM ports."""
        try:
            import serial.tools.list_ports
            ports = []
            for port in serial.tools.list_ports.comports():
                ports.append(port.device)
            return sorted(ports)
        except ImportError:
            return []
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return any errors."""
        errors = []
        
        # Validate serial configuration
        if self.serial.default_baud_rate not in [9600, 38400, 57600, 115200, 230400]:
            errors.append("Invalid baud rate")
        
        if self.serial.timeout <= 0:
            errors.append("Serial timeout must be positive")
        
        # Validate data configuration
        if self.data.max_buffer_size <= 0:
            errors.append("Buffer size must be positive")
        
        if self.data.update_rate_hz <= 0:
            errors.append("Update rate must be positive")
        
        if self.data.max_file_size_mb <= 0:
            errors.append("Max file size must be positive")
        
        if self.data.retention_days < 0:
            errors.append("Retention days cannot be negative")
        
        # Validate visualization configuration
        if self.visualization.chart_dpi <= 0:
            errors.append("Chart DPI must be positive")
        
        if self.visualization.line_width <= 0:
            errors.append("Line width must be positive")
        
        if not (0 <= self.visualization.grid_alpha <= 1):
            errors.append("Grid alpha must be between 0 and 1")
        
        # Validate sensor configuration
        if not (0 <= self.sensor.data_mask <= 255):
            errors.append("Data mask must be between 0 and 255")
        
        if self.sensor.calibration_time <= 0:
            errors.append("Calibration time must be positive")
        
        if not (0 <= self.sensor.altitude_threshold <= 1800):
            errors.append("Altitude threshold must be between 0 and 1800")
        
        if not (0 <= self.sensor.azimuth_threshold <= 3600):
            errors.append("Azimuth threshold must be between 0 and 3600")
        
        # Validate UI configuration
        if self.ui.window_width <= 0 or self.ui.window_height <= 0:
            errors.append("Window dimensions must be positive")
        
        if self.ui.sidebar_width <= 0:
            errors.append("Sidebar width must be positive")
        
        if self.ui.font_size <= 0:
            errors.append("Font size must be positive")
        
        return errors
    
    def export_config(self, filename: str) -> bool:
        """Export configuration to file."""
        try:
            config_data = {
                'serial': asdict(self.serial),
                'data': asdict(self.data),
                'visualization': asdict(self.visualization),
                'filter': asdict(self.filter),
                'sensor': asdict(self.sensor),
                'ui': asdict(self.ui),
                'exported_at': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration exported to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, filename: str) -> bool:
        """Import configuration from file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'serial' in config_data:
                self.serial = SerialConfig(**config_data['serial'])
            
            if 'data' in config_data:
                self.data = DataConfig(**config_data['data'])
            
            if 'visualization' in config_data:
                self.visualization = VisualizationConfig(**config_data['visualization'])
            
            if 'filter' in config_data:
                self.filter = FilterConfig(**config_data['filter'])
            
            if 'sensor' in config_data:
                self.sensor = SensorConfig(**config_data['sensor'])
            
            if 'ui' in config_data:
                self.ui = UIConfig(**config_data['ui'])
            
            self.logger.info(f"Configuration imported from {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'serial': {
                'port': self.serial.default_port,
                'baud_rate': self.serial.default_baud_rate,
                'auto_connect': self.serial.auto_connect
            },
            'data': {
                'logging_enabled': self.data.enable_logging,
                'log_directory': self.data.log_directory,
                'update_rate': self.data.update_rate_hz,
                'buffer_size': self.data.max_buffer_size
            },
            'visualization': {
                'theme': self.visualization.chart_theme,
                'dpi': self.visualization.chart_dpi,
                'auto_scale': self.visualization.auto_scale
            },
            'sensor': {
                'data_mask': self.sensor.data_mask,
                'calibration_time': self.sensor.calibration_time,
                'thresholds': {
                    'altitude': self.sensor.altitude_threshold,
                    'azimuth': self.sensor.azimuth_threshold
                }
            },
            'ui': {
                'window_size': (self.ui.window_width, self.ui.window_height),
                'theme': self.ui.theme,
                'font_size': self.ui.font_size
            }
        }
    
    def __del__(self):
        """Save configuration on destruction."""
        try:
            self.save_config()
        except:
            pass
