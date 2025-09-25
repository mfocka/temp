# ISM330DHCX Data Visualization Tool

A Python-based tooling solution for real-time visualization and analysis of ISM330DHCX sensor data, inspired by MEMS Studio but with a simplified interface.

## Overview

This tool provides a comprehensive interface for:
- **Serial Communication**: Connect to ISM330DHCX via COM port with configurable baud rate
- **Real-time Data Visualization**: Live line charts for accelerometer, gyroscope, and processed data
- **Data Logging**: Save all received data to files for offline analysis
- **Motion Analysis**: Display azimuth, altitude, and zenith angles with visual indicators
- **Filter Monitoring**: Track various filter outputs (MotionDI, Simple Integration, Complementary, Fused)

## Features

### Core Functionality
- **Serial Port Selection**: Choose COM port and baud rate for sensor connection
- **Data Type Selection**: Configure which data streams to receive and display
- **Real-time Charts**: Live line charts for:
  - Raw accelerometer data (X, Y, Z axes)
  - Raw gyroscope data (X, Y, Z axes)
  - MotionDI filter results (ANGLES_DI)
  - Simple Integration filter results (ANGLES_SI)
  - Complementary filter results (ANGLES_CO)
  - Fused output results (ANGLES_FU)
  - Quaternion data (QUAT)
  - Gyroscope bias data (GYRO_BIAS_MDI)

### Data Types Supported
Based on the firmware output, the tool supports:

| Data Type | Description | Format |
|-----------|-------------|---------|
| `RAW_DATA` | Raw accelerometer and gyroscope data | `timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z` |
| `ANGLES` | Final motion detection angles | `timestamp,altitude,azimuth,zenith,state` |
| `QUAT` | Quaternion orientation data | `timestamp,qx,qy,qz,qw` |
| `GYRO_BIAS_MDI` | Gyroscope bias from MotionDI | `timestamp,bias_x,bias_y,bias_z` |
| `ANGLES_DI` | MotionDI filter output | `timestamp,pitch,yaw,roll` |
| `ANGLES_SI` | Simple Integration filter output | `timestamp,pitch,yaw,roll` |
| `ANGLES_CO` | Complementary filter output | `timestamp,pitch,yaw,roll` |
| `ANGLES_FU` | Fused filter output | `timestamp,pitch,yaw,roll` |

### Visual Components
- **Angle Displays**: Circular gauges for azimuth, altitude, and zenith angles
- **Line Charts**: Time-series plots for all sensor data, including a resulting angle plot on Filter Outputs (√(pitch²+yaw²+roll²) from Fused)
- **State Monitoring**: Real-time display of motion detection state (MONITORING, CALIBRATING, etc.)
- **Data Table**: Tabular view of current sensor values

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows/Linux/macOS

### Dependencies
```bash
pip install pyserial matplotlib numpy pandas tkinter
```

### Required Packages
```
pyserial>=3.5
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.3.0
```

## Usage

### Basic Usage
1. **Connect to Sensor**:
   - Select COM port from dropdown
   - Set baud rate (typically 115200)
   - Click "Connect" to establish serial connection

2. **Configure Data Streams**:
   - Check/uncheck desired data types in the configuration panel
   - Enable/disable specific filter outputs
   - Set data logging options

3. **Start Data Collection**:
   - Click "Start" to begin data acquisition
   - Monitor real-time charts and angle displays
   - Data is automatically saved to timestamped files

### Sensor Commands

The ISM330DHCX firmware supports various commands that can be sent via serial connection. These commands allow you to control the sensor behavior and data output.

#### Motion Detection Commands

| Command | Description | Usage | Arguments |
|---------|-------------|-------|-----------|
| `ALCAL` | Start motion detection calibration | `ALCAL` | None |
| `ALTH` | Set altitude/azimuth thresholds | `ALTH <altitude_tenths> <azimuth_tenths>` | 2 (in tenths of degrees) |
| `TTC` | Set validation time threshold | `TTC <minutes>` | 1 (in minutes) |
| `MSTATUS` | Show motion detection status | `MSTATUS` | None |

**Example Usage:**
```
ALCAL                    # Start calibration (keep device still for 60 seconds)
ALTH 50 100             # Set altitude threshold to 5.0°, azimuth to 10.0°
TTC 240                 # Set validation time to 240 minutes (4 hours)
MSTATUS                 # Check current motion detection status
```

#### Data Output Control

| Command | Description | Usage | Arguments |
|---------|-------------|-------|-----------|
| `PRINTRAW` | Control debug data output | `PRINTRAW <mask>` | 1 (bit mask) |

**Debug Output Masks:**
The `PRINTRAW` command uses a bit mask to control which data types are output:

| Bit | Mask Value | Data Type | Description |
|-----|------------|-----------|-------------|
| 0 | 0x01 (1) | RAW_DATA | Raw accelerometer and gyroscope data |
| 1 | 0x02 (2) | INFO | System information messages |
| 2 | 0x04 (4) | ANGLES | Final motion detection angles |
| 3 | 0x08 (8) | QUAT | Quaternion orientation data |
| 4 | 0x10 (16) | EVENTS | Motion detection events |
| 5 | 0x20 (32) | GYRO_BIAS_MDI | Gyroscope bias data |
| 6 | 0x40 (64) | ANGLES_* | Filter outputs (DI, SI, CO, FU) |

**Common Mask Combinations:**
```
PRINTRAW 1              # Raw data only
PRINTRAW 3              # Raw data + info (1+2)
PRINTRAW 7              # Raw data + info + angles (1+2+4)
PRINTRAW 15             # Raw data + info + angles + quat (1+2+4+8)
PRINTRAW 63             # All data types (1+2+4+8+16+32)
PRINTRAW 127            # All data types including filter outputs
```

#### System Commands

| Command | Description | Usage | Arguments |
|---------|-------------|-------|-----------|
| `RESET` | Reset the processor | `RESET` | None |
| `VERSION` | Get version information | `VERSION` | None |
| `GETF` | Get flash configuration | `GETF` | None |
| `COLLECT` | Get/Set sensor collection time | `COLLECT [seconds]` | 0-1 |
| `GETS` | Get sensor availability | `GETS` | None |
| `GETDATA` | Get sensor data | `GETDATA` | None |

#### EEPROM Commands

| Command | Description | Usage | Arguments |
|---------|-------------|-------|-----------|
| `EEWRITE` | Write to EEPROM | `EEWRITE <address> <data>` | 2 |
| `EEREAD` | Read from EEPROM | `EEREAD <address>` | 1 |
| `GETEEID` | Get EEPROM ID | `GETEEID` | None |

### Command Integration in Python Tool

The Python tooling solution should include:

1. **Command Interface**: A text input field to send commands to the sensor
2. **Command History**: Keep track of sent commands for easy re-execution
3. **Auto-Configuration**: Automatically send `PRINTRAW 127` to enable all data types
4. **Status Monitoring**: Parse command responses and update UI accordingly

**Example Python Command Interface:**
```python
def send_command(self, command):
    """Send command to sensor and return response"""
    self.serial_connection.write(f"{command}\r\n".encode())
    response = self.serial_connection.readline().decode().strip()
    return response

def configure_data_output(self, mask=127):
    """Configure sensor to output all data types"""
    response = self.send_command(f"PRINTRAW {mask}")
    return "ERROR" not in response
```

### Advanced Features

#### Data Logging
- **Automatic Logging**: All data is saved to CSV files with timestamps
- **File Naming**: `sensor_data_YYYYMMDD_HHMMSS.csv`
- **Data Format**: CSV with headers for easy import into analysis tools

#### Filter Analysis
The tool displays multiple filter outputs for comparison:
- **MotionDI (ANGLES_DI)**: ST's MotionDI library output
- **Simple Integration (ANGLES_SI)**: Basic gyroscope integration
- **Complementary Filter (ANGLES_CO)**: Accelerometer-gyroscope fusion
- **Fused Output (ANGLES_FU)**: Combined filter result

#### Motion Detection States
Monitor the system state:
- `STARTUP`: Initial system startup
- `IDLE`: Waiting for calibration
- `CALIBRATING`: Performing sensor calibration
- `MONITORING`: Active motion detection
- `VALIDATING`: Validating motion events
- `ERROR`: System error state

## Configuration

### Serial Communication
- **COM Port**: Auto-detected available ports
- **Baud Rate**: 115200 (default), 9600, 38400, 57600, 115200, 230400
- **Data Bits**: 8
- **Stop Bits**: 1
- **Parity**: None

### Data Visualization
- **Chart Update Rate**: 10 Hz (configurable); internal batching to keep UI fast
- **Data Buffer Size**: 1000 samples per chart
- **Angle Display Range**: 0-360° for azimuth, 0-180° for altitude/zenith
- **Events Tab**: Shows event raised/cleared timeline derived from ANGLES state transitions

### File Output
- **Log Directory**: `./logs/` (configurable)
- **File Format**: CSV with timestamp
- **Data Retention**: Configurable (default: 7 days)

## Architecture

### Core Components

#### Serial Interface (`serial_interface.py`)
- Handles COM port communication
- Parses incoming data streams
- Manages connection state

#### Data Parser (`data_parser.py`)
- Parses different data types from serial stream
- Validates data integrity
- Converts units and formats

#### Visualization Engine (`visualization.py`)
- Real-time chart updates
- Angle gauge displays
- Data table management

#### Data Logger (`data_logger.py`)
- File I/O operations
- Data persistence
- Log rotation

### Data Flow
```
ISM330DHCX → Serial Port → Data Parser → Visualization Engine
                                    ↓
                              Data Logger → CSV Files
```

## Example Data Format

### Raw Data Sample
```
[1773.640s] RAW_DATA,1870886700,-136.000,965.000,14.000,0.419,-0.769,-0.490
```
- Timestamp: 1870886700 μs
- Accelerometer: -136.000, 965.000, 14.000 mg
- Gyroscope: 0.419, -0.769, -0.490 dps

### Angle Data Sample
```
[1773.640s] ANGLES,1870886700,0.204,0.750,4.566,MONITORING
```
- Timestamp: 1870886700 μs
- Altitude: 0.204°
- Azimuth: 0.750°
- Zenith: 4.566°
- State: MONITORING

## Troubleshooting

### Common Issues

1. **Connection Failed**:
   - Verify COM port is correct
   - Check baud rate settings
   - Ensure device is powered and connected

2. **No Data Received**:
   - Verify sensor is outputting data
   - Check data format matches expected format
   - Verify baud rate compatibility

3. **Charts Not Updating**:
   - Check data stream configuration
   - Verify data parsing is working
   - Check for error messages in console

### Debug Mode
Enable debug logging by setting `DEBUG=True` in configuration to see detailed data flow information.

## Development

### Project Structure
```
ism330dhcx_tool/
├── main.py                 # Main application entry point
├── serial_interface.py     # Serial communication
├── data_parser.py         # Data parsing and validation
├── visualization.py       # Charts and displays
├── data_logger.py         # File I/O and logging
├── config.py              # Configuration management
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

### Adding New Data Types
1. Add parser function in `data_parser.py`
2. Update visualization in `visualization.py`
3. Add configuration option in `config.py`

## License

This project is part of the ADB Safegate motion detection system. See firmware source files for detailed licensing information.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For technical support or questions about the ISM330DHCX sensor integration, refer to the firmware documentation or contact the development team.

---

**Note**: This tool is designed to work with the specific firmware implementation shown in the provided source files. Ensure your sensor is running compatible firmware for full functionality.
