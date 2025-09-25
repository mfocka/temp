"""
Serial Interface Module
Handles COM port communication with ISM330DHCX sensor.
"""

import serial
import serial.tools.list_ports
import threading
import time
import logging
from typing import Optional, List, Callable

class SerialInterface:
    """Handles serial communication with ISM330DHCX sensor."""
    
    def __init__(self):
        """Initialize serial interface."""
        self.serial_connection: Optional[serial.Serial] = None
        self.is_connected = False
        self.port = None
        self.baud_rate = 115200
        self.timeout = 1.0
        self.logger = logging.getLogger(__name__)
        
        # Data callback for real-time processing
        self.data_callback: Optional[Callable[[str], None]] = None
        
    def get_available_ports(self) -> List[str]:
        """Get list of available COM ports."""
        ports = []
        try:
            available_ports = serial.tools.list_ports.comports()
            for port in available_ports:
                ports.append(port.device)
            self.logger.info(f"Found {len(ports)} available ports: {ports}")
        except Exception as e:
            self.logger.error(f"Error getting available ports: {e}")
            
        return sorted(ports)
    
    def connect(self, port: str, baud_rate: int = 115200) -> bool:
        """
        Connect to serial port.
        
        Args:
            port: COM port name (e.g., 'COM3' or '/dev/ttyUSB0')
            baud_rate: Baud rate for communication
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.is_connected:
                self.disconnect()
                
            self.serial_connection = serial.Serial(
                port=port,
                baudrate=baud_rate,
            )
            
            print(self.serial_connection)
            print(self.serial_connection.is_open)
            
            # Wait for connection to stabilize
            time.sleep(0.5)
            
            # Test connection by sending a simple command
            if self.test_connection():
                self.is_connected = True
                self.port = port
                self.baud_rate = baud_rate
                self.logger.info(f"Connected to {port} at {baud_rate} baud")
                return True
            else:
                self.disconnect()
                return False
                
        except serial.SerialException as e:
            self.logger.error(f"Serial connection error: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from serial port."""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.close()
                self.logger.info("Disconnected from serial port")
            except Exception as e:
                self.logger.error(f"Error during disconnect: {e}")
        
        self.serial_connection = None
        self.is_connected = False
        self.port = None
    
    def test_connection(self) -> bool:
        """Test if connection is working by sending a simple command."""
        try:
            if not self.serial_connection or not self.serial_connection.is_open:
                print("Not connected")
                return False
            self.is_connected = True
            # Send VERSION command to test connection
            self.write_command("VERSION")
            # Try to read response
            response = self.read_line()
            print(response)
            return response is not None and len(response) > 0
            
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def write_command(self, command: str) -> bool:
        """
        Write command to serial port.
        
        Args:
            command: Command string to send
            
        Returns:
            True if command sent successfully, False otherwise
        """
        if not self.is_connected or not self.serial_connection:
            self.logger.warning("Not connected to serial port")
            return False
            
        try:
            # Add carriage return and line feed
            command_with_crlf = command + '\r\n'
            self.serial_connection.write(command_with_crlf.encode('utf-8'))
            self.serial_connection.flush()
            self.logger.debug(f"Sent command: {command}")
            return True
            
        except serial.SerialTimeoutException:
            self.logger.error("Serial write timeout")
            return False
        except Exception as e:
            self.logger.error(f"Error writing command: {e}")
            return False
    
    def read_line(self) -> Optional[str]:
        """
        Read a line from serial port.
        
        Returns:
            Decoded line string or None if no data available
        """
        if not self.is_connected or not self.serial_connection:
            return None
            
        try:
            # Read line with timeout
            line = self.serial_connection.readline()
            print(line)
            if line:
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                if decoded_line:
                    self.logger.debug(f"Received: {decoded_line}")
                    return decoded_line
            return None
            
        except serial.SerialTimeoutException:
            # Timeout is normal when no data is available
            return None
        except Exception as e:
            self.logger.error(f"Error reading from serial port: {e}")
            return None
    
    def read_all_available(self) -> List[str]:
        """
        Read all available lines from serial port.
        
        Returns:
            List of decoded line strings
        """
        lines = []
        if not self.is_connected or not self.serial_connection:
            return lines
            
        try:
            while self.serial_connection.in_waiting > 0:
                line = self.read_line()
                if line:
                    lines.append(line)
                else:
                    break
        except Exception as e:
            self.logger.error(f"Error reading all available data: {e}")
            
        return lines
    
    def set_data_callback(self, callback: Callable[[str], None]):
        """
        Set callback function for real-time data processing.
        
        Args:
            callback: Function to call with each received line
        """
        self.data_callback = callback
    
    def start_data_monitoring(self):
        """Start background thread for data monitoring."""
        if not self.is_connected:
            return
            
        def monitor_data():
            while self.is_connected and self.serial_connection:
                try:
                    line = self.read_line()
                    if line and self.data_callback:
                        self.data_callback(line)
                except Exception as e:
                    self.logger.error(f"Data monitoring error: {e}")
                    break
                time.sleep(0.001)  # 1ms delay to prevent excessive CPU usage
                
        monitor_thread = threading.Thread(target=monitor_data, daemon=True)
        monitor_thread.start()
    
    def get_connection_info(self) -> dict:
        """
        Get current connection information.
        
        Returns:
            Dictionary with connection details
        """
        return {
            'connected': self.is_connected,
            'port': self.port,
            'baud_rate': self.baud_rate,
            'timeout': self.timeout
        }
    
    def is_port_available(self, port: str) -> bool:
        """
        Check if a specific port is available.
        
        Args:
            port: Port name to check
            
        Returns:
            True if port is available, False otherwise
        """
        try:
            test_serial = serial.Serial(port, timeout=0.1)
            test_serial.close()
            return True
        except:
            return False
    
    def get_port_info(self, port: str) -> dict:
        """
        Get information about a specific port.
        
        Args:
            port: Port name to get info for
            
        Returns:
            Dictionary with port information
        """
        try:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                if p.device == port:
                    return {
                        'device': p.device,
                        'description': p.description,
                        'manufacturer': p.manufacturer,
                        'product': p.product,
                        'serial_number': p.serial_number,
                        'vid': p.vid,
                        'pid': p.pid
                    }
        except Exception as e:
            self.logger.error(f"Error getting port info: {e}")
            
        return {}
    
    def set_timeout(self, timeout: float):
        """
        Set read timeout for serial connection.
        
        Args:
            timeout: Timeout in seconds
        """
        self.timeout = timeout
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.timeout = timeout
    
    def flush_buffers(self):
        """Flush input and output buffers."""
        if self.serial_connection and self.serial_connection.is_open:
            try:
                self.serial_connection.flushInput()
                self.serial_connection.flushOutput()
            except Exception as e:
                self.logger.error(f"Error flushing buffers: {e}")
    
    def get_buffer_info(self) -> dict:
        """
        Get information about serial buffers.
        
        Returns:
            Dictionary with buffer information
        """
        if not self.serial_connection or not self.serial_connection.is_open:
            return {'in_waiting': 0, 'out_waiting': 0}
            
        try:
            return {
                'in_waiting': self.serial_connection.in_waiting,
                'out_waiting': self.serial_connection.out_waiting
            }
        except Exception as e:
            self.logger.error(f"Error getting buffer info: {e}")
            return {'in_waiting': 0, 'out_waiting': 0}
