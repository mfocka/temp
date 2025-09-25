"""
Command Interface Module
Handles sensor command sending and response processing.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import re

@dataclass
class Command:
    """Command structure."""
    name: str
    description: str
    usage: str
    arguments: int
    example: str

@dataclass
class CommandResponse:
    """Command response structure."""
    command: str
    response: str
    timestamp: float
    success: bool
    error_message: Optional[str] = None

class CommandInterface:
    """Handles sensor command interface."""
    
    def __init__(self):
        """Initialize command interface."""
        self.logger = logging.getLogger(__name__)
        
        # Command history
        self.command_history: List[CommandResponse] = []
        self.max_history = 1000
        
        # Response callbacks
        self.response_callbacks: List[Callable[[CommandResponse], None]] = []
        
        # Command definitions
        self.commands = self._initialize_commands()
        
        # Response patterns for parsing
        self.response_patterns = {
            'error': re.compile(r'ERROR|FAIL|INVALID', re.IGNORECASE),
            'success': re.compile(r'OK|SUCCESS|DONE', re.IGNORECASE),
            'version': re.compile(r'VERSION\s+(.+)', re.IGNORECASE),
            'status': re.compile(r'STATUS\s+(.+)', re.IGNORECASE),
            'data_mask': re.compile(r'PRINTRAW\s+(\d+)', re.IGNORECASE)
        }
        
    def _initialize_commands(self) -> Dict[str, Command]:
        """Initialize command definitions."""
        commands = {
            'PRINTRAW': Command(
                name='PRINTRAW',
                description='Control debug data output',
                usage='PRINTRAW <mask>',
                arguments=1,
                example='PRINTRAW 127'
            ),
            'ALCAL': Command(
                name='ALCAL',
                description='Start motion detection calibration',
                usage='ALCAL',
                arguments=0,
                example='ALCAL'
            ),
            'ALTH': Command(
                name='ALTH',
                description='Set altitude/azimuth thresholds',
                usage='ALTH <altitude_tenths> <azimuth_tenths>',
                arguments=2,
                example='ALTH 50 100'
            ),
            'TTC': Command(
                name='TTC',
                description='Set validation time threshold',
                usage='TTC <minutes>',
                arguments=1,
                example='TTC 240'
            ),
            'MSTATUS': Command(
                name='MSTATUS',
                description='Show motion detection status',
                usage='MSTATUS',
                arguments=0,
                example='MSTATUS'
            ),
            'RESET': Command(
                name='RESET',
                description='Reset the processor',
                usage='RESET',
                arguments=0,
                example='RESET'
            ),
            'VERSION': Command(
                name='VERSION',
                description='Get version information',
                usage='VERSION',
                arguments=0,
                example='VERSION'
            ),
            'GETF': Command(
                name='GETF',
                description='Get flash configuration',
                usage='GETF',
                arguments=0,
                example='GETF'
            ),
            'COLLECT': Command(
                name='COLLECT',
                description='Get/Set sensor collection time',
                usage='COLLECT [seconds]',
                arguments=0,
                example='COLLECT 10'
            ),
            'GETS': Command(
                name='GETS',
                description='Get sensor availability',
                usage='GETS',
                arguments=0,
                example='GETS'
            ),
            'GETDATA': Command(
                name='GETDATA',
                description='Get sensor data',
                usage='GETDATA',
                arguments=0,
                example='GETDATA'
            ),
            'EEWRITE': Command(
                name='EEWRITE',
                description='Write to EEPROM',
                usage='EEWRITE <address> <data>',
                arguments=2,
                example='EEWRITE 0x10 0x55'
            ),
            'EEREAD': Command(
                name='EEREAD',
                description='Read from EEPROM',
                usage='EEREAD <address>',
                arguments=1,
                example='EEREAD 0x10'
            ),
            'GETEEID': Command(
                name='GETEEID',
                description='Get EEPROM ID',
                usage='GETEEID',
                arguments=0,
                example='GETEEID'
            )
        }
        return commands
    
    def send_command(self, serial_interface, command: str, timeout: float = 2.0) -> CommandResponse:
        """
        Send command to sensor and return response.
        
        Args:
            serial_interface: Serial interface object
            command: Command string to send
            timeout: Response timeout in seconds
            
        Returns:
            CommandResponse object
        """
        if not serial_interface or not serial_interface.is_connected:
            return CommandResponse(
                command=command,
                response="",
                timestamp=time.time(),
                success=False,
                error_message="Not connected to sensor"
            )
        
        # Parse command
        parsed_command = self._parse_command(command)
        if not parsed_command:
            return CommandResponse(
                command=command,
                response="",
                timestamp=time.time(),
                success=False,
                error_message="Invalid command format"
            )
        
        # Validate command
        validation_error = self._validate_command(parsed_command)
        if validation_error:
            return CommandResponse(
                command=command,
                response="",
                timestamp=time.time(),
                success=False,
                error_message=validation_error
            )
        
        # Send command
        try:
            success = serial_interface.write_command(command)
            if not success:
                return CommandResponse(
                    command=command,
                    response="",
                    timestamp=time.time(),
                    success=False,
                    error_message="Failed to send command"
                )
            
            # Wait for response
            response = self._wait_for_response(serial_interface, timeout)
            
            # Parse response
            parsed_response = self._parse_response(command, response)
            
            # Create response object
            cmd_response = CommandResponse(
                command=command,
                response=response,
                timestamp=time.time(),
                success=parsed_response['success'],
                error_message=parsed_response.get('error_message')
            )
            
            # Add to history
            self._add_to_history(cmd_response)
            
            # Notify callbacks
            self._notify_callbacks(cmd_response)
            
            return cmd_response
            
        except Exception as e:
            error_response = CommandResponse(
                command=command,
                response="",
                timestamp=time.time(),
                success=False,
                error_message=f"Command execution error: {str(e)}"
            )
            self._add_to_history(error_response)
            return error_response
    
    def _parse_command(self, command: str) -> Optional[Dict[str, Any]]:
        """Parse command string into components."""
        if not command or not command.strip():
            return None
            
        parts = command.strip().split()
        if not parts:
            return None
            
        return {
            'name': parts[0].upper(),
            'args': parts[1:] if len(parts) > 1 else [],
            'raw': command.strip()
        }
    
    def _validate_command(self, parsed_command: Dict[str, Any]) -> Optional[str]:
        """Validate parsed command."""
        name = parsed_command['name']
        args = parsed_command['args']
        
        if name not in self.commands:
            return f"Unknown command: {name}"
        
        command_def = self.commands[name]
        
        # Check argument count
        if len(args) != command_def.arguments:
            if command_def.arguments == 0 and len(args) > 0:
                # Optional arguments
                pass
            elif command_def.arguments > 0 and len(args) != command_def.arguments:
                return f"Command {name} requires {command_def.arguments} arguments, got {len(args)}"
        
        # Validate specific commands
        if name == 'PRINTRAW':
            if args and not args[0].isdigit():
                return "PRINTRAW mask must be a number"
            if args and not (0 <= int(args[0]) <= 255):
                return "PRINTRAW mask must be between 0 and 255"
        
        elif name == 'ALTH':
            if len(args) >= 2:
                if not args[0].isdigit() or not args[1].isdigit():
                    return "ALTH arguments must be numbers"
                if not (0 <= int(args[0]) <= 1800) or not (0 <= int(args[1]) <= 3600):
                    return "ALTH values must be in valid range (0-1800 for altitude, 0-3600 for azimuth)"
        
        elif name == 'TTC':
            if args and not args[0].isdigit():
                return "TTC value must be a number"
            if args and not (0 <= int(args[0]) <= 1440):
                return "TTC value must be between 0 and 1440 minutes"
        
        return None
    
    def _wait_for_response(self, serial_interface, timeout: float) -> str:
        """Wait for command response."""
        start_time = time.time()
        response_lines = []
        
        while time.time() - start_time < timeout:
            line = serial_interface.read_line()
            if line:
                response_lines.append(line)
                
                # Check if we have a complete response
                if self._is_complete_response(response_lines):
                    break
        
        return '\n'.join(response_lines) if response_lines else ""
    
    def _is_complete_response(self, response_lines: List[str]) -> bool:
        """Check if response is complete."""
        if not response_lines:
            return False
        
        last_line = response_lines[-1].strip()
        
        # Check for common response terminators
        if any(term in last_line.upper() for term in ['OK', 'ERROR', 'DONE', 'READY']):
            return True
        
        # Check for data output patterns
        if any(pattern in last_line for pattern in ['RAW_DATA', 'ANGLES', 'QUAT', 'VERSION']):
            return True
        
        return False
    
    def _parse_response(self, command: str, response: str) -> Dict[str, Any]:
        """Parse command response."""
        if not response:
            return {'success': False, 'error_message': 'No response received'}
        
        # Check for error patterns
        if self.response_patterns['error'].search(response):
            return {'success': False, 'error_message': 'Command returned error'}
        
        # Check for success patterns
        if self.response_patterns['success'].search(response):
            return {'success': True}
        
        # Check for specific command responses
        if command.upper() == 'VERSION':
            version_match = self.response_patterns['version'].search(response)
            if version_match:
                return {'success': True, 'data': version_match.group(1)}
        
        if command.upper() == 'MSTATUS':
            status_match = self.response_patterns['status'].search(response)
            if status_match:
                return {'success': True, 'data': status_match.group(1)}
        
        if command.upper().startswith('PRINTRAW'):
            mask_match = self.response_patterns['data_mask'].search(response)
            if mask_match:
                return {'success': True, 'data': mask_match.group(1)}
        
        # If we got any response, consider it successful
        return {'success': True, 'data': response}
    
    def _add_to_history(self, response: CommandResponse):
        """Add response to command history."""
        self.command_history.append(response)
        
        # Maintain history size
        if len(self.command_history) > self.max_history:
            self.command_history.pop(0)
    
    def _notify_callbacks(self, response: CommandResponse):
        """Notify registered callbacks of command response."""
        for callback in self.response_callbacks:
            try:
                callback(response)
            except Exception as e:
                self.logger.error(f"Error in response callback: {e}")
    
    def add_response_callback(self, callback: Callable[[CommandResponse], None]):
        """Add response callback function."""
        self.response_callbacks.append(callback)
    
    def remove_response_callback(self, callback: Callable[[CommandResponse], None]):
        """Remove response callback function."""
        if callback in self.response_callbacks:
            self.response_callbacks.remove(callback)
    
    def get_command_history(self, limit: Optional[int] = None) -> List[CommandResponse]:
        """Get command history."""
        if limit is None:
            return self.command_history.copy()
        return self.command_history[-limit:] if limit > 0 else []
    
    def get_command_help(self, command_name: Optional[str] = None) -> Dict[str, Any]:
        """Get command help information."""
        if command_name:
            command_name = command_name.upper()
            if command_name in self.commands:
                return {
                    'command': command_name,
                    'info': self.commands[command_name]
                }
            return {}
        
        return {name: cmd for name, cmd in self.commands.items()}
    
    def get_data_mask_help(self) -> Dict[str, Any]:
        """Get help for PRINTRAW data mask."""
        return {
            'description': 'PRINTRAW command controls which data types are output',
            'bits': {
                0: {'mask': 0x01, 'name': 'RAW_DATA', 'description': 'Raw accelerometer and gyroscope data'},
                1: {'mask': 0x02, 'name': 'INFO', 'description': 'System information messages'},
                2: {'mask': 0x04, 'name': 'ANGLES', 'description': 'Final motion detection angles'},
                3: {'mask': 0x08, 'name': 'QUAT', 'description': 'Quaternion orientation data'},
                4: {'mask': 0x10, 'name': 'EVENTS', 'description': 'Motion detection events'},
                5: {'mask': 0x20, 'name': 'GYRO_BIAS_MDI', 'description': 'Gyroscope bias data'},
                6: {'mask': 0x40, 'name': 'ANGLES_*', 'description': 'Filter outputs (DI, SI, CO, FU)'}
            },
            'common_masks': {
                '1': 'Raw data only',
                '3': 'Raw data + info',
                '7': 'Raw data + info + angles',
                '15': 'Raw data + info + angles + quat',
                '63': 'All data types except filters',
                '127': 'All data types including filters'
            }
        }
    
    def create_data_mask(self, enabled_types: List[str]) -> int:
        """Create data mask from enabled data types."""
        mask = 0
        
        type_masks = {
            'RAW_DATA': 0x01,
            'INFO': 0x02,
            'ANGLES': 0x04,
            'QUAT': 0x08,
            'EVENTS': 0x10,
            'GYRO_BIAS_MDI': 0x20,
            'FILTERS': 0x40  # Covers all ANGLES_* types
        }
        
        for data_type in enabled_types:
            if data_type in type_masks:
                mask |= type_masks[data_type]
        
        return mask
    
    def parse_data_mask(self, mask: int) -> List[str]:
        """Parse data mask into enabled data types."""
        enabled_types = []
        
        type_masks = {
            'RAW_DATA': 0x01,
            'INFO': 0x02,
            'ANGLES': 0x04,
            'QUAT': 0x08,
            'EVENTS': 0x10,
            'GYRO_BIAS_MDI': 0x20,
            'FILTERS': 0x40
        }
        
        for data_type, bit_mask in type_masks.items():
            if mask & bit_mask:
                enabled_types.append(data_type)
        
        return enabled_types
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get command interface statistics."""
        total_commands = len(self.command_history)
        successful_commands = sum(1 for cmd in self.command_history if cmd.success)
        failed_commands = total_commands - successful_commands
        
        return {
            'total_commands': total_commands,
            'successful_commands': successful_commands,
            'failed_commands': failed_commands,
            'success_rate': successful_commands / total_commands if total_commands > 0 else 0,
            'available_commands': len(self.commands),
            'response_callbacks': len(self.response_callbacks)
        }
    
    def clear_history(self):
        """Clear command history."""
        self.command_history.clear()
        self.logger.info("Command history cleared")
    
    def export_history(self, filename: str) -> bool:
        """Export command history to file."""
        try:
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['timestamp', 'command', 'response', 'success', 'error_message']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for cmd in self.command_history:
                    writer.writerow({
                        'timestamp': datetime.fromtimestamp(cmd.timestamp).isoformat(),
                        'command': cmd.command,
                        'response': cmd.response,
                        'success': cmd.success,
                        'error_message': cmd.error_message or ''
                    })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting command history: {e}")
            return False
