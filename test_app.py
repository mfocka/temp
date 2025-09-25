#!/usr/bin/env python3
"""
Test script for ISM330DHCX Data Visualization Tool
"""

import sys
import os
import time
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from serial_interface import SerialInterface
        print("✓ SerialInterface imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SerialInterface: {e}")
        return False
    
    try:
        from data_parser import DataParser
        print("✓ DataParser imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DataParser: {e}")
        return False
    
    try:
        from visualization import VisualizationEngine
        print("✓ VisualizationEngine imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import VisualizationEngine: {e}")
        return False
    
    try:
        from data_logger import DataLogger
        print("✓ DataLogger imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DataLogger: {e}")
        return False
    
    try:
        from command_interface import CommandInterface
        print("✓ CommandInterface imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CommandInterface: {e}")
        return False
    
    try:
        from config import Config
        print("✓ Config imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Config: {e}")
        return False
    
    try:
        from utils import setup_logging
        print("✓ Utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Utils: {e}")
        return False
    
    return True

def test_serial_interface():
    """Test serial interface functionality."""
    print("\nTesting SerialInterface...")
    
    try:
        from serial_interface import SerialInterface
        
        # Create interface
        serial = SerialInterface()
        print("✓ SerialInterface created successfully")
        
        # Test port detection
        ports = serial.get_available_ports()
        print(f"✓ Found {len(ports)} available ports: {ports}")
        
        # Test connection info
        info = serial.get_connection_info()
        print(f"✓ Connection info: {info}")
        
        return True
        
    except Exception as e:
        print(f"✗ SerialInterface test failed: {e}")
        return False

def test_data_parser():
    """Test data parser functionality."""
    print("\nTesting DataParser...")
    
    try:
        from data_parser import DataParser
        
        # Create parser
        parser = DataParser()
        print("✓ DataParser created successfully")
        
        # Test parsing raw data
        test_line = "[1773.640s] RAW_DATA,1870886700,-136.000,965.000,14.000,0.419,-0.769,-0.490"
        parsed = parser.parse_line(test_line)
        
        if parsed and parsed['type'] == 'RAW_DATA':
            print("✓ Raw data parsing successful")
        else:
            print("✗ Raw data parsing failed")
            return False
        
        # Test parsing angle data
        test_line = "[1773.640s] ANGLES,1870886700,0.204,0.750,4.566,MONITORING"
        parsed = parser.parse_line(test_line)
        
        if parsed and parsed['type'] == 'ANGLES':
            print("✓ Angle data parsing successful")
        else:
            print("✗ Angle data parsing failed")
            return False
        
        # Test data summary
        summary = parser.get_data_summary()
        print(f"✓ Data summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"✗ DataParser test failed: {e}")
        return False

def test_command_interface():
    """Test command interface functionality."""
    print("\nTesting CommandInterface...")
    
    try:
        from command_interface import CommandInterface
        
        # Create interface
        cmd_interface = CommandInterface()
        print("✓ CommandInterface created successfully")
        
        # Test command parsing
        parsed = cmd_interface._parse_command("PRINTRAW 127")
        if parsed and parsed['name'] == 'PRINTRAW':
            print("✓ Command parsing successful")
        else:
            print("✗ Command parsing failed")
            return False
        
        # Test command validation
        error = cmd_interface._validate_command(parsed)
        if error is None:
            print("✓ Command validation successful")
        else:
            print(f"✗ Command validation failed: {error}")
            return False
        
        # Test data mask creation
        mask = cmd_interface.create_data_mask(['RAW_DATA', 'ANGLES', 'QUAT'])
        print(f"✓ Data mask creation: {mask}")
        
        # Test statistics
        stats = cmd_interface.get_statistics()
        print(f"✓ Command statistics: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ CommandInterface test failed: {e}")
        return False

def test_config():
    """Test configuration functionality."""
    print("\nTesting Config...")
    
    try:
        from config import Config
        
        # Create config
        config = Config()
        print("✓ Config created successfully")
        
        # Test configuration sections
        serial_config = config.get_serial_config()
        print(f"✓ Serial config: {serial_config}")
        
        data_config = config.get_data_config()
        print(f"✓ Data config: {data_config}")
        
        # Test data mask
        mask = config.get_data_mask()
        print(f"✓ Data mask: {mask}")
        
        # Test enabled data types
        enabled_types = config.get_enabled_data_types()
        print(f"✓ Enabled data types: {enabled_types}")
        
        # Test validation
        errors = config.validate_config()
        if not errors:
            print("✓ Configuration validation successful")
        else:
            print(f"⚠ Configuration validation warnings: {errors}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_data_logger():
    """Test data logger functionality."""
    print("\nTesting DataLogger...")
    
    try:
        from data_logger import DataLogger
        
        # Create logger
        logger = DataLogger(log_directory="./test_logs")
        print("✓ DataLogger created successfully")
        
        # Test log files
        log_files = logger.get_log_files()
        print(f"✓ Log files: {len(log_files)}")
        
        # Test statistics
        stats = logger.get_log_statistics()
        print(f"✓ Logger statistics: {stats}")
        
        # Cleanup
        logger.stop_logging_thread()
        print("✓ DataLogger cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"✗ DataLogger test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ISM330DHCX Data Visualization Tool - Test Suite")
    print("=" * 50)
    
    # Setup logging
    from utils import setup_logging
    setup_logging(level=logging.WARNING)
    
    tests = [
        test_imports,
        test_serial_interface,
        test_data_parser,
        test_command_interface,
        test_config,
        test_data_logger
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The application should work correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
