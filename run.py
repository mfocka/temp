#!/usr/bin/env python3
"""
Launcher script for ISM330DHCX Data Visualization Tool
"""

import sys
import os
import subprocess
import platform

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'serial',
        'matplotlib',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main launcher function."""
    print("ISM330DHCX Data Visualization Tool")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return 1
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Add current directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    try:
        # Import and run main application
        from main import main as app_main
        print("Starting application...")
        app_main()
        return 0
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required files are present")
        return 1
        
    except Exception as e:
        print(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
