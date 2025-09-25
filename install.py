#!/usr/bin/env python3
"""
Installation script for ISM330DHCX Data Visualization Tool
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"✗ Python 3.8 or higher is required. Current version: {sys.version}")
        return False
    else:
        print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} is compatible")
        return True

def install_requirements():
    """Install required packages."""
    print("\nInstalling required packages...")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("✗ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = [
        "logs",
        "config",
        "exports"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except OSError as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    return True

def create_desktop_shortcut():
    """Create desktop shortcut (Windows only)."""
    if platform.system() != "Windows":
        return True
    
    print("\nCreating desktop shortcut...")
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "ISM330DHCX Tool.lnk")
        target = os.path.join(os.getcwd(), "run.py")
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{target}"'
        shortcut.WorkingDirectory = os.getcwd()
        shortcut.IconLocation = sys.executable
        shortcut.save()
        
        print("✓ Desktop shortcut created")
        return True
        
    except ImportError:
        print("⚠ Desktop shortcut creation skipped (winshell not available)")
        return True
    except Exception as e:
        print(f"⚠ Desktop shortcut creation failed: {e}")
        return True

def run_tests():
    """Run basic tests."""
    print("\nRunning tests...")
    
    if not run_command(f"{sys.executable} test_app.py", "Running application tests"):
        print("⚠ Some tests failed, but installation may still work")
        return True
    
    return True

def main():
    """Main installation function."""
    print("ISM330DHCX Data Visualization Tool - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Installation failed during package installation")
        return 1
    
    # Create directories
    if not create_directories():
        print("\n✗ Installation failed during directory creation")
        return 1
    
    # Create desktop shortcut
    create_desktop_shortcut()
    
    # Run tests
    run_tests()
    
    print("\n" + "=" * 50)
    print("✓ Installation completed successfully!")
    print("\nTo run the application:")
    print("  python run.py")
    print("  or")
    print("  python main.py")
    print("\nFor help:")
    print("  python run.py --help")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
