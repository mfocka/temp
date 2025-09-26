#!/usr/bin/env python3
"""
ISM330DHCX Data Visualization Tool
Main application entry point for real-time sensor data visualization.
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from serial_interface import SerialInterface
from data_parser import DataParser
from visualization import VisualizationEngine
from data_logger import DataLogger
from command_interface import CommandInterface
from config import Config
from utils import setup_logging

class ISM330DHCXTool:
    """Main application class for ISM330DHCX data visualization tool."""
    
    def __init__(self):
        """Initialize the application."""
        self.root = tk.Tk()
        self.root.title("ISM330DHCX Data Visualization Tool")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize components
        self.config = Config()
        self.serial_interface = SerialInterface()
        self.data_parser = DataParser()
        self.visualization = VisualizationEngine(self.root)
        self.data_logger = DataLogger()
        self.command_interface = CommandInterface()
        
        # Application state
        self.is_connected = False
        self.is_collecting = False
        self.data_thread = None
        self.running = True
        
        # Setup logging
        setup_logging()
        
        # Initialize UI
        self.setup_ui()
        self.setup_bindings()
        
        # Start data processing thread
        self.start_data_thread()
        
    def setup_ui(self):
        """Setup the main user interface."""
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create main frames
        self.create_sidebar()
        self.create_main_area()
        self.create_status_bar()
        
    def create_sidebar(self):
        """Create the left sidebar with controls."""
        self.sidebar = ttk.Frame(self.root, width=300, padding="10")
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.sidebar.grid_propagate(False)
        
        # Connection controls
        self.create_connection_frame()
        
        # Data stream controls
        self.create_data_stream_frame()
        
        # Command interface
        self.create_command_frame()
        
        # Settings
        self.create_settings_frame()
        
    def create_connection_frame(self):
        """Create connection control frame."""
        frame = ttk.LabelFrame(self.sidebar, text="Connection", padding="5")
        frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # COM port selection
        ttk.Label(frame, text="COM Port:").grid(row=0, column=0, sticky="w")
        self.com_port_var = tk.StringVar()
        self.com_port_combo = ttk.Combobox(frame, textvariable=self.com_port_var, width=15)
        self.com_port_combo.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        
        # Refresh button
        ttk.Button(frame, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=(5, 0))
        
        # Baud rate selection
        ttk.Label(frame, text="Baud Rate:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.baud_rate_var = tk.StringVar(value="115200")
        baud_combo = ttk.Combobox(frame, textvariable=self.baud_rate_var, 
                                 values=["9600", "38400", "57600", "115200", "230400"], width=15)
        baud_combo.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(5, 0))
        
        # Connection buttons
        self.connect_btn = ttk.Button(frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky="ew")
        
        # Configure grid weights
        frame.grid_columnconfigure(1, weight=1)
        
    def create_data_stream_frame(self):
        """Create data stream selection frame."""
        frame = ttk.LabelFrame(self.sidebar, text="Data Streams", padding="5")
        frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        # Data type checkboxes
        self.data_stream_vars = {}
        data_types = [
            ("RAW_DATA", "Raw Data"),
            ("ANGLES", "Angles"),
            ("QUAT", "Quaternion"),
            ("GYRO_BIAS_MDI", "Gyro Bias"),
            ("ANGLES_DI", "MotionDI Filter"),
            ("ANGLES_SI", "Simple Integration"),
            ("ANGLES_CO", "Complementary Filter"),
            ("ANGLES_FU", "Fused Output")
        ]
        
        for i, (key, label) in enumerate(data_types):
            var = tk.BooleanVar(value=True)
            self.data_stream_vars[key] = var
            ttk.Checkbutton(frame, text=label, variable=var, 
                           command=self.update_data_streams).grid(row=i, column=0, sticky="w")
        
        # Start/Stop collection
        self.collect_btn = ttk.Button(frame, text="Start Collection", 
                                     command=self.toggle_collection, state="disabled")
        self.collect_btn.grid(row=len(data_types), column=0, pady=(10, 0), sticky="ew")
        
    def create_command_frame(self):
        """Create command interface frame."""
        frame = ttk.LabelFrame(self.sidebar, text="Commands", padding="5")
        frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        # Command input
        ttk.Label(frame, text="Command:").grid(row=0, column=0, sticky="w")
        self.command_var = tk.StringVar()
        self.command_entry = ttk.Entry(frame, textvariable=self.command_var, width=20)
        self.command_entry.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        self.command_entry.bind('<Return>', self.send_command)
        
        # Send button
        ttk.Button(frame, text="Send", command=self.send_command).grid(row=1, column=1, padx=(5, 0), pady=(5, 0))
        
        # Quick commands
        quick_commands = [
            ("PRINTRAW 127", "Enable All Data"),
            ("MSTATUS", "Get Status"),
            ("ALCAL", "Start Calibration")
        ]
        
        for i, (cmd, label) in enumerate(quick_commands):
            ttk.Button(frame, text=label, 
                      command=lambda c=cmd: self.send_quick_command(c)).grid(row=i+2, column=0, columnspan=2, 
                                                                           sticky="ew", pady=(2, 0))
        
        # Configure grid weights
        frame.grid_columnconfigure(0, weight=1)
        
    def create_settings_frame(self):
        """Create settings frame."""
        frame = ttk.LabelFrame(self.sidebar, text="Settings", padding="5")
        frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        
        # Chart update rate
        ttk.Label(frame, text="Update Rate (Hz):").grid(row=0, column=0, sticky="w")
        self.update_rate_var = tk.StringVar(value="10")
        update_rate_combo = ttk.Combobox(frame, textvariable=self.update_rate_var,
                                       values=["1", "5", "10", "20", "50"], width=10)
        update_rate_combo.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        update_rate_combo.bind('<<ComboboxSelected>>', self.on_update_rate_change)
        
        # Data logging
        self.logging_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Enable Logging", variable=self.logging_var).grid(row=1, column=0, sticky="w", pady=(5, 0))

        # Reset view button
        ttk.Button(frame, text="Reset View (Ctrl+R)", command=self.reset_view).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        
        # Configure grid weights
        frame.grid_columnconfigure(1, weight=1)
        
    def create_main_area(self):
        """Create the main visualization area."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Create notebook for different views
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Charts tab
        self.charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.charts_frame, text="Charts")
        
        # Angles tab
        self.angles_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.angles_frame, text="Angles")
        
        # Data table tab
        self.table_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="Data Table")
        
        # Initialize visualization components
        self.visualization.setup_charts(self.charts_frame)
        self.visualization.setup_angles(self.angles_frame)
        self.visualization.setup_data_table(self.table_frame)
        # Start periodic UI update loop (drains queue and redraws)
        self.visualization.start_update_loop()
        # Apply initial update rate
        self.on_update_rate_change()
        
    def create_status_bar(self):
        """Create status bar at bottom."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        
        self.status_label = ttk.Label(self.status_frame, text="Disconnected")
        self.status_label.grid(row=0, column=0, sticky="w")
        
        self.data_count_label = ttk.Label(self.status_frame, text="Data Points: 0")
        self.data_count_label.grid(row=0, column=1, sticky="e")
        
    def setup_bindings(self):
        """Setup event bindings."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind('<Control-r>', self.reset_view)
        
    def refresh_ports(self):
        """Refresh available COM ports."""
        ports = self.serial_interface.get_available_ports()
        self.com_port_combo['values'] = ports
        if ports and not self.com_port_var.get():
            self.com_port_var.set(ports[0])
            
    def toggle_connection(self):
        """Toggle serial connection."""
        if not self.is_connected:
            self.connect()
        else:
            self.disconnect()
            
    def connect(self):
        """Connect to serial port."""
        port = self.com_port_var.get()
        baud_rate = int(self.baud_rate_var.get())
        
        if not port:
            messagebox.showerror("Error", "Please select a COM port")
            return
            
        try:
            if self.serial_interface.connect(port, baud_rate):
                self.is_connected = True
                self.connect_btn.config(text="Disconnect")
                self.collect_btn.config(state="normal")
                self.status_label.config(text=f"Connected to {port}")
                self.update_data_streams()
            else:
                messagebox.showerror("Error", "Failed to connect to serial port")
        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {str(e)}")
            
    def disconnect(self):
        """Disconnect from serial port."""
        self.is_connected = False
        self.is_collecting = False
        self.serial_interface.disconnect()
        self.connect_btn.config(text="Connect")
        self.collect_btn.config(text="Start Collection", state="disabled")
        self.status_label.config(text="Disconnected")
        
    def toggle_collection(self):
        """Toggle data collection."""
        if not self.is_collecting:
            self.start_collection()
        else:
            self.stop_collection()
            
    def start_collection(self):
        """Start data collection."""
        if not self.is_connected:
            messagebox.showerror("Error", "Not connected to sensor")
            return
            
        self.is_collecting = True
        self.collect_btn.config(text="Stop Collection")
        self.status_label.config(text="Collecting data...")
        
        # Enable all data streams
        self.send_command("PRINTRAW 127")
        
    def stop_collection(self):
        """Stop data collection."""
        self.is_collecting = False
        self.collect_btn.config(text="Start Collection")
        self.status_label.config(text="Connected")
        
    def update_data_streams(self):
        """Update enabled data streams."""
        if not self.is_connected:
            return
            
        # Calculate bit mask for enabled streams
        mask = 0
        if self.data_stream_vars["RAW_DATA"].get():
            mask |= 1
        if self.data_stream_vars["ANGLES"].get():
            mask |= 4
        if self.data_stream_vars["QUAT"].get():
            mask |= 8
        if self.data_stream_vars["GYRO_BIAS_MDI"].get():
            mask |= 32
        if any(self.data_stream_vars[f"ANGLES_{f}"].get() for f in ["DI", "SI", "CO", "FU"]):
            mask |= 64
            
        self.send_command(f"PRINTRAW {mask}")
        
    def send_command(self, event=None):
        """Send command to sensor."""
        command = self.command_var.get().strip()
        if not command:
            return
            
        if self.is_connected:
            response = self.command_interface.send_command(self.serial_interface, command)
        else:
            messagebox.showerror("Error", "Not connected to sensor")
            
        self.command_var.set("")
        
    def send_quick_command(self, command):
        """Send a quick command."""
        self.command_var.set(command)
        self.send_command()
        
    def start_data_thread(self):
        """Start the data processing thread."""
        self.data_thread = threading.Thread(target=self.data_processing_loop, daemon=True)
        self.data_thread.start()
        
    def data_processing_loop(self):
        """Main data processing loop."""
        while self.running:
            if self.is_connected and self.is_collecting:
                try:
                    # Prefer batch read for performance
                    lines = self.serial_interface.read_all_available()
                    if not lines:
                        # Fall back to single line
                        single = self.serial_interface.read_line()
                        if single:
                            lines = [single]

                    if lines:
                        for data in lines:
                            parsed_data = self.data_parser.parse_line(data)
                            if parsed_data:
                                # Enqueue for UI thread to process
                                self.visualization.update_data(parsed_data)
                                # Log data if enabled (background thread safe)
                                if self.logging_var.get():
                                    self.data_logger.log_data(parsed_data)

                        # Update status (UI thread)
                        self.root.after(0, self.update_data_count)
                            
                except Exception as e:
                    print(f"Data processing error: {e}")
                    
            time.sleep(0.001)  # Fast loop; UI throttled by visualization.update_interval
            
    def update_data_count(self):
        """Update data count in status bar."""
        count = self.data_parser.get_total_data_count()
        self.data_count_label.config(text=f"Data Points: {count}")

    def on_update_rate_change(self, event=None):
        """Apply update rate to visualization engine."""
        try:
            hz = max(1, int(self.update_rate_var.get()))
            interval_ms = int(1000 / hz)
            self.visualization.set_update_interval(interval_ms)
        except Exception:
            pass

    def reset_view(self, event=None):
        """Reset visualization, parser, and flush serial buffers."""
        try:
            self.visualization.reset()
        except Exception:
            pass
        try:
            self.data_parser.clear_data()
        except Exception:
            pass
        try:
            self.serial_interface.flush_buffers()
        except Exception:
            pass
        self.update_data_count()
        
    def on_closing(self):
        """Handle application closing."""
        self.running = False
        if self.is_connected:
            self.disconnect()
        self.root.destroy()
        
    def run(self):
        """Run the application."""
        # Refresh ports on startup
        self.refresh_ports()
        
        # Start the GUI main loop
        self.root.mainloop()

def main():
    """Main entry point."""
    try:
        app = ISM330DHCXTool()
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        messagebox.showerror("Error", f"Application error: {e}")

if __name__ == "__main__":
    main()
