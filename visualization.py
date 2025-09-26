"""
Visualization Engine Module
Handles real-time data visualization with charts and gauges.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
import queue
import logging
from datetime import datetime, timedelta
import math
import time

class CircularGauge:
    """Circular gauge widget for angle display."""
    
    def __init__(self, parent, title: str, min_val: float = 0, max_val: float = 360, 
                 size: int = 150, unit: str = "°"):
        """Initialize circular gauge."""
        self.parent = parent
        self.title = title
        self.min_val = min_val
        self.max_val = max_val
        self.size = size
        self.unit = unit
        self.current_value = 0.0
        
        # Create canvas
        self.canvas = tk.Canvas(parent, width=size, height=size + 30, bg='white')
        
        # Draw gauge
        self.draw_gauge()
        
    def draw_gauge(self):
        """Draw the circular gauge."""
        center_x = self.size // 2
        center_y = self.size // 2
        radius = (self.size - 20) // 2
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw outer circle
        self.canvas.create_oval(center_x - radius, center_y - radius,
                               center_x + radius, center_y + radius,
                               outline='black', width=2)
        
        # Draw scale marks
        for angle in range(0, 360, 30):
            angle_rad = math.radians(angle)
            x1 = center_x + (radius - 10) * math.cos(angle_rad)
            y1 = center_y + (radius - 10) * math.sin(angle_rad)
            x2 = center_x + radius * math.cos(angle_rad)
            y2 = center_y + radius * math.sin(angle_rad)
            self.canvas.create_line(x1, y1, x2, y2, fill='black', width=2)
            
            # Draw labels
            label_x = center_x + (radius + 15) * math.cos(angle_rad)
            label_y = center_y + (radius + 15) * math.sin(angle_rad)
            value = int(self.min_val + (angle / 360) * (self.max_val - self.min_val))
            self.canvas.create_text(label_x, label_y, text=str(value), font=('Arial', 8))
        
        # Draw needle
        self.update_value(0)
        
        # Draw title
        self.canvas.create_text(center_x, self.size - 10, text=self.title, 
                               font=('Arial', 10, 'bold'))
        
    def update_value(self, value: float):
        """Update gauge value."""
        # Clamp value to valid range
        value = max(self.min_val, min(self.max_val, float(value)))
        self.current_value = value
        
        center_x = self.size // 2
        center_y = self.size // 2
        radius = (self.size - 20) // 2
        
        # Calculate needle angle
        normalized_value = (value - self.min_val) / (self.max_val - self.min_val)
        angle = normalized_value * 360
        angle_rad = math.radians(angle - 90)  # Start from top
        
        # Draw needle
        needle_length = radius - 15
        x = center_x + needle_length * math.cos(angle_rad)
        y = center_y + needle_length * math.sin(angle_rad)
        
        # Clear previous needle and value
        self.canvas.delete("needle")
        self.canvas.delete("value")
        
        # Draw needle
        self.canvas.create_line(center_x, center_y, x, y, 
                               fill='red', width=3, tags="needle")
        
        # Draw center dot
        self.canvas.create_oval(center_x - 5, center_y - 5,
                               center_x + 5, center_y + 5,
                               fill='red', outline='darkred', tags="needle")
        
        # Update value display
        self.canvas.create_text(center_x, center_y + 20, 
                               text=f"{value:.1f}{self.unit}", 
                               font=('Arial', 12, 'bold'), tags="value")

class VisualizationEngine:
    """Main visualization engine for sensor data."""
    
    def __init__(self, root):
        """Initialize visualization engine."""
        self.root = root
        self.logger = logging.getLogger(__name__)
        
        # Chart configuration
        self.max_data_points = 500  # Reduced for better performance at 52Hz
        self.update_interval = 20  # ms - ~50Hz update rate for 52Hz data

        # Relative time baseline (set on first data after reset)
        self.time_zero: Optional[float] = None
        
        # Data storage using deque for efficient pops
        self.data_buffers = {
            'RAW_DATA': {
                'time': deque(maxlen=self.max_data_points),
                'acc_x': deque(maxlen=self.max_data_points),
                'acc_y': deque(maxlen=self.max_data_points),
                'acc_z': deque(maxlen=self.max_data_points),
                'gyro_x': deque(maxlen=self.max_data_points),
                'gyro_y': deque(maxlen=self.max_data_points),
                'gyro_z': deque(maxlen=self.max_data_points)
            },
            'ANGLES': {
                'time': deque(maxlen=self.max_data_points),
                'altitude': deque(maxlen=self.max_data_points),
                'azimuth': deque(maxlen=self.max_data_points),
                'zenith': deque(maxlen=self.max_data_points)
            },
            'QUAT': {
                'time': deque(maxlen=self.max_data_points),
                'qx': deque(maxlen=self.max_data_points),
                'qy': deque(maxlen=self.max_data_points),
                'qz': deque(maxlen=self.max_data_points),
                'qw': deque(maxlen=self.max_data_points)
            },
            'GYRO_BIAS_MDI': {
                'time': deque(maxlen=self.max_data_points),
                'bias_x': deque(maxlen=self.max_data_points),
                'bias_y': deque(maxlen=self.max_data_points),
                'bias_z': deque(maxlen=self.max_data_points)
            },
            'ANGLES_DI': {
                'time': deque(maxlen=self.max_data_points),
                'pitch': deque(maxlen=self.max_data_points),
                'yaw': deque(maxlen=self.max_data_points),
                'roll': deque(maxlen=self.max_data_points)
            },
            'ANGLES_SI': {
                'time': deque(maxlen=self.max_data_points),
                'pitch': deque(maxlen=self.max_data_points),
                'yaw': deque(maxlen=self.max_data_points),
                'roll': deque(maxlen=self.max_data_points)
            },
            'ANGLES_CO': {
                'time': deque(maxlen=self.max_data_points),
                'pitch': deque(maxlen=self.max_data_points),
                'yaw': deque(maxlen=self.max_data_points),
                'roll': deque(maxlen=self.max_data_points)
            },
            'ANGLES_FU': {
                'time': deque(maxlen=self.max_data_points),
                'pitch': deque(maxlen=self.max_data_points),
                'yaw': deque(maxlen=self.max_data_points),
                'roll': deque(maxlen=self.max_data_points)
            }
        }
        
        # Matplotlib style
        plt.style.use('seaborn-v0_8')
        
        # Initialize components
        self.charts = {}
        self.gauges = {}
        self.data_table = None

        # Thread-safe queue for parsed data coming from background thread
        self.update_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1000)  # Smaller queue for faster processing

        # Track last known ANGLES state to derive events
        self._last_angles_state: Optional[str] = None

        # Throttling settings for performance
        self._last_table_update_ms: float = 0.0
        self.table_update_interval_ms: int = 100  # Faster table updates for 52Hz
        self._last_autoscale_ms: float = 0.0
        self.autoscale_interval_ms: int = 500  # Less frequent autoscaling for performance
        
    def setup_charts(self, parent):
        """Setup line charts for time-series data."""
        # Create main frame
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create notebook for different chart groups
        self.chart_notebook = ttk.Notebook(main_frame)
        self.chart_notebook.pack(fill='both', expand=True)
        
        # Raw data tab
        self.setup_raw_data_charts()
        
        # Filter data tab
        self.setup_filter_charts()

        # Events tab
        self.setup_events_tab()
        
        # Quaternion tab
        self.setup_quaternion_charts()
        
    def setup_raw_data_charts(self):
        """Setup raw data charts."""
        frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(frame, text="Raw Data")
        
        # Create figure for raw data
        self.raw_fig = Figure(figsize=(12, 8), dpi=100)
        self.raw_canvas = FigureCanvasTkAgg(self.raw_fig, frame)
        self.raw_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.raw_canvas, frame)
        toolbar.update()
        
        # Create subplots
        self.raw_axes = self.raw_fig.subplots(2, 1, sharex=True)
        self.raw_fig.suptitle('Raw Sensor Data')
        
        # Accelerometer plot
        self.raw_axes[0].set_title('Accelerometer (mg)')
        self.raw_axes[0].set_ylabel('Acceleration (mg)')
        self.raw_axes[0].grid(True, alpha=0.3)
        self.raw_axes[0].legend(['X', 'Y', 'Z'], loc='upper right')
        
        # Gyroscope plot
        self.raw_axes[1].set_title('Gyroscope (dps)')
        self.raw_axes[1].set_xlabel('Time (s)')
        self.raw_axes[1].set_ylabel('Angular Velocity (dps)')
        self.raw_axes[1].grid(True, alpha=0.3)
        self.raw_axes[1].legend(['X', 'Y', 'Z'], loc='upper right')
        
        # Initialize plot lines
        self.raw_lines = {}
        for i, axis in enumerate(['acc_x', 'acc_y', 'acc_z']):
            self.raw_lines[axis] = self.raw_axes[0].plot([], [], 
                color=['red', 'green', 'blue'][i], linewidth=1)[0]
        for i, axis in enumerate(['gyro_x', 'gyro_y', 'gyro_z']):
            self.raw_lines[axis] = self.raw_axes[1].plot([], [], 
                color=['red', 'green', 'blue'][i], linewidth=1)[0]
        
        self.charts['raw'] = {
            'figure': self.raw_fig,
            'canvas': self.raw_canvas,
            'axes': self.raw_axes,
            'lines': self.raw_lines
        }
        
    def setup_filter_charts(self):
        """Setup filter output charts."""
        frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(frame, text="Filter Outputs")
        
        # Create figure for filter data
        self.filter_fig = Figure(figsize=(12, 8), dpi=100)
        self.filter_canvas = FigureCanvasTkAgg(self.filter_fig, frame)
        self.filter_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.filter_canvas, frame)
        toolbar.update()
        
        # Create subplots for different filters
        self.filter_axes = self.filter_fig.subplots(3, 2, sharex=True)
        self.filter_fig.suptitle('Filter Outputs (with Final ANGLES time series)')
        
        filter_types = ['DI', 'SI', 'CO', 'FU']
        filter_names = ['MotionDI', 'Simple Integration', 'Complementary', 'Fused']
        
        self.filter_lines = {}
        # 4 filter plots in rows 0-1, and resulting angle in row 2 spanning both columns
        for i, (filter_type, name) in enumerate(zip(filter_types, filter_names)):
            row, col = i // 2, i % 2
            ax = self.filter_axes[row, col]
            ax.set_title(f'{name} Filter')
            ax.set_ylabel('Angle (degrees)')
            ax.grid(True, alpha=0.3)
            ax.legend(['Pitch', 'Yaw', 'Roll'], loc='upper right')
            
            # Initialize plot lines
            for j, angle in enumerate(['pitch', 'yaw', 'roll']):
                key = f'ANGLES_{filter_type}_{angle}'
                self.filter_lines[key] = ax.plot([], [], 
                    color=['red', 'green', 'blue'][j], linewidth=1)[0]
        
        # Bottom row: show final ANGLES (altitude, azimuth, zenith) time series, not fused magnitude
        self.angles_ax_left = self.filter_axes[2, 0]
        self.angles_ax_right = self.filter_axes[2, 1]
        for ax in (self.angles_ax_left, self.angles_ax_right):
            ax.set_title('Final ANGLES (Altitude, Azimuth, Zenith)')
            ax.set_ylabel('Angle (degrees)')
            ax.grid(True, alpha=0.3)

        # Create separate line sets for both axes (kept in the same registry for clearing)
        self.angles_lines_left = {
            'ANGLES_altitude_left': self.angles_ax_left.plot([], [], color='red', linewidth=1)[0],
            'ANGLES_azimuth_left': self.angles_ax_left.plot([], [], color='green', linewidth=1)[0],
            'ANGLES_zenith_left': self.angles_ax_left.plot([], [], color='blue', linewidth=1)[0],
        }
        self.angles_lines_right = {
            'ANGLES_altitude_right': self.angles_ax_right.plot([], [], color='red', linewidth=1)[0],
            'ANGLES_azimuth_right': self.angles_ax_right.plot([], [], color='green', linewidth=1)[0],
            'ANGLES_zenith_right': self.angles_ax_right.plot([], [], color='blue', linewidth=1)[0],
        }

        # Register all lines, including final ANGLES lines, for consistent clearing
        filter_all_lines = {}
        filter_all_lines.update(self.filter_lines)
        filter_all_lines.update(self.angles_lines_left)
        filter_all_lines.update(self.angles_lines_right)

        self.charts['filter'] = {
            'figure': self.filter_fig,
            'canvas': self.filter_canvas,
            'axes': self.filter_axes,
            'lines': filter_all_lines
        }
        
    def setup_quaternion_charts(self):
        """Setup quaternion charts."""
        frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(frame, text="Quaternion")
        
        # Create figure for quaternion data
        self.quat_fig = Figure(figsize=(12, 6), dpi=100)
        self.quat_canvas = FigureCanvasTkAgg(self.quat_fig, frame)
        self.quat_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.quat_canvas, frame)
        toolbar.update()
        
        # Create subplot
        self.quat_ax = self.quat_fig.subplots(1, 1)
        self.quat_fig.suptitle('Quaternion Components')
        self.quat_ax.set_xlabel('Time (s)')
        self.quat_ax.set_ylabel('Quaternion Value')
        self.quat_ax.grid(True, alpha=0.3)
        self.quat_ax.legend(['QX', 'QY', 'QZ', 'QW'], loc='upper right')
        
        # Initialize plot lines
        self.quat_lines = {}
        for i, component in enumerate(['qx', 'qy', 'qz', 'qw']):
            self.quat_lines[component] = self.quat_ax.plot([], [], 
                color=['red', 'green', 'blue', 'orange'][i], linewidth=1)[0]
        
        self.charts['quaternion'] = {
            'figure': self.quat_fig,
            'canvas': self.quat_canvas,
            'axes': self.quat_ax,
            'lines': self.quat_lines
        }

    def setup_events_tab(self):
        """Setup events tab showing raised/cleared timeline."""
        frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(frame, text="Events")

        # Table
        columns = ('Time (s)', 'Event', 'Action', 'Details')
        self.events_table = ttk.Treeview(frame, columns=columns, show='headings', height=10)
        for col in columns:
            self.events_table.heading(col, text=col)
            self.events_table.column(col, width=160)
        self.events_table.pack(side='top', fill='x', padx=5, pady=5)

        # Timeline figure
        self.events_fig = Figure(figsize=(12, 3), dpi=100)
        self.events_canvas = FigureCanvasTkAgg(self.events_fig, frame)
        self.events_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.events_ax = self.events_fig.subplots(1, 1)
        self.events_ax.set_title('Events Timeline')
        self.events_ax.set_xlabel('Time (s)')
        self.events_ax.set_ylabel('State (0=Cleared, 1=Raised)')
        self.events_ax.grid(True, alpha=0.3)
        self.events_scatter = self.events_ax.plot([], [], 'o', color='#f4c430', markersize=4)[0]
        self.events_data = []  # List of tuples (t, name, action, details)

    def add_event(self, t: float, name: str, action: str, details: str = ""):
        """Add event to table and timeline."""
        self.events_data.append((t, name, action, details))
        self.events_table.insert('', 'end', values=(f"{t:.3f}", name, action, details))
        # Update timeline
        xs = [e[0] for e in self.events_data]
        ys = [1.0 if e[2].lower().startswith('raise') else 0.0 for e in self.events_data]
        self.events_scatter.set_data(xs, ys)
        self.events_ax.relim()
        self.events_ax.autoscale_view()
        self.events_canvas.draw_idle()
        
    def setup_angles(self, parent):
        """Setup circular gauges for angle display."""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create title
        title_label = ttk.Label(main_frame, text="Motion Detection Angles", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Create gauge frame
        gauge_frame = ttk.Frame(main_frame)
        gauge_frame.pack(expand=True)
        
        # Create gauges
        self.gauges['azimuth'] = CircularGauge(gauge_frame, "Azimuth", 0, 360, 150, "°")
        self.gauges['azimuth'].canvas.grid(row=0, column=0, padx=10, pady=10)
        
        self.gauges['altitude'] = CircularGauge(gauge_frame, "Altitude", 0, 180, 150, "°")
        self.gauges['altitude'].canvas.grid(row=0, column=1, padx=10, pady=10)
        
        self.gauges['zenith'] = CircularGauge(gauge_frame, "Zenith", 0, 180, 150, "°")
        self.gauges['zenith'].canvas.grid(row=0, column=2, padx=10, pady=10)
        
        # State display
        state_frame = ttk.Frame(main_frame)
        state_frame.pack(pady=10)
        
        ttk.Label(state_frame, text="Current State:", font=('Arial', 12, 'bold')).pack(side='left')
        self.state_label = ttk.Label(state_frame, text="UNKNOWN", 
                                    font=('Arial', 12), foreground='red')
        self.state_label.pack(side='left', padx=(10, 0))
        
    def setup_data_table(self, parent):
        """Setup data table for current values."""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create title
        title_label = ttk.Label(main_frame, text="Current Sensor Values", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Create treeview for data table
        columns = ('Parameter', 'Value', 'Unit', 'Timestamp')
        self.data_table = ttk.Treeview(main_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        self.data_table.heading('Parameter', text='Parameter')
        self.data_table.heading('Value', text='Value')
        self.data_table.heading('Unit', text='Unit')
        self.data_table.heading('Timestamp', text='Timestamp')
        
        self.data_table.column('Parameter', width=200)
        self.data_table.column('Value', width=150)
        self.data_table.column('Unit', width=100)
        self.data_table.column('Timestamp', width=200)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=self.data_table.yview)
        self.data_table.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.data_table.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Initialize with empty data
        self.update_data_table()
        
    def update_data(self, parsed_data: Dict[str, Any]):
        """Enqueue new parsed data to be processed on the UI thread."""
        if not parsed_data:
            return
        try:
            self.update_queue.put_nowait(parsed_data)
        except queue.Full:
            # Drop oldest to make room for newest
            try:
                _ = self.update_queue.get_nowait()
            except Exception:
                pass
            try:
                self.update_queue.put_nowait(parsed_data)
            except Exception:
                pass

    def _apply_parsed_data(self, parsed_data: Dict[str, Any]):
        """Apply parsed data to buffers and UI state (no drawing here)."""
        data_type = parsed_data['type']
        data = parsed_data.get('data')
        timestamp = parsed_data.get('timestamp')

        if data_type in self.data_buffers and timestamp is not None and data is not None:
            if self.time_zero is None:
                self.time_zero = float(timestamp)
            t_rel = float(timestamp) - float(self.time_zero)
            buffer = self.data_buffers[data_type]
            buffer['time'].append(t_rel)

            if data_type == 'RAW_DATA':
                buffer['acc_x'].append(data.acc_x)
                buffer['acc_y'].append(data.acc_y)
                buffer['acc_z'].append(data.acc_z)
                buffer['gyro_x'].append(data.gyro_x)
                buffer['gyro_y'].append(data.gyro_y)
                buffer['gyro_z'].append(data.gyro_z)

            elif data_type == 'ANGLES':
                buffer['altitude'].append(data.altitude)
                buffer['azimuth'].append(data.azimuth)
                buffer['zenith'].append(data.zenith)

                # Update gauges with proper values
                try:
                    if 'azimuth' in self.gauges and hasattr(data, 'azimuth'):
                        self.gauges['azimuth'].update_value(float(data.azimuth))
                    if 'altitude' in self.gauges and hasattr(data, 'altitude'):
                        self.gauges['altitude'].update_value(float(data.altitude))
                    if 'zenith' in self.gauges and hasattr(data, 'zenith'):
                        self.gauges['zenith'].update_value(float(data.zenith))
                except Exception as e:
                    self.logger.debug(f"Error updating gauges: {e}")

                # Update state label and raise event if changed
                self.state_label.config(text=data.state)
                state_color = 'green' if data.state == 'MONITORING' else 'orange'
                self.state_label.config(foreground=state_color)

                try:
                    if self._last_angles_state is None or self._last_angles_state != data.state:
                        action = 'RAISED' if (data.state and data.state.upper() != 'CLEARED') else 'CLEARED'
                        self.add_event(t_rel, 'ANGLES_STATE', action, data.state)
                        self._last_angles_state = data.state
                except Exception:
                    pass

            elif data_type == 'QUAT':
                buffer['qx'].append(data.qx)
                buffer['qy'].append(data.qy)
                buffer['qz'].append(data.qz)
                buffer['qw'].append(data.qw)

            elif data_type == 'GYRO_BIAS_MDI':
                buffer['bias_x'].append(data.bias_x)
                buffer['bias_y'].append(data.bias_y)
                buffer['bias_z'].append(data.bias_z)

            elif data_type.startswith('ANGLES_'):
                buffer['pitch'].append(data.pitch)
                buffer['yaw'].append(data.yaw)
                buffer['roll'].append(data.roll)

            # Deque maxlen enforces buffer size; no manual truncation needed

    def process_pending_data(self):
        """Drain the pending data queue and update charts and table once."""
        drained = 0
        max_drain = 100  # Limit to prevent UI freezing
        try:
            while drained < max_drain:
                item = self.update_queue.get_nowait()
                self._apply_parsed_data(item)
                drained += 1
        except queue.Empty:
            pass

        if drained > 0:
            self.update_charts()
            self.update_data_table()

    def start_update_loop(self):
        """Start periodic UI update loop based on update_interval."""
        def _loop():
            try:
                self.process_pending_data()
            finally:
                # schedule next
                self.root.after(self.update_interval, _loop)
        # kick off
        self.root.after(self.update_interval, _loop)
        
    def update_charts(self):
        """Update all charts with current data."""
        # Update raw data charts
        if 'raw' in self.charts:
            self.update_raw_charts()
            
        # Update filter charts
        if 'filter' in self.charts:
            self.update_filter_charts()
            
        # Update quaternion charts
        if 'quaternion' in self.charts:
            self.update_quaternion_charts()
    
    def update_raw_charts(self):
        """Update raw data charts."""
        buffer = self.data_buffers['RAW_DATA']
        if not buffer['time']:
            return
            
        # Update accelerometer plot
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            if buffer[axis]:
                self.charts['raw']['lines'][axis].set_data(buffer['time'], buffer[axis])
        
        # Update gyroscope plot
        for axis in ['gyro_x', 'gyro_y', 'gyro_z']:
            if buffer[axis]:
                self.charts['raw']['lines'][axis].set_data(buffer['time'], buffer[axis])
        
        # Auto-scale axes (throttled)
        now_ms = time.time() * 1000.0
        if now_ms - self._last_autoscale_ms >= self.autoscale_interval_ms:
            for ax in self.charts['raw']['axes']:
                ax.relim()
                ax.autoscale_view()
            self._last_autoscale_ms = now_ms
        
        self.charts['raw']['canvas'].draw_idle()
    
    def update_filter_charts(self):
        """Update filter output charts."""
        filter_types = ['DI', 'SI', 'CO', 'FU']
        
        now_ms = time.time() * 1000.0
        do_autoscale = (now_ms - self._last_autoscale_ms) >= self.autoscale_interval_ms

        for i, filter_type in enumerate(filter_types):
            buffer = self.data_buffers[f'ANGLES_{filter_type}']
            if not buffer['time']:
                continue
                
            row, col = i // 2, i % 2
            ax = self.charts['filter']['axes'][row, col]
            
            # Update plot lines
            for j, angle in enumerate(['pitch', 'yaw', 'roll']):
                key = f'ANGLES_{filter_type}_{angle}'
                if buffer[angle]:
                    self.charts['filter']['lines'][key].set_data(buffer['time'], buffer[angle])
            
            # Auto-scale axes (throttled)
            if do_autoscale:
                ax.relim()
                ax.autoscale_view()

        # Update final ANGLES plots (altitude, azimuth, zenith)
        angles = self.data_buffers['ANGLES']
        if angles['time']:
            t = angles['time']
            # Left axis
            self.angles_lines_left['ANGLES_altitude_left'].set_data(t, angles['altitude'])
            self.angles_lines_left['ANGLES_azimuth_left'].set_data(t, angles['azimuth'])
            self.angles_lines_left['ANGLES_zenith_left'].set_data(t, angles['zenith'])
            if do_autoscale:
                self.angles_ax_left.relim()
                self.angles_ax_left.autoscale_view()

            # Right axis (mirror for now)
            self.angles_lines_right['ANGLES_altitude_right'].set_data(t, angles['altitude'])
            self.angles_lines_right['ANGLES_azimuth_right'].set_data(t, angles['azimuth'])
            self.angles_lines_right['ANGLES_zenith_right'].set_data(t, angles['zenith'])
            if do_autoscale:
                self.angles_ax_right.relim()
                self.angles_ax_right.autoscale_view()

        if do_autoscale:
            self._last_autoscale_ms = now_ms
        self.charts['filter']['canvas'].draw_idle()
    
    def update_quaternion_charts(self):
        """Update quaternion charts."""
        buffer = self.data_buffers['QUAT']
        if not buffer['time']:
            return
            
        # Update plot lines
        for component in ['qx', 'qy', 'qz', 'qw']:
            if buffer[component]:
                self.charts['quaternion']['lines'][component].set_data(buffer['time'], buffer[component])
        
        # Auto-scale axes (throttled)
        now_ms = time.time() * 1000.0
        if now_ms - self._last_autoscale_ms >= self.autoscale_interval_ms:
            self.charts['quaternion']['axes'].relim()
            self.charts['quaternion']['axes'].autoscale_view()
            self._last_autoscale_ms = now_ms
        
        self.charts['quaternion']['canvas'].draw_idle()
    
    def update_data_table(self):
        """Update data table with current values."""
        if not self.data_table:
            return
        now_ms = time.time() * 1000.0
        if now_ms - self._last_table_update_ms < self.table_update_interval_ms:
            return
        self._last_table_update_ms = now_ms
            
        # Clear existing items
        for item in self.data_table.get_children():
            self.data_table.delete(item)
        
        # Add current values
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Raw data
        raw_buffer = self.data_buffers['RAW_DATA']
        if raw_buffer['time']:
            latest_time = raw_buffer['time'][-1]
            self.data_table.insert('', 'end', values=('Accelerometer X', f"{raw_buffer['acc_x'][-1]:.2f}", 'mg', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Accelerometer Y', f"{raw_buffer['acc_y'][-1]:.2f}", 'mg', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Accelerometer Z', f"{raw_buffer['acc_z'][-1]:.2f}", 'mg', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Gyroscope X', f"{raw_buffer['gyro_x'][-1]:.2f}", 'dps', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Gyroscope Y', f"{raw_buffer['gyro_y'][-1]:.2f}", 'dps', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Gyroscope Z', f"{raw_buffer['gyro_z'][-1]:.2f}", 'dps', f"{latest_time:.2f}s"))
        
        # Angle data
        angles_buffer = self.data_buffers['ANGLES']
        if angles_buffer['time']:
            latest_time = angles_buffer['time'][-1]
            self.data_table.insert('', 'end', values=('Azimuth', f"{angles_buffer['azimuth'][-1]:.2f}", '°', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Altitude', f"{angles_buffer['altitude'][-1]:.2f}", '°', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Zenith', f"{angles_buffer['zenith'][-1]:.2f}", '°', f"{latest_time:.2f}s"))
        
        # Quaternion data
        quat_buffer = self.data_buffers['QUAT']
        if quat_buffer['time']:
            latest_time = quat_buffer['time'][-1]
            self.data_table.insert('', 'end', values=('Quaternion X', f"{quat_buffer['qx'][-1]:.3f}", '', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Quaternion Y', f"{quat_buffer['qy'][-1]:.3f}", '', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Quaternion Z', f"{quat_buffer['qz'][-1]:.3f}", '', f"{latest_time:.2f}s"))
            self.data_table.insert('', 'end', values=('Quaternion W', f"{quat_buffer['qw'][-1]:.3f}", '', f"{latest_time:.2f}s"))
    
    def clear_data(self):
        """Clear all visualization data."""
        for buffer in self.data_buffers.values():
            for key in buffer:
                buffer[key].clear()
        
        # Reset gauges
        for gauge in self.gauges.values():
            gauge.update_value(0)
        
        # Reset state
        self.state_label.config(text="UNKNOWN", foreground='red')
        
        # Clear charts
        for chart in self.charts.values():
            for line in chart['lines'].values():
                line.set_data([], [])
            chart['canvas'].draw_idle()
        
        # Clear data table
        if self.data_table:
            for item in self.data_table.get_children():
                self.data_table.delete(item)

    def reset(self):
        """Full reset: clear queues, data, events, and re-baseline time."""
        # Drain pending queue
        try:
            while True:
                _ = self.update_queue.get_nowait()
        except queue.Empty:
            pass

        # Reset time baseline and state
        self.time_zero = None
        self._last_angles_state = None
        self._last_table_update_ms = 0.0
        self._last_autoscale_ms = 0.0

        # Clear events
        self.events_data = []
        try:
            if hasattr(self, 'events_table') and self.events_table:
                for item in self.events_table.get_children():
                    self.events_table.delete(item)
            if hasattr(self, 'events_scatter') and hasattr(self, 'events_ax'):
                self.events_scatter.set_data([], [])
                self.events_ax.relim()
                self.events_ax.autoscale_view()
                self.events_canvas.draw_idle()
        except Exception:
            pass

        # Clear all visual data
        self.clear_data()

    def set_update_interval(self, interval_ms: int):
        """Set UI update interval in milliseconds (min 10ms)."""
        try:
            self.update_interval = max(10, int(interval_ms))
        except Exception:
            pass
    
    def export_chart(self, chart_name: str, filename: str):
        """Export chart to file."""
        if chart_name in self.charts:
            self.charts[chart_name]['figure'].savefig(filename, dpi=300, bbox_inches='tight')
            return True
        return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of current data."""
        summary = {}
        for data_type, buffer in self.data_buffers.items():
            if buffer['time']:
                summary[data_type] = {
                    'data_points': len(buffer['time']),
                    'time_range': (buffer['time'][0], buffer['time'][-1]) if len(buffer['time']) > 1 else buffer['time'][0],
                    'latest_timestamp': buffer['time'][-1]
                }
        return summary
