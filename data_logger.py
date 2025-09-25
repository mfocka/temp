"""
Data Logger Module
Handles data logging to CSV files with automatic file management.
"""

import os
import csv
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import threading
import queue
import time
from pathlib import Path

class DataLogger:
    """Handles data logging to CSV files."""
    
    def __init__(self, log_directory: str = "./logs", max_file_size: int = 10*1024*1024, 
                 max_files: int = 10, retention_days: int = 7):
        """
        Initialize data logger.
        
        Args:
            log_directory: Directory to store log files
            max_file_size: Maximum file size in bytes before rotation
            max_files: Maximum number of files to keep
            retention_days: Number of days to keep log files
        """
        self.log_directory = Path(log_directory)
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.retention_days = retention_days
        
        # Create log directory if it doesn't exist
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Data queue for background logging
        self.data_queue: queue.Queue = queue.Queue()
        self.logging_thread = None
        self.running = False
        
        # Current log files
        self.current_files: Dict[str, Path] = {}
        self.file_handles: Dict[Path, Any] = {}
        self.csv_writers: Dict[Path, Any] = {}
        
        # Statistics
        self.logged_records = 0
        self.start_time = None
        
        # Start background logging thread
        self.start_logging_thread()
        
    def start_logging_thread(self):
        """Start background logging thread."""
        if self.logging_thread and self.logging_thread.is_alive():
            return
            
        self.running = True
        self.logging_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.logging_thread.start()
        self.logger.info("Data logging thread started")
        
    def stop_logging_thread(self):
        """Stop background logging thread."""
        self.running = False
        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=5)
        self.logger.info("Data logging thread stopped")
        
    def _logging_worker(self):
        """Background worker for logging data."""
        while self.running:
            try:
                # Get data from queue with timeout
                data = self.data_queue.get(timeout=1.0)
                if data is None:  # Shutdown signal
                    break
                    
                self._write_data(data)
                self.data_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in logging worker: {e}")
                
    def log_data(self, parsed_data: Dict[str, Any]):
        """
        Log parsed data to appropriate CSV file.
        
        Args:
            parsed_data: Parsed data dictionary from data parser
        """
        if not parsed_data or not self.running:
            return
            
        # Add to queue for background processing
        try:
            self.data_queue.put(parsed_data, timeout=0.1)
        except queue.Full:
            self.logger.warning("Data queue full, dropping data")
            
    def _write_data(self, parsed_data: Dict[str, Any]):
        """Write data to CSV file."""
        data_type = parsed_data['type']
        data = parsed_data['data']
        timestamp = parsed_data['timestamp']
        
        # Get or create file for this data type
        file_path = self._get_file_path(data_type)
        
        try:
            # Ensure file exists and has proper headers
            if file_path not in self.current_files:
                self._create_file(file_path, data_type)
                
            # Write data to CSV
            self._write_csv_row(file_path, data_type, data, timestamp)
            self.logged_records += 1
            
            # Check if file needs rotation
            if self._should_rotate_file(file_path):
                self._rotate_file(file_path, data_type)
                
        except Exception as e:
            self.logger.error(f"Error writing data to {file_path}: {e}")
            
    def _get_file_path(self, data_type: str) -> Path:
        """Get current file path for data type."""
        if data_type not in self.current_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp}.csv"
            self.current_files[data_type] = self.log_directory / filename
            
        return self.current_files[data_type]
        
    def _create_file(self, file_path: Path, data_type: str):
        """Create new CSV file with headers."""
        try:
            # Create file handle
            file_handle = open(file_path, 'w', newline='', encoding='utf-8')
            self.file_handles[file_path] = file_handle
            
            # Create CSV writer
            writer = csv.writer(file_handle)
            self.csv_writers[file_path] = writer
            
            # Write headers based on data type
            headers = self._get_headers(data_type)
            writer.writerow(headers)
            file_handle.flush()
            
            self.logger.info(f"Created log file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating file {file_path}: {e}")
            
    def _get_headers(self, data_type: str) -> List[str]:
        """Get CSV headers for data type."""
        if data_type == 'RAW_DATA':
            return ['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        elif data_type == 'ANGLES':
            return ['timestamp', 'altitude', 'azimuth', 'zenith', 'state']
        elif data_type == 'QUAT':
            return ['timestamp', 'qx', 'qy', 'qz', 'qw']
        elif data_type == 'GYRO_BIAS_MDI':
            return ['timestamp', 'bias_x', 'bias_y', 'bias_z']
        elif data_type.startswith('ANGLES_'):
            return ['timestamp', 'pitch', 'yaw', 'roll']
        else:
            return ['timestamp', 'data']
            
    def _write_csv_row(self, file_path: Path, data_type: str, data: Any, timestamp: float):
        """Write a single row to CSV file."""
        if file_path not in self.csv_writers:
            return
            
        writer = self.csv_writers[file_path]
        
        try:
            if data_type == 'RAW_DATA':
                row = [timestamp, data.acc_x, data.acc_y, data.acc_z, 
                      data.gyro_x, data.gyro_y, data.gyro_z]
            elif data_type == 'ANGLES':
                row = [timestamp, data.altitude, data.azimuth, data.zenith, data.state]
            elif data_type == 'QUAT':
                row = [timestamp, data.qx, data.qy, data.qz, data.qw]
            elif data_type == 'GYRO_BIAS_MDI':
                row = [timestamp, data.bias_x, data.bias_y, data.bias_z]
            elif data_type.startswith('ANGLES_'):
                row = [timestamp, data.pitch, data.yaw, data.roll]
            else:
                row = [timestamp, str(data)]
                
            writer.writerow(row)
            self.file_handles[file_path].flush()
            
        except Exception as e:
            self.logger.error(f"Error writing CSV row: {e}")
            
    def _should_rotate_file(self, file_path: Path) -> bool:
        """Check if file should be rotated."""
        try:
            return file_path.stat().st_size > self.max_file_size
        except:
            return False
            
    def _rotate_file(self, file_path: Path, data_type: str):
        """Rotate file to new name."""
        try:
            # Close current file
            if file_path in self.file_handles:
                self.file_handles[file_path].close()
                del self.file_handles[file_path]
            if file_path in self.csv_writers:
                del self.csv_writers[file_path]
                
            # Create new file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{data_type}_{timestamp}.csv"
            new_file_path = self.log_directory / new_filename
            
            self.current_files[data_type] = new_file_path
            self._create_file(new_file_path, data_type)
            
            self.logger.info(f"Rotated file: {file_path} -> {new_file_path}")
            
        except Exception as e:
            self.logger.error(f"Error rotating file: {e}")
            
    def get_log_files(self, data_type: Optional[str] = None) -> List[Path]:
        """
        Get list of log files.
        
        Args:
            data_type: Specific data type to filter by
            
        Returns:
            List of log file paths
        """
        pattern = f"{data_type}_*.csv" if data_type else "*.csv"
        return list(self.log_directory.glob(pattern))
        
    def cleanup_old_files(self):
        """Remove old log files based on retention policy."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            removed_count = 0
            
            for file_path in self.log_directory.glob("*.csv"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_path.unlink()
                    removed_count += 1
                    
            if removed_count > 0:
                self.logger.info(f"Cleaned up {removed_count} old log files")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old files: {e}")
            
    def export_data_range(self, data_type: str, start_time: float, end_time: float, 
                         output_file: str) -> bool:
        """
        Export data from a specific time range.
        
        Args:
            data_type: Type of data to export
            start_time: Start timestamp
            end_time: End timestamp
            output_file: Output file path
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            log_files = self.get_log_files(data_type)
            if not log_files:
                return False
                
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.writer(outfile)
                headers_written = False
                
                for log_file in sorted(log_files):
                    with open(log_file, 'r', encoding='utf-8') as infile:
                        reader = csv.reader(infile)
                        
                        # Write headers only once
                        if not headers_written:
                            headers = next(reader)
                            writer.writerow(headers)
                            headers_written = True
                        else:
                            next(reader)  # Skip headers
                            
                        # Filter data by time range
                        for row in reader:
                            if not row:
                                continue
                            try:
                                timestamp = float(row[0])
                                if start_time <= timestamp <= end_time:
                                    writer.writerow(row)
                            except (ValueError, IndexError):
                                continue
                                
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting data range: {e}")
            return False
            
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            'logged_records': self.logged_records,
            'active_files': len(self.current_files),
            'log_directory': str(self.log_directory),
            'max_file_size': self.max_file_size,
            'retention_days': self.retention_days
        }
        
        # Add file statistics
        total_files = 0
        total_size = 0
        
        for file_path in self.log_directory.glob("*.csv"):
            total_files += 1
            total_size += file_path.stat().st_size
            
        stats['total_files'] = total_files
        stats['total_size_mb'] = total_size / (1024 * 1024)
        
        return stats
        
    def set_log_directory(self, directory: str):
        """Set new log directory."""
        new_dir = Path(directory)
        new_dir.mkdir(parents=True, exist_ok=True)
        self.log_directory = new_dir
        self.logger.info(f"Log directory changed to: {directory}")
        
    def set_max_file_size(self, size_mb: int):
        """Set maximum file size in MB."""
        self.max_file_size = size_mb * 1024 * 1024
        self.logger.info(f"Max file size set to: {size_mb} MB")
        
    def set_retention_days(self, days: int):
        """Set retention period in days."""
        self.retention_days = days
        self.logger.info(f"Retention period set to: {days} days")
        
    def flush_all_files(self):
        """Flush all open files."""
        for file_handle in self.file_handles.values():
            try:
                file_handle.flush()
            except Exception as e:
                self.logger.error(f"Error flushing file: {e}")
                
    def close_all_files(self):
        """Close all open files."""
        for file_handle in self.file_handles.values():
            try:
                file_handle.close()
            except Exception as e:
                self.logger.error(f"Error closing file: {e}")
                
        self.file_handles.clear()
        self.csv_writers.clear()
        self.current_files.clear()
        
    def __del__(self):
        """Cleanup on destruction."""
        self.stop_logging_thread()
        self.close_all_files()
