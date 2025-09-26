#!/usr/bin/env python3
"""
Performance test script for ISM330DHCX Data Visualization Tool
Tests the 52Hz data streaming capability
"""

import time
import threading
import queue
from data_parser import DataParser
from data_logger import DataLogger
from visualization import VisualizationEngine
import tkinter as tk
from datetime import datetime

def generate_test_data(rate_hz=52):
    """Generate test data at specified rate."""
    parser = DataParser()
    logger = DataLogger()
    
    # Test data samples
    test_lines = [
        "[{:.2f}s] RAW_DATA,1000,-77.0,957.0,-174.0,0.49,-0.769,-0.56",
        "[{:.2f}s] ANGLES,1001,45.5,120.3,89.7,MONITORING",
        "[{:.2f}s] QUAT,1002,0.1,0.2,0.3,0.94",
        "[{:.2f}s] GYRO_BIAS_MDI,1003,0.01,-0.02,0.03",
        "[{:.2f}s] ANGLES_DI,1004,10.5,-20.3,30.7",
        "[{:.2f}s] ANGLES_SI,1005,11.5,-21.3,31.7",
        "[{:.2f}s] ANGLES_CO,1006,12.5,-22.3,32.7",
        "[{:.2f}s] ANGLES_FU,1007,13.5,-23.3,33.7"
    ]
    
    start_time = time.time()
    sample_count = 0
    parse_times = []
    log_times = []
    
    print(f"Starting performance test at {rate_hz}Hz...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            loop_start = time.time()
            current_time = loop_start - start_time
            
            # Generate and parse data
            for i, template in enumerate(test_lines):
                line = template.format(current_time + i * 0.001)
                
                # Measure parse time
                parse_start = time.time()
                parsed = parser.parse_line(line)
                parse_time = time.time() - parse_start
                parse_times.append(parse_time)
                
                if parsed:
                    # Measure log time
                    log_start = time.time()
                    logger.log_data(parsed)
                    log_time = time.time() - log_start
                    log_times.append(log_time)
                
                sample_count += 1
            
            # Calculate statistics every second
            if sample_count % (rate_hz * 8) == 0:  # 8 data types per cycle
                avg_parse = sum(parse_times) / len(parse_times) * 1000 if parse_times else 0
                avg_log = sum(log_times) / len(log_times) * 1000 if log_times else 0
                actual_rate = sample_count / (time.time() - start_time)
                
                print(f"Time: {current_time:.1f}s | Samples: {sample_count} | Rate: {actual_rate:.1f}Hz")
                print(f"  Parse: {avg_parse:.3f}ms | Log: {avg_log:.3f}ms")
                print(f"  Queue size: {logger.data_queue.qsize()}")
                print()
                
                # Reset timing arrays to prevent memory growth
                parse_times = parse_times[-100:]
                log_times = log_times[-100:]
            
            # Sleep to maintain rate
            sleep_time = (1.0 / rate_hz) - (time.time() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nTest stopped")
        
    # Final statistics
    total_time = time.time() - start_time
    final_rate = sample_count / total_time
    
    print(f"\nFinal Statistics:")
    print(f"  Total samples: {sample_count}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average rate: {final_rate:.2f}Hz")
    print(f"  Target rate: {rate_hz}Hz")
    print(f"  Performance: {(final_rate/rate_hz)*100:.1f}%")
    
    # Cleanup
    logger.stop_logging_thread()
    logger.close_all_files()

def test_visualization_performance():
    """Test visualization engine performance."""
    print("Testing visualization engine performance...")
    
    root = tk.Tk()
    root.withdraw()  # Hide the window for testing
    
    viz = VisualizationEngine(root)
    parser = DataParser()
    
    # Generate test data
    test_lines = [
        "[{:.2f}s] RAW_DATA,1000,-77.0,957.0,-174.0,0.49,-0.769,-0.56",
        "[{:.2f}s] ANGLES,1001,45.5,120.3,89.7,MONITORING",
    ]
    
    start_time = time.time()
    update_count = 0
    
    # Test for 5 seconds
    while time.time() - start_time < 5:
        current_time = time.time() - start_time
        
        for template in test_lines:
            line = template.format(current_time)
            parsed = parser.parse_line(line)
            if parsed:
                viz.update_data(parsed)
                update_count += 1
        
        # Process pending data
        viz.process_pending_data()
        root.update_idletasks()
        
        time.sleep(0.019)  # ~52Hz
    
    rate = update_count / 5.0
    print(f"Visualization update rate: {rate:.1f}Hz")
    print(f"Queue size: {viz.update_queue.qsize()}")
    
    root.destroy()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "viz":
        test_visualization_performance()
    else:
        # Test data generation and logging at 52Hz
        generate_test_data(52)