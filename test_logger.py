#!/usr/bin/env python3
"""
Test script for the structured logging system

This script demonstrates the usage of the logging infrastructure
and validates that all components work correctly.
"""

import time
import numpy as np
from src.utils.logger import get_logger, log_execution_time, LogContext, setup_logger


def test_basic_logging():
    """Test basic logging functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic Logging Levels")
    print("="*60)
    
    logger = get_logger(__name__)
    
    logger.debug("This is a DEBUG message - detailed information")
    logger.info("This is an INFO message - general information")
    logger.warning("This is a WARNING message - warning about something")
    logger.error("This is an ERROR message - error occurred")
    logger.critical("This is a CRITICAL message - critical system error")
    
    print("✓ Basic logging test completed")


def test_component_logging():
    """Test logging from different components"""
    print("\n" + "="*60)
    print("TEST 2: Component-based Logging")
    print("="*60)
    
    # Simulate different components
    camera_logger = get_logger('camera_module')
    detector_logger = get_logger('car_detector')
    controller_logger = get_logger('traffic_controller')
    
    camera_logger.info("Camera initialized successfully")
    detector_logger.info("ML model loaded: TFLite optimized")
    controller_logger.info("Traffic light controller started")
    
    camera_logger.debug("Camera resolution: 640x480")
    detector_logger.debug("Detection threshold: 0.5")
    controller_logger.warning("High traffic detected on route A")
    
    print("✓ Component logging test completed")


@log_execution_time
def simulated_frame_processing(frame_id: int):
    """Simulate frame processing with performance logging"""
    time.sleep(0.05)  # Simulate processing time
    return f"Frame {frame_id} processed"


def test_performance_logging():
    """Test performance logging decorator"""
    print("\n" + "="*60)
    print("TEST 3: Performance Logging")
    print("="*60)
    
    logger = get_logger(__name__)
    
    for i in range(3):
        result = simulated_frame_processing(i)
        logger.info(result)
    
    print("✓ Performance logging test completed")
    print("  Check logs/performance.log for timing details")


def test_context_manager():
    """Test LogContext context manager"""
    print("\n" + "="*60)
    print("TEST 4: Context Manager Logging")
    print("="*60)
    
    logger = get_logger(__name__)
    
    with LogContext(logger, "Processing batch of frames", log_args=True, batch_size=10):
        time.sleep(0.1)
        logger.info("Processing frame 1")
        logger.info("Processing frame 2")
        logger.info("Processing frame 3")
    
    print("✓ Context manager test completed")


def test_error_logging():
    """Test error logging with exception handling"""
    print("\n" + "="*60)
    print("TEST 5: Error Logging and Exception Handling")
    print("="*60)
    
    logger = get_logger(__name__)
    
    try:
        # Simulate an error condition
        logger.info("Attempting risky operation...")
        result = 10 / 0  # This will raise ZeroDivisionError
    except ZeroDivisionError as e:
        logger.error(f"Operation failed: {e}", exc_info=True)
        logger.warning("Falling back to safe default")
    
    print("✓ Error logging test completed")
    print("  Check logs/errors.log for error details")


def test_structured_data():
    """Test logging with structured data"""
    print("\n" + "="*60)
    print("TEST 6: Structured Data Logging")
    print("="*60)
    
    logger = get_logger('detection')
    
    # Simulate car detection results
    detection_results = {
        'camera': 'A',
        'car_count': 3,
        'confidence': [0.95, 0.87, 0.92],
        'processing_time': 0.045
    }
    
    logger.info(f"Detection completed: {detection_results}")
    
    # Simulate traffic decision
    logger.info(
        f"Traffic decision: Camera A={detection_results['car_count']} cars, "
        f"avg_confidence={np.mean(detection_results['confidence']):.2f}"
    )
    
    print("✓ Structured data logging test completed")


def test_custom_log_levels():
    """Test custom log level configuration"""
    print("\n" + "="*60)
    print("TEST 7: Custom Log Level Configuration")
    print("="*60)
    
    # Create logger with DEBUG level
    debug_logger = setup_logger('debug_component', level='DEBUG')
    debug_logger.debug("This DEBUG message should appear")
    debug_logger.info("This INFO message should appear")
    
    # Create logger with WARNING level
    warn_logger = setup_logger('warning_component', level='WARNING')
    warn_logger.debug("This DEBUG message should NOT appear in console")
    warn_logger.info("This INFO message should NOT appear in console")
    warn_logger.warning("This WARNING message SHOULD appear")
    
    print("✓ Custom log level test completed")


def test_multi_threaded_logging():
    """Test thread-safe logging"""
    print("\n" + "="*60)
    print("TEST 8: Multi-threaded Logging")
    print("="*60)
    
    import threading
    
    logger = get_logger('threading_test')
    
    def worker(worker_id: int):
        for i in range(3):
            logger.info(f"Worker {worker_id}: Processing item {i}")
            time.sleep(0.01)
    
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print("✓ Multi-threaded logging test completed")


def main():
    """Run all logging tests"""
    print("\n" + "#"*60)
    print("# Smart Traffic Light - Logging System Test Suite")
    print("#"*60)
    
    try:
        test_basic_logging()
        test_component_logging()
        test_performance_logging()
        test_context_manager()
        test_error_logging()
        test_structured_data()
        test_custom_log_levels()
        test_multi_threaded_logging()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("="*60)
        print("\nLog files created:")
        print("  - logs/traffic_light.log (main log)")
        print("  - logs/errors.log (errors only)")
        print("  - logs/performance.log (timing metrics)")
        print("\nYou can now:")
        print("  1. Check the log files in the 'logs/' directory")
        print("  2. Integrate the logger into existing modules")
        print("  3. Replace print() statements with proper logging")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())