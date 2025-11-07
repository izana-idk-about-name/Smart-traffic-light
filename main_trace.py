#!/usr/bin/env python3
"""
Trace version - runs actual main.py code with detailed logging
"""
import sys
import os

# Add trace logging
original_print = print
def traced_print(*args, **kwargs):
    import time
    timestamp = time.strftime("%H:%M:%S")
    original_print(f"[{timestamp}]", *args, **kwargs)
    sys.stdout.flush()

print = traced_print

print("Starting main.py with trace logging...")

# Set mode to production to test the actual hanging scenario
os.environ['MODO'] = 'production'

print("Importing main module...")
import main

print("Creating TrafficLightController instance...")
try:
    controller = main.TrafficLightController(
        camera_a_dev='/dev/video0',
        camera_b_dev='/dev/video1',
        orchestrator_host='localhost',
        orchestrator_port=9000
    )
    print("✓ TrafficLightController created successfully")
    
    print("Calling initialize_cameras()...")
    result = controller.initialize_cameras()
    print(f"✓ initialize_cameras() returned: {result}")
    
    if result:
        print("Cameras initialized, would start run_loop() but stopping here for diagnostics")
    else:
        print("Camera initialization failed")
        
    print("Cleaning up...")
    controller.cleanup()
    print("✓ Cleanup complete")
    
except KeyboardInterrupt:
    print("\n✗ Interrupted by user")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nTrace complete!")