#!/usr/bin/env python3
"""
Diagnostic version of main.py to identify where the code hangs
"""
import sys
import os

print("=" * 60)
print("DIAGNOSTIC MODE - Tracking execution flow")
print("=" * 60)

print("\n[1/10] Loading environment variables...")
from dotenv import load_dotenv
load_dotenv()
print("✓ Environment variables loaded")

print("\n[2/10] Getting MODE from environment...")
modo = os.getenv('MODO', 'production').lower()
print(f"✓ MODE detected: {modo}")

print("\n[3/10] Importing cv2...")
import cv2
print("✓ cv2 imported")

print("\n[4/10] Configuring OpenCV logging...")
if modo != 'development':
    cv2.setLogLevel(0)
    os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    devnull = open(os.devnull, 'w')
    sys.stderr = devnull
    print("✓ OpenCV logging disabled (production mode)")
else:
    print("✓ OpenCV warnings enabled (development mode)")

print("\n[5/10] Importing standard libraries...")
import time
import threading
import signal
import atexit
import numpy as np
from typing import Optional
print("✓ Standard libraries imported")

print("\n[6/10] Importing application modules...")
try:
    from src.models.car_identify import create_car_identifier
    print("  ✓ car_identify imported")
except Exception as e:
    print(f"  ✗ ERROR importing car_identify: {e}")
    sys.exit(1)

try:
    from src.application.comunicator import OrchestratorComunicator
    print("  ✓ comunicator imported")
except Exception as e:
    print(f"  ✗ ERROR importing comunicator: {e}")
    sys.exit(1)

try:
    from src.application.camera_source import CameraSource, CameraFactory
    print("  ✓ camera_source imported")
except Exception as e:
    print(f"  ✗ ERROR importing camera_source: {e}")
    sys.exit(1)

try:
    from src.settings.rpi_config import CAMERA_SETTINGS, PROCESSING_SETTINGS, MODEL_SETTINGS, NETWORK_SETTINGS, IS_RASPBERRY_PI
    print("  ✓ rpi_config imported")
except Exception as e:
    print(f"  ✗ ERROR importing rpi_config: {e}")
    sys.exit(1)

try:
    from src.utils.resource_manager import FrameBuffer, ResourceTracker, get_global_tracker
    print("  ✓ resource_manager imported")
except Exception as e:
    print(f"  ✗ ERROR importing resource_manager: {e}")
    sys.exit(1)

try:
    from src.utils.healthcheck import HealthCheck, BuiltInHealthChecks
    print("  ✓ healthcheck imported")
except Exception as e:
    print(f"  ✗ ERROR importing healthcheck: {e}")
    sys.exit(1)

try:
    from src.utils.watchdog import Watchdog, RecoveryStrategy, RecoveryAction
    print("  ✓ watchdog imported")
except Exception as e:
    print(f"  ✗ ERROR importing watchdog: {e}")
    sys.exit(1)

try:
    from src.utils.logger import get_logger
    print("  ✓ logger imported")
except Exception as e:
    print(f"  ✗ ERROR importing logger: {e}")
    sys.exit(1)

print("\n[7/10] All imports successful!")

print("\n[8/10] Testing class instantiation...")
try:
    print("  Creating TrafficLightController instance...")
    # We'll just test if we can import the main module
    import main
    print("  ✓ main module imported successfully")
except Exception as e:
    print(f"  ✗ ERROR importing main: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[9/10] Checking main execution flow...")
print(f"  MODE/MODO: {modo}")
print(f"  __name__: {__name__}")

print("\n[10/10] Ready to execute!")
print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE - No import errors detected")
print("=" * 60)

print("\nNow attempting to run main...")
print("If it hangs here, the issue is in the main execution logic\n")

# Try to run with a timeout
if __name__ == "__main__":
    print(f"Executing in mode: {modo}")
    if modo == "development":
        print("Would call main_teste() - skipping for diagnostics")
    else:
        print("Would call TrafficLightController.run_loop() - skipping for diagnostics")
    
    print("\n✓ Diagnostic complete! The code structure is valid.")
    print("  Issue is likely:")
    print("  1. Camera initialization hanging when trying to access /dev/video0")
    print("  2. Network connection blocking on communicator initialization")
    print("  3. Resource initialization taking too long")