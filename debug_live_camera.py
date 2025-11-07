#!/usr/bin/env python3
"""
Debug script to test live camera detection
"""

import cv2
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models.car_identify import create_car_identifier

def test_live_camera():
    """Test detection on live camera feed"""
    print("🎥 Testing Live Camera Detection")
    print("=" * 60)
    print("Point your camera at:")
    print("  1. An EMPTY scene (wall, floor, ceiling)")
    print("  2. A scene WITH a toy car")
    print("\nPress 'q' to quit\n")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Create identifier
    identifier = create_car_identifier('desktop', use_ml=False, use_custom_model=True, use_tflite=False)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        
        # Test detection every 10 frames
        if frame_count % 10 == 0:
            # Save frame for debugging
            debug_path = f"/tmp/debug_frame_{frame_count}.jpg"
            cv2.imwrite(debug_path, frame)
            
            # Count cars
            car_count = identifier.count_cars(frame)
            
            # Show result
            print(f"Frame {frame_count}: Detected {car_count} car(s) | Debug frame saved: {debug_path}")
            
            # Draw on frame
            color = (0, 255, 0) if car_count > 0 else (0, 0, 255)
            text = f"Cars: {car_count}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show frame
        cv2.imshow('Live Camera Detection', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test complete")

if __name__ == "__main__":
    test_live_camera()