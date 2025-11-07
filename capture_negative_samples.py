#!/usr/bin/env python3
"""
Capture negative samples (images WITHOUT cars) from camera
to improve model training
"""

import cv2
import sys
import os
from pathlib import Path
import time

def capture_negative_samples(num_samples=50):
    """Capture images without cars for training"""
    print("📸 Capturing Negative Samples (Images WITHOUT Cars)")
    print("=" * 60)
    print(f"We will capture {num_samples} images")
    print("\n⚠️  IMPORTANT:")
    print("  - Point camera at scenes WITHOUT cars")
    print("  - Examples: walls, floors, ceilings, furniture, plants")
    print("  - Move camera to capture VARIETY")
    print("\nPress 's' to save image, 'q' to quit\n")
    
    # Create output directory
    output_dir = Path('data/negativo_camera')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    saved_count = 0
    
    while saved_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Show frame
        display_frame = frame.copy()
        text = f"Captured: {saved_count}/{num_samples} | Press 's' to save, 'q' to quit"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Capture Negative Samples - NO CARS', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            # Save frame
            filename = f"no_car_{saved_count:04d}_{int(time.time())}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_count += 1
            print(f"✅ Saved {saved_count}/{num_samples}: {filename}")
            time.sleep(0.3)  # Small delay to avoid duplicates
            
        elif key == ord('q'):
            print("\n❌ Capture cancelled by user")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if saved_count >= num_samples:
        print(f"\n✅ Successfully captured {saved_count} negative samples!")
        print(f"📁 Saved to: {output_dir}")
        print("\n🔄 Next steps:")
        print("   1. Run: PYTHONPATH=/home/doorofhell/projetos/Smart-traffic-light python3 src/training/custom_car_trainer.py")
        print("   2. The model will be retrained with your new negative samples")
        print("   3. Test again with: python3 debug_live_camera.py")
        return True
    else:
        print(f"\n⚠️  Only captured {saved_count}/{num_samples} samples")
        return False

if __name__ == "__main__":
    # Default: capture 50 negative samples
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    capture_negative_samples(num_samples)