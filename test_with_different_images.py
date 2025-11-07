#!/usr/bin/env python3
"""
Test the system with one camera showing cars and another showing no cars
"""

import cv2
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment to use different test images
os.environ['MODO'] = 'development'

from src.models.car_identify import create_car_identifier

def test_differentiation():
    """Test that the system can differentiate between images with and without cars"""
    print("🧪 Testing Car Detection Differentiation")
    print("=" * 60)
    
    # Create identifier
    identifier = create_car_identifier('desktop', use_ml=False, use_custom_model=True, use_tflite=False)
    
    # Test with car image
    car_image_path = "data/kitti/images_real/toy_car/000001.jpg"
    if Path(car_image_path).exists():
        frame_with_car = cv2.imread(car_image_path)
        count_with_car = identifier.count_cars(frame_with_car)
        print(f"\n📸 Image WITH car: {car_image_path}")
        print(f"   Detected: {count_with_car} car(s)")
    else:
        print(f"❌ Car image not found: {car_image_path}")
        return False
    
    # Test with no-car image
    no_car_image_path = "data/negativo/sexta-sem-carro.webp"
    if Path(no_car_image_path).exists():
        frame_without_car = cv2.imread(no_car_image_path)
        count_without_car = identifier.count_cars(frame_without_car)
        print(f"\n📸 Image WITHOUT car: {no_car_image_path}")
        print(f"   Detected: {count_without_car} car(s)")
    else:
        print(f"❌ No-car image not found: {no_car_image_path}")
        return False
    
    # Results
    print("\n" + "=" * 60)
    print("📊 COMPARISON RESULTS:")
    print("=" * 60)
    print(f"Image WITH car:    {count_with_car} car(s) detected")
    print(f"Image WITHOUT car: {count_without_car} car(s) detected")
    
    if count_with_car > 0 and count_without_car == 0:
        print("\n✅ SUCCESS: System correctly distinguishes cars from no-cars!")
        print("\n💡 Traffic Light Logic:")
        print(f"   • If Camera A shows {count_with_car} car(s) and Camera B shows {count_without_car} car(s)")
        print(f"   • Decision: Open GREEN light for Camera A (more traffic)")
        return True
    elif count_with_car == 0 and count_without_car > 0:
        print("\n❌ ERROR: Detection is inverted!")
        return False
    elif count_with_car == 0 and count_without_car == 0:
        print("\n❌ ERROR: Not detecting any cars!")
        return False
    else:
        print("\n❌ ERROR: Both detected as having cars (no differentiation)!")
        return False

if __name__ == "__main__":
    success = test_differentiation()
    
    if success:
        print("\n🎯 To test with real cameras:")
        print("   1. Point Camera A at a scene WITH cars (or toy car)")
        print("   2. Point Camera B at an empty scene (wall, floor, ceiling)")
        print("   3. Run: MODO=development python3 main.py")
        print("   4. The system should prioritize Camera A (with cars)")
    
    sys.exit(0 if success else 1)