#!/usr/bin/env python3
"""
Test real-time detection with the updated custom model
"""

import cv2
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.car_identify import create_car_identifier

def test_realtime_detection():
    """Test detection on various images"""
    print("🧪 Testing Real-Time Detection with Custom SVM Model")
    print("=" * 60)
    
    # Create identifier with custom model enabled
    print("Loading car identifier with custom model...")
    identifier = create_car_identifier('desktop', use_ml=False, use_custom_model=True, use_tflite=False)
    
    # Test images
    test_cases = [
        # Positive samples (with cars)
        ("data/kitti/images_real/toy_car/000001.jpg", True, "Toy car image 1"),
        ("data/kitti/images_real/toy_car/000031.jpg", True, "Toy car image 2"),
        ("data/kitti/images_real/toy_f1/000002.jpg", True, "F1 toy car"),
        
        # Negative samples (without cars)
        ("data/negativo/sexta-sem-carro.webp", False, "Empty street"),
        ("data/negativo/estrada-em-dois-sentidos-6984079.webp", False, "Empty road"),
        ("data/negativo/vista-superior-do-estacionamen", False, "Empty parking"),
    ]
    
    print("\n🔍 Testing Detection on Different Images:")
    print("-" * 60)
    
    correct = 0
    total = 0
    
    for img_path, should_have_car, description in test_cases:
        # Check if file exists
        full_paths = [
            img_path,
            img_path + ".webp",
            img_path + ".avif"
        ]
        
        found_path = None
        for p in full_paths:
            if Path(p).exists():
                found_path = p
                break
        
        if not found_path:
            print(f"⚠️  File not found: {img_path}")
            continue
        
        # Load image
        frame = cv2.imread(found_path)
        if frame is None:
            print(f"⚠️  Could not load: {found_path}")
            continue
        
        # Detect cars
        car_count = identifier.count_cars(frame)
        
        # Check result
        has_car = car_count > 0
        expected = "CAR" if should_have_car else "NO CAR"
        detected = "CAR" if has_car else "NO CAR"
        
        is_correct = (has_car == should_have_car)
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        
        print(f"{status} | {description:25s} | Expected: {expected:6s} | Detected: {detected:6s} (count: {car_count})")
        
        if is_correct:
            correct += 1
        total += 1
    
    # Results
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print("=" * 60)
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
        
        if accuracy >= 80:
            print("\n✅ Model is working correctly!")
            return True
        else:
            print("\n⚠️  Model needs improvement")
            return False
    else:
        print("❌ No tests completed")
        return False

if __name__ == "__main__":
    success = test_realtime_detection()
    sys.exit(0 if success else 1)