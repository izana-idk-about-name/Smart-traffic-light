#!/usr/bin/env python3
"""
Test script to verify car detection model can distinguish
between images WITH cars and WITHOUT cars (negative samples)
"""

import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.custom_car_trainer import LightweightCarTrainer

def test_model_on_samples():
    """Test the trained model on positive and negative samples"""
    print("🧪 Testing Car Detection Model")
    print("=" * 60)
    
    # Load model
    trainer = LightweightCarTrainer()
    if not trainer.load_model('src/models/custom_car_detector.yml'):
        print("❌ Failed to load model!")
        return False
    
    print("✅ Model loaded successfully\n")
    
    # Test on positive samples (images WITH cars)
    print("🚗 Testing POSITIVE samples (images WITH cars):")
    print("-" * 60)
    
    car_dir = Path('data/kitti/images_real/toy_car')
    car_images = list(car_dir.glob('*.jpg'))[:5]  # Test first 5
    
    if not car_images:
        car_images = list(car_dir.glob('*.png'))[:5]
    
    correct_positive = 0
    total_positive = len(car_images)
    
    for img_path in car_images:
        prediction, confidence = trainer.predict(str(img_path))
        result = "✅ CORRECT" if prediction == 1 else "❌ WRONG"
        print(f"{result} | {img_path.name[:30]:30s} | Pred: {prediction} | Conf: {confidence:.3f}")
        if prediction == 1:
            correct_positive += 1
    
    # Test on negative samples (images WITHOUT cars)
    print("\n🚫 Testing NEGATIVE samples (images WITHOUT cars):")
    print("-" * 60)
    
    neg_dir = Path('data/negativo')
    neg_images = list(neg_dir.glob('*.webp')) + list(neg_dir.glob('*.avif'))
    
    correct_negative = 0
    total_negative = len(neg_images)
    
    for img_path in neg_images:
        prediction, confidence = trainer.predict(str(img_path))
        result = "✅ CORRECT" if prediction == 0 else "❌ WRONG"
        print(f"{result} | {img_path.name[:30]:30s} | Pred: {prediction} | Conf: {confidence:.3f}")
        if prediction == 0:
            correct_negative += 1
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print("=" * 60)
    
    if total_positive > 0:
        accuracy_positive = (correct_positive / total_positive) * 100
        print(f"Positive samples (WITH cars):    {correct_positive}/{total_positive} correct ({accuracy_positive:.1f}%)")
    
    if total_negative > 0:
        accuracy_negative = (correct_negative / total_negative) * 100
        print(f"Negative samples (WITHOUT cars): {correct_negative}/{total_negative} correct ({accuracy_negative:.1f}%)")
    
    total = total_positive + total_negative
    correct = correct_positive + correct_negative
    
    if total > 0:
        overall_accuracy = (correct / total) * 100
        print(f"\n🎯 Overall Accuracy: {correct}/{total} ({overall_accuracy:.1f}%)")
        
        if overall_accuracy >= 80:
            print("\n✅ Model performs well on distinguishing cars vs no-cars!")
            return True
        else:
            print("\n⚠️  Model accuracy is below 80%, may need more training data")
            return False
    
    return False

if __name__ == "__main__":
    success = test_model_on_samples()
    sys.exit(0 if success else 1)