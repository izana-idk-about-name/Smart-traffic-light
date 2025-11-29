#!/usr/bin/env python3
"""
TFLite Car Detector for optimized inference on Raspberry Pi
Uses the advanced trained model with quantization and pruning
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path
from typing import Tuple, List, Optional

class TFLiteCarDetector:
    """
    Lightweight car detector using TFLite int8 quantized model
    Optimized for Raspberry Pi performance
    """

    def __init__(self, model_path='src/models/car_detector_int8.tflite'):
        self.model_path = Path(model_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.input_shape = (224, 224, 3)  # MobileNet input size
        self.labels = ['background', 'car']

        # Performance tracking
        self.inference_times = []
        self.confidence_threshold = 0.6  # Higher threshold for quantized model

        self._load_model()

    def _load_model(self):
        """Load TFLite model and allocate tensors"""
        try:
            import tensorflow as tf

            if not self.model_path.exists():
                print(f"❌ TFLite model not found: {self.model_path}")
                print("💡 Please train the advanced model first using advanced_car_trainer.py")
                return False

            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()

            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            print("✅ TFLite car detector loaded successfully!")
            print(f"📏 Model size: {self.model_path.stat().st_size / 1024:.1f} KB")
            print(f"🔢 Input shape: {self.input_details[0]['shape']}")
            print(f"🎯 Output shape: {self.output_details[0]['shape']}")

            return True

        except ImportError:
            print("❌ TensorFlow not installed. Install with: pip install tensorflow")
            return False
        except Exception as e:
            print(f"❌ Error loading TFLite model: {e}")
            return False

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for TFLite model inference"""
        # Resize to model input size
        resized = cv2.resize(image, self.input_shape[:2])

        # Convert BGR to RGB
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        resized = resized.astype(np.float32) / 255.0

        # Handle quantization
        if self.input_details[0]['dtype'] == np.int8:
            # Quantize for int8 model
            scale, zero_point = self.input_details[0]['quantization']
            resized = resized / scale + zero_point
            resized = resized.astype(np.int8)

        # Add batch dimension
        resized = np.expand_dims(resized, axis=0)

        return resized

    def detect_cars(self, image: np.ndarray) -> List[dict]:
        """
        Detect cars in image using TFLite model

        Args:
            image: Input image (BGR format)

        Returns:
            List of detection dictionaries with format:
            [{'class': str, 'confidence': float, 'bbox': [x, y, w, h]}]
        """
        if self.interpreter is None:
            return []

        try:
            start_time = time.time()

            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], processed_image)
            self.interpreter.invoke()

            # Get output
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Handle dequantization
            if self.output_details[0]['dtype'] == np.int8:
                scale, zero_point = self.output_details[0]['quantization']
                output = (output.astype(np.float32) - zero_point) * scale

            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # Process predictions
            predictions = output[0]  # Remove batch dimension

            # Get car probability (class 1)
            car_confidence = predictions[1] if len(predictions) > 1 else predictions[0]

            detections = []

            # Apply confidence threshold
            if car_confidence > self.confidence_threshold:
                # For classification model, create full-frame detection
                height, width = image.shape[:2]
                detection = {
                    'class': 'car',
                    'confidence': float(car_confidence),
                    'bbox': [0, 0, width, height]  # Full frame detection
                }
                detections.append(detection)

            return detections

        except Exception as e:
            print(f"❌ Inference error: {e}")
            return []

    def get_average_inference_time(self) -> float:
        """Get average inference time in milliseconds"""
        if not self.inference_times:
            return 0.0
        return np.mean(self.inference_times) * 1000

    def reset_metrics(self):
        """Reset performance metrics"""
        self.inference_times = []

def demo_tflite_detection():
    """Demo function for TFLite car detection"""
    print("🔴 TFLite Car Detection Demo")
    print("=" * 40)

    detector = TFLiteCarDetector()

    if detector.interpreter is None:
        print("❌ Model not loaded. Exiting demo.")
        return

    # Try to open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available. Using test image.")

        # Create test image with synthetic car-like shape
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (200, 150), (440, 330), (0, 0, 255), -1)  # Red rectangle as "car"

        # Run detection
        detections = detector.detect_cars(test_image)

        print(f"🧪 Test detection results: {len(detections)} cars found")

        if detections:
            for i, det in enumerate(detections):
                print(f"  Car {i+1}: {det['confidence']:.2f} confidence")

        # Save test result
        cv2.imwrite("/tmp/tflite_test_result.jpg", test_image)
        print("📸 Test result saved to /tmp/tflite_test_result.jpg")

        return

    print("📹 Starting camera demo... Press 'q' to quit")

    frame_count = 0
    car_detected_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        detections = detector.detect_cars(frame)
        frame_count += 1

        if detections:
            car_detected_frames += 1

        # Draw detections
        display_frame = frame.copy()
        for det in detections:
            x, y, w, h = det['bbox']
            conf = det['confidence']
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Car: {conf:.2f}",
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add performance info
        avg_time = detector.get_average_inference_time()
        cv2.putText(display_frame, f"TFLite Cars: {len(detections)} | Time: {avg_time:.1f}ms",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Save frame periodically (since we can't display GUI on headless)
        if frame_count % 30 == 0:  # Every 30 frames
            cv2.imwrite(f"/tmp/tflite_frame_{frame_count}.jpg", display_frame)
            print(f"📸 Frame {frame_count} saved | Cars: {len(detections)} | Avg time: {avg_time:.1f}ms")

        # Check for quit (simulate key press detection)
        if frame_count >= 300:  # Run for 10 seconds at 30fps
            break

    cap.release()

    detection_rate = car_detected_frames / frame_count * 100 if frame_count > 0 else 0
    print(".1f")
    print(".1f")
    print("✅ Demo completed!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_tflite_detection()
    else:
        # Test single inference
        detector = TFLiteCarDetector()

        if detector.interpreter is not None:
            # Create test image
            test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

            start_time = time.time()
            detections = detector.detect_cars(test_img)
            total_time = time.time() - start_time

            print("🧪 Single inference test:")
            print(f"   Detections: {len(detections)}")
            print(".1f")
        else:
            print("❌ TFLite model not available for testing")