import cv2
import numpy as np
import time
import os
from pathlib import Path
from typing import Tuple, List, Optional
from src.settings.rpi_config import MODEL_SETTINGS
from src.training.custom_car_trainer import LightweightCarTrainer

class CarIdentifier:
    def __init__(self, optimize_for_rpi=True, use_ml=True, use_custom_model=False):
        """
        Initialize car identifier with AI/ML capabilities

        Args:
            optimize_for_rpi: Whether to optimize for Raspberry Pi performance
            use_ml: Whether to use ML model for detection
            use_custom_model: Whether to use custom trained SVM model instead of pre-trained
        """
        self.optimize_for_rpi = optimize_for_rpi
        self.use_ml = use_ml and MODEL_SETTINGS.get('use_ml_model', False)
        self.use_custom_model = use_custom_model

        # Initialize ML model
        self.ml_model = None
        self.labels = []
        self.model_loaded = False

        # Initialize custom SVM model
        self.custom_trainer = None
        self.custom_model_loaded = False

        if self.use_custom_model:
            self._load_custom_model()
            # Don't load ML model when using custom model - custom handles everything
        elif self.use_ml:
            self._load_ml_model()

        # Fallback: traditional CV background subtractor
        if optimize_for_rpi:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=MODEL_SETTINGS.get('background_history', 50),
                varThreshold=MODEL_SETTINGS.get('var_threshold', 30),
                detectShadows=False
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=MODEL_SETTINGS.get('background_history', 100),
                varThreshold=MODEL_SETTINGS.get('var_threshold', 40),
                detectShadows=True
            )

        # Kernel for morphological operations
        kernel_size = 3 if optimize_for_rpi else 5
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        self.ml_inference_count = 0
        self.ml_inference_time = 0
        
    def preprocess_frame(self, frame):
        """Preprocess frame for better detection and performance"""
        if self.optimize_for_rpi:
            # Resize frame for faster processing on RPi
            frame = cv2.resize(frame, (320, 240))
        
        # Convert to grayscale for background subtraction
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        return gray

    def _extract_cars_from_contours(self, contours, processed_frame, original_frame=None):
        """Extract car count and contour data from contours"""
        car_count = 0
        min_area = 200 if self.optimize_for_rpi else 500
        valid_contours = []

        # Calculate scale factors if original frame provided and frame was resized
        scale_x = 1.0
        scale_y = 1.0
        if original_frame is not None and self.optimize_for_rpi:
            scale_x = original_frame.shape[1] / processed_frame.shape[1]
            scale_y = original_frame.shape[0] / processed_frame.shape[0]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Additional filtering for better accuracy
                x, y, w, h = cv2.boundingRect(cnt)

                # Scale back if original frame provided
                if original_frame is not None and self.optimize_for_rpi:
                    x = int(x * scale_x)
                    y = int(y * scale_y)
                    w = int(w * scale_x)
                    h = int(h * scale_y)

                aspect_ratio = float(w) / h if h > 0 else 0

                # Filter based on aspect ratio (typical car shape)
                if 0.5 < aspect_ratio < 3.0:
                    car_count += 1
                    valid_contours.append((x, y, w, h))

        return car_count, valid_contours
    
    def count_cars(self, frame):
        """
        Counts cars using AI/ML model with CV fallback.
        Uses machine learning for accurate car detection when available.

        Args:
            frame: np.ndarray, image from camera
        Returns:
            int: number of cars detected
        """
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        if frame is None or frame.size == 0:
            if debug_mode:
                print("Debug: Invalid frame provided to count_cars")
            return 0

        if debug_mode:
            print("Debug: 🔄 Starting car counting process...")

        start_time = time.time()

        # Use new ML-based detection
        car_boxes = self.detect_cars(frame)
        car_count = len(car_boxes)

        processing_time = time.time() - start_time

        if debug_mode:
            print(f"Debug: ⏱️  Processing time: {processing_time:.3f}s")
            if car_count > 0:
                print(f"Debug: 🚗 Car detection successful: {car_count} cars found")
                for i, detection in enumerate(car_boxes):
                    print(f"Debug:   Car {i+1}: {detection.get('class', 'unknown')} "
                          f"(confidence: {detection.get('confidence', 0):.2f})")
            else:
                print(f"Debug: 🚫 No cars detected in this frame")

        # Update performance metrics
        self.frame_count += 1
        self.total_processing_time += processing_time

        return car_count
    
    def get_average_processing_time(self):
        """Get average processing time per frame"""
        if self.frame_count == 0:
            return 0
        return self.total_processing_time / self.frame_count
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.frame_count = 0
        self.total_processing_time = 0
    
    def visualize_detection(self, frame, show_contours=False):
        """
        Create visualization of car detection using AI/ML (for debugging)

        Args:
            frame: Input frame
            show_contours: Whether to draw bounding boxes

        Returns:
            tuple: (count, visualization_frame)
        """
        if frame is None or frame.size == 0:
            if os.getenv('MODO', '').lower() == 'development':
                print("Debug: Invalid frame provided to visualize_detection")
            return 0, frame

        # Create visualization frame
        vis_frame = frame.copy()

        # Use ML-based detection
        car_boxes = self.detect_cars(frame)
        car_count = len(car_boxes)

        # Draw detections
        for i, detection in enumerate(car_boxes):
            if show_contours:
                x, y, w, h = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class']

                # Color based on detection method
                color = (0, 255, 0) if 'cv' not in class_name.lower() else (255, 0, 0)  # Green for ML, Red for CV

                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, 2)

                # Label with class and confidence
                label = "02d"
                cv2.putText(vis_frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Add performance info
        avg_time = self.get_average_processing_time()
        method = "ML" if self.model_loaded else "CV"
        cv2.putText(vis_frame, f'{method} Cars: {car_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_frame, f'Avg Time: {avg_time:.3f}s', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Add ML-specific info if available
        if self.ml_inference_count > 0:
            ml_avg_time = self.ml_inference_time / self.ml_inference_count
            cv2.putText(vis_frame, f'ML Time: {ml_avg_time:.3f}s', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return car_count, vis_frame

    def _load_custom_model(self):
        """Load custom trained SVM model for car detection"""
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        try:
            self.custom_trainer = LightweightCarTrainer()
            model_path = 'src/models/custom_car_detector.yml'

            if self.custom_trainer.load_model(model_path):
                self.custom_model_loaded = True
                if debug_mode:
                    print("Debug: Custom SVM model loaded successfully for car detection")
            else:
                if debug_mode:
                    print("Debug: Custom model not found. Using fallback CV method.")
                self.custom_model_loaded = False

        except Exception as e:
            if debug_mode:
                print(f"Debug: Error loading custom model: {e}. Using fallback CV method.")
            self.custom_model_loaded = False

    def _load_ml_model(self):
        """Load ML model for car detection (silent by default)"""
        # Get debug mode from environment
        debug_mode = os.getenv('MODO', '').lower() == 'development'

        try:
            model_path = Path(MODEL_SETTINGS.get('model_path', 'src/models/ssd_mobilenet_v3_large_coco.pb'))
            config_path = Path(MODEL_SETTINGS.get('model_path', 'src/models/ssd_mobilenet_v3_large_coco.pbtxt').replace('.pb', '.pbtxt'))
            labels_path = Path(MODEL_SETTINGS.get('labels_path', 'src/models/coco_labels.txt'))

            # Check if model files exist
            if not model_path.exists() or not config_path.exists():
                if debug_mode:
                    print(f"Debug: ML model files not found. Using fallback CV method.")
                self.model_loaded = False
                return

            # Load OpenCV DNN model
            self.ml_model = cv2.dnn_DetectionModel(str(model_path), str(config_path))
            self.ml_model.setInputSize(320, 320)
            self.ml_model.setInputScale(1.0/127.5)
            self.ml_model.setInputMean((127.5, 127.5, 127.5))
            self.ml_model.setInputSwapRB(True)

            # Load labels
            if labels_path.exists():
                with open(labels_path, 'r') as f:
                    self.labels = [line.strip().split(': ')[-1] for line in f.readlines()]

            self.model_loaded = True
            if debug_mode:
                print("Debug: ML model loaded successfully for car detection")
                print(f"Debug: Available classes: {len(self.labels)}")

        except Exception as e:
            if debug_mode:
                print(f"Debug: Error loading ML model: {e}. Using fallback CV method.")
            self.model_loaded = False

    def _detect_with_custom_model(self, frame):
        """Detect cars using custom trained SVM model with fail-safe behavior"""
        if not self.custom_model_loaded or self.custom_trainer is None:
            return []

        try:
            start_time = time.time()

            # Save frame temporarily for SVM prediction
            temp_path = f"/tmp/frame_{int(time.time()*1000)}.jpg"
            success = cv2.imwrite(temp_path, frame)

            if not success:
                # Fail-safe: return no detection instead of error
                return []

            # Use SVM to predict
            prediction, confidence = self.custom_trainer.predict(temp_path)

            detection_time = time.time() - start_time
            self.ml_inference_count += 1
            self.ml_inference_time += detection_time

            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass

            # Check for error in prediction - return no detection for safety
            if prediction == -1:
                return []

            # Only return detection if we're very confident (car detected with high confidence)
            if prediction == 1 and confidence > 0.8:  # Higher threshold for safety
                # For custom model, we don't have bounding boxes, so create a full-frame detection
                height, width = frame.shape[:2]
                car_boxes = [{
                    'bbox': [0, 0, width, height],  # Full frame
                    'confidence': confidence,
                    'class': 'car_custom'
                }]
                return car_boxes

            # For uncertain cases or no cars detected, return empty list (no false positives)
            return []

        except Exception as e:
            # Fail-safe: return no detection instead of error
            if os.getenv('MODO', '').lower() == 'development':
                print(f"Debug: Custom model inference error: {e}")
            return []

    def _detect_with_ml(self, frame):
        """Detect cars using ML model"""
        if not self.model_loaded or self.ml_model is None:
            return []

        try:
            start_time = time.time()

            # Prepare image for model
            height, width = frame.shape[:2]

            # Detect objects
            classes, confidences, boxes = self.ml_model.detect(frame, confThreshold=MODEL_SETTINGS.get('confidence_threshold', 0.5))

            detection_time = time.time() - start_time
            self.ml_inference_count += 1
            self.ml_inference_time += detection_time

            # Filter for car-related classes
            car_classes = MODEL_SETTINGS.get('car_classes', ['car', 'truck', 'bus'])
            car_boxes = []

            if len(classes) > 0:
                for i, class_id in enumerate(classes.flatten()):
                    if class_id < len(self.labels):
                        class_name = self.labels[class_id]
                        if class_name.lower() in [c.lower() for c in car_classes]:
                            box = boxes[i]
                            car_boxes.append({
                                'bbox': box,
                                'confidence': confidences[i],
                                'class': class_name
                            })

            return car_boxes

        except Exception as e:
            if os.getenv('MODO', '').lower() == 'development':
                print(f"Debug: ML inference error: {e}")
            return []

    def _detect_with_cv_conservative(self, frame):
        """Conservative CV detection with balanced thresholds for reliable detection"""
        processed_frame = self.preprocess_frame(frame)
        fg_mask = self.bg_subtractor.apply(processed_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Balanced area threshold for reliable detection
        min_area = 400 if self.optimize_for_rpi else 800  # Adjusted for better detection
        valid_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Filtering for car-like shapes
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Reasonable aspect ratio for cars
                if 0.4 < aspect_ratio < 3.5 and w > 30 and h > 20:  # Adjusted size requirements
                    # Solidity check to avoid fragmented detections
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0

                    if solidity > 0.6:  # Reasonable solidity threshold
                        valid_contours.append((x, y, w, h))

        # Convert to same format as ML detection
        cv_boxes = []
        for contour in valid_contours:
            x, y, w, h = contour
            cv_boxes.append({
                'bbox': [x, y, w, h],
                'confidence': 0.7,  # Moderate confidence for balanced method
                'class': 'car_cv_conservative'
            })

        return cv_boxes

    def _detect_with_cv(self, frame):
        """Fallback: Detect cars using traditional CV"""
        processed_frame = self.preprocess_frame(frame)
        fg_mask = self.bg_subtractor.apply(processed_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        car_count, valid_contours = self._extract_cars_from_contours(contours, processed_frame, original_frame=frame)

        # Convert to same format as ML detection for consistency
        cv_boxes = []
        for contour in valid_contours:
            x, y, w, h = contour
            cv_boxes.append({
                'bbox': [x, y, w, h],
                'confidence': 0.8,  # Default confidence for CV method
                'class': 'car_cv'
            })

        return cv_boxes

    def detect_cars(self, frame):
        """Main detection method using conservative CV for reliable counting"""
        car_boxes = []

        # Use conservative CV detection for reliable multi-car counting
        # Custom model is only for presence detection, not suitable for counting
        car_boxes = self._detect_with_cv_conservative(frame)

        # Debug: print detection results
        if os.getenv('MODO', '').lower() == 'development':
            print(f"Debug: Detected {len(car_boxes)} cars in frame")

        return car_boxes

# Factory function for creating optimized car identifier
def create_car_identifier(mode='rpi', use_ml=True, use_custom_model=False):
    """
    Factory function to create car identifier based on target platform
    Now includes AI/ML capabilities and custom model support.

    Args:
        mode: 'rpi' for Raspberry Pi, 'desktop' for desktop, 'debug' for debugging
        use_ml: Whether to use ML model for car detection
        use_custom_model: Whether to use custom trained model (recommended for RPi)

    Returns:
        CarIdentifier instance with AI capabilities
    """
    if mode == 'rpi':
        return CarIdentifier(optimize_for_rpi=True, use_ml=use_ml, use_custom_model=use_custom_model)
    elif mode == 'desktop':
        return CarIdentifier(optimize_for_rpi=False, use_ml=use_ml, use_custom_model=use_custom_model)
    elif mode == 'debug':
        return CarIdentifier(optimize_for_rpi=False, use_ml=use_ml, use_custom_model=use_custom_model)
    else:
        return CarIdentifier(optimize_for_rpi=True, use_ml=use_ml, use_custom_model=use_custom_model)
