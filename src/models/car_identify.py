import cv2
import numpy as np
import time
import threading
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from src.settings.rpi_config import MODEL_SETTINGS
from src.utils.logger import get_logger
from src.settings.settings import get_settings
from src.utils.resource_manager import TempFileManager

from collections import deque

# Production mode detection
IS_PRODUCTION = os.getenv('MODE', os.getenv('MODO', 'production')).lower() == 'production'

# Lazy imports (only load when needed)
_OptimizedCarTrainer = None
_TFLiteCarDetector = None

def _get_optimized_trainer():
    """Lazy import OptimizedCarTrainer"""
    global _OptimizedCarTrainer
    if _OptimizedCarTrainer is None:
        from src.training.custom_car_trainer import OptimizedCarTrainer
        _OptimizedCarTrainer = OptimizedCarTrainer
    return _OptimizedCarTrainer

def _get_tflite_detector():
    """Lazy import TFLiteCarDetector"""
    global _TFLiteCarDetector
    if _TFLiteCarDetector is None:
        from src.models.tflite_car_detector import TFLiteCarDetector
        _TFLiteCarDetector = TFLiteCarDetector
    return _TFLiteCarDetector

class CarTracker:
    """
    Thread-safe tracker for car bounding boxes using OpenCV Tracker.
    
    Note: This class is NOT thread-safe on its own. Access to CarTracker
    should be protected by the CarIdentifier's lock.
    """
    def __init__(self, tracker_type="KCF", max_lost=10):
        self.trackers = []
        self.ids = []
        self.next_id = 0
        self.max_lost = max_lost
        self.lost = {}
        self.tracker_type = tracker_type

    def _create_tracker(self):
        # Tenta CSRT, depois KCF, depois erro
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        elif hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        elif hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
        elif hasattr(cv2, "TrackerKCF_create"):
            return cv2.TrackerKCF_create()
        else:
            raise AttributeError(
                "Nenhum tracker disponível (nem CSRT nem KCF). "
                "Verifique se o pacote opencv-contrib-python está instalado corretamente."
            )

    def update(self, frame):
        updated_boxes = []
        remove_idxs = []
        for i, tracker in enumerate(self.trackers):
            ok, bbox = tracker.update(frame)
            if ok:
                updated_boxes.append((self.ids[i], bbox))
                self.lost[self.ids[i]] = 0
            else:
                self.lost[self.ids[i]] += 1
                if self.lost[self.ids[i]] > self.max_lost:
                    remove_idxs.append(i)
        # Remove lost trackers
        for idx in reversed(remove_idxs):
            del self.trackers[idx]
            del self.ids[idx]
        return updated_boxes

    def add(self, frame, bbox):
        tracker = self._create_tracker()
        tracker.init(frame, tuple(bbox))
        self.trackers.append(tracker)
        self.ids.append(self.next_id)
        self.lost[self.next_id] = 0
        self.next_id += 1

    def reset(self):
        self.trackers = []
        self.ids = []
        self.lost = {}
        self.next_id = 0
class CarIdentifier:
    def __init__(self, optimize_for_rpi=True, use_ml=True, use_custom_model=False, use_tflite=False):
        """
        Initialize car identifier with multiple detection strategies
        
        Args:
            optimize_for_rpi: Use Raspberry Pi optimizations
            use_ml: Use ML model (MobileNet SSD)
            use_custom_model: Use custom trained SVM model
            use_tflite: Use TensorFlow Lite model
        """
        # Initialize logger first
        self.logger = get_logger(__name__)
        
        # Load settings
        try:
            self.settings = get_settings()
        except Exception as e:
            self.logger.warning(f"Failed to load settings, using defaults: {e}")
            self.settings = None
        
        self.optimize_for_rpi = optimize_for_rpi
        self.use_ml = use_ml and MODEL_SETTINGS.get('use_ml_model', False)
        self.use_custom_model = use_custom_model
        self.use_tflite = use_tflite

        # Thread safety: Use RLock for recursive locking capability
        # This allows the same thread to acquire the lock multiple times
        self.lock = threading.RLock()
        self.lock_timeout = 5.0  # 5 second timeout to prevent deadlocks
        self.lock_contention_threshold = 1.0  # Log warning if lock wait > 1 second

        # Initialize ML model
        self.ml_model = None
        self.labels = []
        self.model_loaded = False

        # Initialize custom SVM model
        self.custom_trainer = None
        self.custom_model_loaded = False

        # Initialize TFLite detector
        self.tflite_detector = None
        self.tflite_loaded = False
        
        # Frame resize cache (production optimization)
        self._resize_cache = {}
        self._cache_max_size = 10

        # Tracker for custom model detections
        self.tracker = CarTracker(tracker_type="KCF", max_lost=10)
        self.last_tracked_boxes = []

        # Priority: TFLite > Custom SVM > ML model
        if self.use_tflite:
            self._load_tflite_model()
        elif self.use_custom_model:
            self._load_custom_model()
        elif self.use_ml:
            self._load_ml_model()
        
        # Validate models after loading
        self._validate_models()

        # Fallback: traditional CV background subtractor com configuração otimizada
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

        # Frame counter for periodic background subtractor reset (thread-safe)
        self.frame_counter = 0
        if self.settings:
            fps = self.settings.camera.fps
            reset_seconds = self.settings.detection.reset_interval_seconds
            self.reset_interval = fps * reset_seconds
        else:
            self.reset_interval = 300  # Default: 300 frames (~30 seconds at 10fps)
        
        self.logger.info(f"CarIdentifier initialized: optimize_rpi={optimize_for_rpi}, "
                        f"use_ml={use_ml}, use_custom={use_custom_model}, use_tflite={use_tflite}")
        self.logger.debug(f"Reset interval: {self.reset_interval} frames, Lock timeout: {self.lock_timeout}s")

        # Kernel for morphological operations
        kernel_size = 3 if optimize_for_rpi else 5
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Performance tracking (thread-safe with lock)
        self.frame_count = 0
        self.total_processing_time = 0
        self.ml_inference_count = 0
        self.ml_inference_time = 0
    
    def _acquire_lock_with_timeout(self, operation: str) -> bool:
        """
        Acquire lock with timeout and contention logging.
        
        Args:
            operation: Name of operation for logging
            
        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        acquired = self.lock.acquire(timeout=self.lock_timeout)
        wait_time = time.time() - start_time
        
        if not acquired:
            thread_id = threading.get_ident()
            self.logger.error(f"[Thread-{thread_id}] Lock acquisition timeout ({self.lock_timeout}s) "
                            f"for operation: {operation}")
            return False
        
        if wait_time > self.lock_contention_threshold:
            thread_id = threading.get_ident()
            self.logger.warning(f"[Thread-{thread_id}] Lock contention detected: waited {wait_time:.3f}s "
                              f"for operation: {operation}")
        else:
            thread_id = threading.get_ident()
            self.logger.debug(f"[Thread-{thread_id}] Lock acquired in {wait_time:.3f}s for: {operation}")
        
        return True
    
    def _release_lock(self, operation: str):
        """
        Release lock with logging.
        
        Args:
            operation: Name of operation for logging
        """
        thread_id = threading.get_ident()
        self.logger.debug(f"[Thread-{thread_id}] Lock released for: {operation}")
        self.lock.release()
        
    def preprocess_frame(self, frame):
        """Preprocess frame with caching for better performance"""
        frame_key = id(frame)
        
        # Check cache first (production optimization)
        if IS_PRODUCTION and frame_key in self._resize_cache:
            return self._resize_cache[frame_key]
        
        if self.optimize_for_rpi:
            # Resize frame ONCE for faster processing on RPi
            frame_resized = cv2.resize(frame, (320, 240))
        else:
            frame_resized = frame
        
        # Convert to grayscale for background subtraction
        if len(frame_resized.shape) == 3:
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_resized
        
        # Cache result (limited cache size)
        if IS_PRODUCTION:
            if len(self._resize_cache) >= self._cache_max_size:
                # Remove oldest entry
                self._resize_cache.pop(next(iter(self._resize_cache)))
            self._resize_cache[frame_key] = gray
            
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
        
        Thread Safety:
            - Acquires lock with timeout
            - Protects frame counter and background subtractor
            - Releases lock even on error
            - Logs thread ID for debugging

        Args:
            frame: np.ndarray, image from camera
        Returns:
            int: number of cars detected
        """
        thread_id = threading.get_ident()
        
        if frame is None or frame.size == 0:
            self.logger.debug(f"[Thread-{thread_id}] Invalid frame provided to count_cars")
            return 0

        self.logger.debug(f"[Thread-{thread_id}] Starting car counting process...")
        start_time = time.time()

        # Thread-safe access to shared state with timeout
        if not self._acquire_lock_with_timeout("count_cars"):
            self.logger.error(f"[Thread-{thread_id}] Failed to acquire lock for count_cars, returning 0")
            return 0
        
        try:
            # Reset background subtractor periodically to avoid over-adaptation
            self.frame_counter += 1
            if self.frame_counter >= self.reset_interval:
                self.logger.info(f"[Thread-{thread_id}] Resetting background subtractor after "
                               f"{self.frame_counter} frames")
                self._reset_background_subtractor()
                self.frame_counter = 0

            # Use ML-based detection (this releases lock internally if needed)
            car_boxes = self.detect_cars(frame)
            car_count = len(car_boxes)

            processing_time = time.time() - start_time

            self.logger.debug(f"[Thread-{thread_id}] Processing time: {processing_time:.3f}s")
            if car_count > 0:
                self.logger.info(f"[Thread-{thread_id}] Detected {car_count} cars")
                for i, detection in enumerate(car_boxes):
                    self.logger.debug(f"[Thread-{thread_id}]   Car {i+1}: {detection.get('class', 'unknown')} "
                                    f"(confidence: {detection.get('confidence', 0):.2f})")
            else:
                self.logger.debug(f"[Thread-{thread_id}] No cars detected in this frame")

            # Update performance metrics
            self.frame_count += 1
            self.total_processing_time += processing_time
            
            return car_count
            
        except Exception as e:
            self.logger.error(f"[Thread-{thread_id}] Error in count_cars: {e}", exc_info=True)
            return 0
        finally:
            self._release_lock("count_cars")
    
    def _reset_background_subtractor(self):
        """
        Reset background subtractor with proper resource cleanup.
        
        Thread Safety:
            - Assumes caller holds the lock
            - Releases old instance before creating new one
        """
        thread_id = threading.get_ident()
        self.logger.debug(f"[Thread-{thread_id}] Resetting background subtractor...")
        
        # Explicitly delete old instance to free memory
        if hasattr(self, 'bg_subtractor') and self.bg_subtractor is not None:
            del self.bg_subtractor
            self.logger.debug(f"[Thread-{thread_id}] Released old background subtractor")
        
        # Create new instance
        if self.optimize_for_rpi:
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
        
        self.logger.debug(f"[Thread-{thread_id}] New background subtractor created")
    
    def get_average_processing_time(self):
        """Get average frame processing time"""
        if self.frame_count > 0:
            return self.total_processing_time / self.frame_count
        return 0.0
    
    def reset_metrics(self):
        """
        Reset performance metrics (thread-safe).
        """
        if not self._acquire_lock_with_timeout("reset_metrics"):
            self.logger.error("Failed to acquire lock for reset_metrics")
            return
        
        try:
            self.frame_count = 0
            self.total_processing_time = 0
            self.logger.info("Performance metrics reset")
        finally:
            self._release_lock("reset_metrics")
    
    def get_car_count_safe(self):
        """
        Thread-safe getter for current car count.
        
        Returns:
            int: Current frame count (as proxy for activity)
        """
        if not self._acquire_lock_with_timeout("get_car_count_safe"):
            return 0
        
        try:
            return self.frame_count
        finally:
            self._release_lock("get_car_count_safe")
    
    def visualize_detection(self, frame, show_contours=False):
        """
        Create visualization of car detection using AI/ML (thread-safe).

        Thread Safety:
            - Works on frame copy to avoid concurrent modification
            - Acquires lock for detection
            - Drawing happens outside lock for better concurrency
            - Handles lock timeout gracefully

        Args:
            frame: Input frame
            show_contours: Whether to draw bounding boxes

        Returns:
            tuple: (count, visualization_frame)
        """
        thread_id = threading.get_ident()
        
        if frame is None or frame.size == 0:
            self.logger.debug(f"[Thread-{thread_id}] Invalid frame provided to visualize_detection")
            return 0, frame

        try:
            # Create visualization frame (work on copy to avoid concurrent modification)
            vis_frame = frame.copy()

            # Thread-safe access for detection with timeout
            if not self._acquire_lock_with_timeout("visualize_detection"):
                self.logger.warning(f"[Thread-{thread_id}] visualize_detection lock timeout, "
                                  f"returning frame without detection")
                return 0, vis_frame
            
            try:
                # Use ML-based detection
                car_boxes = self.detect_cars(frame)
                car_count = len(car_boxes) if car_boxes else 0

                self.logger.debug(f"[Thread-{thread_id}] Visualization - Detected {car_count} cars")
                
                # Copy detection results for drawing outside lock
                boxes_to_draw = car_boxes.copy() if car_boxes else []
                
            finally:
                self._release_lock("visualize_detection")

            # Draw detections (outside lock for better concurrency)
            if boxes_to_draw and show_contours:
                for i, detection in enumerate(boxes_to_draw):
                    try:
                        if not isinstance(detection, dict) or 'bbox' not in detection:
                            continue

                        x, y, w, h = detection['bbox']
                        confidence = detection.get('confidence', 0.0)
                        class_name = detection.get('class', 'unknown')

                        # Validate bbox coordinates
                        if not all(isinstance(coord, (int, float)) for coord in [x, y, w, h]):
                            continue

                        # Color based on detection method
                        color = (0, 255, 0) if 'cv' not in class_name.lower() else (255, 0, 0)  # Green for ML, Red for CV

                        # Only draw reasonable-sized boxes (avoid full-screen false positives)
                        if w < frame.shape[1] * 0.8 and h < frame.shape[0] * 0.8:  # Box must be less than 80% of frame
                            cv2.rectangle(vis_frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

                            # Label with class and confidence
                            label = f"{class_name[:10]}: {confidence:.2f}"
                            cv2.putText(vis_frame, label, (int(x), int(y - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        else:
                            self.logger.debug(f"[Thread-{thread_id}] Skipping oversized box: "
                                            f"{w}x{h} (frame: {frame.shape[1]}x{frame.shape[0]})")
                    except Exception as bbox_error:
                        self.logger.debug(f"[Thread-{thread_id}] Error drawing bbox {i}: {bbox_error}")
                        continue

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

        except Exception as e:
            self.logger.error(f"[Thread-{thread_id}] Error in visualize_detection: {e}", exc_info=True)
            # Return safe fallback
            return 0, frame

    def _load_tflite_model(self):
        """Load TFLite optimized car detection model (called during init, no lock needed)"""
        try:
            TFLiteDetectorClass = _get_tflite_detector()
            self.tflite_detector = TFLiteDetectorClass()
            if self.tflite_detector.interpreter is not None:
                self.tflite_loaded = True
                self.logger.info("TFLite optimized model loaded successfully")
                self.logger.info("🚀 Using TFLite model - optimized for Raspberry Pi performance!")
            else:
                self.logger.warning("TFLite model not found. Falling back to custom SVM model.")
                self.tflite_loaded = False
                self._load_custom_model()

        except Exception as e:
            self.logger.error(f"Error loading TFLite model: {e}. Falling back to custom SVM model.",
                            exc_info=True)
            self.tflite_loaded = False
            self._load_custom_model()

    def _load_custom_model(self):
        """Load custom trained SVM model (called during init, no lock needed)"""
        try:
            OptimizedCarTrainerClass = _get_optimized_trainer()
            
            # FIX: Carregar novo modelo .pkl em vez de .yml
            model_path = Path('src/models/custom_car_detector_optimized.pkl')
            
            if not model_path.exists():
                # Fallback para modelo antigo se novo não existir
                old_model_path = Path('src/models/custom_car_detector.yml')
                if old_model_path.exists():
                    self.logger.warning(f"Novo modelo não encontrado em {model_path}")
                    self.logger.warning(f"Usando modelo antigo: {old_model_path}")
                    self.logger.warning("RECOMENDADO: Retreine com: python3 src/training/custom_car_trainer.py")
                    self._load_old_custom_model()  # Fallback para modelo antigo
                    return
                else:
                    self.logger.warning(f"Modelo customizado não encontrado em {model_path}")
                    self.logger.info("Treine o modelo com: python3 src/training/custom_car_trainer.py")
                    return

            # Criar trainer e carregar modelo otimizado
            self.custom_trainer = OptimizedCarTrainerClass()
            
            if self.custom_trainer.load_model(str(model_path)):
                self.custom_model_loaded = True
                self.logger.info(f"✅ Modelo SVM otimizado carregado de {model_path}")
                self.logger.info("   Features: HOG + Cor + Textura + Geometria + Momentos")
                self.logger.info("   SVM: RBF kernel com C=1000.0")
            else:
                self.logger.error(f"Falha ao carregar modelo customizado de {model_path}")
                
        except ImportError as e:
            self.logger.error(f"Erro ao importar OptimizedCarTrainer: {e}")
            self.logger.info("Verifique se src/training/custom_car_trainer.py existe")
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo customizado: {e}", exc_info=True)

    def _load_old_custom_model(self):
        """Fallback: Load old .yml custom model"""
        try:
            # FIX: Verificar se existe classe LightweightCarTrainer para compatibilidade
            try:
                from src.training.custom_car_trainer import LightweightCarTrainer
                trainer_class = LightweightCarTrainer
            except ImportError:
                # Se LightweightCarTrainer não existir, usar OptimizedCarTrainer
                # (não será capaz de carregar .yml, mas evita erro de import)
                self.logger.warning("LightweightCarTrainer não encontrado - modelo .yml não suportado")
                return
            
            old_model_path = Path('src/models/custom_car_detector.yml')
            
            # Criar trainer antigo
            self.custom_trainer = trainer_class()
            
            # Tentar carregar modelo antigo
            if hasattr(self.custom_trainer, 'svm') and self.custom_trainer.svm is not None:
                try:
                    self.custom_trainer.svm.load(str(old_model_path))
                    self.custom_model_loaded = True
                    self.logger.info(f"Modelo SVM antigo (.yml) carregado de {old_model_path}")
                    self.logger.warning("⚠️  Usando modelo ANTIGO - performance reduzida")
                    self.logger.warning("   Retreine com: python3 src/training/custom_car_trainer.py")
                except Exception as e:
                    self.logger.error(f"Erro ao carregar modelo .yml: {e}")
            else:
                self.logger.error("Trainer antigo não inicializado corretamente")
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo antigo: {e}")

    def _load_ml_model(self):
        """Load ML model for car detection (called during init, no lock needed)"""
        try:
            model_path = Path(MODEL_SETTINGS.get('model_path', 'src/models/ssd_mobilenet_v3_large_coco.pb'))
            config_path = Path(MODEL_SETTINGS.get('model_path', 'src/models/ssd_mobilenet_v3_large_coco.pbtxt').replace('.pb', '.pbtxt'))
            labels_path = Path(MODEL_SETTINGS.get('labels_path', 'src/models/coco_labels.txt'))

            # Check if model files exist
            if not model_path.exists() or not config_path.exists():
                self.logger.warning("ML model files not found. Using fallback CV method.")
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
            self.logger.info(f"ML model loaded successfully with {len(self.labels)} classes")

        except Exception as e:
            self.logger.error(f"Error loading ML model: {e}. Using fallback CV method.", exc_info=True)
            self.model_loaded = False
    
    def _validate_models(self) -> Dict[str, bool]:
        """
        Validate all ML models are available and working.
        Tests model loading and basic inference capability.
        
        Returns:
            Dictionary with validation status for each model type
        """
        validation = {
            'tflite': False,
            'custom_svm': False,
            'ml_model': False,
            'has_fallback': True  # CV method always available
        }
        
        # Validate TFLite model
        if self.use_tflite:
            try:
                if self.tflite_loaded and self.tflite_detector is not None:
                    # Test with dummy data
                    test_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                    detections = self.tflite_detector.detect_cars(test_frame)
                    validation['tflite'] = True
                    self.logger.info("✅ TFLite model validated successfully")
                else:
                    self.logger.warning("⚠️  TFLite model not loaded")
            except Exception as e:
                self.logger.error(f"❌ TFLite model validation failed: {e}")
        
        # FIX: Validate custom SVM model - verificar .pkl em vez de .yml
        if self.use_custom_model:
            try:
                if self.custom_model_loaded and self.custom_trainer is not None:
                    # FIX: Verificar arquivo .pkl otimizado
                    model_path = Path('src/models/custom_car_detector_optimized.pkl')
                    if model_path.exists():
                        validation['custom_svm'] = True
                        self.logger.info("✅ Custom SVM model validated successfully")
                        self.logger.info(f"   Model file: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
                    else:
                        self.logger.warning(f"⚠️  Custom SVM model file not found at {model_path}")
                else:
                    self.logger.warning("⚠️  Custom SVM model not loaded")
            except Exception as e:
                self.logger.error(f"❌ Custom SVM model validation failed: {e}")
        
        # Validate ML model
        if self.use_ml:
            try:
                if self.model_loaded and self.ml_model is not None:
                    validation['ml_model'] = True
                    self.logger.info(f"✅ ML model validated with {len(self.labels)} classes")
                else:
                    self.logger.warning("⚠️  ML model not loaded")
            except Exception as e:
                self.logger.error(f"❌ ML model validation failed: {e}")
        
        # FIX: Log overall validation status corretamente
        active_models = [k for k, v in validation.items() if v and k != 'has_fallback']
        if active_models:
            self.logger.info(f"🎯 Active models: {', '.join(active_models)}")
        else:
            self.logger.warning("⚠️  No ML models available, using CV fallback only")
        
        return validation

    def _detect_with_tflite(self, frame):
        """Detect cars using TFLite optimized model"""
        if not self.tflite_loaded or self.tflite_detector is None:
            return []

        try:
            start_time = time.time()

            # Run TFLite inference
            detections = self.tflite_detector.detect_cars(frame)

            detection_time = time.time() - start_time
            self.ml_inference_count += 1
            self.ml_inference_time += detection_time

            # Convert to expected format
            car_boxes = []
            for det in detections:
                car_boxes.append({
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'class': 'car_tflite'
                })

            return car_boxes

        except Exception as e:
            self.logger.error(f"TFLite inference error: {e}", exc_info=True)
            return []

    def _detect_with_custom_model(self, frame):
        """
        Detect cars using custom trained SVM model with improved negative sampling.
        """
        if not self.custom_model_loaded or self.custom_trainer is None:
            self.logger.debug("Custom model not loaded, falling back to CV detection")
            return []

        try:
            import tempfile
            
            start_time = time.time()

            with tempfile.NamedTemporaryFile(mode='w+b', suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85 if IS_PRODUCTION else 95]
                cv2.imwrite(temp_path, frame, encode_param)
        
            try:
                # FIX: Reduzir threshold de 0.75 para 0.50 (50% confiança)
                prediction, confidence = self.custom_trainer.predict(temp_path, threshold=0.50)  # ← MUDANÇA AQUI
            
                detection_time = time.time() - start_time
                self.ml_inference_count += 1
                self.ml_inference_time += detection_time
            
                if not IS_PRODUCTION:
                    self.logger.debug(f"Custom SVM prediction: {prediction}, confidence: {confidence:.3f}")
            
                if prediction == 1:
                    h, w = frame.shape[:2]
                    car_boxes = [{
                        'bbox': [int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8)],
                        'confidence': float(confidence),
                        'class': 'car_custom_svm'
                    }]
                    if not IS_PRODUCTION:
                        self.logger.info(f"Custom SVM detected car with confidence {confidence:.3f}")
                    return car_boxes
                else:
                    if not IS_PRODUCTION:
                        self.logger.debug(f"Custom SVM: No car detected (confidence: {confidence:.3f})")
                    return []
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(f"Custom SVM inference error: {e}", exc_info=True)
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
                # Ensure all arrays have the same length to prevent index errors
                min_length = min(len(classes.flatten()), len(boxes), len(confidences.flatten()) if len(confidences.shape) > 1 else len(confidences))
                for i in range(min_length):
                    class_id = classes.flatten()[i]
                    if class_id < len(self.labels):
                        class_name = self.labels[class_id]
                        if class_name.lower() in [c.lower() for c in car_classes]:
                            box = boxes[i]
                            confidence = confidences.flatten()[i] if len(confidences.shape) > 1 else confidences[i]
                            car_boxes.append({
                                'bbox': box,
                                'confidence': confidence,
                                'class': class_name
                            })

            return car_boxes

        except Exception as e:
            self.logger.error(f"ML inference error: {e}", exc_info=True)
            return []

    def _detect_with_cv_conservative(self, frame):
        """
        Conservative CV detection with balanced thresholds.
        
        Thread Safety:
            - Assumes caller holds lock (for bg_subtractor access)
        """
        thread_id = threading.get_ident()
        self.logger.info(f"[Thread-{thread_id}] 🔍 Using CV background subtraction detection")
        
        processed_frame = self.preprocess_frame(frame)
        
        # Background subtractor access - must be protected by lock
        fg_mask = self.bg_subtractor.apply(processed_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Balanced area threshold - not too conservative to miss real cars
        min_area = 1000 if self.optimize_for_rpi else 2000  # Balanced threshold
        valid_contours = []

        self.logger.debug(f"[Thread-{thread_id}] CV Conservative - Found {len(contours)} total contours")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            self.logger.debug(f"[Thread-{thread_id}] Contour area: {area}")

            # Only consider very large contours that are definitely not noise
            if area > min_area:
                # Very strict filtering for car-like shapes
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0

                # Broader aspect ratio range for cars (include various car shapes)
                if 0.6 < aspect_ratio < 3.0 and w > 40 and h > 25:  # More flexible size requirements
                    # Solidity check to avoid fragmented detections
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0

                    if solidity > 0.6:  # Reasonable solidity threshold
                        # Additional circularity check to avoid round objects
                        perimeter = cv2.arcLength(cnt, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                        # Cars should have low circularity (not round)
                        if circularity < 0.8:  # Balanced circularity threshold
                            # Additional check: extent (bounding box fill ratio)
                            rect_area = w * h
                            extent = float(area) / rect_area if rect_area > 0 else 0

                            if extent > 0.3:
                                valid_contours.append((x, y, w, h))
                                self.logger.debug(f"[Thread-{thread_id}] ✅ Valid: area={area}, "
                                                f"aspect={aspect_ratio:.2f}, solidity={solidity:.2f}")

        # Convert to same format as ML detection
        cv_boxes = []
        for contour in valid_contours:
            x, y, w, h = contour
            cv_boxes.append({
                'bbox': [x, y, w, h],
                'confidence': 0.5,  # Very low confidence - extremamente conservador
                'class': 'car_cv_conservative'
            })

        self.logger.debug(f"[Thread-{thread_id}] CV Conservative - Final detections: {len(cv_boxes)}")
        return cv_boxes

    def _detect_with_cv(self, frame):
        """
        Fallback: Detect cars using traditional CV.
        
        Thread Safety:
            - Assumes caller holds lock (for bg_subtractor access)
        """
        thread_id = threading.get_ident()
        self.logger.debug(f"[Thread-{thread_id}] Using standard CV detection")
        
        processed_frame = self.preprocess_frame(frame)
        
        # Background subtractor access - must be protected by lock
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
    def detect_cars_with_tracking(self, frame):
        """
        Detect cars using the custom model WITHOUT tracking (tracking disabled).
        
        Thread Safety:
            - Assumes caller holds lock
            
        Returns:
            list: Detected bounding boxes (tracking disabled due to missing opencv-contrib-python)
        """
        thread_id = threading.get_ident()
        self.logger.debug(f"[Thread-{thread_id}] Detecting cars without tracking (tracking disabled)")
        
        # Detect with custom model (lock already held by caller)
        car_boxes = []
        if self.use_custom_model:
            car_boxes = self._detect_with_custom_model(frame)
        else:
            car_boxes = self._detect_with_ml(frame)

        self.logger.debug(f"[Thread-{thread_id}] Detection - Found: {len(car_boxes)} cars")

        # Return detected boxes without tracking (convert to expected format)
        return [{'id': i, 'bbox': det['bbox']} for i, det in enumerate(car_boxes)]

    def detect_cars(self, frame):
        """
        Main detection method: priority TFLite > Custom SVM > ML > CV fallback.
        
        Thread Safety:
            - Can be called with or without lock held
            - ML inference happens without holding lock (for performance)
            - CV methods require lock (for bg_subtractor access)
            
        Returns:
            list: Detected car bounding boxes with metadata
        """
        thread_id = threading.get_ident()
        self.logger.debug(f"[Thread-{thread_id}] Starting car detection")

        # Priority 1: TFLite optimized model (fastest and most accurate)
        # Priority 1: TFLite (no lock needed - inference is thread-safe)
        if self.use_tflite and self.tflite_loaded:
            car_boxes = self._detect_with_tflite(frame)
            self.logger.debug(f"[Thread-{thread_id}] TFLite detected {len(car_boxes)} cars")
            return car_boxes

        # Priority 2: Custom SVM with tracking (needs lock for tracker)
        elif self.use_custom_model and self.custom_model_loaded:
            # Acquire lock if not already held (for tracker access)
            lock_held = self.lock._is_owned() if hasattr(self.lock, '_is_owned') else False
            
            if not lock_held:
                if not self._acquire_lock_with_timeout("detect_cars_custom"):
                    self.logger.error(f"[Thread-{thread_id}] Failed to acquire lock for custom detection")
                    return []
            
            try:
                tracked = self.detect_cars_with_tracking(frame)
                car_boxes = []
                for t in tracked:
                    x, y, w, h = [int(v) for v in t['bbox']]
                    car_boxes.append({'bbox': [x, y, w, h], 'confidence': 1.0,
                                    'class': 'car_custom_tracked'})
                self.logger.debug(f"[Thread-{thread_id}] Custom SVM detected {len(car_boxes)} cars")
                return car_boxes
            finally:
                if not lock_held:
                    self._release_lock("detect_cars_custom")

        # Priority 3: Standard ML model (no lock needed)
        elif self.use_ml and self.model_loaded:
            car_boxes = self._detect_with_ml(frame)
            self.logger.debug(f"[Thread-{thread_id}] ML model detected {len(car_boxes)} cars")
            return car_boxes

        # Fallback: CV detection (needs lock for bg_subtractor)
        else:
            lock_held = self.lock._is_owned() if hasattr(self.lock, '_is_owned') else False
            
            if not lock_held:
                if not self._acquire_lock_with_timeout("detect_cars_cv"):
                    self.logger.error(f"[Thread-{thread_id}] Failed to acquire lock for CV detection")
                    return []
            
            try:
                car_boxes = self._detect_with_cv_conservative(frame)
                self.logger.debug(f"[Thread-{thread_id}] CV fallback detected {len(car_boxes)} cars")
                return car_boxes
            finally:
                if not lock_held:
                    self._release_lock("detect_cars_cv")
    
    def __del__(self):
        """
        Cleanup method to ensure resources are released.
        
        Thread Safety:
            - Attempts to acquire lock with timeout
            - Logs cleanup operations
            - Releases all OpenCV resources
        """
        try:
            if hasattr(self, 'logger') and not IS_PRODUCTION:
                self.logger.info("CarIdentifier cleanup initiated")
            
            # Clear resize cache
            if hasattr(self, '_resize_cache'):
                self._resize_cache.clear()
            
            # Clean up tracker
            if hasattr(self, 'tracker') and self.tracker:
                self.tracker.reset()
                if hasattr(self, 'logger') and not IS_PRODUCTION:
                    self.logger.debug("Tracker resources released")
            
            # Clean up background subtractor
            if hasattr(self, 'bg_subtractor') and self.bg_subtractor is not None:
                del self.bg_subtractor
                if hasattr(self, 'logger') and not IS_PRODUCTION:
                    self.logger.debug("Background subtractor released")
            
            # Clean up ML models
            if hasattr(self, 'ml_model') and self.ml_model is not None:
                del self.ml_model
                if hasattr(self, 'logger') and not IS_PRODUCTION:
                    self.logger.debug("ML model released")
            
            if hasattr(self, 'tflite_detector') and self.tflite_detector is not None:
                del self.tflite_detector
                if hasattr(self, 'logger') and not IS_PRODUCTION:
                    self.logger.debug("TFLite detector released")
            
            if hasattr(self, 'custom_trainer') and self.custom_trainer is not None:
                del self.custom_trainer
                if hasattr(self, 'logger') and not IS_PRODUCTION:
                    self.logger.debug("Custom trainer released")
            
            # Cleanup orphaned temp files
            try:
                TempFileManager.cleanup_orphaned_files(prefix="svm_detect_", max_age_hours=1)
            except Exception as cleanup_error:
                if hasattr(self, 'logger') and not IS_PRODUCTION:
                    self.logger.debug(f"Temp file cleanup: {cleanup_error}")
            
            if hasattr(self, 'logger') and not IS_PRODUCTION:
                self.logger.info("CarIdentifier cleanup completed")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during cleanup: {e}", exc_info=not IS_PRODUCTION)

# Factory function for creating optimized car identifier
def create_car_identifier(mode='rpi', use_ml=True, use_custom_model=False, use_tflite=False):
    """
    Factory function to create car identifier based on target platform
    Now includes AI/ML capabilities with TFLite optimization support.

    Args:
        mode: 'rpi' for Raspberry Pi, 'desktop' for desktop, 'debug' for debugging
        use_ml: Whether to use ML model for car detection
        use_custom_model: Whether to use custom trained SVM model
        use_tflite: Whether to use TFLite optimized model (recommended for RPi)

    Returns:
        CarIdentifier instance with AI capabilities
    """
    if mode == 'rpi':
        # Raspberry Pi: prioritize TFLite for performance
        return CarIdentifier(optimize_for_rpi=True, use_ml=use_ml,
                           use_custom_model=use_custom_model, use_tflite=use_tflite)
    elif mode == 'desktop':
        return CarIdentifier(optimize_for_rpi=False, use_ml=use_ml,
                           use_custom_model=use_custom_model, use_tflite=use_tflite)
    elif mode == 'debug':
        return CarIdentifier(optimize_for_rpi=False, use_ml=use_ml,
                           use_custom_model=use_custom_model, use_tflite=use_tflite)
    else:
        return CarIdentifier(optimize_for_rpi=True, use_ml=use_ml,
                           use_custom_model=use_custom_model, use_tflite=use_tflite)
