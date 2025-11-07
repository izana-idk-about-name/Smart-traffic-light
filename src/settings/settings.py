"""
Comprehensive configuration management for Smart Traffic Light system.
Provides a single source of truth for all configuration settings with validation.
"""

import os
import platform
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _detect_raspberry_pi() -> bool:
    """Detect if running on Raspberry Pi"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            return 'BCM' in cpuinfo or 'Raspberry Pi' in cpuinfo
    except (FileNotFoundError, PermissionError):
        return False


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def _get_env_int(key: str, default: int, min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Get integer value from environment variable with validation"""
    try:
        value = int(os.getenv(key, str(default)))
        if min_val is not None and value < min_val:
            raise ValueError(f"{key} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{key} must be <= {max_val}, got {value}")
        return value
    except ValueError as e:
        raise ValueError(f"Invalid value for {key}: {e}")


def _get_env_float(key: str, default: float, min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Get float value from environment variable with validation"""
    try:
        value = float(os.getenv(key, str(default)))
        if min_val is not None and value < min_val:
            raise ValueError(f"{key} must be >= {min_val}, got {value}")
        if max_val is not None and value > max_val:
            raise ValueError(f"{key} must be <= {max_val}, got {value}")
        return value
    except ValueError as e:
        raise ValueError(f"Invalid value for {key}: {e}")


@dataclass
class SystemSettings:
    """System-level configuration settings"""
    mode: Literal['development', 'production', 'test']
    platform: Literal['raspberry_pi', 'desktop']
    debug: bool
    
    @classmethod
    def from_env(cls) -> 'SystemSettings':
        """Create SystemSettings from environment variables"""
        # Support both MODO and MODE for backward compatibility
        mode_str = os.getenv('MODE', os.getenv('MODO', 'production')).lower()
        if mode_str not in ('development', 'production', 'test'):
            raise ValueError(f"Invalid MODE: {mode_str}. Must be 'development', 'production', or 'test'")
        
        platform_override = os.getenv('PLATFORM', '').lower()
        if platform_override in ('raspberry_pi', 'desktop'):
            platform_str = platform_override
        else:
            platform_str = 'raspberry_pi' if _detect_raspberry_pi() else 'desktop'
        
        debug = _get_env_bool('DEBUG', mode_str == 'development')
        
        return cls(
            mode=mode_str,  # type: ignore
            platform=platform_str,  # type: ignore
            debug=debug
        )


@dataclass
class CameraSettings:
    """Camera configuration settings"""
    camera_a_index: int
    camera_b_index: int
    width: int
    height: int
    fps: int
    use_test_images: bool
    test_image_path_a: str
    test_image_path_b: str
    buffer_size: int
    
    @classmethod
    def from_env(cls, is_rpi: bool) -> 'CameraSettings':
        """Create CameraSettings from environment variables"""
        # Use optimized defaults based on platform
        if is_rpi:
            default_width, default_height, default_fps = 320, 240, 10
        else:
            default_width, default_height, default_fps = 640, 480, 15
        
        return cls(
            camera_a_index=_get_env_int('CAMERA_A_INDEX', 0, min_val=0),
            camera_b_index=_get_env_int('CAMERA_B_INDEX', 1, min_val=0),
            width=_get_env_int('CAMERA_WIDTH', default_width, min_val=160, max_val=1920),
            height=_get_env_int('CAMERA_HEIGHT', default_height, min_val=120, max_val=1080),
            fps=_get_env_int('CAMERA_FPS', default_fps, min_val=1, max_val=60),
            use_test_images=_get_env_bool('USE_TEST_IMAGES', False),
            test_image_path_a=os.getenv('TEST_IMAGE_PATH_A', 'src/Data/test_frame_a.jpg'),
            test_image_path_b=os.getenv('TEST_IMAGE_PATH_B', 'src/Data/test_frame_b.jpg'),
            buffer_size=_get_env_int('CAMERA_BUFFER_SIZE', 1 if is_rpi else 2, min_val=1, max_val=10)
        )


@dataclass
class DetectionSettings:
    """Detection and ML model configuration settings"""
    use_tflite: bool
    use_ml_model: bool
    use_custom_model: bool
    min_confidence: float
    reset_interval_seconds: int
    enable_tracking: bool
    model_path: str
    labels_path: str
    max_results: int
    car_classes: list[str]
    
    # CV fallback settings
    use_fallback_cv: bool
    background_history: int
    var_threshold: int
    min_car_area: int
    
    @classmethod
    def from_env(cls, is_rpi: bool) -> 'DetectionSettings':
        """Create DetectionSettings from environment variables"""
        return cls(
            use_tflite=_get_env_bool('USE_TFLITE', True),
            use_ml_model=_get_env_bool('USE_ML_MODEL', True),
            use_custom_model=_get_env_bool('USE_CUSTOM_MODEL', True),
            min_confidence=_get_env_float('MIN_CONFIDENCE', 0.5, min_val=0.0, max_val=1.0),
            reset_interval_seconds=_get_env_int('RESET_INTERVAL_SECONDS', 30, min_val=1),
            enable_tracking=_get_env_bool('ENABLE_TRACKING', True),
            model_path=os.getenv('MODEL_PATH', 'src/models/efficientdet_lite2.tflite'),
            labels_path=os.getenv('LABELS_PATH', 'src/models/coco_labels.txt'),
            max_results=_get_env_int('MAX_RESULTS', 10 if is_rpi else 15, min_val=1, max_val=100),
            car_classes=['car', 'truck', 'bus'],
            use_fallback_cv=_get_env_bool('USE_FALLBACK_CV', True),
            background_history=_get_env_int('BACKGROUND_HISTORY', 50 if is_rpi else 100, min_val=10),
            var_threshold=_get_env_int('VAR_THRESHOLD', 30 if is_rpi else 40, min_val=1),
            min_car_area=_get_env_int('MIN_CAR_AREA', 200 if is_rpi else 500, min_val=50)
        )


@dataclass
class PerformanceSettings:
    """Performance and processing configuration settings"""
    max_processing_time: float
    visualization_enabled: bool
    save_frames: bool
    frame_save_interval: int
    decision_interval: float
    thread_count: int
    memory_limit_mb: int
    
    @classmethod
    def from_env(cls, is_rpi: bool) -> 'PerformanceSettings':
        """Create PerformanceSettings from environment variables"""
        cpu_count = os.cpu_count() or 2
        
        return cls(
            max_processing_time=_get_env_float('MAX_PROCESSING_TIME', 1.0 if is_rpi else 0.5, min_val=0.1),
            visualization_enabled=_get_env_bool('VISUALIZATION_ENABLED', True),
            save_frames=_get_env_bool('SAVE_FRAMES', False),
            frame_save_interval=_get_env_int('FRAME_SAVE_INTERVAL', 100, min_val=1),
            decision_interval=_get_env_float('DECISION_INTERVAL', 3.0 if is_rpi else 2.0, min_val=0.5),
            thread_count=_get_env_int('THREAD_COUNT', min(2, cpu_count) if is_rpi else cpu_count, min_val=1),
            memory_limit_mb=_get_env_int('MEMORY_LIMIT_MB', 512 if is_rpi else 1024, min_val=128)
        )


@dataclass
class LoggingSettings:
    """Logging configuration settings"""
    log_level: str
    log_dir: str
    enable_performance_logging: bool
    log_to_console: bool
    log_to_file: bool
    max_log_file_size_mb: int
    backup_count: int
    
    @classmethod
    def from_env(cls) -> 'LoggingSettings':
        """Create LoggingSettings from environment variables"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        valid_levels = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        if log_level not in valid_levels:
            raise ValueError(f"Invalid LOG_LEVEL: {log_level}. Must be one of {valid_levels}")
        
        return cls(
            log_level=log_level,
            log_dir=os.getenv('LOG_DIR', 'logs'),
            enable_performance_logging=_get_env_bool('ENABLE_PERFORMANCE_LOGGING', True),
            log_to_console=_get_env_bool('LOG_TO_CONSOLE', True),
            log_to_file=_get_env_bool('LOG_TO_FILE', True),
            max_log_file_size_mb=_get_env_int('MAX_LOG_FILE_SIZE_MB', 10, min_val=1),
            backup_count=_get_env_int('LOG_BACKUP_COUNT', 5, min_val=1)
        )


@dataclass
class TrafficControlSettings:
    """Traffic light timing configuration settings"""
    min_green_time: int
    max_green_time: int
    yellow_time: int
    red_time: int
    
    @classmethod
    def from_env(cls) -> 'TrafficControlSettings':
        """Create TrafficControlSettings from environment variables"""
        return cls(
            min_green_time=_get_env_int('MIN_GREEN_TIME', 10, min_val=5, max_val=300),
            max_green_time=_get_env_int('MAX_GREEN_TIME', 60, min_val=10, max_val=600),
            yellow_time=_get_env_int('YELLOW_TIME', 3, min_val=1, max_val=10),
            red_time=_get_env_int('RED_TIME', 2, min_val=1, max_val=10)
        )


@dataclass
class NetworkSettings:
    """Network communication configuration settings"""
    orchestrator_host: str
    orchestrator_port: int
    use_websocket: bool
    timeout: int
    retry_count: int
    buffer_size: int
    
    @classmethod
    def from_env(cls, is_rpi: bool) -> 'NetworkSettings':
        """Create NetworkSettings from environment variables"""
        return cls(
            orchestrator_host=os.getenv('ORCHESTRATOR_HOST', 'localhost'),
            orchestrator_port=_get_env_int('ORCHESTRATOR_PORT', 9000, min_val=1, max_val=65535),
            use_websocket=_get_env_bool('USE_WEBSOCKET', is_rpi),
            timeout=_get_env_int('NETWORK_TIMEOUT', 5, min_val=1),
            retry_count=_get_env_int('NETWORK_RETRY_COUNT', 3, min_val=0),
            buffer_size=_get_env_int('NETWORK_BUFFER_SIZE', 1024, min_val=128)
        )


@dataclass
class TrainingSettings:
    """Training and data validation configuration settings"""
    min_samples_per_class: int
    min_image_width: int
    min_image_height: int
    max_class_imbalance: float
    allowed_formats: list[str]
    check_duplicates: bool
    validate_before_training: bool
    
    # Training specific
    enable_data_augmentation: bool
    augmentation_factor: int
    test_split: float
    
    @classmethod
    def from_env(cls, is_rpi: bool) -> 'TrainingSettings':
        """Create TrainingSettings from environment variables"""
        allowed_formats_str = os.getenv('TRAINING_ALLOWED_FORMATS', 'jpg,jpeg,png,bmp,webp')
        allowed_formats = [fmt.strip() for fmt in allowed_formats_str.split(',')]
        
        return cls(
            min_samples_per_class=_get_env_int('MIN_SAMPLES_PER_CLASS', 50 if is_rpi else 100, min_val=10),
            min_image_width=_get_env_int('MIN_IMAGE_WIDTH', 64, min_val=32),
            min_image_height=_get_env_int('MIN_IMAGE_HEIGHT', 64, min_val=32),
            max_class_imbalance=_get_env_float('MAX_CLASS_IMBALANCE', 10.0, min_val=1.0),
            allowed_formats=allowed_formats,
            check_duplicates=_get_env_bool('CHECK_DUPLICATES', True),
            validate_before_training=_get_env_bool('VALIDATE_BEFORE_TRAINING', True),
            enable_data_augmentation=_get_env_bool('ENABLE_DATA_AUGMENTATION', True),
            augmentation_factor=_get_env_int('AUGMENTATION_FACTOR', 3, min_val=1, max_val=10),
            test_split=_get_env_float('TEST_SPLIT', 0.2, min_val=0.1, max_val=0.5)
        )


@dataclass
class Settings:
    """
    Main settings class that consolidates all configuration.
    Implements singleton pattern for global access.
    """
    system: SystemSettings
    camera: CameraSettings
    detection: DetectionSettings
    performance: PerformanceSettings
    logging: LoggingSettings
    traffic_control: TrafficControlSettings
    network: NetworkSettings
    training: TrainingSettings
    
    _instance: Optional['Settings'] = None
    
    @classmethod
    def get_instance(cls, force_reload: bool = False) -> 'Settings':
        """
        Get singleton instance of Settings.
        
        Args:
            force_reload: Force reload from environment variables
            
        Returns:
            Settings instance
        """
        if cls._instance is None or force_reload:
            cls._instance = cls.load_from_env()
        return cls._instance
    
    @classmethod
    def load_from_env(cls) -> 'Settings':
        """
        Load all settings from environment variables with validation.
        
        Returns:
            Settings instance with all configuration loaded
            
        Raises:
            ValueError: If any configuration value is invalid
        """
        try:
            # Load system settings first
            system = SystemSettings.from_env()
            is_rpi = system.platform == 'raspberry_pi'
            
            # Load all other settings
            settings = cls(
                system=system,
                camera=CameraSettings.from_env(is_rpi),
                detection=DetectionSettings.from_env(is_rpi),
                performance=PerformanceSettings.from_env(is_rpi),
                logging=LoggingSettings.from_env(),
                traffic_control=TrafficControlSettings.from_env(),
                network=NetworkSettings.from_env(is_rpi),
                training=TrainingSettings.from_env(is_rpi)
            )
            
            # Validate settings
            settings._validate()
            
            # Log configuration if logger is available
            settings._log_configuration()
            
            return settings
            
        except Exception as e:
            print(f"ERROR: Failed to load configuration: {e}")
            raise
    
    def _validate(self) -> None:
        """Validate configuration consistency"""
        # Validate traffic light timings
        if self.traffic_control.min_green_time >= self.traffic_control.max_green_time:
            raise ValueError(
                f"MIN_GREEN_TIME ({self.traffic_control.min_green_time}) must be less than "
                f"MAX_GREEN_TIME ({self.traffic_control.max_green_time})"
            )
        
        # Validate camera indices
        if not self.camera.use_test_images and self.camera.camera_a_index == self.camera.camera_b_index:
            print("WARNING: CAMERA_A_INDEX and CAMERA_B_INDEX are the same. Using single camera for both directions.")
        
        # Validate test image paths if using test images
        if self.camera.use_test_images:
            if not Path(self.camera.test_image_path_a).exists():
                print(f"WARNING: Test image A not found at {self.camera.test_image_path_a}")
            if not Path(self.camera.test_image_path_b).exists():
                print(f"WARNING: Test image B not found at {self.camera.test_image_path_b}")
        
        # Validate model paths
        if self.detection.use_ml_model:
            if not Path(self.detection.model_path).exists():
                print(f"WARNING: ML model not found at {self.detection.model_path}. Will use CV fallback.")
            if not Path(self.detection.labels_path).exists():
                print(f"WARNING: Labels file not found at {self.detection.labels_path}")
    
    def _log_configuration(self) -> None:
        """Log configuration on startup using the logger if available"""
        # Use simple print to avoid circular dependency issues with logger
        # The logger itself will be configured using these settings
        print("=" * 60)
        print("Smart Traffic Light System Configuration")
        print("=" * 60)
        print(f"System Mode: {self.system.mode}")
        print(f"Platform: {self.system.platform}")
        print(f"Debug: {self.system.debug}")
        print(f"Camera Resolution: {self.camera.width}x{self.camera.height} @ {self.camera.fps}fps")
        print(f"Detection: ML={'Yes' if self.detection.use_ml_model else 'No'}, "
              f"Confidence={self.detection.min_confidence}")
        print(f"Performance: Max Processing Time={self.performance.max_processing_time}s, "
              f"Threads={self.performance.thread_count}")
        print(f"Network: {self.network.orchestrator_host}:{self.network.orchestrator_port}")
        print("=" * 60)
    
    def to_dict(self) -> dict:
        """
        Convert settings to dictionary for debugging.
        
        Returns:
            Dictionary representation of all settings
        """
        return {
            'system': asdict(self.system),
            'camera': asdict(self.camera),
            'detection': asdict(self.detection),
            'performance': asdict(self.performance),
            'logging': asdict(self.logging),
            'traffic_control': asdict(self.traffic_control),
            'network': asdict(self.network),
            'training': asdict(self.training)
        }
    
    def __repr__(self) -> str:
        """String representation of settings"""
        import json
        return json.dumps(self.to_dict(), indent=2)


# Convenience function for quick access
def get_settings(force_reload: bool = False) -> Settings:
    """
    Get the global Settings instance.
    
    Args:
        force_reload: Force reload from environment variables
        
    Returns:
        Settings instance
    """
    return Settings.get_instance(force_reload)


if __name__ == "__main__":
    # Test configuration loading
    print("Loading configuration...")
    try:
        settings = get_settings()
        print("\nConfiguration loaded successfully!")
        print(settings)
    except Exception as e:
        print(f"\nFailed to load configuration: {e}")
        import traceback
        traceback.print_exc()