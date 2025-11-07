"""
Type-Safe Camera Source Abstractions

This module provides a unified interface for different camera sources, eliminating
type mixing issues between cv2.VideoCapture objects and numpy arrays.

Key Features:
- Abstract base class defining consistent interface for all camera types
- LiveCameraSource: Wraps cv2.VideoCapture for real cameras
- StaticImageSource: Wraps static images for testing
- VideoFileSource: Wraps video files with looping support
- CameraFactory: Factory pattern for creating appropriate camera sources
- Full integration with ResourceTracker for leak detection
- Context manager support for safe resource management

Usage:
    # Using factory
    camera = CameraFactory.create(camera_index=0)
    
    # Using context manager
    with LiveCameraSource(0) as camera:
        success, frame = camera.read()
        if success:
            # Process frame
            pass
    
    # Static image for testing
    test_camera = StaticImageSource('test_image.jpg')
    success, frame = test_camera.read()
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from src.utils.logger import get_logger
from src.utils.resource_manager import ResourceTracker, get_global_tracker
from src.settings.settings import get_settings


class CameraSource(ABC):
    """
    Abstract base class for all camera sources.
    
    Provides consistent interface for reading frames regardless of source type.
    All camera sources return frames in the same format: (success: bool, frame: np.ndarray)
    """
    
    def __init__(self):
        """Initialize base camera source."""
        self.logger = get_logger(self.__class__.__name__)
        self._is_opened = False
        
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera source.
        
        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray])
            - success: True if frame was read successfully
            - frame: Frame data as numpy array, or None on failure
        """
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """
        Check if camera source is ready for reading.
        
        Returns:
            True if source is opened and ready
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass
    
    @abstractmethod
    def get_properties(self) -> Dict[str, Any]:
        """
        Get camera source properties.
        
        Returns:
            Dictionary with camera properties (width, height, fps, etc.)
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures resource cleanup."""
        self.release()
        return False  # Don't suppress exceptions


class LiveCameraSource(CameraSource):
    """
    Camera source for live camera feeds via cv2.VideoCapture.
    
    Features:
    - Wraps cv2.VideoCapture with consistent interface
    - Integrates with ResourceTracker for leak detection
    - Configurable camera properties (resolution, FPS, etc.)
    - Safe resource management
    
    Usage:
        with LiveCameraSource(0) as camera:
            success, frame = camera.read()
    """
    
    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        buffer_size: Optional[int] = None,
        resource_tracker: Optional[ResourceTracker] = None
    ):
        """
        Initialize live camera source.
        
        Args:
            camera_index: Camera device index (0, 1, 2, etc.)
            width: Optional frame width
            height: Optional frame height
            fps: Optional frame rate
            buffer_size: Optional buffer size
            resource_tracker: Optional resource tracker (uses global if None)
        """
        super().__init__()
        self.camera_index = camera_index
        self.resource_tracker = resource_tracker or get_global_tracker()
        self.camera: Optional[cv2.VideoCapture] = None
        
        # Load settings if properties not provided
        settings = get_settings()
        self.width = width or settings.camera.width
        self.height = height or settings.camera.height
        self.fps = fps or settings.camera.fps
        self.buffer_size = buffer_size or settings.camera.buffer_size
        
        self._open()
    
    def _open(self) -> None:
        """Open the camera and configure properties."""
        try:
            self.logger.info(f"Opening camera at index {self.camera_index}")
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
            
            # Set camera properties
            self._configure_properties()
            
            # Track with resource tracker
            self.resource_tracker.track_camera(self.camera)
            
            self._is_opened = True
            self.logger.info(f"Camera {self.camera_index} opened successfully")
            
            # Verify camera by reading test frame
            ret, frame = self.camera.read()
            if not ret or frame is None:
                raise RuntimeError(f"Camera {self.camera_index} opened but cannot read frames")
            
            self.logger.debug(f"Camera {self.camera_index} test read successful: {frame.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to open camera {self.camera_index}: {e}")
            self._is_opened = False
            raise
    
    def _configure_properties(self) -> None:
        """Configure camera properties (resolution, FPS, buffer)."""
        if self.camera is None:
            return
        
        try:
            # Set resolution
            if not self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width):
                self.logger.warning(f"Failed to set camera width to {self.width}")
            
            if not self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height):
                self.logger.warning(f"Failed to set camera height to {self.height}")
            
            # Set FPS
            if not self.camera.set(cv2.CAP_PROP_FPS, self.fps):
                self.logger.warning(f"Failed to set camera FPS to {self.fps}")
            
            # Set buffer size
            if not self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size):
                self.logger.warning(f"Failed to set camera buffer size to {self.buffer_size}")
            
            # Log actual properties
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(
                f"Camera {self.camera_index} configured: "
                f"{actual_width:.0f}x{actual_height:.0f} @ {actual_fps:.0f}fps"
            )
            
        except Exception as e:
            self.logger.warning(f"Error configuring camera properties: {e}")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the camera.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self._is_opened or self.camera is None:
            self.logger.error("Camera not opened")
            return False, None
        
        try:
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                self.logger.warning(f"Failed to read frame from camera {self.camera_index}")
                return False, None
            
            return True, frame
            
        except Exception as e:
            self.logger.error(f"Error reading from camera {self.camera_index}: {e}")
            return False, None
    
    def is_opened(self) -> bool:
        """Check if camera is opened."""
        return self._is_opened and self.camera is not None and self.camera.isOpened()
    
    def release(self) -> None:
        """Release camera resources."""
        if self.camera is not None:
            try:
                self.resource_tracker.release_camera(self.camera)
                self.camera = None
                self._is_opened = False
                self.logger.info(f"Camera {self.camera_index} released")
            except Exception as e:
                self.logger.warning(f"Error releasing camera {self.camera_index}: {e}")
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get camera properties.
        
        Returns:
            Dictionary with camera properties
        """
        if not self.is_opened() or self.camera is None:
            return {
                'source_type': 'live_camera',
                'camera_index': self.camera_index,
                'is_opened': False
            }
        
        return {
            'source_type': 'live_camera',
            'camera_index': self.camera_index,
            'width': int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.camera.get(cv2.CAP_PROP_FPS)),
            'buffer_size': self.buffer_size,
            'is_opened': True
        }


class StaticImageSource(CameraSource):
    """
    Camera source for static images (testing/development).
    
    Features:
    - Returns same image on each read() call
    - Simulates camera behavior for testing
    - Supports loading from file path or numpy array
    - Memory efficient (stores single image)
    
    Usage:
        camera = StaticImageSource('test_image.jpg')
        success, frame = camera.read()  # Returns copy of image
    """
    
    def __init__(self, image_source: str | np.ndarray):
        """
        Initialize static image source.
        
        Args:
            image_source: Path to image file or numpy array
        """
        super().__init__()
        self.image: Optional[np.ndarray] = None
        self.image_path: Optional[str] = None
        
        if isinstance(image_source, str):
            self.image_path = image_source
            self._load_image(image_source)
        elif isinstance(image_source, np.ndarray):
            self.image = image_source.copy()
            self._is_opened = True
            self.logger.info(f"Static image loaded from array: {self.image.shape}")
        else:
            raise ValueError(f"Invalid image_source type: {type(image_source)}")
    
    def _load_image(self, image_path: str) -> None:
        """Load image from file."""
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            self.logger.info(f"Loading static image from: {image_path}")
            self.image = cv2.imread(str(path))
            
            if self.image is None or self.image.size == 0:
                raise ValueError(f"Failed to load image: {image_path}")
            
            self._is_opened = True
            self.logger.info(f"Static image loaded successfully: {self.image.shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to load image from {image_path}: {e}")
            self._is_opened = False
            raise
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read frame (returns copy of static image).
        
        Returns:
            Tuple of (success, frame copy)
        """
        if not self._is_opened or self.image is None:
            self.logger.error("Static image not loaded")
            return False, None
        
        # Return a copy to simulate camera behavior
        return True, self.image.copy()
    
    def is_opened(self) -> bool:
        """Check if image is loaded."""
        return self._is_opened and self.image is not None
    
    def release(self) -> None:
        """Release image resources."""
        self.image = None
        self._is_opened = False
        self.logger.debug("Static image source released")
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get image properties.
        
        Returns:
            Dictionary with image properties
        """
        if not self.is_opened() or self.image is None:
            return {
                'source_type': 'static_image',
                'image_path': self.image_path,
                'is_opened': False
            }
        
        height, width = self.image.shape[:2]
        channels = self.image.shape[2] if len(self.image.shape) == 3 else 1
        
        return {
            'source_type': 'static_image',
            'image_path': self.image_path,
            'width': width,
            'height': height,
            'channels': channels,
            'dtype': str(self.image.dtype),
            'is_opened': True
        }


class VideoFileSource(CameraSource):
    """
    Camera source for video files with looping support.
    
    Features:
    - Reads from video files
    - Optional looping for continuous playback
    - Integrates with ResourceTracker
    - Similar interface to live camera
    
    Usage:
        camera = VideoFileSource('video.mp4', loop=True)
        success, frame = camera.read()
    """
    
    def __init__(
        self,
        video_path: str,
        loop: bool = True,
        resource_tracker: Optional[ResourceTracker] = None
    ):
        """
        Initialize video file source.
        
        Args:
            video_path: Path to video file
            loop: Whether to loop video when it ends
            resource_tracker: Optional resource tracker
        """
        super().__init__()
        self.video_path = video_path
        self.loop = loop
        self.resource_tracker = resource_tracker or get_global_tracker()
        self.video: Optional[cv2.VideoCapture] = None
        self.frame_count = 0
        self.total_frames = 0
        
        self._open()
    
    def _open(self) -> None:
        """Open video file."""
        try:
            path = Path(self.video_path)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
            self.logger.info(f"Opening video file: {self.video_path}")
            self.video = cv2.VideoCapture(str(path))
            
            if not self.video.isOpened():
                raise RuntimeError(f"Failed to open video file: {self.video_path}")
            
            # Get video properties
            self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Track with resource tracker
            self.resource_tracker.track_camera(self.video)
            
            self._is_opened = True
            self.logger.info(
                f"Video file opened: {self.video_path} "
                f"({self.total_frames} frames, loop={self.loop})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to open video file {self.video_path}: {e}")
            self._is_opened = False
            raise
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from video.
        
        Returns:
            Tuple of (success, frame)
        """
        if not self._is_opened or self.video is None:
            self.logger.error("Video file not opened")
            return False, None
        
        try:
            ret, frame = self.video.read()
            
            # Handle end of video
            if not ret or frame is None:
                if self.loop and self.total_frames > 0:
                    # Reset to beginning
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_count = 0
                    self.logger.debug("Looping video to start")
                    ret, frame = self.video.read()
                else:
                    self.logger.info("Video file ended")
                    return False, None
            
            if ret and frame is not None:
                self.frame_count += 1
                return True, frame
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error reading from video file: {e}")
            return False, None
    
    def is_opened(self) -> bool:
        """Check if video file is opened."""
        return self._is_opened and self.video is not None and self.video.isOpened()
    
    def release(self) -> None:
        """Release video file resources."""
        if self.video is not None:
            try:
                self.resource_tracker.release_camera(self.video)
                self.video = None
                self._is_opened = False
                self.logger.info(f"Video file released: {self.video_path}")
            except Exception as e:
                self.logger.warning(f"Error releasing video file: {e}")
    
    def get_properties(self) -> Dict[str, Any]:
        """
        Get video file properties.
        
        Returns:
            Dictionary with video properties
        """
        if not self.is_opened() or self.video is None:
            return {
                'source_type': 'video_file',
                'video_path': self.video_path,
                'is_opened': False
            }
        
        return {
            'source_type': 'video_file',
            'video_path': self.video_path,
            'width': int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.video.get(cv2.CAP_PROP_FPS)),
            'total_frames': self.total_frames,
            'current_frame': self.frame_count,
            'loop': self.loop,
            'is_opened': True
        }


class CameraFactory:
    """
    Factory for creating appropriate camera sources based on configuration.
    
    Features:
    - Auto-selects camera type based on inputs
    - Integrates with Settings for configuration
    - Supports all camera source types
    - Proper error handling
    
    Usage:
        # Live camera
        camera = CameraFactory.create(camera_index=0)
        
        # Static image
        camera = CameraFactory.create(test_image_path='test.jpg')
        
        # Video file
        camera = CameraFactory.create(video_file='video.mp4')
        
        # From settings
        camera = CameraFactory.create_from_settings('camera_a')
    """
    
    @staticmethod
    def create(
        camera_index: Optional[int] = None,
        test_image_path: Optional[str] = None,
        video_file: Optional[str] = None,
        **kwargs
    ) -> CameraSource:
        """
        Create appropriate camera source based on inputs.
        
        Args:
            camera_index: Camera device index for live camera
            test_image_path: Path to test image for static source
            video_file: Path to video file for video source
            **kwargs: Additional arguments for camera source
        
        Returns:
            Appropriate CameraSource instance
            
        Raises:
            ValueError: If invalid combination of arguments provided
        """
        logger = get_logger('CameraFactory')
        
        # Priority: test_image > video_file > camera_index
        if test_image_path:
            logger.info(f"Creating StaticImageSource from {test_image_path}")
            return StaticImageSource(test_image_path)
        
        elif video_file:
            logger.info(f"Creating VideoFileSource from {video_file}")
            loop = kwargs.get('loop', True)
            return VideoFileSource(video_file, loop=loop)
        
        elif camera_index is not None:
            logger.info(f"Creating LiveCameraSource for index {camera_index}")
            return LiveCameraSource(camera_index, **kwargs)
        
        else:
            raise ValueError(
                "Must provide one of: camera_index, test_image_path, or video_file"
            )
    
    @staticmethod
    def create_from_settings(camera_name: str = 'camera_a') -> CameraSource:
        """
        Create camera source from Settings configuration.
        
        Args:
            camera_name: Which camera to create ('camera_a' or 'camera_b')
            
        Returns:
            Configured CameraSource instance
        """
        logger = get_logger('CameraFactory')
        settings = get_settings()
        
        # Determine camera index
        if camera_name.lower() == 'camera_a':
            camera_index = settings.camera.camera_a_index
            test_image_path = settings.camera.test_image_path_a
        elif camera_name.lower() == 'camera_b':
            camera_index = settings.camera.camera_b_index
            test_image_path = settings.camera.test_image_path_b
        else:
            raise ValueError(f"Invalid camera_name: {camera_name}. Must be 'camera_a' or 'camera_b'")
        
        # Check if using test images
        if settings.camera.use_test_images:
            logger.info(f"Creating {camera_name} from test image: {test_image_path}")
            try:
                return StaticImageSource(test_image_path)
            except Exception as e:
                logger.warning(f"Failed to load test image, falling back to live camera: {e}")
        
        # Create live camera with settings
        logger.info(f"Creating {camera_name} from live camera index {camera_index}")
        return LiveCameraSource(
            camera_index=camera_index,
            width=settings.camera.width,
            height=settings.camera.height,
            fps=settings.camera.fps,
            buffer_size=settings.camera.buffer_size
        )


# Convenience function for quick camera creation
def create_camera(
    camera_index: Optional[int] = None,
    test_image: Optional[str] = None,
    video_file: Optional[str] = None
) -> CameraSource:
    """
    Convenience function for creating camera sources.
    
    Args:
        camera_index: Camera device index
        test_image: Path to test image
        video_file: Path to video file
        
    Returns:
        CameraSource instance
    """
    return CameraFactory.create(
        camera_index=camera_index,
        test_image_path=test_image,
        video_file=video_file
    )