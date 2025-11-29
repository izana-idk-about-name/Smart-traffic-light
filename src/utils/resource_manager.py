"""
Resource Management Utilities for Smart Traffic Light System

This module provides comprehensive resource management to prevent memory leaks:
- TempFileManager: Safe temporary file handling with automatic cleanup
- ResourceTracker: Track and manage OpenCV resources (cameras, windows, mats)
- FrameBuffer: Bounded buffer for visualization frames with memory limits
- CameraContextManager: Context manager for camera resource lifecycle

All classes are thread-safe and integrate with the logging system.
"""

import os
import cv2
import shutil
import psutil
import threading
import weakref
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from contextlib import contextmanager
from collections import deque
from datetime import datetime

from src.utils.logger import get_logger


class TempFileManager:
    """
    Thread-safe context manager for temporary file handling with automatic cleanup.
    
    Features:
    - Automatic cleanup on exit
    - Tracks all created temp files
    - Periodic cleanup of orphaned files
    - Exception-safe (cleanup even on errors)
    
    Usage:
        with TempFileManager() as tmp:
            temp_path = tmp.create_temp_file('.jpg')
            # Use temp_path...
            # File automatically deleted on exit
    """
    
    def __init__(self, prefix: str = "traffic_", cleanup_on_exit: bool = True):
        """
        Initialize temp file manager.
        
        Args:
            prefix: Prefix for temp file names
            cleanup_on_exit: Whether to clean up on context exit
        """
        self.prefix = prefix
        self.cleanup_on_exit = cleanup_on_exit
        self.temp_files: List[str] = []
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
        
    def create_temp_file(self, suffix: str = "", delete: bool = False) -> str:
        """
        Create a temporary file and track it for cleanup.
        
        Args:
            suffix: File extension (e.g., '.jpg', '.png')
            delete: Whether tempfile module should auto-delete (we handle it)
            
        Returns:
            Path to temporary file
        """
        with self.lock:
            try:
                # Use tempfile for secure temp file creation
                fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=self.prefix)
                os.close(fd)  # Close file descriptor immediately
                
                self.temp_files.append(temp_path)
                self.logger.debug(f"Created temp file: {temp_path}")
                
                return temp_path
                
            except Exception as e:
                self.logger.error(f"Failed to create temp file: {e}", exc_info=True)
                raise
    
    def create_temp_dir(self) -> str:
        """
        Create a temporary directory and track it for cleanup.
        
        Returns:
            Path to temporary directory
        """
        with self.lock:
            try:
                temp_dir = tempfile.mkdtemp(prefix=self.prefix)
                self.temp_files.append(temp_dir)
                self.logger.debug(f"Created temp directory: {temp_dir}")
                
                return temp_dir
                
            except Exception as e:
                self.logger.error(f"Failed to create temp directory: {e}", exc_info=True)
                raise
    
    def cleanup(self) -> None:
        """Clean up all tracked temporary files and directories."""
        with self.lock:
            for temp_path in self.temp_files[:]:  # Copy list to avoid modification during iteration
                try:
                    if os.path.isfile(temp_path):
                        os.remove(temp_path)
                        self.logger.debug(f"Removed temp file: {temp_path}")
                    elif os.path.isdir(temp_path):
                        shutil.rmtree(temp_path)
                        self.logger.debug(f"Removed temp directory: {temp_path}")
                    
                    self.temp_files.remove(temp_path)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp path {temp_path}: {e}")
            
            if self.temp_files:
                self.logger.warning(f"Failed to clean up {len(self.temp_files)} temp files")
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up."""
        if self.cleanup_on_exit:
            self.cleanup()
        return False  # Don't suppress exceptions
    
    @staticmethod
    def cleanup_orphaned_files(directory: str = "/tmp", prefix: str = "traffic_", 
                               max_age_hours: int = 24) -> int:
        """
        Clean up orphaned temp files older than specified age.
        
        Args:
            directory: Directory to search for orphaned files
            prefix: File prefix to match
            max_age_hours: Maximum age in hours before cleanup
            
        Returns:
            Number of files cleaned up
        """
        logger = get_logger(__name__)
        cleaned_count = 0
        
        try:
            current_time = datetime.now().timestamp()
            max_age_seconds = max_age_hours * 3600
            
            for entry in os.scandir(directory):
                if entry.name.startswith(prefix):
                    try:
                        file_age = current_time - entry.stat().st_mtime
                        if file_age > max_age_seconds:
                            if entry.is_file():
                                os.remove(entry.path)
                            elif entry.is_dir():
                                shutil.rmtree(entry.path)
                            
                            cleaned_count += 1
                            logger.debug(f"Cleaned up orphaned file: {entry.path}")
                            
                    except Exception as e:
                        logger.debug(f"Failed to clean up {entry.path}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} orphaned temp files")
            
        except Exception as e:
            logger.error(f"Error during orphaned file cleanup: {e}", exc_info=True)
        
        return cleaned_count


class ResourceTracker:
    """
    Track OpenCV and system resources to detect and prevent leaks.
    
    Features:
    - Track VideoCapture objects
    - Track OpenCV windows
    - Monitor memory usage
    - Provide cleanup methods
    - Detect resource leaks
    
    Usage:
        tracker = ResourceTracker()
        camera = tracker.track_camera(cv2.VideoCapture(0))
        # ... use camera ...
        tracker.release_all()
    """
    
    def __init__(self):
        """Initialize resource tracker."""
        # Use dict instead of weakref set since cv2.VideoCapture doesn't support weakref
        self.cameras: Dict[int, cv2.VideoCapture] = {}
        self.windows: Set[str] = set()
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
        self.initial_memory = self._get_memory_usage()
        
    def track_camera(self, camera: cv2.VideoCapture) -> cv2.VideoCapture:
        """
        Track a VideoCapture object for cleanup.
        
        Args:
            camera: VideoCapture instance to track
            
        Returns:
            The same camera instance
        """
        with self.lock:
            # Use id as key since cv2.VideoCapture doesn't support weakref
            camera_id = id(camera)
            self.cameras[camera_id] = camera
            self.logger.debug(f"Tracking camera: {camera_id}")
        
        return camera
    
    def track_window(self, window_name: str) -> None:
        """
        Track an OpenCV window for cleanup.
        
        Args:
            window_name: Name of the window
        """
        with self.lock:
            self.windows.add(window_name)
            self.logger.debug(f"Tracking window: {window_name}")
    
    def release_camera(self, camera: cv2.VideoCapture) -> None:
        """
        Release a specific camera and stop tracking it.
        
        Args:
            camera: Camera to release
        """
        with self.lock:
            try:
                camera_id = id(camera)
                if camera is not None and camera.isOpened():
                    camera.release()
                    self.logger.debug(f"Released camera: {camera_id}")
                # Remove from tracking
                self.cameras.pop(camera_id, None)
            except Exception as e:
                self.logger.warning(f"Error releasing camera: {e}")
    
    def destroy_window(self, window_name: str) -> None:
        """
        Destroy a specific OpenCV window.
        
        Args:
            window_name: Name of window to destroy
        """
        with self.lock:
            try:
                cv2.destroyWindow(window_name)
                self.windows.discard(window_name)
                self.logger.debug(f"Destroyed window: {window_name}")
            except Exception as e:
                self.logger.warning(f"Error destroying window {window_name}: {e}")
    
    def release_all(self) -> None:
        """Release all tracked resources."""
        with self.lock:
            # Release cameras
            for camera_id, camera in list(self.cameras.items()):
                try:
                    if camera is not None and camera.isOpened():
                        camera.release()
                        self.logger.debug(f"Released camera: {camera_id}")
                except Exception as e:
                    self.logger.warning(f"Error releasing camera {camera_id}: {e}")
            
            self.cameras.clear()
            
            # Destroy windows
            for window_name in list(self.windows):
                try:
                    cv2.destroyWindow(window_name)
                except Exception as e:
                    self.logger.debug(f"Error destroying window {window_name}: {e}")
            
            self.windows.clear()
            cv2.destroyAllWindows()
            
            self.logger.info("Released all tracked resources")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Returns:
            Dictionary with resource statistics
        """
        with self.lock:
            alive_cameras = sum(1 for cam in self.cameras.values() if cam is not None and cam.isOpened())
            current_memory = self._get_memory_usage()
            memory_delta = current_memory - self.initial_memory
            
            return {
                'tracked_cameras': len(self.cameras),
                'alive_cameras': alive_cameras,
                'tracked_windows': len(self.windows),
                'memory_mb': current_memory,
                'memory_delta_mb': memory_delta,
                'potential_leaks': len(self.cameras) - alive_cameras
            }
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in MB
        """
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def log_statistics(self) -> None:
        """Log current resource statistics."""
        stats = self.get_statistics()
        self.logger.info(
            f"Resource Stats - Cameras: {stats['alive_cameras']}/{stats['tracked_cameras']}, "
            f"Windows: {stats['tracked_windows']}, "
            f"Memory: {stats['memory_mb']:.1f}MB (Δ{stats['memory_delta_mb']:+.1f}MB)"
        )


class FrameBuffer:
    """
    Thread-safe bounded buffer for visualization frames with automatic rotation.
    
    Features:
    - Fixed maximum size
    - Automatic rotation when full
    - Memory limit enforcement
    - Disk space checking
    - JPEG compression
    
    Usage:
        buffer = FrameBuffer(max_frames=100, output_dir='frames')
        buffer.add_frame(frame, 'camera_a')
        buffer.save_current()
    """
    
    def __init__(self, max_frames: int = 100, output_dir: str = 'detection_frames',
                 max_memory_mb: int = 100, jpeg_quality: int = 85):
        """
        Initialize frame buffer.
        
        Args:
            max_frames: Maximum number of frames to keep
            output_dir: Directory to save frames
            max_memory_mb: Maximum memory usage in MB
            jpeg_quality: JPEG compression quality (1-100)
        """
        self.max_frames = max_frames
        self.output_dir = Path(output_dir)
        self.max_memory_mb = max_memory_mb
        self.jpeg_quality = jpeg_quality
        self.frames: deque = deque(maxlen=max_frames)
        self.lock = threading.Lock()
        self.logger = get_logger(__name__)
        self.frame_count = 0
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_frame(self, frame, camera_name: str, metadata: Optional[Dict] = None) -> None:
        """
        Add a frame to the buffer.
        
        Args:
            frame: Frame to add
            camera_name: Name of camera
            metadata: Optional metadata to store with frame
        """
        with self.lock:
            self.frames.append({
                'frame': frame,
                'camera': camera_name,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            })
            self.frame_count += 1
    
    def save_current(self, frame, camera_name: str, cycle_number: int) -> Optional[str]:
        """
        Save current frame with rotation and disk space checking.
        
        Args:
            frame: Frame to save
            camera_name: Camera name
            cycle_number: Current cycle number
            
        Returns:
            Path to saved frame or None if not saved
        """
        try:
            # Check disk space
            if not self._check_disk_space():
                self.logger.warning("Insufficient disk space, skipping frame save")
                return None
            
            # Check memory usage
            if not self._check_memory_usage():
                self.logger.warning("Memory limit reached, skipping frame save")
                return None
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_{camera_name}_cycle_{cycle_number}_{timestamp}.jpg"
            filepath = self.output_dir / filename
            
            # Save with compression
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success = cv2.imwrite(str(filepath), frame, encode_params)
            
            if success:
                self.logger.debug(f"Saved frame: {filepath}")
                
                # Add to buffer for tracking
                self.add_frame(None, camera_name, {'path': str(filepath)})
                
                # Rotate old frames if needed
                self._rotate_old_frames()
                
                return str(filepath)
            else:
                self.logger.warning(f"Failed to save frame: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving frame: {e}", exc_info=True)
            return None
    
    def _check_disk_space(self, min_free_percent: float = 10.0) -> bool:
        """
        Check if sufficient disk space is available.
        
        Args:
            min_free_percent: Minimum free space percentage
            
        Returns:
            True if sufficient space available
        """
        try:
            stat = shutil.disk_usage(self.output_dir)
            free_percent = (stat.free / stat.total) * 100
            
            if free_percent < min_free_percent:
                self.logger.warning(
                    f"Low disk space: {free_percent:.1f}% free "
                    f"(threshold: {min_free_percent}%)"
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking disk space: {e}")
            return True  # Fail open
    
    def _check_memory_usage(self) -> bool:
        """
        Check if memory usage is within limits.
        
        Returns:
            True if within memory limits
        """
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.max_memory_mb:
                self.logger.warning(
                    f"Memory limit reached: {memory_mb:.1f}MB "
                    f"(limit: {self.max_memory_mb}MB)"
                )
                return False
            
            return True
            
        except Exception:
            return True  # Fail open
    
    def _rotate_old_frames(self) -> None:
        """Delete old frames to maintain max_frames limit."""
        try:
            # Get all frame files
            frame_files = sorted(self.output_dir.glob("camera_*.jpg"))
            
            # Delete oldest if exceeding limit
            if len(frame_files) > self.max_frames:
                files_to_delete = frame_files[:len(frame_files) - self.max_frames]
                
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        self.logger.debug(f"Rotated old frame: {file_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to delete {file_path}: {e}")
                
                self.logger.info(f"Rotated {len(files_to_delete)} old frames")
                
        except Exception as e:
            self.logger.error(f"Error rotating frames: {e}", exc_info=True)
    
    def clear(self) -> None:
        """Clear all frames from buffer."""
        with self.lock:
            self.frames.clear()
            self.logger.debug("Cleared frame buffer")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        with self.lock:
            try:
                # Estimate frame memory
                total_size = 0
                for frame_data in self.frames:
                    if frame_data['frame'] is not None:
                        total_size += frame_data['frame'].nbytes
                
                # Get disk usage
                disk_files = list(self.output_dir.glob("camera_*.jpg"))
                disk_size = sum(f.stat().st_size for f in disk_files)
                
                return {
                    'buffer_frames': len(self.frames),
                    'buffer_memory_mb': total_size / 1024 / 1024,
                    'disk_frames': len(disk_files),
                    'disk_size_mb': disk_size / 1024 / 1024,
                    'total_frames_processed': self.frame_count
                }
                
            except Exception as e:
                self.logger.error(f"Error getting memory usage: {e}")
                return {}


class CameraContextManager:
    """
    Context manager for safe camera resource management.
    
    Ensures cameras are properly released even on exceptions.
    Integrates with ResourceTracker for leak detection.
    
    Usage:
        with CameraContextManager(camera_index=0, tracker=tracker) as camera:
            ret, frame = camera.read()
            # ... use camera ...
        # Camera automatically released
    """
    
    def __init__(self, camera_index: int = 0, tracker: Optional[ResourceTracker] = None):
        """
        Initialize camera context manager.
        
        Args:
            camera_index: Camera device index
            tracker: Optional resource tracker
        """
        self.camera_index = camera_index
        self.tracker = tracker
        self.camera: Optional[cv2.VideoCapture] = None
        self.logger = get_logger(__name__)
    
    def __enter__(self) -> cv2.VideoCapture:
        """Open and return camera."""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")
            
            # Track with resource tracker if available
            if self.tracker:
                self.tracker.track_camera(self.camera)
            
            self.logger.debug(f"Opened camera {self.camera_index}")
            return self.camera
            
        except Exception as e:
            self.logger.error(f"Failed to open camera {self.camera_index}: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release camera resources."""
        if self.camera is not None:
            try:
                self.camera.release()
                self.logger.debug(f"Released camera {self.camera_index}")
            except Exception as e:
                self.logger.warning(f"Error releasing camera {self.camera_index}: {e}")
        
        return False  # Don't suppress exceptions


# Global resource tracker instance
_global_tracker: Optional[ResourceTracker] = None
_tracker_lock = threading.Lock()


def get_global_tracker() -> ResourceTracker:
    """
    Get or create global resource tracker instance.
    
    Returns:
        Global ResourceTracker instance
    """
    global _global_tracker
    
    with _tracker_lock:
        if _global_tracker is None:
            _global_tracker = ResourceTracker()
        return _global_tracker


@contextmanager
def managed_camera(camera_index: int = 0):
    """
    Convenience context manager for camera access.
    
    Args:
        camera_index: Camera device index
        
    Yields:
        VideoCapture instance
        
    Example:
        with managed_camera(0) as camera:
            ret, frame = camera.read()
    """
    tracker = get_global_tracker()
    with CameraContextManager(camera_index, tracker) as camera:
        yield camera