# Resource Management Guide

## Overview

The Smart Traffic Light system now includes comprehensive resource management to prevent memory leaks and ensure proper cleanup of system resources. This guide explains how to use the resource management utilities.

## Components

### 1. TempFileManager

Safe temporary file handling with automatic cleanup.

**Features:**
- Automatic cleanup on context exit
- Tracks all created temp files
- Periodic cleanup of orphaned files
- Exception-safe (cleanup even on errors)

**Usage:**

```python
from src.utils.resource_manager import TempFileManager

# Basic usage with context manager
with TempFileManager() as tmp:
    temp_path = tmp.create_temp_file('.jpg')
    # Use temp file...
    cv2.imwrite(temp_path, frame)
    # File automatically deleted on exit

# Create temporary directory
with TempFileManager() as tmp:
    temp_dir = tmp.create_temp_dir()
    # Use directory...
    # Directory automatically deleted on exit
```

**Cleanup Orphaned Files:**

```python
# Clean up orphaned files older than 24 hours
cleaned_count = TempFileManager.cleanup_orphaned_files(
    directory="/tmp",
    prefix="traffic_",
    max_age_hours=24
)
print(f"Cleaned up {cleaned_count} orphaned files")
```

### 2. ResourceTracker

Track OpenCV and system resources to detect and prevent leaks.

**Features:**
- Track VideoCapture objects
- Track OpenCV windows
- Monitor memory usage
- Provide cleanup methods
- Detect resource leaks

**Usage:**

```python
from src.utils.resource_manager import ResourceTracker, get_global_tracker

# Get global tracker instance
tracker = get_global_tracker()

# Track a camera
camera = cv2.VideoCapture(0)
tracker.track_camera(camera)

# Track a window
cv2.namedWindow("My Window")
tracker.track_window("My Window")

# Get statistics
stats = tracker.get_statistics()
print(f"Alive cameras: {stats['alive_cameras']}")
print(f"Memory usage: {stats['memory_mb']:.1f}MB")

# Release specific resources
tracker.release_camera(camera)
tracker.destroy_window("My Window")

# Release all tracked resources
tracker.release_all()

# Log statistics
tracker.log_statistics()
```

### 3. FrameBuffer

Bounded buffer for visualization frames with automatic rotation and memory management.

**Features:**
- Fixed maximum size
- Automatic rotation when full
- Memory limit enforcement
- Disk space checking
- JPEG compression

**Usage:**

```python
from src.utils.resource_manager import FrameBuffer

# Create frame buffer
buffer = FrameBuffer(
    max_frames=100,
    output_dir='detection_frames',
    max_memory_mb=100,
    jpeg_quality=85
)

# Save frame with automatic rotation
saved_path = buffer.save_current(frame, camera_name='A', cycle_number=1)
if saved_path:
    print(f"Frame saved: {saved_path}")

# Add frame to memory buffer
buffer.add_frame(frame, camera_name='A', metadata={'count': 5})

# Get memory usage statistics
stats = buffer.get_memory_usage()
print(f"Disk frames: {stats['disk_frames']}")
print(f"Disk size: {stats['disk_size_mb']:.1f}MB")

# Clear buffer
buffer.clear()
```

### 4. CameraContextManager

Context manager for safe camera resource management.

**Features:**
- Automatic camera release on exit
- Integration with ResourceTracker
- Exception-safe cleanup

**Usage:**

```python
from src.utils.resource_manager import CameraContextManager, get_global_tracker

tracker = get_global_tracker()

# Use camera with context manager
with CameraContextManager(camera_index=0, tracker=tracker) as camera:
    ret, frame = camera.read()
    # Use camera...
# Camera automatically released

# Convenience function
from src.utils.resource_manager import managed_camera

with managed_camera(camera_index=0) as camera:
    ret, frame = camera.read()
    # Use camera...
# Camera automatically released
```

### 5. ManagedCamera (in camera.py)

Enhanced camera wrapper with full resource management.

**Usage:**

```python
from src.application.camera import ManagedCamera

# Use managed camera
with ManagedCamera(camera_index=0) as camera:
    ret, frame = camera.read()
    
    # Set properties
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Get properties
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    
    # Check if opened
    if camera.is_opened():
        print("Camera is ready")
# Camera automatically released
```

## Integration Examples

### Example 1: Processing Video with Resource Management

```python
from src.utils.resource_manager import managed_camera, FrameBuffer, get_global_tracker

tracker = get_global_tracker()
frame_buffer = FrameBuffer(max_frames=50, output_dir='output')

with managed_camera(0) as camera:
    cycle = 0
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Process frame...
        processed_frame = process_frame(frame)
        
        # Save with rotation
        frame_buffer.save_current(processed_frame, 'camera_a', cycle)
        cycle += 1
        
        # Check memory periodically
        if cycle % 100 == 0:
            tracker.log_statistics()
            stats = frame_buffer.get_memory_usage()
            print(f"Saved {stats['disk_frames']} frames")

# All resources automatically cleaned up
```

### Example 2: Multi-Camera Processing

```python
from src.utils.resource_manager import get_global_tracker, FrameBuffer
from src.application.camera import ManagedCamera

tracker = get_global_tracker()
buffer_a = FrameBuffer(max_frames=100, output_dir='frames/camera_a')
buffer_b = FrameBuffer(max_frames=100, output_dir='frames/camera_b')

try:
    with ManagedCamera(0) as cam_a, ManagedCamera(1) as cam_b:
        cycle = 0
        while True:
            ret_a, frame_a = cam_a.read()
            ret_b, frame_b = cam_b.read()
            
            if not (ret_a and ret_b):
                break
            
            # Process frames...
            buffer_a.save_current(frame_a, 'A', cycle)
            buffer_b.save_current(frame_b, 'B', cycle)
            cycle += 1
            
finally:
    # Cleanup
    tracker.release_all()
    tracker.log_statistics()
    
    # Print final stats
    print(f"Camera A frames: {buffer_a.get_memory_usage()['disk_frames']}")
    print(f"Camera B frames: {buffer_b.get_memory_usage()['disk_frames']}")
```

### Example 3: Temp File Processing

```python
from src.utils.resource_manager import TempFileManager
import cv2

def process_with_temp_file(frame):
    """Process frame using temporary file."""
    with TempFileManager(prefix="process_") as tmp:
        # Create temp file
        temp_path = tmp.create_temp_file('.jpg')
        
        # Save frame
        cv2.imwrite(temp_path, frame)
        
        # Process file
        result = external_processor(temp_path)
        
        # Temp file automatically deleted on exit
        return result
```

## Memory Leak Prevention

### Best Practices

1. **Always use context managers:**
   ```python
   # Good
   with managed_camera(0) as camera:
       frame = camera.read()
   
   # Avoid
   camera = cv2.VideoCapture(0)
   frame = camera.read()
   # May forget to release
   ```

2. **Track all resources:**
   ```python
   tracker = get_global_tracker()
   
   # Track cameras
   camera = cv2.VideoCapture(0)
   tracker.track_camera(camera)
   
   # Track windows
   cv2.namedWindow("Window")
   tracker.track_window("Window")
   ```

3. **Monitor resource usage:**
   ```python
   # Periodic monitoring
   if cycle % 100 == 0:
       tracker.log_statistics()
       stats = tracker.get_statistics()
       
       if stats['memory_delta_mb'] > 100:
           print("WARNING: Memory usage increased by >100MB")
   ```

4. **Clean up orphaned files:**
   ```python
   # Daily cleanup
   TempFileManager.cleanup_orphaned_files(
       prefix="traffic_",
       max_age_hours=24
   )
   ```

5. **Use frame rotation:**
   ```python
   # Limit disk usage
   frame_buffer = FrameBuffer(
       max_frames=100,  # Keep only last 100
       max_memory_mb=50  # Limit memory
   )
   ```

## Troubleshooting

### Memory Keeps Growing

Check for:
- Unreleased cameras: Use `tracker.get_statistics()` to see alive vs tracked cameras
- Unclosed windows: Check `tracked_windows` in statistics
- Temp files not cleaned: Run orphaned file cleanup
- Frame buffer not rotating: Verify `max_frames` setting

### Resources Not Cleaned Up

Ensure:
- Context managers are used properly (no early returns without cleanup)
- `finally` blocks include cleanup code
- Global tracker's `release_all()` is called on shutdown
- Signal handlers call cleanup methods

### Performance Issues

Optimize:
- Reduce `max_frames` in FrameBuffer
- Increase `jpeg_quality` for smaller files
- Enable disk space checking
- Monitor memory usage periodically

## Configuration

Resource management can be configured through environment variables:

```bash
# Memory limits
MEMORY_LIMIT_MB=512

# Frame buffer settings
FRAME_SAVE_INTERVAL=100
MAX_FRAMES_SAVED=100

# Temp file settings
TEMP_FILE_PREFIX=traffic_
TEMP_FILE_MAX_AGE_HOURS=24
```

## API Reference

### TempFileManager

- `create_temp_file(suffix)` - Create temporary file
- `create_temp_dir()` - Create temporary directory
- `cleanup()` - Clean up all tracked files
- `cleanup_orphaned_files(directory, prefix, max_age_hours)` - Static cleanup method

### ResourceTracker

- `track_camera(camera)` - Track VideoCapture
- `track_window(window_name)` - Track window
- `release_camera(camera)` - Release camera
- `destroy_window(window_name)` - Destroy window
- `release_all()` - Release all resources
- `get_statistics()` - Get resource stats
- `log_statistics()` - Log resource stats

### FrameBuffer

- `add_frame(frame, camera_name, metadata)` - Add frame to buffer
- `save_current(frame, camera_name, cycle_number)` - Save frame with rotation
- `clear()` - Clear buffer
- `get_memory_usage()` - Get memory statistics

### CameraContextManager / ManagedCamera

- Context manager protocol (`__enter__`, `__exit__`)
- `read()` - Read frame
- `get(prop)` - Get property
- `set(prop, value)` - Set property
- `is_opened()` - Check if opened

## Logging

All resource management operations are logged. Check logs for:

- Resource tracking events
- Memory usage warnings
- Cleanup operations
- Potential leaks

Log files:
- `logs/traffic_light.log` - Main log
- `logs/errors.log` - Error log
- `logs/performance.log` - Performance metrics