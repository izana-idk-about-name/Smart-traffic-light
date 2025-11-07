#!/usr/bin/env python3
"""
Resource Management Example for Smart Traffic Light System

This script demonstrates proper usage of resource management utilities
to prevent memory leaks and ensure proper cleanup.
"""

import cv2
import time
import numpy as np
from src.utils.resource_manager import (
    TempFileManager,
    ResourceTracker,
    FrameBuffer,
    managed_camera,
    get_global_tracker
)
from src.utils.logger import get_logger


def example_temp_files():
    """Example: Safe temporary file handling."""
    print("\n" + "="*60)
    print("Example 1: Temporary File Management")
    print("="*60)
    
    # Create a test frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Test Frame", (200, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Use TempFileManager for safe temp file handling
    with TempFileManager(prefix="example_") as tmp:
        temp_path = tmp.create_temp_file('.jpg')
        print(f"Created temp file: {temp_path}")
        
        # Save frame
        cv2.imwrite(temp_path, frame)
        print(f"Saved frame to temp file")
        
        # File will be automatically deleted when exiting context
        print("Temp file will be cleaned up automatically")
    
    print("✓ Temp file cleaned up successfully")


def example_camera_management():
    """Example: Camera resource management."""
    print("\n" + "="*60)
    print("Example 2: Camera Resource Management")
    print("="*60)
    
    tracker = get_global_tracker()
    
    try:
        # Use managed_camera context manager
        print("Opening camera with managed context...")
        with managed_camera(camera_index=0) as camera:
            print("✓ Camera opened successfully")
            
            # Capture a few frames
            for i in range(5):
                ret, frame = camera.read()
                if ret:
                    print(f"  Frame {i+1} captured: {frame.shape}")
                time.sleep(0.1)
            
            # Camera will be automatically released
            print("Camera will be released automatically")
        
        print("✓ Camera released successfully")
        
    except Exception as e:
        print(f"✗ Camera error (might not have camera): {e}")
    
    # Show resource statistics
    print("\nResource Statistics:")
    tracker.log_statistics()


def example_frame_buffer():
    """Example: Frame buffer with rotation."""
    print("\n" + "="*60)
    print("Example 3: Frame Buffer with Rotation")
    print("="*60)
    
    # Create frame buffer with small limit for demo
    buffer = FrameBuffer(
        max_frames=5,  # Keep only 5 frames
        output_dir='examples/output/demo_frames',
        max_memory_mb=10,
        jpeg_quality=85
    )
    
    # Create test frames
    print("Saving 10 frames (buffer limit is 5)...")
    for i in range(10):
        # Create a test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"Frame {i+1}", (200, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save with automatic rotation
        saved_path = buffer.save_current(frame, 'demo', i)
        if saved_path:
            print(f"  Saved frame {i+1}: {saved_path}")
    
    # Get statistics
    stats = buffer.get_memory_usage()
    print(f"\nBuffer Statistics:")
    print(f"  Frames in buffer: {stats['buffer_frames']}")
    print(f"  Frames on disk: {stats['disk_frames']}")
    print(f"  Disk size: {stats['disk_size_mb']:.2f} MB")
    print(f"  Total processed: {stats['total_frames_processed']}")
    
    print("\n✓ Old frames were automatically rotated (kept only last 5)")


def example_resource_tracking():
    """Example: Resource tracking and leak detection."""
    print("\n" + "="*60)
    print("Example 4: Resource Tracking and Leak Detection")
    print("="*60)
    
    tracker = get_global_tracker()
    
    # Get initial statistics
    initial_stats = tracker.get_statistics()
    print("Initial state:")
    print(f"  Memory: {initial_stats['memory_mb']:.1f} MB")
    print(f"  Tracked cameras: {initial_stats['tracked_cameras']}")
    print(f"  Tracked windows: {initial_stats['tracked_windows']}")
    
    # Simulate some resource usage
    print("\nCreating resources...")
    
    # Create some dummy data to increase memory
    data = [np.zeros((1000, 1000)) for _ in range(10)]
    
    # Get updated statistics
    current_stats = tracker.get_statistics()
    print("\nAfter resource allocation:")
    print(f"  Memory: {current_stats['memory_mb']:.1f} MB")
    print(f"  Memory delta: {current_stats['memory_delta_mb']:+.1f} MB")
    
    # Clean up
    del data
    
    print("\n✓ Resource tracking helps detect memory usage patterns")


def example_orphaned_cleanup():
    """Example: Cleanup orphaned files."""
    print("\n" + "="*60)
    print("Example 5: Orphaned File Cleanup")
    print("="*60)
    
    # Create some temp files
    print("Creating temporary files...")
    with TempFileManager(prefix="orphan_test_") as tmp:
        for i in range(3):
            path = tmp.create_temp_file('.txt')
            with open(path, 'w') as f:
                f.write(f"Test file {i}")
            print(f"  Created: {path}")
    
    print("\nFiles cleaned up by context manager")
    
    # Demonstrate orphaned file cleanup
    print("\nOrphaned file cleanup:")
    print("  (This cleans up files older than specified age)")
    
    cleaned_count = TempFileManager.cleanup_orphaned_files(
        prefix="orphan_test_",
        max_age_hours=0  # Clean up immediately for demo
    )
    print(f"  Cleaned up {cleaned_count} orphaned files")


def example_complete_workflow():
    """Example: Complete workflow with all resource management."""
    print("\n" + "="*60)
    print("Example 6: Complete Workflow")
    print("="*60)
    
    tracker = get_global_tracker()
    logger = get_logger(__name__)
    
    # Setup frame buffer
    buffer = FrameBuffer(
        max_frames=10,
        output_dir='examples/output/workflow_frames',
        jpeg_quality=85
    )
    
    print("Starting complete workflow...")
    
    try:
        # Use camera with proper resource management
        with managed_camera(0) as camera:
            logger.info("Camera opened")
            
            # Process frames with temp files
            for cycle in range(3):
                ret, frame = camera.read()
                if not ret:
                    break
                
                # Use temp file for processing
                with TempFileManager(prefix=f"process_{cycle}_") as tmp:
                    temp_path = tmp.create_temp_file('.jpg')
                    cv2.imwrite(temp_path, frame)
                    
                    # Process...
                    processed = cv2.imread(temp_path)
                    
                    # Save with rotation
                    buffer.save_current(processed, 'camera_a', cycle)
                
                logger.info(f"Processed cycle {cycle}")
                time.sleep(0.5)
            
            logger.info("Camera processing complete")
        
        # Show final statistics
        print("\nFinal Statistics:")
        tracker.log_statistics()
        
        buffer_stats = buffer.get_memory_usage()
        print(f"Frames saved: {buffer_stats['disk_frames']}")
        
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        print(f"✗ Error (might not have camera): {e}")
    finally:
        # Cleanup
        tracker.release_all()
        buffer.clear()
        print("\n✓ All resources cleaned up")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Resource Management Examples")
    print("Smart Traffic Light System")
    print("="*60)
    
    # Run examples
    example_temp_files()
    example_camera_management()
    example_frame_buffer()
    example_resource_tracking()
    example_orphaned_cleanup()
    example_complete_workflow()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
    print("\nKey Takeaways:")
    print("1. Always use context managers for resources")
    print("2. Track resources with ResourceTracker")
    print("3. Use FrameBuffer for automatic rotation")
    print("4. Monitor memory usage periodically")
    print("5. Clean up orphaned files regularly")


if __name__ == "__main__":
    main()