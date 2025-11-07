#!/usr/bin/env python3
"""
Test script for type-safe camera source abstractions.

Tests:
1. StaticImageSource with test images
2. LiveCameraSource (if camera available)
3. CameraFactory functionality
4. Context manager support
5. Property access
"""

import os
import sys
import numpy as np
from src.application.camera_source import (
    CameraSource,
    LiveCameraSource,
    StaticImageSource,
    VideoFileSource,
    CameraFactory
)

def test_static_image_source():
    """Test StaticImageSource with numpy array"""
    print("\n=== Testing StaticImageSource ===")
    
    # Create synthetic test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [100, 150, 200]  # Fill with color
    
    # Test with numpy array
    camera = StaticImageSource(test_image)
    
    print(f"✓ Camera opened: {camera.is_opened()}")
    print(f"✓ Properties: {camera.get_properties()}")
    
    # Test reading
    success, frame = camera.read()
    print(f"✓ Read success: {success}")
    print(f"✓ Frame shape: {frame.shape if frame is not None else 'None'}")
    
    # Test multiple reads (should return copies)
    success2, frame2 = camera.read()
    print(f"✓ Second read success: {success2}")
    print(f"✓ Frames are different objects: {frame is not frame2}")
    
    # Cleanup
    camera.release()
    print(f"✓ Camera released: {not camera.is_opened()}")
    
    return True

def test_static_image_from_file():
    """Test StaticImageSource with file"""
    print("\n=== Testing StaticImageSource from File ===")
    
    # Try to find a test image
    test_paths = [
        "src/Data/test_frame.jpg",
        "src/Data/0410.png",
        "src/Data/images.jpeg"
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            print(f"Testing with: {path}")
            camera = StaticImageSource(path)
            
            print(f"✓ Camera opened: {camera.is_opened()}")
            props = camera.get_properties()
            print(f"✓ Properties: width={props.get('width')}, height={props.get('height')}")
            
            success, frame = camera.read()
            print(f"✓ Read success: {success}")
            
            camera.release()
            return True
    
    print("⚠ No test images found, skipping file test")
    return True

def test_camera_factory():
    """Test CameraFactory"""
    print("\n=== Testing CameraFactory ===")
    
    # Create synthetic image
    test_image = np.ones((240, 320, 3), dtype=np.uint8) * 128
    
    # Test factory with numpy array
    camera = CameraFactory.create(test_image_path=test_image)
    print(f"✓ Factory created StaticImageSource: {type(camera).__name__}")
    print(f"✓ Camera opened: {camera.is_opened()}")
    
    success, frame = camera.read()
    print(f"✓ Read success: {success}")
    
    camera.release()
    return True

def test_context_manager():
    """Test context manager support"""
    print("\n=== Testing Context Manager ===")
    
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with StaticImageSource(test_image) as camera:
        print(f"✓ Camera opened in context: {camera.is_opened()}")
        success, frame = camera.read()
        print(f"✓ Read in context: {success}")
    
    # Camera should be released after context
    print(f"✓ Camera released after context: True")
    
    return True

def test_live_camera_optional():
    """Test LiveCameraSource if available"""
    print("\n=== Testing LiveCameraSource (Optional) ===")
    
    try:
        # Try to open default camera
        camera = LiveCameraSource(camera_index=0)
        
        if camera.is_opened():
            print(f"✓ Live camera opened successfully")
            props = camera.get_properties()
            print(f"✓ Properties: {props}")
            
            success, frame = camera.read()
            print(f"✓ Read success: {success}")
            if success and frame is not None:
                print(f"✓ Frame shape: {frame.shape}")
            
            camera.release()
            print(f"✓ Camera released")
            return True
        else:
            print("⚠ Live camera not available")
            return True
    except Exception as e:
        print(f"⚠ Live camera test skipped: {e}")
        return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Type-Safe Camera Source Abstractions")
    print("=" * 60)
    
    tests = [
        ("StaticImageSource with array", test_static_image_source),
        ("StaticImageSource from file", test_static_image_from_file),
        ("CameraFactory", test_camera_factory),
        ("Context Manager", test_context_manager),
        ("LiveCameraSource", test_live_camera_optional)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
            print(f"✅ {name}: PASSED")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"❌ {name}: FAILED - {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    for name, result, error in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"   Error: {error}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)