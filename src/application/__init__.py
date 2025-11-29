"""
Application layer module for Smart Traffic Light System.

This module provides high-level application components including:
- Type-safe camera source abstractions
- Traffic controller
- Communication components

Key exports:
- CameraSource: Abstract base class for all camera sources
- LiveCameraSource: Live camera via cv2.VideoCapture
- StaticImageSource: Static images for testing
- VideoFileSource: Video files with looping
- CameraFactory: Factory for creating camera sources
- create_camera: Convenience function for camera creation
"""

from src.application.camera_source import (
    CameraSource,
    LiveCameraSource,
    StaticImageSource,
    VideoFileSource,
    CameraFactory,
    create_camera
)

__all__ = [
    'CameraSource',
    'LiveCameraSource',
    'StaticImageSource',
    'VideoFileSource',
    'CameraFactory',
    'create_camera'
]