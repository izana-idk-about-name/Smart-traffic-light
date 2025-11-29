#!/usr/bin/env python3
"""
Simple and robust camera visualization module for real-time detection display.
This module handles webcam feed display with car detection overlays.
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable, Tuple
import os


class CameraViewer:
    """Simple camera viewer with detection visualization"""
    
    def __init__(self, camera_name: str = "Camera"):
        """
        Initialize camera viewer.
        
        Args:
            camera_name: Name to display in window title
        """
        self.camera_name = camera_name
        self.window_name = f"Smart Traffic - {camera_name}"
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()
        self.current_frame: Optional[np.ndarray] = None
        self.detection_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
        
    def start(self, camera_source, detector, fps: int = 10):
        """
        Start the visualization in a separate thread.
        
        Args:
            camera_source: Camera source object with read() method
            detector: Detection object with count_cars() and visualize_detection() methods
            fps: Target frames per second
        """
        if self.running:
            print(f"[{self.camera_name}] Viewer already running")
            return
        
        self.running = True
        self.thread = threading.Thread(
            target=self._visualization_loop,
            args=(camera_source, detector, fps),
            daemon=True,
            name=f"Viewer-{self.camera_name}"
        )
        self.thread.start()
        print(f"[{self.camera_name}] Visualization started")
    
    def _visualization_loop(self, camera_source, detector, fps: int):
        """Main visualization loop"""
        frame_delay = 1.0 / fps
        
        try:
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 800, 600)
            print(f"[{self.camera_name}] Window created: {self.window_name}")
            
            while self.running:
                loop_start = time.time()
                
                # Read frame from camera
                ret, frame = camera_source.read()
                
                if not ret or frame is None:
                    print(f"[{self.camera_name}] Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Run detection and get visualization
                try:
                    car_count, vis_frame = detector.visualize_detection(
                        frame, 
                        show_contours=True
                    )
                    self.detection_count = car_count
                except Exception as e:
                    print(f"[{self.camera_name}] Detection error: {e}")
                    vis_frame = frame.copy()
                    self.detection_count = 0
                
                # Calculate FPS
                self.frame_count += 1
                if time.time() - self.last_fps_time >= 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = time.time()
                
                # Add overlay information
                vis_frame = self._add_overlay(vis_frame)
                
                # Display frame
                try:
                    cv2.imshow(self.window_name, vis_frame)
                except Exception as e:
                    print(f"[{self.camera_name}] Display error: {e}")
                    break
                
                # Check for window close or ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    print(f"[{self.camera_name}] ESC pressed, stopping viewer")
                    self.running = False
                    break
                
                # Check if window was closed
                try:
                    if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print(f"[{self.camera_name}] Window closed by user")
                        self.running = False
                        break
                except Exception:
                    pass
                
                # Frame rate control
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except Exception as e:
            print(f"[{self.camera_name}] Visualization loop error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            print(f"[{self.camera_name}] Visualization stopped")
    
    def _add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add information overlay to frame"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for header
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # Camera name and title
        cv2.putText(
            frame, 
            f"{self.camera_name} - Smart Traffic Detection",
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (255, 255, 255), 
            2
        )
        
        # Detection count
        count_text = f"Vehicles: {self.detection_count}"
        cv2.putText(
            frame, 
            count_text,
            (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0) if self.detection_count > 0 else (255, 255, 255), 
            2
        )
        
        # FPS counter
        fps_text = f"FPS: {self.fps}"
        cv2.putText(
            frame, 
            fps_text,
            (w - 120, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (0, 255, 255), 
            2
        )
        
        # Instructions
        cv2.putText(
            frame, 
            "Press ESC to close",
            (10, h - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
        
        return frame
    
    def stop(self):
        """Stop the visualization"""
        if not self.running:
            return
        
        self.running = False
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
    
    def is_running(self) -> bool:
        """Check if viewer is running"""
        return self.running and (self.thread is not None and self.thread.is_alive())


class MultiCameraViewer:
    """Manager for multiple camera viewers"""
    
    def __init__(self):
        self.viewers = {}
    
    def add_camera(self, camera_id: str, camera_source, detector, fps: int = 10):
        """
        Add and start a camera viewer.
        
        Args:
            camera_id: Unique identifier for the camera
            camera_source: Camera source object
            detector: Detection object
            fps: Target frames per second
        """
        if camera_id in self.viewers:
            print(f"Camera {camera_id} already added")
            return
        
        viewer = CameraViewer(camera_name=f"Camera {camera_id}")
        viewer.start(camera_source, detector, fps)
        self.viewers[camera_id] = viewer
        print(f"Added camera viewer: {camera_id}")
    
    def remove_camera(self, camera_id: str):
        """Remove and stop a camera viewer"""
        if camera_id in self.viewers:
            self.viewers[camera_id].stop()
            del self.viewers[camera_id]
            print(f"Removed camera viewer: {camera_id}")
    
    def stop_all(self):
        """Stop all camera viewers"""
        for camera_id in list(self.viewers.keys()):
            self.remove_camera(camera_id)
        cv2.destroyAllWindows()
    
    def is_any_running(self) -> bool:
        """Check if any viewer is still running"""
        return any(viewer.is_running() for viewer in self.viewers.values())
    
    def wait_until_closed(self):
        """Wait until all viewers are closed"""
        print("Viewers active. Press ESC in any window to close all.")
        try:
            while self.is_any_running():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        finally:
            self.stop_all()