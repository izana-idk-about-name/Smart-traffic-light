import cv2
import numpy as np
import time

class CarIdentifier:
    def __init__(self, use_gpu=False, optimize_for_rpi=True):
        """
        Initialize car identifier with optimization options
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
            optimize_for_rpi: Whether to optimize for Raspberry Pi performance
        """
        self.use_gpu = use_gpu
        self.optimize_for_rpi = optimize_for_rpi
        
        # Use background subtractor for simple car detection
        # Optimized parameters for Raspberry Pi
        if optimize_for_rpi:
            # Lower history for faster processing on RPi
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=50, 
                varThreshold=30,
                detectShadows=False  # Disable shadows for better performance
            )
        else:
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=100, 
                varThreshold=40,
                detectShadows=True
            )
        
        # Smaller kernel for morphological operations on RPi
        kernel_size = 3 if optimize_for_rpi else 5
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Performance tracking
        self.frame_count = 0
        self.total_processing_time = 0
        
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
    
    def count_cars(self, frame):
        """
        Counts cars in a given frame using background subtraction and contour detection.
        Optimized for Raspberry Pi performance.
        
        Args:
            frame: np.ndarray, image from camera
        Returns:
            int: estimated number of cars
        """
        start_time = time.time()
        
        # Preprocess frame
        processed_frame = self.preprocess_frame(frame)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(processed_frame)
        
        # Morphological operations to reduce noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        car_count = 0
        min_area = 200 if self.optimize_for_rpi else 500
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                # Additional filtering for better accuracy
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Filter based on aspect ratio (typical car shape)
                if 0.5 < aspect_ratio < 3.0:
                    car_count += 1
        
        # Update performance metrics
        self.frame_count += 1
        processing_time = time.time() - start_time
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
        Create visualization of car detection (for debugging)
        
        Args:
            frame: Input frame
            show_contours: Whether to draw contours
            
        Returns:
            tuple: (count, visualization_frame)
        """
        processed_frame = self.preprocess_frame(frame)
        fg_mask = self.bg_subtractor.apply(processed_frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        car_count = 0
        min_area = 200 if self.optimize_for_rpi else 500
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                if 0.5 < aspect_ratio < 3.0:
                    car_count += 1
                    
                    if show_contours:
                        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(vis_frame, f'Car {car_count}', (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add performance info
        avg_time = self.get_average_processing_time()
        cv2.putText(vis_frame, f'Cars: {car_count}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_frame, f'Avg Time: {avg_time:.3f}s', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return car_count, vis_frame

# Factory function for creating optimized car identifier
def create_car_identifier(mode='rpi'):
    """
    Factory function to create car identifier based on target platform
    
    Args:
        mode: 'rpi' for Raspberry Pi, 'desktop' for desktop, 'debug' for debugging
        
    Returns:
        CarIdentifier instance
    """
    if mode == 'rpi':
        return CarIdentifier(optimize_for_rpi=True, use_gpu=False)
    elif mode == 'desktop':
        return CarIdentifier(optimize_for_rpi=False, use_gpu=True)
    elif mode == 'debug':
        return CarIdentifier(optimize_for_rpi=False, use_gpu=False)
    else:
        return CarIdentifier(optimize_for_rpi=True, use_gpu=False)
