import cv2
import numpy as np

class CarIdentifier:
    def __init__(self):
        # Use background subtractor for simple car detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

    def count_cars(self, frame):
        """
        Counts cars in a given frame using background subtraction and contour detection.
        Args:
            frame: np.ndarray, image from camera
        Returns:
            int: estimated number of cars
        """
        fg_mask = self.bg_subtractor.apply(frame)
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        car_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Threshold area to filter out noise
                car_count += 1
        return car_count
