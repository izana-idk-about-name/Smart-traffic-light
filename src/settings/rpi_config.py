"""
Raspberry Pi specific configuration for traffic light AI system
Optimized for Raspberry Pi 4 with 4GB RAM
"""

import os
import platform

class RPiConfig:
    """Configuration class for Raspberry Pi optimizations"""
    
    def __init__(self):
        self.is_rpi = self._detect_raspberry_pi()
        self.cpu_count = os.cpu_count()
        self.memory_gb = self._get_memory_gb()
        
    def _detect_raspberry_pi(self) -> bool:
        """Detect if running on Raspberry Pi"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'BCM' in cpuinfo or 'Raspberry Pi' in cpuinfo
        except:
            return False
    
    def _get_memory_gb(self) -> float:
        """Get available memory in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except:
            return 4.0  # Default assumption
    
    def get_camera_settings(self):
        """Get optimized camera settings for RPi"""
        if self.is_rpi:
            return {
                'width': 320,
                'height': 240,
                'fps': 10,
                'buffer_size': 1,
                'format': 'MJPG'
            }
        else:
            return {
                'width': 640,
                'height': 480,
                'fps': 15,
                'buffer_size': 2,
                'format': 'MJPG'
            }
    
    def get_processing_settings(self):
        """Get optimized processing settings"""
        if self.is_rpi:
            return {
                'decision_interval': 3,  # seconds
                'max_processing_time': 1.0,  # seconds
                'use_threading': True,
                'thread_count': min(2, self.cpu_count),
                'memory_limit_mb': 512
            }
        else:
            return {
                'decision_interval': 2,
                'max_processing_time': 0.5,
                'use_threading': True,
                'thread_count': self.cpu_count,
                'memory_limit_mb': 1024
            }
    
    def get_model_settings(self):
        """Get optimized model settings"""
        if self.is_rpi:
            return {
                'optimize_for_rpi': True,
                'use_gpu': False,
                'background_history': 50,
                'var_threshold': 30,
                'min_car_area': 200
            }
        else:
            return {
                'optimize_for_rpi': False,
                'use_gpu': True,
                'background_history': 100,
                'var_threshold': 40,
                'min_car_area': 500
            }
    
    def get_network_settings(self):
        """Get optimized network settings"""
        return {
            'timeout': 5,
            'retry_count': 3,
            'buffer_size': 1024,
            'use_websocket': self.is_rpi  # Use WebSocket for RPi efficiency
        }

# Global configuration instance
rpi_config = RPiConfig()

# Export settings for easy access
CAMERA_SETTINGS = rpi_config.get_camera_settings()
PROCESSING_SETTINGS = rpi_config.get_processing_settings()
MODEL_SETTINGS = rpi_config.get_model_settings()
NETWORK_SETTINGS = rpi_config.get_network_settings()

# Environment detection
IS_RASPBERRY_PI = rpi_config.is_rpi
CPU_COUNT = rpi_config.cpu_count
MEMORY_GB = rpi_config.memory_gb

if __name__ == "__main__":
    print("Raspberry Pi Configuration:")
    print(f"  Running on Raspberry Pi: {IS_RASPBERRY_PI}")
    print(f"  CPU Cores: {CPU_COUNT}")
    print(f"  Memory: {MEMORY_GB:.1f} GB")
    print(f"  Camera Settings: {CAMERA_SETTINGS}")
    print(f"  Processing Settings: {PROCESSING_SETTINGS}")
    print(f"  Model Settings: {MODEL_SETTINGS}")
    print(f"  Network Settings: {NETWORK_SETTINGS}")