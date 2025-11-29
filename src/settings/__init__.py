"""
Settings package for Smart Traffic Light system.
Provides unified configuration management.
"""

from src.settings.settings import (
    Settings,
    SystemSettings,
    CameraSettings,
    DetectionSettings,
    PerformanceSettings,
    LoggingSettings,
    TrafficControlSettings,
    NetworkSettings,
    get_settings
)

# For backward compatibility, also export from legacy modules
try:
    from src.settings.config import get_env_mode, ENVIRONMENT
except ImportError:
    pass

try:
    from src.settings.rpi_config import (
        IS_RASPBERRY_PI,
        CPU_COUNT,
        MEMORY_GB,
        CAMERA_SETTINGS,
        PROCESSING_SETTINGS,
        MODEL_SETTINGS,
        NETWORK_SETTINGS
    )
except ImportError:
    pass

__all__ = [
    # New unified settings system
    'Settings',
    'SystemSettings',
    'CameraSettings',
    'DetectionSettings',
    'PerformanceSettings',
    'LoggingSettings',
    'TrafficControlSettings',
    'NetworkSettings',
    'get_settings',
    # Legacy exports (for backward compatibility)
    'get_env_mode',
    'ENVIRONMENT',
    'IS_RASPBERRY_PI',
    'CPU_COUNT',
    'MEMORY_GB',
    'CAMERA_SETTINGS',
    'PROCESSING_SETTINGS',
    'MODEL_SETTINGS',
    'NETWORK_SETTINGS',
]