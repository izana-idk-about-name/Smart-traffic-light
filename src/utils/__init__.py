"""
Utilities package for Smart Traffic Light System
Provides logging, resource management, health monitoring, and watchdog functionality
"""

from src.utils.logger import (
    setup_logger,
    get_logger,
    log_execution_time,
    LogContext
)

from src.utils.resource_manager import (
    TempFileManager,
    ResourceTracker,
    FrameBuffer,
    CameraContextManager,
    get_global_tracker,
    managed_camera
)

from src.utils.healthcheck import (
    HealthCheck,
    HealthCheckResult,
    BuiltInHealthChecks
)

from src.utils.watchdog import (
    Watchdog,
    RecoveryStrategy,
    RecoveryAction
)

__all__ = [
    # Logging utilities
    'setup_logger',
    'get_logger',
    'log_execution_time',
    'LogContext',
    
    # Resource management utilities
    'TempFileManager',
    'ResourceTracker',
    'FrameBuffer',
    'CameraContextManager',
    'get_global_tracker',
    'managed_camera',
    
    # Health monitoring utilities
    'HealthCheck',
    'HealthCheckResult',
    'BuiltInHealthChecks',
    
    # Watchdog utilities
    'Watchdog',
    'RecoveryStrategy',
    'RecoveryAction'
]