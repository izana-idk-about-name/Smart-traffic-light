"""
Health check system for Smart Traffic Light monitoring.

Provides comprehensive health monitoring capabilities including:
- Component health checks (cameras, detection models, memory, disk, threads)
- Configurable check intervals and failure thresholds
- Detailed health status reporting
- Optional HTTP server for external monitoring
"""

import time
import threading
import psutil
import os
from typing import Dict, Callable, Optional, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from src.utils.logger import get_logger


@dataclass
class HealthCheckResult:
    """Result of a single health check"""
    name: str
    healthy: bool
    message: str
    timestamp: float = field(default_factory=time.time)
    check_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthCheck:
    """
    System health monitoring with configurable checks.
    
    Features:
    - Register custom health check functions
    - Track check failures and success rates
    - Configurable failure thresholds
    - Detailed status reporting
    - Thread-safe operation
    """
    
    def __init__(self, max_failures: int = 3):
        """
        Initialize health check system.
        
        Args:
            max_failures: Maximum consecutive failures before marking unhealthy
        """
        self.logger = get_logger(__name__)
        self.checks: Dict[str, Callable[[], bool]] = {}
        self.check_metadata: Dict[str, Dict] = {}
        self.last_check_time: Dict[str, float] = {}
        self.failures: Dict[str, int] = {}
        self.max_failures = max_failures
        self.lock = threading.RLock()
        self.check_history: Dict[str, List[HealthCheckResult]] = {}
        self.max_history = 100  # Keep last 100 results per check
        
        self.logger.info(f"HealthCheck initialized with max_failures={max_failures}")
    
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      description: str = "", critical: bool = True):
        """
        Register a health check function.
        
        Args:
            name: Unique name for the check
            check_func: Function that returns True if healthy, False otherwise
            description: Human-readable description of the check
            critical: Whether failure should be considered critical
        """
        with self.lock:
            self.checks[name] = check_func
            self.check_metadata[name] = {
                'description': description,
                'critical': critical,
                'registered_at': time.time()
            }
            self.failures[name] = 0
            self.check_history[name] = []
            self.logger.info(f"Registered health check: {name} (critical={critical})")
    
    def unregister_check(self, name: str):
        """Remove a health check"""
        with self.lock:
            if name in self.checks:
                del self.checks[name]
                del self.check_metadata[name]
                del self.failures[name]
                if name in self.check_history:
                    del self.check_history[name]
                self.logger.info(f"Unregistered health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.
        
        Args:
            name: Name of the check to run
            
        Returns:
            HealthCheckResult with check outcome
        """
        if name not in self.checks:
            return HealthCheckResult(
                name=name,
                healthy=False,
                message=f"Check '{name}' not registered",
                check_duration=0.0
            )
        
        start_time = time.time()
        try:
            check_func = self.checks[name]
            is_healthy = check_func()
            check_duration = time.time() - start_time
            
            with self.lock:
                self.last_check_time[name] = time.time()
                
                if is_healthy:
                    self.failures[name] = 0
                    message = "Check passed"
                else:
                    self.failures[name] += 1
                    message = f"Check failed ({self.failures[name]}/{self.max_failures} failures)"
                
                result = HealthCheckResult(
                    name=name,
                    healthy=is_healthy,
                    message=message,
                    check_duration=check_duration,
                    metadata={
                        'failures': self.failures[name],
                        'max_failures': self.max_failures,
                        'critical': self.check_metadata[name].get('critical', True)
                    }
                )
                
                # Store in history
                self._add_to_history(name, result)
                
                if not is_healthy:
                    self.logger.warning(f"Health check '{name}' failed: {message}")
                else:
                    self.logger.debug(f"Health check '{name}' passed in {check_duration:.3f}s")
                
                return result
                
        except Exception as e:
            check_duration = time.time() - start_time
            self.logger.error(f"Health check '{name}' raised exception: {e}", exc_info=True)
            
            with self.lock:
                self.failures[name] += 1
                result = HealthCheckResult(
                    name=name,
                    healthy=False,
                    message=f"Exception: {str(e)}",
                    check_duration=check_duration,
                    metadata={'failures': self.failures[name], 'exception': str(e)}
                )
                self._add_to_history(name, result)
                return result
    
    def _add_to_history(self, name: str, result: HealthCheckResult):
        """Add result to check history (assumes lock is held)"""
        if name not in self.check_history:
            self.check_history[name] = []
        
        history = self.check_history[name]
        history.append(result)
        
        # Trim history if too long
        if len(history) > self.max_history:
            self.check_history[name] = history[-self.max_history:]
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary mapping check names to results
        """
        results = {}
        check_names = list(self.checks.keys())
        
        for name in check_names:
            results[name] = self.run_check(name)
        
        return results
    
    def is_healthy(self, include_non_critical: bool = True) -> bool:
        """
        Check if system is healthy based on all checks.
        
        Args:
            include_non_critical: Whether to include non-critical checks
            
        Returns:
            True if all checks pass or failures are below threshold
        """
        with self.lock:
            for name, failure_count in self.failures.items():
                is_critical = self.check_metadata.get(name, {}).get('critical', True)
                
                if not is_critical and not include_non_critical:
                    continue
                
                if failure_count >= self.max_failures:
                    return False
            
            return True
    
    def get_status_report(self, include_history: bool = False) -> Dict[str, Any]:
        """
        Get detailed health status report.
        
        Args:
            include_history: Whether to include check history
            
        Returns:
            Dictionary with comprehensive health status
        """
        with self.lock:
            overall_healthy = self.is_healthy()
            
            checks_status = {}
            for name in self.checks.keys():
                check_info = {
                    'description': self.check_metadata.get(name, {}).get('description', ''),
                    'critical': self.check_metadata.get(name, {}).get('critical', True),
                    'failures': self.failures.get(name, 0),
                    'max_failures': self.max_failures,
                    'last_check': self.last_check_time.get(name, 0),
                    'healthy': self.failures.get(name, 0) < self.max_failures
                }
                
                if include_history and name in self.check_history:
                    history = self.check_history[name]
                    if history:
                        last_result = history[-1]
                        check_info['last_result'] = {
                            'healthy': last_result.healthy,
                            'message': last_result.message,
                            'timestamp': last_result.timestamp,
                            'duration': last_result.check_duration
                        }
                        
                        # Calculate success rate from history
                        successes = sum(1 for r in history if r.healthy)
                        check_info['success_rate'] = successes / len(history) if history else 0.0
                
                checks_status[name] = check_info
            
            report = {
                'overall_healthy': overall_healthy,
                'timestamp': time.time(),
                'total_checks': len(self.checks),
                'failing_checks': sum(1 for c in checks_status.values() if not c['healthy']),
                'checks': checks_status
            }
            
            return report
    
    def reset_failures(self, check_name: Optional[str] = None):
        """
        Reset failure counters.
        
        Args:
            check_name: Specific check to reset, or None for all
        """
        with self.lock:
            if check_name:
                if check_name in self.failures:
                    self.failures[check_name] = 0
                    self.logger.info(f"Reset failures for check: {check_name}")
            else:
                for name in self.failures:
                    self.failures[name] = 0
                self.logger.info("Reset all failure counters")


# Built-in health check functions
class BuiltInHealthChecks:
    """Collection of common health check functions"""
    
    @staticmethod
    def create_camera_health_check(camera, camera_name: str) -> Callable[[], bool]:
        """Create a health check for camera"""
        def check():
            if camera is None:
                return False
            if not hasattr(camera, 'is_opened'):
                return False
            if not camera.is_opened():
                return False
            # Try to read a frame
            ret, frame = camera.read()
            return ret and frame is not None
        return check
    
    @staticmethod
    def create_memory_health_check(max_memory_percent: float = 90.0) -> Callable[[], bool]:
        """Create a health check for memory usage"""
        def check():
            memory = psutil.virtual_memory()
            return memory.percent < max_memory_percent
        return check
    
    @staticmethod
    def create_disk_health_check(min_free_gb: float = 1.0, path: str = '/') -> Callable[[], bool]:
        """Create a health check for disk space"""
        def check():
            try:
                disk = psutil.disk_usage(path)
                free_gb = disk.free / (1024**3)
                return free_gb >= min_free_gb
            except Exception:
                return False
        return check
    
    @staticmethod
    def create_thread_health_check(threads: List[threading.Thread]) -> Callable[[], bool]:
        """Create a health check for thread status"""
        def check():
            return all(t.is_alive() for t in threads if t is not None)
        return check
    
    @staticmethod
    def create_detection_health_check(car_identifier) -> Callable[[], bool]:
        """Create a health check for detection model"""
        def check():
            if car_identifier is None:
                return False
            # Check if model is loaded
            has_model = (
                getattr(car_identifier, 'tflite_loaded', False) or
                getattr(car_identifier, 'custom_model_loaded', False) or
                getattr(car_identifier, 'model_loaded', False)
            )
            return has_model
        return check
    
    @staticmethod
    def create_processing_time_health_check(car_identifier, max_time: float = 1.0) -> Callable[[], bool]:
        """Create a health check for processing time"""
        def check():
            if car_identifier is None or not hasattr(car_identifier, 'get_average_processing_time'):
                return True  # Don't fail if not available
            avg_time = car_identifier.get_average_processing_time()
            return avg_time == 0 or avg_time < max_time
        return check