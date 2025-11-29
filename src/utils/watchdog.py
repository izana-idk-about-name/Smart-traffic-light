"""
Watchdog system for Smart Traffic Light system monitoring and recovery.

Provides automated monitoring and recovery capabilities including:
- Periodic health check execution
- Automatic recovery attempts for failed components
- Configurable recovery strategies
- Graceful shutdown on critical failures
"""

import time
import threading
from typing import Dict, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
from src.utils.logger import get_logger
from src.utils.healthcheck import HealthCheck, HealthCheckResult


class RecoveryAction(Enum):
    """Types of recovery actions"""
    NONE = "none"
    RESTART_COMPONENT = "restart"
    RELOAD_MODEL = "reload"
    REINIT_CAMERA = "reinit_camera"
    FORCE_GC = "force_gc"
    CLEAN_TEMP_FILES = "clean_temp"
    REQUEST_SHUTDOWN = "shutdown"


@dataclass
class RecoveryStrategy:
    """Strategy for recovering from component failure"""
    component: str
    max_attempts: int
    actions: list[RecoveryAction]
    cooldown_seconds: float = 60.0


class Watchdog:
    """
    Monitor system health and perform automatic recovery.
    
    Features:
    - Periodic health check monitoring
    - Configurable recovery strategies per component
    - Tracks recovery attempts and success rates
    - Automatic shutdown on critical failures
    - Thread-safe operation
    """
    
    def __init__(self, health_check: HealthCheck, check_interval: int = 60,
                 shutdown_callback: Optional[Callable[[], None]] = None):
        """
        Initialize watchdog system.
        
        Args:
            health_check: HealthCheck instance to monitor
            check_interval: Seconds between health checks
            shutdown_callback: Function to call for graceful shutdown
        """
        self.logger = get_logger(__name__)
        self.health_check = health_check
        self.check_interval = check_interval
        self.shutdown_callback = shutdown_callback
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Recovery tracking
        self.recovery_attempts: Dict[str, int] = {}
        self.last_recovery_time: Dict[str, float] = {}
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.recovery_callbacks: Dict[str, Callable[[], bool]] = {}
        
        # Statistics
        self.total_recoveries_attempted = 0
        self.total_recoveries_successful = 0
        self.checks_performed = 0
        self.failures_detected = 0
        
        self.logger.info(f"Watchdog initialized with {check_interval}s check interval")
    
    def register_recovery_strategy(self, strategy: RecoveryStrategy,
                                   recovery_callback: Optional[Callable[[], bool]] = None):
        """
        Register a recovery strategy for a component.
        
        Args:
            strategy: RecoveryStrategy defining how to recover
            recovery_callback: Optional function to call for recovery
        """
        with self._lock:
            self.recovery_strategies[strategy.component] = strategy
            if recovery_callback:
                self.recovery_callbacks[strategy.component] = recovery_callback
            self.recovery_attempts[strategy.component] = 0
            
            self.logger.info(f"Registered recovery strategy for '{strategy.component}': "
                           f"{len(strategy.actions)} actions, max {strategy.max_attempts} attempts")
    
    def start(self):
        """Start watchdog monitoring"""
        with self._lock:
            if self._running:
                self.logger.warning("Watchdog already running")
                return
            
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="WatchdogMonitor",
                daemon=True
            )
            self._thread.start()
            self.logger.info("Watchdog monitoring started")
    
    def stop(self, timeout: float = 5.0):
        """
        Stop watchdog monitoring.
        
        Args:
            timeout: Maximum seconds to wait for thread to stop
        """
        with self._lock:
            if not self._running:
                return
            
            self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self.logger.warning("Watchdog thread did not stop within timeout")
            else:
                self.logger.info("Watchdog monitoring stopped")
        
        self._log_statistics()
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        self.logger.info("Watchdog monitor loop started")
        
        while self._running:
            try:
                # Run health checks
                results = self.health_check.run_all_checks()
                self.checks_performed += 1
                
                # Check for failures
                failed_checks = [name for name, result in results.items() 
                                if not result.healthy]
                
                if failed_checks:
                    self.failures_detected += len(failed_checks)
                    self.logger.warning(f"Health check failures detected: {failed_checks}")
                    
                    # Attempt recovery for each failed component
                    for component in failed_checks:
                        self._handle_failure(component, results[component])
                
                # Log status periodically
                if self.checks_performed % 10 == 0:
                    self._log_status()
                
            except Exception as e:
                self.logger.error(f"Error in watchdog monitor loop: {e}", exc_info=True)
            
            # Sleep in small increments to allow quick shutdown
            slept = 0.0
            while slept < self.check_interval and self._running:
                time.sleep(0.5)
                slept += 0.5
        
        self.logger.info("Watchdog monitor loop ended")
    
    def _handle_failure(self, component: str, result: HealthCheckResult):
        """
        Handle component failure with recovery attempts.
        
        Args:
            component: Name of failed component
            result: Health check result with failure details
        """
        with self._lock:
            # Check if we have a recovery strategy
            if component not in self.recovery_strategies:
                self.logger.warning(f"No recovery strategy for component: {component}")
                return
            
            strategy = self.recovery_strategies[component]
            
            # Check cooldown period
            last_recovery = self.last_recovery_time.get(component, 0)
            time_since_last = time.time() - last_recovery
            
            if time_since_last < strategy.cooldown_seconds:
                self.logger.debug(f"Recovery cooldown active for {component} "
                                f"({time_since_last:.1f}s / {strategy.cooldown_seconds}s)")
                return
            
            # Check if max attempts exceeded
            attempts = self.recovery_attempts.get(component, 0)
            if attempts >= strategy.max_attempts:
                self.logger.error(f"Max recovery attempts ({strategy.max_attempts}) exceeded "
                                f"for component: {component}")
                
                # Request shutdown on critical failure
                if result.metadata.get('critical', True):
                    self.logger.critical(f"Critical component '{component}' failed - "
                                       f"requesting shutdown")
                    self._request_shutdown(f"Critical component failure: {component}")
                return
            
            # Attempt recovery
            self.logger.info(f"Attempting recovery for '{component}' "
                           f"(attempt {attempts + 1}/{strategy.max_attempts})")
            
            success = self._attempt_recovery(component, strategy)
            
            # Update tracking
            self.recovery_attempts[component] = attempts + 1
            self.last_recovery_time[component] = time.time()
            self.total_recoveries_attempted += 1
            
            if success:
                self.total_recoveries_successful += 1
                self.recovery_attempts[component] = 0  # Reset on success
                self.logger.info(f"Recovery successful for component: {component}")
            else:
                self.logger.warning(f"Recovery failed for component: {component}")
    
    def _attempt_recovery(self, component: str, strategy: RecoveryStrategy) -> bool:
        """
        Attempt to recover a failed component.
        
        Args:
            component: Name of component to recover
            strategy: Recovery strategy to use
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            # Try custom recovery callback first
            if component in self.recovery_callbacks:
                callback = self.recovery_callbacks[component]
                self.logger.debug(f"Executing custom recovery callback for {component}")
                success = callback()
                if success:
                    return True
            
            # Execute recovery actions
            for action in strategy.actions:
                self.logger.debug(f"Executing recovery action: {action.value}")
                
                if action == RecoveryAction.FORCE_GC:
                    self._action_force_gc()
                elif action == RecoveryAction.CLEAN_TEMP_FILES:
                    self._action_clean_temp_files()
                elif action == RecoveryAction.REQUEST_SHUTDOWN:
                    self._request_shutdown(f"Recovery action for {component}")
                    return False
                # Other actions would need specific implementations
            
            # Verify recovery by running check again
            time.sleep(2.0)  # Give component time to recover
            result = self.health_check.run_check(component)
            return result.healthy
            
        except Exception as e:
            self.logger.error(f"Exception during recovery of {component}: {e}", exc_info=True)
            return False
    
    def _action_force_gc(self):
        """Force garbage collection"""
        import gc
        before = gc.get_count()
        gc.collect()
        after = gc.get_count()
        self.logger.info(f"Forced garbage collection: {before} -> {after}")
    
    def _action_clean_temp_files(self):
        """Clean temporary files"""
        try:
            from src.utils.resource_manager import TempFileManager
            TempFileManager.cleanup_orphaned_files(max_age_hours=1)
            self.logger.info("Cleaned temporary files")
        except Exception as e:
            self.logger.warning(f"Failed to clean temp files: {e}")
    
    def _request_shutdown(self, reason: str):
        """Request graceful shutdown"""
        self.logger.critical(f"Requesting system shutdown: {reason}")
        
        if self.shutdown_callback:
            try:
                self.shutdown_callback()
            except Exception as e:
                self.logger.error(f"Error calling shutdown callback: {e}", exc_info=True)
        else:
            self.logger.warning("No shutdown callback registered")
    
    def _log_status(self):
        """Log current watchdog status"""
        with self._lock:
            active_recoveries = sum(1 for attempts in self.recovery_attempts.values() 
                                   if attempts > 0)
            
            self.logger.info(f"Watchdog Status: {self.checks_performed} checks, "
                           f"{self.failures_detected} failures, "
                           f"{self.total_recoveries_attempted} recovery attempts "
                           f"({self.total_recoveries_successful} successful), "
                           f"{active_recoveries} components in recovery")
    
    def _log_statistics(self):
        """Log final statistics"""
        with self._lock:
            success_rate = (self.total_recoveries_successful / self.total_recoveries_attempted * 100
                          if self.total_recoveries_attempted > 0 else 0)
            
            self.logger.info(f"Watchdog Statistics:")
            self.logger.info(f"  Total checks performed: {self.checks_performed}")
            self.logger.info(f"  Total failures detected: {self.failures_detected}")
            self.logger.info(f"  Recovery attempts: {self.total_recoveries_attempted}")
            self.logger.info(f"  Successful recoveries: {self.total_recoveries_successful} "
                           f"({success_rate:.1f}%)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get watchdog statistics.
        
        Returns:
            Dictionary with watchdog statistics
        """
        with self._lock:
            success_rate = (self.total_recoveries_successful / self.total_recoveries_attempted
                          if self.total_recoveries_attempted > 0 else 0)
            
            return {
                'running': self._running,
                'checks_performed': self.checks_performed,
                'failures_detected': self.failures_detected,
                'recovery_attempts': self.total_recoveries_attempted,
                'successful_recoveries': self.total_recoveries_successful,
                'success_rate': success_rate,
                'active_recoveries': {
                    comp: attempts for comp, attempts in self.recovery_attempts.items()
                    if attempts > 0
                },
                'registered_strategies': list(self.recovery_strategies.keys())
            }
    
    def reset_recovery_attempts(self, component: Optional[str] = None):
        """
        Reset recovery attempt counters.
        
        Args:
            component: Specific component to reset, or None for all
        """
        with self._lock:
            if component:
                if component in self.recovery_attempts:
                    self.recovery_attempts[component] = 0
                    self.logger.info(f"Reset recovery attempts for: {component}")
            else:
                for comp in self.recovery_attempts:
                    self.recovery_attempts[comp] = 0
                self.logger.info("Reset all recovery attempts")