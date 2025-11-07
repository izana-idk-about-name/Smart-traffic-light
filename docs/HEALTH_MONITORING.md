# Health Monitoring and Watchdog System

## Overview

The Smart Traffic Light system now includes comprehensive health monitoring and automatic recovery capabilities through the Health Check and Watchdog systems.

## Components

### 1. ShutdownManager (`main.py`)

Thread-safe shutdown coordination system that ensures graceful termination.

**Features:**
- Thread-safe shutdown requests
- Signal tracking (SIGINT, SIGTERM)
- Blocking wait for shutdown
- Atomic shutdown state management

**Usage:**
```python
shutdown_manager = ShutdownManager()

# Request shutdown
shutdown_manager.request_shutdown(signal.SIGTERM)

# Check if shutdown requested
if shutdown_manager.is_shutdown_requested():
    # Cleanup code here
    pass

# Wait for shutdown signal
shutdown_manager.wait_for_shutdown(timeout=5.0)
```

### 2. HealthCheck System (`src/utils/healthcheck.py`)

Monitors system components and tracks their health status.

**Features:**
- Register custom health check functions
- Track failure counts per component
- Configurable failure thresholds
- Historical check results
- Detailed status reporting
- Built-in checks for common components

**Usage:**
```python
from src.utils.healthcheck import HealthCheck, BuiltInHealthChecks

# Create health check system
health_check = HealthCheck(max_failures=3)

# Register built-in checks
health_check.register_check(
    'memory',
    BuiltInHealthChecks.create_memory_health_check(max_memory_percent=90.0),
    description="System memory usage",
    critical=False
)

health_check.register_check(
    'camera_a',
    BuiltInHealthChecks.create_camera_health_check(camera, "A"),
    description="Camera A status",
    critical=True
)

# Run checks
results = health_check.run_all_checks()

# Check overall health
is_healthy = health_check.is_healthy()

# Get detailed report
report = health_check.get_status_report(include_history=True)
```

**Built-in Health Checks:**

1. **Camera Health Check** - Verifies camera is opened and can read frames
2. **Memory Health Check** - Monitors system memory usage
3. **Disk Health Check** - Monitors available disk space
4. **Thread Health Check** - Verifies threads are alive
5. **Detection Health Check** - Verifies ML models are loaded
6. **Processing Time Check** - Monitors average processing time

### 3. Watchdog System (`src/utils/watchdog.py`)

Automated monitoring and recovery system that takes action on component failures.

**Features:**
- Periodic health check execution
- Automatic recovery attempts
- Configurable recovery strategies
- Cooldown periods between recovery attempts
- Success rate tracking
- Graceful shutdown on critical failures

**Usage:**
```python
from src.utils.watchdog import Watchdog, RecoveryStrategy, RecoveryAction

# Create watchdog
watchdog = Watchdog(
    health_check=health_check,
    check_interval=60,  # Check every 60 seconds
    shutdown_callback=lambda: shutdown_manager.request_shutdown()
)

# Define recovery strategy
memory_strategy = RecoveryStrategy(
    component='memory',
    max_attempts=3,
    actions=[RecoveryAction.FORCE_GC],
    cooldown_seconds=60.0
)

# Register strategy with optional custom callback
def recover_memory():
    # Custom recovery logic
    import gc
    gc.collect()
    return True

watchdog.register_recovery_strategy(memory_strategy, recover_memory)

# Start monitoring
watchdog.start()

# Stop monitoring
watchdog.stop(timeout=5.0)

# Get statistics
stats = watchdog.get_statistics()
```

**Recovery Actions:**

- `NONE` - No action
- `RESTART_COMPONENT` - Restart failed component
- `RELOAD_MODEL` - Reload ML models
- `REINIT_CAMERA` - Reinitialize camera
- `FORCE_GC` - Force garbage collection
- `CLEAN_TEMP_FILES` - Clean temporary files
- `REQUEST_SHUTDOWN` - Request graceful shutdown

### 4. Enhanced Cleanup System

The cleanup system now includes verification to ensure all resources are properly released.

**Features:**
- Thread-safe cleanup
- Timeout-based thread joining
- Resource verification
- Comprehensive logging
- Graceful degradation

**Cleanup Verification:**
```python
def verify_cleanup(self) -> bool:
    checks = {
        'cameras_released': not (camera_a and camera_a.is_opened()),
        'windows_closed': not cv2.getWindowProperty('any', cv2.WND_PROP_VISIBLE) >= 0,
        'threads_stopped': not any(t.is_alive() for t in threads),
        'files_closed': True
    }
    return all(checks.values())
```

## Integration in TrafficLightController

The system automatically sets up health monitoring during initialization:

```python
class TrafficLightController:
    def __init__(self, ...):
        # Create shutdown manager
        self.shutdown_manager = ShutdownManager()
        
        # Create health check system
        self.health_check = HealthCheck(max_failures=3)
        
        # Setup health checks
        self._setup_health_checks()
        
        # Start watchdog after camera initialization
        self._start_watchdog()
```

## Health Check Registration

Health checks are automatically registered for:

1. **System Resources:**
   - Memory usage (non-critical)
   - Disk space (non-critical)

2. **Cameras:**
   - Camera A status (critical)
   - Camera B status (critical, if different from A)

3. **Detection Models:**
   - Detection model A (critical)
   - Detection model B (critical)
   - Processing time A (non-critical)

## Monitoring and Logging

The system provides comprehensive logging:

- Health check results (every check)
- Recovery attempts (when failures occur)
- Periodic health status (every 5 minutes)
- Watchdog statistics (on shutdown)
- Cleanup verification (on shutdown)

**Example Log Output:**
```
[INFO] HealthCheck initialized with max_failures=3
[INFO] Registered health check: memory (critical=False)
[INFO] Watchdog monitoring started
[INFO] System Health: overall_healthy=True, checks=7, failures=0
[WARNING] Health check 'camera_a' failed: Check failed (1/3 failures)
[INFO] Attempting recovery for 'camera_a' (attempt 1/3)
[INFO] Recovery successful for component: camera_a
```

## Production Best Practices

1. **Configure appropriate thresholds:**
   - Set `max_failures` based on system stability
   - Adjust check intervals based on resource constraints
   - Configure cooldown periods to prevent recovery thrashing

2. **Monitor health metrics:**
   - Review health check logs regularly
   - Track recovery success rates
   - Identify recurring failures

3. **Implement custom recovery strategies:**
   - Create component-specific recovery callbacks
   - Test recovery procedures thoroughly
   - Document recovery actions

4. **Handle graceful shutdown:**
   - Respond to shutdown signals promptly
   - Verify cleanup completion
   - Log final statistics

## Testing

Run the system with health monitoring enabled:

```bash
# Development mode (includes visualization)
MODO=development python3 main.py

# Production mode
MODO=production python3 main.py
```

Monitor health status in logs:
```bash
tail -f logs/smart_traffic_*.log | grep -E "(Health|Watchdog|Recovery)"
```

## Configuration

Health monitoring can be configured through:

1. **Environment Variables:** Set `MODO=development` for verbose logging
2. **Settings:** Adjust thresholds in health check registration
3. **Recovery Strategies:** Customize per-component recovery actions

## Troubleshooting

**Issue: Too many recovery attempts**
- Solution: Increase cooldown period or max_attempts

**Issue: Recovery not working**
- Solution: Check recovery callback implementation
- Verify component is actually recoverable

**Issue: False positive failures**
- Solution: Adjust health check thresholds
- Add check-specific logic to handle edge cases

**Issue: Watchdog consuming too many resources**
- Solution: Increase check_interval
- Disable non-critical checks

## API Reference

### HealthCheck

- `register_check(name, check_func, description, critical)` - Register health check
- `run_check(name)` - Run specific check
- `run_all_checks()` - Run all checks
- `is_healthy(include_non_critical)` - Check overall health
- `get_status_report(include_history)` - Get detailed report
- `reset_failures(check_name)` - Reset failure counters

### Watchdog

- `register_recovery_strategy(strategy, callback)` - Register recovery strategy
- `start()` - Start monitoring
- `stop(timeout)` - Stop monitoring
- `get_statistics()` - Get watchdog statistics
- `reset_recovery_attempts(component)` - Reset recovery counters

### ShutdownManager

- `request_shutdown(signum)` - Request shutdown
- `is_shutdown_requested()` - Check shutdown status
- `wait_for_shutdown(timeout)` - Wait for shutdown
- `get_signal()` - Get triggering signal

## Future Enhancements

Potential improvements for the health monitoring system:

1. **HTTP Health Endpoint** - Expose health status via HTTP for external monitoring
2. **Metrics Export** - Export metrics in Prometheus format
3. **Alert System** - Send alerts on critical failures (email, webhook)
4. **Historical Analysis** - Analyze trends and predict failures
5. **Dynamic Thresholds** - Adjust thresholds based on historical data
6. **Component Dependencies** - Model dependencies between components
7. **Recovery Orchestration** - Coordinate recovery across multiple components