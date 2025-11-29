"""
Optimized Logging System for Smart Traffic Light Project

Production-optimized with async logging, unified output, and minimal overhead.
"""

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from queue import Queue
from logging.handlers import QueueHandler, QueueListener

# Simple global logger cache
_loggers = {}

# Default settings
DEFAULT_LOG_DIR = 'logs'
DEFAULT_LOG_LEVEL = 'INFO'

# Production mode detection
IS_PRODUCTION = os.getenv('MODE', os.getenv('MODO', 'production')).lower() == 'production'

# Async logging components (production only)
_log_queue = None
_queue_listener = None


def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Get or create an optimized logger with async support in production
    
    Args:
        name: Logger name
        level: Optional log level
        
    Returns:
        Logger instance
    """
    global _log_queue, _queue_listener
    
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]
    
    # Create new logger
    logger = logging.getLogger(name)
    log_level = level or DEFAULT_LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Simple formatter (no colors in production for performance)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        if IS_PRODUCTION:
            # PRODUCTION MODE: Async logging with unified file
            if _log_queue is None:
                _log_queue = Queue(-1)  # Unlimited queue
                
                # Single unified log file with rotation
                log_dir = Path(DEFAULT_LOG_DIR)
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Larger rotation (50MB instead of 10MB)
                file_handler = RotatingFileHandler(
                    log_dir / 'traffic_light.log',
                    maxBytes=50*1024*1024,  # 50MB
                    backupCount=3  # Keep only 3 backups
                )
                file_handler.setFormatter(formatter)
                
                # Start queue listener (async logging)
                _queue_listener = QueueListener(_log_queue, file_handler, respect_handler_level=True)
                _queue_listener.start()
            
            # Use queue handler for async logging
            queue_handler = QueueHandler(_log_queue)
            logger.addHandler(queue_handler)
            
            # Minimal console output in production (errors only)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.ERROR)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        else:
            # DEVELOPMENT MODE: Synchronous logging with colors
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler for development
            try:
                log_dir = Path(DEFAULT_LOG_DIR)
                log_dir.mkdir(parents=True, exist_ok=True)
                
                file_handler = RotatingFileHandler(
                    log_dir / 'traffic_light.log',
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass
    
    # Cache and return
    _loggers[name] = logger
    return logger


# For backward compatibility
def setup_logger(name: str, **kwargs) -> logging.Logger:
    """Backward compatible setup function"""
    return get_logger(name, kwargs.get('level'))


def log_execution_time(func):
    """Simple decorator for timing"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = get_logger('performance')
        logger.info(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


class LogContext:
    """Simple context manager for logging"""
    
    def __init__(self, logger, context_name: str):
        self.logger = logger
        self.context_name = context_name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"Entering: {self.context_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        if exc_type:
            self.logger.error(f"Failed: {self.context_name} after {elapsed:.4f}s: {exc_val}")
        else:
            self.logger.info(f"Completed: {self.context_name} in {elapsed:.4f}s")
        return False


def shutdown_logging():
    """Shutdown async logging gracefully (production only)"""
    global _queue_listener
    if _queue_listener:
        _queue_listener.stop()
        _queue_listener = None


if __name__ == '__main__':
    # Self-test
    logger = get_logger('test', 'DEBUG')
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print("Logger test complete!")
    shutdown_logging()