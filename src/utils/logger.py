"""
Simple, Functional Logging System for Smart Traffic Light Project

Simplified version without complex threading locks that cause deadlocks.
"""

import logging
import sys
from pathlib import Path

# Simple global logger cache
_loggers = {}

# Default settings
DEFAULT_LOG_DIR = 'logs'
DEFAULT_LOG_LEVEL = 'INFO'


def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Get or create a simple logger
    
    Args:
        name: Logger name
        level: Optional log level
        
    Returns:
        Logger instance
    """
    # Return cached logger if exists
    if name in _loggers:
        return _loggers[name]
    
    # Create new logger
    logger = logging.getLogger(name)
    log_level = level or DEFAULT_LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Simple formatter
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Optional file handler
        try:
            log_dir = Path(DEFAULT_LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / 'traffic_light.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Silently fail if can't create file handler
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


if __name__ == '__main__':
    # Self-test
    logger = get_logger('test', 'DEBUG')
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    print("Logger test complete!")