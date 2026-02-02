"""
Logging module, provides unified logging functionality
"""

import logging
import os
import sys
from datetime import datetime

class Logger:
    """
    Logger class, provides a unified logging interface
    """
    
    # Log level mapping
    LEVELS = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    def __init__(self, name='UIMatch', level='info', log_dir='logs'):
        """
        Initialize the logger

        Args:
            name: Logger name
            level: Log level, options: debug, info, warning, error, critical
            log_dir: Directory for saving log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LEVELS.get(level.lower(), logging.INFO))
        self.logger.propagate = False
        
        # If handlers already exist, do not add more
        if self.logger.handlers:
            return
            
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

            
        # Log filename includes date
        log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.logger.level)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(self.logger.level)
        
        # Create log format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # Set format
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def debug(self, message):
        """Log a debug level message"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log an info level message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log a warning level message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an error level message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log a critical level message"""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log an exception message, including stack trace"""
        self.logger.exception(message)


# Create default logger instance, can be directly imported and used in other modules
logger = Logger()


def get_logger(name=None, level=None, log_dir=None, force_new_handlers=False):
    """
    Get a logger instance

    Args:
        name: Logger name, defaults to 'UIMatch'
        level: Log level, defaults to 'info'
        log_dir: Directory for saving log files, defaults to 'logs'
        force_new_handlers: Whether to force creation of new handlers, even if the logger already exists

    Returns:
        Logger instance
    """
    logger_name = name or 'UIMatch'
    logger_level = level or 'info'
    logger_dir = log_dir or 'logs'
    
    # If forcing new handlers, remove existing handlers first
    if force_new_handlers:
        existing_logger = logging.getLogger(logger_name)
        for handler in list(existing_logger.handlers):
            existing_logger.removeHandler(handler)
    
    return Logger(
        name=logger_name,
        level=logger_level,
        log_dir=logger_dir
    )