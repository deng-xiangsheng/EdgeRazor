"""
Logging utilities for EdgeRazor.

This module provides a centralized logging system for the EdgeRazor lightweight framework.
It includes specialized loggers for different components like QAT, distillation, etc.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


class EdgeRazorFormatter(logging.Formatter):
    """Custom formatter for EdgeRazor logs with color support and structured output."""

    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to log level name
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            colored_levelname = f"{color}{record.levelname:8s}{self.COLORS['RESET']}"
        else:
            colored_levelname = record.levelname

        # Create custom format with component information
        component = getattr(record, 'component', 'EdgeRazor')
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S.%f')[:-3]

        # Format: [HH:MM:SS.mmm] [LEVEL] [Component] Message
        formatted_msg = f"[{timestamp}] [{colored_levelname}] [{component:>8s}] {record.getMessage()}"

        # Add exception info if present
        if record.exc_info:
            formatted_msg += f"\n{self.formatException(record.exc_info)}"

        return formatted_msg


class EdgeRazorLogger:
    """
    Centralized logger for EdgeRazor framework.
    
    Provides component-specific logging with consistent formatting and multiple output options.
    """

    _loggers = {}
    _initialized = False

    @classmethod
    def setup_logging(cls,
                     level: str | int = logging.INFO,
                     log_file: str | Path | None = None,
                     console_output: bool = True) -> None:
        """
        Setup the global logging configuration for EdgeRazor.
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional file path for log output
            console_output: Whether to output logs to console
        """
        if cls._initialized:
            return

        # Convert string level to logging constant
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        # Create custom formatter
        formatter = EdgeRazorFormatter()

        # Setup console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)

        # Setup file handler if specified
        file_handler = None
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)

        # Store handlers for component loggers
        cls._console_handler = console_handler if console_output else None
        cls._file_handler = file_handler
        cls._level = level
        cls._initialized = True

    @classmethod
    def get_logger(cls, component: str) -> logging.Logger:
        """
        Get a logger for a specific component.
        
        Args:
            component: Component name (e.g., 'QAT', 'Distill', 'Config')
            
        Returns:
            Logger instance for the component
        """
        if not cls._initialized:
            cls.setup_logging()

        if component not in cls._loggers:
            logger = logging.getLogger(f"EdgeRazor.{component}")
            logger.setLevel(cls._level)

            # Clear any existing handlers
            logger.handlers.clear()
            logger.propagate = False

            # Add configured handlers
            if cls._console_handler:
                logger.addHandler(cls._console_handler)
            if cls._file_handler:
                logger.addHandler(cls._file_handler)

            # Store original log methods and create component-aware versions
            original_debug = logger.debug
            original_info = logger.info
            original_warning = logger.warning
            original_error = logger.error
            original_critical = logger.critical

            def make_component_method(original_method):
                def component_method(msg, *args, **kwargs):
                    # Add component info to extra data
                    if 'extra' not in kwargs:
                        kwargs['extra'] = {}
                    kwargs['extra']['component'] = component
                    return original_method(msg, *args, **kwargs)
                return component_method

            logger.debug = make_component_method(original_debug)
            logger.info = make_component_method(original_info)
            logger.warning = make_component_method(original_warning)
            logger.error = make_component_method(original_error)
            logger.critical = make_component_method(original_critical)

            cls._loggers[component] = logger

        return cls._loggers[component]


def get_logger(component: str) -> logging.Logger:
    """
    Convenience function to get a component logger.
    
    Args:
        component: Component name
        
    Returns:
        Logger instance
    """
    return EdgeRazorLogger.get_logger(component)


def setup_logging(**kwargs):
    """Convenience function to setup logging.
    Example:
        setup_logging(level='ERROR')
    """
    return EdgeRazorLogger.setup_logging(**kwargs)
