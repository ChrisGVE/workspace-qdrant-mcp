"""
Logging package for workspace-qdrant-mcp using loguru.

This package provides the loguru-based logging system for the application.
Import the get_logger function from loguru_config directly.

Example:
    from common.logging.loguru_config import get_logger
    logger = get_logger(__name__)
"""

# Re-export the main loguru config functions for convenience
from .loguru_config import get_logger, configure_logging

# Import compatibility classes from parent module using absolute path
import sys
import importlib.util
from pathlib import Path

# Load the parent logging module directly to get compatibility classes
parent_logging_path = Path(__file__).parent.parent / "logging.py"
spec = importlib.util.spec_from_file_location("parent_logging", parent_logging_path)
parent_logging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent_logging)

# Re-export compatibility classes
LogContext = parent_logging.LogContext
PerformanceLogger = parent_logging.PerformanceLogger
safe_log_error = parent_logging.safe_log_error

__all__ = ['get_logger', 'configure_logging', 'LogContext', 'PerformanceLogger', 'safe_log_error']