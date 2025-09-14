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

__all__ = ['get_logger', 'configure_logging']