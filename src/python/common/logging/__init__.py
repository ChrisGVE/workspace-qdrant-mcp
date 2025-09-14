"""
Logging package for workspace-qdrant-mcp using loguru.

This package provides the loguru-based logging system for the application.
Use loguru directly for all logging needs.

Example:
    from loguru import logger
    logger.info("Message here")
"""

# Re-export the main loguru config functions for convenience
from .loguru_config import setup_logging
from contextlib import asynccontextmanager
from typing import Optional, Any, Dict


class LogContext:
    """Simple context manager for structured logging."""
    def __init__(self, operation: str, **kwargs):
        self.operation = operation
        self.context = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class PerformanceLogger:
    """Simple performance logger using loguru."""
    def __init__(self):
        pass

    @asynccontextmanager
    async def operation(self, name: str, **kwargs):
        yield


def safe_log_error(message: str, **kwargs) -> None:
    """Safely log an error using loguru."""
    from loguru import logger
    logger.error(message, **kwargs)


__all__ = ['setup_logging', 'LogContext', 'PerformanceLogger', 'safe_log_error']