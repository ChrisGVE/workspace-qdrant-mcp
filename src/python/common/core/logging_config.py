"""
Structured logging configuration for workspace-qdrant-mcp.

This module provides comprehensive logging setup with structured JSON output,
performance monitoring, and integration with observability platforms.
"""

import json
import logging
import logging.config
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog


class PerformanceLogger:
    """Performance monitoring and structured logging."""

    def __init__(self):
        self.setup_structured_logging()
        self.logger = structlog.get_logger(__name__)

    def setup_structured_logging(self):
        """Configure structured logging with JSON output."""

        # Configure structlog for JSON output
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

        # Set up Python logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
        )

    def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.logger.info(
            "performance_metric",
            operation=operation,
            duration_ms=round(duration * 1000, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kwargs,
        )


# Global performance logger instance
perf_logger = PerformanceLogger()


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up comprehensive logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for file logging
    """
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "json": {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "structured",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "workspace_qdrant_mcp": {
                "level": log_level,
                "handlers": ["console"],
                "propagate": False,
            }
        },
        "root": {"level": log_level, "handlers": ["console"]},
    }

    # Add file handler if specified
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "level": log_level,
            "formatter": "json",
            "filename": log_file,
        }
        config["loggers"]["workspace_qdrant_mcp"]["handlers"].append("file")
        config["root"]["handlers"].append("file")

    logging.config.dictConfig(config)


class ContextTimer:
    """Context manager for timing operations."""

    def __init__(self, operation: str, logger: Optional[Any] = None, **kwargs):
        self.operation = operation
        self.logger = logger or perf_logger
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.logger.log_performance(
                self.operation, duration, success=exc_type is None, **self.kwargs
            )


def get_logger(name: str) -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)
