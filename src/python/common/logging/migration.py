"""
Backward compatibility layer and migration utilities.

This module provides drop-in replacement functions for existing logging patterns
and handles the migration from standard logging to the unified logging system.
"""

import logging
import os
from typing import Any, Dict, Optional, Union

from .core import StructuredLogger
from .config import LoggingConfig, detect_mcp_mode


# Global configuration cache
_global_config: Optional[LoggingConfig] = None
_initialized: bool = False


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance (drop-in replacement for logging.getLogger).

    This function provides a drop-in replacement for `logging.getLogger(__name__)`
    that automatically handles MCP stdio detection and console output suppression.

    Args:
        name: Logger name, typically __name__

    Returns:
        StructuredLogger instance with MCP stdio awareness

    Example:
        Replace standard logging imports:

        # OLD:
        import logging
        logger = logging.getLogger(__name__)

        # NEW:
        from common.logging.loguru_config import get_logger
        logger = get_logger(__name__)
    """
    global _global_config

    # Initialize configuration if not already done
    if _global_config is None:
        _global_config = LoggingConfig.from_environment()

    return StructuredLogger(name, _global_config)


def configure_unified_logging(
    level: Union[str, int] = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_output: bool = True,
    force_mcp_detection: bool = False,
) -> None:
    """Configure the unified logging system.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting for structured logs
        log_file: Path to log file (optional)
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output logs to console
        force_mcp_detection: Force MCP stdio mode detection
    """
    global _global_config, _initialized

    # Detect MCP mode
    stdio_mode = detect_mcp_mode() or force_mcp_detection

    # Create configuration
    _global_config = LoggingConfig(
        level=level,
        json_format=json_format,
        log_file=log_file,
        max_file_size=max_file_size,
        backup_count=backup_count,
        console_output=console_output,
        stdio_mode=stdio_mode,
        force_stderr=stdio_mode,
    )

    # Reset the root logger to apply new configuration
    root_logger = logging.getLogger()

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create a dummy logger to trigger root logger configuration
    dummy_logger = StructuredLogger("_config_dummy", _global_config)

    # Mark as initialized
    _initialized = True


def _initialize_logging_if_needed() -> None:
    """Initialize logging configuration if not already done.

    This function is called automatically on module import to ensure
    MCP stdio detection happens early in the application lifecycle.
    """
    global _initialized

    # Skip auto-initialization if explicitly disabled
    if os.getenv("WQM_LOG_INIT", "true").lower() == "false":
        return

    # Skip if already initialized
    if _initialized:
        return

    # Check if root logger already has handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Logging already configured, just set up MCP safety checks
        from .handlers import setup_handler_safety_checks
        setup_handler_safety_checks()
        _initialized = True
        return

    # Initialize with environment configuration
    configure_unified_logging()


def migrate_existing_loggers() -> Dict[str, int]:
    """Migrate existing loggers to use the unified logging system.

    This function scans for existing loggers in the system and replaces
    their handlers with MCP-aware versions.

    Returns:
        Dictionary with migration statistics
    """
    stats = {
        "loggers_migrated": 0,
        "handlers_replaced": 0,
        "handlers_removed": 0,
    }

    # Get all existing loggers
    logger_dict = logging.Logger.manager.loggerDict

    for name, logger_obj in logger_dict.items():
        if not isinstance(logger_obj, logging.Logger):
            continue

        if logger_obj.handlers:
            stats["loggers_migrated"] += 1

            # Count handlers being replaced
            stats["handlers_replaced"] += len(logger_obj.handlers)

            # Clear existing handlers
            for handler in logger_obj.handlers[:]:
                logger_obj.removeHandler(handler)
                stats["handlers_removed"] += 1

    # Reset root logger configuration
    configure_unified_logging()

    return stats


def get_legacy_logger(name: str) -> logging.Logger:
    """Get a legacy logger for compatibility with third-party libraries.

    This function returns a standard logging.Logger instance but ensures
    the root logger is configured with MCP stdio awareness.

    Args:
        name: Logger name

    Returns:
        Standard logging.Logger instance

    Note:
        This function is for compatibility with libraries that expect
        standard logging.Logger instances. For new code, use get_logger().
    """
    # Ensure unified logging is initialized
    _initialize_logging_if_needed()

    return logging.getLogger(name)


def is_mcp_mode() -> bool:
    """Check if the application is running in MCP stdio mode.

    Returns:
        True if MCP stdio mode is detected
    """
    return detect_mcp_mode()


def suppress_console_output() -> None:
    """Manually suppress all console output.

    This function can be called to forcibly suppress console output,
    useful when MCP mode detection fails but we know we're in MCP mode.
    """
    root_logger = logging.getLogger()

    # Remove all console handlers
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            root_logger.removeHandler(handler)


# Context management for backward compatibility
class LogContext:
    """Context manager for adding structured context to all log messages.

    This provides backward compatibility with the existing LogContext from
    the observability module.

    Args:
        **context: Key-value pairs to add to log context

    Example:
        with LogContext(operation="search", user_id="user123"):
            logger.info("Starting search")  # Will include operation and user_id
    """

    def __init__(self, **context):
        self.context = context
        self.token = None

    def __enter__(self):
        from .formatters import _log_context

        # Get current context
        current_context = _log_context.get({})

        # Merge with new context
        new_context = {**current_context, **self.context}

        # Set new context
        self.token = _log_context.set(new_context)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token is not None:
            from .formatters import _log_context
            _log_context.reset(self.token)


class PerformanceLogger:
    """Logger for performance monitoring and timing operations.

    Provides backward compatibility with the existing PerformanceLogger from
    the observability module.
    """

    def __init__(self, logger: StructuredLogger):
        self.logger = logger

    def time_operation(self, operation: str, **context):
        """Context manager for timing operations with structured logging."""
        import time
        from contextlib import contextmanager

        @contextmanager
        def _time_operation():
            start_time = time.perf_counter()

            self.logger.debug("Operation started", operation=operation, **context)

            try:
                yield

                duration = time.perf_counter() - start_time
                self.logger.info(
                    "Operation completed",
                    operation=operation,
                    duration_seconds=duration,
                    success=True,
                    **context,
                )

            except Exception as e:
                duration = time.perf_counter() - start_time
                self.logger.error(
                    "Operation failed",
                    operation=operation,
                    duration_seconds=duration,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    **context,
                    exc_info=True,
                )
                raise

        return _time_operation()


# Compatibility functions for gradual migration
def setup_logging_from_env() -> None:
    """Set up logging configuration from environment variables.

    This provides compatibility with the existing setup_logging_from_env function.
    """
    configure_unified_logging()


def configure_logging(**kwargs) -> None:
    """Configure structured logging (compatibility function).

    This provides compatibility with the existing configure_logging function.
    """
    configure_unified_logging(**kwargs)