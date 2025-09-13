"""
Logging formatters for JSON and console output.

This module provides formatters that are aware of MCP stdio mode and
structure log output appropriately for different contexts.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from contextvars import ContextVar


# Context variables for request/operation tracking
_log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging output.

    Formats log records as JSON with structured fields including:
    - Standard log fields (timestamp, level, logger, message, etc.)
    - Context variables for request/operation tracking
    - Extra fields from log record
    - Exception information if present
    - Thread and process information for debugging
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with structured fields."""
        # Base log entry structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add context from contextvars
        context = _log_context.get({})
        if context:
            log_entry.update(context)

        # Add extra fields from log record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add thread/process info for debugging
        log_entry.update(
            {
                "thread_id": record.thread,
                "thread_name": record.threadName,
                "process_id": record.process,
            }
        )

        return json.dumps(log_entry, ensure_ascii=False, separators=(",", ":"))


class ConsoleFormatter(logging.Formatter):
    """Human-readable console formatter.

    Provides clean, readable output for development and debugging.
    Automatically includes structured fields when available.
    """

    def __init__(self, include_extra: bool = True):
        """Initialize console formatter.

        Args:
            include_extra: Whether to include extra fields in output
        """
        super().__init__(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        # Get basic formatted message
        formatted = super().format(record)

        # Add extra fields if available and requested
        if self.include_extra:
            extra_parts = []

            # Add context from contextvars
            context = _log_context.get({})
            if context:
                for key, value in context.items():
                    extra_parts.append(f"{key}={value}")

            # Add extra fields from log record
            if hasattr(record, "extra_fields"):
                for key, value in record.extra_fields.items():
                    extra_parts.append(f"{key}={value}")

            # Append extra parts to message
            if extra_parts:
                formatted += f" [{', '.join(extra_parts)}]"

        return formatted


def get_formatter(json_format: bool = True) -> logging.Formatter:
    """Get appropriate formatter based on format preference.

    Args:
        json_format: Whether to use JSON formatting

    Returns:
        Configured formatter instance
    """
    if json_format:
        return JSONFormatter()
    else:
        return ConsoleFormatter()


def get_context() -> Dict[str, Any]:
    """Get current log context.

    Returns:
        Current context variables as dictionary
    """
    return _log_context.get({}).copy()


def set_context(**context: Any) -> None:
    """Set log context variables.

    Args:
        **context: Key-value pairs to add to log context
    """
    current_context = _log_context.get({})
    new_context = {**current_context, **context}
    _log_context.set(new_context)


def clear_context() -> None:
    """Clear all log context variables."""
    _log_context.set({})


def bind_context(**context: Any):
    """Context manager for temporarily binding log context.

    Args:
        **context: Key-value pairs to add to log context

    Example:
        with bind_context(operation="search", user_id="123"):
            logger.info("Starting search")  # Will include operation and user_id
    """
    from contextlib import contextmanager

    @contextmanager
    def _bind_context():
        # Get current context
        current_context = _log_context.get({})

        # Merge with new context
        new_context = {**current_context, **context}

        # Set new context
        token = _log_context.set(new_context)

        try:
            yield
        finally:
            # Restore previous context
            _log_context.reset(token)

    return _bind_context()