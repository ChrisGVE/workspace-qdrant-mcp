"""
MCP-aware console and file handlers for the unified logging system.

This module provides handlers that are aware of MCP stdio mode and
avoid interfering with the JSON-RPC protocol on stdout.
"""

import logging
import logging.handlers
import sys
from typing import Optional

from .config import LoggingConfig
from .formatters import get_formatter


class MCPConsoleHandler(logging.StreamHandler):
    """Console handler that is aware of MCP stdio mode.

    This handler automatically routes output to stderr when in MCP stdio mode
    to avoid interfering with the JSON-RPC protocol on stdout.
    """

    def __init__(self, config: LoggingConfig):
        """Initialize MCP-aware console handler.

        Args:
            config: Logging configuration
        """
        # In MCP stdio mode, always use stderr to avoid protocol interference
        # In normal mode, use stdout unless force_stderr is set
        if config.stdio_mode or config.force_stderr:
            stream = sys.stderr
        else:
            stream = sys.stdout

        super().__init__(stream)

        # Set formatter
        formatter = get_formatter(config.json_format)
        self.setFormatter(formatter)

        # Set level
        if isinstance(config.level, str):
            level = getattr(logging, config.level.upper())
        else:
            level = config.level
        self.setLevel(level)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with additional MCP safety checks."""
        try:
            # Additional safety: check if we're in stdio mode and avoid stdout
            if hasattr(self, 'stream') and self.stream == sys.stdout:
                # Double-check environment to be extra safe
                import os
                if (os.getenv("WQM_STDIO_MODE", "false").lower() == "true" or
                    os.getenv("MCP_TRANSPORT") == "stdio"):
                    # Emergency fallback: redirect to stderr
                    self.stream = sys.stderr

            super().emit(record)
        except Exception:
            # Fail silently in MCP mode to avoid breaking the protocol
            # In development mode, we might want to see errors
            import os
            if not os.getenv("WQM_STDIO_MODE", "false").lower() == "true":
                self.handleError(record)


class QuietNullHandler(logging.Handler):
    """A completely quiet handler that discards all log records.

    Used when console output must be completely suppressed in MCP stdio mode.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Discard the log record silently."""
        pass

    def handle(self, record: logging.LogRecord) -> bool:
        """Handle a log record by discarding it."""
        return True

    def createLock(self) -> None:
        """No-op lock creation since we don't need synchronization."""
        self.lock = None


def create_console_handler(config: LoggingConfig) -> Optional[logging.Handler]:
    """Create appropriate console handler based on configuration.

    Args:
        config: Logging configuration

    Returns:
        Console handler instance or None if console output is suppressed
    """
    # Import here to avoid circular imports
    from .config import should_suppress_console

    # If console output should be completely suppressed, return None
    if should_suppress_console(config):
        return None

    # Create MCP-aware console handler
    return MCPConsoleHandler(config)


def create_file_handler(config: LoggingConfig) -> Optional[logging.Handler]:
    """Create file handler with rotation based on configuration.

    Args:
        config: Logging configuration

    Returns:
        File handler instance or None if no log file is configured
    """
    if not config.log_file:
        return None

    # Ensure log directory exists
    config.log_file.parent.mkdir(parents=True, exist_ok=True)

    # Create rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        filename=config.log_file,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count,
        encoding="utf-8",
    )

    # Set formatter (always use JSON for files)
    formatter = get_formatter(json_format=True)
    file_handler.setFormatter(formatter)

    # Set level
    if isinstance(config.level, str):
        level = getattr(logging, config.level.upper())
    else:
        level = config.level
    file_handler.setLevel(level)

    return file_handler


def create_null_handler() -> logging.Handler:
    """Create a null handler that discards all log records.

    Returns:
        Null handler instance
    """
    return QuietNullHandler()


def setup_handler_safety_checks():
    """Set up additional safety checks for existing handlers.

    This function can be called to add MCP stdio detection to existing
    logging handlers in the system.
    """
    import os

    # Check all existing handlers in the root logger
    root_logger = logging.getLogger()
    stdio_mode = os.getenv("WQM_STDIO_MODE", "false").lower() == "true"

    if stdio_mode:
        for handler in root_logger.handlers[:]:
            # Replace any stdout StreamHandlers with stderr versions
            if (isinstance(handler, logging.StreamHandler) and
                hasattr(handler, 'stream') and
                handler.stream == sys.stdout):

                # Create replacement handler using stderr
                new_handler = logging.StreamHandler(sys.stderr)
                new_handler.setLevel(handler.level)
                new_handler.setFormatter(handler.formatter)

                # Replace the handler
                root_logger.removeHandler(handler)
                root_logger.addHandler(new_handler)