"""
Core structured logging implementation with MCP stdio awareness.

This module provides the StructuredLogger class that automatically handles
MCP stdio mode detection and console output suppression.
"""

import logging
import sys
from typing import Any

from .config import LoggingConfig, should_suppress_console
from .handlers import create_console_handler, create_file_handler
from .formatters import get_formatter


# Comprehensive list of third-party loggers to suppress in stdio mode
THIRD_PARTY_LOGGERS = [
    'httpx', 'httpcore', 'qdrant_client', 'urllib3', 'asyncio',
    'fastembed', 'grpc', 'structlog', 'typer', 'pydantic', 'rich',
    'uvicorn', 'fastapi', 'watchdog', 'watchfiles', 'aiohttp',
    'aiofiles', 'requests', 'chardet', 'pypdf', 'python_docx',
    'python_pptx', 'ebooklib', 'beautifulsoup4', 'lxml',
    'markdown', 'pygments', 'psutil', 'tqdm', 'multipart',
    'tokenizers', 'transformers', 'sentence_transformers',
    'bitsandbytes', 'safetensors', 'huggingface_hub'
]


class StructuredLogger:
    """Enhanced logger with structured logging capabilities and MCP stdio awareness.

    This logger automatically detects MCP stdio mode and suppresses console output
    when needed to prevent interference with the MCP JSON-RPC protocol.

    Features:
        - Automatic MCP stdio detection
        - Structured logging with extra fields
        - JSON formatting for structured logs
        - Context preservation for observability
        - Performance monitoring integration
        - Drop-in replacement for standard logging.Logger
    """

    def __init__(self, name: str, config: LoggingConfig = None):
        """Initialize the structured logger.

        Args:
            name: Logger name, typically __name__
            config: Optional logging configuration (defaults to environment)
        """
        self.name = name
        self.config = config or LoggingConfig.from_environment()
        self._logger = logging.getLogger(name)

        # Configure this logger if not already configured
        self._configure_if_needed()

    def _configure_if_needed(self):
        """Configure the underlying logger if not already configured."""
        # Check if the root logger has been configured
        root_logger = logging.getLogger()

        # If root logger has no handlers, we need to configure it
        if not root_logger.handlers:
            self._configure_root_logger()

        # Set level for this specific logger
        if isinstance(self.config.level, str):
            level = getattr(logging, self.config.level.upper())
        else:
            level = self.config.level
        self._logger.setLevel(level)

    def _configure_root_logger(self):
        """Configure the root logger with MCP-aware settings."""
        root_logger = logging.getLogger()

        # Set root logger level
        if isinstance(self.config.level, str):
            level = getattr(logging, self.config.level.upper())
        else:
            level = self.config.level
        root_logger.setLevel(level)

        # Clear existing handlers to avoid duplication
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Determine if console should be suppressed
        suppress_console = should_suppress_console(self.config)

        if suppress_console:
            # If console output is suppressed, add a null handler to prevent
            # Python's logging system from falling back to stderr/stdout
            from .handlers import create_null_handler
            null_handler = create_null_handler()
            root_logger.addHandler(null_handler)
        else:
            # Create console handler only if not suppressed
            console_handler = create_console_handler(self.config)
            if console_handler:
                root_logger.addHandler(console_handler)

        # Create file handler if configured
        if self.config.log_file:
            file_handler = create_file_handler(self.config)
            if file_handler:
                root_logger.addHandler(file_handler)

        # Configure third-party loggers
        self._configure_third_party_loggers(suppress_console)

    def _configure_third_party_loggers(self, suppress_console: bool):
        """Configure third-party library loggers for MCP stdio mode compatibility.

        Args:
            suppress_console: Whether console output should be completely suppressed
        """
        import os
        stdio_mode = os.getenv("WQM_STDIO_MODE", "false").lower() == "true"

        for logger_name in THIRD_PARTY_LOGGERS:
            third_party_logger = logging.getLogger(logger_name)

            if stdio_mode or suppress_console:
                # In stdio mode, completely silence third-party loggers
                from .handlers import create_null_handler
                third_party_logger.handlers.clear()
                third_party_logger.addHandler(create_null_handler())
                third_party_logger.setLevel(logging.CRITICAL + 1)
                third_party_logger.disabled = True
                third_party_logger.propagate = False
            else:
                # In normal mode, just reduce verbosity
                if logger_name in ['httpx', 'httpcore', 'urllib3', 'requests']:
                    third_party_logger.setLevel(logging.WARNING)
                elif logger_name in ['qdrant_client', 'fastembed']:
                    third_party_logger.setLevel(logging.INFO)
                elif logger_name in ['asyncio', 'uvicorn', 'fastapi']:
                    third_party_logger.setLevel(logging.WARNING)
                elif logger_name in ['tokenizers', 'transformers', 'sentence_transformers']:
                    third_party_logger.setLevel(logging.ERROR)
                elif logger_name in ['grpc', 'structlog', 'typer', 'rich']:
                    third_party_logger.setLevel(logging.WARNING)
                else:
                    # Default to WARNING for other libraries
                    third_party_logger.setLevel(logging.WARNING)

    def _log_with_extra(self, level: int, msg: str, *args, **kwargs):
        """Log message with extra structured fields."""
        # Extract extra fields from kwargs
        extra_fields = {}
        log_kwargs = {}

        for key, value in kwargs.items():
            if key in ("exc_info", "stack_info", "stacklevel"):
                log_kwargs[key] = value
            else:
                extra_fields[key] = value

        # Create a custom LogRecord with extra fields
        if extra_fields:
            extra = {"extra_fields": extra_fields}
            log_kwargs["extra"] = extra

        self._logger.log(level, msg, *args, **log_kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with structured fields."""
        self._log_with_extra(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message with structured fields."""
        self._log_with_extra(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with structured fields."""
        self._log_with_extra(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message with structured fields."""
        self._log_with_extra(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with structured fields."""
        self._log_with_extra(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback and structured fields."""
        kwargs.setdefault("exc_info", True)
        self._log_with_extra(logging.ERROR, msg, *args, **kwargs)

    # Provide compatibility methods with standard logger interface
    def setLevel(self, level):
        """Set the logging level."""
        self._logger.setLevel(level)

    def isEnabledFor(self, level):
        """Check if logger is enabled for the given level."""
        return self._logger.isEnabledFor(level)

    def getEffectiveLevel(self):
        """Get the effective logging level."""
        return self._logger.getEffectiveLevel()

    def addHandler(self, handler):
        """Add a handler to the logger."""
        self._logger.addHandler(handler)

    def removeHandler(self, handler):
        """Remove a handler from the logger."""
        self._logger.removeHandler(handler)

    def addFilter(self, filter):
        """Add a filter to the logger."""
        self._logger.addFilter(filter)

    def removeFilter(self, filter):
        """Remove a filter from the logger."""
        self._logger.removeFilter(filter)

    @property
    def handlers(self):
        """Get the list of handlers."""
        return self._logger.handlers

    @property
    def level(self):
        """Get the current logging level."""
        return self._logger.level

    @level.setter
    def level(self, value):
        """Set the logging level."""
        self._logger.setLevel(value)

    def __repr__(self):
        """Return string representation of logger."""
        return f"StructuredLogger(name='{self.name}', level={self.level})"