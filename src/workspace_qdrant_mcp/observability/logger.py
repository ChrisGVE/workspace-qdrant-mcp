"""
Structured logging implementation for workspace-qdrant-mcp.

Provides JSON-formatted structured logging with proper log levels, context management,
and production-ready configuration. Replaces print statements throughout the application
with structured logging that provides actionable insights for operators.

Features:
    - JSON-structured log output for parsing by log aggregators
    - Automatic context injection (request IDs, operation names, etc.)
    - Configurable log levels and output destinations
    - Performance-optimized logging with lazy evaluation
    - Integration with observability systems

Example:
    ```python
    from workspace_qdrant_mcp.observability import get_logger, LogContext
    
    logger = get_logger(__name__)
    
    # Basic structured logging
    logger.info("Processing document", 
                document_id="doc123", 
                collection="my-project", 
                size_bytes=1024)
    
    # Context-aware logging
    with LogContext(operation="search", query_id="q456"):
        logger.debug("Starting hybrid search")
        logger.warning("Low similarity scores", threshold=0.7, max_score=0.65)
    ```
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars


# Context variables for request/operation tracking
_log_context: ContextVar[Dict[str, Any]] = ContextVar('log_context', default={})


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON with structured fields."""
        # Base log entry structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
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
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add thread/process info for debugging
        log_entry.update({
            "thread_id": record.thread,
            "thread_name": record.threadName,
            "process_id": record.process,
        })
        
        return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str):
        self.name = name
        self._logger = logging.getLogger(name)
    
    def _log_with_extra(self, level: int, msg: str, *args, **kwargs):
        """Log message with extra structured fields."""
        # Extract extra fields from kwargs
        extra_fields = {}
        log_kwargs = {}
        
        for key, value in kwargs.items():
            if key in ('exc_info', 'stack_info', 'stacklevel'):
                log_kwargs[key] = value
            else:
                extra_fields[key] = value
        
        # Create a custom LogRecord with extra fields
        if extra_fields:
            extra = {'extra_fields': extra_fields}
            log_kwargs['extra'] = extra
        
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
        kwargs.setdefault('exc_info', True)
        self._log_with_extra(logging.ERROR, msg, *args, **kwargs)


def configure_logging(
    level: Union[str, int] = "INFO",
    json_format: bool = True,
    log_file: Optional[Path] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
) -> None:
    """Configure structured logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting for structured logs
        log_file: Path to log file (optional, logs to console if None)
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output logs to console
    """
    # Convert string level to logging level
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers to avoid excessive verbosity
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for the given name.
    
    Args:
        name: Logger name, typically __name__
        
    Returns:
        StructuredLogger instance with JSON formatting support
    """
    return StructuredLogger(name)


@contextmanager
def LogContext(**context):
    """Context manager for adding structured context to all log messages.
    
    Args:
        **context: Key-value pairs to add to log context
        
    Example:
        ```python
        with LogContext(operation="search", user_id="user123"):
            logger.info("Starting search")  # Will include operation and user_id
            logger.error("Search failed")   # Will include operation and user_id
        ```
    """
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


class PerformanceLogger:
    """Logger for performance monitoring and timing operations."""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    @contextmanager
    def time_operation(self, operation: str, **context):
        """Context manager for timing operations with structured logging.
        
        Args:
            operation: Name of the operation being timed
            **context: Additional context fields
            
        Example:
            ```python
            perf_logger = PerformanceLogger(logger)
            with perf_logger.time_operation("document_search", collection="my-project"):
                # Perform search operation
                results = search_documents(query)
            ```
        """
        start_time = time.perf_counter()
        
        self.logger.debug("Operation started", 
                         operation=operation, 
                         **context)
        
        try:
            yield
            
            duration = time.perf_counter() - start_time
            self.logger.info("Operation completed", 
                           operation=operation,
                           duration_seconds=duration,
                           success=True,
                           **context)
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            self.logger.error("Operation failed",
                            operation=operation,
                            duration_seconds=duration,
                            success=False,
                            error_type=type(e).__name__,
                            error_message=str(e),
                            **context,
                            exc_info=True)
            raise


def setup_logging_from_env() -> None:
    """Set up logging configuration from environment variables.
    
    Environment Variables:
        LOG_LEVEL: Logging level (default: INFO)
        LOG_FORMAT: json or text (default: json)
        LOG_FILE: Path to log file (optional)
        LOG_CONSOLE: Enable console output (default: true)
        LOG_MAX_FILE_SIZE: Max file size in bytes (default: 10MB)
        LOG_BACKUP_COUNT: Number of backup files (default: 5)
    """
    # Get configuration from environment
    level = os.getenv("LOG_LEVEL", "INFO")
    json_format = os.getenv("LOG_FORMAT", "json").lower() == "json"
    log_file_path = os.getenv("LOG_FILE")
    console_output = os.getenv("LOG_CONSOLE", "true").lower() == "true"
    max_file_size = int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024)))
    backup_count = int(os.getenv("LOG_BACKUP_COUNT", "5"))
    
    # Convert log file path if provided
    log_file = Path(log_file_path) if log_file_path else None
    
    # Configure logging
    configure_logging(
        level=level,
        json_format=json_format,
        log_file=log_file,
        max_file_size=max_file_size,
        backup_count=backup_count,
        console_output=console_output,
    )


# Initialize logging from environment if not already configured
if not logging.getLogger().handlers:
    setup_logging_from_env()