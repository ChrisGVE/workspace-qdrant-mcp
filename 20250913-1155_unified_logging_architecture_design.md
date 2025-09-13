# Unified Logging Architecture Design
**Date**: 2025-09-13 11:55 CET
**Task**: 208 - Design centralized logging architecture
**Status**: In Progress

## Executive Summary

This document defines the unified logging architecture for workspace-qdrant-mcp, consolidating the fragmented logging systems into a single, MCP-aware solution based on structlog. The design addresses the critical MCP stdio interference issue while providing rich console output for CLI users and maintaining backward compatibility.

## Architecture Overview

### Core Principles

1. **Single Source of Truth**: `common/logging` module as the sole logging configuration point
2. **Context-Aware**: Runtime detection of MCP vs CLI mode with appropriate behavior
3. **Performance-First**: Async-safe, lazy evaluation, minimal overhead
4. **Standards-Compliant**: ECS/OTEL semantic conventions support
5. **Backward Compatible**: Seamless migration path for existing code

### Module Structure

```
src/python/common/logging/
├── __init__.py              # Public API and convenience functions
├── config.py               # Configuration models and management
├── core.py                 # Core logging implementation
├── context.py              # Context management and request scoping
├── formatters.py           # JSON, console, and ECS formatters
├── handlers.py             # MCP-aware handlers and stream management
├── middleware.py           # Request/response logging middleware
├── performance.py          # Performance monitoring integration
├── testing.py              # Testing utilities and mocks
└── migration.py            # Migration utilities for existing code
```

## Core Architecture Components

### 1. Configuration System (`config.py`)

**Design**: Pydantic-based configuration with environment variable support and runtime reconfiguration capabilities.

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Dict, Any
from pathlib import Path
import os

class FileLoggingConfig(BaseModel):
    """File logging configuration."""
    enabled: bool = True
    path: Optional[Path] = Field(default=None, description="Log file path")
    max_size: int = Field(default=10_000_000, gt=0, description="Max file size in bytes")
    backup_count: int = Field(default=5, ge=0, description="Number of backup files")
    rotation: Literal["size", "time", "daily", "weekly"] = "size"

class ConsoleLoggingConfig(BaseModel):
    """Console logging configuration."""
    enabled: bool = True
    colors: bool = True
    format: Literal["json", "console", "dev"] = "console"
    stream: Literal["stdout", "stderr", "auto"] = "auto"

class MCPConfig(BaseModel):
    """MCP-specific logging configuration."""
    stdio_mode: bool = Field(default_factory=lambda: os.getenv("WQM_STDIO_MODE", "false").lower() == "true")
    quiet_mode: bool = Field(default_factory=lambda: os.getenv("MCP_QUIET_MODE", "true").lower() == "true")
    disable_console_logs: bool = Field(default_factory=lambda: os.getenv("DISABLE_MCP_CONSOLE_LOGS", "false").lower() == "true")

    @validator('disable_console_logs', always=True)
    def auto_disable_console_in_stdio(cls, v, values):
        """Auto-disable console logs in stdio mode if quiet mode is enabled."""
        if values.get('stdio_mode') and values.get('quiet_mode'):
            return True
        return v

class PerformanceConfig(BaseModel):
    """Performance and optimization settings."""
    async_safe: bool = True
    lazy_evaluation: bool = True
    batch_size: int = Field(default=100, gt=0)
    flush_interval: float = Field(default=1.0, gt=0)

class OpenTelemetryConfig(BaseModel):
    """OpenTelemetry integration configuration."""
    enabled: bool = Field(default=False)
    service_name: str = "workspace-qdrant-mcp"
    service_version: str = "1.0.0"
    trace_context: bool = True
    span_context: bool = True

class LoggingConfig(BaseModel):
    """Unified logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "console", "ecs"] = "json"

    file: FileLoggingConfig = FileLoggingConfig()
    console: ConsoleLoggingConfig = ConsoleLoggingConfig()
    mcp: MCPConfig = MCPConfig()
    performance: PerformanceConfig = PerformanceConfig()
    otel: OpenTelemetryConfig = OpenTelemetryConfig()

    # Environment-based overrides
    cli_mode: bool = Field(default_factory=lambda: os.getenv("WQM_CLI_MODE", "false").lower() == "true")

    class Config:
        env_prefix = "LOG_"
        case_sensitive = False
        validate_assignment = True

    @validator('console', always=True)
    def configure_console_for_context(cls, v, values):
        """Auto-configure console settings based on MCP/CLI context."""
        mcp_config = values.get('mcp', MCPConfig())
        cli_mode = values.get('cli_mode', False)

        if mcp_config.stdio_mode:
            # MCP stdio mode: JSON to stderr or disabled
            v.format = "json"
            v.stream = "stderr"
            v.enabled = not (mcp_config.quiet_mode or mcp_config.disable_console_logs)
        elif cli_mode:
            # CLI mode: Rich console output
            v.format = "console"
            v.colors = True
            v.stream = "stdout"
            v.enabled = True

        return v
```

### 2. Core Logging Implementation (`core.py`)

**Design**: Structlog-based implementation with MCP-aware processors and context management.

```python
import structlog
import logging
import sys
from typing import Any, Dict, Optional, Union
from contextvars import ContextVar
from .config import LoggingConfig
from .formatters import JSONFormatter, ConsoleFormatter, ECSFormatter
from .context import LogContext

# Global context variable for request scoping
_log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})

class StructuredLogger:
    """Enhanced structured logger with MCP awareness and context management."""

    def __init__(self, name: str, config: LoggingConfig):
        self.name = name
        self.config = config
        self._structlog_logger = structlog.get_logger(name)
        self._setup_processors()

    def _setup_processors(self):
        """Configure structlog processors based on configuration."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.contextvars.merge_contextvars,
            self._add_context_processor,
            structlog.processors.dict_tracebacks,
        ]

        # Add OpenTelemetry processor if enabled
        if self.config.otel.enabled:
            processors.append(self._otel_processor)

        # Add appropriate formatter
        if self.config.format == "json":
            processors.append(JSONFormatter())
        elif self.config.format == "ecs":
            processors.append(ECSFormatter(self.config.otel))
        else:
            processors.append(ConsoleFormatter(colors=self.config.console.colors))

        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

    def _add_context_processor(self, logger, method_name, event_dict):
        """Add context from contextvars to log event."""
        context = _log_context.get({})
        if context:
            event_dict.update(context)
        return event_dict

    def _otel_processor(self, logger, method_name, event_dict):
        """Add OpenTelemetry trace context to log events."""
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span.is_recording():
                span_context = span.get_span_context()
                event_dict.setdefault("otel", {}).update({
                    "trace_id": f"{span_context.trace_id:032x}",
                    "span_id": f"{span_context.span_id:016x}",
                    "service": {
                        "name": self.config.otel.service_name,
                        "version": self.config.otel.service_version
                    }
                })
        except ImportError:
            pass  # OpenTelemetry not available
        return event_dict

    def bind(self, **kwargs) -> 'StructuredLogger':
        """Create a bound logger with additional context."""
        bound_logger = self._structlog_logger.bind(**kwargs)
        new_logger = StructuredLogger(self.name, self.config)
        new_logger._structlog_logger = bound_logger
        return new_logger

    def debug(self, msg: str, **kwargs):
        """Log debug message with structured context."""
        self._structlog_logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message with structured context."""
        self._structlog_logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message with structured context."""
        self._structlog_logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs):
        """Log error message with structured context."""
        self._structlog_logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        """Log critical message with structured context."""
        self._structlog_logger.critical(msg, **kwargs)

    def exception(self, msg: str, **kwargs):
        """Log exception with traceback and structured context."""
        kwargs.setdefault("exc_info", True)
        self._structlog_logger.error(msg, **kwargs)

class LoggerManager:
    """Central manager for logging configuration and logger instances."""

    def __init__(self):
        self._config: Optional[LoggingConfig] = None
        self._loggers: Dict[str, StructuredLogger] = {}
        self._initialized = False

    def configure(self, config: Optional[LoggingConfig] = None):
        """Configure the logging system."""
        if config is None:
            config = LoggingConfig()

        self._config = config
        self._setup_stdlib_logging()
        self._initialized = True

    def _setup_stdlib_logging(self):
        """Configure standard library logging to work with structlog."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self._config.level))

        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add handlers based on configuration
        if self._config.console.enabled:
            self._add_console_handler(root_logger)

        if self._config.file.enabled and self._config.file.path:
            self._add_file_handler(root_logger)

    def _add_console_handler(self, logger):
        """Add console handler with appropriate stream."""
        stream_map = {
            "stdout": sys.stdout,
            "stderr": sys.stderr,
            "auto": sys.stderr if self._config.mcp.stdio_mode else sys.stdout
        }

        stream = stream_map[self._config.console.stream]
        handler = logging.StreamHandler(stream)
        handler.setLevel(getattr(logging, self._config.level))

        # Use plain formatter for stdlib logging (structlog handles formatting)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def _add_file_handler(self, logger):
        """Add file handler with rotation."""
        from logging.handlers import RotatingFileHandler

        self._config.file.path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            filename=self._config.file.path,
            maxBytes=self._config.file.max_size,
            backupCount=self._config.file.backup_count
        )
        handler.setLevel(getattr(logging, self._config.level))

        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def get_logger(self, name: str) -> StructuredLogger:
        """Get or create a structured logger instance."""
        if not self._initialized:
            self.configure()

        if name not in self._loggers:
            self._loggers[name] = StructuredLogger(name, self._config)

        return self._loggers[name]

# Global logger manager instance
_manager = LoggerManager()
```

### 3. Context Management (`context.py`)

**Design**: Context management for request scoping and performance monitoring.

```python
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional
import time
import uuid

_log_context: ContextVar[Dict[str, Any]] = ContextVar("log_context", default={})

class LogContext:
    """Context manager for structured logging context."""

    def __init__(self, **context):
        self.context = context
        self.token = None

    def __enter__(self):
        current_context = _log_context.get({})
        new_context = {**current_context, **self.context}
        self.token = _log_context.set(new_context)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.token:
            _log_context.reset(self.token)

@contextmanager
def request_context(request_id: Optional[str] = None, **kwargs):
    """Context manager for request-scoped logging."""
    if request_id is None:
        request_id = str(uuid.uuid4())

    with LogContext(request_id=request_id, **kwargs):
        yield request_id

@contextmanager
def operation_timer(operation: str, logger: Optional[Any] = None, **context):
    """Context manager for timing operations with structured logging."""
    start_time = time.perf_counter()

    if logger:
        logger.debug("Operation started", operation=operation, **context)

    try:
        yield
        duration = time.perf_counter() - start_time

        if logger:
            logger.info(
                "Operation completed",
                operation=operation,
                duration_seconds=duration,
                success=True,
                **context
            )

    except Exception as e:
        duration = time.perf_counter() - start_time

        if logger:
            logger.error(
                "Operation failed",
                operation=operation,
                duration_seconds=duration,
                success=False,
                error_type=type(e).__name__,
                error_message=str(e),
                **context,
                exc_info=True
            )
        raise
```

### 4. Backward Compatibility Layer (`migration.py`)

**Design**: Seamless migration path for existing `logging.getLogger()` calls.

```python
import logging
from typing import Optional
from .core import _manager

class CompatibilityLogger:
    """Compatibility wrapper that mimics stdlib Logger interface."""

    def __init__(self, name: str):
        self.name = name
        self._structured_logger = _manager.get_logger(name)

    def debug(self, msg, *args, **kwargs):
        """Debug logging with stdlib compatibility."""
        if args:
            msg = msg % args
        extra = kwargs.pop('extra', {})
        self._structured_logger.debug(msg, **extra)

    def info(self, msg, *args, **kwargs):
        """Info logging with stdlib compatibility."""
        if args:
            msg = msg % args
        extra = kwargs.pop('extra', {})
        self._structured_logger.info(msg, **extra)

    def warning(self, msg, *args, **kwargs):
        """Warning logging with stdlib compatibility."""
        if args:
            msg = msg % args
        extra = kwargs.pop('extra', {})
        self._structured_logger.warning(msg, **extra)

    def error(self, msg, *args, **kwargs):
        """Error logging with stdlib compatibility."""
        if args:
            msg = msg % args
        extra = kwargs.pop('extra', {})
        exc_info = kwargs.pop('exc_info', None)
        if exc_info:
            extra['exc_info'] = exc_info
        self._structured_logger.error(msg, **extra)

    def critical(self, msg, *args, **kwargs):
        """Critical logging with stdlib compatibility."""
        if args:
            msg = msg % args
        extra = kwargs.pop('extra', {})
        self._structured_logger.critical(msg, **extra)

    def exception(self, msg, *args, **kwargs):
        """Exception logging with stdlib compatibility."""
        if args:
            msg = msg % args
        extra = kwargs.pop('extra', {})
        self._structured_logger.exception(msg, **extra)

def get_logger(name: str) -> CompatibilityLogger:
    """Drop-in replacement for logging.getLogger() with structured logging."""
    return CompatibilityLogger(name)

# Monkey patch for seamless migration
def patch_stdlib_logging():
    """Monkey patch stdlib logging to use structured logging."""
    original_getLogger = logging.getLogger
    logging.getLogger = get_logger
    logging._original_getLogger = original_getLogger

def unpatch_stdlib_logging():
    """Restore original stdlib logging."""
    if hasattr(logging, '_original_getLogger'):
        logging.getLogger = logging._original_getLogger
        delattr(logging, '_original_getLogger')
```

### 5. Public API (`__init__.py`)

**Design**: Clean, simple public API for easy adoption.

```python
"""
Unified structured logging for workspace-qdrant-mcp.

This module provides MCP-aware structured logging with automatic context management,
performance monitoring, and OpenTelemetry integration.

Usage:
    from common.logging import get_logger, LogContext, configure_logging

    logger = get_logger(__name__)
    logger.info("Processing document", document_id="doc123", size=1024)

    with LogContext(request_id="req456"):
        logger.debug("Starting operation")
"""

from .core import _manager, StructuredLogger
from .config import LoggingConfig
from .context import LogContext, request_context, operation_timer
from .migration import patch_stdlib_logging, unpatch_stdlib_logging

__all__ = [
    'get_logger',
    'configure_logging',
    'LoggingConfig',
    'LogContext',
    'request_context',
    'operation_timer',
    'patch_stdlib_logging',
    'unpatch_stdlib_logging'
]

def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name, typically __name__

    Returns:
        StructuredLogger with MCP awareness and structured output
    """
    return _manager.get_logger(name)

def configure_logging(config: Optional[LoggingConfig] = None) -> None:
    """Configure the unified logging system.

    Args:
        config: Optional logging configuration. If None, uses environment variables.
    """
    _manager.configure(config)

def setup_logging_from_env() -> None:
    """Set up logging from environment variables (backward compatibility)."""
    configure_logging()

# Auto-configure logging if not explicitly disabled
import os
if os.getenv("WQM_LOG_INIT", "true").lower() != "false":
    configure_logging()
```

## Migration Strategy

### Phase 1: Foundation (Immediate)

1. **Create Module Structure**: Implement `common/logging` module with core components
2. **Maintain Existing Systems**: Keep both logging systems operational during transition
3. **Add Compatibility Layer**: Enable gradual migration of existing code

### Phase 2: Critical Path Migration (Priority)

1. **MCP Server Components**: Migrate `workspace_qdrant_mcp/server.py` first to fix stdio interference
2. **Core Tools**: Update MCP tools to use unified logging
3. **Test MCP Functionality**: Verify Claude Desktop compatibility

### Phase 3: Systematic Migration (Comprehensive)

1. **Automated Migration**: Use script to update import statements
2. **CLI Components**: Migrate wqm_cli components with rich console output
3. **Deprecate Old Systems**: Remove `common/core/logging_config.py`
4. **CLI Wrapper Fix**: Replace `logging.disable()` with proper configuration

### Phase 4: Enhancement (Advanced)

1. **Performance Optimization**: Implement async batching and lazy evaluation
2. **OpenTelemetry Integration**: Add full observability support
3. **Monitoring Integration**: Connect to performance monitoring systems

## Configuration Schema

### Environment Variables

```bash
# Core settings
LOG_LEVEL=INFO                      # Logging level
LOG_FORMAT=json                     # Output format
LOG_FILE_ENABLED=true              # Enable file logging
LOG_FILE_PATH=/var/log/wqm.log     # Log file path

# Console settings
LOG_CONSOLE_ENABLED=true           # Enable console logging
LOG_CONSOLE_COLORS=true            # Enable colors in console
LOG_CONSOLE_STREAM=auto            # Console stream (stdout/stderr/auto)

# MCP settings
WQM_STDIO_MODE=false               # Set by server when in stdio mode
MCP_QUIET_MODE=true                # Disable console in MCP mode
DISABLE_MCP_CONSOLE_LOGS=false     # Alternative console disable

# Performance settings
LOG_PERFORMANCE_ASYNC_SAFE=true    # Enable async safety
LOG_PERFORMANCE_LAZY_EVALUATION=true # Enable lazy evaluation

# OpenTelemetry settings
LOG_OTEL_ENABLED=false             # Enable OTEL integration
LOG_OTEL_SERVICE_NAME=workspace-qdrant-mcp
LOG_OTEL_SERVICE_VERSION=1.0.0
```

### Configuration File Support

```yaml
# logging.yaml
logging:
  level: INFO
  format: json

  file:
    enabled: true
    path: /var/log/workspace-qdrant-mcp.log
    max_size: 50000000
    backup_count: 10
    rotation: size

  console:
    enabled: true
    colors: true
    format: console
    stream: auto

  mcp:
    stdio_mode: false  # Auto-detected
    quiet_mode: true
    disable_console_logs: false

  performance:
    async_safe: true
    lazy_evaluation: true
    batch_size: 100
    flush_interval: 1.0

  otel:
    enabled: true
    service_name: workspace-qdrant-mcp
    service_version: "1.0.0"
    trace_context: true
    span_context: true
```

## Testing Strategy

### Unit Testing Framework

```python
# common/logging/testing.py
from typing import List, Dict, Any
from unittest.mock import MagicMock
import json

class LogCapture:
    """Capture log events for testing."""

    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.mock_logger = MagicMock()
        self.mock_logger.info.side_effect = self._capture_info
        self.mock_logger.error.side_effect = self._capture_error
        # ... other levels

    def _capture_info(self, msg: str, **kwargs):
        self.events.append({
            "level": "INFO",
            "message": msg,
            **kwargs
        })

    def _capture_error(self, msg: str, **kwargs):
        self.events.append({
            "level": "ERROR",
            "message": msg,
            **kwargs
        })

    def assert_logged(self, level: str, message: str, **kwargs):
        """Assert that a specific log event was captured."""
        for event in self.events:
            if (event.get("level") == level and
                event.get("message") == message and
                all(event.get(k) == v for k, v in kwargs.items())):
                return True
        raise AssertionError(f"Expected log event not found: {level} {message} {kwargs}")

    def clear(self):
        """Clear captured events."""
        self.events.clear()

class MCPLoggingTest:
    """Test MCP stdio mode logging behavior."""

    def test_mcp_stdio_mode_suppression(self):
        """Test that console output is suppressed in MCP stdio mode."""
        import os
        import sys
        from io import StringIO

        # Set MCP stdio mode
        os.environ["WQM_STDIO_MODE"] = "true"
        os.environ["MCP_QUIET_MODE"] = "true"

        # Capture stdout/stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()

        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            logger = get_logger("test")
            logger.info("Test message")

        # Assert no output to stdout (MCP protocol stream)
        assert stdout_capture.getvalue() == ""

        # May have output to stderr or file, but not stdout
        # Reset environment
        del os.environ["WQM_STDIO_MODE"]
        del os.environ["MCP_QUIET_MODE"]
```

## Performance Specifications

### Performance Requirements

- **Throughput**: > 10,000 log events/second
- **Memory**: < 100MB baseline memory usage
- **Latency**: < 1ms per log event (async mode)
- **CPU**: < 5% CPU overhead under normal load

### Optimization Features

1. **Lazy Evaluation**: Expensive operations only evaluated if logging level matches
2. **Async Batching**: Batch log events for high-throughput scenarios
3. **Context Caching**: Cache bound loggers for repeated context
4. **Processor Optimization**: Optimized processor pipeline for structlog

## Success Metrics

### Functional Requirements

- ✅ MCP stdio mode works without console interference
- ✅ CLI users get rich, colorized console output
- ✅ All logging flows through single configuration system
- ✅ Backward compatibility maintained for existing code
- ✅ Performance matches or exceeds current implementation

### Technical Validation

- ✅ Claude Desktop integration testing passes
- ✅ All existing tests continue to pass
- ✅ Performance benchmarks meet requirements
- ✅ Configuration validation works across deployment scenarios

---

**Architecture Design Complete**: Ready for implementation (Task 209)