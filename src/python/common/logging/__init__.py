"""
Unified logging module for workspace-qdrant-mcp.

This module provides a centralized logging system that resolves MCP stdio interference
by automatically detecting MCP mode and suppressing console output when needed.

Key Features:
    - Drop-in replacement for logging.getLogger(__name__)
    - Automatic MCP stdio detection and console suppression
    - Backward compatibility with existing logging patterns
    - Structured logging with JSON formatting
    - Context-aware logging with performance monitoring
    - Environment-based configuration

Usage:
    Replace standard logging imports:

    # OLD:
    import logging
    logger = logging.getLogger(__name__)

    # NEW:
    from common.logging import get_logger
    logger = get_logger(__name__)

    # Advanced usage:
    from common.logging import get_logger, LogContext, PerformanceLogger

    logger = get_logger(__name__)
    perf_logger = PerformanceLogger(logger)

    with LogContext(operation="search", user_id="123"):
        with perf_logger.time_operation("document_search"):
            logger.info("Starting search", query="example")

Environment Variables:
    - WQM_STDIO_MODE: Set to 'true' to enable MCP stdio mode (suppresses console)
    - MCP_QUIET_MODE: Alternative way to suppress console in MCP mode
    - DISABLE_MCP_CONSOLE_LOGS: Explicit console suppression flag
    - LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - LOG_FORMAT: 'json' or 'text' formatting
"""

from .core import StructuredLogger
from .config import LoggingConfig, detect_mcp_mode
from .migration import get_logger, LogContext, PerformanceLogger, configure_unified_logging


# Re-export key classes and functions for backward compatibility
__all__ = [
    "get_logger",
    "configure_unified_logging",
    "StructuredLogger",
    "LoggingConfig",
    "detect_mcp_mode",
    "LogContext",
    "PerformanceLogger",
]

# Initialize logging configuration from environment on import
# This ensures MCP stdio detection happens early
from .migration import _initialize_logging_if_needed
_initialize_logging_if_needed()