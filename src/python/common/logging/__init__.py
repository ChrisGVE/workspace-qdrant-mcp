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
import os
from pathlib import Path


def safe_log_error(message: str, **kwargs) -> None:
    """Safely log an error with stdio mode detection.

    This function can be used to log critical errors even in stdio mode
    by writing to a file instead of console.

    Args:
        message: Error message to log
        **kwargs: Additional structured fields
    """
    def is_stdio_mode() -> bool:
        """Check if we're running in MCP stdio mode."""
        return (
            os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
            os.getenv("MCP_QUIET_MODE", "").lower() == "true"
        )

    if is_stdio_mode():
        # In stdio mode, attempt to log to file only
        try:
            import tempfile
            import json
            from datetime import datetime, timezone

            log_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'level': 'ERROR',
                'message': message,
                'mode': 'stdio_safe_log',
                **kwargs
            }

            # Write to temporary log file
            log_dir = Path(tempfile.gettempdir()) / "workspace-qdrant-mcp"
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "stdio_errors.log"

            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception:
            # If even file logging fails, silently continue (stdio mode requirement)
            pass
    else:
        # Normal mode - use regular logger
        logger = get_logger("common.logging")
        logger.error(message, **kwargs)


# Re-export key classes and functions for backward compatibility
__all__ = [
    "get_logger",
    "configure_unified_logging",
    "StructuredLogger",
    "LoggingConfig",
    "detect_mcp_mode",
    "LogContext",
    "PerformanceLogger",
    "safe_log_error",
]

# Initialize logging configuration from environment on import
# This ensures MCP stdio detection happens early
from .migration import _initialize_logging_if_needed
_initialize_logging_if_needed()