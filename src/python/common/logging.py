"""
Unified logging system bridge for workspace-qdrant-mcp.

This module provides a bridge to the actual unified logging system in
common.observability.logger with additional MCP-specific functionality.

This is the centralized entry point for all logging throughout the application,
ensuring consistent MCP stdio mode compliance and structured logging.

Task 215: This module replaces all direct logging.getLogger() calls
with unified logging system for complete MCP stdio compliance.

Example:
    Replace this:
        import logging
        logger = logging.getLogger(__name__)

    With this:
        from common.logging import get_logger
        logger = get_logger(__name__)
"""

import os
from typing import Optional, Union
from pathlib import Path

# Import the actual unified logging implementation
from common.observability.logger import (
    get_logger as _get_logger,
    LogContext,
    PerformanceLogger,
    configure_logging,
    StructuredLogger,
)

# Re-export the main functions
get_logger = _get_logger
__all__ = [
    'get_logger',
    'LogContext',
    'PerformanceLogger',
    'configure_unified_logging',
    'StructuredLogger',
]


def configure_unified_logging(
    level: Union[str, int] = "INFO",
    json_format: bool = True,
    log_file: Optional[Path] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    console_output: bool = True,
    force_mcp_detection: bool = False,
) -> None:
    """Configure unified logging with MCP-specific enhancements.

    This is a wrapper around configure_logging that adds MCP-specific
    detection and configuration for stdio mode compliance.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting for structured logs
        log_file: Path to log file (optional, logs to console if None)
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        console_output: Whether to output logs to console
        force_mcp_detection: Force MCP stdio mode detection regardless of environment

    Task 215: This function ensures complete silence in MCP stdio mode
    while maintaining full functionality in other modes.
    """
    # Enhanced MCP stdio mode detection
    stdio_mode = (
        force_mcp_detection or
        os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
        os.getenv("MCP_QUIET_MODE", "").lower() == "true"
    )

    # In stdio mode, override console output settings for protocol compliance
    if stdio_mode:
        console_output = False
        level = "CRITICAL"  # Effectively disable all logging

    # Call the actual configure_logging function with stdio detection
    configure_logging(
        level=level,
        json_format=json_format,
        log_file=log_file,
        max_file_size=max_file_size,
        backup_count=backup_count,
        console_output=console_output,
        stdio_mode=stdio_mode,
    )


def is_stdio_mode() -> bool:
    """Check if we're running in MCP stdio mode.

    Returns:
        True if in MCP stdio mode (should suppress console output)
    """
    return (
        os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
        os.getenv("MCP_QUIET_MODE", "").lower() == "true"
    )


def safe_log_error(message: str, **kwargs) -> None:
    """Safely log an error with stdio mode detection.

    This function can be used to log critical errors even in stdio mode
    by writing to a file instead of console.

    Args:
        message: Error message to log
        **kwargs: Additional structured fields
    """
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