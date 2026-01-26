"""
Simple loguru configuration for MCP stdio mode compatibility.

Minimal setup that respects MCP protocol requirements.
Uses OS-standard log directories following platform conventions.

Includes automatic sanitization of sensitive data (passwords, API keys,
tokens, secrets) to prevent credential leaks in log files.
"""

import os
import sys
import pathlib

from loguru import logger

from ..utils.os_directories import OSDirectories
from ..utils.log_sanitizer import LogSanitizer, SanitizationLevel

# Global sanitizer instance for log filtering
_sanitizer = LogSanitizer(level=SanitizationLevel.STANDARD)


def _sanitize_log_record(record):
    """Filter function to sanitize log messages before output.

    Automatically redacts sensitive information like passwords, API keys,
    tokens, and other credentials from log messages.

    Args:
        record: Loguru log record

    Returns:
        Sanitized log record
    """
    # Sanitize the message
    if isinstance(record["message"], str):
        record["message"] = _sanitizer.sanitize_string(record["message"])

    # Sanitize extra fields if present
    if "extra" in record and record["extra"]:
        for key, value in record["extra"].items():
            if isinstance(value, str):
                record["extra"][key] = _sanitizer.sanitize_string(value)

    return record


def setup_logging(log_file: str = None, verbose: bool = False):
    """Setup loguru with minimal configuration using OS-standard log directory.

    Args:
        log_file: Optional custom file path for logging. If None, uses OS-standard log directory.
                 For backward compatibility only - prefer OS-standard location.
        verbose: Enable console output (disabled in MCP stdio mode)
    """
    # Remove default handler
    logger.remove()

    # Add file handler using OS-standard log directory
    if log_file is None:
        # Use OS-standard log directory
        os_dirs = OSDirectories()
        os_dirs.ensure_directories()
        log_path = os_dirs.get_log_file("workspace.log")
        logger.info(f"Using OS-standard log directory: {log_path}")
    else:
        # Legacy mode: use custom path (for backward compatibility)
        log_path = pathlib.Path(log_file)
        logger.warning(
            f"Using legacy log path: {log_file}. Consider migrating to OS-standard location."
        )

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(
            "Failed to create log directory %s: %s", log_path.parent, exc
        )
    logger.add(
        log_path,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        filter=_sanitize_log_record  # Automatic credential sanitization
    )

    # Add console handler only if not in MCP stdio mode and verbose requested
    if verbose and not _is_mcp_stdio_mode():
        logger.add(
            sys.stderr,  # Never use stdout (reserved for MCP JSON-RPC)
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            filter=_sanitize_log_record  # Automatic credential sanitization
        )


def _is_mcp_stdio_mode() -> bool:
    """Check if running in MCP stdio mode."""
    return (
        os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
        os.getenv("MCP_QUIET_MODE", "").lower() == "true" or
        (not sys.stdout.isatty() and not sys.stdin.isatty() and not os.getenv("TERM"))
    )
