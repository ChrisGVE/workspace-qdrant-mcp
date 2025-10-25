"""
Simple loguru configuration for MCP stdio mode compatibility.

Minimal setup that respects MCP protocol requirements.
Uses OS-standard log directories following platform conventions.
"""

import os
import sys
from pathlib import Path

from loguru import logger

from ..utils.os_directories import OSDirectories


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
        log_path = Path(log_file)
        logger.warning(f"Using legacy log path: {log_path}. Consider migrating to OS-standard location.")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        rotation="10 MB",
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"
    )

    # Add console handler only if not in MCP stdio mode and verbose requested
    if verbose and not _is_mcp_stdio_mode():
        logger.add(
            sys.stderr,  # Never use stdout (reserved for MCP JSON-RPC)
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )


def _is_mcp_stdio_mode() -> bool:
    """Check if running in MCP stdio mode."""
    return (
        os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
        os.getenv("MCP_QUIET_MODE", "").lower() == "true" or
        (not sys.stdout.isatty() and not sys.stdin.isatty() and not os.getenv("TERM"))
    )
