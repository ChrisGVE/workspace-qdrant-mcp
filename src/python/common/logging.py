"""
Legacy logging bridge for workspace-qdrant-mcp.

This module provides backward compatibility for any remaining imports
from common.logging after the loguru migration. All functionality now
routes through loguru_config.

DEPRECATED: Direct imports from this module are deprecated.
Use 'from common.logging.loguru_config import get_logger' instead.

Task 222: This is a compatibility bridge during final cleanup phase.
"""

import os
import warnings
from typing import Optional, Union
from pathlib import Path

# Import from the new loguru-based system
from common.logging.loguru_config import (
    get_logger as _get_logger,
    configure_logging,
)

# Legacy compatibility - emit deprecation warning
def get_logger(name: str):
    """Legacy get_logger function with deprecation warning."""
    warnings.warn(
        "Importing get_logger from 'common.logging' is deprecated. "
        "Use 'from common.logging.loguru_config import get_logger' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_logger(name)

# Stub classes for backward compatibility
class LogContext:
    """Deprecated LogContext stub for backward compatibility."""
    def __init__(self, **kwargs):
        warnings.warn(
            "LogContext is deprecated with the loguru migration. "
            "Use loguru's contextualization features instead.",
            DeprecationWarning,
            stacklevel=2
        )

class PerformanceLogger:
    """Deprecated PerformanceLogger stub for backward compatibility."""
    def __init__(self, **kwargs):
        warnings.warn(
            "PerformanceLogger is deprecated with the loguru migration. "
            "Use loguru's timing features instead.",
            DeprecationWarning,
            stacklevel=2
        )

class StructuredLogger:
    """Deprecated StructuredLogger stub for backward compatibility."""
    def __init__(self, **kwargs):
        warnings.warn(
            "StructuredLogger is deprecated with the loguru migration. "
            "Use loguru directly instead.",
            DeprecationWarning,
            stacklevel=2
        )

# Re-export for compatibility
__all__ = [
    'get_logger',
    'LogContext',
    'PerformanceLogger',
    'configure_unified_logging',
    'StructuredLogger',
    'safe_log_error',
]

def configure_unified_logging(**kwargs) -> None:
    """Legacy configure_unified_logging with deprecation warning."""
    warnings.warn(
        "configure_unified_logging is deprecated. "
        "Use 'from common.logging.loguru_config import configure_logging' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return configure_logging(**kwargs)

def is_stdio_mode() -> bool:
    """Check if we're running in MCP stdio mode."""
    return (
        os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
        os.getenv("MCP_QUIET_MODE", "").lower() == "true"
    )

def safe_log_error(message: str, **kwargs) -> None:
    """Safely log an error with stdio mode detection."""
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
        logger = _get_logger("common.logging")
        logger.error(message, **kwargs)