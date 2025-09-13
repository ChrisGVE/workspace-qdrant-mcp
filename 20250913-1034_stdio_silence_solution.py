#!/usr/bin/env python3
"""
Comprehensive solution to eliminate ALL console output in MCP stdio mode.

This script implements a complete silence solution that:
1. Completely suppresses all Python logging to stdout/stderr in stdio mode
2. Suppresses ALL third-party library warnings
3. Redirects any system-level output to /dev/null
4. Provides emergency fallbacks for edge cases

Critical for MCP protocol compliance - stdio mode must only output clean JSON-RPC.
"""

import contextlib
import io
import logging
import os
import sys
import warnings
from typing import Optional

# Global state for stdio mode detection
_STDIO_MODE_DETECTED = False
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None
_NULL_DEVICE = None


def detect_stdio_mode() -> bool:
    """Detect if we're running in MCP stdio mode with comprehensive checks."""
    # Check explicit environment variables
    if os.getenv("WQM_STDIO_MODE", "").lower() == "true":
        return True
    if os.getenv("MCP_QUIET_MODE", "").lower() == "true":
        return True
    if os.getenv("DISABLE_MCP_CONSOLE_LOGS", "").lower() == "true":
        return True
    if os.getenv("MCP_TRANSPORT") == "stdio":
        return True

    # Check command line arguments for stdio transport
    if "--transport" in sys.argv:
        try:
            transport_idx = sys.argv.index("--transport")
            if transport_idx + 1 < len(sys.argv):
                if sys.argv[transport_idx + 1] == "stdio":
                    return True
        except (ValueError, IndexError):
            pass

    # Check if stdout appears to be piped (MCP scenario)
    if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
        if os.getenv("TERM") is None:  # No terminal environment
            return True

    return False


def setup_complete_silence():
    """Set up complete silence for MCP stdio mode."""
    global _STDIO_MODE_DETECTED, _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _NULL_DEVICE

    _STDIO_MODE_DETECTED = detect_stdio_mode()

    if not _STDIO_MODE_DETECTED:
        return

    # Store originals for potential restoration
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr

    # Create null device
    _NULL_DEVICE = open(os.devnull, 'w')

    # Redirect all output to null
    sys.stdout = _NULL_DEVICE
    sys.stderr = _NULL_DEVICE

    # Suppress ALL warnings globally
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Suppress specific third-party warnings that bypass the filter
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic_core")
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*PydanticDeprecatedSince20.*")
    warnings.filterwarnings("ignore", message=".*got forked.*parallelism.*")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Set tokenizer parallelism to false to avoid warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Configure Python logging root logger to be completely silent
    root_logger = logging.getLogger()

    # Remove all existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add null handler to prevent fallback to stderr
    null_handler = NullHandler()
    root_logger.addHandler(null_handler)
    root_logger.setLevel(logging.CRITICAL + 1)  # Higher than any normal level

    # Suppress specific third-party loggers
    third_party_loggers = [
        'httpx', 'httpcore', 'urllib3', 'requests',
        'qdrant_client', 'fastmcp', 'uvicorn', 'fastapi',
        'pydantic', 'transformers', 'huggingface_hub',
        'sentence_transformers', 'torch', 'tensorflow'
    ]

    for logger_name in third_party_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        logger.propagate = False

    # Disable root logger propagation to prevent any leakage
    root_logger.disabled = True


class NullHandler(logging.Handler):
    """Handler that completely discards all log records."""

    def emit(self, record):
        """Discard the record silently."""
        pass

    def handle(self, record):
        """Handle by discarding."""
        return True

    def createLock(self):
        """No locking needed."""
        self.lock = None


class StdioSilencer:
    """Context manager to temporarily silence all output."""

    def __init__(self):
        self.original_stdout = None
        self.original_stderr = None
        self.null_device = None

    def __enter__(self):
        if _STDIO_MODE_DETECTED:
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            self.null_device = open(os.devnull, 'w')
            sys.stdout = self.null_device
            sys.stderr = self.null_device
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.null_device:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.null_device.close()


def restore_output():
    """Restore original output streams (for testing/debugging only)."""
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR, _NULL_DEVICE

    if _NULL_DEVICE:
        sys.stdout = _ORIGINAL_STDOUT
        sys.stderr = _ORIGINAL_STDERR
        _NULL_DEVICE.close()
        _NULL_DEVICE = None


def silence_third_party_imports():
    """Silence output during third-party imports."""
    if not _STDIO_MODE_DETECTED:
        return

    # This can be called around problematic imports
    with StdioSilencer():
        # Imports that might produce output can be wrapped
        pass


def is_stdio_mode() -> bool:
    """Check if stdio mode is active."""
    return _STDIO_MODE_DETECTED


# Set up silence as early as possible when this module is imported
setup_complete_silence()


if __name__ == "__main__":
    # Test the silencer
    print("If you see this, stdio mode was not detected")

    # Test with manual activation
    os.environ["WQM_STDIO_MODE"] = "true"
    setup_complete_silence()
    print("You should not see this message")

    # Test logging
    logger = logging.getLogger(__name__)
    logger.error("You should not see this log message")

    # Test warnings
    warnings.warn("You should not see this warning")

    print("Test complete")