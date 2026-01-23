#!/usr/bin/env python3
"""
CLI wrapper that sets environment variables before importing any modules.

This wrapper ensures that WQM_CLI_MODE and WQM_LOG_INIT are set before
any module imports occur, preventing server module initialization and
logging during CLI startup.

Task 215: Migrated to unified logging system for MCP stdio compliance.

DEPRECATED: This Python CLI is deprecated in favor of the Rust CLI.
Task 439: CLI consolidation - Python CLI will be removed in v1.0.0.
Use the Rust 'wqm' binary instead for better performance and daemon integration.
See: docs/CLI_MIGRATION.md for migration guide.
"""

import os
import sys
import warnings

# Set environment variables IMMEDIATELY before any other imports
os.environ.setdefault("WQM_CLI_MODE", "true")
os.environ.setdefault("WQM_LOG_INIT", "false")

# Task 221: Use loguru-based logging system for CLI
# Configure loguru with CLI-appropriate settings
from common.logging.loguru_config import setup_logging

# Configure loguru to be silent in CLI mode but allow file logging for debug
setup_logging(
    log_file=None,  # No file logging by default in CLI mode
    verbose=False,  # No console output in CLI mode
)

# Deprecation warning - shown once per session
_DEPRECATION_MSG = """
================================================================================
DEPRECATION WARNING: Python CLI is deprecated and will be removed in v1.0.0.

Please use the Rust CLI instead:
  - Better performance (<100ms startup)
  - Native daemon integration via gRPC
  - Cross-platform binary available

Most commands have direct equivalents:
  wqm-py service status  →  wqm service status
  wqm-py admin health    →  wqm admin health
  wqm-py library list    →  wqm library list

For full migration guide, see: docs/CLI_MIGRATION.md

To suppress this warning, set WQM_NO_DEPRECATION_WARNING=1
================================================================================
"""


def _show_deprecation_warning():
    """Show deprecation warning unless suppressed."""
    if os.environ.get("WQM_NO_DEPRECATION_WARNING", "").lower() not in ("1", "true", "yes"):
        print(_DEPRECATION_MSG, file=sys.stderr)


def main():
    """Main entry point for wqm CLI.

    Task 215: Enhanced with unified logging system.
    Task 439: Added deprecation warning for Python CLI.
    """
    # Show deprecation warning
    _show_deprecation_warning()

    # Import the actual CLI app only after environment is set
    from .cli.main import app

    # Run the CLI app
    app()

if __name__ == "__main__":
    main()
