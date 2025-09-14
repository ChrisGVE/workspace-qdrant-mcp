#!/usr/bin/env python3
"""
CLI wrapper that sets environment variables before importing any modules.

This wrapper ensures that WQM_CLI_MODE and WQM_LOG_INIT are set before
any module imports occur, preventing server module initialization and
logging during CLI startup.

Task 215: Migrated to unified logging system for MCP stdio compliance.
"""

import os
import sys

# Set environment variables IMMEDIATELY before any other imports
os.environ.setdefault("WQM_CLI_MODE", "true")
os.environ.setdefault("WQM_LOG_INIT", "false")

# Task 221: Use loguru-based logging system for CLI
# Configure loguru with CLI-appropriate settings
from common.logging.loguru_config import configure_logging

# Configure loguru to be silent in CLI mode but allow file logging for debug
configure_logging(
    level="CRITICAL",  # Effectively disable console logging
    console_output=False,  # No console output in CLI mode
    json_format=True,
    log_file=None,  # No file logging by default in CLI mode
)

def main():
    """Main entry point for wqm CLI.

    Task 215: Enhanced with unified logging system.
    """
    # Import the actual CLI app only after environment is set
    from .cli.main import app

    # Run the CLI app
    app()

if __name__ == "__main__":
    main()