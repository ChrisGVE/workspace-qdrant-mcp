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

# Task 215: Replace direct logging with unified system
# Disable all logging immediately using unified system approach
from common.logging import configure_unified_logging

# Configure logging to be completely silent in CLI mode
configure_unified_logging(
    level="CRITICAL",
    console_output=False,
    json_format=False,  # CLI doesn't need JSON format
    force_mcp_detection=True  # Treat CLI mode like MCP mode for silence
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