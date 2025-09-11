#!/usr/bin/env python3
"""
CLI wrapper that sets environment variables before importing any modules.

This wrapper ensures that WQM_CLI_MODE and WQM_LOG_INIT are set before
any module imports occur, preventing server module initialization and
logging during CLI startup.
"""

import os
import sys

# Set environment variables IMMEDIATELY before any other imports
os.environ.setdefault("WQM_CLI_MODE", "true")
os.environ.setdefault("WQM_LOG_INIT", "false")

# Disable all logging immediately
import logging
logging.disable(logging.CRITICAL)

def main():
    """Main entry point for wqm CLI."""
    # Import the actual CLI app only after environment is set
    from .cli.main import app
    
    # Run the CLI app
    app()

if __name__ == "__main__":
    main()