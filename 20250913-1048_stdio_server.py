#!/usr/bin/env python3
"""
Lightweight MCP server entry point optimized for stdio mode.

This version bypasses problematic imports and provides just the essential
MCP functionality needed for Claude Desktop/Code integration.
"""

import asyncio
import os
import sys
import json
import logging

# CRITICAL: Set up complete stdio silence before any imports
if os.getenv("WQM_STDIO_MODE") == "true" or ("--transport" in sys.argv and "stdio" in sys.argv):
    # Set environment variables
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress all warnings
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Redirect stderr to null, keep stdout for MCP protocol
    _null_device = open(os.devnull, 'w')
    sys.stderr = _null_device

    # Configure Python logging to be silent
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    class _NullHandler(logging.Handler):
        def emit(self, record): pass
        def handle(self, record): return True
        def createLock(self): self.lock = None

    root_logger.addHandler(_NullHandler())
    root_logger.setLevel(logging.CRITICAL + 1)
    root_logger.disabled = True

# Import essential components only
from fastmcp import FastMCP

# Create the minimal MCP app
app = FastMCP("workspace-qdrant-mcp")

# Add essential tools
@app.tool()
async def workspace_status() -> dict:
    """Get basic workspace status information."""
    return {
        "status": "active",
        "mode": "stdio",
        "workspace": os.getcwd(),
        "message": "MCP server running in stdio mode with basic functionality"
    }

@app.tool()
async def echo_test(message: str) -> str:
    """Echo test message for MCP protocol validation."""
    return f"Echo: {message}"

# Basic search tool (minimal implementation)
@app.tool()
async def search_workspace(
    query: str,
    limit: int = 10,
    collection: str = "default"
) -> dict:
    """Basic search functionality (placeholder)."""
    return {
        "query": query,
        "results": [],
        "message": "Search functionality requires full server mode",
        "total": 0
    }

def run_stdio_server():
    """Run the lightweight MCP server in stdio mode."""
    try:
        app.run(transport="stdio")
    except Exception as e:
        # Emergency error handling - write to original stderr if possible
        if hasattr(sys, '__stderr__'):
            sys.__stderr__.write(f"MCP server error: {e}\n")
        sys.exit(1)
    finally:
        # Clean up
        if '_null_device' in globals() and not _null_device.closed:
            _null_device.close()

if __name__ == "__main__":
    run_stdio_server()