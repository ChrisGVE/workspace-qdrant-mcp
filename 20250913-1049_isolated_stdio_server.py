#!/usr/bin/env python3
"""
Completely isolated stdio server for MCP protocol compliance testing.

This server is completely independent and doesn't import anything from the
main workspace_qdrant_mcp package to avoid any import issues.
"""

import asyncio
import os
import sys
import logging

# CRITICAL: Set up complete stdio silence before ANY imports
if (os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
    "--transport" in sys.argv and "stdio" in sys.argv or
    len(sys.argv) == 1):  # Default to stdio if no args

    # Set environment variables
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress ALL warnings globally
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Redirect stderr to null device
    _null_device = open(os.devnull, 'w')
    sys.stderr = _null_device

    # Configure logging to be completely silent
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

    # Also disable specific loggers that might interfere
    for logger_name in ['fastmcp', 'uvicorn', 'httpx', 'httpcore', 'qdrant_client']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        logger.propagate = False

# Now import FastMCP
from fastmcp import FastMCP

# Create the minimal MCP app
app = FastMCP("workspace-qdrant-mcp")

# Add essential tools for MCP protocol testing
@app.tool()
async def workspace_status() -> dict:
    """Get basic workspace status information."""
    return {
        "status": "active",
        "mode": "stdio_isolated",
        "workspace": os.getcwd(),
        "server_type": "lightweight_stdio",
        "message": "Isolated MCP server running with minimal functionality",
        "features": ["workspace_status", "echo_test", "search_placeholder"]
    }

@app.tool()
async def echo_test(message: str) -> str:
    """Echo a test message for MCP protocol validation."""
    return f"Echo: {message}"

@app.tool()
async def search_workspace(
    query: str,
    limit: int = 10,
    collection: str = "default"
) -> dict:
    """Placeholder search functionality."""
    return {
        "query": query,
        "results": [],
        "message": "This is an isolated stdio server - search functionality not available",
        "total": 0,
        "collection": collection,
        "limit": limit,
        "server_mode": "isolated_stdio"
    }

def main():
    """Main entry point for the isolated stdio server."""
    try:
        # Run the server
        app.run(transport="stdio")
    except Exception as e:
        # Emergency error handling - write to original stderr if possible
        if hasattr(sys, '__stderr__'):
            sys.__stderr__.write(f"Isolated MCP server error: {e}\n")
        sys.exit(1)
    finally:
        # Clean up
        if '_null_device' in globals() and not _null_device.closed:
            _null_device.close()

if __name__ == "__main__":
    main()