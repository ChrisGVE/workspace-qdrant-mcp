"""
Lightweight MCP server implementation for stdio mode.

This module provides a minimal MCP server that avoids problematic imports
and focuses on core protocol compliance for Claude Desktop/Code integration.

IMPORTANT: This module must remain completely independent and not import
from the main server module to avoid import hangs in stdio mode.
"""

import asyncio
import os
import sys
import logging

# CRITICAL: Set up stdio silence before any FastMCP import
_STDIO_MODE = False
if (os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
    "--transport" in sys.argv and "stdio" in sys.argv):
    _STDIO_MODE = True

    # Set environment variables
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress all warnings
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Redirect stderr to null
    _null_device = open(os.devnull, 'w')
    sys.stderr = _null_device

    # Configure logging to be silent
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

from fastmcp import FastMCP


def run_lightweight_stdio_server():
    """Run the lightweight MCP server optimized for stdio mode.

    This function creates a minimal FastMCP server with essential tools only,
    avoiding heavy imports that can cause hangs in stdio mode.
    """

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
            "message": "MCP server running in stdio mode with basic functionality",
            "features": ["basic_status", "echo_test", "minimal_search"]
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
        """Basic search functionality (placeholder).

        Args:
            query: Search query string
            limit: Maximum number of results to return
            collection: Collection name to search in

        Returns:
            Search results with metadata
        """
        return {
            "query": query,
            "results": [],
            "message": "Search functionality requires full server mode with Qdrant connection",
            "total": 0,
            "collection": collection,
            "limit": limit
        }

    @app.tool()
    async def get_server_info() -> dict:
        """Get server information and capabilities."""
        return {
            "name": "workspace-qdrant-mcp",
            "version": "0.1.0",
            "mode": "stdio_lightweight",
            "description": "Lightweight MCP server for stdio mode",
            "capabilities": [
                "workspace_status",
                "echo_test",
                "basic_search_placeholder"
            ],
            "note": "This is a minimal stdio mode server. Full functionality requires HTTP mode."
        }

    try:
        # Run the server in stdio mode
        app.run(transport="stdio")
    except Exception as e:
        # Emergency error handling
        if hasattr(sys, '__stderr__'):
            sys.__stderr__.write(f"Lightweight MCP server error: {e}\n")
        sys.exit(1)


def main():
    """Main entry point for stdio server."""
    # Set up stdio mode environment if not already set
    if "--transport" not in sys.argv:
        sys.argv.extend(["--transport", "stdio"])

    # Set environment variables for stdio mode
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Run the lightweight server
    run_lightweight_stdio_server()


if __name__ == "__main__":
    main()