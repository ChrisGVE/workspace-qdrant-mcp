"""
Lightweight MCP server implementation for stdio mode.

This module provides a minimal MCP server that avoids problematic imports
and focuses on core protocol compliance for Claude Desktop/Code integration.

IMPORTANT: This module must remain completely independent and not import
from the main server module to avoid import hangs in stdio mode.

Task 215: Migrated to unified logging system for complete MCP stdio compliance.
"""

import os
import sys

# CRITICAL: Set up stdio silence IMMEDIATELY before any imports
_STDIO_MODE = False
if (os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
    "--transport" in sys.argv and "stdio" in sys.argv or
    (len(sys.argv) > 1 and "--transport" not in sys.argv)):  # Default stdio mode
    _STDIO_MODE = True

    # Set environment variables FIRST
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_TRACE"] = ""

    # Redirect stderr to null IMMEDIATELY
    _null_device = open(os.devnull, 'w')
    sys.stderr = _null_device

    # Suppress all warnings immediately
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Suppress Pydantic deprecation warnings specifically
    try:
        from pydantic.warnings import PydanticDeprecatedSince20
        warnings.filterwarnings('ignore', category=PydanticDeprecatedSince20)
    except ImportError:
        pass

# Now safe to import other modules
import asyncio
import logging

# Import unified logging system after stdio setup
from common.logging.loguru_config import get_logger, safe_log_error

# Set up unified logging with stdio mode detection
logger = get_logger(__name__)

# Configure logging to be silent in stdio mode
if _STDIO_MODE:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    class _NullHandler(logging.Handler):
        def emit(self, record): pass
        def handle(self, record): return True
        def createLock(self): self.lock = None

    root_logger.addHandler(_NullHandler())
    root_logger.setLevel(logging.CRITICAL)
    root_logger.disabled = True

    # Configure third-party library loggers to be completely silent
    _THIRD_PARTY_LOGGERS = [
        'httpx', 'httpcore', 'qdrant_client', 'urllib3', 'asyncio',
        'fastembed', 'grpc', 'structlog', 'typer', 'pydantic', 'rich',
        'uvicorn', 'fastapi', 'watchdog', 'watchfiles', 'aiohttp',
        'aiofiles', 'requests', 'chardet', 'pypdf', 'python_docx',
        'python_pptx', 'ebooklib', 'beautifulsoup4', 'lxml',
        'markdown', 'pygments', 'psutil', 'tqdm', 'multipart'
    ]

    for logger_name in _THIRD_PARTY_LOGGERS:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.handlers.clear()
        third_party_logger.addHandler(_NullHandler())
        third_party_logger.setLevel(logging.CRITICAL)
        third_party_logger.disabled = True
        third_party_logger.propagate = False

# Configure Rich console to use null device in stdio mode
if _STDIO_MODE:
    try:
        from rich.console import Console
        # Override Rich console creation to use devnull
        original_console_init = Console.__init__

        def _null_console_init(self, **kwargs):
            kwargs['file'] = _null_device
            kwargs['stderr'] = False
            original_console_init(self, **kwargs)

        Console.__init__ = _null_console_init
    except ImportError:
        # Rich not available, skip
        pass

# Structlog configuration removed - now using loguru via common.logging.loguru_config

# Configure Typer to redirect stderr in stdio mode
if _STDIO_MODE:
    try:
        import typer
        # Override typer's echo functions to use null device
        original_echo = typer.echo
        original_secho = typer.secho

        def _null_echo(*args, **kwargs):
            kwargs['file'] = _null_device
            return original_echo(*args, **kwargs)

        def _null_secho(*args, **kwargs):
            kwargs['file'] = _null_device
            return original_secho(*args, **kwargs)

        typer.echo = _null_echo
        typer.secho = _null_secho
    except ImportError:
        # Typer not available, skip
        pass

from fastmcp import FastMCP


def run_lightweight_stdio_server():
    """Run the lightweight MCP server optimized for stdio mode.

    This function creates a minimal FastMCP server with essential tools only,
    avoiding heavy imports that can cause hangs in stdio mode.

    Task 215: Enhanced with unified logging system for proper error handling.
    """

    # Logging suppressed in stdio mode for MCP compliance

    # Create the minimal MCP app
    app = FastMCP("workspace-qdrant-mcp")

    # Add essential tools
    @app.tool()
    async def workspace_status() -> dict:
        """Get basic workspace status information."""
        # Debug logging suppressed in stdio mode
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
        # Debug logging suppressed in stdio mode
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
        # Debug logging suppressed in stdio mode
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
        # Debug logging suppressed in stdio mode
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
        # Logging suppressed in stdio mode for MCP compliance
        # Run the server in stdio mode
        app.run(transport="stdio")
    except Exception as e:
        # Task 215: Replace sys.__stderr__.write() with unified logging
        # Use safe_log_error which handles stdio mode detection automatically
        safe_log_error(
            "Lightweight MCP server error",
            error_type=type(e).__name__,
            error_message=str(e),
            server_mode="stdio_lightweight"
        )
        logger.exception("MCP server failed", error=str(e))
        sys.exit(1)


def main():
    """Main entry point for stdio server.

    Task 215: Enhanced with unified logging system.
    """
    # Debug logging suppressed in stdio mode

    # Set up stdio mode environment if not already set
    if "--transport" not in sys.argv:
        sys.argv.extend(["--transport", "stdio"])

    # Set environment variables for stdio mode
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_TRACE"] = ""

    # Logging suppressed in stdio mode for MCP compliance

    # Run the lightweight server
    run_lightweight_stdio_server()


if __name__ == "__main__":
    main()