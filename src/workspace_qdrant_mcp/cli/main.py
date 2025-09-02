"""Unified CLI interface for workspace-qdrant-mcp v2.0.

This module implements the complete `wqm` command structure with all
subcommands and features specified in the PRD v2.0.

Usage:
    wqm memory list              # List memory rules
    wqm memory add "rule text"   # Add memory rule
    wqm admin status             # Show system status
    wqm search project "query"   # Search workspace
    wqm ingest file document.pdf # Process document
    wqm library create books     # Create library
    wqm watch add ~/docs         # Watch folder
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer

from ..observability import get_logger, configure_logging

from .commands.admin import admin_app
from .commands.ingest import ingest_app
from .commands.library import library_app

# Import command modules
from .commands.memory import memory_app
from .commands.search import search_app
from .commands.watch import watch_app
from .observability import observability_app

# Initialize main app and logger
app = typer.Typer(
    name="wqm",
    help="Workspace Qdrant MCP - Unified semantic workspace management",
    add_completion=True,
    no_args_is_help=False,  # Allow custom welcome message
)
logger = get_logger(__name__)

# Add subcommand groups
app.add_typer(memory_app, name="memory", help="Memory rules and LLM behavior management")
app.add_typer(admin_app, name="admin", help="System administration and configuration")
app.add_typer(ingest_app, name="ingest", help="Manual document processing")
app.add_typer(search_app, name="search", help="Command-line search interface")
app.add_typer(library_app, name="library", help="Library collection management")
app.add_typer(watch_app, name="watch", help="Folder watching configuration")
app.add_typer(observability_app, name="observability", help="Observability, monitoring, and health checks")

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version information"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Custom configuration file"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug mode with verbose logging"),
) -> None:
    """
    Workspace Qdrant MCP - Unified semantic workspace management

    A comprehensive semantic workspace platform with memory-driven LLM behavior,
    automated document ingestion, and advanced search capabilities.

    Examples:
        wqm admin status                    # Check system status
        wqm memory add "Use uv for Python"  # Add memory rule
        wqm ingest file document.pdf        # Ingest a document
        wqm search project "rust patterns"  # Search current project
        wqm library create technical-books  # Create library collection
        wqm watch add ~/docs --collection=_docs  # Watch folder
    """
    # Configure logging and environment based on debug flag
    import os
    if debug:
        # Enable verbose logging and initialization messages in debug mode
        os.environ["WQM_LOG_INIT"] = "true"
        configure_logging(level="DEBUG", json_format=True, console_output=True)
        logger.debug("Debug mode enabled")
    else:
        # Disable initialization logging and reduce verbosity for CLI usage
        os.environ["WQM_LOG_INIT"] = "false"
        configure_logging(level="ERROR", json_format=False, console_output=False)
    
    if version:
        show_version(debug=debug)
        return

    if config_path:
        # TODO: Load custom config
        if debug:
            logger.debug("Custom config path provided", config_path=config_path)
        pass

    if ctx.invoked_subcommand is None:
        show_welcome(debug=debug)

def show_version(debug: bool = False) -> None:
    """Display version information."""
    try:
        from workspace_qdrant_mcp import __version__
        version_str = __version__
    except ImportError:
        version_str = "0.2.0"  # Fallback version

    if debug:
        # Verbose version info in debug mode
        print(f"Workspace Qdrant MCP {version_str}")
        print(f"Python {sys.version.split()[0]}")
        print(f"Platform: {sys.platform}")
    else:
        # Simple version display
        print(version_str)

def show_welcome(debug: bool = False) -> None:
    """Display welcome message and quick start guide."""
    if debug:
        logger.debug("Displaying welcome message")
    
    print("Workspace Qdrant MCP - Unified semantic workspace management")
    print("")
    print("Available commands:")
    print("  wqm admin status                   # Check system health")
    print("  wqm memory add \"Use uv for Python\"  # Add behavior rule")
    print("  wqm ingest file document.pdf       # Process a document")
    print("  wqm search project \"patterns\"      # Search your project")
    print("  wqm library create technical-books # Create library collection")
    print("  wqm watch add ~/docs               # Watch folder")
    print("")
    print("Use 'wqm COMMAND --help' for detailed information.")
    print("Use 'wqm --debug' for verbose debugging output.")

def handle_async_command(coro, debug: bool = False):
    """Helper to run async commands in CLI context."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        if debug:
            logger.warning("Operation cancelled by user")
        print("\nOperation cancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        if debug:
            logger.error("CLI operation failed", error=str(e), exc_info=True)
            print(f"Error: {e}")
            print(f"Exception type: {type(e).__name__}")
        else:
            print(f"Error: {e}")
        raise typer.Exit(1)

# Make cli available for backward compatibility
cli = app

if __name__ == "__main__":
    app()
