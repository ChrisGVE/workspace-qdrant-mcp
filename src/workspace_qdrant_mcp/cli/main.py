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

import os
import sys

# Suppress warnings for version-only calls before any imports
if len(sys.argv) >= 2 and (sys.argv[1] == "--version" or sys.argv[1] == "-v"):
    verbose_flag = "--verbose" in sys.argv or "--debug" in sys.argv
    if not verbose_flag:
        os.environ["PYTHONWARNINGS"] = "ignore"
        import warnings
        warnings.filterwarnings('ignore')

import asyncio
import warnings
from pathlib import Path
from typing import Optional

# Handle version flag early to avoid heavy imports
if len(sys.argv) >= 2 and (sys.argv[1] == "--version" or sys.argv[1] == "-v"):
    verbose_flag = "--verbose" in sys.argv or "--debug" in sys.argv
    
    # Get version from __init__.py without importing heavy modules
    version_str = "0.2.0"  # Default fallback
    try:
        # Read version from __init__.py file directly to avoid imports
        init_file = Path(__file__).parent.parent / "__init__.py"
        if init_file.exists():
            init_content = init_file.read_text()
            import re
            match = re.search(r'__version__\s*=\s*["\']([^"\'\']*)["\']', init_content)
            if match:
                version_str = match.group(1)
    except Exception:
        pass  # Keep fallback version

    if verbose_flag:
        # Verbose version info with detailed metadata
        print(f"Workspace Qdrant MCP {version_str}")
        print(f"Python {sys.version.split()[0]}")
        print(f"Platform: {sys.platform}")
        print(f"Installation path: {Path(__file__).parent.parent}")
    else:
        # Clean version display - just the version number
        print(version_str)
    
    sys.exit(0)

# Import heavy modules only after version check passes
import typer

from ..observability import get_logger, configure_logging

from .commands.admin import admin_app
# TODO: Fix syntax errors in ingest.py before re-enabling
# from .commands.ingest import ingest_app
from .commands.init import init_app
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
    add_completion=False,  # Use custom init command instead
    no_args_is_help=False,  # Handle this manually to allow version flag
    rich_markup_mode=None,  # Disable Rich formatting completely
)
logger = get_logger(__name__)

# Add subcommand groups
app.add_typer(init_app, name="init", help="Initialize shell completion for wqm")
app.add_typer(memory_app, name="memory", help="Memory rules and LLM behavior management")
app.add_typer(admin_app, name="admin", help="System administration and configuration")
# TODO: Re-enable after fixing ingest.py
# app.add_typer(ingest_app, name="ingest", help="Manual document processing")
app.add_typer(search_app, name="search", help="Command-line search interface")
app.add_typer(library_app, name="library", help="Library collection management")
app.add_typer(watch_app, name="watch", help="Folder watching configuration")
app.add_typer(observability_app, name="observability", help="Observability, monitoring, and health checks")

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version information"),
    verbose: bool = typer.Option(False, "--verbose", help="Show verbose version information"),
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
        wqm init                           # Enable shell completion
    """
    # Handle version flag first, before any configuration loading
    if version:
        # Suppress warnings for clean version output
        if not verbose and not debug:
            import warnings
            warnings.filterwarnings('ignore')
        show_version(verbose=verbose or debug)
        raise typer.Exit()
    
    # If no command is invoked and no version flag, show help
    if not ctx.invoked_subcommand:
        print(ctx.get_help())
        raise typer.Exit()
    
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

    if config_path:
        # TODO: Load custom config
        if debug:
            logger.debug("Custom config path provided", config_path=config_path)
        pass

def show_version(verbose: bool = False) -> None:
    """Display version information.
    
    Args:
        verbose: If True, show detailed version information.
                If False, show only the version number.
    """
    try:
        from workspace_qdrant_mcp import __version__
        version_str = __version__
    except ImportError:
        version_str = "0.2.0"  # Fallback version

    if verbose:
        # Verbose version info with detailed metadata
        print(f"Workspace Qdrant MCP {version_str}")
        print(f"Python {sys.version.split()[0]}")
        print(f"Platform: {sys.platform}")
        print(f"Installation path: {Path(__file__).parent.parent}")
    else:
        # Clean version display - just the version number
        print(version_str)


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
