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
    wqm grammar list             # List tree-sitter grammars
    wqm grammar install python   # Install grammar
    wqm watch add ~/docs         # Watch folder
    wqm messages list            # View error messages
    wqm queue status             # Monitor queue status
    wqm tools status             # Check tool availability
    wqm collections list-types   # View collections by type
    wqm errors stats             # Error statistics
    wqm branch list --project ID # List branches in project
    wqm migrate detect           # Detect old collections
    wqm backup create /path      # Create system backup
    wqm backup info /path        # Show backup information
"""

# Disable all console logging immediately for CLI usage
import logging
import os
import sys

logging.disable(logging.CRITICAL)

# Suppress observability initialization logging for CLI usage
os.environ.setdefault("WQM_LOG_INIT", "false")
# Set CLI mode to prevent server imports in __init__.py
os.environ.setdefault("WQM_CLI_MODE", "true")

# Suppress warnings for version-only calls before any imports
if len(sys.argv) >= 2 and (sys.argv[1] == "--version" or sys.argv[1] == "-v"):
    verbose_flag = "--verbose" in sys.argv or "--debug" in sys.argv
    if not verbose_flag:
        os.environ["PYTHONWARNINGS"] = "ignore"
        import warnings

        warnings.filterwarnings("ignore")

import asyncio
import warnings
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

# Handle version flag early to avoid heavy imports
if len(sys.argv) >= 2 and (sys.argv[1] == "--version" or sys.argv[1] == "-v"):
    verbose_flag = "--verbose" in sys.argv or "--debug" in sys.argv

    # Get version from __init__.py without importing heavy modules
    version_str = "0.3.0"  # Default fallback
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

    # Exit without importing typer for lightweight version check
    sys.exit(0)

# Import heavy modules only after version check passes
import typer
from common.logging.loguru_config import setup_logging
from loguru import logger

from .advanced_features import advanced_features_app
from .commands.admin import admin_app
from .commands.backup import backup_app
from .commands.branch import branch_app
from .commands.collection_types import collection_types_app
from .commands.error_reporting import errors_app
from .commands.grammar import grammar_app
from .commands.ingest import ingest_app
from .commands.init import init_app
from .commands.library import library_app
from .commands.lsp_management import lsp_app

# Import command modules
from .commands.memory import memory_app
from .commands.messages import messages_app
from .commands.migrate import migrate_app
from .commands.project import project_app
from .commands.queue import queue_app
from .commands.search import search_app
from .commands.service import service_app
from .commands.tools import tools_app
from .commands.watch import watch_app
from .error_handling import setup_exception_hook

# Import enhanced CLI features
from .help_system import help_app

# SECURITY: Web UI temporarily disabled due to critical vulnerabilities
# from .commands.web import web_app
from .observability import observability_app
from .status import status_app

# Initialize main app and logger
app = typer.Typer(
    name="wqm",
    help="Workspace Qdrant MCP - Unified semantic workspace management",
    add_completion=False,  # Use custom init command instead
    no_args_is_help=False,  # Handle this manually to allow version flag
    rich_markup_mode=None,  # Disable Rich formatting completely
)
# logger imported from loguru

# Add subcommand groups
app.add_typer(init_app, name="init", help="Initialize shell completion for wqm")
app.add_typer(
    memory_app, name="memory", help="Memory rules and LLM behavior management"
)
app.add_typer(admin_app, name="admin", help="System administration and configuration")
app.add_typer(backup_app, name="backup", help="Backup and restore operations with version validation")
app.add_typer(ingest_app, name="ingest", help="Manual document processing")
app.add_typer(search_app, name="search", help="Command-line search interface")
app.add_typer(library_app, name="library", help="Library collection management")
app.add_typer(lsp_app, name="lsp", help="LSP server management and monitoring")
app.add_typer(grammar_app, name="grammar", help="Tree-sitter grammar management")
app.add_typer(
    service_app, name="service", help="User service management for memexd daemon"
)
app.add_typer(watch_app, name="watch", help="Folder watching configuration")
# SECURITY: Web UI commands temporarily disabled due to critical vulnerabilities
# app.add_typer(
#     web_app, name="web", help="Integrated web UI server with workspace features"
# )
app.add_typer(
    observability_app,
    name="observability",
    help="Observability, monitoring, and health checks",
)
app.add_typer(
    status_app, name="status", help="Processing status and user feedback system"
)
app.add_typer(
    help_app, name="help", help="Interactive help and command discovery system"
)
app.add_typer(
    advanced_features_app, name="wizard", help="Configuration wizards and advanced features"
)
# Add new queue management commands
app.add_typer(
    messages_app, name="messages", help="View and manage error messages from daemon"
)
app.add_typer(
    queue_app, name="queue", help="Monitor queue status and statistics"
)
app.add_typer(
    tools_app, name="tools", help="Check tool availability status"
)
# Add collection type management commands
app.add_typer(
    collection_types_app, name="collections", help="Manage collection types and type-specific behaviors"
)
# Add error reporting and statistics commands
app.add_typer(
    errors_app, name="errors", help="Error statistics, trends, and comprehensive reporting"
)
# Add project alias management commands
app.add_typer(
    project_app, name="project", help="Project collection alias management"
)
# Add branch management commands
app.add_typer(
    branch_app, name="branch", help="Git branch management for project collections"
)
# Add migration tooling commands
app.add_typer(
    migrate_app, name="migrate", help="Migrate old collections to new _{project_id} architecture"
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version information"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Show verbose version information"
    ),
    config_path: str | None = typer.Option(
        None, "--config", "-c", help="Custom configuration file"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug mode with verbose logging"
    ),
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
        wqm status                         # Show processing status
        wqm status --live --interval 10    # Live monitoring
        wqm messages list                  # View error messages
        wqm queue status                   # Monitor queue status
        wqm tools status                   # Check tool availability
        wqm collections list-types         # View collections by type
        wqm errors stats                   # Error statistics
        wqm branch list --project abc123   # List branches in project
        wqm migrate detect                 # Detect old collections
        wqm init                           # Enable shell completion
    """
    # Handle version flag first, before any configuration loading
    if version:
        # Suppress warnings for clean version output
        if not verbose and not debug:
            import warnings

            warnings.filterwarnings("ignore")
        show_version(verbose=verbose or debug)
        raise typer.Exit()

    # If no command is invoked and no version flag, show help
    if not ctx.invoked_subcommand:
        print(ctx.get_help())
        raise typer.Exit()

    # Configure logging and environment based on debug flag
    import os

    if debug:
        # Re-enable logging for debug mode with loguru
        logging.disable(logging.NOTSET)
        # Enable verbose logging and initialization messages in debug mode
        os.environ["WQM_LOG_INIT"] = "true"
        setup_logging(
            log_file=None,
            verbose=True  # Enable console output in debug mode
        )
        logger.debug("Debug mode enabled with loguru")
    else:
        # Keep console logging disabled for normal CLI usage
        # Configure file-only logging for troubleshooting
        from pathlib import Path
        log_dir = Path.home() / ".config" / "workspace-qdrant" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "wqm-cli.log"
        setup_logging(
            log_file=str(log_file),
            verbose=False  # No console output in normal CLI usage
        )

    if config_path:
        # TODO: Load custom config
        if debug:
            logger.debug("Custom config path provided", config_path=config_path)
        pass

    # Setup enhanced error handling
    if not debug:  # Only in non-debug mode to avoid interference
        setup_exception_hook()


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
        version_str = "0.3.0"  # Fallback version

    if verbose:
        # Verbose version info with detailed metadata
        print(f"Workspace Qdrant MCP {version_str}")
        print(f"Python {sys.version.split()[0]}")
        print(f"Platform: {sys.platform}")
        print(f"Installation path: {Path(__file__).parent.parent}")
    else:
        # Clean version display - just the version number
        print(version_str)


def handle_async_command(coro: Coroutine[Any, Any, Any], debug: bool = False) -> Any:
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



def cli_entry_point() -> None:
    """Entry point for the wqm CLI command.

    This function serves as the entry point for the console script
    defined in pyproject.toml. It simply invokes the Typer app.
    """
    app()


if __name__ == "__main__":
    app()
