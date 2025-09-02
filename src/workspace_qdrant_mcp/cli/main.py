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
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from ..observability import get_logger

from .commands.admin import admin_app
from .commands.ingest import ingest_app
from .commands.library import library_app

# Import command modules
from .commands.memory import memory_app
from .commands.search import search_app
from .commands.watch import watch_app
from .observability import observability_app

# Initialize main app, console, and logger
app = typer.Typer(
    name="wqm",
    help="🚀 Workspace Qdrant MCP - Unified semantic workspace management",
    rich_markup_mode="rich",
    add_completion=True,
    no_args_is_help=True,
)
console = Console()
logger = get_logger(__name__)

# Add subcommand groups
app.add_typer(memory_app, name="memory", help="🧠 Memory rules and LLM behavior management")
app.add_typer(admin_app, name="admin", help="⚙️  System administration and configuration")
app.add_typer(ingest_app, name="ingest", help="📁 Manual document processing")
app.add_typer(search_app, name="search", help="🔍 Command-line search interface")
app.add_typer(library_app, name="library", help="📚 Library collection management")
app.add_typer(watch_app, name="watch", help="👀 Folder watching configuration")
app.add_typer(observability_app, name="observability", help="📊 Observability, monitoring, and health checks")

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version information"),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Custom configuration file"),
) -> None:
    """
    🚀 Workspace Qdrant MCP - Unified semantic workspace management

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
    if version:
        show_version()
        return

    if config_path:
        # TODO: Load custom config
        pass

    if ctx.invoked_subcommand is None:
        show_welcome()

def show_version() -> None:
    """Display version information."""
    try:
        from workspace_qdrant_mcp import __version__
        version_str = __version__
    except ImportError:
        version_str = "0.2.0"  # Fallback version

    version_info = Panel.fit(
        f"""[bold blue]Workspace Qdrant MCP[/bold blue]

Version: [green]{version_str}[/green]
Engine: [yellow]Rust v2.0[/yellow]
Python: [cyan]{sys.version.split()[0]}[/cyan]

🚀 High-performance semantic workspace management
📚 Memory-driven LLM behavior
🔍 Advanced hybrid search
⚡ Rust-powered processing engine""",
        title="Version Information",
        border_style="blue",
    )
    console.print(version_info)

def show_welcome() -> None:
    """Display welcome message and quick start guide."""
    welcome = Panel.fit(
        """[bold blue]Welcome to Workspace Qdrant MCP v2.0![/bold blue]

🧠 [yellow]Memory System:[/yellow] Personalize LLM behavior with persistent rules
📁 [yellow]Document Processing:[/yellow] Ingest text, PDF, EPUB, and code files
🔍 [yellow]Semantic Search:[/yellow] Find content across your entire workspace
📚 [yellow]Library Management:[/yellow] Organize and auto-watch document collections
⚡ [yellow]High Performance:[/yellow] Rust-powered processing engine

[dim]Quick Start:[/dim]
  [green]wqm admin status[/green]                 # Check system health
  [green]wqm memory add "Use uv for Python"[/green]   # Add behavior rule
  [green]wqm ingest file document.pdf[/green]     # Process a document
  [green]wqm search project "patterns"[/green]    # Search your project

[dim]Need help?[/dim] Use [cyan]wqm COMMAND --help[/cyan] for detailed information.""",
        title="🚀 Workspace Qdrant MCP",
        border_style="blue",
    )
    console.print(welcome)

def handle_async_command(coro):
    """Helper to run async commands in CLI context."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        logger.error("CLI operation failed", error=str(e), exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

# Make cli available for backward compatibility
cli = app

if __name__ == "__main__":
    app()
