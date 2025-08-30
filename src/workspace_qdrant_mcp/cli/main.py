"""
Main CLI entry point for wqm (workspace-qdrant-mcp) command.

This module provides the unified CLI interface with domain-specific subcommands
for memory management, system administration, ingestion, search, and more.

Usage:
    wqm memory list              # List memory rules
    wqm memory add "rule text"   # Add memory rule
    wqm admin status             # Show system status
    wqm search "query text"      # Search workspace
"""

import click
from rich.console import Console

from .memory import memory

console = Console()


@click.group(name="wqm")
@click.version_option(version="0.1.0", prog_name="workspace-qdrant-mcp")
def cli():
    """
    Workspace Qdrant MCP - Unified command interface.
    
    A comprehensive semantic workspace platform with memory-driven LLM behavior,
    knowledge graph capabilities, and intelligent document management.
    
    Core domains:
    
    • memory: Memory rules and LLM behavior management
    • admin: System administration and configuration  
    • ingest: Manual document processing
    • search: Command-line search interface
    • library: Readonly library collection management
    • watch: Library folder watching
    
    Use 'wqm <domain> --help' for domain-specific commands.
    """
    pass


# Add memory commands
cli.add_command(memory)

# Placeholder commands for other domains - to be implemented in future tasks

@cli.group()
def admin():
    """System administration and configuration."""
    pass


@admin.command()
def status():
    """Show system and engine status."""
    console.print("[yellow]Admin commands not yet implemented - part of unified CLI task[/yellow]")
    console.print("This will be implemented in Task 10: Design unified CLI with wqm command structure")


@cli.group()
def ingest():
    """Manual document processing."""
    pass


@ingest.command()
@click.argument("path")
def file(path: str):
    """Ingest single file."""
    console.print(f"[yellow]Would ingest file: {path}[/yellow]")
    console.print("Ingestion commands will be implemented with the Rust engine in future tasks")


@cli.group()
def search():
    """Command-line search interface."""
    pass


@search.command()
@click.argument("query")
def project(query: str):
    """Search current project collections."""
    console.print(f"[yellow]Would search project collections for: {query}[/yellow]")
    console.print("Advanced search modes will be implemented in Task 13")


@cli.group()
def library():
    """Readonly library collection management."""
    pass


@library.command()
def list():
    """Show all library collections."""
    console.print("[yellow]Library management not yet implemented[/yellow]")
    console.print("This will be part of the library folder watching system")


@cli.group()
def watch():
    """Library folder watching (NOT projects)."""
    pass


@watch.command()
def list():
    """Show active watches."""
    console.print("[yellow]Watch commands not yet implemented[/yellow]")
    console.print("This will be implemented in Task 14: Implement library folder watching system")


if __name__ == "__main__":
    cli()