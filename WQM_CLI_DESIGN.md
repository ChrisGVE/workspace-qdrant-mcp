# WQM Unified CLI Design v2.0

**Version:** 2.0  
**Date:** 2025-08-30  
**Status:** Complete Design Specification

## Overview

The `wqm` (Workspace Qdrant MCP) command provides a unified interface for all v2.0 functionality, replacing the fragmented CLI commands of v1.0 with a single, intuitive command structure. This design implements the complete CLI architecture specified in the PRD v2.0.

## Design Philosophy

### Single Command Interface
- **One command to remember**: `wqm`
- **Domain-specific subcommands**: Organized by functional area
- **Consistent patterns**: Similar flags and behaviors across subcommands
- **Contextual help**: Rich help text with examples and usage patterns

### User Experience Principles
- **Adult user respect**: Provide tools and information, let users decide
- **Clear feedback**: Status, progress, and error information
- **Sensible defaults**: Common operations work with minimal flags
- **Power user friendly**: Advanced options available when needed

## Complete CLI Structure

```bash
wqm                    # Single command to remember
├── memory             # Memory rules and LLM behavior management
│   ├── list           # Show all memory rules
│   ├── add            # Add new rule (preference or behavior)
│   ├── edit           # Edit specific rule
│   ├── remove         # Remove rule
│   ├── tokens         # Show token usage
│   ├── trim           # Interactive token optimization
│   └── --web          # Future: Web interface for curation
│
├── admin              # System administration and configuration
│   ├── status         # System and engine status
│   ├── config         # Configuration management
│   ├── start-engine   # Start Rust engine
│   ├── stop-engine    # Stop Rust engine  
│   └── restart-engine # Restart with new config
│
├── ingest             # Manual document processing
│   ├── file           # Ingest single file
│   ├── folder         # Ingest folder contents
│   ├── yaml           # Process completed YAML metadata
│   └── web            # Crawl web pages from root URL
│
├── search             # Command-line search interface  
│   ├── project        # Search current project collections
│   ├── collection     # Search specific collection
│   ├── global         # Search global collections
│   ├── all            # Search all collections
│   └── memory         # Search memory/knowledge graph
│
├── library            # Readonly library collection management
│   ├── list           # Show all library collections
│   ├── create         # Create new library collection
│   ├── remove         # Remove library collection
│   └── status         # Show library statistics
│
└── watch              # Library folder watching (NOT projects)
    ├── add            # Add folder to watch for collection
    ├── list           # Show active watches
    ├── remove         # Stop watching folder
    ├── status         # Watch activity and statistics
    ├── pause          # Pause all/specific watches
    └── resume         # Resume paused watches
```

## Implementation Strategy

### Python CLI Framework

```python
# src/workspace_qdrant_mcp/cli/unified.py
"""
Unified CLI interface for workspace-qdrant-mcp v2.0.

This module implements the complete `wqm` command structure with all
subcommands and features specified in the PRD v2.0.
"""

import asyncio
import sys
from typing import Optional, List, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from ..rust_engine.manager import get_engine_manager
from ..core.config import Config
from .memory import MemoryCommands
from .admin import AdminCommands
from .ingest import IngestCommands
from .search import SearchCommands
from .library import LibraryCommands
from .watch import WatchCommands

# Initialize CLI app and console
app = typer.Typer(
    name="wqm",
    help="Workspace Qdrant MCP - Unified semantic workspace management",
    rich_markup_mode="rich",
    add_completion=True,
    no_args_is_help=True,
)
console = Console()

# Global configuration
config = Config()

# Add subcommand groups
app.add_typer(MemoryCommands().app, name="memory", help="🧠 Memory rules and LLM behavior management")
app.add_typer(AdminCommands().app, name="admin", help="⚙️  System administration and configuration")  
app.add_typer(IngestCommands().app, name="ingest", help="📁 Manual document processing")
app.add_typer(SearchCommands().app, name="search", help="🔍 Command-line search interface")
app.add_typer(LibraryCommands().app, name="library", help="📚 Library collection management")
app.add_typer(WatchCommands().app, name="watch", help="👀 Folder watching configuration")

@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version information"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Custom configuration file"),
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
    from .. import __version__
    
    version_info = Panel.fit(
        f"""[bold blue]Workspace Qdrant MCP[/bold blue]
        
Version: [green]{__version__}[/green]
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
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
```

## Memory Commands Implementation

### Memory Rule Management

```python
# src/workspace_qdrant_mcp/cli/memory.py
"""Memory system CLI commands for LLM behavior management."""

from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

from ..memory.manager import MemoryManager
from ..memory.types import MemoryRule, AuthorityLevel

class MemoryCommands:
    def __init__(self):
        self.app = typer.Typer(help="🧠 Memory rules and LLM behavior management")
        self.console = Console()
        self.memory_manager = MemoryManager()
        
        # Register commands
        self.app.command("list")(self.list_rules)
        self.app.command("add")(self.add_rule)
        self.app.command("edit")(self.edit_rule)
        self.app.command("remove")(self.remove_rule)
        self.app.command("tokens")(self.token_usage)
        self.app.command("trim")(self.trim_rules)
    
    async def list_rules(
        self,
        authority: Optional[str] = typer.Option(None, "--authority", "-a", help="Filter by authority level"),
        search: Optional[str] = typer.Option(None, "--search", "-s", help="Search rule content"),
        format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, yaml")
    ) -> None:
        """📋 List all memory rules with optional filtering."""
        
        rules = await self.memory_manager.get_rules(
            authority_filter=authority,
            search_query=search
        )
        
        if format == "json":
            import json
            self.console.print(json.dumps([rule.dict() for rule in rules], indent=2))
            return
        
        if format == "yaml":
            import yaml
            self.console.print(yaml.dump([rule.dict() for rule in rules]))
            return
        
        # Table format (default)
        table = Table(title="💭 Memory Rules")
        table.add_column("ID", style="cyan")
        table.add_column("Rule", style="white")
        table.add_column("Authority", justify="center")
        table.add_column("Scope", style="dim")
        table.add_column("Source", style="dim")
        
        for rule in rules:
            authority_style = "red" if rule.authority == "absolute" else "yellow"
            table.add_row(
                str(rule.id),
                rule.rule[:60] + "..." if len(rule.rule) > 60 else rule.rule,
                f"[{authority_style}]{rule.authority}[/{authority_style}]",
                ", ".join(rule.scope) if rule.scope else "all",
                rule.source
            )
        
        self.console.print(table)
        self.console.print(f"\n[dim]Total: {len(rules)} rules[/dim]")
    
    async def add_rule(
        self,
        rule: str = typer.Argument(..., help="The memory rule to add"),
        authority: str = typer.Option("default", "--authority", "-a", help="Authority level: absolute, default"),
        scope: Optional[List[str]] = typer.Option(None, "--scope", "-s", help="Rule scope (can be specified multiple times)"),
        interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive mode for detailed configuration")
    ) -> None:
        """➕ Add a new memory rule for LLM behavior."""
        
        if interactive:
            rule = Prompt.ask("Enter the memory rule")
            authority = Prompt.ask(
                "Authority level", 
                choices=["absolute", "default"], 
                default="default"
            )
            scope_input = Prompt.ask("Scope (comma-separated, or press Enter for all)", default="")
            scope = [s.strip() for s in scope_input.split(",")] if scope_input else None
        
        # Validate authority level
        if authority not in ["absolute", "default"]:
            self.console.print(f"[red]Invalid authority level: {authority}[/red]")
            raise typer.Exit(1)
        
        # Create memory rule
        memory_rule = MemoryRule(
            rule=rule,
            authority=AuthorityLevel(authority),
            scope=scope or [],
            source="user_cli"
        )
        
        # Check for conflicts
        conflicts = await self.memory_manager.check_conflicts(memory_rule)
        if conflicts:
            self.console.print(f"[yellow]⚠️  Potential conflicts detected:[/yellow]")
            for conflict in conflicts:
                self.console.print(f"  • {conflict.rule}")
            
            if not Confirm.ask("Continue anyway?"):
                self.console.print("[yellow]Rule addition cancelled[/yellow]")
                return
        
        # Add the rule
        rule_id = await self.memory_manager.add_rule(memory_rule)
        
        self.console.print(f"[green]✅ Memory rule added successfully (ID: {rule_id})[/green]")
        
        # Show token usage impact
        token_usage = await self.memory_manager.get_token_usage()
        self.console.print(f"[dim]Total memory tokens: {token_usage.total} ({token_usage.percentage:.1f}% of context)[/dim]")
```

## Admin Commands Implementation

### System Administration

```python
# src/workspace_qdrant_mcp/cli/admin.py
"""Administrative CLI commands for system management."""

from typing import Dict, Any
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

from ..rust_engine.manager import get_engine_manager
from ..core.config import Config

class AdminCommands:
    def __init__(self):
        self.app = typer.Typer(help="⚙️ System administration and configuration")
        self.console = Console()
        self.config = Config()
        
        # Register commands
        self.app.command("status")(self.status)
        self.app.command("config")(self.config_management)
        self.app.command("start-engine")(self.start_engine)
        self.app.command("stop-engine")(self.stop_engine)
        self.app.command("restart-engine")(self.restart_engine)
    
    async def status(
        self,
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status"),
        json_output: bool = typer.Option(False, "--json", help="Output as JSON")
    ) -> None:
        """📊 Show comprehensive system status."""
        
        engine_manager = get_engine_manager(self.config)
        status_data = await self._collect_status_data(engine_manager)
        
        if json_output:
            import json
            self.console.print(json.dumps(status_data, indent=2, default=str))
            return
        
        self._display_status_table(status_data, verbose)
    
    async def _collect_status_data(self, engine_manager) -> Dict[str, Any]:
        """Collect comprehensive status information."""
        
        # Engine status
        try:
            engine_status = await engine_manager.get_engine_status()
            engine_healthy = True
        except Exception as e:
            engine_status = {"error": str(e)}
            engine_healthy = False
        
        # Qdrant connectivity
        try:
            from ..core.client import QdrantWorkspaceClient
            client = QdrantWorkspaceClient(self.config)
            await client.health_check()
            qdrant_status = {"status": "healthy", "url": self.config.qdrant_url}
        except Exception as e:
            qdrant_status = {"status": "error", "error": str(e)}
        
        # System resources
        import psutil
        system_status = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": psutil.virtual_memory()._asdict(),
            "disk": psutil.disk_usage('/')._asdict(),
        }
        
        return {
            "timestamp": "now",
            "engine": engine_status,
            "engine_healthy": engine_healthy,
            "qdrant": qdrant_status,
            "system": system_status,
            "config": {
                "qdrant_url": self.config.qdrant_url,
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
            }
        }
    
    def _display_status_table(self, status_data: Dict[str, Any], verbose: bool) -> None:
        """Display status in rich table format."""
        
        # Overall health panel
        engine_healthy = status_data["engine_healthy"]
        qdrant_healthy = status_data["qdrant"]["status"] == "healthy"
        overall_healthy = engine_healthy and qdrant_healthy
        
        health_color = "green" if overall_healthy else "red"
        health_text = "🟢 HEALTHY" if overall_healthy else "🔴 UNHEALTHY"
        
        health_panel = Panel(
            f"[{health_color}]{health_text}[/{health_color}]",
            title="System Health",
            border_style=health_color,
        )
        self.console.print(health_panel)
        
        # Component status table
        table = Table(title="Component Status")
        table.add_column("Component", style="cyan", width=20)
        table.add_column("Status", justify="center", width=15)
        table.add_column("Details", style="dim")
        
        # Rust Engine
        if engine_healthy:
            engine_status = status_data["engine"]
            table.add_row(
                "🦀 Rust Engine",
                "[green]RUNNING[/green]",
                f"Tasks: {engine_status.get('active_tasks', 0)}A/{engine_status.get('queued_tasks', 0)}Q"
            )
        else:
            table.add_row(
                "🦀 Rust Engine", 
                "[red]ERROR[/red]",
                str(status_data["engine"].get("error", "Unknown error"))
            )
        
        # Qdrant Database
        qdrant_status = status_data["qdrant"]
        if qdrant_healthy:
            table.add_row(
                "🔍 Qdrant DB",
                "[green]CONNECTED[/green]",
                qdrant_status["url"]
            )
        else:
            table.add_row(
                "🔍 Qdrant DB",
                "[red]ERROR[/red]",
                str(qdrant_status.get("error", "Connection failed"))
            )
        
        # System resources (if verbose)
        if verbose:
            system = status_data["system"]
            table.add_row(
                "💻 CPU Usage",
                f"[yellow]{system['cpu_percent']:.1f}%[/yellow]",
                f"System load"
            )
            
            memory_percent = (system["memory"]["used"] / system["memory"]["total"]) * 100
            table.add_row(
                "🧠 Memory",
                f"[yellow]{memory_percent:.1f}%[/yellow]",
                f"{system['memory']['used'] // (1024**3):.1f}GB / {system['memory']['total'] // (1024**3):.1f}GB"
            )
        
        self.console.print(table)
        
        # Configuration summary
        if verbose:
            config_table = Table(title="Configuration")
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="white")
            
            config = status_data["config"]
            for key, value in config.items():
                config_table.add_row(key.replace("_", " ").title(), str(value))
            
            self.console.print(config_table)
```

## Library and Watch Commands

### Library Management

```python
# src/workspace_qdrant_mcp/cli/library.py
"""Library collection management CLI commands."""

from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path

class LibraryCommands:
    def __init__(self):
        self.app = typer.Typer(help="📚 Library collection management")
        self.console = Console()
        
        # Register commands
        self.app.command("list")(self.list_libraries)
        self.app.command("create")(self.create_library)
        self.app.command("remove")(self.remove_library)
        self.app.command("status")(self.library_status)
    
    async def create_library(
        self,
        name: str = typer.Argument(..., help="Library name (will be prefixed with _)"),
        description: Optional[str] = typer.Option(None, "--description", "-d", help="Library description"),
        tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Library tags"),
    ) -> None:
        """📚 Create a new library collection."""
        
        collection_name = f"_{name}"
        
        # Validate collection name
        if await self._collection_exists(collection_name):
            self.console.print(f"[red]❌ Collection '{collection_name}' already exists[/red]")
            raise typer.Exit(1)
        
        # Create collection
        await self._create_collection(collection_name, description, tags or [])
        
        self.console.print(f"[green]✅ Library collection '{name}' created successfully[/green]")
        self.console.print(f"[dim]Collection name: {collection_name}[/dim]")
        self.console.print(f"[dim]Use 'wqm watch add PATH --collection={collection_name}' to start monitoring files[/dim]")
```

### File Watching

```python
# src/workspace_qdrant_mcp/cli/watch.py
"""File watching CLI commands for library auto-ingestion."""

from typing import List, Optional
import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path

class WatchCommands:
    def __init__(self):
        self.app = typer.Typer(help="👀 Folder watching configuration")
        self.console = Console()
        
        # Register commands
        self.app.command("add")(self.add_watch)
        self.app.command("list")(self.list_watches)
        self.app.command("remove")(self.remove_watch)
        self.app.command("status")(self.watch_status)
        self.app.command("pause")(self.pause_watch)
        self.app.command("resume")(self.resume_watch)
    
    async def add_watch(
        self,
        path: str = typer.Argument(..., help="Path to watch"),
        collection: str = typer.Option(..., "--collection", "-c", help="Target collection (must start with _)"),
        patterns: Optional[List[str]] = typer.Option(
            ["*.pdf", "*.epub", "*.txt", "*.md"],
            "--pattern", "-p",
            help="File patterns to watch"
        ),
        ignore: Optional[List[str]] = typer.Option(
            [".git/*", "node_modules/*", "__pycache__/*"],
            "--ignore", "-i",
            help="Ignore patterns"
        ),
        auto_ingest: bool = typer.Option(True, "--auto/--no-auto", help="Enable automatic ingestion"),
        recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Watch subdirectories")
    ) -> None:
        """➕ Add a folder to watch for automatic ingestion."""
        
        watch_path = Path(path)
        
        # Validate path
        if not watch_path.exists():
            self.console.print(f"[red]❌ Path does not exist: {path}[/red]")
            raise typer.Exit(1)
        
        if not watch_path.is_dir():
            self.console.print(f"[red]❌ Path is not a directory: {path}[/red]")
            raise typer.Exit(1)
        
        # Validate collection name
        if not collection.startswith('_'):
            self.console.print(f"[red]❌ Collection must start with underscore (library collection): {collection}[/red]")
            raise typer.Exit(1)
        
        # Add watch configuration
        from ..rust_engine.manager import get_engine_manager
        from ..core.config import Config
        
        engine_manager = get_engine_manager(Config())
        
        try:
            result = await engine_manager.start_watching(
                path=str(watch_path.absolute()),
                collection=collection,
                auto_ingest=auto_ingest,
                file_patterns=patterns,
                ignore_patterns=ignore
            )
            
            self.console.print(f"[green]✅ Started watching: {watch_path}[/green]")
            self.console.print(f"[dim]Collection: {collection}[/dim]")
            self.console.print(f"[dim]Patterns: {', '.join(patterns)}[/dim]")
            self.console.print(f"[dim]Auto-ingest: {'enabled' if auto_ingest else 'disabled'}[/dim]")
            
        except Exception as e:
            self.console.print(f"[red]❌ Failed to add watch: {e}[/red]")
            raise typer.Exit(1)
```

## Usage Examples

### Common Workflows

```bash
# Initial setup and status check
wqm admin status                              # Check system health
wqm admin start-engine                        # Start Rust engine if needed

# Memory management
wqm memory add "Always use uv for Python package management"
wqm memory add "Make atomic commits following conventional format" --authority absolute
wqm memory list --authority absolute          # Show only absolute rules
wqm memory tokens                            # Check token usage

# Library setup
wqm library create technical-books --description "Programming and technical books"
wqm watch add ~/Documents/Books --collection=_technical-books --pattern "*.pdf" --pattern "*.epub"
wqm library status _technical-books          # Check ingestion progress

# Document ingestion
wqm ingest file document.pdf --collection=project-docs
wqm ingest folder ~/project/docs --collection=project-docs --recursive
wqm ingest yaml ~/Documents/Books/metadata.yaml  # Process completed metadata

# Search operations
wqm search project "rust async patterns"     # Search current project
wqm search collection _technical-books "design patterns"  # Search specific library
wqm search memory "python preferences"       # Query knowledge graph
wqm search all "microservices architecture"  # Search everything

# Watch management
wqm watch list                               # Show all active watches
wqm watch status                            # Detailed watch statistics
wqm watch pause ~/Documents/Books           # Pause specific watch
wqm watch resume                            # Resume all watches
```

### Advanced Usage

```bash
# Comprehensive system overview
wqm admin status --verbose --json > system-status.json

# Memory rule management with conflict detection
wqm memory add "Use TypeScript strict mode" --interactive
wqm memory trim --max-tokens 2000           # Optimize token usage

# Batch document processing
wqm ingest folder ~/research --collection=research-papers \
  --pattern "*.pdf" --ignore "*draft*" --recursive

# Complex search with relationships
wqm search all "machine learning" --include-relationships --format json

# Library maintenance
wqm library remove old-references --confirm
wqm watch add ~/sync/papers --collection=_papers --auto --pattern "*.pdf"
```

## Error Handling and User Feedback

### Rich Error Messages

```python
def handle_collection_conflict_error():
    """Example of rich error handling."""
    console.print(Panel.fit(
        """[red]❌ Collection Name Conflict[/red]

The collection name '[yellow]library[/yellow]' conflicts with the reserved library
collection '[yellow]_library[/yellow]'.

[bold]Suggestions:[/bold]
• Use '[green]wqm library create library[/green]' to create '[cyan]_library[/cyan]'
• Choose a different name: '[green]my-library[/green]', '[green]documents[/green]'
• Remove existing '[cyan]_library[/cyan]' with '[green]wqm library remove library[/green]'

[dim]Library collections (prefixed with _) are read-only from MCP tools
and managed exclusively through the CLI.[/dim]""",
        title="⚠️ Collection Conflict",
        border_style="red",
    ))
```

### Progress Indicators

```python
async def ingest_with_progress():
    """Example of progress indication during long operations."""
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing documents...", total=None)
        
        async for update in engine_manager.ingest_folder(...):
            if update.get("total"):
                progress.update(task, total=update["total"])
            if update.get("completed"):
                progress.update(task, completed=update["completed"])
            
            console.print(f"[dim]{update.get('message', '')}[/dim]")
```

This unified CLI design provides a comprehensive, user-friendly interface that makes all v2.0 functionality accessible through a single, memorable command while maintaining the power and flexibility needed for advanced users.