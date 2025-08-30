"""File watching CLI commands.

This module provides management for library folder watching,
enabling automatic ingestion of files into library collections.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm

console = Console()

# Create the watch app
watch_app = typer.Typer(help="👀 Folder watching configuration")

def handle_async(coro):
    """Helper to run async commands."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

@watch_app.command("add")
def add_watch(
    path: str = typer.Argument(..., help="Path to watch"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target collection (must start with _)"),
    patterns: Optional[List[str]] = typer.Option(
        ["*.pdf", "*.epub", "*.txt", "*.md"],
        "--pattern", "-p",
        help="File patterns to watch"
    ),
    ignore: Optional[List[str]] = typer.Option(
        [".git/*", "node_modules/*", "__pycache__/*", ".DS_Store"],
        "--ignore", "-i",
        help="Ignore patterns"
    ),
    auto_ingest: bool = typer.Option(True, "--auto/--no-auto", help="Enable automatic ingestion"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Watch subdirectories"),
    debounce: int = typer.Option(5, "--debounce", help="Debounce time in seconds"),
):
    """➕ Add a folder to watch for automatic ingestion."""
    handle_async(_add_watch(path, collection, patterns, ignore, auto_ingest, recursive, debounce))

@watch_app.command("list")
def list_watches(
    active_only: bool = typer.Option(False, "--active", help="Show only active watches"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
):
    """📋 Show all active watches."""
    handle_async(_list_watches(active_only, collection, format))

@watch_app.command("remove")
def remove_watch(
    path: Optional[str] = typer.Argument(None, help="Path to stop watching (or watch ID)"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Remove all watches for collection"),
    all: bool = typer.Option(False, "--all", help="Remove all watches"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """🗑️ Stop watching folder."""
    handle_async(_remove_watch(path, collection, all, force))

@watch_app.command("status")
def watch_status(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics"),
    recent: bool = typer.Option(False, "--recent", help="Show recent activity"),
):
    """📊 Watch activity and statistics."""
    handle_async(_watch_status(detailed, recent))

@watch_app.command("pause")
def pause_watches(
    path: Optional[str] = typer.Argument(None, help="Specific path to pause (or watch ID)"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Pause all watches for collection"),
    all: bool = typer.Option(False, "--all", help="Pause all watches"),
):
    """⏸️ Pause all or specific watches."""
    handle_async(_pause_watches(path, collection, all))

@watch_app.command("resume")
def resume_watches(
    path: Optional[str] = typer.Argument(None, help="Specific path to resume (or watch ID)"),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Resume all watches for collection"),
    all: bool = typer.Option(False, "--all", help="Resume all watches"),
):
    """▶️ Resume paused watches."""
    handle_async(_resume_watches(path, collection, all))

@watch_app.command("sync")
def sync_watched_folders(
    path: Optional[str] = typer.Argument(None, help="Specific path to sync"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be synced"),
    force: bool = typer.Option(False, "--force", help="Force sync all files"),
):
    """🔄 Sync watched folders manually."""
    handle_async(_sync_watched_folders(path, dry_run, force))

# Async implementation functions
async def _add_watch(
    path: str,
    collection: str,
    patterns: Optional[List[str]],
    ignore: Optional[List[str]],
    auto_ingest: bool,
    recursive: bool,
    debounce: int
):
    """Add a folder watch configuration."""
    try:
        watch_path = Path(path).resolve()
        
        # Validate path
        if not watch_path.exists():
            console.print(f"[red]❌ Path does not exist: {path}[/red]")
            raise typer.Exit(1)
        
        if not watch_path.is_dir():
            console.print(f"[red]❌ Path is not a directory: {path}[/red]")
            raise typer.Exit(1)
        
        # Validate collection name (must be library collection)
        if not collection.startswith('_'):
            console.print(f"[red]❌ Collection must start with underscore (library collection): {collection}[/red]")
            console.print("[dim]Library collections are used for reference materials and watched folders[/dim]")
            console.print(f"[dim]Create it first with: wqm library create {collection[1:] if collection.startswith('_') else collection}[/dim]")
            raise typer.Exit(1)
        
        console.print(f"[bold blue]👀 Adding watch configuration[/bold blue]")
        console.print(f"Path: [cyan]{watch_path}[/cyan]")
        console.print(f"Collection: [cyan]{collection}[/cyan]")
        
        # TODO: Implement actual watch management with Rust engine
        # This will be part of Task 14: Library folder watching system
        
        # For now, show what would be configured
        config_info = f"""[bold]Watch Configuration[/bold]

📁 [cyan]Path:[/cyan] {watch_path}
📚 [cyan]Collection:[/cyan] {collection}
🔄 [cyan]Auto-ingest:[/cyan] {'✅ Enabled' if auto_ingest else '❌ Disabled'}
📂 [cyan]Recursive:[/cyan] {'✅ Yes' if recursive else '❌ No'}
⏱️  [cyan]Debounce:[/cyan] {debounce} seconds

[bold]File Patterns:[/bold]
{chr(10).join(f'  • {pattern}' for pattern in patterns)}

[bold]Ignore Patterns:[/bold]
{chr(10).join(f'  • {ignore_pattern}' for ignore_pattern in ignore)}"""
        
        config_panel = Panel(
            config_info,
            title="⚙️ Watch Configuration",
            border_style="blue"
        )
        console.print(config_panel)
        
        console.print("\n[yellow]🚧 File watching is not yet implemented[/yellow]")
        console.print("This feature will be implemented in Task 14: Library folder watching system")
        console.print("\n[dim]The watch configuration shown above will be used when the feature is ready.[/dim]")
        
        # Show current workaround
        console.print(f"\n[bold blue]💡 Current Alternative[/bold blue]")
        console.print("For now, you can manually process folders:")
        console.print(f"  [green]wqm ingest folder \"{watch_path}\" --collection={collection} --recursive[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to add watch: {e}[/red]")
        raise typer.Exit(1)

async def _list_watches(active_only: bool, collection: Optional[str], format: str):
    """List all watch configurations."""
    try:
        console.print("[bold blue]📋 Active Watches[/bold blue]")
        
        # TODO: Implement actual watch listing
        # This will be part of Task 14: Library folder watching system
        
        console.print("[yellow]🚧 File watching is not yet implemented[/yellow]")
        console.print("This feature will be implemented in Task 14: Library folder watching system")
        
        # Show placeholder table structure
        console.print("\n[dim]When implemented, this will show:[/dim]")
        
        table = Table(title="👀 Watch Configurations")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Path", style="white", width=30)
        table.add_column("Collection", style="blue", width=20)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Files", justify="right", width=8)
        table.add_column("Last Activity", width=15)
        
        # Example rows
        table.add_row("1", "[dim]~/Documents/Books[/dim]", "[dim]_technical-books[/dim]", "[dim]ACTIVE[/dim]", "[dim]42[/dim]", "[dim]2 hours ago[/dim]")
        table.add_row("2", "[dim]~/Papers[/dim]", "[dim]_research[/dim]", "[dim]PAUSED[/dim]", "[dim]128[/dim]", "[dim]1 day ago[/dim]")
        
        console.print(table)
        console.print("[dim]^ Example of what the interface will look like[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to list watches: {e}[/red]")
        raise typer.Exit(1)

async def _remove_watch(path: Optional[str], collection: Optional[str], all: bool, force: bool):
    """Remove watch configurations."""
    try:
        if all:
            console.print("[bold red]🗑️ Remove All Watches[/bold red]")
        elif collection:
            console.print(f"[bold red]🗑️ Remove Watches for Collection: {collection}[/bold red]")
        elif path:
            console.print(f"[bold red]🗑️ Remove Watch: {path}[/bold red]")
        else:
            console.print("[red]❌ Must specify --all, --collection, or a path[/red]")
            raise typer.Exit(1)
        
        if not force:
            action = "all watches" if all else f"watches for {collection}" if collection else f"watch for {path}"
            if not Confirm.ask(f"[red]Are you sure you want to remove {action}?[/red]"):
                console.print("[yellow]Operation cancelled[/yellow]")
                return
        
        console.print("[yellow]🚧 Watch removal is not yet implemented[/yellow]")
        console.print("This feature will be implemented in Task 14: Library folder watching system")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to remove watches: {e}[/red]")
        raise typer.Exit(1)

async def _watch_status(detailed: bool, recent: bool):
    """Show watch activity and statistics."""
    try:
        console.print("[bold blue]📊 Watch System Status[/bold blue]")
        
        # TODO: Implement actual watch status
        # This will be part of Task 14: Library folder watching system
        
        console.print("[yellow]🚧 Watch status is not yet implemented[/yellow]")
        console.print("This feature will be implemented in Task 14: Library folder watching system")
        
        # Show what the status would look like
        console.print("\n[dim]When implemented, this will show:[/dim]")
        
        # System status
        status_panel = Panel(
            """[bold]Watch System Status[/bold]

🔍 [cyan]Active Watches:[/cyan] 0
⏸️  [cyan]Paused Watches:[/cyan] 0
📁 [cyan]Monitored Directories:[/cyan] 0
📄 [cyan]Files Tracked:[/cyan] 0
⚡ [cyan]Processing Queue:[/cyan] 0

[bold]Recent Activity (24h)[/bold]
📥 [cyan]Files Added:[/cyan] 0
📝 [cyan]Files Modified:[/cyan] 0
🗑️  [cyan]Files Removed:[/cyan] 0
⚠️  [cyan]Errors:[/cyan] 0""",
            title="📈 System Overview",
            border_style="blue"
        )
        console.print(status_panel)
        
        if detailed:
            console.print("\n[dim]Detailed statistics would include:[/dim]")
            console.print("• Per-collection processing statistics")
            console.print("• File type breakdown")
            console.print("• Processing performance metrics")
            console.print("• Error logs and resolution suggestions")
        
        if recent:
            console.print("\n[dim]Recent activity would show:[/dim]")
            console.print("• Timeline of file changes")
            console.print("• Processing results")
            console.print("• Auto-ingestion outcomes")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to get watch status: {e}[/red]")
        raise typer.Exit(1)

async def _pause_watches(path: Optional[str], collection: Optional[str], all: bool):
    """Pause watch configurations."""
    try:
        if all:
            console.print("[bold yellow]⏸️ Pausing All Watches[/bold yellow]")
        elif collection:
            console.print(f"[bold yellow]⏸️ Pausing Watches for Collection: {collection}[/bold yellow]")
        elif path:
            console.print(f"[bold yellow]⏸️ Pausing Watch: {path}[/bold yellow]")
        else:
            console.print("[red]❌ Must specify --all, --collection, or a path[/red]")
            raise typer.Exit(1)
        
        console.print("[yellow]🚧 Watch pausing is not yet implemented[/yellow]")
        console.print("This feature will be implemented in Task 14: Library folder watching system")
        
        # Show what would happen
        action = "All watches" if all else f"Watches for {collection}" if collection else f"Watch for {path}"
        console.print(f"\n[dim]{action} would be paused (file monitoring stopped but configuration preserved)[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to pause watches: {e}[/red]")
        raise typer.Exit(1)

async def _resume_watches(path: Optional[str], collection: Optional[str], all: bool):
    """Resume watch configurations."""
    try:
        if all:
            console.print("[bold green]▶️ Resuming All Watches[/bold green]")
        elif collection:
            console.print(f"[bold green]▶️ Resuming Watches for Collection: {collection}[/bold green]")
        elif path:
            console.print(f"[bold green]▶️ Resuming Watch: {path}[/bold green]")
        else:
            console.print("[red]❌ Must specify --all, --collection, or a path[/red]")
            raise typer.Exit(1)
        
        console.print("[yellow]🚧 Watch resuming is not yet implemented[/yellow]")
        console.print("This feature will be implemented in Task 14: Library folder watching system")
        
        # Show what would happen
        action = "All watches" if all else f"Watches for {collection}" if collection else f"Watch for {path}"
        console.print(f"\n[dim]{action} would be resumed (file monitoring restarted)[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to resume watches: {e}[/red]")
        raise typer.Exit(1)

async def _sync_watched_folders(path: Optional[str], dry_run: bool, force: bool):
    """Manually sync watched folders."""
    try:
        if path:
            console.print(f"[bold blue]🔄 Syncing Watch: {path}[/bold blue]")
        else:
            console.print("[bold blue]🔄 Syncing All Watched Folders[/bold blue]")
        
        if dry_run:
            console.print("[yellow]DRY RUN - No files will be processed[/yellow]")
        
        console.print("[yellow]🚧 Watch syncing is not yet implemented[/yellow]")
        console.print("This feature will be implemented in Task 14: Library folder watching system")
        
        # Show what would happen
        console.print("\n[dim]When implemented, sync will:[/dim]")
        console.print("• Scan all watched directories for changes")
        console.print("• Process new and modified files")
        console.print("• Update collection indexes")
        console.print("• Report processing results")
        
        if force:
            console.print("• Re-process all files (not just changes)")
        
        # Current alternative
        console.print(f"\n[bold blue]💡 Current Alternative[/bold blue]")
        console.print("Use manual folder ingestion:")
        if path:
            console.print(f"  [green]wqm ingest folder \"{path}\" --collection=<library> --force[/green]")
        else:
            console.print("  [green]wqm ingest folder <path> --collection=<library> --force[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to sync watches: {e}[/red]")
        raise typer.Exit(1)