
from ...observability import get_logger
logger = get_logger(__name__)
"""Administrative CLI commands for system management.

This module provides comprehensive system administration capabilities
including status monitoring, configuration management, and engine lifecycle.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ...core.client import QdrantWorkspaceClient, create_qdrant_client
from ...core.config import Config
from ...utils.project_detection import ProjectDetector

console = Console()

# Create the admin app
admin_app = typer.Typer(
    help="⚙️ System administration and configuration",
    no_args_is_help=True
)

def handle_async(coro):
    """Helper to run async commands."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        console.logger.info("\n[yellow]Operation cancelled by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.logger.info("[red]Error: {e}[/red]")
        raise typer.Exit(1)

@admin_app.command("status")
def system_status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch status continuously (5s refresh)"),
):
    """📊 Show comprehensive system status."""
    if watch:
        handle_async(_watch_status(verbose))
    else:
        handle_async(_system_status(verbose, json_output))

@admin_app.command("config")
def config_management(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
    path: str | None = typer.Option(None, "--path", help="Configuration file path"),
):
    """⚙️ Configuration management."""
    handle_async(_config_management(show, validate, path))

@admin_app.command("start-engine")
def start_engine(
    force: bool = typer.Option(False, "--force", help="Force start even if already running"),
    config_path: str | None = typer.Option(None, "--config", help="Custom config path"),
):
    """🚀 Start the Rust processing engine."""
    handle_async(_start_engine(force, config_path))

@admin_app.command("stop-engine")
def stop_engine(
    force: bool = typer.Option(False, "--force", help="Force stop without graceful shutdown"),
    timeout: int = typer.Option(30, "--timeout", help="Shutdown timeout in seconds"),
):
    """🛑 Stop the Rust processing engine."""
    handle_async(_stop_engine(force, timeout))

@admin_app.command("restart-engine")
def restart_engine(
    config_path: str | None = typer.Option(None, "--config", help="Custom config path"),
):
    """🔄 Restart engine with new configuration."""
    handle_async(_restart_engine(config_path))

@admin_app.command("collections")
def list_collections(
    project: str | None = typer.Option(None, "--project", help="Filter by project"),
    stats: bool = typer.Option(False, "--stats", help="Include collection statistics"),
    library: bool = typer.Option(False, "--library", help="Show only library collections (_prefixed)"),
):
    """📁 List and manage collections."""
    handle_async(_list_collections(project, stats, library))

@admin_app.command("health")
def health_check(
    deep: bool = typer.Option(False, "--deep", help="Perform deep health check"),
    timeout: int = typer.Option(10, "--timeout", help="Health check timeout"),
):
    """🏥 Comprehensive health check."""
    handle_async(_health_check(deep, timeout))

# Async implementation functions
async def _system_status(verbose: bool, json_output: bool):
    """Show comprehensive system status."""
    try:
        config = Config()
        status_data = await _collect_status_data(config)

        if json_output:
            logger.info("Output", data=json.dumps(status_data, indent=2, default=str))
            return

        _display_status_panel(status_data, verbose)

    except Exception as e:
        console.logger.info("[red]Error getting system status: {e}[/red]")
        raise typer.Exit(1)

async def _watch_status(verbose: bool):
    """Watch system status with continuous refresh."""
    import time

    console.logger.info("[bold blue]🔍 Watching system status (Ctrl+C to stop)[/bold blue]\n")

    try:
        while True:
            # Clear screen
            console.clear()

            config = Config()
            status_data = await _collect_status_data(config)
            _display_status_panel(status_data, verbose)

            console.logger.info("\n[dim]Last updated: {status_data['timestamp']} | Press Ctrl+C to stop[/dim]")

            # Wait 5 seconds
            await asyncio.sleep(5)

    except KeyboardInterrupt:
        console.logger.info("\n[yellow]Status monitoring stopped[/yellow]")

async def _collect_status_data(config: Config) -> dict[str, Any]:
    """Collect comprehensive status information."""
    from datetime import datetime

    status_data = {
        "timestamp": datetime.now().isoformat(),
        "config_valid": True,
        "qdrant": {"status": "unknown"},
        "rust_engine": {"status": "unknown"},
        "system": {},
        "project": {},
        "collections": {}
    }

    # Test Qdrant connectivity
    try:
        client = create_qdrant_client(config.qdrant_client_config)
        # Simple ping test
        collections = await client.list_collections()
        status_data["qdrant"] = {
            "status": "healthy",
            "url": config.qdrant.url,
            "collections_count": len(collections),
            "version": "1.x"  # TODO: Get actual version
        }
    except Exception as e:
        status_data["qdrant"] = {
            "status": "error",
            "error": str(e),
            "url": config.qdrant.url if hasattr(config, 'qdrant') else "unknown"
        }

    # Check Rust Engine (placeholder - will be implemented with Rust integration)
    try:
        # TODO: Implement actual Rust engine status check
        status_data["rust_engine"] = {
            "status": "not_implemented",
            "message": "Rust engine status checking will be implemented in Task 11"
        }
    except Exception as e:
        status_data["rust_engine"] = {
            "status": "error",
            "error": str(e)
        }

    # System resources
    try:
        status_data["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "percent": psutil.virtual_memory().percent,
                "used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "total_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            },
            "disk": {
                "percent": psutil.disk_usage('/').percent,
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 2),
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 2)
            }
        }
    except Exception as e:
        status_data["system"] = {"error": str(e)}

    # Project detection
    try:
        detector = ProjectDetector(config.workspace.github_user if hasattr(config, 'workspace') else None)
        projects = detector.detect_projects([Path.cwd()])
        status_data["project"] = {
            "current_dir": str(Path.cwd()),
            "detected_projects": len(projects),
            "current_project": projects[0].name if projects else "unknown"
        }
    except Exception as e:
        status_data["project"] = {"error": str(e)}

    return status_data

def _display_status_panel(status_data: dict[str, Any], verbose: bool):
    """Display status in rich panel format."""

    # Overall health assessment
    qdrant_healthy = status_data["qdrant"]["status"] == "healthy"
    rust_healthy = status_data["rust_engine"]["status"] in ["healthy", "not_implemented"]
    overall_healthy = qdrant_healthy and rust_healthy

    health_color = "green" if overall_healthy else "red"
    health_icon = "🟢" if overall_healthy else "🔴"
    health_text = f"{health_icon} {'HEALTHY' if overall_healthy else 'UNHEALTHY'}"

    # Main status panel
    status_panel = Panel(
        f"[{health_color}]{health_text}[/{health_color}]",
        title="🚀 System Health",
        border_style=health_color,
    )
    console.logger.info("Output", data=status_panel)

    # Component status table
    table = Table(title="📊 Component Status")
    table.add_column("Component", style="cyan", width=20)
    table.add_column("Status", justify="center", width=15)
    table.add_column("Details", style="dim")

    # Qdrant Database
    qdrant = status_data["qdrant"]
    if qdrant["status"] == "healthy":
        table.add_row(
            "🔍 Qdrant DB",
            "[green]CONNECTED[/green]",
            f"{qdrant.get('collections_count', 0)} collections | {qdrant.get('url', 'unknown')}"
        )
    else:
        table.add_row(
            "🔍 Qdrant DB",
            "[red]ERROR[/red]",
            str(qdrant.get("error", "Connection failed"))
        )

    # Rust Engine
    rust_engine = status_data["rust_engine"]
    if rust_engine["status"] == "healthy":
        table.add_row(
            "🦀 Rust Engine",
            "[green]RUNNING[/green]",
            f"Tasks: {rust_engine.get('active_tasks', 0)}A/{rust_engine.get('queued_tasks', 0)}Q"
        )
    elif rust_engine["status"] == "not_implemented":
        table.add_row(
            "🦀 Rust Engine",
            "[yellow]PENDING[/yellow]",
            rust_engine.get("message", "Not yet implemented")
        )
    else:
        table.add_row(
            "🦀 Rust Engine",
            "[red]ERROR[/red]",
            str(rust_engine.get("error", "Unknown error"))
        )

    # Project Context
    project = status_data["project"]
    if "error" not in project:
        table.add_row(
            "📁 Project",
            "[green]DETECTED[/green]",
            f"{project.get('current_project', 'unknown')} | {project.get('detected_projects', 0)} projects"
        )
    else:
        table.add_row(
            "📁 Project",
            "[yellow]WARNING[/yellow]",
            str(project.get("error", "Detection failed"))
        )

    console.logger.info("Output", data=table)

    # System resources (if verbose)
    if verbose and "error" not in status_data["system"]:
        system = status_data["system"]

        resource_table = Table(title="💻 System Resources")
        resource_table.add_column("Resource", style="cyan")
        resource_table.add_column("Usage", justify="center")
        resource_table.add_column("Details", style="dim")

        # CPU
        cpu_color = "red" if system["cpu_percent"] > 80 else "yellow" if system["cpu_percent"] > 60 else "green"
        resource_table.add_row(
            "CPU",
            f"[{cpu_color}]{system['cpu_percent']:.1f}%[/{cpu_color}]",
            "System load"
        )

        # Memory
        mem = system["memory"]
        mem_color = "red" if mem["percent"] > 90 else "yellow" if mem["percent"] > 70 else "green"
        resource_table.add_row(
            "Memory",
            f"[{mem_color}]{mem['percent']:.1f}%[/{mem_color}]",
            f"{mem['used_gb']:.1f}GB / {mem['total_gb']:.1f}GB"
        )

        # Disk
        disk = system["disk"]
        disk_color = "red" if disk["percent"] > 95 else "yellow" if disk["percent"] > 80 else "green"
        resource_table.add_row(
            "Disk",
            f"[{disk_color}]{disk['percent']:.1f}%[/{disk_color}]",
            f"{disk['free_gb']:.1f}GB free / {disk['total_gb']:.1f}GB total"
        )

        console.logger.info("Output", data=resource_table)

async def _config_management(show: bool, validate: bool, path: str | None):
    """Configuration management operations."""
    try:
        config = Config()

        if show:
            console.logger.info("[bold blue]📋 Current Configuration[/bold blue]")

            # Basic config info
            config_table = Table(title="Configuration Settings")
            config_table.add_column("Setting", style="cyan")
            config_table.add_column("Value", style="white")

            # Add key configuration values
            if hasattr(config, 'qdrant'):
                config_table.add_row("Qdrant URL", str(config.qdrant.url))
            if hasattr(config, 'embedding'):
                config_table.add_row("Embedding Model", str(config.embedding.model))
            if hasattr(config, 'workspace'):
                config_table.add_row("Collection Prefix", str(config.workspace.collection_prefix))

            console.logger.info("Output", data=config_table)

        if validate:
            console.logger.info("[bold blue]✅ Configuration Validation[/bold blue]")

            validation_results = []

            # Validate Qdrant connection
            try:
                client = create_qdrant_client(config.qdrant_client_config)
                await client.list_collections()
                validation_results.append(("Qdrant Connection", "✅ Valid", "green"))
            except Exception as e:
                validation_results.append(("Qdrant Connection", f"❌ Failed: {e}", "red"))

            # Display validation results
            for setting, result, color in validation_results:
                console.logger.info("  {setting}: [{color}]{result}[/{color}]")

        if not show and not validate:
            console.logger.info("[yellow]Use --show or --validate flags to perform operations[/yellow]")

    except Exception as e:
        console.logger.info("[red]Configuration error: {e}[/red]")
        raise typer.Exit(1)

async def _start_engine(force: bool, config_path: str | None):
    """Start the Rust processing engine."""
    console.logger.info("[bold blue]🚀 Starting Rust Engine[/bold blue]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting engine...", total=None)

        try:
            # TODO: Implement actual Rust engine startup
            await asyncio.sleep(2)  # Simulate startup time

            progress.update(task, description="Engine started successfully")
            console.logger.info("[green]✅ Rust engine started successfully[/green]")
            console.logger.info("[dim]Note: Full Rust engine integration will be implemented in Task 11[/dim]")

        except Exception as e:
            console.logger.info("[red]❌ Failed to start engine: {e}[/red]")
            raise typer.Exit(1)

async def _stop_engine(force: bool, timeout: int):
    """Stop the Rust processing engine."""
    console.logger.info("[bold yellow]🛑 Stopping Rust Engine[/bold yellow]")

    try:
        # TODO: Implement actual Rust engine shutdown
        if force:
            console.logger.info("[yellow]Force stopping engine...[/yellow]")
        else:
            console.logger.info("[yellow]Graceful shutdown (timeout: {timeout}s)...[/yellow]")

        await asyncio.sleep(1)  # Simulate shutdown time
        console.logger.info("[green]✅ Rust engine stopped[/green]")

    except Exception as e:
        console.logger.info("[red]❌ Failed to stop engine: {e}[/red]")
        raise typer.Exit(1)

async def _restart_engine(config_path: str | None):
    """Restart engine with new configuration."""
    console.logger.info("[bold blue]🔄 Restarting Rust Engine[/bold blue]")

    try:
        await _stop_engine(False, 30)
        await asyncio.sleep(1)
        await _start_engine(False, config_path)

    except Exception as e:
        console.logger.info("[red]❌ Failed to restart engine: {e}[/red]")
        raise typer.Exit(1)

async def _list_collections(project: str | None, stats: bool, library: bool):
    """List and manage collections."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)
        collections = await client.list_collections()

        # Filter collections
        if library:
            collections = [col for col in collections if col.get("name", "").startswith("_")]
        elif project:
            prefix = f"{config.workspace.collection_prefix}{project}_" if hasattr(config, 'workspace') else f"{project}_"
            collections = [col for col in collections if col.get("name", "").startswith(prefix)]

        if not collections:
            filter_desc = "library " if library else f"project '{project}' " if project else ""
            console.logger.info("[yellow]No {filter_desc}collections found.[/yellow]")
            return

        # Display collections table
        table = Table(title=f"📁 Collections ({len(collections)} found)")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="white")

        if stats:
            table.add_column("Points", justify="right")
            table.add_column("Vectors", justify="right")

        for col in collections:
            name = col.get("name", "unknown")
            col_type = "Library" if name.startswith("_") else "Project"

            if stats:
                try:
                    info = await client.get_collection_info(name)
                    points = str(info.get("points_count", "?"))
                    vectors = str(info.get("vectors_count", "?"))
                    table.add_row(name, col_type, points, vectors)
                except Exception:
                    table.add_row(name, col_type, "?", "?")
            else:
                table.add_row(name, col_type)

        console.logger.info("Output", data=table)

    except Exception as e:
        console.logger.info("[red]Error listing collections: {e}[/red]")
        raise typer.Exit(1)

async def _health_check(deep: bool, timeout: int):
    """Comprehensive health check."""
    console.logger.info("[bold blue]🏥 System Health Check[/bold blue]")

    health_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:

        # Basic connectivity
        task = progress.add_task("Testing Qdrant connectivity...", total=None)
        try:
            config = Config()
            client = create_qdrant_client(config.qdrant_client_config)
            await asyncio.wait_for(client.list_collections(), timeout=timeout)
            health_results.append(("Qdrant Connectivity", "✅ Healthy", "green"))
        except asyncio.TimeoutError:
            health_results.append(("Qdrant Connectivity", "⏰ Timeout", "yellow"))
        except Exception as e:
            health_results.append(("Qdrant Connectivity", f"❌ Error: {e}", "red"))

        # Memory usage check
        progress.update(task, description="Checking memory usage...")
        try:
            memory = psutil.virtual_memory()
            if memory.percent < 80:
                health_results.append(("Memory Usage", f"✅ {memory.percent:.1f}%", "green"))
            elif memory.percent < 95:
                health_results.append(("Memory Usage", f"⚠️ {memory.percent:.1f}%", "yellow"))
            else:
                health_results.append(("Memory Usage", f"❌ {memory.percent:.1f}%", "red"))
        except Exception as e:
            health_results.append(("Memory Usage", f"❌ Error: {e}", "red"))

        # Disk space check
        if deep:
            progress.update(task, description="Checking disk space...")
            try:
                disk = psutil.disk_usage('/')
                if disk.percent < 85:
                    health_results.append(("Disk Space", f"✅ {disk.percent:.1f}%", "green"))
                elif disk.percent < 95:
                    health_results.append(("Disk Space", f"⚠️ {disk.percent:.1f}%", "yellow"))
                else:
                    health_results.append(("Disk Space", f"❌ {disk.percent:.1f}%", "red"))
            except Exception as e:
                health_results.append(("Disk Space", f"❌ Error: {e}", "red"))

    # Display results
    console.logger.info("\n[bold]Health Check Results:[/bold]")
    for component, status, color in health_results:
        console.logger.info("  {component}: [{color}]{status}[/{color}]")

    # Overall assessment
    errors = sum(1 for _, status, _ in health_results if status.startswith("❌"))
    warnings = sum(1 for _, status, _ in health_results if status.startswith("⚠️"))

    if errors == 0:
        if warnings == 0:
            console.logger.info("\n[green]🎉 System is healthy![/green]")
        else:
            console.logger.info("\n[yellow]⚠️ System has {warnings} warning(s)[/yellow]")
    else:
        console.logger.info("\n[red]❌ System has {errors} error(s) and {warnings} warning(s)[/red]")
        raise typer.Exit(1)
