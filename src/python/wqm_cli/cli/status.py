"""
Status CLI tools for workspace-qdrant-mcp.

Provides comprehensive status reporting and user feedback for:
- Active processing status with real-time progress
- Processing history with detailed categorization
- Watch folder monitoring and health checks
- Performance metrics and resource usage
- Queue management and analytics

Commands:
    wqm status                           # Show current processing status
    wqm status --history                 # Show processing history
    wqm status --queue                   # Show queue status and statistics
    wqm status --watch                   # Show watch folder status
    wqm status --performance             # Show performance metrics
    wqm status --export json             # Export status data
    wqm status --live                    # Live status monitoring

Examples:
    ```bash
    # Basic status overview
    wqm status

    # Show processing history from last 7 days
    wqm status --history --days 7

    # Live monitoring with 10-second updates
    wqm status --live --interval 10

    # Export detailed status to JSON file
    wqm status --export json --output status_report.json

    # Filter by collection and show failed files
    wqm status --history --collection docs --status failed
    ```
"""

import asyncio
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from common.core.client import QdrantWorkspaceClient
from common.core.config import Config
from loguru import logger
# TEMPORARY FIX: Comment out grpc_tools import that causes CLI to hang
# This needs to be fixed properly by making grpc_tools imports lazy or conditional
# from workspace_qdrant_mcp.tools.grpc_tools import (
#     get_grpc_engine_stats,
#     stream_processing_status_grpc,
#     stream_queue_status_grpc,
#     stream_system_metrics_grpc,
#     test_grpc_connection,
# )
# TEMPORARY FIX: Comment out state_management import that causes hang
# This needs to be fixed by resolving the state_aware_ingestion import issue
# from workspace_qdrant_mcp.tools.state_management import (
#     get_database_stats,
#     get_failed_files,
#     get_processing_analytics,
#     get_processing_status,
#     get_queue_stats,
#     get_watch_folder_configs,
# )
# TEMPORARY FIX: Comment out watch_management import that also causes hang
# from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager

# Temporary stub for WatchToolsManager
class WatchToolsManager:
    """Temporary stub for WatchToolsManager."""
    def __init__(self, *args, **kwargs):
        pass

# logger imported from loguru

# TEMPORARY FIX: Stub functions for disabled state_management imports
async def get_processing_status(workspace_client, watch_manager):
    """Temporary stub for get_processing_status."""
    return {"status": "disabled", "message": "State management temporarily disabled"}

async def get_queue_stats(workspace_client, watch_manager):
    """Temporary stub for get_queue_stats."""
    return {"queue_size": 0, "status": "disabled", "message": "Queue stats temporarily disabled"}

async def get_watch_folder_configs(workspace_client, watch_manager):
    """Temporary stub for get_watch_folder_configs."""
    return {"configs": [], "status": "disabled", "message": "Watch configs temporarily disabled"}

async def get_database_stats(workspace_client, watch_manager):
    """Temporary stub for get_database_stats."""
    return {"database_size": "unknown", "status": "disabled", "message": "Database stats temporarily disabled"}

# CLI app for status commands
status_app = typer.Typer(
    name="status",
    help="Processing status and user feedback system",
    no_args_is_help=False,
)

console = Console()


def format_timestamp(timestamp: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)

    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1

    return f"{size:.1f} {size_names[i]}"


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def create_status_overview(
    processing_status: Dict[str, Any],
    queue_stats: Dict[str, Any],
    grpc_stats: Dict[str, Any] = None,
) -> Panel:
    """Create status overview panel."""

    # Processing overview
    processing_info = processing_status.get("processing_info", {})
    queue_info = queue_stats.get("queue_stats", {})

    overview_table = Table(show_header=False, box=None, padding=(0, 2))
    overview_table.add_column("Metric", style="cyan", min_width=20)
    overview_table.add_column("Value", style="white")

    # Active processing
    active_count = processing_info.get("currently_processing", 0)
    overview_table.add_row("Active Processing", str(active_count))

    # Queue depth
    total_queued = queue_info.get("total", 0)
    overview_table.add_row("Queue Depth", str(total_queued))

    # Recent completions (last hour)
    recent_successful = processing_info.get("recent_successful", 0)
    recent_failed = processing_info.get("recent_failed", 0)
    overview_table.add_row("Recent Success", str(recent_successful))
    overview_table.add_row("Recent Failed", str(recent_failed))

    # Processing rate
    if grpc_stats and grpc_stats.get("success"):
        engine_stats = grpc_stats.get("stats", {}).get("engine_stats", {})
        total_processed = engine_stats.get("total_documents_processed", 0)
        uptime = engine_stats.get("uptime_seconds", 1)
        rate = total_processed / (uptime / 3600) if uptime > 0 else 0
        overview_table.add_row("Processing Rate", f"{rate:.1f} files/hour")

    # Daemon status
    daemon_status = "Running" if grpc_stats and grpc_stats.get("success") else "Offline"
    status_style = "green" if daemon_status == "Running" else "red"
    overview_table.add_row(
        "Daemon Status", f"[{status_style}]{daemon_status}[/{status_style}]"
    )

    return Panel(
        overview_table,
        title="[bold]Processing Status Overview[/bold]",
        border_style="blue",
    )


def create_queue_breakdown(queue_stats: Dict[str, Any]) -> Panel:
    """Create queue breakdown panel."""
    queue_info = queue_stats.get("queue_stats", {})

    if queue_info.get("total", 0) == 0:
        return Panel(
            Text("No files currently in processing queue", style="dim italic"),
            title="[bold]Processing Queue[/bold]",
            border_style="yellow",
        )

    queue_table = Table(show_header=True, box=None)
    queue_table.add_column("Priority", style="cyan")
    queue_table.add_column("Files", justify="right", style="white")
    queue_table.add_column("Collections", style="dim")

    priorities = ["urgent", "high", "normal", "low"]
    for priority in priorities:
        count = queue_info.get(f"{priority}_priority", 0)
        if count > 0:
            collections = queue_info.get(f"{priority}_collections", [])
            collections_str = ", ".join(collections[:3])
            if len(collections) > 3:
                collections_str += f" (+{len(collections) - 3} more)"

            queue_table.add_row(priority.title(), str(count), collections_str)

    total = queue_info.get("total", 0)
    queue_table.add_row("", "", "", style="dim")
    queue_table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]", "", style="bold")

    return Panel(
        queue_table,
        title="[bold]Processing Queue Breakdown[/bold]",
        border_style="yellow",
    )


def create_recent_activity(processing_status: Dict[str, Any]) -> Panel:
    """Create recent activity panel."""
    recent_files = processing_status.get("recent_files", [])

    if not recent_files:
        return Panel(
            Text("No recent processing activity", style="dim italic"),
            title="[bold]Recent Activity[/bold]",
            border_style="green",
        )

    activity_table = Table(show_header=True, box=None)
    activity_table.add_column("Time", style="dim", max_width=16)
    activity_table.add_column("File", style="cyan", max_width=30)
    activity_table.add_column("Status", style="white", max_width=10)
    activity_table.add_column("Collection", style="dim", max_width=15)
    activity_table.add_column("Duration", style="white", max_width=8)

    for file_info in recent_files[:10]:  # Show last 10 files
        filename = Path(file_info.get("file_path", "")).name
        if len(filename) > 30:
            filename = filename[:27] + "..."

        status = file_info.get("status", "unknown")
        status_color = {
            "completed": "green",
            "failed": "red",
            "processing": "yellow",
            "skipped": "dim",
        }.get(status, "white")

        duration = file_info.get("processing_duration")
        duration_str = format_duration(duration) if duration else "-"

        activity_table.add_row(
            format_timestamp(file_info.get("timestamp", "")),
            filename,
            f"[{status_color}]{status}[/{status_color}]",
            file_info.get("collection", ""),
            duration_str,
        )

    return Panel(
        activity_table,
        title="[bold]Recent Processing Activity[/bold]",
        border_style="green",
    )


def create_watch_status(watch_configs: List[Dict[str, Any]]) -> Panel:
    """Create watch folder status panel."""
    if not watch_configs:
        return Panel(
            Text("No watch folders configured", style="dim italic"),
            title="[bold]Watch Folders[/bold]",
            border_style="purple",
        )

    watch_table = Table(show_header=True, box=None)
    watch_table.add_column("Path", style="cyan", max_width=40)
    watch_table.add_column("Collection", style="white", max_width=15)
    watch_table.add_column("Status", style="white", max_width=10)
    watch_table.add_column("Last Scan", style="dim", max_width=16)
    watch_table.add_column("Files", justify="right", style="white")

    for config in watch_configs[:8]:  # Show first 8 watch folders
        path = config.get("path", "")
        if len(path) > 40:
            path = "..." + path[-37:]

        status = "Active" if config.get("enabled") else "Disabled"
        status_color = "green" if config.get("enabled") else "dim"

        last_scan = config.get("last_scan")
        last_scan_str = format_timestamp(last_scan) if last_scan else "Never"

        # TODO: Get actual file counts from watch folder stats
        file_count = "-"

        watch_table.add_row(
            path,
            config.get("collection", ""),
            f"[{status_color}]{status}[/{status_color}]",
            last_scan_str,
            file_count,
        )

    return Panel(
        watch_table, title="[bold]Watch Folder Status[/bold]", border_style="purple"
    )


def create_performance_metrics(
    grpc_stats: Dict[str, Any], db_stats: Dict[str, Any]
) -> Panel:
    """Create performance metrics panel."""
    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Metric", style="cyan", min_width=25)
    metrics_table.add_column("Value", style="white")

    if grpc_stats and grpc_stats.get("success"):
        engine_stats = grpc_stats.get("stats", {}).get("engine_stats", {})

        # Uptime
        uptime_seconds = engine_stats.get("uptime_seconds", 0)
        metrics_table.add_row("Daemon Uptime", format_duration(uptime_seconds))

        # Total processed
        total_processed = engine_stats.get("total_documents_processed", 0)
        metrics_table.add_row("Total Documents Processed", f"{total_processed:,}")

        # Active watches
        active_watches = engine_stats.get("active_watches", 0)
        metrics_table.add_row("Active Watch Folders", str(active_watches))

        # Version
        version = engine_stats.get("version", "Unknown")
        metrics_table.add_row("Engine Version", version)
    else:
        metrics_table.add_row("Daemon Status", "[red]Offline[/red]")
        metrics_table.add_row("Connection", "[red]Cannot connect to gRPC daemon[/red]")

    # Database metrics
    if db_stats and db_stats.get("success"):
        database_stats = db_stats.get("database_stats", {})

        size_mb = database_stats.get("total_size_mb", 0)
        metrics_table.add_row("Database Size", f"{size_mb:.1f} MB")

        total_records = database_stats.get("total_records", 0)
        metrics_table.add_row("Total State Records", f"{total_records:,}")

        # Processing stats from last 24h
        processing_stats = database_stats.get("recent_processing", {})
        if processing_stats:
            successful_24h = processing_stats.get("successful_24h", 0)
            failed_24h = processing_stats.get("failed_24h", 0)

            metrics_table.add_row(
                "24h Success Rate",
                f"{successful_24h}/{successful_24h + failed_24h} "
                + f"({(successful_24h / (successful_24h + failed_24h) * 100) if (successful_24h + failed_24h) > 0 else 0:.1f}%)",
            )

    return Panel(
        metrics_table, title="[bold]Performance Metrics[/bold]", border_style="magenta"
    )


async def get_comprehensive_status() -> Dict[str, Any]:
    """Get comprehensive status data from all sources."""
    try:
        # Initialize clients
        config = Config.get_default_config()
        workspace_client = QdrantWorkspaceClient(config.qdrant)
        watch_manager = WatchToolsManager(workspace_client)

        # Gather data in parallel
        tasks = []

        # Processing status from SQLite
        tasks.append(get_processing_status(workspace_client, watch_manager))

        # Queue statistics
        tasks.append(get_queue_stats(workspace_client, watch_manager))

        # Watch folder configurations
        tasks.append(get_watch_folder_configs(workspace_client, watch_manager))

        # Database statistics
        tasks.append(get_database_stats(workspace_client, watch_manager))

        # gRPC daemon stats
        # TEMPORARY FIX: Skip grpc stats due to import hang
        # tasks.append(get_grpc_engine_stats())
        pass

        # Execute all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            "processing_status": results[0]
            if not isinstance(results[0], Exception)
            else {"success": False, "error": str(results[0])},
            "queue_stats": results[1]
            if not isinstance(results[1], Exception)
            else {"success": False, "error": str(results[1])},
            "watch_configs": results[2]
            if not isinstance(results[2], Exception)
            else {"success": False, "error": str(results[2])},
            "database_stats": results[3]
            if not isinstance(results[3], Exception)
            else {"success": False, "error": str(results[3])},
            "grpc_stats": results[4]
            if not isinstance(results[4], Exception)
            else {"success": False, "error": str(results[4])},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error("Failed to get comprehensive status", error=str(e))
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@status_app.callback(invoke_without_command=True)
def status_main(
    ctx: typer.Context,
    history: bool = typer.Option(False, "--history", help="Show processing history"),
    queue: bool = typer.Option(False, "--queue", help="Show detailed queue statistics"),
    watch: bool = typer.Option(False, "--watch", help="Show watch folder status"),
    performance: bool = typer.Option(
        False, "--performance", help="Show performance metrics"
    ),
    live: bool = typer.Option(False, "--live", help="Enable live monitoring mode"),
    stream: bool = typer.Option(
        False, "--stream", help="Enable real-time gRPC streaming (requires daemon)"
    ),
    interval: int = typer.Option(
        5, "--interval", "-i", help="Live update interval in seconds"
    ),
    grpc_host: str = typer.Option("127.0.0.1", "--grpc-host", help="gRPC daemon host"),
    grpc_port: int = typer.Option(50051, "--grpc-port", help="gRPC daemon port"),
    export: Optional[str] = typer.Option(
        None, "--export", help="Export format: json, csv"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path"
    ),
    collection: Optional[str] = typer.Option(
        None, "--collection", "-c", help="Filter by collection"
    ),
    status_filter: Optional[str] = typer.Option(
        None, "--status", help="Filter by status: success, failed, skipped, pending"
    ),
    days: int = typer.Option(
        7, "--days", "-d", help="Number of days for history (default: 7)"
    ),
    limit: int = typer.Option(
        100, "--limit", "-l", help="Maximum number of records to show"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
) -> None:
    """
    Show comprehensive processing status and user feedback.

    By default, shows current processing overview with active operations,
    queue status, recent activity, and system health.

    Use specific flags to focus on particular aspects of the system status.
    """
    if not ctx.invoked_subcommand:
        # Main status display
        if live or stream:
            if stream:
                asyncio.run(
                    live_streaming_status_monitor(
                        interval, collection, grpc_host, grpc_port
                    )
                )
            else:
                asyncio.run(live_status_monitor(interval, collection))
        else:
            asyncio.run(
                show_status_overview(
                    history=history,
                    queue=queue,
                    watch=watch,
                    performance=performance,
                    export=export,
                    output=output,
                    collection=collection,
                    status_filter=status_filter,
                    days=days,
                    limit=limit,
                    verbose=verbose,
                    quiet=quiet,
                )
            )


async def show_status_overview(
    history: bool = False,
    queue: bool = False,
    watch: bool = False,
    performance: bool = False,
    export: Optional[str] = None,
    output: Optional[Path] = None,
    collection: Optional[str] = None,
    status_filter: Optional[str] = None,
    days: int = 7,
    limit: int = 100,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    """Show status overview with optional detailed sections."""

    if not quiet:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Gathering status information...", total=None)

            status_data = await get_comprehensive_status()

            progress.update(task, completed=True)
    else:
        status_data = await get_comprehensive_status()

    # Handle export first
    if export:
        await export_status_data(
            status_data, export, output, collection, status_filter, days, limit
        )
        return

    if not quiet:
        console.print()  # Add spacing

    # Always show basic overview unless in quiet mode
    if not quiet:
        # Create main layout
        layout = Layout()

        if history or queue or watch or performance:
            # Focused view - show only requested sections
            sections = []

            if history:
                sections.append(
                    create_processing_history_panel(
                        status_data, collection, status_filter, days, limit
                    )
                )
            if queue:
                sections.append(
                    create_queue_breakdown(status_data.get("queue_stats", {}))
                )
            if watch:
                sections.append(
                    create_watch_status(
                        status_data.get("watch_configs", {}).get("watch_configs", [])
                    )
                )
            if performance:
                sections.append(
                    create_performance_metrics(
                        status_data.get("grpc_stats", {}),
                        status_data.get("database_stats", {}),
                    )
                )

            # Display sections vertically
            for section in sections:
                console.print(section)
                console.print()  # Add spacing between sections

        else:
            # Default comprehensive view
            layout.split_column(
                Layout(name="header", size=8),
                Layout(name="main"),
                Layout(name="footer", size=6),
            )

            # Header: Status overview
            layout["header"].update(
                create_status_overview(
                    status_data.get("processing_status", {}),
                    status_data.get("queue_stats", {}),
                    status_data.get("grpc_stats", {}),
                )
            )

            # Main: Split between activity and queue
            layout["main"].split_row(
                Layout(
                    create_recent_activity(status_data.get("processing_status", {})),
                    ratio=2,
                ),
                Layout(
                    create_queue_breakdown(status_data.get("queue_stats", {})), ratio=1
                ),
            )

            # Footer: Performance metrics
            layout["footer"].update(
                create_performance_metrics(
                    status_data.get("grpc_stats", {}),
                    status_data.get("database_stats", {}),
                )
            )

            console.print(layout)
    else:
        # Quiet mode - minimal output
        processing_status = status_data.get("processing_status", {})
        queue_stats = status_data.get("queue_stats", {})
        grpc_stats = status_data.get("grpc_stats", {})

        processing_info = processing_status.get("processing_info", {})
        queue_info = queue_stats.get("queue_stats", {})

        daemon_status = (
            "online" if grpc_stats and grpc_stats.get("success") else "offline"
        )
        active_processing = processing_info.get("currently_processing", 0)
        queue_depth = queue_info.get("total", 0)
        recent_failed = processing_info.get("recent_failed", 0)

        print(
            f"daemon:{daemon_status} active:{active_processing} queued:{queue_depth} failed:{recent_failed}"
        )


def create_processing_history_panel(
    status_data: Dict[str, Any],
    collection: Optional[str] = None,
    status_filter: Optional[str] = None,
    days: int = 7,
    limit: int = 100,
) -> Panel:
    """Create processing history panel."""
    # This would typically get more detailed history from SQLite
    # For now, use the recent activity data
    processing_status = status_data.get("processing_status", {})
    recent_files = processing_status.get("recent_files", [])

    # Filter if needed
    filtered_files = recent_files
    if collection:
        filtered_files = [
            f for f in filtered_files if f.get("collection") == collection
        ]
    if status_filter:
        filtered_files = [f for f in filtered_files if f.get("status") == status_filter]

    if not filtered_files:
        filter_desc = []
        if collection:
            filter_desc.append(f"collection '{collection}'")
        if status_filter:
            filter_desc.append(f"status '{status_filter}'")

        message = "No processing history found"
        if filter_desc:
            message += f" for {' and '.join(filter_desc)}"

        return Panel(
            Text(message, style="dim italic"),
            title="[bold]Processing History[/bold]",
            border_style="blue",
        )

    history_table = Table(show_header=True, box=None)
    history_table.add_column("Timestamp", style="dim", max_width=20)
    history_table.add_column("File Path", style="cyan", max_width=50)
    history_table.add_column("Status", style="white", max_width=10)
    history_table.add_column("Collection", style="white", max_width=15)
    history_table.add_column("Duration", style="dim", max_width=10)
    history_table.add_column("Error", style="red", max_width=30)

    for file_info in filtered_files[:limit]:
        filename = file_info.get("file_path", "")
        if len(filename) > 50:
            filename = "..." + filename[-47:]

        status = file_info.get("status", "unknown")
        status_color = {
            "completed": "green",
            "failed": "red",
            "processing": "yellow",
            "skipped": "dim",
        }.get(status, "white")

        duration = file_info.get("processing_duration")
        duration_str = format_duration(duration) if duration else "-"

        error_msg = file_info.get("error_message", "")
        if error_msg and len(error_msg) > 30:
            error_msg = error_msg[:27] + "..."

        history_table.add_row(
            format_timestamp(file_info.get("timestamp", "")),
            filename,
            f"[{status_color}]{status}[/{status_color}]",
            file_info.get("collection", ""),
            duration_str,
            error_msg if status == "failed" else "",
        )

    title = f"[bold]Processing History (last {days} days)[/bold]"
    if collection or status_filter:
        filters = []
        if collection:
            filters.append(f"collection: {collection}")
        if status_filter:
            filters.append(f"status: {status_filter}")
        title += f" - {', '.join(filters)}"

    return Panel(history_table, title=title, border_style="blue")


async def export_status_data(
    status_data: Dict[str, Any],
    export_format: str,
    output_path: Optional[Path] = None,
    collection: Optional[str] = None,
    status_filter: Optional[str] = None,
    days: int = 7,
    limit: int = 100,
) -> None:
    """Export status data in specified format."""

    if export_format.lower() == "json":
        export_data = {
            "export_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "format": "json",
                "filters": {
                    "collection": collection,
                    "status": status_filter,
                    "days": days,
                    "limit": limit,
                },
            },
            "status_data": status_data,
        }

        json_output = json.dumps(export_data, indent=2, default=str)

        if output_path:
            output_path.write_text(json_output)
            console.print(f"[green]Status data exported to {output_path}[/green]")
        else:
            console.print(json_output)

    elif export_format.lower() == "csv":
        # For CSV, export processing history as a flat table
        import csv
        import io

        processing_status = status_data.get("processing_status", {})
        recent_files = processing_status.get("recent_files", [])

        # Filter if needed
        filtered_files = recent_files
        if collection:
            filtered_files = [
                f for f in filtered_files if f.get("collection") == collection
            ]
        if status_filter:
            filtered_files = [
                f for f in filtered_files if f.get("status") == status_filter
            ]

        output = io.StringIO()
        writer = csv.writer(output)

        # Write headers
        writer.writerow(
            [
                "timestamp",
                "file_path",
                "status",
                "collection",
                "processing_duration",
                "error_message",
                "chunks_added",
            ]
        )

        # Write data
        for file_info in filtered_files[:limit]:
            writer.writerow(
                [
                    file_info.get("timestamp", ""),
                    file_info.get("file_path", ""),
                    file_info.get("status", ""),
                    file_info.get("collection", ""),
                    file_info.get("processing_duration", ""),
                    file_info.get("error_message", ""),
                    file_info.get("chunks_added", ""),
                ]
            )

        csv_output = output.getvalue()

        if output_path:
            output_path.write_text(csv_output)
            console.print(
                f"[green]Processing history exported to {output_path}[/green]"
            )
        else:
            console.print(csv_output)

    else:
        console.print(
            f"[red]Unsupported export format: {export_format}. Use 'json' or 'csv'.[/red]"
        )
        raise typer.Exit(1)


async def live_status_monitor(
    interval: int = 5, collection: Optional[str] = None
) -> None:
    """Run live status monitoring with real-time updates."""

    def create_live_layout(status_data: Dict[str, Any]) -> Layout:
        """Create live monitoring layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=2),
        )

        # Header with timestamp and refresh info
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        grpc_stats = status_data.get("grpc_stats", {})
        daemon_status = (
            "ONLINE" if grpc_stats and grpc_stats.get("success") else "OFFLINE"
        )
        status_color = "green" if daemon_status == "ONLINE" else "red"

        header_text = Text()
        header_text.append("Live Status Monitor - ", style="bold")
        header_text.append(daemon_status, style=f"bold {status_color}")
        header_text.append(f" - {current_time} (refresh: {interval}s)", style="dim")

        layout["header"].update(Panel(header_text, border_style="blue"))

        # Main content: status overview and recent activity
        layout["main"].split_row(
            Layout(
                create_status_overview(
                    status_data.get("processing_status", {}),
                    status_data.get("queue_stats", {}),
                    grpc_stats,
                ),
                ratio=1,
            ),
            Layout(
                create_recent_activity(status_data.get("processing_status", {})),
                ratio=2,
            ),
        )

        # Footer: Queue and performance summary
        queue_info = status_data.get("queue_stats", {}).get("queue_stats", {})
        db_stats = status_data.get("database_stats", {})

        footer_text = f"Queue: {queue_info.get('total', 0)} files | "
        if db_stats and db_stats.get("success"):
            database_stats = db_stats.get("database_stats", {})
            footer_text += f"DB: {database_stats.get('total_size_mb', 0):.1f}MB | "

        footer_text += f"Press Ctrl+C to exit"

        layout["footer"].update(Panel(footer_text, style="dim", border_style="dim"))

        return layout

    console.print(
        f"[cyan]Starting live status monitoring (interval: {interval}s)[/cyan]"
    )
    if collection:
        console.print(f"[dim]Filtered to collection: {collection}[/dim]")
    console.print("Press Ctrl+C to exit\n")

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                try:
                    status_data = await get_comprehensive_status()
                    layout = create_live_layout(status_data)
                    live.update(layout)

                    await asyncio.sleep(interval)

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error updating status: {e}[/red]", err=True)
                    await asyncio.sleep(interval)

    except KeyboardInterrupt:
        console.print("\n[yellow]Live monitoring stopped by user[/yellow]")


async def live_streaming_status_monitor(
    interval: int = 5,
    collection: Optional[str] = None,
    grpc_host: str = "127.0.0.1",
    grpc_port: int = 50051,
) -> None:
    """Run live status monitoring with real-time gRPC streaming updates."""

    def create_streaming_layout(
        processing_update: Optional[Dict[str, Any]] = None,
        metrics_update: Optional[Dict[str, Any]] = None,
        queue_update: Optional[Dict[str, Any]] = None,
    ) -> Layout:
        """Create live streaming layout with real-time data."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=2),
        )

        # Header with real-time timestamp
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        daemon_status = (
            "STREAMING"
            if processing_update or metrics_update or queue_update
            else "CONNECTING"
        )
        status_color = "green" if daemon_status == "STREAMING" else "yellow"

        header_text = Text()
        header_text.append("Live Streaming Monitor - ", style="bold")
        header_text.append(daemon_status, style=f"bold {status_color}")
        header_text.append(f" - {current_time} (gRPC streaming)", style="dim")

        layout["header"].update(Panel(header_text, border_style="blue"))

        # Main content split between processing status and metrics
        layout["main"].split_row(
            Layout(name="processing", ratio=2), Layout(name="metrics", ratio=1)
        )

        # Processing status panel
        if processing_update:
            processing_panel = create_streaming_processing_panel(processing_update)
        else:
            processing_panel = Panel(
                Text("Waiting for processing status updates...", style="dim italic"),
                title="[bold]Processing Status[/bold]",
                border_style="blue",
            )

        layout["main"]["processing"].update(processing_panel)

        # Metrics panel
        if metrics_update:
            metrics_panel = create_streaming_metrics_panel(metrics_update)
        else:
            metrics_panel = Panel(
                Text("Waiting for system metrics...", style="dim italic"),
                title="[bold]System Metrics[/bold]",
                border_style="magenta",
            )

        layout["main"]["metrics"].update(metrics_panel)

        # Footer with queue info
        footer_text = "Real-time gRPC streaming active | "
        if queue_update and queue_update.get("queue_status"):
            queue_status = queue_update["queue_status"]
            footer_text += f"Queue: {queue_status.get('total_queued', 0)} files | "

        footer_text += "Press Ctrl+C to exit"

        layout["footer"].update(Panel(footer_text, style="dim", border_style="dim"))

        return layout

    # Check if gRPC daemon is available
    console.print(f"[cyan]Testing gRPC connection to {grpc_host}:{grpc_port}...[/cyan]")
    # TEMPORARY FIX: Skip grpc connection test due to import hang
    # connection_result = await test_grpc_connection(grpc_host, grpc_port, timeout=5.0)
    connection_result = {"connected": False, "error": "gRPC tools temporarily disabled"}

    if not connection_result.get("connected"):
        console.print(
            f"[red]Error: Cannot connect to gRPC daemon at {grpc_host}:{grpc_port}[/red]"
        )
        console.print(
            f"[dim]Error: {connection_result.get('error', 'Connection failed')}[/dim]"
        )
        console.print("[yellow]Falling back to polling-based monitoring...[/yellow]\n")
        # Fall back to regular monitoring
        await live_status_monitor(interval, collection)
        return

    console.print(f"[green]✓ Connected to gRPC daemon[/green]")
    console.print(f"[cyan]Starting real-time streaming monitor[/cyan]")
    if collection:
        console.print(f"[dim]Filtered to collection: {collection}[/dim]")
    console.print("Press Ctrl+C to exit\n")

    # Storage for latest updates
    latest_processing = None
    latest_metrics = None
    latest_queue = None

    async def update_processing_status():
        """Background task to stream processing status updates."""
        nonlocal latest_processing
        try:
            # TEMPORARY FIX: Skip grpc streaming due to import hang
            # Original call was:
            # result = await stream_processing_status_grpc(
            #     host=grpc_host,
            #     port=grpc_port,
            #     update_interval=interval,
            #     include_history=True,
            #     collection_filter=collection,
            # )
            result = {"success": False, "error": "gRPC tools temporarily disabled"}
            if result.get("success") and result.get("status_updates"):
                for update in result["status_updates"]:
                    latest_processing = update
        except Exception as e:
            logger.warning("Processing status streaming failed", error=str(e))

    async def update_system_metrics():
        """Background task to stream system metrics updates."""
        nonlocal latest_metrics
        try:
            # TEMPORARY FIX: Skip grpc streaming due to import hang
            # Original call was:
            # result = await stream_system_metrics_grpc(
            #     host=grpc_host,
            #     port=grpc_port,
            #     update_interval=max(interval, 10),  # Metrics update less frequently
            #     include_detailed_metrics=True,
            # )
            result = {"success": False, "error": "gRPC tools temporarily disabled"}
            if result.get("success") and result.get("metrics_updates"):
                for update in result["metrics_updates"]:
                    latest_metrics = update
        except Exception as e:
            logger.warning("System metrics streaming failed", error=str(e))

    async def update_queue_status():
        """Background task to stream queue status updates."""
        nonlocal latest_queue
        try:
            # TEMPORARY FIX: Skip grpc streaming due to import hang
            # Original call was:
            # result = await stream_queue_status_grpc(
            #     host=grpc_host,
            #     port=grpc_port,
            #     update_interval=min(interval, 5),  # Queue updates more frequently
            #     collection_filter=collection,
            # )
            result = {"success": False, "error": "gRPC tools temporarily disabled"}
            if result.get("success") and result.get("queue_updates"):
                for update in result["queue_updates"]:
                    latest_queue = update
        except Exception as e:
            logger.warning("Queue status streaming failed", error=str(e))

    try:
        # Start background streaming tasks
        processing_task = asyncio.create_task(update_processing_status())
        metrics_task = asyncio.create_task(update_system_metrics())
        queue_task = asyncio.create_task(update_queue_status())

        with Live(console=console, refresh_per_second=2) as live:
            while True:
                try:
                    # Create layout with latest data
                    layout = create_streaming_layout(
                        latest_processing, latest_metrics, latest_queue
                    )
                    live.update(layout)

                    await asyncio.sleep(1)  # Update display every second

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print(f"[red]Error updating display: {e}[/red]", err=True)
                    await asyncio.sleep(1)

        # Cancel background tasks
        processing_task.cancel()
        metrics_task.cancel()
        queue_task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(
            processing_task, metrics_task, queue_task, return_exceptions=True
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Streaming monitor stopped by user[/yellow]")


def create_streaming_processing_panel(processing_update: Dict[str, Any]) -> Panel:
    """Create a panel for streaming processing status."""

    processing_table = Table(show_header=True, box=None)
    processing_table.add_column("Metric", style="cyan", min_width=20)
    processing_table.add_column("Value", style="white")

    # Current stats
    current_stats = processing_update.get("current_stats", {})
    active_tasks = len(processing_update.get("active_tasks", []))
    recent_completed = len(processing_update.get("recent_completed", []))

    processing_table.add_row("Active Processing", str(active_tasks))
    processing_table.add_row("Recently Completed", str(recent_completed))

    if current_stats:
        processing_table.add_row(
            "Total Processed", f"{current_stats.get('total_files_processed', 0):,}"
        )
        processing_table.add_row(
            "Total Failed", str(current_stats.get("total_files_failed", 0))
        )
        processing_table.add_row(
            "Total Skipped", str(current_stats.get("total_files_skipped", 0))
        )
        processing_table.add_row(
            "Queued Tasks", str(current_stats.get("queued_tasks", 0))
        )

    # Add active task details if available
    active_tasks_list = processing_update.get("active_tasks", [])
    if active_tasks_list:
        processing_table.add_row("", "")  # Separator
        processing_table.add_row("[bold]Active Files:[/bold]", "")

        for task in active_tasks_list[:3]:  # Show first 3
            filename = Path(task.get("file_path", "")).name
            if len(filename) > 25:
                filename = filename[:22] + "..."
            progress = task.get("progress_percent", 0)
            processing_table.add_row(f"  {filename}", f"{progress:.1f}%")

        if len(active_tasks_list) > 3:
            processing_table.add_row(f"  (+{len(active_tasks_list) - 3} more)", "")

    return Panel(
        processing_table,
        title="[bold]Real-time Processing Status[/bold]",
        border_style="blue",
    )


def create_streaming_metrics_panel(metrics_update: Dict[str, Any]) -> Panel:
    """Create a panel for streaming system metrics."""

    metrics_table = Table(show_header=False, box=None, padding=(0, 2))
    metrics_table.add_column("Metric", style="cyan", min_width=20)
    metrics_table.add_column("Value", style="white")

    # Resource usage
    resource_usage = metrics_update.get("resource_usage", {})
    if resource_usage:
        metrics_table.add_row(
            "CPU Usage", f"{resource_usage.get('cpu_percent', 0):.1f}%"
        )

        memory_mb = resource_usage.get("memory_bytes", 0) / 1024 / 1024
        metrics_table.add_row("Memory Usage", f"{memory_mb:.1f} MB")

        metrics_table.add_row("Open Files", str(resource_usage.get("open_files", 0)))
        metrics_table.add_row(
            "Connections", str(resource_usage.get("active_connections", 0))
        )

    # Engine stats
    engine_stats = metrics_update.get("engine_stats", {})
    if engine_stats:
        metrics_table.add_row("", "")  # Separator
        uptime_seconds = engine_stats.get("uptime_seconds", 0)
        metrics_table.add_row("Engine Uptime", format_duration(uptime_seconds))

        total_docs = engine_stats.get("total_documents_processed", 0)
        metrics_table.add_row("Total Documents", f"{total_docs:,}")

        active_watches = engine_stats.get("active_watches", 0)
        metrics_table.add_row("Active Watches", str(active_watches))

    # Performance metrics
    performance_metrics = metrics_update.get("performance_metrics", {})
    if performance_metrics:
        processing_rate = performance_metrics.get("processing_rate_files_per_hour", 0)
        metrics_table.add_row("Processing Rate", f"{processing_rate:.1f} files/hour")

        success_rate = performance_metrics.get("success_rate_percent", 0)
        metrics_table.add_row("Success Rate", f"{success_rate:.1f}%")

        concurrent_tasks = performance_metrics.get("concurrent_tasks", 0)
        metrics_table.add_row("Concurrent Tasks", str(concurrent_tasks))

    return Panel(
        metrics_table,
        title="[bold]Real-time System Metrics[/bold]",
        border_style="magenta",
    )


# Export the status CLI app
__all__ = ["status_app"]
