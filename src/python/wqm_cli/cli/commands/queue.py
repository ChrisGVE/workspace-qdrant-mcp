"""Queue management CLI commands.

This module provides commands to monitor queue status and statistics
from the workspace-qdrant-mcp daemon.

Usage:
    wqm queue status                    # Show queue statistics
    wqm queue status --tenant-id=foo    # Filter by tenant
    wqm queue status --format=json      # JSON output
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from ..utils import (
    create_command_app,
    error_message,
    handle_async,
)

# Create the queue app
queue_app = create_command_app(
    name="queue",
    help_text="""Monitor queue status and statistics.

View queue depths, processing statistics, and wait times for different
queue types in the daemon.

Examples:
    wqm queue status                     # Overall queue statistics
    wqm queue status --tenant-id=default # Filter by tenant
    wqm queue status --format=json       # JSON output
    wqm queue status --verbose           # Show detailed breakdowns""",
    no_args_is_help=True,
)


def get_daemon_db_path() -> Path:
    """Get path to daemon SQLite database.

    Returns:
        Path to state.db file in daemon state directory.

    Raises:
        FileNotFoundError: If database file doesn't exist.
    """
    config_dir = Path.home() / ".config" / "workspace-qdrant"
    db_path = config_dir / "state.db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Daemon database not found at {db_path}. "
            "Is the daemon running? Try: wqm service status"
        )

    return db_path


def format_timestamp(timestamp: Optional[str]) -> str:
    """Format timestamp for display.

    Args:
        timestamp: ISO 8601 timestamp string or None

    Returns:
        Formatted timestamp string or "N/A"
    """
    if not timestamp:
        return "N/A"

    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return timestamp


def calculate_wait_time(timestamp: Optional[str]) -> str:
    """Calculate wait time from timestamp to now.

    Args:
        timestamp: ISO 8601 timestamp string

    Returns:
        Human-readable wait time string
    """
    if not timestamp:
        return "N/A"

    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        delta = now - dt

        if delta.days > 0:
            return f"{delta.days}d {delta.seconds // 3600}h"
        elif delta.seconds >= 3600:
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        elif delta.seconds >= 60:
            return f"{delta.seconds // 60}m"
        else:
            return f"{delta.seconds}s"
    except (ValueError, AttributeError):
        return "N/A"


@queue_app.command("status")
def queue_status(
    tenant_id: Optional[str] = typer.Option(
        None,
        "--tenant-id",
        help="Filter by tenant ID",
    ),
    priority: Optional[int] = typer.Option(
        None,
        "--priority",
        help="Filter by priority level (0-10)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed breakdowns",
    ),
):
    """Show queue monitoring statistics.

    Displays current queue depths, priority distribution, retry statistics,
    and wait times for items in various queues.
    """
    handle_async(_queue_status(tenant_id, priority, format, verbose))


async def _queue_status(
    tenant_id: Optional[str],
    priority: Optional[int],
    format: str,
    verbose: bool,
) -> None:
    """Implementation of queue_status command."""
    try:
        db_path = get_daemon_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build base query with filters
        where_clauses = []
        params: List[Any] = []

        if tenant_id:
            where_clauses.append("tenant_id = ?")
            params.append(tenant_id)

        if priority is not None:
            where_clauses.append("priority = ?")
            params.append(priority)

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Get overall queue statistics
        cursor.execute(
            f"""
            SELECT
                COUNT(*) as total_items,
                SUM(CASE WHEN priority >= 8 THEN 1 ELSE 0 END) as urgent_items,
                SUM(CASE WHEN priority >= 5 AND priority < 8 THEN 1 ELSE 0 END) as high_priority_items,
                SUM(CASE WHEN priority >= 3 AND priority < 5 THEN 1 ELSE 0 END) as normal_priority_items,
                SUM(CASE WHEN priority < 3 THEN 1 ELSE 0 END) as low_priority_items,
                SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) as retry_items,
                SUM(CASE WHEN error_message_id IS NOT NULL THEN 1 ELSE 0 END) as items_with_errors,
                COUNT(DISTINCT collection_name) as unique_collections,
                COUNT(DISTINCT tenant_id) as unique_tenants,
                MIN(queued_timestamp) as oldest_item,
                MAX(queued_timestamp) as newest_item
            FROM ingestion_queue
            WHERE {where_clause}
            """,
            params
        )
        stats = dict(cursor.fetchone())

        # Get missing metadata queue stats
        cursor.execute(
            f"""
            SELECT COUNT(*) as count
            FROM missing_metadata_queue
            WHERE {where_clause}
            """,
            params
        )
        stats["missing_metadata_items"] = cursor.fetchone()["count"]

        # Get per-collection breakdown if verbose
        collection_stats = []
        if verbose:
            cursor.execute(
                f"""
                SELECT
                    collection_name,
                    tenant_id,
                    COUNT(*) as queued_items,
                    AVG(priority) as avg_priority,
                    MIN(queued_timestamp) as oldest_queued,
                    SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) as items_with_retries
                FROM ingestion_queue
                WHERE {where_clause}
                GROUP BY collection_name, tenant_id
                ORDER BY queued_items DESC
                LIMIT 10
                """,
                params
            )
            collection_stats = [dict(row) for row in cursor.fetchall()]

        conn.close()

        # Format output
        if format.lower() == "json":
            output = {
                "overall_stats": stats,
                "collection_stats": collection_stats if verbose else None,
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            # Display as tables
            console = Console()

            # Overall statistics table
            table = Table(title="Queue Status Overview")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="yellow")

            table.add_row("Total Items", str(stats["total_items"]))
            table.add_row("  Urgent (8-10)", str(stats["urgent_items"]))
            table.add_row("  High (5-7)", str(stats["high_priority_items"]))
            table.add_row("  Normal (3-4)", str(stats["normal_priority_items"]))
            table.add_row("  Low (0-2)", str(stats["low_priority_items"]))
            table.add_row("", "")
            table.add_row("Items with Retries", str(stats["retry_items"]))
            table.add_row("Items with Errors", str(stats["items_with_errors"]))
            table.add_row("Missing Metadata Queue", str(stats["missing_metadata_items"]))
            table.add_row("", "")
            table.add_row("Unique Collections", str(stats["unique_collections"]))
            table.add_row("Unique Tenants", str(stats["unique_tenants"]))
            table.add_row("", "")
            table.add_row("Oldest Item", format_timestamp(stats["oldest_item"]))
            table.add_row("Newest Item", format_timestamp(stats["newest_item"]))

            if stats["oldest_item"]:
                wait_time = calculate_wait_time(stats["oldest_item"])
                table.add_row("Max Wait Time", wait_time)

            console.print(table)

            # Per-collection breakdown if verbose
            if verbose and collection_stats:
                console.print("\n")
                coll_table = Table(title="Per-Collection Queue Statistics")
                coll_table.add_column("Collection", style="cyan")
                coll_table.add_column("Tenant", style="blue")
                coll_table.add_column("Items", style="yellow")
                coll_table.add_column("Avg Priority", style="magenta")
                coll_table.add_column("Oldest", style="white")
                coll_table.add_column("Retries", style="red")

                for coll in collection_stats:
                    coll_table.add_row(
                        coll["collection_name"],
                        coll["tenant_id"],
                        str(coll["queued_items"]),
                        f"{coll['avg_priority']:.1f}",
                        calculate_wait_time(coll["oldest_queued"]),
                        str(coll["items_with_retries"]),
                    )

                console.print(coll_table)

    except FileNotFoundError as e:
        error_message(str(e))
        raise typer.Exit(1)
    except Exception as e:
        error_message(f"Failed to get queue status: {e}")
        logger.error("Error getting queue status", error=str(e), exc_info=True)
        raise typer.Exit(1)
