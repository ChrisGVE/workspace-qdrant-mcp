"""Message management CLI commands.

This module provides commands to view error messages and retry failed queue items
from the workspace-qdrant-mcp daemon.

Usage:
    wqm messages list                      # List recent errors
    wqm messages list --severity=error     # Filter by severity
    wqm messages list --format=json        # JSON output
    wqm messages retry 123                 # Retry specific item
    wqm messages retry --queue-type=retry  # Retry all items from retry queue
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from ..utils import (
    create_command_app,
    error_message,
    handle_async,
    success_message,
)

# Create the messages app
messages_app = create_command_app(
    name="messages",
    help_text="""Manage error messages and queue items.

View error messages from the daemon queue processor and retry failed items.

Examples:
    wqm messages list                          # List recent errors
    wqm messages list --severity=error         # Filter by severity
    wqm messages list --limit=100              # Show more errors
    wqm messages retry 123                     # Retry specific item ID
    wqm messages retry --queue-type=retry      # Retry all from retry queue
    wqm messages retry --dry-run               # Preview retry operations""",
    no_args_is_help=True,
)


def get_daemon_db_path() -> Path:
    """Get path to daemon SQLite database.

    Returns:
        Path to state.db file in daemon state directory.

    Raises:
        FileNotFoundError: If database file doesn't exist.
    """
    # Default location: ~/.config/workspace-qdrant/state.db
    config_dir = Path.home() / ".config" / "workspace-qdrant"
    db_path = config_dir / "state.db"

    if not db_path.exists():
        raise FileNotFoundError(
            f"Daemon database not found at {db_path}. "
            "Is the daemon running? Try: wqm service status"
        )

    return db_path


def format_timestamp(timestamp: str, relative: bool = True) -> str:
    """Format timestamp for display.

    Args:
        timestamp: ISO 8601 timestamp string
        relative: If True, show relative time (e.g., "2 hours ago")

    Returns:
        Formatted timestamp string
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

        if relative:
            now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
            delta = now - dt

            if delta.days > 0:
                return f"{delta.days}d ago"
            elif delta.seconds >= 3600:
                hours = delta.seconds // 3600
                return f"{hours}h ago"
            elif delta.seconds >= 60:
                minutes = delta.seconds // 60
                return f"{minutes}m ago"
            else:
                return f"{delta.seconds}s ago"
        else:
            return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return timestamp


def truncate_text(text: str, max_length: int = 60) -> str:
    """Truncate long text with ellipsis.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


@messages_app.command("list")
def list_messages(
    severity: str | None = typer.Option(
        None,
        "--severity",
        help="Filter by severity (error, warning, info)",
    ),
    queue_type: str | None = typer.Option(
        None,
        "--queue-type",
        help="Filter by queue type (processing, missing_metadata, retry)",
    ),
    days: int = typer.Option(
        7,
        "--days",
        help="Show messages from last N days",
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        help="Maximum number of messages to show",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
    tenant_id: str | None = typer.Option(
        None,
        "--tenant-id",
        help="Filter by tenant ID",
    ),
):
    """List error messages from the daemon.

    Shows recent error messages from the queue processor with filtering options.
    """
    handle_async(_list_messages(severity, queue_type, days, limit, format, tenant_id))


async def _list_messages(
    severity: str | None,
    queue_type: str | None,
    days: int,
    limit: int,
    format: str,
    tenant_id: str | None,
) -> None:
    """Implementation of list_messages command."""
    try:
        db_path = get_daemon_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Build query with filters
        query = """
            SELECT
                m.id,
                m.error_type,
                m.error_message,
                m.occurred_timestamp,
                m.file_path,
                m.collection_name,
                m.retry_count
            FROM messages m
            WHERE m.occurred_timestamp >= datetime('now', ?)
        """
        params: list[Any] = [f'-{days} days']

        # Apply filters
        if severity:
            # Map severity to error_type patterns
            severity_map = {
                "error": "%ERROR%",
                "warning": "%WARNING%",
                "info": "%INFO%",
            }
            if severity.lower() in severity_map:
                query += " AND m.error_type LIKE ?"
                params.append(severity_map[severity.lower()])

        if tenant_id:
            # Note: messages table doesn't have tenant_id directly
            # We'd need to join with ingestion_queue for this
            query += """
                AND EXISTS (
                    SELECT 1 FROM ingestion_queue iq
                    WHERE iq.file_absolute_path = m.file_path
                    AND iq.tenant_id = ?
                )
            """
            params.append(tenant_id)

        query += " ORDER BY m.occurred_timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            print("No messages found matching criteria")
            conn.close()
            return

        # Convert rows to dicts
        messages = [dict(row) for row in rows]

        if format.lower() == "json":
            print(json.dumps(messages, indent=2, default=str))
        else:
            # Display as table
            console = Console()
            table = Table(title=f"Error Messages ({len(messages)} found)")

            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Time", style="yellow")
            table.add_column("Type", style="magenta")
            table.add_column("Message", style="white")
            table.add_column("File", style="blue")
            table.add_column("Retries", style="red")

            for msg in messages:
                table.add_row(
                    str(msg["id"]),
                    format_timestamp(msg["occurred_timestamp"]),
                    truncate_text(msg["error_type"], 20),
                    truncate_text(msg["error_message"], 40),
                    truncate_text(msg["file_path"] or "N/A", 30),
                    str(msg["retry_count"]),
                )

            console.print(table)

        conn.close()

    except FileNotFoundError as e:
        error_message(str(e))
        raise typer.Exit(1)
    except Exception as e:
        error_message(f"Failed to list messages: {e}")
        logger.error("Error listing messages", error=str(e), exc_info=True)
        raise typer.Exit(1)


@messages_app.command("retry")
def retry_messages(
    item_id: int | None = typer.Argument(
        None,
        help="Specific message/item ID to retry",
    ),
    queue_type: str | None = typer.Option(
        None,
        "--queue-type",
        help="Retry all items from queue type (retry, missing_metadata)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview what will be retried without making changes",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip confirmation prompts",
    ),
):
    """Retry failed queue items.

    Requeue items from the missing_metadata_queue or retry specific items
    by ID. Use --dry-run to preview operations before executing.
    """
    handle_async(_retry_messages(item_id, queue_type, dry_run, force))


async def _retry_messages(
    item_id: int | None,
    queue_type: str | None,
    dry_run: bool,
    force: bool,
) -> None:
    """Implementation of retry_messages command."""
    try:
        db_path = get_daemon_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if item_id is not None:
            # Retry specific item by ID
            # Find the item in ingestion_queue
            cursor.execute(
                """
                SELECT file_absolute_path, collection_name, operation, priority
                FROM ingestion_queue
                WHERE error_message_id = ?
                """,
                (item_id,)
            )
            row = cursor.fetchone()

            if not row:
                error_message(f"No queue item found with error message ID {item_id}")
                conn.close()
                raise typer.Exit(1)

            if dry_run:
                print(f"Would retry item ID {item_id}:")
                print(f"  File: {row['file_absolute_path']}")
                print(f"  Collection: {row['collection_name']}")
                print(f"  Operation: {row['operation']}")
            else:
                # Clear error_message_id to allow retry
                cursor.execute(
                    """
                    UPDATE ingestion_queue
                    SET error_message_id = NULL,
                        retry_count = retry_count + 1,
                        queued_timestamp = CURRENT_TIMESTAMP
                    WHERE error_message_id = ?
                    """,
                    (item_id,)
                )
                conn.commit()
                success_message(f"Requeued item ID {item_id} for retry")

        elif queue_type:
            # Retry all items from specific queue
            if queue_type == "missing_metadata":
                # Query missing_metadata_queue
                cursor.execute(
                    "SELECT COUNT(*) as count FROM missing_metadata_queue"
                )
                count = cursor.fetchone()["count"]

                if count == 0:
                    print("No items in missing_metadata_queue")
                    conn.close()
                    return

                if dry_run:
                    print(f"Would retry {count} items from missing_metadata_queue")
                    # Show sample
                    cursor.execute(
                        """
                        SELECT file_absolute_path, collection_name, priority
                        FROM missing_metadata_queue
                        LIMIT 5
                        """
                    )
                    rows = cursor.fetchall()
                    print("\nSample items:")
                    for row in rows:
                        print(f"  - {row['file_absolute_path']} ({row['collection_name']})")
                    if count > 5:
                        print(f"  ... and {count - 5} more")
                else:
                    # Confirm bulk operation
                    if not force:
                        response = typer.confirm(
                            f"Retry {count} items from missing_metadata_queue?"
                        )
                        if not response:
                            print("Operation cancelled")
                            conn.close()
                            return

                    # Move items back to ingestion_queue
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO ingestion_queue
                            (file_absolute_path, collection_name, tenant_id, branch,
                             operation, priority, queued_timestamp, retry_count)
                        SELECT
                            file_absolute_path, collection_name, tenant_id, branch,
                            operation, priority, CURRENT_TIMESTAMP, retry_count + 1
                        FROM missing_metadata_queue
                        """
                    )

                    # Delete from missing_metadata_queue
                    cursor.execute("DELETE FROM missing_metadata_queue")
                    conn.commit()
                    success_message(f"Requeued {count} items from missing_metadata_queue")

            elif queue_type == "retry":
                # Items with retry_count > 0
                cursor.execute(
                    """
                    SELECT COUNT(*) as count
                    FROM ingestion_queue
                    WHERE retry_count > 0
                    """
                )
                count = cursor.fetchone()["count"]

                if count == 0:
                    print("No items with retry_count > 0")
                    conn.close()
                    return

                if dry_run:
                    print(f"Would reset retry for {count} items")
                else:
                    if not force:
                        response = typer.confirm(
                            f"Reset retry timestamps for {count} items?"
                        )
                        if not response:
                            print("Operation cancelled")
                            conn.close()
                            return

                    # Reset retry_from and update timestamp
                    cursor.execute(
                        """
                        UPDATE ingestion_queue
                        SET queued_timestamp = CURRENT_TIMESTAMP,
                            error_message_id = NULL
                        WHERE retry_count > 0
                        """
                    )
                    conn.commit()
                    success_message(f"Reset retry for {count} items")
            else:
                error_message(
                    f"Unknown queue type: {queue_type}. "
                    "Use 'missing_metadata' or 'retry'"
                )
                conn.close()
                raise typer.Exit(1)
        else:
            error_message("Specify either --item-id or --queue-type")
            conn.close()
            raise typer.Exit(1)

        conn.close()

    except FileNotFoundError as e:
        error_message(str(e))
        raise typer.Exit(1)
    except Exception as e:
        error_message(f"Failed to retry messages: {e}")
        logger.error("Error retrying messages", error=str(e), exc_info=True)
        raise typer.Exit(1)
