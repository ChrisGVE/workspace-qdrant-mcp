"""Tool availability CLI commands.

This module provides commands to check tool availability status
(tree-sitter parsers, LSP servers) in the workspace-qdrant-mcp daemon.

Usage:
    wqm tools status                   # List all detected tools
    wqm tools status --language=rust   # Filter by language
    wqm tools status --available       # Show only available tools
    wqm tools status --format=json     # JSON output
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

# Create the tools app
tools_app = create_command_app(
    name="tools",
    help_text="""Check tool availability status.

View detected tools (tree-sitter parsers, LSP servers) and their
availability status for different programming languages.

Examples:
    wqm tools status                      # List all tools
    wqm tools status --language=python    # Filter by language
    wqm tools status --available          # Show only available tools
    wqm tools status --unavailable        # Show missing tools
    wqm tools status --format=json        # JSON output""",
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


def format_timestamp(timestamp: Optional[int]) -> str:
    """Format Unix timestamp for display.

    Args:
        timestamp: Unix timestamp (seconds since epoch) or None

    Returns:
        Formatted timestamp string or "Never"
    """
    if not timestamp:
        return "Never"

    try:
        dt = datetime.fromtimestamp(timestamp)
        now = datetime.now()
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
            return "Just now"
    except (ValueError, AttributeError, OSError):
        return "Unknown"


def truncate_path(path: Optional[str], max_length: int = 50) -> str:
    """Truncate long paths with ellipsis in the middle.

    Args:
        path: File path to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated path with ellipsis if needed
    """
    if not path:
        return "N/A"

    if len(path) <= max_length:
        return path

    # Keep start and end of path
    keep_chars = max_length - 3  # Reserve 3 for "..."
    start_chars = keep_chars // 2
    end_chars = keep_chars - start_chars

    return path[:start_chars] + "..." + path[-end_chars:]


@tools_app.command("status")
def tool_status(
    language: Optional[str] = typer.Option(
        None,
        "--language",
        help="Filter by programming language",
    ),
    available: bool = typer.Option(
        False,
        "--available",
        help="Show only available tools",
    ),
    unavailable: bool = typer.Option(
        False,
        "--unavailable",
        help="Show only unavailable tools",
    ),
    tool_type: Optional[str] = typer.Option(
        None,
        "--type",
        help="Filter by tool type (tree-sitter, lsp)",
    ),
    format: str = typer.Option(
        "table",
        "--format",
        help="Output format: table or json",
    ),
):
    """Show tool availability status.

    Lists detected tools (tree-sitter parsers, LSP servers) and their
    availability for different programming languages. Shows when each
    tool was last checked and its current status.
    """
    handle_async(_tool_status(language, available, unavailable, tool_type, format))


async def _tool_status(
    language: Optional[str],
    available: bool,
    unavailable: bool,
    tool_type: Optional[str],
    format: str,
) -> None:
    """Implementation of tool_status command."""
    try:
        db_path = get_daemon_db_path()
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if tool_availability table exists
        cursor.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='tool_availability'
            """
        )
        if not cursor.fetchone():
            error_message(
                "Tool availability table not found. "
                "The daemon may not have run tool monitoring yet."
            )
            conn.close()
            raise typer.Exit(1)

        # Build query with filters
        where_clauses = []
        params: List[Any] = []

        if language:
            where_clauses.append("language = ?")
            params.append(language.lower())

        if available and not unavailable:
            where_clauses.append("is_available = 1")
        elif unavailable and not available:
            where_clauses.append("is_available = 0")

        if tool_type:
            # Match tool_type prefix (case-insensitive)
            where_clauses.append("LOWER(tool_type) LIKE ?")
            params.append(f"{tool_type.lower()}%")

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Query tool_availability table
        cursor.execute(
            f"""
            SELECT
                tool_type,
                language,
                tool_path,
                last_checked_at,
                is_available
            FROM tool_availability
            WHERE {where_clause}
            ORDER BY language, tool_type
            """,
            params
        )
        rows = cursor.fetchall()

        if not rows:
            print("No tools found matching criteria")
            conn.close()
            return

        # Convert rows to dicts
        tools = [dict(row) for row in rows]

        # Get summary statistics
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_tools,
                SUM(CASE WHEN is_available = 1 THEN 1 ELSE 0 END) as available_tools,
                COUNT(DISTINCT language) as languages_supported
            FROM tool_availability
            """
        )
        summary = dict(cursor.fetchone())

        conn.close()

        # Format output
        if format.lower() == "json":
            output = {
                "summary": summary,
                "tools": tools,
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            # Display as table
            console = Console()

            # Summary line
            console.print(
                f"\n[bold]Tool Availability Summary[/bold]: "
                f"{summary['available_tools']}/{summary['total_tools']} available, "
                f"{summary['languages_supported']} languages supported\n"
            )

            # Tools table
            table = Table(title=f"Detected Tools ({len(tools)} found)")
            table.add_column("Language", style="cyan", no_wrap=True)
            table.add_column("Tool Type", style="magenta")
            table.add_column("Status", style="yellow")
            table.add_column("Path", style="blue")
            table.add_column("Last Checked", style="white")

            for tool in tools:
                status_text = "✓ Available" if tool["is_available"] else "✗ Missing"
                status_style = "green" if tool["is_available"] else "red"

                table.add_row(
                    tool["language"],
                    tool["tool_type"],
                    f"[{status_style}]{status_text}[/{status_style}]",
                    truncate_path(tool["tool_path"]),
                    format_timestamp(tool["last_checked_at"]),
                )

            console.print(table)

            # Show hint about missing tools
            unavailable_count = summary["total_tools"] - summary["available_tools"]
            if unavailable_count > 0:
                console.print(
                    f"\n[yellow]Note:[/yellow] {unavailable_count} tools are unavailable. "
                    "Files requiring these tools are in the missing_metadata_queue."
                )
                console.print(
                    "Use [cyan]wqm queue status[/cyan] to see queue statistics."
                )

    except FileNotFoundError as e:
        error_message(str(e))
        raise typer.Exit(1)
    except Exception as e:
        error_message(f"Failed to get tool status: {e}")
        logger.error("Error getting tool status", error=str(e), exc_info=True)
        raise typer.Exit(1)
