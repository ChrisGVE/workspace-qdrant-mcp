"""Library collection management CLI commands.

This module provides management for readonly library collections
(prefixed with _) that are used for reference materials and documents.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from ...core.client import create_qdrant_client
from ...core.collection_naming import CollectionNameError, validate_collection_name
from ...core.config import Config

console = Console()

# Create the library app
library_app = typer.Typer(help="📚 Library collection management")

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

@library_app.command("list")
def list_libraries(
    stats: bool = typer.Option(False, "--stats", help="Include collection statistics"),
    sort_by: str = typer.Option("name", "--sort", help="Sort by: name, size, created"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
):
    """📋 Show all library collections."""
    handle_async(_list_libraries(stats, sort_by, format))

@library_app.command("create")
def create_library(
    name: str = typer.Argument(..., help="Library name (will be prefixed with _)"),
    description: str | None = typer.Option(None, "--description", "-d", help="Library description"),
    tags: list[str] | None = typer.Option(None, "--tag", "-t", help="Library tags"),
    vector_size: int = typer.Option(384, "--vector-size", help="Vector dimension size"),
    distance_metric: str = typer.Option("cosine", "--distance", help="Distance metric: cosine, euclidean, dot"),
):
    """➕ Create a new library collection."""
    handle_async(_create_library(name, description, tags, vector_size, distance_metric))

@library_app.command("remove")
def remove_library(
    name: str = typer.Argument(..., help="Library name (with or without _ prefix)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create backup before removal"),
):
    """🗑️ Remove library collection."""
    handle_async(_remove_library(name, force, backup))

@library_app.command("status")
def library_status(
    name: str | None = typer.Argument(None, help="Specific library name to check"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information"),
    health_check: bool = typer.Option(False, "--health", help="Perform health check"),
):
    """📊 Show library statistics and health."""
    handle_async(_library_status(name, detailed, health_check))

@library_app.command("info")
def library_info(
    name: str = typer.Argument(..., help="Library name to inspect"),
    show_samples: bool = typer.Option(False, "--samples", help="Show sample documents"),
    show_schema: bool = typer.Option(False, "--schema", help="Show collection schema"),
):
    """🔍 Show detailed library information."""
    handle_async(_library_info(name, show_samples, show_schema))

@library_app.command("rename")
def rename_library(
    old_name: str = typer.Argument(..., help="Current library name"),
    new_name: str = typer.Argument(..., help="New library name"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
):
    """✏️ Rename a library collection."""
    handle_async(_rename_library(old_name, new_name, force))

@library_app.command("copy")
def copy_library(
    source: str = typer.Argument(..., help="Source library name"),
    destination: str = typer.Argument(..., help="Destination library name"),
    description: str | None = typer.Option(None, "--description", help="Description for new library"),
):
    """📋 Copy library collection."""
    handle_async(_copy_library(source, destination, description))

# Async implementation functions
async def _list_libraries(stats: bool, sort_by: str, format: str):
    """List all library collections."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Get all collections
        all_collections = await client.list_collections()

        # Filter for library collections (start with _)
        library_collections = [
            col for col in all_collections
            if col.get("name", "").startswith("_")
        ]

        if not library_collections:
            console.print("[yellow]📚 No library collections found.[/yellow]")
            console.print("[dim]Use 'wqm library create <name>' to create one[/dim]")
            return

        # Collect statistics if requested
        library_data = []
        for col in library_collections:
            name = col.get("name", "")
            lib_info = {"name": name, "display_name": name[1:]}  # Remove _ prefix for display

            if stats:
                try:
                    info = await client.get_collection_info(name)
                    lib_info.update({
                        "points_count": info.get("points_count", 0),
                        "vectors_count": info.get("vectors_count", 0),
                        "indexed_points_count": info.get("indexed_points_count", 0),
                        "status": info.get("status", "unknown")
                    })
                except Exception as e:
                    lib_info.update({
                        "points_count": "error",
                        "status": "error",
                        "error": str(e)
                    })

            library_data.append(lib_info)

        # Sort libraries
        if sort_by == "name":
            library_data.sort(key=lambda x: x["name"])
        elif sort_by == "size" and stats:
            library_data.sort(key=lambda x: x.get("points_count", 0) if isinstance(x.get("points_count"), int) else 0, reverse=True)

        if format == "json":
            import json
            print(json.dumps(library_data, indent=2))
            return

        # Display as table
        table = Table(title=f"📚 Library Collections ({len(library_data)} found)")
        table.add_column("Name", style="cyan")
        table.add_column("Status", justify="center", width=12)

        if stats:
            table.add_column("Documents", justify="right", width=12)
            table.add_column("Vectors", justify="right", width=12)
            table.add_column("Indexed", justify="right", width=12)

        for lib in library_data:
            name = lib["display_name"]

            # Determine status
            if stats and lib.get("error"):
                status = "[red]ERROR[/red]"
            elif stats and lib.get("points_count", 0) == 0:
                status = "[yellow]EMPTY[/yellow]"
            elif stats:
                status = "[green]ACTIVE[/green]"
            else:
                status = "[blue]LIBRARY[/blue]"

            if stats:
                points = str(lib.get("points_count", "?"))
                vectors = str(lib.get("vectors_count", "?"))
                indexed = str(lib.get("indexed_points_count", "?"))
                table.add_row(name, status, points, vectors, indexed)
            else:
                table.add_row(name, status)

        console.print(table)

        if stats:
            total_docs = sum(lib.get("points_count", 0) for lib in library_data if isinstance(lib.get("points_count"), int))
            console.print(f"\n[dim]Total documents across all libraries: {total_docs:,}[/dim]")

    except Exception as e:
        console.print(f"[red]❌ Failed to list libraries: {e}[/red]")
        raise typer.Exit(1)

async def _create_library(
    name: str,
    description: str | None,
    tags: list[str] | None,
    vector_size: int,
    distance_metric: str
):
    """Create a new library collection."""
    try:
        # Ensure name starts with underscore
        if not name.startswith("_"):
            collection_name = f"_{name}"
            display_name = name
        else:
            collection_name = name
            display_name = name[1:]

        # Validate collection name
        try:
            validate_collection_name(collection_name, allow_library=True)
        except CollectionNameError as e:
            console.print(f"[red]❌ Invalid collection name: {e}[/red]")
            raise typer.Exit(1)

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Check if collection already exists
        existing_collections = await client.list_collections()
        if any(col.get("name") == collection_name for col in existing_collections):
            console.print(f"[red]❌ Library collection '{display_name}' already exists[/red]")
            raise typer.Exit(1)

        console.print(f"[bold blue]➕ Creating library: {display_name}[/bold blue]")

        # Prepare collection configuration
        collection_config = {
            "vectors": {
                "size": vector_size,
                "distance": distance_metric.upper()
            }
        }

        # Add metadata if provided
        payload_schema = {}
        if description:
            payload_schema["description"] = "text"
        if tags:
            payload_schema["tags"] = "keyword"

        if payload_schema:
            collection_config["payload_schema"] = payload_schema

        # Create the collection
        await client.create_collection(collection_name, collection_config)

        console.print(f"[green]✅ Library collection '{display_name}' created successfully![/green]")

        # Display creation summary
        summary_panel = Panel(
            f"""[bold]Library Collection Created[/bold]

📚 [cyan]Name:[/cyan] {display_name}
🔍 [cyan]Full Name:[/cyan] {collection_name}
📏 [cyan]Vector Size:[/cyan] {vector_size}
📐 [cyan]Distance Metric:[/cyan] {distance_metric.upper()}
📝 [cyan]Description:[/cyan] {description or "None"}
🏷️  [cyan]Tags:[/cyan] {', '.join(tags) if tags else "None"}

[dim]Next steps:[/dim]
• Use [green]wqm watch add PATH --collection={collection_name}[/green] to auto-monitor files
• Use [green]wqm ingest folder PATH --collection={collection_name}[/green] for manual ingestion
• Use [green]wqm search collection {collection_name} "query"[/green] to search the library""",
            title="🎉 Success",
            border_style="green"
        )
        console.print(summary_panel)

    except Exception as e:
        console.print(f"[red]❌ Failed to create library: {e}[/red]")
        raise typer.Exit(1)

async def _remove_library(name: str, force: bool, backup: bool):
    """Remove library collection."""
    try:
        # Ensure name starts with underscore
        if not name.startswith("_"):
            collection_name = f"_{name}"
            display_name = name
        else:
            collection_name = name
            display_name = name[1:]

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Check if collection exists
        existing_collections = await client.list_collections()
        if not any(col.get("name") == collection_name for col in existing_collections):
            console.print(f"[red]❌ Library collection '{display_name}' not found[/red]")
            raise typer.Exit(1)

        # Get collection info for confirmation
        try:
            info = await client.get_collection_info(collection_name)
            doc_count = info.get("points_count", 0)
        except Exception:
            doc_count = "unknown"

        if not force:
            console.print(f"[bold red]🗑️ Remove Library: {display_name}[/bold red]")
            console.print("[yellow]⚠️ This will permanently delete the library collection![/yellow]")
            console.print(f"Collection: {collection_name}")
            console.print(f"Documents: {doc_count}")

            if not Confirm.ask("\n[red]Are you sure you want to delete this library?[/red]"):
                console.print("[yellow]Removal cancelled.[/yellow]")
                return

        # TODO: Implement backup functionality
        if backup:
            console.print("[yellow]📦 Backup functionality not yet implemented[/yellow]")
            console.print("[dim]This will be added in a future update[/dim]")

        # Delete the collection
        console.print(f"[yellow]Removing library '{display_name}'...[/yellow]")
        await client.delete_collection(collection_name)

        console.print(f"[green]✅ Library collection '{display_name}' removed successfully[/green]")

    except Exception as e:
        console.print(f"[red]❌ Failed to remove library: {e}[/red]")
        raise typer.Exit(1)

async def _library_status(name: str | None, detailed: bool, health_check: bool):
    """Show library statistics and health."""
    try:
        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        if name:
            # Status for specific library
            collection_name = name if name.startswith("_") else f"_{name}"
            display_name = name[1:] if name.startswith("_") else name

            console.print(f"[bold blue]📊 Library Status: {display_name}[/bold blue]")

            try:
                info = await client.get_collection_info(collection_name)

                status_table = Table(title=f"📚 {display_name} Status")
                status_table.add_column("Property", style="cyan")
                status_table.add_column("Value", style="white")

                status_table.add_row("Collection Name", collection_name)
                status_table.add_row("Status", info.get("status", "unknown"))
                status_table.add_row("Documents", str(info.get("points_count", 0)))
                status_table.add_row("Vectors", str(info.get("vectors_count", 0)))
                status_table.add_row("Indexed Points", str(info.get("indexed_points_count", 0)))

                if detailed:
                    status_table.add_row("Vector Size", str(info.get("config", {}).get("params", {}).get("vectors", {}).get("size", "unknown")))
                    status_table.add_row("Distance Metric", str(info.get("config", {}).get("params", {}).get("vectors", {}).get("distance", "unknown")))

                console.print(status_table)

            except Exception as e:
                console.print(f"[red]❌ Cannot get status for '{display_name}': {e}[/red]")

        else:
            # Status for all libraries
            console.print("[bold blue]📊 All Libraries Status[/bold blue]")
            await _list_libraries(stats=True, sort_by="name", format="table")

        if health_check:
            console.print("\n[bold blue]🏥 Health Check[/bold blue]")
            # TODO: Implement comprehensive health check
            console.print("[yellow]Comprehensive health check will be implemented in future updates[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Failed to get library status: {e}[/red]")
        raise typer.Exit(1)

async def _library_info(name: str, show_samples: bool, show_schema: bool):
    """Show detailed library information."""
    try:
        collection_name = name if name.startswith("_") else f"_{name}"
        display_name = name[1:] if name.startswith("_") else name

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        console.print(f"[bold blue]🔍 Library Info: {display_name}[/bold blue]")

        # Get collection info
        info = await client.get_collection_info(collection_name)

        # Basic information
        info_panel = Panel(
            f"""[bold]Library Details[/bold]

📚 [cyan]Name:[/cyan] {display_name}
🔍 [cyan]Full Name:[/cyan] {collection_name}
📊 [cyan]Status:[/cyan] {info.get("status", "unknown")}
📄 [cyan]Documents:[/cyan] {info.get("points_count", 0):,}
🔢 [cyan]Vectors:[/cyan] {info.get("vectors_count", 0):,}
✅ [cyan]Indexed:[/cyan] {info.get("indexed_points_count", 0):,}

[bold]Configuration[/bold]
📏 [cyan]Vector Size:[/cyan] {info.get("config", {}).get("params", {}).get("vectors", {}).get("size", "unknown")}
📐 [cyan]Distance:[/cyan] {info.get("config", {}).get("params", {}).get("vectors", {}).get("distance", "unknown")}""",
            title="📚 Library Information",
            border_style="blue"
        )
        console.print(info_panel)

        if show_schema:
            console.print("\n[bold blue]📋 Collection Schema[/bold blue]")
            # TODO: Display collection schema details
            console.print("[yellow]Schema details display will be implemented in future updates[/yellow]")

        if show_samples:
            console.print("\n[bold blue]📄 Sample Documents[/bold blue]")
            try:
                # Get a few sample points
                sample_results = await client.scroll_collection(collection_name, limit=3)

                if sample_results and "points" in sample_results:
                    for i, point in enumerate(sample_results["points"][:3], 1):
                        console.print(f"[cyan]Sample {i}:[/cyan]")
                        payload = point.get("payload", {})

                        # Display key fields
                        for key, value in payload.items():
                            if key in ["title", "content", "filename", "source"]:
                                display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                                console.print(f"  [dim]{key}:[/dim] {display_value}")

                        console.print()
                else:
                    console.print("[yellow]No sample documents found[/yellow]")

            except Exception as e:
                console.print(f"[yellow]Cannot retrieve samples: {e}[/yellow]")

    except Exception as e:
        console.print(f"[red]❌ Failed to get library info: {e}[/red]")
        raise typer.Exit(1)

async def _rename_library(old_name: str, new_name: str, force: bool):
    """Rename a library collection."""
    try:
        console.print("[yellow]🚧 Library renaming is not yet implemented[/yellow]")
        console.print("This feature requires creating a new collection and migrating data.")
        console.print("It will be implemented in a future update.")
        console.print("\n[dim]Current workaround:[/dim]")
        console.print(f"1. [green]wqm library create {new_name}[/green]")
        console.print("2. Manually re-ingest content into the new library")
        console.print(f"3. [red]wqm library remove {old_name}[/red] when ready")

    except Exception as e:
        console.print(f"[red]❌ Rename operation failed: {e}[/red]")
        raise typer.Exit(1)

async def _copy_library(source: str, destination: str, description: str | None):
    """Copy library collection."""
    try:
        console.print("[yellow]🚧 Library copying is not yet implemented[/yellow]")
        console.print("This feature requires advanced collection manipulation.")
        console.print("It will be implemented in a future update.")
        console.print("\n[dim]Current workaround:[/dim]")
        console.print(f"1. [green]wqm library create {destination}[/green]")
        console.print("2. Export data from source library (when export feature is available)")
        console.print("3. Import data into destination library")

    except Exception as e:
        console.print(f"[red]❌ Copy operation failed: {e}[/red]")
        raise typer.Exit(1)
