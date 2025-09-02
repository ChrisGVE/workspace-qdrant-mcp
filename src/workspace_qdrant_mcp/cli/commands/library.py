
from ...observability import get_logger
logger = get_logger(__name__)
"""Library collection management CLI commands.

This module provides management for readonly library collections
(prefixed with _) that are used for reference materials and documents.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.prompt import Confirm

from ...core.client import create_qdrant_client
from ...core.collection_naming import CollectionNameError, validate_collection_name
from ...core.config import Config


# Create the library app
library_app = typer.Typer(
    help=" Library collection management",
    no_args_is_help=True
,
    rich_markup_mode=None  # Disable Rich formatting completely
)

def handle_async(coro):
    """Helper to run async commands."""
    try:
        return asyncio.run(coro)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        raise typer.Exit(1)
    except Exception as e:
        print(f"Error: {e}")
        raise typer.Exit(1)

@library_app.command("list")
def list_libraries(
    stats: bool = typer.Option(False, "--stats", help="Include collection statistics"),
    sort_by: str = typer.Option("name", "--sort", help="Sort by: name, size, created"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
):
    """ Show all library collections."""
    handle_async(_list_libraries(stats, sort_by, format))

@library_app.command("create")
def create_library(
    name: str = typer.Argument(..., help="Library name (will be prefixed with _)"),
    description: str | None = typer.Option(None, "--description", "-d", help="Library description"),
    tags: list[str] | None = typer.Option(None, "--tag", "-t", help="Library tags"),
    vector_size: int = typer.Option(384, "--vector-size", help="Vector dimension size"),
    distance_metric: str = typer.Option("cosine", "--distance", help="Distance metric: cosine, euclidean, dot"),
):
    """ Create a new library collection."""
    handle_async(_create_library(name, description, tags, vector_size, distance_metric))

@library_app.command("remove")
def remove_library(
    name: str = typer.Argument(..., help="Library name (with or without _ prefix)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create backup before removal"),
):
    """ Remove library collection."""
    handle_async(_remove_library(name, force, backup))

@library_app.command("status")
def library_status(
    name: str | None = typer.Argument(None, help="Specific library name to check"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed information"),
    health_check: bool = typer.Option(False, "--health", help="Perform health check"),
):
    """ Show library statistics and health."""
    handle_async(_library_status(name, detailed, health_check))

@library_app.command("info")
def library_info(
    name: str = typer.Argument(..., help="Library name to inspect"),
    show_samples: bool = typer.Option(False, "--samples", help="Show sample documents"),
    show_schema: bool = typer.Option(False, "--schema", help="Show collection schema"),
):
    """ Show detailed library information."""
    handle_async(_library_info(name, show_samples, show_schema))

@library_app.command("rename")
def rename_library(
    old_name: str = typer.Argument(..., help="Current library name"),
    new_name: str = typer.Argument(..., help="New library name"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
):
    """ Rename a library collection."""
    handle_async(_rename_library(old_name, new_name, force))

@library_app.command("copy")
def copy_library(
    source: str = typer.Argument(..., help="Source library name"),
    destination: str = typer.Argument(..., help="Destination library name"),
    description: str | None = typer.Option(None, "--description", help="Description for new library"),
):
    """ Copy library collection."""
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
            print("📚 No library collections found.")
            print("Use 'wqm library create <name>' to create one")
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

        print(table)

        if stats:
            total_docs = sum(lib.get("points_count", 0) for lib in library_data if isinstance(lib.get("points_count"), int))
            print(f"\nTotal documents across all libraries: {total_docs:,}")

    except Exception as e:
        print(f"❌ Failed to list libraries: {e}")
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
            print(f"❌ Invalid collection name: {e}")
            raise typer.Exit(1)

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        # Check if collection already exists
        existing_collections = await client.list_collections()
        if any(col.get("name") == collection_name for col in existing_collections):
            print(f"❌ Library collection '{display_name}' already exists")
            raise typer.Exit(1)

        print(f" Creating library: {display_name}")

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

        print(f" Library collection '{display_name}' created successfully!")

        # Display creation summary
        summary_panel = Panel(
            f"""[bold]Library Collection Created[/bold]

📚 [cyan]Name:[/cyan] {display_name}
 [cyan]Full Name:[/cyan] {collection_name}
📏 [cyan]Vector Size:[/cyan] {vector_size}
📐 [cyan]Distance Metric:[/cyan] {distance_metric.upper()}
📝 [cyan]Description:[/cyan] {description or "None"}
🏷  [cyan]Tags:[/cyan] {', '.join(tags) if tags else "None"}

[dim]Next steps:[/dim]
• Use [green]wqm watch add PATH --collection={collection_name}[/green] to auto-monitor files
• Use [green]wqm ingest folder PATH --collection={collection_name}[/green] for manual ingestion
• Use [green]wqm search collection {collection_name} "query"[/green] to search the library""",
            title="🎉 Success",
            border_style="green"
        )
        print(summary_panel)

    except Exception as e:
        print(f"❌ Failed to create library: {e}")
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
            print(f"❌ Library collection '{display_name}' not found")
            raise typer.Exit(1)

        # Get collection info for confirmation
        try:
            info = await client.get_collection_info(collection_name)
            doc_count = info.get("points_count", 0)
        except Exception:
            doc_count = "unknown"

        if not force:
            print(f" Remove Library: {display_name}")
            print(" This will permanently delete the library collection!")
            print(f"Collection: {collection_name}")
            print(f"Documents: {doc_count}")

            if not get_user_confirmation("\nAre you sure you want to delete this library?"):
                print("Removal cancelled.")
                return

        # TODO: Implement backup functionality
        if backup:
            print("📦 Backup functionality not yet implemented")
            print("This will be added in a future update")

        # Delete the collection
        print(f"Removing library '{display_name}'...")
        await client.delete_collection(collection_name)

        print(f" Library collection '{display_name}' removed successfully")

    except Exception as e:
        print(f"❌ Failed to remove library: {e}")
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

            print(f" Library Status: {display_name}")

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

                print(status_table)

            except Exception as e:
                print(f"❌ Cannot get status for '{display_name}': {e}")

        else:
            # Status for all libraries
            print(" All Libraries Status")
            await _list_libraries(stats=True, sort_by="name", format="table")

        if health_check:
            print("\n🏥 Health Check")
            # TODO: Implement comprehensive health check
            print("Comprehensive health check will be implemented in future updates")

    except Exception as e:
        print(f"❌ Failed to get library status: {e}")
        raise typer.Exit(1)

async def _library_info(name: str, show_samples: bool, show_schema: bool):
    """Show detailed library information."""
    try:
        collection_name = name if name.startswith("_") else f"_{name}"
        display_name = name[1:] if name.startswith("_") else name

        config = Config()
        client = create_qdrant_client(config.qdrant_client_config)

        print(f" Library Info: {display_name}")

        # Get collection info
        info = await client.get_collection_info(collection_name)

        # Basic information
        info_panel = Panel(
            f"""[bold]Library Details[/bold]

📚 [cyan]Name:[/cyan] {display_name}
 [cyan]Full Name:[/cyan] {collection_name}
 [cyan]Status:[/cyan] {info.get("status", "unknown")}
📄 [cyan]Documents:[/cyan] {info.get("points_count", 0):,}
🔢 [cyan]Vectors:[/cyan] {info.get("vectors_count", 0):,}
 [cyan]Indexed:[/cyan] {info.get("indexed_points_count", 0):,}

[bold]Configuration[/bold]
📏 [cyan]Vector Size:[/cyan] {info.get("config", {}).get("params", {}).get("vectors", {}).get("size", "unknown")}
📐 [cyan]Distance:[/cyan] {info.get("config", {}).get("params", {}).get("vectors", {}).get("distance", "unknown")}""",
            title="📚 Library Information",
            border_style="blue"
        )
        print(info_panel)

        if show_schema:
            print("\n Collection Schema")
            # TODO: Display collection schema details
            print("Schema details display will be implemented in future updates")

        if show_samples:
            print("\n📄 Sample Documents")
            try:
                # Get a few sample points
                sample_results = await client.scroll_collection(collection_name, limit=3)

                if sample_results and "points" in sample_results:
                    for i, point in enumerate(sample_results["points"][:3], 1):
                        print(f"Sample {i}:")
                        payload = point.get("payload", {})

                        # Display key fields
                        for key, value in payload.items():
                            if key in ["title", "content", "filename", "source"]:
                                display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                                print(f"  {key}: {display_value}")

                        console.print()
                else:
                    print("No sample documents found")

            except Exception as e:
                print(f"Cannot retrieve samples: {e}")

    except Exception as e:
        print(f"❌ Failed to get library info: {e}")
        raise typer.Exit(1)

async def _rename_library(old_name: str, new_name: str, force: bool):
    """Rename a library collection."""
    try:
        print("🚧 Library renaming is not yet implemented")
        print("This feature requires creating a new collection and migrating data.")
        print("It will be implemented in a future update.")
        print("\nCurrent workaround:")
        print(f"1. wqm library create {new_name}")
        print("2. Manually re-ingest content into the new library")
        print(f"3. wqm library remove {old_name} when ready")

    except Exception as e:
        print(f"❌ Rename operation failed: {e}")
        raise typer.Exit(1)

async def _copy_library(source: str, destination: str, description: str | None):
    """Copy library collection."""
    try:
        print("🚧 Library copying is not yet implemented")
        print("This feature requires advanced collection manipulation.")
        print("It will be implemented in a future update.")
        print("\nCurrent workaround:")
        print(f"1. wqm library create {destination}")
        print("2. Export data from source library (when export feature is available)")
        print("3. Import data into destination library")

    except Exception as e:
        print(f"❌ Copy operation failed: {e}")
        raise typer.Exit(1)
