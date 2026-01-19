"""Library collection management CLI commands.

This module provides management for readonly library collections
(prefixed with _) that are used for reference materials and documents.

Task 399: Multi-tenant library management with watch folder integration.
- `wqm library add` - Add a watch folder for library ingestion
- `wqm library rescan` - Re-ingest all files from a library watch
- `wqm library watches` - List all library watch folders
"""

from pathlib import Path

import typer
from common.core.collection_naming import CollectionNameError, validate_collection_name
from common.core.config import get_config_manager
from common.core.sqlite_state_manager import SQLiteStateManager
from common.grpc.daemon_client import with_daemon_client, get_daemon_client
from loguru import logger

from ..utils import (
    create_command_app,
    handle_async,
)

# logger imported from loguru


# Create the library app using shared utilities
library_app = create_command_app(
    name="library",
    help_text="""Library collection management.

Manage readonly library collections for reference materials.

Examples:
    wqm library list                    # Show all library collections
    wqm library create technical-docs   # Create new library collection
    wqm library status                  # Show library statistics
    wqm library info tech-docs          # Show collection details
    wqm library remove old-docs         # Remove library collection""",
    no_args_is_help=True,
)


@library_app.command("list")
def list_libraries(
    stats: bool = typer.Option(False, "--stats", help="Include collection statistics"),
    sort_by: str = typer.Option("name", "--sort", help="Sort by: name, size, created"),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json"
    ),
):
    """Show all library collections."""
    handle_async(_list_libraries(stats, sort_by, format))


@library_app.command("create")
def create_library(
    name: str = typer.Argument(..., help="Library name (will be prefixed with _)"),
    description: str | None = typer.Option(
        None, "--description", "-d", help="Library description"
    ),
    tags: list[str] | None = typer.Option(None, "--tag", "-t", help="Library tags"),
    vector_size: int = typer.Option(384, "--vector-size", help="Vector dimension size"),
    distance_metric: str = typer.Option(
        "cosine", "--distance", help="Distance metric: cosine, euclidean, dot"
    ),
):
    """Create a new library collection."""
    handle_async(_create_library(name, description, tags, vector_size, distance_metric))


@library_app.command("remove")
def remove_library(
    name: str = typer.Argument(..., help="Library name (with or without _ prefix)"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    backup: bool = typer.Option(
        True, "--backup/--no-backup", help="Create backup before removal"
    ),
):
    """Remove library collection."""
    handle_async(_remove_library(name, force, backup))


@library_app.command("status")
def library_status(
    name: str | None = typer.Argument(None, help="Specific library name to check"),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed information"
    ),
    health_check: bool = typer.Option(False, "--health", help="Perform health check"),
):
    """Show library statistics and health."""
    handle_async(_library_status(name, detailed, health_check))


@library_app.command("info")
def library_info(
    name: str = typer.Argument(..., help="Library name to inspect"),
    show_samples: bool = typer.Option(False, "--samples", help="Show sample documents"),
    show_schema: bool = typer.Option(False, "--schema", help="Show collection schema"),
):
    """Show detailed library information."""
    handle_async(_library_info(name, show_samples, show_schema))


@library_app.command("rename")
def rename_library(
    old_name: str = typer.Argument(..., help="Current library name"),
    new_name: str = typer.Argument(..., help="New library name"),
    force: bool = typer.Option(False, "--force", help="Skip confirmation"),
):
    """Rename a library collection."""
    handle_async(_rename_library(old_name, new_name, force))


@library_app.command("copy")
def copy_library(
    source: str = typer.Argument(..., help="Source library name"),
    destination: str = typer.Argument(..., help="Destination library name"),
    description: str | None = typer.Option(
        None, "--description", help="Description for new library"
    ),
):
    """Copy library collection."""
    handle_async(_copy_library(source, destination, description))


# ============================================================================
# Task 399: Multi-tenant library watch folder commands
# ============================================================================

@library_app.command("add")
def add_library_watch(
    path: str = typer.Argument(..., help="Path to library folder to watch"),
    name: str = typer.Option(..., "--name", "-n", help="Library name (e.g., 'langchain')"),
    patterns: list[str] | None = typer.Option(
        None,
        "--pattern",
        "-p",
        help="File patterns to include (default: *.pdf, *.epub, *.md, *.txt)",
    ),
    ignore: list[str] | None = typer.Option(
        None,
        "--ignore",
        "-i",
        help="Patterns to exclude (default: .git/*, __pycache__/*)",
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Watch subdirectories"
    ),
    depth: int = typer.Option(
        10, "--depth", help="Maximum directory depth to watch"
    ),
    debounce: float = typer.Option(
        5.0, "--debounce", help="Debounce time in seconds"
    ),
    auto_ingest: bool = typer.Option(
        True, "--auto/--no-auto", help="Start initial ingestion immediately"
    ),
):
    """Add a folder to watch for automatic library ingestion.

    This command configures a watch folder that the daemon will monitor
    for changes. Files matching the patterns will be automatically
    ingested into the _libraries unified collection with the specified
    library_name as the tenant identifier.

    Examples:
        wqm library add ~/docs/langchain --name langchain
        wqm library add ~/research/papers --name papers --pattern "*.pdf"
        wqm library add ~/docs --name docs --depth 5 --no-recursive
    """
    handle_async(
        _add_library_watch(path, name, patterns, ignore, recursive, depth, debounce, auto_ingest)
    )


@library_app.command("watches")
def list_library_watches(
    all_watches: bool = typer.Option(
        False, "--all", "-a", help="Include disabled watches"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json"
    ),
):
    """List all library watch folders.

    Shows configured library watch folders with their status,
    document counts, and last scan times.
    """
    handle_async(_list_library_watches(all_watches, format))


@library_app.command("unwatch")
def remove_library_watch(
    name: str = typer.Argument(..., help="Library name to stop watching"),
    delete_collection: bool = typer.Option(
        False, "--delete-collection", help="Also delete the library collection"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation"
    ),
):
    """Stop watching a library folder.

    Removes the watch configuration for the specified library.
    Optionally deletes the associated collection data.
    """
    handle_async(_remove_library_watch(name, delete_collection, force))


@library_app.command("rescan")
def rescan_library(
    name: str = typer.Argument(..., help="Library name to rescan"),
    clear_first: bool = typer.Option(
        False, "--clear", "-c", help="Clear existing documents before rescan"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Skip confirmation"
    ),
):
    """Re-ingest all files from a library watch folder.

    Queues all matching files from the watched folder for re-ingestion.
    Use --clear to remove existing documents first.

    Examples:
        wqm library rescan langchain              # Re-ingest all files
        wqm library rescan papers --clear         # Clear and re-ingest
    """
    handle_async(_rescan_library(name, clear_first, force))


@library_app.command("watch-status")
def library_watch_status(
    name: str = typer.Argument(..., help="Library name to check"),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed statistics"
    ),
):
    """Show detailed status for a specific library watch.

    Displays watch configuration, ingestion stats, and recent activity.
    """
    handle_async(_library_watch_status(name, detailed))


# ============================================================================
# Helper functions
# ============================================================================

async def _get_state_manager() -> SQLiteStateManager:
    """Get initialized state manager for library watch operations."""
    state_manager = SQLiteStateManager()
    await state_manager.initialize()
    return state_manager


# Async implementation functions
async def _list_libraries(stats: bool, sort_by: str, format: str):
    """List all library collections."""
    try:
        config = get_config_manager()

        async def _operation(client):
            # Get all collections
            response = await client.list_collections(include_stats=stats)
            return response.collections

        all_collections = await with_daemon_client(_operation, config)

        # Filter for library collections (start with _)
        library_collections = [
            col for col in all_collections if col.name.startswith("_")
        ]

        if not library_collections:
            print("No library collections found.")
            print("Use 'wqm library create <name>' to create one")
            return

        # Collect statistics
        library_data = []
        for col in library_collections:
            name = col.name
            lib_info = {
                "name": name,
                "display_name": name[1:],  # Remove _ prefix for display
            }

            if stats:
                try:
                    # Stats are already included in the collection response
                    lib_info.update(
                        {
                            "points_count": col.points_count,
                            "vectors_count": col.vectors_count if hasattr(col, 'vectors_count') else col.points_count,
                            "indexed_points_count": col.indexed_points_count if hasattr(col, 'indexed_points_count') else col.points_count,
                            "status": col.status if hasattr(col, 'status') else "active",
                        }
                    )
                except Exception as e:
                    lib_info.update(
                        {"points_count": "error", "status": "error", "error": str(e)}
                    )

            library_data.append(lib_info)

        # Sort libraries
        if sort_by == "name":
            library_data.sort(key=lambda x: x["name"])
        elif sort_by == "size" and stats:
            library_data.sort(
                key=lambda x: x.get("points_count", 0)
                if isinstance(x.get("points_count"), int)
                else 0,
                reverse=True,
            )

        if format == "json":
            import json

            print(json.dumps(library_data, indent=2))
            return

        # Display as table
        print(f"Library Collections ({len(library_data)} found):")
        print(f"{'Name':<25} {'Status':<12}", end="")

        if stats:
            print(f" {'Documents':<12} {'Vectors':<12} {'Indexed':<12}")

        print()
        print("-" * (65 if stats else 40))

        for lib in library_data:
            name = lib["display_name"]

            # Determine status
            if stats and lib.get("error"):
                status = "ERROR"
            elif stats and lib.get("points_count", 0) == 0:
                status = "EMPTY"
            elif stats:
                status = "ACTIVE"
            else:
                status = "LIBRARY"

            if stats:
                points = str(lib.get("points_count", "?"))
                vectors = str(lib.get("vectors_count", "?"))
                indexed = str(lib.get("indexed_points_count", "?"))
                print(
                    f"{name:<25} {status:<12} {points:<12} {vectors:<12} {indexed:<12}"
                )
            else:
                print(f"{name:<25} {status:<12}")

        if stats:
            total_docs = sum(
                lib.get("points_count", 0)
                for lib in library_data
                if isinstance(lib.get("points_count"), int)
            )
            print(f"\nTotal documents across all libraries: {total_docs:,}")

    except Exception as e:
        print(f"Error: Failed to list libraries: {e}")
        raise typer.Exit(1)


async def _create_library(
    name: str,
    description: str | None,
    tags: list[str] | None,
    vector_size: int,
    distance_metric: str,
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
            print(f"Error: Invalid collection name: {e}")
            raise typer.Exit(1)

        config = get_config_manager()

        async def _operation(client):
            # Check if collection already exists
            response = await client.list_collections(include_stats=False)
            existing_names = [col.name for col in response.collections]

            if collection_name in existing_names:
                return None  # Signal that collection exists

            # Create the collection with metadata
            metadata = {}
            if description:
                metadata["description"] = description
            if tags:
                metadata["tags"] = ",".join(tags)
            metadata["vector_size"] = str(vector_size)
            metadata["distance_metric"] = distance_metric.upper()

            return await client.create_collection(
                collection_name=collection_name,
                description=description or "",
                metadata=metadata
            )

        result = await with_daemon_client(_operation, config)

        if result is None:
            print(f"Error: Library collection '{display_name}' already exists")
            raise typer.Exit(1)

        print(f"Creating library: {display_name}")

        print(f"Library collection '{display_name}' created successfully!")

        # Display creation summary
        print("Library Collection Created")
        print(f"Name: {display_name}")
        print(f"Full Name: {collection_name}")
        print(f"Vector Size: {vector_size}")
        print(f"Distance Metric: {distance_metric.upper()}")
        print(f"Description: {description or 'None'}")
        print(f"Tags: {', '.join(tags) if tags else 'None'}")
        print("\nNext steps:")
        print(
            f"- Use 'wqm watch add PATH --collection={collection_name}' to auto-monitor files"
        )
        print(
            f"- Use 'wqm ingest folder PATH --collection={collection_name}' for manual ingestion"
        )
        print(
            f"- Use 'wqm search collection {collection_name} \"query\"' to search the library"
        )

    except Exception as e:
        print(f"Error: Failed to create library: {e}")
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

        config = get_config_manager()

        async def _operation(client):
            # Check if collection exists and get info
            response = await client.list_collections(include_stats=False)
            existing_names = [col.name for col in response.collections]

            if collection_name not in existing_names:
                return None, None  # Signal that collection doesn't exist

            # Get collection info for confirmation
            try:
                info = await client.get_collection_info(collection_name)
                doc_count = info.points_count
            except Exception:
                doc_count = "unknown"

            return collection_name, doc_count

        collection_exists, doc_count = await with_daemon_client(_operation, config)

        if collection_exists is None:
            print(f"Error: Library collection '{display_name}' not found")
            raise typer.Exit(1)

        if not force:
            print(f"Remove Library: {display_name}")
            print("This will permanently delete the library collection!")
            print(f"Collection: {collection_name}")
            print(f"Documents: {doc_count}")

            response = input("\nAre you sure you want to delete this library? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                print("Removal cancelled.")
                return

        # TODO: Implement backup functionality
        if backup:
            print("Backup functionality not yet implemented")
            print("This will be added in a future update")

        # Delete the collection
        print(f"Removing library '{display_name}'...")

        async def _delete_operation(client):
            return await client.delete_collection(collection_name, confirm=True)

        await with_daemon_client(_delete_operation, config)

        print(f"Library collection '{display_name}' removed successfully")

    except Exception as e:
        print(f"Error: Failed to remove library: {e}")
        raise typer.Exit(1)


async def _library_status(name: str | None, detailed: bool, health_check: bool):
    """Show library statistics and health."""
    try:
        config = get_config_manager()

        if name:
            # Status for specific library
            collection_name = name if name.startswith("_") else f"_{name}"
            display_name = name[1:] if name.startswith("_") else name

            print(f"Library Status: {display_name}")

            async def _operation(client):
                try:
                    return await client.get_collection_info(collection_name)
                except Exception as e:
                    return None, str(e)

            result = await with_daemon_client(_operation, config)

            if isinstance(result, tuple):
                print(f"Error: Cannot get status for '{display_name}': {result[1]}")
                raise typer.Exit(1)
            else:
                info = result
                print(f"{display_name} Status:")
                print(f"Collection Name: {collection_name}")
                print("Status: active")
                print(f"Documents: {info.points_count}")
                print(f"Vectors: {info.points_count}")
                print(f"Indexed Points: {info.indexed_points_count if hasattr(info, 'indexed_points_count') else info.points_count}")

                if detailed:
                    print(f"Vector Size: {info.vector_size if hasattr(info, 'vector_size') else 'unknown'}")
                    print(f"Distance Metric: {info.distance_metric if hasattr(info, 'distance_metric') else 'unknown'}")

        else:
            # Status for all libraries
            print("All Libraries Status")
            await _list_libraries(stats=True, sort_by="name", format="table")

        if health_check:
            print("\nHealth Check")
            # TODO: Implement comprehensive health check
            print("Comprehensive health check will be implemented in future updates")

    except Exception as e:
        print(f"Error: Failed to get library status: {e}")
        raise typer.Exit(1)


async def _library_info(name: str, show_samples: bool, show_schema: bool):
    """Show detailed library information."""
    try:
        collection_name = name if name.startswith("_") else f"_{name}"
        display_name = name[1:] if name.startswith("_") else name

        config = get_config_manager()

        print(f"Library Info: {display_name}")

        async def _operation(client):
            # Get collection info with sample documents if requested
            info = await client.get_collection_info(
                collection_name,
                include_sample_documents=show_samples
            )
            return info

        info = await with_daemon_client(_operation, config)

        # Basic information
        print("Library Details:")
        print(f"Name: {display_name}")
        print(f"Full Name: {collection_name}")
        print("Status: active")
        print(f"Documents: {info.points_count:,}")
        print(f"Vectors: {info.points_count:,}")
        print(f"Indexed: {info.indexed_points_count if hasattr(info, 'indexed_points_count') else info.points_count:,}")
        print("\nConfiguration:")
        print(f"Vector Size: {info.vector_size if hasattr(info, 'vector_size') else 'unknown'}")
        print(f"Distance: {info.distance_metric if hasattr(info, 'distance_metric') else 'unknown'}")

        if show_schema:
            print("\nCollection Schema")
            print("Schema details display will be implemented in future updates")

        if show_samples and hasattr(info, 'sample_documents') and info.sample_documents:
            print("\nSample Documents")
            for i, doc in enumerate(info.sample_documents[:3], 1):
                print(f"Sample {i}:")
                # Display key fields from document metadata
                metadata = getattr(doc, 'metadata', {})
                for key in ["title", "content", "filename", "source"]:
                    if key in metadata:
                        value = metadata[key]
                        display_value = (
                            str(value)[:100] + "..."
                            if len(str(value)) > 100
                            else str(value)
                        )
                        print(f"  {key}: {display_value}")
                print()
        elif show_samples:
            print("\nSample Documents")
            print("No sample documents found")

    except Exception as e:
        print(f"Error: Failed to get library info: {e}")
        raise typer.Exit(1)


async def _rename_library(old_name: str, new_name: str, force: bool):
    """Rename a library collection."""
    try:
        print("Warning: Library renaming is not yet implemented")
        print("This feature requires creating a new collection and migrating data.")
        print("It will be implemented in a future update.")
        print("\nCurrent workaround:")
        print(f"1. wqm library create {new_name}")
        print("2. Manually re-ingest content into the new library")
        print(f"3. wqm library remove {old_name} when ready")

    except Exception as e:
        print(f"Error: Rename operation failed: {e}")
        raise typer.Exit(1)


async def _copy_library(source: str, destination: str, description: str | None):
    """Copy library collection."""
    try:
        print("Warning: Library copying is not yet implemented")
        print("This feature requires advanced collection manipulation.")
        print("It will be implemented in a future update.")
        print("\nCurrent workaround:")
        print(f"1. wqm library create {destination}")
        print("2. Export data from source library (when export feature is available)")
        print("3. Import data into destination library")

    except Exception as e:
        print(f"Error: Copy operation failed: {e}")
        raise typer.Exit(1)


# ============================================================================
# Task 399: Multi-tenant library watch folder implementations
# ============================================================================

async def _add_library_watch(
    path: str,
    name: str,
    patterns: list[str] | None,
    ignore: list[str] | None,
    recursive: bool,
    depth: int,
    debounce: float,
    auto_ingest: bool,
):
    """Add a library watch folder configuration."""
    try:
        # Validate and resolve path
        watch_path = Path(path).expanduser().resolve()
        if not watch_path.exists():
            print(f"Error: Path does not exist: {watch_path}")
            raise typer.Exit(1)
        if not watch_path.is_dir():
            print(f"Error: Path is not a directory: {watch_path}")
            raise typer.Exit(1)

        # Normalize library name (remove _ prefix if provided)
        library_name = name.lstrip("_").lower().replace(" ", "-")

        # Set default patterns
        if patterns is None:
            patterns = ["*.pdf", "*.epub", "*.md", "*.txt"]
        if ignore is None:
            ignore = [".git/*", "__pycache__/*", "node_modules/*", ".DS_Store"]

        # Get state manager and save configuration
        state_manager = await _get_state_manager()

        # Check if library watch already exists
        existing = await state_manager.get_library_watch(library_name)
        if existing:
            print(f"Library watch '{library_name}' already exists.")
            print(f"Current path: {existing['path']}")
            response = input("Do you want to update it? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                print("Operation cancelled.")
                return

        # Save library watch configuration
        success = await state_manager.save_library_watch(
            library_name=library_name,
            path=str(watch_path),
            patterns=patterns,
            ignore_patterns=ignore,
            recursive=recursive,
            recursive_depth=depth,
            debounce_seconds=debounce,
            enabled=True,
            metadata={"auto_ingest": auto_ingest},
        )

        if not success:
            print(f"Error: Failed to save library watch configuration")
            raise typer.Exit(1)

        print(f"Library Watch Added: {library_name}")
        print(f"Path: {watch_path}")
        print(f"Patterns: {', '.join(patterns)}")
        print(f"Ignore: {', '.join(ignore)}")
        print(f"Recursive: {recursive} (depth: {depth})")
        print(f"Debounce: {debounce}s")

        # Signal daemon to refresh watches (if available)
        try:
            daemon_client = get_daemon_client()
            await daemon_client.connect()
            # The daemon polls SQLite for watch changes, so no explicit signal needed
            # But we can check if it's running
            print("\nDaemon will detect new watch configuration automatically.")
        except Exception as e:
            logger.debug(f"Daemon not available: {e}")
            print("\nNote: Daemon not running. Start with 'wqm service start'.")

        if auto_ingest:
            print("\nTo start initial ingestion, run:")
            print(f"  wqm library rescan {library_name}")

        print("\nNext steps:")
        print(f"- Search: wqm search --scope all \"{library_name} documentation\"")
        print(f"- Status: wqm library watch-status {library_name}")
        print(f"- List:   wqm library watches")

    except typer.Exit:
        raise
    except Exception as e:
        print(f"Error: Failed to add library watch: {e}")
        raise typer.Exit(1)


async def _list_library_watches(all_watches: bool, format: str):
    """List all library watch configurations."""
    try:
        state_manager = await _get_state_manager()
        watches = await state_manager.list_library_watches(enabled_only=not all_watches)

        if not watches:
            print("No library watches configured.")
            print("Use 'wqm library add <path> --name <name>' to add one.")
            return

        if format == "json":
            import json
            print(json.dumps(watches, indent=2, default=str))
            return

        # Table format
        print(f"Library Watches ({len(watches)} found):")
        print()
        print(f"{'Name':<20} {'Path':<40} {'Docs':<8} {'Status':<10} {'Last Scan':<20}")
        print("-" * 100)

        for watch in watches:
            name = watch["library_name"]
            path = watch["path"]
            if len(path) > 38:
                path = "..." + path[-35:]
            doc_count = watch.get("document_count", 0) or 0
            status = "active" if watch["enabled"] else "disabled"
            last_scan = watch.get("last_scan", "never") or "never"
            if last_scan != "never":
                last_scan = last_scan[:19]  # Trim timestamp

            print(f"{name:<20} {path:<40} {doc_count:<8} {status:<10} {last_scan:<20}")

        print()
        total_docs = sum(w.get("document_count", 0) or 0 for w in watches)
        print(f"Total documents: {total_docs:,}")

    except Exception as e:
        print(f"Error: Failed to list library watches: {e}")
        raise typer.Exit(1)


async def _remove_library_watch(name: str, delete_collection: bool, force: bool):
    """Remove a library watch configuration."""
    try:
        library_name = name.lstrip("_").lower()
        state_manager = await _get_state_manager()

        # Check if watch exists
        watch = await state_manager.get_library_watch(library_name)
        if not watch:
            print(f"Error: Library watch '{library_name}' not found")
            raise typer.Exit(1)

        if not force:
            print(f"Remove Library Watch: {library_name}")
            print(f"Path: {watch['path']}")
            print(f"Documents: {watch.get('document_count', 0) or 0}")
            if delete_collection:
                print("\nWARNING: This will also delete the collection data!")

            response = input("\nAre you sure? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                print("Operation cancelled.")
                return

        # Remove watch configuration
        success = await state_manager.remove_library_watch(library_name)
        if not success:
            print(f"Error: Failed to remove library watch")
            raise typer.Exit(1)

        print(f"Library watch '{library_name}' removed.")

        # Optionally delete collection
        if delete_collection:
            try:
                config = get_config_manager()
                collection_name = f"_{library_name}"

                async def _delete_op(client):
                    return await client.delete_collection(collection_name, confirm=True)

                await with_daemon_client(_delete_op, config)
                print(f"Collection '{collection_name}' deleted.")
            except Exception as e:
                print(f"Warning: Failed to delete collection: {e}")

    except typer.Exit:
        raise
    except Exception as e:
        print(f"Error: Failed to remove library watch: {e}")
        raise typer.Exit(1)


async def _rescan_library(name: str, clear_first: bool, force: bool):
    """Re-ingest all files from a library watch folder."""
    try:
        library_name = name.lstrip("_").lower()
        state_manager = await _get_state_manager()

        # Get watch configuration
        watch = await state_manager.get_library_watch(library_name)
        if not watch:
            print(f"Error: Library watch '{library_name}' not found")
            print("Use 'wqm library watches' to see configured watches.")
            raise typer.Exit(1)

        watch_path = Path(watch["path"])
        if not watch_path.exists():
            print(f"Error: Watch path no longer exists: {watch_path}")
            raise typer.Exit(1)

        # Count files to process
        patterns = watch.get("patterns", ["*.pdf", "*.epub", "*.md", "*.txt"])
        ignore_patterns = watch.get("ignore_patterns", [])
        recursive = watch.get("recursive", True)
        depth = watch.get("recursive_depth", 10)

        # Find matching files
        matching_files = []
        for pattern in patterns:
            if recursive:
                matching_files.extend(watch_path.rglob(pattern))
            else:
                matching_files.extend(watch_path.glob(pattern))

        # Filter out ignored files
        def should_ignore(file_path: Path) -> bool:
            rel_path = str(file_path.relative_to(watch_path))
            for ignore_pattern in ignore_patterns:
                if ignore_pattern.endswith("/*"):
                    dir_name = ignore_pattern[:-2]
                    if rel_path.startswith(dir_name + "/") or f"/{dir_name}/" in rel_path:
                        return True
                elif rel_path.endswith(ignore_pattern.lstrip("*")):
                    return True
            return False

        matching_files = [f for f in matching_files if not should_ignore(f)]

        if not matching_files:
            print(f"No files found matching patterns: {patterns}")
            return

        if not force:
            print(f"Rescan Library: {library_name}")
            print(f"Path: {watch_path}")
            print(f"Files to process: {len(matching_files)}")
            if clear_first:
                print("\nWARNING: This will clear existing documents first!")

            response = input("\nProceed? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                print("Operation cancelled.")
                return

        # Clear collection if requested
        if clear_first:
            print("Clearing existing documents...")
            try:
                config = get_config_manager()

                async def _clear_op(client):
                    # Delete and recreate collection
                    collection_name = f"_{library_name}"
                    try:
                        await client.delete_collection(collection_name, confirm=True)
                    except Exception:
                        pass  # Collection might not exist
                    return True

                await with_daemon_client(_clear_op, config)
                print("Collection cleared.")
            except Exception as e:
                print(f"Warning: Failed to clear collection: {e}")

        # Queue files for ingestion
        print(f"Queueing {len(matching_files)} files for ingestion...")

        queued_count = 0
        for file_path in matching_files:
            try:
                await state_manager.add_to_ingestion_queue(
                    file_path=str(file_path),
                    collection=f"_libraries",  # Unified libraries collection
                    tenant_id=library_name,
                    priority=5,  # Normal priority
                    metadata={"library_name": library_name, "source": "rescan"},
                )
                queued_count += 1
            except Exception as e:
                logger.warning(f"Failed to queue {file_path}: {e}")

        print(f"Queued {queued_count} files for ingestion.")

        # Update last scan time
        await state_manager.update_library_watch_stats(library_name)

        print("\nThe daemon will process queued files automatically.")
        print(f"Monitor progress with: wqm library watch-status {library_name}")

    except typer.Exit:
        raise
    except Exception as e:
        print(f"Error: Failed to rescan library: {e}")
        raise typer.Exit(1)


async def _library_watch_status(name: str, detailed: bool):
    """Show detailed status for a library watch."""
    try:
        library_name = name.lstrip("_").lower()
        state_manager = await _get_state_manager()

        # Get watch configuration
        watch = await state_manager.get_library_watch(library_name)
        if not watch:
            print(f"Error: Library watch '{library_name}' not found")
            raise typer.Exit(1)

        print(f"Library Watch Status: {library_name}")
        print("=" * 50)
        print(f"Path: {watch['path']}")
        print(f"Status: {'active' if watch['enabled'] else 'disabled'}")
        print(f"Documents: {watch.get('document_count', 0) or 0}")
        print(f"Last Scan: {watch.get('last_scan', 'never') or 'never'}")
        print(f"Added: {watch.get('added_at', 'unknown')}")

        print("\nConfiguration:")
        print(f"  Patterns: {', '.join(watch.get('patterns', []))}")
        print(f"  Ignore: {', '.join(watch.get('ignore_patterns', []))}")
        print(f"  Recursive: {watch.get('recursive', True)}")
        print(f"  Max Depth: {watch.get('recursive_depth', 10)}")
        print(f"  Debounce: {watch.get('debounce_seconds', 2.0)}s")

        if detailed:
            # Check if path exists and count files
            watch_path = Path(watch["path"])
            if watch_path.exists():
                patterns = watch.get("patterns", ["*.pdf", "*.epub", "*.md", "*.txt"])
                file_count = 0
                for pattern in patterns:
                    if watch.get("recursive", True):
                        file_count += len(list(watch_path.rglob(pattern)))
                    else:
                        file_count += len(list(watch_path.glob(pattern)))

                print(f"\nFile System:")
                print(f"  Path exists: Yes")
                print(f"  Matching files: {file_count}")
            else:
                print(f"\nFile System:")
                print(f"  Path exists: NO - Path missing!")

            # Get ingestion queue stats
            try:
                queue_stats = await state_manager.get_queue_stats()
                library_queued = sum(
                    1 for item in queue_stats.get("items", [])
                    if item.get("tenant_id") == library_name
                )
                print(f"\nIngestion Queue:")
                print(f"  Pending items: {library_queued}")
            except Exception:
                pass

    except typer.Exit:
        raise
    except Exception as e:
        print(f"Error: Failed to get watch status: {e}")
        raise typer.Exit(1)
