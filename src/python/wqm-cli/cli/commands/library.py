"""Library collection management CLI commands.

This module provides management for readonly library collections
(prefixed with _) that are used for reference materials and documents.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from common.core.collection_naming import CollectionNameError, validate_collection_name
from common.core.daemon_client import get_daemon_client, with_daemon_client
from common.core.yaml_config import load_config
from common.observability import get_logger
from ..utils import (
    confirm,
    create_command_app,
    error_message,
    force_option,
    handle_async,
    json_output_option,
    success_message,
    verbose_option,
    warning_message,
)

logger = get_logger(__name__)


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


# Async implementation functions
async def _list_libraries(stats: bool, sort_by: str, format: str):
    """List all library collections."""
    try:
        config = load_config()
        
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

        config = load_config()
        
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

        config = load_config()
        
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
        config = load_config()

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
            else:
                info = result
                print(f"{display_name} Status:")
                print(f"Collection Name: {collection_name}")
                print(f"Status: active")
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

        config = load_config()

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
        print(f"Status: active")
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