"""File watching CLI commands.

This module provides management for library folder watching,
enabling automatic ingestion of files into library collections.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from ...core.daemon_client import get_daemon_client, with_daemon_client
from ...core.config import Config
from ...observability import get_logger
from ..utils import (
    confirm,
    create_command_app,
    error_message,
    force_option,
    handle_async,
    success_message,
    verbose_option,
    warning_message,
)

logger = get_logger(__name__)

# Create the watch app using shared utilities
watch_app = create_command_app(
    name="watch",
    help_text="""Folder watching configuration.
    
Configure automatic folder watching for library collections.
    
Examples:
    wqm watch list                      # Show all watch configurations
    wqm watch add ~/docs --collection=docs  # Watch folder for changes
    wqm watch remove ~/docs             # Stop watching folder
    wqm watch status                    # Show watch service status
    wqm watch enable --name=docs-watch  # Enable specific watch""",
    no_args_is_help=True,
)


async def _get_daemon_client():
    """Get connected daemon client for watch operations."""
    config = Config()
    client = get_daemon_client(config.workspace_config)
    await client.connect()
    return client


@watch_app.command("add")
def add_watch(
    path: str = typer.Argument(..., help="Path to watch"),
    collection: str = typer.Option(
        ..., "--collection", "-c", help="Target collection (must start with _)"
    ),
    patterns: list[str] | None = typer.Option(
        None,
        "--pattern",
        "-p",
        help="File patterns to watch (default: *.pdf, *.epub, *.txt, *.md)",
    ),
    ignore: list[str] | None = typer.Option(
        None,
        "--ignore",
        "-i",
        help="Ignore patterns (default: .git/*, node_modules/*, __pycache__/*, .DS_Store)",
    ),
    auto_ingest: bool = typer.Option(
        True, "--auto/--no-auto", help="Enable automatic ingestion"
    ),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", help="Watch subdirectories"
    ),
    debounce: int = typer.Option(5, "--debounce", help="Debounce time in seconds"),
):
    """Add a folder to watch for automatic ingestion."""
    handle_async(
        _add_watch(path, collection, patterns, ignore, auto_ingest, recursive, debounce)
    )


@watch_app.command("list")
def list_watches(
    active_only: bool = typer.Option(
        False, "--active", help="Show only active watches"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Filter by collection"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json"
    ),
):
    """Show all active watches."""
    handle_async(_list_watches(active_only, collection, format))


@watch_app.command("remove")
def remove_watch(
    path: str | None = typer.Argument(None, help="Path to stop watching (or watch ID)"),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Remove all watches for collection"
    ),
    all: bool = typer.Option(False, "--all", help="Remove all watches"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Stop watching folder."""
    handle_async(_remove_watch(path, collection, all, force))


@watch_app.command("status")
def watch_status(
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed statistics"
    ),
    recent: bool = typer.Option(False, "--recent", help="Show recent activity"),
):
    """Watch activity and statistics."""
    handle_async(_watch_status(detailed, recent))


@watch_app.command("pause")
def pause_watches(
    path: str | None = typer.Argument(
        None, help="Specific path to pause (or watch ID)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Pause all watches for collection"
    ),
    all: bool = typer.Option(False, "--all", help="Pause all watches"),
):
    """Pause all or specific watches."""
    handle_async(_pause_watches(path, collection, all))


@watch_app.command("resume")
def resume_watches(
    path: str | None = typer.Argument(
        None, help="Specific path to resume (or watch ID)"
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Resume all watches for collection"
    ),
    all: bool = typer.Option(False, "--all", help="Resume all watches"),
):
    """Resume paused watches."""
    handle_async(_resume_watches(path, collection, all))


@watch_app.command("sync")
def sync_watched_folders(
    path: str | None = typer.Argument(None, help="Specific path to sync"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be synced"),
    force: bool = typer.Option(False, "--force", help="Force sync all files"),
):
    """Sync watched folders manually."""
    handle_async(_sync_watched_folders(path, dry_run, force))


# Async implementation functions
async def _add_watch(
    path: str,
    collection: str,
    patterns: list[str] | None,
    ignore: list[str] | None,
    auto_ingest: bool,
    recursive: bool,
    debounce: int,
):
    """Add a folder watch configuration."""
    try:
        watch_path = Path(path).resolve()

        print("Adding watch configuration")
        print(f"Path: {watch_path}")
        print(f"Collection: {collection}")

        # Get daemon client
        client = await _get_daemon_client()
        
        try:
            # Validate path
            if not watch_path.exists():
                print(f"Error: Path does not exist: {path}")
                raise typer.Exit(1)
            if not watch_path.is_dir():
                print(f"Error: Path is not a directory: {path}")
                raise typer.Exit(1)

            # Validate collection (must be library collection)
            if not collection.startswith("_"):
                print(f"Error: Collection must start with underscore (library collection): {collection}")
                raise typer.Exit(1)

            # Set defaults if not provided
            if patterns is None:
                patterns = ["*.pdf", "*.epub", "*.txt", "*.md"]
            if ignore is None:
                ignore = [".git/*", "node_modules/*", "__pycache__/*", ".DS_Store"]

            # Validate collection exists
            collections_response = await client.list_collections()
            collection_names = [c.name for c in collections_response.collections]
            
            if collection not in collection_names:
                print(f"Error: Collection '{collection}' not found. Create it first with: wqm library create {collection[1:]}")
                raise typer.Exit(1)

            # Start watching using daemon client
            watch_updates = client.start_watching(
                path=str(watch_path),
                collection=collection,
                patterns=patterns,
                ignore_patterns=ignore,
                auto_ingest=auto_ingest,
                recursive=recursive,
                recursive_depth=-1 if recursive else 1,
                debounce_seconds=debounce,
                update_frequency_ms=1000,
            )

            # Process first update to get watch ID
            watch_id = None
            async for update in watch_updates:
                if update.watch_id:
                    watch_id = update.watch_id
                    print(f"Watch started with ID: {watch_id}")
                    break

            # Show configuration
            print("Watch Configuration Added")
            print("=" * 50)
            print(f"Path: {watch_path}")
            print(f"Collection: {collection}")
            print(f"Watch ID: {watch_id}")
            print(f"Auto-ingest: {'Enabled' if auto_ingest else 'Disabled'}")
            print(f"Recursive: {'Yes' if recursive else 'No'}")
            print(f"Debounce: {debounce} seconds")
            print("\nFile Patterns:")
            for pattern in patterns:
                print(f"  • {pattern}")
            print("\nIgnore Patterns:")
            for ignore_pattern in ignore:
                print(f"  • {ignore_pattern}")

            if auto_ingest:
                print(f"\nFile monitoring started for {watch_path}")
                print("New files will be automatically ingested into the collection")
            else:
                print("\nWarning: Auto-ingest is disabled")
                print("Files will be detected but not automatically processed")
                print("Enable with: wqm watch resume")

        finally:
            await client.disconnect()

    except Exception as e:
        print(f"Error: Failed to add watch: {e}")
        raise typer.Exit(1)


async def _list_watches(active_only: bool, collection: str | None, format: str):
    """List all watch configurations."""
    try:
        client = await _get_daemon_client()
        
        try:
            watches_response = await client.list_watches(active_only)
            watches = watches_response.watches

            if format == "json":
                # JSON output
                output = []
                for watch in watches:
                    watch_dict = {
                        "watch_id": watch.watch_id,
                        "path": watch.path,
                        "collection": watch.collection,
                        "status": watch.status,
                        "patterns": list(watch.patterns),
                        "ignore_patterns": list(watch.ignore_patterns),
                        "auto_ingest": watch.auto_ingest,
                        "recursive": watch.recursive,
                        "debounce_seconds": watch.debounce_seconds,
                    }
                    output.append(watch_dict)
                print(json.dumps(output, indent=2))
                return

            # Filter by collection if specified
            if collection:
                watches = [w for w in watches if w.collection == collection]

            # Table output
            if not watches:
                print("No watches found")
                if not active_only:
                    print("Add a watch with: wqm watch add <path> --collection=<library>")
                return

            print(f"Watch Configurations ({len(watches)} found)\n")

            # Show summary table in plain text format
            if watches:
                print(
                    f"{'ID':<10} {'Path':<30} {'Collection':<20} {'Status':<15} {'Auto-Ingest'}"
                )
                print("-" * 85)
                for watch in watches:
                    status_str = "Active" if watch.status == 1 else "Stopped"  # Assuming enum values
                    auto_ingest_str = "Yes" if watch.auto_ingest else "No"
                    
                    # Truncate path if too long
                    path = str(watch.path)
                    if len(path) > 28:
                        path = path[:28] + "..."
                    
                    # Truncate collection if too long
                    collection_name = str(watch.collection)
                    if len(collection_name) > 18:
                        collection_name = collection_name[:18] + "..."
                    
                    watch_id = str(watch.watch_id)
                    if len(watch_id) > 8:
                        watch_id = watch_id[:8] + "..."
                    
                    print(
                        f"{watch_id:<10} {path:<30} {collection_name:<20} {status_str:<15} {auto_ingest_str}"
                    )
            else:
                print("No watch configurations found.")

            # Show tips
            print("\nTip: Use 'wqm watch status --detailed' for more information")
            print("Tip: Use 'wqm watch sync' to manually process watched directories")

        finally:
            await client.disconnect()

    except Exception as e:
        print(f"Error: Failed to list watches: {e}")
        raise typer.Exit(1)


async def _remove_watch(
    path: str | None, collection: str | None, all: bool, force: bool
):
    """Remove watch configurations using daemon client."""
    try:
        client = await _get_daemon_client()
        
        try:
            if all:
                print(" Remove All Watches")
            elif collection:
                print(f" Remove Watches for Collection: {collection}")
            elif path:
                print(f" Remove Watch: {path}")
            else:
                print("Error: Must specify --all, --collection, or a path/watch ID")
                raise typer.Exit(1)

            # Get all watches first
            watches_response = await client.list_watches(active_only=False)
            all_watches = watches_response.watches
            
            # Find watches to remove
            watches_to_remove = []

            if all:
                watches_to_remove = all_watches
            elif collection:
                watches_to_remove = [w for w in all_watches if w.collection == collection]
            elif path:
                # Try as watch ID first, then as path
                matches = [
                    w
                    for w in all_watches
                    if w.watch_id == path or Path(w.path) == Path(path).resolve()
                ]
                if matches:
                    watches_to_remove = matches
                else:
                    print(f"Error: No watch found for: {path}")
                    raise typer.Exit(1)

            if not watches_to_remove:
                print("No matching watches found")
                return

            # Show what will be removed
            print(f"\nFound {len(watches_to_remove)} watch(es) to remove:")
            for watch in watches_to_remove:
                print(f"  • {watch.path} -> {watch.collection} ({watch.watch_id})")

            # Confirm removal
            if not force:
                action = (
                    "all watches"
                    if all
                    else f"watches for {collection}"
                    if collection
                    else f"watch for {path}"
                )
                if not confirm(f"\nAre you sure you want to remove {action}?"):
                    print("Operation cancelled")
                    return

            # Remove watches
            removed_count = 0
            for watch in watches_to_remove:
                try:
                    response = await client.stop_watching(watch.watch_id)
                    if response.success:
                        removed_count += 1
                        print(f"Removed watch: {watch.path}")
                    else:
                        print(f"Error: Failed to remove watch: {watch.path} - {response.message}")
                except Exception as e:
                    print(f"Error: Failed to remove watch: {watch.path} - {e}")

            print(f"\nSuccessfully removed {removed_count} watch(es)")

        finally:
            await client.disconnect()

    except Exception as e:
        print(f"Error: Failed to remove watches: {e}")
        raise typer.Exit(1)


async def _watch_status(detailed: bool, recent: bool):
    """Show watch activity and statistics using daemon client."""
    try:
        client = await _get_daemon_client()
        
        try:
            # Get watch statistics
            stats_response = await client.get_stats(include_watch_stats=True)
            watch_stats = stats_response.watch_stats
            
            # Get watches for detailed info
            watches_response = await client.list_watches(active_only=False)
            watches = watches_response.watches

            print("Watch System Status\n")

            # System status overview in plain text format
            total_watches = len(watches)
            active_watches = sum(1 for w in watches if w.status == 1)  # Assuming 1 = active
            stopped_watches = total_watches - active_watches
            
            print(f"Total Watches: {total_watches}")
            print(f"Active: {active_watches}")
            print(f"Stopped: {stopped_watches}")
            
            if watch_stats:
                print(f"Total Files Monitored: {watch_stats.total_files_monitored}")
                print(f"Files Processed: {watch_stats.total_files_processed}")
                print(f"Processing Errors: {watch_stats.processing_errors}")

            if detailed and watches:
                print("\nDetailed Watch Information")
                print("-" * 50)
                for watch in watches:
                    status_str = "Active" if watch.status == 1 else "Stopped"
                    print(f"Watch {watch.watch_id}:")
                    print(f"  Path: {watch.path}")
                    print(f"  Collection: {watch.collection}")
                    print(f"  Status: {status_str}")
                    print(f"  Auto-ingest: {'Yes' if watch.auto_ingest else 'No'}")
                    print(f"  Recursive: {'Yes' if watch.recursive else 'No'}")
                    print(f"  Patterns: {', '.join(watch.patterns)}")
                    print()

            if recent and watch_stats:
                print("\nRecent Activity")
                print("-" * 50)
                # Note: Recent activity details would need to be added to the gRPC response
                print("Recent activity tracking not yet implemented via daemon")

            # Show tips
            if total_watches == 0:
                print("\nNo watches configured yet")
                print("Add one with: wqm watch add <path> --collection=<library>")
            elif active_watches == 0:
                print("\nNo watches are currently active")
                print("Resume them with: wqm watch resume --all")

        finally:
            await client.disconnect()

    except Exception as e:
        print(f"Error: Failed to get watch status: {e}")
        raise typer.Exit(1)


async def _pause_watches(path: str | None, collection: str | None, all: bool):
    """Pause watch configurations using daemon client."""
    try:
        client = await _get_daemon_client()
        
        try:
            # Get all watches first
            watches_response = await client.list_watches(active_only=False)
            watches = watches_response.watches

            if all:
                print("Pausing all watches")
                paused_count = 0
                for watch in watches:
                    try:
                        from ...grpc.ingestion_pb2 import WatchStatus
                        response = await client.configure_watch(
                            watch_id=watch.watch_id,
                            status=WatchStatus.WATCH_STATUS_PAUSED
                        )
                        if response.success:
                            paused_count += 1
                    except Exception as e:
                        print(f"Failed to pause watch {watch.watch_id}: {e}")
                print(f"Paused {paused_count} watch(es)")

            elif collection:
                print(f"Pausing watches for collection: {collection}")
                matching_watches = [w for w in watches if w.collection == collection]
                paused_count = 0
                for watch in matching_watches:
                    try:
                        from ...grpc.ingestion_pb2 import WatchStatus
                        response = await client.configure_watch(
                            watch_id=watch.watch_id,
                            status=WatchStatus.WATCH_STATUS_PAUSED
                        )
                        if response.success:
                            paused_count += 1
                    except Exception as e:
                        print(f"Failed to pause watch {watch.watch_id}: {e}")
                print(f"Paused {paused_count} watch(es)")

            elif path:
                print(f"Pausing watch: {path}")
                # Find watch by path or ID
                matches = [
                    w
                    for w in watches
                    if w.watch_id == path or Path(w.path) == Path(path).resolve()
                ]

                if not matches:
                    print(f"Error: No watch found for: {path}")
                    raise typer.Exit(1)

                for watch in matches:
                    try:
                        from ...grpc.ingestion_pb2 import WatchStatus
                        response = await client.configure_watch(
                            watch_id=watch.watch_id,
                            status=WatchStatus.WATCH_STATUS_PAUSED
                        )
                        if response.success:
                            print(f"Paused watch: {watch.path}")
                        else:
                            print(f"Error: Failed to pause watch: {watch.path} - {response.message}")
                    except Exception as e:
                        print(f"Error: Failed to pause watch: {watch.path} - {e}")
            else:
                print("Error: Must specify --all, --collection, or a path/watch ID")
                raise typer.Exit(1)

            print("\nFile monitoring is paused but configurations are preserved")
            print("Resume with: wqm watch resume")

        finally:
            await client.disconnect()

    except Exception as e:
        print(f"Error: Failed to pause watches: {e}")
        raise typer.Exit(1)


async def _resume_watches(path: str | None, collection: str | None, all: bool):
    """Resume watch configurations using daemon client."""
    try:
        client = await _get_daemon_client()
        
        try:
            # Get all watches first
            watches_response = await client.list_watches(active_only=False)
            watches = watches_response.watches

            if all:
                print("Resuming all watches")
                resumed_count = 0
                for watch in watches:
                    try:
                        from ...grpc.ingestion_pb2 import WatchStatus
                        response = await client.configure_watch(
                            watch_id=watch.watch_id,
                            status=WatchStatus.WATCH_STATUS_ACTIVE
                        )
                        if response.success:
                            resumed_count += 1
                    except Exception as e:
                        print(f"Failed to resume watch {watch.watch_id}: {e}")
                print(f"Resumed {resumed_count} watch(es)")

            elif collection:
                print(f"Resuming watches for collection: {collection}")
                matching_watches = [w for w in watches if w.collection == collection]
                resumed_count = 0
                for watch in matching_watches:
                    try:
                        from ...grpc.ingestion_pb2 import WatchStatus
                        response = await client.configure_watch(
                            watch_id=watch.watch_id,
                            status=WatchStatus.WATCH_STATUS_ACTIVE
                        )
                        if response.success:
                            resumed_count += 1
                    except Exception as e:
                        print(f"Failed to resume watch {watch.watch_id}: {e}")
                print(f"Resumed {resumed_count} watch(es)")

            elif path:
                print(f"Resuming watch: {path}")
                # Find watch by path or ID
                matches = [
                    w
                    for w in watches
                    if w.watch_id == path or Path(w.path) == Path(path).resolve()
                ]

                if not matches:
                    print(f"Error: No watch found for: {path}")
                    raise typer.Exit(1)

                for watch in matches:
                    try:
                        from ...grpc.ingestion_pb2 import WatchStatus
                        response = await client.configure_watch(
                            watch_id=watch.watch_id,
                            status=WatchStatus.WATCH_STATUS_ACTIVE
                        )
                        if response.success:
                            print(f"Resumed watch: {watch.path}")
                        else:
                            print(f"Error: Failed to resume watch: {watch.path} - {response.message}")
                    except Exception as e:
                        print(f"Error: Failed to resume watch: {watch.path} - {e}")
            else:
                print("Error: Must specify --all, --collection, or a path/watch ID")
                raise typer.Exit(1)

            print("\nFile monitoring resumed - new files will be automatically ingested")

        finally:
            await client.disconnect()

    except Exception as e:
        print(f"Error: Failed to resume watches: {e}")
        raise typer.Exit(1)


async def _sync_watched_folders(path: str | None, dry_run: bool, force: bool):
    """Manually sync watched folders using daemon client."""
    try:
        client = await _get_daemon_client()
        
        try:
            # Get all watches
            watches_response = await client.list_watches(active_only=False)
            watches = watches_response.watches

            if path:
                print(f"Syncing watch: {path}")
                # Filter to specific path
                matches = [
                    w for w in watches 
                    if w.watch_id == path or Path(w.path) == Path(path).resolve()
                ]
                if not matches:
                    print(f"Error: No watch found for path: {path}")
                    raise typer.Exit(1)
                watches = matches
            else:
                print("Syncing all watched folders")

            if dry_run:
                print("DRY RUN - No files will be processed")

            # Process each watch by processing its folder
            total_processed = 0
            total_errors = 0

            for watch in watches:
                print(f"\nProcessing watch: {watch.path} -> {watch.collection}")
                
                try:
                    # Use process_folder to sync the watched directory
                    folder_progress = client.process_folder(
                        folder_path=watch.path,
                        collection=watch.collection,
                        include_patterns=list(watch.patterns) if watch.patterns else None,
                        ignore_patterns=list(watch.ignore_patterns) if watch.ignore_patterns else None,
                        recursive=watch.recursive,
                        dry_run=dry_run,
                    )
                    
                    files_processed = 0
                    files_failed = 0
                    
                    async for progress in folder_progress:
                        if progress.file_processed:
                            files_processed += 1
                            if progress.success:
                                print(f"  ✓ {progress.file_path}")
                            else:
                                files_failed += 1
                                print(f"  ✗ {progress.file_path}: {progress.error_message}")
                    
                    total_processed += files_processed
                    total_errors += files_failed
                    
                    print(f"Watch {watch.watch_id}: Processed {files_processed} files")
                    if files_failed > 0:
                        print(f"  {files_failed} files failed")
                        
                except Exception as e:
                    print(f"Error processing watch {watch.watch_id}: {e}")
                    total_errors += 1

            # Summary
            if dry_run:
                print("\nSync Preview Summary")
                print(f"Would process {total_processed} files")
            else:
                print("\nSync Summary")
                print(f"Processed {total_processed} files")

            if total_errors > 0:
                print(f"Errors: {total_errors}")

            if not watches:
                print("\nNo watched folders found to sync")
                print("Add watches with: wqm watch add <path> --collection=<library>")

        finally:
            await client.disconnect()

    except Exception as e:
        print(f"Error: Failed to sync watches: {e}")
        raise typer.Exit(1)
