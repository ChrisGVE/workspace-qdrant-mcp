
from ...observability import get_logger
logger = get_logger(__name__)
"""File watching CLI commands.

This module provides management for library folder watching,
enabling automatic ingestion of files into library collections.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from ...core.client import QdrantWorkspaceClient
from ...core.config import Config
from ..watch_service import WatchService, create_status_table, create_watches_table

# Create the watch app
watch_app = typer.Typer(
    help="""Folder watching configuration
    
    Configure automatic folder watching for library collections.
    
    Examples:
        wqm watch list                      # Show all watch configurations
        wqm watch add ~/docs --collection=docs  # Watch folder for changes
        wqm watch remove ~/docs             # Stop watching folder
        wqm watch status                    # Show watch service status
        wqm watch enable --name=docs-watch  # Enable specific watch
    """,
    no_args_is_help=True,
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


async def _get_watch_service() -> WatchService:
    """Get an initialized watch service."""
    config = Config()
    client = QdrantWorkspaceClient(config)
    await client.initialize()
    service = WatchService(client)
    await service.initialize()
    return service


@watch_app.command("add")
def add_watch(
    path: str = typer.Argument(..., help="Path to watch"),
    collection: str = typer.Option(..., "--collection", "-c", help="Target collection (must start with _)"),
    patterns: list[str] | None = typer.Option(
        None,
        "--pattern", "-p",
        help="File patterns to watch (default: *.pdf, *.epub, *.txt, *.md)"
    ),
    ignore: list[str] | None = typer.Option(
        None,
        "--ignore", "-i",
        help="Ignore patterns (default: .git/*, node_modules/*, __pycache__/*, .DS_Store)"
    ),
    auto_ingest: bool = typer.Option(True, "--auto/--no-auto", help="Enable automatic ingestion"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Watch subdirectories"),
    debounce: int = typer.Option(5, "--debounce", help="Debounce time in seconds"),
):
    """ Add a folder to watch for automatic ingestion."""
    handle_async(_add_watch(path, collection, patterns, ignore, auto_ingest, recursive, debounce))


@watch_app.command("list")
def list_watches(
    active_only: bool = typer.Option(False, "--active", help="Show only active watches"),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Filter by collection"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json"),
):
    """ Show all active watches."""
    handle_async(_list_watches(active_only, collection, format))


@watch_app.command("remove")
def remove_watch(
    path: str | None = typer.Argument(None, help="Path to stop watching (or watch ID)"),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Remove all watches for collection"),
    all: bool = typer.Option(False, "--all", help="Remove all watches"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """ Stop watching folder."""
    handle_async(_remove_watch(path, collection, all, force))


@watch_app.command("status")
def watch_status(
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed statistics"),
    recent: bool = typer.Option(False, "--recent", help="Show recent activity"),
):
    """ Watch activity and statistics."""
    handle_async(_watch_status(detailed, recent))


@watch_app.command("pause")
def pause_watches(
    path: str | None = typer.Argument(None, help="Specific path to pause (or watch ID)"),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Pause all watches for collection"),
    all: bool = typer.Option(False, "--all", help="Pause all watches"),
):
    """Pause all or specific watches."""
    handle_async(_pause_watches(path, collection, all))


@watch_app.command("resume")
def resume_watches(
    path: str | None = typer.Argument(None, help="Specific path to resume (or watch ID)"),
    collection: str | None = typer.Option(None, "--collection", "-c", help="Resume all watches for collection"),
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
    debounce: int
):
    """Add a folder watch configuration."""
    try:
        watch_path = Path(path).resolve()

        print("Adding watch configuration")
        print(f"Path: {watch_path}")
        print(f"Collection: {collection}")

        # Get watch service
        service = await _get_watch_service()

        # Set defaults if not provided
        if patterns is None:
            patterns = ["*.pdf", "*.epub", "*.txt", "*.md"]
        if ignore is None:
            ignore = [".git/*", "node_modules/*", "__pycache__/*", ".DS_Store"]

        # Add the watch
        watch_id = await service.add_watch(
            path=str(watch_path),
            collection=collection,
            patterns=patterns,
            ignore_patterns=ignore,
            auto_ingest=auto_ingest,
            recursive=recursive,
            debounce_seconds=debounce,
        )

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

        # Start monitoring if auto-ingest is enabled
        if auto_ingest:
            await service.start_all_watches()
            print(f"\nFile monitoring started for {watch_path}")
            print("New files will be automatically ingested into the collection")
        else:
            print("\nWarning: Auto-ingest is disabled")
            print("Files will be detected but not automatically processed")
            print("Enable with: wqm watch resume")

    except Exception as e:
        print(f"Error: Failed to add watch: {e}")
        raise typer.Exit(1)


async def _list_watches(active_only: bool, collection: str | None, format: str):
    """List all watch configurations."""
    try:
        service = await _get_watch_service()
        watches = await service.list_watches(active_only, collection)

        if format == "json":
            # JSON output
            output = [watch.to_dict() for watch in watches]
            print(json.dumps(output, indent=2))
            return

        # Table output
        if not watches:
            print("No watches found")
            if not active_only:
                print("Add a watch with: wqm watch add <path> --collection=<library>")
            return

        # Get status information
        status_data = await service.get_watch_status()
        watches_status = status_data['watches']

        print(f"Watch Configurations ({len(watches)} found)\n")

        # Show summary table
        table = create_watches_table(watches_status)
        # TODO: Convert Rich table to plain text format
        print("[Watch table would be displayed here]")

        # Show tips
        print("\nTip: Use 'wqm watch status --detailed' for more information")
        print("Tip: Use 'wqm watch sync' to manually process watched directories")

    except Exception as e:
        print(f"Error: Failed to list watches: {e}")
        raise typer.Exit(1)


async def _remove_watch(path: str | None, collection: str | None, all: bool, force: bool):
    """Remove watch configurations."""
    try:
        service = await _get_watch_service()

        if all:
            print(" Remove All Watches")
        elif collection:
            print(f" Remove Watches for Collection: {collection}")
        elif path:
            print(f" Remove Watch: {path}")
        else:
            print("Error: Must specify --all, --collection, or a path/watch ID")
            raise typer.Exit(1)

        # Find watches to remove
        watches_to_remove = []
        all_watches = await service.list_watches()

        if all:
            watches_to_remove = all_watches
        elif collection:
            watches_to_remove = [w for w in all_watches if w.collection == collection]
        elif path:
            # Try as watch ID first, then as path
            matches = [w for w in all_watches if w.id == path or Path(w.path) == Path(path).resolve()]
            if matches:
                watches_to_remove = matches
            else:
                print(f"Error: No watch found for: {path}")
                raise typer.Exit(1)

        if not watches_to_remove:
            print("No matching watches found")
            return

        # Show what will be removed
        print(f"\nFound {len(watches_to_remove)} watch(es) to remove:[/yellow]")
        for watch in watches_to_remove:
            print(f"  • {watch.path} -> {watch.collection} ({watch.id})")

        # Confirm removal
        if not force:
            action = "all watches" if all else f"watches for {collection}" if collection else f"watch for {path}"
            if not get_user_confirmation("\nAre you sure you want to remove {action}?"):
                print("Operation cancelled")
                return

        # Remove watches
        removed_count = 0
        for watch in watches_to_remove:
            if await service.remove_watch(watch.id):
                removed_count += 1
                print(f" Removed watch: {watch.path}")
            else:
                print(f"Error: Failed to remove watch: {watch.path}")

        print(f"\nSuccessfully removed {removed_count} watch(es)[/green]")

    except Exception as e:
        print(f"Error: Failed to remove watches: {e}")
        raise typer.Exit(1)


async def _watch_status(detailed: bool, recent: bool):
    """Show watch activity and statistics."""
    try:
        service = await _get_watch_service()
        status_data = await service.get_watch_status()

        print(" Watch System Status\n")

        # System status overview
        table = create_status_table(status_data)
        # TODO: Convert Rich table to plain text format
        print("[Watch table would be displayed here]")

        if detailed and status_data['watches']:
            print("\n")
            watches_table = create_watches_table(status_data['watches'])
            print("Output", data=watches_table)

        if recent:
            print("\n Recent Activity")
            recent_events = service.get_recent_activity(limit=20)

            if not recent_events:
                print("No recent activity")
            else:
                # TODO: Replace Rich table with plain text
                print(f"Last {len(recent_events)} Events:")
                print("-" * 80)

                for event in reversed(recent_events[-20:]):
                    # Format timestamp
                    from datetime import datetime
                    try:
                        ts = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
                        time_str = ts.strftime("%H:%M:%S")
                    except:
                        time_str = event.timestamp[:8]

                    # Shorten file path
                    file_path = event.file_path
                    if len(file_path) > 30:
                        file_path = "..." + file_path[-27:]

                    # Simple text output
                    print(f"  {time_str} {event.change_type} {file_path} -> {event.collection}")

        # Show tips
        if status_data['total_watches'] == 0:
            print("\nNo watches configured yet")
            print("Add one with: wqm watch add <path> --collection=<library>")
        elif status_data['running_watches'] == 0:
            print("\nNo watches are currently running")
            print("Start them with: wqm watch resume --all")

    except Exception as e:
        print(f"Error: Failed to get watch status: {e}")
        raise typer.Exit(1)


async def _pause_watches(path: str | None, collection: str | None, all: bool):
    """Pause watch configurations."""
    try:
        service = await _get_watch_service()

        if all:
            print("Pausing all watches")
            await service.stop_all_watches()
            print(" All watches paused")

        elif collection:
            print(f"Pausing watches for collection: {collection}")
            watches = await service.list_watches(collection=collection)
            paused_count = 0
            for watch in watches:
                if await service.pause_watch(watch.id):
                    paused_count += 1
            print(f" Paused {paused_count} watch(es)[/green]")

        elif path:
            print(f"Pausing watch: {path}")
            # Find watch by path or ID
            all_watches = await service.list_watches()
            matches = [w for w in all_watches if w.id == path or Path(w.path) == Path(path).resolve()]

            if not matches:
                print(f"Error: No watch found for: {path}")
                raise typer.Exit(1)

            for watch in matches:
                if await service.pause_watch(watch.id):
                    print(f" Paused watch: {watch.path}")
                else:
                    print(f"Error: Failed to pause watch: {watch.path}")
        else:
            print("Error: Must specify --all, --collection, or a path/watch ID")
            raise typer.Exit(1)

        print("\nFile monitoring is stopped but configurations are preserved")
        print("Resume with: wqm watch resume")

    except Exception as e:
        print(f"Error: Failed to pause watches: {e}")
        raise typer.Exit(1)


async def _resume_watches(path: str | None, collection: str | None, all: bool):
    """Resume watch configurations."""
    try:
        service = await _get_watch_service()

        if all:
            print("Resuming all watches")
            await service.start_all_watches()
            print(" All watches resumed")

        elif collection:
            print(f"Resuming watches for collection: {collection}")
            watches = await service.list_watches(collection=collection)
            resumed_count = 0
            for watch in watches:
                if await service.resume_watch(watch.id):
                    resumed_count += 1
            print(f" Resumed {resumed_count} watch(es)[/green]")

        elif path:
            print(f"Resuming watch: {path}")
            # Find watch by path or ID
            all_watches = await service.list_watches()
            matches = [w for w in all_watches if w.id == path or Path(w.path) == Path(path).resolve()]

            if not matches:
                print(f"Error: No watch found for: {path}")
                raise typer.Exit(1)

            for watch in matches:
                if await service.resume_watch(watch.id):
                    print(f" Resumed watch: {watch.path}")
                else:
                    print(f"Error: Failed to resume watch: {watch.path}")
        else:
            print("Error: Must specify --all, --collection, or a path/watch ID")
            raise typer.Exit(1)

        print("\nFile monitoring restarted - new files will be automatically ingested")

    except Exception as e:
        print(f"Error: Failed to resume watches: {e}")
        raise typer.Exit(1)


async def _sync_watched_folders(path: str | None, dry_run: bool, force: bool):
    """Manually sync watched folders."""
    try:
        service = await _get_watch_service()

        if path:
            print(f"Syncing watch: {path}")
        else:
            print("Syncing all watched folders")

        if dry_run:
            print("DRY RUN - No files will be processed")

        # Perform sync
        results = await service.sync_watched_folders(path=path, dry_run=dry_run, force=force)

        if 'error' in results:
            print(f"Error: {results['error']}")
            raise typer.Exit(1)

        # Show results
        total_processed = 0
        total_errors = 0

        for watch_id, result in results.items():
            if result['success']:
                stats = result['stats']
                print(f"\n Watch {watch_id}: {result['message']}")
                if stats:
                    total_processed += stats.get('files_processed', 0)
                    if stats.get('files_failed', 0) > 0:
                        print(f"   {stats['files_failed']} files failed")
                        total_errors += stats['files_failed']
            else:
                print(f"\nError: Watch {watch_id}: {result['message']}")
                total_errors += 1

        # Summary
        if dry_run:
            print("\n Sync Preview Summary")
            print(f"Would process {total_processed} files")
        else:
            print("\n Sync Summary")
            print(f"Processed {total_processed} files")

        if total_errors > 0:
            print(f"Errors: {total_errors}")

        # Current alternative note
        if not results:
            print("\nNo watched folders found to sync")
            print("Add watches with: wqm watch add <path> --collection=<library>")

    except Exception as e:
        print(f"Error: Failed to sync watches: {e}")
        raise typer.Exit(1)
