"""File watching CLI commands.

This module provides management for library folder watching,
enabling automatic ingestion of files into library collections.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import typer
from common.core.sqlite_state_manager import SQLiteStateManager, WatchFolderConfig
from common.grpc.daemon_client import get_daemon_client
from loguru import logger

# Import PatternManager for default patterns
try:
    from common.core.pattern_manager import PatternManager

    def _get_cli_default_patterns() -> list[str]:
        """Get default patterns for CLI watch commands."""
        try:
            PatternManager()
            # TODO: Get from PatternManager in future, for now use compatible defaults
            return ["*.pdf", "*.epub", "*.txt", "*.md"]
        except Exception as e:
            logger.debug(f"Failed to load PatternManager, using fallback patterns: {e}")
            return ["*.pdf", "*.epub", "*.txt", "*.md"]

    def _get_cli_default_ignore_patterns() -> list[str]:
        """Get default ignore patterns for CLI watch commands."""
        try:
            PatternManager()
            # TODO: Get from PatternManager in future, for now use compatible defaults
            return [".git/*", "node_modules/*", "__pycache__/*", ".DS_Store"]
        except Exception as e:
            logger.debug(f"Failed to load PatternManager, using fallback ignore patterns: {e}")
            return [".git/*", "node_modules/*", "__pycache__/*", ".DS_Store"]

except ImportError:
    logger.debug("PatternManager not available - using hardcoded CLI patterns")

    def _get_cli_default_patterns() -> list[str]:
        """Fallback default patterns for CLI watch commands."""
        return ["*.pdf", "*.epub", "*.txt", "*.md"]

    def _get_cli_default_ignore_patterns() -> list[str]:
        """Fallback default ignore patterns for CLI watch commands."""
        return [".git/*", "node_modules/*", "__pycache__/*", ".DS_Store"]
from ..formatting import (
    create_data_table,
    display_table_or_empty,
    error_panel,
    info_panel,
    simple_info,
    simple_success,
    simple_warning,
    success_panel,
)
from ..utils import (
    confirm,
    create_command_app,
    handle_async,
)

# logger imported from loguru

# Create the watch app using shared utilities
watch_app = create_command_app(
    name="watch",
    help_text="""Folder watching configuration with configurable depth control.

Configure automatic folder watching for library collections with precise depth control
for optimal performance and resource usage.

Depth Configuration:
    --depth=0     Current directory only (no subdirectories)
    --depth=3     Shallow watching (2-3 levels, good performance)
    --depth=10    Deep watching (10+ levels, moderate performance)
    --depth=-1    Unlimited depth (may impact performance on large structures)

Examples:
    # Basic watching with default depth
    wqm watch add ~/docs --collection=docs

    # Shallow watching for performance (recommended for large directories)
    wqm watch add ~/projects --collection=code --depth=3

    # Deep watching for nested structures
    wqm watch add ~/research --collection=papers --depth=15

    # Unlimited depth (use with caution on large file systems)
    wqm watch add ~/archive --collection=backup --depth=-1

    # Configure existing watch depth
    wqm watch configure watch_abc123 --depth=5

    # List and manage watches
    wqm watch list                      # Show all watch configurations
    wqm watch status --detailed         # Show detailed watch statistics
    wqm watch remove ~/docs             # Stop watching folder""",
    no_args_is_help=True,
)


async def _get_state_manager() -> SQLiteStateManager:
    """Get initialized state manager for watch operations."""
    state_manager = SQLiteStateManager()
    await state_manager.initialize()
    return state_manager


async def _get_daemon_client():
    """Get connected daemon client for daemon lifecycle operations only."""
    client = get_daemon_client()
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
    depth: int = typer.Option(
        -1,
        "--depth",
        help="Maximum directory depth to watch. Examples: 0=current only, 3=shallow (recommended), 10=deep, -1=unlimited (use with caution)"
    ),
    debounce: int = typer.Option(5, "--debounce", help="Debounce time in seconds"),
):
    """Add a folder to watch for automatic ingestion."""
    handle_async(
        _add_watch(path, collection, patterns, ignore, auto_ingest, recursive, depth, debounce)
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


@watch_app.command("configure")
def configure_watch(
    watch_id: str = typer.Argument(..., help="Watch ID or path to configure"),
    depth: int | None = typer.Option(
        None,
        "--depth",
        help="Change maximum directory depth. Examples: 0=current only, 3=shallow, 10=deep, -1=unlimited"
    ),
    patterns: list[str] | None = typer.Option(None, "--pattern", "-p", help="File patterns to watch"),
    ignore: list[str] | None = typer.Option(None, "--ignore", "-i", help="Ignore patterns"),
    auto_ingest: bool | None = typer.Option(None, "--auto/--no-auto", help="Enable/disable automatic ingestion"),
    recursive: bool | None = typer.Option(None, "--recursive/--no-recursive", help="Enable/disable recursive watching"),
    debounce: int | None = typer.Option(None, "--debounce", help="Debounce time in seconds"),
):
    """Configure an existing watch."""
    handle_async(_configure_watch(watch_id, depth, patterns, ignore, auto_ingest, recursive, debounce))


@watch_app.command("sync")
def sync_watched_folders(
    path: str | None = typer.Argument(None, help="Specific path to sync"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be synced"),
    force: bool = typer.Option(False, "--force", help="Force sync all files"),
):
    """Sync watched folders manually."""
    handle_async(_sync_watched_folders(path, dry_run, force))


@watch_app.command("health")
def watch_health(
    status: str | None = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by health status: healthy, degraded, backoff, disabled",
    ),
    collection: str | None = typer.Option(
        None, "--collection", "-c", help="Filter by collection"
    ),
    format: str = typer.Option(
        "table", "--format", "-f", help="Output format: table, json"
    ),
):
    """Show health status and error statistics for all watches.

    Examples:
        wqm watch health                    # Show all watches health
        wqm watch health --status=failing   # Show only failing watches
        wqm watch health --collection=_docs # Show health for collection
    """
    handle_async(_watch_health(status, collection, format))


@watch_app.command("errors")
def watch_errors(
    watch_id: str = typer.Argument(..., help="Watch ID or path to show errors for"),
):
    """Show detailed error information for a specific watch.

    Examples:
        wqm watch errors watch_abc123
        wqm watch errors ~/docs
    """
    handle_async(_watch_errors(watch_id))


@watch_app.command("reset-errors")
def reset_watch_errors(
    watch_id: str = typer.Argument(..., help="Watch ID or path to reset errors for"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Clear error state and reset health status for a watch.

    This resets consecutive_errors to 0, clears backoff_until,
    and sets health_status back to 'healthy'.

    Examples:
        wqm watch reset-errors watch_abc123
        wqm watch reset-errors ~/docs --force
    """
    handle_async(_reset_watch_errors(watch_id, force))


# Async implementation functions
async def _configure_watch(
    watch_id: str,
    depth: int | None,
    patterns: list[str] | None,
    ignore: list[str] | None,
    auto_ingest: bool | None,
    recursive: bool | None,
    debounce: int | None,
):
    """Configure an existing watch using SQLiteStateManager."""
    try:
        state_manager = await _get_state_manager()

        try:
            # First, validate that the watch exists by trying to find it
            all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

            # Find the watch by ID or path
            target_watch = None
            for watch in all_watches:
                if watch.watch_id == watch_id or Path(watch.path) == Path(watch_id).resolve():
                    target_watch = watch
                    break

            if not target_watch:
                error_panel(f"No watch found with ID or path: {watch_id}")
                raise typer.Exit(1)

            # Validate depth parameter if provided using comprehensive validation
            if depth is not None:
                from common.core.depth_validation import (
                    format_depth_display,
                    validate_recursive_depth,
                )
                depth_result = validate_recursive_depth(depth)

                if not depth_result.is_valid:
                    error_panel(depth_result.error_message)
                    raise typer.Exit(1)

                # Show warnings for depth selection
                if depth_result.warnings:
                    for warning in depth_result.warnings:
                        simple_warning(warning)

                # Show recommendations
                if depth_result.recommendations:
                    rec_text = "Depth Recommendations:\n"
                    rec_text += "\n".join(f"  • {rec}" for rec in depth_result.recommendations)
                    info_panel(rec_text, "Recommendations")

            # Build updated configuration
            updated_config = WatchFolderConfig(
                watch_id=target_watch.watch_id,
                path=target_watch.path,
                collection=target_watch.collection,
                patterns=patterns if patterns is not None else target_watch.patterns,
                ignore_patterns=ignore if ignore is not None else target_watch.ignore_patterns,
                auto_ingest=auto_ingest if auto_ingest is not None else target_watch.auto_ingest,
                recursive=recursive if recursive is not None else target_watch.recursive,
                recursive_depth=depth if depth is not None else target_watch.recursive_depth,
                debounce_seconds=float(debounce) if debounce is not None else target_watch.debounce_seconds,
                enabled=target_watch.enabled,
                created_at=target_watch.created_at,
                updated_at=datetime.now(timezone.utc),
                last_scan=target_watch.last_scan,
                metadata=target_watch.metadata,
            )

            config_info = f"Configuring watch: {target_watch.watch_id}\n"
            config_info += f"Path: {target_watch.path}\n"
            config_info += f"Collection: {target_watch.collection}"
            info_panel(config_info, "Watch Configuration")

            # Save configuration to database
            success = await state_manager.save_watch_folder_config(updated_config)

            if success:
                # Show what was changed
                settings_text = "Updated Settings:\n"
                if depth is not None:
                    from common.core.depth_validation import format_depth_display
                    settings_text += f"  Depth: {format_depth_display(depth)}\n"
                if patterns is not None:
                    settings_text += f"  Patterns: {', '.join(patterns)}\n"
                if ignore is not None:
                    settings_text += f"  Ignore patterns: {', '.join(ignore)}\n"
                if auto_ingest is not None:
                    settings_text += f"  Auto-ingest: {'Enabled' if auto_ingest else 'Disabled'}\n"
                if recursive is not None:
                    settings_text += f"  Recursive: {'Yes' if recursive else 'No'}\n"
                if debounce is not None:
                    settings_text += f"  Debounce: {debounce} seconds\n"

                success_panel(settings_text.strip(), "Configuration Updated")

                # Show performance warning for unlimited depth
                if depth == -1:
                    simple_warning("Unlimited depth may impact performance on large directory structures")
                    simple_info("Consider using a specific depth limit for better performance")

            else:
                error_panel("Failed to save watch configuration")
                raise typer.Exit(1)

        finally:
            await state_manager.close()

    except Exception as e:
        error_panel(f"Failed to configure watch: {e}")
        raise typer.Exit(1)


async def _add_watch(
    path: str,
    collection: str,
    patterns: list[str] | None,
    ignore: list[str] | None,
    auto_ingest: bool,
    recursive: bool,
    depth: int,
    debounce: int,
):
    """Add a folder watch configuration using SQLiteStateManager."""
    try:
        watch_path = Path(path).resolve()

        watch_info = f"Path: {watch_path}\nCollection: {collection}"
        info_panel(watch_info, "Adding Watch Configuration")

        state_manager = await _get_state_manager()

        try:
            # Validate path
            if not watch_path.exists():
                error_panel(f"Path does not exist: {path}")
                raise typer.Exit(1)
            if not watch_path.is_dir():
                error_panel(f"Path is not a directory: {path}")
                raise typer.Exit(1)

            # Validate depth parameter using comprehensive validation
            from common.core.depth_validation import (
                format_depth_display,
                validate_recursive_depth,
            )
            depth_result = validate_recursive_depth(depth)

            if not depth_result.is_valid:
                error_panel(depth_result.error_message)
                raise typer.Exit(1)

            # Show warnings for depth selection
            if depth_result.warnings:
                for warning in depth_result.warnings:
                    simple_warning(warning)

            # Show recommendations
            if depth_result.recommendations:
                rec_text = "Recommendations:\n"
                rec_text += "\n".join(f"  • {rec}" for rec in depth_result.recommendations)
                info_panel(rec_text, "Depth Recommendations")

            # Validate collection (must be library collection)
            if not collection.startswith("_"):
                error_panel(f"Collection must start with underscore (library collection): {collection}")
                raise typer.Exit(1)

            # Set defaults if not provided
            if patterns is None:
                patterns = _get_cli_default_patterns()
            if ignore is None:
                ignore = _get_cli_default_ignore_patterns()

            # Validate collection exists using daemon client
            client = await _get_daemon_client()
            try:
                collections_response = await client.list_collections()
                collection_names = [c.name for c in collections_response.collections]

                if collection not in collection_names:
                    error_panel(f"Collection '{collection}' not found. Create it first with: wqm library create {collection[1:]}")
                    raise typer.Exit(1)
            finally:
                await client.disconnect()

            # Generate unique watch ID
            watch_id = f"watch_{uuid.uuid4().hex[:8]}"

            # Create watch configuration
            watch_config = WatchFolderConfig(
                watch_id=watch_id,
                path=str(watch_path),
                collection=collection,
                patterns=patterns,
                ignore_patterns=ignore,
                auto_ingest=auto_ingest,
                recursive=recursive,
                recursive_depth=depth,
                debounce_seconds=float(debounce),
                enabled=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                last_scan=None,
                metadata=None,
            )

            # Save to database
            success = await state_manager.save_watch_folder_config(watch_config)

            if success:
                simple_success(f"Watch started with ID: {watch_id}")

                # Show configuration
                config_text = f"Path: {watch_path}\n"
                config_text += f"Collection: {collection}\n"
                config_text += f"Watch ID: {watch_id}\n"
                config_text += f"Auto-ingest: {'Enabled' if auto_ingest else 'Disabled'}\n"
                config_text += f"Recursive: {'Yes' if recursive else 'No'}\n"
                config_text += f"Depth: {format_depth_display(depth)}\n"
                config_text += f"Debounce: {debounce} seconds\n\n"

                config_text += "File Patterns:\n"
                for pattern in patterns:
                    config_text += f"  • {pattern}\n"

                config_text += "\nIgnore Patterns:\n"
                for ignore_pattern in ignore:
                    config_text += f"  • {ignore_pattern}"

                success_panel(config_text, "Watch Configuration Added")

                if auto_ingest:
                    simple_info("File monitoring active - daemon will poll database for changes")
                    simple_info("New files will be automatically ingested into the collection")
                else:
                    simple_warning("Auto-ingest is disabled")
                    simple_info("Files will be detected but not automatically processed")
                    simple_info("Enable with: wqm watch resume")
            else:
                error_panel("Failed to save watch configuration")
                raise typer.Exit(1)

        finally:
            await state_manager.close()

    except Exception as e:
        error_panel(f"Failed to add watch: {e}")
        raise typer.Exit(1)


async def _list_watches(active_only: bool, collection: str | None, format: str):
    """List all watch configurations using SQLiteStateManager."""
    try:
        state_manager = await _get_state_manager()

        try:
            # Get all watches from database
            all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

            if format == "json":
                # JSON output
                output = []
                for watch in all_watches:
                    watch_dict = {
                        "watch_id": watch.watch_id,
                        "path": watch.path,
                        "collection": watch.collection,
                        "enabled": watch.enabled,
                        "patterns": watch.patterns,
                        "ignore_patterns": watch.ignore_patterns,
                        "auto_ingest": watch.auto_ingest,
                        "recursive": watch.recursive,
                        "recursive_depth": watch.recursive_depth,
                        "debounce_seconds": watch.debounce_seconds,
                    }
                    output.append(watch_dict)
                print(json.dumps(output, indent=2))
                return

            # Filter by enabled status if active_only
            if active_only:
                all_watches = [w for w in all_watches if w.enabled]

            # Filter by collection if specified
            if collection:
                all_watches = [w for w in all_watches if w.collection == collection]

            # Table output
            if not all_watches:
                simple_info("No watches found")
                if not active_only:
                    simple_info("Add a watch with: wqm watch add <path> --collection=<library>")
                return

            # Create Rich table
            table = create_data_table(
                f"Watch Configurations ({len(all_watches)} found)",
                ["ID", "Path", "Collection", "Status", "Auto-Ingest"]
            )

            for watch in all_watches:
                status_str = "Active" if watch.enabled else "Paused"
                auto_ingest_str = "Yes" if watch.auto_ingest else "No"

                # Truncate fields if too long
                watch_id = str(watch.watch_id)
                if len(watch_id) > 8:
                    watch_id = watch_id[:8] + "..."

                path = str(watch.path)
                if len(path) > 28:
                    path = path[:28] + "..."

                collection_name = str(watch.collection)
                if len(collection_name) > 18:
                    collection_name = collection_name[:18] + "..."

                table.add_row(
                    watch_id,
                    path,
                    collection_name,
                    status_str,
                    auto_ingest_str
                )

            display_table_or_empty(table, "No watch configurations found.")

            # Show tips
            simple_info("Use 'wqm watch status --detailed' for more information")
            simple_info("Use 'wqm watch sync' to manually process watched directories")

        finally:
            await state_manager.close()

    except Exception as e:
        error_panel(f"Failed to list watches: {e}")
        raise typer.Exit(1)


async def _remove_watch(
    path: str | None, collection: str | None, all: bool, force: bool
):
    """Remove watch configurations using SQLiteStateManager."""
    try:
        state_manager = await _get_state_manager()

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

            # Get all watches from database
            all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

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

            # Remove watches from database
            removed_count = 0
            for watch in watches_to_remove:
                try:
                    success = await state_manager.remove_watch_folder_config(watch.watch_id)
                    if success:
                        removed_count += 1
                        print(f"Removed watch: {watch.path}")
                    else:
                        print(f"Error: Failed to remove watch: {watch.path}")
                except Exception as e:
                    print(f"Error: Failed to remove watch: {watch.path} - {e}")

            print(f"\nSuccessfully removed {removed_count} watch(es)")

        finally:
            await state_manager.close()

    except Exception as e:
        print(f"Error: Failed to remove watches: {e}")
        raise typer.Exit(1)


async def _watch_status(detailed: bool, recent: bool):
    """Show watch activity and statistics using SQLiteStateManager and daemon client."""
    try:
        state_manager = await _get_state_manager()

        try:
            # Get watches from database
            all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

            print("Watch System Status\n")

            # System status overview in plain text format
            total_watches = len(all_watches)
            active_watches = sum(1 for w in all_watches if w.enabled)
            stopped_watches = total_watches - active_watches

            print(f"Total Watches: {total_watches}")
            print(f"Active: {active_watches}")
            print(f"Stopped: {stopped_watches}")

            # Try to get stats from daemon (optional, may not be available)
            try:
                client = await _get_daemon_client()
                try:
                    stats_response = await client.get_stats(include_watch_stats=True)
                    watch_stats = stats_response.watch_stats

                    if watch_stats:
                        print(f"Total Files Monitored: {watch_stats.total_files_monitored}")
                        print(f"Files Processed: {watch_stats.total_files_processed}")
                        print(f"Processing Errors: {watch_stats.processing_errors}")
                finally:
                    await client.disconnect()
            except Exception as e:
                logger.debug(f"Could not get daemon stats: {e}")

            if detailed and all_watches:
                print("\nDetailed Watch Information")
                print("-" * 50)
                for watch in all_watches:
                    status_str = "Active" if watch.enabled else "Paused"
                    print(f"Watch {watch.watch_id}:")
                    print(f"  Path: {watch.path}")
                    print(f"  Collection: {watch.collection}")
                    print(f"  Status: {status_str}")
                    print(f"  Auto-ingest: {'Yes' if watch.auto_ingest else 'No'}")
                    print(f"  Recursive: {'Yes' if watch.recursive else 'No'}")
                    print(f"  Depth: {watch.recursive_depth}")
                    print(f"  Patterns: {', '.join(watch.patterns)}")
                    print()

            if recent:
                print("\nRecent Activity")
                print("-" * 50)
                print("Recent activity tracking available through daemon stats")

            # Show tips
            if total_watches == 0:
                print("\nNo watches configured yet")
                print("Add one with: wqm watch add <path> --collection=<library>")
            elif active_watches == 0:
                print("\nNo watches are currently active")
                print("Resume them with: wqm watch resume --all")

        finally:
            await state_manager.close()

    except Exception as e:
        print(f"Error: Failed to get watch status: {e}")
        raise typer.Exit(1)


async def _pause_watches(path: str | None, collection: str | None, all: bool):
    """Pause watch configurations using SQLiteStateManager."""
    try:
        state_manager = await _get_state_manager()

        try:
            # Get all watches from database
            all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

            if all:
                print("Pausing all watches")
                paused_count = 0
                for watch in all_watches:
                    try:
                        # Update watch to paused state
                        updated_watch = WatchFolderConfig(
                            watch_id=watch.watch_id,
                            path=watch.path,
                            collection=watch.collection,
                            patterns=watch.patterns,
                            ignore_patterns=watch.ignore_patterns,
                            auto_ingest=watch.auto_ingest,
                            recursive=watch.recursive,
                            recursive_depth=watch.recursive_depth,
                            debounce_seconds=watch.debounce_seconds,
                            enabled=False,  # Pause the watch
                            created_at=watch.created_at,
                            updated_at=datetime.now(timezone.utc),
                            last_scan=watch.last_scan,
                            metadata=watch.metadata,
                        )
                        success = await state_manager.save_watch_folder_config(updated_watch)
                        if success:
                            paused_count += 1
                    except Exception as e:
                        print(f"Failed to pause watch {watch.watch_id}: {e}")
                print(f"Paused {paused_count} watch(es)")

            elif collection:
                print(f"Pausing watches for collection: {collection}")
                matching_watches = [w for w in all_watches if w.collection == collection]
                paused_count = 0
                for watch in matching_watches:
                    try:
                        updated_watch = WatchFolderConfig(
                            watch_id=watch.watch_id,
                            path=watch.path,
                            collection=watch.collection,
                            patterns=watch.patterns,
                            ignore_patterns=watch.ignore_patterns,
                            auto_ingest=watch.auto_ingest,
                            recursive=watch.recursive,
                            recursive_depth=watch.recursive_depth,
                            debounce_seconds=watch.debounce_seconds,
                            enabled=False,
                            created_at=watch.created_at,
                            updated_at=datetime.now(timezone.utc),
                            last_scan=watch.last_scan,
                            metadata=watch.metadata,
                        )
                        success = await state_manager.save_watch_folder_config(updated_watch)
                        if success:
                            paused_count += 1
                    except Exception as e:
                        print(f"Failed to pause watch {watch.watch_id}: {e}")
                print(f"Paused {paused_count} watch(es)")

            elif path:
                print(f"Pausing watch: {path}")
                # Find watch by path or ID
                matches = [
                    w
                    for w in all_watches
                    if w.watch_id == path or Path(w.path) == Path(path).resolve()
                ]

                if not matches:
                    print(f"Error: No watch found for: {path}")
                    raise typer.Exit(1)

                for watch in matches:
                    try:
                        updated_watch = WatchFolderConfig(
                            watch_id=watch.watch_id,
                            path=watch.path,
                            collection=watch.collection,
                            patterns=watch.patterns,
                            ignore_patterns=watch.ignore_patterns,
                            auto_ingest=watch.auto_ingest,
                            recursive=watch.recursive,
                            recursive_depth=watch.recursive_depth,
                            debounce_seconds=watch.debounce_seconds,
                            enabled=False,
                            created_at=watch.created_at,
                            updated_at=datetime.now(timezone.utc),
                            last_scan=watch.last_scan,
                            metadata=watch.metadata,
                        )
                        success = await state_manager.save_watch_folder_config(updated_watch)
                        if success:
                            print(f"Paused watch: {watch.path}")
                        else:
                            print(f"Error: Failed to pause watch: {watch.path}")
                    except Exception as e:
                        print(f"Error: Failed to pause watch: {watch.path} - {e}")
            else:
                print("Error: Must specify --all, --collection, or a path/watch ID")
                raise typer.Exit(1)

            print("\nFile monitoring is paused but configurations are preserved")
            print("Resume with: wqm watch resume")

        finally:
            await state_manager.close()

    except Exception as e:
        print(f"Error: Failed to pause watches: {e}")
        raise typer.Exit(1)


async def _resume_watches(path: str | None, collection: str | None, all: bool):
    """Resume watch configurations using SQLiteStateManager."""
    try:
        state_manager = await _get_state_manager()

        try:
            # Get all watches from database
            all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

            if all:
                print("Resuming all watches")
                resumed_count = 0
                for watch in all_watches:
                    try:
                        updated_watch = WatchFolderConfig(
                            watch_id=watch.watch_id,
                            path=watch.path,
                            collection=watch.collection,
                            patterns=watch.patterns,
                            ignore_patterns=watch.ignore_patterns,
                            auto_ingest=watch.auto_ingest,
                            recursive=watch.recursive,
                            recursive_depth=watch.recursive_depth,
                            debounce_seconds=watch.debounce_seconds,
                            enabled=True,  # Resume the watch
                            created_at=watch.created_at,
                            updated_at=datetime.now(timezone.utc),
                            last_scan=watch.last_scan,
                            metadata=watch.metadata,
                        )
                        success = await state_manager.save_watch_folder_config(updated_watch)
                        if success:
                            resumed_count += 1
                    except Exception as e:
                        print(f"Failed to resume watch {watch.watch_id}: {e}")
                print(f"Resumed {resumed_count} watch(es)")

            elif collection:
                print(f"Resuming watches for collection: {collection}")
                matching_watches = [w for w in all_watches if w.collection == collection]
                resumed_count = 0
                for watch in matching_watches:
                    try:
                        updated_watch = WatchFolderConfig(
                            watch_id=watch.watch_id,
                            path=watch.path,
                            collection=watch.collection,
                            patterns=watch.patterns,
                            ignore_patterns=watch.ignore_patterns,
                            auto_ingest=watch.auto_ingest,
                            recursive=watch.recursive,
                            recursive_depth=watch.recursive_depth,
                            debounce_seconds=watch.debounce_seconds,
                            enabled=True,
                            created_at=watch.created_at,
                            updated_at=datetime.now(timezone.utc),
                            last_scan=watch.last_scan,
                            metadata=watch.metadata,
                        )
                        success = await state_manager.save_watch_folder_config(updated_watch)
                        if success:
                            resumed_count += 1
                    except Exception as e:
                        print(f"Failed to resume watch {watch.watch_id}: {e}")
                print(f"Resumed {resumed_count} watch(es)")

            elif path:
                print(f"Resuming watch: {path}")
                # Find watch by path or ID
                matches = [
                    w
                    for w in all_watches
                    if w.watch_id == path or Path(w.path) == Path(path).resolve()
                ]

                if not matches:
                    print(f"Error: No watch found for: {path}")
                    raise typer.Exit(1)

                for watch in matches:
                    try:
                        updated_watch = WatchFolderConfig(
                            watch_id=watch.watch_id,
                            path=watch.path,
                            collection=watch.collection,
                            patterns=watch.patterns,
                            ignore_patterns=watch.ignore_patterns,
                            auto_ingest=watch.auto_ingest,
                            recursive=watch.recursive,
                            recursive_depth=watch.recursive_depth,
                            debounce_seconds=watch.debounce_seconds,
                            enabled=True,
                            created_at=watch.created_at,
                            updated_at=datetime.now(timezone.utc),
                            last_scan=watch.last_scan,
                            metadata=watch.metadata,
                        )
                        success = await state_manager.save_watch_folder_config(updated_watch)
                        if success:
                            print(f"Resumed watch: {watch.path}")
                        else:
                            print(f"Error: Failed to resume watch: {watch.path}")
                    except Exception as e:
                        print(f"Error: Failed to resume watch: {watch.path} - {e}")
            else:
                print("Error: Must specify --all, --collection, or a path/watch ID")
                raise typer.Exit(1)

            print("\nFile monitoring resumed - daemon will poll database for active watches")

        finally:
            await state_manager.close()

    except Exception as e:
        print(f"Error: Failed to resume watches: {e}")
        raise typer.Exit(1)


async def _sync_watched_folders(path: str | None, dry_run: bool, force: bool):
    """Manually sync watched folders using daemon client."""
    try:
        client = await _get_daemon_client()

        try:
            # Get all watches from state manager
            state_manager = await _get_state_manager()
            try:
                all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)
            finally:
                await state_manager.close()

            if path:
                print(f"Syncing watch: {path}")
                # Filter to specific path
                matches = [
                    w for w in all_watches
                    if w.watch_id == path or Path(w.path) == Path(path).resolve()
                ]
                if not matches:
                    print(f"Error: No watch found for path: {path}")
                    raise typer.Exit(1)
                all_watches = matches
            else:
                print("Syncing all watched folders")

            if dry_run:
                print("DRY RUN - No files will be processed")

            # Process each watch by processing its folder
            total_processed = 0
            total_errors = 0

            for watch in all_watches:
                print(f"\nProcessing watch: {watch.path} -> {watch.collection}")

                try:
                    # Use process_folder to sync the watched directory
                    folder_progress = client.process_folder(
                        folder_path=watch.path,
                        collection=watch.collection,
                        include_patterns=watch.patterns if watch.patterns else None,
                        ignore_patterns=watch.ignore_patterns if watch.ignore_patterns else None,
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

            if not all_watches:
                print("\nNo watched folders found to sync")
                print("Add watches with: wqm watch add <path> --collection=<library>")

        finally:
            await client.disconnect()

    except Exception as e:
        print(f"Error: Failed to sync watches: {e}")
        raise typer.Exit(1)


async def _watch_health(status: str | None, collection: str | None, format: str):
    """Show health status and error statistics for all watches (Task 461.14)."""
    try:
        state_manager = await _get_state_manager()

        try:
            # Get watches based on status filter
            if status:
                # Normalize status
                status_map = {
                    "failing": "backoff",
                    "failed": "backoff",
                    "error": "degraded",
                }
                normalized_status = status_map.get(status.lower(), status.lower())

                if normalized_status not in ("healthy", "degraded", "backoff", "disabled"):
                    error_panel(f"Invalid status filter: {status}")
                    simple_info("Valid values: healthy, degraded, backoff, disabled (or: failing, failed, error)")
                    raise typer.Exit(1)

                all_watches = await state_manager.get_watch_folders_by_health_status(normalized_status)
            else:
                all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

            # Filter by collection if specified
            if collection:
                all_watches = [w for w in all_watches if w.collection == collection]

            if format == "json":
                # JSON output
                output = []
                for watch in all_watches:
                    watch_dict = {
                        "watch_id": watch.watch_id,
                        "path": watch.path,
                        "collection": watch.collection,
                        "enabled": watch.enabled,
                        "health_status": watch.health_status,
                        "consecutive_errors": watch.consecutive_errors,
                        "total_errors": watch.total_errors,
                        "last_error_at": watch.last_error_at.isoformat() if watch.last_error_at else None,
                        "last_error_message": watch.last_error_message,
                        "backoff_until": watch.backoff_until.isoformat() if watch.backoff_until else None,
                        "last_success_at": watch.last_success_at.isoformat() if watch.last_success_at else None,
                    }
                    output.append(watch_dict)
                print(json.dumps(output, indent=2))
                return

            # Table output
            if not all_watches:
                if status:
                    simple_info(f"No watches found with status: {status}")
                else:
                    simple_info("No watches found")
                return

            # Health status summary
            health_counts = {"healthy": 0, "degraded": 0, "backoff": 0, "disabled": 0}
            for watch in all_watches:
                health_counts[watch.health_status] = health_counts.get(watch.health_status, 0) + 1

            summary_text = f"Watch Health Summary ({len(all_watches)} total)\n"
            summary_text += f"  Healthy: {health_counts['healthy']}\n"
            summary_text += f"  Degraded: {health_counts['degraded']}\n"
            summary_text += f"  Backoff: {health_counts['backoff']}\n"
            summary_text += f"  Disabled: {health_counts['disabled']}"

            info_panel(summary_text, "Health Overview")

            # Create Rich table
            table = create_data_table(
                "Watch Health Status",
                ["ID", "Path", "Health", "Errors", "Last Error", "Backoff Until"]
            )

            for watch in all_watches:
                # Format watch ID
                watch_id = str(watch.watch_id)
                if len(watch_id) > 12:
                    watch_id = watch_id[:12] + "..."

                # Format path
                path = str(watch.path)
                if len(path) > 25:
                    path = "..." + path[-22:]

                # Format health status with color indicator
                health = watch.health_status.upper()

                # Format errors
                errors = f"{watch.consecutive_errors}/{watch.total_errors}"

                # Format last error time
                if watch.last_error_at:
                    last_error = watch.last_error_at.strftime("%Y-%m-%d %H:%M")
                else:
                    last_error = "-"

                # Format backoff until
                if watch.backoff_until:
                    now = datetime.now(timezone.utc)
                    if watch.backoff_until > now:
                        remaining = watch.backoff_until - now
                        if remaining.total_seconds() > 3600:
                            backoff = f"{int(remaining.total_seconds() // 3600)}h"
                        elif remaining.total_seconds() > 60:
                            backoff = f"{int(remaining.total_seconds() // 60)}m"
                        else:
                            backoff = f"{int(remaining.total_seconds())}s"
                    else:
                        backoff = "expired"
                else:
                    backoff = "-"

                table.add_row(
                    watch_id,
                    path,
                    health,
                    errors,
                    last_error,
                    backoff
                )

            display_table_or_empty(table, "No watches found.")

            # Show tips for unhealthy watches
            if health_counts["backoff"] > 0 or health_counts["disabled"] > 0:
                simple_info("View detailed errors: wqm watch errors <watch_id>")
                simple_info("Reset error state: wqm watch reset-errors <watch_id>")

        finally:
            await state_manager.close()

    except Exception as e:
        error_panel(f"Failed to get watch health: {e}")
        raise typer.Exit(1)


async def _watch_errors(watch_id: str):
    """Show detailed error information for a specific watch (Task 461.14)."""
    try:
        state_manager = await _get_state_manager()

        try:
            # Find the watch by ID or path
            all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

            target_watch = None
            for watch in all_watches:
                if watch.watch_id == watch_id or Path(watch.path) == Path(watch_id).resolve():
                    target_watch = watch
                    break

            if not target_watch:
                error_panel(f"No watch found with ID or path: {watch_id}")
                raise typer.Exit(1)

            # Display watch error details
            info_text = f"Watch ID: {target_watch.watch_id}\n"
            info_text += f"Path: {target_watch.path}\n"
            info_text += f"Collection: {target_watch.collection}"
            info_panel(info_text, "Watch Information")

            # Health status
            health_text = f"Status: {target_watch.health_status.upper()}\n"
            health_text += f"Enabled: {'Yes' if target_watch.enabled else 'No'}\n"

            if target_watch.backoff_until:
                now = datetime.now(timezone.utc)
                if target_watch.backoff_until > now:
                    remaining = target_watch.backoff_until - now
                    health_text += f"Backoff Until: {target_watch.backoff_until.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                    health_text += f"Time Remaining: {int(remaining.total_seconds())} seconds"
                else:
                    health_text += "Backoff: Expired (will resume on next poll)"
            else:
                health_text += "Backoff: None"

            if target_watch.health_status == "healthy":
                success_panel(health_text, "Health Status")
            elif target_watch.health_status == "degraded":
                simple_warning(health_text)
            else:
                error_panel(health_text)

            # Error statistics
            stats_text = f"Consecutive Errors: {target_watch.consecutive_errors}\n"
            stats_text += f"Total Errors: {target_watch.total_errors}\n"

            if target_watch.last_error_at:
                stats_text += f"Last Error: {target_watch.last_error_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            else:
                stats_text += "Last Error: Never"

            info_panel(stats_text, "Error Statistics")

            # Last error message
            if target_watch.last_error_message:
                error_panel(target_watch.last_error_message)
            else:
                simple_info("No error messages recorded")

            # Success information
            if target_watch.last_success_at:
                success_text = f"Last Successful Processing: {target_watch.last_success_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                simple_success(success_text)
            else:
                simple_info("No successful processing recorded")

            # Provide recommendations based on state
            if target_watch.health_status == "backoff":
                simple_info("The watch is in backoff due to repeated errors")
                simple_info("Reset with: wqm watch reset-errors " + target_watch.watch_id)
            elif target_watch.health_status == "disabled":
                simple_warning("This watch has been disabled due to too many errors")
                simple_info("Re-enable with: wqm watch resume " + target_watch.watch_id)
                simple_info("Or reset errors with: wqm watch reset-errors " + target_watch.watch_id)
            elif target_watch.consecutive_errors > 0:
                simple_info(f"Watch has {target_watch.consecutive_errors} consecutive errors")
                simple_info("Errors will reset after successful processing")

        finally:
            await state_manager.close()

    except Exception as e:
        error_panel(f"Failed to get watch errors: {e}")
        raise typer.Exit(1)


async def _reset_watch_errors(watch_id: str, force: bool):
    """Clear error state and reset health status for a watch (Task 461.14)."""
    try:
        state_manager = await _get_state_manager()

        try:
            # Find the watch by ID or path
            all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=False)

            target_watch = None
            for watch in all_watches:
                if watch.watch_id == watch_id or Path(watch.path) == Path(watch_id).resolve():
                    target_watch = watch
                    break

            if not target_watch:
                error_panel(f"No watch found with ID or path: {watch_id}")
                raise typer.Exit(1)

            # Show current state
            current_state = f"Watch ID: {target_watch.watch_id}\n"
            current_state += f"Path: {target_watch.path}\n"
            current_state += f"Current Health: {target_watch.health_status.upper()}\n"
            current_state += f"Consecutive Errors: {target_watch.consecutive_errors}\n"
            current_state += f"Total Errors: {target_watch.total_errors}"

            info_panel(current_state, "Current State")

            # Confirm reset
            if not force:
                if not confirm("Reset error state and set health to 'healthy'?"):
                    simple_info("Operation cancelled")
                    return

            # Reset error state
            success = await state_manager.update_watch_folder_error_state(
                watch_id=target_watch.watch_id,
                consecutive_errors=0,
                health_status="healthy",
                clear_backoff=True,
                # Keep total_errors for statistics, don't reset
            )

            if success:
                success_panel(
                    f"Error state reset for watch: {target_watch.watch_id}\n"
                    f"Consecutive errors: 0\n"
                    f"Health status: healthy\n"
                    f"Backoff: cleared\n"
                    f"Total errors preserved: {target_watch.total_errors}",
                    "Reset Complete"
                )
                simple_info("The watch will resume normal processing on next poll")
            else:
                error_panel("Failed to reset error state")
                raise typer.Exit(1)

        finally:
            await state_manager.close()

    except Exception as e:
        error_panel(f"Failed to reset watch errors: {e}")
        raise typer.Exit(1)
