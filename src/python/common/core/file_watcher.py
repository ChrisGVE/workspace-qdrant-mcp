"""
File watching system for automatic library collection ingestion.

This module provides file system monitoring capabilities for library collections,
enabling automatic ingestion of new and modified files with sophisticated
language-aware filtering.
"""

import asyncio
import json
from loguru import logger
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from watchfiles import Change, awatch

from .language_filters import LanguageAwareFilter
from .sqlite_state_manager import SQLiteStateManager

# logger imported from loguru


@dataclass
class WatchConfiguration:
    """Configuration for a directory watch."""

    id: str
    path: str
    collection: str
    patterns: list[str] = field(
        default_factory=lambda: ["*.pdf", "*.epub", "*.txt", "*.md"]
    )
    ignore_patterns: list[str] = field(
        default_factory=lambda: [
            ".git/*",
            "node_modules/*",
            "__pycache__/*",
            ".DS_Store",
        ]
    )
    auto_ingest: bool = True
    recursive: bool = True
    debounce_seconds: int = 5
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_activity: str | None = None
    status: str = "active"  # active, paused, error
    files_processed: int = 0
    errors_count: int = 0
    files_filtered: int = 0  # New: track filtered files
    use_language_filtering: bool = True  # New: enable/disable language-aware filtering

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WatchConfiguration":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class WatchEvent:
    """Represents a file system watch event."""

    change_type: str  # added, modified, deleted
    file_path: str
    collection: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class FileWatcher:
    """
    File system watcher for a single directory.

    Monitors a directory for changes and enqueues files for ingestion
    using the state manager queue system.
    """

    def __init__(
        self,
        config: WatchConfiguration,
        state_manager: SQLiteStateManager,
        event_callback: Callable[[WatchEvent], None] | None = None,
        filter_config_path: str | None = None,
    ):
        """
        Initialize file watcher.

        Args:
            config: Watch configuration
            state_manager: SQLite state manager for queue operations
            event_callback: Optional callback for watch events
            filter_config_path: Path to language filter configuration directory
        """
        self.config = config
        self.state_manager = state_manager
        self.event_callback = event_callback
        self._running = False
        self._task: asyncio.Task | None = None
        self._debounce_tasks: dict[str, asyncio.Task] = {}

        # Cache for project root detection (per file path)
        self._project_root_cache: dict[str, Path] = {}

        # Initialize language-aware filtering
        self.language_filter: LanguageAwareFilter | None = None
        if config.use_language_filtering:
            self.language_filter = LanguageAwareFilter(filter_config_path)
            # Load configuration asynchronously when starting

        logger.debug(f"FileWatcher initialized with language filtering: {config.use_language_filtering}")

    async def start(self) -> None:
        """Start watching the directory."""
        if self._running:
            logger.warning(f"Watcher for {self.config.path} is already running")
            return

        # Initialize language filter if enabled
        if self.language_filter:
            try:
                await self.language_filter.load_configuration()
                logger.info(f"Language-aware filtering initialized for watcher {self.config.id}")
            except Exception as e:
                logger.error(f"Failed to initialize language filtering: {e}")
                logger.info("Falling back to basic pattern matching")
                self.language_filter = None

        self._running = True
        self.config.status = "active"

        logger.info(
            f"Starting file watcher for {self.config.path} -> {self.config.collection}"
        )

        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self) -> None:
        """Stop watching the directory."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Cancel any pending debounce tasks
        for task in self._debounce_tasks.values():
            if not task.done():
                task.cancel()
        self._debounce_tasks.clear()

        self.config.status = "paused"
        logger.info(f"Stopped file watcher for {self.config.path}")

    async def pause(self) -> None:
        """Pause the watcher without stopping completely."""
        if self._running:
            await self.stop()
            self.config.status = "paused"

    async def resume(self) -> None:
        """Resume a paused watcher."""
        if not self._running and self.config.status == "paused":
            await self.start()

    def is_running(self) -> bool:
        """Check if watcher is currently running."""
        return self._running and self._task is not None and not self._task.done()

    def get_filter_statistics(self) -> Dict[str, Any]:
        """Get language filtering statistics if available."""
        if self.language_filter:
            stats = self.language_filter.get_statistics()
            return {
                "language_filtering": True,
                "filter_config_summary": self.language_filter.get_configuration_summary(),
                "detailed_stats": stats.to_dict(),
            }
        else:
            return {
                "language_filtering": False,
                "files_filtered": self.config.files_filtered,
                "files_processed": self.config.files_processed,
                "filter_method": "basic_patterns"
            }

    def reset_filter_statistics(self) -> None:
        """Reset filtering statistics."""
        if self.language_filter:
            self.language_filter.reset_statistics()
        self.config.files_filtered = 0
        self.config.files_processed = 0

    def _find_project_root(self, file_path: Path) -> Path:
        """
        Find project root by walking up directory tree looking for .git directory.

        Args:
            file_path: Path to file

        Returns:
            Project root path (directory containing .git) or file's parent directory
        """
        # Check cache first
        file_path_str = str(file_path)
        if file_path_str in self._project_root_cache:
            return self._project_root_cache[file_path_str]

        # Walk up directory tree
        current = file_path.parent if file_path.is_file() else file_path

        while current != current.parent:  # Stop at filesystem root
            if (current / ".git").exists():
                # Found git repository
                self._project_root_cache[file_path_str] = current
                logger.debug(f"Found project root for {file_path}: {current}")
                return current
            current = current.parent

        # No .git found, use file's parent directory
        fallback = file_path.parent if file_path.is_file() else file_path
        self._project_root_cache[file_path_str] = fallback
        logger.debug(f"No .git found for {file_path}, using parent: {fallback}")
        return fallback

    async def _watch_loop(self) -> None:
        """Main watching loop."""
        watch_path = Path(self.config.path)

        if not watch_path.exists():
            logger.error(f"Watch path does not exist: {self.config.path}")
            self.config.status = "error"
            return

        try:
            async for changes in awatch(watch_path, recursive=self.config.recursive):
                if not self._running:
                    break

                await self._handle_changes(changes)

        except Exception as e:
            logger.error(f"Error in watch loop for {self.config.path}: {e}")
            self.config.status = "error"
            self.config.errors_count += 1

    def _determine_operation_type(self, change_type: Change, file_path: Path) -> str:
        """
        Determine the operation type for a file change event.

        This method maps watchfiles Change types to queue operation types,
        handling race conditions where files may be deleted between detection
        and processing.

        Args:
            change_type: The watchfiles Change type (added, modified, deleted)
            file_path: Path to the file being changed

        Returns:
            Operation type string: 'ingest', 'update', or 'delete'

        Logic:
            - Change.added → 'ingest' (new file added to watch directory)
            - Change.modified + file exists → 'update' (existing file modified)
            - Change.modified + file missing → 'delete' (file deleted during processing)
            - Change.deleted → 'delete' (file explicitly deleted)

        Edge cases handled:
            - Race condition: File deleted between event and this check
            - Symlinks: Checked via exists() which follows symlinks by default
            - Broken symlinks: Treated as delete since target is unavailable
            - Permission errors: Default to 'update' to allow queue processor to handle
            - Special files: Filtered earlier in _handle_changes(), won't reach here

        Note:
            This method performs filesystem checks and should be called
            during the debounce period to handle transient file states.
            It's called AFTER filtering, so we know the file is relevant.
        """
        if change_type == Change.added:
            operation = 'ingest'
            logger.debug(f"Operation type 'ingest' determined for added file: {file_path}")
            return operation
        elif change_type == Change.deleted:
            operation = 'delete'
            logger.debug(f"Operation type 'delete' determined for deleted file: {file_path}")
            return operation
        elif change_type == Change.modified:
            # Check if file still exists to disambiguate modify vs delete
            # This handles race conditions where a file is deleted between
            # the event being generated and this method being called
            try:
                # Check for symlinks first
                if file_path.is_symlink():
                    # Symlink exists - check if target exists
                    if file_path.exists():
                        operation = 'update'
                        logger.debug(f"Operation type 'update' determined for modified symlink: {file_path}")
                    else:
                        # Broken symlink - treat as delete
                        operation = 'delete'
                        logger.debug(f"Operation type 'delete' determined for broken symlink: {file_path}")
                elif file_path.exists():
                    operation = 'update'
                    logger.debug(f"Operation type 'update' determined for modified file: {file_path}")
                else:
                    # File was deleted between event and check
                    operation = 'delete'
                    logger.debug(
                        f"Operation type 'delete' determined for modified file "
                        f"(race condition - file no longer exists): {file_path}"
                    )
                return operation
            except (OSError, PermissionError) as e:
                # If we can't check the file (permissions, disk error, etc.),
                # assume it's an update and let the queue processor handle errors
                logger.warning(
                    f"Could not check file existence for {file_path}: {e}. "
                    f"Defaulting to 'update' operation."
                )
                return 'update'
        else:
            # Unknown change type - default to ingest for safety
            logger.warning(f"Unknown change type {change_type} for {file_path}, defaulting to 'ingest'")
            return 'ingest'

    async def _handle_changes(self, changes: set[tuple]) -> None:
        """Handle file system changes."""
        for change_info in changes:
            change_type, file_path = change_info
            file_path = Path(file_path)

            # Handle deletions differently since file no longer exists
            if change_type == Change.deleted:
                await self._handle_deletion(file_path)
                continue

            # Skip if not a file
            if not file_path.is_file():
                continue

            # Use language-aware filtering if available, otherwise fall back to basic pattern matching
            if self.language_filter:
                should_process, reason = self.language_filter.should_process_file(file_path)
                if not should_process:
                    self.config.files_filtered += 1
                    logger.debug(f"Language filter rejected {file_path}: {reason}")
                    continue
            else:
                # Fallback to basic pattern matching
                if not self._matches_patterns(file_path):
                    self.config.files_filtered += 1
                    continue

                if self._matches_ignore_patterns(file_path):
                    self.config.files_filtered += 1
                    continue

            # Create watch event
            change_name = self._get_change_name(change_type)
            event = WatchEvent(
                change_type=change_name,
                file_path=str(file_path),
                collection=self.config.collection,
            )

            # Notify event callback
            if self.event_callback:
                try:
                    self.event_callback(event)
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

            # Handle file ingestion with debouncing
            if (
                change_type in (Change.added, Change.modified)
                and self.config.auto_ingest
            ):
                await self._debounce_ingestion(str(file_path), change_type)

    async def _handle_deletion(self, file_path: Path) -> None:
        """
        Handle file deletion events.

        Deletions are enqueued immediately with high priority for timely
        vector DB cleanup. Since the file is already deleted, we can't read
        its metadata, but we can still determine if it should be processed
        based on file patterns.

        Args:
            file_path: Path to the deleted file
        """
        # Check if we would have processed this file based on patterns
        # Since file is deleted, we can only check the path/extension
        should_process = False

        if self.language_filter:
            # Language filter can check patterns even without file access
            # It will use file extension and path patterns
            should_process, reason = self.language_filter.should_process_file(
                file_path,
                allow_missing=True
            )
            if not should_process:
                self.config.files_filtered += 1
                logger.debug(f"Language filter rejected deletion of {file_path}: {reason}")
                return
        else:
            # Fallback to basic pattern matching (works on path alone)
            if not self._matches_patterns(file_path):
                self.config.files_filtered += 1
                return

            if self._matches_ignore_patterns(file_path):
                self.config.files_filtered += 1
                return

            should_process = True

        if not should_process or not self.config.auto_ingest:
            return

        # Create deletion event
        event = WatchEvent(
            change_type="deleted",
            file_path=str(file_path),
            collection=self.config.collection,
        )

        # Notify event callback
        if self.event_callback:
            try:
                self.event_callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

        # Enqueue deletion with high priority
        try:
            logger.info(
                f"Processing deletion: {file_path} from collection {self.config.collection}"
            )

            await self._trigger_operation(
                str(file_path.resolve()),
                self.config.collection,
                "delete"
            )

            # Update stats
            self.config.files_processed += 1
            self.config.last_activity = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error(f"Error processing deletion for {file_path}: {e}")
            self.config.errors_count += 1

    def _matches_patterns(self, file_path: Path) -> bool:
        """Check if file matches include patterns."""
        for pattern in self.config.patterns:
            if file_path.match(pattern):
                return True
        return False

    def _matches_ignore_patterns(self, file_path: Path) -> bool:
        """Check if file matches ignore patterns."""
        for pattern in self.config.ignore_patterns:
            if file_path.match(pattern):
                return True
        return False

    def _get_change_name(self, change_type: Change) -> str:
        """Convert change type to string."""
        if change_type == Change.added:
            return "added"
        elif change_type == Change.modified:
            return "modified"
        elif change_type == Change.deleted:
            return "deleted"
        else:
            return "unknown"

    async def _debounce_ingestion(self, file_path: str, change_type: Change) -> None:
        """Debounce file ingestion to avoid processing rapid changes."""
        # Cancel existing debounce task for this file
        if file_path in self._debounce_tasks:
            self._debounce_tasks[file_path].cancel()

        # Create new debounce task
        self._debounce_tasks[file_path] = asyncio.create_task(
            self._delayed_ingestion(file_path, change_type)
        )

    async def _delayed_ingestion(self, file_path: str, change_type: Change) -> None:
        """Perform delayed file ingestion after debounce period."""
        try:
            await asyncio.sleep(self.config.debounce_seconds)

            # Trigger ingestion (add/update operations)
            # Determine operation type after debounce period
            # File state may have changed during the delay
            path_obj = Path(file_path)
            operation = self._determine_operation_type(change_type, path_obj)

            # Trigger operation with determined type
            await self._trigger_operation(file_path, self.config.collection, operation)

            # Update stats
            self.config.files_processed += 1
            self.config.last_activity = datetime.now(timezone.utc).isoformat()

        except asyncio.CancelledError:
            # Task was cancelled (normal for debouncing)
            pass
        except Exception as e:
            logger.error(f"Error during delayed ingestion of {file_path}: {e}")
            self.config.errors_count += 1
        finally:
            # Clean up task reference
            self._debounce_tasks.pop(file_path, None)

    async def _trigger_operation(
        self,
        file_path: str,
        collection: str,
        operation: str
    ) -> None:
        """
        Trigger file operation by enqueuing to state manager.

        Replaces the old callback pattern with direct queue operations.

        Args:
            file_path: Absolute path to file
            collection: Target collection name
            operation: Operation type ('ingest', 'update', or 'delete')
        """
        try:
            # Detect project root
            file_path_obj = Path(file_path)
            project_root = self._find_project_root(file_path_obj)

            # Calculate tenant_id and branch
            tenant_id = await self.state_manager.calculate_tenant_id(project_root)
            branch = await self.state_manager.get_current_branch(project_root)

            # Default priority: 5 (NORMAL)
            # Higher priority for deletions
            priority = 8 if operation == "delete" else 5

            # Build metadata with event information
            metadata = {
                "watch_id": self.config.id,
                "watch_path": self.config.path,
                "operation": operation,
                "event_type": "file_change",
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "project_root": str(project_root),
            }

            # Enqueue file for processing
            queue_id = await self.state_manager.enqueue(
                file_path=file_path,
                collection=collection,
                priority=priority,
                tenant_id=tenant_id,
                branch=branch,
                metadata=metadata,
            )

            logger.debug(
                f"Enqueued file for {operation}: {file_path} "
                f"(collection={collection}, priority={priority}, "
                f"tenant={tenant_id}, branch={branch}, queue_id={queue_id})"
            )

        except Exception as e:
            logger.error(f"Error enqueueing file {file_path} for {operation}: {e}")
            raise


class WatchManager:
    """
    Manager for multiple file watchers with persistent configuration.

    Handles configuration persistence, watcher lifecycle management,
    and coordination between multiple directory watches.
    """

    def __init__(
        self,
        config_file: str | None = None,
        filter_config_path: str | None = None
    ):
        """
        Initialize watch manager.

        Args:
            config_file: Path to configuration file (default: ~/.wqm/watches.json)
            filter_config_path: Path to language filter configuration directory
        """
        if config_file:
            self.config_file = Path(config_file)
        else:
            # Default to user's home directory
            self.config_file = Path.home() / ".wqm" / "watches.json"

        self.config_file.parent.mkdir(parents=True, exist_ok=True)

        self.watchers: dict[str, FileWatcher] = {}
        self.configurations: dict[str, WatchConfiguration] = {}
        self.state_manager: SQLiteStateManager | None = None
        self.event_callback: Callable[[WatchEvent], None] | None = None
        self.filter_config_path = filter_config_path

    def set_state_manager(self, state_manager: SQLiteStateManager) -> None:
        """Set the state manager for all watchers."""
        self.state_manager = state_manager

    def set_event_callback(self, callback: Callable[[WatchEvent], None]) -> None:
        """Set the event callback for all watchers."""
        self.event_callback = callback

    async def load_configurations(self) -> None:
        """Load watch configurations from file."""
        if not self.config_file.exists():
            logger.info("No watch configuration file found, starting fresh")
            return

        try:
            with open(self.config_file) as f:
                data = json.load(f)

            for config_data in data.get("watches", []):
                config = WatchConfiguration.from_dict(config_data)
                self.configurations[config.id] = config

            logger.info(f"Loaded {len(self.configurations)} watch configurations")

        except Exception as e:
            logger.error(f"Failed to load watch configurations: {e}")

    async def save_configurations(self) -> None:
        """Save watch configurations to file."""
        try:
            data = {
                "watches": [
                    config.to_dict() for config in self.configurations.values()
                ],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            # Write atomically
            temp_file = self.config_file.with_suffix(".json.tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.replace(self.config_file)

            logger.debug(f"Saved {len(self.configurations)} watch configurations")

        except Exception as e:
            logger.error(f"Failed to save watch configurations: {e}")

    async def add_watch(
        self,
        path: str,
        collection: str,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        auto_ingest: bool = True,
        recursive: bool = True,
        debounce_seconds: int = 5,
    ) -> str:
        """
        Add a new directory watch.

        Returns:
            Watch ID
        """
        # Generate unique ID
        watch_id = (
            f"watch_{len(self.configurations) + 1}_{datetime.now().timestamp():.0f}"
        )

        config = WatchConfiguration(
            id=watch_id,
            path=str(Path(path).resolve()),
            collection=collection,
            patterns=patterns or ["*.pdf", "*.epub", "*.txt", "*.md"],
            ignore_patterns=ignore_patterns
            or [".git/*", "node_modules/*", "__pycache__/*", ".DS_Store"],
            auto_ingest=auto_ingest,
            recursive=recursive,
            debounce_seconds=debounce_seconds,
        )

        self.configurations[watch_id] = config
        await self.save_configurations()

        # Start watching if state_manager is set
        if self.state_manager:
            await self._start_watcher(watch_id)

        logger.info(f"Added watch {watch_id} for {path} -> {collection}")
        return watch_id

    async def remove_watch(self, watch_id: str) -> bool:
        """Remove a directory watch."""
        if watch_id not in self.configurations:
            return False

        # Stop watcher if running
        if watch_id in self.watchers:
            await self.watchers[watch_id].stop()
            del self.watchers[watch_id]

        # Remove configuration
        del self.configurations[watch_id]
        await self.save_configurations()

        logger.info(f"Removed watch {watch_id}")
        return True

    async def start_all_watches(self) -> None:
        """Start all configured watches."""
        if not self.state_manager:
            logger.warning("No state manager set, cannot start watches")
            return

        for watch_id in self.configurations:
            try:
                await self._start_watcher(watch_id)
            except Exception as e:
                logger.error(f"Failed to start watcher {watch_id}: {e}")

    async def stop_all_watches(self) -> None:
        """Stop all running watches."""
        for watcher in self.watchers.values():
            try:
                await watcher.stop()
            except Exception as e:
                logger.error(f"Failed to stop watcher: {e}")
        self.watchers.clear()

    async def pause_watch(self, watch_id: str) -> bool:
        """Pause a specific watch."""
        if watch_id in self.watchers:
            await self.watchers[watch_id].pause()
            return True
        elif watch_id in self.configurations:
            self.configurations[watch_id].status = "paused"
            await self.save_configurations()
            return True
        return False

    async def resume_watch(self, watch_id: str) -> bool:
        """Resume a specific watch."""
        if watch_id in self.watchers:
            await self.watchers[watch_id].resume()
            return True
        elif watch_id in self.configurations:
            await self._start_watcher(watch_id)
            return True
        return False

    async def _start_watcher(self, watch_id: str) -> None:
        """Start a specific watcher."""
        if watch_id not in self.configurations:
            raise ValueError(f"Watch {watch_id} not found")

        config = self.configurations[watch_id]

        # Don't start if already running
        if watch_id in self.watchers and self.watchers[watch_id].is_running():
            return

        # Create and start watcher
        watcher = FileWatcher(
            config=config,
            state_manager=self.state_manager,
            event_callback=self.event_callback,
            filter_config_path=self.filter_config_path,
        )

        await watcher.start()
        self.watchers[watch_id] = watcher

    def get_watch_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all watches."""
        status = {}

        for watch_id, config in self.configurations.items():
            watcher = self.watchers.get(watch_id)

            status[watch_id] = {
                "config": config.to_dict(),
                "running": watcher.is_running() if watcher else False,
                "path_exists": Path(config.path).exists(),
            }

        return status

    def list_watches(
        self,
        active_only: bool = False,
        collection: str | None = None,
    ) -> list[WatchConfiguration]:
        """List watch configurations with optional filtering."""
        configs = list(self.configurations.values())

        if active_only:
            configs = [c for c in configs if c.status == "active"]

        if collection:
            configs = [c for c in configs if c.collection == collection]

        return sorted(configs, key=lambda c: c.created_at)

    def get_all_filter_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics from all watchers."""
        all_stats = {}
        total_files_checked = 0
        total_files_processed = 0
        total_files_filtered = 0

        for watch_id, watcher in self.watchers.items():
            watcher_stats = watcher.get_filter_statistics()
            all_stats[watch_id] = watcher_stats

            if watcher_stats.get("language_filtering", False):
                detailed_stats = watcher_stats.get("detailed_stats", {})
                total_files_checked += detailed_stats.get("total_files_checked", 0)
                total_files_processed += detailed_stats.get("files_processed", 0)
                total_files_filtered += detailed_stats.get("files_filtered_out", 0)
            else:
                total_files_processed += watcher_stats.get("files_processed", 0)
                total_files_filtered += watcher_stats.get("files_filtered", 0)

        return {
            "individual_watchers": all_stats,
            "summary": {
                "total_watchers": len(self.watchers),
                "total_files_checked": total_files_checked,
                "total_files_processed": total_files_processed,
                "total_files_filtered": total_files_filtered,
                "filter_efficiency": total_files_filtered / max(1, total_files_checked) if total_files_checked > 0 else 0.0
            }
        }

    def reset_all_filter_statistics(self) -> None:
        """Reset filtering statistics for all watchers."""
        for watcher in self.watchers.values():
            watcher.reset_filter_statistics()
