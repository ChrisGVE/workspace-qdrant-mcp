"""
Enhanced file watching system with persistent configuration integration.

This module provides an enhanced file watcher that integrates with the persistent
configuration system, supporting watch state recovery, automatic startup, and
comprehensive lifecycle management.
"""

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from watchfiles import Change, awatch

from .file_watcher import FileWatcher, WatchEvent
from .watch_config import (
    PersistentWatchConfigManager,
    WatchConfigurationPersistent,
)

logger = logging.getLogger(__name__)


class PersistentFileWatcher(FileWatcher):
    """
    Enhanced file watcher with persistent state management.

    Extends the base FileWatcher to automatically save state changes,
    update statistics, and handle error recovery with persistent storage.
    """

    def __init__(
        self,
        config: WatchConfigurationPersistent,
        config_manager: PersistentWatchConfigManager,
        ingestion_callback: Callable[[str, str], None],
        event_callback: Callable[[WatchEvent], None] | None = None,
    ):
        """
        Initialize persistent file watcher.

        Args:
            config: Persistent watch configuration
            config_manager: Configuration manager for persistence
            ingestion_callback: Callback for file ingestion
            event_callback: Optional callback for watch events
        """
        # Convert persistent config to legacy format for base class
        legacy_config = self._to_legacy_config(config)
        super().__init__(legacy_config, ingestion_callback, event_callback)

        self.persistent_config = config
        self.config_manager = config_manager
        self._last_save_time: Optional[str] = None

    def _to_legacy_config(self, persistent_config: WatchConfigurationPersistent):
        """Convert persistent config to legacy WatchConfiguration format."""
        from .file_watcher import WatchConfiguration

        return WatchConfiguration(
            id=persistent_config.id,
            path=persistent_config.path,
            collection=persistent_config.collection,
            patterns=persistent_config.patterns,
            ignore_patterns=persistent_config.ignore_patterns,
            auto_ingest=persistent_config.auto_ingest,
            recursive=persistent_config.recursive,
            debounce_seconds=persistent_config.debounce_seconds,
            created_at=persistent_config.created_at,
            last_activity=persistent_config.last_activity,
            status=persistent_config.status,
            files_processed=persistent_config.files_processed,
            errors_count=persistent_config.errors_count,
        )

    async def start(self) -> None:
        """Start watching with persistent state updates."""
        await super().start()

        # Update persistent config status
        self.persistent_config.status = "active"
        self.persistent_config.last_activity = datetime.now(timezone.utc).isoformat()
        await self._save_config_state()

    async def stop(self) -> None:
        """Stop watching with persistent state updates."""
        await super().stop()

        # Update persistent config status
        self.persistent_config.status = "paused"
        await self._save_config_state()

    async def pause(self) -> None:
        """Pause watcher with persistent state update."""
        await super().pause()

        self.persistent_config.status = "paused"
        await self._save_config_state()

    async def resume(self) -> None:
        """Resume watcher with persistent state update."""
        await super().resume()

        if self.persistent_config.status == "paused":
            self.persistent_config.status = "active"
            self.persistent_config.last_activity = datetime.now(
                timezone.utc
            ).isoformat()
            await self._save_config_state()

    async def _trigger_ingestion(self, file_path: str) -> None:
        """Enhanced ingestion trigger with statistics tracking."""
        try:
            await super()._trigger_ingestion(file_path)

            # Update statistics on successful ingestion
            self.persistent_config.files_processed += 1
            self.persistent_config.last_activity = datetime.now(
                timezone.utc
            ).isoformat()

            # Periodic save to avoid too frequent writes
            await self._save_config_state(force=False)

        except Exception as e:
            # Update error statistics on failure
            self.persistent_config.errors_count += 1
            self.persistent_config.status = "error"
            await self._save_config_state(force=True)
            raise

    async def _save_config_state(self, force: bool = False) -> None:
        """Save current configuration state to persistent storage."""
        try:
            current_time = datetime.now(timezone.utc).isoformat()

            # Save periodically or when forced
            if (
                force
                or not self._last_save_time
                or (
                    datetime.fromisoformat(current_time.replace("Z", "+00:00"))
                    - datetime.fromisoformat(
                        self._last_save_time.replace("Z", "+00:00")
                    )
                ).total_seconds()
                > 30
            ):  # Save at most once every 30 seconds
                await self.config_manager.update_watch_config(self.persistent_config)
                self._last_save_time = current_time
                logger.debug(
                    f"Saved persistent state for watch {self.persistent_config.id}"
                )

        except Exception as e:
            logger.error(f"Failed to save persistent config state: {e}")


class PersistentWatchManager:
    """
    Enhanced watch manager with persistent configuration and state recovery.

    Provides comprehensive watch lifecycle management with automatic startup,
    state recovery, and persistent configuration integration.
    """

    def __init__(
        self,
        config_manager: Optional[PersistentWatchConfigManager] = None,
        project_dir: Optional[Path] = None,
    ):
        """
        Initialize persistent watch manager.

        Args:
            config_manager: Configuration manager (creates default if not provided)
            project_dir: Project directory for project-specific config
        """
        self.config_manager = config_manager or PersistentWatchConfigManager(
            project_dir=project_dir
        )
        self.watchers: Dict[str, PersistentFileWatcher] = {}
        self.ingestion_callback: Optional[Callable[[str, str], None]] = None
        self.event_callback: Optional[Callable[[WatchEvent], None]] = None
        self._initialized = False
        self._startup_recovery_completed = False

    def set_ingestion_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set the ingestion callback for all watchers."""
        self.ingestion_callback = callback

        # Update existing watchers
        for watcher in self.watchers.values():
            watcher.ingestion_callback = callback

    def set_event_callback(self, callback: Callable[[WatchEvent], None]) -> None:
        """Set the event callback for all watchers."""
        self.event_callback = callback

        # Update existing watchers
        for watcher in self.watchers.values():
            watcher.event_callback = callback

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the watch manager and recover persistent state.

        Returns:
            dict: Initialization results with recovery statistics
        """
        if self._initialized:
            return {"status": "already_initialized"}

        try:
            logger.info("Initializing persistent watch manager...")

            # Perform startup recovery
            recovery_results = await self._startup_recovery()

            self._initialized = True
            self._startup_recovery_completed = True

            logger.info(
                f"Watch manager initialized with {len(self.watchers)} recovered watches"
            )

            return {
                "status": "initialized",
                "recovery_results": recovery_results,
                "active_watches": len(self.watchers),
                "config_file": str(self.config_manager.get_config_file_path()),
            }

        except Exception as e:
            logger.error(f"Failed to initialize watch manager: {e}")
            return {"status": "error", "error": str(e)}

    async def _startup_recovery(self) -> Dict[str, Any]:
        """Recover watch state from persistent configuration."""
        recovery_stats = {
            "total_configs": 0,
            "recovered_active": 0,
            "recovered_paused": 0,
            "validation_errors": 0,
            "recovery_errors": 0,
            "skipped_configs": [],
        }

        try:
            # Load all persistent configurations
            configs = await self.config_manager.list_watch_configs()
            recovery_stats["total_configs"] = len(configs)

            logger.info(f"Attempting to recover {len(configs)} watch configurations")

            for config in configs:
                try:
                    # Validate configuration
                    validation_issues = config.validate()
                    if validation_issues:
                        logger.warning(
                            f"Skipping invalid config {config.id}: {', '.join(validation_issues)}"
                        )
                        recovery_stats["validation_errors"] += 1
                        recovery_stats["skipped_configs"].append(
                            {
                                "id": config.id,
                                "reason": "validation_failed",
                                "issues": validation_issues,
                            }
                        )
                        continue

                    # Skip if ingestion callback not set and auto_ingest is enabled
                    if config.auto_ingest and not self.ingestion_callback:
                        logger.info(f"Skipping {config.id}: ingestion callback not set")
                        recovery_stats["skipped_configs"].append(
                            {"id": config.id, "reason": "no_ingestion_callback"}
                        )
                        continue

                    # Create persistent watcher
                    watcher = PersistentFileWatcher(
                        config=config,
                        config_manager=self.config_manager,
                        ingestion_callback=self.ingestion_callback,
                        event_callback=self.event_callback,
                    )

                    # Start watcher if it was active
                    if config.status == "active":
                        await watcher.start()
                        recovery_stats["recovered_active"] += 1
                        logger.info(
                            f"Recovered active watch: {config.id} ({config.path})"
                        )
                    else:
                        # Store paused watcher for potential later activation
                        recovery_stats["recovered_paused"] += 1
                        logger.info(
                            f"Recovered paused watch: {config.id} ({config.path})"
                        )

                    self.watchers[config.id] = watcher

                except Exception as e:
                    logger.error(f"Failed to recover watch {config.id}: {e}")
                    recovery_stats["recovery_errors"] += 1
                    recovery_stats["skipped_configs"].append(
                        {"id": config.id, "reason": "recovery_error", "error": str(e)}
                    )

            logger.info(
                f"Recovery completed: {recovery_stats['recovered_active']} active, "
                f"{recovery_stats['recovered_paused']} paused, "
                f"{recovery_stats['validation_errors']} validation errors, "
                f"{recovery_stats['recovery_errors']} recovery errors"
            )

        except Exception as e:
            logger.error(f"Startup recovery failed: {e}")
            recovery_stats["recovery_errors"] += 1

        return recovery_stats

    async def start_watch(self, watch_id: str) -> bool:
        """Start a specific watch by ID."""
        if watch_id in self.watchers:
            try:
                await self.watchers[watch_id].start()
                return True
            except Exception as e:
                logger.error(f"Failed to start watch {watch_id}: {e}")
                return False
        else:
            # Try to load from persistent config
            config = await self.config_manager.get_watch_config(watch_id)
            if not config:
                return False

            try:
                watcher = PersistentFileWatcher(
                    config=config,
                    config_manager=self.config_manager,
                    ingestion_callback=self.ingestion_callback,
                    event_callback=self.event_callback,
                )

                await watcher.start()
                self.watchers[watch_id] = watcher
                return True

            except Exception as e:
                logger.error(f"Failed to create and start watch {watch_id}: {e}")
                return False

    async def stop_watch(self, watch_id: str) -> bool:
        """Stop a specific watch by ID."""
        if watch_id in self.watchers:
            try:
                await self.watchers[watch_id].stop()
                return True
            except Exception as e:
                logger.error(f"Failed to stop watch {watch_id}: {e}")
                return False
        return False

    async def pause_watch(self, watch_id: str) -> bool:
        """Pause a specific watch by ID."""
        if watch_id in self.watchers:
            try:
                await self.watchers[watch_id].pause()
                return True
            except Exception as e:
                logger.error(f"Failed to pause watch {watch_id}: {e}")
                return False
        return False

    async def resume_watch(self, watch_id: str) -> bool:
        """Resume a specific watch by ID."""
        if watch_id in self.watchers:
            try:
                await self.watchers[watch_id].resume()
                return True
            except Exception as e:
                logger.error(f"Failed to resume watch {watch_id}: {e}")
                return False
        else:
            # Try to start from persistent config
            return await self.start_watch(watch_id)

    async def remove_watch(self, watch_id: str) -> bool:
        """Remove a watch completely (stop and delete from config)."""
        # Stop watcher if running
        if watch_id in self.watchers:
            try:
                await self.watchers[watch_id].stop()
                del self.watchers[watch_id]
            except Exception as e:
                logger.error(f"Error stopping watch during removal: {e}")

        # Remove from persistent configuration
        return await self.config_manager.remove_watch_config(watch_id)

    async def start_all_active_watches(self) -> Dict[str, bool]:
        """Start all watches that should be active."""
        if not self.ingestion_callback:
            logger.warning("Cannot start watches: ingestion callback not set")
            return {}

        results = {}
        configs = await self.config_manager.list_watch_configs()

        for config in configs:
            if config.status == "active":
                success = await self.start_watch(config.id)
                results[config.id] = success

        return results

    async def stop_all_watches(self) -> Dict[str, bool]:
        """Stop all running watches."""
        results = {}

        for watch_id in list(self.watchers.keys()):
            success = await self.stop_watch(watch_id)
            results[watch_id] = success

        return results

    def get_watch_runtime_status(self) -> Dict[str, Dict[str, Any]]:
        """Get runtime status of all watches."""
        status = {}

        for watch_id, watcher in self.watchers.items():
            status[watch_id] = {
                "is_running": watcher.is_running(),
                "config_status": watcher.persistent_config.status,
                "path": watcher.persistent_config.path,
                "collection": watcher.persistent_config.collection,
                "files_processed": watcher.persistent_config.files_processed,
                "errors_count": watcher.persistent_config.errors_count,
                "last_activity": watcher.persistent_config.last_activity,
            }

        return status

    async def validate_all_watches(self) -> Dict[str, List[str]]:
        """Validate all watch configurations."""
        return await self.config_manager.validate_all_configs()

    async def cleanup(self) -> None:
        """Clean up all resources."""
        logger.info("Cleaning up persistent watch manager...")

        # Stop all watchers
        await self.stop_all_watches()

        # Clear watcher references
        self.watchers.clear()

        logger.info("Persistent watch manager cleanup completed")

    def is_initialized(self) -> bool:
        """Check if the manager has been initialized."""
        return self._initialized

    def get_config_manager(self) -> PersistentWatchConfigManager:
        """Get the configuration manager instance."""
        return self.config_manager
