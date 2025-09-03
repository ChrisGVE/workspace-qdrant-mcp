"""
MCP tools for persistent folder watching management.

This module provides MCP tool functions for managing persistent folder watches,
including adding, removing, listing, and configuring watch settings with
proper validation and error handling.
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.client import QdrantWorkspaceClient
from ..core.persistent_file_watcher import PersistentWatchManager
from ..core.watch_config import (
    PersistentWatchConfigManager,
    WatchConfigurationPersistent,
)
from ..core.watch_sync import (
    ConfigChangeEvent,
    SynchronizedWatchConfigManager,
)
from ..core.watch_validation import (
    WatchErrorRecovery,
    WatchHealthMonitor,
    WatchPathValidator,
)

logger = logging.getLogger(__name__)


class WatchToolsManager:
    """Manager for watch-related MCP tools."""

    def __init__(self, workspace_client: QdrantWorkspaceClient):
        """
        Initialize watch tools manager.

        Args:
            workspace_client: The workspace client for database operations
        """
        self.workspace_client = workspace_client
        self.config_manager = SynchronizedWatchConfigManager()
        self.persistent_watch_manager = PersistentWatchManager(self.config_manager)
        self.error_recovery = WatchErrorRecovery()
        self.health_monitor = WatchHealthMonitor(self.error_recovery)
        self._initialized = False

        # Set up event handling for real-time synchronization
        self.config_manager.subscribe_to_changes(self._handle_config_change)

    async def initialize(self) -> Dict[str, Any]:
        """Initialize the watch tools manager and recover persistent state."""
        if self._initialized:
            return {"status": "already_initialized"}

        try:
            # Set up ingestion callback
            async def ingestion_callback(file_path: str, collection: str):
                """Callback for file ingestion."""
                try:
                    # Read file content
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Add document to collection
                    from ..tools.documents import add_document

                    result = await add_document(
                        self.workspace_client,
                        content=content,
                        collection=collection,
                        metadata={
                            "file_path": file_path,
                            "file_name": Path(file_path).name,
                            "ingestion_source": "file_watcher",
                            "ingestion_time": datetime.now(timezone.utc).isoformat(),
                        },
                        chunk_text=True,
                    )

                    if not result.get("success"):
                        logger.error(
                            f"Failed to ingest file {file_path}: {result.get('error')}"
                        )
                        raise RuntimeError(f"Ingestion failed: {result.get('error')}")

                    logger.info(
                        f"Successfully ingested file: {file_path} -> {collection}"
                    )

                except Exception as e:
                    logger.error(f"Error in ingestion callback for {file_path}: {e}")
                    raise

            # Set callbacks
            self.persistent_watch_manager.set_ingestion_callback(ingestion_callback)

            # Initialize the synchronized config manager
            await self.config_manager.initialize()

            # Initialize the persistent watch manager
            init_results = await self.persistent_watch_manager.initialize()

            # Start health monitoring
            try:
                await self.health_monitor.start_monitoring()
                logger.info("Started watch health monitoring")
            except Exception as e:
                logger.error(f"Failed to start health monitoring: {e}")

            self._initialized = True
            return init_results

        except Exception as e:
            logger.error(f"Failed to initialize watch tools manager: {e}")
            return {"status": "error", "error": str(e)}

    async def add_watch_folder(
        self,
        path: str,
        collection: str,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        auto_ingest: bool = True,
        recursive: bool = True,
        recursive_depth: int = -1,
        debounce_seconds: int = 5,
        update_frequency: int = 1000,
        watch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Add a new folder watch configuration.

        Args:
            path: Directory path to watch
            collection: Target Qdrant collection for ingested files
            patterns: File patterns to include (default: common document types)
            ignore_patterns: File patterns to ignore (default: common ignore patterns)
            auto_ingest: Enable automatic ingestion of matched files
            recursive: Watch subdirectories recursively
            recursive_depth: Maximum depth for recursive watching (-1 for unlimited)
            debounce_seconds: Delay before processing file changes (1-300 seconds)
            update_frequency: File system check frequency in milliseconds (100-10000)
            watch_id: Custom watch identifier (auto-generated if not provided)

        Returns:
            dict: Result of the add operation with success status and details
        """
        try:
            # Comprehensive path validation
            watch_path = Path(path).resolve()
            validation_result = WatchPathValidator.validate_watch_path(watch_path)

            if not validation_result.valid:
                return {
                    "success": False,
                    "error": validation_result.error_message,
                    "error_type": validation_result.error_code,
                    "validation_details": validation_result.to_dict(),
                }

            # Log any validation warnings
            if validation_result.warnings:
                logger.warning(
                    f"Validation warnings for path {path}: {', '.join(validation_result.warnings)}"
                )

            # Validate collection exists
            available_collections = await self.workspace_client.list_collections()
            if collection not in available_collections:
                return {
                    "success": False,
                    "error": f"Collection '{collection}' does not exist. Available: {', '.join(available_collections)}",
                    "error_type": "collection_not_found",
                    "available_collections": available_collections,
                }

            # Generate watch ID if not provided
            if not watch_id:
                watch_id = (
                    f"watch_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
                )

            # Check if watch ID already exists
            existing_config = await self.config_manager.get_watch_config(watch_id)
            if existing_config:
                return {
                    "success": False,
                    "error": f"Watch ID already exists: {watch_id}",
                    "error_type": "duplicate_id",
                }

            # Use defaults if not provided
            if patterns is None:
                patterns = ["*.pdf", "*.epub", "*.txt", "*.md", "*.docx", "*.rtf"]

            if ignore_patterns is None:
                ignore_patterns = [
                    ".git/*",
                    ".svn/*",
                    ".hg/*",
                    "node_modules/*",
                    "__pycache__/*",
                    ".pyc",
                    ".DS_Store",
                    "Thumbs.db",
                    "*.tmp",
                    "*.temp",
                    "*.log",
                ]

            # Create watch configuration
            watch_config = WatchConfigurationPersistent(
                id=watch_id,
                path=str(watch_path),
                collection=collection,
                patterns=patterns,
                ignore_patterns=ignore_patterns,
                auto_ingest=auto_ingest,
                recursive=recursive,
                recursive_depth=recursive_depth,
                debounce_seconds=debounce_seconds,
                update_frequency=update_frequency,
                status="active",
                created_at=datetime.now(timezone.utc).isoformat(),
                files_processed=0,
                errors_count=0,
            )

            # Validate configuration
            validation_issues = watch_config.validate()
            if validation_issues:
                return {
                    "success": False,
                    "error": f"Configuration validation failed: {'; '.join(validation_issues)}",
                    "error_type": "validation_error",
                    "validation_issues": validation_issues,
                }

            # Initialize if not already done
            if not self._initialized:
                await self.initialize()

            # Save configuration
            success = await self.config_manager.add_watch_config(watch_config)
            if not success:
                return {
                    "success": False,
                    "error": "Failed to save watch configuration",
                    "error_type": "save_error",
                }

            # Register with health monitor
            self.health_monitor.register_watch(watch_id, watch_path)

            # Start the watch if auto_ingest is enabled
            if auto_ingest and self._initialized:
                try:
                    await self.persistent_watch_manager.start_watch(watch_id)
                    logger.info(f"Started active watch: {watch_id}")
                except Exception as e:
                    logger.warning(f"Failed to start watch {watch_id}: {e}")
                    # Don't fail the add operation if starting fails
                    # Let error recovery handle this later

            return {
                "success": True,
                "watch_id": watch_id,
                "path": str(watch_path),
                "collection": collection,
                "patterns": patterns,
                "ignore_patterns": ignore_patterns,
                "auto_ingest": auto_ingest,
                "recursive": recursive,
                "recursive_depth": recursive_depth,
                "debounce_seconds": debounce_seconds,
                "update_frequency": update_frequency,
                "status": "active",
                "created_at": watch_config.created_at,
                "message": f"Watch folder added successfully: {watch_id}",
            }

        except Exception as e:
            logger.error(f"Failed to add watch folder: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "internal_error",
            }

    async def remove_watch_folder(self, watch_id: str) -> Dict[str, Any]:
        """
        Remove a folder watch configuration.

        Args:
            watch_id: Unique identifier of the watch to remove

        Returns:
            dict: Result of the remove operation
        """
        try:
            # Check if watch exists
            existing_config = await self.config_manager.get_watch_config(watch_id)
            if not existing_config:
                return {
                    "success": False,
                    "error": f"Watch not found: {watch_id}",
                    "error_type": "watch_not_found",
                }

            # Remove configuration
            success = await self.config_manager.remove_watch_config(watch_id)
            if not success:
                return {
                    "success": False,
                    "error": "Failed to remove watch configuration",
                    "error_type": "remove_error",
                }

            return {
                "success": True,
                "watch_id": watch_id,
                "removed_path": existing_config.path,
                "removed_collection": existing_config.collection,
                "message": f"Watch folder removed successfully: {watch_id}",
            }

        except Exception as e:
            logger.error(f"Failed to remove watch folder: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "internal_error",
            }

    async def list_watched_folders(
        self,
        active_only: bool = False,
        collection: Optional[str] = None,
        include_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        List all configured folder watches.

        Args:
            active_only: Only return active watches (exclude paused/error/disabled)
            collection: Filter by specific collection name
            include_stats: Include processing statistics for each watch

        Returns:
            dict: List of watch configurations and summary information
        """
        try:
            # Get all configurations
            all_configs = await self.config_manager.list_watch_configs()

            # Apply filters
            filtered_configs = []
            for config in all_configs:
                if active_only and config.status != "active":
                    continue
                if collection and config.collection != collection:
                    continue
                filtered_configs.append(config)

            # Convert to dict format for response
            watches = []
            for config in filtered_configs:
                watch_info = {
                    "id": config.id,
                    "path": config.path,
                    "collection": config.collection,
                    "patterns": config.patterns,
                    "ignore_patterns": config.ignore_patterns,
                    "auto_ingest": config.auto_ingest,
                    "recursive": config.recursive,
                    "recursive_depth": config.recursive_depth,
                    "debounce_seconds": config.debounce_seconds,
                    "update_frequency": config.update_frequency,
                    "status": config.status,
                    "created_at": config.created_at,
                    "last_activity": config.last_activity,
                }

                if include_stats:
                    watch_info.update(
                        {
                            "files_processed": config.files_processed,
                            "errors_count": config.errors_count,
                        }
                    )

                # Add validation status
                validation_issues = config.validate()
                watch_info["validation_status"] = {
                    "valid": len(validation_issues) == 0,
                    "issues": validation_issues,
                }

                # Check if path still exists
                watch_info["path_exists"] = Path(config.path).exists()

                watches.append(watch_info)

            # Generate summary statistics
            summary = {
                "total_watches": len(all_configs),
                "filtered_watches": len(filtered_configs),
                "active_watches": len([w for w in all_configs if w.status == "active"]),
                "paused_watches": len([w for w in all_configs if w.status == "paused"]),
                "error_watches": len([w for w in all_configs if w.status == "error"]),
                "disabled_watches": len(
                    [w for w in all_configs if w.status == "disabled"]
                ),
            }

            if include_stats:
                summary.update(
                    {
                        "total_files_processed": sum(
                            w.files_processed for w in all_configs
                        ),
                        "total_errors": sum(w.errors_count for w in all_configs),
                    }
                )

            # Get unique collections
            collections = list(set(config.collection for config in all_configs))

            return {
                "success": True,
                "watches": watches,
                "summary": summary,
                "collections": sorted(collections),
                "filters_applied": {
                    "active_only": active_only,
                    "collection": collection,
                    "include_stats": include_stats,
                },
            }

        except Exception as e:
            logger.error(f"Failed to list watched folders: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "internal_error",
            }

    async def configure_watch_settings(
        self,
        watch_id: str,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        auto_ingest: Optional[bool] = None,
        recursive: Optional[bool] = None,
        recursive_depth: Optional[int] = None,
        debounce_seconds: Optional[int] = None,
        update_frequency: Optional[int] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Configure settings for an existing watch.

        Args:
            watch_id: Unique identifier of the watch to configure
            patterns: New file patterns to include (optional)
            ignore_patterns: New file patterns to ignore (optional)
            auto_ingest: Enable/disable automatic ingestion (optional)
            recursive: Enable/disable recursive watching (optional)
            recursive_depth: Set maximum recursive depth (optional)
            debounce_seconds: Set debounce delay (optional)
            update_frequency: Set check frequency in milliseconds (optional)
            status: Set watch status (active/paused/disabled) (optional)

        Returns:
            dict: Result of the configuration update
        """
        try:
            # Get existing configuration
            existing_config = await self.config_manager.get_watch_config(watch_id)
            if not existing_config:
                return {
                    "success": False,
                    "error": f"Watch not found: {watch_id}",
                    "error_type": "watch_not_found",
                }

            # Track changes
            changes_made = {}

            # Update fields if provided
            if patterns is not None:
                changes_made["patterns"] = {
                    "old": existing_config.patterns,
                    "new": patterns,
                }
                existing_config.patterns = patterns

            if ignore_patterns is not None:
                changes_made["ignore_patterns"] = {
                    "old": existing_config.ignore_patterns,
                    "new": ignore_patterns,
                }
                existing_config.ignore_patterns = ignore_patterns

            if auto_ingest is not None:
                changes_made["auto_ingest"] = {
                    "old": existing_config.auto_ingest,
                    "new": auto_ingest,
                }
                existing_config.auto_ingest = auto_ingest

            if recursive is not None:
                changes_made["recursive"] = {
                    "old": existing_config.recursive,
                    "new": recursive,
                }
                existing_config.recursive = recursive

            if recursive_depth is not None:
                changes_made["recursive_depth"] = {
                    "old": existing_config.recursive_depth,
                    "new": recursive_depth,
                }
                existing_config.recursive_depth = recursive_depth

            if debounce_seconds is not None:
                changes_made["debounce_seconds"] = {
                    "old": existing_config.debounce_seconds,
                    "new": debounce_seconds,
                }
                existing_config.debounce_seconds = debounce_seconds

            if update_frequency is not None:
                changes_made["update_frequency"] = {
                    "old": existing_config.update_frequency,
                    "new": update_frequency,
                }
                existing_config.update_frequency = update_frequency

            if status is not None:
                valid_statuses = ["active", "paused", "error", "disabled"]
                if status not in valid_statuses:
                    return {
                        "success": False,
                        "error": f"Invalid status '{status}'. Valid options: {', '.join(valid_statuses)}",
                        "error_type": "invalid_status",
                    }
                changes_made["status"] = {"old": existing_config.status, "new": status}
                existing_config.status = status

            # Check if any changes were made
            if not changes_made:
                return {
                    "success": True,
                    "watch_id": watch_id,
                    "message": "No changes specified",
                    "config": existing_config.to_dict(),
                }

            # Validate updated configuration
            validation_issues = existing_config.validate()
            if validation_issues:
                return {
                    "success": False,
                    "error": f"Configuration validation failed: {'; '.join(validation_issues)}",
                    "error_type": "validation_error",
                    "validation_issues": validation_issues,
                    "changes_attempted": changes_made,
                }

            # Update last activity timestamp
            existing_config.last_activity = datetime.now(timezone.utc).isoformat()

            # Save updated configuration
            success = await self.config_manager.update_watch_config(existing_config)
            if not success:
                return {
                    "success": False,
                    "error": "Failed to save updated watch configuration",
                    "error_type": "save_error",
                }

            return {
                "success": True,
                "watch_id": watch_id,
                "changes_made": changes_made,
                "updated_config": existing_config.to_dict(),
                "message": f"Watch settings updated successfully: {watch_id}",
            }

        except Exception as e:
            logger.error(f"Failed to configure watch settings: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "internal_error",
            }

    async def get_watch_status(self, watch_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed status information for watches.

        Args:
            watch_id: Specific watch ID to get status for (optional, gets all if None)

        Returns:
            dict: Status information for the specified watch or all watches
        """
        try:
            if watch_id:
                # Get status for specific watch
                config = await self.config_manager.get_watch_config(watch_id)
                if not config:
                    return {
                        "success": False,
                        "error": f"Watch not found: {watch_id}",
                        "error_type": "watch_not_found",
                    }

                # Get validation status
                validation_issues = config.validate()
                path_exists = Path(config.path).exists()

                return {
                    "success": True,
                    "watch_id": watch_id,
                    "status": {
                        "config": config.to_dict(),
                        "validation": {
                            "valid": len(validation_issues) == 0,
                            "issues": validation_issues,
                        },
                        "path_exists": path_exists,
                        "is_running": False,  # Will be updated when watcher integration is complete
                    },
                }
            else:
                # Get status for all watches
                configs = await self.config_manager.list_watch_configs()
                statuses = {}

                for config in configs:
                    validation_issues = config.validate()
                    path_exists = Path(config.path).exists()

                    statuses[config.id] = {
                        "config": config.to_dict(),
                        "validation": {
                            "valid": len(validation_issues) == 0,
                            "issues": validation_issues,
                        },
                        "path_exists": path_exists,
                        "is_running": False,  # Will be updated when watcher integration is complete
                    }

                return {
                    "success": True,
                    "total_watches": len(configs),
                    "statuses": statuses,
                }

        except Exception as e:
            logger.error(f"Failed to get watch status: {e}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "internal_error",
            }

    def _get_runtime_info(self, watch_id: str) -> Dict[str, Any]:
        """Get runtime information for a specific watch."""
        if not self._initialized:
            return {"status": "manager_not_initialized"}

        runtime_status = self.persistent_watch_manager.get_watch_runtime_status()
        if watch_id in runtime_status:
            return runtime_status[watch_id]
        else:
            return {"is_running": False, "reason": "not_loaded_in_manager"}

    async def start_all_active_watches(self) -> Dict[str, Any]:
        """Start all watches that should be active."""
        if not self._initialized:
            init_result = await self.initialize()
            if (
                init_result.get("status") != "initialized"
                and init_result.get("status") != "already_initialized"
            ):
                return {"success": False, "error": "Failed to initialize watch manager"}

        try:
            results = await self.persistent_watch_manager.start_all_active_watches()
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def stop_all_watches(self) -> Dict[str, Any]:
        """Stop all running watches."""
        if not self._initialized:
            return {"success": True, "message": "No watches to stop"}

        try:
            results = await self.persistent_watch_manager.stop_all_watches()
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _handle_config_change(self, event: ConfigChangeEvent) -> None:
        """Handle configuration change events for real-time synchronization."""
        try:
            logger.info(
                f"Handling config change: {event.event_type} for {event.watch_id}"
            )

            if event.event_type == "added":
                # Start new watch if it should be active
                if event.new_config and event.new_config.get("status") == "active":
                    await self.persistent_watch_manager.start_watch(event.watch_id)

            elif event.event_type == "removed":
                # Stop and remove watch
                await self.persistent_watch_manager.stop_watch(event.watch_id)
                self.health_monitor.unregister_watch(event.watch_id)

            elif event.event_type == "modified":
                # Restart watch with new configuration if it's running
                if event.watch_id in self.persistent_watch_manager.watchers:
                    await self.persistent_watch_manager.stop_watch(event.watch_id)
                    if event.new_config and event.new_config.get("status") == "active":
                        await self.persistent_watch_manager.start_watch(event.watch_id)

            elif event.event_type == "status_changed":
                # Handle status changes
                new_status = (
                    event.new_config.get("status") if event.new_config else None
                )
                if new_status == "active":
                    await self.persistent_watch_manager.resume_watch(event.watch_id)
                elif new_status == "paused":
                    await self.persistent_watch_manager.pause_watch(event.watch_id)
                elif new_status == "disabled":
                    await self.persistent_watch_manager.stop_watch(event.watch_id)

        except Exception as e:
            logger.error(f"Error handling config change event: {e}")

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self._initialized:
            # Stop health monitoring
            try:
                await self.health_monitor.stop_monitoring()
                logger.info("Stopped watch health monitoring")
            except Exception as e:
                logger.error(f"Error stopping health monitoring: {e}")

            # Clean up config manager
            try:
                await self.config_manager.cleanup()
                logger.info("Cleaned up synchronized config manager")
            except Exception as e:
                logger.error(f"Error cleaning up config manager: {e}")

            # Clean up watch manager
            await self.persistent_watch_manager.cleanup()

        self._initialized = False
