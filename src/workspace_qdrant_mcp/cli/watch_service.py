"""
Watch service integration layer.

This module provides the integration between file watching and the ingestion engine,
handling automatic file processing for library collections.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table

from ..core.client import QdrantWorkspaceClient
from ..core.file_watcher import WatchEvent, WatchManager
from .ingestion_engine import DocumentIngestionEngine

logger = logging.getLogger(__name__)
console = Console()


class WatchService:
    """
    Service that integrates file watching with document ingestion.

    Provides automatic ingestion of files into library collections
    when changes are detected in watched directories.
    """

    def __init__(self, client: QdrantWorkspaceClient):
        """
        Initialize watch service.

        Args:
            client: Qdrant client for collection operations
        """
        self.client = client
        self.watch_manager = WatchManager()
        self.ingestion_engine = DocumentIngestionEngine(client)
        self.activity_log: list[WatchEvent] = []
        self.max_activity_log = 1000

        # Set up callbacks
        self.watch_manager.set_ingestion_callback(self._handle_file_ingestion)
        self.watch_manager.set_event_callback(self._handle_watch_event)

    async def initialize(self) -> None:
        """Initialize the watch service."""
        await self.watch_manager.load_configurations()
        logger.info("Watch service initialized")

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
        """Add a new directory watch with collection validation."""

        # Validate path
        watch_path = Path(path).resolve()
        if not watch_path.exists():
            raise ValueError(f"Path does not exist: {path}")
        if not watch_path.is_dir():
            raise ValueError(f"Path is not a directory: {path}")

        # Validate collection (must be library collection)
        if not collection.startswith('_'):
            raise ValueError(f"Collection must start with underscore (library collection): {collection}")

        # Check if collection exists
        available_collections = await self.client.list_collections()
        if collection not in available_collections:
            raise ValueError(f"Collection '{collection}' not found. Create it first with: wqm library create {collection[1:]}")

        # Add the watch
        watch_id = await self.watch_manager.add_watch(
            path=str(watch_path),
            collection=collection,
            patterns=patterns,
            ignore_patterns=ignore_patterns,
            auto_ingest=auto_ingest,
            recursive=recursive,
            debounce_seconds=debounce_seconds,
        )

        return watch_id

    async def remove_watch(self, watch_id: str) -> bool:
        """Remove a directory watch."""
        return await self.watch_manager.remove_watch(watch_id)

    async def start_all_watches(self) -> None:
        """Start all configured watches."""
        await self.watch_manager.start_all_watches()
        logger.info("All watches started")

    async def stop_all_watches(self) -> None:
        """Stop all running watches."""
        await self.watch_manager.stop_all_watches()
        logger.info("All watches stopped")

    async def pause_watch(self, watch_id: str) -> bool:
        """Pause a specific watch."""
        return await self.watch_manager.pause_watch(watch_id)

    async def resume_watch(self, watch_id: str) -> bool:
        """Resume a specific watch."""
        return await self.watch_manager.resume_watch(watch_id)

    async def list_watches(
        self,
        active_only: bool = False,
        collection: str | None = None,
    ):
        """List all watch configurations."""
        return self.watch_manager.list_watches(active_only, collection)

    async def get_watch_status(self) -> dict[str, Any]:
        """Get comprehensive watch system status."""
        status = self.watch_manager.get_watch_status()

        # Aggregate statistics
        total_watches = len(status)
        active_watches = sum(1 for s in status.values() if s['config']['status'] == 'active')
        paused_watches = sum(1 for s in status.values() if s['config']['status'] == 'paused')
        running_watches = sum(1 for s in status.values() if s['running'])

        total_files_processed = sum(s['config']['files_processed'] for s in status.values())
        total_errors = sum(s['config']['errors_count'] for s in status.values())

        # Recent activity (last 24 hours)
        from datetime import datetime, timedelta, timezone
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_events = [
            event for event in self.activity_log
            if datetime.fromisoformat(event.timestamp.replace('Z', '+00:00')) > cutoff
        ]

        recent_added = sum(1 for e in recent_events if e.change_type == 'added')
        recent_modified = sum(1 for e in recent_events if e.change_type == 'modified')
        recent_deleted = sum(1 for e in recent_events if e.change_type == 'deleted')

        return {
            'total_watches': total_watches,
            'active_watches': active_watches,
            'paused_watches': paused_watches,
            'running_watches': running_watches,
            'monitored_directories': sum(1 for s in status.values() if s['path_exists']),
            'total_files_processed': total_files_processed,
            'total_errors': total_errors,
            'recent_activity': {
                'files_added': recent_added,
                'files_modified': recent_modified,
                'files_deleted': recent_deleted,
            },
            'watches': status,
        }

    async def sync_watched_folders(
        self,
        path: str | None = None,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        """Manually sync watched folders."""
        results = {}

        watches = self.watch_manager.list_watches()

        if path:
            # Sync specific path
            matching_watches = [w for w in watches if Path(w.path) == Path(path).resolve()]
            if not matching_watches:
                return {'error': f'No watch found for path: {path}'}
            watches = matching_watches

        for watch_config in watches:
            try:
                result = await self.ingestion_engine.process_directory(
                    directory_path=watch_config.path,
                    collection=watch_config.collection,
                    formats=None,  # Use all supported formats
                    dry_run=dry_run,
                    recursive=watch_config.recursive,
                    exclude_patterns=watch_config.ignore_patterns,
                )

                results[watch_config.id] = {
                    'success': result.success,
                    'message': result.message,
                    'stats': result.stats.__dict__,
                }

            except Exception as e:
                results[watch_config.id] = {
                    'success': False,
                    'message': f'Sync failed: {e}',
                    'stats': None,
                }

        return results

    def get_recent_activity(self, limit: int = 50) -> list[WatchEvent]:
        """Get recent watch activity events."""
        return self.activity_log[-limit:]

    async def _handle_file_ingestion(self, file_path: str, collection: str) -> None:
        """Handle automatic file ingestion."""
        try:
            logger.info(f"Auto-ingesting file: {file_path} -> {collection}")

            # Create a temporary directory with just this file for processing
            import shutil
            import tempfile
            from pathlib import Path

            file_path_obj = Path(file_path)

            # Verify the file still exists and is readable
            if not file_path_obj.exists():
                logger.warning(f"File no longer exists, skipping ingestion: {file_path}")
                return

            if not file_path_obj.is_file():
                logger.warning(f"Path is not a file, skipping ingestion: {file_path}")
                return

            # Use a temporary directory approach
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir_path = Path(temp_dir)
                temp_file_path = temp_dir_path / file_path_obj.name

                # Copy the file to temp directory
                shutil.copy2(file_path_obj, temp_file_path)

                # Process the temp directory (which contains only our file)
                result = await self.ingestion_engine.process_directory(
                    directory_path=temp_dir_path,
                    collection=collection,
                    formats=None,  # Let the engine determine format
                    dry_run=False,
                    recursive=False,  # Only process this directory level
                    exclude_patterns=None,  # No exclusions needed
                )

                if result.success:
                    logger.info(f"Successfully ingested {file_path}: {result.message}")
                else:
                    logger.error(f"Failed to ingest {file_path}: {result.message}")

        except Exception as e:
            logger.error(f"Error during automatic ingestion of {file_path}: {e}")

    def _handle_watch_event(self, event: WatchEvent) -> None:
        """Handle watch events for logging and monitoring."""
        # Add to activity log
        self.activity_log.append(event)

        # Keep log size manageable
        if len(self.activity_log) > self.max_activity_log:
            self.activity_log = self.activity_log[-self.max_activity_log//2:]

        # Log the event
        logger.debug(f"Watch event: {event.change_type} {event.file_path} -> {event.collection}")


def create_status_table(status_data: dict[str, Any]) -> Table:
    """Create a formatted table showing watch status."""
    table = Table(title="👀 Watch System Status")

    # Add columns
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", justify="right", width=15)
    table.add_column("Details", style="dim", width=30)

    # System overview
    table.add_row("Total Watches", str(status_data['total_watches']), "Configured watches")
    table.add_row("Active", str(status_data['active_watches']), "Ready to monitor")
    table.add_row("Running", str(status_data['running_watches']), "Currently monitoring")
    table.add_row("Paused", str(status_data['paused_watches']), "Temporarily stopped")

    table.add_section()

    # Activity stats
    table.add_row("Files Processed", str(status_data['total_files_processed']), "All time")
    table.add_row("Errors", str(status_data['total_errors']), "Processing failures")

    table.add_section()

    # Recent activity (24h)
    recent = status_data['recent_activity']
    table.add_row("Added (24h)", str(recent['files_added']), "New files detected")
    table.add_row("Modified (24h)", str(recent['files_modified']), "Changed files")
    table.add_row("Deleted (24h)", str(recent['files_deleted']), "Removed files")

    return table


def create_watches_table(watches_status: dict[str, dict[str, Any]]) -> Table:
    """Create a formatted table showing individual watches."""
    table = Table(title="👀 Individual Watch Status")

    table.add_column("ID", style="cyan", width=12)
    table.add_column("Path", style="white", width=35)
    table.add_column("Collection", style="blue", width=20)
    table.add_column("Status", justify="center", width=10)
    table.add_column("Files", justify="right", width=8)
    table.add_column("Errors", justify="right", width=8)

    for watch_id, watch_info in watches_status.items():
        config = watch_info['config']

        # Status indicator
        if watch_info['running']:
            status = "[green]RUNNING[/green]"
        elif config['status'] == 'paused':
            status = "[yellow]PAUSED[/yellow]"
        elif config['status'] == 'error':
            status = "[red]ERROR[/red]"
        else:
            status = "[dim]STOPPED[/dim]"

        # Path display
        path = config['path']
        if len(path) > 30:
            path = "..." + path[-27:]

        table.add_row(
            watch_id[:10] + "..." if len(watch_id) > 10 else watch_id,
            path,
            config['collection'],
            status,
            str(config['files_processed']),
            str(config['errors_count']),
        )

    return table
