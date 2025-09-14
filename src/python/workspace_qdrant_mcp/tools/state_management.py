"""
State Management Tools for MCP Server.

This module provides MCP tool endpoints for managing SQLite state persistence,
file processing queues, and ingestion analytics.

Key Features:
    - Processing status monitoring and analytics
    - Failed file retry management
    - Queue management and prioritization
    - State cleanup and maintenance operations
    - Crash recovery status reporting
    - Database statistics and health monitoring

Available Tools:
    - get_processing_status: Get comprehensive processing status
    - retry_failed_files: Retry files that failed processing
    - get_queue_stats: Get processing queue statistics
    - cleanup_old_records: Clean up old processing records
    - get_database_stats: Get database size and health statistics
    - process_pending_files: Manually trigger processing of pending files
"""

from typing import Any, Dict, List, Optional

from common.logging.loguru_config import get_logger

from common.core.client import QdrantWorkspaceClient
from common.core.sqlite_state_manager import FileProcessingStatus, ProcessingPriority
from common.core.state_aware_ingestion import get_ingestion_manager
from ..tools.watch_management import WatchToolsManager

logger = get_logger(__name__)


async def get_processing_status(
    workspace_client: QdrantWorkspaceClient, watch_manager: WatchToolsManager
) -> Dict[str, Any]:
    """
    Get comprehensive processing status and analytics.

    Returns:
        Dict containing processing status, queue stats, recent failures,
        and system state information.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Get comprehensive status
        status = await ingestion_manager.get_processing_status()

        return {
            "success": True,
            "status": status,
            "message": "Processing status retrieved successfully",
        }

    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve processing status",
        }


async def retry_failed_files(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    collection: Optional[str] = None,
    max_files: int = 50,
) -> Dict[str, Any]:
    """
    Retry files that failed processing.

    Args:
        collection: Optional collection filter
        max_files: Maximum number of files to retry (default: 50)

    Returns:
        Dict with retry results and counts.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Retry failed files
        result = await ingestion_manager.retry_failed_files(
            collection=collection, max_files=max_files
        )

        if result.get("success"):
            message = f"Scheduled {result['retry_count']} failed files for retry"
        else:
            message = f"Failed to retry files: {result.get('error')}"

        return {
            "success": result.get("success", False),
            "retry_count": result.get("retry_count", 0),
            "total_failed": result.get("total_failed", 0),
            "collection": collection,
            "message": message,
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"Failed to retry failed files: {e}")
        return {
            "success": False,
            "error": str(e),
            "retry_count": 0,
            "message": "Failed to retry failed files",
        }


async def get_queue_stats(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get processing queue statistics.

    Args:
        collection: Optional collection filter

    Returns:
        Dict with queue statistics by priority level.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Get queue stats from state manager
        queue_stats = await ingestion_manager.state_manager.get_queue_stats(collection)

        return {
            "success": True,
            "queue_stats": queue_stats,
            "collection": collection,
            "message": f"Queue contains {queue_stats['total']} pending files",
        }

    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve queue statistics",
        }


async def process_pending_files(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    max_files: Optional[int] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Manually trigger processing of pending files.

    Args:
        max_files: Maximum number of files to process
        collection: Optional collection filter

    Returns:
        Dict with processing results.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Process pending files
        result = await ingestion_manager.process_pending_files(max_files=max_files)

        message = (
            f"Processed {result['processed']} files successfully, "
            f"{result['failed']} failed, {result['total']} total"
        )

        return {
            "success": True,
            "processed": result.get("processed", 0),
            "failed": result.get("failed", 0),
            "total": result.get("total", 0),
            "collection": collection,
            "message": message,
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"Failed to process pending files: {e}")
        return {
            "success": False,
            "error": str(e),
            "processed": 0,
            "failed": 0,
            "total": 0,
            "message": "Failed to process pending files",
        }


async def cleanup_old_records(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    days: int = 30,
) -> Dict[str, Any]:
    """
    Clean up old processing records.

    Args:
        days: Age threshold in days for record cleanup (default: 30)

    Returns:
        Dict with cleanup results and counts.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Perform cleanup
        result = await ingestion_manager.cleanup_old_records(days=days)

        if result.get("success"):
            total_cleaned = result.get("total_cleaned", 0)
            message = f"Cleaned up {total_cleaned} old records older than {days} days"
            if result.get("vacuum_performed"):
                message += " and performed database vacuum"
        else:
            message = f"Failed to cleanup records: {result.get('error')}"

        return {
            "success": result.get("success", False),
            "cleanup_counts": result.get("cleanup_counts", {}),
            "total_cleaned": result.get("total_cleaned", 0),
            "vacuum_performed": result.get("vacuum_performed", False),
            "days": days,
            "message": message,
            "error": result.get("error"),
        }

    except Exception as e:
        logger.error(f"Failed to cleanup old records: {e}")
        return {
            "success": False,
            "error": str(e),
            "total_cleaned": 0,
            "message": "Failed to cleanup old records",
        }


async def get_database_stats(
    workspace_client: QdrantWorkspaceClient, watch_manager: WatchToolsManager
) -> Dict[str, Any]:
    """
    Get database size and health statistics.

    Returns:
        Dict with database statistics and health information.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Get database stats
        db_stats = await ingestion_manager.state_manager.get_database_stats()

        return {
            "success": True,
            "database_stats": db_stats,
            "message": f"Database size: {db_stats.get('total_size_mb', 0):.2f} MB",
        }

    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve database statistics",
        }


async def get_failed_files(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    collection: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Get list of failed files with error details.

    Args:
        collection: Optional collection filter
        limit: Maximum number of failed files to return (default: 100)

    Returns:
        Dict with list of failed files and error details.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Get failed files
        failed_files = await ingestion_manager.state_manager.get_failed_files(
            collection=collection, limit=limit
        )

        return {
            "success": True,
            "failed_files": failed_files,
            "count": len(failed_files),
            "collection": collection,
            "limit": limit,
            "message": f"Found {len(failed_files)} failed files",
        }

    except Exception as e:
        logger.error(f"Failed to get failed files: {e}")
        return {
            "success": False,
            "error": str(e),
            "failed_files": [],
            "count": 0,
            "message": "Failed to retrieve failed files",
        }


async def get_processing_analytics(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    collection: Optional[str] = None,
    days: int = 7,
) -> Dict[str, Any]:
    """
    Get processing analytics and statistics.

    Args:
        collection: Optional collection filter
        days: Number of days to include in analytics (default: 7)

    Returns:
        Dict with processing analytics and performance metrics.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Get processing stats
        analytics = await ingestion_manager.state_manager.get_processing_stats(
            collection=collection, days=days
        )

        return {
            "success": True,
            "analytics": analytics,
            "collection": collection,
            "period_days": days,
            "message": (
                f"Analytics for last {days} days: "
                f"{analytics.get('successful', 0)} successful, "
                f"{analytics.get('failed', 0)} failed files"
            ),
        }

    except Exception as e:
        logger.error(f"Failed to get processing analytics: {e}")
        return {
            "success": False,
            "error": str(e),
            "analytics": {},
            "message": "Failed to retrieve processing analytics",
        }


async def add_files_to_queue(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    file_paths: List[str],
    collection: str,
    priority: str = "normal",
) -> Dict[str, Any]:
    """
    Add files to processing queue with specified priority.

    Args:
        file_paths: List of file paths to add to queue
        collection: Target collection for ingestion
        priority: Processing priority (low, normal, high, urgent)

    Returns:
        Dict with queue addition results.
    """
    try:
        # Parse priority
        priority_map = {
            "low": ProcessingPriority.LOW,
            "normal": ProcessingPriority.NORMAL,
            "high": ProcessingPriority.HIGH,
            "urgent": ProcessingPriority.URGENT,
        }

        priority_enum = priority_map.get(priority.lower(), ProcessingPriority.NORMAL)

        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Add files to queue
        queue_ids = await ingestion_manager.add_files_to_queue(
            file_paths=file_paths, collection=collection, priority=priority_enum
        )

        return {
            "success": True,
            "queue_ids": queue_ids,
            "files_queued": len(queue_ids),
            "collection": collection,
            "priority": priority,
            "message": f"Added {len(queue_ids)} files to processing queue with {priority} priority",
        }

    except Exception as e:
        logger.error(f"Failed to add files to queue: {e}")
        return {
            "success": False,
            "error": str(e),
            "queue_ids": [],
            "files_queued": 0,
            "message": "Failed to add files to processing queue",
        }


async def clear_processing_queue(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Clear processing queue items.

    Args:
        collection: Optional collection filter (clears all if None)

    Returns:
        Dict with clear operation results.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Clear queue
        cleared_count = await ingestion_manager.state_manager.clear_queue(
            collection=collection
        )

        collection_msg = f" for collection '{collection}'" if collection else ""

        return {
            "success": True,
            "cleared_count": cleared_count,
            "collection": collection,
            "message": f"Cleared {cleared_count} items from processing queue{collection_msg}",
        }

    except Exception as e:
        logger.error(f"Failed to clear processing queue: {e}")
        return {
            "success": False,
            "error": str(e),
            "cleared_count": 0,
            "message": "Failed to clear processing queue",
        }


async def get_watch_folder_configs(
    workspace_client: QdrantWorkspaceClient,
    watch_manager: WatchToolsManager,
    enabled_only: bool = True,
) -> Dict[str, Any]:
    """
    Get persistent watch folder configurations.

    Args:
        enabled_only: Return only enabled watch folders (default: True)

    Returns:
        Dict with watch folder configurations.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Get watch folder configs
        configs = await ingestion_manager.state_manager.get_all_watch_folder_configs(
            enabled_only=enabled_only
        )

        # Convert to serializable format
        config_data = []
        for config in configs:
            config_data.append(
                {
                    "watch_id": config.watch_id,
                    "path": config.path,
                    "collection": config.collection,
                    "patterns": config.patterns,
                    "ignore_patterns": config.ignore_patterns,
                    "auto_ingest": config.auto_ingest,
                    "recursive": config.recursive,
                    "recursive_depth": config.recursive_depth,
                    "debounce_seconds": config.debounce_seconds,
                    "enabled": config.enabled,
                    "created_at": config.created_at.isoformat(),
                    "updated_at": config.updated_at.isoformat(),
                    "last_scan": config.last_scan.isoformat()
                    if config.last_scan
                    else None,
                    "metadata": config.metadata,
                }
            )

        return {
            "success": True,
            "watch_configs": config_data,
            "count": len(config_data),
            "enabled_only": enabled_only,
            "message": f"Retrieved {len(config_data)} watch folder configurations",
        }

    except Exception as e:
        logger.error(f"Failed to get watch folder configs: {e}")
        return {
            "success": False,
            "error": str(e),
            "watch_configs": [],
            "count": 0,
            "message": "Failed to retrieve watch folder configurations",
        }


async def vacuum_state_database(
    workspace_client: QdrantWorkspaceClient, watch_manager: WatchToolsManager
) -> Dict[str, Any]:
    """
    Perform database vacuum to reclaim space and optimize performance.

    Returns:
        Dict with vacuum operation results.
    """
    try:
        # Get ingestion manager
        ingestion_manager = await get_ingestion_manager(workspace_client, watch_manager)

        # Get database stats before vacuum
        stats_before = await ingestion_manager.state_manager.get_database_stats()
        size_before_mb = stats_before.get("total_size_mb", 0)

        # Perform vacuum
        success = await ingestion_manager.state_manager.vacuum_database()

        if success:
            # Get stats after vacuum
            stats_after = await ingestion_manager.state_manager.get_database_stats()
            size_after_mb = stats_after.get("total_size_mb", 0)
            space_saved_mb = size_before_mb - size_after_mb

            return {
                "success": True,
                "size_before_mb": size_before_mb,
                "size_after_mb": size_after_mb,
                "space_saved_mb": max(0, space_saved_mb),
                "message": (
                    f"Database vacuum completed. "
                    f"Size reduced from {size_before_mb:.2f}MB to {size_after_mb:.2f}MB "
                    f"(saved {max(0, space_saved_mb):.2f}MB)"
                ),
            }
        else:
            return {
                "success": False,
                "error": "Database vacuum operation failed",
                "message": "Failed to perform database vacuum",
            }

    except Exception as e:
        logger.error(f"Failed to vacuum database: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to perform database vacuum operation",
        }
