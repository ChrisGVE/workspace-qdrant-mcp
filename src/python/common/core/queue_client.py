"""
SQLite Queue Client for MCP Server

Provides high-level queue operations for the ingestion queue system,
with support for priority-based processing, error handling, and retry logic.

Features:
    - Enqueue files with automatic priority calculation
    - Dequeue batches with priority ordering
    - Update priority for existing items
    - Mark items as complete or error
    - Retry logic with exponential backoff
    - Collection metadata management
    - Queue statistics and monitoring

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_client import SQLiteQueueClient

    # Initialize client
    client = SQLiteQueueClient()
    await client.initialize()

    # Enqueue a file
    queue_id = await client.enqueue_file(
        file_path="/path/to/file.py",
        collection="my-project",
        priority=7
    )

    # Dequeue batch for processing
    items = await client.dequeue_batch(batch_size=10)

    # Mark as complete
    await client.mark_complete(queue_id)
    ```
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .queue_connection import QueueConnectionPool, ConnectionConfig
from .collection_types import CollectionTypeClassifier


class QueueOperation(Enum):
    """Queue operation types."""
    INGEST = "ingest"
    UPDATE = "update"
    DELETE = "delete"


class QueueItem:
    """Represents an item in the ingestion queue."""

    def __init__(
        self,
        file_absolute_path: str,
        collection_name: str,
        tenant_id: str = "default",
        branch: str = "main",
        operation: QueueOperation = QueueOperation.INGEST,
        priority: int = 5,
        queued_timestamp: Optional[datetime] = None,
        retry_count: int = 0,
        retry_from: Optional[str] = None,
        error_message_id: Optional[int] = None,
        collection_type: Optional[str] = None,
    ):
        self.file_absolute_path = file_absolute_path
        self.collection_name = collection_name
        self.tenant_id = tenant_id
        self.branch = branch
        self.operation = operation if isinstance(operation, QueueOperation) else QueueOperation(operation)
        self.priority = priority
        self.queued_timestamp = queued_timestamp or datetime.now(timezone.utc)
        self.retry_count = retry_count
        self.retry_from = retry_from
        self.error_message_id = error_message_id
        self.collection_type = collection_type

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "QueueItem":
        """Create QueueItem from database row."""
        return cls(
            file_absolute_path=row["file_absolute_path"],
            collection_name=row["collection_name"],
            tenant_id=row["tenant_id"],
            branch=row["branch"],
            operation=QueueOperation(row["operation"]),
            priority=row["priority"],
            queued_timestamp=datetime.fromisoformat(row["queued_timestamp"]),
            retry_count=row["retry_count"],
            retry_from=row["retry_from"],
            error_message_id=row["error_message_id"],
            collection_type=row.get("collection_type"),  # May be None for legacy items
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_absolute_path": self.file_absolute_path,
            "collection_name": self.collection_name,
            "tenant_id": self.tenant_id,
            "branch": self.branch,
            "operation": self.operation.value,
            "priority": self.priority,
            "queued_timestamp": self.queued_timestamp.isoformat(),
            "retry_count": self.retry_count,
            "retry_from": self.retry_from,
            "error_message_id": self.error_message_id,
            "collection_type": self.collection_type,
        }


class SQLiteQueueClient:
    """
    High-level client for SQLite queue operations.

    Provides methods for enqueueing files, dequeuing batches, updating priorities,
    and managing queue state with proper error handling and retry logic.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        connection_config: Optional[ConnectionConfig] = None
    ):
        """
        Initialize queue client.

        Args:
            db_path: Optional custom database path
            connection_config: Optional connection configuration
        """
        self.connection_pool = QueueConnectionPool(
            db_path=db_path or self._get_default_db_path(),
            config=connection_config or ConnectionConfig()
        )
        self._initialized = False

    def _get_default_db_path(self) -> str:
        """Get default database path from OS directories."""
        from ..utils.os_directories import OSDirectories
        os_dirs = OSDirectories()
        os_dirs.ensure_directories()
        return str(os_dirs.get_state_file("workspace_state.db"))

    async def initialize(self):
        """Initialize the queue client."""
        if self._initialized:
            return

        await self.connection_pool.initialize()
        self._initialized = True
        logger.info("SQLite queue client initialized")

    async def close(self):
        """Close the queue client."""
        if not self._initialized:
            return

        await self.connection_pool.close()
        self._initialized = False
        logger.info("SQLite queue client closed")

    async def enqueue_file(
        self,
        file_path: str,
        collection: str,
        tenant_id: str = "default",
        branch: str = "main",
        operation: QueueOperation = QueueOperation.INGEST,
        priority: int = 5,
        retry_from: Optional[str] = None
    ) -> str:
        """
        Enqueue a file for processing.

        Args:
            file_path: Absolute path to file
            collection: Target collection name
            tenant_id: Tenant identifier
            branch: Branch identifier
            operation: Operation type (ingest/update/delete)
            priority: Priority level (0-10)
            retry_from: Original file path if this is a retry

        Returns:
            File path (serves as queue ID)

        Raises:
            ValueError: If priority is out of range
            sqlite3.IntegrityError: If file already in queue
        """
        if not 0 <= priority <= 10:
            raise ValueError(f"Priority must be between 0 and 10, got {priority}")

        # Normalize file path
        file_absolute_path = str(Path(file_path).resolve())

        # Detect collection type
        classifier = CollectionTypeClassifier()
        collection_type_enum = classifier.classify_collection_type(collection)
        collection_type = collection_type_enum.value  # Convert enum to string

        query = """
            INSERT INTO ingestion_queue (
                file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, retry_from, collection_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        async with self.connection_pool.get_connection_async() as conn:
            try:
                conn.execute(query, (
                    file_absolute_path,
                    collection,
                    tenant_id,
                    branch,
                    operation.value,
                    priority,
                    retry_from,
                    collection_type
                ))
                conn.commit()

                logger.debug(
                    f"Enqueued file: {file_absolute_path} "
                    f"(collection={collection}, priority={priority}, type={collection_type})"
                )

                return file_absolute_path

            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint" in str(e) or "PRIMARY KEY" in str(e):
                    logger.warning(f"File already in queue: {file_absolute_path}")
                    raise
                else:
                    raise

    async def dequeue_batch(
        self,
        batch_size: int = 10,
        tenant_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> List[QueueItem]:
        """
        Dequeue a batch of items for processing.

        Items are selected based on priority (DESC) and queued timestamp (ASC).

        Args:
            batch_size: Maximum number of items to dequeue
            tenant_id: Optional tenant filter
            branch: Optional branch filter

        Returns:
            List of queue items ordered by priority
        """
        # Build query with optional filters
        query = """
            SELECT
                file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, queued_timestamp, retry_count,
                retry_from, error_message_id, collection_type
            FROM ingestion_queue
        """

        conditions = []
        params = []

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if branch:
            conditions.append("branch = ?")
            params.append(branch)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY priority DESC, queued_timestamp ASC LIMIT ?"
        params.append(batch_size)

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()

            items = [QueueItem.from_db_row(row) for row in rows]

            logger.debug(f"Dequeued {len(items)} items from queue")
            return items

    async def update_priority(
        self,
        file_path: str,
        new_priority: int
    ) -> bool:
        """
        Update priority for a queued file.

        Args:
            file_path: File path (queue ID)
            new_priority: New priority level (0-10)

        Returns:
            True if updated, False if not found

        Raises:
            ValueError: If priority is out of range
        """
        if not 0 <= new_priority <= 10:
            raise ValueError(f"Priority must be between 0 and 10, got {new_priority}")

        file_absolute_path = str(Path(file_path).resolve())

        query = """
            UPDATE ingestion_queue
            SET priority = ?
            WHERE file_absolute_path = ?
        """

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, (new_priority, file_absolute_path))
            conn.commit()

            updated = cursor.rowcount > 0

            if updated:
                logger.debug(f"Updated priority for {file_absolute_path}: {new_priority}")
            else:
                logger.warning(f"File not found in queue: {file_absolute_path}")

            return updated

    async def mark_complete(
        self,
        file_path: str
    ) -> bool:
        """
        Mark a file as completed and remove from queue.

        Args:
            file_path: File path (queue ID)

        Returns:
            True if removed, False if not found
        """
        file_absolute_path = str(Path(file_path).resolve())

        query = "DELETE FROM ingestion_queue WHERE file_absolute_path = ?"

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, (file_absolute_path,))
            conn.commit()

            deleted = cursor.rowcount > 0

            if deleted:
                logger.debug(f"Marked complete and removed from queue: {file_absolute_path}")
            else:
                logger.warning(f"File not found in queue: {file_absolute_path}")

            return deleted

    async def mark_error(
        self,
        file_path: str,
        error_type: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        max_retries: int = 3
    ) -> Tuple[bool, Optional[int]]:
        """
        Mark a file as having an error and update retry count.

        If retry count exceeds max_retries, removes from queue.
        Otherwise, increments retry_count and links error message.

        Args:
            file_path: File path (queue ID)
            error_type: Error type classification
            error_message: Error message
            error_details: Optional error details (JSON)
            max_retries: Maximum retry attempts

        Returns:
            Tuple of (should_retry, error_message_id)
        """
        file_absolute_path = str(Path(file_path).resolve())

        async with self.connection_pool.get_connection_async() as conn:
            # First, insert error message
            error_query = """
                INSERT INTO messages (
                    error_type, error_message, error_details,
                    file_path, collection_name
                ) VALUES (?, ?, ?, ?, (
                    SELECT collection_name FROM ingestion_queue
                    WHERE file_absolute_path = ?
                ))
            """

            error_details_json = json.dumps(error_details) if error_details else None

            cursor = conn.execute(error_query, (
                error_type,
                error_message,
                error_details_json,
                file_absolute_path,
                file_absolute_path
            ))

            error_message_id = cursor.lastrowid

            # Get current retry count
            cursor = conn.execute(
                "SELECT retry_count FROM ingestion_queue WHERE file_absolute_path = ?",
                (file_absolute_path,)
            )
            row = cursor.fetchone()

            if not row:
                logger.warning(f"File not found in queue: {file_absolute_path}")
                conn.commit()
                return False, error_message_id

            current_retry_count = row[0]
            new_retry_count = current_retry_count + 1

            if new_retry_count >= max_retries:
                # Max retries reached, remove from queue
                conn.execute(
                    "DELETE FROM ingestion_queue WHERE file_absolute_path = ?",
                    (file_absolute_path,)
                )
                logger.warning(
                    f"Max retries ({max_retries}) reached for {file_absolute_path}, "
                    f"removing from queue"
                )
                conn.commit()
                return False, error_message_id

            else:
                # Update retry count and link error
                update_query = """
                    UPDATE ingestion_queue
                    SET retry_count = ?, error_message_id = ?
                    WHERE file_absolute_path = ?
                """

                conn.execute(update_query, (
                    new_retry_count,
                    error_message_id,
                    file_absolute_path
                ))

                logger.debug(
                    f"Updated error for {file_absolute_path}: "
                    f"retry {new_retry_count}/{max_retries}"
                )

                conn.commit()
                return True, error_message_id

    async def get_queue_stats(
        self,
        tenant_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get queue statistics.

        Args:
            tenant_id: Optional tenant filter
            branch: Optional branch filter

        Returns:
            Dictionary with queue statistics
        """
        query = """
            SELECT
                COUNT(*) as total_items,
                SUM(CASE WHEN priority >= 8 THEN 1 ELSE 0 END) as urgent_items,
                SUM(CASE WHEN priority >= 5 AND priority < 8 THEN 1 ELSE 0 END) as high_items,
                SUM(CASE WHEN priority >= 3 AND priority < 5 THEN 1 ELSE 0 END) as normal_items,
                SUM(CASE WHEN priority < 3 THEN 1 ELSE 0 END) as low_items,
                SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) as retry_items,
                SUM(CASE WHEN error_message_id IS NOT NULL THEN 1 ELSE 0 END) as error_items,
                COUNT(DISTINCT collection_name) as unique_collections,
                MIN(queued_timestamp) as oldest_item,
                MAX(queued_timestamp) as newest_item
            FROM ingestion_queue
        """

        conditions = []
        params = []

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if branch:
            conditions.append("branch = ?")
            params.append(branch)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            row = cursor.fetchone()

            return {
                "total_items": row["total_items"],
                "urgent_items": row["urgent_items"],
                "high_items": row["high_items"],
                "normal_items": row["normal_items"],
                "low_items": row["low_items"],
                "retry_items": row["retry_items"],
                "error_items": row["error_items"],
                "unique_collections": row["unique_collections"],
                "oldest_item": row["oldest_item"],
                "newest_item": row["newest_item"],
            }

    async def clear_queue(
        self,
        collection: Optional[str] = None,
        tenant_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> int:
        """
        Clear items from the queue.

        Args:
            collection: Optional collection filter
            tenant_id: Optional tenant filter
            branch: Optional branch filter

        Returns:
            Number of items removed
        """
        query = "DELETE FROM ingestion_queue"
        conditions = []
        params = []

        if collection:
            conditions.append("collection_name = ?")
            params.append(collection)

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if branch:
            conditions.append("branch = ?")
            params.append(branch)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            conn.commit()

            deleted_count = cursor.rowcount
            logger.info(f"Cleared {deleted_count} items from queue")

            return deleted_count

    async def register_collection(
        self,
        collection_name: str,
        collection_type: str,
        configuration: Optional[Dict[str, Any]] = None,
        tenant_id: str = "default",
        branch: str = "main"
    ) -> bool:
        """
        Register or update collection metadata.

        Args:
            collection_name: Collection name
            collection_type: Type (non-watched/watched-dynamic/watched-cumulative/project)
            configuration: Optional JSON configuration
            tenant_id: Tenant identifier
            branch: Branch identifier

        Returns:
            True if registered/updated

        Raises:
            ValueError: If collection_type is invalid
        """
        valid_types = ["non-watched", "watched-dynamic", "watched-cumulative", "project"]
        if collection_type not in valid_types:
            raise ValueError(f"Invalid collection_type: {collection_type}. Must be one of {valid_types}")

        config_json = json.dumps(configuration) if configuration else "{}"

        query = """
            INSERT INTO collection_metadata (
                collection_name, collection_type, configuration, tenant_id, branch
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(collection_name) DO UPDATE SET
                collection_type = excluded.collection_type,
                configuration = excluded.configuration,
                last_updated = CURRENT_TIMESTAMP
        """

        async with self.connection_pool.get_connection_async() as conn:
            conn.execute(query, (
                collection_name,
                collection_type,
                config_json,
                tenant_id,
                branch
            ))
            conn.commit()

            logger.debug(f"Registered collection: {collection_name} (type={collection_type})")
            return True

    async def get_collection_info(
        self,
        collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get collection metadata.

        Args:
            collection_name: Collection name

        Returns:
            Collection metadata dict or None if not found
        """
        query = """
            SELECT
                collection_name, collection_type, created_timestamp,
                last_updated, configuration, tenant_id, branch
            FROM collection_metadata
            WHERE collection_name = ?
        """

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, (collection_name,))
            row = cursor.fetchone()

            if not row:
                return None

            config = json.loads(row["configuration"]) if row["configuration"] else {}

            return {
                "collection_name": row["collection_name"],
                "collection_type": row["collection_type"],
                "created_timestamp": row["created_timestamp"],
                "last_updated": row["last_updated"],
                "configuration": config,
                "tenant_id": row["tenant_id"],
                "branch": row["branch"],
            }

    async def enqueue_batch(
        self,
        items: List[Dict[str, Any]],
        max_queue_depth: Optional[int] = None,
        overflow_strategy: str = "reject"
    ) -> Tuple[int, List[str]]:
        """
        Enqueue multiple files as a batch with priority calculation.

        Args:
            items: List of dictionaries with keys:
                - file_path: str (required)
                - collection: str (required)
                - priority: int (required, 0-10)
                - operation: QueueOperation (optional, defaults to INGEST)
                - tenant_id: str (optional, defaults to "default")
                - branch: str (optional, defaults to "main")
            max_queue_depth: Maximum queue size (None = unlimited)
            overflow_strategy: Strategy when queue full ("reject" or "replace_lowest")

        Returns:
            Tuple of (successful_count, failed_paths)

        Raises:
            ValueError: If queue depth exceeded and overflow_strategy="reject"
        """
        if not items:
            return 0, []

        # Validate all items first
        for item in items:
            if "file_path" not in item or "collection" not in item or "priority" not in item:
                raise ValueError("Each item must have file_path, collection, and priority")
            if not (0 <= item["priority"] <= 10):
                raise ValueError(f"Priority must be 0-10, got {item['priority']}")

        async with self.connection_pool.get_connection_async() as conn:
            # Check queue depth if limit specified
            if max_queue_depth is not None:
                cursor = conn.execute("SELECT COUNT(*) FROM ingestion_queue")
                current_depth = cursor.fetchone()[0]

                if current_depth + len(items) > max_queue_depth:
                    if overflow_strategy == "reject":
                        raise ValueError(
                            f"Queue depth limit ({max_queue_depth}) would be exceeded. "
                            f"Current: {current_depth}, Adding: {len(items)}"
                        )
                    elif overflow_strategy == "replace_lowest":
                        # Remove lowest priority items to make space
                        items_to_remove = (current_depth + len(items)) - max_queue_depth
                        conn.execute(f"""
                            DELETE FROM ingestion_queue
                            WHERE file_absolute_path IN (
                                SELECT file_absolute_path
                                FROM ingestion_queue
                                ORDER BY priority ASC, queued_timestamp DESC
                                LIMIT {items_to_remove}
                            )
                        """)
                        logger.info(
                            f"Removed {items_to_remove} lowest priority items due to "
                            f"queue depth limit"
                        )

            # Batch insert
            successful = 0
            failed = []

            insert_query = """
                INSERT OR REPLACE INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id, branch,
                    operation, priority, queued_timestamp, collection_type
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            """

            classifier = CollectionTypeClassifier()

            for item in items:
                try:
                    operation = item.get("operation", QueueOperation.INGEST).value \
                        if isinstance(item.get("operation"), QueueOperation) \
                        else item.get("operation", "ingest")

                    # Detect collection type
                    collection_type_enum = classifier.classify_collection_type(item["collection"])
                    collection_type = collection_type_enum.value

                    conn.execute(insert_query, (
                        item["file_path"],
                        item["collection"],
                        item.get("tenant_id", "default"),
                        item.get("branch", "main"),
                        operation,
                        item["priority"],
                        collection_type
                    ))
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to enqueue {item['file_path']}: {e}")
                    failed.append(item["file_path"])

            conn.commit()

            logger.info(
                f"Batch enqueue completed: {successful} successful, {len(failed)} failed"
            )

            return successful, failed

    async def purge_completed_items(
        self,
        retention_hours: int = 24,
        tenant_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> int:
        """
        Purge completed items based on retention policy.

        Removes entries from messages table older than retention period.
        Note: ingestion_queue items are removed immediately on completion,
        this purges historical error/completion logs.

        Args:
            retention_hours: Keep messages newer than this many hours
            tenant_id: Optional tenant filter
            branch: Optional branch filter

        Returns:
            Number of messages purged
        """
        query = """
            DELETE FROM messages
            WHERE created_timestamp < datetime('now', ? || ' hours')
        """

        params = [f"-{retention_hours}"]

        # Add filters if specified
        # Note: messages table doesn't have tenant_id/branch,
        # we filter by file_path pattern if needed
        if tenant_id or branch:
            logger.warning(
                "tenant_id/branch filtering not directly supported for message purging"
            )

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            conn.commit()

            purged_count = cursor.rowcount
            logger.info(
                f"Purged {purged_count} messages older than {retention_hours} hours"
            )

            return purged_count

    async def get_queue_depth(
        self,
        tenant_id: Optional[str] = None,
        branch: Optional[str] = None
    ) -> int:
        """
        Get current queue depth (item count).

        Args:
            tenant_id: Optional tenant filter
            branch: Optional branch filter

        Returns:
            Number of items in queue
        """
        query = "SELECT COUNT(*) FROM ingestion_queue"
        conditions = []
        params = []

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if branch:
            conditions.append("branch = ?")
            params.append(branch)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            return cursor.fetchone()[0]
