"""
Real-Time Queue Statistics Collection

Provides real-time metrics collection and monitoring for the queue system,
tracking processing rates, success/failure rates, queue sizes, and performance metrics.

Features:
    - Real-time queue size monitoring
    - Processing rate calculation with time windows (1min, 5min, 15min)
    - Success/failure rate tracking
    - Average processing time calculation
    - Per-queue, per-collection, per-tenant statistics
    - Background statistics collection task
    - Memory-efficient sliding window implementation

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_statistics import QueueStatisticsCollector

    # Initialize collector
    collector = QueueStatisticsCollector()
    await collector.initialize()

    # Start background collection
    await collector.start_collection(interval_seconds=5)

    # Get current statistics
    stats = await collector.get_current_statistics()
    print(f"Queue size: {stats.queue_size}")
    print(f"Processing rate: {stats.processing_rate} items/min")

    # Get per-collection statistics
    collection_stats = await collector.get_statistics_by_collection("my-project")

    # Stop collection
    await collector.stop_collection()
    ```
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from .queue_connection import ConnectionConfig, QueueConnectionPool


@dataclass
class QueueStatistics:
    """
    Real-time queue statistics snapshot.

    Attributes:
        queue_size: Current number of items in queue
        processing_rate: Items processed per minute (averaged over window)
        success_rate: Percentage of successful processing (0-100)
        failure_rate: Percentage of failed processing (0-100)
        avg_processing_time: Average processing time in milliseconds
        items_added_rate: Items added to queue per minute
        items_removed_rate: Items removed from queue per minute
        priority_distribution: Count by priority level
        retry_count: Number of items with retry_count > 0
        error_count: Number of items with errors
        timestamp: When statistics were collected
    """
    queue_size: int = 0
    processing_rate: float = 0.0  # items/min
    success_rate: float = 0.0  # percentage
    failure_rate: float = 0.0  # percentage
    avg_processing_time: float = 0.0  # milliseconds
    items_added_rate: float = 0.0  # items/min
    items_removed_rate: float = 0.0  # items/min
    priority_distribution: dict[str, int] = field(default_factory=dict)
    retry_count: int = 0
    error_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with all statistics fields
        """
        return {
            "queue_size": self.queue_size,
            "processing_rate": round(self.processing_rate, 2),
            "success_rate": round(self.success_rate, 2),
            "failure_rate": round(self.failure_rate, 2),
            "avg_processing_time": round(self.avg_processing_time, 2),
            "items_added_rate": round(self.items_added_rate, 2),
            "items_removed_rate": round(self.items_removed_rate, 2),
            "priority_distribution": self.priority_distribution,
            "retry_count": self.retry_count,
            "error_count": self.error_count,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ProcessingEvent:
    """
    Represents a single processing event for rate calculation.

    Attributes:
        timestamp: When the event occurred
        event_type: Type of event ('added', 'removed', 'success', 'failure')
        processing_time: Processing time in milliseconds (for completion events)
        queue_type: Queue type (ingestion, retry, missing_metadata, error)
        collection: Optional collection name
        tenant_id: Optional tenant identifier
    """
    timestamp: float
    event_type: str
    processing_time: float | None = None
    queue_type: str = "ingestion_queue"
    collection: str | None = None
    tenant_id: str | None = None


class QueueStatisticsCollector:
    """
    Real-time queue statistics collector with sliding window metrics.

    Collects and calculates queue statistics using time-based sliding windows
    for rate calculations and memory-efficient event tracking.
    """

    def __init__(
        self,
        db_path: str | None = None,
        connection_config: ConnectionConfig | None = None,
        window_1min: int = 60,
        window_5min: int = 300,
        window_15min: int = 900,
        max_events: int = 10000
    ):
        """
        Initialize queue statistics collector.

        Args:
            db_path: Optional custom database path
            connection_config: Optional connection configuration
            window_1min: 1-minute window in seconds
            window_5min: 5-minute window in seconds
            window_15min: 15-minute window in seconds
            max_events: Maximum events to keep in memory (memory limit)
        """
        self.connection_pool = QueueConnectionPool(
            db_path=db_path or self._get_default_db_path(),
            config=connection_config or ConnectionConfig()
        )
        self._initialized = False

        # Time windows for rate calculation
        self.window_1min = window_1min
        self.window_5min = window_5min
        self.window_15min = window_15min

        # Event tracking with sliding windows
        # Using deque for O(1) append and popleft
        self._events: deque[ProcessingEvent] = deque(maxlen=max_events)
        self._lock = asyncio.Lock()

        # Background collection task
        self._collection_task: asyncio.Task | None = None
        self._collection_interval: int = 5  # seconds
        self._shutdown_event = asyncio.Event()

        # Latest snapshot cache
        self._latest_stats: QueueStatistics | None = None

    def _get_default_db_path(self) -> str:
        """Get default database path from OS directories."""
        from ..utils.os_directories import OSDirectories
        os_dirs = OSDirectories()
        os_dirs.ensure_directories()
        return str(os_dirs.get_state_file("workspace_state.db"))

    async def initialize(self):
        """Initialize the statistics collector."""
        if self._initialized:
            return

        await self.connection_pool.initialize()

        self._initialized = True
        logger.info("Queue statistics collector initialized")

    async def close(self):
        """Close the statistics collector."""
        if not self._initialized:
            return

        # Stop collection if running
        await self.stop_collection()

        await self.connection_pool.close()
        self._initialized = False
        logger.info("Queue statistics collector closed")

    async def record_event(
        self,
        event_type: str,
        processing_time: float | None = None,
        queue_type: str = "ingestion_queue",
        collection: str | None = None,
        tenant_id: str | None = None
    ):
        """
        Record a processing event for statistics calculation.

        Args:
            event_type: Event type ('added', 'removed', 'success', 'failure')
            processing_time: Processing time in milliseconds (for completion events)
            queue_type: Queue type
            collection: Optional collection name
            tenant_id: Optional tenant identifier
        """
        event = ProcessingEvent(
            timestamp=time.time(),
            event_type=event_type,
            processing_time=processing_time,
            queue_type=queue_type,
            collection=collection,
            tenant_id=tenant_id
        )

        async with self._lock:
            self._events.append(event)

        logger.debug(f"Recorded event: {event_type} (queue={queue_type})")

    async def get_current_statistics(
        self,
        queue_type: str = "ingestion_queue"
    ) -> QueueStatistics:
        """
        Get current real-time statistics.

        Args:
            queue_type: Queue type to get statistics for

        Returns:
            Current queue statistics
        """
        # Get queue size from database
        queue_size = await self._get_queue_size(queue_type=queue_type)

        # Get priority distribution
        priority_dist = await self._get_priority_distribution(queue_type=queue_type)

        # Get retry and error counts
        retry_count, error_count = await self._get_retry_error_counts(queue_type=queue_type)

        # Calculate rates from events
        processing_rate, success_rate, failure_rate = await self._calculate_rates(
            window_seconds=self.window_1min,
            queue_type=queue_type
        )

        # Calculate average processing time
        avg_processing_time = await self._calculate_avg_processing_time(
            window_seconds=self.window_5min,
            queue_type=queue_type
        )

        # Calculate add/remove rates
        items_added_rate = await self._calculate_event_rate(
            event_type="added",
            window_seconds=self.window_1min,
            queue_type=queue_type
        )

        items_removed_rate = await self._calculate_event_rate(
            event_type="removed",
            window_seconds=self.window_1min,
            queue_type=queue_type
        )

        stats = QueueStatistics(
            queue_size=queue_size,
            processing_rate=processing_rate,
            success_rate=success_rate,
            failure_rate=failure_rate,
            avg_processing_time=avg_processing_time,
            items_added_rate=items_added_rate,
            items_removed_rate=items_removed_rate,
            priority_distribution=priority_dist,
            retry_count=retry_count,
            error_count=error_count,
            timestamp=datetime.now(timezone.utc)
        )

        # Cache latest stats
        self._latest_stats = stats

        return stats

    async def get_statistics_by_queue(
        self,
        queue_type: str
    ) -> QueueStatistics:
        """
        Get statistics for a specific queue type.

        Args:
            queue_type: Queue type (ingestion_queue, retry_queue, etc.)

        Returns:
            Queue statistics for the specified queue
        """
        return await self.get_current_statistics(queue_type=queue_type)

    async def get_statistics_by_collection(
        self,
        collection_name: str
    ) -> QueueStatistics:
        """
        Get statistics filtered by collection.

        Args:
            collection_name: Collection name to filter by

        Returns:
            Queue statistics for the specified collection
        """
        # Get queue size for collection
        queue_size = await self._get_queue_size(collection=collection_name)

        # Get priority distribution for collection
        priority_dist = await self._get_priority_distribution(collection=collection_name)

        # Get retry and error counts for collection
        retry_count, error_count = await self._get_retry_error_counts(collection=collection_name)

        # Calculate rates filtered by collection
        processing_rate, success_rate, failure_rate = await self._calculate_rates(
            window_seconds=self.window_1min,
            collection=collection_name
        )

        avg_processing_time = await self._calculate_avg_processing_time(
            window_seconds=self.window_5min,
            collection=collection_name
        )

        items_added_rate = await self._calculate_event_rate(
            event_type="added",
            window_seconds=self.window_1min,
            collection=collection_name
        )

        items_removed_rate = await self._calculate_event_rate(
            event_type="removed",
            window_seconds=self.window_1min,
            collection=collection_name
        )

        return QueueStatistics(
            queue_size=queue_size,
            processing_rate=processing_rate,
            success_rate=success_rate,
            failure_rate=failure_rate,
            avg_processing_time=avg_processing_time,
            items_added_rate=items_added_rate,
            items_removed_rate=items_removed_rate,
            priority_distribution=priority_dist,
            retry_count=retry_count,
            error_count=error_count,
            timestamp=datetime.now(timezone.utc)
        )

    async def get_statistics_by_tenant(
        self,
        tenant_id: str
    ) -> QueueStatistics:
        """
        Get statistics filtered by tenant.

        Args:
            tenant_id: Tenant identifier to filter by

        Returns:
            Queue statistics for the specified tenant
        """
        # Get queue size for tenant
        queue_size = await self._get_queue_size(tenant_id=tenant_id)

        # Get priority distribution for tenant
        priority_dist = await self._get_priority_distribution(tenant_id=tenant_id)

        # Get retry and error counts for tenant
        retry_count, error_count = await self._get_retry_error_counts(tenant_id=tenant_id)

        # Calculate rates filtered by tenant
        processing_rate, success_rate, failure_rate = await self._calculate_rates(
            window_seconds=self.window_1min,
            tenant_id=tenant_id
        )

        avg_processing_time = await self._calculate_avg_processing_time(
            window_seconds=self.window_5min,
            tenant_id=tenant_id
        )

        items_added_rate = await self._calculate_event_rate(
            event_type="added",
            window_seconds=self.window_1min,
            tenant_id=tenant_id
        )

        items_removed_rate = await self._calculate_event_rate(
            event_type="removed",
            window_seconds=self.window_1min,
            tenant_id=tenant_id
        )

        return QueueStatistics(
            queue_size=queue_size,
            processing_rate=processing_rate,
            success_rate=success_rate,
            failure_rate=failure_rate,
            avg_processing_time=avg_processing_time,
            items_added_rate=items_added_rate,
            items_removed_rate=items_removed_rate,
            priority_distribution=priority_dist,
            retry_count=retry_count,
            error_count=error_count,
            timestamp=datetime.now(timezone.utc)
        )

    async def start_collection(
        self,
        interval_seconds: int = 5
    ) -> bool:
        """
        Start background statistics collection.

        Args:
            interval_seconds: Collection interval in seconds

        Returns:
            True if started successfully, False if already running
        """
        if self._collection_task and not self._collection_task.done():
            logger.warning("Statistics collection already running")
            return False

        self._collection_interval = interval_seconds
        self._shutdown_event.clear()

        self._collection_task = asyncio.create_task(self._collection_loop())

        logger.info(f"Started statistics collection (interval={interval_seconds}s)")
        return True

    async def stop_collection(self) -> bool:
        """
        Stop background statistics collection.

        Returns:
            True if stopped successfully, False if not running
        """
        if not self._collection_task or self._collection_task.done():
            logger.warning("Statistics collection not running")
            return False

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for task to complete
        try:
            await asyncio.wait_for(self._collection_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Collection task did not stop gracefully, cancelling")
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped statistics collection")
        return True

    async def _collection_loop(self):
        """Background task for periodic statistics collection."""
        logger.info("Starting statistics collection loop")

        while not self._shutdown_event.is_set():
            try:
                # Collect current statistics
                await self.get_current_statistics()

                # Cleanup old events
                await self._cleanup_old_events()

                # Wait for next interval
                await asyncio.sleep(self._collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in statistics collection loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

        logger.info("Statistics collection loop stopped")

    async def _cleanup_old_events(self):
        """Remove events older than 15-minute window."""
        cutoff_time = time.time() - self.window_15min

        async with self._lock:
            # Remove old events from left side
            while self._events and self._events[0].timestamp < cutoff_time:
                self._events.popleft()

        logger.debug(f"Event cleanup complete, {len(self._events)} events remaining")

    async def _get_queue_size(
        self,
        queue_type: str = "ingestion_queue",
        collection: str | None = None,
        tenant_id: str | None = None
    ) -> int:
        """
        Get current queue size from database.

        Args:
            queue_type: Queue type to query
            collection: Optional collection filter
            tenant_id: Optional tenant filter

        Returns:
            Number of items in queue
        """
        query = f"SELECT COUNT(*) FROM {queue_type}"
        conditions = []
        params = []

        if collection:
            conditions.append("collection_name = ?")
            params.append(collection)

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            return cursor.fetchone()[0]

    async def _get_priority_distribution(
        self,
        queue_type: str = "ingestion_queue",
        collection: str | None = None,
        tenant_id: str | None = None
    ) -> dict[str, int]:
        """
        Get priority distribution from database.

        Args:
            queue_type: Queue type to query
            collection: Optional collection filter
            tenant_id: Optional tenant filter

        Returns:
            Dictionary with priority distribution
        """
        query = f"""
            SELECT
                CASE
                    WHEN priority >= 8 THEN 'urgent'
                    WHEN priority >= 5 THEN 'high'
                    WHEN priority >= 3 THEN 'normal'
                    ELSE 'low'
                END as priority_level,
                COUNT(*) as count
            FROM {queue_type}
        """

        conditions = []
        params = []

        if collection:
            conditions.append("collection_name = ?")
            params.append(collection)

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY priority_level"

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()

            return {row[0]: row[1] for row in rows}

    async def _get_retry_error_counts(
        self,
        queue_type: str = "ingestion_queue",
        collection: str | None = None,
        tenant_id: str | None = None
    ) -> tuple[int, int]:
        """
        Get retry and error counts from database.

        Args:
            queue_type: Queue type to query
            collection: Optional collection filter
            tenant_id: Optional tenant filter

        Returns:
            Tuple of (retry_count, error_count)
        """
        query = f"""
            SELECT
                SUM(CASE WHEN retry_count > 0 THEN 1 ELSE 0 END) as retry_items,
                SUM(CASE WHEN error_message_id IS NOT NULL THEN 1 ELSE 0 END) as error_items
            FROM {queue_type}
        """

        conditions = []
        params = []

        if collection:
            conditions.append("collection_name = ?")
            params.append(collection)

        if tenant_id:
            conditions.append("tenant_id = ?")
            params.append(tenant_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            row = cursor.fetchone()

            retry_count = row[0] or 0
            error_count = row[1] or 0

            return retry_count, error_count

    async def _calculate_rates(
        self,
        window_seconds: int,
        queue_type: str | None = None,
        collection: str | None = None,
        tenant_id: str | None = None
    ) -> tuple[float, float, float]:
        """
        Calculate processing, success, and failure rates.

        Args:
            window_seconds: Time window in seconds
            queue_type: Optional queue type filter
            collection: Optional collection filter
            tenant_id: Optional tenant filter

        Returns:
            Tuple of (processing_rate, success_rate, failure_rate)
        """
        cutoff_time = time.time() - window_seconds

        async with self._lock:
            # Filter events by time window and filters
            filtered_events = [
                e for e in self._events
                if e.timestamp >= cutoff_time
                and (queue_type is None or e.queue_type == queue_type)
                and (collection is None or e.collection == collection)
                and (tenant_id is None or e.tenant_id == tenant_id)
            ]

        # Count events
        success_events = sum(1 for e in filtered_events if e.event_type == "success")
        failure_events = sum(1 for e in filtered_events if e.event_type == "failure")
        total_processed = success_events + failure_events

        # Calculate rates
        minutes = window_seconds / 60.0
        processing_rate = total_processed / minutes if minutes > 0 else 0.0

        success_rate = (success_events / total_processed * 100) if total_processed > 0 else 0.0
        failure_rate = (failure_events / total_processed * 100) if total_processed > 0 else 0.0

        return processing_rate, success_rate, failure_rate

    async def _calculate_avg_processing_time(
        self,
        window_seconds: int,
        queue_type: str | None = None,
        collection: str | None = None,
        tenant_id: str | None = None
    ) -> float:
        """
        Calculate average processing time in milliseconds.

        Args:
            window_seconds: Time window in seconds
            queue_type: Optional queue type filter
            collection: Optional collection filter
            tenant_id: Optional tenant filter

        Returns:
            Average processing time in milliseconds
        """
        cutoff_time = time.time() - window_seconds

        async with self._lock:
            # Filter events with processing time
            filtered_events = [
                e for e in self._events
                if e.timestamp >= cutoff_time
                and e.processing_time is not None
                and (queue_type is None or e.queue_type == queue_type)
                and (collection is None or e.collection == collection)
                and (tenant_id is None or e.tenant_id == tenant_id)
            ]

        if not filtered_events:
            return 0.0

        total_time = sum(e.processing_time for e in filtered_events)
        return total_time / len(filtered_events)

    async def _calculate_event_rate(
        self,
        event_type: str,
        window_seconds: int,
        queue_type: str | None = None,
        collection: str | None = None,
        tenant_id: str | None = None
    ) -> float:
        """
        Calculate rate for specific event type.

        Args:
            event_type: Event type to calculate rate for
            window_seconds: Time window in seconds
            queue_type: Optional queue type filter
            collection: Optional collection filter
            tenant_id: Optional tenant filter

        Returns:
            Event rate in events per minute
        """
        cutoff_time = time.time() - window_seconds

        async with self._lock:
            # Filter events
            filtered_events = [
                e for e in self._events
                if e.timestamp >= cutoff_time
                and e.event_type == event_type
                and (queue_type is None or e.queue_type == queue_type)
                and (collection is None or e.collection == collection)
                and (tenant_id is None or e.tenant_id == tenant_id)
            ]

        minutes = window_seconds / 60.0
        return len(filtered_events) / minutes if minutes > 0 else 0.0
