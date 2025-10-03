"""
Unit Tests for Queue Statistics Module

Tests real-time queue statistics collection, rate calculations,
and memory-efficient event tracking.
"""

import asyncio
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List
import tempfile

import pytest

from src.python.common.core.queue_statistics import (
    QueueStatistics,
    ProcessingEvent,
    QueueStatisticsCollector
)
from src.python.common.core.queue_connection import QueueConnectionPool, ConnectionConfig


@pytest.fixture
async def test_db():
    """Create temporary test database with schema."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
        db_path = f.name

    # Create schema
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE ingestion_queue (
            file_absolute_path TEXT PRIMARY KEY,
            collection_name TEXT NOT NULL,
            tenant_id TEXT NOT NULL DEFAULT 'default',
            branch TEXT NOT NULL DEFAULT 'main',
            operation TEXT NOT NULL DEFAULT 'ingest',
            priority INTEGER NOT NULL DEFAULT 5,
            queued_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            retry_count INTEGER NOT NULL DEFAULT 0,
            retry_from TEXT,
            error_message_id INTEGER,
            collection_type TEXT
        )
    """)
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
async def collector(test_db):
    """Create queue statistics collector with test database."""
    config = ConnectionConfig(
        busy_timeout=5.0,
        max_connections=5
    )
    collector = QueueStatisticsCollector(
        db_path=test_db,
        connection_config=config,
        window_1min=60,
        window_5min=300,
        window_15min=900,
        max_events=1000
    )
    await collector.initialize()
    yield collector
    await collector.close()


class TestQueueStatistics:
    """Test QueueStatistics dataclass."""

    def test_queue_statistics_creation(self):
        """Test creating QueueStatistics with default values."""
        stats = QueueStatistics()

        assert stats.queue_size == 0
        assert stats.processing_rate == 0.0
        assert stats.success_rate == 0.0
        assert stats.failure_rate == 0.0
        assert stats.avg_processing_time == 0.0
        assert stats.items_added_rate == 0.0
        assert stats.items_removed_rate == 0.0
        assert stats.priority_distribution == {}
        assert stats.retry_count == 0
        assert stats.error_count == 0
        assert isinstance(stats.timestamp, datetime)

    def test_queue_statistics_to_dict(self):
        """Test converting QueueStatistics to dictionary."""
        stats = QueueStatistics(
            queue_size=10,
            processing_rate=5.5,
            success_rate=95.0,
            failure_rate=5.0,
            avg_processing_time=123.45,
            items_added_rate=3.2,
            items_removed_rate=2.8,
            priority_distribution={"urgent": 2, "high": 5, "normal": 3},
            retry_count=1,
            error_count=0
        )

        result = stats.to_dict()

        assert result["queue_size"] == 10
        assert result["processing_rate"] == 5.5
        assert result["success_rate"] == 95.0
        assert result["failure_rate"] == 5.0
        assert result["avg_processing_time"] == 123.45
        assert result["items_added_rate"] == 3.2
        assert result["items_removed_rate"] == 2.8
        assert result["priority_distribution"] == {"urgent": 2, "high": 5, "normal": 3}
        assert result["retry_count"] == 1
        assert result["error_count"] == 0
        assert "timestamp" in result


class TestProcessingEvent:
    """Test ProcessingEvent dataclass."""

    def test_processing_event_creation(self):
        """Test creating ProcessingEvent."""
        event = ProcessingEvent(
            timestamp=time.time(),
            event_type="success",
            processing_time=100.0,
            queue_type="ingestion_queue",
            collection="test-collection",
            tenant_id="test-tenant"
        )

        assert event.event_type == "success"
        assert event.processing_time == 100.0
        assert event.queue_type == "ingestion_queue"
        assert event.collection == "test-collection"
        assert event.tenant_id == "test-tenant"


class TestQueueStatisticsCollector:
    """Test QueueStatisticsCollector."""

    @pytest.mark.asyncio
    async def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector._initialized
        assert collector.connection_pool is not None
        assert collector.window_1min == 60
        assert collector.window_5min == 300
        assert collector.window_15min == 900

    @pytest.mark.asyncio
    async def test_record_event(self, collector):
        """Test recording processing events."""
        await collector.record_event(
            event_type="added",
            queue_type="ingestion_queue",
            collection="test-collection"
        )

        await collector.record_event(
            event_type="success",
            processing_time=150.0,
            queue_type="ingestion_queue",
            collection="test-collection"
        )

        async with collector._lock:
            assert len(collector._events) == 2
            assert collector._events[0].event_type == "added"
            assert collector._events[1].event_type == "success"
            assert collector._events[1].processing_time == 150.0

    @pytest.mark.asyncio
    async def test_get_queue_size_empty(self, collector):
        """Test getting queue size when empty."""
        size = await collector._get_queue_size()
        assert size == 0

    @pytest.mark.asyncio
    async def test_get_queue_size_with_items(self, collector, test_db):
        """Test getting queue size with items."""
        # Insert test items
        conn = sqlite3.connect(test_db)
        for i in range(5):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id, priority
                ) VALUES (?, ?, ?, ?)
                """,
                (f"/test/file{i}.py", "test-collection", "default", 5)
            )
        conn.commit()
        conn.close()

        size = await collector._get_queue_size()
        assert size == 5

    @pytest.mark.asyncio
    async def test_get_queue_size_filtered_by_collection(self, collector, test_db):
        """Test getting queue size filtered by collection."""
        # Insert items with different collections
        conn = sqlite3.connect(test_db)
        for i in range(3):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id
                ) VALUES (?, ?, ?)
                """,
                (f"/test/file{i}.py", "collection-a", "default")
            )
        for i in range(2):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id
                ) VALUES (?, ?, ?)
                """,
                (f"/test/other{i}.py", "collection-b", "default")
            )
        conn.commit()
        conn.close()

        size_a = await collector._get_queue_size(collection="collection-a")
        size_b = await collector._get_queue_size(collection="collection-b")

        assert size_a == 3
        assert size_b == 2

    @pytest.mark.asyncio
    async def test_get_queue_size_filtered_by_tenant(self, collector, test_db):
        """Test getting queue size filtered by tenant."""
        # Insert items with different tenants
        conn = sqlite3.connect(test_db)
        for i in range(4):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id
                ) VALUES (?, ?, ?)
                """,
                (f"/test/file{i}.py", "test-collection", "tenant-1")
            )
        for i in range(2):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id
                ) VALUES (?, ?, ?)
                """,
                (f"/test/other{i}.py", "test-collection", "tenant-2")
            )
        conn.commit()
        conn.close()

        size_1 = await collector._get_queue_size(tenant_id="tenant-1")
        size_2 = await collector._get_queue_size(tenant_id="tenant-2")

        assert size_1 == 4
        assert size_2 == 2

    @pytest.mark.asyncio
    async def test_get_priority_distribution(self, collector, test_db):
        """Test getting priority distribution."""
        # Insert items with different priorities
        conn = sqlite3.connect(test_db)
        priorities = [9, 8, 7, 6, 5, 4, 3, 2, 1]  # Mix of all levels
        for i, priority in enumerate(priorities):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, priority
                ) VALUES (?, ?, ?)
                """,
                (f"/test/file{i}.py", "test-collection", priority)
            )
        conn.commit()
        conn.close()

        dist = await collector._get_priority_distribution()

        # urgent: 8-10 -> priorities 9, 8 = 2
        # high: 5-7 -> priorities 7, 6, 5 = 3
        # normal: 3-4 -> priorities 4, 3 = 2
        # low: 0-2 -> priorities 2, 1 = 2

        assert dist.get("urgent", 0) == 2
        assert dist.get("high", 0) == 3
        assert dist.get("normal", 0) == 2
        assert dist.get("low", 0) == 2

    @pytest.mark.asyncio
    async def test_get_retry_error_counts(self, collector, test_db):
        """Test getting retry and error counts."""
        # Insert items with retries and errors
        conn = sqlite3.connect(test_db)

        # 3 items with retries
        for i in range(3):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, retry_count
                ) VALUES (?, ?, ?)
                """,
                (f"/test/retry{i}.py", "test-collection", i + 1)
            )

        # 2 items with errors
        for i in range(2):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, error_message_id
                ) VALUES (?, ?, ?)
                """,
                (f"/test/error{i}.py", "test-collection", i + 1)
            )

        # 1 item normal
        conn.execute(
            """
            INSERT INTO ingestion_queue (
                file_absolute_path, collection_name
            ) VALUES (?, ?)
            """,
            ("/test/normal.py", "test-collection")
        )

        conn.commit()
        conn.close()

        retry_count, error_count = await collector._get_retry_error_counts()

        assert retry_count == 3
        assert error_count == 2

    @pytest.mark.asyncio
    async def test_calculate_rates_no_events(self, collector):
        """Test calculating rates with no events."""
        processing_rate, success_rate, failure_rate = await collector._calculate_rates(
            window_seconds=60
        )

        assert processing_rate == 0.0
        assert success_rate == 0.0
        assert failure_rate == 0.0

    @pytest.mark.asyncio
    async def test_calculate_rates_with_events(self, collector):
        """Test calculating rates with events."""
        # Record events
        for _ in range(10):
            await collector.record_event(event_type="success")

        for _ in range(2):
            await collector.record_event(event_type="failure")

        processing_rate, success_rate, failure_rate = await collector._calculate_rates(
            window_seconds=60
        )

        # 12 events in 60 seconds = 12 events/min
        assert processing_rate == 12.0
        # 10/12 = 83.33%
        assert abs(success_rate - 83.33) < 0.1
        # 2/12 = 16.67%
        assert abs(failure_rate - 16.67) < 0.1

    @pytest.mark.asyncio
    async def test_calculate_rates_filtered_by_collection(self, collector):
        """Test calculating rates filtered by collection."""
        # Record events for different collections
        for _ in range(5):
            await collector.record_event(
                event_type="success",
                collection="collection-a"
            )

        for _ in range(3):
            await collector.record_event(
                event_type="success",
                collection="collection-b"
            )

        rate_a, _, _ = await collector._calculate_rates(
            window_seconds=60,
            collection="collection-a"
        )

        rate_b, _, _ = await collector._calculate_rates(
            window_seconds=60,
            collection="collection-b"
        )

        assert rate_a == 5.0
        assert rate_b == 3.0

    @pytest.mark.asyncio
    async def test_calculate_avg_processing_time(self, collector):
        """Test calculating average processing time."""
        # Record events with processing times
        times = [100.0, 150.0, 200.0, 120.0, 180.0]
        for t in times:
            await collector.record_event(
                event_type="success",
                processing_time=t
            )

        avg_time = await collector._calculate_avg_processing_time(
            window_seconds=60
        )

        expected_avg = sum(times) / len(times)
        assert abs(avg_time - expected_avg) < 0.1

    @pytest.mark.asyncio
    async def test_calculate_avg_processing_time_no_events(self, collector):
        """Test calculating average processing time with no events."""
        avg_time = await collector._calculate_avg_processing_time(
            window_seconds=60
        )

        assert avg_time == 0.0

    @pytest.mark.asyncio
    async def test_calculate_event_rate(self, collector):
        """Test calculating event rate for specific type."""
        # Record added events
        for _ in range(8):
            await collector.record_event(event_type="added")

        # Record removed events
        for _ in range(5):
            await collector.record_event(event_type="removed")

        added_rate = await collector._calculate_event_rate(
            event_type="added",
            window_seconds=60
        )

        removed_rate = await collector._calculate_event_rate(
            event_type="removed",
            window_seconds=60
        )

        assert added_rate == 8.0
        assert removed_rate == 5.0

    @pytest.mark.asyncio
    async def test_get_current_statistics(self, collector, test_db):
        """Test getting current statistics."""
        # Insert queue items
        conn = sqlite3.connect(test_db)
        for i in range(5):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, priority
                ) VALUES (?, ?, ?)
                """,
                (f"/test/file{i}.py", "test-collection", 5 + i)
            )
        conn.commit()
        conn.close()

        # Record some events
        await collector.record_event(event_type="success", processing_time=100.0)
        await collector.record_event(event_type="success", processing_time=150.0)
        await collector.record_event(event_type="failure", processing_time=200.0)

        stats = await collector.get_current_statistics()

        assert stats.queue_size == 5
        assert stats.processing_rate > 0
        assert stats.success_rate > 0
        assert stats.failure_rate > 0
        assert stats.avg_processing_time > 0
        assert isinstance(stats.priority_distribution, dict)
        assert isinstance(stats.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_get_statistics_by_collection(self, collector, test_db):
        """Test getting statistics by collection."""
        # Insert items for specific collection
        conn = sqlite3.connect(test_db)
        for i in range(3):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, priority
                ) VALUES (?, ?, ?)
                """,
                (f"/test/file{i}.py", "my-collection", 5)
            )
        conn.commit()
        conn.close()

        # Record events for this collection
        await collector.record_event(
            event_type="success",
            collection="my-collection",
            processing_time=100.0
        )

        stats = await collector.get_statistics_by_collection("my-collection")

        assert stats.queue_size == 3
        assert isinstance(stats.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_get_statistics_by_tenant(self, collector, test_db):
        """Test getting statistics by tenant."""
        # Insert items for specific tenant
        conn = sqlite3.connect(test_db)
        for i in range(4):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id, priority
                ) VALUES (?, ?, ?, ?)
                """,
                (f"/test/file{i}.py", "test-collection", "my-tenant", 5)
            )
        conn.commit()
        conn.close()

        # Record events for this tenant
        await collector.record_event(
            event_type="success",
            tenant_id="my-tenant",
            processing_time=120.0
        )

        stats = await collector.get_statistics_by_tenant("my-tenant")

        assert stats.queue_size == 4
        assert isinstance(stats.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_start_stop_collection(self, collector):
        """Test starting and stopping background collection."""
        # Start collection
        started = await collector.start_collection(interval_seconds=1)
        assert started is True
        assert collector._collection_task is not None
        assert not collector._collection_task.done()

        # Try to start again (should fail)
        started_again = await collector.start_collection(interval_seconds=1)
        assert started_again is False

        # Let it run briefly
        await asyncio.sleep(0.5)

        # Stop collection
        stopped = await collector.stop_collection()
        assert stopped is True

        # Try to stop again (should fail)
        stopped_again = await collector.stop_collection()
        assert stopped_again is False

    @pytest.mark.asyncio
    async def test_collection_loop_updates_stats(self, collector, test_db):
        """Test that collection loop updates statistics."""
        # Insert items
        conn = sqlite3.connect(test_db)
        for i in range(3):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name
                ) VALUES (?, ?)
                """,
                (f"/test/file{i}.py", "test-collection")
            )
        conn.commit()
        conn.close()

        # Start collection
        await collector.start_collection(interval_seconds=1)

        # Wait for at least one collection cycle
        await asyncio.sleep(1.5)

        # Check that latest stats were cached
        assert collector._latest_stats is not None
        assert collector._latest_stats.queue_size == 3

        # Stop collection
        await collector.stop_collection()

    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, collector):
        """Test cleanup of old events."""
        # Record events with old timestamps
        current_time = time.time()

        # Create old event (16 minutes ago)
        old_event = ProcessingEvent(
            timestamp=current_time - 960,  # 16 minutes
            event_type="success"
        )

        # Create recent event
        recent_event = ProcessingEvent(
            timestamp=current_time,
            event_type="success"
        )

        async with collector._lock:
            collector._events.append(old_event)
            collector._events.append(recent_event)

        # Run cleanup
        await collector._cleanup_old_events()

        async with collector._lock:
            # Old event should be removed (older than 15 min window)
            assert len(collector._events) == 1
            assert collector._events[0].timestamp == recent_event.timestamp

    @pytest.mark.asyncio
    async def test_memory_efficiency_max_events(self, collector):
        """Test that max_events limit is enforced."""
        # Collector has max_events=1000 from fixture

        # Record more than max events
        for i in range(1500):
            await collector.record_event(event_type="success")

        async with collector._lock:
            # Should be limited to max_events
            assert len(collector._events) == 1000

    @pytest.mark.asyncio
    async def test_edge_case_empty_queue(self, collector):
        """Test edge case with empty queue."""
        stats = await collector.get_current_statistics()

        assert stats.queue_size == 0
        assert stats.processing_rate == 0.0
        assert stats.success_rate == 0.0
        assert stats.failure_rate == 0.0
        assert stats.avg_processing_time == 0.0
        assert stats.retry_count == 0
        assert stats.error_count == 0

    @pytest.mark.asyncio
    async def test_edge_case_no_activity(self, collector, test_db):
        """Test edge case with queue items but no processing activity."""
        # Insert items
        conn = sqlite3.connect(test_db)
        for i in range(5):
            conn.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name
                ) VALUES (?, ?)
                """,
                (f"/test/file{i}.py", "test-collection")
            )
        conn.commit()
        conn.close()

        stats = await collector.get_current_statistics()

        # Should have queue size but zero rates
        assert stats.queue_size == 5
        assert stats.processing_rate == 0.0
        assert stats.success_rate == 0.0
        assert stats.failure_rate == 0.0

    @pytest.mark.asyncio
    async def test_multiple_queue_types(self, collector, test_db):
        """Test handling multiple queue types."""
        # Create additional queue table
        conn = sqlite3.connect(test_db)
        conn.execute("""
            CREATE TABLE retry_queue (
                file_absolute_path TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT 'default',
                branch TEXT NOT NULL DEFAULT 'main',
                operation TEXT NOT NULL DEFAULT 'ingest',
                priority INTEGER NOT NULL DEFAULT 5,
                queued_timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                retry_count INTEGER NOT NULL DEFAULT 0,
                retry_from TEXT,
                error_message_id INTEGER,
                collection_type TEXT
            )
        """)

        # Insert items into different queues
        conn.execute(
            """
            INSERT INTO ingestion_queue (
                file_absolute_path, collection_name
            ) VALUES (?, ?)
            """,
            ("/test/ingest.py", "test-collection")
        )

        conn.execute(
            """
            INSERT INTO retry_queue (
                file_absolute_path, collection_name
            ) VALUES (?, ?)
            """,
            ("/test/retry.py", "test-collection")
        )

        conn.commit()
        conn.close()

        # Get stats for different queue types
        ingest_stats = await collector.get_statistics_by_queue("ingestion_queue")
        retry_stats = await collector.get_statistics_by_queue("retry_queue")

        assert ingest_stats.queue_size == 1
        assert retry_stats.queue_size == 1
