"""
Unit Tests for Queue Performance Metrics Module

Tests comprehensive performance metrics collection including percentile calculations,
throughput analysis, latency tracking, and metrics export.
"""

import asyncio
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import pytest

from src.python.common.core.queue_performance_metrics import (
    ThroughputMetrics,
    LatencyMetrics,
    MetricsAggregator,
    PerformanceMetrics,
    QueuePerformanceCollector
)
from src.python.common.core.queue_connection import ConnectionConfig


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
    """Create performance collector with test database."""
    config = ConnectionConfig(
        busy_timeout=5.0,
        max_connections=5
    )
    collector = QueuePerformanceCollector(
        db_path=test_db,
        connection_config=config,
        window_minutes=5,
        max_events=1000,
        enable_resource_tracking=True
    )
    await collector.initialize()
    yield collector
    await collector.close()


class TestThroughputMetrics:
    """Test ThroughputMetrics dataclass."""

    def test_throughput_metrics_creation(self):
        """Test creating ThroughputMetrics with default values."""
        metrics = ThroughputMetrics()

        assert metrics.items_per_second == 0.0
        assert metrics.items_per_minute == 0.0
        assert metrics.total_items == 0
        assert metrics.window_seconds == 300
        assert isinstance(metrics.timestamp, datetime)

    def test_throughput_metrics_to_dict(self):
        """Test converting ThroughputMetrics to dictionary."""
        metrics = ThroughputMetrics(
            items_per_second=5.5,
            items_per_minute=330.0,
            total_items=100,
            window_seconds=300
        )

        result = metrics.to_dict()

        assert result["items_per_second"] == 5.5
        assert result["items_per_minute"] == 330.0
        assert result["total_items"] == 100
        assert result["window_seconds"] == 300
        assert "timestamp" in result


class TestLatencyMetrics:
    """Test LatencyMetrics dataclass."""

    def test_latency_metrics_creation(self):
        """Test creating LatencyMetrics with default values."""
        metrics = LatencyMetrics()

        assert metrics.avg_latency_ms == 0.0
        assert metrics.min_latency_ms == 0.0
        assert metrics.max_latency_ms == 0.0
        assert metrics.total_items == 0
        assert isinstance(metrics.timestamp, datetime)

    def test_latency_metrics_to_dict(self):
        """Test converting LatencyMetrics to dictionary."""
        metrics = LatencyMetrics(
            avg_latency_ms=150.5,
            min_latency_ms=50.0,
            max_latency_ms=300.0,
            total_items=100
        )

        result = metrics.to_dict()

        assert result["avg_latency_ms"] == 150.5
        assert result["min_latency_ms"] == 50.0
        assert result["max_latency_ms"] == 300.0
        assert result["total_items"] == 100
        assert "timestamp" in result


class TestMetricsAggregator:
    """Test MetricsAggregator dataclass."""

    def test_metrics_aggregator_creation(self):
        """Test creating MetricsAggregator with default values."""
        aggregator = MetricsAggregator()

        assert aggregator.min == 0.0
        assert aggregator.max == 0.0
        assert aggregator.avg == 0.0
        assert aggregator.p50 == 0.0
        assert aggregator.p95 == 0.0
        assert aggregator.p99 == 0.0
        assert aggregator.count == 0

    def test_metrics_aggregator_to_dict(self):
        """Test converting MetricsAggregator to dictionary."""
        aggregator = MetricsAggregator(
            min=50.0,
            max=500.0,
            avg=150.0,
            p50=120.0,
            p95=400.0,
            p99=480.0,
            count=100
        )

        result = aggregator.to_dict()

        assert result["min"] == 50.0
        assert result["max"] == 500.0
        assert result["avg"] == 150.0
        assert result["p50"] == 120.0
        assert result["p95"] == 400.0
        assert result["p99"] == 480.0
        assert result["count"] == 100


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics with default values."""
        metrics = PerformanceMetrics()

        assert isinstance(metrics.throughput, ThroughputMetrics)
        assert isinstance(metrics.latency, LatencyMetrics)
        assert isinstance(metrics.processing_time, MetricsAggregator)
        assert metrics.resource_usage == {}
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert isinstance(metrics.timestamp, datetime)

    def test_performance_metrics_to_dict(self):
        """Test converting PerformanceMetrics to dictionary."""
        metrics = PerformanceMetrics(
            success_count=95,
            failure_count=5,
            resource_usage={"cpu_percent": 25.5, "memory_mb": 150.0}
        )

        result = metrics.to_dict()

        assert "throughput" in result
        assert "latency" in result
        assert "processing_time" in result
        assert result["resource_usage"]["cpu_percent"] == 25.5
        assert result["success_count"] == 95
        assert result["failure_count"] == 5
        assert "timestamp" in result


class TestQueuePerformanceCollector:
    """Test QueuePerformanceCollector."""

    @pytest.mark.asyncio
    async def test_initialization(self, collector):
        """Test collector initialization."""
        assert collector._initialized
        assert collector.statistics_collector is not None
        assert collector.window_minutes == 5
        assert collector.enable_resource_tracking is True

    @pytest.mark.asyncio
    async def test_record_processing_event(self, collector):
        """Test recording processing events."""
        await collector.record_processing_event(
            duration_ms=150.5,
            success=True,
            metadata={
                "collection": "test-collection",
                "tenant_id": "test-tenant"
            }
        )

        await collector.record_processing_event(
            duration_ms=200.0,
            success=False,
            metadata={
                "collection": "test-collection",
                "tenant_id": "test-tenant"
            }
        )

        # Events should be recorded in base statistics collector
        async with collector.statistics_collector._lock:
            assert len(collector.statistics_collector._events) == 2

    @pytest.mark.asyncio
    async def test_record_processing_event_with_latency(self, collector):
        """Test recording events with latency tracking."""
        enqueue_time = time.time() - 0.1  # 100ms ago

        await collector.record_processing_event(
            duration_ms=50.0,
            success=True,
            metadata={
                "enqueue_time": enqueue_time,
                "collection": "test-collection"
            }
        )

        # Latency should be recorded
        async with collector._lock:
            assert len(collector._latency_events) == 1
            timestamp, latency = collector._latency_events[0]
            assert latency >= 100.0  # At least 100ms

    @pytest.mark.asyncio
    async def test_get_throughput_metrics_no_events(self, collector):
        """Test getting throughput metrics with no events."""
        metrics = await collector.get_throughput_metrics(window_minutes=5)

        assert metrics.items_per_second == 0.0
        assert metrics.items_per_minute == 0.0
        assert metrics.total_items == 0
        assert metrics.window_seconds == 300

    @pytest.mark.asyncio
    async def test_get_throughput_metrics_with_events(self, collector):
        """Test getting throughput metrics with events."""
        # Record 10 successful events
        for _ in range(10):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True
            )

        # Record 2 failed events
        for _ in range(2):
            await collector.record_processing_event(
                duration_ms=150.0,
                success=False
            )

        metrics = await collector.get_throughput_metrics(window_minutes=5)

        assert metrics.total_items == 12
        assert metrics.items_per_second > 0
        assert metrics.items_per_minute == metrics.items_per_second * 60

    @pytest.mark.asyncio
    async def test_get_throughput_metrics_filtered_by_collection(self, collector):
        """Test getting throughput filtered by collection."""
        # Record events for different collections
        for _ in range(5):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True,
                metadata={"collection": "collection-a"}
            )

        for _ in range(3):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True,
                metadata={"collection": "collection-b"}
            )

        metrics_a = await collector.get_throughput_metrics(
            window_minutes=5,
            collection="collection-a"
        )
        metrics_b = await collector.get_throughput_metrics(
            window_minutes=5,
            collection="collection-b"
        )

        assert metrics_a.total_items == 5
        assert metrics_b.total_items == 3

    @pytest.mark.asyncio
    async def test_get_throughput_metrics_filtered_by_tenant(self, collector):
        """Test getting throughput filtered by tenant."""
        # Record events for different tenants
        for _ in range(4):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True,
                metadata={"tenant_id": "tenant-1"}
            )

        for _ in range(2):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True,
                metadata={"tenant_id": "tenant-2"}
            )

        metrics_1 = await collector.get_throughput_metrics(
            window_minutes=5,
            tenant_id="tenant-1"
        )
        metrics_2 = await collector.get_throughput_metrics(
            window_minutes=5,
            tenant_id="tenant-2"
        )

        assert metrics_1.total_items == 4
        assert metrics_2.total_items == 2

    @pytest.mark.asyncio
    async def test_get_latency_metrics_no_events(self, collector):
        """Test getting latency metrics with no events."""
        metrics = await collector.get_latency_metrics(window_minutes=5)

        assert metrics.avg_latency_ms == 0.0
        assert metrics.min_latency_ms == 0.0
        assert metrics.max_latency_ms == 0.0
        assert metrics.total_items == 0

    @pytest.mark.asyncio
    async def test_get_latency_metrics_with_events(self, collector):
        """Test getting latency metrics with events."""
        # Record events with different latencies
        enqueue_times = [
            time.time() - 0.1,  # 100ms ago
            time.time() - 0.2,  # 200ms ago
            time.time() - 0.05  # 50ms ago
        ]

        for enqueue_time in enqueue_times:
            await collector.record_processing_event(
                duration_ms=50.0,
                success=True,
                metadata={"enqueue_time": enqueue_time}
            )

        metrics = await collector.get_latency_metrics(window_minutes=5)

        assert metrics.total_items == 3
        assert metrics.avg_latency_ms > 0
        assert metrics.min_latency_ms > 0
        assert metrics.max_latency_ms > metrics.min_latency_ms

    @pytest.mark.asyncio
    async def test_get_processing_time_stats_no_events(self, collector):
        """Test getting processing time stats with no events."""
        stats = await collector.get_processing_time_stats()

        assert stats.min == 0.0
        assert stats.max == 0.0
        assert stats.avg == 0.0
        assert stats.p50 == 0.0
        assert stats.p95 == 0.0
        assert stats.p99 == 0.0
        assert stats.count == 0

    @pytest.mark.asyncio
    async def test_get_processing_time_stats_with_events(self, collector):
        """Test getting processing time stats with events."""
        # Record events with various processing times
        times = [100.0, 150.0, 200.0, 120.0, 180.0, 90.0, 300.0, 250.0, 110.0, 130.0]
        for t in times:
            await collector.record_processing_event(
                duration_ms=t,
                success=True
            )

        stats = await collector.get_processing_time_stats()

        assert stats.count == 10
        assert stats.min == 90.0
        assert stats.max == 300.0
        assert stats.avg == sum(times) / len(times)
        assert stats.p50 > 0
        assert stats.p95 > stats.p50
        assert stats.p99 > stats.p95

    @pytest.mark.asyncio
    async def test_percentile_calculation_edge_cases(self, collector):
        """Test percentile calculation edge cases."""
        # Single event
        await collector.record_processing_event(duration_ms=100.0, success=True)
        stats = await collector.get_processing_time_stats()
        assert stats.p50 == 100.0
        assert stats.p95 == 100.0
        assert stats.p99 == 100.0

    @pytest.mark.asyncio
    async def test_percentile_calculation_accuracy(self, collector):
        """Test percentile calculation accuracy."""
        # Create known distribution: 1, 2, 3, ..., 100
        for i in range(1, 101):
            await collector.record_processing_event(
                duration_ms=float(i),
                success=True
            )

        stats = await collector.get_processing_time_stats()

        # For 1-100, median should be ~50.5
        assert 49 <= stats.p50 <= 51
        # P95 should be ~95
        assert 94 <= stats.p95 <= 96
        # P99 should be ~99
        assert 98 <= stats.p99 <= 100

    @pytest.mark.asyncio
    async def test_get_processing_time_stats_filtered_by_collection(self, collector):
        """Test getting processing time stats filtered by collection."""
        # Collection A: faster processing
        for i in range(5):
            await collector.record_processing_event(
                duration_ms=100.0 + i * 10,
                success=True,
                metadata={"collection": "collection-a"}
            )

        # Collection B: slower processing
        for i in range(5):
            await collector.record_processing_event(
                duration_ms=200.0 + i * 20,
                success=True,
                metadata={"collection": "collection-b"}
            )

        stats_a = await collector.get_processing_time_stats(collection="collection-a")
        stats_b = await collector.get_processing_time_stats(collection="collection-b")

        assert stats_a.count == 5
        assert stats_b.count == 5
        assert stats_a.avg < stats_b.avg  # Collection A is faster

    @pytest.mark.asyncio
    async def test_get_processing_time_stats_filtered_by_tenant(self, collector):
        """Test getting processing time stats filtered by tenant."""
        # Tenant 1: fast processing
        for _ in range(3):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True,
                metadata={"tenant_id": "tenant-1"}
            )

        # Tenant 2: slow processing
        for _ in range(3):
            await collector.record_processing_event(
                duration_ms=500.0,
                success=True,
                metadata={"tenant_id": "tenant-2"}
            )

        stats_1 = await collector.get_processing_time_stats(tenant_id="tenant-1")
        stats_2 = await collector.get_processing_time_stats(tenant_id="tenant-2")

        assert stats_1.count == 3
        assert stats_2.count == 3
        assert stats_1.avg < stats_2.avg

    @pytest.mark.asyncio
    async def test_get_metrics_by_collection(self, collector):
        """Test getting comprehensive metrics by collection."""
        # Record events for collection
        for _ in range(5):
            await collector.record_processing_event(
                duration_ms=150.0,
                success=True,
                metadata={"collection": "my-collection"}
            )

        for _ in range(2):
            await collector.record_processing_event(
                duration_ms=200.0,
                success=False,
                metadata={"collection": "my-collection"}
            )

        metrics = await collector.get_metrics_by_collection("my-collection")

        assert metrics.throughput.total_items == 7
        assert metrics.processing_time.count == 7
        assert metrics.success_count == 5
        assert metrics.failure_count == 2
        assert isinstance(metrics.resource_usage, dict)

    @pytest.mark.asyncio
    async def test_get_metrics_by_tenant(self, collector):
        """Test getting comprehensive metrics by tenant."""
        # Record events for tenant
        for _ in range(10):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True,
                metadata={"tenant_id": "my-tenant"}
            )

        for _ in range(1):
            await collector.record_processing_event(
                duration_ms=150.0,
                success=False,
                metadata={"tenant_id": "my-tenant"}
            )

        metrics = await collector.get_metrics_by_tenant("my-tenant")

        assert metrics.throughput.total_items == 11
        assert metrics.processing_time.count == 11
        assert metrics.success_count == 10
        assert metrics.failure_count == 1

    @pytest.mark.asyncio
    async def test_export_metrics_json_format(self, collector):
        """Test exporting metrics in JSON format."""
        # Record some events
        for _ in range(5):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True
            )

        export = await collector.export_metrics(format='json')

        # Parse JSON
        data = json.loads(export)

        assert "metadata" in data
        assert "throughput" in data
        assert "latency" in data
        assert "processing_time" in data
        assert "success_count" in data
        assert "failure_count" in data
        assert "success_rate" in data

        # Check metadata
        assert data["metadata"]["window_minutes"] == 5
        assert "timestamp" in data["metadata"]

    @pytest.mark.asyncio
    async def test_export_metrics_prometheus_format(self, collector):
        """Test exporting metrics in Prometheus format."""
        # Record some events
        for _ in range(5):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True
            )

        export = await collector.export_metrics(format='prometheus')

        # Check Prometheus format
        assert "# HELP" in export
        assert "# TYPE" in export
        assert "queue_throughput_items_per_second" in export
        assert "queue_latency_milliseconds" in export
        assert "queue_processing_time_milliseconds" in export
        assert "queue_processing_total" in export

        # Check labels
        assert 'stat="p50"' in export
        assert 'stat="p95"' in export
        assert 'stat="p99"' in export
        assert 'status="success"' in export
        assert 'status="failure"' in export

    @pytest.mark.asyncio
    async def test_export_metrics_invalid_format(self, collector):
        """Test exporting with invalid format raises error."""
        with pytest.raises(ValueError, match="Unsupported format"):
            await collector.export_metrics(format='xml')

    @pytest.mark.asyncio
    async def test_export_metrics_includes_resource_usage(self, collector):
        """Test that exports include resource usage when enabled."""
        export = await collector.export_metrics(format='json')
        data = json.loads(export)

        # Should have resource usage (if psutil is available)
        assert "resource_usage" in data
        # Resource usage might be empty dict if psutil fails, but key should exist
        assert isinstance(data["resource_usage"], dict)

    @pytest.mark.asyncio
    async def test_memory_efficiency_max_events(self, collector):
        """Test that max_events limit is enforced."""
        # Collector has max_events=1000 from fixture

        # Record more than max events
        for i in range(1500):
            await collector.record_processing_event(
                duration_ms=100.0,
                success=True
            )

        async with collector.statistics_collector._lock:
            # Should be limited to max_events
            assert len(collector.statistics_collector._events) == 1000

    @pytest.mark.asyncio
    async def test_edge_case_outliers(self, collector):
        """Test handling of outlier values."""
        # Record normal values
        for _ in range(10):
            await collector.record_processing_event(duration_ms=100.0, success=True)

        # Record outlier
        await collector.record_processing_event(duration_ms=10000.0, success=True)

        stats = await collector.get_processing_time_stats()

        # Outlier should affect max and p99 but not median much
        assert stats.max == 10000.0
        assert stats.p99 > stats.p50
        assert stats.p50 < 200.0  # Median should still be low

    @pytest.mark.asyncio
    async def test_window_sliding_behavior(self, collector):
        """Test sliding window behavior."""
        # Record old event (manually add to deque with old timestamp)
        old_time = time.time() - 400  # 400 seconds ago (outside 5-min window)
        from src.python.common.core.queue_statistics import ProcessingEvent

        old_event = ProcessingEvent(
            timestamp=old_time,
            event_type="success",
            processing_time=100.0
        )

        async with collector.statistics_collector._lock:
            collector.statistics_collector._events.append(old_event)

        # Record recent event
        await collector.record_processing_event(duration_ms=200.0, success=True)

        # Get stats with 5-minute window
        stats = await collector.get_processing_time_stats(window_minutes=5)

        # Should only include recent event (old event is outside window)
        assert stats.count == 1
        assert stats.avg == 200.0

    @pytest.mark.asyncio
    async def test_concurrent_recording(self, collector):
        """Test concurrent event recording."""
        async def record_events(count: int):
            for _ in range(count):
                await collector.record_processing_event(
                    duration_ms=100.0,
                    success=True
                )

        # Record concurrently
        await asyncio.gather(
            record_events(10),
            record_events(10),
            record_events(10)
        )

        async with collector.statistics_collector._lock:
            assert len(collector.statistics_collector._events) == 30

    @pytest.mark.asyncio
    async def test_custom_window_minutes(self, collector):
        """Test using custom window_minutes parameter."""
        # Record events
        for _ in range(5):
            await collector.record_processing_event(duration_ms=100.0, success=True)

        # Get metrics with different windows
        metrics_1min = await collector.get_throughput_metrics(window_minutes=1)
        metrics_5min = await collector.get_throughput_metrics(window_minutes=5)

        # Both should have same total_items
        assert metrics_1min.total_items == 5
        assert metrics_5min.total_items == 5

        # But different window_seconds
        assert metrics_1min.window_seconds == 60
        assert metrics_5min.window_seconds == 300

        # Items/second should differ based on window
        assert metrics_1min.items_per_second > metrics_5min.items_per_second
