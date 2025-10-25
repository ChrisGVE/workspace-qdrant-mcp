"""
Unit tests for Queue Resource Monitor

Tests resource utilization tracking, correlation analysis, and bottleneck detection.
"""

import asyncio
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Import the module to test
from common.core.queue_resource_monitor import (
    PSUTIL_AVAILABLE,
    QueueResourceMonitor,
    ResourceBottleneck,
    ResourceCorrelation,
    ResourceMetrics,
    ResourceSnapshot,
)
from common.core.queue_statistics import QueueStatistics


class TestResourceMetrics:
    """Test ResourceMetrics dataclass."""

    def test_resource_metrics_creation(self):
        """Test creating ResourceMetrics with default values."""
        metrics = ResourceMetrics()

        assert metrics.cpu_percent == 0.0
        assert metrics.memory_mb == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.db_connections == 0
        assert metrics.thread_count == 0
        assert metrics.file_descriptors is None
        assert metrics.disk_io_read_mb is None
        assert metrics.disk_io_write_mb is None

    def test_resource_metrics_with_values(self):
        """Test creating ResourceMetrics with custom values."""
        metrics = ResourceMetrics(
            cpu_percent=45.5,
            memory_mb=512.75,
            memory_percent=25.3,
            db_connections=5,
            thread_count=10,
            file_descriptors=100,
            disk_io_read_mb=1024.5,
            disk_io_write_mb=512.25
        )

        assert metrics.cpu_percent == 45.5
        assert metrics.memory_mb == 512.75
        assert metrics.memory_percent == 25.3
        assert metrics.db_connections == 5
        assert metrics.thread_count == 10
        assert metrics.file_descriptors == 100
        assert metrics.disk_io_read_mb == 1024.5
        assert metrics.disk_io_write_mb == 512.25

    def test_resource_metrics_to_dict(self):
        """Test converting ResourceMetrics to dictionary."""
        metrics = ResourceMetrics(
            cpu_percent=45.567,
            memory_mb=512.789,
            memory_percent=25.345,
            db_connections=5,
            thread_count=10
        )

        result = metrics.to_dict()

        assert result["cpu_percent"] == 45.57  # Rounded to 2 decimals
        assert result["memory_mb"] == 512.79
        assert result["memory_percent"] == 25.34
        assert result["db_connections"] == 5
        assert result["thread_count"] == 10
        assert result["file_descriptors"] is None
        assert result["disk_io_read_mb"] is None
        assert result["disk_io_write_mb"] is None


class TestResourceSnapshot:
    """Test ResourceSnapshot dataclass."""

    def test_resource_snapshot_creation(self):
        """Test creating ResourceSnapshot."""
        timestamp = datetime.now(timezone.utc)
        metrics = ResourceMetrics(cpu_percent=50.0, memory_mb=1024.0)
        queue_stats = QueueStatistics(queue_size=100, processing_rate=10.0)

        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            metrics=metrics,
            queue_stats=queue_stats
        )

        assert snapshot.timestamp == timestamp
        assert snapshot.metrics == metrics
        assert snapshot.queue_stats == queue_stats

    def test_resource_snapshot_to_dict(self):
        """Test converting ResourceSnapshot to dictionary."""
        timestamp = datetime(2025, 10, 3, 12, 0, 0, tzinfo=timezone.utc)
        metrics = ResourceMetrics(cpu_percent=50.0, memory_mb=1024.0)
        queue_stats = QueueStatistics(queue_size=100, processing_rate=10.0)

        snapshot = ResourceSnapshot(
            timestamp=timestamp,
            metrics=metrics,
            queue_stats=queue_stats
        )

        result = snapshot.to_dict()

        assert "timestamp" in result
        assert "metrics" in result
        assert "queue_stats" in result
        assert result["timestamp"] == timestamp.isoformat()


class TestResourceCorrelation:
    """Test ResourceCorrelation dataclass."""

    def test_resource_correlation_creation(self):
        """Test creating ResourceCorrelation."""
        corr = ResourceCorrelation(
            resource_metric="cpu_percent",
            queue_metric="processing_rate",
            correlation_coefficient=-0.75,
            sample_size=100
        )

        assert corr.resource_metric == "cpu_percent"
        assert corr.queue_metric == "processing_rate"
        assert corr.correlation_coefficient == -0.75
        assert corr.sample_size == 100

    def test_correlation_strength_strong(self):
        """Test correlation strength classification - strong."""
        corr = ResourceCorrelation(
            resource_metric="cpu_percent",
            queue_metric="processing_rate",
            correlation_coefficient=-0.85,
            sample_size=50
        )

        assert corr._get_correlation_strength() == "strong"

    def test_correlation_strength_moderate(self):
        """Test correlation strength classification - moderate."""
        corr = ResourceCorrelation(
            resource_metric="memory_mb",
            queue_metric="queue_size",
            correlation_coefficient=0.55,
            sample_size=50
        )

        assert corr._get_correlation_strength() == "moderate"

    def test_correlation_strength_weak(self):
        """Test correlation strength classification - weak."""
        corr = ResourceCorrelation(
            resource_metric="memory_mb",
            queue_metric="error_rate",
            correlation_coefficient=0.25,
            sample_size=50
        )

        assert corr._get_correlation_strength() == "weak"

    def test_correlation_strength_negligible(self):
        """Test correlation strength classification - negligible."""
        corr = ResourceCorrelation(
            resource_metric="thread_count",
            queue_metric="processing_rate",
            correlation_coefficient=0.05,
            sample_size=50
        )

        assert corr._get_correlation_strength() == "negligible"

    def test_resource_correlation_to_dict(self):
        """Test converting ResourceCorrelation to dictionary."""
        corr = ResourceCorrelation(
            resource_metric="cpu_percent",
            queue_metric="processing_rate",
            correlation_coefficient=-0.7532,
            sample_size=100
        )

        result = corr.to_dict()

        assert result["resource_metric"] == "cpu_percent"
        assert result["queue_metric"] == "processing_rate"
        assert result["correlation_coefficient"] == -0.7532  # Rounded to 4 decimals
        assert result["sample_size"] == 100
        assert result["strength"] == "strong"


class TestResourceBottleneck:
    """Test ResourceBottleneck dataclass."""

    def test_resource_bottleneck_creation(self):
        """Test creating ResourceBottleneck."""
        bottleneck = ResourceBottleneck(
            resource_type="CPU",
            current_value=95.5,
            threshold_value=90.0,
            severity="critical",
            queue_impact="Low processing rate with high CPU usage",
            recommendation="Increase CPU resources"
        )

        assert bottleneck.resource_type == "CPU"
        assert bottleneck.current_value == 95.5
        assert bottleneck.threshold_value == 90.0
        assert bottleneck.severity == "critical"
        assert bottleneck.queue_impact == "Low processing rate with high CPU usage"
        assert bottleneck.recommendation == "Increase CPU resources"

    def test_resource_bottleneck_to_dict(self):
        """Test converting ResourceBottleneck to dictionary."""
        bottleneck = ResourceBottleneck(
            resource_type="Memory",
            current_value=2048.567,
            threshold_value=2000.0,
            severity="critical",
            queue_impact="Queue backlog with high memory usage",
            recommendation="Increase memory allocation"
        )

        result = bottleneck.to_dict()

        assert result["resource_type"] == "Memory"
        assert result["current_value"] == 2048.57  # Rounded to 2 decimals
        assert result["threshold_value"] == 2000.0
        assert result["severity"] == "critical"
        assert result["queue_impact"] == "Queue backlog with high memory usage"
        assert result["recommendation"] == "Increase memory allocation"


class TestQueueResourceMonitor:
    """Test QueueResourceMonitor main class."""

    @pytest.fixture
    async def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        # Initialize database schema
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_queue (
                id INTEGER PRIMARY KEY,
                collection_name TEXT,
                priority INTEGER DEFAULT 5,
                retry_count INTEGER DEFAULT 0,
                error_message_id INTEGER,
                created_at TEXT,
                tenant_id TEXT
            )
        """)
        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        try:
            os.unlink(db_path)
        except Exception:
            pass

    @pytest.fixture
    async def monitor(self, temp_db):
        """Create QueueResourceMonitor instance for testing."""
        monitor = QueueResourceMonitor(
            db_path=temp_db,
            snapshot_retention=10,
            monitoring_interval=1
        )
        await monitor.initialize()
        yield monitor
        await monitor.close()

    @pytest.mark.asyncio
    async def test_monitor_initialization(self, temp_db):
        """Test monitor initialization."""
        monitor = QueueResourceMonitor(db_path=temp_db)

        assert not monitor._initialized

        await monitor.initialize()

        assert monitor._initialized
        assert monitor.connection_pool is not None
        assert monitor.stats_collector is not None

        await monitor.close()

    @pytest.mark.asyncio
    async def test_monitor_close(self, temp_db):
        """Test monitor cleanup."""
        monitor = QueueResourceMonitor(db_path=temp_db)
        await monitor.initialize()

        assert monitor._initialized

        await monitor.close()

        assert not monitor._initialized

    @pytest.mark.asyncio
    async def test_get_current_resources_without_psutil(self, monitor):
        """Test resource collection when psutil is unavailable."""
        with patch('common.core.queue_resource_monitor.PSUTIL_AVAILABLE', False):
            metrics = await monitor.get_current_resources()

            # Should return empty metrics
            assert metrics.cpu_percent == 0.0
            assert metrics.memory_mb == 0.0
            assert metrics.thread_count == 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    async def test_get_current_resources_with_psutil(self, monitor):
        """Test resource collection with psutil available."""
        with patch.object(monitor, '_process') as mock_process:
            # Mock psutil methods
            mock_process.cpu_percent.return_value = 45.5
            mock_process.memory_info.return_value = Mock(rss=1024 * 1024 * 512)  # 512 MB
            mock_process.memory_percent.return_value = 25.3
            mock_process.num_threads.return_value = 10

            # Mock connection pool
            monitor.connection_pool.active_connections = 5

            metrics = await monitor.get_current_resources()

            assert metrics.cpu_percent == 45.5
            assert metrics.memory_mb == 512.0
            assert metrics.memory_percent == 25.3
            assert metrics.thread_count == 10
            assert metrics.db_connections == 5

    @pytest.mark.asyncio
    async def test_take_snapshot(self, monitor):
        """Test taking a resource snapshot."""
        # Mock methods
        mock_metrics = ResourceMetrics(cpu_percent=50.0, memory_mb=1024.0)
        mock_stats = QueueStatistics(queue_size=100, processing_rate=10.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        snapshot = await monitor.take_snapshot()

        assert snapshot.metrics == mock_metrics
        assert snapshot.queue_stats == mock_stats
        assert len(monitor._snapshots) == 1

    @pytest.mark.asyncio
    async def test_snapshot_sliding_window(self, monitor):
        """Test snapshot retention with sliding window."""
        # Set retention to 3
        from collections import deque; monitor._snapshots = deque(maxlen=3)

        mock_metrics = ResourceMetrics(cpu_percent=50.0, memory_mb=1024.0)
        mock_stats = QueueStatistics(queue_size=100, processing_rate=10.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        # Take 5 snapshots
        for _i in range(5):
            await monitor.take_snapshot()

        # Should only retain last 3
        assert len(monitor._snapshots) == 3

    @pytest.mark.asyncio
    async def test_get_resource_history(self, monitor):
        """Test retrieving resource history."""
        # Create snapshots with different timestamps
        now = datetime.now(timezone.utc)

        snapshot1 = ResourceSnapshot(
            timestamp=now - timedelta(minutes=90),
            metrics=ResourceMetrics(cpu_percent=40.0),
            queue_stats=QueueStatistics()
        )
        snapshot2 = ResourceSnapshot(
            timestamp=now - timedelta(minutes=30),
            metrics=ResourceMetrics(cpu_percent=50.0),
            queue_stats=QueueStatistics()
        )
        snapshot3 = ResourceSnapshot(
            timestamp=now - timedelta(minutes=5),
            metrics=ResourceMetrics(cpu_percent=60.0),
            queue_stats=QueueStatistics()
        )

        monitor._snapshots.extend([snapshot1, snapshot2, snapshot3])

        # Get last 60 minutes
        history = await monitor.get_resource_history(minutes=60)

        # Should only include snapshot2 and snapshot3
        assert len(history) == 2
        assert snapshot2 in history
        assert snapshot3 in history

    @pytest.mark.asyncio
    async def test_pearson_correlation(self, monitor):
        """Test Pearson correlation calculation."""
        # Perfect positive correlation
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]

        corr = monitor._pearson_correlation(x, y)
        assert abs(corr - 1.0) < 0.01  # Should be very close to 1.0

        # Perfect negative correlation
        y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
        corr_neg = monitor._pearson_correlation(x, y_neg)
        assert abs(corr_neg - (-1.0)) < 0.01  # Should be very close to -1.0

        # No correlation
        y_rand = [5.0, 2.0, 8.0, 1.0, 9.0]
        corr_none = monitor._pearson_correlation(x, y_rand)
        # Should be closer to 0 (not exactly 0 due to small sample)
        assert abs(corr_none) < 0.5

    @pytest.mark.asyncio
    async def test_pearson_correlation_edge_cases(self, monitor):
        """Test Pearson correlation with edge cases."""
        # Empty lists
        assert monitor._pearson_correlation([], []) == 0.0

        # Single value
        assert monitor._pearson_correlation([1.0], [2.0]) == 0.0

        # Different lengths
        assert monitor._pearson_correlation([1.0, 2.0], [1.0]) == 0.0

        # Zero variance in one variable
        assert monitor._pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0

    @pytest.mark.asyncio
    async def test_correlate_with_queue_performance(self, monitor):
        """Test correlation analysis with queue performance."""
        # Create snapshots with varying metrics
        now = datetime.now(timezone.utc)

        for i in range(10):
            snapshot = ResourceSnapshot(
                timestamp=now - timedelta(minutes=i),
                metrics=ResourceMetrics(
                    cpu_percent=50.0 + i * 5,
                    memory_mb=1000.0 + i * 100
                ),
                queue_stats=QueueStatistics(
                    queue_size=100 - i * 10,
                    processing_rate=10.0 + i,
                    failure_rate=5.0 - i * 0.5
                )
            )
            monitor._snapshots.append(snapshot)

        correlations = await monitor.correlate_with_queue_performance()

        # Should have 5 correlations
        assert len(correlations) == 5

        # Check correlation types
        metric_pairs = [(c.resource_metric, c.queue_metric) for c in correlations]
        assert ("cpu_percent", "processing_rate") in metric_pairs
        assert ("cpu_percent", "queue_size") in metric_pairs
        assert ("memory_mb", "processing_rate") in metric_pairs
        assert ("memory_mb", "queue_size") in metric_pairs
        assert ("memory_mb", "error_rate") in metric_pairs

        # All should have sample size of 10
        for corr in correlations:
            assert corr.sample_size == 10

    @pytest.mark.asyncio
    async def test_correlate_insufficient_samples(self, monitor):
        """Test correlation with insufficient samples."""
        # Only one snapshot
        snapshot = ResourceSnapshot(
            timestamp=datetime.now(timezone.utc),
            metrics=ResourceMetrics(cpu_percent=50.0),
            queue_stats=QueueStatistics(queue_size=100)
        )
        monitor._snapshots.append(snapshot)

        correlations = await monitor.correlate_with_queue_performance()

        # Should return empty list
        assert len(correlations) == 0

    @pytest.mark.asyncio
    async def test_detect_bottleneck_cpu_critical(self, monitor):
        """Test CPU bottleneck detection - critical."""
        # Mock high CPU with low processing rate
        mock_metrics = ResourceMetrics(cpu_percent=95.0, memory_mb=500.0)
        mock_stats = QueueStatistics(queue_size=100, processing_rate=5.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        bottleneck = await monitor.detect_resource_bottleneck()

        assert bottleneck is not None
        assert bottleneck.resource_type == "CPU"
        assert bottleneck.severity == "critical"
        assert bottleneck.current_value == 95.0

    @pytest.mark.asyncio
    async def test_detect_bottleneck_cpu_warning(self, monitor):
        """Test CPU bottleneck detection - warning."""
        # Mock elevated CPU with queue backlog
        mock_metrics = ResourceMetrics(cpu_percent=75.0, memory_mb=500.0)
        mock_stats = QueueStatistics(queue_size=2000, processing_rate=15.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        bottleneck = await monitor.detect_resource_bottleneck()

        assert bottleneck is not None
        assert bottleneck.resource_type == "CPU"
        assert bottleneck.severity == "warning"

    @pytest.mark.asyncio
    async def test_detect_bottleneck_memory_critical(self, monitor):
        """Test memory bottleneck detection - critical."""
        # Mock high memory with queue backlog
        mock_metrics = ResourceMetrics(cpu_percent=50.0, memory_mb=2500.0)
        mock_stats = QueueStatistics(queue_size=3000, processing_rate=15.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        bottleneck = await monitor.detect_resource_bottleneck()

        assert bottleneck is not None
        assert bottleneck.resource_type == "Memory"
        assert bottleneck.severity == "critical"

    @pytest.mark.asyncio
    async def test_detect_bottleneck_memory_warning(self, monitor):
        """Test memory bottleneck detection - warning."""
        # Mock elevated memory with low processing rate
        mock_metrics = ResourceMetrics(cpu_percent=50.0, memory_mb=1500.0)
        mock_stats = QueueStatistics(queue_size=500, processing_rate=5.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        bottleneck = await monitor.detect_resource_bottleneck()

        assert bottleneck is not None
        assert bottleneck.resource_type == "Memory"
        assert bottleneck.severity == "warning"

    @pytest.mark.asyncio
    async def test_detect_bottleneck_connections(self, monitor):
        """Test database connection bottleneck detection."""
        # Mock connection pool near capacity
        mock_metrics = ResourceMetrics(
            cpu_percent=50.0,
            memory_mb=500.0,
            db_connections=18  # 90% of 20
        )
        mock_stats = QueueStatistics(queue_size=500, processing_rate=5.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)
        monitor.connection_pool.config.max_connections = 20

        bottleneck = await monitor.detect_resource_bottleneck()

        assert bottleneck is not None
        assert bottleneck.resource_type == "Database Connections"
        assert bottleneck.severity == "critical"

    @pytest.mark.asyncio
    async def test_detect_bottleneck_threads(self, monitor):
        """Test thread bottleneck detection."""
        # Mock high thread count with queue backlog
        mock_metrics = ResourceMetrics(
            cpu_percent=50.0,
            memory_mb=500.0,
            thread_count=150
        )
        mock_stats = QueueStatistics(queue_size=6000, processing_rate=15.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        bottleneck = await monitor.detect_resource_bottleneck()

        assert bottleneck is not None
        assert bottleneck.resource_type == "Threads"
        assert bottleneck.severity == "warning"

    @pytest.mark.asyncio
    async def test_detect_no_bottleneck(self, monitor):
        """Test when no bottleneck is detected."""
        # Mock healthy metrics
        mock_metrics = ResourceMetrics(cpu_percent=30.0, memory_mb=500.0)
        mock_stats = QueueStatistics(queue_size=50, processing_rate=20.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        bottleneck = await monitor.detect_resource_bottleneck()

        assert bottleneck is None

    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitor):
        """Test starting background monitoring."""
        result = await monitor.start_monitoring(interval_seconds=1)

        assert result is True
        assert monitor._monitoring_task is not None
        assert not monitor._monitoring_task.done()

        # Stop monitoring
        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_start_monitoring_already_running(self, monitor):
        """Test starting monitoring when already running."""
        await monitor.start_monitoring(interval_seconds=1)

        # Try to start again
        result = await monitor.start_monitoring(interval_seconds=1)

        assert result is False

        # Stop monitoring
        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitor):
        """Test stopping background monitoring."""
        await monitor.start_monitoring(interval_seconds=1)

        result = await monitor.stop_monitoring()

        assert result is True
        assert monitor._monitoring_task.done()

    @pytest.mark.asyncio
    async def test_stop_monitoring_not_running(self, monitor):
        """Test stopping monitoring when not running."""
        result = await monitor.stop_monitoring()

        assert result is False

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, monitor):
        """Test background monitoring loop."""
        # Mock methods
        mock_metrics = ResourceMetrics(cpu_percent=50.0, memory_mb=1024.0)
        mock_stats = QueueStatistics(queue_size=100, processing_rate=10.0)

        monitor.get_current_resources = AsyncMock(return_value=mock_metrics)
        monitor.stats_collector.get_current_statistics = AsyncMock(return_value=mock_stats)

        # Start monitoring
        await monitor.start_monitoring(interval_seconds=0.1)

        # Wait for a few iterations
        await asyncio.sleep(0.3)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Should have collected snapshots
        assert len(monitor._snapshots) > 0

    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, monitor):
        """Test monitoring loop handles errors gracefully."""
        # Mock method to raise exception
        monitor.get_current_resources = AsyncMock(side_effect=Exception("Test error"))

        # Start monitoring
        await monitor.start_monitoring(interval_seconds=0.1)

        # Wait a bit
        await asyncio.sleep(0.3)

        # Stop monitoring - should not crash
        await monitor.stop_monitoring()

        # Monitoring should have stopped gracefully
        assert monitor._monitoring_task.done()
