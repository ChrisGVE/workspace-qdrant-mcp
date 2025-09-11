"""
Tests for the performance monitoring and optimization system.

This module contains comprehensive tests for the performance metrics collection,
analytics, storage, and optimization recommendation components.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.workspace_qdrant_mcp.core.performance_metrics import (
    MetricType, PerformanceMetric, PerformanceMetricsCollector,
    OperationTrace, PerformanceLevel
)
from src.workspace_qdrant_mcp.core.performance_analytics import (
    PerformanceAnalyzer, OptimizationType, Priority
)
from src.workspace_qdrant_mcp.core.performance_storage import PerformanceStorage
from src.workspace_qdrant_mcp.core.performance_monitor import PerformanceMonitor


class TestPerformanceMetrics:
    """Test performance metrics collection functionality."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for testing."""
        return PerformanceMetricsCollector("test_project")
    
    @pytest.mark.asyncio
    async def test_metric_recording(self, metrics_collector):
        """Test basic metric recording."""
        await metrics_collector.record_metric(
            MetricType.SEARCH_LATENCY,
            150.5,
            "ms",
            context={"query": "test"}
        )
        
        metrics = await metrics_collector.buffer.get_metrics(
            metric_type=MetricType.SEARCH_LATENCY
        )
        
        assert len(metrics) == 1
        assert metrics[0].value == 150.5
        assert metrics[0].unit == "ms"
        assert metrics[0].context["query"] == "test"
    
    @pytest.mark.asyncio
    async def test_search_performance_recording(self, metrics_collector):
        """Test search performance metric recording."""
        await metrics_collector.record_search_performance(
            query="test query",
            result_count=25,
            latency_ms=120.0,
            relevance_score=0.85
        )
        
        # Check search latency metric
        latency_metrics = await metrics_collector.buffer.get_metrics(
            metric_type=MetricType.SEARCH_LATENCY
        )
        assert len(latency_metrics) == 1
        assert latency_metrics[0].value == 120.0
        
        # Check throughput metric
        throughput_metrics = await metrics_collector.buffer.get_metrics(
            metric_type=MetricType.SEARCH_THROUGHPUT
        )
        assert len(throughput_metrics) == 1
        assert throughput_metrics[0].value > 0
    
    @pytest.mark.asyncio
    async def test_operation_profiling(self, metrics_collector):
        """Test operation profiling context manager."""
        async with metrics_collector.profile_operation("file_processing") as trace:
            trace.add_metric(MetricType.FILE_PROCESSING_RATE, 45.0, "files/min")
            await asyncio.sleep(0.1)  # Simulate work
        
        # Check that operation was recorded
        recent_ops = await metrics_collector.get_recent_operations(limit=1)
        assert len(recent_ops) == 1
        assert recent_ops[0].operation_type == "file_processing"
        assert recent_ops[0].status == "completed"
        assert recent_ops[0].duration is not None
    
    @pytest.mark.asyncio
    async def test_metric_summary_calculation(self, metrics_collector):
        """Test metric summary statistics calculation."""
        # Record multiple metrics
        values = [100, 150, 200, 120, 180]
        for value in values:
            await metrics_collector.record_metric(
                MetricType.SEARCH_LATENCY,
                value,
                "ms"
            )
        
        summary = await metrics_collector.get_metric_summary(MetricType.SEARCH_LATENCY)
        
        assert summary is not None
        assert summary.count == 5
        assert summary.min_value == 100
        assert summary.max_value == 200
        assert abs(summary.mean_value - 150) < 0.1
        assert summary.performance_level in [PerformanceLevel.GOOD, PerformanceLevel.AVERAGE]


class TestPerformanceAnalytics:
    """Test performance analytics and optimization recommendations."""
    
    @pytest.fixture
    def mock_metrics_collector(self):
        """Create a mock metrics collector."""
        collector = MagicMock(spec=PerformanceMetricsCollector)
        collector.project_id = "test_project"
        return collector
    
    @pytest.fixture
    def analyzer(self, mock_metrics_collector):
        """Create a performance analyzer for testing."""
        return PerformanceAnalyzer(mock_metrics_collector)
    
    @pytest.mark.asyncio
    async def test_performance_analysis(self, analyzer, mock_metrics_collector):
        """Test performance analysis report generation."""
        from src.workspace_qdrant_mcp.core.performance_metrics import MetricSummary
        
        # Mock metric summaries
        mock_summary = MetricSummary(
            metric_type=MetricType.SEARCH_LATENCY,
            count=100,
            min_value=50,
            max_value=300,
            mean_value=150,
            median_value=140,
            std_dev=45,
            percentile_95=250,
            percentile_99=280,
            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
            performance_level=PerformanceLevel.AVERAGE,
            trend="stable"
        )
        
        mock_metrics_collector.get_metric_summary = AsyncMock(return_value=mock_summary)
        
        report = await analyzer.analyze_performance(time_range_hours=1)
        
        assert report.project_id == "test_project"
        assert 0 <= report.overall_performance_score <= 100
        assert len(report.insights) >= 0
        assert len(report.recommendations) >= 0
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, analyzer, mock_metrics_collector):
        """Test optimization recommendation generation."""
        from src.workspace_qdrant_mcp.core.performance_metrics import MetricSummary
        
        # Mock high latency scenario
        high_latency_summary = MetricSummary(
            metric_type=MetricType.SEARCH_LATENCY,
            count=50,
            min_value=200,
            max_value=600,
            mean_value=400,
            median_value=380,
            std_dev=80,
            percentile_95=550,
            percentile_99=580,
            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
            performance_level=PerformanceLevel.POOR,
            trend="degrading"
        )
        
        mock_metrics_collector.get_metric_summary = AsyncMock(return_value=high_latency_summary)
        
        report = await analyzer.analyze_performance()
        
        # Should generate search optimization recommendations
        search_recommendations = [
            rec for rec in report.recommendations
            if rec.optimization_type == OptimizationType.SEARCH_OPTIMIZATION
        ]
        
        assert len(search_recommendations) > 0
        assert search_recommendations[0].priority in [Priority.HIGH, Priority.CRITICAL]


class TestPerformanceStorage:
    """Test performance data storage and retrieval."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = PerformanceStorage("test_project", Path(temp_dir))
            yield storage
            storage.close()
    
    @pytest.mark.asyncio
    async def test_metric_storage_and_retrieval(self, temp_storage):
        """Test storing and retrieving metrics."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SEARCH_LATENCY,
            value=125.5,
            unit="ms",
            project_id="test_project",
            context={"query": "test"},
            tags=["search"]
        )
        
        await temp_storage.store_metric(metric)
        
        retrieved_metrics = await temp_storage.get_metrics(
            metric_type=MetricType.SEARCH_LATENCY
        )
        
        assert len(retrieved_metrics) == 1
        assert retrieved_metrics[0].value == 125.5
        assert retrieved_metrics[0].context["query"] == "test"
    
    @pytest.mark.asyncio
    async def test_batch_metric_storage(self, temp_storage):
        """Test batch metric storage for performance."""
        metrics = []
        for i in range(50):
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.CPU_USAGE,
                value=float(i),
                unit="%",
                project_id="test_project"
            )
            metrics.append(metric)
        
        await temp_storage.store_metrics_batch(metrics)
        
        retrieved_metrics = await temp_storage.get_metrics(
            metric_type=MetricType.CPU_USAGE
        )
        
        assert len(retrieved_metrics) == 50
        assert all(m.metric_type == MetricType.CPU_USAGE for m in retrieved_metrics)
    
    @pytest.mark.asyncio
    async def test_operation_trace_storage(self, temp_storage):
        """Test operation trace storage."""
        trace = OperationTrace(
            operation_id="test_op_001",
            operation_type="file_processing",
            start_time=datetime.now() - timedelta(seconds=30),
            end_time=datetime.now(),
            project_id="test_project",
            status="completed"
        )
        
        await temp_storage.store_operation_trace(trace)
        
        retrieved_traces = await temp_storage.get_operation_traces(
            operation_type="file_processing"
        )
        
        assert len(retrieved_traces) == 1
        assert retrieved_traces[0].operation_id == "test_op_001"
        assert retrieved_traces[0].status == "completed"
    
    @pytest.mark.asyncio
    async def test_data_cleanup(self, temp_storage):
        """Test automatic data cleanup functionality."""
        # Store old metric
        old_metric = PerformanceMetric(
            timestamp=datetime.now() - timedelta(days=8),
            metric_type=MetricType.MEMORY_USAGE,
            value=100.0,
            unit="MB",
            project_id="test_project"
        )
        
        await temp_storage.store_metric(old_metric)
        
        # Store recent metric
        recent_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.MEMORY_USAGE,
            value=200.0,
            unit="MB",
            project_id="test_project"
        )
        
        await temp_storage.store_metric(recent_metric)
        
        # Cleanup old data
        await temp_storage.cleanup_old_data()
        
        # Check that only recent metric remains
        metrics = await temp_storage.get_metrics(metric_type=MetricType.MEMORY_USAGE)
        assert len(metrics) == 1
        assert metrics[0].value == 200.0


class TestPerformanceMonitor:
    """Test integrated performance monitoring system."""
    
    @pytest.fixture
    def mock_storage(self):
        """Create mock storage."""
        storage = MagicMock(spec=PerformanceStorage)
        storage.store_metric = AsyncMock()
        storage.store_metrics_batch = AsyncMock()
        storage.store_operation_trace = AsyncMock()
        storage.store_performance_report = AsyncMock()
        storage.get_storage_stats = AsyncMock(return_value={
            "database_size_mb": 1.5,
            "total_size_mb": 2.0
        })
        return storage
    
    @pytest.fixture
    def performance_monitor(self, mock_storage):
        """Create performance monitor for testing."""
        monitor = PerformanceMonitor("test_project")
        monitor.storage = mock_storage
        return monitor
    
    @pytest.mark.asyncio
    async def test_performance_monitor_lifecycle(self, performance_monitor):
        """Test performance monitor start/stop lifecycle."""
        # Test start
        with patch('src.workspace_qdrant_mcp.core.performance_monitor.get_performance_storage') as mock_get_storage:
            mock_get_storage.return_value = performance_monitor.storage
            
            await performance_monitor.start()
            assert performance_monitor.running is True
            
            # Test stop
            await performance_monitor.stop()
            assert performance_monitor.running is False
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, performance_monitor):
        """Test performance alert generation."""
        alert_received = False
        
        def alert_callback(alert):
            nonlocal alert_received
            alert_received = True
        
        performance_monitor.add_alert_callback(alert_callback)
        
        # Create high latency metric that should trigger alert
        high_latency_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SEARCH_LATENCY,
            value=600.0,  # Above critical threshold
            unit="ms",
            project_id="test_project"
        )
        
        await performance_monitor._check_metric_alerts(high_latency_metric)
        
        # Should have generated an alert
        assert len(performance_monitor.active_alerts) > 0
        assert alert_received is True
    
    @pytest.mark.asyncio
    async def test_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        with patch.object(performance_monitor.analyzer, 'analyze_performance') as mock_analyze:
            from src.workspace_qdrant_mcp.core.performance_analytics import PerformanceReport
            
            mock_report = PerformanceReport(
                project_id="test_project",
                generated_at=datetime.now(),
                time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                overall_performance_score=85.0,
                performance_level=PerformanceLevel.GOOD
            )
            mock_analyze.return_value = mock_report
            
            summary = await performance_monitor.get_performance_summary()
            
            assert summary["project_id"] == "test_project"
            assert summary["performance_score"] == 85.0
            assert summary["performance_level"] == "good"
            assert "storage_stats" in summary


@pytest.mark.asyncio
async def test_performance_monitor_integration():
    """Test integration between all performance monitoring components."""
    from src.workspace_qdrant_mcp.core.performance_monitor import get_performance_monitor
    
    # Get performance monitor
    monitor = await get_performance_monitor("integration_test_project")
    
    try:
        # Record some performance data
        await monitor.record_search_performance(
            query="integration test",
            result_count=15,
            latency_ms=95.0
        )
        
        await monitor.record_file_processing(
            file_path="/test/file.txt",
            processing_time_seconds=2.5,
            file_size_bytes=1024,
            success=True
        )
        
        # Use operation profiler
        async with monitor.profile_operation("test_operation") as trace:
            await asyncio.sleep(0.05)  # Simulate work
        
        # Get performance summary
        summary = await monitor.get_performance_summary()
        assert summary["project_id"] == "integration_test_project"
        
        # Get recommendations
        recommendations = await monitor.get_optimization_recommendations()
        assert isinstance(recommendations, list)
        
    finally:
        # Cleanup
        await monitor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])