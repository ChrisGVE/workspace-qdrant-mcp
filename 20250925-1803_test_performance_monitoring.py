#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Performance Monitoring System

Tests all components with edge cases, error conditions, and boundary scenarios.
Ensures 90%+ test coverage with meaningful assertions for performance monitoring,
optimization, health reporting, and benchmarking systems.

Created: 2025-09-25T18:03:00+02:00
"""

import asyncio
import threading
import time
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import statistics

# Import the performance monitoring system
from performance_monitoring import (
    MetricsCollector,
    PerformanceOptimizer,
    HealthMonitor,
    PerformanceBenchmark,
    PerformanceMonitoringSystem,
    PerformanceMetric,
    HealthReport,
    OptimizationRecommendation,
    BenchmarkResult,
    MetricType,
    HealthStatus,
    OptimizationType
)


class TestMetricsCollector:
    """Test suite for MetricsCollector"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'buffer_size': 100,
            'collection_interval': 0.1,
            'enable_memory_profiling': False
        }
        self.collector = MetricsCollector(self.config)

    def teardown_method(self):
        """Cleanup after tests"""
        if self.collector.is_collecting:
            self.collector.stop_collection()

    def test_record_metric_basic(self):
        """Test basic metric recording"""
        timestamp = datetime.now()
        self.collector.record_metric("test.metric", 42.5, MetricType.GAUGE, timestamp)

        metrics = self.collector.get_metrics("test.metric")
        assert len(metrics) == 1
        assert metrics[0].name == "test.metric"
        assert metrics[0].value == 42.5
        assert metrics[0].metric_type == MetricType.GAUGE
        assert metrics[0].timestamp == timestamp

    def test_record_metric_with_tags_and_metadata(self):
        """Test metric recording with tags and metadata"""
        tags = {"component": "test", "version": "1.0"}
        metadata = {"source": "unit_test", "extra_info": "sample"}

        self.collector.record_metric(
            "test.tagged", 100.0, MetricType.COUNTER,
            tags=tags, metadata=metadata
        )

        metrics = self.collector.get_metrics("test.tagged")
        assert len(metrics) == 1
        assert metrics[0].tags == tags
        assert metrics[0].metadata == metadata

    def test_record_metric_thread_safety(self):
        """Test thread safety of metric recording"""
        num_threads = 10
        metrics_per_thread = 50
        threads = []
        errors = []

        def record_metrics(thread_id):
            try:
                for i in range(metrics_per_thread):
                    self.collector.record_metric(
                        f"thread.{thread_id}",
                        float(i),
                        MetricType.COUNTER,
                        tags={"thread_id": str(thread_id)}
                    )
            except Exception as e:
                errors.append(e)

        # Start threads
        for thread_id in range(num_threads):
            thread = threading.Thread(target=record_metrics, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no errors and correct count
        assert len(errors) == 0
        total_metrics = len(self.collector.get_metrics())
        assert total_metrics == num_threads * metrics_per_thread

    def test_buffer_size_limit(self):
        """Test that buffer respects size limit"""
        buffer_size = 10
        collector = MetricsCollector({'buffer_size': buffer_size})

        # Add more metrics than buffer size
        for i in range(buffer_size + 5):
            collector.record_metric(f"test.{i}", float(i), MetricType.GAUGE)

        # Should only keep the last buffer_size metrics
        all_metrics = collector.get_metrics()
        assert len(all_metrics) == buffer_size

        # Should contain the most recent metrics
        metric_values = [m.value for m in all_metrics]
        expected_values = list(range(5, buffer_size + 5))  # Last 10 values
        assert sorted(metric_values) == sorted(expected_values)

    def test_get_metrics_filtering_by_name(self):
        """Test metric retrieval filtering by name"""
        # Add different metrics
        self.collector.record_metric("cpu.usage", 50.0, MetricType.GAUGE)
        self.collector.record_metric("memory.usage", 80.0, MetricType.GAUGE)
        self.collector.record_metric("cpu.temperature", 65.0, MetricType.GAUGE)

        # Filter by name
        cpu_metrics = self.collector.get_metrics("cpu.usage")
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0].name == "cpu.usage"

        memory_metrics = self.collector.get_metrics("memory.usage")
        assert len(memory_metrics) == 1
        assert memory_metrics[0].name == "memory.usage"

    def test_get_metrics_filtering_by_time_range(self):
        """Test metric retrieval filtering by time range"""
        base_time = datetime.now()

        # Add metrics with different timestamps
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            self.collector.record_metric("test.time", float(i), MetricType.GAUGE, timestamp)

        # Filter by time range
        start_time = base_time + timedelta(minutes=1)
        end_time = base_time + timedelta(minutes=3)
        filtered_metrics = self.collector.get_metrics(time_range=(start_time, end_time))

        # Should get metrics at minutes 1, 2, 3
        assert len(filtered_metrics) == 3
        values = sorted([m.value for m in filtered_metrics])
        assert values == [1.0, 2.0, 3.0]

    def test_get_metric_summary_basic(self):
        """Test basic metric summary calculation"""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for value in values:
            self.collector.record_metric("test.summary", value, MetricType.GAUGE)

        summary = self.collector.get_metric_summary("test.summary")

        assert summary["count"] == 5
        assert summary["min"] == 10.0
        assert summary["max"] == 50.0
        assert summary["mean"] == 30.0
        assert summary["median"] == 30.0
        assert summary["latest"] == 50.0

    def test_get_metric_summary_empty(self):
        """Test metric summary with no metrics - edge case"""
        summary = self.collector.get_metric_summary("nonexistent.metric")

        assert "error" in summary
        assert summary["error"] == "No metrics found"
        assert summary["count"] == 0

    def test_get_metric_summary_single_value(self):
        """Test metric summary with single value - edge case"""
        self.collector.record_metric("test.single", 42.0, MetricType.GAUGE)

        summary = self.collector.get_metric_summary("test.single")

        assert summary["count"] == 1
        assert summary["min"] == 42.0
        assert summary["max"] == 42.0
        assert summary["mean"] == 42.0
        assert summary["median"] == 42.0
        assert summary["std_dev"] == 0.0

    def test_get_metric_summary_with_percentiles(self):
        """Test metric summary with percentiles for large dataset"""
        # Add enough values to trigger percentile calculation
        values = list(range(1, 101))  # 1 to 100
        for value in values:
            self.collector.record_metric("test.percentiles", float(value), MetricType.GAUGE)

        summary = self.collector.get_metric_summary("test.percentiles")

        assert "p50" in summary
        assert "p95" in summary
        assert "p99" in summary
        assert summary["p50"] == 50.5  # Median of 1-100
        assert summary["p95"] == 95.0
        assert summary["p99"] == 99.0

    @patch('psutil.Process')
    def test_collect_system_metrics_error_handling(self, mock_process):
        """Test error handling in system metrics collection"""
        # Mock process to raise exception
        mock_process.return_value.memory_info.side_effect = Exception("Process error")

        collector = MetricsCollector()

        # Should not raise exception
        collector._collect_system_metrics()

        # Should have logged error but not crashed

    def test_start_stop_collection(self):
        """Test starting and stopping metrics collection"""
        assert not self.collector.is_collecting

        # Start collection
        self.collector.start_collection()
        assert self.collector.is_collecting
        assert self.collector._collection_thread is not None

        # Let it run briefly
        time.sleep(0.2)

        # Stop collection
        self.collector.stop_collection()
        assert not self.collector.is_collecting

        # Should have collected some metrics
        metrics = self.collector.get_metrics()
        assert len(metrics) > 0

    def test_collection_interval_respected(self):
        """Test that collection respects configured interval"""
        collector = MetricsCollector({'collection_interval': 0.05})  # 50ms

        start_time = time.time()
        collector.start_collection()
        time.sleep(0.2)  # Run for 200ms
        collector.stop_collection()
        end_time = time.time()

        # Should have collected metrics at ~50ms intervals
        metrics = collector.get_metrics()
        duration = end_time - start_time

        # Expect roughly 4 collection cycles in 200ms with 50ms intervals
        expected_collections = int(duration / 0.05)
        # Allow some variance due to timing precision
        assert len(metrics) >= expected_collections - 2

    def test_register_aggregator(self):
        """Test custom metric aggregator registration and execution"""
        aggregated_values = []

        def test_aggregator(metric: PerformanceMetric):
            aggregated_values.append(metric.value * 2)  # Double the value

        self.collector.register_aggregator("test.aggregated", test_aggregator)

        # Record metrics
        for i in range(3):
            self.collector.record_metric("test.aggregated", float(i), MetricType.GAUGE)

        # Check that aggregator was called
        assert len(aggregated_values) == 3
        assert aggregated_values == [0.0, 2.0, 4.0]  # Original values doubled

    def test_aggregator_error_handling(self):
        """Test error handling in custom aggregators"""
        def failing_aggregator(metric: PerformanceMetric):
            raise Exception("Aggregator failed")

        self.collector.register_aggregator("test.failing", failing_aggregator)

        # Should not raise exception when recording metric
        self.collector.record_metric("test.failing", 42.0, MetricType.GAUGE)

        # Metric should still be recorded despite aggregator failure
        metrics = self.collector.get_metrics("test.failing")
        assert len(metrics) == 1


class TestPerformanceOptimizer:
    """Test suite for PerformanceOptimizer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.metrics_collector = MetricsCollector({'buffer_size': 1000})
        self.config = {
            'memory_threshold_mb': 100,
            'cpu_threshold_percent': 70,
            'response_time_threshold_ms': 50
        }
        self.optimizer = PerformanceOptimizer(self.metrics_collector, self.config)

    def add_sample_metrics(self):
        """Add sample metrics for testing"""
        base_time = datetime.now() - timedelta(minutes=10)

        # Memory metrics
        memory_values = [50, 75, 120, 140, 130]  # Exceeds threshold
        for i, value in enumerate(memory_values):
            timestamp = base_time + timedelta(minutes=i)
            self.metrics_collector.record_metric("memory.rss", value, MetricType.GAUGE, timestamp)

        # CPU metrics
        cpu_values = [30, 50, 80, 85, 75]  # Exceeds threshold
        for i, value in enumerate(cpu_values):
            timestamp = base_time + timedelta(minutes=i)
            self.metrics_collector.record_metric("cpu.usage", value, MetricType.GAUGE, timestamp)

        # Response time metrics
        response_values = [20, 40, 60, 80, 100]  # Some exceed threshold
        for i, value in enumerate(response_values):
            timestamp = base_time + timedelta(minutes=i)
            self.metrics_collector.record_metric("api.response_time", value, MetricType.TIMER, timestamp)

    def test_analyze_performance_with_issues(self):
        """Test performance analysis with various issues"""
        self.add_sample_metrics()

        recommendations = self.optimizer.analyze_performance()

        # Should generate recommendations for high memory, CPU, and response times
        assert len(recommendations) > 0

        # Check for memory recommendations
        memory_recs = [r for r in recommendations if r.optimization_type == OptimizationType.MEMORY]
        assert len(memory_recs) > 0

        # Check for CPU recommendations
        cpu_recs = [r for r in recommendations if r.optimization_type == OptimizationType.CPU]
        assert len(cpu_recs) > 0

        # Verify recommendations are sorted by priority
        priorities = [r.priority for r in recommendations]
        assert priorities == sorted(priorities, reverse=True)

    def test_analyze_performance_no_issues(self):
        """Test performance analysis with no issues"""
        base_time = datetime.now() - timedelta(minutes=10)

        # Add metrics within thresholds
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            self.metrics_collector.record_metric("memory.rss", 50.0, MetricType.GAUGE, timestamp)
            self.metrics_collector.record_metric("cpu.usage", 30.0, MetricType.GAUGE, timestamp)

        recommendations = self.optimizer.analyze_performance()

        # Should generate few or no recommendations
        assert len(recommendations) == 0 or all(r.priority <= 3 for r in recommendations)

    def test_analyze_memory_usage_high_current(self):
        """Test memory analysis with high current usage"""
        # Add metrics with high current memory
        self.metrics_collector.record_metric("memory.rss", 150.0, MetricType.GAUGE)  # Above threshold

        recommendations = self.optimizer._analyze_memory_usage()

        memory_high_recs = [r for r in recommendations if "exceeding threshold" in r.description]
        assert len(memory_high_recs) > 0
        assert memory_high_recs[0].priority >= 6

    def test_analyze_memory_usage_growth_trend(self):
        """Test memory analysis with growth trend"""
        base_time = datetime.now() - timedelta(minutes=10)

        # Add metrics showing growth trend
        memory_values = [50, 60, 70, 80, 90]  # Increasing trend
        for i, value in enumerate(memory_values):
            timestamp = base_time + timedelta(minutes=i)
            self.metrics_collector.record_metric("memory.rss", value, MetricType.GAUGE, timestamp)

        recommendations = self.optimizer._analyze_memory_usage()

        growth_recs = [r for r in recommendations if "trend" in r.description.lower()]
        assert len(growth_recs) > 0

    def test_analyze_cpu_usage_high_current(self):
        """Test CPU analysis with high current usage"""
        self.metrics_collector.record_metric("cpu.usage", 85.0, MetricType.GAUGE)  # Above threshold

        recommendations = self.optimizer._analyze_cpu_usage()

        cpu_high_recs = [r for r in recommendations if "exceeding threshold" in r.description]
        assert len(cpu_high_recs) > 0
        assert cpu_high_recs[0].priority >= 5

    def test_analyze_cpu_usage_sustained_high(self):
        """Test CPU analysis with sustained high usage"""
        base_time = datetime.now() - timedelta(minutes=10)

        # Add metrics showing sustained high CPU
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            self.metrics_collector.record_metric("cpu.usage", 65.0, MetricType.GAUGE, timestamp)  # Above 70% of threshold

        recommendations = self.optimizer._analyze_cpu_usage()

        sustained_recs = [r for r in recommendations if "sustained" in r.description.lower()]
        assert len(sustained_recs) > 0

    def test_analyze_response_times_high_p95(self):
        """Test response time analysis with high P95"""
        base_time = datetime.now() - timedelta(minutes=10)

        # Add response time metrics with some high values
        response_values = [20] * 18 + [100] * 2  # P95 will be 100ms (above 50ms threshold)
        for i, value in enumerate(response_values):
            timestamp = base_time + timedelta(seconds=i*30)
            self.metrics_collector.record_metric("api.response_time", value, MetricType.TIMER, timestamp)

        recommendations = self.optimizer._analyze_response_times()

        response_recs = [r for r in recommendations if "response time" in r.description.lower()]
        assert len(response_recs) > 0

    def test_analyze_io_performance_high_activity(self):
        """Test I/O analysis with high activity"""
        base_time = datetime.now() - timedelta(minutes=10)

        # Add I/O metrics showing high activity
        start_bytes = 100 * 1024 * 1024  # 100MB
        end_bytes = 250 * 1024 * 1024    # 250MB (150MB transferred)

        self.metrics_collector.record_metric("io.read_bytes", start_bytes, MetricType.COUNTER, base_time)
        self.metrics_collector.record_metric("io.read_bytes", end_bytes, MetricType.COUNTER, base_time + timedelta(minutes=5))

        recommendations = self.optimizer._analyze_io_performance()

        io_recs = [r for r in recommendations if r.optimization_type == OptimizationType.IO]
        assert len(io_recs) > 0

    def test_apply_optimization(self):
        """Test applying optimization recommendations"""
        recommendation = OptimizationRecommendation(
            recommendation_id="test_opt_001",
            optimization_type=OptimizationType.CONFIGURATION,
            description="Test optimization",
            impact_estimate=10.0,
            implementation_difficulty=3,
            estimated_benefit="Test benefit",
            code_changes_required=["change1", "change2"],
            configuration_changes={"setting1": "value1", "setting2": "value2"},
            priority=5
        )

        result = self.optimizer.apply_optimization(recommendation)

        assert result is True
        assert recommendation.recommendation_id in self.optimizer.applied_optimizations

        # Check that configuration was updated
        assert self.optimizer.config["setting1"] == "value1"
        assert self.optimizer.config["setting2"] == "value2"

    def test_get_optimization_history(self):
        """Test getting optimization history"""
        # Add some recommendations to history
        self.add_sample_metrics()
        recommendations = self.optimizer.analyze_performance()

        history = self.optimizer.get_optimization_history()

        assert len(history) == len(recommendations)
        assert all(isinstance(r, OptimizationRecommendation) for r in history)

    def test_error_handling_in_analysis(self):
        """Test error handling during performance analysis"""
        # Mock metrics collector to raise exception
        with patch.object(self.optimizer.metrics_collector, 'get_metric_summary', side_effect=Exception("Test error")):
            recommendations = self.optimizer.analyze_performance()

            # Should not crash and return empty recommendations
            assert isinstance(recommendations, list)


class TestHealthMonitor:
    """Test suite for HealthMonitor"""

    def setup_method(self):
        """Setup test fixtures"""
        self.metrics_collector = MetricsCollector({'buffer_size': 1000})
        self.config = {
            'memory_warning_mb': 100,
            'memory_critical_mb': 200,
            'cpu_warning_percent': 60,
            'cpu_critical_percent': 80,
            'response_warning_ms': 100,
            'response_critical_ms': 500
        }
        self.health_monitor = HealthMonitor(self.metrics_collector, self.config)

    def add_healthy_metrics(self):
        """Add metrics indicating healthy system"""
        self.metrics_collector.record_metric("memory.rss", 50.0, MetricType.GAUGE)
        self.metrics_collector.record_metric("cpu.usage", 30.0, MetricType.GAUGE)
        self.metrics_collector.record_metric("uptime", 3600.0, MetricType.GAUGE)

    def add_warning_metrics(self):
        """Add metrics indicating warning conditions"""
        self.metrics_collector.record_metric("memory.rss", 150.0, MetricType.GAUGE)  # Warning level
        self.metrics_collector.record_metric("cpu.usage", 70.0, MetricType.GAUGE)   # Warning level

    def add_critical_metrics(self):
        """Add metrics indicating critical conditions"""
        self.metrics_collector.record_metric("memory.rss", 250.0, MetricType.GAUGE)  # Critical level
        self.metrics_collector.record_metric("cpu.usage", 90.0, MetricType.GAUGE)   # Critical level

    def test_generate_health_report_healthy(self):
        """Test health report generation with healthy system"""
        self.add_healthy_metrics()

        report = self.health_monitor.generate_health_report()

        assert report.overall_status == HealthStatus.HEALTHY
        assert report.component_health["memory"] == HealthStatus.HEALTHY
        assert report.component_health["cpu"] == HealthStatus.HEALTHY
        assert len(report.alerts) == 0
        assert report.uptime > 0

    def test_generate_health_report_warning(self):
        """Test health report generation with warning conditions"""
        self.add_warning_metrics()

        report = self.health_monitor.generate_health_report()

        assert report.overall_status == HealthStatus.WARNING
        assert len(report.alerts) > 0
        assert len(report.recommendations) > 0
        assert any("Warning" in alert for alert in report.alerts)

    def test_generate_health_report_critical(self):
        """Test health report generation with critical conditions"""
        self.add_critical_metrics()

        report = self.health_monitor.generate_health_report()

        assert report.overall_status == HealthStatus.CRITICAL
        assert len(report.alerts) > 0
        assert any("Critical" in alert for alert in report.alerts)
        assert any("Immediate" in rec for rec in report.recommendations)

    def test_assess_memory_health_levels(self):
        """Test memory health assessment at different levels"""
        # Test healthy
        self.metrics_collector.record_metric("memory.rss", 50.0, MetricType.GAUGE)
        status, alerts, recs, metrics = self.health_monitor._assess_memory_health()
        assert status == HealthStatus.HEALTHY
        assert len(alerts) == 0

        # Test warning
        self.metrics_collector.record_metric("memory.rss", 150.0, MetricType.GAUGE)
        status, alerts, recs, metrics = self.health_monitor._assess_memory_health()
        assert status == HealthStatus.WARNING
        assert len(alerts) > 0

        # Test critical
        self.metrics_collector.record_metric("memory.rss", 250.0, MetricType.GAUGE)
        status, alerts, recs, metrics = self.health_monitor._assess_memory_health()
        assert status == HealthStatus.CRITICAL
        assert len(alerts) > 0

    def test_assess_memory_health_growth_trend(self):
        """Test memory health assessment with growth trend"""
        base_time = datetime.now() - timedelta(minutes=10)

        # Add metrics showing growth trend
        memory_values = [50, 60, 70, 80, 120]  # Growing trend, last value high
        for i, value in enumerate(memory_values):
            timestamp = base_time + timedelta(minutes=i)
            self.metrics_collector.record_metric("memory.rss", value, MetricType.GAUGE, timestamp)

        status, alerts, recs, metrics = self.health_monitor._assess_memory_health()

        # Should detect warning due to growth trend
        assert status in [HealthStatus.WARNING, HealthStatus.CRITICAL]
        trend_alerts = [a for a in alerts if "trend" in a.lower()]
        assert len(trend_alerts) > 0

    def test_assess_cpu_health_levels(self):
        """Test CPU health assessment at different levels"""
        # Test healthy
        self.metrics_collector.record_metric("cpu.usage", 30.0, MetricType.GAUGE)
        status, alerts, recs, metrics = self.health_monitor._assess_cpu_health()
        assert status == HealthStatus.HEALTHY

        # Test warning
        self.metrics_collector.record_metric("cpu.usage", 70.0, MetricType.GAUGE)
        status, alerts, recs, metrics = self.health_monitor._assess_cpu_health()
        assert status == HealthStatus.WARNING

        # Test critical
        self.metrics_collector.record_metric("cpu.usage", 90.0, MetricType.GAUGE)
        status, alerts, recs, metrics = self.health_monitor._assess_cpu_health()
        assert status == HealthStatus.CRITICAL

    def test_assess_cpu_health_sustained_high(self):
        """Test CPU health assessment with sustained high usage"""
        base_time = datetime.now() - timedelta(minutes=10)

        # Add sustained high CPU metrics
        for i in range(5):
            timestamp = base_time + timedelta(minutes=i)
            self.metrics_collector.record_metric("cpu.usage", 65.0, MetricType.GAUGE, timestamp)

        status, alerts, recs, metrics = self.health_monitor._assess_cpu_health()

        # Should detect warning due to sustained high usage
        assert status in [HealthStatus.WARNING, HealthStatus.CRITICAL]
        sustained_alerts = [a for a in alerts if "sustained" in a.lower()]
        assert len(sustained_alerts) > 0

    def test_assess_response_health_with_metrics(self):
        """Test response time health assessment"""
        base_time = datetime.now() - timedelta(minutes=5)

        # Add response time metrics
        response_values = [50] * 18 + [600] * 2  # P95 will be high
        for i, value in enumerate(response_values):
            timestamp = base_time + timedelta(seconds=i*15)
            self.metrics_collector.record_metric("api.response_time", value, MetricType.TIMER, timestamp)

        status, alerts, recs, metrics = self.health_monitor._assess_response_health()

        assert status in [HealthStatus.WARNING, HealthStatus.CRITICAL]
        assert len(alerts) > 0

    def test_assess_response_health_no_metrics(self):
        """Test response time health assessment with no metrics"""
        status, alerts, recs, metrics = self.health_monitor._assess_response_health()

        assert status == HealthStatus.UNKNOWN
        assert "No response time" in alerts[0]

    def test_get_health_trends_with_data(self):
        """Test health trends analysis with historical data"""
        # Add some health reports to history
        for i in range(5):
            self.add_healthy_metrics()
            report = self.health_monitor.generate_health_report()
            # Modify timestamp to simulate time progression
            report.timestamp = datetime.now() - timedelta(hours=i)
            self.health_monitor.health_history.append(report)

        trends = self.health_monitor.get_health_trends(hours=6)

        assert "time_period_hours" in trends
        assert trends["time_period_hours"] == 6
        assert trends["total_reports"] == 5
        assert "status_distribution" in trends
        assert trends["status_distribution"]["healthy"] == 5

    def test_get_health_trends_no_data(self):
        """Test health trends analysis with no data"""
        trends = self.health_monitor.get_health_trends(hours=24)

        assert "error" in trends
        assert "No health data available" in trends["error"]

    def test_health_report_error_handling(self):
        """Test error handling during health report generation"""
        # Mock metrics collector to raise exception
        with patch.object(self.health_monitor, '_assess_memory_health', side_effect=Exception("Test error")):
            report = self.health_monitor.generate_health_report()

            assert report.overall_status == HealthStatus.UNKNOWN
            assert len(report.alerts) > 0
            assert "Health monitoring error" in report.alerts[0]


class TestPerformanceBenchmark:
    """Test suite for PerformanceBenchmark"""

    def setup_method(self):
        """Setup test fixtures"""
        self.metrics_collector = MetricsCollector()
        self.benchmark = PerformanceBenchmark(self.metrics_collector)

    def simple_sync_function(self, delay: float = 0.001):
        """Simple synchronous function for testing"""
        time.sleep(delay)
        return "completed"

    async def simple_async_function(self, delay: float = 0.001):
        """Simple asynchronous function for testing"""
        await asyncio.sleep(delay)
        return "completed"

    def failing_function(self):
        """Function that always fails for error testing"""
        raise Exception("Test failure")

    @pytest.mark.asyncio
    async def test_run_benchmark_sync_function(self):
        """Test benchmark with synchronous function"""
        result = await self.benchmark.run_benchmark(
            "sync_test", self.simple_sync_function, iterations=10, delay=0.001
        )

        assert result.test_name == "sync_test"
        assert result.duration > 0
        assert result.throughput > 0
        assert result.success_rate == 1.0
        assert result.error_count == 0
        assert result.metadata["iterations"] == 10
        assert result.metadata["successful_runs"] == 10

    @pytest.mark.asyncio
    async def test_run_benchmark_async_function(self):
        """Test benchmark with asynchronous function"""
        result = await self.benchmark.run_benchmark(
            "async_test", self.simple_async_function, iterations=5, delay=0.001
        )

        assert result.test_name == "async_test"
        assert result.success_rate == 1.0
        assert result.error_count == 0

    @pytest.mark.asyncio
    async def test_run_benchmark_with_failures(self):
        """Test benchmark with failing function"""
        result = await self.benchmark.run_benchmark(
            "failing_test", self.failing_function, iterations=5
        )

        assert result.test_name == "failing_test"
        assert result.success_rate == 0.0
        assert result.error_count == 5
        assert result.throughput == 0.0

    @pytest.mark.asyncio
    async def test_run_benchmark_mixed_success_failure(self):
        """Test benchmark with mixed success and failure"""
        call_count = 0

        def mixed_function():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:  # Fail every other call
                raise Exception("Intermittent failure")
            return "success"

        result = await self.benchmark.run_benchmark(
            "mixed_test", mixed_function, iterations=10
        )

        assert result.test_name == "mixed_test"
        assert result.success_rate == 0.5  # 50% success rate
        assert result.error_count == 5

    def test_get_benchmark_history_all(self):
        """Test getting all benchmark history"""
        # Add some results manually
        for i in range(3):
            result = BenchmarkResult(
                test_name=f"test_{i}",
                timestamp=datetime.now(),
                duration=1.0,
                throughput=10.0,
                memory_usage={},
                cpu_usage=50.0,
                success_rate=1.0,
                error_count=0,
                metadata={}
            )
            self.benchmark.benchmark_results.append(result)

        history = self.benchmark.get_benchmark_history()
        assert len(history) == 3

    def test_get_benchmark_history_filtered(self):
        """Test getting filtered benchmark history"""
        # Add results with different names
        for i in range(3):
            result = BenchmarkResult(
                test_name="test_a" if i % 2 == 0 else "test_b",
                timestamp=datetime.now(),
                duration=1.0,
                throughput=10.0,
                memory_usage={},
                cpu_usage=50.0,
                success_rate=1.0,
                error_count=0,
                metadata={}
            )
            self.benchmark.benchmark_results.append(result)

        # Filter for test_a only
        filtered_history = self.benchmark.get_benchmark_history("test_a")
        assert len(filtered_history) == 2
        assert all(r.test_name == "test_a" for r in filtered_history)

    def test_compare_benchmarks_improvement(self):
        """Test benchmark comparison showing improvement"""
        base_time = datetime.now()

        # Add baseline result (slower)
        baseline = BenchmarkResult(
            test_name="comparison_test",
            timestamp=base_time - timedelta(hours=1),
            duration=2.0,
            throughput=50.0,
            memory_usage={"delta_mb": 10.0},
            cpu_usage=60.0,
            success_rate=0.95,
            error_count=5,
            metadata={}
        )
        self.benchmark.benchmark_results.append(baseline)

        # Add current result (faster)
        current = BenchmarkResult(
            test_name="comparison_test",
            timestamp=base_time,
            duration=1.0,
            throughput=100.0,
            memory_usage={"delta_mb": 5.0},
            cpu_usage=40.0,
            success_rate=1.0,
            error_count=0,
            metadata={}
        )
        self.benchmark.benchmark_results.append(current)

        comparison = self.benchmark.compare_benchmarks("comparison_test")

        assert "changes" in comparison
        assert comparison["changes"]["throughput_percent"] == 100.0  # 100% improvement
        assert comparison["changes"]["duration_percent"] == -50.0    # 50% reduction
        assert comparison["performance_assessment"]["rating"] in ["good", "excellent"]

    def test_compare_benchmarks_regression(self):
        """Test benchmark comparison showing regression"""
        base_time = datetime.now()

        # Add baseline result (faster)
        baseline = BenchmarkResult(
            test_name="regression_test",
            timestamp=base_time - timedelta(hours=1),
            duration=1.0,
            throughput=100.0,
            memory_usage={"delta_mb": 5.0},
            cpu_usage=40.0,
            success_rate=1.0,
            error_count=0,
            metadata={}
        )
        self.benchmark.benchmark_results.append(baseline)

        # Add current result (slower)
        current = BenchmarkResult(
            test_name="regression_test",
            timestamp=base_time,
            duration=2.0,
            throughput=50.0,
            memory_usage={"delta_mb": 15.0},
            cpu_usage=70.0,
            success_rate=0.9,
            error_count=10,
            metadata={}
        )
        self.benchmark.benchmark_results.append(current)

        comparison = self.benchmark.compare_benchmarks("regression_test")

        assert comparison["changes"]["throughput_percent"] == -50.0  # 50% regression
        assert comparison["changes"]["duration_percent"] == 100.0    # 100% increase (worse)
        assert comparison["performance_assessment"]["rating"] == "needs_improvement"

    def test_compare_benchmarks_no_data(self):
        """Test benchmark comparison with no data"""
        comparison = self.benchmark.compare_benchmarks("nonexistent_test")

        assert "error" in comparison
        assert "No benchmark results found" in comparison["error"]

    def test_compare_benchmarks_insufficient_data(self):
        """Test benchmark comparison with insufficient data"""
        # Add only one result
        result = BenchmarkResult(
            test_name="single_test",
            timestamp=datetime.now(),
            duration=1.0,
            throughput=50.0,
            memory_usage={},
            cpu_usage=50.0,
            success_rate=1.0,
            error_count=0,
            metadata={}
        )
        self.benchmark.benchmark_results.append(result)

        comparison = self.benchmark.compare_benchmarks("single_test")

        assert "error" in comparison
        assert "Insufficient data" in comparison["error"]


class TestPerformanceMonitoringSystem:
    """Test suite for PerformanceMonitoringSystem integration"""

    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'auto_start': False,  # Don't auto-start for testing
            'metrics': {'collection_interval': 0.1},
            'optimizer': {'memory_threshold_mb': 50},
            'health': {'memory_warning_mb': 40}
        }
        self.system = PerformanceMonitoringSystem(self.config)

    def teardown_method(self):
        """Cleanup after tests"""
        self.system.shutdown()

    def test_initialization(self):
        """Test system initialization"""
        assert self.system.metrics_collector is not None
        assert self.system.optimizer is not None
        assert self.system.health_monitor is not None
        assert self.system.benchmark is not None

        # Should not auto-start collection due to config
        assert not self.system.metrics_collector.is_collecting

    def test_get_performance_dashboard(self):
        """Test performance dashboard generation"""
        # Add some metrics
        self.system.metrics_collector.record_metric("memory.rss", 100.0, MetricType.GAUGE)
        self.system.metrics_collector.record_metric("cpu.usage", 50.0, MetricType.GAUGE)
        self.system.metrics_collector.record_metric("uptime", 3600.0, MetricType.GAUGE)

        dashboard = self.system.get_performance_dashboard()

        assert "timestamp" in dashboard
        assert "health_report" in dashboard
        assert "optimization_recommendations" in dashboard
        assert "key_metrics" in dashboard
        assert "system_info" in dashboard

        # Verify health report structure
        health_report = dashboard["health_report"]
        assert "overall_status" in health_report
        assert "component_health" in health_report

        # Verify key metrics
        key_metrics = dashboard["key_metrics"]
        assert "memory" in key_metrics
        assert "cpu" in key_metrics
        assert "uptime" in key_metrics

    def test_context_manager(self):
        """Test context manager functionality"""
        config = {'auto_start': False}

        with PerformanceMonitoringSystem(config) as system:
            assert system.metrics_collector is not None
            # System should be operational

        # After context exit, system should be shutdown
        # (This is tested implicitly by no exceptions being raised)

    def test_shutdown(self):
        """Test system shutdown"""
        # Start collection first
        self.system.metrics_collector.start_collection()
        assert self.system.metrics_collector.is_collecting

        # Shutdown system
        self.system.shutdown()
        assert not self.system.metrics_collector.is_collecting

    def test_error_handling_in_dashboard(self):
        """Test error handling during dashboard generation"""
        # Mock health monitor to raise exception
        with patch.object(self.system.health_monitor, 'generate_health_report', side_effect=Exception("Test error")):
            dashboard = self.system.get_performance_dashboard()

            assert "error" in dashboard
            assert "Test error" in dashboard["error"]


class TestIntegrationScenarios:
    """Integration tests for complete system scenarios"""

    @pytest.mark.asyncio
    async def test_complete_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        config = {
            'auto_start': True,
            'metrics': {'collection_interval': 0.05, 'buffer_size': 100},
            'optimizer': {'memory_threshold_mb': 50, 'cpu_threshold_percent': 60},
            'health': {'memory_warning_mb': 40, 'cpu_warning_percent': 50}
        }

        with PerformanceMonitoringSystem(config) as system:
            # Let system collect metrics
            await asyncio.sleep(0.2)

            # Run a benchmark
            def cpu_intensive_task():
                result = sum(i**2 for i in range(1000))
                return result

            benchmark_result = await system.benchmark.run_benchmark(
                "cpu_test", cpu_intensive_task, iterations=10
            )

            assert benchmark_result.success_rate == 1.0

            # Generate dashboard
            dashboard = system.get_performance_dashboard()
            assert "health_report" in dashboard

            # Analyze performance
            recommendations = system.optimizer.analyze_performance()
            assert isinstance(recommendations, list)

    @pytest.mark.asyncio
    async def test_high_load_scenario(self):
        """Test system behavior under high load"""
        config = {
            'auto_start': True,
            'metrics': {'collection_interval': 0.01, 'buffer_size': 1000}
        }

        with PerformanceMonitoringSystem(config) as system:
            # Simulate high metric load
            start_time = time.time()
            metric_count = 0

            while time.time() - start_time < 0.5:  # Run for 0.5 seconds
                system.metrics_collector.record_metric(
                    f"load_test.{metric_count % 10}",
                    metric_count,
                    MetricType.COUNTER
                )
                metric_count += 1

            # System should handle high load without issues
            dashboard = system.get_performance_dashboard()
            assert "error" not in dashboard

            # Should have collected many metrics
            all_metrics = system.metrics_collector.get_metrics()
            assert len(all_metrics) > 100

    def test_error_recovery_scenarios(self):
        """Test error recovery in various scenarios"""
        config = {'auto_start': False}
        system = PerformanceMonitoringSystem(config)

        # Test metrics collection with invalid data
        system.metrics_collector.record_metric("test.invalid", float('inf'), MetricType.GAUGE)
        system.metrics_collector.record_metric("test.nan", float('nan'), MetricType.GAUGE)

        # Should not crash
        summary = system.metrics_collector.get_metric_summary("test.invalid")
        assert isinstance(summary, dict)

        # Test health monitoring with no metrics
        health_report = system.health_monitor.generate_health_report()
        assert health_report.overall_status in [HealthStatus.HEALTHY, HealthStatus.UNKNOWN]

        system.shutdown()

    def test_concurrent_access_safety(self):
        """Test thread safety under concurrent access"""
        config = {'auto_start': True, 'metrics': {'collection_interval': 0.01}}

        with PerformanceMonitoringSystem(config) as system:
            errors = []

            def record_metrics(thread_id):
                try:
                    for i in range(100):
                        system.metrics_collector.record_metric(
                            f"concurrent.{thread_id}",
                            float(i),
                            MetricType.GAUGE
                        )
                except Exception as e:
                    errors.append(e)

            # Start multiple threads
            threads = []
            for thread_id in range(5):
                thread = threading.Thread(target=record_metrics, args=(thread_id,))
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # Should have no errors
            assert len(errors) == 0

            # Should have recorded all metrics
            metrics = system.metrics_collector.get_metrics()
            assert len(metrics) >= 500  # 5 threads * 100 metrics each


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])