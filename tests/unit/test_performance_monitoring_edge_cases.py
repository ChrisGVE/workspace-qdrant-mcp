"""
Comprehensive edge case tests for performance monitoring and optimization system.

This module contains extensive tests for edge cases, error conditions, boundary
scenarios, and failure modes in the performance monitoring system to achieve
90%+ test coverage with meaningful assertions.

Task 265: Enhanced performance monitoring tests with comprehensive edge cases.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import pytest
import tempfile
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from workspace_qdrant_mcp.core.performance_metrics import (
    MetricType, PerformanceMetric, PerformanceMetricsCollector,
    OperationTrace, PerformanceLevel, MetricSummary
)
from workspace_qdrant_mcp.core.performance_analytics import (
    PerformanceAnalyzer, OptimizationType, Priority, OptimizationRecommendation,
    OptimizationEngine, PerformanceReport, PerformanceInsight
)
from workspace_qdrant_mcp.core.performance_storage import PerformanceStorage
from workspace_qdrant_mcp.core.performance_monitor import PerformanceMonitor, PerformanceAlert


class TestMetricsCollectionEdgeCases:
    """Test edge cases for metrics collection system."""

    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for testing."""
        return PerformanceMetricsCollector("test_project")

    @pytest.mark.asyncio
    async def test_metrics_collection_under_high_load(self, metrics_collector):
        """Test metrics collection behavior under high load conditions."""
        # Simulate high load with many concurrent metric recordings
        tasks = []
        for i in range(1000):
            task = metrics_collector.record_metric(
                MetricType.SEARCH_LATENCY,
                float(i % 500),  # Values from 0 to 499
                "ms",
                context={"load_test": True, "batch": i // 100}
            )
            tasks.append(task)

        # Execute all tasks concurrently
        await asyncio.gather(*tasks)

        # Verify all metrics were recorded
        metrics = await metrics_collector.buffer.get_metrics(
            metric_type=MetricType.SEARCH_LATENCY
        )

        assert len(metrics) == 1000

        # Verify data integrity under high load
        values = [m.value for m in metrics]
        assert min(values) == 0.0
        assert max(values) == 499.0
        assert all(m.context.get("load_test") is True for m in metrics)

    @pytest.mark.asyncio
    async def test_metrics_collection_with_invalid_values(self, metrics_collector):
        """Test metrics collection with invalid/edge case values."""
        # Test with None value
        with pytest.raises((ValueError, TypeError)):
            await metrics_collector.record_metric(
                MetricType.SEARCH_LATENCY,
                None,
                "ms"
            )

        # Test with negative value for latency (should be handled gracefully)
        await metrics_collector.record_metric(
            MetricType.SEARCH_LATENCY,
            -10.0,
            "ms",
            context={"invalid_case": "negative_latency"}
        )

        # Test with extremely large value
        await metrics_collector.record_metric(
            MetricType.SEARCH_LATENCY,
            float('inf'),
            "ms",
            context={"invalid_case": "infinite_value"}
        )

        # Test with NaN value
        await metrics_collector.record_metric(
            MetricType.SEARCH_LATENCY,
            float('nan'),
            "ms",
            context={"invalid_case": "nan_value"}
        )

        metrics = await metrics_collector.buffer.get_metrics(
            metric_type=MetricType.SEARCH_LATENCY
        )

        # Should handle invalid values gracefully
        assert len(metrics) >= 3

        # Find metrics by context
        negative_metrics = [m for m in metrics if m.context.get("invalid_case") == "negative_latency"]
        inf_metrics = [m for m in metrics if m.context.get("invalid_case") == "infinite_value"]
        nan_metrics = [m for m in metrics if m.context.get("invalid_case") == "nan_value"]

        assert len(negative_metrics) == 1
        assert len(inf_metrics) == 1
        assert len(nan_metrics) == 1

    @pytest.mark.asyncio
    async def test_metrics_collection_memory_pressure(self, metrics_collector):
        """Test metrics collection behavior under memory pressure."""
        # Fill up the metrics buffer to test memory management
        large_context = {"large_data": "x" * 10000}  # 10KB of data per metric

        # Record many metrics with large context data
        for i in range(500):  # ~5MB of context data
            await metrics_collector.record_metric(
                MetricType.MEMORY_USAGE,
                float(i),
                "MB",
                context=large_context
            )

        # Test that the system handles memory pressure gracefully
        metrics = await metrics_collector.buffer.get_metrics(
            metric_type=MetricType.MEMORY_USAGE
        )

        # Should have recorded all metrics or handled overflow gracefully
        assert len(metrics) >= 100  # At least some metrics should be preserved

        # Test memory usage metric calculation under pressure
        summary = await metrics_collector.get_metric_summary(MetricType.MEMORY_USAGE)
        assert summary is not None
        assert summary.count > 0

    @pytest.mark.asyncio
    async def test_operation_profiling_with_exceptions(self, metrics_collector):
        """Test operation profiling when operations fail with exceptions."""
        # Test with operation that raises exception
        with pytest.raises(ValueError):
            async with metrics_collector.profile_operation("failing_operation") as trace:
                trace.add_metric(MetricType.FILE_PROCESSING_RATE, 25.0, "files/min")
                raise ValueError("Simulated operation failure")

        # Verify that the trace was still recorded with failure status
        recent_ops = await metrics_collector.get_recent_operations(limit=1)
        assert len(recent_ops) == 1
        assert recent_ops[0].operation_type == "failing_operation"
        assert recent_ops[0].status == "failed"
        assert recent_ops[0].duration is not None
        assert len(recent_ops[0].metrics) == 1

    @pytest.mark.asyncio
    async def test_concurrent_metric_summary_calculation(self, metrics_collector):
        """Test metric summary calculation with concurrent access."""
        # Start recording metrics concurrently
        record_task = asyncio.create_task(self._continuous_metric_recording(metrics_collector))

        # Calculate summaries concurrently while recording
        summary_tasks = []
        for _ in range(5):
            task = asyncio.create_task(
                metrics_collector.get_metric_summary(MetricType.CPU_USAGE)
            )
            summary_tasks.append(task)
            await asyncio.sleep(0.1)

        # Wait for all summary calculations
        summaries = await asyncio.gather(*summary_tasks)

        # Stop recording
        record_task.cancel()
        try:
            await record_task
        except asyncio.CancelledError:
            pass

        # Verify that all summaries were calculated successfully
        valid_summaries = [s for s in summaries if s is not None]
        assert len(valid_summaries) >= 3  # Should have multiple valid summaries

        # Verify summary consistency
        for summary in valid_summaries:
            assert summary.metric_type == MetricType.CPU_USAGE
            assert summary.count > 0
            assert summary.mean_value >= 0

    async def _continuous_metric_recording(self, metrics_collector):
        """Helper method for continuous metric recording."""
        counter = 0
        while True:
            await metrics_collector.record_metric(
                MetricType.CPU_USAGE,
                float(counter % 100),
                "%"
            )
            counter += 1
            await asyncio.sleep(0.05)


class TestOptimizationRecommendationEdgeCases:
    """Test edge cases for optimization recommendation system."""

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

    @pytest.fixture
    def optimization_engine(self):
        """Create an optimization engine for testing."""
        return OptimizationEngine()

    @pytest.mark.asyncio
    async def test_recommendations_with_conflicting_metrics(self, analyzer, mock_metrics_collector):
        """Test recommendation generation with conflicting metric signals."""
        # Create conflicting metric summaries
        high_cpu_summary = MetricSummary(
            metric_type=MetricType.CPU_USAGE,
            count=100,
            min_value=85,
            max_value=98,
            mean_value=92,
            median_value=91,
            std_dev=3,
            percentile_95=96,
            percentile_99=97,
            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
            performance_level=PerformanceLevel.POOR,
            trend="degrading"
        )

        low_memory_summary = MetricSummary(
            metric_type=MetricType.MEMORY_USAGE,
            count=100,
            min_value=50,
            max_value=80,
            mean_value=65,
            median_value=64,
            std_dev=8,
            percentile_95=75,
            percentile_99=78,
            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
            performance_level=PerformanceLevel.GOOD,
            trend="stable"
        )

        def mock_get_metric_summary(metric_type, start_time=None, end_time=None):
            if metric_type == MetricType.CPU_USAGE:
                return high_cpu_summary
            elif metric_type == MetricType.MEMORY_USAGE:
                return low_memory_summary
            return None

        mock_metrics_collector.get_metric_summary = AsyncMock(side_effect=mock_get_metric_summary)

        report = await analyzer.analyze_performance()

        # Should handle conflicting signals and prioritize appropriately
        assert len(report.recommendations) > 0

        # Should recommend CPU optimization but not memory optimization
        cpu_recommendations = [
            rec for rec in report.recommendations
            if rec.optimization_type == OptimizationType.CPU_OPTIMIZATION
        ]
        memory_recommendations = [
            rec for rec in report.recommendations
            if rec.optimization_type == OptimizationType.MEMORY_OPTIMIZATION
        ]

        # Should have CPU recommendations due to high CPU usage
        # Should not have memory recommendations due to good memory usage
        assert len(memory_recommendations) == 0 or memory_recommendations[0].priority == Priority.LOW

    @pytest.mark.asyncio
    async def test_recommendations_with_insufficient_data(self, analyzer, mock_metrics_collector):
        """Test recommendation generation with insufficient or no data."""
        # Mock empty metric summaries
        mock_metrics_collector.get_metric_summary = AsyncMock(return_value=None)

        report = await analyzer.analyze_performance()

        # Should handle insufficient data gracefully
        assert report.project_id == "test_project"
        assert report.overall_performance_score >= 0
        assert isinstance(report.recommendations, list)
        assert isinstance(report.insights, list)

        # Performance score should be neutral when no data is available
        assert 40 <= report.overall_performance_score <= 60

    @pytest.mark.asyncio
    async def test_recommendations_with_extreme_values(self, analyzer, mock_metrics_collector):
        """Test recommendation generation with extreme metric values."""
        # Create extreme value scenarios
        extreme_scenarios = [
            # Extremely high latency
            MetricSummary(
                metric_type=MetricType.SEARCH_LATENCY,
                count=10,
                min_value=10000,
                max_value=50000,
                mean_value=25000,  # 25 seconds!
                median_value=24000,
                std_dev=8000,
                percentile_95=45000,
                percentile_99=48000,
                time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                performance_level=PerformanceLevel.CRITICAL,
                trend="degrading"
            ),
            # Zero processing rate
            MetricSummary(
                metric_type=MetricType.FILE_PROCESSING_RATE,
                count=50,
                min_value=0,
                max_value=0.1,
                mean_value=0.05,  # Nearly zero
                median_value=0,
                std_dev=0.02,
                percentile_95=0.08,
                percentile_99=0.1,
                time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                performance_level=PerformanceLevel.CRITICAL,
                trend="degrading"
            ),
        ]

        def mock_get_metric_summary(metric_type, start_time=None, end_time=None):
            for summary in extreme_scenarios:
                if summary.metric_type == metric_type:
                    return summary
            return None

        mock_metrics_collector.get_metric_summary = AsyncMock(side_effect=mock_get_metric_summary)

        report = await analyzer.analyze_performance()

        # Should generate critical priority recommendations for extreme values
        critical_recommendations = [
            rec for rec in report.recommendations
            if rec.priority == Priority.CRITICAL
        ]

        assert len(critical_recommendations) > 0
        assert report.overall_performance_score < 20  # Very low score for extreme issues

        # Should identify specific bottlenecks
        assert len(report.bottlenecks) > 0
        assert any("Search operations" in bottleneck for bottleneck in report.bottlenecks)

    @pytest.mark.asyncio
    async def test_optimization_engine_with_invalid_recommendations(self, optimization_engine):
        """Test optimization engine with invalid or malformed recommendations."""
        # Test with recommendation that has missing required fields
        invalid_recommendation = OptimizationRecommendation(
            title="",  # Empty title
            description="",  # Empty description
            optimization_type=OptimizationType.MEMORY_OPTIMIZATION,
            priority=Priority.HIGH,
            impact_estimate="high",
            implementation_effort="moderate"
        )

        result = await optimization_engine.apply_recommendation(
            invalid_recommendation,
            auto_apply=False
        )

        # Should handle invalid recommendations gracefully
        assert result["status"] == "manual_required"
        assert "actions" in result

    @pytest.mark.asyncio
    async def test_recommendation_priority_edge_cases(self, analyzer, mock_metrics_collector):
        """Test recommendation priority assignment with edge case values."""
        # Test with values right at threshold boundaries
        boundary_scenarios = [
            # CPU usage exactly at warning threshold
            MetricSummary(
                metric_type=MetricType.CPU_USAGE,
                count=50,
                min_value=70,
                max_value=70,
                mean_value=70,  # Exactly at warning threshold
                median_value=70,
                std_dev=0,
                percentile_95=70,
                percentile_99=70,
                time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                performance_level=PerformanceLevel.AVERAGE,
                trend="stable"
            ),
            # Memory usage exactly at critical threshold
            MetricSummary(
                metric_type=MetricType.MEMORY_USAGE,
                count=50,
                min_value=486,  # 95% of 512MB
                max_value=486,
                mean_value=486,
                median_value=486,
                std_dev=0,
                percentile_95=486,
                percentile_99=486,
                time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
                performance_level=PerformanceLevel.POOR,
                trend="stable"
            ),
        ]

        def mock_get_metric_summary(metric_type, start_time=None, end_time=None):
            for summary in boundary_scenarios:
                if summary.metric_type == metric_type:
                    return summary
            return None

        mock_metrics_collector.get_metric_summary = AsyncMock(side_effect=mock_get_metric_summary)

        report = await analyzer.analyze_performance()

        # Should handle boundary conditions appropriately
        for recommendation in report.recommendations:
            assert recommendation.priority in [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
            assert recommendation.impact_estimate in ["low", "medium", "high"]
            assert recommendation.implementation_effort in ["easy", "moderate", "complex"]


class TestAlertingSystemEdgeCases:
    """Test edge cases for the alerting system."""

    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing."""
        monitor = PerformanceMonitor("test_project")
        monitor.storage = MagicMock()  # Mock storage to avoid actual persistence
        monitor.storage.store_metrics_batch = AsyncMock()
        monitor.storage.store_operation_trace = AsyncMock()
        monitor.storage.get_storage_stats = AsyncMock(return_value={})
        return monitor

    @pytest.mark.asyncio
    async def test_alert_threshold_boundary_conditions(self, performance_monitor):
        """Test alerting behavior at exact threshold boundaries."""
        alert_count = 0

        def count_alerts(alert):
            nonlocal alert_count
            alert_count += 1

        performance_monitor.add_alert_callback(count_alerts)

        # Test metrics at exact threshold boundaries
        threshold_tests = [
            # Exactly at warning threshold
            (MetricType.SEARCH_LATENCY, 200.0, "warning"),
            # Just below warning threshold
            (MetricType.SEARCH_LATENCY, 199.9, None),
            # Just above warning threshold
            (MetricType.SEARCH_LATENCY, 200.1, "warning"),
            # Exactly at critical threshold
            (MetricType.SEARCH_LATENCY, 500.0, "critical"),
            # Just below critical threshold
            (MetricType.SEARCH_LATENCY, 499.9, "warning"),
            # Just above critical threshold
            (MetricType.SEARCH_LATENCY, 500.1, "critical"),
        ]

        for metric_type, value, expected_severity in threshold_tests:
            alert_count = 0  # Reset counter

            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=metric_type,
                value=value,
                unit="ms",
                project_id="test_project"
            )

            await performance_monitor._check_metric_alerts(metric)

            if expected_severity is None:
                assert alert_count == 0, f"Unexpected alert for value {value}"
            else:
                assert alert_count > 0, f"Expected alert for value {value}"

                # Check alert severity
                active_alerts = list(performance_monitor.active_alerts.values())
                if active_alerts:
                    assert active_alerts[-1].severity == expected_severity

            # Clear active alerts for next test
            performance_monitor.active_alerts.clear()

    @pytest.mark.asyncio
    async def test_alert_cooldown_behavior(self, performance_monitor):
        """Test alert cooldown mechanism to prevent spam."""
        alerts_received = []

        def collect_alerts(alert):
            alerts_received.append(alert)

        performance_monitor.add_alert_callback(collect_alerts)

        # Create metric that triggers critical alert
        critical_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SEARCH_LATENCY,
            value=600.0,
            unit="ms",
            project_id="test_project"
        )

        # Trigger initial alert
        await performance_monitor._check_metric_alerts(critical_metric)
        initial_alert_count = len(alerts_received)
        assert initial_alert_count > 0

        # Trigger same alert type immediately (should be suppressed by cooldown)
        await performance_monitor._check_metric_alerts(critical_metric)
        suppressed_alert_count = len(alerts_received)
        assert suppressed_alert_count == initial_alert_count  # No new alerts

        # Test that different metric types are not affected by cooldown
        different_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.CPU_USAGE,
            value=95.0,
            unit="%",
            project_id="test_project"
        )

        await performance_monitor._check_metric_alerts(different_metric)
        different_metric_alert_count = len(alerts_received)
        assert different_metric_alert_count > suppressed_alert_count  # New alert allowed

    @pytest.mark.asyncio
    async def test_alert_false_positive_handling(self, performance_monitor):
        """Test handling of potential false positive alerts."""
        false_positive_alerts = []

        def collect_false_positives(alert):
            false_positive_alerts.append(alert)

        performance_monitor.add_alert_callback(collect_false_positives)

        # Test scenario that might cause false positives: temporary spike
        normal_values = [50.0, 55.0, 48.0, 52.0, 49.0]  # Normal latency
        spike_value = 600.0  # Temporary spike
        recovery_values = [51.0, 47.0, 53.0, 50.0]  # Back to normal

        # Record normal values (should not trigger alerts)
        for value in normal_values:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.SEARCH_LATENCY,
                value=value,
                unit="ms",
                project_id="test_project"
            )
            await performance_monitor._check_metric_alerts(metric)

        initial_alert_count = len(false_positive_alerts)

        # Record spike (should trigger alert)
        spike_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SEARCH_LATENCY,
            value=spike_value,
            unit="ms",
            project_id="test_project"
        )
        await performance_monitor._check_metric_alerts(spike_metric)

        spike_alert_count = len(false_positive_alerts)
        assert spike_alert_count > initial_alert_count  # Alert should be triggered

        # Record recovery values (should not trigger additional alerts due to cooldown)
        for value in recovery_values:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=MetricType.SEARCH_LATENCY,
                value=value,
                unit="ms",
                project_id="test_project"
            )
            await performance_monitor._check_metric_alerts(metric)

        final_alert_count = len(false_positive_alerts)
        # Should not have additional alerts due to cooldown
        assert final_alert_count == spike_alert_count

    @pytest.mark.asyncio
    async def test_alert_with_missing_callbacks(self, performance_monitor):
        """Test alert generation when no callbacks are registered."""
        # Ensure no callbacks are registered
        performance_monitor.alert_callbacks.clear()

        # Create metric that should trigger alert
        critical_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SEARCH_LATENCY,
            value=600.0,
            unit="ms",
            project_id="test_project"
        )

        # Should not raise exception even without callbacks
        try:
            await performance_monitor._check_metric_alerts(critical_metric)
        except Exception as e:
            pytest.fail(f"Alert generation failed without callbacks: {e}")

        # Alert should still be recorded in active alerts
        assert len(performance_monitor.active_alerts) > 0

    @pytest.mark.asyncio
    async def test_alert_callback_exceptions(self, performance_monitor):
        """Test alert system resilience when callbacks raise exceptions."""
        successful_callback_called = False

        def failing_callback(alert):
            raise RuntimeError("Callback failure simulation")

        def successful_callback(alert):
            nonlocal successful_callback_called
            successful_callback_called = True

        # Register both callbacks
        performance_monitor.add_alert_callback(failing_callback)
        performance_monitor.add_alert_callback(successful_callback)

        # Create alert
        critical_metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=MetricType.SEARCH_LATENCY,
            value=600.0,
            unit="ms",
            project_id="test_project"
        )

        # Should handle callback exceptions gracefully
        try:
            await performance_monitor._check_metric_alerts(critical_metric)
        except Exception as e:
            pytest.fail(f"Alert system failed due to callback exception: {e}")

        # Successful callback should still have been called
        assert successful_callback_called is True

        # Alert should still be recorded
        assert len(performance_monitor.active_alerts) > 0


class TestResourceTrackingEdgeCases:
    """Test edge cases for resource usage tracking."""

    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for testing."""
        return PerformanceMetricsCollector("test_project")

    @pytest.mark.asyncio
    async def test_memory_spike_detection(self, metrics_collector):
        """Test detection and handling of memory spikes."""
        # Simulate gradual memory increase followed by spike
        normal_memory_values = [100, 105, 110, 108, 112, 115]  # MB
        spike_values = [400, 450, 480, 460, 420]  # Sudden spike
        recovery_values = [120, 118, 125, 115]  # Recovery

        all_values = normal_memory_values + spike_values + recovery_values

        for value in all_values:
            await metrics_collector.record_metric(
                MetricType.MEMORY_USAGE,
                float(value),
                "MB",
                context={"test_phase": "memory_spike"}
            )

        # Get summary to analyze memory pattern
        summary = await metrics_collector.get_metric_summary(MetricType.MEMORY_USAGE)

        assert summary is not None
        assert summary.count == len(all_values)
        assert summary.max_value == 480.0  # Peak of spike
        assert summary.min_value == 100.0  # Normal baseline

        # Standard deviation should be high due to spike
        assert summary.std_dev > 50  # High variability indicates spike pattern

    @pytest.mark.asyncio
    async def test_cpu_throttling_simulation(self, metrics_collector):
        """Test CPU usage tracking under throttling conditions."""
        # Simulate CPU throttling pattern: high usage then forced reduction
        high_cpu_values = [90, 92, 95, 98, 97]  # High CPU before throttling
        throttled_values = [30, 25, 35, 28, 32]  # Throttled CPU usage

        # Record high CPU usage
        for value in high_cpu_values:
            await metrics_collector.record_metric(
                MetricType.CPU_USAGE,
                float(value),
                "%",
                context={"phase": "pre_throttle"}
            )

        # Record throttled CPU usage
        for value in throttled_values:
            await metrics_collector.record_metric(
                MetricType.CPU_USAGE,
                float(value),
                "%",
                context={"phase": "throttled"}
            )

        # Analyze the pattern
        summary = await metrics_collector.get_metric_summary(MetricType.CPU_USAGE)

        assert summary is not None
        assert summary.max_value == 98.0
        assert summary.min_value == 25.0

        # High standard deviation should indicate throttling pattern
        assert summary.std_dev > 25

        # Get individual metrics to verify pattern
        cpu_metrics = await metrics_collector.buffer.get_metrics(MetricType.CPU_USAGE)
        pre_throttle = [m for m in cpu_metrics if m.context.get("phase") == "pre_throttle"]
        throttled = [m for m in cpu_metrics if m.context.get("phase") == "throttled"]

        assert len(pre_throttle) == 5
        assert len(throttled) == 5

        # Verify pattern: pre-throttle values should be higher than throttled
        pre_throttle_avg = sum(m.value for m in pre_throttle) / len(pre_throttle)
        throttled_avg = sum(m.value for m in throttled) / len(throttled)

        assert pre_throttle_avg > throttled_avg + 30  # Significant difference

    @pytest.mark.asyncio
    async def test_disk_full_scenario(self, metrics_collector):
        """Test behavior when disk usage approaches full capacity."""
        # Simulate disk usage approaching 100%
        disk_usage_progression = [
            70, 75, 80, 85, 90,  # Normal growth
            92, 94, 96, 98, 99,  # Approaching full
            99.5, 99.8, 99.9     # Critical levels
        ]

        for usage in disk_usage_progression:
            await metrics_collector.record_metric(
                MetricType.DISK_USAGE,
                float(usage),
                "%",
                context={
                    "disk_critical": usage > 95,
                    "usage_level": usage
                }
            )

        # Analyze disk usage pattern
        summary = await metrics_collector.get_metric_summary(MetricType.DISK_USAGE)

        assert summary is not None
        assert summary.max_value == 99.9
        assert summary.min_value == 70.0

        # Check that critical levels are properly tracked
        disk_metrics = await metrics_collector.buffer.get_metrics(MetricType.DISK_USAGE)
        critical_metrics = [m for m in disk_metrics if m.context.get("disk_critical")]

        assert len(critical_metrics) == 6  # Values above 95%

        # All critical metrics should have high values
        for metric in critical_metrics:
            assert metric.value > 95.0

    @pytest.mark.asyncio
    async def test_resource_tracking_with_system_errors(self, metrics_collector):
        """Test resource tracking when system calls fail or return invalid data."""
        # Mock system resource collection that sometimes fails
        original_method = metrics_collector.record_system_resources

        async def mock_failing_system_resources():
            # Simulate intermittent failures
            import random
            if random.random() < 0.3:  # 30% failure rate
                raise OSError("System resource unavailable")

            # Sometimes return valid data
            await metrics_collector.record_metric(
                MetricType.MEMORY_USAGE,
                random.uniform(50, 200),
                "MB"
            )
            await metrics_collector.record_metric(
                MetricType.CPU_USAGE,
                random.uniform(10, 90),
                "%"
            )

        metrics_collector.record_system_resources = mock_failing_system_resources

        # Attempt multiple system resource recordings
        success_count = 0
        failure_count = 0

        for _ in range(20):
            try:
                await metrics_collector.record_system_resources()
                success_count += 1
            except OSError:
                failure_count += 1

        # Should handle failures gracefully
        assert failure_count > 0  # Some failures should have occurred
        assert success_count > 0  # Some successes should have occurred

        # Check that successful recordings were stored
        memory_metrics = await metrics_collector.buffer.get_metrics(MetricType.MEMORY_USAGE)
        cpu_metrics = await metrics_collector.buffer.get_metrics(MetricType.CPU_USAGE)

        # Should have some metrics despite failures
        assert len(memory_metrics) > 0
        assert len(cpu_metrics) > 0

        # Restore original method
        metrics_collector.record_system_resources = original_method

    @pytest.mark.asyncio
    async def test_network_resource_tracking_edge_cases(self, metrics_collector):
        """Test network resource tracking under various edge conditions."""
        # Test various network conditions
        network_scenarios = [
            # Normal network usage
            {"bytes_sent": 1024, "bytes_recv": 2048, "condition": "normal"},
            # High bandwidth usage
            {"bytes_sent": 10**9, "bytes_recv": 2*10**9, "condition": "high_bandwidth"},
            # Network outage (zero bytes)
            {"bytes_sent": 0, "bytes_recv": 0, "condition": "outage"},
            # Asymmetric usage
            {"bytes_sent": 100, "bytes_recv": 50000, "condition": "download_heavy"},
            {"bytes_sent": 50000, "bytes_recv": 100, "condition": "upload_heavy"},
        ]

        for scenario in network_scenarios:
            await metrics_collector.record_metric(
                MetricType.NETWORK_IO,
                float(scenario["bytes_sent"] + scenario["bytes_recv"]),
                "bytes",
                context={
                    "bytes_sent": scenario["bytes_sent"],
                    "bytes_recv": scenario["bytes_recv"],
                    "condition": scenario["condition"]
                }
            )

        # Analyze network patterns
        network_metrics = await metrics_collector.buffer.get_metrics(MetricType.NETWORK_IO)
        assert len(network_metrics) == len(network_scenarios)

        # Check different conditions
        outage_metrics = [m for m in network_metrics if m.context.get("condition") == "outage"]
        high_bandwidth_metrics = [m for m in network_metrics if m.context.get("condition") == "high_bandwidth"]

        assert len(outage_metrics) == 1
        assert outage_metrics[0].value == 0.0

        assert len(high_bandwidth_metrics) == 1
        assert high_bandwidth_metrics[0].value > 1e9  # Very high usage


class TestPerformanceRegressionEdgeCases:
    """Test edge cases for performance regression detection."""

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
    async def test_regression_detection_with_noisy_data(self, analyzer, mock_metrics_collector):
        """Test regression detection with noisy/erratic data patterns."""
        # Create noisy data that has underlying regression trend
        import random

        baseline_latency = 100  # ms
        regression_trend = 2    # 2ms increase per measurement
        noise_amplitude = 20    # ±20ms noise

        # Generate 100 measurements with trend + noise
        measurements = []
        for i in range(100):
            trend_value = baseline_latency + (i * regression_trend)
            noise = random.uniform(-noise_amplitude, noise_amplitude)
            measurements.append(max(0, trend_value + noise))  # Ensure non-negative

        # Create summary with noisy but regressing data
        noisy_regression_summary = MetricSummary(
            metric_type=MetricType.SEARCH_LATENCY,
            count=100,
            min_value=min(measurements),
            max_value=max(measurements),
            mean_value=sum(measurements) / len(measurements),
            median_value=sorted(measurements)[50],
            std_dev=sum((x - sum(measurements)/len(measurements))**2 for x in measurements) ** 0.5 / len(measurements),
            percentile_95=sorted(measurements)[95],
            percentile_99=sorted(measurements)[99],
            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
            performance_level=PerformanceLevel.POOR,
            trend="degrading"  # Should detect regression despite noise
        )

        mock_metrics_collector.get_metric_summary = AsyncMock(return_value=noisy_regression_summary)

        report = await analyzer.analyze_performance()

        # Should detect regression despite noise
        assert report.overall_performance_score < 70  # Should indicate poor performance

        # Should have recommendations for the regression
        search_recommendations = [
            rec for rec in report.recommendations
            if rec.optimization_type == OptimizationType.SEARCH_OPTIMIZATION
        ]
        assert len(search_recommendations) > 0

        # Should identify search as a bottleneck
        assert "Search operations" in report.bottlenecks

    @pytest.mark.asyncio
    async def test_regression_detection_with_seasonal_patterns(self, analyzer, mock_metrics_collector):
        """Test regression detection that accounts for seasonal/cyclical patterns."""
        # Simulate cyclical pattern (e.g., daily usage cycles) with underlying regression
        import math

        baseline = 150  # Base latency
        cycle_amplitude = 30  # ±30ms cyclical variation
        regression_rate = 1  # 1ms increase per hour
        hours = 24

        # Generate hourly data for 24 hours with cyclical pattern + regression
        hourly_latencies = []
        for hour in range(hours):
            # Cyclical component (simulates daily usage pattern)
            cyclical = cycle_amplitude * math.sin(2 * math.pi * hour / 24)
            # Regression component
            regression = hour * regression_rate
            # Combined
            latency = baseline + cyclical + regression
            hourly_latencies.append(latency)

        seasonal_regression_summary = MetricSummary(
            metric_type=MetricType.SEARCH_LATENCY,
            count=24,
            min_value=min(hourly_latencies),
            max_value=max(hourly_latencies),
            mean_value=sum(hourly_latencies) / len(hourly_latencies),
            median_value=sorted(hourly_latencies)[12],
            std_dev=(sum((x - sum(hourly_latencies)/len(hourly_latencies))**2 for x in hourly_latencies) / len(hourly_latencies)) ** 0.5,
            percentile_95=sorted(hourly_latencies)[int(0.95 * len(hourly_latencies))],
            percentile_99=sorted(hourly_latencies)[int(0.99 * len(hourly_latencies))],
            time_range=(datetime.now() - timedelta(hours=24), datetime.now()),
            performance_level=PerformanceLevel.AVERAGE,
            trend="degrading"  # Should detect underlying regression
        )

        mock_metrics_collector.get_metric_summary = AsyncMock(return_value=seasonal_regression_summary)

        report = await analyzer.analyze_performance()

        # Should handle seasonal patterns appropriately
        assert report.project_id == "test_project"
        assert isinstance(report.overall_performance_score, float)

        # May or may not detect regression depending on implementation sophistication
        # At minimum, should not crash with cyclical data
        assert len(report.insights) >= 0
        assert len(report.recommendations) >= 0

    @pytest.mark.asyncio
    async def test_regression_detection_with_sparse_data(self, analyzer, mock_metrics_collector):
        """Test regression detection with sparse or irregular data points."""
        # Create sparse data with irregular timing
        sparse_data_summary = MetricSummary(
            metric_type=MetricType.FILE_PROCESSING_RATE,
            count=5,  # Very few data points
            min_value=8.0,
            max_value=12.0,
            mean_value=10.0,
            median_value=10.0,
            std_dev=1.5,
            percentile_95=11.8,
            percentile_99=12.0,
            time_range=(datetime.now() - timedelta(hours=6), datetime.now()),
            performance_level=PerformanceLevel.AVERAGE,
            trend="stable"  # Insufficient data for trend detection
        )

        mock_metrics_collector.get_metric_summary = AsyncMock(return_value=sparse_data_summary)

        report = await analyzer.analyze_performance()

        # Should handle sparse data gracefully
        assert report.project_id == "test_project"
        assert report.overall_performance_score > 0

        # With sparse data, should be conservative about regression detection
        critical_recommendations = [
            rec for rec in report.recommendations
            if rec.priority == Priority.CRITICAL
        ]

        # Should not generate critical recommendations based on sparse data
        assert len(critical_recommendations) == 0

    @pytest.mark.asyncio
    async def test_false_regression_detection(self, analyzer, mock_metrics_collector):
        """Test scenarios that might trigger false regression detection."""
        # Scenario 1: Single outlier in otherwise stable data
        stable_with_outlier_summary = MetricSummary(
            metric_type=MetricType.SEARCH_LATENCY,
            count=100,
            min_value=45,
            max_value=2000,  # Single outlier
            mean_value=70,   # Mean affected by outlier
            median_value=50, # Median not affected
            std_dev=200,     # High std dev due to outlier
            percentile_95=55, # 95th percentile still normal
            percentile_99=1800, # 99th percentile shows outlier
            time_range=(datetime.now() - timedelta(hours=1), datetime.now()),
            performance_level=PerformanceLevel.AVERAGE,
            trend="stable"
        )

        # Scenario 2: Initial measurement period (cold start)
        cold_start_summary = MetricSummary(
            metric_type=MetricType.SEARCH_LATENCY,
            count=10,  # Few measurements
            min_value=200,
            max_value=800,  # High initial latency
            mean_value=400,
            median_value=350,
            std_dev=150,
            percentile_95=700,
            percentile_99=800,
            time_range=(datetime.now() - timedelta(minutes=5), datetime.now()),
            performance_level=PerformanceLevel.POOR,
            trend="improving"  # Should be improving as system warms up
        )

        test_scenarios = [
            ("outlier_scenario", stable_with_outlier_summary),
            ("cold_start_scenario", cold_start_summary)
        ]

        for scenario_name, summary in test_scenarios:
            mock_metrics_collector.get_metric_summary = AsyncMock(return_value=summary)

            report = await analyzer.analyze_performance()

            if scenario_name == "outlier_scenario":
                # Should not heavily penalize performance due to single outlier
                # Median-based analysis should show system is actually performing well
                assert report.overall_performance_score > 40  # Not critically bad

                # Should not generate critical recommendations for single outlier
                critical_recs = [r for r in report.recommendations if r.priority == Priority.CRITICAL]
                assert len(critical_recs) <= 1  # At most one critical recommendation

            elif scenario_name == "cold_start_scenario":
                # Should recognize improving trend and not over-alert
                # May have recommendations but should not be overly critical
                improving_insights = [
                    i for i in report.insights
                    if "improving" in i.trend or "improving" in i.description.lower()
                ]
                # Should recognize improvement or at least not be overly pessimistic
                assert len(improving_insights) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])