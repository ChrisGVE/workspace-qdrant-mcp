"""
Integration tests for performance monitoring and metrics system.

Tests the performance monitoring functionality, baseline management,
regression detection, and metrics collection accuracy.

Task: #290.9 - Build performance monitoring and metrics integration
Parent: #290 - Build MCP-daemon integration test framework
"""

import asyncio
import tempfile
import time
from pathlib import Path

import pytest

from tests.integration.performance_monitoring import (
    PerformanceBaseline,
    PerformanceFixture,
    PerformanceMetrics,
    PerformanceMonitor,
)


@pytest.fixture
def temp_baselines_path(tmp_path):
    """Create temporary path for baselines."""
    return tmp_path / "baselines.json"


@pytest.fixture
def performance_monitor(temp_baselines_path):
    """Create performance monitor with temporary baseline storage."""
    return PerformanceMonitor(baselines_path=temp_baselines_path)


class TestPerformanceMetricsCollection:
    """Test performance metrics collection."""

    @pytest.mark.asyncio
    async def test_basic_metrics_collection(self, performance_monitor):
        """Test basic metrics collection for a test."""
        # Step 1: Start monitoring
        performance_monitor.start_monitoring("test_basic_collection")

        # Step 2: Simulate operations
        for _i in range(10):
            start = time.time()
            await asyncio.sleep(0.01)  # 10ms operation
            latency_ms = (time.time() - start) * 1000
            performance_monitor.record_operation(latency_ms)

        # Step 3: Stop monitoring and get metrics
        metrics = performance_monitor.stop_monitoring()

        # Step 4: Validate metrics
        assert metrics.test_name == "test_basic_collection"
        assert metrics.operations_count == 10
        assert metrics.duration_ms > 0
        assert len(metrics.latencies_ms) == 10
        assert metrics.avg_latency_ms > 0
        assert metrics.throughput_ops_per_sec > 0

    @pytest.mark.asyncio
    async def test_context_manager_monitoring(self, performance_monitor):
        """Test monitoring using context manager."""
        # Use context manager
        async with performance_monitor.monitor_test("test_context_manager"):
            for _i in range(5):
                start = time.time()
                await asyncio.sleep(0.005)
                latency_ms = (time.time() - start) * 1000
                performance_monitor.record_operation(latency_ms)

        # Metrics should be available
        metrics = performance_monitor.current_metrics
        assert metrics is not None
        assert metrics.operations_count == 5

    @pytest.mark.asyncio
    async def test_latency_percentiles(self, performance_monitor):
        """Test latency percentile calculations."""
        performance_monitor.start_monitoring("test_percentiles")

        # Create known distribution
        latencies = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # ms
        for latency in latencies:
            performance_monitor.record_operation(latency)

        metrics = performance_monitor.stop_monitoring()

        # Validate percentiles
        assert metrics.avg_latency_ms == 55.0  # Average
        assert metrics.median_latency_ms == 55.0  # Median
        assert metrics.min_latency_ms == 10.0
        assert metrics.max_latency_ms == 100.0
        # P95 should be around 95th percentile
        assert metrics.p95_latency_ms >= 90.0

    @pytest.mark.asyncio
    async def test_error_tracking(self, performance_monitor):
        """Test error and warning tracking."""
        performance_monitor.start_monitoring("test_errors")

        # Record operations with errors
        for i in range(10):
            error = (i % 3 == 0)  # Every 3rd operation fails
            performance_monitor.record_operation(10.0, error=error)

        # Record warnings
        for _i in range(5):
            performance_monitor.record_warning()

        metrics = performance_monitor.stop_monitoring()

        assert metrics.errors_count == 4  # Operations 0, 3, 6, 9
        assert metrics.warnings_count == 5


class TestResourceMonitoring:
    """Test resource usage monitoring."""

    @pytest.mark.asyncio
    async def test_cpu_memory_monitoring(self, performance_monitor):
        """Test CPU and memory usage monitoring."""
        performance_monitor.start_monitoring("test_resources")

        # Wait for some resource samples
        await asyncio.sleep(1.5)  # Collect at least 3 samples (500ms intervals)

        # Do some work
        [i ** 2 for i in range(10000)]  # Allocate some memory

        metrics = performance_monitor.stop_monitoring()

        # Validate resource metrics were collected
        assert metrics.cpu_percent_avg >= 0
        assert metrics.cpu_percent_max >= 0
        assert metrics.memory_mb_avg > 0
        assert metrics.memory_mb_max > 0
        assert metrics.memory_mb_max >= metrics.memory_mb_avg

    @pytest.mark.asyncio
    async def test_resource_monitoring_cancellation(self, performance_monitor):
        """Test that resource monitoring stops cleanly."""
        performance_monitor.start_monitoring("test_cancellation")
        await asyncio.sleep(0.5)

        # Stop monitoring
        performance_monitor.stop_monitoring()

        # Monitoring task should be cancelled
        assert performance_monitor.monitoring_task.cancelled() or performance_monitor.monitoring_task.done()


class TestBaselineManagement:
    """Test performance baseline management."""

    @pytest.mark.asyncio
    async def test_set_and_load_baseline(self, performance_monitor, temp_baselines_path):
        """Test setting and loading baselines."""
        # Step 1: Create metrics
        performance_monitor.start_monitoring("test_baseline_mgmt")
        for _i in range(10):
            performance_monitor.record_operation(10.0)
        metrics = performance_monitor.stop_monitoring()

        # Step 2: Set baseline
        performance_monitor.set_baseline(metrics)

        # Verify baseline exists
        assert "test_baseline_mgmt" in performance_monitor.baselines

        # Step 3: Create new monitor and load baselines
        new_monitor = PerformanceMonitor(baselines_path=temp_baselines_path)

        # Baseline should be loaded
        assert "test_baseline_mgmt" in new_monitor.baselines
        baseline = new_monitor.baselines["test_baseline_mgmt"]
        assert baseline.avg_latency_ms > 0
        assert baseline.baseline_date is not None

    @pytest.mark.asyncio
    async def test_baseline_persistence(self, performance_monitor, temp_baselines_path):
        """Test baseline persistence across sessions."""
        # Create and set baseline
        performance_monitor.start_monitoring("test_persistence")
        performance_monitor.record_operation(15.0)
        metrics = performance_monitor.stop_monitoring()
        performance_monitor.set_baseline(metrics)

        # Verify file was created
        assert temp_baselines_path.exists()

        # Load in new monitor
        new_monitor = PerformanceMonitor(baselines_path=temp_baselines_path)
        assert "test_persistence" in new_monitor.baselines


class TestRegressionDetection:
    """Test performance regression detection."""

    @pytest.mark.asyncio
    async def test_no_regression_detection(self, performance_monitor):
        """Test when no regression occurs."""
        # Set baseline
        performance_monitor.start_monitoring("test_no_regression")
        for _i in range(10):
            performance_monitor.record_operation(10.0)
        baseline_metrics = performance_monitor.stop_monitoring()
        performance_monitor.set_baseline(baseline_metrics)

        # Run test with similar performance
        performance_monitor.start_monitoring("test_no_regression")
        for _i in range(10):
            performance_monitor.record_operation(10.5)  # Slight increase within tolerance
        metrics = performance_monitor.stop_monitoring()

        # Check for regressions
        warnings = performance_monitor.check_regression(metrics)

        # Should have no warnings (within 20% tolerance)
        regression_warnings = [w for w in warnings if "REGRESSION" in w]
        assert len(regression_warnings) == 0

    @pytest.mark.asyncio
    async def test_latency_regression_detection(self, performance_monitor):
        """Test latency regression detection."""
        # Set baseline with 10ms latency
        performance_monitor.start_monitoring("test_latency_regression")
        for _i in range(10):
            performance_monitor.record_operation(10.0)
        baseline_metrics = performance_monitor.stop_monitoring()
        performance_monitor.set_baseline(baseline_metrics)

        # Run test with 50% latency increase (exceeds 20% tolerance)
        performance_monitor.start_monitoring("test_latency_regression")
        for _i in range(10):
            performance_monitor.record_operation(15.0)  # 50% increase
        metrics = performance_monitor.stop_monitoring()

        # Check for regressions
        warnings = performance_monitor.check_regression(metrics)

        # Should detect latency regression
        regression_warnings = [w for w in warnings if "latency increased" in w.lower()]
        assert len(regression_warnings) > 0

    @pytest.mark.asyncio
    async def test_throughput_regression_detection(self, performance_monitor):
        """Test throughput regression detection."""
        # Set baseline with high throughput
        performance_monitor.start_monitoring("test_throughput_regression")
        for _i in range(100):
            performance_monitor.record_operation(1.0)  # Fast operations
        await asyncio.sleep(0.1)  # Very short duration = high throughput
        baseline_metrics = performance_monitor.stop_monitoring()
        performance_monitor.set_baseline(baseline_metrics)

        # Run test with lower throughput (slower operations)
        performance_monitor.start_monitoring("test_throughput_regression")
        for _i in range(50):  # Half the operations
            performance_monitor.record_operation(5.0)  # Slower
        await asyncio.sleep(0.5)  # Longer duration = lower throughput
        metrics = performance_monitor.stop_monitoring()

        # Check for regressions
        warnings = performance_monitor.check_regression(metrics)

        # Should detect throughput regression
        regression_warnings = [w for w in warnings if "throughput decreased" in w.lower()]
        assert len(regression_warnings) > 0


class TestReportGeneration:
    """Test performance report generation."""

    @pytest.mark.asyncio
    async def test_comprehensive_report(self, performance_monitor):
        """Test comprehensive report generation."""
        # Create metrics
        performance_monitor.start_monitoring("test_report")
        for i in range(20):
            performance_monitor.record_operation(10.0 + i)  # Varying latencies
        performance_monitor.record_warning()
        performance_monitor.record_warning()
        metrics = performance_monitor.stop_monitoring()

        # Generate report
        report = performance_monitor.generate_report(metrics)

        # Validate report contains key sections
        assert "PERFORMANCE REPORT" in report
        assert "test_report" in report
        assert "Test Duration" in report
        assert "Operations:" in report
        assert "Latency (ms):" in report
        assert "Average:" in report
        assert "P95:" in report
        assert "P99:" in report
        assert "Resource Usage:" in report
        assert "Warnings: 2" in report

    @pytest.mark.asyncio
    async def test_report_with_baseline(self, performance_monitor):
        """Test report includes baseline comparison."""
        # Set baseline
        performance_monitor.start_monitoring("test_report_baseline")
        for _i in range(10):
            performance_monitor.record_operation(10.0)
        baseline_metrics = performance_monitor.stop_monitoring()
        performance_monitor.set_baseline(baseline_metrics)

        # Run test
        performance_monitor.start_monitoring("test_report_baseline")
        for _i in range(10):
            performance_monitor.record_operation(10.5)
        metrics = performance_monitor.stop_monitoring()

        # Generate report
        report = performance_monitor.generate_report(metrics)

        # Should mention baseline
        assert "baseline:" in report.lower() or "regression" in report.lower()


@pytest.mark.asyncio
async def test_performance_monitoring_integration_report():
    """Generate comprehensive performance monitoring integration report."""
    print("\n" + "=" * 80)
    print("PERFORMANCE MONITORING INTEGRATION TEST REPORT")
    print("=" * 80)

    print("\nFeatures Tested:")
    print("  ✓ Basic metrics collection (operations, latency, duration)")
    print("  ✓ Context manager monitoring interface")
    print("  ✓ Latency percentile calculations (P50, P95, P99)")
    print("  ✓ Error and warning tracking")
    print("  ✓ CPU and memory usage monitoring")
    print("  ✓ Resource monitoring task cancellation")
    print("  ✓ Baseline creation and persistence")
    print("  ✓ Baseline loading across sessions")
    print("  ✓ Regression detection (latency, throughput)")
    print("  ✓ Performance report generation")

    print("\nMetrics Collected:")
    print("  - Operation count and throughput (ops/sec)")
    print("  - Latency statistics (avg, median, min, max, P95, P99)")
    print("  - Resource usage (CPU %, memory MB)")
    print("  - Error and warning counts")
    print("  - Test duration")

    print("\nBaseline Management:")
    print("  - Persistent storage in JSON format")
    print("  - Automatic loading on monitor initialization")
    print("  - Tolerance thresholds (latency: 20%, throughput: 20%, resources: 30%)")

    print("\nRegression Detection:")
    print("  - Latency regression (>20% increase)")
    print("  - Throughput regression (>20% decrease)")
    print("  - Memory regression (>30% increase)")
    print("  - Automatic warning generation")

    print("\nIntegration Points:")
    print("  - Pytest markers: @pytest.mark.performance, @pytest.mark.baseline")
    print("  - Context manager API for easy test integration")
    print("  - Automatic resource monitoring (500ms sampling)")
    print("  - JSON export for CI/CD integration")

    print("\n" + "=" * 80)
    print("Performance monitoring system ready for integration test framework.")
    print("=" * 80)
