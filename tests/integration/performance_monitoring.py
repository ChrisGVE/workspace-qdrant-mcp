"""
Performance monitoring and metrics integration for integration tests.

Provides performance metrics collection, latency monitoring, throughput measurement,
resource usage tracking, and performance regression detection for the integration
test framework.

Features:
1. Real-time performance metrics collection
2. Latency percentile tracking (P50, P95, P99)
3. Throughput measurement (requests/second)
4. Resource usage monitoring (CPU, memory, disk I/O)
5. Performance regression detection
6. Baseline management and comparison
7. Alert generation for performance degradation
8. Test performance profiling

Task: #290.9 - Build performance monitoring and metrics integration
Parent: #290 - Build MCP-daemon integration test framework
"""

import asyncio
import json
import statistics
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil


@dataclass
class PerformanceMetrics:
    """Performance metrics for a test run."""
    test_name: str
    start_time: float
    end_time: float
    duration_ms: float

    # Latency metrics
    latencies_ms: list[float] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0

    # Throughput metrics
    operations_count: int = 0
    throughput_ops_per_sec: float = 0.0

    # Resource metrics
    cpu_percent_avg: float = 0.0
    cpu_percent_max: float = 0.0
    memory_mb_avg: float = 0.0
    memory_mb_max: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0

    # Additional metrics
    errors_count: int = 0
    warnings_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_statistics(self):
        """Calculate statistical metrics from collected data."""
        if self.latencies_ms:
            self.avg_latency_ms = statistics.mean(self.latencies_ms)
            self.median_latency_ms = statistics.median(self.latencies_ms)
            self.min_latency_ms = min(self.latencies_ms)
            self.max_latency_ms = max(self.latencies_ms)

            # Calculate percentiles
            if len(self.latencies_ms) >= 20:
                quantiles = statistics.quantiles(self.latencies_ms, n=100)
                self.p95_latency_ms = quantiles[94]
                self.p99_latency_ms = quantiles[98]
            else:
                # Fallback for small samples
                sorted_latencies = sorted(self.latencies_ms)
                self.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
                self.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        # Calculate throughput
        duration_sec = self.duration_ms / 1000
        if duration_sec > 0:
            self.throughput_ops_per_sec = self.operations_count / duration_sec

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary, excluding raw latencies."""
        data = asdict(self)
        # Remove raw latencies list (too large for storage)
        data.pop('latencies_ms', None)
        return data


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    test_name: str
    baseline_date: str
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_ops_per_sec: float
    memory_mb_max: float
    cpu_percent_max: float

    # Tolerance thresholds (percentage)
    latency_tolerance: float = 20.0  # 20% degradation allowed
    throughput_tolerance: float = 20.0
    resource_tolerance: float = 30.0


class PerformanceMonitor:
    """
    Performance monitoring system for integration tests.

    Collects metrics, tracks baselines, and detects regressions.
    """

    def __init__(self, baselines_path: Path | None = None):
        """
        Initialize performance monitor.

        Args:
            baselines_path: Path to store baseline metrics
        """
        self.baselines_path = baselines_path or Path("test_results/performance_baselines.json")
        self.baselines_path.parent.mkdir(parents=True, exist_ok=True)

        self.current_metrics: PerformanceMetrics | None = None
        self.baselines: dict[str, PerformanceBaseline] = {}
        self.load_baselines()

        # Resource monitoring
        self.process = psutil.Process()
        self.resource_samples: list[dict[str, float]] = []
        self.monitoring_task: asyncio.Task | None = None

    def load_baselines(self):
        """Load performance baselines from file."""
        if self.baselines_path.exists():
            try:
                with open(self.baselines_path) as f:
                    data = json.load(f)
                    self.baselines = {
                        name: PerformanceBaseline(**baseline)
                        for name, baseline in data.items()
                    }
            except Exception as e:
                print(f"Warning: Could not load baselines: {e}")

    def save_baselines(self):
        """Save performance baselines to file."""
        try:
            data = {
                name: asdict(baseline)
                for name, baseline in self.baselines.items()
            }
            with open(self.baselines_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save baselines: {e}")

    def start_monitoring(self, test_name: str):
        """
        Start monitoring performance for a test.

        Args:
            test_name: Name of the test being monitored
        """
        self.current_metrics = PerformanceMetrics(
            test_name=test_name,
            start_time=time.time(),
            end_time=0.0,
            duration_ms=0.0,
        )

        # Start resource monitoring
        self.resource_samples = []
        self.monitoring_task = asyncio.create_task(self._monitor_resources())

    async def _monitor_resources(self):
        """Monitor resource usage periodically."""
        while True:
            try:
                cpu_percent = self.process.cpu_percent(interval=0.1)
                memory_mb = self.process.memory_info().rss / 1024 / 1024

                self.resource_samples.append({
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'timestamp': time.time(),
                })

                await asyncio.sleep(0.5)  # Sample every 500ms
            except asyncio.CancelledError:
                break
            except Exception:
                # Continue monitoring even if errors occur
                pass

    def record_operation(self, latency_ms: float, error: bool = False):
        """
        Record a single operation.

        Args:
            latency_ms: Operation latency in milliseconds
            error: Whether the operation resulted in an error
        """
        if self.current_metrics:
            self.current_metrics.latencies_ms.append(latency_ms)
            self.current_metrics.operations_count += 1

            if error:
                self.current_metrics.errors_count += 1

    def record_warning(self):
        """Record a warning event."""
        if self.current_metrics:
            self.current_metrics.warnings_count += 1

    def stop_monitoring(self) -> PerformanceMetrics:
        """
        Stop monitoring and calculate final metrics.

        Returns:
            Final performance metrics
        """
        if not self.current_metrics:
            raise RuntimeError("Monitoring not started")

        # Stop resource monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()

        # Calculate end time and duration
        self.current_metrics.end_time = time.time()
        self.current_metrics.duration_ms = (
            (self.current_metrics.end_time - self.current_metrics.start_time) * 1000
        )

        # Calculate resource metrics
        if self.resource_samples:
            cpu_samples = [s['cpu_percent'] for s in self.resource_samples]
            memory_samples = [s['memory_mb'] for s in self.resource_samples]

            self.current_metrics.cpu_percent_avg = statistics.mean(cpu_samples)
            self.current_metrics.cpu_percent_max = max(cpu_samples)
            self.current_metrics.memory_mb_avg = statistics.mean(memory_samples)
            self.current_metrics.memory_mb_max = max(memory_samples)

        # Calculate statistical metrics
        self.current_metrics.calculate_statistics()

        return self.current_metrics

    @asynccontextmanager
    async def monitor_test(self, test_name: str):
        """
        Context manager for monitoring a test.

        Usage:
            async with monitor.monitor_test("test_name"):
                # test code here

        Args:
            test_name: Name of the test

        Yields:
            Performance monitor instance
        """
        self.start_monitoring(test_name)
        try:
            yield self
        finally:
            metrics = self.stop_monitoring()
            # Check for regressions
            self.check_regression(metrics)

    def set_baseline(self, metrics: PerformanceMetrics):
        """
        Set performance baseline for a test.

        Args:
            metrics: Metrics to use as baseline
        """
        baseline = PerformanceBaseline(
            test_name=metrics.test_name,
            baseline_date=datetime.now().isoformat(),
            avg_latency_ms=metrics.avg_latency_ms,
            p95_latency_ms=metrics.p95_latency_ms,
            p99_latency_ms=metrics.p99_latency_ms,
            throughput_ops_per_sec=metrics.throughput_ops_per_sec,
            memory_mb_max=metrics.memory_mb_max,
            cpu_percent_max=metrics.cpu_percent_max,
        )

        self.baselines[metrics.test_name] = baseline
        self.save_baselines()

    def check_regression(self, metrics: PerformanceMetrics) -> list[str]:
        """
        Check for performance regressions against baseline.

        Args:
            metrics: Current test metrics

        Returns:
            List of regression warnings
        """
        warnings = []

        baseline = self.baselines.get(metrics.test_name)
        if not baseline:
            # No baseline, suggest setting one
            warnings.append(
                f"No baseline for {metrics.test_name}. "
                f"Consider setting baseline with current metrics."
            )
            return warnings

        # Check latency regression
        if metrics.avg_latency_ms > 0:
            latency_increase = (
                (metrics.avg_latency_ms - baseline.avg_latency_ms) /
                baseline.avg_latency_ms * 100
            )

            if latency_increase > baseline.latency_tolerance:
                warnings.append(
                    f"REGRESSION: Average latency increased by {latency_increase:.1f}% "
                    f"(baseline: {baseline.avg_latency_ms:.2f}ms, "
                    f"current: {metrics.avg_latency_ms:.2f}ms, "
                    f"tolerance: {baseline.latency_tolerance}%)"
                )

        # Check P95 latency regression
        if metrics.p95_latency_ms > 0:
            p95_increase = (
                (metrics.p95_latency_ms - baseline.p95_latency_ms) /
                baseline.p95_latency_ms * 100
            )

            if p95_increase > baseline.latency_tolerance:
                warnings.append(
                    f"REGRESSION: P95 latency increased by {p95_increase:.1f}% "
                    f"(baseline: {baseline.p95_latency_ms:.2f}ms, "
                    f"current: {metrics.p95_latency_ms:.2f}ms)"
                )

        # Check throughput regression
        if metrics.throughput_ops_per_sec > 0:
            throughput_decrease = (
                (baseline.throughput_ops_per_sec - metrics.throughput_ops_per_sec) /
                baseline.throughput_ops_per_sec * 100
            )

            if throughput_decrease > baseline.throughput_tolerance:
                warnings.append(
                    f"REGRESSION: Throughput decreased by {throughput_decrease:.1f}% "
                    f"(baseline: {baseline.throughput_ops_per_sec:.2f} ops/s, "
                    f"current: {metrics.throughput_ops_per_sec:.2f} ops/s, "
                    f"tolerance: {baseline.throughput_tolerance}%)"
                )

        # Check memory regression
        if metrics.memory_mb_max > 0:
            memory_increase = (
                (metrics.memory_mb_max - baseline.memory_mb_max) /
                baseline.memory_mb_max * 100
            )

            if memory_increase > baseline.resource_tolerance:
                warnings.append(
                    f"REGRESSION: Peak memory increased by {memory_increase:.1f}% "
                    f"(baseline: {baseline.memory_mb_max:.2f}MB, "
                    f"current: {metrics.memory_mb_max:.2f}MB, "
                    f"tolerance: {baseline.resource_tolerance}%)"
                )

        return warnings

    def generate_report(self, metrics: PerformanceMetrics) -> str:
        """
        Generate performance report for metrics.

        Args:
            metrics: Metrics to report

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append(f"PERFORMANCE REPORT: {metrics.test_name}")
        report.append("=" * 80)

        # Duration
        report.append(f"\nTest Duration: {metrics.duration_ms:.2f}ms ({metrics.duration_ms/1000:.2f}s)")

        # Operations
        report.append("\nOperations:")
        report.append(f"  Total: {metrics.operations_count}")
        report.append(f"  Errors: {metrics.errors_count}")
        report.append(f"  Warnings: {metrics.warnings_count}")
        report.append(f"  Throughput: {metrics.throughput_ops_per_sec:.2f} ops/s")

        # Latency
        if metrics.latencies_ms:
            report.append("\nLatency (ms):")
            report.append(f"  Average: {metrics.avg_latency_ms:.2f}")
            report.append(f"  Median: {metrics.median_latency_ms:.2f}")
            report.append(f"  Min: {metrics.min_latency_ms:.2f}")
            report.append(f"  Max: {metrics.max_latency_ms:.2f}")
            report.append(f"  P95: {metrics.p95_latency_ms:.2f}")
            report.append(f"  P99: {metrics.p99_latency_ms:.2f}")

        # Resources
        report.append("\nResource Usage:")
        report.append(f"  CPU Average: {metrics.cpu_percent_avg:.2f}%")
        report.append(f"  CPU Peak: {metrics.cpu_percent_max:.2f}%")
        report.append(f"  Memory Average: {metrics.memory_mb_avg:.2f}MB")
        report.append(f"  Memory Peak: {metrics.memory_mb_max:.2f}MB")

        # Regression check
        regressions = self.check_regression(metrics)
        if regressions:
            report.append("\n⚠️  PERFORMANCE REGRESSIONS DETECTED:")
            for regression in regressions:
                report.append(f"  - {regression}")
        else:
            baseline = self.baselines.get(metrics.test_name)
            if baseline:
                report.append(f"\n✓ No regressions detected (baseline: {baseline.baseline_date})")

        report.append("=" * 80)

        return "\n".join(report)


# Pytest fixtures for performance monitoring
def pytest_configure(config):
    """Configure pytest with performance monitoring markers."""
    config.addinivalue_line(
        "markers",
        "performance: mark test for performance monitoring"
    )
    config.addinivalue_line(
        "markers",
        "baseline: mark test to set performance baseline"
    )


class PerformanceFixture:
    """Pytest fixture helper for performance monitoring."""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.current_test_metrics: PerformanceMetrics | None = None

    @asynccontextmanager
    async def track_performance(self, test_name: str):
        """Track performance for a test."""
        async with self.monitor.monitor_test(test_name):
            yield self.monitor

        # Store metrics for reporting
        self.current_test_metrics = self.monitor.current_metrics

    def get_metrics(self) -> PerformanceMetrics | None:
        """Get metrics from last tracked test."""
        return self.current_test_metrics

    def set_baseline(self):
        """Set baseline from current metrics."""
        if self.current_test_metrics:
            self.monitor.set_baseline(self.current_test_metrics)

    def print_report(self):
        """Print performance report."""
        if self.current_test_metrics:
            print(self.monitor.generate_report(self.current_test_metrics))


# Example usage in tests
async def example_test_with_monitoring():
    """
    Example of how to use performance monitoring in tests.
    """
    monitor = PerformanceMonitor()

    async with monitor.monitor_test("example_concurrent_requests"):
        # Simulate operations
        for _i in range(100):
            start = time.time()
            await asyncio.sleep(0.01)  # Simulate work
            latency_ms = (time.time() - start) * 1000
            monitor.record_operation(latency_ms)

    # Get final metrics
    metrics = monitor.current_metrics

    # Generate report
    report = monitor.generate_report(metrics)
    print(report)

    # Set as baseline if this is first run
    if not monitor.baselines.get(metrics.test_name):
        monitor.set_baseline(metrics)
