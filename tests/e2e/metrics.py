"""
Performance Metrics Collection Framework (Task 293.6).

Comprehensive metrics collection and analysis for workflow simulation tests.
Provides throughput measurement, latency tracking, resource monitoring, and
automated performance reporting.

Features:
    - Latency tracking (p50, p95, p99, p99.9)
    - Throughput measurement (ops/sec, docs/sec)
    - Resource utilization monitoring (CPU, memory, disk)
    - Error rate tracking
    - Performance baselines and regression detection
    - Automated report generation
    - Statistical analysis and trend detection

Usage:
    ```python
    from tests.e2e.metrics import MetricsCollector, PerformanceReport

    # Initialize collector
    collector = MetricsCollector("workflow_name")

    # Record operations
    with collector.measure("operation_name"):
        # Perform operation
        pass

    # Generate report
    report = PerformanceReport(collector)
    print(report.summary())
    ```
"""

import asyncio
import json
import psutil
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
import numpy as np


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class LatencyMetrics:
    """Latency metrics with percentiles."""

    count: int
    min: float
    max: float
    mean: float
    median: float
    p50: float
    p95: float
    p99: float
    p99_9: float
    std_dev: float

    @classmethod
    def from_measurements(cls, measurements: List[float]) -> 'LatencyMetrics':
        """Calculate latency metrics from measurements."""
        if not measurements:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        measurements = sorted(measurements)
        count = len(measurements)

        return cls(
            count=count,
            min=min(measurements),
            max=max(measurements),
            mean=statistics.mean(measurements),
            median=statistics.median(measurements),
            p50=np.percentile(measurements, 50),
            p95=np.percentile(measurements, 95),
            p99=np.percentile(measurements, 99),
            p99_9=np.percentile(measurements, 99.9) if count >= 1000 else measurements[-1],
            std_dev=statistics.stdev(measurements) if count > 1 else 0.0
        )


@dataclass
class ThroughputMetrics:
    """Throughput metrics."""

    total_operations: int
    duration_seconds: float
    ops_per_second: float
    operations_per_minute: float
    peak_ops_per_second: float
    average_batch_size: float

    @classmethod
    def from_operations(
        cls,
        operations: List[float],
        duration: float,
        batch_sizes: Optional[List[int]] = None
    ) -> 'ThroughputMetrics':
        """Calculate throughput metrics from operations."""
        if not operations or duration <= 0:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0)

        total_ops = len(operations)
        ops_per_sec = total_ops / duration

        # Calculate peak throughput (operations per second in best 1-second window)
        if duration >= 1.0:
            window_ops = []
            for i in range(len(operations) - 1):
                if operations[i + 1] - operations[i] <= 1.0:
                    # Count operations in 1-second windows
                    window_count = sum(
                        1 for op in operations
                        if operations[i] <= op <= operations[i] + 1.0
                    )
                    window_ops.append(window_count)
            peak_ops = max(window_ops) if window_ops else ops_per_sec
        else:
            peak_ops = ops_per_sec

        avg_batch = statistics.mean(batch_sizes) if batch_sizes else 1.0

        return cls(
            total_operations=total_ops,
            duration_seconds=duration,
            ops_per_second=ops_per_sec,
            operations_per_minute=ops_per_sec * 60,
            peak_ops_per_second=peak_ops,
            average_batch_size=avg_batch
        )


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""

    cpu_percent_avg: float
    cpu_percent_peak: float
    memory_mb_avg: float
    memory_mb_peak: float
    memory_percent_avg: float
    memory_percent_peak: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_received_mb: float

    @classmethod
    def from_samples(cls, samples: List[Dict[str, Any]]) -> 'ResourceMetrics':
        """Calculate resource metrics from samples."""
        if not samples:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        cpu_vals = [s['cpu_percent'] for s in samples]
        mem_mb_vals = [s['memory_mb'] for s in samples]
        mem_pct_vals = [s['memory_percent'] for s in samples]

        return cls(
            cpu_percent_avg=statistics.mean(cpu_vals),
            cpu_percent_peak=max(cpu_vals),
            memory_mb_avg=statistics.mean(mem_mb_vals),
            memory_mb_peak=max(mem_mb_vals),
            memory_percent_avg=statistics.mean(mem_pct_vals),
            memory_percent_peak=max(mem_pct_vals),
            disk_io_read_mb=sum(s.get('disk_read_mb', 0.0) for s in samples),
            disk_io_write_mb=sum(s.get('disk_write_mb', 0.0) for s in samples),
            network_sent_mb=sum(s.get('net_sent_mb', 0.0) for s in samples),
            network_received_mb=sum(s.get('net_recv_mb', 0.0) for s in samples)
        )


@dataclass
class ErrorMetrics:
    """Error rate metrics."""

    total_operations: int
    total_errors: int
    error_rate: float
    error_types: Dict[str, int]
    first_error_time: Optional[float]
    last_error_time: Optional[float]

    @classmethod
    def from_errors(
        cls,
        total_ops: int,
        errors: List[Dict[str, Any]]
    ) -> 'ErrorMetrics':
        """Calculate error metrics from error records."""
        error_count = len(errors)
        error_rate = error_count / total_ops if total_ops > 0 else 0.0

        error_types = defaultdict(int)
        for error in errors:
            error_type = error.get('type', 'unknown')
            error_types[error_type] += 1

        first_error = min((e['timestamp'] for e in errors), default=None)
        last_error = max((e['timestamp'] for e in errors), default=None)

        return cls(
            total_operations=total_ops,
            total_errors=error_count,
            error_rate=error_rate,
            error_types=dict(error_types),
            first_error_time=first_error,
            last_error_time=last_error
        )


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """
    Comprehensive metrics collection for workflow simulations.

    Collects timing, throughput, resource, and error metrics during test execution.
    """

    def __init__(self, workflow_name: str):
        """
        Initialize metrics collector.

        Args:
            workflow_name: Name of the workflow being measured
        """
        self.workflow_name = workflow_name
        self.start_time = time.time()

        # Timing measurements
        self.operation_timings: Dict[str, List[float]] = defaultdict(list)
        self.operation_timestamps: Dict[str, List[float]] = defaultdict(list)

        # Throughput measurements
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.batch_sizes: Dict[str, List[int]] = defaultdict(list)

        # Resource samples
        self.resource_samples: List[Dict[str, Any]] = []
        self._resource_monitor_task: Optional[asyncio.Task] = None

        # Error tracking
        self.errors: List[Dict[str, Any]] = []

        # Custom metrics
        self.custom_metrics: Dict[str, Any] = {}

    @contextmanager
    def measure(self, operation_name: str):
        """
        Context manager for measuring operation latency.

        Args:
            operation_name: Name of the operation being measured

        Example:
            ```python
            with collector.measure("database_query"):
                result = db.query()
            ```
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.operation_timings[operation_name].append(duration)
            self.operation_timestamps[operation_name].append(time.time())
            self.operation_counts[operation_name] += 1

    def record_operation(
        self,
        operation_name: str,
        duration: float,
        batch_size: int = 1,
        success: bool = True,
        error: Optional[Exception] = None
    ):
        """
        Manually record an operation.

        Args:
            operation_name: Name of the operation
            duration: Operation duration in seconds
            batch_size: Number of items processed
            success: Whether operation succeeded
            error: Error if operation failed
        """
        self.operation_timings[operation_name].append(duration)
        self.operation_timestamps[operation_name].append(time.time())
        self.operation_counts[operation_name] += 1
        self.batch_sizes[operation_name].append(batch_size)

        if not success and error:
            self.record_error(operation_name, error)

    def record_error(self, operation_name: str, error: Exception):
        """Record an error."""
        self.errors.append({
            'timestamp': time.time(),
            'operation': operation_name,
            'type': type(error).__name__,
            'message': str(error)
        })

    def record_custom_metric(self, name: str, value: Any):
        """Record a custom metric."""
        if name not in self.custom_metrics:
            self.custom_metrics[name] = []
        self.custom_metrics[name].append(value)

    async def start_resource_monitoring(self, interval: float = 1.0):
        """
        Start async resource monitoring.

        Args:
            interval: Sampling interval in seconds
        """
        async def monitor():
            process = psutil.Process()
            while True:
                try:
                    sample = {
                        'timestamp': time.time(),
                        'cpu_percent': process.cpu_percent(),
                        'memory_mb': process.memory_info().rss / 1024 / 1024,
                        'memory_percent': process.memory_percent(),
                    }
                    self.resource_samples.append(sample)
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break

        self._resource_monitor_task = asyncio.create_task(monitor())

    async def stop_resource_monitoring(self):
        """Stop async resource monitoring."""
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
            try:
                await self._resource_monitor_task
            except asyncio.CancelledError:
                pass

    def get_duration(self) -> float:
        """Get total measurement duration."""
        return time.time() - self.start_time


# ============================================================================
# Performance Report
# ============================================================================


class PerformanceReport:
    """
    Generate comprehensive performance reports from collected metrics.
    """

    def __init__(self, collector: MetricsCollector):
        """
        Initialize performance report.

        Args:
            collector: MetricsCollector with collected data
        """
        self.collector = collector
        self.workflow_name = collector.workflow_name
        self.duration = collector.get_duration()

        # Calculate metrics
        self.latency_metrics = self._calculate_latency_metrics()
        self.throughput_metrics = self._calculate_throughput_metrics()
        self.resource_metrics = self._calculate_resource_metrics()
        self.error_metrics = self._calculate_error_metrics()

    def _calculate_latency_metrics(self) -> Dict[str, LatencyMetrics]:
        """Calculate latency metrics for all operations."""
        metrics = {}
        for op_name, timings in self.collector.operation_timings.items():
            if timings:
                # Convert to milliseconds
                timings_ms = [t * 1000 for t in timings]
                metrics[op_name] = LatencyMetrics.from_measurements(timings_ms)
        return metrics

    def _calculate_throughput_metrics(self) -> Dict[str, ThroughputMetrics]:
        """Calculate throughput metrics for all operations."""
        metrics = {}
        for op_name, timestamps in self.collector.operation_timestamps.items():
            if timestamps:
                relative_timestamps = [t - self.collector.start_time for t in timestamps]
                batch_sizes = self.collector.batch_sizes.get(op_name)
                metrics[op_name] = ThroughputMetrics.from_operations(
                    relative_timestamps,
                    self.duration,
                    batch_sizes
                )
        return metrics

    def _calculate_resource_metrics(self) -> Optional[ResourceMetrics]:
        """Calculate resource utilization metrics."""
        if not self.collector.resource_samples:
            return None
        return ResourceMetrics.from_samples(self.collector.resource_samples)

    def _calculate_error_metrics(self) -> ErrorMetrics:
        """Calculate error rate metrics."""
        total_ops = sum(self.collector.operation_counts.values())
        return ErrorMetrics.from_errors(total_ops, self.collector.errors)

    def summary(self) -> str:
        """Generate human-readable summary report."""
        lines = [
            f"Performance Report: {self.workflow_name}",
            "=" * 80,
            f"Duration: {self.duration:.2f}s",
            "",
            "Latency Metrics:",
            "-" * 80,
        ]

        for op_name, metrics in self.latency_metrics.items():
            lines.extend([
                f"\n{op_name}:",
                f"  Count: {metrics.count}",
                f"  Mean: {metrics.mean:.2f}ms",
                f"  p50: {metrics.p50:.2f}ms",
                f"  p95: {metrics.p95:.2f}ms",
                f"  p99: {metrics.p99:.2f}ms",
                f"  p99.9: {metrics.p99_9:.2f}ms",
            ])

        lines.extend([
            "",
            "Throughput Metrics:",
            "-" * 80,
        ])

        for op_name, metrics in self.throughput_metrics.items():
            lines.extend([
                f"\n{op_name}:",
                f"  Total Operations: {metrics.total_operations}",
                f"  Ops/sec: {metrics.ops_per_second:.2f}",
                f"  Peak Ops/sec: {metrics.peak_ops_per_second:.2f}",
            ])

        if self.resource_metrics:
            lines.extend([
                "",
                "Resource Utilization:",
                "-" * 80,
                f"CPU Average: {self.resource_metrics.cpu_percent_avg:.2f}%",
                f"CPU Peak: {self.resource_metrics.cpu_percent_peak:.2f}%",
                f"Memory Average: {self.resource_metrics.memory_mb_avg:.2f} MB",
                f"Memory Peak: {self.resource_metrics.memory_mb_peak:.2f} MB",
            ])

        if self.error_metrics.total_errors > 0:
            lines.extend([
                "",
                "Error Metrics:",
                "-" * 80,
                f"Total Errors: {self.error_metrics.total_errors}",
                f"Error Rate: {self.error_metrics.error_rate * 100:.2f}%",
                f"Error Types: {self.error_metrics.error_types}",
            ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as dictionary."""
        return {
            'workflow_name': self.workflow_name,
            'duration': self.duration,
            'latency': {
                name: asdict(metrics)
                for name, metrics in self.latency_metrics.items()
            },
            'throughput': {
                name: asdict(metrics)
                for name, metrics in self.throughput_metrics.items()
            },
            'resources': asdict(self.resource_metrics) if self.resource_metrics else None,
            'errors': asdict(self.error_metrics)
        }

    def to_json(self, filepath: Optional[Path] = None) -> str:
        """
        Export metrics as JSON.

        Args:
            filepath: Optional file path to save JSON

        Returns:
            JSON string
        """
        json_data = json.dumps(self.to_dict(), indent=2)

        if filepath:
            filepath.write_text(json_data)

        return json_data

    def check_regression(
        self,
        baseline: Dict[str, Any],
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, List[str]]:
        """
        Check for performance regressions against baseline.

        Args:
            baseline: Baseline metrics dictionary
            thresholds: Custom thresholds for regression detection

        Returns:
            Dictionary of regressions found
        """
        if thresholds is None:
            thresholds = {
                'latency_p95': 1.2,  # 20% increase
                'latency_p99': 1.25,  # 25% increase
                'throughput': 0.85,  # 15% decrease
                'error_rate': 2.0,  # 2x increase
            }

        regressions = defaultdict(list)

        # Check latency regressions
        baseline_latency = baseline.get('latency', {})
        for op_name, metrics in self.latency_metrics.items():
            if op_name in baseline_latency:
                base_metrics = baseline_latency[op_name]

                if metrics.p95 > base_metrics['p95'] * thresholds['latency_p95']:
                    regressions['latency'].append(
                        f"{op_name} p95: {metrics.p95:.2f}ms "
                        f"(baseline: {base_metrics['p95']:.2f}ms)"
                    )

                if metrics.p99 > base_metrics['p99'] * thresholds['latency_p99']:
                    regressions['latency'].append(
                        f"{op_name} p99: {metrics.p99:.2f}ms "
                        f"(baseline: {base_metrics['p99']:.2f}ms)"
                    )

        # Check throughput regressions
        baseline_throughput = baseline.get('throughput', {})
        for op_name, metrics in self.throughput_metrics.items():
            if op_name in baseline_throughput:
                base_metrics = baseline_throughput[op_name]

                if metrics.ops_per_second < base_metrics['ops_per_second'] * thresholds['throughput']:
                    regressions['throughput'].append(
                        f"{op_name}: {metrics.ops_per_second:.2f} ops/sec "
                        f"(baseline: {base_metrics['ops_per_second']:.2f} ops/sec)"
                    )

        # Check error rate regressions
        baseline_errors = baseline.get('errors', {})
        if 'error_rate' in baseline_errors:
            if self.error_metrics.error_rate > baseline_errors['error_rate'] * thresholds['error_rate']:
                regressions['errors'].append(
                    f"Error rate: {self.error_metrics.error_rate * 100:.2f}% "
                    f"(baseline: {baseline_errors['error_rate'] * 100:.2f}%)"
                )

        return dict(regressions)
