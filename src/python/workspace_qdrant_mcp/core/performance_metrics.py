"""
Performance Metrics Collection and Management System.

This module provides comprehensive performance metrics collection, buffering,
profiling, and analysis capabilities for the workspace-qdrant-mcp system.

Key Features:
    - Real-time metrics collection with low overhead
    - Operation profiling with context management
    - Metric buffering and aggregation
    - Performance level classification
    - Statistical analysis and trend detection

Task 265: Core performance metrics system for monitoring and optimization.
"""

import asyncio
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path

from loguru import logger
import psutil


class MetricType(Enum):
    """Types of performance metrics collected by the system."""

    # Search and query metrics
    SEARCH_LATENCY = "search_latency"
    SEARCH_THROUGHPUT = "search_throughput"
    SEARCH_ACCURACY = "search_accuracy"

    # System resource metrics
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_USAGE = "disk_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"

    # File processing metrics
    FILE_PROCESSING_RATE = "file_processing_rate"
    FILE_PROCESSING_LATENCY = "file_processing_latency"

    # LSP and language server metrics
    LSP_REQUEST_LATENCY = "lsp_request_latency"
    LSP_REQUEST_THROUGHPUT = "lsp_request_throughput"
    LSP_ERROR_RATE = "lsp_error_rate"

    # Collection and storage metrics
    COLLECTION_SIZE = "collection_size"
    INDEX_BUILD_TIME = "index_build_time"

    # Cache metrics
    CACHE_HIT_RATE = "cache_hit_rate"
    CACHE_SIZE = "cache_size"


class PerformanceLevel(Enum):
    """Performance level classification for metrics."""

    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""

    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    project_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    operation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "project_id": self.project_id,
            "context": self.context,
            "tags": self.tags,
            "operation_id": self.operation_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetric":
        """Create metric from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metric_type=MetricType(data["metric_type"]),
            value=data["value"],
            unit=data["unit"],
            project_id=data["project_id"],
            context=data.get("context", {}),
            tags=data.get("tags", []),
            operation_id=data.get("operation_id")
        )


@dataclass
class OperationTrace:
    """Trace of a complete operation with associated metrics."""

    operation_id: str
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime]
    project_id: str
    status: str = "in_progress"  # in_progress, completed, failed
    metrics: List[PerformanceMetric] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Calculate operation duration in seconds."""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def add_metric(self, metric_type: MetricType, value: float, unit: str, **context):
        """Add a metric to this operation trace."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            unit=unit,
            project_id=self.project_id,
            operation_id=self.operation_id,
            context=context
        )
        self.metrics.append(metric)

    def complete(self, status: str = "completed"):
        """Mark operation as complete."""
        self.end_time = datetime.now()
        self.status = status

    def fail(self, error_message: str):
        """Mark operation as failed."""
        self.end_time = datetime.now()
        self.status = "failed"
        self.error_message = error_message


@dataclass
class MetricSummary:
    """Statistical summary of metrics for a specific type and time period."""

    metric_type: MetricType
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    percentile_95: float
    percentile_99: float
    time_range: tuple[datetime, datetime]
    performance_level: PerformanceLevel
    trend: str = "stable"  # "improving", "degrading", "stable"

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary."""
        return {
            "metric_type": self.metric_type.value,
            "count": self.count,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "mean_value": self.mean_value,
            "median_value": self.median_value,
            "std_dev": self.std_dev,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "time_range": [self.time_range[0].isoformat(), self.time_range[1].isoformat()],
            "performance_level": self.performance_level.value,
            "trend": self.trend
        }


class MetricsBuffer:
    """Thread-safe buffer for storing and retrieving metrics."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque[PerformanceMetric] = deque(maxlen=max_size)
        self._metrics_by_type: Dict[MetricType, List[PerformanceMetric]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add_metric(self, metric: PerformanceMetric):
        """Add a metric to the buffer."""
        async with self._lock:
            self._buffer.append(metric)
            self._metrics_by_type[metric.metric_type].append(metric)

            # Maintain type-specific lists within size limits
            if len(self._metrics_by_type[metric.metric_type]) > self.max_size:
                self._metrics_by_type[metric.metric_type] = \
                    self._metrics_by_type[metric.metric_type][-self.max_size:]

    async def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """Retrieve metrics with optional filtering."""
        async with self._lock:
            if metric_type:
                metrics = list(self._metrics_by_type[metric_type])
            else:
                metrics = list(self._buffer)

            # Filter by time range
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_metrics.append(metric)
                metrics = filtered_metrics

            # Apply limit
            if limit:
                metrics = metrics[-limit:]

            return metrics

    async def get_metric_count(self, metric_type: Optional[MetricType] = None) -> int:
        """Get count of metrics in buffer."""
        async with self._lock:
            if metric_type:
                return len(self._metrics_by_type[metric_type])
            return len(self._buffer)

    async def clear(self):
        """Clear all metrics from buffer."""
        async with self._lock:
            self._buffer.clear()
            self._metrics_by_type.clear()


class PerformanceProfiler:
    """Context manager for profiling operations with automatic metric collection."""

    def __init__(
        self,
        operation_type: str,
        project_id: str,
        metrics_collector: "PerformanceMetricsCollector",
        context: Optional[Dict[str, Any]] = None
    ):
        self.operation_type = operation_type
        self.project_id = project_id
        self.metrics_collector = metrics_collector
        self.context = context or {}

        self.operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        self.trace: Optional[OperationTrace] = None

    async def __aenter__(self) -> OperationTrace:
        """Start profiling operation."""
        self.trace = OperationTrace(
            operation_id=self.operation_id,
            operation_type=self.operation_type,
            start_time=datetime.now(),
            end_time=None,
            project_id=self.project_id,
            context=self.context
        )

        logger.debug(f"Started profiling operation: {self.operation_id}")
        return self.trace

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Complete profiling operation."""
        if not self.trace:
            return

        if exc_type:
            # Operation failed
            self.trace.fail(str(exc_val))
            logger.warning(f"Operation {self.operation_id} failed: {exc_val}")
        else:
            # Operation completed successfully
            self.trace.complete("completed")
            logger.debug(f"Completed profiling operation: {self.operation_id}")

        # Add trace to the collector's operation traces
        self.metrics_collector.operation_traces.append(self.trace)

        # Also call the callback if one is set
        if hasattr(self.metrics_collector, '_operation_callback') and self.metrics_collector._operation_callback:
            self.metrics_collector._operation_callback(self.trace)


class PerformanceMetricsCollector:
    """Main performance metrics collection system."""

    def __init__(self, project_id: str, buffer_size: int = 10000):
        self.project_id = project_id
        self.buffer = MetricsBuffer(buffer_size)
        self.operation_traces: deque[OperationTrace] = deque(maxlen=1000)

        # Callbacks for metric and operation events
        self._metric_callbacks: List[Callable[[PerformanceMetric], None]] = []
        self._operation_callback: Optional[Callable[[OperationTrace], None]] = None

        logger.info(f"Initialized performance metrics collector for project: {project_id}")

    def add_metric_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add callback for when metrics are collected."""
        self._metric_callbacks.append(callback)

    def add_operation_callback(self, callback: Callable[[OperationTrace], None]):
        """Add callback for when operations complete."""
        self._operation_callback = callback

    async def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        unit: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        operation_id: Optional[str] = None
    ):
        """Record a performance metric."""
        # Validate and sanitize the value
        if value is None:
            raise ValueError("Metric value cannot be None")

        # Handle special float values
        if not isinstance(value, (int, float)):
            raise TypeError(f"Metric value must be numeric, got {type(value)}")

        # Convert inf and nan to finite values for storage
        if value == float('inf'):
            value = 1e12  # Large but finite value
        elif value == float('-inf'):
            value = -1e12
        elif value != value:  # NaN check
            value = 0.0  # Default to 0 for NaN

        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=float(value),
            unit=unit,
            project_id=self.project_id,
            context=context or {},
            tags=tags or [],
            operation_id=operation_id
        )

        await self.buffer.add_metric(metric)

        # Notify callbacks
        for callback in self._metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.error(f"Metric callback failed: {e}")

        logger.debug(f"Recorded metric: {metric_type.value} = {value} {unit}")

    async def record_search_performance(
        self,
        query: str,
        result_count: int,
        latency_ms: float,
        relevance_score: Optional[float] = None,
        operation_id: Optional[str] = None
    ):
        """Record search performance metrics."""
        context = {
            "query": query,
            "result_count": result_count,
            "relevance_score": relevance_score
        }

        # Record latency
        await self.record_metric(
            MetricType.SEARCH_LATENCY,
            latency_ms,
            "ms",
            context=context,
            operation_id=operation_id
        )

        # Calculate and record throughput (queries per second)
        if latency_ms > 0:
            throughput = 1000.0 / latency_ms  # Convert ms to qps
            await self.record_metric(
                MetricType.SEARCH_THROUGHPUT,
                throughput,
                "qps",
                context=context,
                operation_id=operation_id
            )

    async def record_lsp_performance(
        self,
        method: str,
        latency_ms: float,
        response_size_bytes: int,
        success: bool,
        operation_id: Optional[str] = None
    ):
        """Record LSP operation performance."""
        context = {
            "method": method,
            "response_size_bytes": response_size_bytes,
            "success": success
        }

        await self.record_metric(
            MetricType.LSP_REQUEST_LATENCY,
            latency_ms,
            "ms",
            context=context,
            operation_id=operation_id
        )

        # Record error rate (0.0 for success, 1.0 for failure)
        error_rate = 0.0 if success else 1.0
        await self.record_metric(
            MetricType.LSP_ERROR_RATE,
            error_rate,
            "rate",
            context=context,
            operation_id=operation_id
        )

    async def record_file_processing(
        self,
        file_path: str,
        processing_time_seconds: float,
        file_size_bytes: int,
        success: bool,
        operation_id: Optional[str] = None
    ):
        """Record file processing performance."""
        context = {
            "file_path": file_path,
            "file_size_bytes": file_size_bytes,
            "success": success
        }

        # Record processing latency
        await self.record_metric(
            MetricType.FILE_PROCESSING_LATENCY,
            processing_time_seconds * 1000,  # Convert to ms
            "ms",
            context=context,
            operation_id=operation_id
        )

        # Calculate and record processing rate
        if processing_time_seconds > 0:
            rate = 60.0 / processing_time_seconds  # Files per minute
            await self.record_metric(
                MetricType.FILE_PROCESSING_RATE,
                rate,
                "files/min",
                context=context,
                operation_id=operation_id
            )

    async def record_system_resources(self):
        """Record system resource metrics."""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            await self.record_metric(
                MetricType.MEMORY_USAGE,
                memory.used / (1024 * 1024),  # Convert to MB
                "MB",
                context={"total_mb": memory.total / (1024 * 1024)}
            )

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            await self.record_metric(
                MetricType.CPU_USAGE,
                cpu_percent,
                "%"
            )

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.record_metric(
                MetricType.DISK_USAGE,
                disk_percent,
                "%",
                context={"total_gb": disk.total / (1024**3)}
            )

        except Exception as e:
            logger.error(f"Failed to record system resources: {e}")
            raise

    def profile_operation(
        self,
        operation_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PerformanceProfiler:
        """Create a performance profiler for an operation."""
        return PerformanceProfiler(
            operation_type=operation_type,
            project_id=self.project_id,
            metrics_collector=self,
            context=context
        )

    async def get_metric_summary(
        self,
        metric_type: MetricType,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[MetricSummary]:
        """Calculate statistical summary for a metric type."""
        metrics = await self.buffer.get_metrics(
            metric_type=metric_type,
            start_time=start_time,
            end_time=end_time
        )

        if not metrics:
            return None

        values = [m.value for m in metrics]

        # Calculate statistics
        count = len(values)
        min_val = min(values)
        max_val = max(values)
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_dev = statistics.stdev(values) if count > 1 else 0.0

        # Calculate percentiles
        sorted_values = sorted(values)
        p95_idx = int(0.95 * len(sorted_values))
        p99_idx = int(0.99 * len(sorted_values))
        percentile_95 = sorted_values[p95_idx] if p95_idx < len(sorted_values) else max_val
        percentile_99 = sorted_values[p99_idx] if p99_idx < len(sorted_values) else max_val

        # Determine performance level
        performance_level = self._classify_performance_level(metric_type, mean_val)

        # Determine trend (simplified)
        trend = self._analyze_trend(values)

        time_range = (
            start_time or min(m.timestamp for m in metrics),
            end_time or max(m.timestamp for m in metrics)
        )

        return MetricSummary(
            metric_type=metric_type,
            count=count,
            min_value=min_val,
            max_value=max_val,
            mean_value=mean_val,
            median_value=median_val,
            std_dev=std_dev,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            time_range=time_range,
            performance_level=performance_level,
            trend=trend
        )

    def _classify_performance_level(self, metric_type: MetricType, value: float) -> PerformanceLevel:
        """Classify performance level based on metric type and value."""
        thresholds = {
            MetricType.SEARCH_LATENCY: {
                PerformanceLevel.EXCELLENT: 50,
                PerformanceLevel.GOOD: 100,
                PerformanceLevel.AVERAGE: 200,
                PerformanceLevel.POOR: 500,
            },
            MetricType.CPU_USAGE: {
                PerformanceLevel.EXCELLENT: 25,
                PerformanceLevel.GOOD: 50,
                PerformanceLevel.AVERAGE: 75,
                PerformanceLevel.POOR: 90,
            },
            MetricType.MEMORY_USAGE: {
                PerformanceLevel.EXCELLENT: 100,  # MB
                PerformanceLevel.GOOD: 200,
                PerformanceLevel.AVERAGE: 300,
                PerformanceLevel.POOR: 400,
            }
        }

        metric_thresholds = thresholds.get(metric_type)
        if not metric_thresholds:
            return PerformanceLevel.AVERAGE

        for level in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD,
                      PerformanceLevel.AVERAGE, PerformanceLevel.POOR]:
            if value <= metric_thresholds[level]:
                return level

        return PerformanceLevel.CRITICAL

    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend in metric values."""
        if len(values) < 5:
            return "stable"

        # Simple linear regression to detect trend
        n = len(values)
        x_vals = list(range(n))

        # Calculate slope
        x_mean = sum(x_vals) / n
        y_mean = sum(values) / n

        numerator = sum((x_vals[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Classify trend based on slope
        if slope > 0.1:
            return "degrading"  # Values increasing (bad for latency)
        elif slope < -0.1:
            return "improving"  # Values decreasing (good for latency)
        else:
            return "stable"

    async def get_recent_operations(self, limit: int = 10) -> List[OperationTrace]:
        """Get recent operation traces."""
        return list(self.operation_traces)[-limit:]

    async def clear_metrics(self):
        """Clear all collected metrics."""
        await self.buffer.clear()
        self.operation_traces.clear()
        logger.info(f"Cleared all metrics for project: {self.project_id}")


# Export main classes
__all__ = [
    "MetricType",
    "PerformanceLevel",
    "PerformanceMetric",
    "OperationTrace",
    "MetricSummary",
    "MetricsBuffer",
    "PerformanceProfiler",
    "PerformanceMetricsCollector"
]