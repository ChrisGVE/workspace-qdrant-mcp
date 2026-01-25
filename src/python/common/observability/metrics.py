"""
Metrics collection system for workspace-qdrant-mcp.

Provides comprehensive metrics collection compatible with Prometheus, StatsD, and other
monitoring systems. Tracks operational metrics, performance data, and business metrics
for production observability.

Key Metrics:
    - Request counts and rates (by operation, collection, status)
    - Response times and latency percentiles
    - Error rates and types
    - Resource utilization (memory, connections, queue lengths)
    - Business metrics (documents processed, search queries, embeddings generated)

Integration Support:
    - Prometheus exposition format
    - StatsD protocol for real-time metrics
    - Custom metrics exporters
    - Dashboard-ready metric structure

Example:
    ```python
    from workspace_qdrant_mcp.observability import metrics_instance, record_operation

    # Record operation metrics
    with record_operation("search", collection="my-project"):
        results = perform_search()

    # Manual metric recording
    metrics_instance.increment_counter("documents_processed", collection="docs")
    metrics_instance.record_histogram("embedding_generation_time", 0.45)
    ```
"""

import os
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import psutil
from loguru import logger

# logger imported from loguru


@dataclass
class MetricValue:
    """Container for a metric value with metadata."""

    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class HistogramBucket:
    """Histogram bucket for latency and size distributions."""

    le: float  # Less than or equal to
    count: int = 0


class Counter:
    """Thread-safe counter metric."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = Lock()
        self._labels_values: dict[str, float] = defaultdict(float)

    def increment(self, amount: float = 1.0, **labels):
        """Increment counter by amount with optional labels."""
        with self._lock:
            self._value += amount
            if labels:
                label_key = self._serialize_labels(labels)
                self._labels_values[label_key] += amount

    def get_value(self, **labels) -> float:
        """Get current counter value, optionally filtered by labels."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                return self._labels_values.get(label_key, 0.0)
            return self._value

    def get_all_labeled_values(self) -> dict[str, float]:
        """Get all labeled counter values."""
        with self._lock:
            return dict(self._labels_values)

    def _serialize_labels(self, labels: dict[str, Any]) -> str:
        """Serialize labels to a consistent string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class Gauge:
    """Thread-safe gauge metric for current values."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = Lock()
        self._labels_values: dict[str, float] = defaultdict(float)

    def set(self, value: float, **labels):
        """Set gauge to a specific value with optional labels."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                self._labels_values[label_key] = value
            else:
                self._value = value

    def add(self, amount: float, **labels):
        """Add amount to gauge with optional labels."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                self._labels_values[label_key] += amount
            else:
                self._value += amount

    def subtract(self, amount: float, **labels):
        """Subtract amount from gauge with optional labels."""
        self.add(-amount, **labels)

    def get_value(self, **labels) -> float:
        """Get current gauge value, optionally filtered by labels."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                return self._labels_values.get(label_key, 0.0)
            return self._value

    def get_all_labeled_values(self) -> dict[str, float]:
        """Get all labeled gauge values."""
        with self._lock:
            return dict(self._labels_values)

    def _serialize_labels(self, labels: dict[str, Any]) -> str:
        """Serialize labels to a consistent string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class Histogram:
    """Thread-safe histogram for tracking distributions."""

    # Default buckets for latency measurements (seconds)
    DEFAULT_LATENCY_BUCKETS = [
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    ]

    def __init__(
        self, name: str, description: str = "", buckets: list[float] | None = None
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_LATENCY_BUCKETS
        self._lock = Lock()

        # Initialize buckets
        self._histogram_buckets: dict[float, int] = dict.fromkeys(self.buckets, 0)
        self._histogram_buckets[float("inf")] = 0  # +Inf bucket

        # Track sum and count for average calculation
        self._sum = 0.0
        self._count = 0

        # Labeled histograms
        self._labeled_histograms: dict[str, dict[float, int]] = defaultdict(
            lambda: dict.fromkeys(self.buckets + [float("inf")], 0)
        )
        self._labeled_sums: dict[str, float] = defaultdict(float)
        self._labeled_counts: dict[str, int] = defaultdict(int)

    def observe(self, value: float, **labels):
        """Record an observation in the histogram."""
        with self._lock:
            # Update buckets
            for bucket in self.buckets:
                if value <= bucket:
                    if labels:
                        label_key = self._serialize_labels(labels)
                        self._labeled_histograms[label_key][bucket] += 1
                    else:
                        self._histogram_buckets[bucket] += 1

            # Always increment +Inf bucket
            if labels:
                label_key = self._serialize_labels(labels)
                self._labeled_histograms[label_key][float("inf")] += 1
                self._labeled_sums[label_key] += value
                self._labeled_counts[label_key] += 1
            else:
                self._histogram_buckets[float("inf")] += 1
                self._sum += value
                self._count += 1

    def get_buckets(self, **labels) -> dict[float, int]:
        """Get histogram buckets, optionally filtered by labels."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                return dict(self._labeled_histograms.get(label_key, {}))
            return dict(self._histogram_buckets)

    def get_count(self, **labels) -> int:
        """Get total number of observations."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                return self._labeled_counts.get(label_key, 0)
            return self._count

    def get_sum(self, **labels) -> float:
        """Get sum of all observations."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                return self._labeled_sums.get(label_key, 0.0)
            return self._sum

    def get_average(self, **labels) -> float:
        """Get average of all observations."""
        count = self.get_count(**labels)
        if count == 0:
            return 0.0
        return self.get_sum(**labels) / count

    def _serialize_labels(self, labels: dict[str, Any]) -> str:
        """Serialize labels to a consistent string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class MetricsCollector:
    """Main metrics collection and management system."""

    def __init__(self):
        self.counters: dict[str, Counter] = {}
        self.gauges: dict[str, Gauge] = {}
        self.histograms: dict[str, Histogram] = {}
        self._lock = Lock()

        # System metrics tracking
        self._system_metrics_enabled = True
        self._last_system_update = 0

        # Active operations tracking
        self._active_operations: dict[str, float] = {}  # operation_id -> start_time

        # Initialize standard metrics
        self._initialize_standard_metrics()

        # Only log initialization if explicitly requested or in server mode

        if (
            os.getenv("WQM_LOG_INIT", "false").lower() == "true"
            or os.getenv("WQM_SERVER_MODE", "false").lower() == "true"
        ):
            logger.info("MetricsCollector initialized", metrics_enabled=True)

    def _initialize_standard_metrics(self):
        """Initialize standard application metrics."""
        # Request metrics
        self.create_counter("requests_total", "Total number of requests")
        self.create_counter("requests_errors_total", "Total number of request errors")

        # Operation metrics
        self.create_histogram(
            "operation_duration_seconds", "Duration of operations in seconds"
        )
        self.create_counter("operations_total", "Total number of operations")

        # Search metrics
        self.create_histogram("search_duration_seconds", "Search operation duration")
        self.create_counter("search_queries_total", "Total search queries")
        self.create_histogram(
            "search_results_count", "Number of search results returned"
        )

        # Document metrics
        self.create_counter("documents_processed_total", "Total documents processed")
        self.create_histogram("document_size_bytes", "Size of processed documents")
        self.create_counter("embeddings_generated_total", "Total embeddings generated")

        # Collection metrics
        self.create_gauge("collections_active", "Number of active collections")
        self.create_gauge(
            "collection_points_total", "Total points across all collections"
        )

        # System metrics
        self.create_gauge("memory_usage_bytes", "Memory usage in bytes")
        self.create_gauge("cpu_usage_percent", "CPU usage percentage")
        self.create_gauge("active_connections", "Number of active connections")

        # Watch system metrics
        self.create_counter("watch_events_total", "Total file watch events")
        self.create_gauge("watches_active", "Number of active file watches")
        self.create_counter("watch_errors_total", "Total watch system errors")

    def create_counter(self, name: str, description: str = "") -> Counter:
        """Create or get existing counter metric."""
        with self._lock:
            if name not in self.counters:
                self.counters[name] = Counter(name, description)
                logger.debug("Counter created", metric_name=name)
            return self.counters[name]

    def create_gauge(self, name: str, description: str = "") -> Gauge:
        """Create or get existing gauge metric."""
        with self._lock:
            if name not in self.gauges:
                self.gauges[name] = Gauge(name, description)
                logger.debug("Gauge created", metric_name=name)
            return self.gauges[name]

    def create_histogram(
        self, name: str, description: str = "", buckets: list[float] | None = None
    ) -> Histogram:
        """Create or get existing histogram metric."""
        with self._lock:
            if name not in self.histograms:
                self.histograms[name] = Histogram(name, description, buckets)
                logger.debug("Histogram created", metric_name=name)
            return self.histograms[name]

    def increment_counter(self, name: str, amount: float = 1.0, **labels):
        """Increment a counter metric."""
        counter = self.create_counter(name)
        counter.increment(amount, **labels)

    def set_gauge(self, name: str, value: float, **labels):
        """Set a gauge metric value."""
        gauge = self.create_gauge(name)
        gauge.set(value, **labels)

    def add_gauge(self, name: str, amount: float, **labels):
        """Add to a gauge metric value."""
        gauge = self.create_gauge(name)
        gauge.add(amount, **labels)

    def record_histogram(self, name: str, value: float, **labels):
        """Record a value in a histogram."""
        histogram = self.create_histogram(name)
        histogram.observe(value, **labels)

    @contextmanager
    def timer(self, metric_name: str, **labels):
        """Context manager for timing operations and recording to histogram."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record_histogram(metric_name, duration, **labels)

    def start_operation(self, operation_id: str, operation_type: str, **context):
        """Start tracking an operation."""
        start_time = time.perf_counter()
        self._active_operations[operation_id] = start_time

        # Record operation start
        self.increment_counter("operations_total", operation=operation_type, **context)
        self.add_gauge("active_operations", 1, operation=operation_type)

        logger.debug(
            "Operation started",
            operation_id=operation_id,
            operation_type=operation_type,
            **context,
        )

    def complete_operation(
        self, operation_id: str, operation_type: str, success: bool = True, **context
    ):
        """Complete tracking an operation."""
        if operation_id not in self._active_operations:
            logger.warning(
                "Operation not found for completion", operation_id=operation_id
            )
            return

        start_time = self._active_operations.pop(operation_id)
        duration = time.perf_counter() - start_time

        # Record completion metrics
        self.record_histogram(
            "operation_duration_seconds", duration, operation=operation_type, **context
        )
        self.add_gauge("active_operations", -1, operation=operation_type)

        if not success:
            self.increment_counter(
                "requests_errors_total", operation=operation_type, **context
            )

        logger.debug(
            "Operation completed",
            operation_id=operation_id,
            operation_type=operation_type,
            duration_seconds=duration,
            success=success,
            **context,
        )

    def update_system_metrics(self):
        """Update system resource metrics."""
        if not self._system_metrics_enabled:
            return

        current_time = time.time()
        if current_time - self._last_system_update < 1.0:  # Update max once per second
            return

        try:
            # Memory metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            self.set_gauge("memory_usage_bytes", memory_info.rss)

            # CPU metrics
            cpu_percent = process.cpu_percent()
            self.set_gauge("cpu_usage_percent", cpu_percent)

            # Connection metrics (approximation via file descriptors)
            try:
                num_fds = process.num_fds()
                self.set_gauge("active_connections", num_fds)
            except (AttributeError, OSError):
                # num_fds not available on all platforms
                pass

            self._last_system_update = current_time

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning("Failed to update system metrics", error=str(e))

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of all current metrics."""
        summary = {
            "timestamp": time.time(),
            "counters": {},
            "gauges": {},
            "histograms": {},
        }

        # Update system metrics before reporting
        self.update_system_metrics()

        # Collect counter values
        for name, counter in self.counters.items():
            summary["counters"][name] = {
                "value": counter.get_value(),
                "labeled_values": counter.get_all_labeled_values(),
                "description": counter.description,
            }

        # Collect gauge values
        for name, gauge in self.gauges.items():
            summary["gauges"][name] = {
                "value": gauge.get_value(),
                "labeled_values": gauge.get_all_labeled_values(),
                "description": gauge.description,
            }

        # Collect histogram values
        for name, histogram in self.histograms.items():
            summary["histograms"][name] = {
                "count": histogram.get_count(),
                "sum": histogram.get_sum(),
                "average": histogram.get_average(),
                "buckets": histogram.get_buckets(),
                "description": histogram.description,
            }

        return summary

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus exposition format."""
        lines = []

        # Add metadata
        lines.append("# Workspace Qdrant MCP Metrics")
        lines.append(f"# Generated at {time.time()}")
        lines.append("")

        # Export counters
        for name, counter in self.counters.items():
            if counter.description:
                lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")

            # Main counter
            lines.append(f"{name} {counter.get_value()}")

            # Labeled counters
            for label_str, value in counter.get_all_labeled_values().items():
                labels = "{" + label_str.replace("=", '="').replace(",", '", ') + '"}'
                lines.append(f"{name}{labels} {value}")
            lines.append("")

        # Export gauges
        for name, gauge in self.gauges.items():
            if gauge.description:
                lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")

            lines.append(f"{name} {gauge.get_value()}")

            for label_str, value in gauge.get_all_labeled_values().items():
                labels = "{" + label_str.replace("=", '="').replace(",", '", ') + '"}'
                lines.append(f"{name}{labels} {value}")
            lines.append("")

        # Export histograms
        for name, histogram in self.histograms.items():
            if histogram.description:
                lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")

            # Buckets
            buckets = histogram.get_buckets()
            for bucket_value, count in buckets.items():
                if bucket_value == float("inf"):
                    lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                else:
                    lines.append(f'{name}_bucket{{le="{bucket_value}"}} {count}')

            lines.append(f"{name}_sum {histogram.get_sum()}")
            lines.append(f"{name}_count {histogram.get_count()}")
            lines.append("")

        return "\n".join(lines)

    def reset_metrics(self):
        """Reset all metrics to initial state."""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self._active_operations.clear()
            self._initialize_standard_metrics()
            logger.info("All metrics reset")


# Global metrics instance (lazy initialization)
_metrics_instance: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance with lazy initialization.

    Returns:
        The global MetricsCollector instance, created on first access.
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


class LazyMetricsCollector:
    """Lazy proxy for metrics collector instance."""

    def __getattr__(self, name):
        return getattr(get_metrics_collector(), name)

    def __call__(self, *args, **kwargs):
        return get_metrics_collector()(*args, **kwargs)


# Maintain API compatibility with lazy initialization
metrics_instance = LazyMetricsCollector()


@contextmanager
def record_operation(operation_type: str, **context):
    """Context manager for recording operation metrics.

    Args:
        operation_type: Type of operation (e.g., 'search', 'ingest', 'embed')
        **context: Additional context labels

    Example:
        ```python
        with record_operation("search", collection="my-project", mode="hybrid"):
            results = perform_search()
        ```
    """
    operation_id = f"{operation_type}_{int(time.time() * 1000000)}"

    metrics_instance.start_operation(operation_id, operation_type, **context)

    try:
        yield
        metrics_instance.complete_operation(
            operation_id, operation_type, True, **context
        )
    except Exception as e:
        metrics_instance.complete_operation(
            operation_id, operation_type, False, error_type=type(e).__name__, **context
        )
        raise


# =====================================================================
# MCP Tool Metrics (Task 412.10)
# =====================================================================

def _initialize_tool_metrics(collector: MetricsCollector) -> None:
    """Initialize MCP tool-specific metrics.

    Creates:
        - wqm_tool_calls_total: Counter for tool call counts
        - wqm_tool_duration_seconds: Histogram for tool durations
        - wqm_search_scope_total: Counter for search scopes
        - wqm_search_results: Histogram for search result counts
        - wqm_search_latency_seconds: Histogram for search latency
    """
    collector.create_counter(
        "wqm_tool_calls_total",
        "Total MCP tool calls by tool name and status"
    )
    collector.create_histogram(
        "wqm_tool_duration_seconds",
        "MCP tool call duration in seconds",
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
    )
    collector.create_counter(
        "wqm_search_scope_total",
        "Search scope counts (project, global, all)"
    )
    collector.create_histogram(
        "wqm_search_results",
        "Number of search results returned",
        buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]
    )
    collector.create_histogram(
        "wqm_search_latency_seconds",
        "Search latency by scope",
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )


# Initialize tool metrics when metrics collector is created
_tool_metrics_initialized = False


def _ensure_tool_metrics() -> None:
    """Ensure tool metrics are initialized (once)."""
    global _tool_metrics_initialized
    if not _tool_metrics_initialized:
        _initialize_tool_metrics(get_metrics_collector())
        _tool_metrics_initialized = True


def record_tool_call(tool_name: str, status: str, duration_seconds: float) -> None:
    """Record an MCP tool call metric.

    Args:
        tool_name: Name of the tool (store, search, manage, retrieve)
        status: Call status ("success" or "error")
        duration_seconds: Duration of the call in seconds
    """
    _ensure_tool_metrics()
    metrics = get_metrics_collector()
    metrics.increment_counter("wqm_tool_calls_total", tool_name=tool_name, status=status)
    metrics.record_histogram("wqm_tool_duration_seconds", duration_seconds, tool_name=tool_name)


def record_search_scope(scope: str) -> None:
    """Record a search scope metric.

    Args:
        scope: Search scope ("project", "global", "all")
    """
    _ensure_tool_metrics()
    get_metrics_collector().increment_counter("wqm_search_scope_total", scope=scope)


def record_search_results(scope: str, result_count: int, latency_seconds: float) -> None:
    """Record search result metrics.

    Args:
        scope: Search scope ("project", "global", "all")
        result_count: Number of results returned
        latency_seconds: Search latency in seconds
    """
    _ensure_tool_metrics()
    metrics = get_metrics_collector()
    metrics.record_histogram("wqm_search_results", result_count, scope=scope)
    metrics.record_histogram("wqm_search_latency_seconds", latency_seconds, scope=scope)


async def _async_tool_wrapper(tool_name: str, coro):
    """Async wrapper for recording tool metrics.

    Internal function - use track_tool decorator instead.
    """
    start_time = time.perf_counter()
    try:
        result = await coro
        duration = time.perf_counter() - start_time
        record_tool_call(tool_name, "success", duration)
        return result
    except Exception as e:
        duration = time.perf_counter() - start_time
        record_tool_call(tool_name, "error", duration)
        raise


def track_tool(tool_name: str, operation_callback=None):
    """Decorator for tracking MCP tool call metrics.

    Task 412.10: Wraps async tool functions to record:
    - wqm_tool_calls_total{tool_name, status}
    - wqm_tool_duration_seconds{tool_name}

    Task 452: Enhanced to track logical success/failure from return values.
    If the result is a dict with a 'success' key, tracks logical success/failure.

    Args:
        tool_name: Name of the tool being tracked
        operation_callback: Optional callback(success: bool) for operation tracking

    Example:
        ```python
        @track_tool("store")
        async def store(content: str, ...) -> dict:
            ...
        ```
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration = time.perf_counter() - start_time

                # Task 452: Check for logical success/failure in return value
                logical_success = True
                if isinstance(result, dict):
                    logical_success = result.get("success", True)

                status = "success" if logical_success else "logical_failure"
                record_tool_call(tool_name, status, duration)

                # Task 452: Call operation callback if provided
                if operation_callback:
                    try:
                        operation_callback(logical_success)
                    except Exception:
                        pass  # Don't let callback failures affect tool result

                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                record_tool_call(tool_name, "error", duration)

                # Task 452: Call operation callback on exception
                if operation_callback:
                    try:
                        operation_callback(False)
                    except Exception:
                        pass

                raise
        return wrapper
    return decorator


def get_tool_metrics_summary() -> dict[str, Any]:
    """Get a summary of MCP tool metrics.

    Returns:
        Dict with tool calls, durations, and search metrics
    """
    _ensure_tool_metrics()
    metrics = get_metrics_collector()

    return {
        "tool_calls": metrics.counters.get("wqm_tool_calls_total", Counter("wqm_tool_calls_total")).get_all_labeled_values(),
        "tool_durations": {
            "count": metrics.histograms.get("wqm_tool_duration_seconds", Histogram("wqm_tool_duration_seconds")).get_count(),
            "average": metrics.histograms.get("wqm_tool_duration_seconds", Histogram("wqm_tool_duration_seconds")).get_average(),
        },
        "search_scopes": metrics.counters.get("wqm_search_scope_total", Counter("wqm_search_scope_total")).get_all_labeled_values(),
        "search_results": {
            "count": metrics.histograms.get("wqm_search_results", Histogram("wqm_search_results")).get_count(),
            "average": metrics.histograms.get("wqm_search_results", Histogram("wqm_search_results")).get_average(),
        },
        "search_latency": {
            "count": metrics.histograms.get("wqm_search_latency_seconds", Histogram("wqm_search_latency_seconds")).get_count(),
            "average": metrics.histograms.get("wqm_search_latency_seconds", Histogram("wqm_search_latency_seconds")).get_average(),
        },
    }
