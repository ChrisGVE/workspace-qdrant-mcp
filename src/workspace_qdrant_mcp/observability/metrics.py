
from ...observability import get_logger
logger = get_logger(__name__)
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

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Any, Set
import asyncio
import weakref
from collections import defaultdict, deque
import psutil
import os

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricValue:
    """Container for a metric value with metadata."""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
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
        self._labels_values: Dict[str, float] = defaultdict(float)
    
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
    
    def get_all_labeled_values(self) -> Dict[str, float]:
        """Get all labeled counter values."""
        with self._lock:
            return dict(self._labels_values)
    
    def _serialize_labels(self, labels: Dict[str, Any]) -> str:
        """Serialize labels to a consistent string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class Gauge:
    """Thread-safe gauge metric for current values."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = Lock()
        self._labels_values: Dict[str, float] = defaultdict(float)
    
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
    
    def get_all_labeled_values(self) -> Dict[str, float]:
        """Get all labeled gauge values."""
        with self._lock:
            return dict(self._labels_values)
    
    def _serialize_labels(self, labels: Dict[str, Any]) -> str:
        """Serialize labels to a consistent string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class Histogram:
    """Thread-safe histogram for tracking distributions."""
    
    # Default buckets for latency measurements (seconds)
    DEFAULT_LATENCY_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    def __init__(self, name: str, description: str = "", buckets: Optional[List[float]] = None):
        self.name = name
        self.description = description
        self.buckets = buckets or self.DEFAULT_LATENCY_BUCKETS
        self._lock = Lock()
        
        # Initialize buckets
        self._histogram_buckets: Dict[float, int] = {bucket: 0 for bucket in self.buckets}
        self._histogram_buckets[float('inf')] = 0  # +Inf bucket
        
        # Track sum and count for average calculation
        self._sum = 0.0
        self._count = 0
        
        # Labeled histograms
        self._labeled_histograms: Dict[str, Dict[float, int]] = defaultdict(
            lambda: {bucket: 0 for bucket in self.buckets + [float('inf')]}
        )
        self._labeled_sums: Dict[str, float] = defaultdict(float)
        self._labeled_counts: Dict[str, int] = defaultdict(int)
    
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
                self._labeled_histograms[label_key][float('inf')] += 1
                self._labeled_sums[label_key] += value
                self._labeled_counts[label_key] += 1
            else:
                self._histogram_buckets[float('inf')] += 1
                self._sum += value
                self._count += 1
    
    def get_buckets(self, **labels) -> Dict[float, int]:
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
    
    def _serialize_labels(self, labels: Dict[str, Any]) -> str:
        """Serialize labels to a consistent string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class MetricsCollector:
    """Main metrics collection and management system."""
    
    def __init__(self):
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self._lock = Lock()
        
        # System metrics tracking
        self._system_metrics_enabled = True
        self._last_system_update = 0
        
        # Active operations tracking
        self._active_operations: Dict[str, float] = {}  # operation_id -> start_time
        
        # Initialize standard metrics
        self._initialize_standard_metrics()
        
        logger.info("MetricsCollector initialized", metrics_enabled=True)
    
    def _initialize_standard_metrics(self):
        """Initialize standard application metrics."""
        # Request metrics
        self.create_counter("requests_total", "Total number of requests")
        self.create_counter("requests_errors_total", "Total number of request errors")
        
        # Operation metrics
        self.create_histogram("operation_duration_seconds", "Duration of operations in seconds")
        self.create_counter("operations_total", "Total number of operations")
        
        # Search metrics
        self.create_histogram("search_duration_seconds", "Search operation duration")
        self.create_counter("search_queries_total", "Total search queries")
        self.create_histogram("search_results_count", "Number of search results returned")
        
        # Document metrics
        self.create_counter("documents_processed_total", "Total documents processed")
        self.create_histogram("document_size_bytes", "Size of processed documents")
        self.create_counter("embeddings_generated_total", "Total embeddings generated")
        
        # Collection metrics
        self.create_gauge("collections_active", "Number of active collections")
        self.create_gauge("collection_points_total", "Total points across all collections")
        
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
    
    def create_histogram(self, name: str, description: str = "", 
                        buckets: Optional[List[float]] = None) -> Histogram:
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
        
        logger.debug("Operation started", 
                    operation_id=operation_id, 
                    operation_type=operation_type, 
                    **context)
    
    def complete_operation(self, operation_id: str, operation_type: str, 
                          success: bool = True, **context):
        """Complete tracking an operation."""
        if operation_id not in self._active_operations:
            logger.warning("Operation not found for completion", operation_id=operation_id)
            return
        
        start_time = self._active_operations.pop(operation_id)
        duration = time.perf_counter() - start_time
        
        # Record completion metrics
        self.record_histogram("operation_duration_seconds", duration, 
                             operation=operation_type, **context)
        self.add_gauge("active_operations", -1, operation=operation_type)
        
        if not success:
            self.increment_counter("requests_errors_total", 
                                 operation=operation_type, **context)
        
        logger.debug("Operation completed",
                    operation_id=operation_id,
                    operation_type=operation_type, 
                    duration_seconds=duration,
                    success=success,
                    **context)
    
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
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all current metrics."""
        summary = {
            "timestamp": time.time(),
            "counters": {},
            "gauges": {},
            "histograms": {}
        }
        
        # Update system metrics before reporting
        self.update_system_metrics()
        
        # Collect counter values
        for name, counter in self.counters.items():
            summary["counters"][name] = {
                "value": counter.get_value(),
                "labeled_values": counter.get_all_labeled_values(),
                "description": counter.description
            }
        
        # Collect gauge values
        for name, gauge in self.gauges.items():
            summary["gauges"][name] = {
                "value": gauge.get_value(),
                "labeled_values": gauge.get_all_labeled_values(),
                "description": gauge.description
            }
        
        # Collect histogram values
        for name, histogram in self.histograms.items():
            summary["histograms"][name] = {
                "count": histogram.get_count(),
                "sum": histogram.get_sum(),
                "average": histogram.get_average(),
                "buckets": histogram.get_buckets(),
                "description": histogram.description
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
                if bucket_value == float('inf'):
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


# Global metrics instance
metrics_instance = MetricsCollector()


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
        metrics_instance.complete_operation(operation_id, operation_type, True, **context)
    except Exception as e:
        metrics_instance.complete_operation(operation_id, operation_type, False, 
                                          error_type=type(e).__name__, **context)
        raise