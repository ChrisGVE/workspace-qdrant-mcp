"""
Performance Metrics Collection System for Workspace Qdrant MCP.

This module provides comprehensive performance metrics collection for daemon instances,
including file processing rates, search performance, LSP operation timing, and
system resource utilization with trend analysis capabilities.
"""

import asyncio
import json
from common.logging import get_logger
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from weakref import WeakSet

import psutil

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""
    
    # Resource metrics
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    
    # Processing metrics
    FILE_PROCESSING_RATE = "file_processing_rate"
    DOCUMENT_INGESTION_TIME = "document_ingestion_time"
    BATCH_PROCESSING_TIME = "batch_processing_time"
    
    # Search metrics
    SEARCH_LATENCY = "search_latency"
    SEARCH_THROUGHPUT = "search_throughput"
    QUERY_COMPLEXITY = "query_complexity"
    RESULT_RELEVANCE = "result_relevance"
    
    # LSP metrics
    LSP_REQUEST_LATENCY = "lsp_request_latency"
    LSP_RESPONSE_SIZE = "lsp_response_size"
    LSP_CONNECTION_HEALTH = "lsp_connection_health"
    LSP_ERROR_RATE = "lsp_error_rate"
    
    # Daemon metrics
    DAEMON_STARTUP_TIME = "daemon_startup_time"
    DAEMON_HEALTH_CHECK_TIME = "daemon_health_check_time"
    DAEMON_RESTART_COUNT = "daemon_restart_count"
    
    # Collection metrics
    COLLECTION_SIZE = "collection_size"
    COLLECTION_OPERATIONS = "collection_operations"
    EMBEDDING_GENERATION_TIME = "embedding_generation_time"


class PerformanceLevel(Enum):
    """Performance levels for categorization."""
    
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    
    timestamp: datetime
    metric_type: MetricType
    value: float
    unit: str
    project_id: str
    operation_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric_type": self.metric_type.value,
            "value": self.value,
            "unit": self.unit,
            "project_id": self.project_id,
            "operation_id": self.operation_id,
            "context": self.context,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceMetric":
        """Create from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metric_type=MetricType(data["metric_type"]),
            value=data["value"],
            unit=data["unit"],
            project_id=data["project_id"],
            operation_id=data.get("operation_id"),
            context=data.get("context", {}),
            tags=data.get("tags", [])
        )


@dataclass
class MetricSummary:
    """Statistical summary of metrics over a time period."""
    
    metric_type: MetricType
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    percentile_95: float
    percentile_99: float
    time_range: Tuple[datetime, datetime]
    performance_level: PerformanceLevel
    trend: str  # "improving", "degrading", "stable"


@dataclass
class OperationTrace:
    """Trace of a complete operation for performance analysis."""
    
    operation_id: str
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    project_id: str = ""
    metrics: List[PerformanceMetric] = field(default_factory=list)
    status: str = "in_progress"  # in_progress, completed, failed
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    def add_metric(self, metric_type: MetricType, value: float, unit: str, **kwargs):
        """Add a metric to this operation trace."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            unit=unit,
            project_id=self.project_id,
            operation_id=self.operation_id,
            context=kwargs.get("context", {}),
            tags=kwargs.get("tags", [])
        )
        self.metrics.append(metric)
    
    def complete(self, status: str = "completed", error: Optional[str] = None):
        """Mark operation as complete."""
        self.end_time = datetime.now()
        self.status = status
        self.error = error


class MetricsBuffer:
    """Thread-safe buffer for collecting metrics."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        
    async def add(self, metric: PerformanceMetric):
        """Add a metric to the buffer."""
        async with self._lock:
            self._buffer.append(metric)
    
    async def get_metrics(
        self, 
        metric_type: Optional[MetricType] = None,
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """Get metrics from buffer with filtering."""
        async with self._lock:
            metrics = list(self._buffer)
        
        # Apply filters
        if metric_type:
            metrics = [m for m in metrics if m.metric_type == metric_type]
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        # Sort by timestamp (newest first)
        metrics.sort(key=lambda m: m.timestamp, reverse=True)
        
        if limit:
            metrics = metrics[:limit]
        
        return metrics
    
    async def clear(self):
        """Clear the buffer."""
        async with self._lock:
            self._buffer.clear()
    
    async def get_size(self) -> int:
        """Get current buffer size."""
        async with self._lock:
            return len(self._buffer)


class PerformanceProfiler:
    """Context manager for profiling operations."""
    
    def __init__(
        self, 
        operation_type: str, 
        project_id: str,
        collector: "PerformanceMetricsCollector",
        context: Optional[Dict[str, Any]] = None
    ):
        self.operation_type = operation_type
        self.project_id = project_id
        self.collector = collector
        self.context = context or {}
        self.operation_id = f"{operation_type}_{int(time.time() * 1000)}"
        self.trace: Optional[OperationTrace] = None
    
    async def __aenter__(self) -> OperationTrace:
        """Start operation profiling."""
        self.trace = OperationTrace(
            operation_id=self.operation_id,
            operation_type=self.operation_type,
            start_time=datetime.now(),
            project_id=self.project_id,
            context=self.context
        )
        
        # Record initial resource state
        try:
            process = psutil.Process()
            self.trace.add_metric(
                MetricType.MEMORY_USAGE,
                process.memory_info().rss / (1024 * 1024),  # MB
                "MB",
                context={"phase": "start"}
            )
            self.trace.add_metric(
                MetricType.CPU_USAGE,
                process.cpu_percent(),
                "%",
                context={"phase": "start"}
            )
        except Exception as e:
            logger.debug(f"Failed to collect initial metrics: {e}")
        
        return self.trace
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Complete operation profiling."""
        if not self.trace:
            return
        
        # Record final resource state
        try:
            process = psutil.Process()
            self.trace.add_metric(
                MetricType.MEMORY_USAGE,
                process.memory_info().rss / (1024 * 1024),  # MB
                "MB",
                context={"phase": "end"}
            )
            self.trace.add_metric(
                MetricType.CPU_USAGE,
                process.cpu_percent(),
                "%",
                context={"phase": "end"}
            )
        except Exception as e:
            logger.debug(f"Failed to collect final metrics: {e}")
        
        # Complete the trace
        status = "failed" if exc_type else "completed"
        error = str(exc_val) if exc_val else None
        self.trace.complete(status, error)
        
        # Submit to collector
        await self.collector.record_operation_trace(self.trace)


class PerformanceMetricsCollector:
    """Main performance metrics collection system."""
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.buffer = MetricsBuffer()
        self.operation_traces: Dict[str, OperationTrace] = {}
        self.active_operations: Set[str] = set()
        
        # Metric-specific collectors
        self._collection_start_times: Dict[str, float] = {}
        self._last_disk_io: Optional[psutil._common.sdiskio] = None
        self._last_network_io: Optional[psutil._common.snetio] = None
        self._last_disk_io_time: Optional[float] = None
        self._last_network_io_time: Optional[float] = None
        
        # Callbacks for metric events
        self._metric_callbacks: List[Callable[[PerformanceMetric], None]] = []
        self._operation_callbacks: List[Callable[[OperationTrace], None]] = []
        
        # Performance thresholds
        self.thresholds = {
            MetricType.SEARCH_LATENCY: {
                PerformanceLevel.EXCELLENT: 50,  # ms
                PerformanceLevel.GOOD: 100,
                PerformanceLevel.AVERAGE: 250,
                PerformanceLevel.POOR: 500,
            },
            MetricType.FILE_PROCESSING_RATE: {
                PerformanceLevel.EXCELLENT: 100,  # files/min
                PerformanceLevel.GOOD: 50,
                PerformanceLevel.AVERAGE: 25,
                PerformanceLevel.POOR: 10,
            },
            MetricType.LSP_REQUEST_LATENCY: {
                PerformanceLevel.EXCELLENT: 20,  # ms
                PerformanceLevel.GOOD: 50,
                PerformanceLevel.AVERAGE: 100,
                PerformanceLevel.POOR: 200,
            }
        }
    
    def add_metric_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add callback for metric events."""
        self._metric_callbacks.append(callback)
    
    def add_operation_callback(self, callback: Callable[[OperationTrace], None]):
        """Add callback for operation completion events."""
        self._operation_callbacks.append(callback)
    
    async def record_metric(
        self,
        metric_type: MetricType,
        value: float,
        unit: str,
        operation_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ):
        """Record a performance metric."""
        metric = PerformanceMetric(
            timestamp=datetime.now(),
            metric_type=metric_type,
            value=value,
            unit=unit,
            project_id=self.project_id,
            operation_id=operation_id,
            context=context or {},
            tags=tags or []
        )
        
        await self.buffer.add(metric)
        
        # Notify callbacks
        for callback in self._metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                logger.warning(f"Metric callback failed: {e}")
    
    async def record_operation_trace(self, trace: OperationTrace):
        """Record a completed operation trace."""
        self.operation_traces[trace.operation_id] = trace
        self.active_operations.discard(trace.operation_id)
        
        # Add all trace metrics to buffer
        for metric in trace.metrics:
            await self.buffer.add(metric)
        
        # Record operation duration if completed
        if trace.duration is not None:
            await self.record_metric(
                MetricType.BATCH_PROCESSING_TIME,
                trace.duration,
                "seconds",
                operation_id=trace.operation_id,
                context={"operation_type": trace.operation_type, "status": trace.status}
            )
        
        # Notify callbacks
        for callback in self._operation_callbacks:
            try:
                callback(trace)
            except Exception as e:
                logger.warning(f"Operation callback failed: {e}")
    
    def profile_operation(
        self, 
        operation_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> PerformanceProfiler:
        """Create a performance profiler for an operation."""
        return PerformanceProfiler(operation_type, self.project_id, self, context)
    
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
            "query_length": len(query),
            "result_count": result_count,
            "query_hash": hash(query) % 10000
        }
        
        await self.record_metric(
            MetricType.SEARCH_LATENCY,
            latency_ms,
            "ms",
            operation_id=operation_id,
            context=context
        )
        
        await self.record_metric(
            MetricType.SEARCH_THROUGHPUT,
            result_count / (latency_ms / 1000) if latency_ms > 0 else 0,
            "results/second",
            operation_id=operation_id,
            context=context
        )
        
        if relevance_score is not None:
            await self.record_metric(
                MetricType.RESULT_RELEVANCE,
                relevance_score,
                "score",
                operation_id=operation_id,
                context=context
            )
    
    async def record_lsp_performance(
        self,
        method: str,
        latency_ms: float,
        response_size_bytes: int,
        success: bool,
        operation_id: Optional[str] = None
    ):
        """Record LSP operation performance metrics."""
        context = {
            "method": method,
            "response_size_bytes": response_size_bytes,
            "success": success
        }
        
        await self.record_metric(
            MetricType.LSP_REQUEST_LATENCY,
            latency_ms,
            "ms",
            operation_id=operation_id,
            context=context
        )
        
        await self.record_metric(
            MetricType.LSP_RESPONSE_SIZE,
            response_size_bytes,
            "bytes",
            operation_id=operation_id,
            context=context
        )
        
        if not success:
            await self.record_metric(
                MetricType.LSP_ERROR_RATE,
                1.0,
                "errors",
                operation_id=operation_id,
                context=context
            )
    
    async def record_file_processing(
        self,
        file_path: str,
        processing_time_seconds: float,
        file_size_bytes: int,
        success: bool,
        operation_id: Optional[str] = None
    ):
        """Record file processing performance metrics."""
        context = {
            "file_path": str(file_path),
            "file_size_bytes": file_size_bytes,
            "success": success,
            "file_extension": Path(file_path).suffix
        }
        
        await self.record_metric(
            MetricType.DOCUMENT_INGESTION_TIME,
            processing_time_seconds,
            "seconds",
            operation_id=operation_id,
            context=context
        )
        
        # Calculate processing rate
        if processing_time_seconds > 0:
            rate = file_size_bytes / processing_time_seconds / 1024  # KB/s
            await self.record_metric(
                MetricType.FILE_PROCESSING_RATE,
                rate,
                "KB/s",
                operation_id=operation_id,
                context=context
            )
    
    async def record_system_resources(self):
        """Record current system resource usage."""
        try:
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            await self.record_metric(
                MetricType.MEMORY_USAGE,
                memory_info.rss / (1024 * 1024),  # MB
                "MB"
            )
            
            # CPU usage
            cpu_percent = process.cpu_percent()
            await self.record_metric(
                MetricType.CPU_USAGE,
                cpu_percent,
                "%"
            )
            
            # Disk I/O
            disk_io = process.io_counters()
            current_time = time.time()
            
            if self._last_disk_io and self._last_disk_io_time:
                time_delta = current_time - self._last_disk_io_time
                if time_delta > 0:
                    read_rate = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_delta
                    write_rate = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_delta
                    
                    await self.record_metric(
                        MetricType.DISK_IO,
                        read_rate / 1024,  # KB/s
                        "KB/s",
                        context={"operation": "read"}
                    )
                    
                    await self.record_metric(
                        MetricType.DISK_IO,
                        write_rate / 1024,  # KB/s
                        "KB/s",
                        context={"operation": "write"}
                    )
            
            self._last_disk_io = disk_io
            self._last_disk_io_time = current_time
            
            # Network I/O
            try:
                net_io = psutil.net_io_counters()
                if self._last_network_io and time_delta > 0:
                    bytes_sent_rate = (net_io.bytes_sent - self._last_network_io.bytes_sent) / time_delta
                    bytes_recv_rate = (net_io.bytes_recv - self._last_network_io.bytes_recv) / time_delta
                    
                    await self.record_metric(
                        MetricType.NETWORK_IO,
                        bytes_sent_rate / 1024,  # KB/s
                        "KB/s",
                        context={"direction": "sent"}
                    )
                    
                    await self.record_metric(
                        MetricType.NETWORK_IO,
                        bytes_recv_rate / 1024,  # KB/s
                        "KB/s",
                        context={"direction": "received"}
                    )
                
                self._last_network_io = net_io
            except Exception as e:
                logger.debug(f"Failed to collect network I/O: {e}")
                
        except Exception as e:
            logger.warning(f"Failed to collect system resources: {e}")
    
    async def get_metric_summary(
        self,
        metric_type: MetricType,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> Optional[MetricSummary]:
        """Get statistical summary for a metric type."""
        # Default to last hour if no time range specified
        if since is None:
            since = datetime.now() - timedelta(hours=1)
        if until is None:
            until = datetime.now()
        
        # Get metrics from buffer
        metrics = await self.buffer.get_metrics(metric_type=metric_type, since=since)
        metrics = [m for m in metrics if m.timestamp <= until]
        
        if not metrics:
            return None
        
        values = [m.value for m in metrics]
        
        # Calculate statistics
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Calculate percentiles
        sorted_values = sorted(values)
        p95_index = int(0.95 * len(sorted_values))
        p99_index = int(0.99 * len(sorted_values))
        
        percentile_95 = sorted_values[min(p95_index, len(sorted_values) - 1)]
        percentile_99 = sorted_values[min(p99_index, len(sorted_values) - 1)]
        
        # Determine performance level
        performance_level = self._calculate_performance_level(metric_type, mean_val)
        
        # Determine trend (simple implementation)
        trend = self._calculate_trend(metrics)
        
        return MetricSummary(
            metric_type=metric_type,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=mean_val,
            median_value=median_val,
            std_dev=std_dev,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            time_range=(since, until),
            performance_level=performance_level,
            trend=trend
        )
    
    def _calculate_performance_level(self, metric_type: MetricType, value: float) -> PerformanceLevel:
        """Calculate performance level based on value and thresholds."""
        if metric_type not in self.thresholds:
            return PerformanceLevel.AVERAGE
        
        thresholds = self.thresholds[metric_type]
        
        if value <= thresholds.get(PerformanceLevel.EXCELLENT, float('inf')):
            return PerformanceLevel.EXCELLENT
        elif value <= thresholds.get(PerformanceLevel.GOOD, float('inf')):
            return PerformanceLevel.GOOD
        elif value <= thresholds.get(PerformanceLevel.AVERAGE, float('inf')):
            return PerformanceLevel.AVERAGE
        elif value <= thresholds.get(PerformanceLevel.POOR, float('inf')):
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.CRITICAL
    
    def _calculate_trend(self, metrics: List[PerformanceMetric]) -> str:
        """Calculate trend from metrics (simple linear regression)."""
        if len(metrics) < 3:
            return "stable"
        
        # Sort by timestamp
        metrics.sort(key=lambda m: m.timestamp)
        
        # Calculate simple trend
        values = [m.value for m in metrics]
        n = len(values)
        
        # Linear regression slope
        x_mean = n / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        # Classify trend
        if slope > 0.01:  # Arbitrary threshold
            return "degrading" if metrics[0].metric_type in [
                MetricType.SEARCH_LATENCY, 
                MetricType.LSP_REQUEST_LATENCY,
                MetricType.MEMORY_USAGE
            ] else "improving"
        elif slope < -0.01:
            return "improving" if metrics[0].metric_type in [
                MetricType.SEARCH_LATENCY, 
                MetricType.LSP_REQUEST_LATENCY,
                MetricType.MEMORY_USAGE
            ] else "degrading"
        else:
            return "stable"
    
    async def get_recent_operations(self, limit: int = 10) -> List[OperationTrace]:
        """Get recent operation traces."""
        traces = list(self.operation_traces.values())
        traces.sort(key=lambda t: t.start_time, reverse=True)
        return traces[:limit]
    
    async def cleanup_old_data(self, max_age_hours: int = 24):
        """Clean up old metrics and operation traces."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean operation traces
        old_operations = [
            op_id for op_id, trace in self.operation_traces.items()
            if trace.start_time < cutoff_time
        ]
        
        for op_id in old_operations:
            del self.operation_traces[op_id]
        
        logger.info(f"Cleaned up {len(old_operations)} old operation traces")