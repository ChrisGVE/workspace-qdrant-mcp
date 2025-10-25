"""
Queue Performance Metrics Collection

Provides comprehensive performance metrics collection for the queue system,
including percentile calculations, throughput analysis, latency tracking,
and metrics export in multiple formats.

Features:
    - Processing time percentile calculations (p50, p95, p99)
    - Throughput metrics (items/second, items/minute)
    - Latency tracking (queue wait time)
    - Per-collection and per-tenant metrics
    - Success/failure breakdown
    - Export to JSON and Prometheus formats
    - Memory-efficient sliding window storage

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_performance_metrics import (
        QueuePerformanceCollector
    )

    # Initialize collector
    collector = QueuePerformanceCollector()
    await collector.initialize()

    # Record processing event
    await collector.record_processing_event(
        duration_ms=150.5,
        success=True,
        metadata={
            "collection": "my-project",
            "tenant_id": "tenant-1"
        }
    )

    # Get throughput metrics
    throughput = await collector.get_throughput_metrics(window_minutes=5)
    print(f"Items/sec: {throughput.items_per_second}")

    # Get latency metrics
    latency = await collector.get_latency_metrics(window_minutes=5)
    print(f"Avg latency: {latency.avg_latency_ms}ms")

    # Get processing time percentiles
    stats = await collector.get_processing_time_stats()
    print(f"P95: {stats.p95}ms, P99: {stats.p99}ms")

    # Export metrics
    json_export = await collector.export_metrics(format='json')
    prometheus_export = await collector.export_metrics(format='prometheus')
    ```
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import psutil
from loguru import logger

from .queue_statistics import QueueStatisticsCollector


@dataclass
class ThroughputMetrics:
    """
    Throughput metrics for queue processing.

    Attributes:
        items_per_second: Processing throughput in items/second
        items_per_minute: Processing throughput in items/minute
        total_items: Total items processed in window
        window_seconds: Time window for calculation
        timestamp: When metrics were collected
    """
    items_per_second: float = 0.0
    items_per_minute: float = 0.0
    total_items: int = 0
    window_seconds: int = 300
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "items_per_second": round(self.items_per_second, 2),
            "items_per_minute": round(self.items_per_minute, 2),
            "total_items": self.total_items,
            "window_seconds": self.window_seconds,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class LatencyMetrics:
    """
    Latency metrics for queue wait times.

    Attributes:
        avg_latency_ms: Average latency in milliseconds
        min_latency_ms: Minimum latency in milliseconds
        max_latency_ms: Maximum latency in milliseconds
        total_items: Total items measured
        timestamp: When metrics were collected
    """
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    total_items: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "total_items": self.total_items,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class MetricsAggregator:
    """
    Statistical aggregator for metrics with percentiles.

    Attributes:
        min: Minimum value
        max: Maximum value
        avg: Average value
        p50: 50th percentile (median)
        p95: 95th percentile
        p99: 99th percentile
        count: Sample count
    """
    min: float = 0.0
    max: float = 0.0
    avg: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "avg": round(self.avg, 2),
            "p50": round(self.p50, 2),
            "p95": round(self.p95, 2),
            "p99": round(self.p99, 2),
            "count": self.count
        }


@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics.

    Attributes:
        throughput: Throughput metrics
        latency: Latency metrics
        processing_time: Processing time statistics with percentiles
        resource_usage: Resource usage metrics (CPU, memory)
        success_count: Number of successful processing events
        failure_count: Number of failed processing events
        timestamp: When metrics were collected
    """
    throughput: ThroughputMetrics = field(default_factory=ThroughputMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)
    processing_time: MetricsAggregator = field(default_factory=MetricsAggregator)
    resource_usage: dict[str, float] = field(default_factory=dict)
    success_count: int = 0
    failure_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "throughput": self.throughput.to_dict(),
            "latency": self.latency.to_dict(),
            "processing_time": self.processing_time.to_dict(),
            "resource_usage": self.resource_usage,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timestamp": self.timestamp.isoformat()
        }


class QueuePerformanceCollector:
    """
    Advanced performance metrics collector for queue system.

    Extends QueueStatisticsCollector with detailed performance tracking,
    percentile calculations, and metrics export capabilities.
    """

    def __init__(
        self,
        db_path: str | None = None,
        connection_config: Any | None = None,
        window_minutes: int = 5,
        max_events: int = 10000,
        enable_resource_tracking: bool = True
    ):
        """
        Initialize performance collector.

        Args:
            db_path: Optional custom database path
            connection_config: Optional connection configuration
            window_minutes: Default time window in minutes
            max_events: Maximum events to keep in memory
            enable_resource_tracking: Whether to track CPU/memory usage
        """
        # Initialize base statistics collector
        self.statistics_collector = QueueStatisticsCollector(
            db_path=db_path,
            connection_config=connection_config,
            window_1min=60,
            window_5min=300,
            window_15min=900,
            max_events=max_events
        )

        self.window_minutes = window_minutes
        self.enable_resource_tracking = enable_resource_tracking
        self._initialized = False

        # Latency tracking (enqueue to processing start time)
        self._latency_events: deque[tuple[float, float]] = deque(maxlen=max_events)
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize the performance collector."""
        if self._initialized:
            return

        await self.statistics_collector.initialize()
        self._initialized = True
        logger.info("Queue performance collector initialized")

    async def close(self):
        """Close the performance collector."""
        if not self._initialized:
            return

        await self.statistics_collector.close()
        self._initialized = False
        logger.info("Queue performance collector closed")

    async def record_processing_event(
        self,
        duration_ms: float,
        success: bool,
        metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Record a processing event with performance data.

        Args:
            duration_ms: Processing duration in milliseconds
            success: Whether processing was successful
            metadata: Optional metadata (collection, tenant_id, etc.)
        """
        metadata = metadata or {}
        event_type = "success" if success else "failure"

        # Record in base statistics collector
        await self.statistics_collector.record_event(
            event_type=event_type,
            processing_time=duration_ms,
            queue_type=metadata.get("queue_type", "ingestion_queue"),
            collection=metadata.get("collection"),
            tenant_id=metadata.get("tenant_id")
        )

        # Record latency if provided
        if "enqueue_time" in metadata:
            latency_ms = (time.time() - metadata["enqueue_time"]) * 1000
            async with self._lock:
                self._latency_events.append((time.time(), latency_ms))

        logger.debug(
            f"Recorded processing event: {event_type}, duration={duration_ms}ms"
        )

    async def get_throughput_metrics(
        self,
        window_minutes: int = 5,
        collection: str | None = None,
        tenant_id: str | None = None
    ) -> ThroughputMetrics:
        """
        Get throughput metrics for specified time window.

        Args:
            window_minutes: Time window in minutes
            collection: Optional collection filter
            tenant_id: Optional tenant filter

        Returns:
            Throughput metrics
        """
        window_seconds = window_minutes * 60
        cutoff_time = time.time() - window_seconds

        async with self.statistics_collector._lock:
            # Filter events by time window and filters
            filtered_events = [
                e for e in self.statistics_collector._events
                if e.timestamp >= cutoff_time
                and e.event_type in ("success", "failure")
                and (collection is None or e.collection == collection)
                and (tenant_id is None or e.tenant_id == tenant_id)
            ]

        total_items = len(filtered_events)
        items_per_second = total_items / window_seconds if window_seconds > 0 else 0.0
        items_per_minute = items_per_second * 60

        return ThroughputMetrics(
            items_per_second=items_per_second,
            items_per_minute=items_per_minute,
            total_items=total_items,
            window_seconds=window_seconds,
            timestamp=datetime.now(timezone.utc)
        )

    async def get_latency_metrics(
        self,
        window_minutes: int = 5
    ) -> LatencyMetrics:
        """
        Get latency metrics for queue wait times.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Latency metrics
        """
        window_seconds = window_minutes * 60
        cutoff_time = time.time() - window_seconds

        async with self._lock:
            # Filter latency events by time window
            filtered_latencies = [
                latency for timestamp, latency in self._latency_events
                if timestamp >= cutoff_time
            ]

        if not filtered_latencies:
            return LatencyMetrics(
                timestamp=datetime.now(timezone.utc)
            )

        return LatencyMetrics(
            avg_latency_ms=sum(filtered_latencies) / len(filtered_latencies),
            min_latency_ms=min(filtered_latencies),
            max_latency_ms=max(filtered_latencies),
            total_items=len(filtered_latencies),
            timestamp=datetime.now(timezone.utc)
        )

    async def get_processing_time_stats(
        self,
        window_minutes: int | None = None,
        collection: str | None = None,
        tenant_id: str | None = None
    ) -> MetricsAggregator:
        """
        Get processing time statistics with percentiles.

        Args:
            window_minutes: Optional time window in minutes (default: use collector default)
            collection: Optional collection filter
            tenant_id: Optional tenant filter

        Returns:
            Metrics aggregator with percentiles
        """
        window_minutes = window_minutes or self.window_minutes
        window_seconds = window_minutes * 60
        cutoff_time = time.time() - window_seconds

        async with self.statistics_collector._lock:
            # Filter events with processing time
            filtered_times = [
                e.processing_time for e in self.statistics_collector._events
                if e.timestamp >= cutoff_time
                and e.processing_time is not None
                and (collection is None or e.collection == collection)
                and (tenant_id is None or e.tenant_id == tenant_id)
            ]

        if not filtered_times:
            return MetricsAggregator()

        # Sort for percentile calculation
        sorted_times = sorted(filtered_times)
        count = len(sorted_times)

        # Calculate percentiles
        p50 = self._calculate_percentile(sorted_times, 50)
        p95 = self._calculate_percentile(sorted_times, 95)
        p99 = self._calculate_percentile(sorted_times, 99)

        return MetricsAggregator(
            min=min(sorted_times),
            max=max(sorted_times),
            avg=sum(sorted_times) / count,
            p50=p50,
            p95=p95,
            p99=p99,
            count=count
        )

    def _calculate_percentile(self, sorted_values: list[float], percentile: float) -> float:
        """
        Calculate percentile from sorted values.

        Uses linear interpolation between closest ranks.

        Args:
            sorted_values: Sorted list of values
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0.0

        if len(sorted_values) == 1:
            return sorted_values[0]

        # Calculate rank (0-indexed)
        rank = (percentile / 100) * (len(sorted_values) - 1)
        lower_index = int(rank)
        upper_index = min(lower_index + 1, len(sorted_values) - 1)

        # Linear interpolation
        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]
        fraction = rank - lower_index

        return lower_value + fraction * (upper_value - lower_value)

    async def get_metrics_by_collection(
        self,
        collection_name: str,
        window_minutes: int | None = None
    ) -> PerformanceMetrics:
        """
        Get performance metrics for specific collection.

        Args:
            collection_name: Collection name to filter by
            window_minutes: Optional time window in minutes

        Returns:
            Performance metrics for collection
        """
        window_minutes = window_minutes or self.window_minutes

        throughput = await self.get_throughput_metrics(
            window_minutes=window_minutes,
            collection=collection_name
        )

        processing_time = await self.get_processing_time_stats(
            window_minutes=window_minutes,
            collection=collection_name
        )

        # Get success/failure counts
        window_seconds = window_minutes * 60
        cutoff_time = time.time() - window_seconds

        async with self.statistics_collector._lock:
            filtered_events = [
                e for e in self.statistics_collector._events
                if e.timestamp >= cutoff_time
                and e.collection == collection_name
            ]

        success_count = sum(1 for e in filtered_events if e.event_type == "success")
        failure_count = sum(1 for e in filtered_events if e.event_type == "failure")

        # Get resource usage if enabled
        resource_usage = {}
        if self.enable_resource_tracking:
            resource_usage = self._get_resource_usage()

        return PerformanceMetrics(
            throughput=throughput,
            latency=LatencyMetrics(),  # Latency not collection-specific
            processing_time=processing_time,
            resource_usage=resource_usage,
            success_count=success_count,
            failure_count=failure_count,
            timestamp=datetime.now(timezone.utc)
        )

    async def get_metrics_by_tenant(
        self,
        tenant_id: str,
        window_minutes: int | None = None
    ) -> PerformanceMetrics:
        """
        Get performance metrics for specific tenant.

        Args:
            tenant_id: Tenant identifier to filter by
            window_minutes: Optional time window in minutes

        Returns:
            Performance metrics for tenant
        """
        window_minutes = window_minutes or self.window_minutes

        throughput = await self.get_throughput_metrics(
            window_minutes=window_minutes,
            tenant_id=tenant_id
        )

        processing_time = await self.get_processing_time_stats(
            window_minutes=window_minutes,
            tenant_id=tenant_id
        )

        # Get success/failure counts
        window_seconds = window_minutes * 60
        cutoff_time = time.time() - window_seconds

        async with self.statistics_collector._lock:
            filtered_events = [
                e for e in self.statistics_collector._events
                if e.timestamp >= cutoff_time
                and e.tenant_id == tenant_id
            ]

        success_count = sum(1 for e in filtered_events if e.event_type == "success")
        failure_count = sum(1 for e in filtered_events if e.event_type == "failure")

        # Get resource usage if enabled
        resource_usage = {}
        if self.enable_resource_tracking:
            resource_usage = self._get_resource_usage()

        return PerformanceMetrics(
            throughput=throughput,
            latency=LatencyMetrics(),  # Latency not tenant-specific
            processing_time=processing_time,
            resource_usage=resource_usage,
            success_count=success_count,
            failure_count=failure_count,
            timestamp=datetime.now(timezone.utc)
        )

    def _get_resource_usage(self) -> dict[str, float]:
        """
        Get current resource usage metrics.

        Returns:
            Dictionary with CPU and memory usage
        """
        try:
            process = psutil.Process()
            return {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "memory_percent": process.memory_percent()
            }
        except Exception as e:
            logger.warning(f"Failed to get resource usage: {e}")
            return {}

    async def export_metrics(
        self,
        format: str = 'json',
        window_minutes: int | None = None
    ) -> str:
        """
        Export metrics in specified format.

        Args:
            format: Export format ('json' or 'prometheus')
            window_minutes: Optional time window in minutes

        Returns:
            Formatted metrics string

        Raises:
            ValueError: If format is not supported
        """
        if format not in ('json', 'prometheus'):
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'prometheus'")

        window_minutes = window_minutes or self.window_minutes

        # Collect all metrics
        throughput = await self.get_throughput_metrics(window_minutes=window_minutes)
        latency = await self.get_latency_metrics(window_minutes=window_minutes)
        processing_time = await self.get_processing_time_stats(window_minutes=window_minutes)

        # Get success/failure counts
        window_seconds = window_minutes * 60
        cutoff_time = time.time() - window_seconds

        async with self.statistics_collector._lock:
            filtered_events = [
                e for e in self.statistics_collector._events
                if e.timestamp >= cutoff_time
            ]

        success_count = sum(1 for e in filtered_events if e.event_type == "success")
        failure_count = sum(1 for e in filtered_events if e.event_type == "failure")

        # Get resource usage
        resource_usage = {}
        if self.enable_resource_tracking:
            resource_usage = self._get_resource_usage()

        if format == 'json':
            return self._export_json(
                throughput=throughput,
                latency=latency,
                processing_time=processing_time,
                resource_usage=resource_usage,
                success_count=success_count,
                failure_count=failure_count,
                window_minutes=window_minutes
            )
        else:
            return self._export_prometheus(
                throughput=throughput,
                latency=latency,
                processing_time=processing_time,
                resource_usage=resource_usage,
                success_count=success_count,
                failure_count=failure_count
            )

    def _export_json(
        self,
        throughput: ThroughputMetrics,
        latency: LatencyMetrics,
        processing_time: MetricsAggregator,
        resource_usage: dict[str, float],
        success_count: int,
        failure_count: int,
        window_minutes: int
    ) -> str:
        """
        Export metrics as JSON.

        Args:
            throughput: Throughput metrics
            latency: Latency metrics
            processing_time: Processing time statistics
            resource_usage: Resource usage metrics
            success_count: Success count
            failure_count: Failure count
            window_minutes: Time window in minutes

        Returns:
            JSON string
        """
        data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "window_minutes": window_minutes,
                "collection_period_seconds": window_minutes * 60
            },
            "throughput": throughput.to_dict(),
            "latency": latency.to_dict(),
            "processing_time": processing_time.to_dict(),
            "resource_usage": resource_usage,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": round(
                (success_count / (success_count + failure_count) * 100)
                if (success_count + failure_count) > 0 else 0.0,
                2
            )
        }

        return json.dumps(data, indent=2)

    def _export_prometheus(
        self,
        throughput: ThroughputMetrics,
        latency: LatencyMetrics,
        processing_time: MetricsAggregator,
        resource_usage: dict[str, float],
        success_count: int,
        failure_count: int
    ) -> str:
        """
        Export metrics in Prometheus text format.

        Args:
            throughput: Throughput metrics
            latency: Latency metrics
            processing_time: Processing time statistics
            resource_usage: Resource usage metrics
            success_count: Success count
            failure_count: Failure count

        Returns:
            Prometheus format string
        """
        lines = []

        # Throughput metrics
        lines.append("# HELP queue_throughput_items_per_second Queue processing throughput")
        lines.append("# TYPE queue_throughput_items_per_second gauge")
        lines.append(f"queue_throughput_items_per_second {throughput.items_per_second}")

        lines.append("# HELP queue_throughput_items_per_minute Queue processing throughput per minute")
        lines.append("# TYPE queue_throughput_items_per_minute gauge")
        lines.append(f"queue_throughput_items_per_minute {throughput.items_per_minute}")

        # Latency metrics
        lines.append("# HELP queue_latency_milliseconds Queue wait time latency")
        lines.append("# TYPE queue_latency_milliseconds gauge")
        lines.append(f"queue_latency_milliseconds{{stat=\"avg\"}} {latency.avg_latency_ms}")
        lines.append(f"queue_latency_milliseconds{{stat=\"min\"}} {latency.min_latency_ms}")
        lines.append(f"queue_latency_milliseconds{{stat=\"max\"}} {latency.max_latency_ms}")

        # Processing time metrics
        lines.append("# HELP queue_processing_time_milliseconds Processing time statistics")
        lines.append("# TYPE queue_processing_time_milliseconds gauge")
        lines.append(f"queue_processing_time_milliseconds{{stat=\"min\"}} {processing_time.min}")
        lines.append(f"queue_processing_time_milliseconds{{stat=\"max\"}} {processing_time.max}")
        lines.append(f"queue_processing_time_milliseconds{{stat=\"avg\"}} {processing_time.avg}")
        lines.append(f"queue_processing_time_milliseconds{{stat=\"p50\"}} {processing_time.p50}")
        lines.append(f"queue_processing_time_milliseconds{{stat=\"p95\"}} {processing_time.p95}")
        lines.append(f"queue_processing_time_milliseconds{{stat=\"p99\"}} {processing_time.p99}")

        # Success/failure counts
        lines.append("# HELP queue_processing_total Total processing events")
        lines.append("# TYPE queue_processing_total counter")
        lines.append(f"queue_processing_total{{status=\"success\"}} {success_count}")
        lines.append(f"queue_processing_total{{status=\"failure\"}} {failure_count}")

        # Resource usage
        if resource_usage:
            lines.append("# HELP queue_resource_usage Resource usage metrics")
            lines.append("# TYPE queue_resource_usage gauge")
            for key, value in resource_usage.items():
                metric_name = key.replace("_", "_")
                lines.append(f"queue_resource_usage{{resource=\"{metric_name}\"}} {value}")

        return "\n".join(lines) + "\n"
