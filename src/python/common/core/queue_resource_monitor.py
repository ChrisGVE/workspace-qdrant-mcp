"""
Queue Resource Utilization Tracking

Monitors system resource usage and correlates it with queue performance metrics
to detect resource bottlenecks and optimize queue processing.

Features:
    - CPU and memory usage tracking using psutil
    - Database connection pool monitoring
    - Thread count monitoring
    - Historical resource data with sliding window
    - Correlation analysis between resources and queue performance
    - Automatic bottleneck detection (CPU, memory, connections, threads)
    - Background monitoring with configurable intervals
    - Memory-efficient snapshot storage

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_resource_monitor import QueueResourceMonitor

    # Initialize monitor
    monitor = QueueResourceMonitor()
    await monitor.initialize()

    # Start background monitoring
    await monitor.start_monitoring(interval_seconds=30)

    # Get current resources
    metrics = await monitor.get_current_resources()
    print(f"CPU: {metrics.cpu_percent}%")
    print(f"Memory: {metrics.memory_mb}MB")

    # Check for bottlenecks
    bottleneck = await monitor.detect_resource_bottleneck()
    if bottleneck:
        print(f"Bottleneck detected: {bottleneck.resource_type}")
        print(f"Recommendation: {bottleneck.recommendation}")

    # Get correlations with queue performance
    correlations = await monitor.correlate_with_queue_performance()
    for corr in correlations:
        print(f"{corr.resource_metric} vs {corr.queue_metric}: {corr.correlation_coefficient}")

    # Stop monitoring
    await monitor.stop_monitoring()
    ```
"""

import asyncio
import os
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    logger.warning("psutil not available, resource monitoring will be limited")
    PSUTIL_AVAILABLE = False

from .queue_connection import QueueConnectionPool, ConnectionConfig
from .queue_statistics import QueueStatisticsCollector, QueueStatistics


@dataclass
class ResourceMetrics:
    """
    System resource metrics snapshot.

    Attributes:
        cpu_percent: Process CPU usage percentage (0-100)
        memory_mb: Process memory usage in megabytes
        memory_percent: Process memory usage percentage (0-100)
        db_connections: Number of active database connections
        thread_count: Number of active threads
        file_descriptors: Number of open file descriptors (if available)
        disk_io_read_mb: Cumulative disk read in MB (if available)
        disk_io_write_mb: Cumulative disk write in MB (if available)
    """
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    db_connections: int = 0
    thread_count: int = 0
    file_descriptors: Optional[int] = None
    disk_io_read_mb: Optional[float] = None
    disk_io_write_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_mb": round(self.memory_mb, 2),
            "memory_percent": round(self.memory_percent, 2),
            "db_connections": self.db_connections,
            "thread_count": self.thread_count,
            "file_descriptors": self.file_descriptors,
            "disk_io_read_mb": round(self.disk_io_read_mb, 2) if self.disk_io_read_mb else None,
            "disk_io_write_mb": round(self.disk_io_write_mb, 2) if self.disk_io_write_mb else None,
        }


@dataclass
class ResourceSnapshot:
    """
    Combined resource and queue statistics snapshot.

    Attributes:
        timestamp: When the snapshot was taken
        metrics: Resource metrics at this moment
        queue_stats: Queue statistics at this moment
    """
    timestamp: datetime
    metrics: ResourceMetrics
    queue_stats: QueueStatistics

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics.to_dict(),
            "queue_stats": self.queue_stats.to_dict()
        }


@dataclass
class ResourceCorrelation:
    """
    Correlation between a resource metric and queue metric.

    Attributes:
        resource_metric: Name of the resource metric
        queue_metric: Name of the queue metric
        correlation_coefficient: Pearson correlation coefficient (-1 to 1)
        sample_size: Number of data points used for correlation
    """
    resource_metric: str
    queue_metric: str
    correlation_coefficient: float
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "resource_metric": self.resource_metric,
            "queue_metric": self.queue_metric,
            "correlation_coefficient": round(self.correlation_coefficient, 4),
            "sample_size": self.sample_size,
            "strength": self._get_correlation_strength()
        }

    def _get_correlation_strength(self) -> str:
        """Get human-readable correlation strength."""
        abs_corr = abs(self.correlation_coefficient)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "negligible"


@dataclass
class ResourceBottleneck:
    """
    Detected resource bottleneck.

    Attributes:
        resource_type: Type of resource causing bottleneck
        current_value: Current value of the resource metric
        threshold_value: Threshold that was exceeded
        severity: Severity level (warning, critical)
        queue_impact: Description of impact on queue performance
        recommendation: Recommendation to address bottleneck
    """
    resource_type: str
    current_value: float
    threshold_value: float
    severity: str
    queue_impact: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "resource_type": self.resource_type,
            "current_value": round(self.current_value, 2),
            "threshold_value": round(self.threshold_value, 2),
            "severity": self.severity,
            "queue_impact": self.queue_impact,
            "recommendation": self.recommendation
        }


class QueueResourceMonitor:
    """
    Queue resource utilization monitor with correlation analysis.

    Monitors system resources and correlates them with queue performance
    to detect bottlenecks and optimize processing.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        connection_config: Optional[ConnectionConfig] = None,
        snapshot_retention: int = 200,
        monitoring_interval: int = 30,
        cpu_warning_threshold: float = 70.0,
        cpu_critical_threshold: float = 90.0,
        memory_warning_mb: float = 1000.0,
        memory_critical_mb: float = 2000.0,
        low_processing_rate_threshold: float = 10.0
    ):
        """
        Initialize queue resource monitor.

        Args:
            db_path: Optional custom database path
            connection_config: Optional connection configuration
            snapshot_retention: Number of snapshots to retain in memory
            monitoring_interval: Default monitoring interval in seconds
            cpu_warning_threshold: CPU percentage for warning threshold
            cpu_critical_threshold: CPU percentage for critical threshold
            memory_warning_mb: Memory MB for warning threshold
            memory_critical_mb: Memory MB for critical threshold
            low_processing_rate_threshold: Minimum acceptable processing rate
        """
        self.connection_pool = QueueConnectionPool(
            db_path=db_path or self._get_default_db_path(),
            config=connection_config or ConnectionConfig()
        )
        self.stats_collector = QueueStatisticsCollector(
            db_path=db_path,
            connection_config=connection_config
        )

        self._initialized = False
        self._process: Optional['psutil.Process'] = None

        # Snapshot storage with sliding window
        self._snapshots: deque[ResourceSnapshot] = deque(maxlen=snapshot_retention)
        self._lock = asyncio.Lock()

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitoring_interval = monitoring_interval
        self._shutdown_event = asyncio.Event()

        # Bottleneck thresholds
        self.cpu_warning_threshold = cpu_warning_threshold
        self.cpu_critical_threshold = cpu_critical_threshold
        self.memory_warning_mb = memory_warning_mb
        self.memory_critical_mb = memory_critical_mb
        self.low_processing_rate_threshold = low_processing_rate_threshold

    def _get_default_db_path(self) -> str:
        """Get default database path from OS directories."""
        from ..utils.os_directories import OSDirectories
        os_dirs = OSDirectories()
        os_dirs.ensure_directories()
        return str(os_dirs.get_state_file("workspace_state.db"))

    async def initialize(self):
        """Initialize the resource monitor."""
        if self._initialized:
            return

        await self.connection_pool.initialize()
        await self.stats_collector.initialize()

        # Initialize psutil process handle
        if PSUTIL_AVAILABLE:
            self._process = psutil.Process(os.getpid())
            # Initialize CPU percent (first call returns 0.0)
            self._process.cpu_percent(interval=None)

        self._initialized = True
        logger.info("Queue resource monitor initialized")

    async def close(self):
        """Close the resource monitor."""
        if not self._initialized:
            return

        # Stop monitoring if running
        await self.stop_monitoring()

        await self.stats_collector.close()
        await self.connection_pool.close()

        self._initialized = False
        logger.info("Queue resource monitor closed")

    async def get_current_resources(self) -> ResourceMetrics:
        """
        Get current system resource metrics.

        Returns:
            Current resource metrics
        """
        metrics = ResourceMetrics()

        if not PSUTIL_AVAILABLE or self._process is None:
            logger.warning("psutil not available, returning empty metrics")
            return metrics

        try:
            # CPU usage (non-blocking)
            metrics.cpu_percent = self._process.cpu_percent(interval=None)

            # Memory usage
            mem_info = self._process.memory_info()
            metrics.memory_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
            metrics.memory_percent = self._process.memory_percent()

            # Thread count
            metrics.thread_count = self._process.num_threads()

            # Database connections (from connection pool)
            metrics.db_connections = self.connection_pool.active_connections

            # File descriptors (Unix-like systems only)
            try:
                metrics.file_descriptors = self._process.num_fds()
            except (AttributeError, NotImplementedError):
                # Not available on Windows
                pass

            # Disk I/O (if available)
            try:
                io_counters = self._process.io_counters()
                metrics.disk_io_read_mb = io_counters.read_bytes / (1024 * 1024)
                metrics.disk_io_write_mb = io_counters.write_bytes / (1024 * 1024)
            except (AttributeError, NotImplementedError):
                # Not available on all platforms
                pass

        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")

        return metrics

    async def take_snapshot(self) -> ResourceSnapshot:
        """
        Take a snapshot of current resources and queue statistics.

        Returns:
            Combined resource and queue snapshot
        """
        # Collect metrics
        resource_metrics = await self.get_current_resources()
        queue_stats = await self.stats_collector.get_current_statistics()

        # Create snapshot
        snapshot = ResourceSnapshot(
            timestamp=datetime.now(timezone.utc),
            metrics=resource_metrics,
            queue_stats=queue_stats
        )

        # Store in sliding window
        async with self._lock:
            self._snapshots.append(snapshot)

        logger.debug(f"Snapshot taken: CPU={resource_metrics.cpu_percent}%, "
                    f"Memory={resource_metrics.memory_mb}MB, "
                    f"Queue={queue_stats.queue_size}")

        return snapshot

    async def get_resource_history(self, minutes: int = 60) -> List[ResourceSnapshot]:
        """
        Get historical resource snapshots.

        Args:
            minutes: Number of minutes of history to retrieve

        Returns:
            List of resource snapshots within the time window
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (minutes * 60)

        async with self._lock:
            return [
                snapshot for snapshot in self._snapshots
                if snapshot.timestamp.timestamp() >= cutoff_time
            ]

    async def correlate_with_queue_performance(self) -> List[ResourceCorrelation]:
        """
        Correlate resource metrics with queue performance metrics.

        Uses Pearson correlation to identify relationships between resource
        usage and queue performance (e.g., high CPU + low throughput).

        Returns:
            List of correlation results
        """
        async with self._lock:
            if len(self._snapshots) < 2:
                logger.warning("Not enough snapshots for correlation analysis")
                return []

            snapshots = list(self._snapshots)

        correlations = []

        # Extract time series data
        cpu_values = [s.metrics.cpu_percent for s in snapshots]
        memory_values = [s.metrics.memory_mb for s in snapshots]
        processing_rates = [s.queue_stats.processing_rate for s in snapshots]
        queue_sizes = [s.queue_stats.queue_size for s in snapshots]
        error_rates = [s.queue_stats.failure_rate for s in snapshots]

        # Calculate correlations
        sample_size = len(snapshots)

        # CPU vs Processing Rate
        corr_cpu_proc = self._pearson_correlation(cpu_values, processing_rates)
        correlations.append(ResourceCorrelation(
            resource_metric="cpu_percent",
            queue_metric="processing_rate",
            correlation_coefficient=corr_cpu_proc,
            sample_size=sample_size
        ))

        # CPU vs Queue Size
        corr_cpu_queue = self._pearson_correlation(cpu_values, queue_sizes)
        correlations.append(ResourceCorrelation(
            resource_metric="cpu_percent",
            queue_metric="queue_size",
            correlation_coefficient=corr_cpu_queue,
            sample_size=sample_size
        ))

        # Memory vs Processing Rate
        corr_mem_proc = self._pearson_correlation(memory_values, processing_rates)
        correlations.append(ResourceCorrelation(
            resource_metric="memory_mb",
            queue_metric="processing_rate",
            correlation_coefficient=corr_mem_proc,
            sample_size=sample_size
        ))

        # Memory vs Queue Size
        corr_mem_queue = self._pearson_correlation(memory_values, queue_sizes)
        correlations.append(ResourceCorrelation(
            resource_metric="memory_mb",
            queue_metric="queue_size",
            correlation_coefficient=corr_mem_queue,
            sample_size=sample_size
        ))

        # Memory vs Error Rate
        corr_mem_error = self._pearson_correlation(memory_values, error_rates)
        correlations.append(ResourceCorrelation(
            resource_metric="memory_mb",
            queue_metric="error_rate",
            correlation_coefficient=corr_mem_error,
            sample_size=sample_size
        ))

        logger.debug(f"Calculated {len(correlations)} correlations from {sample_size} snapshots")

        return correlations

    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Pearson correlation coefficient.

        Args:
            x: First variable values
            y: Second variable values

        Returns:
            Correlation coefficient (-1 to 1), or 0 if calculation fails
        """
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)

        # Calculate means
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Calculate correlation components
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))

        sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(n))

        denominator = (sum_sq_x * sum_sq_y) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    async def detect_resource_bottleneck(self) -> Optional[ResourceBottleneck]:
        """
        Detect resource bottlenecks affecting queue performance.

        Analyzes current resource usage and queue performance to identify
        if a resource constraint is limiting throughput.

        Returns:
            ResourceBottleneck if detected, None otherwise
        """
        # Get current metrics
        resources = await self.get_current_resources()
        queue_stats = await self.stats_collector.get_current_statistics()

        # Check CPU bottleneck
        if resources.cpu_percent >= self.cpu_critical_threshold:
            if queue_stats.processing_rate < self.low_processing_rate_threshold:
                return ResourceBottleneck(
                    resource_type="CPU",
                    current_value=resources.cpu_percent,
                    threshold_value=self.cpu_critical_threshold,
                    severity="critical",
                    queue_impact=f"Low processing rate ({queue_stats.processing_rate:.1f} items/min) with high CPU usage",
                    recommendation="Consider increasing CPU resources or optimizing processing logic"
                )
        elif resources.cpu_percent >= self.cpu_warning_threshold:
            if queue_stats.queue_size > 1000:
                return ResourceBottleneck(
                    resource_type="CPU",
                    current_value=resources.cpu_percent,
                    threshold_value=self.cpu_warning_threshold,
                    severity="warning",
                    queue_impact=f"Queue backlog ({queue_stats.queue_size} items) with elevated CPU usage",
                    recommendation="Monitor CPU usage and consider scaling if backlog grows"
                )

        # Check memory bottleneck
        if resources.memory_mb >= self.memory_critical_mb:
            if queue_stats.queue_size > 1000:
                return ResourceBottleneck(
                    resource_type="Memory",
                    current_value=resources.memory_mb,
                    threshold_value=self.memory_critical_mb,
                    severity="critical",
                    queue_impact=f"Queue backlog ({queue_stats.queue_size} items) with high memory usage",
                    recommendation="Increase memory allocation or optimize memory usage in processing"
                )
        elif resources.memory_mb >= self.memory_warning_mb:
            if queue_stats.processing_rate < self.low_processing_rate_threshold:
                return ResourceBottleneck(
                    resource_type="Memory",
                    current_value=resources.memory_mb,
                    threshold_value=self.memory_warning_mb,
                    severity="warning",
                    queue_impact=f"Low processing rate with elevated memory usage",
                    recommendation="Monitor memory usage and check for memory leaks"
                )

        # Check connection pool bottleneck (if we know the max)
        max_connections = self.connection_pool.config.max_connections
        if resources.db_connections >= max_connections * 0.9:  # 90% of max
            if queue_stats.processing_rate < self.low_processing_rate_threshold:
                return ResourceBottleneck(
                    resource_type="Database Connections",
                    current_value=float(resources.db_connections),
                    threshold_value=float(max_connections),
                    severity="critical",
                    queue_impact="Low processing rate with connection pool near capacity",
                    recommendation=f"Increase connection pool size (current max: {max_connections})"
                )

        # Check thread bottleneck (basic heuristic)
        if resources.thread_count > 100:  # Arbitrary threshold
            if queue_stats.queue_size > 5000:
                return ResourceBottleneck(
                    resource_type="Threads",
                    current_value=float(resources.thread_count),
                    threshold_value=100.0,
                    severity="warning",
                    queue_impact=f"High thread count ({resources.thread_count}) with queue backlog",
                    recommendation="Review thread usage and consider async patterns"
                )

        # No bottleneck detected
        return None

    async def start_monitoring(self, interval_seconds: int = 30) -> bool:
        """
        Start background resource monitoring.

        Args:
            interval_seconds: Monitoring interval in seconds

        Returns:
            True if started successfully, False if already running
        """
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Resource monitoring already running")
            return False

        self._monitoring_interval = interval_seconds
        self._shutdown_event.clear()

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info(f"Started resource monitoring (interval={interval_seconds}s)")
        return True

    async def stop_monitoring(self) -> bool:
        """
        Stop background resource monitoring.

        Returns:
            True if stopped successfully, False if not running
        """
        if not self._monitoring_task or self._monitoring_task.done():
            logger.warning("Resource monitoring not running")
            return False

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for task to complete
        try:
            await asyncio.wait_for(self._monitoring_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning("Monitoring task did not stop gracefully, cancelling")
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped resource monitoring")
        return True

    async def _monitoring_loop(self):
        """Background task for periodic resource monitoring."""
        logger.info("Starting resource monitoring loop")

        while not self._shutdown_event.is_set():
            try:
                # Take snapshot
                await self.take_snapshot()

                # Check for bottlenecks
                bottleneck = await self.detect_resource_bottleneck()
                if bottleneck:
                    logger.warning(
                        f"Resource bottleneck detected: {bottleneck.resource_type} "
                        f"({bottleneck.severity}) - {bottleneck.queue_impact}"
                    )

                # Wait for next interval
                await asyncio.sleep(self._monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

        logger.info("Resource monitoring loop stopped")
