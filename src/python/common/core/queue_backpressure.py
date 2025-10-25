"""
Queue Backpressure Detection and Alerting

Provides real-time backpressure detection for the queue system, monitoring queue
growth rates, processing capacity, and generating alerts when backpressure is detected.

Features:
    - Real-time backpressure indicator calculation
    - Queue growth rate monitoring
    - Processing capacity utilization tracking
    - Estimated drain time calculation
    - Queue size prediction based on trends
    - Configurable severity-based alerting (LOW, MEDIUM, HIGH, CRITICAL)
    - Async callback system for alert notifications
    - Background monitoring task with configurable intervals

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_backpressure import BackpressureDetector
    from workspace_qdrant_mcp.core.queue_statistics import QueueStatisticsCollector

    # Initialize statistics collector and backpressure detector
    stats_collector = QueueStatisticsCollector()
    await stats_collector.initialize()
    await stats_collector.start_collection(interval_seconds=5)

    detector = BackpressureDetector(stats_collector)
    await detector.initialize()

    # Register callback for alerts
    async def on_alert(alert):
        print(f"Backpressure alert: {alert.severity} - {alert.recommended_actions}")

    await detector.register_backpressure_callback(on_alert)

    # Start monitoring
    await detector.start_monitoring(check_interval_seconds=60)

    # Check current backpressure
    alert = await detector.detect_backpressure()
    if alert:
        print(f"Current backpressure: {alert.severity}")

    # Get indicators
    indicators = await detector.get_backpressure_indicators()
    print(f"Queue growth rate: {indicators.queue_growth_rate} items/min")
    print(f"Estimated drain time: {indicators.estimated_drain_time}")

    # Predict future queue size
    predicted_size = await detector.predict_queue_size(minutes_ahead=30)
    print(f"Predicted queue size in 30min: {predicted_size}")

    # Stop monitoring
    await detector.stop_monitoring()
    ```
"""

import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

from loguru import logger

from .queue_statistics import QueueStatisticsCollector


class BackpressureSeverity(str, Enum):
    """Backpressure alert severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QueueTrend(str, Enum):
    """Queue depth trend indicators."""

    GROWING = "growing"
    STABLE = "stable"
    DRAINING = "draining"


@dataclass
class BackpressureIndicators:
    """
    Real-time backpressure indicators.

    Attributes:
        queue_growth_rate: Items per minute (positive = growing, negative = draining)
        processing_capacity_used: Percentage of max throughput (0-100)
        estimated_drain_time: Time to empty queue at current processing rate
        queue_size: Current queue size
        queue_trend: Direction of queue size change
        timestamp: When indicators were calculated
    """

    queue_growth_rate: float  # items/min
    processing_capacity_used: float  # percentage 0-100
    estimated_drain_time: timedelta
    queue_size: int = 0
    queue_trend: QueueTrend = QueueTrend.STABLE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "queue_growth_rate": round(self.queue_growth_rate, 2),
            "processing_capacity_used": round(self.processing_capacity_used, 2),
            "estimated_drain_time_seconds": int(self.estimated_drain_time.total_seconds()),
            "estimated_drain_time_human": self._format_timedelta(self.estimated_drain_time),
            "queue_size": self.queue_size,
            "queue_trend": self.queue_trend.value,
            "timestamp": self.timestamp.isoformat(),
        }

    @staticmethod
    def _format_timedelta(td: timedelta) -> str:
        """Format timedelta for human readability."""
        total_seconds = int(td.total_seconds())
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            return f"{total_seconds // 60}m"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
        else:
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            return f"{days}d {hours}h" if hours > 0 else f"{days}d"


@dataclass
class BackpressureAlert:
    """
    Backpressure alert with severity and recommended actions.

    Attributes:
        severity: Alert severity level
        indicators: Current backpressure indicators
        recommended_actions: List of recommended actions to address backpressure
        timestamp: When alert was generated
    """

    severity: BackpressureSeverity
    indicators: BackpressureIndicators
    recommended_actions: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "indicators": self.indicators.to_dict(),
            "recommended_actions": self.recommended_actions,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BackpressureThresholds:
    """
    Configurable thresholds for backpressure detection.

    Attributes:
        low_growth_rate: Growth rate threshold for LOW severity (items/min)
        low_drain_time_minutes: Drain time threshold for LOW severity
        medium_growth_rate: Growth rate threshold for MEDIUM severity (items/min)
        medium_drain_time_minutes: Drain time threshold for MEDIUM severity
        high_growth_rate: Growth rate threshold for HIGH severity (items/min)
        high_drain_time_minutes: Drain time threshold for HIGH severity
        critical_growth_rate: Growth rate threshold for CRITICAL severity (items/min)
        critical_drain_time_minutes: Drain time threshold for CRITICAL severity
        max_processing_capacity: Maximum processing capacity (items/min)
    """

    # LOW severity thresholds
    low_growth_rate: float = 10.0  # items/min
    low_drain_time_minutes: float = 30.0  # < 30min drain time

    # MEDIUM severity thresholds
    medium_growth_rate: float = 50.0  # items/min
    medium_drain_time_min_minutes: float = 30.0  # 30-60min drain time
    medium_drain_time_max_minutes: float = 60.0

    # HIGH severity thresholds
    high_growth_rate: float = 100.0  # items/min
    high_drain_time_min_minutes: float = 60.0  # 1-2hr drain time
    high_drain_time_max_minutes: float = 120.0

    # CRITICAL severity thresholds
    critical_growth_rate: float = 200.0  # items/min
    critical_drain_time_minutes: float = 120.0  # > 2hr drain time

    # Processing capacity
    max_processing_capacity: float = 1000.0  # items/min (adjust based on system)


class BackpressureDetector:
    """
    Real-time backpressure detector with monitoring and alerting.

    Monitors queue statistics, calculates backpressure indicators, and generates
    alerts when backpressure thresholds are exceeded.
    """

    def __init__(
        self,
        stats_collector: QueueStatisticsCollector,
        thresholds: BackpressureThresholds | None = None,
    ):
        """
        Initialize backpressure detector.

        Args:
            stats_collector: Queue statistics collector for metrics
            thresholds: Optional custom thresholds (uses defaults if None)
        """
        self.stats_collector = stats_collector
        self.thresholds = thresholds or BackpressureThresholds()
        self._initialized = False

        # Historical data for trend analysis
        self._queue_size_history: deque[tuple[datetime, int]] = deque(maxlen=100)
        self._lock = asyncio.Lock()

        # Alert callbacks
        self._callbacks: list[Callable] = []

        # Background monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._monitoring_interval: int = 60  # seconds
        self._shutdown_event = asyncio.Event()

        # Latest indicators cache
        self._latest_indicators: BackpressureIndicators | None = None

    async def initialize(self):
        """Initialize the backpressure detector."""
        if self._initialized:
            return

        # Ensure stats collector is initialized
        if not self.stats_collector._initialized:
            await self.stats_collector.initialize()

        self._initialized = True
        logger.info("Backpressure detector initialized")

    async def close(self):
        """Close the backpressure detector."""
        if not self._initialized:
            return

        # Stop monitoring if running
        await self.stop_monitoring()

        self._initialized = False
        logger.info("Backpressure detector closed")

    async def detect_backpressure(
        self, queue_type: str = "ingestion_queue"
    ) -> BackpressureAlert | None:
        """
        Detect current backpressure state.

        Args:
            queue_type: Queue type to check for backpressure

        Returns:
            BackpressureAlert if backpressure detected, None otherwise
        """
        # Get current indicators
        indicators = await self.get_backpressure_indicators(queue_type=queue_type)

        # Determine severity based on thresholds
        severity = self._calculate_severity(indicators)

        if severity == BackpressureSeverity.NONE:
            return None

        # Get recommended actions based on severity
        recommended_actions = self._get_recommended_actions(severity, indicators)

        alert = BackpressureAlert(
            severity=severity,
            indicators=indicators,
            recommended_actions=recommended_actions,
        )

        logger.warning(
            f"Backpressure detected: {severity.value} "
            f"(growth_rate={indicators.queue_growth_rate:.2f} items/min, "
            f"drain_time={indicators.estimated_drain_time})"
        )

        return alert

    async def get_backpressure_indicators(
        self, queue_type: str = "ingestion_queue"
    ) -> BackpressureIndicators:
        """
        Get current backpressure indicators.

        Args:
            queue_type: Queue type to analyze

        Returns:
            Current backpressure indicators
        """
        # Get current statistics
        stats = await self.stats_collector.get_current_statistics(queue_type=queue_type)

        # Record queue size in history
        async with self._lock:
            self._queue_size_history.append((datetime.now(timezone.utc), stats.queue_size))

        # Calculate queue growth rate
        growth_rate = await self._calculate_growth_rate()

        # Calculate processing capacity used
        capacity_used = await self._calculate_capacity_used(stats.processing_rate)

        # Calculate estimated drain time
        drain_time = await self.calculate_drain_time(queue_type=queue_type)

        # Determine queue trend
        trend = self._determine_trend(growth_rate)

        indicators = BackpressureIndicators(
            queue_growth_rate=growth_rate,
            processing_capacity_used=capacity_used,
            estimated_drain_time=drain_time,
            queue_size=stats.queue_size,
            queue_trend=trend,
        )

        # Cache latest indicators
        self._latest_indicators = indicators

        return indicators

    async def calculate_drain_time(
        self, queue_type: str = "ingestion_queue"
    ) -> timedelta:
        """
        Calculate estimated time to drain queue at current processing rate.

        Args:
            queue_type: Queue type to calculate drain time for

        Returns:
            Estimated drain time as timedelta
        """
        stats = await self.stats_collector.get_current_statistics(queue_type=queue_type)

        # If processing rate is zero or negative, return max timedelta
        if stats.processing_rate <= 0:
            # Queue is not being processed
            return timedelta(days=365)  # Effectively infinite

        # Calculate drain time in minutes
        drain_minutes = stats.queue_size / stats.processing_rate

        return timedelta(minutes=drain_minutes)

    async def predict_queue_size(
        self, minutes_ahead: int, queue_type: str = "ingestion_queue"
    ) -> int:
        """
        Predict queue size at a future time based on current trends.

        Args:
            minutes_ahead: Number of minutes into the future to predict
            queue_type: Queue type to predict for

        Returns:
            Predicted queue size (cannot go below 0)
        """
        stats = await self.stats_collector.get_current_statistics(queue_type=queue_type)

        # Net growth rate = items added - items processed
        growth_rate = await self._calculate_growth_rate()

        # Predict queue size
        predicted_size = stats.queue_size + (growth_rate * minutes_ahead)

        # Cannot go below 0
        return max(0, int(predicted_size))

    async def register_backpressure_callback(self, callback: Callable) -> bool:
        """
        Register a callback for backpressure alerts.

        Args:
            callback: Async callback function that receives BackpressureAlert

        Returns:
            True if registered successfully
        """
        async with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                logger.debug(f"Registered backpressure callback: {callback.__name__}")
                return True
            return False

    async def unregister_backpressure_callback(self, callback: Callable) -> bool:
        """
        Unregister a backpressure callback.

        Args:
            callback: Callback to remove

        Returns:
            True if unregistered successfully
        """
        async with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                logger.debug(f"Unregistered backpressure callback: {callback.__name__}")
                return True
            return False

    async def start_monitoring(self, check_interval_seconds: int = 60) -> bool:
        """
        Start background backpressure monitoring.

        Args:
            check_interval_seconds: Interval between backpressure checks

        Returns:
            True if started successfully, False if already running
        """
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Backpressure monitoring already running")
            return False

        self._monitoring_interval = check_interval_seconds
        self._shutdown_event.clear()

        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info(
            f"Started backpressure monitoring (interval={check_interval_seconds}s)"
        )
        return True

    async def stop_monitoring(self) -> bool:
        """
        Stop background backpressure monitoring.

        Returns:
            True if stopped successfully, False if not running
        """
        if not self._monitoring_task or self._monitoring_task.done():
            logger.warning("Backpressure monitoring not running")
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

        logger.info("Stopped backpressure monitoring")
        return True

    async def _monitoring_loop(self):
        """Background task for periodic backpressure detection."""
        logger.info("Starting backpressure monitoring loop")

        while not self._shutdown_event.is_set():
            try:
                # Detect backpressure
                alert = await self.detect_backpressure()

                # Trigger callbacks if alert exists
                if alert:
                    await self._trigger_callbacks(alert)

                # Wait for next interval
                await asyncio.sleep(self._monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in backpressure monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

        logger.info("Backpressure monitoring loop stopped")

    async def _trigger_callbacks(self, alert: BackpressureAlert):
        """Trigger registered callbacks with alert."""
        async with self._lock:
            callbacks = self._callbacks.copy()

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in backpressure callback {callback.__name__}: {e}")

    async def _calculate_growth_rate(self) -> float:
        """
        Calculate queue growth rate from historical data.

        Returns:
            Growth rate in items per minute (positive = growing, negative = draining)
        """
        async with self._lock:
            if len(self._queue_size_history) < 2:
                return 0.0

            # Get first and last data points
            first_time, first_size = self._queue_size_history[0]
            last_time, last_size = self._queue_size_history[-1]

            # Calculate time difference in minutes
            time_diff = (last_time - first_time).total_seconds() / 60.0

            if time_diff <= 0:
                return 0.0

            # Calculate growth rate (items per minute)
            growth_rate = (last_size - first_size) / time_diff

            return growth_rate

    async def _calculate_capacity_used(self, processing_rate: float) -> float:
        """
        Calculate processing capacity utilization percentage.

        Args:
            processing_rate: Current processing rate (items/min)

        Returns:
            Capacity used as percentage (0-100)
        """
        if self.thresholds.max_processing_capacity <= 0:
            return 0.0

        capacity_used = (
            processing_rate / self.thresholds.max_processing_capacity
        ) * 100

        # Cap at 100%
        return min(100.0, capacity_used)

    def _determine_trend(self, growth_rate: float) -> QueueTrend:
        """
        Determine queue trend from growth rate.

        Args:
            growth_rate: Queue growth rate (items/min)

        Returns:
            Queue trend indicator
        """
        # Thresholds for trend determination
        STABLE_THRESHOLD = 5.0  # items/min

        if growth_rate > STABLE_THRESHOLD:
            return QueueTrend.GROWING
        elif growth_rate < -STABLE_THRESHOLD:
            return QueueTrend.DRAINING
        else:
            return QueueTrend.STABLE

    def _calculate_severity(self, indicators: BackpressureIndicators) -> BackpressureSeverity:
        """
        Calculate backpressure severity from indicators.

        Args:
            indicators: Current backpressure indicators

        Returns:
            Severity level
        """
        growth_rate = indicators.queue_growth_rate
        drain_minutes = indicators.estimated_drain_time.total_seconds() / 60.0

        # CRITICAL: Very high growth rate OR very long drain time
        if (
            growth_rate >= self.thresholds.critical_growth_rate
            or drain_minutes >= self.thresholds.critical_drain_time_minutes
        ):
            return BackpressureSeverity.CRITICAL

        # HIGH: High growth rate AND long drain time
        if (
            growth_rate >= self.thresholds.high_growth_rate
            and drain_minutes >= self.thresholds.high_drain_time_min_minutes
            and drain_minutes <= self.thresholds.high_drain_time_max_minutes
        ):
            return BackpressureSeverity.HIGH

        # MEDIUM: Medium growth rate AND moderate drain time
        if (
            growth_rate >= self.thresholds.medium_growth_rate
            and drain_minutes >= self.thresholds.medium_drain_time_min_minutes
            and drain_minutes <= self.thresholds.medium_drain_time_max_minutes
        ):
            return BackpressureSeverity.MEDIUM

        # LOW: Low growth rate AND reasonable drain time (not too short, not too long)
        # Only trigger if growth is significant AND queue has some depth
        if growth_rate >= self.thresholds.low_growth_rate:
            return BackpressureSeverity.LOW

        return BackpressureSeverity.NONE

    def _get_recommended_actions(
        self, severity: BackpressureSeverity, indicators: BackpressureIndicators
    ) -> list[str]:
        """
        Get recommended actions based on severity.

        Args:
            severity: Alert severity level
            indicators: Current backpressure indicators

        Returns:
            List of recommended actions
        """
        actions = []

        if severity == BackpressureSeverity.LOW:
            actions.append("Monitor queue growth, consider increasing workers if trend continues")
            actions.append(f"Current growth rate: {indicators.queue_growth_rate:.1f} items/min")

        elif severity == BackpressureSeverity.MEDIUM:
            actions.append("Increase processing capacity immediately")
            actions.append("Check for bottlenecks in processing pipeline")
            actions.append(f"Queue will drain in ~{indicators.estimated_drain_time}")

        elif severity == BackpressureSeverity.HIGH:
            actions.append("Scale up workers immediately")
            actions.append("Investigate slow processing operations")
            actions.append("Consider temporarily pausing low-priority ingestion")
            actions.append(f"Estimated drain time: {indicators.estimated_drain_time}")

        elif severity == BackpressureSeverity.CRITICAL:
            actions.append("EMERGENCY: Scale up processing capacity immediately")
            actions.append("Pause non-critical document ingestion")
            actions.append("Investigate system bottlenecks and resource constraints")
            actions.append(f"Critical backpressure: {indicators.queue_growth_rate:.1f} items/min growth")

        return actions
