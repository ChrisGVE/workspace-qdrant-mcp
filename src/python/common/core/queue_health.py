"""
Queue Health Status Calculation

Provides comprehensive health assessment for the queue system, combining
statistics, performance metrics, and backpressure indicators into an overall
health score with actionable recommendations.

Features:
    - Overall health score calculation (0-100)
    - Individual health indicator checks
    - Status classification (HEALTHY, DEGRADED, UNHEALTHY, CRITICAL)
    - Weighted health metrics
    - Custom health check registration
    - Context-aware recommendations
    - Configurable thresholds

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_health import QueueHealthCalculator
    from workspace_qdrant_mcp.core.queue_statistics import QueueStatisticsCollector
    from workspace_qdrant_mcp.core.queue_backpressure import BackpressureDetector
    from workspace_qdrant_mcp.core.queue_performance_metrics import QueuePerformanceCollector

    # Initialize components
    stats_collector = QueueStatisticsCollector()
    await stats_collector.initialize()

    backpressure_detector = BackpressureDetector(stats_collector)
    await backpressure_detector.initialize()

    performance_collector = QueuePerformanceCollector()
    await performance_collector.initialize()

    # Initialize health calculator
    health_calculator = QueueHealthCalculator(
        stats_collector=stats_collector,
        backpressure_detector=backpressure_detector,
        performance_collector=performance_collector
    )
    await health_calculator.initialize()

    # Calculate overall health
    health_status = await health_calculator.calculate_health()
    print(f"Health: {health_status.overall_status}")
    print(f"Score: {health_status.score}/100")
    print(f"Recommendations: {health_status.recommendations}")

    # Get individual indicators
    indicators = await health_calculator.get_health_indicators()
    for indicator in indicators:
        print(f"{indicator.name}: {indicator.status} (value={indicator.value})")

    # Check if healthy
    is_healthy = await health_calculator.is_healthy()
    print(f"Is healthy: {is_healthy}")

    # Register custom health check
    async def check_custom_metric() -> tuple:
        # Custom check logic
        status = HealthStatus.HEALTHY
        value = 95.0
        threshold = 80.0
        message = "Custom metric is good"
        return status, value, threshold, message

    health_calculator.register_health_check(
        name="custom_metric",
        check_func=check_custom_metric,
        weight=10.0
    )
    ```
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

from .queue_backpressure import BackpressureDetector, BackpressureSeverity
from .queue_performance_metrics import QueuePerformanceCollector
from .queue_statistics import QueueStatisticsCollector


class HealthStatus(str, Enum):
    """Queue health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthIndicator:
    """
    Individual health indicator measurement.

    Attributes:
        name: Indicator name (e.g., "error_rate", "backlog")
        status: Health status for this indicator
        value: Current measured value
        threshold: Threshold value for comparison
        message: Human-readable status message
        score: Normalized score 0-100 (100 = perfect health)
    """

    name: str
    status: HealthStatus
    value: float
    threshold: float
    message: str
    score: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "value": round(self.value, 2),
            "threshold": round(self.threshold, 2),
            "message": self.message,
            "score": round(self.score, 2),
        }


@dataclass
class QueueHealthStatus:
    """
    Overall queue health status assessment.

    Attributes:
        overall_status: Overall health classification
        indicators: List of individual health indicators
        score: Overall health score 0-100 (100 = perfect health)
        recommendations: List of recommended actions
        timestamp: When health was assessed
    """

    overall_status: HealthStatus
    indicators: list[HealthIndicator]
    score: float
    recommendations: list[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_status": self.overall_status.value,
            "score": round(self.score, 2),
            "indicators": [ind.to_dict() for ind in self.indicators],
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class HealthThresholds:
    """
    Configurable thresholds for health checks.

    Attributes:
        backlog_normal: Normal queue size baseline
        backlog_warning: Queue size warning threshold
        backlog_critical: Queue size critical threshold
        processing_rate_min: Minimum acceptable processing rate (items/min)
        error_rate_max: Maximum acceptable error rate (%)
        latency_warning_ms: Warning threshold for latency
        latency_critical_ms: Critical threshold for latency
        success_rate_min: Minimum acceptable success rate (%)
        cpu_warning_percent: CPU usage warning threshold (%)
        cpu_critical_percent: CPU usage critical threshold (%)
        memory_warning_mb: Memory usage warning threshold (MB)
        memory_critical_mb: Memory usage critical threshold (MB)
    """

    # Backlog thresholds
    backlog_normal: int = 1000
    backlog_warning: int = 5000
    backlog_critical: int = 10000

    # Processing rate thresholds
    processing_rate_min: float = 10.0  # items/min

    # Error rate thresholds
    error_rate_max: float = 5.0  # percentage

    # Latency thresholds
    latency_warning_ms: float = 1000.0  # 1 second
    latency_critical_ms: float = 5000.0  # 5 seconds

    # Success rate thresholds
    success_rate_min: float = 95.0  # percentage

    # Resource usage thresholds
    cpu_warning_percent: float = 70.0
    cpu_critical_percent: float = 90.0
    memory_warning_mb: float = 1000.0
    memory_critical_mb: float = 2000.0


@dataclass
class HealthWeights:
    """
    Weights for health indicator contribution to overall score.

    All weights should sum to 100 for proper weighted average.

    Attributes:
        backlog: Weight for queue backlog indicator
        processing_rate: Weight for processing rate indicator
        error_rate: Weight for error rate indicator
        latency: Weight for latency indicator
        success_rate: Weight for success rate indicator
        backpressure: Weight for backpressure indicator
        resource_usage: Weight for resource usage indicator
    """

    backlog: float = 20.0
    processing_rate: float = 20.0
    error_rate: float = 25.0
    latency: float = 15.0
    success_rate: float = 20.0
    backpressure: float = 0.0  # Optional, adds to total if used
    resource_usage: float = 0.0  # Optional, adds to total if used


class QueueHealthCalculator:
    """
    Queue health status calculator.

    Combines statistics, performance metrics, and backpressure detection
    to provide comprehensive health assessment with actionable recommendations.
    """

    def __init__(
        self,
        stats_collector: QueueStatisticsCollector,
        backpressure_detector: BackpressureDetector | None = None,
        performance_collector: QueuePerformanceCollector | None = None,
        thresholds: HealthThresholds | None = None,
        weights: HealthWeights | None = None,
    ):
        """
        Initialize queue health calculator.

        Args:
            stats_collector: Queue statistics collector
            backpressure_detector: Optional backpressure detector
            performance_collector: Optional performance metrics collector
            thresholds: Optional custom thresholds (uses defaults if None)
            weights: Optional custom weights (uses defaults if None)
        """
        self.stats_collector = stats_collector
        self.backpressure_detector = backpressure_detector
        self.performance_collector = performance_collector

        self.thresholds = thresholds or HealthThresholds()
        self.weights = weights or HealthWeights()

        # Custom health checks
        self._custom_checks: dict[str, tuple[Callable, float]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the health calculator."""
        if self._initialized:
            return

        # Ensure stats collector is initialized
        if not self.stats_collector._initialized:
            await self.stats_collector.initialize()

        # Initialize optional components if provided
        if self.backpressure_detector and not self.backpressure_detector._initialized:
            await self.backpressure_detector.initialize()

        if self.performance_collector and not self.performance_collector._initialized:
            await self.performance_collector.initialize()

        self._initialized = True
        logger.info("Queue health calculator initialized")

    async def close(self):
        """Close the health calculator."""
        if not self._initialized:
            return

        self._initialized = False
        logger.info("Queue health calculator closed")

    async def calculate_health(
        self, queue_type: str = "ingestion_queue"
    ) -> QueueHealthStatus:
        """
        Calculate overall queue health status.

        Args:
            queue_type: Queue type to assess

        Returns:
            Comprehensive health status assessment
        """
        # Get all health indicators
        indicators = await self.get_health_indicators(queue_type=queue_type)

        # Calculate overall health score
        overall_score = await self.calculate_health_score(indicators)

        # Determine overall status
        overall_status = self._determine_health_status(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(overall_status, indicators)

        return QueueHealthStatus(
            overall_status=overall_status,
            indicators=indicators,
            score=overall_score,
            recommendations=recommendations,
        )

    async def get_health_indicators(
        self, queue_type: str = "ingestion_queue"
    ) -> list[HealthIndicator]:
        """
        Get all individual health indicators.

        Args:
            queue_type: Queue type to assess

        Returns:
            List of health indicators
        """
        indicators = []

        # Get current statistics
        stats = await self.stats_collector.get_current_statistics(queue_type=queue_type)

        # Check queue backlog
        indicators.append(await self._check_queue_backlog(stats))

        # Check processing rate
        indicators.append(await self._check_processing_rate(stats))

        # Check error rate
        indicators.append(await self._check_error_rate(stats))

        # Check success rate
        indicators.append(await self._check_success_rate(stats))

        # Check backpressure if detector available
        if self.backpressure_detector:
            indicators.append(
                await self._check_backpressure(queue_type=queue_type)
            )

        # Check latency and resource usage if performance collector available
        if self.performance_collector:
            indicators.append(await self._check_latency())
            indicators.append(await self._check_resource_usage())

        # Run custom health checks
        custom_indicators = await self._run_custom_checks()
        indicators.extend(custom_indicators)

        return indicators

    async def calculate_health_score(
        self, indicators: list[HealthIndicator]
    ) -> float:
        """
        Calculate weighted overall health score from indicators.

        Args:
            indicators: List of health indicators

        Returns:
            Overall health score 0-100 (100 = perfect health)
        """
        if not indicators:
            return 0.0

        # Build weight mapping
        weight_map = {
            "queue_backlog": self.weights.backlog,
            "processing_rate": self.weights.processing_rate,
            "error_rate": self.weights.error_rate,
            "latency": self.weights.latency,
            "success_rate": self.weights.success_rate,
            "backpressure": self.weights.backpressure,
            "resource_usage": self.weights.resource_usage,
        }

        total_weighted_score = 0.0
        total_weight = 0.0

        for indicator in indicators:
            # Get weight for this indicator
            weight = weight_map.get(indicator.name, 0.0)

            # Check if this is a custom check
            if indicator.name not in weight_map:
                async with self._lock:
                    if indicator.name in self._custom_checks:
                        _, weight = self._custom_checks[indicator.name]

            if weight > 0:
                total_weighted_score += indicator.score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.0

        # Calculate weighted average
        overall_score = total_weighted_score / total_weight

        # Ensure score is in 0-100 range
        return max(0.0, min(100.0, overall_score))

    async def is_healthy(self, queue_type: str = "ingestion_queue") -> bool:
        """
        Check if queue is in healthy state.

        Args:
            queue_type: Queue type to check

        Returns:
            True if health status is HEALTHY, False otherwise
        """
        health_status = await self.calculate_health(queue_type=queue_type)
        return health_status.overall_status == HealthStatus.HEALTHY

    def register_health_check(
        self, name: str, check_func: Callable, weight: float
    ) -> bool:
        """
        Register a custom health check.

        The check function should be async and return a tuple:
        (status: HealthStatus, value: float, threshold: float, message: str)

        Args:
            name: Unique name for this health check
            check_func: Async callable that returns health check result
            weight: Weight for this check in overall score calculation

        Returns:
            True if registered successfully, False if name already exists
        """
        if not callable(check_func):
            raise ValueError(f"check_func must be callable, got {type(check_func)}")

        if weight < 0:
            raise ValueError(f"weight must be non-negative, got {weight}")

        # Store check function and weight
        if name in self._custom_checks:
            logger.warning(f"Health check '{name}' already registered, overwriting")
            return False

        self._custom_checks[name] = (check_func, weight)
        logger.info(f"Registered custom health check: {name} (weight={weight})")
        return True

    def unregister_health_check(self, name: str) -> bool:
        """
        Unregister a custom health check.

        Args:
            name: Name of health check to remove

        Returns:
            True if unregistered, False if not found
        """
        if name in self._custom_checks:
            del self._custom_checks[name]
            logger.info(f"Unregistered custom health check: {name}")
            return True
        return False

    async def _check_queue_backlog(self, stats) -> HealthIndicator:
        """Check queue backlog health."""
        queue_size = stats.queue_size
        threshold = self.thresholds.backlog_normal

        # Determine status and score based on backlog
        if queue_size <= self.thresholds.backlog_normal:
            status = HealthStatus.HEALTHY
            score = 100.0
            message = f"Queue backlog normal ({queue_size} items)"
        elif queue_size <= self.thresholds.backlog_warning:
            status = HealthStatus.DEGRADED
            # Linear interpolation between normal and warning
            score = 80.0 - (
                (queue_size - self.thresholds.backlog_normal)
                / (self.thresholds.backlog_warning - self.thresholds.backlog_normal)
                * 20.0
            )
            message = f"Queue backlog elevated ({queue_size} items)"
        elif queue_size <= self.thresholds.backlog_critical:
            status = HealthStatus.UNHEALTHY
            # Linear interpolation between warning and critical
            score = 40.0 - (
                (queue_size - self.thresholds.backlog_warning)
                / (self.thresholds.backlog_critical - self.thresholds.backlog_warning)
                * 20.0
            )
            message = f"Queue backlog high ({queue_size} items)"
        else:
            status = HealthStatus.CRITICAL
            score = 0.0
            message = f"Queue backlog critical ({queue_size} items)"

        return HealthIndicator(
            name="queue_backlog",
            status=status,
            value=float(queue_size),
            threshold=threshold,
            message=message,
            score=score,
        )

    async def _check_processing_rate(self, stats) -> HealthIndicator:
        """Check processing rate health."""
        processing_rate = stats.processing_rate
        threshold = self.thresholds.processing_rate_min

        # Determine status and score
        if processing_rate >= threshold:
            status = HealthStatus.HEALTHY
            # Score increases with processing rate
            score = min(100.0, 80.0 + (processing_rate / threshold) * 20.0)
            message = f"Processing rate good ({processing_rate:.1f} items/min)"
        elif processing_rate >= threshold * 0.5:
            status = HealthStatus.DEGRADED
            score = 60.0 + (processing_rate / threshold) * 20.0
            message = f"Processing rate below target ({processing_rate:.1f} items/min)"
        elif processing_rate > 0:
            status = HealthStatus.UNHEALTHY
            score = (processing_rate / threshold) * 40.0
            message = f"Processing rate low ({processing_rate:.1f} items/min)"
        else:
            status = HealthStatus.CRITICAL
            score = 0.0
            message = "Processing stopped (0 items/min)"

        return HealthIndicator(
            name="processing_rate",
            status=status,
            value=processing_rate,
            threshold=threshold,
            message=message,
            score=score,
        )

    async def _check_error_rate(self, stats) -> HealthIndicator:
        """Check error rate health."""
        error_rate = stats.failure_rate
        threshold = self.thresholds.error_rate_max

        # Determine status and score (lower is better for error rate)
        if error_rate <= threshold * 0.2:
            status = HealthStatus.HEALTHY
            score = 100.0 - (error_rate / threshold) * 20.0
            message = f"Error rate low ({error_rate:.1f}%)"
        elif error_rate <= threshold:
            status = HealthStatus.DEGRADED
            score = 80.0 - (error_rate / threshold) * 20.0
            message = f"Error rate elevated ({error_rate:.1f}%)"
        elif error_rate <= threshold * 2:
            status = HealthStatus.UNHEALTHY
            score = 40.0 - ((error_rate - threshold) / threshold) * 20.0
            message = f"Error rate high ({error_rate:.1f}%)"
        else:
            status = HealthStatus.CRITICAL
            score = 0.0
            message = f"Error rate critical ({error_rate:.1f}%)"

        return HealthIndicator(
            name="error_rate",
            status=status,
            value=error_rate,
            threshold=threshold,
            message=message,
            score=max(0.0, score),
        )

    async def _check_latency(self) -> HealthIndicator:
        """Check latency health."""
        if not self.performance_collector:
            return HealthIndicator(
                name="latency",
                status=HealthStatus.HEALTHY,
                value=0.0,
                threshold=0.0,
                message="Latency monitoring not available",
                score=100.0,
            )

        latency_metrics = await self.performance_collector.get_latency_metrics()
        avg_latency = latency_metrics.avg_latency_ms
        threshold = self.thresholds.latency_warning_ms

        # Determine status and score
        if avg_latency <= threshold * 0.5:
            status = HealthStatus.HEALTHY
            score = 100.0 - (avg_latency / threshold) * 20.0
            message = f"Latency good ({avg_latency:.0f}ms)"
        elif avg_latency <= threshold:
            status = HealthStatus.DEGRADED
            score = 80.0 - ((avg_latency - threshold * 0.5) / (threshold * 0.5)) * 20.0
            message = f"Latency elevated ({avg_latency:.0f}ms)"
        elif avg_latency <= self.thresholds.latency_critical_ms:
            status = HealthStatus.UNHEALTHY
            score = 40.0 - (
                (avg_latency - threshold)
                / (self.thresholds.latency_critical_ms - threshold)
                * 40.0
            )
            message = f"Latency high ({avg_latency:.0f}ms)"
        else:
            status = HealthStatus.CRITICAL
            score = 0.0
            message = f"Latency critical ({avg_latency:.0f}ms)"

        return HealthIndicator(
            name="latency",
            status=status,
            value=avg_latency,
            threshold=threshold,
            message=message,
            score=max(0.0, score),
        )

    async def _check_success_rate(self, stats) -> HealthIndicator:
        """Check success rate health."""
        success_rate = stats.success_rate
        threshold = self.thresholds.success_rate_min

        # Determine status and score
        if success_rate >= threshold:
            status = HealthStatus.HEALTHY
            score = 80.0 + (success_rate - threshold) / (100.0 - threshold) * 20.0
            message = f"Success rate good ({success_rate:.1f}%)"
        elif success_rate >= threshold - 10.0:
            status = HealthStatus.DEGRADED
            score = 60.0 + (success_rate - (threshold - 10.0)) / 10.0 * 20.0
            message = f"Success rate below target ({success_rate:.1f}%)"
        elif success_rate >= threshold - 20.0:
            status = HealthStatus.UNHEALTHY
            score = 20.0 + (success_rate - (threshold - 20.0)) / 10.0 * 40.0
            message = f"Success rate low ({success_rate:.1f}%)"
        else:
            status = HealthStatus.CRITICAL
            score = 0.0
            message = f"Success rate critical ({success_rate:.1f}%)"

        return HealthIndicator(
            name="success_rate",
            status=status,
            value=success_rate,
            threshold=threshold,
            message=message,
            score=max(0.0, min(100.0, score)),
        )

    async def _check_backpressure(self, queue_type: str) -> HealthIndicator:
        """Check backpressure health."""
        if not self.backpressure_detector:
            return HealthIndicator(
                name="backpressure",
                status=HealthStatus.HEALTHY,
                value=0.0,
                threshold=0.0,
                message="Backpressure monitoring not available",
                score=100.0,
            )

        alert = await self.backpressure_detector.detect_backpressure(queue_type=queue_type)

        if not alert or alert.severity == BackpressureSeverity.NONE:
            return HealthIndicator(
                name="backpressure",
                status=HealthStatus.HEALTHY,
                value=0.0,
                threshold=0.0,
                message="No backpressure detected",
                score=100.0,
            )

        # Map backpressure severity to health status
        severity_map = {
            BackpressureSeverity.LOW: (HealthStatus.DEGRADED, 70.0),
            BackpressureSeverity.MEDIUM: (HealthStatus.UNHEALTHY, 50.0),
            BackpressureSeverity.HIGH: (HealthStatus.UNHEALTHY, 30.0),
            BackpressureSeverity.CRITICAL: (HealthStatus.CRITICAL, 0.0),
        }

        status, score = severity_map.get(
            alert.severity, (HealthStatus.HEALTHY, 100.0)
        )

        return HealthIndicator(
            name="backpressure",
            status=status,
            value=alert.indicators.queue_growth_rate,
            threshold=0.0,
            message=f"Backpressure: {alert.severity.value}",
            score=score,
        )

    async def _check_resource_usage(self) -> HealthIndicator:
        """Check resource usage health."""
        if not self.performance_collector or not self.performance_collector.enable_resource_tracking:
            return HealthIndicator(
                name="resource_usage",
                status=HealthStatus.HEALTHY,
                value=0.0,
                threshold=0.0,
                message="Resource monitoring not available",
                score=100.0,
            )

        resource_usage = self.performance_collector._get_resource_usage()

        if not resource_usage:
            return HealthIndicator(
                name="resource_usage",
                status=HealthStatus.HEALTHY,
                value=0.0,
                threshold=0.0,
                message="Resource data unavailable",
                score=100.0,
            )

        cpu_percent = resource_usage.get("cpu_percent", 0.0)
        memory_mb = resource_usage.get("memory_mb", 0.0)

        # Check CPU
        cpu_status = HealthStatus.HEALTHY
        cpu_score = 100.0
        if cpu_percent >= self.thresholds.cpu_critical_percent:
            cpu_status = HealthStatus.CRITICAL
            cpu_score = 0.0
        elif cpu_percent >= self.thresholds.cpu_warning_percent:
            cpu_status = HealthStatus.DEGRADED
            cpu_score = 60.0

        # Check memory
        memory_status = HealthStatus.HEALTHY
        memory_score = 100.0
        if memory_mb >= self.thresholds.memory_critical_mb:
            memory_status = HealthStatus.CRITICAL
            memory_score = 0.0
        elif memory_mb >= self.thresholds.memory_warning_mb:
            memory_status = HealthStatus.DEGRADED
            memory_score = 60.0

        # Overall resource status is worst of CPU and memory
        if cpu_status == HealthStatus.CRITICAL or memory_status == HealthStatus.CRITICAL:
            status = HealthStatus.CRITICAL
            score = 0.0
        elif cpu_status == HealthStatus.DEGRADED or memory_status == HealthStatus.DEGRADED:
            status = HealthStatus.DEGRADED
            score = (cpu_score + memory_score) / 2
        else:
            status = HealthStatus.HEALTHY
            score = 100.0

        message = f"CPU: {cpu_percent:.1f}%, Memory: {memory_mb:.0f}MB"

        return HealthIndicator(
            name="resource_usage",
            status=status,
            value=cpu_percent,  # Use CPU as primary value
            threshold=self.thresholds.cpu_warning_percent,
            message=message,
            score=score,
        )

    async def _run_custom_checks(self) -> list[HealthIndicator]:
        """Run all registered custom health checks."""
        indicators = []

        async with self._lock:
            custom_checks = list(self._custom_checks.items())

        for name, (check_func, _) in custom_checks:
            try:
                # Call custom check function
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()

                # Unpack result
                status, value, threshold, message = result

                # Calculate score based on status
                status_scores = {
                    HealthStatus.HEALTHY: 100.0,
                    HealthStatus.DEGRADED: 70.0,
                    HealthStatus.UNHEALTHY: 40.0,
                    HealthStatus.CRITICAL: 0.0,
                }
                score = status_scores.get(status, 50.0)

                indicators.append(
                    HealthIndicator(
                        name=name,
                        status=status,
                        value=value,
                        threshold=threshold,
                        message=message,
                        score=score,
                    )
                )

            except Exception as e:
                logger.error(f"Error running custom health check '{name}': {e}")
                # Add failed indicator
                indicators.append(
                    HealthIndicator(
                        name=name,
                        status=HealthStatus.CRITICAL,
                        value=0.0,
                        threshold=0.0,
                        message=f"Health check failed: {e}",
                        score=0.0,
                    )
                )

        return indicators

    def _determine_health_status(self, score: float) -> HealthStatus:
        """
        Determine overall health status from score.

        Args:
            score: Health score 0-100

        Returns:
            Health status classification
        """
        if score >= 80.0:
            return HealthStatus.HEALTHY
        elif score >= 60.0:
            return HealthStatus.DEGRADED
        elif score >= 40.0:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL

    def _generate_recommendations(
        self, status: HealthStatus, indicators: list[HealthIndicator]
    ) -> list[str]:
        """
        Generate actionable recommendations based on health status.

        Args:
            status: Overall health status
            indicators: List of health indicators

        Returns:
            List of recommended actions
        """
        recommendations = []

        # Overall status recommendations
        if status == HealthStatus.HEALTHY:
            recommendations.append("Queue operating normally")
            return recommendations

        # Analyze failing indicators
        failing_indicators = [
            ind
            for ind in indicators
            if ind.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL)
        ]

        degraded_indicators = [
            ind for ind in indicators if ind.status == HealthStatus.DEGRADED
        ]

        # Critical status recommendations
        if status == HealthStatus.CRITICAL:
            recommendations.append("URGENT: Immediate action required")
            for ind in failing_indicators:
                if ind.name == "queue_backlog":
                    recommendations.append("Emergency: Clear queue backlog immediately")
                    recommendations.append("Consider pausing ingestion temporarily")
                elif ind.name == "processing_rate":
                    recommendations.append("Emergency: Scale up processing workers")
                    recommendations.append("Investigate processing bottlenecks")
                elif ind.name == "error_rate":
                    recommendations.append("Critical: High error rate detected")
                    recommendations.append("Review recent error messages immediately")
                elif ind.name == "latency":
                    recommendations.append("Critical: Very high latency detected")
                    recommendations.append("Check for resource constraints")
                elif ind.name == "resource_usage":
                    recommendations.append("Critical: Resource limits reached")
                    recommendations.append("Add more resources or scale horizontally")

        # Unhealthy status recommendations
        elif status == HealthStatus.UNHEALTHY:
            recommendations.append("Action needed: Queue health degraded")
            for ind in failing_indicators:
                if ind.name == "queue_backlog":
                    recommendations.append("Scale up workers to reduce backlog")
                elif ind.name == "processing_rate":
                    recommendations.append("Increase processing capacity")
                elif ind.name == "error_rate":
                    recommendations.append("Investigate and fix recurring errors")
                elif ind.name == "success_rate":
                    recommendations.append("Review failed items and processing logic")

        # Degraded status recommendations
        elif status == HealthStatus.DEGRADED:
            recommendations.append("Monitor closely, consider scaling")
            for ind in degraded_indicators:
                if ind.name == "queue_backlog":
                    recommendations.append("Monitor backlog growth trend")
                elif ind.name == "processing_rate":
                    recommendations.append("Consider adding workers if trend continues")
                elif ind.name == "latency":
                    recommendations.append("Monitor latency, check for slow operations")

        # Add backpressure-specific recommendations
        backpressure_ind = next(
            (ind for ind in indicators if ind.name == "backpressure"), None
        )
        if backpressure_ind and backpressure_ind.status != HealthStatus.HEALTHY:
            recommendations.append("Review backpressure alerts for detailed guidance")

        # Ensure at least one recommendation
        if not recommendations:
            recommendations.append("Monitor queue metrics closely")

        return recommendations
