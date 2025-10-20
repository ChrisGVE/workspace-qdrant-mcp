"""
Performance Degradation Tracking System (Task 305.8)

Comprehensive performance monitoring and degradation detection system for identifying
performance regressions during stress tests. Provides continuous monitoring, threshold
detection, trend analysis, and automated alerting.

Features:
1. Continuous performance metric tracking
2. Baseline establishment and comparison
3. Degradation threshold detection
4. Performance trend analysis (improving/degrading/stable)
5. Automated alerting for performance regressions
6. Multi-metric support (latency, throughput, error rate, resource usage)
7. Historical performance tracking
8. Percentile-based analysis (p50/p90/p95/p99)
"""

import time
import statistics
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque


logger = logging.getLogger(__name__)


class PerformanceMetricType(Enum):
    """Type of performance metric."""
    LATENCY = "latency"                # Lower is better (ms)
    THROUGHPUT = "throughput"          # Higher is better (ops/s)
    ERROR_RATE = "error_rate"          # Lower is better (%)
    CPU_USAGE = "cpu_usage"            # Lower is better (%)
    MEMORY_USAGE = "memory_usage"      # Lower is better (MB)
    DISK_IO = "disk_io"                # Context-dependent (MB/s)
    NETWORK_IO = "network_io"          # Context-dependent (MB/s)


class PerformanceTrend(Enum):
    """Performance trend direction."""
    IMPROVING = "improving"      # Performance getting better
    STABLE = "stable"            # Performance stable
    DEGRADING = "degrading"      # Performance getting worse
    UNKNOWN = "unknown"          # Not enough data


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class PerformanceBaseline:
    """Performance baseline metrics."""
    metric_name: str
    metric_type: PerformanceMetricType

    # Baseline statistics
    mean: float = 0.0
    median: float = 0.0
    stddev: float = 0.0
    min_value: float = float('inf')
    max_value: float = 0.0

    # Percentiles
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    # Sample info
    sample_count: int = 0
    sample_period_seconds: float = 0.0
    baseline_timestamp: float = 0.0


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class PerformanceDegradation:
    """Detected performance degradation."""
    metric_name: str
    metric_type: PerformanceMetricType
    timestamp: float

    # Degradation metrics
    baseline_value: float
    current_value: float
    degradation_percent: float
    threshold_percent: float

    # Severity
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""

    def __str__(self) -> str:
        """String representation."""
        return (
            f"{self.severity.value.upper()}: {self.metric_name} degradation detected - "
            f"baseline: {self.baseline_value:.2f}, current: {self.current_value:.2f} "
            f"({self.degradation_percent:+.1f}%, threshold: {self.threshold_percent:.1f}%) "
            f"at {self.timestamp:.2f}"
        )


@dataclass
class PerformanceTrendAnalysis:
    """Performance trend analysis results."""
    metric_name: str
    trend: PerformanceTrend = PerformanceTrend.UNKNOWN

    # Trend statistics
    trend_start_value: float = 0.0
    trend_end_value: float = 0.0
    trend_change_percent: float = 0.0
    trend_confidence: float = 0.0  # 0-1, higher is more confident

    # Regression analysis
    slope: float = 0.0  # Rate of change
    r_squared: float = 0.0  # How well trend fits data

    sample_count: int = 0
    analysis_period_seconds: float = 0.0


class PerformanceDegradationTracker:
    """
    Tracks performance degradation with continuous monitoring and alerting.

    Monitors performance metrics continuously, compares against baselines,
    detects degradation, analyzes trends, and triggers alerts.
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[PerformanceDegradation], None]] = None,
        default_degradation_threshold_percent: float = 20.0,
        trend_window_size: int = 50,
        min_samples_for_trend: int = 10
    ):
        """
        Initialize performance degradation tracker.

        Args:
            alert_callback: Callback function for alerts (receives PerformanceDegradation)
            default_degradation_threshold_percent: Default degradation threshold
            trend_window_size: Number of samples for trend analysis
            min_samples_for_trend: Minimum samples needed for trend analysis
        """
        self.alert_callback = alert_callback
        self.default_threshold = default_degradation_threshold_percent
        self.trend_window_size = trend_window_size
        self.min_samples_for_trend = min_samples_for_trend

        # Baselines (metric_name -> baseline)
        self.baselines: Dict[str, PerformanceBaseline] = {}

        # Custom thresholds (metric_name -> threshold_percent)
        self.thresholds: Dict[str, float] = {}

        # Performance history (metric_name -> deque of (timestamp, value))
        self.history: Dict[str, deque] = {}

        # Detected degradations
        self.degradations: List[PerformanceDegradation] = []

        # Snapshots history
        self.snapshots: List[PerformanceSnapshot] = []

        # Metric type registry (metric_name -> type)
        self.metric_types: Dict[str, PerformanceMetricType] = {}

    def register_metric(
        self,
        metric_name: str,
        metric_type: PerformanceMetricType,
        threshold_percent: Optional[float] = None
    ):
        """
        Register a performance metric for tracking.

        Args:
            metric_name: Name of the metric
            metric_type: Type of performance metric
            threshold_percent: Custom degradation threshold (None for default)
        """
        self.metric_types[metric_name] = metric_type
        self.history[metric_name] = deque(maxlen=self.trend_window_size)

        if threshold_percent is not None:
            self.thresholds[metric_name] = threshold_percent

        logger.info(
            f"Registered metric '{metric_name}' (type: {metric_type.value}, "
            f"threshold: {threshold_percent or self.default_threshold}%)"
        )

    def establish_baseline(
        self,
        metric_name: str,
        samples: List[float],
        metric_type: Optional[PerformanceMetricType] = None
    ) -> PerformanceBaseline:
        """
        Establish performance baseline from samples.

        Args:
            metric_name: Name of the metric
            samples: Sample values for baseline
            metric_type: Type of metric (auto-detected if registered)

        Returns:
            PerformanceBaseline with statistics
        """
        if not samples:
            raise ValueError(f"Cannot establish baseline for {metric_name}: no samples")

        # Get or auto-detect metric type
        if metric_type is None:
            if metric_name not in self.metric_types:
                raise ValueError(f"Metric {metric_name} not registered and no type provided")
            metric_type = self.metric_types[metric_name]
        else:
            self.metric_types[metric_name] = metric_type

        # Calculate baseline statistics
        baseline = PerformanceBaseline(
            metric_name=metric_name,
            metric_type=metric_type,
            mean=statistics.mean(samples),
            median=statistics.median(samples),
            min_value=min(samples),
            max_value=max(samples),
            sample_count=len(samples),
            baseline_timestamp=time.time()
        )

        # Calculate stddev if enough samples
        if len(samples) >= 2:
            baseline.stddev = statistics.stdev(samples)

        # Calculate percentiles if enough samples
        if len(samples) >= 10:
            sorted_samples = sorted(samples)
            baseline.p50 = sorted_samples[int(len(sorted_samples) * 0.50)]
            baseline.p90 = sorted_samples[min(int(len(sorted_samples) * 0.90), len(sorted_samples) - 1)]
            baseline.p95 = sorted_samples[min(int(len(sorted_samples) * 0.95), len(sorted_samples) - 1)]
            baseline.p99 = sorted_samples[min(int(len(sorted_samples) * 0.99), len(sorted_samples) - 1)]

        self.baselines[metric_name] = baseline

        logger.info(
            f"Established baseline for '{metric_name}': "
            f"mean={baseline.mean:.2f}, median={baseline.median:.2f}, "
            f"p95={baseline.p95:.2f}, samples={baseline.sample_count}"
        )

        return baseline

    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[float] = None,
        check_degradation: bool = True
    ):
        """
        Record a performance metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Timestamp (defaults to now)
            check_degradation: Whether to check for degradation
        """
        timestamp = timestamp or time.time()

        # Ensure metric is registered
        if metric_name not in self.history:
            logger.warning(f"Metric '{metric_name}' not registered, auto-registering with LATENCY type")
            self.register_metric(metric_name, PerformanceMetricType.LATENCY)

        # Add to history
        self.history[metric_name].append((timestamp, value))

        # Check for degradation if enabled and baseline exists
        if check_degradation and metric_name in self.baselines:
            degradation = self._check_degradation(metric_name, value, timestamp)
            if degradation:
                self.degradations.append(degradation)
                if self.alert_callback:
                    self.alert_callback(degradation)

    def record_snapshot(
        self,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, any]] = None
    ) -> PerformanceSnapshot:
        """
        Record a snapshot of multiple metrics at once.

        Args:
            metrics: Dictionary of metric_name -> value
            metadata: Optional metadata

        Returns:
            PerformanceSnapshot
        """
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            metrics=metrics.copy(),
            metadata=metadata or {}
        )

        # Record each metric
        for metric_name, value in metrics.items():
            self.record_metric(metric_name, value, snapshot.timestamp)

        self.snapshots.append(snapshot)
        return snapshot

    def _check_degradation(
        self,
        metric_name: str,
        current_value: float,
        timestamp: float
    ) -> Optional[PerformanceDegradation]:
        """
        Check if current value represents degradation from baseline.

        Args:
            metric_name: Name of the metric
            current_value: Current metric value
            timestamp: Current timestamp

        Returns:
            PerformanceDegradation if degradation detected, None otherwise
        """
        baseline = self.baselines.get(metric_name)
        if not baseline:
            return None

        metric_type = baseline.metric_type
        baseline_value = baseline.mean
        threshold = self.thresholds.get(metric_name, self.default_threshold)

        # Calculate degradation based on metric type
        # For LATENCY, ERROR_RATE, CPU, MEMORY: higher is worse
        # For THROUGHPUT: lower is worse
        if metric_type in [
            PerformanceMetricType.LATENCY,
            PerformanceMetricType.ERROR_RATE,
            PerformanceMetricType.CPU_USAGE,
            PerformanceMetricType.MEMORY_USAGE
        ]:
            # Higher is worse - check if current > baseline
            if baseline_value == 0:
                return None
            degradation_percent = ((current_value - baseline_value) / baseline_value) * 100
            is_degraded = degradation_percent > threshold

        elif metric_type == PerformanceMetricType.THROUGHPUT:
            # Lower is worse - check if current < baseline
            if baseline_value == 0:
                return None
            degradation_percent = ((baseline_value - current_value) / baseline_value) * 100
            is_degraded = degradation_percent > threshold

        else:
            # For DISK_IO, NETWORK_IO: context-dependent, skip for now
            return None

        if not is_degraded:
            return None

        # Determine severity
        if degradation_percent >= threshold * 2:
            severity = AlertSeverity.CRITICAL
        elif degradation_percent >= threshold:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        degradation = PerformanceDegradation(
            metric_name=metric_name,
            metric_type=metric_type,
            timestamp=timestamp,
            baseline_value=baseline_value,
            current_value=current_value,
            degradation_percent=degradation_percent,
            threshold_percent=threshold,
            severity=severity,
            message=f"{metric_name} degraded by {degradation_percent:.1f}%"
        )

        logger.warning(str(degradation))
        return degradation

    def analyze_trend(
        self,
        metric_name: str,
        lookback_count: Optional[int] = None
    ) -> PerformanceTrendAnalysis:
        """
        Analyze performance trend for a metric.

        Args:
            metric_name: Name of the metric
            lookback_count: Number of recent samples to analyze (None for all)

        Returns:
            PerformanceTrendAnalysis
        """
        if metric_name not in self.history or not self.history[metric_name]:
            return PerformanceTrendAnalysis(
                metric_name=metric_name,
                trend=PerformanceTrend.UNKNOWN
            )

        # Get samples
        samples = list(self.history[metric_name])
        if lookback_count:
            samples = samples[-lookback_count:]

        if len(samples) < self.min_samples_for_trend:
            return PerformanceTrendAnalysis(
                metric_name=metric_name,
                trend=PerformanceTrend.UNKNOWN,
                sample_count=len(samples)
            )

        # Extract values and calculate linear regression
        timestamps = [t for t, _ in samples]
        values = [v for _, v in samples]

        # Simple linear regression: y = mx + b
        n = len(values)
        x = list(range(n))  # Use indices as x values
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(values)

        # Calculate slope (m)
        numerator = sum((x[i] - mean_x) * (values[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Calculate R-squared
        y_pred = [mean_y + slope * (x[i] - mean_x) for i in range(n)]
        ss_tot = sum((values[i] - mean_y) ** 2 for i in range(n))
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))

        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)

        # Calculate trend
        start_value = values[0]
        end_value = values[-1]
        if start_value == 0:
            trend_change_percent = 0
        else:
            trend_change_percent = ((end_value - start_value) / start_value) * 100

        # Determine trend direction based on metric type
        metric_type = self.metric_types.get(metric_name, PerformanceMetricType.LATENCY)

        if metric_type in [
            PerformanceMetricType.LATENCY,
            PerformanceMetricType.ERROR_RATE,
            PerformanceMetricType.CPU_USAGE,
            PerformanceMetricType.MEMORY_USAGE
        ]:
            # For these metrics, decreasing is improving
            if abs(trend_change_percent) < 5:
                trend = PerformanceTrend.STABLE
            elif trend_change_percent < 0:
                trend = PerformanceTrend.IMPROVING
            else:
                trend = PerformanceTrend.DEGRADING
        elif metric_type == PerformanceMetricType.THROUGHPUT:
            # For throughput, increasing is improving
            if abs(trend_change_percent) < 5:
                trend = PerformanceTrend.STABLE
            elif trend_change_percent > 0:
                trend = PerformanceTrend.IMPROVING
            else:
                trend = PerformanceTrend.DEGRADING
        else:
            trend = PerformanceTrend.UNKNOWN

        # Confidence based on R-squared and sample count
        confidence = r_squared * min(len(samples) / 50, 1.0)

        analysis = PerformanceTrendAnalysis(
            metric_name=metric_name,
            trend=trend,
            trend_start_value=start_value,
            trend_end_value=end_value,
            trend_change_percent=trend_change_percent,
            trend_confidence=confidence,
            slope=slope,
            r_squared=r_squared,
            sample_count=len(samples),
            analysis_period_seconds=timestamps[-1] - timestamps[0] if timestamps else 0
        )

        logger.info(
            f"Trend analysis for '{metric_name}': {trend.value} "
            f"({trend_change_percent:+.1f}%, confidence: {confidence:.2f})"
        )

        return analysis

    def get_degradations(
        self,
        severity: Optional[AlertSeverity] = None,
        since_timestamp: Optional[float] = None
    ) -> List[PerformanceDegradation]:
        """
        Get detected degradations.

        Args:
            severity: Filter by severity (None for all)
            since_timestamp: Only degradations after this timestamp

        Returns:
            List of PerformanceDegradation
        """
        degradations = self.degradations

        if severity:
            degradations = [d for d in degradations if d.severity == severity]

        if since_timestamp:
            degradations = [d for d in degradations if d.timestamp >= since_timestamp]

        return degradations

    def reset(self):
        """Reset all tracking data."""
        self.baselines.clear()
        self.history.clear()
        self.degradations.clear()
        self.snapshots.clear()
        logger.info("Performance degradation tracker reset")
