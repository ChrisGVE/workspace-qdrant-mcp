"""
Recovery Time Measurement System (Task 305.7)

Comprehensive recovery time tracking and analysis system for measuring system
resilience and recovery characteristics. Implements MTTR (Mean Time To Recovery)
and MTTF (Mean Time To Failure) calculations with detailed statistical analysis.

Features:
1. MTTR/MTTF calculation and tracking
2. Automated recovery detection
3. Performance baseline restoration tracking
4. Recovery time statistical analysis (percentiles, trends, distributions)
5. Component-level and system-level metrics
6. Historical trend analysis
7. Recovery success rate tracking
8. Performance degradation during recovery tracking
"""

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class RecoveryState(Enum):
    """Recovery state for a component or system."""
    HEALTHY = "healthy"                  # Operating normally
    DEGRADED = "degraded"                # Performance degraded but functional
    FAILED = "failed"                    # Failed and not recovering
    RECOVERING = "recovering"            # In process of recovery
    RECOVERED = "recovered"              # Successfully recovered


class RecoveryDetectionStrategy(Enum):
    """Strategy for detecting recovery completion."""
    HEALTH_CHECK = "health_check"        # Component health check passes
    PERFORMANCE_BASELINE = "baseline"     # Performance returns to baseline
    HYBRID = "hybrid"                    # Both health and performance criteria


@dataclass
class RecoveryEvent:
    """Single recovery event record."""
    component_name: str
    failure_time: float
    detection_time: float
    recovery_start_time: float
    recovery_complete_time: float | None = None
    recovery_strategy: RecoveryDetectionStrategy = RecoveryDetectionStrategy.HEALTH_CHECK

    # Recovery metrics
    recovery_duration: float | None = None  # Time from start to complete
    time_to_detect: float | None = None     # Time from failure to detection
    time_to_recover: float | None = None    # Total time from failure to recovery

    # Performance metrics during recovery
    baseline_performance: dict[str, float] | None = None
    recovery_performance: dict[str, float] | None = None
    performance_degradation_percent: float = 0.0

    # Recovery outcome
    recovery_successful: bool = False
    recovery_state: RecoveryState = RecoveryState.FAILED
    failure_reason: str | None = None
    recovery_notes: str = ""

    def complete_recovery(
        self,
        recovery_time: float,
        successful: bool = True,
        performance: dict[str, float] | None = None
    ):
        """Mark recovery as complete."""
        self.recovery_complete_time = recovery_time
        self.recovery_successful = successful
        self.recovery_state = RecoveryState.RECOVERED if successful else RecoveryState.FAILED

        # Calculate durations
        self.recovery_duration = recovery_time - self.recovery_start_time
        self.time_to_detect = self.detection_time - self.failure_time
        self.time_to_recover = recovery_time - self.failure_time

        # Calculate performance degradation if baseline and current available
        if performance and self.baseline_performance:
            self.recovery_performance = performance
            self._calculate_degradation()

    def _calculate_degradation(self):
        """Calculate performance degradation percentage."""
        if not self.baseline_performance or not self.recovery_performance:
            return

        degradations = []
        for metric, baseline_val in self.baseline_performance.items():
            if baseline_val == 0:
                continue

            recovery_val = self.recovery_performance.get(metric, baseline_val)
            # For latency/time metrics, higher is worse
            if "latency" in metric.lower() or "time" in metric.lower():
                deg = ((recovery_val - baseline_val) / baseline_val) * 100
            # For throughput/rate metrics, lower is worse
            else:
                deg = ((baseline_val - recovery_val) / baseline_val) * 100

            degradations.append(max(0, deg))  # Only positive degradation

        if degradations:
            self.performance_degradation_percent = statistics.mean(degradations)


@dataclass
class MTTRAnalysis:
    """Mean Time To Recovery analysis results."""
    component_name: str

    # Basic MTTR metrics
    mean_recovery_time: float = 0.0
    median_recovery_time: float = 0.0
    min_recovery_time: float = float('inf')
    max_recovery_time: float = 0.0

    # Statistical analysis
    stddev_recovery_time: float = 0.0
    p50_recovery_time: float = 0.0
    p90_recovery_time: float = 0.0
    p95_recovery_time: float = 0.0
    p99_recovery_time: float = 0.0

    # Count metrics
    total_recoveries: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0
    recovery_success_rate: float = 0.0

    # Timing breakdown
    mean_detection_time: float = 0.0
    mean_recovery_duration: float = 0.0

    # Performance impact
    mean_degradation_percent: float = 0.0

    # Trend analysis
    recovery_time_trend: str = "stable"  # improving, degrading, stable


@dataclass
class MTTFAnalysis:
    """Mean Time To Failure analysis results."""
    component_name: str

    # Basic MTTF metrics
    mean_time_to_failure: float = 0.0
    median_time_to_failure: float = 0.0
    min_time_to_failure: float = float('inf')
    max_time_to_failure: float = 0.0

    # Statistical analysis
    stddev_time_to_failure: float = 0.0
    p50_time_to_failure: float = 0.0
    p90_time_to_failure: float = 0.0
    p95_time_to_failure: float = 0.0

    # Failure rate
    failure_rate_per_hour: float = 0.0
    total_failures: int = 0
    observation_period_hours: float = 0.0

    # Reliability metrics
    availability_percent: float = 100.0  # Uptime percentage


@dataclass
class RecoveryMetricsReport:
    """Comprehensive recovery metrics report."""
    component_metrics: dict[str, tuple[MTTRAnalysis, MTTFAnalysis]] = field(default_factory=dict)
    system_mttr: float = 0.0
    system_mttf: float = 0.0
    system_availability: float = 100.0
    total_recovery_events: int = 0
    overall_success_rate: float = 0.0
    observation_start_time: float = 0.0
    observation_end_time: float = 0.0


class RecoveryTimeTracker:
    """
    Tracks recovery times and calculates MTTR/MTTF metrics.

    Provides comprehensive recovery time tracking with automated recovery
    detection, performance baseline restoration tracking, and detailed
    statistical analysis.
    """

    def __init__(
        self,
        baseline_performance: dict[str, float] | None = None,
        recovery_detection_strategy: RecoveryDetectionStrategy = RecoveryDetectionStrategy.HEALTH_CHECK
    ):
        """
        Initialize recovery time tracker.

        Args:
            baseline_performance: Baseline performance metrics for comparison
            recovery_detection_strategy: Strategy for detecting recovery completion
        """
        self.baseline_performance = baseline_performance or {}
        self.recovery_strategy = recovery_detection_strategy

        # Active recovery events (component -> event)
        self.active_recoveries: dict[str, RecoveryEvent] = {}

        # Historical recovery events
        self.recovery_history: list[RecoveryEvent] = []

        # Failure timestamps for MTTF calculation (component -> [timestamps])
        self.failure_times: dict[str, deque] = {}

        # Last healthy timestamp (component -> timestamp)
        self.last_healthy_time: dict[str, float] = {}

        # Observation period
        self.observation_start = time.time()

    def record_failure(
        self,
        component_name: str,
        failure_time: float | None = None,
        failure_reason: str | None = None
    ):
        """
        Record component failure.

        Args:
            component_name: Name of failed component
            failure_time: Timestamp of failure (defaults to now)
            failure_reason: Reason for failure
        """
        failure_time = failure_time or time.time()

        # Record for MTTF calculation
        if component_name not in self.failure_times:
            self.failure_times[component_name] = deque(maxlen=100)  # Keep last 100
        self.failure_times[component_name].append(failure_time)

        # Create recovery event (use failure_time as base for detection/recovery start)
        event = RecoveryEvent(
            component_name=component_name,
            failure_time=failure_time,
            detection_time=failure_time,  # Assume immediate detection
            recovery_start_time=failure_time,  # Start recovery immediately
            recovery_strategy=self.recovery_strategy,
            baseline_performance=self.baseline_performance.copy(),
            failure_reason=failure_reason
        )

        self.active_recoveries[component_name] = event

        logger.info(
            f"Recorded failure for {component_name} at {failure_time:.2f}. "
            f"Reason: {failure_reason or 'unknown'}"
        )

    def record_recovery_start(
        self,
        component_name: str,
        start_time: float | None = None
    ):
        """
        Record start of recovery process.

        Args:
            component_name: Name of recovering component
            start_time: Timestamp of recovery start (defaults to now)
        """
        start_time = start_time or time.time()

        if component_name in self.active_recoveries:
            self.active_recoveries[component_name].recovery_start_time = start_time
            self.active_recoveries[component_name].recovery_state = RecoveryState.RECOVERING
            logger.info(f"Recovery started for {component_name} at {start_time:.2f}")

    def record_recovery_complete(
        self,
        component_name: str,
        recovery_time: float | None = None,
        successful: bool = True,
        current_performance: dict[str, float] | None = None
    ) -> RecoveryEvent | None:
        """
        Record completion of recovery.

        Args:
            component_name: Name of recovered component
            recovery_time: Timestamp of recovery completion (defaults to now)
            successful: Whether recovery was successful
            current_performance: Current performance metrics after recovery

        Returns:
            Completed RecoveryEvent or None if no active recovery
        """
        recovery_time = recovery_time or time.time()

        if component_name not in self.active_recoveries:
            logger.warning(f"No active recovery for {component_name}")
            return None

        event = self.active_recoveries.pop(component_name)
        event.complete_recovery(recovery_time, successful, current_performance)

        # Add to history
        self.recovery_history.append(event)

        # Update last healthy time
        if successful:
            self.last_healthy_time[component_name] = recovery_time

        logger.info(
            f"Recovery {'succeeded' if successful else 'failed'} for {component_name}. "
            f"Duration: {event.recovery_duration:.2f}s, "
            f"Total time: {event.time_to_recover:.2f}s"
        )

        return event

    def calculate_mttr(
        self,
        component_name: str | None = None,
        lookback_count: int | None = None
    ) -> MTTRAnalysis:
        """
        Calculate MTTR analysis.

        Args:
            component_name: Component to analyze (None for all)
            lookback_count: Number of recent events to analyze (None for all)

        Returns:
            MTTRAnalysis with recovery time statistics
        """
        # Filter events
        events = [
            e for e in self.recovery_history
            if (component_name is None or e.component_name == component_name)
            and e.recovery_successful
            and e.recovery_duration is not None
        ]

        if lookback_count:
            events = events[-lookback_count:]

        if not events:
            return MTTRAnalysis(
                component_name=component_name or "all",
                total_recoveries=0
            )

        # Extract metrics
        recovery_times = [e.recovery_duration for e in events]
        detection_times = [e.time_to_detect for e in events if e.time_to_detect]
        degradations = [e.performance_degradation_percent for e in events]

        # Calculate statistics
        analysis = MTTRAnalysis(
            component_name=component_name or "all",
            mean_recovery_time=statistics.mean(recovery_times),
            median_recovery_time=statistics.median(recovery_times),
            min_recovery_time=min(recovery_times),
            max_recovery_time=max(recovery_times),
            total_recoveries=len(events),
            successful_recoveries=sum(1 for e in events if e.recovery_successful),
            failed_recoveries=sum(1 for e in events if not e.recovery_successful)
        )

        # Calculate stddev and percentiles if enough data
        if len(recovery_times) >= 2:
            analysis.stddev_recovery_time = statistics.stdev(recovery_times)

        if len(recovery_times) >= 10:
            sorted_times = sorted(recovery_times)
            # Use proper percentile calculation (0-based indexing with rounding)
            analysis.p50_recovery_time = sorted_times[min(int(len(sorted_times) * 0.50), len(sorted_times) - 1)]
            analysis.p90_recovery_time = sorted_times[min(int(len(sorted_times) * 0.90), len(sorted_times) - 1)]
            analysis.p95_recovery_time = sorted_times[min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)]
            analysis.p99_recovery_time = sorted_times[min(int(len(sorted_times) * 0.99), len(sorted_times) - 1)]

        # Calculate success rate
        total_attempts = len([
            e for e in self.recovery_history
            if component_name is None or e.component_name == component_name
        ])
        if total_attempts > 0:
            analysis.recovery_success_rate = (analysis.successful_recoveries / total_attempts) * 100

        # Calculate mean detection and recovery duration
        if detection_times:
            analysis.mean_detection_time = statistics.mean(detection_times)
        analysis.mean_recovery_duration = analysis.mean_recovery_time

        # Calculate mean degradation
        if degradations:
            analysis.mean_degradation_percent = statistics.mean(degradations)

        # Analyze trend (simple: compare first half to second half)
        if len(recovery_times) >= 10:
            mid = len(recovery_times) // 2
            first_half_mean = statistics.mean(recovery_times[:mid])
            second_half_mean = statistics.mean(recovery_times[mid:])

            change_percent = ((second_half_mean - first_half_mean) / first_half_mean) * 100
            if abs(change_percent) < 10:
                analysis.recovery_time_trend = "stable"
            elif change_percent > 0:
                analysis.recovery_time_trend = "degrading"
            else:
                analysis.recovery_time_trend = "improving"

        return analysis

    def calculate_mttf(
        self,
        component_name: str | None = None,
        observation_period_hours: float | None = None
    ) -> MTTFAnalysis:
        """
        Calculate MTTF analysis.

        Args:
            component_name: Component to analyze (None for all)
            observation_period_hours: Hours of observation (None for auto)

        Returns:
            MTTFAnalysis with failure time statistics
        """
        # Determine observation period
        if observation_period_hours is None:
            # If we have recovery events, use them to determine observation period
            # (handles synthetic test timestamps better than wall clock)
            if self.recovery_history:
                earliest_time = min(e.failure_time for e in self.recovery_history)
                latest_time = max(
                    e.recovery_complete_time
                    for e in self.recovery_history
                    if e.recovery_complete_time is not None
                )
                observation_period_hours = (latest_time - earliest_time) / 3600
            else:
                # Fall back to wall clock
                observation_period_hours = (time.time() - self.observation_start) / 3600

        # Get failure times for component(s)
        if component_name:
            failure_timestamps = list(self.failure_times.get(component_name, []))
        else:
            failure_timestamps = []
            for times in self.failure_times.values():
                failure_timestamps.extend(times)

        if len(failure_timestamps) < 2:
            return MTTFAnalysis(
                component_name=component_name or "all",
                observation_period_hours=observation_period_hours,
                total_failures=len(failure_timestamps)
            )

        # Sort timestamps
        failure_timestamps = sorted(failure_timestamps)

        # Calculate time between failures
        times_between_failures = [
            failure_timestamps[i+1] - failure_timestamps[i]
            for i in range(len(failure_timestamps) - 1)
        ]

        analysis = MTTFAnalysis(
            component_name=component_name or "all",
            mean_time_to_failure=statistics.mean(times_between_failures),
            median_time_to_failure=statistics.median(times_between_failures),
            min_time_to_failure=min(times_between_failures),
            max_time_to_failure=max(times_between_failures),
            total_failures=len(failure_timestamps),
            observation_period_hours=observation_period_hours
        )

        # Calculate stddev and percentiles if enough data
        if len(times_between_failures) >= 2:
            analysis.stddev_time_to_failure = statistics.stdev(times_between_failures)

        if len(times_between_failures) >= 10:
            sorted_times = sorted(times_between_failures)
            analysis.p50_time_to_failure = sorted_times[int(len(sorted_times) * 0.50)]
            analysis.p90_time_to_failure = sorted_times[int(len(sorted_times) * 0.90)]
            analysis.p95_time_to_failure = sorted_times[int(len(sorted_times) * 0.95)]

        # Calculate failure rate (failures per hour)
        if observation_period_hours > 0:
            analysis.failure_rate_per_hour = len(failure_timestamps) / observation_period_hours

        # Calculate availability
        # Availability = (Total Time - Downtime) / Total Time
        total_time = observation_period_hours * 3600  # seconds
        downtime = sum(
            e.time_to_recover for e in self.recovery_history
            if (component_name is None or e.component_name == component_name)
            and e.time_to_recover is not None
        )

        if total_time > 0:
            analysis.availability_percent = ((total_time - downtime) / total_time) * 100

        return analysis

    def generate_report(self) -> RecoveryMetricsReport:
        """
        Generate comprehensive recovery metrics report.

        Returns:
            RecoveryMetricsReport with all components and system-level metrics
        """
        report = RecoveryMetricsReport(
            observation_start_time=self.observation_start,
            observation_end_time=time.time()
        )

        # Get all components
        all_components = set()
        all_components.update(self.failure_times.keys())
        all_components.update(e.component_name for e in self.recovery_history)

        # Calculate per-component metrics
        for component in all_components:
            mttr = self.calculate_mttr(component)
            mttf = self.calculate_mttf(component)
            report.component_metrics[component] = (mttr, mttf)

        # Calculate system-level metrics
        all_mttr = self.calculate_mttr()
        all_mttf = self.calculate_mttf()

        report.system_mttr = all_mttr.mean_recovery_time
        report.system_mttf = all_mttf.mean_time_to_failure
        report.system_availability = all_mttf.availability_percent
        report.total_recovery_events = len(self.recovery_history)
        report.overall_success_rate = all_mttr.recovery_success_rate

        return report

    def reset(self):
        """Reset all tracking data."""
        self.active_recoveries.clear()
        self.recovery_history.clear()
        self.failure_times.clear()
        self.last_healthy_time.clear()
        self.observation_start = time.time()
        logger.info("Recovery time tracker reset")
