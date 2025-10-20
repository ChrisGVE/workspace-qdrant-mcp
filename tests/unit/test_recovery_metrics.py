"""
Unit Tests for Recovery Time Measurement System (Task 305.7)

Comprehensive tests for recovery metrics tracking including MTTR/MTTF calculation,
automated recovery detection, and statistical analysis.
"""

import pytest
import time
from tests.framework.recovery_metrics import (
    RecoveryTimeTracker,
    RecoveryEvent,
    RecoveryState,
    RecoveryDetectionStrategy,
    MTTRAnalysis,
    MTTFAnalysis,
    RecoveryMetricsReport,
)


class TestRecoveryEvent:
    """Test RecoveryEvent dataclass."""

    def test_event_creation(self):
        """Test creating a recovery event."""
        event = RecoveryEvent(
            component_name="test-component",
            failure_time=100.0,
            detection_time=101.0,
            recovery_start_time=102.0
        )

        assert event.component_name == "test-component"
        assert event.failure_time == 100.0
        assert event.detection_time == 101.0
        assert event.recovery_start_time == 102.0
        assert event.recovery_complete_time is None
        assert event.recovery_successful is False
        assert event.recovery_state == RecoveryState.FAILED

    def test_complete_recovery_successful(self):
        """Test completing a successful recovery."""
        event = RecoveryEvent(
            component_name="test-component",
            failure_time=100.0,
            detection_time=101.0,
            recovery_start_time=102.0,
            baseline_performance={"latency_ms": 100.0}
        )

        event.complete_recovery(
            recovery_time=110.0,
            successful=True,
            performance={"latency_ms": 105.0}
        )

        assert event.recovery_complete_time == 110.0
        assert event.recovery_successful is True
        assert event.recovery_state == RecoveryState.RECOVERED
        assert event.recovery_duration == 8.0  # 110 - 102
        assert event.time_to_detect == 1.0     # 101 - 100
        assert event.time_to_recover == 10.0   # 110 - 100

    def test_complete_recovery_failed(self):
        """Test completing a failed recovery."""
        event = RecoveryEvent(
            component_name="test-component",
            failure_time=100.0,
            detection_time=101.0,
            recovery_start_time=102.0
        )

        event.complete_recovery(
            recovery_time=110.0,
            successful=False
        )

        assert event.recovery_successful is False
        assert event.recovery_state == RecoveryState.FAILED

    def test_performance_degradation_calculation(self):
        """Test performance degradation calculation."""
        event = RecoveryEvent(
            component_name="test-component",
            failure_time=100.0,
            detection_time=101.0,
            recovery_start_time=102.0,
            baseline_performance={
                "latency_ms": 100.0,
                "throughput_rps": 1000.0
            }
        )

        # Latency increased by 20%, throughput decreased by 10%
        event.complete_recovery(
            recovery_time=110.0,
            successful=True,
            performance={
                "latency_ms": 120.0,    # 20% worse (higher is worse)
                "throughput_rps": 900.0  # 10% worse (lower is worse)
            }
        )

        # Average degradation should be (20 + 10) / 2 = 15%
        assert event.performance_degradation_percent == pytest.approx(15.0, abs=0.1)


class TestRecoveryTimeTracker:
    """Test RecoveryTimeTracker functionality."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = RecoveryTimeTracker()

        assert len(tracker.active_recoveries) == 0
        assert len(tracker.recovery_history) == 0
        assert len(tracker.failure_times) == 0
        assert tracker.observation_start > 0

    def test_record_failure(self):
        """Test recording a failure."""
        tracker = RecoveryTimeTracker()
        tracker.record_failure("component1", failure_time=100.0, failure_reason="timeout")

        assert "component1" in tracker.active_recoveries
        assert "component1" in tracker.failure_times
        assert len(tracker.failure_times["component1"]) == 1
        assert tracker.failure_times["component1"][0] == 100.0

        event = tracker.active_recoveries["component1"]
        assert event.component_name == "component1"
        assert event.failure_time == 100.0
        assert event.failure_reason == "timeout"

    def test_record_recovery_start(self):
        """Test recording recovery start."""
        tracker = RecoveryTimeTracker()
        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_start("component1", start_time=105.0)

        event = tracker.active_recoveries["component1"]
        assert event.recovery_start_time == 105.0
        assert event.recovery_state == RecoveryState.RECOVERING

    def test_record_recovery_complete_successful(self):
        """Test recording successful recovery completion."""
        tracker = RecoveryTimeTracker(
            baseline_performance={"latency_ms": 100.0}
        )

        tracker.record_failure("component1", failure_time=100.0)
        event = tracker.record_recovery_complete(
            "component1",
            recovery_time=110.0,
            successful=True,
            current_performance={"latency_ms": 105.0}
        )

        assert event is not None
        assert event.recovery_successful is True
        assert event.recovery_duration == pytest.approx(10.0, abs=0.1)
        assert "component1" not in tracker.active_recoveries
        assert len(tracker.recovery_history) == 1
        assert "component1" in tracker.last_healthy_time

    def test_record_recovery_complete_failed(self):
        """Test recording failed recovery completion."""
        tracker = RecoveryTimeTracker()
        tracker.record_failure("component1", failure_time=100.0)
        event = tracker.record_recovery_complete(
            "component1",
            recovery_time=110.0,
            successful=False
        )

        assert event is not None
        assert event.recovery_successful is False
        assert "component1" not in tracker.active_recoveries
        assert len(tracker.recovery_history) == 1

    def test_multiple_failures_same_component(self):
        """Test tracking multiple failures for same component."""
        tracker = RecoveryTimeTracker()

        # First failure and recovery
        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        # Second failure and recovery
        tracker.record_failure("component1", failure_time=200.0)
        tracker.record_recovery_complete("component1", recovery_time=215.0, successful=True)

        assert len(tracker.recovery_history) == 2
        assert len(tracker.failure_times["component1"]) == 2

    def test_multiple_components(self):
        """Test tracking multiple components."""
        tracker = RecoveryTimeTracker()

        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_failure("component2", failure_time=105.0)

        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)
        tracker.record_recovery_complete("component2", recovery_time=120.0, successful=True)

        assert len(tracker.recovery_history) == 2
        assert "component1" in tracker.failure_times
        assert "component2" in tracker.failure_times


class TestMTTRCalculation:
    """Test MTTR calculation."""

    def test_mttr_no_data(self):
        """Test MTTR with no recovery data."""
        tracker = RecoveryTimeTracker()
        mttr = tracker.calculate_mttr("component1")

        assert mttr.component_name == "component1"
        assert mttr.total_recoveries == 0
        assert mttr.mean_recovery_time == 0.0

    def test_mttr_single_recovery(self):
        """Test MTTR with single recovery."""
        tracker = RecoveryTimeTracker()

        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        mttr = tracker.calculate_mttr("component1")

        assert mttr.total_recoveries == 1
        assert mttr.successful_recoveries == 1
        assert mttr.mean_recovery_time == pytest.approx(10.0, abs=0.1)
        assert mttr.min_recovery_time == pytest.approx(10.0, abs=0.1)
        assert mttr.max_recovery_time == pytest.approx(10.0, abs=0.1)

    def test_mttr_multiple_recoveries(self):
        """Test MTTR with multiple recoveries."""
        tracker = RecoveryTimeTracker()

        # Recovery times: 10s, 15s, 12s
        recoveries = [
            (100.0, 110.0),  # 10s
            (200.0, 215.0),  # 15s
            (300.0, 312.0),  # 12s
        ]

        for failure_time, recovery_time in recoveries:
            tracker.record_failure("component1", failure_time=failure_time)
            tracker.record_recovery_complete("component1", recovery_time=recovery_time, successful=True)

        mttr = tracker.calculate_mttr("component1")

        assert mttr.total_recoveries == 3
        assert mttr.mean_recovery_time == pytest.approx(12.333, abs=0.1)  # (10+15+12)/3
        assert mttr.median_recovery_time == 12.0
        assert mttr.min_recovery_time == 10.0
        assert mttr.max_recovery_time == 15.0

    def test_mttr_with_failures(self):
        """Test MTTR calculation with mixed success/failure."""
        tracker = RecoveryTimeTracker()

        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        tracker.record_failure("component1", failure_time=200.0)
        tracker.record_recovery_complete("component1", recovery_time=215.0, successful=False)

        tracker.record_failure("component1", failure_time=300.0)
        tracker.record_recovery_complete("component1", recovery_time=312.0, successful=True)

        mttr = tracker.calculate_mttr("component1")

        # Only successful recoveries counted in MTTR
        assert mttr.total_recoveries == 2
        assert mttr.successful_recoveries == 2
        assert mttr.failed_recoveries == 0
        assert mttr.recovery_success_rate == pytest.approx(66.67, abs=0.1)  # 2/3

    def test_mttr_percentiles(self):
        """Test MTTR percentile calculations."""
        tracker = RecoveryTimeTracker()

        # Create 20 recoveries with varying times
        recovery_times = list(range(10, 30))  # 10s to 29s
        base_time = 100.0

        for i, duration in enumerate(recovery_times):
            failure_time = base_time + (i * 100)
            recovery_time = failure_time + duration
            tracker.record_failure("component1", failure_time=failure_time)
            tracker.record_recovery_complete("component1", recovery_time=recovery_time, successful=True)

        mttr = tracker.calculate_mttr("component1")

        assert mttr.total_recoveries == 20
        assert mttr.p50_recovery_time == 20  # 50th percentile (10th index of 20 items = 11th value = 20)
        assert mttr.p90_recovery_time == 28
        assert mttr.p95_recovery_time == 29
        assert mttr.p99_recovery_time == 29

    def test_mttr_lookback_limit(self):
        """Test MTTR with lookback count limit."""
        tracker = RecoveryTimeTracker()

        # Create 10 recoveries
        for i in range(10):
            failure_time = 100.0 + (i * 100)
            recovery_time = failure_time + (i + 10)  # Increasing recovery times
            tracker.record_failure("component1", failure_time=failure_time)
            tracker.record_recovery_complete("component1", recovery_time=recovery_time, successful=True)

        # Calculate MTTR for last 5 recoveries
        mttr = tracker.calculate_mttr("component1", lookback_count=5)

        assert mttr.total_recoveries == 5
        # Last 5 recovery times: 15, 16, 17, 18, 19
        assert mttr.mean_recovery_time == pytest.approx(17.0, abs=0.1)

    def test_mttr_trend_analysis(self):
        """Test MTTR trend analysis."""
        tracker = RecoveryTimeTracker()

        # Create improving trend (decreasing recovery times)
        recovery_times = list(range(20, 10, -1))  # 20s down to 11s
        base_time = 100.0

        for i, duration in enumerate(recovery_times):
            failure_time = base_time + (i * 100)
            recovery_time = failure_time + duration
            tracker.record_failure("component1", failure_time=failure_time)
            tracker.record_recovery_complete("component1", recovery_time=recovery_time, successful=True)

        mttr = tracker.calculate_mttr("component1")

        assert mttr.recovery_time_trend == "improving"

    def test_mttr_all_components(self):
        """Test MTTR calculation across all components."""
        tracker = RecoveryTimeTracker()

        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        tracker.record_failure("component2", failure_time=200.0)
        tracker.record_recovery_complete("component2", recovery_time=215.0, successful=True)

        mttr = tracker.calculate_mttr()  # All components

        assert mttr.component_name == "all"
        assert mttr.total_recoveries == 2
        assert mttr.mean_recovery_time == pytest.approx(12.5, abs=0.1)  # (10+15)/2


class TestMTTFCalculation:
    """Test MTTF calculation."""

    def test_mttf_no_data(self):
        """Test MTTF with no failure data."""
        tracker = RecoveryTimeTracker()
        mttf = tracker.calculate_mttf("component1")

        assert mttf.component_name == "component1"
        assert mttf.total_failures == 0

    def test_mttf_single_failure(self):
        """Test MTTF with single failure."""
        tracker = RecoveryTimeTracker()
        tracker.record_failure("component1", failure_time=100.0)

        mttf = tracker.calculate_mttf("component1")

        assert mttf.total_failures == 1
        # Can't calculate MTTF with only one failure

    def test_mttf_multiple_failures(self):
        """Test MTTF with multiple failures."""
        tracker = RecoveryTimeTracker()

        # Failures at: 100, 200, 350, 600
        # Time between: 100s, 150s, 250s
        failure_times = [100.0, 200.0, 350.0, 600.0]

        for ft in failure_times:
            tracker.record_failure("component1", failure_time=ft)

        mttf = tracker.calculate_mttf("component1", observation_period_hours=1.0)

        assert mttf.total_failures == 4
        assert mttf.mean_time_to_failure == pytest.approx(166.67, abs=0.1)  # (100+150+250)/3
        assert mttf.median_time_to_failure == 150.0
        assert mttf.min_time_to_failure == 100.0
        assert mttf.max_time_to_failure == 250.0

    def test_mttf_failure_rate(self):
        """Test failure rate calculation."""
        tracker = RecoveryTimeTracker()

        # 10 failures over 10 hours = 1 failure/hour
        for i in range(10):
            tracker.record_failure("component1", failure_time=i * 3600.0)

        mttf = tracker.calculate_mttf("component1", observation_period_hours=10.0)

        assert mttf.total_failures == 10
        assert mttf.failure_rate_per_hour == pytest.approx(1.0, abs=0.1)

    def test_mttf_availability_calculation(self):
        """Test availability percentage calculation."""
        tracker = RecoveryTimeTracker()

        # Simulate 3 failures with recoveries
        # Total downtime: 30s
        # Observation: 1 hour = 3600s
        # Availability = (3600 - 30) / 3600 = 99.17%

        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        tracker.record_failure("component1", failure_time=200.0)
        tracker.record_recovery_complete("component1", recovery_time=210.0, successful=True)

        tracker.record_failure("component1", failure_time=300.0)
        tracker.record_recovery_complete("component1", recovery_time=310.0, successful=True)

        mttf = tracker.calculate_mttf("component1", observation_period_hours=1.0)

        # Total downtime = 10 + 10 + 10 = 30s out of 3600s
        expected_availability = ((3600 - 30) / 3600) * 100
        assert mttf.availability_percent == pytest.approx(expected_availability, abs=0.1)

    def test_mttf_all_components(self):
        """Test MTTF calculation across all components."""
        tracker = RecoveryTimeTracker()

        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_failure("component1", failure_time=200.0)

        tracker.record_failure("component2", failure_time=150.0)
        tracker.record_failure("component2", failure_time=300.0)

        mttf = tracker.calculate_mttf(observation_period_hours=1.0)

        assert mttf.component_name == "all"
        assert mttf.total_failures == 4
        # Times between: 100, 150 (component1 and component2 combined)


class TestRecoveryMetricsReport:
    """Test comprehensive recovery metrics reporting."""

    def test_generate_report_empty(self):
        """Test generating report with no data."""
        tracker = RecoveryTimeTracker()
        report = tracker.generate_report()

        assert isinstance(report, RecoveryMetricsReport)
        assert len(report.component_metrics) == 0
        assert report.total_recovery_events == 0

    def test_generate_report_single_component(self):
        """Test generating report for single component."""
        tracker = RecoveryTimeTracker()

        # Create some failures and recoveries
        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        tracker.record_failure("component1", failure_time=200.0)
        tracker.record_recovery_complete("component1", recovery_time=215.0, successful=True)

        report = tracker.generate_report()

        assert len(report.component_metrics) == 1
        assert "component1" in report.component_metrics

        mttr, mttf = report.component_metrics["component1"]
        assert mttr.total_recoveries == 2
        assert mttf.total_failures == 2

    def test_generate_report_multiple_components(self):
        """Test generating report for multiple components."""
        tracker = RecoveryTimeTracker()

        # Component 1
        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        # Component 2
        tracker.record_failure("component2", failure_time=200.0)
        tracker.record_recovery_complete("component2", recovery_time=220.0, successful=True)

        report = tracker.generate_report()

        assert len(report.component_metrics) == 2
        assert "component1" in report.component_metrics
        assert "component2" in report.component_metrics

    def test_generate_report_system_metrics(self):
        """Test system-level metrics in report."""
        tracker = RecoveryTimeTracker()

        # Create multiple recoveries
        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        tracker.record_failure("component1", failure_time=200.0)
        tracker.record_recovery_complete("component1", recovery_time=215.0, successful=True)

        tracker.record_failure("component2", failure_time=300.0)
        tracker.record_recovery_complete("component2", recovery_time=320.0, successful=True)

        report = tracker.generate_report()

        assert report.system_mttr > 0
        assert report.system_availability > 0
        assert report.total_recovery_events == 3
        assert report.overall_success_rate == 100.0


class TestRecoveryTimeTracker_Reset:
    """Test tracker reset functionality."""

    def test_reset(self):
        """Test resetting tracker state."""
        tracker = RecoveryTimeTracker()

        # Add some data
        tracker.record_failure("component1", failure_time=100.0)
        tracker.record_recovery_complete("component1", recovery_time=110.0, successful=True)

        # Reset
        tracker.reset()

        assert len(tracker.active_recoveries) == 0
        assert len(tracker.recovery_history) == 0
        assert len(tracker.failure_times) == 0
        assert len(tracker.last_healthy_time) == 0
