"""
Unit Tests for Performance Degradation Tracking System (Task 305.8)

Comprehensive tests for performance degradation monitoring including baseline
establishment, degradation detection, trend analysis, and alerting.
"""

import pytest
import time
from tests.framework.performance_degradation import (
    PerformanceDegradationTracker,
    PerformanceMetricType,
    PerformanceTrend,
    AlertSeverity,
    PerformanceBaseline,
    PerformanceSnapshot,
    PerformanceDegradation,
    PerformanceTrendAnalysis,
)


class TestPerformanceDegradationTracker:
    """Test PerformanceDegradationTracker initialization and basic functionality."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = PerformanceDegradationTracker()

        assert len(tracker.baselines) == 0
        assert len(tracker.history) == 0
        assert len(tracker.degradations) == 0
        assert tracker.default_threshold == 20.0

    def test_initialization_with_custom_threshold(self):
        """Test tracker initialization with custom threshold."""
        tracker = PerformanceDegradationTracker(default_degradation_threshold_percent=30.0)

        assert tracker.default_threshold == 30.0

    def test_register_metric(self):
        """Test registering a metric."""
        tracker = PerformanceDegradationTracker()

        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        assert "latency_ms" in tracker.metric_types
        assert tracker.metric_types["latency_ms"] == PerformanceMetricType.LATENCY
        assert "latency_ms" in tracker.history

    def test_register_metric_with_custom_threshold(self):
        """Test registering a metric with custom threshold."""
        tracker = PerformanceDegradationTracker()

        tracker.register_metric(
            "latency_ms",
            PerformanceMetricType.LATENCY,
            threshold_percent=15.0
        )

        assert "latency_ms" in tracker.thresholds
        assert tracker.thresholds["latency_ms"] == 15.0


class TestBaselineEstablishment:
    """Test baseline establishment."""

    def test_establish_baseline_simple(self):
        """Test establishing a simple baseline."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        samples = [100.0, 110.0, 105.0, 95.0, 100.0]
        baseline = tracker.establish_baseline("latency_ms", samples)

        assert baseline.metric_name == "latency_ms"
        assert baseline.mean == pytest.approx(102.0, abs=0.1)
        assert baseline.median == 100.0
        assert baseline.min_value == 95.0
        assert baseline.max_value == 110.0
        assert baseline.sample_count == 5

    def test_establish_baseline_with_percentiles(self):
        """Test establishing baseline with percentiles."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Create enough samples for percentiles (>=10)
        samples = list(range(10, 30))  # 10-29
        baseline = tracker.establish_baseline("latency_ms", samples)

        assert baseline.sample_count == 20
        assert baseline.p50 == 20  # Median
        assert baseline.p90 == 28
        assert baseline.p95 == 29
        assert baseline.p99 == 29

    def test_establish_baseline_empty_samples(self):
        """Test establishing baseline with empty samples raises error."""
        tracker = PerformanceDegradationTracker()

        with pytest.raises(ValueError, match="no samples"):
            tracker.establish_baseline("latency_ms", [])

    def test_establish_baseline_updates_registry(self):
        """Test that establishing baseline updates the baseline registry."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        samples = [100.0, 110.0, 105.0]
        tracker.establish_baseline("latency_ms", samples)

        assert "latency_ms" in tracker.baselines
        assert tracker.baselines["latency_ms"].mean == pytest.approx(105.0, abs=0.1)


class TestMetricRecording:
    """Test metric recording."""

    def test_record_metric(self):
        """Test recording a metric value."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        tracker.record_metric("latency_ms", 100.0, timestamp=1000.0, check_degradation=False)

        assert len(tracker.history["latency_ms"]) == 1
        assert tracker.history["latency_ms"][0] == (1000.0, 100.0)

    def test_record_multiple_metrics(self):
        """Test recording multiple metric values."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        for i in range(5):
            tracker.record_metric("latency_ms", 100.0 + i, timestamp=1000.0 + i, check_degradation=False)

        assert len(tracker.history["latency_ms"]) == 5

    def test_record_unregistered_metric(self):
        """Test recording an unregistered metric auto-registers it."""
        tracker = PerformanceDegradationTracker()

        tracker.record_metric("unknown_metric", 100.0, timestamp=1000.0, check_degradation=False)

        assert "unknown_metric" in tracker.metric_types
        assert "unknown_metric" in tracker.history

    def test_record_snapshot(self):
        """Test recording a snapshot of multiple metrics."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)
        tracker.register_metric("throughput_rps", PerformanceMetricType.THROUGHPUT)

        metrics = {
            "latency_ms": 100.0,
            "throughput_rps": 1000.0
        }

        snapshot = tracker.record_snapshot(metrics)

        assert len(tracker.snapshots) == 1
        assert snapshot.metrics == metrics
        assert len(tracker.history["latency_ms"]) == 1
        assert len(tracker.history["throughput_rps"]) == 1


class TestDegradationDetection:
    """Test degradation detection."""

    def test_detect_latency_degradation(self):
        """Test detecting latency degradation."""
        alerts = []
        tracker = PerformanceDegradationTracker(
            alert_callback=lambda d: alerts.append(d),
            default_degradation_threshold_percent=20.0
        )
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Establish baseline: mean = 100ms
        baseline_samples = [100.0] * 10
        tracker.establish_baseline("latency_ms", baseline_samples)

        # Record value with 25% degradation (should trigger alert)
        tracker.record_metric("latency_ms", 125.0, timestamp=2000.0)

        assert len(alerts) == 1
        assert alerts[0].metric_name == "latency_ms"
        assert alerts[0].degradation_percent == pytest.approx(25.0, abs=0.1)
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_detect_critical_degradation(self):
        """Test detecting critical degradation."""
        alerts = []
        tracker = PerformanceDegradationTracker(
            alert_callback=lambda d: alerts.append(d),
            default_degradation_threshold_percent=20.0
        )
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Establish baseline: mean = 100ms
        tracker.establish_baseline("latency_ms", [100.0] * 10)

        # Record value with 50% degradation (2x threshold = critical)
        tracker.record_metric("latency_ms", 150.0, timestamp=2000.0)

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    def test_no_degradation_within_threshold(self):
        """Test no degradation detected within threshold."""
        alerts = []
        tracker = PerformanceDegradationTracker(
            alert_callback=lambda d: alerts.append(d),
            default_degradation_threshold_percent=20.0
        )
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Establish baseline: mean = 100ms
        tracker.establish_baseline("latency_ms", [100.0] * 10)

        # Record value with 15% degradation (below 20% threshold)
        tracker.record_metric("latency_ms", 115.0, timestamp=2000.0)

        assert len(alerts) == 0

    def test_throughput_degradation(self):
        """Test detecting throughput degradation."""
        alerts = []
        tracker = PerformanceDegradationTracker(
            alert_callback=lambda d: alerts.append(d),
            default_degradation_threshold_percent=20.0
        )
        tracker.register_metric("throughput_rps", PerformanceMetricType.THROUGHPUT)

        # Establish baseline: mean = 1000 rps
        tracker.establish_baseline("throughput_rps", [1000.0] * 10)

        # Record value with 25% degradation (lower is worse for throughput)
        tracker.record_metric("throughput_rps", 750.0, timestamp=2000.0)

        assert len(alerts) == 1
        assert alerts[0].metric_name == "throughput_rps"
        assert alerts[0].degradation_percent == pytest.approx(25.0, abs=0.1)

    def test_improvement_not_detected_as_degradation(self):
        """Test that performance improvements are not flagged as degradation."""
        alerts = []
        tracker = PerformanceDegradationTracker(
            alert_callback=lambda d: alerts.append(d)
        )
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Establish baseline: mean = 100ms
        tracker.establish_baseline("latency_ms", [100.0] * 10)

        # Record improved value (lower latency)
        tracker.record_metric("latency_ms", 80.0, timestamp=2000.0)

        assert len(alerts) == 0

    def test_custom_threshold(self):
        """Test degradation detection with custom threshold."""
        alerts = []
        tracker = PerformanceDegradationTracker(
            alert_callback=lambda d: alerts.append(d)
        )
        tracker.register_metric(
            "latency_ms",
            PerformanceMetricType.LATENCY,
            threshold_percent=10.0  # Custom 10% threshold
        )

        # Establish baseline: mean = 100ms
        tracker.establish_baseline("latency_ms", [100.0] * 10)

        # Record value with 12% degradation (above 10% custom threshold)
        tracker.record_metric("latency_ms", 112.0, timestamp=2000.0)

        assert len(alerts) == 1


class TestTrendAnalysis:
    """Test performance trend analysis."""

    def test_trend_improving(self):
        """Test detecting improving trend."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Record decreasing latency (improving)
        for i in range(20):
            latency = 100.0 - (i * 2)  # 100, 98, 96, ...
            tracker.record_metric("latency_ms", latency, timestamp=1000.0 + i, check_degradation=False)

        analysis = tracker.analyze_trend("latency_ms")

        assert analysis.trend == PerformanceTrend.IMPROVING
        assert analysis.sample_count == 20
        assert analysis.trend_change_percent < -5  # More than 5% improvement

    def test_trend_degrading(self):
        """Test detecting degrading trend."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Record increasing latency (degrading)
        for i in range(20):
            latency = 100.0 + (i * 2)  # 100, 102, 104, ...
            tracker.record_metric("latency_ms", latency, timestamp=1000.0 + i, check_degradation=False)

        analysis = tracker.analyze_trend("latency_ms")

        assert analysis.trend == PerformanceTrend.DEGRADING
        assert analysis.trend_change_percent > 5  # More than 5% degradation

    def test_trend_stable(self):
        """Test detecting stable trend."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Record stable latency (small variations)
        for i in range(20):
            latency = 100.0 + (i % 3 - 1)  # 99, 100, 101, 99, 100, 101, ...
            tracker.record_metric("latency_ms", latency, timestamp=1000.0 + i, check_degradation=False)

        analysis = tracker.analyze_trend("latency_ms")

        assert analysis.trend == PerformanceTrend.STABLE
        assert abs(analysis.trend_change_percent) < 5

    def test_trend_unknown_insufficient_data(self):
        """Test trend analysis with insufficient data."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Record only a few samples (less than min_samples_for_trend)
        for i in range(5):
            tracker.record_metric("latency_ms", 100.0, timestamp=1000.0 + i, check_degradation=False)

        analysis = tracker.analyze_trend("latency_ms")

        assert analysis.trend == PerformanceTrend.UNKNOWN

    def test_trend_throughput_improving(self):
        """Test detecting improving trend for throughput."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("throughput_rps", PerformanceMetricType.THROUGHPUT)

        # Record increasing throughput (improving)
        for i in range(20):
            throughput = 1000.0 + (i * 10)  # 1000, 1010, 1020, ...
            tracker.record_metric("throughput_rps", throughput, timestamp=1000.0 + i, check_degradation=False)

        analysis = tracker.analyze_trend("throughput_rps")

        assert analysis.trend == PerformanceTrend.IMPROVING

    def test_trend_confidence(self):
        """Test trend confidence calculation."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)

        # Record clear linear trend
        for i in range(50):
            latency = 100.0 + i  # Perfect linear increase
            tracker.record_metric("latency_ms", latency, timestamp=1000.0 + i, check_degradation=False)

        analysis = tracker.analyze_trend("latency_ms")

        # Should have high confidence due to clear linear trend and 50 samples
        assert analysis.trend_confidence > 0.8
        assert analysis.r_squared > 0.95


class TestDegradationRetrieval:
    """Test retrieving degradations."""

    def test_get_all_degradations(self):
        """Test getting all degradations."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)
        tracker.establish_baseline("latency_ms", [100.0] * 10)

        # Trigger 3 degradations
        for i in range(3):
            tracker.record_metric("latency_ms", 125.0, timestamp=2000.0 + i)

        degradations = tracker.get_degradations()

        assert len(degradations) == 3

    def test_get_degradations_by_severity(self):
        """Test getting degradations filtered by severity."""
        tracker = PerformanceDegradationTracker(
            default_degradation_threshold_percent=20.0
        )
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)
        tracker.establish_baseline("latency_ms", [100.0] * 10)

        # WARNING: 25% degradation
        tracker.record_metric("latency_ms", 125.0, timestamp=2000.0)

        # CRITICAL: 50% degradation (2x threshold)
        tracker.record_metric("latency_ms", 150.0, timestamp=2001.0)

        warnings = tracker.get_degradations(severity=AlertSeverity.WARNING)
        criticals = tracker.get_degradations(severity=AlertSeverity.CRITICAL)

        assert len(warnings) == 1
        assert len(criticals) == 1

    def test_get_degradations_since_timestamp(self):
        """Test getting degradations after a timestamp."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)
        tracker.establish_baseline("latency_ms", [100.0] * 10)

        # Create degradations at different times
        tracker.record_metric("latency_ms", 125.0, timestamp=1000.0)
        tracker.record_metric("latency_ms", 125.0, timestamp=2000.0)
        tracker.record_metric("latency_ms", 125.0, timestamp=3000.0)

        # Get only degradations after timestamp 2000
        recent = tracker.get_degradations(since_timestamp=2000.0)

        assert len(recent) == 2  # Only 2000 and 3000


class TestReset:
    """Test tracker reset."""

    def test_reset(self):
        """Test resetting tracker state."""
        tracker = PerformanceDegradationTracker()
        tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)
        tracker.establish_baseline("latency_ms", [100.0] * 10)
        tracker.record_metric("latency_ms", 125.0, timestamp=2000.0)

        # Reset
        tracker.reset()

        assert len(tracker.baselines) == 0
        assert len(tracker.history) == 0
        assert len(tracker.degradations) == 0
        assert len(tracker.snapshots) == 0
