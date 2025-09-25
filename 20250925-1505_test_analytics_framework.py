"""
Comprehensive tests for the test analytics framework.

Tests all edge cases, error conditions, and functionality of the analytics system.
"""

import math
import sqlite3
import tempfile
import time
from collections import deque
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the analytics framework
import sys
src_path = Path(__file__).parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

framework_path = Path(__file__).parent / "tests" / "framework"
if str(framework_path) not in sys.path:
    sys.path.insert(0, str(framework_path))

from tests.framework.analytics import (
    TestAnalytics, TestMetrics, SuiteMetrics, FlakeDetector, TrendAnalyzer,
    AlertManager, Alert, HealthStatus, TrendDirection, AlertLevel
)
from tests.framework.discovery import TestMetadata, TestCategory, TestComplexity
from tests.framework.execution import ExecutionResult, ExecutionStatus


class TestFlakeDetector:
    """Test suite for FlakeDetector."""

    @pytest.fixture
    def flake_detector(self):
        """Create FlakeDetector instance."""
        return FlakeDetector(min_runs=5, significance_level=0.05)

    def test_initialization(self, flake_detector):
        """Test FlakeDetector initialization."""
        assert flake_detector.min_runs == 5
        assert flake_detector.significance_level == 0.05

    def test_stable_tests_not_flaky(self, flake_detector):
        """Test that consistently passing or failing tests are not considered flaky."""
        # All passing
        all_pass = [True] * 20
        flakiness = flake_detector.calculate_flakiness_score(all_pass)
        assert flakiness == 0.0

        # All failing
        all_fail = [False] * 20
        flakiness = flake_detector.calculate_flakiness_score(all_fail)
        assert flakiness == 0.0

    def test_alternating_pattern_is_flaky(self, flake_detector):
        """Test that alternating pass/fail pattern is detected as flaky."""
        alternating = [True, False] * 10  # Perfect alternating pattern
        flakiness = flake_detector.calculate_flakiness_score(alternating)
        assert flakiness > 0.5  # Should be considered flaky

    def test_mostly_stable_with_occasional_failures(self, flake_detector):
        """Test mostly stable tests with occasional failures."""
        mostly_pass = [True] * 18 + [False] * 2
        flakiness = flake_detector.calculate_flakiness_score(mostly_pass)
        assert 0.0 <= flakiness <= 0.3  # Should be low flakiness

    def test_insufficient_data(self, flake_detector):
        """Test handling of insufficient data."""
        insufficient_data = [True, False, True]  # Less than min_runs
        flakiness = flake_detector.calculate_flakiness_score(insufficient_data)
        assert flakiness == 0.0

    def test_empty_data(self, flake_detector):
        """Test handling of empty data."""
        flakiness = flake_detector.calculate_flakiness_score([])
        assert flakiness == 0.0

    def test_run_calculation(self, flake_detector):
        """Test run calculation logic."""
        # Single run (all same value)
        single_run = [1, 1, 1, 1, 1]
        runs = flake_detector._calculate_runs(single_run)
        assert runs == 1

        # Two runs
        two_runs = [1, 1, 0, 0, 0]
        runs = flake_detector._calculate_runs(two_runs)
        assert runs == 2

        # Many runs (alternating)
        many_runs = [1, 0, 1, 0, 1, 0]
        runs = flake_detector._calculate_runs(many_runs)
        assert runs == 6

        # Empty list
        empty_runs = flake_detector._calculate_runs([])
        assert empty_runs == 0

    def test_detect_flaky_tests_integration(self, flake_detector):
        """Test integration with TestMetrics."""
        metrics = {
            "stable_test": TestMetrics(
                test_name="stable_test",
                total_runs=20,
                result_history=deque([ExecutionStatus.COMPLETED] * 20, maxlen=100)
            ),
            "flaky_test": TestMetrics(
                test_name="flaky_test",
                total_runs=20,
                passed_runs=10,
                failed_runs=10,
                result_history=deque([ExecutionStatus.COMPLETED, ExecutionStatus.FAILED] * 10, maxlen=100)
            ),
            "insufficient_test": TestMetrics(
                test_name="insufficient_test",
                total_runs=3,
                result_history=deque([ExecutionStatus.COMPLETED] * 3, maxlen=100)
            )
        }

        # Update success rates
        for test_metrics in metrics.values():
            if test_metrics.total_runs > 0:
                test_metrics.success_rate = test_metrics.passed_runs / test_metrics.total_runs

        flaky_tests = flake_detector.detect_flaky_tests(metrics)

        # Should detect the flaky test but not the others
        assert "flaky_test" in flaky_tests
        assert "stable_test" not in flaky_tests
        assert "insufficient_test" not in flaky_tests


class TestTrendAnalyzer:
    """Test suite for TrendAnalyzer."""

    @pytest.fixture
    def trend_analyzer(self):
        """Create TrendAnalyzer instance."""
        return TrendAnalyzer(window_size=10)

    def test_initialization(self, trend_analyzer):
        """Test TrendAnalyzer initialization."""
        assert trend_analyzer.window_size == 10

    def test_performance_trend_improving(self, trend_analyzer):
        """Test detection of improving performance trend."""
        # Decreasing durations (improving performance)
        improving_data = [10.0, 9.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5]
        trend = trend_analyzer.analyze_performance_trend(improving_data)
        assert trend == TrendDirection.IMPROVING

    def test_performance_trend_degrading(self, trend_analyzer):
        """Test detection of degrading performance trend."""
        # Increasing durations (degrading performance)
        degrading_data = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5]
        trend = trend_analyzer.analyze_performance_trend(degrading_data)
        assert trend == TrendDirection.DEGRADING

    def test_performance_trend_stable(self, trend_analyzer):
        """Test detection of stable performance trend."""
        # Stable durations with minor fluctuations
        stable_data = [5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0]
        trend = trend_analyzer.analyze_performance_trend(stable_data)
        assert trend == TrendDirection.STABLE

    def test_performance_trend_insufficient_data(self, trend_analyzer):
        """Test handling of insufficient performance data."""
        insufficient_data = [5.0, 5.1, 4.9]
        trend = trend_analyzer.analyze_performance_trend(insufficient_data)
        assert trend == TrendDirection.STABLE

    def test_reliability_trend_improving(self, trend_analyzer):
        """Test detection of improving reliability trend."""
        # More successes at the end
        improving_data = [False, False, False, True, True, True, True, True, True, True]
        trend = trend_analyzer.analyze_reliability_trend(improving_data)
        assert trend == TrendDirection.IMPROVING

    def test_reliability_trend_degrading(self, trend_analyzer):
        """Test detection of degrading reliability trend."""
        # More failures at the end
        degrading_data = [True, True, True, True, True, True, False, False, False, False]
        trend = trend_analyzer.analyze_reliability_trend(degrading_data)
        assert trend == TrendDirection.DEGRADING

    def test_reliability_trend_volatile(self, trend_analyzer):
        """Test detection of volatile reliability trend."""
        # High switching rate
        volatile_data = [True, False, True, False, True, False, True, False, True, False]
        trend = trend_analyzer.analyze_reliability_trend(volatile_data)
        assert trend == TrendDirection.VOLATILE

    def test_reliability_trend_stable(self, trend_analyzer):
        """Test detection of stable reliability trend."""
        stable_data = [True] * 10
        trend = trend_analyzer.analyze_reliability_trend(stable_data)
        assert trend == TrendDirection.STABLE

    def test_volatility_calculation(self, trend_analyzer):
        """Test volatility calculation."""
        # High volatility (alternating)
        high_volatility = [True, False, True, False, True, False]
        volatility = trend_analyzer._calculate_volatility(high_volatility)
        assert volatility > 0.8

        # Low volatility (stable)
        low_volatility = [True, True, True, False, False, False]
        volatility = trend_analyzer._calculate_volatility(low_volatility)
        assert volatility < 0.5

        # No volatility (all same)
        no_volatility = [True, True, True, True]
        volatility = trend_analyzer._calculate_volatility(no_volatility)
        assert volatility == 0.0

        # Edge cases
        single_item = [True]
        volatility = trend_analyzer._calculate_volatility(single_item)
        assert volatility == 0.0

        empty_list = []
        volatility = trend_analyzer._calculate_volatility(empty_list)
        assert volatility == 0.0


class TestAlertManager:
    """Test suite for AlertManager."""

    @pytest.fixture
    def alert_manager(self):
        """Create AlertManager instance."""
        return AlertManager()

    def test_initialization(self, alert_manager):
        """Test AlertManager initialization."""
        assert len(alert_manager.alerts) == 0
        assert alert_manager._alert_thresholds['success_rate_warning'] == 0.90
        assert alert_manager._alert_thresholds['success_rate_critical'] == 0.70

    def test_suite_health_critical_success_rate(self, alert_manager):
        """Test critical success rate alert generation."""
        suite_metrics = SuiteMetrics(
            total_tests=100,
            executed_tests=100,
            passed_tests=60,
            failed_tests=40,
            overall_success_rate=0.60  # Below critical threshold
        )

        alerts = alert_manager.check_suite_health(suite_metrics)

        assert len(alerts) > 0
        critical_alerts = [a for a in alerts if a.level == AlertLevel.CRITICAL]
        assert len(critical_alerts) > 0
        assert "Critical" in critical_alerts[0].message

    def test_suite_health_warning_success_rate(self, alert_manager):
        """Test warning success rate alert generation."""
        suite_metrics = SuiteMetrics(
            total_tests=100,
            executed_tests=100,
            passed_tests=85,
            failed_tests=15,
            overall_success_rate=0.85  # Below warning threshold
        )

        alerts = alert_manager.check_suite_health(suite_metrics)

        warning_alerts = [a for a in alerts if a.level == AlertLevel.WARNING]
        assert len(warning_alerts) > 0

    def test_flaky_test_alerts(self, alert_manager):
        """Test flaky test alert generation."""
        suite_metrics = SuiteMetrics(
            total_tests=100,
            executed_tests=100,
            passed_tests=90,
            failed_tests=10,
            overall_success_rate=0.90,
            flaky_test_count=15  # 15% flaky tests
        )

        alerts = alert_manager.check_suite_health(suite_metrics)

        flaky_alerts = [a for a in alerts if "flaky" in a.message.lower()]
        assert len(flaky_alerts) > 0

    def test_performance_degradation_alerts(self, alert_manager):
        """Test performance degradation alert generation."""
        suite_metrics = SuiteMetrics(
            total_tests=100,
            executed_tests=100,
            passed_tests=95,
            failed_tests=5,
            overall_success_rate=0.95,
            performance_trend=TrendDirection.DEGRADING
        )

        alerts = alert_manager.check_suite_health(suite_metrics)

        perf_alerts = [a for a in alerts if "performance" in a.message.lower()]
        assert len(perf_alerts) > 0

    def test_test_health_flaky_alerts(self, alert_manager):
        """Test individual test flaky alerts."""
        test_metrics = TestMetrics(
            test_name="flaky_test",
            total_runs=50,
            passed_runs=30,
            failed_runs=20,
            success_rate=0.6,
            flakiness_score=0.7  # High flakiness
        )

        alerts = alert_manager.check_test_health(test_metrics)

        flaky_alerts = [a for a in alerts if a.test_name == "flaky_test"]
        assert len(flaky_alerts) > 0

    def test_test_health_high_failure_rate(self, alert_manager):
        """Test high failure rate alerts."""
        test_metrics = TestMetrics(
            test_name="failing_test",
            total_runs=20,
            passed_runs=8,
            failed_runs=12,
            success_rate=0.4  # High failure rate
        )

        alerts = alert_manager.check_test_health(test_metrics)

        failure_alerts = [a for a in alerts if "failure rate" in a.message.lower()]
        assert len(failure_alerts) > 0

    def test_no_alerts_for_healthy_suite(self, alert_manager):
        """Test that healthy suites don't generate alerts."""
        healthy_suite = SuiteMetrics(
            total_tests=100,
            executed_tests=100,
            passed_tests=98,
            failed_tests=2,
            overall_success_rate=0.98,
            flaky_test_count=1,
            performance_trend=TrendDirection.STABLE,
            reliability_trend=TrendDirection.STABLE
        )

        alerts = alert_manager.check_suite_health(healthy_suite)

        # Should have minimal or no alerts
        assert len(alerts) == 0


class TestTestAnalytics:
    """Test suite for TestAnalytics."""

    @pytest.fixture
    def temp_analytics(self):
        """Create TestAnalytics with temporary database."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        analytics = TestAnalytics(database_path=db_path)
        yield analytics
        analytics.close()

        # Clean up
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def sample_results(self):
        """Create sample test execution results."""
        return {
            "test_a": ExecutionResult(
                test_name="test_a",
                status=ExecutionStatus.COMPLETED,
                duration=1.5,
                start_time=time.time(),
                end_time=time.time() + 1.5
            ),
            "test_b": ExecutionResult(
                test_name="test_b",
                status=ExecutionStatus.FAILED,
                duration=2.0,
                start_time=time.time(),
                end_time=time.time() + 2.0,
                error_message="AssertionError: Test failed"
            ),
            "test_c": ExecutionResult(
                test_name="test_c",
                status=ExecutionStatus.TIMEOUT,
                duration=30.0,
                start_time=time.time(),
                end_time=time.time() + 30.0,
                error_message="Test timed out after 30 seconds"
            )
        }

    @pytest.fixture
    def sample_metadata(self):
        """Create sample test metadata."""
        return {
            "test_a": TestMetadata(
                name="test_a",
                file_path=Path("/fake/test_a.py"),
                category=TestCategory.UNIT,
                complexity=TestComplexity.LOW,
                estimated_duration=1.0
            ),
            "test_b": TestMetadata(
                name="test_b",
                file_path=Path("/fake/test_b.py"),
                category=TestCategory.INTEGRATION,
                complexity=TestComplexity.MEDIUM,
                estimated_duration=2.0
            ),
            "test_c": TestMetadata(
                name="test_c",
                file_path=Path("/fake/test_c.py"),
                category=TestCategory.E2E,
                complexity=TestComplexity.HIGH,
                estimated_duration=10.0
            )
        }

    def test_initialization(self, temp_analytics):
        """Test TestAnalytics initialization."""
        assert temp_analytics.database_path.exists()
        assert len(temp_analytics.test_metrics) == 0
        assert len(temp_analytics.suite_history) == 0

    def test_database_initialization(self, temp_analytics):
        """Test database table creation."""
        with sqlite3.connect(temp_analytics.database_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

        expected_tables = ['test_results', 'suite_metrics', 'alerts']
        for table in expected_tables:
            assert table in tables

    def test_process_execution_results(self, temp_analytics, sample_results, sample_metadata):
        """Test processing of execution results."""
        suite_metrics = temp_analytics.process_execution_results(
            sample_results, sample_metadata
        )

        # Check suite metrics
        assert suite_metrics.total_tests == 3
        assert suite_metrics.executed_tests == 3
        assert suite_metrics.passed_tests == 1
        assert suite_metrics.failed_tests == 1
        assert suite_metrics.timeout_tests == 1

        # Check individual test metrics were created
        assert len(temp_analytics.test_metrics) == 3
        assert "test_a" in temp_analytics.test_metrics
        assert "test_b" in temp_analytics.test_metrics
        assert "test_c" in temp_analytics.test_metrics

    def test_test_metrics_update(self, temp_analytics):
        """Test individual test metrics update."""
        result = ExecutionResult(
            test_name="test_example",
            status=ExecutionStatus.COMPLETED,
            duration=1.5,
            start_time=time.time(),
            end_time=time.time() + 1.5
        )

        temp_analytics._update_test_metrics("test_example", result)

        metrics = temp_analytics.test_metrics["test_example"]
        assert metrics.total_runs == 1
        assert metrics.passed_runs == 1
        assert metrics.failed_runs == 0
        assert metrics.success_rate == 1.0
        assert metrics.avg_duration == 1.5

    def test_multiple_results_same_test(self, temp_analytics):
        """Test accumulation of multiple results for the same test."""
        # First result - pass
        result1 = ExecutionResult(
            test_name="test_example",
            status=ExecutionStatus.COMPLETED,
            duration=1.0,
            start_time=time.time(),
            end_time=time.time() + 1.0
        )

        # Second result - fail
        result2 = ExecutionResult(
            test_name="test_example",
            status=ExecutionStatus.FAILED,
            duration=2.0,
            start_time=time.time(),
            end_time=time.time() + 2.0,
            error_message="Test failed"
        )

        temp_analytics._update_test_metrics("test_example", result1)
        temp_analytics._update_test_metrics("test_example", result2)

        metrics = temp_analytics.test_metrics["test_example"]
        assert metrics.total_runs == 2
        assert metrics.passed_runs == 1
        assert metrics.failed_runs == 1
        assert metrics.success_rate == 0.5
        assert metrics.avg_duration == 1.5

    def test_error_type_extraction(self, temp_analytics):
        """Test error type extraction from error messages."""
        assert temp_analytics._extract_error_type("AssertionError: Test failed") == "AssertionError"
        assert temp_analytics._extract_error_type("TimeoutError: Operation timed out") == "TimeoutError"
        assert temp_analytics._extract_error_type("") == "Unknown"
        assert temp_analytics._extract_error_type(None) == "Unknown"
        assert temp_analytics._extract_error_type("Some random error message") == "Some random error message"[:50]

    def test_health_status_calculation(self, temp_analytics):
        """Test health status calculation."""
        assert temp_analytics._calculate_health_status(0.98) == HealthStatus.EXCELLENT
        assert temp_analytics._calculate_health_status(0.92) == HealthStatus.GOOD
        assert temp_analytics._calculate_health_status(0.85) == HealthStatus.FAIR
        assert temp_analytics._calculate_health_status(0.75) == HealthStatus.POOR
        assert temp_analytics._calculate_health_status(0.65) == HealthStatus.CRITICAL

    def test_percentile_calculation(self, temp_analytics):
        """Test percentile calculation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Test various percentiles
        assert temp_analytics._percentile(data, 50) == 5.5  # Median
        assert temp_analytics._percentile(data, 0) == 1     # Minimum
        assert temp_analytics._percentile(data, 100) == 10  # Maximum
        assert temp_analytics._percentile(data, 25) == 3.25 # First quartile

        # Edge cases
        assert temp_analytics._percentile([], 50) == 0.0
        assert temp_analytics._percentile([5], 50) == 5.0

    def test_get_test_report(self, temp_analytics, sample_results, sample_metadata):
        """Test individual test report generation."""
        temp_analytics.process_execution_results(sample_results, sample_metadata)

        report = temp_analytics.get_test_report("test_a")

        assert report is not None
        assert report['test_name'] == "test_a"
        assert 'metrics' in report
        assert 'recent_failures' in report
        assert 'performance_statistics' in report
        assert 'recommendations' in report

        # Non-existent test
        assert temp_analytics.get_test_report("non_existent") is None

    def test_get_suite_report(self, temp_analytics, sample_results, sample_metadata):
        """Test suite report generation."""
        temp_analytics.process_execution_results(sample_results, sample_metadata)

        report = temp_analytics.get_suite_report()

        assert 'current_metrics' in report
        assert 'trending_tests' in report
        assert 'flaky_tests' in report
        assert 'performance_insights' in report
        assert 'recommendations' in report
        assert 'alerts' in report

    def test_data_export_json(self, temp_analytics, sample_results, sample_metadata):
        """Test JSON data export."""
        temp_analytics.process_execution_results(sample_results, sample_metadata)

        # Export as string
        json_data = temp_analytics.export_data(format='json')
        assert isinstance(json_data, str)
        assert 'test_metrics' in json_data
        assert 'suite_history' in json_data

        # Export to file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_path = Path(f.name)

        result_path = temp_analytics.export_data(format='json', file_path=output_path)
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Clean up
        output_path.unlink()

    def test_data_export_unsupported_format(self, temp_analytics):
        """Test unsupported export format handling."""
        with pytest.raises(ValueError, match="Unsupported export format"):
            temp_analytics.export_data(format='xml')

    def test_database_storage_and_retrieval(self, temp_analytics):
        """Test database storage and retrieval operations."""
        result = ExecutionResult(
            test_name="test_db",
            status=ExecutionStatus.COMPLETED,
            duration=1.5,
            start_time=1234567890.0,
            end_time=1234567891.5
        )

        temp_analytics._store_test_result("test_db", result)

        # Verify data was stored
        with sqlite3.connect(temp_analytics.database_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM test_results WHERE test_name = ?",
                ("test_db",)
            )
            rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][0] == "test_db"  # test_name
        assert rows[0][2] == "COMPLETED"  # status

    def test_suite_metrics_storage(self, temp_analytics):
        """Test suite metrics database storage."""
        suite_metrics = SuiteMetrics(
            total_tests=10,
            passed_tests=8,
            failed_tests=2,
            total_duration=50.0,
            overall_success_rate=0.8,
            health_status=HealthStatus.GOOD
        )

        temp_analytics._store_suite_metrics(suite_metrics)

        # Verify data was stored
        with sqlite3.connect(temp_analytics.database_path) as conn:
            cursor = conn.execute("SELECT * FROM suite_metrics")
            rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0][1] == 10  # total_tests
        assert rows[0][2] == 8   # passed_tests

    def test_database_error_handling(self, temp_analytics):
        """Test graceful handling of database errors."""
        # Close the database file to simulate error
        temp_analytics.database_path.unlink()

        # These operations should not crash
        result = ExecutionResult(
            test_name="test_error",
            status=ExecutionStatus.COMPLETED,
            duration=1.0,
            start_time=time.time(),
            end_time=time.time() + 1.0
        )

        temp_analytics._store_test_result("test_error", result)  # Should not crash

        suite_metrics = SuiteMetrics()
        temp_analytics._store_suite_metrics(suite_metrics)  # Should not crash

    def test_trending_tests_identification(self, temp_analytics):
        """Test identification of trending tests."""
        # Create test with improving trend
        improving_metrics = TestMetrics(
            test_name="improving_test",
            performance_trend=TrendDirection.IMPROVING
        )

        # Create test with degrading trend
        degrading_metrics = TestMetrics(
            test_name="degrading_test",
            reliability_trend=TrendDirection.DEGRADING
        )

        # Create test with volatile trend
        volatile_metrics = TestMetrics(
            test_name="volatile_test",
            reliability_trend=TrendDirection.VOLATILE
        )

        temp_analytics.test_metrics = {
            "improving_test": improving_metrics,
            "degrading_test": degrading_metrics,
            "volatile_test": volatile_metrics
        }

        trends = temp_analytics._get_trending_tests()

        assert "improving_test" in trends['improving']
        assert "degrading_test" in trends['degrading']
        assert "volatile_test" in trends['volatile']

    def test_flaky_test_summary(self, temp_analytics):
        """Test flaky test summary generation."""
        # Create flaky test metrics
        flaky_metrics = TestMetrics(
            test_name="flaky_test",
            flakiness_score=0.7,
            success_rate=0.6,
            total_runs=50
        )

        # Create stable test metrics
        stable_metrics = TestMetrics(
            test_name="stable_test",
            flakiness_score=0.1,
            success_rate=0.98,
            total_runs=50
        )

        temp_analytics.test_metrics = {
            "flaky_test": flaky_metrics,
            "stable_test": stable_metrics
        }

        summary = temp_analytics._get_flaky_test_summary()

        # Should only include flaky test
        assert len(summary) == 1
        assert summary[0]['test_name'] == "flaky_test"
        assert summary[0]['flakiness_score'] == 0.7

    def test_performance_insights(self, temp_analytics):
        """Test performance insights generation."""
        # Add some suite history
        suite1 = SuiteMetrics(total_duration=100.0, suite_efficiency=10.0)
        suite2 = SuiteMetrics(total_duration=120.0, suite_efficiency=8.0)
        temp_analytics.suite_history = [suite1, suite2]

        # Add test metrics
        fast_test = TestMetrics(test_name="fast_test", avg_duration=1.0, max_duration=1.5)
        slow_test = TestMetrics(test_name="slow_test", avg_duration=30.0, max_duration=35.0)
        temp_analytics.test_metrics = {"fast_test": fast_test, "slow_test": slow_test}

        insights = temp_analytics._get_performance_insights()

        assert 'average_suite_duration' in insights
        assert 'suite_efficiency_trend' in insights
        assert 'slowest_tests' in insights

        # Slow test should be in slowest tests
        slowest = insights['slowest_tests']
        assert len(slowest) > 0
        assert slowest[0]['test_name'] == "slow_test"


def test_test_metrics_creation():
    """Test TestMetrics dataclass creation and defaults."""
    metrics = TestMetrics(test_name="test_example")

    assert metrics.test_name == "test_example"
    assert metrics.total_runs == 0
    assert metrics.passed_runs == 0
    assert metrics.failed_runs == 0
    assert metrics.success_rate == 0.0
    assert metrics.flakiness_score == 0.0
    assert isinstance(metrics.duration_history, deque)
    assert isinstance(metrics.result_history, deque)


def test_suite_metrics_creation():
    """Test SuiteMetrics dataclass creation and defaults."""
    metrics = SuiteMetrics()

    assert metrics.total_tests == 0
    assert metrics.executed_tests == 0
    assert metrics.overall_success_rate == 0.0
    assert metrics.health_status == HealthStatus.EXCELLENT
    assert isinstance(metrics.category_breakdown, dict)
    assert isinstance(metrics.alerts, list)


def test_alert_creation():
    """Test Alert dataclass creation."""
    alert = Alert(
        level=AlertLevel.WARNING,
        message="Test alert",
        test_name="test_example"
    )

    assert alert.level == AlertLevel.WARNING
    assert alert.message == "Test alert"
    assert alert.test_name == "test_example"
    assert not alert.acknowledged
    assert alert.timestamp > 0


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])