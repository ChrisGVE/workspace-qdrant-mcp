"""
Unit tests for performance dashboard and trend analyzer.

Tests the TrendAnalyzer and PerformanceDashboard classes.
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from tests.reporting.models import (
    CoverageMetrics,
    PerformanceMetrics,
    TestCase,
    TestResult,
    TestRun,
    TestSource,
    TestStatus,
    TestSuite,
    TestType,
)
from tests.reporting.performance_dashboard import PerformanceDashboard
from tests.reporting.storage import TestResultStorage
from tests.reporting.trend_analyzer import HealthStatus, TrendAnalyzer, TrendDirection


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        storage = TestResultStorage(db_path)
        yield storage


@pytest.fixture
def sample_test_run():
    """Create a sample test run."""
    test_run = TestRun.create(
        source=TestSource.PYTEST, timestamp=datetime.now(), metadata={"test": True}
    )

    # Add unit test suite
    suite = TestSuite(
        suite_id=str(uuid4()), name="unit_tests", test_type=TestType.UNIT
    )

    # Add test cases
    for i in range(10):
        case = TestCase(case_id=str(uuid4()), name=f"test_function_{i}")
        result = TestResult(
            test_id=str(uuid4()),
            name=f"test_function_{i}",
            status=TestStatus.PASSED if i < 9 else TestStatus.FAILED,
            duration_ms=100.0 + i * 10,
            timestamp=datetime.now(),
        )
        case.add_result(result)
        suite.add_test_case(case)

    test_run.add_suite(suite)

    # Add coverage
    test_run.coverage = CoverageMetrics(
        line_coverage_percent=85.0, lines_covered=850, lines_total=1000
    )

    return test_run


class TestTrendAnalyzer:
    """Tests for TrendAnalyzer."""

    def test_initialization(self, temp_storage):
        """Test TrendAnalyzer initialization."""
        analyzer = TrendAnalyzer(temp_storage)
        assert analyzer.storage == temp_storage

    def test_calculate_health_status_excellent(self, temp_storage):
        """Test health status calculation - excellent."""
        # Create test run with 98% success rate and 85% coverage
        test_run = TestRun.create(source=TestSource.PYTEST, timestamp=datetime.now())
        suite = TestSuite(
            suite_id=str(uuid4()), name="tests", test_type=TestType.UNIT
        )

        # 98% pass rate (49/50)
        for i in range(50):
            case = TestCase(case_id=str(uuid4()), name=f"test_{i}")
            result = TestResult(
                test_id=str(uuid4()),
                name=f"test_{i}",
                status=TestStatus.PASSED if i < 49 else TestStatus.FAILED,
                duration_ms=100.0,
                timestamp=datetime.now(),
            )
            case.add_result(result)
            suite.add_test_case(case)

        test_run.add_suite(suite)
        test_run.coverage = CoverageMetrics(
            line_coverage_percent=85.0, lines_covered=850, lines_total=1000
        )

        analyzer = TrendAnalyzer(temp_storage)
        status = analyzer.calculate_health_status(test_run)

        assert status == HealthStatus.EXCELLENT

    def test_calculate_health_status_good(self, temp_storage):
        """Test health status calculation - good."""
        # Create test run with 92% success rate
        test_run = TestRun.create(source=TestSource.PYTEST, timestamp=datetime.now())
        suite = TestSuite(
            suite_id=str(uuid4()), name="tests", test_type=TestType.UNIT
        )

        # 92% pass rate (46/50)
        for i in range(50):
            case = TestCase(case_id=str(uuid4()), name=f"test_{i}")
            result = TestResult(
                test_id=str(uuid4()),
                name=f"test_{i}",
                status=TestStatus.PASSED if i < 46 else TestStatus.FAILED,
                duration_ms=100.0,
                timestamp=datetime.now(),
            )
            case.add_result(result)
            suite.add_test_case(case)

        test_run.add_suite(suite)
        test_run.coverage = CoverageMetrics(
            line_coverage_percent=75.0, lines_covered=750, lines_total=1000
        )

        analyzer = TrendAnalyzer(temp_storage)
        status = analyzer.calculate_health_status(test_run)

        assert status == HealthStatus.GOOD

    def test_calculate_health_status_warning(self, temp_storage):
        """Test health status calculation - warning."""
        # Create test run with 80% success rate
        test_run = TestRun.create(source=TestSource.PYTEST, timestamp=datetime.now())
        suite = TestSuite(
            suite_id=str(uuid4()), name="tests", test_type=TestType.UNIT
        )

        # 80% pass rate
        for i in range(50):
            case = TestCase(case_id=str(uuid4()), name=f"test_{i}")
            result = TestResult(
                test_id=str(uuid4()),
                name=f"test_{i}",
                status=TestStatus.PASSED if i < 40 else TestStatus.FAILED,
                duration_ms=100.0,
                timestamp=datetime.now(),
            )
            case.add_result(result)
            suite.add_test_case(case)

        test_run.add_suite(suite)

        analyzer = TrendAnalyzer(temp_storage)
        status = analyzer.calculate_health_status(test_run)

        assert status == HealthStatus.WARNING

    def test_calculate_health_status_critical(self, temp_storage):
        """Test health status calculation - critical."""
        # Create test run with 70% success rate
        test_run = TestRun.create(source=TestSource.PYTEST, timestamp=datetime.now())
        suite = TestSuite(
            suite_id=str(uuid4()), name="tests", test_type=TestType.UNIT
        )

        # 70% pass rate
        for i in range(50):
            case = TestCase(case_id=str(uuid4()), name=f"test_{i}")
            result = TestResult(
                test_id=str(uuid4()),
                name=f"test_{i}",
                status=TestStatus.PASSED if i < 35 else TestStatus.FAILED,
                duration_ms=100.0,
                timestamp=datetime.now(),
            )
            case.add_result(result)
            suite.add_test_case(case)

        test_run.add_suite(suite)

        analyzer = TrendAnalyzer(temp_storage)
        status = analyzer.calculate_health_status(test_run)

        assert status == HealthStatus.CRITICAL

    def test_calculate_trend_direction(self, temp_storage):
        """Test trend direction calculation."""
        analyzer = TrendAnalyzer(temp_storage)

        # Improving trend
        improving_values = [80, 82, 85, 88, 90, 92, 95]
        direction = analyzer._calculate_trend_direction(improving_values)
        assert direction == TrendDirection.IMPROVING

        # Declining trend
        declining_values = [95, 92, 90, 88, 85, 82, 80]
        direction = analyzer._calculate_trend_direction(declining_values)
        assert direction == TrendDirection.DECLINING

        # Stable trend
        stable_values = [90, 90.5, 89.5, 90, 90.2, 89.8, 90]
        direction = analyzer._calculate_trend_direction(stable_values)
        assert direction == TrendDirection.STABLE

    def test_analyze_success_rate_trend_insufficient_data(self, temp_storage):
        """Test success rate trend with insufficient data."""
        analyzer = TrendAnalyzer(temp_storage)

        # No test runs yet
        trend = analyzer.analyze_success_rate_trend(days=7, min_runs=3)

        assert trend["insufficient_data"] is True
        assert trend["trend_direction"] == TrendDirection.STABLE
        assert len(trend["data_points"]) == 0

    def test_get_action_items_critical(self, temp_storage):
        """Test action item generation for critical health."""
        analyzer = TrendAnalyzer(temp_storage)

        action_items = analyzer.get_action_items(HealthStatus.CRITICAL, {})

        assert any("URGENT" in item for item in action_items)
        assert any("failing tests" in item.lower() for item in action_items)

    def test_get_action_items_excellent(self, temp_storage):
        """Test action item generation for excellent health."""
        analyzer = TrendAnalyzer(temp_storage)

        action_items = analyzer.get_action_items(HealthStatus.EXCELLENT, {})

        assert any("healthy" in item.lower() for item in action_items)


class TestPerformanceDashboard:
    """Tests for PerformanceDashboard."""

    def test_initialization(self, temp_storage):
        """Test PerformanceDashboard initialization."""
        dashboard = PerformanceDashboard(temp_storage)
        assert dashboard.storage == temp_storage
        assert dashboard.trend_analyzer is not None

    def test_generate_executive_summary(self, temp_storage, sample_test_run):
        """Test executive summary generation."""
        dashboard = PerformanceDashboard(temp_storage)

        # Save test run
        temp_storage.save_test_run(sample_test_run)

        # Generate summary
        summary = dashboard.generate_executive_summary(
            sample_test_run, include_trends=False
        )

        assert "health_status" in summary
        assert "key_metrics" in summary
        assert summary["key_metrics"]["total_tests"] == 10
        assert summary["key_metrics"]["passed_tests"] == 9
        assert summary["key_metrics"]["failed_tests"] == 1
        assert summary["key_metrics"]["line_coverage"] == 85.0

    def test_generate_health_indicators(self, temp_storage, sample_test_run):
        """Test health indicator generation."""
        dashboard = PerformanceDashboard(temp_storage)
        temp_storage.save_test_run(sample_test_run)

        indicators = dashboard.generate_health_indicators(sample_test_run)

        assert "success_rate" in indicators
        assert "coverage" in indicators
        assert "failed_tests" in indicators

        # Check success rate indicator
        assert indicators["success_rate"]["value"] == sample_test_run.success_rate
        assert indicators["success_rate"]["status"] in [
            "excellent",
            "good",
            "warning",
            "critical",
        ]
        assert indicators["success_rate"]["unit"] == "%"

    def test_generate_dashboard_charts(self, temp_storage, sample_test_run):
        """Test dashboard chart generation."""
        dashboard = PerformanceDashboard(temp_storage)
        temp_storage.save_test_run(sample_test_run)

        charts = dashboard.generate_dashboard_charts(
            sample_test_run, time_windows=[7, 30]
        )

        # Should have health gauge
        assert "health_status" in charts

        # Check chart structure
        health_chart = charts["health_status"]
        assert health_chart["type"] == "doughnut"
        assert "data" in health_chart
        assert "options" in health_chart

    def test_get_trend_arrow(self):
        """Test trend arrow symbol generation."""
        assert PerformanceDashboard._get_trend_arrow(TrendDirection.IMPROVING) == "↑"
        assert PerformanceDashboard._get_trend_arrow(TrendDirection.DECLINING) == "↓"
        assert PerformanceDashboard._get_trend_arrow(TrendDirection.STABLE) == "→"

    def test_get_success_rate_status(self):
        """Test success rate status color determination."""
        assert PerformanceDashboard._get_success_rate_status(98) == "excellent"
        assert PerformanceDashboard._get_success_rate_status(92) == "good"
        assert PerformanceDashboard._get_success_rate_status(80) == "warning"
        assert PerformanceDashboard._get_success_rate_status(70) == "critical"

    def test_get_coverage_status(self):
        """Test coverage status color determination."""
        assert PerformanceDashboard._get_coverage_status(85) == "excellent"
        assert PerformanceDashboard._get_coverage_status(75) == "good"
        assert PerformanceDashboard._get_coverage_status(65) == "warning"
        assert PerformanceDashboard._get_coverage_status(55) == "critical"

    def test_get_failed_tests_status(self):
        """Test failed tests status color determination."""
        assert PerformanceDashboard._get_failed_tests_status(0) == "excellent"
        assert PerformanceDashboard._get_failed_tests_status(1) == "good"
        assert PerformanceDashboard._get_failed_tests_status(3) == "warning"
        assert PerformanceDashboard._get_failed_tests_status(10) == "critical"

    def test_get_flakiness_status(self):
        """Test flakiness status color determination."""
        assert PerformanceDashboard._get_flakiness_status(0) == "excellent"
        assert PerformanceDashboard._get_flakiness_status(1) == "good"
        assert PerformanceDashboard._get_flakiness_status(2) == "warning"
        assert PerformanceDashboard._get_flakiness_status(5) == "critical"


class TestDashboardIntegration:
    """Integration tests for dashboard features."""

    def test_full_dashboard_generation(self, temp_storage):
        """Test complete dashboard generation workflow."""
        # Create multiple test runs over time
        runs = []
        for i in range(5):
            timestamp = datetime.now() - timedelta(days=5 - i)
            test_run = TestRun.create(
                source=TestSource.PYTEST,
                timestamp=timestamp,
                metadata={"run": i + 1},
            )

            suite = TestSuite(
                suite_id=str(uuid4()), name="tests", test_type=TestType.UNIT
            )

            # Gradually improving success rate
            total_tests = 50
            passed_tests = int(40 + i * 2)  # 80% -> 96%

            for j in range(total_tests):
                case = TestCase(case_id=str(uuid4()), name=f"test_{j}")
                result = TestResult(
                    test_id=str(uuid4()),
                    name=f"test_{j}",
                    status=TestStatus.PASSED
                    if j < passed_tests
                    else TestStatus.FAILED,
                    duration_ms=100.0,
                    timestamp=timestamp,
                )
                case.add_result(result)
                suite.add_test_case(case)

            test_run.add_suite(suite)
            test_run.coverage = CoverageMetrics(
                line_coverage_percent=70.0 + i * 3,  # Improving coverage
                lines_covered=int((70.0 + i * 3) * 10),
                lines_total=1000,
            )

            temp_storage.save_test_run(test_run)
            runs.append(test_run)

        # Generate dashboard for latest run
        dashboard = PerformanceDashboard(temp_storage)
        latest_run = runs[-1]

        summary = dashboard.generate_executive_summary(
            latest_run, include_trends=True
        )
        indicators = dashboard.generate_health_indicators(latest_run)
        charts = dashboard.generate_dashboard_charts(latest_run, time_windows=[7])

        # Verify summary
        assert summary["health_status"] in [
            HealthStatus.EXCELLENT,
            HealthStatus.GOOD,
            HealthStatus.WARNING,
            HealthStatus.CRITICAL,
        ]
        assert "key_metrics" in summary
        assert "trends" in summary

        # Verify indicators
        assert len(indicators) >= 3  # At least success rate, coverage, failed tests

        # Verify charts
        assert "health_status" in charts
        assert charts["health_status"]["type"] == "doughnut"
