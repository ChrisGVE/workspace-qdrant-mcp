"""
Performance trend analyzer for test results.

Analyzes historical test data to identify trends in:
- Success rates over time
- Execution time trends
- Coverage trends
- Flakiness trends
- Benchmark performance trends

Provides health indicators and trend directions for executive dashboards.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .models import TestRun
from .storage import TestResultStorage


class HealthStatus(str, Enum):
    """Health status indicators."""

    EXCELLENT = "excellent"  # Green - everything great
    GOOD = "good"  # Light green - minor issues
    WARNING = "warning"  # Yellow - attention needed
    CRITICAL = "critical"  # Red - urgent action required


class TrendDirection(str, Enum):
    """Trend direction indicators."""

    IMPROVING = "improving"  # Upward arrow - getting better
    STABLE = "stable"  # Flat arrow - no significant change
    DECLINING = "declining"  # Downward arrow - getting worse


class TrendAnalyzer:
    """
    Analyze historical trends in test results.

    Provides trend analysis for success rates, execution times, coverage,
    flakiness, and benchmark performance over configurable time windows.
    """

    def __init__(self, storage: Optional[TestResultStorage] = None):
        """
        Initialize trend analyzer.

        Args:
            storage: Storage backend for test results
        """
        self.storage = storage or TestResultStorage()

    def analyze_success_rate_trend(
        self, days: int = 30, min_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze success rate trend over time.

        Args:
            days: Number of days to analyze
            min_runs: Minimum number of runs required for trend analysis

        Returns:
            Trend analysis with data points, direction, and statistics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get test runs in time window
        runs = self.storage.list_test_runs(
            limit=1000, start_date=start_date, end_date=end_date
        )

        if len(runs) < min_runs:
            return {
                "trend_direction": TrendDirection.STABLE,
                "data_points": [],
                "insufficient_data": True,
                "message": f"Need at least {min_runs} test runs for trend analysis",
            }

        # Extract success rate data points
        data_points = []
        for run in runs:
            data_points.append(
                {
                    "timestamp": run["timestamp"],
                    "success_rate": run["success_rate"],
                    "total_tests": run["total_tests"],
                    "passed_tests": run["passed_tests"],
                    "failed_tests": run["failed_tests"],
                }
            )

        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])

        # Calculate trend direction
        trend_direction = self._calculate_trend_direction(
            [p["success_rate"] for p in data_points]
        )

        # Calculate statistics
        success_rates = [p["success_rate"] for p in data_points]
        stats = {
            "current": success_rates[-1] if success_rates else 0,
            "average": sum(success_rates) / len(success_rates) if success_rates else 0,
            "min": min(success_rates) if success_rates else 0,
            "max": max(success_rates) if success_rates else 0,
            "change_from_first": success_rates[-1] - success_rates[0]
            if len(success_rates) >= 2
            else 0,
        }

        return {
            "trend_direction": trend_direction,
            "data_points": data_points,
            "statistics": stats,
            "days_analyzed": days,
            "total_runs": len(data_points),
            "insufficient_data": False,
        }

    def analyze_execution_time_trend(
        self, days: int = 30, min_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze test execution time trend over time.

        Args:
            days: Number of days to analyze
            min_runs: Minimum number of runs required for trend analysis

        Returns:
            Trend analysis with data points, direction, and statistics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get test runs in time window
        runs = self.storage.list_test_runs(
            limit=1000, start_date=start_date, end_date=end_date
        )

        if len(runs) < min_runs:
            return {
                "trend_direction": TrendDirection.STABLE,
                "data_points": [],
                "insufficient_data": True,
                "message": f"Need at least {min_runs} test runs for trend analysis",
            }

        # Calculate total execution time for each run
        data_points = []
        for run_summary in runs:
            # Load full test run to calculate total duration
            test_run = self.storage.get_test_run(run_summary["run_id"])
            if not test_run:
                continue

            # Sum all test durations
            total_duration_ms = 0
            for suite in test_run.suites:
                for case in suite.test_cases:
                    for result in case.results:
                        total_duration_ms += result.duration_ms

            data_points.append(
                {
                    "timestamp": run_summary["timestamp"],
                    "total_duration_ms": total_duration_ms,
                    "total_duration_seconds": total_duration_ms / 1000,
                    "total_tests": test_run.total_tests,
                    "avg_test_duration_ms": total_duration_ms / test_run.total_tests
                    if test_run.total_tests > 0
                    else 0,
                }
            )

        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])

        # Calculate trend direction (inverted - lower is better)
        trend_direction = self._calculate_trend_direction(
            [p["total_duration_ms"] for p in data_points], lower_is_better=True
        )

        # Calculate statistics
        durations = [p["total_duration_ms"] for p in data_points]
        stats = {
            "current_ms": durations[-1] if durations else 0,
            "average_ms": sum(durations) / len(durations) if durations else 0,
            "min_ms": min(durations) if durations else 0,
            "max_ms": max(durations) if durations else 0,
            "change_from_first_ms": durations[-1] - durations[0]
            if len(durations) >= 2
            else 0,
        }

        return {
            "trend_direction": trend_direction,
            "data_points": data_points,
            "statistics": stats,
            "days_analyzed": days,
            "total_runs": len(data_points),
            "insufficient_data": False,
        }

    def analyze_coverage_trend(
        self, days: int = 30, min_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze code coverage trend over time.

        Args:
            days: Number of days to analyze
            min_runs: Minimum number of runs required for trend analysis

        Returns:
            Trend analysis with data points, direction, and statistics
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get test runs in time window
        runs = self.storage.list_test_runs(
            limit=1000, start_date=start_date, end_date=end_date
        )

        # Filter runs with coverage data
        data_points = []
        for run_summary in runs:
            test_run = self.storage.get_test_run(run_summary["run_id"])
            if not test_run or not test_run.coverage:
                continue

            coverage = test_run.coverage
            data_points.append(
                {
                    "timestamp": run_summary["timestamp"],
                    "line_coverage": coverage.line_coverage_percent,
                    "function_coverage": coverage.function_coverage_percent,
                    "branch_coverage": coverage.branch_coverage_percent,
                    "lines_covered": coverage.lines_covered,
                    "lines_total": coverage.lines_total,
                }
            )

        if len(data_points) < min_runs:
            return {
                "trend_direction": TrendDirection.STABLE,
                "data_points": [],
                "insufficient_data": True,
                "message": f"Need at least {min_runs} test runs with coverage for trend analysis",
            }

        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])

        # Calculate trend direction
        trend_direction = self._calculate_trend_direction(
            [p["line_coverage"] for p in data_points]
        )

        # Calculate statistics
        line_coverages = [p["line_coverage"] for p in data_points]
        stats = {
            "current": line_coverages[-1] if line_coverages else 0,
            "average": sum(line_coverages) / len(line_coverages)
            if line_coverages
            else 0,
            "min": min(line_coverages) if line_coverages else 0,
            "max": max(line_coverages) if line_coverages else 0,
            "change_from_first": line_coverages[-1] - line_coverages[0]
            if len(line_coverages) >= 2
            else 0,
        }

        # Add function and branch coverage if available
        function_coverages = [
            p["function_coverage"]
            for p in data_points
            if p["function_coverage"] is not None
        ]
        if function_coverages:
            stats["function_current"] = function_coverages[-1]
            stats["function_average"] = sum(function_coverages) / len(
                function_coverages
            )

        branch_coverages = [
            p["branch_coverage"]
            for p in data_points
            if p["branch_coverage"] is not None
        ]
        if branch_coverages:
            stats["branch_current"] = branch_coverages[-1]
            stats["branch_average"] = sum(branch_coverages) / len(branch_coverages)

        return {
            "trend_direction": trend_direction,
            "data_points": data_points,
            "statistics": stats,
            "days_analyzed": days,
            "total_runs": len(data_points),
            "insufficient_data": False,
        }

    def analyze_flakiness_trend(
        self, days: int = 30, min_runs: int = 5
    ) -> Dict[str, Any]:
        """
        Analyze test flakiness trend over time.

        Args:
            days: Number of days to analyze
            min_runs: Minimum number of runs required for flakiness analysis

        Returns:
            Trend analysis with flaky test counts and trend direction
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get failure analysis reports in time window
        reports = self.storage.list_failure_analysis_reports(limit=1000)

        # Filter by time window
        filtered_reports = []
        for report_summary in reports:
            report_time = datetime.fromisoformat(report_summary["timestamp"])
            if start_date <= report_time <= end_date:
                filtered_reports.append(report_summary)

        if len(filtered_reports) < min_runs:
            return {
                "trend_direction": TrendDirection.STABLE,
                "data_points": [],
                "insufficient_data": True,
                "message": f"Need at least {min_runs} failure analysis reports for trend analysis",
            }

        # Extract flaky test counts
        data_points = []
        for report_summary in filtered_reports:
            data_points.append(
                {
                    "timestamp": report_summary["timestamp"],
                    "total_flaky_tests": report_summary["total_flaky_tests"],
                    "total_failure_patterns": report_summary[
                        "total_failure_patterns"
                    ],
                }
            )

        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])

        # Calculate trend direction (inverted - lower is better)
        trend_direction = self._calculate_trend_direction(
            [p["total_flaky_tests"] for p in data_points], lower_is_better=True
        )

        # Calculate statistics
        flaky_counts = [p["total_flaky_tests"] for p in data_points]
        stats = {
            "current": flaky_counts[-1] if flaky_counts else 0,
            "average": sum(flaky_counts) / len(flaky_counts) if flaky_counts else 0,
            "min": min(flaky_counts) if flaky_counts else 0,
            "max": max(flaky_counts) if flaky_counts else 0,
            "change_from_first": flaky_counts[-1] - flaky_counts[0]
            if len(flaky_counts) >= 2
            else 0,
        }

        return {
            "trend_direction": trend_direction,
            "data_points": data_points,
            "statistics": stats,
            "days_analyzed": days,
            "total_reports": len(data_points),
            "insufficient_data": False,
        }

    def analyze_benchmark_performance_trend(
        self, benchmark_name: str, metric: str = "avg_ms", days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze performance trend for a specific benchmark.

        Args:
            benchmark_name: Name of benchmark to analyze
            metric: Metric to track (avg_ms, p95_ms, operations_per_second, etc.)
            days: Number of days to analyze

        Returns:
            Trend analysis with data points and trend direction
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Get test runs in time window
        runs = self.storage.list_test_runs(
            limit=1000, start_date=start_date, end_date=end_date
        )

        # Find benchmark results
        data_points = []
        for run_summary in runs:
            test_run = self.storage.get_test_run(run_summary["run_id"])
            if not test_run:
                continue

            # Look for benchmark in test suites
            for suite in test_run.suites:
                for case in suite.test_cases:
                    if case.name == benchmark_name:
                        for result in case.results:
                            if result.performance:
                                metric_value = getattr(
                                    result.performance, metric, None
                                )
                                if metric_value is not None:
                                    data_points.append(
                                        {
                                            "timestamp": result.timestamp.isoformat(),
                                            "metric": metric,
                                            "value": metric_value,
                                            "benchmark_name": benchmark_name,
                                        }
                                    )

        if len(data_points) < 3:
            return {
                "trend_direction": TrendDirection.STABLE,
                "data_points": [],
                "insufficient_data": True,
                "message": f"Insufficient data for benchmark '{benchmark_name}'",
            }

        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])

        # Calculate trend direction (depends on metric - lower is better for time metrics)
        lower_is_better = metric.endswith("_ms") or metric in [
            "duration_ms",
            "latency_ms",
        ]
        trend_direction = self._calculate_trend_direction(
            [p["value"] for p in data_points], lower_is_better=lower_is_better
        )

        # Calculate statistics
        values = [p["value"] for p in data_points]
        stats = {
            "current": values[-1] if values else 0,
            "average": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "change_from_first": values[-1] - values[0] if len(values) >= 2 else 0,
        }

        return {
            "trend_direction": trend_direction,
            "data_points": data_points,
            "statistics": stats,
            "benchmark_name": benchmark_name,
            "metric": metric,
            "days_analyzed": days,
            "total_data_points": len(data_points),
            "insufficient_data": False,
        }

    def calculate_health_status(self, test_run: TestRun) -> HealthStatus:
        """
        Calculate overall health status for a test run.

        Args:
            test_run: Test run to evaluate

        Returns:
            Health status (excellent, good, warning, critical)
        """
        success_rate = test_run.success_rate

        # Coverage check
        has_coverage = test_run.coverage is not None
        coverage_percent = (
            test_run.coverage.line_coverage_percent if has_coverage else 0
        )

        # Determine health status based on success rate and coverage
        if success_rate >= 95 and (not has_coverage or coverage_percent >= 80):
            return HealthStatus.EXCELLENT
        elif success_rate >= 90 and (not has_coverage or coverage_percent >= 70):
            return HealthStatus.GOOD
        elif success_rate >= 75:
            return HealthStatus.WARNING
        else:
            return HealthStatus.CRITICAL

    def _calculate_trend_direction(
        self, values: List[float], lower_is_better: bool = False
    ) -> TrendDirection:
        """
        Calculate trend direction from a series of values.

        Uses linear regression slope to determine if values are improving,
        declining, or stable over time.

        Args:
            values: List of metric values over time
            lower_is_better: If True, declining values indicate improvement

        Returns:
            Trend direction (improving, stable, declining)
        """
        if len(values) < 2:
            return TrendDirection.STABLE

        # Calculate linear regression slope
        n = len(values)
        x = list(range(n))
        y = values

        # Calculate slope: (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return TrendDirection.STABLE

        slope = (n * sum_xy - sum_x * sum_y) / denominator

        # Determine significance threshold (2% of average value)
        avg_value = sum_y / n
        significance_threshold = abs(avg_value) * 0.02

        # Classify trend
        if abs(slope) < significance_threshold:
            return TrendDirection.STABLE
        elif slope > 0:
            # Positive slope
            return TrendDirection.DECLINING if lower_is_better else TrendDirection.IMPROVING
        else:
            # Negative slope
            return TrendDirection.IMPROVING if lower_is_better else TrendDirection.DECLINING

    def get_action_items(
        self, health_status: HealthStatus, trends: Dict[str, Any]
    ) -> List[str]:
        """
        Generate action items based on health status and trends.

        Args:
            health_status: Current health status
            trends: Dictionary of trend analyses

        Returns:
            List of recommended action items
        """
        action_items = []

        # Health-based actions
        if health_status == HealthStatus.CRITICAL:
            action_items.append("URGENT: Investigate and fix failing tests immediately")
            action_items.append("Review recent code changes for breaking changes")
        elif health_status == HealthStatus.WARNING:
            action_items.append("Address failing tests before next release")
            action_items.append("Review test stability and flakiness")

        # Success rate trend actions
        success_trend = trends.get("success_rate", {})
        if success_trend.get("trend_direction") == TrendDirection.DECLINING:
            action_items.append(
                "Success rate declining - review recent test failures"
            )

        # Coverage trend actions
        coverage_trend = trends.get("coverage", {})
        if coverage_trend.get("trend_direction") == TrendDirection.DECLINING:
            action_items.append("Code coverage declining - add tests for new code")
        if not coverage_trend.get("insufficient_data"):
            stats = coverage_trend.get("statistics", {})
            if stats.get("current", 100) < 80:
                action_items.append(
                    "Coverage below 80% - prioritize test coverage improvements"
                )

        # Execution time trend actions
        execution_trend = trends.get("execution_time", {})
        if execution_trend.get("trend_direction") == TrendDirection.DECLINING:
            action_items.append(
                "Test execution time increasing - optimize slow tests"
            )

        # Flakiness trend actions
        flakiness_trend = trends.get("flakiness", {})
        if not flakiness_trend.get("insufficient_data"):
            stats = flakiness_trend.get("statistics", {})
            if stats.get("current", 0) > 0:
                action_items.append(
                    f"Fix {int(stats['current'])} flaky tests to improve reliability"
                )

        # Default action if all is well
        if not action_items:
            action_items.append("All metrics healthy - maintain current quality standards")

        return action_items
