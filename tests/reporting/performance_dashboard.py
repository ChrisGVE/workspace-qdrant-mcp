"""
Executive performance dashboard generator.

Creates comprehensive executive dashboards with:
- Executive summary with key metrics
- Health indicators (green/yellow/red status)
- Performance trend visualizations
- Interactive charts with drill-down capabilities
- Action items and recommendations
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Union

from .models import TestRun
from .storage import TestResultStorage
from .trend_analyzer import HealthStatus, TrendAnalyzer, TrendDirection


class PerformanceDashboard:
    """
    Generate executive performance dashboards.

    Provides high-level overview of test health, trends, and actionable insights
    for stakeholders and management.
    """

    def __init__(
        self,
        storage: TestResultStorage | None = None,
        trend_analyzer: TrendAnalyzer | None = None,
    ):
        """
        Initialize performance dashboard.

        Args:
            storage: Storage backend for test results
            trend_analyzer: Trend analyzer instance
        """
        self.storage = storage or TestResultStorage()
        self.trend_analyzer = trend_analyzer or TrendAnalyzer(self.storage)

    def generate_executive_summary(
        self, test_run: TestRun, include_trends: bool = True
    ) -> dict[str, Any]:
        """
        Generate executive summary for a test run.

        Args:
            test_run: Test run to summarize
            include_trends: Include trend analysis (7, 30, 90 day windows)

        Returns:
            Executive summary dictionary
        """
        # Basic metrics
        summary = {
            "run_id": test_run.run_id,
            "timestamp": test_run.timestamp,
            "health_status": self.trend_analyzer.calculate_health_status(test_run),
            "key_metrics": {
                "total_tests": test_run.total_tests,
                "success_rate": test_run.success_rate,
                "passed_tests": test_run.passed_tests,
                "failed_tests": test_run.failed_tests,
                "skipped_tests": test_run.skipped_tests,
                "error_tests": test_run.error_tests,
            },
        }

        # Coverage metrics
        if test_run.coverage:
            summary["key_metrics"]["line_coverage"] = (
                test_run.coverage.line_coverage_percent
            )
            summary["key_metrics"]["lines_covered"] = test_run.coverage.lines_covered
            summary["key_metrics"]["lines_total"] = test_run.coverage.lines_total

        # Calculate total execution time
        total_duration_ms = 0
        for suite in test_run.suites:
            for case in suite.test_cases:
                for result in case.results:
                    total_duration_ms += result.duration_ms

        summary["key_metrics"]["total_execution_time_ms"] = total_duration_ms
        summary["key_metrics"]["total_execution_time_seconds"] = (
            total_duration_ms / 1000
        )

        # Trend analysis
        if include_trends:
            trends = self._analyze_all_trends()
            summary["trends"] = trends

            # Calculate trend indicators for key metrics
            summary["trend_indicators"] = self._generate_trend_indicators(trends)

            # Generate action items
            summary["action_items"] = self.trend_analyzer.get_action_items(
                summary["health_status"], trends
            )

        return summary

    def generate_dashboard_charts(
        self, test_run: TestRun, time_windows: list[int] | None = None
    ) -> dict[str, Any]:
        """
        Generate all dashboard charts.

        Args:
            test_run: Test run to generate charts for
            time_windows: List of time windows in days (default: [7, 30, 90])

        Returns:
            Dictionary of Chart.js chart configurations
        """
        if time_windows is None:
            time_windows = [7, 30, 90]

        charts = {}

        # Health status gauge
        charts["health_status"] = self._generate_health_gauge(test_run)

        # Success rate trend charts for multiple time windows
        for days in time_windows:
            trend_data = self.trend_analyzer.analyze_success_rate_trend(days=days)
            if not trend_data.get("insufficient_data"):
                charts[f"success_rate_trend_{days}d"] = (
                    self._generate_success_rate_trend_chart(trend_data, days)
                )

        # Execution time trend charts
        for days in time_windows:
            trend_data = self.trend_analyzer.analyze_execution_time_trend(days=days)
            if not trend_data.get("insufficient_data"):
                charts[f"execution_time_trend_{days}d"] = (
                    self._generate_execution_time_trend_chart(trend_data, days)
                )

        # Coverage trend charts
        for days in time_windows:
            trend_data = self.trend_analyzer.analyze_coverage_trend(days=days)
            if not trend_data.get("insufficient_data"):
                charts[f"coverage_trend_{days}d"] = (
                    self._generate_coverage_trend_chart(trend_data, days)
                )

        # Flakiness trend chart (30 days)
        flakiness_trend = self.trend_analyzer.analyze_flakiness_trend(days=30)
        if not flakiness_trend.get("insufficient_data"):
            charts["flakiness_trend"] = self._generate_flakiness_trend_chart(
                flakiness_trend
            )

        # Benchmark performance trends (if benchmarks exist)
        benchmark_charts = self._generate_benchmark_trend_charts(test_run, days=30)
        charts.update(benchmark_charts)

        return charts

    def generate_health_indicators(
        self, test_run: TestRun
    ) -> dict[str, dict[str, Any]]:
        """
        Generate health indicators for dashboard.

        Args:
            test_run: Test run to generate indicators for

        Returns:
            Dictionary of health indicators with status, value, and trend
        """
        indicators = {}

        # Success rate indicator
        success_rate_trend = self.trend_analyzer.analyze_success_rate_trend(days=7)
        indicators["success_rate"] = {
            "value": test_run.success_rate,
            "status": self._get_success_rate_status(test_run.success_rate),
            "trend": success_rate_trend.get("trend_direction", TrendDirection.STABLE),
            "label": "Success Rate",
            "unit": "%",
        }

        # Coverage indicator
        if test_run.coverage:
            coverage_trend = self.trend_analyzer.analyze_coverage_trend(days=7)
            indicators["coverage"] = {
                "value": test_run.coverage.line_coverage_percent,
                "status": self._get_coverage_status(
                    test_run.coverage.line_coverage_percent
                ),
                "trend": coverage_trend.get("trend_direction", TrendDirection.STABLE),
                "label": "Line Coverage",
                "unit": "%",
            }

        # Failed tests indicator
        indicators["failed_tests"] = {
            "value": test_run.failed_tests,
            "status": self._get_failed_tests_status(test_run.failed_tests),
            "trend": TrendDirection.STABLE,  # Could enhance with historical comparison
            "label": "Failed Tests",
            "unit": "",
        }

        # Flakiness indicator
        flakiness_trend = self.trend_analyzer.analyze_flakiness_trend(days=30)
        if not flakiness_trend.get("insufficient_data"):
            flaky_count = flakiness_trend.get("statistics", {}).get("current", 0)
            indicators["flakiness"] = {
                "value": flaky_count,
                "status": self._get_flakiness_status(flaky_count),
                "trend": flakiness_trend.get("trend_direction", TrendDirection.STABLE),
                "label": "Flaky Tests",
                "unit": "",
            }

        return indicators

    def _analyze_all_trends(self) -> dict[str, Any]:
        """Analyze all trends (7, 30, 90 days)."""
        trends = {}

        # Success rate trends
        trends["success_rate_7d"] = self.trend_analyzer.analyze_success_rate_trend(
            days=7
        )
        trends["success_rate_30d"] = self.trend_analyzer.analyze_success_rate_trend(
            days=30
        )
        trends["success_rate_90d"] = self.trend_analyzer.analyze_success_rate_trend(
            days=90
        )

        # Execution time trends
        trends["execution_time_7d"] = (
            self.trend_analyzer.analyze_execution_time_trend(days=7)
        )
        trends["execution_time_30d"] = (
            self.trend_analyzer.analyze_execution_time_trend(days=30)
        )

        # Coverage trends
        trends["coverage_7d"] = self.trend_analyzer.analyze_coverage_trend(days=7)
        trends["coverage_30d"] = self.trend_analyzer.analyze_coverage_trend(days=30)

        # Flakiness trends
        trends["flakiness_30d"] = self.trend_analyzer.analyze_flakiness_trend(days=30)

        return trends

    def _generate_trend_indicators(self, trends: dict[str, Any]) -> dict[str, str]:
        """
        Generate simple trend indicators for key metrics.

        Args:
            trends: Dictionary of trend analyses

        Returns:
            Dictionary mapping metric names to trend arrows/symbols
        """
        indicators = {}

        # Success rate (7 day)
        sr_trend = trends.get("success_rate_7d", {}).get(
            "trend_direction", TrendDirection.STABLE
        )
        indicators["success_rate"] = self._get_trend_arrow(sr_trend)

        # Coverage (7 day)
        cov_trend = trends.get("coverage_7d", {}).get(
            "trend_direction", TrendDirection.STABLE
        )
        indicators["coverage"] = self._get_trend_arrow(cov_trend)

        # Execution time (7 day)
        exec_trend = trends.get("execution_time_7d", {}).get(
            "trend_direction", TrendDirection.STABLE
        )
        indicators["execution_time"] = self._get_trend_arrow(exec_trend)

        # Flakiness (30 day)
        flake_trend = trends.get("flakiness_30d", {}).get(
            "trend_direction", TrendDirection.STABLE
        )
        indicators["flakiness"] = self._get_trend_arrow(flake_trend)

        return indicators

    def _generate_health_gauge(self, test_run: TestRun) -> dict[str, Any]:
        """Generate health status gauge chart."""
        health_status = self.trend_analyzer.calculate_health_status(test_run)

        # Map health status to colors and values
        status_config = {
            HealthStatus.EXCELLENT: {"color": "#10b981", "value": 100, "label": "Excellent"},
            HealthStatus.GOOD: {"color": "#84cc16", "value": 75, "label": "Good"},
            HealthStatus.WARNING: {"color": "#f59e0b", "value": 50, "label": "Warning"},
            HealthStatus.CRITICAL: {"color": "#ef4444", "value": 25, "label": "Critical"},
        }

        config = status_config[health_status]

        return {
            "type": "doughnut",
            "data": {
                "labels": ["Health", ""],
                "datasets": [
                    {
                        "data": [config["value"], 100 - config["value"]],
                        "backgroundColor": [config["color"], "#e5e7eb"],
                        "borderWidth": 0,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "circumference": 180,
                "rotation": 270,
                "cutout": "75%",
                "plugins": {
                    "legend": {"display": False},
                    "title": {
                        "display": True,
                        "text": f"Overall Health: {config['label']}",
                        "font": {"size": 16, "weight": "bold"},
                    },
                    "tooltip": {"enabled": False},
                },
            },
        }

    def _generate_success_rate_trend_chart(
        self, trend_data: dict[str, Any], days: int
    ) -> dict[str, Any]:
        """Generate success rate trend line chart."""
        data_points = trend_data.get("data_points", [])

        # Extract timestamps and values
        labels = [
            datetime.fromisoformat(p["timestamp"]).strftime("%m/%d %H:%M")
            for p in data_points
        ]
        values = [p["success_rate"] for p in data_points]

        return {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Success Rate (%)",
                        "data": values,
                        "borderColor": "#3b82f6",
                        "backgroundColor": "rgba(59, 130, 246, 0.1)",
                        "tension": 0.4,
                        "fill": True,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "interaction": {"intersect": False, "mode": "index"},
                "scales": {"y": {"beginAtZero": True, "max": 100}},
                "plugins": {
                    "legend": {"display": False},
                    "title": {
                        "display": True,
                        "text": f"Success Rate Trend (Last {days} Days)",
                    },
                },
            },
        }

    def _generate_execution_time_trend_chart(
        self, trend_data: dict[str, Any], days: int
    ) -> dict[str, Any]:
        """Generate execution time trend area chart."""
        data_points = trend_data.get("data_points", [])

        labels = [
            datetime.fromisoformat(p["timestamp"]).strftime("%m/%d %H:%M")
            for p in data_points
        ]
        values = [p["total_duration_seconds"] for p in data_points]

        return {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Execution Time (seconds)",
                        "data": values,
                        "borderColor": "#8b5cf6",
                        "backgroundColor": "rgba(139, 92, 246, 0.1)",
                        "tension": 0.4,
                        "fill": True,
                    }
                ],
            },
            "options": {
                "responsive": True,
                "interaction": {"intersect": False, "mode": "index"},
                "scales": {"y": {"beginAtZero": True}},
                "plugins": {
                    "legend": {"display": False},
                    "title": {
                        "display": True,
                        "text": f"Test Execution Time Trend (Last {days} Days)",
                    },
                },
            },
        }

    def _generate_coverage_trend_chart(
        self, trend_data: dict[str, Any], days: int
    ) -> dict[str, Any]:
        """Generate coverage trend chart."""
        data_points = trend_data.get("data_points", [])

        labels = [
            datetime.fromisoformat(p["timestamp"]).strftime("%m/%d %H:%M")
            for p in data_points
        ]
        line_coverage = [p["line_coverage"] for p in data_points]

        # Check if function/branch coverage is available
        has_function = any(p["function_coverage"] is not None for p in data_points)
        has_branch = any(p["branch_coverage"] is not None for p in data_points)

        datasets = [
            {
                "label": "Line Coverage (%)",
                "data": line_coverage,
                "borderColor": "#10b981",
                "backgroundColor": "rgba(16, 185, 129, 0.1)",
                "tension": 0.4,
                "fill": True,
            }
        ]

        if has_function:
            function_coverage = [
                p["function_coverage"] if p["function_coverage"] is not None else 0
                for p in data_points
            ]
            datasets.append(
                {
                    "label": "Function Coverage (%)",
                    "data": function_coverage,
                    "borderColor": "#3b82f6",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                    "tension": 0.4,
                    "fill": True,
                }
            )

        if has_branch:
            branch_coverage = [
                p["branch_coverage"] if p["branch_coverage"] is not None else 0
                for p in data_points
            ]
            datasets.append(
                {
                    "label": "Branch Coverage (%)",
                    "data": branch_coverage,
                    "borderColor": "#f59e0b",
                    "backgroundColor": "rgba(245, 158, 11, 0.1)",
                    "tension": 0.4,
                    "fill": True,
                }
            )

        return {
            "type": "line",
            "data": {"labels": labels, "datasets": datasets},
            "options": {
                "responsive": True,
                "interaction": {"intersect": False, "mode": "index"},
                "scales": {"y": {"beginAtZero": True, "max": 100}},
                "plugins": {
                    "legend": {"position": "bottom"},
                    "title": {
                        "display": True,
                        "text": f"Coverage Trend (Last {days} Days)",
                    },
                },
            },
        }

    def _generate_flakiness_trend_chart(
        self, trend_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate flakiness trend chart."""
        data_points = trend_data.get("data_points", [])

        labels = [
            datetime.fromisoformat(p["timestamp"]).strftime("%m/%d %H:%M")
            for p in data_points
        ]
        flaky_counts = [p["total_flaky_tests"] for p in data_points]
        pattern_counts = [p["total_failure_patterns"] for p in data_points]

        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [
                    {
                        "label": "Flaky Tests",
                        "data": flaky_counts,
                        "backgroundColor": "#ef4444",
                    },
                    {
                        "label": "Failure Patterns",
                        "data": pattern_counts,
                        "backgroundColor": "#f59e0b",
                    },
                ],
            },
            "options": {
                "responsive": True,
                "interaction": {"intersect": False, "mode": "index"},
                "scales": {"y": {"beginAtZero": True}},
                "plugins": {
                    "legend": {"position": "bottom"},
                    "title": {
                        "display": True,
                        "text": "Flakiness Trend (Last 30 Days)",
                    },
                },
            },
        }

    def _generate_benchmark_trend_charts(
        self, test_run: TestRun, days: int = 30
    ) -> dict[str, Any]:
        """
        Generate benchmark performance trend charts.

        Args:
            test_run: Test run containing benchmarks
            days: Number of days to analyze

        Returns:
            Dictionary of benchmark trend charts
        """
        charts = {}

        # Find all benchmarks in the test run
        benchmark_names = set()
        for suite in test_run.suites:
            for case in suite.test_cases:
                for result in case.results:
                    if result.performance:
                        benchmark_names.add(case.name)

        # Generate trend chart for each benchmark (limit to top 5)
        for idx, benchmark_name in enumerate(sorted(benchmark_names)[:5]):
            trend_data = self.trend_analyzer.analyze_benchmark_performance_trend(
                benchmark_name=benchmark_name, metric="avg_ms", days=days
            )

            if not trend_data.get("insufficient_data"):
                data_points = trend_data.get("data_points", [])
                labels = [
                    datetime.fromisoformat(p["timestamp"]).strftime("%m/%d %H:%M")
                    for p in data_points
                ]
                values = [p["value"] for p in data_points]

                chart_id = f"benchmark_{idx}_{benchmark_name.replace(' ', '_')[:30]}"
                charts[chart_id] = {
                    "type": "line",
                    "data": {
                        "labels": labels,
                        "datasets": [
                            {
                                "label": f"{benchmark_name} (ms)",
                                "data": values,
                                "borderColor": "#6366f1",
                                "backgroundColor": "rgba(99, 102, 241, 0.1)",
                                "tension": 0.4,
                                "fill": True,
                            }
                        ],
                    },
                    "options": {
                        "responsive": True,
                        "interaction": {"intersect": False, "mode": "index"},
                        "scales": {"y": {"beginAtZero": True}},
                        "plugins": {
                            "legend": {"display": False},
                            "title": {
                                "display": True,
                                "text": f"{benchmark_name} Performance Trend",
                            },
                        },
                    },
                }

        return charts

    @staticmethod
    def _get_success_rate_status(success_rate: float) -> str:
        """Get status color for success rate."""
        if success_rate >= 95:
            return "excellent"
        elif success_rate >= 90:
            return "good"
        elif success_rate >= 75:
            return "warning"
        else:
            return "critical"

    @staticmethod
    def _get_coverage_status(coverage: float) -> str:
        """Get status color for coverage."""
        if coverage >= 80:
            return "excellent"
        elif coverage >= 70:
            return "good"
        elif coverage >= 60:
            return "warning"
        else:
            return "critical"

    @staticmethod
    def _get_failed_tests_status(failed_count: int) -> str:
        """Get status color for failed test count."""
        if failed_count == 0:
            return "excellent"
        elif failed_count <= 2:
            return "good"
        elif failed_count <= 5:
            return "warning"
        else:
            return "critical"

    @staticmethod
    def _get_flakiness_status(flaky_count: int) -> str:
        """Get status color for flaky test count."""
        if flaky_count == 0:
            return "excellent"
        elif flaky_count <= 1:
            return "good"
        elif flaky_count <= 3:
            return "warning"
        else:
            return "critical"

    @staticmethod
    def _get_trend_arrow(trend_direction: TrendDirection) -> str:
        """Get arrow symbol for trend direction."""
        if trend_direction == TrendDirection.IMPROVING:
            return "↑"
        elif trend_direction == TrendDirection.DECLINING:
            return "↓"
        else:
            return "→"
