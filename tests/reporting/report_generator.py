"""
HTML and PDF test report generator.

Generates professional, customizable test reports from aggregated test results.
Supports HTML output with interactive charts and PDF export.
"""

import base64
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .failure_analyzer import FailureAnalyzer
from .models import TestRun
from .performance_dashboard import PerformanceDashboard
from .query import TestResultQuery
from .storage import TestResultStorage
from .trend_analyzer import TrendAnalyzer


class ReportGenerator:
    """
    Generate HTML and PDF test reports from aggregated test results.

    Features:
    - Executive summary with pass/fail stats
    - Test suite breakdown with expandable details
    - Failed test details with error messages
    - Performance benchmarks with charts
    - Test execution timeline
    - Trend analysis graphs
    """

    def __init__(
        self,
        storage: Optional[TestResultStorage] = None,
        template_dir: Optional[Path] = None,
    ):
        """
        Initialize report generator.

        Args:
            storage: Storage backend for test results
            template_dir: Custom template directory (uses default if None)
        """
        self.storage = storage or TestResultStorage()
        self.query = TestResultQuery(self.storage)
        self.failure_analyzer = FailureAnalyzer(self.storage)
        self.trend_analyzer = TrendAnalyzer(self.storage)
        self.dashboard = PerformanceDashboard(self.storage, self.trend_analyzer)

        # Set up Jinja2 environment
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )

        # Register custom filters
        self.env.filters["format_duration"] = self._format_duration
        self.env.filters["format_percentage"] = self._format_percentage
        self.env.filters["format_timestamp"] = self._format_timestamp

    def generate_html_report(
        self,
        run_id: str,
        output_path: Optional[Union[str, Path]] = None,
        include_charts: bool = True,
        include_trends: bool = True,
        include_failure_analysis: bool = True,
        template_name: str = "report.html",
        custom_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate HTML test report.

        Args:
            run_id: Test run ID to generate report for
            output_path: Path to save HTML file (returns HTML string if None)
            include_charts: Include performance charts
            include_trends: Include trend analysis
            include_failure_analysis: Include failure pattern analysis and flakiness detection
            template_name: Template file to use
            custom_context: Additional template context variables

        Returns:
            HTML content as string

        Raises:
            ValueError: If test run not found
        """
        # Retrieve test run
        test_run = self.storage.get_test_run(run_id)
        if not test_run:
            raise ValueError(f"Test run not found: {run_id}")

        # Gather report data
        context = self._build_report_context(
            test_run,
            include_charts=include_charts,
            include_trends=include_trends,
            include_failure_analysis=include_failure_analysis,
        )

        # Add custom context
        if custom_context:
            context.update(custom_context)

        # Render template
        template = self.env.get_template(template_name)
        html_content = template.render(**context)

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content, encoding="utf-8")

        return html_content

    def generate_pdf_report(
        self,
        run_id: str,
        output_path: Union[str, Path],
        include_charts: bool = True,
        include_trends: bool = True,
        include_failure_analysis: bool = True,
        template_name: str = "report.html",
        custom_context: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Generate PDF test report.

        Args:
            run_id: Test run ID to generate report for
            output_path: Path to save PDF file
            include_charts: Include performance charts
            include_trends: Include trend analysis
            include_failure_analysis: Include failure pattern analysis and flakiness detection
            template_name: Template file to use
            custom_context: Additional template context variables

        Returns:
            Path to generated PDF file

        Raises:
            ValueError: If test run not found
            ImportError: If weasyprint is not installed
        """
        try:
            from weasyprint import HTML
        except ImportError as e:
            raise ImportError(
                "weasyprint is required for PDF generation. "
                "Install with: pip install weasyprint"
            ) from e

        # Generate HTML content
        html_content = self.generate_html_report(
            run_id=run_id,
            output_path=None,
            include_charts=include_charts,
            include_trends=include_trends,
            include_failure_analysis=include_failure_analysis,
            template_name=template_name,
            custom_context=custom_context,
        )

        # Convert to PDF
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        HTML(string=html_content, base_url=str(self.template_dir)).write_pdf(
            str(output_path)
        )

        return output_path

    def generate_dashboard_report(
        self,
        run_id: str,
        output_path: Optional[Union[str, Path]] = None,
        time_windows: Optional[List[int]] = None,
        template_name: str = "dashboard.html",
        custom_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate executive dashboard HTML report.

        Args:
            run_id: Test run ID to generate dashboard for
            output_path: Path to save HTML file (returns HTML string if None)
            time_windows: List of time windows in days (default: [7, 30, 90])
            template_name: Template file to use
            custom_context: Additional template context variables

        Returns:
            HTML content as string

        Raises:
            ValueError: If test run not found
        """
        # Retrieve test run
        test_run = self.storage.get_test_run(run_id)
        if not test_run:
            raise ValueError(f"Test run not found: {run_id}")

        # Generate executive summary
        executive_summary = self.dashboard.generate_executive_summary(
            test_run, include_trends=True
        )

        # Generate health indicators
        health_indicators = self.dashboard.generate_health_indicators(test_run)

        # Generate dashboard charts
        dashboard_charts = self.dashboard.generate_dashboard_charts(
            test_run, time_windows=time_windows
        )

        # Build context
        context = {
            "run": test_run,
            "executive_summary": executive_summary,
            "health_indicators": health_indicators,
            "dashboard_charts": dashboard_charts,
            "generated_at": datetime.now(),
        }

        # Add custom context
        if custom_context:
            context.update(custom_context)

        # Render template (fallback to regular report if dashboard template doesn't exist)
        try:
            template = self.env.get_template(template_name)
        except Exception:
            # Dashboard template doesn't exist, use regular report with dashboard data
            template = self.env.get_template("report.html")
            # Add dashboard data to regular report context
            context.update(
                self._build_report_context(
                    test_run,
                    include_charts=True,
                    include_trends=True,
                    include_failure_analysis=True,
                )
            )

        html_content = template.render(**context)

        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(html_content, encoding="utf-8")

        return html_content

    def _build_report_context(
        self,
        test_run: TestRun,
        include_charts: bool = True,
        include_trends: bool = True,
        include_failure_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Build template context from test run data.

        Args:
            test_run: Test run to generate context for
            include_charts: Include chart data
            include_trends: Include trend data
            include_failure_analysis: Include failure pattern analysis

        Returns:
            Template context dictionary
        """
        # Basic summary
        run_summary = self.query.get_run_summary(test_run.run_id)

        # Failed tests
        failed_tests = self.query.get_failed_tests(test_run.run_id)

        # Slow tests
        slow_tests = self.query.get_slow_tests(test_run.run_id, threshold_ms=1000)

        # Benchmarks
        benchmarks = self.query.get_benchmark_results(test_run.run_id)

        # Build context
        context = {
            "run": test_run,
            "summary": run_summary,
            "failed_tests": failed_tests,
            "slow_tests": slow_tests,
            "benchmarks": benchmarks,
            "generated_at": datetime.now(),
            "include_charts": include_charts,
            "include_trends": include_trends,
            "include_failure_analysis": include_failure_analysis,
            "has_coverage": test_run.coverage is not None,
        }

        # Add coverage data if present
        if test_run.coverage:
            context["coverage"] = test_run.coverage
            context["coverage_summary"] = self._generate_coverage_summary(test_run.coverage)

        # Add chart data if requested
        if include_charts:
            context["chart_data"] = self._generate_chart_data(test_run)

        # Add trend data if requested
        if include_trends:
            context["trend_data"] = self._generate_trend_data()
            # Add executive summary and health indicators
            context["executive_summary"] = self.dashboard.generate_executive_summary(
                test_run, include_trends=True
            )
            context["health_indicators"] = self.dashboard.generate_health_indicators(
                test_run
            )
            # Add dashboard charts
            context["dashboard_charts"] = self.dashboard.generate_dashboard_charts(
                test_run, time_windows=[7, 30, 90]
            )

        # Add failure analysis if requested
        if include_failure_analysis:
            failure_report = self.failure_analyzer.analyze_test_runs(
                run_ids=[test_run.run_id],
                min_flakiness_score=5.0,
            )
            context["failure_analysis"] = failure_report
            context["has_failure_analysis"] = (
                failure_report.total_flaky_tests > 0
                or failure_report.total_failure_patterns > 0
            )

            # Save the failure analysis report
            self.storage.save_failure_analysis_report(failure_report)

            # Add failure analysis charts if charts enabled
            if include_charts:
                failure_charts = self._generate_failure_analysis_charts(failure_report)
                context["chart_data"].update(failure_charts)

        return context

    def _generate_chart_data(self, test_run: TestRun) -> Dict[str, Any]:
        """
        Generate data for charts.

        Args:
            test_run: Test run to generate charts for

        Returns:
            Chart data dictionary for Chart.js
        """
        # Status distribution pie chart
        status_chart = {
            "type": "pie",
            "data": {
                "labels": ["Passed", "Failed", "Skipped", "Error"],
                "datasets": [
                    {
                        "data": [
                            test_run.passed_tests,
                            test_run.failed_tests,
                            test_run.skipped_tests,
                            test_run.error_tests,
                        ],
                        "backgroundColor": [
                            "#10b981",  # green
                            "#ef4444",  # red
                            "#f59e0b",  # amber
                            "#8b5cf6",  # purple
                        ],
                    }
                ],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"position": "bottom"},
                    "title": {"display": True, "text": "Test Status Distribution"},
                },
            },
        }

        # Test suite breakdown
        suite_labels = []
        suite_passed = []
        suite_failed = []
        for suite in test_run.suites:
            suite_labels.append(suite.name)
            suite_passed.append(suite.passed_tests)
            suite_failed.append(suite.failed_tests)

        suite_chart = {
            "type": "bar",
            "data": {
                "labels": suite_labels,
                "datasets": [
                    {
                        "label": "Passed",
                        "data": suite_passed,
                        "backgroundColor": "#10b981",
                    },
                    {
                        "label": "Failed",
                        "data": suite_failed,
                        "backgroundColor": "#ef4444",
                    },
                ],
            },
            "options": {
                "responsive": True,
                "scales": {"y": {"beginAtZero": True}},
                "plugins": {
                    "legend": {"position": "bottom"},
                    "title": {"display": True, "text": "Results by Test Suite"},
                },
            },
        }

        charts = {
            "status_distribution": status_chart,
            "suite_breakdown": suite_chart,
        }

        # Add coverage charts if coverage data is present
        if test_run.coverage:
            charts.update(self._generate_coverage_charts(test_run.coverage))

        return charts

    def _generate_trend_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate trend analysis data.

        Args:
            days: Number of days to analyze

        Returns:
            Trend data dictionary
        """
        trend_data = self.query.get_trend_data(days=days, metric="success_rate")

        # Format for Chart.js
        labels = []
        values = []
        for point in trend_data:
            timestamp = datetime.fromisoformat(point["timestamp"])
            labels.append(timestamp.strftime("%Y-%m-%d %H:%M"))
            values.append(point["value"])

        trend_chart = {
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
                    }
                ],
            },
            "options": {
                "responsive": True,
                "scales": {"y": {"beginAtZero": True, "max": 100}},
                "plugins": {
                    "legend": {"position": "bottom"},
                    "title": {
                        "display": True,
                        "text": f"Success Rate Trend (Last {days} Days)",
                    },
                },
            },
        }

        return {"success_rate_trend": trend_chart}

    def _generate_coverage_summary(self, coverage) -> Dict[str, Any]:
        """
        Generate summary statistics for coverage.

        Args:
            coverage: CoverageMetrics object

        Returns:
            Coverage summary dictionary
        """
        summary = {
            "line_coverage": coverage.line_coverage_percent,
            "lines_covered": coverage.lines_covered,
            "lines_total": coverage.lines_total,
            "lines_uncovered": coverage.lines_total - coverage.lines_covered,
        }

        if coverage.function_coverage_percent is not None:
            summary["function_coverage"] = coverage.function_coverage_percent
            summary["functions_covered"] = coverage.functions_covered
            summary["functions_total"] = coverage.functions_total
            summary["functions_uncovered"] = coverage.functions_total - coverage.functions_covered

        if coverage.branch_coverage_percent is not None:
            summary["branch_coverage"] = coverage.branch_coverage_percent
            summary["branches_covered"] = coverage.branches_covered
            summary["branches_total"] = coverage.branches_total
            summary["branches_uncovered"] = coverage.branches_total - coverage.branches_covered

        # Find files with lowest coverage
        files_sorted = sorted(
            coverage.file_coverage,
            key=lambda f: f.line_coverage_percent
        )
        summary["lowest_coverage_files"] = [
            {
                "path": f.file_path,
                "coverage": f.line_coverage_percent,
                "uncovered_lines": len(f.uncovered_lines),
            }
            for f in files_sorted[:10]  # Top 10 files needing attention
        ]

        # Find files with 100% coverage
        summary["fully_covered_files"] = [
            f.file_path
            for f in coverage.file_coverage
            if f.line_coverage_percent == 100.0
        ]

        return summary

    def _generate_coverage_charts(self, coverage) -> Dict[str, Any]:
        """
        Generate charts for coverage visualization.

        Args:
            coverage: CoverageMetrics object

        Returns:
            Dictionary of Chart.js chart configurations
        """
        charts = {}

        # Overall coverage gauge chart
        coverage_gauge = {
            "type": "doughnut",
            "data": {
                "labels": ["Covered", "Uncovered"],
                "datasets": [
                    {
                        "data": [
                            coverage.lines_covered,
                            coverage.lines_total - coverage.lines_covered,
                        ],
                        "backgroundColor": ["#10b981", "#ef4444"],
                    }
                ],
            },
            "options": {
                "responsive": True,
                "plugins": {
                    "legend": {"position": "bottom"},
                    "title": {
                        "display": True,
                        "text": f"Line Coverage: {coverage.line_coverage_percent:.2f}%",
                    },
                },
            },
        }
        charts["coverage_gauge"] = coverage_gauge

        # Coverage type breakdown (if available)
        if coverage.function_coverage_percent is not None or coverage.branch_coverage_percent is not None:
            labels = ["Lines"]
            data = [coverage.line_coverage_percent]

            if coverage.function_coverage_percent is not None:
                labels.append("Functions")
                data.append(coverage.function_coverage_percent)

            if coverage.branch_coverage_percent is not None:
                labels.append("Branches")
                data.append(coverage.branch_coverage_percent)

            coverage_breakdown = {
                "type": "bar",
                "data": {
                    "labels": labels,
                    "datasets": [
                        {
                            "label": "Coverage %",
                            "data": data,
                            "backgroundColor": "#3b82f6",
                        }
                    ],
                },
                "options": {
                    "responsive": True,
                    "scales": {"y": {"beginAtZero": True, "max": 100}},
                    "plugins": {
                        "legend": {"display": False},
                        "title": {"display": True, "text": "Coverage by Type"},
                    },
                },
            }
            charts["coverage_breakdown"] = coverage_breakdown

        # Per-file coverage bar chart (top 20 files by lines)
        if coverage.file_coverage:
            files_sorted = sorted(
                coverage.file_coverage,
                key=lambda f: f.lines_total,
                reverse=True
            )[:20]

            file_labels = [f.file_path.split("/")[-1] for f in files_sorted]  # Just filename
            file_coverage_data = [f.line_coverage_percent for f in files_sorted]

            # Color based on coverage percentage
            colors = [
                "#10b981" if cov >= 80 else "#f59e0b" if cov >= 60 else "#ef4444"
                for cov in file_coverage_data
            ]

            file_coverage_chart = {
                "type": "bar",
                "data": {
                    "labels": file_labels,
                    "datasets": [
                        {
                            "label": "Coverage %",
                            "data": file_coverage_data,
                            "backgroundColor": colors,
                        }
                    ],
                },
                "options": {
                    "responsive": True,
                    "indexAxis": "y",  # Horizontal bar chart
                    "scales": {"x": {"beginAtZero": True, "max": 100}},
                    "plugins": {
                        "legend": {"display": False},
                        "title": {
                            "display": True,
                            "text": "Coverage by File (Top 20 by Size)",
                        },
                    },
                },
            }
            charts["file_coverage"] = file_coverage_chart

        return charts

    def _generate_failure_analysis_charts(self, failure_report) -> Dict[str, Any]:
        """
        Generate charts for failure analysis visualization.

        Args:
            failure_report: FailureAnalysisReport object

        Returns:
            Dictionary of Chart.js chart configurations
        """
        charts = {}

        # Failure category distribution pie chart
        if failure_report.category_distribution:
            category_labels = list(failure_report.category_distribution.keys())
            category_counts = list(failure_report.category_distribution.values())

            category_chart = {
                "type": "pie",
                "data": {
                    "labels": category_labels,
                    "datasets": [
                        {
                            "data": category_counts,
                            "backgroundColor": [
                                "#ef4444",  # red
                                "#f59e0b",  # amber
                                "#8b5cf6",  # purple
                                "#3b82f6",  # blue
                                "#10b981",  # green
                                "#6b7280",  # gray
                            ],
                        }
                    ],
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "legend": {"position": "bottom"},
                        "title": {"display": True, "text": "Failure Category Distribution"},
                    },
                },
            }
            charts["failure_categories"] = category_chart

        # Flaky tests bar chart (top 10)
        if failure_report.flaky_tests:
            top_flaky = failure_report.flaky_tests[:10]
            flaky_labels = [f.test_case_name.split("::")[-1] for f in top_flaky]  # Just test name
            flaky_scores = [f.flakiness_score for f in top_flaky]

            # Color based on severity
            colors = [
                "#ef4444" if score >= 40 else "#f59e0b" if score >= 20 else "#fbbf24"
                for score in flaky_scores
            ]

            flaky_chart = {
                "type": "bar",
                "data": {
                    "labels": flaky_labels,
                    "datasets": [
                        {
                            "label": "Flakiness Score",
                            "data": flaky_scores,
                            "backgroundColor": colors,
                        }
                    ],
                },
                "options": {
                    "responsive": True,
                    "indexAxis": "y",  # Horizontal bar chart
                    "scales": {"x": {"beginAtZero": True, "max": 100}},
                    "plugins": {
                        "legend": {"display": False},
                        "title": {
                            "display": True,
                            "text": "Top 10 Flaky Tests (by Flakiness Score)",
                        },
                    },
                },
            }
            charts["flaky_tests"] = flaky_chart

        # Failure patterns bar chart (top 10)
        if failure_report.failure_patterns:
            top_patterns = failure_report.failure_patterns[:10]
            pattern_labels = [
                p.error_signature[:50] + "..." if len(p.error_signature) > 50 else p.error_signature
                for p in top_patterns
            ]
            pattern_counts = [p.occurrences for p in top_patterns]

            # Color by category
            category_colors = {
                "assertion": "#ef4444",
                "timeout": "#f59e0b",
                "setup_teardown": "#8b5cf6",
                "external_dependency": "#3b82f6",
                "resource_exhaustion": "#10b981",
                "unknown": "#6b7280",
            }
            colors = [category_colors.get(p.category.value, "#6b7280") for p in top_patterns]

            pattern_chart = {
                "type": "bar",
                "data": {
                    "labels": pattern_labels,
                    "datasets": [
                        {
                            "label": "Occurrences",
                            "data": pattern_counts,
                            "backgroundColor": colors,
                        }
                    ],
                },
                "options": {
                    "responsive": True,
                    "indexAxis": "y",  # Horizontal bar chart
                    "scales": {"x": {"beginAtZero": True}},
                    "plugins": {
                        "legend": {"display": False},
                        "title": {
                            "display": True,
                            "text": "Top 10 Failure Patterns (by Occurrence)",
                        },
                    },
                },
            }
            charts["failure_patterns"] = pattern_chart

        return charts

    @staticmethod
    def _format_duration(ms: float) -> str:
        """Format duration in milliseconds to human-readable string."""
        if ms < 1000:
            return f"{ms:.2f}ms"
        elif ms < 60000:
            return f"{ms / 1000:.2f}s"
        else:
            minutes = int(ms / 60000)
            seconds = (ms % 60000) / 1000
            return f"{minutes}m {seconds:.2f}s"

    @staticmethod
    def _format_percentage(value: float) -> str:
        """Format percentage value."""
        return f"{value:.2f}%"

    @staticmethod
    def _format_timestamp(dt: datetime) -> str:
        """Format timestamp."""
        return dt.strftime("%Y-%m-%d %H:%M:%S")


# Convenience function for quick report generation
def generate_test_report(
    run_id: str,
    output_path: Union[str, Path],
    format: str = "html",
    storage_path: Optional[Path] = None,
) -> Path:
    """
    Quick helper to generate a test report.

    Args:
        run_id: Test run ID
        output_path: Path to save report
        format: Report format ('html' or 'pdf')
        storage_path: Optional path to SQLite database

    Returns:
        Path to generated report file

    Example:
        >>> from tests.reporting.report_generator import generate_test_report
        >>> report_path = generate_test_report('run-123', 'report.html', format='html')
        >>> print(f"Report generated: {report_path}")
    """
    storage = TestResultStorage(storage_path) if storage_path else TestResultStorage()
    generator = ReportGenerator(storage)

    if format.lower() == "pdf":
        return generator.generate_pdf_report(run_id, output_path)
    else:
        html = generator.generate_html_report(run_id, output_path)
        return Path(output_path)
