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

from .models import TestRun
from .query import TestResultQuery
from .storage import TestResultStorage


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
            test_run, include_charts=include_charts, include_trends=include_trends
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

    def _build_report_context(
        self, test_run: TestRun, include_charts: bool = True, include_trends: bool = True
    ) -> Dict[str, Any]:
        """
        Build template context from test run data.

        Args:
            test_run: Test run to generate context for
            include_charts: Include chart data
            include_trends: Include trend data

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
        }

        # Add chart data if requested
        if include_charts:
            context["chart_data"] = self._generate_chart_data(test_run)

        # Add trend data if requested
        if include_trends:
            context["trend_data"] = self._generate_trend_data()

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

        return {
            "status_distribution": status_chart,
            "suite_breakdown": suite_chart,
        }

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
