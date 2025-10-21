"""
Unit tests for test report generator.

Tests HTML and PDF report generation functionality.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from tests.reporting.models import (
    PerformanceMetrics,
    TestCase,
    TestResult,
    TestRun,
    TestSource,
    TestStatus,
    TestSuite,
    TestType,
)
from tests.reporting.report_generator import ReportGenerator, generate_test_report
from tests.reporting.storage import TestResultStorage


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary test storage."""
    db_path = tmp_path / "test_reports.db"
    storage = TestResultStorage(db_path)
    return storage


@pytest.fixture
def sample_test_run():
    """Create a sample test run with various test results."""
    # Create test results
    passed_result = TestResult(
        test_id="test-1",
        name="test_passing_case",
        status=TestStatus.PASSED,
        duration_ms=150.5,
        timestamp=datetime.now(),
    )

    failed_result = TestResult(
        test_id="test-2",
        name="test_failing_case",
        status=TestStatus.FAILED,
        duration_ms=250.3,
        timestamp=datetime.now(),
        error_message="AssertionError: Expected 5 but got 3",
        error_traceback="Traceback (most recent call last):\n  ...",
    )

    slow_result = TestResult(
        test_id="test-3",
        name="test_slow_operation",
        status=TestStatus.PASSED,
        duration_ms=2500.0,
        timestamp=datetime.now(),
    )

    benchmark_result = TestResult(
        test_id="bench-1",
        name="benchmark_search_performance",
        status=TestStatus.PASSED,
        duration_ms=50.0,
        timestamp=datetime.now(),
        performance=PerformanceMetrics(
            min_ms=45.2,
            max_ms=55.8,
            avg_ms=50.1,
            median_ms=49.9,
            p95_ms=54.5,
            p99_ms=55.5,
            operations_per_second=20.0,
            memory_mb=128.5,
            cpu_percent=45.2,
        ),
    )

    # Create test cases
    passing_case = TestCase(
        case_id="case-1",
        name="test_passing_case",
        file_path="tests/unit/test_example.py",
        line_number=42,
        test_type=TestType.UNIT,
        markers=["unit"],
        results=[passed_result],
    )

    failing_case = TestCase(
        case_id="case-2",
        name="test_failing_case",
        file_path="tests/unit/test_example.py",
        line_number=58,
        test_type=TestType.UNIT,
        markers=["unit"],
        results=[failed_result],
    )

    slow_case = TestCase(
        case_id="case-3",
        name="test_slow_operation",
        file_path="tests/integration/test_slow.py",
        line_number=15,
        test_type=TestType.INTEGRATION,
        markers=["integration", "slow"],
        results=[slow_result],
    )

    benchmark_case = TestCase(
        case_id="bench-case-1",
        name="benchmark_search_performance",
        file_path="tests/benchmarks/test_search.py",
        line_number=25,
        test_type=TestType.BENCHMARK,
        markers=["benchmark"],
        results=[benchmark_result],
    )

    # Create test suites
    unit_suite = TestSuite(
        suite_id="suite-unit",
        name="Unit Tests",
        test_type=TestType.UNIT,
        test_cases=[passing_case, failing_case],
    )

    integration_suite = TestSuite(
        suite_id="suite-integration",
        name="Integration Tests",
        test_type=TestType.INTEGRATION,
        test_cases=[slow_case],
    )

    benchmark_suite = TestSuite(
        suite_id="suite-benchmark",
        name="Benchmarks",
        test_type=TestType.BENCHMARK,
        test_cases=[benchmark_case],
    )

    # Create test run
    test_run = TestRun.create(
        source=TestSource.PYTEST,
        run_id="test-run-001",
        metadata={
            "branch": "main",
            "commit": "abc123",
            "ci_pipeline": "github-actions",
        },
        environment={
            "python_version": "3.10.12",
            "os": "Linux",
            "platform": "ubuntu-22.04",
        },
    )

    test_run.add_suite(unit_suite)
    test_run.add_suite(integration_suite)
    test_run.add_suite(benchmark_suite)

    return test_run


class TestReportGenerator:
    """Test ReportGenerator class."""

    def test_initialization(self, temp_storage):
        """Test report generator initialization."""
        generator = ReportGenerator(storage=temp_storage)
        assert generator.storage == temp_storage
        assert generator.env is not None
        assert generator.template_dir.exists()

    def test_custom_template_dir(self, temp_storage, tmp_path):
        """Test initialization with custom template directory."""
        custom_dir = tmp_path / "custom_templates"
        custom_dir.mkdir()
        generator = ReportGenerator(storage=temp_storage, template_dir=custom_dir)
        assert generator.template_dir == custom_dir

    def test_format_duration_filter(self):
        """Test duration formatting filter."""
        generator = ReportGenerator()

        assert generator._format_duration(50.5) == "50.50ms"
        assert generator._format_duration(1500.0) == "1.50s"
        assert generator._format_duration(75000.0) == "1m 15.00s"
        assert generator._format_duration(125000.0) == "2m 5.00s"

    def test_format_percentage_filter(self):
        """Test percentage formatting filter."""
        generator = ReportGenerator()

        assert generator._format_percentage(95.5) == "95.50%"
        assert generator._format_percentage(100.0) == "100.00%"
        assert generator._format_percentage(0.0) == "0.00%"

    def test_format_timestamp_filter(self):
        """Test timestamp formatting filter."""
        generator = ReportGenerator()

        dt = datetime(2024, 10, 21, 14, 30, 45)
        assert generator._format_timestamp(dt) == "2024-10-21 14:30:45"

    def test_generate_html_report(self, temp_storage, sample_test_run, tmp_path):
        """Test HTML report generation."""
        # Save test run to storage
        temp_storage.save_test_run(sample_test_run)

        # Generate report
        generator = ReportGenerator(storage=temp_storage)
        output_path = tmp_path / "test_report.html"

        html = generator.generate_html_report(
            run_id=sample_test_run.run_id, output_path=output_path
        )

        # Verify HTML was generated
        assert html is not None
        assert len(html) > 0
        assert "Test Report" in html
        assert sample_test_run.run_id in html

        # Verify file was written
        assert output_path.exists()
        content = output_path.read_text()
        assert content == html

    def test_generate_html_report_without_file(self, temp_storage, sample_test_run):
        """Test HTML report generation without saving to file."""
        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        html = generator.generate_html_report(run_id=sample_test_run.run_id)

        assert html is not None
        assert "Test Report" in html

    def test_generate_html_report_with_charts(self, temp_storage, sample_test_run):
        """Test HTML report includes chart data."""
        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        html = generator.generate_html_report(
            run_id=sample_test_run.run_id, include_charts=True
        )

        assert "Chart.js" in html
        assert "statusChart" in html
        assert "suiteChart" in html

    def test_generate_html_report_without_charts(self, temp_storage, sample_test_run):
        """Test HTML report without charts."""
        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        html = generator.generate_html_report(
            run_id=sample_test_run.run_id, include_charts=False, include_trends=False
        )

        assert "Chart.js" not in html

    def test_generate_html_report_with_trends(self, temp_storage, sample_test_run):
        """Test HTML report includes trend data."""
        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        html = generator.generate_html_report(
            run_id=sample_test_run.run_id, include_trends=True
        )

        # Trend section should be present (even if empty due to single run)
        assert "include_trends" in html or "Trend Analysis" in html

    def test_generate_html_report_invalid_run_id(self, temp_storage):
        """Test error handling for invalid run ID."""
        generator = ReportGenerator(storage=temp_storage)

        with pytest.raises(ValueError, match="Test run not found"):
            generator.generate_html_report(run_id="nonexistent-run")

    def test_generate_html_report_custom_context(
        self, temp_storage, sample_test_run, tmp_path
    ):
        """Test HTML report with custom context variables."""
        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        html = generator.generate_html_report(
            run_id=sample_test_run.run_id,
            custom_context={"custom_var": "custom_value"},
        )

        # Custom context should be available to template
        assert html is not None

    def test_generate_pdf_report(self, temp_storage, sample_test_run, tmp_path):
        """Test PDF report generation."""
        pytest.importorskip("weasyprint", reason="weasyprint required for PDF generation")

        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        output_path = tmp_path / "test_report.pdf"

        result_path = generator.generate_pdf_report(
            run_id=sample_test_run.run_id, output_path=output_path
        )

        # Verify PDF was created
        assert result_path == output_path
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify it's a PDF file (starts with PDF magic bytes)
        with open(output_path, "rb") as f:
            header = f.read(4)
            assert header == b"%PDF"

    def test_generate_pdf_report_without_weasyprint(
        self, temp_storage, sample_test_run, tmp_path, monkeypatch
    ):
        """Test PDF generation error when weasyprint not available."""

        def mock_import_error(*args, **kwargs):
            raise ImportError("No module named 'weasyprint'")

        # Mock weasyprint import to fail
        import builtins

        original_import = builtins.__import__

        def custom_import(name, *args, **kwargs):
            if name == "weasyprint":
                raise ImportError("No module named 'weasyprint'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", custom_import)

        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        output_path = tmp_path / "test_report.pdf"

        with pytest.raises(ImportError, match="weasyprint is required"):
            generator.generate_pdf_report(
                run_id=sample_test_run.run_id, output_path=output_path
            )

    def test_chart_data_generation(self, temp_storage, sample_test_run):
        """Test chart data generation."""
        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        chart_data = generator._generate_chart_data(sample_test_run)

        # Verify chart data structure
        assert "status_distribution" in chart_data
        assert "suite_breakdown" in chart_data

        # Verify status distribution chart
        status_chart = chart_data["status_distribution"]
        assert status_chart["type"] == "pie"
        assert "data" in status_chart
        assert "labels" in status_chart["data"]
        assert status_chart["data"]["labels"] == ["Passed", "Failed", "Skipped", "Error"]

        # Verify suite breakdown chart
        suite_chart = chart_data["suite_breakdown"]
        assert suite_chart["type"] == "bar"
        assert "data" in suite_chart

    def test_trend_data_generation(self, temp_storage, sample_test_run):
        """Test trend data generation."""
        temp_storage.save_test_run(sample_test_run)

        generator = ReportGenerator(storage=temp_storage)
        trend_data = generator._generate_trend_data(days=30)

        # Verify trend data structure
        assert "success_rate_trend" in trend_data
        trend_chart = trend_data["success_rate_trend"]
        assert trend_chart["type"] == "line"
        assert "data" in trend_chart


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_generate_test_report_html(self, temp_storage, sample_test_run, tmp_path):
        """Test convenience function for HTML generation."""
        temp_storage.save_test_run(sample_test_run)

        output_path = tmp_path / "report.html"
        result = generate_test_report(
            run_id=sample_test_run.run_id,
            output_path=output_path,
            format="html",
            storage_path=temp_storage.db_path,
        )

        assert result == output_path
        assert output_path.exists()

    def test_generate_test_report_pdf(self, temp_storage, sample_test_run, tmp_path):
        """Test convenience function for PDF generation."""
        pytest.importorskip("weasyprint", reason="weasyprint required for PDF generation")

        temp_storage.save_test_run(sample_test_run)

        output_path = tmp_path / "report.pdf"
        result = generate_test_report(
            run_id=sample_test_run.run_id,
            output_path=output_path,
            format="pdf",
            storage_path=temp_storage.db_path,
        )

        assert result == output_path
        assert output_path.exists()
