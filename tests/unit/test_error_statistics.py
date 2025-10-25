"""
Unit tests for error statistics and reporting system.

Tests comprehensive error statistics generation, trend analysis,
top errors detection, resolution metrics, and report formatting.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from common.core.error_categorization import ErrorCategory, ErrorSeverity
from common.core.error_message_manager import ErrorMessageManager
from common.core.error_statistics import (
    DetailedErrorStatistics,
    ErrorReportGenerator,
    ResolutionMetrics,
    ResolutionReport,
    SummaryReport,
    TimePeriodData,
    TopError,
    TopErrorsReport,
    TrendReport,
)


@pytest.fixture
async def test_db_path(tmp_path):
    """Create a temporary test database."""
    db_path = tmp_path / "test_error_stats.db"
    return str(db_path)


@pytest.fixture
async def error_manager(test_db_path):
    """Create and initialize error message manager."""
    manager = ErrorMessageManager(db_path=test_db_path)
    await manager.initialize()

    # Create messages table with enhanced schema
    async with manager.connection_pool.get_connection_async() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                context TEXT,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_at TEXT,
                acknowledged_by TEXT,
                retry_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()

    yield manager
    await manager.close()


@pytest.fixture
async def report_generator(error_manager):
    """Create error report generator."""
    generator = ErrorReportGenerator(error_manager=error_manager)
    await generator.initialize()
    yield generator
    await generator.close()


async def create_test_error(
    manager: ErrorMessageManager,
    message: str,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    timestamp: datetime = None,
    acknowledged: bool = False,
    acknowledged_by: str = None,
    context: dict = None
) -> int:
    """
    Helper to create a test error message.

    Args:
        manager: Error message manager
        message: Error message
        severity: Error severity
        category: Error category
        timestamp: Optional timestamp (defaults to now)
        acknowledged: Whether error is acknowledged
        acknowledged_by: Who acknowledged the error
        context: Optional context dictionary

    Returns:
        Error ID
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    # Insert error
    query = """
        INSERT INTO messages (
            timestamp, severity, category, message, context,
            acknowledged, acknowledged_at, acknowledged_by, retry_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    acknowledged_at = None
    if acknowledged and acknowledged_by:
        # Acknowledged 1 hour after error
        acknowledged_at = (timestamp + timedelta(hours=1)).isoformat()

    context_json = json.dumps(context) if context else None

    async with manager.connection_pool.get_connection_async() as conn:
        cursor = conn.execute(query, (
            timestamp.isoformat(),
            severity.value,
            category.value,
            message,
            context_json,
            1 if acknowledged else 0,
            acknowledged_at,
            acknowledged_by,
            0
        ))
        conn.commit()
        return cursor.lastrowid


class TestDetailedErrorStatistics:
    """Test DetailedErrorStatistics dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        now = datetime.now(timezone.utc)
        stats = DetailedErrorStatistics(
            total_count=100,
            by_severity={'error': 60, 'warning': 30, 'info': 10},
            by_category={'file_corrupt': 40, 'network': 30, 'unknown': 30},
            unacknowledged_count=20,
            last_error_at=now,
            time_range_start=now - timedelta(days=7),
            time_range_end=now,
            error_rate_per_hour=0.6,
            error_rate_per_day=14.3,
            top_errors=[]
        )

        result = stats.to_dict()

        assert result['total_count'] == 100
        assert result['by_severity'] == {'error': 60, 'warning': 30, 'info': 10}
        assert result['by_category'] == {'file_corrupt': 40, 'network': 30, 'unknown': 30}
        assert result['unacknowledged_count'] == 20
        assert result['error_rate_per_hour'] == 0.6
        assert result['error_rate_per_day'] == 14.3
        assert 'time_range_start' in result
        assert 'time_range_end' in result


class TestErrorReportGenerator:
    """Test ErrorReportGenerator class."""

    @pytest.mark.asyncio
    async def test_initialize_and_close(self, test_db_path):
        """Test generator initialization and cleanup."""
        generator = ErrorReportGenerator(db_path=test_db_path)

        assert not generator._initialized

        await generator.initialize()
        assert generator._initialized

        await generator.close()
        assert not generator._initialized

    @pytest.mark.asyncio
    async def test_generate_summary_report_empty(self, report_generator):
        """Test summary report generation with no errors."""
        report = await report_generator.generate_summary_report(days=7)

        assert isinstance(report, SummaryReport)
        assert report.time_range_days == 7
        assert report.statistics.total_count == 0
        assert report.statistics.unacknowledged_count == 0
        assert report.statistics.error_rate_per_hour == 0.0
        assert report.statistics.error_rate_per_day == 0.0
        assert len(report.statistics.top_errors) == 0

    @pytest.mark.asyncio
    async def test_generate_summary_report_with_errors(
        self,
        error_manager,
        report_generator
    ):
        """Test summary report with actual errors."""
        now = datetime.now(timezone.utc)

        # Create test errors
        await create_test_error(
            error_manager,
            "File not found",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.FILE_CORRUPT,
            timestamp=now - timedelta(hours=2)
        )
        await create_test_error(
            error_manager,
            "Connection timeout",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.TIMEOUT,
            timestamp=now - timedelta(hours=1)
        )
        await create_test_error(
            error_manager,
            "Permission denied",
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.PERMISSION_DENIED,
            timestamp=now - timedelta(minutes=30),
            acknowledged=True,
            acknowledged_by="admin"
        )

        # Generate report
        report = await report_generator.generate_summary_report(days=7)

        assert report.statistics.total_count == 3
        assert report.statistics.by_severity['error'] == 2
        assert report.statistics.by_severity['warning'] == 1
        assert report.statistics.by_category['file_corrupt'] == 1
        assert report.statistics.by_category['timeout'] == 1
        assert report.statistics.by_category['permission_denied'] == 1
        assert report.statistics.unacknowledged_count == 2
        assert report.statistics.error_rate_per_hour > 0
        assert report.statistics.error_rate_per_day > 0

    @pytest.mark.asyncio
    async def test_generate_summary_report_time_range(
        self,
        error_manager,
        report_generator
    ):
        """Test summary report respects time range."""
        now = datetime.now(timezone.utc)

        # Error within range
        await create_test_error(
            error_manager,
            "Recent error",
            timestamp=now - timedelta(days=3)
        )

        # Error outside range
        await create_test_error(
            error_manager,
            "Old error",
            timestamp=now - timedelta(days=10)
        )

        # Generate report for last 7 days
        report = await report_generator.generate_summary_report(days=7)

        # Should only include recent error
        assert report.statistics.total_count == 1

    @pytest.mark.asyncio
    async def test_generate_trend_report_daily(
        self,
        error_manager,
        report_generator
    ):
        """Test daily trend report generation."""
        now = datetime.now(timezone.utc)

        # Create errors across multiple days
        for i in range(5):
            day_offset = i
            await create_test_error(
                error_manager,
                f"Error day {i}",
                timestamp=now - timedelta(days=day_offset)
            )

        # Generate daily trend report
        report = await report_generator.generate_trend_report(
            days=7,
            granularity='daily'
        )

        assert isinstance(report, TrendReport)
        assert report.granularity == 'daily'
        assert report.time_range_days == 7
        assert len(report.time_series) > 0

        # Verify time series structure
        for period_data in report.time_series:
            assert isinstance(period_data, TimePeriodData)
            assert isinstance(period_data.period, str)
            assert isinstance(period_data.count, int)
            assert isinstance(period_data.by_severity, dict)
            assert isinstance(period_data.by_category, dict)

    @pytest.mark.asyncio
    async def test_generate_trend_report_hourly(
        self,
        error_manager,
        report_generator
    ):
        """Test hourly trend report generation."""
        now = datetime.now(timezone.utc)

        # Create errors across multiple hours
        for i in range(5):
            await create_test_error(
                error_manager,
                f"Error hour {i}",
                timestamp=now - timedelta(hours=i)
            )

        # Generate hourly trend report
        report = await report_generator.generate_trend_report(
            days=1,
            granularity='hourly'
        )

        assert report.granularity == 'hourly'
        assert len(report.time_series) == 24  # 24 hours in 1 day

    @pytest.mark.asyncio
    async def test_generate_trend_report_weekly(
        self,
        error_manager,
        report_generator
    ):
        """Test weekly trend report generation."""
        now = datetime.now(timezone.utc)

        # Create errors across multiple weeks
        for i in range(3):
            await create_test_error(
                error_manager,
                f"Error week {i}",
                timestamp=now - timedelta(weeks=i)
            )

        # Generate weekly trend report
        report = await report_generator.generate_trend_report(
            days=21,  # 3 weeks
            granularity='weekly'
        )

        assert report.granularity == 'weekly'
        assert len(report.time_series) == 3

    @pytest.mark.asyncio
    async def test_generate_top_errors_report(
        self,
        error_manager,
        report_generator
    ):
        """Test top errors report generation."""
        now = datetime.now(timezone.utc)

        # Create errors with varying frequencies
        error_counts = {
            "Connection timeout": 10,
            "File not found": 5,
            "Permission denied": 3,
            "Parse error": 1
        }

        for message, count in error_counts.items():
            for i in range(count):
                await create_test_error(
                    error_manager,
                    message,
                    timestamp=now - timedelta(minutes=i)
                )

        # Generate top errors report
        report = await report_generator.generate_top_errors_report(limit=3)

        assert isinstance(report, TopErrorsReport)
        assert report.limit == 3
        assert len(report.top_errors) == 3

        # Verify sorted by count descending
        assert report.top_errors[0].message == "Connection timeout"
        assert report.top_errors[0].count == 10
        assert report.top_errors[1].message == "File not found"
        assert report.top_errors[1].count == 5
        assert report.top_errors[2].message == "Permission denied"
        assert report.top_errors[2].count == 3

        # Verify TopError structure
        top_error = report.top_errors[0]
        assert isinstance(top_error, TopError)
        assert isinstance(top_error.first_seen, datetime)
        assert isinstance(top_error.last_seen, datetime)
        assert top_error.category
        assert top_error.severity

    @pytest.mark.asyncio
    async def test_generate_top_errors_with_time_range(
        self,
        error_manager,
        report_generator
    ):
        """Test top errors report with time range filter."""
        now = datetime.now(timezone.utc)

        # Recent errors
        for _i in range(5):
            await create_test_error(
                error_manager,
                "Recent error",
                timestamp=now - timedelta(days=2)
            )

        # Old errors (more frequent but outside range)
        for _i in range(10):
            await create_test_error(
                error_manager,
                "Old error",
                timestamp=now - timedelta(days=10)
            )

        # Generate report for last 7 days
        report = await report_generator.generate_top_errors_report(
            limit=10,
            days=7
        )

        # Should only include recent errors
        assert len(report.top_errors) == 1
        assert report.top_errors[0].message == "Recent error"
        assert report.top_errors[0].count == 5

    @pytest.mark.asyncio
    async def test_generate_resolution_report(
        self,
        error_manager,
        report_generator
    ):
        """Test resolution report generation."""
        now = datetime.now(timezone.utc)

        # Create mix of acknowledged and unacknowledged errors
        for i in range(7):
            await create_test_error(
                error_manager,
                f"Error {i}",
                acknowledged=(i < 5),  # First 5 acknowledged
                acknowledged_by="admin" if i < 5 else None,
                timestamp=now - timedelta(hours=i)
            )

        # Generate resolution report
        report = await report_generator.generate_resolution_report()

        assert isinstance(report, ResolutionReport)
        assert isinstance(report.metrics, ResolutionMetrics)
        assert report.metrics.total_errors == 7
        assert report.metrics.acknowledged_count == 5
        assert report.metrics.unacknowledged_count == 2
        assert report.metrics.acknowledgment_rate == pytest.approx(71.4, rel=0.1)
        assert report.metrics.avg_time_to_acknowledge is not None
        assert report.metrics.avg_time_to_acknowledge > 0

    @pytest.mark.asyncio
    async def test_generate_resolution_report_no_acknowledgments(
        self,
        error_manager,
        report_generator
    ):
        """Test resolution report with no acknowledgments."""
        now = datetime.now(timezone.utc)

        # Create unacknowledged errors
        for i in range(3):
            await create_test_error(
                error_manager,
                f"Error {i}",
                timestamp=now - timedelta(hours=i)
            )

        # Generate resolution report
        report = await report_generator.generate_resolution_report()

        assert report.metrics.total_errors == 3
        assert report.metrics.acknowledged_count == 0
        assert report.metrics.unacknowledged_count == 3
        assert report.metrics.acknowledgment_rate == 0.0
        assert report.metrics.avg_time_to_acknowledge is None

    @pytest.mark.asyncio
    async def test_generate_resolution_report_time_range(
        self,
        error_manager,
        report_generator
    ):
        """Test resolution report with time range."""
        now = datetime.now(timezone.utc)

        # Recent errors
        for i in range(3):
            await create_test_error(
                error_manager,
                f"Recent error {i}",
                timestamp=now - timedelta(days=2),
                acknowledged=True,
                acknowledged_by="admin"
            )

        # Old errors
        for i in range(5):
            await create_test_error(
                error_manager,
                f"Old error {i}",
                timestamp=now - timedelta(days=10)
            )

        # Generate report for last 7 days
        report = await report_generator.generate_resolution_report(days=7)

        # Should only include recent errors
        assert report.metrics.total_errors == 3
        assert report.time_range_days == 7


class TestReportExport:
    """Test report export functionality."""

    @pytest.mark.asyncio
    async def test_export_summary_json(
        self,
        error_manager,
        report_generator
    ):
        """Test JSON export of summary report."""
        # Create test error
        await create_test_error(error_manager, "Test error")

        report = await report_generator.generate_summary_report(days=7)
        json_output = report_generator.export_report(report, format='json')

        # Verify valid JSON
        data = json.loads(json_output)
        assert data['type'] == 'summary'
        assert 'statistics' in data
        assert 'time_range_days' in data
        assert 'generated_at' in data

    @pytest.mark.asyncio
    async def test_export_summary_markdown(
        self,
        error_manager,
        report_generator
    ):
        """Test Markdown export of summary report."""
        await create_test_error(error_manager, "Test error")

        report = await report_generator.generate_summary_report(days=7)
        markdown_output = report_generator.export_report(report, format='markdown')

        # Verify Markdown structure
        assert '# Error Summary Report' in markdown_output
        assert '## Overview' in markdown_output
        assert '## By Severity' in markdown_output
        assert '## By Category' in markdown_output
        assert 'Total Errors:' in markdown_output

    @pytest.mark.asyncio
    async def test_export_trend_json(
        self,
        error_manager,
        report_generator
    ):
        """Test JSON export of trend report."""
        await create_test_error(error_manager, "Test error")

        report = await report_generator.generate_trend_report(
            days=7,
            granularity='daily'
        )
        json_output = report_generator.export_report(report, format='json')

        data = json.loads(json_output)
        assert data['type'] == 'trend'
        assert 'time_series' in data
        assert 'granularity' in data
        assert data['granularity'] == 'daily'

    @pytest.mark.asyncio
    async def test_export_trend_markdown(
        self,
        error_manager,
        report_generator
    ):
        """Test Markdown export of trend report."""
        await create_test_error(error_manager, "Test error")

        report = await report_generator.generate_trend_report(
            days=7,
            granularity='daily'
        )
        markdown_output = report_generator.export_report(report, format='markdown')

        assert '# Error Trend Report' in markdown_output
        assert '## Time Series' in markdown_output
        assert '| Period | Total | Error | Warning | Info |' in markdown_output

    @pytest.mark.asyncio
    async def test_export_top_errors_json(
        self,
        error_manager,
        report_generator
    ):
        """Test JSON export of top errors report."""
        await create_test_error(error_manager, "Test error")

        report = await report_generator.generate_top_errors_report(limit=10)
        json_output = report_generator.export_report(report, format='json')

        data = json.loads(json_output)
        assert data['type'] == 'top_errors'
        assert 'top_errors' in data
        assert 'limit' in data

    @pytest.mark.asyncio
    async def test_export_resolution_json(
        self,
        error_manager,
        report_generator
    ):
        """Test JSON export of resolution report."""
        await create_test_error(
            error_manager,
            "Test error",
            acknowledged=True,
            acknowledged_by="admin"
        )

        report = await report_generator.generate_resolution_report()
        json_output = report_generator.export_report(report, format='json')

        data = json.loads(json_output)
        assert data['type'] == 'resolution'
        assert 'metrics' in data
        assert 'total_errors' in data['metrics']
        assert 'acknowledged_count' in data['metrics']
        assert 'acknowledgment_rate' in data['metrics']

    @pytest.mark.asyncio
    async def test_export_invalid_format(self, report_generator):
        """Test export with invalid format raises error."""
        report = SummaryReport(
            statistics=DetailedErrorStatistics(
                total_count=0,
                by_severity={},
                by_category={},
                unacknowledged_count=0
            ),
            time_range_days=7
        )

        with pytest.raises(ValueError, match="Unsupported format"):
            report_generator.export_report(report, format='xml')


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_database(self, report_generator):
        """Test all reports with empty database."""
        summary = await report_generator.generate_summary_report(days=7)
        assert summary.statistics.total_count == 0

        trends = await report_generator.generate_trend_report(days=7)
        assert len(trends.time_series) > 0
        assert all(tp.count == 0 for tp in trends.time_series)

        top_errors = await report_generator.generate_top_errors_report(limit=10)
        assert len(top_errors.top_errors) == 0

        resolution = await report_generator.generate_resolution_report()
        assert resolution.metrics.total_errors == 0
        assert resolution.metrics.acknowledgment_rate == 0.0

    @pytest.mark.asyncio
    async def test_single_error(self, error_manager, report_generator):
        """Test reports with single error."""
        await create_test_error(error_manager, "Single error")

        summary = await report_generator.generate_summary_report(days=7)
        assert summary.statistics.total_count == 1

        top_errors = await report_generator.generate_top_errors_report(limit=10)
        assert len(top_errors.top_errors) == 1
        assert top_errors.top_errors[0].count == 1

    @pytest.mark.asyncio
    async def test_invalid_granularity(self, report_generator):
        """Test trend report with invalid granularity."""
        with pytest.raises(ValueError, match="Invalid granularity"):
            await report_generator.generate_trend_report(
                days=7,
                granularity='monthly'  # type: ignore
            )

    @pytest.mark.asyncio
    async def test_zero_days(self, report_generator):
        """Test reports with zero days should handle gracefully."""
        # This should handle edge case without errors
        summary = await report_generator.generate_summary_report(days=0)
        assert summary.time_range_days == 0


class TestContextPreservation:
    """Test that context is preserved in top errors."""

    @pytest.mark.asyncio
    async def test_top_error_context(self, error_manager, report_generator):
        """Test that context is included in top errors."""
        context = {
            'file_path': '/path/to/file.py',
            'line_number': 42,
            'function': 'process_data'
        }

        await create_test_error(
            error_manager,
            "Test error with context",
            context=context
        )

        report = await report_generator.generate_top_errors_report(limit=10)

        assert len(report.top_errors) == 1
        top_error = report.top_errors[0]
        assert top_error.sample_context is not None
        assert top_error.sample_context['file_path'] == '/path/to/file.py'
        assert top_error.sample_context['line_number'] == 42
        assert top_error.sample_context['function'] == 'process_data'
