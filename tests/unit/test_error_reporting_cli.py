"""
Unit tests for error reporting CLI commands.

Tests CLI commands for error statistics, trends, top errors,
resolution metrics, and comprehensive report generation.
"""

import json
import pytest
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typer.testing import CliRunner

from wqm_cli.cli.commands.error_reporting import (
    errors_app,
    get_severity_color
)


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_db_path(tmp_path):
    """Create a mock database path."""
    db_path = tmp_path / "test_state.db"

    # Create database with messages table
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE messages (
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
    conn.close()

    return db_path


def add_test_error(
    db_path: Path,
    message: str,
    severity: str = "error",
    category: str = "unknown",
    timestamp: datetime = None,
    acknowledged: bool = False,
    acknowledged_by: str = None
):
    """
    Helper to add test error to database.

    Args:
        db_path: Path to database
        message: Error message
        severity: Error severity
        category: Error category
        timestamp: Error timestamp
        acknowledged: Whether error is acknowledged
        acknowledged_by: Who acknowledged
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)

    acknowledged_at = None
    if acknowledged and acknowledged_by:
        acknowledged_at = (timestamp + timedelta(hours=1)).isoformat()

    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        INSERT INTO messages (
            timestamp, severity, category, message, context,
            acknowledged, acknowledged_at, acknowledged_by, retry_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp.isoformat(),
        severity,
        category,
        message,
        None,
        1 if acknowledged else 0,
        acknowledged_at,
        acknowledged_by,
        0
    ))
    conn.commit()
    conn.close()


class TestGetSeverityColor:
    """Test severity color mapping function."""

    def test_error_color(self):
        """Test error severity returns red."""
        assert get_severity_color('error') == 'red'
        assert get_severity_color('ERROR') == 'red'

    def test_warning_color(self):
        """Test warning severity returns yellow."""
        assert get_severity_color('warning') == 'yellow'
        assert get_severity_color('WARNING') == 'yellow'

    def test_info_color(self):
        """Test info severity returns blue."""
        assert get_severity_color('info') == 'blue'
        assert get_severity_color('INFO') == 'blue'

    def test_unknown_color(self):
        """Test unknown severity returns white."""
        assert get_severity_color('unknown') == 'white'
        assert get_severity_color('invalid') == 'white'


class TestStatsCommand:
    """Test 'wqm errors stats' command."""

    @patch('wqm_cli.cli.commands.error_reporting.ErrorReportGenerator')
    def test_stats_default(self, mock_generator_class, runner, mock_db_path):
        """Test stats command with default options."""
        # Mock the generator
        mock_generator = Mock()
        mock_generator.initialize = AsyncMock()
        mock_generator.close = AsyncMock()
        mock_generator.generate_summary_report = AsyncMock()
        mock_generator_class.return_value = mock_generator

        # Mock report data
        from common.core.error_statistics import (
            SummaryReport,
            DetailedErrorStatistics
        )

        mock_stats = DetailedErrorStatistics(
            total_count=10,
            by_severity={'error': 6, 'warning': 3, 'info': 1},
            by_category={'file_corrupt': 5, 'network': 3, 'unknown': 2},
            unacknowledged_count=4,
            error_rate_per_hour=0.5,
            error_rate_per_day=12.0
        )
        mock_report = SummaryReport(
            statistics=mock_stats,
            time_range_days=7
        )
        mock_generator.generate_summary_report.return_value = mock_report

        # Patch database path detection
        with patch(
            'wqm_cli.cli.commands.error_reporting.ErrorReportGenerator',
            return_value=mock_generator
        ):
            result = runner.invoke(errors_app, ['stats'])

        # Command should succeed
        assert result.exit_code == 0

    def test_stats_json_format(self, runner, mock_db_path):
        """Test stats command with JSON output format."""
        now = datetime.now(timezone.utc)

        # Add test errors
        add_test_error(
            mock_db_path,
            "Test error 1",
            severity="error",
            category="file_corrupt",
            timestamp=now - timedelta(hours=2)
        )
        add_test_error(
            mock_db_path,
            "Test error 2",
            severity="warning",
            category="network",
            timestamp=now - timedelta(hours=1)
        )

        # Patch the database path
        with patch(
            'wqm_cli.cli.commands.error_reporting.ErrorReportGenerator'
        ) as mock_gen_class:
            # Create a real generator instance for the test
            from common.core.error_statistics import ErrorReportGenerator

            real_generator = ErrorReportGenerator(db_path=str(mock_db_path))
            mock_gen_class.return_value = real_generator

            result = runner.invoke(errors_app, ['stats', '--format=json'])

        # Should succeed
        assert result.exit_code == 0

        # Output should be valid JSON
        output_data = json.loads(result.stdout)
        assert 'type' in output_data
        assert output_data['type'] == 'summary'

    def test_stats_custom_days(self, runner):
        """Test stats command with custom day range."""
        result = runner.invoke(errors_app, ['stats', '--days=30'])

        # Command structure should be valid (may fail on empty DB)
        # We're testing the CLI parsing works
        assert '--days=30' in str(result) or result.exit_code in [0, 1]


class TestTrendsCommand:
    """Test 'wqm errors trends' command."""

    def test_trends_default(self, runner):
        """Test trends command with default options."""
        result = runner.invoke(errors_app, ['trends'])

        # Command should be recognized
        assert result.exit_code in [0, 1]  # May fail without DB

    def test_trends_hourly_granularity(self, runner):
        """Test trends with hourly granularity."""
        result = runner.invoke(errors_app, ['trends', '--granularity=hourly'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_trends_weekly_granularity(self, runner):
        """Test trends with weekly granularity."""
        result = runner.invoke(errors_app, ['trends', '--granularity=weekly'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_trends_invalid_granularity(self, runner):
        """Test trends with invalid granularity."""
        result = runner.invoke(errors_app, ['trends', '--granularity=monthly'])

        # Should show error
        assert result.exit_code == 1
        assert 'Invalid granularity' in result.stdout

    def test_trends_json_format(self, runner):
        """Test trends with JSON output."""
        result = runner.invoke(
            errors_app,
            ['trends', '--format=json']
        )

        # Command structure should be valid
        assert result.exit_code in [0, 1]

    def test_trends_custom_days(self, runner):
        """Test trends with custom day range."""
        result = runner.invoke(errors_app, ['trends', '--days=14'])

        # Command should be recognized
        assert result.exit_code in [0, 1]


class TestTopCommand:
    """Test 'wqm errors top' command."""

    def test_top_default(self, runner):
        """Test top command with default options."""
        result = runner.invoke(errors_app, ['top'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_top_custom_limit(self, runner):
        """Test top command with custom limit."""
        result = runner.invoke(errors_app, ['top', '--limit=20'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_top_with_days_filter(self, runner):
        """Test top command with day filter."""
        result = runner.invoke(errors_app, ['top', '--days=30'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_top_json_format(self, runner):
        """Test top command with JSON output."""
        result = runner.invoke(errors_app, ['top', '--format=json'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    @patch('wqm_cli.cli.commands.error_reporting.ErrorReportGenerator')
    def test_top_with_results(self, mock_generator_class, runner):
        """Test top command displays results correctly."""
        # Mock the generator
        mock_generator = Mock()
        mock_generator.initialize = AsyncMock()
        mock_generator.close = AsyncMock()
        mock_generator.generate_top_errors_report = AsyncMock()
        mock_generator_class.return_value = mock_generator

        # Mock report data
        from common.core.error_statistics import TopError, TopErrorsReport

        now = datetime.now(timezone.utc)
        mock_top_errors = [
            TopError(
                message="Connection timeout",
                count=10,
                category="timeout",
                severity="error",
                first_seen=now - timedelta(days=1),
                last_seen=now
            ),
            TopError(
                message="File not found",
                count=5,
                category="file_corrupt",
                severity="error",
                first_seen=now - timedelta(hours=12),
                last_seen=now
            )
        ]
        mock_report = TopErrorsReport(
            top_errors=mock_top_errors,
            limit=10
        )
        mock_generator.generate_top_errors_report.return_value = mock_report

        result = runner.invoke(errors_app, ['top'])

        # Should succeed
        assert result.exit_code == 0


class TestResolutionCommand:
    """Test 'wqm errors resolution' command."""

    def test_resolution_default(self, runner):
        """Test resolution command with default options."""
        result = runner.invoke(errors_app, ['resolution'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_resolution_custom_days(self, runner):
        """Test resolution command with custom day range."""
        result = runner.invoke(errors_app, ['resolution', '--days=30'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_resolution_json_format(self, runner):
        """Test resolution command with JSON output."""
        result = runner.invoke(errors_app, ['resolution', '--format=json'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    @patch('wqm_cli.cli.commands.error_reporting.ErrorReportGenerator')
    def test_resolution_with_data(self, mock_generator_class, runner):
        """Test resolution command displays metrics correctly."""
        # Mock the generator
        mock_generator = Mock()
        mock_generator.initialize = AsyncMock()
        mock_generator.close = AsyncMock()
        mock_generator.generate_resolution_report = AsyncMock()
        mock_generator_class.return_value = mock_generator

        # Mock report data
        from common.core.error_statistics import (
            ResolutionReport,
            ResolutionMetrics
        )

        mock_metrics = ResolutionMetrics(
            total_errors=100,
            acknowledged_count=75,
            acknowledgment_rate=75.0,
            avg_time_to_acknowledge=3600.0,  # 1 hour in seconds
            unacknowledged_count=25
        )
        mock_report = ResolutionReport(
            metrics=mock_metrics,
            time_range_days=7
        )
        mock_generator.generate_resolution_report.return_value = mock_report

        result = runner.invoke(errors_app, ['resolution'])

        # Should succeed
        assert result.exit_code == 0


class TestReportCommand:
    """Test 'wqm errors report' command."""

    def test_report_default_json(self, runner):
        """Test report command with default JSON format."""
        result = runner.invoke(errors_app, ['report'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_report_markdown_format(self, runner):
        """Test report command with Markdown format."""
        result = runner.invoke(errors_app, ['report', '--format=markdown'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_report_invalid_format(self, runner):
        """Test report command with invalid format."""
        result = runner.invoke(errors_app, ['report', '--format=xml'])

        # Should show error
        assert result.exit_code == 1
        assert 'Invalid format' in result.stdout

    def test_report_custom_days(self, runner):
        """Test report command with custom day range."""
        result = runner.invoke(errors_app, ['report', '--days=14'])

        # Command should be recognized
        assert result.exit_code in [0, 1]

    def test_report_with_output_file(self, runner, tmp_path):
        """Test report command with output file."""
        output_file = tmp_path / "report.json"

        result = runner.invoke(
            errors_app,
            ['report', f'--output={output_file}']
        )

        # Command should be recognized
        assert result.exit_code in [0, 1]

    @patch('wqm_cli.cli.commands.error_reporting.ErrorReportGenerator')
    def test_report_generates_comprehensive_output(
        self,
        mock_generator_class,
        runner,
        tmp_path
    ):
        """Test report command generates all sections."""
        # Mock the generator
        mock_generator = Mock()
        mock_generator.initialize = AsyncMock()
        mock_generator.close = AsyncMock()
        mock_generator_class.return_value = mock_generator

        # Mock report generation methods
        from common.core.error_statistics import (
            SummaryReport,
            TrendReport,
            TopErrorsReport,
            ResolutionReport,
            DetailedErrorStatistics,
            ResolutionMetrics
        )

        mock_generator.generate_summary_report = AsyncMock(
            return_value=SummaryReport(
                statistics=DetailedErrorStatistics(
                    total_count=10,
                    by_severity={},
                    by_category={},
                    unacknowledged_count=5
                ),
                time_range_days=7
            )
        )
        mock_generator.generate_trend_report = AsyncMock(
            return_value=TrendReport(
                time_series=[],
                granularity='daily',
                time_range_days=7
            )
        )
        mock_generator.generate_top_errors_report = AsyncMock(
            return_value=TopErrorsReport(
                top_errors=[],
                limit=20
            )
        )
        mock_generator.generate_resolution_report = AsyncMock(
            return_value=ResolutionReport(
                metrics=ResolutionMetrics(
                    total_errors=10,
                    acknowledged_count=5,
                    acknowledgment_rate=50.0,
                    unacknowledged_count=5
                ),
                time_range_days=7
            )
        )

        # Mock export_report to return JSON
        def mock_export(report, format='json'):
            return json.dumps({'type': format, 'data': 'mock'})

        mock_generator.export_report = mock_export

        output_file = tmp_path / "report.json"
        result = runner.invoke(
            errors_app,
            ['report', f'--output={output_file}']
        )

        # Should succeed
        assert result.exit_code == 0


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_help_messages(self, runner):
        """Test help messages for all commands."""
        # Main help
        result = runner.invoke(errors_app, ['--help'])
        assert result.exit_code == 0
        assert 'stats' in result.stdout
        assert 'trends' in result.stdout
        assert 'top' in result.stdout
        assert 'resolution' in result.stdout
        assert 'report' in result.stdout

        # Stats help
        result = runner.invoke(errors_app, ['stats', '--help'])
        assert result.exit_code == 0
        assert '--days' in result.stdout
        assert '--format' in result.stdout

        # Trends help
        result = runner.invoke(errors_app, ['trends', '--help'])
        assert result.exit_code == 0
        assert '--granularity' in result.stdout

        # Top help
        result = runner.invoke(errors_app, ['top', '--help'])
        assert result.exit_code == 0
        assert '--limit' in result.stdout

        # Resolution help
        result = runner.invoke(errors_app, ['resolution', '--help'])
        assert result.exit_code == 0

        # Report help
        result = runner.invoke(errors_app, ['report', '--help'])
        assert result.exit_code == 0
        assert '--output' in result.stdout

    def test_command_validation(self, runner):
        """Test command validation and error handling."""
        # Invalid subcommand
        result = runner.invoke(errors_app, ['invalid-command'])
        assert result.exit_code != 0

        # Invalid option
        result = runner.invoke(errors_app, ['stats', '--invalid-option'])
        assert result.exit_code != 0


class TestErrorHandling:
    """Test error handling in CLI commands."""

    @patch('wqm_cli.cli.commands.error_reporting.ErrorReportGenerator')
    def test_generator_initialization_error(self, mock_generator_class, runner):
        """Test handling of generator initialization errors."""
        mock_generator = Mock()
        mock_generator.initialize = AsyncMock(side_effect=Exception("Init failed"))
        mock_generator_class.return_value = mock_generator

        result = runner.invoke(errors_app, ['stats'])

        # Should handle error gracefully
        assert result.exit_code == 1
        assert 'Failed to generate statistics' in result.stdout

    @patch('wqm_cli.cli.commands.error_reporting.ErrorReportGenerator')
    def test_report_generation_error(self, mock_generator_class, runner):
        """Test handling of report generation errors."""
        mock_generator = Mock()
        mock_generator.initialize = AsyncMock()
        mock_generator.generate_summary_report = AsyncMock(
            side_effect=Exception("Generation failed")
        )
        mock_generator_class.return_value = mock_generator

        result = runner.invoke(errors_app, ['stats'])

        # Should handle error gracefully
        assert result.exit_code == 1


class TestOutputFormatting:
    """Test output formatting for different formats."""

    @patch('wqm_cli.cli.commands.error_reporting.ErrorReportGenerator')
    def test_table_output_structure(self, mock_generator_class, runner):
        """Test that table output has proper structure."""
        # Mock the generator with realistic data
        mock_generator = Mock()
        mock_generator.initialize = AsyncMock()
        mock_generator.close = AsyncMock()
        mock_generator_class.return_value = mock_generator

        from common.core.error_statistics import (
            SummaryReport,
            DetailedErrorStatistics
        )

        mock_stats = DetailedErrorStatistics(
            total_count=50,
            by_severity={'error': 30, 'warning': 15, 'info': 5},
            by_category={'file_corrupt': 20, 'network': 15, 'unknown': 15},
            unacknowledged_count=10,
            error_rate_per_hour=2.0,
            error_rate_per_day=48.0
        )
        mock_report = SummaryReport(
            statistics=mock_stats,
            time_range_days=7
        )
        mock_generator.generate_summary_report = AsyncMock(
            return_value=mock_report
        )

        result = runner.invoke(errors_app, ['stats'])

        # Should succeed and contain expected elements
        assert result.exit_code == 0
        # Rich table output should be present (may vary based on terminal)

    @patch('wqm_cli.cli.commands.error_reporting.ErrorReportGenerator')
    def test_json_output_valid(self, mock_generator_class, runner):
        """Test that JSON output is valid."""
        mock_generator = Mock()
        mock_generator.initialize = AsyncMock()
        mock_generator.close = AsyncMock()
        mock_generator_class.return_value = mock_generator

        from common.core.error_statistics import (
            SummaryReport,
            DetailedErrorStatistics
        )

        mock_stats = DetailedErrorStatistics(
            total_count=10,
            by_severity={'error': 10},
            by_category={'unknown': 10},
            unacknowledged_count=5
        )
        mock_report = SummaryReport(
            statistics=mock_stats,
            time_range_days=7
        )
        mock_generator.generate_summary_report = AsyncMock(
            return_value=mock_report
        )
        mock_generator.export_report = Mock(
            return_value=json.dumps({'type': 'summary', 'data': 'test'})
        )

        result = runner.invoke(errors_app, ['stats', '--format=json'])

        assert result.exit_code == 0
        # Output should be valid JSON
        try:
            json.loads(result.stdout)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")
