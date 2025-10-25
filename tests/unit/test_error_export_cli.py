"""Unit tests for error export CLI commands.

Tests CLI commands for error export and debug bundle generation:
- wqm errors export - Export to CSV/JSON
- wqm errors debug-bundle - Create debug bundles
"""

import json
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from typer.testing import CliRunner
from wqm_cli.cli.commands.error_reporting import errors_app


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_error_manager():
    """Create mock error manager."""
    manager = Mock()
    manager.initialize = AsyncMock()
    manager.close = AsyncMock()
    manager._initialized = True
    return manager


@pytest.fixture
def mock_exporter():
    """Create mock error exporter."""
    exporter = Mock()
    exporter.initialize = AsyncMock()
    exporter.close = AsyncMock()
    exporter.export_to_csv = AsyncMock(return_value=True)
    exporter.export_to_json = AsyncMock(return_value=True)
    exporter.export_filtered = AsyncMock(return_value=True)
    exporter.export_date_range = AsyncMock(return_value=True)
    return exporter


@pytest.fixture
def mock_bundle_generator():
    """Create mock debug bundle generator."""
    generator = Mock()
    generator.initialize = AsyncMock()
    generator.close = AsyncMock()

    # Mock bundle
    mock_bundle = Mock()
    mock_bundle.bundle_id = "test_bundle_123"
    mock_bundle.error_ids = ["1", "2"]

    generator.create_debug_bundle = AsyncMock(return_value=mock_bundle)
    generator.bundle_to_archive = AsyncMock(return_value="/tmp/debug.tar.gz")

    return generator


class TestExportCommand:
    """Test 'wqm errors export' command."""

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_csv_basic(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test CSV export with basic options."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'csv',
            '--output', str(output_file)
        ])

        assert result.exit_code == 0
        assert "Successfully exported" in result.output
        mock_exporter.export_filtered.assert_called_once()

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_json_basic(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test JSON export with basic options."""
        output_file = tmp_path / "errors.json"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'json',
            '--output', str(output_file)
        ])

        assert result.exit_code == 0
        assert "Successfully exported" in result.output

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_with_severity_filter(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test export with severity filter."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'csv',
            '--output', str(output_file),
            '--severity', 'error'
        ])

        assert result.exit_code == 0
        mock_exporter.export_filtered.assert_called_once()

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_with_category_filter(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test export with category filter."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'csv',
            '--output', str(output_file),
            '--category', 'file_corrupt'
        ])

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_with_days_filter(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test export with days filter."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'csv',
            '--output', str(output_file),
            '--days', '30'
        ])

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_with_date_range(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test export with explicit date range."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'csv',
            '--output', str(output_file),
            '--start-date', '2025-01-01',
            '--end-date', '2025-01-31'
        ])

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_with_acknowledged_filter(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test export with acknowledged filter."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'csv',
            '--output', str(output_file),
            '--acknowledged'
        ])

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_with_limit(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test export with custom limit."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'csv',
            '--output', str(output_file),
            '--limit', '500'
        ])

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_invalid_format(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test export with invalid format shows error."""
        output_file = tmp_path / "errors.txt"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'txt',
            '--output', str(output_file)
        ])

        assert result.exit_code != 0
        assert "Invalid format" in result.output

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_invalid_severity(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager, mock_exporter):
        """Test export with invalid severity shows error."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_exporter_class.return_value = mock_exporter

        # Mock severity validation to raise ValueError
        from common.core.error_categorization import ErrorSeverity
        with patch.object(ErrorSeverity, 'from_string', side_effect=ValueError("Invalid severity")):
            result = runner.invoke(errors_app, [
                'export',
                '--format', 'csv',
                '--output', str(output_file),
                '--severity', 'invalid'
            ])

        assert result.exit_code != 0

    @patch('wqm_cli.cli.commands.error_export_cli.ErrorExporter')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_export_no_matches(self, mock_manager_class, mock_exporter_class, runner, tmp_path, mock_error_manager):
        """Test export with no matching errors shows error."""
        output_file = tmp_path / "errors.csv"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager

        exporter = Mock()
        exporter.initialize = AsyncMock()
        exporter.close = AsyncMock()
        exporter.export_filtered = AsyncMock(side_effect=ValueError("No errors match"))
        mock_exporter_class.return_value = exporter

        result = runner.invoke(errors_app, [
            'export',
            '--format', 'csv',
            '--output', str(output_file),
            '--severity', 'error'
        ])

        assert result.exit_code != 0


class TestDebugBundleCommand:
    """Test 'wqm errors debug-bundle' command."""

    @patch('wqm_cli.cli.commands.error_export_cli.DebugBundleGenerator')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    @patch('shutil.rmtree')
    def test_debug_bundle_single_error(self, mock_rmtree, mock_manager_class, mock_generator_class, runner, tmp_path, mock_error_manager, mock_bundle_generator):
        """Test debug bundle with single error ID."""
        output_file = tmp_path / "debug.tar.gz"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager

        # Mock get_error_by_id to return a valid error
        from common.core.error_categorization import ErrorCategory, ErrorSeverity
        from common.core.error_message_manager import ErrorMessage
        mock_error = ErrorMessage(
            id=123,
            timestamp=datetime.now(),
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.FILE_CORRUPT,
            message="Test error",
            retry_count=0
        )
        mock_error_manager.get_error_by_id = AsyncMock(return_value=mock_error)

        mock_generator_class.return_value = mock_bundle_generator

        result = runner.invoke(errors_app, [
            'debug-bundle',
            '--error-id', '123',
            '--output', str(output_file)
        ])

        assert result.exit_code == 0
        assert "Debug bundle created" in result.output
        assert "test_bundle_123" in result.output

    @patch('wqm_cli.cli.commands.error_export_cli.DebugBundleGenerator')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    @patch('shutil.rmtree')
    def test_debug_bundle_last_n(self, mock_rmtree, mock_manager_class, mock_generator_class, runner, tmp_path, mock_error_manager, mock_bundle_generator):
        """Test debug bundle with last N errors."""
        output_file = tmp_path / "debug.tar.gz"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager

        # Mock get_errors to return a list
        from common.core.error_categorization import ErrorCategory, ErrorSeverity
        from common.core.error_message_manager import ErrorMessage
        mock_errors = [
            ErrorMessage(id=i, timestamp=datetime.now(), severity=ErrorSeverity.ERROR,
                        category=ErrorCategory.FILE_CORRUPT, message=f"Error {i}",
                        retry_count=0)
            for i in range(1, 11)
        ]
        mock_error_manager.get_errors = AsyncMock(return_value=mock_errors)

        mock_generator_class.return_value = mock_bundle_generator

        result = runner.invoke(errors_app, [
            'debug-bundle',
            '--last-n', '10',
            '--output', str(output_file)
        ])

        assert result.exit_code == 0
        assert "Including 10 errors" in result.output
        assert "Debug bundle created" in result.output

    @patch('wqm_cli.cli.commands.error_export_cli.DebugBundleGenerator')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    @patch('shutil.rmtree')
    def test_debug_bundle_with_severity_filter(self, mock_rmtree, mock_manager_class, mock_generator_class, runner, tmp_path, mock_error_manager, mock_bundle_generator):
        """Test debug bundle with severity filter."""
        output_file = tmp_path / "debug.tar.gz"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager

        # Mock get_errors
        from common.core.error_categorization import ErrorCategory, ErrorSeverity
        from common.core.error_message_manager import ErrorMessage
        mock_errors = [
            ErrorMessage(id=1, timestamp=datetime.now(), severity=ErrorSeverity.ERROR,
                        category=ErrorCategory.FILE_CORRUPT, message="Error 1",
                        retry_count=0)
        ]
        mock_error_manager.get_errors = AsyncMock(return_value=mock_errors)

        mock_generator_class.return_value = mock_bundle_generator

        result = runner.invoke(errors_app, [
            'debug-bundle',
            '--last-n', '10',
            '--severity', 'error',
            '--output', str(output_file)
        ])

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.error_export_cli.DebugBundleGenerator')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_debug_bundle_error_not_found(self, mock_manager_class, mock_generator_class, runner, tmp_path, mock_error_manager):
        """Test debug bundle with non-existent error ID."""
        output_file = tmp_path / "debug.tar.gz"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_error_manager.get_error_by_id = AsyncMock(return_value=None)

        generator = Mock()
        generator.initialize = AsyncMock()
        generator.close = AsyncMock()
        mock_generator_class.return_value = generator

        result = runner.invoke(errors_app, [
            'debug-bundle',
            '--error-id', '999',
            '--output', str(output_file)
        ])

        assert result.exit_code != 0
        assert "not found" in result.output

    @patch('wqm_cli.cli.commands.error_export_cli.DebugBundleGenerator')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_debug_bundle_no_matching_errors(self, mock_manager_class, mock_generator_class, runner, tmp_path, mock_error_manager):
        """Test debug bundle with no matching errors."""
        output_file = tmp_path / "debug.tar.gz"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager
        mock_error_manager.get_errors = AsyncMock(return_value=[])

        generator = Mock()
        generator.initialize = AsyncMock()
        generator.close = AsyncMock()
        mock_generator_class.return_value = generator

        result = runner.invoke(errors_app, [
            'debug-bundle',
            '--last-n', '10',
            '--output', str(output_file)
        ])

        assert result.exit_code != 0
        assert "No errors found" in result.output

    @patch('wqm_cli.cli.commands.error_export_cli.DebugBundleGenerator')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_debug_bundle_missing_params(self, mock_manager_class, mock_generator_class, runner, tmp_path):
        """Test debug bundle without required parameters."""
        output_file = tmp_path / "debug.tar.gz"

        result = runner.invoke(errors_app, [
            'debug-bundle',
            '--output', str(output_file)
        ])

        assert result.exit_code != 0
        assert "Specify either --error-id or --last-n" in result.output

    @patch('wqm_cli.cli.commands.error_export_cli.DebugBundleGenerator')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    def test_debug_bundle_both_params(self, mock_manager_class, mock_generator_class, runner, tmp_path):
        """Test debug bundle with both error-id and last-n shows error."""
        output_file = tmp_path / "debug.tar.gz"

        result = runner.invoke(errors_app, [
            'debug-bundle',
            '--error-id', '123',
            '--last-n', '10',
            '--output', str(output_file)
        ])

        assert result.exit_code != 0
        assert "Cannot specify both" in result.output

    @patch('wqm_cli.cli.commands.error_export_cli.DebugBundleGenerator')
    @patch('wqm_cli.cli.commands.error_export_cli.ErrorMessageManager')
    @patch('shutil.rmtree')
    def test_debug_bundle_creation_failure(self, mock_rmtree, mock_manager_class, mock_generator_class, runner, tmp_path, mock_error_manager):
        """Test debug bundle handles creation failures."""
        output_file = tmp_path / "debug.tar.gz"

        # Setup mocks
        mock_manager_class.return_value = mock_error_manager

        from common.core.error_categorization import ErrorCategory, ErrorSeverity
        from common.core.error_message_manager import ErrorMessage
        mock_error = ErrorMessage(
            id=123,
            timestamp=datetime.now(),
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.FILE_CORRUPT,
            message="Test error",
            retry_count=0
        )
        mock_error_manager.get_error_by_id = AsyncMock(return_value=mock_error)

        generator = Mock()
        generator.initialize = AsyncMock()
        generator.close = AsyncMock()
        generator.create_debug_bundle = AsyncMock(side_effect=Exception("Bundle creation failed"))
        mock_generator_class.return_value = generator

        result = runner.invoke(errors_app, [
            'debug-bundle',
            '--error-id', '123',
            '--output', str(output_file)
        ])

        assert result.exit_code != 0
        assert "Failed to create debug bundle" in result.output
