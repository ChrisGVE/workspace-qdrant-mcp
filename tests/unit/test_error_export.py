"""Unit tests for error export and debug bundle generation.

Tests ErrorExporter and DebugBundleGenerator classes including:
- CSV export with proper escaping
- JSON export
- Filtered exports
- Date range exports
- Debug bundle creation
- System info collection
- Context gathering
- Log extraction
- Archive creation
- Edge cases
"""

import csv
import json
import sqlite3
import tarfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.python.common.core.error_categorization import ErrorCategory, ErrorSeverity
from src.python.common.core.error_export import (
    DebugBundle,
    DebugBundleGenerator,
    ErrorExporter,
)
from src.python.common.core.error_filtering import ErrorFilter
from src.python.common.core.error_message_manager import ErrorMessage


@pytest.fixture
def sample_errors():
    """Create sample error messages for testing."""
    now = datetime.now()

    return [
        ErrorMessage(
            id=1,
            timestamp=now,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.FILE_CORRUPT,
            message="File not found",
            context={"file_path": "/path/to/file.txt", "collection": "test", "tenant_id": "default"},
            acknowledged=False,
            retry_count=0
        ),
        ErrorMessage(
            id=2,
            timestamp=now - timedelta(hours=1),
            severity=ErrorSeverity.WARNING,
            category=ErrorCategory.NETWORK,
            message='Message with "quotes" and, commas',
            context={"file_path": "/other/file.py", "collection": "test2", "tenant_id": "tenant1"},
            acknowledged=True,
            acknowledged_at=now,
            acknowledged_by="admin",
            retry_count=1
        ),
        ErrorMessage(
            id=3,
            timestamp=now - timedelta(days=1),
            severity=ErrorSeverity.INFO,
            category=ErrorCategory.PROCESSING_FAILED,
            message="Multi-line\nmessage\nhere",
            context=None,
            acknowledged=False,
            retry_count=2
        ),
    ]


@pytest.fixture
async def error_manager_mock():
    """Create mock error manager."""
    manager = AsyncMock()
    manager._initialized = True
    manager.connection_pool = MagicMock()
    manager.connection_pool.db_path = "/tmp/test.db"
    return manager


@pytest.fixture
async def exporter(error_manager_mock):
    """Create ErrorExporter instance."""
    exporter = ErrorExporter(error_manager_mock)
    exporter._initialized = True
    exporter.filter_manager._initialized = True
    return exporter


@pytest.fixture
async def bundle_generator(error_manager_mock):
    """Create DebugBundleGenerator instance."""
    generator = DebugBundleGenerator(error_manager_mock)
    generator._initialized = True
    generator._db_path = "/tmp/test.db"
    return generator


class TestErrorExporter:
    """Tests for ErrorExporter class."""

    @pytest.mark.asyncio
    async def test_export_to_csv_basic(self, exporter, sample_errors, tmp_path):
        """Test basic CSV export."""
        output_file = tmp_path / "errors.csv"

        result = await exporter.export_to_csv(sample_errors, str(output_file))

        assert result is True
        assert output_file.exists()

        # Read and verify CSV
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert rows[0]['id'] == '1'
        assert rows[0]['severity'] == 'error'
        assert rows[0]['message'] == 'File not found'
        assert rows[0]['file_path'] == '/path/to/file.txt'

    @pytest.mark.asyncio
    async def test_export_to_csv_escaping(self, exporter, sample_errors, tmp_path):
        """Test CSV export with proper field escaping."""
        output_file = tmp_path / "errors.csv"

        result = await exporter.export_to_csv(sample_errors, str(output_file))

        assert result is True

        # Read and verify escaping
        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check row with quotes and commas
        row_with_quotes = rows[1]
        assert row_with_quotes['message'] == 'Message with "quotes" and, commas'

        # Check multi-line message
        row_multiline = rows[2]
        assert "Multi-line\nmessage\nhere" in row_multiline['message']

    @pytest.mark.asyncio
    async def test_export_to_csv_empty_context(self, exporter, sample_errors, tmp_path):
        """Test CSV export with errors having empty context."""
        output_file = tmp_path / "errors.csv"

        result = await exporter.export_to_csv(sample_errors, str(output_file))

        assert result is True

        with open(output_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Row 3 has no context
        assert rows[2]['file_path'] == ''
        assert rows[2]['collection'] == ''
        assert rows[2]['tenant_id'] == ''

    @pytest.mark.asyncio
    async def test_export_to_csv_empty_list(self, exporter, tmp_path):
        """Test CSV export with empty error list raises ValueError."""
        output_file = tmp_path / "errors.csv"

        with pytest.raises(ValueError, match="Cannot export empty error list"):
            await exporter.export_to_csv([], str(output_file))

    @pytest.mark.asyncio
    async def test_export_to_csv_creates_directories(self, exporter, sample_errors, tmp_path):
        """Test CSV export creates parent directories if needed."""
        output_file = tmp_path / "subdir" / "nested" / "errors.csv"

        result = await exporter.export_to_csv(sample_errors, str(output_file))

        assert result is True
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_export_to_json_basic(self, exporter, sample_errors, tmp_path):
        """Test basic JSON export."""
        output_file = tmp_path / "errors.json"

        result = await exporter.export_to_json(sample_errors, str(output_file))

        assert result is True
        assert output_file.exists()

        # Read and verify JSON
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]['id'] == 1
        assert data[0]['severity'] == 'error'
        assert data[0]['message'] == 'File not found'
        assert data[0]['context']['file_path'] == '/path/to/file.txt'

    @pytest.mark.asyncio
    async def test_export_to_json_full_details(self, exporter, sample_errors, tmp_path):
        """Test JSON export includes all error details."""
        output_file = tmp_path / "errors.json"

        result = await exporter.export_to_json(sample_errors, str(output_file))

        assert result is True

        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check acknowledged error has all fields
        ack_error = data[1]
        assert ack_error['acknowledged'] is True
        assert ack_error['acknowledged_by'] == 'admin'
        assert ack_error['acknowledged_at'] is not None
        assert ack_error['retry_count'] == 1

    @pytest.mark.asyncio
    async def test_export_to_json_empty_list(self, exporter, tmp_path):
        """Test JSON export with empty error list raises ValueError."""
        output_file = tmp_path / "errors.json"

        with pytest.raises(ValueError, match="Cannot export empty error list"):
            await exporter.export_to_json([], str(output_file))

    @pytest.mark.asyncio
    async def test_export_filtered_csv(self, exporter, sample_errors, tmp_path):
        """Test filtered export to CSV."""
        output_file = tmp_path / "filtered.csv"

        # Mock filter_manager
        exporter.filter_manager.filter_errors = AsyncMock(
            return_value=MagicMock(errors=sample_errors[:2])
        )

        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        result = await exporter.export_filtered(filter, "csv", str(output_file))

        assert result is True
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_export_filtered_json(self, exporter, sample_errors, tmp_path):
        """Test filtered export to JSON."""
        output_file = tmp_path / "filtered.json"

        # Mock filter_manager
        exporter.filter_manager.filter_errors = AsyncMock(
            return_value=MagicMock(errors=sample_errors[:1])
        )

        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        result = await exporter.export_filtered(filter, "json", str(output_file))

        assert result is True
        assert output_file.exists()

    @pytest.mark.asyncio
    async def test_export_filtered_invalid_format(self, exporter, tmp_path):
        """Test filtered export with invalid format raises ValueError."""
        output_file = tmp_path / "errors.txt"

        filter = ErrorFilter()

        with pytest.raises(ValueError, match="Invalid format"):
            await exporter.export_filtered(filter, "txt", str(output_file))

    @pytest.mark.asyncio
    async def test_export_filtered_no_matches(self, exporter, tmp_path):
        """Test filtered export with no matches raises ValueError."""
        output_file = tmp_path / "errors.csv"

        # Mock filter_manager to return empty results
        exporter.filter_manager.filter_errors = AsyncMock(
            return_value=MagicMock(errors=[])
        )

        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])

        with pytest.raises(ValueError, match="No errors match"):
            await exporter.export_filtered(filter, "csv", str(output_file))

    @pytest.mark.asyncio
    async def test_export_date_range(self, exporter, sample_errors, tmp_path):
        """Test date range export."""
        output_file = tmp_path / "range.csv"

        # Mock filter_manager
        exporter.filter_manager.filter_errors = AsyncMock(
            return_value=MagicMock(errors=sample_errors)
        )

        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        result = await exporter.export_date_range(
            start_date, end_date, "csv", str(output_file)
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_export_date_range_invalid(self, exporter, tmp_path):
        """Test date range export with invalid dates raises ValueError."""
        output_file = tmp_path / "range.csv"

        start_date = datetime.now()
        end_date = datetime.now() - timedelta(days=7)

        with pytest.raises(ValueError, match="start_date must be before"):
            await exporter.export_date_range(
                start_date, end_date, "csv", str(output_file)
            )

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, error_manager_mock, sample_errors, tmp_path):
        """Test operations on uninitialized exporter raise RuntimeError."""
        exporter = ErrorExporter(error_manager_mock)
        exporter._initialized = False

        output_file = tmp_path / "errors.csv"

        with pytest.raises(RuntimeError, match="not initialized"):
            await exporter.export_to_csv(sample_errors, str(output_file))


class TestDebugBundleGenerator:
    """Tests for DebugBundleGenerator class."""

    @pytest.mark.asyncio
    async def test_include_system_info(self, bundle_generator):
        """Test system information collection."""
        info = bundle_generator.include_system_info()

        assert 'os' in info
        assert 'python_version' in info
        assert 'platform' in info
        assert 'architecture' in info
        assert 'package_versions' in info

        # Verify data types
        assert isinstance(info['os'], str)
        assert isinstance(info['python_version'], str)
        assert isinstance(info['package_versions'], dict)

    @pytest.mark.asyncio
    async def test_include_error_context(self, bundle_generator, error_manager_mock, sample_errors):
        """Test error context gathering."""
        # Mock error manager
        error_manager_mock.get_error_by_id = AsyncMock(return_value=sample_errors[0])

        # Mock database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall = MagicMock(return_value=[])
        mock_conn.execute = MagicMock(return_value=mock_cursor)

        error_manager_mock.connection_pool.get_connection_async = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock())
        )

        context = await bundle_generator.include_error_context("1")

        assert 'error' in context
        assert context['error']['id'] == 1
        assert 'related_queue_items' in context

    @pytest.mark.asyncio
    async def test_include_error_context_not_found(self, bundle_generator, error_manager_mock):
        """Test error context when error not found."""
        error_manager_mock.get_error_by_id = AsyncMock(return_value=None)

        context = await bundle_generator.include_error_context("999")

        assert 'error' in context
        assert 'Error ID 999 not found' in context['error']

    @pytest.mark.asyncio
    async def test_include_logs_no_log_file(self, bundle_generator, error_manager_mock, sample_errors):
        """Test log extraction when no log file exists."""
        error_manager_mock.get_error_by_id = AsyncMock(return_value=sample_errors[0])

        logs = await bundle_generator.include_logs("1", lines=100)

        assert isinstance(logs, str)
        assert "No log files found" in logs or "Searched:" in logs

    @pytest.mark.asyncio
    async def test_include_logs_with_log_file(self, bundle_generator, error_manager_mock, sample_errors, tmp_path):
        """Test log extraction with existing log file."""
        # Create temporary log file
        log_file = tmp_path / "daemon.log"
        log_content = "\n".join([f"Log line {i}" for i in range(200)])
        log_file.write_text(log_content)

        error_manager_mock.get_error_by_id = AsyncMock(return_value=sample_errors[0])

        # Patch log paths to include our temp file
        with patch.object(bundle_generator, 'include_logs') as mock_include:
            mock_include.return_value = f"=== Log file: {log_file} ===\nLog line 100\nLog line 101\n"

            logs = await mock_include("1", lines=100)

            assert str(log_file) in logs

    @pytest.mark.asyncio
    async def test_create_debug_bundle(self, bundle_generator, error_manager_mock, sample_errors, tmp_path):
        """Test debug bundle creation."""
        output_dir = tmp_path / "bundle"

        # Mock methods
        error_manager_mock.get_error_by_id = AsyncMock(side_effect=lambda id: sample_errors[int(id)-1])

        bundle_generator.include_system_info = MagicMock(return_value={"os": "test", "python_version": "3.11"})
        bundle_generator.include_error_context = AsyncMock(return_value={"error": sample_errors[0].to_dict()})
        bundle_generator.include_logs = AsyncMock(return_value="Test log content")

        bundle = await bundle_generator.create_debug_bundle(
            error_ids=["1", "2"],
            output_path=str(output_dir)
        )

        assert isinstance(bundle, DebugBundle)
        assert bundle.bundle_id.startswith("debug_")
        assert len(bundle.error_ids) == 2
        assert bundle.output_path == output_dir

        # Verify files were created
        assert (output_dir / "error_details.json").exists()
        assert (output_dir / "system_info.json").exists()
        assert (output_dir / "context.json").exists()
        assert (output_dir / "logs.txt").exists()
        assert (output_dir / "metadata.json").exists()

    @pytest.mark.asyncio
    async def test_create_debug_bundle_empty_error_ids(self, bundle_generator, tmp_path):
        """Test debug bundle creation with empty error_ids raises ValueError."""
        output_dir = tmp_path / "bundle"

        with pytest.raises(ValueError, match="error_ids cannot be empty"):
            await bundle_generator.create_debug_bundle(
                error_ids=[],
                output_path=str(output_dir)
            )

    @pytest.mark.asyncio
    async def test_bundle_to_archive(self, bundle_generator, tmp_path):
        """Test archive creation from debug bundle."""
        # Create mock bundle directory
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "test.txt").write_text("Test content")

        bundle = DebugBundle(
            bundle_id="test_bundle",
            created_at=datetime.now(),
            error_ids=["1"],
            error_details=[],
            system_info={},
            context_data={},
            log_excerpts={},
            output_path=bundle_dir
        )

        archive_path = tmp_path / "bundle.tar.gz"

        result = await bundle_generator.bundle_to_archive(bundle, str(archive_path))

        assert result == str(archive_path)
        assert Path(archive_path).exists()

        # Verify archive contents
        with tarfile.open(archive_path, 'r:gz') as tar:
            members = tar.getnames()
            assert 'test_bundle/test.txt' in members

    @pytest.mark.asyncio
    async def test_bundle_to_archive_no_output_path(self, bundle_generator):
        """Test archive creation with no output_path raises ValueError."""
        bundle = DebugBundle(
            bundle_id="test",
            created_at=datetime.now(),
            error_ids=[],
            error_details=[],
            system_info={},
            context_data={},
            log_excerpts={},
            output_path=None
        )

        with pytest.raises(ValueError, match="Bundle has no output_path"):
            await bundle_generator.bundle_to_archive(bundle, "/tmp/output.tar.gz")

    @pytest.mark.asyncio
    async def test_not_initialized_error(self, error_manager_mock, tmp_path):
        """Test operations on uninitialized generator raise RuntimeError."""
        generator = DebugBundleGenerator(error_manager_mock)
        generator._initialized = False

        with pytest.raises(RuntimeError, match="not initialized"):
            await generator.create_debug_bundle(["1"], str(tmp_path))
