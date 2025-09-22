"""Integration tests for migration reporting CLI commands."""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from wqm_cli.cli.commands.admin import admin_app
from workspace_qdrant_mcp.utils.migration import ConfigMigrator, MigrationReport, ChangeEntry, ChangeType


class TestMigrationReportingCLI:
    """Integration tests for migration reporting CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_migration_report_latest(self):
        """Test migration-report command with --latest flag."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            # Create a mock report
            mock_report = MigrationReport(
                migration_id="test-latest-123",
                source_version="v1",
                target_version="v2",
                success=True
            )
            mock_report.add_change(
                ChangeEntry(
                    change_type=ChangeType.ADDED,
                    field_path="collections.test",
                    new_value={"model": "test-model"}
                )
            )

            mock_migrator.get_latest_migration_report.return_value = mock_report

            # Mock the report generator
            mock_report_gen = MagicMock()
            mock_report_gen.format_report_text.return_value = "Mock Migration Report\ntest-latest-123"
            mock_migrator.report_generator = mock_report_gen

            result = self.runner.invoke(admin_app, ["migration-report", "--latest"])

            assert result.exit_code == 0
            assert "Mock Migration Report" in result.stdout
            assert "test-latest-123" in result.stdout
            mock_migrator.get_latest_migration_report.assert_called_once()

    def test_migration_report_specific_id(self):
        """Test migration-report command with specific migration ID."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_report = MigrationReport(migration_id="specific-id-456")
            mock_migrator.get_migration_report.return_value = mock_report

            mock_report_gen = MagicMock()
            mock_report_gen.format_report_text.return_value = "Specific Report: specific-id-456"
            mock_migrator.report_generator = mock_report_gen

            result = self.runner.invoke(admin_app, ["migration-report", "specific-id-456"])

            assert result.exit_code == 0
            assert "Specific Report: specific-id-456" in result.stdout
            mock_migrator.get_migration_report.assert_called_once_with("specific-id-456")

    def test_migration_report_not_found(self):
        """Test migration-report command when report is not found."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator
            mock_migrator.get_migration_report.return_value = None

            result = self.runner.invoke(admin_app, ["migration-report", "nonexistent-id"])

            assert result.exit_code == 0
            assert "Migration report 'nonexistent-id' not found" in result.stdout

    def test_migration_report_json_format(self):
        """Test migration-report command with JSON format."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_report = MigrationReport(migration_id="json-test")
            mock_migrator.get_latest_migration_report.return_value = mock_report

            mock_report_gen = MagicMock()
            mock_report_gen.format_report_json.return_value = '{"migration_id": "json-test"}'
            mock_migrator.report_generator = mock_report_gen

            result = self.runner.invoke(admin_app, ["migration-report", "--latest", "--format", "json"])

            assert result.exit_code == 0
            assert '"migration_id": "json-test"' in result.stdout

    def test_migration_report_export(self):
        """Test migration-report command with export functionality."""
        export_path = self.temp_path / "exported_report.txt"

        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_report = MigrationReport(migration_id="export-test")
            mock_migrator.get_latest_migration_report.return_value = mock_report

            mock_report_gen = MagicMock()
            mock_report_gen.format_report_text.return_value = "Exported Report Content"
            mock_migrator.report_generator = mock_report_gen

            result = self.runner.invoke(admin_app, [
                "migration-report", "--latest", "--export", str(export_path)
            ])

            assert result.exit_code == 0
            assert f"Migration report exported to: {export_path}" in result.stdout
            assert export_path.exists()
            assert export_path.read_text() == "Exported Report Content"

    def test_migration_history_basic(self):
        """Test migration-history command basic functionality."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_history = [
                {
                    "migration_id": "hist-001",
                    "timestamp": "2023-01-01T10:00:00",
                    "source_version": "v1",
                    "target_version": "v2",
                    "success": True,
                    "changes_count": 5,
                },
                {
                    "migration_id": "hist-002",
                    "timestamp": "2023-01-02T11:00:00",
                    "source_version": "v2",
                    "target_version": "v3",
                    "success": False,
                    "changes_count": 3,
                },
            ]
            mock_migrator.get_migration_history.return_value = mock_history

            result = self.runner.invoke(admin_app, ["migration-history"])

            assert result.exit_code == 0
            assert "Migration History (2 records)" in result.stdout
            assert "hist-001" in result.stdout
            assert "hist-002" in result.stdout
            assert "SUCCESS" in result.stdout
            assert "FAILED" in result.stdout

    def test_migration_history_with_filters(self):
        """Test migration-history command with filtering options."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_filtered_history = [
                {
                    "migration_id": "filtered-001",
                    "timestamp": "2023-01-01T10:00:00",
                    "source_version": "v1",
                    "target_version": "v2",
                    "success": True,
                    "changes_count": 2,
                }
            ]
            mock_migrator.search_migration_history.return_value = mock_filtered_history

            result = self.runner.invoke(admin_app, [
                "migration-history",
                "--source", "v1",
                "--success-only", "true",
                "--days", "7"
            ])

            assert result.exit_code == 0
            assert "Migration History (1 records)" in result.stdout
            assert "filtered-001" in result.stdout
            mock_migrator.search_migration_history.assert_called_once_with(
                source_version="v1",
                target_version=None,
                success_only=True,
                days_back=7
            )

    def test_migration_history_json_format(self):
        """Test migration-history command with JSON format."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_history = [{"migration_id": "json-hist-001", "success": True}]
            mock_migrator.get_migration_history.return_value = mock_history

            result = self.runner.invoke(admin_app, ["migration-history", "--format", "json"])

            assert result.exit_code == 0
            # Should be valid JSON
            output_data = json.loads(result.stdout.strip())
            assert len(output_data) == 1
            assert output_data[0]["migration_id"] == "json-hist-001"

    def test_migration_history_empty(self):
        """Test migration-history command with no history."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator
            mock_migrator.get_migration_history.return_value = []

            result = self.runner.invoke(admin_app, ["migration-history"])

            assert result.exit_code == 0
            assert "No migrations found matching the criteria" in result.stdout

    def test_validate_backup_success(self):
        """Test validate-backup command with successful validation."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_migrator.validate_backup.return_value = True
            mock_migrator.get_backup_info.return_value = {
                "timestamp": "2023-01-01T10:00:00",
                "version": "v1",
                "file_size": 1024,
            }

            result = self.runner.invoke(admin_app, ["validate-backup", "test-backup-id"])

            assert result.exit_code == 0
            assert "✅ Backup validation successful" in result.stdout
            assert "Checksum matches, JSON is valid" in result.stdout
            assert "2023-01-01T10:00:00" in result.stdout
            mock_migrator.validate_backup.assert_called_once_with("test-backup-id")

    def test_validate_backup_failure(self):
        """Test validate-backup command with validation failure."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_migrator.validate_backup.return_value = False

            result = self.runner.invoke(admin_app, ["validate-backup", "invalid-backup-id"])

            assert result.exit_code == 1
            assert "❌ Backup validation failed" in result.stdout
            assert "Backup file is corrupted or missing" in result.stdout

    def test_rollback_config_success(self):
        """Test rollback-config command with successful rollback."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_migrator.get_backup_info.return_value = {
                "timestamp": "2023-01-01T10:00:00",
                "version": "v1",
            }
            mock_migrator.validate_backup.return_value = True
            mock_migrator.rollback_config.return_value = {"test": "config"}

            result = self.runner.invoke(admin_app, ["rollback-config", "rollback-id", "--force"])

            assert result.exit_code == 0
            assert "✅ Configuration rollback completed successfully" in result.stdout
            assert "2023-01-01T10:00:00" in result.stdout
            mock_migrator.rollback_config.assert_called_once_with("rollback-id")

    def test_rollback_config_backup_not_found(self):
        """Test rollback-config command when backup is not found."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_migrator.get_backup_info.return_value = None

            result = self.runner.invoke(admin_app, ["rollback-config", "nonexistent-backup", "--force"])

            assert result.exit_code == 1
            assert "Backup 'nonexistent-backup' not found" in result.stdout

    def test_rollback_config_validation_failed(self):
        """Test rollback-config command when backup validation fails."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_migrator.get_backup_info.return_value = {"timestamp": "2023-01-01T10:00:00"}
            mock_migrator.validate_backup.return_value = False

            result = self.runner.invoke(admin_app, ["rollback-config", "invalid-backup", "--force"])

            assert result.exit_code == 1
            assert "❌ Backup validation failed - cannot rollback" in result.stdout

    def test_backup_info_command(self):
        """Test backup-info command."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_backup_info = {
                "timestamp": "2023-01-01T10:00:00",
                "version": "v1",
                "description": "Test backup",
                "file_path": "/test/backup.json",
                "file_size": 2048,
                "checksum": "abc123def456",
                "config_path": "/test/config.json",
            }
            mock_migrator.get_backup_info.return_value = mock_backup_info
            mock_migrator.validate_backup.return_value = True

            result = self.runner.invoke(admin_app, ["backup-info", "info-test-id"])

            assert result.exit_code == 0
            assert "Backup Information: info-test-id" in result.stdout
            assert "2023-01-01T10:00:00" in result.stdout
            assert "Test backup" in result.stdout
            assert "2048 bytes" in result.stdout
            assert "✅ Backup is valid and can be used for rollback" in result.stdout

    def test_backup_info_not_found(self):
        """Test backup-info command when backup is not found."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_migrator.get_backup_info.return_value = None

            result = self.runner.invoke(admin_app, ["backup-info", "missing-backup"])

            assert result.exit_code == 0
            assert "Backup 'missing-backup' not found" in result.stdout

    def test_cleanup_migration_history_no_cleanup_needed(self):
        """Test cleanup-migration-history command when no cleanup is needed."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_migrator.get_migration_history.return_value = [
                {"migration_id": "keep-1"},
                {"migration_id": "keep-2"},
            ]

            result = self.runner.invoke(admin_app, ["cleanup-migration-history", "--keep", "5", "--force"])

            assert result.exit_code == 0
            assert "No cleanup needed. Current reports: 2, keep: 5" in result.stdout

    def test_cleanup_migration_history_with_cleanup(self):
        """Test cleanup-migration-history command with actual cleanup."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            # Mock 10 existing reports
            mock_history = [{"migration_id": f"report-{i}"} for i in range(10)]
            mock_migrator.get_migration_history.side_effect = [
                mock_history,  # First call for count check
                [{"migration_id": f"report-{i}"} for i in range(3)]  # Second call after cleanup
            ]
            mock_migrator.cleanup_old_migration_reports.return_value = 7

            result = self.runner.invoke(admin_app, ["cleanup-migration-history", "--keep", "3", "--force"])

            assert result.exit_code == 0
            assert "✅ Cleanup completed" in result.stdout
            assert "Removed: 7 migration reports" in result.stdout
            assert "Remaining: 3 reports" in result.stdout
            mock_migrator.cleanup_old_migration_reports.assert_called_once_with(3)

    def test_cli_error_handling(self):
        """Test CLI error handling for migration commands."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator_class.side_effect = Exception("Test error")

            result = self.runner.invoke(admin_app, ["migration-report", "--latest"])

            assert result.exit_code == 1
            assert "Error retrieving migration report: Test error" in result.stdout

    def test_migration_history_limit_parameter(self):
        """Test migration-history command with limit parameter."""
        with patch('python.wqm_cli.cli.commands.admin.ConfigMigrator') as mock_migrator_class:
            mock_migrator = MagicMock()
            mock_migrator_class.return_value = mock_migrator

            mock_history = [{"migration_id": f"limit-test-{i}"} for i in range(5)]
            mock_migrator.get_migration_history.return_value = mock_history

            result = self.runner.invoke(admin_app, ["migration-history", "--limit", "3"])

            assert result.exit_code == 0
            assert "Migration History (5 records)" in result.stdout  # Shows all found
            mock_migrator.get_migration_history.assert_called_once_with(limit=3)