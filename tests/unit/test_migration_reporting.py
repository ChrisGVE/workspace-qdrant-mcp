"""Tests for migration reporting and notification system."""

import json
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from common.utils.migration import (
    ChangeEntry,
    ChangeType,
    ConfigMigrator,
    MigrationReport,
    NotificationLevel,
    NotificationSystem,
    ReportGenerator,
    ValidationResult,
)


class TestMigrationReport:
    """Test MigrationReport data class."""

    def test_migration_report_initialization(self):
        """Test basic migration report initialization."""
        report = MigrationReport()

        assert report.migration_id is not None
        assert report.timestamp is not None
        assert report.source_version == "unknown"
        assert report.target_version == "v2_current"
        assert report.success is True
        assert report.changes_made == []
        assert report.deprecated_fields_handled == {}
        assert report.validation_results == []

    def test_add_change(self):
        """Test adding changes to migration report."""
        report = MigrationReport()
        change = ChangeEntry(
            change_type=ChangeType.ADDED,
            field_path="collections.test",
            new_value={"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"},
        )

        report.add_change(change)

        assert len(report.changes_made) == 1
        assert report.changes_made[0] == change

    def test_add_validation_result(self):
        """Test adding validation results."""
        report = MigrationReport()
        validation = ValidationResult(
            is_valid=True, warnings=["Minor warning"], recommendations=["Consider X"]
        )

        report.add_validation_result(validation)

        assert len(report.validation_results) == 1
        assert report.validation_results[0] == validation

    def test_add_warning_and_error(self):
        """Test adding warnings and errors."""
        report = MigrationReport()

        report.add_warning("Test warning")
        report.add_error("Test error")

        assert "Test warning" in report.warnings
        assert "Test error" in report.errors
        assert report.success is False  # Should be set to False when error added

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        report = MigrationReport(
            migration_id="test-id",
            source_version="v1",
            target_version="v2",
        )

        change = ChangeEntry(
            change_type=ChangeType.MODIFIED,
            field_path="test.field",
            old_value="old",
            new_value="new",
        )
        report.add_change(change)

        validation = ValidationResult(is_valid=True, warnings=["warning"])
        report.add_validation_result(validation)

        result = report.to_dict()

        assert isinstance(result, dict)
        assert result["migration_id"] == "test-id"
        assert result["source_version"] == "v1"
        assert result["target_version"] == "v2"
        assert len(result["changes_made"]) == 1
        assert len(result["validation_results"]) == 1


class TestChangeEntry:
    """Test ChangeEntry data class."""

    def test_change_entry_initialization(self):
        """Test basic change entry initialization."""
        change = ChangeEntry(
            change_type=ChangeType.ADDED,
            field_path="test.field",
            new_value="test_value",
        )

        assert change.change_type == ChangeType.ADDED
        assert change.field_path == "test.field"
        assert change.new_value == "test_value"
        assert change.timestamp is not None

    def test_to_dict_serialization(self):
        """Test change entry serialization."""
        change = ChangeEntry(
            change_type=ChangeType.REMOVED,
            field_path="old.field",
            old_value="old_value",
            reason="Deprecated field removed",
        )

        result = change.to_dict()

        assert isinstance(result, dict)
        assert result["change_type"] == "removed"
        assert result["field_path"] == "old.field"
        assert result["old_value"] == "old_value"
        assert result["reason"] == "Deprecated field removed"


class TestReportGenerator:
    """Test ReportGenerator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()
        self.generator = ReportGenerator(self.mock_logger)

    def test_generate_diff(self):
        """Test configuration diff generation."""
        before = {"collections": {"old": {"model": "old-model"}}}
        after = {"collections": {"new": {"model": "new-model"}}}

        diff = self.generator.generate_diff(before, after)

        assert isinstance(diff, str)
        assert "before_migration.json" in diff
        assert "after_migration.json" in diff
        # Should contain unified diff markers
        assert any(line.startswith("-") or line.startswith("+") for line in diff.split("\n"))

    def test_generate_diff_error_handling(self):
        """Test diff generation error handling."""
        # Test with non-serializable objects
        before = {"test": object()}  # Non-serializable object
        after = {"test": "value"}

        diff = self.generator.generate_diff(before, after)

        assert "Error generating diff:" in diff
        self.mock_logger.error.assert_called_once()

    def test_format_report_text(self):
        """Test text report formatting."""
        report = MigrationReport(
            migration_id="test-123",
            source_version="v1",
            target_version="v2",
            success=True,
        )

        # Add some changes
        report.add_change(
            ChangeEntry(
                change_type=ChangeType.ADDED,
                field_path="collections.new_collection",
                new_value={"model": "test"},
                section="collections",
            )
        )

        report.deprecated_fields_handled["old_field"] = "Replaced by new_field"

        formatted = self.generator.format_report_text(report)

        assert isinstance(formatted, str)
        assert "CONFIGURATION MIGRATION REPORT" in formatted
        assert "test-123" in formatted
        assert "SUCCESS" in formatted
        assert "v1 → v2" in formatted
        assert "collections.new_collection" in formatted
        assert "old_field" in formatted

    def test_format_report_json(self):
        """Test JSON report formatting."""
        report = MigrationReport(migration_id="test-456")

        formatted = self.generator.format_report_json(report)

        assert isinstance(formatted, str)
        data = json.loads(formatted)  # Should be valid JSON
        assert data["migration_id"] == "test-456"

    def test_format_report_json_error(self):
        """Test JSON formatting error handling."""
        with patch("json.dumps", side_effect=TypeError("Test error")):
            report = MigrationReport()
            formatted = self.generator.format_report_json(report)

            data = json.loads(formatted)
            assert "error" in data


class TestNotificationSystem:
    """Test NotificationSystem functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_logger = MagicMock()
        self.notifier = NotificationSystem(self.mock_logger)

    def test_notify_migration_started(self):
        """Test migration start notification."""
        self.notifier.notify_migration_started("test-id", "v1", "v2")

        self.mock_logger.info.assert_any_call("🚀 Starting configuration migration test-id")
        self.mock_logger.info.assert_any_call("   Migrating from v1 to v2")

    def test_notify_migration_success(self):
        """Test successful migration notification."""
        report = MigrationReport(
            migration_id="success-id",
            backup_id="backup-123",
            changes_made=[
                ChangeEntry(ChangeType.ADDED, "test.field"),
                ChangeEntry(ChangeType.REMOVED, "old.field"),
            ],
            warnings=["Minor warning"],
            deprecated_fields_handled={"old": "Replaced by new"},
        )

        self.notifier.notify_migration_success(report)

        # Check that success message was logged
        self.mock_logger.info.assert_any_call("✅ Configuration migration completed successfully")
        self.mock_logger.info.assert_any_call("   Migration ID: success-id")
        self.mock_logger.info.assert_any_call("   Changes made: 2")
        self.mock_logger.info.assert_any_call("   Configuration backed up: backup-123")
        self.mock_logger.warning.assert_any_call("   ⚠️  1 warnings (see report for details)")

    def test_notify_migration_failure(self):
        """Test failed migration notification."""
        report = MigrationReport(
            migration_id="failure-id",
            backup_id="backup-456",
            success=False,
            errors=["Critical error 1", "Critical error 2", "Error 3", "Error 4"],
        )

        self.notifier.notify_migration_failure(report)

        self.mock_logger.error.assert_any_call("❌ Configuration migration failed")
        self.mock_logger.error.assert_any_call("   Migration ID: failure-id")
        # Should show first 3 errors
        self.mock_logger.error.assert_any_call("     • Critical error 1")
        self.mock_logger.error.assert_any_call("     • Critical error 2")
        self.mock_logger.error.assert_any_call("     • Error 3")
        self.mock_logger.error.assert_any_call("     ... and 1 more errors")

    def test_notify_deprecated_features(self):
        """Test deprecated features notification."""
        deprecated_fields = {
            "old.field1": "Replaced by new.field1",
            "old.field2": "Removed (no replacement)",
        }

        self.notifier.notify_deprecated_features(deprecated_fields)

        self.mock_logger.warning.assert_any_call("🔄 Deprecated configuration fields updated:")
        self.mock_logger.warning.assert_any_call("   • old.field1 → Replaced by new.field1")
        self.mock_logger.warning.assert_any_call("   • old.field2 → Removed (no replacement)")

    def test_format_notification(self):
        """Test notification formatting."""
        message = self.notifier.format_notification(
            NotificationLevel.WARNING,
            "Test message",
            ["Detail 1", "Detail 2"]
        )

        assert "⚠️ Test message" in message
        assert "   • Detail 1" in message
        assert "   • Detail 2" in message


class TestConfigMigratorReporting:
    """Test ConfigMigrator reporting functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        self.mock_logger = MagicMock()
        self.migrator = ConfigMigrator(logger=self.mock_logger)
        self.migrator.set_migration_history_dir(self.temp_path / "migration_history")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_migration_history_dir_property(self):
        """Test migration history directory property."""
        history_dir = self.migrator.migration_history_dir

        assert isinstance(history_dir, Path)
        assert history_dir.exists()
        assert history_dir.name == "migration_history"

    def test_generate_migration_report_basic(self):
        """Test basic migration report generation."""
        before_config = {"collections": {"old": {"model": "old-model"}}}
        after_config = {"collections": {"new": {"model": "new-model"}}}

        report = self.migrator.generate_migration_report(
            before_config=before_config,
            after_config=after_config,
            migration_methods=["test_method"],
        )

        assert isinstance(report, MigrationReport)
        assert report.source_version == "unknown"
        assert report.target_version == "v2_current"
        assert "test_method" in report.migration_methods_used
        assert len(report.changes_made) > 0

    def test_generate_migration_report_with_backup(self):
        """Test report generation with backup information."""
        before_config = {"test": "value"}
        after_config = {"test": "new_value"}

        # Mock backup info
        with patch.object(self.migrator, 'get_backup_info', return_value={
            "file_path": "/test/backup.json",
            "timestamp": "2023-01-01T00:00:00",
        }):
            report = self.migrator.generate_migration_report(
                before_config=before_config,
                after_config=after_config,
                backup_id="test-backup",
                migration_duration=1.5,
            )

        assert report.backup_id == "test-backup"
        assert report.backup_location == "/test/backup.json"
        assert report.migration_duration_seconds == 1.5

    def test_analyze_configuration_changes(self):
        """Test configuration change analysis."""
        before_config = {
            "collections": {"old_col": {"model": "old"}},
            "patterns": {"include": ["*.py"]},
        }
        after_config = {
            "collections": {"new_col": {"model": "new"}},
            "patterns": {"include": ["*.py", "*.js"]},
        }

        report = MigrationReport()
        self.migrator._analyze_configuration_changes(before_config, after_config, report)

        assert len(report.changes_made) > 0

        # Check for removed collection
        removed_changes = [c for c in report.changes_made if c.change_type == ChangeType.REMOVED]
        assert any("old_col" in c.field_path for c in removed_changes)

        # Check for added collection
        added_changes = [c for c in report.changes_made if c.change_type == ChangeType.ADDED]
        assert any("new_col" in c.field_path for c in added_changes)

    def test_save_and_load_migration_report(self):
        """Test saving and loading migration reports."""
        original_report = MigrationReport(
            migration_id="test-save-load",
            source_version="v1",
            target_version="v2",
        )
        original_report.add_change(
            ChangeEntry(ChangeType.ADDED, "test.field", new_value="value")
        )

        # Save the report
        self.migrator._save_migration_report(original_report)

        # Load the report
        loaded_report = self.migrator.get_migration_report("test-save-load")

        assert loaded_report is not None
        assert loaded_report.migration_id == "test-save-load"
        assert loaded_report.source_version == "v1"
        assert loaded_report.target_version == "v2"
        assert len(loaded_report.changes_made) == 1
        assert loaded_report.changes_made[0].field_path == "test.field"

    def test_get_migration_history(self):
        """Test getting migration history."""
        # Create multiple reports
        for i in range(5):
            report = MigrationReport(migration_id=f"test-{i}")
            self.migrator._save_migration_report(report)

        history = self.migrator.get_migration_history()

        assert len(history) == 5
        # Should be sorted by timestamp (most recent first)
        assert all("migration_id" in entry for entry in history)

    def test_get_migration_history_with_limit(self):
        """Test getting migration history with limit."""
        # Create multiple reports
        for i in range(10):
            report = MigrationReport(migration_id=f"limited-test-{i}")
            self.migrator._save_migration_report(report)

        history = self.migrator.get_migration_history(limit=3)

        assert len(history) == 3

    def test_search_migration_history(self):
        """Test searching migration history with filters."""
        # Create reports with different characteristics
        success_report = MigrationReport(
            migration_id="success-test",
            source_version="v1",
            success=True,
        )
        self.migrator._save_migration_report(success_report)

        failure_report = MigrationReport(
            migration_id="failure-test",
            source_version="v2",
            success=False,
        )
        self.migrator._save_migration_report(failure_report)

        # Search by success status
        successful_migrations = self.migrator.search_migration_history(success_only=True)
        assert len(successful_migrations) == 1
        assert successful_migrations[0]["migration_id"] == "success-test"

        # Search by source version
        v2_migrations = self.migrator.search_migration_history(source_version="v2")
        assert len(v2_migrations) == 1
        assert v2_migrations[0]["migration_id"] == "failure-test"

    def test_cleanup_old_migration_reports(self):
        """Test cleaning up old migration reports."""
        # Create multiple reports
        report_ids = []
        for i in range(10):
            report = MigrationReport(migration_id=f"cleanup-test-{i}")
            self.migrator._save_migration_report(report)
            report_ids.append(f"cleanup-test-{i}")

        # Cleanup, keeping only 3 most recent
        removed_count = self.migrator.cleanup_old_migration_reports(keep_count=3)

        assert removed_count == 7

        # Check that only 3 remain
        remaining_history = self.migrator.get_migration_history()
        assert len(remaining_history) == 3

    def test_get_latest_migration_report(self):
        """Test getting the latest migration report."""
        # Initially no reports
        latest = self.migrator.get_latest_migration_report()
        assert latest is None

        # Add a report
        report = MigrationReport(migration_id="latest-test")
        self.migrator._save_migration_report(report)

        latest = self.migrator.get_latest_migration_report()
        assert latest is not None
        assert latest.migration_id == "latest-test"

    def test_generate_migration_report_error_handling(self):
        """Test error handling in migration report generation."""
        with pytest.raises(ValueError, match="must be dictionaries"):
            self.migrator.generate_migration_report("not a dict", {})

        with pytest.raises(ValueError, match="must be dictionaries"):
            self.migrator.generate_migration_report({}, "not a dict")

    def test_field_replacement_info(self):
        """Test deprecated field replacement information."""
        replacement = self.migrator._get_field_replacement_info("collection_prefix")
        assert "collections.project_suffixes" in replacement

        replacement = self.migrator._get_field_replacement_info("unknown_field")
        assert "See documentation for replacement" in replacement

    def test_generate_rollback_instructions(self):
        """Test rollback instruction generation."""
        instructions = self.migrator._generate_rollback_instructions("test-backup-id")

        assert "wqm admin rollback-config test-backup-id" in instructions
        assert "wqm admin validate-config" in instructions
        assert "wqm admin validate-backup test-backup-id" in instructions

    def test_dict_to_migration_report_conversion(self):
        """Test converting dictionary back to MigrationReport object."""
        # Create original report
        original = MigrationReport(migration_id="conversion-test")
        original.add_change(
            ChangeEntry(ChangeType.MODIFIED, "test.field", "old", "new")
        )
        original.add_validation_result(
            ValidationResult(is_valid=True, warnings=["warning"])
        )

        # Convert to dict and back
        dict_data = original.to_dict()
        converted = self.migrator._dict_to_migration_report(dict_data)

        assert converted.migration_id == original.migration_id
        assert len(converted.changes_made) == len(original.changes_made)
        assert len(converted.validation_results) == len(original.validation_results)
        assert converted.changes_made[0].field_path == "test.field"

    @patch('workspace_qdrant_mcp.utils.migration.time.time')
    def test_migrate_with_backup_integration(self, mock_time):
        """Test migration with backup includes comprehensive reporting."""
        mock_time.side_effect = [1000.0, 1002.5]  # Start and end times

        # Mock the backup creation
        with patch.object(self.migrator, 'backup_config', return_value="test-backup-id"):
            with patch.object(self.migrator, 'get_backup_info', return_value={
                "file_path": "/test/backup.json"
            }):
                # Mock individual migration methods to return modified config
                original_config = {"collections": {"old": "value"}}
                modified_config = {"collections": {"new": "value"}}

                with patch.object(self.migrator, 'migrate_collection_config', return_value=modified_config):
                    with patch.object(self.migrator, 'migrate_pattern_config', return_value=modified_config):
                        with patch.object(self.migrator, 'remove_deprecated_fields', return_value=modified_config):
                            with patch.object(self.migrator, '_validate_cleaned_config', return_value={"is_valid": True, "error": None}):

                                result = self.migrator.migrate_with_backup(original_config)

                                assert result == modified_config

                                # Check that a migration report was created
                                latest_report = self.migrator.get_latest_migration_report()
                                assert latest_report is not None
                                assert latest_report.success is True
                                assert latest_report.backup_id == "test-backup-id"
                                assert latest_report.migration_duration_seconds == 2.5