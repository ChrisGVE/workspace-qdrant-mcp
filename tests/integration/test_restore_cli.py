"""
Restore CLI Integration Tests (Task 376.14).

Comprehensive integration tests for restore CLI command including:
- restore command with various options
- Version compatibility validation
- Dry-run mode
- Interactive confirmation
- Error handling and validation
- Integration with RestoreManager
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.core.backup import BackupManager, RestoreManager
from typer.testing import CliRunner
from wqm_cli import __version__
from wqm_cli.cli.main import app


@pytest.fixture
def cli_runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_backup_dir():
    """Create temporary directory for backups."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def compatible_backup(temp_backup_dir):
    """Create a compatible backup for testing."""
    backup_path = temp_backup_dir / "compatible_backup"

    # Create valid backup with current version
    backup_manager = BackupManager(current_version=__version__)
    backup_manager.prepare_backup_directory(backup_path)

    metadata = backup_manager.create_backup_metadata(
        collections={"test-collection": 100},
        total_documents=100,
        description="Compatible test backup"
    )
    backup_manager.save_backup_manifest(metadata, backup_path)

    return backup_path


@pytest.fixture
def incompatible_backup(temp_backup_dir):
    """Create an incompatible backup for testing."""
    backup_path = temp_backup_dir / "incompatible_backup"

    # Create backup with incompatible version (major version bump)
    major, minor, patch = __version__.split('.')
    incompatible_version = f"{int(major) + 1}.0.0"

    backup_manager = BackupManager(current_version=incompatible_version)
    backup_manager.prepare_backup_directory(backup_path)

    metadata = backup_manager.create_backup_metadata(
        collections={"test-collection": 100},
        total_documents=100,
        description="Incompatible test backup"
    )
    backup_manager.save_backup_manifest(metadata, backup_path)

    return backup_path


@pytest.fixture
def newer_patch_backup(temp_backup_dir):
    """Create a newer patch version backup for testing downgrade."""
    from common.core.backup import VersionValidator

    backup_path = temp_backup_dir / "newer_patch_backup"

    # Create backup with newer patch version (same major.minor)
    # Use VersionValidator to properly parse version with dev suffix
    major, minor, patch = VersionValidator.parse_version(__version__)
    newer_version = f"{major}.{minor}.{patch + 1}"

    backup_manager = BackupManager(current_version=newer_version)
    backup_manager.prepare_backup_directory(backup_path)

    metadata = backup_manager.create_backup_metadata(
        collections={"test-collection": 100},
        total_documents=100,
        description="Newer patch version backup"
    )
    backup_manager.save_backup_manifest(metadata, backup_path)

    return backup_path


class TestRestoreCommand:
    """Test restore command functionality."""

    def test_restore_missing_path(self, cli_runner):
        """Test restore command without backup path fails."""
        result = cli_runner.invoke(app, ["backup", "restore"])

        assert result.exit_code != 0
        output = result.output.lower()
        assert "missing argument" in output or "required" in output

    def test_restore_help(self, cli_runner):
        """Test restore help display."""
        result = cli_runner.invoke(app, ["backup", "restore", "--help"])

        assert result.exit_code == 0
        assert "Restore system from backup" in result.output
        assert "--dry-run" in result.output
        assert "--allow-downgrade" in result.output
        assert "--force" in result.output
        assert "--verbose" in result.output

    def test_restore_nonexistent_backup(self, cli_runner, temp_backup_dir):
        """Test restore on non-existent backup."""
        nonexistent = temp_backup_dir / "nonexistent"

        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(nonexistent), "--force"]
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_restore_invalid_structure(self, cli_runner, temp_backup_dir):
        """Test restore on invalid backup structure."""
        invalid_backup = temp_backup_dir / "invalid"
        invalid_backup.mkdir()

        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(invalid_backup), "--force"]
        )

        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "missing" in result.output.lower()

    def test_restore_incompatible_version(self, cli_runner, incompatible_backup):
        """Test restore fails on incompatible version."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(incompatible_backup), "--force"]
        )

        assert result.exit_code != 0
        output = result.output.lower()
        assert "incompatible" in output or "version" in output

    def test_restore_dry_run(self, cli_runner, compatible_backup):
        """Test restore dry-run mode."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup), "--dry-run"]
        )

        # Dry-run should succeed and show plan
        assert result.exit_code == 0
        output = result.output.lower()
        assert "restore plan" in output or "dry-run complete" in output

    def test_restore_with_verbose(self, cli_runner, compatible_backup):
        """Test restore with verbose output."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup), "--dry-run", "--verbose"]
        )

        assert result.exit_code == 0
        # Verbose should show more details
        output = result.output
        assert "compatible" in output.lower()

    def test_restore_newer_patch_rejected(self, cli_runner, newer_patch_backup):
        """Test restore rejects newer patch version by default."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(newer_patch_backup), "--force"]
        )

        assert result.exit_code != 0
        output = result.output.lower()
        assert "downgrade" in output or "newer" in output

    def test_restore_newer_patch_with_allow_downgrade(
        self, cli_runner, newer_patch_backup
    ):
        """Test restore accepts newer patch version with --allow-downgrade."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(newer_patch_backup),
             "--allow-downgrade", "--dry-run"]
        )

        # Should succeed with allow-downgrade flag in dry-run
        assert result.exit_code == 0
        output = result.output.lower()
        assert "restore plan" in output or "dry-run complete" in output

    def test_restore_without_force_requires_confirmation(
        self, cli_runner, compatible_backup
    ):
        """Test restore without --force shows confirmation prompt."""
        # Simulate user declining confirmation by providing 'n'
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup)],
            input="n\n"
        )

        # Should abort when user declines
        output = result.output.lower()
        assert "confirm" in output or "abort" in output or "cancelled" in output


class TestRestoreErrorHandling:
    """Test restore CLI error handling."""

    def test_restore_with_invalid_flags_combination(
        self, cli_runner, compatible_backup
    ):
        """Test restore handles invalid flag combinations gracefully."""
        # Dry-run with force should still work (force is just ignored)
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup),
             "--dry-run", "--force"]
        )

        # Should succeed - dry-run overrides force
        assert result.exit_code == 0

    def test_restore_displays_backup_info(self, cli_runner, compatible_backup):
        """Test restore displays backup information."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup), "--dry-run"]
        )

        assert result.exit_code == 0
        output = result.output
        # Should show backup details
        assert "compatible" in output.lower() or "version" in output.lower()


class TestRestoreIntegration:
    """Test restore CLI integration scenarios."""

    def test_restore_shows_detailed_plan(self, cli_runner, compatible_backup):
        """Test restore shows detailed restore plan."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup), "--dry-run"]
        )

        assert result.exit_code == 0
        output = result.output

        # Should show various components of the restore plan
        # At minimum, should mention compatibility
        assert len(output) > 100  # Non-trivial output

    def test_restore_dry_run_does_not_modify(
        self, cli_runner, compatible_backup, temp_backup_dir
    ):
        """Test dry-run does not actually restore anything."""
        # Create a marker file to verify no changes
        marker = temp_backup_dir / "marker.txt"
        marker.write_text("original")

        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup), "--dry-run"]
        )

        assert result.exit_code == 0
        # Marker should still exist unchanged
        assert marker.exists()
        assert marker.read_text() == "original"

    def test_restore_validates_before_showing_plan(
        self, cli_runner, incompatible_backup
    ):
        """Test restore validates version before showing restore plan."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(incompatible_backup), "--dry-run"]
        )

        # Should fail validation before showing plan
        assert result.exit_code != 0
        output = result.output.lower()
        assert "incompatible" in output or "version" in output


class TestRestoreVerbosity:
    """Test restore verbose output levels."""

    def test_restore_normal_output(self, cli_runner, compatible_backup):
        """Test restore with normal verbosity."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup), "--dry-run"]
        )

        assert result.exit_code == 0
        # Normal output should be concise
        output = result.output
        assert "compatible" in output.lower() or "restore" in output.lower()

    def test_restore_verbose_shows_details(self, cli_runner, compatible_backup):
        """Test restore verbose mode shows additional details."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup),
             "--dry-run", "--verbose"]
        )

        assert result.exit_code == 0
        # Verbose should have more output
        assert len(result.output) > 50


class TestRestoreCompatibilityChecks:
    """Test restore version compatibility checking."""

    def test_restore_checks_version_first(self, cli_runner, incompatible_backup):
        """Test restore validates version before proceeding."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(incompatible_backup), "--force"]
        )

        assert result.exit_code != 0
        # Should fail on version check
        output = result.output.lower()
        assert "version" in output or "incompatible" in output

    def test_restore_allows_compatible_versions(
        self, cli_runner, compatible_backup
    ):
        """Test restore accepts compatible versions."""
        result = cli_runner.invoke(
            app,
            ["backup", "restore", str(compatible_backup), "--dry-run"]
        )

        assert result.exit_code == 0
        output = result.output.lower()
        assert "restore plan" in output or "dry-run complete" in output

    def test_restore_downgrade_requires_flag(self, cli_runner, newer_patch_backup):
        """Test restoring from newer patch version requires explicit flag."""
        # Without flag - should fail
        result1 = cli_runner.invoke(
            app,
            ["backup", "restore", str(newer_patch_backup), "--force"]
        )
        assert result1.exit_code != 0

        # With flag - should succeed in dry-run
        result2 = cli_runner.invoke(
            app,
            ["backup", "restore", str(newer_patch_backup),
             "--allow-downgrade", "--dry-run"]
        )
        assert result2.exit_code == 0
