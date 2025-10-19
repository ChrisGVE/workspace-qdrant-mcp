"""
Unit tests for backup module.

Tests BackupMetadata dataclass and related backup/restore utilities.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from common.core.backup import BackupMetadata, VersionValidator, CompatibilityStatus
from common.core.error_handling import IncompatibleVersionError


class TestBackupMetadata:
    """Test suite for BackupMetadata dataclass."""

    def test_backup_metadata_minimal(self):
        """Test BackupMetadata with only required fields."""
        metadata = BackupMetadata(
            version="0.2.1",
            timestamp=time.time()
        )

        assert metadata.version == "0.2.1"
        assert metadata.timestamp > 0
        assert metadata.collections is None
        assert metadata.total_documents is None

    def test_backup_metadata_full(self):
        """Test BackupMetadata with all fields populated."""
        now = time.time()
        metadata = BackupMetadata(
            version="0.2.1",
            timestamp=now,
            collections=["project-code", "project-docs"],
            total_documents=100,
            database_version="3.41.0",
            python_version="3.12.9",
            during_operations=True,
            partial_backup=False,
            description="Full system backup",
            created_by="admin",
            backup_path="/backups/2024-01-01"
        )

        assert metadata.version == "0.2.1"
        assert metadata.timestamp == now
        assert metadata.collections == ["project-code", "project-docs"]
        assert metadata.total_documents == 100
        assert metadata.database_version == "3.41.0"
        assert metadata.during_operations is True

    def test_backup_metadata_collections_as_dict(self):
        """Test BackupMetadata with collections as count dictionary."""
        metadata = BackupMetadata(
            version="0.2.1",
            timestamp=time.time(),
            collections={"project-code": 50, "project-docs": 30}
        )

        assert isinstance(metadata.collections, dict)
        assert metadata.collections["project-code"] == 50
        assert metadata.collections["project-docs"] == 30

    def test_backup_metadata_to_dict(self):
        """Test conversion to dictionary."""
        metadata = BackupMetadata(
            version="0.2.1",
            timestamp=1234567890.0,
            collections=["test-collection"],
            total_documents=10
        )

        data = metadata.to_dict()

        assert data["version"] == "0.2.1"
        assert data["timestamp"] == 1234567890.0
        assert data["collections"] == ["test-collection"]
        assert data["total_documents"] == 10
        # None values should be excluded
        assert "database_version" not in data
        assert "python_version" not in data

    def test_backup_metadata_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "version": "0.2.1",
            "timestamp": 1234567890.0,
            "collections": ["test"],
            "total_documents": 5
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.version == "0.2.1"
        assert metadata.timestamp == 1234567890.0
        assert metadata.collections == ["test"]
        assert metadata.total_documents == 5

    def test_backup_metadata_from_dict_with_additional_fields(self):
        """Test from_dict with extra fields stored in additional_metadata."""
        data = {
            "version": "0.2.1",
            "timestamp": time.time(),
            "custom_field": "custom_value",
            "another_field": 123
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.version == "0.2.1"
        assert metadata.additional_metadata["custom_field"] == "custom_value"
        assert metadata.additional_metadata["another_field"] == 123

    def test_backup_metadata_save_and_load(self):
        """Test saving to and loading from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "manifest.json"

            # Create and save metadata
            original = BackupMetadata(
                version="0.2.1",
                timestamp=1234567890.0,
                collections=["test-collection"],
                total_documents=10,
                description="Test backup"
            )
            original.save_to_file(file_path)

            # Verify file exists and contains valid JSON
            assert file_path.exists()
            data = json.loads(file_path.read_text())
            assert data["version"] == "0.2.1"

            # Load and verify
            loaded = BackupMetadata.load_from_file(file_path)
            assert loaded.version == original.version
            assert loaded.timestamp == original.timestamp
            assert loaded.collections == original.collections
            assert loaded.total_documents == original.total_documents
            assert loaded.description == original.description

    def test_backup_metadata_save_creates_directory(self):
        """Test that save_to_file creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "path" / "manifest.json"

            metadata = BackupMetadata(version="0.2.1", timestamp=time.time())
            metadata.save_to_file(nested_path)

            assert nested_path.exists()
            assert nested_path.parent.exists()

    def test_backup_metadata_get_version_tuple(self):
        """Test parsing version string into tuple."""
        metadata = BackupMetadata(version="0.2.1", timestamp=time.time())
        version_tuple = metadata.get_version_tuple()

        assert version_tuple == (0, 2, 1)

    def test_backup_metadata_get_version_tuple_dev_version(self):
        """Test parsing development version with suffix."""
        metadata = BackupMetadata(version="0.2.1dev1", timestamp=time.time())
        version_tuple = metadata.get_version_tuple()

        # Should extract numeric part only
        assert version_tuple == (0, 2, 1)

    def test_backup_metadata_get_version_tuple_no_patch(self):
        """Test parsing version without patch number."""
        metadata = BackupMetadata(version="0.2", timestamp=time.time())
        version_tuple = metadata.get_version_tuple()

        assert version_tuple == (0, 2, 0)

    def test_backup_metadata_get_version_tuple_invalid(self):
        """Test parsing invalid version string raises ValueError."""
        metadata = BackupMetadata(version="invalid", timestamp=time.time())

        with pytest.raises(ValueError, match="Invalid version format"):
            metadata.get_version_tuple()

    def test_backup_metadata_format_timestamp(self):
        """Test timestamp formatting."""
        # Use known timestamp for reproducibility
        metadata = BackupMetadata(
            version="0.2.1",
            timestamp=1234567890.0  # 2009-02-13 23:31:30 UTC
        )

        formatted = metadata.format_timestamp()

        # Should be ISO format
        assert "2009-02-13" in formatted
        assert "T" in formatted  # ISO format separator

    def test_backup_metadata_partial_backup_fields(self):
        """Test partial backup specific fields."""
        metadata = BackupMetadata(
            version="0.2.1",
            timestamp=time.time(),
            partial_backup=True,
            selected_collections=["project-code", "project-docs"]
        )

        assert metadata.partial_backup is True
        assert metadata.selected_collections == ["project-code", "project-docs"]

    def test_backup_metadata_during_operations_flag(self):
        """Test during_operations flag."""
        metadata = BackupMetadata(
            version="0.2.1",
            timestamp=time.time(),
            during_operations=True
        )

        assert metadata.during_operations is True

    def test_backup_metadata_load_from_missing_file(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            BackupMetadata.load_from_file("/nonexistent/path/manifest.json")

    def test_backup_metadata_load_from_invalid_json(self):
        """Test loading from file with invalid JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "invalid.json"
            file_path.write_text("not valid json {}")

            with pytest.raises(json.JSONDecodeError):
                BackupMetadata.load_from_file(file_path)

    def test_backup_metadata_from_dict_missing_required_fields(self):
        """Test from_dict with missing required fields raises KeyError."""
        with pytest.raises(TypeError):  # Missing required positional argument
            BackupMetadata.from_dict({"timestamp": time.time()})  # Missing version

    def test_backup_metadata_roundtrip_with_additional_metadata(self):
        """Test save/load preserves additional metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "manifest.json"

            original = BackupMetadata(
                version="0.2.1",
                timestamp=time.time(),
                additional_metadata={"custom_key": "custom_value", "number": 42}
            )
            original.save_to_file(file_path)

            loaded = BackupMetadata.load_from_file(file_path)

            assert loaded.additional_metadata["custom_key"] == "custom_value"
            assert loaded.additional_metadata["number"] == 42


class TestVersionValidator:
    """Test suite for VersionValidator utility class."""

    def test_parse_version_standard(self):
        """Test parsing standard semver version."""
        major, minor, patch = VersionValidator.parse_version("0.2.1")
        assert (major, minor, patch) == (0, 2, 1)

    def test_parse_version_dev_suffix(self):
        """Test parsing version with dev suffix."""
        major, minor, patch = VersionValidator.parse_version("0.2.1dev1")
        assert (major, minor, patch) == (0, 2, 1)

    def test_parse_version_no_patch(self):
        """Test parsing version without patch number."""
        major, minor, patch = VersionValidator.parse_version("1.0")
        assert (major, minor, patch) == (1, 0, 0)

    def test_parse_version_invalid(self):
        """Test parsing invalid version raises ValueError."""
        with pytest.raises(ValueError, match="Invalid version format"):
            VersionValidator.parse_version("invalid")

    def test_check_compatibility_identical(self):
        """Test identical versions are compatible."""
        status = VersionValidator.check_compatibility("0.2.1", "0.2.1")
        assert status == CompatibilityStatus.COMPATIBLE

    def test_check_compatibility_patch_difference(self):
        """Test same major.minor with different patch is compatible."""
        status = VersionValidator.check_compatibility("0.2.0", "0.2.1")
        assert status == CompatibilityStatus.UPGRADE_AVAILABLE

    def test_check_compatibility_downgrade(self):
        """Test newer backup version is downgrade."""
        status = VersionValidator.check_compatibility("0.2.2", "0.2.1")
        assert status == CompatibilityStatus.DOWNGRADE

    def test_check_compatibility_minor_version_mismatch(self):
        """Test different minor version is incompatible."""
        status = VersionValidator.check_compatibility("0.1.0", "0.2.0")
        assert status == CompatibilityStatus.INCOMPATIBLE

    def test_check_compatibility_major_version_mismatch(self):
        """Test different major version is incompatible."""
        status = VersionValidator.check_compatibility("1.0.0", "2.0.0")
        assert status == CompatibilityStatus.INCOMPATIBLE

    def test_check_compatibility_dev_versions(self):
        """Test compatibility with dev suffixes."""
        status = VersionValidator.check_compatibility("0.2.1dev1", "0.2.1")
        assert status == CompatibilityStatus.COMPATIBLE

    def test_validate_compatibility_success(self):
        """Test validate_compatibility doesn't raise on compatible versions."""
        metadata = BackupMetadata(version="0.2.1", timestamp=time.time())
        # Should not raise
        VersionValidator.validate_compatibility(metadata, "0.2.1")

    def test_validate_compatibility_upgrade_allowed(self):
        """Test validate_compatibility allows patch upgrades."""
        metadata = BackupMetadata(version="0.2.0", timestamp=time.time())
        # Should not raise (backup is older)
        VersionValidator.validate_compatibility(metadata, "0.2.1")

    def test_validate_compatibility_incompatible_raises(self):
        """Test validate_compatibility raises on incompatible versions."""
        metadata = BackupMetadata(version="0.1.0", timestamp=time.time())

        with pytest.raises(IncompatibleVersionError) as exc_info:
            VersionValidator.validate_compatibility(metadata, "0.2.0")

        error = exc_info.value
        assert "incompatible" in str(error).lower()
        assert error.context["backup_version"] == "0.1.0"
        assert error.context["current_version"] == "0.2.0"

    def test_validate_compatibility_downgrade_blocked(self):
        """Test validate_compatibility blocks downgrade by default."""
        metadata = BackupMetadata(version="0.2.2", timestamp=time.time())

        with pytest.raises(IncompatibleVersionError) as exc_info:
            VersionValidator.validate_compatibility(metadata, "0.2.1")

        error = exc_info.value
        assert "newer backup version" in str(error).lower()

    def test_validate_compatibility_downgrade_allowed(self):
        """Test validate_compatibility allows downgrade with flag."""
        metadata = BackupMetadata(version="0.2.2", timestamp=time.time())
        # Should not raise with allow_downgrade=True
        VersionValidator.validate_compatibility(metadata, "0.2.1", allow_downgrade=True)

    def test_get_compatibility_message_compatible(self):
        """Test compatibility message for compatible versions."""
        message = VersionValidator.get_compatibility_message("0.2.1", "0.2.1")
        assert "compatible" in message.lower()
        assert "0.2.1" in message

    def test_get_compatibility_message_upgrade(self):
        """Test compatibility message for upgrade available."""
        message = VersionValidator.get_compatibility_message("0.2.0", "0.2.1")
        assert "older" in message.lower()
        assert "0.2.0" in message
        assert "0.2.1" in message

    def test_get_compatibility_message_downgrade(self):
        """Test compatibility message for downgrade."""
        message = VersionValidator.get_compatibility_message("0.2.2", "0.2.1")
        assert "newer" in message.lower()

    def test_get_compatibility_message_incompatible(self):
        """Test compatibility message for incompatible versions."""
        message = VersionValidator.get_compatibility_message("0.1.0", "0.2.0")
        assert "incompatible" in message.lower()
        assert "major or minor" in message.lower()

    def test_version_validator_with_real_metadata(self):
        """Test VersionValidator with actual BackupMetadata."""
        metadata = BackupMetadata(
            version="0.2.1",
            timestamp=time.time(),
            collections=["test"]
        )

        # Compatible version should pass
        VersionValidator.validate_compatibility(metadata, "0.2.1")

        # Incompatible version should fail
        with pytest.raises(IncompatibleVersionError):
            VersionValidator.validate_compatibility(metadata, "0.3.0")
