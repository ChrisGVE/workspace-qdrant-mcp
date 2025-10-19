"""
Unit tests for backup module.

Tests BackupMetadata dataclass and related backup/restore utilities.
"""

import json
import tempfile
import time
from pathlib import Path

import pytest

from common.core.backup import BackupMetadata


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
