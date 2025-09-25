"""Comprehensive unit tests for versioning system with edge cases."""

import pytest
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import shutil

from docs.framework.deployment.versioning import (
    VersionManager,
    Version,
    VersioningStrategy,
    VersionStatus,
    VersioningConfig
)


class TestVersion:
    """Test Version data class."""

    def test_version_initialization(self):
        """Test basic version initialization."""
        version = Version(
            version_string="1.0.0",
            strategy=VersioningStrategy.SEMANTIC
        )

        assert version.version_string == "1.0.0"
        assert version.strategy == VersioningStrategy.SEMANTIC
        assert version.status == VersionStatus.DRAFT
        assert isinstance(version.created_at, datetime)

    def test_version_with_metadata(self):
        """Test version with full metadata."""
        created_at = datetime.now()
        deployed_at = created_at + timedelta(hours=1)

        version = Version(
            version_string="2.1.0",
            strategy=VersioningStrategy.SEMANTIC,
            status=VersionStatus.ACTIVE,
            created_at=created_at,
            deployed_at=deployed_at,
            title="Major Release",
            description="New features and improvements",
            author="test@example.com",
            git_commit="abc123",
            size_bytes=1024,
            file_count=42
        )

        assert version.title == "Major Release"
        assert version.deployed_at == deployed_at
        assert version.git_commit == "abc123"
        assert version.size_bytes == 1024
        assert version.file_count == 42

    def test_version_string_deployment_path(self):
        """Test version with string deployment path."""
        version = Version(
            version_string="1.0.0",
            strategy=VersioningStrategy.SEMANTIC,
            deployment_path="/deploy/path"
        )

        assert isinstance(version.deployment_path, Path)
        assert version.deployment_path == Path("/deploy/path")

    def test_version_to_dict(self):
        """Test version serialization to dictionary."""
        created_at = datetime.now()
        version = Version(
            version_string="1.0.0",
            strategy=VersioningStrategy.SEMANTIC,
            status=VersionStatus.ACTIVE,
            created_at=created_at,
            deployment_path=Path("/deploy")
        )

        data = version.to_dict()

        assert data['version_string'] == "1.0.0"
        assert data['strategy'] == "semantic"
        assert data['status'] == "active"
        assert data['created_at'] == created_at.isoformat()
        assert data['deployment_path'] == "/deploy"

    def test_version_from_dict(self):
        """Test version deserialization from dictionary."""
        created_at = datetime.now()
        data = {
            'version_string': '1.0.0',
            'strategy': 'semantic',
            'status': 'active',
            'created_at': created_at.isoformat(),
            'deployment_path': '/deploy',
            'title': 'Test Version'
        }

        version = Version.from_dict(data)

        assert version.version_string == "1.0.0"
        assert version.strategy == VersioningStrategy.SEMANTIC
        assert version.status == VersionStatus.ACTIVE
        assert version.created_at == created_at
        assert version.deployment_path == Path("/deploy")
        assert version.title == "Test Version"

    def test_version_properties(self):
        """Test version properties."""
        # Active version
        active_version = Version("1.0.0", VersioningStrategy.SEMANTIC, VersionStatus.ACTIVE)
        assert active_version.is_active
        assert not active_version.is_deprecated

        # Deprecated version
        deprecated_version = Version("0.9.0", VersioningStrategy.SEMANTIC, VersionStatus.DEPRECATED)
        assert not deprecated_version.is_active
        assert deprecated_version.is_deprecated

    def test_version_age_calculation(self):
        """Test version age calculation."""
        old_date = datetime.now() - timedelta(days=5)
        version = Version("1.0.0", VersioningStrategy.SEMANTIC, created_at=old_date)

        assert version.age_days >= 4  # Account for small timing differences

    def test_compare_semantic_versions(self):
        """Test semantic version comparison."""
        v1 = Version("1.0.0", VersioningStrategy.SEMANTIC)
        v2 = Version("1.0.1", VersioningStrategy.SEMANTIC)
        v3 = Version("1.1.0", VersioningStrategy.SEMANTIC)
        v4 = Version("2.0.0", VersioningStrategy.SEMANTIC)

        assert v1.compare_version(v2) == -1  # v1 < v2
        assert v2.compare_version(v1) == 1   # v2 > v1
        assert v1.compare_version(v1) == 0   # v1 == v1
        assert v2.compare_version(v3) == -1  # v2 < v3
        assert v3.compare_version(v4) == -1  # v3 < v4

    def test_compare_semantic_versions_with_prefix(self):
        """Test semantic version comparison with 'v' prefix."""
        v1 = Version("v1.0.0", VersioningStrategy.SEMANTIC)
        v2 = Version("v1.0.1", VersioningStrategy.SEMANTIC)

        assert v1.compare_version(v2) == -1

    def test_compare_sequential_versions(self):
        """Test sequential version comparison."""
        v1 = Version("v1", VersioningStrategy.SEQUENTIAL)
        v2 = Version("v2", VersioningStrategy.SEQUENTIAL)
        v10 = Version("v10", VersioningStrategy.SEQUENTIAL)

        assert v1.compare_version(v2) == -1
        assert v2.compare_version(v10) == -1
        assert v10.compare_version(v1) == 1

    def test_compare_timestamp_versions(self):
        """Test timestamp version comparison."""
        v1 = Version("20240101-120000", VersioningStrategy.TIMESTAMP)
        v2 = Version("20240101-130000", VersioningStrategy.TIMESTAMP)

        assert v1.compare_version(v2) == -1

    def test_compare_different_strategies(self):
        """Test comparison between different versioning strategies."""
        now = datetime.now()
        later = now + timedelta(minutes=30)

        v1 = Version("1.0.0", VersioningStrategy.SEMANTIC, created_at=now)
        v2 = Version("v1", VersioningStrategy.SEQUENTIAL, created_at=later)

        assert v1.compare_version(v2) == -1  # Falls back to timestamp

    def test_compare_invalid_semantic_versions(self):
        """Test comparison of invalid semantic versions."""
        v1 = Version("invalid", VersioningStrategy.SEMANTIC)
        v2 = Version("also-invalid", VersioningStrategy.SEMANTIC)

        result = v1.compare_version(v2)
        assert result in [-1, 0, 1]  # Should handle gracefully


class TestVersioningConfig:
    """Test VersioningConfig data class."""

    def test_versioning_config_defaults(self):
        """Test default configuration values."""
        config = VersioningConfig()

        assert config.strategy == VersioningStrategy.SEMANTIC
        assert config.auto_increment is True
        assert config.max_versions == 50
        assert config.use_git_info is True

    def test_versioning_config_custom(self):
        """Test custom configuration."""
        config = VersioningConfig(
            strategy=VersioningStrategy.TIMESTAMP,
            max_versions=10,
            auto_archive_days=30,
            custom_format="v{date}-{time}"
        )

        assert config.strategy == VersioningStrategy.TIMESTAMP
        assert config.max_versions == 10
        assert config.auto_archive_days == 30
        assert config.custom_format == "v{date}-{time}"

    def test_versioning_config_version_file_path(self):
        """Test version file path conversion."""
        config = VersioningConfig(version_file="/path/to/version.txt")

        assert isinstance(config.version_file, Path)
        assert config.version_file == Path("/path/to/version.txt")


class TestVersionManager:
    """Test VersionManager with comprehensive edge cases."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def temp_source(self):
        """Create temporary source directory with files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir)
            (source_path / "index.html").write_text("<html>Test</html>")
            (source_path / "style.css").write_text("body { color: red; }")
            (source_path / "subdir").mkdir()
            (source_path / "subdir" / "page.html").write_text("<html>Subpage</html>")
            yield source_path

    @pytest.fixture
    def version_manager(self, temp_storage):
        """Create version manager for testing."""
        return VersionManager(temp_storage)

    def test_version_manager_initialization(self, temp_storage):
        """Test version manager initialization."""
        config = VersioningConfig(strategy=VersioningStrategy.TIMESTAMP)
        manager = VersionManager(temp_storage, config)

        assert manager.storage_path == temp_storage
        assert manager.config.strategy == VersioningStrategy.TIMESTAMP
        assert manager._versions_file == temp_storage / "versions.json"
        assert temp_storage.exists()

    def test_version_manager_string_storage_path(self, temp_storage):
        """Test version manager with string storage path."""
        manager = VersionManager(str(temp_storage))

        assert isinstance(manager.storage_path, Path)
        assert manager.storage_path == temp_storage

    def test_create_version_basic(self, version_manager, temp_source):
        """Test basic version creation."""
        version = version_manager.create_version(temp_source)

        assert version.version_string == "1.0.0"  # Default semantic
        assert version.strategy == VersioningStrategy.SEMANTIC
        assert version.status == VersionStatus.DRAFT
        assert version.file_count == 3
        assert version.size_bytes > 0
        assert version.source_hash is not None

    def test_create_version_explicit_string(self, version_manager, temp_source):
        """Test version creation with explicit version string."""
        version = version_manager.create_version(
            temp_source,
            version_string="2.1.0",
            title="Custom Version",
            description="Test description"
        )

        assert version.version_string == "2.1.0"
        assert version.title == "Custom Version"
        assert version.description == "Test description"

    def test_create_version_nonexistent_source(self, version_manager):
        """Test version creation with nonexistent source."""
        with pytest.raises(ValueError, match="does not exist"):
            version_manager.create_version(Path("/nonexistent/path"))

    @patch('subprocess.run')
    def test_create_version_with_git_info(self, mock_run, version_manager, temp_source):
        """Test version creation with Git information."""
        # Mock git commands
        mock_run.side_effect = [
            Mock(stdout="abc123def456\n", returncode=0),  # commit hash
            Mock(stdout="main\n", returncode=0),          # branch
            Mock(stdout="Test Author\n", returncode=0)    # author
        ]

        version = version_manager.create_version(temp_source)

        assert version.git_commit == "abc123def456"
        assert version.git_branch == "main"
        assert version.author == "Test Author"

    @patch('subprocess.run')
    def test_create_version_git_unavailable(self, mock_run, version_manager, temp_source):
        """Test version creation when Git is unavailable."""
        mock_run.side_effect = subprocess.CalledProcessError(128, ['git'])

        version = version_manager.create_version(temp_source)

        # Should still create version without Git info
        assert version.git_commit is None
        assert version.git_branch is None

    def test_create_version_git_disabled(self, temp_storage, temp_source):
        """Test version creation with Git info disabled."""
        config = VersioningConfig(use_git_info=False)
        manager = VersionManager(temp_storage, config)

        version = manager.create_version(temp_source)

        assert version.git_commit is None
        assert version.git_branch is None

    def test_deploy_version_success(self, version_manager, temp_source, temp_storage):
        """Test successful version deployment."""
        version = version_manager.create_version(temp_source)
        deploy_path = temp_storage / "deploy"
        deploy_path.mkdir()

        success = version_manager.deploy_version(
            version.version_string,
            deploy_path,
            deployment_id="deploy_123"
        )

        assert success
        assert version.status == VersionStatus.ACTIVE
        assert version.deployed_at is not None
        assert version.deployment_path == deploy_path
        assert version.deployment_id == "deploy_123"

    def test_deploy_version_nonexistent(self, version_manager, temp_storage):
        """Test deploying nonexistent version."""
        with pytest.raises(ValueError, match="Version not found"):
            version_manager.deploy_version("nonexistent", temp_storage / "deploy")

    def test_deploy_version_deactivates_current(self, version_manager, temp_source, temp_storage):
        """Test deploying version deactivates current active version."""
        # Create and deploy first version
        version1 = version_manager.create_version(temp_source, "1.0.0")
        version_manager.deploy_version(version1.version_string, temp_storage / "deploy1")

        # Create and deploy second version
        version2 = version_manager.create_version(temp_source, "2.0.0")
        version_manager.deploy_version(version2.version_string, temp_storage / "deploy2")

        # First version should be deactivated
        assert version1.status == VersionStatus.STAGED
        assert version2.status == VersionStatus.ACTIVE

    def test_rollback_to_version_success(self, version_manager, temp_source, temp_storage):
        """Test successful rollback to version."""
        # Create and stage a version
        version = version_manager.create_version(temp_source)
        version.status = VersionStatus.STAGED
        version_manager._save_versions()

        success = version_manager.rollback_to_version(version.version_string)

        assert success
        assert version.status == VersionStatus.ACTIVE
        assert version.deployed_at is not None

    def test_rollback_to_version_nonexistent(self, version_manager):
        """Test rollback to nonexistent version."""
        with pytest.raises(ValueError, match="Version not found"):
            version_manager.rollback_to_version("nonexistent")

    def test_rollback_to_draft_version(self, version_manager, temp_source):
        """Test rollback to draft version."""
        version = version_manager.create_version(temp_source)

        with pytest.raises(ValueError, match="Cannot rollback to draft version"):
            version_manager.rollback_to_version(version.version_string)

    def test_deprecate_version_success(self, version_manager, temp_source):
        """Test successful version deprecation."""
        version = version_manager.create_version(temp_source)
        version.status = VersionStatus.STAGED
        version_manager._save_versions()

        success = version_manager.deprecate_version(
            version.version_string,
            reason="Security vulnerability"
        )

        assert success
        assert version.status == VersionStatus.DEPRECATED
        assert version.deprecated_at is not None
        assert "Security vulnerability" in version.description

    def test_deprecate_active_version(self, version_manager, temp_source, temp_storage):
        """Test deprecating active version."""
        version = version_manager.create_version(temp_source)
        version_manager.deploy_version(version.version_string, temp_storage / "deploy")

        with pytest.raises(ValueError, match="Cannot deprecate active version"):
            version_manager.deprecate_version(version.version_string)

    def test_archive_version_success(self, version_manager, temp_source):
        """Test successful version archiving."""
        version = version_manager.create_version(temp_source)

        success = version_manager.archive_version(version.version_string)

        assert success
        assert version.status == VersionStatus.ARCHIVED

    def test_archive_active_version(self, version_manager, temp_source, temp_storage):
        """Test archiving active version."""
        version = version_manager.create_version(temp_source)
        version_manager.deploy_version(version.version_string, temp_storage / "deploy")

        with pytest.raises(ValueError, match="Cannot archive active version"):
            version_manager.archive_version(version.version_string)

    def test_delete_version_success(self, version_manager, temp_source):
        """Test successful version deletion."""
        version = version_manager.create_version(temp_source)

        success = version_manager.delete_version(version.version_string)

        assert success
        assert version.version_string not in version_manager._versions

    def test_delete_active_version_without_force(self, version_manager, temp_source, temp_storage):
        """Test deleting active version without force."""
        version = version_manager.create_version(temp_source)
        version_manager.deploy_version(version.version_string, temp_storage / "deploy")

        with pytest.raises(ValueError, match="Cannot delete active version without force"):
            version_manager.delete_version(version.version_string)

    def test_delete_active_version_with_force(self, version_manager, temp_source, temp_storage):
        """Test deleting active version with force."""
        version = version_manager.create_version(temp_source)
        deploy_path = temp_storage / "deploy"
        deploy_path.mkdir()
        version_manager.deploy_version(version.version_string, deploy_path)

        success = version_manager.delete_version(version.version_string, force=True)

        assert success
        assert version.version_string not in version_manager._versions

    def test_delete_version_with_deployment_files(self, version_manager, temp_source, temp_storage):
        """Test deleting version with deployment files."""
        version = version_manager.create_version(temp_source)
        deploy_path = temp_storage / "deploy_files"
        deploy_path.mkdir()
        (deploy_path / "test.html").write_text("Test")

        version.deployment_path = deploy_path
        version.status = VersionStatus.STAGED
        version_manager._save_versions()

        success = version_manager.delete_version(version.version_string)

        assert success
        assert not deploy_path.exists()

    def test_get_version(self, version_manager, temp_source):
        """Test getting specific version."""
        version = version_manager.create_version(temp_source, "1.0.0")

        retrieved = version_manager.get_version("1.0.0")
        assert retrieved == version

        nonexistent = version_manager.get_version("nonexistent")
        assert nonexistent is None

    def test_get_active_version(self, version_manager, temp_source, temp_storage):
        """Test getting active version."""
        # No active version initially
        active = version_manager.get_active_version()
        assert active is None

        # Create and deploy version
        version = version_manager.create_version(temp_source)
        version_manager.deploy_version(version.version_string, temp_storage / "deploy")

        active = version_manager.get_active_version()
        assert active == version

    def test_list_versions_empty(self, version_manager):
        """Test listing versions when none exist."""
        versions = version_manager.list_versions()
        assert len(versions) == 0

    def test_list_versions_basic(self, version_manager, temp_source):
        """Test basic version listing."""
        version1 = version_manager.create_version(temp_source, "1.0.0")
        version2 = version_manager.create_version(temp_source, "2.0.0")

        versions = version_manager.list_versions()

        assert len(versions) == 2
        # Should be sorted by creation time (newest first)
        assert versions[0].version_string == "2.0.0"
        assert versions[1].version_string == "1.0.0"

    def test_list_versions_filtered_by_status(self, version_manager, temp_source, temp_storage):
        """Test listing versions filtered by status."""
        version1 = version_manager.create_version(temp_source, "1.0.0")
        version2 = version_manager.create_version(temp_source, "2.0.0")
        version_manager.deploy_version(version2.version_string, temp_storage / "deploy")

        # Filter by draft status
        draft_versions = version_manager.list_versions(status=VersionStatus.DRAFT)
        assert len(draft_versions) == 1
        assert draft_versions[0] == version1

        # Filter by active status
        active_versions = version_manager.list_versions(status=VersionStatus.ACTIVE)
        assert len(active_versions) == 1
        assert active_versions[0] == version2

    def test_list_versions_with_limit(self, version_manager, temp_source):
        """Test listing versions with limit."""
        for i in range(5):
            version_manager.create_version(temp_source, f"1.{i}.0")

        versions = version_manager.list_versions(limit=3)
        assert len(versions) == 3

    def test_cleanup_old_versions(self, version_manager, temp_source):
        """Test cleaning up old versions."""
        config = VersioningConfig(
            max_versions=3,
            auto_archive_days=1,
            auto_deprecate_days=1
        )
        version_manager.config = config

        # Create versions with different ages
        old_date = datetime.now() - timedelta(days=2)
        for i in range(5):
            version = version_manager.create_version(temp_source, f"1.{i}.0")
            if i < 3:  # Make first 3 versions old
                version.created_at = old_date
                version.status = VersionStatus.STAGED

        cleaned = version_manager.cleanup_old_versions()

        assert cleaned > 0

        # Check remaining versions
        remaining = version_manager.list_versions()
        assert len(remaining) <= 3

    def test_export_version_history(self, version_manager, temp_source, temp_storage):
        """Test exporting version history."""
        # Create some versions
        version_manager.create_version(temp_source, "1.0.0")
        version_manager.create_version(temp_source, "2.0.0")

        output_path = temp_storage / "history.json"
        success = version_manager.export_version_history(output_path)

        assert success
        assert output_path.exists()

        # Verify exported data
        with open(output_path, 'r') as f:
            data = json.load(f)

        assert 'export_timestamp' in data
        assert 'versions' in data
        assert 'statistics' in data
        assert len(data['versions']) == 2

    def test_export_version_history_failure(self, version_manager, temp_storage):
        """Test export failure handling."""
        # Try to export to invalid path
        invalid_path = Path("/invalid/path/history.json")
        success = version_manager.export_version_history(invalid_path)

        assert not success

    def test_generate_semantic_version(self, version_manager, temp_source):
        """Test semantic version generation."""
        # First version
        version1 = version_manager.create_version(temp_source)
        assert version1.version_string == "1.0.0"

        # Second version should increment
        version2 = version_manager.create_version(temp_source)
        assert version2.version_string == "1.0.1"

    def test_generate_timestamp_version(self, temp_storage, temp_source):
        """Test timestamp version generation."""
        config = VersioningConfig(strategy=VersioningStrategy.TIMESTAMP)
        manager = VersionManager(temp_storage, config)

        version = manager.create_version(temp_source)

        # Should be in format YYYYMMDD-HHMMSS
        assert len(version.version_string) == 15
        assert '-' in version.version_string

    def test_generate_sequential_version(self, temp_storage, temp_source):
        """Test sequential version generation."""
        config = VersioningConfig(strategy=VersioningStrategy.SEQUENTIAL)
        manager = VersionManager(temp_storage, config)

        version1 = manager.create_version(temp_source)
        assert version1.version_string == "v1"

        version2 = manager.create_version(temp_source)
        assert version2.version_string == "v2"

    @patch('subprocess.run')
    def test_generate_git_version(self, mock_run, temp_storage, temp_source):
        """Test Git hash version generation."""
        config = VersioningConfig(strategy=VersioningStrategy.GIT_HASH)
        manager = VersionManager(temp_storage, config)

        mock_run.return_value = Mock(stdout="abc123\n", returncode=0)

        version = manager.create_version(temp_source)

        assert version.version_string == "git-abc123"

    @patch('subprocess.run')
    def test_generate_git_version_fallback(self, mock_run, temp_storage, temp_source):
        """Test Git version generation fallback to timestamp."""
        config = VersioningConfig(strategy=VersioningStrategy.GIT_HASH)
        manager = VersionManager(temp_storage, config)

        mock_run.side_effect = subprocess.CalledProcessError(128, ['git'])

        version = manager.create_version(temp_source)

        # Should fall back to timestamp format
        assert len(version.version_string) == 15
        assert '-' in version.version_string

    def test_generate_custom_version(self, temp_storage, temp_source):
        """Test custom version generation."""
        config = VersioningConfig(
            strategy=VersioningStrategy.CUSTOM,
            custom_format="v{date}-{time}"
        )
        manager = VersionManager(temp_storage, config)

        version = manager.create_version(temp_source)

        assert version.version_string.startswith("v")
        assert len(version.version_string) > 10

    def test_generate_custom_version_no_format(self, temp_storage, temp_source):
        """Test custom version generation without format."""
        config = VersioningConfig(strategy=VersioningStrategy.CUSTOM)
        manager = VersionManager(temp_storage, config)

        with pytest.raises(ValueError, match="Custom format not specified"):
            manager.create_version(temp_source)

    def test_calculate_directory_hash(self, version_manager, temp_source):
        """Test directory hash calculation."""
        hash1 = version_manager._calculate_directory_hash(temp_source)
        assert len(hash1) == 32  # MD5 hex digest

        # Same directory should produce same hash
        hash2 = version_manager._calculate_directory_hash(temp_source)
        assert hash1 == hash2

        # Different content should produce different hash
        (temp_source / "new_file.txt").write_text("New content")
        hash3 = version_manager._calculate_directory_hash(temp_source)
        assert hash1 != hash3

    def test_calculate_directory_hash_unreadable_file(self, version_manager, temp_source):
        """Test directory hash with unreadable file."""
        # Create a file and make it unreadable (on Unix systems)
        test_file = temp_source / "unreadable.txt"
        test_file.write_text("Test")

        # Mock open to raise PermissionError
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            hash_result = version_manager._calculate_directory_hash(temp_source)

        # Should still produce a hash despite unreadable file
        assert len(hash_result) == 32

    def test_load_versions_empty_storage(self, temp_storage):
        """Test loading versions from empty storage."""
        manager = VersionManager(temp_storage)
        assert len(manager._versions) == 0

    def test_load_versions_with_existing_data(self, temp_storage, temp_source):
        """Test loading versions from existing storage."""
        # Create manager and add version
        manager1 = VersionManager(temp_storage)
        version = manager1.create_version(temp_source, "1.0.0")

        # Create new manager, should load existing version
        manager2 = VersionManager(temp_storage)
        assert len(manager2._versions) == 1
        assert "1.0.0" in manager2._versions

    def test_load_versions_corrupt_data(self, temp_storage):
        """Test loading versions with corrupt data."""
        # Create corrupt versions file
        versions_file = temp_storage / "versions.json"
        versions_file.write_text("invalid json content")

        # Should handle gracefully
        manager = VersionManager(temp_storage)
        assert len(manager._versions) == 0

    def test_save_versions_error_handling(self, version_manager, temp_source):
        """Test version saving error handling."""
        version = version_manager.create_version(temp_source)

        # Mock file operations to fail
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                version_manager._save_versions()

    def test_get_statistics(self, version_manager, temp_source, temp_storage):
        """Test getting version statistics."""
        # Create versions with different statuses
        version1 = version_manager.create_version(temp_source, "1.0.0")
        version2 = version_manager.create_version(temp_source, "2.0.0")
        version_manager.deploy_version(version2.version_string, temp_storage / "deploy")
        version_manager.deprecate_version(version1.version_string)

        stats = version_manager._get_statistics()

        assert stats['total_versions'] == 2
        assert stats['status_counts']['deprecated'] == 1
        assert stats['status_counts']['active'] == 1
        assert stats['active_version'] == "2.0.0"
        assert stats['total_size_bytes'] > 0
        assert stats['total_files'] > 0

    def test_edge_case_empty_source_directory(self, version_manager, temp_storage):
        """Test creating version from empty source directory."""
        empty_source = temp_storage / "empty_source"
        empty_source.mkdir()

        version = version_manager.create_version(empty_source)

        assert version.file_count == 0
        assert version.size_bytes == 0

    def test_edge_case_large_number_of_files(self, version_manager, temp_storage):
        """Test creating version with many files."""
        large_source = temp_storage / "large_source"
        large_source.mkdir()

        # Create many small files
        for i in range(100):
            (large_source / f"file_{i}.txt").write_text(f"Content {i}")

        version = version_manager.create_version(large_source)

        assert version.file_count == 100
        assert version.size_bytes > 0

    def test_edge_case_unicode_filenames(self, version_manager, temp_storage):
        """Test creating version with Unicode filenames."""
        unicode_source = temp_storage / "unicode_source"
        unicode_source.mkdir()

        (unicode_source / "测试.html").write_text("<html>Unicode test</html>")
        (unicode_source / "файл.css").write_text("body { color: blue; }")

        version = version_manager.create_version(unicode_source)

        assert version.file_count == 2
        assert version.source_hash is not None

    def test_edge_case_version_string_collision(self, version_manager, temp_source):
        """Test handling version string collision."""
        # Create version with explicit string
        version1 = version_manager.create_version(temp_source, "1.0.0")

        # Creating another version with same string should be allowed
        # (in practice, this might be prevented by application logic)
        version2 = version_manager.create_version(temp_source, "1.0.0")

        # The second version should overwrite the first in the dictionary
        assert len(version_manager._versions) == 1

    def test_persistence_across_instances(self, temp_storage, temp_source):
        """Test version persistence across manager instances."""
        # Create version with first manager
        manager1 = VersionManager(temp_storage)
        version = manager1.create_version(temp_source, "1.0.0", title="Persistent Version")

        # Create second manager, should load the version
        manager2 = VersionManager(temp_storage)
        loaded_version = manager2.get_version("1.0.0")

        assert loaded_version is not None
        assert loaded_version.title == "Persistent Version"
        assert loaded_version.version_string == "1.0.0"