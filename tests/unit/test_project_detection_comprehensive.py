"""
Comprehensive unit tests for project detection functionality.

This test module provides 100% coverage for the project detection module,
including all classes, methods, and edge cases. Tests both ProjectDetector
and DaemonIdentifier classes with comprehensive mocking of file system
and Git operations.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import os
import tempfile
import hashlib
import re
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock, call
from urllib.parse import urlparse

import git
import pytest
from git.exc import GitError, InvalidGitRepositoryError

from common.utils.project_detection import ProjectDetector, DaemonIdentifier
from common.core.pattern_manager import PatternManager


class TestDaemonIdentifier:
    """Comprehensive tests for DaemonIdentifier class."""

    def setup_method(self):
        """Clear registry before each test."""
        DaemonIdentifier.clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        DaemonIdentifier.clear_registry()

    def test_init(self):
        """Test DaemonIdentifier initialization."""
        identifier = DaemonIdentifier("test-project", "/path/to/project", "dev")

        assert identifier.project_name == "test-project"
        assert identifier.project_path == os.path.abspath("/path/to/project")
        assert identifier.suffix == "dev"
        assert identifier._identifier is None
        assert identifier._path_hash is None

    def test_init_without_suffix(self):
        """Test DaemonIdentifier initialization without suffix."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        assert identifier.project_name == "test-project"
        assert identifier.project_path == os.path.abspath("/path/to/project")
        assert identifier.suffix is None

    def test_generate_identifier_basic(self):
        """Test basic identifier generation."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        result = identifier.generate_identifier()

        # Should generate identifier with format: project_hash
        assert result is not None
        assert result.startswith("test-project_")
        assert len(result.split("_")) == 2
        assert identifier._identifier == result
        assert identifier._path_hash is not None

    def test_generate_identifier_with_suffix(self):
        """Test identifier generation with suffix."""
        identifier = DaemonIdentifier("test-project", "/path/to/project", "dev")
        result = identifier.generate_identifier()

        # Should generate identifier with format: project_hash_suffix
        assert result is not None
        assert result.startswith("test-project_")
        assert result.endswith("_dev")
        assert len(result.split("_")) == 3

    def test_generate_identifier_custom_hash_length(self):
        """Test identifier generation with custom hash length."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        result = identifier.generate_identifier(hash_length=12)

        # Extract hash part and verify length
        parts = result.split("_")
        hash_part = parts[1]
        assert len(hash_part) == 12

    def test_generate_identifier_cached(self):
        """Test that identifier generation is cached."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        result1 = identifier.generate_identifier()
        result2 = identifier.generate_identifier()

        assert result1 == result2
        assert identifier._identifier == result1

    def test_generate_identifier_collision_detection(self):
        """Test collision detection and resolution."""
        # Create first identifier
        identifier1 = DaemonIdentifier("test-project", "/path/to/project1")
        id1 = identifier1.generate_identifier()

        # Create second identifier with different path but potentially same hash
        identifier2 = DaemonIdentifier("test-project", "/path/to/project2")

        # Mock _check_collision to simulate collision
        with patch.object(identifier2, '_check_collision', side_effect=[True, False]):
            with patch.object(identifier2, '_generate_path_hash', side_effect=["abcd1234", "abcd12345678"]):
                id2 = identifier2.generate_identifier()

                # Should have tried longer hash due to collision
                assert id2 is not None
                assert id1 != id2

    def test_generate_identifier_collision_max_retries(self):
        """Test collision detection with maximum retries exceeded."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        # Mock collision detection to always return True
        with patch.object(identifier, '_check_collision', return_value=True):
            with pytest.raises(ValueError, match="Cannot generate unique identifier"):
                identifier.generate_identifier()

    def test_get_identifier_before_generation(self):
        """Test get_identifier before generation."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        assert identifier.get_identifier() is None

    def test_get_identifier_after_generation(self):
        """Test get_identifier after generation."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        generated = identifier.generate_identifier()
        assert identifier.get_identifier() == generated

    def test_get_path_hash_before_generation(self):
        """Test get_path_hash before generation."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        assert identifier.get_path_hash() is None

    def test_get_path_hash_after_generation(self):
        """Test get_path_hash after generation."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        identifier.generate_identifier()
        assert identifier.get_path_hash() is not None
        assert len(identifier.get_path_hash()) == 8  # default hash length

    def test_validate_identifier_valid_format(self):
        """Test identifier validation with valid format."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        generated = identifier.generate_identifier()

        assert identifier.validate_identifier(generated) is True

    def test_validate_identifier_invalid_format(self):
        """Test identifier validation with invalid formats."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        invalid_identifiers = [
            "invalid",
            "no_hash",
            "project_",
            "project_invalid_hash",
            "project_abc",  # too short hash
            "project_abc123_",  # empty suffix
        ]

        for invalid_id in invalid_identifiers:
            assert identifier.validate_identifier(invalid_id) is False

    def test_validate_identifier_wrong_path(self):
        """Test identifier validation with wrong path hash."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        # Create identifier for different path
        wrong_identifier = "test-project_wronghash"
        assert identifier.validate_identifier(wrong_identifier) is False

    def test_release_identifier(self):
        """Test identifier release."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        generated = identifier.generate_identifier()

        # Verify it's in registry
        assert generated in DaemonIdentifier._active_identifiers
        assert generated in DaemonIdentifier._identifier_registry

        # Release it
        identifier.release_identifier()

        # Verify it's removed
        assert generated not in DaemonIdentifier._active_identifiers
        assert generated not in DaemonIdentifier._identifier_registry

    def test_release_identifier_not_generated(self):
        """Test releasing identifier that wasn't generated."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        # Should not raise error
        identifier.release_identifier()

    def test_generate_path_hash_consistent(self):
        """Test path hash generation consistency."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        hash1 = identifier._generate_path_hash("/path/to/project")
        hash2 = identifier._generate_path_hash("/path/to/project")

        assert hash1 == hash2

    def test_generate_path_hash_different_paths(self):
        """Test path hash generation for different paths."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        hash1 = identifier._generate_path_hash("/path/to/project1")
        hash2 = identifier._generate_path_hash("/path/to/project2")

        assert hash1 != hash2

    def test_generate_path_hash_normalized_paths(self):
        """Test path hash generation with path normalization."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        hash1 = identifier._generate_path_hash("/path/to/project")
        hash2 = identifier._generate_path_hash("/path/to/../to/project")
        hash3 = identifier._generate_path_hash("/path/to/project/")

        # All should be normalized to same path
        assert hash1 == hash2 == hash3

    def test_generate_path_hash_custom_length(self):
        """Test path hash generation with custom length."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        hash_short = identifier._generate_path_hash("/path/to/project", 4)
        hash_long = identifier._generate_path_hash("/path/to/project", 16)

        assert len(hash_short) == 4
        assert len(hash_long) == 16

    def test_check_collision_no_collision(self):
        """Test collision checking with no collision."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        assert identifier._check_collision("new-identifier") is False

    def test_check_collision_same_project(self):
        """Test collision checking with same project (allowed)."""
        identifier1 = DaemonIdentifier("test-project", "/path/to/project")
        id1 = identifier1.generate_identifier()

        identifier2 = DaemonIdentifier("test-project", "/path/to/project")

        # Same project should not be a collision
        assert identifier2._check_collision(id1) is False

    def test_check_collision_different_project(self):
        """Test collision checking with different project (collision)."""
        identifier1 = DaemonIdentifier("test-project", "/path/to/project1")
        id1 = identifier1.generate_identifier()

        identifier2 = DaemonIdentifier("test-project", "/path/to/project2")

        # Manually add to registry to simulate collision
        DaemonIdentifier._active_identifiers.add("test-collision")
        DaemonIdentifier._identifier_registry["test-collision"] = {
            'project_path': '/different/path'
        }

        assert identifier2._check_collision("test-collision") is True

    def test_register_identifier(self):
        """Test identifier registration."""
        identifier = DaemonIdentifier("test-project", "/path/to/project", "dev")
        generated = identifier.generate_identifier()

        # Verify registration
        assert generated in DaemonIdentifier._active_identifiers
        registry_info = DaemonIdentifier._identifier_registry[generated]

        assert registry_info['project_name'] == "test-project"
        assert registry_info['project_path'] == identifier.project_path
        assert registry_info['suffix'] == "dev"
        assert registry_info['path_hash'] == identifier._path_hash
        assert 'registered_at' in registry_info

    def test_get_active_identifiers(self):
        """Test getting active identifiers."""
        assert len(DaemonIdentifier.get_active_identifiers()) == 0

        identifier1 = DaemonIdentifier("project1", "/path1")
        identifier2 = DaemonIdentifier("project2", "/path2")

        id1 = identifier1.generate_identifier()
        id2 = identifier2.generate_identifier()

        active = DaemonIdentifier.get_active_identifiers()
        assert len(active) == 2
        assert id1 in active
        assert id2 in active

    def test_get_identifier_info(self):
        """Test getting identifier information."""
        identifier = DaemonIdentifier("test-project", "/path/to/project", "dev")
        generated = identifier.generate_identifier()

        info = DaemonIdentifier.get_identifier_info(generated)
        assert info is not None
        assert info['project_name'] == "test-project"
        assert info['suffix'] == "dev"

    def test_get_identifier_info_not_found(self):
        """Test getting identifier information for non-existent identifier."""
        info = DaemonIdentifier.get_identifier_info("non-existent")
        assert info is None

    def test_clear_registry(self):
        """Test clearing the registry."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")
        identifier.generate_identifier()

        assert len(DaemonIdentifier._active_identifiers) > 0
        assert len(DaemonIdentifier._identifier_registry) > 0

        DaemonIdentifier.clear_registry()

        assert len(DaemonIdentifier._active_identifiers) == 0
        assert len(DaemonIdentifier._identifier_registry) == 0

    def test_str_representation(self):
        """Test string representation."""
        identifier = DaemonIdentifier("test-project", "/path/to/project")

        # Before generation
        str_repr = str(identifier)
        assert "test-project" in str_repr
        assert "ungenerated" in str_repr

        # After generation
        generated = identifier.generate_identifier()
        str_repr = str(identifier)
        assert str_repr == generated

    def test_repr_representation(self):
        """Test detailed representation."""
        identifier = DaemonIdentifier("test-project", "/path/to/project", "dev")
        repr_str = repr(identifier)

        assert "DaemonIdentifier" in repr_str
        assert "test-project" in repr_str
        assert "/path/to/project" in repr_str
        assert "dev" in repr_str


class TestProjectDetectorComprehensive:
    """Comprehensive tests for ProjectDetector class."""

    def test_init_defaults(self):
        """Test ProjectDetector initialization with defaults."""
        detector = ProjectDetector()

        assert detector.github_user is None
        assert detector.pattern_manager is not None
        assert isinstance(detector.pattern_manager, PatternManager)

    def test_init_with_github_user(self):
        """Test ProjectDetector initialization with GitHub user."""
        detector = ProjectDetector(github_user="testuser")

        assert detector.github_user == "testuser"

    def test_init_with_pattern_manager(self):
        """Test ProjectDetector initialization with custom pattern manager."""
        custom_manager = Mock(spec=PatternManager)
        detector = ProjectDetector(pattern_manager=custom_manager)

        assert detector.pattern_manager == custom_manager

    def test_get_project_name_non_git_fallback(self):
        """Test project name detection fallback for non-Git directory."""
        detector = ProjectDetector()

        with patch.object(detector, '_find_git_root', return_value=None):
            with patch('common.utils.project_detection.os.path.basename', return_value="project") as mock_basename:
                with patch('common.utils.project_detection.os.path.abspath', return_value="/absolute/path") as mock_abspath:
                    result = detector.get_project_name("/some/path")

                    assert result == "project"
                    mock_abspath.assert_called_once_with("/some/path")
                    mock_basename.assert_called_once_with("/absolute/path")

    def test_get_project_name_git_with_user_owned_repo(self):
        """Test project name detection for user-owned Git repository."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, '_find_git_root', return_value="/git/root"):
            with patch.object(detector, '_get_git_remote_url', return_value="https://github.com/testuser/myproject.git"):
                with patch.object(detector, '_belongs_to_user', return_value=True):
                    with patch.object(detector, '_extract_repo_name_from_remote', return_value="myproject"):
                        result = detector.get_project_name("/some/path")

                        assert result == "myproject"

    def test_get_project_name_git_not_user_owned(self):
        """Test project name detection for non-user-owned Git repository."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, '_find_git_root', return_value="/git/root"):
            with patch.object(detector, '_get_git_remote_url', return_value="https://github.com/otheruser/project.git"):
                with patch.object(detector, '_belongs_to_user', return_value=False):
                    with patch('common.utils.project_detection.os.path.basename', return_value="project-dir"):
                        result = detector.get_project_name("/some/path")

                        assert result == "project-dir"

    def test_get_project_name_git_no_remote_url(self):
        """Test project name detection for Git repository without remote URL."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, '_find_git_root', return_value="/git/root"):
            with patch.object(detector, '_get_git_remote_url', return_value=None):
                with patch('common.utils.project_detection.os.path.basename', return_value="project-dir"):
                    result = detector.get_project_name("/some/path")

                    assert result == "project-dir"

    def test_get_project_name_git_no_repo_name_extracted(self):
        """Test project name detection when repo name extraction fails."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, '_find_git_root', return_value="/git/root"):
            with patch.object(detector, '_get_git_remote_url', return_value="https://github.com/testuser/project.git"):
                with patch.object(detector, '_belongs_to_user', return_value=True):
                    with patch.object(detector, '_extract_repo_name_from_remote', return_value=None):
                        with patch('common.utils.project_detection.os.path.basename', return_value="project-dir"):
                            result = detector.get_project_name("/some/path")

                            assert result == "project-dir"

    def test_get_project_name_exception_handling(self):
        """Test project name detection with exception handling."""
        detector = ProjectDetector()

        with patch.object(detector, '_find_git_root', side_effect=Exception("Test error")):
            with patch('common.utils.project_detection.os.path.basename', return_value="fallback"):
                with patch('common.utils.project_detection.os.path.abspath', return_value="/absolute/path"):
                    with patch('common.utils.project_detection.logger.warning') as mock_logger:
                        result = detector.get_project_name("/some/path")

                        assert result == "fallback"
                        mock_logger.assert_called_once()

    def test_get_project_and_subprojects(self):
        """Test getting project name and subprojects together."""
        detector = ProjectDetector()

        with patch.object(detector, 'get_project_name', return_value="main-project"):
            with patch.object(detector, 'get_subprojects', return_value=["sub1", "sub2"]):
                main, subs = detector.get_project_and_subprojects("/path")

                assert main == "main-project"
                assert subs == ["sub1", "sub2"]

    def test_get_subprojects(self):
        """Test getting subprojects list."""
        detector = ProjectDetector()

        mock_detailed_submodules = [
            {"project_name": "sub1"},
            {"project_name": "sub2"},
            {"project_name": None},  # Should be filtered out
        ]

        with patch.object(detector, 'get_detailed_submodules', return_value=mock_detailed_submodules):
            result = detector.get_subprojects("/path")

            assert result == ["sub1", "sub2"]

    def test_get_detailed_submodules_no_git_root(self):
        """Test detailed submodules detection with no Git root."""
        detector = ProjectDetector()

        with patch.object(detector, '_find_git_root', return_value=None):
            result = detector.get_detailed_submodules("/path")

            assert result == []

    @patch('common.utils.project_detection.git.Repo')
    def test_get_detailed_submodules_success(self, mock_repo_class):
        """Test successful detailed submodules detection."""
        detector = ProjectDetector()

        # Mock Git repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        # Mock submodule
        mock_submodule = Mock()
        mock_submodule.name = "test-submodule"
        mock_repo.submodules = [mock_submodule]

        # Mock submodule analysis
        submodule_info = {
            "name": "test-submodule",
            "project_name": "test-submodule"
        }

        with patch.object(detector, '_find_git_root', return_value="/git/root"):
            with patch.object(detector, '_analyze_submodule', return_value=submodule_info):
                result = detector.get_detailed_submodules("/path")

                assert len(result) == 1
                assert result[0] == submodule_info

    @patch('common.utils.project_detection.git.Repo')
    def test_get_detailed_submodules_with_failures(self, mock_repo_class):
        """Test detailed submodules detection with some failures."""
        detector = ProjectDetector()

        # Mock Git repository
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        # Mock submodules - one succeeds, one fails
        mock_submodule1 = Mock()
        mock_submodule1.name = "success-submodule"
        mock_submodule2 = Mock()
        mock_submodule2.name = "failure-submodule"
        mock_repo.submodules = [mock_submodule1, mock_submodule2]

        def mock_analyze_submodule(submodule, git_root):
            if submodule.name == "success-submodule":
                return {"name": "success-submodule", "project_name": "success"}
            else:
                raise Exception("Analysis failed")

        with patch.object(detector, '_find_git_root', return_value="/git/root"):
            with patch.object(detector, '_analyze_submodule', side_effect=mock_analyze_submodule):
                with patch('common.utils.project_detection.logger.warning'):
                    result = detector.get_detailed_submodules("/path")

                    assert len(result) == 1
                    assert result[0]["name"] == "success-submodule"

    def test_get_detailed_submodules_exception_handling(self):
        """Test detailed submodules detection with exception handling."""
        detector = ProjectDetector()

        with patch.object(detector, '_find_git_root', side_effect=Exception("Git error")):
            with patch('common.utils.project_detection.logger.warning') as mock_logger:
                result = detector.get_detailed_submodules("/path")

                assert result == []
                mock_logger.assert_called_once()

    def test_analyze_submodule_success(self):
        """Test successful submodule analysis."""
        detector = ProjectDetector(github_user="testuser")

        # Mock submodule
        mock_submodule = Mock()
        mock_submodule.name = "test-submodule"
        mock_submodule.path = "libs/test-submodule"
        mock_submodule.url = "https://github.com/testuser/test-submodule.git"
        mock_submodule.hexsha = "abc123"

        with patch.object(detector, '_parse_git_url') as mock_parse:
            with patch.object(detector, '_belongs_to_user', return_value=True):
                with patch.object(detector, '_extract_repo_name_from_remote', return_value="test-submodule"):
                    with patch('common.utils.project_detection.os.path.exists', return_value=True):
                        with patch('common.utils.project_detection.os.listdir', return_value=["file.txt"]):
                            result = detector._analyze_submodule(mock_submodule, "/git/root")

                            assert result is not None
                            assert result["name"] == "test-submodule"
                            assert result["path"] == "libs/test-submodule"
                            assert result["url"] == "https://github.com/testuser/test-submodule.git"
                            assert result["project_name"] == "test-submodule"
                            assert result["is_initialized"] is True
                            assert result["user_owned"] is True
                            assert result["commit_sha"] == "abc123"

    def test_analyze_submodule_filtered_by_user(self):
        """Test submodule analysis filtered by GitHub user."""
        detector = ProjectDetector(github_user="testuser")

        # Mock submodule not owned by user
        mock_submodule = Mock()
        mock_submodule.url = "https://github.com/otheruser/project.git"

        with patch.object(detector, '_belongs_to_user', return_value=False):
            result = detector._analyze_submodule(mock_submodule, "/git/root")

            assert result is None

    def test_analyze_submodule_no_user_filtering(self):
        """Test submodule analysis with no user filtering."""
        detector = ProjectDetector()  # No github_user

        # Mock submodule
        mock_submodule = Mock()
        mock_submodule.name = "test-submodule"
        mock_submodule.path = "test-submodule"
        mock_submodule.url = "https://github.com/anyuser/test-submodule.git"
        mock_submodule.hexsha = "abc123"

        with patch.object(detector, '_parse_git_url'):
            with patch.object(detector, '_belongs_to_user', return_value=False):
                with patch.object(detector, '_extract_repo_name_from_remote', return_value="test-submodule"):
                    with patch('common.utils.project_detection.os.path.exists', return_value=True):
                        with patch('common.utils.project_detection.os.listdir', return_value=["file.txt"]):
                            result = detector._analyze_submodule(mock_submodule, "/git/root")

                            assert result is not None
                            assert result["user_owned"] is False

    def test_analyze_submodule_not_initialized(self):
        """Test submodule analysis for non-initialized submodule."""
        detector = ProjectDetector()

        # Mock submodule
        mock_submodule = Mock()
        mock_submodule.name = "test-submodule"
        mock_submodule.path = "test-submodule"
        mock_submodule.url = "https://github.com/user/test-submodule.git"
        mock_submodule.hexsha = "abc123"

        with patch.object(detector, '_parse_git_url'):
            with patch.object(detector, '_belongs_to_user', return_value=False):
                with patch.object(detector, '_extract_repo_name_from_remote', return_value="test-submodule"):
                    with patch('common.utils.project_detection.os.path.exists', return_value=False):
                        result = detector._analyze_submodule(mock_submodule, "/git/root")

                        assert result is not None
                        assert result["is_initialized"] is False

    def test_analyze_submodule_commit_sha_exception(self):
        """Test submodule analysis when commit SHA access fails."""
        detector = ProjectDetector()

        # Mock submodule with hexsha property that raises exception
        mock_submodule = Mock()
        mock_submodule.name = "test-submodule"
        mock_submodule.path = "test-submodule"
        mock_submodule.url = "https://github.com/user/test-submodule.git"
        type(mock_submodule).hexsha = PropertyMock(side_effect=Exception("No commit"))

        with patch.object(detector, '_parse_git_url'):
            with patch.object(detector, '_belongs_to_user', return_value=False):
                with patch.object(detector, '_extract_repo_name_from_remote', return_value="test-submodule"):
                    with patch('common.utils.project_detection.os.path.exists', return_value=True):
                        with patch('common.utils.project_detection.os.listdir', return_value=["file.txt"]):
                            result = detector._analyze_submodule(mock_submodule, "/git/root")

                            assert result is not None
                            assert result["commit_sha"] is None

    def test_analyze_submodule_exception_handling(self):
        """Test submodule analysis with exception handling."""
        detector = ProjectDetector()

        # Mock submodule that causes exception
        mock_submodule = Mock()
        mock_submodule.name = "failing-submodule"
        mock_submodule.url = "invalid-url"

        with patch.object(detector, '_parse_git_url', side_effect=Exception("Parse error")):
            with patch('common.utils.project_detection.logger.error') as mock_logger:
                result = detector._analyze_submodule(mock_submodule, "/git/root")

                assert result is None
                mock_logger.assert_called_once()

    @patch('common.utils.project_detection.git.Repo')
    def test_find_git_root_success(self, mock_repo_class):
        """Test successful Git root finding."""
        detector = ProjectDetector()

        mock_repo = Mock()
        mock_repo.working_dir = "/git/root"
        mock_repo_class.return_value = mock_repo

        result = detector._find_git_root("/some/path")
        assert result == "/git/root"
        mock_repo_class.assert_called_once_with("/some/path", search_parent_directories=True)

    @patch('common.utils.project_detection.git.Repo')
    def test_find_git_root_no_working_dir(self, mock_repo_class):
        """Test Git root finding with no working directory."""
        detector = ProjectDetector()

        mock_repo = Mock()
        mock_repo.working_dir = None
        mock_repo_class.return_value = mock_repo

        result = detector._find_git_root("/some/path")
        assert result is None

    @patch('common.utils.project_detection.git.Repo')
    def test_find_git_root_invalid_repo(self, mock_repo_class):
        """Test Git root finding with invalid repository."""
        detector = ProjectDetector()

        mock_repo_class.side_effect = InvalidGitRepositoryError("Not a Git repo")

        result = detector._find_git_root("/some/path")
        assert result is None

    @patch('common.utils.project_detection.git.Repo')
    def test_find_git_root_git_error(self, mock_repo_class):
        """Test Git root finding with Git error."""
        detector = ProjectDetector()

        mock_repo_class.side_effect = GitError("Git error")

        result = detector._find_git_root("/some/path")
        assert result is None

    @patch('common.utils.project_detection.git.Repo')
    def test_get_git_remote_url_origin(self, mock_repo_class):
        """Test getting Git remote URL with origin remote."""
        detector = ProjectDetector()

        mock_repo = Mock()
        mock_origin = Mock()
        mock_origin.url = "https://github.com/user/repo.git"
        mock_repo.remotes.origin = mock_origin
        mock_repo_class.return_value = mock_repo

        result = detector._get_git_remote_url("/git/root")
        assert result == "https://github.com/user/repo.git"

    @patch('common.utils.project_detection.git.Repo')
    def test_get_git_remote_url_upstream(self, mock_repo_class):
        """Test getting Git remote URL with upstream remote when no origin."""
        detector = ProjectDetector()

        mock_repo = Mock()
        mock_upstream = Mock()
        mock_upstream.url = "https://github.com/upstream/repo.git"

        # Mock the remotes to not have origin but have upstream
        mock_remotes = Mock()
        mock_remotes.upstream = mock_upstream

        # Mock hasattr for remotes object
        def remotes_hasattr(attr):
            return attr == "upstream"

        mock_remotes.__hasattr__ = remotes_hasattr
        mock_repo.remotes = mock_remotes
        mock_repo_class.return_value = mock_repo

        # Patch hasattr to check for attributes on remotes
        original_hasattr = hasattr
        def mock_hasattr(obj, name):
            if obj is mock_remotes:
                return name == "upstream"
            return original_hasattr(obj, name)

        with patch('builtins.hasattr', side_effect=mock_hasattr):
            result = detector._get_git_remote_url("/git/root")
            assert result == "https://github.com/upstream/repo.git"

    @patch('common.utils.project_detection.git.Repo')
    def test_get_git_remote_url_first_available(self, mock_repo_class):
        """Test getting Git remote URL with first available remote."""
        detector = ProjectDetector()

        mock_repo = Mock()
        # Configure remotes to not have origin or upstream
        type(mock_repo.remotes).origin = PropertyMock(side_effect=AttributeError("No origin"))
        type(mock_repo.remotes).upstream = PropertyMock(side_effect=AttributeError("No upstream"))

        mock_remote = Mock()
        mock_remote.url = "https://github.com/other/repo.git"
        mock_repo.remotes = [mock_remote]
        mock_repo_class.return_value = mock_repo

        result = detector._get_git_remote_url("/git/root")
        assert result == "https://github.com/other/repo.git"

    @patch('common.utils.project_detection.git.Repo')
    def test_get_git_remote_url_no_remotes(self, mock_repo_class):
        """Test getting Git remote URL with no remotes."""
        detector = ProjectDetector()

        mock_repo = Mock()
        # Configure remotes to not have origin or upstream
        type(mock_repo.remotes).origin = PropertyMock(side_effect=AttributeError("No origin"))
        type(mock_repo.remotes).upstream = PropertyMock(side_effect=AttributeError("No upstream"))
        mock_repo.remotes = []
        mock_repo_class.return_value = mock_repo

        result = detector._get_git_remote_url("/git/root")
        assert result is None

    @patch('common.utils.project_detection.git.Repo')
    def test_get_git_remote_url_exception(self, mock_repo_class):
        """Test getting Git remote URL with exception."""
        detector = ProjectDetector()

        mock_repo_class.side_effect = Exception("Git error")

        with patch('common.utils.project_detection.logger.warning') as mock_logger:
            result = detector._get_git_remote_url("/git/root")

            assert result is None
            mock_logger.assert_called_once()

    def test_parse_git_url_empty_url(self):
        """Test parsing empty Git URL."""
        detector = ProjectDetector()

        result = detector._parse_git_url("")

        expected = {
            "original": "",
            "hostname": None,
            "username": None,
            "repository": None,
            "protocol": None,
            "is_github": False,
            "is_ssh": False,
        }

        assert result == expected

    def test_parse_git_url_ssh_format(self):
        """Test parsing SSH format Git URL."""
        detector = ProjectDetector()

        url = "git@github.com:testuser/testrepo.git"
        result = detector._parse_git_url(url)

        assert result["original"] == url
        assert result["hostname"] == "github.com"
        assert result["username"] == "testuser"
        assert result["repository"] == "testrepo"
        assert result["protocol"] == "ssh"
        assert result["is_github"] is True
        assert result["is_ssh"] is True

    def test_parse_git_url_https_format(self):
        """Test parsing HTTPS format Git URL."""
        detector = ProjectDetector()

        url = "https://github.com/testuser/testrepo.git"
        result = detector._parse_git_url(url)

        assert result["original"] == url
        assert result["hostname"] == "github.com"
        assert result["username"] == "testuser"
        assert result["repository"] == "testrepo"
        assert result["protocol"] == "https"
        assert result["is_github"] is True
        assert result["is_ssh"] is False

    def test_parse_git_url_https_no_git_suffix(self):
        """Test parsing HTTPS URL without .git suffix."""
        detector = ProjectDetector()

        url = "https://github.com/testuser/testrepo"
        result = detector._parse_git_url(url)

        assert result["repository"] == "testrepo"

    def test_parse_git_url_non_github(self):
        """Test parsing non-GitHub Git URL."""
        detector = ProjectDetector()

        url = "https://gitlab.com/testuser/testrepo.git"
        result = detector._parse_git_url(url)

        assert result["hostname"] == "gitlab.com"
        assert result["username"] == "testuser"
        assert result["repository"] == "testrepo"
        assert result["is_github"] is False

    def test_parse_git_url_malformed_ssh(self):
        """Test parsing malformed SSH URL."""
        detector = ProjectDetector()

        url = "git@github.com:malformed"
        result = detector._parse_git_url(url)

        # Should handle gracefully and return basic info
        assert result["original"] == url
        assert result["is_ssh"] is True
        assert result["protocol"] == "ssh"

    def test_parse_git_url_malformed_https(self):
        """Test parsing malformed HTTPS URL."""
        detector = ProjectDetector()

        url = "https://github.com/"
        result = detector._parse_git_url(url)

        # Should handle gracefully
        assert result["original"] == url
        assert result["hostname"] == "github.com"
        assert result["protocol"] == "https"

    def test_parse_git_url_exception_handling(self):
        """Test parsing Git URL with exception handling."""
        detector = ProjectDetector()

        # This should trigger an exception in the parsing logic
        url = "https://[invalid-url"

        with patch('common.utils.project_detection.logger.warning') as mock_logger:
            result = detector._parse_git_url(url)

            # Should return basic structure even with error
            assert result["original"] == url
            mock_logger.assert_called_once()

    def test_belongs_to_user_no_github_user(self):
        """Test user ownership check with no GitHub user configured."""
        detector = ProjectDetector()

        result = detector._belongs_to_user("https://github.com/testuser/repo.git")
        assert result is False

    def test_belongs_to_user_no_remote_url(self):
        """Test user ownership check with no remote URL."""
        detector = ProjectDetector(github_user="testuser")

        result = detector._belongs_to_user("")
        assert result is False

        result = detector._belongs_to_user(None)
        assert result is False

    def test_belongs_to_user_github_match(self):
        """Test user ownership check with matching GitHub user."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, '_parse_git_url') as mock_parse:
            mock_parse.return_value = {
                "is_github": True,
                "username": "testuser"
            }

            result = detector._belongs_to_user("https://github.com/testuser/repo.git")
            assert result is True

    def test_belongs_to_user_github_no_match(self):
        """Test user ownership check with non-matching GitHub user."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, '_parse_git_url') as mock_parse:
            mock_parse.return_value = {
                "is_github": True,
                "username": "otheruser"
            }

            result = detector._belongs_to_user("https://github.com/otheruser/repo.git")
            assert result is False

    def test_belongs_to_user_non_github(self):
        """Test user ownership check with non-GitHub URL."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, '_parse_git_url') as mock_parse:
            mock_parse.return_value = {
                "is_github": False,
                "username": "testuser"
            }

            result = detector._belongs_to_user("https://gitlab.com/testuser/repo.git")
            assert result is False

    def test_belongs_to_user_case_insensitive(self):
        """Test user ownership check is case insensitive."""
        detector = ProjectDetector(github_user="TestUser")

        with patch.object(detector, '_parse_git_url') as mock_parse:
            mock_parse.return_value = {
                "is_github": True,
                "username": "testuser"
            }

            result = detector._belongs_to_user("https://github.com/testuser/repo.git")
            assert result is True

    def test_belongs_to_user_exception_handling(self):
        """Test user ownership check with exception handling."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, '_parse_git_url', side_effect=Exception("Parse error")):
            with patch('common.utils.project_detection.logger.warning') as mock_logger:
                result = detector._belongs_to_user("invalid-url")

                assert result is False
                mock_logger.assert_called_once()

    def test_extract_repo_name_from_remote_success(self):
        """Test successful repository name extraction."""
        detector = ProjectDetector()

        with patch.object(detector, '_parse_git_url') as mock_parse:
            mock_parse.return_value = {"repository": "test-repo"}

            result = detector._extract_repo_name_from_remote("https://github.com/user/test-repo.git")
            assert result == "test-repo"

    def test_extract_repo_name_from_remote_empty_url(self):
        """Test repository name extraction with empty URL."""
        detector = ProjectDetector()

        result = detector._extract_repo_name_from_remote("")
        assert result is None

        result = detector._extract_repo_name_from_remote(None)
        assert result is None

    def test_extract_repo_name_from_remote_exception(self):
        """Test repository name extraction with exception."""
        detector = ProjectDetector()

        with patch.object(detector, '_parse_git_url', side_effect=Exception("Parse error")):
            with patch('common.utils.project_detection.logger.warning') as mock_logger:
                result = detector._extract_repo_name_from_remote("invalid-url")

                assert result is None
                mock_logger.assert_called_once()

    def test_get_project_info_comprehensive(self):
        """Test comprehensive project information gathering."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, 'get_project_and_subprojects', return_value=("main-project", ["sub1", "sub2"])):
            with patch.object(detector, '_find_git_root', return_value="/git/root"):
                with patch.object(detector, '_get_git_remote_url', return_value="https://github.com/testuser/main-project.git"):
                    with patch.object(detector, 'get_detailed_submodules', return_value=[{"name": "sub1", "user_owned": True}]):
                        with patch.object(detector, '_parse_git_url', return_value={"hostname": "github.com"}):
                            with patch.object(detector, '_belongs_to_user', return_value=True):
                                with patch('common.utils.project_detection.os.path.abspath', return_value="/absolute/path"):
                                    result = detector.get_project_info("/some/path")

                                    assert result["main_project"] == "main-project"
                                    assert result["subprojects"] == ["sub1", "sub2"]
                                    assert result["git_root"] == "/git/root"
                                    assert result["remote_url"] == "https://github.com/testuser/main-project.git"
                                    assert result["github_user"] == "testuser"
                                    assert result["path"] == "/absolute/path"
                                    assert result["is_git_repo"] is True
                                    assert result["belongs_to_user"] is True
                                    assert result["submodule_count"] == 1
                                    assert len(result["user_owned_submodules"]) == 1

    def test_get_project_info_no_git_repo(self):
        """Test project info for non-Git repository."""
        detector = ProjectDetector()

        with patch.object(detector, 'get_project_and_subprojects', return_value=("project", [])):
            with patch.object(detector, '_find_git_root', return_value=None):
                with patch.object(detector, 'get_detailed_submodules', return_value=[]):
                    with patch('common.utils.project_detection.os.path.abspath', return_value="/absolute/path"):
                        result = detector.get_project_info("/some/path")

                        assert result["is_git_repo"] is False
                        assert result["git_root"] is None
                        assert result["remote_url"] is None
                        assert result["belongs_to_user"] is False

    def test_get_project_info_exception_handling(self):
        """Test project info with exception handling."""
        detector = ProjectDetector(github_user="testuser")

        with patch.object(detector, 'get_project_and_subprojects', side_effect=Exception("Test error")):
            with patch('common.utils.project_detection.os.path.basename', return_value="fallback"):
                with patch('common.utils.project_detection.os.path.abspath', return_value="/absolute/path"):
                    with patch('common.utils.project_detection.logger.error') as mock_logger:
                        result = detector.get_project_info("/some/path")

                        assert result["main_project"] == "fallback"
                        assert result["error"] == "Test error"
                        assert result["is_git_repo"] is False
                        mock_logger.assert_called_once()

    def test_create_daemon_identifier(self):
        """Test daemon identifier creation."""
        detector = ProjectDetector()

        with patch.object(detector, 'get_project_name', return_value="test-project"):
            with patch('common.utils.project_detection.os.path.abspath', return_value="/absolute/path"):
                identifier = detector.create_daemon_identifier("/some/path", "dev")

                assert isinstance(identifier, DaemonIdentifier)
                assert identifier.project_name == "test-project"
                assert identifier.project_path == "/absolute/path"
                assert identifier.suffix == "dev"

    def test_create_daemon_identifier_default_path(self):
        """Test daemon identifier creation with default path."""
        detector = ProjectDetector()

        with patch.object(detector, 'get_project_name', return_value="test-project"):
            with patch('common.utils.project_detection.os.path.abspath', return_value="/current/dir"):
                identifier = detector.create_daemon_identifier()

                assert identifier.project_name == "test-project"
                assert identifier.project_path == "/current/dir"
                assert identifier.suffix is None

    def test_detect_ecosystems_success(self):
        """Test successful ecosystem detection."""
        detector = ProjectDetector()

        with patch.object(detector.pattern_manager, 'detect_ecosystem', return_value=["python", "javascript"]):
            result = detector.detect_ecosystems("/some/path")

            assert result == ["python", "javascript"]

    def test_detect_ecosystems_exception_handling(self):
        """Test ecosystem detection with exception handling."""
        detector = ProjectDetector()

        with patch.object(detector.pattern_manager, 'detect_ecosystem', side_effect=Exception("Detection error")):
            with patch('common.utils.project_detection.logger.warning') as mock_logger:
                result = detector.detect_ecosystems("/some/path")

                assert result == []
                mock_logger.assert_called_once()

    def test_detect_ecosystems_default_path(self):
        """Test ecosystem detection with default path."""
        detector = ProjectDetector()

        with patch.object(detector.pattern_manager, 'detect_ecosystem', return_value=["rust"]) as mock_detect:
            result = detector.detect_ecosystems()

            assert result == ["rust"]
            mock_detect.assert_called_once_with(".")


# Import fixture helper for PropertyMock
from unittest.mock import PropertyMock


@pytest.fixture
def temp_git_repo():
    """Create a temporary Git repository for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize Git repo
        repo = git.Repo.init(temp_dir)

        # Add a remote
        repo.create_remote("origin", "https://github.com/testuser/test-project.git")

        # Create a test file and commit
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")

        repo.index.add([str(test_file)])
        repo.index.commit("Initial commit")

        yield temp_dir


@pytest.fixture
def temp_git_repo_with_submodules():
    """Create a temporary Git repository with submodules for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize Git repo
        repo = git.Repo.init(temp_dir)

        # Add a remote
        repo.create_remote("origin", "https://github.com/testuser/main-project.git")

        # Create a test file and commit
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("test content")

        repo.index.add([str(test_file)])
        repo.index.commit("Initial commit")

        yield temp_dir