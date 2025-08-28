"""
Unit tests for project detection functionality.

Tests Git repository detection, project naming, and submodule handling.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import pytest
from git.exc import GitError

from workspace_qdrant_mcp.utils.project_detection import ProjectDetector


class TestProjectDetector:
    """Test ProjectDetector class."""

    def test_init_without_github_user(self):
        """Test initialization without GitHub user."""
        detector = ProjectDetector()
        assert detector.github_user is None

    def test_init_with_github_user(self):
        """Test initialization with GitHub user."""
        detector = ProjectDetector(github_user="testuser")
        assert detector.github_user == "testuser"

    def test_get_project_name_non_git_directory(self):
        """Test project name detection for non-Git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = ProjectDetector()
            project_name = detector.get_project_name(temp_dir)

            expected_name = os.path.basename(temp_dir)
            assert project_name == expected_name

    def test_get_project_name_git_directory_no_remote(self, temp_git_repo):
        """Test project name detection for Git directory without remote."""
        # Remove the remote we added in the fixture
        repo = git.Repo(temp_git_repo)
        repo.delete_remote("origin")

        detector = ProjectDetector()
        project_name = detector.get_project_name(temp_git_repo)

        expected_name = os.path.basename(temp_git_repo)
        assert project_name == expected_name

    def test_get_project_name_git_with_github_user_match(self, temp_git_repo):
        """Test project name detection with matching GitHub user."""
        detector = ProjectDetector(github_user="testuser")
        project_name = detector.get_project_name(temp_git_repo)

        assert project_name == "test-project"

    def test_get_project_name_git_with_github_user_no_match(self, temp_git_repo):
        """Test project name detection with non-matching GitHub user."""
        detector = ProjectDetector(github_user="otheruser")
        project_name = detector.get_project_name(temp_git_repo)

        # Should fall back to directory name
        expected_name = os.path.basename(temp_git_repo)
        assert project_name == expected_name

    def test_get_project_name_git_no_github_user(self, temp_git_repo):
        """Test project name detection without GitHub user configured."""
        detector = ProjectDetector()
        project_name = detector.get_project_name(temp_git_repo)

        # Should fall back to directory name
        expected_name = os.path.basename(temp_git_repo)
        assert project_name == expected_name

    @patch("workspace_qdrant_mcp.utils.project_detection.git.Repo")
    def test_get_project_name_git_error(self, mock_repo_class):
        """Test project name detection when Git operations fail."""
        mock_repo_class.side_effect = GitError("Git error")

        detector = ProjectDetector()
        project_name = detector.get_project_name("/some/path")

        assert project_name == "path"

    def test_get_subprojects_no_git(self):
        """Test subproject detection for non-Git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = ProjectDetector()
            subprojects = detector.get_subprojects(temp_dir)

            assert subprojects == []

    def test_get_subprojects_no_submodules(self, temp_git_repo):
        """Test subproject detection for Git repo without submodules."""
        detector = ProjectDetector()
        subprojects = detector.get_subprojects(temp_git_repo)

        assert subprojects == []

    def test_get_subprojects_with_submodules_no_filter(
        self, temp_git_repo_with_submodules
    ):
        """Test subproject detection with submodules, no user filtering."""
        detector = ProjectDetector()

        # Mock the submodules
        with patch(
            "workspace_qdrant_mcp.utils.project_detection.git.Repo"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            # Create mock submodules
            mock_submodule1 = MagicMock()
            mock_submodule1.name = "subproject1"
            mock_submodule1.path = "subproject1"
            mock_submodule1.url = "https://github.com/testuser/subproject1.git"
            mock_submodule1.hexsha = "abc123"

            mock_submodule2 = MagicMock()
            mock_submodule2.name = "subproject2"
            mock_submodule2.path = "libs/subproject2"
            mock_submodule2.url = "https://github.com/otheruser/subproject2.git"
            mock_submodule2.hexsha = "def456"

            mock_repo.submodules = [mock_submodule1, mock_submodule2]

            # Mock os.path.exists and os.listdir for initialized check
            with (
                patch("os.path.exists", return_value=True),
                patch("os.listdir", return_value=["file.txt"]),
            ):
                subprojects = detector.get_subprojects(temp_git_repo_with_submodules)

                # Should return both subprojects when no user filtering
                expected = ["subproject1", "subproject2"]
                assert sorted(subprojects) == sorted(expected)

    def test_get_subprojects_with_user_filtering(self, temp_git_repo_with_submodules):
        """Test subproject detection with GitHub user filtering."""
        detector = ProjectDetector(github_user="testuser")

        # Mock the submodules
        with patch(
            "workspace_qdrant_mcp.utils.project_detection.git.Repo"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            # Create mock submodules - one matching user, one not
            mock_submodule1 = MagicMock()
            mock_submodule1.name = "subproject1"
            mock_submodule1.path = "subproject1"
            mock_submodule1.url = "https://github.com/testuser/subproject1.git"
            mock_submodule1.hexsha = "abc123"

            mock_submodule2 = MagicMock()
            mock_submodule2.name = "subproject2"
            mock_submodule2.path = "libs/subproject2"
            mock_submodule2.url = "https://github.com/otheruser/subproject2.git"
            mock_submodule2.hexsha = "def456"

            mock_repo.submodules = [mock_submodule1, mock_submodule2]

            # Mock os.path.exists and os.listdir for initialized check
            with (
                patch("os.path.exists", return_value=True),
                patch("os.listdir", return_value=["file.txt"]),
            ):
                subprojects = detector.get_subprojects(temp_git_repo_with_submodules)

                # Should only return user-owned subproject
                assert subprojects == ["subproject1"]

    def test_get_project_and_subprojects(self, temp_git_repo):
        """Test getting both main project and subprojects."""
        detector = ProjectDetector(github_user="testuser")

        main_project, subprojects = detector.get_project_and_subprojects(temp_git_repo)

        assert main_project == "test-project"
        assert isinstance(subprojects, list)

    def test_parse_git_url_https(self):
        """Test parsing HTTPS Git URL."""
        detector = ProjectDetector()
        url = "https://github.com/testuser/testrepo.git"

        url_info = detector._parse_git_url(url)

        assert url_info["original"] == url
        assert url_info["hostname"] == "github.com"
        assert url_info["username"] == "testuser"
        assert url_info["repository"] == "testrepo"
        assert url_info["protocol"] == "https"
        assert url_info["is_github"] is True
        assert url_info["is_ssh"] is False

    def test_parse_git_url_ssh(self):
        """Test parsing SSH Git URL."""
        detector = ProjectDetector()
        url = "git@github.com:testuser/testrepo.git"

        url_info = detector._parse_git_url(url)

        assert url_info["original"] == url
        assert url_info["hostname"] == "github.com"
        assert url_info["username"] == "testuser"
        assert url_info["repository"] == "testrepo"
        assert url_info["protocol"] == "ssh"
        assert url_info["is_github"] is True
        assert url_info["is_ssh"] is True

    def test_parse_git_url_non_github(self):
        """Test parsing non-GitHub Git URL."""
        detector = ProjectDetector()
        url = "https://gitlab.com/testuser/testrepo.git"

        url_info = detector._parse_git_url(url)

        assert url_info["hostname"] == "gitlab.com"
        assert url_info["username"] == "testuser"
        assert url_info["repository"] == "testrepo"
        assert url_info["is_github"] is False

    def test_parse_git_url_no_git_suffix(self):
        """Test parsing Git URL without .git suffix."""
        detector = ProjectDetector()
        url = "https://github.com/testuser/testrepo"

        url_info = detector._parse_git_url(url)

        assert url_info["repository"] == "testrepo"

    def test_parse_git_url_invalid(self):
        """Test parsing invalid Git URL."""
        detector = ProjectDetector()
        url = "invalid-url"

        url_info = detector._parse_git_url(url)

        assert url_info["original"] == url
        assert url_info["hostname"] is None
        assert url_info["username"] is None
        assert url_info["repository"] is None

    def test_parse_git_url_empty(self):
        """Test parsing empty Git URL."""
        detector = ProjectDetector()
        url_info = detector._parse_git_url("")

        assert url_info["original"] == ""
        assert url_info["hostname"] is None

    def test_belongs_to_user_github_match(self):
        """Test user ownership check with matching GitHub user."""
        detector = ProjectDetector(github_user="testuser")
        url = "https://github.com/testuser/testrepo.git"

        assert detector._belongs_to_user(url) is True

    def test_belongs_to_user_github_no_match(self):
        """Test user ownership check with non-matching GitHub user."""
        detector = ProjectDetector(github_user="testuser")
        url = "https://github.com/otheruser/testrepo.git"

        assert detector._belongs_to_user(url) is False

    def test_belongs_to_user_no_github_user_configured(self):
        """Test user ownership check with no GitHub user configured."""
        detector = ProjectDetector()
        url = "https://github.com/testuser/testrepo.git"

        assert detector._belongs_to_user(url) is False

    def test_belongs_to_user_non_github(self):
        """Test user ownership check with non-GitHub URL."""
        detector = ProjectDetector(github_user="testuser")
        url = "https://gitlab.com/testuser/testrepo.git"

        assert detector._belongs_to_user(url) is False

    def test_extract_repo_name_from_remote(self):
        """Test repository name extraction from remote URL."""
        detector = ProjectDetector()

        test_cases = [
            ("https://github.com/user/repo.git", "repo"),
            ("git@github.com:user/repo.git", "repo"),
            ("https://github.com/user/repo", "repo"),
            ("", None),
            ("invalid-url", None),
        ]

        for url, expected in test_cases:
            result = detector._extract_repo_name_from_remote(url)
            assert result == expected, f"Failed for URL: {url}"

    def test_find_git_root_valid_repo(self, temp_git_repo):
        """Test finding Git root in valid repository."""
        detector = ProjectDetector()
        git_root = detector._find_git_root(temp_git_repo)

        assert git_root == temp_git_repo

    def test_find_git_root_non_git_directory(self):
        """Test finding Git root in non-Git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = ProjectDetector()
            git_root = detector._find_git_root(temp_dir)

            assert git_root is None

    def test_find_git_root_subdirectory(self, temp_git_repo):
        """Test finding Git root from subdirectory."""
        # Create a subdirectory
        subdir = Path(temp_git_repo) / "subdir"
        subdir.mkdir()

        detector = ProjectDetector()
        git_root = detector._find_git_root(str(subdir))

        assert git_root == temp_git_repo

    def test_get_git_remote_url_with_origin(self, temp_git_repo):
        """Test getting remote URL when origin exists."""
        detector = ProjectDetector()
        remote_url = detector._get_git_remote_url(temp_git_repo)

        assert remote_url == "https://github.com/testuser/test-project.git"

    def test_get_git_remote_url_no_origin(self, temp_git_repo):
        """Test getting remote URL when no origin exists."""
        # Remove origin and add a different remote
        repo = git.Repo(temp_git_repo)
        repo.delete_remote("origin")
        repo.create_remote("upstream", "https://github.com/upstream/project.git")

        detector = ProjectDetector()
        remote_url = detector._get_git_remote_url(temp_git_repo)

        assert remote_url == "https://github.com/upstream/project.git"

    def test_get_git_remote_url_no_remotes(self, temp_git_repo):
        """Test getting remote URL when no remotes exist."""
        # Remove all remotes
        repo = git.Repo(temp_git_repo)
        for remote in repo.remotes:
            repo.delete_remote(remote)

        detector = ProjectDetector()
        remote_url = detector._get_git_remote_url(temp_git_repo)

        assert remote_url is None

    def test_get_project_info_comprehensive(self, temp_git_repo):
        """Test comprehensive project information gathering."""
        detector = ProjectDetector(github_user="testuser")
        project_info = detector.get_project_info(temp_git_repo)

        # Check all expected fields
        expected_fields = [
            "main_project",
            "subprojects",
            "git_root",
            "remote_url",
            "main_url_info",
            "github_user",
            "path",
            "is_git_repo",
            "belongs_to_user",
            "detailed_submodules",
            "submodule_count",
            "user_owned_submodules",
        ]

        for field in expected_fields:
            assert field in project_info, f"Missing field: {field}"

        # Check specific values
        assert project_info["main_project"] == "test-project"
        assert project_info["git_root"] == temp_git_repo
        assert project_info["is_git_repo"] is True
        assert project_info["belongs_to_user"] is True
        assert project_info["github_user"] == "testuser"

    def test_get_project_info_non_git_directory(self):
        """Test project info for non-Git directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = ProjectDetector()
            project_info = detector.get_project_info(temp_dir)

            assert project_info["is_git_repo"] is False
            assert project_info["git_root"] is None
            assert project_info["remote_url"] is None
            assert project_info["belongs_to_user"] is False
            assert project_info["main_project"] == os.path.basename(temp_dir)

    @patch("workspace_qdrant_mcp.utils.project_detection.logger")
    def test_get_project_info_with_error(self, mock_logger):
        """Test project info gathering when errors occur."""
        detector = ProjectDetector()

        with patch.object(
            detector, "_find_git_root", side_effect=Exception("Test error")
        ):
            project_info = detector.get_project_info("/some/path")

            # Should return error info but not crash
            assert "error" in project_info
            assert project_info["main_project"] == "path"
            mock_logger.error.assert_called_once()

    def test_detailed_submodules_analysis(self, temp_git_repo_with_submodules):
        """Test detailed submodule analysis."""
        detector = ProjectDetector(github_user="testuser")

        # Mock the submodules with detailed info
        with patch(
            "workspace_qdrant_mcp.utils.project_detection.git.Repo"
        ) as mock_repo_class:
            mock_repo = MagicMock()
            mock_repo_class.return_value = mock_repo

            # Create detailed mock submodule
            mock_submodule = MagicMock()
            mock_submodule.name = "test-submodule"
            mock_submodule.path = "libs/test-submodule"
            mock_submodule.url = "https://github.com/testuser/test-submodule.git"
            mock_submodule.hexsha = "abc123def456"

            mock_repo.submodules = [mock_submodule]

            # Mock file system checks
            with (
                patch("os.path.exists", return_value=True),
                patch("os.listdir", return_value=["file1.txt", "file2.py"]),
            ):
                detailed = detector.get_detailed_submodules(
                    temp_git_repo_with_submodules
                )

                assert len(detailed) == 1
                submodule_info = detailed[0]

                expected_fields = [
                    "name",
                    "path",
                    "url",
                    "project_name",
                    "is_initialized",
                    "user_owned",
                    "commit_sha",
                    "url_info",
                    "local_path",
                ]

                for field in expected_fields:
                    assert field in submodule_info, f"Missing field: {field}"

                assert submodule_info["name"] == "test-submodule"
                assert submodule_info["project_name"] == "test-submodule"
                assert submodule_info["is_initialized"] is True
                assert submodule_info["user_owned"] is True

    @pytest.mark.parametrize(
        "url,expected_components",
        [
            (
                "https://github.com/user/repo.git",
                {
                    "hostname": "github.com",
                    "username": "user",
                    "repository": "repo",
                    "protocol": "https",
                    "is_github": True,
                    "is_ssh": False,
                },
            ),
            (
                "git@github.com:user/repo.git",
                {
                    "hostname": "github.com",
                    "username": "user",
                    "repository": "repo",
                    "protocol": "ssh",
                    "is_github": True,
                    "is_ssh": True,
                },
            ),
            (
                "https://gitlab.com/user/repo",
                {
                    "hostname": "gitlab.com",
                    "username": "user",
                    "repository": "repo",
                    "protocol": "https",
                    "is_github": False,
                    "is_ssh": False,
                },
            ),
        ],
    )
    def test_parse_git_url_parametrized(self, url, expected_components):
        """Test Git URL parsing with various URL formats."""
        detector = ProjectDetector()
        url_info = detector._parse_git_url(url)

        for key, expected_value in expected_components.items():
            assert url_info[key] == expected_value, f"Failed for {key} in URL {url}"
