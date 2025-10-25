"""Unit tests for git_utils module.

This test suite validates the Git branch detection functionality with comprehensive
coverage of edge cases and error conditions. It mirrors the test coverage from the
Rust implementation.

Test Coverage:
    - Branch detection in git repositories
    - Non-git directories
    - Detached HEAD state
    - Empty repositories (no commits)
    - Branch switching scenarios
    - Subdirectory detection
    - Repository root detection
    - Git repository validation
"""

import os
from pathlib import Path

import pytest
from common.utils.git_utils import (
    DEFAULT_BRANCH,
    get_current_branch,
    get_repository_root,
    is_git_repository,
)
from git import Repo


@pytest.fixture
def temp_git_repo(tmp_path):
    """Create a temporary Git repository with an initial commit.

    This fixture creates a clean Git repository for testing. It includes
    an initial commit so that the repository has a HEAD reference.

    Yields:
        Path to the temporary repository directory
    """
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Initialize Git repository
    repo = Repo.init(repo_path)

    # Configure user for commits
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@example.com").release()

    # Create initial commit (required for branch to exist)
    readme_file = repo_path / "README.md"
    readme_file.write_text("# Test Repository\n")
    repo.index.add([str(readme_file)])
    repo.index.commit("Initial commit")

    yield repo_path

    # Cleanup is automatic with tmp_path


@pytest.fixture
def temp_empty_git_repo(tmp_path):
    """Create a temporary Git repository with no commits.

    This fixture creates a Git repository without any commits to test
    the handling of empty repositories.

    Yields:
        Path to the empty repository directory
    """
    repo_path = tmp_path / "empty_repo"
    repo_path.mkdir()

    # Initialize Git repository but don't create any commits
    Repo.init(repo_path)

    yield repo_path


class TestGetCurrentBranch:
    """Test suite for get_current_branch function."""

    @pytest.mark.requires_git
    def test_detect_branch_in_git_repo(self, temp_git_repo):
        """Test branch detection in a normal Git repository.

        The branch name should be either "main" or "master" depending on
        the Git version (2.28+ defaults to "main", older versions use "master").
        """
        branch = get_current_branch(temp_git_repo)

        # Git 2.28+ defaults to "main", older versions use "master"
        assert branch in ["main", "master"], f"Expected 'main' or 'master', got '{branch}'"

    @pytest.mark.requires_git
    def test_non_git_directory(self, tmp_path):
        """Test branch detection in a non-Git directory.

        Should return DEFAULT_BRANCH ("main") when called on a directory
        that is not part of a Git repository.
        """
        non_git_path = tmp_path / "not_a_repo"
        non_git_path.mkdir()

        branch = get_current_branch(non_git_path)

        assert branch == DEFAULT_BRANCH

    @pytest.mark.requires_git
    def test_empty_repository(self, temp_empty_git_repo):
        """Test branch detection in a repository with no commits.

        Git repositories without commits don't have a valid HEAD reference,
        so the function should return DEFAULT_BRANCH.
        """
        branch = get_current_branch(temp_empty_git_repo)

        assert branch == DEFAULT_BRANCH

    @pytest.mark.requires_git
    def test_detached_head_state(self, temp_git_repo):
        """Test branch detection when repository is in detached HEAD state.

        When HEAD points directly to a commit instead of a branch,
        the function should return DEFAULT_BRANCH.
        """
        repo = Repo(temp_git_repo)

        # Get the commit SHA to checkout
        head_commit = repo.head.commit

        # Detach HEAD by checking out the commit directly
        repo.head.reference = head_commit
        repo.head.reset(index=True, working_tree=True)

        branch = get_current_branch(temp_git_repo)

        assert branch == DEFAULT_BRANCH

    @pytest.mark.requires_git
    def test_branch_switching(self, temp_git_repo):
        """Test branch detection after switching branches.

        Creates a new branch, switches to it, and verifies that the
        branch detection follows the switch.
        """
        repo = Repo(temp_git_repo)

        # Get initial branch name
        initial_branch = get_current_branch(temp_git_repo)

        # Create and switch to new branch
        new_branch_name = "feature/test-branch"
        new_branch = repo.create_head(new_branch_name)
        new_branch.checkout()

        # Verify branch detection shows new branch
        current_branch = get_current_branch(temp_git_repo)

        assert current_branch == new_branch_name
        assert current_branch != initial_branch

    @pytest.mark.requires_git
    def test_subdirectory_detection(self, temp_git_repo):
        """Test branch detection from a subdirectory within the repository.

        GitPython should search parent directories to find the repository root,
        so branch detection should work from any subdirectory.
        """
        # Create nested subdirectory structure
        subdir = temp_git_repo / "src" / "lib" / "utils"
        subdir.mkdir(parents=True)

        # Branch detection should work from subdirectory
        branch = get_current_branch(subdir)

        assert branch in ["main", "master"]

    @pytest.mark.requires_git
    def test_nonexistent_path(self, tmp_path):
        """Test branch detection with a path that doesn't exist.

        Should handle the error gracefully and return DEFAULT_BRANCH.
        """
        nonexistent_path = tmp_path / "does_not_exist"

        branch = get_current_branch(nonexistent_path)

        assert branch == DEFAULT_BRANCH

    @pytest.mark.requires_git
    def test_branch_with_slashes(self, temp_git_repo):
        """Test branch detection with branch names containing slashes.

        Branch names like "feature/new-api" are common and should be
        detected correctly.
        """
        repo = Repo(temp_git_repo)

        # Create branch with slashes
        branch_name = "feature/auth/jwt-tokens"
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()

        detected_branch = get_current_branch(temp_git_repo)

        assert detected_branch == branch_name


class TestGetRepositoryRoot:
    """Test suite for get_repository_root function."""

    @pytest.mark.requires_git
    def test_get_root_from_repo_directory(self, temp_git_repo):
        """Test getting repository root from the repository directory itself."""
        root = get_repository_root(temp_git_repo)

        assert root == temp_git_repo

    @pytest.mark.requires_git
    def test_get_root_from_subdirectory(self, temp_git_repo):
        """Test getting repository root from a subdirectory."""
        subdir = temp_git_repo / "src" / "lib"
        subdir.mkdir(parents=True)

        root = get_repository_root(subdir)

        assert root == temp_git_repo

    @pytest.mark.requires_git
    def test_get_root_from_non_git_directory(self, tmp_path):
        """Test getting repository root from a non-Git directory."""
        non_git_path = tmp_path / "not_a_repo"
        non_git_path.mkdir()

        root = get_repository_root(non_git_path)

        assert root is None

    @pytest.mark.requires_git
    def test_get_root_from_nonexistent_path(self, tmp_path):
        """Test getting repository root from a nonexistent path."""
        nonexistent = tmp_path / "does_not_exist"

        root = get_repository_root(nonexistent)

        assert root is None


class TestIsGitRepository:
    """Test suite for is_git_repository function."""

    @pytest.mark.requires_git
    def test_is_git_repo_positive(self, temp_git_repo):
        """Test that a Git repository is correctly identified."""
        assert is_git_repository(temp_git_repo) is True

    @pytest.mark.requires_git
    def test_is_git_repo_from_subdirectory(self, temp_git_repo):
        """Test Git repository detection from a subdirectory."""
        subdir = temp_git_repo / "src"
        subdir.mkdir()

        assert is_git_repository(subdir) is True

    @pytest.mark.requires_git
    def test_is_git_repo_negative(self, tmp_path):
        """Test that a non-Git directory is correctly identified."""
        non_git_path = tmp_path / "not_a_repo"
        non_git_path.mkdir()

        assert is_git_repository(non_git_path) is False

    @pytest.mark.requires_git
    def test_is_git_repo_nonexistent_path(self, tmp_path):
        """Test Git repository detection with nonexistent path."""
        nonexistent = tmp_path / "does_not_exist"

        assert is_git_repository(nonexistent) is False

    @pytest.mark.requires_git
    def test_is_git_repo_empty_repo(self, temp_empty_git_repo):
        """Test that an empty Git repository is still identified as a repo."""
        assert is_git_repository(temp_empty_git_repo) is True


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    @pytest.mark.requires_git
    def test_symlink_to_git_repo(self, temp_git_repo, tmp_path):
        """Test branch detection through a symbolic link to a Git repository."""
        symlink_path = tmp_path / "repo_symlink"
        symlink_path.symlink_to(temp_git_repo)

        branch = get_current_branch(symlink_path)

        assert branch in ["main", "master"]

    @pytest.mark.requires_git
    def test_relative_path(self, temp_git_repo):
        """Test branch detection with a relative path.

        The function should handle relative paths correctly by resolving them.
        """
        # Change to parent directory and use relative path
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_git_repo.parent)
            relative_path = Path(temp_git_repo.name)

            branch = get_current_branch(relative_path)

            assert branch in ["main", "master"]
        finally:
            os.chdir(original_cwd)

    @pytest.mark.requires_git
    def test_multiple_calls_same_repo(self, temp_git_repo):
        """Test that multiple calls to get_current_branch return consistent results.

        This validates that the function is stateless and doesn't have
        side effects that affect subsequent calls.
        """
        branch1 = get_current_branch(temp_git_repo)
        branch2 = get_current_branch(temp_git_repo)
        branch3 = get_current_branch(temp_git_repo)

        assert branch1 == branch2 == branch3

    @pytest.mark.requires_git
    def test_current_directory_is_repo(self, temp_git_repo):
        """Test branch detection using current directory (Path('.'))."""
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_git_repo)

            branch = get_current_branch(Path("."))

            assert branch in ["main", "master"]
        finally:
            os.chdir(original_cwd)


class TestDefaultBranchConstant:
    """Test suite for the DEFAULT_BRANCH constant."""

    def test_default_branch_value(self):
        """Verify that DEFAULT_BRANCH is set to 'main'."""
        assert DEFAULT_BRANCH == "main"


# Integration test to verify Python implementation matches Rust behavior
class TestPythonRustParity:
    """Test suite to verify Python implementation matches Rust behavior.

    These tests ensure that the Python implementation behaves identically
    to the Rust implementation in all scenarios.
    """

    @pytest.mark.requires_git
    def test_all_error_cases_return_main(self, tmp_path):
        """Verify that all error conditions return 'main' as default.

        This matches the Rust implementation's behavior of returning "main"
        for any error condition.
        """
        # Non-git directory
        assert get_current_branch(tmp_path / "not_a_repo") == "main"

        # Nonexistent path
        assert get_current_branch(tmp_path / "does_not_exist") == "main"

    @pytest.mark.requires_git
    def test_branch_detection_consistency(self, temp_git_repo):
        """Verify consistent branch detection across different call patterns."""
        # Direct call
        branch1 = get_current_branch(temp_git_repo)

        # Call from subdirectory reference
        subdir = temp_git_repo / "src"
        subdir.mkdir()
        branch2 = get_current_branch(subdir)

        # Both should return the same branch
        assert branch1 == branch2
        assert branch1 in ["main", "master"]
