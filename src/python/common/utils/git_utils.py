"""Git utility functions for branch detection and repository operations.

This module provides Git-related utility functions with a focus on branch detection
for multi-branch project support. It mirrors the functionality of the Rust
git_integration module.

Key Features:
    - Current branch detection from any directory within a Git repository
    - Graceful fallback to "main" for edge cases
    - Handles detached HEAD states, non-git directories, and empty repositories
    - Uses GitPython for reliable Git operations
    - Logging for debugging and error tracking

The implementation follows these principles:
    1. Return "main" as the default for all error cases
    2. Log warnings for expected edge cases (non-git, detached HEAD)
    3. Log errors for unexpected failures
    4. Never raise exceptions - always return a branch name

Example:
    ```python
    from common.utils.git_utils import get_current_branch
    from pathlib import Path

    # Detect branch in git repository
    branch = get_current_branch(Path("/path/to/repo"))
    # Returns: "feature/auth" or "main"

    # Non-git directory
    branch = get_current_branch(Path("/tmp"))
    # Returns: "main" (with warning logged)

    # Detached HEAD
    branch = get_current_branch(Path("/repo/in/detached/state"))
    # Returns: "main" (with warning logged)
    ```
"""

import logging
from pathlib import Path
from typing import Optional

from git import InvalidGitRepositoryError, Repo
from git.exc import GitCommandError

logger = logging.getLogger(__name__)

DEFAULT_BRANCH = "main"


def get_current_branch(repo_path: Path) -> str:
    """Get the current Git branch name for a repository.

    This function detects the current Git branch for any directory within a Git
    repository. It handles various edge cases gracefully by returning "main" as
    the default branch name.

    The function follows this detection algorithm:
        1. Search for Git repository starting from repo_path (walks up parent dirs)
        2. Get HEAD reference to determine current branch
        3. Return branch name if on a branch
        4. Return "main" for all error/edge cases

    Edge Cases Handled:
        - **Non-Git directory**: Returns "main" with warning log
        - **Git repo with no commits**: Returns "main" with warning log
        - **Detached HEAD state**: Returns "main" with warning log
        - **Permission errors**: Returns "main" with error log
        - **Invalid UTF-8 in branch name**: Returns "main" with error log
        - **Any other Git error**: Returns "main" with error log

    Args:
        repo_path: Path to a directory within a Git repository. Can be any
                  subdirectory - GitPython will search parent directories
                  to find the repository root.

    Returns:
        Current Git branch name (e.g., "main", "feature/auth", "develop")
        or "main" if detection fails or repo is not a Git repository.

    Notes:
        - This function never raises exceptions
        - All error cases are logged and return "main"
        - The function is safe to call on any directory
        - Symbolic refs are resolved to branch names

    Example:
        >>> from pathlib import Path
        >>> get_current_branch(Path("/path/to/repo"))
        'feature/new-api'

        >>> get_current_branch(Path("/not/a/git/repo"))
        'main'  # With warning logged

        >>> get_current_branch(Path("/repo/in/detached/head"))
        'main'  # With warning logged
    """
    try:
        # Try to open the Git repository
        # search_parent_directories=True allows detecting repo from subdirectories
        repo = Repo(repo_path, search_parent_directories=True)

        # Check if repository has any commits
        # An empty repository (no commits) will have an empty HEAD
        try:
            _ = repo.head.commit
        except ValueError as e:
            # Repository exists but has no commits yet
            logger.warning(
                "Git repository has no commits yet, defaulting to '%s': %s",
                DEFAULT_BRANCH,
                repo_path,
            )
            return DEFAULT_BRANCH

        # Check if HEAD is detached
        if repo.head.is_detached:
            logger.warning(
                "Git repository in detached HEAD state, defaulting to '%s': %s",
                DEFAULT_BRANCH,
                repo_path,
            )
            return DEFAULT_BRANCH

        # Get the current branch reference
        # This returns the symbolic reference (e.g., "refs/heads/main")
        head_ref = repo.head.ref

        # Extract branch name from the reference
        # head_ref.name gives us just the branch name without "refs/heads/"
        branch_name = head_ref.name

        logger.debug(
            "Detected Git branch '%s' for repository at %s",
            branch_name,
            repo_path,
        )

        return branch_name

    except InvalidGitRepositoryError:
        # Not a Git repository - this is expected for non-git directories
        logger.warning(
            "Not a Git repository, defaulting to '%s': %s",
            DEFAULT_BRANCH,
            repo_path,
        )
        return DEFAULT_BRANCH

    except GitCommandError as e:
        # Git command failed - could be permissions, corrupted repo, etc.
        logger.error(
            "Git command failed, defaulting to '%s': %s - Error: %s",
            DEFAULT_BRANCH,
            repo_path,
            e,
        )
        return DEFAULT_BRANCH

    except (OSError, IOError) as e:
        # File system errors - permissions, missing files, etc.
        logger.error(
            "File system error accessing Git repository, defaulting to '%s': %s - Error: %s",
            DEFAULT_BRANCH,
            repo_path,
            e,
        )
        return DEFAULT_BRANCH

    except Exception as e:
        # Catch-all for any other unexpected errors
        logger.error(
            "Unexpected error detecting Git branch, defaulting to '%s': %s - Error: %s",
            DEFAULT_BRANCH,
            repo_path,
            e,
            exc_info=True,  # Include stack trace for unexpected errors
        )
        return DEFAULT_BRANCH


def get_repository_root(repo_path: Path) -> Optional[Path]:
    """Get the root directory of a Git repository.

    This is a utility function that returns the repository root directory
    for a given path within a Git repository. Returns None if the path
    is not within a Git repository.

    Args:
        repo_path: Path to a directory within a Git repository

    Returns:
        Path to the repository root, or None if not a Git repository

    Example:
        >>> get_repository_root(Path("/path/to/repo/src/utils"))
        Path("/path/to/repo")

        >>> get_repository_root(Path("/not/a/repo"))
        None
    """
    try:
        repo = Repo(repo_path, search_parent_directories=True)
        if repo.working_dir:
            return Path(repo.working_dir)
        return None
    except (InvalidGitRepositoryError, GitCommandError, Exception):
        return None


def is_git_repository(repo_path: Path) -> bool:
    """Check if a path is within a Git repository.

    Args:
        repo_path: Path to check

    Returns:
        True if path is within a Git repository, False otherwise

    Example:
        >>> is_git_repository(Path("/path/to/repo"))
        True

        >>> is_git_repository(Path("/tmp"))
        False
    """
    try:
        Repo(repo_path, search_parent_directories=True)
        return True
    except (InvalidGitRepositoryError, Exception):
        return False
