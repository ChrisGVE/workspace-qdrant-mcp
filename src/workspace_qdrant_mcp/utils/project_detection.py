
from ...observability import get_logger
logger = get_logger(__name__)
"""
Intelligent project detection with Git and GitHub integration.

This module provides sophisticated project structure detection capabilities that analyze
Git repositories, GitHub ownership, and directory structures to automatically identify
project hierarchies and relationships. It's designed to work seamlessly with monorepos,
multi-project workspaces, and nested Git repositories.

Key Features:
    - Automatic project name detection from Git remotes and directory structure
    - GitHub user ownership verification for accurate project identification
    - Submodule and nested repository discovery
    - Monorepo support with subproject detection
    - Configurable project naming strategies
    - Robust error handling for various Git repository states

Detection Algorithm:
    1. Traverses directory tree to find Git repository root
    2. Analyzes Git remote URLs for GitHub ownership information
    3. Applies user-specific naming rules when GitHub user is configured
    4. Discovers submodules and nested projects
    5. Generates hierarchical project structure
    6. Falls back to directory-based naming when Git is unavailable

Supported Scenarios:
    - Standard Git repositories with GitHub remotes
    - Monorepos with multiple logical projects
    - Nested Git repositories and submodules
    - Local repositories without remotes
    - Directories without Git initialization
    - Complex ownership scenarios with multiple users

Example:
    ```python
    from workspace_qdrant_mcp.utils.project_detection import ProjectDetector

    # Basic project detection
    detector = ProjectDetector()
    project_name = detector.get_project_name("/path/to/project")

    # GitHub user-aware detection
    detector = ProjectDetector(github_user="username")
    project_info = detector.get_project_info()
    logger.info("Main project: {project_info['main_project']}")
    logger.info("Subprojects: {project_info['subprojects']}")
    ```
"""

import logging
import os
import re
from typing import Any, Optional
from urllib.parse import urlparse

import git
from git.exc import GitError, InvalidGitRepositoryError

logger = logging.getLogger(__name__)


class ProjectDetector:
    """
    Advanced project detection engine with Git and GitHub integration.

    This class provides comprehensive project structure analysis by examining Git
    repositories, remote configurations, directory structures, and ownership patterns.
    It's designed to automatically discover project hierarchies in complex development
    environments including monorepos, nested projects, and multi-user repositories.

    The detector implements a sophisticated algorithm that:
    - Analyzes Git repository structure and remote configurations
    - Applies GitHub user ownership filtering when configured
    - Discovers subprojects through submodules and directory analysis
    - Handles edge cases like missing remotes or complex repository structures
    - Provides fallback mechanisms for non-Git environments

    Attributes:
        github_user (Optional[str]): GitHub username for ownership filtering.
                                   When specified, only repositories owned by this
                                   user will be included as subprojects

    Detection Strategy:
        1. **Git-based**: Uses Git remote URL to determine project name
        2. **Directory-based**: Falls back to directory name when Git unavailable
        3. **User-filtered**: Applies GitHub user ownership rules when configured
        4. **Hierarchical**: Discovers subprojects and maintains relationships

    Example:
        ```python
        # Basic usage
        detector = ProjectDetector()
        name = detector.get_project_name()  # Current directory

        # With GitHub user filtering
        detector = ProjectDetector(github_user="myusername")
        info = detector.get_project_info()

        # Custom path analysis
        subprojects = detector.get_subprojects("/path/to/monorepo")
        ```
    """

    def __init__(self, github_user: str | None = None) -> None:
        """Initialize the project detector with optional GitHub user filtering.

        Args:
            github_user: GitHub username for ownership-based project naming.
                        When provided, repositories owned by this user will use
                        remote-based names, while others use directory names
        """
        self.github_user = github_user

    def get_project_name(self, path: str = ".") -> str:
        """
        Get project name following the PRD algorithm.

        Args:
            path: Path to analyze (defaults to current directory)

        Returns:
            Project name string
        """
        try:
            git_root = self._find_git_root(path)
            if not git_root:
                return os.path.basename(os.path.abspath(path))

            remote_url = self._get_git_remote_url(git_root)
            if self.github_user and remote_url and self._belongs_to_user(remote_url):
                repo_name = self._extract_repo_name_from_remote(remote_url)
                return repo_name if repo_name else os.path.basename(git_root)
            else:
                return os.path.basename(git_root)

        except Exception as e:
            logger.warning("Failed to detect project name from %s: %s", path, e)
            return os.path.basename(os.path.abspath(path))

    def get_project_and_subprojects(self, path: str = ".") -> tuple[str, list[str]]:
        """
        Get main project name and filtered subprojects.

        Args:
            path: Path to analyze

        Returns:
            Tuple of (main_project_name, list_of_subproject_names)
        """
        main_project = self.get_project_name(path)
        subprojects = self.get_subprojects(path)

        return main_project, subprojects

    def get_subprojects(self, path: str = ".") -> list[str]:
        """
        Get list of subprojects (Git submodules filtered by GitHub user).

        Args:
            path: Path to analyze

        Returns:
            List of subproject names
        """
        submodules = self.get_detailed_submodules(path)
        return [sm["project_name"] for sm in submodules if sm["project_name"]]

    def get_detailed_submodules(self, path: str = ".") -> list[dict[str, Any]]:
        """
        Get detailed information about submodules.

        Args:
            path: Path to analyze

        Returns:
            List of submodule information dictionaries
        """
        try:
            git_root = self._find_git_root(path)
            if not git_root:
                return []

            repo = git.Repo(git_root)
            submodules = []

            # Get all submodules
            for submodule in repo.submodules:
                try:
                    submodule_info = self._analyze_submodule(submodule, git_root)
                    if submodule_info:
                        submodules.append(submodule_info)

                except Exception as e:
                    logger.warning(
                        "Failed to process submodule %s: %s", submodule.name, e
                    )
                    continue

            # Sort by project name
            submodules.sort(key=lambda x: x.get("project_name", ""))

            return submodules

        except Exception as e:
            logger.warning("Failed to get submodules from %s: %s", path, e)
            return []

    def _analyze_submodule(
        self, submodule: Any, git_root: str
    ) -> dict[str, Any] | None:
        """Analyze a single submodule and extract information."""
        try:
            submodule_url = submodule.url
            submodule_path = os.path.join(git_root, submodule.path)

            # Parse URL information
            url_info = self._parse_git_url(submodule_url)

            # Check if this submodule belongs to the configured user
            user_owned = self._belongs_to_user(submodule_url)

            # If github_user is configured, only include user-owned subprojects
            # If no github_user is configured, include all subprojects (no filtering)
            if self.github_user and not user_owned:
                return None

            # Extract project name
            project_name = self._extract_repo_name_from_remote(submodule_url)

            # Check if submodule is initialized
            is_initialized = os.path.exists(submodule_path) and bool(
                os.listdir(submodule_path)
            )

            # Try to get commit info
            commit_sha = None
            try:
                commit_sha = submodule.hexsha
            except Exception:
                pass

            return {
                "name": submodule.name,
                "path": submodule.path,
                "url": submodule_url,
                "project_name": project_name,
                "is_initialized": is_initialized,
                "user_owned": user_owned,
                "commit_sha": commit_sha,
                "url_info": url_info,
                "local_path": submodule_path,
            }

        except Exception as e:
            logger.error("Failed to analyze submodule %s: %s", submodule.name, e)
            return None

    def _find_git_root(self, path: str) -> str | None:
        """
        Find the root directory of a Git repository.

        Args:
            path: Starting path

        Returns:
            Git root directory path or None
        """
        try:
            repo = git.Repo(path, search_parent_directories=True)
            working_dir = repo.working_dir
            return str(working_dir) if working_dir else None
        except (InvalidGitRepositoryError, GitError):
            return None

    def _get_git_remote_url(self, git_root: str) -> str | None:
        """
        Get the remote URL for the Git repository.

        Args:
            git_root: Git repository root directory

        Returns:
            Remote URL string or None
        """
        try:
            repo = git.Repo(git_root)

            # Try origin first, then any remote
            for remote_name in ["origin", "upstream"]:
                if hasattr(repo.remotes, remote_name):
                    remote = getattr(repo.remotes, remote_name)
                    return str(remote.url)

            # Fall back to first available remote
            if repo.remotes:
                return str(repo.remotes[0].url)

            return None

        except Exception as e:
            logger.warning("Failed to get remote URL from %s: %s", git_root, e)
            return None

    def _parse_git_url(self, remote_url: str) -> dict[str, Any]:
        """
        Parse a Git remote URL and extract components.

        Args:
            remote_url: Git remote URL

        Returns:
            Dictionary with URL components
        """
        url_info = {
            "original": remote_url,
            "hostname": None,
            "username": None,
            "repository": None,
            "protocol": None,
            "is_github": False,
            "is_ssh": False,
        }

        if not remote_url:
            return url_info

        try:
            # SSH format: git@github.com:user/repo.git
            if remote_url.startswith("git@"):
                url_info["is_ssh"] = True
                url_info["protocol"] = "ssh"

                # Parse SSH format
                ssh_match = re.match(
                    r"git@([^:]+):([^/]+)/(.+?)(?:\.git)?$", remote_url
                )
                if ssh_match:
                    url_info["hostname"] = ssh_match.group(1)
                    url_info["username"] = ssh_match.group(2)
                    url_info["repository"] = ssh_match.group(3)

            # HTTPS/HTTP format: https://github.com/user/repo.git
            elif remote_url.startswith(("http://", "https://")):
                parsed = urlparse(remote_url)
                url_info["protocol"] = parsed.scheme
                url_info["hostname"] = parsed.hostname

                if parsed.path:
                    path_parts = parsed.path.strip("/").split("/")
                    if len(path_parts) >= 2:
                        url_info["username"] = path_parts[0]
                        repo_name = path_parts[1]
                        if repo_name.endswith(".git"):
                            repo_name = repo_name[:-4]
                        url_info["repository"] = repo_name

            # Check if it's GitHub
            if url_info["hostname"] == "github.com":
                url_info["is_github"] = True

        except Exception as e:
            logger.warning("Failed to parse Git URL %s: %s", remote_url, e)

        return url_info

    def _belongs_to_user(self, remote_url: str) -> bool:
        """
        Check if a remote URL belongs to the configured GitHub user.

        Args:
            remote_url: Git remote URL

        Returns:
            True if URL belongs to the user
        """
        if not self.github_user or not remote_url:
            return False

        try:
            url_info = self._parse_git_url(remote_url)
            is_github = url_info.get("is_github", False)
            username = url_info.get("username")
            return bool(is_github and username == self.github_user)

        except Exception as e:
            logger.warning(
                "Failed to check user ownership for URL %s: %s", remote_url, e
            )
            return False

    def _extract_repo_name_from_remote(self, remote_url: str) -> str | None:
        """
        Extract repository name from remote URL.

        Args:
            remote_url: Git remote URL

        Returns:
            Repository name or None
        """
        if not remote_url:
            return None

        try:
            url_info = self._parse_git_url(remote_url)
            return url_info.get("repository")

        except Exception as e:
            logger.warning("Failed to extract repo name from %s: %s", remote_url, e)
            return None

    def get_project_info(self, path: str = ".") -> dict[str, Any]:
        """
        Get comprehensive project information.

        Args:
            path: Path to analyze

        Returns:
            Dictionary with project information
        """
        try:
            main_project, subprojects = self.get_project_and_subprojects(path)
            git_root = self._find_git_root(path)
            remote_url = self._get_git_remote_url(git_root) if git_root else None
            detailed_submodules = self.get_detailed_submodules(path)

            # Parse main project URL info
            main_url_info = self._parse_git_url(remote_url) if remote_url else {}

            return {
                "main_project": main_project,
                "subprojects": subprojects,
                "git_root": git_root,
                "remote_url": remote_url,
                "main_url_info": main_url_info,
                "github_user": self.github_user,
                "path": os.path.abspath(path),
                "is_git_repo": git_root is not None,
                "belongs_to_user": self._belongs_to_user(remote_url)
                if remote_url
                else False,
                "detailed_submodules": detailed_submodules,
                "submodule_count": len(detailed_submodules),
                "user_owned_submodules": [
                    sm for sm in detailed_submodules if sm.get("user_owned", False)
                ],
            }

        except Exception as e:
            logger.error("Failed to get project info from %s: %s", path, e)
            return {
                "main_project": os.path.basename(os.path.abspath(path)),
                "subprojects": [],
                "git_root": None,
                "remote_url": None,
                "main_url_info": {},
                "github_user": self.github_user,
                "path": os.path.abspath(path),
                "is_git_repo": False,
                "belongs_to_user": False,
                "detailed_submodules": [],
                "submodule_count": 0,
                "user_owned_submodules": [],
                "error": str(e),
            }
