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

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import git
from git.exc import GitError, InvalidGitRepositoryError

from ..core.pattern_manager import PatternManager

logger = logging.getLogger(__name__)


class DaemonIdentifier:
    """
    Project-specific daemon identifier with collision detection and validation.

    This class generates unique, consistent identifiers for daemon instances based on
    project information, with built-in collision detection and validation to ensure
    proper isolation between multiple daemon instances.

    Key Features:
        - Consistent identifier generation from project path and name
        - Hash-based collision detection with configurable length
        - Validation for duplicate identifiers across projects
        - Configurable naming strategies with user-defined suffixes
        - Registry tracking for active daemon identifiers

    Identifier Format:
        {project_name}_{path_hash}[_{suffix}]

    Example:
        workspace-qdrant-mcp_a1b2c3d4
        my-project_x9y8z7w6_dev
    """

    # Class-level registry to track active identifiers
    _active_identifiers: set[str] = set()
    _identifier_registry: dict[str, dict[str, Any]] = {}

    def __init__(self, project_name: str, project_path: str, suffix: str | None = None):
        """Initialize daemon identifier with project information.

        Args:
            project_name: Base project name for the identifier
            project_path: Full path to the project directory
            suffix: Optional suffix for custom identification (e.g., 'dev', 'test')
        """
        self.project_name = project_name
        self.project_path = os.path.abspath(project_path)
        self.suffix = suffix
        self._identifier = None
        self._path_hash = None

    def generate_identifier(self, hash_length: int = 8) -> str:
        """Generate a unique daemon identifier for this project.

        Args:
            hash_length: Length of the path hash component (default: 8)

        Returns:
            Unique daemon identifier string

        Raises:
            ValueError: If identifier collision is detected
        """
        if self._identifier:
            return self._identifier

        # Generate path hash from absolute path
        self._path_hash = self._generate_path_hash(self.project_path, hash_length)

        # Build identifier components
        base_identifier = f"{self.project_name}_{self._path_hash}"

        if self.suffix:
            full_identifier = f"{base_identifier}_{self.suffix}"
        else:
            full_identifier = base_identifier

        # Validate uniqueness
        if self._check_collision(full_identifier):
            # Try with longer hash if collision detected
            if hash_length < 16:
                logger.warning(
                    "Identifier collision detected for %s, trying longer hash",
                    full_identifier
                )
                return self.generate_identifier(hash_length + 4)
            else:
                raise ValueError(
                    f"Cannot generate unique identifier for project {self.project_name} "
                    f"at path {self.project_path}. Consider using a suffix."
                )

        self._identifier = full_identifier
        self._register_identifier()

        logger.debug(
            "Generated daemon identifier",
            identifier=self._identifier,
            project=self.project_name,
            path=self.project_path,
            hash=self._path_hash
        )

        return self._identifier

    def get_identifier(self) -> str | None:
        """Get the current identifier without generating a new one.

        Returns:
            Current identifier or None if not yet generated
        """
        return self._identifier

    def get_path_hash(self) -> str | None:
        """Get the path hash component of the identifier.

        Returns:
            Path hash string or None if not yet generated
        """
        return self._path_hash

    def validate_identifier(self, identifier: str) -> bool:
        """Validate that an identifier follows the expected format.

        Args:
            identifier: Identifier string to validate

        Returns:
            True if identifier is valid, False otherwise
        """
        # Basic format validation: name_hash or name_hash_suffix
        pattern = r'^[a-zA-Z0-9_-]+_[a-f0-9]{4,16}(?:_[a-zA-Z0-9_-]+)?$'

        if not re.match(pattern, identifier):
            return False

        # Additional validation: check if it matches our project
        if self._identifier and identifier == self._identifier:
            return True

        # Check if it's a valid identifier for this project path
        parts = identifier.split('_')
        if len(parts) >= 2:
            parts[0]
            hash_part = parts[1]

            # Verify the hash matches our project path
            expected_hash = self._generate_path_hash(self.project_path, len(hash_part))
            return hash_part == expected_hash

        return False

    def release_identifier(self) -> None:
        """Release the current identifier from the active registry."""
        if self._identifier and self._identifier in self._active_identifiers:
            self._active_identifiers.remove(self._identifier)
            if self._identifier in self._identifier_registry:
                del self._identifier_registry[self._identifier]

            logger.debug(
                "Released daemon identifier",
                identifier=self._identifier,
                project=self.project_name
            )

    def _generate_path_hash(self, path: str, length: int = 8) -> str:
        """Generate a consistent hash from the project path.

        Args:
            path: Project path to hash
            length: Desired hash length

        Returns:
            Hexadecimal hash string
        """
        # Normalize path for consistent hashing
        normalized_path = os.path.normpath(os.path.abspath(path))

        # Use SHA-256 for consistent, collision-resistant hashing
        hash_obj = hashlib.sha256(normalized_path.encode('utf-8'))
        return hash_obj.hexdigest()[:length]

    def _check_collision(self, identifier: str) -> bool:
        """Check if an identifier collides with existing ones.

        Args:
            identifier: Identifier to check

        Returns:
            True if collision detected, False otherwise
        """
        if identifier in self._active_identifiers:
            # Check if it's the same project path (allowed)
            existing_info = self._identifier_registry.get(identifier)
            if existing_info and existing_info['project_path'] == self.project_path:
                return False  # Same project, not a collision
            return True  # Different project, collision detected
        return False

    def _register_identifier(self) -> None:
        """Register the current identifier in the active registry."""
        if self._identifier:
            self._active_identifiers.add(self._identifier)
            self._identifier_registry[self._identifier] = {
                'project_name': self.project_name,
                'project_path': self.project_path,
                'suffix': self.suffix,
                'path_hash': self._path_hash,
                'registered_at': os.getcwd(),  # Current working directory when registered
            }

    @classmethod
    def get_active_identifiers(cls) -> set[str]:
        """Get all currently active daemon identifiers.

        Returns:
            Set of active identifier strings
        """
        return cls._active_identifiers.copy()

    @classmethod
    def get_identifier_info(cls, identifier: str) -> dict[str, Any] | None:
        """Get information about a registered identifier.

        Args:
            identifier: Identifier to look up

        Returns:
            Dictionary with identifier information or None if not found
        """
        return cls._identifier_registry.get(identifier)

    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered identifiers (for testing/cleanup)."""
        cls._active_identifiers.clear()
        cls._identifier_registry.clear()

    def __str__(self) -> str:
        """String representation of the daemon identifier."""
        if self._identifier:
            return self._identifier
        return f"DaemonIdentifier(project={self.project_name}, ungenerated)"

    def __repr__(self) -> str:
        """Detailed representation of the daemon identifier."""
        return (f"DaemonIdentifier(project_name='{self.project_name}', "
                f"project_path='{self.project_path}', suffix='{self.suffix}', "
                f"identifier='{self._identifier}')")


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

    def __init__(
        self,
        github_user: str | None = None,
        pattern_manager: PatternManager | None = None
    ) -> None:
        """Initialize the project detector with optional GitHub user filtering.

        Args:
            github_user: GitHub username for ownership-based project naming.
                        When provided, repositories owned by this user will use
                        remote-based names, while others use directory names
            pattern_manager: Pattern management system for ecosystem detection
        """
        self.github_user = github_user
        self.pattern_manager = pattern_manager or PatternManager()

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
            return bool(
                is_github
                and username
                and self.github_user
                and username.lower() == self.github_user.lower()
            )

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

    def create_daemon_identifier(self, path: str = ".", suffix: str | None = None) -> DaemonIdentifier:
        """Create a DaemonIdentifier for the project at the specified path.

        Args:
            path: Path to analyze (defaults to current directory)
            suffix: Optional suffix for the identifier

        Returns:
            DaemonIdentifier instance for the project
        """
        project_name = self.get_project_name(path)
        project_path = os.path.abspath(path)

        return DaemonIdentifier(
            project_name=project_name,
            project_path=project_path,
            suffix=suffix
        )

    def detect_ecosystems(self, path: str = ".") -> list[str]:
        """
        Detect project ecosystems using PatternManager.

        Args:
            path: Path to analyze (defaults to current directory)

        Returns:
            List of detected ecosystem names
        """
        try:
            return self.pattern_manager.detect_ecosystem(path)
        except Exception as e:
            logger.warning("Failed to detect ecosystems for %s: %s", path, e)
            return []


def calculate_tenant_id(project_root: Path) -> str:
    """
    Calculate a unique tenant ID for a project root directory.

    This function implements the tenant ID calculation algorithm:
    1. Try to get git remote URL (prefer origin, fallback to upstream)
    2. If remote exists: Sanitize URL to create tenant ID
       - Remove protocol (https://, git@, ssh://)
       - Replace separators (/, ., :, @) with underscores
       - Example: github.com/user/repo → github_com_user_repo
    3. If no remote: Use SHA256 hash of absolute path
       - Hash first 16 chars: abc123def456789a
       - Add prefix: path_abc123def456789a

    Args:
        project_root: Path object pointing to the project root directory

    Returns:
        Unique tenant ID string

    Examples:
        >>> calculate_tenant_id(Path("/path/to/repo"))  # with git remote
        'github_com_user_repo'

        >>> calculate_tenant_id(Path("/path/to/local"))  # without git remote
        'path_abc123def456789a'
    """
    try:
        # Convert to Path object if string
        if isinstance(project_root, str):
            project_root = Path(project_root)
        # Convert to absolute path string
        abs_path = str(project_root.resolve())

        # Try to find git repository and get remote URL
        try:
            repo = git.Repo(abs_path, search_parent_directories=True)

            # Try origin first, then upstream, then any remote
            remote_url = None
            for remote_name in ["origin", "upstream"]:
                if hasattr(repo.remotes, remote_name):
                    remote = getattr(repo.remotes, remote_name)
                    remote_url = str(remote.url)
                    break

            # Fall back to first available remote if origin/upstream not found
            if not remote_url and repo.remotes:
                remote_url = str(repo.remotes[0].url)

            # If we have a remote URL, sanitize it
            if remote_url:
                tenant_id = _sanitize_remote_url(remote_url)
                logger.debug(
                    "Generated tenant ID from git remote",
                    project_root=abs_path,
                    remote_url=remote_url,
                    tenant_id=tenant_id
                )
                return tenant_id

        except (InvalidGitRepositoryError, GitError):
            # Not a git repository or no remotes, fall through to path hash
            pass

        # No git remote available, use path hash
        tenant_id = _generate_path_hash_tenant_id(abs_path)
        logger.debug(
            "Generated tenant ID from path hash",
            project_root=abs_path,
            tenant_id=tenant_id
        )
        return tenant_id

    except Exception as e:
        # On any error, fall back to path hash
        logger.warning(
            "Error calculating tenant ID for %s: %s. Falling back to path hash.",
            project_root, e
        )
        abs_path = str(project_root.resolve())
        return _generate_path_hash_tenant_id(abs_path)


def _sanitize_remote_url(remote_url: str) -> str:
    """
    Sanitize a git remote URL to create a tenant ID.

    Removes protocols and replaces separators with underscores.

    Args:
        remote_url: Git remote URL (HTTPS or SSH format)

    Returns:
        Sanitized tenant ID string

    Examples:
        >>> _sanitize_remote_url("https://github.com/user/repo.git")
        'github_com_user_repo'

        >>> _sanitize_remote_url("git@github.com:user/repo.git")
        'github_com_user_repo'

        >>> _sanitize_remote_url("ssh://git@gitlab.com:2222/user/project.git")
        'gitlab_com_2222_user_project'
    """
    # Remove common protocols
    url = remote_url
    for protocol in ["https://", "http://", "ssh://", "git://"]:
        if url.startswith(protocol):
            url = url[len(protocol):]
            break

    # Remove git@ prefix (SSH format)
    if url.startswith("git@"):
        url = url[4:]  # Remove "git@"

    # Remove .git suffix if present
    if url.endswith(".git"):
        url = url[:-4]

    # Replace all separators with underscores and normalize case
    url = url.lower()
    # Handle separators: / . : @ -
    url = url.replace("/", "_")
    url = url.replace(".", "_")
    url = url.replace(":", "_")
    url = url.replace("@", "_")
    url = url.replace("-", "_")

    # Remove any duplicate underscores
    while "__" in url:
        url = url.replace("__", "_")

    # Remove leading/trailing underscores
    url = url.strip("_")

    return url


def _generate_path_hash_tenant_id(abs_path: str) -> str:
    """
    Generate a tenant ID from an absolute path using SHA256 hash.

    Creates a hash-based tenant ID with the format: path_{16_char_hash}

    Args:
        abs_path: Absolute path to the project directory

    Returns:
        Tenant ID with path_ prefix and 16-character hash

    Examples:
        >>> _generate_path_hash_tenant_id("/home/user/project")
        'path_a1b2c3d4e5f67890'
    """
    # Normalize path for consistent hashing
    normalized_path = os.path.normpath(abs_path)

    # Generate SHA256 hash
    hash_obj = hashlib.sha256(normalized_path.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()

    # Take first 16 characters of the hash
    hash_prefix = hash_hex[:16]

    # Return with path_ prefix
    return f"path_{hash_prefix}"
