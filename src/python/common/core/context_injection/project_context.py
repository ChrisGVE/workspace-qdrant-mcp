"""
Project context detection and rule application for LLM context injection.

This module provides project boundary detection and context-aware rule selection
using the existing project identification infrastructure.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from loguru import logger

from ..memory import MemoryRule
from .authority_filter import AuthorityFilter
from .rule_retrieval import RuleRetrieval, RuleFilter


@dataclass
class ProjectContext:
    """
    Project context information for rule application.

    Attributes:
        project_id: Unique project identifier
        project_root: Absolute path to project root
        current_path: Current working directory or file path
        scope: Current scope contexts (e.g., ["python", "testing"])
        is_submodule: Whether this is a git submodule
        parent_project_id: Parent project ID if this is a submodule
    """

    project_id: str
    project_root: Path
    current_path: Path
    scope: List[str]
    is_submodule: bool = False
    parent_project_id: Optional[str] = None


class ProjectContextDetector:
    """
    Detects project context and boundaries for rule application.

    This class uses the existing project identification infrastructure to
    detect project boundaries, determine project IDs, and provide context
    for context-aware rule selection.
    """

    def __init__(self):
        """Initialize the project context detector."""
        self._git_cache = {}  # Cache for git repository detection

    def detect_project_context(
        self, path: Optional[Path] = None
    ) -> Optional[ProjectContext]:
        """
        Detect project context for a given path.

        Args:
            path: Path to detect context for (defaults to current directory)

        Returns:
            ProjectContext if project detected, None otherwise
        """
        if path is None:
            path = Path.cwd()
        else:
            path = Path(path).resolve()

        # Find project root
        project_root = self._find_project_root(path)
        if not project_root:
            logger.debug(f"No project detected for path: {path}")
            return None

        # Determine project ID
        project_id = self._get_project_id(project_root)

        # Detect scope from path
        scope = self._detect_scope(path, project_root)

        # Check for submodule
        is_submodule, parent_project_id = self._check_submodule(project_root)

        context = ProjectContext(
            project_id=project_id,
            project_root=project_root,
            current_path=path,
            scope=scope,
            is_submodule=is_submodule,
            parent_project_id=parent_project_id,
        )

        logger.debug(
            f"Detected project context: {project_id} at {project_root}, "
            f"scope: {scope}, submodule: {is_submodule}"
        )

        return context

    def _find_project_root(self, path: Path) -> Optional[Path]:
        """
        Find project root by walking up directory tree.

        Looks for project signatures:
        - .git directory
        - pyproject.toml
        - package.json
        - Cargo.toml
        - go.mod

        Args:
            path: Starting path to search from

        Returns:
            Path to project root or None if not found
        """
        current = path.resolve()

        # Walk up the directory tree
        while current != current.parent:
            # Check for project signature files
            if self._is_project_root(current):
                return current
            current = current.parent

        # Check root directory as last resort
        if self._is_project_root(current):
            return current

        return None

    def _is_project_root(self, path: Path) -> bool:
        """
        Check if path is a project root.

        Args:
            path: Path to check

        Returns:
            True if path is a project root
        """
        signatures = [
            ".git",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            ".taskmaster",
        ]

        for signature in signatures:
            if (path / signature).exists():
                return True

        return False

    def _get_project_id(self, project_root: Path) -> str:
        """
        Get project ID for a project root.

        Uses the same logic as the main codebase:
        - If git remote exists, use normalized remote URL
        - Otherwise, use absolute path

        Args:
            project_root: Project root path

        Returns:
            Project ID string
        """
        # Check for git remote
        git_dir = project_root / ".git"
        if git_dir.exists():
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=project_root,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    remote_url = result.stdout.strip()
                    # Normalize URL (remove .git, https://, etc.)
                    normalized = self._normalize_git_url(remote_url)
                    return normalized
            except Exception as e:
                logger.debug(f"Failed to get git remote: {e}")

        # Fall back to absolute path
        return str(project_root)

    def _normalize_git_url(self, url: str) -> str:
        """
        Normalize git remote URL to consistent format.

        Args:
            url: Git remote URL

        Returns:
            Normalized URL
        """
        # Remove .git suffix
        if url.endswith(".git"):
            url = url[:-4]

        # Remove protocol prefixes
        for prefix in ["https://", "http://", "git@", "ssh://"]:
            if url.startswith(prefix):
                url = url[len(prefix) :]

        # Replace : with / for SSH URLs (git@github.com:user/repo -> git@github.com/user/repo)
        if ":" in url and not url.startswith("http"):
            url = url.replace(":", "/", 1)

        return url

    def _detect_scope(self, path: Path, project_root: Path) -> List[str]:
        """
        Detect scope contexts from path.

        Infers scope from:
        - Directory structure (tests/, src/, docs/)
        - File extensions (.py, .js, .rs)
        - Current directory name

        Args:
            path: Current path
            project_root: Project root path

        Returns:
            List of scope contexts
        """
        scope = []

        # Get relative path from project root
        try:
            rel_path = path.relative_to(project_root)
        except ValueError:
            # Path is not relative to project root
            return scope

        # Check directory structure
        parts = rel_path.parts
        if parts:
            # First directory indicates scope
            first_dir = parts[0]
            if first_dir in ["tests", "test"]:
                scope.append("testing")
            elif first_dir in ["src", "lib"]:
                scope.append("source")
            elif first_dir in ["docs", "documentation"]:
                scope.append("documentation")

        # Detect language from file extension
        if path.is_file():
            ext = path.suffix.lower()
            ext_to_lang = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".rs": "rust",
                ".go": "go",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".rb": "ruby",
                ".php": "php",
                ".md": "markdown",
            }
            if ext in ext_to_lang:
                scope.append(ext_to_lang[ext])

        return scope

    def _check_submodule(self, project_root: Path) -> tuple[bool, Optional[str]]:
        """
        Check if project is a git submodule.

        Args:
            project_root: Project root path

        Returns:
            Tuple of (is_submodule, parent_project_id)
        """
        git_dir = project_root / ".git"

        # If .git is a file (not directory), it's likely a submodule
        if git_dir.is_file():
            try:
                # Read .git file to find actual git directory
                with open(git_dir) as f:
                    git_content = f.read().strip()

                # Parse gitdir path
                if git_content.startswith("gitdir:"):
                    gitdir_path = git_content[7:].strip()
                    # Submodule git dirs are usually in parent's .git/modules/
                    if ".git/modules/" in gitdir_path:
                        # Find parent project
                        parent_root = self._find_project_root(project_root.parent)
                        if parent_root and parent_root != project_root:
                            parent_id = self._get_project_id(parent_root)
                            return True, parent_id
            except Exception as e:
                logger.debug(f"Failed to check submodule status: {e}")

        return False, None


class ProjectRuleApplicator:
    """
    Applies project-specific rule filtering and overrides.

    Integrates RuleRetrieval, AuthorityFilter, and ProjectContextDetector
    to provide context-aware rule selection with project-specific overrides.
    """

    def __init__(
        self,
        rule_retrieval: RuleRetrieval,
        authority_filter: Optional[AuthorityFilter] = None,
        context_detector: Optional[ProjectContextDetector] = None,
    ):
        """
        Initialize the project rule applicator.

        Args:
            rule_retrieval: RuleRetrieval instance for fetching rules
            authority_filter: Optional AuthorityFilter (creates default if None)
            context_detector: Optional ProjectContextDetector (creates default if None)
        """
        self.rule_retrieval = rule_retrieval
        self.authority_filter = authority_filter or AuthorityFilter()
        self.context_detector = context_detector or ProjectContextDetector()

    async def get_applicable_rules(
        self, path: Optional[Path] = None, include_parent_project: bool = True
    ) -> List[MemoryRule]:
        """
        Get rules applicable to current project context.

        Args:
            path: Path to detect context for (defaults to current directory)
            include_parent_project: Include parent project rules for submodules

        Returns:
            List of applicable rules with project-specific overrides applied
        """
        # Detect project context
        context = self.context_detector.detect_project_context(path)
        if not context:
            # No project context - return global rules only
            logger.debug("No project context detected, returning global rules only")
            filter = RuleFilter(scope=[])
            result = await self.rule_retrieval.get_rules(filter)
            return result.rules

        # Build filter for project-specific rules
        filter = RuleFilter(
            project_id=context.project_id, scope=context.scope, limit=100
        )

        # Get rules matching project context
        result = await self.rule_retrieval.get_rules(filter)
        rules = result.rules

        # Include parent project rules if this is a submodule
        if (
            include_parent_project
            and context.is_submodule
            and context.parent_project_id
        ):
            parent_filter = RuleFilter(
                project_id=context.parent_project_id, scope=context.scope, limit=100
            )
            parent_result = await self.rule_retrieval.get_rules(parent_filter)
            rules.extend(parent_result.rules)

        # Apply authority filtering with project context
        effective_rules = self.authority_filter.get_effective_rules(
            rules, project_id=context.project_id, scope=context.scope
        )

        logger.info(
            f"Retrieved {len(effective_rules)} applicable rules for project "
            f"{context.project_id} (from {len(rules)} total)"
        )

        return effective_rules

    async def get_project_overrides(
        self, global_rules: List[MemoryRule], project_id: str
    ) -> List[MemoryRule]:
        """
        Get project-specific overrides for global rules.

        Args:
            global_rules: List of global rules
            project_id: Project ID to check for overrides

        Returns:
            List of rules that override global rules for this project
        """
        # Get all rules for this project
        filter = RuleFilter(project_id=project_id, limit=100)
        result = await self.rule_retrieval.get_rules(filter)
        project_rules = result.rules

        # Find rules that override global rules (same category + scope)
        overrides = []
        global_contexts = set()

        # Build set of global rule contexts
        for rule in global_rules:
            context_key = self.authority_filter._get_context_key(rule, project_id)
            global_contexts.add(context_key)

        # Find project rules that match global contexts
        for rule in project_rules:
            context_key = self.authority_filter._get_context_key(rule, project_id)
            if context_key in global_contexts:
                overrides.append(rule)

        logger.debug(
            f"Found {len(overrides)} project-specific overrides for {len(global_rules)} global rules"
        )

        return overrides
