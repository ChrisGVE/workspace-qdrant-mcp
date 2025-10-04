"""
Grammar Installation System for Tree-sitter.

This module provides functionality to install tree-sitter grammars from Git repositories
with support for version pinning, dependency management, and automatic compilation.

Key features:
- Clone grammars from Git repositories
- Support version pinning (tags, branches, commits)
- Manage installation location (~/.config/tree-sitter/grammars/)
- Integrate with grammar discovery system
- Handle installation conflicts and updates
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class InstallationResult:
    """Result of a grammar installation operation."""

    success: bool
    """Whether installation succeeded"""

    grammar_name: str
    """Name of the grammar (e.g., 'python')"""

    installation_path: Path
    """Path where grammar was installed"""

    version: Optional[str] = None
    """Installed version (tag, branch, or commit)"""

    message: str = ""
    """Human-readable status message"""

    error: Optional[str] = None
    """Error message if installation failed"""


class GrammarInstaller:
    """
    Manages installation of tree-sitter grammars from Git repositories.

    Handles cloning, version management, and integration with the grammar
    discovery system.
    """

    def __init__(self, installation_dir: Optional[Path] = None):
        """
        Initialize grammar installer.

        Args:
            installation_dir: Directory for grammar installations.
                            Defaults to ~/.config/tree-sitter/grammars/
        """
        if installation_dir is None:
            config_dir = Path.home() / ".config" / "tree-sitter"
            self.installation_dir = config_dir / "grammars"
        else:
            self.installation_dir = Path(installation_dir)

        # Ensure installation directory exists
        self.installation_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Grammar installation directory: {self.installation_dir}")

    def install(
        self,
        grammar_url: str,
        grammar_name: Optional[str] = None,
        version: Optional[str] = None,
        force: bool = False
    ) -> InstallationResult:
        """
        Install a tree-sitter grammar from a Git repository.

        Args:
            grammar_url: Git repository URL (e.g., https://github.com/tree-sitter/tree-sitter-python)
            grammar_name: Override grammar name (auto-detected from URL if None)
            version: Version to install (tag, branch, or commit hash). Uses default branch if None.
            force: If True, overwrite existing installation

        Returns:
            InstallationResult with status and details

        Example:
            >>> installer = GrammarInstaller()
            >>> result = installer.install(
            ...     "https://github.com/tree-sitter/tree-sitter-python",
            ...     version="v0.20.0"
            ... )
            >>> if result.success:
            ...     print(f"Installed {result.grammar_name} at {result.installation_path}")
        """
        # Auto-detect grammar name from URL if not provided
        if grammar_name is None:
            grammar_name = self._extract_grammar_name(grammar_url)

        logger.info(f"Installing grammar '{grammar_name}' from {grammar_url}")

        # Check if already installed
        install_path = self.installation_dir / f"tree-sitter-{grammar_name}"
        if install_path.exists() and not force:
            return InstallationResult(
                success=False,
                grammar_name=grammar_name,
                installation_path=install_path,
                error=f"Grammar '{grammar_name}' already installed at {install_path}. Use force=True to reinstall."
            )

        # Clone to temporary directory first
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / f"tree-sitter-{grammar_name}"

                # Clone repository
                clone_result = self._clone_repository(grammar_url, temp_path, version)
                if not clone_result[0]:
                    return InstallationResult(
                        success=False,
                        grammar_name=grammar_name,
                        installation_path=install_path,
                        error=clone_result[1]
                    )

                # Verify it's a valid tree-sitter grammar
                if not self._verify_grammar(temp_path):
                    return InstallationResult(
                        success=False,
                        grammar_name=grammar_name,
                        installation_path=install_path,
                        error=f"Repository does not appear to be a valid tree-sitter grammar (missing grammar.js or src/grammar.json)"
                    )

                # Remove existing installation if force=True
                if install_path.exists():
                    logger.info(f"Removing existing installation at {install_path}")
                    shutil.rmtree(install_path)

                # Move from temp to final location
                shutil.move(str(temp_path), str(install_path))

                # Get installed version
                installed_version = self._get_installed_version(install_path)

                return InstallationResult(
                    success=True,
                    grammar_name=grammar_name,
                    installation_path=install_path,
                    version=installed_version or version,
                    message=f"Successfully installed grammar '{grammar_name}' to {install_path}"
                )

        except Exception as e:
            logger.error(f"Failed to install grammar '{grammar_name}': {e}")
            return InstallationResult(
                success=False,
                grammar_name=grammar_name,
                installation_path=install_path,
                error=f"Installation failed: {str(e)}"
            )

    def uninstall(self, grammar_name: str) -> Tuple[bool, str]:
        """
        Uninstall a tree-sitter grammar.

        Args:
            grammar_name: Name of grammar to uninstall

        Returns:
            Tuple of (success, message)
        """
        install_path = self.installation_dir / f"tree-sitter-{grammar_name}"

        if not install_path.exists():
            return False, f"Grammar '{grammar_name}' is not installed"

        try:
            shutil.rmtree(install_path)
            logger.info(f"Uninstalled grammar '{grammar_name}' from {install_path}")
            return True, f"Successfully uninstalled grammar '{grammar_name}'"
        except Exception as e:
            logger.error(f"Failed to uninstall grammar '{grammar_name}': {e}")
            return False, f"Uninstallation failed: {str(e)}"

    def list_installed(self) -> List[str]:
        """
        List all installed grammars.

        Returns:
            List of installed grammar names
        """
        if not self.installation_dir.exists():
            return []

        installed = []
        for item in self.installation_dir.iterdir():
            if item.is_dir() and item.name.startswith("tree-sitter-"):
                # Extract grammar name (remove "tree-sitter-" prefix)
                grammar_name = item.name[len("tree-sitter-"):]
                installed.append(grammar_name)

        return sorted(installed)

    def get_installation_path(self, grammar_name: str) -> Optional[Path]:
        """
        Get installation path for a grammar.

        Args:
            grammar_name: Name of grammar

        Returns:
            Installation path if installed, None otherwise
        """
        install_path = self.installation_dir / f"tree-sitter-{grammar_name}"
        return install_path if install_path.exists() else None

    def is_installed(self, grammar_name: str) -> bool:
        """
        Check if a grammar is installed.

        Args:
            grammar_name: Name of grammar

        Returns:
            True if installed, False otherwise
        """
        return self.get_installation_path(grammar_name) is not None

    def _extract_grammar_name(self, url: str) -> str:
        """
        Extract grammar name from repository URL.

        Args:
            url: Git repository URL

        Returns:
            Grammar name (e.g., "python" from "tree-sitter-python")

        Examples:
            >>> installer = GrammarInstaller()
            >>> installer._extract_grammar_name("https://github.com/tree-sitter/tree-sitter-python")
            'python'
            >>> installer._extract_grammar_name("git@github.com:tree-sitter/tree-sitter-rust.git")
            'rust'
        """
        # Extract last part of URL (handle both / and : separators for SSH URLs)
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]

        # Split by either / or : to handle both HTTPS and SSH URLs
        if ":" in url and "/" in url:
            # Could be SSH URL like git@github.com:user/repo
            url = url.replace(":", "/")

        repo_name = url.split("/")[-1]

        # Remove "tree-sitter-" prefix if present
        if repo_name.startswith("tree-sitter-"):
            return repo_name[len("tree-sitter-"):]

        return repo_name

    def _clone_repository(
        self,
        url: str,
        destination: Path,
        version: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Clone a Git repository.

        Args:
            url: Repository URL
            destination: Where to clone
            version: Version to checkout (tag, branch, or commit)

        Returns:
            Tuple of (success, message)
        """
        try:
            # Clone repository
            logger.info(f"Cloning {url} to {destination}")

            # Use --depth 1 for faster cloning if no specific version
            clone_cmd = ["git", "clone"]
            if version is None:
                clone_cmd.extend(["--depth", "1"])
            clone_cmd.extend([url, str(destination)])

            result = subprocess.run(
                clone_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                return False, f"Git clone failed: {result.stderr}"

            # Checkout specific version if requested
            if version is not None:
                logger.info(f"Checking out version: {version}")
                checkout_result = subprocess.run(
                    ["git", "checkout", version],
                    cwd=destination,
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if checkout_result.returncode != 0:
                    return False, f"Git checkout failed: {checkout_result.stderr}"

            return True, "Clone successful"

        except subprocess.TimeoutExpired:
            return False, "Git clone timed out"
        except FileNotFoundError:
            return False, "Git is not installed or not in PATH"
        except Exception as e:
            return False, f"Clone failed: {str(e)}"

    def _verify_grammar(self, path: Path) -> bool:
        """
        Verify that a directory contains a valid tree-sitter grammar.

        Args:
            path: Path to check

        Returns:
            True if valid grammar, False otherwise
        """
        # Check for grammar.js (source grammar)
        grammar_js = path / "grammar.js"
        if grammar_js.exists():
            return True

        # Check for src/grammar.json (generated grammar)
        grammar_json = path / "src" / "grammar.json"
        if grammar_json.exists():
            return True

        return False

    def _get_installed_version(self, path: Path) -> Optional[str]:
        """
        Get version of installed grammar from Git.

        Args:
            path: Grammar installation path

        Returns:
            Version string (tag or commit hash) if available
        """
        try:
            # Try to get current tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return result.stdout.strip()

            # Fall back to commit hash
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return result.stdout.strip()

        except Exception:
            pass

        return None


# Export main class and result type
__all__ = ["GrammarInstaller", "InstallationResult"]
