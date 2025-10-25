"""
Unit tests for Tree-sitter Grammar Installer.

Tests grammar installation, uninstallation, version management,
and error handling.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.python.common.core.grammar_installer import (
    GrammarInstaller,
    InstallationResult,
)


class TestGrammarInstaller:
    """Test suite for GrammarInstaller class."""

    @pytest.fixture
    def temp_install_dir(self):
        """Create temporary installation directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def installer(self, temp_install_dir):
        """Create GrammarInstaller with temporary directory."""
        return GrammarInstaller(installation_dir=temp_install_dir)

    def test_installer_initialization_default_dir(self):
        """Test installer initializes with default directory."""
        installer = GrammarInstaller()
        expected_dir = Path.home() / ".config" / "tree-sitter" / "grammars"
        assert installer.installation_dir == expected_dir

    def test_installer_initialization_custom_dir(self, temp_install_dir):
        """Test installer initializes with custom directory."""
        installer = GrammarInstaller(installation_dir=temp_install_dir)
        assert installer.installation_dir == temp_install_dir
        assert temp_install_dir.exists()

    def test_extract_grammar_name_https(self, installer):
        """Test grammar name extraction from HTTPS URL."""
        url = "https://github.com/tree-sitter/tree-sitter-python"
        assert installer._extract_grammar_name(url) == "python"

    def test_extract_grammar_name_ssh(self, installer):
        """Test grammar name extraction from SSH URL."""
        url = "git@github.com:tree-sitter/tree-sitter-rust.git"
        assert installer._extract_grammar_name(url) == "rust"

    def test_extract_grammar_name_trailing_slash(self, installer):
        """Test grammar name extraction with trailing slash."""
        url = "https://github.com/tree-sitter/tree-sitter-javascript/"
        assert installer._extract_grammar_name(url) == "javascript"

    def test_extract_grammar_name_no_prefix(self, installer):
        """Test grammar name extraction without tree-sitter prefix."""
        url = "https://github.com/user/mylang"
        assert installer._extract_grammar_name(url) == "mylang"

    def test_verify_grammar_with_grammar_js(self, installer, temp_install_dir):
        """Test grammar verification with grammar.js file."""
        grammar_dir = temp_install_dir / "test-grammar"
        grammar_dir.mkdir()
        (grammar_dir / "grammar.js").touch()

        assert installer._verify_grammar(grammar_dir) is True

    def test_verify_grammar_with_grammar_json(self, installer, temp_install_dir):
        """Test grammar verification with src/grammar.json file."""
        grammar_dir = temp_install_dir / "test-grammar"
        grammar_dir.mkdir()
        src_dir = grammar_dir / "src"
        src_dir.mkdir()
        (src_dir / "grammar.json").touch()

        assert installer._verify_grammar(grammar_dir) is True

    def test_verify_grammar_invalid(self, installer, temp_install_dir):
        """Test grammar verification fails for invalid directory."""
        grammar_dir = temp_install_dir / "test-grammar"
        grammar_dir.mkdir()

        assert installer._verify_grammar(grammar_dir) is False

    def test_is_installed_true(self, installer, temp_install_dir):
        """Test is_installed returns True for installed grammar."""
        grammar_dir = temp_install_dir / "tree-sitter-python"
        grammar_dir.mkdir()

        assert installer.is_installed("python") is True

    def test_is_installed_false(self, installer):
        """Test is_installed returns False for not installed grammar."""
        assert installer.is_installed("python") is False

    def test_get_installation_path_exists(self, installer, temp_install_dir):
        """Test get_installation_path returns path when installed."""
        grammar_dir = temp_install_dir / "tree-sitter-python"
        grammar_dir.mkdir()

        path = installer.get_installation_path("python")
        assert path == grammar_dir

    def test_get_installation_path_not_exists(self, installer):
        """Test get_installation_path returns None when not installed."""
        path = installer.get_installation_path("python")
        assert path is None

    def test_list_installed_empty(self, installer):
        """Test list_installed returns empty list when nothing installed."""
        assert installer.list_installed() == []

    def test_list_installed_multiple(self, installer, temp_install_dir):
        """Test list_installed returns all installed grammars."""
        (temp_install_dir / "tree-sitter-python").mkdir()
        (temp_install_dir / "tree-sitter-rust").mkdir()
        (temp_install_dir / "tree-sitter-javascript").mkdir()

        installed = installer.list_installed()
        assert sorted(installed) == ["javascript", "python", "rust"]

    def test_list_installed_ignores_non_grammar_dirs(self, installer, temp_install_dir):
        """Test list_installed ignores non-grammar directories."""
        (temp_install_dir / "tree-sitter-python").mkdir()
        (temp_install_dir / "other-dir").mkdir()
        (temp_install_dir / "file.txt").touch()

        installed = installer.list_installed()
        assert installed == ["python"]

    @patch('subprocess.run')
    def test_clone_repository_success(self, mock_run, installer, temp_install_dir):
        """Test successful repository cloning."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        dest = temp_install_dir / "test-repo"
        success, message = installer._clone_repository(
            "https://github.com/test/repo",
            dest
        )

        assert success is True
        assert "successful" in message.lower()
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_clone_repository_with_version(self, mock_run, installer, temp_install_dir):
        """Test repository cloning with version checkout."""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        dest = temp_install_dir / "test-repo"
        success, message = installer._clone_repository(
            "https://github.com/test/repo",
            dest,
            version="v1.0.0"
        )

        assert success is True
        # Should call git clone and git checkout
        assert mock_run.call_count == 2

    @patch('subprocess.run')
    def test_clone_repository_clone_failure(self, mock_run, installer, temp_install_dir):
        """Test clone failure handling."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="fatal: repository not found"
        )

        dest = temp_install_dir / "test-repo"
        success, message = installer._clone_repository(
            "https://github.com/test/repo",
            dest
        )

        assert success is False
        assert "clone failed" in message.lower()

    @patch('subprocess.run')
    def test_clone_repository_git_not_found(self, mock_run, installer, temp_install_dir):
        """Test error when git is not installed."""
        mock_run.side_effect = FileNotFoundError()

        dest = temp_install_dir / "test-repo"
        success, message = installer._clone_repository(
            "https://github.com/test/repo",
            dest
        )

        assert success is False
        assert "git is not installed" in message.lower()

    @patch('subprocess.run')
    def test_clone_repository_timeout(self, mock_run, installer, temp_install_dir):
        """Test clone timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired("git", 300)

        dest = temp_install_dir / "test-repo"
        success, message = installer._clone_repository(
            "https://github.com/test/repo",
            dest
        )

        assert success is False
        assert "timed out" in message.lower()

    @patch('subprocess.run')
    def test_get_installed_version_with_tag(self, mock_run, installer, temp_install_dir):
        """Test getting version from git tag."""
        mock_run.return_value = Mock(returncode=0, stdout="v1.0.0\n", stderr="")

        grammar_dir = temp_install_dir / "test-grammar"
        grammar_dir.mkdir()

        version = installer._get_installed_version(grammar_dir)
        assert version == "v1.0.0"

    @patch('subprocess.run')
    def test_get_installed_version_with_commit(self, mock_run, installer, temp_install_dir):
        """Test getting version from commit hash when no tag."""
        # First call (git describe) fails, second call (git rev-parse) succeeds
        mock_run.side_effect = [
            Mock(returncode=1, stdout="", stderr=""),
            Mock(returncode=0, stdout="abc1234\n", stderr="")
        ]

        grammar_dir = temp_install_dir / "test-grammar"
        grammar_dir.mkdir()

        version = installer._get_installed_version(grammar_dir)
        assert version == "abc1234"

    @patch('subprocess.run')
    def test_get_installed_version_no_git(self, mock_run, installer, temp_install_dir):
        """Test getting version returns None when git fails."""
        mock_run.side_effect = Exception("Git error")

        grammar_dir = temp_install_dir / "test-grammar"
        grammar_dir.mkdir()

        version = installer._get_installed_version(grammar_dir)
        assert version is None

    def test_uninstall_success(self, installer, temp_install_dir):
        """Test successful grammar uninstallation."""
        grammar_dir = temp_install_dir / "tree-sitter-python"
        grammar_dir.mkdir()
        (grammar_dir / "grammar.js").touch()

        success, message = installer.uninstall("python")

        assert success is True
        assert "successfully uninstalled" in message.lower()
        assert not grammar_dir.exists()

    def test_uninstall_not_installed(self, installer):
        """Test uninstall fails for non-installed grammar."""
        success, message = installer.uninstall("python")

        assert success is False
        assert "not installed" in message.lower()

    @patch.object(GrammarInstaller, '_clone_repository')
    @patch.object(GrammarInstaller, '_verify_grammar')
    @patch.object(GrammarInstaller, '_get_installed_version')
    def test_install_success(
        self,
        mock_get_version,
        mock_verify,
        mock_clone,
        installer,
        temp_install_dir
    ):
        """Test successful grammar installation."""
        # Mock _clone_repository to create the temp directory with grammar.js
        def create_temp_grammar(url, dest, version=None):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "grammar.js").touch()
            return (True, "Clone successful")

        mock_clone.side_effect = create_temp_grammar
        mock_verify.return_value = True
        mock_get_version.return_value = "v1.0.0"

        result = installer.install("https://github.com/tree-sitter/tree-sitter-python")

        assert result.success is True
        assert result.grammar_name == "python"
        assert result.version == "v1.0.0"
        assert result.installation_path.exists()

    @patch.object(GrammarInstaller, '_clone_repository')
    def test_install_clone_failure(self, mock_clone, installer):
        """Test installation fails when clone fails."""
        mock_clone.return_value = (False, "Clone failed")

        result = installer.install("https://github.com/tree-sitter/tree-sitter-python")

        assert result.success is False
        assert "clone failed" in result.error.lower()

    @patch.object(GrammarInstaller, '_clone_repository')
    @patch.object(GrammarInstaller, '_verify_grammar')
    def test_install_invalid_grammar(self, mock_verify, mock_clone, installer):
        """Test installation fails for invalid grammar."""
        mock_clone.return_value = (True, "Clone successful")
        mock_verify.return_value = False

        result = installer.install("https://github.com/test/not-a-grammar")

        assert result.success is False
        assert "not appear to be a valid" in result.error.lower()

    def test_install_already_installed(self, installer, temp_install_dir):
        """Test installation fails when grammar already installed."""
        grammar_dir = temp_install_dir / "tree-sitter-python"
        grammar_dir.mkdir()

        result = installer.install("https://github.com/tree-sitter/tree-sitter-python")

        assert result.success is False
        assert "already installed" in result.error.lower()

    @patch.object(GrammarInstaller, '_clone_repository')
    @patch.object(GrammarInstaller, '_verify_grammar')
    @patch.object(GrammarInstaller, '_get_installed_version')
    def test_install_force_reinstall(
        self,
        mock_get_version,
        mock_verify,
        mock_clone,
        installer,
        temp_install_dir
    ):
        """Test force reinstallation overwrites existing installation."""
        # Create existing installation
        existing_dir = temp_install_dir / "tree-sitter-python"
        existing_dir.mkdir()
        (existing_dir / "old_file.txt").touch()

        # Mock _clone_repository to create the temp directory with grammar.js
        def create_temp_grammar(url, dest, version=None):
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "grammar.js").touch()
            return (True, "Clone successful")

        mock_clone.side_effect = create_temp_grammar
        mock_verify.return_value = True
        mock_get_version.return_value = "v2.0.0"

        result = installer.install(
            "https://github.com/tree-sitter/tree-sitter-python",
            force=True
        )

        assert result.success is True
        assert result.grammar_name == "python"
        # Old installation should be replaced
        assert not (existing_dir / "old_file.txt").exists()

    def test_install_custom_grammar_name(self, installer):
        """Test installation with custom grammar name."""
        with patch.object(GrammarInstaller, '_clone_repository') as mock_clone, \
             patch.object(GrammarInstaller, '_verify_grammar') as mock_verify, \
             patch.object(GrammarInstaller, '_get_installed_version') as mock_version:

            # Mock _clone_repository to create the temp directory with grammar.js
            def create_temp_grammar(url, dest, version=None):
                dest.mkdir(parents=True, exist_ok=True)
                (dest / "grammar.js").touch()
                return (True, "Clone successful")

            mock_clone.side_effect = create_temp_grammar
            mock_verify.return_value = True
            mock_version.return_value = None

            result = installer.install(
                "https://github.com/user/custom-lang",
                grammar_name="mylang"
            )

            assert result.success is True
            assert result.grammar_name == "mylang"


class TestInstallationResult:
    """Test suite for InstallationResult dataclass."""

    def test_installation_result_success(self):
        """Test creating successful installation result."""
        result = InstallationResult(
            success=True,
            grammar_name="python",
            installation_path=Path("/path/to/python"),
            version="v1.0.0",
            message="Installation successful"
        )

        assert result.success is True
        assert result.grammar_name == "python"
        assert result.version == "v1.0.0"
        assert result.error is None

    def test_installation_result_failure(self):
        """Test creating failed installation result."""
        result = InstallationResult(
            success=False,
            grammar_name="python",
            installation_path=Path("/path/to/python"),
            error="Installation failed"
        )

        assert result.success is False
        assert result.error == "Installation failed"
        assert result.version is None
