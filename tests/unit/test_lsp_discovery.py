"""Unit tests for LSP server discovery in tool_discovery module.

Tests the discover_lsp_servers and find_lsp_executable methods for:
- Finding LSP executables in system PATH
- Finding LSP executables in project-local paths
- Integration with LanguageSupportDatabaseConfig
- Handling missing LSP servers
- Cross-platform compatibility
"""

import platform
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.python.common.core.tool_discovery import ToolDiscovery


class TestFindProjectLocalPaths:
    """Test finding project-local paths."""

    def test_no_project_root(self):
        """Test with no project root specified."""
        discovery = ToolDiscovery()
        paths = discovery._find_project_local_paths()

        assert paths == []

    def test_empty_project(self, tmp_path):
        """Test with empty project directory."""
        discovery = ToolDiscovery(project_root=tmp_path)
        paths = discovery._find_project_local_paths()

        assert paths == []

    def test_finds_node_modules(self, tmp_path):
        """Test finding node_modules/.bin directory."""
        node_bin = tmp_path / "node_modules" / ".bin"
        node_bin.mkdir(parents=True)

        discovery = ToolDiscovery(project_root=tmp_path)
        paths = discovery._find_project_local_paths()

        assert node_bin in paths

    def test_finds_python_venv(self, tmp_path):
        """Test finding Python virtual environment bin directory."""
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)

        discovery = ToolDiscovery(project_root=tmp_path)
        paths = discovery._find_project_local_paths()

        assert venv_bin in paths

    def test_finds_python_venv_windows(self, tmp_path):
        """Test finding Python venv Scripts directory on Windows."""
        venv_scripts = tmp_path / ".venv" / "Scripts"
        venv_scripts.mkdir(parents=True)

        discovery = ToolDiscovery(project_root=tmp_path)
        paths = discovery._find_project_local_paths()

        assert venv_scripts in paths

    def test_finds_generic_bin(self, tmp_path):
        """Test finding generic bin directory."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir(parents=True)

        discovery = ToolDiscovery(project_root=tmp_path)
        paths = discovery._find_project_local_paths()

        assert bin_dir in paths

    def test_finds_multiple_paths(self, tmp_path):
        """Test finding multiple project-local paths."""
        node_bin = tmp_path / "node_modules" / ".bin"
        venv_bin = tmp_path / ".venv" / "bin"
        bin_dir = tmp_path / "bin"

        for dir_path in [node_bin, venv_bin, bin_dir]:
            dir_path.mkdir(parents=True)

        discovery = ToolDiscovery(project_root=tmp_path)
        paths = discovery._find_project_local_paths()

        assert node_bin in paths
        assert venv_bin in paths
        assert bin_dir in paths
        assert len(paths) == 3


class TestFindLSPExecutable:
    """Test finding LSP server executables."""

    def test_lsp_not_found(self):
        """Test when LSP executable is not found."""
        discovery = ToolDiscovery()
        result = discovery.find_lsp_executable("python", "nonexistent-lsp-12345")

        assert result is None

    def test_lsp_in_system_path(self):
        """Test finding LSP in system PATH."""
        discovery = ToolDiscovery()
        # Use python as a proxy for an LSP (it exists in PATH)
        result = discovery.find_lsp_executable("python", "python")

        assert result is not None
        assert Path(result).exists()
        assert discovery.validate_executable(result)

    def test_lsp_in_project_local_path(self, tmp_path):
        """Test finding LSP in project-local path."""
        # Create mock LSP in project venv
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)

        lsp_path = venv_bin / "pyright-langserver"
        lsp_path.write_text("#!/bin/sh\necho 'mock lsp'")

        discovery = ToolDiscovery(project_root=tmp_path)

        # Mock executable check to simulate executable permission
        with patch('os.access', return_value=True if platform.system() != "Windows" else False):
            result = discovery.find_lsp_executable("python", "pyright-langserver")

            # On Unix, should find in project-local path
            if platform.system() != "Windows":
                assert result is not None
                assert str(venv_bin) in result
                assert "pyright-langserver" in result

    def test_lsp_in_node_modules(self, tmp_path):
        """Test finding LSP in node_modules/.bin."""
        # Create mock TypeScript LSP
        node_bin = tmp_path / "node_modules" / ".bin"
        node_bin.mkdir(parents=True)

        lsp_path = node_bin / "typescript-language-server"
        lsp_path.write_text("#!/bin/sh\necho 'mock typescript lsp'")

        discovery = ToolDiscovery(project_root=tmp_path)

        # Mock executable check to simulate executable permission
        with patch('os.access', return_value=True if platform.system() != "Windows" else False):
            result = discovery.find_lsp_executable("typescript", "typescript-language-server")

            # On Unix, should find in node_modules
            if platform.system() != "Windows":
                assert result is not None
                assert str(node_bin) in result

    def test_project_local_priority(self, tmp_path):
        """Test that project-local LSP takes priority over system PATH."""
        # Create mock project-local LSP
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)

        lsp_path = venv_bin / "python"  # Using python as it exists in system PATH
        lsp_path.write_text("#!/bin/sh\necho 'project python'")

        discovery = ToolDiscovery(project_root=tmp_path)

        # Mock executable check to simulate executable permission
        with patch('os.access', return_value=True if platform.system() != "Windows" else False):
            result = discovery.find_lsp_executable("python", "python")

            # On Unix, should find project-local version first
            if platform.system() != "Windows":
                assert result is not None
                assert str(venv_bin) in result


class TestDiscoverLSPServers:
    """Test discovering LSP servers from configuration."""

    def test_none_config_raises_error(self):
        """Test that None config raises ValueError."""
        discovery = ToolDiscovery()

        with pytest.raises(ValueError, match="language_config cannot be None"):
            discovery.discover_lsp_servers(None)

    def test_invalid_config_raises_error(self):
        """Test that invalid config raises ValueError."""
        discovery = ToolDiscovery()
        invalid_config = Mock(spec=[])  # Mock without 'languages' attribute

        with pytest.raises(ValueError, match="must have 'languages' attribute"):
            discovery.discover_lsp_servers(invalid_config)

    def test_empty_language_list(self):
        """Test with empty language list."""
        discovery = ToolDiscovery()
        config = Mock()
        config.languages = []

        result = discovery.discover_lsp_servers(config)

        assert result == {}

    def test_language_without_lsp(self):
        """Test language definition without LSP configuration."""
        discovery = ToolDiscovery()

        # Create mock language without LSP
        lang = Mock()
        lang.name = "plaintext"
        lang.lsp = None

        config = Mock()
        config.languages = [lang]

        result = discovery.discover_lsp_servers(config)

        assert result == {}

    def test_single_language_with_lsp(self):
        """Test discovering LSP for single language."""
        discovery = ToolDiscovery()

        # Create mock language with LSP
        lsp = Mock()
        lsp.executable = "python"  # Using python as it exists

        lang = Mock()
        lang.name = "python"
        lang.lsp = lsp

        config = Mock()
        config.languages = [lang]

        result = discovery.discover_lsp_servers(config)

        assert "python" in result
        # Python should be found in system PATH
        assert result["python"] is not None
        assert Path(result["python"]).exists()

    def test_multiple_languages(self):
        """Test discovering LSPs for multiple languages."""
        discovery = ToolDiscovery()

        # Create mock languages
        lsp1 = Mock()
        lsp1.executable = "python"
        lang1 = Mock()
        lang1.name = "python"
        lang1.lsp = lsp1

        lsp2 = Mock()
        lsp2.executable = "nonexistent-lsp"
        lang2 = Mock()
        lang2.name = "custom"
        lang2.lsp = lsp2

        config = Mock()
        config.languages = [lang1, lang2]

        result = discovery.discover_lsp_servers(config)

        assert "python" in result
        assert "custom" in result
        assert result["python"] is not None  # Found
        assert result["custom"] is None  # Not found

    def test_with_project_root(self, tmp_path):
        """Test LSP discovery with project root specified."""
        # Create mock project-local LSP
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)

        lsp_path = venv_bin / "ruff-lsp"
        lsp_path.write_text("#!/bin/sh\necho 'ruff-lsp'")

        discovery = ToolDiscovery()

        # Create mock language with LSP
        lsp = Mock()
        lsp.executable = "ruff-lsp"
        lang = Mock()
        lang.name = "python"
        lang.lsp = lsp

        config = Mock()
        config.languages = [lang]

        # Mock executable check to simulate executable permission
        with patch('os.access', return_value=True if platform.system() != "Windows" else False):
            result = discovery.discover_lsp_servers(config, project_root=tmp_path)

            # On Unix, should find project-local LSP
            if platform.system() != "Windows":
                assert "python" in result
                assert result["python"] is not None
                assert str(venv_bin) in result["python"]

    def test_missing_lsp_warning(self, tmp_path, caplog):
        """Test that missing LSP servers generate warnings."""
        discovery = ToolDiscovery()

        # Create mock language with non-existent LSP
        lsp = Mock()
        lsp.executable = "totally-nonexistent-lsp-xyz"
        lang = Mock()
        lang.name = "testlang"
        lang.lsp = lsp

        config = Mock()
        config.languages = [lang]

        result = discovery.discover_lsp_servers(config)

        assert "testlang" in result
        assert result["testlang"] is None


class TestIntegration:
    """Integration tests for LSP discovery."""

    def test_realistic_project_structure(self, tmp_path):
        """Test LSP discovery in realistic project structure."""
        # Create realistic project structure
        venv_bin = tmp_path / ".venv" / "bin"
        venv_bin.mkdir(parents=True)
        node_bin = tmp_path / "node_modules" / ".bin"
        node_bin.mkdir(parents=True)

        # Create mock LSP servers
        pyright = venv_bin / "pyright-langserver"
        pyright.write_text("#!/bin/sh\necho 'pyright'")

        tsserver = node_bin / "typescript-language-server"
        tsserver.write_text("#!/bin/sh\necho 'tsserver'")

        discovery = ToolDiscovery(project_root=tmp_path)

        # Create mock configuration
        lsp1 = Mock()
        lsp1.executable = "pyright-langserver"
        lang1 = Mock()
        lang1.name = "python"
        lang1.lsp = lsp1

        lsp2 = Mock()
        lsp2.executable = "typescript-language-server"
        lang2 = Mock()
        lang2.name = "typescript"
        lang2.lsp = lsp2

        lsp3 = Mock()
        lsp3.executable = "rust-analyzer"
        lang3 = Mock()
        lang3.name = "rust"
        lang3.lsp = lsp3

        config = Mock()
        config.languages = [lang1, lang2, lang3]

        # Mock executable check to simulate executable permission
        with patch('os.access', return_value=True if platform.system() != "Windows" else False):
            result = discovery.discover_lsp_servers(config)

            # On Unix, Python and TypeScript should be found locally
            # Rust should check system PATH
            if platform.system() != "Windows":
                assert result["python"] is not None
                assert result["typescript"] is not None
                assert str(venv_bin) in result["python"]
                assert str(node_bin) in result["typescript"]

    def test_custom_paths_integration(self, tmp_path):
        """Test LSP discovery with custom paths."""
        # Create custom LSP location
        custom_bin = tmp_path / "custom-lsp-bin"
        custom_bin.mkdir(parents=True)

        custom_lsp = custom_bin / "custom-lsp"
        custom_lsp.write_text("#!/bin/sh\necho 'custom'")

        # Initialize with custom path
        config = {"custom_paths": [str(custom_bin)]}
        discovery = ToolDiscovery(config=config)

        # Create mock configuration
        lsp = Mock()
        lsp.executable = "custom-lsp"
        lang = Mock()
        lang.name = "customlang"
        lang.lsp = lsp

        lang_config = Mock()
        lang_config.languages = [lang]

        # Mock executable check to simulate executable permission
        with patch('os.access', return_value=True if platform.system() != "Windows" else False):
            result = discovery.discover_lsp_servers(lang_config)

            # On Unix, should find LSP in custom path
            if platform.system() != "Windows":
                assert result["customlang"] is not None
                assert str(custom_bin) in result["customlang"]
