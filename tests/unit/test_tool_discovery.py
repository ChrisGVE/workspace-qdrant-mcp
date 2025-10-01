"""Unit tests for tool discovery module.

Tests the ToolDiscovery class for finding, validating, and retrieving
version information from executables. Includes tests for:
- Finding executables in PATH
- Custom path support
- Executable validation
- Version retrieval
- Pattern-based scanning
- Cross-platform compatibility
"""

import os
import platform
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.python.common.core.tool_discovery import ToolDiscovery


class TestToolDiscoveryInit:
    """Test ToolDiscovery initialization."""

    def test_init_default(self):
        """Test initialization with default settings."""
        discovery = ToolDiscovery()
        assert discovery.timeout == 5
        assert discovery.custom_paths == []

    def test_init_with_timeout(self):
        """Test initialization with custom timeout."""
        discovery = ToolDiscovery(timeout=10)
        assert discovery.timeout == 10

    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {
            "custom_paths": ["/custom/path1", "/custom/path2"],
            "timeout": 15
        }
        discovery = ToolDiscovery(config=config)
        assert discovery.timeout == 15
        assert discovery.custom_paths == ["/custom/path1", "/custom/path2"]

    def test_init_with_partial_config(self):
        """Test initialization with partial configuration."""
        config = {"custom_paths": ["/custom/path"]}
        discovery = ToolDiscovery(config=config, timeout=8)
        assert discovery.timeout == 8
        assert discovery.custom_paths == ["/custom/path"]


class TestFindExecutable:
    """Test finding executables in PATH."""

    def test_find_python(self):
        """Test finding python executable."""
        discovery = ToolDiscovery()
        python_path = discovery.find_executable("python")

        # Python should be available in test environment
        assert python_path is not None
        assert Path(python_path).exists()
        assert discovery.validate_executable(python_path)

    def test_find_python3(self):
        """Test finding python3 executable."""
        discovery = ToolDiscovery()
        python3_path = discovery.find_executable("python3")

        # python3 might not be available on Windows
        if python3_path:
            assert Path(python3_path).exists()
            assert discovery.validate_executable(python3_path)

    def test_find_nonexistent(self):
        """Test finding non-existent executable."""
        discovery = ToolDiscovery()
        result = discovery.find_executable("nonexistent-tool-12345")
        assert result is None

    def test_find_in_custom_path(self, tmp_path):
        """Test finding executable in custom path."""
        # Create a mock executable in temporary directory
        mock_exe = tmp_path / "mock-tool"
        mock_exe.write_text("#!/bin/sh\necho 'mock'")
        mock_exe.chmod(0o755)

        config = {"custom_paths": [str(tmp_path)]}
        discovery = ToolDiscovery(config=config)

        result = discovery.find_executable("mock-tool")

        # Should find in custom path
        if platform.system() != "Windows":
            assert result is not None
            assert Path(result).exists()

    def test_custom_path_priority(self, tmp_path):
        """Test that custom paths are checked before system PATH."""
        # Create a mock executable with same name as system tool
        mock_python = tmp_path / "python"
        mock_python.write_text("#!/bin/sh\necho 'custom'")
        mock_python.chmod(0o755)

        config = {"custom_paths": [str(tmp_path)]}
        discovery = ToolDiscovery(config=config)

        result = discovery.find_executable("python")

        # On Unix, should find custom path first
        if platform.system() != "Windows":
            assert result is not None
            # Should be in our tmp_path
            assert str(tmp_path) in result


class TestValidateExecutable:
    """Test executable validation."""

    def test_validate_python(self):
        """Test validating python executable."""
        discovery = ToolDiscovery()
        python_path = discovery.find_executable("python")
        assert python_path is not None
        assert discovery.validate_executable(python_path)

    def test_validate_nonexistent(self):
        """Test validating non-existent file."""
        discovery = ToolDiscovery()
        assert not discovery.validate_executable("/nonexistent/path/tool")

    def test_validate_directory(self, tmp_path):
        """Test validating a directory returns False."""
        discovery = ToolDiscovery()
        assert not discovery.validate_executable(str(tmp_path))

    def test_validate_non_executable_file(self, tmp_path):
        """Test validating non-executable file."""
        non_exec = tmp_path / "not-executable.txt"
        non_exec.write_text("not executable")

        discovery = ToolDiscovery()
        result = discovery.validate_executable(str(non_exec))

        # On Unix, should be False (no exec bit)
        # On Windows, should be False (wrong extension)
        assert not result

    def test_validate_executable_file(self, tmp_path):
        """Test validating executable file."""
        if platform.system() == "Windows":
            # On Windows, create .exe file
            exe_file = tmp_path / "tool.exe"
            exe_file.write_text("mock exe")
            expected = True
        else:
            # On Unix, create file with exec bit
            exe_file = tmp_path / "tool"
            exe_file.write_text("#!/bin/sh\necho test")
            exe_file.chmod(0o755)
            expected = True

        discovery = ToolDiscovery()
        assert discovery.validate_executable(str(exe_file)) == expected


class TestGetVersion:
    """Test version retrieval."""

    def test_get_python_version(self):
        """Test getting python version."""
        discovery = ToolDiscovery()
        version = discovery.get_version("python")

        assert version is not None
        # Should contain version number
        assert any(c.isdigit() for c in version)

    def test_get_python_version_with_path(self):
        """Test getting version with full path."""
        discovery = ToolDiscovery()
        python_path = discovery.find_executable("python")
        assert python_path is not None

        version = discovery.get_version(python_path)
        assert version is not None
        assert any(c.isdigit() for c in version)

    def test_get_version_custom_flag(self):
        """Test getting version with custom flag."""
        discovery = ToolDiscovery()
        # Python also supports -V
        version = discovery.get_version("python", "-V")

        # Should work with either flag
        if version:
            assert any(c.isdigit() for c in version)

    def test_get_version_nonexistent(self):
        """Test getting version of non-existent tool."""
        discovery = ToolDiscovery()
        version = discovery.get_version("nonexistent-tool-12345")
        assert version is None

    def test_get_version_timeout(self, tmp_path):
        """Test version retrieval with timeout."""
        if platform.system() == "Windows":
            pytest.skip("Timeout test not suitable for Windows")

        # Create a script that sleeps forever
        sleep_script = tmp_path / "sleeper"
        sleep_script.write_text("#!/bin/sh\nsleep 60")
        sleep_script.chmod(0o755)

        discovery = ToolDiscovery(timeout=1)
        version = discovery.get_version(str(sleep_script))

        # Should timeout and return None
        assert version is None


class TestScanPathForExecutables:
    """Test scanning PATH for executables."""

    def test_scan_for_python(self):
        """Test scanning for python executables."""
        discovery = ToolDiscovery()
        results = discovery.scan_path_for_executables("python*")

        # Should find at least one python executable
        assert len(results) > 0
        # All results should be valid executables
        for result in results:
            assert discovery.validate_executable(result)

    def test_scan_for_nonexistent_pattern(self):
        """Test scanning for non-existent pattern."""
        discovery = ToolDiscovery()
        results = discovery.scan_path_for_executables("nonexistent-pattern-*")
        assert len(results) == 0

    def test_scan_in_custom_paths(self, tmp_path):
        """Test scanning in custom paths."""
        # Create multiple mock executables
        mock1 = tmp_path / "test-tool-1"
        mock2 = tmp_path / "test-tool-2"
        mock3 = tmp_path / "other-tool"

        if platform.system() == "Windows":
            mock1 = tmp_path / "test-tool-1.exe"
            mock2 = tmp_path / "test-tool-2.exe"
            mock3 = tmp_path / "other-tool.exe"

        for mock in [mock1, mock2, mock3]:
            mock.write_text("mock")
            if platform.system() != "Windows":
                mock.chmod(0o755)

        config = {"custom_paths": [str(tmp_path)]}
        discovery = ToolDiscovery(config=config)

        results = discovery.scan_path_for_executables("test-tool-*")

        # Should find both test-tool-1 and test-tool-2
        assert len(results) == 2
        assert any("test-tool-1" in r for r in results)
        assert any("test-tool-2" in r for r in results)
        assert not any("other-tool" in r for r in results)

    def test_scan_deduplication(self, tmp_path):
        """Test that scanning deduplicates results."""
        # Create mock executable
        mock = tmp_path / "duplicate-tool"
        mock.write_text("#!/bin/sh\necho test")
        if platform.system() != "Windows":
            mock.chmod(0o755)

        # Add same path twice to custom paths
        config = {"custom_paths": [str(tmp_path), str(tmp_path)]}
        discovery = ToolDiscovery(config=config)

        results = discovery.scan_path_for_executables("duplicate-*")

        # Should only find once despite duplicate paths
        if platform.system() != "Windows":
            assert len(results) == 1


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test complete workflow: find, validate, get version."""
        discovery = ToolDiscovery()

        # Find python
        python_path = discovery.find_executable("python")
        assert python_path is not None

        # Validate it
        assert discovery.validate_executable(python_path)

        # Get version
        version = discovery.get_version(python_path)
        assert version is not None
        assert "python" in version.lower() or any(c.isdigit() for c in version)

    def test_with_git(self):
        """Test with git executable if available."""
        discovery = ToolDiscovery()

        git_path = discovery.find_executable("git")
        if git_path:
            # Validate
            assert discovery.validate_executable(git_path)

            # Get version
            version = discovery.get_version(git_path)
            assert version is not None
            assert "git" in version.lower()
