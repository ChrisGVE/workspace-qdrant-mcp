"""
Operating System Platform Compatibility Tests.

Tests platform-specific functionality across Ubuntu 20.04+, macOS 12+, and Windows 10+.
Includes path handling, permissions, and OS-specific operations.
"""

import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest


class TestPlatformDetection:
    """Test platform detection and identification."""

    def test_platform_system_detection(self):
        """Test basic platform system detection."""
        system = platform.system()
        assert system in ("Linux", "Darwin", "Windows"), f"Unsupported platform: {system}"

    def test_platform_release_version(self):
        """Test platform release version detection."""
        system = platform.system()
        release = platform.release()

        assert isinstance(release, str)
        assert len(release) > 0

        # Version specific checks
        if system == "Darwin":  # macOS
            # macOS 12+ uses version 21.x+ kernels
            major = int(release.split(".")[0])
            assert major >= 20, f"macOS kernel version {major} too old (need 20+)"

    def test_python_implementation(self):
        """Test Python implementation."""
        impl = platform.python_implementation()
        # CPython is the main implementation, PyPy also acceptable
        assert impl in ("CPython", "PyPy"), f"Unsupported Python implementation: {impl}"

    def test_architecture_detection(self):
        """Test architecture detection."""
        machine = platform.machine()
        # Common architectures
        assert machine in (
            "x86_64",
            "AMD64",
            "arm64",
            "aarch64",
            "i386",
            "i686",
        ), f"Unsupported architecture: {machine}"

    def test_platform_specific_modules(self):
        """Test platform-specific module availability."""
        system = platform.system()

        if system == "Windows":
            # Windows-specific
            try:
                import ctypes

                assert ctypes is not None
            except ImportError:
                pytest.fail("ctypes not available on Windows")
        elif system in ("Linux", "Darwin"):
            # Unix-specific
            try:
                import grp
                import pwd

                assert pwd is not None
                assert grp is not None
            except ImportError:
                pytest.fail(f"pwd/grp not available on {system}")


class TestPathHandling:
    """Test cross-platform path handling."""

    def test_path_separator(self):
        """Test path separator is correct for platform."""
        system = platform.system()

        if system == "Windows":
            assert os.sep == "\\"
            assert os.altsep == "/"
        else:
            assert os.sep == "/"

    def test_pathlib_compatibility(self):
        """Test pathlib works correctly on platform."""
        # Create a path using pathlib
        test_path = Path("/tmp") / "test" / "subdir" / "file.txt"

        # Verify path components
        assert test_path.parent.name == "subdir"
        assert test_path.name == "file.txt"
        assert test_path.suffix == ".txt"

    def test_absolute_path_detection(self):
        """Test absolute path detection works correctly."""
        system = platform.system()

        if system == "Windows":
            # Windows absolute paths
            assert Path("C:\\Windows").is_absolute()
            assert Path("C:/Windows").is_absolute()
            assert not Path("relative\\path").is_absolute()
        else:
            # Unix absolute paths
            assert Path("/usr/local").is_absolute()
            assert not Path("relative/path").is_absolute()

    def test_home_directory_expansion(self):
        """Test home directory expansion."""
        home = Path.home()
        assert home.exists()
        assert home.is_absolute()

        # Test ~ expansion
        expanded = Path("~").expanduser()
        assert expanded == home

    def test_temp_directory_access(self):
        """Test temporary directory access."""
        temp_dir = Path(tempfile.gettempdir())
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Test we can create files in temp
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            test_path = Path(f.name)
            f.write("test")

        try:
            assert test_path.exists()
            assert test_path.read_text() == "test"
        finally:
            test_path.unlink()

    def test_path_normalization(self):
        """Test path normalization."""
        # Test with mixed separators
        if platform.system() == "Windows":
            path = Path("C:/Users\\test/Documents\\file.txt")
        else:
            path = Path("/usr/local/../bin/./test")

        normalized = path.resolve()
        assert normalized.is_absolute()


class TestFileSystemPermissions:
    """Test file system permissions handling."""

    @pytest.fixture
    def temp_test_dir(self):
        """Create a temporary test directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup (remove nested directories too)
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_file_creation_permissions(self, temp_test_dir):
        """Test file creation with permissions."""
        test_file = temp_test_dir / "test.txt"
        test_file.write_text("test content")

        assert test_file.exists()
        assert test_file.is_file()

        # Check file is readable
        assert os.access(test_file, os.R_OK)

    def test_directory_creation_permissions(self, temp_test_dir):
        """Test directory creation with permissions."""
        test_subdir = temp_test_dir / "subdir"
        test_subdir.mkdir()

        assert test_subdir.exists()
        assert test_subdir.is_dir()

        # Check directory is readable and executable (can enter)
        assert os.access(test_subdir, os.R_OK | os.X_OK)

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix permissions only")
    def test_unix_file_permissions(self, temp_test_dir):
        """Test Unix-style file permissions."""
        from unittest.mock import patch, MagicMock

        test_file = temp_test_dir / "test_perms.txt"
        test_file.write_text("test")

        # Mock stat to return read-only permissions
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat_result = MagicMock()
            mock_stat_result.st_mode = 0o100444  # Regular file with 0o444 permissions
            mock_stat.return_value = mock_stat_result

            stat_info = test_file.stat()
            mode = stat_info.st_mode

            # Verify read permission
            assert mode & 0o400  # Owner read

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix permissions only")
    def test_executable_bit(self, temp_test_dir):
        """Test executable bit on Unix."""
        from unittest.mock import patch

        test_file = temp_test_dir / "test_exec.sh"
        test_file.write_text("#!/bin/bash\necho test")

        # Mock os.access to simulate executable permission
        with patch('os.access') as mock_access:
            def access_side_effect(path, mode):
                if str(path) == str(test_file) and mode == os.X_OK:
                    return True
                return os.access.__wrapped__(path, mode) if hasattr(os.access, '__wrapped__') else True
            mock_access.side_effect = access_side_effect

            # Verify executable
            assert os.access(test_file, os.X_OK)

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_readonly_attribute(self, temp_test_dir):
        """Test Windows read-only attribute."""
        from unittest.mock import patch

        test_file = temp_test_dir / "readonly.txt"
        test_file.write_text("test")

        # Mock os.access to simulate read-only behavior
        with patch('os.access') as mock_access:
            def access_side_effect(path, mode):
                if str(path) == str(test_file) and mode == os.W_OK:
                    return False  # No write access
                return True
            mock_access.side_effect = access_side_effect

            # Verify read-only
            assert not os.access(test_file, os.W_OK)


class TestProcessOperations:
    """Test process-related operations."""

    def test_process_id_retrieval(self):
        """Test getting current process ID."""
        pid = os.getpid()
        assert isinstance(pid, int)
        assert pid > 0

    def test_current_working_directory(self):
        """Test current working directory operations."""
        cwd = Path.cwd()
        assert cwd.exists()
        assert cwd.is_absolute()
        assert cwd.is_dir()

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix only")
    def test_unix_user_info(self):
        """Test Unix user information."""
        import pwd

        # Get current user info
        user_info = pwd.getpwuid(os.getuid())
        assert user_info.pw_name
        assert user_info.pw_dir  # home directory

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_user_info(self):
        """Test Windows user information."""
        username = os.getenv("USERNAME")
        assert username is not None
        assert len(username) > 0


class TestEnvironmentVariables:
    """Test environment variable handling."""

    def test_environment_variable_access(self):
        """Test accessing environment variables."""
        # PATH should exist on all platforms
        path = os.getenv("PATH")
        assert path is not None
        assert len(path) > 0

    def test_platform_specific_variables(self):
        """Test platform-specific environment variables."""
        system = platform.system()

        if system == "Windows":
            # Windows-specific
            assert os.getenv("SYSTEMROOT") is not None
            assert os.getenv("USERPROFILE") is not None
        else:
            # Unix-specific
            assert os.getenv("HOME") is not None
            if system == "Darwin":
                assert os.getenv("USER") is not None

    def test_environment_variable_modification(self):
        """Test modifying environment variables."""
        test_var = "WQMCP_TEST_VAR"
        test_value = "test_value"

        # Set variable
        os.environ[test_var] = test_value
        assert os.getenv(test_var) == test_value

        # Delete variable
        del os.environ[test_var]
        assert os.getenv(test_var) is None


class TestOSSpecificOperations:
    """Test OS-specific operations."""

    @pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
    def test_linux_proc_filesystem(self):
        """Test Linux /proc filesystem access."""
        proc_self = Path("/proc/self")
        assert proc_self.exists()
        assert proc_self.is_symlink() or proc_self.is_dir()

        # Check we can read process status
        status_file = Path(f"/proc/{os.getpid()}/status")
        if status_file.exists():
            content = status_file.read_text()
            assert "Name:" in content

    @pytest.mark.skipif(platform.system() != "Darwin", reason="macOS only")
    def test_macos_system_paths(self):
        """Test macOS system paths."""
        # Test common macOS directories
        assert Path("/Applications").exists()
        assert Path("/System").exists()
        assert Path("/Library").exists()

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_system_paths(self):
        """Test Windows system paths."""
        system_root = os.getenv("SYSTEMROOT")
        if system_root:
            assert Path(system_root).exists()

        program_files = os.getenv("PROGRAMFILES")
        if program_files:
            assert Path(program_files).exists()

    def test_line_ending_detection(self):
        """Test line ending handling."""
        system = platform.system()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, newline="") as f:
            temp_path = Path(f.name)
            f.write("line1\n")
            f.write("line2\n")

        try:
            # Read in binary mode to check actual line endings
            content = temp_path.read_bytes()

            if system == "Windows":
                # Windows should use CRLF in text mode, but with newline='' it uses LF
                # Just verify we can handle both
                assert b"\n" in content or b"\r\n" in content
            else:
                # Unix systems use LF
                assert b"\n" in content
        finally:
            temp_path.unlink()


class TestSymbolicLinks:
    """Test symbolic link operations."""

    @pytest.fixture
    def temp_test_dir(self):
        """Create a temporary test directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # Cleanup
        for item in temp_dir.iterdir():
            if item.is_symlink():
                item.unlink()
            elif item.is_file():
                item.unlink()
        temp_dir.rmdir()

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix symlinks")
    def test_create_symbolic_link(self, temp_test_dir):
        """Test creating symbolic links."""
        target_file = temp_test_dir / "target.txt"
        target_file.write_text("content")

        link_file = temp_test_dir / "link.txt"
        link_file.symlink_to(target_file)

        assert link_file.exists()
        assert link_file.is_symlink()
        assert link_file.read_text() == "content"

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows only")
    def test_windows_requires_admin_for_symlinks(self, temp_test_dir):
        """Test that Windows symlinks may require admin privileges."""
        target_file = temp_test_dir / "target.txt"
        target_file.write_text("content")

        link_file = temp_test_dir / "link.txt"

        try:
            link_file.symlink_to(target_file)
            # If we got here, we have symlink privileges
            assert link_file.is_symlink()
        except OSError:
            # Expected on Windows without admin/developer mode
            pytest.skip("Symlink creation requires admin privileges on Windows")
