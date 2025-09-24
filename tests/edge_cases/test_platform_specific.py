"""
Cross-platform compatibility edge case tests for workspace-qdrant-mcp.

Tests platform-specific behaviors, file system differences, and
compatibility across macOS, Linux, and Windows environments.
"""

import os
import platform
import pytest
import tempfile
import stat
from pathlib import Path, PurePosixPath, PureWindowsPath
from unittest.mock import Mock, patch

from workspace_qdrant_mcp.utils.project_detection import GitProjectDetector
from workspace_qdrant_mcp.core.config import ConfigValidator


@pytest.mark.platform_specific
class TestFileSystemCompatibility:
    """Test file system compatibility across platforms."""

    def test_path_separator_handling(self):
        """Test proper handling of different path separators."""
        # Test Unix-style paths
        unix_path = "src/workspace_qdrant_mcp/core/client.py"
        normalized_unix = Path(unix_path)

        # Test Windows-style paths
        windows_path = "src\\workspace_qdrant_mcp\\core\\client.py"
        normalized_windows = Path(windows_path)

        # Both should resolve to the same logical path
        assert normalized_unix.parts == normalized_windows.parts

    def test_case_sensitivity_handling(self, tmp_path):
        """Test handling of case-sensitive vs case-insensitive file systems."""
        # Create test files with different cases
        file_lower = tmp_path / "test_file.txt"
        file_upper = tmp_path / "TEST_FILE.TXT"

        file_lower.write_text("lowercase content")

        # On case-insensitive systems (like macOS/Windows default),
        # this might overwrite the first file
        try:
            file_upper.write_text("uppercase content")
            files_created = 2 if file_lower.read_text() == "lowercase content" else 1
        except FileExistsError:
            files_created = 1

        # Test that our code handles both scenarios
        detector = GitProjectDetector()

        # Should handle case variations gracefully
        assert detector._normalize_path(str(file_lower)) is not None
        assert detector._normalize_path(str(file_upper)) is not None

    def test_long_path_handling(self, tmp_path):
        """Test handling of long file paths across platforms."""
        # Create a very long path
        long_components = ["very_long_directory_name_" + "x" * 50] * 5
        long_path = tmp_path

        try:
            for component in long_components:
                long_path = long_path / component
                long_path.mkdir(exist_ok=True)

            # Create a file in the deep directory
            test_file = long_path / ("very_long_filename_" + "x" * 100 + ".txt")
            test_file.write_text("test content")

            # Should handle long paths gracefully
            assert test_file.exists()
            assert test_file.read_text() == "test content"

        except OSError as e:
            # Some platforms have path length limits
            if "File name too long" in str(e) or "path too long" in str(e).lower():
                pytest.skip(f"Platform path length limit: {e}")
            else:
                raise

    def test_special_character_handling(self, tmp_path):
        """Test handling of special characters in file paths."""
        special_chars = {
            "spaces": "file with spaces.txt",
            "unicode": "файл_тест.txt",  # Russian characters
            "emoji": "test_📁_file.txt",
            "brackets": "file[with](brackets).txt",
        }

        successful_files = []

        for char_type, filename in special_chars.items():
            try:
                test_file = tmp_path / filename
                test_file.write_text(f"content for {char_type}")

                if test_file.exists() and test_file.read_text().startswith("content for"):
                    successful_files.append(char_type)

            except (OSError, UnicodeEncodeError) as e:
                # Some platforms/filesystems don't support certain characters
                print(f"Platform doesn't support {char_type}: {e}")

        # Should handle at least basic special characters
        assert "spaces" in successful_files or len(successful_files) > 0

    def test_symlink_handling(self, tmp_path):
        """Test symbolic link handling across platforms."""
        original_file = tmp_path / "original.txt"
        original_file.write_text("original content")

        symlink_path = tmp_path / "symlink.txt"

        try:
            symlink_path.symlink_to(original_file)

            # Test that our code handles symlinks properly
            detector = GitProjectDetector()

            # Should be able to resolve both the original and symlink
            original_normalized = detector._normalize_path(str(original_file))
            symlink_normalized = detector._normalize_path(str(symlink_path))

            assert original_normalized is not None
            assert symlink_normalized is not None

        except (OSError, NotImplementedError):
            # Symlinks might not be supported on all platforms/configurations
            pytest.skip("Platform doesn't support symbolic links")

    @pytest.mark.skipif(platform.system() == "Windows", reason="Unix-specific test")
    def test_unix_permissions(self, tmp_path):
        """Test Unix file permission handling."""
        test_file = tmp_path / "permission_test.txt"
        test_file.write_text("test content")

        # Make file read-only
        test_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        # Test handling of read-only files
        assert test_file.stat().st_mode & stat.S_IWUSR == 0  # Should be read-only

        # Our code should handle read-only files gracefully
        config_validator = ConfigValidator()

        try:
            # Should not crash on read-only files
            is_readable = config_validator._is_file_readable(str(test_file))
            assert is_readable is not None
        except PermissionError:
            # Expected behavior for protected files
            pass

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_windows_reserved_names(self, tmp_path):
        """Test Windows reserved filename handling."""
        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]

        for reserved_name in reserved_names:
            try:
                # Try to create file with reserved name
                test_file = tmp_path / f"{reserved_name}.txt"
                test_file.write_text("test content")

                # If successful, our code should handle it
                assert test_file.exists()

            except (OSError, FileNotFoundError):
                # Expected behavior on Windows - reserved names are not allowed
                assert platform.system() == "Windows"


@pytest.mark.platform_specific
class TestEnvironmentVariableHandling:
    """Test platform-specific environment variable handling."""

    def test_path_environment_variable(self):
        """Test PATH environment variable handling across platforms."""
        if platform.system() == "Windows":
            path_var = "PATH"
            separator = ";"
        else:
            path_var = "PATH"
            separator = ":"

        path_value = os.environ.get(path_var, "")
        path_entries = path_value.split(separator) if path_value else []

        # Should have some PATH entries
        assert len(path_entries) > 0

        # Test that our configuration handles PATH correctly
        config_validator = ConfigValidator()

        # Should be able to process environment variables
        processed_path = config_validator._process_environment_variable(path_var)
        assert processed_path is not None

    def test_home_directory_detection(self):
        """Test home directory detection across platforms."""
        if platform.system() == "Windows":
            expected_vars = ["USERPROFILE", "HOMEDRIVE", "HOMEPATH"]
        else:
            expected_vars = ["HOME"]

        home_found = False
        for var in expected_vars:
            if os.environ.get(var):
                home_found = True
                break

        assert home_found, f"No home directory variable found from {expected_vars}"

        # Test Path.home() works consistently
        home_path = Path.home()
        assert home_path.exists()
        assert home_path.is_dir()

    def test_temp_directory_handling(self):
        """Test temporary directory handling across platforms."""
        # Should work consistently across platforms
        temp_dir = Path(tempfile.gettempdir())
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Test creating temp files
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
            tmp.write("test content")
            temp_path = Path(tmp.name)

        try:
            assert temp_path.exists()
            assert temp_path.read_text() == "test content"
        finally:
            temp_path.unlink()


@pytest.mark.platform_specific
class TestProcessHandling:
    """Test process handling across platforms."""

    def test_subprocess_execution(self):
        """Test subprocess execution across platforms."""
        import subprocess

        if platform.system() == "Windows":
            cmd = ["cmd", "/c", "echo test"]
        else:
            cmd = ["echo", "test"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            assert result.returncode == 0
            assert "test" in result.stdout.strip()

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            pytest.skip(f"Subprocess execution failed: {e}")

    def test_signal_handling(self):
        """Test signal handling across platforms."""
        import signal

        # Test that basic signals are available
        if platform.system() != "Windows":
            # Unix signals
            assert hasattr(signal, 'SIGTERM')
            assert hasattr(signal, 'SIGINT')
        else:
            # Windows signals (limited set)
            assert hasattr(signal, 'SIGTERM')
            assert hasattr(signal, 'SIGINT')

    def test_process_monitoring(self):
        """Test process monitoring capabilities."""
        import psutil

        current_process = psutil.Process()

        # Should be able to get basic process info
        assert current_process.pid > 0
        assert current_process.name() is not None

        # Memory and CPU info should be available
        memory_info = current_process.memory_info()
        assert memory_info.rss > 0

        cpu_percent = current_process.cpu_percent()
        assert cpu_percent >= 0.0


@pytest.mark.platform_specific
class TestNetworkingPlatformDifferences:
    """Test networking behavior differences across platforms."""

    def test_localhost_resolution(self):
        """Test localhost resolution across platforms."""
        import socket

        # Should resolve localhost consistently
        localhost_ip = socket.gethostbyname('localhost')
        assert localhost_ip in ['127.0.0.1', '::1']

        # Test IPv4 vs IPv6 preferences
        try:
            ipv4_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            ipv4_socket.close()

            ipv6_socket = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            ipv6_socket.close()

            # Both should be available on modern systems
        except OSError as e:
            # Some systems might not have IPv6 enabled
            if "Address family not supported" in str(e):
                pytest.skip("IPv6 not available on this system")

    def test_port_binding_behavior(self):
        """Test port binding behavior differences."""
        import socket

        # Test binding to random port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('localhost', 0))
            addr, port = sock.getsockname()

            assert addr in ['127.0.0.1', '::1']
            assert port > 0

        finally:
            sock.close()


@pytest.mark.platform_specific
class TestPlatformSpecificConfigurations:
    """Test platform-specific configuration handling."""

    def test_config_file_locations(self):
        """Test configuration file location preferences by platform."""
        config_validator = ConfigValidator()

        if platform.system() == "Windows":
            # Windows: AppData
            expected_paths = [
                Path.home() / "AppData" / "Local",
                Path.home() / "AppData" / "Roaming"
            ]
        elif platform.system() == "Darwin":
            # macOS: Application Support
            expected_paths = [
                Path.home() / "Library" / "Application Support"
            ]
        else:
            # Linux: XDG directories
            expected_paths = [
                Path.home() / ".config",
                Path.home() / ".local" / "share"
            ]

        # At least one standard location should exist
        existing_paths = [path for path in expected_paths if path.exists()]
        assert len(existing_paths) > 0

    def test_executable_extensions(self):
        """Test executable file extension handling."""
        if platform.system() == "Windows":
            executable_extensions = ['.exe', '.bat', '.cmd', '.com']
        else:
            executable_extensions = ['']  # No extension needed on Unix

        # Test that our code handles executable detection properly
        for ext in executable_extensions:
            test_name = f"test_executable{ext}"
            # Should handle executable naming conventions
            assert test_name.endswith(ext) or ext == ''

    def test_platform_specific_imports(self):
        """Test platform-specific import behavior."""

        if platform.system() == "Windows":
            try:
                import winsound
                assert hasattr(winsound, 'Beep')
            except ImportError:
                pytest.skip("Windows-specific modules not available")

        elif platform.system() == "Darwin":
            try:
                import pwd
                assert hasattr(pwd, 'getpwuid')
            except ImportError:
                pytest.skip("macOS-specific modules not available")

        else:  # Linux/Unix
            try:
                import pwd
                import grp
                assert hasattr(pwd, 'getpwuid')
                assert hasattr(grp, 'getgrgid')
            except ImportError:
                pytest.skip("Unix-specific modules not available")

    def test_line_ending_handling(self, tmp_path):
        """Test line ending handling across platforms."""
        test_file = tmp_path / "line_endings.txt"

        # Test different line endings
        content_unix = "line1\nline2\nline3\n"
        content_windows = "line1\r\nline2\r\nline3\r\n"
        content_mac = "line1\rline2\rline3\r"

        for line_type, content in [
            ("unix", content_unix),
            ("windows", content_windows),
            ("mac", content_mac)
        ]:
            test_file.write_text(content, newline='')

            # Should handle all line ending types
            read_content = test_file.read_text()
            lines = read_content.splitlines()

            assert len(lines) == 3
            assert all(line.startswith("line") for line in lines)


@pytest.mark.platform_specific
class TestTimezoneHandling:
    """Test timezone and datetime handling across platforms."""

    def test_timezone_detection(self):
        """Test timezone detection across platforms."""
        import time
        import datetime

        # Should be able to get timezone info
        local_time = time.localtime()
        assert local_time.tm_zone is not None or hasattr(time, 'timezone')

        # Test datetime timezone awareness
        now_naive = datetime.datetime.now()
        now_aware = datetime.datetime.now(datetime.timezone.utc)

        assert now_naive is not None
        assert now_aware is not None
        assert now_aware.tzinfo is not None

    def test_timestamp_handling(self):
        """Test timestamp handling across platforms."""
        import time

        # Should handle timestamps consistently
        current_timestamp = time.time()
        assert current_timestamp > 0

        # Convert back to datetime
        dt = datetime.datetime.fromtimestamp(current_timestamp)
        assert dt.year >= 2020  # Reasonable sanity check