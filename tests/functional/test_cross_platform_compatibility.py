"""
Cross-Platform Compatibility Testing

Tests that validate functionality across Windows, macOS, and Linux with different
Python versions and system configurations to ensure consistent behavior.

This module implements subtask 203.4 of the End-to-End Functional Testing Framework.
"""

import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch

import pytest


class CrossPlatformTestEnvironment:
    """Test environment for cross-platform compatibility validation."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.platform_info = self._get_platform_info()
        self.config_dir = self._get_platform_config_dir()
        self.cli_executable = "uv run wqm"

        self.setup_environment()

    def setup_environment(self):
        """Set up platform-specific test environment."""
        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create platform-specific configuration
        config_content = self._get_platform_config()
        config_file = self.config_dir / "config.yaml"

        with open(config_file, 'w', encoding='utf-8') as f:
            import yaml
            yaml.dump(config_content, f)

    def _get_platform_info(self) -> dict[str, Any]:
        """Get comprehensive platform information."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "architecture": platform.architecture(),
            "node": platform.node(),
            "platform": platform.platform(),
            "is_windows": platform.system() == "Windows",
            "is_macos": platform.system() == "Darwin",
            "is_linux": platform.system() == "Linux",
            "path_separator": os.path.sep,
            "pathsep": os.pathsep,
            "linesep": os.linesep,
            "encoding": sys.getfilesystemencoding(),
            "max_path_length": self._get_max_path_length()
        }

    def _get_max_path_length(self) -> int:
        """Get maximum path length for the platform."""
        system = platform.system()
        if system == "Windows":
            return 260  # Traditional Windows limit (can be longer with long path support)
        elif system == "Darwin":
            return 1024  # macOS limit
        else:
            return 4096  # Linux and other Unix-like systems

    def _get_platform_config_dir(self) -> Path:
        """Get platform-appropriate configuration directory."""
        if self.platform_info["is_windows"]:
            # Windows: %APPDATA%/workspace-qdrant
            Path(os.environ.get("APPDATA", self.tmp_path))
        elif self.platform_info["is_macos"]:
            # macOS: ~/Library/Application Support/workspace-qdrant
            Path.home() / "Library" / "Application Support"
        else:
            # Linux/Unix: ~/.config/workspace-qdrant
            Path.home() / ".config"

        return self.tmp_path / "platform_config" / "workspace-qdrant"

    def _get_platform_config(self) -> dict[str, Any]:
        """Get platform-specific configuration."""
        config = {
            "qdrant_url": "http://localhost:6333",
            "platform": {
                "system": self.platform_info["system"],
                "python_version": self.platform_info["python_version"],
                "encoding": self.platform_info["encoding"],
                "path_separator": self.platform_info["path_separator"]
            },
            "paths": {
                "log_dir": str(self._get_platform_log_dir()),
                "cache_dir": str(self._get_platform_cache_dir()),
                "data_dir": str(self._get_platform_data_dir())
            }
        }

        # Platform-specific adjustments
        if self.platform_info["is_windows"]:
            config["daemon"] = {
                "service_name": "WorkspaceQdrantService",
                "startup_type": "automatic"
            }
        elif self.platform_info["is_macos"]:
            config["daemon"] = {
                "plist_name": "com.workspace.qdrant",
                "launch_agent": True
            }
        else:  # Linux
            config["daemon"] = {
                "service_name": "workspace-qdrant",
                "systemd_user": True
            }

        return config

    def _get_platform_log_dir(self) -> Path:
        """Get platform-appropriate log directory."""
        if self.platform_info["is_windows"]:
            return self.tmp_path / "logs"
        elif self.platform_info["is_macos"]:
            return self.tmp_path / "logs"
        else:
            return self.tmp_path / "logs"

    def _get_platform_cache_dir(self) -> Path:
        """Get platform-appropriate cache directory."""
        if self.platform_info["is_windows"]:
            return self.tmp_path / "cache"
        elif self.platform_info["is_macos"]:
            return self.tmp_path / "cache"
        else:
            return self.tmp_path / "cache"

    def _get_platform_data_dir(self) -> Path:
        """Get platform-appropriate data directory."""
        if self.platform_info["is_windows"]:
            return self.tmp_path / "data"
        elif self.platform_info["is_macos"]:
            return self.tmp_path / "data"
        else:
            return self.tmp_path / "data"

    def run_cli_command(
        self,
        command: str,
        cwd: Path | None = None,
        timeout: int = 30,
        env_vars: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """Execute CLI command with platform-specific handling."""
        if cwd is None:
            cwd = self.tmp_path

        # Set up environment with platform-specific variables
        env = os.environ.copy()
        env.update({
            "WQM_CONFIG_DIR": str(self.config_dir),
            "PYTHONPATH": str(Path.cwd()),
            "PYTHONIOENCODING": "utf-8",  # Ensure consistent encoding
        })

        # Platform-specific environment adjustments
        if self.platform_info["is_windows"]:
            env["PATHEXT"] = env.get("PATHEXT", "") + ";.PY"

        if env_vars:
            env.update(env_vars)

        # Execute command with platform-appropriate shell
        shell = self.platform_info["is_windows"]

        try:
            result = subprocess.run(
                f"{self.cli_executable} {command}",
                shell=shell,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                encoding='utf-8',
                errors='replace'  # Handle encoding issues gracefully
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", f"Command execution failed: {e}"

    def create_test_files_with_platform_paths(self) -> dict[str, Path]:
        """Create test files with platform-specific path handling."""
        test_files = {}

        # Standard files
        standard_files = {
            "simple.txt": "Simple text file content",
            "unicode_content.txt": "Unicode content: ñoño αβγ 中文 🚀",
            "long_name_file.txt": "File with a long name for testing",
        }

        for filename, content in standard_files.items():
            file_path = self.tmp_path / filename
            file_path.write_text(content, encoding='utf-8')
            test_files[filename] = file_path

        # Platform-specific path tests
        if self.platform_info["is_windows"]:
            # Test Windows-specific path issues
            test_files.update(self._create_windows_specific_files())
        else:
            # Test Unix-like path issues
            test_files.update(self._create_unix_specific_files())

        return test_files

    def _create_windows_specific_files(self) -> dict[str, Path]:
        """Create Windows-specific test files."""
        files = {}

        # Test files with Windows-problematic characters (when safe)
        safe_windows_file = self.tmp_path / "windows_safe_file.txt"
        safe_windows_file.write_text("Windows-safe file content", encoding='utf-8')
        files["windows_safe"] = safe_windows_file

        # Test long path (but within limits for compatibility)
        long_dir = self.tmp_path / "very" / "long" / "directory" / "structure"
        long_dir.mkdir(parents=True, exist_ok=True)
        long_file = long_dir / "long_path_file.txt"
        long_file.write_text("File in long path", encoding='utf-8')
        files["long_path"] = long_file

        return files

    def _create_unix_specific_files(self) -> dict[str, Path]:
        """Create Unix-specific test files."""
        files = {}

        # Test files with spaces and special characters
        space_file = self.tmp_path / "file with spaces.txt"
        space_file.write_text("File with spaces in name", encoding='utf-8')
        files["spaces"] = space_file

        # Test hidden file
        hidden_file = self.tmp_path / ".hidden_file.txt"
        hidden_file.write_text("Hidden file content", encoding='utf-8')
        files["hidden"] = hidden_file

        return files

    def test_path_handling(self, test_path: Path) -> bool:
        """Test platform-specific path handling."""
        try:
            # Test path existence
            test_path.exists()

            # Test path resolution
            test_path.resolve()

            # Test path string representation
            str(test_path)

            # Test path conversion
            if self.platform_info["is_windows"]:
                # Test Windows path conversion
                test_path.as_posix()
                PureWindowsPath(test_path)
            else:
                # Test Unix path handling
                PurePosixPath(test_path)

            return True
        except Exception:
            return False


class CrossPlatformValidator:
    """Validates cross-platform functionality and compatibility."""

    @staticmethod
    def validate_platform_detection(platform_info: dict[str, Any]) -> bool:
        """Validate platform detection accuracy."""
        required_fields = [
            "system", "python_version", "architecture",
            "encoding", "path_separator"
        ]
        return all(field in platform_info for field in required_fields)

    @staticmethod
    def validate_path_compatibility(paths: list[Path]) -> dict[str, bool]:
        """Validate path compatibility across platforms."""
        results = {}

        for path in paths:
            try:
                # Test basic path operations
                path.exists()
                path.resolve()

                results[str(path)] = True
            except Exception:
                results[str(path)] = False

        return results

    @staticmethod
    def validate_encoding_handling(content: str, file_path: Path) -> bool:
        """Validate Unicode and encoding handling."""
        try:
            # Write and read with explicit encoding
            file_path.write_text(content, encoding='utf-8')
            read_content = file_path.read_text(encoding='utf-8')
            return content == read_content
        except (UnicodeError, OSError):
            return False

    @staticmethod
    def validate_command_execution(
        return_code: int,
        stdout: str,
        stderr: str
    ) -> dict[str, Any]:
        """Validate command execution results."""
        return {
            "executed": return_code != -1,
            "successful": return_code == 0,
            "has_output": len(stdout + stderr) > 0,
            "encoding_issues": any(char in (stdout + stderr) for char in ['\ufffd', '?'])
        }


@pytest.mark.functional
@pytest.mark.cross_platform
class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility and consistency."""

    @pytest.fixture
    def platform_env(self, tmp_path):
        """Create cross-platform test environment."""
        return CrossPlatformTestEnvironment(tmp_path)

    @pytest.fixture
    def validator(self):
        """Create cross-platform validator."""
        return CrossPlatformValidator()

    def test_platform_detection(self, platform_env, validator):
        """Test platform detection and information gathering."""
        platform_info = platform_env.platform_info

        # Validate platform detection
        assert validator.validate_platform_detection(platform_info)

        # Test platform-specific attributes
        if platform_info["is_windows"]:
            assert platform_info["path_separator"] == "\\"
            assert "Windows" in platform_info["system"]
        elif platform_info["is_macos"]:
            assert platform_info["path_separator"] == "/"
            assert platform_info["system"] == "Darwin"
        elif platform_info["is_linux"]:
            assert platform_info["path_separator"] == "/"
            assert platform_info["system"] == "Linux"

        # Validate Python version format
        version_parts = platform_info["python_version"].split(".")
        assert len(version_parts) >= 2
        assert all(part.isdigit() for part in version_parts[:2])

    def test_cli_command_execution_consistency(self, platform_env, validator):
        """Test CLI command execution consistency across platforms."""
        # Test basic commands that should work on all platforms
        basic_commands = [
            "--version",
            "--help",
            "admin status"
        ]

        results = {}
        for command in basic_commands:
            return_code, stdout, stderr = platform_env.run_cli_command(command)
            validation = validator.validate_command_execution(return_code, stdout, stderr)
            results[command] = validation

        # Validate command execution
        assert all(result["executed"] for result in results.values())

        # Version and help should work consistently
        assert results["--version"]["successful"]
        assert results["--help"]["successful"]

        # All commands should produce output
        assert all(result["has_output"] for result in results.values())

        # No encoding issues
        assert not any(result["encoding_issues"] for result in results.values())

    def test_path_handling_consistency(self, platform_env, validator):
        """Test path handling consistency across platforms."""
        test_files = platform_env.create_test_files_with_platform_paths()
        paths = list(test_files.values())

        # Validate path compatibility
        path_results = validator.validate_path_compatibility(paths)

        # All paths should be handled correctly
        assert all(result for result in path_results.values())

        # Test CLI operations with different path types
        for _file_type, file_path in test_files.items():
            if file_path.exists():
                return_code, stdout, stderr = platform_env.run_cli_command(
                    f"ingest file {file_path}"
                )

                # Should handle file ingestion attempt
                validation = validator.validate_command_execution(return_code, stdout, stderr)
                assert validation["executed"]
                assert validation["has_output"]

    def test_unicode_and_encoding_support(self, platform_env, validator):
        """Test Unicode and encoding support across platforms."""
        # Test various Unicode content
        unicode_tests = [
            "Simple ASCII text",
            "Latin characters: ñoño café naïve",
            "Greek: αβγδε ζηθικ λμνξο",
            "Cyrillic: абвгд еёжзи йклмн",
            "Chinese: 你好世界 测试文档",
            "Japanese: こんにちは テスト",
            "Emoji: 🚀 🐍 💻 📁 🔍",
            "Mixed: Hello 世界 🌍 Testing αβγ"
        ]

        for i, content in enumerate(unicode_tests):
            test_file = platform_env.tmp_path / f"unicode_test_{i}.txt"

            # Test encoding handling
            encoding_valid = validator.validate_encoding_handling(content, test_file)
            assert encoding_valid, f"Failed to handle Unicode content: {content[:50]}..."

            # Test CLI operations with Unicode files
            if test_file.exists():
                return_code, stdout, stderr = platform_env.run_cli_command(
                    f"ingest file {test_file}"
                )

                validation = validator.validate_command_execution(return_code, stdout, stderr)
                assert validation["executed"]
                assert not validation["encoding_issues"]

    def test_configuration_file_compatibility(self, platform_env, validator):
        """Test configuration file compatibility across platforms."""
        # Test configuration loading
        return_code, stdout, stderr = platform_env.run_cli_command("config show")

        validation = validator.validate_command_execution(return_code, stdout, stderr)
        assert validation["executed"]
        assert validation["has_output"]

        # Test custom configuration with platform-specific paths
        custom_config = platform_env.config_dir / "custom-config.yaml"
        config_content = {
            "qdrant_url": "http://localhost:6333",
            "log_file": str(platform_env._get_platform_log_dir() / "test.log"),
            "data_dir": str(platform_env._get_platform_data_dir()),
            "platform_specific": {
                "system": platform_env.platform_info["system"],
                "encoding": platform_env.platform_info["encoding"]
            }
        }

        import yaml
        with open(custom_config, 'w', encoding='utf-8') as f:
            yaml.dump(config_content, f)

        # Test with custom configuration
        return_code, stdout, stderr = platform_env.run_cli_command(
            f"--config {custom_config} admin status"
        )

        validation = validator.validate_command_execution(return_code, stdout, stderr)
        assert validation["executed"]

    def test_service_management_platform_differences(self, platform_env, validator):
        """Test service management across different platforms."""
        # Test service status (should handle platform differences gracefully)
        return_code, stdout, stderr = platform_env.run_cli_command("service status")

        validation = validator.validate_command_execution(return_code, stdout, stderr)
        assert validation["executed"]
        assert validation["has_output"]

        # Platform-specific service operations
        if platform_env.platform_info["is_windows"]:
            # Windows service operations
            commands = ["service install", "service uninstall"]
        elif platform_env.platform_info["is_macos"]:
            # macOS launchd operations
            commands = ["service install", "service uninstall"]
        else:
            # Linux systemd operations
            commands = ["service install", "service uninstall"]

        for command in commands:
            return_code, stdout, stderr = platform_env.run_cli_command(command)
            validation = validator.validate_command_execution(return_code, stdout, stderr)

            # Should execute and provide feedback (may fail without permissions)
            assert validation["executed"]
            assert validation["has_output"]

    def test_file_system_permissions(self, platform_env, validator):
        """Test file system permissions handling across platforms."""
        # Create test file with content
        test_file = platform_env.tmp_path / "permissions_test.txt"
        test_file.write_text("Permission test content", encoding='utf-8')

        # Test basic file operations
        assert test_file.exists()
        assert test_file.is_file()

        # Platform-specific permission tests
        if not platform_env.platform_info["is_windows"]:
            # Unix-like systems: Mock permission checking
            # Mock os.access to simulate read-only file
            with patch('os.access', return_value=False):
                # Test CLI operations with read-only file
                return_code, stdout, stderr = platform_env.run_cli_command(
                    f"ingest file {test_file}"
                )

                validation = validator.validate_command_execution(return_code, stdout, stderr)
                assert validation["executed"]

    def test_python_version_compatibility(self, platform_env, validator):
        """Test compatibility with different Python versions."""
        python_info = platform_env.platform_info

        # Validate minimum Python version support
        version_parts = python_info["python_version"].split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])

        # Should support Python 3.8+
        assert major >= 3
        if major == 3:
            assert minor >= 8

        # Test Python implementation compatibility
        implementation = python_info["python_implementation"]
        assert implementation in ["CPython", "PyPy"]

        # Test basic CLI functionality with current Python version
        return_code, stdout, stderr = platform_env.run_cli_command("--version")
        validation = validator.validate_command_execution(return_code, stdout, stderr)

        assert validation["successful"]
        assert not validation["encoding_issues"]

    def test_environment_variable_handling(self, platform_env, validator):
        """Test environment variable handling across platforms."""
        # Test with various environment variables
        env_tests = {
            "WQM_DEBUG": "true",
            "WQM_LOG_LEVEL": "DEBUG",
            "QDRANT_URL": "http://localhost:6333"
        }

        for env_var, value in env_tests.items():
            return_code, stdout, stderr = platform_env.run_cli_command(
                "admin status",
                env_vars={env_var: value}
            )

            validation = validator.validate_command_execution(return_code, stdout, stderr)
            assert validation["executed"]
            assert validation["has_output"]

    @pytest.mark.slow
    def test_long_path_handling(self, platform_env, validator):
        """Test long path handling across platforms."""
        # Create progressively longer paths
        max_length = min(platform_env.platform_info["max_path_length"] - 50, 200)

        # Build long directory structure
        long_path = platform_env.tmp_path
        path_components = []

        while len(str(long_path)) < max_length:
            component = f"dir_{len(path_components)}"
            path_components.append(component)
            long_path = long_path / component

            try:
                long_path.mkdir(exist_ok=True)
            except OSError:
                # Path too long, stop here
                break

        # Create file in long path
        if long_path.exists():
            long_file = long_path / "long_path_test.txt"
            try:
                long_file.write_text("Long path test content", encoding='utf-8')

                # Test CLI operations with long path
                return_code, stdout, stderr = platform_env.run_cli_command(
                    f"ingest file {long_file}"
                )

                validation = validator.validate_command_execution(return_code, stdout, stderr)
                assert validation["executed"]

            except OSError:
                # Expected on some platforms with path length limits
                pass

    def test_concurrent_operations_platform_stability(self, platform_env, validator):
        """Test concurrent operations stability across platforms."""
        import queue
        import threading

        results_queue = queue.Queue()

        def worker(worker_id):
            try:
                # Test multiple operations per worker
                operations = [
                    "admin status",
                    "--version",
                    "config show"
                ]

                worker_results = []
                for op in operations:
                    return_code, stdout, stderr = platform_env.run_cli_command(op)
                    validation = validator.validate_command_execution(return_code, stdout, stderr)
                    worker_results.append(validation["executed"])

                results_queue.put({
                    "worker_id": worker_id,
                    "success": all(worker_results),
                    "operations": len(operations)
                })

            except Exception as e:
                results_queue.put({
                    "worker_id": worker_id,
                    "error": str(e)
                })

        # Start multiple threads
        num_workers = 3
        threads = []

        for i in range(num_workers):
            thread = threading.Thread(target=worker, args=(i,))
            thread.start()
            threads.append(thread)

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        # Validate concurrent operations
        assert len(results) == num_workers
        assert all("error" not in result for result in results)
        assert all(result["success"] for result in results)

    def test_system_integration_points(self, platform_env, validator):
        """Test system integration points across platforms."""
        # Test system-specific integration points
        system_commands = []

        if platform_env.platform_info["is_windows"]:
            system_commands = [
                "admin status",  # Should work on Windows
            ]
        elif platform_env.platform_info["is_macos"]:
            system_commands = [
                "admin status",  # Should work on macOS
            ]
        else:  # Linux
            system_commands = [
                "admin status",  # Should work on Linux
            ]

        for command in system_commands:
            return_code, stdout, stderr = platform_env.run_cli_command(command)
            validation = validator.validate_command_execution(return_code, stdout, stderr)

            assert validation["executed"]
            assert validation["has_output"]
            assert not validation["encoding_issues"]
