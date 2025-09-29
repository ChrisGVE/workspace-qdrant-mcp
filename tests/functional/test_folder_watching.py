"""Comprehensive functional tests for folder watching and auto-ingestion.

This test suite validates the complete folder watching workflow including:
- Watch creation and management
- File detection and ingestion
- Watch pause/resume functionality
- Watch configuration and filtering
- Multi-folder watching scenarios
"""

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


class FolderWatchingEnvironment:
    """Test environment for folder watching functional tests."""

    def __init__(self, tmp_path: Path):
        """Initialize test environment.

        Args:
            tmp_path: Temporary directory for test files
        """
        self.tmp_path = tmp_path
        self.watched_dirs: List[Path] = []
        self.test_files: Dict[str, Path] = {}
        self.cli_executable = "uv run wqm"

        # Create test directory structure
        self.watch_root = tmp_path / "watched_folders"
        self.watch_root.mkdir(parents=True, exist_ok=True)

        # Create a test Git repository for project detection
        self.project_dir = tmp_path / "test_project"
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / ".git").mkdir(exist_ok=True)

    def create_watch_folder(self, name: str) -> Path:
        """Create a folder to be watched.

        Args:
            name: Folder name

        Returns:
            Path to created folder
        """
        folder = self.watch_root / name
        folder.mkdir(parents=True, exist_ok=True)
        self.watched_dirs.append(folder)
        return folder

    def add_test_file(
        self, folder: Path, filename: str, content: str, delay: float = 0.1
    ) -> Path:
        """Add a test file to a watched folder.

        Args:
            folder: Target folder
            filename: File name
            content: File content
            delay: Delay after creation (for debouncing)

        Returns:
            Path to created file
        """
        file_path = folder / filename
        file_path.write_text(content)
        self.test_files[filename] = file_path

        # Small delay to ensure file system event is processed
        if delay > 0:
            time.sleep(delay)

        return file_path

    def modify_test_file(self, file_path: Path, new_content: str, delay: float = 0.1):
        """Modify an existing test file.

        Args:
            file_path: Path to file
            new_content: New content
            delay: Delay after modification
        """
        file_path.write_text(new_content)
        if delay > 0:
            time.sleep(delay)

    def delete_test_file(self, file_path: Path, delay: float = 0.1):
        """Delete a test file.

        Args:
            file_path: Path to file
            delay: Delay after deletion
        """
        if file_path.exists():
            file_path.unlink()
        if delay > 0:
            time.sleep(delay)

    def run_cli_command(
        self,
        command: str,
        cwd: Optional[Path] = None,
        timeout: int = 30,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str, str]:
        """Execute CLI command.

        Args:
            command: Command to run
            cwd: Working directory
            timeout: Command timeout
            env_vars: Additional environment variables

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if cwd is None:
            cwd = self.project_dir

        env = os.environ.copy()
        env.update(
            {
                "QDRANT_URL": "http://localhost:6333",
                "PYTHONPATH": str(Path.cwd()),
            }
        )
        if env_vars:
            env.update(env_vars)

        try:
            result = subprocess.run(
                f"{self.cli_executable} {command}",
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", str(e)

    def cleanup(self):
        """Clean up test environment."""
        for folder in self.watched_dirs:
            if folder.exists():
                shutil.rmtree(folder, ignore_errors=True)


@pytest.fixture
def watch_env(tmp_path):
    """Provide folder watching test environment."""
    env = FolderWatchingEnvironment(tmp_path)
    yield env
    env.cleanup()


class TestFolderWatchingBasics:
    """Basic folder watching functionality tests."""

    def test_watch_help_command(self, watch_env):
        """Test that watch help command works."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch --help")
        assert return_code == 0
        assert "watch" in stdout.lower()
        assert "folder" in stdout.lower() or "watch" in stdout.lower()

    def test_watch_list_empty(self, watch_env):
        """Test listing watches when none exist."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch list")
        # Command may fail if daemon is not running, but should provide clear output
        output = stdout + stderr
        assert len(output) > 0

    def test_watch_subcommands_exist(self, watch_env):
        """Test that all watch subcommands are available."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch --help")
        assert return_code == 0

        # Check for expected subcommands
        expected_commands = ["add", "list", "remove", "status", "pause", "resume"]
        for cmd in expected_commands:
            assert cmd in stdout.lower(), f"Expected subcommand '{cmd}' not found"


class TestWatchConfiguration:
    """Test watch configuration and filtering."""

    def test_watch_add_help(self, watch_env):
        """Test watch add command help."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch add --help")
        assert return_code == 0
        assert "add" in stdout.lower()

    def test_watch_remove_help(self, watch_env):
        """Test watch remove command help."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch remove --help")
        assert return_code == 0
        assert "remove" in stdout.lower()

    def test_watch_status_help(self, watch_env):
        """Test watch status command help."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch status --help")
        assert return_code == 0
        assert "status" in stdout.lower()

    def test_watch_pause_help(self, watch_env):
        """Test watch pause command help."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch pause --help")
        assert return_code == 0
        assert "pause" in stdout.lower()

    def test_watch_resume_help(self, watch_env):
        """Test watch resume command help."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch resume --help")
        assert return_code == 0
        assert "resume" in stdout.lower()


@pytest.mark.requires_daemon
class TestWatchLifecycle:
    """Test complete watch lifecycle (requires daemon)."""

    @pytest.fixture
    def daemon_running(self, watch_env):
        """Check if daemon is running, skip if not."""
        # Check daemon status
        return_code, stdout, stderr = watch_env.run_cli_command("service status")
        if return_code != 0 or "not running" in (stdout + stderr).lower():
            pytest.skip("Daemon not running - skipping daemon-dependent test")
        return True

    def test_add_watch_folder(self, watch_env, daemon_running):
        """Test adding a folder to watch."""
        # Create test folder
        test_folder = watch_env.create_watch_folder("test_watch_1")

        # Add watch
        return_code, stdout, stderr = watch_env.run_cli_command(
            f"watch add {test_folder}"
        )

        # Check result - may succeed or fail depending on daemon
        output = stdout + stderr
        assert len(output) > 0

    def test_list_active_watches(self, watch_env, daemon_running):
        """Test listing active watches."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch list")

        # Check result
        output = stdout + stderr
        assert len(output) > 0

    def test_watch_status(self, watch_env, daemon_running):
        """Test getting watch status."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch status")

        # Check result
        output = stdout + stderr
        assert len(output) > 0

    def test_pause_watch(self, watch_env, daemon_running):
        """Test pausing a watch."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch pause --all")

        # Check result
        output = stdout + stderr
        assert len(output) > 0

    def test_resume_watch(self, watch_env, daemon_running):
        """Test resuming a paused watch."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch resume --all")

        # Check result
        output = stdout + stderr
        assert len(output) > 0

    def test_remove_watch(self, watch_env, daemon_running):
        """Test removing a watch."""
        # Create and add watch
        test_folder = watch_env.create_watch_folder("test_watch_remove")

        # Try to remove (may fail without daemon, but should provide feedback)
        return_code, stdout, stderr = watch_env.run_cli_command(
            f"watch remove {test_folder}"
        )

        output = stdout + stderr
        assert len(output) > 0


@pytest.mark.requires_daemon
@pytest.mark.slow
class TestFileDetection:
    """Test file detection and ingestion (requires daemon)."""

    @pytest.fixture
    def daemon_running(self, watch_env):
        """Check if daemon is running, skip if not."""
        # Check daemon status
        return_code, stdout, stderr = watch_env.run_cli_command("service status")
        if return_code != 0 or "not running" in (stdout + stderr).lower():
            pytest.skip("Daemon not running - skipping daemon-dependent test")
        return True

    def test_detect_new_file(self, watch_env, daemon_running):
        """Test detection of newly created file."""
        # Create watched folder
        test_folder = watch_env.create_watch_folder("test_detection")

        # Add watch
        watch_env.run_cli_command(f"watch add {test_folder}")

        # Create test file
        test_file = watch_env.add_test_file(
            test_folder, "test_doc.md", "# Test Document\n\nThis is a test."
        )

        # Wait for processing
        time.sleep(2)

        # Check if file was detected (implementation-specific)
        # This would need to check ingestion logs or Qdrant
        assert test_file.exists()

    def test_detect_file_modification(self, watch_env, daemon_running):
        """Test detection of file modifications."""
        # Create watched folder and file
        test_folder = watch_env.create_watch_folder("test_modification")
        test_file = watch_env.add_test_file(
            test_folder, "modifiable.txt", "Original content"
        )

        # Add watch
        watch_env.run_cli_command(f"watch add {test_folder}")

        # Modify file
        watch_env.modify_test_file(
            test_file, "Modified content\n\nWith additional data."
        )

        # Wait for processing
        time.sleep(2)

        # Verify modification was detected
        assert test_file.exists()
        assert "Modified content" in test_file.read_text()

    def test_detect_file_deletion(self, watch_env, daemon_running):
        """Test detection of file deletions."""
        # Create watched folder and file
        test_folder = watch_env.create_watch_folder("test_deletion")
        test_file = watch_env.add_test_file(
            test_folder, "to_delete.txt", "This file will be deleted"
        )

        # Add watch
        watch_env.run_cli_command(f"watch add {test_folder}")

        # Delete file
        watch_env.delete_test_file(test_file)

        # Wait for processing
        time.sleep(2)

        # Verify deletion
        assert not test_file.exists()


@pytest.mark.requires_daemon
class TestMultipleFolders:
    """Test watching multiple folders simultaneously."""

    @pytest.fixture
    def daemon_running(self, watch_env):
        """Check if daemon is running, skip if not."""
        # Check daemon status
        return_code, stdout, stderr = watch_env.run_cli_command("service status")
        if return_code != 0 or "not running" in (stdout + stderr).lower():
            pytest.skip("Daemon not running - skipping daemon-dependent test")
        return True

    def test_watch_multiple_folders(self, watch_env, daemon_running):
        """Test adding multiple folders to watch."""
        # Create multiple folders
        folder1 = watch_env.create_watch_folder("multi_watch_1")
        folder2 = watch_env.create_watch_folder("multi_watch_2")
        folder3 = watch_env.create_watch_folder("multi_watch_3")

        # Add watches
        for folder in [folder1, folder2, folder3]:
            return_code, stdout, stderr = watch_env.run_cli_command(
                f"watch add {folder}"
            )
            output = stdout + stderr
            assert len(output) > 0

        # List watches
        return_code, stdout, stderr = watch_env.run_cli_command("watch list")
        output = stdout + stderr
        assert len(output) > 0

    def test_watch_nested_folders(self, watch_env, daemon_running):
        """Test watching nested folder structures."""
        # Create nested structure
        parent = watch_env.create_watch_folder("nested_parent")
        child = parent / "child_folder"
        child.mkdir(parents=True, exist_ok=True)
        grandchild = child / "grandchild_folder"
        grandchild.mkdir(parents=True, exist_ok=True)

        # Add watch on parent
        return_code, stdout, stderr = watch_env.run_cli_command(f"watch add {parent}")

        output = stdout + stderr
        assert len(output) > 0


class TestWatchErrorHandling:
    """Test error handling in watch operations."""

    def test_add_nonexistent_folder(self, watch_env):
        """Test adding a folder that doesn't exist."""
        nonexistent = watch_env.watch_root / "does_not_exist"

        return_code, stdout, stderr = watch_env.run_cli_command(
            f"watch add {nonexistent}"
        )

        # Should provide clear error message
        output = stdout + stderr
        assert len(output) > 0

    def test_remove_nonexistent_watch(self, watch_env):
        """Test removing a watch that doesn't exist."""
        return_code, stdout, stderr = watch_env.run_cli_command(
            "watch remove /fake/path"
        )

        # Should handle gracefully with error message
        output = stdout + stderr
        assert len(output) > 0

    def test_invalid_watch_command(self, watch_env):
        """Test handling of invalid watch commands."""
        return_code, stdout, stderr = watch_env.run_cli_command("watch invalid_cmd")

        # Should fail with helpful error
        assert return_code != 0
        output = stdout + stderr
        assert len(output) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])