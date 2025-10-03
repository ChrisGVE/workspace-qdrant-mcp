"""
File system operation mocking for comprehensive testing.

Provides sophisticated mocking for file system operations including
reading, writing, directory operations, and failure scenarios.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
from unittest.mock import AsyncMock, Mock, mock_open

from .error_injection import ErrorInjector, FailureScenarios


class FileSystemErrorInjector(ErrorInjector):
    """Specialized error injector for file system operations."""

    def __init__(self):
        super().__init__()
        self.failure_modes = {
            "permission_denied": {"probability": 0.0, "errno": 13},
            "file_not_found": {"probability": 0.0, "errno": 2},
            "disk_full": {"probability": 0.0, "errno": 28},
            "io_error": {"probability": 0.0, "errno": 5},
            "access_denied": {"probability": 0.0, "errno": 13},
            "path_too_long": {"probability": 0.0, "errno": 36},
            "too_many_open_files": {"probability": 0.0, "errno": 24},
            "device_busy": {"probability": 0.0, "errno": 16},
            "read_only_filesystem": {"probability": 0.0, "errno": 30},
            "interrupted_system_call": {"probability": 0.0, "errno": 4},
        }

    def configure_permission_issues(self, probability: float = 0.1):
        """Configure permission-related failures."""
        self.failure_modes["permission_denied"]["probability"] = probability
        self.failure_modes["access_denied"]["probability"] = probability / 2

    def configure_resource_issues(self, probability: float = 0.1):
        """Configure resource-related failures."""
        self.failure_modes["disk_full"]["probability"] = probability
        self.failure_modes["too_many_open_files"]["probability"] = probability / 2

    def configure_io_issues(self, probability: float = 0.1):
        """Configure I/O-related failures."""
        self.failure_modes["io_error"]["probability"] = probability
        self.failure_modes["interrupted_system_call"]["probability"] = probability / 2


class FileSystemMock:
    """Comprehensive file system operations mock."""

    def __init__(self, error_injector: Optional[FileSystemErrorInjector] = None):
        self.error_injector = error_injector or FileSystemErrorInjector()
        self.virtual_filesystem: Dict[str, Any] = {}
        self.operation_history: List[Dict[str, Any]] = []
        self.open_files: Set[str] = set()

        # Setup method mocks
        self._setup_file_operations()
        self._setup_directory_operations()
        self._setup_path_operations()

    def _setup_file_operations(self):
        """Setup file operation mocks."""
        self.open = mock_open()
        self.read_text = Mock(side_effect=self._mock_read_text)
        self.write_text = Mock(side_effect=self._mock_write_text)
        self.read_bytes = Mock(side_effect=self._mock_read_bytes)
        self.write_bytes = Mock(side_effect=self._mock_write_bytes)

    def _setup_directory_operations(self):
        """Setup directory operation mocks."""
        self.mkdir = Mock(side_effect=self._mock_mkdir)
        self.rmdir = Mock(side_effect=self._mock_rmdir)
        self.listdir = Mock(side_effect=self._mock_listdir)
        self.walk = Mock(side_effect=self._mock_walk)

    def _setup_path_operations(self):
        """Setup path operation mocks."""
        self.exists = Mock(side_effect=self._mock_exists)
        self.is_file = Mock(side_effect=self._mock_is_file)
        self.is_dir = Mock(side_effect=self._mock_is_dir)
        self.stat = Mock(side_effect=self._mock_stat)
        self.chmod = Mock(side_effect=self._mock_chmod)

    def _inject_filesystem_error(self, operation: str, path: str) -> None:
        """Inject filesystem errors based on configuration."""
        if not self.error_injector.should_inject_error():
            return

        error_type = self.error_injector.get_random_error()
        error_config = self.error_injector.failure_modes.get(error_type, {})
        errno = error_config.get("errno", 1)

        if error_type == "permission_denied":
            raise PermissionError(f"Permission denied: '{path}'")
        elif error_type == "file_not_found":
            raise FileNotFoundError(f"No such file or directory: '{path}'")
        elif error_type == "disk_full":
            raise OSError(errno, f"No space left on device: '{path}'")
        elif error_type == "io_error":
            raise OSError(errno, f"I/O error: '{path}'")
        elif error_type == "access_denied":
            raise PermissionError(f"Access denied: '{path}'")
        elif error_type == "path_too_long":
            raise OSError(errno, f"File name too long: '{path}'")
        elif error_type == "too_many_open_files":
            raise OSError(errno, "Too many open files")
        elif error_type == "device_busy":
            raise OSError(errno, f"Device or resource busy: '{path}'")
        elif error_type == "read_only_filesystem":
            raise OSError(errno, f"Read-only file system: '{path}'")
        elif error_type == "interrupted_system_call":
            raise OSError(errno, f"Interrupted system call: '{path}'")

    def _mock_read_text(self, path: Union[str, Path], encoding: str = "utf-8") -> str:
        """Mock text file reading."""
        path_str = str(path)
        self._inject_filesystem_error("read", path_str)

        self.operation_history.append({
            "operation": "read_text",
            "path": path_str,
            "encoding": encoding
        })

        if path_str in self.virtual_filesystem:
            content = self.virtual_filesystem[path_str].get("content", "")
            if isinstance(content, str):
                return content
            else:
                raise ValueError(f"Binary content cannot be read as text: '{path_str}'")
        else:
            raise FileNotFoundError(f"No such file: '{path_str}'")

    def _mock_write_text(self, path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
        """Mock text file writing."""
        path_str = str(path)
        self._inject_filesystem_error("write", path_str)

        self.operation_history.append({
            "operation": "write_text",
            "path": path_str,
            "content_length": len(content),
            "encoding": encoding
        })

        self.virtual_filesystem[path_str] = {
            "content": content,
            "type": "file",
            "encoding": encoding,
            "size": len(content.encode(encoding))
        }

    def _mock_read_bytes(self, path: Union[str, Path]) -> bytes:
        """Mock binary file reading."""
        path_str = str(path)
        self._inject_filesystem_error("read", path_str)

        self.operation_history.append({
            "operation": "read_bytes",
            "path": path_str
        })

        if path_str in self.virtual_filesystem:
            content = self.virtual_filesystem[path_str].get("content", b"")
            if isinstance(content, bytes):
                return content
            elif isinstance(content, str):
                return content.encode("utf-8")
            else:
                return b""
        else:
            raise FileNotFoundError(f"No such file: '{path_str}'")

    def _mock_write_bytes(self, path: Union[str, Path], content: bytes) -> None:
        """Mock binary file writing."""
        path_str = str(path)
        self._inject_filesystem_error("write", path_str)

        self.operation_history.append({
            "operation": "write_bytes",
            "path": path_str,
            "content_length": len(content)
        })

        self.virtual_filesystem[path_str] = {
            "content": content,
            "type": "file",
            "size": len(content)
        }

    def _mock_mkdir(self, path: Union[str, Path], mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        """Mock directory creation."""
        path_str = str(path)
        self._inject_filesystem_error("mkdir", path_str)

        self.operation_history.append({
            "operation": "mkdir",
            "path": path_str,
            "mode": mode,
            "parents": parents,
            "exist_ok": exist_ok
        })

        if path_str in self.virtual_filesystem and not exist_ok:
            raise FileExistsError(f"Directory already exists: '{path_str}'")

        self.virtual_filesystem[path_str] = {
            "type": "directory",
            "mode": mode,
            "contents": []
        }

    def _mock_rmdir(self, path: Union[str, Path]) -> None:
        """Mock directory removal."""
        path_str = str(path)
        self._inject_filesystem_error("rmdir", path_str)

        self.operation_history.append({
            "operation": "rmdir",
            "path": path_str
        })

        if path_str not in self.virtual_filesystem:
            raise FileNotFoundError(f"No such directory: '{path_str}'")

        if self.virtual_filesystem[path_str].get("type") != "directory":
            raise NotADirectoryError(f"Not a directory: '{path_str}'")

        del self.virtual_filesystem[path_str]

    def _mock_listdir(self, path: Union[str, Path]) -> List[str]:
        """Mock directory listing."""
        path_str = str(path)
        self._inject_filesystem_error("listdir", path_str)

        self.operation_history.append({
            "operation": "listdir",
            "path": path_str
        })

        if path_str not in self.virtual_filesystem:
            raise FileNotFoundError(f"No such directory: '{path_str}'")

        if self.virtual_filesystem[path_str].get("type") != "directory":
            raise NotADirectoryError(f"Not a directory: '{path_str}'")

        # Return mock directory contents
        return [
            "file1.txt",
            "file2.py",
            "subdirectory",
            ".hidden_file"
        ]

    def _mock_walk(self, path: Union[str, Path]):
        """Mock directory tree walking."""
        path_str = str(path)
        self._inject_filesystem_error("walk", path_str)

        self.operation_history.append({
            "operation": "walk",
            "path": path_str
        })

        # Generate mock walk results
        yield (path_str, ["subdir1", "subdir2"], ["file1.txt", "file2.py"])
        yield (f"{path_str}/subdir1", [], ["nested_file.md"])
        yield (f"{path_str}/subdir2", ["deeper"], ["config.json"])

    def _mock_exists(self, path: Union[str, Path]) -> bool:
        """Mock path existence check."""
        path_str = str(path)

        self.operation_history.append({
            "operation": "exists",
            "path": path_str
        })

        return path_str in self.virtual_filesystem

    def _mock_is_file(self, path: Union[str, Path]) -> bool:
        """Mock file type check."""
        path_str = str(path)

        self.operation_history.append({
            "operation": "is_file",
            "path": path_str
        })

        if path_str in self.virtual_filesystem:
            return self.virtual_filesystem[path_str].get("type") == "file"
        return False

    def _mock_is_dir(self, path: Union[str, Path]) -> bool:
        """Mock directory type check."""
        path_str = str(path)

        self.operation_history.append({
            "operation": "is_dir",
            "path": path_str
        })

        if path_str in self.virtual_filesystem:
            return self.virtual_filesystem[path_str].get("type") == "directory"
        return False

    def _mock_stat(self, path: Union[str, Path]):
        """Mock file status."""
        path_str = str(path)
        self._inject_filesystem_error("stat", path_str)

        self.operation_history.append({
            "operation": "stat",
            "path": path_str
        })

        if path_str not in self.virtual_filesystem:
            raise FileNotFoundError(f"No such file or directory: '{path_str}'")

        # Return mock stat result
        return Mock(
            st_size=self.virtual_filesystem[path_str].get("size", 1024),
            st_mode=self.virtual_filesystem[path_str].get("mode", 0o644),
            st_mtime=1640995200,  # 2022-01-01
            st_atime=1640995200,
            st_ctime=1640995200
        )

    def _mock_chmod(self, path: Union[str, Path], mode: int) -> None:
        """Mock file permission change."""
        path_str = str(path)
        self._inject_filesystem_error("chmod", path_str)

        self.operation_history.append({
            "operation": "chmod",
            "path": path_str,
            "mode": mode
        })

        if path_str not in self.virtual_filesystem:
            raise FileNotFoundError(f"No such file or directory: '{path_str}'")

        self.virtual_filesystem[path_str]["mode"] = mode

    def add_file(self, path: str, content: Union[str, bytes], file_type: str = "file") -> None:
        """Add a file to the virtual filesystem."""
        self.virtual_filesystem[path] = {
            "content": content,
            "type": file_type,
            "size": len(content) if isinstance(content, (str, bytes)) else 0
        }

    def add_directory(self, path: str) -> None:
        """Add a directory to the virtual filesystem."""
        self.virtual_filesystem[path] = {
            "type": "directory",
            "contents": []
        }

    def get_operation_history(self) -> List[Dict[str, Any]]:
        """Get history of file operations."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset mock state."""
        self.virtual_filesystem.clear()
        self.operation_history.clear()
        self.open_files.clear()
        self.error_injector.reset()


class DirectoryOperationMock:
    """Mock for directory-specific operations."""

    def __init__(self, error_injector: Optional[FileSystemErrorInjector] = None):
        self.error_injector = error_injector or FileSystemErrorInjector()
        self.operation_history: List[Dict[str, Any]] = []

        # Setup method mocks
        self.create_directory_tree = Mock(side_effect=self._mock_create_directory_tree)
        self.copy_directory = AsyncMock(side_effect=self._mock_copy_directory)
        self.move_directory = AsyncMock(side_effect=self._mock_move_directory)
        self.delete_directory_tree = AsyncMock(side_effect=self._mock_delete_directory_tree)

    def _mock_create_directory_tree(self, base_path: str, structure: Dict[str, Any]) -> None:
        """Mock creating a directory tree from structure."""
        if self.error_injector.should_inject_error():
            raise OSError(f"Failed to create directory tree at: {base_path}")

        self.operation_history.append({
            "operation": "create_directory_tree",
            "base_path": base_path,
            "structure_keys": list(structure.keys())
        })

    async def _mock_copy_directory(self, src: str, dst: str) -> None:
        """Mock directory copying."""
        if self.error_injector.should_inject_error():
            raise OSError(f"Failed to copy directory from {src} to {dst}")

        await asyncio.sleep(0.1)  # Simulate operation time

        self.operation_history.append({
            "operation": "copy_directory",
            "src": src,
            "dst": dst
        })

    async def _mock_move_directory(self, src: str, dst: str) -> None:
        """Mock directory moving."""
        if self.error_injector.should_inject_error():
            raise OSError(f"Failed to move directory from {src} to {dst}")

        await asyncio.sleep(0.1)  # Simulate operation time

        self.operation_history.append({
            "operation": "move_directory",
            "src": src,
            "dst": dst
        })

    async def _mock_delete_directory_tree(self, path: str) -> None:
        """Mock recursive directory deletion."""
        if self.error_injector.should_inject_error():
            raise OSError(f"Failed to delete directory tree: {path}")

        await asyncio.sleep(0.1)  # Simulate operation time

        self.operation_history.append({
            "operation": "delete_directory_tree",
            "path": path
        })

    def reset_state(self) -> None:
        """Reset operation state."""
        self.operation_history.clear()
        self.error_injector.reset()


def create_filesystem_mock(
    with_error_injection: bool = False,
    error_probability: float = 0.1
) -> FileSystemMock:
    """
    Create a filesystem mock with optional error injection.

    Args:
        with_error_injection: Enable error injection
        error_probability: Probability of errors (0.0 to 1.0)

    Returns:
        Configured FileSystemMock instance
    """
    error_injector = None
    if with_error_injection:
        error_injector = FileSystemErrorInjector()
        error_injector.configure_permission_issues(error_probability)
        error_injector.configure_resource_issues(error_probability)
        error_injector.configure_io_issues(error_probability)

    return FileSystemMock(error_injector)


def create_directory_operation_mock(
    with_error_injection: bool = False,
    error_probability: float = 0.1
) -> DirectoryOperationMock:
    """Create a directory operations mock with optional error injection."""
    error_injector = None
    if with_error_injection:
        error_injector = FileSystemErrorInjector()
        error_injector.configure_permission_issues(error_probability)
        error_injector.configure_resource_issues(error_probability)

    return DirectoryOperationMock(error_injector)
