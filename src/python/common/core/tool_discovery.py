"""Tool discovery system for finding and validating executables.

This module provides comprehensive tool discovery capabilities for finding
executables in PATH, validating their presence and executability, and
retrieving version information. It supports cross-platform operations
(Windows/macOS/Linux) with timeout handling and custom path support.

Architecture:
    - PATH scanning with glob-style pattern matching
    - Executable validation with cross-platform compatibility
    - Version detection with configurable flags and parsing
    - Timeout handling for subprocess operations
    - Custom path support for non-standard installations
    - Comprehensive logging for debugging

Example:
    ```python
    from workspace_qdrant_mcp.core.tool_discovery import ToolDiscovery

    # Initialize with default settings
    discovery = ToolDiscovery()

    # Find an executable
    python_path = discovery.find_executable("python")

    # Validate executable
    if discovery.validate_executable(python_path):
        version = discovery.get_version(python_path)
        print(f"Python version: {version}")

    # Scan for executables matching pattern
    lsp_servers = discovery.scan_path_for_executables("*-language-server")
    ```
"""

import fnmatch
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class ToolDiscovery:
    """Tool discovery system for finding and validating executables.

    This class provides methods to discover executables in system PATH,
    validate their presence and executability, and retrieve version
    information. It supports custom paths, timeout handling, and
    cross-platform operations.

    Attributes:
        custom_paths: List of custom paths to search before system PATH
        timeout: Timeout in seconds for subprocess operations
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        timeout: int = 5
    ):
        """Initialize tool discovery system.

        Args:
            config: Optional configuration dictionary containing:
                - custom_paths: List of custom paths to search
                - timeout: Override default timeout
            timeout: Default timeout in seconds for subprocess operations
        """
        self.timeout = timeout
        self.custom_paths: List[str] = []

        if config:
            # Extract custom paths from config
            if "custom_paths" in config:
                self.custom_paths = config["custom_paths"]
                logger.debug(
                    f"Initialized with {len(self.custom_paths)} custom paths"
                )

            # Override timeout if specified
            if "timeout" in config:
                self.timeout = config["timeout"]
                logger.debug(f"Using custom timeout: {self.timeout}s")

        logger.debug(
            f"ToolDiscovery initialized with timeout={self.timeout}s, "
            f"custom_paths={len(self.custom_paths)}"
        )

    def find_executable(self, name: str) -> Optional[str]:
        """Find an executable in PATH or custom paths.

        Searches for the executable first in custom paths, then in system PATH.
        Returns the absolute path if found, None otherwise.

        Args:
            name: Name of the executable to find (e.g., "python", "git")

        Returns:
            Absolute path to executable if found, None otherwise
        """
        # First check custom paths
        for custom_path in self.custom_paths:
            custom_path_obj = Path(custom_path)
            if not custom_path_obj.exists():
                continue

            # Check if custom_path is a directory or a file
            if custom_path_obj.is_dir():
                # Search for executable in directory
                potential_path = custom_path_obj / name
                if platform.system() == "Windows":
                    # Windows may need .exe extension
                    for ext in ["", ".exe", ".bat", ".cmd"]:
                        candidate = Path(str(potential_path) + ext)
                        if candidate.exists() and self.validate_executable(str(candidate)):
                            logger.debug(
                                f"Found '{name}' in custom path: {candidate}"
                            )
                            return str(candidate.resolve())
                else:
                    if potential_path.exists() and self.validate_executable(str(potential_path)):
                        logger.debug(
                            f"Found '{name}' in custom path: {potential_path}"
                        )
                        return str(potential_path.resolve())
            else:
                # custom_path is a file, check if it matches
                if custom_path_obj.name == name or custom_path_obj.stem == name:
                    if self.validate_executable(custom_path):
                        logger.debug(
                            f"Found '{name}' at custom path: {custom_path}"
                        )
                        return str(custom_path_obj.resolve())

        # Fall back to system PATH
        result = shutil.which(name)

        if result:
            logger.debug(f"Found '{name}' in system PATH: {result}")
            return str(Path(result).resolve())
        else:
            logger.debug(f"Executable '{name}' not found in PATH or custom paths")
            return None

    def scan_path_for_executables(self, pattern: str) -> List[str]:
        """Scan PATH for executables matching a glob-style pattern.

        Searches both custom paths and system PATH for executables that
        match the provided glob pattern. Returns a deduplicated list of
        absolute paths.

        Args:
            pattern: Glob-style pattern (e.g., "*-language-server", "python*")

        Returns:
            List of absolute paths to matching executables
        """
        found_executables: List[str] = []
        seen_paths: set = set()

        # Collect all paths to search (custom + system PATH)
        search_paths = self.custom_paths.copy()

        # Add system PATH directories
        system_path = os.environ.get("PATH", "")
        if system_path:
            search_paths.extend(system_path.split(os.pathsep))

        logger.debug(
            f"Scanning {len(search_paths)} directories for pattern '{pattern}'"
        )

        for path_str in search_paths:
            path = Path(path_str)

            # Skip non-existent directories
            if not path.exists() or not path.is_dir():
                continue

            try:
                # List all files in directory
                for entry in path.iterdir():
                    # Skip if already seen (handle duplicates in PATH)
                    if str(entry.resolve()) in seen_paths:
                        continue

                    # Check if filename matches pattern
                    if fnmatch.fnmatch(entry.name, pattern):
                        # Validate it's actually executable
                        if self.validate_executable(str(entry)):
                            resolved_path = str(entry.resolve())
                            found_executables.append(resolved_path)
                            seen_paths.add(resolved_path)
                            logger.debug(f"Found matching executable: {resolved_path}")

            except (PermissionError, OSError) as e:
                # Skip directories we can't read
                logger.debug(f"Cannot read directory {path}: {e}")
                continue

        logger.debug(
            f"Found {len(found_executables)} executables matching '{pattern}'"
        )
        return found_executables

    def validate_executable(self, path: str) -> bool:
        """Check if a path exists and is executable.

        Performs cross-platform validation of executable files. On Unix-like
        systems, checks the executable bit. On Windows, checks for common
        executable extensions.

        Args:
            path: Path to validate

        Returns:
            True if path exists and is executable, False otherwise
        """
        path_obj = Path(path)

        # Check if path exists
        if not path_obj.exists():
            return False

        # Check if it's a file (not a directory)
        if not path_obj.is_file():
            return False

        # Cross-platform executable check
        if platform.system() == "Windows":
            # On Windows, check file extension
            executable_extensions = {".exe", ".bat", ".cmd", ".com", ".ps1"}
            return path_obj.suffix.lower() in executable_extensions
        else:
            # On Unix-like systems, check executable bit
            return os.access(path, os.X_OK)

    def get_version(
        self,
        executable: str,
        version_flag: str = "--version"
    ) -> Optional[str]:
        """Get version information from an executable.

        Runs the executable with the specified version flag and attempts
        to parse the version string from the output. Uses timeout to
        prevent hangs and handles errors gracefully.

        Args:
            executable: Path to executable or name if in PATH
            version_flag: Flag to retrieve version (default: "--version")

        Returns:
            Version string if successfully retrieved, None otherwise
        """
        # Resolve executable path if needed
        if not Path(executable).exists():
            resolved_path = self.find_executable(executable)
            if not resolved_path:
                logger.debug(f"Cannot get version: '{executable}' not found")
                return None
            executable = resolved_path

        # Validate executable
        if not self.validate_executable(executable):
            logger.debug(f"Cannot get version: '{executable}' not executable")
            return None

        try:
            # Run executable with version flag
            logger.debug(
                f"Getting version for '{executable}' with flag '{version_flag}'"
            )

            result = subprocess.run(
                [executable, version_flag],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False  # Don't raise on non-zero exit
            )

            # Try stdout first, then stderr
            output = result.stdout.strip() or result.stderr.strip()

            if output:
                logger.debug(f"Version output for '{executable}': {output[:100]}")
                return output
            else:
                logger.debug(f"No version output from '{executable}'")
                return None

        except subprocess.TimeoutExpired:
            logger.warning(
                f"Timeout ({self.timeout}s) getting version for '{executable}'"
            )
            return None

        except FileNotFoundError:
            logger.debug(f"Executable not found: '{executable}'")
            return None

        except (OSError, subprocess.SubprocessError) as e:
            logger.debug(f"Error getting version for '{executable}': {e}")
            return None
