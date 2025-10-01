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

    def discover_compilers(self) -> Dict[str, Optional[str]]:
        """Discover available compilers on the system.

        Searches for common C/C++ and other language compilers in PATH
        and custom paths. Returns a dictionary mapping compiler names to
        their absolute paths.

        Returns:
            Dictionary mapping compiler name to absolute path or None if not found.
            Example:
            {
                "gcc": "/usr/bin/gcc",
                "g++": "/usr/bin/g++",
                "clang": "/usr/bin/clang",
                "clang++": "/usr/bin/clang++",
                "cl": None,  # Not found (Windows-specific)
                "zig": None,  # Not found
                "cc": "/usr/bin/cc"
            }
        """
        # Define compilers to search for
        compiler_names = ["gcc", "g++", "clang", "clang++", "cc", "zig"]

        # Add Windows-specific compilers if on Windows
        if platform.system() == "Windows":
            compiler_names.extend(["cl", "cl.exe"])

        compilers: Dict[str, Optional[str]] = {}

        logger.debug("Discovering compilers...")

        for compiler_name in compiler_names:
            path = self.find_executable(compiler_name)

            # Validate the found path
            if path and self.validate_executable(path):
                compilers[compiler_name] = path
                # Try to get version for logging
                version = self.get_version(path)
                if version:
                    # Log first line of version output
                    version_line = version.split('\n')[0]
                    logger.info(f"Found compiler '{compiler_name}': {path} ({version_line})")
                else:
                    logger.info(f"Found compiler '{compiler_name}': {path}")
            else:
                compilers[compiler_name] = None
                logger.debug(f"Compiler '{compiler_name}' not found")

        found_count = sum(1 for v in compilers.values() if v is not None)
        logger.info(f"Discovered {found_count}/{len(compilers)} compilers")

        return compilers

    def discover_build_tools(self) -> Dict[str, Optional[str]]:
        """Discover available build tools on the system.

        Searches for common build tools, package managers, and version control
        systems in PATH and custom paths. Returns a dictionary mapping tool
        names to their absolute paths.

        Returns:
            Dictionary mapping tool name to absolute path or None if not found.
            Example:
            {
                "git": "/usr/bin/git",
                "make": "/usr/bin/make",
                "cmake": "/usr/local/bin/cmake",
                "cargo": "/Users/user/.cargo/bin/cargo",
                "npm": "/usr/local/bin/npm",
                "yarn": None,  # Not found
                "pip": "/usr/bin/pip",
                "uv": "/usr/local/bin/uv"
            }
        """
        # Define build tools to search for
        tool_names = [
            "git",
            "make",
            "cmake",
            "cargo",
            "npm",
            "yarn",
            "pip",
            "uv",
        ]

        # Add Windows-specific tools if on Windows
        if platform.system() == "Windows":
            tool_names.extend(["nmake", "nmake.exe", "msbuild", "msbuild.exe"])

        build_tools: Dict[str, Optional[str]] = {}

        logger.debug("Discovering build tools...")

        for tool_name in tool_names:
            path = self.find_executable(tool_name)

            # Validate the found path
            if path and self.validate_executable(path):
                build_tools[tool_name] = path
                # Try to get version for logging
                version = self.get_version(path)
                if version:
                    # Log first line of version output
                    version_line = version.split('\n')[0]
                    logger.info(f"Found build tool '{tool_name}': {path} ({version_line})")
                else:
                    logger.info(f"Found build tool '{tool_name}': {path}")
            else:
                build_tools[tool_name] = None
                logger.debug(f"Build tool '{tool_name}' not found")

        found_count = sum(1 for v in build_tools.values() if v is not None)
        logger.info(f"Discovered {found_count}/{len(build_tools)} build tools")

        return build_tools
def __init__(
self,
config: Optional[Dict] = None,
timeout: int = 5,
project_root: Optional[Path] = None,
):
"""Initialize tool discovery system.
Args:
config: Optional configuration dictionary containing:
- custom_paths: List of custom paths to search
- timeout: Override default timeout
- project_root: Project root directory for local tool discovery
timeout: Default timeout in seconds for subprocess operations
project_root: Optional project root directory for local tool discovery
"""
self.timeout = timeout
self.custom_paths: List[str] = []
self.project_root = project_root
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
# Set project root if specified
if "project_root" in config:
self.project_root = Path(config["project_root"])
logger.debug(f"Using project root: {self.project_root}")
logger.debug(
f"ToolDiscovery initialized with timeout={self.timeout}s, "
f"custom_paths={len(self.custom_paths)}, "
f"project_root={self.project_root}"
)
def _find_project_local_paths(self, project_root: Optional[Path] = None) -> List[Path]:
"""Find project-local paths for tool discovery.
Searches for common project-local tool directories such as:
- node_modules/.bin (JavaScript/TypeScript)
- .venv/bin or venv/bin (Python virtual environments)
- bin/ (generic project binaries)
Args:
project_root: Optional project root to override instance project_root
Returns:
List of paths to search for project-local tools
"""
local_paths: List[Path] = []
root = project_root or self.project_root
if not root:
return local_paths
# Check for node_modules/.bin (JavaScript/TypeScript LSPs)
node_bin = root / "node_modules" / ".bin"
if node_bin.exists() and node_bin.is_dir():
local_paths.append(node_bin)
logger.debug(f"Found node_modules/.bin at {node_bin}")
# Check for Python virtual environments
for venv_name in [".venv", "venv", "env"]:
venv_path = root / venv_name
if venv_path.exists() and venv_path.is_dir():
# Unix-like systems use bin/
venv_bin = venv_path / "bin"
if venv_bin.exists() and venv_bin.is_dir():
local_paths.append(venv_bin)
logger.debug(f"Found Python venv bin at {venv_bin}")
# Windows uses Scripts/
venv_scripts = venv_path / "Scripts"
if venv_scripts.exists() and venv_scripts.is_dir():
local_paths.append(venv_scripts)
logger.debug(f"Found Python venv Scripts at {venv_scripts}")
# Check for generic bin/ directory
bin_path = root / "bin"
if bin_path.exists() and bin_path.is_dir():
local_paths.append(bin_path)
logger.debug(f"Found project bin at {bin_path}")
return local_paths
def find_lsp_executable(
self,
language_name: str,
lsp_executable: str,
project_root: Optional[Path] = None,
) -> Optional[str]:
"""Find LSP server executable for a specific language.
Searches for the LSP executable in the following order:
1. Project-local paths (node_modules/.bin, .venv/bin, etc.)
2. Custom paths configured in ToolDiscovery
3. System PATH
Args:
language_name: Name of the language (used for logging)
lsp_executable: Name of the LSP executable to find
project_root: Optional project root to override instance project_root
Returns:
Absolute path to LSP executable if found, None otherwise
"""
# First, try project-local paths
local_paths = self._find_project_local_paths(project_root)
for local_path in local_paths:
potential_path = local_path / lsp_executable
# Handle Windows executable extensions
if platform.system() == "Windows":
for ext in ["", ".exe", ".bat", ".cmd"]:
candidate = Path(str(potential_path) + ext)
if candidate.exists() and self.validate_executable(str(candidate)):
logger.info(
f"Found LSP '{lsp_executable}' for {language_name} "
f"in project-local path: {candidate}"
)
return str(candidate.resolve())
else:
if potential_path.exists() and self.validate_executable(str(potential_path)):
logger.info(
f"Found LSP '{lsp_executable}' for {language_name} "
f"in project-local path: {potential_path}"
)
return str(potential_path.resolve())
# Fall back to global search (custom paths + system PATH)
result = self.find_executable(lsp_executable)
if result:
logger.info(
f"Found LSP '{lsp_executable}' for {language_name} in global PATH: {result}"
)
else:
logger.warning(
f"LSP server '{lsp_executable}' for {language_name} not found"
)
return result
def discover_lsp_servers(
self,
language_config,
project_root: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
"""Discover LSP servers for languages in configuration.
Searches for LSP server executables for each language that has an
LSP configuration. Returns a dictionary mapping language names to
their LSP executable paths (or None if not found).
The search order is:
1. Project-local paths (node_modules/.bin, .venv/bin, etc.)
2. Custom paths configured in ToolDiscovery
3. System PATH
Args:
language_config: LanguageSupportDatabaseConfig containing language
definitions with LSP configurations
project_root: Optional project root to override instance project_root
Returns:
Dictionary mapping language name to absolute LSP path or None.
Example:
{
"python": "/usr/local/bin/pyright-langserver",
"rust": "/usr/bin/rust-analyzer",
"typescript": None,  # Not found
}
Raises:
ValueError: If language_config is None or invalid
"""
if language_config is None:
raise ValueError("language_config cannot be None")
if not hasattr(language_config, "languages"):
raise ValueError(
"language_config must have 'languages' attribute "
"(LanguageSupportDatabaseConfig)"
)
lsp_paths: Dict[str, Optional[str]] = {}
found_count = 0
missing_count = 0
logger.info(
f"Discovering LSP servers for {len(language_config.languages)} languages"
)
for language in language_config.languages:
language_name = language.name
# Skip languages without LSP configuration
if language.lsp is None:
logger.debug(f"Language '{language_name}' has no LSP configuration")
continue
lsp_executable = language.lsp.executable
# Find the LSP executable
lsp_path = self.find_lsp_executable(
language_name, lsp_executable, project_root
)
if lsp_path:
# Validate the discovered path
if self.validate_executable(lsp_path):
lsp_paths[language_name] = lsp_path
found_count += 1
else:
logger.warning(
f"Found '{lsp_executable}' for {language_name} but it's not executable: {lsp_path}"
)
lsp_paths[language_name] = None
missing_count += 1
else:
lsp_paths[language_name] = None
missing_count += 1
logger.info(
f"LSP discovery complete: {found_count} found, {missing_count} missing"
)
if missing_count > 0:
missing_languages = [
lang for lang, path in lsp_paths.items() if path is None
]
logger.warning(
f"Missing LSP servers for: {', '.join(missing_languages)}"
)
return lsp_paths
