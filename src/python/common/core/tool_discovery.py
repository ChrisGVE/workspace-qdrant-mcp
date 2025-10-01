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
   1 
   2     def __init__(
   3         self,
   4         config: Optional[Dict] = None,
   5         timeout: int = 5,
   6         project_root: Optional[Path] = None,
   7     ):
   8         """Initialize tool discovery system.
   9 
  10         Args:
  11             config: Optional configuration dictionary containing:
  12                 - custom_paths: List of custom paths to search
  13                 - timeout: Override default timeout
  14                 - project_root: Project root directory for local tool discovery
  15             timeout: Default timeout in seconds for subprocess operations
  16             project_root: Optional project root directory for local tool discovery
  17         """
  18         self.timeout = timeout
  19         self.custom_paths: List[str] = []
  20         self.project_root = project_root
  21 
  22         if config:
  23             # Extract custom paths from config
  24             if "custom_paths" in config:
  25                 self.custom_paths = config["custom_paths"]
  26                 logger.debug(
  27                     f"Initialized with {len(self.custom_paths)} custom paths"
  28                 )
  29 
  30             # Override timeout if specified
  31             if "timeout" in config:
  32                 self.timeout = config["timeout"]
  33                 logger.debug(f"Using custom timeout: {self.timeout}s")
  34 
  35             # Set project root if specified
  36             if "project_root" in config:
  37                 self.project_root = Path(config["project_root"])
  38                 logger.debug(f"Using project root: {self.project_root}")
  39 
  40         logger.debug(
  41             f"ToolDiscovery initialized with timeout={self.timeout}s, "
  42             f"custom_paths={len(self.custom_paths)}, "
  43             f"project_root={self.project_root}"
  44         )
  45 
  46     def _find_project_local_paths(self, project_root: Optional[Path] = None) -> List[Path]:
  47         """Find project-local paths for tool discovery.
  48 
  49         Searches for common project-local tool directories such as:
  50         - node_modules/.bin (JavaScript/TypeScript)
  51         - .venv/bin or venv/bin (Python virtual environments)
  52         - bin/ (generic project binaries)
  53 
  54         Args:
  55             project_root: Optional project root to override instance project_root
  56 
  57         Returns:
  58             List of paths to search for project-local tools
  59         """
  60         local_paths: List[Path] = []
  61         root = project_root or self.project_root
  62 
  63         if not root:
  64             return local_paths
  65 
  66         # Check for node_modules/.bin (JavaScript/TypeScript LSPs)
  67         node_bin = root / "node_modules" / ".bin"
  68         if node_bin.exists() and node_bin.is_dir():
  69             local_paths.append(node_bin)
  70             logger.debug(f"Found node_modules/.bin at {node_bin}")
  71 
  72         # Check for Python virtual environments
  73         for venv_name in [".venv", "venv", "env"]:
  74             venv_path = root / venv_name
  75             if venv_path.exists() and venv_path.is_dir():
  76                 # Unix-like systems use bin/
  77                 venv_bin = venv_path / "bin"
  78                 if venv_bin.exists() and venv_bin.is_dir():
  79                     local_paths.append(venv_bin)
  80                     logger.debug(f"Found Python venv bin at {venv_bin}")
  81 
  82                 # Windows uses Scripts/
  83                 venv_scripts = venv_path / "Scripts"
  84                 if venv_scripts.exists() and venv_scripts.is_dir():
  85                     local_paths.append(venv_scripts)
  86                     logger.debug(f"Found Python venv Scripts at {venv_scripts}")
  87 
  88         # Check for generic bin/ directory
  89         bin_path = root / "bin"
  90         if bin_path.exists() and bin_path.is_dir():
  91             local_paths.append(bin_path)
  92             logger.debug(f"Found project bin at {bin_path}")
  93 
  94         return local_paths
  95 
  96     def find_lsp_executable(
  97         self,
  98         language_name: str,
  99         lsp_executable: str,
 100         project_root: Optional[Path] = None,
 101     ) -> Optional[str]:
 102         """Find LSP server executable for a specific language.
 103 
 104         Searches for the LSP executable in the following order:
 105         1. Project-local paths (node_modules/.bin, .venv/bin, etc.)
 106         2. Custom paths configured in ToolDiscovery
 107         3. System PATH
 108 
 109         Args:
 110             language_name: Name of the language (used for logging)
 111             lsp_executable: Name of the LSP executable to find
 112             project_root: Optional project root to override instance project_root
 113 
 114         Returns:
 115             Absolute path to LSP executable if found, None otherwise
 116         """
 117         # First, try project-local paths
 118         local_paths = self._find_project_local_paths(project_root)
 119         for local_path in local_paths:
 120             potential_path = local_path / lsp_executable
 121 
 122             # Handle Windows executable extensions
 123             if platform.system() == "Windows":
 124                 for ext in ["", ".exe", ".bat", ".cmd"]:
 125                     candidate = Path(str(potential_path) + ext)
 126                     if candidate.exists() and self.validate_executable(str(candidate)):
 127                         logger.info(
 128                             f"Found LSP '{lsp_executable}' for {language_name} "
 129                             f"in project-local path: {candidate}"
 130                         )
 131                         return str(candidate.resolve())
 132             else:
 133                 if potential_path.exists() and self.validate_executable(str(potential_path)):
 134                     logger.info(
 135                         f"Found LSP '{lsp_executable}' for {language_name} "
 136                         f"in project-local path: {potential_path}"
 137                     )
 138                     return str(potential_path.resolve())
 139 
 140         # Fall back to global search (custom paths + system PATH)
 141         result = self.find_executable(lsp_executable)
 142 
 143         if result:
 144             logger.info(
 145                 f"Found LSP '{lsp_executable}' for {language_name} in global PATH: {result}"
 146             )
 147         else:
 148             logger.warning(
 149                 f"LSP server '{lsp_executable}' for {language_name} not found"
 150             )
 151 
 152         return result
 153 
 154     def discover_lsp_servers(
 155         self,
 156         language_config,
 157         project_root: Optional[Path] = None,
 158     ) -> Dict[str, Optional[str]]:
 159         """Discover LSP servers for languages in configuration.
 160 
 161         Searches for LSP server executables for each language that has an
 162         LSP configuration. Returns a dictionary mapping language names to
 163         their LSP executable paths (or None if not found).
 164 
 165         The search order is:
 166         1. Project-local paths (node_modules/.bin, .venv/bin, etc.)
 167         2. Custom paths configured in ToolDiscovery
 168         3. System PATH
 169 
 170         Args:
 171             language_config: LanguageSupportDatabaseConfig containing language
 172                 definitions with LSP configurations
 173             project_root: Optional project root to override instance project_root
 174 
 175         Returns:
 176             Dictionary mapping language name to absolute LSP path or None.
 177             Example:
 178                 {
 179                     "python": "/usr/local/bin/pyright-langserver",
 180                     "rust": "/usr/bin/rust-analyzer",
 181                     "typescript": None,  # Not found
 182                 }
 183 
 184         Raises:
 185             ValueError: If language_config is None or invalid
 186         """
 187         if language_config is None:
 188             raise ValueError("language_config cannot be None")
 189 
 190         if not hasattr(language_config, "languages"):
 191             raise ValueError(
 192                 "language_config must have 'languages' attribute "
 193                 "(LanguageSupportDatabaseConfig)"
 194             )
 195 
 196         lsp_paths: Dict[str, Optional[str]] = {}
 197         found_count = 0
 198         missing_count = 0
 199 
 200         logger.info(
 201             f"Discovering LSP servers for {len(language_config.languages)} languages"
 202         )
 203 
 204         for language in language_config.languages:
 205             language_name = language.name
 206 
 207             # Skip languages without LSP configuration
 208             if language.lsp is None:
 209                 logger.debug(f"Language '{language_name}' has no LSP configuration")
 210                 continue
 211 
 212             lsp_executable = language.lsp.executable
 213 
 214             # Find the LSP executable
 215             lsp_path = self.find_lsp_executable(
 216                 language_name, lsp_executable, project_root
 217             )
 218 
 219             if lsp_path:
 220                 # Validate the discovered path
 221                 if self.validate_executable(lsp_path):
 222                     lsp_paths[language_name] = lsp_path
 223                     found_count += 1
 224                 else:
 225                     logger.warning(
 226                         f"Found '{lsp_executable}' for {language_name} but it's not executable: {lsp_path}"
 227                     )
 228                     lsp_paths[language_name] = None
 229                     missing_count += 1
 230             else:
 231                 lsp_paths[language_name] = None
 232                 missing_count += 1
 233 
 234         logger.info(
 235             f"LSP discovery complete: {found_count} found, {missing_count} missing"
 236         )
 237 
 238         if missing_count > 0:
 239             missing_languages = [
 240                 lang for lang, path in lsp_paths.items() if path is None
 241             ]
 242             logger.warning(
 243                 f"Missing LSP servers for: {', '.join(missing_languages)}"
 244             )
 245 
 246         return lsp_paths
