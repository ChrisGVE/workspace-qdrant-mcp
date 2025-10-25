"""
OS-Standard Directory Utilities

This module provides OS-standard directory paths following platform conventions:
- XDG Base Directory Specification for Linux/Unix
- macOS standard directories (~/Library/...)
- Windows standard directories (%APPDATA%, %LOCALAPPDATA%)

Supported directory types:
- config: Configuration files
- data: Application data files
- cache: Temporary cache files
- state: State files (databases, etc.)
- logs: Log files

Example:
    ```python
    from workspace_qdrant_mcp.utils.os_directories import OSDirectories

    os_dirs = OSDirectories()

    # Get standard directories
    config_dir = os_dirs.get_config_dir()
    cache_dir = os_dirs.get_cache_dir()
    state_dir = os_dirs.get_state_dir()
    log_dir = os_dirs.get_log_dir()

    # Create directories automatically
    os_dirs.ensure_directories()

    # Get application-specific paths
    db_path = os_dirs.get_state_file("workspace_state.db")
    log_path = os_dirs.get_log_file("workspace.log")
    ```
"""

import os
import platform
from pathlib import Path


class OSDirectories:
    """OS-standard directory manager following platform conventions."""

    APP_NAME = "workspace-qdrant"

    def __init__(self, app_name: str | None = None):
        """Initialize OS directories manager.

        Args:
            app_name: Application name for directory structure (default: workspace-qdrant)
        """
        self.app_name = app_name or self.APP_NAME
        self.system = platform.system().lower()

    def get_config_dir(self) -> Path:
        """Get OS-standard configuration directory.

        Returns:
            Path to configuration directory:
                - Linux/Unix: $XDG_CONFIG_HOME/workspace-qdrant or ~/.config/workspace-qdrant
                - macOS: ~/Library/Application Support/workspace-qdrant
                - Windows: %APPDATA%/workspace-qdrant
        """
        if self.system == 'darwin':  # macOS
            return Path.home() / 'Library' / 'Application Support' / self.app_name
        elif self.system == 'windows':
            appdata = os.environ.get('APPDATA')
            if appdata:
                return Path(appdata) / self.app_name
            else:
                # Fallback if APPDATA is not set
                return Path.home() / 'AppData' / 'Roaming' / self.app_name
        else:  # Linux/Unix and other Unix-like systems
            # Check XDG_CONFIG_HOME first
            xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
            if xdg_config_home:
                return Path(xdg_config_home) / self.app_name
            else:
                return Path.home() / '.config' / self.app_name

    def get_data_dir(self) -> Path:
        """Get OS-standard data directory.

        Returns:
            Path to data directory:
                - Linux/Unix: $XDG_DATA_HOME/workspace-qdrant or ~/.local/share/workspace-qdrant
                - macOS: ~/Library/Application Support/workspace-qdrant (same as config)
                - Windows: %LOCALAPPDATA%/workspace-qdrant
        """
        if self.system == 'darwin':  # macOS
            # On macOS, data and config are typically in the same location
            return Path.home() / 'Library' / 'Application Support' / self.app_name
        elif self.system == 'windows':
            localappdata = os.environ.get('LOCALAPPDATA')
            if localappdata:
                return Path(localappdata) / self.app_name
            else:
                # Fallback if LOCALAPPDATA is not set
                return Path.home() / 'AppData' / 'Local' / self.app_name
        else:  # Linux/Unix
            # Check XDG_DATA_HOME first
            xdg_data_home = os.environ.get('XDG_DATA_HOME')
            if xdg_data_home:
                return Path(xdg_data_home) / self.app_name
            else:
                return Path.home() / '.local' / 'share' / self.app_name

    def get_cache_dir(self) -> Path:
        """Get OS-standard cache directory.

        Returns:
            Path to cache directory:
                - Linux/Unix: $XDG_CACHE_HOME/workspace-qdrant or ~/.cache/workspace-qdrant
                - macOS: ~/Library/Caches/workspace-qdrant
                - Windows: %LOCALAPPDATA%/workspace-qdrant/cache
        """
        if self.system == 'darwin':  # macOS
            return Path.home() / 'Library' / 'Caches' / self.app_name
        elif self.system == 'windows':
            localappdata = os.environ.get('LOCALAPPDATA')
            if localappdata:
                return Path(localappdata) / self.app_name / 'cache'
            else:
                # Fallback if LOCALAPPDATA is not set
                return Path.home() / 'AppData' / 'Local' / self.app_name / 'cache'
        else:  # Linux/Unix
            # Check XDG_CACHE_HOME first
            xdg_cache_home = os.environ.get('XDG_CACHE_HOME')
            if xdg_cache_home:
                return Path(xdg_cache_home) / self.app_name
            else:
                return Path.home() / '.cache' / self.app_name

    def get_state_dir(self) -> Path:
        """Get OS-standard state directory.

        Returns:
            Path to state directory:
                - Linux/Unix: $XDG_STATE_HOME/workspace-qdrant or ~/.local/state/workspace-qdrant
                - macOS: ~/Library/Application Support/workspace-qdrant (same as data)
                - Windows: %LOCALAPPDATA%/workspace-qdrant/state
        """
        if self.system == 'darwin':  # macOS
            # On macOS, state is typically stored with application data
            return Path.home() / 'Library' / 'Application Support' / self.app_name
        elif self.system == 'windows':
            localappdata = os.environ.get('LOCALAPPDATA')
            if localappdata:
                return Path(localappdata) / self.app_name / 'state'
            else:
                # Fallback if LOCALAPPDATA is not set
                return Path.home() / 'AppData' / 'Local' / self.app_name / 'state'
        else:  # Linux/Unix
            # Check XDG_STATE_HOME first (newer XDG spec)
            xdg_state_home = os.environ.get('XDG_STATE_HOME')
            if xdg_state_home:
                return Path(xdg_state_home) / self.app_name
            else:
                return Path.home() / '.local' / 'state' / self.app_name

    def get_log_dir(self) -> Path:
        """Get OS-standard log directory.

        Returns:
            Path to log directory:
                - Linux/Unix: ~/.local/state/workspace-qdrant/logs (following XDG state spec)
                - macOS: ~/Library/Logs/workspace-qdrant
                - Windows: %LOCALAPPDATA%/workspace-qdrant/logs
        """
        if self.system == 'darwin':  # macOS
            return Path.home() / 'Library' / 'Logs' / self.app_name
        elif self.system == 'windows':
            localappdata = os.environ.get('LOCALAPPDATA')
            if localappdata:
                return Path(localappdata) / self.app_name / 'logs'
            else:
                # Fallback if LOCALAPPDATA is not set
                return Path.home() / 'AppData' / 'Local' / self.app_name / 'logs'
        else:  # Linux/Unix
            # Logs are considered state files under XDG spec
            return self.get_state_dir() / 'logs'

    def get_config_file(self, filename: str) -> Path:
        """Get path for a configuration file.

        Args:
            filename: Configuration file name

        Returns:
            Path to configuration file in OS-standard config directory
        """
        return self.get_config_dir() / filename

    def get_data_file(self, filename: str) -> Path:
        """Get path for a data file.

        Args:
            filename: Data file name

        Returns:
            Path to data file in OS-standard data directory
        """
        return self.get_data_dir() / filename

    def get_cache_file(self, filename: str) -> Path:
        """Get path for a cache file.

        Args:
            filename: Cache file name

        Returns:
            Path to cache file in OS-standard cache directory
        """
        return self.get_cache_dir() / filename

    def get_state_file(self, filename: str) -> Path:
        """Get path for a state file.

        Args:
            filename: State file name (e.g., database file)

        Returns:
            Path to state file in OS-standard state directory
        """
        return self.get_state_dir() / filename

    def get_log_file(self, filename: str) -> Path:
        """Get path for a log file.

        Args:
            filename: Log file name

        Returns:
            Path to log file in OS-standard log directory
        """
        return self.get_log_dir() / filename

    def ensure_directories(self) -> None:
        """Create all OS-standard directories if they don't exist.

        Creates all standard directories with appropriate permissions.
        """
        directories = [
            self.get_config_dir(),
            self.get_data_dir(),
            self.get_cache_dir(),
            self.get_state_dir(),
            self.get_log_dir()
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

            # Set restrictive permissions for data and state directories
            if directory in [self.get_data_dir(), self.get_state_dir()]:
                try:
                    # Make directories readable only by owner (700)
                    if self.system != 'windows':  # Windows doesn't support chmod
                        os.chmod(directory, 0o700)
                except (OSError, PermissionError):
                    # Ignore permission errors if we can't set them
                    pass

    def get_runtime_dir(self) -> Path | None:
        """Get OS-standard runtime directory (for sockets, etc.).

        Returns:
            Path to runtime directory:
                - Linux/Unix: $XDG_RUNTIME_DIR/workspace-qdrant or None
                - macOS: None (no standard runtime directory)
                - Windows: None (no standard runtime directory)
        """
        if self.system != 'linux':
            return None

        xdg_runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
        if xdg_runtime_dir:
            return Path(xdg_runtime_dir) / self.app_name
        else:
            return None

    def migrate_from_legacy_paths(self, legacy_paths: dict[str, Path]) -> dict[str, list[Path]]:
        """Migrate files from legacy paths to OS-standard locations.

        Args:
            legacy_paths: Dictionary mapping file types to legacy paths
                         e.g., {"config": Path("./config.yaml"), "state": Path("./workspace_state.db")}

        Returns:
            Dictionary mapping file types to lists of migrated files

        Note:
            This method only identifies files that should be migrated.
            Actual migration should be performed by the calling code.
        """
        migration_plan = {}

        for file_type, legacy_path in legacy_paths.items():
            if not legacy_path.exists():
                continue

            if file_type == 'config':
                new_path = self.get_config_file(legacy_path.name)
            elif file_type == 'data':
                new_path = self.get_data_file(legacy_path.name)
            elif file_type == 'cache':
                new_path = self.get_cache_file(legacy_path.name)
            elif file_type == 'state':
                new_path = self.get_state_file(legacy_path.name)
            elif file_type == 'logs':
                new_path = self.get_log_file(legacy_path.name)
            else:
                # Unknown file type, skip
                continue

            if file_type not in migration_plan:
                migration_plan[file_type] = []

            migration_plan[file_type].append({
                'from': legacy_path,
                'to': new_path,
                'exists': legacy_path.exists(),
                'target_exists': new_path.exists()
            })

        return migration_plan

    def get_directory_info(self) -> dict[str, str]:
        """Get information about all OS-standard directories.

        Returns:
            Dictionary with directory type as key and path as string value
        """
        return {
            'system': self.system,
            'app_name': self.app_name,
            'config': str(self.get_config_dir()),
            'data': str(self.get_data_dir()),
            'cache': str(self.get_cache_dir()),
            'state': str(self.get_state_dir()),
            'logs': str(self.get_log_dir()),
            'runtime': str(self.get_runtime_dir()) if self.get_runtime_dir() else None
        }
