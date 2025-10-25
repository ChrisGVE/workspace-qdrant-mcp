"""
Comprehensive unit tests for OS-standard directory utilities.

This module provides 100% test coverage for the OS directories system,
including all platform-specific directory paths and operations.

Test coverage:
- OSDirectories: all directory methods for different platforms
- Platform-specific path generation (Linux, macOS, Windows)
- Environment variable handling and fallbacks
- Directory creation and permission setting
- Migration planning and file operations
"""

import os
import platform

# Ensure proper imports from the project structure
import sys
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))
from common.utils.os_directories import OSDirectories


class TestOSDirectories:
    """Comprehensive tests for OSDirectories class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default_app_name(self):
        """Test initialization with default app name."""
        os_dirs = OSDirectories()

        assert os_dirs.app_name == "workspace-qdrant"
        assert os_dirs.system == platform.system().lower()

    def test_init_custom_app_name(self):
        """Test initialization with custom app name."""
        os_dirs = OSDirectories("custom-app")

        assert os_dirs.app_name == "custom-app"

    def test_init_system_detection(self):
        """Test system detection during initialization."""
        with patch('platform.system', return_value='Linux'):
            os_dirs = OSDirectories()
            assert os_dirs.system == 'linux'

        with patch('platform.system', return_value='Darwin'):
            os_dirs = OSDirectories()
            assert os_dirs.system == 'darwin'

        with patch('platform.system', return_value='Windows'):
            os_dirs = OSDirectories()
            assert os_dirs.system == 'windows'

    @patch('platform.system', return_value='Darwin')
    def test_get_config_dir_macos(self, mock_system):
        """Test config directory on macOS."""
        os_dirs = OSDirectories("test-app")

        with patch('pathlib.Path.home', return_value=Path('/Users/testuser')):
            config_dir = os_dirs.get_config_dir()
            expected = Path('/Users/testuser/Library/Application Support/test-app')
            assert config_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_config_dir_windows_with_appdata(self, mock_system):
        """Test config directory on Windows with APPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'APPDATA': 'C:\\Users\\testuser\\AppData\\Roaming'}):
            config_dir = os_dirs.get_config_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Roaming\\test-app')
            assert config_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_config_dir_windows_no_appdata(self, mock_system):
        """Test config directory on Windows without APPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('C:\\Users\\testuser')):
            config_dir = os_dirs.get_config_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Roaming\\test-app')
            assert config_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_config_dir_linux_with_xdg(self, mock_system):
        """Test config directory on Linux with XDG_CONFIG_HOME."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'XDG_CONFIG_HOME': '/home/testuser/.config'}):
            config_dir = os_dirs.get_config_dir()
            expected = Path('/home/testuser/.config/test-app')
            assert config_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_config_dir_linux_no_xdg(self, mock_system):
        """Test config directory on Linux without XDG_CONFIG_HOME."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('/home/testuser')):
            config_dir = os_dirs.get_config_dir()
            expected = Path('/home/testuser/.config/test-app')
            assert config_dir == expected

    @patch('platform.system', return_value='Darwin')
    def test_get_data_dir_macos(self, mock_system):
        """Test data directory on macOS."""
        os_dirs = OSDirectories("test-app")

        with patch('pathlib.Path.home', return_value=Path('/Users/testuser')):
            data_dir = os_dirs.get_data_dir()
            expected = Path('/Users/testuser/Library/Application Support/test-app')
            assert data_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_data_dir_windows_with_localappdata(self, mock_system):
        """Test data directory on Windows with LOCALAPPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'LOCALAPPDATA': 'C:\\Users\\testuser\\AppData\\Local'}):
            data_dir = os_dirs.get_data_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Local\\test-app')
            assert data_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_data_dir_windows_no_localappdata(self, mock_system):
        """Test data directory on Windows without LOCALAPPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('C:\\Users\\testuser')):
            data_dir = os_dirs.get_data_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Local\\test-app')
            assert data_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_data_dir_linux_with_xdg(self, mock_system):
        """Test data directory on Linux with XDG_DATA_HOME."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'XDG_DATA_HOME': '/home/testuser/.local/share'}):
            data_dir = os_dirs.get_data_dir()
            expected = Path('/home/testuser/.local/share/test-app')
            assert data_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_data_dir_linux_no_xdg(self, mock_system):
        """Test data directory on Linux without XDG_DATA_HOME."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('/home/testuser')):
            data_dir = os_dirs.get_data_dir()
            expected = Path('/home/testuser/.local/share/test-app')
            assert data_dir == expected

    @patch('platform.system', return_value='Darwin')
    def test_get_cache_dir_macos(self, mock_system):
        """Test cache directory on macOS."""
        os_dirs = OSDirectories("test-app")

        with patch('pathlib.Path.home', return_value=Path('/Users/testuser')):
            cache_dir = os_dirs.get_cache_dir()
            expected = Path('/Users/testuser/Library/Caches/test-app')
            assert cache_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_cache_dir_windows_with_localappdata(self, mock_system):
        """Test cache directory on Windows with LOCALAPPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'LOCALAPPDATA': 'C:\\Users\\testuser\\AppData\\Local'}):
            cache_dir = os_dirs.get_cache_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Local\\test-app\\cache')
            assert cache_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_cache_dir_windows_no_localappdata(self, mock_system):
        """Test cache directory on Windows without LOCALAPPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('C:\\Users\\testuser')):
            cache_dir = os_dirs.get_cache_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Local\\test-app\\cache')
            assert cache_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_cache_dir_linux_with_xdg(self, mock_system):
        """Test cache directory on Linux with XDG_CACHE_HOME."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'XDG_CACHE_HOME': '/home/testuser/.cache'}):
            cache_dir = os_dirs.get_cache_dir()
            expected = Path('/home/testuser/.cache/test-app')
            assert cache_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_cache_dir_linux_no_xdg(self, mock_system):
        """Test cache directory on Linux without XDG_CACHE_HOME."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('/home/testuser')):
            cache_dir = os_dirs.get_cache_dir()
            expected = Path('/home/testuser/.cache/test-app')
            assert cache_dir == expected

    @patch('platform.system', return_value='Darwin')
    def test_get_state_dir_macos(self, mock_system):
        """Test state directory on macOS."""
        os_dirs = OSDirectories("test-app")

        with patch('pathlib.Path.home', return_value=Path('/Users/testuser')):
            state_dir = os_dirs.get_state_dir()
            expected = Path('/Users/testuser/Library/Application Support/test-app')
            assert state_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_state_dir_windows_with_localappdata(self, mock_system):
        """Test state directory on Windows with LOCALAPPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'LOCALAPPDATA': 'C:\\Users\\testuser\\AppData\\Local'}):
            state_dir = os_dirs.get_state_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Local\\test-app\\state')
            assert state_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_state_dir_windows_no_localappdata(self, mock_system):
        """Test state directory on Windows without LOCALAPPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('C:\\Users\\testuser')):
            state_dir = os_dirs.get_state_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Local\\test-app\\state')
            assert state_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_state_dir_linux_with_xdg(self, mock_system):
        """Test state directory on Linux with XDG_STATE_HOME."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'XDG_STATE_HOME': '/home/testuser/.local/state'}):
            state_dir = os_dirs.get_state_dir()
            expected = Path('/home/testuser/.local/state/test-app')
            assert state_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_state_dir_linux_no_xdg(self, mock_system):
        """Test state directory on Linux without XDG_STATE_HOME."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('/home/testuser')):
            state_dir = os_dirs.get_state_dir()
            expected = Path('/home/testuser/.local/state/test-app')
            assert state_dir == expected

    @patch('platform.system', return_value='Darwin')
    def test_get_log_dir_macos(self, mock_system):
        """Test log directory on macOS."""
        os_dirs = OSDirectories("test-app")

        with patch('pathlib.Path.home', return_value=Path('/Users/testuser')):
            log_dir = os_dirs.get_log_dir()
            expected = Path('/Users/testuser/Library/Logs/test-app')
            assert log_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_log_dir_windows_with_localappdata(self, mock_system):
        """Test log directory on Windows with LOCALAPPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'LOCALAPPDATA': 'C:\\Users\\testuser\\AppData\\Local'}):
            log_dir = os_dirs.get_log_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Local\\test-app\\logs')
            assert log_dir == expected

    @patch('platform.system', return_value='Windows')
    def test_get_log_dir_windows_no_localappdata(self, mock_system):
        """Test log directory on Windows without LOCALAPPDATA."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('C:\\Users\\testuser')):
            log_dir = os_dirs.get_log_dir()
            expected = Path('C:\\Users\\testuser\\AppData\\Local\\test-app\\logs')
            assert log_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_log_dir_linux(self, mock_system):
        """Test log directory on Linux (uses state dir + logs)."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('/home/testuser')):
            log_dir = os_dirs.get_log_dir()
            expected = Path('/home/testuser/.local/state/test-app/logs')
            assert log_dir == expected

    def test_get_config_file(self):
        """Test config file path generation."""
        os_dirs = OSDirectories("test-app")

        with patch.object(os_dirs, 'get_config_dir', return_value=Path('/config')):
            config_file = os_dirs.get_config_file("app.conf")
            assert config_file == Path('/config/app.conf')

    def test_get_data_file(self):
        """Test data file path generation."""
        os_dirs = OSDirectories("test-app")

        with patch.object(os_dirs, 'get_data_dir', return_value=Path('/data')):
            data_file = os_dirs.get_data_file("data.db")
            assert data_file == Path('/data/data.db')

    def test_get_cache_file(self):
        """Test cache file path generation."""
        os_dirs = OSDirectories("test-app")

        with patch.object(os_dirs, 'get_cache_dir', return_value=Path('/cache')):
            cache_file = os_dirs.get_cache_file("cache.tmp")
            assert cache_file == Path('/cache/cache.tmp')

    def test_get_state_file(self):
        """Test state file path generation."""
        os_dirs = OSDirectories("test-app")

        with patch.object(os_dirs, 'get_state_dir', return_value=Path('/state')):
            state_file = os_dirs.get_state_file("state.db")
            assert state_file == Path('/state/state.db')

    def test_get_log_file(self):
        """Test log file path generation."""
        os_dirs = OSDirectories("test-app")

        with patch.object(os_dirs, 'get_log_dir', return_value=Path('/logs')):
            log_file = os_dirs.get_log_file("app.log")
            assert log_file == Path('/logs/app.log')

    @patch('platform.system', return_value='Linux')
    def test_get_runtime_dir_linux_with_xdg(self, mock_system):
        """Test runtime directory on Linux with XDG_RUNTIME_DIR."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {'XDG_RUNTIME_DIR': '/run/user/1000'}):
            runtime_dir = os_dirs.get_runtime_dir()
            expected = Path('/run/user/1000/test-app')
            assert runtime_dir == expected

    @patch('platform.system', return_value='Linux')
    def test_get_runtime_dir_linux_no_xdg(self, mock_system):
        """Test runtime directory on Linux without XDG_RUNTIME_DIR."""
        os_dirs = OSDirectories("test-app")

        with patch.dict(os.environ, {}, clear=True):
            runtime_dir = os_dirs.get_runtime_dir()
            assert runtime_dir is None

    @patch('platform.system', return_value='Darwin')
    def test_get_runtime_dir_macos(self, mock_system):
        """Test runtime directory on macOS (should return None)."""
        os_dirs = OSDirectories("test-app")

        runtime_dir = os_dirs.get_runtime_dir()
        assert runtime_dir is None

    @patch('platform.system', return_value='Windows')
    def test_get_runtime_dir_windows(self, mock_system):
        """Test runtime directory on Windows (should return None)."""
        os_dirs = OSDirectories("test-app")

        runtime_dir = os_dirs.get_runtime_dir()
        assert runtime_dir is None

    def test_ensure_directories(self):
        """Test directory creation."""
        os_dirs = OSDirectories("test-app")

        mock_directories = [
            Path(self.temp_dir) / "config",
            Path(self.temp_dir) / "data",
            Path(self.temp_dir) / "cache",
            Path(self.temp_dir) / "state",
            Path(self.temp_dir) / "logs"
        ]

        with patch.object(os_dirs, 'get_config_dir', return_value=mock_directories[0]), \
             patch.object(os_dirs, 'get_data_dir', return_value=mock_directories[1]), \
             patch.object(os_dirs, 'get_cache_dir', return_value=mock_directories[2]), \
             patch.object(os_dirs, 'get_state_dir', return_value=mock_directories[3]), \
             patch.object(os_dirs, 'get_log_dir', return_value=mock_directories[4]):

            os_dirs.ensure_directories()

            # Check that all directories were created
            for directory in mock_directories:
                assert directory.exists()

    @patch('platform.system', return_value='Linux')
    def test_ensure_directories_permissions_linux(self, mock_system):
        """Test directory creation with permissions on Linux."""
        os_dirs = OSDirectories("test-app")

        data_dir = Path(self.temp_dir) / "data"
        state_dir = Path(self.temp_dir) / "state"
        other_dir = Path(self.temp_dir) / "cache"

        with patch.object(os_dirs, 'get_config_dir', return_value=other_dir), \
             patch.object(os_dirs, 'get_data_dir', return_value=data_dir), \
             patch.object(os_dirs, 'get_cache_dir', return_value=other_dir), \
             patch.object(os_dirs, 'get_state_dir', return_value=state_dir), \
             patch.object(os_dirs, 'get_log_dir', return_value=other_dir), \
             patch('os.chmod') as mock_chmod:

            os_dirs.ensure_directories()

            # Check that chmod was called for data and state directories
            expected_calls = [
                call(data_dir, 0o700),
                call(state_dir, 0o700)
            ]
            mock_chmod.assert_has_calls(expected_calls, any_order=True)

    @patch('platform.system', return_value='Windows')
    def test_ensure_directories_permissions_windows(self, mock_system):
        """Test directory creation without permissions on Windows."""
        os_dirs = OSDirectories("test-app")

        data_dir = Path(self.temp_dir) / "data"
        state_dir = Path(self.temp_dir) / "state"

        with patch.object(os_dirs, 'get_config_dir', return_value=Path(self.temp_dir) / "config"), \
             patch.object(os_dirs, 'get_data_dir', return_value=data_dir), \
             patch.object(os_dirs, 'get_cache_dir', return_value=Path(self.temp_dir) / "cache"), \
             patch.object(os_dirs, 'get_state_dir', return_value=state_dir), \
             patch.object(os_dirs, 'get_log_dir', return_value=Path(self.temp_dir) / "logs"), \
             patch('os.chmod') as mock_chmod:

            os_dirs.ensure_directories()

            # chmod should not be called on Windows
            mock_chmod.assert_not_called()

    def test_ensure_directories_permissions_error(self):
        """Test directory creation with permission errors."""
        os_dirs = OSDirectories("test-app")

        data_dir = Path(self.temp_dir) / "data"

        with patch.object(os_dirs, 'get_config_dir', return_value=Path(self.temp_dir) / "config"), \
             patch.object(os_dirs, 'get_data_dir', return_value=data_dir), \
             patch.object(os_dirs, 'get_cache_dir', return_value=Path(self.temp_dir) / "cache"), \
             patch.object(os_dirs, 'get_state_dir', return_value=Path(self.temp_dir) / "state"), \
             patch.object(os_dirs, 'get_log_dir', return_value=Path(self.temp_dir) / "logs"), \
             patch('os.chmod', side_effect=PermissionError("Permission denied")):

            # Should not raise exception even if chmod fails
            os_dirs.ensure_directories()

            # Directory should still be created
            assert data_dir.exists()

    def test_migrate_from_legacy_paths_config(self):
        """Test migration planning for config files."""
        os_dirs = OSDirectories("test-app")

        legacy_config = Path(self.temp_dir) / "config.yaml"
        legacy_config.touch()

        legacy_paths = {"config": legacy_config}

        with patch.object(os_dirs, 'get_config_file', return_value=Path(self.temp_dir) / "new_config.yaml"):
            migration_plan = os_dirs.migrate_from_legacy_paths(legacy_paths)

            assert "config" in migration_plan
            assert len(migration_plan["config"]) == 1
            assert migration_plan["config"][0]["from"] == legacy_config
            assert migration_plan["config"][0]["exists"]

    def test_migrate_from_legacy_paths_all_types(self):
        """Test migration planning for all file types."""
        os_dirs = OSDirectories("test-app")

        legacy_files = {
            "config": Path(self.temp_dir) / "config.yaml",
            "data": Path(self.temp_dir) / "data.db",
            "cache": Path(self.temp_dir) / "cache.tmp",
            "state": Path(self.temp_dir) / "state.db",
            "logs": Path(self.temp_dir) / "app.log"
        }

        # Create legacy files
        for file_path in legacy_files.values():
            file_path.touch()

        new_files = {
            "config": Path(self.temp_dir) / "new_config.yaml",
            "data": Path(self.temp_dir) / "new_data.db",
            "cache": Path(self.temp_dir) / "new_cache.tmp",
            "state": Path(self.temp_dir) / "new_state.db",
            "logs": Path(self.temp_dir) / "new_app.log"
        }

        with patch.object(os_dirs, 'get_config_file', return_value=new_files["config"]), \
             patch.object(os_dirs, 'get_data_file', return_value=new_files["data"]), \
             patch.object(os_dirs, 'get_cache_file', return_value=new_files["cache"]), \
             patch.object(os_dirs, 'get_state_file', return_value=new_files["state"]), \
             patch.object(os_dirs, 'get_log_file', return_value=new_files["logs"]):

            migration_plan = os_dirs.migrate_from_legacy_paths(legacy_files)

            # Check all file types are in migration plan
            for file_type in legacy_files.keys():
                assert file_type in migration_plan
                assert len(migration_plan[file_type]) == 1
                assert migration_plan[file_type][0]["exists"]

    def test_migrate_from_legacy_paths_nonexistent_file(self):
        """Test migration planning for non-existent legacy files."""
        os_dirs = OSDirectories("test-app")

        legacy_config = Path(self.temp_dir) / "nonexistent.yaml"
        legacy_paths = {"config": legacy_config}

        migration_plan = os_dirs.migrate_from_legacy_paths(legacy_paths)

        # Should not include non-existent files in migration plan
        assert len(migration_plan) == 0

    def test_migrate_from_legacy_paths_unknown_type(self):
        """Test migration planning for unknown file type."""
        os_dirs = OSDirectories("test-app")

        legacy_file = Path(self.temp_dir) / "unknown.file"
        legacy_file.touch()
        legacy_paths = {"unknown": legacy_file}

        migration_plan = os_dirs.migrate_from_legacy_paths(legacy_paths)

        # Should not include unknown file types in migration plan
        assert len(migration_plan) == 0

    def test_migrate_from_legacy_paths_target_exists(self):
        """Test migration planning when target file already exists."""
        os_dirs = OSDirectories("test-app")

        legacy_config = Path(self.temp_dir) / "config.yaml"
        legacy_config.touch()

        new_config = Path(self.temp_dir) / "new_config.yaml"
        new_config.touch()  # Target already exists

        legacy_paths = {"config": legacy_config}

        with patch.object(os_dirs, 'get_config_file', return_value=new_config):
            migration_plan = os_dirs.migrate_from_legacy_paths(legacy_paths)

            assert migration_plan["config"][0]["target_exists"]

    def test_get_directory_info(self):
        """Test directory information generation."""
        os_dirs = OSDirectories("test-app")

        mock_paths = {
            'config': '/home/user/.config/test-app',
            'data': '/home/user/.local/share/test-app',
            'cache': '/home/user/.cache/test-app',
            'state': '/home/user/.local/state/test-app',
            'logs': '/home/user/.local/state/test-app/logs',
            'runtime': '/run/user/1000/test-app'
        }

        with patch.object(os_dirs, 'get_config_dir', return_value=Path(mock_paths['config'])), \
             patch.object(os_dirs, 'get_data_dir', return_value=Path(mock_paths['data'])), \
             patch.object(os_dirs, 'get_cache_dir', return_value=Path(mock_paths['cache'])), \
             patch.object(os_dirs, 'get_state_dir', return_value=Path(mock_paths['state'])), \
             patch.object(os_dirs, 'get_log_dir', return_value=Path(mock_paths['logs'])), \
             patch.object(os_dirs, 'get_runtime_dir', return_value=Path(mock_paths['runtime'])):

            info = os_dirs.get_directory_info()

            assert info['system'] == os_dirs.system
            assert info['app_name'] == "test-app"
            assert info['config'] == mock_paths['config']
            assert info['data'] == mock_paths['data']
            assert info['cache'] == mock_paths['cache']
            assert info['state'] == mock_paths['state']
            assert info['logs'] == mock_paths['logs']
            assert info['runtime'] == mock_paths['runtime']

    def test_get_directory_info_no_runtime(self):
        """Test directory information with no runtime directory."""
        os_dirs = OSDirectories("test-app")

        with patch.object(os_dirs, 'get_config_dir', return_value=Path('/config')), \
             patch.object(os_dirs, 'get_data_dir', return_value=Path('/data')), \
             patch.object(os_dirs, 'get_cache_dir', return_value=Path('/cache')), \
             patch.object(os_dirs, 'get_state_dir', return_value=Path('/state')), \
             patch.object(os_dirs, 'get_log_dir', return_value=Path('/logs')), \
             patch.object(os_dirs, 'get_runtime_dir', return_value=None):

            info = os_dirs.get_directory_info()

            assert info['runtime'] is None

    def test_class_constant_app_name(self):
        """Test class constant APP_NAME."""
        assert OSDirectories.APP_NAME == "workspace-qdrant"

    @patch('platform.system', return_value='UnknownOS')
    def test_unknown_system_behavior(self, mock_system):
        """Test behavior on unknown operating system."""
        os_dirs = OSDirectories("test-app")

        # Should default to Linux-like behavior for unknown systems
        with patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=Path('/home/testuser')):

            config_dir = os_dirs.get_config_dir()
            expected = Path('/home/testuser/.config/test-app')
            assert config_dir == expected

    def test_path_consistency(self):
        """Test that paths are consistent across method calls."""
        os_dirs = OSDirectories("test-app")

        # Multiple calls should return the same paths
        config_dir1 = os_dirs.get_config_dir()
        config_dir2 = os_dirs.get_config_dir()
        assert config_dir1 == config_dir2

        data_dir1 = os_dirs.get_data_dir()
        data_dir2 = os_dirs.get_data_dir()
        assert data_dir1 == data_dir2

    def test_file_path_methods_consistency(self):
        """Test that file path methods are consistent with directory methods."""
        os_dirs = OSDirectories("test-app")

        config_dir = os_dirs.get_config_dir()
        config_file = os_dirs.get_config_file("test.conf")
        assert config_file.parent == config_dir

        data_dir = os_dirs.get_data_dir()
        data_file = os_dirs.get_data_file("test.db")
        assert data_file.parent == data_dir

        cache_dir = os_dirs.get_cache_dir()
        cache_file = os_dirs.get_cache_file("test.cache")
        assert cache_file.parent == cache_dir

        state_dir = os_dirs.get_state_dir()
        state_file = os_dirs.get_state_file("test.state")
        assert state_file.parent == state_dir

        log_dir = os_dirs.get_log_dir()
        log_file = os_dirs.get_log_file("test.log")
        assert log_file.parent == log_dir


if __name__ == "__main__":
    pytest.main([__file__])
