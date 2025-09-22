"""
Tests for OS-standard directory utilities.

Tests OS-standard directory detection and creation across different platforms.
"""

import os
import platform
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from workspace_qdrant_mcp.utils.os_directories import OSDirectories


class TestOSDirectories:
    """Test OS-standard directory functionality."""

    def test_initialization(self):
        """Test OSDirectories initialization."""
        # Default initialization
        os_dirs = OSDirectories()
        assert os_dirs.app_name == "workspace-qdrant"
        assert os_dirs.system == platform.system().lower()

        # Custom app name
        custom_dirs = OSDirectories("custom-app")
        assert custom_dirs.app_name == "custom-app"

    @pytest.mark.parametrize("mock_system,expected_config", [
        ("darwin", "Library/Application Support/workspace-qdrant"),
        ("windows", "AppData/Roaming/workspace-qdrant"),
        ("linux", ".config/workspace-qdrant"),
    ])
    def test_config_dir_by_platform(self, mock_system, expected_config):
        """Test config directory selection by platform."""
        with patch('platform.system', return_value=mock_system.title()):
            os_dirs = OSDirectories()

            # Mock environment variables to ensure predictable results
            with patch.dict(os.environ, {}, clear=True):
                config_dir = os_dirs.get_config_dir()

                # Check that the path contains the expected suffix
                assert expected_config in str(config_dir)

    def test_xdg_config_home_override(self):
        """Test XDG_CONFIG_HOME environment variable override on Linux."""
        with patch('platform.system', return_value='Linux'):
            os_dirs = OSDirectories()

            # Test with XDG_CONFIG_HOME set
            with patch.dict(os.environ, {'XDG_CONFIG_HOME': '/custom/config'}):
                config_dir = os_dirs.get_config_dir()
                assert config_dir == Path('/custom/config/workspace-qdrant')

    def test_windows_appdata_override(self):
        """Test Windows APPDATA environment variable override."""
        with patch('platform.system', return_value='Windows'):
            os_dirs = OSDirectories()

            # Test with APPDATA set
            with patch.dict(os.environ, {'APPDATA': '/custom/appdata'}):
                config_dir = os_dirs.get_config_dir()
                assert config_dir == Path('/custom/appdata/workspace-qdrant')

    @pytest.mark.parametrize("mock_system,expected_cache", [
        ("darwin", "Library/Caches/workspace-qdrant"),
        ("windows", "AppData/Local/workspace-qdrant/cache"),
        ("linux", ".cache/workspace-qdrant"),
    ])
    def test_cache_dir_by_platform(self, mock_system, expected_cache):
        """Test cache directory selection by platform."""
        with patch('platform.system', return_value=mock_system.title()):
            os_dirs = OSDirectories()

            with patch.dict(os.environ, {}, clear=True):
                cache_dir = os_dirs.get_cache_dir()
                assert expected_cache in str(cache_dir)

    @pytest.mark.parametrize("mock_system,expected_state", [
        ("darwin", "Library/Application Support/workspace-qdrant"),
        ("windows", "AppData/Local/workspace-qdrant/state"),
        ("linux", ".local/state/workspace-qdrant"),
    ])
    def test_state_dir_by_platform(self, mock_system, expected_state):
        """Test state directory selection by platform."""
        with patch('platform.system', return_value=mock_system.title()):
            os_dirs = OSDirectories()

            with patch.dict(os.environ, {}, clear=True):
                state_dir = os_dirs.get_state_dir()
                assert expected_state in str(state_dir)

    @pytest.mark.parametrize("mock_system,expected_log", [
        ("darwin", "Library/Logs/workspace-qdrant"),
        ("windows", "AppData/Local/workspace-qdrant/logs"),
        ("linux", ".local/state/workspace-qdrant/logs"),
    ])
    def test_log_dir_by_platform(self, mock_system, expected_log):
        """Test log directory selection by platform."""
        with patch('platform.system', return_value=mock_system.title()):
            os_dirs = OSDirectories()

            with patch.dict(os.environ, {}, clear=True):
                log_dir = os_dirs.get_log_dir()
                assert expected_log in str(log_dir)

    def test_file_path_methods(self):
        """Test file path generation methods."""
        os_dirs = OSDirectories()

        # Test config file
        config_file = os_dirs.get_config_file("config.yaml")
        assert config_file.name == "config.yaml"
        assert config_file.parent == os_dirs.get_config_dir()

        # Test data file
        data_file = os_dirs.get_data_file("data.db")
        assert data_file.name == "data.db"
        assert data_file.parent == os_dirs.get_data_dir()

        # Test cache file
        cache_file = os_dirs.get_cache_file("cache.tmp")
        assert cache_file.name == "cache.tmp"
        assert cache_file.parent == os_dirs.get_cache_dir()

        # Test state file
        state_file = os_dirs.get_state_file("state.db")
        assert state_file.name == "state.db"
        assert state_file.parent == os_dirs.get_state_dir()

        # Test log file
        log_file = os_dirs.get_log_file("app.log")
        assert log_file.name == "app.log"
        assert log_file.parent == os_dirs.get_log_dir()

    def test_ensure_directories(self):
        """Test directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock home directory to use temp directory
            with patch('pathlib.Path.home', return_value=Path(temp_dir)):
                os_dirs = OSDirectories()

                # Ensure directories don't exist initially
                config_dir = os_dirs.get_config_dir()
                assert not config_dir.exists()

                # Create directories
                os_dirs.ensure_directories()

                # Verify all directories were created
                assert os_dirs.get_config_dir().exists()
                assert os_dirs.get_data_dir().exists()
                assert os_dirs.get_cache_dir().exists()
                assert os_dirs.get_state_dir().exists()
                assert os_dirs.get_log_dir().exists()

    def test_runtime_dir_linux(self):
        """Test runtime directory on Linux."""
        with patch('platform.system', return_value='Linux'):
            os_dirs = OSDirectories()

            # Test with XDG_RUNTIME_DIR set
            with patch.dict(os.environ, {'XDG_RUNTIME_DIR': '/run/user/1000'}):
                runtime_dir = os_dirs.get_runtime_dir()
                assert runtime_dir == Path('/run/user/1000/workspace-qdrant')

            # Test without XDG_RUNTIME_DIR
            with patch.dict(os.environ, {}, clear=True):
                runtime_dir = os_dirs.get_runtime_dir()
                assert runtime_dir is None

    def test_runtime_dir_non_linux(self):
        """Test runtime directory on non-Linux systems."""
        for system in ['Darwin', 'Windows']:
            with patch('platform.system', return_value=system):
                os_dirs = OSDirectories()
                runtime_dir = os_dirs.get_runtime_dir()
                assert runtime_dir is None

    def test_migration_plan(self):
        """Test migration plan generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create some legacy files
            legacy_config = temp_path / "config.yaml"
            legacy_db = temp_path / "database.db"
            legacy_config.write_text("test config")
            legacy_db.write_text("test db")

            os_dirs = OSDirectories()

            legacy_paths = {
                "config": legacy_config,
                "state": legacy_db,
                "nonexistent": temp_path / "nonexistent.txt"
            }

            migration_plan = os_dirs.migrate_from_legacy_paths(legacy_paths)

            # Should have migration plans for existing files
            assert "config" in migration_plan
            assert "state" in migration_plan

            # Should not have plan for nonexistent file
            assert "nonexistent" not in migration_plan

            # Verify migration plan structure
            config_plan = migration_plan["config"][0]
            assert config_plan["from"] == legacy_config
            assert config_plan["exists"] is True
            assert "to" in config_plan

    def test_directory_info(self):
        """Test directory information retrieval."""
        os_dirs = OSDirectories()
        info = os_dirs.get_directory_info()

        # Verify required keys are present
        required_keys = ['system', 'app_name', 'config', 'data', 'cache', 'state', 'logs']
        for key in required_keys:
            assert key in info

        # Verify values are strings (except runtime which can be None)
        for key in required_keys:
            assert isinstance(info[key], str)

        # Verify system matches current platform
        assert info['system'] == platform.system().lower()
        assert info['app_name'] == 'workspace-qdrant'

    def test_custom_app_name_in_paths(self):
        """Test that custom app name is used in all paths."""
        custom_name = "my-custom-app"
        os_dirs = OSDirectories(custom_name)

        # All directory paths should contain the custom app name
        assert custom_name in str(os_dirs.get_config_dir())
        assert custom_name in str(os_dirs.get_data_dir())
        assert custom_name in str(os_dirs.get_cache_dir())
        assert custom_name in str(os_dirs.get_state_dir())
        assert custom_name in str(os_dirs.get_log_dir())

        # Runtime dir should also use custom name (if supported)
        runtime_dir = os_dirs.get_runtime_dir()
        if runtime_dir:
            assert custom_name in str(runtime_dir)

    def test_xdg_data_home_override(self):
        """Test XDG_DATA_HOME environment variable override on Linux."""
        with patch('platform.system', return_value='Linux'):
            os_dirs = OSDirectories()

            # Test with XDG_DATA_HOME set
            with patch.dict(os.environ, {'XDG_DATA_HOME': '/custom/data'}):
                data_dir = os_dirs.get_data_dir()
                assert data_dir == Path('/custom/data/workspace-qdrant')

    def test_xdg_cache_home_override(self):
        """Test XDG_CACHE_HOME environment variable override on Linux."""
        with patch('platform.system', return_value='Linux'):
            os_dirs = OSDirectories()

            # Test with XDG_CACHE_HOME set
            with patch.dict(os.environ, {'XDG_CACHE_HOME': '/custom/cache'}):
                cache_dir = os_dirs.get_cache_dir()
                assert cache_dir == Path('/custom/cache/workspace-qdrant')

    def test_xdg_state_home_override(self):
        """Test XDG_STATE_HOME environment variable override on Linux."""
        with patch('platform.system', return_value='Linux'):
            os_dirs = OSDirectories()

            # Test with XDG_STATE_HOME set
            with patch.dict(os.environ, {'XDG_STATE_HOME': '/custom/state'}):
                state_dir = os_dirs.get_state_dir()
                assert state_dir == Path('/custom/state/workspace-qdrant')

    def test_windows_localappdata_fallback(self):
        """Test Windows LOCALAPPDATA fallback behavior."""
        with patch('platform.system', return_value='Windows'):
            os_dirs = OSDirectories()

            # Test fallback when LOCALAPPDATA is not set
            with patch.dict(os.environ, {}, clear=True):
                data_dir = os_dirs.get_data_dir()
                cache_dir = os_dirs.get_cache_dir()
                state_dir = os_dirs.get_state_dir()
                log_dir = os_dirs.get_log_dir()

                # Should use AppData/Local fallback
                assert "AppData/Local" in str(data_dir)
                assert "AppData/Local" in str(cache_dir)
                assert "AppData/Local" in str(state_dir)
                assert "AppData/Local" in str(log_dir)

    def test_windows_appdata_fallback(self):
        """Test Windows APPDATA fallback behavior."""
        with patch('platform.system', return_value='Windows'):
            os_dirs = OSDirectories()

            # Test fallback when APPDATA is not set
            with patch.dict(os.environ, {}, clear=True):
                config_dir = os_dirs.get_config_dir()

                # Should use AppData/Roaming fallback
                assert "AppData/Roaming" in str(config_dir)