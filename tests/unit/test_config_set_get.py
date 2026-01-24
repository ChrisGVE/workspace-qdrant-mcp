"""Unit tests for ConfigManager set/get functionality (Task 466).

Tests the ConfigManager.set(), save_to_file(), get_config_file_path(),
requires_restart() methods, and CLI --set/--get options.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from common.core.config import ConfigManager, reset_config


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            "server": {"host": "127.0.0.1", "port": 8000},
            "qdrant": {"url": "http://localhost:6333"},
            "embedding": {"model": "test-model", "chunk_size": 800},
        }, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def config_manager(temp_config_file):
    """Create a ConfigManager instance with temp config."""
    # Reset any existing singleton
    reset_config()
    os.environ['WQM_TEST_MODE'] = 'true'

    manager = ConfigManager(config_file=temp_config_file)

    yield manager

    # Cleanup
    reset_config()
    if 'WQM_TEST_MODE' in os.environ:
        del os.environ['WQM_TEST_MODE']


class TestConfigManagerSet:
    """Tests for ConfigManager.set() method."""

    def test_set_simple_value(self, config_manager):
        """Test setting a simple configuration value."""
        config_manager.set("server.port", 9000)
        assert config_manager.get("server.port") == 9000

    def test_set_string_value(self, config_manager):
        """Test setting a string value."""
        config_manager.set("qdrant.url", "http://newhost:6333")
        assert config_manager.get("qdrant.url") == "http://newhost:6333"

    def test_set_boolean_value(self, config_manager):
        """Test setting a boolean value."""
        config_manager.set("server.debug", True)
        assert config_manager.get("server.debug") is True

    def test_set_nested_value(self, config_manager):
        """Test setting a deeply nested value."""
        config_manager.set("embedding.advanced.feature", "enabled")
        assert config_manager.get("embedding.advanced.feature") == "enabled"

    def test_set_creates_missing_parents(self, config_manager):
        """Test that set creates missing parent dictionaries."""
        config_manager.set("new.nested.deeply.value", 42)
        assert config_manager.get("new.nested.deeply.value") == 42
        assert config_manager.get("new.nested.deeply") == {"value": 42}

    def test_set_overwrites_existing(self, config_manager):
        """Test that set overwrites existing values."""
        original = config_manager.get("server.host")
        assert original == "127.0.0.1"

        config_manager.set("server.host", "0.0.0.0")
        assert config_manager.get("server.host") == "0.0.0.0"

    def test_set_list_value(self, config_manager):
        """Test setting a list value."""
        patterns = ["*.py", "*.js", "*.ts"]
        config_manager.set("workspace.patterns", patterns)
        assert config_manager.get("workspace.patterns") == patterns

    def test_set_dict_value(self, config_manager):
        """Test setting a dictionary value."""
        config = {"enabled": True, "timeout": 30}
        config_manager.set("custom.config", config)
        assert config_manager.get("custom.config") == config

    def test_set_empty_path_raises_error(self, config_manager):
        """Test that setting with empty path raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            config_manager.set("", "value")

    def test_set_none_value(self, config_manager):
        """Test setting a None value."""
        config_manager.set("server.optional", None)
        assert config_manager.get("server.optional") is None


class TestConfigManagerSaveToFile:
    """Tests for ConfigManager.save_to_file() method."""

    def test_save_to_original_file(self, config_manager, temp_config_file):
        """Test saving back to the original config file."""
        config_manager.set("server.port", 9999)
        saved_path = config_manager.save_to_file()

        assert saved_path == temp_config_file

        # Verify the file was updated
        with open(temp_config_file, 'r') as f:
            saved_config = yaml.safe_load(f)
        assert saved_config["server"]["port"] == 9999

    def test_save_to_new_file(self, config_manager):
        """Test saving to a new file path."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            new_path = f.name

        try:
            config_manager.set("server.port", 8888)
            saved_path = config_manager.save_to_file(new_path)

            assert saved_path == new_path

            with open(new_path, 'r') as f:
                saved_config = yaml.safe_load(f)
            assert saved_config["server"]["port"] == 8888
        finally:
            if os.path.exists(new_path):
                os.unlink(new_path)

    def test_save_creates_parent_directories(self, config_manager):
        """Test that save_to_file creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_path = os.path.join(tmpdir, "subdir", "config.yaml")
            saved_path = config_manager.save_to_file(new_path)

            assert saved_path == new_path
            assert os.path.exists(new_path)

    def test_save_without_path_uses_loaded_file(self, config_manager, temp_config_file):
        """Test that save without path uses the originally loaded file."""
        saved_path = config_manager.save_to_file()
        assert saved_path == temp_config_file

    def test_save_updates_tracked_path(self, config_manager):
        """Test that save_to_file updates the tracked config path."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            new_path = f.name

        try:
            config_manager.save_to_file(new_path)
            assert config_manager.get_config_file_path() == new_path
        finally:
            if os.path.exists(new_path):
                os.unlink(new_path)


class TestConfigManagerGetConfigFilePath:
    """Tests for ConfigManager.get_config_file_path() method."""

    def test_returns_loaded_file_path(self, config_manager, temp_config_file):
        """Test that get_config_file_path returns the loaded file path."""
        assert config_manager.get_config_file_path() == temp_config_file

    def test_returns_none_when_no_file_loaded(self):
        """Test that get_config_file_path returns None when no file was loaded."""
        reset_config()
        os.environ['WQM_TEST_MODE'] = 'true'

        try:
            manager = ConfigManager()
            assert manager.get_config_file_path() is None
        finally:
            reset_config()
            if 'WQM_TEST_MODE' in os.environ:
                del os.environ['WQM_TEST_MODE']


class TestConfigManagerRequiresRestart:
    """Tests for ConfigManager.requires_restart() method."""

    def test_server_host_requires_restart(self, config_manager):
        """Test that server.host requires restart."""
        assert config_manager.requires_restart("server.host") is True

    def test_server_port_requires_restart(self, config_manager):
        """Test that server.port requires restart."""
        assert config_manager.requires_restart("server.port") is True

    def test_qdrant_url_requires_restart(self, config_manager):
        """Test that qdrant.url requires restart."""
        assert config_manager.requires_restart("qdrant.url") is True

    def test_qdrant_api_key_requires_restart(self, config_manager):
        """Test that qdrant.api_key requires restart."""
        assert config_manager.requires_restart("qdrant.api_key") is True

    def test_grpc_host_requires_restart(self, config_manager):
        """Test that grpc.host requires restart."""
        assert config_manager.requires_restart("grpc.host") is True

    def test_grpc_port_requires_restart(self, config_manager):
        """Test that grpc.port requires restart."""
        assert config_manager.requires_restart("grpc.port") is True

    def test_embedding_model_requires_restart(self, config_manager):
        """Test that embedding.model requires restart."""
        assert config_manager.requires_restart("embedding.model") is True

    def test_auto_ingestion_enabled_requires_restart(self, config_manager):
        """Test that auto_ingestion.enabled requires restart."""
        assert config_manager.requires_restart("auto_ingestion.enabled") is True

    def test_other_settings_do_not_require_restart(self, config_manager):
        """Test that other settings do not require restart."""
        assert config_manager.requires_restart("workspace.patterns") is False
        assert config_manager.requires_restart("logging.level") is False
        assert config_manager.requires_restart("embedding.chunk_size") is False


class TestParseConfigValue:
    """Tests for _parse_config_value helper function."""

    def test_parse_true_values(self):
        """Test parsing various true values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("true") is True
        assert _parse_config_value("True") is True
        assert _parse_config_value("TRUE") is True
        assert _parse_config_value("yes") is True
        assert _parse_config_value("on") is True
        assert _parse_config_value("1") == 1  # Note: numeric "1" becomes int

    def test_parse_false_values(self):
        """Test parsing various false values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("false") is False
        assert _parse_config_value("False") is False
        assert _parse_config_value("FALSE") is False
        assert _parse_config_value("no") is False
        assert _parse_config_value("off") is False
        assert _parse_config_value("0") == 0  # Note: numeric "0" becomes int

    def test_parse_integer(self):
        """Test parsing integer values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("42") == 42
        assert _parse_config_value("8080") == 8080
        assert _parse_config_value("-10") == -10

    def test_parse_float(self):
        """Test parsing float values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("3.14") == 3.14
        assert _parse_config_value("0.5") == 0.5
        assert _parse_config_value("-2.5") == -2.5

    def test_parse_list(self):
        """Test parsing comma-separated list."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        result = _parse_config_value("*.py,*.js,*.ts")
        assert result == ["*.py", "*.js", "*.ts"]

    def test_parse_list_with_spaces(self):
        """Test parsing list with spaces around commas."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        result = _parse_config_value("foo , bar , baz")
        assert result == ["foo", "bar", "baz"]

    def test_parse_string(self):
        """Test parsing regular strings."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("http://localhost:6333") == "http://localhost:6333"
        assert _parse_config_value("my-model") == "my-model"


class TestRestartRequiredSettings:
    """Tests for RESTART_REQUIRED_SETTINGS constant."""

    def test_restart_required_settings_is_set(self):
        """Test that RESTART_REQUIRED_SETTINGS is a set."""
        assert isinstance(ConfigManager.RESTART_REQUIRED_SETTINGS, set)

    def test_restart_required_settings_not_empty(self):
        """Test that RESTART_REQUIRED_SETTINGS is not empty."""
        assert len(ConfigManager.RESTART_REQUIRED_SETTINGS) > 0

    def test_restart_required_settings_contains_expected(self):
        """Test that RESTART_REQUIRED_SETTINGS contains expected values."""
        expected = {
            "server.host",
            "server.port",
            "qdrant.url",
            "qdrant.api_key",
            "grpc.host",
            "grpc.port",
            "embedding.model",
            "auto_ingestion.enabled",
        }
        assert ConfigManager.RESTART_REQUIRED_SETTINGS == expected
