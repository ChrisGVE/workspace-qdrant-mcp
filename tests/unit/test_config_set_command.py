"""Unit tests for ConfigManager.set() and related CLI config set command.

Tests the new configuration setting functionality including:
- ConfigManager.set() method
- ConfigManager.save_to_file() method
- ConfigManager.get_config_file_path() method
- ConfigManager.requires_restart() classmethod
- _parse_config_value() helper function
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from common.core.config import ConfigManager, get_config_manager


class TestConfigManagerSet:
    """Test ConfigManager.set() method."""

    def test_set_simple_value(self):
        """Test setting a simple top-level value."""
        manager = ConfigManager()

        # Set a new value
        result = manager.set("server.debug", True)

        assert result is True
        assert manager.get("server.debug") is True

    def test_set_nested_value(self):
        """Test setting a nested value."""
        manager = ConfigManager()

        # Set a nested value
        result = manager.set("qdrant.timeout", 60000)

        assert result is True
        assert manager.get("qdrant.timeout") == 60000

    def test_set_deeply_nested_value(self):
        """Test setting a deeply nested value creates intermediate dicts."""
        manager = ConfigManager()

        # Set a deeply nested value that may not exist
        manager.set("custom.deep.nested.value", "test")

        assert manager.get("custom.deep.nested.value") == "test"

    def test_set_string_value(self):
        """Test setting a string value."""
        manager = ConfigManager()

        manager.set("qdrant.url", "http://custom:6333")

        assert manager.get("qdrant.url") == "http://custom:6333"

    def test_set_integer_value(self):
        """Test setting an integer value."""
        manager = ConfigManager()

        manager.set("server.port", 9000)

        assert manager.get("server.port") == 9000

    def test_set_list_value(self):
        """Test setting a list value."""
        manager = ConfigManager()

        manager.set("custom.list", ["a", "b", "c"])

        assert manager.get("custom.list") == ["a", "b", "c"]

    def test_set_empty_path_raises(self):
        """Test that setting with empty path raises ValueError."""
        manager = ConfigManager()

        with pytest.raises(ValueError, match="Configuration path cannot be empty"):
            manager.set("", "value")


class TestConfigManagerSaveToFile:
    """Test ConfigManager.save_to_file() method."""

    def test_save_to_file_creates_file(self):
        """Test that save_to_file creates a YAML file."""
        manager = ConfigManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "test_config.yaml"

            manager.save_to_file(config_path)

            assert config_path.exists()

            # Verify it's valid YAML
            with config_path.open("r") as f:
                saved_config = yaml.safe_load(f)

            assert isinstance(saved_config, dict)

    def test_save_to_file_creates_parent_dirs(self):
        """Test that save_to_file creates parent directories."""
        manager = ConfigManager()

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nested" / "dir" / "config.yaml"

            manager.save_to_file(config_path)

            assert config_path.exists()

    def test_save_preserves_set_values(self):
        """Test that saved file preserves values set with set()."""
        manager = ConfigManager()

        # Set a custom value
        manager.set("custom.test_key", "test_value")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"

            manager.save_to_file(config_path)

            # Reload and verify
            with config_path.open("r") as f:
                saved_config = yaml.safe_load(f)

            assert saved_config.get("custom", {}).get("test_key") == "test_value"


class TestConfigManagerRequiresRestart:
    """Test ConfigManager.requires_restart() classmethod."""

    def test_server_host_requires_restart(self):
        """Test that server.host requires restart."""
        assert ConfigManager.requires_restart("server.host") is True

    def test_server_port_requires_restart(self):
        """Test that server.port requires restart."""
        assert ConfigManager.requires_restart("server.port") is True

    def test_qdrant_url_requires_restart(self):
        """Test that qdrant.url requires restart."""
        assert ConfigManager.requires_restart("qdrant.url") is True

    def test_grpc_port_requires_restart(self):
        """Test that grpc.port requires restart."""
        assert ConfigManager.requires_restart("grpc.port") is True

    def test_embedding_model_requires_restart(self):
        """Test that embedding.model requires restart."""
        assert ConfigManager.requires_restart("embedding.model") is True

    def test_server_debug_does_not_require_restart(self):
        """Test that server.debug does NOT require restart."""
        assert ConfigManager.requires_restart("server.debug") is False

    def test_logging_level_does_not_require_restart(self):
        """Test that logging.level does NOT require restart."""
        assert ConfigManager.requires_restart("logging.level") is False

    def test_child_of_restart_required_requires_restart(self):
        """Test that children of restart-required settings also require restart."""
        # If server.host requires restart, server.host.something should too
        # (though this is an edge case)
        assert ConfigManager.requires_restart("server.host.nested") is True

    def test_unknown_setting_does_not_require_restart(self):
        """Test that unknown settings don't require restart."""
        assert ConfigManager.requires_restart("some.unknown.setting") is False


class TestParseConfigValue:
    """Test _parse_config_value() helper function."""

    def test_parse_true_values(self):
        """Test parsing true boolean values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("true") is True
        assert _parse_config_value("True") is True
        assert _parse_config_value("TRUE") is True
        assert _parse_config_value("yes") is True
        assert _parse_config_value("on") is True
        assert _parse_config_value("1") is True

    def test_parse_false_values(self):
        """Test parsing false boolean values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("false") is False
        assert _parse_config_value("False") is False
        assert _parse_config_value("FALSE") is False
        assert _parse_config_value("no") is False
        assert _parse_config_value("off") is False
        assert _parse_config_value("0") is False

    def test_parse_null_values(self):
        """Test parsing null/none values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("null") is None
        assert _parse_config_value("None") is None
        assert _parse_config_value("~") is None

    def test_parse_integer(self):
        """Test parsing integer values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("42") == 42
        assert _parse_config_value("-10") == -10
        assert _parse_config_value("8000") == 8000

    def test_parse_float(self):
        """Test parsing float values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("3.14") == 3.14
        assert _parse_config_value("-2.5") == -2.5

    def test_parse_list(self):
        """Test parsing comma-separated list values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        result = _parse_config_value("a,b,c")
        assert result == ["a", "b", "c"]

        # With spaces
        result = _parse_config_value("a, b, c")
        assert result == ["a", "b", "c"]

    def test_parse_string(self):
        """Test parsing plain string values."""
        from wqm_cli.cli.commands.admin import _parse_config_value

        assert _parse_config_value("hello") == "hello"
        assert _parse_config_value("http://localhost:6333") == "http://localhost:6333"


class TestConfigSetCLICommand:
    """Test the CLI config set command integration."""

    @patch('wqm_cli.cli.commands.admin.handle_async')
    def test_config_set_command_invokes_handler(self, mock_handle_async):
        """Test that config --set invokes the async handler."""
        from typer.testing import CliRunner
        from wqm_cli.cli.commands.admin import admin_app

        runner = CliRunner()
        result = runner.invoke(admin_app, ["config", "--set", "server.debug=true"])

        # Handler should be called
        mock_handle_async.assert_called_once()

    @patch('wqm_cli.cli.commands.admin.handle_async')
    def test_config_get_command_invokes_handler(self, mock_handle_async):
        """Test that config --get invokes the async handler."""
        from typer.testing import CliRunner
        from wqm_cli.cli.commands.admin import admin_app

        runner = CliRunner()
        result = runner.invoke(admin_app, ["config", "--get", "server.port"])

        # Handler should be called
        mock_handle_async.assert_called_once()

    @patch('wqm_cli.cli.commands.admin.handle_async')
    def test_config_show_command_invokes_handler(self, mock_handle_async):
        """Test that config --show invokes the async handler."""
        from typer.testing import CliRunner
        from wqm_cli.cli.commands.admin import admin_app

        runner = CliRunner()
        result = runner.invoke(admin_app, ["config", "--show"])

        # Handler should be called
        mock_handle_async.assert_called_once()
