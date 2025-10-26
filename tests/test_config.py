"""Comprehensive unit tests for the configuration system.

This module tests all aspects of the configuration system including:
- XDG configuration directory resolution
- TOML file loading and validation
- Configuration precedence handling
- Environment variable parsing
- Configuration merging and validation
- Error handling for various failure scenarios
"""

import os
import platform
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import mock_open, patch

import pytest
import toml
from core.config import (
    ConfigFileNotFoundError,
    ConfigPaths,
    ConfigType,
    ConfigurationError,
    ConfigValidationError,
    DaemonConfig,
    McpConfig,
    create_default_config_files,
    get_config_paths,
    load_daemon_config,
    load_mcp_config,
    load_toml_file,
    merge_config_sources,
    parse_env_vars_for_daemon,
    parse_env_vars_for_mcp,
    resolve_config_directory,
    validate_config,
)


class TestConfigDirectoryResolution:
    """Test configuration directory resolution with XDG compliance."""

    @pytest.fixture
    def mock_home(self, tmp_path):
        """Fixture providing a mock home directory."""
        return tmp_path / "home"

    def test_resolve_config_directory_macos(self, mock_home):
        """Test macOS configuration directory resolution."""
        with patch('platform.system', return_value='Darwin'), \
             patch('pathlib.Path.home', return_value=mock_home):
            config_dir = resolve_config_directory()
            expected = mock_home / "Library" / "Application Support" / "workspace-qdrant-mcp"
            assert config_dir == expected
            assert config_dir.exists()

    def test_resolve_config_directory_linux_with_xdg(self, mock_home, tmp_path):
        """Test Linux configuration directory with XDG_CONFIG_HOME set."""
        xdg_config = tmp_path / "xdg_config"
        xdg_config.mkdir()

        with patch('platform.system', return_value='Linux'), \
             patch.dict(os.environ, {'XDG_CONFIG_HOME': str(xdg_config)}), \
             patch('pathlib.Path.home', return_value=mock_home):
            config_dir = resolve_config_directory()
            expected = xdg_config / "workspace-qdrant-mcp"
            assert config_dir == expected
            assert config_dir.exists()

    def test_resolve_config_directory_linux_without_xdg(self, mock_home):
        """Test Linux configuration directory without XDG_CONFIG_HOME."""
        with patch('platform.system', return_value='Linux'), \
             patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=mock_home):
            config_dir = resolve_config_directory()
            expected = mock_home / ".config" / "workspace-qdrant-mcp"
            assert config_dir == expected
            assert config_dir.exists()

    def test_resolve_config_directory_windows_with_appdata(self, mock_home, tmp_path):
        """Test Windows configuration directory with APPDATA set."""
        appdata = tmp_path / "appdata"
        appdata.mkdir()

        with patch('platform.system', return_value='Windows'), \
             patch.dict(os.environ, {'APPDATA': str(appdata)}), \
             patch('pathlib.Path.home', return_value=mock_home):
            config_dir = resolve_config_directory()
            expected = appdata / "workspace-qdrant-mcp"
            assert config_dir == expected
            assert config_dir.exists()

    def test_resolve_config_directory_windows_without_appdata(self, mock_home):
        """Test Windows configuration directory without APPDATA."""
        with patch('platform.system', return_value='Windows'), \
             patch.dict(os.environ, {}, clear=True), \
             patch('pathlib.Path.home', return_value=mock_home):
            config_dir = resolve_config_directory()
            expected = mock_home / "AppData" / "Roaming" / "workspace-qdrant-mcp"
            assert config_dir == expected
            assert config_dir.exists()

    def test_resolve_config_directory_other_os(self, mock_home):
        """Test configuration directory for other operating systems."""
        with patch('platform.system', return_value='FreeBSD'), \
             patch('pathlib.Path.home', return_value=mock_home):
            config_dir = resolve_config_directory()
            expected = mock_home / ".workspace-qdrant-mcp"
            assert config_dir == expected
            assert config_dir.exists()

    def test_resolve_config_directory_invalid_xdg(self, mock_home):
        """Test Linux with invalid XDG_CONFIG_HOME falls back to default."""
        with patch('platform.system', return_value='Linux'), \
             patch.dict(os.environ, {'XDG_CONFIG_HOME': '/nonexistent/path'}), \
             patch('pathlib.Path.home', return_value=mock_home):
            # This should still work as mkdir(parents=True, exist_ok=True) creates the path
            config_dir = resolve_config_directory()
            # Should fall back to the provided XDG path and create it
            expected = Path('/nonexistent/path') / "workspace-qdrant-mcp"
            assert config_dir == expected

    def test_resolve_config_directory_permission_error(self, mock_home):
        """Test configuration directory resolution with permission error."""
        with patch('platform.system', return_value='Darwin'), \
             patch('pathlib.Path.home', return_value=mock_home), \
             patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with pytest.raises(ConfigurationError, match="Failed to create configuration directory"):
                resolve_config_directory()


class TestConfigPaths:
    """Test configuration paths resolution."""

    def test_get_config_paths(self, tmp_path):
        """Test getting all configuration paths."""
        with patch('core.config.resolve_config_directory', return_value=tmp_path):
            paths = get_config_paths()

            assert isinstance(paths, ConfigPaths)
            assert paths.config_dir == tmp_path
            assert paths.mcp_config_file == tmp_path / "workspace-qdrant-mcp-config.toml"
            assert paths.daemon_config_file == tmp_path / "workspace-qdrant-daemon.toml"
            assert paths.cache_dir == tmp_path / "cache"
            assert paths.data_dir == tmp_path / "data"


class TestTomlFileLoading:
    """Test TOML file loading functionality."""

    def test_load_toml_file_success(self, tmp_path):
        """Test successful TOML file loading."""
        config_file = tmp_path / "config.toml"
        test_config = {
            "server_name": "test-server",
            "timeout": 5000,
            "enable_feature": True
        }

        with open(config_file, 'w') as f:
            toml.dump(test_config, f)

        result = load_toml_file(config_file)
        assert result == test_config

    def test_load_toml_file_not_found_optional(self, tmp_path):
        """Test loading non-existent optional TOML file."""
        config_file = tmp_path / "nonexistent.toml"
        result = load_toml_file(config_file, required=False)
        assert result == {}

    def test_load_toml_file_not_found_required(self, tmp_path):
        """Test loading non-existent required TOML file."""
        config_file = tmp_path / "nonexistent.toml"
        with pytest.raises(ConfigFileNotFoundError):
            load_toml_file(config_file, required=True)

    def test_load_toml_file_invalid_syntax(self, tmp_path):
        """Test loading TOML file with invalid syntax."""
        config_file = tmp_path / "invalid.toml"
        with open(config_file, 'w') as f:
            f.write("invalid toml content [[[")

        with pytest.raises(ConfigurationError, match="Invalid TOML syntax"):
            load_toml_file(config_file)

    def test_load_toml_file_read_error(self, tmp_path):
        """Test TOML file loading with read permission error."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("test = true")

        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            with pytest.raises(ConfigurationError, match="Failed to read configuration file"):
                load_toml_file(config_file)


class TestConfigMerging:
    """Test configuration merging with proper precedence."""

    def test_merge_config_sources_precedence(self):
        """Test configuration merging respects precedence order."""
        defaults = {
            "server_name": "default-server",
            "timeout": 1000,
            "feature_a": True,
            "nested": {"value1": "default", "value2": "default"}
        }

        user_config = {
            "server_name": "user-server",
            "timeout": 2000,
            "nested": {"value1": "user"}
        }

        env_vars = {
            "timeout": 3000,
            "feature_b": True,
            "nested": {"value2": "env"}
        }

        cli_args = {
            "timeout": 4000,
            "feature_c": True
        }

        result = merge_config_sources(defaults, user_config, env_vars, cli_args)

        # CLI args should have highest precedence
        assert result["timeout"] == 4000
        assert result["feature_c"] is True

        # Env vars should override user config and defaults
        assert result["feature_b"] is True

        # User config should override defaults
        assert result["server_name"] == "user-server"

        # Defaults should be preserved when not overridden
        assert result["feature_a"] is True

        # Nested merging should work correctly
        assert result["nested"]["value1"] == "user"  # From user config
        assert result["nested"]["value2"] == "env"   # From env vars

    def test_merge_config_sources_deep_merge(self):
        """Test deep merging of nested dictionaries."""
        defaults = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "options": {
                    "timeout": 30,
                    "pool_size": 10
                }
            }
        }

        user_config = {
            "database": {
                "host": "user-host",
                "options": {
                    "timeout": 60
                }
            }
        }

        result = merge_config_sources(defaults, user_config, {}, {})

        assert result["database"]["host"] == "user-host"  # Overridden
        assert result["database"]["port"] == 5432         # Preserved
        assert result["database"]["options"]["timeout"] == 60  # Overridden
        assert result["database"]["options"]["pool_size"] == 10  # Preserved

    def test_merge_config_sources_empty_sources(self):
        """Test merging with empty configuration sources."""
        defaults = {"key": "value"}
        result = merge_config_sources(defaults, {}, {}, {})
        assert result == defaults


class TestMcpConfigLoading:
    """Test MCP configuration loading."""

    def test_load_mcp_config_defaults(self, tmp_path):
        """Test loading MCP config with defaults only."""
        config_file = tmp_path / "mcp-config.toml"

        with patch('core.config.get_config_paths') as mock_paths:
            mock_paths.return_value = ConfigPaths(
                config_dir=tmp_path,
                mcp_config_file=config_file,
                daemon_config_file=tmp_path / "daemon.toml"
            )

            config = load_mcp_config()

            assert isinstance(config, McpConfig)
            assert config.server_name == "workspace-qdrant-mcp"
            assert config.qdrant_url == "http://localhost:6333"
            assert config.max_concurrent_tasks == 4
            assert config.enable_lsp is True
            assert config.enable_web_ui is False

    def test_load_mcp_config_from_file(self, tmp_path):
        """Test loading MCP config from TOML file."""
        config_file = tmp_path / "mcp-config.toml"
        file_config = {
            "server_name": "custom-server",
            "qdrant_url": "http://custom-host:6333",
            "max_concurrent_tasks": 8,
            "custom_field": "custom_value"
        }

        with open(config_file, 'w') as f:
            toml.dump(file_config, f)

        config = load_mcp_config(config_file)

        assert config.server_name == "custom-server"
        assert config.qdrant_url == "http://custom-host:6333"
        assert config.max_concurrent_tasks == 8
        assert config.additional_config["custom_field"] == "custom_value"

    def test_load_mcp_config_with_env_vars(self, tmp_path):
        """Test loading MCP config with environment variable overrides."""
        config_file = tmp_path / "mcp-config.toml"
        env_vars = {
            "server_name": "env-server",
            "max_concurrent_tasks": 16
        }

        config = load_mcp_config(config_file, env_vars=env_vars)

        assert config.server_name == "env-server"
        assert config.max_concurrent_tasks == 16

    def test_load_mcp_config_with_cli_args(self, tmp_path):
        """Test loading MCP config with CLI argument overrides."""
        config_file = tmp_path / "mcp-config.toml"
        cli_args = {
            "server_name": "cli-server",
            "enable_web_ui": True
        }

        config = load_mcp_config(config_file, cli_args=cli_args)

        assert config.server_name == "cli-server"
        assert config.enable_web_ui is True

    def test_load_mcp_config_precedence(self, tmp_path):
        """Test MCP config loading with all sources and precedence."""
        config_file = tmp_path / "mcp-config.toml"
        file_config = {"server_name": "file-server", "max_concurrent_tasks": 2}

        with open(config_file, 'w') as f:
            toml.dump(file_config, f)

        env_vars = {"server_name": "env-server", "qdrant_timeout_ms": 45000}
        cli_args = {"server_name": "cli-server"}

        config = load_mcp_config(config_file, env_vars=env_vars, cli_args=cli_args)

        # CLI should have highest precedence
        assert config.server_name == "cli-server"
        # Env vars should override file config
        assert config.qdrant_timeout_ms == 45000
        # File config should override defaults
        assert config.max_concurrent_tasks == 2

    def test_load_mcp_config_validation_error(self, tmp_path):
        """Test MCP config loading with validation errors."""
        config_file = tmp_path / "mcp-config.toml"

        # Invalid config that would cause TypeError in McpConfig constructor
        with patch('core.config.McpConfig', side_effect=TypeError("Invalid config")):
            with pytest.raises(ConfigValidationError, match="Invalid MCP configuration structure"):
                load_mcp_config(config_file)


class TestDaemonConfigLoading:
    """Test daemon configuration loading."""

    def test_load_daemon_config_defaults(self, tmp_path):
        """Test loading daemon config with defaults only."""
        config_file = tmp_path / "daemon-config.toml"

        with patch('core.config.get_config_paths') as mock_paths:
            mock_paths.return_value = ConfigPaths(
                config_dir=tmp_path,
                mcp_config_file=tmp_path / "mcp.toml",
                daemon_config_file=config_file
            )

            config = load_daemon_config()

            assert isinstance(config, DaemonConfig)
            assert config.daemon_name == "workspace-qdrant-daemon"
            assert config.log_level == "info"
            assert config.auto_ingestion_enabled is True
            assert config.enable_preemption is True

    def test_load_daemon_config_from_file(self, tmp_path):
        """Test loading daemon config from TOML file."""
        config_file = tmp_path / "daemon-config.toml"
        file_config = {
            "daemon_name": "custom-daemon",
            "log_level": "debug",
            "auto_ingestion_enabled": False,
            "custom_setting": "value"
        }

        with open(config_file, 'w') as f:
            toml.dump(file_config, f)

        config = load_daemon_config(config_file)

        assert config.daemon_name == "custom-daemon"
        assert config.log_level == "debug"
        assert config.auto_ingestion_enabled is False
        assert config.additional_config["custom_setting"] == "value"

    def test_load_daemon_config_precedence(self, tmp_path):
        """Test daemon config loading with precedence."""
        config_file = tmp_path / "daemon-config.toml"
        file_config = {"daemon_name": "file-daemon", "log_level": "warn"}

        with open(config_file, 'w') as f:
            toml.dump(file_config, f)

        env_vars = {"daemon_name": "env-daemon"}
        cli_args = {"log_level": "error"}

        config = load_daemon_config(config_file, env_vars=env_vars, cli_args=cli_args)

        # CLI should override env vars
        assert config.log_level == "error"
        # Env vars should override file config
        assert config.daemon_name == "env-daemon"


class TestConfigValidation:
    """Test configuration validation."""

    def test_validate_mcp_config_valid(self):
        """Test validation of valid MCP configuration."""
        config = McpConfig(
            server_name="test-server",
            qdrant_url="http://localhost:6333",
            qdrant_timeout_ms=30000,
            max_concurrent_tasks=4,
            chunk_size=1000
        )

        errors = validate_config(config)
        assert errors == []

    def test_validate_mcp_config_invalid_server_name(self):
        """Test validation with invalid server name."""
        config = McpConfig(server_name="")
        errors = validate_config(config)
        assert "server_name must be a non-empty string" in errors

    def test_validate_mcp_config_invalid_qdrant_url(self):
        """Test validation with invalid Qdrant URL."""
        config = McpConfig(qdrant_url="")
        errors = validate_config(config)
        assert "qdrant_url must be a non-empty string" in errors

    def test_validate_mcp_config_negative_values(self):
        """Test validation with negative numeric values."""
        config = McpConfig(
            qdrant_timeout_ms=-1,
            max_concurrent_tasks=0,
            chunk_size=-100
        )
        errors = validate_config(config)

        assert "qdrant_timeout_ms must be positive" in errors
        assert "max_concurrent_tasks must be positive" in errors
        assert "chunk_size must be positive" in errors

    def test_validate_daemon_config_valid(self):
        """Test validation of valid daemon configuration."""
        config = DaemonConfig(
            daemon_name="test-daemon",
            log_level="info",
            qdrant_url="http://localhost:6333"
        )

        errors = validate_config(config)
        assert errors == []

    def test_validate_daemon_config_invalid_log_level(self):
        """Test validation with invalid log level."""
        config = DaemonConfig(log_level="invalid")
        errors = validate_config(config)
        assert "log_level must be one of: debug, info, warn, error" in errors

    def test_validate_config_invalid_additional_config(self):
        """Test validation with invalid additional config."""
        config = McpConfig()
        config.additional_config = "not a dict"  # Should be dict

        errors = validate_config(config)
        assert "additional_config must be a dictionary" in errors


class TestEnvVarParsing:
    """Test environment variable parsing."""

    @pytest.fixture
    def clean_env(self):
        """Fixture to provide clean environment without WQM_ variables."""
        original_env = os.environ.copy()
        # Remove any existing WQM_ variables
        for key in list(os.environ.keys()):
            if key.startswith(('WQM_MCP_', 'WQM_DAEMON_')):
                del os.environ[key]
        yield
        # Restore original environment (preserve pytest internal variables)
        current_keys = set(os.environ.keys())
        original_keys = set(original_env.keys())

        # Remove keys added during test (except pytest internal variables)
        for key in current_keys - original_keys:
            if not key.startswith('PYTEST_'):
                os.environ.pop(key, None)

        # Restore original values
        for key in original_keys:
            if key in original_env:
                os.environ[key] = original_env[key]
            else:
                os.environ.pop(key, None)

    def test_parse_env_vars_for_mcp_empty(self, clean_env):
        """Test parsing MCP environment variables when none are set."""
        result = parse_env_vars_for_mcp()
        assert result == {}

    def test_parse_env_vars_for_mcp_various_types(self, clean_env):
        """Test parsing MCP environment variables of different types."""
        env_vars = {
            'WQM_MCP_SERVER_NAME': 'test-server',
            'WQM_MCP_MAX_CONCURRENT_TASKS': '8',
            'WQM_MCP_ENABLE_LSP': 'true',
            'WQM_MCP_ENABLE_WEB_UI': 'false',
            'WQM_MCP_TIMEOUT_RATIO': '1.5'
        }

        with patch.dict(os.environ, env_vars):
            result = parse_env_vars_for_mcp()

            assert result['server_name'] == 'test-server'
            assert result['max_concurrent_tasks'] == 8
            assert result['enable_lsp'] is True
            assert result['enable_web_ui'] is False
            assert result['timeout_ratio'] == 1.5

    def test_parse_env_vars_for_mcp_case_conversion(self, clean_env):
        """Test MCP environment variable case conversion."""
        with patch.dict(os.environ, {'WQM_MCP_QDRANT_TIMEOUT_MS': '45000'}):
            result = parse_env_vars_for_mcp()
            assert result['qdrant_timeout_ms'] == 45000

    def test_parse_env_vars_for_daemon_various_types(self, clean_env):
        """Test parsing daemon environment variables of different types."""
        env_vars = {
            'WQM_DAEMON_DAEMON_NAME': 'custom-daemon',
            'WQM_DAEMON_MAX_CONCURRENT_TASKS': '12',
            'WQM_DAEMON_AUTO_INGESTION_ENABLED': 'false',
            'WQM_DAEMON_ENABLE_PREEMPTION': 'true'
        }

        with patch.dict(os.environ, env_vars):
            result = parse_env_vars_for_daemon()

            assert result['daemon_name'] == 'custom-daemon'
            assert result['max_concurrent_tasks'] == 12
            assert result['auto_ingestion_enabled'] is False
            assert result['enable_preemption'] is True

    @pytest.mark.parametrize("env_value,expected_type,expected_value", [
        ("true", bool, True),
        ("false", bool, False),
        ("TRUE", bool, True),
        ("False", bool, False),
        ("123", int, 123),
        ("0", int, 0),
        ("3.14", float, 3.14),
        ("0.0", float, 0.0),
        ("hello", str, "hello"),
        ("not_a_number", str, "not_a_number")
    ])
    def test_env_var_type_parsing(self, clean_env, env_value, expected_type, expected_value):
        """Test environment variable type parsing for various values."""
        with patch.dict(os.environ, {'WQM_MCP_TEST_VALUE': env_value}):
            result = parse_env_vars_for_mcp()
            assert isinstance(result['test_value'], expected_type)
            assert result['test_value'] == expected_value


class TestCreateDefaultConfigs:
    """Test default configuration file creation."""

    def test_create_default_config_files_new(self, tmp_path):
        """Test creating default configuration files when they don't exist."""
        with patch('core.config.get_config_paths') as mock_paths:
            mock_paths.return_value = ConfigPaths(
                config_dir=tmp_path,
                mcp_config_file=tmp_path / "mcp-config.toml",
                daemon_config_file=tmp_path / "daemon-config.toml"
            )

            paths = create_default_config_files()

            assert paths.mcp_config_file.exists()
            assert paths.daemon_config_file.exists()

            # Verify MCP config content
            mcp_config = toml.load(paths.mcp_config_file)
            assert mcp_config["server_name"] == "workspace-qdrant-mcp"
            assert mcp_config["enable_web_ui"] is False

            # Verify daemon config content
            daemon_config = toml.load(paths.daemon_config_file)
            assert daemon_config["daemon_name"] == "workspace-qdrant-daemon"
            assert daemon_config["auto_ingestion"]["enabled"] is True

    def test_create_default_config_files_existing(self, tmp_path):
        """Test creating default config files when they already exist."""
        mcp_config_file = tmp_path / "mcp-config.toml"
        daemon_config_file = tmp_path / "daemon-config.toml"

        # Create existing files with custom content
        existing_mcp = {"server_name": "existing-server"}
        existing_daemon = {"daemon_name": "existing-daemon"}

        with open(mcp_config_file, 'w') as f:
            toml.dump(existing_mcp, f)
        with open(daemon_config_file, 'w') as f:
            toml.dump(existing_daemon, f)

        with patch('core.config.get_config_paths') as mock_paths:
            mock_paths.return_value = ConfigPaths(
                config_dir=tmp_path,
                mcp_config_file=mcp_config_file,
                daemon_config_file=daemon_config_file
            )

            paths = create_default_config_files()

            # Files should still exist with original content
            mcp_config = toml.load(paths.mcp_config_file)
            daemon_config = toml.load(paths.daemon_config_file)

            assert mcp_config["server_name"] == "existing-server"
            assert daemon_config["daemon_name"] == "existing-daemon"

    def test_create_default_config_files_write_error(self, tmp_path):
        """Test error handling when default config files cannot be written."""
        with patch('core.config.get_config_paths') as mock_paths:
            mock_paths.return_value = ConfigPaths(
                config_dir=tmp_path,
                mcp_config_file=tmp_path / "mcp-config.toml",
                daemon_config_file=tmp_path / "daemon-config.toml"
            )

            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                with pytest.raises(ConfigurationError, match="Failed to create MCP config file"):
                    create_default_config_files()


@pytest.mark.parametrize("platform_name,expected_dir", [
    ("Darwin", "Library/Application Support/workspace-qdrant-mcp"),
    ("Linux", ".config/workspace-qdrant-mcp"),
    ("Windows", "AppData/Roaming/workspace-qdrant-mcp"),
    ("FreeBSD", ".workspace-qdrant-mcp"),
])
def test_resolve_config_directory_platforms(platform_name, expected_dir, tmp_path):
    """Parameterized test for different OS platforms."""
    mock_home = tmp_path / "home"
    mock_home.mkdir()

    with patch('platform.system', return_value=platform_name), \
         patch('pathlib.Path.home', return_value=mock_home), \
         patch.dict(os.environ, {}, clear=True):

        config_dir = resolve_config_directory()
        expected_path = mock_home / Path(expected_dir)
        assert config_dir == expected_path
        assert config_dir.exists()


class TestIntegration:
    """Integration tests combining multiple configuration aspects."""

    def test_full_mcp_config_loading_workflow(self, tmp_path):
        """Test complete MCP configuration loading workflow."""
        # Setup directory structure
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "mcp-config.toml"

        # Create config file
        file_config = {
            "server_name": "integration-server",
            "max_concurrent_tasks": 6,
            "custom_section": {
                "nested_value": "test"
            }
        }
        with open(config_file, 'w') as f:
            toml.dump(file_config, f)

        # Setup environment variables
        env_vars = {
            'WQM_MCP_ENABLE_METRICS': 'false',
            'WQM_MCP_QDRANT_TIMEOUT_MS': '60000'
        }

        # CLI arguments
        cli_args = {
            "enable_web_ui": True,
            "server_name": "cli-override-server"
        }

        with patch.dict(os.environ, env_vars):
            env_config = parse_env_vars_for_mcp()
            config = load_mcp_config(config_file, env_vars=env_config, cli_args=cli_args)

        # Verify precedence
        assert config.server_name == "cli-override-server"  # CLI highest
        assert config.qdrant_timeout_ms == 60000  # Env var
        assert config.max_concurrent_tasks == 6   # File config
        assert config.chunk_size == 1000          # Default

        # Verify additional config
        assert config.additional_config["custom_section"]["nested_value"] == "test"

        # Verify validation passes
        errors = validate_config(config)
        assert errors == []

    def test_error_handling_chain(self, tmp_path):
        """Test error handling across the configuration loading chain."""
        # Test file not found -> should create defaults
        non_existent = tmp_path / "nonexistent.toml"
        config = load_mcp_config(non_existent)
        assert config.server_name == "workspace-qdrant-mcp"  # Default

        # Test invalid TOML -> should raise ConfigurationError
        invalid_file = tmp_path / "invalid.toml"
        invalid_file.write_text("invalid [[[[ toml")

        with pytest.raises(ConfigurationError):
            load_mcp_config(invalid_file)
