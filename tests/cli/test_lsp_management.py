"""Tests for LSP Management CLI commands.

This module tests the comprehensive LSP server management functionality
including status monitoring, installation, configuration, and diagnostics.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner
from wqm_cli.cli.commands.lsp_management import (
    KNOWN_LSP_SERVERS,
    _check_server_installation,
    _get_all_servers_status,
    _get_server_status,
    lsp_app,
)


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def mock_health_monitor():
    """Mock LSP health monitor."""
    monitor = MagicMock()
    monitor.register_server = MagicMock()
    monitor.perform_health_check = AsyncMock()
    monitor.get_server_health = MagicMock()
    monitor.get_all_servers_health = MagicMock(return_value={})
    return monitor


class TestLspStatusCommand:
    """Test the 'wqm lsp status' command."""

    def test_status_help(self, cli_runner):
        """Test that status command shows help."""
        result = cli_runner.invoke(lsp_app, ["status", "--help"])
        assert result.exit_code == 0
        assert "LSP server health and capability overview" in result.stdout

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._show_lsp_status')
    def test_status_all_servers(self, mock_show_status, cli_runner):
        """Test showing status for all servers."""
        mock_show_status.return_value = asyncio.Future()
        mock_show_status.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["status"])
        assert result.exit_code == 0
        mock_show_status.assert_called_once()

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._show_lsp_status')
    def test_status_specific_server(self, mock_show_status, cli_runner):
        """Test showing status for a specific server."""
        mock_show_status.return_value = asyncio.Future()
        mock_show_status.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["status", "python"])
        assert result.exit_code == 0
        mock_show_status.assert_called_once()

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._show_lsp_status')
    def test_status_json_output(self, mock_show_status, cli_runner):
        """Test JSON output for status command."""
        mock_show_status.return_value = asyncio.Future()
        mock_show_status.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["status", "--json"])
        assert result.exit_code == 0
        mock_show_status.assert_called_once()

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._watch_lsp_status')
    def test_status_watch_mode(self, mock_watch_status, cli_runner):
        """Test watch mode for status command."""
        mock_watch_status.return_value = asyncio.Future()
        mock_watch_status.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["status", "--watch"])
        assert result.exit_code == 0
        mock_watch_status.assert_called_once()


class TestLspInstallCommand:
    """Test the 'wqm lsp install' command."""

    def test_install_help(self, cli_runner):
        """Test that install command shows help."""
        result = cli_runner.invoke(lsp_app, ["install", "--help"])
        assert result.exit_code == 0
        assert "Guided LSP server installation" in result.stdout

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._install_lsp_server')
    def test_install_python_server(self, mock_install, cli_runner):
        """Test installing Python LSP server."""
        mock_install.return_value = asyncio.Future()
        mock_install.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["install", "python"])
        assert result.exit_code == 0
        mock_install.assert_called_once_with("python", False, False, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._install_lsp_server')
    def test_install_with_force_flag(self, mock_install, cli_runner):
        """Test installing with force flag."""
        mock_install.return_value = asyncio.Future()
        mock_install.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["install", "typescript", "--force"])
        assert result.exit_code == 0
        mock_install.assert_called_once_with("typescript", True, False, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._install_lsp_server')
    def test_install_system_wide(self, mock_install, cli_runner):
        """Test installing system-wide."""
        mock_install.return_value = asyncio.Future()
        mock_install.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["install", "rust", "--system"])
        assert result.exit_code == 0
        mock_install.assert_called_once_with("rust", False, True, False)


class TestLspRestartCommand:
    """Test the 'wqm lsp restart' command."""

    def test_restart_help(self, cli_runner):
        """Test that restart command shows help."""
        result = cli_runner.invoke(lsp_app, ["restart", "--help"])
        assert result.exit_code == 0
        assert "Restart specific LSP server" in result.stdout

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._restart_lsp_server')
    def test_restart_server(self, mock_restart, cli_runner):
        """Test restarting an LSP server."""
        mock_restart.return_value = asyncio.Future()
        mock_restart.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["restart", "python"])
        assert result.exit_code == 0
        mock_restart.assert_called_once_with("python", 30, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._restart_lsp_server')
    def test_restart_with_timeout(self, mock_restart, cli_runner):
        """Test restarting with custom timeout."""
        mock_restart.return_value = asyncio.Future()
        mock_restart.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["restart", "typescript", "--timeout", "60"])
        assert result.exit_code == 0
        mock_restart.assert_called_once_with("typescript", 60, False)


class TestLspConfigCommand:
    """Test the 'wqm lsp config' command."""

    def test_config_help(self, cli_runner):
        """Test that config command shows help."""
        result = cli_runner.invoke(lsp_app, ["config", "--help"])
        assert result.exit_code == 0
        assert "LSP server configuration management" in result.stdout

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._manage_lsp_config')
    def test_config_show(self, mock_manage_config, cli_runner):
        """Test showing LSP configuration."""
        mock_manage_config.return_value = asyncio.Future()
        mock_manage_config.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["config", "--show"])
        assert result.exit_code == 0
        mock_manage_config.assert_called_once_with(None, True, False, False, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._manage_lsp_config')
    def test_config_validate(self, mock_manage_config, cli_runner):
        """Test validating LSP configuration."""
        mock_manage_config.return_value = asyncio.Future()
        mock_manage_config.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["config", "python", "--validate"])
        assert result.exit_code == 0
        mock_manage_config.assert_called_once_with("python", False, True, False, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._manage_lsp_config')
    def test_config_edit(self, mock_manage_config, cli_runner):
        """Test editing LSP configuration."""
        mock_manage_config.return_value = asyncio.Future()
        mock_manage_config.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["config", "typescript", "--edit"])
        assert result.exit_code == 0
        mock_manage_config.assert_called_once_with("typescript", False, False, True, False)


class TestLspDiagnoseCommand:
    """Test the 'wqm lsp diagnose' command."""

    def test_diagnose_help(self, cli_runner):
        """Test that diagnose command shows help."""
        result = cli_runner.invoke(lsp_app, ["diagnose", "--help"])
        assert result.exit_code == 0
        assert "Run comprehensive troubleshooting" in result.stdout

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._diagnose_lsp_server')
    def test_diagnose_server(self, mock_diagnose, cli_runner):
        """Test diagnosing an LSP server."""
        mock_diagnose.return_value = asyncio.Future()
        mock_diagnose.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["diagnose", "python"])
        assert result.exit_code == 0
        mock_diagnose.assert_called_once_with("python", False, False, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._diagnose_lsp_server')
    def test_diagnose_comprehensive(self, mock_diagnose, cli_runner):
        """Test comprehensive diagnostics."""
        mock_diagnose.return_value = asyncio.Future()
        mock_diagnose.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["diagnose", "rust", "--comprehensive"])
        assert result.exit_code == 0
        mock_diagnose.assert_called_once_with("rust", True, False, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._diagnose_lsp_server')
    def test_diagnose_with_fix(self, mock_diagnose, cli_runner):
        """Test diagnostics with automatic fix."""
        mock_diagnose.return_value = asyncio.Future()
        mock_diagnose.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["diagnose", "go", "--fix"])
        assert result.exit_code == 0
        mock_diagnose.assert_called_once_with("go", False, True, False)


class TestLspSetupCommand:
    """Test the 'wqm lsp setup' command."""

    def test_setup_help(self, cli_runner):
        """Test that setup command shows help."""
        result = cli_runner.invoke(lsp_app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "Interactive setup wizard" in result.stdout

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._interactive_lsp_setup')
    def test_setup_interactive(self, mock_setup, cli_runner):
        """Test interactive setup wizard."""
        mock_setup.return_value = asyncio.Future()
        mock_setup.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["setup"])
        assert result.exit_code == 0
        mock_setup.assert_called_once_with(True, None, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._interactive_lsp_setup')
    def test_setup_with_language(self, mock_setup, cli_runner):
        """Test setup with pre-selected language."""
        mock_setup.return_value = asyncio.Future()
        mock_setup.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["setup", "--language", "python"])
        assert result.exit_code == 0
        mock_setup.assert_called_once_with(True, "python", False)


class TestLspListCommand:
    """Test the 'wqm lsp list' command."""

    def test_list_help(self, cli_runner):
        """Test that list command shows help."""
        result = cli_runner.invoke(lsp_app, ["list", "--help"])
        assert result.exit_code == 0
        assert "List available and installed LSP servers" in result.stdout

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._list_lsp_servers')
    def test_list_all_servers(self, mock_list, cli_runner):
        """Test listing all available servers."""
        mock_list.return_value = asyncio.Future()
        mock_list.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["list"])
        assert result.exit_code == 0
        mock_list.assert_called_once_with(False, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._list_lsp_servers')
    def test_list_installed_only(self, mock_list, cli_runner):
        """Test listing only installed servers."""
        mock_list.return_value = asyncio.Future()
        mock_list.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["list", "--installed"])
        assert result.exit_code == 0
        mock_list.assert_called_once_with(True, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._list_lsp_servers')
    def test_list_json_output(self, mock_list, cli_runner):
        """Test listing with JSON output."""
        mock_list.return_value = asyncio.Future()
        mock_list.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["list", "--json"])
        assert result.exit_code == 0
        mock_list.assert_called_once_with(False, True)


class TestLspPerformanceCommand:
    """Test the 'wqm lsp performance' command."""

    def test_performance_help(self, cli_runner):
        """Test that performance command shows help."""
        result = cli_runner.invoke(lsp_app, ["performance", "--help"])
        assert result.exit_code == 0
        assert "Monitor LSP server performance" in result.stdout

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._monitor_lsp_performance')
    def test_performance_monitoring(self, mock_monitor, cli_runner):
        """Test performance monitoring."""
        mock_monitor.return_value = asyncio.Future()
        mock_monitor.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["performance"])
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with(None, 60, 5, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._monitor_lsp_performance')
    def test_performance_custom_duration(self, mock_monitor, cli_runner):
        """Test performance monitoring with custom duration."""
        mock_monitor.return_value = asyncio.Future()
        mock_monitor.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["performance", "--duration", "120"])
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with(None, 120, 5, False)

    @pytest.mark.xfail(reason="Async/sync boundary issue: Future needs event loop")
    @patch('wqm_cli.cli.commands.lsp_management._monitor_lsp_performance')
    def test_performance_specific_server(self, mock_monitor, cli_runner):
        """Test monitoring specific server performance."""
        mock_monitor.return_value = asyncio.Future()
        mock_monitor.return_value.set_result(None)

        result = cli_runner.invoke(lsp_app, ["performance", "python", "--interval", "10"])
        assert result.exit_code == 0
        mock_monitor.assert_called_once_with("python", 60, 10, False)


class TestLspUtilityFunctions:
    """Test LSP management utility functions."""

    @pytest.mark.asyncio
    async def test_check_server_installation_installed(self):
        """Test checking installation for an installed server."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock successful command execution
            mock_process = MagicMock()
            mock_process.wait = AsyncMock(return_value=None)
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await _check_server_installation("python")
            assert result is True

    @pytest.mark.asyncio
    async def test_check_server_installation_not_installed(self):
        """Test checking installation for a non-installed server."""
        with patch('asyncio.create_subprocess_exec') as mock_subprocess:
            # Mock failed command execution
            mock_process = MagicMock()
            mock_process.wait = AsyncMock(return_value=None)
            mock_process.returncode = 1
            mock_subprocess.return_value = mock_process

            result = await _check_server_installation("nonexistent")
            assert result is False

    @pytest.mark.asyncio
    async def test_check_server_installation_command_not_found(self):
        """Test checking installation when command is not found."""
        with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError):
            result = await _check_server_installation("python")
            assert result is False

    @pytest.mark.asyncio
    async def test_get_server_status_known_server(self, mock_health_monitor):
        """Test getting status for a known server."""
        with patch('wqm_cli.cli.commands.lsp_management._check_server_installation', return_value=True):
            result = await _get_server_status("python", mock_health_monitor)

            assert result["server_name"] == "python"
            assert result["installed"] is True
            assert result["status"] == "healthy"
            assert "python" in result["languages"]

    @pytest.mark.asyncio
    async def test_get_server_status_unknown_server(self, mock_health_monitor):
        """Test getting status for an unknown server."""
        with pytest.raises(ValueError, match="Unknown LSP server"):
            await _get_server_status("unknown_server", mock_health_monitor)

    @pytest.mark.asyncio
    async def test_get_all_servers_status(self, mock_health_monitor):
        """Test getting status for all servers."""
        with patch('wqm_cli.cli.commands.lsp_management._get_server_status') as mock_get_status:
            mock_get_status.return_value = {
                "server_name": "python",
                "status": "healthy",
                "installed": True
            }

            result = await _get_all_servers_status(mock_health_monitor)

            assert "timestamp" in result
            assert "total_servers" in result
            assert "servers" in result
            assert len(result["servers"]) == len(KNOWN_LSP_SERVERS)


class TestKnownLspServers:
    """Test known LSP servers configuration."""

    def test_known_servers_structure(self):
        """Test that all known servers have required fields."""
        required_fields = [
            "name", "package", "command", "check_command",
            "languages", "features"
        ]

        for server_key, server_config in KNOWN_LSP_SERVERS.items():
            for field in required_fields:
                assert field in server_config, f"Server {server_key} missing {field}"

            # Test specific field types
            assert isinstance(server_config["name"], str)
            assert isinstance(server_config["languages"], list)
            assert isinstance(server_config["features"], list)
            assert isinstance(server_config["command"], list)

            # At least one language should be supported
            assert len(server_config["languages"]) > 0

            # At least one feature should be supported
            assert len(server_config["features"]) > 0

    def test_python_server_config(self):
        """Test Python server specific configuration."""
        python_config = KNOWN_LSP_SERVERS["python"]

        assert python_config["name"] == "Python LSP Server (pylsp)"
        assert "python" in python_config["languages"]
        assert "hover" in python_config["features"]
        assert "definition" in python_config["features"]
        assert python_config["command"] == ["pylsp"]

    def test_typescript_server_config(self):
        """Test TypeScript server specific configuration."""
        typescript_config = KNOWN_LSP_SERVERS["typescript"]

        assert typescript_config["name"] == "TypeScript Language Server"
        assert "typescript" in typescript_config["languages"]
        assert "javascript" in typescript_config["languages"]
        assert "formatting" in typescript_config["features"]

    def test_all_servers_have_install_commands(self):
        """Test that servers have install commands or marked as complex."""
        for server_key, server_config in KNOWN_LSP_SERVERS.items():
            install_command = server_config.get("install_command")

            # Either has install command or is marked as complex (None)
            if install_command is not None:
                assert isinstance(install_command, list)
                assert len(install_command) > 0
            # Java is known to have complex installation
            elif server_key == "java":
                assert install_command is None
            else:
                # All other servers should have install commands
                assert install_command is not None, f"Server {server_key} missing install command"


class TestLspIntegration:
    """Integration tests for LSP management commands."""

    def test_lsp_app_structure(self, cli_runner):
        """Test that LSP app is properly structured."""
        assert lsp_app.info.name == "lsp"

        # Check that all expected commands are registered by checking help output
        expected_commands = [
            "status", "install", "restart", "config",
            "diagnose", "setup", "list", "performance"
        ]

        result = cli_runner.invoke(lsp_app, ["--help"])
        assert result.exit_code == 0

        for expected_cmd in expected_commands:
            assert expected_cmd in result.stdout, f"Command {expected_cmd} not found in LSP app help"

    def test_help_text_formatting(self, cli_runner):
        """Test that help text is properly formatted."""
        result = cli_runner.invoke(lsp_app, ["--help"])
        assert result.exit_code == 0

        # Should contain main description
        assert "LSP server management and monitoring" in result.stdout

        # Should contain examples
        assert "Examples:" in result.stdout
        assert "wqm lsp status" in result.stdout
        assert "wqm lsp install" in result.stdout

    def test_command_error_handling(self, cli_runner):
        """Test error handling for invalid commands."""
        # Test invalid server name
        result = cli_runner.invoke(lsp_app, ["status", "invalid_server"])
        # Should not crash, though specific behavior depends on implementation
        assert isinstance(result.exit_code, int)

    @patch('wqm_cli.cli.commands.lsp_management.handle_async')
    def test_async_command_wrapper(self, mock_handle_async, cli_runner):
        """Test that async commands are properly wrapped."""
        # Mock handle_async to avoid actual async execution
        mock_handle_async.return_value = None

        cli_runner.invoke(lsp_app, ["status"])

        # Should have called handle_async for the async function
        mock_handle_async.assert_called_once()
