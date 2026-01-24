"""Unit tests for CLI service management commands.

Tests all service management CLI commands using CliRunner:
- Service install/uninstall commands
- Start/stop/restart/status commands
- Service logs retrieval
- Cross-platform support (macOS/Linux)
- Error handling and exit codes
- User interaction scenarios

Test coverage:
- MemexdServiceManager class and all methods
- All CLI command entry points with typer.testing.CliRunner
- Argument parsing and validation
- Error handling and proper exit codes
- Mock external dependencies (subprocess, file system, etc.)
"""

import asyncio
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

from wqm_cli.cli.binary_security import BinarySecurityError


class TestMemexdServiceManager:
    """Test MemexdServiceManager class methods."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('platform.system')
    @patch.object(Path, 'exists')
    @patch('os.access')
    def test_service_manager_init_success(self, mock_access, mock_exists, mock_system):
        """Test successful MemexdServiceManager initialization."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        mock_access.return_value = True

        from wqm_cli.cli.commands.service import MemexdServiceManager

        manager = MemexdServiceManager()
        assert manager.system == "darwin"
        assert manager.service_name == "memexd"
        assert manager.service_id == "com.workspace-qdrant.memexd"
        # Binary path should be OS-specific user path
        assert ".local/bin" in str(manager.memexd_binary)
        assert manager.memexd_binary.name == "memexd"

    @patch('platform.system')
    @patch.object(Path, 'exists')
    def test_service_manager_init_binary_not_found(self, mock_exists, mock_system):
        """Test MemexdServiceManager initialization with missing binary."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = False

        from wqm_cli.cli.commands.service import MemexdServiceManager

        with pytest.raises(FileNotFoundError, match="memexd binary not found"):
            MemexdServiceManager()

        # Error message should include helpful installation guidance
        try:
            MemexdServiceManager()
        except FileNotFoundError as e:
            assert "Preferred installation path:" in str(e)
            assert "wqm service install --build" in str(e)

    @patch('platform.system')
    @patch.object(Path, 'exists')
    @patch('os.access')
    def test_service_manager_init_binary_not_executable(self, mock_access, mock_exists, mock_system):
        """Test MemexdServiceManager initialization with non-executable binary."""
        mock_system.return_value = "Darwin"
        mock_exists.return_value = True
        mock_access.return_value = False

        from wqm_cli.cli.commands.service import MemexdServiceManager

        with pytest.raises(BinarySecurityError, match="not executable"):
            MemexdServiceManager()

    @patch('platform.system')
    @patch.object(Path, 'exists', return_value=True)
    @patch('os.access')
    def test_get_config_path(self, mock_access, mock_exists, mock_system):
        """Test configuration path resolution."""
        mock_system.return_value = "Darwin"
        mock_access.return_value = True

        from wqm_cli.cli.commands.service import MemexdServiceManager

        manager = MemexdServiceManager()
        config_path = manager.get_config_path()

        assert config_path.name in ["workspace_qdrant_config.yaml", "config.yaml"]
        assert ".config/workspace-qdrant" in str(config_path)

    @patch('platform.system')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access')
    def test_get_log_path(self, mock_access, mock_exists, mock_system):
        """Test log path resolution."""
        mock_system.return_value = "Darwin"
        mock_access.return_value = True

        from wqm_cli.cli.commands.service import MemexdServiceManager

        manager = MemexdServiceManager()
        log_path = manager.get_log_path()

        assert log_path.name == "memexd.log"
        assert ".local/var/log/workspace-qdrant" in str(log_path)

    @patch('platform.system')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access')
    def test_get_pid_path(self, mock_access, mock_exists, mock_system):
        """Test PID path resolution."""
        mock_system.return_value = "Darwin"
        mock_access.return_value = True

        from wqm_cli.cli.commands.service import MemexdServiceManager

        manager = MemexdServiceManager()
        pid_path = manager.get_pid_path()

        assert pid_path.name == "memexd.pid"
        assert ".local/var/run/workspace-qdrant" in str(pid_path)


class TestServiceInstallCommand:
    """Test service install command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_install_service_success(self, mock_service_manager):
        """Test successful service installation."""
        from wqm_cli.cli.commands.service import service_app

        # Mock successful installation
        mock_service_manager.install_service = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "binary_path": "/usr/local/bin/memexd",
            "message": "Service installed successfully"
        })

        result = self.runner.invoke(service_app, ["install"])

        assert result.exit_code == 0
        assert "Service installed successfully" in result.stdout
        assert "✅" in result.stdout
        mock_service_manager.install_service.assert_called_once_with(auto_start=True)

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_install_service_no_auto_start(self, mock_service_manager):
        """Test service installation without auto-start."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.install_service = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "binary_path": "/usr/local/bin/memexd",
            "message": "Service installed successfully"
        })

        result = self.runner.invoke(service_app, ["install", "--no-auto-start"])

        assert result.exit_code == 0
        mock_service_manager.install_service.assert_called_once_with(auto_start=False)

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_install_service_failure(self, mock_service_manager):
        """Test service installation failure."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.install_service = AsyncMock(return_value={
            "success": False,
            "error": "memexd binary not found",
            "suggestion": "Please install memexd first"
        })

        result = self.runner.invoke(service_app, ["install"])

        assert result.exit_code == 1
        assert "Installation failed" in result.stdout
        assert "❌" in result.stdout
        assert "memexd binary not found" in result.stdout
        assert "Please install memexd first" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_install_service_exception(self, mock_service_manager):
        """Test service installation with exception."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.install_service = AsyncMock(side_effect=Exception("Unexpected error"))

        result = self.runner.invoke(service_app, ["install"])

        assert result.exit_code == 1
        assert "Unexpected error" in result.stdout


class TestServiceUninstallCommand:
    """Test service uninstall command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_uninstall_service_success(self, mock_service_manager):
        """Test successful service uninstallation."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.uninstall_service = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "message": "Service uninstalled successfully"
        })

        result = self.runner.invoke(service_app, ["uninstall"])

        assert result.exit_code == 0
        assert "Service uninstalled successfully" in result.stdout
        assert "✅" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_uninstall_service_failure(self, mock_service_manager):
        """Test service uninstallation failure."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.uninstall_service = AsyncMock(return_value={
            "success": False,
            "error": "Service not found"
        })

        result = self.runner.invoke(service_app, ["uninstall"])

        assert result.exit_code == 1
        assert "Uninstall failed" in result.stdout
        assert "❌" in result.stdout


class TestServiceStartCommand:
    """Test service start command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_start_service_success(self, mock_service_manager):
        """Test successful service start."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.start_service = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "message": "Service started successfully"
        })

        result = self.runner.invoke(service_app, ["start"])

        assert result.exit_code == 0
        assert "Service started successfully" in result.stdout
        assert "✅" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_start_service_failure_with_suggestion(self, mock_service_manager):
        """Test service start failure with suggestion."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.start_service = AsyncMock(return_value={
            "success": False,
            "error": "Service not installed",
            "suggestion": "Run 'wqm service install' first"
        })

        result = self.runner.invoke(service_app, ["start"])

        assert result.exit_code == 1
        assert "Failed to start service" in result.stdout
        assert "Service not installed" in result.stdout
        assert "Run 'wqm service install' first" in result.stdout


class TestServiceStopCommand:
    """Test service stop command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_stop_service_success(self, mock_service_manager):
        """Test successful service stop."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.stop_service = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "message": "Service stopped successfully"
        })

        result = self.runner.invoke(service_app, ["stop"])

        assert result.exit_code == 0
        assert "Service stopped successfully" in result.stdout
        assert "✅" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_stop_service_failure(self, mock_service_manager):
        """Test service stop failure."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.stop_service = AsyncMock(return_value={
            "success": False,
            "error": "Service not running"
        })

        result = self.runner.invoke(service_app, ["stop"])

        assert result.exit_code == 1
        assert "Failed to stop service" in result.stdout
        assert "❌" in result.stdout


class TestServiceRestartCommand:
    """Test service restart command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_restart_service_success(self, mock_service_manager):
        """Test successful service restart."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.restart_service = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "message": "Service restarted successfully"
        })

        result = self.runner.invoke(service_app, ["restart"])

        assert result.exit_code == 0
        assert "Service restarted successfully" in result.stdout
        assert "✅" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_restart_service_failure(self, mock_service_manager):
        """Test service restart failure."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.restart_service = AsyncMock(return_value={
            "success": False,
            "error": "Failed to stop service"
        })

        result = self.runner.invoke(service_app, ["restart"])

        assert result.exit_code == 1
        assert "Failed to restart service" in result.stdout
        assert "❌" in result.stdout


class TestServiceStatusCommand:
    """Test service status command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_status_service_running(self, mock_service_manager):
        """Test status command with running service."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_status = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "status": "running",
            "running": True,
            "pid": 1234,
            "platform": "macOS"
        })
        mock_service_manager.memexd_binary = Path("/usr/local/bin/memexd")

        result = self.runner.invoke(service_app, ["status"])

        assert result.exit_code == 0
        assert "Running" in result.stdout
        assert "1234" in result.stdout
        assert "macOS" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_status_service_not_running(self, mock_service_manager):
        """Test status command with stopped service."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_status = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "status": "loaded",
            "running": False,
            "platform": "macOS"
        })
        mock_service_manager.memexd_binary = Path("/usr/local/bin/memexd")

        result = self.runner.invoke(service_app, ["status"])

        assert result.exit_code == 0
        assert "Loaded (Not Running)" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_status_service_not_installed(self, mock_service_manager):
        """Test status command with uninstalled service."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_status = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "status": "not_installed",
            "running": False,
            "platform": "macOS"
        })
        mock_service_manager.memexd_binary = Path("/usr/local/bin/memexd")

        result = self.runner.invoke(service_app, ["status"])

        assert result.exit_code == 0
        assert "Not Installed" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_status_verbose_mode(self, mock_service_manager):
        """Test status command with verbose flag."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_status = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "status": "running",
            "running": True,
            "pid": 1234,
            "platform": "macOS"
        })
        mock_service_manager.memexd_binary = Path("/usr/local/bin/memexd")

        result = self.runner.invoke(service_app, ["status", "--verbose"])

        assert result.exit_code == 0
        assert "Service is running and processing workspace events" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_status_service_failure(self, mock_service_manager):
        """Test status command failure."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_status = AsyncMock(return_value={
            "success": False,
            "error": "Cannot connect to service manager"
        })

        result = self.runner.invoke(service_app, ["status"])

        assert result.exit_code == 1
        assert "Failed to get service status" in result.stdout
        assert "❌" in result.stdout


class TestServiceLogsCommand:
    """Test service logs command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_logs_service_success(self, mock_service_manager):
        """Test successful logs retrieval."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_logs = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "logs": [
                "2024-01-15 10:30:00 INFO Service started",
                "2024-01-15 10:30:01 INFO Processing events",
                "2024-01-15 10:30:02 INFO All systems operational"
            ]
        })

        result = self.runner.invoke(service_app, ["logs"])

        assert result.exit_code == 0
        assert "Recent logs" in result.stdout
        assert "Service started" in result.stdout
        assert "Processing events" in result.stdout
        mock_service_manager.get_service_logs.assert_called_once_with(50)

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_logs_service_custom_lines(self, mock_service_manager):
        """Test logs retrieval with custom line count."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_logs = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "logs": ["Log line 1", "Log line 2"]
        })

        result = self.runner.invoke(service_app, ["logs", "--lines", "100"])

        assert result.exit_code == 0
        mock_service_manager.get_service_logs.assert_called_once_with(100)

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_logs_service_no_logs(self, mock_service_manager):
        """Test logs retrieval with no available logs."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_logs = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.memexd",
            "logs": []
        })

        result = self.runner.invoke(service_app, ["logs"])

        assert result.exit_code == 0
        assert "No logs available" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_logs_service_failure(self, mock_service_manager):
        """Test logs retrieval failure."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_logs = AsyncMock(return_value={
            "success": False,
            "error": "Log file not accessible"
        })

        result = self.runner.invoke(service_app, ["logs"])

        assert result.exit_code == 1
        assert "Failed to get service logs" in result.stdout
        assert "❌" in result.stdout


class TestServiceAppHelp:
    """Test service app help and argument parsing."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    def test_service_app_help(self):
        """Test service app shows help when no command given."""
        from wqm_cli.cli.commands.service import service_app

        result = self.runner.invoke(service_app, [])

        # With no_args_is_help=True, Typer shows help but exits with code 2
        # (indicating missing required command)
        assert result.exit_code == 2
        assert "Usage:" in result.output
        assert "install" in result.output
        assert "start" in result.output
        assert "stop" in result.output
        assert "status" in result.output

    def test_service_app_help_flag(self):
        """Test service app help flag."""
        from wqm_cli.cli.commands.service import service_app

        result = self.runner.invoke(service_app, ["--help"])

        assert result.exit_code == 0
        assert "User service management" in result.stdout

    def test_invalid_command(self):
        """Test invalid service command."""
        from wqm_cli.cli.commands.service import service_app

        result = self.runner.invoke(service_app, ["invalid-command"])

        assert result.exit_code != 0

    def test_install_help(self):
        """Test install command help."""
        from wqm_cli.cli.commands.service import service_app

        result = self.runner.invoke(service_app, ["install", "--help"])

        assert result.exit_code == 0
        assert "Install the workspace daemon" in result.stdout
        assert "--auto-start" in result.stdout

    def test_status_help(self):
        """Test status command help."""
        from wqm_cli.cli.commands.service import service_app

        result = self.runner.invoke(service_app, ["status", "--help"])

        assert result.exit_code == 0
        assert "Show workspace daemon service status" in result.stdout
        assert "--verbose" in result.stdout

    def test_logs_help(self):
        """Test logs command help."""
        from wqm_cli.cli.commands.service import service_app

        result = self.runner.invoke(service_app, ["logs", "--help"])

        assert result.exit_code == 0
        assert "Show workspace daemon service logs" in result.stdout
        assert "--lines" in result.stdout


class TestServiceManagerAsyncMethods:
    """Test async methods of MemexdServiceManager with proper mocking."""

    def setup_method(self):
        """Set up test environment."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def teardown_method(self):
        """Clean up test environment."""
        self.loop.close()

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_install_macos_service_success(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test successful macOS service installation."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock subprocess success
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Success", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()
        result = await manager.install_service(auto_start=True)

        assert result["success"] is True
        assert "service_id" in result
        assert result["auto_start"] is True

    @patch('platform.system', return_value='Linux')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_install_linux_service_success(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test successful Linux service installation."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock subprocess success
        mock_process = AsyncMock()
        mock_process.wait.return_value = 0
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()
        result = await manager.install_service(auto_start=True)

        assert result["success"] is True
        assert "service_name" in result

    @patch('platform.system', return_value='Windows')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    async def test_install_unsupported_platform(self, mock_access, mock_exists, mock_system):
        """Test installation on unsupported platform."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        manager = MemexdServiceManager()
        result = await manager.install_service()

        assert result["success"] is False
        assert "Windows support not implemented" in result["error"]

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_get_service_status_running(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test getting status of running service."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock launchctl list output
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b'"PID" = 1234;', b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()
        result = await manager.get_service_status()

        assert result["success"] is True
        assert result["running"] is True
        assert result["pid"] == 1234

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_stop_service_with_force_kill(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test stopping service that requires force kill."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock subprocess calls for stop and force kill
        responses = [
            # First call: launchctl kill TERM (graceful stop)
            (b"", b""),
            # Second call: pgrep -f memexd (check if still running)
            (b"1234\n", b""),
            # Third call: launchctl kill KILL (force kill)
            (b"", b""),
            # Fourth call: final pgrep check (should be empty)
            (b"", b""),
            # Fifth call: launchctl list (final status check)
            (b'"PID" = -;', b"")
        ]

        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            mock_process = AsyncMock()
            stdout, stderr = responses[call_count]
            call_count += 1
            mock_process.communicate.return_value = (stdout, stderr)
            mock_process.returncode = 0
            return mock_process

        mock_subprocess.side_effect = side_effect

        manager = MemexdServiceManager()
        result = await manager.stop_service()

        assert result["success"] is True
        assert "method" in result


class TestServiceManagerCrossPlatform:
    """Test cross-platform service manager functionality."""

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    def test_macos_service_paths(self, mock_access, mock_exists, mock_system):
        """Test macOS-specific service paths."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        manager = MemexdServiceManager()

        # Test service paths are correctly set for macOS
        assert manager.system == "darwin"
        assert manager.service_id == "com.workspace-qdrant.memexd"

    @patch('platform.system', return_value='Linux')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    def test_linux_service_paths(self, mock_access, mock_exists, mock_system):
        """Test Linux-specific service paths."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        manager = MemexdServiceManager()

        # Test service paths are correctly set for Linux
        assert manager.system == "linux"
        assert manager.service_name == "memexd"

    @pytest.mark.xfail(reason="Path.exists mock cannot receive path argument - needs rewrite with temp directory")
    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    def test_config_path_precedence(self, mock_access, mock_exists, mock_system):
        """Test configuration path precedence."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        manager = MemexdServiceManager()

        # Mock workspace_qdrant_config.yaml exists
        with patch('pathlib.Path.exists') as mock_path_exists:
            mock_path_exists.return_value = True
            config_path = manager.get_config_path()
            assert "workspace_qdrant_config.yaml" in str(config_path)

        # Mock only config.yaml exists - this test needs rewrite with actual files
        with patch('pathlib.Path.exists') as mock_path_exists:
            mock_path_exists.return_value = False
            config_path = manager.get_config_path()
            assert "config.yaml" in str(config_path)
