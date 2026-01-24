"""Enhanced unit tests for CLI service management commands.

Comprehensive test coverage for service management including:
- Service lifecycle operations (install, start, stop, restart, status, logs, uninstall)
- Edge cases: service already running, permission issues, daemon not found
- Platform-specific variations (Linux systemd vs macOS launchd)
- Binary build functionality with --build flag
- Mock systemctl and launchctl commands
- Error handling and recovery scenarios

This test suite ensures complete coverage of wqm service commands.
"""

import asyncio
import os
import platform
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from typer.testing import CliRunner

# Add the project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))


class TestServiceInstallWithBuild:
    """Test service install command with --build flag."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_install_service_with_build_flag(self, mock_service_manager):
        """Test service installation with --build flag."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.install_service = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.workspace-qdrant-daemon",
            "binary_path": "/Users/test/.local/bin/memexd",
            "message": "Service installed successfully"
        })

        result = self.runner.invoke(service_app, ["install", "--build"])

        assert result.exit_code == 0
        assert "Service installed successfully" in result.stdout
        mock_service_manager.install_service.assert_called_once_with(auto_start=True, build=True)

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_install_with_build_failure(self, mock_service_manager):
        """Test service installation when build fails."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.install_service = AsyncMock(return_value={
            "success": False,
            "error": "Cargo build failed: error compiling",
            "suggestion": "Check rust toolchain installation"
        })

        result = self.runner.invoke(service_app, ["install", "--build"])

        assert result.exit_code == 1
        assert "Installation failed" in result.stdout
        assert "Cargo build failed" in result.stdout


class TestServicePermissionErrors:
    """Test service commands with permission errors."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_install_permission_denied(self, mock_service_manager):
        """Test service installation with permission denied."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.install_service = AsyncMock(return_value={
            "success": False,
            "error": "No write permission to /Users/test/Library/LaunchAgents",
            "suggestion": "Check permissions: ls -la /Users/test/Library/LaunchAgents"
        })

        result = self.runner.invoke(service_app, ["install"])

        assert result.exit_code == 1
        assert "No write permission" in result.stdout
        assert "Check permissions" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_start_permission_denied(self, mock_service_manager):
        """Test service start with permission denied."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.start_service = AsyncMock(return_value={
            "success": False,
            "error": "Failed to start service: Permission denied",
            "suggestion": "Check service permissions"
        })

        result = self.runner.invoke(service_app, ["start"])

        assert result.exit_code == 1
        assert "Permission denied" in result.stdout


class TestServiceAlreadyRunning:
    """Test service commands when service is already running."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_start_already_running(self, mock_service_manager):
        """Test starting an already running service."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.start_service = AsyncMock(return_value={
            "success": False,
            "error": "Service is already running with PID 1234",
            "suggestion": "Use 'wqm service restart' to restart the service"
        })

        result = self.runner.invoke(service_app, ["start"])

        assert result.exit_code == 1
        assert "already running" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_install_already_installed(self, mock_service_manager):
        """Test installing when service is already installed."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.install_service = AsyncMock(return_value={
            "success": False,
            "error": "Service already installed",
            "suggestion": "Use 'wqm service uninstall' first or 'wqm service restart' to restart"
        })

        result = self.runner.invoke(service_app, ["install"])

        assert result.exit_code == 1
        assert "already installed" in result.stdout


class TestServiceNotFound:
    """Test service commands when daemon is not found."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_start_service_not_installed(self, mock_service_manager):
        """Test starting a service that's not installed."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.start_service = AsyncMock(return_value={
            "success": False,
            "error": "Service not installed",
            "suggestion": "Run 'wqm service install' first"
        })

        result = self.runner.invoke(service_app, ["start"])

        assert result.exit_code == 1
        assert "Service not installed" in result.stdout
        assert "Run 'wqm service install' first" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_stop_service_not_found(self, mock_service_manager):
        """Test stopping a service that doesn't exist."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.stop_service = AsyncMock(return_value={
            "success": False,
            "error": "No workspace-qdrant service found",
            "suggestion": "Service may not be installed"
        })

        result = self.runner.invoke(service_app, ["stop"])

        assert result.exit_code == 1
        assert "No workspace-qdrant service found" in result.stdout


class TestServiceBinaryBuild:
    """Test binary building and installation."""

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=False)
    @patch('asyncio.create_subprocess_exec')
    async def test_install_binary_from_source_success(self, mock_subprocess, mock_exists, mock_system):
        """Test successful binary build from source."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock successful cargo build
        mock_build_process = AsyncMock()
        mock_build_process.communicate.return_value = (b"Compiling memexd", b"")
        mock_build_process.returncode = 0
        mock_subprocess.return_value = mock_build_process

        # Create a temporary manager without validation
        with patch.object(MemexdServiceManager, 'validate_binary'):
            manager = MemexdServiceManager()

            # Mock the paths
            with patch('pathlib.Path.exists', return_value=True):
                with patch('shutil.copy2'):
                    with patch('os.chmod'):
                        result = await manager.install_binary_from_source()

                        assert result["success"] is True
                        assert "install_path" in result
                        assert "memexd binary built and installed successfully" in result["message"]

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)  # Directory must exist for build to be attempted
    @patch('asyncio.create_subprocess_exec')
    async def test_install_binary_from_source_build_failure(self, mock_subprocess, mock_exists, mock_system):
        """Test binary build failure."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock failed cargo build
        mock_build_process = AsyncMock()
        mock_build_process.communicate.return_value = (b"", b"error: could not compile")
        mock_build_process.returncode = 1
        mock_subprocess.return_value = mock_build_process

        with patch.object(MemexdServiceManager, 'validate_binary'):
            manager = MemexdServiceManager()
            result = await manager.install_binary_from_source()

            assert result["success"] is False
            assert "Cargo build failed" in result["error"]

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=False)
    async def test_install_binary_source_not_found(self, mock_exists, mock_system):
        """Test binary build when src/rust/daemon directory not found."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        with patch.object(MemexdServiceManager, 'validate_binary'):
            manager = MemexdServiceManager()
            result = await manager.install_binary_from_source()

            assert result["success"] is False
            assert "src/rust/daemon directory not found" in result["error"]


class TestPlatformSpecificMacOS:
    """Test macOS-specific service operations with launchctl mocking."""

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_macos_install_bootstrap_success(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test macOS service installation with launchctl bootstrap."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock successful bootstrap
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()

        with patch('pathlib.Path.write_text'):
            with patch('pathlib.Path.mkdir'):
                result = await manager._install_macos_service(auto_start=True)

                assert result["success"] is True
                assert "service_id" in result
                assert result["auto_start"] is True

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_macos_install_already_loaded(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test macOS installation when service already loaded."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock "service already loaded" error (which is actually OK)
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"service already loaded")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()

        with patch('pathlib.Path.write_text'):
            with patch('pathlib.Path.mkdir'):
                result = await manager._install_macos_service(auto_start=True)

                # Should succeed even with "already loaded" error
                assert result["success"] is True

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_macos_start_with_kickstart(self, mock_subprocess, mock_access, mock_is_file, mock_exists, mock_system):
        """Test macOS service start with launchctl kickstart."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock successful kickstart
        call_count = [0]

        def subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_process = AsyncMock()

            if call_count[0] == 1:  # launchctl list check
                mock_process.communicate.return_value = (b"", b"")
                mock_process.returncode = 0
            elif call_count[0] == 2:  # kickstart command
                mock_process.communicate.return_value = (b"", b"")
                mock_process.returncode = 0
            elif call_count[0] == 3:  # status check
                mock_process.communicate.return_value = (b'"PID" = 1234;', b"")
                mock_process.returncode = 0
            else:  # any other calls
                mock_process.communicate.return_value = (b"", b"")
                mock_process.returncode = 0

            return mock_process

        mock_subprocess.side_effect = subprocess_side_effect

        manager = MemexdServiceManager()

        with patch('pathlib.Path.glob', return_value=[Path("/test/com.workspace-qdrant.plist")]):
            result = await manager._start_macos_service()

            assert result["success"] is True
            assert "service_id" in result

    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_file', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_macos_stop_with_force_kill(self, mock_subprocess, mock_access, mock_is_file, mock_exists, mock_system):
        """Test macOS service stop with force kill for memexd shutdown bug."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        call_count = [0]

        def subprocess_side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_process = AsyncMock()

            if call_count[0] == 1:  # Initial kill TERM
                mock_process.communicate.return_value = (b"", b"")
                mock_process.returncode = 0
            elif call_count[0] == 2:  # pgrep check (still running)
                mock_process.communicate.return_value = (b"1234", b"")
                mock_process.returncode = 0
            elif call_count[0] == 3:  # Force kill KILL
                mock_process.communicate.return_value = (b"", b"")
                mock_process.returncode = 0
            elif call_count[0] == 4:  # Final pgrep check (stopped)
                mock_process.communicate.return_value = (b"", b"")
                mock_process.returncode = 1
            else:  # Final status check
                mock_process.communicate.return_value = (b'"PID" = -;', b"")
                mock_process.returncode = 0

            return mock_process

        mock_subprocess.side_effect = subprocess_side_effect

        manager = MemexdServiceManager()

        with patch('pathlib.Path.glob', return_value=[Path("/test/com.workspace-qdrant.plist")]):
            result = await manager._stop_macos_service()

            assert result["success"] is True
            assert "method" in result


class TestPlatformSpecificLinux:
    """Test Linux-specific service operations with systemctl mocking."""

    @patch('platform.system', return_value='Linux')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_linux_install_with_daemon_reload(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test Linux service installation with systemctl daemon-reload."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock successful systemctl commands
        mock_process = AsyncMock()
        mock_process.wait.return_value = 0
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()

        with patch('pathlib.Path.write_text'):
            with patch('pathlib.Path.mkdir'):
                result = await manager._install_linux_service(auto_start=True)

                assert result["success"] is True
                assert "service_name" in result

    @patch('platform.system', return_value='Linux')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_linux_start_service(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test Linux service start with systemctl start."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock successful start
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()
        result = await manager._start_linux_service()

        assert result["success"] is True
        assert "service_name" in result

    @patch('platform.system', return_value='Linux')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_linux_stop_service(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test Linux service stop with systemctl stop."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock successful stop
        mock_process = AsyncMock()
        mock_process.wait.return_value = 0
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()
        result = await manager._stop_linux_service()

        assert result["success"] is True
        assert "service_name" in result

    @patch('platform.system', return_value='Linux')
    @patch('pathlib.Path.exists', return_value=True)
    @patch('os.access', return_value=True)
    @patch('asyncio.create_subprocess_exec')
    async def test_linux_status_active(self, mock_subprocess, mock_access, mock_exists, mock_system):
        """Test Linux service status when active."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        # Mock active status
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"active", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        manager = MemexdServiceManager()
        result = await manager._get_linux_service_status()

        assert result["success"] is True
        assert result["running"] is True
        assert result["status"] == "active"


class TestServiceStatusVariations:
    """Test various service status scenarios."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_status_service_loaded_but_not_running(self, mock_service_manager):
        """Test status when service is loaded but not running."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_status = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.workspace-qdrant-daemon",
            "status": "loaded_but_stopped",
            "running": False,
            "platform": "macOS"
        })
        mock_service_manager.memexd_binary = Path("/usr/local/bin/memexd")

        result = self.runner.invoke(service_app, ["status"])

        assert result.exit_code == 0
        # The status output should indicate the service is loaded but not running

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_status_service_running_unmanaged(self, mock_service_manager):
        """Test status when service is running but not managed by launchd/systemd."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_status = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.workspace-qdrant-daemon",
            "status": "running_unmanaged",
            "running": True,
            "pid": 5678,
            "platform": "macOS",
            "message": "Process running but not managed by launchd"
        })
        mock_service_manager.memexd_binary = Path("/usr/local/bin/memexd")

        result = self.runner.invoke(service_app, ["status"])

        assert result.exit_code == 0
        assert "5678" in result.stdout


class TestServiceLogsEdgeCases:
    """Test edge cases for service logs command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_logs_with_short_lines(self, mock_service_manager):
        """Test logs with minimal line count."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.get_service_logs = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.workspace-qdrant-daemon",
            "logs": ["Line 1", "Line 2", "Line 3"]
        })

        result = self.runner.invoke(service_app, ["logs", "--lines", "5"])

        assert result.exit_code == 0
        mock_service_manager.get_service_logs.assert_called_once_with(5)

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_logs_with_many_lines(self, mock_service_manager):
        """Test logs with large line count."""
        from wqm_cli.cli.commands.service import service_app

        mock_logs = [f"Log line {i}" for i in range(500)]
        mock_service_manager.get_service_logs = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.workspace-qdrant-daemon",
            "logs": mock_logs
        })

        result = self.runner.invoke(service_app, ["logs", "--lines", "500"])

        assert result.exit_code == 0
        mock_service_manager.get_service_logs.assert_called_once_with(500)


class TestServiceRestartEdgeCases:
    """Test edge cases for service restart command."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_restart_when_stopped(self, mock_service_manager):
        """Test restarting a stopped service."""
        from wqm_cli.cli.commands.service import service_app

        # Simulate restart of stopped service (stop succeeds, start succeeds)
        mock_service_manager.restart_service = AsyncMock(return_value={
            "success": True,
            "service_id": "com.workspace-qdrant.workspace-qdrant-daemon",
            "message": "Service restarted successfully"
        })

        result = self.runner.invoke(service_app, ["restart"])

        assert result.exit_code == 0
        assert "Service restarted successfully" in result.stdout

    @patch('wqm_cli.cli.commands.service.service_manager')
    def test_restart_stop_fails(self, mock_service_manager):
        """Test restart when stop operation fails."""
        from wqm_cli.cli.commands.service import service_app

        mock_service_manager.restart_service = AsyncMock(return_value={
            "success": False,
            "error": "Failed to stop service before restart"
        })

        result = self.runner.invoke(service_app, ["restart"])

        assert result.exit_code == 1
        assert "Failed to restart service" in result.stdout


class TestServiceConfigPathResolution:
    """Test configuration path resolution edge cases."""

    @pytest.mark.xfail(reason="Path.exists mock cannot receive path argument - needs rewrite with temp directory")
    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists')
    @patch('os.access', return_value=True)
    def test_config_path_workspace_qdrant_config_exists(self, mock_access, mock_exists, mock_system):
        """Test config path when workspace_qdrant_config.yaml exists."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        def exists_side_effect(path_self):
            path_str = str(path_self)
            return (path_str.endswith("memexd") or
                    "workspace_qdrant_config.yaml" in path_str)

        mock_exists.side_effect = lambda: exists_side_effect(mock_exists._mock_self)

        with patch.object(Path, 'exists', side_effect=exists_side_effect):
            manager = MemexdServiceManager()
            config_path = manager.get_config_path()

            assert "workspace_qdrant_config.yaml" in str(config_path)

    @pytest.mark.xfail(reason="Path.exists mock cannot receive path argument - needs rewrite with temp directory")
    @patch('platform.system', return_value='Darwin')
    @patch('pathlib.Path.exists')
    @patch('os.access', return_value=True)
    def test_config_path_default_config_exists(self, mock_access, mock_exists, mock_system):
        """Test config path when only config.yaml exists."""
        from wqm_cli.cli.commands.service import MemexdServiceManager

        def exists_side_effect(path_self):
            path_str = str(path_self)
            return (path_str.endswith("memexd") or
                    ("config.yaml" in path_str and "workspace_qdrant_config.yaml" not in path_str))

        with patch.object(Path, 'exists', side_effect=exists_side_effect):
            manager = MemexdServiceManager()
            config_path = manager.get_config_path()

            assert "config.yaml" in str(config_path)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
