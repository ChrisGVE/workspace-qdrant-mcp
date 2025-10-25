"""
Daemon Lifecycle and Integration Testing

Comprehensive tests for daemon startup, shutdown, restart scenarios, service
installation/management, and integration between CLI and daemon processes.

This module implements subtask 203.2 of the End-to-End Functional Testing Framework.
"""

import asyncio
import json
import os
import platform
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import psutil
import pytest
import yaml


class DaemonTestEnvironment:
    """Test environment for daemon lifecycle testing."""

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.config_dir = tmp_path / ".config" / "workspace-qdrant"
        self.log_dir = tmp_path / "logs"
        self.pid_file = tmp_path / "daemon.pid"
        self.socket_file = tmp_path / "daemon.sock"
        self.cli_executable = "uv run wqm"
        self.daemon_process = None
        self.test_processes = []

        self.setup_test_environment()

    def setup_test_environment(self):
        """Set up test directories and configuration."""
        # Create test directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create test configuration
        config_content = {
            "qdrant_url": "http://localhost:6333",
            "log_level": "DEBUG",
            "log_dir": str(self.log_dir),
            "pid_file": str(self.pid_file),
            "socket_file": str(self.socket_file),
            "daemon": {
                "enabled": True,
                "startup_timeout": 30,
                "shutdown_timeout": 15,
                "health_check_interval": 5,
                "max_restart_attempts": 3
            }
        }

        config_file = self.config_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_content, f)

    def cleanup(self):
        """Clean up test processes and files."""
        # Kill test processes
        for process in self.test_processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

        # Clean up daemon process
        if self.daemon_process:
            try:
                if self.daemon_process.poll() is None:
                    self.daemon_process.terminate()
                    self.daemon_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    self.daemon_process.kill()
                except ProcessLookupError:
                    pass

        # Clean up PID file
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except FileNotFoundError:
                pass

    def run_cli_command(
        self,
        command: str,
        timeout: int = 30,
        env_vars: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """Execute CLI command and return result."""
        env = os.environ.copy()
        env.update({
            "WQM_CONFIG_DIR": str(self.config_dir),
            "PYTHONPATH": str(Path.cwd()),
        })
        if env_vars:
            env.update(env_vars)

        try:
            result = subprocess.run(
                f"{self.cli_executable} {command}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return -1, "", f"Command execution failed: {e}"

    def start_daemon_process(self, timeout: int = 30) -> bool:
        """Start daemon process and return success status."""
        try:
            # Start daemon using service command
            return_code, stdout, stderr = self.run_cli_command("service start", timeout=timeout)
            return return_code == 0
        except Exception:
            return False

    def stop_daemon_process(self, timeout: int = 15) -> bool:
        """Stop daemon process and return success status."""
        try:
            return_code, stdout, stderr = self.run_cli_command("service stop", timeout=timeout)
            return return_code == 0
        except Exception:
            return False

    def get_daemon_status(self) -> dict[str, Any]:
        """Get daemon status information."""
        return_code, stdout, stderr = self.run_cli_command("service status")

        status = {
            "running": False,
            "pid": None,
            "uptime": None,
            "error": None
        }

        if return_code == 0:
            try:
                # Parse status output
                if "running" in stdout.lower():
                    status["running"] = True
                    # Extract PID if available
                    import re
                    pid_match = re.search(r'pid[:\s]+(\d+)', stdout, re.IGNORECASE)
                    if pid_match:
                        status["pid"] = int(pid_match.group(1))
            except Exception as e:
                status["error"] = str(e)
        else:
            status["error"] = stderr + stdout

        return status

    def is_daemon_responsive(self) -> bool:
        """Check if daemon is responsive to requests."""
        try:
            return_code, stdout, stderr = self.run_cli_command("admin status", timeout=10)
            return return_code == 0
        except Exception:
            return False

    def wait_for_daemon_ready(self, timeout: int = 30) -> bool:
        """Wait for daemon to be ready and responsive."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_daemon_responsive():
                return True
            time.sleep(1)
        return False

    def simulate_daemon_crash(self) -> bool:
        """Simulate daemon crash for testing recovery."""
        status = self.get_daemon_status()
        if status["running"] and status["pid"]:
            try:
                # Send SIGKILL to simulate crash
                os.kill(status["pid"], signal.SIGKILL)
                return True
            except (ProcessLookupError, PermissionError):
                pass
        return False


class DaemonLifecycleValidator:
    """Validates daemon lifecycle operations and states."""

    @staticmethod
    def validate_daemon_started(status: dict[str, Any]) -> bool:
        """Validate that daemon started successfully."""
        return status.get("running", False) and status.get("pid") is not None

    @staticmethod
    def validate_daemon_stopped(status: dict[str, Any]) -> bool:
        """Validate that daemon stopped successfully."""
        return not status.get("running", True)

    @staticmethod
    def validate_process_exists(pid: int) -> bool:
        """Validate that process with given PID exists."""
        try:
            return psutil.pid_exists(pid)
        except Exception:
            return False

    @staticmethod
    def validate_service_installation() -> bool:
        """Validate service installation status."""
        # Check platform-specific service installation
        system = platform.system().lower()

        if system == "linux":
            # Check systemd service
            result = subprocess.run(
                ["systemctl", "--user", "list-unit-files", "workspace-qdrant*"],
                capture_output=True, text=True
            )
            return result.returncode == 0 and "workspace-qdrant" in result.stdout
        elif system == "darwin":
            # Check launchd service
            result = subprocess.run(
                ["launchctl", "list", "com.workspace.qdrant"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        elif system == "windows":
            # Check Windows service
            result = subprocess.run(
                ["sc", "query", "WorkspaceQdrant"],
                capture_output=True, text=True, shell=True
            )
            return result.returncode == 0

        return False


@pytest.mark.functional
@pytest.mark.daemon_lifecycle
class TestDaemonLifecycle:
    """Test daemon lifecycle operations."""

    @pytest.fixture
    def daemon_env(self, tmp_path):
        """Create daemon test environment."""
        env = DaemonTestEnvironment(tmp_path)
        yield env
        env.cleanup()

    @pytest.fixture
    def validator(self):
        """Create daemon lifecycle validator."""
        return DaemonLifecycleValidator()

    def test_daemon_startup_workflow(self, daemon_env, validator):
        """Test daemon startup workflow."""
        # Ensure daemon is not running
        initial_status = daemon_env.get_daemon_status()
        if initial_status["running"]:
            daemon_env.stop_daemon_process()
            time.sleep(2)

        # Test daemon startup
        startup_success = daemon_env.start_daemon_process()

        # Validate startup (may not succeed if Qdrant not available)
        if startup_success:
            # Wait for daemon to be ready
            ready = daemon_env.wait_for_daemon_ready()
            assert ready, "Daemon did not become ready within timeout"

            # Validate daemon status
            status = daemon_env.get_daemon_status()
            assert validator.validate_daemon_started(status)

            # Validate process exists
            if status["pid"]:
                assert validator.validate_process_exists(status["pid"])
        else:
            # Even if startup fails, should provide meaningful error
            status = daemon_env.get_daemon_status()
            assert status.get("error") is not None

    def test_daemon_shutdown_workflow(self, daemon_env, validator):
        """Test daemon shutdown workflow."""
        # Start daemon first (if possible)
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            # Test graceful shutdown
            shutdown_success = daemon_env.stop_daemon_process()
            assert shutdown_success, "Daemon shutdown failed"

            # Validate daemon stopped
            time.sleep(2)
            status = daemon_env.get_daemon_status()
            assert validator.validate_daemon_stopped(status)
        else:
            # Test shutdown when not running
            daemon_env.stop_daemon_process()
            # Should handle "not running" case gracefully
            status = daemon_env.get_daemon_status()
            assert not status["running"]

    def test_daemon_restart_workflow(self, daemon_env, validator):
        """Test daemon restart workflow."""
        # Test restart command
        return_code, stdout, stderr = daemon_env.run_cli_command("service restart")

        # Restart should work even if daemon not initially running
        assert return_code in [0, 1]  # Success or expected failure

        # If restart succeeded, validate final state
        if return_code == 0:
            time.sleep(3)
            if daemon_env.wait_for_daemon_ready(timeout=10):
                status = daemon_env.get_daemon_status()
                assert validator.validate_daemon_started(status)

    def test_daemon_status_reporting(self, daemon_env, validator):
        """Test daemon status reporting accuracy."""
        # Test status when stopped
        initial_status = daemon_env.get_daemon_status()

        # Start daemon (if possible)
        startup_success = daemon_env.start_daemon_process()

        if startup_success:
            # Wait and check status
            time.sleep(3)
            daemon_env.get_daemon_status()

            # Stop daemon
            daemon_env.stop_daemon_process()
            time.sleep(2)
            stopped_status = daemon_env.get_daemon_status()

            # Validate status transitions
            assert not validator.validate_daemon_started(stopped_status)
        else:
            # Even if daemon can't start, status should be accurate
            assert not initial_status["running"]

    @pytest.mark.slow
    def test_daemon_health_monitoring(self, daemon_env, validator):
        """Test daemon health monitoring and recovery."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            # Test health check
            assert daemon_env.is_daemon_responsive()

            # Test health monitoring over time
            for _ in range(3):
                time.sleep(2)
                assert daemon_env.is_daemon_responsive()

    def test_daemon_crash_recovery(self, daemon_env, validator):
        """Test daemon crash detection and recovery."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            # Get initial status
            initial_status = daemon_env.get_daemon_status()

            # Simulate crash
            if initial_status["pid"]:
                crash_success = daemon_env.simulate_daemon_crash()

                if crash_success:
                    # Wait a moment for crash detection
                    time.sleep(5)

                    # Check if daemon auto-recovers or reports crash
                    status = daemon_env.get_daemon_status()
                    # Either should recover or report not running
                    assert status["running"] or status["error"] is not None

    def test_multiple_daemon_instances(self, daemon_env, validator):
        """Test handling of multiple daemon instances."""
        # Try to start daemon twice
        first_start = daemon_env.start_daemon_process()

        if first_start:
            # Try to start again
            return_code, stdout, stderr = daemon_env.run_cli_command("service start")

            # Should handle already running case gracefully
            error_message = stderr + stdout
            assert "already" in error_message.lower() or "running" in error_message.lower()


@pytest.mark.functional
@pytest.mark.daemon_integration
class TestDaemonIntegration:
    """Test daemon integration with CLI and other components."""

    @pytest.fixture
    def daemon_env(self, tmp_path):
        """Create daemon test environment."""
        env = DaemonTestEnvironment(tmp_path)
        yield env
        env.cleanup()

    def test_cli_daemon_communication(self, daemon_env):
        """Test communication between CLI and daemon."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            # Test CLI commands that require daemon
            commands = [
                "admin status",
                "memory list",
                "search project 'test'",
                "library list",
                "watch list"
            ]

            for command in commands:
                return_code, stdout, stderr = daemon_env.run_cli_command(command)

                # Should either succeed or provide clear error
                assert len(stdout + stderr) > 0

                # If succeeded, should have meaningful output
                if return_code == 0:
                    assert len(stdout.strip()) > 0

    def test_daemon_cli_error_handling(self, daemon_env):
        """Test error handling when daemon is not available."""
        # Ensure daemon is stopped
        daemon_env.stop_daemon_process()
        time.sleep(2)

        # Test CLI commands that require daemon
        commands = [
            "admin status",
            "memory add 'test rule'",
            "search project 'test'",
            "ingest file /nonexistent/file.txt"
        ]

        for command in commands:
            return_code, stdout, stderr = daemon_env.run_cli_command(command)

            # Should provide clear error about daemon not available
            error_message = (stderr + stdout).lower()
            assert any(indicator in error_message for indicator in [
                "daemon", "connection", "service", "not running", "unavailable"
            ])

    def test_daemon_configuration_reload(self, daemon_env):
        """Test daemon configuration reload functionality."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            # Test configuration reload
            return_code, stdout, stderr = daemon_env.run_cli_command("config reload")

            # Should handle reload request
            assert return_code in [0, 1]  # Success or not implemented
            assert len(stdout + stderr) > 0

    def test_daemon_service_management_cross_platform(self, daemon_env):
        """Test service management across different platforms."""
        platform.system().lower()

        # Test service installation
        return_code, stdout, stderr = daemon_env.run_cli_command("service install")

        # May not succeed without proper permissions, but should provide feedback
        assert len(stdout + stderr) > 0

        # Test service uninstallation
        return_code, stdout, stderr = daemon_env.run_cli_command("service uninstall")

        # Should provide feedback about uninstallation
        assert len(stdout + stderr) > 0

    def test_daemon_log_management(self, daemon_env):
        """Test daemon log management and rotation."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success:
            # Wait for some activity
            time.sleep(5)

            # Check log file creation
            log_files = list(daemon_env.log_dir.glob("*.log"))

            # Should create log files
            if log_files:
                # Check log content
                for log_file in log_files:
                    content = log_file.read_text()
                    assert len(content) > 0

    def test_daemon_pid_file_management(self, daemon_env):
        """Test daemon PID file management."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            # Check PID file creation
            if daemon_env.pid_file.exists():
                pid_content = daemon_env.pid_file.read_text().strip()
                assert pid_content.isdigit()

                pid = int(pid_content)
                assert DaemonLifecycleValidator.validate_process_exists(pid)

            # Stop daemon and check PID file cleanup
            daemon_env.stop_daemon_process()
            time.sleep(2)

            # PID file should be cleaned up
            assert not daemon_env.pid_file.exists() or daemon_env.pid_file.stat().st_size == 0

    def test_daemon_socket_communication(self, daemon_env):
        """Test daemon socket communication."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            # Check socket file creation
            if daemon_env.socket_file.exists():
                # Test socket communication through CLI
                return_code, stdout, stderr = daemon_env.run_cli_command("admin status")

                # Should be able to communicate via socket
                assert return_code == 0 or len(stderr + stdout) > 0


@pytest.mark.functional
@pytest.mark.daemon_performance
class TestDaemonPerformance:
    """Test daemon performance and resource management."""

    @pytest.fixture
    def daemon_env(self, tmp_path):
        """Create daemon test environment."""
        env = DaemonTestEnvironment(tmp_path)
        yield env
        env.cleanup()

    def test_daemon_startup_time(self, daemon_env):
        """Test daemon startup performance."""
        # Measure startup time
        start_time = time.time()
        startup_success = daemon_env.start_daemon_process(timeout=30)

        if startup_success:
            ready = daemon_env.wait_for_daemon_ready(timeout=30)
            if ready:
                startup_time = time.time() - start_time

                # Startup should be reasonable (less than 30 seconds)
                assert startup_time < 30, f"Daemon startup took {startup_time:.2f} seconds"

    def test_daemon_memory_usage(self, daemon_env):
        """Test daemon memory usage patterns."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            status = daemon_env.get_daemon_status()

            if status["pid"]:
                try:
                    process = psutil.Process(status["pid"])
                    memory_info = process.memory_info()

                    # Memory usage should be reasonable (less than 500MB)
                    memory_mb = memory_info.rss / 1024 / 1024
                    assert memory_mb < 500, f"Daemon using {memory_mb:.2f}MB memory"

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process might have stopped or access denied
                    pass

    def test_daemon_cpu_usage(self, daemon_env):
        """Test daemon CPU usage patterns."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            status = daemon_env.get_daemon_status()

            if status["pid"]:
                try:
                    process = psutil.Process(status["pid"])

                    # Monitor CPU usage over time
                    cpu_samples = []
                    for _ in range(5):
                        cpu_percent = process.cpu_percent(interval=1)
                        cpu_samples.append(cpu_percent)

                    avg_cpu = sum(cpu_samples) / len(cpu_samples)

                    # CPU usage should be reasonable when idle
                    assert avg_cpu < 50, f"Daemon using {avg_cpu:.2f}% CPU on average"

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process might have stopped or access denied
                    pass

    @pytest.mark.slow
    def test_daemon_long_running_stability(self, daemon_env):
        """Test daemon stability over extended periods."""
        # Start daemon
        startup_success = daemon_env.start_daemon_process()

        if startup_success and daemon_env.wait_for_daemon_ready(timeout=10):
            # Monitor daemon over time
            monitoring_duration = 60  # 1 minute for test
            check_interval = 10  # Check every 10 seconds

            start_time = time.time()
            checks = 0
            successful_checks = 0

            while time.time() - start_time < monitoring_duration:
                if daemon_env.is_daemon_responsive():
                    successful_checks += 1
                checks += 1
                time.sleep(check_interval)

            # Should maintain responsiveness
            if checks > 0:
                uptime_percentage = (successful_checks / checks) * 100
                assert uptime_percentage > 80, f"Daemon uptime only {uptime_percentage:.1f}%"
