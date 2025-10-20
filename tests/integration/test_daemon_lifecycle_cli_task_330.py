"""
Daemon Lifecycle Control Integration Tests (Task 330.1).

Real integration tests for CLI service management commands controlling daemon
lifecycle. Tests actual subprocess execution of wqm service commands against
a real (or test) daemon process.

Test Coverage:
- Service start/stop/restart/status commands
- PID handling and process management
- Graceful vs forced shutdown scenarios
- Service state transitions
- Idempotent operations (start when running, stop when stopped)
- Error handling (daemon crashes, invalid states)

Architecture:
- Uses subprocess.run() to execute real wqm CLI commands
- Manages actual daemon processes (not mocked)
- Verifies process existence via psutil
- Tests real service state via service management
"""

import asyncio
import os
import psutil
import pytest
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any


@pytest.fixture
def service_env():
    """
    Provide environment for service testing.

    Uses the system service without isolation for pragmatic testing.
    """
    env = os.environ.copy()

    yield {
        "env": env
    }


def run_wqm_command(command: list, env: Optional[Dict] = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """
    Run wqm CLI command via subprocess.

    Args:
        command: Command arguments (e.g., ['service', 'start'])
        env: Environment variables
        timeout: Command timeout in seconds

    Returns:
        CompletedProcess with result
    """
    full_command = ["uv", "run", "wqm"] + command
    result = subprocess.run(
        full_command,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result


def get_daemon_pid_from_status(env: Dict) -> Optional[int]:
    """
    Extract daemon PID from wqm service status command.

    Args:
        env: Environment variables

    Returns:
        PID if daemon running, None otherwise
    """
    result = run_wqm_command(["service", "status"], env=env)

    if result.returncode == 0:
        # Parse output for PID
        # Expected format: "Status: running (PID: 12345)" or similar
        output = result.stdout
        if "PID" in output or "pid" in output:
            # Extract PID from output
            import re
            match = re.search(r'[Pp][Ii][Dd][:=\s]+(\d+)', output)
            if match:
                return int(match.group(1))

    return None


def is_process_running(pid: int) -> bool:
    """
    Check if process with given PID is running.

    Args:
        pid: Process ID to check

    Returns:
        True if process exists and is running
    """
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def wait_for_daemon_ready(env: Dict, timeout: int = 10) -> bool:
    """
    Wait for daemon to be ready and responding.

    Args:
        env: Environment variables
        timeout: Maximum wait time in seconds

    Returns:
        True if daemon ready, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        pid = get_daemon_pid_from_status(env)
        if pid and is_process_running(pid):
            # Additional check: daemon is responding
            time.sleep(0.5)  # Grace period for daemon to fully initialize
            return True
        time.sleep(0.2)
    return False


def wait_for_daemon_stopped(env: Dict, timeout: int = 10) -> bool:
    """
    Wait for daemon to fully stop.

    Args:
        env: Environment variables
        timeout: Maximum wait time in seconds

    Returns:
        True if daemon stopped, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        pid = get_daemon_pid_from_status(env)
        if pid is None or not is_process_running(pid):
            return True
        time.sleep(0.2)
    return False


@pytest.mark.integration
@pytest.mark.slow
class TestServiceStartCommand:
    """Test 'wqm service start' command."""

    def test_start_daemon_from_stopped_state(self, service_env):
        """Test starting daemon when not running."""
        env = service_env["env"]

        # Ensure daemon is stopped
        run_wqm_command(["service", "stop"], env=env)
        wait_for_daemon_stopped(env)

        # Start daemon
        result = run_wqm_command(["service", "start"], env=env)

        # Verify command succeeded
        assert result.returncode == 0, f"Start failed: {result.stderr}"

        # Verify daemon is running
        assert wait_for_daemon_ready(env, timeout=15), "Daemon did not start within timeout"

        # Verify PID is valid
        pid = get_daemon_pid_from_status(env)
        assert pid is not None, "No PID found after start"
        assert is_process_running(pid), f"Process {pid} not running"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)

    def test_start_daemon_when_already_running_is_idempotent(self, service_env):
        """Test starting daemon when already running (idempotent operation)."""
        env = service_env["env"]

        # Start daemon first time
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Initial start failed"

        # Get PID after first start
        pid_first = get_daemon_pid_from_status(env)

        # Start again (should be idempotent)
        result = run_wqm_command(["service", "start"], env=env)

        # Should succeed (not error)
        assert result.returncode == 0, "Second start should be idempotent"

        # PID should remain the same (same process)
        pid_second = get_daemon_pid_from_status(env)
        assert pid_first == pid_second, "PID changed on idempotent start"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)

    def test_start_makes_daemon_running(self, service_env):
        """Test that starting daemon makes it show as running."""
        env = service_env["env"]

        # Ensure stopped first
        run_wqm_command(["service", "stop"], env=env)
        wait_for_daemon_stopped(env)

        # Start daemon
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        # Verify running
        pid = get_daemon_pid_from_status(env)
        assert pid is not None, "No PID reported after start"
        assert is_process_running(pid), f"Process {pid} not running"

        # Don't cleanup - leave in running state


@pytest.mark.integration
@pytest.mark.slow
class TestServiceStopCommand:
    """Test 'wqm service stop' command."""

    def test_stop_daemon_when_running(self, service_env):
        """Test stopping daemon when running."""
        env = service_env["env"]

        # Start daemon first
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        pid = get_daemon_pid_from_status(env)
        assert pid is not None

        # Stop daemon
        result = run_wqm_command(["service", "stop"], env=env)

        # Verify command succeeded
        assert result.returncode == 0, f"Stop failed: {result.stderr}"

        # Verify daemon stopped
        assert wait_for_daemon_stopped(env, timeout=15), "Daemon did not stop within timeout"

        # Verify process is dead
        assert not is_process_running(pid), f"Process {pid} still running after stop"

    def test_stop_daemon_when_not_running_is_idempotent(self, service_env):
        """Test stopping daemon when not running (idempotent operation)."""
        env = service_env["env"]

        # Ensure daemon is stopped
        run_wqm_command(["service", "stop"], env=env)
        wait_for_daemon_stopped(env)

        # Stop again (should be idempotent)
        result = run_wqm_command(["service", "stop"], env=env)

        # Should succeed (not error)
        assert result.returncode == 0, "Stop when not running should be idempotent"

    def test_stop_makes_daemon_not_running(self, service_env):
        """Test that stopping daemon makes it not running."""
        env = service_env["env"]

        # Start daemon first
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        pid_before = get_daemon_pid_from_status(env)
        assert pid_before is not None

        # Stop daemon
        run_wqm_command(["service", "stop"], env=env)
        assert wait_for_daemon_stopped(env), "Daemon did not stop"

        # Verify process is dead
        assert not is_process_running(pid_before), f"Process {pid_before} still running"

        # Verify status shows not running
        pid_after = get_daemon_pid_from_status(env)
        assert pid_after is None, "Status still showing PID after stop"

    def test_graceful_shutdown_completes_in_flight_operations(self, service_env):
        """Test that graceful shutdown waits for in-flight operations."""
        env = service_env["env"]

        # Start daemon
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        # TODO: Trigger long-running operation (e.g., large file ingestion)
        # For now, just test that stop waits reasonable amount of time

        stop_start = time.time()
        result = run_wqm_command(["service", "stop"], env=env, timeout=30)
        stop_duration = time.time() - stop_start

        # Graceful shutdown should complete within reasonable time
        assert result.returncode == 0, "Graceful stop failed"
        assert stop_duration < 25, f"Stop took too long: {stop_duration}s (possible hang)"


@pytest.mark.integration
@pytest.mark.slow
class TestServiceRestartCommand:
    """Test 'wqm service restart' command."""

    def test_restart_daemon_when_running(self, service_env):
        """Test restarting daemon when running."""
        env = service_env["env"]

        # Start daemon
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        pid_before = get_daemon_pid_from_status(env)

        # Restart daemon
        result = run_wqm_command(["service", "restart"], env=env)

        # Verify command succeeded
        assert result.returncode == 0, f"Restart failed: {result.stderr}"

        # Verify daemon restarted (new PID)
        assert wait_for_daemon_ready(env, timeout=15), "Daemon did not restart"

        pid_after = get_daemon_pid_from_status(env)
        assert pid_after is not None
        assert pid_after != pid_before, "PID did not change after restart"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)

    def test_restart_daemon_when_not_running_starts_it(self, service_env):
        """Test restarting daemon when not running (should start it)."""
        env = service_env["env"]

        # Ensure stopped
        run_wqm_command(["service", "stop"], env=env)
        wait_for_daemon_stopped(env)

        # Restart (should start since not running)
        result = run_wqm_command(["service", "restart"], env=env)

        # Should succeed
        assert result.returncode == 0, "Restart failed when not running"

        # Daemon should be running now
        assert wait_for_daemon_ready(env), "Daemon not running after restart from stopped state"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)

    def test_restart_preserves_configuration(self, service_env):
        """Test that restart preserves daemon configuration."""
        env = service_env["env"]

        # Start daemon with specific configuration
        # (In real implementation, would set config before start)
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        # TODO: Verify specific configuration is active

        # Restart
        run_wqm_command(["service", "restart"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not restart"

        # TODO: Verify configuration still active after restart

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)


@pytest.mark.integration
class TestServiceStatusCommand:
    """Test 'wqm service status' command."""

    def test_status_when_daemon_running(self, service_env):
        """Test status command when daemon is running."""
        env = service_env["env"]

        # Start daemon
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        # Get status
        result = run_wqm_command(["service", "status"], env=env)

        # Status should succeed
        assert result.returncode == 0, f"Status failed: {result.stderr}"

        # Output should indicate running
        output = result.stdout.lower()
        assert "running" in output or "active" in output, f"Output does not show running: {result.stdout}"

        # Should include PID
        assert "pid" in output, f"Output missing PID: {result.stdout}"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)

    def test_status_when_daemon_not_running(self, service_env):
        """Test status command when daemon is not running."""
        env = service_env["env"]

        # Ensure stopped
        run_wqm_command(["service", "stop"], env=env)
        wait_for_daemon_stopped(env)

        # Get status
        result = run_wqm_command(["service", "status"], env=env)

        # Status command should still succeed (but show not running)
        # Note: Some implementations return non-zero when not running
        # Either behavior is acceptable

        # Output should indicate stopped
        output = result.stdout.lower()
        is_stopped = (
            "stopped" in output or
            "not running" in output or
            "inactive" in output or
            "dead" in output
        )
        assert is_stopped, f"Output does not show stopped state: {result.stdout}"

    def test_status_includes_useful_information(self, service_env):
        """Test that status output includes useful diagnostic information."""
        env = service_env["env"]

        # Start daemon
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        # Get status
        result = run_wqm_command(["service", "status"], env=env)
        output = result.stdout

        # Should include multiple diagnostic fields
        # (Specific fields depend on implementation)
        has_info = (
            len(output) > 50  # Non-trivial output
        )
        assert has_info, f"Status output seems too minimal: {output}"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)


@pytest.mark.integration
@pytest.mark.slow
class TestServiceLifecycleTransitions:
    """Test daemon state transitions and edge cases."""

    def test_rapid_start_stop_cycles(self, service_env):
        """Test rapid start/stop cycles for stability."""
        env = service_env["env"]

        # Ensure clean state
        run_wqm_command(["service", "stop"], env=env)
        wait_for_daemon_stopped(env)

        # Perform 5 rapid start/stop cycles
        for i in range(5):
            # Start
            result_start = run_wqm_command(["service", "start"], env=env)
            assert result_start.returncode == 0, f"Start {i+1} failed"
            assert wait_for_daemon_ready(env, timeout=10), f"Daemon not ready on cycle {i+1}"

            # Brief operation
            time.sleep(0.5)

            # Stop
            result_stop = run_wqm_command(["service", "stop"], env=env)
            assert result_stop.returncode == 0, f"Stop {i+1} failed"
            assert wait_for_daemon_stopped(env, timeout=10), f"Daemon not stopped on cycle {i+1}"

    def test_service_commands_during_startup_transition(self, service_env):
        """Test commands issued during daemon startup."""
        env = service_env["env"]

        # Ensure stopped
        run_wqm_command(["service", "stop"], env=env)
        wait_for_daemon_stopped(env)

        # Start daemon without waiting
        import threading
        start_thread = threading.Thread(
            target=lambda: run_wqm_command(["service", "start"], env=env)
        )
        start_thread.start()

        # Immediately check status (during startup)
        time.sleep(0.1)  # Brief delay to ensure startup initiated
        result = run_wqm_command(["service", "status"], env=env)

        # Status should handle startup transition gracefully
        # (Either show starting, not ready, or already running)
        assert result.returncode in [0, 3], "Status failed during startup transition"

        # Wait for startup to complete
        start_thread.join(timeout=15)
        assert wait_for_daemon_ready(env), "Daemon did not complete startup"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)

    def test_service_commands_during_shutdown_transition(self, service_env):
        """Test commands issued during daemon shutdown."""
        env = service_env["env"]

        # Start daemon
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        # Stop daemon without waiting
        import threading
        stop_thread = threading.Thread(
            target=lambda: run_wqm_command(["service", "stop"], env=env)
        )
        stop_thread.start()

        # Immediately check status (during shutdown)
        time.sleep(0.1)  # Brief delay to ensure shutdown initiated
        result = run_wqm_command(["service", "status"], env=env)

        # Status should handle shutdown transition gracefully
        assert result.returncode in [0, 3], "Status failed during shutdown transition"

        # Wait for shutdown to complete
        stop_thread.join(timeout=15)
        assert wait_for_daemon_stopped(env), "Daemon did not complete shutdown"

    @pytest.mark.skip(reason="Requires daemon crash simulation mechanism")
    def test_recovery_from_daemon_crash(self, service_env):
        """Test daemon recovery after unexpected crash."""
        env = service_env["env"]

        # Start daemon
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        pid = get_daemon_pid_from_status(env)

        # Simulate crash (kill -9)
        os.kill(pid, 9)

        # Wait for crash to be detected
        time.sleep(2)

        # Status should reflect crashed state
        result = run_wqm_command(["service", "status"], env=env)
        # Should show not running or crashed

        # Restart should work
        restart_result = run_wqm_command(["service", "restart"], env=env)
        assert restart_result.returncode == 0, "Could not restart after crash"
        assert wait_for_daemon_ready(env), "Daemon not running after crash recovery"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)


@pytest.mark.integration
class TestServiceErrorHandling:
    """Test error handling in service commands."""

    def test_error_message_quality_when_command_fails(self, service_env):
        """Test that error messages are clear and actionable."""
        env = service_env["env"]

        # Induce error (implementation-specific)
        # For example: try to start with invalid configuration
        # This is a placeholder - actual implementation depends on CLI

        # Error messages should:
        # 1. Explain what went wrong
        # 2. Suggest remediation steps
        # 3. Include relevant context (PID, paths, etc.)

        # This test would need specific error scenarios to verify

    def test_concurrent_service_commands(self, service_env):
        """Test multiple concurrent service commands."""
        env = service_env["env"]

        # Start daemon
        run_wqm_command(["service", "start"], env=env)
        assert wait_for_daemon_ready(env), "Daemon did not start"

        # Issue multiple status checks concurrently
        import concurrent.futures

        def check_status():
            result = run_wqm_command(["service", "status"], env=env)
            return result.returncode == 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(check_status) for _ in range(10)]
            results = [f.result() for f in futures]

        # All status checks should succeed
        assert all(results), "Some concurrent status checks failed"

        # Cleanup
        run_wqm_command(["service", "stop"], env=env)
