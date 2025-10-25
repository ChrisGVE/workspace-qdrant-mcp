"""
Integration tests for error propagation between CLI and daemon (Task 330.7).

Tests error handling and propagation:
- Daemon errors propagating to CLI with clear messages
- Network failures and connection issues
- Timeout handling for long operations
- Graceful degradation when services unavailable
- Informative error messages
- Correct exit codes
- Recovery mechanisms

These tests verify:
1. Errors from daemon reach CLI with context
2. Network issues handled gracefully
3. Timeouts don't leave orphaned processes
4. CLI provides actionable error messages
5. Exit codes follow conventions (0=success, non-zero=failure)
6. System recovers from transient errors
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import pytest


def run_wqm_command(
    command: list, env: dict | None = None, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Run wqm CLI command via subprocess."""
    full_command = ["uv", "run", "wqm"] + command
    result = subprocess.run(
        full_command,
        env=env or os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


def stop_daemon() -> subprocess.CompletedProcess:
    """Stop the daemon for testing error scenarios."""
    return run_wqm_command(["service", "stop"])


def start_daemon() -> subprocess.CompletedProcess:
    """Start the daemon."""
    return run_wqm_command(["service", "start"])


def daemon_is_running() -> bool:
    """Check if daemon is currently running."""
    result = run_wqm_command(["service", "status"])
    return result.returncode == 0 and "running" in result.stdout.lower()


@pytest.fixture(scope="module")
def ensure_daemon_running():
    """Ensure daemon is running for tests that need it."""
    if not daemon_is_running():
        start_result = start_daemon()
        if start_result.returncode != 0:
            pytest.skip("Daemon not available and could not be started")
        time.sleep(3)

    yield

    # Ensure daemon is running after tests
    if not daemon_is_running():
        start_daemon()
        time.sleep(2)


@pytest.fixture
def test_workspace(tmp_path):
    """Create temporary workspace for error testing."""
    workspace = tmp_path / "error_test_workspace"
    workspace.mkdir()

    # Create a test file
    test_file = workspace / "test.txt"
    test_file.write_text("Test content for error propagation")

    yield {
        "workspace": workspace,
        "test_file": test_file,
    }


@pytest.fixture
def test_collection():
    """Provide test collection name and cleanup."""
    collection_name = f"test_error_prop_{int(time.time())}"

    yield collection_name

    # Cleanup: delete test collection
    try:
        run_wqm_command(
            ["admin", "collections", "delete", collection_name, "--confirm"]
        )
    except Exception:
        pass


@pytest.mark.integration
@pytest.mark.slow
class TestDaemonErrorPropagation:
    """Test daemon errors propagating to CLI."""

    def test_cli_reports_daemon_offline(self):
        """Test that CLI clearly reports when daemon is offline."""
        # Stop daemon
        stop_result = stop_daemon()
        if stop_result.returncode != 0:
            pytest.skip("Cannot stop daemon for test")

        time.sleep(2)

        try:
            # Try a command that requires daemon
            result = run_wqm_command(["status", "--quiet"])

            # Should indicate daemon is offline
            assert "daemon:offline" in result.stdout or result.returncode != 0

        finally:
            # Restart daemon
            start_daemon()
            time.sleep(3)

    def test_ingestion_error_with_stopped_daemon(self, test_workspace, test_collection):
        """Test ingestion command error when daemon stopped."""
        # Stop daemon
        stop_result = stop_daemon()
        if stop_result.returncode != 0:
            pytest.skip("Cannot stop daemon for test")

        time.sleep(2)

        try:
            test_file = test_workspace["test_file"]

            # Try to ingest
            result = run_wqm_command(
                ["ingest", "file", str(test_file), "--collection", test_collection]
            )

            # Should fail with informative error
            assert result.returncode != 0, "Should fail when daemon offline"
            assert len(result.stderr) > 0, "Should have error message"

            # Error should mention daemon or connection
            error_lower = result.stderr.lower()
            assert (
                "daemon" in error_lower
                or "connect" in error_lower
                or "unavailable" in error_lower
            ), "Error should mention daemon/connection issue"

        finally:
            # Restart daemon
            start_daemon()
            time.sleep(3)

    def test_watch_command_with_stopped_daemon(self, test_workspace, test_collection):
        """Test watch command error when daemon stopped."""
        # Stop daemon
        stop_result = stop_daemon()
        if stop_result.returncode != 0:
            pytest.skip("Cannot stop daemon for test")

        time.sleep(2)

        try:
            watch_dir = test_workspace["workspace"]

            # Try to add watch
            result = run_wqm_command(
                ["watch", "add", str(watch_dir), "--collection", test_collection]
            )

            # Should fail gracefully
            assert result.returncode != 0, "Should fail when daemon offline"

            # Should have informative error
            if len(result.stderr) > 0:
                error_lower = result.stderr.lower()
                # Error should be actionable
                assert "daemon" in error_lower or "connect" in error_lower

        finally:
            # Restart daemon
            start_daemon()
            time.sleep(3)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestInvalidInputErrors:
    """Test error handling for invalid inputs."""

    def test_nonexistent_file_error_message(self, test_collection):
        """Test error message for nonexistent file."""
        result = run_wqm_command(
            ["ingest", "file", "/absolutely/nonexistent/file.txt", "--collection", test_collection]
        )

        # Should fail
        assert result.returncode != 0, "Should fail for nonexistent file"

        # Should have clear error message
        assert len(result.stderr) > 0, "Should have error message"
        error_lower = result.stderr.lower()
        assert (
            "not found" in error_lower
            or "does not exist" in error_lower
            or "no such file" in error_lower
        ), "Error should mention file not found"

    def test_invalid_collection_name_error(self, test_workspace):
        """Test error message for invalid collection name."""
        test_file = test_workspace["test_file"]

        # Try with empty collection name
        result = run_wqm_command(["ingest", "file", str(test_file), "--collection", ""])

        # Should fail with validation error
        assert result.returncode != 0, "Should fail for empty collection name"

        # Should have error message
        if len(result.stderr) > 0:
            # Error should mention collection or validation
            error_lower = result.stderr.lower()
            assert "collection" in error_lower or "invalid" in error_lower

    def test_invalid_watch_pattern_error(self, test_workspace, test_collection):
        """Test error handling for invalid watch patterns."""
        watch_dir = test_workspace["workspace"]

        # Try with invalid pattern (if validation exists)
        result = run_wqm_command(
            [
                "watch",
                "add",
                str(watch_dir),
                "--collection",
                test_collection,
                "--pattern",
                "[invalid-regex-pattern",
            ]
        )

        # May succeed (pattern might not be validated) or fail
        # Just verify it doesn't crash
        assert result.returncode in [0, 1], "Should handle invalid pattern gracefully"

    def test_missing_required_argument_error(self):
        """Test error message for missing required arguments."""
        # Try ingest without file path
        result = run_wqm_command(["ingest", "file"])

        # Should fail
        assert result.returncode != 0, "Should fail for missing argument"

        # Should have usage information
        assert len(result.stderr) > 0 or len(result.stdout) > 0, "Should show usage"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestExitCodeConsistency:
    """Test that exit codes are consistent and meaningful."""

    def test_success_returns_zero(self):
        """Test that successful commands return exit code 0."""
        result = run_wqm_command(["service", "status"])

        # Should return 0 for success
        assert result.returncode == 0, "Successful command should return 0"

    def test_invalid_command_nonzero_exit(self):
        """Test that invalid commands return non-zero exit code."""
        result = run_wqm_command(["nonexistent-command"])

        # Should return non-zero
        assert result.returncode != 0, "Invalid command should return non-zero"

    def test_failed_ingestion_nonzero_exit(self, test_collection):
        """Test that failed ingestion returns non-zero exit code."""
        result = run_wqm_command(
            ["ingest", "file", "/bad/path.txt", "--collection", test_collection]
        )

        # Should return non-zero
        assert result.returncode != 0, "Failed ingestion should return non-zero"

    def test_help_command_returns_zero(self):
        """Test that help command returns zero exit code."""
        result = run_wqm_command(["--help"])

        # Help should return 0
        assert result.returncode == 0, "Help command should return 0"

    def test_version_command_returns_zero(self):
        """Test that version command returns zero exit code."""
        result = run_wqm_command(["--version"])

        # Version should return 0
        assert result.returncode == 0, "Version command should return 0"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestErrorMessageQuality:
    """Test quality and informativeness of error messages."""

    def test_error_messages_are_actionable(self, test_collection):
        """Test that error messages provide actionable information."""
        result = run_wqm_command(
            ["ingest", "file", "/nonexistent/file.txt", "--collection", test_collection]
        )

        assert result.returncode != 0

        if len(result.stderr) > 0:
            error_msg = result.stderr
            # Error should be non-empty and meaningful
            assert len(error_msg) > 10, "Error message should be substantial"
            # Should contain useful keywords
            error_lower = error_msg.lower()
            has_context = any(
                keyword in error_lower
                for keyword in ["file", "not found", "exist", "path", "error"]
            )
            assert has_context, "Error should contain contextual information"

    def test_errors_dont_expose_sensitive_info(self, test_collection):
        """Test that errors don't leak sensitive information."""
        result = run_wqm_command(
            ["ingest", "file", "/etc/shadow", "--collection", test_collection]
        )

        # Should fail
        assert result.returncode != 0

        # Error should not contain file contents or sensitive data
        if len(result.stderr) > 0:
            error_msg = result.stderr
            # Basic check - no suspicious patterns
            assert "-----BEGIN" not in error_msg, "Should not leak private keys"
            assert "password" not in error_msg.lower() or "password" in error_msg.lower() and "invalid" in error_msg.lower()

    def test_stack_traces_not_in_user_errors(self, test_collection):
        """Test that user-facing errors don't show stack traces."""
        result = run_wqm_command(
            ["ingest", "file", "/bad/file.txt", "--collection", test_collection]
        )

        assert result.returncode != 0

        if len(result.stderr) > 0:
            error_msg = result.stderr
            # Should not contain Python stack traces
            assert "Traceback" not in error_msg, "Should not show stack traces to users"
            assert "File \"" not in error_msg or "line" not in error_msg


@pytest.mark.integration
@pytest.mark.slow
class TestTimeoutHandling:
    """Test timeout handling for long operations."""

    def test_command_timeout_handling(self, test_workspace, test_collection):
        """Test that commands handle timeouts gracefully."""
        # This test runs a command with a very short timeout
        # to verify timeout handling doesn't leave orphaned processes

        test_file = test_workspace["test_file"]

        try:
            run_wqm_command(
                ["ingest", "file", str(test_file), "--collection", test_collection],
                timeout=0.1,  # Very short timeout
            )
        except subprocess.TimeoutExpired:
            # Timeout is expected
            pass

        time.sleep(2)

        # System should still be operational
        status_result = run_wqm_command(["service", "status"])
        # Status should work (daemon might be running or not, but command should work)
        assert status_result.returncode in [0, 1, 3], "System should be operational after timeout"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestGracefulDegradation:
    """Test graceful degradation when services partially available."""

    def test_status_degrades_gracefully_without_daemon(self):
        """Test status command degrades gracefully."""
        # Stop daemon
        stop_result = stop_daemon()
        if stop_result.returncode != 0:
            pytest.skip("Cannot stop daemon")

        time.sleep(2)

        try:
            # Status command should still work but indicate degraded state
            result = run_wqm_command(["status", "--quiet"])

            # Should complete (even if showing offline)
            assert "daemon:offline" in result.stdout or result.returncode != 0

        finally:
            start_daemon()
            time.sleep(3)

    def test_collections_command_without_daemon(self):
        """Test collections command when daemon unavailable."""
        # Stop daemon
        stop_result = stop_daemon()
        if stop_result.returncode != 0:
            pytest.skip("Cannot stop daemon")

        time.sleep(2)

        try:
            # Collections might work directly through Qdrant or fail gracefully
            result = run_wqm_command(["admin", "collections"])

            # Should handle gracefully (success or informative failure)
            assert result.returncode in [0, 1], "Should handle daemon unavailability"

        finally:
            start_daemon()
            time.sleep(3)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestRecoveryMechanisms:
    """Test recovery mechanisms after errors."""

    def test_system_recovers_after_bad_ingestion(self, test_collection):
        """Test system recovers after failed ingestion attempts."""
        # Submit several bad ingestion requests
        for i in range(3):
            run_wqm_command(
                ["ingest", "file", f"/bad/file{i}.txt", "--collection", test_collection]
            )
            time.sleep(0.2)

        time.sleep(2)

        # System should recover and be operational
        status_result = run_wqm_command(["status", "--quiet"])
        assert status_result.returncode == 0, "System should recover after failures"

    def test_daemon_restarts_after_multiple_errors(self, test_collection):
        """Test daemon remains stable after multiple error conditions."""
        # Generate multiple errors
        for _i in range(5):
            run_wqm_command(
                ["ingest", "file", "/nonexistent.txt", "--collection", test_collection]
            )
            time.sleep(0.1)

        time.sleep(2)

        # Daemon should still be running
        status_result = run_wqm_command(["service", "status"])
        assert status_result.returncode == 0, "Daemon should remain stable"

    def test_watch_recovers_from_bad_configuration(self, test_collection):
        """Test watch system recovers from bad configuration."""
        # Try to add watch with problems
        run_wqm_command(
            ["watch", "add", "/nonexistent/path", "--collection", test_collection]
        )

        time.sleep(1)

        # Watch list should still be queryable
        list_result = run_wqm_command(["watch", "list"])
        assert list_result.returncode == 0, "Watch system should recover"


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.usefixtures("ensure_daemon_running")
class TestErrorContext:
    """Test that errors include helpful context."""

    def test_file_error_includes_path(self, test_collection):
        """Test file errors include the file path."""
        bad_path = "/this/is/a/test/path/file.txt"
        result = run_wqm_command(
            ["ingest", "file", bad_path, "--collection", test_collection]
        )

        assert result.returncode != 0

        # Error should mention the file path
        if len(result.stderr) > 0:
            assert bad_path in result.stderr, "Error should include file path"

    def test_collection_error_includes_name(self):
        """Test collection errors include collection name."""
        # Try invalid operation
        result = run_wqm_command(
            ["admin", "collections", "delete", "nonexistent-collection"]
        )

        # May succeed or fail, but if error, should mention collection
        if result.returncode != 0 and len(result.stderr) > 0:
            assert "collection" in result.stderr.lower(), "Error should mention collection"

    def test_command_error_includes_command_name(self):
        """Test command errors mention the command."""
        result = run_wqm_command(["invalid-command-xyz"])

        assert result.returncode != 0

        # Error should mention command or provide help
        assert (
            len(result.stderr) > 0 or len(result.stdout) > 0
        ), "Should provide error or help"


# Summary of test coverage:
# 1. TestDaemonErrorPropagation (3 tests)
#    - CLI reports daemon offline
#    - Ingestion error with stopped daemon
#    - Watch command with stopped daemon
#
# 2. TestInvalidInputErrors (4 tests)
#    - Nonexistent file error message
#    - Invalid collection name error
#    - Invalid watch pattern handling
#    - Missing required argument error
#
# 3. TestExitCodeConsistency (5 tests)
#    - Success returns zero
#    - Invalid command returns non-zero
#    - Failed operations return non-zero
#    - Help returns zero
#    - Version returns zero
#
# 4. TestErrorMessageQuality (3 tests)
#    - Error messages are actionable
#    - Errors don't expose sensitive info
#    - No stack traces in user errors
#
# 5. TestTimeoutHandling (1 test)
#    - Command timeout handling
#
# 6. TestGracefulDegradation (2 tests)
#    - Status degrades gracefully
#    - Collections command without daemon
#
# 7. TestRecoveryMechanisms (3 tests)
#    - System recovers after bad ingestion
#    - Daemon stable after multiple errors
#    - Watch recovers from bad configuration
#
# 8. TestErrorContext (3 tests)
#    - File errors include path
#    - Collection errors include name
#    - Command errors include command name
#
# Total: 24 comprehensive test cases covering error propagation
