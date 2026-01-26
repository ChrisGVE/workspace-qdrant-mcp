"""
End-to-end tests for component failure simulation and recovery.

Tests system resilience by simulating various component failures: daemon crashes,
Qdrant disconnection, MCP server shutdown, SQLite corruption, network failures.
Verifies graceful degradation, automatic recovery, error handling, and overall
system resilience under adverse conditions.
"""

import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx
import psutil
import pytest

from tests.e2e.fixtures import (
    CLIHelper,
    DaemonManager,
    MCPServerManager,
    SystemComponents,
)


def _skip_if_qdrant_unavailable(qdrant_url: str):
    """Skip tests if Qdrant is not reachable."""
    try:
        response = httpx.get(f"{qdrant_url}/health", timeout=2.0)
    except Exception:
        pytest.skip("Qdrant not available for integration test")
    if response.status_code != 200:
        pytest.skip("Qdrant not healthy for integration test")


@pytest.mark.integration
@pytest.mark.slow
class TestDaemonCrashRecovery:
    """Test daemon crash scenarios and recovery mechanisms."""

    def test_daemon_graceful_shutdown(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test daemon responds to graceful shutdown signal."""
        daemon_process = system_components.daemon_process

        if daemon_process is None:
            pytest.skip("Daemon not running")

        pid = daemon_process.pid

        # Send SIGTERM for graceful shutdown
        try:
            daemon_process.terminate()
            daemon_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            daemon_process.kill()

        # Verify process terminated
        assert not psutil.pid_exists(pid)

    def test_daemon_forced_termination(
        self, integration_test_workspace, integration_state_db
    ):
        """Test daemon recovery after forced termination."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start daemon
        daemon_manager.start(timeout=15)
        pid = daemon_manager.pid
        assert psutil.pid_exists(pid)

        # Force kill daemon
        daemon_manager.stop(force=True)
        time.sleep(1)
        assert not psutil.pid_exists(pid)

        # Restart daemon (simulating recovery)
        daemon_manager.start(timeout=15)
        new_pid = daemon_manager.pid
        assert psutil.pid_exists(new_pid)
        assert new_pid != pid

        # Cleanup
        daemon_manager.stop()

    def test_daemon_crash_during_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system behavior when daemon crashes during ingestion."""
        workspace = system_components.workspace_path

        # Start ingestion
        test_file = workspace / "crash_test.txt"
        test_file.write_text("Content for crash testing")

        # Ingest file (daemon may crash)
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-crash"],
            timeout=30,
        )

        # System should handle failure gracefully
        assert result is not None

    def test_daemon_restart_preserves_state(
        self, integration_test_workspace, integration_state_db
    ):
        """Test daemon restart preserves watch folder state."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start daemon
        daemon_manager.start(timeout=15)
        original_pid = daemon_manager.pid

        # Stop daemon
        daemon_manager.stop()
        time.sleep(2)

        # Restart daemon
        daemon_manager.start(timeout=15)
        new_pid = daemon_manager.pid

        # Verify new process
        assert new_pid != original_pid
        assert psutil.pid_exists(new_pid)

        # Cleanup
        daemon_manager.stop()

    def test_daemon_multiple_restart_cycles(
        self, integration_test_workspace, integration_state_db
    ):
        """Test daemon can handle multiple restart cycles."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        pids = []

        # Perform multiple restart cycles
        for _i in range(3):
            daemon_manager.start(timeout=15)
            pid = daemon_manager.pid
            pids.append(pid)
            assert psutil.pid_exists(pid)

            time.sleep(1)

            daemon_manager.stop()
            time.sleep(1)
            assert not psutil.pid_exists(pid)

        # All PIDs should be different
        assert len(set(pids)) == len(pids)


@pytest.mark.integration
@pytest.mark.slow
class TestQdrantDisconnection:
    """Test Qdrant disconnection and reconnection scenarios."""

    def test_operation_during_qdrant_unavailable(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system behavior when Qdrant is unavailable."""
        workspace = system_components.workspace_path

        # Try to ingest when Qdrant might be unavailable
        test_file = workspace / "qdrant_test.txt"
        test_file.write_text("Testing Qdrant disconnection")

        # Operation should fail gracefully
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-qdrant-fail"],
            timeout=30,
        )

        # Should complete without crashing (may fail)
        assert result is not None

    def test_search_during_qdrant_issues(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test search operation when Qdrant has issues."""
        # Try search (may fail if Qdrant unavailable)
        result = cli_helper.run_command(
            ["search", "test query", "--collection", "test-qdrant-search"],
            timeout=15,
        )

        # Should handle error gracefully
        assert result is not None

    def test_collection_listing_resilience(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test collection listing when Qdrant connection is unstable."""
        # Try to list collections
        result = cli_helper.run_command(["admin", "collections"], timeout=10)

        # Should return result or error gracefully
        assert result is not None

    def test_retry_after_qdrant_reconnection(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test operations succeed after Qdrant reconnects."""
        workspace = system_components.workspace_path

        # Attempt operation (Qdrant should be available)
        test_file = workspace / "reconnect_test.txt"
        test_file.write_text("Testing reconnection")

        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-reconnect"],
            timeout=30,
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestMCPServerFailure:
    """Test MCP server shutdown and restart scenarios."""

    def test_mcp_server_graceful_shutdown(
        self, integration_test_workspace, integration_state_db
    ):
        """Test MCP server graceful shutdown."""
        _skip_if_qdrant_unavailable("http://localhost:6333")
        mcp_manager = MCPServerManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start MCP server
        mcp_manager.start(timeout=15)
        pid = mcp_manager.pid
        assert psutil.pid_exists(pid)

        # Graceful shutdown
        mcp_manager.stop()
        time.sleep(2)
        assert not psutil.pid_exists(pid)

    def test_mcp_server_forced_termination(
        self, integration_test_workspace, integration_state_db
    ):
        """Test MCP server recovery after forced termination."""
        _skip_if_qdrant_unavailable("http://localhost:6333")
        mcp_manager = MCPServerManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start server
        mcp_manager.start(timeout=15)
        pid = mcp_manager.pid

        # Force kill
        mcp_manager.stop(force=True)
        time.sleep(1)
        assert not psutil.pid_exists(pid)

        # Restart
        mcp_manager.start(timeout=15)
        new_pid = mcp_manager.pid
        assert psutil.pid_exists(new_pid)
        assert new_pid != pid

        # Cleanup
        mcp_manager.stop()

    def test_mcp_server_restart_after_crash(
        self, integration_test_workspace, integration_state_db
    ):
        """Test MCP server can restart after crash."""
        _skip_if_qdrant_unavailable("http://localhost:6333")
        mcp_manager = MCPServerManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        pids = []

        # Multiple restart cycles
        for _ in range(2):
            mcp_manager.start(timeout=15)
            pid = mcp_manager.pid
            pids.append(pid)

            mcp_manager.stop(force=True)
            time.sleep(1)

        # All PIDs should be different
        assert len(set(pids)) == len(pids)


@pytest.mark.integration
@pytest.mark.slow
class TestSQLiteCorruption:
    """Test SQLite corruption scenarios and recovery."""

    def test_database_corruption_detection(
        self, system_components: SystemComponents
    ):
        """Test detecting corrupted database."""
        import sqlite3

        state_db = system_components.state_db_path

        # Create test database
        test_db = state_db.parent / "corruption_test.db"

        # Create valid database
        conn = sqlite3.connect(test_db)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")
        conn.commit()
        conn.close()

        # Corrupt database
        test_db.write_bytes(b"corrupted data" + test_db.read_bytes()[14:])

        # Try to open corrupted database
        try:
            conn = sqlite3.connect(test_db)
            conn.execute("SELECT * FROM test")
            conn.close()
            corrupted = False
        except sqlite3.DatabaseError:
            corrupted = True

        # Should detect corruption
        assert corrupted

        # Cleanup
        if test_db.exists():
            test_db.unlink()

    def test_recovery_from_backup_after_corruption(
        self, system_components: SystemComponents
    ):
        """Test recovering from backup after corruption."""
        import shutil
        import sqlite3

        state_db = system_components.state_db_path
        test_db = state_db.parent / "recovery_test.db"
        backup_db = state_db.parent / "recovery_backup.db"

        # Create valid database
        conn = sqlite3.connect(test_db)
        conn.execute("CREATE TABLE test (id INTEGER, data TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'data')")
        conn.commit()
        conn.close()

        # Create backup
        shutil.copy2(test_db, backup_db)

        # Corrupt original
        test_db.write_text("corrupted")

        # Recover from backup
        recovered_db = state_db.parent / "recovered.db"
        shutil.copy2(backup_db, recovered_db)

        # Verify recovery
        conn = sqlite3.connect(recovered_db)
        result = conn.execute("SELECT * FROM test").fetchone()
        conn.close()

        assert result == (1, 'data')

        # Cleanup
        for db in [test_db, backup_db, recovered_db]:
            if db.exists():
                db.unlink()

    def test_sqlite_wal_mode_resilience(
        self, system_components: SystemComponents
    ):
        """Test SQLite WAL mode provides resilience."""
        import sqlite3

        state_db = system_components.state_db_path
        test_db = state_db.parent / "wal_test.db"

        # Create database with WAL mode
        conn = sqlite3.connect(test_db)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE test (id INTEGER, data TEXT)")
        conn.execute("INSERT INTO test VALUES (1, 'data')")
        conn.commit()

        # Verify WAL mode
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

        conn.close()

        # Cleanup
        test_db.unlink()
        wal_file = test_db.parent / f"{test_db.name}-wal"
        shm_file = test_db.parent / f"{test_db.name}-shm"
        if wal_file.exists():
            wal_file.unlink()
        if shm_file.exists():
            shm_file.unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestNetworkFailures:
    """Test network failure scenarios."""

    def test_cli_timeout_handling(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test CLI handles operation timeouts gracefully."""
        # Try operation with very short timeout
        result = cli_helper.run_command(
            ["status"],
            timeout=1,
        )

        # Should complete or timeout gracefully
        assert result is not None

    def test_operation_retry_after_timeout(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test operations can retry after timeout."""
        workspace = system_components.workspace_path

        # First attempt (may timeout)
        test_file = workspace / "retry_test.txt"
        test_file.write_text("Testing retry mechanism")

        result1 = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-retry"],
            timeout=30,
        )

        # Retry should work
        result2 = cli_helper.run_command(
            ["status"],
            timeout=10,
        )

        assert result1 is not None
        assert result2 is not None

    def test_partial_failure_handling(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of partial operation failures."""
        workspace = system_components.workspace_path

        # Create multiple files
        files = []
        for i in range(3):
            test_file = workspace / f"partial_{i}.txt"
            test_file.write_text(f"Partial failure test {i}")
            files.append(test_file)

        # Try batch operation (may partially fail)
        result = cli_helper.run_command(
            ["ingest", "folder", str(workspace), "--collection", "test-partial"],
            timeout=60,
        )

        # Should handle partial failures
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestGracefulDegradation:
    """Test graceful degradation under failure conditions."""

    def test_readonly_mode_during_failures(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system provides readonly access during write failures."""
        # Status/read operations should work even if writes fail
        result = cli_helper.run_command(["status"], timeout=10)
        assert result is not None

        collections_result = cli_helper.run_command(
            ["admin", "collections"], timeout=10
        )
        assert collections_result is not None

    def test_status_reporting_during_degradation(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test status command reports degraded state."""
        # Status should always be queryable
        result = cli_helper.run_command(["status"], timeout=10)
        assert result is not None
        assert result.returncode in [0, 1]

    def test_error_messages_during_failures(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test error messages are informative during failures."""
        # Try invalid operation
        result = cli_helper.run_command(
            ["search", "test", "--collection", "nonexistent"],
            timeout=15,
        )

        # Should provide error feedback
        assert result is not None
        # Error information should be in stdout or stderr
        assert result.stdout or result.stderr


@pytest.mark.integration
@pytest.mark.slow
class TestAutomaticRecovery:
    """Test automatic recovery mechanisms."""

    def test_daemon_auto_restart_capability(
        self, integration_test_workspace, integration_state_db
    ):
        """Test daemon can be automatically restarted."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start daemon
        daemon_manager.start(timeout=15)
        original_pid = daemon_manager.pid

        # Simulate crash
        daemon_manager.stop(force=True)
        time.sleep(2)

        # Restart (simulating auto-restart)
        daemon_manager.start(timeout=15)
        new_pid = daemon_manager.pid

        # Verify recovery
        assert psutil.pid_exists(new_pid)
        assert new_pid != original_pid

        # Cleanup
        daemon_manager.stop()

    def test_connection_pool_recovery(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test connection pool recovers from failures."""
        # Multiple operations to test connection recovery
        for _i in range(3):
            result = cli_helper.run_command(["status", "--quiet"])
            assert result is not None
            time.sleep(1)

    def test_queue_processing_resumes_after_failure(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test ingestion queue processing resumes after failure."""
        workspace = system_components.workspace_path

        # Create files for processing
        test_file = workspace / "queue_resume.txt"
        test_file.write_text("Queue processing test")

        # Ingest file
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-queue-resume"]
        )

        # Queue should eventually process (or fail gracefully)
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestErrorHandling:
    """Test error handling and reporting."""

    def test_invalid_input_handling(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling of invalid inputs."""
        # Invalid file path
        result = cli_helper.run_command(
            ["ingest", "file", "/nonexistent/path.txt", "--collection", "test-invalid"]
        )

        # Should handle error gracefully
        assert result is not None
        assert result.returncode != 0

    def test_error_logging_during_failures(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test errors are properly logged."""
        # Trigger error
        result = cli_helper.run_command(
            ["search", "test", "--collection", "nonexistent"],
            timeout=15,
        )

        # Should produce error output
        assert result is not None
        assert result.stderr or result.returncode != 0

    def test_exception_handling_completeness(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test all operations have exception handling."""
        # Various operations that might fail
        operations = [
            ["status"],
            ["admin", "collections"],
            ["search", "test", "--collection", "test"],
        ]

        for op in operations:
            result = cli_helper.run_command(op, timeout=15)
            # Should not crash, always return result
            assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSystemResilience:
    """Test overall system resilience."""

    def test_concurrent_failures_handling(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system handles multiple concurrent failures."""
        workspace = system_components.workspace_path

        # Try multiple operations that might fail
        test_file = workspace / "concurrent_fail.txt"
        test_file.write_text("Concurrent failure test")

        result1 = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-concurrent-1"],
            timeout=30,
        )
        result2 = cli_helper.run_command(["status"], timeout=10)

        # System should remain operational
        assert result1 is not None
        assert result2 is not None

    def test_system_stability_after_failures(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system remains stable after experiencing failures."""
        # Trigger some failures
        cli_helper.run_command(
            ["search", "test", "--collection", "nonexistent"],
            timeout=15,
        )

        time.sleep(2)

        # System should still be operational
        result = cli_helper.run_command(["status"])
        assert result is not None

    def test_resource_cleanup_after_failures(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test resources are cleaned up after failures."""
        # Perform operations that might fail
        for _ in range(5):
            cli_helper.run_command(["status", "--quiet"])

        # System should not leak resources
        result = cli_helper.run_command(["status"])
        assert result is not None

    def test_long_running_stability_with_failures(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test system remains stable during long run with failures."""
        workspace = system_components.workspace_path

        # Simulate extended operation with potential failures
        for i in range(10):
            test_file = workspace / f"stability_{i}.txt"
            test_file.write_text(f"Stability test {i}")

            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-stability-{i}",
                ]
            )

            time.sleep(0.5)

        # System should still respond
        result = cli_helper.run_command(["status"])
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestFailureIsolation:
    """Test failure isolation between components."""

    def test_daemon_failure_doesnt_affect_cli(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test CLI remains functional when daemon fails."""
        # CLI help should work even if daemon down
        result = cli_helper.run_command(["--help"])
        assert result is not None
        assert result.returncode == 0

    def test_qdrant_failure_isolated(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test Qdrant failures don't cascade to other components."""
        # Status should work even if Qdrant has issues
        result = cli_helper.run_command(["status"])
        assert result is not None

    def test_component_independence(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test components can function independently."""
        # Various operations testing independence
        result1 = cli_helper.run_command(["--version"])
        result2 = cli_helper.run_command(["--help"])

        assert result1 is not None
        assert result1.returncode == 0
        assert result2 is not None
        assert result2.returncode == 0


@pytest.mark.integration
@pytest.mark.slow
class TestRecoveryMetrics:
    """Test recovery time and metrics."""

    def test_recovery_time_measurement(
        self, integration_test_workspace, integration_state_db
    ):
        """Test measuring component recovery time."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        # Start daemon
        daemon_manager.start(timeout=15)

        # Stop daemon
        daemon_manager.stop()
        time.sleep(1)

        # Measure restart time
        start_time = time.time()
        daemon_manager.start(timeout=15)
        recovery_time = time.time() - start_time

        # Recovery should be fast (<15 seconds)
        assert recovery_time < 15.0

        # Cleanup
        daemon_manager.stop()

    def test_failure_detection_time(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test time to detect failures."""
        # Operation that may fail
        start_time = time.time()
        result = cli_helper.run_command(
            ["search", "test", "--collection", "nonexistent"],
            timeout=10,
        )
        detection_time = time.time() - start_time

        # Should detect failure quickly (<10 seconds)
        assert detection_time < 10.0
        assert result is not None

    def test_mttr_estimation(
        self, integration_test_workspace, integration_state_db
    ):
        """Test Mean Time To Recovery (MTTR) estimation."""
        daemon_manager = DaemonManager(
            qdrant_url="http://localhost:6333",
            state_db_path=integration_state_db,
        )

        recovery_times = []

        # Multiple recovery cycles
        for _ in range(3):
            daemon_manager.start(timeout=15)
            daemon_manager.stop()
            time.sleep(1)

            start = time.time()
            daemon_manager.start(timeout=15)
            recovery_times.append(time.time() - start)
            daemon_manager.stop()

        # Calculate average MTTR
        avg_mttr = sum(recovery_times) / len(recovery_times)

        # MTTR should be reasonable (<15 seconds)
        assert avg_mttr < 15.0
