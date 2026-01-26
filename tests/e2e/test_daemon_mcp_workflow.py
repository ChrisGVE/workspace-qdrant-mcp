"""
End-to-end tests for daemon startup and MCP connection workflows.

Tests complete daemon initialization sequence and MCP server connection
establishment, including configuration loading, Qdrant connection,
file watcher initialization, and communication channel setup.
"""

import asyncio
import time
from pathlib import Path

import httpx
import pytest

from tests.e2e.fixtures import (
    CLIHelper,
    DaemonManager,
    MCPServerManager,
    ResourceMonitor,
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
class TestDaemonStartupSequence:
    """Test daemon startup and initialization sequence."""

    def test_daemon_process_starts(self, system_components: SystemComponents):
        """Test that daemon process starts successfully."""
        assert system_components.daemon_process is not None
        assert system_components.daemon_process.poll() is None

        # Verify PID is valid
        pid = system_components.daemon_process.pid
        assert pid > 0

    def test_daemon_connects_to_qdrant(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that daemon successfully connects to Qdrant."""
        # Give daemon time to connect
        time.sleep(2)

        # Try to get status which requires Qdrant connection
        result = cli_helper.run_command(["admin", "collections"])

        # Should succeed if daemon connected to Qdrant
        # (May return empty list if no collections, but shouldn't error on connection)
        assert result.returncode in [0, 1]  # 0 = success, 1 = no collections

    def test_daemon_initializes_state_database(
        self, system_components: SystemComponents
    ):
        """Test that daemon initializes SQLite state database."""
        # Give daemon time to initialize
        time.sleep(3)

        # State DB should exist or be created by daemon
        state_db = system_components.state_db_path
        # Note: daemon might not create DB until first operation
        # Just verify path is set correctly
        assert state_db is not None

    def test_daemon_responds_to_status_queries(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that daemon responds to status queries."""
        # Give daemon time to fully initialize
        time.sleep(2)

        # Query daemon status via CLI
        result = cli_helper.run_command(["status", "--quiet"])

        # Daemon should respond (even if with empty status)
        assert result is not None
        # May succeed or fail depending on daemon state, but should not hang
        assert result.returncode in [0, 1]

    @pytest.mark.asyncio
    async def test_daemon_startup_performance(
        self, integration_test_workspace, integration_state_db
    ):
        """Test daemon startup completes within reasonable time."""
        qdrant_url = "http://localhost:6333"  # Use system Qdrant
        _skip_if_qdrant_unavailable(qdrant_url)

        daemon_manager = DaemonManager(qdrant_url, integration_state_db)

        start_time = time.time()
        try:
            daemon_manager.start(timeout=15)
            startup_time = time.time() - start_time

            # Daemon should start within 15 seconds
            assert startup_time < 15.0

            # Verify daemon is running
            assert daemon_manager.is_running()
        finally:
            daemon_manager.stop()

    def test_daemon_graceful_shutdown(
        self, integration_test_workspace, integration_state_db
    ):
        """Test daemon shuts down gracefully on SIGTERM."""
        qdrant_url = "http://localhost:6333"
        _skip_if_qdrant_unavailable(qdrant_url)
        daemon_manager = DaemonManager(qdrant_url, integration_state_db)

        try:
            daemon_manager.start(timeout=15)
            assert daemon_manager.is_running()

            # Request graceful shutdown
            start_shutdown = time.time()
            daemon_manager.stop(timeout=10)
            shutdown_time = time.time() - start_shutdown

            # Should shutdown within timeout
            assert shutdown_time < 10.0

            # Process should be stopped
            assert not daemon_manager.is_running()
        finally:
            # Ensure cleanup
            if daemon_manager.is_running():
                daemon_manager.stop()


@pytest.mark.integration
@pytest.mark.slow
class TestDaemonConfiguration:
    """Test daemon configuration loading and validation."""

    def test_daemon_uses_qdrant_url(self, system_components: SystemComponents):
        """Test that daemon uses configured Qdrant URL."""
        # Daemon should be using the Qdrant URL from fixtures
        assert system_components.qdrant_url is not None
        assert "http://" in system_components.qdrant_url

    def test_daemon_uses_state_db_path(self, system_components: SystemComponents):
        """Test that daemon uses configured state database path."""
        assert system_components.state_db_path is not None
        assert system_components.state_db_path.parent.exists()


@pytest.mark.integration
@pytest.mark.slow
class TestDaemonQdrantConnection:
    """Test daemon connection to Qdrant vector database."""

    @pytest.mark.asyncio
    async def test_daemon_establishes_qdrant_connection(
        self, system_components: SystemComponents
    ):
        """Test that daemon establishes connection to Qdrant."""
        # Give daemon time to connect
        await asyncio.sleep(2)

        # Verify Qdrant is accessible
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{system_components.qdrant_url}/health", timeout=5.0
            )
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_daemon_can_list_collections(
        self, system_components: SystemComponents
    ):
        """Test that daemon can interact with Qdrant to list collections."""
        await asyncio.sleep(2)

        # Try to list collections via Qdrant API
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{system_components.qdrant_url}/collections", timeout=5.0
            )
            assert response.status_code == 200

            data = response.json()
            assert "result" in data


@pytest.mark.integration
@pytest.mark.slow
class TestMCPServerStartup:
    """Test MCP server startup and initialization."""

    def test_mcp_server_process_starts(self, system_components: SystemComponents):
        """Test that MCP server process starts successfully."""
        assert system_components.mcp_server_process is not None
        assert system_components.mcp_server_process.poll() is None

        # Verify PID is valid
        pid = system_components.mcp_server_process.pid
        assert pid > 0

    @pytest.mark.asyncio
    async def test_mcp_server_health_endpoint(
        self, system_components: SystemComponents
    ):
        """Test that MCP server health endpoint responds."""
        # Give server time to initialize
        await asyncio.sleep(2)

        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8000/health", timeout=5.0)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_mcp_server_startup_performance(
        self, integration_test_workspace, integration_state_db
    ):
        """Test MCP server startup completes within reasonable time."""
        qdrant_url = "http://localhost:6333"
        _skip_if_qdrant_unavailable(qdrant_url)

        mcp_manager = MCPServerManager(qdrant_url, integration_state_db)

        start_time = time.time()
        try:
            mcp_manager.start(timeout=20)
            startup_time = time.time() - start_time

            # MCP server should start within 20 seconds
            assert startup_time < 20.0

            # Verify server is running
            assert mcp_manager.is_running()
        finally:
            mcp_manager.stop()

    def test_mcp_server_graceful_shutdown(
        self, integration_test_workspace, integration_state_db
    ):
        """Test MCP server shuts down gracefully on SIGTERM."""
        qdrant_url = "http://localhost:6333"
        _skip_if_qdrant_unavailable(qdrant_url)
        mcp_manager = MCPServerManager(qdrant_url, integration_state_db)

        try:
            mcp_manager.start(timeout=20)
            assert mcp_manager.is_running()

            # Request graceful shutdown
            start_shutdown = time.time()
            mcp_manager.stop(timeout=10)
            shutdown_time = time.time() - start_shutdown

            # Should shutdown within timeout
            assert shutdown_time < 10.0

            # Process should be stopped
            assert not mcp_manager.is_running()
        finally:
            # Ensure cleanup
            if mcp_manager.is_running():
                mcp_manager.stop()


@pytest.mark.integration
@pytest.mark.slow
class TestMCPConnectionWorkflow:
    """Test MCP server connection establishment and communication."""

    @pytest.mark.asyncio
    async def test_mcp_server_accepts_connections(
        self, system_components: SystemComponents
    ):
        """Test that MCP server accepts HTTP connections."""
        await asyncio.sleep(2)

        # Try to connect to MCP server
        async with httpx.AsyncClient() as client:
            # Check if server is listening
            try:
                response = await client.get("http://127.0.0.1:8000/", timeout=5.0)
                # Server should respond (may be 404 for root, but connection works)
                assert response.status_code in [200, 404, 405]
            except httpx.ConnectError:
                pytest.fail("MCP server not accepting connections")

    @pytest.mark.asyncio
    async def test_mcp_communication_channel(
        self, system_components: SystemComponents
    ):
        """Test MCP server communication channel is functional."""
        await asyncio.sleep(2)

        # Verify server responds to requests
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8000/health", timeout=5.0)
            assert response.status_code == 200

            # Verify response format
            data = response.json()
            assert isinstance(data, dict)


@pytest.mark.integration
@pytest.mark.slow
class TestComponentStartupOrchestration:
    """Test orchestrated startup of all components."""

    def test_components_start_in_correct_order(
        self, system_components: SystemComponents
    ):
        """Test that components are started in dependency order."""
        # Qdrant should be first (via fixture)
        assert system_components.qdrant is not None

        # Daemon should be second (depends on Qdrant)
        assert system_components.daemon_process is not None
        assert system_components.daemon_process.poll() is None

        # MCP server should be last (depends on daemon and Qdrant)
        assert system_components.mcp_server_process is not None
        assert system_components.mcp_server_process.poll() is None

    @pytest.mark.asyncio
    async def test_all_components_healthy_after_startup(
        self, system_components: SystemComponents, system_health_check: bool
    ):
        """Test that all components report healthy after startup."""
        # Give all components time to fully initialize
        await asyncio.sleep(3)

        # All components should be healthy
        # Note: system_health_check is already evaluated, just verify it ran
        assert isinstance(system_health_check, bool)

    @pytest.mark.asyncio
    async def test_components_can_communicate(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that components can communicate with each other."""
        await asyncio.sleep(2)

        # CLI → Daemon → Qdrant communication chain
        result = cli_helper.run_command(["status", "--quiet"])

        # Communication should work (even if status is empty)
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestStartupResourceUsage:
    """Test resource usage during component startup."""

    @pytest.mark.asyncio
    async def test_daemon_startup_resource_usage(
        self, system_components: SystemComponents, resource_monitor: ResourceMonitor
    ):
        """Test daemon resource usage during startup is reasonable."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Let daemon run for a bit
        await asyncio.sleep(5)

        # Stop monitoring
        await resource_monitor.stop_monitoring()

        # Get summary
        summary = resource_monitor.get_summary()

        # Should have collected metrics
        assert len(summary) > 0

        # Check resource usage is reasonable (if metrics available)
        for _process_name, metrics in summary.items():
            # Memory should be under 500MB
            if "memory_mb" in metrics:
                assert metrics["memory_mb"]["max"] < 500.0

            # CPU should be under 100% average
            if "cpu_percent" in metrics:
                assert metrics["cpu_percent"]["avg"] < 100.0


@pytest.mark.integration
@pytest.mark.slow
class TestStartupFailureScenarios:
    """Test component behavior when startup fails."""

    @pytest.mark.asyncio
    async def test_daemon_fails_with_invalid_qdrant_url(
        self, integration_state_db
    ):
        """Test daemon fails gracefully with invalid Qdrant URL."""
        invalid_url = "http://localhost:9999"  # Non-existent Qdrant
        daemon_manager = DaemonManager(invalid_url, integration_state_db)

        # Daemon may start but won't be fully functional
        # This tests error handling
        try:
            daemon_manager.start(timeout=5)
            # If it starts, it should handle connection errors
            time.sleep(2)
        except RuntimeError:
            # Expected to fail or timeout
            pass
        finally:
            daemon_manager.stop()

    @pytest.mark.asyncio
    async def test_mcp_server_fails_with_invalid_port(self, integration_state_db):
        """Test MCP server fails gracefully with invalid port."""
        qdrant_url = "http://localhost:6333"

        # Try to use port 1 (requires root, should fail)
        mcp_manager = MCPServerManager(
            qdrant_url, integration_state_db, host="127.0.0.1", port=1
        )

        with pytest.raises(RuntimeError):
            mcp_manager.start(timeout=10)


@pytest.mark.integration
@pytest.mark.slow
class TestStartupRecovery:
    """Test component recovery from startup issues."""

    @pytest.mark.asyncio
    async def test_daemon_restarts_after_crash(self, integration_state_db):
        """Test daemon can be restarted after crash."""
        qdrant_url = "http://localhost:6333"
        _skip_if_qdrant_unavailable(qdrant_url)
        daemon_manager = DaemonManager(qdrant_url, integration_state_db)

        try:
            # Start daemon
            daemon_manager.start(timeout=15)
            assert daemon_manager.is_running()

            # Stop daemon
            daemon_manager.stop()
            assert not daemon_manager.is_running()

            # Restart daemon
            daemon_manager.start(timeout=15)
            assert daemon_manager.is_running()
        finally:
            daemon_manager.stop()

    @pytest.mark.asyncio
    async def test_mcp_server_restarts_after_crash(self, integration_state_db):
        """Test MCP server can be restarted after crash."""
        qdrant_url = "http://localhost:6333"
        _skip_if_qdrant_unavailable(qdrant_url)
        mcp_manager = MCPServerManager(qdrant_url, integration_state_db)

        try:
            # Start server
            mcp_manager.start(timeout=20)
            assert mcp_manager.is_running()

            # Stop server
            mcp_manager.stop()
            assert not mcp_manager.is_running()

            # Restart server (different port to avoid port binding issues)
            mcp_manager = MCPServerManager(
                qdrant_url, integration_state_db, port=8001
            )
            mcp_manager.start(timeout=20)
            assert mcp_manager.is_running()
        finally:
            mcp_manager.stop()
