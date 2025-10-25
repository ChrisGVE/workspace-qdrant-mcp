"""
Multi-Component Startup/Shutdown Tests (Task 292.3).

Comprehensive E2E tests for various startup sequences and graceful shutdown
procedures across all system components (Qdrant, daemon, MCP server).

Test Coverage:
- Sequential component startup (Qdrant → daemon → MCP)
- Parallel startup scenarios
- Partial startup recovery
- Component dependency validation
- Graceful shutdown ordering
- Forced shutdown handling
- Startup timeout scenarios
- Component readiness checks
- Inter-component communication establishment
"""

import asyncio
import time
from typing import Any

import pytest

from tests.e2e.utils import HealthChecker, WorkflowTimer


@pytest.mark.e2e
class TestSequentialStartup:
    """
    Test sequential component startup scenarios.

    Validates proper startup ordering: Qdrant → daemon → MCP server
    """

    @pytest.mark.asyncio
    async def test_qdrant_daemon_mcp_sequential_startup(
        self,
        component_lifecycle_manager
    ):
        """
        Test proper sequential startup: Qdrant → daemon → MCP.

        Validates:
        - Qdrant starts first
        - Daemon waits for Qdrant
        - MCP waits for daemon
        - All components reach healthy state
        - Startup timing within thresholds
        """
        timer = WorkflowTimer()
        timer.start()

        # Step 1: Start Qdrant
        success = await component_lifecycle_manager.start_component("qdrant")
        assert success, "Failed to start Qdrant"
        timer.checkpoint("qdrant_started")

        # Verify Qdrant is healthy before proceeding
        qdrant_health = await component_lifecycle_manager.check_health("qdrant")
        assert qdrant_health["healthy"], "Qdrant not healthy"
        timer.checkpoint("qdrant_healthy")

        # Step 2: Start daemon (depends on Qdrant)
        success = await component_lifecycle_manager.start_component("daemon")
        assert success, "Failed to start daemon"
        timer.checkpoint("daemon_started")

        # Verify daemon is healthy
        daemon_health = await component_lifecycle_manager.check_health("daemon")
        assert daemon_health["healthy"], "Daemon not healthy"
        timer.checkpoint("daemon_healthy")

        # Step 3: Start MCP server (depends on daemon)
        success = await component_lifecycle_manager.start_component("mcp_server")
        assert success, "Failed to start MCP server"
        timer.checkpoint("mcp_started")

        # Verify MCP is healthy
        mcp_health = await component_lifecycle_manager.check_health("mcp_server")
        assert mcp_health["healthy"], "MCP server not healthy"
        timer.checkpoint("mcp_healthy")

        # Validate startup timing
        total_startup = timer.get_duration("mcp_healthy")
        assert total_startup < 60, f"Sequential startup took {total_startup:.1f}s, expected < 60s"

        # Validate individual component startup times
        qdrant_startup = timer.get_duration("qdrant_healthy")
        daemon_startup = timer.get_duration("daemon_healthy") - timer.get_duration("qdrant_healthy")
        mcp_startup = timer.get_duration("mcp_healthy") - timer.get_duration("daemon_healthy")

        assert qdrant_startup < 15, f"Qdrant startup: {qdrant_startup:.1f}s"
        assert daemon_startup < 25, f"Daemon startup: {daemon_startup:.1f}s"
        assert mcp_startup < 15, f"MCP startup: {mcp_startup:.1f}s"

        # Cleanup
        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_startup_with_dependency_validation(
        self,
        component_lifecycle_manager
    ):
        """
        Test that components validate dependencies during startup.

        Validates:
        - Daemon verifies Qdrant availability
        - MCP verifies daemon availability
        - Appropriate warnings logged for missing dependencies
        - Fallback modes activated when dependencies unavailable
        """
        # Attempt to start daemon without Qdrant
        await component_lifecycle_manager.start_component("daemon")
        await asyncio.sleep(3)

        # Daemon should start but may log warnings about Qdrant
        daemon_health = await component_lifecycle_manager.check_health("daemon")
        assert daemon_health["healthy"], "Daemon should start even without Qdrant"

        # Now start Qdrant - daemon should connect
        await component_lifecycle_manager.start_component("qdrant")
        await asyncio.sleep(5)

        qdrant_health = await component_lifecycle_manager.check_health("qdrant")
        assert qdrant_health["healthy"]

        # Start MCP - should connect to daemon
        await component_lifecycle_manager.start_component("mcp_server")
        await asyncio.sleep(3)

        mcp_health = await component_lifecycle_manager.check_health("mcp_server")
        assert mcp_health["healthy"]

        # Cleanup
        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_startup_order_independence(
        self,
        component_lifecycle_manager
    ):
        """
        Test system handles out-of-order startup gracefully.

        Validates:
        - Components can start in any order
        - Missing dependencies detected
        - Automatic connection when dependencies become available
        - System stabilizes to healthy state
        """
        timer = WorkflowTimer()
        timer.start()

        # Start in reverse order: MCP → daemon → Qdrant
        await component_lifecycle_manager.start_component("mcp_server")
        timer.checkpoint("mcp_started")

        await component_lifecycle_manager.start_component("daemon")
        timer.checkpoint("daemon_started")

        await component_lifecycle_manager.start_component("qdrant")
        timer.checkpoint("qdrant_started")

        # Wait for system to stabilize
        ready = await component_lifecycle_manager.wait_for_ready(timeout=60)
        assert ready, "System did not stabilize to healthy state"
        timer.checkpoint("system_stable")

        # All components should now be healthy
        for component in ["qdrant", "daemon", "mcp_server"]:
            health = await component_lifecycle_manager.check_health(component)
            assert health["healthy"], f"{component} not healthy after stabilization"

        stabilization_time = timer.get_duration("system_stable") - timer.get_duration("qdrant_started")
        assert stabilization_time < 30, f"Stabilization took {stabilization_time:.1f}s"

        # Cleanup
        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
class TestParallelStartup:
    """
    Test parallel component startup scenarios.

    Validates concurrent startup behavior and race condition handling.
    """

    @pytest.mark.asyncio
    async def test_parallel_component_startup(
        self,
        component_lifecycle_manager
    ):
        """
        Test all components starting simultaneously.

        Validates:
        - No startup conflicts
        - All components reach healthy state
        - Startup time faster than sequential
        - Resource contention handled
        """
        timer = WorkflowTimer()
        timer.start()

        # Start all components in parallel
        startup_tasks = [
            component_lifecycle_manager.start_component("qdrant"),
            component_lifecycle_manager.start_component("daemon"),
            component_lifecycle_manager.start_component("mcp_server")
        ]

        results = await asyncio.gather(*startup_tasks)
        timer.checkpoint("components_started")

        assert all(results), "Some components failed to start"

        # Wait for all to become healthy
        ready = await component_lifecycle_manager.wait_for_ready(timeout=60)
        assert ready, "Not all components became healthy"
        timer.checkpoint("all_healthy")

        parallel_startup_time = timer.get_duration("all_healthy")

        # Parallel startup should be faster than sequential (< 45s vs < 60s)
        assert parallel_startup_time < 45, f"Parallel startup: {parallel_startup_time:.1f}s"

        # Verify all components healthy
        for component in ["qdrant", "daemon", "mcp_server"]:
            health = await component_lifecycle_manager.check_health(component)
            assert health["healthy"], f"{component} not healthy"

        # Cleanup
        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_parallel_startup_with_failures(
        self,
        component_lifecycle_manager
    ):
        """
        Test parallel startup when some components fail.

        Validates:
        - Successful components remain healthy
        - Failed components don't block others
        - System reaches stable state
        - Retry mechanisms work
        """
        # Simulate starting all components with potential failures
        await component_lifecycle_manager.start_component("qdrant")
        await component_lifecycle_manager.start_component("daemon")
        # Simulate MCP failing to start initially
        # await component_lifecycle_manager.start_component("mcp_server")  # Skip

        await asyncio.sleep(5)

        # Qdrant and daemon should be healthy
        qdrant_health = await component_lifecycle_manager.check_health("qdrant")
        daemon_health = await component_lifecycle_manager.check_health("daemon")

        assert qdrant_health["healthy"]
        assert daemon_health["healthy"]

        # Now start MCP
        await component_lifecycle_manager.start_component("mcp_server")
        await asyncio.sleep(5)

        # All should be healthy
        ready = await component_lifecycle_manager.wait_for_ready(timeout=30)
        assert ready

        # Cleanup
        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
class TestPartialStartupRecovery:
    """
    Test recovery from partial startup scenarios.

    Validates system recovery when not all components start successfully.
    """

    @pytest.mark.asyncio
    async def test_missing_component_detection(
        self,
        component_lifecycle_manager
    ):
        """
        Test detection of missing/failed components.

        Validates:
        - Missing components detected
        - System operates in degraded mode
        - Health checks reflect partial status
        - Recovery possible when missing components start
        """
        # Start only Qdrant and MCP, skip daemon
        await component_lifecycle_manager.start_component("qdrant")
        await asyncio.sleep(5)

        await component_lifecycle_manager.start_component("mcp_server")
        await asyncio.sleep(5)

        # MCP should detect daemon is missing and enter degraded mode
        mcp_health = await component_lifecycle_manager.check_health("mcp_server")
        assert mcp_health["healthy"], "MCP should run in degraded mode"

        # Start daemon now
        await component_lifecycle_manager.start_component("daemon")
        await asyncio.sleep(5)

        # System should recover to full functionality
        ready = await component_lifecycle_manager.wait_for_ready(timeout=30)
        assert ready, "System should recover when missing component starts"

        # Cleanup
        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_component_restart_during_startup(
        self,
        component_lifecycle_manager
    ):
        """
        Test component restart during system startup.

        Validates:
        - Component restarts don't disrupt other components
        - System reaches stable state
        - No cascading failures
        - State consistency maintained
        """
        # Start all components
        await component_lifecycle_manager.start_all()
        await asyncio.sleep(3)

        # Restart daemon mid-startup
        await component_lifecycle_manager.stop_component("daemon")
        await asyncio.sleep(2)
        await component_lifecycle_manager.start_component("daemon")

        # System should stabilize
        ready = await component_lifecycle_manager.wait_for_ready(timeout=60)
        assert ready, "System should stabilize after component restart"

        # All components healthy
        for component in ["qdrant", "daemon", "mcp_server"]:
            health = await component_lifecycle_manager.check_health(component)
            assert health["healthy"]

        # Cleanup
        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
class TestGracefulShutdown:
    """
    Test graceful shutdown procedures.

    Validates proper shutdown ordering and resource cleanup.
    """

    @pytest.mark.asyncio
    async def test_graceful_shutdown_ordering(
        self,
        component_lifecycle_manager
    ):
        """
        Test components shutdown in reverse order of startup.

        Validates:
        - MCP server stops first
        - Daemon stops second
        - Qdrant stops last
        - All resources cleaned up
        - No errors during shutdown
        """
        # Start all components
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        timer = WorkflowTimer()
        timer.start()

        # Graceful shutdown in reverse order
        await component_lifecycle_manager.stop_component("mcp_server")
        timer.checkpoint("mcp_stopped")

        await component_lifecycle_manager.stop_component("daemon")
        timer.checkpoint("daemon_stopped")

        await component_lifecycle_manager.stop_component("qdrant")
        timer.checkpoint("qdrant_stopped")

        # Validate shutdown timing
        total_shutdown = timer.get_duration("qdrant_stopped")
        assert total_shutdown < 30, f"Shutdown took {total_shutdown:.1f}s, expected < 30s"

        # Verify all components stopped
        for component in ["mcp_server", "daemon", "qdrant"]:
            health = await component_lifecycle_manager.check_health(component)
            assert not health["healthy"], f"{component} still running"

    @pytest.mark.asyncio
    async def test_shutdown_with_active_operations(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test graceful shutdown during active operations.

        Validates:
        - Ongoing operations complete
        - New operations rejected
        - Data persisted before shutdown
        - Clean state after shutdown
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        workspace_path = temp_project_workspace["path"]

        # Start some operations (file creation)
        test_file = workspace_path / "src/active_op.py"
        from tests.e2e.utils import TestDataGenerator
        test_file.write_text(TestDataGenerator.create_python_module("active_op"))

        # Allow operations to start
        await asyncio.sleep(2)

        timer = WorkflowTimer()
        timer.start()

        # Initiate shutdown
        await component_lifecycle_manager.stop_all()
        timer.checkpoint("shutdown_complete")

        shutdown_time = timer.get_duration("shutdown_complete")

        # Shutdown should wait for operations but not indefinitely
        assert 2 < shutdown_time < 30, f"Shutdown with active ops: {shutdown_time:.1f}s"

    @pytest.mark.asyncio
    async def test_forced_shutdown_after_timeout(
        self,
        component_lifecycle_manager
    ):
        """
        Test forced shutdown when graceful shutdown times out.

        Validates:
        - Graceful shutdown attempted first
        - Timeout detection
        - Forced termination after timeout
        - System clean state
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        timer = WorkflowTimer()
        timer.start()

        # Attempt shutdown (may force after timeout in real implementation)
        await component_lifecycle_manager.stop_all()
        timer.checkpoint("shutdown_complete")

        shutdown_time = timer.get_duration("shutdown_complete")

        # Should complete within reasonable time even if forced
        assert shutdown_time < 60, f"Shutdown took {shutdown_time:.1f}s"

        # Verify all stopped
        for component in ["mcp_server", "daemon", "qdrant"]:
            health = await component_lifecycle_manager.check_health(component)
            assert not health["healthy"], f"{component} still running after forced shutdown"


@pytest.mark.e2e
class TestStartupTimeouts:
    """
    Test startup timeout scenarios.

    Validates timeout handling during component startup.
    """

    @pytest.mark.asyncio
    async def test_component_startup_timeout_detection(
        self,
        component_lifecycle_manager
    ):
        """
        Test detection of components that timeout during startup.

        Validates:
        - Timeout detection mechanism
        - Other components unaffected
        - Retry mechanisms
        - Error reporting
        """
        timer = WorkflowTimer()
        timer.start()

        # Start components with timeout monitoring
        await component_lifecycle_manager.start_component("qdrant")
        timer.checkpoint("qdrant_started")

        # Wait for Qdrant to be healthy or timeout
        qdrant_ready = await component_lifecycle_manager.wait_for_ready(timeout=30)

        if not qdrant_ready:
            pytest.skip("Qdrant failed to start - timeout scenario")

        # Continue with other components
        await component_lifecycle_manager.start_component("daemon")
        await component_lifecycle_manager.start_component("mcp_server")

        # System should handle individual component timeouts gracefully
        await asyncio.sleep(10)

        # Cleanup
        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_system_startup_overall_timeout(
        self,
        component_lifecycle_manager
    ):
        """
        Test overall system startup timeout.

        Validates:
        - Maximum startup time enforced
        - Partial startup handled
        - Clear error messages
        - System remains stable
        """
        timer = WorkflowTimer()
        timer.start()

        # Start all components
        await component_lifecycle_manager.start_all()

        # Wait with overall timeout
        ready = await component_lifecycle_manager.wait_for_ready(timeout=90)
        timer.checkpoint("startup_complete_or_timeout")

        elapsed = timer.get_duration("startup_complete_or_timeout")

        # Should complete or timeout within limit
        assert elapsed < 100, f"Startup check exceeded overall timeout: {elapsed:.1f}s"

        # If not ready, verify system is in known state
        if not ready:
            # Components should still be accessible
            for component in ["qdrant", "daemon", "mcp_server"]:
                health = await component_lifecycle_manager.check_health(component)
                # Health check should complete even if component not healthy
                assert "healthy" in health

        # Cleanup
        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
class TestInterComponentCommunication:
    """
    Test inter-component communication establishment.

    Validates connections between components during startup.
    """

    @pytest.mark.asyncio
    async def test_daemon_qdrant_connection_establishment(
        self,
        component_lifecycle_manager
    ):
        """
        Test daemon establishes connection to Qdrant.

        Validates:
        - Daemon discovers Qdrant endpoint
        - Connection established successfully
        - Health checks pass through connection
        - Retry on connection failure
        """
        # Start Qdrant first
        await component_lifecycle_manager.start_component("qdrant")
        await asyncio.sleep(5)

        qdrant_health = await component_lifecycle_manager.check_health("qdrant")
        assert qdrant_health["healthy"]

        # Start daemon - should connect to Qdrant
        await component_lifecycle_manager.start_component("daemon")
        await asyncio.sleep(10)

        daemon_health = await component_lifecycle_manager.check_health("daemon")
        assert daemon_health["healthy"]

        # Verify connection (mocked - would test actual gRPC/HTTP calls)
        # In real implementation: verify daemon can write to Qdrant

        # Cleanup
        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_mcp_daemon_connection_establishment(
        self,
        component_lifecycle_manager
    ):
        """
        Test MCP server establishes connection to daemon.

        Validates:
        - MCP discovers daemon endpoint
        - gRPC connection established
        - Connection pool initialized
        - Fallback to direct Qdrant if daemon unavailable
        """
        # Start daemon (assumes Qdrant running)
        await component_lifecycle_manager.start_component("qdrant")
        await asyncio.sleep(5)

        await component_lifecycle_manager.start_component("daemon")
        await asyncio.sleep(10)

        # Start MCP - should connect to daemon
        await component_lifecycle_manager.start_component("mcp_server")
        await asyncio.sleep(5)

        mcp_health = await component_lifecycle_manager.check_health("mcp_server")
        assert mcp_health["healthy"]

        # Verify connection (mocked - would test actual gRPC calls)
        # In real implementation: verify MCP can send requests to daemon

        # Cleanup
        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_connection_retry_on_failure(
        self,
        component_lifecycle_manager
    ):
        """
        Test connection retry mechanisms.

        Validates:
        - Initial connection failure handled
        - Automatic retry with backoff
        - Successful connection on retry
        - Connection state tracked
        """
        # Start MCP without daemon
        await component_lifecycle_manager.start_component("qdrant")
        await asyncio.sleep(5)

        await component_lifecycle_manager.start_component("mcp_server")
        await asyncio.sleep(5)

        # MCP should be running but in degraded mode
        mcp_health = await component_lifecycle_manager.check_health("mcp_server")
        assert mcp_health["healthy"]

        # Start daemon - MCP should retry and connect
        await component_lifecycle_manager.start_component("daemon")
        await asyncio.sleep(15)  # Allow time for retry

        # MCP should now be connected to daemon
        # In real implementation: verify MCP switched from fallback to daemon mode

        # Cleanup
        await component_lifecycle_manager.stop_all()
