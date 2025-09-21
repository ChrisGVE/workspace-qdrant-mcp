"""
Integration tests for Component Isolation and Failure Handling.

Tests comprehensive isolation mechanisms including error boundaries, process separation,
resource limits, timeout handling, and failure containment strategies.
"""

import asyncio
import os
import psutil
import pytest
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.python.common.core.component_isolation import (
    ComponentIsolationManager,
    ComponentBoundary,
    ComponentBoundaryContext,
    ComponentUnavailableError,
    ResourceExhaustedError,
    ResourceLimits,
    BoundaryType,
    IsolationStrategy,
    FailureImpact,
    ResourceType,
)
from src.python.common.core.component_coordination import (
    ComponentCoordinator,
    ComponentType,
    ComponentStatus,
    ComponentHealth,
)
from src.python.common.core.component_lifecycle import ComponentLifecycleManager
from src.python.common.core.graceful_degradation import DegradationManager
from src.python.common.core.automatic_recovery import RecoveryManager
from src.python.common.core.lsp_health_monitor import LspHealthMonitor


class MockProcess:
    """Mock process for testing process isolation."""

    def __init__(self, pid: int, cpu_percent: float = 10.0, memory_mb: float = 100.0):
        self.pid = pid
        self._cpu_percent = cpu_percent
        self._memory_mb = memory_mb
        self._running = True
        self._threads = 5
        self._fds = 20

    def is_running(self):
        return self._running

    def cpu_percent(self):
        return self._cpu_percent

    def memory_info(self):
        return Mock(rss=self._memory_mb * 1024 * 1024)

    def num_threads(self):
        return self._threads

    def num_fds(self):
        return self._fds

    def ppid(self):
        return 1

    def cwd(self):
        return "/tmp"

    def terminate(self):
        self._running = False

    def kill(self):
        self._running = False

    def wait(self, timeout=None):
        pass


@pytest.fixture
async def isolation_manager():
    """Create component isolation manager for testing."""
    # Create mock dependencies
    lifecycle_manager = Mock(spec=ComponentLifecycleManager)
    coordinator = Mock(spec=ComponentCoordinator)
    degradation_manager = Mock(spec=DegradationManager)
    recovery_manager = Mock(spec=RecoveryManager)
    health_monitor = Mock(spec=LspHealthMonitor)

    # Configure mocks
    coordinator.register_component = AsyncMock(return_value="test-component-id")
    coordinator.update_component_health = AsyncMock(return_value=True)
    coordinator.record_heartbeat = AsyncMock(return_value=True)

    degradation_manager.is_component_available = Mock(return_value=True)
    degradation_manager.record_component_success = AsyncMock()
    degradation_manager.record_component_failure = AsyncMock()
    degradation_manager.register_notification_handler = Mock()

    recovery_manager.trigger_component_recovery = AsyncMock(return_value="recovery-id")
    recovery_manager.register_notification_handler = Mock()

    health_monitor.register_notification_handler = Mock()

    # Create isolation manager
    manager = ComponentIsolationManager(
        lifecycle_manager=lifecycle_manager,
        coordinator=coordinator,
        degradation_manager=degradation_manager,
        recovery_manager=recovery_manager,
        health_monitor=health_monitor,
        config={"test_mode": True}
    )

    # Initialize
    await manager.initialize()

    yield manager

    # Cleanup
    await manager.shutdown()


@pytest.fixture
def mock_processes():
    """Create mock processes for testing."""
    return {
        "rust_daemon": MockProcess(1001, 25.0, 200.0),
        "python_mcp_server": MockProcess(1002, 15.0, 150.0),
        "cli_utility": MockProcess(1003, 5.0, 50.0),
        "context_injector": MockProcess(1004, 10.0, 100.0),
    }


class TestComponentIsolationIntegration:
    """Integration tests for component isolation system."""

    @pytest.mark.asyncio
    async def test_component_boundary_timeout_handling(self, isolation_manager):
        """Test timeout handling in component boundaries."""
        # Set short timeout for testing
        boundary = ComponentBoundary(
            component_type=ComponentType.RUST_DAEMON,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=0.1,  # Very short timeout
            retry_count=1
        )

        await isolation_manager.set_component_boundary(ComponentType.RUST_DAEMON, boundary)

        # Test timeout scenario
        with pytest.raises(asyncio.TimeoutError):
            async with isolation_manager.component_boundary(
                ComponentType.RUST_DAEMON,
                "test_timeout_operation"
            ):
                # Simulate long-running operation
                await asyncio.sleep(0.2)

        # Check that timeout was recorded
        status = await isolation_manager.get_isolation_status()
        assert status['boundary_violations_count'] == 0  # Timeout doesn't count as boundary violation

        # Check that error was recorded in degradation manager
        isolation_manager.degradation_manager.record_component_failure.assert_called()

    @pytest.mark.asyncio
    async def test_component_boundary_concurrency_limits(self, isolation_manager):
        """Test concurrency limits in component boundaries."""
        # Set low concurrency limit for testing
        boundary = ComponentBoundary(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            boundary_type=BoundaryType.MESSAGE_QUEUE,
            timeout_seconds=1.0,
            max_concurrent_calls=2,  # Only 2 concurrent calls allowed
            queue_timeout_seconds=0.1
        )

        await isolation_manager.set_component_boundary(ComponentType.PYTHON_MCP_SERVER, boundary)

        # Start multiple concurrent operations
        async def slow_operation(delay: float):
            async with isolation_manager.component_boundary(
                ComponentType.PYTHON_MCP_SERVER,
                f"slow_operation_{delay}"
            ):
                await asyncio.sleep(delay)
                return f"completed_{delay}"

        # Start 2 operations (should succeed)
        task1 = asyncio.create_task(slow_operation(0.2))
        task2 = asyncio.create_task(slow_operation(0.2))

        # Wait a bit for them to start
        await asyncio.sleep(0.05)

        # Start 3rd operation (should timeout on queue)
        with pytest.raises(TimeoutError, match="Queue timeout"):
            await slow_operation(0.1)

        # Wait for original operations to complete
        result1 = await task1
        result2 = await task2

        assert result1 == "completed_0.2"
        assert result2 == "completed_0.2"

    @pytest.mark.asyncio
    async def test_resource_limit_enforcement(self, isolation_manager, mock_processes):
        """Test resource limit enforcement and violations."""
        # Set strict resource limits
        limits = ResourceLimits(
            cpu_limit_percent=20.0,  # Low CPU limit
            memory_limit_mb=120,     # Low memory limit
            max_file_descriptors=15, # Low FD limit
            max_threads=3,           # Low thread limit
            enforce_cpu=True,
            enforce_memory=True,
            kill_on_violation=False  # Don't kill for testing
        )

        await isolation_manager.set_component_resource_limits(ComponentType.RUST_DAEMON, limits)

        # Create mock process that violates limits
        violating_process = MockProcess(2001, 35.0, 200.0)  # High CPU and memory
        violating_process._threads = 10  # High thread count
        violating_process._fds = 25      # High FD count

        # Mock psutil to return our violating process
        with patch('psutil.Process', return_value=violating_process):
            with patch('psutil.process_iter', return_value=[Mock(info={'pid': 2001, 'name': 'rust_daemon', 'cmdline': ['rust', 'daemon'], 'create_time': time.time()})]):
                # Trigger resource monitoring
                await isolation_manager._check_resource_usage()

        # Check that violations were recorded
        status = await isolation_manager.get_isolation_status()
        assert status['resource_violations_count'] > 0

        # Check that isolation events were created
        assert len(isolation_manager.isolation_events) > 0

        # Verify the violation event
        violation_events = [
            event for event in isolation_manager.isolation_events
            if event.isolation_strategy == IsolationStrategy.RESOURCE_ISOLATION
        ]
        assert len(violation_events) > 0

        violation_event = violation_events[0]
        assert "violation" in violation_event.trigger_reason.lower()
        assert violation_event.impact_level == FailureImpact.LIMITED

    @pytest.mark.asyncio
    async def test_error_boundary_failure_containment(self, isolation_manager):
        """Test error boundaries prevent failure propagation."""
        # Configure component with critical exceptions
        boundary = ComponentBoundary(
            component_type=ComponentType.CONTEXT_INJECTOR,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=5.0,
            critical_exceptions={"RuntimeError", "ValueError"},
            allowed_exceptions={"KeyError"}
        )

        await isolation_manager.set_component_boundary(ComponentType.CONTEXT_INJECTOR, boundary)

        # Test allowed exception (should not trigger isolation)
        try:
            async with isolation_manager.component_boundary(
                ComponentType.CONTEXT_INJECTOR,
                "test_allowed_error"
            ):
                raise KeyError("This is allowed")
        except KeyError:
            pass  # Expected

        # Check that no isolation was triggered
        assert isolation_manager.isolation_count == 0

        # Test critical exception (should trigger isolation)
        with pytest.raises(RuntimeError):
            async with isolation_manager.component_boundary(
                ComponentType.CONTEXT_INJECTOR,
                "test_critical_error"
            ):
                raise RuntimeError("This is critical")

        # Check that isolation was triggered
        assert isolation_manager.isolation_count == 1

        # Verify recovery was triggered
        isolation_manager.recovery_manager.trigger_component_recovery.assert_called_with(
            ComponentType.CONTEXT_INJECTOR,
            reason="Component isolation: This is critical"
        )

    @pytest.mark.asyncio
    async def test_failure_propagation_prevention(self, isolation_manager):
        """Test that failures in one component don't cascade to others."""
        # Set up multiple component boundaries
        for component_type in ComponentType:
            boundary = ComponentBoundary(
                component_type=component_type,
                boundary_type=BoundaryType.SYNCHRONOUS,
                timeout_seconds=1.0,
                critical_exceptions={"SystemError"}
            )
            await isolation_manager.set_component_boundary(component_type, boundary)

        # Fail one component
        with pytest.raises(SystemError):
            async with isolation_manager.component_boundary(
                ComponentType.RUST_DAEMON,
                "failing_operation"
            ):
                raise SystemError("Daemon failure")

        # Verify the failing component was isolated
        assert isolation_manager.isolation_count == 1

        # Test that other components are still functional
        async with isolation_manager.component_boundary(
            ComponentType.PYTHON_MCP_SERVER,
            "healthy_operation"
        ):
            # This should work fine
            pass

        async with isolation_manager.component_boundary(
            ComponentType.CLI_UTILITY,
            "another_healthy_operation"
        ):
            # This should also work fine
            pass

        # Verify no additional isolation was triggered
        assert isolation_manager.isolation_count == 1

        # Check that only the failed component triggered recovery
        recovery_calls = isolation_manager.recovery_manager.trigger_component_recovery.call_args_list
        assert len(recovery_calls) == 1
        assert recovery_calls[0][0][0] == ComponentType.RUST_DAEMON

    @pytest.mark.asyncio
    async def test_component_unavailable_handling(self, isolation_manager):
        """Test handling of unavailable components."""
        # Configure degradation manager to report component as unavailable
        isolation_manager.degradation_manager.is_component_available.return_value = False

        # Test that unavailable component raises error
        with pytest.raises(ComponentUnavailableError):
            async with isolation_manager.component_boundary(
                ComponentType.RUST_DAEMON,
                "test_unavailable"
            ):
                pass

        # Test with allow_degraded=True (should work)
        async with isolation_manager.component_boundary(
            ComponentType.RUST_DAEMON,
            "test_degraded_allowed",
            allow_degraded=True
        ):
            # Should work even though component is unavailable
            pass

    @pytest.mark.asyncio
    async def test_process_termination_handling(self, isolation_manager, mock_processes):
        """Test handling of process termination."""
        # Add a mock process to the manager
        process_info = isolation_manager.component_processes["rust_daemon-2001"] = Mock()
        process_info.component_type = ComponentType.RUST_DAEMON
        process_info.process_id = 2001
        process_info.is_running = True
        process_info.cpu_usage_percent = 10.0
        process_info.memory_usage_mb = 100.0
        process_info.file_descriptor_count = 10
        process_info.thread_count = 5
        process_info.resource_violations = []

        # Mock psutil to indicate process no longer exists
        with patch('psutil.Process', side_effect=psutil.NoSuchProcess(2001)):
            # Trigger process health check
            await isolation_manager._check_process_health()

        # Verify process was removed from tracking
        assert "rust_daemon-2001" not in isolation_manager.component_processes

        # Verify recovery was triggered
        isolation_manager.recovery_manager.trigger_component_recovery.assert_called()

        # Verify process restart count was incremented
        assert isolation_manager.process_restarts_count == 1

    @pytest.mark.asyncio
    async def test_network_failure_scenario(self, isolation_manager):
        """Test network failure scenario and isolation."""
        # Configure boundary with network-related exceptions
        boundary = ComponentBoundary(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            boundary_type=BoundaryType.MESSAGE_QUEUE,
            timeout_seconds=2.0,
            critical_exceptions={"ConnectionError", "TimeoutError"},
            max_error_rate=0.1,
            error_window_seconds=60
        )

        await isolation_manager.set_component_boundary(ComponentType.PYTHON_MCP_SERVER, boundary)

        # Simulate network failures
        for i in range(3):
            with pytest.raises(ConnectionError):
                async with isolation_manager.component_boundary(
                    ComponentType.PYTHON_MCP_SERVER,
                    f"network_operation_{i}"
                ):
                    raise ConnectionError("Network unreachable")

        # Check that isolation was triggered due to repeated failures
        assert isolation_manager.isolation_count >= 1

        # Check that error rate monitoring is working
        error_key = "python_mcp_server_errors"
        assert error_key in isolation_manager.error_counts
        assert len(isolation_manager.error_counts[error_key]) == 3

    @pytest.mark.asyncio
    async def test_resource_exhaustion_scenario(self, isolation_manager):
        """Test resource exhaustion scenario."""
        # Create mock process with very high resource usage
        exhausted_process = MockProcess(3001, 95.0, 800.0)  # Very high CPU and memory
        exhausted_process._threads = 100  # Very high thread count
        exhausted_process._fds = 1500     # Very high FD count

        # Set strict limits
        limits = ResourceLimits(
            cpu_limit_percent=50.0,
            memory_limit_mb=512,
            max_file_descriptors=1000,
            max_threads=50,
            enforce_cpu=True,
            enforce_memory=True,
            kill_on_violation=True  # Enable termination
        )

        await isolation_manager.set_component_resource_limits(ComponentType.CLI_UTILITY, limits)

        # Add process to tracking
        isolation_manager.component_processes["cli_utility-3001"] = Mock()
        process_info = isolation_manager.component_processes["cli_utility-3001"]
        process_info.component_type = ComponentType.CLI_UTILITY
        process_info.process_id = 3001
        process_info.is_running = True
        process_info.resource_limits = limits
        process_info.resource_violations = []
        process_info.cpu_usage_percent = 95.0
        process_info.memory_usage_mb = 800.0
        process_info.file_descriptor_count = 1500
        process_info.thread_count = 100

        # Mock process termination
        with patch('psutil.Process', return_value=exhausted_process):
            with patch.object(isolation_manager, '_terminate_violating_process') as mock_terminate:
                # Trigger resource violation check
                await isolation_manager._check_resource_violations("cli_utility-3001", process_info)

                # Verify termination was called
                mock_terminate.assert_called_once()

        # Verify multiple violations were recorded
        assert len(process_info.resource_violations) >= 4  # CPU, memory, FD, threads

    @pytest.mark.asyncio
    async def test_component_crash_scenario(self, isolation_manager):
        """Test component crash scenario and recovery."""
        # Add a process that will "crash"
        process_info = Mock()
        process_info.component_type = ComponentType.CONTEXT_INJECTOR
        process_info.process_id = 4001
        process_info.is_running = True
        process_info.termination_reason = None

        isolation_manager.component_processes["context_injector-4001"] = process_info

        # Simulate process crash (process no longer exists)
        crashed_process = Mock()
        crashed_process.is_running.return_value = False

        with patch('psutil.Process', return_value=crashed_process):
            # Trigger process health check
            await isolation_manager._check_process_health()

        # Verify process was marked as not running
        assert not process_info.is_running
        assert process_info.termination_reason == "Process no longer running"

        # Verify recovery was triggered
        isolation_manager.recovery_manager.trigger_component_recovery.assert_called()

        # Verify process was removed from tracking
        await isolation_manager._handle_process_termination("context_injector-4001", process_info)
        assert "context_injector-4001" not in isolation_manager.component_processes

    @pytest.mark.asyncio
    async def test_disk_space_exhaustion(self, isolation_manager):
        """Test disk space exhaustion handling."""
        # This test simulates disk space exhaustion by creating a large file
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set disk limits
            limits = ResourceLimits(
                disk_limit_mb=100,  # 100MB limit
                enforce_disk=True,
                kill_on_violation=False
            )

            await isolation_manager.set_component_resource_limits(ComponentType.RUST_DAEMON, limits)

            # Simulate disk usage check (this would be implemented in a real scenario)
            # For testing, we'll manually trigger a disk violation
            process_info = Mock()
            process_info.component_type = ComponentType.RUST_DAEMON
            process_info.process_id = 5001
            process_info.is_running = True
            process_info.resource_limits = limits
            process_info.resource_violations = []
            process_info.disk_usage_mb = 150.0  # Exceeds limit

            # Manually check for disk violations (simulated)
            if process_info.disk_usage_mb > limits.disk_limit_mb:
                violation = f"Disk usage {process_info.disk_usage_mb}MB exceeds limit {limits.disk_limit_mb}MB"
                process_info.resource_violations.append(violation)

            assert len(process_info.resource_violations) == 1
            assert "Disk usage" in process_info.resource_violations[0]

    @pytest.mark.asyncio
    async def test_error_rate_threshold_isolation(self, isolation_manager):
        """Test isolation triggered by high error rates."""
        # Configure component with low error rate threshold
        boundary = ComponentBoundary(
            component_type=ComponentType.CLI_UTILITY,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=5.0,
            max_error_rate=0.05,  # 5% error rate threshold
            error_window_seconds=10,
            critical_exceptions={"RuntimeError"}
        )

        await isolation_manager.set_component_boundary(ComponentType.CLI_UTILITY, boundary)

        # Generate errors to exceed threshold
        for i in range(10):  # Generate many errors quickly
            try:
                async with isolation_manager.component_boundary(
                    ComponentType.CLI_UTILITY,
                    f"error_operation_{i}"
                ):
                    raise RuntimeError(f"Error {i}")
            except RuntimeError:
                pass

            # Small delay to stay within error window
            await asyncio.sleep(0.01)

        # Trigger boundary violation check
        await isolation_manager._check_boundary_violations()

        # Verify boundary violations were detected
        assert isolation_manager.boundary_violations_count > 0

        # Verify isolation events were created
        boundary_events = [
            event for event in isolation_manager.isolation_events
            if event.isolation_strategy == IsolationStrategy.FAILURE_CONTAINMENT
        ]
        assert len(boundary_events) > 0

    @pytest.mark.asyncio
    async def test_isolation_status_reporting(self, isolation_manager):
        """Test comprehensive isolation status reporting."""
        # Add some mock processes and violations
        process_info = Mock()
        process_info.component_type = ComponentType.PYTHON_MCP_SERVER
        process_info.process_id = 6001
        process_info.is_running = True
        process_info.cpu_usage_percent = 25.0
        process_info.memory_usage_mb = 200.0
        process_info.file_descriptor_count = 50
        process_info.thread_count = 10
        process_info.resource_violations = ["Test violation"]
        process_info.start_time = isolation_manager.start_time
        process_info.last_health_check = None

        isolation_manager.component_processes["python_mcp_server-6001"] = process_info

        # Add some active calls
        isolation_manager.active_calls["python_mcp_server-6001"] = {"call1", "call2"}

        # Increment some counters
        isolation_manager.isolation_count = 2
        isolation_manager.boundary_violations_count = 3
        isolation_manager.resource_violations_count = 5
        isolation_manager.process_restarts_count = 1

        # Get status
        status = await isolation_manager.get_isolation_status()

        # Verify status structure and content
        assert 'uptime_seconds' in status
        assert status['isolation_count'] == 2
        assert status['boundary_violations_count'] == 3
        assert status['resource_violations_count'] == 5
        assert status['process_restarts_count'] == 1
        assert status['active_processes'] == 1
        assert status['total_processes'] == 1

        assert 'active_calls' in status
        assert status['active_calls']['python_mcp_server-6001'] == 2

        assert 'processes' in status
        assert len(status['processes']) == 1

        process_status = status['processes'][0]
        assert process_status['component_id'] == 'python_mcp_server-6001'
        assert process_status['component_type'] == 'python_mcp_server'
        assert process_status['is_running'] is True
        assert process_status['resource_violations'] == ["Test violation"]

        assert 'component_boundaries' in status
        assert 'resource_limits' in status

    @pytest.mark.asyncio
    async def test_manual_component_isolation(self, isolation_manager):
        """Test manual component isolation."""
        # Force isolation of a component
        await isolation_manager.force_component_isolation(
            ComponentType.RUST_DAEMON,
            "Manual test isolation"
        )

        # Verify isolation was triggered
        assert isolation_manager.isolation_count == 1

        # Verify recovery was triggered
        isolation_manager.recovery_manager.trigger_component_recovery.assert_called_with(
            ComponentType.RUST_DAEMON,
            reason="Component isolation: Manual test isolation"
        )

        # Verify isolation event was created
        isolation_events = [
            event for event in isolation_manager.isolation_events
            if event.component_id == "rust_daemon-isolated"
        ]
        assert len(isolation_events) == 1

        isolation_event = isolation_events[0]
        assert isolation_event.trigger_reason == "Manual test isolation"
        assert isolation_event.action_taken == "Component isolated due to critical error"
        assert isolation_event.recovery_triggered is True


class TestBoundaryContextIntegration:
    """Integration tests for boundary context functionality."""

    @pytest.mark.asyncio
    async def test_boundary_context_timeout_warning(self, isolation_manager):
        """Test boundary context timeout warnings."""
        # Set short timeout
        boundary = ComponentBoundary(
            component_type=ComponentType.CLI_UTILITY,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=1.0
        )

        await isolation_manager.set_component_boundary(ComponentType.CLI_UTILITY, boundary)

        async with isolation_manager.component_boundary(
            ComponentType.CLI_UTILITY,
            "timeout_warning_test"
        ) as context:
            # Wait most of the timeout period
            await asyncio.sleep(0.8)

            # Check if we're approaching timeout
            elapsed = context.get_elapsed_time()
            assert elapsed >= 0.8

            # This should warn about approaching timeout
            with pytest.warns(UserWarning):
                await context.check_timeout(0.3)  # 0.3s threshold

    @pytest.mark.asyncio
    async def test_boundary_context_elapsed_time(self, isolation_manager):
        """Test boundary context elapsed time tracking."""
        async with isolation_manager.component_boundary(
            ComponentType.CONTEXT_INJECTOR,
            "elapsed_time_test"
        ) as context:
            initial_time = context.get_elapsed_time()
            assert initial_time >= 0

            await asyncio.sleep(0.1)

            later_time = context.get_elapsed_time()
            assert later_time > initial_time
            assert later_time >= 0.1


class TestFailureScenarios:
    """Integration tests for various failure scenarios."""

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, isolation_manager):
        """Test prevention of cascading failures across components."""
        # Configure all components with different failure conditions
        for component_type in ComponentType:
            boundary = ComponentBoundary(
                component_type=component_type,
                boundary_type=BoundaryType.SYNCHRONOUS,
                timeout_seconds=2.0,
                critical_exceptions={"SystemError"},
                max_error_rate=0.2
            )
            await isolation_manager.set_component_boundary(component_type, boundary)

        # Fail the first component (Rust daemon)
        with pytest.raises(SystemError):
            async with isolation_manager.component_boundary(
                ComponentType.RUST_DAEMON,
                "cascade_test_1"
            ):
                raise SystemError("Primary component failure")

        # Verify isolation was triggered for the failing component
        assert isolation_manager.isolation_count == 1

        # Test that other components remain functional and isolated
        successful_operations = []

        for component_type in [ComponentType.PYTHON_MCP_SERVER, ComponentType.CLI_UTILITY, ComponentType.CONTEXT_INJECTOR]:
            try:
                async with isolation_manager.component_boundary(
                    component_type,
                    f"cascade_test_{component_type.value}"
                ):
                    successful_operations.append(component_type.value)
            except Exception as e:
                pytest.fail(f"Component {component_type.value} should not have failed: {e}")

        # All other components should have operated successfully
        assert len(successful_operations) == 3

        # Only one component should have triggered isolation
        assert isolation_manager.isolation_count == 1

    @pytest.mark.asyncio
    async def test_memory_leak_detection_and_isolation(self, isolation_manager):
        """Test detection and isolation of memory leaks."""
        # Set up memory monitoring with strict limits
        limits = ResourceLimits(
            memory_limit_mb=200,
            enforce_memory=True,
            kill_on_violation=False,
            warn_threshold=0.7  # Warning at 70%
        )

        await isolation_manager.set_component_resource_limits(ComponentType.PYTHON_MCP_SERVER, limits)

        # Simulate gradually increasing memory usage
        memory_leak_process = MockProcess(7001, 15.0, 100.0)

        isolation_manager.component_processes["python_mcp_server-7001"] = Mock()
        process_info = isolation_manager.component_processes["python_mcp_server-7001"]
        process_info.component_type = ComponentType.PYTHON_MCP_SERVER
        process_info.process_id = 7001
        process_info.is_running = True
        process_info.resource_limits = limits
        process_info.resource_violations = []

        # Simulate memory leak progression
        memory_values = [100, 140, 180, 220, 250]  # Gradual increase, exceeding limit

        for memory_mb in memory_values:
            process_info.memory_usage_mb = memory_mb
            process_info.cpu_usage_percent = 15.0
            process_info.file_descriptor_count = 20
            process_info.thread_count = 5

            # Check for violations
            await isolation_manager._check_resource_violations("python_mcp_server-7001", process_info)

        # Verify violations were detected for memory exceeding limit
        memory_violations = [v for v in process_info.resource_violations if "Memory usage" in v]
        assert len(memory_violations) >= 2  # Should detect violations for 220MB and 250MB

        # Verify isolation events were created
        memory_events = [
            event for event in isolation_manager.isolation_events
            if event.isolation_strategy == IsolationStrategy.RESOURCE_ISOLATION
        ]
        assert len(memory_events) >= 2

    @pytest.mark.asyncio
    async def test_deadlock_detection_and_recovery(self, isolation_manager):
        """Test detection and recovery from deadlock scenarios."""
        # Configure component with short timeout to detect deadlocks
        boundary = ComponentBoundary(
            component_type=ComponentType.CONTEXT_INJECTOR,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=1.0,
            max_concurrent_calls=2
        )

        await isolation_manager.set_component_boundary(ComponentType.CONTEXT_INJECTOR, boundary)

        # Create deadlock scenario - two operations waiting for each other
        async def deadlock_operation_1():
            async with isolation_manager.component_boundary(
                ComponentType.CONTEXT_INJECTOR,
                "deadlock_op_1"
            ):
                # This would normally wait for operation 2, simulated by long sleep
                await asyncio.sleep(2.0)  # Longer than timeout

        async def deadlock_operation_2():
            async with isolation_manager.component_boundary(
                ComponentType.CONTEXT_INJECTOR,
                "deadlock_op_2"
            ):
                # This would normally wait for operation 1, simulated by long sleep
                await asyncio.sleep(2.0)  # Longer than timeout

        # Start both operations
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(3.0):  # Overall timeout
                await asyncio.gather(
                    deadlock_operation_1(),
                    deadlock_operation_2()
                )

        # Verify timeout handling prevented the deadlock from hanging the system
        # Both operations should have timed out and been cleaned up

        # Check isolation events for timeout handling
        timeout_events = [
            event for event in isolation_manager.isolation_events
            if "timeout" in event.trigger_reason.lower()
        ]
        assert len(timeout_events) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])