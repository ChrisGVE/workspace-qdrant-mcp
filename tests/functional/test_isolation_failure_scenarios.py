"""
Functional tests for Component Isolation Failure Scenarios.

Tests comprehensive failure scenarios including network issues, resource exhaustion,
component crashes, and validation of isolation effectiveness.
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
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from workspace_qdrant_mcp.core.component_isolation import (
    ComponentIsolationManager,
    ComponentBoundary,
    ResourceLimits,
    BoundaryType,
    IsolationStrategy,
    FailureImpact,
)
from workspace_qdrant_mcp.core.component_coordination import (
    ComponentCoordinator,
    ComponentType,
    ComponentStatus,
    ComponentHealth,
)
from workspace_qdrant_mcp.core.component_lifecycle import ComponentLifecycleManager
from workspace_qdrant_mcp.core.graceful_degradation import DegradationManager
from workspace_qdrant_mcp.core.automatic_recovery import RecoveryManager
from workspace_qdrant_mcp.core.lsp_health_monitor import LspHealthMonitor, UserNotification, NotificationLevel


class NetworkFailureSimulator:
    """Simulates various network failure conditions."""

    @staticmethod
    async def simulate_network_timeout():
        """Simulate network timeout."""
        await asyncio.sleep(0.1)
        raise asyncio.TimeoutError("Network timeout")

    @staticmethod
    async def simulate_connection_refused():
        """Simulate connection refused."""
        await asyncio.sleep(0.05)
        raise ConnectionRefusedError("Connection refused")

    @staticmethod
    async def simulate_dns_failure():
        """Simulate DNS resolution failure."""
        await asyncio.sleep(0.02)
        raise OSError("Name resolution failed")

    @staticmethod
    async def simulate_intermittent_network():
        """Simulate intermittent network issues."""
        import random
        if random.random() < 0.7:  # 70% failure rate
            await asyncio.sleep(0.05)
            raise ConnectionError("Intermittent network failure")
        await asyncio.sleep(0.1)
        return "success"


class ResourceExhaustionSimulator:
    """Simulates various resource exhaustion scenarios."""

    def __init__(self):
        self.cpu_load = 0.0
        self.memory_usage = 0.0
        self.fd_count = 0
        self.thread_count = 0

    def simulate_cpu_spike(self, target_percent: float):
        """Simulate CPU usage spike."""
        self.cpu_load = target_percent

    def simulate_memory_leak(self, increase_mb: float):
        """Simulate memory leak."""
        self.memory_usage += increase_mb

    def simulate_fd_leak(self, increase_count: int):
        """Simulate file descriptor leak."""
        self.fd_count += increase_count

    def simulate_thread_explosion(self, increase_count: int):
        """Simulate thread count explosion."""
        self.thread_count += increase_count

    def get_usage_stats(self):
        """Get current usage statistics."""
        return {
            'cpu_percent': self.cpu_load,
            'memory_mb': self.memory_usage,
            'fd_count': self.fd_count,
            'thread_count': self.thread_count
        }


@pytest.fixture
async def full_isolation_system():
    """Create complete isolation system with all dependencies."""
    # Create real-like mocks for all dependencies
    lifecycle_manager = AsyncMock(spec=ComponentLifecycleManager)
    coordinator = AsyncMock(spec=ComponentCoordinator)
    degradation_manager = Mock(spec=DegradationManager)
    recovery_manager = AsyncMock(spec=RecoveryManager)
    health_monitor = Mock(spec=LspHealthMonitor)

    # Configure coordinator
    coordinator.register_component.return_value = "test-component-id"
    coordinator.update_component_health.return_value = True
    coordinator.record_heartbeat.return_value = True
    coordinator.get_component_status.return_value = {
        "components": {
            "rust_daemon": {"state": "operational", "health": "healthy"},
            "python_mcp_server": {"state": "operational", "health": "healthy"},
            "cli_utility": {"state": "operational", "health": "healthy"},
            "context_injector": {"state": "operational", "health": "healthy"},
        }
    }

    # Configure degradation manager
    degradation_manager.is_component_available.return_value = True
    degradation_manager.record_component_success = AsyncMock()
    degradation_manager.record_component_failure = AsyncMock()
    degradation_manager.register_notification_handler.return_value = None

    # Configure recovery manager
    recovery_manager.trigger_component_recovery.return_value = "recovery-attempt-id"
    recovery_manager.register_notification_handler.return_value = None

    # Configure health monitor
    health_monitor.register_notification_handler.return_value = None

    # Create isolation manager
    isolation_manager = ComponentIsolationManager(
        lifecycle_manager=lifecycle_manager,
        coordinator=coordinator,
        degradation_manager=degradation_manager,
        recovery_manager=recovery_manager,
        health_monitor=health_monitor,
        config={
            "test_mode": True,
            "monitoring_interval": 0.1,  # Fast monitoring for tests
            "resource_check_interval": 0.1
        }
    )

    await isolation_manager.initialize()

    yield {
        'isolation_manager': isolation_manager,
        'lifecycle_manager': lifecycle_manager,
        'coordinator': coordinator,
        'degradation_manager': degradation_manager,
        'recovery_manager': recovery_manager,
        'health_monitor': health_monitor
    }

    await isolation_manager.shutdown()


@pytest.fixture
def network_simulator():
    """Create network failure simulator."""
    return NetworkFailureSimulator()


@pytest.fixture
def resource_simulator():
    """Create resource exhaustion simulator."""
    return ResourceExhaustionSimulator()


class TestNetworkFailureScenarios:
    """Test various network failure scenarios and isolation effectiveness."""

    @pytest.mark.asyncio
    async def test_network_timeout_isolation(self, full_isolation_system, network_simulator):
        """Test isolation during network timeout scenarios."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Configure component for network operations
        boundary = ComponentBoundary(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=0.5,  # Short timeout for testing
            retry_count=2,
            critical_exceptions={"TimeoutError", "asyncio.TimeoutError"}
        )

        await isolation_manager.set_component_boundary(ComponentType.PYTHON_MCP_SERVER, boundary)

        # Simulate network timeout scenario
        timeout_count = 0
        for i in range(3):  # Multiple timeout attempts
            with pytest.raises((asyncio.TimeoutError, TimeoutError)):
                async with isolation_manager.component_boundary(
                    ComponentType.PYTHON_MCP_SERVER,
                    f"network_operation_{i}"
                ):
                    await network_simulator.simulate_network_timeout()
                timeout_count += 1

        # Verify timeouts were handled properly
        assert timeout_count == 3

        # Check that component was eventually isolated due to repeated failures
        assert isolation_manager.isolation_count >= 1

        # Verify recovery was triggered
        recovery_manager = full_isolation_system['recovery_manager']
        recovery_manager.trigger_component_recovery.assert_called()

    @pytest.mark.asyncio
    async def test_connection_refused_handling(self, full_isolation_system, network_simulator):
        """Test handling of connection refused errors."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Configure component with connection error handling
        boundary = ComponentBoundary(
            component_type=ComponentType.RUST_DAEMON,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=1.0,
            critical_exceptions={"ConnectionRefusedError"},
            max_error_rate=0.3
        )

        await isolation_manager.set_component_boundary(ComponentType.RUST_DAEMON, boundary)

        # Simulate connection refused scenarios
        for i in range(5):
            with pytest.raises(ConnectionRefusedError):
                async with isolation_manager.component_boundary(
                    ComponentType.RUST_DAEMON,
                    f"connection_operation_{i}"
                ):
                    await network_simulator.simulate_connection_refused()

        # Verify isolation was triggered
        assert isolation_manager.isolation_count >= 1

        # Check that error rate threshold was exceeded
        error_key = "rust_daemon_errors"
        assert error_key in isolation_manager.error_counts
        assert len(isolation_manager.error_counts[error_key]) == 5

    @pytest.mark.asyncio
    async def test_intermittent_network_resilience(self, full_isolation_system, network_simulator):
        """Test resilience to intermittent network issues."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Configure component with retry logic
        boundary = ComponentBoundary(
            component_type=ComponentType.CONTEXT_INJECTOR,
            boundary_type=BoundaryType.ASYNCHRONOUS,
            timeout_seconds=2.0,
            retry_count=3,
            retry_delay_seconds=0.1,
            max_error_rate=0.8  # High tolerance for intermittent issues
        )

        await isolation_manager.set_component_boundary(ComponentType.CONTEXT_INJECTOR, boundary)

        # Test intermittent network operations
        success_count = 0
        failure_count = 0

        for i in range(10):
            try:
                async with isolation_manager.component_boundary(
                    ComponentType.CONTEXT_INJECTOR,
                    f"intermittent_operation_{i}"
                ):
                    result = await network_simulator.simulate_intermittent_network()
                    if result == "success":
                        success_count += 1
            except ConnectionError:
                failure_count += 1

        # Should have some successes and failures
        assert success_count > 0
        assert failure_count > 0

        # Should not trigger isolation due to high tolerance
        assert isolation_manager.isolation_count == 0

    @pytest.mark.asyncio
    async def test_dns_failure_isolation(self, full_isolation_system, network_simulator):
        """Test isolation during DNS resolution failures."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Configure component for DNS-dependent operations
        boundary = ComponentBoundary(
            component_type=ComponentType.CLI_UTILITY,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=1.0,
            critical_exceptions={"OSError"},
            max_error_rate=0.2
        )

        await isolation_manager.set_component_boundary(ComponentType.CLI_UTILITY, boundary)

        # Simulate DNS failures
        for i in range(3):
            with pytest.raises(OSError):
                async with isolation_manager.component_boundary(
                    ComponentType.CLI_UTILITY,
                    f"dns_operation_{i}"
                ):
                    await network_simulator.simulate_dns_failure()

        # Verify isolation was triggered
        assert isolation_manager.isolation_count >= 1

        # Verify DNS failures were properly recorded
        dns_events = [
            event for event in isolation_manager.isolation_events
            if "dns" in event.trigger_reason.lower() or "name resolution" in event.trigger_reason.lower()
        ]
        # DNS failures should be recorded as isolation events
        assert len(isolation_manager.isolation_events) > 0


class TestResourceExhaustionScenarios:
    """Test various resource exhaustion scenarios."""

    @pytest.mark.asyncio
    async def test_cpu_spike_handling(self, full_isolation_system, resource_simulator):
        """Test handling of CPU usage spikes."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Set strict CPU limits
        limits = ResourceLimits(
            cpu_limit_percent=30.0,
            enforce_cpu=True,
            kill_on_violation=False
        )

        await isolation_manager.set_component_resource_limits(ComponentType.RUST_DAEMON, limits)

        # Simulate CPU spike
        resource_simulator.simulate_cpu_spike(80.0)  # Spike to 80%

        # Create mock process with high CPU usage
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 80.0
        mock_process.memory_info.return_value = Mock(rss=100 * 1024 * 1024)  # 100MB
        mock_process.num_threads.return_value = 5
        mock_process.num_fds.return_value = 20
        mock_process.is_running.return_value = True

        # Add process to tracking
        process_info = Mock()
        process_info.component_type = ComponentType.RUST_DAEMON
        process_info.process_id = 8001
        process_info.is_running = True
        process_info.resource_limits = limits
        process_info.resource_violations = []
        process_info.cpu_usage_percent = 80.0
        process_info.memory_usage_mb = 100.0
        process_info.file_descriptor_count = 20
        process_info.thread_count = 5

        isolation_manager.component_processes["rust_daemon-8001"] = process_info

        # Trigger resource monitoring
        await isolation_manager._check_resource_violations("rust_daemon-8001", process_info)

        # Verify CPU violation was detected
        cpu_violations = [v for v in process_info.resource_violations if "CPU usage" in v]
        assert len(cpu_violations) >= 1

        # Verify isolation event was created
        cpu_events = [
            event for event in isolation_manager.isolation_events
            if event.isolation_strategy == IsolationStrategy.RESOURCE_ISOLATION
        ]
        assert len(cpu_events) >= 1

    @pytest.mark.asyncio
    async def test_memory_leak_progression(self, full_isolation_system, resource_simulator):
        """Test detection of progressive memory leaks."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Set memory limits with progression monitoring
        limits = ResourceLimits(
            memory_limit_mb=300,
            enforce_memory=True,
            warn_threshold=0.7,  # Warning at 70% (210MB)
            kill_on_violation=False
        )

        await isolation_manager.set_component_resource_limits(ComponentType.PYTHON_MCP_SERVER, limits)

        # Simulate progressive memory leak
        process_info = Mock()
        process_info.component_type = ComponentType.PYTHON_MCP_SERVER
        process_info.process_id = 8002
        process_info.is_running = True
        process_info.resource_limits = limits
        process_info.resource_violations = []

        isolation_manager.component_processes["python_mcp_server-8002"] = process_info

        # Simulate memory growth over time
        memory_progression = [100, 150, 200, 250, 320, 400]  # MB
        violation_count = 0

        for memory_mb in memory_progression:
            resource_simulator.memory_usage = memory_mb
            process_info.memory_usage_mb = memory_mb
            process_info.cpu_usage_percent = 20.0
            process_info.file_descriptor_count = 30
            process_info.thread_count = 8

            # Check for violations
            await isolation_manager._check_resource_violations("python_mcp_server-8002", process_info)

            if memory_mb > limits.memory_limit_mb:
                violation_count += 1

        # Verify progressive violations were detected
        memory_violations = [v for v in process_info.resource_violations if "Memory usage" in v]
        assert len(memory_violations) >= 2  # Should detect violations for 320MB and 400MB

        # Verify resource usage history was recorded
        assert "python_mcp_server-8002" in isolation_manager.resource_usage_history

    @pytest.mark.asyncio
    async def test_file_descriptor_leak(self, full_isolation_system, resource_simulator):
        """Test detection of file descriptor leaks."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Set strict FD limits
        limits = ResourceLimits(
            max_file_descriptors=50,
            enforce_memory=True,  # Use memory enforcement flag
            kill_on_violation=True
        )

        await isolation_manager.set_component_resource_limits(ComponentType.CLI_UTILITY, limits)

        # Simulate file descriptor leak
        process_info = Mock()
        process_info.component_type = ComponentType.CLI_UTILITY
        process_info.process_id = 8003
        process_info.is_running = True
        process_info.resource_limits = limits
        process_info.resource_violations = []

        isolation_manager.component_processes["cli_utility-8003"] = process_info

        # Simulate progressive FD leak
        fd_progression = [20, 40, 60, 80, 100]
        for fd_count in fd_progression:
            resource_simulator.simulate_fd_leak(5)
            process_info.file_descriptor_count = fd_count
            process_info.cpu_usage_percent = 5.0
            process_info.memory_usage_mb = 50.0
            process_info.thread_count = 3

            await isolation_manager._check_resource_violations("cli_utility-8003", process_info)

        # Verify FD violations were detected
        fd_violations = [v for v in process_info.resource_violations if "File descriptors" in v]
        assert len(fd_violations) >= 3  # Should detect violations for 60, 80, 100

        # Verify termination would be triggered due to kill_on_violation=True
        # (In actual implementation, process would be terminated)

    @pytest.mark.asyncio
    async def test_thread_explosion(self, full_isolation_system, resource_simulator):
        """Test detection of thread explosion scenarios."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Set thread limits
        limits = ResourceLimits(
            max_threads=20,
            enforce_memory=True,
            kill_on_violation=False
        )

        await isolation_manager.set_component_resource_limits(ComponentType.CONTEXT_INJECTOR, limits)

        # Simulate thread explosion
        process_info = Mock()
        process_info.component_type = ComponentType.CONTEXT_INJECTOR
        process_info.process_id = 8004
        process_info.is_running = True
        process_info.resource_limits = limits
        process_info.resource_violations = []

        isolation_manager.component_processes["context_injector-8004"] = process_info

        # Simulate rapid thread creation
        thread_progression = [5, 15, 25, 50, 100]
        for thread_count in thread_progression:
            resource_simulator.simulate_thread_explosion(10)
            process_info.thread_count = thread_count
            process_info.cpu_usage_percent = 15.0
            process_info.memory_usage_mb = 120.0
            process_info.file_descriptor_count = 25

            await isolation_manager._check_resource_violations("context_injector-8004", process_info)

        # Verify thread violations were detected
        thread_violations = [v for v in process_info.resource_violations if "Thread count" in v]
        assert len(thread_violations) >= 3  # Should detect violations for 25, 50, 100


class TestComponentCrashScenarios:
    """Test various component crash scenarios."""

    @pytest.mark.asyncio
    async def test_sudden_process_termination(self, full_isolation_system):
        """Test handling of sudden process termination."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Add a process that will suddenly terminate
        process_info = Mock()
        process_info.component_type = ComponentType.RUST_DAEMON
        process_info.process_id = 9001
        process_info.is_running = True
        process_info.termination_reason = None
        process_info.cpu_usage_percent = 20.0
        process_info.memory_usage_mb = 150.0
        process_info.file_descriptor_count = 30
        process_info.thread_count = 8
        process_info.resource_violations = []
        process_info.last_health_check = None

        isolation_manager.component_processes["rust_daemon-9001"] = process_info

        # Simulate process termination (process no longer exists)
        with patch('psutil.Process', side_effect=psutil.NoSuchProcess(9001)):
            await isolation_manager._check_process_health()

        # Verify process was marked as not running
        assert not process_info.is_running
        assert process_info.termination_reason == "Process not found"

        # Verify recovery was triggered
        recovery_manager = full_isolation_system['recovery_manager']
        recovery_manager.trigger_component_recovery.assert_called()

    @pytest.mark.asyncio
    async def test_zombie_process_handling(self, full_isolation_system):
        """Test handling of zombie processes."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Add a process that becomes a zombie
        process_info = Mock()
        process_info.component_type = ComponentType.PYTHON_MCP_SERVER
        process_info.process_id = 9002
        process_info.is_running = True
        process_info.termination_reason = None

        isolation_manager.component_processes["python_mcp_server-9002"] = process_info

        # Simulate zombie process
        with patch('psutil.Process', side_effect=psutil.ZombieProcess(9002)):
            await isolation_manager._check_process_health()

        # Verify zombie process was handled
        assert not process_info.is_running
        assert process_info.termination_reason == "Process not found"

    @pytest.mark.asyncio
    async def test_segmentation_fault_simulation(self, full_isolation_system):
        """Test handling of segmentation fault scenarios."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Simulate a process that crashes with segfault
        process_info = Mock()
        process_info.component_type = ComponentType.CONTEXT_INJECTOR
        process_info.process_id = 9003
        process_info.is_running = True
        process_info.termination_reason = None

        isolation_manager.component_processes["context_injector-9003"] = process_info

        # Simulate segfault by making process unavailable
        mock_process = Mock()
        mock_process.is_running.return_value = False
        mock_process.returncode = -11  # SIGSEGV

        with patch('psutil.Process', return_value=mock_process):
            await isolation_manager._check_process_health()

        # Verify crash was handled
        assert not process_info.is_running
        assert "Process no longer running" in process_info.termination_reason

        # Verify isolation event was created
        crash_events = [
            event for event in isolation_manager.isolation_events
            if event.isolation_strategy == IsolationStrategy.PROCESS_SEPARATION
        ]
        assert len(crash_events) >= 1

    @pytest.mark.asyncio
    async def test_access_denied_handling(self, full_isolation_system):
        """Test handling of access denied scenarios."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Add a process that becomes inaccessible
        process_info = Mock()
        process_info.component_type = ComponentType.CLI_UTILITY
        process_info.process_id = 9004
        process_info.is_running = True
        process_info.resource_violations = []

        isolation_manager.component_processes["cli_utility-9004"] = process_info

        # Simulate access denied when checking process
        with patch('psutil.Process', side_effect=psutil.AccessDenied(9004)):
            # This should not crash the monitoring system
            await isolation_manager._check_process_health()

        # Verify the system handled the access denied gracefully
        # Process should still be marked as running since we couldn't verify otherwise
        assert process_info.is_running


class TestIsolationEffectiveness:
    """Test overall isolation effectiveness and system resilience."""

    @pytest.mark.asyncio
    async def test_multiple_simultaneous_failures(self, full_isolation_system):
        """Test system resilience under multiple simultaneous failures."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Configure all components with different failure thresholds
        component_configs = {
            ComponentType.RUST_DAEMON: ComponentBoundary(
                component_type=ComponentType.RUST_DAEMON,
                boundary_type=BoundaryType.SYNCHRONOUS,
                timeout_seconds=1.0,
                critical_exceptions={"SystemError"},
                max_error_rate=0.1
            ),
            ComponentType.PYTHON_MCP_SERVER: ComponentBoundary(
                component_type=ComponentType.PYTHON_MCP_SERVER,
                boundary_type=BoundaryType.MESSAGE_QUEUE,
                timeout_seconds=2.0,
                critical_exceptions={"RuntimeError"},
                max_error_rate=0.15
            ),
            ComponentType.CLI_UTILITY: ComponentBoundary(
                component_type=ComponentType.CLI_UTILITY,
                boundary_type=BoundaryType.SYNCHRONOUS,
                timeout_seconds=0.5,
                critical_exceptions={"ValueError"},
                max_error_rate=0.2
            ),
            ComponentType.CONTEXT_INJECTOR: ComponentBoundary(
                component_type=ComponentType.CONTEXT_INJECTOR,
                boundary_type=BoundaryType.EVENT_STREAM,
                timeout_seconds=3.0,
                critical_exceptions={"OSError"},
                max_error_rate=0.05
            )
        }

        # Apply configurations
        for component_type, boundary in component_configs.items():
            await isolation_manager.set_component_boundary(component_type, boundary)

        # Simulate simultaneous failures in multiple components
        failure_tasks = []

        async def fail_component(component_type: ComponentType, exception_type: type, error_msg: str):
            try:
                async with isolation_manager.component_boundary(
                    component_type,
                    f"failing_operation_{component_type.value}"
                ):
                    raise exception_type(error_msg)
            except Exception:
                pass  # Expected to fail

        # Start failures simultaneously
        failure_tasks.extend([
            fail_component(ComponentType.RUST_DAEMON, SystemError, "Daemon system error"),
            fail_component(ComponentType.PYTHON_MCP_SERVER, RuntimeError, "MCP runtime error"),
            fail_component(ComponentType.CLI_UTILITY, ValueError, "CLI value error"),
            fail_component(ComponentType.CONTEXT_INJECTOR, OSError, "Injector OS error")
        ])

        # Execute all failures simultaneously
        await asyncio.gather(*failure_tasks, return_exceptions=True)

        # Verify that multiple components were isolated
        assert isolation_manager.isolation_count >= 2

        # Verify that each failed component triggered its own recovery
        recovery_manager = full_isolation_system['recovery_manager']
        assert recovery_manager.trigger_component_recovery.call_count >= 2

        # Verify that isolation events were created for multiple components
        isolation_events = [
            event for event in isolation_manager.isolation_events
            if event.isolation_strategy == IsolationStrategy.FAILURE_CONTAINMENT
        ]
        assert len(isolation_events) >= 2

    @pytest.mark.asyncio
    async def test_isolation_under_load(self, full_isolation_system):
        """Test isolation effectiveness under high load conditions."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Configure component with load-appropriate settings
        boundary = ComponentBoundary(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            boundary_type=BoundaryType.MESSAGE_QUEUE,
            timeout_seconds=5.0,
            max_concurrent_calls=50,
            queue_timeout_seconds=1.0,
            max_error_rate=0.3
        )

        await isolation_manager.set_component_boundary(ComponentType.PYTHON_MCP_SERVER, boundary)

        # Generate high load with mixed success/failure
        async def load_operation(operation_id: int):
            try:
                async with isolation_manager.component_boundary(
                    ComponentType.PYTHON_MCP_SERVER,
                    f"load_operation_{operation_id}"
                ):
                    # Simulate work with occasional failures
                    if operation_id % 10 == 0:  # 10% failure rate
                        raise RuntimeError(f"Load operation {operation_id} failed")
                    await asyncio.sleep(0.01)  # Brief work simulation
                    return f"success_{operation_id}"
            except Exception:
                return f"failed_{operation_id}"

        # Generate 100 concurrent operations
        load_tasks = [load_operation(i) for i in range(100)]
        results = await asyncio.gather(*load_tasks, return_exceptions=True)

        # Analyze results
        successes = [r for r in results if isinstance(r, str) and r.startswith("success")]
        failures = [r for r in results if isinstance(r, str) and r.startswith("failed")]

        # Should have mostly successes with some expected failures
        assert len(successes) >= 80  # At least 80% success rate
        assert len(failures) <= 20   # At most 20% failure rate

        # Error rate should not trigger isolation due to tolerance
        assert isolation_manager.isolation_count == 0

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention_validation(self, full_isolation_system):
        """Validate that cascading failures are actually prevented."""
        isolation_manager = full_isolation_system['isolation_manager']

        # Configure components with dependencies
        boundaries = {
            ComponentType.RUST_DAEMON: ComponentBoundary(
                component_type=ComponentType.RUST_DAEMON,
                boundary_type=BoundaryType.SYNCHRONOUS,
                timeout_seconds=1.0,
                critical_exceptions={"ConnectionError"},
                max_error_rate=0.1
            ),
            ComponentType.PYTHON_MCP_SERVER: ComponentBoundary(
                component_type=ComponentType.PYTHON_MCP_SERVER,
                boundary_type=BoundaryType.MESSAGE_QUEUE,
                timeout_seconds=2.0,
                critical_exceptions={"RuntimeError"},
                max_error_rate=0.2
            ),
            ComponentType.CLI_UTILITY: ComponentBoundary(
                component_type=ComponentType.CLI_UTILITY,
                boundary_type=BoundaryType.SYNCHRONOUS,
                timeout_seconds=1.5,
                critical_exceptions={"SystemError"},
                max_error_rate=0.15
            )
        }

        for component_type, boundary in boundaries.items():
            await isolation_manager.set_component_boundary(component_type, boundary)

        # Phase 1: Fail the primary component (Rust daemon)
        with pytest.raises(ConnectionError):
            async with isolation_manager.component_boundary(
                ComponentType.RUST_DAEMON,
                "primary_failure"
            ):
                raise ConnectionError("Primary component failed")

        # Verify primary component was isolated
        initial_isolation_count = isolation_manager.isolation_count
        assert initial_isolation_count >= 1

        # Phase 2: Test that dependent components can still operate independently
        dependent_successes = 0

        try:
            async with isolation_manager.component_boundary(
                ComponentType.PYTHON_MCP_SERVER,
                "dependent_operation_1"
            ):
                # This should succeed independently
                await asyncio.sleep(0.01)
                dependent_successes += 1
        except Exception as e:
            pytest.fail(f"Dependent component should not fail due to primary failure: {e}")

        try:
            async with isolation_manager.component_boundary(
                ComponentType.CLI_UTILITY,
                "dependent_operation_2"
            ):
                # This should also succeed independently
                await asyncio.sleep(0.01)
                dependent_successes += 1
        except Exception as e:
            pytest.fail(f"CLI component should not fail due to primary failure: {e}")

        # Verify dependent components operated successfully
        assert dependent_successes == 2

        # Verify no additional isolation was triggered by dependent operations
        assert isolation_manager.isolation_count == initial_isolation_count

        # Phase 3: Verify system can continue operating with reduced functionality
        try:
            async with isolation_manager.component_boundary(
                ComponentType.PYTHON_MCP_SERVER,
                "continued_operation"
            ):
                await asyncio.sleep(0.01)
        except Exception as e:
            pytest.fail(f"System should continue operating with reduced functionality: {e}")

    @pytest.mark.asyncio
    async def test_recovery_after_isolation(self, full_isolation_system):
        """Test system recovery after component isolation."""
        isolation_manager = full_isolation_system['isolation_manager']
        recovery_manager = full_isolation_system['recovery_manager']

        # Configure component
        boundary = ComponentBoundary(
            component_type=ComponentType.CONTEXT_INJECTOR,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=1.0,
            critical_exceptions={"RuntimeError"}
        )

        await isolation_manager.set_component_boundary(ComponentType.CONTEXT_INJECTOR, boundary)

        # Phase 1: Trigger isolation
        with pytest.raises(RuntimeError):
            async with isolation_manager.component_boundary(
                ComponentType.CONTEXT_INJECTOR,
                "failure_before_recovery"
            ):
                raise RuntimeError("Component failure triggering isolation")

        # Verify isolation occurred
        assert isolation_manager.isolation_count >= 1
        recovery_manager.trigger_component_recovery.assert_called()

        # Phase 2: Simulate successful recovery
        # Reset error counts to simulate recovery
        error_key = "context_injector_errors"
        if error_key in isolation_manager.error_counts:
            isolation_manager.error_counts[error_key].clear()

        # Simulate recovery completion by resetting degradation manager response
        degradation_manager = full_isolation_system['degradation_manager']
        degradation_manager.is_component_available.return_value = True

        # Phase 3: Test that component can operate normally after recovery
        try:
            async with isolation_manager.component_boundary(
                ComponentType.CONTEXT_INJECTOR,
                "operation_after_recovery"
            ):
                # This should work normally after recovery
                await asyncio.sleep(0.01)
        except Exception as e:
            pytest.fail(f"Component should work normally after recovery: {e}")

        # Verify no additional isolation was triggered
        initial_isolation_count = isolation_manager.isolation_count

        # Multiple successful operations should not trigger additional isolation
        for i in range(5):
            try:
                async with isolation_manager.component_boundary(
                    ComponentType.CONTEXT_INJECTOR,
                    f"post_recovery_operation_{i}"
                ):
                    await asyncio.sleep(0.01)
            except Exception as e:
                pytest.fail(f"Post-recovery operation {i} should succeed: {e}")

        # Isolation count should remain the same
        assert isolation_manager.isolation_count == initial_isolation_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])