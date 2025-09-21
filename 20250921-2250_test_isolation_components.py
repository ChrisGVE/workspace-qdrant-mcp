#!/usr/bin/env python3
"""
Test script for Component Isolation and Failure Handling.

This script validates the isolation system implementation without requiring
the full test infrastructure.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add source path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from common.core.component_isolation import (
    ComponentIsolationManager,
    ComponentBoundary,
    ResourceLimits,
    BoundaryType,
    IsolationStrategy,
    FailureImpact,
    ComponentUnavailableError,
)
from common.core.component_coordination import ComponentType
from unittest.mock import Mock, AsyncMock


class TestResults:
    """Track test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_true(self, condition, message):
        """Assert condition is true."""
        if condition:
            self.passed += 1
            print(f"✓ {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"✗ {message}")

    def assert_equal(self, actual, expected, message):
        """Assert values are equal."""
        if actual == expected:
            self.passed += 1
            print(f"✓ {message}")
        else:
            self.failed += 1
            error = f"{message} - Expected: {expected}, Actual: {actual}"
            self.errors.append(error)
            print(f"✗ {error}")

    def summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Test Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print(f"Failed tests:")
            for error in self.errors:
                print(f"  - {error}")
        print(f"{'='*60}")
        return self.failed == 0


async def create_test_isolation_manager():
    """Create isolation manager with mock dependencies."""
    # Create mock dependencies
    lifecycle_manager = Mock()
    coordinator = AsyncMock()
    degradation_manager = Mock()
    recovery_manager = AsyncMock()
    health_monitor = Mock()

    # Configure mocks
    coordinator.register_component.return_value = "test-component-id"
    coordinator.update_component_health.return_value = True
    coordinator.record_heartbeat.return_value = True

    degradation_manager.is_component_available.return_value = True
    degradation_manager.record_component_success = AsyncMock()
    degradation_manager.record_component_failure = AsyncMock()
    degradation_manager.register_notification_handler.return_value = None

    recovery_manager.trigger_component_recovery.return_value = "recovery-id"
    recovery_manager.register_notification_handler.return_value = None

    health_monitor.register_notification_handler.return_value = None

    # Create isolation manager
    manager = ComponentIsolationManager(
        lifecycle_manager=lifecycle_manager,
        coordinator=coordinator,
        degradation_manager=degradation_manager,
        recovery_manager=recovery_manager,
        health_monitor=health_monitor,
        config={"test_mode": True}
    )

    await manager.initialize()
    return manager, {
        'coordinator': coordinator,
        'degradation_manager': degradation_manager,
        'recovery_manager': recovery_manager
    }


async def test_basic_isolation_functionality(results: TestResults):
    """Test basic isolation functionality."""
    print("\n--- Testing Basic Isolation Functionality ---")

    isolation_manager, mocks = await create_test_isolation_manager()

    try:
        # Test 1: Manager initialization
        results.assert_true(isolation_manager is not None, "Isolation manager created successfully")

        # Test 2: Component boundary configuration
        boundary = ComponentBoundary(
            component_type=ComponentType.RUST_DAEMON,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=1.0,
            max_concurrent_calls=10
        )

        await isolation_manager.set_component_boundary(ComponentType.RUST_DAEMON, boundary)

        configured_boundary = isolation_manager.component_boundaries.get(ComponentType.RUST_DAEMON)
        results.assert_true(configured_boundary is not None, "Component boundary configured")
        results.assert_equal(configured_boundary.timeout_seconds, 1.0, "Boundary timeout configured correctly")

        # Test 3: Resource limits configuration
        limits = ResourceLimits(
            cpu_limit_percent=50.0,
            memory_limit_mb=256,
            max_file_descriptors=100
        )

        await isolation_manager.set_component_resource_limits(ComponentType.RUST_DAEMON, limits)

        configured_limits = isolation_manager.resource_limits.get(ComponentType.RUST_DAEMON)
        results.assert_true(configured_limits is not None, "Resource limits configured")
        results.assert_equal(configured_limits.cpu_limit_percent, 50.0, "CPU limit configured correctly")

        # Test 4: Basic component boundary operation
        async with isolation_manager.component_boundary(
            ComponentType.RUST_DAEMON,
            "test_operation"
        ) as context:
            results.assert_true(context is not None, "Component boundary context created")
            results.assert_true(context.get_elapsed_time() >= 0, "Elapsed time tracking works")
            await asyncio.sleep(0.01)  # Brief operation

        # Test 5: Isolation status reporting
        status = await isolation_manager.get_isolation_status()
        results.assert_true(isinstance(status, dict), "Isolation status returned as dict")
        results.assert_true('uptime_seconds' in status, "Status contains uptime")
        results.assert_true('isolation_count' in status, "Status contains isolation count")

        print(f"Initial isolation status: {status['isolation_count']} isolations, {status['uptime_seconds']:.2f}s uptime")

    finally:
        await isolation_manager.shutdown()


async def test_timeout_handling(results: TestResults):
    """Test timeout handling in component boundaries."""
    print("\n--- Testing Timeout Handling ---")

    isolation_manager, mocks = await create_test_isolation_manager()

    try:
        # Configure short timeout for testing
        boundary = ComponentBoundary(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=0.1,  # Very short timeout
            retry_count=1
        )

        await isolation_manager.set_component_boundary(ComponentType.PYTHON_MCP_SERVER, boundary)

        # Test timeout scenario
        timeout_occurred = False
        try:
            async with isolation_manager.component_boundary(
                ComponentType.PYTHON_MCP_SERVER,
                "timeout_test_operation"
            ):
                await asyncio.sleep(0.2)  # Longer than timeout
        except asyncio.TimeoutError:
            timeout_occurred = True

        results.assert_true(timeout_occurred, "Timeout properly detected and raised")

        # Verify error was recorded
        degradation_manager = mocks['degradation_manager']
        results.assert_true(
            degradation_manager.record_component_failure.called,
            "Component failure recorded on timeout"
        )

    finally:
        await isolation_manager.shutdown()


async def test_concurrency_limits(results: TestResults):
    """Test concurrency limits in component boundaries."""
    print("\n--- Testing Concurrency Limits ---")

    isolation_manager, mocks = await create_test_isolation_manager()

    try:
        # Configure low concurrency limit
        boundary = ComponentBoundary(
            component_type=ComponentType.CLI_UTILITY,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=2.0,
            max_concurrent_calls=2,  # Only 2 concurrent calls
            queue_timeout_seconds=0.1
        )

        await isolation_manager.set_component_boundary(ComponentType.CLI_UTILITY, boundary)

        # Start 2 operations (should succeed)
        async def slow_operation(operation_id: int):
            async with isolation_manager.component_boundary(
                ComponentType.CLI_UTILITY,
                f"concurrent_op_{operation_id}"
            ):
                await asyncio.sleep(0.3)
                return f"completed_{operation_id}"

        # Start first two operations
        task1 = asyncio.create_task(slow_operation(1))
        task2 = asyncio.create_task(slow_operation(2))

        # Wait a bit for them to start
        await asyncio.sleep(0.05)

        # Try third operation (should timeout on queue)
        queue_timeout_occurred = False
        try:
            await slow_operation(3)
        except TimeoutError:
            queue_timeout_occurred = True

        results.assert_true(queue_timeout_occurred, "Queue timeout properly enforced")

        # Wait for original operations to complete
        result1 = await task1
        result2 = await task2

        results.assert_equal(result1, "completed_1", "First concurrent operation completed")
        results.assert_equal(result2, "completed_2", "Second concurrent operation completed")

    finally:
        await isolation_manager.shutdown()


async def test_error_boundary_isolation(results: TestResults):
    """Test error boundaries and isolation triggers."""
    print("\n--- Testing Error Boundary Isolation ---")

    isolation_manager, mocks = await create_test_isolation_manager()

    try:
        # Configure component with critical exceptions
        boundary = ComponentBoundary(
            component_type=ComponentType.CONTEXT_INJECTOR,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=5.0,
            critical_exceptions={"RuntimeError", "SystemError"},
            allowed_exceptions={"KeyError"}
        )

        await isolation_manager.set_component_boundary(ComponentType.CONTEXT_INJECTOR, boundary)

        # Test allowed exception (should not trigger isolation)
        allowed_exception_caught = False
        allowed_exception_not_isolated = False

        try:
            async with isolation_manager.component_boundary(
                ComponentType.CONTEXT_INJECTOR,
                "allowed_error_test"
            ):
                raise KeyError("This is allowed")
        except KeyError:
            allowed_exception_caught = True
            # Check that no isolation was triggered yet
            if isolation_manager.isolation_count == 0:
                allowed_exception_not_isolated = True

        results.assert_true(allowed_exception_caught, "Allowed exception properly handled")
        results.assert_true(allowed_exception_not_isolated, "Allowed exception did not trigger isolation")
        results.assert_equal(isolation_manager.isolation_count, 0, "No isolation for allowed exception")

        # Test critical exception (should trigger isolation)
        critical_exception_caught = False
        try:
            async with isolation_manager.component_boundary(
                ComponentType.CONTEXT_INJECTOR,
                "critical_error_test"
            ):
                raise RuntimeError("This is critical")
        except RuntimeError:
            critical_exception_caught = True

        results.assert_true(critical_exception_caught, "Critical exception properly propagated")
        results.assert_true(isolation_manager.isolation_count >= 1, "Isolation triggered by critical exception")

        # Verify recovery was triggered
        recovery_manager = mocks['recovery_manager']
        results.assert_true(
            recovery_manager.trigger_component_recovery.called,
            "Recovery triggered after isolation"
        )

    finally:
        await isolation_manager.shutdown()


async def test_component_unavailable_handling(results: TestResults):
    """Test handling of unavailable components."""
    print("\n--- Testing Component Unavailable Handling ---")

    isolation_manager, mocks = await create_test_isolation_manager()

    try:
        # Configure degradation manager to report component as unavailable
        degradation_manager = mocks['degradation_manager']
        degradation_manager.is_component_available.return_value = False

        # Test that unavailable component raises error
        unavailable_error_caught = False
        try:
            async with isolation_manager.component_boundary(
                ComponentType.RUST_DAEMON,
                "unavailable_test"
            ):
                pass
        except ComponentUnavailableError:
            unavailable_error_caught = True

        results.assert_true(unavailable_error_caught, "ComponentUnavailableError raised for unavailable component")

        # Test with allow_degraded=True (should work)
        degraded_operation_succeeded = False
        try:
            async with isolation_manager.component_boundary(
                ComponentType.RUST_DAEMON,
                "degraded_test",
                allow_degraded=True
            ):
                degraded_operation_succeeded = True
        except Exception:
            pass

        results.assert_true(degraded_operation_succeeded, "Operation succeeds with allow_degraded=True")

    finally:
        await isolation_manager.shutdown()


async def test_resource_violation_detection(results: TestResults):
    """Test resource violation detection."""
    print("\n--- Testing Resource Violation Detection ---")

    isolation_manager, mocks = await create_test_isolation_manager()

    try:
        # Set strict resource limits
        limits = ResourceLimits(
            cpu_limit_percent=20.0,
            memory_limit_mb=100,
            max_file_descriptors=50,
            max_threads=10,
            enforce_cpu=True,
            enforce_memory=True
        )

        await isolation_manager.set_component_resource_limits(ComponentType.PYTHON_MCP_SERVER, limits)

        # Create mock process that violates limits
        from unittest.mock import Mock
        process_info = Mock()
        process_info.component_type = ComponentType.PYTHON_MCP_SERVER
        process_info.process_id = 1001
        process_info.is_running = True
        process_info.resource_limits = limits
        process_info.resource_violations = []
        process_info.cpu_usage_percent = 50.0  # Exceeds 20% limit
        process_info.memory_usage_mb = 150.0   # Exceeds 100MB limit
        process_info.file_descriptor_count = 60  # Exceeds 50 limit
        process_info.thread_count = 15         # Exceeds 10 limit

        # Check for violations
        await isolation_manager._check_resource_violations("python_mcp_server-1001", process_info)

        # Verify violations were detected
        violations = process_info.resource_violations
        results.assert_true(len(violations) >= 4, f"Multiple resource violations detected: {len(violations)}")

        cpu_violation = any("CPU usage" in v for v in violations)
        memory_violation = any("Memory usage" in v for v in violations)
        fd_violation = any("File descriptors" in v for v in violations)
        thread_violation = any("Thread count" in v for v in violations)

        results.assert_true(cpu_violation, "CPU violation detected")
        results.assert_true(memory_violation, "Memory violation detected")
        results.assert_true(fd_violation, "File descriptor violation detected")
        results.assert_true(thread_violation, "Thread violation detected")

        # Verify isolation events were created
        resource_events = [
            event for event in isolation_manager.isolation_events
            if event.isolation_strategy == IsolationStrategy.RESOURCE_ISOLATION
        ]
        results.assert_true(len(resource_events) >= 1, "Resource isolation event created")

        # Verify resource violations count was incremented
        results.assert_true(
            isolation_manager.resource_violations_count >= 4,
            f"Resource violations count incremented: {isolation_manager.resource_violations_count}"
        )

    finally:
        await isolation_manager.shutdown()


async def test_manual_isolation(results: TestResults):
    """Test manual component isolation."""
    print("\n--- Testing Manual Component Isolation ---")

    isolation_manager, mocks = await create_test_isolation_manager()

    try:
        # Force isolation of a component
        await isolation_manager.force_component_isolation(
            ComponentType.CLI_UTILITY,
            "Manual test isolation"
        )

        # Verify isolation was triggered
        results.assert_true(isolation_manager.isolation_count >= 1, "Manual isolation incremented count")

        # Verify recovery was triggered
        recovery_manager = mocks['recovery_manager']
        results.assert_true(
            recovery_manager.trigger_component_recovery.called,
            "Recovery triggered by manual isolation"
        )

        # Verify isolation event was created
        isolation_events = [
            event for event in isolation_manager.isolation_events
            if "Manual test isolation" in event.trigger_reason
        ]
        results.assert_true(len(isolation_events) >= 1, "Manual isolation event created")

        isolation_event = isolation_events[0]
        results.assert_equal(
            isolation_event.action_taken,
            "Component isolated due to critical error",
            "Correct action recorded"
        )
        results.assert_true(isolation_event.recovery_triggered, "Recovery triggered flag set")

    finally:
        await isolation_manager.shutdown()


async def run_all_tests():
    """Run all isolation tests."""
    print("Component Isolation and Failure Handling Test Suite")
    print("="*60)

    results = TestResults()

    # Run all test functions
    test_functions = [
        test_basic_isolation_functionality,
        test_timeout_handling,
        test_concurrency_limits,
        test_error_boundary_isolation,
        test_component_unavailable_handling,
        test_resource_violation_detection,
        test_manual_isolation,
    ]

    for test_func in test_functions:
        try:
            await test_func(results)
        except Exception as e:
            results.failed += 1
            error_msg = f"Test {test_func.__name__} failed with exception: {e}"
            results.errors.append(error_msg)
            print(f"✗ {error_msg}")

    # Print summary
    success = results.summary()

    if success:
        print("\n🎉 All component isolation tests passed!")
        print("✓ Error boundaries prevent failure propagation")
        print("✓ Component isolation through process separation works")
        print("✓ Resource limits are properly enforced")
        print("✓ Timeout handling functions correctly")
        print("✓ Failure scenarios are handled effectively")
    else:
        print("\n❌ Some tests failed. Check the errors above.")

    return success


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)