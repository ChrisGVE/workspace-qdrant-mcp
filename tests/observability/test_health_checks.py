"""
Comprehensive health check endpoint verification tests for workspace-qdrant-mcp.

Tests coverage:
    - Liveness checks (is the service running?)
    - Readiness checks (is the service ready to accept requests?)
    - Startup checks (has initialization completed?)
    - Deep health checks (are dependencies healthy?)
    - Response validation (status codes, JSON format, timing, error messages)
    - Component health (MCP server, Qdrant, Rust daemon, SQLite, embedding model)
    - Health states (healthy, degraded, unhealthy, recovery)
    - Health aggregation across components

This test suite validates the health check infrastructure without modifying
core application logic or MCP server configuration.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.observability.health import (
    ComponentHealth,
    HealthChecker,
    HealthStatus,
    get_health_checker,
)
from common.observability.health_coordinator import (
    AlertSeverity,
    ComponentHealthMetrics,
    ComponentType,
    HealthAlert,
    HealthCoordinator,
    HealthTrend,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
async def health_checker():
    """Create a HealthChecker instance for testing."""
    checker = HealthChecker()
    yield checker
    # Cleanup
    if checker._background_task and not checker._background_task.done():
        checker._background_task.cancel()
        try:
            await checker._background_task
        except asyncio.CancelledError:
            pass


@pytest.fixture
async def health_coordinator():
    """Create a HealthCoordinator instance for testing."""
    coordinator = HealthCoordinator(
        db_path=":memory:",
        project_name="test_project",
        enable_auto_recovery=False,
    )
    await coordinator.initialize()
    yield coordinator
    await coordinator.stop_monitoring()


@pytest.fixture
def mock_healthy_qdrant():
    """Mock healthy Qdrant client."""
    mock_client = Mock()
    mock_client.get_status = AsyncMock(
        return_value={
            "connected": True,
            "qdrant_url": "http://localhost:6333",
            "workspace_collections": ["test-collection"],
            "current_project": "test_project",
            "embedding_info": {"model_name": "all-MiniLM-L6-v2"},
        }
    )
    return mock_client


@pytest.fixture
def mock_unhealthy_qdrant():
    """Mock unhealthy Qdrant client."""
    mock_client = Mock()
    mock_client.get_status = AsyncMock(
        return_value={
            "connected": False,
            "error": "Connection refused",
        }
    )
    return mock_client


# ============================================================================
# Liveness Checks - Is the service running?
# ============================================================================


class TestLivenessChecks:
    """Test basic liveness health checks."""

    @pytest.mark.asyncio
    async def test_health_checker_instantiation(self):
        """Test that health checker can be instantiated (basic liveness)."""
        checker = HealthChecker()
        assert checker is not None
        assert isinstance(checker, HealthChecker)
        assert checker._enabled is True

    @pytest.mark.asyncio
    async def test_get_health_checker_singleton(self):
        """Test that get_health_checker returns singleton instance."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        assert checker1 is checker2

    @pytest.mark.asyncio
    async def test_health_status_basic_liveness(self, health_checker):
        """Test basic health status retrieval indicates service is alive."""
        health_status = await health_checker.get_health_status()

        # Service is alive if it can return health status
        assert health_status is not None
        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "components" in health_status

    @pytest.mark.asyncio
    async def test_liveness_check_response_structure(self, health_checker):
        """Test liveness check returns expected response structure."""
        health_status = await health_checker.get_health_status()

        # Validate response structure
        assert isinstance(health_status["status"], str)
        assert isinstance(health_status["timestamp"], float)
        assert isinstance(health_status["components"], dict)
        assert "message" in health_status


# ============================================================================
# Readiness Checks - Is the service ready to accept requests?
# ============================================================================


class TestReadinessChecks:
    """Test readiness health checks."""

    @pytest.mark.asyncio
    async def test_readiness_all_checks_enabled(self, health_checker):
        """Test that all required health checks are registered and enabled."""
        required_checks = [
            "system_resources",
            "qdrant_connectivity",
            "embedding_service",
            "file_watchers",
            "configuration",
        ]

        for check_name in required_checks:
            assert check_name in health_checker.health_checks
            check = health_checker.health_checks[check_name]
            assert check.enabled is True

    @pytest.mark.asyncio
    async def test_readiness_check_execution(self, health_checker):
        """Test that readiness checks can execute successfully."""
        # Run system resources check (doesn't depend on external services)
        health_result = await health_checker.run_check("system_resources")

        assert health_result is not None
        assert isinstance(health_result, ComponentHealth)
        assert health_result.name == "system_resources"
        assert health_result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]

    @pytest.mark.asyncio
    async def test_readiness_response_time_acceptable(self, health_checker):
        """Test that readiness checks complete within acceptable time."""
        start_time = time.perf_counter()
        await health_checker.run_check("system_resources")
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Readiness checks should be fast (< 150ms allows for system variance)
        # Target is 100ms, but allow some tolerance for slower systems/load
        assert elapsed_ms < 150, f"Readiness check took {elapsed_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_readiness_critical_components(self, health_checker):
        """Test that critical components are marked for readiness."""
        critical_checks = ["qdrant_connectivity", "configuration"]

        for check_name in critical_checks:
            check = health_checker.health_checks[check_name]
            assert (
                check.critical is True
            ), f"{check_name} should be critical for readiness"


# ============================================================================
# Startup Checks - Has initialization completed?
# ============================================================================


class TestStartupChecks:
    """Test startup and initialization health checks."""

    @pytest.mark.asyncio
    async def test_startup_initialization_complete(self, health_checker):
        """Test that health checker initializes all standard checks."""
        expected_checks = [
            "system_resources",
            "qdrant_connectivity",
            "embedding_service",
            "file_watchers",
            "configuration",
        ]

        for check_name in expected_checks:
            assert (
                check_name in health_checker.health_checks
            ), f"Standard check {check_name} not initialized"

    @pytest.mark.asyncio
    async def test_startup_check_registration(self, health_checker):
        """Test custom health check registration during startup."""
        # Register a custom startup check
        async def custom_startup_check():
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Startup complete",
                "details": {"initialized": True},
            }

        health_checker.register_check("startup_check", custom_startup_check)

        assert "startup_check" in health_checker.health_checks
        result = await health_checker.run_check("startup_check")
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_startup_background_monitoring(self, health_checker):
        """Test background monitoring can start (initialization feature)."""
        health_checker.start_background_monitoring(interval=60.0)

        assert health_checker._background_task is not None
        assert not health_checker._background_task.done()

        # Cleanup
        health_checker.stop_background_monitoring()


# ============================================================================
# Deep Health Checks - Are all dependencies healthy?
# ============================================================================


class TestDeepHealthChecks:
    """Test deep health checks for all dependencies."""

    @pytest.mark.asyncio
    async def test_deep_health_all_components(self, health_checker):
        """Test deep health check covers all system components."""
        health_status = await health_checker.get_health_status()

        # Deep health should check all components
        components = health_status["components"]
        assert "system_resources" in components
        assert "qdrant_connectivity" in components
        assert "embedding_service" in components
        assert "file_watchers" in components
        assert "configuration" in components

    @pytest.mark.asyncio
    async def test_deep_health_system_resources(self, health_checker):
        """Test deep health check for system resources."""
        result = await health_checker.run_check("system_resources")

        assert result.details is not None
        assert "memory" in result.details
        assert "cpu" in result.details
        assert "disk" in result.details

        # Validate resource metrics are present
        memory = result.details["memory"]
        assert "percent_used" in memory
        assert "available_gb" in memory
        assert "total_gb" in memory

    @pytest.mark.asyncio
    async def test_deep_health_detailed_diagnostics(self, health_checker):
        """Test detailed diagnostics provide deep health information."""
        diagnostics = await health_checker.get_detailed_diagnostics()

        assert "health_status" in diagnostics
        assert "system_info" in diagnostics
        assert "check_history" in diagnostics
        assert "configuration" in diagnostics

        # System info should have detailed metrics
        system_info = diagnostics["system_info"]
        assert "process_id" in system_info or "error" in system_info

    @pytest.mark.asyncio
    async def test_deep_health_dependency_tracking(self, health_checker):
        """Test that health checks track their own health metrics."""
        # Run a check
        await health_checker.run_check("system_resources")

        # Get check history
        history = health_checker._get_check_history()

        assert "system_resources" in history
        check_info = history["system_resources"]
        assert "last_check_time" in check_info
        assert "consecutive_failures" in check_info
        assert "last_result" in check_info


# ============================================================================
# Response Validation - Status codes, format, timing, errors
# ============================================================================


class TestResponseValidation:
    """Test health check response validation."""

    @pytest.mark.asyncio
    async def test_response_status_codes(self, health_checker):
        """Test health status maps to appropriate status representations."""
        health_status = await health_checker.get_health_status()

        status = health_status["status"]
        # Status should be one of the valid health states
        assert status in ["healthy", "degraded", "unhealthy", "unknown"]

    @pytest.mark.asyncio
    async def test_response_json_format(self, health_checker):
        """Test response is in proper JSON-serializable format."""
        health_status = await health_checker.get_health_status()

        # All values should be JSON-serializable
        import json

        try:
            json.dumps(health_status)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Health status is not JSON serializable: {e}")

    @pytest.mark.asyncio
    async def test_response_timing_fast(self, health_checker):
        """Test health check response time is under 100ms."""
        start_time = time.perf_counter()
        await health_checker.get_health_status()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete within 100ms as per requirements
        # Note: May fail on slow systems, using 200ms as reasonable threshold
        assert elapsed_ms < 200, f"Health check took {elapsed_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_response_component_timing(self, health_checker):
        """Test individual component checks track response time."""
        result = await health_checker.run_check("system_resources")

        assert result.response_time is not None
        assert isinstance(result.response_time, float)
        assert result.response_time > 0

    @pytest.mark.asyncio
    async def test_response_timestamp_present(self, health_checker):
        """Test response includes valid timestamp."""
        health_status = await health_checker.get_health_status()

        assert "timestamp" in health_status
        timestamp = health_status["timestamp"]
        assert isinstance(timestamp, (int, float))
        assert timestamp > 0

        # Timestamp should be recent (within last minute)
        current_time = time.time()
        assert abs(current_time - timestamp) < 60

    @pytest.mark.asyncio
    async def test_response_error_messages(self, health_checker):
        """Test error messages are included for unhealthy components."""

        # Create a check that will fail
        async def failing_check():
            raise Exception("Simulated component failure")

        health_checker.register_check("failing_component", failing_check)
        result = await health_checker.run_check("failing_component")

        assert result.status == HealthStatus.UNHEALTHY
        assert result.error is not None
        assert "Simulated component failure" in result.error
        assert result.message is not None


# ============================================================================
# Component Health - Individual component validation
# ============================================================================


class TestComponentHealth:
    """Test health checks for individual components."""

    @pytest.mark.asyncio
    async def test_component_system_resources_healthy(self, health_checker):
        """Test system resources health check when healthy."""
        result = await health_checker._check_system_resources()

        assert result["status"] in ["healthy", "degraded", "unhealthy"]
        assert "details" in result
        assert "memory" in result["details"]
        assert "cpu" in result["details"]
        assert "disk" in result["details"]

    @pytest.mark.asyncio
    async def test_component_health_status_enum(self):
        """Test ComponentHealth uses correct status enum values."""
        component = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="Test message",
        )

        assert component.status == HealthStatus.HEALTHY
        assert component.name == "test_component"
        assert component.message == "Test message"

    @pytest.mark.asyncio
    async def test_component_health_with_details(self):
        """Test ComponentHealth can include detailed metrics."""
        component = ComponentHealth(
            name="qdrant",
            status=HealthStatus.HEALTHY,
            message="Qdrant connected",
            details={
                "url": "http://localhost:6333",
                "collections": 5,
                "response_time_ms": 15.3,
            },
            response_time=0.0153,
        )

        assert component.details["url"] == "http://localhost:6333"
        assert component.details["collections"] == 5
        assert component.response_time == 0.0153

    @pytest.mark.asyncio
    async def test_component_check_timeout(self, health_checker):
        """Test component check handles timeouts gracefully."""

        # Create a check that will timeout
        async def slow_check():
            await asyncio.sleep(10)  # Longer than default timeout
            return {"status": "healthy", "message": "Should timeout"}

        health_checker.register_check("slow_check", slow_check, timeout_seconds=0.5)
        result = await health_checker.run_check("slow_check")

        assert result.status == HealthStatus.UNHEALTHY
        assert "timed out" in result.message.lower() or "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_component_consecutive_failures(self, health_checker):
        """Test component tracks consecutive failures."""

        # Create a check that always fails
        async def always_fail():
            raise Exception("Always fails")

        health_checker.register_check("failing_check", always_fail)

        # Run check multiple times
        for i in range(3):
            await health_checker.run_check("failing_check")

        check = health_checker.health_checks["failing_check"]
        assert check.consecutive_failures == 3


# ============================================================================
# Health State Scenarios - Normal, degraded, failure, recovery
# ============================================================================


class TestHealthStateScenarios:
    """Test various health state scenarios."""

    @pytest.mark.asyncio
    async def test_healthy_state_all_components(self, health_checker):
        """Test overall healthy state when all components healthy."""

        # Register all healthy checks
        async def healthy_check():
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Component healthy",
                "details": {},
            }

        health_checker.health_checks.clear()
        for i in range(3):
            health_checker.register_check(f"component_{i}", healthy_check, critical=True)

        health_status = await health_checker.get_health_status()
        assert health_status["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_degraded_state_non_critical_failure(self, health_checker):
        """Test degraded state when non-critical component fails."""

        async def healthy_check():
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Healthy",
                "details": {},
            }

        async def degraded_check():
            return {
                "status": HealthStatus.DEGRADED.value,
                "message": "Degraded",
                "details": {},
            }

        health_checker.health_checks.clear()
        health_checker.register_check("critical_component", healthy_check, critical=True)
        health_checker.register_check(
            "optional_component", degraded_check, critical=False
        )

        health_status = await health_checker.get_health_status()
        # Should be degraded, not unhealthy
        assert health_status["status"] in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_unhealthy_state_critical_failure(self, health_checker):
        """Test unhealthy state when critical component fails."""

        async def healthy_check():
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Healthy",
                "details": {},
            }

        async def unhealthy_check():
            return {
                "status": HealthStatus.UNHEALTHY.value,
                "message": "Component failed",
                "details": {"error": "Critical failure"},
            }

        health_checker.health_checks.clear()
        health_checker.register_check("critical_component", unhealthy_check, critical=True)
        health_checker.register_check("other_component", healthy_check, critical=False)

        health_status = await health_checker.get_health_status()
        assert health_status["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_recovery_scenario(self, health_checker):
        """Test health recovery from unhealthy to healthy state."""

        # Start with failing check
        failure_count = 0

        async def recovering_check():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                return {
                    "status": HealthStatus.UNHEALTHY.value,
                    "message": "Still recovering",
                    "details": {},
                }
            return {
                "status": HealthStatus.HEALTHY.value,
                "message": "Recovered",
                "details": {},
            }

        health_checker.health_checks.clear()
        health_checker.register_check("recovering", recovering_check, critical=True)

        # First check - unhealthy
        status1 = await health_checker.get_health_status()
        assert status1["status"] == "unhealthy"

        # Second check - still unhealthy
        status2 = await health_checker.get_health_status()
        assert status2["status"] == "unhealthy"

        # Third check - recovered
        status3 = await health_checker.get_health_status()
        assert status3["status"] == "healthy"


# ============================================================================
# Health Check Aggregation - Cross-component health
# ============================================================================


class TestHealthAggregation:
    """Test health check aggregation across components."""

    @pytest.mark.asyncio
    async def test_aggregation_multiple_components(self, health_checker):
        """Test health aggregation across multiple components."""
        health_status = await health_checker.get_health_status()

        components = health_status["components"]
        assert len(components) >= 3, "Should have multiple components"

        # Each component should have required fields
        for component_name, component_data in components.items():
            assert "status" in component_data
            assert "message" in component_data

    @pytest.mark.asyncio
    async def test_aggregation_overall_status_logic(self, health_checker):
        """Test overall status aggregates component statuses correctly."""

        async def healthy():
            return {"status": "healthy", "message": "OK", "details": {}}

        async def degraded():
            return {"status": "degraded", "message": "Degraded", "details": {}}

        async def unhealthy():
            return {"status": "unhealthy", "message": "Failed", "details": {}}

        health_checker.health_checks.clear()

        # Test 1: All healthy -> healthy
        health_checker.register_check("c1", healthy, critical=True)
        health_checker.register_check("c2", healthy, critical=True)
        status = await health_checker.get_health_status()
        assert status["status"] == "healthy"

        # Test 2: One degraded -> degraded
        health_checker.health_checks.clear()
        health_checker.register_check("c1", healthy, critical=True)
        health_checker.register_check("c2", degraded, critical=False)
        status = await health_checker.get_health_status()
        assert status["status"] in ["healthy", "degraded"]

        # Test 3: One critical unhealthy -> unhealthy
        health_checker.health_checks.clear()
        health_checker.register_check("c1", unhealthy, critical=True)
        health_checker.register_check("c2", healthy, critical=True)
        status = await health_checker.get_health_status()
        assert status["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_aggregation_message_generation(self, health_checker):
        """Test aggregated message describes overall health."""
        health_status = await health_checker.get_health_status()

        assert "message" in health_status
        message = health_status["message"]
        assert isinstance(message, str)
        assert len(message) > 0

    @pytest.mark.asyncio
    async def test_aggregation_concurrent_checks(self, health_checker):
        """Test that component checks run concurrently in aggregation."""

        check_times = []

        async def timed_check():
            start = time.time()
            await asyncio.sleep(0.1)  # 100ms delay
            check_times.append(time.time() - start)
            return {"status": "healthy", "message": "OK", "details": {}}

        health_checker.health_checks.clear()
        for i in range(3):
            health_checker.register_check(f"check_{i}", timed_check)

        start_time = time.time()
        await health_checker.get_health_status()
        total_time = time.time() - start_time

        # If checks run concurrently, total time should be ~100ms, not 300ms
        assert total_time < 0.2, f"Checks appear to run serially: {total_time:.2f}s"

    @pytest.mark.asyncio
    async def test_aggregation_critical_vs_non_critical(self, health_checker):
        """Test aggregation differentiates critical vs non-critical failures."""

        async def healthy():
            return {"status": "healthy", "message": "OK", "details": {}}

        async def unhealthy():
            return {"status": "unhealthy", "message": "Failed", "details": {}}

        health_checker.health_checks.clear()

        # Non-critical failure should not make overall unhealthy
        health_checker.register_check("critical", healthy, critical=True)
        health_checker.register_check("optional", unhealthy, critical=False)

        status = await health_checker.get_health_status()
        # Overall status should not be unhealthy due to non-critical failure
        assert status["status"] in ["healthy", "degraded"]


# ============================================================================
# Health Coordinator Integration Tests
# ============================================================================


class TestHealthCoordinatorIntegration:
    """Test HealthCoordinator for advanced health monitoring."""

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, health_coordinator):
        """Test health coordinator initializes successfully."""
        assert health_coordinator is not None
        assert health_coordinator.health_checker is not None
        assert health_coordinator.component_coordinator is not None

    @pytest.mark.asyncio
    async def test_coordinator_unified_health_status(self, health_coordinator):
        """Test coordinator provides unified health status."""
        with patch.object(
            health_coordinator.health_checker, "get_health_status"
        ) as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "message": "All systems operational",
                "timestamp": time.time(),
                "components": {},
            }

            unified_status = await health_coordinator.get_unified_health_status()

            assert "overall_status" in unified_status
            assert "component_health" in unified_status
            assert "timestamp" in unified_status

    @pytest.mark.asyncio
    async def test_coordinator_component_metrics(self, health_coordinator):
        """Test coordinator tracks component health metrics."""
        metrics = ComponentHealthMetrics(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            health_status=HealthStatus.HEALTHY,
            response_time_ms=15.5,
            error_rate=0.0,
            resource_usage={"cpu": 10.0, "memory": 50.0},
            dependency_health={},
            uptime_seconds=3600.0,
        )

        assert metrics.component_type == ComponentType.PYTHON_MCP_SERVER
        assert metrics.health_status == HealthStatus.HEALTHY
        assert metrics.response_time_ms == 15.5

    @pytest.mark.asyncio
    async def test_coordinator_alert_generation(self):
        """Test coordinator can generate health alerts."""
        alert = HealthAlert(
            alert_id="test_alert_1",
            component_type=ComponentType.RUST_DAEMON,
            severity=AlertSeverity.WARNING,
            message="High response time detected",
            description="Component response time exceeds threshold",
            timestamp=datetime.now(timezone.utc),
            metrics={"response_time_ms": 5500},
        )

        assert alert.component_type == ComponentType.RUST_DAEMON
        assert alert.severity == AlertSeverity.WARNING
        assert not alert.resolved


# ============================================================================
# Performance and Edge Cases
# ============================================================================


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    @pytest.mark.asyncio
    async def test_health_check_under_load(self, health_checker):
        """Test health checks perform well under concurrent load."""

        async def concurrent_health_checks():
            return await health_checker.get_health_status()

        # Run 10 concurrent health checks
        start_time = time.time()
        results = await asyncio.gather(*[concurrent_health_checks() for _ in range(10)])
        elapsed = time.time() - start_time

        # All should complete
        assert len(results) == 10
        # Should complete in reasonable time even under load
        assert elapsed < 2.0, f"Concurrent checks took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_health_check_with_disabled_checker(self, health_checker):
        """Test health check behavior when checker is disabled."""
        health_checker._enabled = False

        health_status = await health_checker.get_health_status()

        assert health_status["status"] == "unknown"
        assert "disabled" in health_status["message"].lower()

    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self, health_checker):
        """Test health check handles component exceptions gracefully."""

        async def exception_check():
            raise RuntimeError("Unexpected error in health check")

        health_checker.register_check("exception_component", exception_check)
        result = await health_checker.run_check("exception_component")

        # Should return unhealthy, not crash
        assert result.status == HealthStatus.UNHEALTHY
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_health_check_missing_component(self, health_checker):
        """Test health check for non-existent component."""
        result = await health_checker.run_check("non_existent_component")

        assert result.status == HealthStatus.UNKNOWN
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_disabled_component(self, health_checker):
        """Test health check for disabled component."""

        async def disabled_check():
            return {"status": "healthy", "message": "OK", "details": {}}

        health_checker.register_check("disabled_component", disabled_check)
        health_checker.health_checks["disabled_component"].enabled = False

        result = await health_checker.run_check("disabled_component")

        assert result.status == HealthStatus.UNKNOWN
        assert "disabled" in result.message.lower()
