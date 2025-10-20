"""
Unit tests for stress testing orchestration module.

Tests multi-component coordination, failure injection, recovery measurement,
and performance degradation tracking.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from tests.framework.stress_orchestration import (
    LoadPattern,
    FailureMode,
    StressPipelineStage,
    StressTestConfig,
    ComponentStressConfig,
    StressTestResult,
    MultiComponentCoordinator,
    StressTestOrchestrator,
)
from tests.framework.orchestration import (
    OrchestrationMode,
    OrchestrationPriority,
    PipelineStage,
)
from tests.framework.integration import (
    IntegrationTestCoordinator,
    ComponentConfig,
    ComponentType,
    ComponentInstance,
    ComponentState,
    IsolationLevel,
)


class TestLoadPattern:
    """Test LoadPattern enum."""

    def test_load_patterns(self):
        """Test all load patterns are defined."""
        assert LoadPattern.CONSTANT.value == "constant"
        assert LoadPattern.RAMP_UP.value == "ramp_up"
        assert LoadPattern.SPIKE.value == "spike"
        assert LoadPattern.SUSTAINED.value == "sustained"


class TestFailureMode:
    """Test FailureMode enum."""

    def test_failure_modes(self):
        """Test all failure modes are defined."""
        assert FailureMode.CRASH.value == "crash"
        assert FailureMode.HANG.value == "hang"
        assert FailureMode.SLOW.value == "slow"
        assert FailureMode.NETWORK_PARTITION.value == "network_partition"


class TestStressPipelineStage:
    """Test StressPipelineStage enum."""

    def test_stress_stages(self):
        """Test all stress pipeline stages are defined."""
        assert StressPipelineStage.RESOURCE_BASELINE.value == "resource_baseline"
        assert StressPipelineStage.LOAD_RAMP.value == "load_ramp"
        assert StressPipelineStage.STRESS_EXECUTION.value == "stress_execution"
        assert StressPipelineStage.FAILURE_INJECTION.value == "failure_injection"
        assert StressPipelineStage.RECOVERY_VALIDATION.value == "recovery_validation"
        assert StressPipelineStage.DEGRADATION_ANALYSIS.value == "degradation_analysis"


class TestStressTestConfig:
    """Test StressTestConfig dataclass."""

    def test_default_config(self):
        """Test default stress test configuration."""
        config = StressTestConfig()

        assert config.load_pattern == LoadPattern.CONSTANT
        assert config.duration_hours == 24.0
        assert config.resource_constraints["memory_mb"] == 1024
        assert config.resource_constraints["cpu_percent"] == 80
        assert config.resource_constraints["disk_io_mb_s"] == 100
        assert config.failure_injection_enabled is False
        assert config.performance_thresholds["p50_ms"] == 100.0
        assert config.performance_thresholds["p95_ms"] == 500.0
        assert config.performance_thresholds["p99_ms"] == 1000.0
        assert config.performance_thresholds["error_rate_percent"] == 1.0
        assert config.stability_checkpoints_minutes == 30

    def test_custom_config(self):
        """Test custom stress test configuration."""
        config = StressTestConfig(
            load_pattern=LoadPattern.RAMP_UP,
            duration_hours=12.0,
            failure_injection_enabled=True,
            stability_checkpoints_minutes=15
        )

        assert config.load_pattern == LoadPattern.RAMP_UP
        assert config.duration_hours == 12.0
        assert config.failure_injection_enabled is True
        assert config.stability_checkpoints_minutes == 15

    def test_extends_orchestration_config(self):
        """Test that StressTestConfig extends OrchestrationConfig."""
        config = StressTestConfig(
            mode=OrchestrationMode.FULL_PIPELINE,
            priority=OrchestrationPriority.HIGH,
            max_workers=8
        )

        assert config.mode == OrchestrationMode.FULL_PIPELINE
        assert config.priority == OrchestrationPriority.HIGH
        assert config.max_workers == 8


class TestComponentStressConfig:
    """Test ComponentStressConfig dataclass."""

    def test_default_config(self):
        """Test default component stress configuration."""
        config = ComponentStressConfig(component_name="test-component")

        assert config.component_name == "test-component"
        assert config.resource_limits["memory_mb"] == 512
        assert config.resource_limits["cpu_percent"] == 50
        assert config.failure_modes == ["crash"]
        assert config.health_check_endpoint is None
        assert config.recovery_timeout_seconds == 30.0

    def test_custom_config(self):
        """Test custom component stress configuration."""
        config = ComponentStressConfig(
            component_name="api-service",
            resource_limits={"memory_mb": 1024, "cpu_percent": 80},
            failure_modes=["crash", "hang", "slow"],
            health_check_endpoint="http://localhost:8080/health",
            recovery_timeout_seconds=60.0
        )

        assert config.component_name == "api-service"
        assert config.resource_limits["memory_mb"] == 1024
        assert config.resource_limits["cpu_percent"] == 80
        assert len(config.failure_modes) == 3
        assert "crash" in config.failure_modes
        assert "hang" in config.failure_modes
        assert "slow" in config.failure_modes
        assert config.health_check_endpoint == "http://localhost:8080/health"
        assert config.recovery_timeout_seconds == 60.0


class TestStressTestResult:
    """Test StressTestResult dataclass."""

    def test_default_result(self):
        """Test default stress test result."""
        result = StressTestResult(
            orchestration_id="test_123",
            start_time=time.time()
        )

        assert result.orchestration_id == "test_123"
        assert result.baseline_metrics == {}
        assert result.recovery_times == {}
        assert result.performance_samples == {}
        assert result.failure_injections == []
        assert result.stability_violations == []

    def test_avg_recovery_time(self):
        """Test average recovery time calculation."""
        result = StressTestResult(
            orchestration_id="test_123",
            start_time=time.time(),
            recovery_times={
                "service-a": 5.0,
                "service-b": 10.0,
                "service-c": 15.0
            }
        )

        assert result.avg_recovery_time == 10.0

    def test_avg_recovery_time_empty(self):
        """Test average recovery time with no data."""
        result = StressTestResult(
            orchestration_id="test_123",
            start_time=time.time()
        )

        assert result.avg_recovery_time == 0.0

    def test_max_recovery_time(self):
        """Test maximum recovery time calculation."""
        result = StressTestResult(
            orchestration_id="test_123",
            start_time=time.time(),
            recovery_times={
                "service-a": 5.0,
                "service-b": 15.0,
                "service-c": 10.0
            }
        )

        assert result.max_recovery_time == 15.0

    def test_max_recovery_time_empty(self):
        """Test maximum recovery time with no data."""
        result = StressTestResult(
            orchestration_id="test_123",
            start_time=time.time()
        )

        assert result.max_recovery_time == 0.0


class TestMultiComponentCoordinator:
    """Test MultiComponentCoordinator class."""

    @pytest.fixture
    def integration_coordinator(self):
        """Create mock integration coordinator."""
        coordinator = Mock(spec=IntegrationTestCoordinator)
        coordinator._components = {}
        coordinator.start_component = AsyncMock(return_value=True)
        coordinator.stop_component = AsyncMock(return_value=True)
        return coordinator

    @pytest.fixture
    def multi_coordinator(self, integration_coordinator):
        """Create multi-component coordinator."""
        return MultiComponentCoordinator(integration_coordinator)

    def test_initialization(self, integration_coordinator):
        """Test coordinator initialization."""
        coordinator = MultiComponentCoordinator(integration_coordinator)

        assert coordinator.integration == integration_coordinator
        assert coordinator.components == {}
        assert coordinator.failure_timestamps == {}
        assert coordinator.recovery_timestamps == {}

    @pytest.mark.asyncio
    async def test_start_all_components_success(self, multi_coordinator, integration_coordinator):
        """Test starting all components successfully."""
        # Setup mock components
        component_config = ComponentConfig(
            name="test-service",
            component_type=ComponentType.PYTHON_SERVICE
        )
        component_instance = ComponentInstance(config=component_config)

        integration_coordinator._components = {
            "test-service": component_instance
        }

        stress_configs = [
            ComponentStressConfig(component_name="test-service")
        ]

        results = await multi_coordinator.start_all_components(stress_configs)

        assert results["test-service"] is True
        assert "test-service" in multi_coordinator.components
        integration_coordinator.start_component.assert_called_once_with("test-service")

    @pytest.mark.asyncio
    async def test_start_all_components_not_registered(self, multi_coordinator):
        """Test starting component that is not registered."""
        stress_configs = [
            ComponentStressConfig(component_name="unknown-service")
        ]

        results = await multi_coordinator.start_all_components(stress_configs)

        assert results["unknown-service"] is False

    @pytest.mark.asyncio
    async def test_start_all_components_failure(self, multi_coordinator, integration_coordinator):
        """Test handling component start failure."""
        component_config = ComponentConfig(
            name="test-service",
            component_type=ComponentType.PYTHON_SERVICE
        )
        component_instance = ComponentInstance(config=component_config)

        integration_coordinator._components = {
            "test-service": component_instance
        }
        integration_coordinator.start_component = AsyncMock(return_value=False)

        stress_configs = [
            ComponentStressConfig(component_name="test-service")
        ]

        results = await multi_coordinator.start_all_components(stress_configs)

        assert results["test-service"] is False

    @pytest.mark.asyncio
    async def test_stop_component_crash(self, multi_coordinator):
        """Test stopping component with CRASH failure mode."""
        # Create mock component with process
        mock_process = Mock()
        mock_process.kill = Mock()

        component_config = ComponentConfig(
            name="test-service",
            component_type=ComponentType.PYTHON_SERVICE
        )
        component_instance = ComponentInstance(
            config=component_config,
            process=mock_process,
            pid=12345
        )

        multi_coordinator.components["test-service"] = component_instance
        multi_coordinator.integration._get_controller = Mock()

        timestamp = await multi_coordinator.stop_component(
            "test-service",
            FailureMode.CRASH.value
        )

        assert timestamp > 0
        assert "test-service" in multi_coordinator.failure_timestamps
        assert component_instance.state == ComponentState.FAILED
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_component_not_found(self, multi_coordinator):
        """Test stopping non-existent component."""
        with pytest.raises(ValueError, match="Component unknown not found"):
            await multi_coordinator.stop_component("unknown", FailureMode.CRASH.value)

    @pytest.mark.asyncio
    async def test_check_component_health_success(self, multi_coordinator, integration_coordinator):
        """Test checking component health when healthy."""
        mock_controller = Mock()
        mock_controller.health_check = AsyncMock(return_value=True)

        component_config = ComponentConfig(
            name="test-service",
            component_type=ComponentType.PYTHON_SERVICE
        )
        component_instance = ComponentInstance(config=component_config)

        multi_coordinator.components["test-service"] = component_instance
        integration_coordinator._get_controller = Mock(return_value=mock_controller)

        is_healthy = await multi_coordinator.check_component_health("test-service")

        assert is_healthy is True
        mock_controller.health_check.assert_called_once_with(component_instance)

    @pytest.mark.asyncio
    async def test_check_component_health_not_found(self, multi_coordinator):
        """Test checking health of non-existent component."""
        is_healthy = await multi_coordinator.check_component_health("unknown")

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_check_component_health_exception(self, multi_coordinator, integration_coordinator):
        """Test handling exception during health check."""
        mock_controller = Mock()
        mock_controller.health_check = AsyncMock(side_effect=Exception("Health check failed"))

        component_config = ComponentConfig(
            name="test-service",
            component_type=ComponentType.PYTHON_SERVICE
        )
        component_instance = ComponentInstance(config=component_config)

        multi_coordinator.components["test-service"] = component_instance
        integration_coordinator._get_controller = Mock(return_value=mock_controller)

        is_healthy = await multi_coordinator.check_component_health("test-service")

        assert is_healthy is False

    def test_get_all_component_statuses(self, multi_coordinator):
        """Test getting all component statuses."""
        component1 = ComponentInstance(
            config=ComponentConfig(name="service-a", component_type=ComponentType.PYTHON_SERVICE),
            state=ComponentState.RUNNING
        )
        component2 = ComponentInstance(
            config=ComponentConfig(name="service-b", component_type=ComponentType.RUST_SERVICE),
            state=ComponentState.STOPPED
        )

        multi_coordinator.components = {
            "service-a": component1,
            "service-b": component2
        }

        statuses = multi_coordinator.get_all_component_statuses()

        assert statuses["service-a"] == "RUNNING"
        assert statuses["service-b"] == "STOPPED"

    def test_get_all_component_statuses_empty(self, multi_coordinator):
        """Test getting statuses when no components."""
        statuses = multi_coordinator.get_all_component_statuses()

        assert statuses == {}


class TestStressTestOrchestrator:
    """Test StressTestOrchestrator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def orchestrator(self, temp_dir):
        """Create stress test orchestrator."""
        project_root = temp_dir / "project"
        project_root.mkdir()

        test_dir = project_root / "tests"
        test_dir.mkdir()

        config = StressTestConfig(
            duration_hours=0.001,  # Very short for testing
            failure_injection_enabled=True
        )

        return StressTestOrchestrator(
            project_root=project_root,
            test_directory=test_dir,
            config=config
        )

    def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert isinstance(orchestrator.stress_config, StressTestConfig)
        assert orchestrator.multi_coordinator is None
        assert orchestrator.baseline_metrics == {}
        assert orchestrator.performance_samples == {}

    def test_stress_database_tables_created(self, orchestrator):
        """Test stress-specific database tables are created."""
        import sqlite3

        conn = sqlite3.connect(orchestrator.database_path)
        cursor = conn.cursor()

        # Check stress_test_runs table
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='stress_test_runs'
        """)
        assert cursor.fetchone() is not None

        # Check component_recovery_times table
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='component_recovery_times'
        """)
        assert cursor.fetchone() is not None

        # Check stress_performance_metrics table
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='stress_performance_metrics'
        """)
        assert cursor.fetchone() is not None

        conn.close()

    @pytest.mark.asyncio
    async def test_orchestrate_stress_test_basic(self, orchestrator):
        """Test basic stress test orchestration."""
        # Mock integration coordinator
        mock_integration = Mock(spec=IntegrationTestCoordinator)
        mock_integration._components = {}
        mock_integration.register_component = Mock()

        orchestrator._get_component_integration = Mock(return_value=mock_integration)

        # Mock multi-coordinator
        mock_multi = Mock(spec=MultiComponentCoordinator)
        mock_multi.start_all_components = AsyncMock(return_value={"test-service": True})
        mock_multi.check_component_health = AsyncMock(return_value=True)
        mock_multi.failure_timestamps = {}
        mock_multi.recovery_timestamps = {}

        with patch.object(
            orchestrator,
            'multi_coordinator',
            mock_multi
        ):
            components = [
                ComponentStressConfig(component_name="test-service")
            ]

            result = await orchestrator.orchestrate_stress_test(components)

            assert result.orchestration_id.startswith("stress_")
            assert result.start_time > 0
            assert result.end_time > result.start_time

    def test_track_performance_degradation(self, orchestrator):
        """Test tracking performance degradation."""
        orchestrator.performance_samples["service-a"] = [100.0, 110.0, 120.0]
        orchestrator.performance_samples["service-b"] = [50.0, 55.0, 60.0]

        degradation = orchestrator.track_performance_degradation()

        assert "service-a" in degradation
        assert "service-b" in degradation
        assert len(degradation["service-a"]) == 3
        assert len(degradation["service-b"]) == 3

    @pytest.mark.asyncio
    async def test_inject_component_failure_without_coordinator(self, orchestrator):
        """Test failure injection without initialized coordinator."""
        with pytest.raises(RuntimeError, match="Multi-component coordinator not initialized"):
            await orchestrator.inject_component_failure("test", FailureMode.CRASH.value)

    @pytest.mark.asyncio
    async def test_measure_recovery_time_without_coordinator(self, orchestrator):
        """Test recovery measurement without initialized coordinator."""
        with pytest.raises(RuntimeError, match="Multi-component coordinator not initialized"):
            await orchestrator.measure_recovery_time("test")

    @pytest.mark.asyncio
    async def test_inject_component_failure_with_coordinator(self, orchestrator):
        """Test failure injection with coordinator."""
        mock_multi = Mock(spec=MultiComponentCoordinator)
        mock_multi.stop_component = AsyncMock()

        orchestrator.multi_coordinator = mock_multi

        await orchestrator.inject_component_failure("test-service", FailureMode.CRASH.value)

        mock_multi.stop_component.assert_called_once_with(
            "test-service",
            FailureMode.CRASH.value
        )

    @pytest.mark.asyncio
    async def test_measure_recovery_time_with_coordinator(self, orchestrator):
        """Test recovery measurement with coordinator."""
        mock_multi = Mock(spec=MultiComponentCoordinator)
        mock_multi.restart_component = AsyncMock(return_value=5.0)

        orchestrator.multi_coordinator = mock_multi

        recovery_time = await orchestrator.measure_recovery_time("test-service")

        assert recovery_time == 5.0
        mock_multi.restart_component.assert_called_once_with("test-service")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
