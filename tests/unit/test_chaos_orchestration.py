"""
Unit Tests for Chaos Engineering Orchestration (Task 305.9)

Comprehensive tests for chaos experiment scheduling, result correlation,
and orchestration coordination.
"""

import pytest
import asyncio
import time
from tests.framework.chaos_orchestration import (
    ChaosOrchestrator,
    ChaosOrchestrationConfig,
    ChaosExperimentScheduler,
    ChaosResultCorrelator,
    ChaosExperimentConfig,
    ChaosExperimentResult,
    ChaosExperimentType,
    ExperimentStatus,
    TriggerCondition,
    AlertSeverity,
)
from tests.framework.resource_exhaustion import ResourceExhaustionScenario
from tests.framework.network_instability import NetworkInstabilityScenario
from tests.framework.cascading_failures import CascadeScenario, FailureNode, FailureType
from tests.framework.recovery_metrics import (
    RecoveryTimeTracker,
    RecoveryEvent,
    RecoveryState,
    MTTRAnalysis,
)
from tests.framework.performance_degradation import (
    PerformanceDegradationTracker,
    PerformanceMetricType,
    PerformanceDegradation,
    PerformanceTrend,
)


class TestChaosExperimentScheduler:
    """Test ChaosExperimentScheduler functionality."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = ChaosExperimentScheduler()

        assert len(scheduler.pending_experiments) == 0
        assert len(scheduler.running_experiments) == 0
        assert len(scheduler.completed_experiments) == 0

    def test_schedule_experiment(self):
        """Test scheduling a single experiment."""
        scheduler = ChaosExperimentScheduler()

        config = ChaosExperimentConfig(
            experiment_id="test-001",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.IMMEDIATE
        )

        scheduler.schedule_experiment(config)

        assert len(scheduler.pending_experiments) == 1
        assert scheduler.pending_experiments[0].experiment_id == "test-001"

    def test_immediate_trigger(self):
        """Test immediate trigger condition."""
        scheduler = ChaosExperimentScheduler()

        config = ChaosExperimentConfig(
            experiment_id="test-immediate",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.IMMEDIATE
        )
        scheduler.schedule_experiment(config)

        next_exp = scheduler.get_next_experiment()

        assert next_exp is not None
        assert next_exp.experiment_id == "test-immediate"
        assert len(scheduler.running_experiments) == 1
        assert len(scheduler.pending_experiments) == 0

    def test_scheduled_time_trigger_not_ready(self):
        """Test scheduled time trigger when time hasn't come yet."""
        scheduler = ChaosExperimentScheduler()

        future_time = time.time() + 3600  # 1 hour in future

        config = ChaosExperimentConfig(
            experiment_id="test-scheduled",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.SCHEDULED_TIME,
            trigger_params={"scheduled_time": future_time}
        )
        scheduler.schedule_experiment(config)

        next_exp = scheduler.get_next_experiment()

        assert next_exp is None
        assert len(scheduler.pending_experiments) == 1

    def test_scheduled_time_trigger_ready(self):
        """Test scheduled time trigger when time has come."""
        scheduler = ChaosExperimentScheduler()

        past_time = time.time() - 60  # 1 minute ago

        config = ChaosExperimentConfig(
            experiment_id="test-scheduled",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.SCHEDULED_TIME,
            trigger_params={"scheduled_time": past_time}
        )
        scheduler.schedule_experiment(config)

        next_exp = scheduler.get_next_experiment()

        assert next_exp is not None
        assert next_exp.experiment_id == "test-scheduled"

    def test_load_threshold_trigger(self):
        """Test load threshold trigger condition."""
        scheduler = ChaosExperimentScheduler()

        config = ChaosExperimentConfig(
            experiment_id="test-load",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.LOAD_THRESHOLD,
            trigger_params={"load_threshold": 0.8}  # 80% threshold
        )
        scheduler.schedule_experiment(config)

        # Test with low load
        next_exp = scheduler.get_next_experiment(
            current_metrics={"cpu_percent": 50.0}  # 50%
        )
        assert next_exp is None

        # Test with high load
        next_exp = scheduler.get_next_experiment(
            current_metrics={"cpu_percent": 85.0}  # 85%
        )
        assert next_exp is not None

    def test_metric_threshold_trigger(self):
        """Test metric threshold trigger condition."""
        scheduler = ChaosExperimentScheduler()

        config = ChaosExperimentConfig(
            experiment_id="test-metric",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.METRIC_THRESHOLD,
            trigger_params={
                "metric_name": "latency_ms",
                "threshold": 500.0
            }
        )
        scheduler.schedule_experiment(config)

        # Test below threshold
        next_exp = scheduler.get_next_experiment(
            current_metrics={"latency_ms": 200.0}
        )
        assert next_exp is None

        # Test above threshold
        next_exp = scheduler.get_next_experiment(
            current_metrics={"latency_ms": 600.0}
        )
        assert next_exp is not None

    def test_max_concurrent_experiments(self):
        """Test max concurrent experiment limit."""
        scheduler = ChaosExperimentScheduler()

        # Schedule two experiments
        for i in range(2):
            config = ChaosExperimentConfig(
                experiment_id=f"test-{i}",
                experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
                trigger_condition=TriggerCondition.IMMEDIATE
            )
            scheduler.schedule_experiment(config)

        # Get first experiment (should succeed)
        exp1 = scheduler.get_next_experiment(max_concurrent=1)
        assert exp1 is not None
        assert len(scheduler.running_experiments) == 1

        # Try to get second experiment with max_concurrent=1 (should fail)
        exp2 = scheduler.get_next_experiment(max_concurrent=1)
        assert exp2 is None
        assert len(scheduler.running_experiments) == 1

    def test_complete_experiment(self):
        """Test completing an experiment."""
        scheduler = ChaosExperimentScheduler()

        config = ChaosExperimentConfig(
            experiment_id="test-complete",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.IMMEDIATE
        )
        scheduler.schedule_experiment(config)

        # Get experiment
        exp = scheduler.get_next_experiment()
        assert len(scheduler.running_experiments) == 1

        # Complete experiment
        result = ChaosExperimentResult(
            experiment_id="test-complete",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            status=ExperimentStatus.COMPLETED,
            start_time=time.time(),
            success=True
        )
        scheduler.complete_experiment(result)

        assert len(scheduler.running_experiments) == 0
        assert len(scheduler.completed_experiments) == 1

    def test_experiment_summary(self):
        """Test getting experiment summary."""
        scheduler = ChaosExperimentScheduler()

        # Add pending experiments
        for i in range(3):
            config = ChaosExperimentConfig(
                experiment_id=f"test-{i}",
                experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
                trigger_condition=TriggerCondition.IMMEDIATE
            )
            scheduler.schedule_experiment(config)

        # Start one
        scheduler.get_next_experiment()

        # Complete one successfully
        result1 = ChaosExperimentResult(
            experiment_id="test-success",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            status=ExperimentStatus.COMPLETED,
            start_time=time.time(),
            success=True
        )
        scheduler.complete_experiment(result1)

        # Complete one failed
        result2 = ChaosExperimentResult(
            experiment_id="test-failed",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            status=ExperimentStatus.FAILED,
            start_time=time.time(),
            success=False
        )
        scheduler.complete_experiment(result2)

        summary = scheduler.get_experiment_summary()

        assert summary["pending"] == 2
        assert summary["running"] == 1
        assert summary["completed"] == 2
        assert summary["success_rate"] == 0.5  # 1 success, 1 failure

    def test_cancel_all_experiments(self):
        """Test cancelling all pending experiments."""
        scheduler = ChaosExperimentScheduler()

        # Schedule multiple experiments
        for i in range(5):
            config = ChaosExperimentConfig(
                experiment_id=f"test-{i}",
                experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
                trigger_condition=TriggerCondition.IMMEDIATE
            )
            scheduler.schedule_experiment(config)

        assert len(scheduler.pending_experiments) == 5

        scheduler.cancel_all_experiments()

        assert len(scheduler.pending_experiments) == 0


class TestChaosResultCorrelator:
    """Test ChaosResultCorrelator functionality."""

    def test_initialization(self):
        """Test correlator initialization."""
        correlator = ChaosResultCorrelator()
        assert correlator is not None

    def test_correlate_performance_impact_no_degradations(self):
        """Test performance correlation with no degradations."""
        correlator = ChaosResultCorrelator()
        perf_tracker = PerformanceDegradationTracker()

        result = ChaosExperimentResult(
            experiment_id="test-perf",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            status=ExperimentStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time() + 60,
            success=True
        )

        correlation = correlator.correlate_performance_impact(result, perf_tracker)

        assert correlation["degradation_count"] == 0
        assert correlation["max_degradation_percent"] == 0.0
        assert correlation["critical_degradations"] == 0

    def test_correlate_performance_impact_with_degradations(self):
        """Test performance correlation with degradations."""
        correlator = ChaosResultCorrelator()
        perf_tracker = PerformanceDegradationTracker()

        # Register metric and establish baseline
        perf_tracker.register_metric("latency_ms", PerformanceMetricType.LATENCY)
        perf_tracker.establish_baseline("latency_ms", [100.0] * 10)

        # Create experiment result
        result = ChaosExperimentResult(
            experiment_id="test-perf",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            status=ExperimentStatus.COMPLETED,
            start_time=time.time(),
            end_time=None,  # Still running
            success=True
        )

        # Record degradations during experiment
        perf_tracker.record_metric("latency_ms", 130.0, timestamp=result.start_time + 10)  # 30% degradation

        # End experiment
        result.end_time = time.time() + 60

        # Correlate
        correlation = correlator.correlate_performance_impact(result, perf_tracker)

        assert correlation["degradation_count"] == 1
        assert correlation["max_degradation_percent"] == pytest.approx(30.0, abs=1.0)
        assert correlation["avg_degradation_percent"] == pytest.approx(30.0, abs=1.0)

    def test_correlate_recovery_metrics_no_failures(self):
        """Test recovery correlation with no failures."""
        correlator = ChaosResultCorrelator()
        recovery_tracker = RecoveryTimeTracker()

        result = ChaosExperimentResult(
            experiment_id="test-recovery",
            experiment_type=ChaosExperimentType.CASCADING_FAILURE,
            status=ExperimentStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time() + 60,
            success=True
        )

        correlation = correlator.correlate_recovery_metrics(result, recovery_tracker)

        assert correlation["failure_count"] == 0
        assert correlation["recovery_count"] == 0
        assert correlation["avg_recovery_time"] == 0.0

    def test_correlate_recovery_metrics_with_failures(self):
        """Test recovery correlation with failures."""
        correlator = ChaosResultCorrelator()
        recovery_tracker = RecoveryTimeTracker()

        # Create experiment result
        start_time = time.time()
        result = ChaosExperimentResult(
            experiment_id="test-recovery",
            experiment_type=ChaosExperimentType.CASCADING_FAILURE,
            status=ExperimentStatus.COMPLETED,
            start_time=start_time,
            end_time=start_time + 120,
            success=True
        )

        # Record failures during experiment
        recovery_tracker.record_failure("component_a", start_time + 10, "test failure")
        recovery_tracker.record_recovery_start("component_a", start_time + 15)
        recovery_tracker.record_recovery_complete("component_a", start_time + 30, True, 1.0)

        # Correlate
        correlation = correlator.correlate_recovery_metrics(result, recovery_tracker)

        assert correlation["failure_count"] == 1
        assert correlation["recovery_count"] == 1
        assert correlation["avg_recovery_time"] > 0.0

    def test_generate_correlation_report(self):
        """Test generating complete correlation report."""
        correlator = ChaosResultCorrelator()

        result = ChaosExperimentResult(
            experiment_id="test-report",
            experiment_type=ChaosExperimentType.COMBINED,
            status=ExperimentStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time() + 60,
            duration_seconds=60.0,
            success=True,
            max_degradation_percent=25.0,
            avg_degradation_percent=15.0
        )

        # Add some results
        result.resource_exhaustion_results = {"memory_mb": 512}
        result.network_instability_results = {"latency_ms": 200}

        report = correlator.generate_correlation_report(result)

        assert report["experiment_id"] == "test-report"
        assert report["experiment_type"] == "combined"
        assert report["success"] is True
        assert report["performance_impact"]["max_degradation_percent"] == 25.0
        assert "experiment_results" in report


class TestChaosOrchestrator:
    """Test ChaosOrchestrator integration."""

    def test_initialization(self):
        """Test orchestrator initialization."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        assert orchestrator.config == config
        assert orchestrator.scheduler is not None
        assert orchestrator.correlator is not None
        assert orchestrator.running is False

    def test_add_experiment(self):
        """Test adding experiments to orchestrator."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        exp_config = ChaosExperimentConfig(
            experiment_id="test-exp",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.IMMEDIATE
        )

        orchestrator.add_experiment(exp_config)

        assert len(orchestrator.scheduler.pending_experiments) == 1

    @pytest.mark.asyncio
    async def test_run_experiments_empty(self):
        """Test running with no experiments."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        results = await orchestrator.run_experiments(duration_seconds=1.0)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_execute_experiment_resource_exhaustion(self):
        """Test executing resource exhaustion experiment."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        # Create minimal resource exhaustion scenario
        scenario = ResourceExhaustionScenario(
            duration_seconds=0.5,
            memory_target_mb=10  # Very small
        )

        exp_config = ChaosExperimentConfig(
            experiment_id="test-resource",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.IMMEDIATE,
            duration_seconds=1.0,
            baseline_duration_seconds=0.1,
            resource_exhaustion_scenario=scenario
        )

        result = await orchestrator._execute_experiment(exp_config)

        assert result.experiment_id == "test-resource"
        assert result.status == ExperimentStatus.COMPLETED
        assert result.success is True
        assert result.duration_seconds is not None
        assert result.baseline_metrics is not None

    @pytest.mark.asyncio
    async def test_execute_experiment_network_instability(self):
        """Test executing network instability experiment."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        # Create minimal network scenario
        scenario = NetworkInstabilityScenario(
            duration_seconds=0.5,
            latency_ms=50
        )

        exp_config = ChaosExperimentConfig(
            experiment_id="test-network",
            experiment_type=ChaosExperimentType.NETWORK_INSTABILITY,
            trigger_condition=TriggerCondition.IMMEDIATE,
            duration_seconds=1.0,
            baseline_duration_seconds=0.1,
            network_instability_scenario=scenario
        )

        result = await orchestrator._execute_experiment(exp_config)

        assert result.experiment_id == "test-network"
        assert result.status == ExperimentStatus.COMPLETED
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_experiment_cascading_failure(self):
        """Test executing cascading failure experiment."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        # Create minimal cascade scenario
        node_a = FailureNode(
            node_id="node_a",
            failure_type=FailureType.COMPONENT_CRASH,
            trigger_delay_seconds=0.1,
            recovery_delay_seconds=0.2
        )

        scenario = CascadeScenario(
            failure_nodes=[node_a],
            initial_trigger="node_a",
            max_cascade_depth=1
        )

        exp_config = ChaosExperimentConfig(
            experiment_id="test-cascade",
            experiment_type=ChaosExperimentType.CASCADING_FAILURE,
            trigger_condition=TriggerCondition.IMMEDIATE,
            duration_seconds=1.0,
            baseline_duration_seconds=0.1,
            cascade_scenario=scenario
        )

        result = await orchestrator._execute_experiment(exp_config)

        assert result.experiment_id == "test-cascade"
        assert result.status == ExperimentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_experiments_with_duration_limit(self):
        """Test running experiments with duration limit."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        # Add a quick experiment
        scenario = ResourceExhaustionScenario(
            duration_seconds=0.1,
            memory_target_mb=10
        )

        exp_config = ChaosExperimentConfig(
            experiment_id="test-quick",
            experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
            trigger_condition=TriggerCondition.IMMEDIATE,
            duration_seconds=0.1,
            baseline_duration_seconds=0.05,
            resource_exhaustion_scenario=scenario
        )

        orchestrator.add_experiment(exp_config)

        # Run with short duration that allows experiment to complete
        start = time.time()
        results = await orchestrator.run_experiments(duration_seconds=2.0)
        elapsed = time.time() - start

        # Should complete the quick experiment within duration limit
        assert elapsed < 2.0
        assert len(results) == 1
        assert results[0].success is True

    def test_orchestrator_stop(self):
        """Test stopping orchestrator."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        # Add some experiments
        for i in range(3):
            exp_config = ChaosExperimentConfig(
                experiment_id=f"test-{i}",
                experiment_type=ChaosExperimentType.RESOURCE_EXHAUSTION,
                trigger_condition=TriggerCondition.IMMEDIATE
            )
            orchestrator.add_experiment(exp_config)

        orchestrator.running = True
        orchestrator.stop()

        assert orchestrator.running is False
        assert len(orchestrator.scheduler.pending_experiments) == 0

    def test_get_summary(self):
        """Test getting orchestrator summary."""
        config = ChaosOrchestrationConfig()
        orchestrator = ChaosOrchestrator(config)

        summary = orchestrator.get_summary()

        assert "orchestrator_status" in summary
        assert summary["orchestrator_status"] == "stopped"
        assert "experiments" in summary
        assert summary["current_experiment"] is None
