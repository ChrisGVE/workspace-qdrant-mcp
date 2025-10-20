"""
Chaos Engineering Orchestration for Stress Testing (Task 305.9)

This module provides comprehensive chaos engineering orchestration by integrating
all chaos testing components with automated experiment scheduling, result correlation,
and integration with the stress testing framework.

Features:
- Automated chaos experiment scheduling with configurable triggers
- Experiment result correlation with performance metrics and recovery times
- Integration with resource exhaustion, network instability, and cascading failures
- Performance degradation tracking during chaos experiments
- Recovery metrics collection and analysis
- Experiment reproducibility and reporting
- Multi-component chaos coordination
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import json

from .resource_exhaustion import (
    ResourceExhaustionSimulator,
    ResourceExhaustionScenario,
)
from .network_instability import (
    NetworkInstabilitySimulator,
    NetworkInstabilityScenario,
)
from .cascading_failures import (
    CascadingFailureSimulator,
    CascadeScenario,
)
from .recovery_metrics import (
    RecoveryTimeTracker,
    RecoveryEvent,
    MTTRAnalysis,
    MTTFAnalysis,
)
from .performance_degradation import (
    PerformanceDegradationTracker,
    PerformanceMetricType,
    PerformanceTrend,
    AlertSeverity,
)


class ChaosExperimentType(Enum):
    """Types of chaos experiments."""
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_INSTABILITY = "network_instability"
    CASCADING_FAILURE = "cascading_failure"
    COMBINED = "combined"  # Multiple chaos types simultaneously


class ExperimentStatus(Enum):
    """Status of a chaos experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TriggerCondition(Enum):
    """Conditions that trigger chaos experiments."""
    IMMEDIATE = "immediate"  # Start immediately
    SCHEDULED_TIME = "scheduled_time"  # Start at specific time
    LOAD_THRESHOLD = "load_threshold"  # Start when load exceeds threshold
    PERIODIC = "periodic"  # Run at regular intervals
    METRIC_THRESHOLD = "metric_threshold"  # Start when metric exceeds threshold


@dataclass
class ChaosExperimentConfig:
    """Configuration for a single chaos experiment."""
    experiment_id: str
    experiment_type: ChaosExperimentType
    trigger_condition: TriggerCondition

    # Experiment parameters
    duration_seconds: float = 60.0
    target_components: List[str] = field(default_factory=list)

    # Trigger parameters (varies by condition type)
    trigger_params: Dict[str, Any] = field(default_factory=dict)

    # Experiment-specific configs (varies by type)
    resource_exhaustion_scenario: Optional[ResourceExhaustionScenario] = None
    network_instability_scenario: Optional[NetworkInstabilityScenario] = None
    cascade_scenario: Optional[CascadeScenario] = None

    # Monitoring and correlation
    collect_recovery_metrics: bool = True
    track_performance_degradation: bool = True
    baseline_duration_seconds: float = 30.0  # Collect baseline before experiment

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ChaosExperimentResult:
    """Results from a chaos experiment execution."""
    experiment_id: str
    experiment_type: ChaosExperimentType
    status: ExperimentStatus

    # Timing
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Baseline metrics (captured before experiment)
    baseline_metrics: Dict[str, float] = field(default_factory=dict)

    # Recovery metrics
    recovery_events: List[RecoveryEvent] = field(default_factory=list)
    mttr_analysis: Optional[MTTRAnalysis] = None
    mttf_analysis: Optional[MTTFAnalysis] = None

    # Performance degradation
    performance_degradations: List[Dict[str, Any]] = field(default_factory=list)
    max_degradation_percent: float = 0.0
    avg_degradation_percent: float = 0.0

    # Experiment-specific results
    resource_exhaustion_results: Dict[str, Any] = field(default_factory=dict)
    network_instability_results: Dict[str, Any] = field(default_factory=dict)
    cascade_failure_results: Dict[str, Any] = field(default_factory=dict)

    # Success criteria
    success: bool = True
    failure_reason: Optional[str] = None

    # Correlation data
    correlated_metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Metadata
    notes: str = ""


@dataclass
class ChaosOrchestrationConfig:
    """Configuration for chaos engineering orchestration."""
    # Experiment scheduling
    experiments: List[ChaosExperimentConfig] = field(default_factory=list)

    # Global settings
    max_concurrent_experiments: int = 1  # Safety limit
    auto_start_experiments: bool = True
    stop_on_critical_failure: bool = True

    # Monitoring
    performance_baseline_samples: int = 10
    recovery_timeout_seconds: float = 300.0

    # Result correlation
    correlation_window_seconds: float = 60.0

    # Reporting
    generate_detailed_reports: bool = True
    report_callback: Optional[Callable[[ChaosExperimentResult], None]] = None


class ChaosExperimentScheduler:
    """
    Schedules and manages chaos experiment execution.

    Handles experiment triggers, queuing, and execution coordination.
    """

    def __init__(self):
        """Initialize experiment scheduler."""
        self.pending_experiments: List[ChaosExperimentConfig] = []
        self.running_experiments: Dict[str, ChaosExperimentConfig] = {}
        self.completed_experiments: List[ChaosExperimentResult] = []
        self.logger = logging.getLogger(__name__)

    def schedule_experiment(self, config: ChaosExperimentConfig) -> None:
        """Schedule an experiment for execution.

        Args:
            config: Experiment configuration
        """
        self.pending_experiments.append(config)
        self.logger.info(f"Scheduled experiment: {config.experiment_id} ({config.experiment_type.value})")

    def get_next_experiment(
        self,
        current_metrics: Optional[Dict[str, float]] = None,
        max_concurrent: int = 1
    ) -> Optional[ChaosExperimentConfig]:
        """Get the next experiment that should be executed.

        Args:
            current_metrics: Current system metrics for threshold checking
            max_concurrent: Maximum number of concurrent experiments allowed

        Returns:
            Next experiment to execute, or None if none are ready
        """
        if len(self.running_experiments) >= max_concurrent:
            return None

        current_time = time.time()

        for exp in self.pending_experiments:
            # Check if experiment should start based on trigger condition
            if self._should_start_experiment(exp, current_time, current_metrics):
                self.pending_experiments.remove(exp)
                self.running_experiments[exp.experiment_id] = exp
                return exp

        return None

    def _should_start_experiment(
        self,
        exp: ChaosExperimentConfig,
        current_time: float,
        current_metrics: Optional[Dict[str, float]]
    ) -> bool:
        """Determine if an experiment should start.

        Args:
            exp: Experiment configuration
            current_time: Current timestamp
            current_metrics: Current system metrics

        Returns:
            True if experiment should start
        """
        if exp.trigger_condition == TriggerCondition.IMMEDIATE:
            return True

        elif exp.trigger_condition == TriggerCondition.SCHEDULED_TIME:
            scheduled_time = exp.trigger_params.get("scheduled_time", 0.0)
            return current_time >= scheduled_time

        elif exp.trigger_condition == TriggerCondition.PERIODIC:
            interval = exp.trigger_params.get("interval_seconds", 3600)
            last_run = exp.trigger_params.get("last_run_time", 0.0)
            return (current_time - last_run) >= interval

        elif exp.trigger_condition == TriggerCondition.LOAD_THRESHOLD:
            if not current_metrics:
                return False
            threshold = exp.trigger_params.get("load_threshold", 0.8)
            current_load = current_metrics.get("cpu_percent", 0.0) / 100.0
            return current_load >= threshold

        elif exp.trigger_condition == TriggerCondition.METRIC_THRESHOLD:
            if not current_metrics:
                return False
            metric_name = exp.trigger_params.get("metric_name", "")
            threshold = exp.trigger_params.get("threshold", 0.0)
            current_value = current_metrics.get(metric_name, 0.0)
            return current_value >= threshold

        return False

    def complete_experiment(self, result: ChaosExperimentResult) -> None:
        """Mark an experiment as completed.

        Args:
            result: Experiment result
        """
        if result.experiment_id in self.running_experiments:
            del self.running_experiments[result.experiment_id]

        self.completed_experiments.append(result)
        self.logger.info(
            f"Completed experiment: {result.experiment_id} "
            f"(status={result.status.value}, success={result.success})"
        )

    def cancel_all_experiments(self) -> None:
        """Cancel all pending experiments."""
        self.pending_experiments.clear()
        self.logger.info("Cancelled all pending experiments")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments.

        Returns:
            Dictionary with experiment counts and status
        """
        return {
            "pending": len(self.pending_experiments),
            "running": len(self.running_experiments),
            "completed": len(self.completed_experiments),
            "success_rate": self._calculate_success_rate()
        }

    def _calculate_success_rate(self) -> float:
        """Calculate success rate of completed experiments.

        Returns:
            Success rate (0.0-1.0)
        """
        if not self.completed_experiments:
            return 0.0

        successful = sum(1 for r in self.completed_experiments if r.success)
        return successful / len(self.completed_experiments)


class ChaosResultCorrelator:
    """
    Correlates chaos experiment results with system metrics.

    Analyzes the impact of chaos experiments on performance, recovery,
    and overall system behavior.
    """

    def __init__(self):
        """Initialize result correlator."""
        self.logger = logging.getLogger(__name__)

    def correlate_performance_impact(
        self,
        result: ChaosExperimentResult,
        perf_tracker: PerformanceDegradationTracker
    ) -> Dict[str, Any]:
        """Correlate experiment with performance degradation.

        Args:
            result: Experiment result
            perf_tracker: Performance degradation tracker

        Returns:
            Dictionary with correlation analysis
        """
        # Get degradations during experiment window
        degradations = perf_tracker.get_degradations(
            since_timestamp=result.start_time
        )

        # Filter to experiment duration
        if result.end_time:
            degradations = [
                d for d in degradations
                if d.timestamp <= result.end_time
            ]

        # Analyze degradations
        if degradations:
            degradation_percents = [d.degradation_percent for d in degradations]
            result.max_degradation_percent = max(degradation_percents)
            result.avg_degradation_percent = sum(degradation_percents) / len(degradation_percents)
            result.performance_degradations = [
                {
                    "metric_name": d.metric_name,
                    "degradation_percent": d.degradation_percent,
                    "severity": d.severity.value,
                    "timestamp": d.timestamp
                }
                for d in degradations
            ]

        return {
            "degradation_count": len(degradations),
            "max_degradation_percent": result.max_degradation_percent,
            "avg_degradation_percent": result.avg_degradation_percent,
            "critical_degradations": sum(
                1 for d in degradations if d.severity == AlertSeverity.CRITICAL
            )
        }

    def correlate_recovery_metrics(
        self,
        result: ChaosExperimentResult,
        recovery_tracker: RecoveryTimeTracker
    ) -> Dict[str, Any]:
        """Correlate experiment with recovery metrics.

        Args:
            result: Experiment result
            recovery_tracker: Recovery time tracker

        Returns:
            Dictionary with recovery correlation
        """
        # Get recovery events during experiment
        all_events = recovery_tracker.recovery_history
        experiment_events = [
            e for e in all_events
            if result.start_time <= e.failure_time <= (result.end_time or time.time())
        ]

        result.recovery_events = experiment_events

        # Calculate MTTR if we have recoveries
        if experiment_events:
            completed_recoveries = [
                e for e in experiment_events
                if e.recovery_complete_time is not None
            ]

            if completed_recoveries:
                # Calculate MTTR for all components (don't filter by component name)
                result.mttr_analysis = recovery_tracker.calculate_mttr(
                    component_name=None,  # Calculate for all components
                    lookback_count=len(completed_recoveries)
                )

        return {
            "failure_count": len(experiment_events),
            "recovery_count": sum(1 for e in experiment_events if e.recovery_complete_time),
            "avg_recovery_time": result.mttr_analysis.mean_recovery_time if result.mttr_analysis else 0.0
        }

    def generate_correlation_report(
        self,
        result: ChaosExperimentResult
    ) -> Dict[str, Any]:
        """Generate comprehensive correlation report.

        Args:
            result: Experiment result with correlated data

        Returns:
            Dictionary with full correlation analysis
        """
        return {
            "experiment_id": result.experiment_id,
            "experiment_type": result.experiment_type.value,
            "duration_seconds": result.duration_seconds,
            "success": result.success,

            "performance_impact": {
                "max_degradation_percent": result.max_degradation_percent,
                "avg_degradation_percent": result.avg_degradation_percent,
                "degradation_count": len(result.performance_degradations)
            },

            "recovery_metrics": {
                "failure_count": len(result.recovery_events),
                "mttr_seconds": result.mttr_analysis.mean_recovery_time if result.mttr_analysis else None,
                "p95_recovery_seconds": result.mttr_analysis.p95_recovery_time if result.mttr_analysis else None
            },

            "experiment_results": {
                "resource_exhaustion": result.resource_exhaustion_results,
                "network_instability": result.network_instability_results,
                "cascade_failures": result.cascade_failure_results
            }
        }


class ChaosOrchestrator:
    """
    Main chaos engineering orchestration coordinator.

    Integrates experiment scheduling, execution, result correlation,
    and reporting for comprehensive chaos engineering workflows.
    """

    def __init__(self, config: ChaosOrchestrationConfig):
        """Initialize chaos orchestrator.

        Args:
            config: Orchestration configuration
        """
        self.config = config
        self.scheduler = ChaosExperimentScheduler()
        self.correlator = ChaosResultCorrelator()

        # Simulators
        self.resource_simulator = ResourceExhaustionSimulator()
        self.network_simulator = NetworkInstabilitySimulator()
        self.cascade_simulator = CascadingFailureSimulator()

        # Monitoring and tracking
        self.recovery_tracker = RecoveryTimeTracker()
        self.perf_tracker = PerformanceDegradationTracker(
            alert_callback=self._handle_performance_alert
        )

        # State
        self.running = False
        self.current_experiment: Optional[ChaosExperimentConfig] = None
        self.logger = logging.getLogger(__name__)

    def add_experiment(self, config: ChaosExperimentConfig) -> None:
        """Add a chaos experiment to the schedule.

        Args:
            config: Experiment configuration
        """
        self.scheduler.schedule_experiment(config)

    async def run_experiments(self, duration_seconds: Optional[float] = None) -> List[ChaosExperimentResult]:
        """Run all scheduled experiments.

        Args:
            duration_seconds: Optional total duration limit

        Returns:
            List of experiment results
        """
        self.running = True
        start_time = time.time()

        self.logger.info(f"Starting chaos orchestration with {len(self.scheduler.pending_experiments)} experiments")

        try:
            while self.running:
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    self.logger.info("Duration limit reached, stopping orchestration")
                    break

                # Get next experiment to run
                experiment = self.scheduler.get_next_experiment(
                    current_metrics=self._get_current_metrics(),
                    max_concurrent=self.config.max_concurrent_experiments
                )

                if experiment:
                    # Run experiment
                    result = await self._execute_experiment(experiment)

                    # Correlate results
                    self._correlate_experiment_results(result)

                    # Complete experiment
                    self.scheduler.complete_experiment(result)

                    # Report if callback provided
                    if self.config.report_callback:
                        self.config.report_callback(result)

                    # Check for critical failures
                    if self.config.stop_on_critical_failure and not result.success:
                        self.logger.warning(f"Critical failure in experiment {result.experiment_id}, stopping")
                        break

                # Check if we have more work to do
                if not self.scheduler.pending_experiments and not self.scheduler.running_experiments:
                    self.logger.info("All experiments completed")
                    break

                # Brief pause between experiment checks
                await asyncio.sleep(1.0)

        finally:
            self.running = False

        return self.scheduler.completed_experiments

    async def _execute_experiment(self, config: ChaosExperimentConfig) -> ChaosExperimentResult:
        """Execute a single chaos experiment.

        Args:
            config: Experiment configuration

        Returns:
            Experiment result
        """
        result = ChaosExperimentResult(
            experiment_id=config.experiment_id,
            experiment_type=config.experiment_type,
            status=ExperimentStatus.RUNNING,
            start_time=time.time()
        )

        try:
            # Collect baseline metrics
            if config.baseline_duration_seconds > 0:
                await self._collect_baseline_metrics(config, result)

            # Execute experiment based on type
            if config.experiment_type == ChaosExperimentType.RESOURCE_EXHAUSTION:
                await self._execute_resource_exhaustion(config, result)

            elif config.experiment_type == ChaosExperimentType.NETWORK_INSTABILITY:
                await self._execute_network_instability(config, result)

            elif config.experiment_type == ChaosExperimentType.CASCADING_FAILURE:
                await self._execute_cascading_failure(config, result)

            elif config.experiment_type == ChaosExperimentType.COMBINED:
                await self._execute_combined_chaos(config, result)

            result.status = ExperimentStatus.COMPLETED
            result.success = True

        except Exception as e:
            self.logger.error(f"Experiment {config.experiment_id} failed: {e}")
            result.status = ExperimentStatus.FAILED
            result.success = False
            result.failure_reason = str(e)

        finally:
            result.end_time = time.time()
            result.duration_seconds = result.end_time - result.start_time

        return result

    async def _collect_baseline_metrics(
        self,
        config: ChaosExperimentConfig,
        result: ChaosExperimentResult
    ) -> None:
        """Collect baseline metrics before experiment.

        Args:
            config: Experiment configuration
            result: Result to populate with baseline
        """
        self.logger.info(f"Collecting baseline metrics for {config.experiment_id}")

        # Collect samples for baseline
        for i in range(self.config.performance_baseline_samples):
            metrics = self._get_current_metrics()
            for metric_name, value in metrics.items():
                if metric_name not in result.baseline_metrics:
                    result.baseline_metrics[metric_name] = []
                result.baseline_metrics[metric_name].append(value)

            await asyncio.sleep(config.baseline_duration_seconds / self.config.performance_baseline_samples)

        # Calculate baseline averages
        result.baseline_metrics = {
            name: sum(values) / len(values)
            for name, values in result.baseline_metrics.items()
        }

    async def _execute_resource_exhaustion(
        self,
        config: ChaosExperimentConfig,
        result: ChaosExperimentResult
    ) -> None:
        """Execute resource exhaustion experiment.

        Args:
            config: Experiment configuration
            result: Result to populate
        """
        if not config.resource_exhaustion_scenario:
            raise ValueError("Resource exhaustion scenario not provided")

        self.logger.info(f"Executing resource exhaustion: {config.experiment_id}")

        # Run resource exhaustion
        await self.resource_simulator.execute_scenario(
            config.resource_exhaustion_scenario
        )

        # Resource exhaustion doesn't return a result, just executes
        result.resource_exhaustion_results = {"executed": True}

    async def _execute_network_instability(
        self,
        config: ChaosExperimentConfig,
        result: ChaosExperimentResult
    ) -> None:
        """Execute network instability experiment.

        Args:
            config: Experiment configuration
            result: Result to populate
        """
        if not config.network_instability_scenario:
            raise ValueError("Network instability scenario not provided")

        self.logger.info(f"Executing network instability: {config.experiment_id}")

        # Run network instability
        await self.network_simulator.execute_scenario(
            config.network_instability_scenario
        )

        # Store execution results
        result.network_instability_results = {"executed": True}

    async def _execute_cascading_failure(
        self,
        config: ChaosExperimentConfig,
        result: ChaosExperimentResult
    ) -> None:
        """Execute cascading failure experiment.

        Args:
            config: Experiment configuration
            result: Result to populate
        """
        if not config.cascade_scenario:
            raise ValueError("Cascade scenario not provided")

        self.logger.info(f"Executing cascading failure: {config.experiment_id}")

        # Run cascading failure simulation
        cascade_result = await self.cascade_simulator.execute_scenario(
            config.cascade_scenario
        )

        result.cascade_failure_results = cascade_result

    async def _execute_combined_chaos(
        self,
        config: ChaosExperimentConfig,
        result: ChaosExperimentResult
    ) -> None:
        """Execute combined chaos experiment.

        Args:
            config: Experiment configuration
            result: Result to populate
        """
        self.logger.info(f"Executing combined chaos: {config.experiment_id}")

        # Run all configured chaos types concurrently
        tasks = []

        if config.resource_exhaustion_scenario:
            tasks.append(self._execute_resource_exhaustion(config, result))

        if config.network_instability_scenario:
            tasks.append(self._execute_network_instability(config, result))

        if config.cascade_scenario:
            tasks.append(self._execute_cascading_failure(config, result))

        # Execute all in parallel
        if tasks:
            await asyncio.gather(*tasks)

    def _correlate_experiment_results(self, result: ChaosExperimentResult) -> None:
        """Correlate experiment results with monitoring data.

        Args:
            result: Experiment result to correlate
        """
        # Correlate with performance degradation
        self.correlator.correlate_performance_impact(result, self.perf_tracker)

        # Correlate with recovery metrics
        self.correlator.correlate_recovery_metrics(result, self.recovery_tracker)

    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current system metrics.

        Returns:
            Dictionary of current metric values
        """
        # Placeholder - would integrate with actual monitoring
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_io_mb_s": 0.0
        }

    def _handle_performance_alert(self, degradation) -> None:
        """Handle performance degradation alert.

        Args:
            degradation: Performance degradation event
        """
        self.logger.warning(
            f"Performance degradation detected: {degradation.metric_name} "
            f"degraded by {degradation.degradation_percent:.1f}% "
            f"(severity={degradation.severity.value})"
        )

    def stop(self) -> None:
        """Stop chaos orchestration."""
        self.running = False
        self.scheduler.cancel_all_experiments()
        self.logger.info("Stopped chaos orchestration")

    def get_summary(self) -> Dict[str, Any]:
        """Get orchestration summary.

        Returns:
            Dictionary with orchestration state and results
        """
        scheduler_summary = self.scheduler.get_experiment_summary()

        return {
            "orchestrator_status": "running" if self.running else "stopped",
            "experiments": scheduler_summary,
            "current_experiment": self.current_experiment.experiment_id if self.current_experiment else None
        }
