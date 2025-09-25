"""
Central Test Orchestration System

This module provides comprehensive orchestration of all testing framework components,
coordinating discovery, execution, analytics, and integration testing in a unified
workflow with intelligent scheduling and resource optimization.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from threading import Lock, Event
import json
import sqlite3
from datetime import datetime, timedelta

from .discovery import TestDiscovery, TestCategory, TestComplexity, TestMetadata
from .execution import ParallelTestExecutor, ExecutionStrategy, ExecutionResult, ExecutionStatus
from .analytics import TestAnalytics, TestMetrics, SuiteMetrics, HealthStatus
from .integration import IntegrationTestCoordinator, ComponentConfig, IsolationLevel


class OrchestrationMode(Enum):
    """Test orchestration execution modes."""
    DISCOVERY_ONLY = "discovery_only"
    EXECUTION_ONLY = "execution_only"
    FULL_PIPELINE = "full_pipeline"
    ANALYSIS_ONLY = "analysis_only"
    INTEGRATION_ONLY = "integration_only"
    CUSTOM = "custom"


class OrchestrationPriority(Enum):
    """Test execution priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class PipelineStage(Enum):
    """Test pipeline stages."""
    INITIALIZATION = "initialization"
    DISCOVERY = "discovery"
    CATEGORIZATION = "categorization"
    DEPENDENCY_ANALYSIS = "dependency_analysis"
    RESOURCE_ALLOCATION = "resource_allocation"
    INTEGRATION_SETUP = "integration_setup"
    EXECUTION = "execution"
    ANALYTICS = "analytics"
    REPORTING = "reporting"
    CLEANUP = "cleanup"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OrchestrationConfig:
    """Configuration for test orchestration."""
    mode: OrchestrationMode = OrchestrationMode.FULL_PIPELINE
    priority: OrchestrationPriority = OrchestrationPriority.NORMAL
    max_workers: int = 4
    execution_strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL_SMART
    isolation_level: IsolationLevel = IsolationLevel.THREAD
    enable_analytics: bool = True
    enable_integration: bool = True
    enable_caching: bool = True
    timeout_seconds: float = 3600.0  # 1 hour default
    retry_failed_tests: bool = True
    generate_reports: bool = True
    cleanup_on_completion: bool = True
    custom_stages: List[str] = field(default_factory=list)
    stage_callbacks: Dict[str, List[Callable]] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Results from test orchestration execution."""
    orchestration_id: str
    start_time: float
    end_time: Optional[float] = None
    status: PipelineStage = PipelineStage.INITIALIZATION
    discovered_tests: Dict[str, TestMetadata] = field(default_factory=dict)
    execution_results: Dict[str, ExecutionResult] = field(default_factory=dict)
    suite_metrics: Optional[SuiteMetrics] = None
    integration_results: Dict[str, Any] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def duration(self) -> Optional[float]:
        """Get total orchestration duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if not self.execution_results:
            return 0.0

        successful = sum(1 for result in self.execution_results.values()
                        if result.status == ExecutionStatus.COMPLETED)
        return successful / len(self.execution_results)


class TestOrchestrator:
    """
    Central orchestrator for comprehensive test execution workflows.

    Coordinates all testing framework components to provide unified,
    intelligent test execution with advanced scheduling and optimization.
    """

    def __init__(self,
                 project_root: Path,
                 test_directory: Path,
                 config: Optional[OrchestrationConfig] = None,
                 database_path: Optional[Path] = None):
        """Initialize the test orchestrator.

        Args:
            project_root: Root directory of the project
            test_directory: Directory containing test files
            config: Orchestration configuration
            database_path: Path to orchestration database
        """
        self.project_root = Path(project_root)
        self.test_directory = Path(test_directory)
        self.config = config or OrchestrationConfig()
        self.database_path = database_path or self.project_root / ".test_orchestration.db"

        # Core components
        self._discovery: Optional[TestDiscovery] = None
        self._executor: Optional[ParallelTestExecutor] = None
        self._analytics: Optional[TestAnalytics] = None
        self._integration: Optional[IntegrationTestCoordinator] = None

        # Orchestration state
        self._current_orchestration: Optional[str] = None
        self._pipeline_lock = Lock()
        self._stop_event = Event()
        self._stage_callbacks: Dict[str, List[Callable]] = {}

        # Initialize database
        self._init_database()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize orchestration database."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orchestration_runs (
                    id TEXT PRIMARY KEY,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    status TEXT NOT NULL,
                    config TEXT NOT NULL,
                    results TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS stage_timings (
                    orchestration_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    duration REAL,
                    status TEXT,
                    FOREIGN KEY (orchestration_id) REFERENCES orchestration_runs (id)
                )
            """)

    def register_stage_callback(self, stage: str, callback: Callable):
        """Register callback for pipeline stage."""
        if stage not in self._stage_callbacks:
            self._stage_callbacks[stage] = []
        self._stage_callbacks[stage].append(callback)

    def _execute_stage_callbacks(self, stage: str, context: Dict[str, Any]):
        """Execute callbacks for pipeline stage."""
        if stage in self._stage_callbacks:
            for callback in self._stage_callbacks[stage]:
                try:
                    callback(stage, context)
                except Exception as e:
                    self.logger.warning(f"Stage callback failed for {stage}: {e}")

    def _get_component_discovery(self) -> TestDiscovery:
        """Get or create discovery component."""
        if not self._discovery:
            self._discovery = TestDiscovery(
                project_root=self.project_root,
                test_directory=self.test_directory
            )
        return self._discovery

    def _get_component_executor(self) -> ParallelTestExecutor:
        """Get or create execution component."""
        if not self._executor:
            self._executor = ParallelTestExecutor(
                max_workers=self.config.max_workers,
                strategy=self.config.execution_strategy,
                retry_failed=self.config.retry_failed_tests
            )
        return self._executor

    def _get_component_analytics(self) -> TestAnalytics:
        """Get or create analytics component."""
        if not self._analytics:
            analytics_db = self.project_root / ".test_analytics.db"
            self._analytics = TestAnalytics(database_path=analytics_db)
        return self._analytics

    def _get_component_integration(self) -> IntegrationTestCoordinator:
        """Get or create integration component."""
        if not self._integration:
            self._integration = IntegrationTestCoordinator(
                isolation_level=self.config.isolation_level
            )
        return self._integration

    async def orchestrate_tests(self,
                               components: Optional[List[ComponentConfig]] = None,
                               filters: Optional[Dict[str, Any]] = None) -> OrchestrationResult:
        """
        Execute comprehensive test orchestration pipeline.

        Args:
            components: Integration test components to orchestrate
            filters: Test filtering criteria

        Returns:
            Complete orchestration results
        """
        orchestration_id = f"orch_{int(time.time() * 1000)}"
        self._current_orchestration = orchestration_id

        result = OrchestrationResult(
            orchestration_id=orchestration_id,
            start_time=time.time()
        )

        try:
            with self._pipeline_lock:
                # Execute pipeline stages
                await self._execute_pipeline(result, components, filters)

        except Exception as e:
            result.status = PipelineStage.FAILED
            result.errors.append(f"Orchestration failed: {str(e)}")
            self.logger.error(f"Orchestration {orchestration_id} failed: {e}")

        finally:
            result.end_time = time.time()
            self._save_orchestration_result(result)
            self._current_orchestration = None

        return result

    async def _execute_pipeline(self,
                               result: OrchestrationResult,
                               components: Optional[List[ComponentConfig]],
                               filters: Optional[Dict[str, Any]]):
        """Execute the complete orchestration pipeline."""

        stages = self._get_pipeline_stages()

        for stage in stages:
            if self._stop_event.is_set():
                result.status = PipelineStage.FAILED
                result.errors.append("Orchestration stopped by user request")
                return

            stage_start = time.time()
            result.status = stage

            try:
                await self._execute_pipeline_stage(stage, result, components, filters)
                result.stage_timings[stage.value] = time.time() - stage_start

            except Exception as e:
                result.status = PipelineStage.FAILED
                result.errors.append(f"Stage {stage.value} failed: {str(e)}")
                self.logger.error(f"Pipeline stage {stage.value} failed: {e}")
                return

        result.status = PipelineStage.COMPLETED

    def _get_pipeline_stages(self) -> List[PipelineStage]:
        """Get pipeline stages based on configuration."""
        if self.config.mode == OrchestrationMode.DISCOVERY_ONLY:
            return [
                PipelineStage.INITIALIZATION,
                PipelineStage.DISCOVERY,
                PipelineStage.CATEGORIZATION,
                PipelineStage.REPORTING
            ]
        elif self.config.mode == OrchestrationMode.EXECUTION_ONLY:
            return [
                PipelineStage.INITIALIZATION,
                PipelineStage.RESOURCE_ALLOCATION,
                PipelineStage.EXECUTION,
                PipelineStage.REPORTING
            ]
        elif self.config.mode == OrchestrationMode.ANALYSIS_ONLY:
            return [
                PipelineStage.INITIALIZATION,
                PipelineStage.ANALYTICS,
                PipelineStage.REPORTING
            ]
        elif self.config.mode == OrchestrationMode.INTEGRATION_ONLY:
            return [
                PipelineStage.INITIALIZATION,
                PipelineStage.INTEGRATION_SETUP,
                PipelineStage.EXECUTION,
                PipelineStage.CLEANUP
            ]
        else:  # FULL_PIPELINE
            return [
                PipelineStage.INITIALIZATION,
                PipelineStage.DISCOVERY,
                PipelineStage.CATEGORIZATION,
                PipelineStage.DEPENDENCY_ANALYSIS,
                PipelineStage.RESOURCE_ALLOCATION,
                PipelineStage.INTEGRATION_SETUP,
                PipelineStage.EXECUTION,
                PipelineStage.ANALYTICS,
                PipelineStage.REPORTING,
                PipelineStage.CLEANUP
            ]

    async def _execute_pipeline_stage(self,
                                     stage: PipelineStage,
                                     result: OrchestrationResult,
                                     components: Optional[List[ComponentConfig]],
                                     filters: Optional[Dict[str, Any]]):
        """Execute specific pipeline stage."""

        context = {
            'result': result,
            'components': components,
            'filters': filters
        }

        # Execute pre-stage callbacks
        self._execute_stage_callbacks(f"pre_{stage.value}", context)

        if stage == PipelineStage.INITIALIZATION:
            await self._stage_initialization(result)

        elif stage == PipelineStage.DISCOVERY:
            await self._stage_discovery(result, filters)

        elif stage == PipelineStage.CATEGORIZATION:
            await self._stage_categorization(result)

        elif stage == PipelineStage.DEPENDENCY_ANALYSIS:
            await self._stage_dependency_analysis(result)

        elif stage == PipelineStage.RESOURCE_ALLOCATION:
            await self._stage_resource_allocation(result)

        elif stage == PipelineStage.INTEGRATION_SETUP:
            await self._stage_integration_setup(result, components)

        elif stage == PipelineStage.EXECUTION:
            await self._stage_execution(result)

        elif stage == PipelineStage.ANALYTICS:
            await self._stage_analytics(result)

        elif stage == PipelineStage.REPORTING:
            await self._stage_reporting(result)

        elif stage == PipelineStage.CLEANUP:
            await self._stage_cleanup(result)

        # Execute post-stage callbacks
        self._execute_stage_callbacks(f"post_{stage.value}", context)

    async def _stage_initialization(self, result: OrchestrationResult):
        """Initialize orchestration components."""
        self.logger.info(f"Initializing orchestration {result.orchestration_id}")

        # Initialize components based on configuration
        if self.config.mode in [OrchestrationMode.FULL_PIPELINE, OrchestrationMode.DISCOVERY_ONLY]:
            self._get_component_discovery()

        if self.config.mode in [OrchestrationMode.FULL_PIPELINE, OrchestrationMode.EXECUTION_ONLY]:
            self._get_component_executor()

        if self.config.enable_analytics:
            self._get_component_analytics()

        if self.config.enable_integration:
            self._get_component_integration()

    async def _stage_discovery(self, result: OrchestrationResult, filters: Optional[Dict[str, Any]]):
        """Execute test discovery stage."""
        discovery = self._get_component_discovery()

        self.logger.info("Discovering tests...")
        discovered_tests = discovery.discover_tests(
            parallel=True,
            filters=filters or {}
        )

        result.discovered_tests = discovered_tests
        self.logger.info(f"Discovered {len(discovered_tests)} tests")

    async def _stage_categorization(self, result: OrchestrationResult):
        """Execute test categorization stage."""
        if not result.discovered_tests:
            return

        self.logger.info("Categorizing tests...")

        # Analyze test categories and complexities
        categories = {}
        complexities = {}

        for test_name, metadata in result.discovered_tests.items():
            categories[metadata.category] = categories.get(metadata.category, 0) + 1
            complexities[metadata.complexity] = complexities.get(metadata.complexity, 0) + 1

        self.logger.info(f"Test categories: {dict(categories)}")
        self.logger.info(f"Test complexities: {dict(complexities)}")

    async def _stage_dependency_analysis(self, result: OrchestrationResult):
        """Analyze test dependencies."""
        if not result.discovered_tests:
            return

        self.logger.info("Analyzing test dependencies...")

        # Analyze resource requirements and potential conflicts
        resource_conflicts = set()
        database_tests = []
        network_tests = []

        for test_name, metadata in result.discovered_tests.items():
            if hasattr(metadata, 'resource_requirements'):
                for req in metadata.resource_requirements:
                    if req.resource_type == "database":
                        database_tests.append(test_name)
                    elif req.resource_type == "network":
                        network_tests.append(test_name)

        if database_tests:
            self.logger.info(f"Found {len(database_tests)} database-dependent tests")
        if network_tests:
            self.logger.info(f"Found {len(network_tests)} network-dependent tests")

    async def _stage_resource_allocation(self, result: OrchestrationResult):
        """Allocate execution resources."""
        executor = self._get_component_executor()

        if result.discovered_tests:
            self.logger.info("Allocating execution resources...")

            # Create execution plan
            execution_plan = executor.create_execution_plan(result.discovered_tests)

            self.logger.info(f"Created execution plan with {len(execution_plan.test_batches)} batches")
            self.logger.info(f"Estimated execution duration: {execution_plan.estimated_duration:.2f}s")

    async def _stage_integration_setup(self,
                                      result: OrchestrationResult,
                                      components: Optional[List[ComponentConfig]]):
        """Setup integration test environment."""
        if not self.config.enable_integration or not components:
            return

        integration = self._get_component_integration()

        self.logger.info("Setting up integration test environment...")

        # Register components
        integration.register_components(components)

        # Start components
        for component in components:
            try:
                await integration.start_component(component.name)
                result.integration_results[component.name] = {"status": "started"}
            except Exception as e:
                result.warnings.append(f"Failed to start component {component.name}: {e}")
                result.integration_results[component.name] = {"status": "failed", "error": str(e)}

    async def _stage_execution(self, result: OrchestrationResult):
        """Execute tests."""
        if not result.discovered_tests:
            self.logger.warning("No tests to execute")
            return

        executor = self._get_component_executor()

        self.logger.info(f"Executing {len(result.discovered_tests)} tests...")

        # Execute tests
        execution_results = await executor.execute_tests(result.discovered_tests)
        result.execution_results = execution_results

        # Log execution summary
        successful = sum(1 for r in execution_results.values() if r.status == ExecutionStatus.COMPLETED)
        failed = len(execution_results) - successful

        self.logger.info(f"Test execution completed: {successful} passed, {failed} failed")

    async def _stage_analytics(self, result: OrchestrationResult):
        """Process test analytics."""
        if not self.config.enable_analytics or not result.execution_results:
            return

        analytics = self._get_component_analytics()

        self.logger.info("Processing test analytics...")

        # Process results
        suite_metrics = analytics.process_execution_results(
            result.execution_results,
            result.discovered_tests
        )

        result.suite_metrics = suite_metrics

        self.logger.info(f"Analytics complete: {suite_metrics.overall_success_rate:.1%} success rate")
        self.logger.info(f"Suite health status: {suite_metrics.health_status.name}")

    async def _stage_reporting(self, result: OrchestrationResult):
        """Generate test reports."""
        if not self.config.generate_reports:
            return

        self.logger.info("Generating test reports...")

        # Generate summary report
        report_data = {
            "orchestration_id": result.orchestration_id,
            "duration": result.duration,
            "discovered_tests": len(result.discovered_tests),
            "executed_tests": len(result.execution_results),
            "success_rate": result.success_rate,
            "stage_timings": result.stage_timings
        }

        if result.suite_metrics:
            report_data.update({
                "health_status": result.suite_metrics.health_status.name,
                "flaky_tests": result.suite_metrics.flaky_test_count,
                "total_duration": result.suite_metrics.total_duration
            })

        # Save report
        report_path = self.project_root / f"test_report_{result.orchestration_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Test report saved to {report_path}")

    async def _stage_cleanup(self, result: OrchestrationResult):
        """Cleanup orchestration resources."""
        if not self.config.cleanup_on_completion:
            return

        self.logger.info("Cleaning up orchestration resources...")

        # Cleanup integration components
        if self._integration:
            await self._integration.cleanup_all()

        # Close component connections
        if self._discovery:
            self._discovery.close()

        if self._analytics:
            self._analytics.close()

    def _save_orchestration_result(self, result: OrchestrationResult):
        """Save orchestration result to database."""
        with sqlite3.connect(self.database_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO orchestration_runs
                (id, start_time, end_time, status, config, results)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result.orchestration_id,
                result.start_time,
                result.end_time,
                result.status.value,
                json.dumps(self.config.__dict__, default=str),
                json.dumps({
                    'discovered_tests': len(result.discovered_tests),
                    'execution_results': len(result.execution_results),
                    'success_rate': result.success_rate,
                    'errors': result.errors,
                    'warnings': result.warnings
                })
            ))

            # Save stage timings
            for stage, duration in result.stage_timings.items():
                conn.execute("""
                    INSERT INTO stage_timings
                    (orchestration_id, stage, start_time, end_time, duration, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    result.orchestration_id,
                    stage,
                    result.start_time,
                    result.start_time + duration,
                    duration,
                    "completed"
                ))

    def stop_orchestration(self):
        """Stop current orchestration."""
        self._stop_event.set()
        self.logger.info("Orchestration stop requested")

    def get_orchestration_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get orchestration execution history."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT id, start_time, end_time, status, results
                FROM orchestration_runs
                ORDER BY start_time DESC
                LIMIT ?
            """, (limit,))

            history = []
            for row in cursor.fetchall():
                history.append({
                    'id': row[0],
                    'start_time': row[1],
                    'end_time': row[2],
                    'status': row[3],
                    'results': json.loads(row[4]) if row[4] else {}
                })

            return history

    def get_stage_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics by pipeline stage."""
        with sqlite3.connect(self.database_path) as conn:
            cursor = conn.execute("""
                SELECT stage, AVG(duration) as avg_duration,
                       MIN(duration) as min_duration, MAX(duration) as max_duration,
                       COUNT(*) as execution_count
                FROM stage_timings
                WHERE status = 'completed'
                GROUP BY stage
            """)

            stats = {}
            for row in cursor.fetchall():
                stats[row[0]] = {
                    'avg_duration': row[1],
                    'min_duration': row[2],
                    'max_duration': row[3],
                    'execution_count': row[4]
                }

            return stats

    def close(self):
        """Close orchestrator and cleanup resources."""
        self.stop_orchestration()

        # Close components
        if self._discovery:
            self._discovery.close()
        if self._analytics:
            self._analytics.close()

        self.logger.info("Test orchestrator closed")


class OrchestrationScheduler:
    """
    Advanced scheduler for automated test orchestration.

    Provides cron-like scheduling, priority queues, and intelligent
    test execution timing based on resource availability.
    """

    def __init__(self, orchestrator: TestOrchestrator):
        """Initialize orchestration scheduler.

        Args:
            orchestrator: Test orchestrator instance
        """
        self.orchestrator = orchestrator
        self.scheduled_runs: Dict[str, Dict[str, Any]] = {}
        self.running_orchestrations: Set[str] = set()
        self.scheduler_executor = ThreadPoolExecutor(max_workers=2)
        self.logger = logging.getLogger(__name__)

    def schedule_orchestration(self,
                              schedule_id: str,
                              cron_expression: str,
                              config: OrchestrationConfig,
                              components: Optional[List[ComponentConfig]] = None,
                              filters: Optional[Dict[str, Any]] = None):
        """Schedule recurring test orchestration.

        Args:
            schedule_id: Unique identifier for scheduled run
            cron_expression: Cron-like scheduling expression
            config: Orchestration configuration
            components: Integration test components
            filters: Test filtering criteria
        """
        self.scheduled_runs[schedule_id] = {
            'cron_expression': cron_expression,
            'config': config,
            'components': components,
            'filters': filters,
            'last_run': None,
            'next_run': self._calculate_next_run(cron_expression)
        }

        self.logger.info(f"Scheduled orchestration '{schedule_id}' with cron '{cron_expression}'")

    def _calculate_next_run(self, cron_expression: str) -> datetime:
        """Calculate next run time from cron expression."""
        # Simplified cron parsing - in production, use croniter or similar
        # For demo, schedule every hour
        return datetime.now() + timedelta(hours=1)

    async def run_scheduled_orchestrations(self):
        """Execute scheduled orchestrations."""
        current_time = datetime.now()

        for schedule_id, schedule_info in self.scheduled_runs.items():
            if (schedule_info['next_run'] <= current_time and
                schedule_id not in self.running_orchestrations):

                self.logger.info(f"Executing scheduled orchestration '{schedule_id}'")

                # Mark as running
                self.running_orchestrations.add(schedule_id)

                try:
                    # Execute orchestration
                    result = await self.orchestrator.orchestrate_tests(
                        components=schedule_info['components'],
                        filters=schedule_info['filters']
                    )

                    # Update schedule
                    schedule_info['last_run'] = current_time
                    schedule_info['next_run'] = self._calculate_next_run(
                        schedule_info['cron_expression']
                    )

                    self.logger.info(f"Scheduled orchestration '{schedule_id}' completed with status {result.status.value}")

                finally:
                    self.running_orchestrations.remove(schedule_id)

    def cancel_scheduled_orchestration(self, schedule_id: str):
        """Cancel scheduled orchestration."""
        if schedule_id in self.scheduled_runs:
            del self.scheduled_runs[schedule_id]
            self.logger.info(f"Cancelled scheduled orchestration '{schedule_id}'")

    def get_schedule_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all scheduled orchestrations."""
        status = {}
        for schedule_id, schedule_info in self.scheduled_runs.items():
            status[schedule_id] = {
                'cron_expression': schedule_info['cron_expression'],
                'last_run': schedule_info['last_run'],
                'next_run': schedule_info['next_run'],
                'is_running': schedule_id in self.running_orchestrations
            }
        return status

    def close(self):
        """Close scheduler and cleanup resources."""
        self.scheduler_executor.shutdown(wait=True)
        self.logger.info("Orchestration scheduler closed")