"""
Comprehensive Tests for Test Orchestration Framework

This module provides extensive testing for the central orchestration system,
covering all pipeline stages, error conditions, and integration scenarios.
"""

import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import sqlite3
import json

# Import the framework components
import sys
src_path = Path(__file__).parent / "src" / "python"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

framework_path = Path(__file__).parent / "tests" / "framework"
if str(framework_path) not in sys.path:
    sys.path.insert(0, str(framework_path))

from tests.framework.orchestration import (
    TestOrchestrator, OrchestrationConfig, OrchestrationMode,
    OrchestrationPriority, PipelineStage, OrchestrationResult,
    OrchestrationScheduler
)
from tests.framework.discovery import TestDiscovery, TestCategory, TestComplexity, TestMetadata
from tests.framework.execution import ParallelTestExecutor, ExecutionStrategy, ExecutionResult, ExecutionStatus
from tests.framework.analytics import TestAnalytics, TestMetrics, SuiteMetrics, HealthStatus
from tests.framework.integration import IntegrationTestCoordinator, ComponentConfig, ComponentType, IsolationLevel


class TestOrchestrationConfig:
    """Test orchestration configuration."""

    def test_default_configuration(self):
        """Test default orchestration configuration."""
        config = OrchestrationConfig()

        assert config.mode == OrchestrationMode.FULL_PIPELINE
        assert config.priority == OrchestrationPriority.NORMAL
        assert config.max_workers == 4
        assert config.execution_strategy == ExecutionStrategy.PARALLEL_SMART
        assert config.isolation_level == IsolationLevel.PROCESS
        assert config.enable_analytics is True
        assert config.enable_integration is True
        assert config.timeout_seconds == 3600.0

    def test_custom_configuration(self):
        """Test custom orchestration configuration."""
        config = OrchestrationConfig(
            mode=OrchestrationMode.EXECUTION_ONLY,
            priority=OrchestrationPriority.HIGH,
            max_workers=8,
            execution_strategy=ExecutionStrategy.PARALLEL_AGGRESSIVE,
            timeout_seconds=7200.0
        )

        assert config.mode == OrchestrationMode.EXECUTION_ONLY
        assert config.priority == OrchestrationPriority.HIGH
        assert config.max_workers == 8
        assert config.execution_strategy == ExecutionStrategy.PARALLEL_AGGRESSIVE
        assert config.timeout_seconds == 7200.0

    def test_stage_callbacks_initialization(self):
        """Test stage callbacks initialization."""
        config = OrchestrationConfig()
        assert isinstance(config.stage_callbacks, dict)
        assert len(config.stage_callbacks) == 0
        assert isinstance(config.custom_stages, list)


class TestOrchestrationResult:
    """Test orchestration result handling."""

    def test_result_creation(self):
        """Test orchestration result creation."""
        result = OrchestrationResult(
            orchestration_id="test_orch_123",
            start_time=time.time()
        )

        assert result.orchestration_id == "test_orch_123"
        assert result.status == PipelineStage.INITIALIZATION
        assert isinstance(result.discovered_tests, dict)
        assert isinstance(result.execution_results, dict)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)

    def test_duration_calculation(self):
        """Test duration calculation."""
        start_time = time.time()
        result = OrchestrationResult(
            orchestration_id="test_duration",
            start_time=start_time
        )

        # No end time set
        assert result.duration is None

        # Set end time
        result.end_time = start_time + 10.5
        assert result.duration == 10.5

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = OrchestrationResult(
            orchestration_id="test_success_rate",
            start_time=time.time()
        )

        # No execution results
        assert result.success_rate == 0.0

        # Add execution results
        result.execution_results = {
            "test1": ExecutionResult(
                test_name="test1",
                status=ExecutionStatus.COMPLETED,
                duration=1.0,
                start_time=time.time(),
                end_time=time.time() + 1.0
            ),
            "test2": ExecutionResult(
                test_name="test2",
                status=ExecutionStatus.FAILED,
                duration=0.5,
                start_time=time.time(),
                end_time=time.time() + 0.5
            ),
            "test3": ExecutionResult(
                test_name="test3",
                status=ExecutionStatus.COMPLETED,
                duration=2.0,
                start_time=time.time(),
                end_time=time.time() + 2.0
            )
        }

        # 2 out of 3 successful = 66.67%
        assert abs(result.success_rate - (2/3)) < 0.001

    def test_success_rate_all_successful(self):
        """Test success rate with all successful tests."""
        result = OrchestrationResult(
            orchestration_id="test_all_success",
            start_time=time.time()
        )

        result.execution_results = {
            "test1": ExecutionResult(
                test_name="test1",
                status=ExecutionStatus.COMPLETED,
                duration=1.0,
                start_time=time.time(),
                end_time=time.time() + 1.0
            ),
            "test2": ExecutionResult(
                test_name="test2",
                status=ExecutionStatus.COMPLETED,
                duration=1.5,
                start_time=time.time(),
                end_time=time.time() + 1.5
            )
        }

        assert result.success_rate == 1.0


class TestTestOrchestrator:
    """Test the main orchestration system."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            test_dir = project_root / "tests"
            test_dir.mkdir()

            # Create sample test file
            test_file = test_dir / "test_sample.py"
            test_file.write_text('''
import pytest

def test_simple():
    assert True

@pytest.mark.asyncio
async def test_async():
    assert True
''')

            yield project_root, test_dir

    @pytest.fixture
    def orchestrator(self, temp_project):
        """Create test orchestrator."""
        project_root, test_dir = temp_project
        return TestOrchestrator(project_root, test_dir)

    def test_orchestrator_initialization(self, temp_project):
        """Test orchestrator initialization."""
        project_root, test_dir = temp_project

        orchestrator = TestOrchestrator(project_root, test_dir)

        assert orchestrator.project_root == project_root
        assert orchestrator.test_directory == test_dir
        assert isinstance(orchestrator.config, OrchestrationConfig)
        assert orchestrator._current_orchestration is None

    def test_orchestrator_with_custom_config(self, temp_project):
        """Test orchestrator with custom configuration."""
        project_root, test_dir = temp_project

        config = OrchestrationConfig(
            mode=OrchestrationMode.DISCOVERY_ONLY,
            max_workers=8
        )

        orchestrator = TestOrchestrator(project_root, test_dir, config)

        assert orchestrator.config.mode == OrchestrationMode.DISCOVERY_ONLY
        assert orchestrator.config.max_workers == 8

    def test_database_initialization(self, orchestrator):
        """Test database initialization."""
        # Check that database file exists
        assert orchestrator.database_path.exists()

        # Check that tables were created
        with sqlite3.connect(orchestrator.database_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('orchestration_runs', 'stage_timings')
            """)
            tables = [row[0] for row in cursor.fetchall()]

            assert 'orchestration_runs' in tables
            assert 'stage_timings' in tables

    def test_stage_callback_registration(self, orchestrator):
        """Test stage callback registration."""
        callback_executed = False

        def test_callback(stage, context):
            nonlocal callback_executed
            callback_executed = True
            assert stage == "pre_discovery"
            assert "result" in context

        orchestrator.register_stage_callback("pre_discovery", test_callback)

        # Execute callback
        orchestrator._execute_stage_callbacks("pre_discovery", {"result": Mock()})

        assert callback_executed

    def test_stage_callback_error_handling(self, orchestrator):
        """Test stage callback error handling."""
        def failing_callback(stage, context):
            raise Exception("Callback failed")

        orchestrator.register_stage_callback("test_stage", failing_callback)

        # Should not raise exception
        orchestrator._execute_stage_callbacks("test_stage", {})

    def test_get_pipeline_stages_full_pipeline(self, orchestrator):
        """Test pipeline stages for full pipeline mode."""
        orchestrator.config.mode = OrchestrationMode.FULL_PIPELINE

        stages = orchestrator._get_pipeline_stages()

        expected_stages = [
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

        assert stages == expected_stages

    def test_get_pipeline_stages_discovery_only(self, orchestrator):
        """Test pipeline stages for discovery only mode."""
        orchestrator.config.mode = OrchestrationMode.DISCOVERY_ONLY

        stages = orchestrator._get_pipeline_stages()

        expected_stages = [
            PipelineStage.INITIALIZATION,
            PipelineStage.DISCOVERY,
            PipelineStage.CATEGORIZATION,
            PipelineStage.REPORTING
        ]

        assert stages == expected_stages

    def test_get_pipeline_stages_execution_only(self, orchestrator):
        """Test pipeline stages for execution only mode."""
        orchestrator.config.mode = OrchestrationMode.EXECUTION_ONLY

        stages = orchestrator._get_pipeline_stages()

        expected_stages = [
            PipelineStage.INITIALIZATION,
            PipelineStage.RESOURCE_ALLOCATION,
            PipelineStage.EXECUTION,
            PipelineStage.REPORTING
        ]

        assert stages == expected_stages

    def test_component_creation_lazy_loading(self, orchestrator):
        """Test lazy loading of components."""
        # Initially, components should be None
        assert orchestrator._discovery is None
        assert orchestrator._executor is None
        assert orchestrator._analytics is None
        assert orchestrator._integration is None

        # Get components - should create them
        discovery = orchestrator._get_component_discovery()
        executor = orchestrator._get_component_executor()
        analytics = orchestrator._get_component_analytics()
        integration = orchestrator._get_component_integration()

        assert discovery is not None
        assert executor is not None
        assert analytics is not None
        assert integration is not None

        # Getting again should return same instances
        assert orchestrator._get_component_discovery() is discovery
        assert orchestrator._get_component_executor() is executor

    @pytest.mark.asyncio
    async def test_orchestrate_tests_basic(self, orchestrator):
        """Test basic test orchestration."""
        with patch.object(orchestrator, '_execute_pipeline') as mock_execute:
            result = await orchestrator.orchestrate_tests()

            assert result.orchestration_id is not None
            assert result.start_time > 0
            assert result.end_time is not None
            assert result.duration is not None
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrate_tests_with_components(self, orchestrator):
        """Test orchestration with integration components."""
        components = [
            ComponentConfig(
                name="test_service",
                component_type=ComponentType.PYTHON_SERVICE,
                start_command=["echo", "test"],
                health_check_command=["echo", "healthy"],
                startup_timeout=5.0,
                ports=[8000]
            )
        ]

        with patch.object(orchestrator, '_execute_pipeline') as mock_execute:
            result = await orchestrator.orchestrate_tests(components=components)

            mock_execute.assert_called_once()
            args, kwargs = mock_execute.call_args
            assert args[1] == components  # components argument

    @pytest.mark.asyncio
    async def test_orchestrate_tests_error_handling(self, orchestrator):
        """Test orchestration error handling."""
        with patch.object(orchestrator, '_execute_pipeline') as mock_execute:
            mock_execute.side_effect = Exception("Pipeline failed")

            result = await orchestrator.orchestrate_tests()

            assert result.status == PipelineStage.FAILED
            assert len(result.errors) > 0
            assert "Orchestration failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_stage_initialization(self, orchestrator):
        """Test initialization stage."""
        result = OrchestrationResult(
            orchestration_id="test_init",
            start_time=time.time()
        )

        await orchestrator._stage_initialization(result)

        # Components should be created based on mode
        if orchestrator.config.mode == OrchestrationMode.FULL_PIPELINE:
            assert orchestrator._discovery is not None
            assert orchestrator._executor is not None
        if orchestrator.config.enable_analytics:
            assert orchestrator._analytics is not None
        if orchestrator.config.enable_integration:
            assert orchestrator._integration is not None

    @pytest.mark.asyncio
    async def test_stage_discovery(self, orchestrator):
        """Test discovery stage."""
        result = OrchestrationResult(
            orchestration_id="test_discovery",
            start_time=time.time()
        )

        # Mock discovery component
        mock_discovery = Mock()
        mock_discovery.discover_tests.return_value = {
            "test_sample": TestMetadata(
                name="test_sample",
                category=TestCategory.UNIT,
                complexity=TestComplexity.LOW,
                estimated_duration=1.0
            )
        }

        with patch.object(orchestrator, '_get_component_discovery', return_value=mock_discovery):
            await orchestrator._stage_discovery(result, None)

            assert len(result.discovered_tests) == 1
            assert "test_sample" in result.discovered_tests
            mock_discovery.discover_tests.assert_called_once_with(
                parallel=True,
                filters={}
            )

    @pytest.mark.asyncio
    async def test_stage_categorization(self, orchestrator):
        """Test categorization stage."""
        result = OrchestrationResult(
            orchestration_id="test_categorization",
            start_time=time.time()
        )

        # Add some discovered tests
        result.discovered_tests = {
            "test_unit": TestMetadata(
                name="test_unit",
                category=TestCategory.UNIT,
                complexity=TestComplexity.LOW,
                estimated_duration=1.0
            ),
            "test_integration": TestMetadata(
                name="test_integration",
                category=TestCategory.INTEGRATION,
                complexity=TestComplexity.HIGH,
                estimated_duration=5.0
            )
        }

        await orchestrator._stage_categorization(result)

        # Should complete without error

    @pytest.mark.asyncio
    async def test_stage_execution(self, orchestrator):
        """Test execution stage."""
        result = OrchestrationResult(
            orchestration_id="test_execution",
            start_time=time.time()
        )

        # Add discovered tests
        result.discovered_tests = {
            "test1": TestMetadata(
                name="test1",
                category=TestCategory.UNIT,
                complexity=TestComplexity.LOW,
                estimated_duration=1.0
            )
        }

        # Mock executor
        mock_executor = Mock()
        mock_execution_result = {
            "test1": ExecutionResult(
                test_name="test1",
                status=ExecutionStatus.COMPLETED,
                duration=1.0,
                start_time=time.time(),
                end_time=time.time() + 1.0
            )
        }
        mock_executor.execute_tests.return_value = mock_execution_result

        with patch.object(orchestrator, '_get_component_executor', return_value=mock_executor):
            await orchestrator._stage_execution(result)

            assert len(result.execution_results) == 1
            assert "test1" in result.execution_results
            assert result.execution_results["test1"].status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_stage_execution_no_tests(self, orchestrator):
        """Test execution stage with no tests."""
        result = OrchestrationResult(
            orchestration_id="test_no_tests",
            start_time=time.time()
        )

        await orchestrator._stage_execution(result)

        # Should complete without error
        assert len(result.execution_results) == 0

    @pytest.mark.asyncio
    async def test_stage_analytics(self, orchestrator):
        """Test analytics stage."""
        result = OrchestrationResult(
            orchestration_id="test_analytics",
            start_time=time.time()
        )

        # Add execution results
        result.execution_results = {
            "test1": ExecutionResult(
                test_name="test1",
                status=ExecutionStatus.COMPLETED,
                duration=1.0,
                start_time=time.time(),
                end_time=time.time() + 1.0
            )
        }

        # Mock analytics
        mock_analytics = Mock()
        mock_suite_metrics = SuiteMetrics(
            suite_id="test_suite",
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            total_duration=1.0,
            overall_success_rate=1.0,
            health_status=HealthStatus.HEALTHY,
            flaky_test_count=0
        )
        mock_analytics.process_execution_results.return_value = mock_suite_metrics

        with patch.object(orchestrator, '_get_component_analytics', return_value=mock_analytics):
            await orchestrator._stage_analytics(result)

            assert result.suite_metrics is not None
            assert result.suite_metrics.overall_success_rate == 1.0
            assert result.suite_metrics.health_status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_stage_reporting(self, orchestrator):
        """Test reporting stage."""
        result = OrchestrationResult(
            orchestration_id="test_reporting",
            start_time=time.time()
        )
        result.end_time = result.start_time + 10.0

        # Add some test data
        result.discovered_tests = {"test1": Mock()}
        result.execution_results = {"test1": Mock()}
        result.stage_timings = {"discovery": 2.0, "execution": 5.0}

        await orchestrator._stage_reporting(result)

        # Check if report file was created
        report_files = list(orchestrator.project_root.glob("test_report_*.json"))
        assert len(report_files) == 1

        # Check report content
        with open(report_files[0]) as f:
            report_data = json.load(f)

            assert report_data["orchestration_id"] == result.orchestration_id
            assert report_data["duration"] == 10.0
            assert report_data["discovered_tests"] == 1
            assert report_data["executed_tests"] == 1

    @pytest.mark.asyncio
    async def test_stage_cleanup(self, orchestrator):
        """Test cleanup stage."""
        result = OrchestrationResult(
            orchestration_id="test_cleanup",
            start_time=time.time()
        )

        # Mock components
        orchestrator._discovery = Mock()
        orchestrator._analytics = Mock()
        orchestrator._integration = Mock()
        orchestrator._integration.cleanup_all = AsyncMock()

        await orchestrator._stage_cleanup(result)

        orchestrator._discovery.close.assert_called_once()
        orchestrator._analytics.close.assert_called_once()
        orchestrator._integration.cleanup_all.assert_called_once()

    def test_stop_orchestration(self, orchestrator):
        """Test orchestration stopping."""
        assert not orchestrator._stop_event.is_set()

        orchestrator.stop_orchestration()

        assert orchestrator._stop_event.is_set()

    def test_get_orchestration_history(self, orchestrator):
        """Test orchestration history retrieval."""
        # Add some test data to database
        with sqlite3.connect(orchestrator.database_path) as conn:
            conn.execute("""
                INSERT INTO orchestration_runs
                (id, start_time, end_time, status, config, results)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                "test_orch_1",
                time.time() - 3600,
                time.time() - 3500,
                "completed",
                "{}",
                '{"test": "data"}'
            ))

        history = orchestrator.get_orchestration_history(limit=5)

        assert len(history) >= 1
        assert history[0]["id"] == "test_orch_1"
        assert history[0]["status"] == "completed"
        assert "test" in history[0]["results"]

    def test_get_stage_performance_stats(self, orchestrator):
        """Test stage performance statistics."""
        # Add test data
        with sqlite3.connect(orchestrator.database_path) as conn:
            for i in range(3):
                conn.execute("""
                    INSERT INTO stage_timings
                    (orchestration_id, stage, start_time, end_time, duration, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    f"test_orch_{i}",
                    "discovery",
                    time.time(),
                    time.time() + (i + 1),
                    i + 1,
                    "completed"
                ))

        stats = orchestrator.get_stage_performance_stats()

        assert "discovery" in stats
        assert stats["discovery"]["execution_count"] == 3
        assert stats["discovery"]["avg_duration"] == 2.0  # (1+2+3)/3
        assert stats["discovery"]["min_duration"] == 1.0
        assert stats["discovery"]["max_duration"] == 3.0


class TestOrchestrationScheduler:
    """Test orchestration scheduler."""

    @pytest.fixture
    def temp_project(self):
        """Create temporary project structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            test_dir = project_root / "tests"
            test_dir.mkdir()
            yield project_root, test_dir

    @pytest.fixture
    def scheduler(self, temp_project):
        """Create test scheduler."""
        project_root, test_dir = temp_project
        orchestrator = TestOrchestrator(project_root, test_dir)
        return OrchestrationScheduler(orchestrator)

    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization."""
        assert scheduler.orchestrator is not None
        assert isinstance(scheduler.scheduled_runs, dict)
        assert isinstance(scheduler.running_orchestrations, set)
        assert scheduler.scheduler_executor is not None

    def test_schedule_orchestration(self, scheduler):
        """Test scheduling orchestration."""
        config = OrchestrationConfig(mode=OrchestrationMode.DISCOVERY_ONLY)

        scheduler.schedule_orchestration(
            schedule_id="daily_tests",
            cron_expression="0 9 * * *",  # Daily at 9 AM
            config=config
        )

        assert "daily_tests" in scheduler.scheduled_runs
        schedule_info = scheduler.scheduled_runs["daily_tests"]
        assert schedule_info["cron_expression"] == "0 9 * * *"
        assert schedule_info["config"] == config
        assert schedule_info["last_run"] is None
        assert schedule_info["next_run"] is not None

    def test_cancel_scheduled_orchestration(self, scheduler):
        """Test canceling scheduled orchestration."""
        config = OrchestrationConfig()

        scheduler.schedule_orchestration(
            schedule_id="test_schedule",
            cron_expression="0 * * * *",
            config=config
        )

        assert "test_schedule" in scheduler.scheduled_runs

        scheduler.cancel_scheduled_orchestration("test_schedule")

        assert "test_schedule" not in scheduler.scheduled_runs

    def test_get_schedule_status(self, scheduler):
        """Test getting schedule status."""
        config = OrchestrationConfig()

        scheduler.schedule_orchestration(
            schedule_id="status_test",
            cron_expression="0 12 * * *",
            config=config
        )

        status = scheduler.get_schedule_status()

        assert "status_test" in status
        schedule_status = status["status_test"]
        assert schedule_status["cron_expression"] == "0 12 * * *"
        assert schedule_status["last_run"] is None
        assert schedule_status["next_run"] is not None
        assert schedule_status["is_running"] is False

    def test_scheduler_cleanup(self, scheduler):
        """Test scheduler cleanup."""
        scheduler.close()

        # Executor should be shut down
        assert scheduler.scheduler_executor._shutdown


class TestOrchestrationIntegration:
    """Integration tests for orchestration system."""

    @pytest.fixture
    def temp_project_with_tests(self):
        """Create temporary project with actual test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            test_dir = project_root / "tests"
            test_dir.mkdir()

            # Create multiple test files
            (test_dir / "test_unit.py").write_text('''
import pytest

def test_addition():
    assert 1 + 1 == 2

def test_subtraction():
    assert 5 - 3 == 2

@pytest.mark.parametrize("a,b,expected", [(1,2,3), (2,3,5)])
def test_parametrized(a, b, expected):
    assert a + b == expected
''')

            (test_dir / "test_integration.py").write_text('''
import pytest

@pytest.mark.integration
def test_database_connection():
    # Mock database test
    assert True

@pytest.mark.integration
def test_api_endpoint():
    # Mock API test
    assert True
''')

            yield project_root, test_dir

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, temp_project_with_tests):
        """Test complete orchestration pipeline."""
        project_root, test_dir = temp_project_with_tests

        config = OrchestrationConfig(
            mode=OrchestrationMode.FULL_PIPELINE,
            enable_integration=False,  # Skip integration for this test
            cleanup_on_completion=True
        )

        orchestrator = TestOrchestrator(project_root, test_dir, config)

        # Mock the execution to avoid actually running tests
        with patch.object(orchestrator, '_get_component_executor') as mock_get_executor:
            mock_executor = Mock()
            mock_executor.execute_tests.return_value = {
                "test_addition": ExecutionResult(
                    test_name="test_addition",
                    status=ExecutionStatus.COMPLETED,
                    duration=0.1,
                    start_time=time.time(),
                    end_time=time.time() + 0.1
                )
            }
            mock_executor.create_execution_plan.return_value = Mock(
                test_batches=1,
                estimated_duration=1.0,
                max_parallelism=1
            )
            mock_get_executor.return_value = mock_executor

            result = await orchestrator.orchestrate_tests()

            assert result.status == PipelineStage.COMPLETED
            assert len(result.discovered_tests) > 0
            assert result.success_rate >= 0.0
            assert result.duration is not None

    @pytest.mark.asyncio
    async def test_discovery_only_mode(self, temp_project_with_tests):
        """Test discovery-only orchestration mode."""
        project_root, test_dir = temp_project_with_tests

        config = OrchestrationConfig(mode=OrchestrationMode.DISCOVERY_ONLY)
        orchestrator = TestOrchestrator(project_root, test_dir, config)

        result = await orchestrator.orchestrate_tests()

        assert result.status == PipelineStage.COMPLETED
        assert len(result.discovered_tests) > 0
        assert len(result.execution_results) == 0  # No execution in discovery mode

    @pytest.mark.asyncio
    async def test_error_recovery(self, temp_project_with_tests):
        """Test orchestration error recovery."""
        project_root, test_dir = temp_project_with_tests

        orchestrator = TestOrchestrator(project_root, test_dir)

        # Mock discovery to fail
        with patch.object(orchestrator, '_get_component_discovery') as mock_get_discovery:
            mock_discovery = Mock()
            mock_discovery.discover_tests.side_effect = Exception("Discovery failed")
            mock_get_discovery.return_value = mock_discovery

            result = await orchestrator.orchestrate_tests()

            assert result.status == PipelineStage.FAILED
            assert len(result.errors) > 0
            assert "Discovery failed" in str(result.errors)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])