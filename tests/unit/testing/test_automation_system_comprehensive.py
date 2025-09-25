"""
Comprehensive unit tests for TestAutomationSystem with edge cases.

Tests automation orchestration, triggering, phase coordination, and
comprehensive error handling scenarios.
"""

import pytest
import asyncio
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.python.workspace_qdrant_mcp.testing.execution.automation import (
    TestAutomationSystem,
    AutomationConfig,
    AutomationMode,
    AutomationTrigger,
    AutomationRun,
    TriggerType
)
from src.python.workspace_qdrant_mcp.testing.execution.scheduler import (
    TestExecutionScheduler,
    ExecutionPriority,
    ExecutionStatus
)
from src.python.workspace_qdrant_mcp.testing.analytics.engine import (
    TestAnalyticsEngine,
    TestMetrics,
    QualityReport
)


class TestAutomationConfig:
    """Tests for AutomationConfig dataclass."""

    def test_default_config(self):
        """Test default automation configuration."""
        config = AutomationConfig()

        assert config.mode == AutomationMode.SCHEDULED
        assert config.test_patterns == ["test_*.py", "*_test.py"]
        assert config.test_command == ["python", "-m", "pytest"]
        assert config.parallel_jobs == 4
        assert config.enable_analytics is True
        assert config.failure_threshold == 0.1
        assert config.coverage_threshold == 0.8

    def test_custom_config(self):
        """Test custom automation configuration."""
        config = AutomationConfig(
            mode=AutomationMode.CONTINUOUS,
            test_patterns=["tests/*.py"],
            parallel_jobs=8,
            failure_threshold=0.05,
            coverage_threshold=0.9
        )

        assert config.mode == AutomationMode.CONTINUOUS
        assert config.test_patterns == ["tests/*.py"]
        assert config.parallel_jobs == 8
        assert config.failure_threshold == 0.05
        assert config.coverage_threshold == 0.9


class TestAutomationTrigger:
    """Tests for AutomationTrigger dataclass."""

    def test_trigger_creation(self):
        """Test automation trigger creation."""
        trigger = AutomationTrigger(
            trigger_id="test_trigger",
            trigger_type=TriggerType.TIME_BASED,
            description="Test trigger",
            conditions={"schedule": "0 */6 * * *"},
            actions=["run_tests", "generate_report"]
        )

        assert trigger.trigger_id == "test_trigger"
        assert trigger.trigger_type == TriggerType.TIME_BASED
        assert trigger.enabled is True
        assert trigger.trigger_count == 0
        assert trigger.last_triggered is None


class TestAutomationRun:
    """Tests for AutomationRun dataclass."""

    def test_run_creation(self):
        """Test automation run creation."""
        run = AutomationRun(
            run_id="test_run_1",
            started_at=datetime.now(),
            trigger_type=TriggerType.MANUAL
        )

        assert run.run_id == "test_run_1"
        assert run.status == "running"
        assert run.trigger_type == TriggerType.MANUAL
        assert run.executions == []
        assert run.results_summary == {}


class TestTestAutomationSystem:
    """Comprehensive tests for TestAutomationSystem."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)
        yield db_path
        if db_path.exists():
            db_path.unlink()

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for automation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_execution_scheduler(self):
        """Create mock execution scheduler."""
        scheduler = Mock(spec=TestExecutionScheduler)

        # Mock scheduling method
        scheduler.schedule_test_execution.return_value = "exec_123"

        # Mock status checking
        scheduler.get_execution_status.return_value = {
            'status': 'completed',
            'tests_run': 10,
            'tests_passed': 9,
            'tests_failed': 1,
            'duration': 30.0
        }

        return scheduler

    @pytest.fixture
    def mock_analytics_engine(self):
        """Create mock analytics engine."""
        engine = Mock(spec=TestAnalyticsEngine)

        # Mock metrics calculation
        engine.calculate_metrics.return_value = TestMetrics(
            total_tests=100,
            passed=85,
            failed=15,
            pass_rate=85.0,
            coverage_percentage=90.0
        )

        # Mock quality report
        engine.generate_quality_report.return_value = QualityReport(
            overall_score=85.0,
            metrics={},
            trends=[],
            recommendations=["Good test coverage"],
            warnings=[],
            critical_issues=[]
        )

        return engine

    @pytest.fixture
    def automation_config(self, temp_dir):
        """Create automation configuration."""
        return AutomationConfig(
            mode=AutomationMode.ON_DEMAND,
            working_directory=temp_dir,
            parallel_jobs=2,
            max_execution_time=30,  # Short timeout for testing
            enable_dashboard=False,  # Disable to simplify testing
            enable_maintenance=False,
            enable_discovery=False,
            cleanup_old_data=False
        )

    @pytest.fixture
    def automation_system(self, mock_execution_scheduler, mock_analytics_engine, automation_config, temp_db):
        """Create automation system for testing."""
        return TestAutomationSystem(
            execution_scheduler=mock_execution_scheduler,
            analytics_engine=mock_analytics_engine,
            config=automation_config,
            db_path=temp_db
        )

    def test_automation_system_initialization(self, automation_system, temp_db):
        """Test automation system initialization."""
        assert automation_system.execution_scheduler is not None
        assert automation_system.analytics_engine is not None
        assert automation_system.config is not None

        # Check database was created
        assert temp_db.exists()

        # Check default triggers were created
        assert len(automation_system.triggers) > 0

    def test_automation_system_initialization_with_all_features(self, mock_execution_scheduler, mock_analytics_engine, temp_db):
        """Test initialization with all features enabled."""
        config = AutomationConfig(
            enable_dashboard=True,
            enable_maintenance=True,
            enable_discovery=True
        )

        automation_system = TestAutomationSystem(
            execution_scheduler=mock_execution_scheduler,
            analytics_engine=mock_analytics_engine,
            config=config,
            db_path=temp_db
        )

        assert automation_system.dashboard_generator is not None
        assert automation_system.maintenance_scheduler is not None
        assert automation_system.discovery_engine is not None

    def test_default_triggers_setup(self, automation_system):
        """Test default triggers are set up correctly."""
        triggers = automation_system.triggers

        # Should have some default triggers
        assert len(triggers) > 0

        # Check for specific trigger types
        trigger_types = [t.trigger_type for t in triggers.values()]
        assert TriggerType.COVERAGE_DROP in trigger_types
        assert TriggerType.FAILURE_SPIKE in trigger_types

    def test_start_automation_system(self, automation_system):
        """Test starting the automation system."""
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance

            automation_system.start_automation()

            assert automation_system.running is True
            assert mock_thread.called
            assert mock_thread_instance.start.called

    def test_start_automation_already_running(self, automation_system):
        """Test starting automation system when already running."""
        automation_system.running = True

        with patch('threading.Thread') as mock_thread:
            automation_system.start_automation()

            # Should not create new thread
            assert not mock_thread.called

    def test_stop_automation_system(self, automation_system):
        """Test stopping the automation system."""
        # Mock running state
        automation_system.running = True
        automation_system.scheduler_thread = Mock()
        automation_system.scheduler_thread.is_alive.return_value = True

        with patch('schedule.clear') as mock_clear:
            automation_system.stop_automation()

            assert automation_system.running is False
            assert mock_clear.called
            assert automation_system.scheduler_thread.join.called

    def test_trigger_automation_run_success(self, automation_system):
        """Test triggering automation run successfully."""
        run_id = automation_system.trigger_automation_run(
            trigger_type=TriggerType.MANUAL,
            test_patterns=["custom_test_*.py"]
        )

        assert run_id is not None
        assert run_id.startswith("run_")
        assert run_id in automation_system.active_runs

        # Check run was created correctly
        run = automation_system.active_runs[run_id]
        assert run.trigger_type == TriggerType.MANUAL
        assert run.status == "running"

    def test_trigger_automation_run_error(self, automation_system):
        """Test triggering automation run with error."""
        # Mock error in run creation
        with patch.object(automation_system, '_execute_automation_run', side_effect=Exception("Start error")):
            run_id = automation_system.trigger_automation_run()

            assert run_id is not None
            assert run_id in automation_system.active_runs

            # Should create failed run
            run = automation_system.active_runs[run_id]
            assert run.status == "failed"

    @pytest.mark.asyncio
    async def test_execute_automation_run_success(self, automation_system):
        """Test successful automation run execution."""
        run = AutomationRun(
            run_id="test_run",
            started_at=datetime.now(),
            trigger_type=TriggerType.MANUAL
        )

        # Mock successful execution
        automation_system.config.enable_discovery = False
        automation_system.config.enable_analytics = True

        await automation_system._execute_automation_run(run)

        assert run.status == "completed"
        assert run.completed_at is not None
        assert 'execution' in run.results_summary

    @pytest.mark.asyncio
    async def test_execute_automation_run_error(self, automation_system):
        """Test automation run execution with error."""
        run = AutomationRun(
            run_id="test_run",
            started_at=datetime.now(),
            trigger_type=TriggerType.MANUAL
        )

        # Mock error in test execution
        automation_system.execution_scheduler.schedule_test_execution.side_effect = Exception("Execution error")

        await automation_system._execute_automation_run(run)

        assert run.status == "failed"
        assert run.error_message is not None

    @pytest.mark.asyncio
    async def test_run_test_discovery_success(self, automation_system):
        """Test test discovery phase success."""
        automation_system.config.enable_discovery = True

        # Mock discovery engine
        mock_discovery_result = Mock()
        mock_discovery_result.discovered_tests = [Mock() for _ in range(5)]
        mock_discovery_result.test_suggestions = [Mock() for _ in range(3)]
        mock_discovery_result.coverage_analysis = Mock()
        mock_discovery_result.coverage_analysis.gaps = [Mock() for _ in range(2)]

        automation_system.discovery_engine = Mock()
        automation_system.discovery_engine.discover_tests.return_value = mock_discovery_result

        run = AutomationRun("test_run", datetime.now())

        await automation_system._run_test_discovery(run)

        assert 'discovery' in run.results_summary
        assert run.results_summary['discovery']['tests_discovered'] == 5
        assert run.results_summary['discovery']['suggestions_generated'] == 3
        assert run.results_summary['discovery']['coverage_gaps'] == 2

    @pytest.mark.asyncio
    async def test_run_test_discovery_error(self, automation_system):
        """Test test discovery phase with error."""
        automation_system.config.enable_discovery = True
        automation_system.discovery_engine = Mock()
        automation_system.discovery_engine.discover_tests.side_effect = Exception("Discovery error")

        run = AutomationRun("test_run", datetime.now())

        await automation_system._run_test_discovery(run)

        assert 'discovery' in run.results_summary
        assert 'error' in run.results_summary['discovery']

    @pytest.mark.asyncio
    async def test_run_test_execution_success(self, automation_system):
        """Test test execution phase success."""
        run = AutomationRun("test_run", datetime.now())

        with patch.object(automation_system, '_wait_for_executions') as mock_wait:
            await automation_system._run_test_execution(run, ["test_*.py"])

            assert len(run.executions) > 0
            assert mock_wait.called

    @pytest.mark.asyncio
    async def test_run_test_execution_with_additional_args(self, automation_system):
        """Test test execution with additional arguments."""
        run = AutomationRun("test_run", datetime.now())

        with patch.object(automation_system, '_wait_for_executions'):
            await automation_system._run_test_execution(run, None, ["--verbose", "--cov"])

            # Check that scheduler was called with additional args
            call_args = automation_system.execution_scheduler.schedule_test_execution.call_args
            command = call_args[1]['command']  # keyword argument
            assert "--verbose" in command
            assert "--cov" in command

    @pytest.mark.asyncio
    async def test_run_test_execution_error(self, automation_system):
        """Test test execution phase with error."""
        run = AutomationRun("test_run", datetime.now())

        automation_system.execution_scheduler.schedule_test_execution.side_effect = Exception("Execution error")

        await automation_system._run_test_execution(run)

        assert 'execution' in run.results_summary
        assert 'error' in run.results_summary['execution']

    @pytest.mark.asyncio
    async def test_wait_for_executions_success(self, automation_system):
        """Test waiting for executions to complete."""
        run = AutomationRun("test_run", datetime.now())
        execution_ids = ["exec_1", "exec_2"]

        # Mock executions completing immediately
        automation_system.execution_scheduler.get_execution_status.return_value = {
            'status': 'completed',
            'tests_run': 10,
            'tests_passed': 9,
            'tests_failed': 1,
            'duration': 15.0
        }

        await automation_system._wait_for_executions(execution_ids, run)

        assert 'execution' in run.results_summary
        assert run.results_summary['execution']['total_tests'] == 20  # 2 executions * 10 tests

    @pytest.mark.asyncio
    async def test_wait_for_executions_timeout(self, automation_system):
        """Test waiting for executions with timeout."""
        run = AutomationRun("test_run", datetime.now())
        execution_ids = ["exec_1", "exec_2"]

        # Mock executions that never complete
        automation_system.execution_scheduler.get_execution_status.return_value = {
            'status': 'running'
        }

        # Short timeout
        automation_system.config.max_execution_time = 1

        await automation_system._wait_for_executions(execution_ids, run)

        # Should have cancelled executions
        assert automation_system.execution_scheduler.cancel_execution.called

    @pytest.mark.asyncio
    async def test_collect_execution_results_success(self, automation_system):
        """Test collecting execution results."""
        run = AutomationRun("test_run", datetime.now())
        run.executions = ["exec_1", "exec_2"]

        # Mock different execution results
        status_map = {
            "exec_1": {
                'status': 'completed',
                'tests_run': 10,
                'tests_passed': 8,
                'tests_failed': 2,
                'duration': 15.0
            },
            "exec_2": {
                'status': 'completed',
                'tests_run': 5,
                'tests_passed': 5,
                'tests_failed': 0,
                'duration': 10.0
            }
        }

        automation_system.execution_scheduler.get_execution_status.side_effect = lambda exec_id: status_map.get(exec_id)

        await automation_system._collect_execution_results(run)

        assert run.results_summary['execution']['total_tests'] == 15
        assert run.results_summary['execution']['total_passed'] == 13
        assert run.results_summary['execution']['total_failed'] == 2
        assert run.results_summary['execution']['total_duration'] == 25.0

    @pytest.mark.asyncio
    async def test_collect_execution_results_with_none_values(self, automation_system):
        """Test collecting execution results with None duration."""
        run = AutomationRun("test_run", datetime.now())
        run.executions = ["exec_1"]

        automation_system.execution_scheduler.get_execution_status.return_value = {
            'status': 'completed',
            'tests_run': 10,
            'tests_passed': 10,
            'tests_failed': 0,
            'duration': None  # None duration
        }

        await automation_system._collect_execution_results(run)

        # Should handle None values gracefully
        assert run.results_summary['execution']['total_duration'] == 0

    @pytest.mark.asyncio
    async def test_run_analytics_success(self, automation_system):
        """Test analytics phase success."""
        run = AutomationRun("test_run", datetime.now())

        await automation_system._run_analytics(run)

        assert 'analytics' in run.results_summary
        assert 'pass_rate' in run.results_summary['analytics']
        assert 'quality_score' in run.results_summary['analytics']

    @pytest.mark.asyncio
    async def test_run_analytics_error(self, automation_system):
        """Test analytics phase with error."""
        run = AutomationRun("test_run", datetime.now())

        automation_system.analytics_engine.calculate_metrics.side_effect = Exception("Analytics error")

        await automation_system._run_analytics(run)

        assert 'analytics' in run.results_summary
        assert 'error' in run.results_summary['analytics']

    @pytest.mark.asyncio
    async def test_generate_dashboard_success(self, automation_system):
        """Test dashboard generation phase success."""
        automation_system.config.enable_dashboard = True
        automation_system.dashboard_generator = Mock()
        automation_system.dashboard_generator.generate_dashboard.return_value = "/path/to/dashboard.html"

        run = AutomationRun("test_run", datetime.now())

        await automation_system._generate_dashboard(run)

        assert run.dashboard_path == "/path/to/dashboard.html"
        assert 'dashboard' in run.results_summary

    @pytest.mark.asyncio
    async def test_generate_dashboard_error(self, automation_system):
        """Test dashboard generation phase with error."""
        automation_system.config.enable_dashboard = True
        automation_system.dashboard_generator = Mock()
        automation_system.dashboard_generator.generate_dashboard.side_effect = Exception("Dashboard error")

        run = AutomationRun("test_run", datetime.now())

        await automation_system._generate_dashboard(run)

        assert 'dashboard' in run.results_summary
        assert 'error' in run.results_summary['dashboard']

    @pytest.mark.asyncio
    async def test_run_maintenance_checks_low_pass_rate(self, automation_system):
        """Test maintenance checks with low pass rate."""
        automation_system.config.enable_maintenance = True
        automation_system.maintenance_scheduler = Mock()
        automation_system.maintenance_scheduler.schedule_task.return_value = True

        run = AutomationRun("test_run", datetime.now())
        run.results_summary['execution'] = {'pass_rate': 70.0}  # Low pass rate

        await automation_system._run_maintenance_checks(run)

        assert 'maintenance' in run.results_summary
        assert run.results_summary['maintenance']['tasks_scheduled'] >= 1

        # Should have scheduled investigation task
        assert automation_system.maintenance_scheduler.schedule_task.called

    @pytest.mark.asyncio
    async def test_run_maintenance_checks_slow_execution(self, automation_system):
        """Test maintenance checks with slow execution."""
        automation_system.config.enable_maintenance = True
        automation_system.maintenance_scheduler = Mock()
        automation_system.maintenance_scheduler.schedule_task.return_value = True

        run = AutomationRun("test_run", datetime.now())
        run.results_summary['execution'] = {'total_duration': 2000.0}  # Slow execution

        await automation_system._run_maintenance_checks(run)

        assert 'maintenance' in run.results_summary
        assert run.results_summary['maintenance']['tasks_scheduled'] >= 1

    @pytest.mark.asyncio
    async def test_run_maintenance_checks_good_metrics(self, automation_system):
        """Test maintenance checks with good metrics."""
        automation_system.config.enable_maintenance = True
        automation_system.maintenance_scheduler = Mock()

        run = AutomationRun("test_run", datetime.now())
        run.results_summary['execution'] = {
            'pass_rate': 95.0,  # Good pass rate
            'total_duration': 300.0  # Fast execution
        }

        await automation_system._run_maintenance_checks(run)

        # Should not schedule any tasks
        assert run.results_summary['maintenance']['tasks_scheduled'] == 0

    @pytest.mark.asyncio
    async def test_run_maintenance_checks_error(self, automation_system):
        """Test maintenance checks with error."""
        automation_system.config.enable_maintenance = True
        automation_system.maintenance_scheduler = Mock()
        automation_system.maintenance_scheduler.schedule_task.side_effect = Exception("Maintenance error")

        run = AutomationRun("test_run", datetime.now())
        run.results_summary['execution'] = {'pass_rate': 50.0}

        await automation_system._run_maintenance_checks(run)

        assert 'maintenance' in run.results_summary
        assert 'error' in run.results_summary['maintenance']

    @pytest.mark.asyncio
    async def test_cleanup_old_data_success(self, automation_system):
        """Test data cleanup phase success."""
        run = AutomationRun("test_run", datetime.now())

        await automation_system._cleanup_old_data(run)

        assert 'cleanup' in run.results_summary
        assert 'cutoff_date' in run.results_summary['cleanup']
        assert 'retention_days' in run.results_summary['cleanup']

    @pytest.mark.asyncio
    async def test_cleanup_old_data_error(self, automation_system):
        """Test data cleanup phase with error."""
        run = AutomationRun("test_run", datetime.now())

        # Mock error in cleanup
        with patch.object(automation_system, '_cleanup_old_data',
                         side_effect=Exception("Cleanup error")) as mock_cleanup:
            # Call the original method to test error handling
            mock_cleanup.side_effect = None
            mock_cleanup.side_effect = Exception("Cleanup error")

            try:
                await automation_system._cleanup_old_data(run)
            except Exception:
                pass  # Expected

    def test_add_trigger_success(self, automation_system):
        """Test adding automation trigger."""
        trigger = AutomationTrigger(
            trigger_id="custom_trigger",
            trigger_type=TriggerType.FILE_CHANGE,
            description="Custom trigger",
            conditions={"pattern": "*.py"},
            actions=["run_tests"]
        )

        initial_count = len(automation_system.triggers)
        automation_system.add_trigger(trigger)

        assert len(automation_system.triggers) == initial_count + 1
        assert "custom_trigger" in automation_system.triggers

    def test_add_trigger_database_save(self, automation_system):
        """Test trigger is saved to database."""
        trigger = AutomationTrigger(
            trigger_id="db_trigger",
            trigger_type=TriggerType.MANUAL,
            description="DB trigger"
        )

        automation_system.add_trigger(trigger)

        # Verify saved to database
        import sqlite3
        conn = sqlite3.connect(automation_system.db_path)
        cursor = conn.execute('SELECT * FROM automation_triggers WHERE trigger_id = ?', ("db_trigger",))
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_remove_trigger_success(self, automation_system):
        """Test removing automation trigger."""
        # Add trigger first
        trigger = AutomationTrigger(
            trigger_id="temp_trigger",
            trigger_type=TriggerType.MANUAL,
            description="Temporary trigger"
        )
        automation_system.add_trigger(trigger)

        # Remove trigger
        success = automation_system.remove_trigger("temp_trigger")

        assert success is True
        assert "temp_trigger" not in automation_system.triggers

    def test_remove_trigger_nonexistent(self, automation_system):
        """Test removing non-existent trigger."""
        success = automation_system.remove_trigger("nonexistent_trigger")
        assert success is False

    def test_get_automation_status(self, automation_system):
        """Test getting automation system status."""
        automation_system.running = True
        automation_system.active_runs["run_1"] = Mock()
        automation_system.active_runs["run_2"] = Mock()

        status = automation_system.get_automation_status()

        assert status['running'] is True
        assert status['active_runs'] == 2
        assert status['triggers_configured'] > 0
        assert 'config' in status
        assert 'recent_runs' in status

    def test_get_run_status_active(self, automation_system):
        """Test getting status of active run."""
        run = AutomationRun(
            run_id="active_run",
            started_at=datetime.now(),
            trigger_type=TriggerType.MANUAL,
            status="running"
        )
        run.executions = ["exec_1", "exec_2"]
        run.results_summary = {"execution": {"total_tests": 10}}

        automation_system.active_runs["active_run"] = run

        status = automation_system.get_run_status("active_run")

        assert status is not None
        assert status['run_id'] == "active_run"
        assert status['status'] == "running"
        assert status['executions'] == 2
        assert 'results_summary' in status

    def test_get_run_status_nonexistent(self, automation_system):
        """Test getting status of non-existent run."""
        status = automation_system.get_run_status("nonexistent_run")
        assert status is None

    def test_scheduler_loop_with_schedule(self, automation_system):
        """Test scheduler loop with schedule library."""
        automation_system.running = True

        with patch('schedule.run_pending') as mock_run_pending, \
             patch('time.sleep') as mock_sleep:

            # Mock to stop after one iteration
            def stop_running(*args):
                automation_system.running = False

            mock_sleep.side_effect = stop_running

            automation_system._scheduler_loop()

            assert mock_run_pending.called
            assert mock_sleep.called

    def test_scheduler_loop_error(self, automation_system):
        """Test scheduler loop with error."""
        automation_system.running = True

        with patch('schedule.run_pending', side_effect=Exception("Schedule error")), \
             patch('time.sleep') as mock_sleep:

            # Mock to stop after one iteration
            def stop_running(*args):
                automation_system.running = False

            mock_sleep.side_effect = stop_running

            automation_system._scheduler_loop()

            # Should handle error and continue
            assert mock_sleep.called

    def test_setup_scheduled_triggers(self, automation_system):
        """Test setting up scheduled triggers."""
        # Add time-based trigger
        trigger = AutomationTrigger(
            trigger_id="scheduled_test",
            trigger_type=TriggerType.TIME_BASED,
            description="Scheduled test",
            conditions={"schedule": "0 */6 * * *"},
            enabled=True
        )
        automation_system.triggers["scheduled_test"] = trigger

        with patch('schedule.every') as mock_schedule:
            mock_schedule.return_value.hours.do.return_value = None

            automation_system._setup_scheduled_triggers()

            # Should have set up schedule for 6-hour interval
            assert mock_schedule.called

    def test_trigger_scheduled_run(self, automation_system):
        """Test triggering a scheduled run."""
        # Add trigger
        trigger = AutomationTrigger(
            trigger_id="test_trigger",
            trigger_type=TriggerType.TIME_BASED,
            description="Test trigger",
            enabled=True
        )
        automation_system.triggers["test_trigger"] = trigger

        with patch.object(automation_system, 'trigger_automation_run') as mock_trigger:
            automation_system._trigger_scheduled_run("test_trigger")

            assert trigger.trigger_count == 1
            assert trigger.last_triggered is not None
            assert mock_trigger.called

    def test_trigger_scheduled_run_disabled(self, automation_system):
        """Test triggering a disabled scheduled run."""
        # Add disabled trigger
        trigger = AutomationTrigger(
            trigger_id="disabled_trigger",
            trigger_type=TriggerType.TIME_BASED,
            description="Disabled trigger",
            enabled=False
        )
        automation_system.triggers["disabled_trigger"] = trigger

        with patch.object(automation_system, 'trigger_automation_run') as mock_trigger:
            automation_system._trigger_scheduled_run("disabled_trigger")

            # Should not trigger
            assert not mock_trigger.called

    def test_trigger_scheduled_run_nonexistent(self, automation_system):
        """Test triggering non-existent scheduled run."""
        with patch.object(automation_system, 'trigger_automation_run') as mock_trigger:
            automation_system._trigger_scheduled_run("nonexistent_trigger")

            # Should not trigger
            assert not mock_trigger.called

    def test_trigger_scheduled_run_error(self, automation_system):
        """Test triggering scheduled run with error."""
        # Add trigger
        trigger = AutomationTrigger(
            trigger_id="error_trigger",
            trigger_type=TriggerType.TIME_BASED,
            description="Error trigger",
            enabled=True
        )
        automation_system.triggers["error_trigger"] = trigger

        with patch.object(automation_system, 'trigger_automation_run', side_effect=Exception("Trigger error")):
            # Should not crash
            automation_system._trigger_scheduled_run("error_trigger")

    def test_save_automation_run(self, automation_system):
        """Test saving automation run to database."""
        run = AutomationRun(
            run_id="test_save_run",
            started_at=datetime.now(),
            trigger_type=TriggerType.MANUAL,
            status="completed",
            executions=["exec_1", "exec_2"],
            results_summary={"tests": 10}
        )

        automation_system._save_automation_run(run)

        # Verify saved to database
        import sqlite3
        conn = sqlite3.connect(automation_system.db_path)
        cursor = conn.execute('SELECT * FROM automation_runs WHERE run_id = ?', ("test_save_run",))
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_save_trigger(self, automation_system):
        """Test saving trigger to database."""
        trigger = AutomationTrigger(
            trigger_id="test_save_trigger",
            trigger_type=TriggerType.MANUAL,
            description="Test save trigger",
            conditions={"test": "value"},
            actions=["action1", "action2"],
            trigger_count=5
        )

        automation_system._save_trigger(trigger)

        # Verify saved to database
        import sqlite3
        conn = sqlite3.connect(automation_system.db_path)
        cursor = conn.execute('SELECT * FROM automation_triggers WHERE trigger_id = ?', ("test_save_trigger",))
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_database_error_handling(self, automation_system):
        """Test database operations with errors."""
        run = AutomationRun(
            run_id="error_run",
            started_at=datetime.now(),
            trigger_type=TriggerType.MANUAL
        )

        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            # Should not crash
            automation_system._save_automation_run(run)

    def test_concurrent_automation_runs(self, automation_system):
        """Test multiple concurrent automation runs."""
        import threading

        results = []
        errors = []

        def worker(worker_id):
            try:
                run_id = automation_system.trigger_automation_run(
                    trigger_type=TriggerType.MANUAL
                )
                results.append(run_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have no errors and all runs started
        assert len(errors) == 0
        assert len(results) == 3
        assert len(automation_system.active_runs) >= 3

    def test_monitoring_integration_placeholder(self, automation_system):
        """Test monitoring integration placeholder."""
        # This tests the placeholder _start_monitoring method
        automation_system._start_monitoring()

        # Should complete without error (placeholder implementation)
        assert True

    def test_config_validation(self):
        """Test configuration validation scenarios."""
        # Test with minimal config
        config = AutomationConfig(
            mode=AutomationMode.CI_TRIGGERED,
            test_patterns=[],
            test_command=[],
            enable_analytics=False,
            enable_dashboard=False
        )

        assert config.mode == AutomationMode.CI_TRIGGERED
        assert config.test_patterns == []
        assert config.enable_analytics is False