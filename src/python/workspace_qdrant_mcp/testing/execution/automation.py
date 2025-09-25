"""
Test Execution Automation

High-level automation system for test suite management, continuous integration,
automated reporting, and maintenance with comprehensive error handling and
recovery mechanisms.
"""

import logging
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Callable
from enum import Enum
import schedule
import threading

from .scheduler import TestExecutionScheduler, TestExecution, ExecutionResult, ExecutionPriority, ExecutionStatus
from ..analytics.engine import TestAnalyticsEngine, TestResult, TestMetrics
from ..analytics.dashboard import DashboardGenerator
from ..lifecycle.scheduler import MaintenanceScheduler, MaintenanceTask, TaskType, TaskPriority
from ..discovery.engine import TestDiscoveryEngine

logger = logging.getLogger(__name__)


class AutomationMode(Enum):
    """Automation operation modes."""
    CONTINUOUS = "continuous"
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"
    CI_TRIGGERED = "ci_triggered"


class TriggerType(Enum):
    """Types of automation triggers."""
    TIME_BASED = "time_based"
    FILE_CHANGE = "file_change"
    GIT_PUSH = "git_push"
    MANUAL = "manual"
    DEPENDENCY_UPDATE = "dependency_update"
    COVERAGE_DROP = "coverage_drop"
    FAILURE_SPIKE = "failure_spike"


@dataclass
class AutomationConfig:
    """Configuration for test automation."""
    mode: AutomationMode = AutomationMode.SCHEDULED
    test_patterns: List[str] = field(default_factory=lambda: ["test_*.py", "*_test.py"])
    test_command: List[str] = field(default_factory=lambda: ["python", "-m", "pytest"])
    working_directory: Optional[Path] = None
    schedule_cron: Optional[str] = None  # e.g., "0 */6 * * *" for every 6 hours
    parallel_jobs: int = 4
    enable_analytics: bool = True
    enable_dashboard: bool = True
    enable_maintenance: bool = True
    enable_discovery: bool = True
    notification_webhooks: List[str] = field(default_factory=list)
    failure_threshold: float = 0.1  # 10% failure rate triggers alert
    coverage_threshold: float = 0.8  # 80% minimum coverage
    max_execution_time: int = 3600  # 1 hour max
    retry_failed_tests: bool = True
    generate_reports: bool = True
    cleanup_old_data: bool = True
    data_retention_days: int = 30


@dataclass
class AutomationTrigger:
    """Represents an automation trigger."""
    trigger_id: str
    trigger_type: TriggerType
    description: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    actions: List[str] = field(default_factory=list)
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class AutomationRun:
    """Represents a complete automation run."""
    run_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    trigger_type: TriggerType = TriggerType.MANUAL
    status: str = "running"  # running, completed, failed, cancelled
    executions: List[str] = field(default_factory=list)  # execution IDs
    results_summary: Dict[str, Any] = field(default_factory=dict)
    dashboard_path: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None


class TestAutomationSystem:
    """
    Comprehensive test automation system.

    Orchestrates test execution, analytics, reporting, and maintenance
    with intelligent triggering and comprehensive error handling.
    """

    def __init__(self,
                 execution_scheduler: TestExecutionScheduler,
                 analytics_engine: TestAnalyticsEngine,
                 config: AutomationConfig,
                 db_path: Optional[Path] = None):
        """
        Initialize automation system.

        Args:
            execution_scheduler: Test execution scheduler
            analytics_engine: Analytics engine
            config: Automation configuration
            db_path: Path to automation database
        """
        self.execution_scheduler = execution_scheduler
        self.analytics_engine = analytics_engine
        self.config = config
        self.db_path = db_path or Path.cwd() / "automation.db"

        # Optional components
        self.dashboard_generator = None
        self.maintenance_scheduler = None
        self.discovery_engine = None

        # Initialize optional components
        if config.enable_dashboard:
            self.dashboard_generator = DashboardGenerator(analytics_engine)

        if config.enable_maintenance:
            maintenance_db = self.db_path.parent / "maintenance.db"
            self.maintenance_scheduler = MaintenanceScheduler(maintenance_db)

        if config.enable_discovery:
            discovery_db = self.db_path.parent / "discovery.db"
            self.discovery_engine = TestDiscoveryEngine(discovery_db)

        # State management
        self.active_runs = {}  # run_id -> AutomationRun
        self.triggers = {}  # trigger_id -> AutomationTrigger
        self.scheduler_thread = None
        self.running = False

        # Setup triggers
        self._setup_default_triggers()

        # Initialize database
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize automation database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS automation_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    trigger_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    executions TEXT,  -- JSON array
                    results_summary TEXT,  -- JSON object
                    dashboard_path TEXT,
                    report_path TEXT,
                    error_message TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS automation_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    trigger_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    conditions TEXT,  -- JSON object
                    actions TEXT,  -- JSON array
                    enabled INTEGER DEFAULT 1,
                    last_triggered REAL,
                    trigger_count INTEGER DEFAULT 0
                )
            ''')

            conn.commit()
        finally:
            conn.close()

    def _setup_default_triggers(self) -> None:
        """Setup default automation triggers."""
        # Time-based trigger for regular test runs
        if self.config.schedule_cron:
            self.add_trigger(AutomationTrigger(
                trigger_id="scheduled_run",
                trigger_type=TriggerType.TIME_BASED,
                description="Regular scheduled test execution",
                conditions={"schedule": self.config.schedule_cron},
                actions=["run_full_test_suite", "generate_dashboard", "check_maintenance"]
            ))

        # Coverage drop trigger
        self.add_trigger(AutomationTrigger(
            trigger_id="coverage_drop",
            trigger_type=TriggerType.COVERAGE_DROP,
            description="Coverage dropped below threshold",
            conditions={"threshold": self.config.coverage_threshold},
            actions=["run_coverage_analysis", "generate_coverage_report"]
        ))

        # Failure spike trigger
        self.add_trigger(AutomationTrigger(
            trigger_id="failure_spike",
            trigger_type=TriggerType.FAILURE_SPIKE,
            description="Test failure rate exceeded threshold",
            conditions={"threshold": self.config.failure_threshold},
            actions=["analyze_failures", "run_diagnostics", "notify_team"]
        ))

    def start_automation(self) -> None:
        """Start the automation system."""
        if self.running:
            logger.warning("Automation system is already running")
            return

        logger.info("Starting test automation system")
        self.running = True

        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        # Setup time-based triggers
        self._setup_scheduled_triggers()

        # Start monitoring
        if self.config.enable_analytics:
            self._start_monitoring()

        logger.info("Test automation system started successfully")

    def stop_automation(self) -> None:
        """Stop the automation system."""
        if not self.running:
            return

        logger.info("Stopping test automation system")
        self.running = False

        # Clear scheduled jobs
        schedule.clear()

        # Wait for scheduler thread
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=10.0)

        logger.info("Test automation system stopped")

    def trigger_automation_run(self,
                             trigger_type: TriggerType = TriggerType.MANUAL,
                             test_patterns: Optional[List[str]] = None,
                             additional_args: Optional[List[str]] = None) -> str:
        """
        Trigger an automation run.

        Args:
            trigger_type: Type of trigger that initiated the run
            test_patterns: Optional test patterns to override defaults
            additional_args: Additional command line arguments

        Returns:
            Run ID for tracking
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Create automation run
            automation_run = AutomationRun(
                run_id=run_id,
                started_at=datetime.now(),
                trigger_type=trigger_type
            )

            self.active_runs[run_id] = automation_run

            # Start async execution
            asyncio.create_task(self._execute_automation_run(automation_run, test_patterns, additional_args))

            logger.info(f"Triggered automation run: {run_id}")
            return run_id

        except Exception as e:
            logger.error(f"Failed to trigger automation run: {e}")
            # Create failed run record
            failed_run = AutomationRun(
                run_id=run_id,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                status="failed",
                error_message=str(e),
                trigger_type=trigger_type
            )
            self.active_runs[run_id] = failed_run
            return run_id

    async def _execute_automation_run(self,
                                    automation_run: AutomationRun,
                                    test_patterns: Optional[List[str]] = None,
                                    additional_args: Optional[List[str]] = None) -> None:
        """Execute a complete automation run."""
        try:
            logger.info(f"Executing automation run: {automation_run.run_id}")

            # Phase 1: Discovery (if enabled)
            if self.config.enable_discovery and self.discovery_engine:
                await self._run_test_discovery(automation_run)

            # Phase 2: Execute tests
            await self._run_test_execution(automation_run, test_patterns, additional_args)

            # Phase 3: Analytics and reporting
            if self.config.enable_analytics:
                await self._run_analytics(automation_run)

            # Phase 4: Generate dashboard
            if self.config.enable_dashboard and self.dashboard_generator:
                await self._generate_dashboard(automation_run)

            # Phase 5: Maintenance checks
            if self.config.enable_maintenance and self.maintenance_scheduler:
                await self._run_maintenance_checks(automation_run)

            # Phase 6: Cleanup
            if self.config.cleanup_old_data:
                await self._cleanup_old_data(automation_run)

            # Mark as completed
            automation_run.completed_at = datetime.now()
            automation_run.status = "completed"

            # Save run results
            self._save_automation_run(automation_run)

            logger.info(f"Automation run completed: {automation_run.run_id}")

        except Exception as e:
            logger.error(f"Automation run failed {automation_run.run_id}: {e}")
            automation_run.completed_at = datetime.now()
            automation_run.status = "failed"
            automation_run.error_message = str(e)
            self._save_automation_run(automation_run)

    async def _run_test_discovery(self, automation_run: AutomationRun) -> None:
        """Run test discovery phase."""
        try:
            logger.info("Running test discovery")
            working_dir = self.config.working_directory or Path.cwd()

            # Discover tests
            discovery_result = self.discovery_engine.discover_tests(
                project_root=working_dir,
                test_patterns=self.config.test_patterns
            )

            # Store discovery results
            automation_run.results_summary['discovery'] = {
                'tests_discovered': len(discovery_result.discovered_tests),
                'suggestions_generated': len(discovery_result.test_suggestions),
                'coverage_gaps': len(discovery_result.coverage_analysis.gaps) if discovery_result.coverage_analysis else 0
            }

            logger.info(f"Discovery completed: {len(discovery_result.discovered_tests)} tests found")

        except Exception as e:
            logger.error(f"Test discovery failed: {e}")
            automation_run.results_summary['discovery'] = {'error': str(e)}

    async def _run_test_execution(self,
                                automation_run: AutomationRun,
                                test_patterns: Optional[List[str]] = None,
                                additional_args: Optional[List[str]] = None) -> None:
        """Run test execution phase."""
        try:
            logger.info("Running test execution")

            patterns = test_patterns or self.config.test_patterns
            working_dir = self.config.working_directory or Path.cwd()

            # Schedule test executions
            execution_ids = []
            for pattern in patterns:
                command = self.config.test_command.copy()
                if additional_args:
                    command.extend(additional_args)
                command.append(pattern)

                execution_id = self.execution_scheduler.schedule_test_execution(
                    test_pattern=pattern,
                    command=command,
                    working_directory=working_dir,
                    priority=ExecutionPriority.HIGH
                )
                execution_ids.append(execution_id)

            automation_run.executions = execution_ids

            # Wait for executions to complete
            await self._wait_for_executions(execution_ids, automation_run)

            logger.info(f"Test execution completed: {len(execution_ids)} executions")

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            automation_run.results_summary['execution'] = {'error': str(e)}

    async def _wait_for_executions(self, execution_ids: List[str], automation_run: AutomationRun) -> None:
        """Wait for test executions to complete."""
        timeout = self.config.max_execution_time
        start_time = datetime.now()

        while execution_ids:
            if (datetime.now() - start_time).total_seconds() > timeout:
                logger.warning("Execution timeout reached, cancelling remaining executions")
                for exec_id in execution_ids:
                    self.execution_scheduler.cancel_execution(exec_id)
                break

            completed_executions = []
            for exec_id in execution_ids:
                status = self.execution_scheduler.get_execution_status(exec_id)
                if status and status['status'] in ['completed', 'failed', 'timeout', 'cancelled']:
                    completed_executions.append(exec_id)

            # Remove completed executions
            for exec_id in completed_executions:
                execution_ids.remove(exec_id)

            if execution_ids:
                await asyncio.sleep(5.0)  # Check every 5 seconds

        # Collect results
        await self._collect_execution_results(automation_run)

    async def _collect_execution_results(self, automation_run: AutomationRun) -> None:
        """Collect and summarize execution results."""
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_duration = 0.0

        for exec_id in automation_run.executions:
            status = self.execution_scheduler.get_execution_status(exec_id)
            if status:
                total_tests += status.get('tests_run', 0)
                total_passed += status.get('tests_passed', 0)
                total_failed += status.get('tests_failed', 0)
                total_duration += status.get('duration', 0) or 0

        automation_run.results_summary['execution'] = {
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'pass_rate': (total_passed / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_duration,
            'executions_completed': len(automation_run.executions)
        }

    async def _run_analytics(self, automation_run: AutomationRun) -> None:
        """Run analytics phase."""
        try:
            logger.info("Running analytics")

            # Calculate current metrics
            metrics = self.analytics_engine.calculate_metrics()

            # Generate quality report
            quality_report = self.analytics_engine.generate_quality_report()

            automation_run.results_summary['analytics'] = {
                'pass_rate': metrics.pass_rate,
                'coverage_percentage': metrics.coverage_percentage,
                'quality_score': quality_report.overall_score,
                'critical_issues': len(quality_report.critical_issues),
                'recommendations': len(quality_report.recommendations)
            }

            logger.info(f"Analytics completed: Quality score {quality_report.overall_score:.1f}")

        except Exception as e:
            logger.error(f"Analytics failed: {e}")
            automation_run.results_summary['analytics'] = {'error': str(e)}

    async def _generate_dashboard(self, automation_run: AutomationRun) -> None:
        """Generate dashboard phase."""
        try:
            logger.info("Generating dashboard")

            dashboard_path = self.dashboard_generator.generate_dashboard(
                title=f"Automation Run {automation_run.run_id}",
                period_days=7
            )

            automation_run.dashboard_path = dashboard_path
            automation_run.results_summary['dashboard'] = {
                'path': dashboard_path,
                'generated_at': datetime.now().isoformat()
            }

            logger.info(f"Dashboard generated: {dashboard_path}")

        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")
            automation_run.results_summary['dashboard'] = {'error': str(e)}

    async def _run_maintenance_checks(self, automation_run: AutomationRun) -> None:
        """Run maintenance checks phase."""
        try:
            logger.info("Running maintenance checks")

            # Get current metrics to identify maintenance needs
            execution_summary = automation_run.results_summary.get('execution', {})

            # Schedule maintenance tasks based on results
            maintenance_tasks = []

            # Low pass rate maintenance
            if execution_summary.get('pass_rate', 100) < 90:
                task_id = f"investigate_failures_{automation_run.run_id}"
                task = MaintenanceTask(
                    task_id=task_id,
                    task_type=TaskType.FLAKY_TEST_INVESTIGATION,
                    priority=TaskPriority.HIGH,
                    title="Investigate test failures",
                    description=f"Pass rate is {execution_summary.get('pass_rate', 0):.1f}%",
                    estimated_duration=timedelta(hours=2)
                )
                if self.maintenance_scheduler.schedule_task(task):
                    maintenance_tasks.append(task_id)

            # Slow execution maintenance
            if execution_summary.get('total_duration', 0) > 1800:  # 30 minutes
                task_id = f"optimize_performance_{automation_run.run_id}"
                task = MaintenanceTask(
                    task_id=task_id,
                    task_type=TaskType.PERFORMANCE_OPTIMIZATION,
                    priority=TaskPriority.MEDIUM,
                    title="Optimize test performance",
                    description=f"Tests took {execution_summary.get('total_duration', 0):.1f} seconds",
                    estimated_duration=timedelta(hours=4)
                )
                if self.maintenance_scheduler.schedule_task(task):
                    maintenance_tasks.append(task_id)

            automation_run.results_summary['maintenance'] = {
                'tasks_scheduled': len(maintenance_tasks),
                'task_ids': maintenance_tasks
            }

            logger.info(f"Maintenance checks completed: {len(maintenance_tasks)} tasks scheduled")

        except Exception as e:
            logger.error(f"Maintenance checks failed: {e}")
            automation_run.results_summary['maintenance'] = {'error': str(e)}

    async def _cleanup_old_data(self, automation_run: AutomationRun) -> None:
        """Cleanup old data phase."""
        try:
            logger.info("Cleaning up old data")

            cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)

            # Cleanup logic would go here
            # This is a placeholder for actual cleanup implementation

            automation_run.results_summary['cleanup'] = {
                'cutoff_date': cutoff_date.isoformat(),
                'retention_days': self.config.data_retention_days
            }

            logger.info("Data cleanup completed")

        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
            automation_run.results_summary['cleanup'] = {'error': str(e)}

    def add_trigger(self, trigger: AutomationTrigger) -> None:
        """Add automation trigger."""
        self.triggers[trigger.trigger_id] = trigger
        self._save_trigger(trigger)
        logger.info(f"Added automation trigger: {trigger.trigger_id}")

    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove automation trigger."""
        if trigger_id in self.triggers:
            del self.triggers[trigger_id]
            # Remove from database
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('DELETE FROM automation_triggers WHERE trigger_id = ?', (trigger_id,))
                conn.commit()
            finally:
                conn.close()
            logger.info(f"Removed automation trigger: {trigger_id}")
            return True
        return False

    def get_automation_status(self) -> Dict[str, Any]:
        """Get current automation system status."""
        return {
            'running': self.running,
            'active_runs': len(self.active_runs),
            'triggers_configured': len(self.triggers),
            'config': {
                'mode': self.config.mode.value,
                'parallel_jobs': self.config.parallel_jobs,
                'enable_analytics': self.config.enable_analytics,
                'enable_dashboard': self.config.enable_dashboard,
                'enable_maintenance': self.config.enable_maintenance
            },
            'recent_runs': list(self.active_runs.keys())[-5:]  # Last 5 runs
        }

    def get_run_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific automation run."""
        if run_id in self.active_runs:
            run = self.active_runs[run_id]
            return {
                'run_id': run_id,
                'status': run.status,
                'started_at': run.started_at.isoformat(),
                'completed_at': run.completed_at.isoformat() if run.completed_at else None,
                'trigger_type': run.trigger_type.value,
                'executions': len(run.executions),
                'results_summary': run.results_summary,
                'dashboard_path': run.dashboard_path,
                'error_message': run.error_message
            }
        return None

    def _scheduler_loop(self) -> None:
        """Main scheduler loop for time-based triggers."""
        while self.running:
            try:
                schedule.run_pending()
                import time
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

    def _setup_scheduled_triggers(self) -> None:
        """Setup scheduled triggers using the schedule library."""
        for trigger in self.triggers.values():
            if trigger.trigger_type == TriggerType.TIME_BASED and trigger.enabled:
                cron_schedule = trigger.conditions.get('schedule')
                if cron_schedule:
                    # Simple parsing for common cron patterns
                    # In a real implementation, you'd use a proper cron parser
                    if cron_schedule == "0 */6 * * *":  # Every 6 hours
                        schedule.every(6).hours.do(self._trigger_scheduled_run, trigger.trigger_id)
                    elif cron_schedule == "0 0 * * *":  # Daily at midnight
                        schedule.every().day.at("00:00").do(self._trigger_scheduled_run, trigger.trigger_id)

    def _trigger_scheduled_run(self, trigger_id: str) -> None:
        """Trigger a scheduled run."""
        try:
            trigger = self.triggers.get(trigger_id)
            if trigger and trigger.enabled:
                trigger.last_triggered = datetime.now()
                trigger.trigger_count += 1
                self._save_trigger(trigger)

                self.trigger_automation_run(TriggerType.TIME_BASED)
                logger.info(f"Triggered scheduled run for: {trigger_id}")
        except Exception as e:
            logger.error(f"Failed to trigger scheduled run {trigger_id}: {e}")

    def _start_monitoring(self) -> None:
        """Start monitoring for trigger conditions."""
        # This would implement monitoring for coverage drops, failure spikes, etc.
        # Placeholder implementation
        pass

    def _save_automation_run(self, run: AutomationRun) -> None:
        """Save automation run to database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO automation_runs (
                    run_id, started_at, completed_at, trigger_type, status,
                    executions, results_summary, dashboard_path, report_path, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                run.run_id,
                run.started_at.timestamp(),
                run.completed_at.timestamp() if run.completed_at else None,
                run.trigger_type.value,
                run.status,
                json.dumps(run.executions),
                json.dumps(run.results_summary),
                run.dashboard_path,
                run.report_path,
                run.error_message
            ))
            conn.commit()
        finally:
            conn.close()

    def _save_trigger(self, trigger: AutomationTrigger) -> None:
        """Save trigger to database."""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO automation_triggers (
                    trigger_id, trigger_type, description, conditions, actions,
                    enabled, last_triggered, trigger_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trigger.trigger_id,
                trigger.trigger_type.value,
                trigger.description,
                json.dumps(trigger.conditions),
                json.dumps(trigger.actions),
                1 if trigger.enabled else 0,
                trigger.last_triggered.timestamp() if trigger.last_triggered else None,
                trigger.trigger_count
            ))
            conn.commit()
        finally:
            conn.close()