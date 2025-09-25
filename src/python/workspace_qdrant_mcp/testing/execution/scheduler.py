"""
Test Execution Scheduler

Automated test execution scheduling with resource management, parallel execution,
result collection, and comprehensive error handling for execution failures.
"""

import logging
import subprocess
import asyncio
import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Callable, AsyncGenerator
from enum import Enum
import concurrent.futures
import threading
import signal
import os
import psutil
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of test execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ExecutionPriority(Enum):
    """Priority levels for test execution."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class ExecutionConstraints:
    """Resource constraints for test execution."""
    max_parallel_jobs: int = 4
    max_memory_mb: int = 2048
    max_cpu_percent: int = 80
    timeout_seconds: int = 300
    max_retries: int = 2
    retry_delay_seconds: int = 30
    kill_timeout_seconds: int = 10


@dataclass
class TestExecution:
    """Represents a test execution job."""
    execution_id: str
    test_pattern: str  # Pattern to match tests (e.g., "test_*.py", specific test file)
    command: List[str]  # Command to execute
    working_directory: Path
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    return_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    error_message: Optional[str] = None
    retry_count: int = 0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)  # Other execution IDs this depends on

    def __lt__(self, other):
        """Enable priority queue ordering."""
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)


@dataclass
class ExecutionResult:
    """Results of test execution."""
    execution: TestExecution
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    coverage_percentage: Optional[float] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[Path] = field(default_factory=list)


@dataclass
class ResourceMonitor:
    """Monitors system resources during execution."""
    cpu_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    memory_samples: deque = field(default_factory=lambda: deque(maxlen=100))
    last_update: Optional[datetime] = None
    monitoring_active: bool = False


class TestExecutionScheduler:
    """
    Advanced test execution scheduler with resource management.

    Provides automated test scheduling, parallel execution, resource monitoring,
    result collection, and comprehensive error handling with retry logic.
    """

    def __init__(self,
                 db_path: Path,
                 constraints: Optional[ExecutionConstraints] = None,
                 enable_monitoring: bool = True):
        """
        Initialize test execution scheduler.

        Args:
            db_path: Path to SQLite database for execution data
            constraints: Resource constraints for execution
            enable_monitoring: Enable system resource monitoring
        """
        self.db_path = db_path
        self.constraints = constraints or ExecutionConstraints()
        self.enable_monitoring = enable_monitoring

        # Execution state
        self.pending_executions = []  # Priority queue
        self.running_executions = {}  # execution_id -> TestExecution
        self.completed_executions = {}  # execution_id -> ExecutionResult
        self.failed_executions = {}  # execution_id -> TestExecution

        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
        self.monitor_thread = None
        self.shutdown_event = threading.Event()

        # Execution tracking
        self.execution_lock = threading.Lock()
        self.active_processes = {}  # execution_id -> subprocess.Popen
        self.execution_callbacks = []  # Callbacks for execution events

        self._initialize_db()
        if enable_monitoring:
            self._start_resource_monitoring()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS test_executions (
                    execution_id TEXT PRIMARY KEY,
                    test_pattern TEXT NOT NULL,
                    command TEXT NOT NULL,  -- JSON array
                    working_directory TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    duration_seconds REAL,
                    return_code INTEGER,
                    stdout TEXT,
                    stderr TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    resource_usage TEXT,  -- JSON object
                    metadata TEXT,  -- JSON object
                    tags TEXT,  -- JSON array
                    dependencies TEXT  -- JSON array
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS execution_results (
                    execution_id TEXT PRIMARY KEY,
                    tests_run INTEGER DEFAULT 0,
                    tests_passed INTEGER DEFAULT 0,
                    tests_failed INTEGER DEFAULT 0,
                    tests_skipped INTEGER DEFAULT 0,
                    coverage_percentage REAL,
                    performance_metrics TEXT,  -- JSON object
                    artifacts TEXT,  -- JSON array
                    FOREIGN KEY (execution_id) REFERENCES test_executions (execution_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS resource_history (
                    timestamp REAL NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    active_executions INTEGER,
                    PRIMARY KEY (timestamp)
                )
            ''')

            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_executions_status ON test_executions(status)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_executions_created ON test_executions(created_at)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_resource_timestamp ON resource_history(timestamp)')

            conn.commit()
        finally:
            conn.close()

    def schedule_test_execution(self,
                              test_pattern: str,
                              command: List[str],
                              working_directory: Union[str, Path],
                              priority: ExecutionPriority = ExecutionPriority.NORMAL,
                              tags: Optional[Set[str]] = None,
                              dependencies: Optional[Set[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Schedule a test execution.

        Args:
            test_pattern: Pattern identifying tests to run
            command: Command to execute tests
            working_directory: Directory to run tests in
            priority: Execution priority
            tags: Optional tags for categorization
            dependencies: Optional execution dependencies
            metadata: Optional metadata

        Returns:
            Execution ID for tracking

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate inputs
        if not test_pattern or not command:
            raise ValueError("test_pattern and command are required")

        working_dir = Path(working_directory)
        if not working_dir.exists():
            raise ValueError(f"Working directory does not exist: {working_dir}")

        # Generate execution ID
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.pending_executions)}"

        # Create execution object
        execution = TestExecution(
            execution_id=execution_id,
            test_pattern=test_pattern,
            command=command,
            working_directory=working_dir,
            priority=priority,
            tags=tags or set(),
            dependencies=dependencies or set(),
            metadata=metadata or {}
        )

        with self.execution_lock:
            # Add to pending queue
            import heapq
            heapq.heappush(self.pending_executions, execution)

            # Save to database
            self._save_execution(execution)

        logger.info(f"Scheduled test execution: {execution_id}")
        return execution_id

    def start_scheduler(self) -> None:
        """Start the execution scheduler."""
        logger.info("Starting test execution scheduler")
        asyncio.run(self._scheduler_loop())

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while not self.shutdown_event.is_set():
            try:
                # Check for ready executions
                ready_execution = self._get_next_ready_execution()

                if ready_execution:
                    # Start execution
                    await self._start_execution(ready_execution)

                # Clean up completed executions
                self._cleanup_completed_executions()

                # Wait before next iteration
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5.0)

    def _get_next_ready_execution(self) -> Optional[TestExecution]:
        """Get next execution ready to run."""
        with self.execution_lock:
            if not self.pending_executions:
                return None

            # Check resource constraints
            if len(self.running_executions) >= self.constraints.max_parallel_jobs:
                return None

            if not self._check_resource_availability():
                return None

            # Get highest priority execution
            import heapq
            next_execution = heapq.heappop(self.pending_executions)

            # Check dependencies
            if not self._are_dependencies_satisfied(next_execution):
                # Re-queue execution
                heapq.heappush(self.pending_executions, next_execution)
                return None

            return next_execution

    def _are_dependencies_satisfied(self, execution: TestExecution) -> bool:
        """Check if execution dependencies are satisfied."""
        if not execution.dependencies:
            return True

        for dep_id in execution.dependencies:
            if dep_id in self.running_executions:
                return False  # Still running
            if dep_id in self.failed_executions:
                return False  # Failed
            if dep_id not in self.completed_executions:
                return False  # Not completed

        return True

    def _check_resource_availability(self) -> bool:
        """Check if system resources are available."""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.constraints.max_cpu_percent:
                logger.debug(f"CPU usage too high: {cpu_percent}%")
                return False

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:  # Conservative threshold
                logger.debug(f"Memory usage too high: {memory.percent}%")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to check resource availability: {e}")
            return True  # Default to allowing execution

    async def _start_execution(self, execution: TestExecution) -> None:
        """Start executing a test."""
        try:
            execution.status = ExecutionStatus.RUNNING
            execution.started_at = datetime.now()

            with self.execution_lock:
                self.running_executions[execution.execution_id] = execution

            # Update database
            self._save_execution(execution)

            # Start execution process
            asyncio.create_task(self._execute_test(execution))

            logger.info(f"Started execution: {execution.execution_id}")

        except Exception as e:
            logger.error(f"Failed to start execution {execution.execution_id}: {e}")
            execution.status = ExecutionStatus.FAILED
            execution.error_message = str(e)
            self._handle_execution_failure(execution)

    async def _execute_test(self, execution: TestExecution) -> None:
        """Execute a test asynchronously."""
        process = None
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(execution.metadata.get('env', {}))

            # Start process
            process = await asyncio.create_subprocess_exec(
                *execution.command,
                cwd=execution.working_directory,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )

            # Store process for potential cancellation
            with self.execution_lock:
                self.active_processes[execution.execution_id] = process

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.constraints.timeout_seconds
                )

                execution.stdout = stdout.decode('utf-8', errors='ignore')
                execution.stderr = stderr.decode('utf-8', errors='ignore')
                execution.return_code = process.returncode

            except asyncio.TimeoutError:
                logger.warning(f"Execution timeout: {execution.execution_id}")
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=self.constraints.kill_timeout_seconds)
                except asyncio.TimeoutError:
                    logger.error(f"Failed to kill process for {execution.execution_id}")

                execution.status = ExecutionStatus.TIMEOUT
                execution.error_message = "Execution timed out"

            # Calculate duration
            execution.completed_at = datetime.now()
            execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()

            # Determine final status
            if execution.status != ExecutionStatus.TIMEOUT:
                if execution.return_code == 0:
                    execution.status = ExecutionStatus.COMPLETED
                else:
                    execution.status = ExecutionStatus.FAILED
                    if not execution.error_message:
                        execution.error_message = f"Process exited with code {execution.return_code}"

            # Process results
            result = self._process_execution_results(execution)

            # Move to appropriate completion queue
            with self.execution_lock:
                self.running_executions.pop(execution.execution_id, None)
                self.active_processes.pop(execution.execution_id, None)

                if execution.status == ExecutionStatus.COMPLETED:
                    self.completed_executions[execution.execution_id] = result
                else:
                    self.failed_executions[execution.execution_id] = execution

            # Save results
            self._save_execution(execution)
            if execution.status == ExecutionStatus.COMPLETED:
                self._save_execution_result(result)

            # Handle retries for failed executions
            if execution.status == ExecutionStatus.FAILED and execution.retry_count < self.constraints.max_retries:
                await self._schedule_retry(execution)

            # Notify callbacks
            for callback in self.execution_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(execution, result if execution.status == ExecutionStatus.COMPLETED else None)
                    else:
                        callback(execution, result if execution.status == ExecutionStatus.COMPLETED else None)
                except Exception as e:
                    logger.error(f"Callback error for {execution.execution_id}: {e}")

            logger.info(f"Completed execution: {execution.execution_id} - {execution.status.value}")

        except Exception as e:
            logger.error(f"Execution error for {execution.execution_id}: {e}")
            execution.status = ExecutionStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            if execution.started_at:
                execution.duration_seconds = (execution.completed_at - execution.started_at).total_seconds()

            self._handle_execution_failure(execution)

    def _process_execution_results(self, execution: TestExecution) -> ExecutionResult:
        """Process execution results and extract test metrics."""
        result = ExecutionResult(execution=execution)

        try:
            # Parse test output for metrics (simplified parsing)
            output = execution.stdout + execution.stderr

            # Common test result patterns
            import re

            # pytest style results
            pytest_match = re.search(r'(\d+) passed.*?(\d+) failed.*?(\d+) skipped', output)
            if pytest_match:
                result.tests_passed = int(pytest_match.group(1))
                result.tests_failed = int(pytest_match.group(2))
                result.tests_skipped = int(pytest_match.group(3))
                result.tests_run = result.tests_passed + result.tests_failed + result.tests_skipped

            # unittest style results
            unittest_match = re.search(r'Ran (\d+) tests.*?OK', output)
            if unittest_match:
                result.tests_run = int(unittest_match.group(1))
                result.tests_passed = result.tests_run  # Assuming all passed if OK

            # Coverage information
            coverage_match = re.search(r'Total coverage: (\d+)%', output)
            if coverage_match:
                result.coverage_percentage = float(coverage_match.group(1))

            # Performance metrics from execution
            if execution.duration_seconds:
                result.performance_metrics['execution_time'] = execution.duration_seconds
                result.performance_metrics['tests_per_second'] = result.tests_run / execution.duration_seconds if result.tests_run > 0 else 0

            # Resource usage
            if execution.resource_usage:
                result.performance_metrics.update(execution.resource_usage)

        except Exception as e:
            logger.error(f"Failed to process execution results for {execution.execution_id}: {e}")

        return result

    async def _schedule_retry(self, execution: TestExecution) -> None:
        """Schedule execution retry."""
        execution.retry_count += 1
        execution.status = ExecutionStatus.PENDING

        # Add delay before retry
        delay = self.constraints.retry_delay_seconds * (2 ** (execution.retry_count - 1))  # Exponential backoff
        logger.info(f"Scheduling retry {execution.retry_count} for {execution.execution_id} in {delay} seconds")

        await asyncio.sleep(delay)

        with self.execution_lock:
            import heapq
            heapq.heappush(self.pending_executions, execution)

    def _handle_execution_failure(self, execution: TestExecution) -> None:
        """Handle execution failure."""
        with self.execution_lock:
            self.running_executions.pop(execution.execution_id, None)
            self.active_processes.pop(execution.execution_id, None)
            self.failed_executions[execution.execution_id] = execution

        self._save_execution(execution)

    def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel a running or pending execution.

        Args:
            execution_id: ID of execution to cancel

        Returns:
            True if cancelled successfully
        """
        with self.execution_lock:
            # Check running executions
            if execution_id in self.running_executions:
                execution = self.running_executions[execution_id]
                execution.status = ExecutionStatus.CANCELLED

                # Kill process if running
                if execution_id in self.active_processes:
                    process = self.active_processes[execution_id]
                    try:
                        if hasattr(process, 'terminate'):
                            process.terminate()
                        else:
                            process.kill()
                    except Exception as e:
                        logger.error(f"Failed to kill process for {execution_id}: {e}")

                self.running_executions.pop(execution_id, None)
                self.active_processes.pop(execution_id, None)
                self._save_execution(execution)
                return True

            # Check pending executions
            import heapq
            pending_list = []
            found = False

            while self.pending_executions:
                execution = heapq.heappop(self.pending_executions)
                if execution.execution_id == execution_id:
                    execution.status = ExecutionStatus.CANCELLED
                    self._save_execution(execution)
                    found = True
                else:
                    pending_list.append(execution)

            # Rebuild heap
            self.pending_executions = pending_list
            heapq.heapify(self.pending_executions)

            return found

    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an execution."""
        # Check running executions
        if execution_id in self.running_executions:
            execution = self.running_executions[execution_id]
            return {
                'execution_id': execution_id,
                'status': execution.status.value,
                'started_at': execution.started_at.isoformat() if execution.started_at else None,
                'duration': (datetime.now() - execution.started_at).total_seconds() if execution.started_at else None
            }

        # Check completed executions
        if execution_id in self.completed_executions:
            result = self.completed_executions[execution_id]
            return {
                'execution_id': execution_id,
                'status': result.execution.status.value,
                'tests_run': result.tests_run,
                'tests_passed': result.tests_passed,
                'tests_failed': result.tests_failed,
                'duration': result.execution.duration_seconds
            }

        # Check failed executions
        if execution_id in self.failed_executions:
            execution = self.failed_executions[execution_id]
            return {
                'execution_id': execution_id,
                'status': execution.status.value,
                'error_message': execution.error_message,
                'retry_count': execution.retry_count
            }

        # Check database
        return self._get_execution_from_db(execution_id)

    def _cleanup_completed_executions(self) -> None:
        """Clean up old completed executions to free memory."""
        cutoff_time = datetime.now() - timedelta(hours=1)  # Keep last hour in memory

        with self.execution_lock:
            # Clean completed executions
            to_remove = [
                exec_id for exec_id, result in self.completed_executions.items()
                if result.execution.completed_at and result.execution.completed_at < cutoff_time
            ]
            for exec_id in to_remove:
                self.completed_executions.pop(exec_id, None)

            # Clean failed executions
            to_remove = [
                exec_id for exec_id, execution in self.failed_executions.items()
                if execution.completed_at and execution.completed_at < cutoff_time
            ]
            for exec_id in to_remove:
                self.failed_executions.pop(exec_id, None)

    def add_execution_callback(self, callback: Callable) -> None:
        """Add callback for execution events."""
        self.execution_callbacks.append(callback)

    def get_scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self.execution_lock:
            return {
                'pending_executions': len(self.pending_executions),
                'running_executions': len(self.running_executions),
                'completed_executions': len(self.completed_executions),
                'failed_executions': len(self.failed_executions),
                'resource_monitor_active': self.resource_monitor.monitoring_active,
                'last_resource_update': self.resource_monitor.last_update.isoformat() if self.resource_monitor.last_update else None
            }

    def _start_resource_monitoring(self) -> None:
        """Start resource monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return

        self.resource_monitor.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._resource_monitor_loop, daemon=True)
        self.monitor_thread.start()

    def _resource_monitor_loop(self) -> None:
        """Resource monitoring loop."""
        while self.resource_monitor.monitoring_active and not self.shutdown_event.is_set():
            try:
                # Sample resource usage
                cpu_percent = psutil.cpu_percent(interval=1.0)
                memory_percent = psutil.virtual_memory().percent

                self.resource_monitor.cpu_samples.append(cpu_percent)
                self.resource_monitor.memory_samples.append(memory_percent)
                self.resource_monitor.last_update = datetime.now()

                # Save to database periodically
                if len(self.resource_monitor.cpu_samples) % 10 == 0:  # Every 10 samples
                    self._save_resource_history(cpu_percent, memory_percent)

            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")

    def _save_resource_history(self, cpu_percent: float, memory_percent: float) -> None:
        """Save resource history to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            try:
                with self.execution_lock:
                    active_count = len(self.running_executions)

                conn.execute('''
                    INSERT INTO resource_history (timestamp, cpu_percent, memory_percent, active_executions)
                    VALUES (?, ?, ?, ?)
                ''', (datetime.now().timestamp(), cpu_percent, memory_percent, active_count))
                conn.commit()
            finally:
                conn.close()
        except Exception as e:
            logger.error(f"Failed to save resource history: {e}")

    def _save_execution(self, execution: TestExecution) -> None:
        """Save execution to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO test_executions (
                    execution_id, test_pattern, command, working_directory, priority, status,
                    created_at, started_at, completed_at, duration_seconds, return_code,
                    stdout, stderr, error_message, retry_count, resource_usage, metadata, tags, dependencies
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id,
                execution.test_pattern,
                json.dumps(execution.command),
                str(execution.working_directory),
                execution.priority.value,
                execution.status.value,
                execution.created_at.timestamp(),
                execution.started_at.timestamp() if execution.started_at else None,
                execution.completed_at.timestamp() if execution.completed_at else None,
                execution.duration_seconds,
                execution.return_code,
                execution.stdout,
                execution.stderr,
                execution.error_message,
                execution.retry_count,
                json.dumps(execution.resource_usage),
                json.dumps(execution.metadata),
                json.dumps(list(execution.tags)),
                json.dumps(list(execution.dependencies))
            ))
            conn.commit()
        finally:
            conn.close()

    def _save_execution_result(self, result: ExecutionResult) -> None:
        """Save execution result to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO execution_results (
                    execution_id, tests_run, tests_passed, tests_failed, tests_skipped,
                    coverage_percentage, performance_metrics, artifacts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.execution.execution_id,
                result.tests_run,
                result.tests_passed,
                result.tests_failed,
                result.tests_skipped,
                result.coverage_percentage,
                json.dumps(result.performance_metrics),
                json.dumps([str(p) for p in result.artifacts])
            ))
            conn.commit()
        finally:
            conn.close()

    def _get_execution_from_db(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution from database."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute('''
                SELECT status, created_at, started_at, completed_at, duration_seconds, error_message
                FROM test_executions WHERE execution_id = ?
            ''', (execution_id,))
            row = cursor.fetchone()

            if row:
                return {
                    'execution_id': execution_id,
                    'status': row[0],
                    'created_at': datetime.fromtimestamp(row[1]).isoformat(),
                    'started_at': datetime.fromtimestamp(row[2]).isoformat() if row[2] else None,
                    'completed_at': datetime.fromtimestamp(row[3]).isoformat() if row[3] else None,
                    'duration_seconds': row[4],
                    'error_message': row[5]
                }
        finally:
            conn.close()

        return None

    def shutdown(self) -> None:
        """Shutdown scheduler and cleanup resources."""
        logger.info("Shutting down test execution scheduler")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel running executions
        with self.execution_lock:
            for execution_id in list(self.running_executions.keys()):
                self.cancel_execution(execution_id)

        # Stop resource monitoring
        if self.resource_monitor.monitoring_active:
            self.resource_monitor.monitoring_active = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        logger.info("Scheduler shutdown complete")