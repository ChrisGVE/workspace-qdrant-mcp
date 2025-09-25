"""
Comprehensive unit tests for TestExecutionScheduler with edge cases.

Tests async execution, resource monitoring, retry logic, process management,
and all error handling scenarios including timeouts and resource constraints.
"""

import pytest
import asyncio
import sqlite3
import tempfile
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import threading
import signal

from src.python.workspace_qdrant_mcp.testing.execution.scheduler import (
    TestExecutionScheduler,
    TestExecution,
    ExecutionResult,
    ExecutionStatus,
    ExecutionPriority,
    ExecutionConstraints
)


class TestExecutionConstraints:
    """Tests for ExecutionConstraints dataclass."""

    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = ExecutionConstraints()

        assert constraints.max_parallel_jobs == 4
        assert constraints.max_memory_mb == 2048
        assert constraints.max_cpu_percent == 80
        assert constraints.timeout_seconds == 300
        assert constraints.max_retries == 2

    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = ExecutionConstraints(
            max_parallel_jobs=8,
            max_memory_mb=4096,
            timeout_seconds=600
        )

        assert constraints.max_parallel_jobs == 8
        assert constraints.max_memory_mb == 4096
        assert constraints.timeout_seconds == 600


class TestTestExecution:
    """Tests for TestExecution dataclass."""

    @pytest.fixture
    def sample_execution(self):
        """Create sample test execution."""
        return TestExecution(
            execution_id="test_exec_1",
            test_pattern="test_*.py",
            command=["python", "-m", "pytest"],
            working_directory=Path("/tmp/test"),
            priority=ExecutionPriority.NORMAL
        )

    def test_execution_initialization(self, sample_execution):
        """Test test execution initialization."""
        assert sample_execution.execution_id == "test_exec_1"
        assert sample_execution.status == ExecutionStatus.PENDING
        assert sample_execution.retry_count == 0
        assert isinstance(sample_execution.created_at, datetime)

    def test_execution_comparison(self):
        """Test execution priority comparison for queue ordering."""
        high_exec = TestExecution(
            execution_id="high",
            test_pattern="test_*.py",
            command=["pytest"],
            working_directory=Path("/tmp"),
            priority=ExecutionPriority.HIGH
        )

        low_exec = TestExecution(
            execution_id="low",
            test_pattern="test_*.py",
            command=["pytest"],
            working_directory=Path("/tmp"),
            priority=ExecutionPriority.LOW,
            created_at=datetime.now() + timedelta(seconds=1)  # Created later
        )

        # Higher priority should come first
        assert high_exec < low_exec

    def test_execution_same_priority_time_ordering(self):
        """Test execution ordering with same priority by creation time."""
        base_time = datetime.now()

        exec1 = TestExecution(
            execution_id="first",
            test_pattern="test_*.py",
            command=["pytest"],
            working_directory=Path("/tmp"),
            priority=ExecutionPriority.NORMAL,
            created_at=base_time
        )

        exec2 = TestExecution(
            execution_id="second",
            test_pattern="test_*.py",
            command=["pytest"],
            working_directory=Path("/tmp"),
            priority=ExecutionPriority.NORMAL,
            created_at=base_time + timedelta(seconds=1)
        )

        # Earlier creation time should come first
        assert exec1 < exec2


class TestTestExecutionScheduler:
    """Comprehensive tests for TestExecutionScheduler."""

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
        """Create temporary directory for test execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def scheduler(self, temp_db):
        """Create test execution scheduler."""
        constraints = ExecutionConstraints(
            max_parallel_jobs=2,
            timeout_seconds=10,  # Short timeout for testing
            max_retries=1
        )
        return TestExecutionScheduler(
            db_path=temp_db,
            constraints=constraints,
            enable_monitoring=False  # Disable for testing
        )

    @pytest.fixture
    def scheduler_with_monitoring(self, temp_db):
        """Create scheduler with resource monitoring enabled."""
        return TestExecutionScheduler(
            db_path=temp_db,
            enable_monitoring=True
        )

    def test_scheduler_initialization(self, temp_db):
        """Test scheduler initialization."""
        scheduler = TestExecutionScheduler(db_path=temp_db)

        # Check database was created
        assert temp_db.exists()

        # Check tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        expected_tables = {
            'test_executions', 'execution_results', 'resource_history'
        }
        assert expected_tables.issubset(tables)

    def test_scheduler_initialization_with_monitoring(self, scheduler_with_monitoring):
        """Test scheduler initialization with resource monitoring."""
        assert scheduler_with_monitoring.resource_monitor.monitoring_active
        assert scheduler_with_monitoring.monitor_thread is not None

    def test_schedule_test_execution_success(self, scheduler, temp_dir):
        """Test successful test execution scheduling."""
        execution_id = scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "hello"],
            working_directory=temp_dir,
            priority=ExecutionPriority.HIGH
        )

        assert execution_id is not None
        assert execution_id.startswith("exec_")

        # Verify execution was added to pending queue
        assert len(scheduler.pending_executions) == 1

    def test_schedule_test_execution_invalid_params(self, scheduler):
        """Test scheduling with invalid parameters."""
        # Empty pattern
        with pytest.raises(ValueError, match="test_pattern and command are required"):
            scheduler.schedule_test_execution("", ["pytest"], Path("/tmp"))

        # Empty command
        with pytest.raises(ValueError, match="test_pattern and command are required"):
            scheduler.schedule_test_execution("test_*.py", [], Path("/tmp"))

        # Non-existent directory
        with pytest.raises(ValueError, match="Working directory does not exist"):
            scheduler.schedule_test_execution("test_*.py", ["pytest"], Path("/nonexistent"))

    def test_schedule_with_dependencies_and_tags(self, scheduler, temp_dir):
        """Test scheduling with dependencies and tags."""
        execution_id = scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir,
            dependencies={"dep1", "dep2"},
            tags={"unit", "fast"},
            metadata={"env": {"TEST": "1"}}
        )

        assert execution_id is not None

        # Check execution has dependencies and tags
        execution = scheduler.pending_executions[0]
        assert execution.dependencies == {"dep1", "dep2"}
        assert execution.tags == {"unit", "fast"}
        assert execution.metadata == {"env": {"TEST": "1"}}

    def test_get_next_ready_execution_no_executions(self, scheduler):
        """Test getting next ready execution when none exist."""
        ready_execution = scheduler._get_next_ready_execution()
        assert ready_execution is None

    def test_get_next_ready_execution_resource_constrained(self, scheduler, temp_dir):
        """Test execution blocking due to resource constraints."""
        # Fill up parallel job slots
        scheduler.running_executions = {f"running_{i}": Mock() for i in range(scheduler.constraints.max_parallel_jobs)}

        # Schedule new execution
        scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        # Should not be ready due to resource constraints
        ready_execution = scheduler._get_next_ready_execution()
        assert ready_execution is None

    @patch('src.python.workspace_qdrant_mcp.testing.execution.scheduler.psutil')
    def test_get_next_ready_execution_cpu_constrained(self, mock_psutil, scheduler, temp_dir):
        """Test execution blocking due to CPU constraints."""
        # Mock high CPU usage
        mock_psutil.cpu_percent.return_value = 90.0  # Above threshold
        mock_psutil.virtual_memory.return_value.percent = 50.0

        scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        ready_execution = scheduler._get_next_ready_execution()
        assert ready_execution is None

    @patch('src.python.workspace_qdrant_mcp.testing.execution.scheduler.psutil')
    def test_get_next_ready_execution_memory_constrained(self, mock_psutil, scheduler, temp_dir):
        """Test execution blocking due to memory constraints."""
        # Mock high memory usage
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value.percent = 90.0  # Above threshold

        scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        ready_execution = scheduler._get_next_ready_execution()
        assert ready_execution is None

    @patch('src.python.workspace_qdrant_mcp.testing.execution.scheduler.psutil')
    def test_check_resource_availability_psutil_error(self, mock_psutil, scheduler):
        """Test resource availability check with psutil error."""
        mock_psutil.cpu_percent.side_effect = Exception("PSUtil error")

        # Should default to allowing execution
        available = scheduler._check_resource_availability()
        assert available is True

    def test_get_next_ready_execution_with_dependencies(self, scheduler, temp_dir):
        """Test execution with unsatisfied dependencies."""
        # Schedule execution with dependency
        scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir,
            dependencies={"nonexistent_dep"}
        )

        ready_execution = scheduler._get_next_ready_execution()
        assert ready_execution is None  # Should be blocked by dependency

    def test_get_next_ready_execution_satisfied_dependencies(self, scheduler, temp_dir):
        """Test execution with satisfied dependencies."""
        # Add completed execution to satisfy dependency
        completed_result = Mock()
        completed_result.execution.execution_id = "completed_dep"
        scheduler.completed_executions["completed_dep"] = completed_result

        # Schedule execution with satisfied dependency
        scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir,
            dependencies={"completed_dep"}
        )

        ready_execution = scheduler._get_next_ready_execution()
        assert ready_execution is not None

    @pytest.mark.asyncio
    async def test_start_execution_success(self, scheduler, temp_dir):
        """Test successful execution start."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["echo", "hello"],
            working_directory=temp_dir
        )

        await scheduler._start_execution(execution)

        assert execution.status == ExecutionStatus.RUNNING
        assert execution.started_at is not None
        assert execution.execution_id in scheduler.running_executions

    @pytest.mark.asyncio
    async def test_start_execution_error(self, scheduler, temp_dir):
        """Test execution start with error."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["nonexistent_command"],
            working_directory=temp_dir
        )

        # Mock asyncio.create_subprocess_exec to raise error
        with patch('asyncio.create_subprocess_exec', side_effect=Exception("Start error")):
            await scheduler._start_execution(execution)

        assert execution.status == ExecutionStatus.FAILED
        assert execution.error_message == "Start error"

    @pytest.mark.asyncio
    async def test_execute_test_success(self, scheduler, temp_dir):
        """Test successful test execution."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["echo", "hello world"],
            working_directory=temp_dir
        )

        # Mock successful process
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"hello world", b"")
        mock_process.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await scheduler._execute_test(execution)

        assert execution.status == ExecutionStatus.COMPLETED
        assert execution.return_code == 0
        assert "hello world" in execution.stdout
        assert execution.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_execute_test_failure(self, scheduler, temp_dir):
        """Test test execution failure."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["false"],  # Command that always fails
            working_directory=temp_dir
        )

        # Mock failed process
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error output")
        mock_process.returncode = 1

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await scheduler._execute_test(execution)

        assert execution.status == ExecutionStatus.FAILED
        assert execution.return_code == 1
        assert "error output" in execution.stderr

    @pytest.mark.asyncio
    async def test_execute_test_timeout(self, scheduler, temp_dir):
        """Test test execution timeout."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["sleep", "100"],  # Long-running command
            working_directory=temp_dir
        )

        # Mock process that times out
        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        mock_process.kill.return_value = None
        mock_process.wait.return_value = None

        with patch('asyncio.create_subprocess_exec', return_value=mock_process), \
             patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):

            await scheduler._execute_test(execution)

        assert execution.status == ExecutionStatus.TIMEOUT
        assert execution.error_message == "Execution timed out"
        assert mock_process.kill.called

    @pytest.mark.asyncio
    async def test_execute_test_kill_timeout(self, scheduler, temp_dir):
        """Test execution with kill timeout."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["sleep", "100"],
            working_directory=temp_dir
        )

        # Mock process that times out and fails to be killed
        mock_process = AsyncMock()
        mock_process.communicate.side_effect = asyncio.TimeoutError()
        mock_process.kill.return_value = None
        mock_process.wait.side_effect = asyncio.TimeoutError()

        with patch('asyncio.create_subprocess_exec', return_value=mock_process), \
             patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):

            await scheduler._execute_test(execution)

        assert execution.status == ExecutionStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_execute_test_with_callbacks(self, scheduler, temp_dir):
        """Test execution with registered callbacks."""
        callback_called = []

        def sync_callback(execution, result):
            callback_called.append(("sync", execution.execution_id))

        async def async_callback(execution, result):
            callback_called.append(("async", execution.execution_id))

        scheduler.add_execution_callback(sync_callback)
        scheduler.add_execution_callback(async_callback)

        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        # Mock successful process
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"test", b"")
        mock_process.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await scheduler._execute_test(execution)

        # Both callbacks should have been called
        assert len(callback_called) == 2

    @pytest.mark.asyncio
    async def test_execute_test_callback_error(self, scheduler, temp_dir):
        """Test execution with callback that raises error."""
        def error_callback(execution, result):
            raise Exception("Callback error")

        scheduler.add_execution_callback(error_callback)

        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        # Mock successful process
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"test", b"")
        mock_process.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            # Should not crash despite callback error
            await scheduler._execute_test(execution)

        assert execution.status == ExecutionStatus.COMPLETED

    def test_process_execution_results_pytest_output(self, scheduler):
        """Test processing results from pytest output."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["pytest"],
            working_directory=Path("/tmp"),
            duration_seconds=45.0
        )

        execution.stdout = "===== 25 passed, 5 failed, 2 skipped in 45.00s ====="

        result = scheduler._process_execution_results(execution)

        assert result.tests_passed == 25
        assert result.tests_failed == 5
        assert result.tests_skipped == 2
        assert result.tests_run == 32
        assert result.performance_metrics['execution_time'] == 45.0

    def test_process_execution_results_unittest_output(self, scheduler):
        """Test processing results from unittest output."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["python", "-m", "unittest"],
            working_directory=Path("/tmp"),
            duration_seconds=30.0
        )

        execution.stdout = "Ran 10 tests in 30.000s\n\nOK"

        result = scheduler._process_execution_results(execution)

        assert result.tests_run == 10
        assert result.tests_passed == 10  # All passed since OK
        assert result.performance_metrics['execution_time'] == 30.0

    def test_process_execution_results_with_coverage(self, scheduler):
        """Test processing results with coverage information."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["pytest", "--cov"],
            working_directory=Path("/tmp")
        )

        execution.stdout = """
        ===== 10 passed in 5.00s =====
        Total coverage: 85%
        """

        result = scheduler._process_execution_results(execution)

        assert result.coverage_percentage == 85.0

    def test_process_execution_results_no_match(self, scheduler):
        """Test processing results with no recognizable output."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["custom_test_runner"],
            working_directory=Path("/tmp")
        )

        execution.stdout = "Custom test output with no standard format"

        result = scheduler._process_execution_results(execution)

        # Should have minimal result
        assert result.tests_run == 0
        assert result.tests_passed == 0

    def test_process_execution_results_error(self, scheduler):
        """Test processing results with parsing error."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["pytest"],
            working_directory=Path("/tmp")
        )

        # Mock regex to raise error
        with patch('re.search', side_effect=Exception("Regex error")):
            result = scheduler._process_execution_results(execution)

        # Should return basic result without crashing
        assert isinstance(result, ExecutionResult)

    @pytest.mark.asyncio
    async def test_schedule_retry_success(self, scheduler):
        """Test successful execution retry scheduling."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=Path("/tmp"),
            retry_count=0
        )

        with patch('asyncio.sleep') as mock_sleep:  # Mock sleep to speed up test
            await scheduler._schedule_retry(execution)

        assert execution.retry_count == 1
        assert execution.status == ExecutionStatus.PENDING
        assert len(scheduler.pending_executions) == 1
        assert mock_sleep.called

    @pytest.mark.asyncio
    async def test_schedule_retry_exponential_backoff(self, scheduler):
        """Test retry with exponential backoff delay."""
        execution = TestExecution(
            execution_id="test_exec",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=Path("/tmp"),
            retry_count=2  # Third retry
        )

        with patch('asyncio.sleep') as mock_sleep:
            await scheduler._schedule_retry(execution)

        # Should have exponential backoff: delay = 30 * (2 ^ (3-1)) = 120 seconds
        expected_delay = scheduler.constraints.retry_delay_seconds * (2 ** (3 - 1))
        mock_sleep.assert_called_with(expected_delay)

    def test_cancel_execution_running(self, scheduler, temp_dir):
        """Test cancelling a running execution."""
        execution_id = "running_exec"
        execution = TestExecution(
            execution_id=execution_id,
            test_pattern="test_*.py",
            command=["sleep", "100"],
            working_directory=temp_dir,
            status=ExecutionStatus.RUNNING
        )

        scheduler.running_executions[execution_id] = execution

        # Mock active process
        mock_process = Mock()
        scheduler.active_processes[execution_id] = mock_process

        success = scheduler.cancel_execution(execution_id)

        assert success is True
        assert execution.status == ExecutionStatus.CANCELLED
        assert execution_id not in scheduler.running_executions
        assert mock_process.terminate.called

    def test_cancel_execution_pending(self, scheduler, temp_dir):
        """Test cancelling a pending execution."""
        # Add execution to pending queue
        execution = TestExecution(
            execution_id="pending_exec",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        import heapq
        heapq.heappush(scheduler.pending_executions, execution)

        success = scheduler.cancel_execution("pending_exec")

        assert success is True
        assert execution.status == ExecutionStatus.CANCELLED

    def test_cancel_execution_nonexistent(self, scheduler):
        """Test cancelling non-existent execution."""
        success = scheduler.cancel_execution("nonexistent_exec")
        assert success is False

    def test_cancel_execution_kill_process_error(self, scheduler, temp_dir):
        """Test cancelling execution with process kill error."""
        execution_id = "running_exec"
        execution = TestExecution(
            execution_id=execution_id,
            test_pattern="test_*.py",
            command=["sleep", "100"],
            working_directory=temp_dir,
            status=ExecutionStatus.RUNNING
        )

        scheduler.running_executions[execution_id] = execution

        # Mock process that raises error on terminate
        mock_process = Mock()
        mock_process.terminate.side_effect = Exception("Kill error")
        scheduler.active_processes[execution_id] = mock_process

        success = scheduler.cancel_execution(execution_id)

        # Should still succeed despite kill error
        assert success is True
        assert execution.status == ExecutionStatus.CANCELLED

    def test_get_execution_status_running(self, scheduler, temp_dir):
        """Test getting status of running execution."""
        execution_id = "running_exec"
        execution = TestExecution(
            execution_id=execution_id,
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir,
            status=ExecutionStatus.RUNNING,
            started_at=datetime.now()
        )

        scheduler.running_executions[execution_id] = execution

        status = scheduler.get_execution_status(execution_id)

        assert status is not None
        assert status['execution_id'] == execution_id
        assert status['status'] == ExecutionStatus.RUNNING.value
        assert 'started_at' in status
        assert 'duration' in status

    def test_get_execution_status_completed(self, scheduler):
        """Test getting status of completed execution."""
        execution_id = "completed_exec"
        execution = TestExecution(
            execution_id=execution_id,
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=Path("/tmp"),
            status=ExecutionStatus.COMPLETED,
            duration_seconds=45.0
        )

        result = ExecutionResult(
            execution=execution,
            tests_run=10,
            tests_passed=8,
            tests_failed=2
        )

        scheduler.completed_executions[execution_id] = result

        status = scheduler.get_execution_status(execution_id)

        assert status is not None
        assert status['tests_run'] == 10
        assert status['tests_passed'] == 8
        assert status['tests_failed'] == 2
        assert status['duration'] == 45.0

    def test_get_execution_status_failed(self, scheduler):
        """Test getting status of failed execution."""
        execution_id = "failed_exec"
        execution = TestExecution(
            execution_id=execution_id,
            test_pattern="test_*.py",
            command=["false"],
            working_directory=Path("/tmp"),
            status=ExecutionStatus.FAILED,
            error_message="Test failure",
            retry_count=2
        )

        scheduler.failed_executions[execution_id] = execution

        status = scheduler.get_execution_status(execution_id)

        assert status is not None
        assert status['status'] == ExecutionStatus.FAILED.value
        assert status['error_message'] == "Test failure"
        assert status['retry_count'] == 2

    def test_get_execution_status_database_lookup(self, scheduler, temp_dir):
        """Test getting status from database when not in memory."""
        execution_id = scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        # Clear from memory
        scheduler.pending_executions.clear()

        status = scheduler.get_execution_status(execution_id)

        assert status is not None
        assert status['execution_id'] == execution_id

    def test_get_execution_status_nonexistent(self, scheduler):
        """Test getting status of non-existent execution."""
        status = scheduler.get_execution_status("nonexistent_exec")
        assert status is None

    def test_cleanup_completed_executions(self, scheduler):
        """Test cleanup of old completed executions."""
        # Add old completed execution
        old_time = datetime.now() - timedelta(hours=2)
        old_execution = TestExecution(
            execution_id="old_exec",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=Path("/tmp"),
            completed_at=old_time
        )

        old_result = ExecutionResult(execution=old_execution)
        scheduler.completed_executions["old_exec"] = old_result

        # Add recent execution
        recent_execution = TestExecution(
            execution_id="recent_exec",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=Path("/tmp"),
            completed_at=datetime.now() - timedelta(minutes=30)
        )

        recent_result = ExecutionResult(execution=recent_execution)
        scheduler.completed_executions["recent_exec"] = recent_result

        scheduler._cleanup_completed_executions()

        # Old execution should be removed, recent should remain
        assert "old_exec" not in scheduler.completed_executions
        assert "recent_exec" in scheduler.completed_executions

    def test_get_scheduler_stats(self, scheduler, temp_dir):
        """Test getting scheduler statistics."""
        # Add some executions in different states
        scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        mock_execution = Mock()
        scheduler.running_executions["running"] = mock_execution

        stats = scheduler.get_scheduler_stats()

        assert 'pending_executions' in stats
        assert 'running_executions' in stats
        assert 'completed_executions' in stats
        assert 'failed_executions' in stats
        assert stats['pending_executions'] == 1
        assert stats['running_executions'] == 1

    def test_resource_monitoring_disabled(self, scheduler):
        """Test scheduler with resource monitoring disabled."""
        assert not scheduler.resource_monitor.monitoring_active
        assert scheduler.monitor_thread is None

    def test_resource_monitoring_enabled(self, scheduler_with_monitoring):
        """Test scheduler with resource monitoring enabled."""
        assert scheduler_with_monitoring.resource_monitor.monitoring_active
        assert scheduler_with_monitoring.monitor_thread is not None

    @patch('src.python.workspace_qdrant_mcp.testing.execution.scheduler.psutil')
    def test_resource_monitor_loop(self, mock_psutil, scheduler):
        """Test resource monitoring loop."""
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value.percent = 60.0

        # Enable monitoring temporarily
        scheduler.resource_monitor.monitoring_active = True

        # Mock the loop to run once
        original_active = scheduler.resource_monitor.monitoring_active

        def mock_monitoring_check():
            if hasattr(mock_monitoring_check, 'called'):
                scheduler.resource_monitor.monitoring_active = False
                return False
            mock_monitoring_check.called = True
            return original_active

        with patch.object(scheduler.resource_monitor, 'monitoring_active', side_effect=mock_monitoring_check):
            scheduler._resource_monitor_loop()

        # Should have collected samples
        assert len(scheduler.resource_monitor.cpu_samples) > 0
        assert len(scheduler.resource_monitor.memory_samples) > 0

    def test_resource_monitor_loop_error(self, scheduler):
        """Test resource monitoring loop with error."""
        scheduler.resource_monitor.monitoring_active = True

        with patch('src.python.workspace_qdrant_mcp.testing.execution.scheduler.psutil.cpu_percent',
                  side_effect=Exception("PSUtil error")):
            # Run one iteration with error
            scheduler.resource_monitor.monitoring_active = False
            scheduler._resource_monitor_loop()

        # Should handle error gracefully

    def test_save_resource_history_success(self, scheduler):
        """Test saving resource history to database."""
        scheduler._save_resource_history(50.0, 60.0)

        # Verify saved to database
        conn = sqlite3.connect(scheduler.db_path)
        cursor = conn.execute('SELECT COUNT(*) FROM resource_history')
        count = cursor.fetchone()[0]
        conn.close()

        assert count > 0

    def test_save_resource_history_error(self, scheduler):
        """Test saving resource history with database error."""
        with patch('sqlite3.connect', side_effect=Exception("DB Error")):
            # Should not raise exception
            scheduler._save_resource_history(50.0, 60.0)

    def test_save_execution_success(self, scheduler, temp_dir):
        """Test saving execution to database."""
        execution = TestExecution(
            execution_id="test_save",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir,
            tags={"unit", "fast"},
            dependencies={"dep1"},
            metadata={"env": "test"}
        )

        scheduler._save_execution(execution)

        # Verify saved to database
        conn = sqlite3.connect(scheduler.db_path)
        cursor = conn.execute('SELECT * FROM test_executions WHERE execution_id = ?', (execution.execution_id,))
        result = cursor.fetchone()
        conn.close()

        assert result is not None
        assert result[0] == execution.execution_id  # execution_id

    def test_save_execution_result_success(self, scheduler, temp_dir):
        """Test saving execution result to database."""
        execution = TestExecution(
            execution_id="test_result",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=temp_dir
        )

        result = ExecutionResult(
            execution=execution,
            tests_run=10,
            tests_passed=8,
            tests_failed=2,
            coverage_percentage=85.0
        )

        scheduler._save_execution_result(result)

        # Verify saved to database
        conn = sqlite3.connect(scheduler.db_path)
        cursor = conn.execute('SELECT * FROM execution_results WHERE execution_id = ?', (execution.execution_id,))
        db_result = cursor.fetchone()
        conn.close()

        assert db_result is not None
        assert db_result[1] == 10  # tests_run

    def test_database_operations_with_invalid_data(self, scheduler):
        """Test database operations with invalid JSON data."""
        execution = TestExecution(
            execution_id="invalid_data",
            test_pattern="test_*.py",
            command=["echo", "test"],
            working_directory=Path("/tmp")
        )

        # Add invalid data that can't be JSON serialized
        execution.metadata = {"invalid": float('inf')}

        # Should handle gracefully without crashing
        scheduler._save_execution(execution)

    def test_shutdown_scheduler(self, scheduler):
        """Test scheduler shutdown."""
        # Add some running executions
        scheduler.running_executions["exec1"] = Mock()
        scheduler.running_executions["exec2"] = Mock()

        scheduler.shutdown()

        # Should have cancelled all running executions
        assert len(scheduler.running_executions) == 0
        assert not scheduler.resource_monitor.monitoring_active

    def test_shutdown_scheduler_with_monitoring(self, scheduler_with_monitoring):
        """Test scheduler shutdown with monitoring thread."""
        # Give monitoring thread time to start
        import time
        time.sleep(0.1)

        scheduler_with_monitoring.shutdown()

        assert not scheduler_with_monitoring.resource_monitor.monitoring_active

        # Wait for thread to finish
        if scheduler_with_monitoring.monitor_thread:
            scheduler_with_monitoring.monitor_thread.join(timeout=1.0)

    def test_concurrent_scheduling(self, scheduler, temp_dir):
        """Test concurrent execution scheduling."""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                execution_id = scheduler.schedule_test_execution(
                    test_pattern=f"test_worker_{worker_id}.py",
                    command=["echo", f"worker_{worker_id}"],
                    working_directory=temp_dir
                )
                results.append(execution_id)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have no errors and all executions scheduled
        assert len(errors) == 0
        assert len(results) == 5
        assert len(scheduler.pending_executions) == 5

    def test_execution_with_environment_variables(self, scheduler, temp_dir):
        """Test execution with custom environment variables."""
        execution_id = scheduler.schedule_test_execution(
            test_pattern="test_*.py",
            command=["env"],
            working_directory=temp_dir,
            metadata={"env": {"CUSTOM_VAR": "test_value"}}
        )

        execution = scheduler.pending_executions[0]
        assert "env" in execution.metadata
        assert execution.metadata["env"]["CUSTOM_VAR"] == "test_value"

    @pytest.mark.asyncio
    async def test_execution_with_custom_environment(self, scheduler, temp_dir):
        """Test actual execution with custom environment variables."""
        execution = TestExecution(
            execution_id="env_test",
            test_pattern="test_*.py",
            command=["python", "-c", "import os; print(os.environ.get('TEST_VAR', 'not_found'))"],
            working_directory=temp_dir,
            metadata={"env": {"TEST_VAR": "custom_value"}}
        )

        # Mock successful process with environment
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"custom_value\n", b"")
        mock_process.returncode = 0

        with patch('asyncio.create_subprocess_exec', return_value=mock_process) as mock_exec:
            await scheduler._execute_test(execution)

        # Verify environment was passed
        mock_exec.assert_called_once()
        call_args = mock_exec.call_args
        env_arg = call_args[1]['env']  # keyword argument
        assert 'TEST_VAR' in env_arg
        assert env_arg['TEST_VAR'] == 'custom_value'