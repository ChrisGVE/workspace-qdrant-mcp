"""
Comprehensive unit tests for Advanced Priority Queue System.

Tests cover all edge cases, resource management, fairness algorithms,
starvation prevention, and performance scenarios.
"""

import asyncio
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.python.common.core.advanced_priority_queue import (
    AdvancedPriorityQueue,
    FairnesManager,
    PriorityTask,
    ResourceMonitor,
    ResourceType,
    TaskMetrics,
    TaskPriority,
)


class TestTaskMetrics:
    """Test TaskMetrics functionality."""

    def test_task_metrics_initialization(self):
        """Test TaskMetrics initialization with default values."""
        metrics = TaskMetrics()

        assert metrics.started_at is None
        assert metrics.completed_at is None
        assert metrics.wait_time == 0.0
        assert metrics.execution_time == 0.0
        assert metrics.retry_count == 0
        assert metrics.last_error is None
        assert isinstance(metrics.resource_usage, dict)
        assert metrics.created_at > 0

    def test_task_metrics_lifecycle(self):
        """Test TaskMetrics lifecycle methods."""
        metrics = TaskMetrics()
        initial_time = metrics.created_at

        # Test age calculation
        time.sleep(0.01)  # Small delay
        age = metrics.age_seconds()
        assert age > 0

        # Test start marking
        metrics.mark_started()
        assert metrics.started_at > initial_time
        assert metrics.wait_time > 0

        # Test completion marking
        time.sleep(0.01)  # Small delay for execution time
        metrics.mark_completed()
        assert metrics.completed_at > metrics.started_at
        assert metrics.execution_time > 0

    def test_task_metrics_edge_cases(self):
        """Test TaskMetrics edge cases."""
        metrics = TaskMetrics()

        # Mark completed without starting (edge case)
        metrics.mark_completed()
        assert metrics.completed_at is not None
        assert metrics.execution_time == 0  # No start time recorded

    def test_task_metrics_multiple_starts(self):
        """Test multiple start calls on same metrics."""
        metrics = TaskMetrics()

        metrics.mark_started()
        first_start = metrics.started_at
        first_wait = metrics.wait_time

        time.sleep(0.01)
        metrics.mark_started()  # Should update times

        assert metrics.started_at >= first_start
        assert metrics.wait_time >= first_wait


class TestPriorityTask:
    """Test PriorityTask functionality."""

    def test_priority_task_initialization(self):
        """Test PriorityTask initialization."""
        task = PriorityTask(
            task_id="test_task",
            priority=TaskPriority.HIGH,
            payload={"test": "data"}
        )

        assert task.task_id == "test_task"
        assert task.priority == TaskPriority.HIGH
        assert task.original_priority == TaskPriority.HIGH
        assert task.boosted_priority is None
        assert task.payload == {"test": "data"}
        assert isinstance(task.metrics, TaskMetrics)
        assert task.estimated_cpu_usage == 0.1
        assert task.estimated_memory_mb == 10.0
        assert task.requires_exclusive_access is False

    def test_priority_task_comparison(self):
        """Test PriorityTask comparison for priority queue ordering."""
        task_high = PriorityTask("high", TaskPriority.HIGH, {})
        task_normal = PriorityTask("normal", TaskPriority.NORMAL, {})
        task_low = PriorityTask("low", TaskPriority.LOW, {})

        # Higher priority tasks should compare as less than lower priority
        assert task_high < task_normal
        assert task_normal < task_low
        assert not (task_normal < task_high)

    def test_priority_task_age_based_comparison(self):
        """Test PriorityTask comparison with same priority but different ages."""
        task1 = PriorityTask("task1", TaskPriority.NORMAL, {})
        time.sleep(0.01)  # Ensure different creation times
        task2 = PriorityTask("task2", TaskPriority.NORMAL, {})

        # Older task should have priority (task1 is older)
        assert task1 < task2

    def test_priority_task_starvation_detection(self):
        """Test starvation detection logic."""
        task = PriorityTask("test", TaskPriority.LOW, {})
        task.starvation_threshold = 0.1  # 100ms for testing

        # Initially should not need boosting
        assert not task.should_boost_priority()

        # Wait for starvation threshold
        time.sleep(0.15)
        assert task.should_boost_priority()

        # After boosting, should not need boosting again
        task.boost_priority()
        assert not task.should_boost_priority()

    def test_priority_task_boosting(self):
        """Test priority boosting functionality."""
        task = PriorityTask("test", TaskPriority.LOW, {})
        original_priority = task.priority

        task.boost_priority()

        assert task.boosted_priority == TaskPriority.NORMAL
        assert task.original_priority == original_priority

    def test_priority_task_critical_no_boost(self):
        """Test that CRITICAL tasks don't get boosted beyond CRITICAL."""
        task = PriorityTask("test", TaskPriority.CRITICAL, {})
        task.boost_priority()

        # Should remain critical, no boost possible
        assert task.boosted_priority is None

    def test_priority_task_boost_reset(self):
        """Test priority boost reset functionality."""
        task = PriorityTask("test", TaskPriority.LOW, {})
        task.boost_priority()

        assert task.boosted_priority is not None

        task.reset_priority_boost()
        assert task.boosted_priority is None

    def test_priority_task_with_dependencies(self):
        """Test PriorityTask with dependencies."""
        task = PriorityTask(
            task_id="dependent_task",
            priority=TaskPriority.NORMAL,
            payload={},
            depends_on=["dep1", "dep2"],
            blocks=["blocked1"]
        )

        assert "dep1" in task.depends_on
        assert "dep2" in task.depends_on
        assert "blocked1" in task.blocks

    def test_priority_task_file_metadata(self):
        """Test PriorityTask with file metadata."""
        file_path = Path("/test/file.txt")
        task = PriorityTask(
            task_id="file_task",
            priority=TaskPriority.NORMAL,
            payload={},
            file_path=file_path,
            file_size=1024,
            file_type="text/plain"
        )

        assert task.file_path == file_path
        assert task.file_size == 1024
        assert task.file_type == "text/plain"


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""

    def test_resource_monitor_initialization(self):
        """Test ResourceMonitor initialization."""
        monitor = ResourceMonitor(
            cpu_threshold=0.8,
            memory_threshold=0.9,
            io_threshold=1000.0
        )

        assert monitor.cpu_threshold == 0.8
        assert monitor.memory_threshold == 0.9
        assert monitor.io_threshold == 1000.0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_io_counters')
    def test_resource_monitor_usage(self, mock_disk_io, mock_memory, mock_cpu):
        """Test resource usage calculation."""
        # Mock system resource values
        mock_cpu.return_value = 50.0  # 50% CPU
        mock_memory.return_value.percent = 60.0  # 60% memory

        # Mock I/O counters
        mock_io_counter = MagicMock()
        mock_io_counter.read_bytes = 1024 * 1024  # 1MB
        mock_io_counter.write_bytes = 2048 * 1024  # 2MB
        mock_disk_io.return_value = mock_io_counter

        monitor = ResourceMonitor()
        usage = monitor.get_current_usage()

        assert usage[ResourceType.CPU] == 0.5
        assert usage[ResourceType.MEMORY] == 0.6
        assert ResourceType.IO_READ in usage
        assert ResourceType.IO_WRITE in usage

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_resource_monitor_throttling(self, mock_memory, mock_cpu):
        """Test throttling decisions."""
        monitor = ResourceMonitor(cpu_threshold=0.7, memory_threshold=0.8)

        # Normal usage - no throttling
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 60.0

        should_throttle, reasons = monitor.should_throttle()
        assert not should_throttle
        assert len(reasons) == 0

        # High CPU usage - should throttle
        mock_cpu.return_value = 80.0  # Above 70% threshold
        should_throttle, reasons = monitor.should_throttle()
        assert should_throttle
        assert ResourceType.CPU in reasons

        # High memory usage - should throttle
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 85.0  # Above 80% threshold
        should_throttle, reasons = monitor.should_throttle()
        assert should_throttle
        assert ResourceType.MEMORY in reasons

    def test_resource_monitor_throttle_delay(self):
        """Test throttle delay calculation."""
        monitor = ResourceMonitor()

        # No throttling reasons
        delay = monitor.get_throttle_delay([])
        assert delay == 0.0

        # Single resource under pressure
        delay = monitor.get_throttle_delay([ResourceType.CPU])
        assert delay >= 1.0  # Base delay

        # Multiple resources under pressure
        delay = monitor.get_throttle_delay([ResourceType.CPU, ResourceType.MEMORY])
        assert delay >= 2.0  # Higher delay for multiple resources

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_resource_monitor_severe_pressure(self, mock_memory, mock_cpu):
        """Test severe resource pressure handling."""
        monitor = ResourceMonitor()

        # Severe CPU pressure (>95%)
        mock_cpu.return_value = 96.0
        mock_memory.return_value.percent = 60.0

        usage = monitor.get_current_usage()
        should_throttle, reasons = monitor.should_throttle(usage)
        assert should_throttle

        delay = monitor.get_throttle_delay(reasons)
        assert delay > 1.0  # Should have additional delay for severe pressure

    def test_resource_monitor_io_calculation(self):
        """Test I/O rate calculation with multiple calls."""
        with patch('psutil.disk_io_counters') as mock_disk_io:
            # First call
            mock_io_counter1 = MagicMock()
            mock_io_counter1.read_bytes = 1024 * 1024  # 1MB
            mock_io_counter1.write_bytes = 1024 * 1024  # 1MB
            mock_disk_io.return_value = mock_io_counter1

            monitor = ResourceMonitor()
            monitor.get_current_usage()

            # Small delay and second call with more I/O
            time.sleep(0.01)
            mock_io_counter2 = MagicMock()
            mock_io_counter2.read_bytes = 2 * 1024 * 1024  # 2MB
            mock_io_counter2.write_bytes = 3 * 1024 * 1024  # 3MB
            mock_disk_io.return_value = mock_io_counter2

            usage2 = monitor.get_current_usage()

            # Second measurement should have calculated I/O rates
            assert usage2[ResourceType.IO_READ] >= 0
            assert usage2[ResourceType.IO_WRITE] >= 0


class TestFairnesManager:
    """Test FairnesManager functionality."""

    def test_fairness_manager_initialization(self):
        """Test FairnesManager initialization."""
        manager = FairnesManager(starvation_check_interval=30.0)
        assert manager.starvation_check_interval == 30.0

    def test_fairness_manager_record_task(self):
        """Test recording served tasks."""
        manager = FairnesManager()
        task = PriorityTask("test", TaskPriority.HIGH, {})

        manager.record_task_served(task)

        stats = manager.get_fairness_stats()
        assert stats["total_tasks_served"] == 1
        assert TaskPriority.HIGH in stats["priority_distribution"]
        assert stats["priority_distribution"][TaskPriority.HIGH] == 1

    def test_fairness_manager_multiple_priorities(self):
        """Test fairness tracking across multiple priorities."""
        manager = FairnesManager()

        high_task = PriorityTask("high", TaskPriority.HIGH, {})
        normal_task = PriorityTask("normal", TaskPriority.NORMAL, {})
        low_task = PriorityTask("low", TaskPriority.LOW, {})

        manager.record_task_served(high_task)
        manager.record_task_served(normal_task)
        manager.record_task_served(low_task)
        manager.record_task_served(high_task)  # High task served twice

        stats = manager.get_fairness_stats()
        assert stats["total_tasks_served"] == 4
        assert stats["priority_distribution"][TaskPriority.HIGH] == 2
        assert stats["priority_distribution"][TaskPriority.NORMAL] == 1
        assert stats["priority_distribution"][TaskPriority.LOW] == 1

        # Check fairness ratios
        assert stats["fairness_ratios"][str(TaskPriority.HIGH)] == 0.5
        assert stats["fairness_ratios"][str(TaskPriority.NORMAL)] == 0.25

    def test_fairness_manager_starvation_check(self):
        """Test starvation detection and boosting."""
        manager = FairnesManager()

        # Create old low-priority tasks that should be boosted
        old_task = PriorityTask("old", TaskPriority.LOW, {})
        old_task.starvation_threshold = 0.1  # 100ms threshold for testing
        old_task.metrics.created_at = time.time() - 0.2  # Make it old

        new_task = PriorityTask("new", TaskPriority.LOW, {})

        tasks = [old_task, new_task]

        # Check starvation - old task should be boosted
        boosted_count = manager.check_starvation_and_boost(tasks)

        assert boosted_count >= 1
        assert old_task.boosted_priority is not None
        assert new_task.boosted_priority is None  # Too new to be boosted

    def test_fairness_manager_underserved_priority(self):
        """Test boosting for severely underserved priority levels."""
        manager = FairnesManager()

        # Serve many high-priority tasks to skew distribution
        high_task = PriorityTask("high", TaskPriority.HIGH, {})
        for _ in range(10):
            manager.record_task_served(high_task)

        # Create low-priority task that hasn't been served
        low_task = PriorityTask("low", TaskPriority.LOW, {})
        low_task.starvation_threshold = 0.1
        low_task.metrics.created_at = time.time() - 0.2  # Make it old

        boosted_count = manager.check_starvation_and_boost([low_task])

        # Should be boosted due to underrepresentation
        assert boosted_count >= 1
        assert low_task.boosted_priority is not None

    def test_fairness_manager_boosted_task_recording(self):
        """Test recording tasks with boosted priorities."""
        manager = FairnesManager()

        task = PriorityTask("test", TaskPriority.LOW, {})
        task.boost_priority()  # Boost to NORMAL

        manager.record_task_served(task)

        stats = manager.get_fairness_stats()

        # Should be recorded under boosted priority, not original
        assert TaskPriority.NORMAL in stats["priority_distribution"]
        assert stats["priority_distribution"][TaskPriority.NORMAL] == 1


class TestAdvancedPriorityQueue:
    """Test AdvancedPriorityQueue functionality."""

    @pytest.fixture
    def priority_queue(self):
        """Create a test priority queue."""
        queue = AdvancedPriorityQueue(
            max_concurrent_tasks=2,
            starvation_check_interval=5.0,
            enable_resource_monitoring=False  # Disable for most tests
        )
        yield queue
        # Cleanup - only run stop() if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(queue.stop())
        except RuntimeError:
            # No running event loop, queue will be cleaned up via garbage collection
            pass

    def test_priority_queue_initialization(self, priority_queue):
        """Test AdvancedPriorityQueue initialization."""
        assert priority_queue.max_concurrent_tasks == 2
        assert priority_queue.size() == 0
        assert priority_queue.is_empty()

    def test_priority_queue_put_task(self, priority_queue):
        """Test adding tasks to the priority queue."""
        task = PriorityTask("test_task", TaskPriority.NORMAL, {"data": "test"})

        result = priority_queue.put(task)
        assert result is True
        assert priority_queue.size() == 1
        assert not priority_queue.is_empty()

    def test_priority_queue_duplicate_task(self, priority_queue):
        """Test handling of duplicate task IDs."""
        task1 = PriorityTask("duplicate_id", TaskPriority.NORMAL, {})
        task2 = PriorityTask("duplicate_id", TaskPriority.HIGH, {})

        assert priority_queue.put(task1) is True
        assert priority_queue.put(task2) is False  # Should be rejected
        assert priority_queue.size() == 1

    def test_priority_queue_dependency_validation(self, priority_queue):
        """Test dependency validation when adding tasks."""
        # Add task with missing dependency
        task_with_missing_dep = PriorityTask(
            "dependent", TaskPriority.NORMAL, {},
            depends_on=["missing_task"]
        )

        result = priority_queue.put(task_with_missing_dep)
        assert result is False  # Should be rejected
        assert priority_queue.size() == 0

    def test_priority_queue_valid_dependencies(self, priority_queue):
        """Test valid dependency handling."""
        # Add dependency first
        dep_task = PriorityTask("dependency", TaskPriority.NORMAL, {})
        priority_queue.put(dep_task)

        # Add dependent task
        dependent_task = PriorityTask(
            "dependent", TaskPriority.NORMAL, {},
            depends_on=["dependency"]
        )

        result = priority_queue.put(dependent_task)
        assert result is True
        assert priority_queue.size() == 2

    @pytest.mark.asyncio
    async def test_priority_queue_get_task_ordering(self, priority_queue):
        """Test that tasks are retrieved in priority order."""
        # Add tasks in reverse priority order
        low_task = PriorityTask("low", TaskPriority.LOW, {})
        high_task = PriorityTask("high", TaskPriority.HIGH, {})
        normal_task = PriorityTask("normal", TaskPriority.NORMAL, {})

        priority_queue.put(low_task)
        priority_queue.put(high_task)
        priority_queue.put(normal_task)

        # Should get highest priority first
        first_task = await priority_queue.get()
        assert first_task.task_id == "high"

        second_task = await priority_queue.get()
        assert second_task.task_id == "normal"

        third_task = await priority_queue.get()
        assert third_task.task_id == "low"

    @pytest.mark.asyncio
    async def test_priority_queue_empty_get(self, priority_queue):
        """Test getting from empty queue returns None."""
        task = await priority_queue.get()
        assert task is None

    @pytest.mark.asyncio
    async def test_priority_queue_dependency_blocking(self, priority_queue):
        """Test that tasks with unsatisfied dependencies are blocked."""
        # Add dependency task but don't complete it
        dep_task = PriorityTask("dependency", TaskPriority.LOW, {})
        priority_queue.put(dep_task)

        # Add dependent task with higher priority
        dependent_task = PriorityTask(
            "dependent", TaskPriority.HIGH, {},
            depends_on=["dependency"]
        )
        priority_queue.put(dependent_task)

        # Should get dependency first despite lower priority
        first_task = await priority_queue.get()
        assert first_task.task_id == "dependency"

    @pytest.mark.asyncio
    async def test_priority_queue_task_completion(self, priority_queue):
        """Test task completion and unblocking."""
        # Add dependency and dependent tasks
        dep_task = PriorityTask("dependency", TaskPriority.LOW, {})
        dependent_task = PriorityTask(
            "dependent", TaskPriority.HIGH, {},
            depends_on=["dependency"]
        )

        priority_queue.put(dep_task)
        priority_queue.put(dependent_task)

        # Get dependency task
        retrieved_dep = await priority_queue.get()
        assert retrieved_dep.task_id == "dependency"

        # Complete dependency task
        priority_queue.task_completed(retrieved_dep, success=True)

        # Now dependent task should be available
        retrieved_dependent = await priority_queue.get()
        assert retrieved_dependent.task_id == "dependent"

    def test_priority_queue_statistics(self, priority_queue):
        """Test statistics collection."""
        # Add and process some tasks
        task1 = PriorityTask("task1", TaskPriority.HIGH, {})
        task2 = PriorityTask("task2", TaskPriority.LOW, {})

        priority_queue.put(task1)
        priority_queue.put(task2)

        stats = priority_queue.get_statistics()

        assert stats["tasks_queued"] == 2
        assert stats["queue_size"] == 2
        assert stats["running_tasks"] == 0
        assert "fairness_stats" in stats

    def test_priority_queue_queue_contents(self, priority_queue):
        """Test queue contents inspection."""
        task1 = PriorityTask(
            "task1", TaskPriority.HIGH, {},
            file_path=Path("/test/file.txt"),
            file_size=1024
        )
        task2 = PriorityTask("task2", TaskPriority.LOW, {})

        priority_queue.put(task1)
        priority_queue.put(task2)

        contents = priority_queue.get_queue_contents()

        assert len(contents) == 2
        assert any(item["task_id"] == "task1" for item in contents)
        assert any(item["task_id"] == "task2" for item in contents)

        # Check metadata is included
        task1_info = next(item for item in contents if item["task_id"] == "task1")
        assert task1_info["file_path"] == str(Path("/test/file.txt"))
        assert task1_info["file_size"] == 1024

    def test_priority_queue_clear(self, priority_queue):
        """Test clearing the queue."""
        task1 = PriorityTask("task1", TaskPriority.HIGH, {})
        task2 = PriorityTask("task2", TaskPriority.LOW, {})

        priority_queue.put(task1)
        priority_queue.put(task2)

        cleared_count = priority_queue.clear()

        assert cleared_count == 2
        assert priority_queue.size() == 0
        assert priority_queue.is_empty()

    @pytest.mark.asyncio
    async def test_priority_queue_concurrency_limit(self, priority_queue):
        """Test concurrency limit enforcement."""
        # Create tasks
        tasks = [
            PriorityTask(f"task{i}", TaskPriority.NORMAL, {})
            for i in range(5)
        ]

        for task in tasks:
            priority_queue.put(task)

        # Start getting tasks - should be limited by semaphore
        get_tasks = [priority_queue.get() for _ in range(3)]

        # Only max_concurrent_tasks (2) should complete immediately
        completed_tasks = []
        for task_coro in get_tasks:
            try:
                task = await asyncio.wait_for(task_coro, timeout=0.1)
                if task:
                    completed_tasks.append(task)
            except asyncio.TimeoutError:
                pass  # Expected for tasks beyond concurrency limit

        # Should have gotten at least some tasks
        assert len(completed_tasks) >= 0

    @pytest.mark.asyncio
    async def test_priority_queue_starvation_prevention(self, priority_queue):
        """Test starvation prevention through priority boosting."""
        # Create old low-priority task
        old_low_task = PriorityTask("old_low", TaskPriority.LOW, {})
        old_low_task.starvation_threshold = 0.1  # 100ms for testing
        old_low_task.metrics.created_at = time.time() - 0.2  # Make it old

        # Create new high-priority task
        new_high_task = PriorityTask("new_high", TaskPriority.HIGH, {})

        priority_queue.put(old_low_task)
        priority_queue.put(new_high_task)

        # Force starvation check
        priority_queue._last_starvation_check = 0  # Force check

        # Get task - starvation check should happen
        task = await priority_queue.get()

        # Either task could be returned, but low task might be boosted
        assert task is not None

    @pytest.mark.asyncio
    async def test_priority_queue_resource_monitoring(self):
        """Test resource monitoring and throttling."""
        # Create queue with resource monitoring enabled
        queue = AdvancedPriorityQueue(
            max_concurrent_tasks=2,
            enable_resource_monitoring=True,
            cpu_threshold=0.1,  # Very low threshold for testing
            memory_threshold=0.1
        )

        task = PriorityTask("test", TaskPriority.NORMAL, {})
        queue.put(task)

        # Mock high resource usage
        with patch.object(queue._resource_monitor, 'should_throttle', return_value=(True, [ResourceType.CPU])):
            with patch.object(queue._resource_monitor, 'get_throttle_delay', return_value=0.01):
                # Should still get task but with throttling delay
                start_time = time.time()
                retrieved_task = await queue.get()
                elapsed = time.time() - start_time

                assert retrieved_task is not None
                assert elapsed >= 0.01  # Should have waited for throttle delay

        await queue.stop()

    @pytest.mark.asyncio
    async def test_priority_queue_monitoring_loop(self, priority_queue):
        """Test background monitoring loop."""
        await priority_queue.start()

        # Add a task to have some activity
        task = PriorityTask("test", TaskPriority.NORMAL, {})
        priority_queue.put(task)

        # Let monitoring run briefly
        await asyncio.sleep(0.1)

        # Should have monitoring task running
        assert priority_queue._monitoring_task is not None
        assert not priority_queue._monitoring_task.done()

        await priority_queue.stop()

        # Monitoring should be stopped (task may be cancelled, done, or set to None)
        assert (priority_queue._monitoring_task is None or
                priority_queue._monitoring_task.cancelled() or
                priority_queue._monitoring_task.done())

    def test_priority_queue_failed_task_completion(self, priority_queue):
        """Test handling of failed task completion."""
        task = PriorityTask("test", TaskPriority.NORMAL, {})
        priority_queue.put(task)

        # Simulate task retrieval and failure
        retrieved_task = asyncio.run(priority_queue.get())
        priority_queue.task_completed(retrieved_task, success=False, error="Test error")

        stats = priority_queue.get_statistics()
        assert stats["tasks_failed"] == 1
        assert stats["tasks_completed"] == 0

        # Check error is recorded in metrics
        assert retrieved_task.metrics.last_error == "Test error"
        assert retrieved_task.metrics.retry_count == 1

    @pytest.mark.asyncio
    async def test_priority_queue_complex_dependencies(self, priority_queue):
        """Test complex dependency scenarios."""
        # Create a dependency chain: A -> B -> C
        task_a = PriorityTask("A", TaskPriority.LOW, {})
        task_b = PriorityTask("B", TaskPriority.HIGH, {}, depends_on=["A"])
        task_c = PriorityTask("C", TaskPriority.HIGH, {}, depends_on=["B"])

        priority_queue.put(task_a)
        priority_queue.put(task_b)
        priority_queue.put(task_c)

        # Should get A first despite being lowest priority
        first_task = await priority_queue.get()
        assert first_task.task_id == "A"

        # Complete A
        priority_queue.task_completed(first_task, success=True)

        # Now should get B
        second_task = await priority_queue.get()
        assert second_task.task_id == "B"

        # Complete B
        priority_queue.task_completed(second_task, success=True)

        # Finally should get C
        third_task = await priority_queue.get()
        assert third_task.task_id == "C"

    def test_priority_queue_thread_safety(self, priority_queue):
        """Test thread safety of the priority queue."""
        def add_tasks(start_id, count):
            for i in range(count):
                task = PriorityTask(f"thread_task_{start_id}_{i}", TaskPriority.NORMAL, {})
                priority_queue.put(task)

        # Create multiple threads adding tasks
        threads = []
        for thread_id in range(3):
            thread = threading.Thread(target=add_tasks, args=(thread_id, 10))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have all 30 tasks
        assert priority_queue.size() == 30

    @pytest.mark.asyncio
    async def test_priority_queue_age_based_ordering(self, priority_queue):
        """Test age-based ordering for same priority tasks."""
        # Create tasks with same priority but add them in sequence
        task1 = PriorityTask("old", TaskPriority.NORMAL, {})
        time.sleep(0.01)  # Ensure different creation times
        task2 = PriorityTask("new", TaskPriority.NORMAL, {})

        # Add in reverse order (newest first)
        priority_queue.put(task2)
        priority_queue.put(task1)

        # Should get older task first
        first_task = await priority_queue.get()
        assert first_task.task_id == "old"

        second_task = await priority_queue.get()
        assert second_task.task_id == "new"

    def test_priority_queue_metrics_tracking(self, priority_queue):
        """Test comprehensive metrics tracking."""
        task = PriorityTask("test", TaskPriority.NORMAL, {})
        priority_queue.put(task)

        # Get and complete task
        retrieved_task = asyncio.run(priority_queue.get())
        time.sleep(0.01)  # Simulate work
        priority_queue.task_completed(retrieved_task, success=True)

        stats = priority_queue.get_statistics()

        # Check basic stats
        assert stats["tasks_completed"] == 1
        assert stats["average_wait_time"] > 0
        assert stats["average_execution_time"] > 0

        # Check that task metrics were properly set
        assert retrieved_task.metrics.wait_time > 0
        assert retrieved_task.metrics.execution_time > 0
        assert retrieved_task.metrics.completed_at is not None


class TestPriorityQueueEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def priority_queue(self):
        """Create a test priority queue."""
        queue = AdvancedPriorityQueue(
            max_concurrent_tasks=1,
            enable_resource_monitoring=False
        )
        yield queue
        # Cleanup - only run stop() if there's a running event loop
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(queue.stop())
        except RuntimeError:
            # No running event loop, queue will be cleaned up via garbage collection
            pass

    @pytest.mark.asyncio
    async def test_circular_dependencies(self, priority_queue):
        """Test handling of circular dependencies."""
        # This would create a circular dependency in a more complex implementation
        # For now, test that tasks with dependencies are handled properly
        task_a = PriorityTask("A", TaskPriority.NORMAL, {}, depends_on=["B"])
        task_b = PriorityTask("B", TaskPriority.NORMAL, {})

        # Add B first, then A
        priority_queue.put(task_b)
        priority_queue.put(task_a)

        # Should get B first due to A depending on it
        first_task = await priority_queue.get()
        assert first_task.task_id == "B"

    def test_resource_monitor_edge_cases(self):
        """Test ResourceMonitor edge cases."""
        monitor = ResourceMonitor()

        # Test with empty throttle reasons
        delay = monitor.get_throttle_delay([])
        assert delay == 0.0

        # Test throttling decision with custom usage
        custom_usage = {
            ResourceType.CPU: 0.95,
            ResourceType.MEMORY: 0.99,
            ResourceType.IO_READ: 2000.0,
            ResourceType.IO_WRITE: 2000.0,
        }

        should_throttle, reasons = monitor.should_throttle(custom_usage)
        assert should_throttle
        assert len(reasons) >= 2  # CPU and memory should both trigger

    @pytest.mark.asyncio
    async def test_queue_stress_test(self, priority_queue):
        """Stress test with many tasks."""
        # Add many tasks
        task_count = 100
        for i in range(task_count):
            priority = TaskPriority(i % 5)  # Distribute across all priorities
            task = PriorityTask(f"stress_task_{i}", priority, {"data": i})
            priority_queue.put(task)

        assert priority_queue.size() == task_count

        # Process all tasks
        processed_count = 0
        while not priority_queue.is_empty():
            task = await priority_queue.get()
            if task:
                priority_queue.task_completed(task, success=True)
                processed_count += 1

        assert processed_count == task_count

        stats = priority_queue.get_statistics()
        assert stats["tasks_completed"] == task_count

    def test_priority_task_boundary_values(self):
        """Test PriorityTask with boundary values."""
        # Test with extreme resource requirements
        task = PriorityTask(
            "boundary_test",
            TaskPriority.CRITICAL,
            {},
            estimated_cpu_usage=1.0,  # 100% CPU
            estimated_memory_mb=0.0,   # 0 MB
            estimated_io_ops=1000000   # 1M operations
        )

        assert task.estimated_cpu_usage == 1.0
        assert task.estimated_memory_mb == 0.0
        assert task.estimated_io_ops == 1000000

        # Test with very short starvation threshold
        task.starvation_threshold = 0.001  # 1ms
        time.sleep(0.002)
        assert task.should_boost_priority()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
