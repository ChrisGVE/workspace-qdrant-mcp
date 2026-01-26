"""
Advanced Priority Queue System for File Watching with Fairness and Resource Management.

This module provides a sophisticated priority queue implementation that prevents starvation,
manages resource allocation, and provides dynamic priority adjustment based on file
characteristics and system load.
"""

import asyncio
import heapq
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

import psutil
from loguru import logger


class TaskPriority(IntEnum):
    """Task priority levels with numerical values for comparison."""
    CRITICAL = 0      # System critical files, config changes
    HIGH = 1          # User-modified files, important documents
    NORMAL = 2        # Regular file updates, new files
    LOW = 3           # Background tasks, bulk operations
    BACKGROUND = 4    # Cleanup, maintenance tasks


class ResourceType(Enum):
    """Types of system resources to monitor and manage."""
    CPU = "cpu"
    MEMORY = "memory"
    IO_READ = "io_read"
    IO_WRITE = "io_write"
    NETWORK = "network"


@dataclass
class TaskMetrics:
    """Metrics for tracking task performance and resource usage."""
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    wait_time: float = 0.0
    execution_time: float = 0.0
    retry_count: int = 0
    last_error: str | None = None
    resource_usage: dict[ResourceType, float] = field(default_factory=dict)

    def mark_started(self) -> None:
        """Mark task as started and calculate wait time."""
        self.started_at = time.time()
        self.wait_time = self.started_at - self.created_at

    def mark_completed(self) -> None:
        """Mark task as completed and calculate execution time."""
        self.completed_at = time.time()
        if self.started_at:
            self.execution_time = self.completed_at - self.started_at

    def age_seconds(self) -> float:
        """Get task age in seconds."""
        return time.time() - self.created_at


@dataclass
class PriorityTask:
    """Task wrapper with priority, metadata, and fairness tracking."""

    # Core task data
    task_id: str
    priority: TaskPriority
    payload: Any
    callback: Callable | None = None

    # Priority and fairness management
    original_priority: TaskPriority = field(init=False)
    boosted_priority: TaskPriority | None = None
    starvation_threshold: float = 300.0  # 5 minutes default

    # Resource requirements and constraints
    estimated_cpu_usage: float = 0.1  # 0.0 to 1.0
    estimated_memory_mb: float = 10.0
    estimated_io_ops: int = 100
    requires_exclusive_access: bool = False

    # Metadata and tracking
    file_path: Path | None = None
    file_size: int | None = None
    file_type: str | None = None
    depends_on: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)

    # Metrics and monitoring
    metrics: TaskMetrics = field(default_factory=TaskMetrics)

    # Internal priority queue fields
    _queue_index: int = field(default=0, init=False)

    def __post_init__(self):
        """Initialize derived fields."""
        self.original_priority = self.priority

    def __lt__(self, other: 'PriorityTask') -> bool:
        """Compare tasks for priority queue ordering."""
        # Use boosted priority if available, otherwise original
        self_priority = self.boosted_priority if self.boosted_priority is not None else self.priority
        other_priority = other.boosted_priority if other.boosted_priority is not None else other.priority

        # Primary comparison: priority level
        if self_priority != other_priority:
            return self_priority < other_priority

        # Secondary comparison: age (older tasks get priority within same level)
        return self.metrics.created_at < other.metrics.created_at

    def should_boost_priority(self) -> bool:
        """Check if task should have its priority boosted due to age."""
        age = self.metrics.age_seconds()
        return age >= self.starvation_threshold and self.boosted_priority is None

    def boost_priority(self) -> None:
        """Boost task priority to prevent starvation."""
        if self.priority > TaskPriority.CRITICAL:
            # Boost by one level, but never exceed CRITICAL
            new_priority_value = max(TaskPriority.CRITICAL, self.priority - 1)
            self.boosted_priority = TaskPriority(new_priority_value)
            logger.info(f"Boosted task {self.task_id} priority from {self.original_priority} to {self.boosted_priority}")

    def reset_priority_boost(self) -> None:
        """Reset priority boost (used after task completion)."""
        self.boosted_priority = None


class ResourceMonitor:
    """Monitors system resource usage and provides throttling recommendations."""

    def __init__(self,
                 cpu_threshold: float = 0.8,
                 memory_threshold: float = 0.9,
                 io_threshold: float = 1000.0):
        """Initialize resource monitor with thresholds."""
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.io_threshold = io_threshold  # MB/s

        self._last_io_counters = psutil.disk_io_counters()
        self._last_io_time = time.time()

    def get_current_usage(self) -> dict[ResourceType, float]:
        """Get current system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1) / 100.0
        memory = psutil.virtual_memory()
        memory_percent = memory.percent / 100.0

        # Calculate IO rate
        current_io = psutil.disk_io_counters()
        current_time = time.time()
        time_delta = current_time - self._last_io_time

        if time_delta > 0.5 and self._last_io_counters:
            read_bytes_per_sec = (current_io.read_bytes - self._last_io_counters.read_bytes) / time_delta
            write_bytes_per_sec = (current_io.write_bytes - self._last_io_counters.write_bytes) / time_delta
            io_read_mb_per_sec = read_bytes_per_sec / (1024 * 1024)
            io_write_mb_per_sec = write_bytes_per_sec / (1024 * 1024)
        else:
            io_read_mb_per_sec = 0.0
            io_write_mb_per_sec = 0.0

        self._last_io_counters = current_io
        self._last_io_time = current_time

        return {
            ResourceType.CPU: cpu_percent,
            ResourceType.MEMORY: memory_percent,
            ResourceType.IO_READ: io_read_mb_per_sec,
            ResourceType.IO_WRITE: io_write_mb_per_sec,
        }

    def should_throttle(self, usage: dict[ResourceType, float] | None = None) -> tuple[bool, list[ResourceType]]:
        """Check if system should throttle due to resource pressure."""
        if usage is None:
            usage = self.get_current_usage()

        throttle_reasons = []

        if usage[ResourceType.CPU] > self.cpu_threshold:
            throttle_reasons.append(ResourceType.CPU)

        if usage[ResourceType.MEMORY] > self.memory_threshold:
            throttle_reasons.append(ResourceType.MEMORY)

        if usage[ResourceType.IO_READ] > self.io_threshold or usage[ResourceType.IO_WRITE] > self.io_threshold:
            throttle_reasons.append(ResourceType.IO_READ)

        return len(throttle_reasons) > 0, throttle_reasons

    def get_throttle_delay(self, throttle_reasons: list[ResourceType]) -> float:
        """Calculate throttle delay based on resource pressure."""
        if not throttle_reasons:
            return 0.0

        # Base delay of 1 second, increased for each resource under pressure
        base_delay = 1.0
        multiplier = 1.0 + (len(throttle_reasons) * 0.5)

        # Additional delay for severe resource pressure
        usage = self.get_current_usage()
        severe_multiplier = 1.0

        for resource in throttle_reasons:
            if resource == ResourceType.CPU and usage[ResourceType.CPU] > 0.95:
                severe_multiplier += 1.0
            elif resource == ResourceType.MEMORY and usage[ResourceType.MEMORY] > 0.98:
                severe_multiplier += 2.0  # Memory pressure is more critical

        return base_delay * multiplier * severe_multiplier


class FairnesManager:
    """Manages task fairness and prevents starvation."""

    def __init__(self, starvation_check_interval: float = 30.0):
        """Initialize fairness manager."""
        self.starvation_check_interval = starvation_check_interval
        self._priority_counts = defaultdict(int)
        self._priority_last_served = defaultdict(float)
        self._total_tasks_served = 0

    def record_task_served(self, task: PriorityTask) -> None:
        """Record that a task of given priority was served."""
        effective_priority = task.boosted_priority if task.boosted_priority else task.priority
        self._priority_counts[effective_priority] += 1
        self._priority_last_served[effective_priority] = time.time()
        self._total_tasks_served += 1

    def check_starvation_and_boost(self, tasks: list[PriorityTask]) -> int:
        """Check for task starvation and boost priorities as needed."""
        current_time = time.time()
        boosted_count = 0

        for task in tasks:
            if task.should_boost_priority():
                # Additional fairness check: has this priority level been severely underserved?
                priority_ratio = self._priority_counts[task.priority] / max(1, self._total_tasks_served)
                time_since_served = current_time - self._priority_last_served.get(task.priority, 0)

                # Boost if severely underserved or hasn't been served recently
                if priority_ratio < 0.1 or time_since_served > task.starvation_threshold:
                    task.boost_priority()
                    boosted_count += 1

        return boosted_count

    def get_fairness_stats(self) -> dict[str, Any]:
        """Get current fairness statistics."""
        stats = {
            "total_tasks_served": self._total_tasks_served,
            "priority_distribution": dict(self._priority_counts),
            "priority_last_served": {str(k): v for k, v in self._priority_last_served.items()}
        }

        # Calculate fairness ratios
        if self._total_tasks_served > 0:
            stats["fairness_ratios"] = {
                str(priority): count / self._total_tasks_served
                for priority, count in self._priority_counts.items()
            }
        else:
            stats["fairness_ratios"] = {}

        return stats


class AdvancedPriorityQueue:
    """
    Advanced priority queue with fairness, resource management, and starvation prevention.

    Features:
    - Priority-based task ordering with fairness algorithms
    - Resource usage monitoring and throttling
    - Starvation prevention through priority boosting
    - Dependency management and conflict resolution
    - Comprehensive metrics and monitoring
    """

    def __init__(self,
                 max_concurrent_tasks: int = 4,
                 starvation_check_interval: float = 30.0,
                 enable_resource_monitoring: bool = True,
                 cpu_threshold: float = 0.8,
                 memory_threshold: float = 0.9):
        """Initialize advanced priority queue."""

        # Core queue and task management
        self._heap: list[PriorityTask] = []
        self._task_dict: dict[str, PriorityTask] = {}
        self._running_tasks: dict[str, PriorityTask] = {}
        self._completed_tasks: deque = deque(maxlen=1000)  # Keep last 1000 for stats

        # Concurrency and resource limits
        self.max_concurrent_tasks = max_concurrent_tasks
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._lock = threading.RLock()

        # Fairness and starvation prevention
        self._fairness_manager = FairnesManager(starvation_check_interval)
        self._last_starvation_check = time.time()

        # Resource monitoring and throttling
        self._resource_monitor = ResourceMonitor(
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold
        ) if enable_resource_monitoring else None

        # Dependency tracking
        self._dependency_graph: dict[str, set[str]] = defaultdict(set)  # task_id -> dependencies
        self._blocking_graph: dict[str, set[str]] = defaultdict(set)    # task_id -> blocked tasks

        # Background monitoring task
        self._monitoring_task: asyncio.Task | None = None
        self._shutdown = False

        # Metrics and statistics
        self.stats = {
            "tasks_queued": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_wait_time": 0.0,
            "average_execution_time": 0.0,
            "priority_boosts": 0,
            "throttle_events": 0,
            "resource_usage": {},
        }

    async def start(self) -> None:
        """Start the priority queue's background monitoring."""
        if self._monitoring_task is None:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Advanced priority queue monitoring started")

    async def stop(self) -> None:
        """Stop the priority queue and clean up resources."""
        self._shutdown = True
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("Advanced priority queue stopped")

    def put(self, task: PriorityTask) -> bool:
        """
        Add a task to the priority queue.

        Returns:
            bool: True if task was added, False if rejected (e.g., due to dependencies)
        """
        with self._lock:
            # Check if task already exists
            if task.task_id in self._task_dict:
                logger.warning(f"Task {task.task_id} already in queue, ignoring duplicate")
                return False

            # Validate dependencies
            missing_deps = []
            for dep_id in task.depends_on:
                if dep_id not in self._task_dict and dep_id not in self._completed_tasks:
                    missing_deps.append(dep_id)

            if missing_deps:
                logger.error(f"Task {task.task_id} has missing dependencies: {missing_deps}")
                return False

            # Update dependency graphs
            for dep_id in task.depends_on:
                self._dependency_graph[task.task_id].add(dep_id)
                self._blocking_graph[dep_id].add(task.task_id)

            # Add to queue
            heapq.heappush(self._heap, task)
            self._task_dict[task.task_id] = task
            self.stats["tasks_queued"] += 1

            logger.debug(f"Added task {task.task_id} with priority {task.priority} to queue")
            return True

    async def get(self) -> PriorityTask | None:
        """
        Get the next task from the priority queue.

        Returns:
            Optional[PriorityTask]: Next task to execute, or None if queue is empty
        """
        async with self._semaphore:  # Enforce concurrency limit
            with self._lock:
                # Check for starvation and boost priorities if needed
                current_time = time.time()
                if current_time - self._last_starvation_check >= self._fairness_manager.starvation_check_interval:
                    boosted = self._fairness_manager.check_starvation_and_boost(list(self._task_dict.values()))
                    if boosted > 0:
                        self.stats["priority_boosts"] += boosted
                        # Re-heapify after priority changes
                        heapq.heapify(self._heap)
                    self._last_starvation_check = current_time

                # Get next available task (considering dependencies)
                task = self._get_next_ready_task()
                if not task:
                    return None

                # Check resource usage and throttle if necessary
                if self._resource_monitor:
                    should_throttle, throttle_reasons = self._resource_monitor.should_throttle()
                    if should_throttle:
                        delay = self._resource_monitor.get_throttle_delay(throttle_reasons)
                        logger.info(f"Throttling due to resource pressure ({throttle_reasons}), waiting {delay:.1f}s")
                        self.stats["throttle_events"] += 1
                        await asyncio.sleep(delay)

                # Mark task as started
                task.metrics.mark_started()
                self._running_tasks[task.task_id] = task
                del self._task_dict[task.task_id]

                logger.debug(f"Retrieved task {task.task_id} from queue (waited {task.metrics.wait_time:.1f}s)")
                return task

    def _get_next_ready_task(self) -> PriorityTask | None:
        """Get the next task that has all dependencies satisfied."""
        ready_tasks = []

        # Find all tasks with satisfied dependencies
        for task in self._heap:
            if self._are_dependencies_satisfied(task):
                ready_tasks.append(task)

        if not ready_tasks:
            return None

        # Get highest priority task
        ready_tasks.sort()  # This will use the __lt__ method
        next_task = ready_tasks[0]

        # Remove from heap
        self._heap.remove(next_task)
        heapq.heapify(self._heap)

        return next_task

    def _are_dependencies_satisfied(self, task: PriorityTask) -> bool:
        """Check if all task dependencies are satisfied."""
        for dep_id in task.depends_on:
            # Dependency is satisfied if it's completed or currently running
            if dep_id not in self._running_tasks and not any(
                completed_task.task_id == dep_id for completed_task in self._completed_tasks
            ):
                return False
        return True

    def task_completed(self, task: PriorityTask, success: bool = True, error: str | None = None) -> None:
        """Mark a task as completed and update statistics."""
        with self._lock:
            # Remove from running tasks
            if task.task_id in self._running_tasks:
                del self._running_tasks[task.task_id]

            # Update task metrics
            task.metrics.mark_completed()
            if error:
                task.metrics.last_error = error
                task.metrics.retry_count += 1

            # Update statistics
            if success:
                self.stats["tasks_completed"] += 1
                self._completed_tasks.append(task)

                # Record for fairness tracking
                self._fairness_manager.record_task_served(task)
            else:
                self.stats["tasks_failed"] += 1

            # Update average times
            self._update_average_stats()

            # Reset priority boost
            task.reset_priority_boost()

            # Unblock dependent tasks
            self._unblock_dependent_tasks(task.task_id)

            logger.debug(f"Task {task.task_id} completed (success={success}, execution_time={task.metrics.execution_time:.1f}s)")

    def _update_average_stats(self) -> None:
        """Update average wait and execution time statistics."""
        if not self._completed_tasks:
            return

        total_wait_time = sum(task.metrics.wait_time for task in self._completed_tasks)
        total_execution_time = sum(task.metrics.execution_time for task in self._completed_tasks)
        count = len(self._completed_tasks)

        self.stats["average_wait_time"] = total_wait_time / count
        self.stats["average_execution_time"] = total_execution_time / count

    def _unblock_dependent_tasks(self, completed_task_id: str) -> None:
        """Unblock tasks that were waiting for the completed task."""
        blocked_tasks = self._blocking_graph.get(completed_task_id, set())
        for blocked_id in blocked_tasks:
            if blocked_id in self._dependency_graph:
                self._dependency_graph[blocked_id].discard(completed_task_id)

        # Clean up blocking graph
        if completed_task_id in self._blocking_graph:
            del self._blocking_graph[completed_task_id]

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive queue statistics."""
        with self._lock:
            current_stats = self.stats.copy()
            current_stats.update({
                "queue_size": len(self._heap),
                "running_tasks": len(self._running_tasks),
                "completed_tasks_history": len(self._completed_tasks),
                "fairness_stats": self._fairness_manager.get_fairness_stats(),
                "dependency_graph_size": len(self._dependency_graph),
                "blocking_graph_size": len(self._blocking_graph),
            })

            # Add resource usage if monitoring is enabled
            if self._resource_monitor:
                current_stats["resource_usage"] = self._resource_monitor.get_current_usage()

            return current_stats

    def get_queue_contents(self) -> list[dict[str, Any]]:
        """Get information about all tasks currently in the queue."""
        with self._lock:
            return [
                {
                    "task_id": task.task_id,
                    "priority": str(task.priority),
                    "boosted_priority": str(task.boosted_priority) if task.boosted_priority else None,
                    "age_seconds": task.metrics.age_seconds(),
                    "file_path": str(task.file_path) if task.file_path else None,
                    "file_size": task.file_size,
                    "depends_on": task.depends_on,
                    "blocks": task.blocks,
                    "estimated_resources": {
                        "cpu": task.estimated_cpu_usage,
                        "memory_mb": task.estimated_memory_mb,
                        "io_ops": task.estimated_io_ops,
                    }
                }
                for task in self._heap
            ]

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop for resource usage and queue health."""
        while not self._shutdown:
            try:
                # Update resource usage stats
                if self._resource_monitor:
                    self.stats["resource_usage"] = self._resource_monitor.get_current_usage()

                # Log queue health periodically
                stats = self.get_statistics()
                logger.debug(f"Queue health: {stats['queue_size']} queued, {stats['running_tasks']} running, "
                           f"avg_wait={stats['average_wait_time']:.1f}s")

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in priority queue monitoring: {e}")
                await asyncio.sleep(5)  # Brief pause before retrying

    def clear(self) -> int:
        """Clear all tasks from the queue and return the number of tasks cleared."""
        with self._lock:
            cleared_count = len(self._heap) + len(self._running_tasks)
            self._heap.clear()
            self._task_dict.clear()
            self._running_tasks.clear()
            self._dependency_graph.clear()
            self._blocking_graph.clear()
            return cleared_count

    def size(self) -> int:
        """Get the total number of tasks in the queue."""
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._heap) == 0
