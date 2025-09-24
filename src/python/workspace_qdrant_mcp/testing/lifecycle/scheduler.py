"""
Maintenance Scheduler

Schedules and manages test maintenance tasks including refactoring alerts,
cleanup recommendations, and test health monitoring with conflict resolution
and resource constraint handling.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Callable
import sqlite3
import json
from threading import Lock
import heapq
from uuid import uuid4

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for maintenance tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class TaskStatus(Enum):
    """Status of maintenance tasks."""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"


class TaskType(Enum):
    """Types of maintenance tasks."""
    TEST_REFACTOR = "test_refactor"
    TEST_CLEANUP = "test_cleanup"
    COVERAGE_IMPROVEMENT = "coverage_improvement"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DEPENDENCY_UPDATE = "dependency_update"
    DOCUMENTATION_UPDATE = "documentation_update"
    OBSOLETE_TEST_REMOVAL = "obsolete_test_removal"
    FLAKY_TEST_INVESTIGATION = "flaky_test_investigation"
    SECURITY_UPDATE = "security_update"
    HEALTH_CHECK = "health_check"


@dataclass
class ResourceConstraint:
    """Resource constraints for task execution."""
    max_concurrent_tasks: int = 3
    max_memory_mb: int = 1024
    max_cpu_percent: int = 80
    excluded_time_ranges: List[tuple] = field(default_factory=list)  # (start_hour, end_hour)
    required_tools: Set[str] = field(default_factory=set)


@dataclass
class MaintenanceTask:
    """Represents a test maintenance task."""
    task_id: str
    task_type: TaskType
    priority: TaskPriority
    title: str
    description: str
    estimated_duration: timedelta
    target_files: List[Path] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)  # Task IDs this depends on
    prerequisites: List[str] = field(default_factory=list)  # Requirements to run
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_for: Optional[datetime] = None
    deadline: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_to: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None

    def __lt__(self, other):
        """Enable priority queue ordering."""
        return (self.priority.value, self.scheduled_for or datetime.max) < \
               (other.priority.value, other.scheduled_for or datetime.max)


@dataclass
class SchedulingConflict:
    """Represents a scheduling conflict."""
    conflict_id: str
    conflicting_tasks: List[str]  # Task IDs
    conflict_type: str  # resource, dependency, time
    description: str
    suggested_resolution: str
    severity: str = "medium"  # low, medium, high


class MaintenanceScheduler:
    """
    Schedules and manages test maintenance tasks.

    Handles task prioritization, resource constraints, dependency resolution,
    and conflict detection with comprehensive error handling.
    """

    def __init__(self,
                 db_path: Path,
                 resource_constraints: Optional[ResourceConstraint] = None):
        """
        Initialize maintenance scheduler.

        Args:
            db_path: Path to SQLite database for persistence
            resource_constraints: Resource limits for task execution
        """
        self.db_path = db_path
        self.resource_constraints = resource_constraints or ResourceConstraint()
        self.lock = Lock()
        self.task_queue = []  # Priority queue
        self.running_tasks = {}  # task_id -> task
        self.task_handlers = {}  # task_type -> handler function
        self.observers = []  # Task completion observers

        self._initialize_db()
        self._load_tasks()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS maintenance_tasks (
                        task_id TEXT PRIMARY KEY,
                        task_type TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        estimated_duration_seconds INTEGER,
                        target_files TEXT,  -- JSON array
                        dependencies TEXT,  -- JSON array
                        prerequisites TEXT,  -- JSON array
                        created_at REAL NOT NULL,
                        scheduled_for REAL,
                        deadline REAL,
                        status TEXT NOT NULL,
                        assigned_to TEXT,
                        progress REAL DEFAULT 0.0,
                        metadata TEXT,  -- JSON object
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        error_message TEXT,
                        completed_at REAL
                    )
                ''')

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS scheduling_conflicts (
                        conflict_id TEXT PRIMARY KEY,
                        conflicting_tasks TEXT NOT NULL,  -- JSON array
                        conflict_type TEXT NOT NULL,
                        description TEXT NOT NULL,
                        suggested_resolution TEXT NOT NULL,
                        severity TEXT DEFAULT 'medium',
                        created_at REAL DEFAULT (julianday('now')),
                        resolved_at REAL
                    )
                ''')

                conn.execute('''
                    CREATE TABLE IF NOT EXISTS task_history (
                        history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        details TEXT,
                        timestamp REAL DEFAULT (julianday('now'))
                    )
                ''')

                # Create indexes for performance
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_tasks_status
                    ON maintenance_tasks(status)
                ''')

                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_tasks_scheduled
                    ON maintenance_tasks(scheduled_for)
                ''')

                conn.commit()
            finally:
                conn.close()

    def schedule_task(self, task: MaintenanceTask) -> bool:
        """
        Schedule a maintenance task.

        Args:
            task: Task to schedule

        Returns:
            True if successfully scheduled, False otherwise

        Raises:
            ValueError: If task validation fails
        """
        try:
            # Validate task
            self._validate_task(task)

            # Check for conflicts
            conflicts = self._detect_conflicts(task)
            if conflicts:
                # Try to resolve conflicts automatically
                resolved = self._attempt_conflict_resolution(task, conflicts)
                if not resolved:
                    logger.warning(f"Cannot schedule task {task.task_id} due to conflicts")
                    for conflict in conflicts:
                        self._save_conflict(conflict)
                    return False

            # Determine optimal scheduling time
            optimal_time = self._calculate_optimal_schedule_time(task)
            task.scheduled_for = optimal_time
            task.status = TaskStatus.SCHEDULED

            # Add to queue and database
            with self.lock:
                heapq.heappush(self.task_queue, task)
                self._save_task(task)
                self._log_task_action(task.task_id, "scheduled",
                                    f"Scheduled for {optimal_time}")

            logger.info(f"Task {task.task_id} scheduled for {optimal_time}")
            return True

        except Exception as e:
            logger.error(f"Failed to schedule task {task.task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            self._save_task(task)
            return False

    def get_next_task(self) -> Optional[MaintenanceTask]:
        """
        Get the next task ready for execution.

        Returns:
            Next task to execute or None if none available
        """
        with self.lock:
            now = datetime.now()

            while self.task_queue:
                # Check if highest priority task is ready
                next_task = self.task_queue[0]

                if next_task.status != TaskStatus.SCHEDULED:
                    heapq.heappop(self.task_queue)
                    continue

                if next_task.scheduled_for and next_task.scheduled_for > now:
                    # Not time yet
                    return None

                # Check dependencies
                if not self._are_dependencies_satisfied(next_task):
                    # Try to find another task
                    self._requeue_task(heapq.heappop(self.task_queue))
                    continue

                # Check resource constraints
                if not self._check_resource_availability(next_task):
                    return None

                # Task is ready
                task = heapq.heappop(self.task_queue)
                task.status = TaskStatus.IN_PROGRESS
                self.running_tasks[task.task_id] = task
                self._save_task(task)
                self._log_task_action(task.task_id, "started", "Task execution started")

                return task

            return None

    def complete_task(self, task_id: str, success: bool = True,
                     result: Optional[str] = None) -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: ID of completed task
            success: Whether task completed successfully
            result: Optional result description

        Returns:
            True if task was marked completed, False otherwise
        """
        with self.lock:
            if task_id not in self.running_tasks:
                logger.warning(f"Task {task_id} not found in running tasks")
                return False

            task = self.running_tasks.pop(task_id)
            task.completed_at = datetime.now()
            task.progress = 1.0

            if success:
                task.status = TaskStatus.COMPLETED
                self._log_task_action(task_id, "completed", result or "Task completed successfully")

                # Notify observers
                for observer in self.observers:
                    try:
                        observer(task, success, result)
                    except Exception as e:
                        logger.error(f"Observer error for task {task_id}: {e}")

            else:
                task.retry_count += 1
                if task.retry_count < task.max_retries:
                    # Retry task
                    task.status = TaskStatus.PENDING
                    task.error_message = result
                    delay = timedelta(minutes=5 * (2 ** task.retry_count))  # Exponential backoff
                    task.scheduled_for = datetime.now() + delay
                    heapq.heappush(self.task_queue, task)
                    self._log_task_action(task_id, "retrying", f"Retry {task.retry_count}/{task.max_retries}")
                else:
                    # Max retries exceeded
                    task.status = TaskStatus.FAILED
                    task.error_message = result or "Task failed after max retries"
                    self._log_task_action(task_id, "failed", task.error_message)

            self._save_task(task)
            return True

    def cancel_task(self, task_id: str, reason: str = "") -> bool:
        """
        Cancel a scheduled or running task.

        Args:
            task_id: ID of task to cancel
            reason: Reason for cancellation

        Returns:
            True if task was cancelled, False otherwise
        """
        with self.lock:
            # Check running tasks first
            if task_id in self.running_tasks:
                task = self.running_tasks.pop(task_id)
                task.status = TaskStatus.CANCELLED
                task.error_message = reason
                self._save_task(task)
                self._log_task_action(task_id, "cancelled", reason)
                return True

            # Check queued tasks
            for i, task in enumerate(self.task_queue):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    task.error_message = reason
                    # Remove from queue (heap property will be maintained)
                    self.task_queue[i] = self.task_queue[-1]
                    self.task_queue.pop()
                    if self.task_queue:
                        heapq.heapify(self.task_queue)
                    self._save_task(task)
                    self._log_task_action(task_id, "cancelled", reason)
                    return True

            return False

    def get_task_status(self, task_id: str) -> Optional[MaintenanceTask]:
        """Get current status of a task."""
        # Check running tasks first
        if task_id in self.running_tasks:
            return self.running_tasks[task_id]

        # Check database
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                'SELECT * FROM maintenance_tasks WHERE task_id = ?',
                (task_id,)
            )
            result = cursor.fetchone()
            if result:
                return self._row_to_task(result)
        finally:
            conn.close()

        return None

    def get_pending_tasks(self, task_type: Optional[TaskType] = None) -> List[MaintenanceTask]:
        """Get list of pending tasks."""
        conn = sqlite3.connect(self.db_path)
        try:
            if task_type:
                cursor = conn.execute('''
                    SELECT * FROM maintenance_tasks
                    WHERE status IN ('pending', 'scheduled') AND task_type = ?
                    ORDER BY priority DESC, created_at ASC
                ''', (task_type.value,))
            else:
                cursor = conn.execute('''
                    SELECT * FROM maintenance_tasks
                    WHERE status IN ('pending', 'scheduled')
                    ORDER BY priority DESC, created_at ASC
                ''')

            return [self._row_to_task(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_conflicts(self, resolved: bool = False) -> List[SchedulingConflict]:
        """Get scheduling conflicts."""
        conn = sqlite3.connect(self.db_path)
        try:
            if resolved:
                cursor = conn.execute('''
                    SELECT * FROM scheduling_conflicts
                    WHERE resolved_at IS NOT NULL
                    ORDER BY created_at DESC
                ''')
            else:
                cursor = conn.execute('''
                    SELECT * FROM scheduling_conflicts
                    WHERE resolved_at IS NULL
                    ORDER BY severity DESC, created_at DESC
                ''')

            conflicts = []
            for row in cursor.fetchall():
                conflict = SchedulingConflict(
                    conflict_id=row[0],
                    conflicting_tasks=json.loads(row[1]),
                    conflict_type=row[2],
                    description=row[3],
                    suggested_resolution=row[4],
                    severity=row[5]
                )
                conflicts.append(conflict)

            return conflicts
        finally:
            conn.close()

    def register_task_handler(self, task_type: TaskType, handler: Callable) -> None:
        """Register a handler for a task type."""
        self.task_handlers[task_type] = handler

    def add_observer(self, observer: Callable) -> None:
        """Add a task completion observer."""
        self.observers.append(observer)

    def _validate_task(self, task: MaintenanceTask) -> None:
        """Validate task before scheduling."""
        if not task.task_id:
            raise ValueError("Task ID is required")

        if not task.title:
            raise ValueError("Task title is required")

        if task.estimated_duration.total_seconds() <= 0:
            raise ValueError("Task duration must be positive")

        if task.deadline and task.deadline <= datetime.now():
            raise ValueError("Task deadline cannot be in the past")

    def _detect_conflicts(self, task: MaintenanceTask) -> List[SchedulingConflict]:
        """Detect scheduling conflicts for a task."""
        conflicts = []

        # Check resource conflicts
        if self._would_exceed_resource_limits(task):
            conflicts.append(SchedulingConflict(
                conflict_id=str(uuid4()),
                conflicting_tasks=[task.task_id],
                conflict_type="resource",
                description="Task would exceed available resource constraints",
                suggested_resolution="Schedule during off-peak hours or reduce resource requirements"
            ))

        # Check dependency conflicts
        circular_deps = self._detect_circular_dependencies(task)
        if circular_deps:
            conflicts.append(SchedulingConflict(
                conflict_id=str(uuid4()),
                conflicting_tasks=circular_deps,
                conflict_type="dependency",
                description="Circular dependency detected",
                suggested_resolution="Remove circular dependency or reschedule dependent tasks"
            ))

        # Check time conflicts (same files being modified)
        file_conflicts = self._detect_file_conflicts(task)
        if file_conflicts:
            conflicts.append(SchedulingConflict(
                conflict_id=str(uuid4()),
                conflicting_tasks=file_conflicts,
                conflict_type="time",
                description="Multiple tasks targeting same files",
                suggested_resolution="Stagger execution times or merge related tasks"
            ))

        return conflicts

    def _attempt_conflict_resolution(self, task: MaintenanceTask,
                                   conflicts: List[SchedulingConflict]) -> bool:
        """Attempt to automatically resolve conflicts."""
        for conflict in conflicts:
            if conflict.conflict_type == "resource":
                # Try scheduling during off-peak hours
                off_peak_time = self._find_off_peak_time()
                if off_peak_time:
                    task.scheduled_for = off_peak_time
                    logger.info(f"Resolved resource conflict for {task.task_id} by rescheduling")
                    return True

            elif conflict.conflict_type == "time":
                # Try to delay task
                delay = timedelta(hours=1)
                optimal_time = self._calculate_optimal_schedule_time(task, min_delay=delay)
                if optimal_time:
                    task.scheduled_for = optimal_time
                    logger.info(f"Resolved time conflict for {task.task_id} by delaying")
                    return True

        return False

    def _calculate_optimal_schedule_time(self, task: MaintenanceTask,
                                       min_delay: timedelta = None) -> datetime:
        """Calculate optimal scheduling time for a task."""
        now = datetime.now()
        earliest_time = now + (min_delay or timedelta(minutes=5))

        # Consider dependencies
        dep_completion_time = self._get_dependency_completion_time(task)
        if dep_completion_time:
            earliest_time = max(earliest_time, dep_completion_time)

        # Consider resource constraints and time restrictions
        optimal_time = earliest_time
        for excluded_range in self.resource_constraints.excluded_time_ranges:
            start_hour, end_hour = excluded_range
            if start_hour <= optimal_time.hour < end_hour:
                # Move to after excluded period
                optimal_time = optimal_time.replace(
                    hour=end_hour, minute=0, second=0, microsecond=0
                )

        # Consider deadline
        if task.deadline:
            if optimal_time + task.estimated_duration > task.deadline:
                # Try to fit before deadline
                optimal_time = task.deadline - task.estimated_duration
                if optimal_time < now:
                    logger.warning(f"Task {task.task_id} cannot meet deadline")

        return optimal_time

    def _are_dependencies_satisfied(self, task: MaintenanceTask) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True

        conn = sqlite3.connect(self.db_path)
        try:
            for dep_id in task.dependencies:
                cursor = conn.execute(
                    'SELECT status FROM maintenance_tasks WHERE task_id = ?',
                    (dep_id,)
                )
                result = cursor.fetchone()
                if not result or result[0] != TaskStatus.COMPLETED.value:
                    return False
            return True
        finally:
            conn.close()

    def _check_resource_availability(self, task: MaintenanceTask) -> bool:
        """Check if resources are available for task execution."""
        # Check concurrent task limit
        if len(self.running_tasks) >= self.resource_constraints.max_concurrent_tasks:
            return False

        # Check time restrictions
        now = datetime.now()
        for excluded_range in self.resource_constraints.excluded_time_ranges:
            start_hour, end_hour = excluded_range
            if start_hour <= now.hour < end_hour:
                return False

        return True

    def _would_exceed_resource_limits(self, task: MaintenanceTask) -> bool:
        """Check if task would exceed resource limits."""
        # Simplified check - in practice would check actual resource usage
        return len(self.running_tasks) >= self.resource_constraints.max_concurrent_tasks

    def _detect_circular_dependencies(self, task: MaintenanceTask) -> List[str]:
        """Detect circular dependencies."""
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False

            visited.add(task_id)
            rec_stack.add(task_id)

            # Get dependencies for current task
            deps = self._get_task_dependencies(task_id)
            for dep_id in deps:
                if has_cycle(dep_id):
                    return True

            rec_stack.remove(task_id)
            return False

        if has_cycle(task.task_id):
            return list(rec_stack)
        return []

    def _detect_file_conflicts(self, task: MaintenanceTask) -> List[str]:
        """Detect file conflicts with other scheduled tasks."""
        if not task.target_files:
            return []

        conflicts = []
        task_files = set(str(f) for f in task.target_files)

        # Check queued tasks
        for queued_task in self.task_queue:
            if queued_task.task_id == task.task_id:
                continue

            queued_files = set(str(f) for f in queued_task.target_files)
            if task_files & queued_files:  # Intersection
                conflicts.append(queued_task.task_id)

        # Check running tasks
        for running_task in self.running_tasks.values():
            running_files = set(str(f) for f in running_task.target_files)
            if task_files & running_files:
                conflicts.append(running_task.task_id)

        return conflicts

    def _find_off_peak_time(self) -> Optional[datetime]:
        """Find next available off-peak time slot."""
        now = datetime.now()
        # Assume off-peak is between 22:00 and 06:00
        if now.hour >= 22 or now.hour < 6:
            return now + timedelta(minutes=5)
        else:
            # Schedule for 22:00 today or tomorrow
            target_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
            if target_time <= now:
                target_time += timedelta(days=1)
            return target_time

    def _get_dependency_completion_time(self, task: MaintenanceTask) -> Optional[datetime]:
        """Get expected completion time of dependencies."""
        if not task.dependencies:
            return None

        max_completion_time = None
        conn = sqlite3.connect(self.db_path)
        try:
            for dep_id in task.dependencies:
                cursor = conn.execute('''
                    SELECT scheduled_for, estimated_duration_seconds, status
                    FROM maintenance_tasks WHERE task_id = ?
                ''', (dep_id,))
                result = cursor.fetchone()

                if result:
                    scheduled_for, duration_seconds, status = result
                    if status == TaskStatus.COMPLETED.value:
                        continue

                    if scheduled_for and duration_seconds:
                        completion_time = datetime.fromtimestamp(scheduled_for) + \
                                        timedelta(seconds=duration_seconds)
                        if max_completion_time is None or completion_time > max_completion_time:
                            max_completion_time = completion_time

        finally:
            conn.close()

        return max_completion_time

    def _get_task_dependencies(self, task_id: str) -> Set[str]:
        """Get dependencies for a task."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                'SELECT dependencies FROM maintenance_tasks WHERE task_id = ?',
                (task_id,)
            )
            result = cursor.fetchone()
            if result and result[0]:
                return set(json.loads(result[0]))
            return set()
        finally:
            conn.close()

    def _requeue_task(self, task: MaintenanceTask) -> None:
        """Requeue a task that couldn't be executed."""
        # Delay execution slightly to avoid busy waiting
        task.scheduled_for = datetime.now() + timedelta(minutes=5)
        heapq.heappush(self.task_queue, task)

    def _save_task(self, task: MaintenanceTask) -> None:
        """Save task to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT OR REPLACE INTO maintenance_tasks (
                    task_id, task_type, priority, title, description,
                    estimated_duration_seconds, target_files, dependencies, prerequisites,
                    created_at, scheduled_for, deadline, status, assigned_to,
                    progress, metadata, retry_count, max_retries, error_message, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id,
                task.task_type.value,
                task.priority.value,
                task.title,
                task.description,
                int(task.estimated_duration.total_seconds()),
                json.dumps([str(f) for f in task.target_files]),
                json.dumps(list(task.dependencies)),
                json.dumps(task.prerequisites),
                task.created_at.timestamp(),
                task.scheduled_for.timestamp() if task.scheduled_for else None,
                task.deadline.timestamp() if task.deadline else None,
                task.status.value,
                task.assigned_to,
                task.progress,
                json.dumps(task.metadata),
                task.retry_count,
                task.max_retries,
                task.error_message,
                task.completed_at.timestamp() if task.completed_at else None
            ))
            conn.commit()
        finally:
            conn.close()

    def _save_conflict(self, conflict: SchedulingConflict) -> None:
        """Save scheduling conflict to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT INTO scheduling_conflicts (
                    conflict_id, conflicting_tasks, conflict_type,
                    description, suggested_resolution, severity
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conflict.conflict_id,
                json.dumps(conflict.conflicting_tasks),
                conflict.conflict_type,
                conflict.description,
                conflict.suggested_resolution,
                conflict.severity
            ))
            conn.commit()
        finally:
            conn.close()

    def _log_task_action(self, task_id: str, action: str, details: str = "") -> None:
        """Log task action to history."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute('''
                INSERT INTO task_history (task_id, action, details)
                VALUES (?, ?, ?)
            ''', (task_id, action, details))
            conn.commit()
        finally:
            conn.close()

    def _load_tasks(self) -> None:
        """Load pending and scheduled tasks from database."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute('''
                SELECT * FROM maintenance_tasks
                WHERE status IN ('pending', 'scheduled', 'in_progress')
                ORDER BY priority DESC
            ''')

            for row in cursor.fetchall():
                task = self._row_to_task(row)
                if task.status == TaskStatus.IN_PROGRESS:
                    self.running_tasks[task.task_id] = task
                else:
                    heapq.heappush(self.task_queue, task)

        finally:
            conn.close()

    def _row_to_task(self, row) -> MaintenanceTask:
        """Convert database row to MaintenanceTask."""
        return MaintenanceTask(
            task_id=row[0],
            task_type=TaskType(row[1]),
            priority=TaskPriority(row[2]),
            title=row[3],
            description=row[4] or "",
            estimated_duration=timedelta(seconds=row[5]),
            target_files=[Path(f) for f in json.loads(row[6] or '[]')],
            dependencies=set(json.loads(row[7] or '[]')),
            prerequisites=json.loads(row[8] or '[]'),
            created_at=datetime.fromtimestamp(row[9]),
            scheduled_for=datetime.fromtimestamp(row[10]) if row[10] else None,
            deadline=datetime.fromtimestamp(row[11]) if row[11] else None,
            status=TaskStatus(row[12]),
            assigned_to=row[13],
            progress=row[14] or 0.0,
            metadata=json.loads(row[15] or '{}'),
            retry_count=row[16] or 0,
            max_retries=row[17] or 3,
            error_message=row[18],
            completed_at=datetime.fromtimestamp(row[19]) if row[19] else None
        )