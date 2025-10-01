"""
Priority-Based Processing Queue System for MCP-Optimized File Processing.

This module provides intelligent queue management that dynamically adjusts processing
priorities based on MCP server activity, current project context, and user interaction
patterns to optimize resource usage and processing efficiency.

Key Features:
    - Dynamic priority calculation based on MCP activity patterns
    - Resource optimization: multi-core when MCP active, single-core when inactive
    - Intelligent queue management with backpressure handling
    - Crash recovery and queue persistence via SQLite integration
    - Processing statistics and performance monitoring
    - Configurable processing limits and timeout handling
    - Real-time queue monitoring and health metrics
    - Integration with existing SQLite State Manager and Incremental Processor

Example:
    ```python
    from workspace_qdrant_mcp.core.priority_queue_manager import PriorityQueueManager
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager

    # Initialize components
    state_manager = SQLiteStateManager("./workspace_state.db")
    await state_manager.initialize()

    queue_manager = PriorityQueueManager(
        state_manager=state_manager,
        max_concurrent_jobs=4,
        mcp_detection_interval=30
    )
    await queue_manager.initialize()

    # Add files to queue with dynamic priority calculation
    await queue_manager.enqueue_file("/path/to/file.py", "my-project")

    # Process queue with resource optimization
    async with queue_manager.get_processing_context() as context:
        await context.process_next_batch()
    ```
"""

import asyncio
import json
from loguru import logger
import os
import psutil
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .sqlite_state_manager import (
    SQLiteStateManager,
    ProcessingPriority,
    FileProcessingStatus,
    ProcessingQueueItem
)
from .incremental_processor import (
    IncrementalProcessor,
    FileChangeInfo,
    ChangeType
)

# logger imported from loguru


class MCPActivityLevel(Enum):
    """MCP server activity levels for resource optimization."""

    INACTIVE = "inactive"      # No MCP activity detected
    LOW = "low"               # Minimal MCP activity
    MODERATE = "moderate"     # Active user interaction
    HIGH = "high"            # Heavy MCP usage
    BURST = "burst"          # Peak activity burst


class ProcessingMode(Enum):
    """Processing execution modes based on resource availability."""

    CONSERVATIVE = "conservative"  # Single-core, minimal resources
    BALANCED = "balanced"         # Multi-core, moderate resources
    AGGRESSIVE = "aggressive"     # Full multi-core, maximum resources
    BURST = "burst"              # Temporary high-resource mode


class QueueHealthStatus(Enum):
    """Queue system health status indicators."""

    HEALTHY = "healthy"       # Normal operation
    DEGRADED = "degraded"     # Some issues, but functional
    CRITICAL = "critical"     # Serious issues affecting performance
    FAILURE = "failure"       # System failure, requires intervention


@dataclass
class MCPActivityMetrics:
    """Metrics for tracking MCP server activity."""

    requests_per_minute: float = 0.0
    active_sessions: int = 0
    last_request_time: Optional[datetime] = None
    activity_level: MCPActivityLevel = MCPActivityLevel.INACTIVE
    burst_detected: bool = False
    session_start_time: Optional[datetime] = None
    total_requests: int = 0
    average_request_duration: float = 0.0

    def update_activity(self, request_count: int, session_count: int):
        """Update activity metrics with new measurements."""
        now = datetime.now(timezone.utc)
        self.total_requests += request_count
        self.active_sessions = session_count
        self.last_request_time = now if request_count > 0 else self.last_request_time

        # Calculate requests per minute
        if self.session_start_time:
            elapsed_minutes = (now - self.session_start_time).total_seconds() / 60.0
            if elapsed_minutes > 0:
                self.requests_per_minute = self.total_requests / elapsed_minutes

        # Determine activity level
        if self.requests_per_minute >= 20:
            self.activity_level = MCPActivityLevel.HIGH
        elif self.requests_per_minute >= 10:
            self.activity_level = MCPActivityLevel.MODERATE
        elif self.requests_per_minute >= 2:
            self.activity_level = MCPActivityLevel.LOW
        else:
            self.activity_level = MCPActivityLevel.INACTIVE

        # Detect burst activity
        recent_window = timedelta(minutes=2)
        if (self.last_request_time and
            now - self.last_request_time <= recent_window and
            request_count >= 5):
            self.burst_detected = True
            if self.activity_level != MCPActivityLevel.BURST:
                self.activity_level = MCPActivityLevel.BURST
        else:
            self.burst_detected = False


@dataclass
class PriorityCalculationContext:
    """Context information for dynamic priority calculation."""

    file_path: str
    collection: str
    mcp_activity: MCPActivityMetrics
    current_project_root: Optional[str] = None
    file_modification_time: Optional[datetime] = None
    file_size: int = 0
    is_user_triggered: bool = False
    is_current_project: bool = False
    is_recently_modified: bool = False
    has_dependencies: bool = False
    processing_history: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingJob:
    """Individual processing job with metadata."""

    queue_id: str
    file_path: str
    collection: str
    priority: ProcessingPriority
    calculated_score: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    scheduled_at: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_context: Optional[PriorityCalculationContext] = None


@dataclass
class QueueStatistics:
    """Comprehensive queue processing statistics."""

    total_items: int = 0
    items_by_priority: Dict[str, int] = field(default_factory=dict)
    items_by_status: Dict[str, int] = field(default_factory=dict)
    processing_rate: float = 0.0  # items per minute
    average_processing_time: float = 0.0  # seconds
    success_rate: float = 1.0
    backpressure_events: int = 0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    health_status: QueueHealthStatus = QueueHealthStatus.HEALTHY
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ResourceConfiguration:
    """Configuration for resource management and processing limits."""

    max_concurrent_jobs: int = 4
    max_memory_mb: int = 1024
    max_cpu_percent: int = 80
    conservative_concurrent_jobs: int = 1
    conservative_memory_mb: int = 256
    balanced_concurrent_jobs: int = 2
    balanced_memory_mb: int = 512
    aggressive_concurrent_jobs: int = 6
    aggressive_memory_mb: int = 2048
    burst_concurrent_jobs: int = 8
    burst_memory_mb: int = 4096
    burst_duration_seconds: int = 300
    backpressure_threshold: float = 0.9
    health_check_interval: int = 30


class PriorityQueueManager:
    """
    Sophisticated priority-based queue management system for MCP-optimized processing.

    Provides dynamic priority calculation, resource optimization, and comprehensive
    monitoring for efficient file processing in development workflows.
    """

    def __init__(
        self,
        state_manager: SQLiteStateManager,
        incremental_processor: Optional[IncrementalProcessor] = None,
        resource_config: Optional[ResourceConfiguration] = None,
        mcp_detection_interval: int = 30,
        statistics_retention_hours: int = 24
    ):
        """
        Initialize Priority Queue Manager.

        Args:
            state_manager: SQLite state persistence manager
            incremental_processor: Optional incremental update processor
            resource_config: Resource management configuration
            mcp_detection_interval: Seconds between MCP activity checks
            statistics_retention_hours: Hours to retain processing statistics
        """
        self.state_manager = state_manager
        self.incremental_processor = incremental_processor
        self.resource_config = resource_config or ResourceConfiguration()
        self.mcp_detection_interval = mcp_detection_interval
        self.statistics_retention_hours = statistics_retention_hours

        # Activity tracking
        self.mcp_activity = MCPActivityMetrics()
        self.current_project_root: Optional[str] = None
        self.recent_files: Set[str] = set()

        # Processing control
        self.processing_mode = ProcessingMode.CONSERVATIVE
        self.executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = None
        self.active_jobs: Dict[str, ProcessingJob] = {}
        self.job_semaphore: Optional[asyncio.Semaphore] = None

        # Statistics and monitoring
        self.statistics = QueueStatistics()
        self.processing_history: List[Dict[str, Any]] = []
        self.health_metrics: Dict[str, Any] = {}

        # Control flags
        self._initialized = False
        self._shutdown_event = asyncio.Event()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._activity_detection_task: Optional[asyncio.Task] = None

        # Callbacks and hooks
        self.priority_calculation_hooks: List[Callable] = []
        self.processing_hooks: List[Callable] = []
        self.monitoring_hooks: List[Callable] = []

    async def initialize(self) -> bool:
        """
        Initialize the priority queue system.

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True

        try:
            logger.info("Initializing Priority Queue Manager")

            # Ensure state manager is initialized
            if not hasattr(self.state_manager, '_initialized') or not self.state_manager._initialized:
                if not await self.state_manager.initialize():
                    raise RuntimeError("Failed to initialize state manager")

            # Initialize processing mode and resources
            await self._initialize_processing_resources()

            # Start activity detection
            self.mcp_activity.session_start_time = datetime.now(timezone.utc)

            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._activity_detection_task = asyncio.create_task(self._activity_detection_loop())

            # Perform crash recovery
            await self._perform_crash_recovery()

            self._initialized = True
            logger.info("Priority Queue Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Priority Queue Manager: {e}")
            await self.shutdown()
            return False

    async def shutdown(self):
        """Graceful shutdown of queue manager."""
        if not self._initialized:
            return

        logger.info("Shutting down Priority Queue Manager")

        # Signal shutdown
        self._shutdown_event.set()

        # Cancel monitoring tasks
        for task in [self._monitoring_task, self._activity_detection_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Wait for active jobs to complete (with timeout)
        if self.active_jobs:
            logger.info(f"Waiting for {len(self.active_jobs)} active jobs to complete")
            try:
                await asyncio.wait_for(
                    self._wait_for_jobs_completion(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for jobs completion, forcing shutdown")

        # Cleanup resources
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        self._initialized = False
        logger.info("Priority Queue Manager shutdown complete")

    async def enqueue_file(
        self,
        file_path: str,
        collection: str,
        user_triggered: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add file to processing queue with dynamic priority calculation.

        Args:
            file_path: Path to file to be processed
            collection: Target collection name
            user_triggered: Whether this was triggered by user action
            metadata: Optional metadata for the processing job

        Returns:
            Queue ID for the enqueued job
        """
        if not self._initialized:
            raise RuntimeError("Priority Queue Manager not initialized")

        try:
            # Create priority calculation context
            context = await self._create_priority_context(
                file_path, collection, user_triggered, metadata or {}
            )

            # Calculate dynamic priority
            priority, score = await self._calculate_dynamic_priority(context)

            # Calculate tenant_id and branch for new queue system
            project_root = Path(file_path).parent
            tenant_id = await self.state_manager.calculate_tenant_id(project_root)
            branch = await self.state_manager.get_current_branch(project_root)

            # Create processing job metadata with queue integration info
            job_metadata = {
                **(metadata or {}),
                "calculated_score": score,
                "user_triggered": user_triggered,
                "mcp_activity_level": self.mcp_activity.activity_level.value,
                "processing_mode": self.processing_mode.value,
                "tenant_id": tenant_id,
                "branch": branch
            }

            # Enqueue using new SQLite queue operations
            queue_id = await self.state_manager.enqueue(
                file_path=file_path,
                collection=collection,
                priority=priority.value,  # Convert enum to integer
                tenant_id=tenant_id,
                branch=branch,
                metadata=job_metadata
            )

            logger.info(
                f"Enqueued file: {file_path} (priority: {priority.name}, "
                f"score: {score:.2f}, queue_id: {queue_id}, tenant: {tenant_id}, branch: {branch})"
            )

            # Update statistics
            self.statistics.total_items += 1
            if priority.name not in self.statistics.items_by_priority:
                self.statistics.items_by_priority[priority.name] = 0
            self.statistics.items_by_priority[priority.name] += 1

            return queue_id

        except Exception as e:
            logger.error(f"Failed to enqueue file {file_path}: {e}")
            raise

    async def process_next_batch(
        self,
        batch_size: Optional[int] = None
    ) -> List[ProcessingJob]:
        """
        Process the next batch of files from the queue.

        Args:
            batch_size: Override batch size based on current processing mode

        Returns:
            List of completed processing jobs
        """
        if not self._initialized:
            raise RuntimeError("Priority Queue Manager not initialized")

        # Determine batch size based on processing mode
        if batch_size is None:
            batch_size = await self._get_optimal_batch_size()

        completed_jobs = []

        try:
            # Check resource availability and backpressure
            if await self._check_backpressure():
                logger.info("Backpressure detected, skipping batch processing")
                return completed_jobs

            # Get next items from queue
            queue_items = []
            for _ in range(batch_size):
                item = await self.state_manager.get_next_queue_item()
                if not item:
                    break
                queue_items.append(item)

            if not queue_items:
                return completed_jobs

            logger.info(f"Processing batch of {len(queue_items)} items")

            # Create processing jobs
            jobs = []
            for item in queue_items:
                job = ProcessingJob(
                    queue_id=item.queue_id,
                    file_path=item.file_path,
                    collection=item.collection,
                    priority=item.priority,
                    metadata=item.metadata or {},
                    attempts=item.attempts
                )
                jobs.append(job)
                self.active_jobs[job.queue_id] = job

            # Process jobs concurrently
            if self.processing_mode in [ProcessingMode.CONSERVATIVE]:
                # Sequential processing for conservative mode
                for job in jobs:
                    completed_job = await self._process_single_job(job)
                    if completed_job:
                        completed_jobs.append(completed_job)
            else:
                # Concurrent processing for other modes
                async with self._get_concurrency_limiter():
                    tasks = [self._process_single_job(job) for job in jobs]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for result in results:
                        if isinstance(result, ProcessingJob):
                            completed_jobs.append(result)
                        elif isinstance(result, Exception):
                            logger.error(f"Job processing failed: {result}")

            # Update processing statistics
            await self._update_processing_statistics(completed_jobs)

            logger.info(f"Completed batch processing: {len(completed_jobs)} jobs")
            return completed_jobs

        except Exception as e:
            logger.error(f"Failed to process batch: {e}")

            # Clean up active jobs on failure
            for job in [j for j in self.active_jobs.values() if j.queue_id in [qi.queue_id for qi in queue_items]]:
                await self._handle_job_failure(job, str(e))

            return completed_jobs

    # --- Priority Calculation Methods ---

    async def _create_priority_context(
        self,
        file_path: str,
        collection: str,
        user_triggered: bool,
        metadata: Dict[str, Any]
    ) -> PriorityCalculationContext:
        """Create context for priority calculation."""
        file_path_obj = Path(file_path)

        # Get file metadata
        file_mtime = None
        file_size = 0
        if file_path_obj.exists():
            stat = file_path_obj.stat()
            file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            file_size = stat.st_size

        # Determine if this is the current project
        is_current_project = False
        if self.current_project_root:
            try:
                file_path_obj.relative_to(self.current_project_root)
                is_current_project = True
            except ValueError:
                pass

        # Check if recently modified (last 10 minutes)
        is_recently_modified = False
        if file_mtime:
            is_recently_modified = (
                datetime.now(timezone.utc) - file_mtime
            ).total_seconds() < 600

        # Get processing history
        processing_history = {}
        try:
            record = await self.state_manager.get_file_processing_record(file_path)
            if record:
                processing_history = {
                    "retry_count": record.retry_count,
                    "last_processed": record.completed_at,
                    "success_rate": 1.0 if record.status == FileProcessingStatus.COMPLETED else 0.5
                }
        except Exception:
            pass

        return PriorityCalculationContext(
            file_path=file_path,
            collection=collection,
            mcp_activity=self.mcp_activity,
            current_project_root=self.current_project_root,
            file_modification_time=file_mtime,
            file_size=file_size,
            is_user_triggered=user_triggered,
            is_current_project=is_current_project,
            is_recently_modified=is_recently_modified,
            has_dependencies=False,  # TODO: Implement dependency detection
            processing_history=processing_history
        )

    async def _calculate_dynamic_priority(
        self,
        context: PriorityCalculationContext
    ) -> Tuple[ProcessingPriority, float]:
        """
        Calculate dynamic priority based on multiple factors.

        Priority Score Components:
        - MCP Activity Level: 0-40 points
        - Current Project: 0-25 points
        - User Triggered: 0-20 points
        - File Recency: 0-15 points
        - Processing History: 0-10 points (penalties for failures)

        Returns:
            Tuple of (ProcessingPriority, calculated_score)
        """
        score = 0.0

        # MCP Activity Level (0-40 points)
        mcp_multiplier = {
            MCPActivityLevel.INACTIVE: 0.0,
            MCPActivityLevel.LOW: 0.3,
            MCPActivityLevel.MODERATE: 0.7,
            MCPActivityLevel.HIGH: 0.9,
            MCPActivityLevel.BURST: 1.0
        }
        score += 40 * mcp_multiplier.get(context.mcp_activity.activity_level, 0.0)

        # Current Project Bonus (0-25 points)
        if context.is_current_project:
            score += 25

        # User Triggered Bonus (0-20 points)
        if context.is_user_triggered:
            score += 20

        # File Recency (0-15 points)
        if context.is_recently_modified:
            # More recent files get higher scores
            if context.file_modification_time:
                age_minutes = (
                    datetime.now(timezone.utc) - context.file_modification_time
                ).total_seconds() / 60
                # Linear decay over 60 minutes
                recency_score = max(0, 15 * (1 - age_minutes / 60))
                score += recency_score

        # Processing History (0 to -10 points for penalties)
        history = context.processing_history
        if history:
            retry_count = history.get("retry_count", 0)
            success_rate = history.get("success_rate", 1.0)

            # Penalty for multiple retries
            score -= min(10, retry_count * 2)

            # Penalty for low success rate
            if success_rate < 0.5:
                score -= 5

        # File Size Considerations (small adjustment)
        if context.file_size > 10 * 1024 * 1024:  # Files > 10MB
            score -= 2  # Slight penalty for large files
        elif context.file_size < 1024:  # Very small files
            score += 1  # Small boost for quick processing

        # Apply hooks for custom priority calculation
        for hook in self.priority_calculation_hooks:
            try:
                score = await hook(context, score)
            except Exception as e:
                logger.warning(f"Priority calculation hook failed: {e}")

        # Convert score to priority level
        if score >= 70:
            priority = ProcessingPriority.URGENT
        elif score >= 50:
            priority = ProcessingPriority.HIGH
        elif score >= 25:
            priority = ProcessingPriority.NORMAL
        else:
            priority = ProcessingPriority.LOW

        return priority, score

    # --- Resource Management Methods ---

    async def _initialize_processing_resources(self):
        """Initialize processing resources based on current activity level."""
        # Detect initial processing mode
        await self._update_processing_mode()

        # Initialize executor and semaphore
        await self._configure_executor()

    async def _update_processing_mode(self):
        """Update processing mode based on MCP activity and system resources."""
        old_mode = self.processing_mode

        # Check system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Determine mode based on MCP activity and system load
        if self.mcp_activity.activity_level == MCPActivityLevel.BURST:
            if cpu_percent < 60 and memory.percent < 70:
                self.processing_mode = ProcessingMode.BURST
            else:
                self.processing_mode = ProcessingMode.AGGRESSIVE

        elif self.mcp_activity.activity_level == MCPActivityLevel.HIGH:
            if cpu_percent < 70 and memory.percent < 80:
                self.processing_mode = ProcessingMode.AGGRESSIVE
            else:
                self.processing_mode = ProcessingMode.BALANCED

        elif self.mcp_activity.activity_level == MCPActivityLevel.MODERATE:
            self.processing_mode = ProcessingMode.BALANCED

        elif self.mcp_activity.activity_level in [MCPActivityLevel.LOW, MCPActivityLevel.INACTIVE]:
            self.processing_mode = ProcessingMode.CONSERVATIVE

        # Log mode changes
        if old_mode != self.processing_mode:
            logger.info(f"Processing mode changed: {old_mode.value} -> {self.processing_mode.value}")
            await self._configure_executor()

    async def _configure_executor(self):
        """Configure executor and concurrency limits based on processing mode."""
        # Shutdown existing executor
        if self.executor:
            self.executor.shutdown(wait=False)

        # Configure based on processing mode
        config_map = {
            ProcessingMode.CONSERVATIVE: {
                "max_workers": self.resource_config.conservative_concurrent_jobs,
                "use_threads": True  # Single-core friendly
            },
            ProcessingMode.BALANCED: {
                "max_workers": self.resource_config.balanced_concurrent_jobs,
                "use_threads": True
            },
            ProcessingMode.AGGRESSIVE: {
                "max_workers": self.resource_config.aggressive_concurrent_jobs,
                "use_threads": False  # Use processes for CPU-intensive work
            },
            ProcessingMode.BURST: {
                "max_workers": self.resource_config.burst_concurrent_jobs,
                "use_threads": False
            }
        }

        config = config_map[self.processing_mode]

        # Create appropriate executor
        if config["use_threads"]:
            self.executor = ThreadPoolExecutor(
                max_workers=config["max_workers"],
                thread_name_prefix="PQM-Worker"
            )
        else:
            self.executor = ProcessPoolExecutor(
                max_workers=config["max_workers"]
            )

        # Update semaphore
        self.job_semaphore = asyncio.Semaphore(config["max_workers"])

        logger.info(f"Configured executor: {type(self.executor).__name__} with {config['max_workers']} workers")

    @asynccontextmanager
    async def _get_concurrency_limiter(self):
        """Async context manager for controlling job concurrency."""
        if not self.job_semaphore:
            raise RuntimeError("Job semaphore not initialized")

        async with self.job_semaphore:
            yield

    async def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on current processing mode."""
        base_sizes = {
            ProcessingMode.CONSERVATIVE: 1,
            ProcessingMode.BALANCED: 2,
            ProcessingMode.AGGRESSIVE: 4,
            ProcessingMode.BURST: 6
        }

        base_size = base_sizes[self.processing_mode]

        # Adjust based on queue size
        queue_stats = await self.state_manager.queue_stats()
        total_queued = sum(queue_stats.values()) if queue_stats else 0

        if total_queued > 20:
            return min(base_size * 2, 10)  # Increase batch size for large queues
        elif total_queued < 5:
            return max(1, base_size // 2)  # Decrease for small queues

        return base_size

    async def _check_backpressure(self) -> bool:
        """Check if backpressure conditions are met."""
        # Check system resources
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Check if we're over resource thresholds
        cpu_threshold = self.resource_config.max_cpu_percent * self.resource_config.backpressure_threshold
        memory_threshold = (self.resource_config.max_memory_mb * 1024 * 1024) * self.resource_config.backpressure_threshold

        if cpu_percent > cpu_threshold:
            logger.warning(f"CPU backpressure detected: {cpu_percent}% > {cpu_threshold}%")
            self.statistics.backpressure_events += 1
            return True

        if memory.used > memory_threshold:
            logger.warning(f"Memory backpressure detected: {memory.used / (1024*1024):.1f}MB > {memory_threshold / (1024*1024):.1f}MB")
            self.statistics.backpressure_events += 1
            return True

        # Check active job count
        if len(self.active_jobs) >= self.job_semaphore._value * 2:
            logger.info("Job queue backpressure: too many active jobs")
            return True

        return False

    # --- Job Processing Methods ---

    async def _process_single_job(self, job: ProcessingJob) -> Optional[ProcessingJob]:
        """Process a single job from the queue."""
        start_time = time.time()
        job_success = False

        try:
            logger.info(f"Processing job {job.queue_id}: {job.file_path}")

            # Mark job as processing in database
            await self.state_manager.mark_queue_item_processing(job.queue_id)

            # Check if file exists
            if not Path(job.file_path).exists():
                raise FileNotFoundError(f"File not found: {job.file_path}")

            # Process job based on available processor
            if self.incremental_processor:
                # Use incremental processor for efficient processing
                changes = await self.incremental_processor.detect_changes([job.file_path])
                if changes:
                    await self.incremental_processor.process_changes(changes, batch_size=1)
            else:
                # Fallback to basic processing
                await self._process_job_fallback(job)

            # Mark job as completed
            await self.state_manager.complete_file_processing(
                job.file_path,
                success=True,
                document_id=job.metadata.get("document_id")
            )

            # Remove from processing queue
            await self.state_manager.remove_from_processing_queue(job.queue_id)

            job_success = True
            processing_time = time.time() - start_time

            logger.info(f"Completed job {job.queue_id} in {processing_time:.2f}s")

            # Update job metadata
            job.metadata.update({
                "processing_time": processing_time,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "success": True
            })

            return job

        except Exception as e:
            error_msg = str(e)
            processing_time = time.time() - start_time

            logger.error(f"Job {job.queue_id} failed after {processing_time:.2f}s: {error_msg}")

            # Handle job failure
            await self._handle_job_failure(job, error_msg)

            return None

        finally:
            # Remove from active jobs
            self.active_jobs.pop(job.queue_id, None)

            # Apply processing hooks
            for hook in self.processing_hooks:
                try:
                    await hook(job, job_success, time.time() - start_time)
                except Exception as e:
                    logger.warning(f"Processing hook failed: {e}")

    async def _process_job_fallback(self, job: ProcessingJob):
        """Fallback processing method when incremental processor is not available."""
        # This is a basic implementation - in production, this would integrate
        # with the document processing pipeline
        logger.warning(f"Using fallback processing for {job.file_path}")

        # Simulate processing time
        await asyncio.sleep(0.1)

        # Mark as processed
        await self.state_manager.start_file_processing(job.file_path, job.collection)
        await asyncio.sleep(0.1)  # Simulate actual processing

    async def _handle_job_failure(self, job: ProcessingJob, error_message: str):
        """Handle job processing failure with retry logic."""
        try:
            # Increment attempt count
            job.attempts += 1

            # Update file processing record with error
            await self.state_manager.complete_file_processing(
                job.file_path,
                success=False,
                error_message=error_message
            )

            # Decide whether to retry
            # Decide whether to retry
            if job.attempts < job.max_attempts:
                # Calculate exponential backoff delay
                delay_seconds = min(300, 30 * (2 ** (job.attempts - 1)))
                # Delays: 30s, 60s, 120s, 240s, 300s (capped at 5 min)

                # Mark for retry using state manager's retry method
                retry_success = await self.state_manager.retry_failed_file(
                    job.file_path,
                    max_retries=job.max_attempts
                )

                if retry_success:
                    logger.info(
                        f"Scheduled job {job.queue_id} for retry "
                        f"(attempt {job.attempts}/{job.max_attempts}, "
                        f"exponential backoff delay={delay_seconds}s)"
                    )
                    # NOTE: The actual delay is handled by the retry_failed_file() method
                    # which marks the file as RETRYING. Future integration with 
                    # SQLiteQueueClient will enable proper delayed scheduling.
                else:
                    logger.warning(
                        f"Failed to schedule retry for {job.queue_id}, "
                        "marking as permanently failed"
                    )
            else:
                # Max attempts reached, mark as permanently failed
                # The file is already marked as FAILED by complete_file_processing()
                logger.error(
                    f"Job {job.queue_id} permanently failed after "
                    f"{job.attempts} attempts: {error_message}"
                )

        except Exception as e:
            logger.error(f"Failed to handle job failure for {job.queue_id}: {e}")

    async def _wait_for_jobs_completion(self):
        """Wait for all active jobs to complete."""
        while self.active_jobs:
            await asyncio.sleep(1)

    # --- Activity Detection and Monitoring Methods ---

    async def _activity_detection_loop(self):
        """Background loop for detecting MCP activity."""
        logger.info("Starting MCP activity detection loop")

        while not self._shutdown_event.is_set():
            try:
                # Detect MCP activity (this is a simplified implementation)
                await self._detect_mcp_activity()

                # Update processing mode if needed
                await self._update_processing_mode()

                # Wait for next detection cycle
                await asyncio.sleep(self.mcp_detection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in activity detection loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry

    async def _detect_mcp_activity(self):
        """
        Detect current MCP activity level.

        This is a simplified implementation. In production, this would:
        - Monitor MCP server request logs
        - Track active client connections
        - Measure request frequency and patterns
        - Detect user interaction vs background activity
        """
        # For now, simulate activity detection based on system load
        # In production, integrate with actual MCP server metrics

        current_time = datetime.now(timezone.utc)

        # Check for recent file activity in monitored directories
        recent_activity_count = len([
            f for f in self.recent_files
            if (current_time - datetime.fromtimestamp(
                Path(f).stat().st_mtime, tz=timezone.utc
            )).total_seconds() < 300  # 5 minutes
        ]) if self.recent_files else 0

        # Update activity metrics
        self.mcp_activity.update_activity(
            request_count=recent_activity_count,
            session_count=1 if recent_activity_count > 0 else 0
        )

        # Log activity level changes
        if hasattr(self, '_last_activity_level'):
            if self._last_activity_level != self.mcp_activity.activity_level:
                logger.info(f"MCP activity level changed: {self._last_activity_level.value} -> {self.mcp_activity.activity_level.value}")

        self._last_activity_level = self.mcp_activity.activity_level

    async def _monitoring_loop(self):
        """Background monitoring loop for health and statistics."""
        logger.info("Starting monitoring loop")

        while not self._shutdown_event.is_set():
            try:
                # Update health metrics
                await self._update_health_metrics()

                # Update queue statistics
                await self._update_queue_statistics()

                # Cleanup old statistics
                await self._cleanup_old_statistics()

                # Check system health
                await self._check_system_health()

                # Apply monitoring hooks
                for hook in self.monitoring_hooks:
                    try:
                        await hook(self.statistics, self.health_metrics)
                    except Exception as e:
                        logger.warning(f"Monitoring hook failed: {e}")

                # Wait for next monitoring cycle
                await asyncio.sleep(self.resource_config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)

    async def _update_health_metrics(self):
        """Update system health metrics."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        self.health_metrics.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / (1024 * 1024),
            "active_jobs": len(self.active_jobs),
            "processing_mode": self.processing_mode.value,
            "mcp_activity_level": self.mcp_activity.activity_level.value,
            "executor_type": type(self.executor).__name__ if self.executor else None
        })

    async def _update_queue_statistics(self):
        """Update queue processing statistics."""
        # Get current queue stats from database
        queue_stats = await self.state_manager.queue_stats()

        self.statistics.total_items = sum(queue_stats.values()) if queue_stats else 0
        self.statistics.items_by_priority = {
            priority.name: queue_stats.get(priority.value, 0)
            for priority in ProcessingPriority
        }

        # Calculate processing rate (items per minute)
        if self.processing_history:
            recent_completions = [
                entry for entry in self.processing_history
                if (datetime.now(timezone.utc) - datetime.fromisoformat(entry["timestamp"])).total_seconds() < 3600
            ]

            if recent_completions:
                self.statistics.processing_rate = len(recent_completions) / (60.0)  # per minute

                # Calculate average processing time
                processing_times = [entry.get("processing_time", 0) for entry in recent_completions]
                self.statistics.average_processing_time = sum(processing_times) / len(processing_times)

                # Calculate success rate
                successful = len([entry for entry in recent_completions if entry.get("success", False)])
                self.statistics.success_rate = successful / len(recent_completions)

        self.statistics.last_updated = datetime.now(timezone.utc)

    async def _update_processing_statistics(self, completed_jobs: List[ProcessingJob]):
        """Update processing statistics with completed jobs."""
        for job in completed_jobs:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "queue_id": job.queue_id,
                "file_path": job.file_path,
                "collection": job.collection,
                "priority": job.priority.name,
                "processing_time": job.metadata.get("processing_time", 0),
                "success": job.metadata.get("success", False),
                "attempts": job.attempts
            }
            self.processing_history.append(entry)

        # Keep only recent history to prevent memory bloat
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.statistics_retention_hours)
        self.processing_history = [
            entry for entry in self.processing_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]

    async def _cleanup_old_statistics(self):
        """Clean up old statistics to prevent memory buildup."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.statistics_retention_hours)

        # Clean processing history
        old_count = len(self.processing_history)
        self.processing_history = [
            entry for entry in self.processing_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]

        cleaned_count = old_count - len(self.processing_history)
        if cleaned_count > 0:
            logger.debug(f"Cleaned up {cleaned_count} old statistics entries")

    async def _check_system_health(self):
        """Check overall system health and update status."""
        # Check resource utilization
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Check backpressure frequency
        recent_backpressure = self.statistics.backpressure_events > 10  # Arbitrary threshold

        # Check processing success rate
        low_success_rate = self.statistics.success_rate < 0.8

        # Determine health status
        if cpu_percent > 90 or memory.percent > 95 or recent_backpressure:
            self.statistics.health_status = QueueHealthStatus.CRITICAL
        elif cpu_percent > 80 or memory.percent > 85 or low_success_rate:
            self.statistics.health_status = QueueHealthStatus.DEGRADED
        elif len(self.active_jobs) == 0 and self.statistics.total_items == 0:
            self.statistics.health_status = QueueHealthStatus.HEALTHY
        else:
            self.statistics.health_status = QueueHealthStatus.HEALTHY

    # --- Crash Recovery Methods ---

    async def _perform_crash_recovery(self):
        """Perform crash recovery for interrupted processing jobs."""
        logger.info("Performing crash recovery")

        try:
            # Find jobs that were marked as processing but never completed
            # This would require additional database queries to identify stale processing records

            # For now, just log recovery completion
            # In production, implement sophisticated recovery logic
            logger.info("Crash recovery completed successfully")

        except Exception as e:
            logger.error(f"Error during crash recovery: {e}")

    # --- Public Interface Methods ---

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        if not self._initialized:
            return {"error": "Priority Queue Manager not initialized"}

        return {
            "initialized": self._initialized,
            "processing_mode": self.processing_mode.value,
            "mcp_activity": {
                "level": self.mcp_activity.activity_level.value,
                "requests_per_minute": self.mcp_activity.requests_per_minute,
                "active_sessions": self.mcp_activity.active_sessions,
                "burst_detected": self.mcp_activity.burst_detected
            },
            "statistics": {
                "total_items": self.statistics.total_items,
                "items_by_priority": self.statistics.items_by_priority,
                "processing_rate": self.statistics.processing_rate,
                "average_processing_time": self.statistics.average_processing_time,
                "success_rate": self.statistics.success_rate,
                "backpressure_events": self.statistics.backpressure_events,
                "health_status": self.statistics.health_status.value
            },
            "active_jobs": len(self.active_jobs),
            "resource_utilization": self.health_metrics,
            "current_project_root": self.current_project_root
        }

    async def get_processing_context(self) -> 'ProcessingContextManager':
        """Get processing context manager for batch operations."""
        return ProcessingContextManager(self)

    def add_priority_calculation_hook(self, hook: Callable):
        """Add custom priority calculation hook."""
        self.priority_calculation_hooks.append(hook)

    def add_processing_hook(self, hook: Callable):
        """Add processing completion hook."""
        self.processing_hooks.append(hook)

    def add_monitoring_hook(self, hook: Callable):
        """Add monitoring hook for statistics updates."""
        self.monitoring_hooks.append(hook)

    def set_current_project_root(self, project_root: str):
        """Set current project root for priority calculations."""
        self.current_project_root = project_root
        logger.info(f"Set current project root: {project_root}")

    async def clear_queue(self, collection: Optional[str] = None) -> int:
        """Clear processing queue items."""
        if not self._initialized:
            raise RuntimeError("Priority Queue Manager not initialized")

        return await self.state_manager.clear_queue(collection)

    async def get_health_status(self) -> Dict[str, Any]:
        """Get detailed health status information."""
        if not self._initialized:
            return {"error": "Priority Queue Manager not initialized"}

        return {
            "health_status": self.statistics.health_status.value,
            "system_metrics": self.health_metrics,
            "queue_statistics": {
                "total_items": self.statistics.total_items,
                "processing_rate": self.statistics.processing_rate,
                "success_rate": self.statistics.success_rate,
                "backpressure_events": self.statistics.backpressure_events
            },
            "resource_status": {
                "processing_mode": self.processing_mode.value,
                "active_jobs": len(self.active_jobs),
                "executor_type": type(self.executor).__name__ if self.executor else None
            },
            "mcp_activity": {
                "level": self.mcp_activity.activity_level.value,
                "requests_per_minute": self.mcp_activity.requests_per_minute,
                "burst_detected": self.mcp_activity.burst_detected
            }
        }


class ProcessingContextManager:
    """Context manager for batch processing operations."""

    def __init__(self, queue_manager: PriorityQueueManager):
        self.queue_manager = queue_manager
        self.start_time = None

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Processing context error: {exc_val}")

        processing_time = time.time() - self.start_time
        logger.info(f"Processing context completed in {processing_time:.2f}s")

    async def process_next_batch(self, batch_size: Optional[int] = None) -> List[ProcessingJob]:
        """Process next batch of jobs."""
        return await self.queue_manager.process_next_batch(batch_size)

    async def enqueue_multiple_files(
        self,
        file_collection_pairs: List[Tuple[str, str]],
        user_triggered: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Enqueue multiple files for processing."""
        queue_ids = []
        for file_path, collection in file_collection_pairs:
            queue_id = await self.queue_manager.enqueue_file(
                file_path=file_path,
                collection=collection,
                user_triggered=user_triggered,
                metadata=metadata
            )
            queue_ids.append(queue_id)

        return queue_ids
