"""
Batch Processing Manager for High-Performance File Watching

This module provides intelligent batch processing capabilities for handling
bulk file changes efficiently. It implements adaptive batching strategies,
queue management, and performance optimization for high-throughput scenarios.

Features:
    - Adaptive batch sizing based on system load and file types
    - Intelligent queuing with priority and deduplication
    - Memory-efficient processing with backpressure control
    - Performance monitoring and throughput optimization
    - Graceful degradation under high load
    - Configurable processing strategies (time-based, count-based, hybrid)
    - Error handling and retry mechanisms

Example:
    ```python
    from workspace_qdrant_mcp.core.batch_processing_manager import BatchProcessingManager

    # Initialize with adaptive configuration
    manager = BatchProcessingManager(
        max_batch_size=100,
        max_batch_age_seconds=5.0,
        max_queue_size=10000,
        processing_strategy="adaptive"
    )

    # Set processing callback
    async def process_batch(files):
        for file_path, collection in files:
            await ingest_document(file_path, collection)

    manager.set_processing_callback(process_batch)
    await manager.start()

    # Add files for batch processing
    await manager.add_file("/path/to/file.txt", "collection-name")
    ```
"""

import asyncio
import gc
import heapq
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple, NamedTuple
import hashlib
import weakref

from loguru import logger

# Performance monitoring imports
try:
    from .performance_monitor import PerformanceMonitor
    from .performance_metrics import PerformanceMetricsCollector, MetricType
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Graceful degradation imports
try:
    from .graceful_degradation import GracefulDegradationManager
    DEGRADATION_AVAILABLE = True
except ImportError:
    DEGRADATION_AVAILABLE = False


class ProcessingStrategy(Enum):
    """Processing strategy for batch operations."""
    TIME_BASED = "time_based"      # Flush batches based on age
    COUNT_BASED = "count_based"    # Flush batches based on size
    HYBRID = "hybrid"              # Combined time and count-based
    ADAPTIVE = "adaptive"          # Dynamic adaptation based on load
    PRIORITY = "priority"          # Priority-based processing


class BatchPriority(Enum):
    """Priority levels for batch processing."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class FileOperation(NamedTuple):
    """Represents a file operation for batch processing."""
    file_path: str
    collection: str
    operation_type: str  # "add", "modify", "delete"
    priority: BatchPriority = BatchPriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BatchItem:
    """Individual item in a processing batch."""
    file_operation: FileOperation
    retry_count: int = 0
    last_error: Optional[str] = None
    added_at: float = field(default_factory=time.time)
    processing_duration: float = 0.0

    def __hash__(self) -> int:
        """Hash based on file path and operation type for deduplication."""
        return hash((self.file_operation.file_path, self.file_operation.operation_type))

    def __eq__(self, other) -> bool:
        """Equality based on file path and operation type."""
        if not isinstance(other, BatchItem):
            return False
        return (
            self.file_operation.file_path == other.file_operation.file_path and
            self.file_operation.operation_type == other.file_operation.operation_type
        )


@dataclass
class ProcessingBatch:
    """A batch of items for processing."""
    items: List[BatchItem] = field(default_factory=list)
    batch_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    created_at: float = field(default_factory=time.time)
    priority: BatchPriority = BatchPriority.NORMAL
    collection: str = ""
    estimated_processing_time: float = 0.0

    def add_item(self, item: BatchItem) -> None:
        """Add an item to the batch with deduplication."""
        # Remove duplicate if exists (keep the newer one)
        existing_items = [i for i in self.items if i == item]
        for existing_item in existing_items:
            self.items.remove(existing_item)

        # Add new item
        self.items.append(item)

        # Update batch priority to highest item priority
        if item.file_operation.priority.value > self.priority.value:
            self.priority = item.file_operation.priority

        # Update collection if not set
        if not self.collection:
            self.collection = item.file_operation.collection

    def age_seconds(self) -> float:
        """Get age of batch in seconds."""
        return time.time() - self.created_at

    def size(self) -> int:
        """Get number of items in batch."""
        return len(self.items)

    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return len(self.items) == 0

    def get_file_paths(self) -> List[str]:
        """Get list of file paths in batch."""
        return [item.file_operation.file_path for item in self.items]

    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # Rough estimate based on number of items and average path length
        avg_path_length = sum(len(item.file_operation.file_path) for item in self.items) / max(1, len(self.items))
        base_overhead = len(self.items) * 200  # Base object overhead per item
        path_memory = len(self.items) * avg_path_length * 2  # Unicode string overhead
        metadata_memory = sum(
            len(str(item.file_operation.metadata)) if item.file_operation.metadata else 0
            for item in self.items
        )
        total_bytes = base_overhead + path_memory + metadata_memory
        return total_bytes / (1024 * 1024)  # Convert to MB


@dataclass
class BatchProcessingStatistics:
    """Statistics for batch processing operations."""
    total_files_processed: int = 0
    total_batches_processed: int = 0
    total_files_failed: int = 0
    total_batches_failed: int = 0
    avg_batch_size: float = 0.0
    avg_batch_processing_time: float = 0.0
    avg_queue_wait_time: float = 0.0
    throughput_files_per_second: float = 0.0
    throughput_batches_per_second: float = 0.0
    current_queue_size: int = 0
    current_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0
    duplicate_files_merged: int = 0
    priority_promotions: int = 0

    # Rolling windows for recent performance
    recent_batch_sizes: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_processing_times: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_wait_times: deque = field(default_factory=lambda: deque(maxlen=100))

    def add_batch_processed(self, batch_size: int, processing_time: float, wait_time: float) -> None:
        """Add a processed batch to statistics."""
        self.total_files_processed += batch_size
        self.total_batches_processed += 1

        self.recent_batch_sizes.append(batch_size)
        self.recent_processing_times.append(processing_time)
        self.recent_wait_times.append(wait_time)

        # Update averages
        if self.recent_batch_sizes:
            self.avg_batch_size = sum(self.recent_batch_sizes) / len(self.recent_batch_sizes)
        if self.recent_processing_times:
            self.avg_batch_processing_time = sum(self.recent_processing_times) / len(self.recent_processing_times)
        if self.recent_wait_times:
            self.avg_queue_wait_time = sum(self.recent_wait_times) / len(self.recent_wait_times)

        # Calculate throughput
        if processing_time > 0:
            self.throughput_files_per_second = batch_size / processing_time
            self.throughput_batches_per_second = 1.0 / processing_time

    def add_batch_failed(self, batch_size: int) -> None:
        """Add a failed batch to statistics."""
        self.total_files_failed += batch_size
        self.total_batches_failed += 1

    def update_memory_usage(self, current_mb: float) -> None:
        """Update memory usage statistics."""
        self.current_memory_usage_mb = current_mb
        if current_mb > self.peak_memory_usage_mb:
            self.peak_memory_usage_mb = current_mb

    def get_efficiency_metrics(self) -> Dict[str, float]:
        """Get efficiency and performance metrics."""
        success_rate = (
            (self.total_files_processed / max(1, self.total_files_processed + self.total_files_failed)) * 100
        )

        batch_success_rate = (
            (self.total_batches_processed / max(1, self.total_batches_processed + self.total_batches_failed)) * 100
        )

        return {
            "file_success_rate_percent": success_rate,
            "batch_success_rate_percent": batch_success_rate,
            "avg_batch_utilization": self.avg_batch_size,
            "processing_efficiency": self.throughput_files_per_second,
            "memory_efficiency": self.current_memory_usage_mb / max(1, self.current_queue_size),
            "queue_efficiency": max(0, 100 - (self.avg_queue_wait_time / max(1, self.avg_batch_processing_time)) * 100)
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        base_stats = {
            "total_files_processed": self.total_files_processed,
            "total_batches_processed": self.total_batches_processed,
            "total_files_failed": self.total_files_failed,
            "total_batches_failed": self.total_batches_failed,
            "avg_batch_size": self.avg_batch_size,
            "avg_batch_processing_time": self.avg_batch_processing_time,
            "avg_queue_wait_time": self.avg_queue_wait_time,
            "throughput_files_per_second": self.throughput_files_per_second,
            "throughput_batches_per_second": self.throughput_batches_per_second,
            "current_queue_size": self.current_queue_size,
            "current_memory_usage_mb": self.current_memory_usage_mb,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,
            "duplicate_files_merged": self.duplicate_files_merged,
            "priority_promotions": self.priority_promotions,
        }
        base_stats.update(self.get_efficiency_metrics())
        return base_stats

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_files_processed = 0
        self.total_batches_processed = 0
        self.total_files_failed = 0
        self.total_batches_failed = 0
        self.avg_batch_size = 0.0
        self.avg_batch_processing_time = 0.0
        self.avg_queue_wait_time = 0.0
        self.throughput_files_per_second = 0.0
        self.throughput_batches_per_second = 0.0
        self.current_queue_size = 0
        self.current_memory_usage_mb = 0.0
        self.peak_memory_usage_mb = 0.0
        self.duplicate_files_merged = 0
        self.priority_promotions = 0
        self.recent_batch_sizes.clear()
        self.recent_processing_times.clear()
        self.recent_wait_times.clear()


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing manager."""

    # Batch sizing
    max_batch_size: int = 50
    min_batch_size: int = 1
    adaptive_batch_sizing: bool = True

    # Timing
    max_batch_age_seconds: float = 5.0
    min_batch_age_seconds: float = 0.1
    batch_processing_timeout_seconds: float = 60.0

    # Queue management
    max_queue_size: int = 10000
    max_memory_usage_mb: float = 512.0
    queue_backpressure_threshold: float = 0.8

    # Processing strategy
    processing_strategy: ProcessingStrategy = ProcessingStrategy.ADAPTIVE
    enable_deduplication: bool = True
    enable_priority_processing: bool = True

    # Performance tuning
    max_concurrent_batches: int = 3
    retry_max_attempts: int = 3
    retry_delay_seconds: float = 1.0

    # Monitoring
    enable_performance_monitoring: bool = True
    statistics_window_size: int = 1000

    # Graceful degradation
    enable_graceful_degradation: bool = True
    high_load_threshold: float = 0.8
    critical_load_threshold: float = 0.95

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.min_batch_size <= 0 or self.min_batch_size > self.max_batch_size:
            raise ValueError("min_batch_size must be positive and <= max_batch_size")
        if self.max_batch_age_seconds <= 0:
            raise ValueError("max_batch_age_seconds must be positive")
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.max_memory_usage_mb <= 0:
            raise ValueError("max_memory_usage_mb must be positive")


class BatchProcessingManager:
    """
    High-performance batch processing manager for file operations.

    Manages intelligent batching of file operations with adaptive strategies,
    priority handling, and performance optimization for high-throughput scenarios.
    """

    def __init__(self, config: Union[BatchProcessingConfig, Dict[str, Any]]):
        """Initialize batch processing manager.

        Args:
            config: Configuration as BatchProcessingConfig object or dictionary
        """
        if isinstance(config, dict):
            self.config = BatchProcessingConfig(**config)
        else:
            self.config = config

        self.config.validate()

        self.statistics = BatchProcessingStatistics()
        self.processing_callback: Optional[Callable[[List[BatchItem]], Any]] = None

        # Queue management
        self._processing_queue: List[ProcessingBatch] = []  # Priority queue (heapq)
        self._current_batches: Dict[str, ProcessingBatch] = {}  # Collection -> current batch
        self._pending_items: Dict[str, Set[str]] = defaultdict(set)  # Track pending file paths per collection

        # Processing control
        self._running = False
        self._processing_task: Optional[asyncio.Task] = None
        self._batch_flush_task: Optional[asyncio.Task] = None
        self._concurrent_processors: Set[asyncio.Task] = set()

        # Performance monitoring
        self.performance_monitor = None
        if MONITORING_AVAILABLE and self.config.enable_performance_monitoring:
            try:
                self.performance_monitor = PerformanceMonitor(
                    "batch_processing_manager",
                    alert_thresholds={
                        "queue_size": self.config.max_queue_size * 0.8,
                        "memory_usage_mb": self.config.max_memory_usage_mb * 0.8,
                        "batch_processing_time": self.config.batch_processing_timeout_seconds * 0.5
                    }
                )
            except Exception as e:
                logger.debug(f"Performance monitoring not available: {e}")

        # Graceful degradation
        self.degradation_manager = None
        if DEGRADATION_AVAILABLE and self.config.enable_graceful_degradation:
            try:
                self.degradation_manager = GracefulDegradationManager(
                    high_load_threshold=self.config.high_load_threshold,
                    critical_load_threshold=self.config.critical_load_threshold
                )
            except Exception as e:
                logger.debug(f"Graceful degradation not available: {e}")

        # Adaptive parameters
        self._adaptive_batch_size = self.config.max_batch_size // 2
        self._system_load_history: deque = deque(maxlen=10)
        self._last_performance_adjustment = time.time()

        logger.info(f"BatchProcessingManager initialized with strategy: {self.config.processing_strategy.value}")

    def set_processing_callback(self, callback: Callable[[List[BatchItem]], Any]) -> None:
        """Set the callback function for processing batches.

        Args:
            callback: Async function that takes a list of BatchItem objects
        """
        self.processing_callback = callback

    async def start(self) -> None:
        """Start the batch processing manager."""
        if self._running:
            logger.warning("BatchProcessingManager is already running")
            return

        if not self.processing_callback:
            raise ValueError("Processing callback must be set before starting")

        self._running = True

        # Start processing tasks
        self._processing_task = asyncio.create_task(self._processing_loop())
        self._batch_flush_task = asyncio.create_task(self._batch_flush_loop())

        logger.info("BatchProcessingManager started")

    async def stop(self) -> None:
        """Stop the batch processing manager and process remaining batches."""
        self._running = False

        # Cancel tasks
        for task in [self._processing_task, self._batch_flush_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Wait for concurrent processors to finish
        if self._concurrent_processors:
            await asyncio.gather(*self._concurrent_processors, return_exceptions=True)

        # Process remaining batches
        await self._flush_all_batches()

        logger.info("BatchProcessingManager stopped")

    async def add_file(
        self,
        file_path: str,
        collection: str,
        operation_type: str = "add",
        priority: BatchPriority = BatchPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Add a file to the processing queue.

        Args:
            file_path: Path to the file
            collection: Collection name
            operation_type: Type of operation ("add", "modify", "delete")
            priority: Processing priority
            metadata: Optional metadata

        Returns:
            True if added successfully, False if rejected due to capacity
        """
        if not self._running:
            logger.warning("Cannot add file: BatchProcessingManager not running")
            return False

        # Check queue capacity
        if await self._is_queue_at_capacity():
            if priority == BatchPriority.CRITICAL:
                await self._make_room_for_critical_item()
            else:
                logger.warning(f"Queue at capacity, rejecting file: {file_path}")
                return False

        # Create file operation
        file_operation = FileOperation(
            file_path=file_path,
            collection=collection,
            operation_type=operation_type,
            priority=priority,
            timestamp=time.time(),
            metadata=metadata
        )

        # Create batch item
        batch_item = BatchItem(file_operation=file_operation)

        # Check for duplicates if deduplication is enabled
        if self.config.enable_deduplication:
            if await self._is_duplicate_pending(file_path, collection, operation_type):
                self.statistics.duplicate_files_merged += 1
                logger.debug(f"Merged duplicate file operation: {file_path}")
                return True

        # Add to appropriate batch
        await self._add_to_batch(batch_item)

        # Update pending tracking
        self._pending_items[collection].add(file_path)

        # Update statistics
        self.statistics.current_queue_size += 1

        logger.debug(f"Added file to processing queue: {file_path} (priority: {priority.name})")
        return True

    async def _add_to_batch(self, batch_item: BatchItem) -> None:
        """Add item to appropriate batch based on collection and strategy."""
        collection = batch_item.file_operation.collection

        # Get or create current batch for collection
        if collection not in self._current_batches:
            self._current_batches[collection] = ProcessingBatch(
                collection=collection,
                priority=batch_item.file_operation.priority
            )

        current_batch = self._current_batches[collection]
        current_batch.add_item(batch_item)

        # Check if batch should be flushed
        should_flush = await self._should_flush_batch(current_batch)
        if should_flush:
            await self._flush_batch(current_batch)

    async def _should_flush_batch(self, batch: ProcessingBatch) -> bool:
        """Determine if a batch should be flushed based on strategy."""
        strategy = self.config.processing_strategy

        if strategy == ProcessingStrategy.COUNT_BASED:
            return batch.size() >= await self._get_target_batch_size()

        elif strategy == ProcessingStrategy.TIME_BASED:
            return batch.age_seconds() >= self.config.max_batch_age_seconds

        elif strategy == ProcessingStrategy.HYBRID:
            target_size = await self._get_target_batch_size()
            return (
                batch.size() >= target_size or
                batch.age_seconds() >= self.config.max_batch_age_seconds
            )

        elif strategy == ProcessingStrategy.ADAPTIVE:
            return await self._adaptive_should_flush(batch)

        elif strategy == ProcessingStrategy.PRIORITY:
            return await self._priority_should_flush(batch)

        return batch.size() >= self.config.max_batch_size

    async def _adaptive_should_flush(self, batch: ProcessingBatch) -> bool:
        """Adaptive flushing based on system load and performance."""
        current_load = await self._get_system_load()
        target_size = await self._get_target_batch_size()

        # Under low load, use smaller batches for lower latency
        if current_load < 0.3:
            return batch.size() >= max(1, target_size // 2) or batch.age_seconds() >= 1.0

        # Under high load, use larger batches for better throughput
        elif current_load > 0.7:
            return batch.size() >= target_size or batch.age_seconds() >= self.config.max_batch_age_seconds * 2

        # Normal load - standard hybrid approach
        else:
            return (
                batch.size() >= target_size or
                batch.age_seconds() >= self.config.max_batch_age_seconds
            )

    async def _priority_should_flush(self, batch: ProcessingBatch) -> bool:
        """Priority-based flushing with immediate processing for high priority."""
        if batch.priority == BatchPriority.CRITICAL:
            return batch.size() >= 1  # Flush immediately

        elif batch.priority == BatchPriority.HIGH:
            return batch.size() >= max(1, self.config.max_batch_size // 4) or batch.age_seconds() >= 1.0

        else:
            # Normal/low priority - use standard rules
            target_size = await self._get_target_batch_size()
            return batch.size() >= target_size or batch.age_seconds() >= self.config.max_batch_age_seconds

    async def _flush_batch(self, batch: ProcessingBatch) -> None:
        """Flush a batch to the processing queue."""
        if batch.is_empty():
            return

        # Remove from current batches
        if batch.collection in self._current_batches:
            del self._current_batches[batch.collection]

        # Add to processing queue with priority
        heapq.heappush(self._processing_queue, (-batch.priority.value, batch.created_at, batch))

        logger.debug(f"Flushed batch {batch.batch_id} with {batch.size()} items to processing queue")

    async def _processing_loop(self) -> None:
        """Main processing loop for handling batches."""
        logger.debug("Batch processing loop started")

        while self._running or self._processing_queue:
            try:
                if not self._processing_queue:
                    await asyncio.sleep(0.1)
                    continue

                # Check if we can start a new processor
                if len(self._concurrent_processors) >= self.config.max_concurrent_batches:
                    await asyncio.sleep(0.1)
                    continue

                # Get next batch from priority queue
                try:
                    priority, created_at, batch = heapq.heappop(self._processing_queue)
                except IndexError:
                    continue

                # Start concurrent processor for batch
                processor_task = asyncio.create_task(self._process_batch(batch))
                self._concurrent_processors.add(processor_task)

                # Clean up completed processors
                completed_processors = {task for task in self._concurrent_processors if task.done()}
                for task in completed_processors:
                    self._concurrent_processors.remove(task)
                    try:
                        await task  # Retrieve any exceptions
                    except Exception as e:
                        logger.error(f"Error in batch processor: {e}")

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)

        logger.debug("Batch processing loop ended")

    async def _batch_flush_loop(self) -> None:
        """Loop for periodic batch flushing based on age."""
        logger.debug("Batch flush loop started")

        while self._running:
            try:
                current_time = time.time()

                # Check all current batches for age-based flushing
                batches_to_flush = []
                for collection, batch in self._current_batches.items():
                    if (
                        batch.age_seconds() >= self.config.max_batch_age_seconds or
                        (batch.priority == BatchPriority.CRITICAL and batch.age_seconds() >= 0.1)
                    ):
                        batches_to_flush.append(batch)

                # Flush aged batches
                for batch in batches_to_flush:
                    await self._flush_batch(batch)

                # Update system performance metrics
                await self._update_performance_metrics()

                # Sleep until next check
                await asyncio.sleep(min(1.0, self.config.max_batch_age_seconds / 4))

            except Exception as e:
                logger.error(f"Error in batch flush loop: {e}")
                await asyncio.sleep(1)

        logger.debug("Batch flush loop ended")

    async def _process_batch(self, batch: ProcessingBatch) -> None:
        """Process a single batch of items."""
        batch_start_time = time.time()
        wait_time = batch_start_time - batch.created_at

        logger.debug(f"Processing batch {batch.batch_id} with {batch.size()} items")

        try:
            # Update memory usage before processing
            memory_usage = batch.estimate_memory_usage()
            self.statistics.update_memory_usage(memory_usage)

            # Call processing callback
            if asyncio.iscoroutinefunction(self.processing_callback):
                await asyncio.wait_for(
                    self.processing_callback(batch.items),
                    timeout=self.config.batch_processing_timeout_seconds
                )
            else:
                # Run sync callback in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.processing_callback, batch.items)

            # Update statistics on success
            processing_time = time.time() - batch_start_time
            self.statistics.add_batch_processed(batch.size(), processing_time, wait_time)

            # Remove items from pending tracking
            for item in batch.items:
                collection = item.file_operation.collection
                file_path = item.file_operation.file_path
                self._pending_items[collection].discard(file_path)
                self.statistics.current_queue_size -= 1

            # Performance monitoring
            if self.performance_monitor:
                self.performance_monitor.record_metric("batch_size", batch.size())
                self.performance_monitor.record_metric("batch_processing_time", processing_time)
                self.performance_monitor.record_metric("queue_wait_time", wait_time)

            logger.debug(f"Successfully processed batch {batch.batch_id} in {processing_time:.2f}s")

        except asyncio.TimeoutError:
            logger.error(f"Batch {batch.batch_id} processing timed out after {self.config.batch_processing_timeout_seconds}s")
            self.statistics.add_batch_failed(batch.size())
            await self._handle_batch_failure(batch, "timeout")

        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            self.statistics.add_batch_failed(batch.size())
            await self._handle_batch_failure(batch, str(e))

    async def _handle_batch_failure(self, batch: ProcessingBatch, error_message: str) -> None:
        """Handle batch processing failure with retry logic."""
        # Update item error information
        for item in batch.items:
            item.retry_count += 1
            item.last_error = error_message

        # Determine if retry should be attempted
        retryable_items = [
            item for item in batch.items
            if item.retry_count < self.config.retry_max_attempts
        ]

        if retryable_items and self._running:
            # Create new batch for retry
            retry_batch = ProcessingBatch(
                items=retryable_items,
                collection=batch.collection,
                priority=batch.priority
            )

            # Add delay before retry
            await asyncio.sleep(self.config.retry_delay_seconds * batch.items[0].retry_count)

            # Re-queue for processing
            heapq.heappush(self._processing_queue, (-retry_batch.priority.value, retry_batch.created_at, retry_batch))

            logger.info(f"Retrying batch {batch.batch_id} with {len(retryable_items)} items (attempt {batch.items[0].retry_count + 1})")

        else:
            # Remove failed items from pending tracking
            for item in batch.items:
                collection = item.file_operation.collection
                file_path = item.file_operation.file_path
                self._pending_items[collection].discard(file_path)
                self.statistics.current_queue_size -= 1

            logger.error(f"Giving up on batch {batch.batch_id} after {self.config.retry_max_attempts} attempts")

    async def _flush_all_batches(self) -> None:
        """Flush all remaining batches during shutdown."""
        logger.info("Flushing all remaining batches...")

        # Flush current batches
        for batch in list(self._current_batches.values()):
            if not batch.is_empty():
                await self._flush_batch(batch)

        # Process all remaining batches
        while self._processing_queue:
            try:
                _, _, batch = heapq.heappop(self._processing_queue)
                await self._process_batch(batch)
            except Exception as e:
                logger.error(f"Error flushing batch during shutdown: {e}")

        logger.info("All batches flushed")

    async def _is_queue_at_capacity(self) -> bool:
        """Check if queue is at capacity considering memory and size limits."""
        # Check size limit
        if self.statistics.current_queue_size >= self.config.max_queue_size:
            return True

        # Check memory limit
        current_memory = await self._estimate_total_memory_usage()
        if current_memory >= self.config.max_memory_usage_mb:
            return True

        # Check backpressure threshold
        size_pressure = self.statistics.current_queue_size / self.config.max_queue_size
        memory_pressure = current_memory / self.config.max_memory_usage_mb
        max_pressure = max(size_pressure, memory_pressure)

        return max_pressure >= self.config.queue_backpressure_threshold

    async def _make_room_for_critical_item(self) -> None:
        """Make room for critical priority item by removing low priority items."""
        # Find low priority items to remove
        items_to_remove = []

        for collection, batch in self._current_batches.items():
            low_priority_items = [
                item for item in batch.items
                if item.file_operation.priority == BatchPriority.LOW
            ]
            items_to_remove.extend(low_priority_items)

        # Remove low priority items
        for item in items_to_remove[:10]:  # Remove up to 10 items
            collection = item.file_operation.collection
            file_path = item.file_operation.file_path

            if collection in self._current_batches:
                batch = self._current_batches[collection]
                if item in batch.items:
                    batch.items.remove(item)

            self._pending_items[collection].discard(file_path)
            self.statistics.current_queue_size -= 1

        if items_to_remove:
            logger.info(f"Removed {len(items_to_remove)} low priority items to make room for critical item")

    async def _is_duplicate_pending(self, file_path: str, collection: str, operation_type: str) -> bool:
        """Check if a file operation is already pending."""
        return file_path in self._pending_items[collection]

    async def _get_target_batch_size(self) -> int:
        """Get target batch size based on current configuration and load."""
        if not self.config.adaptive_batch_sizing:
            return self.config.max_batch_size

        # Adjust based on system load
        system_load = await self._get_system_load()

        if system_load < 0.3:
            # Low load - smaller batches for lower latency
            target = max(self.config.min_batch_size, self._adaptive_batch_size // 2)
        elif system_load > 0.7:
            # High load - larger batches for better throughput
            target = min(self.config.max_batch_size, int(self._adaptive_batch_size * 1.5))
        else:
            # Normal load
            target = self._adaptive_batch_size

        return max(self.config.min_batch_size, min(self.config.max_batch_size, target))

    async def _get_system_load(self) -> float:
        """Get current system load (0.0 to 1.0)."""
        try:
            # Combine CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None) / 100.0
            memory_percent = psutil.virtual_memory().percent / 100.0
            load = max(cpu_percent, memory_percent)

            self._system_load_history.append(load)
            return sum(self._system_load_history) / len(self._system_load_history)

        except Exception:
            return 0.5  # Default moderate load

    async def _estimate_total_memory_usage(self) -> float:
        """Estimate total memory usage of current queue in MB."""
        total_memory = 0.0

        # Memory from current batches
        for batch in self._current_batches.values():
            total_memory += batch.estimate_memory_usage()

        # Memory from processing queue
        for _, _, batch in self._processing_queue:
            total_memory += batch.estimate_memory_usage()

        return total_memory

    async def _update_performance_metrics(self) -> None:
        """Update performance metrics and adaptive parameters."""
        current_time = time.time()

        # Only adjust every few seconds to avoid thrashing
        if current_time - self._last_performance_adjustment < 5.0:
            return

        self._last_performance_adjustment = current_time

        # Adjust adaptive batch size based on recent performance
        if (
            self.statistics.recent_processing_times and
            self.statistics.recent_batch_sizes and
            self.config.adaptive_batch_sizing
        ):
            avg_time = self.statistics.avg_batch_processing_time
            avg_size = self.statistics.avg_batch_size

            # If processing is fast, we can use larger batches
            if avg_time < 1.0 and avg_size < self.config.max_batch_size:
                self._adaptive_batch_size = min(
                    self.config.max_batch_size,
                    int(self._adaptive_batch_size * 1.1)
                )

            # If processing is slow, use smaller batches
            elif avg_time > 5.0 and avg_size > self.config.min_batch_size:
                self._adaptive_batch_size = max(
                    self.config.min_batch_size,
                    int(self._adaptive_batch_size * 0.9)
                )

        # Update memory usage
        memory_usage = await self._estimate_total_memory_usage()
        self.statistics.update_memory_usage(memory_usage)

        # Performance monitoring alerts
        if self.performance_monitor:
            self.performance_monitor.record_metric("queue_size", self.statistics.current_queue_size)
            self.performance_monitor.record_metric("memory_usage_mb", memory_usage)
            self.performance_monitor.record_metric("adaptive_batch_size", self._adaptive_batch_size)

    def get_statistics(self) -> BatchProcessingStatistics:
        """Get current batch processing statistics."""
        return self.statistics

    def get_queue_status(self) -> Dict[str, Any]:
        """Get detailed queue status information."""
        return {
            "running": self._running,
            "current_queue_size": self.statistics.current_queue_size,
            "processing_queue_batches": len(self._processing_queue),
            "current_batches": len(self._current_batches),
            "concurrent_processors": len(self._concurrent_processors),
            "adaptive_batch_size": self._adaptive_batch_size,
            "collections": list(self._current_batches.keys()),
            "memory_usage_mb": self.statistics.current_memory_usage_mb,
            "system_load": self._system_load_history[-1] if self._system_load_history else 0.0
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        stats_dict = self.statistics.to_dict()
        queue_status = self.get_queue_status()

        return {
            "statistics": stats_dict,
            "queue_status": queue_status,
            "configuration": {
                "processing_strategy": self.config.processing_strategy.value,
                "max_batch_size": self.config.max_batch_size,
                "max_queue_size": self.config.max_queue_size,
                "max_concurrent_batches": self.config.max_concurrent_batches,
                "adaptive_batch_sizing": self.config.adaptive_batch_sizing,
                "enable_deduplication": self.config.enable_deduplication,
                "enable_priority_processing": self.config.enable_priority_processing
            },
            "performance_insights": {
                "queue_utilization_percent": (self.statistics.current_queue_size / self.config.max_queue_size) * 100,
                "memory_utilization_percent": (self.statistics.current_memory_usage_mb / self.config.max_memory_usage_mb) * 100,
                "batch_size_efficiency": self.statistics.avg_batch_size / self.config.max_batch_size,
                "processing_efficiency": self.statistics.throughput_files_per_second
            }
        }

    async def close(self) -> None:
        """Clean up resources and stop processing."""
        await self.stop()

        if self.performance_monitor:
            await self.performance_monitor.cleanup()

        logger.info("BatchProcessingManager closed")


# Convenience functions for common batch processing scenarios

def create_high_throughput_batch_processor() -> BatchProcessingManager:
    """Create a batch processor optimized for high-throughput scenarios."""
    config = BatchProcessingConfig(
        max_batch_size=200,
        max_batch_age_seconds=10.0,
        max_queue_size=50000,
        processing_strategy=ProcessingStrategy.ADAPTIVE,
        max_concurrent_batches=5,
        adaptive_batch_sizing=True,
        enable_deduplication=True,
        max_memory_usage_mb=1024.0
    )
    return BatchProcessingManager(config)


def create_low_latency_batch_processor() -> BatchProcessingManager:
    """Create a batch processor optimized for low latency."""
    config = BatchProcessingConfig(
        max_batch_size=10,
        max_batch_age_seconds=1.0,
        processing_strategy=ProcessingStrategy.PRIORITY,
        max_concurrent_batches=10,
        enable_priority_processing=True,
        adaptive_batch_sizing=True
    )
    return BatchProcessingManager(config)


def create_memory_efficient_batch_processor() -> BatchProcessingManager:
    """Create a batch processor optimized for memory efficiency."""
    config = BatchProcessingConfig(
        max_batch_size=25,
        max_queue_size=5000,
        max_memory_usage_mb=256.0,
        processing_strategy=ProcessingStrategy.HYBRID,
        queue_backpressure_threshold=0.6,
        enable_graceful_degradation=True
    )
    return BatchProcessingManager(config)