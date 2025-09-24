"""
Enhanced File Watcher with High-Performance Auto-Ingestion

This module provides an enhanced file watching system that combines advanced filtering,
intelligent batch processing, and performance optimization for high-throughput scenarios.

Features:
    - High-performance file system monitoring with intelligent event handling
    - Advanced file filtering with regex, content-based, and MIME type filtering
    - Intelligent batch processing with adaptive strategies
    - Smart debouncing algorithms (time-based, statistical, adaptive)
    - Performance monitoring and resource management
    - Graceful degradation under high load
    - Comprehensive error handling and recovery
    - Integration with existing performance monitoring systems

Example:
    ```python
    from workspace_qdrant_mcp.core.enhanced_file_watcher import EnhancedFileWatcher

    # Create watcher with advanced configuration
    config = {
        "watch_directories": {
            "/path/to/docs": "documents-collection",
            "/path/to/code": "code-collection"
        },
        "filtering": {
            "include_patterns": ["*.py", "*.js", "*.md", "*.txt"],
            "exclude_patterns": [r"__pycache__.*", r"node_modules.*"],
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "content_filters": ["import", "function", "class"]
        },
        "batch_processing": {
            "max_batch_size": 50,
            "max_batch_age_seconds": 5.0,
            "processing_strategy": "adaptive"
        },
        "debouncing": {
            "strategy": "adaptive",
            "base_delay_seconds": 1.0,
            "max_delay_seconds": 10.0
        }
    }

    # Set ingestion callback
    async def process_files(file_operations):
        for operation in file_operations:
            await ingest_document(operation.file_path, operation.collection)

    watcher = EnhancedFileWatcher(config)
    watcher.set_ingestion_callback(process_files)
    await watcher.start()
    ```
"""

import asyncio
import gc
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
import weakref
from contextlib import asynccontextmanager

from loguru import logger
from watchfiles import Change, awatch

# Import our advanced components
from .advanced_file_filters import AdvancedFileFilter, FilterConfig
from .batch_processing_manager import (
    BatchProcessingManager,
    BatchProcessingConfig,
    BatchItem,
    FileOperation,
    ProcessingStrategy,
    BatchPriority
)

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


class DebouncingStrategy:
    """Enumeration of debouncing strategies."""
    SIMPLE = "simple"           # Fixed delay
    ADAPTIVE = "adaptive"       # Adapts based on file change frequency
    STATISTICAL = "statistical" # Based on change pattern analysis
    NONE = "none"              # No debouncing


@dataclass
class DebouncingConfig:
    """Configuration for debouncing behavior."""
    strategy: str = DebouncingStrategy.ADAPTIVE
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 10.0
    min_delay_seconds: float = 0.1

    # Adaptive parameters
    frequency_window_seconds: float = 60.0
    high_frequency_threshold: int = 10  # Changes per window to trigger adaptive behavior

    # Statistical parameters
    pattern_analysis_window: int = 100  # Number of changes to analyze
    variance_threshold: float = 0.5     # Threshold for pattern detection


@dataclass
class WatcherStatistics:
    """Comprehensive statistics for enhanced file watcher."""

    # File system events
    total_events_received: int = 0
    files_added: int = 0
    files_modified: int = 0
    files_deleted: int = 0
    files_filtered_out: int = 0

    # Processing statistics
    files_processed_successfully: int = 0
    files_processed_with_errors: int = 0
    batches_processed: int = 0

    # Performance metrics
    avg_event_processing_time_ms: float = 0.0
    avg_debounce_delay_ms: float = 0.0
    events_per_second: float = 0.0
    processing_throughput_files_per_sec: float = 0.0

    # Resource usage
    current_debounce_tasks: int = 0
    peak_debounce_tasks: int = 0
    current_memory_usage_mb: float = 0.0
    peak_memory_usage_mb: float = 0.0

    # Error tracking
    permission_errors: int = 0
    file_not_found_errors: int = 0
    processing_errors: int = 0
    filter_errors: int = 0

    # Performance tracking
    recent_processing_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    def add_processing_time(self, time_ms: float) -> None:
        """Add a processing time measurement."""
        self.recent_processing_times.append(time_ms)
        if self.recent_processing_times:
            self.avg_event_processing_time_ms = sum(self.recent_processing_times) / len(self.recent_processing_times)

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            # File system events
            "total_events_received": self.total_events_received,
            "files_added": self.files_added,
            "files_modified": self.files_modified,
            "files_deleted": self.files_deleted,
            "files_filtered_out": self.files_filtered_out,

            # Processing statistics
            "files_processed_successfully": self.files_processed_successfully,
            "files_processed_with_errors": self.files_processed_with_errors,
            "batches_processed": self.batches_processed,

            # Performance metrics
            "avg_event_processing_time_ms": self.avg_event_processing_time_ms,
            "avg_debounce_delay_ms": self.avg_debounce_delay_ms,
            "events_per_second": self.events_per_second,
            "processing_throughput_files_per_sec": self.processing_throughput_files_per_sec,

            # Resource usage
            "current_debounce_tasks": self.current_debounce_tasks,
            "peak_debounce_tasks": self.peak_debounce_tasks,
            "current_memory_usage_mb": self.current_memory_usage_mb,
            "peak_memory_usage_mb": self.peak_memory_usage_mb,

            # Error tracking
            "permission_errors": self.permission_errors,
            "file_not_found_errors": self.file_not_found_errors,
            "processing_errors": self.processing_errors,
            "filter_errors": self.filter_errors,

            # Derived metrics
            "success_rate_percent": (
                (self.files_processed_successfully /
                 max(1, self.files_processed_successfully + self.files_processed_with_errors)) * 100
            ),
            "filter_efficiency_percent": (
                (self.files_filtered_out / max(1, self.total_events_received)) * 100
            )
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_events_received = 0
        self.files_added = 0
        self.files_modified = 0
        self.files_deleted = 0
        self.files_filtered_out = 0
        self.files_processed_successfully = 0
        self.files_processed_with_errors = 0
        self.batches_processed = 0
        self.avg_event_processing_time_ms = 0.0
        self.avg_debounce_delay_ms = 0.0
        self.events_per_second = 0.0
        self.processing_throughput_files_per_sec = 0.0
        self.current_debounce_tasks = 0
        self.peak_debounce_tasks = 0
        self.current_memory_usage_mb = 0.0
        self.peak_memory_usage_mb = 0.0
        self.permission_errors = 0
        self.file_not_found_errors = 0
        self.processing_errors = 0
        self.filter_errors = 0
        self.recent_processing_times.clear()


@dataclass
class EnhancedWatcherConfig:
    """Configuration for enhanced file watcher."""

    # Watch directories (path -> collection mapping)
    watch_directories: Dict[str, str] = field(default_factory=dict)

    # Filtering configuration
    filtering: Dict[str, Any] = field(default_factory=dict)

    # Batch processing configuration
    batch_processing: Dict[str, Any] = field(default_factory=dict)

    # Debouncing configuration
    debouncing: Dict[str, Any] = field(default_factory=dict)

    # Performance settings
    max_concurrent_watchers: int = 10
    enable_performance_monitoring: bool = True
    enable_graceful_degradation: bool = True

    # Resource limits
    max_memory_usage_mb: float = 512.0
    max_debounce_tasks: int = 1000

    # Error handling
    max_retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    continue_on_error: bool = True

    def __post_init__(self):
        """Set default configurations for sub-components."""
        if not self.filtering:
            self.filtering = {
                "include_patterns": ["*.txt", "*.md", "*.py", "*.js", "*.json"],
                "exclude_patterns": [r"__pycache__.*", r"node_modules.*", r"\.git.*"],
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "enable_content_filtering": False,
                "enable_mime_type_detection": True
            }

        if not self.batch_processing:
            self.batch_processing = {
                "max_batch_size": 50,
                "max_batch_age_seconds": 5.0,
                "processing_strategy": ProcessingStrategy.ADAPTIVE.value,
                "max_concurrent_batches": 3,
                "enable_deduplication": True
            }

        if not self.debouncing:
            self.debouncing = {
                "strategy": DebouncingStrategy.ADAPTIVE,
                "base_delay_seconds": 1.0,
                "max_delay_seconds": 10.0
            }


class SmartDebouncer:
    """Smart debouncing system with multiple strategies."""

    def __init__(self, config: DebouncingConfig):
        self.config = config
        self.file_change_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=config.pattern_analysis_window))
        self.file_delays: Dict[str, float] = {}
        self.pending_tasks: Dict[str, asyncio.Task] = {}

    async def debounce_file_change(
        self,
        file_path: str,
        callback: Callable[[], Any],
        change_type: str = "modify"
    ) -> None:
        """Apply debouncing to a file change event."""

        # Cancel any existing task for this file
        if file_path in self.pending_tasks:
            self.pending_tasks[file_path].cancel()

        # Calculate delay based on strategy
        delay = await self._calculate_delay(file_path, change_type)

        # Create new debounce task
        self.pending_tasks[file_path] = asyncio.create_task(
            self._delayed_callback(file_path, callback, delay)
        )

    async def _calculate_delay(self, file_path: str, change_type: str) -> float:
        """Calculate debounce delay based on strategy."""
        if self.config.strategy == DebouncingStrategy.NONE:
            return 0.0

        elif self.config.strategy == DebouncingStrategy.SIMPLE:
            return self.config.base_delay_seconds

        elif self.config.strategy == DebouncingStrategy.ADAPTIVE:
            return await self._calculate_adaptive_delay(file_path)

        elif self.config.strategy == DebouncingStrategy.STATISTICAL:
            return await self._calculate_statistical_delay(file_path)

        else:
            return self.config.base_delay_seconds

    async def _calculate_adaptive_delay(self, file_path: str) -> float:
        """Calculate adaptive delay based on change frequency."""
        current_time = time.time()

        # Get recent changes for this file
        history = self.file_change_history[file_path]
        history.append(current_time)

        # Count changes in the frequency window
        window_start = current_time - self.config.frequency_window_seconds
        recent_changes = sum(1 for timestamp in history if timestamp >= window_start)

        # Adaptive delay calculation
        if recent_changes >= self.config.high_frequency_threshold:
            # High frequency - increase delay to reduce processing overhead
            delay_multiplier = min(recent_changes / self.config.high_frequency_threshold, 5.0)
            delay = self.config.base_delay_seconds * delay_multiplier
        else:
            # Low frequency - use shorter delay for responsiveness
            delay = self.config.base_delay_seconds * 0.5

        # Clamp to configured limits
        delay = max(self.config.min_delay_seconds, min(self.config.max_delay_seconds, delay))

        self.file_delays[file_path] = delay
        return delay

    async def _calculate_statistical_delay(self, file_path: str) -> float:
        """Calculate delay based on statistical analysis of change patterns."""
        history = self.file_change_history[file_path]

        if len(history) < 3:
            return self.config.base_delay_seconds

        # Calculate time intervals between changes
        intervals = []
        for i in range(1, len(history)):
            intervals.append(history[i] - history[i-1])

        if not intervals:
            return self.config.base_delay_seconds

        # Statistical analysis
        mean_interval = sum(intervals) / len(intervals)

        if len(intervals) > 1:
            variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
            std_dev = variance ** 0.5

            # If changes are regular (low variance), use shorter delay
            # If changes are irregular (high variance), use longer delay
            if std_dev < self.config.variance_threshold:
                delay = self.config.base_delay_seconds * 0.7  # Regular pattern
            else:
                delay = self.config.base_delay_seconds * 1.3  # Irregular pattern
        else:
            delay = self.config.base_delay_seconds

        # Clamp to configured limits
        delay = max(self.config.min_delay_seconds, min(self.config.max_delay_seconds, delay))

        return delay

    async def _delayed_callback(self, file_path: str, callback: Callable[[], Any], delay: float) -> None:
        """Execute callback after delay."""
        try:
            await asyncio.sleep(delay)

            # Execute callback
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()

        except asyncio.CancelledError:
            # Task was cancelled (normal for debouncing)
            pass
        except Exception as e:
            logger.error(f"Error in debounced callback for {file_path}: {e}")
        finally:
            # Clean up
            self.pending_tasks.pop(file_path, None)

    def get_statistics(self) -> Dict[str, Any]:
        """Get debouncing statistics."""
        active_tasks = len(self.pending_tasks)
        avg_delay = sum(self.file_delays.values()) / max(1, len(self.file_delays))

        return {
            "active_debounce_tasks": active_tasks,
            "files_being_debounced": len(self.file_delays),
            "average_delay_seconds": avg_delay,
            "strategy": self.config.strategy,
            "total_files_tracked": len(self.file_change_history)
        }

    async def cleanup(self) -> None:
        """Clean up pending tasks."""
        for task in self.pending_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks.values(), return_exceptions=True)

        self.pending_tasks.clear()


class EnhancedFileWatcher:
    """
    Enhanced file watcher with advanced filtering, batch processing, and performance optimization.

    Combines high-performance file system monitoring with intelligent processing strategies
    for optimal performance in high-throughput scenarios.
    """

    def __init__(self, config: Union[EnhancedWatcherConfig, Dict[str, Any]]):
        """Initialize enhanced file watcher.

        Args:
            config: Configuration as EnhancedWatcherConfig object or dictionary
        """
        if isinstance(config, dict):
            self.config = EnhancedWatcherConfig(**config)
        else:
            self.config = config

        self.statistics = WatcherStatistics()
        self.ingestion_callback: Optional[Callable[[List[BatchItem]], Any]] = None

        # Core components
        self.file_filter: Optional[AdvancedFileFilter] = None
        self.batch_processor: Optional[BatchProcessingManager] = None
        self.debouncer: Optional[SmartDebouncer] = None

        # Watcher management
        self._running = False
        self._watch_tasks: Dict[str, asyncio.Task] = {}
        self._processing_semaphore = asyncio.Semaphore(self.config.max_concurrent_watchers)

        # Performance monitoring
        self.performance_monitor = None
        if MONITORING_AVAILABLE and self.config.enable_performance_monitoring:
            try:
                self.performance_monitor = PerformanceMonitor(
                    "enhanced_file_watcher",
                    alert_thresholds={
                        "event_processing_time_ms": 100,
                        "memory_usage_mb": self.config.max_memory_usage_mb * 0.8,
                        "debounce_tasks": self.config.max_debounce_tasks * 0.8
                    }
                )
            except Exception as e:
                logger.debug(f"Performance monitoring not available: {e}")

        # Graceful degradation
        self.degradation_manager = None
        if DEGRADATION_AVAILABLE and self.config.enable_graceful_degradation:
            try:
                self.degradation_manager = GracefulDegradationManager()
            except Exception as e:
                logger.debug(f"Graceful degradation not available: {e}")

        # Initialize components
        self._initialize_components()

        logger.info(f"EnhancedFileWatcher initialized for {len(self.config.watch_directories)} directories")

    def _initialize_components(self) -> None:
        """Initialize filter, batch processor, and debouncer components."""

        # Initialize file filter
        filter_config = FilterConfig(**self.config.filtering)
        self.file_filter = AdvancedFileFilter(filter_config)

        # Initialize batch processor
        batch_config = BatchProcessingConfig(**self.config.batch_processing)
        self.batch_processor = BatchProcessingManager(batch_config)
        self.batch_processor.set_processing_callback(self._process_file_batch)

        # Initialize debouncer
        debounce_config = DebouncingConfig(**self.config.debouncing)
        self.debouncer = SmartDebouncer(debounce_config)

    def set_ingestion_callback(self, callback: Callable[[List[BatchItem]], Any]) -> None:
        """Set the callback function for processing file batches.

        Args:
            callback: Async function that takes a list of BatchItem objects
        """
        self.ingestion_callback = callback

    async def start(self) -> None:
        """Start the enhanced file watcher."""
        if self._running:
            logger.warning("EnhancedFileWatcher is already running")
            return

        if not self.ingestion_callback:
            raise ValueError("Ingestion callback must be set before starting")

        self._running = True

        # Start batch processor
        await self.batch_processor.start()

        # Start watching directories
        for directory_path, collection in self.config.watch_directories.items():
            await self._start_directory_watcher(directory_path, collection)

        logger.info(f"EnhancedFileWatcher started, monitoring {len(self.config.watch_directories)} directories")

    async def stop(self) -> None:
        """Stop the enhanced file watcher."""
        self._running = False

        # Stop directory watchers
        for task in self._watch_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for watch tasks to complete
        if self._watch_tasks:
            await asyncio.gather(*self._watch_tasks.values(), return_exceptions=True)
        self._watch_tasks.clear()

        # Stop batch processor
        if self.batch_processor:
            await self.batch_processor.stop()

        # Cleanup debouncer
        if self.debouncer:
            await self.debouncer.cleanup()

        logger.info("EnhancedFileWatcher stopped")

    async def _start_directory_watcher(self, directory_path: str, collection: str) -> None:
        """Start watching a specific directory."""
        dir_path = Path(directory_path)

        if not dir_path.exists():
            logger.error(f"Watch directory does not exist: {directory_path}")
            return

        if not dir_path.is_dir():
            logger.error(f"Watch path is not a directory: {directory_path}")
            return

        # Create watch task
        watch_task = asyncio.create_task(
            self._directory_watch_loop(directory_path, collection)
        )
        self._watch_tasks[directory_path] = watch_task

        logger.info(f"Started watching directory: {directory_path} -> {collection}")

    async def _directory_watch_loop(self, directory_path: str, collection: str) -> None:
        """Watch loop for a specific directory."""
        logger.debug(f"Watch loop started for {directory_path}")

        try:
            async for changes in awatch(directory_path, recursive=True):
                if not self._running:
                    break

                # Process changes with semaphore to limit concurrency
                async with self._processing_semaphore:
                    await self._handle_file_changes(changes, collection)

        except Exception as e:
            logger.error(f"Error in watch loop for {directory_path}: {e}")
            if self.config.continue_on_error:
                # Try to restart the watcher after a delay
                await asyncio.sleep(self.config.retry_delay_seconds)
                if self._running:
                    logger.info(f"Restarting watcher for {directory_path}")
                    await self._start_directory_watcher(directory_path, collection)

        logger.debug(f"Watch loop ended for {directory_path}")

    async def _handle_file_changes(self, changes: Set[Tuple], collection: str) -> None:
        """Handle a set of file system changes."""
        start_time = time.perf_counter()

        for change_info in changes:
            change_type, file_path_str = change_info
            file_path = Path(file_path_str)

            self.statistics.total_events_received += 1

            try:
                # Update change type statistics
                await self._update_change_statistics(change_type)

                # Skip if not a file
                if not file_path.is_file() and change_type != Change.deleted:
                    continue

                # Apply file filtering
                if change_type != Change.deleted:  # Can't filter deleted files
                    should_process, reason = await self.file_filter.should_process_file(file_path)
                    if not should_process:
                        self.statistics.files_filtered_out += 1
                        logger.debug(f"File filtered out: {file_path} ({reason})")
                        continue

                # Create file operation
                operation_type = self._get_operation_type(change_type)
                file_operation = FileOperation(
                    file_path=str(file_path),
                    collection=collection,
                    operation_type=operation_type,
                    priority=self._determine_priority(file_path, operation_type),
                    timestamp=time.time()
                )

                # Apply debouncing
                await self.debouncer.debounce_file_change(
                    str(file_path),
                    lambda fo=file_operation: asyncio.create_task(self._queue_file_operation(fo)),
                    operation_type
                )

            except Exception as e:
                logger.error(f"Error handling file change {file_path}: {e}")
                await self._update_error_statistics(e)

        # Update processing time statistics
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        self.statistics.add_processing_time(processing_time_ms)

        # Performance monitoring
        if self.performance_monitor:
            self.performance_monitor.record_metric("event_processing_time_ms", processing_time_ms)
            self.performance_monitor.record_metric("events_processed", len(changes))

    async def _queue_file_operation(self, file_operation: FileOperation) -> None:
        """Queue a file operation for batch processing."""
        try:
            success = await self.batch_processor.add_file(
                file_operation.file_path,
                file_operation.collection,
                file_operation.operation_type,
                file_operation.priority,
                file_operation.metadata
            )

            if success:
                logger.debug(f"Queued file operation: {file_operation.file_path} ({file_operation.operation_type})")
            else:
                logger.warning(f"Failed to queue file operation: {file_operation.file_path}")

        except Exception as e:
            logger.error(f"Error queuing file operation {file_operation.file_path}: {e}")
            self.statistics.processing_errors += 1

    async def _process_file_batch(self, batch_items: List[BatchItem]) -> None:
        """Process a batch of file items (called by batch processor)."""
        if not self.ingestion_callback:
            logger.error("No ingestion callback set for processing batch")
            return

        try:
            # Call the ingestion callback
            if asyncio.iscoroutinefunction(self.ingestion_callback):
                await self.ingestion_callback(batch_items)
            else:
                # Run sync callback in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.ingestion_callback, batch_items)

            # Update statistics
            self.statistics.batches_processed += 1
            self.statistics.files_processed_successfully += len(batch_items)

            logger.debug(f"Successfully processed batch of {len(batch_items)} items")

        except Exception as e:
            logger.error(f"Error processing file batch: {e}")
            self.statistics.files_processed_with_errors += len(batch_items)
            raise  # Re-raise to trigger batch processor retry logic

    async def _update_change_statistics(self, change_type: Change) -> None:
        """Update statistics based on change type."""
        if change_type == Change.added:
            self.statistics.files_added += 1
        elif change_type == Change.modified:
            self.statistics.files_modified += 1
        elif change_type == Change.deleted:
            self.statistics.files_deleted += 1

    async def _update_error_statistics(self, error: Exception) -> None:
        """Update error statistics based on exception type."""
        if isinstance(error, PermissionError):
            self.statistics.permission_errors += 1
        elif isinstance(error, FileNotFoundError):
            self.statistics.file_not_found_errors += 1
        else:
            self.statistics.processing_errors += 1

    def _get_operation_type(self, change_type: Change) -> str:
        """Convert watchfiles change type to operation type."""
        if change_type == Change.added:
            return "add"
        elif change_type == Change.modified:
            return "modify"
        elif change_type == Change.deleted:
            return "delete"
        else:
            return "unknown"

    def _determine_priority(self, file_path: Path, operation_type: str) -> BatchPriority:
        """Determine processing priority based on file characteristics."""
        # Priority rules can be customized based on requirements

        # Critical files (configuration, etc.)
        if any(pattern in str(file_path).lower() for pattern in ['config', 'settings', '.env']):
            return BatchPriority.HIGH

        # Delete operations are typically high priority
        if operation_type == "delete":
            return BatchPriority.HIGH

        # Large files get lower priority
        try:
            if file_path.exists() and file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                return BatchPriority.LOW
        except (OSError, IOError):
            pass

        return BatchPriority.NORMAL

    def get_statistics(self) -> WatcherStatistics:
        """Get current watcher statistics."""
        # Update current resource usage
        self._update_resource_statistics()

        return self.statistics

    def _update_resource_statistics(self) -> None:
        """Update resource usage statistics."""
        # Update debounce task counts
        if self.debouncer:
            debounce_stats = self.debouncer.get_statistics()
            self.statistics.current_debounce_tasks = debounce_stats["active_debounce_tasks"]

            if self.statistics.current_debounce_tasks > self.statistics.peak_debounce_tasks:
                self.statistics.peak_debounce_tasks = self.statistics.current_debounce_tasks

        # Update memory usage (rough estimation)
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.statistics.current_memory_usage_mb = memory_mb

            if memory_mb > self.statistics.peak_memory_usage_mb:
                self.statistics.peak_memory_usage_mb = memory_mb
        except ImportError:
            pass

    def get_component_statistics(self) -> Dict[str, Any]:
        """Get statistics from all components."""
        stats = {
            "watcher": self.statistics.to_dict(),
        }

        if self.file_filter:
            stats["file_filter"] = self.file_filter.get_statistics().to_dict()

        if self.batch_processor:
            stats["batch_processor"] = self.batch_processor.get_statistics().to_dict()

        if self.debouncer:
            stats["debouncer"] = self.debouncer.get_statistics()

        return stats

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        stats = self.get_component_statistics()

        return {
            "component_statistics": stats,
            "configuration": {
                "watch_directories": len(self.config.watch_directories),
                "max_concurrent_watchers": self.config.max_concurrent_watchers,
                "filtering_enabled": bool(self.file_filter),
                "batch_processing_enabled": bool(self.batch_processor),
                "debouncing_strategy": self.config.debouncing.get("strategy", "unknown"),
                "performance_monitoring": self.config.enable_performance_monitoring,
                "graceful_degradation": self.config.enable_graceful_degradation
            },
            "performance_insights": {
                "overall_success_rate": stats["watcher"]["success_rate_percent"],
                "filter_efficiency": stats["watcher"]["filter_efficiency_percent"],
                "processing_throughput": stats["watcher"]["processing_throughput_files_per_sec"],
                "resource_utilization": {
                    "memory_usage_mb": stats["watcher"]["current_memory_usage_mb"],
                    "debounce_tasks": stats["watcher"]["current_debounce_tasks"]
                }
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current watcher status."""
        return {
            "running": self._running,
            "active_watchers": len([t for t in self._watch_tasks.values() if not t.done()]),
            "watch_directories": list(self.config.watch_directories.keys()),
            "component_status": {
                "file_filter": self.file_filter is not None,
                "batch_processor": self.batch_processor is not None and self.batch_processor._running,
                "debouncer": self.debouncer is not None,
                "performance_monitor": self.performance_monitor is not None,
                "degradation_manager": self.degradation_manager is not None
            }
        }

    async def add_watch_directory(self, directory_path: str, collection: str) -> bool:
        """Add a new directory to watch.

        Args:
            directory_path: Path to directory to watch
            collection: Collection name for files in this directory

        Returns:
            True if directory was added successfully
        """
        if directory_path in self.config.watch_directories:
            logger.warning(f"Directory already being watched: {directory_path}")
            return False

        # Validate directory
        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.error(f"Invalid directory: {directory_path}")
            return False

        # Add to configuration
        self.config.watch_directories[directory_path] = collection

        # Start watching if watcher is running
        if self._running:
            await self._start_directory_watcher(directory_path, collection)

        logger.info(f"Added watch directory: {directory_path} -> {collection}")
        return True

    async def remove_watch_directory(self, directory_path: str) -> bool:
        """Remove a directory from watching.

        Args:
            directory_path: Path to directory to stop watching

        Returns:
            True if directory was removed successfully
        """
        if directory_path not in self.config.watch_directories:
            logger.warning(f"Directory not being watched: {directory_path}")
            return False

        # Stop watch task
        if directory_path in self._watch_tasks:
            task = self._watch_tasks[directory_path]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self._watch_tasks[directory_path]

        # Remove from configuration
        del self.config.watch_directories[directory_path]

        logger.info(f"Removed watch directory: {directory_path}")
        return True

    async def pause(self) -> None:
        """Pause the watcher (stop monitoring but keep configuration)."""
        if not self._running:
            return

        # Pause directory watchers
        for task in self._watch_tasks.values():
            if not task.done():
                task.cancel()

        # Don't stop batch processor - let it finish current batches

        logger.info("EnhancedFileWatcher paused")

    async def resume(self) -> None:
        """Resume the watcher (restart monitoring)."""
        if self._running and self._watch_tasks:
            logger.warning("Watcher is already running")
            return

        self._running = True

        # Restart directory watchers
        for directory_path, collection in self.config.watch_directories.items():
            await self._start_directory_watcher(directory_path, collection)

        logger.info("EnhancedFileWatcher resumed")

    async def close(self) -> None:
        """Close the watcher and clean up all resources."""
        await self.stop()

        # Close components
        if self.file_filter:
            await self.file_filter.close()

        if self.batch_processor:
            await self.batch_processor.close()

        if self.performance_monitor:
            await self.performance_monitor.cleanup()

        logger.info("EnhancedFileWatcher closed")


# Convenience functions for common watcher configurations

def create_high_performance_watcher(
    watch_directories: Dict[str, str],
    **kwargs
) -> EnhancedFileWatcher:
    """Create a watcher optimized for high-performance scenarios."""

    config = EnhancedWatcherConfig(
        watch_directories=watch_directories,
        filtering={
            "include_patterns": ["*"],  # Accept all files
            "exclude_patterns": [r"__pycache__.*", r"node_modules.*", r"\.git.*"],
            "max_file_size": 50 * 1024 * 1024,  # 50MB
            "enable_content_filtering": False,  # Disable for speed
            "enable_mime_type_detection": False,
            **kwargs.get("filtering", {})
        },
        batch_processing={
            "max_batch_size": 100,
            "max_batch_age_seconds": 2.0,
            "processing_strategy": ProcessingStrategy.ADAPTIVE.value,
            "max_concurrent_batches": 8,
            "enable_deduplication": True,
            **kwargs.get("batch_processing", {})
        },
        debouncing={
            "strategy": DebouncingStrategy.ADAPTIVE,
            "base_delay_seconds": 0.5,
            "max_delay_seconds": 5.0,
            **kwargs.get("debouncing", {})
        },
        max_concurrent_watchers=20,
        max_memory_usage_mb=1024.0,
        **{k: v for k, v in kwargs.items() if k not in ["filtering", "batch_processing", "debouncing"]}
    )

    return EnhancedFileWatcher(config)


def create_selective_watcher(
    watch_directories: Dict[str, str],
    file_patterns: List[str],
    **kwargs
) -> EnhancedFileWatcher:
    """Create a watcher with selective file filtering."""

    config = EnhancedWatcherConfig(
        watch_directories=watch_directories,
        filtering={
            "include_patterns": file_patterns,
            "exclude_patterns": [
                r"__pycache__.*", r"node_modules.*", r"\.git.*",
                r".*\.tmp$", r".*\.temp$", r".*\.lock$"
            ],
            "max_file_size": 10 * 1024 * 1024,  # 10MB
            "enable_content_filtering": True,
            "enable_mime_type_detection": True,
            "content_filters": ["import", "function", "class", "def"],
            **kwargs.get("filtering", {})
        },
        batch_processing={
            "max_batch_size": 25,
            "max_batch_age_seconds": 3.0,
            "processing_strategy": ProcessingStrategy.HYBRID.value,
            **kwargs.get("batch_processing", {})
        },
        debouncing={
            "strategy": DebouncingStrategy.STATISTICAL,
            "base_delay_seconds": 2.0,
            **kwargs.get("debouncing", {})
        },
        **{k: v for k, v in kwargs.items() if k not in ["filtering", "batch_processing", "debouncing"]}
    )

    return EnhancedFileWatcher(config)


def create_memory_efficient_watcher(
    watch_directories: Dict[str, str],
    **kwargs
) -> EnhancedFileWatcher:
    """Create a watcher optimized for memory efficiency."""

    config = EnhancedWatcherConfig(
        watch_directories=watch_directories,
        filtering={
            "max_file_size": 5 * 1024 * 1024,  # 5MB limit
            "enable_content_filtering": False,  # Reduce memory usage
            **kwargs.get("filtering", {})
        },
        batch_processing={
            "max_batch_size": 10,
            "max_batch_age_seconds": 1.0,
            "processing_strategy": ProcessingStrategy.TIME_BASED.value,
            "max_concurrent_batches": 2,
            **kwargs.get("batch_processing", {})
        },
        debouncing={
            "strategy": DebouncingStrategy.SIMPLE,
            "base_delay_seconds": 1.0,
            **kwargs.get("debouncing", {})
        },
        max_concurrent_watchers=5,
        max_memory_usage_mb=128.0,
        max_debounce_tasks=100,
        **{k: v for k, v in kwargs.items() if k not in ["filtering", "batch_processing", "debouncing"]}
    )

    return EnhancedFileWatcher(config)