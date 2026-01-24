"""
Optimized Auto-Ingestion for High-Performance File Processing

This module provides optimized auto-ingestion capabilities for handling large
file sets efficiently by integrating with BatchProcessingManager.

Features:
    - Adaptive batch sizing based on system resources
    - Progress reporting callbacks for large operations
    - Priority-based file processing queue
    - File deduplication within sessions
    - Configurable concurrency limits
    - Cancellation support for long-running operations
    - Directory ingestion with pattern matching

Example:
    ```python
    from common.core.optimized_auto_ingestion import (
        OptimizedAutoIngestion,
        OptimizedIngestionConfig,
        IngestionProgress,
        IngestionPriority
    )

    config = OptimizedIngestionConfig(
        max_concurrent_files=10,
        batch_size=50,
        enable_deduplication=True,
        progress_interval_seconds=1.0
    )

    ingestion = OptimizedAutoIngestion(config)

    # Set progress callback
    def on_progress(progress: IngestionProgress):
        print(f"Progress: {progress.percent_complete:.1f}%")

    ingestion.set_progress_callback(on_progress)

    # Ingest a directory
    await ingestion.ingest_directory(
        "/path/to/project",
        collection="my-project",
        patterns=["*.py", "*.md"],
        priority=IngestionPriority.HIGH
    )
    ```
"""

import asyncio
import fnmatch
import hashlib
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import psutil
from loguru import logger

from .batch_processing_manager import (
    BatchItem,
    BatchPriority,
    BatchProcessingConfig,
    BatchProcessingManager,
    BatchProcessingStatistics,
    FileOperation,
    ProcessingStrategy,
)


class IngestionPriority(Enum):
    """Priority levels for file ingestion."""

    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()

    def to_batch_priority(self) -> BatchPriority:
        """Convert to BatchPriority for BatchProcessingManager."""
        mapping = {
            IngestionPriority.LOW: BatchPriority.LOW,
            IngestionPriority.NORMAL: BatchPriority.NORMAL,
            IngestionPriority.HIGH: BatchPriority.HIGH,
            IngestionPriority.CRITICAL: BatchPriority.CRITICAL,
        }
        return mapping[self]


class IngestionStatus(Enum):
    """Status of ingestion operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class IngestionProgress:
    """Progress information for ingestion operations."""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    deduplicated_files: int = 0
    current_file: str = ""
    status: IngestionStatus = IngestionStatus.PENDING
    start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    throughput_files_per_second: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def percent_complete(self) -> float:
        """Calculate percentage completion."""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files + self.failed_files + self.skipped_files) / self.total_files * 100

    def update_timing(self) -> None:
        """Update timing-related fields."""
        now = time.time()
        self.elapsed_seconds = now - self.start_time
        self.last_update_time = now

        # Calculate throughput and ETA
        completed = self.processed_files + self.failed_files + self.skipped_files
        if self.elapsed_seconds > 0 and completed > 0:
            self.throughput_files_per_second = completed / self.elapsed_seconds
            remaining = self.total_files - completed
            if self.throughput_files_per_second > 0:
                self.estimated_remaining_seconds = remaining / self.throughput_files_per_second
            else:
                self.estimated_remaining_seconds = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert progress to dictionary."""
        return {
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "failed_files": self.failed_files,
            "skipped_files": self.skipped_files,
            "deduplicated_files": self.deduplicated_files,
            "current_file": self.current_file,
            "status": self.status.value,
            "percent_complete": self.percent_complete,
            "elapsed_seconds": self.elapsed_seconds,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "throughput_files_per_second": self.throughput_files_per_second,
            "error_count": len(self.errors),
        }


@dataclass
class OptimizedIngestionConfig:
    """Configuration for optimized auto-ingestion."""

    # Concurrency and batching
    max_concurrent_files: int = 10
    batch_size: int = 50
    adaptive_batch_sizing: bool = True

    # Deduplication
    enable_deduplication: bool = True
    deduplication_strategy: str = "content_hash"  # "path", "content_hash", "mtime"

    # Progress reporting
    progress_interval_seconds: float = 1.0
    enable_progress_callbacks: bool = True

    # Resource management
    max_memory_usage_mb: float = 512.0
    max_cpu_percent: float = 80.0
    adaptive_resource_scaling: bool = True

    # Processing timeouts
    file_timeout_seconds: float = 60.0
    batch_timeout_seconds: float = 300.0
    operation_timeout_seconds: float = 3600.0  # 1 hour max for entire operation

    # Retry behavior
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # File filtering
    max_file_size_mb: float = 50.0
    default_patterns: list[str] = field(
        default_factory=lambda: ["*.py", "*.js", "*.ts", "*.md", "*.txt", "*.yaml", "*.json"]
    )
    default_ignore_patterns: list[str] = field(
        default_factory=lambda: [
            "*.pyc",
            "__pycache__/*",
            ".git/*",
            "node_modules/*",
            ".venv/*",
            "venv/*",
            "*.egg-info/*",
            "dist/*",
            "build/*",
            ".tox/*",
            "*.so",
            "*.dylib",
        ]
    )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.max_concurrent_files <= 0:
            raise ValueError("max_concurrent_files must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.progress_interval_seconds < 0:
            raise ValueError("progress_interval_seconds must be non-negative")
        if self.max_memory_usage_mb <= 0:
            raise ValueError("max_memory_usage_mb must be positive")
        if self.max_cpu_percent <= 0 or self.max_cpu_percent > 100:
            raise ValueError("max_cpu_percent must be between 0 and 100")


@dataclass
class FileEntry:
    """Represents a file to be ingested."""

    path: str
    collection: str
    priority: IngestionPriority = IngestionPriority.NORMAL
    metadata: dict[str, Any] | None = None
    size_bytes: int = 0
    mtime: float = 0.0
    content_hash: str | None = None

    def compute_content_hash(self, chunk_size: int = 8192) -> str:
        """Compute content hash for deduplication."""
        if self.content_hash:
            return self.content_hash

        hasher = hashlib.sha256()
        try:
            with open(self.path, "rb") as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            self.content_hash = hasher.hexdigest()
        except (OSError, IOError) as e:
            logger.warning(f"Failed to hash file {self.path}: {e}")
            self.content_hash = hashlib.sha256(self.path.encode()).hexdigest()

        return self.content_hash


class OptimizedAutoIngestion:
    """
    High-performance auto-ingestion system for large file sets.

    Integrates with BatchProcessingManager to provide optimized
    file processing with progress tracking, deduplication, and
    adaptive resource management.
    """

    def __init__(
        self,
        config: OptimizedIngestionConfig | dict[str, Any] | None = None,
        processing_callback: Callable[[list[FileEntry]], Any] | None = None,
    ):
        """Initialize optimized auto-ingestion.

        Args:
            config: Configuration as OptimizedIngestionConfig or dictionary
            processing_callback: Optional callback for processing files
        """
        if config is None:
            self.config = OptimizedIngestionConfig()
        elif isinstance(config, dict):
            self.config = OptimizedIngestionConfig(**config)
        else:
            self.config = config

        self.config.validate()

        # Initialize batch processing manager
        batch_config = BatchProcessingConfig(
            max_batch_size=self.config.batch_size,
            adaptive_batch_sizing=self.config.adaptive_batch_sizing,
            max_memory_usage_mb=self.config.max_memory_usage_mb,
            enable_deduplication=self.config.enable_deduplication,
            enable_priority_processing=True,
            processing_strategy=ProcessingStrategy.ADAPTIVE,
            max_concurrent_batches=max(1, self.config.max_concurrent_files // self.config.batch_size),
            retry_max_attempts=self.config.max_retries,
            retry_delay_seconds=self.config.retry_delay_seconds,
            batch_processing_timeout_seconds=self.config.batch_timeout_seconds,
        )
        self._batch_manager = BatchProcessingManager(batch_config)

        # Processing callback
        self._processing_callback = processing_callback
        if processing_callback:
            self._batch_manager.set_processing_callback(self._wrap_processing_callback)

        # Progress tracking
        self._progress = IngestionProgress()
        self._progress_callback: Callable[[IngestionProgress], None] | None = None
        self._progress_task: asyncio.Task | None = None

        # Deduplication tracking (session-based)
        self._session_files: dict[str, set[str]] = defaultdict(set)  # collection -> set of dedup keys
        self._session_id: str = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

        # Cancellation support
        self._cancel_requested = False
        self._pause_requested = False
        self._operation_lock = asyncio.Lock()

        # Active operations
        self._active_ingestions: dict[str, IngestionProgress] = {}

        # Resource monitoring
        self._last_resource_check = 0.0
        self._current_batch_size = self.config.batch_size

        logger.info(f"OptimizedAutoIngestion initialized (session: {self._session_id})")

    def set_processing_callback(self, callback: Callable[[list[FileEntry]], Any]) -> None:
        """Set the callback function for processing files.

        Args:
            callback: Async function that takes a list of FileEntry objects
        """
        self._processing_callback = callback
        self._batch_manager.set_processing_callback(self._wrap_processing_callback)

    def set_progress_callback(self, callback: Callable[[IngestionProgress], None]) -> None:
        """Set the callback function for progress updates.

        Args:
            callback: Function that receives IngestionProgress updates
        """
        self._progress_callback = callback

    async def _wrap_processing_callback(self, batch_items: list[BatchItem]) -> None:
        """Wrap batch items for the processing callback."""
        if not self._processing_callback:
            return

        file_entries = []
        for item in batch_items:
            entry = FileEntry(
                path=item.file_operation.file_path,
                collection=item.file_operation.collection,
                priority=self._batch_to_ingestion_priority(item.file_operation.priority),
                metadata=item.file_operation.metadata,
            )
            file_entries.append(entry)

        await self._processing_callback(file_entries)

    def _batch_to_ingestion_priority(self, batch_priority: BatchPriority) -> IngestionPriority:
        """Convert BatchPriority to IngestionPriority."""
        mapping = {
            BatchPriority.LOW: IngestionPriority.LOW,
            BatchPriority.NORMAL: IngestionPriority.NORMAL,
            BatchPriority.HIGH: IngestionPriority.HIGH,
            BatchPriority.CRITICAL: IngestionPriority.CRITICAL,
        }
        return mapping.get(batch_priority, IngestionPriority.NORMAL)

    async def start(self) -> None:
        """Start the auto-ingestion system."""
        await self._batch_manager.start()

        # Start progress reporting task if enabled
        if self.config.enable_progress_callbacks and self._progress_callback:
            self._progress_task = asyncio.create_task(self._progress_reporter())

        logger.info("OptimizedAutoIngestion started")

    async def stop(self) -> None:
        """Stop the auto-ingestion system."""
        self._cancel_requested = True

        # Stop progress reporter
        if self._progress_task and not self._progress_task.done():
            self._progress_task.cancel()
            try:
                await self._progress_task
            except asyncio.CancelledError:
                pass

        # Stop batch manager
        await self._batch_manager.stop()

        logger.info("OptimizedAutoIngestion stopped")

    async def _progress_reporter(self) -> None:
        """Background task to report progress at configured intervals."""
        while not self._cancel_requested:
            try:
                await asyncio.sleep(self.config.progress_interval_seconds)
                if self._progress_callback and self._progress.status == IngestionStatus.IN_PROGRESS:
                    self._progress.update_timing()
                    self._progress_callback(self._progress)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Progress reporter error: {e}")

    def _get_dedup_key(self, entry: FileEntry) -> str:
        """Generate deduplication key based on strategy."""
        if self.config.deduplication_strategy == "path":
            return entry.path
        elif self.config.deduplication_strategy == "content_hash":
            return entry.compute_content_hash()
        elif self.config.deduplication_strategy == "mtime":
            return f"{entry.path}:{entry.mtime}"
        else:
            return entry.path

    def _is_duplicate(self, entry: FileEntry) -> bool:
        """Check if file is a duplicate within the session."""
        if not self.config.enable_deduplication:
            return False

        dedup_key = self._get_dedup_key(entry)
        return dedup_key in self._session_files[entry.collection]

    def _mark_processed(self, entry: FileEntry) -> None:
        """Mark file as processed for deduplication."""
        if self.config.enable_deduplication:
            dedup_key = self._get_dedup_key(entry)
            self._session_files[entry.collection].add(dedup_key)

    async def _check_resources(self) -> bool:
        """Check if system resources allow processing."""
        now = time.time()
        if now - self._last_resource_check < 1.0:  # Check at most once per second
            return True

        self._last_resource_check = now

        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)

            # Check memory
            memory_used_mb = (memory.total - memory.available) / (1024 * 1024)
            if memory_used_mb > self.config.max_memory_usage_mb * 0.9:
                logger.warning(f"Memory usage high: {memory_used_mb:.1f}MB")
                if self.config.adaptive_resource_scaling:
                    self._current_batch_size = max(1, self._current_batch_size // 2)
                return False

            # Check CPU
            if cpu_percent > self.config.max_cpu_percent:
                logger.warning(f"CPU usage high: {cpu_percent:.1f}%")
                if self.config.adaptive_resource_scaling:
                    self._current_batch_size = max(1, self._current_batch_size // 2)
                return False

            # Resources OK, gradually increase batch size
            if self.config.adaptive_resource_scaling:
                if self._current_batch_size < self.config.batch_size:
                    self._current_batch_size = min(
                        self.config.batch_size, int(self._current_batch_size * 1.1)
                    )

            return True

        except Exception as e:
            logger.debug(f"Resource check failed: {e}")
            return True  # Assume OK if check fails

    def _matches_patterns(
        self, file_path: str, patterns: list[str], ignore_patterns: list[str]
    ) -> bool:
        """Check if file matches include patterns and doesn't match ignore patterns."""
        rel_path = os.path.basename(file_path)
        full_path = file_path

        # Check ignore patterns first
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(full_path, pattern):
                return False

        # Check include patterns
        if not patterns:
            return True

        for pattern in patterns:
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(full_path, pattern):
                return True

        return False

    async def ingest_directory(
        self,
        directory: str,
        collection: str,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        priority: IngestionPriority = IngestionPriority.NORMAL,
        recursive: bool = True,
        max_depth: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionProgress:
        """Ingest all matching files from a directory.

        Args:
            directory: Path to directory to ingest
            collection: Target collection name
            patterns: File patterns to include (e.g., ["*.py", "*.md"])
            ignore_patterns: File patterns to exclude
            priority: Processing priority for all files
            recursive: Whether to recurse into subdirectories
            max_depth: Maximum recursion depth
            metadata: Optional metadata for all files

        Returns:
            IngestionProgress with final statistics
        """
        if patterns is None:
            patterns = self.config.default_patterns
        if ignore_patterns is None:
            ignore_patterns = self.config.default_ignore_patterns

        async with self._operation_lock:
            self._cancel_requested = False
            self._pause_requested = False

            # Reset progress
            self._progress = IngestionProgress(status=IngestionStatus.PENDING)

            try:
                # Discover files
                files = await self._discover_files(
                    directory, patterns, ignore_patterns, recursive, max_depth
                )

                if not files:
                    self._progress.status = IngestionStatus.COMPLETED
                    return self._progress

                self._progress.total_files = len(files)
                self._progress.status = IngestionStatus.IN_PROGRESS

                logger.info(f"Starting ingestion of {len(files)} files from {directory}")

                # Process files in batches
                await self._process_files(
                    files, collection, priority, metadata
                )

                # Final status
                if self._cancel_requested:
                    self._progress.status = IngestionStatus.CANCELLED
                elif self._progress.failed_files > 0 and self._progress.processed_files == 0:
                    self._progress.status = IngestionStatus.FAILED
                else:
                    self._progress.status = IngestionStatus.COMPLETED

                self._progress.update_timing()

                logger.info(
                    f"Ingestion complete: {self._progress.processed_files} processed, "
                    f"{self._progress.failed_files} failed, "
                    f"{self._progress.skipped_files} skipped, "
                    f"{self._progress.deduplicated_files} deduplicated"
                )

                return self._progress

            except asyncio.TimeoutError:
                self._progress.status = IngestionStatus.FAILED
                self._progress.errors.append(("", "Operation timed out"))
                return self._progress
            except Exception as e:
                self._progress.status = IngestionStatus.FAILED
                self._progress.errors.append(("", str(e)))
                logger.error(f"Ingestion failed: {e}")
                return self._progress

    async def _discover_files(
        self,
        directory: str,
        patterns: list[str],
        ignore_patterns: list[str],
        recursive: bool,
        max_depth: int,
    ) -> list[str]:
        """Discover files matching patterns in directory."""
        files = []
        directory_path = Path(directory)

        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")

        def scan_directory(path: Path, current_depth: int) -> None:
            if current_depth > max_depth:
                return

            try:
                for entry in path.iterdir():
                    if self._cancel_requested:
                        return

                    if entry.is_file():
                        if self._matches_patterns(str(entry), patterns, ignore_patterns):
                            # Check file size
                            try:
                                size_mb = entry.stat().st_size / (1024 * 1024)
                                if size_mb <= self.config.max_file_size_mb:
                                    files.append(str(entry))
                            except OSError:
                                pass

                    elif entry.is_dir() and recursive:
                        # Check ignore patterns for directories
                        dir_name = entry.name
                        should_ignore = False
                        for pattern in ignore_patterns:
                            if fnmatch.fnmatch(dir_name, pattern.rstrip("/*")):
                                should_ignore = True
                                break
                            if fnmatch.fnmatch(f"{dir_name}/", pattern):
                                should_ignore = True
                                break
                        if not should_ignore:
                            scan_directory(entry, current_depth + 1)
            except PermissionError:
                logger.debug(f"Permission denied: {path}")
            except OSError as e:
                logger.debug(f"Error scanning {path}: {e}")

        # Run discovery in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, scan_directory, directory_path, 0)

        return files

    async def _process_files(
        self,
        files: list[str],
        collection: str,
        priority: IngestionPriority,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Process files through batch manager."""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_files)

        async def process_single_file(file_path: str) -> None:
            if self._cancel_requested:
                return

            # Wait while paused
            while self._pause_requested and not self._cancel_requested:
                await asyncio.sleep(0.1)

            if self._cancel_requested:
                return

            async with semaphore:
                await self._check_resources()

                self._progress.current_file = file_path

                try:
                    # Create file entry
                    stat = os.stat(file_path)
                    entry = FileEntry(
                        path=file_path,
                        collection=collection,
                        priority=priority,
                        metadata=metadata,
                        size_bytes=stat.st_size,
                        mtime=stat.st_mtime,
                    )

                    # Check for duplicates
                    if self._is_duplicate(entry):
                        self._progress.deduplicated_files += 1
                        self._progress.skipped_files += 1
                        return

                    # Add to batch manager
                    added = await self._batch_manager.add_file(
                        file_path=file_path,
                        collection=collection,
                        operation_type="add",
                        priority=priority.to_batch_priority(),
                        metadata=metadata,
                    )

                    if added:
                        self._mark_processed(entry)
                        self._progress.processed_files += 1
                    else:
                        self._progress.skipped_files += 1

                except FileNotFoundError:
                    self._progress.skipped_files += 1
                except Exception as e:
                    self._progress.failed_files += 1
                    self._progress.errors.append((file_path, str(e)))
                    logger.debug(f"Failed to process {file_path}: {e}")

        # Process files concurrently with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*[process_single_file(f) for f in files], return_exceptions=True),
                timeout=self.config.operation_timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Operation timed out, some files may not have been processed")

    async def ingest_files(
        self,
        files: list[str],
        collection: str,
        priority: IngestionPriority = IngestionPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
    ) -> IngestionProgress:
        """Ingest a specific list of files.

        Args:
            files: List of file paths to ingest
            collection: Target collection name
            priority: Processing priority
            metadata: Optional metadata for all files

        Returns:
            IngestionProgress with final statistics
        """
        async with self._operation_lock:
            self._cancel_requested = False
            self._pause_requested = False

            # Reset progress
            self._progress = IngestionProgress(
                total_files=len(files),
                status=IngestionStatus.IN_PROGRESS,
            )

            if not files:
                self._progress.status = IngestionStatus.COMPLETED
                return self._progress

            try:
                await self._process_files(files, collection, priority, metadata)

                if self._cancel_requested:
                    self._progress.status = IngestionStatus.CANCELLED
                elif self._progress.failed_files > 0 and self._progress.processed_files == 0:
                    self._progress.status = IngestionStatus.FAILED
                else:
                    self._progress.status = IngestionStatus.COMPLETED

                self._progress.update_timing()
                return self._progress

            except Exception as e:
                self._progress.status = IngestionStatus.FAILED
                self._progress.errors.append(("", str(e)))
                return self._progress

    async def cancel(self) -> None:
        """Cancel the current ingestion operation."""
        self._cancel_requested = True
        logger.info("Ingestion cancellation requested")

    async def pause(self) -> None:
        """Pause the current ingestion operation."""
        self._pause_requested = True
        self._progress.status = IngestionStatus.PAUSED
        logger.info("Ingestion paused")

    async def resume(self) -> None:
        """Resume a paused ingestion operation."""
        self._pause_requested = False
        self._progress.status = IngestionStatus.IN_PROGRESS
        logger.info("Ingestion resumed")

    def get_progress(self) -> IngestionProgress:
        """Get current progress information."""
        self._progress.update_timing()
        return self._progress

    def get_statistics(self) -> dict[str, Any]:
        """Get combined statistics from batch manager and ingestion."""
        batch_stats = self._batch_manager.statistics.to_dict()

        ingestion_stats = {
            "session_id": self._session_id,
            "current_batch_size": self._current_batch_size,
            "deduplication_enabled": self.config.enable_deduplication,
            "deduplication_strategy": self.config.deduplication_strategy,
            "progress": self._progress.to_dict(),
            "collections_tracked": list(self._session_files.keys()),
            "files_tracked_per_collection": {
                k: len(v) for k, v in self._session_files.items()
            },
        }

        return {
            "batch_processing": batch_stats,
            "ingestion": ingestion_stats,
        }

    def clear_session(self) -> None:
        """Clear session data and start fresh."""
        self._session_files.clear()
        self._session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        self._progress = IngestionProgress()
        logger.info(f"Session cleared, new session: {self._session_id}")

    @property
    def is_running(self) -> bool:
        """Check if ingestion system is running."""
        return self._batch_manager._running

    @property
    def is_paused(self) -> bool:
        """Check if ingestion is paused."""
        return self._pause_requested

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_requested
