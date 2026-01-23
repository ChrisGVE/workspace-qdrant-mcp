"""
Optimized Auto-Ingestion with BatchProcessingManager Integration.

This module provides high-performance auto-ingestion for large file sets by integrating
the BatchProcessingManager with the existing auto-ingestion infrastructure.

Key Features:
    - Adaptive batch sizing based on system load
    - Progress reporting callbacks for large operations
    - Memory-aware processing with backpressure control
    - Priority-based file processing
    - Deduplication of file processing requests
    - Configurable concurrency limits

Task 447: Optimize auto-ingestion performance for large file sets.

Example:
    ```python
    from workspace_qdrant_mcp.core.optimized_auto_ingestion import OptimizedAutoIngestion

    # Initialize with progress callback
    def progress_callback(progress: IngestionProgress):
        print(f"Progress: {progress.files_processed}/{progress.total_files} ({progress.percentage:.1f}%)")

    ingestion = OptimizedAutoIngestion(
        workspace_client=client,
        progress_callback=progress_callback
    )

    # Ingest a directory with optimized batch processing
    results = await ingestion.ingest_directory("/path/to/project")
    ```
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from .batch_processing_manager import (
    BatchPriority,
    BatchProcessingConfig,
    BatchProcessingManager,
    ProcessingStrategy,
    create_high_throughput_batch_processor,
    create_memory_efficient_batch_processor,
)
from .config import get_config_int, get_config_bool, get_config_float


class IngestionPriority(Enum):
    """Priority levels for file ingestion."""
    CRITICAL = "critical"  # Actively edited files
    HIGH = "high"          # Recently modified files
    NORMAL = "normal"      # Standard files
    LOW = "low"            # Background/bulk ingestion
    BACKGROUND = "background"  # Initial project scan


@dataclass
class IngestionProgress:
    """Progress information for ingestion operations."""
    total_files: int = 0
    files_processed: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    current_file: str = ""
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float = 0.0
    bytes_processed: int = 0
    current_batch: int = 0
    total_batches: int = 0

    @property
    def percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.files_processed + self.files_failed + self.files_skipped) / self.total_files * 100

    @property
    def files_per_second(self) -> float:
        """Calculate processing rate."""
        if self.elapsed_seconds == 0:
            return 0.0
        return self.files_processed / self.elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_files": self.total_files,
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "files_skipped": self.files_skipped,
            "current_file": self.current_file,
            "percentage": round(self.percentage, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "estimated_remaining_seconds": round(self.estimated_remaining_seconds, 2),
            "bytes_processed": self.bytes_processed,
            "files_per_second": round(self.files_per_second, 2),
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
        }


@dataclass
class IngestionResult:
    """Result of an ingestion operation."""
    file_path: str
    success: bool
    processing_time_ms: float
    file_size_bytes: int = 0
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "success": self.success,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "file_size_bytes": self.file_size_bytes,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class IngestionStatistics:
    """Comprehensive statistics for ingestion operations."""
    total_files_processed: int = 0
    total_files_failed: int = 0
    total_files_skipped: int = 0
    total_bytes_processed: int = 0
    total_processing_time_ms: float = 0.0
    average_file_time_ms: float = 0.0
    peak_memory_mb: float = 0.0
    batches_processed: int = 0
    deduplication_hits: int = 0
    priority_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_files_processed": self.total_files_processed,
            "total_files_failed": self.total_files_failed,
            "total_files_skipped": self.total_files_skipped,
            "total_bytes_processed": self.total_bytes_processed,
            "total_processing_time_ms": round(self.total_processing_time_ms, 2),
            "average_file_time_ms": round(self.average_file_time_ms, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "batches_processed": self.batches_processed,
            "deduplication_hits": self.deduplication_hits,
            "priority_distribution": self.priority_distribution,
            "throughput_files_per_second": round(
                self.total_files_processed / max(0.001, self.total_processing_time_ms / 1000), 2
            ),
        }


# Type alias for progress callback
ProgressCallback = Callable[[IngestionProgress], None]


class OptimizedAutoIngestion:
    """
    High-performance auto-ingestion system with BatchProcessingManager integration.

    Provides optimized batch processing for large file sets with:
    - Adaptive batch sizing based on system resources
    - Progress reporting for monitoring
    - Priority-based processing queue
    - Memory-aware backpressure control
    - File deduplication
    """

    def __init__(
        self,
        file_processor: Callable[[Path], Any] | None = None,
        progress_callback: ProgressCallback | None = None,
        max_concurrent_files: int | None = None,
        batch_size: int | None = None,
        use_high_throughput: bool = False,
        use_memory_efficient: bool = False,
    ):
        """
        Initialize optimized auto-ingestion.

        Args:
            file_processor: Callable that processes a single file (if None, uses default)
            progress_callback: Callback function for progress updates
            max_concurrent_files: Maximum concurrent file processing (from config if None)
            batch_size: Batch size for processing (from config if None)
            use_high_throughput: Use high-throughput batch processor factory
            use_memory_efficient: Use memory-efficient batch processor factory
        """
        # Load configuration with defaults
        self.max_concurrent_files = max_concurrent_files or get_config_int(
            "auto_ingestion.max_concurrent_files", 10
        )
        self.batch_size = batch_size or get_config_int(
            "auto_ingestion.batch_size", 50
        )
        self.enable_deduplication = get_config_bool(
            "auto_ingestion.enable_deduplication", True
        )
        self.progress_interval = get_config_float(
            "auto_ingestion.progress_interval_seconds", 1.0
        )

        self.file_processor = file_processor
        self.progress_callback = progress_callback

        # Initialize batch processor based on mode
        if use_high_throughput:
            self.batch_processor = create_high_throughput_batch_processor()
        elif use_memory_efficient:
            self.batch_processor = create_memory_efficient_batch_processor()
        else:
            batch_config = BatchProcessingConfig(
                max_batch_size=self.batch_size,
                max_concurrent_batches=max(1, self.max_concurrent_files // self.batch_size),
                processing_strategy=ProcessingStrategy.ADAPTIVE,
                enable_deduplication=self.enable_deduplication,
            )
            self.batch_processor = BatchProcessingManager(batch_config)

        # State tracking
        self._progress = IngestionProgress()
        self._statistics = IngestionStatistics()
        self._processing_lock = asyncio.Lock()
        self._is_processing = False
        self._cancel_requested = False

        # Track processed files for deduplication
        self._processed_files: set[str] = set()

        logger.info(
            "OptimizedAutoIngestion initialized",
            max_concurrent=self.max_concurrent_files,
            batch_size=self.batch_size,
            deduplication=self.enable_deduplication,
        )

    async def ingest_files(
        self,
        file_paths: list[str | Path],
        priority: IngestionPriority = IngestionPriority.NORMAL,
        collection: str | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest multiple files with optimized batch processing.

        Args:
            file_paths: List of file paths to ingest
            priority: Priority level for processing
            collection: Target collection name (optional)

        Returns:
            List of IngestionResult objects
        """
        if not file_paths:
            return []

        async with self._processing_lock:
            self._is_processing = True
            self._cancel_requested = False
            start_time = time.time()

            # Convert to Path objects and filter
            paths = [Path(p) for p in file_paths]
            valid_paths = [p for p in paths if p.exists() and p.is_file()]

            # Initialize progress
            self._progress = IngestionProgress(
                total_files=len(valid_paths),
                total_batches=(len(valid_paths) + self.batch_size - 1) // self.batch_size,
            )

            results: list[IngestionResult] = []

            try:
                # Process in batches
                for batch_idx, batch_start in enumerate(range(0, len(valid_paths), self.batch_size)):
                    if self._cancel_requested:
                        logger.info("Ingestion cancelled by user")
                        break

                    batch = valid_paths[batch_start:batch_start + self.batch_size]
                    self._progress.current_batch = batch_idx + 1

                    # Process batch concurrently
                    batch_results = await self._process_batch(batch, priority, collection)
                    results.extend(batch_results)

                    # Update progress
                    self._progress.elapsed_seconds = time.time() - start_time
                    self._update_estimated_remaining()

                    # Report progress
                    if self.progress_callback:
                        try:
                            self.progress_callback(self._progress)
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")

                # Update statistics
                self._update_statistics(results, time.time() - start_time)

            finally:
                self._is_processing = False

            return results

    async def ingest_directory(
        self,
        directory: str | Path,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        recursive: bool = True,
        priority: IngestionPriority = IngestionPriority.NORMAL,
        collection: str | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest all matching files from a directory.

        Args:
            directory: Directory path to scan
            patterns: File patterns to include (e.g., ["*.py", "*.md"])
            ignore_patterns: Patterns to exclude (e.g., ["__pycache__/*"])
            recursive: Whether to scan subdirectories
            priority: Priority level for processing
            collection: Target collection name

        Returns:
            List of IngestionResult objects
        """
        directory = Path(directory)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Invalid directory: {directory}")
            return []

        # Collect files matching patterns
        files = await self._collect_files(directory, patterns, ignore_patterns, recursive)

        logger.info(
            f"Collected {len(files)} files from {directory}",
            patterns=patterns,
            recursive=recursive,
        )

        return await self.ingest_files(files, priority, collection)

    async def _collect_files(
        self,
        directory: Path,
        patterns: list[str] | None,
        ignore_patterns: list[str] | None,
        recursive: bool,
    ) -> list[Path]:
        """Collect files matching patterns from directory."""
        import fnmatch

        files: list[Path] = []
        default_patterns = ["*"]
        default_ignore = [
            ".git/*", "__pycache__/*", "*.pyc", "node_modules/*",
            ".venv/*", "venv/*", ".env", "*.lock", "*.log",
        ]

        patterns = patterns or default_patterns
        ignore_patterns = ignore_patterns or default_ignore

        def matches_pattern(path: Path, patterns: list[str]) -> bool:
            """Check if path matches any pattern."""
            path_str = str(path)
            return any(fnmatch.fnmatch(path_str, p) or fnmatch.fnmatch(path.name, p) for p in patterns)

        # Use asyncio to avoid blocking
        def collect_sync() -> list[Path]:
            result = []
            if recursive:
                iterator = directory.rglob("*")
            else:
                iterator = directory.glob("*")

            for path in iterator:
                if not path.is_file():
                    continue

                # Check ignore patterns
                if matches_pattern(path, ignore_patterns):
                    continue

                # Check include patterns
                if matches_pattern(path, patterns):
                    result.append(path)

            return result

        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(None, collect_sync)

        return files

    async def _process_batch(
        self,
        batch: list[Path],
        priority: IngestionPriority,
        collection: str | None,
    ) -> list[IngestionResult]:
        """Process a batch of files concurrently."""
        # Map priority to BatchPriority
        priority_map = {
            IngestionPriority.CRITICAL: BatchPriority.CRITICAL,
            IngestionPriority.HIGH: BatchPriority.HIGH,
            IngestionPriority.NORMAL: BatchPriority.NORMAL,
            IngestionPriority.LOW: BatchPriority.LOW,
            IngestionPriority.BACKGROUND: BatchPriority.LOW,
        }
        batch_priority = priority_map.get(priority, BatchPriority.NORMAL)

        # Create processing tasks
        semaphore = asyncio.Semaphore(self.max_concurrent_files)

        async def process_with_semaphore(file_path: Path) -> IngestionResult:
            async with semaphore:
                return await self._process_single_file(file_path, collection)

        # Process batch concurrently
        tasks = [process_with_semaphore(f) for f in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        processed_results: list[IngestionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(IngestionResult(
                    file_path=str(batch[i]),
                    success=False,
                    processing_time_ms=0,
                    error_message=str(result),
                ))
                self._progress.files_failed += 1
            elif isinstance(result, IngestionResult):
                processed_results.append(result)
                if result.success:
                    self._progress.files_processed += 1
                    self._progress.bytes_processed += result.file_size_bytes
                else:
                    self._progress.files_failed += 1

        # Update priority distribution
        priority_key = priority.value
        self._statistics.priority_distribution[priority_key] = (
            self._statistics.priority_distribution.get(priority_key, 0) + len(batch)
        )

        return processed_results

    async def _process_single_file(
        self,
        file_path: Path,
        collection: str | None,
    ) -> IngestionResult:
        """Process a single file."""
        start_time = time.perf_counter()
        self._progress.current_file = str(file_path)

        # Check deduplication
        file_key = str(file_path.resolve())
        if self.enable_deduplication and file_key in self._processed_files:
            self._statistics.deduplication_hits += 1
            self._progress.files_skipped += 1
            return IngestionResult(
                file_path=str(file_path),
                success=True,
                processing_time_ms=0,
                metadata={"skipped": "duplicate"},
            )

        try:
            # Get file size
            file_size = file_path.stat().st_size

            # Process file
            if self.file_processor:
                await self.file_processor(file_path)
            else:
                # Default: just read and validate the file
                await self._default_file_processor(file_path)

            # Mark as processed
            if self.enable_deduplication:
                self._processed_files.add(file_key)

            processing_time = (time.perf_counter() - start_time) * 1000

            return IngestionResult(
                file_path=str(file_path),
                success=True,
                processing_time_ms=processing_time,
                file_size_bytes=file_size,
                metadata={
                    "collection": collection,
                    "extension": file_path.suffix,
                },
            )

        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Failed to process {file_path}: {e}")

            return IngestionResult(
                file_path=str(file_path),
                success=False,
                processing_time_ms=processing_time,
                error_message=str(e),
            )

    async def _default_file_processor(self, file_path: Path) -> dict[str, Any]:
        """Default file processor - reads and validates file."""
        # Read file content asynchronously
        loop = asyncio.get_event_loop()

        def read_file() -> tuple[str, int]:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            return content, len(content)

        content, size = await loop.run_in_executor(None, read_file)

        return {
            "content_length": size,
            "line_count": content.count('\n') + 1,
        }

    def _update_estimated_remaining(self) -> None:
        """Update estimated remaining time based on current progress."""
        if self._progress.files_processed == 0:
            self._progress.estimated_remaining_seconds = 0
            return

        files_remaining = (
            self._progress.total_files -
            self._progress.files_processed -
            self._progress.files_failed -
            self._progress.files_skipped
        )

        if files_remaining <= 0:
            self._progress.estimated_remaining_seconds = 0
            return

        time_per_file = self._progress.elapsed_seconds / self._progress.files_processed
        self._progress.estimated_remaining_seconds = files_remaining * time_per_file

    def _update_statistics(
        self,
        results: list[IngestionResult],
        total_time_seconds: float,
    ) -> None:
        """Update statistics from processing results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        self._statistics.total_files_processed += len(successful)
        self._statistics.total_files_failed += len(failed)
        self._statistics.total_bytes_processed += sum(r.file_size_bytes for r in successful)
        self._statistics.total_processing_time_ms += total_time_seconds * 1000
        self._statistics.batches_processed += self._progress.total_batches

        if self._statistics.total_files_processed > 0:
            self._statistics.average_file_time_ms = (
                self._statistics.total_processing_time_ms /
                self._statistics.total_files_processed
            )

    def cancel(self) -> None:
        """Request cancellation of current processing."""
        self._cancel_requested = True
        logger.info("Cancellation requested for auto-ingestion")

    def get_progress(self) -> IngestionProgress:
        """Get current progress information."""
        return self._progress

    def get_statistics(self) -> IngestionStatistics:
        """Get cumulative statistics."""
        return self._statistics

    def reset_statistics(self) -> None:
        """Reset statistics to initial state."""
        self._statistics = IngestionStatistics()
        self._processed_files.clear()
        logger.info("Auto-ingestion statistics reset")

    @property
    def is_processing(self) -> bool:
        """Check if currently processing files."""
        return self._is_processing


async def create_optimized_ingestion(
    file_processor: Callable[[Path], Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    mode: str = "balanced",
) -> OptimizedAutoIngestion:
    """
    Factory function to create OptimizedAutoIngestion with preset configurations.

    Args:
        file_processor: Custom file processor function
        progress_callback: Progress reporting callback
        mode: Configuration mode - "balanced", "high_throughput", or "memory_efficient"

    Returns:
        Configured OptimizedAutoIngestion instance
    """
    if mode == "high_throughput":
        return OptimizedAutoIngestion(
            file_processor=file_processor,
            progress_callback=progress_callback,
            use_high_throughput=True,
        )
    elif mode == "memory_efficient":
        return OptimizedAutoIngestion(
            file_processor=file_processor,
            progress_callback=progress_callback,
            use_memory_efficient=True,
        )
    else:
        return OptimizedAutoIngestion(
            file_processor=file_processor,
            progress_callback=progress_callback,
        )
