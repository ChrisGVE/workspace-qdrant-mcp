from common.observability import get_logger

logger = get_logger(__name__)
"""
Progress reporting infrastructure for document parsing operations.

This module provides comprehensive progress tracking for large file processing
and batch operations, including memory usage monitoring and performance metrics.
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import psutil

logger = logging.getLogger(__name__)


class ProgressPhase(Enum):
    """Phases of document processing operation."""

    INITIALIZING = "initializing"
    DETECTING_TYPE = "detecting_type"
    VALIDATING = "validating"
    LOADING = "loading"
    PARSING = "parsing"
    PROCESSING = "processing"
    EXTRACTING = "extracting"
    CLEANING = "cleaning"
    ANALYZING = "analyzing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProgressUnit(Enum):
    """Units for progress measurement."""

    BYTES = "bytes"
    PAGES = "pages"
    DOCUMENTS = "documents"
    OPERATIONS = "operations"
    PERCENT = "percent"


@dataclass
class ProgressMetrics:
    """Container for progress and performance metrics."""

    # Progress tracking
    current: int = 0
    total: int = 0
    phase: ProgressPhase = ProgressPhase.INITIALIZING
    unit: ProgressUnit = ProgressUnit.BYTES

    # Performance metrics
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    elapsed_time: float = 0.0
    estimated_remaining: float = 0.0
    throughput: float = 0.0  # units per second

    # Memory metrics
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_limit_mb: Optional[float] = None

    # Error tracking
    warnings_count: int = 0
    errors_count: int = 0
    recoverable_errors: int = 0

    # Additional context
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    current_operation: Optional[str] = None

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total <= 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)

    @property
    def is_complete(self) -> bool:
        """Check if operation is complete."""
        return self.phase in (ProgressPhase.COMPLETED, ProgressPhase.FAILED)

    def update_timing(self) -> None:
        """Update timing metrics."""
        now = time.time()
        self.elapsed_time = now - self.start_time
        self.last_update = now

        # Calculate throughput
        if self.elapsed_time > 0:
            self.throughput = self.current / self.elapsed_time

        # Estimate remaining time
        if self.throughput > 0 and self.current > 0:
            remaining_work = self.total - self.current
            self.estimated_remaining = remaining_work / self.throughput

    def update_memory(self) -> None:
        """Update memory usage metrics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_usage_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            self.peak_memory_mb = max(self.peak_memory_mb, self.memory_usage_mb)
        except Exception as e:
            logger.debug(f"Failed to update memory metrics: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "current": self.current,
            "total": self.total,
            "progress_percent": self.progress_percent,
            "phase": self.phase.value,
            "unit": self.unit.value,
            "elapsed_time": self.elapsed_time,
            "estimated_remaining": self.estimated_remaining,
            "throughput": self.throughput,
            "memory_usage_mb": self.memory_usage_mb,
            "peak_memory_mb": self.peak_memory_mb,
            "warnings_count": self.warnings_count,
            "errors_count": self.errors_count,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "current_operation": self.current_operation,
            "is_complete": self.is_complete,
        }


class ProgressCallback(ABC):
    """Abstract base class for progress callbacks."""

    @abstractmethod
    def on_progress_update(self, metrics: ProgressMetrics) -> None:
        """Called when progress is updated."""
        pass

    @abstractmethod
    def on_phase_change(
        self, old_phase: ProgressPhase, new_phase: ProgressPhase
    ) -> None:
        """Called when processing phase changes."""
        pass

    @abstractmethod
    def on_error(self, error: Exception, metrics: ProgressMetrics) -> None:
        """Called when an error occurs."""
        pass


class ConsoleProgressCallback(ProgressCallback):
    """Progress callback that outputs to console."""

    def __init__(self, show_memory: bool = True, show_throughput: bool = True):
        """
        Initialize console progress callback.

        Args:
            show_memory: Whether to display memory usage
            show_throughput: Whether to display throughput information
        """
        self.show_memory = show_memory
        self.show_throughput = show_throughput

    def on_progress_update(self, metrics: ProgressMetrics) -> None:
        """Display progress update to console."""
        progress_bar = self._create_progress_bar(metrics.progress_percent)

        output_parts = [
            f"{progress_bar} {metrics.progress_percent:.1f}%",
            f"({metrics.current}/{metrics.total} {metrics.unit.value})",
        ]

        if metrics.current_operation:
            output_parts.append(f"- {metrics.current_operation}")

        if self.show_throughput and metrics.throughput > 0:
            throughput_str = f"{metrics.throughput:.1f} {metrics.unit.value}/s"
            if metrics.estimated_remaining > 0:
                eta_str = f"ETA: {self._format_time(metrics.estimated_remaining)}"
                output_parts.append(f"({throughput_str}, {eta_str})")
            else:
                output_parts.append(f"({throughput_str})")

        if self.show_memory and metrics.memory_usage_mb > 0:
            output_parts.append(f"Memory: {metrics.memory_usage_mb:.1f}MB")

        logger.info("Output", data=" ".join(output_parts), end="\r", flush=True)

    def on_phase_change(
        self, old_phase: ProgressPhase, new_phase: ProgressPhase
    ) -> None:
        """Display phase change."""
        logger.info("\n{new_phase.value.replace('_', ' ').title()}...")

    def on_error(self, error: Exception, metrics: ProgressMetrics) -> None:
        """Display error information."""
        logger.info("\nError during {metrics.phase.value}: {error}")

    def _create_progress_bar(self, percent: float, width: int = 30) -> str:
        """Create ASCII progress bar."""
        filled = int(width * percent / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"

    def _format_time(self, seconds: float) -> str:
        """Format time duration as human-readable string."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"


class LoggingProgressCallback(ProgressCallback):
    """Progress callback that uses logging system."""

    def __init__(
        self,
        logger_instance: Optional[logging.Logger] = None,
        log_interval: float = 5.0,
    ):
        """
        Initialize logging progress callback.

        Args:
            logger_instance: Logger to use for output
            log_interval: Minimum interval between log messages (seconds)
        """
        self.logger = logger_instance or logger
        self.log_interval = log_interval
        self.last_log_time = 0.0

    def on_progress_update(self, metrics: ProgressMetrics) -> None:
        """Log progress update if interval has passed."""
        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            self.logger.info(
                f"Progress: {metrics.progress_percent:.1f}% "
                f"({metrics.current}/{metrics.total} {metrics.unit.value}) "
                f"- {metrics.phase.value}"
            )
            self.last_log_time = now

    def on_phase_change(
        self, old_phase: ProgressPhase, new_phase: ProgressPhase
    ) -> None:
        """Log phase change."""
        self.logger.info(f"Phase changed: {old_phase.value} -> {new_phase.value}")

    def on_error(self, error: Exception, metrics: ProgressMetrics) -> None:
        """Log error."""
        self.logger.error(f"Error during {metrics.phase.value}: {error}")


class ProgressTracker:
    """
    Comprehensive progress tracking system.

    Provides progress reporting for large file processing and batch operations
    with memory monitoring and performance metrics.
    """

    def __init__(
        self,
        total: int,
        unit: ProgressUnit = ProgressUnit.BYTES,
        callbacks: Optional[list[ProgressCallback]] = None,
        memory_limit_mb: Optional[float] = None,
        auto_update_memory: bool = True,
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total units of work to be done
            unit: Unit of measurement for progress
            callbacks: List of progress callbacks
            memory_limit_mb: Memory limit in MB (None for no limit)
            auto_update_memory: Whether to automatically update memory metrics
        """
        self.metrics = ProgressMetrics(
            total=total,
            unit=unit,
            memory_limit_mb=memory_limit_mb,
        )
        self.callbacks = callbacks or []
        self.auto_update_memory = auto_update_memory
        self._lock = threading.Lock()
        self._memory_monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

        if auto_update_memory:
            self._start_memory_monitoring()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

        if exc_type is not None:
            self.set_phase(ProgressPhase.FAILED)
            for callback in self.callbacks:
                try:
                    callback.on_error(exc_val, self.metrics)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
        else:
            self.set_phase(ProgressPhase.COMPLETED)

    def update(self, current: int, operation: Optional[str] = None) -> None:
        """
        Update progress.

        Args:
            current: Current progress value
            operation: Description of current operation
        """
        with self._lock:
            old_current = self.metrics.current
            self.metrics.current = min(current, self.metrics.total)
            self.metrics.current_operation = operation
            self.metrics.update_timing()

            # Check memory limit
            if (
                self.metrics.memory_limit_mb
                and self.metrics.memory_usage_mb > self.metrics.memory_limit_mb
            ):
                logger.warning(
                    f"Memory usage ({self.metrics.memory_usage_mb:.1f}MB) exceeds limit "
                    f"({self.metrics.memory_limit_mb}MB)"
                )

            # Notify callbacks if progress actually changed
            if self.metrics.current != old_current:
                self._notify_callbacks()

    def increment(self, amount: int = 1, operation: Optional[str] = None) -> None:
        """
        Increment progress by specified amount.

        Args:
            amount: Amount to increment
            operation: Description of current operation
        """
        self.update(self.metrics.current + amount, operation)

    def set_phase(self, phase: ProgressPhase) -> None:
        """
        Set current processing phase.

        Args:
            phase: New processing phase
        """
        with self._lock:
            old_phase = self.metrics.phase
            self.metrics.phase = phase

            if old_phase != phase:
                for callback in self.callbacks:
                    try:
                        callback.on_phase_change(old_phase, phase)
                    except Exception as e:
                        logger.warning(f"Progress callback error: {e}")

    def set_file_info(
        self, file_path: Union[str, Path], file_size: Optional[int] = None
    ) -> None:
        """
        Set file information for context.

        Args:
            file_path: Path to the file being processed
            file_size: Size of the file in bytes
        """
        with self._lock:
            self.metrics.file_path = str(file_path)
            self.metrics.file_size = file_size

    def add_warning(self) -> None:
        """Increment warning count."""
        with self._lock:
            self.metrics.warnings_count += 1

    def add_error(self, recoverable: bool = False) -> None:
        """
        Increment error count.

        Args:
            recoverable: Whether the error was recoverable
        """
        with self._lock:
            self.metrics.errors_count += 1
            if recoverable:
                self.metrics.recoverable_errors += 1

    def get_metrics(self) -> ProgressMetrics:
        """Get copy of current metrics."""
        with self._lock:
            return ProgressMetrics(**self.metrics.__dict__)

    def stop(self) -> None:
        """Stop progress tracking and cleanup."""
        if self._memory_monitor_thread and self._memory_monitor_thread.is_alive():
            self._stop_monitoring.set()
            self._memory_monitor_thread.join(timeout=1.0)

    def _start_memory_monitoring(self) -> None:
        """Start background memory monitoring thread."""

        def monitor_memory():
            while not self._stop_monitoring.wait(1.0):  # Check every second
                try:
                    self.metrics.update_memory()
                except Exception as e:
                    logger.debug(f"Memory monitoring error: {e}")

        self._memory_monitor_thread = threading.Thread(
            target=monitor_memory, daemon=True
        )
        self._memory_monitor_thread.start()

    def _notify_callbacks(self) -> None:
        """Notify all callbacks of progress update."""
        for callback in self.callbacks:
            try:
                callback.on_progress_update(self.metrics)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")


class BatchProgressTracker:
    """Progress tracker for batch operations with multiple files."""

    def __init__(
        self,
        total_files: int,
        callbacks: Optional[list[ProgressCallback]] = None,
        show_individual_progress: bool = False,
    ):
        """
        Initialize batch progress tracker.

        Args:
            total_files: Total number of files to process
            callbacks: List of progress callbacks
            show_individual_progress: Whether to show progress for individual files
        """
        self.total_files = total_files
        self.completed_files = 0
        self.callbacks = callbacks or []
        self.show_individual_progress = show_individual_progress
        self.current_file_tracker: Optional[ProgressTracker] = None
        self.batch_start_time = time.time()
        self.file_results: list[Dict[str, Any]] = []

        # Create batch-level progress tracker
        self.batch_tracker = ProgressTracker(
            total=total_files,
            unit=ProgressUnit.DOCUMENTS,
            callbacks=callbacks,
        )
        self.batch_tracker.set_phase(ProgressPhase.INITIALIZING)

    def start_file(
        self,
        file_path: Union[str, Path],
        file_size: Optional[int] = None,
    ) -> ProgressTracker:
        """
        Start processing a new file.

        Args:
            file_path: Path to the file being processed
            file_size: Size of the file in bytes

        Returns:
            ProgressTracker for the individual file
        """
        # Complete previous file if any
        if self.current_file_tracker:
            self.complete_current_file()

        # Create new file tracker
        callbacks = self.callbacks if self.show_individual_progress else []
        total_size = file_size or 1  # Use 1 if size unknown

        self.current_file_tracker = ProgressTracker(
            total=total_size,
            unit=ProgressUnit.BYTES,
            callbacks=callbacks,
        )

        self.current_file_tracker.set_file_info(file_path, file_size)
        self.current_file_tracker.set_phase(ProgressPhase.LOADING)

        # Update batch progress
        self.batch_tracker.update(
            self.completed_files, operation=f"Processing {Path(file_path).name}"
        )

        return self.current_file_tracker

    def complete_current_file(
        self, success: bool = True, error: Optional[str] = None
    ) -> None:
        """
        Complete processing of current file.

        Args:
            success: Whether file processing succeeded
            error: Error message if processing failed
        """
        if not self.current_file_tracker:
            return

        # Record file results
        metrics = self.current_file_tracker.get_metrics()
        result = {
            "file_path": metrics.file_path,
            "success": success,
            "error": error,
            "elapsed_time": metrics.elapsed_time,
            "memory_peak": metrics.peak_memory_mb,
            "warnings": metrics.warnings_count,
            "errors": metrics.errors_count,
        }
        self.file_results.append(result)

        # Update counters
        self.completed_files += 1

        if not success:
            self.batch_tracker.add_error()

        # Cleanup file tracker
        self.current_file_tracker.stop()
        self.current_file_tracker = None

        # Update batch progress
        self.batch_tracker.update(self.completed_files)

    def get_batch_summary(self) -> Dict[str, Any]:
        """Get summary of batch processing results."""
        total_time = time.time() - self.batch_start_time
        successful_files = sum(1 for result in self.file_results if result["success"])
        failed_files = len(self.file_results) - successful_files

        return {
            "total_files": self.total_files,
            "completed_files": self.completed_files,
            "successful_files": successful_files,
            "failed_files": failed_files,
            "total_time": total_time,
            "average_time_per_file": total_time / max(1, self.completed_files),
            "results": self.file_results,
        }

    def __enter__(self):
        """Context manager entry."""
        self.batch_tracker.set_phase(ProgressPhase.PROCESSING)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current_file_tracker:
            self.complete_current_file(
                success=exc_type is None, error=str(exc_val) if exc_val else None
            )

        if exc_type is None:
            self.batch_tracker.set_phase(ProgressPhase.COMPLETED)
        else:
            self.batch_tracker.set_phase(ProgressPhase.FAILED)

        self.batch_tracker.stop()


# Convenience functions
def create_progress_tracker(
    total: int,
    unit: ProgressUnit = ProgressUnit.BYTES,
    show_console: bool = True,
    show_memory: bool = True,
    memory_limit_mb: Optional[float] = None,
) -> ProgressTracker:
    """
    Create a progress tracker with default console output.

    Args:
        total: Total units of work
        unit: Unit of measurement
        show_console: Whether to show console progress
        show_memory: Whether to show memory usage
        memory_limit_mb: Memory limit in MB

    Returns:
        Configured ProgressTracker instance
    """
    callbacks = []

    if show_console:
        callbacks.append(ConsoleProgressCallback(show_memory=show_memory))

    # Always add logging callback
    callbacks.append(LoggingProgressCallback())

    return ProgressTracker(
        total=total,
        unit=unit,
        callbacks=callbacks,
        memory_limit_mb=memory_limit_mb,
    )


def create_batch_progress_tracker(
    total_files: int,
    show_console: bool = True,
    show_individual_files: bool = False,
) -> BatchProgressTracker:
    """
    Create a batch progress tracker with default console output.

    Args:
        total_files: Total number of files to process
        show_console: Whether to show console progress
        show_individual_files: Whether to show individual file progress

    Returns:
        Configured BatchProgressTracker instance
    """
    callbacks = []

    if show_console:
        callbacks.append(ConsoleProgressCallback())

    # Always add logging callback
    callbacks.append(LoggingProgressCallback())

    return BatchProgressTracker(
        total_files=total_files,
        callbacks=callbacks,
        show_individual_progress=show_individual_files,
    )
