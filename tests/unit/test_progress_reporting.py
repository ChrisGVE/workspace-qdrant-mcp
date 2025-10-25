"""
Tests for progress reporting infrastructure.
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest
from wqm_cli.cli.parsers.progress import (
    BatchProgressTracker,
    ConsoleProgressCallback,
    LoggingProgressCallback,
    ProgressMetrics,
    ProgressPhase,
    ProgressTracker,
    ProgressUnit,
    create_batch_progress_tracker,
    create_progress_tracker,
)


class TestProgressMetrics:
    """Test ProgressMetrics functionality."""

    def test_progress_metrics_creation(self):
        """Test creating progress metrics."""
        metrics = ProgressMetrics(
            current=50,
            total=100,
            phase=ProgressPhase.PROCESSING,
            unit=ProgressUnit.BYTES
        )

        assert metrics.current == 50
        assert metrics.total == 100
        assert metrics.phase == ProgressPhase.PROCESSING
        assert metrics.unit == ProgressUnit.BYTES
        assert metrics.progress_percent == 50.0

    def test_progress_percent_calculation(self):
        """Test progress percentage calculation."""
        metrics = ProgressMetrics(current=25, total=100)
        assert metrics.progress_percent == 25.0

        metrics = ProgressMetrics(current=0, total=100)
        assert metrics.progress_percent == 0.0

        metrics = ProgressMetrics(current=100, total=100)
        assert metrics.progress_percent == 100.0

        metrics = ProgressMetrics(current=150, total=100)
        assert metrics.progress_percent == 100.0  # Clamped to 100%

        metrics = ProgressMetrics(current=50, total=0)
        assert metrics.progress_percent == 0.0  # Handle zero division

    def test_is_complete_property(self):
        """Test is_complete property."""
        metrics = ProgressMetrics(phase=ProgressPhase.PROCESSING)
        assert not metrics.is_complete

        metrics.phase = ProgressPhase.COMPLETED
        assert metrics.is_complete

        metrics.phase = ProgressPhase.FAILED
        assert metrics.is_complete

    def test_update_timing(self):
        """Test timing metrics update."""
        metrics = ProgressMetrics(current=0, total=100)
        start_time = metrics.start_time

        # Simulate some progress
        time.sleep(0.01)  # Small delay
        metrics.current = 50
        metrics.update_timing()

        assert metrics.elapsed_time > 0
        assert metrics.last_update > start_time
        assert metrics.throughput > 0

    def test_estimated_remaining_time(self):
        """Test estimated remaining time calculation."""
        metrics = ProgressMetrics(current=25, total=100)
        metrics.elapsed_time = 10.0  # 10 seconds elapsed
        metrics.update_timing()

        # Should estimate ~30 seconds remaining (75% left, took 10s for 25%)
        assert metrics.estimated_remaining > 20.0
        assert metrics.estimated_remaining < 40.0

    @patch('psutil.Process')
    def test_update_memory(self, mock_process):
        """Test memory metrics update."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 100  # 100MB in bytes
        mock_process.return_value.memory_info.return_value = mock_memory_info

        metrics = ProgressMetrics()
        metrics.update_memory()

        assert metrics.memory_usage_mb == 100.0
        assert metrics.peak_memory_mb == 100.0

    def test_to_dict(self):
        """Test converting metrics to dictionary."""
        metrics = ProgressMetrics(
            current=75,
            total=100,
            phase=ProgressPhase.ANALYZING,
            unit=ProgressUnit.PAGES,
            file_path="test.pdf",
            warnings_count=2,
            errors_count=1
        )

        data = metrics.to_dict()

        assert data["current"] == 75
        assert data["total"] == 100
        assert data["progress_percent"] == 75.0
        assert data["phase"] == "analyzing"
        assert data["unit"] == "pages"
        assert data["file_path"] == "test.pdf"
        assert data["warnings_count"] == 2
        assert data["errors_count"] == 1
        assert isinstance(data["is_complete"], bool)


class TestProgressCallbacks:
    """Test progress callback implementations."""

    def test_console_progress_callback(self, capsys):
        """Test console progress callback output."""
        callback = ConsoleProgressCallback(show_memory=False, show_throughput=False)

        metrics = ProgressMetrics(current=50, total=100, unit=ProgressUnit.BYTES)
        callback.on_progress_update(metrics)

        captured = capsys.readouterr()
        assert "50.0%" in captured.out
        assert "(50/100 bytes)" in captured.out

    def test_console_callback_with_throughput(self, capsys):
        """Test console callback with throughput display."""
        callback = ConsoleProgressCallback(show_throughput=True)

        metrics = ProgressMetrics(current=50, total=100, unit=ProgressUnit.BYTES)
        metrics.throughput = 10.5
        metrics.estimated_remaining = 5.0

        callback.on_progress_update(metrics)

        captured = capsys.readouterr()
        assert "10.5 bytes/s" in captured.out
        assert "ETA:" in captured.out

    def test_console_callback_with_memory(self, capsys):
        """Test console callback with memory display."""
        callback = ConsoleProgressCallback(show_memory=True)

        metrics = ProgressMetrics(current=25, total=100)
        metrics.memory_usage_mb = 128.5

        callback.on_progress_update(metrics)

        captured = capsys.readouterr()
        assert "Memory: 128.5MB" in captured.out

    def test_console_callback_phase_change(self, capsys):
        """Test console callback phase change display."""
        callback = ConsoleProgressCallback()

        callback.on_phase_change(ProgressPhase.LOADING, ProgressPhase.PROCESSING)

        captured = capsys.readouterr()
        assert "Processing..." in captured.out

    def test_console_callback_error(self, capsys):
        """Test console callback error display."""
        callback = ConsoleProgressCallback()
        metrics = ProgressMetrics(phase=ProgressPhase.PARSING)
        error = ValueError("Test error")

        callback.on_error(error, metrics)

        captured = capsys.readouterr()
        assert "Error during parsing: Test error" in captured.out

    def test_logging_progress_callback(self, caplog):
        """Test logging progress callback."""
        callback = LoggingProgressCallback(log_interval=0.0)  # No interval for testing

        metrics = ProgressMetrics(
            current=30,
            total=100,
            unit=ProgressUnit.PAGES,
            phase=ProgressPhase.EXTRACTING
        )

        callback.on_progress_update(metrics)

        assert "Progress: 30.0%" in caplog.text
        assert "(30/100 pages)" in caplog.text
        assert "extracting" in caplog.text

    def test_logging_callback_interval(self, caplog):
        """Test logging callback respects interval."""
        callback = LoggingProgressCallback(log_interval=60.0)  # 1 minute interval

        metrics = ProgressMetrics(current=50, total=100)

        # First call should log
        callback.on_progress_update(metrics)
        assert len(caplog.records) == 1

        # Immediate second call should not log
        caplog.clear()
        callback.on_progress_update(metrics)
        assert len(caplog.records) == 0


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    def test_progress_tracker_creation(self):
        """Test creating a progress tracker."""
        tracker = ProgressTracker(total=100, unit=ProgressUnit.BYTES)

        assert tracker.metrics.total == 100
        assert tracker.metrics.unit == ProgressUnit.BYTES
        assert tracker.metrics.current == 0
        assert tracker.metrics.phase == ProgressPhase.INITIALIZING

    def test_progress_update(self):
        """Test updating progress."""
        tracker = ProgressTracker(total=100)

        tracker.update(25, "Processing data")

        assert tracker.metrics.current == 25
        assert tracker.metrics.current_operation == "Processing data"
        assert tracker.metrics.progress_percent == 25.0

    def test_progress_increment(self):
        """Test incrementing progress."""
        tracker = ProgressTracker(total=100)

        tracker.increment(10)
        assert tracker.metrics.current == 10

        tracker.increment(15, "More processing")
        assert tracker.metrics.current == 25
        assert tracker.metrics.current_operation == "More processing"

    def test_phase_setting(self):
        """Test setting processing phase."""
        mock_callback = Mock()
        tracker = ProgressTracker(total=100, callbacks=[mock_callback])

        tracker.set_phase(ProgressPhase.LOADING)

        assert tracker.metrics.phase == ProgressPhase.LOADING
        mock_callback.on_phase_change.assert_called_once_with(
            ProgressPhase.INITIALIZING, ProgressPhase.LOADING
        )

    def test_file_info_setting(self):
        """Test setting file information."""
        tracker = ProgressTracker(total=100)

        tracker.set_file_info("test.pdf", 2048)

        assert tracker.metrics.file_path == "test.pdf"
        assert tracker.metrics.file_size == 2048

    def test_warning_and_error_tracking(self):
        """Test warning and error counting."""
        tracker = ProgressTracker(total=100)

        tracker.add_warning()
        tracker.add_warning()
        tracker.add_error(recoverable=True)
        tracker.add_error(recoverable=False)

        assert tracker.metrics.warnings_count == 2
        assert tracker.metrics.errors_count == 2
        assert tracker.metrics.recoverable_errors == 1

    def test_memory_limit_warning(self, caplog):
        """Test memory limit warning."""
        tracker = ProgressTracker(total=100, memory_limit_mb=100.0)
        tracker.metrics.memory_usage_mb = 150.0  # Exceed limit

        with caplog.at_level("WARNING"):
            tracker.update(50)

            assert "Memory usage" in caplog.text
            assert "exceeds limit" in caplog.text

    def test_progress_callbacks(self):
        """Test progress callback notifications."""
        mock_callback = Mock()
        tracker = ProgressTracker(total=100, callbacks=[mock_callback])

        tracker.update(50, "Test operation")

        mock_callback.on_progress_update.assert_called_once()
        args = mock_callback.on_progress_update.call_args[0]
        assert args[0].current == 50
        assert args[0].current_operation == "Test operation"

    def test_context_manager(self):
        """Test using tracker as context manager."""
        with ProgressTracker(total=100) as tracker:
            tracker.update(50)
            assert tracker.metrics.current == 50

        # Should be completed after context exit
        assert tracker.metrics.phase == ProgressPhase.COMPLETED

    def test_context_manager_with_exception(self):
        """Test context manager with exception."""
        mock_callback = Mock()

        try:
            with ProgressTracker(total=100, callbacks=[mock_callback]) as tracker:
                tracker.update(25)
                raise ValueError("Test error")
        except ValueError:
            pass

        assert tracker.metrics.phase == ProgressPhase.FAILED
        mock_callback.on_error.assert_called_once()

    def test_memory_monitoring_thread(self):
        """Test background memory monitoring."""
        with patch('psutil.Process') as mock_process:
            # Setup mock
            mock_memory = Mock()
            mock_memory.rss = 1024 * 1024 * 50  # 50MB
            mock_process.return_value.memory_info.return_value = mock_memory

            tracker = ProgressTracker(total=100, auto_update_memory=True)

            # Give some time for background thread to run
            time.sleep(0.1)

            tracker.stop()

            # Memory should have been updated
            assert tracker.metrics.memory_usage_mb > 0

    def test_get_metrics_copy(self):
        """Test getting copy of metrics."""
        tracker = ProgressTracker(total=100)
        tracker.update(75)

        metrics_copy = tracker.get_metrics()

        assert metrics_copy.current == 75
        assert metrics_copy is not tracker.metrics  # Should be a copy

    def test_stop_cleanup(self):
        """Test stopping tracker cleanup."""
        tracker = ProgressTracker(total=100, auto_update_memory=True)

        # Start and then stop
        tracker.stop()

        # Should complete without hanging
        assert True


class TestBatchProgressTracker:
    """Test BatchProgressTracker functionality."""

    def test_batch_tracker_creation(self):
        """Test creating batch progress tracker."""
        batch_tracker = BatchProgressTracker(total_files=5)

        assert batch_tracker.total_files == 5
        assert batch_tracker.completed_files == 0
        assert len(batch_tracker.file_results) == 0

    def test_start_file_processing(self):
        """Test starting file processing."""
        batch_tracker = BatchProgressTracker(total_files=3)

        file_tracker = batch_tracker.start_file("file1.txt", 1024)

        assert file_tracker is not None
        assert file_tracker.metrics.file_path == "file1.txt"
        assert file_tracker.metrics.file_size == 1024
        assert batch_tracker.current_file_tracker == file_tracker

    def test_complete_current_file_success(self):
        """Test completing file successfully."""
        batch_tracker = BatchProgressTracker(total_files=2)

        # Start and complete a file
        file_tracker = batch_tracker.start_file("file1.txt")
        file_tracker.update(100)  # Simulate some progress

        batch_tracker.complete_current_file(success=True)

        assert batch_tracker.completed_files == 1
        assert len(batch_tracker.file_results) == 1
        assert batch_tracker.file_results[0]["success"] is True
        assert batch_tracker.current_file_tracker is None

    def test_complete_current_file_failure(self):
        """Test completing file with failure."""
        batch_tracker = BatchProgressTracker(total_files=2)

        batch_tracker.start_file("file1.txt")
        batch_tracker.complete_current_file(success=False, error="Parse error")

        assert batch_tracker.completed_files == 1
        result = batch_tracker.file_results[0]
        assert result["success"] is False
        assert result["error"] == "Parse error"

    def test_batch_summary(self):
        """Test getting batch summary."""
        batch_tracker = BatchProgressTracker(total_files=3)

        # Process some files
        batch_tracker.start_file("file1.txt")
        batch_tracker.complete_current_file(success=True)

        batch_tracker.start_file("file2.txt")
        batch_tracker.complete_current_file(success=False, error="Error")

        summary = batch_tracker.get_batch_summary()

        assert summary["total_files"] == 3
        assert summary["completed_files"] == 2
        assert summary["successful_files"] == 1
        assert summary["failed_files"] == 1
        assert "total_time" in summary
        assert len(summary["results"]) == 2

    def test_batch_context_manager(self):
        """Test batch tracker as context manager."""
        with BatchProgressTracker(total_files=2) as batch_tracker:
            file_tracker = batch_tracker.start_file("file1.txt")
            file_tracker.update(50)

        # Should auto-complete current file on exit
        assert batch_tracker.completed_files == 1

    def test_multiple_files_processing(self):
        """Test processing multiple files in sequence."""
        batch_tracker = BatchProgressTracker(total_files=3)

        # Process first file
        tracker1 = batch_tracker.start_file("file1.txt", 100)
        tracker1.update(100)
        batch_tracker.complete_current_file(success=True)

        # Process second file
        tracker2 = batch_tracker.start_file("file2.txt", 200)
        tracker2.update(150)
        batch_tracker.complete_current_file(success=True)

        # Verify batch progress
        assert batch_tracker.completed_files == 2
        assert len(batch_tracker.file_results) == 2

        # Verify individual file results
        assert batch_tracker.file_results[0]["file_path"] == "file1.txt"
        assert batch_tracker.file_results[1]["file_path"] == "file2.txt"


class TestConvenienceFunctions:
    """Test convenience functions for creating trackers."""

    def test_create_progress_tracker(self):
        """Test create_progress_tracker convenience function."""
        tracker = create_progress_tracker(
            total=100,
            unit=ProgressUnit.PAGES,
            show_console=True,
            show_memory=True,
            memory_limit_mb=500.0
        )

        assert tracker.metrics.total == 100
        assert tracker.metrics.unit == ProgressUnit.PAGES
        assert tracker.metrics.memory_limit_mb == 500.0
        assert len(tracker.callbacks) >= 2  # Console + Logging

    def test_create_progress_tracker_no_console(self):
        """Test creating tracker without console output."""
        tracker = create_progress_tracker(
            total=50,
            show_console=False
        )

        assert tracker.metrics.total == 50
        assert len(tracker.callbacks) == 1  # Only logging callback

    def test_create_batch_progress_tracker(self):
        """Test create_batch_progress_tracker convenience function."""
        batch_tracker = create_batch_progress_tracker(
            total_files=5,
            show_console=True,
            show_individual_files=True
        )

        assert batch_tracker.total_files == 5
        assert batch_tracker.show_individual_progress is True
        assert len(batch_tracker.callbacks) >= 2

    def test_create_batch_tracker_no_individual(self):
        """Test creating batch tracker without individual progress."""
        batch_tracker = create_batch_progress_tracker(
            total_files=10,
            show_individual_files=False
        )

        assert batch_tracker.show_individual_progress is False


class TestProgressIntegration:
    """Integration tests for progress system."""

    def test_progress_with_threading(self):
        """Test progress tracking with multiple threads."""
        tracker = ProgressTracker(total=1000)
        results = []

        def update_progress(start, end):
            for i in range(start, end):
                tracker.increment(1, f"Processing item {i}")
                results.append(i)
                time.sleep(0.001)  # Small delay

        # Create threads to update progress
        thread1 = threading.Thread(target=update_progress, args=(0, 50))
        thread2 = threading.Thread(target=update_progress, args=(50, 100))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        assert tracker.metrics.current == 100
        assert len(results) == 100

    def test_real_file_processing_simulation(self):
        """Test simulating real file processing with progress."""
        with BatchProgressTracker(total_files=3) as batch_tracker:
            # Simulate processing 3 files
            for i in range(3):
                file_name = f"file{i+1}.txt"
                file_size = 1000 * (i + 1)  # Varying file sizes

                with batch_tracker.start_file(file_name, file_size) as file_tracker:
                    # Simulate file processing phases
                    file_tracker.set_phase(ProgressPhase.LOADING)
                    file_tracker.update(file_size // 4, "Loading file")

                    file_tracker.set_phase(ProgressPhase.PARSING)
                    file_tracker.update(file_size // 2, "Parsing content")

                    file_tracker.set_phase(ProgressPhase.PROCESSING)
                    file_tracker.update(file_size * 3 // 4, "Processing text")

                    file_tracker.set_phase(ProgressPhase.FINALIZING)
                    file_tracker.update(file_size, "Creating document")

                batch_tracker.complete_current_file(success=True)

        # Verify all files were processed
        summary = batch_tracker.get_batch_summary()
        assert summary["successful_files"] == 3
        assert summary["failed_files"] == 0
        assert summary["completed_files"] == 3

    def test_error_handling_with_progress(self):
        """Test progress tracking with error scenarios."""
        mock_callback = Mock()

        try:
            with ProgressTracker(total=100, callbacks=[mock_callback]) as tracker:
                tracker.update(25)
                tracker.add_warning()
                tracker.update(50)
                tracker.add_error()
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass

        # Verify error was reported to callback
        mock_callback.on_error.assert_called_once()

        # Verify final state
        assert tracker.metrics.phase == ProgressPhase.FAILED
        assert tracker.metrics.warnings_count == 1
        assert tracker.metrics.errors_count == 1
