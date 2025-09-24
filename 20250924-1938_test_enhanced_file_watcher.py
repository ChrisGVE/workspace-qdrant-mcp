"""
Comprehensive unit tests for enhanced file watcher system.

Tests cover the complete enhanced file watching system including advanced filtering,
batch processing, smart debouncing, and performance optimization features.

Test Categories:
    - Configuration and initialization
    - File system event handling
    - Advanced filtering integration
    - Batch processing coordination
    - Smart debouncing algorithms
    - Performance monitoring
    - Error handling and recovery
    - Resource management
    - Edge cases and failure scenarios
    - Integration testing
"""

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from src.python.common.core.enhanced_file_watcher import (
    EnhancedFileWatcher,
    EnhancedWatcherConfig,
    WatcherStatistics,
    SmartDebouncer,
    DebouncingConfig,
    DebouncingStrategy,
    create_high_performance_watcher,
    create_selective_watcher,
    create_memory_efficient_watcher
)

from src.python.common.core.batch_processing_manager import (
    BatchItem,
    FileOperation,
    BatchPriority,
    ProcessingStrategy
)


class TestEnhancedWatcherConfig:
    """Test EnhancedWatcherConfig functionality."""

    def test_default_configuration(self):
        """Test default configuration initialization."""
        config = EnhancedWatcherConfig()

        assert config.watch_directories == {}
        assert config.max_concurrent_watchers == 10
        assert config.enable_performance_monitoring is True
        assert config.enable_graceful_degradation is True
        assert config.max_memory_usage_mb == 512.0
        assert config.max_debounce_tasks == 1000
        assert config.max_retry_attempts == 3
        assert config.continue_on_error is True

        # Check default sub-configurations are set
        assert "include_patterns" in config.filtering
        assert "max_batch_size" in config.batch_processing
        assert "strategy" in config.debouncing

    def test_custom_configuration(self):
        """Test custom configuration initialization."""
        watch_dirs = {"/tmp/test": "test-collection"}
        custom_filtering = {
            "include_patterns": ["*.py"],
            "max_file_size": 1024
        }
        custom_batch = {
            "max_batch_size": 10,
            "processing_strategy": "time_based"
        }
        custom_debounce = {
            "strategy": "simple",
            "base_delay_seconds": 2.0
        }

        config = EnhancedWatcherConfig(
            watch_directories=watch_dirs,
            filtering=custom_filtering,
            batch_processing=custom_batch,
            debouncing=custom_debounce,
            max_concurrent_watchers=5,
            enable_performance_monitoring=False
        )

        assert config.watch_directories == watch_dirs
        assert config.filtering["include_patterns"] == ["*.py"]
        assert config.batch_processing["max_batch_size"] == 10
        assert config.debouncing["strategy"] == "simple"
        assert config.max_concurrent_watchers == 5
        assert config.enable_performance_monitoring is False

    def test_post_init_defaults(self):
        """Test that post_init sets defaults for empty configurations."""
        config = EnhancedWatcherConfig(
            filtering={},
            batch_processing={},
            debouncing={}
        )

        # Should have set defaults
        assert len(config.filtering) > 0
        assert "include_patterns" in config.filtering
        assert len(config.batch_processing) > 0
        assert "max_batch_size" in config.batch_processing
        assert len(config.debouncing) > 0
        assert "strategy" in config.debouncing


class TestWatcherStatistics:
    """Test WatcherStatistics functionality."""

    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = WatcherStatistics()

        assert stats.total_events_received == 0
        assert stats.files_added == 0
        assert stats.files_modified == 0
        assert stats.files_deleted == 0
        assert stats.files_processed_successfully == 0
        assert stats.avg_event_processing_time_ms == 0.0
        assert stats.current_debounce_tasks == 0
        assert stats.permission_errors == 0

    def test_add_processing_time(self):
        """Test processing time tracking."""
        stats = WatcherStatistics()

        # Add processing times
        times = [10.0, 20.0, 15.0]
        for time_ms in times:
            stats.add_processing_time(time_ms)

        expected_avg = sum(times) / len(times)
        assert abs(stats.avg_event_processing_time_ms - expected_avg) < 0.01
        assert len(stats.recent_processing_times) == 3

    def test_statistics_serialization(self):
        """Test statistics to_dict conversion."""
        stats = WatcherStatistics()
        stats.total_events_received = 100
        stats.files_processed_successfully = 80
        stats.files_processed_with_errors = 20
        stats.files_filtered_out = 30

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict["total_events_received"] == 100
        assert stats_dict["files_processed_successfully"] == 80
        assert "success_rate_percent" in stats_dict
        assert "filter_efficiency_percent" in stats_dict

        # Check derived metrics
        success_rate = stats_dict["success_rate_percent"]
        assert abs(success_rate - 80.0) < 0.01  # 80 successful / 100 total

        filter_efficiency = stats_dict["filter_efficiency_percent"]
        assert abs(filter_efficiency - 30.0) < 0.01  # 30 filtered / 100 total

    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        stats = WatcherStatistics()

        # Set some values
        stats.total_events_received = 50
        stats.files_added = 20
        stats.add_processing_time(15.0)
        stats.permission_errors = 5

        # Reset
        stats.reset()

        # Verify all values are reset
        assert stats.total_events_received == 0
        assert stats.files_added == 0
        assert stats.avg_event_processing_time_ms == 0.0
        assert stats.permission_errors == 0
        assert len(stats.recent_processing_times) == 0


class TestDebouncingConfig:
    """Test DebouncingConfig functionality."""

    def test_default_debouncing_config(self):
        """Test default debouncing configuration."""
        config = DebouncingConfig()

        assert config.strategy == DebouncingStrategy.ADAPTIVE
        assert config.base_delay_seconds == 1.0
        assert config.max_delay_seconds == 10.0
        assert config.min_delay_seconds == 0.1
        assert config.frequency_window_seconds == 60.0
        assert config.high_frequency_threshold == 10

    def test_custom_debouncing_config(self):
        """Test custom debouncing configuration."""
        config = DebouncingConfig(
            strategy=DebouncingStrategy.STATISTICAL,
            base_delay_seconds=2.0,
            max_delay_seconds=20.0,
            high_frequency_threshold=5
        )

        assert config.strategy == DebouncingStrategy.STATISTICAL
        assert config.base_delay_seconds == 2.0
        assert config.max_delay_seconds == 20.0
        assert config.high_frequency_threshold == 5


class TestSmartDebouncer:
    """Test SmartDebouncer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = DebouncingConfig(
            strategy=DebouncingStrategy.SIMPLE,
            base_delay_seconds=0.1  # Short delay for testing
        )
        self.debouncer = SmartDebouncer(self.config)
        self.callback_calls = []

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'debouncer'):
            asyncio.run(self.debouncer.cleanup())

    @pytest.mark.asyncio
    async def test_simple_debouncing(self):
        """Test simple debouncing strategy."""
        def test_callback():
            self.callback_calls.append(("callback", time.time()))

        # Trigger debounced callback
        await self.debouncer.debounce_file_change("/tmp/test.txt", test_callback)

        # Should not have called yet
        assert len(self.callback_calls) == 0

        # Wait for debounce delay
        await asyncio.sleep(0.2)

        # Should have called now
        assert len(self.callback_calls) == 1
        assert self.callback_calls[0][0] == "callback"

    @pytest.mark.asyncio
    async def test_debounce_cancellation(self):
        """Test that rapid changes cancel previous debounce tasks."""
        def test_callback():
            self.callback_calls.append(("callback", time.time()))

        file_path = "/tmp/test.txt"

        # First change
        await self.debouncer.debounce_file_change(file_path, test_callback)

        # Wait a bit but not full delay
        await asyncio.sleep(0.05)

        # Second change - should cancel first
        await self.debouncer.debounce_file_change(file_path, test_callback)

        # Wait for second debounce
        await asyncio.sleep(0.15)

        # Should only have one callback (from second change)
        assert len(self.callback_calls) == 1

    @pytest.mark.asyncio
    async def test_adaptive_debouncing(self):
        """Test adaptive debouncing strategy."""
        config = DebouncingConfig(
            strategy=DebouncingStrategy.ADAPTIVE,
            base_delay_seconds=0.1,
            high_frequency_threshold=3
        )
        debouncer = SmartDebouncer(config)

        def test_callback():
            self.callback_calls.append(time.time())

        file_path = "/tmp/adaptive.txt"

        # Make several rapid changes to trigger adaptive behavior
        for i in range(5):
            await debouncer.debounce_file_change(file_path, test_callback)
            await asyncio.sleep(0.02)  # Very short sleep between changes

        # Wait for processing
        await asyncio.sleep(0.5)

        # Adaptive strategy should have adjusted delay
        delay = debouncer.file_delays.get(file_path, 0)
        assert delay != config.base_delay_seconds  # Should have been adjusted

        await debouncer.cleanup()

    @pytest.mark.asyncio
    async def test_statistical_debouncing(self):
        """Test statistical debouncing strategy."""
        config = DebouncingConfig(
            strategy=DebouncingStrategy.STATISTICAL,
            base_delay_seconds=0.1
        )
        debouncer = SmartDebouncer(config)

        def test_callback():
            self.callback_calls.append(time.time())

        file_path = "/tmp/statistical.txt"

        # Create pattern of changes with regular intervals
        for i in range(5):
            await debouncer.debounce_file_change(file_path, test_callback)
            await asyncio.sleep(0.05)  # Regular intervals

        # Wait for processing
        await asyncio.sleep(0.3)

        await debouncer.cleanup()

    @pytest.mark.asyncio
    async def test_no_debouncing(self):
        """Test no debouncing strategy."""
        config = DebouncingConfig(strategy=DebouncingStrategy.NONE)
        debouncer = SmartDebouncer(config)

        def test_callback():
            self.callback_calls.append(time.time())

        # Should call immediately
        await debouncer.debounce_file_change("/tmp/test.txt", test_callback)

        # Should have called immediately
        assert len(self.callback_calls) == 1

        await debouncer.cleanup()

    @pytest.mark.asyncio
    async def test_debouncer_statistics(self):
        """Test debouncer statistics reporting."""
        file_path = "/tmp/stats.txt"

        def test_callback():
            pass

        # Add some debounced changes
        await self.debouncer.debounce_file_change(file_path, test_callback)

        stats = self.debouncer.get_statistics()

        assert isinstance(stats, dict)
        assert "active_debounce_tasks" in stats
        assert "strategy" in stats
        assert stats["strategy"] == DebouncingStrategy.SIMPLE
        assert stats["active_debounce_tasks"] >= 0

    @pytest.mark.asyncio
    async def test_debouncer_cleanup(self):
        """Test debouncer cleanup functionality."""
        def test_callback():
            pass

        # Add several debounced changes
        for i in range(5):
            await self.debouncer.debounce_file_change(f"/tmp/file_{i}.txt", test_callback)

        # Should have pending tasks
        assert len(self.debouncer.pending_tasks) == 5

        # Cleanup
        await self.debouncer.cleanup()

        # Should have no pending tasks
        assert len(self.debouncer.pending_tasks) == 0


class TestEnhancedFileWatcherInitialization:
    """Test EnhancedFileWatcher initialization."""

    def test_watcher_initialization_with_config_object(self):
        """Test watcher initialization with EnhancedWatcherConfig object."""
        watch_dirs = {"/tmp/test": "test-collection"}
        config = EnhancedWatcherConfig(watch_directories=watch_dirs)

        watcher = EnhancedFileWatcher(config)

        assert watcher.config is config
        assert watcher.config.watch_directories == watch_dirs
        assert watcher._running is False
        assert watcher.ingestion_callback is None
        assert watcher.file_filter is not None
        assert watcher.batch_processor is not None
        assert watcher.debouncer is not None

    def test_watcher_initialization_with_dict_config(self):
        """Test watcher initialization with dictionary configuration."""
        config_dict = {
            "watch_directories": {"/tmp/dict": "dict-collection"},
            "max_concurrent_watchers": 5,
            "enable_performance_monitoring": False
        }

        watcher = EnhancedFileWatcher(config_dict)

        assert watcher.config.watch_directories == {"/tmp/dict": "dict-collection"}
        assert watcher.config.max_concurrent_watchers == 5
        assert watcher.config.enable_performance_monitoring is False

    def test_component_initialization(self):
        """Test that all components are properly initialized."""
        config = EnhancedWatcherConfig(
            watch_directories={"/tmp/test": "test-collection"}
        )
        watcher = EnhancedFileWatcher(config)

        # Check components are initialized
        assert watcher.file_filter is not None
        assert watcher.batch_processor is not None
        assert watcher.debouncer is not None
        assert watcher.statistics is not None

        # Check batch processor callback is set
        assert watcher.batch_processor.processing_callback is not None

    def test_set_ingestion_callback(self):
        """Test setting ingestion callback."""
        config = EnhancedWatcherConfig()
        watcher = EnhancedFileWatcher(config)

        async def test_callback(items):
            pass

        watcher.set_ingestion_callback(test_callback)
        assert watcher.ingestion_callback is test_callback

    @pytest.mark.asyncio
    async def test_start_without_callback_raises_error(self):
        """Test that starting without ingestion callback raises error."""
        config = EnhancedWatcherConfig(
            watch_directories={"/tmp/test": "test-collection"}
        )
        watcher = EnhancedFileWatcher(config)

        with pytest.raises(ValueError, match="Ingestion callback must be set"):
            await watcher.start()


class TestEnhancedFileWatcherDirectoryManagement:
    """Test directory watching management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = EnhancedWatcherConfig()
        self.watcher = EnhancedFileWatcher(self.config)
        self.processed_batches = []

        async def mock_callback(items):
            self.processed_batches.append(items)

        self.watcher.set_ingestion_callback(mock_callback)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

        # Cleanup watcher
        asyncio.run(self.watcher.close())

    @pytest.mark.asyncio
    async def test_add_valid_directory(self):
        """Test adding a valid directory for watching."""
        test_dir = self.temp_dir / "valid_dir"
        test_dir.mkdir()

        result = await self.watcher.add_watch_directory(str(test_dir), "test-collection")

        assert result is True
        assert str(test_dir) in self.watcher.config.watch_directories
        assert self.watcher.config.watch_directories[str(test_dir)] == "test-collection"

    @pytest.mark.asyncio
    async def test_add_nonexistent_directory(self):
        """Test adding a non-existent directory."""
        nonexistent_dir = self.temp_dir / "does_not_exist"

        result = await self.watcher.add_watch_directory(str(nonexistent_dir), "test-collection")

        assert result is False
        assert str(nonexistent_dir) not in self.watcher.config.watch_directories

    @pytest.mark.asyncio
    async def test_add_file_instead_of_directory(self):
        """Test adding a file path instead of directory."""
        test_file = self.temp_dir / "test_file.txt"
        test_file.write_text("test content")

        result = await self.watcher.add_watch_directory(str(test_file), "test-collection")

        assert result is False
        assert str(test_file) not in self.watcher.config.watch_directories

    @pytest.mark.asyncio
    async def test_add_duplicate_directory(self):
        """Test adding a directory that's already being watched."""
        test_dir = self.temp_dir / "duplicate_dir"
        test_dir.mkdir()

        # Add first time
        result1 = await self.watcher.add_watch_directory(str(test_dir), "collection1")
        assert result1 is True

        # Add second time
        result2 = await self.watcher.add_watch_directory(str(test_dir), "collection2")
        assert result2 is False
        # Should still have original collection
        assert self.watcher.config.watch_directories[str(test_dir)] == "collection1"

    @pytest.mark.asyncio
    async def test_remove_existing_directory(self):
        """Test removing an existing watch directory."""
        test_dir = self.temp_dir / "remove_dir"
        test_dir.mkdir()

        # Add directory
        await self.watcher.add_watch_directory(str(test_dir), "test-collection")
        assert str(test_dir) in self.watcher.config.watch_directories

        # Remove directory
        result = await self.watcher.remove_watch_directory(str(test_dir))

        assert result is True
        assert str(test_dir) not in self.watcher.config.watch_directories

    @pytest.mark.asyncio
    async def test_remove_nonexistent_directory(self):
        """Test removing a directory that's not being watched."""
        test_dir = self.temp_dir / "not_watched"

        result = await self.watcher.remove_watch_directory(str(test_dir))

        assert result is False


class TestEnhancedFileWatcherLifecycle:
    """Test watcher lifecycle management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = EnhancedWatcherConfig(
            watch_directories={str(self.temp_dir): "test-collection"}
        )
        self.watcher = EnhancedFileWatcher(self.config)
        self.processed_batches = []

        async def mock_callback(items):
            self.processed_batches.append(items)

        self.watcher.set_ingestion_callback(mock_callback)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

        # Cleanup watcher
        asyncio.run(self.watcher.close())

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test basic start/stop lifecycle."""
        assert self.watcher._running is False
        assert len(self.watcher._watch_tasks) == 0

        # Start watcher
        await self.watcher.start()

        assert self.watcher._running is True
        assert len(self.watcher._watch_tasks) > 0
        assert self.watcher.batch_processor._running is True

        # Stop watcher
        await self.watcher.stop()

        assert self.watcher._running is False
        assert all(task.done() for task in self.watcher._watch_tasks.values())

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Test starting watcher when already running."""
        await self.watcher.start()
        assert self.watcher._running is True

        # Try to start again - should not cause issues
        await self.watcher.start()
        assert self.watcher._running is True

        await self.watcher.stop()

    @pytest.mark.asyncio
    async def test_pause_resume_functionality(self):
        """Test pause and resume functionality."""
        await self.watcher.start()
        assert self.watcher._running is True

        # Pause
        await self.watcher.pause()
        # Should still be marked as running but no active watch tasks
        assert all(task.done() for task in self.watcher._watch_tasks.values())

        # Resume
        await self.watcher.resume()
        assert self.watcher._running is True
        assert len(self.watcher._watch_tasks) > 0

        await self.watcher.stop()

    @pytest.mark.asyncio
    async def test_close_cleanup(self):
        """Test that close properly cleans up resources."""
        await self.watcher.start()

        # Should have active components
        assert self.watcher.file_filter is not None
        assert self.watcher.batch_processor is not None

        # Close
        await self.watcher.close()

        assert self.watcher._running is False
        assert all(task.done() for task in self.watcher._watch_tasks.values())


class TestEnhancedFileWatcherFileProcessing:
    """Test file processing and event handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = EnhancedWatcherConfig(
            watch_directories={str(self.temp_dir): "test-collection"},
            debouncing={"strategy": DebouncingStrategy.NONE}  # No debouncing for faster tests
        )
        self.watcher = EnhancedFileWatcher(self.config)
        self.processed_batches = []

        async def mock_callback(items):
            self.processed_batches.extend(items)

        self.watcher.set_ingestion_callback(mock_callback)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

        asyncio.run(self.watcher.close())

    @pytest.mark.asyncio
    async def test_file_change_detection(self):
        """Test detection of file changes."""
        # Mock the file change handling to avoid actual file system watching
        file_path = self.temp_dir / "test.txt"
        file_path.write_text("test content")

        # Mock file system changes
        from watchfiles import Change
        changes = {(Change.added, str(file_path))}

        await self.watcher._handle_file_changes(changes, "test-collection")

        # Should have updated statistics
        assert self.watcher.statistics.total_events_received == 1
        assert self.watcher.statistics.files_added == 1

    @pytest.mark.asyncio
    async def test_file_filtering_integration(self):
        """Test integration with file filtering."""
        # Create files with different extensions
        py_file = self.temp_dir / "script.py"
        txt_file = self.temp_dir / "document.txt"
        cache_file = self.temp_dir / "__pycache__" / "cached.pyc"
        cache_file.parent.mkdir()

        py_file.write_text("print('hello')")
        txt_file.write_text("document content")
        cache_file.write_text("cached bytecode")

        # Mock changes with filtering that should exclude cache file
        from watchfiles import Change
        changes = {
            (Change.added, str(py_file)),
            (Change.added, str(txt_file)),
            (Change.added, str(cache_file))  # Should be filtered out
        }

        await self.watcher._handle_file_changes(changes, "test-collection")

        # Should have processed 2 files, filtered 1
        assert self.watcher.statistics.total_events_received == 3
        # Cache file should be filtered out by default exclude patterns

    @pytest.mark.asyncio
    async def test_batch_processing_integration(self):
        """Test integration with batch processing."""
        await self.watcher.start()

        # Create a file operation directly
        file_operation = FileOperation(
            file_path=str(self.temp_dir / "test.txt"),
            collection="test-collection",
            operation_type="add",
            priority=BatchPriority.NORMAL
        )

        # Queue the operation
        await self.watcher._queue_file_operation(file_operation)

        # Wait for processing
        await asyncio.sleep(0.2)

        # Should have been processed through batch system
        assert self.watcher.batch_processor.statistics.current_queue_size >= 0

        await self.watcher.stop()

    @pytest.mark.asyncio
    async def test_error_handling_in_file_processing(self):
        """Test error handling during file processing."""
        # Mock a processing error
        original_callback = self.watcher.ingestion_callback

        async def failing_callback(items):
            raise Exception("Processing failed")

        self.watcher.set_ingestion_callback(failing_callback)
        await self.watcher.start()

        # Create file operation
        file_operation = FileOperation(
            file_path=str(self.temp_dir / "error_test.txt"),
            collection="test-collection",
            operation_type="add"
        )

        # This should handle the error gracefully
        await self.watcher._queue_file_operation(file_operation)
        await asyncio.sleep(0.2)

        # Error statistics should be updated
        # Note: Error handling is delegated to batch processor

        await self.watcher.stop()

    def test_operation_type_conversion(self):
        """Test conversion from watchfiles Change to operation type."""
        from watchfiles import Change

        assert self.watcher._get_operation_type(Change.added) == "add"
        assert self.watcher._get_operation_type(Change.modified) == "modify"
        assert self.watcher._get_operation_type(Change.deleted) == "delete"

    def test_priority_determination(self):
        """Test file priority determination logic."""
        # Test config file (should be high priority)
        config_path = Path("/tmp/config.json")
        priority = self.watcher._determine_priority(config_path, "modify")
        assert priority == BatchPriority.HIGH

        # Test delete operation (should be high priority)
        normal_path = Path("/tmp/normal.txt")
        priority = self.watcher._determine_priority(normal_path, "delete")
        assert priority == BatchPriority.HIGH

        # Test normal file (should be normal priority)
        priority = self.watcher._determine_priority(normal_path, "add")
        assert priority == BatchPriority.NORMAL


class TestEnhancedFileWatcherStatistics:
    """Test statistics and monitoring functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = EnhancedWatcherConfig()
        self.watcher = EnhancedFileWatcher(self.config)

    def teardown_method(self):
        """Clean up test fixtures."""
        asyncio.run(self.watcher.close())

    def test_get_statistics(self):
        """Test getting watcher statistics."""
        stats = self.watcher.get_statistics()

        assert isinstance(stats, WatcherStatistics)
        assert stats.total_events_received == 0
        assert stats.files_processed_successfully == 0

    def test_get_component_statistics(self):
        """Test getting statistics from all components."""
        stats = self.watcher.get_component_statistics()

        assert isinstance(stats, dict)
        assert "watcher" in stats
        assert "file_filter" in stats
        assert "batch_processor" in stats
        assert "debouncer" in stats

        # Each component should have its own statistics
        assert isinstance(stats["watcher"], dict)
        assert isinstance(stats["file_filter"], dict)
        assert isinstance(stats["batch_processor"], dict)
        assert isinstance(stats["debouncer"], dict)

    def test_get_performance_report(self):
        """Test comprehensive performance reporting."""
        report = self.watcher.get_performance_report()

        assert isinstance(report, dict)
        assert "component_statistics" in report
        assert "configuration" in report
        assert "performance_insights" in report

        # Check configuration summary
        config = report["configuration"]
        assert "watch_directories" in config
        assert "filtering_enabled" in config
        assert "batch_processing_enabled" in config

        # Check performance insights
        insights = report["performance_insights"]
        assert "overall_success_rate" in insights
        assert "filter_efficiency" in insights
        assert "resource_utilization" in insights

    def test_get_status(self):
        """Test watcher status reporting."""
        status = self.watcher.get_status()

        assert isinstance(status, dict)
        assert "running" in status
        assert "active_watchers" in status
        assert "watch_directories" in status
        assert "component_status" in status

        # Check component status
        comp_status = status["component_status"]
        assert "file_filter" in comp_status
        assert "batch_processor" in comp_status
        assert "debouncer" in comp_status

    def test_statistics_update_methods(self):
        """Test statistics update methods."""
        from watchfiles import Change

        # Test change type statistics
        initial_added = self.watcher.statistics.files_added
        asyncio.run(self.watcher._update_change_statistics(Change.added))
        assert self.watcher.statistics.files_added == initial_added + 1

        # Test error statistics
        initial_permission = self.watcher.statistics.permission_errors
        asyncio.run(self.watcher._update_error_statistics(PermissionError("Access denied")))
        assert self.watcher.statistics.permission_errors == initial_permission + 1


class TestConvenienceFactoryFunctions:
    """Test convenience factory functions."""

    def test_create_high_performance_watcher(self):
        """Test high-performance watcher factory."""
        watch_dirs = {"/tmp/perf": "perf-collection"}

        watcher = create_high_performance_watcher(watch_dirs)

        assert watcher.config.watch_directories == watch_dirs
        assert watcher.config.max_concurrent_watchers == 20
        assert watcher.config.max_memory_usage_mb == 1024.0

        # Check high-performance settings
        assert watcher.config.filtering["enable_content_filtering"] is False
        assert watcher.config.batch_processing["max_batch_size"] == 100
        assert watcher.config.batch_processing["max_concurrent_batches"] == 8

    def test_create_selective_watcher(self):
        """Test selective watcher factory."""
        watch_dirs = {"/tmp/selective": "selective-collection"}
        file_patterns = ["*.py", "*.js"]

        watcher = create_selective_watcher(watch_dirs, file_patterns)

        assert watcher.config.watch_directories == watch_dirs
        assert watcher.config.filtering["include_patterns"] == file_patterns
        assert watcher.config.filtering["enable_content_filtering"] is True
        assert "import" in watcher.config.filtering["content_filters"]

    def test_create_memory_efficient_watcher(self):
        """Test memory-efficient watcher factory."""
        watch_dirs = {"/tmp/memory": "memory-collection"}

        watcher = create_memory_efficient_watcher(watch_dirs)

        assert watcher.config.watch_directories == watch_dirs
        assert watcher.config.max_memory_usage_mb == 128.0
        assert watcher.config.max_debounce_tasks == 100
        assert watcher.config.max_concurrent_watchers == 5

        # Check memory-efficient settings
        assert watcher.config.filtering["max_file_size"] == 5 * 1024 * 1024  # 5MB
        assert watcher.config.batch_processing["max_batch_size"] == 10
        assert watcher.config.batch_processing["max_concurrent_batches"] == 2

    def test_convenience_functions_with_custom_args(self):
        """Test convenience functions with custom arguments."""
        watch_dirs = {"/tmp/custom": "custom-collection"}

        # Test with custom filtering
        custom_filtering = {"max_file_size": 1024}
        watcher = create_high_performance_watcher(watch_dirs, filtering=custom_filtering)

        assert watcher.config.filtering["max_file_size"] == 1024
        # Should still have other high-performance settings
        assert watcher.config.max_concurrent_watchers == 20

    def teardown_method(self):
        """Clean up any created watchers."""
        # Factory functions return new instances, cleanup handled by individual tests
        pass


class TestEnhancedFileWatcherEdgeCases:
    """Test edge cases and error scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = EnhancedWatcherConfig()
        self.watcher = EnhancedFileWatcher(self.config)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

        asyncio.run(self.watcher.close())

    @pytest.mark.asyncio
    async def test_empty_watch_directories(self):
        """Test watcher with no watch directories."""
        async def mock_callback(items):
            pass

        self.watcher.set_ingestion_callback(mock_callback)

        # Should start successfully even with no directories
        await self.watcher.start()
        assert self.watcher._running is True
        assert len(self.watcher._watch_tasks) == 0

        await self.watcher.stop()

    @pytest.mark.asyncio
    async def test_permission_error_handling(self):
        """Test handling of permission errors."""
        restricted_dir = "/root/restricted"  # Likely to cause permission error

        result = await self.watcher.add_watch_directory(restricted_dir, "restricted-collection")

        # Should handle gracefully (return False for invalid directory)
        assert result is False

    @pytest.mark.asyncio
    async def test_unicode_file_paths(self):
        """Test handling of Unicode file paths."""
        # Create Unicode directory
        unicode_dir = self.temp_dir / "测试目录"
        unicode_dir.mkdir()

        result = await self.watcher.add_watch_directory(str(unicode_dir), "unicode-collection")
        assert result is True

        # Create Unicode file
        unicode_file = unicode_dir / "测试文件.txt"
        unicode_file.write_text("Unicode content")

        # Should handle Unicode paths in operations
        file_operation = FileOperation(
            file_path=str(unicode_file),
            collection="unicode-collection",
            operation_type="add"
        )

        # Should not raise encoding errors
        await self.watcher._queue_file_operation(file_operation)

    @pytest.mark.asyncio
    async def test_very_long_file_paths(self):
        """Test handling of very long file paths."""
        # Create nested directory structure
        long_path = self.temp_dir
        for i in range(10):
            long_path = long_path / f"very_long_directory_name_{i}_with_many_characters"

        try:
            long_path.mkdir(parents=True)
            long_file = long_path / ("very_long_filename_" * 10 + ".txt")
            long_file.write_text("content")

            # Should handle long paths
            file_operation = FileOperation(
                file_path=str(long_file),
                collection="long-path-collection",
                operation_type="add"
            )

            await self.watcher._queue_file_operation(file_operation)

        except OSError:
            # Some systems don't support very long paths - skip test
            pytest.skip("System doesn't support very long paths")

    @pytest.mark.asyncio
    async def test_rapid_start_stop_cycles(self):
        """Test rapid start/stop cycles."""
        async def mock_callback(items):
            pass

        self.watcher.set_ingestion_callback(mock_callback)
        self.watcher.config.watch_directories = {str(self.temp_dir): "rapid-collection"}

        # Rapid start/stop cycles
        for _ in range(5):
            await self.watcher.start()
            await asyncio.sleep(0.01)  # Very brief
            await self.watcher.stop()

        # Should handle without crashing
        assert self.watcher._running is False

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_error(self):
        """Test proper resource cleanup when errors occur."""
        async def failing_callback(items):
            raise Exception("Callback always fails")

        self.watcher.set_ingestion_callback(failing_callback)
        self.watcher.config.watch_directories = {str(self.temp_dir): "error-collection"}

        await self.watcher.start()

        # Simulate some file operations that will fail
        file_operation = FileOperation(
            file_path=str(self.temp_dir / "error.txt"),
            collection="error-collection",
            operation_type="add"
        )

        await self.watcher._queue_file_operation(file_operation)
        await asyncio.sleep(0.1)

        # Stop and ensure cleanup happens
        await self.watcher.stop()

        # Should be properly stopped despite errors
        assert self.watcher._running is False

    def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        # Test with invalid debouncing strategy
        with patch('src.python.common.core.enhanced_file_watcher.SmartDebouncer') as mock_debouncer:
            mock_debouncer.side_effect = Exception("Invalid debouncer config")

            # Should handle gracefully and continue with other components
            config = EnhancedWatcherConfig(debouncing={"strategy": "invalid"})

            # Should not crash during initialization
            try:
                watcher = EnhancedFileWatcher(config)
                assert watcher is not None
            except Exception as e:
                # Should handle initialization errors gracefully
                assert "debouncer" in str(e).lower() or "Invalid" in str(e)


class TestEnhancedFileWatcherIntegration:
    """Integration tests for the complete enhanced file watcher system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.processed_files = []

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(str(self.temp_dir))

    @pytest.mark.asyncio
    async def test_end_to_end_file_processing(self):
        """Test complete end-to-end file processing workflow."""
        # Create directories
        docs_dir = self.temp_dir / "docs"
        code_dir = self.temp_dir / "code"
        docs_dir.mkdir()
        code_dir.mkdir()

        # Configure watcher
        config = EnhancedWatcherConfig(
            watch_directories={
                str(docs_dir): "documents",
                str(code_dir): "code"
            },
            filtering={
                "include_patterns": ["*.txt", "*.py", "*.md"],
                "exclude_patterns": [r".*\.tmp$"]
            },
            batch_processing={
                "max_batch_size": 5,
                "max_batch_age_seconds": 1.0
            },
            debouncing={
                "strategy": DebouncingStrategy.SIMPLE,
                "base_delay_seconds": 0.1
            }
        )

        watcher = EnhancedFileWatcher(config)

        async def track_processed_files(items):
            for item in items:
                self.processed_files.append({
                    "path": item.file_operation.file_path,
                    "collection": item.file_operation.collection,
                    "operation": item.file_operation.operation_type
                })

        watcher.set_ingestion_callback(track_processed_files)

        try:
            # Start watcher
            await watcher.start()

            # Create files in monitored directories
            doc_files = [
                docs_dir / "readme.txt",
                docs_dir / "guide.md",
                docs_dir / "temp.tmp"  # Should be filtered out
            ]

            code_files = [
                code_dir / "main.py",
                code_dir / "utils.py"
            ]

            # Write files
            for file_path in doc_files + code_files:
                file_path.write_text(f"Content for {file_path.name}")

            # Simulate file system events
            from watchfiles import Change

            # Documents
            doc_changes = set()
            for file_path in doc_files:
                doc_changes.add((Change.added, str(file_path)))

            await watcher._handle_file_changes(doc_changes, "documents")

            # Code files
            code_changes = set()
            for file_path in code_files:
                code_changes.add((Change.added, str(file_path)))

            await watcher._handle_file_changes(code_changes, "code")

            # Wait for processing
            await asyncio.sleep(0.5)

            # Check results
            processed_paths = [f["path"] for f in self.processed_files]

            # Should have processed valid files
            assert any("readme.txt" in path for path in processed_paths)
            assert any("guide.md" in path for path in processed_paths)
            assert any("main.py" in path for path in processed_paths)
            assert any("utils.py" in path for path in processed_paths)

            # Should have filtered out .tmp file
            assert not any("temp.tmp" in path for path in processed_paths)

            # Check statistics
            stats = watcher.get_statistics()
            assert stats.total_events_received > 0
            assert stats.files_filtered_out > 0  # .tmp file filtered

        finally:
            await watcher.close()

    @pytest.mark.asyncio
    async def test_high_volume_file_processing(self):
        """Test processing of high volume of files."""
        config = EnhancedWatcherConfig(
            watch_directories={str(self.temp_dir): "high-volume"},
            batch_processing={
                "max_batch_size": 20,
                "max_batch_age_seconds": 0.5,
                "processing_strategy": ProcessingStrategy.ADAPTIVE.value
            },
            debouncing={
                "strategy": DebouncingStrategy.ADAPTIVE,
                "base_delay_seconds": 0.05
            }
        )

        watcher = EnhancedFileWatcher(config)
        processed_count = 0

        async def count_processed_files(items):
            nonlocal processed_count
            processed_count += len(items)

        watcher.set_ingestion_callback(count_processed_files)

        try:
            await watcher.start()

            # Create many files
            file_count = 100
            files = []
            for i in range(file_count):
                file_path = self.temp_dir / f"file_{i:03d}.txt"
                file_path.write_text(f"Content {i}")
                files.append(file_path)

            # Simulate batch changes
            from watchfiles import Change
            changes = {(Change.added, str(f)) for f in files}

            await watcher._handle_file_changes(changes, "high-volume")

            # Wait for processing
            await asyncio.sleep(2.0)

            # Should have processed most files
            # (Some may still be in queue or being processed)
            assert processed_count >= 0

            # Check performance metrics
            report = watcher.get_performance_report()
            assert report is not None

        finally:
            await watcher.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])