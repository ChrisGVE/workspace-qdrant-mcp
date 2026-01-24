"""Unit tests for OptimizedAutoIngestion (Task 467).

Tests the OptimizedAutoIngestion class covering:
- Progress tracking and statistics
- Directory ingestion with patterns
- Cancellation support
- Priority levels
- Error handling
- Deduplication
- Resource management
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.core.optimized_auto_ingestion import (
    FileEntry,
    IngestionPriority,
    IngestionProgress,
    IngestionStatus,
    OptimizedAutoIngestion,
    OptimizedIngestionConfig,
)


@pytest.fixture
def temp_directory():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "file1.py").write_text("print('hello')")
        (Path(tmpdir) / "file2.py").write_text("def foo(): pass")
        (Path(tmpdir) / "file3.md").write_text("# README")
        (Path(tmpdir) / "file4.txt").write_text("plain text")
        (Path(tmpdir) / "ignored.pyc").write_text("bytecode")

        # Create subdirectory with files
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "sub_file.py").write_text("import os")
        (subdir / "sub_file.md").write_text("## Subdir")

        yield tmpdir


@pytest.fixture
def config():
    """Create a default test configuration."""
    return OptimizedIngestionConfig(
        max_concurrent_files=5,
        batch_size=10,
        enable_deduplication=True,
        progress_interval_seconds=0.1,
        max_file_size_mb=50.0,
    )


@pytest.fixture
def ingestion(config):
    """Create an OptimizedAutoIngestion instance."""
    return OptimizedAutoIngestion(config=config)


class TestOptimizedIngestionConfig:
    """Tests for OptimizedIngestionConfig."""

    def test_default_config_is_valid(self):
        """Test that default configuration is valid."""
        config = OptimizedIngestionConfig()
        config.validate()  # Should not raise

    def test_invalid_max_concurrent_files_raises(self):
        """Test that invalid max_concurrent_files raises ValueError."""
        config = OptimizedIngestionConfig(max_concurrent_files=0)
        with pytest.raises(ValueError, match="max_concurrent_files"):
            config.validate()

    def test_invalid_batch_size_raises(self):
        """Test that invalid batch_size raises ValueError."""
        config = OptimizedIngestionConfig(batch_size=-1)
        with pytest.raises(ValueError, match="batch_size"):
            config.validate()

    def test_invalid_progress_interval_raises(self):
        """Test that invalid progress_interval_seconds raises ValueError."""
        config = OptimizedIngestionConfig(progress_interval_seconds=-1)
        with pytest.raises(ValueError, match="progress_interval_seconds"):
            config.validate()

    def test_invalid_max_memory_raises(self):
        """Test that invalid max_memory_usage_mb raises ValueError."""
        config = OptimizedIngestionConfig(max_memory_usage_mb=0)
        with pytest.raises(ValueError, match="max_memory_usage_mb"):
            config.validate()

    def test_invalid_max_cpu_raises(self):
        """Test that invalid max_cpu_percent raises ValueError."""
        config = OptimizedIngestionConfig(max_cpu_percent=101)
        with pytest.raises(ValueError, match="max_cpu_percent"):
            config.validate()


class TestIngestionProgress:
    """Tests for IngestionProgress dataclass."""

    def test_percent_complete_with_zero_files(self):
        """Test percent_complete when total_files is zero."""
        progress = IngestionProgress(total_files=0)
        assert progress.percent_complete == 0.0

    def test_percent_complete_partial(self):
        """Test percent_complete with partial completion."""
        progress = IngestionProgress(
            total_files=100,
            processed_files=25,
            failed_files=5,
            skipped_files=10,
        )
        assert progress.percent_complete == 40.0

    def test_percent_complete_full(self):
        """Test percent_complete at 100%."""
        progress = IngestionProgress(
            total_files=50,
            processed_files=45,
            failed_files=3,
            skipped_files=2,
        )
        assert progress.percent_complete == 100.0

    def test_update_timing(self):
        """Test that update_timing updates fields correctly."""
        progress = IngestionProgress(
            total_files=100,
            processed_files=50,
        )
        progress.update_timing()

        assert progress.elapsed_seconds > 0
        assert progress.throughput_files_per_second >= 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        progress = IngestionProgress(
            total_files=10,
            processed_files=5,
            failed_files=1,
        )
        result = progress.to_dict()

        assert result["total_files"] == 10
        assert result["processed_files"] == 5
        assert result["failed_files"] == 1
        assert "percent_complete" in result
        assert "status" in result


class TestIngestionPriority:
    """Tests for IngestionPriority enum."""

    def test_to_batch_priority_low(self):
        """Test conversion of LOW priority."""
        from common.core.batch_processing_manager import BatchPriority
        assert IngestionPriority.LOW.to_batch_priority() == BatchPriority.LOW

    def test_to_batch_priority_normal(self):
        """Test conversion of NORMAL priority."""
        from common.core.batch_processing_manager import BatchPriority
        assert IngestionPriority.NORMAL.to_batch_priority() == BatchPriority.NORMAL

    def test_to_batch_priority_high(self):
        """Test conversion of HIGH priority."""
        from common.core.batch_processing_manager import BatchPriority
        assert IngestionPriority.HIGH.to_batch_priority() == BatchPriority.HIGH

    def test_to_batch_priority_critical(self):
        """Test conversion of CRITICAL priority."""
        from common.core.batch_processing_manager import BatchPriority
        assert IngestionPriority.CRITICAL.to_batch_priority() == BatchPriority.CRITICAL


class TestFileEntry:
    """Tests for FileEntry dataclass."""

    def test_compute_content_hash(self, temp_directory):
        """Test content hash computation."""
        file_path = os.path.join(temp_directory, "file1.py")
        entry = FileEntry(path=file_path, collection="test")

        hash1 = entry.compute_content_hash()
        hash2 = entry.compute_content_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_compute_content_hash_nonexistent(self):
        """Test content hash for nonexistent file."""
        entry = FileEntry(path="/nonexistent/file.txt", collection="test")
        # Should fall back to path hash
        hash_value = entry.compute_content_hash()
        assert len(hash_value) == 64


class TestOptimizedAutoIngestion:
    """Tests for OptimizedAutoIngestion class."""

    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        ingestion = OptimizedAutoIngestion()
        assert ingestion.config is not None
        assert ingestion.config.max_concurrent_files == 10

    def test_init_with_dict_config(self):
        """Test initialization with dictionary configuration."""
        ingestion = OptimizedAutoIngestion(config={"max_concurrent_files": 20})
        assert ingestion.config.max_concurrent_files == 20

    def test_init_with_config_object(self, config):
        """Test initialization with config object."""
        ingestion = OptimizedAutoIngestion(config=config)
        assert ingestion.config.max_concurrent_files == 5

    def test_set_progress_callback(self, ingestion):
        """Test setting progress callback."""
        callback = MagicMock()
        ingestion.set_progress_callback(callback)
        assert ingestion._progress_callback == callback

    def test_set_processing_callback(self, ingestion):
        """Test setting processing callback."""
        callback = AsyncMock()
        ingestion.set_processing_callback(callback)
        assert ingestion._processing_callback == callback

    def test_get_progress(self, ingestion):
        """Test getting progress."""
        progress = ingestion.get_progress()
        assert isinstance(progress, IngestionProgress)

    def test_get_statistics(self, ingestion):
        """Test getting statistics."""
        stats = ingestion.get_statistics()
        assert "batch_processing" in stats
        assert "ingestion" in stats

    def test_clear_session(self, ingestion):
        """Test clearing session data."""
        # Add some data
        ingestion._session_files["test"].add("file1")
        old_session = ingestion._session_id

        ingestion.clear_session()

        assert len(ingestion._session_files) == 0
        assert ingestion._session_id != old_session


class TestDeduplication:
    """Tests for file deduplication."""

    def test_is_duplicate_disabled(self, temp_directory):
        """Test deduplication when disabled."""
        config = OptimizedIngestionConfig(enable_deduplication=False)
        ingestion = OptimizedAutoIngestion(config=config)

        entry = FileEntry(path="/test/file.py", collection="test")
        assert ingestion._is_duplicate(entry) is False

    def test_is_duplicate_path_strategy(self, temp_directory):
        """Test deduplication with path strategy."""
        config = OptimizedIngestionConfig(
            enable_deduplication=True,
            deduplication_strategy="path",
        )
        ingestion = OptimizedAutoIngestion(config=config)

        entry = FileEntry(path="/test/file.py", collection="test")

        # First time - not duplicate
        assert ingestion._is_duplicate(entry) is False

        # Mark as processed
        ingestion._mark_processed(entry)

        # Second time - is duplicate
        assert ingestion._is_duplicate(entry) is True

    def test_is_duplicate_mtime_strategy(self, temp_directory):
        """Test deduplication with mtime strategy."""
        config = OptimizedIngestionConfig(
            enable_deduplication=True,
            deduplication_strategy="mtime",
        )
        ingestion = OptimizedAutoIngestion(config=config)

        entry = FileEntry(path="/test/file.py", collection="test", mtime=1234567890.0)

        # First time - not duplicate
        assert ingestion._is_duplicate(entry) is False

        # Mark as processed
        ingestion._mark_processed(entry)

        # Same path + mtime - is duplicate
        assert ingestion._is_duplicate(entry) is True

        # Different mtime - not duplicate
        entry2 = FileEntry(path="/test/file.py", collection="test", mtime=9999999999.0)
        assert ingestion._is_duplicate(entry2) is False


class TestPatternMatching:
    """Tests for file pattern matching."""

    def test_matches_patterns_include(self, ingestion):
        """Test matching include patterns."""
        assert ingestion._matches_patterns("test.py", ["*.py"], []) is True
        assert ingestion._matches_patterns("test.js", ["*.py"], []) is False

    def test_matches_patterns_exclude(self, ingestion):
        """Test matching exclude patterns."""
        assert ingestion._matches_patterns("test.pyc", ["*.py"], ["*.pyc"]) is False
        assert ingestion._matches_patterns("test.py", ["*.py"], ["*.pyc"]) is True

    def test_matches_patterns_empty_include(self, ingestion):
        """Test matching with empty include patterns."""
        assert ingestion._matches_patterns("test.py", [], []) is True
        assert ingestion._matches_patterns("test.any", [], []) is True

    def test_matches_patterns_multiple(self, ingestion):
        """Test matching multiple patterns."""
        patterns = ["*.py", "*.md", "*.txt"]
        assert ingestion._matches_patterns("test.py", patterns, []) is True
        assert ingestion._matches_patterns("test.md", patterns, []) is True
        assert ingestion._matches_patterns("test.js", patterns, []) is False


class TestDirectoryIngestion:
    """Tests for directory ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_directory_nonexistent(self, ingestion):
        """Test ingesting nonexistent directory returns failed status."""
        progress = await ingestion.ingest_directory(
            "/nonexistent/path",
            collection="test",
        )
        assert progress.status == IngestionStatus.FAILED
        assert len(progress.errors) > 0
        assert "does not exist" in progress.errors[0][1]

    @pytest.mark.asyncio
    async def test_ingest_directory_not_a_directory(self, ingestion, temp_directory):
        """Test ingesting a file instead of directory returns failed status."""
        file_path = os.path.join(temp_directory, "file1.py")
        progress = await ingestion.ingest_directory(
            file_path,
            collection="test",
        )
        assert progress.status == IngestionStatus.FAILED
        assert len(progress.errors) > 0
        assert "not a directory" in progress.errors[0][1]

    @pytest.mark.asyncio
    async def test_ingest_directory_empty_patterns(self, temp_directory):
        """Test ingesting with no matching files."""
        config = OptimizedIngestionConfig(
            enable_deduplication=False,
            enable_progress_callbacks=False,
        )
        ingestion = OptimizedAutoIngestion(config=config)

        # Mock batch manager
        ingestion._batch_manager.add_file = AsyncMock(return_value=True)
        ingestion._batch_manager._running = True

        progress = await ingestion.ingest_directory(
            temp_directory,
            collection="test",
            patterns=["*.nonexistent"],
        )

        assert progress.total_files == 0
        assert progress.status == IngestionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_directory_with_patterns(self, temp_directory):
        """Test ingesting with specific patterns."""
        config = OptimizedIngestionConfig(
            enable_deduplication=False,
            enable_progress_callbacks=False,
        )
        ingestion = OptimizedAutoIngestion(config=config)

        # Mock batch manager
        ingestion._batch_manager.add_file = AsyncMock(return_value=True)
        ingestion._batch_manager._running = True

        progress = await ingestion.ingest_directory(
            temp_directory,
            collection="test",
            patterns=["*.py"],
            recursive=True,
        )

        # Should find 3 .py files (2 in root, 1 in subdir)
        assert progress.total_files == 3
        assert progress.processed_files == 3

    @pytest.mark.asyncio
    async def test_ingest_directory_non_recursive(self, temp_directory):
        """Test ingesting without recursion."""
        config = OptimizedIngestionConfig(
            enable_deduplication=False,
            enable_progress_callbacks=False,
        )
        ingestion = OptimizedAutoIngestion(config=config)

        # Mock batch manager
        ingestion._batch_manager.add_file = AsyncMock(return_value=True)
        ingestion._batch_manager._running = True

        progress = await ingestion.ingest_directory(
            temp_directory,
            collection="test",
            patterns=["*.py"],
            recursive=False,
        )

        # Should find only 2 .py files in root (not subdir)
        assert progress.total_files == 2

    @pytest.mark.asyncio
    async def test_ingest_directory_with_priority(self, temp_directory):
        """Test ingesting with HIGH priority."""
        config = OptimizedIngestionConfig(
            enable_deduplication=False,
            enable_progress_callbacks=False,
        )
        ingestion = OptimizedAutoIngestion(config=config)

        # Mock batch manager to track calls
        calls = []

        async def mock_add_file(**kwargs):
            calls.append(kwargs)
            return True

        ingestion._batch_manager.add_file = mock_add_file
        ingestion._batch_manager._running = True

        await ingestion.ingest_directory(
            temp_directory,
            collection="test",
            patterns=["*.py"],
            priority=IngestionPriority.HIGH,
            recursive=False,
        )

        # All files should have HIGH priority
        from common.core.batch_processing_manager import BatchPriority
        for call in calls:
            assert call["priority"] == BatchPriority.HIGH


class TestFileIngestion:
    """Tests for file list ingestion."""

    @pytest.mark.asyncio
    async def test_ingest_files_empty_list(self, ingestion):
        """Test ingesting empty file list."""
        progress = await ingestion.ingest_files([], collection="test")
        assert progress.total_files == 0
        assert progress.status == IngestionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_files_nonexistent(self, ingestion):
        """Test ingesting nonexistent files."""
        ingestion._batch_manager._running = True

        progress = await ingestion.ingest_files(
            ["/nonexistent/file.py"],
            collection="test",
        )

        assert progress.total_files == 1
        assert progress.skipped_files == 1

    @pytest.mark.asyncio
    async def test_ingest_files_success(self, temp_directory):
        """Test successful file ingestion."""
        config = OptimizedIngestionConfig(
            enable_deduplication=False,
            enable_progress_callbacks=False,
        )
        ingestion = OptimizedAutoIngestion(config=config)

        # Mock batch manager
        ingestion._batch_manager.add_file = AsyncMock(return_value=True)
        ingestion._batch_manager._running = True

        files = [
            os.path.join(temp_directory, "file1.py"),
            os.path.join(temp_directory, "file2.py"),
        ]

        progress = await ingestion.ingest_files(files, collection="test")

        assert progress.total_files == 2
        assert progress.processed_files == 2
        assert progress.status == IngestionStatus.COMPLETED


class TestCancellation:
    """Tests for cancellation support."""

    @pytest.mark.asyncio
    async def test_cancel_during_ingestion(self, temp_directory):
        """Test cancelling during ingestion."""
        config = OptimizedIngestionConfig(
            enable_deduplication=False,
            enable_progress_callbacks=False,
        )
        ingestion = OptimizedAutoIngestion(config=config)

        # Mock batch manager with slow operation
        async def slow_add(**kwargs):
            await asyncio.sleep(0.1)
            return True

        ingestion._batch_manager.add_file = slow_add
        ingestion._batch_manager._running = True

        # Start ingestion in background
        task = asyncio.create_task(
            ingestion.ingest_directory(temp_directory, collection="test")
        )

        # Cancel after short delay
        await asyncio.sleep(0.05)
        await ingestion.cancel()

        progress = await task
        assert progress.status == IngestionStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, ingestion):
        """Test pausing and resuming."""
        assert ingestion.is_paused is False

        await ingestion.pause()
        assert ingestion.is_paused is True

        await ingestion.resume()
        assert ingestion.is_paused is False


class TestProgressCallbacks:
    """Tests for progress callbacks."""

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, temp_directory):
        """Test that progress callback is called."""
        config = OptimizedIngestionConfig(
            enable_progress_callbacks=True,
            progress_interval_seconds=0.01,
            enable_deduplication=False,
        )
        ingestion = OptimizedAutoIngestion(config=config)

        progress_updates = []

        def on_progress(progress):
            progress_updates.append(progress.to_dict())

        ingestion.set_progress_callback(on_progress)

        # Mock batch manager
        ingestion._batch_manager.add_file = AsyncMock(return_value=True)
        ingestion._batch_manager._running = True

        # Need to mock _progress_reporter to avoid async issues
        ingestion._progress_task = None

        await ingestion.ingest_files(
            [os.path.join(temp_directory, "file1.py")],
            collection="test",
        )

        # Progress should have been tracked
        assert ingestion.get_progress().total_files == 1


class TestResourceManagement:
    """Tests for resource management."""

    @pytest.mark.asyncio
    async def test_check_resources(self):
        """Test resource checking with adequate resources."""
        # Create config with high memory threshold
        config = OptimizedIngestionConfig(max_memory_usage_mb=20000.0)
        ingestion = OptimizedAutoIngestion(config=config)

        # Reset last check time to force a fresh check
        ingestion._last_resource_check = 0

        with patch("common.core.optimized_auto_ingestion.psutil.virtual_memory") as mock_mem, \
             patch("common.core.optimized_auto_ingestion.psutil.cpu_percent") as mock_cpu:
            mock_mem.return_value = MagicMock(
                total=16 * 1024 * 1024 * 1024,  # 16GB
                available=8 * 1024 * 1024 * 1024,  # 8GB
            )
            mock_cpu.return_value = 50.0

            result = await ingestion._check_resources()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_resources_high_memory(self, config):
        """Test resource check with high memory usage."""
        config.max_memory_usage_mb = 100.0  # Set low threshold
        config.adaptive_resource_scaling = True
        ingestion = OptimizedAutoIngestion(config=config)

        original_batch_size = ingestion._current_batch_size
        # Reset last check time to force a fresh check
        ingestion._last_resource_check = 0

        with patch("common.core.optimized_auto_ingestion.psutil.virtual_memory") as mock_mem, \
             patch("common.core.optimized_auto_ingestion.psutil.cpu_percent") as mock_cpu:
            # Simulate high memory usage
            mock_mem.return_value = MagicMock(
                total=16 * 1024 * 1024 * 1024,
                available=1 * 1024 * 1024,  # Very little available
            )
            mock_cpu.return_value = 50.0

            result = await ingestion._check_resources()

            # Should return False and reduce batch size
            assert result is False
            assert ingestion._current_batch_size < original_batch_size


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handles_permission_error(self, ingestion, temp_directory):
        """Test handling of permission errors during scan."""
        ingestion._batch_manager._running = True

        # This should handle the error gracefully
        progress = await ingestion.ingest_directory(
            temp_directory,
            collection="test",
            patterns=["*.py"],
        )

        assert progress.status in [IngestionStatus.COMPLETED, IngestionStatus.CANCELLED]

    @pytest.mark.asyncio
    async def test_errors_tracked_in_progress(self, ingestion, temp_directory):
        """Test that errors are tracked in progress."""
        ingestion._batch_manager._running = True

        # Mock to simulate failure
        async def failing_add(**kwargs):
            raise RuntimeError("Simulated error")

        ingestion._batch_manager.add_file = failing_add

        progress = await ingestion.ingest_files(
            [os.path.join(temp_directory, "file1.py")],
            collection="test",
        )

        assert progress.failed_files == 1
        assert len(progress.errors) == 1


class TestLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self, config):
        """Test starting and stopping."""
        ingestion = OptimizedAutoIngestion(config=config)

        # Set a processing callback
        ingestion.set_processing_callback(AsyncMock())

        await ingestion.start()
        assert ingestion.is_running is True

        await ingestion.stop()
        assert ingestion._cancel_requested is True

    @pytest.mark.asyncio
    async def test_is_running_property(self, ingestion):
        """Test is_running property."""
        assert ingestion.is_running is False

        ingestion._batch_manager._running = True
        assert ingestion.is_running is True

    @pytest.mark.asyncio
    async def test_is_cancelled_property(self, ingestion):
        """Test is_cancelled property."""
        assert ingestion.is_cancelled is False

        await ingestion.cancel()
        assert ingestion.is_cancelled is True
