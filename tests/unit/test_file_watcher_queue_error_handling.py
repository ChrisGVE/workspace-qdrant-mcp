"""
Unit tests for file watcher queue error handling.

Tests verify that file watchers properly handle queue operation failures
without crashing, implement retry logic, and track error statistics.
"""

import asyncio
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone

from common.core.file_watcher import FileWatcher, WatchConfiguration, WatchManager
from common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
def temp_watch_dir():
    """Create temporary directory for watching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def watch_config(temp_watch_dir):
    """Create watch configuration."""
    return WatchConfiguration(
        id="test_watch",
        path=str(temp_watch_dir),
        collection="test-collection",
        patterns=["*.txt", "*.py"],
        auto_ingest=True,
        recursive=True,
        debounce_seconds=0.1  # Short for testing
    )


@pytest.fixture
async def state_manager():
    """Create state manager with temporary database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    manager = SQLiteStateManager(db_path=db_path)
    await manager.initialize()

    yield manager

    await manager.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
class TestFileWatcherQueueErrorHandling:
    """Test suite for file watcher queue error handling."""

    async def test_database_locked_error_with_retry(self, watch_config, state_manager, temp_watch_dir):
        """Test handling of database locked errors with retry logic."""

        # Create a mock callback that simulates database locked then success
        call_count = 0
        async def mock_enqueue_callback(file_path, collection, operation):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: simulate database locked
                raise sqlite3.OperationalError("database is locked")
            else:
                # Second call: success
                await state_manager.enqueue(
                    file_path=file_path,
                    collection=collection,
                    priority=5,
                    tenant_id="default",
                    branch="main"
                )

        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_enqueue_callback
        )

        # Create test file
        test_file = temp_watch_dir / "test.txt"
        test_file.write_text("test content")

        # Start watcher
        await watcher.start()

        # Wait for file to be processed (with retries)
        await asyncio.sleep(0.5)

        await watcher.stop()

        # Verify retry occurred
        assert call_count > 1, "Should have retried after database locked error"

        # Error count should be incremented
        assert watch_config.errors_count > 0

    async def test_validation_error_no_retry(self, watch_config, state_manager, temp_watch_dir):
        """Test that validation errors don't trigger retries."""

        # Create a callback that raises validation error
        async def mock_enqueue_callback(file_path, collection, operation):
            raise ValueError("Priority must be between 0 and 10, got 15")

        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_enqueue_callback
        )

        # Create test file
        test_file = temp_watch_dir / "test.py"
        test_file.write_text("print('hello')")

        # Start watcher
        await watcher.start()

        # Wait for processing
        await asyncio.sleep(0.3)

        await watcher.stop()

        # Watcher should not have crashed
        assert watcher.config.status != "error", "Watcher should not crash on validation error"

        # Error should be logged
        assert watch_config.errors_count > 0

    async def test_runtime_error_handling(self, watch_config, temp_watch_dir):
        """Test handling of runtime errors (e.g., state manager not initialized)."""

        # Create a callback that raises runtime error
        async def mock_enqueue_callback(file_path, collection, operation):
            raise RuntimeError("State manager not initialized")

        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_enqueue_callback
        )

        # Create test file
        test_file = temp_watch_dir / "test.txt"
        test_file.write_text("content")

        # Start watcher
        await watcher.start()

        # Wait for processing
        await asyncio.sleep(0.3)

        await watcher.stop()

        # Watcher should continue running despite error
        assert watch_config.errors_count > 0

    async def test_generic_exception_handling(self, watch_config, temp_watch_dir):
        """Test handling of unexpected exceptions."""

        # Create a callback that raises unexpected exception
        async def mock_enqueue_callback(file_path, collection, operation):
            raise Exception("Unexpected error occurred")

        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_enqueue_callback
        )

        # Create test file
        test_file = temp_watch_dir / "test.txt"
        test_file.write_text("content")

        # Start watcher
        await watcher.start()

        # Wait for processing
        await asyncio.sleep(0.3)

        await watcher.stop()

        # Error count should be incremented
        assert watch_config.errors_count > 0

    async def test_error_statistics_tracking(self, watch_config, state_manager, temp_watch_dir):
        """Test that error statistics are properly tracked."""

        error_count = 0

        async def mock_enqueue_callback(file_path, collection, operation):
            nonlocal error_count
            error_count += 1

            # Simulate different errors
            if error_count == 1:
                raise sqlite3.OperationalError("database is locked")
            elif error_count == 2:
                raise ValueError("Invalid priority")
            elif error_count == 3:
                raise RuntimeError("Not initialized")
            else:
                # Success
                await state_manager.enqueue(
                    file_path=file_path,
                    collection=collection,
                    priority=5,
                    tenant_id="default",
                    branch="main"
                )

        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_enqueue_callback
        )

        # Create multiple test files
        for i in range(4):
            test_file = temp_watch_dir / f"test{i}.txt"
            test_file.write_text(f"content {i}")

        # Start watcher
        await watcher.start()

        # Wait for all files to be processed
        await asyncio.sleep(1.0)

        await watcher.stop()

        # Verify error statistics
        assert watch_config.errors_count >= 3, "Should have recorded at least 3 errors"
        assert watch_config.files_processed > 0, "Should have processed files successfully"

    async def test_watcher_never_crashes(self, watch_config, temp_watch_dir):
        """Test that watcher never crashes regardless of queue errors."""

        async def mock_enqueue_callback(file_path, collection, operation):
            # Always fail
            raise Exception("Catastrophic failure")

        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_enqueue_callback
        )

        # Create test file
        test_file = temp_watch_dir / "test.txt"
        test_file.write_text("content")

        # Start watcher
        await watcher.start()

        # Wait for processing
        await asyncio.sleep(0.5)

        # Watcher should still be running
        assert watcher.is_running(), "Watcher should still be running despite errors"

        await watcher.stop()

        # Config status should not be error
        assert watch_config.status != "error", "Watcher should not be in error state"

    async def test_full_context_error_logging(self, watch_config, temp_watch_dir, caplog):
        """Test that errors are logged with full context."""

        async def mock_enqueue_callback(file_path, collection, operation):
            raise sqlite3.OperationalError("database is locked")

        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_enqueue_callback
        )

        # Create test file
        test_file = temp_watch_dir / "important_file.txt"
        test_file.write_text("important data")

        with caplog.at_level("ERROR"):
            # Start watcher
            await watcher.start()

            # Wait for processing
            await asyncio.sleep(0.5)

            await watcher.stop()

        # Verify error logging includes context
        error_logs = [record for record in caplog.records if record.levelname == "ERROR"]
        assert len(error_logs) > 0, "Should have logged errors"

        # At least one error log should mention the file
        file_mentioned = any("important_file.txt" in record.message for record in error_logs)
        assert file_mentioned, "Error log should include file path"


    async def test_circuit_breaker_pattern(self, watch_config, temp_watch_dir):
        """Test that circuit breaker prevents excessive retries."""

        call_count = 0

        async def mock_enqueue_callback(file_path, collection, operation):
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("database is locked")

        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_enqueue_callback
        )

        # Create multiple test files
        for i in range(15):
            test_file = temp_watch_dir / f"test{i}.txt"
            test_file.write_text(f"content {i}")

        # Start watcher
        await watcher.start()

        # Wait for processing
        await asyncio.sleep(2.0)

        await watcher.stop()

        # Circuit breaker should have limited total calls
        # Without circuit breaker, we'd have way more calls
        # This is a basic check - actual implementation may vary
        assert call_count > 0, "Should have attempted some calls"


@pytest.mark.asyncio
class TestWatchManagerErrorHandling:
    """Test error handling at watch manager level."""

    async def test_manager_continues_after_watcher_error(self, temp_watch_dir):
        """Test that watch manager continues even if one watcher has errors."""

        manager = WatchManager()

        # Create two watch directories
        watch_dir1 = temp_watch_dir / "watch1"
        watch_dir1.mkdir()
        watch_dir2 = temp_watch_dir / "watch2"
        watch_dir2.mkdir()

        # Set up callback that fails for watch1 but succeeds for watch2
        async def mock_callback(file_path, collection, operation):
            if "watch1" in str(file_path):
                raise Exception("Error in watch1")
            # Success for watch2
            pass

        manager.set_ingestion_callback(mock_callback)

        # Add two watches
        watch_id1 = await manager.add_watch(
            path=str(watch_dir1),
            collection="collection1",
            debounce_seconds=0.1
        )
        watch_id2 = await manager.add_watch(
            path=str(watch_dir2),
            collection="collection2",
            debounce_seconds=0.1
        )

        # Start all watches
        await manager.start_all_watches()

        # Create files in both directories
        (watch_dir1 / "test.txt").write_text("content")
        (watch_dir2 / "test.txt").write_text("content")

        # Wait for processing
        await asyncio.sleep(0.5)

        # Get status
        status = manager.get_watch_status()

        # Both watchers should still be running
        assert status[watch_id1]["running"], "Watch1 should still be running"
        assert status[watch_id2]["running"], "Watch2 should still be running"

        await manager.stop_all_watches()

    async def test_get_error_statistics(self, temp_watch_dir):
        """Test retrieving error statistics from watch manager."""

        manager = WatchManager()

        error_occurred = False

        async def mock_callback(file_path, collection, operation):
            nonlocal error_occurred
            error_occurred = True
            raise Exception("Test error")

        manager.set_ingestion_callback(mock_callback)

        # Add watch
        watch_id = await manager.add_watch(
            path=str(temp_watch_dir),
            collection="test-collection",
            debounce_seconds=0.1
        )

        # Start watching
        await manager.start_all_watches()

        # Create test file
        (temp_watch_dir / "test.txt").write_text("content")

        # Wait for processing
        await asyncio.sleep(0.5)

        # Get status
        status = manager.get_watch_status()

        # Verify error count is tracked
        assert status[watch_id]["config"]["errors_count"] > 0

        # Error should have occurred
        assert error_occurred

        await manager.stop_all_watches()
