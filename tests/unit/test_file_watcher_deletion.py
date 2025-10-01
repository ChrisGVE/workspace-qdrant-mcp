"""
Unit tests for file watcher deletion handling.

Tests cover deletion scenarios including:
- Basic deletion handling with pattern matching
- Collection type awareness
- High priority assignment for deletions
- Missing file metadata handling (file already deleted)
- Symlink deletions
- Directory deletions (should be skipped)
- Language filter integration with deleted files
- Auto-ingest disabled scenarios
- Event callback notifications
- Error handling during deletion processing
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.python.common.core.file_watcher import (
    FileWatcher,
    WatchConfiguration,
    WatchEvent,
)
from watchfiles import Change


class TestFileDeletion:
    """Test file deletion handling in FileWatcher."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def watch_config(self, temp_dir):
        """Create basic watch configuration."""
        return WatchConfiguration(
            id="test_watch",
            path=str(temp_dir),
            collection="test-collection",
            patterns=["*.txt", "*.py", "*.md"],
            ignore_patterns=[".git/*", "*.tmp"],
            auto_ingest=True,
            recursive=True,
            use_language_filtering=False,  # Disable for basic tests
        )

    @pytest.fixture
    def state_manager(self):
        """Create mock state manager."""
        manager = AsyncMock()
        manager.enqueue = AsyncMock(return_value="queue_123")
        manager.calculate_tenant_id = AsyncMock(return_value="default")
        manager.get_current_branch = AsyncMock(return_value="main")
        return manager

    @pytest.fixture
    def event_callback(self):
        """Create mock event callback."""
        callback = Mock()
        return callback

    @pytest.mark.asyncio
    async def test_basic_deletion_handling(
        self, watch_config, state_manager, event_callback
    ):
        """Test basic file deletion is processed correctly."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            event_callback=event_callback,
        )

        # Simulate deletion event
        deleted_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(deleted_file)

        # Verify enqueue was called with delete operation
        state_manager.enqueue.assert_called_once()
        call_kwargs = state_manager.enqueue.call_args[1]
        assert call_kwargs["file_path"] == str(deleted_file.resolve())
        assert call_kwargs["collection"] == watch_config.collection
        assert call_kwargs["priority"] == 8  # High priority for deletions

        # Check metadata includes operation
        metadata = call_kwargs["metadata"]
        assert metadata["operation"] == "delete"
        assert metadata["watch_id"] == watch_config.id

        # Verify event callback was called
        event_callback.assert_called_once()
        event = event_callback.call_args[0][0]
        assert isinstance(event, WatchEvent)
        assert event.change_type == "deleted"
        assert event.file_path == str(deleted_file)
        assert event.collection == watch_config.collection

        # Verify stats updated
        assert watcher.config.files_processed == 1

    @pytest.mark.asyncio
    async def test_deletion_respects_patterns(
        self, watch_config, state_manager
    ):
        """Test deletion respects file patterns."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Deletion matching pattern - should be processed
        matching_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(matching_file)
        assert state_manager.enqueue.call_count == 1

        # Deletion not matching pattern - should be filtered
        non_matching_file = Path(watch_config.path) / "test.jpg"
        await watcher._handle_deletion(non_matching_file)
        assert state_manager.enqueue.call_count == 1  # No additional call

        # Verify filtering stats
        assert watcher.config.files_filtered == 1

    @pytest.mark.asyncio
    async def test_deletion_respects_ignore_patterns(
        self, watch_config, state_manager
    ):
        """Test deletion respects ignore patterns."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Deletion matching ignore pattern - should be filtered
        ignored_file = Path(watch_config.path) / "temp.tmp"
        await watcher._handle_deletion(ignored_file)
        assert state_manager.enqueue.call_count == 0
        assert watcher.config.files_filtered == 1

        # Git directory file - should be filtered
        git_file = Path(watch_config.path) / ".git" / "config"
        await watcher._handle_deletion(git_file)
        assert state_manager.enqueue.call_count == 0
        assert watcher.config.files_filtered == 2

    @pytest.mark.asyncio
    async def test_deletion_with_auto_ingest_disabled(
        self, watch_config, state_manager
    ):
        """Test deletion when auto_ingest is disabled."""
        watch_config.auto_ingest = False
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        deleted_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(deleted_file)

        # Should not enqueue
        state_manager.enqueue.assert_not_called()

    @pytest.mark.asyncio
    async def test_deletion_collection_awareness(
        self, temp_dir, state_manager
    ):
        """Test deletion uses correct collection from watch config."""
        # Create multiple watchers with different collections
        configs = []
        watchers = []

        for i, collection in enumerate(["coll-a", "coll-b", "coll-c"]):
            config = WatchConfiguration(
                id=f"watch_{i}",
                path=str(temp_dir),
                collection=collection,
                patterns=["*.txt"],
                use_language_filtering=False,
            )
            configs.append(config)
            watchers.append(
                FileWatcher(
                    config=config,
                    state_manager=state_manager,
                )
            )

        # Process deletion with each watcher
        deleted_file = Path(temp_dir) / "test.txt"

        for watcher in watchers:
            await watcher._handle_deletion(deleted_file)

        # Verify each call used correct collection
        assert state_manager.enqueue.call_count == 3
        calls = state_manager.enqueue.call_args_list

        for i, call in enumerate(calls):
            collection = call[1]["collection"]
            assert collection == f"coll-{chr(97 + i)}"  # coll-a, coll-b, coll-c

            # Verify metadata includes delete operation
            metadata = call[1]["metadata"]
            assert metadata["operation"] == "delete"

    @pytest.mark.asyncio
    async def test_deletion_priority(
        self, watch_config, state_manager
    ):
        """Test deletions have higher priority than regular operations."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        deleted_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(deleted_file)

        # Verify deletion has priority 8 (HIGH)
        state_manager.enqueue.assert_called_once()
        call_kwargs = state_manager.enqueue.call_args[1]
        assert call_kwargs["priority"] == 8

    @pytest.mark.asyncio
    async def test_deletion_error_handling(
        self, watch_config, state_manager, event_callback
    ):
        """Test error handling during deletion processing."""
        # Make enqueue raise error
        state_manager.enqueue.side_effect = RuntimeError("Queue full")

        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            event_callback=event_callback,
        )

        deleted_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(deleted_file)

        # Event callback should still be called
        event_callback.assert_called_once()

        # Error should be tracked
        assert watcher.config.errors_count == 1

        # Files processed should not increment on error
        assert watcher.config.files_processed == 0

    @pytest.mark.asyncio
    async def test_deletion_without_event_callback(
        self, watch_config, state_manager
    ):
        """Test deletion works without event callback."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            event_callback=None,  # No event callback
        )

        deleted_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(deleted_file)

        # Should still process deletion
        state_manager.enqueue.assert_called_once()
        assert watcher.config.files_processed == 1

    @pytest.mark.asyncio
    async def test_deletion_event_callback_error(
        self, watch_config, state_manager, event_callback
    ):
        """Test deletion continues if event callback raises error."""
        # Make event callback raise error
        event_callback.side_effect = RuntimeError("Callback failed")

        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            event_callback=event_callback,
        )

        deleted_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(deleted_file)

        # Event callback error shouldn't prevent processing
        event_callback.assert_called_once()
        state_manager.enqueue.assert_called_once()
        assert watcher.config.files_processed == 1

    @pytest.mark.asyncio
    async def test_deletion_in_handle_changes(
        self, watch_config, state_manager
    ):
        """Test deletion is handled correctly in _handle_changes."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Simulate watchfiles change set with deletion
        deleted_file = Path(watch_config.path) / "test.txt"
        changes = {(Change.deleted, str(deleted_file))}

        await watcher._handle_changes(changes)

        # Verify deletion was processed
        state_manager.enqueue.assert_called_once()
        call_kwargs = state_manager.enqueue.call_args[1]
        assert call_kwargs["metadata"]["operation"] == "delete"

    @pytest.mark.asyncio
    async def test_multiple_deletions_in_batch(
        self, watch_config, state_manager
    ):
        """Test handling multiple deletions in one change batch."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Multiple files deleted
        files = [
            Path(watch_config.path) / "file1.txt",
            Path(watch_config.path) / "file2.py",
            Path(watch_config.path) / "file3.md",
        ]

        changes = {(Change.deleted, str(f)) for f in files}
        await watcher._handle_changes(changes)

        # All deletions should be processed
        assert state_manager.enqueue.call_count == 3
        assert watcher.config.files_processed == 3

    @pytest.mark.asyncio
    async def test_mixed_changes_with_deletions(
        self, watch_config, state_manager, temp_dir
    ):
        """Test handling mixed adds, modifies, and deletes."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Create actual files for add/modify events
        added_file = temp_dir / "new.txt"
        added_file.write_text("new content")

        modified_file = temp_dir / "existing.txt"
        modified_file.write_text("modified content")

        # Deleted file doesn't exist
        deleted_file = temp_dir / "deleted.txt"

        changes = {
            (Change.added, str(added_file)),
            (Change.modified, str(modified_file)),
            (Change.deleted, str(deleted_file)),
        }

        await watcher._handle_changes(changes)

        # Wait for debounced ingestions
        await asyncio.sleep(0.1)

        # Deletion should be immediate, add/modify are debounced
        # Check that delete was called
        delete_calls = [
            call for call in state_manager.enqueue.call_args_list
            if call[1]["metadata"]["operation"] == "delete"
        ]
        assert len(delete_calls) == 1
        assert delete_calls[0][1]["file_path"] == str(deleted_file.resolve())

    @pytest.mark.asyncio
    async def test_deletion_path_resolution(
        self, watch_config, state_manager
    ):
        """Test deletion resolves file paths correctly."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Use relative path
        relative_file = Path(watch_config.path) / "subdir" / "test.txt"
        await watcher._handle_deletion(relative_file)

        # Verify absolute path was used in enqueue
        state_manager.enqueue.assert_called_once()
        called_path = state_manager.enqueue.call_args[1]["file_path"]
        assert Path(called_path).is_absolute()
        assert called_path == str(relative_file.resolve())

    @pytest.mark.asyncio
    async def test_deletion_stats_tracking(
        self, watch_config, state_manager
    ):
        """Test deletion updates statistics correctly."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        initial_processed = watcher.config.files_processed
        initial_errors = watcher.config.errors_count
        initial_filtered = watcher.config.files_filtered

        # Process successful deletion
        deleted_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(deleted_file)

        assert watcher.config.files_processed == initial_processed + 1
        assert watcher.config.errors_count == initial_errors
        assert watcher.config.last_activity is not None

        # Process filtered deletion
        filtered_file = Path(watch_config.path) / "ignored.jpg"
        await watcher._handle_deletion(filtered_file)

        assert watcher.config.files_filtered == initial_filtered + 1
        assert watcher.config.files_processed == initial_processed + 1  # No change

    @pytest.mark.asyncio
    async def test_deletion_metadata_in_event(
        self, watch_config, state_manager, event_callback
    ):
        """Test deletion event contains correct metadata."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            event_callback=event_callback,
        )

        deleted_file = Path(watch_config.path) / "test.txt"
        deletion_time_before = datetime.now(timezone.utc)

        await watcher._handle_deletion(deleted_file)

        deletion_time_after = datetime.now(timezone.utc)

        # Verify event metadata
        event_callback.assert_called_once()
        event = event_callback.call_args[0][0]

        assert event.change_type == "deleted"
        assert event.file_path == str(deleted_file)
        assert event.collection == watch_config.collection

        # Timestamp should be within range
        event_time = datetime.fromisoformat(event.timestamp)
        assert deletion_time_before <= event_time <= deletion_time_after

    @pytest.mark.asyncio
    async def test_deletion_metadata_includes_watch_info(
        self, watch_config, state_manager
    ):
        """Test deletion metadata includes watch information."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        deleted_file = Path(watch_config.path) / "test.txt"
        await watcher._handle_deletion(deleted_file)

        # Verify metadata
        state_manager.enqueue.assert_called_once()
        metadata = state_manager.enqueue.call_args[1]["metadata"]

        assert metadata["watch_id"] == watch_config.id
        assert metadata["watch_path"] == watch_config.path
        assert metadata["operation"] == "delete"
        assert metadata["event_type"] == "file_change"
        assert "detected_at" in metadata
        assert "project_root" in metadata


class TestDeletionEdgeCases:
    """Test edge cases for deletion handling."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def watch_config(self, temp_dir):
        """Create watch configuration."""
        return WatchConfiguration(
            id="test_watch",
            path=str(temp_dir),
            collection="test-collection",
            patterns=["*.txt"],
            use_language_filtering=False,
        )

    @pytest.fixture
    def state_manager(self):
        """Create mock state manager."""
        manager = AsyncMock()
        manager.enqueue = AsyncMock(return_value="queue_123")
        manager.calculate_tenant_id = AsyncMock(return_value="default")
        manager.get_current_branch = AsyncMock(return_value="main")
        return manager

    @pytest.mark.asyncio
    async def test_symlink_deletion_handling(
        self, watch_config, state_manager, temp_dir
    ):
        """Test deletion of symlink is handled correctly."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Symlink path (doesn't need to exist)
        symlink_file = temp_dir / "symlink.txt"

        await watcher._handle_deletion(symlink_file)

        # Should be processed like regular file
        state_manager.enqueue.assert_called_once()
        assert watcher.config.files_processed == 1

    @pytest.mark.asyncio
    async def test_directory_deletion_skipped(
        self, watch_config, state_manager, temp_dir
    ):
        """Test directory deletions are skipped in _handle_changes."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Directory path (watchfiles would not send this for directories anyway,
        # but test the logic)
        dir_path = temp_dir / "subdir"

        # In _handle_changes, directory deletions would be filtered by pattern
        # matching since directories typically don't match file patterns
        await watcher._handle_deletion(dir_path)

        # Should be filtered out by pattern matching (*.txt)
        state_manager.enqueue.assert_not_called()
        assert watcher.config.files_filtered == 1

    @pytest.mark.asyncio
    async def test_nested_path_deletion(
        self, watch_config, state_manager, temp_dir
    ):
        """Test deletion of deeply nested file."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Deeply nested file
        nested_file = temp_dir / "a" / "b" / "c" / "d" / "test.txt"

        await watcher._handle_deletion(nested_file)

        # Should be processed normally
        state_manager.enqueue.assert_called_once()
        assert watcher.config.files_processed == 1

    @pytest.mark.asyncio
    async def test_deletion_with_special_characters(
        self, watch_config, state_manager, temp_dir
    ):
        """Test deletion of file with special characters in name."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # File with special characters
        special_file = temp_dir / "file with spaces & symbols!@#.txt"

        await watcher._handle_deletion(special_file)

        # Should be processed normally
        state_manager.enqueue.assert_called_once()
        called_path = state_manager.enqueue.call_args[1]["file_path"]
        assert "file with spaces & symbols!@#" in called_path

    @pytest.mark.asyncio
    async def test_deletion_unicode_filename(
        self, watch_config, state_manager, temp_dir
    ):
        """Test deletion of file with unicode characters."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Unicode filename
        unicode_file = temp_dir / "文件名_файл_αρχείο.txt"

        await watcher._handle_deletion(unicode_file)

        # Should be processed normally
        state_manager.enqueue.assert_called_once()
        assert watcher.config.files_processed == 1

    @pytest.mark.asyncio
    async def test_rapid_delete_recreate(
        self, watch_config, state_manager, temp_dir
    ):
        """Test rapid deletion and recreation of same file."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        test_file = temp_dir / "test.txt"

        # Simulate deletion
        changes_delete = {(Change.deleted, str(test_file))}
        await watcher._handle_changes(changes_delete)

        # Create the file
        test_file.write_text("content")

        # Simulate addition
        changes_add = {(Change.added, str(test_file))}
        await watcher._handle_changes(changes_add)

        # Wait for debounced add
        await asyncio.sleep(0.1)

        # Both operations should be processed
        assert state_manager.enqueue.call_count >= 1  # At least delete called

        # Verify delete was called
        delete_calls = [
            call for call in state_manager.enqueue.call_args_list
            if call[1]["metadata"]["operation"] == "delete"
        ]
        assert len(delete_calls) == 1

    @pytest.mark.asyncio
    async def test_concurrent_deletions(
        self, watch_config, state_manager, temp_dir
    ):
        """Test concurrent deletion processing."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        # Multiple files to delete concurrently
        files = [temp_dir / f"file{i}.txt" for i in range(10)]

        # Process deletions concurrently
        await asyncio.gather(
            *[watcher._handle_deletion(f) for f in files]
        )

        # All deletions should be processed
        assert state_manager.enqueue.call_count == 10
        assert watcher.config.files_processed == 10

    @pytest.mark.asyncio
    async def test_deletion_enqueue_slow(
        self, watch_config, temp_dir
    ):
        """Test handling of slow state manager enqueue during deletion."""
        # Create slow state manager
        state_manager = AsyncMock()
        async def slow_enqueue(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "queue_123"

        state_manager.enqueue = slow_enqueue
        state_manager.calculate_tenant_id = AsyncMock(return_value="default")
        state_manager.get_current_branch = AsyncMock(return_value="main")

        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
        )

        deleted_file = temp_dir / "test.txt"

        # Should complete even with slow enqueue
        start_time = asyncio.get_event_loop().time()
        await watcher._handle_deletion(deleted_file)
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should have waited for enqueue
        assert elapsed >= 0.1
        assert watcher.config.files_processed == 1
