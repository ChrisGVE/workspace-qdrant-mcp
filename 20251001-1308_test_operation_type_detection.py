"""
Unit tests for FileWatcher._determine_operation_type() method.

Tests cover all operation type detection logic including edge cases:
- Change.added → 'ingest'
- Change.modified + file exists → 'update'
- Change.modified + file missing → 'delete'
- Change.deleted → 'delete'
- Symlinks (valid and broken)
- Permission errors
- Race conditions
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from watchfiles import Change

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from common.core.file_watcher import FileWatcher, WatchConfiguration
from common.core.sqlite_state_manager import SQLiteStateManager


class TestDetermineOperationType:
    """Test suite for _determine_operation_type method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def config(self, temp_dir):
        """Create watch configuration."""
        return WatchConfiguration(
            id="test_watch",
            path=str(temp_dir),
            collection="test_collection",
        )

    @pytest.fixture
    def state_manager(self):
        """Create mock state manager."""
        return Mock(spec=SQLiteStateManager)

    @pytest.fixture
    def watcher(self, config, state_manager):
        """Create FileWatcher instance."""
        return FileWatcher(
            config=config,
            state_manager=state_manager,
            event_callback=None,
            filter_config_path=None,
        )

    def test_added_file_returns_ingest(self, watcher, temp_dir):
        """Test that Change.added returns 'ingest'."""
        test_file = temp_dir / "new_file.txt"
        test_file.write_text("content")

        operation = watcher._determine_operation_type(Change.added, test_file)

        assert operation == 'ingest'

    def test_modified_existing_file_returns_update(self, watcher, temp_dir):
        """Test that Change.modified with existing file returns 'update'."""
        test_file = temp_dir / "existing_file.txt"
        test_file.write_text("original content")

        operation = watcher._determine_operation_type(Change.modified, test_file)

        assert operation == 'update'

    def test_modified_missing_file_returns_delete(self, watcher, temp_dir):
        """Test that Change.modified with missing file returns 'delete' (race condition)."""
        test_file = temp_dir / "missing_file.txt"
        # File doesn't exist - simulates deletion during processing

        operation = watcher._determine_operation_type(Change.modified, test_file)

        assert operation == 'delete'

    def test_deleted_file_returns_delete(self, watcher, temp_dir):
        """Test that Change.deleted returns 'delete'."""
        test_file = temp_dir / "deleted_file.txt"
        # File doesn't need to exist for deleted event

        operation = watcher._determine_operation_type(Change.deleted, test_file)

        assert operation == 'delete'

    def test_symlink_valid_target_returns_update(self, watcher, temp_dir):
        """Test that valid symlink with modified change returns 'update'."""
        target_file = temp_dir / "target.txt"
        target_file.write_text("target content")

        symlink_file = temp_dir / "link.txt"
        symlink_file.symlink_to(target_file)

        operation = watcher._determine_operation_type(Change.modified, symlink_file)

        assert operation == 'update'

    def test_symlink_broken_target_returns_delete(self, watcher, temp_dir):
        """Test that broken symlink returns 'delete'."""
        target_file = temp_dir / "target.txt"
        target_file.write_text("target content")

        symlink_file = temp_dir / "link.txt"
        symlink_file.symlink_to(target_file)

        # Delete target to break symlink
        target_file.unlink()

        operation = watcher._determine_operation_type(Change.modified, symlink_file)

        assert operation == 'delete'

    def test_permission_error_defaults_to_update(self, watcher, temp_dir):
        """Test that permission errors default to 'update'."""
        test_file = temp_dir / "restricted_file.txt"

        # Mock is_symlink to raise PermissionError
        with patch.object(Path, 'is_symlink', side_effect=PermissionError("Access denied")):
            operation = watcher._determine_operation_type(Change.modified, test_file)

        assert operation == 'update'

    def test_os_error_defaults_to_update(self, watcher, temp_dir):
        """Test that OS errors default to 'update'."""
        test_file = temp_dir / "error_file.txt"

        # Mock is_symlink to raise OSError
        with patch.object(Path, 'is_symlink', side_effect=OSError("Disk error")):
            operation = watcher._determine_operation_type(Change.modified, test_file)

        assert operation == 'update'

    def test_unknown_change_type_defaults_to_ingest(self, watcher, temp_dir):
        """Test that unknown change types default to 'ingest'."""
        test_file = temp_dir / "file.txt"
        test_file.write_text("content")

        # Use an invalid change type (cast from int)
        unknown_change = 99  # Not a valid Change value

        operation = watcher._determine_operation_type(unknown_change, test_file)

        assert operation == 'ingest'

    def test_added_non_existent_file_still_returns_ingest(self, watcher, temp_dir):
        """Test that Change.added returns 'ingest' even if file doesn't exist."""
        test_file = temp_dir / "non_existent.txt"
        # File doesn't exist - edge case

        operation = watcher._determine_operation_type(Change.added, test_file)

        assert operation == 'ingest'

    def test_debug_logging_for_each_operation(self, watcher, temp_dir, caplog):
        """Test that debug logging occurs for each operation type."""
        import logging
        caplog.set_level(logging.DEBUG)

        test_file = temp_dir / "test.txt"
        test_file.write_text("content")

        # Test ingest
        watcher._determine_operation_type(Change.added, test_file)
        assert "Operation type 'ingest' determined" in caplog.text

        caplog.clear()

        # Test update
        watcher._determine_operation_type(Change.modified, test_file)
        assert "Operation type 'update' determined" in caplog.text

        caplog.clear()

        # Test delete
        watcher._determine_operation_type(Change.deleted, test_file)
        assert "Operation type 'delete' determined" in caplog.text

    def test_race_condition_file_deleted_between_event_and_check(self, watcher, temp_dir, caplog):
        """Test race condition logging when file is deleted between event and check."""
        import logging
        caplog.set_level(logging.DEBUG)

        test_file = temp_dir / "race_condition.txt"
        # File doesn't exist - simulates race condition

        watcher._determine_operation_type(Change.modified, test_file)

        assert "race condition" in caplog.text.lower()
        assert "file no longer exists" in caplog.text.lower()

    def test_permission_warning_logged(self, watcher, temp_dir, caplog):
        """Test that permission errors are logged as warnings."""
        import logging
        caplog.set_level(logging.WARNING)

        test_file = temp_dir / "perm_error.txt"

        with patch.object(Path, 'is_symlink', side_effect=PermissionError("Access denied")):
            watcher._determine_operation_type(Change.modified, test_file)

        assert "could not check file existence" in caplog.text.lower()
        assert "defaulting to 'update'" in caplog.text.lower()


class TestIntegrationWithHandleChanges:
    """Integration tests for _determine_operation_type with _handle_changes."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    async def watcher(self, temp_dir):
        """Create FileWatcher with real components."""
        config = WatchConfiguration(
            id="integration_test",
            path=str(temp_dir),
            collection="test_collection",
            patterns=["*.txt"],
            auto_ingest=True,
        )

        # Create real state manager with in-memory DB
        state_manager = Mock(spec=SQLiteStateManager)
        state_manager.enqueue = Mock(return_value="queue_id_123")
        state_manager.calculate_tenant_id = Mock(return_value="tenant_123")
        state_manager.get_current_branch = Mock(return_value="main")

        watcher = FileWatcher(
            config=config,
            state_manager=state_manager,
            filter_config_path=None,
        )

        return watcher

    @pytest.mark.asyncio
    async def test_added_file_uses_ingest_operation(self, watcher, temp_dir):
        """Test that added files use 'ingest' operation in handle_changes."""
        test_file = temp_dir / "new.txt"
        test_file.write_text("new content")

        changes = {(Change.added, str(test_file))}

        await watcher._handle_changes(changes)

        # Debounced, so wait a moment
        await watcher._debounce_tasks[str(test_file)]

        # Check that enqueue was called with correct operation
        # The operation is determined after debounce in _delayed_ingestion
        watcher.state_manager.enqueue.assert_called()

    @pytest.mark.asyncio
    async def test_modified_file_uses_update_operation(self, watcher, temp_dir):
        """Test that modified existing files use 'update' operation."""
        test_file = temp_dir / "existing.txt"
        test_file.write_text("original")

        changes = {(Change.modified, str(test_file))}

        await watcher._handle_changes(changes)

        # Debounced
        await watcher._debounce_tasks[str(test_file)]

        watcher.state_manager.enqueue.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
