#!/usr/bin/env python3
"""
Integration test for operation type detection in FileWatcher.

Tests the full flow from file event → operation type determination → queue enqueueing.
"""

import asyncio
import tempfile
from pathlib import Path
import sys
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from watchfiles import Change
from common.core.file_watcher import FileWatcher, WatchConfiguration
from unittest.mock import Mock, AsyncMock, call


async def test_integration():
    """Test the full integration of operation type detection."""

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create config
        config = WatchConfiguration(
            id="test_watch",
            path=str(temp_dir),
            collection="test_collection",
            patterns=["*.txt"],
            debounce_seconds=0.1,  # Short debounce for testing
        )

        # Create mock state manager with async methods
        state_manager = Mock()
        state_manager.enqueue = AsyncMock(return_value="queue_id_123")
        state_manager.calculate_tenant_id = AsyncMock(return_value="tenant_123")
        state_manager.get_current_branch = AsyncMock(return_value="main")

        # Create watcher
        watcher = FileWatcher(
            config=config,
            state_manager=state_manager,
            event_callback=None,
            filter_config_path=None,
        )

        # Test 1: Add a file (should result in 'ingest' operation)
        print("Test 1: Adding a new file...")
        new_file = temp_dir / "new.txt"
        new_file.write_text("new content")

        changes = {(Change.added, str(new_file))}
        await watcher._handle_changes(changes)

        # Wait for debounce
        await asyncio.sleep(0.2)

        # Check that enqueue was called
        assert state_manager.enqueue.called, "enqueue should have been called"
        last_call = state_manager.enqueue.call_args
        metadata = last_call.kwargs.get('metadata', {})
        assert metadata['operation'] == 'ingest', f"Expected 'ingest', got {metadata['operation']}"
        print(f"✓ Test 1 passed: New file enqueued with 'ingest' operation")

        # Reset mock
        state_manager.enqueue.reset_mock()

        # Test 2: Modify existing file (should result in 'update' operation)
        print("\nTest 2: Modifying existing file...")
        new_file.write_text("modified content")

        changes = {(Change.modified, str(new_file))}
        await watcher._handle_changes(changes)

        # Wait for debounce
        await asyncio.sleep(0.2)

        # Check that enqueue was called
        assert state_manager.enqueue.called, "enqueue should have been called"
        last_call = state_manager.enqueue.call_args
        metadata = last_call.kwargs.get('metadata', {})
        assert metadata['operation'] == 'update', f"Expected 'update', got {metadata['operation']}"
        print(f"✓ Test 2 passed: Modified file enqueued with 'update' operation")

        # Reset mock
        state_manager.enqueue.reset_mock()

        # Test 3: Delete file (should result in 'delete' operation)
        print("\nTest 3: Deleting file...")
        new_file.unlink()

        changes = {(Change.deleted, str(new_file))}
        await watcher._handle_changes(changes)

        # Wait a moment (deletions are immediate, no debounce)
        await asyncio.sleep(0.1)

        # Check that enqueue was called
        assert state_manager.enqueue.called, "enqueue should have been called"
        last_call = state_manager.enqueue.call_args
        metadata = last_call.kwargs.get('metadata', {})
        assert metadata['operation'] == 'delete', f"Expected 'delete', got {metadata['operation']}"
        # Deletions should have higher priority
        assert last_call.kwargs.get('priority', 5) == 8, "Delete operations should have priority 8"
        print(f"✓ Test 3 passed: Deleted file enqueued with 'delete' operation and priority 8")

        # Reset mock
        state_manager.enqueue.reset_mock()

        # Test 4: Race condition - file deleted during debounce
        print("\nTest 4: Race condition - file deleted during debounce...")
        race_file = temp_dir / "race.txt"
        race_file.write_text("will be deleted")

        changes = {(Change.modified, str(race_file))}
        await watcher._handle_changes(changes)

        # Delete the file before debounce completes
        await asyncio.sleep(0.05)
        race_file.unlink()

        # Wait for debounce to complete
        await asyncio.sleep(0.15)

        # Check that enqueue was called
        assert state_manager.enqueue.called, "enqueue should have been called"
        last_call = state_manager.enqueue.call_args
        metadata = last_call.kwargs.get('metadata', {})
        # Should detect deletion during debounce period
        assert metadata['operation'] == 'delete', f"Expected 'delete' for race condition, got {metadata['operation']}"
        print(f"✓ Test 4 passed: Race condition correctly detected 'delete' operation")

        print("\n✅ All integration tests passed!")


if __name__ == "__main__":
    asyncio.run(test_integration())
