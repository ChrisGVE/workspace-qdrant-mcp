#!/usr/bin/env python3
"""Quick verification that _determine_operation_type works correctly."""

import tempfile
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from watchfiles import Change
from common.core.file_watcher import FileWatcher, WatchConfiguration
from unittest.mock import Mock

def test_operation_type_detection():
    """Test the _determine_operation_type method."""

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)

        # Create config
        config = WatchConfiguration(
            id="test_watch",
            path=str(temp_dir),
            collection="test_collection",
        )

        # Create mock state manager
        state_manager = Mock()

        # Create watcher
        watcher = FileWatcher(
            config=config,
            state_manager=state_manager,
            event_callback=None,
            filter_config_path=None,
        )

        # Test 1: Added file returns 'ingest'
        test_file = temp_dir / "new_file.txt"
        test_file.write_text("content")
        operation = watcher._determine_operation_type(Change.added, test_file)
        assert operation == 'ingest', f"Expected 'ingest', got {operation}"
        print("✓ Test 1 passed: Added file returns 'ingest'")

        # Test 2: Modified existing file returns 'update'
        operation = watcher._determine_operation_type(Change.modified, test_file)
        assert operation == 'update', f"Expected 'update', got {operation}"
        print("✓ Test 2 passed: Modified existing file returns 'update'")

        # Test 3: Modified missing file returns 'delete' (race condition)
        missing_file = temp_dir / "missing.txt"
        operation = watcher._determine_operation_type(Change.modified, missing_file)
        assert operation == 'delete', f"Expected 'delete', got {operation}"
        print("✓ Test 3 passed: Modified missing file returns 'delete'")

        # Test 4: Deleted file returns 'delete'
        operation = watcher._determine_operation_type(Change.deleted, test_file)
        assert operation == 'delete', f"Expected 'delete', got {operation}"
        print("✓ Test 4 passed: Deleted file returns 'delete'")

        # Test 5: Valid symlink returns 'update'
        target = temp_dir / "target.txt"
        target.write_text("target")
        link = temp_dir / "link.txt"
        link.symlink_to(target)
        operation = watcher._determine_operation_type(Change.modified, link)
        assert operation == 'update', f"Expected 'update', got {operation}"
        print("✓ Test 5 passed: Valid symlink returns 'update'")

        # Test 6: Broken symlink returns 'delete'
        target.unlink()
        operation = watcher._determine_operation_type(Change.modified, link)
        assert operation == 'delete', f"Expected 'delete', got {operation}"
        print("✓ Test 6 passed: Broken symlink returns 'delete'")

        print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_operation_type_detection()
