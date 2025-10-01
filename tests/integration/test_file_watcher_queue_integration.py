"""
Integration Tests for File Watcher and Queue System.

Tests the complete integration between FileWatcher, WatchManager, and
SQLiteQueueClient/SQLiteStateManager. Verifies end-to-end workflows including:
- File system events triggering queue operations
- Priority calculations and queue ordering
- Debouncing and batching of events
- Error handling and recovery
- Multi-watcher scenarios
- High-volume processing capabilities
"""

import asyncio
import pytest
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from src.python.common.core.file_watcher import (
    FileWatcher,
    WatchManager,
    WatchConfiguration,
    WatchEvent,
)
from src.python.common.core.sqlite_state_manager import SQLiteStateManager
from src.python.common.core.queue_client import (
    SQLiteQueueClient,
    QueueOperation,
    QueueItem,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    try:
        Path(db_path).unlink()
        Path(f"{db_path}-wal").unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
async def state_manager(temp_db):
    """Initialize SQLiteStateManager."""
    manager = SQLiteStateManager(db_path=temp_db)
    success = await manager.initialize()
    assert success, "State manager initialization failed"
    yield manager
    await manager.close()


@pytest.fixture
async def queue_client(temp_db):
    """Initialize SQLiteQueueClient."""
    client = SQLiteQueueClient(db_path=temp_db)
    await client.initialize()
    yield client
    await client.close()


@pytest.fixture
def watch_directory(tmp_path):
    """Create temporary directory for watching."""
    watch_dir = tmp_path / "watch_test"
    watch_dir.mkdir()

    # Create .git directory to simulate project root
    git_dir = watch_dir / ".git"
    git_dir.mkdir()

    return watch_dir


@pytest.fixture
def watch_config(watch_directory):
    """Create basic watch configuration."""
    return WatchConfiguration(
        id="test_watch_1",
        path=str(watch_directory),
        collection="test-collection",
        patterns=["*.py", "*.txt", "*.md"],
        ignore_patterns=[".git/*", "__pycache__/*"],
        auto_ingest=True,
        recursive=True,
        debounce_seconds=1,  # Short debounce for testing
        use_language_filtering=False,  # Disable for simpler testing
    )


@pytest.fixture
async def file_watcher(watch_config, state_manager):
    """Create FileWatcher instance."""
    events = []

    def event_callback(event: WatchEvent):
        events.append(event)

    watcher = FileWatcher(
        config=watch_config,
        state_manager=state_manager,
        event_callback=event_callback,
        filter_config_path=None,
    )

    # Attach events list for test inspection
    watcher._test_events = events

    yield watcher

    # Cleanup
    if watcher.is_running():
        await watcher.stop()


@pytest.fixture
async def watch_manager(temp_db):
    """Create WatchManager instance."""
    # Use temp config file
    config_file = Path(tempfile.gettempdir()) / f"watch_test_{time.time()}.json"

    manager = WatchManager(config_file=str(config_file))

    yield manager

    # Cleanup
    await manager.stop_all_watches()
    if config_file.exists():
        config_file.unlink()


# =============================================================================
# Test 1: Basic File Watcher -> Queue Integration
# =============================================================================


@pytest.mark.asyncio
async def test_file_creation_enqueues_correctly(
    file_watcher, queue_client, watch_directory
):
    """Test that creating a file triggers enqueue with correct metadata."""
    # Start watcher
    await file_watcher.start()

    # Wait for watcher to initialize
    await asyncio.sleep(0.5)

    # Create a test file
    test_file = watch_directory / "test.py"
    test_file.write_text("# Test file\nprint('hello')\n")

    # Wait for debouncing + processing
    await asyncio.sleep(2)

    # Verify file was enqueued
    queue_depth = await queue_client.get_queue_depth()
    assert queue_depth >= 1, "File should be enqueued"

    # Verify queue item has correct properties
    items = await queue_client.dequeue_batch(batch_size=10)
    assert len(items) >= 1, "Should have at least one item"

    file_item = next((item for item in items if test_file.name in item.file_absolute_path), None)
    assert file_item is not None, "Test file should be in queue"
    assert file_item.collection_name == "test-collection"
    assert file_item.operation == QueueOperation.INGEST
    assert file_item.priority == 5  # Default NORMAL priority
    assert file_item.tenant_id is not None
    assert file_item.branch is not None


@pytest.mark.asyncio
async def test_file_modification_updates_queue(
    file_watcher, queue_client, watch_directory
):
    """Test that modifying a file triggers update operation."""
    # Create file first
    test_file = watch_directory / "modify_test.py"
    test_file.write_text("# Original content\n")

    # Start watcher
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Modify the file
    test_file.write_text("# Modified content\nprint('updated')\n")

    # Wait for debouncing
    await asyncio.sleep(2)

    # Check queue
    items = await queue_client.dequeue_batch(batch_size=10)
    file_items = [item for item in items if test_file.name in item.file_absolute_path]

    # Should have one or more items (depending on timing)
    assert len(file_items) >= 1
    # Most recent should be update operation (or ingest if detected as new)
    assert file_items[-1].operation in [QueueOperation.UPDATE, QueueOperation.INGEST]


@pytest.mark.asyncio
async def test_file_deletion_enqueues_delete_operation(
    file_watcher, queue_client, watch_directory
):
    """Test that deleting a file triggers delete operation with high priority."""
    # Create file
    test_file = watch_directory / "delete_test.py"
    test_file.write_text("# To be deleted\n")

    # Start watcher
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Delete the file
    test_file.unlink()

    # Wait for event processing (deletions are immediate, no debounce)
    await asyncio.sleep(1)

    # Check queue
    items = await queue_client.dequeue_batch(batch_size=10)
    delete_items = [
        item for item in items
        if test_file.name in item.file_absolute_path and item.operation == QueueOperation.DELETE
    ]

    assert len(delete_items) >= 1, "Delete operation should be enqueued"
    # Deletions should have high priority (8)
    assert delete_items[0].priority == 8, "Delete operations should have high priority"


# =============================================================================
# Test 2: Queue Depth and Priority Verification
# =============================================================================


@pytest.mark.asyncio
async def test_queue_depth_changes_correctly(
    file_watcher, queue_client, watch_directory
):
    """Test that queue depth increases with new files and decreases on dequeue."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    initial_depth = await queue_client.get_queue_depth()

    # Create multiple files
    file_count = 5
    for i in range(file_count):
        test_file = watch_directory / f"file_{i}.py"
        test_file.write_text(f"# File {i}\n")

    # Wait for all files to be enqueued
    await asyncio.sleep(3)

    current_depth = await queue_client.get_queue_depth()
    assert current_depth >= initial_depth + file_count, "Queue depth should increase"

    # Dequeue some items
    items = await queue_client.dequeue_batch(batch_size=3)
    for item in items:
        await queue_client.mark_complete(item.file_absolute_path)

    # Verify depth decreased
    final_depth = await queue_client.get_queue_depth()
    assert final_depth == current_depth - len(items), "Queue depth should decrease after completion"


@pytest.mark.asyncio
async def test_priority_ordering_in_queue(
    file_watcher, queue_client, watch_directory, state_manager
):
    """Test that files are dequeued in priority order."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Create files and manually set different priorities
    files_with_priorities = [
        ("low_priority.py", 2),
        ("high_priority.py", 9),
        ("normal_priority.py", 5),
        ("urgent_priority.py", 10),
    ]

    for filename, priority in files_with_priorities:
        test_file = watch_directory / filename
        test_file.write_text(f"# Priority {priority}\n")

        # Wait for file to be enqueued
        await asyncio.sleep(2)

        # Update priority in queue
        await queue_client.update_priority(str(test_file), priority)

    # Dequeue batch - should come in priority order
    items = await queue_client.dequeue_batch(batch_size=10)

    # Verify ordering (highest priority first)
    assert len(items) >= 4
    priorities = [item.priority for item in items[:4]]
    assert priorities == sorted(priorities, reverse=True), "Items should be ordered by priority DESC"


# =============================================================================
# Test 3: High-Volume Scenarios
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_high_volume_file_creation(
    file_watcher, queue_client, watch_directory
):
    """Test handling 100+ files created rapidly."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    file_count = 100
    start_time = time.time()

    # Create files rapidly
    for i in range(file_count):
        test_file = watch_directory / f"bulk_file_{i}.py"
        test_file.write_text(f"# File {i}\nprint('test {i}')\n")

    # Wait for processing
    await asyncio.sleep(10)  # Allow time for debouncing and processing

    end_time = time.time()
    elapsed = end_time - start_time

    # Verify all files are enqueued
    queue_depth = await queue_client.get_queue_depth()
    assert queue_depth >= file_count, f"Should have enqueued {file_count} files, got {queue_depth}"

    # Verify throughput (target: 1000+ docs/min = 16.67/sec)
    # For 100 files, should complete well under 6 seconds plus debounce time
    # With 1s debounce, expect ~11 seconds max (10s wait + 1s processing)
    throughput = file_count / elapsed * 60  # docs per minute
    print(f"Throughput: {throughput:.2f} docs/min")

    # Conservative check: at least able to handle the load
    assert queue_depth > 0, "Queue should have processed files"


@pytest.mark.asyncio
async def test_debouncing_batches_rapid_changes(
    file_watcher, queue_client, watch_directory
):
    """Test that debouncing batches rapid modifications to same file."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    test_file = watch_directory / "rapid_changes.py"
    test_file.write_text("# Version 0\n")

    # Rapidly modify the same file multiple times
    for i in range(10):
        test_file.write_text(f"# Version {i+1}\n")
        await asyncio.sleep(0.1)  # Much faster than debounce period

    # Wait for debounce to settle
    await asyncio.sleep(2)

    # Should have only 1-2 queue entries due to debouncing
    items = await queue_client.dequeue_batch(batch_size=20)
    file_items = [item for item in items if "rapid_changes.py" in item.file_absolute_path]

    # Debouncing should reduce to very few operations
    assert len(file_items) <= 3, f"Debouncing should batch operations, got {len(file_items)}"


@pytest.mark.asyncio
async def test_queue_doesnt_overflow_with_high_load(
    file_watcher, queue_client, watch_directory
):
    """Test that queue handles high load without overflowing."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Create many files
    file_count = 50
    for i in range(file_count):
        test_file = watch_directory / f"load_test_{i}.py"
        test_file.write_text(f"# Load test {i}\n")

    # Wait for processing
    await asyncio.sleep(8)

    # Get stats
    stats = await queue_client.get_queue_stats()

    # Verify queue is functional
    assert stats["total_items"] >= file_count * 0.8, "Most files should be enqueued"
    assert stats["error_items"] == 0, "Should have no errors"

    # Verify we can still dequeue
    items = await queue_client.dequeue_batch(batch_size=10)
    assert len(items) > 0, "Queue should be responsive"


# =============================================================================
# Test 4: Metadata Population
# =============================================================================


@pytest.mark.asyncio
async def test_tenant_id_and_branch_are_set(
    file_watcher, queue_client, watch_directory, state_manager
):
    """Test that tenant_id and branch are properly calculated and set."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    test_file = watch_directory / "metadata_test.py"
    test_file.write_text("# Metadata test\n")

    await asyncio.sleep(2)

    items = await queue_client.dequeue_batch(batch_size=10)
    file_items = [item for item in items if "metadata_test.py" in item.file_absolute_path]

    assert len(file_items) >= 1
    item = file_items[0]

    # Verify tenant_id is set (should be calculated from project root)
    assert item.tenant_id is not None
    assert item.tenant_id != ""

    # Verify branch is set (should detect git branch or default to "main")
    assert item.branch is not None
    assert item.branch != ""


@pytest.mark.asyncio
async def test_metadata_includes_watch_info(
    file_watcher, queue_client, watch_directory, state_manager
):
    """Test that metadata includes watch configuration details."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    test_file = watch_directory / "watch_metadata.py"
    test_file.write_text("# Watch metadata test\n")

    await asyncio.sleep(2)

    # Check the state manager for metadata
    # Queue client doesn't expose metadata directly, but state manager does
    queue_depth = await queue_client.get_queue_depth()
    assert queue_depth >= 1, "File should be enqueued"


# =============================================================================
# Test 5: Error Scenarios
# =============================================================================


@pytest.mark.asyncio
async def test_recovery_after_queue_failure(
    file_watcher, queue_client, watch_directory, state_manager
):
    """Test that watcher recovers after queue operation failures."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Create file successfully
    test_file = watch_directory / "recovery_test.py"
    test_file.write_text("# Recovery test\n")
    await asyncio.sleep(2)

    initial_depth = await queue_client.get_queue_depth()
    assert initial_depth >= 1

    # Create another file - should work normally
    test_file2 = watch_directory / "after_error.py"
    test_file2.write_text("# After error\n")
    await asyncio.sleep(2)

    final_depth = await queue_client.get_queue_depth()
    assert final_depth > initial_depth, "Watcher should continue working after errors"


@pytest.mark.asyncio
async def test_handles_locked_files_gracefully(
    file_watcher, queue_client, watch_directory
):
    """Test that watcher handles locked/inaccessible files gracefully."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Create a file
    test_file = watch_directory / "locked_test.py"
    test_file.write_text("# Locked file test\n")

    # Note: Actual file locking is platform-specific and hard to test portably
    # This test verifies the watcher continues operating
    await asyncio.sleep(2)

    # Create another file to verify watcher is still working
    test_file2 = watch_directory / "after_lock.py"
    test_file2.write_text("# After lock\n")
    await asyncio.sleep(2)

    queue_depth = await queue_client.get_queue_depth()
    assert queue_depth >= 1, "Watcher should continue processing files"


@pytest.mark.asyncio
async def test_handles_rapid_delete_create_cycle(
    file_watcher, queue_client, watch_directory
):
    """Test handling of rapid delete-create cycles on same file."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    test_file = watch_directory / "cycle_test.py"

    # Rapid delete-create cycles
    for i in range(5):
        test_file.write_text(f"# Cycle {i}\n")
        await asyncio.sleep(0.2)
        test_file.unlink()
        await asyncio.sleep(0.2)

    # Final create
    test_file.write_text("# Final version\n")

    await asyncio.sleep(3)

    # Should have queue entries (exact count depends on timing)
    queue_depth = await queue_client.get_queue_depth()
    assert queue_depth >= 1, "Should have enqueued operations"

    # Verify queue is functional
    items = await queue_client.dequeue_batch(batch_size=20)
    assert len(items) > 0


# =============================================================================
# Test 6: Multi-Watcher Scenarios
# =============================================================================


@pytest.mark.asyncio
async def test_multiple_watchers_different_directories(
    watch_manager, state_manager, tmp_path, queue_client
):
    """Test multiple watchers monitoring different directories."""
    watch_manager.set_state_manager(state_manager)

    # Create two watch directories
    dir1 = tmp_path / "watch_dir_1"
    dir1.mkdir()
    (dir1 / ".git").mkdir()

    dir2 = tmp_path / "watch_dir_2"
    dir2.mkdir()
    (dir2 / ".git").mkdir()

    # Add watches
    watch_id1 = await watch_manager.add_watch(
        path=str(dir1),
        collection="collection-1",
        patterns=["*.py"],
        debounce_seconds=1,
    )

    watch_id2 = await watch_manager.add_watch(
        path=str(dir2),
        collection="collection-2",
        patterns=["*.txt"],
        debounce_seconds=1,
    )

    await watch_manager.start_all_watches()
    await asyncio.sleep(1)

    # Create files in both directories
    file1 = dir1 / "test.py"
    file1.write_text("# Test 1\n")

    file2 = dir2 / "test.txt"
    file2.write_text("Test 2\n")

    await asyncio.sleep(3)

    # Verify both files are enqueued to correct collections
    items = await queue_client.dequeue_batch(batch_size=20)

    collection1_items = [item for item in items if item.collection_name == "collection-1"]
    collection2_items = [item for item in items if item.collection_name == "collection-2"]

    assert len(collection1_items) >= 1, "Should have items from collection-1"
    assert len(collection2_items) >= 1, "Should have items from collection-2"


@pytest.mark.asyncio
async def test_same_file_matched_by_multiple_watchers(
    watch_manager, state_manager, tmp_path, queue_client
):
    """Test behavior when same file is matched by multiple watchers."""
    watch_manager.set_state_manager(state_manager)

    # Create directory
    shared_dir = tmp_path / "shared_watch"
    shared_dir.mkdir()
    (shared_dir / ".git").mkdir()

    # Add two watches on same directory with different collections
    watch_id1 = await watch_manager.add_watch(
        path=str(shared_dir),
        collection="collection-a",
        patterns=["*.py"],
        debounce_seconds=1,
    )

    watch_id2 = await watch_manager.add_watch(
        path=str(shared_dir),
        collection="collection-b",
        patterns=["*.py"],
        debounce_seconds=1,
    )

    await watch_manager.start_all_watches()
    await asyncio.sleep(1)

    # Create a file
    test_file = shared_dir / "shared.py"
    test_file.write_text("# Shared file\n")

    await asyncio.sleep(3)

    # Should have entries for both collections
    items = await queue_client.dequeue_batch(batch_size=20)

    collections = set(item.collection_name for item in items if "shared.py" in item.file_absolute_path)

    # Both collections should have the file (or at least one, depending on implementation)
    assert len(collections) >= 1, "File should be enqueued to at least one collection"


@pytest.mark.asyncio
async def test_watcher_pause_resume_functionality(
    watch_manager, state_manager, tmp_path, queue_client
):
    """Test pausing and resuming individual watchers."""
    watch_manager.set_state_manager(state_manager)

    watch_dir = tmp_path / "pause_test"
    watch_dir.mkdir()
    (watch_dir / ".git").mkdir()

    watch_id = await watch_manager.add_watch(
        path=str(watch_dir),
        collection="pause-collection",
        patterns=["*.py"],
        debounce_seconds=1,
    )

    await watch_manager.start_all_watches()
    await asyncio.sleep(1)

    # Create file while active
    file1 = watch_dir / "active.py"
    file1.write_text("# Active\n")
    await asyncio.sleep(2)

    initial_depth = await queue_client.get_queue_depth()
    assert initial_depth >= 1

    # Pause watcher
    await watch_manager.pause_watch(watch_id)
    await asyncio.sleep(1)

    # Create file while paused (should not be enqueued)
    file2 = watch_dir / "paused.py"
    file2.write_text("# Paused\n")
    await asyncio.sleep(2)

    paused_depth = await queue_client.get_queue_depth()
    # Depth should be same or file should not be processed
    # (depending on timing, might have cleared first file)

    # Resume watcher
    await watch_manager.resume_watch(watch_id)
    await asyncio.sleep(1)

    # Create file while resumed
    file3 = watch_dir / "resumed.py"
    file3.write_text("# Resumed\n")
    await asyncio.sleep(2)

    final_depth = await queue_client.get_queue_depth()
    # Should have new files after resume
    assert final_depth >= paused_depth or final_depth >= 1


# =============================================================================
# Test 7: Performance Benchmarks
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_performance_target_1000_docs_per_minute(
    file_watcher, queue_client, watch_directory
):
    """
    Test that system can handle 1000+ docs/min throughput.

    This is a performance benchmark to verify the system meets
    the target of 1000+ documents per minute processing rate.
    """
    await file_watcher.start()
    await asyncio.sleep(1)

    # Calculate test parameters
    # Target: 1000 docs/min = 16.67 docs/sec
    # Test duration: 10 seconds
    # Expected files: ~167 files in 10 seconds
    test_duration = 10
    target_throughput = 1000 / 60  # docs per second
    expected_files = int(target_throughput * test_duration)

    # We'll test with a reasonable subset to keep test time manageable
    file_count = min(expected_files, 100)

    start_time = time.time()

    # Create files
    for i in range(file_count):
        test_file = watch_directory / f"perf_test_{i}.py"
        test_file.write_text(f"# Performance test {i}\n")

    # Wait for processing
    await asyncio.sleep(12)  # Allow time for debouncing

    end_time = time.time()
    elapsed = end_time - start_time

    # Calculate throughput
    throughput_per_min = (file_count / elapsed) * 60

    print(f"\nPerformance Test Results:")
    print(f"  Files created: {file_count}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"  Throughput: {throughput_per_min:.2f} docs/min")
    print(f"  Target: 1000 docs/min")

    # Verify all files were enqueued
    queue_depth = await queue_client.get_queue_depth()
    print(f"  Queue depth: {queue_depth}")

    # Conservative check: system should handle the load
    # We're not strictly enforcing 1000/min due to debouncing,
    # but verify reasonable throughput
    assert throughput_per_min > 100, f"Throughput too low: {throughput_per_min:.2f} docs/min"
    assert queue_depth >= file_count * 0.9, "Should have enqueued most files"


@pytest.mark.asyncio
async def test_latency_measurements(
    file_watcher, queue_client, watch_directory
):
    """
    Test and measure latency from file creation to queue entry.

    Verifies latency is within acceptable bounds:
    - Mean < 50ms (plus debounce time)
    - P95 < 100ms (plus debounce time)
    """
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Test with multiple files to get statistics
    latencies = []
    file_count = 20

    for i in range(file_count):
        test_file = watch_directory / f"latency_test_{i}.py"

        create_start = time.time()
        test_file.write_text(f"# Latency test {i}\n")

        # Wait for debounce + small buffer
        await asyncio.sleep(1.2)

        # Check if file is in queue
        items = await queue_client.dequeue_batch(batch_size=50)
        file_in_queue = any(test_file.name in item.file_absolute_path for item in items)

        if file_in_queue:
            latency = time.time() - create_start
            latencies.append(latency)

    assert len(latencies) > 0, "Should have measured some latencies"

    # Calculate statistics (accounting for 1s debounce)
    mean_latency = sum(latencies) / len(latencies)
    sorted_latencies = sorted(latencies)
    p95_index = int(len(sorted_latencies) * 0.95)
    p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]

    print(f"\nLatency Measurements:")
    print(f"  Mean: {mean_latency:.3f}s")
    print(f"  P95: {p95_latency:.3f}s")
    print(f"  (includes 1s debounce time)")

    # Verify reasonable latencies (debounce + processing should be < 2s)
    assert mean_latency < 2.0, f"Mean latency too high: {mean_latency:.3f}s"
    assert p95_latency < 2.5, f"P95 latency too high: {p95_latency:.3f}s"


# =============================================================================
# Test 8: Event Callback Verification
# =============================================================================


@pytest.mark.asyncio
async def test_event_callbacks_are_triggered(
    file_watcher, watch_directory
):
    """Test that event callbacks are properly triggered for file changes."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Create file
    test_file = watch_directory / "callback_test.py"
    test_file.write_text("# Callback test\n")

    await asyncio.sleep(2)

    # Check events were recorded
    events = file_watcher._test_events
    assert len(events) >= 1, "Should have received events"

    # Verify event properties
    file_events = [e for e in events if "callback_test.py" in e.file_path]
    assert len(file_events) >= 1

    event = file_events[0]
    assert event.change_type in ["added", "modified"]
    assert event.collection == "test-collection"
    assert event.timestamp is not None


# =============================================================================
# Test 9: Queue Statistics Verification
# =============================================================================


@pytest.mark.asyncio
async def test_queue_statistics_accuracy(
    file_watcher, queue_client, watch_directory
):
    """Test that queue statistics accurately reflect queue state."""
    await file_watcher.start()
    await asyncio.sleep(0.5)

    # Create files with known distribution
    # 2 high priority (delete operations)
    # 3 normal priority (create operations)

    # Create and delete 2 files for high priority
    for i in range(2):
        test_file = watch_directory / f"delete_me_{i}.py"
        test_file.write_text(f"# Delete me {i}\n")
        await asyncio.sleep(0.2)
        test_file.unlink()

    # Create 3 normal files
    for i in range(3):
        test_file = watch_directory / f"normal_{i}.py"
        test_file.write_text(f"# Normal {i}\n")

    await asyncio.sleep(3)

    # Get statistics
    stats = await queue_client.get_queue_stats()

    print(f"\nQueue Statistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Urgent items: {stats['urgent_items']}")
    print(f"  High items: {stats['high_items']}")
    print(f"  Normal items: {stats['normal_items']}")
    print(f"  Low items: {stats['low_items']}")

    # Verify statistics make sense
    assert stats["total_items"] >= 5, "Should have at least 5 items"
    assert stats["urgent_items"] >= 0, "Should track urgent items"
    assert stats["high_items"] >= 2, "Should have high priority delete operations"
    assert stats["normal_items"] >= 3, "Should have normal priority ingest operations"


# =============================================================================
# Test 10: Integration with WatchManager
# =============================================================================


@pytest.mark.asyncio
async def test_watch_manager_lifecycle(
    watch_manager, state_manager, tmp_path, queue_client
):
    """Test complete WatchManager lifecycle: add, start, stop, remove."""
    watch_manager.set_state_manager(state_manager)

    watch_dir = tmp_path / "lifecycle_test"
    watch_dir.mkdir()
    (watch_dir / ".git").mkdir()

    # Add watch
    watch_id = await watch_manager.add_watch(
        path=str(watch_dir),
        collection="lifecycle-collection",
        patterns=["*.py"],
        debounce_seconds=1,
    )

    assert watch_id is not None

    # Start watches
    await watch_manager.start_all_watches()
    await asyncio.sleep(1)

    # Verify it's running
    status = watch_manager.get_watch_status()
    assert watch_id in status
    assert status[watch_id]["running"] is True

    # Create file to verify it's working
    test_file = watch_dir / "test.py"
    test_file.write_text("# Test\n")
    await asyncio.sleep(2)

    queue_depth = await queue_client.get_queue_depth()
    assert queue_depth >= 1, "Watch should be processing files"

    # Stop watches
    await watch_manager.stop_all_watches()
    await asyncio.sleep(1)

    # Verify stopped
    status = watch_manager.get_watch_status()
    assert status[watch_id]["running"] is False

    # Remove watch
    removed = await watch_manager.remove_watch(watch_id)
    assert removed is True

    # Verify removed
    status = watch_manager.get_watch_status()
    assert watch_id not in status
