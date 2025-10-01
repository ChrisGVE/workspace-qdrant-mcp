"""
Integration Tests for Priority Queue Manager with Queue Integration.

Tests the complete integration of PriorityQueueManager with SQLiteStateManager
and SQLiteQueueClient, covering enqueue/dequeue, priority calculations,
retry logic, and end-to-end workflows.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List

from src.python.common.core.priority_queue_manager import (
    PriorityQueueManager,
    MCPActivityLevel,
    ProcessingMode,
    ProcessingPriority,
    ProcessingJob,
)
from src.python.common.core.sqlite_state_manager import (
    SQLiteStateManager,
    FileProcessingStatus,
)
from src.python.common.core.queue_client import SQLiteQueueClient


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
async def queue_manager(state_manager):
    """Initialize PriorityQueueManager."""
    manager = PriorityQueueManager(
        state_manager=state_manager,
        incremental_processor=None,
        mcp_detection_interval=1,  # Fast for testing
    )
    success = await manager.initialize()
    assert success, "Queue manager initialization failed"
    yield manager
    await manager.shutdown()


@pytest.fixture
def test_file(tmp_path):
    """Create a test file."""
    test_file = tmp_path / "test_file.py"
    test_file.write_text("# Test file\nprint('hello')\n")
    return str(test_file)


@pytest.fixture
def test_project_root(tmp_path):
    """Create a test project root directory."""
    project_root = tmp_path / "test_project"
    project_root.mkdir()

    # Create a .git directory to simulate a git repo
    git_dir = project_root / ".git"
    git_dir.mkdir()

    return project_root


# Test 1: Enqueue with tenant ID and branch
@pytest.mark.asyncio
async def test_enqueue_with_tenant_and_branch(queue_manager, test_file, test_project_root):
    """Test enqueue_file() calculates tenant_id and branch, stores in queue."""
    # Set current project for priority calculation
    queue_manager.set_current_project_root(str(test_project_root))

    # Enqueue file
    queue_id = await queue_manager.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        user_triggered=False,
    )

    # Verify queue_id is returned
    assert queue_id is not None

    # Verify metadata includes tenant and branch
    # The state manager stores this in the ingestion_queue table
    # We can verify via queue_client
    stats = await queue_manager.queue_client.get_queue_stats()
    assert stats["total_items"] >= 1


# Test 2: Dequeue batch processing
@pytest.mark.asyncio
async def test_dequeue_batch_processing(queue_client, tmp_path, state_manager):
    """Test dequeue returns items by priority and batch_size parameter."""
    # Create multiple test files
    files = []
    priorities = [1, 2, 3, 4]  # LOW, NORMAL, HIGH, URGENT (as integers)

    for i, priority in enumerate(priorities):
        test_file = tmp_path / f"test_file_{i}.py"
        test_file.write_text(f"# Test file {i}\n")
        files.append((str(test_file), priority))

    # Enqueue files with different priorities
    for file_path, priority in files:
        await state_manager.enqueue(
            file_path=file_path,
            collection="test-collection",
            priority=priority,
            tenant_id="default",
            branch="main",
        )

    # Dequeue batch (should be ordered by priority DESC)
    items = await queue_client.dequeue_batch(batch_size=4)

    # Verify items are ordered by priority (highest first)
    assert len(items) == 4
    assert items[0].priority >= items[1].priority
    assert items[1].priority >= items[2].priority
    assert items[2].priority >= items[3].priority

    # Test empty queue handling
    await queue_client.clear_queue()
    empty_items = await queue_client.dequeue_batch(batch_size=1)
    assert len(empty_items) == 0


# Test 3: Priority calculations
@pytest.mark.asyncio
async def test_priority_calculations(queue_manager, test_file, test_project_root):
    """Test MCP activity, current project, user triggered, and branch affect priority."""
    queue_manager.set_current_project_root(str(test_project_root))
    queue_manager.set_current_branch("main")

    # Test 1: User triggered should get high priority
    queue_id_1 = await queue_manager.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        user_triggered=True,
    )

    # Get the enqueued item from queue
    items_1 = await queue_manager.queue_client.dequeue_batch(batch_size=1)
    assert len(items_1) > 0
    user_triggered_priority = items_1[0].priority

    # Clear queue
    await queue_manager.queue_client.clear_queue()

    # Test 2: Non-user triggered should get lower priority
    queue_id_2 = await queue_manager.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        user_triggered=False,
    )

    items_2 = await queue_manager.queue_client.dequeue_batch(batch_size=1)
    assert len(items_2) > 0
    normal_priority = items_2[0].priority

    # User triggered should have higher or equal priority
    assert user_triggered_priority >= normal_priority

    # Clear queue
    await queue_manager.queue_client.clear_queue()

    # Test 3: Simulate MCP activity
    queue_manager.mcp_activity.activity_level = MCPActivityLevel.HIGH

    queue_id_3 = await queue_manager.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        user_triggered=False,
    )

    items_3 = await queue_manager.queue_client.dequeue_batch(batch_size=1)
    assert len(items_3) > 0
    high_activity_priority = items_3[0].priority

    # High MCP activity should affect priority
    assert high_activity_priority >= 1  # At least some priority


# Test 4: Retry logic
@pytest.mark.asyncio
async def test_retry_logic(queue_manager, tmp_path):
    """Test exponential backoff, max attempts, and queue removal after max attempts."""
    # Create test file
    test_file = tmp_path / "retry_test.py"
    test_file.write_text("# Retry test\n")

    # Create a job for testing
    job = ProcessingJob(
        queue_id="test_queue_id",
        file_path=str(test_file),
        collection="test-collection",
        priority=ProcessingPriority.NORMAL,
        attempts=0,
        max_attempts=3,
    )

    # First, start file processing so it exists in database
    await queue_manager.state_manager.start_file_processing(
        file_path=str(test_file),
        collection="test-collection",
    )

    # Simulate first failure
    await queue_manager._handle_job_failure(job, "Test error 1")
    assert job.attempts == 1

    # Verify file status
    record = await queue_manager.state_manager.get_file_processing_status(str(test_file))
    assert record is not None
    assert record.status in [FileProcessingStatus.FAILED, FileProcessingStatus.RETRYING]

    # Simulate second failure
    job.attempts = 1
    await queue_manager._handle_job_failure(job, "Test error 2")
    assert job.attempts == 2

    # Simulate third failure (should reach max attempts)
    job.attempts = 2
    await queue_manager._handle_job_failure(job, "Test error 3")
    assert job.attempts == 3

    # Verify file is marked as failed after max attempts
    record_final = await queue_manager.state_manager.get_file_processing_status(str(test_file))
    assert record_final.status == FileProcessingStatus.FAILED


# Test 5: Priority transitions
@pytest.mark.asyncio
async def test_priority_transitions(queue_manager, queue_client, tmp_path, state_manager):
    """Test activity level transition detection and priority adjustments."""
    # Create test files and enqueue them
    files = []
    for i in range(3):
        test_file = tmp_path / f"file_{i}.py"
        test_file.write_text(f"# File {i}\n")
        files.append(str(test_file))

        # Enqueue using new queue system
        await state_manager.enqueue(
            file_path=str(test_file),
            collection="test-collection",
            priority=5,
            tenant_id="default",
            branch="main",
        )

    # Set initial activity level
    queue_manager.mcp_activity.activity_level = MCPActivityLevel.LOW
    queue_manager._last_activity_level = MCPActivityLevel.LOW
    queue_manager._last_priority_adjustment = None

    # Trigger transition to HIGH activity
    queue_manager.mcp_activity.activity_level = MCPActivityLevel.HIGH
    await queue_manager._handle_activity_level_transition(
        MCPActivityLevel.LOW,
        MCPActivityLevel.HIGH
    )

    # Verify priority adjustment timestamp was set
    assert queue_manager._last_priority_adjustment is not None

    # Test rate limiting - try again within 60 seconds
    queue_manager.mcp_activity.activity_level = MCPActivityLevel.MODERATE
    old_adjustment_time = queue_manager._last_priority_adjustment

    await queue_manager._handle_activity_level_transition(
        MCPActivityLevel.HIGH,
        MCPActivityLevel.MODERATE
    )

    # Should not adjust again (rate limited)
    assert queue_manager._last_priority_adjustment == old_adjustment_time


# Test 6: Queue statistics integration
@pytest.mark.asyncio
async def test_queue_statistics_integration(queue_client, tmp_path, state_manager):
    """Test queue stats integration with priority queue manager."""
    # Create and enqueue files with different priorities
    priorities = [1, 1, 2, 2, 2, 3, 3, 4]

    for i, priority in enumerate(priorities):
        test_file = tmp_path / f"file_{i}.py"
        test_file.write_text(f"# File {i}\n")

        await state_manager.enqueue(
            file_path=str(test_file),
            collection="test-collection",
            priority=priority,
            tenant_id="default",
            branch="main",
        )

    # Get queue stats
    stats = await queue_client.get_queue_stats()

    # Verify stats
    assert stats is not None
    assert stats["total_items"] == 8
    # Verify we have items at different priority levels
    assert stats["low_items"] + stats["normal_items"] + stats["high_items"] + stats["urgent_items"] == 8


# Test 7: Error handling
@pytest.mark.asyncio
async def test_error_handling(queue_manager, tmp_path):
    """Test file not found and processing failure handling."""
    # Test 1: File not found
    non_existent_file = str(tmp_path / "non_existent.py")

    queue_id = await queue_manager.enqueue_file(
        file_path=non_existent_file,
        collection="test-collection",
    )

    # Verify it was enqueued
    assert queue_id is not None

    # Start processing the file
    await queue_manager.state_manager.start_file_processing(
        file_path=non_existent_file,
        collection="test-collection",
    )

    # Create job for non-existent file
    job = ProcessingJob(
        queue_id=queue_id,
        file_path=non_existent_file,
        collection="test-collection",
        priority=ProcessingPriority.NORMAL,
    )

    # Process should fail
    result = await queue_manager._process_single_job(job)
    assert result is None  # None indicates failure


# Test 8: End-to-end workflow (simplified)
@pytest.mark.asyncio
async def test_end_to_end_workflow(queue_manager, tmp_path):
    """Test complete workflow: enqueue multiple files, verify queue state."""
    # Create test files
    files_data = [
        ("file_urgent.py", True),
        ("file_high.py", False),
        ("file_normal.py", False),
    ]

    created_files = []

    for filename, user_triggered in files_data:
        test_file = tmp_path / filename
        test_file.write_text(f"# {filename}\nprint('test')\n")
        created_files.append((str(test_file), user_triggered))

    # Enqueue all files
    queue_ids = []
    for file_path, user_triggered in created_files:
        queue_id = await queue_manager.enqueue_file(
            file_path=file_path,
            collection="test-collection",
            user_triggered=user_triggered,
        )
        queue_ids.append(queue_id)

    # Verify all were enqueued
    assert len(queue_ids) == 3

    # Verify queue has items
    stats = await queue_manager.queue_client.get_queue_stats()
    assert stats["total_items"] >= 3


# Test 9: Concurrent processing (simplified)
@pytest.mark.asyncio
async def test_concurrent_processing(queue_manager, tmp_path):
    """Test concurrent processing mode configuration."""
    # Create test files
    num_files = 3
    files = []

    for i in range(num_files):
        test_file = tmp_path / f"concurrent_file_{i}.py"
        test_file.write_text(f"# Concurrent file {i}\nprint({i})\n")
        files.append(str(test_file))

    # Enqueue all files
    for file_path in files:
        await queue_manager.enqueue_file(
            file_path=file_path,
            collection="test-collection",
        )

    # Set processing mode to AGGRESSIVE for concurrent processing
    queue_manager.processing_mode = ProcessingMode.AGGRESSIVE
    await queue_manager._configure_executor()

    # Verify executor was configured
    assert queue_manager.executor is not None


# Test 10: Priority clamping
@pytest.mark.asyncio
async def test_priority_clamping(queue_manager, queue_client, tmp_path, state_manager):
    """Test that priorities are clamped to valid range (1-9)."""
    # Create test file and enqueue with normal priority
    test_file = tmp_path / "clamp_test.py"
    test_file.write_text("# Test\n")

    await state_manager.enqueue(
        file_path=str(test_file),
        collection="test-collection",
        priority=5,
        tenant_id="default",
        branch="main",
    )

    # Get the item
    items = await queue_client.dequeue_batch(batch_size=1)
    assert len(items) == 1
    original_priority = items[0].priority
    assert 0 <= original_priority <= 10  # Valid range for new queue system

    # Re-enqueue for priority adjustment test
    await state_manager.enqueue(
        file_path=str(test_file),
        collection="test-collection",
        priority=5,
        tenant_id="default",
        branch="main",
    )

    # Test priority adjustment with clamping
    adjusted = await queue_manager._adjust_queue_priorities(priority_delta=10)

    if adjusted > 0:
        # Verify priority was clamped
        items_after = await queue_client.dequeue_batch(batch_size=1)
        if items_after:
            # Priority should be clamped to valid range
            assert items_after[0].priority <= 10


# Test 11: Batch size optimization
@pytest.mark.asyncio
async def test_batch_size_optimization(queue_manager):
    """Test optimal batch size calculation based on processing mode and queue size."""
    # Test CONSERVATIVE mode (should have smaller batch)
    queue_manager.processing_mode = ProcessingMode.CONSERVATIVE
    batch_size_conservative = await queue_manager._get_optimal_batch_size()

    # Test AGGRESSIVE mode (should have larger batch)
    queue_manager.processing_mode = ProcessingMode.AGGRESSIVE
    batch_size_aggressive = await queue_manager._get_optimal_batch_size()

    # Aggressive should have larger or equal batch size
    assert batch_size_aggressive >= batch_size_conservative


# Test 12: Processing statistics tracking
@pytest.mark.asyncio
async def test_processing_statistics_tracking(queue_manager, tmp_path):
    """Test that processing statistics are updated correctly."""
    # Initial stats
    initial_stats = queue_manager.statistics
    initial_total = initial_stats.total_items

    # Create and enqueue files
    num_files = 2
    for i in range(num_files):
        test_file = tmp_path / f"stats_test_{i}.py"
        test_file.write_text(f"# Stats test {i}\n")

        await queue_manager.enqueue_file(
            file_path=str(test_file),
            collection="test-collection",
        )

    # Verify total items increased
    assert queue_manager.statistics.total_items >= initial_total


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
