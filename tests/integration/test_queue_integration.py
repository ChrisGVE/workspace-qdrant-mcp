"""
Integration Tests for Queue System End-to-End Workflows.

Tests complete integration of the queue system including:
- End-to-end file processing workflows
- Priority management and transitions
- Error handling and recovery
- Tool discovery integration
- Multi-tenant operations
- Branch switching scenarios

Test Coverage:
    - File change → queue → process → complete workflows
    - Error scenarios with retry and recovery
    - Priority transitions based on MCP activity
    - Branch switching and priority updates
    - Tool discovery integration
    - Concurrent processing scenarios
    - Real-world simulation tests
"""

import asyncio
import json
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from src.python.common.core.queue_client import (
    SQLiteQueueClient,
    QueueOperation,
    QueueItem,
)
from src.python.common.core.priority_queue_manager import (
    PriorityQueueManager,
    MCPActivityLevel,
    ProcessingMode,
    ProcessingPriority,
)
from src.python.common.core.sqlite_state_manager import (
    SQLiteStateManager,
    FileProcessingStatus,
)
from src.python.common.core.error_message_manager import ErrorMessageManager
from src.python.common.core.queue_connection import ConnectionConfig


# =============================================================================
# FIXTURES
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
async def initialized_db(temp_db):
    """Initialize database with all required schemas."""
    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA journal_mode=WAL")

    # Load queue schema
    queue_schema_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "python"
        / "common"
        / "core"
        / "queue_schema.sql"
    )
    with open(queue_schema_path, "r") as f:
        conn.executescript(f.read())

    # Load error messages schema
    error_schema_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "python"
        / "common"
        / "core"
        / "error_messages_schema.sql"
    )
    with open(error_schema_path, "r") as f:
        conn.executescript(f.read())

    # Load state manager schema
    state_schema_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "python"
        / "common"
        / "core"
        / "migrations"
        / "state_manager_schema.sql"
    )
    if state_schema_path.exists():
        with open(state_schema_path, "r") as f:
            conn.executescript(f.read())

    conn.commit()
    conn.close()

    yield temp_db


@pytest.fixture
async def queue_client(initialized_db):
    """Initialize SQLiteQueueClient."""
    client = SQLiteQueueClient(
        db_path=initialized_db,
        connection_config=ConnectionConfig(max_connections=5),
    )
    await client.initialize()
    yield client
    await client.close()


@pytest.fixture
async def state_manager(initialized_db):
    """Initialize SQLiteStateManager."""
    manager = SQLiteStateManager(db_path=initialized_db)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def priority_queue_manager(state_manager):
    """Initialize PriorityQueueManager."""
    manager = PriorityQueueManager(
        state_manager=state_manager,
        mcp_detection_interval=1,  # Short interval for testing
    )
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
def test_files(tmp_path):
    """Create multiple test files with various extensions."""
    files = []
    extensions = ["py", "md", "txt", "json", "yaml"]

    for i, ext in enumerate(extensions):
        test_file = tmp_path / f"test_file_{i}.{ext}"
        content = f"# Test file {i}\n" + ("test content\n" * 10)
        test_file.write_text(content)
        files.append(str(test_file))

    return files


@pytest.fixture
def mock_git_repo(tmp_path):
    """Create mock git repository structure."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()

    # Create .git directory
    (repo_path / ".git").mkdir()
    (repo_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    # Create test files
    src_dir = repo_path / "src"
    src_dir.mkdir()

    files = []
    for i in range(3):
        test_file = src_dir / f"module_{i}.py"
        test_file.write_text(f"# Module {i}\ndef func_{i}():\n    pass\n")
        files.append(str(test_file))

    return {
        "repo_path": str(repo_path),
        "files": files,
        "branch": "main",
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


async def wait_for_queue_processing(
    queue_client: SQLiteQueueClient,
    timeout: float = 5.0,
    target_depth: int = 0,
) -> bool:
    """Wait for queue to be processed to target depth."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        depth = await queue_client.get_queue_depth()
        if depth <= target_depth:
            return True
        await asyncio.sleep(0.1)

    return False


async def enqueue_test_files(
    queue_client: SQLiteQueueClient,
    files: List[str],
    collection: str = "test-collection",
    priority: int = 5,
) -> List[str]:
    """Enqueue multiple test files."""
    queue_ids = []

    for file_path in files:
        queue_id = await queue_client.enqueue_file(
            file_path=file_path,
            collection=collection,
            priority=priority,
        )
        queue_ids.append(queue_id)

    return queue_ids


async def get_queue_stats(queue_client: SQLiteQueueClient) -> Dict[str, Any]:
    """Get comprehensive queue statistics."""
    return await queue_client.get_queue_stats()


# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_basic_enqueue_dequeue_workflow(queue_client, test_files):
    """Test basic file enqueue and dequeue workflow."""
    # Enqueue files
    queue_ids = await enqueue_test_files(
        queue_client, test_files[:3], priority=5
    )

    assert len(queue_ids) == 3

    # Verify queue depth
    depth = await queue_client.get_queue_depth()
    assert depth == 3

    # Dequeue batch
    items = await queue_client.dequeue_batch(batch_size=3)
    assert len(items) == 3

    # Verify items are ordered by priority
    for item in items:
        assert item.priority == 5
        assert item.collection_name == "test-collection"


@pytest.mark.asyncio
async def test_end_to_end_file_processing(queue_client, test_files):
    """Test complete file processing workflow: enqueue → process → complete."""
    test_file = test_files[0]

    # Step 1: Enqueue file
    queue_id = await queue_client.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        priority=7,
    )

    assert queue_id == str(Path(test_file).resolve())

    # Step 2: Dequeue for processing
    items = await queue_client.dequeue_batch(batch_size=1)
    assert len(items) == 1
    assert items[0].file_absolute_path == queue_id

    # Step 3: Simulate processing
    await asyncio.sleep(0.1)  # Simulate work

    # Step 4: Mark complete
    processing_time_ms = 100.0
    success = await queue_client.mark_complete(
        file_path=queue_id,
        processing_time_ms=processing_time_ms,
    )

    assert success is True

    # Step 5: Verify removal from queue
    depth = await queue_client.get_queue_depth()
    assert depth == 0


@pytest.mark.asyncio
async def test_batch_processing_workflow(queue_client, test_files):
    """Test batch processing of multiple files."""
    # Enqueue multiple files with varying priorities
    priorities = [3, 7, 5, 9, 4]

    for file_path, priority in zip(test_files, priorities):
        await queue_client.enqueue_file(
            file_path=file_path,
            collection="test-collection",
            priority=priority,
        )

    # Dequeue batch (should be ordered by priority DESC)
    items = await queue_client.dequeue_batch(batch_size=5)
    assert len(items) == 5

    # Verify priority ordering
    priorities_received = [item.priority for item in items]
    assert priorities_received == sorted(priorities, reverse=True)

    # Process all items
    for item in items:
        await queue_client.mark_complete(
            file_path=item.file_absolute_path,
            processing_time_ms=50.0,
        )

    # Verify queue is empty
    depth = await queue_client.get_queue_depth()
    assert depth == 0


@pytest.mark.asyncio
async def test_multi_tenant_workflow(queue_client, test_files):
    """Test multi-tenant queue operations."""
    tenants = ["tenant-a", "tenant-b", "tenant-c"]

    # Enqueue different files for different tenants (file_path is PRIMARY KEY)
    file_idx = 0
    for tenant in tenants:
        for i in range(2):
            if file_idx < len(test_files):
                await queue_client.enqueue_file(
                    file_path=test_files[file_idx],
                    collection=f"{tenant}-collection",
                    tenant_id=tenant,
                    priority=5,
                )
                file_idx += 1

    # Total queue depth should be 6 (3 tenants * 2 files)
    total_depth = await queue_client.get_queue_depth()
    assert total_depth == 6

    # Dequeue for specific tenant
    tenant_items = await queue_client.dequeue_batch(
        batch_size=10,
        tenant_id="tenant-a",
    )

    assert len(tenant_items) == 2
    for item in tenant_items:
        assert item.tenant_id == "tenant-a"


@pytest.mark.asyncio
async def test_branch_specific_workflow(queue_client, test_files):
    """Test branch-specific queue operations."""
    branches = ["main", "develop", "feature/test"]

    # Enqueue files for different branches
    for branch in branches:
        await queue_client.enqueue_file(
            file_path=test_files[0],
            collection="test-collection",
            branch=branch,
            priority=5,
        )

    # Dequeue for specific branch
    main_items = await queue_client.dequeue_batch(
        batch_size=10,
        branch="main",
    )

    assert len(main_items) == 1
    assert main_items[0].branch == "main"

    # Dequeue for develop branch
    develop_items = await queue_client.dequeue_batch(
        batch_size=10,
        branch="develop",
    )

    assert len(develop_items) == 1
    assert develop_items[0].branch == "develop"


# =============================================================================
# ERROR SCENARIO TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_error_handling_with_retry(queue_client, test_files):
    """Test error handling and retry workflow."""
    test_file = test_files[0]

    # Enqueue file
    queue_id = await queue_client.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        priority=5,
    )

    # Simulate processing error
    error = ValueError("Processing failed: tool not found")
    should_retry, error_id = await queue_client.mark_error(
        file_path=queue_id,
        exception=error,
        error_context={"tool": "missing-parser"},
        max_retries=3,
    )

    assert should_retry is True
    assert error_id is not None

    # Verify item still in queue with updated retry count
    items = await queue_client.dequeue_batch(batch_size=1)
    assert len(items) == 1
    assert items[0].retry_count == 1
    assert items[0].error_message_id == error_id


@pytest.mark.asyncio
async def test_max_retries_exceeded(queue_client, test_files):
    """Test behavior when max retries is exceeded."""
    test_file = test_files[0]

    # Enqueue file
    queue_id = await queue_client.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        priority=5,
    )

    # Exhaust retries
    max_retries = 3
    for attempt in range(max_retries):
        error = ValueError(f"Attempt {attempt + 1} failed")
        should_retry, error_id = await queue_client.mark_error(
            file_path=queue_id,
            exception=error,
            max_retries=max_retries,
        )

        if attempt < max_retries - 1:
            assert should_retry is True
        else:
            assert should_retry is False

    # Verify item removed from queue
    depth = await queue_client.get_queue_depth()
    assert depth == 0


@pytest.mark.asyncio
async def test_tool_missing_recovery_workflow(queue_client, test_files):
    """Test workflow: tool missing → track error → tool available → reprocess."""
    test_file = test_files[0]

    # Step 1: Enqueue file
    queue_id = await queue_client.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        priority=5,
    )

    # Step 2: Dequeue and simulate tool missing error
    items = await queue_client.dequeue_batch(batch_size=1)
    assert len(items) == 1

    tool_error = FileNotFoundError("Parser tool not found: python-parser")
    should_retry, error_id = await queue_client.mark_error(
        file_path=queue_id,
        exception=tool_error,
        error_context={"missing_tool": "python-parser"},
        max_retries=3,
    )

    assert should_retry is True

    # Step 3: Simulate tool becoming available (boost priority)
    await queue_client.update_priority(file_path=queue_id, new_priority=9)

    # Step 4: Reprocess with higher priority
    items = await queue_client.dequeue_batch(batch_size=1)
    assert len(items) == 1
    assert items[0].priority == 9
    assert items[0].retry_count == 1

    # Step 5: Successful processing
    await queue_client.mark_complete(
        file_path=queue_id,
        processing_time_ms=150.0,
    )

    # Verify successful completion
    depth = await queue_client.get_queue_depth()
    assert depth == 0


@pytest.mark.asyncio
async def test_partial_batch_failure(queue_client, test_files):
    """Test handling of partial batch failures."""
    # Enqueue batch of files
    queue_ids = await enqueue_test_files(queue_client, test_files[:5], priority=5)

    # Dequeue batch
    items = await queue_client.dequeue_batch(batch_size=5)
    assert len(items) == 5

    # Process with some failures
    for i, item in enumerate(items):
        if i % 2 == 0:
            # Success
            await queue_client.mark_complete(
                file_path=item.file_absolute_path,
                processing_time_ms=100.0,
            )
        else:
            # Failure
            error = RuntimeError(f"Processing failed for item {i}")
            await queue_client.mark_error(
                file_path=item.file_absolute_path,
                exception=error,
                max_retries=3,
            )

    # Verify queue contains only failed items for retry
    depth = await queue_client.get_queue_depth()
    assert depth == 2  # Items at indices 1 and 3

    # Verify retry counts
    retry_items = await queue_client.dequeue_batch(batch_size=10)
    for item in retry_items:
        assert item.retry_count == 1


# =============================================================================
# PRIORITY TRANSITION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_priority_boost_on_mcp_activity(priority_queue_manager, test_files):
    """Test priority boost when MCP activity increases."""
    # Set initial low activity
    priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.INACTIVE
    priority_queue_manager._last_activity_level = MCPActivityLevel.INACTIVE

    # Enqueue files with normal priority
    for file_path in test_files[:3]:
        await priority_queue_manager.enqueue_file(
            file_path=file_path,
            collection="test-collection",
            user_triggered=False,
        )

    # Simulate MCP activity increase
    priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.HIGH

    # Trigger priority adjustment
    await priority_queue_manager._handle_activity_level_transition(
        MCPActivityLevel.INACTIVE,
        MCPActivityLevel.HIGH,
    )

    # Allow some time for adjustment
    await asyncio.sleep(0.2)

    # Verify processing mode changed
    assert priority_queue_manager.processing_mode in [
        ProcessingMode.AGGRESSIVE,
        ProcessingMode.BALANCED,
    ]


@pytest.mark.asyncio
async def test_priority_decrease_on_mcp_inactivity(
    priority_queue_manager, test_files
):
    """Test priority decrease when MCP activity decreases."""
    # Set initial high activity
    priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.HIGH
    priority_queue_manager._last_activity_level = MCPActivityLevel.HIGH
    priority_queue_manager.processing_mode = ProcessingMode.AGGRESSIVE

    # Enqueue files
    for file_path in test_files[:3]:
        await priority_queue_manager.enqueue_file(
            file_path=file_path,
            collection="test-collection",
        )

    # Simulate MCP activity decrease
    priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.INACTIVE

    # Trigger priority adjustment
    await priority_queue_manager._handle_activity_level_transition(
        MCPActivityLevel.HIGH,
        MCPActivityLevel.INACTIVE,
    )

    # Allow processing mode update
    await asyncio.sleep(0.2)

    # Verify processing mode changed to conservative
    assert priority_queue_manager.processing_mode == ProcessingMode.CONSERVATIVE


@pytest.mark.asyncio
async def test_branch_switching_priority_update(queue_client, test_files):
    """Test priority updates when switching branches."""
    test_file = test_files[0]

    # Enqueue file for main branch
    queue_id = await queue_client.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        branch="main",
        priority=5,
    )

    # Simulate branch switch - boost priority for current branch
    await queue_client.update_priority(file_path=queue_id, new_priority=8)

    # Verify priority updated
    items = await queue_client.dequeue_batch(batch_size=1, branch="main")
    assert len(items) == 1
    assert items[0].priority == 8


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_error_manager_integration(queue_client, test_files):
    """Test integration with ErrorMessageManager."""
    test_file = test_files[0]

    # Enqueue and fail with error
    queue_id = await queue_client.enqueue_file(
        file_path=test_file,
        collection="test-collection",
        priority=5,
    )

    error = RuntimeError("Integration test error")
    error_context = {
        "component": "integration_test",
        "operation": "test_processing",
    }

    should_retry, error_id = await queue_client.mark_error(
        file_path=queue_id,
        exception=error,
        error_context=error_context,
        max_retries=3,
    )

    assert should_retry is True
    assert error_id is not None

    # Verify error was recorded in error manager
    error_manager = queue_client.error_manager
    error_msg = await error_manager.get_error_message(error_id)

    assert error_msg is not None
    assert "Integration test error" in error_msg["message"]
    assert error_msg["context"]["component"] == "integration_test"


@pytest.mark.asyncio
async def test_statistics_collection_integration(queue_client, test_files):
    """Test integration with queue statistics collection."""
    # Enqueue multiple files
    for file_path in test_files[:5]:
        await queue_client.enqueue_file(
            file_path=file_path,
            collection="test-collection",
            priority=5,
        )

    # Get initial stats
    stats = await queue_client.get_queue_stats()
    assert stats["total_items"] == 5

    # Process some files
    items = await queue_client.dequeue_batch(batch_size=3)
    for item in items:
        await queue_client.mark_complete(
            file_path=item.file_absolute_path,
            processing_time_ms=100.0,
        )

    # Get updated stats
    stats = await queue_client.get_queue_stats()
    assert stats["total_items"] == 2


@pytest.mark.asyncio
async def test_collection_metadata_integration(queue_client, test_files):
    """Test integration with collection metadata management."""
    collection_name = "test-collection-metadata"

    # Register collection
    success = await queue_client.register_collection(
        collection_name=collection_name,
        collection_type="watched-dynamic",
        configuration={"auto_process": True, "retention_days": 30},
    )

    assert success is True

    # Enqueue file for this collection
    await queue_client.enqueue_file(
        file_path=test_files[0],
        collection=collection_name,
        priority=5,
    )

    # Get collection info
    info = await queue_client.get_collection_info(collection_name)

    assert info is not None
    assert info["collection_name"] == collection_name
    assert info["collection_type"] == "watched-dynamic"
    assert info["configuration"]["auto_process"] is True


# =============================================================================
# CONCURRENCY AND PERFORMANCE TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_concurrent_enqueue_operations(queue_client, test_files):
    """Test concurrent enqueue operations."""

    async def enqueue_batch(files, collection_prefix):
        tasks = []
        for i, file_path in enumerate(files):
            task = queue_client.enqueue_file(
                file_path=file_path,
                collection=f"{collection_prefix}-{i}",
                priority=5,
            )
            tasks.append(task)
        return await asyncio.gather(*tasks)

    # Enqueue concurrently from multiple "workers"
    results = await asyncio.gather(
        enqueue_batch(test_files[:3], "worker-1"),
        enqueue_batch(test_files[:3], "worker-2"),
        enqueue_batch(test_files[:3], "worker-3"),
    )

    # Verify all items enqueued
    depth = await queue_client.get_queue_depth()
    assert depth == 9  # 3 workers * 3 files


@pytest.mark.asyncio
async def test_concurrent_processing(queue_client, test_files):
    """Test concurrent dequeue and processing operations."""
    # Enqueue files
    for file_path in test_files:
        await queue_client.enqueue_file(
            file_path=file_path,
            collection="test-collection",
            priority=5,
        )

    async def process_batch():
        items = await queue_client.dequeue_batch(batch_size=2)
        for item in items:
            await asyncio.sleep(0.05)  # Simulate processing
            await queue_client.mark_complete(
                file_path=item.file_absolute_path,
                processing_time_ms=50.0,
            )
        return len(items)

    # Process concurrently
    results = await asyncio.gather(
        process_batch(),
        process_batch(),
        process_batch(),
    )

    # Verify all items processed
    total_processed = sum(results)
    assert total_processed == len(test_files)

    depth = await queue_client.get_queue_depth()
    assert depth == 0


@pytest.mark.asyncio
async def test_queue_depth_management(queue_client, test_files):
    """Test queue depth limits and overflow handling."""
    # Enqueue with depth limit
    items_data = [
        {"file_path": fp, "collection": "test-collection", "priority": i}
        for i, fp in enumerate(test_files[:5])
    ]

    # Test reject strategy
    with pytest.raises(ValueError, match="Queue depth limit"):
        await queue_client.enqueue_batch(
            items=items_data,
            max_queue_depth=3,
            overflow_strategy="reject",
        )

    # Clear queue
    await queue_client.clear_queue()

    # Test replace_lowest strategy
    successful, failed = await queue_client.enqueue_batch(
        items=items_data,
        max_queue_depth=3,
        overflow_strategy="replace_lowest",
    )

    # Should have 3 items with highest priorities
    depth = await queue_client.get_queue_depth()
    assert depth == 3


# =============================================================================
# REAL-WORLD SIMULATION TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_realistic_development_workflow(
    priority_queue_manager, mock_git_repo
):
    """Simulate realistic development workflow with file changes."""
    repo_files = mock_git_repo["files"]

    # Set current project context
    priority_queue_manager.set_current_project_root(mock_git_repo["repo_path"])
    priority_queue_manager.set_current_branch("main")

    # Simulate user editing files (high priority)
    for file_path in repo_files[:2]:
        queue_id = await priority_queue_manager.enqueue_file(
            file_path=file_path,
            collection="code-project",
            user_triggered=True,
        )
        assert queue_id is not None

    # Simulate background file watcher detecting changes (normal priority)
    for file_path in repo_files[2:]:
        queue_id = await priority_queue_manager.enqueue_file(
            file_path=file_path,
            collection="code-project",
            user_triggered=False,
        )
        assert queue_id is not None

    # Process high priority items first
    status = await priority_queue_manager.get_queue_status()
    assert status["statistics"]["total_items"] == 3


@pytest.mark.asyncio
async def test_multi_project_processing(queue_client, tmp_path):
    """Test processing files from multiple projects simultaneously."""
    # Create multiple project structures
    projects = []
    for i in range(3):
        project_dir = tmp_path / f"project_{i}"
        project_dir.mkdir()

        files = []
        for j in range(2):
            test_file = project_dir / f"file_{j}.py"
            test_file.write_text(f"# Project {i}, File {j}\n")
            files.append(str(test_file))

        projects.append({
            "tenant_id": f"project-{i}",
            "files": files,
        })

    # Enqueue files from all projects
    for project in projects:
        for file_path in project["files"]:
            await queue_client.enqueue_file(
                file_path=file_path,
                collection=f"{project['tenant_id']}-code",
                tenant_id=project["tenant_id"],
                priority=5,
            )

    # Verify total queue depth
    depth = await queue_client.get_queue_depth()
    assert depth == 6  # 3 projects * 2 files

    # Process each project independently
    for project in projects:
        items = await queue_client.dequeue_batch(
            batch_size=10,
            tenant_id=project["tenant_id"],
        )

        assert len(items) == 2

        for item in items:
            assert item.tenant_id == project["tenant_id"]
            await queue_client.mark_complete(
                file_path=item.file_absolute_path,
                processing_time_ms=100.0,
            )

    # Verify all processed
    depth = await queue_client.get_queue_depth()
    assert depth == 0


@pytest.mark.asyncio
async def test_error_recovery_after_restart(queue_client, test_files):
    """Test queue recovery after simulated restart."""
    # Enqueue files
    queue_ids = await enqueue_test_files(queue_client, test_files[:3], priority=5)

    # Mark some as errored
    await queue_client.mark_error(
        file_path=queue_ids[0],
        exception=RuntimeError("Crash during processing"),
        max_retries=3,
    )

    # Simulate restart by closing and reopening
    await queue_client.close()

    new_client = SQLiteQueueClient(db_path=queue_client.connection_pool.db_path)
    await new_client.initialize()

    try:
        # Verify queue state preserved
        depth = await new_client.get_queue_depth()
        assert depth == 3  # All items still in queue

        # Verify retry count preserved
        items = await new_client.dequeue_batch(batch_size=3)
        errored_item = next(
            item for item in items if item.file_absolute_path == queue_ids[0]
        )
        assert errored_item.retry_count == 1

    finally:
        await new_client.close()
