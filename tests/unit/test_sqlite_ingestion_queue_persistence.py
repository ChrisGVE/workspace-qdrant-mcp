"""
Ingestion Queue Persistence Tests for SQLite State Manager.

Tests the persistence of ingestion queue items across database restarts,
priority ordering, filtering, metadata serialization, and concurrent access.
"""

import asyncio
import json
import sqlite3
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from src.python.common.core.sqlite_state_manager import (
    ProcessingPriority,
    ProcessingQueueItem,
    SQLiteStateManager,
)


@pytest.fixture
async def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    state_manager = SQLiteStateManager(db_path=db_path)
    await state_manager.initialize()

    yield state_manager

    await state_manager.close()
    # Clean up
    Path(db_path).unlink(missing_ok=True)
    Path(db_path + "-wal").unlink(missing_ok=True)
    Path(db_path + "-shm").unlink(missing_ok=True)


@pytest.fixture
async def db_path_only():
    """Provide just a database path for restart tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Clean up
    Path(db_path).unlink(missing_ok=True)
    Path(db_path + "-wal").unlink(missing_ok=True)
    Path(db_path + "-shm").unlink(missing_ok=True)


class TestIngestionQueuePersistence:
    """Test ingestion queue item persistence across database restarts."""

    @pytest.mark.asyncio
    async def test_queue_items_survive_restart(self, db_path_only):
        """Test that queue items persist across database restarts."""
        # Phase 1: Create and enqueue items
        state_manager = SQLiteStateManager(db_path=db_path_only)
        await state_manager.initialize()

        files_to_enqueue = [
            ("/path/to/file1.py", "collection1", 5, "tenant1", "main", {"key": "value1"}),
            ("/path/to/file2.py", "collection2", 7, "tenant2", "dev", {"key": "value2"}),
            ("/path/to/file3.py", "collection1", 3, "tenant1", "main", None),
        ]

        for file_path, collection, priority, tenant_id, branch, metadata in files_to_enqueue:
            await state_manager.enqueue(file_path, collection, priority, tenant_id, branch, metadata)

        await state_manager.close()

        # Phase 2: Reopen database and verify items exist
        state_manager2 = SQLiteStateManager(db_path=db_path_only)
        await state_manager2.initialize()

        queue_depth = await state_manager2.get_queue_depth()
        assert queue_depth == 3, "All queue items should persist after restart"

        items = await state_manager2.dequeue(batch_size=10)
        assert len(items) == 3, "All items should be retrievable"

        # Verify metadata survived serialization
        item_with_metadata = next(item for item in items if item.metadata is not None)
        assert "key" in item_with_metadata.metadata
        assert item_with_metadata.metadata["key"] in ["value1", "value2"]

        await state_manager2.close()

    @pytest.mark.asyncio
    async def test_priority_ordering_persists(self, db_path_only):
        """Test that priority-based ordering is maintained after restart."""
        # Phase 1: Enqueue items with different priorities
        state_manager = SQLiteStateManager(db_path=db_path_only)
        await state_manager.initialize()

        # Enqueue in random order
        await state_manager.enqueue("/path/low.py", "col", 2, "tenant", "main")  # LOW
        await state_manager.enqueue("/path/urgent.py", "col", 9, "tenant", "main")  # URGENT
        await state_manager.enqueue("/path/normal.py", "col", 5, "tenant", "main")  # NORMAL
        await state_manager.enqueue("/path/high.py", "col", 7, "tenant", "main")  # HIGH

        await state_manager.close()

        # Phase 2: Reopen and verify priority order
        state_manager2 = SQLiteStateManager(db_path=db_path_only)
        await state_manager2.initialize()

        items = await state_manager2.dequeue(batch_size=10)

        # Should be ordered: URGENT (9), HIGH (7), NORMAL (5), LOW (2)
        assert len(items) == 4
        assert "urgent" in items[0].file_path
        assert "high" in items[1].file_path
        assert "normal" in items[2].file_path
        assert "low" in items[3].file_path

        await state_manager2.close()

    @pytest.mark.asyncio
    async def test_tenant_filtering_persists(self, db_path_only):
        """Test that tenant_id filtering works across restarts."""
        # Phase 1: Enqueue items for different tenants
        state_manager = SQLiteStateManager(db_path=db_path_only)
        await state_manager.initialize()

        await state_manager.enqueue("/path/tenant1_a.py", "col", 5, "tenant1", "main")
        await state_manager.enqueue("/path/tenant2_a.py", "col", 5, "tenant2", "main")
        await state_manager.enqueue("/path/tenant1_b.py", "col", 5, "tenant1", "main")
        await state_manager.enqueue("/path/tenant2_b.py", "col", 5, "tenant2", "main")

        await state_manager.close()

        # Phase 2: Reopen and filter by tenant
        state_manager2 = SQLiteStateManager(db_path=db_path_only)
        await state_manager2.initialize()

        tenant1_items = await state_manager2.dequeue(batch_size=10, tenant_id="tenant1")
        assert len(tenant1_items) == 2
        assert all("tenant1" in item.file_path for item in tenant1_items)

        tenant2_items = await state_manager2.dequeue(batch_size=10, tenant_id="tenant2")
        assert len(tenant2_items) == 2
        assert all("tenant2" in item.file_path for item in tenant2_items)

        await state_manager2.close()

    @pytest.mark.asyncio
    async def test_branch_filtering_persists(self, db_path_only):
        """Test that branch filtering works across restarts."""
        # Phase 1: Enqueue items for different branches
        state_manager = SQLiteStateManager(db_path=db_path_only)
        await state_manager.initialize()

        await state_manager.enqueue("/path/main_a.py", "col", 5, "tenant", "main")
        await state_manager.enqueue("/path/dev_a.py", "col", 5, "tenant", "dev")
        await state_manager.enqueue("/path/main_b.py", "col", 5, "tenant", "main")
        await state_manager.enqueue("/path/feature.py", "col", 5, "tenant", "feature/xyz")

        await state_manager.close()

        # Phase 2: Reopen and filter by branch
        state_manager2 = SQLiteStateManager(db_path=db_path_only)
        await state_manager2.initialize()

        main_items = await state_manager2.dequeue(batch_size=10, branch="main")
        assert len(main_items) == 2
        assert all("main" in item.file_path for item in main_items)

        dev_items = await state_manager2.dequeue(batch_size=10, branch="dev")
        assert len(dev_items) == 1
        assert "dev" in dev_items[0].file_path

        await state_manager2.close()

    @pytest.mark.asyncio
    async def test_retry_count_preservation(self, db_path_only):
        """Test that retry_count is preserved across restarts."""
        # Phase 1: Enqueue item with retry_count
        state_manager = SQLiteStateManager(db_path=db_path_only)
        await state_manager.initialize()

        file_path = "/path/retried.py"
        await state_manager.enqueue(file_path, "col", 5, "tenant", "main")

        # Manually update retry_count to simulate retries
        with state_manager._lock:
            state_manager.connection.execute(
                "UPDATE ingestion_queue SET retry_count = ? WHERE file_absolute_path = ?",
                (3, str(Path(file_path).resolve()))
            )
            state_manager.connection.commit()

        await state_manager.close()

        # Phase 2: Reopen and verify retry_count
        state_manager2 = SQLiteStateManager(db_path=db_path_only)
        await state_manager2.initialize()

        items = await state_manager2.dequeue(batch_size=1)
        assert len(items) == 1
        assert items[0].attempts == 3, "Retry count should persist"

        await state_manager2.close()

    @pytest.mark.asyncio
    async def test_queue_depth_calculations(self, temp_db):
        """Test queue depth calculations with various filters."""
        # Enqueue items with different tenant/branch combinations
        await temp_db.enqueue("/path/t1_main_a.py", "col", 5, "tenant1", "main")
        await temp_db.enqueue("/path/t1_main_b.py", "col", 5, "tenant1", "main")
        await temp_db.enqueue("/path/t1_dev.py", "col", 5, "tenant1", "dev")
        await temp_db.enqueue("/path/t2_main.py", "col", 5, "tenant2", "main")

        # Test total depth
        total_depth = await temp_db.get_queue_depth()
        assert total_depth == 4

        # Test tenant filtering
        t1_depth = await temp_db.get_queue_depth(tenant_id="tenant1")
        assert t1_depth == 3

        t2_depth = await temp_db.get_queue_depth(tenant_id="tenant2")
        assert t2_depth == 1

        # Test branch filtering
        main_depth = await temp_db.get_queue_depth(branch="main")
        assert main_depth == 3

        dev_depth = await temp_db.get_queue_depth(branch="dev")
        assert dev_depth == 1

        # Test combined filtering
        t1_main_depth = await temp_db.get_queue_depth(tenant_id="tenant1", branch="main")
        assert t1_main_depth == 2

    @pytest.mark.asyncio
    async def test_dequeue_with_filters(self, temp_db):
        """Test dequeue operations with tenant and branch filters."""
        # Enqueue items
        await temp_db.enqueue("/path/t1_main.py", "col", 8, "tenant1", "main")
        await temp_db.enqueue("/path/t2_main.py", "col", 6, "tenant2", "main")
        await temp_db.enqueue("/path/t1_dev.py", "col", 7, "tenant1", "dev")

        # Dequeue for tenant1 only
        items = await temp_db.dequeue(batch_size=10, tenant_id="tenant1")
        assert len(items) == 2
        assert all(item.file_path.startswith("/path/t1_") for item in items)

        # Dequeue for main branch only
        items = await temp_db.dequeue(batch_size=10, branch="main")
        assert len(items) == 2
        assert all("main" in item.file_path for item in items)

        # Dequeue with both filters
        items = await temp_db.dequeue(batch_size=10, tenant_id="tenant1", branch="dev")
        assert len(items) == 1
        assert items[0].file_path == "/path/t1_dev.py"

    @pytest.mark.asyncio
    async def test_unique_constraint_handling(self, temp_db):
        """Test UNIQUE constraint handling on file_absolute_path."""
        file_path = "/path/duplicate.py"

        # First enqueue
        queue_id1 = await temp_db.enqueue(file_path, "col1", 5, "tenant1", "main", {"version": 1})

        # Second enqueue with same path should update priority
        queue_id2 = await temp_db.enqueue(file_path, "col2", 8, "tenant2", "dev", {"version": 2})

        # Should still have only one item
        depth = await temp_db.get_queue_depth()
        assert depth == 1, "Duplicate path should update existing item, not create new one"

        # Verify the priority was updated (higher priority wins)
        items = await temp_db.dequeue(batch_size=1)
        assert len(items) == 1
        # Priority 8 maps to HIGH (6-8 range)
        assert items[0].priority == ProcessingPriority.HIGH

    @pytest.mark.asyncio
    async def test_metadata_serialization(self, db_path_only):
        """Test that complex metadata survives JSON serialization."""
        # Phase 1: Enqueue with complex metadata
        state_manager = SQLiteStateManager(db_path=db_path_only)
        await state_manager.initialize()

        complex_metadata = {
            "nested": {"key": "value", "list": [1, 2, 3]},
            "types": {"str": "text", "int": 42, "float": 3.14, "bool": True, "none": None},
            "unicode": "こんにちは世界",
        }

        await state_manager.enqueue(
            "/path/complex.py",
            "col",
            5,
            "tenant",
            "main",
            complex_metadata
        )

        await state_manager.close()

        # Phase 2: Reopen and verify metadata
        state_manager2 = SQLiteStateManager(db_path=db_path_only)
        await state_manager2.initialize()

        items = await state_manager2.dequeue(batch_size=1)
        assert len(items) == 1

        metadata = items[0].metadata
        assert metadata["nested"]["key"] == "value"
        assert metadata["nested"]["list"] == [1, 2, 3]
        assert metadata["types"]["int"] == 42
        assert metadata["types"]["float"] == 3.14
        assert metadata["types"]["bool"] is True
        assert metadata["types"]["none"] is None
        assert metadata["unicode"] == "こんにちは世界"

        await state_manager2.close()

    @pytest.mark.asyncio
    async def test_concurrent_enqueue_operations(self, temp_db):
        """Test concurrent enqueue operations from multiple threads."""
        results = []
        errors = []

        async def enqueue_task(index: int):
            try:
                await temp_db.enqueue(
                    f"/path/concurrent_{index}.py",
                    "col",
                    index % 10,  # Vary priority
                    f"tenant_{index % 3}",  # 3 different tenants
                    f"branch_{index % 2}",  # 2 different branches
                    {"index": index}
                )
                results.append(index)
            except Exception as e:
                errors.append(e)

        # Launch 50 concurrent enqueue operations
        tasks = [enqueue_task(i) for i in range(50)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"No errors should occur during concurrent enqueue: {errors}"
        assert len(results) == 50, "All enqueue operations should succeed"

        # Verify all items are in queue
        depth = await temp_db.get_queue_depth()
        assert depth == 50, "All items should be in queue"

    @pytest.mark.asyncio
    async def test_concurrent_dequeue_operations(self, temp_db):
        """Test concurrent dequeue operations from multiple tasks."""
        # Enqueue items
        for i in range(20):
            await temp_db.enqueue(
                f"/path/dequeue_{i}.py",
                "col",
                5,
                "tenant",
                "main"
            )

        dequeued_items: List[ProcessingQueueItem] = []
        errors = []

        async def dequeue_task():
            try:
                items = await temp_db.dequeue(batch_size=5)
                dequeued_items.extend(items)
            except Exception as e:
                errors.append(e)

        # Launch 10 concurrent dequeue operations
        tasks = [dequeue_task() for _ in range(10)]
        await asyncio.gather(*tasks)

        assert len(errors) == 0, f"No errors should occur during concurrent dequeue: {errors}"
        # Note: dequeued_items may contain duplicates since we're just reading, not removing
        # This is expected behavior for the dequeue method

    @pytest.mark.asyncio
    async def test_remove_from_queue(self, temp_db):
        """Test removing items from queue."""
        # Enqueue items
        file_path1 = "/path/remove1.py"
        file_path2 = "/path/remove2.py"

        await temp_db.enqueue(file_path1, "col", 5, "tenant", "main")
        await temp_db.enqueue(file_path2, "col", 5, "tenant", "main")

        # Verify both in queue
        depth = await temp_db.get_queue_depth()
        assert depth == 2

        # Remove first item (queue_id is the absolute path)
        queue_id = str(Path(file_path1).resolve())
        removed = await temp_db.remove_from_queue(queue_id)
        assert removed is True

        # Verify only one item remains
        depth = await temp_db.get_queue_depth()
        assert depth == 1

        # Try to remove again (should fail)
        removed = await temp_db.remove_from_queue(queue_id)
        assert removed is False

    @pytest.mark.asyncio
    async def test_priority_timestamp_ordering(self, temp_db):
        """Test that items are ordered by priority DESC, then timestamp ASC."""
        # Enqueue items with same priority but different timestamps
        await temp_db.enqueue("/path/first.py", "col", 5, "tenant", "main")
        await asyncio.sleep(0.01)  # Ensure different timestamp
        await temp_db.enqueue("/path/second.py", "col", 5, "tenant", "main")
        await asyncio.sleep(0.01)
        await temp_db.enqueue("/path/third.py", "col", 5, "tenant", "main")

        # Enqueue a higher priority item after all others
        await asyncio.sleep(0.01)
        await temp_db.enqueue("/path/urgent.py", "col", 9, "tenant", "main")

        # Dequeue should get urgent first, then by timestamp order
        items = await temp_db.dequeue(batch_size=10)
        assert len(items) == 4
        assert "urgent" in items[0].file_path
        assert "first" in items[1].file_path
        assert "second" in items[2].file_path
        assert "third" in items[3].file_path

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, db_path_only):
        """Test that WAL mode is properly enabled for crash resistance."""
        state_manager = SQLiteStateManager(db_path=db_path_only)
        await state_manager.initialize()

        # Verify WAL mode
        with state_manager._lock:
            cursor = state_manager.connection.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.upper() == "WAL", "Database should be in WAL mode"

        await state_manager.close()
