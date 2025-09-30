"""
Comprehensive unit tests for SQLiteQueueClient operations.

Tests cover all queue operations with edge cases and error conditions:
- enqueue_file() - test deduplication, priority updates, metadata handling
- dequeue_batch() - test priority ordering, batch size, tenant/branch filtering
- update_priority() - test single and bulk updates, validation
- mark_complete() - test successful completion and not-found cases
- mark_error() - test retry logic, max retries, error tracking
- get_queue_stats() - test statistics accuracy and filtering
- enqueue_batch() - test bulk operations, overflow strategies
- purge_completed_items() - test retention policy cleanup
- get_queue_depth() - test queue depth calculations with filters
- clear_queue() - test queue clearing with various filters
- register_collection() and get_collection_info() - test metadata management
- Transaction rollback scenarios - test atomic operations
"""

import asyncio
import json
import pytest
import sqlite3
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any

from src.python.common.core.queue_client import (
    SQLiteQueueClient,
    QueueOperation,
    QueueItem,
)
from src.python.common.core.queue_connection import ConnectionConfig


class TestQueueItem:
    """Test QueueItem creation and serialization."""

    def test_queue_item_creation(self):
        """Test basic queue item creation with defaults."""
        item = QueueItem(
            file_absolute_path="/test/file.py",
            collection_name="test-collection",
        )

        assert item.file_absolute_path == "/test/file.py"
        assert item.collection_name == "test-collection"
        assert item.tenant_id == "default"
        assert item.branch == "main"
        assert item.operation == QueueOperation.INGEST
        assert item.priority == 5
        assert item.retry_count == 0
        assert item.retry_from is None
        assert item.error_message_id is None
        assert isinstance(item.queued_timestamp, datetime)

    def test_queue_item_full_params(self):
        """Test queue item with all parameters specified."""
        timestamp = datetime.now(timezone.utc)
        item = QueueItem(
            file_absolute_path="/test/file.py",
            collection_name="test-collection",
            tenant_id="tenant1",
            branch="dev",
            operation=QueueOperation.UPDATE,
            priority=8,
            queued_timestamp=timestamp,
            retry_count=2,
            retry_from="/old/path.py",
            error_message_id=123,
        )

        assert item.tenant_id == "tenant1"
        assert item.branch == "dev"
        assert item.operation == QueueOperation.UPDATE
        assert item.priority == 8
        assert item.queued_timestamp == timestamp
        assert item.retry_count == 2
        assert item.retry_from == "/old/path.py"
        assert item.error_message_id == 123

    def test_queue_item_operation_from_string(self):
        """Test queue item accepts operation as string."""
        item = QueueItem(
            file_absolute_path="/test/file.py",
            collection_name="test-collection",
            operation="delete",
        )

        assert item.operation == QueueOperation.DELETE

    def test_queue_item_to_dict(self):
        """Test queue item serialization to dictionary."""
        item = QueueItem(
            file_absolute_path="/test/file.py",
            collection_name="test-collection",
            tenant_id="tenant1",
            priority=7,
        )

        item_dict = item.to_dict()

        assert item_dict["file_absolute_path"] == "/test/file.py"
        assert item_dict["collection_name"] == "test-collection"
        assert item_dict["tenant_id"] == "tenant1"
        assert item_dict["priority"] == 7
        assert item_dict["operation"] == "ingest"
        assert "queued_timestamp" in item_dict


class TestSQLiteQueueClient:
    """Test SQLiteQueueClient operations."""

    @pytest.fixture
    async def queue_client(self):
        """Create a temporary queue client for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_queue.db"
            client = SQLiteQueueClient(str(db_path))
            await client.initialize()
            
            # Create database schema
            schema_path = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "queue_schema.sql"
            with open(schema_path) as f:
                schema_sql = f.read()
            
            async with client.connection_pool.get_connection_async() as conn:
                conn.executescript(schema_sql)
                conn.commit()
            yield client
            await client.close()

    @pytest.mark.asyncio
    async def test_enqueue_file_basic(self, queue_client):
        """Test basic file enqueue operation."""
        file_path = await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
            priority=5,
        )

        assert file_path == str(Path("/test/file.py").resolve())

        # Verify item is in queue
        items = await queue_client.dequeue_batch(batch_size=1)
        assert len(items) == 1
        assert items[0].file_absolute_path == str(Path("/test/file.py").resolve())
        assert items[0].collection_name == "test-collection"
        assert items[0].priority == 5

    @pytest.mark.asyncio
    async def test_enqueue_file_with_all_params(self, queue_client):
        """Test enqueue with all parameters."""
        file_path = await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
            tenant_id="tenant1",
            branch="dev",
            operation=QueueOperation.UPDATE,
            priority=8,
            retry_from="/old/file.py",
        )

        items = await queue_client.dequeue_batch(batch_size=1)
        assert len(items) == 1
        assert items[0].tenant_id == "tenant1"
        assert items[0].branch == "dev"
        assert items[0].operation == QueueOperation.UPDATE
        assert items[0].priority == 8
        assert items[0].retry_from == "/old/file.py"

    @pytest.mark.asyncio
    async def test_enqueue_file_duplicate_raises_error(self, queue_client):
        """Test enqueueing the same file twice raises IntegrityError."""
        await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
        )

        with pytest.raises(sqlite3.IntegrityError):
            await queue_client.enqueue_file(
                file_path="/test/file.py",
                collection="test-collection",
            )

    @pytest.mark.asyncio
    async def test_enqueue_file_invalid_priority(self, queue_client):
        """Test enqueue with invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await queue_client.enqueue_file(
                file_path="/test/file.py",
                collection="test-collection",
                priority=11,
            )

        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await queue_client.enqueue_file(
                file_path="/test/file.py",
                collection="test-collection",
                priority=-1,
            )

    @pytest.mark.asyncio
    async def test_enqueue_file_normalizes_path(self, queue_client):
        """Test enqueue normalizes file paths."""
        # Use relative path
        file_path = await queue_client.enqueue_file(
            file_path="./test/file.py",
            collection="test-collection",
        )

        # Should be resolved to absolute path
        assert Path(file_path).is_absolute()

    @pytest.mark.asyncio
    async def test_dequeue_batch_basic(self, queue_client):
        """Test basic dequeue operation."""
        # Enqueue multiple items
        for i in range(5):
            await queue_client.enqueue_file(
                file_path=f"/test/file{i}.py",
                collection="test-collection",
            )

        items = await queue_client.dequeue_batch(batch_size=3)
        assert len(items) == 3
        assert all(isinstance(item, QueueItem) for item in items)

    @pytest.mark.asyncio
    async def test_dequeue_batch_priority_ordering(self, queue_client):
        """Test dequeue returns items ordered by priority DESC."""
        # Enqueue with different priorities
        await queue_client.enqueue_file(
            file_path="/test/low.py",
            collection="test-collection",
            priority=2,
        )
        await queue_client.enqueue_file(
            file_path="/test/high.py",
            collection="test-collection",
            priority=9,
        )
        await queue_client.enqueue_file(
            file_path="/test/medium.py",
            collection="test-collection",
            priority=5,
        )

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 3
        assert items[0].file_absolute_path == str(Path("/test/high.py").resolve())
        assert items[1].file_absolute_path == str(Path("/test/medium.py").resolve())
        assert items[2].file_absolute_path == str(Path("/test/low.py").resolve())

    @pytest.mark.asyncio
    async def test_dequeue_batch_with_tenant_filter(self, queue_client):
        """Test dequeue with tenant_id filtering."""
        await queue_client.enqueue_file(
            file_path="/test/tenant1.py",
            collection="test-collection",
            tenant_id="tenant1",
        )
        await queue_client.enqueue_file(
            file_path="/test/tenant2.py",
            collection="test-collection",
            tenant_id="tenant2",
        )
        await queue_client.enqueue_file(
            file_path="/test/default.py",
            collection="test-collection",
        )

        items = await queue_client.dequeue_batch(batch_size=10, tenant_id="tenant1")
        assert len(items) == 1
        assert items[0].tenant_id == "tenant1"

    @pytest.mark.asyncio
    async def test_dequeue_batch_with_branch_filter(self, queue_client):
        """Test dequeue with branch filtering."""
        await queue_client.enqueue_file(
            file_path="/test/main.py",
            collection="test-collection",
            branch="main",
        )
        await queue_client.enqueue_file(
            file_path="/test/dev.py",
            collection="test-collection",
            branch="dev",
        )

        items = await queue_client.dequeue_batch(batch_size=10, branch="dev")
        assert len(items) == 1
        assert items[0].branch == "dev"

    @pytest.mark.asyncio
    async def test_dequeue_batch_combined_filters(self, queue_client):
        """Test dequeue with multiple filters."""
        await queue_client.enqueue_file(
            file_path="/test/match.py",
            collection="test-collection",
            tenant_id="tenant1",
            branch="dev",
        )
        await queue_client.enqueue_file(
            file_path="/test/wrong_tenant.py",
            collection="test-collection",
            tenant_id="tenant2",
            branch="dev",
        )
        await queue_client.enqueue_file(
            file_path="/test/wrong_branch.py",
            collection="test-collection",
            tenant_id="tenant1",
            branch="main",
        )

        items = await queue_client.dequeue_batch(
            batch_size=10, tenant_id="tenant1", branch="dev"
        )
        assert len(items) == 1
        assert items[0].file_absolute_path == str(Path("/test/match.py").resolve())

    @pytest.mark.asyncio
    async def test_dequeue_batch_empty_queue(self, queue_client):
        """Test dequeue returns empty list when queue is empty."""
        items = await queue_client.dequeue_batch(batch_size=10)
        assert items == []

    @pytest.mark.asyncio
    async def test_dequeue_batch_respects_batch_size(self, queue_client):
        """Test dequeue respects batch_size parameter."""
        for i in range(15):
            await queue_client.enqueue_file(
                file_path=f"/test/file{i}.py",
                collection="test-collection",
            )

        items = await queue_client.dequeue_batch(batch_size=5)
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_update_priority_basic(self, queue_client):
        """Test basic priority update."""
        await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
            priority=5,
        )

        updated = await queue_client.update_priority("/test/file.py", 9)
        assert updated is True

        items = await queue_client.dequeue_batch(batch_size=1)
        assert items[0].priority == 9

    @pytest.mark.asyncio
    async def test_update_priority_not_found(self, queue_client):
        """Test update priority returns False for non-existent file."""
        updated = await queue_client.update_priority("/test/nonexistent.py", 5)
        assert updated is False

    @pytest.mark.asyncio
    async def test_update_priority_invalid(self, queue_client):
        """Test update priority with invalid values raises ValueError."""
        await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
        )

        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await queue_client.update_priority("/test/file.py", 15)

        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await queue_client.update_priority("/test/file.py", -5)

    @pytest.mark.asyncio
    async def test_mark_complete_basic(self, queue_client):
        """Test mark complete removes item from queue."""
        await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
        )

        removed = await queue_client.mark_complete("/test/file.py")
        assert removed is True

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_mark_complete_not_found(self, queue_client):
        """Test mark complete returns False for non-existent file."""
        removed = await queue_client.mark_complete("/test/nonexistent.py")
        assert removed is False

    @pytest.mark.asyncio
    async def test_mark_error_basic(self, queue_client):
        """Test mark error increments retry count and links error."""
        await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
        )

        should_retry, error_id = await queue_client.mark_error(
            file_path="/test/file.py",
            error_type="ProcessingError",
            error_message="Test error",
            error_details={"detail": "test"},
            max_retries=3,
        )

        assert should_retry is True
        assert error_id is not None

        items = await queue_client.dequeue_batch(batch_size=1)
        assert len(items) == 1
        assert items[0].retry_count == 1
        assert items[0].error_message_id == error_id

    @pytest.mark.asyncio
    async def test_mark_error_max_retries_removes_item(self, queue_client):
        """Test item is removed when max retries reached."""
        await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
        )

        # Mark error 3 times
        for i in range(3):
            should_retry, _ = await queue_client.mark_error(
                file_path="/test/file.py",
                error_type="ProcessingError",
                error_message=f"Error {i}",
                max_retries=3,
            )

        # After 3 errors, should_retry should be False
        assert should_retry is False

        # Item should be removed from queue
        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_mark_error_not_found(self, queue_client):
        """Test mark error returns False for non-existent file."""
        should_retry, error_id = await queue_client.mark_error(
            file_path="/test/nonexistent.py",
            error_type="ProcessingError",
            error_message="Test error",
        )

        assert should_retry is False
        assert error_id is not None  # Error message is still created

    @pytest.mark.asyncio
    async def test_get_queue_stats_basic(self, queue_client):
        """Test get queue statistics."""
        # Enqueue items with different priorities
        await queue_client.enqueue_file(
            file_path="/test/urgent.py",
            collection="test-collection",
            priority=9,
        )
        await queue_client.enqueue_file(
            file_path="/test/high.py",
            collection="test-collection",
            priority=6,
        )
        await queue_client.enqueue_file(
            file_path="/test/normal.py",
            collection="test-collection",
            priority=4,
        )
        await queue_client.enqueue_file(
            file_path="/test/low.py",
            collection="test-collection",
            priority=1,
        )

        stats = await queue_client.get_queue_stats()

        assert stats["total_items"] == 4
        assert stats["urgent_items"] == 1  # priority >= 8
        assert stats["high_items"] == 1  # priority >= 5 and < 8
        assert stats["normal_items"] == 1  # priority >= 3 and < 5
        assert stats["low_items"] == 1  # priority < 3
        assert stats["retry_items"] == 0
        assert stats["error_items"] == 0
        assert stats["unique_collections"] == 1

    @pytest.mark.asyncio
    async def test_get_queue_stats_with_errors(self, queue_client):
        """Test queue stats includes error and retry counts."""
        await queue_client.enqueue_file(
            file_path="/test/file1.py",
            collection="test-collection",
        )
        await queue_client.enqueue_file(
            file_path="/test/file2.py",
            collection="test-collection",
        )

        # Mark one with error
        await queue_client.mark_error(
            file_path="/test/file1.py",
            error_type="TestError",
            error_message="Test",
        )

        stats = await queue_client.get_queue_stats()

        assert stats["total_items"] == 2
        assert stats["retry_items"] == 1
        assert stats["error_items"] == 1

    @pytest.mark.asyncio
    async def test_get_queue_stats_with_filters(self, queue_client):
        """Test queue stats with tenant and branch filters."""
        await queue_client.enqueue_file(
            file_path="/test/tenant1_main.py",
            collection="test-collection",
            tenant_id="tenant1",
            branch="main",
        )
        await queue_client.enqueue_file(
            file_path="/test/tenant1_dev.py",
            collection="test-collection",
            tenant_id="tenant1",
            branch="dev",
        )
        await queue_client.enqueue_file(
            file_path="/test/tenant2_main.py",
            collection="test-collection",
            tenant_id="tenant2",
            branch="main",
        )

        stats = await queue_client.get_queue_stats(tenant_id="tenant1", branch="main")

        assert stats["total_items"] == 1

    @pytest.mark.asyncio
    async def test_get_queue_stats_empty_queue(self, queue_client):
        """Test queue stats on empty queue."""
        stats = await queue_client.get_queue_stats()

        assert stats["total_items"] == 0
        assert stats["urgent_items"] == 0
        assert stats["unique_collections"] == 0

    @pytest.mark.asyncio
    async def test_enqueue_batch_basic(self, queue_client):
        """Test batch enqueue operation."""
        items = [
            {"file_path": f"/test/file{i}.py", "collection": "test-collection", "priority": 5}
            for i in range(5)
        ]

        successful, failed = await queue_client.enqueue_batch(items)

        assert successful == 5
        assert len(failed) == 0

        queue_items = await queue_client.dequeue_batch(batch_size=10)
        assert len(queue_items) == 5

    @pytest.mark.asyncio
    async def test_enqueue_batch_with_queue_depth_limit_reject(self, queue_client):
        """Test batch enqueue rejects when queue depth limit reached."""
        # Add some existing items
        for i in range(5):
            await queue_client.enqueue_file(
                file_path=f"/existing/file{i}.py",
                collection="test-collection",
            )

        items = [
            {"file_path": f"/test/file{i}.py", "collection": "test-collection", "priority": 5}
            for i in range(10)
        ]

        with pytest.raises(ValueError, match="Queue depth limit"):
            await queue_client.enqueue_batch(
                items, max_queue_depth=10, overflow_strategy="reject"
            )

    @pytest.mark.asyncio
    async def test_enqueue_batch_with_queue_depth_limit_replace(self, queue_client):
        """Test batch enqueue replaces lowest priority items when limit reached."""
        # Add existing low priority items
        for i in range(5):
            await queue_client.enqueue_file(
                file_path=f"/existing/file{i}.py",
                collection="test-collection",
                priority=1,
            )

        # Try to add high priority items
        items = [
            {"file_path": f"/test/file{i}.py", "collection": "test-collection", "priority": 9}
            for i in range(3)
        ]

        successful, failed = await queue_client.enqueue_batch(
            items, max_queue_depth=6, overflow_strategy="replace_lowest"
        )

        assert successful == 3
        assert len(failed) == 0

        # Check that we have 6 items total
        depth = await queue_client.get_queue_depth()
        assert depth == 6

        # High priority items should be in queue
        queue_items = await queue_client.dequeue_batch(batch_size=10)
        high_priority_count = sum(1 for item in queue_items if item.priority == 9)
        assert high_priority_count == 3

    @pytest.mark.asyncio
    async def test_enqueue_batch_empty_list(self, queue_client):
        """Test batch enqueue with empty list."""
        successful, failed = await queue_client.enqueue_batch([])

        assert successful == 0
        assert len(failed) == 0

    @pytest.mark.asyncio
    async def test_enqueue_batch_invalid_items(self, queue_client):
        """Test batch enqueue validates required fields."""
        items = [
            {"file_path": "/test/file1.py", "collection": "test-collection"},  # Missing priority
        ]

        with pytest.raises(ValueError, match="must have file_path, collection, and priority"):
            await queue_client.enqueue_batch(items)

    @pytest.mark.asyncio
    async def test_enqueue_batch_invalid_priority(self, queue_client):
        """Test batch enqueue validates priority range."""
        items = [
            {"file_path": "/test/file1.py", "collection": "test-collection", "priority": 15},
        ]

        with pytest.raises(ValueError, match="Priority must be 0-10"):
            await queue_client.enqueue_batch(items)

    @pytest.mark.asyncio
    async def test_purge_completed_items(self, queue_client):
        """Test purging old error messages."""
        # Create some test messages by marking errors
        await queue_client.enqueue_file(
            file_path="/test/file1.py",
            collection="test-collection",
        )

        await queue_client.mark_error(
            file_path="/test/file1.py",
            error_type="TestError",
            error_message="Test error",
        )

        # Purge with very short retention (should purge nothing recent)
        purged = await queue_client.purge_completed_items(retention_hours=24)
        assert purged == 0

        # Purge with negative retention (should purge all)
        purged = await queue_client.purge_completed_items(retention_hours=-1)
        assert purged >= 1

    @pytest.mark.asyncio
    async def test_get_queue_depth_basic(self, queue_client):
        """Test get queue depth."""
        for i in range(7):
            await queue_client.enqueue_file(
                file_path=f"/test/file{i}.py",
                collection="test-collection",
            )

        depth = await queue_client.get_queue_depth()
        assert depth == 7

    @pytest.mark.asyncio
    async def test_get_queue_depth_with_filters(self, queue_client):
        """Test get queue depth with filters."""
        await queue_client.enqueue_file(
            file_path="/test/tenant1.py",
            collection="test-collection",
            tenant_id="tenant1",
            branch="main",
        )
        await queue_client.enqueue_file(
            file_path="/test/tenant2.py",
            collection="test-collection",
            tenant_id="tenant2",
            branch="main",
        )

        depth = await queue_client.get_queue_depth(tenant_id="tenant1")
        assert depth == 1

    @pytest.mark.asyncio
    async def test_get_queue_depth_empty(self, queue_client):
        """Test get queue depth on empty queue."""
        depth = await queue_client.get_queue_depth()
        assert depth == 0

    @pytest.mark.asyncio
    async def test_clear_queue_all(self, queue_client):
        """Test clearing entire queue."""
        for i in range(5):
            await queue_client.enqueue_file(
                file_path=f"/test/file{i}.py",
                collection="test-collection",
            )

        cleared = await queue_client.clear_queue()
        assert cleared == 5

        depth = await queue_client.get_queue_depth()
        assert depth == 0

    @pytest.mark.asyncio
    async def test_clear_queue_with_collection_filter(self, queue_client):
        """Test clearing queue with collection filter."""
        await queue_client.enqueue_file(
            file_path="/test/file1.py",
            collection="collection1",
        )
        await queue_client.enqueue_file(
            file_path="/test/file2.py",
            collection="collection2",
        )

        cleared = await queue_client.clear_queue(collection="collection1")
        assert cleared == 1

        depth = await queue_client.get_queue_depth()
        assert depth == 1

    @pytest.mark.asyncio
    async def test_clear_queue_with_multiple_filters(self, queue_client):
        """Test clearing queue with multiple filters."""
        await queue_client.enqueue_file(
            file_path="/test/file1.py",
            collection="test-collection",
            tenant_id="tenant1",
            branch="main",
        )
        await queue_client.enqueue_file(
            file_path="/test/file2.py",
            collection="test-collection",
            tenant_id="tenant1",
            branch="dev",
        )
        await queue_client.enqueue_file(
            file_path="/test/file3.py",
            collection="test-collection",
            tenant_id="tenant2",
            branch="main",
        )

        cleared = await queue_client.clear_queue(tenant_id="tenant1", branch="main")
        assert cleared == 1

        depth = await queue_client.get_queue_depth()
        assert depth == 2

    @pytest.mark.asyncio
    async def test_register_collection_basic(self, queue_client):
        """Test registering collection metadata."""
        registered = await queue_client.register_collection(
            collection_name="test-collection",
            collection_type="watched-dynamic",
            configuration={"watch_path": "/test"},
        )

        assert registered is True

        info = await queue_client.get_collection_info("test-collection")
        assert info is not None
        assert info["collection_name"] == "test-collection"
        assert info["collection_type"] == "watched-dynamic"
        assert info["configuration"]["watch_path"] == "/test"

    @pytest.mark.asyncio
    async def test_register_collection_update_existing(self, queue_client):
        """Test updating existing collection metadata."""
        await queue_client.register_collection(
            collection_name="test-collection",
            collection_type="watched-dynamic",
        )

        # Update with new type
        await queue_client.register_collection(
            collection_name="test-collection",
            collection_type="project",
        )

        info = await queue_client.get_collection_info("test-collection")
        assert info["collection_type"] == "project"

    @pytest.mark.asyncio
    async def test_register_collection_invalid_type(self, queue_client):
        """Test registering collection with invalid type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid collection_type"):
            await queue_client.register_collection(
                collection_name="test-collection",
                collection_type="invalid-type",
            )

    @pytest.mark.asyncio
    async def test_get_collection_info_not_found(self, queue_client):
        """Test get collection info returns None for non-existent collection."""
        info = await queue_client.get_collection_info("nonexistent")
        assert info is None

    @pytest.mark.asyncio
    async def test_initialization_idempotent(self, queue_client):
        """Test that calling initialize multiple times is safe."""
        # queue_client is already initialized
        await queue_client.initialize()
        await queue_client.initialize()

        # Should still work
        await queue_client.enqueue_file(
            file_path="/test/file.py",
            collection="test-collection",
        )

    @pytest.mark.asyncio
    async def test_close_idempotent(self, queue_client):
        """Test that calling close multiple times is safe."""
        await queue_client.close()
        await queue_client.close()

    @pytest.mark.asyncio
    async def test_concurrent_enqueue(self, queue_client):
        """Test concurrent enqueue operations."""
        async def enqueue_task(i):
            await queue_client.enqueue_file(
                file_path=f"/test/file{i}.py",
                collection="test-collection",
            )

        # Enqueue 10 files concurrently
        await asyncio.gather(*[enqueue_task(i) for i in range(10)])

        depth = await queue_client.get_queue_depth()
        assert depth == 10

    @pytest.mark.asyncio
    async def test_concurrent_dequeue(self, queue_client):
        """Test concurrent dequeue operations."""
        # Enqueue items
        for i in range(20):
            await queue_client.enqueue_file(
                file_path=f"/test/file{i}.py",
                collection="test-collection",
            )

        # Dequeue concurrently
        results = await asyncio.gather(
            queue_client.dequeue_batch(batch_size=5),
            queue_client.dequeue_batch(batch_size=5),
            queue_client.dequeue_batch(batch_size=5),
        )

        # All dequeues should succeed
        total_items = sum(len(items) for items in results)
        assert total_items == 15

    @pytest.mark.asyncio
    async def test_operation_enum_values(self, queue_client):
        """Test all QueueOperation enum values can be enqueued."""
        for operation in QueueOperation:
            await queue_client.enqueue_file(
                file_path=f"/test/{operation.value}.py",
                collection="test-collection",
                operation=operation,
            )

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 3

        operations = {item.operation for item in items}
        assert operations == {
            QueueOperation.INGEST,
            QueueOperation.UPDATE,
            QueueOperation.DELETE,
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
