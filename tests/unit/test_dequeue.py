"""Unit tests for SQLiteStateManager dequeue method."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta

from src.python.common.core.sqlite_state_manager import (
    SQLiteStateManager,
    ProcessingPriority,
    ProcessingQueueItem,
)


class TestDequeue:
    """Test dequeue method functionality."""

    @pytest.fixture
    async def state_manager(self):
        """Create a temporary state manager for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state.db"
            manager = SQLiteStateManager(str(db_path))
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_dequeue_single_item(self, state_manager):
        """Test dequeuing a single item from the queue."""
        # Add an item to the queue
        await state_manager.add_to_processing_queue(
            file_path="/test/file1.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
        )

        # Dequeue items
        items = await state_manager.dequeue(batch_size=1)
        assert len(items) == 1
        assert items[0].file_path == "/test/file1.txt"
        assert items[0].collection == "test-collection"
        assert items[0].priority == ProcessingPriority.NORMAL

    @pytest.mark.asyncio
    async def test_dequeue_multiple_items(self, state_manager):
        """Test dequeuing multiple items in batch."""
        # Add multiple items to the queue
        for i in range(5):
            await state_manager.add_to_processing_queue(
                file_path=f"/test/file{i}.txt",
                collection="test-collection",
                priority=ProcessingPriority.NORMAL,
            )

        # Dequeue batch
        items = await state_manager.dequeue(batch_size=3)
        assert len(items) == 3
        assert all(isinstance(item, ProcessingQueueItem) for item in items)

    @pytest.mark.asyncio
    async def test_dequeue_priority_ordering(self, state_manager):
        """Test dequeue returns items ordered by priority DESC, scheduled_at ASC."""
        # Add items with different priorities
        await state_manager.add_to_processing_queue(
            file_path="/test/low.txt",
            collection="test-collection",
            priority=ProcessingPriority.LOW,
        )
        await asyncio.sleep(0.01)  # Small delay to ensure different timestamps

        await state_manager.add_to_processing_queue(
            file_path="/test/urgent.txt",
            collection="test-collection",
            priority=ProcessingPriority.URGENT,
        )
        await asyncio.sleep(0.01)

        await state_manager.add_to_processing_queue(
            file_path="/test/high.txt",
            collection="test-collection",
            priority=ProcessingPriority.HIGH,
        )

        # Dequeue all items
        items = await state_manager.dequeue(batch_size=10)
        assert len(items) == 3
        # Should be ordered by priority DESC
        assert items[0].file_path == "/test/urgent.txt"
        assert items[1].file_path == "/test/high.txt"
        assert items[2].file_path == "/test/low.txt"

    @pytest.mark.asyncio
    async def test_dequeue_scheduled_at_filtering(self, state_manager):
        """Test dequeue filters out items scheduled in the future."""
        # Add item scheduled for now
        await state_manager.add_to_processing_queue(
            file_path="/test/now.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            scheduled_at=datetime.now(timezone.utc),
        )

        # Add item scheduled for future
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)
        await state_manager.add_to_processing_queue(
            file_path="/test/future.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            scheduled_at=future_time,
        )

        # Dequeue - should only get the current item
        items = await state_manager.dequeue(batch_size=10)
        assert len(items) == 1
        assert items[0].file_path == "/test/now.txt"

    @pytest.mark.asyncio
    async def test_dequeue_with_retry_from_filter(self, state_manager):
        """Test dequeue with retry_from parameter."""
        # Add items with different scheduled times
        past_time = datetime.now(timezone.utc) - timedelta(hours=2)
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        await state_manager.add_to_processing_queue(
            file_path="/test/old.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            scheduled_at=past_time,
        )

        await state_manager.add_to_processing_queue(
            file_path="/test/recent.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            scheduled_at=recent_time,
        )

        # Dequeue with retry_from filter
        retry_from = datetime.now(timezone.utc) - timedelta(hours=1)
        items = await state_manager.dequeue(batch_size=10, retry_from=retry_from)

        # Should only get items scheduled after retry_from
        assert len(items) == 1
        assert items[0].file_path == "/test/recent.txt"

    @pytest.mark.asyncio
    async def test_dequeue_with_tenant_id_filter(self, state_manager):
        """Test dequeue with tenant_id filtering."""
        # Add items with different tenant_ids in metadata
        await state_manager.add_to_processing_queue(
            file_path="/test/tenant1.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            metadata={"tenant_id": "tenant1"},
        )

        await state_manager.add_to_processing_queue(
            file_path="/test/tenant2.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            metadata={"tenant_id": "tenant2"},
        )

        await state_manager.add_to_processing_queue(
            file_path="/test/no_tenant.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            metadata={},
        )

        # Dequeue with tenant_id filter
        items = await state_manager.dequeue(batch_size=10, tenant_id="tenant1")

        # Should only get items for tenant1
        assert len(items) == 1
        assert items[0].file_path == "/test/tenant1.txt"

    @pytest.mark.asyncio
    async def test_dequeue_with_branch_filter(self, state_manager):
        """Test dequeue with branch filtering."""
        # Add items with different branches in metadata
        await state_manager.add_to_processing_queue(
            file_path="/test/main.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            metadata={"branch": "main"},
        )

        await state_manager.add_to_processing_queue(
            file_path="/test/dev.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            metadata={"branch": "dev"},
        )

        # Dequeue with branch filter
        items = await state_manager.dequeue(batch_size=10, branch="main")

        # Should only get items for main branch
        assert len(items) == 1
        assert items[0].file_path == "/test/main.txt"

    @pytest.mark.asyncio
    async def test_dequeue_combined_filters(self, state_manager):
        """Test dequeue with multiple filters combined."""
        # Add items with various metadata
        await state_manager.add_to_processing_queue(
            file_path="/test/match.txt",
            collection="test-collection",
            priority=ProcessingPriority.HIGH,
            metadata={"tenant_id": "tenant1", "branch": "main"},
        )

        await state_manager.add_to_processing_queue(
            file_path="/test/wrong_tenant.txt",
            collection="test-collection",
            priority=ProcessingPriority.HIGH,
            metadata={"tenant_id": "tenant2", "branch": "main"},
        )

        await state_manager.add_to_processing_queue(
            file_path="/test/wrong_branch.txt",
            collection="test-collection",
            priority=ProcessingPriority.HIGH,
            metadata={"tenant_id": "tenant1", "branch": "dev"},
        )

        # Dequeue with both filters
        items = await state_manager.dequeue(
            batch_size=10, tenant_id="tenant1", branch="main"
        )

        # Should only get the matching item
        assert len(items) == 1
        assert items[0].file_path == "/test/match.txt"

    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self, state_manager):
        """Test dequeue returns empty list when queue is empty."""
        items = await state_manager.dequeue(batch_size=10)
        assert items == []

    @pytest.mark.asyncio
    async def test_dequeue_respects_batch_size(self, state_manager):
        """Test dequeue respects batch_size parameter."""
        # Add more items than batch_size
        for i in range(15):
            await state_manager.add_to_processing_queue(
                file_path=f"/test/file{i}.txt",
                collection="test-collection",
                priority=ProcessingPriority.NORMAL,
            )

        # Dequeue with specific batch size
        items = await state_manager.dequeue(batch_size=5)
        assert len(items) == 5

    @pytest.mark.asyncio
    async def test_dequeue_default_batch_size(self, state_manager):
        """Test dequeue uses default batch_size of 10."""
        # Add more items than default batch_size
        for i in range(15):
            await state_manager.add_to_processing_queue(
                file_path=f"/test/file{i}.txt",
                collection="test-collection",
                priority=ProcessingPriority.NORMAL,
            )

        # Dequeue with default batch size
        items = await state_manager.dequeue()
        assert len(items) == 10

    @pytest.mark.asyncio
    async def test_dequeue_preserves_metadata(self, state_manager):
        """Test dequeue preserves metadata from queue items."""
        test_metadata = {"key": "value", "number": 42, "tenant_id": "test"}

        await state_manager.add_to_processing_queue(
            file_path="/test/file.txt",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            metadata=test_metadata,
        )

        items = await state_manager.dequeue(batch_size=1)
        assert len(items) == 1
        assert items[0].metadata == test_metadata

    @pytest.mark.asyncio
    async def test_dequeue_scheduled_at_asc_ordering(self, state_manager):
        """Test items with same priority are ordered by scheduled_at ASC."""
        # Add items with same priority but different scheduled times
        times = [
            datetime.now(timezone.utc) - timedelta(hours=3),
            datetime.now(timezone.utc) - timedelta(hours=1),
            datetime.now(timezone.utc) - timedelta(hours=2),
        ]

        for i, scheduled_time in enumerate(times):
            await state_manager.add_to_processing_queue(
                file_path=f"/test/file{i}.txt",
                collection="test-collection",
                priority=ProcessingPriority.NORMAL,
                scheduled_at=scheduled_time,
            )

        items = await state_manager.dequeue(batch_size=10)
        assert len(items) == 3

        # Should be ordered by scheduled_at ASC
        assert items[0].file_path == "/test/file0.txt"  # Oldest (3 hours ago)
        assert items[1].file_path == "/test/file2.txt"  # Middle (2 hours ago)
        assert items[2].file_path == "/test/file1.txt"  # Recent (1 hour ago)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
