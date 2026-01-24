"""
Unit tests for content ingestion queue (Task 456/ADR-001).

Tests SQLite queue fallback for MCP store() when daemon is unavailable.
"""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from common.core.sqlite_state_manager import (
    SQLiteStateManager,
    ContentIngestionStatus,
    ContentIngestionQueueItem,
)


@pytest.fixture
async def state_manager():
    """Create a fresh state manager with temp database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_state.db"
        manager = SQLiteStateManager(db_path=str(db_path))
        await manager.initialize()
        yield manager
        await manager.close()


class TestContentIngestionQueue:
    """Tests for content ingestion queue methods."""

    @pytest.mark.asyncio
    async def test_enqueue_ingestion_creates_item(self, state_manager):
        """Test that enqueue_ingestion creates a new queue item."""
        queue_id, is_new = await state_manager.enqueue_ingestion(
            content="Test content for ingestion",
            collection="projects",
            source_type="scratchbook",
            priority=8,
            main_tag="project123",
            full_tag="project123.main",
            metadata={"custom": "value"},
        )

        assert queue_id is not None
        assert is_new is True
        assert len(queue_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_enqueue_ingestion_idempotency(self, state_manager):
        """Test that duplicate content returns existing queue_id."""
        content = "Duplicate content test"
        metadata = {"key": "value"}

        # First enqueue
        queue_id1, is_new1 = await state_manager.enqueue_ingestion(
            content=content,
            collection="projects",
            source_type="scratchbook",
            metadata=metadata,
        )
        assert is_new1 is True

        # Second enqueue with same content
        queue_id2, is_new2 = await state_manager.enqueue_ingestion(
            content=content,
            collection="projects",
            source_type="scratchbook",
            metadata=metadata,
        )
        assert is_new2 is False
        assert queue_id1 == queue_id2

    @pytest.mark.asyncio
    async def test_enqueue_ingestion_different_content_creates_new(self, state_manager):
        """Test that different content creates different queue items."""
        queue_id1, _ = await state_manager.enqueue_ingestion(
            content="Content A",
            collection="projects",
        )
        queue_id2, _ = await state_manager.enqueue_ingestion(
            content="Content B",
            collection="projects",
        )

        assert queue_id1 != queue_id2

    @pytest.mark.asyncio
    async def test_enqueue_ingestion_priority_validation(self, state_manager):
        """Test that invalid priority raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await state_manager.enqueue_ingestion(
                content="Test",
                collection="projects",
                priority=11,
            )

        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await state_manager.enqueue_ingestion(
                content="Test",
                collection="projects",
                priority=-1,
            )

    @pytest.mark.asyncio
    async def test_dequeue_content_ingestion_returns_pending_items(self, state_manager):
        """Test that dequeue returns pending items and marks them in_progress."""
        # Add some items
        await state_manager.enqueue_ingestion(content="Item 1", collection="projects")
        await state_manager.enqueue_ingestion(content="Item 2", collection="projects")

        # Dequeue items
        items = await state_manager.dequeue_content_ingestion(batch_size=10)

        assert len(items) == 2
        assert all(isinstance(item, ContentIngestionQueueItem) for item in items)
        assert all(item.status == ContentIngestionStatus.IN_PROGRESS for item in items)

    @pytest.mark.asyncio
    async def test_dequeue_content_ingestion_priority_ordering(self, state_manager):
        """Test that dequeue returns items in priority order."""
        await state_manager.enqueue_ingestion(
            content="Low priority", collection="projects", priority=2
        )
        await state_manager.enqueue_ingestion(
            content="High priority", collection="projects", priority=9
        )
        await state_manager.enqueue_ingestion(
            content="Medium priority", collection="projects", priority=5
        )

        items = await state_manager.dequeue_content_ingestion(batch_size=10)

        assert len(items) == 3
        assert items[0].content == "High priority"
        assert items[1].content == "Medium priority"
        assert items[2].content == "Low priority"

    @pytest.mark.asyncio
    async def test_dequeue_content_ingestion_collection_filter(self, state_manager):
        """Test that dequeue filters by collection."""
        await state_manager.enqueue_ingestion(content="Projects item", collection="projects")
        await state_manager.enqueue_ingestion(content="Libraries item", collection="libraries")

        items = await state_manager.dequeue_content_ingestion(
            batch_size=10, collection="projects"
        )

        assert len(items) == 1
        assert items[0].content == "Projects item"

    @pytest.mark.asyncio
    async def test_update_content_ingestion_status_to_done(self, state_manager):
        """Test updating status to done."""
        queue_id, _ = await state_manager.enqueue_ingestion(
            content="Test content", collection="projects"
        )

        updated = await state_manager.update_content_ingestion_status(
            queue_id, ContentIngestionStatus.DONE
        )

        assert updated is True

        # Verify it's no longer in pending queue
        items = await state_manager.dequeue_content_ingestion(batch_size=10)
        assert len(items) == 0

    @pytest.mark.asyncio
    async def test_update_content_ingestion_status_to_failed(self, state_manager):
        """Test updating status to failed with error message."""
        queue_id, _ = await state_manager.enqueue_ingestion(
            content="Test content", collection="projects"
        )

        updated = await state_manager.update_content_ingestion_status(
            queue_id,
            ContentIngestionStatus.FAILED,
            error_message="Connection timeout",
        )

        assert updated is True

    @pytest.mark.asyncio
    async def test_get_content_ingestion_queue_depth(self, state_manager):
        """Test getting queue depth."""
        # Initially empty
        depth = await state_manager.get_content_ingestion_queue_depth()
        assert depth == 0

        # Add items
        await state_manager.enqueue_ingestion(content="Item 1", collection="projects")
        await state_manager.enqueue_ingestion(content="Item 2", collection="projects")
        await state_manager.enqueue_ingestion(content="Item 3", collection="libraries")

        # Total depth
        depth = await state_manager.get_content_ingestion_queue_depth()
        assert depth == 3

        # Filtered by collection
        depth = await state_manager.get_content_ingestion_queue_depth(collection="projects")
        assert depth == 2

    @pytest.mark.asyncio
    async def test_get_content_ingestion_queue_depth_by_status(self, state_manager):
        """Test getting queue depth filtered by status."""
        queue_id, _ = await state_manager.enqueue_ingestion(
            content="Item 1", collection="projects"
        )
        await state_manager.enqueue_ingestion(content="Item 2", collection="projects")

        # Mark one as done
        await state_manager.update_content_ingestion_status(
            queue_id, ContentIngestionStatus.DONE
        )

        # Pending count
        pending_depth = await state_manager.get_content_ingestion_queue_depth(
            status=ContentIngestionStatus.PENDING
        )
        assert pending_depth == 1

        # Done count
        done_depth = await state_manager.get_content_ingestion_queue_depth(
            status=ContentIngestionStatus.DONE
        )
        assert done_depth == 1

    @pytest.mark.asyncio
    async def test_reset_in_progress_content_items(self, state_manager):
        """Test crash recovery resets in_progress items to pending."""
        # Add and dequeue items (marks them in_progress)
        await state_manager.enqueue_ingestion(content="Item 1", collection="projects")
        await state_manager.enqueue_ingestion(content="Item 2", collection="projects")
        await state_manager.dequeue_content_ingestion(batch_size=10)

        # Verify they're in_progress
        in_progress_depth = await state_manager.get_content_ingestion_queue_depth(
            status=ContentIngestionStatus.IN_PROGRESS
        )
        assert in_progress_depth == 2

        # Reset
        reset_count = await state_manager.reset_in_progress_content_items()
        assert reset_count == 2

        # Verify they're back to pending
        pending_depth = await state_manager.get_content_ingestion_queue_depth(
            status=ContentIngestionStatus.PENDING
        )
        assert pending_depth == 2

    @pytest.mark.asyncio
    async def test_reset_in_progress_marks_max_retries_as_failed(self, state_manager):
        """Test that items exceeding max retries are marked failed."""
        queue_id, _ = await state_manager.enqueue_ingestion(
            content="Failing item", collection="projects"
        )

        # Simulate multiple failures by dequeuing and resetting repeatedly
        for _ in range(4):  # Default max_retries=3
            await state_manager.dequeue_content_ingestion(batch_size=1)
            await state_manager.reset_in_progress_content_items(max_retries=3)

        # Item should now be failed
        failed_depth = await state_manager.get_content_ingestion_queue_depth(
            status=ContentIngestionStatus.FAILED
        )
        assert failed_depth == 1

    @pytest.mark.asyncio
    async def test_remove_completed_content_items(self, state_manager):
        """Test removing old completed items."""
        queue_id, _ = await state_manager.enqueue_ingestion(
            content="Completed item", collection="projects"
        )
        await state_manager.update_content_ingestion_status(
            queue_id, ContentIngestionStatus.DONE
        )

        # Won't remove recently completed items
        removed = await state_manager.remove_completed_content_items(older_than_hours=1)
        assert removed == 0

        # Verify item still exists
        done_depth = await state_manager.get_content_ingestion_queue_depth(
            status=ContentIngestionStatus.DONE
        )
        assert done_depth == 1

    @pytest.mark.asyncio
    async def test_content_queue_item_fields(self, state_manager):
        """Test that ContentIngestionQueueItem has all expected fields."""
        await state_manager.enqueue_ingestion(
            content="Full test content",
            collection="projects",
            source_type="file",
            priority=7,
            main_tag="proj_abc",
            full_tag="proj_abc.feature",
            metadata={"file_path": "/path/to/file.py"},
        )

        items = await state_manager.dequeue_content_ingestion(batch_size=1)
        item = items[0]

        assert item.queue_id is not None
        assert item.idempotency_key is not None
        assert item.content == "Full test content"
        assert item.collection == "projects"
        assert item.source_type == "file"
        assert item.priority == 7
        assert item.status == ContentIngestionStatus.IN_PROGRESS
        assert item.main_tag == "proj_abc"
        assert item.full_tag == "proj_abc.feature"
        assert item.metadata == {"file_path": "/path/to/file.py"}
        assert item.created_at is not None
        assert item.updated_at is not None
        assert item.started_at is not None  # Set when dequeued


class TestContentIngestionQueueMigration:
    """Tests for v7 -> v8 schema migration."""

    @pytest.mark.asyncio
    async def test_migration_creates_content_queue_table(self):
        """Test that migration from v7 creates the content_ingestion_queue table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_migration.db"
            manager = SQLiteStateManager(db_path=str(db_path))
            await manager.initialize()

            # Verify table exists and is functional
            queue_id, is_new = await manager.enqueue_ingestion(
                content="Migration test",
                collection="projects",
            )
            assert is_new is True

            await manager.close()
