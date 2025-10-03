"""
Unit tests for collection_type integration in queue system.

Tests collection type detection, storage, and retrieval across Python
queue client and database operations.
"""

import pytest
import sqlite3
import tempfile
from pathlib import Path

from src.python.common.core.queue_client import SQLiteQueueClient, QueueItem, QueueOperation
from src.python.common.core.collection_types import CollectionTypeClassifier, CollectionType
from src.python.common.core.queue_connection import ConnectionConfig


@pytest.fixture
async def queue_client():
    """Create a temporary queue client for testing."""
    # Create temporary database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    db_path = temp_db.name
    temp_db.close()

    # Initialize queue client
    config = ConnectionConfig(
        max_connections=1,
        min_connections=1,
        connection_timeout=30.0
    )
    client = SQLiteQueueClient(db_path=db_path, connection_config=config)
    await client.initialize()

    # Load schema
    schema_path = Path(__file__).parent.parent.parent / "src/python/common/core/queue_schema.sql"
    with open(schema_path) as f:
        schema_sql = f.read()

    conn = sqlite3.connect(db_path)
    conn.executescript(schema_sql)
    conn.close()

    yield client

    # Cleanup
    await client.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
class TestCollectionTypeDetection:
    """Test collection type detection during enqueue operations."""

    async def test_enqueue_system_collection(self, queue_client):
        """Test enqueueing with system collection (__prefix)."""
        await queue_client.enqueue_file(
            file_path="/test/file.txt",
            collection="__user_preferences",
            priority=5
        )

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 1
        assert items[0].collection_type == "system"

    async def test_enqueue_library_collection(self, queue_client):
        """Test enqueueing with library collection (_prefix)."""
        await queue_client.enqueue_file(
            file_path="/test/lib.py",
            collection="_python_stdlib",
            priority=5
        )

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 1
        assert items[0].collection_type == "library"

    async def test_enqueue_project_collection(self, queue_client):
        """Test enqueueing with project collection ({name}-{suffix})."""
        await queue_client.enqueue_file(
            file_path="/test/project.py",
            collection="myproject-docs",
            priority=5
        )

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 1
        assert items[0].collection_type == "project"

    async def test_enqueue_global_collection(self, queue_client):
        """Test enqueueing with global collection."""
        await queue_client.enqueue_file(
            file_path="/test/algo.py",
            collection="algorithms",
            priority=5
        )

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 1
        assert items[0].collection_type == "global"

    async def test_enqueue_unknown_collection(self, queue_client):
        """Test enqueueing with unknown collection type."""
        await queue_client.enqueue_file(
            file_path="/test/unknown.txt",
            collection="someunknowncollection",
            priority=5
        )

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 1
        assert items[0].collection_type == "unknown"


@pytest.mark.asyncio
class TestCollectionTypeBatch:
    """Test collection type handling in batch operations."""

    async def test_enqueue_batch_mixed_types(self, queue_client):
        """Test batch enqueue with different collection types."""
        items_to_enqueue = [
            {
                "file_path": "/test/file1.txt",
                "collection": "__system_config",
                "priority": 5
            },
            {
                "file_path": "/test/file2.txt",
                "collection": "_library_docs",
                "priority": 5
            },
            {
                "file_path": "/test/file3.txt",
                "collection": "project-code",
                "priority": 5
            },
            {
                "file_path": "/test/file4.txt",
                "collection": "algorithms",
                "priority": 5
            }
        ]

        successful, failed = await queue_client.enqueue_batch(items_to_enqueue)
        assert successful == 4
        assert len(failed) == 0

        # Verify types
        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 4

        types_by_collection = {item.collection_name: item.collection_type for item in items}
        assert types_by_collection["__system_config"] == "system"
        assert types_by_collection["_library_docs"] == "library"
        assert types_by_collection["project-code"] == "project"
        assert types_by_collection["algorithms"] == "global"


@pytest.mark.asyncio
class TestQueueItemSerialization:
    """Test QueueItem serialization with collection_type."""

    async def test_to_dict_includes_collection_type(self, queue_client):
        """Test QueueItem.to_dict() includes collection_type."""
        await queue_client.enqueue_file(
            file_path="/test/file.txt",
            collection="__test_collection",
            priority=5
        )

        items = await queue_client.dequeue_batch(batch_size=1)
        item_dict = items[0].to_dict()

        assert "collection_type" in item_dict
        assert item_dict["collection_type"] == "system"

    async def test_from_db_row_handles_null_collection_type(self, queue_client):
        """Test QueueItem.from_db_row() handles NULL collection_type gracefully."""
        # Manually insert item without collection_type (simulating legacy data)
        conn = sqlite3.connect(queue_client.connection_pool.db_path)
        conn.execute("""
            INSERT INTO ingestion_queue (
                file_absolute_path, collection_name, operation, priority
            ) VALUES (?, ?, ?, ?)
        """, ("/test/legacy.txt", "legacy-collection", "ingest", 5))
        conn.commit()
        conn.close()

        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 1
        assert items[0].collection_type is None  # Should be None, not crash


@pytest.mark.asyncio
class TestBackwardCompatibility:
    """Test backward compatibility with legacy queue items."""

    async def test_null_collection_type_handled(self, queue_client):
        """Test that NULL collection_type values are handled gracefully."""
        # Insert legacy item without collection_type
        conn = sqlite3.connect(queue_client.connection_pool.db_path)
        conn.execute("""
            INSERT INTO ingestion_queue (
                file_absolute_path, collection_name, operation, priority
            ) VALUES (?, ?, ?, ?)
        """, ("/test/legacy.txt", "test-collection", "ingest", 7))
        conn.commit()
        conn.close()

        # Should dequeue without error
        items = await queue_client.dequeue_batch(batch_size=10)
        assert len(items) == 1
        assert items[0].file_absolute_path == "/test/legacy.txt"
        assert items[0].collection_type is None


class TestCollectionTypeClassifier:
    """Test CollectionTypeClassifier directly."""

    def setup_method(self):
        """Setup classifier for each test."""
        self.classifier = CollectionTypeClassifier()

    def test_classify_system_collection(self):
        """Test classification of system collections."""
        result = self.classifier.classify_collection_type("__user_settings")
        assert result == CollectionType.SYSTEM

    def test_classify_library_collection(self):
        """Test classification of library collections."""
        result = self.classifier.classify_collection_type("_python_docs")
        assert result == CollectionType.LIBRARY

    def test_classify_project_collection(self):
        """Test classification of project collections."""
        result = self.classifier.classify_collection_type("myproject-documents")
        assert result == CollectionType.PROJECT

    def test_classify_global_collection(self):
        """Test classification of global collections."""
        for name in ["algorithms", "codebase", "documents", "workspace"]:
            result = self.classifier.classify_collection_type(name)
            assert result == CollectionType.GLOBAL

    def test_classify_unknown_collection(self):
        """Test classification of unknown collections."""
        result = self.classifier.classify_collection_type("unknowncollection")
        assert result == CollectionType.UNKNOWN

    def test_empty_collection_name(self):
        """Test classification of empty string."""
        result = self.classifier.classify_collection_type("")
        assert result == CollectionType.UNKNOWN
