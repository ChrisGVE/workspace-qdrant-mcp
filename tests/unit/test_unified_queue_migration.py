"""
Tests for unified_queue schema migration (v12 -> v13).

Validates fresh database migration, legacy data preservation,
idempotency, rollback behavior, and concurrent migration safety.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.python.common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def temp_db_path() -> Path:
    """Create temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    yield db_path
    for suffix in ["", "-shm", "-wal"]:
        file = Path(str(db_path) + suffix)
        if file.exists():
            file.unlink()


async def create_v12_manager(db_path: Path) -> SQLiteStateManager:
    """Create a SQLiteStateManager initialized to schema version 12."""
    original_version = SQLiteStateManager.SCHEMA_VERSION
    SQLiteStateManager.SCHEMA_VERSION = SQLiteStateManager.BASE_SCHEMA_VERSION
    manager = SQLiteStateManager(db_path=str(db_path))
    try:
        await manager.initialize()
    finally:
        SQLiteStateManager.SCHEMA_VERSION = original_version
    return manager


def table_exists(manager: SQLiteStateManager, table_name: str) -> bool:
    """Check if a table exists in sqlite_master."""
    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cursor.fetchone() is not None


def schema_versions(manager: SQLiteStateManager) -> list[int]:
    """Fetch ordered schema versions."""
    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT version FROM schema_version ORDER BY version ASC"
        )
        rows = cursor.fetchall()
    return [row["version"] for row in rows]


@pytest.mark.asyncio
async def test_fresh_db_migrates_to_v13(temp_db_path: Path) -> None:
    """Fresh databases should migrate from v12 base schema to v13."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    assert table_exists(manager, "unified_queue")
    versions = schema_versions(manager)
    assert SQLiteStateManager.BASE_SCHEMA_VERSION in versions
    assert SQLiteStateManager.SCHEMA_VERSION in versions

    await manager.close()


@pytest.mark.asyncio
async def test_migration_preserves_legacy_queues(temp_db_path: Path) -> None:
    """Migration should add unified_queue without altering legacy queues."""
    v12_manager = await create_v12_manager(temp_db_path)

    file_path = str(temp_db_path.parent / "legacy_file.txt")
    await v12_manager.enqueue(
        file_path=file_path,
        collection="legacy-collection",
        priority=5,
        tenant_id="default",
        branch="main",
        metadata={"source": "legacy"},
    )
    content_queue_id, _ = await v12_manager.enqueue_ingestion(
        content="legacy content",
        collection="legacy-collection",
        source_type="scratchbook",
        priority=7,
        metadata={"tag": "legacy"},
    )

    with v12_manager._lock:
        legacy_ingestion = v12_manager.connection.execute(
            "SELECT file_absolute_path, collection_name, tenant_id, branch, priority "
            "FROM ingestion_queue"
        ).fetchall()
        legacy_content = v12_manager.connection.execute(
            "SELECT queue_id, collection, source_type, priority "
            "FROM content_ingestion_queue"
        ).fetchall()

    await v12_manager.close()

    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    assert table_exists(manager, "unified_queue")
    with manager._lock:
        ingestion_after = manager.connection.execute(
            "SELECT file_absolute_path, collection_name, tenant_id, branch, priority "
            "FROM ingestion_queue"
        ).fetchall()
        content_after = manager.connection.execute(
            "SELECT queue_id, collection, source_type, priority "
            "FROM content_ingestion_queue"
        ).fetchall()
        unified_count = manager.connection.execute(
            "SELECT COUNT(*) AS count FROM unified_queue"
        ).fetchone()["count"]

    assert legacy_ingestion == ingestion_after
    assert legacy_content == content_after
    assert any(row["queue_id"] == content_queue_id for row in content_after)
    assert unified_count == 0

    await manager.close()


@pytest.mark.asyncio
async def test_migration_idempotency(temp_db_path: Path) -> None:
    """Re-running the migration should not create duplicates or errors."""
    v12_manager = await create_v12_manager(temp_db_path)
    await v12_manager.close()

    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    await manager._migrate_schema(
        SQLiteStateManager.BASE_SCHEMA_VERSION, SQLiteStateManager.SCHEMA_VERSION
    )

    with manager._lock:
        count = manager.connection.execute(
            "SELECT COUNT(*) AS count FROM schema_version WHERE version = ?",
            (SQLiteStateManager.SCHEMA_VERSION,),
        ).fetchone()["count"]

    assert count == 1

    await manager.close()


@pytest.mark.asyncio
async def test_rollback_restores_v12_state(temp_db_path: Path) -> None:
    """Rollback should remove unified_queue and schema_version 13."""
    v12_manager = await create_v12_manager(temp_db_path)
    file_path = str(temp_db_path.parent / "legacy_file_rollback.txt")
    await v12_manager.enqueue(
        file_path=file_path,
        collection="legacy-collection",
        priority=4,
        tenant_id="default",
        branch="main",
        metadata={"source": "legacy"},
    )
    await v12_manager.enqueue_ingestion(
        content="legacy rollback",
        collection="legacy-collection",
        source_type="scratchbook",
        priority=6,
        metadata={"tag": "legacy"},
    )
    await v12_manager.close()

    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    with manager._lock:
        manager.connection.execute(
            """
            INSERT INTO unified_queue
            (item_type, op, tenant_id, collection, idempotency_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("file", "ingest", "default", "legacy-collection", "rollback-test"),
        )

    rollback_statements = manager._load_migration_statements(
        "003_unified_queue_rollback.sql"
    )
    with manager._lock:
        for statement in rollback_statements:
            manager.connection.execute(statement)

    assert not table_exists(manager, "unified_queue")
    versions = schema_versions(manager)
    assert SQLiteStateManager.SCHEMA_VERSION not in versions

    with manager._lock:
        legacy_count = manager.connection.execute(
            "SELECT COUNT(*) AS count FROM ingestion_queue"
        ).fetchone()["count"]
        content_count = manager.connection.execute(
            "SELECT COUNT(*) AS count FROM content_ingestion_queue"
        ).fetchone()["count"]

    assert legacy_count == 1
    assert content_count == 1

    await manager.close()


@pytest.mark.asyncio
async def test_initialize_migrates_v12_database(temp_db_path: Path) -> None:
    """initialize() should auto-migrate existing v12 databases to v13."""
    v12_manager = await create_v12_manager(temp_db_path)
    await v12_manager.close()

    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    assert table_exists(manager, "unified_queue")
    versions = schema_versions(manager)
    assert versions[-1] == SQLiteStateManager.SCHEMA_VERSION

    await manager.close()


@pytest.mark.asyncio
async def test_concurrent_migration_v12_to_v13(temp_db_path: Path) -> None:
    """Concurrent initializes should migrate safely with single v13 entry."""
    v12_manager = await create_v12_manager(temp_db_path)
    await v12_manager.close()

    managers = [
        SQLiteStateManager(db_path=str(temp_db_path)),
        SQLiteStateManager(db_path=str(temp_db_path)),
        SQLiteStateManager(db_path=str(temp_db_path)),
    ]

    await asyncio.gather(*(manager.initialize() for manager in managers))

    with managers[0]._lock:
        version_count = managers[0].connection.execute(
            "SELECT COUNT(*) AS count FROM schema_version WHERE version = ?",
            (SQLiteStateManager.SCHEMA_VERSION,),
        ).fetchone()["count"]

    assert version_count == 1

    await asyncio.gather(*(manager.close() for manager in managers))


def test_migration_statement_ordering() -> None:
    """Schema version insert should be the final migration statement."""
    manager = SQLiteStateManager(db_path=":memory:")
    statements = manager._load_migration_statements("003_unified_queue.sql")

    schema_inserts = [
        idx for idx, stmt in enumerate(statements) if "schema_version" in stmt.lower()
    ]

    assert schema_inserts, "Expected schema_version insert statement in migration file"
    assert schema_inserts[-1] == len(statements) - 1
    assert all(
        "schema_version" not in stmt.lower() for stmt in statements[:-1]
    )


# ============================================================================
# enqueue_unified Tests (Task 25)
# ============================================================================

from src.python.common.core.sqlite_state_manager import (
    UnifiedQueueItemType,
    UnifiedQueueOperation,
    generate_unified_idempotency_key,
)


@pytest.mark.asyncio
async def test_enqueue_unified_basic(temp_db_path: Path) -> None:
    """Test basic enqueue_unified functionality."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    queue_id, is_new = await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_test123",
        collection="test-project-code",
        payload={"file_path": "/path/to/test.py"},
        priority=7,
    )

    assert is_new is True
    assert queue_id is not None
    assert len(queue_id) == 36  # UUID format

    # Verify item is in database
    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT * FROM unified_queue WHERE queue_id = ?",
            (queue_id,),
        )
        row = cursor.fetchone()

    assert row is not None
    assert row["item_type"] == "file"
    assert row["op"] == "ingest"
    assert row["tenant_id"] == "proj_test123"
    assert row["collection"] == "test-project-code"
    assert row["priority"] == 7
    assert row["status"] == "pending"

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_idempotency(temp_db_path: Path) -> None:
    """Test that duplicate enqueues return existing item."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    # First enqueue
    queue_id1, is_new1 = await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_idem",
        collection="idem-collection",
        payload={"file_path": "/path/to/duplicate.py"},
    )

    # Second enqueue with same inputs
    queue_id2, is_new2 = await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_idem",
        collection="idem-collection",
        payload={"file_path": "/path/to/duplicate.py"},
    )

    assert is_new1 is True
    assert is_new2 is False
    assert queue_id1 == queue_id2

    # Verify only one row exists
    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT COUNT(*) as count FROM unified_queue WHERE tenant_id = ?",
            ("proj_idem",),
        )
        count = cursor.fetchone()["count"]

    assert count == 1

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_validation_priority_range(temp_db_path: Path) -> None:
    """Test that priority must be between 0 and 10."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    # Priority too low
    with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
        await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.FILE,
            op=UnifiedQueueOperation.INGEST,
            tenant_id="proj_valid",
            collection="valid-collection",
            payload={},
            priority=-1,
        )

    # Priority too high
    with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
        await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.FILE,
            op=UnifiedQueueOperation.INGEST,
            tenant_id="proj_valid",
            collection="valid-collection",
            payload={},
            priority=11,
        )

    # Valid edge cases
    queue_id0, _ = await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_edge0",
        collection="edge-collection",
        payload={},
        priority=0,
    )
    queue_id10, _ = await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_edge10",
        collection="edge-collection",
        payload={},
        priority=10,
    )

    assert queue_id0 is not None
    assert queue_id10 is not None

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_validation_empty_tenant_id(temp_db_path: Path) -> None:
    """Test that empty tenant_id raises ValueError."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    with pytest.raises(ValueError, match="tenant_id cannot be empty"):
        await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.FILE,
            op=UnifiedQueueOperation.INGEST,
            tenant_id="",
            collection="test-collection",
            payload={},
        )

    with pytest.raises(ValueError, match="tenant_id cannot be empty"):
        await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.FILE,
            op=UnifiedQueueOperation.INGEST,
            tenant_id="   ",
            collection="test-collection",
            payload={},
        )

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_validation_empty_collection(temp_db_path: Path) -> None:
    """Test that empty collection raises ValueError."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    with pytest.raises(ValueError, match="collection cannot be empty"):
        await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.FILE,
            op=UnifiedQueueOperation.INGEST,
            tenant_id="proj_test",
            collection="",
            payload={},
        )

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_validation_invalid_item_type(temp_db_path: Path) -> None:
    """Test that invalid item_type raises ValueError."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    with pytest.raises(ValueError, match="Invalid item_type"):
        await manager.enqueue_unified(
            item_type="invalid_type",
            op=UnifiedQueueOperation.INGEST,
            tenant_id="proj_test",
            collection="test-collection",
            payload={},
        )

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_validation_invalid_operation(temp_db_path: Path) -> None:
    """Test that invalid operation raises ValueError."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    with pytest.raises(ValueError, match="Invalid operation"):
        await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.FILE,
            op="invalid_op",
            tenant_id="proj_test",
            collection="test-collection",
            payload={},
        )

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_validation_invalid_op_for_item_type(temp_db_path: Path) -> None:
    """Test that invalid operation for item type raises ValueError."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    # DELETE_TENANT only supports DELETE operation
    with pytest.raises(ValueError, match="not valid for item type"):
        await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.DELETE_TENANT,
            op=UnifiedQueueOperation.INGEST,
            tenant_id="proj_test",
            collection="test-collection",
            payload={},
        )

    # RENAME only supports UPDATE operation
    with pytest.raises(ValueError, match="not valid for item type"):
        await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.RENAME,
            op=UnifiedQueueOperation.DELETE,
            tenant_id="proj_test",
            collection="test-collection",
            payload={},
        )

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_string_enums(temp_db_path: Path) -> None:
    """Test that string values for item_type and op work correctly."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    queue_id, is_new = await manager.enqueue_unified(
        item_type="content",
        op="ingest",
        tenant_id="proj_string",
        collection="string-collection",
        payload={"content": "test content"},
    )

    assert is_new is True
    assert queue_id is not None

    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT item_type, op FROM unified_queue WHERE queue_id = ?",
            (queue_id,),
        )
        row = cursor.fetchone()

    assert row["item_type"] == "content"
    assert row["op"] == "ingest"

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_complex_payload(temp_db_path: Path) -> None:
    """Test enqueue with complex nested payload."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    complex_payload = {
        "file_path": "/path/to/complex.py",
        "metadata": {
            "author": "test_user",
            "tags": ["python", "test", "complex"],
            "nested": {"deep": {"value": 42}},
        },
        "unicode_text": "Hello 世界 🌍",
        "special_chars": "quotes: \"'` and slashes: /\\",
    }

    queue_id, is_new = await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_complex",
        collection="complex-collection",
        payload=complex_payload,
    )

    assert is_new is True

    # Verify payload is stored correctly
    import json
    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT payload_json FROM unified_queue WHERE queue_id = ?",
            (queue_id,),
        )
        row = cursor.fetchone()
        stored_payload = json.loads(row["payload_json"])

    assert stored_payload["file_path"] == complex_payload["file_path"]
    assert stored_payload["unicode_text"] == complex_payload["unicode_text"]
    assert stored_payload["metadata"]["tags"] == complex_payload["metadata"]["tags"]

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_with_metadata(temp_db_path: Path) -> None:
    """Test enqueue with optional metadata."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    queue_id, is_new = await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_meta",
        collection="meta-collection",
        payload={"file_path": "/path/to/file.py"},
        metadata={"source": "test", "version": "1.0"},
    )

    assert is_new is True

    import json
    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT metadata FROM unified_queue WHERE queue_id = ?",
            (queue_id,),
        )
        row = cursor.fetchone()
        stored_metadata = json.loads(row["metadata"])

    assert stored_metadata["source"] == "test"
    assert stored_metadata["version"] == "1.0"

    await manager.close()


@pytest.mark.asyncio
async def test_enqueue_unified_with_branch(temp_db_path: Path) -> None:
    """Test enqueue with custom branch."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    queue_id, is_new = await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_branch",
        collection="branch-collection",
        payload={"file_path": "/path/to/file.py"},
        branch="feature/new-feature",
    )

    assert is_new is True

    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT branch FROM unified_queue WHERE queue_id = ?",
            (queue_id,),
        )
        row = cursor.fetchone()

    assert row["branch"] == "feature/new-feature"

    await manager.close()


@pytest.mark.asyncio
async def test_get_unified_queue_depth(temp_db_path: Path) -> None:
    """Test get_unified_queue_depth method."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    # Initial depth should be 0
    depth = await manager.get_unified_queue_depth()
    assert depth == 0

    # Add some items
    await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_depth1",
        collection="depth-collection",
        payload={"file_path": "/file1.py"},
    )
    await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.CONTENT,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_depth2",
        collection="depth-collection",
        payload={"content": "test"},
    )
    await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_depth3",
        collection="other-collection",
        payload={"file_path": "/file2.py"},
    )

    # Total depth
    depth = await manager.get_unified_queue_depth()
    assert depth == 3

    # Filter by collection
    depth = await manager.get_unified_queue_depth(collection="depth-collection")
    assert depth == 2

    # Filter by item_type
    depth = await manager.get_unified_queue_depth(item_type="file")
    assert depth == 2

    await manager.close()


@pytest.mark.asyncio
async def test_concurrent_enqueue_same_item(temp_db_path: Path) -> None:
    """Test concurrent enqueues of the same item."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    async def enqueue_item():
        return await manager.enqueue_unified(
            item_type=UnifiedQueueItemType.FILE,
            op=UnifiedQueueOperation.INGEST,
            tenant_id="proj_concurrent",
            collection="concurrent-collection",
            payload={"file_path": "/concurrent/file.py"},
        )

    # Run 10 concurrent enqueues
    results = await asyncio.gather(*[enqueue_item() for _ in range(10)])

    # All should return the same queue_id
    queue_ids = [r[0] for r in results]
    assert len(set(queue_ids)) == 1  # All same ID

    # Only one should be new
    new_counts = sum(1 for r in results if r[1] is True)
    assert new_counts == 1

    # Verify only one row exists
    with manager._lock:
        cursor = manager.connection.execute(
            "SELECT COUNT(*) as count FROM unified_queue WHERE tenant_id = ?",
            ("proj_concurrent",),
        )
        count = cursor.fetchone()["count"]

    assert count == 1

    await manager.close()


def test_generate_unified_idempotency_key_consistency() -> None:
    """Test that idempotency key generation is deterministic."""
    key1 = generate_unified_idempotency_key(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_test",
        collection="test-collection",
        payload={"file_path": "/test.py"},
    )
    key2 = generate_unified_idempotency_key(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_test",
        collection="test-collection",
        payload={"file_path": "/test.py"},
    )

    assert key1 == key2
    assert len(key1) == 32  # SHA256 truncated to 32 hex chars


def test_generate_unified_idempotency_key_uniqueness() -> None:
    """Test that different inputs produce different keys."""
    key1 = generate_unified_idempotency_key(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_test",
        collection="collection-a",
        payload={"file_path": "/test.py"},
    )
    key2 = generate_unified_idempotency_key(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj_test",
        collection="collection-b",  # Different collection
        payload={"file_path": "/test.py"},
    )

    assert key1 != key2


@pytest.mark.asyncio
async def test_get_unified_queue_stats_empty(temp_db_path: Path) -> None:
    """Test get_unified_queue_stats on empty queue."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    stats = await manager.get_unified_queue_stats()

    assert stats["total"] == 0
    assert stats["by_status"] == {}
    assert stats["by_item_type"] == {}
    assert stats["by_operation"] == {}
    assert stats["oldest_pending_age_seconds"] is None
    assert stats["collections_with_pending"] == []

    await manager.close()


@pytest.mark.asyncio
async def test_get_unified_queue_stats_with_items(temp_db_path: Path) -> None:
    """Test get_unified_queue_stats with various queue items."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()

    # Add various items
    await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj1",
        collection="collection-a",
        payload={"file_path": "/file1.py"},
    )
    await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.CONTENT,
        op=UnifiedQueueOperation.INGEST,
        tenant_id="proj2",
        collection="collection-a",
        payload={"content": "test"},
    )
    await manager.enqueue_unified(
        item_type=UnifiedQueueItemType.FILE,
        op=UnifiedQueueOperation.DELETE,
        tenant_id="proj3",
        collection="collection-b",
        payload={"file_path": "/file2.py"},
    )

    stats = await manager.get_unified_queue_stats()

    assert stats["total"] == 3
    assert stats["by_status"]["pending"] == 3
    assert stats["by_item_type"]["file"] == 2
    assert stats["by_item_type"]["content"] == 1
    assert stats["by_operation"]["ingest"] == 2
    assert stats["by_operation"]["delete"] == 1
    assert stats["oldest_pending_age_seconds"] is not None
    assert stats["oldest_pending_age_seconds"] >= 0
    assert set(stats["collections_with_pending"]) == {"collection-a", "collection-b"}

    await manager.close()
