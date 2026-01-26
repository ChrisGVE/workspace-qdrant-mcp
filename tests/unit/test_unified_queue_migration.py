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
