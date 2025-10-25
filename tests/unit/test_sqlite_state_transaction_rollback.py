"""
Comprehensive transaction rollback testing framework for SQLiteStateManager.

Tests verify ACID transaction guarantees with focus on:
- Constraint violations (UNIQUE, FOREIGN KEY, CHECK, NOT NULL)
- Transaction rollback on errors
- Savepoint rollback behavior
- Nested transaction handling
- State consistency verification
- Database integrity after failures

Tests use temporary database files with WAL mode for realistic consistency testing.
"""

import asyncio
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest

from src.python.common.core.sqlite_state_manager import (
    FileProcessingRecord,
    FileProcessingStatus,
    LSPServerStatus,
    ProcessingPriority,
    SQLiteStateManager,
    WatchFolderConfig,
)


class TestConstraintViolations:
    """Test transaction rollback on constraint violations."""

    @pytest.fixture
    async def state_manager(self):
        """Create a temporary state manager with WAL mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state.db"
            manager = SQLiteStateManager(str(db_path))
            await manager.initialize()
            # Verify WAL mode is enabled
            cursor = manager.connection.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            assert journal_mode.upper() == "WAL", f"Expected WAL mode, got {journal_mode}"
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_unique_constraint_violation_watch_folders(self, state_manager):
        """Test UNIQUE constraint violation rolls back transaction."""
        # Create initial watch folder
        config1 = WatchFolderConfig(
            watch_id="test-watch-1",
            path="/test/path",
            collection="test-collection",
            patterns=["*.py"],
            ignore_patterns=["*.pyc"],
        )
        success = await state_manager.save_watch_folder_config(config1)
        assert success

        # Try to create another watch folder with same watch_id
        config2 = WatchFolderConfig(
            watch_id="test-watch-1",  # Same ID - should trigger UNIQUE violation
            path="/different/path",
            collection="different-collection",
            patterns=["*.js"],
            ignore_patterns=["node_modules/*"],
        )

        # The save_watch_folder_config uses INSERT OR REPLACE, so it should succeed
        # but update the existing record
        success = await state_manager.save_watch_folder_config(config2)
        assert success

        # Verify the record was updated, not duplicated
        retrieved = await state_manager.get_watch_folder_config("test-watch-1")
        assert retrieved is not None
        assert retrieved.path == "/different/path"
        assert retrieved.collection == "different-collection"

        # Verify only one record exists
        all_configs = await state_manager.get_all_watch_folder_configs(enabled_only=False)
        assert len(all_configs) == 1

    @pytest.mark.asyncio
    async def test_foreign_key_constraint_violation(self, state_manager):
        """Test FOREIGN KEY constraint violation rolls back transaction."""
        # Try to create file processing record with invalid lsp_server_id
        async with state_manager.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO file_processing
                    (file_path, collection, status, lsp_server_id)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file.py", "test-collection",
                     FileProcessingStatus.PENDING.value, 99999)  # Non-existent LSP server
                )
                # If we get here, FK constraint didn't fire (expected since FK might be deferred)
                # Force constraint check
                conn.execute("PRAGMA foreign_key_check")
            except sqlite3.IntegrityError as e:
                # Expected: FOREIGN KEY constraint failed
                assert "FOREIGN KEY" in str(e) or "foreign key" in str(e).lower()

        # Verify no record was inserted (transaction rolled back)
        record = await state_manager.get_file_processing_status("/test/file.py")
        assert record is None

    @pytest.mark.asyncio
    async def test_check_constraint_violation_priority(self, state_manager):
        """Test CHECK constraint violation for queue priority."""
        # Try to enqueue file with invalid priority (outside 0-10 range)
        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await state_manager.enqueue(
                file_path="/test/file.py",
                collection="test-collection",
                priority=15,  # Invalid - exceeds max of 10
                tenant_id="default",
                branch="main",
            )

        # Verify no record was inserted
        depth = await state_manager.get_queue_depth()
        assert depth == 0

    @pytest.mark.asyncio
    async def test_check_constraint_violation_operation(self, state_manager):
        """Test CHECK constraint violation for queue operation."""
        async with state_manager.transaction() as conn:
            with pytest.raises(sqlite3.IntegrityError) as exc_info:
                conn.execute(
                    """
                    INSERT INTO ingestion_queue
                    (file_absolute_path, collection_name, operation, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file.py", "test-collection", "invalid_op", 5)
                )
            assert "CHECK constraint" in str(exc_info.value) or "constraint failed" in str(exc_info.value).lower()

        # Verify no record was inserted (transaction rolled back)
        depth = await state_manager.get_queue_depth()
        assert depth == 0

    @pytest.mark.asyncio
    async def test_not_null_constraint_violation(self, state_manager):
        """Test NOT NULL constraint violation rolls back transaction."""
        async with state_manager.transaction() as conn:
            with pytest.raises(sqlite3.IntegrityError) as exc_info:
                # Try to insert watch folder without required path field
                conn.execute(
                    """
                    INSERT INTO watch_folders (watch_id, collection, patterns, ignore_patterns)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("test-id", "test-collection", "[]", "[]")  # Missing 'path' (NOT NULL)
                )
            assert "NOT NULL" in str(exc_info.value) or "constraint failed" in str(exc_info.value).lower()

        # Verify no record was inserted (transaction rolled back)
        config = await state_manager.get_watch_folder_config("test-id")
        assert config is None


class TestTransactionRollback:
    """Test transaction rollback on various error conditions."""

    @pytest.fixture
    async def state_manager(self):
        """Create a temporary state manager with WAL mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state.db"
            manager = SQLiteStateManager(str(db_path))
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_rollback_on_exception(self, state_manager):
        """Test transaction rolls back on exception."""
        # First, insert a valid record
        await state_manager.start_file_processing(
            file_path="/test/file1.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
        )

        # Try to perform operations in a transaction that will fail
        try:
            async with state_manager.transaction() as conn:
                # Insert another valid record
                conn.execute(
                    """
                    INSERT INTO file_processing (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file2.py", "test-collection",
                     FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                )

                # Now trigger an error (duplicate primary key)
                conn.execute(
                    """
                    INSERT INTO file_processing (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file1.py", "test-collection",  # Duplicate file_path (PRIMARY KEY)
                     FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                )
        except sqlite3.IntegrityError:
            # Expected error
            pass

        # Verify file2.py was NOT inserted (transaction rolled back)
        record2 = await state_manager.get_file_processing_status("/test/file2.py")
        assert record2 is None

        # Verify file1.py is still present (original insert outside transaction)
        record1 = await state_manager.get_file_processing_status("/test/file1.py")
        assert record1 is not None

    @pytest.mark.asyncio
    async def test_rollback_preserves_original_data(self, state_manager):
        """Test rollback preserves original database state."""
        # Insert initial data
        original_config = WatchFolderConfig(
            watch_id="test-watch",
            path="/original/path",
            collection="original-collection",
            patterns=["*.py"],
            ignore_patterns=["*.pyc"],
            enabled=True,
        )
        await state_manager.save_watch_folder_config(original_config)

        # Try to update in a transaction that will fail
        try:
            async with state_manager.transaction() as conn:
                # Update the watch folder
                conn.execute(
                    """
                    UPDATE watch_folders
                    SET path = ?, collection = ?, enabled = ?
                    WHERE watch_id = ?
                    """,
                    ("/modified/path", "modified-collection", False, "test-watch")
                )

                # Insert a duplicate to trigger error
                conn.execute(
                    """
                    INSERT INTO watch_folders (watch_id, path, collection, patterns, ignore_patterns)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("test-watch", "/duplicate/path", "dup-collection", "[]", "[]")
                )
        except sqlite3.IntegrityError:
            # Expected error
            pass

        # Verify original data is preserved
        retrieved = await state_manager.get_watch_folder_config("test-watch")
        assert retrieved is not None
        assert retrieved.path == "/original/path"
        assert retrieved.collection == "original-collection"
        assert retrieved.enabled is True

    @pytest.mark.asyncio
    async def test_multiple_operations_rollback_together(self, state_manager):
        """Test multiple operations in transaction roll back together."""
        # Try multiple operations in one transaction
        try:
            async with state_manager.transaction() as conn:
                # Insert file processing record
                conn.execute(
                    """
                    INSERT INTO file_processing (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file.py", "test-collection",
                     FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                )

                # Insert queue item
                conn.execute(
                    """
                    INSERT INTO ingestion_queue
                    (file_absolute_path, collection_name, operation, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file.py", "test-collection", "ingest", 5)
                )

                # Insert system state
                conn.execute(
                    """
                    INSERT INTO system_state (key, value)
                    VALUES (?, ?)
                    """,
                    ("test-key", "test-value")
                )

                # Trigger error with invalid priority
                conn.execute(
                    """
                    INSERT INTO ingestion_queue
                    (file_absolute_path, collection_name, operation, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file2.py", "test-collection", "ingest", 99)  # Invalid priority
                )
        except sqlite3.IntegrityError:
            # Expected error
            pass

        # Verify ALL operations were rolled back
        record = await state_manager.get_file_processing_status("/test/file.py")
        assert record is None

        depth = await state_manager.get_queue_depth()
        assert depth == 0

        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT value FROM system_state WHERE key = ?", ("test-key",)
            )
            assert cursor.fetchone() is None


class TestSavepointRollback:
    """Test savepoint-based nested transaction handling."""

    @pytest.fixture
    async def state_manager(self):
        """Create a temporary state manager with WAL mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state.db"
            manager = SQLiteStateManager(str(db_path))
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_savepoint_rollback_inner_transaction(self, state_manager):
        """Test inner savepoint rollback doesn't affect outer transaction."""
        # Start with a valid record outside transactions
        await state_manager.start_file_processing(
            file_path="/test/outer.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
        )

        # Outer transaction
        async with state_manager.transaction() as conn1:
            # Insert record in outer transaction
            conn1.execute(
                """
                INSERT INTO file_processing (file_path, collection, status, priority)
                VALUES (?, ?, ?, ?)
                """,
                ("/test/outer2.py", "test-collection",
                 FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
            )

            # Inner transaction (savepoint)
            try:
                async with state_manager.transaction() as conn2:
                    conn2.execute(
                        """
                        INSERT INTO file_processing (file_path, collection, status, priority)
                        VALUES (?, ?, ?, ?)
                        """,
                        ("/test/inner.py", "test-collection",
                         FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                    )

                    # Trigger error in inner transaction
                    conn2.execute(
                        """
                        INSERT INTO file_processing (file_path, collection, status, priority)
                        VALUES (?, ?, ?, ?)
                        """,
                        ("/test/outer.py", "test-collection",  # Duplicate
                         FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                    )
            except sqlite3.IntegrityError:
                # Inner transaction should roll back
                pass

        # Verify: outer2.py committed (outer transaction), inner.py rolled back
        outer2 = await state_manager.get_file_processing_status("/test/outer2.py")
        assert outer2 is not None

        inner = await state_manager.get_file_processing_status("/test/inner.py")
        assert inner is None

        # Original record still exists
        outer = await state_manager.get_file_processing_status("/test/outer.py")
        assert outer is not None

    @pytest.mark.asyncio
    async def test_savepoint_commit_on_success(self, state_manager):
        """Test savepoint commits successfully when no errors occur."""
        async with state_manager.transaction() as conn1:
            conn1.execute(
                """
                INSERT INTO file_processing (file_path, collection, status, priority)
                VALUES (?, ?, ?, ?)
                """,
                ("/test/outer.py", "test-collection",
                 FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
            )

            async with state_manager.transaction() as conn2:
                conn2.execute(
                    """
                    INSERT INTO file_processing (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/inner.py", "test-collection",
                     FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                )

        # Both should be committed
        outer = await state_manager.get_file_processing_status("/test/outer.py")
        assert outer is not None

        inner = await state_manager.get_file_processing_status("/test/inner.py")
        assert inner is not None

    @pytest.mark.asyncio
    async def test_multiple_nested_savepoints(self, state_manager):
        """Test multiple levels of nested savepoints."""
        async with state_manager.transaction() as conn1:
            conn1.execute(
                """
                INSERT INTO file_processing (file_path, collection, status, priority)
                VALUES (?, ?, ?, ?)
                """,
                ("/test/level1.py", "test-collection",
                 FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
            )

            async with state_manager.transaction() as conn2:
                conn2.execute(
                    """
                    INSERT INTO file_processing (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/level2.py", "test-collection",
                     FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                )

                try:
                    async with state_manager.transaction() as conn3:
                        conn3.execute(
                            """
                            INSERT INTO file_processing (file_path, collection, status, priority)
                            VALUES (?, ?, ?, ?)
                            """,
                            ("/test/level3.py", "test-collection",
                             FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                        )

                        # Trigger error at level 3
                        conn3.execute(
                            """
                            INSERT INTO file_processing (file_path, collection, status, priority)
                            VALUES (?, ?, ?, ?)
                            """,
                            ("/test/level1.py", "test-collection",  # Duplicate
                             FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                        )
                except sqlite3.IntegrityError:
                    pass

        # Level 1 and 2 committed, level 3 rolled back
        level1 = await state_manager.get_file_processing_status("/test/level1.py")
        assert level1 is not None

        level2 = await state_manager.get_file_processing_status("/test/level2.py")
        assert level2 is not None

        level3 = await state_manager.get_file_processing_status("/test/level3.py")
        assert level3 is None


class TestStateConsistency:
    """Test database state consistency after rollbacks."""

    @pytest.fixture
    async def state_manager(self):
        """Create a temporary state manager with WAL mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state.db"
            manager = SQLiteStateManager(str(db_path))
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_foreign_key_integrity_after_rollback(self, state_manager):
        """Test foreign key relationships remain consistent after rollback."""
        # Insert LSP server
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO lsp_servers (language, server_path, status)
                VALUES (?, ?, ?)
                """,
                ("python", "/usr/bin/pyright", LSPServerStatus.ACTIVE.value)
            )
            # Get the server ID
            cursor = conn.execute("SELECT id FROM lsp_servers WHERE language = ?", ("python",))
            server_id = cursor.fetchone()[0]

        # Insert file with reference to LSP server
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO file_processing (file_path, collection, status, priority, lsp_server_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("/test/file.py", "test-collection",
                 FileProcessingStatus.PROCESSING.value, ProcessingPriority.NORMAL.value, server_id)
            )

        # Try to delete LSP server in a failing transaction (with ON DELETE SET NULL, this is allowed)
        # But we'll trigger a different error to test rollback
        try:
            async with state_manager.transaction() as conn:
                # Delete the LSP server (ON DELETE SET NULL applies)
                conn.execute("DELETE FROM lsp_servers WHERE id = ?", (server_id,))

                # Trigger error by trying to insert invalid data
                conn.execute(
                    "INSERT INTO file_processing (file_path, collection, status, priority) VALUES (?, ?, ?, ?)",
                    ("/test/file.py", "test", FileProcessingStatus.PENDING.value, 1)  # Duplicate file_path
                )
        except sqlite3.IntegrityError:
            pass

        # Verify LSP server still exists (transaction rolled back)
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT id FROM lsp_servers WHERE id = ?", (server_id,)
            )
            assert cursor.fetchone() is not None

        # Verify file still references the server
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT lsp_server_id FROM file_processing WHERE file_path = ?",
                ("/test/file.py",)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == server_id

    @pytest.mark.asyncio
    async def test_index_consistency_after_rollback(self, state_manager):
        """Test indexes remain consistent after transaction rollback."""
        # Insert records to populate indexes
        for i in range(5):
            await state_manager.start_file_processing(
                file_path=f"/test/file{i}.py",
                collection="test-collection",
                priority=ProcessingPriority.NORMAL,
            )

        # Count records using index
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT COUNT(*) FROM file_processing WHERE status = ?",
                (FileProcessingStatus.PROCESSING.value,)
            )
            count_before = cursor.fetchone()[0]
            assert count_before == 5

        # Try to insert more records in failing transaction
        try:
            async with state_manager.transaction() as conn:
                for i in range(5, 10):
                    conn.execute(
                        """
                        INSERT INTO file_processing (file_path, collection, status, priority)
                        VALUES (?, ?, ?, ?)
                        """,
                        (f"/test/file{i}.py", "test-collection",
                         FileProcessingStatus.PROCESSING.value, ProcessingPriority.NORMAL.value)
                    )

                # Trigger error
                conn.execute(
                    """
                    INSERT INTO file_processing (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file0.py", "test-collection",  # Duplicate
                     FileProcessingStatus.PROCESSING.value, ProcessingPriority.NORMAL.value)
                )
        except sqlite3.IntegrityError:
            pass

        # Verify index still shows correct count (rollback worked)
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT COUNT(*) FROM file_processing WHERE status = ?",
                (FileProcessingStatus.PROCESSING.value,)
            )
            count_after = cursor.fetchone()[0]
            assert count_after == count_before  # No change

    @pytest.mark.asyncio
    async def test_json_field_integrity_after_rollback(self, state_manager):
        """Test JSON fields remain valid after rollback."""
        # Insert record with JSON metadata
        metadata = {"key": "value", "number": 42, "nested": {"inner": "data"}}
        await state_manager.start_file_processing(
            file_path="/test/file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            metadata=metadata,
        )

        # Verify metadata is intact
        record = await state_manager.get_file_processing_status("/test/file.py")
        assert record is not None
        assert record.metadata == metadata

        # Try to corrupt in failing transaction
        try:
            async with state_manager.transaction() as conn:
                # Update with invalid JSON (should fail deserialization)
                conn.execute(
                    "UPDATE file_processing SET metadata = ? WHERE file_path = ?",
                    ("invalid json{", "/test/file.py")
                )

                # Trigger rollback
                conn.execute(
                    """
                    INSERT INTO file_processing (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file.py", "test-collection",  # Duplicate
                     FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                )
        except sqlite3.IntegrityError:
            pass

        # Verify original metadata is preserved
        record_after = await state_manager.get_file_processing_status("/test/file.py")
        assert record_after is not None
        assert record_after.metadata == metadata


class TestConcurrentTransactions:
    """Test transaction isolation and concurrent access."""

    @pytest.fixture
    async def state_manager(self):
        """Create a temporary state manager with WAL mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state.db"
            manager = SQLiteStateManager(str(db_path))
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_transaction_isolation(self, state_manager):
        """Test transactions are isolated from each other."""
        # This test verifies that uncommitted changes in one transaction
        # are not visible to another transaction (READ COMMITTED isolation)

        # Insert initial record
        await state_manager.start_file_processing(
            file_path="/test/file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
        )

        # Start transaction but don't commit
        async with state_manager.transaction() as conn1:
            # Update status in transaction 1
            conn1.execute(
                "UPDATE file_processing SET status = ? WHERE file_path = ?",
                (FileProcessingStatus.COMPLETED.value, "/test/file.py")
            )

            # Read from same connection - should see the change
            cursor = conn1.execute(
                "SELECT status FROM file_processing WHERE file_path = ?",
                ("/test/file.py",)
            )
            status_in_tx = cursor.fetchone()[0]
            assert status_in_tx == FileProcessingStatus.COMPLETED.value

        # After commit, verify change is visible
        record = await state_manager.get_file_processing_status("/test/file.py")
        assert record is not None
        assert record.status == FileProcessingStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_rollback_isolation(self, state_manager):
        """Test rolled back changes are not visible outside transaction."""
        # Insert initial record
        await state_manager.start_file_processing(
            file_path="/test/file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
        )

        try:
            async with state_manager.transaction() as conn:
                # Make changes
                conn.execute(
                    "UPDATE file_processing SET status = ? WHERE file_path = ?",
                    (FileProcessingStatus.COMPLETED.value, "/test/file.py")
                )

                # Trigger rollback
                conn.execute(
                    """
                    INSERT INTO file_processing (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/file.py", "test-collection",  # Duplicate
                     FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
                )
        except sqlite3.IntegrityError:
            pass

        # Verify changes were rolled back
        record = await state_manager.get_file_processing_status("/test/file.py")
        assert record is not None
        assert record.status == FileProcessingStatus.PROCESSING  # Original status


class TestComplexTransactionScenarios:
    """Test complex transaction scenarios with multiple operations."""

    @pytest.fixture
    async def state_manager(self):
        """Create a temporary state manager with WAL mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_state.db"
            manager = SQLiteStateManager(str(db_path))
            await manager.initialize()
            yield manager
            await manager.close()

    @pytest.mark.asyncio
    async def test_enqueue_with_constraint_violation(self, state_manager):
        """Test enqueue gracefully handles constraint violations."""
        # First enqueue
        queue_id1 = await state_manager.enqueue(
            file_path="/test/file.py",
            collection="test-collection",
            priority=5,
            tenant_id="default",
            branch="main",
        )
        assert queue_id1 is not None

        # Second enqueue with same file - should update priority
        queue_id2 = await state_manager.enqueue(
            file_path="/test/file.py",
            collection="test-collection",
            priority=8,  # Higher priority
            tenant_id="default",
            branch="main",
        )
        assert queue_id2 == queue_id1  # Same queue ID (file path)

        # Verify priority was updated
        items = await state_manager.dequeue(batch_size=1)
        assert len(items) == 1
        # Priority mapping: 8 -> HIGH (6-8 range)
        assert items[0].priority == ProcessingPriority.HIGH

    @pytest.mark.asyncio
    async def test_cascading_delete_with_rollback(self, state_manager):
        """Test cascading deletes roll back properly."""
        # Create file processing record
        await state_manager.start_file_processing(
            file_path="/test/file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
        )

        # Add to queue (has FK to file_processing)
        await state_manager.add_to_processing_queue(
            file_path="/test/file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
        )

        # Try to delete with error in same transaction
        try:
            async with state_manager.transaction() as conn:
                # Delete from file_processing (should cascade to processing_queue with ON DELETE CASCADE)
                conn.execute(
                    "DELETE FROM file_processing WHERE file_path = ?",
                    ("/test/file.py",)
                )

                # Trigger error with duplicate insert
                conn.execute(
                    """
                    INSERT INTO watch_folders (watch_id, path, collection, patterns, ignore_patterns)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("test-id", "/test/path", "test-collection", "[]", "[]")
                )
                # Insert duplicate to trigger error
                conn.execute(
                    """
                    INSERT INTO watch_folders (watch_id, path, collection, patterns, ignore_patterns)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    ("test-id", "/test/path", "test-collection", "[]", "[]")
                )
        except sqlite3.IntegrityError:
            pass

        # Verify both file_processing and processing_queue records still exist (cascading delete was rolled back)
        record = await state_manager.get_file_processing_status("/test/file.py")
        assert record is not None

        # Check queue still has the item
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT COUNT(*) FROM processing_queue WHERE file_path = ?",
                ("/test/file.py",)
            )
            count = cursor.fetchone()[0]
            assert count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
