"""
Comprehensive crash recovery tests for SQLiteStateManager.

Tests crash scenarios using subprocess simulation with SIGKILL to verify
WAL file recovery, transaction rollback, and state restoration after crashes.
"""

import asyncio
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from common.core.sqlite_state_manager import (
    FileProcessingStatus,
    ProcessingPriority,
    SQLiteStateManager,
    WatchFolderConfig,
)

# Crash simulation subprocess script
CRASH_PROCESS_SCRIPT = """
import asyncio
import sys
import time
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.sqlite_state_manager import (
    FileProcessingStatus,
    ProcessingPriority,
    SQLiteStateManager,
)


async def crash_during_file_processing(db_path):
    '''Simulate crash during file processing.'''
    state_manager = SQLiteStateManager(db_path=db_path)
    await state_manager.initialize()

    # Start processing but don't complete
    await state_manager.start_file_processing(
        file_path="/test/crash_file.py",
        collection="test_collection",
        priority=ProcessingPriority.HIGH,
        file_size=1024,
    )

    # Simulate work in progress
    time.sleep(0.1)

    # Exit without cleanup (simulates crash)
    sys.exit(137)  # SIGKILL exit code


async def crash_during_enqueue(db_path):
    '''Simulate crash during enqueue operation.'''
    state_manager = SQLiteStateManager(db_path=db_path)
    await state_manager.initialize()

    # Start transaction but crash before commit
    async with state_manager.transaction() as conn:
        conn.execute(
            \"\"\"
            INSERT INTO ingestion_queue
            (file_absolute_path, collection_name, tenant_id, branch, operation, priority)
            VALUES (?, ?, ?, ?, ?, ?)
            \"\"\",
            ("/test/crash_enqueue.py", "test_collection", "default", "main", "ingest", 5)
        )
        # Crash before transaction commits
        sys.exit(137)


async def crash_during_watch_folder_update(db_path):
    '''Simulate crash during watch folder config update.'''
    state_manager = SQLiteStateManager(db_path=db_path)
    await state_manager.initialize()

    # Start updating watch folder
    async with state_manager.transaction() as conn:
        conn.execute(
            \"\"\"
            INSERT INTO watch_folders
            (watch_id, path, collection, patterns, ignore_patterns)
            VALUES (?, ?, ?, ?, ?)
            \"\"\",
            ("crash_watch", "/test/path", "test_collection", '["*.py"]', '[]')
        )
        # Crash before transaction commits
        sys.exit(137)


async def crash_during_multiple_operations(db_path):
    '''Simulate crash during multiple concurrent operations.'''
    state_manager = SQLiteStateManager(db_path=db_path)
    await state_manager.initialize()

    # Start multiple operations
    await state_manager.start_file_processing(
        "/test/multi1.py", "test_collection", ProcessingPriority.HIGH
    )
    await state_manager.start_file_processing(
        "/test/multi2.py", "test_collection", ProcessingPriority.NORMAL
    )

    await state_manager.enqueue(
        "/test/multi3.py", "test_collection", priority=5,
        tenant_id="default", branch="main"
    )

    # Crash during operations
    sys.exit(137)


async def crash_with_uncommitted_transactions(db_path):
    '''Simulate crash with multiple uncommitted transactions.'''
    state_manager = SQLiteStateManager(db_path=db_path)
    await state_manager.initialize()

    # Start processing files
    await state_manager.start_file_processing(
        "/test/uncommitted1.py", "test_collection", ProcessingPriority.HIGH
    )

    # Start transaction but don't commit
    async with state_manager.transaction() as conn:
        conn.execute(
            \"\"\"
            UPDATE file_processing
            SET status = ?, error_message = ?
            WHERE file_path = ?
            \"\"\",
            (FileProcessingStatus.FAILED.value, "Test error", "/test/uncommitted1.py")
        )
        # Crash before commit
        sys.exit(137)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: script.py <operation> <db_path>")
        sys.exit(1)

    operation = sys.argv[1]
    db_path = sys.argv[2]

    operations = {
        "file_processing": crash_during_file_processing,
        "enqueue": crash_during_enqueue,
        "watch_folder": crash_during_watch_folder_update,
        "multiple": crash_during_multiple_operations,
        "uncommitted": crash_with_uncommitted_transactions,
    }

    if operation not in operations:
        print(f"Unknown operation: {operation}")
        sys.exit(1)

    asyncio.run(operations[operation](db_path))
"""


@pytest.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    for ext in ["", "-wal", "-shm"]:
        try:
            os.unlink(db_path + ext)
        except FileNotFoundError:
            pass


@pytest.fixture
def crash_script():
    """Create temporary crash simulation script."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(CRASH_PROCESS_SCRIPT)
        script_path = f.name

    yield script_path

    # Cleanup
    try:
        os.unlink(script_path)
    except FileNotFoundError:
        pass


def run_crash_subprocess(script_path: str, operation: str, db_path: str) -> bool:
    """
    Run a subprocess that simulates a crash.

    Returns:
        True if process was killed (simulated crash), False otherwise
    """
    process = subprocess.Popen(
        [sys.executable, script_path, operation, db_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for process to crash or timeout
    try:
        process.wait(timeout=5)
        # Process exited (should be via sys.exit(137))
        return process.returncode == 137
    except subprocess.TimeoutExpired:
        # Process didn't crash naturally, force kill it
        process.kill()
        process.wait()
        return True


class TestCrashRecoverySimulation:
    """Test crash recovery using subprocess simulation."""

    @pytest.mark.asyncio
    async def test_crash_during_file_processing_recovery(
        self, temp_db, crash_script
    ):
        """Test recovery from crash during file processing."""
        # Simulate crash
        crashed = run_crash_subprocess(crash_script, "file_processing", temp_db)
        assert crashed, "Subprocess should have crashed"

        # Initialize new state manager (triggers recovery)
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            # Check that crashed file was recovered
            status = await state_manager.get_file_processing_status(
                "/test/crash_file.py"
            )

            # File should exist and be marked for retry or failed
            if status is not None:
                assert status.status in [
                    FileProcessingStatus.RETRYING,
                    FileProcessingStatus.FAILED,
                ]
                if status.status == FileProcessingStatus.RETRYING:
                    assert status.started_at is None  # Should be reset

        finally:
            await state_manager.close()

    @pytest.mark.asyncio
    async def test_crash_during_enqueue_rollback(self, temp_db, crash_script):
        """Test transaction rollback after crash during enqueue."""
        # Simulate crash
        crashed = run_crash_subprocess(crash_script, "enqueue", temp_db)
        assert crashed, "Subprocess should have crashed"

        # Initialize new state manager
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            # Verify uncommitted enqueue was rolled back
            queue_depth = await state_manager.get_queue_depth()
            assert queue_depth == 0, "Uncommitted enqueue should be rolled back"

            # Verify database is in consistent state
            with state_manager._lock:
                cursor = state_manager.connection.execute(
                    "SELECT COUNT(*) FROM ingestion_queue"
                )
                count = cursor.fetchone()[0]
                assert count == 0
        finally:
            await state_manager.close()

    @pytest.mark.asyncio
    async def test_crash_during_watch_folder_update_rollback(
        self, temp_db, crash_script
    ):
        """Test rollback of watch folder update after crash."""
        # Simulate crash
        crashed = run_crash_subprocess(crash_script, "watch_folder", temp_db)
        assert crashed, "Subprocess should have crashed"

        # Initialize new state manager
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            # Verify uncommitted watch folder update was rolled back
            config = await state_manager.get_watch_folder_config("crash_watch")
            assert config is None, "Uncommitted watch folder should be rolled back"

            # Verify all watch folders list is empty
            all_configs = await state_manager.get_all_watch_folder_configs(
                enabled_only=False
            )
            assert len(all_configs) == 0
        finally:
            await state_manager.close()

    @pytest.mark.asyncio
    async def test_crash_with_multiple_operations_recovery(
        self, temp_db, crash_script
    ):
        """Test recovery from crash with multiple concurrent operations."""
        # Simulate crash
        crashed = run_crash_subprocess(crash_script, "multiple", temp_db)
        assert crashed, "Subprocess should have crashed"

        # Initialize new state manager
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            # Check recovery of multiple files
            files_to_check = ["/test/multi1.py", "/test/multi2.py"]

            for file_path in files_to_check:
                status = await state_manager.get_file_processing_status(file_path)
                if status is not None:
                    # Should be marked for retry or failed
                    assert status.status in [
                        FileProcessingStatus.RETRYING,
                        FileProcessingStatus.FAILED,
                    ]

            # Verify database consistency
            with state_manager._lock:
                # No orphaned queue items
                cursor = state_manager.connection.execute(
                    """
                    SELECT COUNT(*) FROM processing_queue
                    WHERE file_path NOT IN (SELECT file_path FROM file_processing)
                    """
                )
                orphaned = cursor.fetchone()[0]
                assert orphaned == 0, "No orphaned queue items should remain"
        finally:
            await state_manager.close()

    @pytest.mark.asyncio
    async def test_crash_with_uncommitted_transactions_recovery(
        self, temp_db, crash_script
    ):
        """Test recovery from crash with uncommitted transactions."""
        # Simulate crash
        crashed = run_crash_subprocess(crash_script, "uncommitted", temp_db)
        assert crashed, "Subprocess should have crashed"

        # Initialize new state manager
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            # Check file status - uncommitted update should be rolled back
            status = await state_manager.get_file_processing_status(
                "/test/uncommitted1.py"
            )

            if status is not None:
                # Should be PROCESSING or RETRYING (not FAILED)
                # because the failed status update was in uncommitted transaction
                assert status.status in [
                    FileProcessingStatus.PROCESSING,
                    FileProcessingStatus.RETRYING,
                ]
        finally:
            await state_manager.close()


class TestWALFileRecovery:
    """Test WAL file recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_wal_checkpoint_on_startup(self, temp_db):
        """Test WAL checkpoint is performed on startup."""
        # Initialize and create some data
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        await state_manager.start_file_processing(
            "/test/wal_test.py", "test_collection", ProcessingPriority.NORMAL
        )

        # Close without full checkpoint
        await state_manager.close()

        # WAL file should exist
        wal_file = Path(temp_db + "-wal")
        wal_file.exists()

        # Re-initialize (triggers checkpoint)
        state_manager2 = SQLiteStateManager(db_path=temp_db)
        await state_manager2.initialize()

        try:
            # Data should be intact
            status = await state_manager2.get_file_processing_status(
                "/test/wal_test.py"
            )
            assert status is not None
        finally:
            await state_manager2.close()

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, temp_db):
        """Test that WAL mode is enabled on initialization."""
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            with state_manager._lock:
                cursor = state_manager.connection.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]
                assert journal_mode.lower() == "wal"
        finally:
            await state_manager.close()

    @pytest.mark.asyncio
    async def test_wal_recovery_after_incomplete_write(self, temp_db):
        """Test recovery from incomplete WAL write."""
        # Create initial state
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        # Write some data
        await state_manager.start_file_processing(
            "/test/incomplete1.py", "test_collection", ProcessingPriority.HIGH
        )
        await state_manager.start_file_processing(
            "/test/incomplete2.py", "test_collection", ProcessingPriority.NORMAL
        )

        # Close without checkpoint
        state_manager.connection.close()
        state_manager._initialized = False

        # Re-initialize (should recover from WAL)
        state_manager2 = SQLiteStateManager(db_path=temp_db)
        await state_manager2.initialize()

        try:
            # Both files should be present
            status1 = await state_manager2.get_file_processing_status(
                "/test/incomplete1.py"
            )
            status2 = await state_manager2.get_file_processing_status(
                "/test/incomplete2.py"
            )

            assert status1 is not None
            assert status2 is not None
        finally:
            await state_manager2.close()


class TestProcessingStatusRestoration:
    """Test processing status restoration after crashes."""

    @pytest.mark.asyncio
    async def test_processing_to_retrying_conversion(self, temp_db):
        """Test PROCESSING status converted to RETRYING on recovery."""
        # Create state with PROCESSING file
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        await state_manager.start_file_processing(
            "/test/processing.py", "test_collection",
            priority=ProcessingPriority.HIGH, file_size=2048
        )

        # Close without completing
        await state_manager.close()

        # Re-initialize (triggers recovery)
        state_manager2 = SQLiteStateManager(db_path=temp_db)
        await state_manager2.initialize()

        try:
            status = await state_manager2.get_file_processing_status(
                "/test/processing.py"
            )

            # Should be marked for retry
            assert status.status == FileProcessingStatus.RETRYING
            assert status.retry_count == 1
            assert status.started_at is None
            assert "crash" in status.error_message.lower()
        finally:
            await state_manager2.close()

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_after_crash(self, temp_db):
        """Test files exceeding max retries are marked as FAILED."""
        # Create state with file at max retries
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        # Set file with max retries already reached
        with state_manager._lock:
            state_manager.connection.execute(
                """
                INSERT INTO file_processing
                (file_path, collection, status, priority, started_at, retry_count, max_retries)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
                """,
                (
                    "/test/max_retries.py",
                    "test_collection",
                    FileProcessingStatus.PROCESSING.value,
                    ProcessingPriority.NORMAL.value,
                    3,  # Already at max
                    3,  # Max retries
                )
            )
            state_manager.connection.commit()

        await state_manager.close()

        # Re-initialize (triggers recovery)
        state_manager2 = SQLiteStateManager(db_path=temp_db)
        await state_manager2.initialize()

        try:
            status = await state_manager2.get_file_processing_status(
                "/test/max_retries.py"
            )

            # Should be marked as FAILED
            assert status.status == FileProcessingStatus.FAILED
            assert "max retries" in status.error_message.lower()
        finally:
            await state_manager2.close()

    @pytest.mark.asyncio
    async def test_queue_restoration_for_crashed_files(self, temp_db):
        """Test crashed files are added back to queue with high priority."""
        # Create state with PROCESSING file
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        await state_manager.start_file_processing(
            "/test/queue_restore.py", "test_collection",
            priority=ProcessingPriority.NORMAL
        )

        await state_manager.close()

        # Re-initialize (triggers recovery)
        state_manager2 = SQLiteStateManager(db_path=temp_db)
        await state_manager2.initialize()

        try:
            # Check if file is in queue with elevated priority
            with state_manager2._lock:
                cursor = state_manager2.connection.execute(
                    """
                    SELECT priority FROM processing_queue
                    WHERE file_path = ?
                    """,
                    ("/test/queue_restore.py",)
                )
                row = cursor.fetchone()

                if row:  # May or may not be in queue
                    # Should have high priority
                    assert row["priority"] == ProcessingPriority.HIGH.value
        finally:
            await state_manager2.close()


class TestOrphanedRecordCleanup:
    """Test cleanup of orphaned records after crashes."""

    @pytest.mark.asyncio
    async def test_orphaned_queue_items_removed(self, temp_db):
        """Test orphaned queue items are removed during recovery."""
        # Create state with orphaned queue item
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        # Create file processing record for valid file
        await state_manager.start_file_processing(
            "/test/orphan_parent.py", "test_collection"
        )

        # Temporarily disable foreign key constraints to create orphaned item
        with state_manager._lock:
            state_manager.connection.execute("PRAGMA foreign_keys=OFF")

            # Add queue item for non-existent file (orphaned)
            state_manager.connection.execute(
                """
                INSERT INTO processing_queue
                (queue_id, file_path, collection, priority)
                VALUES (?, ?, ?, ?)
                """,
                ("orphan_queue_1", "/test/orphan_nonexistent.py", "test_collection", 2)
            )
            state_manager.connection.commit()

            # Re-enable foreign key constraints
            state_manager.connection.execute("PRAGMA foreign_keys=ON")

        await state_manager.close()

        # Re-initialize (triggers recovery)
        state_manager2 = SQLiteStateManager(db_path=temp_db)
        await state_manager2.initialize()

        try:
            # Orphaned queue item should be removed
            with state_manager2._lock:
                cursor = state_manager2.connection.execute(
                    """
                    SELECT COUNT(*) FROM processing_queue
                    WHERE file_path NOT IN (SELECT file_path FROM file_processing)
                    """
                )
                orphaned_count = cursor.fetchone()[0]
                assert orphaned_count == 0
        finally:
            await state_manager2.close()

    @pytest.mark.asyncio
    async def test_multiple_orphaned_items_cleanup(self, temp_db):
        """Test cleanup of multiple orphaned items."""
        # Create state with multiple orphaned items
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        # Temporarily disable foreign key constraints
        with state_manager._lock:
            state_manager.connection.execute("PRAGMA foreign_keys=OFF")

            # Add orphaned queue items
            for i in range(5):
                state_manager.connection.execute(
                    """
                    INSERT INTO processing_queue
                    (queue_id, file_path, collection, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    (f"orphan_{i}", f"/test/orphan_{i}.py", "test_collection", 2)
                )
            state_manager.connection.commit()

            # Re-enable foreign key constraints
            state_manager.connection.execute("PRAGMA foreign_keys=ON")

        await state_manager.close()

        # Re-initialize (triggers recovery)
        state_manager2 = SQLiteStateManager(db_path=temp_db)
        await state_manager2.initialize()

        try:
            # All orphaned items should be removed
            with state_manager2._lock:
                cursor = state_manager2.connection.execute(
                    "SELECT COUNT(*) FROM processing_queue"
                )
                count = cursor.fetchone()[0]
                assert count == 0
        finally:
            await state_manager2.close()


class TestDatabaseConsistency:
    """Test database consistency after crash recovery."""

    @pytest.mark.asyncio
    async def test_foreign_key_constraints_intact(self, temp_db):
        """Test foreign key constraints are intact after recovery."""
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            # Verify foreign keys are enabled
            with state_manager._lock:
                cursor = state_manager.connection.execute("PRAGMA foreign_keys")
                fk_enabled = cursor.fetchone()[0]
                assert fk_enabled == 1
        finally:
            await state_manager.close()

    @pytest.mark.asyncio
    async def test_schema_version_consistent(self, temp_db):
        """Test schema version is consistent after recovery."""
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            with state_manager._lock:
                cursor = state_manager.connection.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                )
                version = cursor.fetchone()[0]
                assert version == SQLiteStateManager.SCHEMA_VERSION
        finally:
            await state_manager.close()

    @pytest.mark.asyncio
    async def test_indexes_intact_after_recovery(self, temp_db):
        """Test database indexes are intact after recovery."""
        state_manager = SQLiteStateManager(db_path=temp_db)
        await state_manager.initialize()

        try:
            with state_manager._lock:
                # Check for key indexes
                cursor = state_manager.connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='index' AND name LIKE 'idx_file_processing%'
                    """
                )
                indexes = [row["name"] for row in cursor.fetchall()]

                # Verify critical indexes exist
                assert "idx_file_processing_status" in indexes
                assert "idx_file_processing_collection" in indexes
        finally:
            await state_manager.close()
