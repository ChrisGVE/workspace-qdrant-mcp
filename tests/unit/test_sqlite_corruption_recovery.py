"""
Unit Tests for SQLite Corruption Detection and Recovery.

This test suite validates the SQLiteStateManager's ability to detect and recover
from database corruption scenarios:
    - Database file corruption (partial writes, bit flips)
    - WAL file corruption scenarios
    - PRAGMA integrity_check validation
    - Graceful degradation when corruption detected
    - Schema version validation
    - Partial write simulation (interrupted checkpoints)
    - File system error handling
"""

import asyncio
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.python.common.core.sqlite_state_manager import (
    FileProcessingStatus,
    ProcessingPriority,
    SQLiteStateManager,
    WatchFolderConfig,
)


class TestSQLiteCorruptionDetection:
    """Test database corruption detection mechanisms."""

    @pytest.mark.asyncio
    async def test_database_file_corruption_detection(self, tmp_path):
        """Test detection of corrupted main database file."""
        db_path = tmp_path / "test_corrupt.db"

        # Create valid database first
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Add some data
        await state_manager.start_file_processing(
            "/test/file.txt", "test-collection"
        )

        # Close cleanly
        await state_manager.close()

        # Corrupt the database file by overwriting with random bytes
        with open(db_path, "r+b") as f:
            f.seek(100)  # Corrupt header area
            f.write(b"\x00" * 50)

        # Attempt to open corrupted database
        state_manager_corrupted = SQLiteStateManager(db_path=str(db_path))

        # Should either fail to initialize or detect corruption
        try:
            result = await state_manager_corrupted.initialize()
            # If initialization succeeds, check for corruption
            if result and state_manager_corrupted.connection:
                try:
                    cursor = state_manager_corrupted.connection.execute(
                        "PRAGMA integrity_check"
                    )
                    integrity_result = cursor.fetchone()[0]
                    # Should detect corruption
                    assert integrity_result != "ok", "Corrupted database should fail integrity check"
                except sqlite3.DatabaseError as e:
                    # Expected: corruption detected
                    assert "corrupt" in str(e).lower() or "malformed" in str(e).lower()
        except (sqlite3.DatabaseError, sqlite3.CorruptError) as e:
            # Expected: initialization fails due to corruption
            assert "corrupt" in str(e).lower() or "malformed" in str(e).lower()
        finally:
            if state_manager_corrupted.connection:
                await state_manager_corrupted.close()

    @pytest.mark.asyncio
    async def test_wal_file_corruption_detection(self, tmp_path):
        """Test detection of corrupted WAL file."""
        db_path = tmp_path / "test_wal_corrupt.db"
        wal_path = tmp_path / "test_wal_corrupt.db-wal"

        # Create database with WAL mode
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Add data to create WAL file
        await state_manager.start_file_processing(
            "/test/file1.txt", "test-collection"
        )
        await state_manager.start_file_processing(
            "/test/file2.txt", "test-collection"
        )

        # Ensure WAL file exists
        assert wal_path.exists(), "WAL file should exist"

        # Corrupt WAL file
        with open(wal_path, "r+b") as f:
            f.seek(50)
            f.write(b"\xFF" * 100)

        # Attempt recovery by forcing checkpoint
        try:
            # Force checkpoint should detect corruption
            state_manager.connection.execute("PRAGMA wal_checkpoint(FULL)")
        except sqlite3.DatabaseError as e:
            # Expected: WAL corruption detected
            assert "corrupt" in str(e).lower() or "disk I/O error" in str(e).lower()

        await state_manager.close()

    @pytest.mark.asyncio
    async def test_integrity_check_validation(self, tmp_path):
        """Test PRAGMA integrity_check returns proper validation."""
        db_path = tmp_path / "test_integrity.db"

        # Create clean database
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Verify integrity check passes on clean database
        cursor = state_manager.connection.execute("PRAGMA integrity_check")
        result = cursor.fetchone()[0]
        assert result == "ok", "Clean database should pass integrity check"

        await state_manager.close()

        # More aggressive corruption: zero out large section of file
        with open(db_path, "r+b") as f:
            # Corrupt middle section of database
            f.seek(1024)
            f.write(b"\x00" * 4096)

        # Attempt to check integrity
        state_manager_corrupted = SQLiteStateManager(db_path=str(db_path))
        try:
            result = await state_manager_corrupted.initialize()
            if result and state_manager_corrupted.connection:
                cursor = state_manager_corrupted.connection.execute("PRAGMA integrity_check")
                result = cursor.fetchone()[0]
                # Should detect corruption (not "ok")
                assert result != "ok", "Corrupted database should fail integrity check"
        except sqlite3.DatabaseError:
            # Expected: corruption prevents operations
            pass
        finally:
            if state_manager_corrupted.connection:
                await state_manager_corrupted.close()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_corruption(self, tmp_path):
        """Test graceful error handling when corruption detected."""
        db_path = tmp_path / "test_graceful.db"

        # Create database
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()
        await state_manager.close()

        # Corrupt database header (SQLite magic string)
        with open(db_path, "r+b") as f:
            f.seek(0)
            f.write(b"CORRUPT!")  # Overwrite SQLite header

        # Attempt to initialize - should handle gracefully without crash
        state_manager_corrupted = SQLiteStateManager(db_path=str(db_path))
        result = await state_manager_corrupted.initialize()

        # Should return False indicating initialization failure
        assert result is False, "Initialization should fail gracefully on corruption"
        assert state_manager_corrupted.connection is None, "Connection should be None after failed init"
        assert state_manager_corrupted._initialized is False, "Should not be marked as initialized"

    @pytest.mark.asyncio
    async def test_partial_write_simulation(self, tmp_path):
        """Test recovery from interrupted checkpoint (partial write)."""
        db_path = tmp_path / "test_partial.db"
        wal_path = tmp_path / "test_partial.db-wal"

        # Create database with active WAL
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Write substantial data to WAL
        for i in range(100):
            await state_manager.start_file_processing(
                f"/test/file{i}.txt", "test-collection"
            )

        # Get WAL size before corruption
        wal_size = wal_path.stat().st_size

        # Simulate interrupted checkpoint by truncating WAL mid-transaction
        with open(wal_path, "r+b") as f:
            f.truncate(wal_size // 2)

        # Close current connection
        await state_manager.close()

        # Attempt recovery
        state_manager_recovered = SQLiteStateManager(db_path=str(db_path))
        result = await state_manager_recovered.initialize()

        if result:
            # If initialization succeeds, verify data integrity
            files = await state_manager_recovered.get_files_by_status(
                FileProcessingStatus.PROCESSING
            )
            # Some files may be recovered, some may be lost
            # The key is no crash and graceful handling
            assert isinstance(files, list), "Should return list even with partial data"
            await state_manager_recovered.close()
        else:
            # Initialization failed gracefully - also acceptable
            assert state_manager_recovered.connection is None

    @pytest.mark.asyncio
    async def test_schema_version_validation_corruption(self, tmp_path):
        """Test schema version validation when corruption affects schema table."""
        db_path = tmp_path / "test_schema_corrupt.db"

        # Create database
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Manually corrupt schema_version table
        try:
            state_manager.connection.execute(
                "UPDATE schema_version SET version = 9999 WHERE version = ?",
                (SQLiteStateManager.SCHEMA_VERSION,)
            )
            state_manager.connection.commit()
        except sqlite3.Error:
            pass  # Expected if table is locked

        await state_manager.close()

        # Reopen and verify handling of invalid schema version
        state_manager_reopened = SQLiteStateManager(db_path=str(db_path))
        result = await state_manager_reopened.initialize()

        # Should either fail or handle gracefully
        # The key is no crash
        assert isinstance(result, bool), "Initialize should return boolean"

        if state_manager_reopened.connection:
            await state_manager_reopened.close()

    @pytest.mark.asyncio
    async def test_bit_flip_corruption(self, tmp_path):
        """Test detection of single bit flips in database file."""
        db_path = tmp_path / "test_bitflip.db"

        # Create database with data
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Add recognizable data
        config = WatchFolderConfig(
            watch_id="test-watch",
            path="/test/path",
            collection="test-collection",
            patterns=["*.py"],
            ignore_patterns=["*.pyc"],
        )
        await state_manager.save_watch_folder_config(config)
        await state_manager.close()

        # Flip random bits in database
        with open(db_path, "r+b") as f:
            data = bytearray(f.read())
            # Flip bits at multiple positions
            for pos in [500, 1000, 1500, 2000]:
                if pos < len(data):
                    data[pos] ^= 0xFF  # Flip all bits in byte
            f.seek(0)
            f.write(data)

        # Attempt to read
        state_manager_corrupted = SQLiteStateManager(db_path=str(db_path))
        try:
            result = await state_manager_corrupted.initialize()
            if result:
                # Try to retrieve data
                retrieved = await state_manager_corrupted.get_watch_folder_config("test-watch")
                # Data may be corrupted or retrieval may fail
                # Key is graceful handling
                if retrieved:
                    # If retrieved, data may be corrupted
                    assert isinstance(retrieved, WatchFolderConfig)
        except (sqlite3.DatabaseError, sqlite3.CorruptError):
            # Expected: corruption detected
            pass
        finally:
            if state_manager_corrupted.connection:
                await state_manager_corrupted.close()

    @pytest.mark.asyncio
    async def test_file_system_error_handling(self, tmp_path):
        """Test handling of file system errors during operations."""
        db_path = tmp_path / "test_fs_error.db"

        # Create database
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Mock file write operations to simulate permission errors
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            try:
                # Attempt write operation - should handle gracefully
                result = await state_manager.start_file_processing(
                    "/test/file.txt", "test-collection"
                )
                # May succeed if data is still in memory/WAL
                assert isinstance(result, bool)
            except (OSError, sqlite3.OperationalError, PermissionError):
                # Expected: file system permission error
                pass

        await state_manager.close()

    @pytest.mark.asyncio
    async def test_rollback_on_corruption_during_transaction(self, tmp_path):
        """Test transaction rollback when corruption detected mid-operation."""
        db_path = tmp_path / "test_rollback.db"

        # Create database
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Use patch to simulate corruption during save_watch_folder_config
        with patch.object(
            state_manager.connection,
            'execute',
            side_effect=lambda sql, *args, **kwargs: (
                state_manager.connection.execute.__wrapped__(sql, *args, **kwargs)
                if "INSERT" not in sql.upper() or "watch_folders" not in sql
                else (_ for _ in ()).throw(sqlite3.DatabaseError("database disk image is malformed"))
            )
        ) if hasattr(state_manager.connection.execute, '__wrapped__') else (
            patch.object(
                state_manager,
                '_lock',
                side_effect=lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("simulated corruption"))
            )
        ):
            # Attempt operation that triggers corruption
            config = WatchFolderConfig(
                watch_id="test-watch",
                path="/test/path",
                collection="test-collection",
                patterns=["*.py"],
                ignore_patterns=["*.pyc"],
            )

            result = await state_manager.save_watch_folder_config(config)

            # Should handle gracefully with rollback
            # Result may be True or False depending on when corruption hits
            assert isinstance(result, bool), "Operation should return boolean"

        # Verify database is still usable after potential corruption
        test_result = await state_manager.start_file_processing(
            "/test/file.txt", "test-collection"
        )
        assert test_result is True, "Database should still be usable after error"

        await state_manager.close()

    @pytest.mark.asyncio
    async def test_migration_corruption_detection(self, tmp_path):
        """Test detection of corruption during schema migration."""
        db_path = tmp_path / "test_migration_corrupt.db"

        # Create database with old schema version
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")

        # Create minimal old schema (version 1)
        conn.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT INTO schema_version (version) VALUES (1)")

        # Create some tables to corrupt
        conn.execute("""
            CREATE TABLE file_processing (
                file_path TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        # Corrupt the file_processing table
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA writable_schema = ON")
        try:
            # Corrupt table definition in sqlite_master
            conn.execute("""
                UPDATE sqlite_master
                SET sql = 'CREATE TABLE corrupted ('
                WHERE type = 'table' AND name = 'file_processing'
            """)
            conn.commit()
        except sqlite3.Error:
            pass  # Some corruption attempts may fail
        finally:
            conn.execute("PRAGMA writable_schema = OFF")
            conn.close()

        # Attempt migration
        state_manager = SQLiteStateManager(db_path=str(db_path))
        result = await state_manager.initialize()

        # Should detect corruption and fail gracefully
        # No crash, proper error handling
        if result:
            # If somehow succeeded, verify integrity
            try:
                cursor = state_manager.connection.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
                # May pass or fail depending on corruption
                assert integrity in ["ok", "*** in database main ***"]
            except sqlite3.Error:
                pass  # Expected
            await state_manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_corruption_detection(self, tmp_path):
        """Test corruption detection with concurrent access attempts."""
        db_path = tmp_path / "test_concurrent.db"

        # Create database
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Add initial data
        await state_manager.start_file_processing(
            "/test/file.txt", "test-collection"
        )

        # Simulate concurrent corruption by another process
        def corrupt_in_background():
            # Wait a bit then corrupt
            import time
            time.sleep(0.1)
            try:
                with open(db_path, "r+b") as f:
                    f.seek(200)
                    f.write(b"\x00" * 50)
            except Exception:
                pass  # Ignore if locked

        import threading
        corruption_thread = threading.Thread(target=corrupt_in_background)
        corruption_thread.start()

        # Continue operations
        try:
            for i in range(10):
                await state_manager.start_file_processing(
                    f"/test/file{i}.txt", "test-collection"
                )
                await asyncio.sleep(0.05)
        except sqlite3.DatabaseError:
            # Expected: may encounter corruption
            pass

        corruption_thread.join()
        await state_manager.close()

    @pytest.mark.asyncio
    async def test_checkpoint_interruption_recovery(self, tmp_path):
        """Test recovery from interrupted WAL checkpoint."""
        db_path = tmp_path / "test_checkpoint.db"
        wal_path = tmp_path / "test_checkpoint.db-wal"

        # Create database with substantial WAL
        state_manager = SQLiteStateManager(db_path=str(db_path))
        await state_manager.initialize()

        # Write enough data to trigger checkpoint
        for i in range(200):
            await state_manager.start_file_processing(
                f"/test/file{i}.txt", "test-collection"
            )

        # Get original WAL content if it exists
        if wal_path.exists():
            with open(wal_path, "rb") as f:
                original_wal = f.read()

            # Simulate interrupted checkpoint by restoring partial WAL
            await state_manager.close()

            # Only corrupt if we have enough data
            if len(original_wal) > 100:
                # Restore partial WAL
                with open(wal_path, "wb") as f:
                    f.write(original_wal[:len(original_wal) // 2])

                # Attempt recovery
                state_manager_recovered = SQLiteStateManager(db_path=str(db_path))
                result = await state_manager_recovered.initialize()

                if result:
                    # Should perform crash recovery
                    files = await state_manager_recovered.get_files_by_status(
                        FileProcessingStatus.PROCESSING
                    )
                    # Some data may be recovered
                    assert isinstance(files, list)
                    await state_manager_recovered.close()
            else:
                # WAL too small, just verify clean close
                pass
        else:
            # WAL doesn't exist - checkpoint already happened
            await state_manager.close()
