"""
Tests for SQLite concurrent access and schema migration.

This module tests:
- WAL mode concurrent reader-writer operations
- Lock contention and busy timeout handling
- Schema version validation
- Data preservation during migrations
"""

import asyncio
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.python.common.core.sqlite_state_manager import (
    FileProcessingStatus,
    ProcessingPriority,
    SQLiteStateManager,
    WatchFolderConfig,
)


@pytest.fixture
async def temp_db_path():
    """Create temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    yield db_path
    # Cleanup
    for suffix in ["", "-shm", "-wal"]:
        file = Path(str(db_path) + suffix)
        if file.exists():
            file.unlink()


@pytest.fixture
async def state_manager(temp_db_path):
    """Create and initialize state manager."""
    manager = SQLiteStateManager(db_path=str(temp_db_path))
    await manager.initialize()
    yield manager
    await manager.close()


class TestConcurrentAccess:
    """Test WAL mode concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_readers(self, temp_db_path):
        """Test multiple concurrent readers can access database."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        # Insert test data
        config = WatchFolderConfig(
            watch_id="test-watch",
            path="/test/path",
            collection="test-collection",
            patterns=["*.py"],
            ignore_patterns=["*.pyc"],
        )
        await manager.save_watch_folder_config(config)

        # Function for concurrent reads
        async def read_watch_config(reader_id: int, results: list):
            await asyncio.sleep(0.01)  # Small delay to ensure overlap
            config = await manager.get_watch_folder_config("test-watch")
            results.append((reader_id, config is not None, config.watch_id if config else None))

        # Launch 10 concurrent readers
        results = []
        tasks = [read_watch_config(i, results) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all readers succeeded
        assert len(results) == 10
        for _reader_id, success, watch_id in results:
            assert success is True
            assert watch_id == "test-watch"

        await manager.close()

    @pytest.mark.asyncio
    async def test_reader_writer_concurrency(self, temp_db_path):
        """Test concurrent readers and single writer."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        write_count = 0
        read_results = []

        async def writer():
            """Write watch configs continuously."""
            nonlocal write_count
            for i in range(5):
                config = WatchFolderConfig(
                    watch_id=f"watch-{i}",
                    path=f"/test/path/{i}",
                    collection=f"collection-{i}",
                    patterns=["*.py"],
                    ignore_patterns=["*.pyc"],
                )
                await manager.save_watch_folder_config(config)
                write_count += 1
                await asyncio.sleep(0.01)

        async def reader(reader_id: int):
            """Read watch configs continuously."""
            for _ in range(10):
                configs = await manager.get_all_watch_folder_configs(enabled_only=False)
                read_results.append((reader_id, len(configs)))
                await asyncio.sleep(0.005)

        # Run 1 writer and 5 readers concurrently
        tasks = [writer()] + [reader(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Verify writes completed
        assert write_count == 5

        # Verify reads succeeded
        assert len(read_results) == 50  # 5 readers × 10 reads each
        for _reader_id, count in read_results:
            assert count >= 0  # Should read successfully

        await manager.close()

    @pytest.mark.asyncio
    async def test_lock_contention_handling(self, temp_db_path):
        """Test that lock contention is handled gracefully."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        errors = []
        success_count = 0

        async def concurrent_writer(writer_id: int):
            """Attempt concurrent writes."""
            nonlocal success_count
            try:
                config = WatchFolderConfig(
                    watch_id=f"watch-{writer_id}",
                    path=f"/test/path/{writer_id}",
                    collection=f"collection-{writer_id}",
                    patterns=["*.py"],
                    ignore_patterns=["*.pyc"],
                )
                await manager.save_watch_folder_config(config)
                success_count += 1
            except Exception as e:
                errors.append((writer_id, str(e)))

        # Launch 10 concurrent writers
        tasks = [concurrent_writer(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # All writes should succeed (WAL mode handles concurrency)
        assert success_count == 10
        assert len(errors) == 0

        await manager.close()

    @pytest.mark.asyncio
    async def test_busy_timeout_behavior(self, temp_db_path):
        """Test busy timeout during long-running transaction."""
        manager1 = SQLiteStateManager(db_path=str(temp_db_path))
        await manager1.initialize()

        manager2 = SQLiteStateManager(db_path=str(temp_db_path))
        await manager2.initialize()

        # Start a transaction in manager1
        async with manager1.transaction() as conn:
            # Insert data
            conn.execute(
                """
                INSERT INTO watch_folders
                (watch_id, path, collection, patterns, ignore_patterns)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("test-watch", "/test", "test-collection", "[]", "[]"),
            )

            # Meanwhile, try to read from manager2 (should succeed in WAL mode)
            config = await manager2.get_watch_folder_config("test-watch")
            # Won't see uncommitted data
            assert config is None

        # After transaction commits, manager2 should see the data
        await asyncio.sleep(0.1)
        config = await manager2.get_watch_folder_config("test-watch")
        assert config is not None
        assert config.watch_id == "test-watch"

        await manager1.close()
        await manager2.close()

    @pytest.mark.asyncio
    async def test_multiple_writers_queuing(self, temp_db_path):
        """Test that multiple writers queue properly."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        write_order = []

        async def sequential_writer(writer_id: int):
            """Write and record order."""
            config = WatchFolderConfig(
                watch_id=f"watch-{writer_id}",
                path=f"/test/path/{writer_id}",
                collection=f"collection-{writer_id}",
                patterns=["*.py"],
                ignore_patterns=["*.pyc"],
            )
            await manager.save_watch_folder_config(config)
            write_order.append(writer_id)

        # Launch writers
        tasks = [sequential_writer(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # All writes should complete
        assert len(write_order) == 5
        assert set(write_order) == {0, 1, 2, 3, 4}

        # Verify all configs were saved
        configs = await manager.get_all_watch_folder_configs(enabled_only=False)
        assert len(configs) == 5

        await manager.close()

    @pytest.mark.asyncio
    async def test_concurrent_read_operations(self, temp_db_path):
        """Test various concurrent read operations."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        # Setup test data
        for i in range(5):
            config = WatchFolderConfig(
                watch_id=f"watch-{i}",
                path=f"/test/path/{i}",
                collection=f"collection-{i}",
                patterns=["*.py"],
                ignore_patterns=["*.pyc"],
            )
            await manager.save_watch_folder_config(config)

            await manager.start_file_processing(
                file_path=f"/test/file-{i}.py",
                collection=f"collection-{i}",
                priority=ProcessingPriority.NORMAL,
            )

        read_results = []

        async def read_watch_configs():
            """Read watch configs."""
            configs = await manager.get_all_watch_folder_configs(enabled_only=False)
            read_results.append(("watch", len(configs)))

        async def read_file_status():
            """Read file processing status."""
            files = await manager.get_files_by_status(FileProcessingStatus.PROCESSING)
            read_results.append(("files", len(files)))

        async def read_queue_depth():
            """Read queue depth."""
            depth = await manager.get_queue_depth()
            read_results.append(("queue", depth))

        # Run different read operations concurrently
        tasks = []
        for _ in range(3):
            tasks.extend([read_watch_configs(), read_file_status(), read_queue_depth()])

        await asyncio.gather(*tasks)

        # Verify all reads succeeded
        assert len(read_results) == 9
        watch_results = [r for r in read_results if r[0] == "watch"]
        assert all(count == 5 for _, count in watch_results)

        await manager.close()


class TestSchemaMigration:
    """Test schema version tracking and validation."""

    @pytest.mark.asyncio
    async def test_initial_schema_version(self, temp_db_path):
        """Test initial schema creates correct version."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        # Check schema version
        with manager._lock:
            cursor = manager.connection.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            version = cursor.fetchone()[0]

        assert version == SQLiteStateManager.SCHEMA_VERSION

        await manager.close()

    @pytest.mark.asyncio
    async def test_schema_version_validation(self, temp_db_path):
        """Test schema version is correctly tracked."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        # Check schema version table
        with manager._lock:
            cursor = manager.connection.execute(
                "SELECT version, applied_at FROM schema_version ORDER BY version ASC"
            )
            rows = cursor.fetchall()

        # Should have at least one version entry
        assert len(rows) >= 1

        # Latest version should match current schema
        latest_version = rows[-1]["version"]
        assert latest_version == SQLiteStateManager.SCHEMA_VERSION

        # All versions should have timestamps
        for row in rows:
            assert row["applied_at"] is not None

        await manager.close()

    @pytest.mark.asyncio
    async def test_wal_mode_enabled(self, temp_db_path):
        """Test that WAL mode is properly enabled."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        # Check WAL mode
        with manager._lock:
            cursor = manager.connection.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]

        assert mode.lower() == "wal"

        await manager.close()

    @pytest.mark.asyncio
    async def test_database_pragmas(self, temp_db_path):
        """Test that database pragmas are correctly set."""
        manager = SQLiteStateManager(db_path=str(temp_db_path))
        await manager.initialize()

        with manager._lock:
            # Check synchronous mode
            cursor = manager.connection.execute("PRAGMA synchronous")
            sync_mode = cursor.fetchone()[0]
            assert sync_mode == 1  # NORMAL

            # Check foreign keys are enabled
            cursor = manager.connection.execute("PRAGMA foreign_keys")
            fk_enabled = cursor.fetchone()[0]
            assert fk_enabled == 1

        await manager.close()
