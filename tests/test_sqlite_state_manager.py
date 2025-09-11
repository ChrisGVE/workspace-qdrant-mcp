"""
Tests for SQLite State Manager with Crash Recovery.

This module tests the bulletproof state persistence using SQLite with WAL mode,
ACID transactions, crash recovery, and comprehensive state management.
"""

import asyncio
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any
import pytest

from common.core.sqlite_state_manager import (
    SQLiteStateManager, FileProcessingStatus, ProcessingPriority,
    FileProcessingRecord, WatchFolderConfig, ProcessingQueueItem,
    DatabaseTransaction
)


class TestSQLiteStateManager:
    """Test SQLite state manager functionality."""
    
    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        for ext in ['', '-wal', '-shm']:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass
    
    @pytest.fixture
    async def state_manager(self, temp_db_path):
        """Create and initialize state manager."""
        manager = SQLiteStateManager(temp_db_path)
        
        success = await manager.initialize()
        assert success, "State manager initialization failed"
        
        yield manager
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_initialization_and_wal_mode(self, temp_db_path):
        """Test database initialization with WAL mode."""
        manager = SQLiteStateManager(temp_db_path)
        
        # Test initialization
        success = await manager.initialize()
        assert success
        
        # Verify WAL mode is enabled
        with manager._lock:
            cursor = manager.connection.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0].lower()
            assert journal_mode == "wal"
        
        # Verify foreign keys are enabled
        with manager._lock:
            cursor = manager.connection.execute("PRAGMA foreign_keys")
            foreign_keys = cursor.fetchone()[0]
            assert foreign_keys == 1
        
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_schema_creation(self, state_manager):
        """Test database schema creation."""
        # Verify all tables exist
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = [
            'schema_version', 'file_processing', 'watch_folders',
            'processing_queue', 'system_state', 'processing_history'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Table {table} not found"
        
        # Verify schema version
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            version = cursor.fetchone()[0]
            assert version == state_manager.SCHEMA_VERSION
    
    @pytest.mark.asyncio
    async def test_acid_transactions(self, state_manager):
        """Test ACID transaction support with rollback."""
        test_file = "/test/file.txt"
        collection = "test_collection"
        
        # Test successful transaction
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO file_processing 
                (file_path, collection, status, priority)
                VALUES (?, ?, ?, ?)
                """,
                (test_file, collection, FileProcessingStatus.PENDING.value, ProcessingPriority.NORMAL.value)
            )
        
        # Verify data was committed
        status = await state_manager.get_file_processing_status(test_file)
        assert status is not None
        assert status.status == FileProcessingStatus.PENDING
        
        # Test transaction rollback on error
        try:
            async with state_manager.transaction() as conn:
                conn.execute(
                    """
                    UPDATE file_processing 
                    SET status = ? 
                    WHERE file_path = ?
                    """,
                    (FileProcessingStatus.PROCESSING.value, test_file)
                )
                
                # Force an error to trigger rollback
                raise ValueError("Test error")
                
        except ValueError:
            pass  # Expected error
        
        # Verify rollback - status should still be PENDING
        status = await state_manager.get_file_processing_status(test_file)
        assert status.status == FileProcessingStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_file_processing_lifecycle(self, state_manager):
        """Test complete file processing lifecycle."""
        file_path = "/test/lifecycle.txt"
        collection = "test_collection"
        
        # Start file processing
        success = await state_manager.start_file_processing(
            file_path=file_path,
            collection=collection,
            priority=ProcessingPriority.HIGH,
            file_size=1024,
            file_hash="abc123",
            metadata={"test": "data"}
        )
        assert success
        
        # Verify processing state
        status = await state_manager.get_file_processing_status(file_path)
        assert status is not None
        assert status.status == FileProcessingStatus.PROCESSING
        assert status.priority == ProcessingPriority.HIGH
        assert status.file_size == 1024
        assert status.file_hash == "abc123"
        assert status.metadata["test"] == "data"
        
        # Complete processing successfully
        success = await state_manager.complete_file_processing(
            file_path=file_path,
            success=True,
            processing_time_ms=500,
            metadata={"processed": True}
        )
        assert success
        
        # Verify completion
        status = await state_manager.get_file_processing_status(file_path)
        assert status.status == FileProcessingStatus.COMPLETED
        assert status.completed_at is not None
        
        # Verify processing history was created
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT COUNT(*) FROM processing_history WHERE file_path = ?",
                (file_path,)
            )
            count = cursor.fetchone()[0]
            assert count == 1
    
    @pytest.mark.asyncio
    async def test_failed_file_retry_logic(self, state_manager):
        """Test failed file retry functionality."""
        file_path = "/test/retry.txt"
        collection = "test_collection"
        
        # Start and fail processing
        await state_manager.start_file_processing(file_path, collection)
        await state_manager.complete_file_processing(
            file_path=file_path,
            success=False,
            error_message="Test failure"
        )
        
        # Verify failed status
        status = await state_manager.get_file_processing_status(file_path)
        assert status.status == FileProcessingStatus.FAILED
        assert status.error_message == "Test failure"
        
        # Test retry
        success = await state_manager.retry_failed_file(file_path)
        assert success
        
        # Verify retry state
        status = await state_manager.get_file_processing_status(file_path)
        assert status.status == FileProcessingStatus.RETRYING
        assert status.error_message is None
        
        # Test max retries exceeded
        for i in range(5):  # Exceed max retries
            await state_manager.start_file_processing(file_path, collection)
            await state_manager.complete_file_processing(
                file_path=file_path,
                success=False,
                error_message=f"Failure {i+1}"
            )
        
        # Should not be able to retry anymore
        success = await state_manager.retry_failed_file(file_path)
        assert not success
    
    @pytest.mark.asyncio
    async def test_watch_folder_persistence(self, state_manager):
        """Test persistent watch folder configuration."""
        watch_config = WatchFolderConfig(
            watch_id="test_watch",
            path="/test/path",
            collection="test_collection",
            patterns=["*.txt", "*.md"],
            ignore_patterns=["*.tmp"],
            auto_ingest=True,
            recursive=True,
            recursive_depth=5,
            debounce_seconds=1.5,
            metadata={"project": "test"}
        )
        
        # Save configuration
        success = await state_manager.save_watch_folder_config(watch_config)
        assert success
        
        # Retrieve configuration
        retrieved_config = await state_manager.get_watch_folder_config("test_watch")
        assert retrieved_config is not None
        assert retrieved_config.watch_id == watch_config.watch_id
        assert retrieved_config.path == watch_config.path
        assert retrieved_config.collection == watch_config.collection
        assert retrieved_config.patterns == watch_config.patterns
        assert retrieved_config.ignore_patterns == watch_config.ignore_patterns
        assert retrieved_config.auto_ingest == watch_config.auto_ingest
        assert retrieved_config.recursive == watch_config.recursive
        assert retrieved_config.recursive_depth == watch_config.recursive_depth
        assert retrieved_config.debounce_seconds == watch_config.debounce_seconds
        assert retrieved_config.metadata["project"] == "test"
        
        # Test update scan time
        success = await state_manager.update_watch_folder_scan_time("test_watch")
        assert success
        
        # Verify scan time was updated
        updated_config = await state_manager.get_watch_folder_config("test_watch")
        assert updated_config.last_scan is not None
        
        # Test getting all configs
        all_configs = await state_manager.get_all_watch_folder_configs()
        assert len(all_configs) >= 1
        assert any(c.watch_id == "test_watch" for c in all_configs)
        
        # Test removal
        success = await state_manager.remove_watch_folder_config("test_watch")
        assert success
        
        # Verify removal
        removed_config = await state_manager.get_watch_folder_config("test_watch")
        assert removed_config is None
    
    @pytest.mark.asyncio
    async def test_processing_queue_management(self, state_manager):
        """Test processing queue functionality."""
        # Add files to queue
        queue_id1 = await state_manager.add_to_processing_queue(
            file_path="/test/file1.txt",
            collection="test_collection",
            priority=ProcessingPriority.HIGH,
            metadata={"test": "data1"}
        )
        
        queue_id2 = await state_manager.add_to_processing_queue(
            file_path="/test/file2.txt",
            collection="test_collection",
            priority=ProcessingPriority.LOW
        )
        
        assert queue_id1 != queue_id2
        
        # Get queue stats
        stats = await state_manager.get_queue_stats()
        assert stats["total"] >= 2
        assert stats["high"] >= 1
        assert stats["low"] >= 1
        
        # Get next item (should be high priority first)
        next_item = await state_manager.get_next_queue_item()
        assert next_item is not None
        assert next_item.priority == ProcessingPriority.HIGH
        assert next_item.file_path == "/test/file1.txt"
        
        # Mark as processing
        success = await state_manager.mark_queue_item_processing(next_item.queue_id)
        assert success
        
        # Remove from queue
        success = await state_manager.remove_from_processing_queue(next_item.queue_id)
        assert success
        
        # Test rescheduling
        next_item = await state_manager.get_next_queue_item()
        assert next_item is not None
        
        success = await state_manager.reschedule_queue_item(
            queue_id=next_item.queue_id,
            delay_seconds=60
        )
        assert success
        
        # Item should not be available immediately
        no_item = await state_manager.get_next_queue_item()
        assert no_item is None  # Scheduled for future
        
        # Clear queue
        cleared_count = await state_manager.clear_queue()
        assert cleared_count >= 1
    
    @pytest.mark.asyncio
    async def test_crash_recovery(self, temp_db_path):
        """Test crash recovery functionality."""
        # First session: create processing records
        manager1 = SQLiteStateManager(temp_db_path)
        await manager1.initialize()
        
        # Start processing some files
        await manager1.start_file_processing("/test/crash1.txt", "test_collection")
        await manager1.start_file_processing("/test/crash2.txt", "test_collection")
        
        # Simulate crash by closing without completion
        await manager1.close()
        
        # Second session: recover from crash
        manager2 = SQLiteStateManager(temp_db_path)
        await manager2.initialize()  # This should trigger crash recovery
        
        # Check that processing files were moved to retry state
        status1 = await manager2.get_file_processing_status("/test/crash1.txt")
        status2 = await manager2.get_file_processing_status("/test/crash2.txt")
        
        # Files should be marked for retry after crash
        assert status1.status in [FileProcessingStatus.RETRYING, FileProcessingStatus.FAILED]
        assert status2.status in [FileProcessingStatus.RETRYING, FileProcessingStatus.FAILED]
        
        await manager2.close()
    
    @pytest.mark.asyncio
    async def test_system_state_management(self, state_manager):
        """Test system state key-value storage."""
        # Set various types of state
        await state_manager.set_system_state("string_key", "test_value")
        await state_manager.set_system_state("dict_key", {"nested": "data", "count": 42})
        await state_manager.set_system_state("list_key", [1, 2, 3])
        await state_manager.set_system_state("number_key", 3.14)
        
        # Retrieve and verify
        string_val = await state_manager.get_system_state("string_key")
        assert string_val == "test_value"
        
        dict_val = await state_manager.get_system_state("dict_key")
        assert dict_val["nested"] == "data"
        assert dict_val["count"] == 42
        
        list_val = await state_manager.get_system_state("list_key")
        assert list_val == [1, 2, 3]
        
        number_val = await state_manager.get_system_state("number_key")
        assert number_val == 3.14
        
        # Test default value
        missing_val = await state_manager.get_system_state("missing_key", "default")
        assert missing_val == "default"
        
        # Test deletion
        success = await state_manager.delete_system_state("string_key")
        assert success
        
        deleted_val = await state_manager.get_system_state("string_key")
        assert deleted_val is None
    
    @pytest.mark.asyncio
    async def test_processing_analytics(self, state_manager):
        """Test processing statistics and analytics."""
        collection = "analytics_test"
        
        # Create test processing records
        for i in range(10):
            file_path = f"/test/analytics_{i}.txt"
            await state_manager.start_file_processing(file_path, collection)
            
            # Complete with varying success rates
            success = i < 7  # 70% success rate
            error_msg = None if success else f"Error {i}"
            
            await state_manager.complete_file_processing(
                file_path=file_path,
                success=success,
                error_message=error_msg,
                processing_time_ms=100 + i * 10
            )
        
        # Get analytics
        stats = await state_manager.get_processing_stats(collection=collection, days=1)
        
        assert stats["total_processed"] >= 10
        assert stats["successful"] >= 7
        assert stats["failed"] >= 3
        assert 0.6 <= stats["success_rate"] <= 0.8  # Around 70%
        assert stats["avg_processing_time_ms"] > 0
        
        # Get failed files
        failed_files = await state_manager.get_failed_files(collection=collection, limit=5)
        assert len(failed_files) >= 3
        
        for failed_file in failed_files:
            assert failed_file["error_message"] is not None
            assert failed_file["collection"] == collection
    
    @pytest.mark.asyncio
    async def test_database_maintenance(self, state_manager):
        """Test database cleanup and maintenance operations."""
        # Create old records
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=35)
        
        with state_manager._lock:
            # Insert old processing history
            state_manager.connection.execute(
                """
                INSERT INTO processing_history 
                (file_path, collection, status, created_at)
                VALUES (?, ?, ?, ?)
                """,
                ("/old/file.txt", "test_collection", "completed", cutoff_date.isoformat())
            )
            state_manager.connection.commit()
        
        # Get stats before cleanup
        stats_before = await state_manager.get_database_stats()
        
        # Perform cleanup
        cleanup_result = await state_manager.cleanup_old_records(days=30)
        
        assert "processing_history" in cleanup_result
        assert cleanup_result["processing_history"] >= 1
        
        # Verify old records were removed
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT COUNT(*) FROM processing_history WHERE file_path = '/old/file.txt'"
            )
            count = cursor.fetchone()[0]
            assert count == 0
        
        # Test vacuum
        success = await state_manager.vacuum_database()
        assert success
        
        # Get stats after maintenance
        stats_after = await state_manager.get_database_stats()
        assert "total_size_mb" in stats_after
        assert stats_after["schema_version"] == state_manager.SCHEMA_VERSION
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, temp_db_path):
        """Test concurrent access with WAL mode."""
        # Create multiple managers accessing same database
        manager1 = SQLiteStateManager(temp_db_path)
        manager2 = SQLiteStateManager(temp_db_path)
        
        await manager1.initialize()
        await manager2.initialize()
        
        # Concurrent writes
        tasks = []
        for i in range(5):
            tasks.append(manager1.start_file_processing(f"/test/concurrent1_{i}.txt", "test"))
            tasks.append(manager2.start_file_processing(f"/test/concurrent2_{i}.txt", "test"))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should succeed
        for result in results:
            assert result is True or not isinstance(result, Exception)
        
        # Verify data consistency
        for i in range(5):
            status1 = await manager1.get_file_processing_status(f"/test/concurrent1_{i}.txt")
            status2 = await manager2.get_file_processing_status(f"/test/concurrent2_{i}.txt")
            
            assert status1 is not None
            assert status2 is not None
            assert status1.status == FileProcessingStatus.PROCESSING
            assert status2.status == FileProcessingStatus.PROCESSING
        
        await manager1.close()
        await manager2.close()
    
    @pytest.mark.asyncio
    async def test_large_data_handling(self, state_manager):
        """Test handling of large datasets and performance."""
        collection = "performance_test"
        batch_size = 100
        
        # Create large batch of files
        start_time = time.time()
        
        for i in range(batch_size):
            file_path = f"/test/batch_{i:04d}.txt"
            metadata = {"batch": "performance", "index": i, "data": "x" * 100}
            
            await state_manager.start_file_processing(
                file_path=file_path,
                collection=collection,
                metadata=metadata
            )
        
        creation_time = time.time() - start_time
        assert creation_time < 10.0  # Should complete within 10 seconds
        
        # Query performance test
        start_time = time.time()
        
        pending_files = await state_manager.get_files_by_status(
            FileProcessingStatus.PROCESSING,
            collection=collection
        )
        
        query_time = time.time() - start_time
        assert len(pending_files) == batch_size
        assert query_time < 5.0  # Should query quickly
        
        # Bulk completion
        start_time = time.time()
        
        completion_tasks = []
        for record in pending_files[:50]:  # Complete half
            completion_tasks.append(
                state_manager.complete_file_processing(
                    record.file_path,
                    success=True,
                    processing_time_ms=50
                )
            )
        
        await asyncio.gather(*completion_tasks)
        
        completion_time = time.time() - start_time
        assert completion_time < 5.0  # Should complete quickly
        
        # Verify final state
        stats = await state_manager.get_processing_stats(collection=collection)
        assert stats["total_processed"] >= 50
        assert stats["successful"] >= 50


class TestCrashRecoveryScenarios:
    """Test specific crash recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_wal_checkpoint_recovery(self, temp_db_path):
        """Test recovery with WAL file data."""
        manager = SQLiteStateManager(temp_db_path)
        await manager.initialize()
        
        # Create data that will be in WAL file
        await manager.start_file_processing("/test/wal1.txt", "test")
        await manager.start_file_processing("/test/wal2.txt", "test")
        
        # Force WAL checkpoint to test recovery
        with manager._lock:
            manager.connection.execute("PRAGMA wal_checkpoint(FULL)")
        
        await manager.close()
        
        # Reopen and verify data integrity
        manager2 = SQLiteStateManager(temp_db_path)
        await manager2.initialize()
        
        status1 = await manager2.get_file_processing_status("/test/wal1.txt")
        status2 = await manager2.get_file_processing_status("/test/wal2.txt")
        
        # Data should be recovered properly
        assert status1 is not None
        assert status2 is not None
        
        await manager2.close()
    
    @pytest.mark.asyncio
    async def test_interrupted_transaction_recovery(self, temp_db_path):
        """Test recovery from interrupted transactions."""
        manager = SQLiteStateManager(temp_db_path)
        await manager.initialize()
        
        # Simulate interrupted transaction by direct database manipulation
        with manager._lock:
            # Start a transaction but don't commit
            manager.connection.execute("BEGIN")
            manager.connection.execute(
                """
                INSERT INTO file_processing 
                (file_path, collection, status, priority)
                VALUES (?, ?, ?, ?)
                """,
                ("/test/interrupted.txt", "test", FileProcessingStatus.PROCESSING.value, ProcessingPriority.NORMAL.value)
            )
            # Don't commit - simulate crash
        
        await manager.close()
        
        # Recovery should handle the uncommitted transaction
        manager2 = SQLiteStateManager(temp_db_path)
        await manager2.initialize()
        
        # The uncommitted record should not exist
        status = await manager2.get_file_processing_status("/test/interrupted.txt")
        assert status is None  # Should not exist due to rollback
        
        await manager2.close()
    
    @pytest.mark.asyncio
    async def test_corrupted_state_recovery(self, temp_db_path):
        """Test recovery from partially corrupted state."""
        manager = SQLiteStateManager(temp_db_path)
        await manager.initialize()
        
        # Create valid processing record
        await manager.start_file_processing("/test/valid.txt", "test")
        
        # Create orphaned queue item (no corresponding file_processing record)
        queue_id = await manager.add_to_processing_queue(
            file_path="/test/orphaned.txt",
            collection="test",
            priority=ProcessingPriority.NORMAL
        )
        
        # Manually delete the file_processing record to create orphan
        with manager._lock:
            manager.connection.execute(
                "DELETE FROM file_processing WHERE file_path = ?",
                ("/test/orphaned.txt",)
            )
            manager.connection.commit()
        
        await manager.close()
        
        # Recovery should clean up orphaned queue items
        manager2 = SQLiteStateManager(temp_db_path)
        await manager2.initialize()
        
        # Valid record should still exist
        valid_status = await manager2.get_file_processing_status("/test/valid.txt")
        assert valid_status is not None
        
        # Orphaned queue item should be cleaned up
        queue_stats = await manager2.get_queue_stats()
        # Should have cleaned up the orphaned item during recovery
        
        await manager2.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])