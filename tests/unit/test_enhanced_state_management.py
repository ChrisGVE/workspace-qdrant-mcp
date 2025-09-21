"""
Tests for Enhanced SQLite State Management.

This module tests the enhanced state management features including:
- Transaction safety and rollback capabilities
- Atomic multi-table operations  
- Database backup and restore functionality
- Performance monitoring and metrics
- Migration tools and validation
- Concurrent access patterns
- State recovery scenarios
"""

import asyncio
import json
import pytest
import sqlite3
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from common.core.sqlite_state_manager import SQLiteStateManager, FileProcessingStatus, ProcessingPriority
from workspace_qdrant_mcp.core.state_enhancements import (
    EnhancedStateManager, 
    MigrationStep, 
    BackupMetadata,
    PerformanceMetric,
    create_enhanced_state_manager
)


@pytest.fixture
async def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_enhanced.db"
    yield str(db_path)
    # Cleanup handled by tempfile


@pytest.fixture
async def base_state_manager(temp_db):
    """Create a base state manager for testing."""
    manager = SQLiteStateManager(temp_db)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def enhanced_manager(base_state_manager):
    """Create an enhanced state manager for testing."""
    manager = EnhancedStateManager(base_state_manager)
    yield manager
    

class TestEnhancedTransactions:
    """Test enhanced transaction functionality."""
    
    async def test_enhanced_transaction_success(self, enhanced_manager):
        """Test successful enhanced transaction with audit logging."""
        async with enhanced_manager.enhanced_transaction("test_operation") as (conn, txn_id):
            # Insert test data
            conn.execute(
                "INSERT INTO file_processing (file_path, collection, status) VALUES (?, ?, ?)",
                ("/test/file.txt", "test_collection", "processing")
            )
            
            # Verify transaction ID is generated
            assert txn_id.startswith("txn_")
            
            # Check audit log entry was created
            cursor = conn.execute(
                "SELECT COUNT(*) FROM transaction_audit WHERE transaction_id = ? AND operation_type = 'BEGIN'",
                (txn_id,)
            )
            assert cursor.fetchone()[0] > 0
    
    async def test_enhanced_transaction_rollback(self, enhanced_manager):
        """Test transaction rollback with audit logging."""
        txn_id = None
        
        try:
            async with enhanced_manager.enhanced_transaction("test_rollback") as (conn, transaction_id):
                txn_id = transaction_id
                
                # Insert test data
                conn.execute(
                    "INSERT INTO file_processing (file_path, collection, status) VALUES (?, ?, ?)",
                    ("/test/rollback.txt", "test_collection", "processing")
                )
                
                # Force an error to trigger rollback
                raise ValueError("Intentional test error")
                
        except ValueError:
            pass  # Expected error
        
        # Verify rollback was logged
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM transaction_audit WHERE transaction_id = ? AND operation_type = 'ROLLBACK'",
                (txn_id,)
            )
            assert cursor.fetchone()[0] > 0
            
            # Verify data was not committed
            cursor = conn.execute(
                "SELECT COUNT(*) FROM file_processing WHERE file_path = ?",
                ("/test/rollback.txt",)
            )
            assert cursor.fetchone()[0] == 0
    
    async def test_atomic_bulk_operation_success(self, enhanced_manager):
        """Test successful atomic bulk operations."""
        operations = [
            {
                'type': 'insert',
                'table': 'file_processing',
                'data': {
                    'file_path': '/test/bulk1.txt',
                    'collection': 'test_collection',
                    'status': 'pending'
                }
            },
            {
                'type': 'insert',
                'table': 'file_processing', 
                'data': {
                    'file_path': '/test/bulk2.txt',
                    'collection': 'test_collection',
                    'status': 'pending'
                }
            },
            {
                'type': 'update',
                'table': 'file_processing',
                'data': {'status': 'processing'},
                'conditions': {'file_path': '/test/bulk1.txt'}
            }
        ]
        
        result = await enhanced_manager.atomic_bulk_operation(operations, "test_bulk")
        assert result is True
        
        # Verify all operations were applied
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute(
                "SELECT status FROM file_processing WHERE file_path = ?",
                ("/test/bulk1.txt",)
            )
            assert cursor.fetchone()[0] == "processing"
            
            cursor = conn.execute(
                "SELECT status FROM file_processing WHERE file_path = ?", 
                ("/test/bulk2.txt",)
            )
            assert cursor.fetchone()[0] == "pending"
    
    async def test_atomic_bulk_operation_rollback(self, enhanced_manager):
        """Test atomic bulk operation rollback on error."""
        operations = [
            {
                'type': 'insert',
                'table': 'file_processing',
                'data': {
                    'file_path': '/test/bulk_fail1.txt',
                    'collection': 'test_collection', 
                    'status': 'pending'
                }
            },
            {
                'type': 'insert',
                'table': 'nonexistent_table',  # This will cause an error
                'data': {'invalid': 'data'}
            }
        ]
        
        result = await enhanced_manager.atomic_bulk_operation(operations, "test_bulk_fail")
        assert result is False
        
        # Verify no partial data was committed
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM file_processing WHERE file_path = ?",
                ("/test/bulk_fail1.txt",)
            )
            assert cursor.fetchone()[0] == 0


class TestBackupAndRestore:
    """Test backup and restore functionality."""
    
    async def test_create_backup(self, enhanced_manager):
        """Test creating a database backup."""
        # Add some test data
        async with enhanced_manager.base_manager.transaction() as conn:
            conn.execute(
                "INSERT INTO file_processing (file_path, collection, status) VALUES (?, ?, ?)",
                ("/test/backup_test.txt", "test_collection", "completed")
            )
        
        backup_id = await enhanced_manager.create_backup("Test backup")
        assert backup_id is not None
        assert backup_id.startswith("backup_")
        
        # Verify backup file exists
        backup_list = await enhanced_manager.get_backup_list()
        assert len(backup_list) > 0
        assert any(b['backup_id'] == backup_id for b in backup_list)
        
        # Verify backup metadata
        backup_info = next(b for b in backup_list if b['backup_id'] == backup_id)
        assert backup_info['exists'] is True
        assert backup_info['file_size_mb'] > 0
        assert backup_info['description'] == "Test backup"
    
    async def test_restore_backup(self, enhanced_manager):
        """Test restoring from a backup."""
        # Create initial data
        async with enhanced_manager.base_manager.transaction() as conn:
            conn.execute(
                "INSERT INTO file_processing (file_path, collection, status) VALUES (?, ?, ?)",
                ("/test/original.txt", "test_collection", "completed")
            )
        
        # Create backup
        backup_id = await enhanced_manager.create_backup("Before modification")
        
        # Modify data
        async with enhanced_manager.base_manager.transaction() as conn:
            conn.execute(
                "INSERT INTO file_processing (file_path, collection, status) VALUES (?, ?, ?)",
                ("/test/modified.txt", "test_collection", "completed")
            )
        
        # Verify modified data exists
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM file_processing")
            count_before_restore = cursor.fetchone()[0]
            assert count_before_restore == 2
        
        # Restore backup
        result = await enhanced_manager.restore_backup(backup_id)
        assert result is True
        
        # Verify data was restored (should only have original data)
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM file_processing")
            count_after_restore = cursor.fetchone()[0]
            assert count_after_restore == 1
            
            cursor = conn.execute("SELECT file_path FROM file_processing")
            file_path = cursor.fetchone()[0]
            assert file_path == "/test/original.txt"


class TestMigrations:
    """Test migration functionality."""
    
    async def test_migration_validation(self, enhanced_manager):
        """Test migration validation."""
        # Test validating existing migration
        result = await enhanced_manager.validate_migration(4)
        assert result is True
        
        # Test validating non-existent migration
        result = await enhanced_manager.validate_migration(999)
        assert result is False
    
    async def test_apply_migration_success(self, enhanced_manager):
        """Test successful migration application."""
        # Apply migration to version 4 (performance monitoring)
        result = await enhanced_manager.apply_migration(4, create_backup=False)
        assert result is True
        
        # Verify tables were created
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('performance_logs', 'backup_registry')"
            )
            tables = [row[0] for row in cursor.fetchall()]
            assert 'performance_logs' in tables
            assert 'backup_registry' in tables
            
            # Verify schema version was updated
            cursor = conn.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            )
            version = cursor.fetchone()[0]
            assert version >= 4
    
    async def test_migration_rollback_on_error(self, enhanced_manager):
        """Test migration rollback when errors occur."""
        # Register a migration that will fail
        enhanced_manager.migration_registry[999] = [
            MigrationStep(
                step_id="failing_step",
                description="This step will fail",
                forward_sql=["CREATE TABLE test_table (id INTEGER)"],
                rollback_sql=["DROP TABLE IF EXISTS test_table"]
            ),
            MigrationStep(
                step_id="error_step", 
                description="This step causes an error",
                forward_sql=["INVALID SQL SYNTAX"],  # This will cause an error
                rollback_sql=[]
            )
        ]
        
        result = await enhanced_manager.apply_migration(999, create_backup=False)
        assert result is False
        
        # Verify rollback occurred - test_table should not exist
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test_table'"
            )
            assert cursor.fetchone() is None


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    async def test_performance_metric_recording(self, enhanced_manager):
        """Test recording performance metrics."""
        # Apply migration to ensure performance_logs table exists
        await enhanced_manager.apply_migration(4, create_backup=False)
        
        # Record a performance metric
        await enhanced_manager._record_performance_metric("test_operation", 150.5, 10)
        
        # Verify metric was recorded
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute(
                "SELECT operation, duration_ms, rows_affected FROM performance_logs WHERE operation = ?",
                ("test_operation",)
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "test_operation"
            assert row[1] == 150.5
            assert row[2] == 10
    
    async def test_performance_stats_collection(self, enhanced_manager):
        """Test collecting performance statistics."""
        # Apply migration to ensure performance_logs table exists
        await enhanced_manager.apply_migration(4, create_backup=False)
        
        # Record multiple metrics
        await enhanced_manager._record_performance_metric("operation_a", 100.0, 5)
        await enhanced_manager._record_performance_metric("operation_a", 200.0, 10)
        await enhanced_manager._record_performance_metric("operation_b", 50.0, 2)
        
        # Get performance stats
        stats = await enhanced_manager.get_performance_stats(hours=1)
        
        assert 'overall_stats' in stats
        assert 'by_operation' in stats
        assert stats['overall_stats']['total_operations'] == 3
        assert stats['overall_stats']['total_rows_affected'] == 17
        
        # Check operation-specific stats
        op_a_stats = next((op for op in stats['by_operation'] if op['operation'] == 'operation_a'), None)
        assert op_a_stats is not None
        assert op_a_stats['count'] == 2
        assert op_a_stats['avg_duration_ms'] == 150.0
        assert op_a_stats['total_rows_affected'] == 15
    
    async def test_performance_logs_cleanup(self, enhanced_manager):
        """Test cleaning up old performance logs."""
        # Apply migration to ensure performance_logs table exists
        await enhanced_manager.apply_migration(4, create_backup=False)
        
        # Insert old performance log entry manually
        old_timestamp = datetime.now(timezone.utc) - timedelta(days=35)
        async with enhanced_manager.base_manager.transaction() as conn:
            conn.execute(
                "INSERT INTO performance_logs (operation, duration_ms, rows_affected, timestamp) VALUES (?, ?, ?, ?)",
                ("old_operation", 100.0, 5, old_timestamp)
            )
            
            # Insert recent entry
            conn.execute(
                "INSERT INTO performance_logs (operation, duration_ms, rows_affected) VALUES (?, ?, ?)",
                ("recent_operation", 50.0, 3)
            )
        
        # Clean up logs older than 30 days
        deleted_count = await enhanced_manager.cleanup_performance_logs(days_to_keep=30)
        assert deleted_count == 1
        
        # Verify only recent entry remains
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM performance_logs")
            remaining_count = cursor.fetchone()[0]
            assert remaining_count == 1
            
            cursor = conn.execute("SELECT operation FROM performance_logs")
            operation = cursor.fetchone()[0]
            assert operation == "recent_operation"


class TestDatabaseIntegrity:
    """Test database integrity checking functionality."""
    
    async def test_database_integrity_check(self, enhanced_manager):
        """Test comprehensive database integrity check."""
        # Add test data
        async with enhanced_manager.base_manager.transaction() as conn:
            conn.execute(
                "INSERT INTO file_processing (file_path, collection, status) VALUES (?, ?, ?)",
                ("/test/integrity.txt", "test_collection", "completed")
            )
        
        integrity_results = await enhanced_manager.verify_database_integrity()
        
        assert 'sqlite_integrity' in integrity_results
        assert 'foreign_key_violations' in integrity_results
        assert 'orphaned_records' in integrity_results
        assert 'table_counts' in integrity_results
        assert 'database_info' in integrity_results
        
        # Check SQLite integrity
        assert integrity_results['sqlite_integrity'][0] == 'ok'
        
        # Check table counts
        assert integrity_results['table_counts']['file_processing'] >= 1
        
        # Check database info
        db_info = integrity_results['database_info']
        assert db_info['total_pages'] > 0
        assert db_info['page_size_bytes'] > 0
        assert db_info['total_size_mb'] > 0


class TestConcurrentAccess:
    """Test concurrent access patterns."""
    
    async def test_concurrent_transactions(self, enhanced_manager):
        """Test concurrent transaction handling."""
        
        async def worker_task(worker_id: int):
            """Worker task that performs database operations."""
            for i in range(5):
                async with enhanced_manager.enhanced_transaction(f"worker_{worker_id}_op_{i}") as (conn, txn_id):
                    conn.execute(
                        "INSERT INTO file_processing (file_path, collection, status) VALUES (?, ?, ?)",
                        (f"/test/worker_{worker_id}_file_{i}.txt", "test_collection", "processing")
                    )
                    # Small delay to increase chance of concurrent access
                    await asyncio.sleep(0.001)
        
        # Run multiple workers concurrently
        workers = [worker_task(i) for i in range(3)]
        await asyncio.gather(*workers)
        
        # Verify all operations completed successfully
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM file_processing WHERE file_path LIKE '/test/worker_%'")
            total_count = cursor.fetchone()[0]
            assert total_count == 15  # 3 workers * 5 operations each
    
    async def test_concurrent_backup_operations(self, enhanced_manager):
        """Test concurrent backup operations."""
        
        async def create_backup_task(backup_num: int):
            """Task to create a backup."""
            return await enhanced_manager.create_backup(f"Concurrent backup {backup_num}")
        
        # Create multiple backups concurrently
        backup_tasks = [create_backup_task(i) for i in range(3)]
        backup_ids = await asyncio.gather(*backup_tasks)
        
        # Verify all backups were created
        assert len(backup_ids) == 3
        assert all(backup_id is not None for backup_id in backup_ids)
        assert len(set(backup_ids)) == 3  # All backup IDs should be unique
        
        # Verify backups are listed
        backup_list = await enhanced_manager.get_backup_list()
        created_backup_ids = {b['backup_id'] for b in backup_list}
        for backup_id in backup_ids:
            assert backup_id in created_backup_ids


class TestStateRecovery:
    """Test state recovery scenarios."""
    
    async def test_crash_recovery_simulation(self, enhanced_manager):
        """Test state recovery after simulated crash."""
        # Start some file processing
        await enhanced_manager.base_manager.start_file_processing(
            "/test/crash_test.txt", "test_collection", priority=ProcessingPriority.HIGH
        )
        
        # Simulate crash by closing and reopening database
        await enhanced_manager.base_manager.close()
        
        # Reinitialize (this should trigger crash recovery)
        new_manager = SQLiteStateManager(enhanced_manager.base_manager.db_path)
        await new_manager.initialize()
        
        # Check that file processing record was recovered and marked appropriately
        async with new_manager.transaction() as conn:
            cursor = conn.execute(
                "SELECT status FROM file_processing WHERE file_path = ?",
                ("/test/crash_test.txt",)
            )
            row = cursor.fetchone()
            assert row is not None
            # File should be marked for retry after crash recovery
            assert row[0] in [FileProcessingStatus.RETRYING.value, FileProcessingStatus.FAILED.value]
        
        await new_manager.close()
    
    async def test_transaction_recovery(self, enhanced_manager):
        """Test recovery from incomplete transactions."""
        # Apply migration to ensure audit table exists
        await enhanced_manager.apply_migration(5, create_backup=False)
        
        # Start transaction but don't complete it (simulate interruption)
        async with enhanced_manager.base_manager.transaction() as conn:
            # Insert audit record for incomplete transaction
            incomplete_txn_id = "txn_incomplete_12345"
            conn.execute(
                """
                INSERT INTO transaction_audit 
                (transaction_id, operation_type, table_name, timestamp)
                VALUES (?, 'BEGIN', 'file_processing', CURRENT_TIMESTAMP)
                """,
                (incomplete_txn_id,)
            )
        
        # Check for incomplete transactions
        async with enhanced_manager.base_manager.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT transaction_id 
                FROM transaction_audit 
                WHERE transaction_id NOT IN (
                    SELECT transaction_id 
                    FROM transaction_audit 
                    WHERE operation_type IN ('COMMIT', 'ROLLBACK')
                )
                """
            )
            incomplete_transactions = [row[0] for row in cursor.fetchall()]
            
            # Should find our incomplete transaction
            assert incomplete_txn_id in incomplete_transactions


@pytest.mark.asyncio
async def test_create_enhanced_state_manager():
    """Test creation and initialization of enhanced state manager."""
    temp_dir = tempfile.mkdtemp()
    db_path = str(Path(temp_dir) / "test_enhanced_creation.db")
    
    manager = await create_enhanced_state_manager(db_path)
    
    assert isinstance(manager, EnhancedStateManager)
    assert isinstance(manager.base_manager, SQLiteStateManager)
    
    # Verify enhanced tables were created
    async with manager.base_manager.transaction() as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('performance_logs', 'backup_registry', 'transaction_audit')"
        )
        tables = [row[0] for row in cursor.fetchall()]
        
        # Should have at least performance_logs and backup_registry
        assert 'performance_logs' in tables
        assert 'backup_registry' in tables
    
    await manager.base_manager.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])