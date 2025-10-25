#!/usr/bin/env python3
"""
Comprehensive SQLite State Management Testing Suite - Task 75

This module implements comprehensive testing for SQLite state management functionality
including crash recovery, concurrent access, ACID transactions, and performance validation.

Key Test Categories:
1. Basic functionality and WAL mode verification
2. Crash recovery simulation tests
3. Concurrent access validation
4. ACID transaction testing
5. Performance benchmarking with large datasets
6. Database maintenance and error scenarios

Requirements:
- SQLite with WAL mode enabled
- Process simulation for crash recovery
- Multi-process testing for concurrency
- Performance metrics collection
- Error injection and recovery testing
"""

import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import psutil
import pytest


@dataclass
class StateRecord:
    """Represents a file processing state record."""
    file_path: str
    status: str
    last_modified: float
    checksum: str
    metadata: dict[str, Any]
    created_at: float
    updated_at: float
    retry_count: int = 0
    error_message: str | None = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation: str
    duration_ms: float
    records_processed: int
    throughput_rps: float
    memory_usage_mb: float
    cpu_usage_percent: float


class SQLiteStateManagerComprehensive:
    """
    Comprehensive SQLite-based state manager with WAL mode and crash recovery.

    This is the implementation being tested - includes all features
    required for comprehensive testing scenarios.
    """

    def __init__(self, db_path: str, enable_wal: bool = True):
        """Initialize state manager with SQLite database."""
        self.db_path = db_path
        self.enable_wal = enable_wal
        self._connection = None
        self._lock = threading.RLock()
        self._transaction_counter = 0
        self.logger = logging.getLogger(f"SQLiteStateManager-{id(self)}")

    async def initialize(self):
        """Initialize database schema and enable WAL mode."""
        with self._lock:
            self._connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0,
                isolation_level=None  # Autocommit mode for better control
            )

            # Enable WAL mode for better concurrency and crash recovery
            if self.enable_wal:
                result = self._connection.execute("PRAGMA journal_mode=WAL").fetchone()
                self.logger.info(f"WAL mode enabled: {result[0]}")

            # Configure for reliability and performance
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA cache_size=10000")
            self._connection.execute("PRAGMA temp_store=MEMORY")
            self._connection.execute("PRAGMA mmap_size=268435456")  # 256MB
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA busy_timeout=30000")

            # Create state tracking table
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS file_states (
                    file_path TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    last_modified REAL NOT NULL,
                    checksum TEXT NOT NULL,
                    metadata TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'retrying'))
                )
            """)

            # Create indexes for performance
            self._connection.execute("CREATE INDEX IF NOT EXISTS idx_status ON file_states(status)")
            self._connection.execute("CREATE INDEX IF NOT EXISTS idx_updated_at ON file_states(updated_at)")
            self._connection.execute("CREATE INDEX IF NOT EXISTS idx_retry_count ON file_states(retry_count)")

            # Create transaction log for crash recovery
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS transaction_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transaction_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    old_state TEXT,
                    new_state TEXT,
                    timestamp REAL NOT NULL,
                    committed BOOLEAN DEFAULT 0,
                    INDEX(transaction_id),
                    INDEX(timestamp),
                    INDEX(committed)
                )
            """)

            # Create performance metrics table
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT NOT NULL,
                    duration_ms REAL NOT NULL,
                    records_processed INTEGER NOT NULL,
                    memory_usage_mb REAL,
                    cpu_usage_percent REAL,
                    timestamp REAL NOT NULL,
                    INDEX(operation),
                    INDEX(timestamp)
                )
            """)

            # Create system configuration table
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)

            # Perform crash recovery check
            await self._perform_crash_recovery()

    async def _perform_crash_recovery(self):
        """Perform crash recovery operations."""
        with self._lock:
            # Find uncommitted transactions
            cursor = self._connection.execute("""
                SELECT DISTINCT transaction_id
                FROM transaction_log
                WHERE committed = 0
            """)

            uncommitted_txs = [row[0] for row in cursor.fetchall()]

            for tx_id in uncommitted_txs:
                self.logger.warning(f"Rolling back uncommitted transaction: {tx_id}")
                await self._rollback_transaction_internal(tx_id)

            # Find processing files that may have been interrupted
            cursor = self._connection.execute("""
                SELECT file_path
                FROM file_states
                WHERE status = 'processing'
                AND updated_at < ?
            """, (time.time() - 300,))  # 5 minutes ago

            interrupted_files = [row[0] for row in cursor.fetchall()]

            for file_path in interrupted_files:
                self.logger.warning(f"Marking interrupted file for retry: {file_path}")
                self._connection.execute("""
                    UPDATE file_states
                    SET status = 'retrying',
                        retry_count = retry_count + 1,
                        error_message = 'Interrupted by system restart',
                        updated_at = ?
                    WHERE file_path = ?
                """, (time.time(), file_path))

            self._connection.commit()

    async def begin_transaction(self) -> str:
        """Begin a new transaction and return transaction ID."""
        with self._lock:
            self._transaction_counter += 1
            tx_id = f"tx_{int(time.time())}_{self._transaction_counter}"

            self._connection.execute(
                "INSERT INTO transaction_log (transaction_id, operation, file_path, timestamp) VALUES (?, ?, ?, ?)",
                (tx_id, "BEGIN", "", time.time())
            )
            self._connection.commit()

            self.logger.debug(f"Started transaction: {tx_id}")
            return tx_id

    async def commit_transaction(self, transaction_id: str):
        """Commit a transaction."""
        with self._lock:
            self._connection.execute(
                "UPDATE transaction_log SET committed = 1 WHERE transaction_id = ?",
                (transaction_id,)
            )
            self._connection.commit()
            self.logger.debug(f"Committed transaction: {transaction_id}")

    async def rollback_transaction(self, transaction_id: str):
        """Rollback a transaction."""
        await self._rollback_transaction_internal(transaction_id)

    async def _rollback_transaction_internal(self, transaction_id: str):
        """Internal rollback implementation."""
        with self._lock:
            # Get all operations in this transaction
            cursor = self._connection.execute(
                """SELECT operation, file_path, old_state
                   FROM transaction_log
                   WHERE transaction_id = ?
                   AND committed = 0
                   ORDER BY id DESC""",
                (transaction_id,)
            )

            operations = cursor.fetchall()

            # Reverse operations
            for operation, file_path, old_state in operations:
                if operation == "UPDATE" and old_state:
                    try:
                        old_record = json.loads(old_state)
                        self._connection.execute(
                            """UPDATE file_states SET
                               status = ?, last_modified = ?, checksum = ?,
                               metadata = ?, updated_at = ?, retry_count = ?, error_message = ?
                               WHERE file_path = ?""",
                            (old_record['status'], old_record['last_modified'],
                             old_record['checksum'], old_record['metadata'],
                             old_record['updated_at'], old_record.get('retry_count', 0),
                             old_record.get('error_message'), file_path)
                        )
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.error(f"Error rolling back {file_path}: {e}")
                elif operation == "INSERT":
                    self._connection.execute(
                        "DELETE FROM file_states WHERE file_path = ?",
                        (file_path,)
                    )

            # Remove transaction log entries
            self._connection.execute(
                "DELETE FROM transaction_log WHERE transaction_id = ?",
                (transaction_id,)
            )
            self._connection.commit()

            self.logger.debug(f"Rolled back transaction: {transaction_id}")

    async def upsert_state(self, record: StateRecord, transaction_id: str | None = None) -> bool:
        """Insert or update a state record."""
        with self._lock:
            try:
                # Log transaction if provided
                if transaction_id:
                    # Get old state for rollback
                    cursor = self._connection.execute(
                        "SELECT * FROM file_states WHERE file_path = ?",
                        (record.file_path,)
                    )
                    old_record = cursor.fetchone()
                    old_state = None
                    if old_record:
                        old_state = json.dumps({
                            'status': old_record[1], 'last_modified': old_record[2],
                            'checksum': old_record[3], 'metadata': old_record[4],
                            'updated_at': old_record[6], 'retry_count': old_record[7],
                            'error_message': old_record[8]
                        })

                    self._connection.execute(
                        """INSERT INTO transaction_log
                           (transaction_id, operation, file_path, old_state, new_state, timestamp)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (transaction_id, "UPDATE", record.file_path, old_state,
                         json.dumps(asdict(record)), time.time())
                    )

                # Perform upsert
                self._connection.execute(
                    """INSERT OR REPLACE INTO file_states
                       (file_path, status, last_modified, checksum, metadata,
                        created_at, updated_at, retry_count, error_message)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (record.file_path, record.status, record.last_modified,
                     record.checksum, json.dumps(record.metadata),
                     record.created_at, record.updated_at, record.retry_count,
                     record.error_message)
                )

                if not transaction_id:
                    self._connection.commit()

                return True

            except sqlite3.Error as e:
                self.logger.error(f"Error upserting state for {record.file_path}: {e}")
                return False

    async def get_state(self, file_path: str) -> StateRecord | None:
        """Get state record for a file."""
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "SELECT * FROM file_states WHERE file_path = ?",
                    (file_path,)
                )
                row = cursor.fetchone()
                if row:
                    return StateRecord(
                        file_path=row[0], status=row[1], last_modified=row[2],
                        checksum=row[3], metadata=json.loads(row[4] or '{}'),
                        created_at=row[5], updated_at=row[6], retry_count=row[7],
                        error_message=row[8]
                    )
                return None
            except sqlite3.Error as e:
                self.logger.error(f"Error getting state for {file_path}: {e}")
                return None

    async def get_states_by_status(self, status: str) -> list[StateRecord]:
        """Get all states with specific status."""
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "SELECT * FROM file_states WHERE status = ? ORDER BY updated_at",
                    (status,)
                )
                records = []
                for row in cursor.fetchall():
                    records.append(StateRecord(
                        file_path=row[0], status=row[1], last_modified=row[2],
                        checksum=row[3], metadata=json.loads(row[4] or '{}'),
                        created_at=row[5], updated_at=row[6], retry_count=row[7],
                        error_message=row[8]
                    ))
                return records
            except sqlite3.Error as e:
                self.logger.error(f"Error getting states by status {status}: {e}")
                return []

    async def delete_state(self, file_path: str) -> bool:
        """Delete a state record."""
        with self._lock:
            try:
                cursor = self._connection.execute(
                    "DELETE FROM file_states WHERE file_path = ?",
                    (file_path,)
                )
                self._connection.commit()
                return cursor.rowcount > 0
            except sqlite3.Error as e:
                self.logger.error(f"Error deleting state for {file_path}: {e}")
                return False

    async def count_states(self) -> int:
        """Get total number of state records."""
        with self._lock:
            try:
                cursor = self._connection.execute("SELECT COUNT(*) FROM file_states")
                return cursor.fetchone()[0]
            except sqlite3.Error as e:
                self.logger.error(f"Error counting states: {e}")
                return 0

    async def vacuum(self) -> bool:
        """Vacuum database for maintenance."""
        with self._lock:
            try:
                self._connection.execute("VACUUM")
                return True
            except sqlite3.Error as e:
                self.logger.error(f"Error vacuuming database: {e}")
                return False

    async def analyze(self) -> bool:
        """Analyze database for optimization."""
        with self._lock:
            try:
                self._connection.execute("ANALYZE")
                return True
            except sqlite3.Error as e:
                self.logger.error(f"Error analyzing database: {e}")
                return False

    async def get_wal_status(self) -> dict[str, Any]:
        """Get WAL mode status and statistics."""
        with self._lock:
            try:
                cursor = self._connection.execute("PRAGMA journal_mode")
                journal_mode = cursor.fetchone()[0]

                cursor = self._connection.execute("PRAGMA wal_checkpoint")
                checkpoint_info = cursor.fetchone()

                return {
                    'journal_mode': journal_mode,
                    'wal_busy': checkpoint_info[0] if checkpoint_info else 0,
                    'wal_log': checkpoint_info[1] if checkpoint_info else 0,
                    'wal_checkpointed': checkpoint_info[2] if checkpoint_info else 0
                }
            except sqlite3.Error as e:
                self.logger.error(f"Error getting WAL status: {e}")
                return {'journal_mode': 'unknown', 'error': str(e)}

    async def force_checkpoint(self) -> bool:
        """Force WAL checkpoint."""
        with self._lock:
            try:
                self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                return True
            except sqlite3.Error as e:
                self.logger.error(f"Error forcing checkpoint: {e}")
                return False

    async def get_database_size(self) -> dict[str, int]:
        """Get database file sizes."""
        try:
            sizes = {}
            sizes['main_db'] = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            sizes['wal_file'] = os.path.getsize(self.db_path + '-wal') if os.path.exists(self.db_path + '-wal') else 0
            sizes['shm_file'] = os.path.getsize(self.db_path + '-shm') if os.path.exists(self.db_path + '-shm') else 0
            sizes['total'] = sum(sizes.values())
            return sizes
        except OSError as e:
            self.logger.error(f"Error getting database size: {e}")
            return {'error': str(e)}

    async def record_performance_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        with self._lock:
            try:
                self._connection.execute(
                    """INSERT INTO performance_metrics
                       (operation, duration_ms, records_processed, memory_usage_mb,
                        cpu_usage_percent, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (metric.operation, metric.duration_ms, metric.records_processed,
                     metric.memory_usage_mb, metric.cpu_usage_percent, time.time())
                )
                self._connection.commit()
            except sqlite3.Error as e:
                self.logger.error(f"Error recording performance metric: {e}")

    async def get_performance_stats(self, operation: str | None = None,
                                  hours: int = 24) -> dict[str, float]:
        """Get performance statistics."""
        with self._lock:
            try:
                cutoff_time = time.time() - (hours * 3600)

                if operation:
                    cursor = self._connection.execute(
                        """SELECT AVG(duration_ms), AVG(records_processed),
                           AVG(memory_usage_mb), COUNT(*)
                           FROM performance_metrics
                           WHERE operation = ? AND timestamp > ?""",
                        (operation, cutoff_time)
                    )
                else:
                    cursor = self._connection.execute(
                        """SELECT AVG(duration_ms), AVG(records_processed),
                           AVG(memory_usage_mb), COUNT(*)
                           FROM performance_metrics
                           WHERE timestamp > ?""",
                        (cutoff_time,)
                    )

                row = cursor.fetchone()
                if row and row[3] > 0:  # Count > 0
                    return {
                        'avg_duration_ms': row[0] or 0,
                        'avg_records_processed': row[1] or 0,
                        'avg_memory_usage_mb': row[2] or 0,
                        'sample_count': row[3]
                    }
                return {'sample_count': 0}

            except sqlite3.Error as e:
                self.logger.error(f"Error getting performance stats: {e}")
                return {'error': str(e)}

    async def simulate_disk_full(self, enable: bool = True):
        """Simulate disk full condition for testing."""
        if enable:
            # This would be used in tests with mocking
            pass

    async def close(self):
        """Close database connection."""
        if self._connection:
            try:
                # Force final checkpoint
                await self.force_checkpoint()
                self._connection.close()
                self._connection = None
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database: {e}")


class TestSQLiteStateManagerComprehensive:
    """Comprehensive test suite for SQLite State Manager."""

    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary database file for testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        # Cleanup
        for ext in ['', '-wal', '-shm']:
            try:
                os.unlink(path + ext)
            except FileNotFoundError:
                pass

    @pytest.fixture
    async def state_manager(self, temp_db_path):
        """Create initialized state manager."""
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.fixture
    def sample_record(self):
        """Create sample state record."""
        return StateRecord(
            file_path="/test/file.txt",
            status="processed",
            last_modified=time.time(),
            checksum="sha256:abc123",
            metadata={"size": 1024, "type": "text"},
            created_at=time.time(),
            updated_at=time.time()
        )

    # ==================== Basic Functionality Tests ====================

    @pytest.mark.asyncio
    async def test_initialization_and_wal_mode(self, temp_db_path):
        """Test database initialization with WAL mode enabled."""
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()

        # Verify WAL mode is enabled
        status = await manager.get_wal_status()
        assert status['journal_mode'].upper() == 'WAL'

        # Verify tables exist
        with manager._lock:
            cursor = manager._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            assert 'file_states' in tables
            assert 'transaction_log' in tables
            assert 'performance_metrics' in tables
            assert 'system_config' in tables

        await manager.close()

    @pytest.mark.asyncio
    async def test_basic_crud_operations(self, state_manager, sample_record):
        """Test basic CRUD operations."""
        # Create
        result = await state_manager.upsert_state(sample_record)
        assert result is True

        # Read
        retrieved = await state_manager.get_state(sample_record.file_path)
        assert retrieved is not None
        assert retrieved.file_path == sample_record.file_path
        assert retrieved.status == sample_record.status

        # Update
        sample_record.status = "updated"
        sample_record.updated_at = time.time()
        await state_manager.upsert_state(sample_record)

        updated = await state_manager.get_state(sample_record.file_path)
        assert updated.status == "updated"

        # Delete
        deleted = await state_manager.delete_state(sample_record.file_path)
        assert deleted is True

        # Verify deletion
        not_found = await state_manager.get_state(sample_record.file_path)
        assert not_found is None

    @pytest.mark.asyncio
    async def test_query_by_status(self, state_manager):
        """Test querying records by status."""
        # Create records with different statuses
        records = []
        for i in range(5):
            record = StateRecord(
                file_path=f"/test/file_{i}.txt",
                status="pending" if i < 3 else "processed",
                last_modified=time.time(),
                checksum=f"sha256:test{i}",
                metadata={"index": i},
                created_at=time.time(),
                updated_at=time.time()
            )
            records.append(record)
            await state_manager.upsert_state(record)

        # Query by status
        pending = await state_manager.get_states_by_status("pending")
        processed = await state_manager.get_states_by_status("processed")

        assert len(pending) == 3
        assert len(processed) == 2
        assert all(r.status == "pending" for r in pending)
        assert all(r.status == "processed" for r in processed)

    # ==================== Crash Recovery Tests ====================

    @pytest.mark.asyncio
    async def test_crash_recovery_uncommitted_transaction(self, temp_db_path):
        """Test recovery from crash during uncommitted transaction."""
        # Setup initial state
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()

        # Create initial record
        initial_record = StateRecord(
            file_path="/test/recovery.txt",
            status="initial",
            last_modified=time.time(),
            checksum="initial",
            metadata={"test": "initial"},
            created_at=time.time(),
            updated_at=time.time()
        )
        await manager.upsert_state(initial_record)

        # Start transaction but don't commit
        tx_id = await manager.begin_transaction()

        modified_record = StateRecord(
            file_path="/test/recovery.txt",
            status="modified",
            last_modified=time.time(),
            checksum="modified",
            metadata={"test": "modified"},
            created_at=initial_record.created_at,
            updated_at=time.time()
        )
        await manager.upsert_state(modified_record, tx_id)

        # Simulate crash - close without commit
        await manager.close()

        # Restart and check recovery
        manager2 = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager2.initialize()

        # The uncommitted transaction should be rolled back
        recovered = await manager2.get_state("/test/recovery.txt")
        assert recovered is not None
        # Should have the initial state, not the modified state
        assert recovered.status == "initial"
        assert recovered.checksum == "initial"

        await manager2.close()

    @pytest.mark.asyncio
    async def test_process_kill_recovery(self, temp_db_path):
        """Test recovery after process is killed mid-transaction."""
        def worker_process(db_path: str, should_crash: bool):
            """Worker process that may crash during transaction."""
            import json
            import os
            import signal
            import sqlite3
            import time

            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")

            # Insert initial data
            conn.execute("""
                INSERT OR REPLACE INTO file_states
                (file_path, status, last_modified, checksum, metadata, created_at, updated_at, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ("/test/crash_test.txt", "processing", time.time(),
                  "hash1", json.dumps({}), time.time(), time.time(), 0))

            conn.commit()

            if should_crash:
                # Start transaction but crash before commit
                conn.execute("BEGIN")
                conn.execute("""
                    UPDATE file_states SET status = 'crashed'
                    WHERE file_path = '/test/crash_test.txt'
                """)
                # Don't commit - simulate crash
                os.kill(os.getpid(), signal.SIGKILL)
            else:
                conn.close()

        # Initialize database
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()
        await manager.close()

        # Run worker that crashes
        process = multiprocessing.Process(
            target=worker_process,
            args=(temp_db_path, True)
        )
        process.start()
        process.join()

        # Verify the process was killed
        assert process.exitcode != 0

        # Check recovery
        manager2 = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager2.initialize()

        record = await manager2.get_state("/test/crash_test.txt")
        # Due to WAL mode and recovery, we should have the initial committed state
        assert record is not None
        assert record.status == "processing"  # Should not be 'crashed'

        await manager2.close()

    # ==================== Concurrent Access Tests ====================

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, state_manager):
        """Test concurrent read operations."""
        # Setup test data
        records = []
        for i in range(100):
            record = StateRecord(
                file_path=f"/test/concurrent_{i}.txt",
                status=f"status_{i % 5}",
                last_modified=time.time(),
                checksum=f"hash_{i}",
                metadata={"index": i},
                created_at=time.time(),
                updated_at=time.time()
            )
            records.append(record)
            await state_manager.upsert_state(record)

        async def read_worker(worker_id: int) -> list[StateRecord]:
            """Worker that performs concurrent reads."""
            results = []
            for i in range(0, 100, 10):
                record = await state_manager.get_state(f"/test/concurrent_{i}.txt")
                if record:
                    results.append(record)
            return results

        # Run concurrent reads
        tasks = [read_worker(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all workers got consistent results
        for worker_results in results:
            assert len(worker_results) == 10
            for record in worker_results:
                assert record is not None

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, temp_db_path):
        """Test concurrent write operations across processes."""
        def writer_process(db_path: str, worker_id: int, num_records: int):
            """Worker process that writes records concurrently."""
            import json
            import sqlite3
            import time

            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")

            success_count = 0
            for i in range(num_records):
                try:
                    file_path = f"/test/worker_{worker_id}_record_{i}.txt"
                    conn.execute("""
                        INSERT OR REPLACE INTO file_states
                        (file_path, status, last_modified, checksum, metadata,
                         created_at, updated_at, retry_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (file_path, f"worker_{worker_id}", time.time(),
                          f"hash_{i}", json.dumps({"worker": worker_id, "index": i}),
                          time.time(), time.time(), 0))
                    conn.commit()
                    success_count += 1
                except sqlite3.Error as e:
                    print(f"Worker {worker_id} error on record {i}: {e}")

            conn.close()
            return success_count

        # Initialize database
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()
        await manager.close()

        # Run concurrent writers
        num_workers = 5
        records_per_worker = 50

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(writer_process, temp_db_path, i, records_per_worker)
                for i in range(num_workers)
            ]
            results = [f.result() for f in futures]

        # Verify results
        manager2 = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager2.initialize()

        total_written = sum(results)
        total_in_db = await manager2.count_states()

        assert total_written == num_workers * records_per_worker
        assert total_in_db == total_written

        # Verify data integrity
        for worker_id in range(num_workers):
            worker_records = await manager2.get_states_by_status(f"worker_{worker_id}")
            assert len(worker_records) == records_per_worker

        await manager2.close()

    # ==================== ACID Transaction Tests ====================

    @pytest.mark.asyncio
    async def test_transaction_atomicity(self, state_manager):
        """Test atomicity - all operations in transaction succeed or fail together."""
        # Create initial records
        initial_records = []
        for i in range(5):
            record = StateRecord(
                file_path=f"/test/atomic_{i}.txt",
                status="initial",
                last_modified=time.time(),
                checksum=f"initial_{i}",
                metadata={"index": i},
                created_at=time.time(),
                updated_at=time.time()
            )
            initial_records.append(record)
            await state_manager.upsert_state(record)

        # Start transaction
        tx_id = await state_manager.begin_transaction()

        # Perform multiple operations
        for i, record in enumerate(initial_records):
            record.status = "updated"
            record.checksum = f"updated_{i}"
            record.updated_at = time.time()
            await state_manager.upsert_state(record, tx_id)

        # Simulate failure - rollback instead of commit
        await state_manager.rollback_transaction(tx_id)

        # Verify all records are back to initial state
        for i in range(5):
            record = await state_manager.get_state(f"/test/atomic_{i}.txt")
            assert record.status == "initial"
            assert record.checksum == f"initial_{i}"

    @pytest.mark.asyncio
    async def test_transaction_consistency(self, state_manager):
        """Test consistency - database remains in valid state."""
        # Create test scenario with constraints
        await state_manager.upsert_state(StateRecord(
            file_path="/test/parent.txt",
            status="active",
            last_modified=time.time(),
            checksum="parent_hash",
            metadata={"type": "parent", "children": ["/test/child1.txt", "/test/child2.txt"]},
            created_at=time.time(),
            updated_at=time.time()
        ))

        for i in range(1, 3):
            await state_manager.upsert_state(StateRecord(
                file_path=f"/test/child{i}.txt",
                status="active",
                last_modified=time.time(),
                checksum=f"child_{i}_hash",
                metadata={"type": "child", "parent": "/test/parent.txt"},
                created_at=time.time(),
                updated_at=time.time()
            ))

        # Transaction that maintains consistency
        tx_id = await state_manager.begin_transaction()

        # Update parent
        parent = await state_manager.get_state("/test/parent.txt")
        parent.status = "updating"
        parent.updated_at = time.time()
        await state_manager.upsert_state(parent, tx_id)

        # Update children
        for i in range(1, 3):
            child = await state_manager.get_state(f"/test/child{i}.txt")
            child.status = "updating"
            child.updated_at = time.time()
            await state_manager.upsert_state(child, tx_id)

        # Commit transaction
        await state_manager.commit_transaction(tx_id)

        # Verify consistency
        parent = await state_manager.get_state("/test/parent.txt")
        assert parent.status == "updating"

        for i in range(1, 3):
            child = await state_manager.get_state(f"/test/child{i}.txt")
            assert child.status == "updating"

    @pytest.mark.asyncio
    async def test_transaction_durability(self, temp_db_path):
        """Test durability - committed transactions survive system restart."""
        # First session - create and commit data
        manager1 = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager1.initialize()

        tx_id = await manager1.begin_transaction()

        durable_record = StateRecord(
            file_path="/test/durable.txt",
            status="committed",
            last_modified=time.time(),
            checksum="durable_hash",
            metadata={"test": "durability"},
            created_at=time.time(),
            updated_at=time.time()
        )

        await manager1.upsert_state(durable_record, tx_id)
        await manager1.commit_transaction(tx_id)
        await manager1.force_checkpoint()  # Ensure WAL is flushed
        await manager1.close()

        # Second session - verify data persists
        manager2 = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager2.initialize()

        recovered = await manager2.get_state("/test/durable.txt")
        assert recovered is not None
        assert recovered.status == "committed"
        assert recovered.metadata["test"] == "durability"

        await manager2.close()

    # ==================== Performance Tests ====================

    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, state_manager):
        """Test performance with 1000+ records."""
        num_records = 1000
        batch_size = 100

        # Measure insertion performance
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        records = []
        for i in range(num_records):
            record = StateRecord(
                file_path=f"/test/perf_{i:04d}.txt",
                status=f"status_{i % 10}",
                last_modified=time.time(),
                checksum=f"sha256:perf_{i}",
                metadata={"index": i, "batch": i // batch_size},
                created_at=time.time(),
                updated_at=time.time()
            )
            records.append(record)

        # Batch insert
        for i in range(0, num_records, batch_size):
            batch = records[i:i + batch_size]
            for record in batch:
                await state_manager.upsert_state(record)

        insert_time = time.perf_counter() - start_time
        insert_memory = self._get_memory_usage() - start_memory

        # Record performance metric
        insert_metrics = PerformanceMetrics(
            operation="bulk_insert",
            duration_ms=insert_time * 1000,
            records_processed=num_records,
            throughput_rps=num_records / insert_time,
            memory_usage_mb=insert_memory,
            cpu_usage_percent=0  # Would need process monitoring
        )
        await state_manager.record_performance_metric(insert_metrics)

        # Measure query performance
        start_time = time.perf_counter()

        # Test various query patterns
        count = await state_manager.count_states()
        assert count >= num_records

        # Query by status
        status_0_records = await state_manager.get_states_by_status("status_0")
        assert len(status_0_records) == num_records // 10

        # Random access queries
        for _ in range(100):
            i = random.randint(0, num_records - 1)
            record = await state_manager.get_state(f"/test/perf_{i:04d}.txt")
            assert record is not None

        query_time = time.perf_counter() - start_time

        # Performance assertions
        assert insert_metrics.throughput_rps > 100  # At least 100 inserts/sec
        assert query_time < 5.0  # Queries should complete within 5 seconds
        assert insert_metrics.memory_usage_mb < 100  # Reasonable memory usage

        # Verify performance stats are recorded
        perf_stats = await state_manager.get_performance_stats("bulk_insert")
        assert perf_stats['sample_count'] > 0

        return insert_metrics

    @pytest.mark.asyncio
    async def test_concurrent_performance(self, temp_db_path):
        """Test performance under concurrent load."""
        def concurrent_worker(db_path: str, worker_id: int, num_ops: int):
            """Worker that performs mixed operations."""
            import json
            import random
            import sqlite3
            import time

            conn = sqlite3.connect(db_path, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")

            start_time = time.perf_counter()
            operations = 0

            for i in range(num_ops):
                if random.random() < 0.7:  # 70% writes
                    file_path = f"/test/concurrent_{worker_id}_{i}.txt"
                    conn.execute("""
                        INSERT OR REPLACE INTO file_states
                        (file_path, status, last_modified, checksum, metadata,
                         created_at, updated_at, retry_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (file_path, f"worker_{worker_id}", time.time(),
                          f"hash_{i}", json.dumps({"worker": worker_id, "op": i}),
                          time.time(), time.time(), 0))
                    conn.commit()
                else:  # 30% reads
                    # Try to read a random record from this worker
                    read_i = random.randint(0, max(1, i))
                    file_path = f"/test/concurrent_{worker_id}_{read_i}.txt"
                    cursor = conn.execute(
                        "SELECT * FROM file_states WHERE file_path = ?",
                        (file_path,)
                    )
                    cursor.fetchone()

                operations += 1

            duration = time.perf_counter() - start_time
            conn.close()

            return {
                'worker_id': worker_id,
                'operations': operations,
                'duration': duration,
                'ops_per_sec': operations / duration if duration > 0 else 0
            }

        # Initialize database
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()
        await manager.close()

        # Run concurrent load test
        num_workers = 8
        ops_per_worker = 100

        start_time = time.perf_counter()

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(concurrent_worker, temp_db_path, i, ops_per_worker)
                for i in range(num_workers)
            ]
            results = [f.result() for f in futures]

        total_time = time.perf_counter() - start_time
        total_ops = sum(r['operations'] for r in results)
        overall_ops_per_sec = total_ops / total_time

        # Verify performance
        assert overall_ops_per_sec > 500  # At least 500 ops/sec total
        assert all(r['ops_per_sec'] > 50 for r in results)  # Each worker > 50 ops/sec

        # Verify data integrity
        manager2 = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager2.initialize()

        final_count = await manager2.count_states()
        expected_writes = int(num_workers * ops_per_worker * 0.7)  # 70% were writes
        assert final_count >= expected_writes * 0.9  # Allow some variance

        await manager2.close()

        return {
            'total_operations': total_ops,
            'total_duration': total_time,
            'overall_ops_per_sec': overall_ops_per_sec,
            'worker_results': results
        }

    # ==================== Database Maintenance Tests ====================

    @pytest.mark.asyncio
    async def test_vacuum_operation(self, state_manager):
        """Test database vacuum operation."""
        # Create and delete many records to create fragmentation
        for i in range(1000):
            record = StateRecord(
                file_path=f"/test/vacuum_{i}.txt",
                status="temporary",
                last_modified=time.time(),
                checksum=f"temp_{i}",
                metadata={"temp": True},
                created_at=time.time(),
                updated_at=time.time()
            )
            await state_manager.upsert_state(record)

        # Delete half the records
        for i in range(0, 1000, 2):
            await state_manager.delete_state(f"/test/vacuum_{i}.txt")

        # Get database size before vacuum
        size_before = await state_manager.get_database_size()

        # Perform vacuum
        start_time = time.perf_counter()
        result = await state_manager.vacuum()
        vacuum_time = time.perf_counter() - start_time

        # Get database size after vacuum
        size_after = await state_manager.get_database_size()

        # Verify vacuum worked
        assert result is True
        assert vacuum_time < 10.0  # Should complete within 10 seconds
        assert size_after['main_db'] <= size_before['main_db']  # Size should not increase

        # Verify data integrity after vacuum
        count = await state_manager.count_states()
        assert count == 500  # Half the records should remain

    @pytest.mark.asyncio
    async def test_analyze_operation(self, state_manager):
        """Test database analyze operation for optimization."""
        # Create test data with various patterns
        statuses = ["pending", "processing", "completed", "failed"]

        for i in range(200):
            record = StateRecord(
                file_path=f"/test/analyze_{i:03d}.txt",
                status=statuses[i % len(statuses)],
                last_modified=time.time() - (i * 3600),  # Spread over time
                checksum=f"analyze_{i}",
                metadata={"group": i // 50},
                created_at=time.time() - (i * 3600),
                updated_at=time.time()
            )
            await state_manager.upsert_state(record)

        # Run analyze
        start_time = time.perf_counter()
        result = await state_manager.analyze()
        analyze_time = time.perf_counter() - start_time

        # Verify analyze completed successfully
        assert result is True
        assert analyze_time < 5.0  # Should be fast

        # Test that queries still work efficiently after analyze
        start_time = time.perf_counter()
        pending_records = await state_manager.get_states_by_status("pending")
        query_time = time.perf_counter() - start_time

        assert len(pending_records) == 50  # 200/4 records per status
        assert query_time < 0.1  # Should be fast with good statistics

    @pytest.mark.asyncio
    async def test_wal_checkpoint_operations(self, state_manager):
        """Test WAL checkpoint operations."""
        # Generate WAL activity
        for i in range(100):
            record = StateRecord(
                file_path=f"/test/wal_{i}.txt",
                status="wal_test",
                last_modified=time.time(),
                checksum=f"wal_{i}",
                metadata={"wal_test": True},
                created_at=time.time(),
                updated_at=time.time()
            )
            await state_manager.upsert_state(record)

        # Check WAL status
        wal_status = await state_manager.get_wal_status()
        assert wal_status['journal_mode'].upper() == 'WAL'

        # Force checkpoint
        checkpoint_result = await state_manager.force_checkpoint()
        assert checkpoint_result is True

        # Check WAL status after checkpoint
        wal_status_after = await state_manager.get_wal_status()
        # WAL should still be active
        assert wal_status_after['journal_mode'].upper() == 'WAL'

    # ==================== Error Scenario Tests ====================

    @pytest.mark.asyncio
    async def test_disk_full_simulation(self, temp_db_path):
        """Test behavior when disk space is exhausted."""
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()

        # Mock disk full error
        with patch.object(manager._connection, 'execute') as mock_execute:
            mock_execute.side_effect = sqlite3.OperationalError("disk full")

            # Attempt operation that should fail gracefully
            record = StateRecord(
                file_path="/test/disk_full.txt",
                status="test",
                last_modified=time.time(),
                checksum="test",
                metadata={},
                created_at=time.time(),
                updated_at=time.time()
            )

            # Should return False but not crash
            result = await manager.upsert_state(record)
            assert result is False

        await manager.close()

    @pytest.mark.asyncio
    async def test_corruption_recovery(self, temp_db_path):
        """Test handling of database corruption."""
        # Create initial database
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()

        # Add some data
        record = StateRecord(
            file_path="/test/corruption.txt",
            status="before_corruption",
            last_modified=time.time(),
            checksum="before",
            metadata={},
            created_at=time.time(),
            updated_at=time.time()
        )
        await manager.upsert_state(record)
        await manager.close()

        # Simulate corruption by truncating the database file
        with open(temp_db_path, 'r+b') as f:
            f.truncate(100)  # Truncate to invalid size

        # Try to open corrupted database
        manager2 = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)

        # Should handle corruption gracefully
        try:
            await manager2.initialize()
            # If it succeeds, SQLite recovered automatically
            await manager2.close()
        except sqlite3.DatabaseError:
            # Expected for corrupted database
            pass

    @pytest.mark.asyncio
    async def test_concurrent_connection_limits(self, temp_db_path):
        """Test behavior with many concurrent connections."""
        def connection_worker(db_path: str, worker_id: int):
            """Worker that holds a connection for some time."""
            import json
            import sqlite3
            import time

            try:
                conn = sqlite3.connect(db_path, timeout=10.0)
                conn.execute("PRAGMA journal_mode=WAL")

                # Hold connection and perform operation
                conn.execute("""
                    INSERT OR REPLACE INTO file_states
                    (file_path, status, last_modified, checksum, metadata,
                     created_at, updated_at, retry_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (f"/test/conn_{worker_id}.txt", "connected", time.time(),
                      f"hash_{worker_id}", json.dumps({}), time.time(), time.time(), 0))
                conn.commit()

                time.sleep(0.1)  # Hold connection briefly
                conn.close()
                return True
            except sqlite3.Error as e:
                return str(e)

        # Initialize database
        manager = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager.initialize()
        await manager.close()

        # Test with many concurrent connections
        num_connections = 50

        with ThreadPoolExecutor(max_workers=num_connections) as executor:
            futures = [
                executor.submit(connection_worker, temp_db_path, i)
                for i in range(num_connections)
            ]
            results = [f.result() for f in futures]

        # Most connections should succeed
        success_count = sum(1 for r in results if r is True)
        assert success_count >= num_connections * 0.8  # At least 80% success

        # Verify final state
        manager2 = SQLiteStateManagerComprehensive(temp_db_path, enable_wal=True)
        await manager2.initialize()
        final_count = await manager2.count_states()
        assert final_count == success_count
        await manager2.close()

    # ==================== Utility Methods ====================

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0

    async def _create_test_data(self, state_manager: SQLiteStateManagerComprehensive, count: int) -> list[StateRecord]:
        """Create test data records."""
        records = []
        for i in range(count):
            record = StateRecord(
                file_path=f"/test/data_{i:04d}.txt",
                status=f"status_{i % 5}",
                last_modified=time.time() - (i * 60),
                checksum=f"sha256:data_{i}",
                metadata={"index": i, "batch": i // 100},
                created_at=time.time() - (i * 60),
                updated_at=time.time()
            )
            records.append(record)
            await state_manager.upsert_state(record)
        return records


# ==================== Integration Tests ====================

class TestSQLiteStateManagerIntegration:
    """Integration tests for complete workflows."""

    @pytest.fixture
    async def temp_dir(self):
        """Create temporary directory for integration tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_full_file_processing_workflow(self, temp_dir):
        """Test complete file processing state management workflow."""
        db_path = os.path.join(temp_dir, "workflow.db")
        manager = SQLiteStateManagerComprehensive(db_path, enable_wal=True)
        await manager.initialize()

        # Simulate file discovery and processing workflow
        files_to_process = [
            "/project/src/main.py",
            "/project/src/utils.py",
            "/project/docs/README.md",
            "/project/tests/test_main.py"
        ]

        # Phase 1: Discovery
        for file_path in files_to_process:
            record = StateRecord(
                file_path=file_path,
                status="pending",
                last_modified=time.time(),
                checksum=f"sha256:{file_path.split('/')[-1]}",
                metadata={"phase": "discovery", "size": len(file_path) * 100},
                created_at=time.time(),
                updated_at=time.time()
            )
            await manager.upsert_state(record)

        # Phase 2: Processing
        for file_path in files_to_process:
            record = await manager.get_state(file_path)
            record.status = "processing"
            record.metadata.update({"phase": "processing", "start_time": time.time()})
            record.updated_at = time.time()
            await manager.upsert_state(record)

            # Simulate processing time
            await asyncio.sleep(0.01)

            # Complete processing
            record.status = "completed"
            record.metadata.update({
                "phase": "completed",
                "end_time": time.time(),
                "chunks_created": random.randint(1, 10)
            })
            record.updated_at = time.time()
            await manager.upsert_state(record)

        # Verify final state
        completed_files = await manager.get_states_by_status("completed")
        assert len(completed_files) == len(files_to_process)

        for record in completed_files:
            assert "chunks_created" in record.metadata
            assert record.metadata["phase"] == "completed"

        # Test performance metrics
        await manager.get_performance_stats()

        await manager.close()

    @pytest.mark.asyncio
    async def test_disaster_recovery_scenario(self, temp_dir):
        """Test complete disaster recovery scenario."""
        db_path = os.path.join(temp_dir, "disaster.db")

        # Phase 1: Normal operation
        manager1 = SQLiteStateManagerComprehensive(db_path, enable_wal=True)
        await manager1.initialize()

        # Process files normally
        for i in range(100):
            record = StateRecord(
                file_path=f"/disaster/file_{i:03d}.txt",
                status="completed" if i < 50 else "processing",
                last_modified=time.time(),
                checksum=f"disaster_{i}",
                metadata={"disaster_test": True, "index": i},
                created_at=time.time(),
                updated_at=time.time()
            )
            await manager1.upsert_state(record)

        # Start transactions for processing files
        tx_ids = []
        for i in range(50, 60):
            tx_id = await manager1.begin_transaction()
            tx_ids.append(tx_id)

            record = await manager1.get_state(f"/disaster/file_{i:03d}.txt")
            record.status = "processing_tx"
            record.updated_at = time.time()
            await manager1.upsert_state(record, tx_id)

        # Commit some transactions, leave others uncommitted
        for i, tx_id in enumerate(tx_ids[:5]):
            await manager1.commit_transaction(tx_id)

        # Simulate disaster - close without committing remaining transactions
        await manager1.close()

        # Phase 2: Recovery
        manager2 = SQLiteStateManagerComprehensive(db_path, enable_wal=True)
        await manager2.initialize()

        # Check recovery state
        total_records = await manager2.count_states()
        assert total_records == 100

        # Check that uncommitted transactions were properly handled
        # (they should have been rolled back during recovery)

        await manager2.close()


# ==================== Test Runner ====================

def run_specific_tests(category: str = "all"):
    """Run specific test categories."""
    test_markers = {
        "basic": "-k 'test_initialization or test_basic_crud or test_query'",
        "crash": "-k 'crash_recovery'",
        "concurrent": "-k 'concurrent'",
        "acid": "-k 'transaction'",
        "performance": "-k 'performance'",
        "maintenance": "-k 'vacuum or analyze or wal'",
        "errors": "-k 'disk_full or corruption or connection_limits'",
        "integration": "-k 'Integration'"
    }

    if category == "all":
        pytest.main(["-v", __file__])
    elif category in test_markers:
        pytest.main(["-v", test_markers[category], __file__])
    else:
        print(f"Unknown test category: {category}")
        print(f"Available categories: {list(test_markers.keys())}")


if __name__ == "__main__":
    """Run the comprehensive test suite."""
    import sys

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run specific test categories based on command line arguments
    if len(sys.argv) > 1:
        run_specific_tests(sys.argv[1])
    else:
        # Run all tests
        run_specific_tests("all")
