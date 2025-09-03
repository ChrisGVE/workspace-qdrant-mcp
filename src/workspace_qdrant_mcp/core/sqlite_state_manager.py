"""
SQLite State Persistence Manager with Crash Recovery.

This module provides bulletproof state persistence using SQLite with WAL mode
for tracking ingestion progress, watch folders, and processing status with
full crash recovery capabilities.

Key Features:
    - SQLite database with WAL mode for crash resistance and concurrent access
    - ACID transaction support with proper rollback handling
    - Ingestion progress tracking with atomic markers
    - Persistent watch folder configurations that survive daemon restarts
    - Failed/skipped file registry with detailed error tracking
    - Processing queue state management with priority handling
    - Graceful shutdown handlers for state preservation
    - Startup recovery procedures for interrupted operations
    - Database schema migrations for future updates
    - State cleanup and maintenance procedures

Example:
    ```python
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager

    # Initialize state manager
    state_manager = SQLiteStateManager(db_path="./workspace_state.db")
    await state_manager.initialize()

    # Track file ingestion progress
    await state_manager.start_file_processing(file_path, collection)
    await state_manager.complete_file_processing(file_path, success=True)

    # Persist watch folder configuration
    await state_manager.save_watch_folder_config(watch_id, config)
    ```
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class FileProcessingStatus(Enum):
    """File processing status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ProcessingPriority(Enum):
    """Processing priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class FileProcessingRecord:
    """Record for tracking file processing state."""

    file_path: str
    collection: str
    status: FileProcessingStatus
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    created_at: datetime = None
    updated_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class WatchFolderConfig:
    """Configuration for persistent watch folders."""

    watch_id: str
    path: str
    collection: str
    patterns: List[str]
    ignore_patterns: List[str]
    auto_ingest: bool = True
    recursive: bool = True
    recursive_depth: int = 10
    debounce_seconds: float = 2.0
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    last_scan: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class ProcessingQueueItem:
    """Item in the processing queue."""

    queue_id: str
    file_path: str
    collection: str
    priority: ProcessingPriority
    created_at: datetime = None
    scheduled_at: Optional[datetime] = None
    attempts: int = 0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at


class DatabaseTransaction:
    """Context manager for ACID transactions with proper error handling."""

    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self.savepoint_name = None

    def __enter__(self):
        # Use savepoints for nested transaction support
        self.savepoint_name = f"sp_{int(time.time() * 1000000)}"
        self.connection.execute(f"SAVEPOINT {self.savepoint_name}")
        return self.connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Commit the savepoint
            self.connection.execute(f"RELEASE SAVEPOINT {self.savepoint_name}")
        else:
            # Rollback the savepoint
            self.connection.execute(f"ROLLBACK TO SAVEPOINT {self.savepoint_name}")
            logger.error(f"Transaction rolled back due to error: {exc_val}")
        return False  # Don't suppress exceptions


class SQLiteStateManager:
    """SQLite-based state persistence manager with crash recovery."""

    SCHEMA_VERSION = 1
    WAL_CHECKPOINT_INTERVAL = 300  # 5 minutes
    MAINTENANCE_INTERVAL = 3600  # 1 hour

    def __init__(self, db_path: str = "workspace_state.db"):
        """
        Initialize SQLite state manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.connection: Optional[sqlite3.Connection] = None
        self._lock = threading.RLock()
        self._shutdown_event = asyncio.Event()
        self._maintenance_task: Optional[asyncio.Task] = None
        self._initialized = False

    async def initialize(self) -> bool:
        """
        Initialize the database and set up WAL mode.

        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True

        try:
            logger.info(f"Initializing SQLite state manager: {self.db_path}")

            # Create database directory if needed
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Initialize connection with WAL mode
            self.connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,
                check_same_thread=False,
                isolation_level=None,  # autocommit mode, we'll use explicit transactions
            )

            # Enable WAL mode for crash resistance and concurrent access
            self.connection.execute("PRAGMA journal_mode=WAL")

            # Optimize for performance and reliability
            self.connection.execute(
                "PRAGMA synchronous=NORMAL"
            )  # Balance between safety and speed
            self.connection.execute("PRAGMA cache_size=10000")  # 10MB cache
            self.connection.execute(
                "PRAGMA temp_store=MEMORY"
            )  # Use memory for temp tables
            self.connection.execute(
                "PRAGMA mmap_size=268435456"
            )  # 256MB memory mapping
            self.connection.execute(
                "PRAGMA wal_autocheckpoint=1000"
            )  # Auto-checkpoint every 1000 pages

            # Enable foreign key constraints
            self.connection.execute("PRAGMA foreign_keys=ON")

            # Set row factory for dict-like access
            self.connection.row_factory = sqlite3.Row

            # Create or migrate schema
            await self._setup_schema()

            # Perform crash recovery
            await self._perform_crash_recovery()

            # Start maintenance tasks
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())

            self._initialized = True
            logger.info("SQLite state manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize SQLite state manager: {e}")
            if self.connection:
                self.connection.close()
                self.connection = None
            return False

    async def close(self):
        """Close the database connection and cleanup resources."""
        if not self._initialized:
            return

        logger.info("Closing SQLite state manager")

        # Signal shutdown
        self._shutdown_event.set()

        # Stop maintenance task
        if self._maintenance_task and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass

        # Perform final checkpoint and close connection
        if self.connection:
            try:
                with self._lock:
                    # Final WAL checkpoint
                    self.connection.execute("PRAGMA wal_checkpoint(FULL)")
                    self.connection.close()
                    self.connection = None

                logger.info("SQLite state manager closed successfully")
            except Exception as e:
                logger.error(f"Error closing SQLite connection: {e}")

        self._initialized = False

    @asynccontextmanager
    async def transaction(self):
        """Async context manager for database transactions."""
        if not self.connection:
            raise RuntimeError("Database not initialized")

        with self._lock:
            transaction = DatabaseTransaction(self.connection)
            with transaction as conn:
                yield conn

    async def _setup_schema(self):
        """Create or migrate database schema."""
        with self._lock:
            # Check current schema version
            cursor = self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )

            if not cursor.fetchone():
                # First time setup
                await self._create_initial_schema()
            else:
                # Check for migrations
                cursor = self.connection.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                )
                current_version = cursor.fetchone()[0]

                if current_version < self.SCHEMA_VERSION:
                    await self._migrate_schema(current_version, self.SCHEMA_VERSION)

    async def _create_initial_schema(self):
        """Create initial database schema."""
        logger.info("Creating initial database schema")

        schema_sql = [
            # Schema version tracking
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # File processing records
            """
            CREATE TABLE file_processing (
                file_path TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 2,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                error_message TEXT,
                file_size INTEGER,
                file_hash TEXT,
                metadata TEXT  -- JSON
            )
            """,
            # Indexes for file_processing
            "CREATE INDEX idx_file_processing_status ON file_processing(status)",
            "CREATE INDEX idx_file_processing_collection ON file_processing(collection)",
            "CREATE INDEX idx_file_processing_updated_at ON file_processing(updated_at)",
            "CREATE INDEX idx_file_processing_priority ON file_processing(priority)",
            # Watch folder configurations
            """
            CREATE TABLE watch_folders (
                watch_id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                collection TEXT NOT NULL,
                patterns TEXT NOT NULL,  -- JSON array
                ignore_patterns TEXT NOT NULL,  -- JSON array
                auto_ingest BOOLEAN NOT NULL DEFAULT 1,
                recursive BOOLEAN NOT NULL DEFAULT 1,
                recursive_depth INTEGER NOT NULL DEFAULT 10,
                debounce_seconds REAL NOT NULL DEFAULT 2.0,
                enabled BOOLEAN NOT NULL DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_scan TIMESTAMP,
                metadata TEXT  -- JSON
            )
            """,
            # Indexes for watch_folders
            "CREATE INDEX idx_watch_folders_path ON watch_folders(path)",
            "CREATE INDEX idx_watch_folders_enabled ON watch_folders(enabled)",
            "CREATE INDEX idx_watch_folders_collection ON watch_folders(collection)",
            # Processing queue
            """
            CREATE TABLE processing_queue (
                queue_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                collection TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 2,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                scheduled_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                attempts INTEGER NOT NULL DEFAULT 0,
                metadata TEXT,  -- JSON
                FOREIGN KEY (file_path) REFERENCES file_processing (file_path) ON DELETE CASCADE
            )
            """,
            # Indexes for processing_queue
            "CREATE INDEX idx_processing_queue_priority ON processing_queue(priority DESC, scheduled_at ASC)",
            "CREATE INDEX idx_processing_queue_file_path ON processing_queue(file_path)",
            "CREATE INDEX idx_processing_queue_scheduled_at ON processing_queue(scheduled_at)",
            # System state table for tracking overall system status
            """
            CREATE TABLE system_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Processing history for analytics and debugging
            """
            CREATE TABLE processing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                collection TEXT NOT NULL,
                status TEXT NOT NULL,
                processing_time_ms INTEGER,
                file_size INTEGER,
                error_message TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON
            )
            """,
            # Indexes for processing_history
            "CREATE INDEX idx_processing_history_file_path ON processing_history(file_path)",
            "CREATE INDEX idx_processing_history_status ON processing_history(status)",
            "CREATE INDEX idx_processing_history_created_at ON processing_history(created_at)",
            # Insert initial schema version
            f"INSERT INTO schema_version (version) VALUES ({self.SCHEMA_VERSION})",
        ]

        with self.connection:
            for sql in schema_sql:
                self.connection.execute(sql)

        logger.info(f"Created database schema version {self.SCHEMA_VERSION}")

    async def _migrate_schema(self, from_version: int, to_version: int):
        """Migrate database schema between versions."""
        logger.info(f"Migrating database schema from {from_version} to {to_version}")

        # Future migrations will be implemented here
        # For now, we only have version 1

        with self.connection:
            self.connection.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (to_version,)
            )

    async def _perform_crash_recovery(self):
        """Perform crash recovery operations on startup."""
        logger.info("Performing crash recovery")

        recovery_operations = 0

        async with self.transaction() as conn:
            # Find files that were being processed when system crashed
            cursor = conn.execute(
                """
                SELECT file_path, collection, started_at, retry_count, max_retries
                FROM file_processing 
                WHERE status = ? AND started_at IS NOT NULL
                """,
                (FileProcessingStatus.PROCESSING.value,),
            )

            crashed_files = cursor.fetchall()

            for row in crashed_files:
                file_path = row["file_path"]
                collection = row["collection"]
                retry_count = row["retry_count"]
                max_retries = row["max_retries"]

                if retry_count < max_retries:
                    # Mark for retry
                    conn.execute(
                        """
                        UPDATE file_processing 
                        SET status = ?, retry_count = retry_count + 1, 
                            updated_at = CURRENT_TIMESTAMP, started_at = NULL,
                            error_message = 'Recovered from crash, retrying'
                        WHERE file_path = ?
                        """,
                        (FileProcessingStatus.RETRYING.value, file_path),
                    )

                    # Add back to processing queue with higher priority
                    queue_id = (
                        f"recovery_{int(time.time() * 1000)}_{hash(file_path) % 10000}"
                    )
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO processing_queue 
                        (queue_id, file_path, collection, priority, attempts)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            queue_id,
                            file_path,
                            collection,
                            ProcessingPriority.HIGH.value,
                            retry_count + 1,
                        ),
                    )

                    recovery_operations += 1
                    logger.info(f"Marked crashed file for retry: {file_path}")
                else:
                    # Max retries exceeded, mark as failed
                    conn.execute(
                        """
                        UPDATE file_processing 
                        SET status = ?, updated_at = CURRENT_TIMESTAMP, started_at = NULL,
                            completed_at = CURRENT_TIMESTAMP,
                            error_message = 'Max retries exceeded after crash recovery'
                        WHERE file_path = ?
                        """,
                        (FileProcessingStatus.FAILED.value, file_path),
                    )

                    recovery_operations += 1
                    logger.warning(
                        f"Marked crashed file as failed (max retries): {file_path}"
                    )

            # Clean up orphaned queue items
            cursor = conn.execute(
                """
                DELETE FROM processing_queue 
                WHERE file_path NOT IN (SELECT file_path FROM file_processing)
                """
            )
            orphaned_items = cursor.rowcount

            if orphaned_items > 0:
                recovery_operations += orphaned_items
                logger.info(f"Cleaned up {orphaned_items} orphaned queue items")

        if recovery_operations > 0:
            logger.info(f"Crash recovery completed: {recovery_operations} operations")
        else:
            logger.info("Crash recovery completed: no issues found")

    async def _maintenance_loop(self):
        """Background maintenance loop."""
        logger.info("Starting maintenance loop")

        try:
            while not self._shutdown_event.is_set():
                try:
                    # Perform maintenance operations
                    await self._perform_maintenance()

                    # Wait for next maintenance interval
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=self.MAINTENANCE_INTERVAL
                    )

                except asyncio.TimeoutError:
                    # Normal timeout, continue maintenance
                    continue
                except Exception as e:
                    logger.error(f"Error in maintenance loop: {e}")
                    # Brief pause before retrying
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info("Maintenance loop cancelled")
        except Exception as e:
            logger.error(f"Maintenance loop failed: {e}")

    async def _perform_maintenance(self):
        """Perform regular maintenance operations."""
        logger.debug("Performing database maintenance")

        try:
            with self._lock:
                # WAL checkpoint
                self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")

                # Analyze query plans for optimization
                self.connection.execute("ANALYZE")

                # Clean up old processing history (keep last 30 days)
                cutoff_date = datetime.now(timezone.utc).timestamp() - (30 * 24 * 3600)
                cursor = self.connection.execute(
                    "DELETE FROM processing_history WHERE created_at < datetime(?, 'unixepoch')",
                    (cutoff_date,),
                )

                if cursor.rowcount > 0:
                    logger.info(
                        f"Cleaned up {cursor.rowcount} old processing history records"
                    )

                # Update system state with last maintenance time
                self.connection.execute(
                    "INSERT OR REPLACE INTO system_state (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    ("last_maintenance", datetime.now(timezone.utc).isoformat()),
                )

                self.connection.commit()

        except Exception as e:
            logger.error(f"Error during maintenance: {e}")

    def _serialize_json(self, data: Any) -> Optional[str]:
        """Serialize data to JSON string."""
        if data is None:
            return None
        try:
            return json.dumps(data)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize data to JSON: {e}")
            return None

    def _deserialize_json(self, data: Optional[str]) -> Any:
        """Deserialize JSON string to data."""
        if not data:
            return None
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to deserialize JSON data: {e}")
            return None

    # File Processing State Management

    async def start_file_processing(
        self,
        file_path: str,
        collection: str,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        file_size: Optional[int] = None,
        file_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a file as starting processing with atomic transaction."""
        try:
            async with self.transaction() as conn:
                # Insert or update file processing record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO file_processing 
                    (file_path, collection, status, priority, started_at, updated_at, 
                     file_size, file_hash, metadata, retry_count)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, 
                           COALESCE((SELECT retry_count FROM file_processing WHERE file_path = ?), 0))
                    """,
                    (
                        file_path,
                        collection,
                        FileProcessingStatus.PROCESSING.value,
                        priority.value,
                        file_size,
                        file_hash,
                        self._serialize_json(metadata),
                        file_path,
                    ),
                )

                # Remove from processing queue if exists
                conn.execute(
                    "DELETE FROM processing_queue WHERE file_path = ?", (file_path,)
                )

            logger.debug(f"Started processing file: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to start file processing {file_path}: {e}")
            return False

    async def complete_file_processing(
        self,
        file_path: str,
        success: bool,
        error_message: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a file as completed (success or failure) with atomic transaction."""
        try:
            status = (
                FileProcessingStatus.COMPLETED
                if success
                else FileProcessingStatus.FAILED
            )

            async with self.transaction() as conn:
                # Update file processing record
                cursor = conn.execute(
                    """
                    UPDATE file_processing 
                    SET status = ?, completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP,
                        error_message = ?, metadata = ?
                    WHERE file_path = ?
                    """,
                    (
                        status.value,
                        error_message,
                        self._serialize_json(metadata),
                        file_path,
                    ),
                )

                if cursor.rowcount == 0:
                    logger.warning(f"File not found in processing records: {file_path}")
                    return False

                # Get file info for history record
                cursor = conn.execute(
                    "SELECT collection, file_size FROM file_processing WHERE file_path = ?",
                    (file_path,),
                )
                row = cursor.fetchone()

                if row:
                    # Add to processing history
                    conn.execute(
                        """
                        INSERT INTO processing_history 
                        (file_path, collection, status, processing_time_ms, file_size, 
                         error_message, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            file_path,
                            row["collection"],
                            status.value,
                            processing_time_ms,
                            row["file_size"],
                            error_message,
                            self._serialize_json(metadata),
                        ),
                    )

                # Clean up processing queue
                conn.execute(
                    "DELETE FROM processing_queue WHERE file_path = ?", (file_path,)
                )

            logger.debug(f"Completed file processing: {file_path} (success: {success})")
            return True

        except Exception as e:
            logger.error(f"Failed to complete file processing {file_path}: {e}")
            return False

    async def get_file_processing_status(
        self, file_path: str
    ) -> Optional[FileProcessingRecord]:
        """Get current processing status for a file."""
        try:
            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT file_path, collection, status, priority, created_at, updated_at,
                           started_at, completed_at, retry_count, max_retries, error_message,
                           file_size, file_hash, metadata
                    FROM file_processing
                    WHERE file_path = ?
                    """,
                    (file_path,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return FileProcessingRecord(
                    file_path=row["file_path"],
                    collection=row["collection"],
                    status=FileProcessingStatus(row["status"]),
                    priority=ProcessingPriority(row["priority"]),
                    created_at=datetime.fromisoformat(
                        row["created_at"].replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        row["updated_at"].replace("Z", "+00:00")
                    ),
                    started_at=datetime.fromisoformat(
                        row["started_at"].replace("Z", "+00:00")
                    )
                    if row["started_at"]
                    else None,
                    completed_at=datetime.fromisoformat(
                        row["completed_at"].replace("Z", "+00:00")
                    )
                    if row["completed_at"]
                    else None,
                    retry_count=row["retry_count"],
                    max_retries=row["max_retries"],
                    error_message=row["error_message"],
                    file_size=row["file_size"],
                    file_hash=row["file_hash"],
                    metadata=self._deserialize_json(row["metadata"]),
                )

        except Exception as e:
            logger.error(f"Failed to get file processing status {file_path}: {e}")
            return None

    async def get_files_by_status(
        self,
        status: FileProcessingStatus,
        collection: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[FileProcessingRecord]:
        """Get files with a specific processing status."""
        try:
            with self._lock:
                sql = """
                    SELECT file_path, collection, status, priority, created_at, updated_at,
                           started_at, completed_at, retry_count, max_retries, error_message,
                           file_size, file_hash, metadata
                    FROM file_processing
                    WHERE status = ?
                """

                params = [status.value]

                if collection:
                    sql += " AND collection = ?"
                    params.append(collection)

                sql += " ORDER BY priority DESC, created_at ASC"

                if limit:
                    sql += " LIMIT ?"
                    params.append(limit)

                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()

                records = []
                for row in rows:
                    records.append(
                        FileProcessingRecord(
                            file_path=row["file_path"],
                            collection=row["collection"],
                            status=FileProcessingStatus(row["status"]),
                            priority=ProcessingPriority(row["priority"]),
                            created_at=datetime.fromisoformat(
                                row["created_at"].replace("Z", "+00:00")
                            ),
                            updated_at=datetime.fromisoformat(
                                row["updated_at"].replace("Z", "+00:00")
                            ),
                            started_at=datetime.fromisoformat(
                                row["started_at"].replace("Z", "+00:00")
                            )
                            if row["started_at"]
                            else None,
                            completed_at=datetime.fromisoformat(
                                row["completed_at"].replace("Z", "+00:00")
                            )
                            if row["completed_at"]
                            else None,
                            retry_count=row["retry_count"],
                            max_retries=row["max_retries"],
                            error_message=row["error_message"],
                            file_size=row["file_size"],
                            file_hash=row["file_hash"],
                            metadata=self._deserialize_json(row["metadata"]),
                        )
                    )

                return records

        except Exception as e:
            logger.error(f"Failed to get files by status {status}: {e}")
            return []

    async def retry_failed_file(
        self, file_path: str, max_retries: Optional[int] = None
    ) -> bool:
        """Mark a failed file for retry."""
        try:
            async with self.transaction() as conn:
                # Check current status
                cursor = conn.execute(
                    "SELECT status, retry_count, max_retries FROM file_processing WHERE file_path = ?",
                    (file_path,),
                )
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"File not found for retry: {file_path}")
                    return False

                current_status = row["status"]
                retry_count = row["retry_count"]
                current_max_retries = row["max_retries"]

                if current_status not in [
                    FileProcessingStatus.FAILED.value,
                    FileProcessingStatus.SKIPPED.value,
                ]:
                    logger.warning(
                        f"File not in failed/skipped state for retry: {file_path} (status: {current_status})"
                    )
                    return False

                # Update max retries if provided
                new_max_retries = (
                    max_retries if max_retries is not None else current_max_retries
                )

                if retry_count >= new_max_retries:
                    logger.warning(
                        f"File already at max retries: {file_path} ({retry_count}/{new_max_retries})"
                    )
                    return False

                # Mark for retry
                conn.execute(
                    """
                    UPDATE file_processing 
                    SET status = ?, updated_at = CURRENT_TIMESTAMP, max_retries = ?,
                        started_at = NULL, completed_at = NULL, error_message = NULL
                    WHERE file_path = ?
                    """,
                    (FileProcessingStatus.RETRYING.value, new_max_retries, file_path),
                )

            logger.info(f"Marked file for retry: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to retry file {file_path}: {e}")
            return False

    # Watch Folder Management

    async def save_watch_folder_config(self, config: WatchFolderConfig) -> bool:
        """Save watch folder configuration persistently."""
        try:
            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO watch_folders 
                    (watch_id, path, collection, patterns, ignore_patterns, auto_ingest,
                     recursive, recursive_depth, debounce_seconds, enabled, 
                     created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                           COALESCE((SELECT created_at FROM watch_folders WHERE watch_id = ?), CURRENT_TIMESTAMP),
                           CURRENT_TIMESTAMP, ?)
                    """,
                    (
                        config.watch_id,
                        config.path,
                        config.collection,
                        self._serialize_json(config.patterns),
                        self._serialize_json(config.ignore_patterns),
                        config.auto_ingest,
                        config.recursive,
                        config.recursive_depth,
                        config.debounce_seconds,
                        config.enabled,
                        config.watch_id,
                        self._serialize_json(config.metadata),
                    ),
                )

            logger.debug(f"Saved watch folder config: {config.watch_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save watch folder config {config.watch_id}: {e}")
            return False

    async def get_watch_folder_config(
        self, watch_id: str
    ) -> Optional[WatchFolderConfig]:
        """Get watch folder configuration."""
        try:
            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT watch_id, path, collection, patterns, ignore_patterns, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, enabled,
                           created_at, updated_at, last_scan, metadata
                    FROM watch_folders
                    WHERE watch_id = ?
                    """,
                    (watch_id,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return WatchFolderConfig(
                    watch_id=row["watch_id"],
                    path=row["path"],
                    collection=row["collection"],
                    patterns=self._deserialize_json(row["patterns"]) or [],
                    ignore_patterns=self._deserialize_json(row["ignore_patterns"])
                    or [],
                    auto_ingest=bool(row["auto_ingest"]),
                    recursive=bool(row["recursive"]),
                    recursive_depth=row["recursive_depth"],
                    debounce_seconds=row["debounce_seconds"],
                    enabled=bool(row["enabled"]),
                    created_at=datetime.fromisoformat(
                        row["created_at"].replace("Z", "+00:00")
                    ),
                    updated_at=datetime.fromisoformat(
                        row["updated_at"].replace("Z", "+00:00")
                    ),
                    last_scan=datetime.fromisoformat(
                        row["last_scan"].replace("Z", "+00:00")
                    )
                    if row["last_scan"]
                    else None,
                    metadata=self._deserialize_json(row["metadata"]),
                )

        except Exception as e:
            logger.error(f"Failed to get watch folder config {watch_id}: {e}")
            return None

    async def get_all_watch_folder_configs(
        self, enabled_only: bool = True
    ) -> List[WatchFolderConfig]:
        """Get all watch folder configurations."""
        try:
            with self._lock:
                sql = """
                    SELECT watch_id, path, collection, patterns, ignore_patterns, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, enabled,
                           created_at, updated_at, last_scan, metadata
                    FROM watch_folders
                """

                if enabled_only:
                    sql += " WHERE enabled = 1"

                sql += " ORDER BY created_at ASC"

                cursor = self.connection.execute(sql)
                rows = cursor.fetchall()

                configs = []
                for row in rows:
                    configs.append(
                        WatchFolderConfig(
                            watch_id=row["watch_id"],
                            path=row["path"],
                            collection=row["collection"],
                            patterns=self._deserialize_json(row["patterns"]) or [],
                            ignore_patterns=self._deserialize_json(
                                row["ignore_patterns"]
                            )
                            or [],
                            auto_ingest=bool(row["auto_ingest"]),
                            recursive=bool(row["recursive"]),
                            recursive_depth=row["recursive_depth"],
                            debounce_seconds=row["debounce_seconds"],
                            enabled=bool(row["enabled"]),
                            created_at=datetime.fromisoformat(
                                row["created_at"].replace("Z", "+00:00")
                            ),
                            updated_at=datetime.fromisoformat(
                                row["updated_at"].replace("Z", "+00:00")
                            ),
                            last_scan=datetime.fromisoformat(
                                row["last_scan"].replace("Z", "+00:00")
                            )
                            if row["last_scan"]
                            else None,
                            metadata=self._deserialize_json(row["metadata"]),
                        )
                    )

                return configs

        except Exception as e:
            logger.error(f"Failed to get all watch folder configs: {e}")
            return []

    async def remove_watch_folder_config(self, watch_id: str) -> bool:
        """Remove watch folder configuration."""
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM watch_folders WHERE watch_id = ?", (watch_id,)
                )

                success = cursor.rowcount > 0
                if success:
                    logger.debug(f"Removed watch folder config: {watch_id}")
                else:
                    logger.warning(f"Watch folder config not found: {watch_id}")

                return success

        except Exception as e:
            logger.error(f"Failed to remove watch folder config {watch_id}: {e}")
            return False

    async def update_watch_folder_scan_time(self, watch_id: str) -> bool:
        """Update the last scan time for a watch folder."""
        try:
            with self._lock:
                cursor = self.connection.execute(
                    "UPDATE watch_folders SET last_scan = CURRENT_TIMESTAMP WHERE watch_id = ?",
                    (watch_id,),
                )

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to update scan time for watch folder {watch_id}: {e}")
            return False

    # Processing Queue Management

    async def add_to_processing_queue(
        self,
        file_path: str,
        collection: str,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        scheduled_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add file to processing queue with priority."""
        try:
            queue_id = f"queue_{int(time.time() * 1000000)}_{hash(file_path) % 100000}"
            scheduled_at = scheduled_at or datetime.now(timezone.utc)

            async with self.transaction() as conn:
                # Ensure file processing record exists
                conn.execute(
                    """
                    INSERT OR IGNORE INTO file_processing 
                    (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        file_path,
                        collection,
                        FileProcessingStatus.PENDING.value,
                        priority.value,
                    ),
                )

                # Add to queue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO processing_queue 
                    (queue_id, file_path, collection, priority, scheduled_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        queue_id,
                        file_path,
                        collection,
                        priority.value,
                        scheduled_at.isoformat(),
                        self._serialize_json(metadata),
                    ),
                )

            logger.debug(
                f"Added to processing queue: {file_path} (queue_id: {queue_id})"
            )
            return queue_id

        except Exception as e:
            logger.error(f"Failed to add to processing queue {file_path}: {e}")
            raise

    async def get_next_queue_item(
        self, collection: Optional[str] = None, max_attempts: int = 3
    ) -> Optional[ProcessingQueueItem]:
        """Get next item from processing queue ordered by priority."""
        try:
            with self._lock:
                sql = """
                    SELECT queue_id, file_path, collection, priority, created_at, 
                           scheduled_at, attempts, metadata
                    FROM processing_queue
                    WHERE attempts < ? AND scheduled_at <= CURRENT_TIMESTAMP
                """

                params = [max_attempts]

                if collection:
                    sql += " AND collection = ?"
                    params.append(collection)

                sql += " ORDER BY priority DESC, scheduled_at ASC LIMIT 1"

                cursor = self.connection.execute(sql, params)
                row = cursor.fetchone()

                if not row:
                    return None

                return ProcessingQueueItem(
                    queue_id=row["queue_id"],
                    file_path=row["file_path"],
                    collection=row["collection"],
                    priority=ProcessingPriority(row["priority"]),
                    created_at=datetime.fromisoformat(
                        row["created_at"].replace("Z", "+00:00")
                    ),
                    scheduled_at=datetime.fromisoformat(
                        row["scheduled_at"].replace("Z", "+00:00")
                    ),
                    attempts=row["attempts"],
                    metadata=self._deserialize_json(row["metadata"]),
                )

        except Exception as e:
            logger.error(f"Failed to get next queue item: {e}")
            return None

    async def mark_queue_item_processing(self, queue_id: str) -> bool:
        """Mark a queue item as being processed."""
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "UPDATE processing_queue SET attempts = attempts + 1 WHERE queue_id = ?",
                    (queue_id,),
                )

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to mark queue item processing {queue_id}: {e}")
            return False

    async def remove_from_processing_queue(self, queue_id: str) -> bool:
        """Remove item from processing queue."""
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM processing_queue WHERE queue_id = ?", (queue_id,)
                )

                success = cursor.rowcount > 0
                if success:
                    logger.debug(f"Removed from processing queue: {queue_id}")

                return success

        except Exception as e:
            logger.error(f"Failed to remove from processing queue {queue_id}: {e}")
            return False

    async def reschedule_queue_item(
        self, queue_id: str, delay_seconds: int = 300, max_attempts: int = 3
    ) -> bool:
        """Reschedule a queue item for later processing."""
        try:
            new_scheduled_at = datetime.now(timezone.utc) + timedelta(
                seconds=delay_seconds
            )

            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE processing_queue 
                    SET scheduled_at = ?, attempts = CASE 
                        WHEN attempts >= ? THEN attempts 
                        ELSE attempts
                    END
                    WHERE queue_id = ? AND attempts < ?
                    """,
                    (
                        new_scheduled_at.isoformat(),
                        max_attempts,
                        queue_id,
                        max_attempts,
                    ),
                )

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to reschedule queue item {queue_id}: {e}")
            return False

    async def get_queue_stats(self, collection: Optional[str] = None) -> Dict[str, int]:
        """Get processing queue statistics."""
        try:
            with self._lock:
                base_sql = "SELECT priority, COUNT(*) as count FROM processing_queue"
                params = []

                if collection:
                    base_sql += " WHERE collection = ?"
                    params.append(collection)

                base_sql += " GROUP BY priority"

                cursor = self.connection.execute(base_sql, params)
                rows = cursor.fetchall()

                stats = {"total": 0, "low": 0, "normal": 0, "high": 0, "urgent": 0}

                priority_names = {
                    ProcessingPriority.LOW.value: "low",
                    ProcessingPriority.NORMAL.value: "normal",
                    ProcessingPriority.HIGH.value: "high",
                    ProcessingPriority.URGENT.value: "urgent",
                }

                for row in rows:
                    priority = row["priority"]
                    count = row["count"]
                    priority_name = priority_names.get(priority, "unknown")
                    if priority_name != "unknown":
                        stats[priority_name] = count
                        stats["total"] += count

                return stats

        except Exception as e:
            logger.error(f"Failed to get queue stats: {e}")
            return {"total": 0, "low": 0, "normal": 0, "high": 0, "urgent": 0}

    async def clear_queue(self, collection: Optional[str] = None) -> int:
        """Clear processing queue items."""
        try:
            async with self.transaction() as conn:
                if collection:
                    cursor = conn.execute(
                        "DELETE FROM processing_queue WHERE collection = ?",
                        (collection,),
                    )
                else:
                    cursor = conn.execute("DELETE FROM processing_queue")

                cleared_count = cursor.rowcount
                logger.info(f"Cleared {cleared_count} items from processing queue")
                return cleared_count

        except Exception as e:
            logger.error(f"Failed to clear processing queue: {e}")
            return 0

    # System State Management

    async def set_system_state(self, key: str, value: Any) -> bool:
        """Set system state value."""
        try:
            serialized_value = (
                self._serialize_json(value) if not isinstance(value, str) else value
            )

            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO system_state (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (key, serialized_value),
                )

            return True

        except Exception as e:
            logger.error(f"Failed to set system state {key}: {e}")
            return False

    async def get_system_state(self, key: str, default: Any = None) -> Any:
        """Get system state value."""
        try:
            with self._lock:
                cursor = self.connection.execute(
                    "SELECT value FROM system_state WHERE key = ?", (key,)
                )

                row = cursor.fetchone()
                if not row:
                    return default

                value = row["value"]

                # Try to deserialize JSON, fall back to string value
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value

        except Exception as e:
            logger.error(f"Failed to get system state {key}: {e}")
            return default

    async def delete_system_state(self, key: str) -> bool:
        """Delete system state value."""
        try:
            async with self.transaction() as conn:
                cursor = conn.execute("DELETE FROM system_state WHERE key = ?", (key,))

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to delete system state {key}: {e}")
            return False

    # Analytics and Reporting

    async def get_processing_stats(
        self, collection: Optional[str] = None, days: int = 7
    ) -> Dict[str, Any]:
        """Get processing statistics for the specified time period."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

            with self._lock:
                # File processing stats
                sql = """
                    SELECT status, COUNT(*) as count
                    FROM file_processing
                    WHERE updated_at >= ?
                """
                params = [cutoff_date.isoformat()]

                if collection:
                    sql += " AND collection = ?"
                    params.append(collection)

                sql += " GROUP BY status"

                cursor = self.connection.execute(sql, params)
                status_stats = {
                    row["status"]: row["count"] for row in cursor.fetchall()
                }

                # Processing history stats
                sql = """
                    SELECT 
                        COUNT(*) as total_processed,
                        AVG(processing_time_ms) as avg_processing_time,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                    FROM processing_history
                    WHERE created_at >= ?
                """
                params = [cutoff_date.isoformat()]

                if collection:
                    sql += " AND collection = ?"
                    params.append(collection)

                cursor = self.connection.execute(sql, params)
                history_stats = cursor.fetchone()

                return {
                    "period_days": days,
                    "collection": collection,
                    "file_status_counts": status_stats,
                    "total_processed": history_stats["total_processed"] or 0,
                    "successful": history_stats["successful"] or 0,
                    "failed": history_stats["failed"] or 0,
                    "success_rate": (
                        (history_stats["successful"] or 0)
                        / max(1, history_stats["total_processed"] or 1)
                    ),
                    "avg_processing_time_ms": history_stats["avg_processing_time"] or 0,
                    "queue_stats": await self.get_queue_stats(collection),
                }

        except Exception as e:
            logger.error(f"Failed to get processing stats: {e}")
            return {
                "period_days": days,
                "collection": collection,
                "file_status_counts": {},
                "total_processed": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_processing_time_ms": 0.0,
                "queue_stats": {
                    "total": 0,
                    "low": 0,
                    "normal": 0,
                    "high": 0,
                    "urgent": 0,
                },
            }

    async def get_failed_files(
        self, collection: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of failed files with error details."""
        try:
            with self._lock:
                sql = """
                    SELECT file_path, collection, error_message, retry_count, max_retries,
                           updated_at, file_size
                    FROM file_processing
                    WHERE status = 'failed'
                """
                params = []

                if collection:
                    sql += " AND collection = ?"
                    params.append(collection)

                sql += " ORDER BY updated_at DESC LIMIT ?"
                params.append(limit)

                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()

                failed_files = []
                for row in rows:
                    failed_files.append(
                        {
                            "file_path": row["file_path"],
                            "collection": row["collection"],
                            "error_message": row["error_message"],
                            "retry_count": row["retry_count"],
                            "max_retries": row["max_retries"],
                            "failed_at": row["updated_at"],
                            "file_size": row["file_size"],
                        }
                    )

                return failed_files

        except Exception as e:
            logger.error(f"Failed to get failed files: {e}")
            return []

    # Cleanup and Maintenance

    async def cleanup_old_records(self, days: int = 30) -> Dict[str, int]:
        """Clean up old processing records."""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            cleanup_counts = {}

            async with self.transaction() as conn:
                # Clean up completed file processing records
                cursor = conn.execute(
                    """
                    DELETE FROM file_processing 
                    WHERE status IN ('completed', 'skipped') AND updated_at < ?
                    """,
                    (cutoff_date.isoformat(),),
                )
                cleanup_counts["file_processing"] = cursor.rowcount

                # Clean up old processing history
                cursor = conn.execute(
                    "DELETE FROM processing_history WHERE created_at < ?",
                    (cutoff_date.isoformat(),),
                )
                cleanup_counts["processing_history"] = cursor.rowcount

                # Clean up orphaned queue items
                cursor = conn.execute(
                    """
                    DELETE FROM processing_queue 
                    WHERE file_path NOT IN (SELECT file_path FROM file_processing)
                    """
                )
                cleanup_counts["orphaned_queue_items"] = cursor.rowcount

            total_cleaned = sum(cleanup_counts.values())
            if total_cleaned > 0:
                logger.info(f"Cleaned up {total_cleaned} old records: {cleanup_counts}")

            return cleanup_counts

        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return {}

    async def vacuum_database(self) -> bool:
        """Perform database vacuum to reclaim space."""
        try:
            logger.info("Performing database vacuum")

            with self._lock:
                # Close and reopen connection for VACUUM (required)
                if self.connection:
                    self.connection.close()

                temp_connection = sqlite3.connect(str(self.db_path), timeout=30.0)
                temp_connection.execute("VACUUM")
                temp_connection.close()

                # Reopen with original settings
                self.connection = sqlite3.connect(
                    str(self.db_path),
                    timeout=30.0,
                    check_same_thread=False,
                    isolation_level=None,
                )

                # Restore settings
                self.connection.execute("PRAGMA journal_mode=WAL")
                self.connection.execute("PRAGMA synchronous=NORMAL")
                self.connection.execute("PRAGMA cache_size=10000")
                self.connection.execute("PRAGMA temp_store=MEMORY")
                self.connection.execute("PRAGMA mmap_size=268435456")
                self.connection.execute("PRAGMA wal_autocheckpoint=1000")
                self.connection.execute("PRAGMA foreign_keys=ON")
                self.connection.row_factory = sqlite3.Row

            logger.info("Database vacuum completed")
            return True

        except Exception as e:
            logger.error(f"Failed to vacuum database: {e}")
            return False

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database size and statistics."""
        try:
            with self._lock:
                # Get table counts
                tables = {
                    "file_processing": "SELECT COUNT(*) FROM file_processing",
                    "watch_folders": "SELECT COUNT(*) FROM watch_folders",
                    "processing_queue": "SELECT COUNT(*) FROM processing_queue",
                    "processing_history": "SELECT COUNT(*) FROM processing_history",
                    "system_state": "SELECT COUNT(*) FROM system_state",
                }

                table_counts = {}
                for table, sql in tables.items():
                    cursor = self.connection.execute(sql)
                    table_counts[table] = cursor.fetchone()[0]

                # Get database size info
                cursor = self.connection.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]

                cursor = self.connection.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]

                total_size_bytes = page_count * page_size

                # Get WAL info
                cursor = self.connection.execute("PRAGMA wal_checkpoint")
                wal_info = cursor.fetchone()

                return {
                    "database_file": str(self.db_path),
                    "total_size_bytes": total_size_bytes,
                    "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
                    "page_count": page_count,
                    "page_size": page_size,
                    "wal_mode": True,
                    "wal_pages": wal_info[1] if wal_info else 0,
                    "table_counts": table_counts,
                    "schema_version": self.SCHEMA_VERSION,
                }

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}


# Graceful shutdown handler
class StateManagerShutdownHandler:
    """Handles graceful shutdown of state manager."""

    def __init__(self, state_manager: SQLiteStateManager):
        self.state_manager = state_manager
        self._shutdown_handlers: List[Callable[[], None]] = []

    def add_shutdown_handler(self, handler: Callable[[], None]):
        """Add a shutdown handler."""
        self._shutdown_handlers.append(handler)

    async def shutdown(self):
        """Perform graceful shutdown."""
        logger.info("Starting graceful shutdown of state manager")

        # Run custom shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler()
                else:
                    handler()
            except Exception as e:
                logger.error(f"Error in shutdown handler: {e}")

        # Close state manager
        await self.state_manager.close()
        logger.info("State manager shutdown completed")


# Global state manager instance
_state_manager: Optional[SQLiteStateManager] = None
_shutdown_handler: Optional[StateManagerShutdownHandler] = None


async def get_state_manager(db_path: str = "workspace_state.db") -> SQLiteStateManager:
    """Get or create global state manager instance."""
    global _state_manager, _shutdown_handler

    if _state_manager is None:
        _state_manager = SQLiteStateManager(db_path)

        if not await _state_manager.initialize():
            raise RuntimeError("Failed to initialize SQLite state manager")

        # Set up shutdown handler
        _shutdown_handler = StateManagerShutdownHandler(_state_manager)

        # Register with signal handlers
        import atexit

        atexit.register(lambda: asyncio.create_task(_shutdown_handler.shutdown()))

    return _state_manager


async def shutdown_state_manager():
    """Shutdown global state manager."""
    global _state_manager, _shutdown_handler

    if _shutdown_handler:
        await _shutdown_handler.shutdown()
        _shutdown_handler = None
        _state_manager = None
