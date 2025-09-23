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
# Use unified logging system to prevent console interference in MCP mode
from loguru import logger
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..utils.os_directories import OSDirectories

# logger imported from loguru


class FileProcessingStatus(Enum):
    """File processing status enumeration."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"
    OCR_REQUIRED = "ocr_required"  # Added for PDF processing with OCR detection


class ProcessingPriority(Enum):
    """Processing priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class LSPServerStatus(Enum):
    """LSP Server status enumeration."""

    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    ERROR = "error"
    UNAVAILABLE = "unavailable"


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
    document_id: Optional[str] = None  # For multi-component testing
    # LSP-specific fields
    language_id: Optional[str] = None
    lsp_extracted: bool = False
    symbols_count: int = 0
    lsp_server_id: Optional[int] = None
    last_lsp_analysis: Optional[datetime] = None
    lsp_metadata: Optional[Dict[str, Any]] = None

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


@dataclass
class ProjectRecord:
    """Record for tracking LSP-enabled projects."""

    id: Optional[int]
    name: str
    root_path: str
    collection_name: str
    lsp_enabled: bool = False
    last_scan: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class LSPServerRecord:
    """Record for tracking LSP servers."""

    id: Optional[int]
    language: str
    server_path: str
    version: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    status: LSPServerStatus = LSPServerStatus.INACTIVE
    last_health_check: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


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

    SCHEMA_VERSION = 3  # Updated for LSP table additions and file metadata extensions
    WAL_CHECKPOINT_INTERVAL = 300  # 5 minutes
    MAINTENANCE_INTERVAL = 3600  # 1 hour

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite state manager with OS-standard state directory.

        Args:
            db_path: Optional custom path to SQLite database file.
                    If None, uses OS-standard state directory with default filename.
                    For backward compatibility only - prefer OS-standard location.
        """
        if db_path is None:
            # Use OS-standard state directory
            os_dirs = OSDirectories()
            os_dirs.ensure_directories()
            self.db_path = os_dirs.get_state_file("workspace_state.db")
            logger.info(f"Using OS-standard state directory: {self.db_path}")
        else:
            # Legacy mode: use custom path (for backward compatibility)
            self.db_path = Path(db_path)
            logger.warning(f"Using legacy database path: {self.db_path}. Consider migrating to OS-standard location.")

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
                document_id TEXT,
                metadata TEXT,  -- JSON
                -- LSP-specific fields (Schema Version 3+)
                language_id TEXT,
                lsp_extracted BOOLEAN NOT NULL DEFAULT 0,
                symbols_count INTEGER DEFAULT 0,
                lsp_server_id INTEGER,
                last_lsp_analysis TIMESTAMP,
                lsp_metadata TEXT,  -- JSON for LSP-specific data
                FOREIGN KEY (lsp_server_id) REFERENCES lsp_servers (id) ON DELETE SET NULL
            )
            """,
            # Indexes for file_processing
            "CREATE INDEX idx_file_processing_status ON file_processing(status)",
            "CREATE INDEX idx_file_processing_collection ON file_processing(collection)",
            "CREATE INDEX idx_file_processing_updated_at ON file_processing(updated_at)",
            "CREATE INDEX idx_file_processing_priority ON file_processing(priority)",
            # LSP-specific indexes
            "CREATE INDEX idx_file_processing_language_id ON file_processing(language_id)",
            "CREATE INDEX idx_file_processing_lsp_extracted ON file_processing(lsp_extracted)",
            "CREATE INDEX idx_file_processing_lsp_server_id ON file_processing(lsp_server_id)",
            "CREATE INDEX idx_file_processing_last_lsp_analysis ON file_processing(last_lsp_analysis)",
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
            # Multi-component integration tables
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                file_path TEXT,
                component TEXT,
                data TEXT,  -- JSON
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX idx_events_type ON events(event_type)",
            "CREATE INDEX idx_events_file_path ON events(file_path)",
            "CREATE INDEX idx_events_component ON events(component)",
            "CREATE INDEX idx_events_timestamp ON events(timestamp)",
            """
            CREATE TABLE search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                results_count INTEGER,
                source TEXT,
                response_time_ms INTEGER,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON
            )
            """,
            "CREATE INDEX idx_search_history_query ON search_history(query)",
            "CREATE INDEX idx_search_history_source ON search_history(source)",
            "CREATE INDEX idx_search_history_timestamp ON search_history(timestamp)",
            """
            CREATE TABLE memory_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT UNIQUE NOT NULL,
                rule_data TEXT NOT NULL,  -- JSON
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX idx_memory_rules_rule_id ON memory_rules(rule_id)",
            """
            CREATE TABLE configuration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_data TEXT NOT NULL,  -- JSON
                source TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX idx_config_history_timestamp ON configuration_history(timestamp)",
            "CREATE INDEX idx_config_history_source ON configuration_history(source)",
            """
            CREATE TABLE error_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                source TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON
            )
            """,
            "CREATE INDEX idx_error_log_type ON error_log(error_type)",
            "CREATE INDEX idx_error_log_source ON error_log(source)",
            "CREATE INDEX idx_error_log_timestamp ON error_log(timestamp)",
            """
            CREATE TABLE performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation TEXT NOT NULL,
                metric_data TEXT NOT NULL,  -- JSON
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX idx_performance_metrics_operation ON performance_metrics(operation)",
            "CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp)",
            """
            CREATE TABLE resource_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usage_data TEXT NOT NULL,  -- JSON
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "CREATE INDEX idx_resource_usage_timestamp ON resource_usage(timestamp)",
            # LSP Integration Tables (Schema Version 2+)
            """
            CREATE TABLE projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                root_path TEXT NOT NULL UNIQUE,
                collection_name TEXT NOT NULL,
                lsp_enabled BOOLEAN NOT NULL DEFAULT 0,
                last_scan TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON for additional project configuration
            )
            """,
            # Indexes for projects
            "CREATE INDEX idx_projects_name ON projects(name)",
            "CREATE INDEX idx_projects_root_path ON projects(root_path)",
            "CREATE INDEX idx_projects_collection_name ON projects(collection_name)",
            "CREATE INDEX idx_projects_lsp_enabled ON projects(lsp_enabled)",
            "CREATE INDEX idx_projects_last_scan ON projects(last_scan)",
            # LSP Servers table
            """
            CREATE TABLE lsp_servers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language TEXT NOT NULL,
                server_path TEXT NOT NULL,
                version TEXT,
                capabilities TEXT,  -- JSON for LSP server capabilities
                status TEXT NOT NULL DEFAULT 'inactive',
                last_health_check TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,  -- JSON for additional server configuration
                UNIQUE(language, server_path)
            )
            """,
            # Indexes for lsp_servers
            "CREATE INDEX idx_lsp_servers_language ON lsp_servers(language)",
            "CREATE INDEX idx_lsp_servers_status ON lsp_servers(status)",
            "CREATE INDEX idx_lsp_servers_last_health_check ON lsp_servers(last_health_check)",
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

        with self.connection:
            # Migrate from version 1 to version 2 - Add LSP tables
            if from_version == 1 and to_version >= 2:
                migration_sql = [
                    # Add projects table for LSP integration
                    """
                    CREATE TABLE projects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        root_path TEXT NOT NULL UNIQUE,
                        collection_name TEXT NOT NULL,
                        lsp_enabled BOOLEAN NOT NULL DEFAULT 0,
                        last_scan TIMESTAMP,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT  -- JSON for additional project configuration
                    )
                    """,
                    # Add indexes for projects table
                    "CREATE INDEX idx_projects_name ON projects(name)",
                    "CREATE INDEX idx_projects_root_path ON projects(root_path)", 
                    "CREATE INDEX idx_projects_collection_name ON projects(collection_name)",
                    "CREATE INDEX idx_projects_lsp_enabled ON projects(lsp_enabled)",
                    "CREATE INDEX idx_projects_last_scan ON projects(last_scan)",
                    # Add lsp_servers table for LSP server management
                    """
                    CREATE TABLE lsp_servers (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        language TEXT NOT NULL,
                        server_path TEXT NOT NULL,
                        version TEXT,
                        capabilities TEXT,  -- JSON for LSP server capabilities
                        status TEXT NOT NULL DEFAULT 'inactive',
                        last_health_check TIMESTAMP,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,  -- JSON for additional server configuration
                        UNIQUE(language, server_path)
                    )
                    """,
                    # Add indexes for lsp_servers table
                    "CREATE INDEX idx_lsp_servers_language ON lsp_servers(language)",
                    "CREATE INDEX idx_lsp_servers_status ON lsp_servers(status)",
                    "CREATE INDEX idx_lsp_servers_last_health_check ON lsp_servers(last_health_check)",
                ]
                
                for sql in migration_sql:
                    self.connection.execute(sql)
                
                logger.info("Successfully migrated to schema version 2 (added projects and lsp_servers tables)")

            # Migrate from version 2 to version 3 - Add LSP fields to file_processing
            if from_version <= 2 and to_version >= 3:
                lsp_fields_sql = [
                    # Add LSP-specific columns to file_processing table
                    "ALTER TABLE file_processing ADD COLUMN language_id TEXT",
                    "ALTER TABLE file_processing ADD COLUMN lsp_extracted BOOLEAN NOT NULL DEFAULT 0",
                    "ALTER TABLE file_processing ADD COLUMN symbols_count INTEGER DEFAULT 0",
                    "ALTER TABLE file_processing ADD COLUMN lsp_server_id INTEGER",
                    "ALTER TABLE file_processing ADD COLUMN last_lsp_analysis TIMESTAMP",
                    "ALTER TABLE file_processing ADD COLUMN lsp_metadata TEXT",
                    # Add indexes for new LSP fields
                    "CREATE INDEX idx_file_processing_language_id ON file_processing(language_id)",
                    "CREATE INDEX idx_file_processing_lsp_extracted ON file_processing(lsp_extracted)",
                    "CREATE INDEX idx_file_processing_lsp_server_id ON file_processing(lsp_server_id)",
                    "CREATE INDEX idx_file_processing_last_lsp_analysis ON file_processing(last_lsp_analysis)",
                ]
                
                for sql in lsp_fields_sql:
                    self.connection.execute(sql)
                
                logger.info("Successfully migrated to schema version 3 (added LSP fields to file_processing)")

            # Record the migration
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

    # Multi-Component Communication Support Methods

    async def update_processing_state(
        self,
        file_path: str,
        status: str,
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update processing state for multi-component testing."""
        try:
            async with self.transaction() as conn:
                if status == "processing":
                    # Start processing
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO file_processing 
                        (file_path, collection, status, started_at, updated_at, metadata)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                        """,
                        (
                            file_path,
                            collection_name or "default",
                            status,
                            self._serialize_json(metadata),
                        ),
                    )
                else:
                    # Update existing record
                    update_fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
                    params = [status]
                    
                    if document_id:
                        update_fields.append("document_id = ?")
                        params.append(document_id)
                    
                    if metadata:
                        update_fields.append("metadata = ?")
                        params.append(self._serialize_json(metadata))
                    
                    if status in ["completed", "failed"]:
                        update_fields.append("completed_at = CURRENT_TIMESTAMP")
                    
                    params.append(file_path)
                    
                    conn.execute(
                        f"UPDATE file_processing SET {', '.join(update_fields)} WHERE file_path = ?",
                        params,
                    )

            return True
        except Exception as e:
            logger.error(f"Failed to update processing state {file_path}: {e}")
            return False

    async def get_processing_states(
        self, filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get processing states with optional filtering."""
        try:
            with self._lock:
                sql = """
                    SELECT file_path, collection, status, document_id, metadata, 
                           created_at, updated_at, completed_at
                    FROM file_processing
                """
                
                params = []
                if filter_params:
                    conditions = []
                    for key, value in filter_params.items():
                        conditions.append(f"{key} = ?")
                        params.append(value)
                    
                    if conditions:
                        sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY updated_at DESC"
                
                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "file_path": row["file_path"],
                        "collection": row["collection"],
                        "status": row["status"],
                        "document_id": row["document_id"],
                        "metadata": self._deserialize_json(row["metadata"]) or {},
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "completed_at": row["completed_at"],
                    })

                return results
        except Exception as e:
            logger.error(f"Failed to get processing states: {e}")
            return []

    async def record_search_operation(
        self,
        query: str,
        results_count: int,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record search operation for multi-component tracking."""
        try:
            async with self.transaction() as conn:
                response_time_ms = metadata.get("response_time_ms") if metadata else None
                
                conn.execute(
                    """
                    INSERT INTO search_history 
                    (query, results_count, source, response_time_ms, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        query,
                        results_count,
                        source,
                        response_time_ms,
                        self._serialize_json(metadata),
                    ),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to record search operation: {e}")
            return False

    async def get_search_history(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get search history."""
        try:
            with self._lock:
                sql = """
                    SELECT query, results_count, source, response_time_ms, 
                           timestamp, metadata
                    FROM search_history
                    ORDER BY timestamp DESC
                """
                
                params = []
                if limit:
                    sql += " LIMIT ?"
                    params.append(limit)
                
                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "query": row["query"],
                        "results_count": row["results_count"],
                        "source": row["source"],
                        "response_time_ms": row["response_time_ms"],
                        "timestamp": row["timestamp"],
                        "metadata": self._deserialize_json(row["metadata"]) or {},
                    })

                return results
        except Exception as e:
            logger.error(f"Failed to get search history: {e}")
            return []

    async def store_memory_rule(
        self, rule_id: str, rule_data: Dict[str, Any]
    ) -> bool:
        """Store memory rule."""
        try:
            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memory_rules 
                    (rule_id, rule_data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                    (rule_id, self._serialize_json(rule_data)),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to store memory rule {rule_id}: {e}")
            return False

    async def get_memory_rules(self) -> List[Dict[str, Any]]:
        """Get all memory rules."""
        try:
            with self._lock:
                cursor = self.connection.execute(
                    "SELECT rule_id, rule_data, created_at, updated_at FROM memory_rules ORDER BY created_at"
                )
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "rule_id": row["rule_id"],
                        "rule_data": self._deserialize_json(row["rule_data"]) or {},
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    })

                return results
        except Exception as e:
            logger.error(f"Failed to get memory rules: {e}")
            return []

    async def record_event(self, event: Dict[str, Any]) -> bool:
        """Record event for multi-component tracking."""
        try:
            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO events 
                    (event_type, file_path, component, data, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        event.get("type"),
                        event.get("file_path"),
                        event.get("component"),
                        self._serialize_json(event),
                        event.get("timestamp", time.time()),
                    ),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to record event: {e}")
            return False

    async def get_events(
        self, filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get events with optional filtering."""
        try:
            with self._lock:
                sql = "SELECT event_type, file_path, component, data, timestamp FROM events"
                
                params = []
                if filter_params:
                    conditions = []
                    for key, value in filter_params.items():
                        if key in ["event_type", "file_path", "component"]:
                            conditions.append(f"{key} = ?")
                            params.append(value)
                        elif key == "type":  # Handle 'type' -> 'event_type' mapping
                            conditions.append("event_type = ?")
                            params.append(value)
                    
                    if conditions:
                        sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY timestamp DESC"
                
                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    event_data = self._deserialize_json(row["data"]) or {}
                    event_data.update({
                        "type": row["event_type"],
                        "file_path": row["file_path"],
                        "component": row["component"],
                        "timestamp": row["timestamp"],
                    })
                    results.append(event_data)

                return results
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []

    async def record_configuration_change(
        self, config_data: Dict[str, Any], source: str, timestamp: Optional[float] = None
    ) -> bool:
        """Record configuration change."""
        try:
            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO configuration_history 
                    (config_data, source, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (
                        self._serialize_json(config_data),
                        source,
                        timestamp or time.time(),
                    ),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to record configuration change: {e}")
            return False

    async def get_configuration_history(self) -> List[Dict[str, Any]]:
        """Get configuration history."""
        try:
            with self._lock:
                cursor = self.connection.execute(
                    "SELECT config_data, source, timestamp FROM configuration_history ORDER BY timestamp DESC"
                )
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "config_data": self._deserialize_json(row["config_data"]) or {},
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                    })

                return results
        except Exception as e:
            logger.error(f"Failed to get configuration history: {e}")
            return []

    async def record_error(
        self,
        error_type: str,
        error_message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record error."""
        try:
            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO error_log 
                    (error_type, error_message, source, metadata)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        error_type,
                        error_message,
                        source,
                        self._serialize_json(metadata),
                    ),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to record error: {e}")
            return False

    async def get_errors(
        self, filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get errors with optional filtering."""
        try:
            with self._lock:
                sql = "SELECT error_type, error_message, source, timestamp, metadata FROM error_log"
                
                params = []
                if filter_params:
                    conditions = []
                    for key, value in filter_params.items():
                        conditions.append(f"{key} = ?")
                        params.append(value)
                    
                    if conditions:
                        sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY timestamp DESC"
                
                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append({
                        "error_type": row["error_type"],
                        "error_message": row["error_message"],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "metadata": self._deserialize_json(row["metadata"]) or {},
                    })

                return results
        except Exception as e:
            logger.error(f"Failed to get errors: {e}")
            return []

    async def record_performance_metric(self, metric: Dict[str, Any]) -> bool:
        """Record performance metric."""
        try:
            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO performance_metrics 
                    (operation, metric_data, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (
                        metric.get("operation"),
                        self._serialize_json(metric),
                        metric.get("timestamp", time.time()),
                    ),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}")
            return False

    async def get_performance_metrics(
        self, filter_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Get performance metrics with optional filtering."""
        try:
            with self._lock:
                sql = "SELECT operation, metric_data, timestamp FROM performance_metrics"
                
                params = []
                if filter_params:
                    conditions = []
                    for key, value in filter_params.items():
                        conditions.append(f"{key} = ?")
                        params.append(value)
                    
                    if conditions:
                        sql += " WHERE " + " AND ".join(conditions)
                
                sql += " ORDER BY timestamp DESC"
                
                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    metric_data = self._deserialize_json(row["metric_data"]) or {}
                    metric_data.update({
                        "operation": row["operation"],
                        "timestamp": row["timestamp"],
                    })
                    results.append(metric_data)

                return results
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return []

    async def record_resource_usage(self, usage_data: Dict[str, Any]) -> bool:
        """Record resource usage."""
        try:
            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO resource_usage 
                    (usage_data, timestamp)
                    VALUES (?, ?)
                    """,
                    (
                        self._serialize_json(usage_data),
                        usage_data.get("timestamp", time.time()),
                    ),
                )

            return True
        except Exception as e:
            logger.error(f"Failed to record resource usage: {e}")
            return False

    async def get_resource_usage_history(self) -> List[Dict[str, Any]]:
        """Get resource usage history."""
        try:
            with self._lock:
                cursor = self.connection.execute(
                    "SELECT usage_data, timestamp FROM resource_usage ORDER BY timestamp DESC"
                )
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    usage_data = self._deserialize_json(row["usage_data"]) or {}
                    usage_data["timestamp"] = row["timestamp"]
                    results.append(usage_data)

                return results
        except Exception as e:
            logger.error(f"Failed to get resource usage history: {e}")
            return []

    # Original methods continue here...
    
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
                           file_size, file_hash, document_id, metadata
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
                    document_id=row["document_id"],
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
                           file_size, file_hash, document_id, metadata
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
                            document_id=row["document_id"],
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
                    "events": "SELECT COUNT(*) FROM events",
                    "search_history": "SELECT COUNT(*) FROM search_history",
                    "memory_rules": "SELECT COUNT(*) FROM memory_rules",
                    "configuration_history": "SELECT COUNT(*) FROM configuration_history",
                    "error_log": "SELECT COUNT(*) FROM error_log",
                    "performance_metrics": "SELECT COUNT(*) FROM performance_metrics",
                    "resource_usage": "SELECT COUNT(*) FROM resource_usage",
                }

                table_counts = {}
                for table, sql in tables.items():
                    try:
                        cursor = self.connection.execute(sql)
                        table_counts[table] = cursor.fetchone()[0]
                    except sqlite3.Error as e:
                        table_counts[table] = f"Error: {e}"

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

    # LSP Integration - Project Management Methods
    
    async def create_project(self, project: ProjectRecord) -> Optional[int]:
        """
        Create a new project record.
        
        Args:
            project: ProjectRecord to create
            
        Returns:
            Project ID if created successfully, None otherwise
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO projects (name, root_path, collection_name, lsp_enabled, 
                                        last_scan, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        project.name,
                        project.root_path,
                        project.collection_name,
                        project.lsp_enabled,
                        project.last_scan,
                        json.dumps(project.metadata) if project.metadata else None,
                        project.created_at,
                        project.updated_at,
                    ),
                )
                
                project_id = cursor.lastrowid
                logger.info(f"Created project: {project.name} with ID {project_id}")
                return project_id
                
        except Exception as e:
            logger.error(f"Failed to create project {project.name}: {e}")
            return None

    async def get_project(self, project_id: int) -> Optional[ProjectRecord]:
        """
        Get project by ID.
        
        Args:
            project_id: Project ID to retrieve
            
        Returns:
            ProjectRecord if found, None otherwise
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, name, root_path, collection_name, lsp_enabled, 
                           last_scan, created_at, updated_at, metadata
                    FROM projects WHERE id = ?
                    """,
                    (project_id,),
                )
                
                row = cursor.fetchone()
                if row:
                    return ProjectRecord(
                        id=row[0],
                        name=row[1],
                        root_path=row[2],
                        collection_name=row[3],
                        lsp_enabled=bool(row[4]),
                        last_scan=datetime.fromisoformat(row[5]) if row[5] else None,
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                        metadata=json.loads(row[8]) if row[8] else None,
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get project {project_id}: {e}")
        
        return None
        
    async def get_project_by_path(self, root_path: str) -> Optional[ProjectRecord]:
        """
        Get project by root path.
        
        Args:
            root_path: Project root path
            
        Returns:
            ProjectRecord if found, None otherwise
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, name, root_path, collection_name, lsp_enabled, 
                           last_scan, created_at, updated_at, metadata
                    FROM projects WHERE root_path = ?
                    """,
                    (root_path,),
                )
                
                row = cursor.fetchone()
                if row:
                    return ProjectRecord(
                        id=row[0],
                        name=row[1],
                        root_path=row[2],
                        collection_name=row[3],
                        lsp_enabled=bool(row[4]),
                        last_scan=datetime.fromisoformat(row[5]) if row[5] else None,
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                        metadata=json.loads(row[8]) if row[8] else None,
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get project by path {root_path}: {e}")
        
        return None

    async def update_project(self, project: ProjectRecord) -> bool:
        """
        Update an existing project.
        
        Args:
            project: ProjectRecord with updated values
            
        Returns:
            True if updated successfully
        """
        if not project.id:
            logger.error("Cannot update project without ID")
            return False
            
        try:
            project.updated_at = datetime.now(timezone.utc)
            
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE projects 
                    SET name = ?, root_path = ?, collection_name = ?, 
                        lsp_enabled = ?, last_scan = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        project.name,
                        project.root_path,
                        project.collection_name,
                        project.lsp_enabled,
                        project.last_scan,
                        json.dumps(project.metadata) if project.metadata else None,
                        project.updated_at,
                        project.id,
                    ),
                )
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated project {project.id}: {project.name}")
                    return True
                else:
                    logger.warning(f"No project found with ID {project.id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to update project {project.id}: {e}")
            return False

    async def delete_project(self, project_id: int) -> bool:
        """
        Delete a project.
        
        Args:
            project_id: Project ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted project {project_id}")
                    return True
                else:
                    logger.warning(f"No project found with ID {project_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}")
            return False

    async def list_projects(self, lsp_enabled_only: bool = False) -> List[ProjectRecord]:
        """
        List all projects.
        
        Args:
            lsp_enabled_only: If True, only return LSP-enabled projects
            
        Returns:
            List of ProjectRecord objects
        """
        projects = []
        
        try:
            async with self.transaction() as conn:
                query = """
                    SELECT id, name, root_path, collection_name, lsp_enabled, 
                           last_scan, created_at, updated_at, metadata
                    FROM projects
                """
                
                params = []
                if lsp_enabled_only:
                    query += " WHERE lsp_enabled = 1"
                    
                query += " ORDER BY name"
                
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    projects.append(ProjectRecord(
                        id=row[0],
                        name=row[1],
                        root_path=row[2],
                        collection_name=row[3],
                        lsp_enabled=bool(row[4]),
                        last_scan=datetime.fromisoformat(row[5]) if row[5] else None,
                        created_at=datetime.fromisoformat(row[6]),
                        updated_at=datetime.fromisoformat(row[7]),
                        metadata=json.loads(row[8]) if row[8] else None,
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            
        return projects

    async def update_project_scan_time(self, project_id: int) -> bool:
        """
        Update the last scan time for a project.
        
        Args:
            project_id: Project ID to update
            
        Returns:
            True if updated successfully
        """
        try:
            now = datetime.now(timezone.utc)
            
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "UPDATE projects SET last_scan = ?, updated_at = ? WHERE id = ?",
                    (now, now, project_id),
                )
                
                if cursor.rowcount > 0:
                    logger.debug(f"Updated scan time for project {project_id}")
                    return True
                else:
                    logger.warning(f"No project found with ID {project_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to update scan time for project {project_id}: {e}")
            return False

    # LSP Integration - LSP Server Management Methods
    
    async def create_lsp_server(self, server: LSPServerRecord) -> Optional[int]:
        """
        Create a new LSP server record.
        
        Args:
            server: LSPServerRecord to create
            
        Returns:
            Server ID if created successfully, None otherwise
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO lsp_servers (language, server_path, version, capabilities, 
                                           status, last_health_check, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        server.language,
                        server.server_path,
                        server.version,
                        json.dumps(server.capabilities) if server.capabilities else None,
                        server.status.value,
                        server.last_health_check,
                        json.dumps(server.metadata) if server.metadata else None,
                        server.created_at,
                        server.updated_at,
                    ),
                )
                
                server_id = cursor.lastrowid
                logger.info(f"Created LSP server: {server.language} at {server.server_path} with ID {server_id}")
                return server_id
                
        except Exception as e:
            logger.error(f"Failed to create LSP server {server.language}: {e}")
            return None

    async def get_lsp_server(self, server_id: int) -> Optional[LSPServerRecord]:
        """
        Get LSP server by ID.
        
        Args:
            server_id: Server ID to retrieve
            
        Returns:
            LSPServerRecord if found, None otherwise
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, language, server_path, version, capabilities, status, 
                           last_health_check, created_at, updated_at, metadata
                    FROM lsp_servers WHERE id = ?
                    """,
                    (server_id,),
                )
                
                row = cursor.fetchone()
                if row:
                    return LSPServerRecord(
                        id=row[0],
                        language=row[1],
                        server_path=row[2],
                        version=row[3],
                        capabilities=json.loads(row[4]) if row[4] else None,
                        status=LSPServerStatus(row[5]),
                        last_health_check=datetime.fromisoformat(row[6]) if row[6] else None,
                        created_at=datetime.fromisoformat(row[7]),
                        updated_at=datetime.fromisoformat(row[8]),
                        metadata=json.loads(row[9]) if row[9] else None,
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get LSP server {server_id}: {e}")
        
        return None

    async def get_lsp_server_by_language(self, language: str) -> Optional[LSPServerRecord]:
        """
        Get LSP server by language.
        
        Args:
            language: Programming language identifier
            
        Returns:
            LSPServerRecord if found, None otherwise
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, language, server_path, version, capabilities, status, 
                           last_health_check, created_at, updated_at, metadata
                    FROM lsp_servers WHERE language = ? AND status != 'unavailable'
                    ORDER BY last_health_check DESC
                    LIMIT 1
                    """,
                    (language,),
                )
                
                row = cursor.fetchone()
                if row:
                    return LSPServerRecord(
                        id=row[0],
                        language=row[1],
                        server_path=row[2],
                        version=row[3],
                        capabilities=json.loads(row[4]) if row[4] else None,
                        status=LSPServerStatus(row[5]),
                        last_health_check=datetime.fromisoformat(row[6]) if row[6] else None,
                        created_at=datetime.fromisoformat(row[7]),
                        updated_at=datetime.fromisoformat(row[8]),
                        metadata=json.loads(row[9]) if row[9] else None,
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get LSP server for language {language}: {e}")
        
        return None

    async def update_lsp_server(self, server: LSPServerRecord) -> bool:
        """
        Update an existing LSP server.
        
        Args:
            server: LSPServerRecord with updated values
            
        Returns:
            True if updated successfully
        """
        if not server.id:
            logger.error("Cannot update LSP server without ID")
            return False
            
        try:
            server.updated_at = datetime.now(timezone.utc)
            
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE lsp_servers 
                    SET language = ?, server_path = ?, version = ?, capabilities = ?, 
                        status = ?, last_health_check = ?, metadata = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        server.language,
                        server.server_path,
                        server.version,
                        json.dumps(server.capabilities) if server.capabilities else None,
                        server.status.value,
                        server.last_health_check,
                        json.dumps(server.metadata) if server.metadata else None,
                        server.updated_at,
                        server.id,
                    ),
                )
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated LSP server {server.id}: {server.language}")
                    return True
                else:
                    logger.warning(f"No LSP server found with ID {server.id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to update LSP server {server.id}: {e}")
            return False

    async def delete_lsp_server(self, server_id: int) -> bool:
        """
        Delete an LSP server.
        
        Args:
            server_id: Server ID to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute("DELETE FROM lsp_servers WHERE id = ?", (server_id,))
                
                if cursor.rowcount > 0:
                    logger.info(f"Deleted LSP server {server_id}")
                    return True
                else:
                    logger.warning(f"No LSP server found with ID {server_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to delete LSP server {server_id}: {e}")
            return False

    async def list_lsp_servers(self, language: str = None, status: LSPServerStatus = None) -> List[LSPServerRecord]:
        """
        List LSP servers with optional filtering.
        
        Args:
            language: Filter by programming language
            status: Filter by server status
            
        Returns:
            List of LSPServerRecord objects
        """
        servers = []
        
        try:
            async with self.transaction() as conn:
                query = """
                    SELECT id, language, server_path, version, capabilities, status, 
                           last_health_check, created_at, updated_at, metadata
                    FROM lsp_servers
                """
                
                conditions = []
                params = []
                
                if language:
                    conditions.append("language = ?")
                    params.append(language)
                    
                if status:
                    conditions.append("status = ?")
                    params.append(status.value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                    
                query += " ORDER BY language, created_at"
                
                cursor = conn.execute(query, params)
                
                for row in cursor:
                    servers.append(LSPServerRecord(
                        id=row[0],
                        language=row[1],
                        server_path=row[2],
                        version=row[3],
                        capabilities=json.loads(row[4]) if row[4] else None,
                        status=LSPServerStatus(row[5]),
                        last_health_check=datetime.fromisoformat(row[6]) if row[6] else None,
                        created_at=datetime.fromisoformat(row[7]),
                        updated_at=datetime.fromisoformat(row[8]),
                        metadata=json.loads(row[9]) if row[9] else None,
                    ))
                    
        except Exception as e:
            logger.error(f"Failed to list LSP servers: {e}")
            
        return servers

    async def update_lsp_server_health(self, server_id: int, status: LSPServerStatus) -> bool:
        """
        Update the health status of an LSP server.
        
        Args:
            server_id: Server ID to update
            status: New server status
            
        Returns:
            True if updated successfully
        """
        try:
            now = datetime.now(timezone.utc)
            
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE lsp_servers 
                    SET status = ?, last_health_check = ?, updated_at = ? 
                    WHERE id = ?
                    """,
                    (status.value, now, now, server_id),
                )
                
                if cursor.rowcount > 0:
                    logger.debug(f"Updated health for LSP server {server_id}: {status.value}")
                    return True
                else:
                    logger.warning(f"No LSP server found with ID {server_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to update health for LSP server {server_id}: {e}")
            return False

    async def get_active_lsp_servers(self) -> List[LSPServerRecord]:
        """
        Get all currently active LSP servers.
        
        Returns:
            List of active LSPServerRecord objects
        """
        return await self.list_lsp_servers(status=LSPServerStatus.ACTIVE)

    # LSP Integration - File Metadata Management Methods

    async def update_file_lsp_metadata(
        self,
        file_path: str,
        language_id: str,
        lsp_server_id: int,
        symbols_count: int = 0,
        lsp_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update LSP-specific metadata for a file.
        
        Args:
            file_path: Path to the file
            language_id: Programming language identifier
            lsp_server_id: ID of LSP server used for analysis
            symbols_count: Number of symbols extracted
            lsp_metadata: Additional LSP-specific metadata
            
        Returns:
            True if updated successfully
        """
        try:
            now = datetime.now(timezone.utc)
            
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE file_processing 
                    SET language_id = ?, lsp_extracted = 1, symbols_count = ?, 
                        lsp_server_id = ?, last_lsp_analysis = ?, lsp_metadata = ?, updated_at = ?
                    WHERE file_path = ?
                    """,
                    (
                        language_id,
                        symbols_count,
                        lsp_server_id,
                        now,
                        json.dumps(lsp_metadata) if lsp_metadata else None,
                        now,
                        file_path,
                    ),
                )
                
                if cursor.rowcount > 0:
                    logger.debug(f"Updated LSP metadata for file {file_path}: {symbols_count} symbols")
                    return True
                else:
                    logger.warning(f"No file processing record found for {file_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to update LSP metadata for file {file_path}: {e}")
            return False

    async def get_file_lsp_metadata(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get LSP metadata for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with LSP metadata if found, None otherwise
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT language_id, lsp_extracted, symbols_count, lsp_server_id, 
                           last_lsp_analysis, lsp_metadata
                    FROM file_processing 
                    WHERE file_path = ?
                    """,
                    (file_path,),
                )
                
                row = cursor.fetchone()
                if row:
                    return {
                        "language_id": row[0],
                        "lsp_extracted": bool(row[1]),
                        "symbols_count": row[2],
                        "lsp_server_id": row[3],
                        "last_lsp_analysis": row[4],
                        "lsp_metadata": json.loads(row[5]) if row[5] else None,
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get LSP metadata for file {file_path}: {e}")
            
        return None

    async def get_files_by_language(self, language_id: str) -> List[str]:
        """
        Get all files for a specific programming language.
        
        Args:
            language_id: Programming language identifier
            
        Returns:
            List of file paths for the specified language
        """
        files = []
        
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT file_path 
                    FROM file_processing 
                    WHERE language_id = ? AND lsp_extracted = 1
                    ORDER BY last_lsp_analysis DESC
                    """,
                    (language_id,),
                )
                
                files = [row[0] for row in cursor]
                    
        except Exception as e:
            logger.error(f"Failed to get files by language {language_id}: {e}")
            
        return files

    async def get_files_needing_lsp_analysis(self, language_id: str = None) -> List[str]:
        """
        Get files that need LSP analysis.
        
        Args:
            language_id: Optional language filter
            
        Returns:
            List of file paths needing LSP analysis
        """
        files = []
        
        try:
            async with self.transaction() as conn:
                query = """
                    SELECT file_path 
                    FROM file_processing 
                    WHERE lsp_extracted = 0 AND status = 'completed'
                """
                
                params = []
                if language_id:
                    query += " AND language_id = ?"
                    params.append(language_id)
                    
                query += " ORDER BY updated_at DESC"
                
                cursor = conn.execute(query, params)
                files = [row[0] for row in cursor]
                    
        except Exception as e:
            logger.error(f"Failed to get files needing LSP analysis: {e}")
            
        return files

    async def mark_file_lsp_failed(self, file_path: str, error_message: str) -> bool:
        """
        Mark a file as having failed LSP analysis.
        
        Args:
            file_path: Path to the file
            error_message: Error message describing the failure
            
        Returns:
            True if marked successfully
        """
        try:
            now = datetime.now(timezone.utc)
            
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE file_processing 
                    SET lsp_extracted = 0, last_lsp_analysis = ?, updated_at = ?,
                        lsp_metadata = ?
                    WHERE file_path = ?
                    """,
                    (
                        now,
                        now,
                        json.dumps({"error": error_message, "failed_at": now.isoformat()}),
                        file_path,
                    ),
                )
                
                if cursor.rowcount > 0:
                    logger.warning(f"Marked LSP analysis as failed for {file_path}: {error_message}")
                    return True
                else:
                    logger.warning(f"No file processing record found for {file_path}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to mark LSP failure for file {file_path}: {e}")
            return False

    async def get_lsp_analysis_stats(self) -> Dict[str, Any]:
        """
        Get statistics about LSP analysis across all files.
        
        Returns:
            Dictionary with LSP analysis statistics
        """
        try:
            async with self.transaction() as conn:
                # Get overall stats
                cursor = conn.execute(
                    """
                    SELECT 
                        COUNT(*) as total_files,
                        SUM(CASE WHEN lsp_extracted = 1 THEN 1 ELSE 0 END) as analyzed_files,
                        SUM(symbols_count) as total_symbols,
                        COUNT(DISTINCT language_id) as languages_count
                    FROM file_processing
                    WHERE language_id IS NOT NULL
                    """
                )
                
                overall_stats = cursor.fetchone()
                
                # Get per-language stats
                cursor = conn.execute(
                    """
                    SELECT 
                        language_id,
                        COUNT(*) as file_count,
                        SUM(CASE WHEN lsp_extracted = 1 THEN 1 ELSE 0 END) as analyzed_count,
                        SUM(symbols_count) as symbols_count,
                        AVG(symbols_count) as avg_symbols
                    FROM file_processing
                    WHERE language_id IS NOT NULL
                    GROUP BY language_id
                    ORDER BY file_count DESC
                    """
                )
                
                language_stats = [
                    {
                        "language": row[0],
                        "file_count": row[1],
                        "analyzed_count": row[2],
                        "symbols_count": row[3],
                        "avg_symbols": round(row[4], 2) if row[4] else 0,
                    }
                    for row in cursor
                ]
                
                return {
                    "total_files": overall_stats[0],
                    "analyzed_files": overall_stats[1],
                    "total_symbols": overall_stats[2],
                    "languages_count": overall_stats[3],
                    "analysis_rate": round(overall_stats[1] / overall_stats[0] * 100, 2) if overall_stats[0] > 0 else 0,
                    "language_breakdown": language_stats,
                }
                
        except Exception as e:
            logger.error(f"Failed to get LSP analysis stats: {e}")
            return {"error": str(e)}

    @classmethod
    def migrate_from_legacy_path(cls, legacy_db_path: str) -> bool:
        """Migrate existing database from legacy path to OS-standard location.

        Args:
            legacy_db_path: Path to existing legacy database file

        Returns:
            bool: True if migration successful or not needed, False on error
        """
        try:
            legacy_path = Path(legacy_db_path)
            if not legacy_path.exists():
                logger.info(f"Legacy database not found at {legacy_path}, no migration needed")
                return True

            # Get OS-standard location
            os_dirs = OSDirectories()
            os_dirs.ensure_directories()
            new_path = os_dirs.get_state_file("workspace_state.db")

            if new_path.exists():
                logger.info(f"OS-standard database already exists at {new_path}, skipping migration")
                return True

            # Perform migration
            logger.info(f"Migrating database from {legacy_path} to {new_path}")

            # Copy the file
            import shutil
            shutil.copy2(legacy_path, new_path)

            # Verify the migration
            if new_path.exists() and new_path.stat().st_size == legacy_path.stat().st_size:
                logger.info(f"Database migration successful. Legacy file remains at {legacy_path}")
                logger.info("You can safely delete the legacy database file after verifying the new location works")
                return True
            else:
                logger.error("Database migration verification failed")
                # Clean up partial migration
                if new_path.exists():
                    new_path.unlink()
                return False

        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            return False


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