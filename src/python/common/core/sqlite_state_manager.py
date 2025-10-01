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
import hashlib
import json
# Use unified logging system to prevent console interference in MCP mode
import re
import subprocess
from loguru import logger
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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

    SCHEMA_VERSION = 4  # Updated for language support schema additions
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
            # Ingestion queue table (new queue system with tenant/branch support)
            """
            CREATE TABLE IF NOT EXISTS ingestion_queue (
                file_absolute_path TEXT PRIMARY KEY NOT NULL,
                collection_name TEXT NOT NULL,
                tenant_id TEXT DEFAULT 'default',
                branch TEXT DEFAULT 'main',
                operation TEXT NOT NULL CHECK (operation IN ('ingest', 'update', 'delete')),
                priority INTEGER NOT NULL DEFAULT 5 CHECK (priority BETWEEN 0 AND 10),
                queued_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                retry_count INTEGER NOT NULL DEFAULT 0,
                retry_from TEXT,
                error_message_id INTEGER,
                metadata TEXT,
                FOREIGN KEY (retry_from) REFERENCES ingestion_queue(file_absolute_path) ON DELETE SET NULL
            )
            """,
            # Indexes for ingestion_queue
            "CREATE INDEX IF NOT EXISTS idx_ingestion_queue_priority_time ON ingestion_queue(priority DESC, queued_timestamp ASC)",
            "CREATE INDEX IF NOT EXISTS idx_ingestion_queue_collection ON ingestion_queue(collection_name, tenant_id, branch)",
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
            # Language Support Tables (Schema Version 4+)
            """
            CREATE TABLE languages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language_name TEXT NOT NULL UNIQUE,
                file_extensions TEXT,  -- JSON array of file extensions
                lsp_name TEXT,
                lsp_executable TEXT,
                lsp_absolute_path TEXT,
                lsp_missing BOOLEAN NOT NULL DEFAULT 0,
                ts_grammar TEXT,
                ts_cli_absolute_path TEXT,
                ts_missing BOOLEAN NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Indexes for languages
            "CREATE INDEX idx_languages_language_name ON languages(language_name)",
            "CREATE INDEX idx_languages_lsp_missing ON languages(lsp_missing)",
            "CREATE INDEX idx_languages_ts_missing ON languages(ts_missing)",
            """
            CREATE TABLE files_missing_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_absolute_path TEXT NOT NULL UNIQUE,
                language_name TEXT,
                branch TEXT,
                missing_lsp_metadata BOOLEAN NOT NULL DEFAULT 0,
                missing_ts_metadata BOOLEAN NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (language_name) REFERENCES languages(language_name) ON DELETE SET NULL
            )
            """,
            # Indexes for files_missing_metadata
            "CREATE INDEX idx_files_missing_metadata_file_path ON files_missing_metadata(file_absolute_path)",
            "CREATE INDEX idx_files_missing_metadata_language ON files_missing_metadata(language_name)",
            "CREATE INDEX idx_files_missing_metadata_missing ON files_missing_metadata(language_name, missing_lsp_metadata, missing_ts_metadata)",
            """
            CREATE TABLE tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL UNIQUE,
                tool_type TEXT NOT NULL CHECK (tool_type IN ('lsp_server', 'tree_sitter_cli')),
                absolute_path TEXT,
                version TEXT,
                missing BOOLEAN NOT NULL DEFAULT 0,
                last_check_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
            # Indexes for tools
            "CREATE INDEX idx_tools_tool_name ON tools(tool_name)",
            "CREATE INDEX idx_tools_tool_type_missing ON tools(tool_type, missing)",
            """
            CREATE TABLE language_support_version (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                yaml_hash TEXT NOT NULL UNIQUE,
                loaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                language_count INTEGER NOT NULL DEFAULT 0,
                last_checked_at TIMESTAMP
            )
            """,
            # Indexes for language_support_version
            "CREATE INDEX idx_language_support_version_yaml_hash ON language_support_version(yaml_hash)",
            "CREATE INDEX idx_language_support_version_loaded_at ON language_support_version(loaded_at)",
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

        async with self.transaction() as conn:
            # Migrate from version 1 to version 2 - Add LSP tables
            if from_version == 1 and to_version >= 2:
                logger.info("Applying migration: v1 -> v2 (LSP integration tables)")
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
                    conn.execute(sql)

                logger.info("Successfully migrated to schema version 2 (added projects and lsp_servers tables)")

            # Migrate from version 2 to version 3 - Add LSP fields to file_processing
            if from_version <= 2 and to_version >= 3:
                logger.info("Applying migration: v2 -> v3 (LSP file processing fields)")
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
                    conn.execute(sql)

                logger.info("Successfully migrated to schema version 3 (added LSP fields to file_processing)")

            # Migrate from version 3 to version 4 - Add language support tables
            if from_version <= 3 and to_version >= 4:
                logger.info("Applying migration: v3 -> v4 (language support schema)")
                language_support_sql = [
                    # Add languages table
                    """
                    CREATE TABLE languages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        language_name TEXT NOT NULL UNIQUE,
                        file_extensions TEXT,  -- JSON array of file extensions
                        lsp_name TEXT,
                        lsp_executable TEXT,
                        lsp_absolute_path TEXT,
                        lsp_missing BOOLEAN NOT NULL DEFAULT 0,
                        ts_grammar TEXT,
                        ts_cli_absolute_path TEXT,
                        ts_missing BOOLEAN NOT NULL DEFAULT 0,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                    # Add indexes for languages
                    "CREATE INDEX idx_languages_language_name ON languages(language_name)",
                    "CREATE INDEX idx_languages_lsp_missing ON languages(lsp_missing)",
                    "CREATE INDEX idx_languages_ts_missing ON languages(ts_missing)",
                    # Add files_missing_metadata table
                    """
                    CREATE TABLE files_missing_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_absolute_path TEXT NOT NULL UNIQUE,
                        language_name TEXT,
                        branch TEXT,
                        missing_lsp_metadata BOOLEAN NOT NULL DEFAULT 0,
                        missing_ts_metadata BOOLEAN NOT NULL DEFAULT 0,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (language_name) REFERENCES languages(language_name) ON DELETE SET NULL
                    )
                    """,
                    # Add indexes for files_missing_metadata
                    "CREATE INDEX idx_files_missing_metadata_file_path ON files_missing_metadata(file_absolute_path)",
                    "CREATE INDEX idx_files_missing_metadata_language ON files_missing_metadata(language_name)",
                    "CREATE INDEX idx_files_missing_metadata_missing ON files_missing_metadata(language_name, missing_lsp_metadata, missing_ts_metadata)",
                    # Add tools table
                    """
                    CREATE TABLE tools (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tool_name TEXT NOT NULL UNIQUE,
                        tool_type TEXT NOT NULL CHECK (tool_type IN ('lsp_server', 'tree_sitter_cli')),
                        absolute_path TEXT,
                        version TEXT,
                        missing BOOLEAN NOT NULL DEFAULT 0,
                        last_check_at TIMESTAMP,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                    # Add indexes for tools
                    "CREATE INDEX idx_tools_tool_name ON tools(tool_name)",
                    "CREATE INDEX idx_tools_tool_type_missing ON tools(tool_type, missing)",
                    # Add language_support_version table
                    """
                    CREATE TABLE language_support_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        yaml_hash TEXT NOT NULL UNIQUE,
                        loaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        language_count INTEGER NOT NULL DEFAULT 0,
                        last_checked_at TIMESTAMP
                    )
                    """,
                    # Add indexes for language_support_version
                    "CREATE INDEX idx_language_support_version_yaml_hash ON language_support_version(yaml_hash)",
                    "CREATE INDEX idx_language_support_version_loaded_at ON language_support_version(loaded_at)",
                ]

                for sql in language_support_sql:
                    conn.execute(sql)

                logger.info("Successfully migrated to schema version 4 (added language support tables)")

            # Record the migration
            conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)", (to_version,)
            )

        logger.info(f"Database migration completed: v{from_version} -> v{to_version}")

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

    async def calculate_tenant_id(self, project_root: Path) -> str:
        """
        Calculate a consistent tenant ID for a project.

        Uses git remote URL if available (sanitized), otherwise falls back to
        a hash of the project root path.

        Args:
            project_root: Path to the project root directory

        Returns:
            Consistent tenant_id string for the project
        """
        try:
            # Try to get git remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                remote_url = result.stdout.strip()
                # Sanitize the URL to create a valid tenant ID
                # Remove protocol prefixes
                sanitized = re.sub(r'^(https?://|git@|ssh://)', '', remote_url)
                # Replace special characters with underscores
                sanitized = re.sub(r'[:/\.]+', '_', sanitized)
                # Remove .git suffix if present
                sanitized = re.sub(r'_git$', '', sanitized)
                # Convert to lowercase and remove leading/trailing underscores
                tenant_id = sanitized.lower().strip('_')
                logger.debug(f"Generated tenant_id from git remote: {tenant_id}")
                return tenant_id
            else:
                logger.debug(f"No git remote found for {project_root}, using path hash")

        except subprocess.TimeoutExpired:
            logger.warning(f"Git command timeout for {project_root}, using path hash")
        except FileNotFoundError:
            logger.debug(f"Git not found in PATH, using path hash for {project_root}")
        except Exception as e:
            logger.warning(f"Error getting git remote for {project_root}: {e}, using path hash")

        # Fallback: use hash of project root path
        path_str = str(project_root.resolve())
        path_hash = hashlib.sha256(path_str.encode('utf-8')).hexdigest()[:16]
        tenant_id = f"path_{path_hash}"
        logger.debug(f"Generated tenant_id from path hash: {tenant_id}")
        return tenant_id

    async def get_current_branch(self, project_root: Path) -> str:
        """
        Get the current git branch for a project.

        Args:
            project_root: Path to the project root directory

        Returns:
            Current branch name, defaults to 'main' if not in a git repository
        """
        try:
            # Try to get current branch from git
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0 and result.stdout.strip():
                branch = result.stdout.strip()
                logger.debug(f"Detected git branch: {branch}")
                return branch
            else:
                logger.debug(f"No git branch found for {project_root}, using default 'main'")
                return "main"

        except subprocess.TimeoutExpired:
            logger.warning(f"Git command timeout for {project_root}, using default 'main'")
            return "main"
        except FileNotFoundError:
            logger.debug(f"Git not found in PATH for {project_root}, using default 'main'")
            return "main"
        except Exception as e:
            logger.warning(f"Error getting git branch for {project_root}: {e}, using default 'main'")
            return "main"

    async def enqueue(
        self,
        file_path: str,
        collection: str,
        priority: int,
        tenant_id: str,
        branch: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Enqueue a file to the ingestion queue.

        Handles UNIQUE constraint violations gracefully by updating the priority
        of existing items instead of raising an error.

        Args:
            file_path: Absolute path to the file to enqueue
            collection: Target collection name
            priority: Priority level (0-10, where 10 is highest)
            tenant_id: Tenant identifier for multi-tenancy support
            branch: Branch identifier for multi-branch support
            metadata: Optional metadata dictionary to store with the queue item

        Returns:
            Queue ID (file_absolute_path) of the enqueued item

        Raises:
            ValueError: If priority is out of valid range (0-10)
        """
        try:
            # Validate priority
            if not 0 <= priority <= 10:
                raise ValueError(f"Priority must be between 0 and 10, got {priority}")

            # Normalize file path to absolute path
            file_absolute_path = str(Path(file_path).resolve())

            # File path serves as queue ID
            queue_id = file_absolute_path

            async with self.transaction() as conn:
                # Try to insert the new queue item
                # If it already exists (UNIQUE constraint on file_absolute_path),
                # update its priority instead
                try:
                    conn.execute(
                        """
                        INSERT INTO ingestion_queue
                        (file_absolute_path, collection_name, tenant_id, branch,
                         operation, priority, metadata)
                        VALUES (?, ?, ?, ?, 'ingest', ?, ?)
                        """,
                        (
                            file_absolute_path,
                            collection,
                            tenant_id,
                            branch,
                            priority,
                            self._serialize_json(metadata) if metadata else None,
                        ),
                    )
                    logger.debug(
                        f"Enqueued file: {file_absolute_path} "
                        f"(collection={collection}, priority={priority}, "
                        f"tenant={tenant_id}, branch={branch})"
                    )
                except sqlite3.IntegrityError as e:
                    # Handle UNIQUE constraint violation by updating priority
                    if "UNIQUE constraint" in str(e) or "PRIMARY KEY" in str(e):
                        conn.execute(
                            """
                            UPDATE ingestion_queue
                            SET priority = ?, queued_timestamp = CURRENT_TIMESTAMP,
                                metadata = COALESCE(?, metadata)
                            WHERE file_absolute_path = ?
                            """,
                            (
                                priority,
                                self._serialize_json(metadata) if metadata else None,
                                file_absolute_path,
                            ),
                        )
                        logger.debug(
                            f"Updated existing queue item: {file_absolute_path} "
                            f"(new priority={priority})"
                        )
                    else:
                        # Re-raise if it's a different integrity error
                        raise

            return queue_id

        except ValueError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Failed to enqueue file {file_path}: {e}")
            raise

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

                # Clean up new ingestion_queue (uses absolute path as primary key)
                file_absolute_path = str(Path(file_path).resolve())
                cursor = conn.execute(
                    "DELETE FROM ingestion_queue WHERE file_absolute_path = ?",
                    (file_absolute_path,)
                )

                if cursor.rowcount > 0:
                    logger.debug(f"Removed from ingestion queue: {file_absolute_path}")

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
                    INSERT INTO processing_queue
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

            logger.debug(f"Added file to processing queue: {file_path} (queue_id: {queue_id})")
            return queue_id

        except Exception as e:
            logger.error(f"Failed to add file to processing queue {file_path}: {e}")
            return ""
