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
import uuid

# Use unified logging system to prevent console interference in MCP mode
import re
import sqlite3
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

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


class ContentIngestionStatus(Enum):
    """Content ingestion queue status (Task 456/ADR-001).

    Used for MCP store() content that goes through SQLite queue
    when daemon is unavailable.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ContentIngestionQueueItem:
    """Item in the content ingestion queue (Task 456/ADR-001).

    Used to queue content from MCP store() operations when daemon
    is unavailable, ensuring daemon-only writes per First Principle 10.

    The idempotency_key prevents duplicate processing if the same
    content is submitted multiple times.
    """

    queue_id: str  # UUID for queue item
    idempotency_key: str  # SHA256(content + collection + source_type + metadata)[:32]
    content: str  # The actual content to store
    collection: str  # Target collection name
    source_type: str  # "scratchbook", "file", "web", "chat", etc.
    priority: int = 8  # 0-10, default HIGH for MCP context
    status: ContentIngestionStatus = ContentIngestionStatus.PENDING
    main_tag: str | None = None  # project_id or library_name
    full_tag: str | None = None  # main_tag.branch or main_tag.version
    metadata: dict[str, Any] | None = None
    created_at: datetime = None
    updated_at: datetime = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: str | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class FileProcessingRecord:
    """Record for tracking file processing state."""

    file_path: str
    collection: str
    status: FileProcessingStatus
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    created_at: datetime = None
    updated_at: datetime = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: str | None = None
    file_size: int | None = None
    file_hash: str | None = None
    metadata: dict[str, Any] | None = None
    document_id: str | None = None  # For multi-component testing
    # LSP-specific fields
    language_id: str | None = None
    lsp_extracted: bool = False
    symbols_count: int = 0
    lsp_server_id: int | None = None
    last_lsp_analysis: datetime | None = None
    lsp_metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class WatchFolderConfig:
    """Configuration for persistent watch folders.

    Multi-tenant routing (Task 402):
    - watch_type: "project" routes to _projects collection with project_id as tenant
    - watch_type: "library" routes to _libraries collection with library_name as tenant

    Error tracking (Task 461):
    - consecutive_errors: Number of consecutive processing errors
    - total_errors: Cumulative error count since watch created
    - last_error_at: Timestamp of most recent error
    - last_error_message: Description of most recent error
    - backoff_until: When to resume after backoff period
    - last_success_at: Timestamp of most recent successful processing
    - health_status: Current health state (healthy/degraded/backoff/disabled)
    """

    watch_id: str
    path: str
    collection: str
    patterns: list[str]
    ignore_patterns: list[str]
    auto_ingest: bool = True
    recursive: bool = True
    recursive_depth: int = 10
    debounce_seconds: float = 2.0
    enabled: bool = True
    # Multi-tenant routing fields (Task 402)
    watch_type: str = "project"  # "project" or "library"
    library_name: str | None = None  # Required for watch_type="library"
    created_at: datetime = None
    updated_at: datetime = None
    last_scan: datetime | None = None
    metadata: dict[str, Any] | None = None
    # Error tracking fields (Task 461)
    consecutive_errors: int = 0
    total_errors: int = 0
    last_error_at: datetime | None = None
    last_error_message: str | None = None
    backoff_until: datetime | None = None
    last_success_at: datetime | None = None
    health_status: str = "healthy"  # healthy, degraded, backoff, disabled

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at
        # Validate watch_type
        if self.watch_type not in ("project", "library"):
            self.watch_type = "project"  # Default to project
        # Validate health_status (Task 461)
        if self.health_status not in ("healthy", "degraded", "backoff", "disabled"):
            self.health_status = "healthy"


@dataclass
class ProcessingQueueItem:
    """Item in the processing queue."""

    queue_id: str
    file_path: str
    collection: str
    priority: ProcessingPriority
    created_at: datetime = None
    scheduled_at: datetime | None = None
    attempts: int = 0
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at


@dataclass
class ProjectRecord:
    """Record for tracking LSP-enabled projects."""

    id: int | None
    name: str
    root_path: str
    collection_name: str
    project_id: str | None = None  # 12-char hex hash from root_path
    lsp_enabled: bool = False
    last_scan: datetime | None = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class LSPServerRecord:
    """Record for tracking LSP servers."""

    id: int | None
    language: str
    server_path: str
    version: str | None = None
    capabilities: dict[str, Any] | None = None
    status: LSPServerStatus = LSPServerStatus.INACTIVE
    last_health_check: datetime | None = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: dict[str, Any] | None = None

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

    SCHEMA_VERSION = 9  # v9: Add error tracking fields to watch_folders (Task 461)
    WAL_CHECKPOINT_INTERVAL = 300  # 5 minutes
    MAINTENANCE_INTERVAL = 3600  # 1 hour

    def __init__(self, db_path: str | None = None):
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

        self.connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._shutdown_event = asyncio.Event()
        self._maintenance_task: asyncio.Task | None = None
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
                watch_type TEXT NOT NULL DEFAULT 'project',  -- 'project' or 'library' (Task 402)
                library_name TEXT,  -- Required for library watch type (Task 402)
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_scan TIMESTAMP,
                metadata TEXT,  -- JSON
                -- Error tracking fields (Task 461)
                consecutive_errors INTEGER NOT NULL DEFAULT 0,
                total_errors INTEGER NOT NULL DEFAULT 0,
                last_error_at TIMESTAMP,
                last_error_message TEXT,
                backoff_until TIMESTAMP,
                last_success_at TIMESTAMP,
                health_status TEXT NOT NULL DEFAULT 'healthy'  -- healthy, degraded, backoff, disabled
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
            "CREATE INDEX IF NOT EXISTS idx_ingestion_queue_branch ON ingestion_queue(branch)",
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
                project_id TEXT,
                lsp_enabled BOOLEAN NOT NULL DEFAULT 0,
                last_scan TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,  -- JSON for additional project configuration
                -- Multi-tenant architecture columns (v7)
                priority TEXT DEFAULT 'normal' CHECK (priority IN ('high', 'normal', 'low')),
                active_sessions INTEGER DEFAULT 0,
                git_remote TEXT,
                registered_at TIMESTAMP,
                last_active TIMESTAMP
            )
            """,
            # Indexes for projects
            "CREATE INDEX idx_projects_name ON projects(name)",
            "CREATE INDEX idx_projects_root_path ON projects(root_path)",
            "CREATE INDEX idx_projects_collection_name ON projects(collection_name)",
            "CREATE INDEX idx_projects_project_id ON projects(project_id)",
            "CREATE INDEX idx_projects_lsp_enabled ON projects(lsp_enabled)",
            "CREATE INDEX idx_projects_last_scan ON projects(last_scan)",
            "CREATE INDEX idx_projects_priority ON projects(priority)",
            "CREATE INDEX idx_projects_active_sessions ON projects(active_sessions)",
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
            # Library watches table for reference documentation (v7)
            """
            CREATE TABLE library_watches (
                library_name TEXT PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                patterns TEXT NOT NULL DEFAULT '["*.pdf", "*.epub", "*.md", "*.txt"]',
                ignore_patterns TEXT NOT NULL DEFAULT '[".git/*", "__pycache__/*"]',
                enabled INTEGER DEFAULT 1,
                recursive INTEGER DEFAULT 1,
                recursive_depth INTEGER DEFAULT 10,
                debounce_seconds REAL DEFAULT 2.0,
                added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_scan TIMESTAMP,
                document_count INTEGER DEFAULT 0,
                metadata TEXT  -- JSON for extensibility
            )
            """,
            # Indexes for library_watches
            "CREATE INDEX idx_library_watches_path ON library_watches(path)",
            "CREATE INDEX idx_library_watches_enabled ON library_watches(enabled)",
            # Content ingestion queue for MCP store fallback (v8, Task 456/ADR-001)
            """
            CREATE TABLE content_ingestion_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                collection TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'scratchbook',
                priority INTEGER NOT NULL DEFAULT 8 CHECK (priority BETWEEN 0 AND 10),
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'done', 'failed')),
                main_tag TEXT,
                full_tag TEXT,
                metadata TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                error_message TEXT
            )
            """,
            # Indexes for content_ingestion_queue
            "CREATE INDEX idx_content_queue_status_priority ON content_ingestion_queue(status, priority DESC, created_at ASC)",
            "CREATE INDEX idx_content_queue_idempotency ON content_ingestion_queue(idempotency_key)",
            "CREATE INDEX idx_content_queue_collection ON content_ingestion_queue(collection)",
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

            # Migrate from version 4 to version 5 - Add ingestion_queue table
            if from_version <= 4 and to_version >= 5:
                logger.info("Applying migration: v4 -> v5 (ingestion queue table)")
                ingestion_queue_sql = [
                    # Add ingestion_queue table with tenant/branch support
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
                    # Add indexes for ingestion_queue
                    "CREATE INDEX IF NOT EXISTS idx_ingestion_queue_priority_time ON ingestion_queue(priority DESC, queued_timestamp ASC)",
                    "CREATE INDEX IF NOT EXISTS idx_ingestion_queue_collection ON ingestion_queue(collection_name, tenant_id, branch)",
                ]

                for sql in ingestion_queue_sql:
                    conn.execute(sql)

                logger.info("Successfully migrated to schema version 5 (added ingestion_queue table)")

            # Migrate from version 5 to version 6 - Add project_id column to projects table
            if from_version <= 5 and to_version >= 6:
                logger.info("Applying migration: v5 -> v6 (add project_id to projects table)")
                project_id_sql = [
                    # Add project_id column to projects table
                    "ALTER TABLE projects ADD COLUMN project_id TEXT",
                    # Add index for project_id
                    "CREATE INDEX IF NOT EXISTS idx_projects_project_id ON projects(project_id)",
                    # Add branch index to ingestion_queue if not exists
                    "CREATE INDEX IF NOT EXISTS idx_ingestion_queue_branch ON ingestion_queue(branch)",
                ]

                for sql in project_id_sql:
                    conn.execute(sql)

                # Update existing projects to populate project_id from root_path hash
                cursor = conn.execute("SELECT id, root_path FROM projects WHERE project_id IS NULL")
                rows = cursor.fetchall()
                for row in rows:
                    project_id = row["id"]
                    root_path = row["root_path"]
                    # Calculate project_id as 12-char hex hash from root_path
                    path_hash = hashlib.sha256(root_path.encode('utf-8')).hexdigest()[:12]
                    conn.execute(
                        "UPDATE projects SET project_id = ? WHERE id = ?",
                        (path_hash, project_id)
                    )

                logger.info("Successfully migrated to schema version 6 (added project_id column to projects table)")

            # Migrate from version 6 to version 7 - Multi-tenant architecture support
            if from_version <= 6 and to_version >= 7:
                logger.info("Applying migration: v6 -> v7 (multi-tenant architecture support)")
                multi_tenant_sql = [
                    # Add priority and session tracking to projects table
                    "ALTER TABLE projects ADD COLUMN priority TEXT DEFAULT 'normal' CHECK (priority IN ('high', 'normal', 'low'))",
                    "ALTER TABLE projects ADD COLUMN active_sessions INTEGER DEFAULT 0",
                    "ALTER TABLE projects ADD COLUMN git_remote TEXT",
                    "ALTER TABLE projects ADD COLUMN registered_at TIMESTAMP",
                    "ALTER TABLE projects ADD COLUMN last_active TIMESTAMP",
                    # Add indexes for new columns
                    "CREATE INDEX IF NOT EXISTS idx_projects_priority ON projects(priority)",
                    "CREATE INDEX IF NOT EXISTS idx_projects_active_sessions ON projects(active_sessions)",
                    # Create library_watches table for reference documentation
                    """
                    CREATE TABLE IF NOT EXISTS library_watches (
                        library_name TEXT PRIMARY KEY,
                        path TEXT NOT NULL UNIQUE,
                        patterns TEXT NOT NULL DEFAULT '["*.pdf", "*.epub", "*.md", "*.txt"]',
                        ignore_patterns TEXT NOT NULL DEFAULT '[".git/*", "__pycache__/*"]',
                        enabled INTEGER DEFAULT 1,
                        recursive INTEGER DEFAULT 1,
                        recursive_depth INTEGER DEFAULT 10,
                        debounce_seconds REAL DEFAULT 2.0,
                        added_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_scan TIMESTAMP,
                        document_count INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                    """,
                    # Add indexes for library_watches
                    "CREATE INDEX IF NOT EXISTS idx_library_watches_path ON library_watches(path)",
                    "CREATE INDEX IF NOT EXISTS idx_library_watches_enabled ON library_watches(enabled)",
                ]

                for sql in multi_tenant_sql:
                    try:
                        conn.execute(sql)
                    except sqlite3.OperationalError as e:
                        # Ignore "duplicate column" errors for ALTER TABLE (column may already exist)
                        if "duplicate column" not in str(e).lower():
                            raise

                # Populate registered_at for existing projects
                conn.execute(
                    "UPDATE projects SET registered_at = created_at WHERE registered_at IS NULL"
                )

                logger.info("Successfully migrated to schema version 7 (multi-tenant architecture support)")

            # Migrate from version 7 to version 8 - Content ingestion queue for daemon fallback (Task 456/ADR-001)
            if from_version <= 7 and to_version >= 8:
                logger.info("Applying migration: v7 -> v8 (content ingestion queue for daemon fallback)")
                content_queue_sql = [
                    # Content ingestion queue for MCP store fallback when daemon unavailable
                    """
                    CREATE TABLE IF NOT EXISTS content_ingestion_queue (
                        queue_id TEXT PRIMARY KEY,
                        idempotency_key TEXT NOT NULL UNIQUE,
                        content TEXT NOT NULL,
                        collection TEXT NOT NULL,
                        source_type TEXT NOT NULL DEFAULT 'scratchbook',
                        priority INTEGER NOT NULL DEFAULT 8 CHECK (priority BETWEEN 0 AND 10),
                        status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'done', 'failed')),
                        main_tag TEXT,
                        full_tag TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        max_retries INTEGER NOT NULL DEFAULT 3,
                        error_message TEXT
                    )
                    """,
                    # Indexes for content_ingestion_queue
                    "CREATE INDEX IF NOT EXISTS idx_content_queue_status_priority ON content_ingestion_queue(status, priority DESC, created_at ASC)",
                    "CREATE INDEX IF NOT EXISTS idx_content_queue_idempotency ON content_ingestion_queue(idempotency_key)",
                    "CREATE INDEX IF NOT EXISTS idx_content_queue_collection ON content_ingestion_queue(collection)",
                ]

                for sql in content_queue_sql:
                    conn.execute(sql)

                logger.info("Successfully migrated to schema version 8 (content ingestion queue)")

            # Migrate from version 8 to version 9 - Error tracking fields for watch_folders (Task 461)
            if from_version <= 8 and to_version >= 9:
                logger.info("Applying migration: v8 -> v9 (watch folder error tracking)")
                error_tracking_sql = [
                    # Add error tracking columns to watch_folders table
                    "ALTER TABLE watch_folders ADD COLUMN consecutive_errors INTEGER NOT NULL DEFAULT 0",
                    "ALTER TABLE watch_folders ADD COLUMN total_errors INTEGER NOT NULL DEFAULT 0",
                    "ALTER TABLE watch_folders ADD COLUMN last_error_at TIMESTAMP",
                    "ALTER TABLE watch_folders ADD COLUMN last_error_message TEXT",
                    "ALTER TABLE watch_folders ADD COLUMN backoff_until TIMESTAMP",
                    "ALTER TABLE watch_folders ADD COLUMN last_success_at TIMESTAMP",
                    "ALTER TABLE watch_folders ADD COLUMN health_status TEXT NOT NULL DEFAULT 'healthy'",
                    # Add index for health monitoring
                    "CREATE INDEX IF NOT EXISTS idx_watch_folders_health_status ON watch_folders(health_status)",
                ]

                for sql in error_tracking_sql:
                    try:
                        conn.execute(sql)
                    except Exception as e:
                        # Ignore if column already exists (idempotent migration)
                        if "duplicate column" not in str(e).lower():
                            raise

                logger.info("Successfully migrated to schema version 9 (watch folder error tracking)")

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

    def _serialize_json(self, data: Any) -> str | None:
        """Serialize data to JSON string."""
        if data is None:
            return None
        try:
            return json.dumps(data)
        except (TypeError, ValueError) as e:
            logger.warning(f"Failed to serialize data to JSON: {e}")
            return None

    def _deserialize_json(self, data: str | None) -> Any:
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
        metadata: dict | None = None,
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

    async def dequeue(
        self,
        batch_size: int = 10,
        tenant_id: str | None = None,
        branch: str | None = None,
    ) -> list[ProcessingQueueItem]:
        """
        Retrieve items from ingestion queue by priority (DESC) and scheduled_at (ASC).

        Args:
            batch_size: Maximum number of items to retrieve
            tenant_id: Filter by tenant ID
            branch: Filter by branch

        Returns:
            List of ProcessingQueueItem objects
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            # Build query with filters
            query = """
                SELECT file_absolute_path, collection_name, priority, tenant_id, branch,
                       metadata, retry_count, queued_timestamp
                FROM ingestion_queue
                WHERE 1=1
            """
            params = []

            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)

            if branch:
                query += " AND branch = ?"
                params.append(branch)

            query += " ORDER BY priority DESC, queued_timestamp ASC LIMIT ?"
            params.append(batch_size)

            with self._lock:
                cursor = self.connection.execute(query, params)
                rows = cursor.fetchall()

                items = []
                for row in rows:
                    metadata = self._deserialize_json(row["metadata"]) or {}

                    # Map integer priority (0-10) to ProcessingPriority enum
                    # Priority mapping: 0-2 -> LOW, 3-5 -> NORMAL, 6-8 -> HIGH, 9-10 -> URGENT
                    int_priority = row["priority"]
                    if int_priority <= 2:
                        priority = ProcessingPriority.LOW
                    elif int_priority <= 5:
                        priority = ProcessingPriority.NORMAL
                    elif int_priority <= 8:
                        priority = ProcessingPriority.HIGH
                    else:
                        priority = ProcessingPriority.URGENT

                    item = ProcessingQueueItem(
                        queue_id=row["file_absolute_path"],  # Use file path as queue ID
                        file_path=row["file_absolute_path"],
                        collection=row["collection_name"],
                        priority=priority,
                        scheduled_at=datetime.fromisoformat(row["queued_timestamp"]) if row["queued_timestamp"] else None,
                        metadata=metadata,
                        attempts=row["retry_count"],
                        created_at=datetime.fromisoformat(row["queued_timestamp"]) if row["queued_timestamp"] else None,
                    )
                    items.append(item)

                return items

        except Exception as e:
            logger.error(f"Failed to dequeue items: {e}")
            raise

    async def get_queue_depth(
        self,
        tenant_id: str | None = None,
        branch: str | None = None,
    ) -> int:
        """
        Get the current depth (count) of the ingestion queue.

        Args:
            tenant_id: Optional filter by tenant ID
            branch: Optional filter by branch

        Returns:
            Number of items in the queue matching the filters
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            query = "SELECT COUNT(*) FROM ingestion_queue WHERE 1=1"
            params = []

            if tenant_id:
                query += " AND tenant_id = ?"
                params.append(tenant_id)

            if branch:
                query += " AND branch = ?"
                params.append(branch)

            with self._lock:
                cursor = self.connection.execute(query, params)
                count = cursor.fetchone()[0]
                return count

        except Exception as e:
            logger.error(f"Failed to get queue depth: {e}")
            raise

    async def remove_from_queue(
        self,
        queue_id: str,
    ) -> bool:
        """
        Remove an item from the ingestion queue.

        Args:
            queue_id: Queue ID (file_absolute_path) of the item to remove

        Returns:
            True if the item was removed, False if it didn't exist

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM ingestion_queue WHERE file_absolute_path = ?",
                    (queue_id,)
                )
                deleted = cursor.rowcount > 0

                if deleted:
                    logger.debug(f"Removed item from queue: {queue_id}")
                else:
                    logger.warning(f"Queue item not found: {queue_id}")

                return deleted

        except Exception as e:
            logger.error(f"Failed to remove from queue {queue_id}: {e}")
            raise

    # Content Ingestion Queue Methods (Task 456/ADR-001)
    # These methods support MCP store() fallback when daemon is unavailable

    async def enqueue_ingestion(
        self,
        content: str,
        collection: str,
        source_type: str = "scratchbook",
        priority: int = 8,
        main_tag: str | None = None,
        full_tag: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """
        Enqueue content for ingestion with idempotency support.

        Uses SHA256 hash of (content + collection + source_type + metadata) as idempotency key.
        If content with same idempotency key already exists, returns existing queue_id without
        creating a duplicate.

        Args:
            content: Text content to ingest
            collection: Target collection name (canonical per ADR-001)
            source_type: Content source type (scratchbook, file, etc.)
            priority: Priority level (0-10, where 10 is highest)
            main_tag: Main tag for hierarchical organization (e.g., project_id)
            full_tag: Full tag with subtag (e.g., project_id.branch)
            metadata: Optional metadata dictionary

        Returns:
            Tuple of (queue_id, is_new) where is_new indicates if item was newly created

        Raises:
            ValueError: If priority is out of valid range (0-10)
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        if not 0 <= priority <= 10:
            raise ValueError(f"Priority must be between 0 and 10, got {priority}")

        try:
            # Calculate idempotency key from content + collection + source_type + metadata
            idempotency_input = f"{content}|{collection}|{source_type}|{json.dumps(metadata or {}, sort_keys=True)}"
            idempotency_key = hashlib.sha256(idempotency_input.encode("utf-8")).hexdigest()[:32]

            # Generate queue_id
            queue_id = str(uuid.uuid4())

            async with self.transaction() as conn:
                # Check if item with same idempotency key already exists
                cursor = conn.execute(
                    "SELECT queue_id FROM content_ingestion_queue WHERE idempotency_key = ?",
                    (idempotency_key,)
                )
                existing = cursor.fetchone()

                if existing:
                    logger.debug(f"Content already queued (idempotency): {existing['queue_id']}")
                    return existing["queue_id"], False

                # Insert new queue item
                conn.execute(
                    """
                    INSERT INTO content_ingestion_queue
                    (queue_id, idempotency_key, content, collection, source_type, priority,
                     status, main_tag, full_tag, metadata, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        queue_id,
                        idempotency_key,
                        content,
                        collection,
                        source_type,
                        priority,
                        main_tag,
                        full_tag,
                        self._serialize_json(metadata) if metadata else None,
                    ),
                )

                logger.debug(
                    f"Enqueued content: {queue_id} "
                    f"(collection={collection}, priority={priority}, source={source_type})"
                )

                return queue_id, True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to enqueue content ingestion: {e}")
            raise

    async def dequeue_content_ingestion(
        self,
        batch_size: int = 10,
        collection: str | None = None,
    ) -> list[ContentIngestionQueueItem]:
        """
        Retrieve pending content items from queue for processing.

        Items are returned in priority order (DESC) then by creation time (ASC).
        Retrieved items are marked as 'in_progress'.

        Args:
            batch_size: Maximum number of items to retrieve
            collection: Optional filter by collection name

        Returns:
            List of ContentIngestionQueueItem objects ready for processing

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            async with self.transaction() as conn:
                # Build query with optional collection filter
                query = """
                    SELECT queue_id, idempotency_key, content, collection, source_type,
                           priority, status, main_tag, full_tag, metadata, created_at,
                           updated_at, started_at, completed_at, retry_count, max_retries,
                           error_message
                    FROM content_ingestion_queue
                    WHERE status = 'pending'
                """
                params: list[Any] = []

                if collection:
                    query += " AND collection = ?"
                    params.append(collection)

                query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
                params.append(batch_size)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                items = []
                queue_ids = []

                for row in rows:
                    queue_ids.append(row["queue_id"])

                # Mark retrieved items as in_progress first
                now = datetime.now(timezone.utc)
                if queue_ids:
                    placeholders = ",".join("?" * len(queue_ids))
                    conn.execute(
                        f"""
                        UPDATE content_ingestion_queue
                        SET status = 'in_progress', started_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE queue_id IN ({placeholders})
                        """,
                        queue_ids,
                    )

                # Now create items with the updated status
                for row in rows:
                    item = ContentIngestionQueueItem(
                        queue_id=row["queue_id"],
                        idempotency_key=row["idempotency_key"],
                        content=row["content"],
                        collection=row["collection"],
                        source_type=row["source_type"],
                        priority=row["priority"],
                        status=ContentIngestionStatus.IN_PROGRESS,  # Status after update
                        main_tag=row["main_tag"],
                        full_tag=row["full_tag"],
                        metadata=self._deserialize_json(row["metadata"]),
                        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                        updated_at=now,  # Updated timestamp
                        started_at=now,  # Started timestamp (set by update)
                        completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                        retry_count=row["retry_count"],
                        max_retries=row["max_retries"],
                        error_message=row["error_message"],
                    )
                    items.append(item)

                logger.debug(f"Dequeued {len(items)} content ingestion items")
                return items

        except Exception as e:
            logger.error(f"Failed to dequeue content ingestion items: {e}")
            raise

    async def update_content_ingestion_status(
        self,
        queue_id: str,
        status: ContentIngestionStatus,
        error_message: str | None = None,
    ) -> bool:
        """
        Update the status of a content ingestion queue item.

        Args:
            queue_id: Queue ID of the item to update
            status: New status (PENDING, IN_PROGRESS, DONE, FAILED)
            error_message: Optional error message (for FAILED status)

        Returns:
            True if item was updated, False if item not found

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            async with self.transaction() as conn:
                update_fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
                params: list[Any] = [status.value]

                if status == ContentIngestionStatus.IN_PROGRESS:
                    update_fields.append("started_at = CURRENT_TIMESTAMP")

                if status in (ContentIngestionStatus.DONE, ContentIngestionStatus.FAILED):
                    update_fields.append("completed_at = CURRENT_TIMESTAMP")

                if error_message:
                    update_fields.append("error_message = ?")
                    params.append(error_message)

                if status == ContentIngestionStatus.FAILED:
                    update_fields.append("retry_count = retry_count + 1")

                params.append(queue_id)

                cursor = conn.execute(
                    f"UPDATE content_ingestion_queue SET {', '.join(update_fields)} WHERE queue_id = ?",
                    params,
                )

                updated = cursor.rowcount > 0

                if updated:
                    logger.debug(f"Updated content ingestion status: {queue_id} -> {status.value}")
                else:
                    logger.warning(f"Content ingestion item not found: {queue_id}")

                return updated

        except Exception as e:
            logger.error(f"Failed to update content ingestion status {queue_id}: {e}")
            raise

    async def get_content_ingestion_queue_depth(
        self,
        collection: str | None = None,
        status: ContentIngestionStatus | None = None,
    ) -> int:
        """
        Get the current depth (count) of the content ingestion queue.

        Args:
            collection: Optional filter by collection name
            status: Optional filter by status

        Returns:
            Number of items in the queue matching the filters

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            query = "SELECT COUNT(*) FROM content_ingestion_queue WHERE 1=1"
            params: list[Any] = []

            if collection:
                query += " AND collection = ?"
                params.append(collection)

            if status:
                query += " AND status = ?"
                params.append(status.value)

            with self._lock:
                cursor = self.connection.execute(query, params)
                count = cursor.fetchone()[0]
                return count

        except Exception as e:
            logger.error(f"Failed to get content ingestion queue depth: {e}")
            raise

    async def reset_in_progress_content_items(
        self,
        max_retries: int = 3,
    ) -> int:
        """
        Reset 'in_progress' content items back to 'pending' for crash recovery.

        Items that have exceeded max_retries are marked as 'failed' instead.

        Args:
            max_retries: Maximum retry count before marking as failed

        Returns:
            Number of items reset

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            reset_count = 0

            async with self.transaction() as conn:
                # Reset items within retry limit back to pending (increment retry_count)
                cursor = conn.execute(
                    """
                    UPDATE content_ingestion_queue
                    SET status = 'pending', started_at = NULL, updated_at = CURRENT_TIMESTAMP,
                        retry_count = retry_count + 1
                    WHERE status = 'in_progress' AND retry_count < ?
                    """,
                    (max_retries,),
                )
                reset_count = cursor.rowcount

                # Mark items that will exceed retry limit after increment as failed
                cursor = conn.execute(
                    """
                    UPDATE content_ingestion_queue
                    SET status = 'failed', error_message = 'Max retries exceeded during crash recovery',
                        completed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP,
                        retry_count = retry_count + 1
                    WHERE status = 'in_progress' AND retry_count >= ?
                    """,
                    (max_retries,),
                )
                failed_count = cursor.rowcount

            if reset_count > 0 or failed_count > 0:
                logger.info(
                    f"Content ingestion crash recovery: reset {reset_count} items, "
                    f"marked {failed_count} as failed"
                )

            return reset_count

        except Exception as e:
            logger.error(f"Failed to reset in-progress content items: {e}")
            raise

    async def remove_completed_content_items(
        self,
        older_than_hours: int = 24,
    ) -> int:
        """
        Remove completed content ingestion items older than specified hours.

        Args:
            older_than_hours: Remove items completed more than this many hours ago

        Returns:
            Number of items removed

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM content_ingestion_queue
                    WHERE status = 'done'
                    AND completed_at < datetime('now', '-' || ? || ' hours')
                    """,
                    (older_than_hours,),
                )
                removed = cursor.rowcount

                if removed > 0:
                    logger.debug(f"Removed {removed} completed content ingestion items")

                return removed

        except Exception as e:
            logger.error(f"Failed to remove completed content items: {e}")
            raise

    # Multi-Component Communication Support Methods

    async def update_processing_state(
        self,
        file_path: str,
        status: str,
        collection_name: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
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
        self, filter_params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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
        metadata: dict[str, Any] | None = None,
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
        self, limit: int | None = None
    ) -> list[dict[str, Any]]:
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
        self, rule_id: str, rule_data: dict[str, Any]
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

    async def get_memory_rules(self) -> list[dict[str, Any]]:
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

    async def record_event(self, event: dict[str, Any]) -> bool:
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
        self, filter_params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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
        self, config_data: dict[str, Any], source: str, timestamp: float | None = None
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

    async def get_configuration_history(self) -> list[dict[str, Any]]:
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
        metadata: dict[str, Any] | None = None,
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
        self, filter_params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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

    async def record_performance_metric(self, metric: dict[str, Any]) -> bool:
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
        self, filter_params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
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

    async def record_resource_usage(self, usage_data: dict[str, Any]) -> bool:
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

    async def get_resource_usage_history(self) -> list[dict[str, Any]]:
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
        file_size: int | None = None,
        file_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
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
        error_message: str | None = None,
        processing_time_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
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
    ) -> FileProcessingRecord | None:
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
        collection: str | None = None,
        limit: int | None = None,
    ) -> list[FileProcessingRecord]:
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
        self, file_path: str, max_retries: int | None = None
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
                     watch_type, library_name,
                     created_at, updated_at, metadata,
                     consecutive_errors, total_errors, last_error_at, last_error_message,
                     backoff_until, last_success_at, health_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           COALESCE((SELECT created_at FROM watch_folders WHERE watch_id = ?), CURRENT_TIMESTAMP),
                           CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        config.watch_type,
                        config.library_name,
                        config.watch_id,
                        self._serialize_json(config.metadata),
                        config.consecutive_errors,
                        config.total_errors,
                        config.last_error_at.isoformat() if config.last_error_at else None,
                        config.last_error_message,
                        config.backoff_until.isoformat() if config.backoff_until else None,
                        config.last_success_at.isoformat() if config.last_success_at else None,
                        config.health_status,
                    ),
                )

            logger.debug(f"Saved watch folder config: {config.watch_id} (type={config.watch_type}, health={config.health_status})")
            return True

        except Exception as e:
            logger.error(f"Failed to save watch folder config {config.watch_id}: {e}")
            return False

    async def get_watch_folder_config(
        self, watch_id: str
    ) -> WatchFolderConfig | None:
        """Get watch folder configuration."""
        try:
            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT watch_id, path, collection, patterns, ignore_patterns, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, enabled,
                           watch_type, library_name,
                           created_at, updated_at, last_scan, metadata,
                           consecutive_errors, total_errors, last_error_at, last_error_message,
                           backoff_until, last_success_at, health_status
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
                    watch_type=row["watch_type"] or "project",
                    library_name=row["library_name"],
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
                    # Error tracking fields (Task 461)
                    consecutive_errors=row["consecutive_errors"] or 0,
                    total_errors=row["total_errors"] or 0,
                    last_error_at=datetime.fromisoformat(
                        row["last_error_at"].replace("Z", "+00:00")
                    )
                    if row["last_error_at"]
                    else None,
                    last_error_message=row["last_error_message"],
                    backoff_until=datetime.fromisoformat(
                        row["backoff_until"].replace("Z", "+00:00")
                    )
                    if row["backoff_until"]
                    else None,
                    last_success_at=datetime.fromisoformat(
                        row["last_success_at"].replace("Z", "+00:00")
                    )
                    if row["last_success_at"]
                    else None,
                    health_status=row["health_status"] or "healthy",
                )

        except Exception as e:
            logger.error(f"Failed to get watch folder config {watch_id}: {e}")
            return None

    async def get_all_watch_folder_configs(
        self, enabled_only: bool = True
    ) -> list[WatchFolderConfig]:
        """Get all watch folder configurations."""
        try:
            with self._lock:
                sql = """
                    SELECT watch_id, path, collection, patterns, ignore_patterns, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, enabled,
                           watch_type, library_name,
                           created_at, updated_at, last_scan, metadata,
                           consecutive_errors, total_errors, last_error_at, last_error_message,
                           backoff_until, last_success_at, health_status
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
                            watch_type=row["watch_type"] or "project",
                            library_name=row["library_name"],
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
                            # Error tracking fields (Task 461)
                            consecutive_errors=row["consecutive_errors"] or 0,
                            total_errors=row["total_errors"] or 0,
                            last_error_at=datetime.fromisoformat(
                                row["last_error_at"].replace("Z", "+00:00")
                            )
                            if row["last_error_at"]
                            else None,
                            last_error_message=row["last_error_message"],
                            backoff_until=datetime.fromisoformat(
                                row["backoff_until"].replace("Z", "+00:00")
                            )
                            if row["backoff_until"]
                            else None,
                            last_success_at=datetime.fromisoformat(
                                row["last_success_at"].replace("Z", "+00:00")
                            )
                            if row["last_success_at"]
                            else None,
                            health_status=row["health_status"] or "healthy",
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

    async def update_watch_folder_error_state(
        self,
        watch_id: str,
        consecutive_errors: int | None = None,
        total_errors: int | None = None,
        last_error_at: datetime | None = None,
        last_error_message: str | None = None,
        backoff_until: datetime | None = None,
        last_success_at: datetime | None = None,
        health_status: str | None = None,
        clear_backoff: bool = False,
    ) -> bool:
        """Update only error tracking fields for a watch folder (Task 461).

        This method efficiently updates error tracking fields without
        reading/writing the entire config. Useful for the Rust daemon
        to update error state frequently.

        Args:
            watch_id: The watch folder identifier
            consecutive_errors: Number of consecutive errors (None = no change)
            total_errors: Cumulative error count (None = no change)
            last_error_at: Timestamp of most recent error (None = no change)
            last_error_message: Description of most recent error (None = no change)
            backoff_until: When to resume after backoff (None = no change)
            last_success_at: Timestamp of most recent success (None = no change)
            health_status: Health state (None = no change)
            clear_backoff: If True, clears backoff_until (sets to NULL)

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            # Build dynamic UPDATE statement
            updates = []
            params = []

            if consecutive_errors is not None:
                updates.append("consecutive_errors = ?")
                params.append(consecutive_errors)

            if total_errors is not None:
                updates.append("total_errors = ?")
                params.append(total_errors)

            if last_error_at is not None:
                updates.append("last_error_at = ?")
                params.append(last_error_at.isoformat())

            if last_error_message is not None:
                updates.append("last_error_message = ?")
                params.append(last_error_message)

            if backoff_until is not None:
                updates.append("backoff_until = ?")
                params.append(backoff_until.isoformat())
            elif clear_backoff:
                updates.append("backoff_until = NULL")

            if last_success_at is not None:
                updates.append("last_success_at = ?")
                params.append(last_success_at.isoformat())

            if health_status is not None:
                if health_status not in ("healthy", "degraded", "backoff", "disabled"):
                    logger.warning(f"Invalid health_status '{health_status}', using 'healthy'")
                    health_status = "healthy"
                updates.append("health_status = ?")
                params.append(health_status)

            if not updates:
                logger.debug(f"No error state updates for watch folder {watch_id}")
                return True

            # Always update updated_at
            updates.append("updated_at = CURRENT_TIMESTAMP")

            sql = f"UPDATE watch_folders SET {', '.join(updates)} WHERE watch_id = ?"
            params.append(watch_id)

            with self._lock:
                cursor = self.connection.execute(sql, params)
                self.connection.commit()

                success = cursor.rowcount > 0
                if success:
                    logger.debug(f"Updated error state for watch folder {watch_id}: health={health_status}")
                else:
                    logger.warning(f"Watch folder not found for error state update: {watch_id}")

                return success

        except Exception as e:
            logger.error(f"Failed to update error state for watch folder {watch_id}: {e}")
            return False

    async def record_watch_folder_error(
        self,
        watch_id: str,
        error_message: str,
        health_status: str = "degraded",
        backoff_until: datetime | None = None,
    ) -> bool:
        """Record an error for a watch folder (Task 461).

        Convenience method that increments error counters and updates
        error tracking fields atomically.

        Args:
            watch_id: The watch folder identifier
            error_message: Description of the error
            health_status: New health status (default: degraded)
            backoff_until: When to resume after backoff (optional)

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            now = datetime.now(timezone.utc)

            # Build UPDATE with increment
            sql = """
                UPDATE watch_folders
                SET consecutive_errors = consecutive_errors + 1,
                    total_errors = total_errors + 1,
                    last_error_at = ?,
                    last_error_message = ?,
                    health_status = ?,
                    backoff_until = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE watch_id = ?
            """
            params = [
                now.isoformat(),
                error_message,
                health_status,
                backoff_until.isoformat() if backoff_until else None,
                watch_id,
            ]

            with self._lock:
                cursor = self.connection.execute(sql, params)
                self.connection.commit()

                success = cursor.rowcount > 0
                if success:
                    logger.debug(f"Recorded error for watch folder {watch_id}: {error_message[:50]}...")
                else:
                    logger.warning(f"Watch folder not found for error recording: {watch_id}")

                return success

        except Exception as e:
            logger.error(f"Failed to record error for watch folder {watch_id}: {e}")
            return False

    async def record_watch_folder_success(self, watch_id: str) -> bool:
        """Record a successful processing for a watch folder (Task 461).

        Resets consecutive error count and clears backoff, keeping
        total_errors for statistics.

        Args:
            watch_id: The watch folder identifier

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            now = datetime.now(timezone.utc)

            sql = """
                UPDATE watch_folders
                SET consecutive_errors = 0,
                    last_success_at = ?,
                    health_status = 'healthy',
                    backoff_until = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE watch_id = ?
            """

            with self._lock:
                cursor = self.connection.execute(sql, [now.isoformat(), watch_id])
                self.connection.commit()

                success = cursor.rowcount > 0
                if success:
                    logger.debug(f"Recorded success for watch folder {watch_id}")
                else:
                    logger.warning(f"Watch folder not found for success recording: {watch_id}")

                return success

        except Exception as e:
            logger.error(f"Failed to record success for watch folder {watch_id}: {e}")
            return False

    async def get_watch_folders_by_health_status(
        self, health_status: str
    ) -> list[WatchFolderConfig]:
        """Get all watch folders with a specific health status (Task 461).

        Args:
            health_status: The health status to filter by (healthy, degraded, backoff, disabled)

        Returns:
            List of watch folder configurations matching the health status
        """
        try:
            if health_status not in ("healthy", "degraded", "backoff", "disabled"):
                logger.warning(f"Invalid health_status filter '{health_status}'")
                return []

            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT watch_id, path, collection, patterns, ignore_patterns, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, enabled,
                           watch_type, library_name,
                           created_at, updated_at, last_scan, metadata,
                           consecutive_errors, total_errors, last_error_at, last_error_message,
                           backoff_until, last_success_at, health_status
                    FROM watch_folders
                    WHERE health_status = ?
                    ORDER BY created_at ASC
                    """,
                    (health_status,),
                )

                rows = cursor.fetchall()
                configs = []

                for row in rows:
                    configs.append(
                        WatchFolderConfig(
                            watch_id=row["watch_id"],
                            path=row["path"],
                            collection=row["collection"],
                            patterns=self._deserialize_json(row["patterns"]) or [],
                            ignore_patterns=self._deserialize_json(row["ignore_patterns"]) or [],
                            auto_ingest=bool(row["auto_ingest"]),
                            recursive=bool(row["recursive"]),
                            recursive_depth=row["recursive_depth"],
                            debounce_seconds=row["debounce_seconds"],
                            enabled=bool(row["enabled"]),
                            watch_type=row["watch_type"] or "project",
                            library_name=row["library_name"],
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
                            consecutive_errors=row["consecutive_errors"] or 0,
                            total_errors=row["total_errors"] or 0,
                            last_error_at=datetime.fromisoformat(
                                row["last_error_at"].replace("Z", "+00:00")
                            )
                            if row["last_error_at"]
                            else None,
                            last_error_message=row["last_error_message"],
                            backoff_until=datetime.fromisoformat(
                                row["backoff_until"].replace("Z", "+00:00")
                            )
                            if row["backoff_until"]
                            else None,
                            last_success_at=datetime.fromisoformat(
                                row["last_success_at"].replace("Z", "+00:00")
                            )
                            if row["last_success_at"]
                            else None,
                            health_status=row["health_status"] or "healthy",
                        )
                    )

                return configs

        except Exception as e:
            logger.error(f"Failed to get watch folders by health status {health_status}: {e}")
            return []

    async def get_watch_folders_in_backoff(self) -> list[WatchFolderConfig]:
        """Get all watch folders currently in backoff state (Task 461).

        Returns watch folders where backoff_until is set and in the future.

        Returns:
            List of watch folder configurations in backoff
        """
        try:
            now = datetime.now(timezone.utc)

            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT watch_id, path, collection, patterns, ignore_patterns, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, enabled,
                           watch_type, library_name,
                           created_at, updated_at, last_scan, metadata,
                           consecutive_errors, total_errors, last_error_at, last_error_message,
                           backoff_until, last_success_at, health_status
                    FROM watch_folders
                    WHERE backoff_until IS NOT NULL AND backoff_until > ?
                    ORDER BY backoff_until ASC
                    """,
                    (now.isoformat(),),
                )

                rows = cursor.fetchall()
                configs = []

                for row in rows:
                    configs.append(
                        WatchFolderConfig(
                            watch_id=row["watch_id"],
                            path=row["path"],
                            collection=row["collection"],
                            patterns=self._deserialize_json(row["patterns"]) or [],
                            ignore_patterns=self._deserialize_json(row["ignore_patterns"]) or [],
                            auto_ingest=bool(row["auto_ingest"]),
                            recursive=bool(row["recursive"]),
                            recursive_depth=row["recursive_depth"],
                            debounce_seconds=row["debounce_seconds"],
                            enabled=bool(row["enabled"]),
                            watch_type=row["watch_type"] or "project",
                            library_name=row["library_name"],
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
                            consecutive_errors=row["consecutive_errors"] or 0,
                            total_errors=row["total_errors"] or 0,
                            last_error_at=datetime.fromisoformat(
                                row["last_error_at"].replace("Z", "+00:00")
                            )
                            if row["last_error_at"]
                            else None,
                            last_error_message=row["last_error_message"],
                            backoff_until=datetime.fromisoformat(
                                row["backoff_until"].replace("Z", "+00:00")
                            )
                            if row["backoff_until"]
                            else None,
                            last_success_at=datetime.fromisoformat(
                                row["last_success_at"].replace("Z", "+00:00")
                            )
                            if row["last_success_at"]
                            else None,
                            health_status=row["health_status"] or "healthy",
                        )
                    )

                return configs

        except Exception as e:
            logger.error(f"Failed to get watch folders in backoff: {e}")
            return []

    # Processing Queue Management

    async def add_to_processing_queue(
        self,
        file_path: str,
        collection: str,
        priority: ProcessingPriority = ProcessingPriority.NORMAL,
        scheduled_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
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

    async def list_watch_folders(
        self, enabled_only: bool = True
    ) -> list[WatchFolderConfig]:
        """List all watch folder configs (alias for get_all_watch_folder_configs)."""
        return await self.get_all_watch_folder_configs(enabled_only=enabled_only)

    # =========================================================================
    # Library Watch Management (Multi-Tenant Architecture v7)
    # =========================================================================

    async def save_library_watch(
        self,
        library_name: str,
        path: str,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        recursive: bool = True,
        recursive_depth: int = 10,
        debounce_seconds: float = 2.0,
        enabled: bool = True,
        metadata: dict | None = None,
    ) -> bool:
        """
        Save or update a library watch configuration.

        Args:
            library_name: Unique identifier for the library (e.g., "color-science")
            path: Absolute path to the library folder
            patterns: File patterns to include (default: PDF, EPUB, MD, TXT)
            ignore_patterns: Patterns to exclude (default: .git, __pycache__)
            recursive: Watch subdirectories
            recursive_depth: Maximum recursion depth
            debounce_seconds: Wait time before processing changes
            enabled: Whether the watch is active
            metadata: Optional JSON metadata

        Returns:
            True if saved successfully
        """
        if patterns is None:
            patterns = ["*.pdf", "*.epub", "*.md", "*.txt"]
        if ignore_patterns is None:
            ignore_patterns = [".git/*", "__pycache__/*"]

        try:
            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO library_watches (
                        library_name, path, patterns, ignore_patterns,
                        enabled, recursive, recursive_depth, debounce_seconds,
                        added_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                    ON CONFLICT(library_name) DO UPDATE SET
                        path = excluded.path,
                        patterns = excluded.patterns,
                        ignore_patterns = excluded.ignore_patterns,
                        enabled = excluded.enabled,
                        recursive = excluded.recursive,
                        recursive_depth = excluded.recursive_depth,
                        debounce_seconds = excluded.debounce_seconds,
                        metadata = excluded.metadata
                    """,
                    (
                        library_name,
                        path,
                        json.dumps(patterns),
                        json.dumps(ignore_patterns),
                        1 if enabled else 0,
                        1 if recursive else 0,
                        recursive_depth,
                        debounce_seconds,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
            logger.info(f"Saved library watch: {library_name} -> {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save library watch {library_name}: {e}")
            return False

    async def get_library_watch(self, library_name: str) -> dict | None:
        """Get a library watch configuration by name."""
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "SELECT * FROM library_watches WHERE library_name = ?",
                    (library_name,),
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "library_name": row["library_name"],
                        "path": row["path"],
                        "patterns": json.loads(row["patterns"]),
                        "ignore_patterns": json.loads(row["ignore_patterns"]),
                        "enabled": bool(row["enabled"]),
                        "recursive": bool(row["recursive"]),
                        "recursive_depth": row["recursive_depth"],
                        "debounce_seconds": row["debounce_seconds"],
                        "added_at": row["added_at"],
                        "last_scan": row["last_scan"],
                        "document_count": row["document_count"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get library watch {library_name}: {e}")
            return None

    async def list_library_watches(self, enabled_only: bool = True) -> list[dict]:
        """List all library watch configurations."""
        try:
            async with self.transaction() as conn:
                query = "SELECT * FROM library_watches"
                if enabled_only:
                    query += " WHERE enabled = 1"
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                return [
                    {
                        "library_name": row["library_name"],
                        "path": row["path"],
                        "patterns": json.loads(row["patterns"]),
                        "ignore_patterns": json.loads(row["ignore_patterns"]),
                        "enabled": bool(row["enabled"]),
                        "recursive": bool(row["recursive"]),
                        "recursive_depth": row["recursive_depth"],
                        "debounce_seconds": row["debounce_seconds"],
                        "added_at": row["added_at"],
                        "last_scan": row["last_scan"],
                        "document_count": row["document_count"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to list library watches: {e}")
            return []

    async def remove_library_watch(self, library_name: str) -> bool:
        """Remove a library watch configuration."""
        try:
            async with self.transaction() as conn:
                conn.execute(
                    "DELETE FROM library_watches WHERE library_name = ?",
                    (library_name,),
                )
            logger.info(f"Removed library watch: {library_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove library watch {library_name}: {e}")
            return False

    async def update_library_watch_stats(
        self, library_name: str, document_count: int | None = None
    ) -> bool:
        """Update library watch statistics (last_scan, document_count)."""
        try:
            async with self.transaction() as conn:
                if document_count is not None:
                    conn.execute(
                        """
                        UPDATE library_watches
                        SET last_scan = CURRENT_TIMESTAMP, document_count = ?
                        WHERE library_name = ?
                        """,
                        (document_count, library_name),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE library_watches
                        SET last_scan = CURRENT_TIMESTAMP
                        WHERE library_name = ?
                        """,
                        (library_name,),
                    )
            return True
        except Exception as e:
            logger.error(f"Failed to update library watch stats {library_name}: {e}")
            return False

    # =========================================================================
    # Project Session Management (Multi-Tenant Architecture v7)
    # =========================================================================

    async def register_project(
        self,
        project_id: str,
        path: str,
        name: str | None = None,
        git_remote: str | None = None,
    ) -> bool:
        """
        Register a project for multi-tenant tracking.

        Called by MCP server when agent starts in a project folder.

        Args:
            project_id: 12-character hex identifier
            path: Absolute path to project root
            name: Human-readable project name (derived from folder/git if not provided)
            git_remote: Normalized git remote URL if available

        Returns:
            True if registered/updated successfully
        """
        try:
            if name is None:
                name = path.rstrip("/").split("/")[-1]  # Use folder name

            async with self.transaction() as conn:
                # Check if project exists by project_id
                cursor = conn.execute(
                    "SELECT id FROM projects WHERE project_id = ?",
                    (project_id,),
                )
                existing = cursor.fetchone()

                if existing:
                    # Update existing project - increment sessions and set high priority
                    conn.execute(
                        """
                        UPDATE projects
                        SET active_sessions = active_sessions + 1,
                            priority = 'high',
                            last_active = CURRENT_TIMESTAMP,
                            git_remote = COALESCE(?, git_remote),
                            updated_at = CURRENT_TIMESTAMP
                        WHERE project_id = ?
                        """,
                        (git_remote, project_id),
                    )
                    logger.info(f"Updated project registration: {project_id} (incremented sessions)")
                else:
                    # Create new project registration
                    conn.execute(
                        """
                        INSERT INTO projects (
                            name, root_path, collection_name, project_id,
                            priority, active_sessions, git_remote,
                            registered_at, last_active, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, 'high', 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """,
                        (name, path, f"projects", project_id, git_remote),
                    )
                    logger.info(f"Registered new project: {project_id} at {path}")

            return True
        except Exception as e:
            logger.error(f"Failed to register project {project_id}: {e}")
            return False

    async def deprioritize_project(self, project_id: str) -> tuple[bool, int, str]:
        """
        Deprioritize a project when agent session ends.

        Called by MCP server on shutdown.

        Args:
            project_id: 12-character hex identifier

        Returns:
            Tuple of (success, remaining_sessions, new_priority)
        """
        try:
            async with self.transaction() as conn:
                # Get current session count
                cursor = conn.execute(
                    "SELECT active_sessions FROM projects WHERE project_id = ?",
                    (project_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False, 0, "normal"

                current_sessions = row["active_sessions"]
                new_sessions = max(0, current_sessions - 1)
                new_priority = "high" if new_sessions > 0 else "normal"

                conn.execute(
                    """
                    UPDATE projects
                    SET active_sessions = ?,
                        priority = ?,
                        last_active = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE project_id = ?
                    """,
                    (new_sessions, new_priority, project_id),
                )

            logger.info(f"Deprioritized project {project_id}: sessions={new_sessions}, priority={new_priority}")
            return True, new_sessions, new_priority
        except Exception as e:
            logger.error(f"Failed to deprioritize project {project_id}: {e}")
            return False, 0, "normal"

    async def get_project_by_id(self, project_id: str) -> dict | None:
        """Get project details by project_id."""
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "SELECT * FROM projects WHERE project_id = ?",
                    (project_id,),
                )
                row = cursor.fetchone()
                if row:
                    return {
                        "id": row["id"],
                        "name": row["name"],
                        "root_path": row["root_path"],
                        "collection_name": row["collection_name"],
                        "project_id": row["project_id"],
                        "priority": row["priority"],
                        "active_sessions": row["active_sessions"],
                        "git_remote": row["git_remote"],
                        "registered_at": row["registered_at"],
                        "last_active": row["last_active"],
                        "lsp_enabled": bool(row["lsp_enabled"]),
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get project {project_id}: {e}")
            return None

    async def list_projects_by_priority(
        self, priority: str | None = None
    ) -> list[dict]:
        """
        List projects, optionally filtered by priority.

        Args:
            priority: Filter by priority ('high', 'normal', 'low') or None for all

        Returns:
            List of project dictionaries
        """
        try:
            async with self.transaction() as conn:
                if priority:
                    cursor = conn.execute(
                        "SELECT * FROM projects WHERE priority = ? ORDER BY last_active DESC",
                        (priority,),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM projects ORDER BY priority DESC, last_active DESC"
                    )
                rows = cursor.fetchall()
                return [
                    {
                        "id": row["id"],
                        "name": row["name"],
                        "root_path": row["root_path"],
                        "project_id": row["project_id"],
                        "priority": row["priority"],
                        "active_sessions": row["active_sessions"],
                        "git_remote": row["git_remote"],
                        "registered_at": row["registered_at"],
                        "last_active": row["last_active"],
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            return []

    async def get_high_priority_projects(self) -> list[dict]:
        """Get all projects with high priority (active agent sessions)."""
        return await self.list_projects_by_priority(priority="high")

