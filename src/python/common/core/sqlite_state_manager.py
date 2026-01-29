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
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from ..observability.metrics import (
    record_dual_write_failure,
    record_dual_write_success,
    record_processing_duration,
    record_queue_enqueue,
    record_queue_processed,
    record_queue_retry,
    record_wait_duration,
)
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


class UnifiedQueueItemType(Enum):
    """Item types for the unified queue (Task 22/24).

    Cross-language compatible with Rust ItemType enum.
    """

    CONTENT = "content"
    FILE = "file"
    FOLDER = "folder"
    PROJECT = "project"
    LIBRARY = "library"
    DELETE_TENANT = "delete_tenant"
    DELETE_DOCUMENT = "delete_document"
    RENAME = "rename"


class UnifiedQueueOperation(Enum):
    """Operation types for the unified queue (Task 22/24).

    Cross-language compatible with Rust QueueOperation enum.
    """

    INGEST = "ingest"
    UPDATE = "update"
    DELETE = "delete"
    SCAN = "scan"

    def is_valid_for(self, item_type: "UnifiedQueueItemType") -> bool:
        """Check if this operation is valid for the given item type.

        Matches Rust's QueueOperation::is_valid_for() implementation.
        """
        validity_map = {
            UnifiedQueueItemType.CONTENT: {
                UnifiedQueueOperation.INGEST,
                UnifiedQueueOperation.UPDATE,
                UnifiedQueueOperation.DELETE,
            },
            UnifiedQueueItemType.FILE: {
                UnifiedQueueOperation.INGEST,
                UnifiedQueueOperation.UPDATE,
                UnifiedQueueOperation.DELETE,
            },
            UnifiedQueueItemType.FOLDER: {
                UnifiedQueueOperation.INGEST,
                UnifiedQueueOperation.DELETE,
                UnifiedQueueOperation.SCAN,
            },
            UnifiedQueueItemType.PROJECT: {
                UnifiedQueueOperation.INGEST,
                UnifiedQueueOperation.DELETE,
                UnifiedQueueOperation.SCAN,
            },
            UnifiedQueueItemType.LIBRARY: {
                UnifiedQueueOperation.INGEST,
                UnifiedQueueOperation.UPDATE,
                UnifiedQueueOperation.DELETE,
            },
            UnifiedQueueItemType.DELETE_TENANT: {UnifiedQueueOperation.DELETE},
            UnifiedQueueItemType.DELETE_DOCUMENT: {UnifiedQueueOperation.DELETE},
            UnifiedQueueItemType.RENAME: {UnifiedQueueOperation.UPDATE},
        }
        return self in validity_map.get(item_type, set())


class IdempotencyKeyError(Exception):
    """Errors that can occur during idempotency key generation."""

    pass


def generate_unified_idempotency_key(
    item_type: UnifiedQueueItemType | str,
    op: UnifiedQueueOperation | str,
    tenant_id: str,
    collection: str,
    payload: dict[str, Any] | str,
) -> str:
    """Generate a comprehensive idempotency key for unified queue deduplication.

    Creates a deterministic key from all relevant queue item attributes to prevent
    duplicate processing. This function is cross-language compatible with the
    matching Rust implementation in unified_queue_schema.rs.

    Format:
        Input string: {item_type}|{op}|{tenant_id}|{collection}|{payload_json}
        Output: SHA256 hash truncated to 32 hex characters

    Args:
        item_type: Type of queue item (content, file, folder, etc.)
        op: Operation type (ingest, update, delete, scan)
        tenant_id: Project/tenant identifier
        collection: Target Qdrant collection name
        payload: Payload dict (will be JSON serialized with sorted keys) or JSON string

    Returns:
        32-character hexadecimal string

    Raises:
        IdempotencyKeyError: If tenant_id or collection is empty, or operation invalid

    Example:
        >>> key = generate_unified_idempotency_key(
        ...     UnifiedQueueItemType.FILE,
        ...     UnifiedQueueOperation.INGEST,
        ...     "proj_abc123",
        ...     "my-project-code",
        ...     {"file_path": "/path/to/file.rs"}
        ... )
        >>> len(key)
        32
    """
    # Convert enums to strings if needed
    if isinstance(item_type, UnifiedQueueItemType):
        item_type_str = item_type.value
    else:
        item_type_str = str(item_type)

    if isinstance(op, UnifiedQueueOperation):
        op_str = op.value
    else:
        op_str = str(op)

    # Validate inputs
    if not tenant_id:
        raise IdempotencyKeyError("tenant_id cannot be empty")
    if not collection:
        raise IdempotencyKeyError("collection cannot be empty")

    # Validate operation for item type
    try:
        item_type_enum = (
            item_type
            if isinstance(item_type, UnifiedQueueItemType)
            else UnifiedQueueItemType(item_type_str)
        )
        op_enum = (
            op if isinstance(op, UnifiedQueueOperation) else UnifiedQueueOperation(op_str)
        )
        if not op_enum.is_valid_for(item_type_enum):
            raise IdempotencyKeyError(
                f"operation '{op_str}' is not valid for item type '{item_type_str}'"
            )
    except ValueError:
        # If enum conversion fails, skip validation (allows flexibility)
        pass

    # Convert payload to sorted JSON string if it's a dict
    if isinstance(payload, dict):
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    else:
        payload_json = str(payload)

    # Construct canonical input string (must match Rust implementation exactly)
    input_string = f"{item_type_str}|{op_str}|{tenant_id}|{collection}|{payload_json}"

    # Hash and truncate to 32 hex chars (16 bytes)
    hash_result = hashlib.sha256(input_string.encode("utf-8")).hexdigest()[:32]

    return hash_result


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
class UnifiedQueueItem:
    """Item in the unified queue (Task 37).

    Represents a queue item that consolidates content_ingestion_queue and
    ingestion_queue into a single unified_queue table. This is the canonical
    queue item type for all new queue operations.

    Supports lease-based processing for concurrent workers.
    """

    queue_id: str  # UUID for queue item
    idempotency_key: str  # SHA256 hash for deduplication
    item_type: str  # content, file, folder, project, library, delete_tenant, etc.
    op: str  # ingest, update, delete, scan
    tenant_id: str  # Project/tenant identifier
    collection: str  # Target collection name
    priority: int = 5  # 0-10, default 5 (medium)
    status: str = "pending"  # pending, in_progress, done, failed
    branch: str = "main"  # Git branch context
    payload_json: str = "{}"  # JSON-encoded payload
    metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    lease_until: datetime | None = None  # Lease expiry for in_progress items
    worker_id: str | None = None  # Worker processing this item
    retry_count: int = 0
    max_retries: int = 3
    error_message: str | None = None
    last_error_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at

    @property
    def payload(self) -> dict[str, Any]:
        """Parse payload_json into a dictionary."""
        if isinstance(self.payload_json, dict):
            return self.payload_json
        return json.loads(self.payload_json) if self.payload_json else {}

    def is_lease_expired(self) -> bool:
        """Check if the lease has expired."""
        if self.lease_until is None:
            return True
        return datetime.now(timezone.utc) > self.lease_until


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

    Priority adjustment (Task 461.17):
    - watch_priority: Priority level 0-10 (default: 5)
      - degraded watches get -1 priority
      - backoff watches get -2 priority
      - healthy watches with recent successes get +1 priority boost
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
    # Priority adjustment fields (Task 461.17)
    watch_priority: int = 5  # 0-10, default 5

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
        # Validate watch_priority (Task 461.17)
        if not isinstance(self.watch_priority, int) or self.watch_priority < 0:
            self.watch_priority = 5
        elif self.watch_priority > 10:
            self.watch_priority = 10

    def calculate_effective_priority(self) -> int:
        """Calculate effective priority based on health status (Task 461.17).

        Returns:
            Effective priority (0-10) adjusted for health status
        """
        base_priority = self.watch_priority

        # Adjust based on health status
        if self.health_status == "degraded":
            base_priority -= 1
        elif self.health_status == "backoff":
            base_priority -= 2
        elif self.health_status == "disabled":
            base_priority = 0  # Lowest priority
        elif self.health_status == "healthy":
            # Boost priority if recent success and no errors
            if self.last_success_at and self.consecutive_errors == 0:
                # Check if success was recent (within last hour)
                now = datetime.now(timezone.utc)
                if (now - self.last_success_at).total_seconds() < 3600:
                    base_priority += 1

        # Clamp to valid range
        return max(0, min(10, base_priority))


class ErrorPatternType(Enum):
    """Types of error patterns that can be detected (Task 461.18)."""

    FILE_REPEATED = "file_repeated"  # Same file failing repeatedly
    FILE_TYPE = "file_type"  # Specific file types causing errors
    TIME_BASED = "time_based"  # Time-based patterns (e.g., network issues)
    NETWORK = "network"  # Network-related errors
    PERMISSION = "permission"  # Permission-related errors


class ExclusionType(Enum):
    """Types of exclusions for files that should not be processed (Task 461.18)."""

    FILE = "file"  # Specific file path
    PATTERN = "pattern"  # Glob pattern (e.g., *.corrupted)
    DIRECTORY = "directory"  # Entire directory


@dataclass
class ErrorPattern:
    """Record for tracking error patterns (Task 461.18).

    Stores detected patterns in errors to identify systematic vs transient failures.
    """

    id: int | None
    watch_id: str
    pattern_type: ErrorPatternType
    pattern_key: str  # File path, extension, time window, etc.
    occurrence_count: int = 1
    first_seen_at: datetime = None
    last_seen_at: datetime = None
    is_systematic: bool = False  # True if pattern indicates systematic failure
    confidence_score: float = 0.0  # 0.0-1.0 confidence in pattern
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.first_seen_at is None:
            self.first_seen_at = datetime.now(timezone.utc)
        if self.last_seen_at is None:
            self.last_seen_at = self.first_seen_at


@dataclass
class WatchExclusion:
    """Record for files/patterns excluded from processing (Task 461.18).

    Systematic failures result in permanent exclusions.
    Transient failures may have temporary exclusions with expiry.
    """

    id: int | None
    watch_id: str
    exclusion_type: ExclusionType
    exclusion_value: str  # File path, pattern, or directory
    reason: str  # Why this was excluded
    error_count: int = 1  # Number of errors that led to exclusion
    created_at: datetime = None
    expires_at: datetime | None = None  # None for permanent exclusions
    is_permanent: bool = False
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        """Check if this exclusion has expired."""
        if self.is_permanent or self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at


class DegradationLevel(Enum):
    """System degradation level (Task 461.19)."""

    NORMAL = "normal"  # Operating normally
    LIGHT = "light"  # Slight load increase, minor adjustments
    MODERATE = "moderate"  # Moderate load, throttling enabled
    SEVERE = "severe"  # High load, aggressive throttling
    CRITICAL = "critical"  # System overloaded, emergency measures


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation (Task 461.19)."""

    # Queue depth thresholds
    queue_depth_light: int = 1000  # Enter light degradation
    queue_depth_moderate: int = 3000  # Enter moderate degradation
    queue_depth_severe: int = 5000  # Enter severe degradation
    queue_depth_critical: int = 10000  # Enter critical degradation

    # Throughput thresholds (as percentage of target)
    throughput_target: float = 100.0  # Items per second target
    throughput_warning: float = 0.75  # 75% of target
    throughput_critical: float = 0.50  # 50% of target

    # Memory thresholds (as percentage of available)
    memory_warning: float = 0.70  # 70% memory usage
    memory_critical: float = 0.85  # 85% memory usage

    # Polling interval adjustments (multipliers)
    polling_interval_light: float = 1.5  # 50% slower
    polling_interval_moderate: float = 2.0  # 2x slower
    polling_interval_severe: float = 4.0  # 4x slower
    polling_interval_critical: float = 10.0  # 10x slower

    # Priority thresholds for pausing watches
    pause_priority_moderate: int = 3  # Pause watches with priority <= 3 in moderate
    pause_priority_severe: int = 5  # Pause watches with priority <= 5 in severe
    pause_priority_critical: int = 8  # Pause watches with priority <= 8 in critical

    # Recovery parameters
    recovery_cooldown_seconds: int = 60  # Wait before recovering
    recovery_steps: int = 3  # Number of steps to full recovery


@dataclass
class DegradationState:
    """Current state of graceful degradation (Task 461.19)."""

    level: DegradationLevel = DegradationLevel.NORMAL
    entered_at: datetime = None
    last_check_at: datetime = None
    queue_depth: int = 0
    throughput: float = 0.0
    memory_usage: float = 0.0
    paused_watch_ids: list[str] = None
    polling_interval_multiplier: float = 1.0
    recovery_step: int = 0  # 0 = not recovering, 1-3 = in recovery

    def __post_init__(self):
        if self.entered_at is None:
            self.entered_at = datetime.now(timezone.utc)
        if self.last_check_at is None:
            self.last_check_at = self.entered_at
        if self.paused_watch_ids is None:
            self.paused_watch_ids = []

    def is_degraded(self) -> bool:
        """Check if system is in any degraded state."""
        return self.level != DegradationLevel.NORMAL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "entered_at": self.entered_at.isoformat() if self.entered_at else None,
            "last_check_at": self.last_check_at.isoformat() if self.last_check_at else None,
            "queue_depth": self.queue_depth,
            "throughput": self.throughput,
            "memory_usage": self.memory_usage,
            "paused_watch_ids": self.paused_watch_ids,
            "polling_interval_multiplier": self.polling_interval_multiplier,
            "recovery_step": self.recovery_step,
        }


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


@dataclass
class ActiveProjectState:
    """Active project state for fairness scheduler (Task 36 - code audit round 2).

    Tracks currently active projects for queue priority scheduling.
    Projects with recent activity get higher priority in the processing queue.

    Used by:
    - Unified queue processor: Updates items_processed_count and last_activity_at
    - Watch folder scanner: Registers project activity
    - Garbage collector: Removes stale projects (inactive > 24 hours)
    - Fairness scheduler: Determines processing priority
    """

    project_id: str  # Unique identifier (typically tenant_id or normalized path)
    tenant_id: str  # Tenant identifier for multi-tenant isolation
    last_activity_at: datetime = None
    items_processed_count: int = 0
    items_in_queue: int = 0
    watch_enabled: bool = False
    watch_folder_id: str | None = None
    created_at: datetime = None
    updated_at: datetime = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.last_activity_at is None:
            self.last_activity_at = self.created_at


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

    BASE_SCHEMA_VERSION = 12  # Base schema in _create_initial_schema (unified_queue added via migration)
    SCHEMA_VERSION = 14  # v14: Add active_projects table (Task 36 - code audit round 2)
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
            # Use ~/.workspace-qdrant/state.db to match Rust daemon
            # This ensures Python MCP server and Rust daemon share the same database
            home = Path.home()
            wq_dir = home / ".workspace-qdrant"
            wq_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = wq_dir / "state.db"
            logger.info(f"Using shared state database: {self.db_path}")
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

            # Ensure all expected columns exist (handles partial migrations)
            await self._ensure_schema_columns()

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
                if self.BASE_SCHEMA_VERSION < self.SCHEMA_VERSION:
                    await self._migrate_schema(self.BASE_SCHEMA_VERSION, self.SCHEMA_VERSION)
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
                health_status TEXT NOT NULL DEFAULT 'healthy',  -- healthy, degraded, backoff, disabled
                -- Priority adjustment (Task 461.17)
                watch_priority INTEGER NOT NULL DEFAULT 5 CHECK (watch_priority >= 0 AND watch_priority <= 10)
            )
            """,
            # Indexes for watch_folders
            "CREATE INDEX idx_watch_folders_path ON watch_folders(path)",
            "CREATE INDEX idx_watch_folders_enabled ON watch_folders(enabled)",
            "CREATE INDEX idx_watch_folders_collection ON watch_folders(collection)",
            "CREATE INDEX idx_watch_folders_priority ON watch_folders(watch_priority DESC)",
            "CREATE INDEX idx_watch_folders_health_status ON watch_folders(health_status)",
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
            # Error pattern detection tables (v11, Task 461.18)
            """
            CREATE TABLE watch_error_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL CHECK (pattern_type IN ('file_repeated', 'file_type', 'time_based', 'network', 'permission')),
                pattern_key TEXT NOT NULL,
                occurrence_count INTEGER NOT NULL DEFAULT 1,
                first_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_systematic BOOLEAN NOT NULL DEFAULT 0,
                confidence_score REAL NOT NULL DEFAULT 0.0,
                metadata TEXT,
                UNIQUE(watch_id, pattern_type, pattern_key),
                FOREIGN KEY (watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
            )
            """,
            # Indexes for watch_error_patterns
            "CREATE INDEX idx_error_patterns_watch_id ON watch_error_patterns(watch_id)",
            "CREATE INDEX idx_error_patterns_type ON watch_error_patterns(pattern_type)",
            "CREATE INDEX idx_error_patterns_systematic ON watch_error_patterns(is_systematic)",
            "CREATE INDEX idx_error_patterns_last_seen ON watch_error_patterns(last_seen_at)",
            # Watch exclusions for permanently excluded files (v11, Task 461.18)
            """
            CREATE TABLE watch_exclusions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                watch_id TEXT NOT NULL,
                exclusion_type TEXT NOT NULL CHECK (exclusion_type IN ('file', 'pattern', 'directory')),
                exclusion_value TEXT NOT NULL,
                reason TEXT NOT NULL,
                error_count INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                is_permanent BOOLEAN NOT NULL DEFAULT 0,
                metadata TEXT,
                UNIQUE(watch_id, exclusion_type, exclusion_value),
                FOREIGN KEY (watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
            )
            """,
            # Indexes for watch_exclusions
            "CREATE INDEX idx_exclusions_watch_id ON watch_exclusions(watch_id)",
            "CREATE INDEX idx_exclusions_type ON watch_exclusions(exclusion_type)",
            "CREATE INDEX idx_exclusions_permanent ON watch_exclusions(is_permanent)",
            "CREATE INDEX idx_exclusions_expires ON watch_exclusions(expires_at)",
            # Dead letter queue for permanently failed items (v12, Task 14)
            """
            CREATE TABLE dead_letter_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                tenant_id TEXT,
                branch TEXT,
                operation TEXT NOT NULL DEFAULT 'ingest',
                error_category TEXT NOT NULL CHECK (error_category IN ('permanent', 'max_retries', 'circuit_breaker')),
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                original_priority INTEGER DEFAULT 5,
                retry_count INTEGER NOT NULL DEFAULT 0,
                retry_history TEXT,  -- JSON array of retry attempts
                failed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                original_queued_at TIMESTAMP,
                reprocessed_at TIMESTAMP,
                reprocess_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT  -- JSON for additional context
            )
            """,
            # Indexes for dead_letter_queue
            "CREATE INDEX idx_dlq_error_category ON dead_letter_queue(error_category)",
            "CREATE INDEX idx_dlq_error_type ON dead_letter_queue(error_type)",
            "CREATE INDEX idx_dlq_collection ON dead_letter_queue(collection_name)",
            "CREATE INDEX idx_dlq_failed_at ON dead_letter_queue(failed_at)",
            "CREATE INDEX idx_dlq_file_path ON dead_letter_queue(file_path)",
            # Insert initial schema version
            f"INSERT INTO schema_version (version) VALUES ({self.BASE_SCHEMA_VERSION})",
        ]

        with self.connection:
            for sql in schema_sql:
                self.connection.execute(sql)

        logger.info(f"Created database schema version {self.BASE_SCHEMA_VERSION}")

    def _load_migration_statements(self, filename: str) -> list[str]:
        migrations_dir = Path(__file__).parent / "migrations"
        migration_file = migrations_dir / filename
        if not migration_file.exists():
            raise FileNotFoundError(f"Missing migration file: {migration_file}")

        raw_sql = migration_file.read_text()
        statements: list[str] = []
        buffer: list[str] = []

        for line in raw_sql.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            buffer.append(line)
            if stripped.endswith(";"):
                statement = "\n".join(buffer).strip().rstrip(";").strip()
                if statement:
                    statements.append(statement)
                buffer = []

        if buffer:
            statement = "\n".join(buffer).strip().rstrip(";").strip()
            if statement:
                statements.append(statement)

        if not statements:
            raise ValueError(f"No SQL statements found in migration file: {migration_file}")

        return statements

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
                    # Add watch_type and library_name columns (Task 402 - multi-tenant routing)
                    "ALTER TABLE watch_folders ADD COLUMN watch_type TEXT NOT NULL DEFAULT 'project'",
                    "ALTER TABLE watch_folders ADD COLUMN library_name TEXT",
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

            # Migrate from version 9 to version 10 - Watch priority adjustment (Task 461.17)
            if from_version <= 9 and to_version >= 10:
                logger.info("Applying migration: v9 -> v10 (watch priority adjustment)")
                priority_sql = [
                    # Add watch_priority column to watch_folders table
                    "ALTER TABLE watch_folders ADD COLUMN watch_priority INTEGER NOT NULL DEFAULT 5",
                    # Add index for priority-based queries
                    "CREATE INDEX IF NOT EXISTS idx_watch_folders_priority ON watch_folders(watch_priority DESC)",
                ]

                for sql in priority_sql:
                    try:
                        conn.execute(sql)
                    except Exception as e:
                        # Ignore if column already exists (idempotent migration)
                        if "duplicate column" not in str(e).lower():
                            raise

                logger.info("Successfully migrated to schema version 10 (watch priority adjustment)")

            # Migrate from version 10 to version 11 - Error pattern detection (Task 461.18)
            if from_version <= 10 and to_version >= 11:
                logger.info("Applying migration: v10 -> v11 (error pattern detection)")
                pattern_sql = [
                    # Error pattern detection table
                    """
                    CREATE TABLE IF NOT EXISTS watch_error_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        watch_id TEXT NOT NULL,
                        pattern_type TEXT NOT NULL CHECK (pattern_type IN ('file_repeated', 'file_type', 'time_based', 'network', 'permission')),
                        pattern_key TEXT NOT NULL,
                        occurrence_count INTEGER NOT NULL DEFAULT 1,
                        first_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        is_systematic BOOLEAN NOT NULL DEFAULT 0,
                        confidence_score REAL NOT NULL DEFAULT 0.0,
                        metadata TEXT,
                        UNIQUE(watch_id, pattern_type, pattern_key),
                        FOREIGN KEY (watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
                    )
                    """,
                    "CREATE INDEX IF NOT EXISTS idx_error_patterns_watch_id ON watch_error_patterns(watch_id)",
                    "CREATE INDEX IF NOT EXISTS idx_error_patterns_type ON watch_error_patterns(pattern_type)",
                    "CREATE INDEX IF NOT EXISTS idx_error_patterns_systematic ON watch_error_patterns(is_systematic)",
                    "CREATE INDEX IF NOT EXISTS idx_error_patterns_last_seen ON watch_error_patterns(last_seen_at)",
                    # Watch exclusions table for permanently excluded files
                    """
                    CREATE TABLE IF NOT EXISTS watch_exclusions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        watch_id TEXT NOT NULL,
                        exclusion_type TEXT NOT NULL CHECK (exclusion_type IN ('file', 'pattern', 'directory')),
                        exclusion_value TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        error_count INTEGER NOT NULL DEFAULT 1,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        is_permanent BOOLEAN NOT NULL DEFAULT 0,
                        metadata TEXT,
                        UNIQUE(watch_id, exclusion_type, exclusion_value),
                        FOREIGN KEY (watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
                    )
                    """,
                    "CREATE INDEX IF NOT EXISTS idx_exclusions_watch_id ON watch_exclusions(watch_id)",
                    "CREATE INDEX IF NOT EXISTS idx_exclusions_type ON watch_exclusions(exclusion_type)",
                    "CREATE INDEX IF NOT EXISTS idx_exclusions_permanent ON watch_exclusions(is_permanent)",
                    "CREATE INDEX IF NOT EXISTS idx_exclusions_expires ON watch_exclusions(expires_at)",
                ]

                for sql in pattern_sql:
                    try:
                        conn.execute(sql)
                    except Exception as e:
                        # Ignore if table already exists (idempotent migration)
                        if "already exists" not in str(e).lower():
                            raise

                logger.info("Successfully migrated to schema version 11 (error pattern detection)")

            # Migrate from version 11 to version 12 - Add dead_letter_queue table
            if from_version < 12 <= to_version:
                dlq_sql = [
                    # Dead letter queue for permanently failed items
                    """
                    CREATE TABLE IF NOT EXISTS dead_letter_queue (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL,
                        collection_name TEXT NOT NULL,
                        tenant_id TEXT,
                        branch TEXT,
                        operation TEXT NOT NULL DEFAULT 'ingest',
                        error_category TEXT NOT NULL CHECK (error_category IN ('permanent', 'max_retries', 'circuit_breaker')),
                        error_type TEXT NOT NULL,
                        error_message TEXT NOT NULL,
                        original_priority INTEGER DEFAULT 5,
                        retry_count INTEGER NOT NULL DEFAULT 0,
                        retry_history TEXT,
                        failed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        original_queued_at TIMESTAMP,
                        reprocessed_at TIMESTAMP,
                        reprocess_count INTEGER NOT NULL DEFAULT 0,
                        metadata TEXT
                    )
                    """,
                    # Indexes for dead_letter_queue
                    "CREATE INDEX IF NOT EXISTS idx_dlq_error_category ON dead_letter_queue(error_category)",
                    "CREATE INDEX IF NOT EXISTS idx_dlq_error_type ON dead_letter_queue(error_type)",
                    "CREATE INDEX IF NOT EXISTS idx_dlq_collection ON dead_letter_queue(collection_name)",
                    "CREATE INDEX IF NOT EXISTS idx_dlq_failed_at ON dead_letter_queue(failed_at)",
                    "CREATE INDEX IF NOT EXISTS idx_dlq_file_path ON dead_letter_queue(file_path)",
                ]

                for sql in dlq_sql:
                    try:
                        conn.execute(sql)
                    except Exception as e:
                        # Ignore if table already exists (idempotent migration)
                        if "already exists" not in str(e).lower():
                            raise

                logger.info("Successfully migrated to schema version 12 (dead letter queue)")

            # Migrate from version 12 to version 13 - Add unified_queue table (Task 22/23)
            if from_version < 13 <= to_version:
                logger.info("Applying migration: v12 -> v13 (unified queue table)")
                migration_sql = self._load_migration_statements(
                    "003_unified_queue.sql"
                )
                logged_table = False
                logged_indexes = False
                logged_schema_version = False

                try:
                    for statement in migration_sql:
                        normalized = statement.lstrip().upper()
                        if normalized.startswith("CREATE TABLE") and not logged_table:
                            logger.debug("Creating unified_queue table")
                            logged_table = True
                        elif normalized.startswith(
                            "CREATE INDEX"
                        ) or normalized.startswith("CREATE UNIQUE INDEX"):
                            if not logged_indexes:
                                logger.debug("Creating indexes for unified_queue")
                                logged_indexes = True
                        elif normalized.startswith(
                            "INSERT INTO SCHEMA_VERSION"
                        ) and not logged_schema_version:
                            logger.debug("Updating schema_version to 13")
                            logged_schema_version = True

                        conn.execute(statement)
                except Exception:
                    logger.exception(
                        "Migration v12 -> v13 failed; transaction will be rolled back"
                    )
                    raise

                logger.info("Successfully migrated to schema version 13 (unified queue table)")

            # Migrate from version 13 to version 14 - Add active_projects table (Task 36 - code audit round 2)
            if from_version < 14 <= to_version:
                logger.info("Applying migration: v13 -> v14 (active projects table)")
                migration_sql = self._load_migration_statements(
                    "004_active_projects.sql"
                )
                logged_table = False
                logged_indexes = False
                logged_views = False

                try:
                    for statement in migration_sql:
                        normalized = statement.lstrip().upper()
                        if normalized.startswith("CREATE TABLE") and not logged_table:
                            logger.debug("Creating active_projects table")
                            logged_table = True
                        elif normalized.startswith("CREATE INDEX") and not logged_indexes:
                            logger.debug("Creating indexes for active_projects")
                            logged_indexes = True
                        elif normalized.startswith("CREATE VIEW") and not logged_views:
                            logger.debug("Creating views for active_projects")
                            logged_views = True

                        conn.execute(statement)

                    # Add trigger inline (parser can't handle semicolons inside BEGIN...END)
                    logger.debug("Creating trigger for active_projects")
                    conn.execute("""
                        CREATE TRIGGER IF NOT EXISTS trg_active_projects_updated_at
                            AFTER UPDATE ON active_projects
                            FOR EACH ROW
                            BEGIN
                                UPDATE active_projects
                                SET updated_at = datetime('now')
                                WHERE project_id = NEW.project_id;
                            END
                    """)

                except Exception:
                    logger.exception(
                        "Migration v13 -> v14 failed; transaction will be rolled back"
                    )
                    raise

                logger.info("Successfully migrated to schema version 14 (active projects table)")

            # Record the migration
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (to_version,),
            )

        logger.info(f"Database migration completed: v{from_version} -> v{to_version}")

    async def _ensure_schema_columns(self):
        """
        Ensure all expected columns exist in tables.

        This handles cases where columns were added to the base schema
        but existing databases might not have been properly migrated.
        Called after migration to catch any missing columns.
        """
        # Expected columns for watch_folders table (added in various versions)
        watch_folders_columns = [
            ("watch_type", "TEXT NOT NULL DEFAULT 'project'"),
            ("library_name", "TEXT"),
            ("consecutive_errors", "INTEGER NOT NULL DEFAULT 0"),
            ("total_errors", "INTEGER NOT NULL DEFAULT 0"),
            ("last_error_at", "TIMESTAMP"),
            ("last_error_message", "TEXT"),
            ("backoff_until", "TIMESTAMP"),
            ("last_success_at", "TIMESTAMP"),
            ("health_status", "TEXT NOT NULL DEFAULT 'healthy'"),
            ("watch_priority", "INTEGER NOT NULL DEFAULT 5"),
        ]

        try:
            with self._lock:
                # Get existing columns for watch_folders
                cursor = self.connection.execute("PRAGMA table_info(watch_folders)")
                existing_columns = {row[1] for row in cursor.fetchall()}

                # Add missing columns
                for column_name, column_def in watch_folders_columns:
                    if column_name not in existing_columns:
                        try:
                            sql = f"ALTER TABLE watch_folders ADD COLUMN {column_name} {column_def}"
                            self.connection.execute(sql)
                            logger.info(f"Added missing column to watch_folders: {column_name}")
                        except Exception as e:
                            # Ignore if column already exists (race condition)
                            if "duplicate column" not in str(e).lower():
                                logger.warning(f"Failed to add column {column_name}: {e}")

                self.connection.commit()
        except Exception as e:
            logger.error(f"Schema column check failed: {e}")

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

    # Dead Letter Queue Methods (Task 14 - code audit)
    # These methods manage permanently failed items for debugging and reprocessing

    async def move_to_dead_letter_queue(
        self,
        file_path: str,
        collection_name: str,
        error_category: str,
        error_type: str,
        error_message: str,
        tenant_id: str | None = None,
        branch: str | None = None,
        operation: str = "ingest",
        original_priority: int = 5,
        retry_count: int = 0,
        retry_history: list[dict] | None = None,
        original_queued_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """
        Move a failed item to the dead letter queue.

        Args:
            file_path: Path of the failed file
            collection_name: Target collection name
            error_category: Category of failure (permanent, max_retries, circuit_breaker)
            error_type: Specific error type
            error_message: Human-readable error message
            tenant_id: Optional tenant identifier
            branch: Optional git branch
            operation: Operation type (ingest, update, delete)
            original_priority: Original queue priority
            retry_count: Number of retry attempts made
            retry_history: JSON-serializable history of retry attempts
            original_queued_at: When item was originally queued
            metadata: Additional context

        Returns:
            ID of the dead letter queue entry
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO dead_letter_queue (
                        file_path, collection_name, tenant_id, branch, operation,
                        error_category, error_type, error_message,
                        original_priority, retry_count, retry_history,
                        original_queued_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_path,
                        collection_name,
                        tenant_id,
                        branch,
                        operation,
                        error_category,
                        error_type,
                        error_message,
                        original_priority,
                        retry_count,
                        json.dumps(retry_history) if retry_history else None,
                        original_queued_at.isoformat() if original_queued_at else None,
                        json.dumps(metadata) if metadata else None,
                    )
                )

                dlq_id = cursor.lastrowid
                logger.info(f"Moved to dead letter queue: {file_path} (id={dlq_id}, category={error_category})")
                return dlq_id

        except Exception as e:
            logger.error(f"Failed to move to dead letter queue: {e}")
            raise

    async def list_dead_letter_items(
        self,
        error_category: str | None = None,
        error_type: str | None = None,
        collection: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        List items in the dead letter queue.

        Args:
            error_category: Filter by error category
            error_type: Filter by error type
            collection: Filter by collection name
            limit: Maximum items to return
            offset: Offset for pagination

        Returns:
            List of dead letter queue items
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            query = "SELECT * FROM dead_letter_queue WHERE 1=1"
            params = []

            if error_category:
                query += " AND error_category = ?"
                params.append(error_category)

            if error_type:
                query += " AND error_type = ?"
                params.append(error_type)

            if collection:
                query += " AND collection_name = ?"
                params.append(collection)

            query += " ORDER BY failed_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            with self._lock:
                cursor = self.connection.execute(query, params)
                rows = cursor.fetchall()

                items = []
                for row in rows:
                    item = dict(row)
                    # Parse JSON fields
                    if item.get("retry_history"):
                        item["retry_history"] = json.loads(item["retry_history"])
                    if item.get("metadata"):
                        item["metadata"] = json.loads(item["metadata"])
                    items.append(item)

                return items

        except Exception as e:
            logger.error(f"Failed to list dead letter items: {e}")
            raise

    async def reprocess_dead_letter_item(
        self,
        dlq_id: int,
        priority: int | None = None,
    ) -> str | None:
        """
        Move a dead letter item back to the main queue for reprocessing.

        Args:
            dlq_id: ID of the dead letter queue item
            priority: Override priority for reprocessing (default: original priority)

        Returns:
            Queue ID if successfully requeued, None if item not found
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            async with self.transaction() as conn:
                # Get the DLQ item
                cursor = conn.execute(
                    "SELECT * FROM dead_letter_queue WHERE id = ?",
                    (dlq_id,)
                )
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"Dead letter item not found: {dlq_id}")
                    return None

                item = dict(row)
                use_priority = priority if priority is not None else item["original_priority"]

                # Re-enqueue to the ingestion queue
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ingestion_queue (
                        file_absolute_path, collection_name, tenant_id, branch,
                        operation, priority, retry_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        item["file_path"],
                        item["collection_name"],
                        item["tenant_id"],
                        item["branch"],
                        item["operation"],
                        use_priority,
                        item["metadata"],
                    )
                )

                # Update DLQ item to mark as reprocessed
                conn.execute(
                    """
                    UPDATE dead_letter_queue
                    SET reprocessed_at = CURRENT_TIMESTAMP,
                        reprocess_count = reprocess_count + 1
                    WHERE id = ?
                    """,
                    (dlq_id,)
                )

                logger.info(f"Reprocessing dead letter item {dlq_id}: {item['file_path']}")
                return item["file_path"]

        except Exception as e:
            logger.error(f"Failed to reprocess dead letter item {dlq_id}: {e}")
            raise

    async def get_dead_letter_stats(self) -> dict[str, Any]:
        """
        Get statistics about the dead letter queue.

        Returns:
            Dictionary with DLQ statistics
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            with self._lock:
                stats = {}

                # Total count
                cursor = self.connection.execute(
                    "SELECT COUNT(*) FROM dead_letter_queue"
                )
                stats["total_count"] = cursor.fetchone()[0]

                # Count by error category
                cursor = self.connection.execute(
                    """
                    SELECT error_category, COUNT(*) as count
                    FROM dead_letter_queue
                    GROUP BY error_category
                    """
                )
                stats["by_category"] = {row[0]: row[1] for row in cursor.fetchall()}

                # Count by error type
                cursor = self.connection.execute(
                    """
                    SELECT error_type, COUNT(*) as count
                    FROM dead_letter_queue
                    GROUP BY error_type
                    ORDER BY count DESC
                    LIMIT 10
                    """
                )
                stats["by_error_type"] = {row[0]: row[1] for row in cursor.fetchall()}

                # Count by collection
                cursor = self.connection.execute(
                    """
                    SELECT collection_name, COUNT(*) as count
                    FROM dead_letter_queue
                    GROUP BY collection_name
                    """
                )
                stats["by_collection"] = {row[0]: row[1] for row in cursor.fetchall()}

                # Reprocess stats
                cursor = self.connection.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (WHERE reprocessed_at IS NOT NULL) as reprocessed,
                        SUM(reprocess_count) as total_reprocess_attempts
                    FROM dead_letter_queue
                    """
                )
                row = cursor.fetchone()
                stats["reprocessed_count"] = row[0] or 0
                stats["total_reprocess_attempts"] = row[1] or 0

                # Recent failures (last 24 hours)
                cursor = self.connection.execute(
                    """
                    SELECT COUNT(*) FROM dead_letter_queue
                    WHERE failed_at > datetime('now', '-1 day')
                    """
                )
                stats["recent_failures_24h"] = cursor.fetchone()[0]

                return stats

        except Exception as e:
            logger.error(f"Failed to get dead letter stats: {e}")
            raise

    async def delete_dead_letter_item(self, dlq_id: int) -> bool:
        """
        Delete an item from the dead letter queue.

        Args:
            dlq_id: ID of the dead letter queue item

        Returns:
            True if deleted, False if not found
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM dead_letter_queue WHERE id = ?",
                    (dlq_id,)
                )
                deleted = cursor.rowcount > 0

                if deleted:
                    logger.info(f"Deleted dead letter item: {dlq_id}")
                else:
                    logger.warning(f"Dead letter item not found: {dlq_id}")

                return deleted

        except Exception as e:
            logger.error(f"Failed to delete dead letter item {dlq_id}: {e}")
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

    # Unified Queue Methods (Task 25 - Queue Consolidation)
    # These methods provide the new unified queue interface that consolidates
    # content_ingestion_queue and ingestion_queue into a single unified_queue table.

    async def enqueue_unified(
        self,
        item_type: UnifiedQueueItemType | str,
        op: UnifiedQueueOperation | str,
        tenant_id: str,
        collection: str,
        payload: dict[str, Any],
        priority: int = 5,
        dual_write: bool | None = None,
        branch: str = "main",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """
        Enqueue an item to the unified queue with idempotency support.

        This is the primary enqueue method for the consolidated queue system.
        Uses SHA256 hash of input parameters as idempotency key to prevent
        duplicate processing.

        Args:
            item_type: Type of queue item (content, file, folder, project, etc.)
            op: Operation type (ingest, update, delete, scan)
            tenant_id: Project/tenant identifier
            collection: Target Qdrant collection name
            payload: Payload dictionary with item-specific data
            priority: Priority level 0-10 (10 is highest, default 5)
            dual_write: If True, also write to legacy queue. If None, uses config default.
            branch: Git branch context (default: 'main')
            metadata: Optional additional metadata

        Returns:
            Tuple of (queue_id, is_new) where is_new indicates if item was newly created.
            If item already exists (same idempotency key), returns (existing_id, False).

        Raises:
            ValueError: If priority is out of range or inputs invalid
            RuntimeError: If state manager not initialized

        Example:
            >>> queue_id, is_new = await state_manager.enqueue_unified(
            ...     item_type=UnifiedQueueItemType.FILE,
            ...     op=UnifiedQueueOperation.INGEST,
            ...     tenant_id="proj_abc123",
            ...     collection="my-project-code",
            ...     payload={"file_path": "/path/to/file.py"},
            ...     priority=7
            ... )
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        # Validate priority range
        if not 0 <= priority <= 10:
            raise ValueError(f"Priority must be between 0 and 10, got {priority}")

        # Validate tenant_id and collection
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id cannot be empty")
        if not collection or not collection.strip():
            raise ValueError("collection cannot be empty")

        # Convert enums to string values
        if isinstance(item_type, UnifiedQueueItemType):
            item_type_str = item_type.value
        else:
            item_type_str = str(item_type)
            # Validate against enum values
            try:
                UnifiedQueueItemType(item_type_str)
            except ValueError:
                valid_types = [t.value for t in UnifiedQueueItemType]
                raise ValueError(
                    f"Invalid item_type '{item_type_str}'. Must be one of: {valid_types}"
                )

        if isinstance(op, UnifiedQueueOperation):
            op_str = op.value
        else:
            op_str = str(op)
            # Validate against enum values
            try:
                UnifiedQueueOperation(op_str)
            except ValueError:
                valid_ops = [o.value for o in UnifiedQueueOperation]
                raise ValueError(
                    f"Invalid operation '{op_str}'. Must be one of: {valid_ops}"
                )

        # Validate operation is valid for item type
        item_type_enum = UnifiedQueueItemType(item_type_str)
        op_enum = UnifiedQueueOperation(op_str)
        if not op_enum.is_valid_for(item_type_enum):
            raise ValueError(
                f"Operation '{op_str}' is not valid for item type '{item_type_str}'"
            )

        try:
            # Generate idempotency key using shared function
            idempotency_key = generate_unified_idempotency_key(
                item_type=item_type_str,
                op=op_str,
                tenant_id=tenant_id,
                collection=collection,
                payload=payload,
            )

            # Generate queue_id
            queue_id = str(uuid.uuid4())

            # Serialize payload to JSON
            payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))

            # Serialize metadata if provided
            metadata_json = (
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                if metadata
                else "{}"
            )

            async with self.transaction() as conn:
                # Check if item with same idempotency key already exists
                cursor = conn.execute(
                    "SELECT queue_id FROM unified_queue WHERE idempotency_key = ?",
                    (idempotency_key,),
                )
                existing = cursor.fetchone()

                if existing:
                    logger.debug(
                        f"Unified queue item already exists (idempotency): {existing['queue_id']}"
                    )
                    return existing["queue_id"], False

                # Insert new queue item
                conn.execute(
                    """
                    INSERT INTO unified_queue
                    (queue_id, item_type, op, tenant_id, collection, priority, status,
                     idempotency_key, payload_json, branch, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                    """,
                    (
                        queue_id,
                        item_type_str,
                        op_str,
                        tenant_id,
                        collection,
                        priority,
                        idempotency_key,
                        payload_json,
                        branch,
                        metadata_json,
                    ),
                )

                logger.debug(
                    f"Enqueued unified item: {queue_id} "
                    f"(type={item_type_str}, op={op_str}, collection={collection}, priority={priority})"
                )

                # Record enqueue metric
                record_queue_enqueue(item_type_str, op_str)

                # Handle dual-write if enabled
                should_dual_write = dual_write
                if should_dual_write is None:
                    # Check config for default (fall back to False)
                    try:
                        from .config import ConfigManager
                        config = ConfigManager.get_instance()
                        should_dual_write = config.get(
                            "queue_processor.enable_dual_write", False
                        )
                    except Exception:
                        should_dual_write = False

                if should_dual_write:
                    await self._dual_write_to_legacy_queue(
                        conn=conn,
                        item_type=item_type_str,
                        op=op_str,
                        tenant_id=tenant_id,
                        collection=collection,
                        payload=payload,
                        priority=priority,
                        branch=branch,
                        metadata=metadata,
                    )

                return queue_id, True

        except ValueError:
            raise
        except IdempotencyKeyError as e:
            raise ValueError(str(e))
        except Exception as e:
            logger.error(f"Failed to enqueue unified item: {e}")
            raise

    # TODO: Remove in v0.5.0 (Phase 4 cleanup) - dual_write and legacy queues
    async def _dual_write_to_legacy_queue(
        self,
        conn,
        item_type: str,
        op: str,
        tenant_id: str,
        collection: str,
        payload: dict[str, Any],
        priority: int,
        branch: str,
        metadata: dict[str, Any] | None,
    ) -> None:
        """
        Write to legacy queue for dual-write migration support.

        .. deprecated:: 0.4.0
            Legacy queues (ingestion_queue, content_ingestion_queue) are deprecated.
            Use unified_queue exclusively. This method will be removed in v0.5.0.
            See docs/MIGRATION.md for migration guidance.

        Routes items to appropriate legacy queue based on item_type:
        - content items -> content_ingestion_queue
        - file/folder items -> ingestion_queue

        Args:
            conn: Database connection (within transaction)
            item_type: Type of queue item
            op: Operation type
            tenant_id: Project/tenant identifier
            collection: Target collection name
            payload: Payload dictionary
            priority: Priority level
            branch: Git branch
            metadata: Optional metadata
        """
        try:
            if item_type == "content":
                # Write to content_ingestion_queue
                content = payload.get("content", "")
                source_type = payload.get("source_type", "scratchbook")
                main_tag = payload.get("main_tag")
                full_tag = payload.get("full_tag")

                # Calculate legacy idempotency key
                idempotency_input = (
                    f"{content}|{collection}|{source_type}|"
                    f"{json.dumps(metadata or {}, sort_keys=True)}"
                )
                idempotency_key = hashlib.sha256(
                    idempotency_input.encode("utf-8")
                ).hexdigest()[:32]

                queue_id = str(uuid.uuid4())

                conn.execute(
                    """
                    INSERT OR IGNORE INTO content_ingestion_queue
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
                logger.debug(f"Dual-write to content_ingestion_queue: {queue_id}")
                record_dual_write_success(item_type, "content_ingestion_queue")

            elif item_type in ("file", "folder"):
                # Write to ingestion_queue
                file_path = payload.get("file_path", "")

                # Generate a unique ID for the legacy queue
                queue_id = str(uuid.uuid4())

                # Map operation to legacy format
                operation_map = {"ingest": "ingest", "update": "update", "delete": "delete"}
                legacy_op = operation_map.get(op, "ingest")

                conn.execute(
                    """
                    INSERT OR IGNORE INTO ingestion_queue
                    (file_absolute_path, collection_name, tenant_id, branch, operation,
                     priority, status, queued_timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP, ?)
                    """,
                    (
                        file_path,
                        collection,
                        tenant_id,
                        branch,
                        legacy_op,
                        priority,
                        self._serialize_json(metadata) if metadata else None,
                    ),
                )
                logger.debug(f"Dual-write to ingestion_queue: {file_path}")
                record_dual_write_success(item_type, "ingestion_queue")

            else:
                # Other item types don't have legacy queue equivalents
                logger.debug(
                    f"Skipping dual-write for item_type={item_type} (no legacy equivalent)"
                )

        except Exception as e:
            # Log but don't fail the main enqueue operation
            logger.warning(f"Dual-write to legacy queue failed (non-fatal): {e}")
            # Determine target queue for metrics
            target_queue = (
                "content_ingestion_queue"
                if item_type == "content"
                else "ingestion_queue"
            )
            record_dual_write_failure(item_type, target_queue, type(e).__name__)

    async def get_unified_queue_depth(
        self,
        collection: str | None = None,
        status: str | None = None,
        item_type: str | None = None,
    ) -> int:
        """
        Get the current depth (count) of the unified queue.

        Args:
            collection: Optional filter by collection name
            status: Optional filter by status ('pending', 'in_progress', 'done', 'failed')
            item_type: Optional filter by item type

        Returns:
            Number of items in the queue matching the filters

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            query = "SELECT COUNT(*) FROM unified_queue WHERE 1=1"
            params: list[Any] = []

            if collection:
                query += " AND collection = ?"
                params.append(collection)

            if status:
                query += " AND status = ?"
                params.append(status)

            if item_type:
                query += " AND item_type = ?"
                params.append(item_type)

            with self._lock:
                cursor = self.connection.execute(query, params)
                count = cursor.fetchone()[0]
                return count

        except Exception as e:
            logger.error(f"Failed to get unified queue depth: {e}")
            raise

    async def get_unified_queue_stats(self) -> dict[str, Any]:
        """
        Get comprehensive statistics for the unified queue.

        Returns a dictionary with counts by status, item_type, and operation,
        plus oldest pending item age and other diagnostic information.

        Returns:
            Dict with queue statistics:
                - total: Total items in queue
                - by_status: Count per status (pending, in_progress, done, failed)
                - by_item_type: Count per item type
                - by_operation: Count per operation type
                - oldest_pending_age_seconds: Age of oldest pending item (or None)
                - collections_with_pending: List of collections with pending items

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            stats: dict[str, Any] = {
                "total": 0,
                "by_status": {},
                "by_item_type": {},
                "by_operation": {},
                "oldest_pending_age_seconds": None,
                "collections_with_pending": [],
            }

            with self._lock:
                # Total count
                cursor = self.connection.execute(
                    "SELECT COUNT(*) FROM unified_queue"
                )
                stats["total"] = cursor.fetchone()[0]

                # Count by status
                cursor = self.connection.execute(
                    "SELECT status, COUNT(*) FROM unified_queue GROUP BY status"
                )
                stats["by_status"] = {row[0]: row[1] for row in cursor.fetchall()}

                # Count by item_type
                cursor = self.connection.execute(
                    "SELECT item_type, COUNT(*) FROM unified_queue GROUP BY item_type"
                )
                stats["by_item_type"] = {row[0]: row[1] for row in cursor.fetchall()}

                # Count by operation
                cursor = self.connection.execute(
                    "SELECT op, COUNT(*) FROM unified_queue GROUP BY op"
                )
                stats["by_operation"] = {row[0]: row[1] for row in cursor.fetchall()}

                # Oldest pending item age
                cursor = self.connection.execute(
                    """
                    SELECT MIN(created_at) FROM unified_queue
                    WHERE status = 'pending'
                    """
                )
                oldest_created = cursor.fetchone()[0]
                if oldest_created:
                    from datetime import datetime, timezone
                    oldest_dt = datetime.fromisoformat(oldest_created.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    stats["oldest_pending_age_seconds"] = (now - oldest_dt).total_seconds()

                # Collections with pending items
                cursor = self.connection.execute(
                    """
                    SELECT DISTINCT collection FROM unified_queue
                    WHERE status = 'pending'
                    ORDER BY collection
                    """
                )
                stats["collections_with_pending"] = [row[0] for row in cursor.fetchall()]

            return stats

        except Exception as e:
            logger.error(f"Failed to get unified queue stats: {e}")
            raise

    async def detect_queue_drift(self) -> dict[str, Any]:
        """
        Detect drift between unified_queue and legacy queues.

        Compares idempotency keys and statuses between the unified_queue and
        both legacy queues (ingestion_queue and content_ingestion_queue) to
        identify discrepancies in the dual-write migration.

        Returns:
            Dict with drift information:
                - missing_in_unified: Items in legacy but not unified
                - missing_in_legacy: Items in unified but not legacy
                - status_mismatch: Items with different statuses
                - total_drift_count: Total number of drifted items
                - by_queue: Breakdown by legacy queue type
                - checked_at: Timestamp of check

        Raises:
            RuntimeError: If state manager not initialized
        """
        from ..observability.metrics import record_drift_detected, set_drift_gauge

        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            drift_report: dict[str, Any] = {
                "missing_in_unified": [],
                "missing_in_legacy": [],
                "status_mismatch": [],
                "total_drift_count": 0,
                "by_queue": {
                    "ingestion_queue": {"missing": 0, "status_mismatch": 0},
                    "content_ingestion_queue": {"missing": 0, "status_mismatch": 0},
                },
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

            with self._lock:
                # Get unified queue items with their idempotency keys
                # (only file/folder items have legacy equivalents in ingestion_queue)
                cursor = self.connection.execute(
                    """
                    SELECT idempotency_key, status, item_type, collection,
                           json_extract(payload_json, '$.file_path') as file_path
                    FROM unified_queue
                    WHERE item_type IN ('file', 'folder')
                    """
                )
                unified_file_items = {
                    row[0]: {"status": row[1], "item_type": row[2], "collection": row[3], "file_path": row[4]}
                    for row in cursor.fetchall()
                    if row[0]  # Skip items without idempotency key
                }

                # Get unified content items
                cursor = self.connection.execute(
                    """
                    SELECT idempotency_key, status, collection
                    FROM unified_queue
                    WHERE item_type = 'content'
                    """
                )
                unified_content_items = {
                    row[0]: {"status": row[1], "collection": row[2]}
                    for row in cursor.fetchall()
                    if row[0]
                }

                # Check ingestion_queue (file/folder items)
                try:
                    cursor = self.connection.execute(
                        """
                        SELECT file_absolute_path, status, collection_name
                        FROM ingestion_queue
                        """
                    )
                    legacy_file_items = {
                        row[0]: {"status": row[1], "collection": row[2]}
                        for row in cursor.fetchall()
                    }

                    # Map file_path to idempotency key for comparison
                    unified_by_path = {
                        v["file_path"]: {"key": k, **v}
                        for k, v in unified_file_items.items()
                        if v.get("file_path")
                    }

                    # Find items missing in unified (present in legacy but not unified)
                    for file_path, legacy_info in legacy_file_items.items():
                        if file_path not in unified_by_path:
                            drift_report["missing_in_unified"].append({
                                "type": "file",
                                "legacy_queue": "ingestion_queue",
                                "file_path": file_path,
                                "legacy_status": legacy_info["status"],
                            })
                            drift_report["by_queue"]["ingestion_queue"]["missing"] += 1

                    # Find items missing in legacy (present in unified but not legacy)
                    for file_path, unified_info in unified_by_path.items():
                        if file_path not in legacy_file_items:
                            drift_report["missing_in_legacy"].append({
                                "type": "file",
                                "legacy_queue": "ingestion_queue",
                                "file_path": file_path,
                                "unified_status": unified_info["status"],
                            })
                            drift_report["by_queue"]["ingestion_queue"]["missing"] += 1
                        elif legacy_file_items[file_path]["status"] != unified_info["status"]:
                            # Status mismatch
                            drift_report["status_mismatch"].append({
                                "type": "file",
                                "legacy_queue": "ingestion_queue",
                                "file_path": file_path,
                                "unified_status": unified_info["status"],
                                "legacy_status": legacy_file_items[file_path]["status"],
                            })
                            drift_report["by_queue"]["ingestion_queue"]["status_mismatch"] += 1

                except sqlite3.OperationalError:
                    # ingestion_queue table doesn't exist
                    logger.debug("ingestion_queue table not found, skipping file drift check")

                # Check content_ingestion_queue (content items)
                try:
                    cursor = self.connection.execute(
                        """
                        SELECT idempotency_key, status, collection
                        FROM content_ingestion_queue
                        """
                    )
                    legacy_content_items = {
                        row[0]: {"status": row[1], "collection": row[2]}
                        for row in cursor.fetchall()
                        if row[0]
                    }

                    # Find content items missing in unified
                    for idem_key, legacy_info in legacy_content_items.items():
                        if idem_key not in unified_content_items:
                            drift_report["missing_in_unified"].append({
                                "type": "content",
                                "legacy_queue": "content_ingestion_queue",
                                "idempotency_key": idem_key,
                                "legacy_status": legacy_info["status"],
                            })
                            drift_report["by_queue"]["content_ingestion_queue"]["missing"] += 1

                    # Find content items missing in legacy
                    for idem_key, unified_info in unified_content_items.items():
                        if idem_key not in legacy_content_items:
                            drift_report["missing_in_legacy"].append({
                                "type": "content",
                                "legacy_queue": "content_ingestion_queue",
                                "idempotency_key": idem_key,
                                "unified_status": unified_info["status"],
                            })
                            drift_report["by_queue"]["content_ingestion_queue"]["missing"] += 1
                        elif legacy_content_items[idem_key]["status"] != unified_info["status"]:
                            drift_report["status_mismatch"].append({
                                "type": "content",
                                "legacy_queue": "content_ingestion_queue",
                                "idempotency_key": idem_key,
                                "unified_status": unified_info["status"],
                                "legacy_status": legacy_content_items[idem_key]["status"],
                            })
                            drift_report["by_queue"]["content_ingestion_queue"]["status_mismatch"] += 1

                except sqlite3.OperationalError:
                    # content_ingestion_queue table doesn't exist
                    logger.debug("content_ingestion_queue table not found, skipping content drift check")

            # Calculate total drift count
            drift_report["total_drift_count"] = (
                len(drift_report["missing_in_unified"])
                + len(drift_report["missing_in_legacy"])
                + len(drift_report["status_mismatch"])
            )

            # Record metrics
            if drift_report["missing_in_unified"]:
                record_drift_detected("missing_in_unified", len(drift_report["missing_in_unified"]))
            if drift_report["missing_in_legacy"]:
                record_drift_detected("missing_in_legacy", len(drift_report["missing_in_legacy"]))
            if drift_report["status_mismatch"]:
                record_drift_detected("status_mismatch", len(drift_report["status_mismatch"]))

            # Update gauges with current drift counts
            set_drift_gauge("missing_in_unified", len(drift_report["missing_in_unified"]))
            set_drift_gauge("missing_in_legacy", len(drift_report["missing_in_legacy"]))
            set_drift_gauge("status_mismatch", len(drift_report["status_mismatch"]))

            logger.info(
                f"Queue drift check complete: {drift_report['total_drift_count']} discrepancies found"
            )

            return drift_report

        except Exception as e:
            logger.error(f"Failed to detect queue drift: {e}")
            raise

    async def dequeue_unified(
        self,
        batch_size: int = 10,
        worker_id: str | None = None,
        lease_duration_seconds: int = 300,
        collection: str | None = None,
        item_type: str | None = None,
    ) -> list[UnifiedQueueItem]:
        """
        Dequeue items from the unified queue with lease-based locking.

        Items are returned in priority order (DESC) then creation time (ASC).
        Each dequeued item is locked with a lease to prevent concurrent processing.

        Args:
            batch_size: Maximum number of items to dequeue
            worker_id: Identifier for the worker (for tracking)
            lease_duration_seconds: How long to hold the lease (default 5 minutes)
            collection: Optional filter by collection name
            item_type: Optional filter by item type

        Returns:
            List of UnifiedQueueItem objects ready for processing

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        if worker_id is None:
            worker_id = f"worker_{uuid.uuid4().hex[:8]}"

        try:
            now = datetime.now(timezone.utc)
            lease_until = now + timedelta(seconds=lease_duration_seconds)

            async with self.transaction() as conn:
                # Build query with filters
                query = """
                    SELECT queue_id, item_type, op, tenant_id, collection, priority, status,
                           idempotency_key, payload_json, branch, metadata, created_at,
                           updated_at, lease_until, worker_id, retry_count, max_retries,
                           error_message, last_error_at
                    FROM unified_queue
                    WHERE status = 'pending'
                """
                params: list[Any] = []

                if collection:
                    query += " AND collection = ?"
                    params.append(collection)

                if item_type:
                    query += " AND item_type = ?"
                    params.append(item_type)

                query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
                params.append(batch_size)

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                items = []
                queue_ids = []

                for row in rows:
                    queue_ids.append(row["queue_id"])

                # Update items to in_progress with lease
                if queue_ids:
                    placeholders = ",".join("?" * len(queue_ids))
                    conn.execute(
                        f"""
                        UPDATE unified_queue
                        SET status = 'in_progress',
                            lease_until = ?,
                            worker_id = ?,
                            updated_at = ?
                        WHERE queue_id IN ({placeholders})
                        """,
                        [lease_until.isoformat(), worker_id, now.isoformat()] + queue_ids,
                    )

                # Build item objects
                for row in rows:
                    item = UnifiedQueueItem(
                        queue_id=row["queue_id"],
                        idempotency_key=row["idempotency_key"],
                        item_type=row["item_type"],
                        op=row["op"],
                        tenant_id=row["tenant_id"],
                        collection=row["collection"],
                        priority=row["priority"],
                        status="in_progress",  # Status after update
                        branch=row["branch"] or "main",
                        payload_json=row["payload_json"],
                        metadata=self._deserialize_json(row["metadata"]),
                        created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else None,
                        updated_at=now,
                        lease_until=lease_until,
                        worker_id=worker_id,
                        retry_count=row["retry_count"],
                        max_retries=row["max_retries"],
                        error_message=row["error_message"],
                        last_error_at=datetime.fromisoformat(row["last_error_at"]) if row["last_error_at"] else None,
                    )
                    items.append(item)

                    # Record wait duration (time from enqueue to dequeue)
                    if item.created_at:
                        try:
                            wait_duration = (now - item.created_at).total_seconds()
                            if wait_duration >= 0:
                                record_wait_duration(item.item_type, wait_duration)
                        except (ValueError, TypeError):
                            pass  # Skip if calculation fails

                logger.debug(f"Dequeued {len(items)} unified queue items for worker {worker_id}")
                return items

        except Exception as e:
            logger.error(f"Failed to dequeue unified items: {e}")
            raise

    async def mark_unified_item_done(
        self,
        queue_id: str,
    ) -> bool:
        """
        Mark a unified queue item as successfully completed.

        Args:
            queue_id: Queue ID of the item to mark done

        Returns:
            True if item was updated, False if not found

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            now = datetime.now(timezone.utc)

            async with self.transaction() as conn:
                # Fetch item details for metrics
                cursor = conn.execute(
                    "SELECT item_type, op, created_at FROM unified_queue WHERE queue_id = ?",
                    (queue_id,),
                )
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"Unified queue item not found: {queue_id}")
                    return False

                item_type = row["item_type"]
                op = row["op"]
                created_at_str = row["created_at"]

                cursor = conn.execute(
                    """
                    UPDATE unified_queue
                    SET status = 'done',
                        updated_at = ?,
                        lease_until = NULL,
                        worker_id = NULL
                    WHERE queue_id = ?
                    """,
                    (now.isoformat(), queue_id),
                )

                updated = cursor.rowcount > 0

                if updated:
                    logger.debug(f"Marked unified queue item done: {queue_id}")

                    # Record metrics
                    record_queue_processed(item_type, op, "done")

                    # Calculate and record processing duration
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_str.replace("Z", "+00:00")
                            )
                            duration = (now - created_at).total_seconds()
                            record_processing_duration(item_type, op, duration)
                        except (ValueError, TypeError):
                            pass  # Skip if timestamp parsing fails

                return updated

        except Exception as e:
            logger.error(f"Failed to mark unified item done {queue_id}: {e}")
            raise

    async def mark_unified_item_failed(
        self,
        queue_id: str,
        error_message: str,
        retry: bool = True,
    ) -> bool:
        """
        Mark a unified queue item as failed, optionally scheduling for retry.

        Args:
            queue_id: Queue ID of the item
            error_message: Error message to record
            retry: If True and retry_count < max_retries, reschedule as pending

        Returns:
            True if item was updated, False if not found

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            now = datetime.now(timezone.utc)

            async with self.transaction() as conn:
                # Check current retry state and get item details for metrics
                cursor = conn.execute(
                    """
                    SELECT retry_count, max_retries, item_type, op, created_at
                    FROM unified_queue WHERE queue_id = ?
                    """,
                    (queue_id,),
                )
                row = cursor.fetchone()

                if not row:
                    logger.warning(f"Unified queue item not found: {queue_id}")
                    return False

                retry_count = row["retry_count"]
                max_retries = row["max_retries"]
                item_type = row["item_type"]
                op = row["op"]
                created_at_str = row["created_at"]

                if retry and retry_count < max_retries:
                    # Reschedule for retry
                    cursor = conn.execute(
                        """
                        UPDATE unified_queue
                        SET status = 'pending',
                            retry_count = retry_count + 1,
                            error_message = ?,
                            last_error_at = ?,
                            updated_at = ?,
                            lease_until = NULL,
                            worker_id = NULL
                        WHERE queue_id = ?
                        """,
                        (error_message, now.isoformat(), now.isoformat(), queue_id),
                    )
                    logger.debug(
                        f"Rescheduled unified queue item for retry: {queue_id} "
                        f"(attempt {retry_count + 1}/{max_retries})"
                    )

                    # Record retry metric
                    record_queue_retry(item_type)
                else:
                    # Mark as permanently failed
                    cursor = conn.execute(
                        """
                        UPDATE unified_queue
                        SET status = 'failed',
                            retry_count = retry_count + 1,
                            error_message = ?,
                            last_error_at = ?,
                            updated_at = ?,
                            lease_until = NULL,
                            worker_id = NULL
                        WHERE queue_id = ?
                        """,
                        (error_message, now.isoformat(), now.isoformat(), queue_id),
                    )
                    logger.warning(f"Marked unified queue item as failed (max retries): {queue_id}")

                    # Record failure metric
                    record_queue_processed(item_type, op, "failed")

                    # Calculate and record processing duration
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_str.replace("Z", "+00:00")
                            )
                            duration = (now - created_at).total_seconds()
                            record_processing_duration(item_type, op, duration)
                        except (ValueError, TypeError):
                            pass  # Skip if timestamp parsing fails

                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to mark unified item failed {queue_id}: {e}")
            raise

    async def recover_stale_unified_leases(
        self,
        max_retries: int = 3,
    ) -> int:
        """
        Recover items with expired leases back to pending status.

        Called during daemon startup to recover items that were being processed
        when the previous daemon instance crashed.

        Args:
            max_retries: Maximum retries before marking as failed

        Returns:
            Number of items recovered

        Raises:
            RuntimeError: If state manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("State manager not initialized")

        try:
            now = datetime.now(timezone.utc)
            recovered_count = 0
            failed_count = 0

            async with self.transaction() as conn:
                # Find items with expired leases
                cursor = conn.execute(
                    """
                    SELECT queue_id, retry_count, max_retries
                    FROM unified_queue
                    WHERE status = 'in_progress'
                    AND lease_until < ?
                    """,
                    (now.isoformat(),),
                )
                expired_items = cursor.fetchall()

                for row in expired_items:
                    queue_id = row["queue_id"]
                    retry_count = row["retry_count"]
                    item_max_retries = row["max_retries"] or max_retries

                    if retry_count < item_max_retries:
                        # Reschedule for retry
                        conn.execute(
                            """
                            UPDATE unified_queue
                            SET status = 'pending',
                                retry_count = retry_count + 1,
                                error_message = 'Recovered from stale lease',
                                last_error_at = ?,
                                updated_at = ?,
                                lease_until = NULL,
                                worker_id = NULL
                            WHERE queue_id = ?
                            """,
                            (now.isoformat(), now.isoformat(), queue_id),
                        )
                        recovered_count += 1
                    else:
                        # Mark as permanently failed
                        conn.execute(
                            """
                            UPDATE unified_queue
                            SET status = 'failed',
                                retry_count = retry_count + 1,
                                error_message = 'Max retries exceeded during lease recovery',
                                last_error_at = ?,
                                updated_at = ?,
                                lease_until = NULL,
                                worker_id = NULL
                            WHERE queue_id = ?
                            """,
                            (now.isoformat(), now.isoformat(), queue_id),
                        )
                        failed_count += 1

            if recovered_count > 0 or failed_count > 0:
                logger.info(
                    f"Unified queue lease recovery: recovered {recovered_count} items, "
                    f"marked {failed_count} as failed"
                )

            return recovered_count

        except Exception as e:
            logger.error(f"Failed to recover stale unified leases: {e}")
            raise

    async def cleanup_completed_unified_items(
        self,
        older_than_hours: int = 24,
    ) -> int:
        """
        Remove completed (done) unified queue items older than specified hours.

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
                    DELETE FROM unified_queue
                    WHERE status = 'done'
                    AND updated_at < datetime('now', '-' || ? || ' hours')
                    """,
                    (older_than_hours,),
                )
                removed = cursor.rowcount

                if removed > 0:
                    logger.debug(f"Removed {removed} completed unified queue items")

                return removed

        except Exception as e:
            logger.error(f"Failed to cleanup completed unified items: {e}")
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
                     backoff_until, last_success_at, health_status, watch_priority)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           COALESCE((SELECT created_at FROM watch_folders WHERE watch_id = ?), CURRENT_TIMESTAMP),
                           CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        config.watch_priority,
                    ),
                )

            logger.debug(f"Saved watch folder config: {config.watch_id} (type={config.watch_type}, health={config.health_status}, priority={config.watch_priority})")
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
                           backoff_until, last_success_at, health_status, watch_priority
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
                    # Priority adjustment (Task 461.17)
                    watch_priority=row["watch_priority"] if row["watch_priority"] is not None else 5,
                )

        except Exception as e:
            logger.error(f"Failed to get watch folder config {watch_id}: {e}")
            return None

    async def get_all_watch_folder_configs(
        self, enabled_only: bool = True, order_by_priority: bool = False
    ) -> list[WatchFolderConfig]:
        """Get all watch folder configurations.

        Args:
            enabled_only: If True, only return enabled watches (default: True)
            order_by_priority: If True, order by watch_priority DESC (Task 461.17)
        """
        try:
            with self._lock:
                sql = """
                    SELECT watch_id, path, collection, patterns, ignore_patterns, auto_ingest,
                           recursive, recursive_depth, debounce_seconds, enabled,
                           watch_type, library_name,
                           created_at, updated_at, last_scan, metadata,
                           consecutive_errors, total_errors, last_error_at, last_error_message,
                           backoff_until, last_success_at, health_status, watch_priority
                    FROM watch_folders
                """

                if enabled_only:
                    sql += " WHERE enabled = 1"

                if order_by_priority:
                    sql += " ORDER BY watch_priority DESC, created_at ASC"
                else:
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
                            # Priority adjustment (Task 461.17)
                            watch_priority=row["watch_priority"] if row["watch_priority"] is not None else 5,
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
                           backoff_until, last_success_at, health_status, watch_priority
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
                            watch_priority=row["watch_priority"] if row["watch_priority"] is not None else 5,
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
                           backoff_until, last_success_at, health_status, watch_priority
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
                            watch_priority=row["watch_priority"] if row["watch_priority"] is not None else 5,
                        )
                    )

                return configs

        except Exception as e:
            logger.error(f"Failed to get watch folders in backoff: {e}")
            return []

    # Error Pattern Analysis (Task 461.18)

    async def record_error_for_pattern_analysis(
        self,
        watch_id: str,
        file_path: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """Record an error for pattern analysis (Task 461.18).

        Tracks errors to detect patterns like:
        - Same file failing repeatedly
        - Specific file types causing errors
        - Time-based error patterns

        Args:
            watch_id: The watch folder ID
            file_path: Path of the file that caused the error
            error_type: Type of error (network, permission, parsing, etc.)
            error_message: Detailed error message
        """
        try:
            now = datetime.now(timezone.utc)
            file_ext = Path(file_path).suffix.lower() or "no_extension"
            hour_of_day = now.hour

            async with self.transaction() as conn:
                # Track file-specific pattern
                conn.execute(
                    """
                    INSERT INTO watch_error_patterns
                    (watch_id, pattern_type, pattern_key, occurrence_count, first_seen_at, last_seen_at, metadata)
                    VALUES (?, 'file_repeated', ?, 1, ?, ?, ?)
                    ON CONFLICT(watch_id, pattern_type, pattern_key) DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        last_seen_at = excluded.last_seen_at,
                        metadata = excluded.metadata
                    """,
                    (
                        watch_id,
                        file_path,
                        now.isoformat(),
                        now.isoformat(),
                        self._serialize_json({"error_type": error_type, "error_message": error_message[:500]}),
                    ),
                )

                # Track file type pattern
                conn.execute(
                    """
                    INSERT INTO watch_error_patterns
                    (watch_id, pattern_type, pattern_key, occurrence_count, first_seen_at, last_seen_at, metadata)
                    VALUES (?, 'file_type', ?, 1, ?, ?, ?)
                    ON CONFLICT(watch_id, pattern_type, pattern_key) DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        last_seen_at = excluded.last_seen_at
                    """,
                    (watch_id, file_ext, now.isoformat(), now.isoformat(), None),
                )

                # Track time-based pattern (by hour of day)
                time_key = f"hour_{hour_of_day:02d}"
                conn.execute(
                    """
                    INSERT INTO watch_error_patterns
                    (watch_id, pattern_type, pattern_key, occurrence_count, first_seen_at, last_seen_at)
                    VALUES (?, 'time_based', ?, 1, ?, ?)
                    ON CONFLICT(watch_id, pattern_type, pattern_key) DO UPDATE SET
                        occurrence_count = occurrence_count + 1,
                        last_seen_at = excluded.last_seen_at
                    """,
                    (watch_id, time_key, now.isoformat(), now.isoformat()),
                )

                # Track error type pattern (network, permission, etc.)
                pattern_type_map = {
                    "network": "network",
                    "timeout": "network",
                    "connection": "network",
                    "permission": "permission",
                    "access": "permission",
                    "denied": "permission",
                }
                detected_type = "file_repeated"  # Default
                error_lower = error_type.lower() + " " + error_message.lower()
                for keyword, ptype in pattern_type_map.items():
                    if keyword in error_lower:
                        detected_type = ptype
                        break

                if detected_type in ("network", "permission"):
                    conn.execute(
                        """
                        INSERT INTO watch_error_patterns
                        (watch_id, pattern_type, pattern_key, occurrence_count, first_seen_at, last_seen_at, metadata)
                        VALUES (?, ?, ?, 1, ?, ?, ?)
                        ON CONFLICT(watch_id, pattern_type, pattern_key) DO UPDATE SET
                            occurrence_count = occurrence_count + 1,
                            last_seen_at = excluded.last_seen_at
                        """,
                        (
                            watch_id,
                            detected_type,
                            error_type,
                            now.isoformat(),
                            now.isoformat(),
                            self._serialize_json({"sample_message": error_message[:200]}),
                        ),
                    )

            logger.debug(f"Recorded error pattern for watch {watch_id}: {file_path}")

        except Exception as e:
            logger.error(f"Failed to record error pattern: {e}")

    async def analyze_error_patterns(
        self,
        watch_id: str,
        threshold_file_repeated: int = 5,
        threshold_file_type: int = 10,
        threshold_time_based: int = 20,
    ) -> list[ErrorPattern]:
        """Analyze error patterns for a watch folder (Task 461.18).

        Identifies systematic vs transient failures based on pattern thresholds.

        Args:
            watch_id: The watch folder ID
            threshold_file_repeated: Errors needed to mark file as systematic
            threshold_file_type: Errors needed to mark file type as systematic
            threshold_time_based: Errors needed to mark time pattern as systematic

        Returns:
            List of error patterns with systematic flag updated
        """
        try:
            thresholds = {
                "file_repeated": threshold_file_repeated,
                "file_type": threshold_file_type,
                "time_based": threshold_time_based,
                "network": 10,
                "permission": 3,
            }

            patterns = []

            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT id, watch_id, pattern_type, pattern_key, occurrence_count,
                           first_seen_at, last_seen_at, is_systematic, confidence_score, metadata
                    FROM watch_error_patterns
                    WHERE watch_id = ?
                    ORDER BY occurrence_count DESC
                    """,
                    (watch_id,),
                )

                for row in cursor.fetchall():
                    pattern_type_str = row["pattern_type"]
                    occurrence_count = row["occurrence_count"]
                    threshold = thresholds.get(pattern_type_str, 10)

                    # Calculate confidence score based on threshold
                    confidence = min(1.0, occurrence_count / threshold)
                    is_systematic = occurrence_count >= threshold

                    # Update the pattern if classification changed
                    if is_systematic != row["is_systematic"] or abs(confidence - (row["confidence_score"] or 0)) > 0.1:
                        self.connection.execute(
                            """
                            UPDATE watch_error_patterns
                            SET is_systematic = ?, confidence_score = ?
                            WHERE id = ?
                            """,
                            (is_systematic, confidence, row["id"]),
                        )
                        self.connection.commit()

                    patterns.append(
                        ErrorPattern(
                            id=row["id"],
                            watch_id=row["watch_id"],
                            pattern_type=ErrorPatternType(pattern_type_str),
                            pattern_key=row["pattern_key"],
                            occurrence_count=occurrence_count,
                            first_seen_at=datetime.fromisoformat(
                                row["first_seen_at"].replace("Z", "+00:00")
                            ) if row["first_seen_at"] else None,
                            last_seen_at=datetime.fromisoformat(
                                row["last_seen_at"].replace("Z", "+00:00")
                            ) if row["last_seen_at"] else None,
                            is_systematic=is_systematic,
                            confidence_score=confidence,
                            metadata=self._deserialize_json(row["metadata"]),
                        )
                    )

            return patterns

        except Exception as e:
            logger.error(f"Failed to analyze error patterns: {e}")
            return []

    async def get_systematic_patterns(self, watch_id: str) -> list[ErrorPattern]:
        """Get patterns classified as systematic failures (Task 461.18).

        Args:
            watch_id: The watch folder ID

        Returns:
            List of error patterns that are systematic
        """
        try:
            patterns = []

            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT id, watch_id, pattern_type, pattern_key, occurrence_count,
                           first_seen_at, last_seen_at, is_systematic, confidence_score, metadata
                    FROM watch_error_patterns
                    WHERE watch_id = ? AND is_systematic = 1
                    ORDER BY confidence_score DESC
                    """,
                    (watch_id,),
                )

                for row in cursor.fetchall():
                    patterns.append(
                        ErrorPattern(
                            id=row["id"],
                            watch_id=row["watch_id"],
                            pattern_type=ErrorPatternType(row["pattern_type"]),
                            pattern_key=row["pattern_key"],
                            occurrence_count=row["occurrence_count"],
                            first_seen_at=datetime.fromisoformat(
                                row["first_seen_at"].replace("Z", "+00:00")
                            ) if row["first_seen_at"] else None,
                            last_seen_at=datetime.fromisoformat(
                                row["last_seen_at"].replace("Z", "+00:00")
                            ) if row["last_seen_at"] else None,
                            is_systematic=True,
                            confidence_score=row["confidence_score"] or 0.0,
                            metadata=self._deserialize_json(row["metadata"]),
                        )
                    )

            return patterns

        except Exception as e:
            logger.error(f"Failed to get systematic patterns: {e}")
            return []

    async def add_watch_exclusion(
        self,
        watch_id: str,
        exclusion_type: ExclusionType,
        exclusion_value: str,
        reason: str,
        error_count: int = 1,
        is_permanent: bool = False,
        expiry_hours: int | None = None,
    ) -> bool:
        """Add a file/pattern/directory to the exclusion list (Task 461.18).

        Args:
            watch_id: The watch folder ID
            exclusion_type: Type of exclusion (file, pattern, directory)
            exclusion_value: The value to exclude
            reason: Why this is being excluded
            error_count: Number of errors that led to exclusion
            is_permanent: Whether this is a permanent exclusion
            expiry_hours: Hours until exclusion expires (for transient failures)

        Returns:
            True if exclusion was added successfully
        """
        try:
            now = datetime.now(timezone.utc)
            expires_at = None
            if expiry_hours and not is_permanent:
                from datetime import timedelta
                expires_at = now + timedelta(hours=expiry_hours)

            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO watch_exclusions
                    (watch_id, exclusion_type, exclusion_value, reason, error_count, created_at, expires_at, is_permanent)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(watch_id, exclusion_type, exclusion_value) DO UPDATE SET
                        error_count = error_count + excluded.error_count,
                        reason = excluded.reason,
                        expires_at = excluded.expires_at,
                        is_permanent = excluded.is_permanent
                    """,
                    (
                        watch_id,
                        exclusion_type.value,
                        exclusion_value,
                        reason,
                        error_count,
                        now.isoformat(),
                        expires_at.isoformat() if expires_at else None,
                        is_permanent,
                    ),
                )

            logger.info(
                f"Added exclusion for watch {watch_id}: {exclusion_type.value}={exclusion_value} "
                f"(permanent={is_permanent})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to add watch exclusion: {e}")
            return False

    async def get_watch_exclusions(
        self,
        watch_id: str,
        include_expired: bool = False,
    ) -> list[WatchExclusion]:
        """Get all exclusions for a watch folder (Task 461.18).

        Args:
            watch_id: The watch folder ID
            include_expired: Whether to include expired exclusions

        Returns:
            List of exclusions
        """
        try:
            exclusions = []
            now = datetime.now(timezone.utc)

            with self._lock:
                if include_expired:
                    cursor = self.connection.execute(
                        """
                        SELECT id, watch_id, exclusion_type, exclusion_value, reason,
                               error_count, created_at, expires_at, is_permanent, metadata
                        FROM watch_exclusions
                        WHERE watch_id = ?
                        ORDER BY created_at DESC
                        """,
                        (watch_id,),
                    )
                else:
                    cursor = self.connection.execute(
                        """
                        SELECT id, watch_id, exclusion_type, exclusion_value, reason,
                               error_count, created_at, expires_at, is_permanent, metadata
                        FROM watch_exclusions
                        WHERE watch_id = ? AND (is_permanent = 1 OR expires_at IS NULL OR expires_at > ?)
                        ORDER BY created_at DESC
                        """,
                        (watch_id, now.isoformat()),
                    )

                for row in cursor.fetchall():
                    exclusions.append(
                        WatchExclusion(
                            id=row["id"],
                            watch_id=row["watch_id"],
                            exclusion_type=ExclusionType(row["exclusion_type"]),
                            exclusion_value=row["exclusion_value"],
                            reason=row["reason"],
                            error_count=row["error_count"],
                            created_at=datetime.fromisoformat(
                                row["created_at"].replace("Z", "+00:00")
                            ) if row["created_at"] else None,
                            expires_at=datetime.fromisoformat(
                                row["expires_at"].replace("Z", "+00:00")
                            ) if row["expires_at"] else None,
                            is_permanent=bool(row["is_permanent"]),
                            metadata=self._deserialize_json(row["metadata"]),
                        )
                    )

            return exclusions

        except Exception as e:
            logger.error(f"Failed to get watch exclusions: {e}")
            return []

    async def is_file_excluded(self, watch_id: str, file_path: str) -> bool:
        """Check if a file is excluded from processing (Task 461.18).

        Checks against:
        - Exact file path exclusions
        - Pattern exclusions (glob matching)
        - Directory exclusions

        Args:
            watch_id: The watch folder ID
            file_path: Path to check

        Returns:
            True if file should be excluded
        """
        try:
            import fnmatch
            now = datetime.now(timezone.utc)
            file_path_obj = Path(file_path)

            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT exclusion_type, exclusion_value
                    FROM watch_exclusions
                    WHERE watch_id = ? AND (is_permanent = 1 OR expires_at IS NULL OR expires_at > ?)
                    """,
                    (watch_id, now.isoformat()),
                )

                for row in cursor.fetchall():
                    exclusion_type = row["exclusion_type"]
                    exclusion_value = row["exclusion_value"]

                    if exclusion_type == "file" and file_path == exclusion_value:
                        return True
                    elif exclusion_type == "pattern" and fnmatch.fnmatch(file_path, exclusion_value):
                        return True
                    elif exclusion_type == "directory":
                        try:
                            if file_path_obj.is_relative_to(exclusion_value):
                                return True
                        except (ValueError, TypeError):
                            if file_path.startswith(exclusion_value):
                                return True

            return False

        except Exception as e:
            logger.error(f"Failed to check file exclusion: {e}")
            return False

    async def remove_watch_exclusion(
        self,
        watch_id: str,
        exclusion_type: ExclusionType,
        exclusion_value: str,
    ) -> bool:
        """Remove an exclusion (Task 461.18).

        Args:
            watch_id: The watch folder ID
            exclusion_type: Type of exclusion
            exclusion_value: Value to remove

        Returns:
            True if exclusion was removed
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM watch_exclusions
                    WHERE watch_id = ? AND exclusion_type = ? AND exclusion_value = ?
                    """,
                    (watch_id, exclusion_type.value, exclusion_value),
                )
                return cursor.rowcount > 0

        except Exception as e:
            logger.error(f"Failed to remove watch exclusion: {e}")
            return False

    async def cleanup_expired_exclusions(self) -> int:
        """Remove expired exclusions from all watches (Task 461.18).

        Returns:
            Number of exclusions removed
        """
        try:
            now = datetime.now(timezone.utc)

            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    DELETE FROM watch_exclusions
                    WHERE is_permanent = 0 AND expires_at IS NOT NULL AND expires_at < ?
                    """,
                    (now.isoformat(),),
                )
                removed = cursor.rowcount

            if removed > 0:
                logger.info(f"Cleaned up {removed} expired exclusions")

            return removed

        except Exception as e:
            logger.error(f"Failed to cleanup expired exclusions: {e}")
            return 0

    async def auto_exclude_systematic_failures(
        self,
        watch_id: str,
        threshold_file_repeated: int = 5,
    ) -> int:
        """Auto-exclude files with systematic failures (Task 461.18).

        Analyzes error patterns and automatically adds exclusions for
        files that repeatedly fail.

        Args:
            watch_id: The watch folder ID
            threshold_file_repeated: Errors needed to auto-exclude a file

        Returns:
            Number of files excluded
        """
        try:
            excluded_count = 0

            # Analyze patterns first
            patterns = await self.analyze_error_patterns(
                watch_id,
                threshold_file_repeated=threshold_file_repeated,
            )

            for pattern in patterns:
                if pattern.is_systematic and pattern.pattern_type == ErrorPatternType.FILE_REPEATED:
                    # This is a file that repeatedly fails - add to exclusions
                    file_path = pattern.pattern_key
                    reason = f"Systematic failure: {pattern.occurrence_count} errors"
                    if pattern.metadata and pattern.metadata.get("error_message"):
                        reason += f" - {pattern.metadata['error_message'][:100]}"

                    success = await self.add_watch_exclusion(
                        watch_id=watch_id,
                        exclusion_type=ExclusionType.FILE,
                        exclusion_value=file_path,
                        reason=reason,
                        error_count=pattern.occurrence_count,
                        is_permanent=True,  # Systematic failures are permanent
                    )
                    if success:
                        excluded_count += 1

            if excluded_count > 0:
                logger.info(f"Auto-excluded {excluded_count} files for watch {watch_id}")

            return excluded_count

        except Exception as e:
            logger.error(f"Failed to auto-exclude systematic failures: {e}")
            return 0

    async def get_error_pattern_summary(self, watch_id: str) -> dict[str, Any]:
        """Get summary of error patterns for a watch (Task 461.18).

        Args:
            watch_id: The watch folder ID

        Returns:
            Summary dictionary with pattern statistics
        """
        try:
            summary = {
                "watch_id": watch_id,
                "total_patterns": 0,
                "systematic_patterns": 0,
                "by_type": {},
                "top_failing_files": [],
                "top_failing_extensions": [],
            }

            with self._lock:
                # Count patterns by type
                cursor = self.connection.execute(
                    """
                    SELECT pattern_type, COUNT(*) as count,
                           SUM(CASE WHEN is_systematic = 1 THEN 1 ELSE 0 END) as systematic_count,
                           SUM(occurrence_count) as total_occurrences
                    FROM watch_error_patterns
                    WHERE watch_id = ?
                    GROUP BY pattern_type
                    """,
                    (watch_id,),
                )

                for row in cursor.fetchall():
                    summary["by_type"][row["pattern_type"]] = {
                        "patterns": row["count"],
                        "systematic": row["systematic_count"],
                        "total_occurrences": row["total_occurrences"],
                    }
                    summary["total_patterns"] += row["count"]
                    summary["systematic_patterns"] += row["systematic_count"]

                # Get top failing files
                cursor = self.connection.execute(
                    """
                    SELECT pattern_key, occurrence_count, confidence_score
                    FROM watch_error_patterns
                    WHERE watch_id = ? AND pattern_type = 'file_repeated'
                    ORDER BY occurrence_count DESC
                    LIMIT 10
                    """,
                    (watch_id,),
                )
                summary["top_failing_files"] = [
                    {"file": row["pattern_key"], "errors": row["occurrence_count"], "confidence": row["confidence_score"]}
                    for row in cursor.fetchall()
                ]

                # Get top failing extensions
                cursor = self.connection.execute(
                    """
                    SELECT pattern_key, occurrence_count, confidence_score
                    FROM watch_error_patterns
                    WHERE watch_id = ? AND pattern_type = 'file_type'
                    ORDER BY occurrence_count DESC
                    LIMIT 10
                    """,
                    (watch_id,),
                )
                summary["top_failing_extensions"] = [
                    {"extension": row["pattern_key"], "errors": row["occurrence_count"], "confidence": row["confidence_score"]}
                    for row in cursor.fetchall()
                ]

            return summary

        except Exception as e:
            logger.error(f"Failed to get error pattern summary: {e}")
            return {"watch_id": watch_id, "error": str(e)}

    async def clear_error_patterns(self, watch_id: str) -> int:
        """Clear all error patterns for a watch (Task 461.18).

        Useful when resetting a watch's error state.

        Args:
            watch_id: The watch folder ID

        Returns:
            Number of patterns cleared
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM watch_error_patterns WHERE watch_id = ?",
                    (watch_id,),
                )
                cleared = cursor.rowcount

            if cleared > 0:
                logger.info(f"Cleared {cleared} error patterns for watch {watch_id}")

            return cleared

        except Exception as e:
            logger.error(f"Failed to clear error patterns: {e}")
            return 0

    # Graceful Degradation Management (Task 461.19)

    def _get_degradation_state_key(self) -> str:
        """Get the system state key for degradation state."""
        return "graceful_degradation_state"

    async def get_degradation_state(self) -> DegradationState:
        """Get current degradation state (Task 461.19).

        Returns:
            Current degradation state
        """
        try:
            with self._lock:
                cursor = self.connection.execute(
                    "SELECT value FROM system_state WHERE key = ?",
                    (self._get_degradation_state_key(),),
                )
                row = cursor.fetchone()

                if row and row["value"]:
                    data = self._deserialize_json(row["value"])
                    if data:
                        return DegradationState(
                            level=DegradationLevel(data.get("level", "normal")),
                            entered_at=datetime.fromisoformat(data["entered_at"].replace("Z", "+00:00"))
                                if data.get("entered_at") else datetime.now(timezone.utc),
                            last_check_at=datetime.fromisoformat(data["last_check_at"].replace("Z", "+00:00"))
                                if data.get("last_check_at") else datetime.now(timezone.utc),
                            queue_depth=data.get("queue_depth", 0),
                            throughput=data.get("throughput", 0.0),
                            memory_usage=data.get("memory_usage", 0.0),
                            paused_watch_ids=data.get("paused_watch_ids", []),
                            polling_interval_multiplier=data.get("polling_interval_multiplier", 1.0),
                            recovery_step=data.get("recovery_step", 0),
                        )

            return DegradationState()

        except Exception as e:
            logger.error(f"Failed to get degradation state: {e}")
            return DegradationState()

    async def save_degradation_state(self, state: DegradationState) -> bool:
        """Save degradation state (Task 461.19).

        Args:
            state: The degradation state to save

        Returns:
            True if saved successfully
        """
        try:
            state.last_check_at = datetime.now(timezone.utc)

            async with self.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO system_state (key, value, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    (self._get_degradation_state_key(), self._serialize_json(state.to_dict())),
                )

            return True

        except Exception as e:
            logger.error(f"Failed to save degradation state: {e}")
            return False

    async def check_and_update_degradation(
        self,
        config: DegradationConfig | None = None,
    ) -> DegradationState:
        """Check system metrics and update degradation state (Task 461.19).

        Monitors queue depth, throughput, and memory usage to determine
        the appropriate degradation level.

        Args:
            config: Degradation configuration (uses defaults if None)

        Returns:
            Updated degradation state
        """
        try:
            config = config or DegradationConfig()
            current_state = await self.get_degradation_state()
            now = datetime.now(timezone.utc)

            # Get current metrics
            queue_depth = await self._get_total_queue_depth()
            throughput = await self._get_recent_throughput()
            memory_usage = self._get_memory_usage()

            # Determine new degradation level
            new_level = self._calculate_degradation_level(
                config, queue_depth, throughput, memory_usage
            )

            # Check if we should transition
            old_level = current_state.level
            should_transition = False

            if new_level.value != old_level.value:
                # Determine transition direction
                level_order = [
                    DegradationLevel.NORMAL,
                    DegradationLevel.LIGHT,
                    DegradationLevel.MODERATE,
                    DegradationLevel.SEVERE,
                    DegradationLevel.CRITICAL,
                ]
                old_idx = level_order.index(old_level)
                new_idx = level_order.index(new_level)

                if new_idx > old_idx:
                    # Degrading - transition immediately
                    should_transition = True
                    current_state.recovery_step = 0
                elif new_idx < old_idx:
                    # Recovering - check cooldown and use gradual recovery
                    time_since_entered = (now - current_state.entered_at).total_seconds()
                    if time_since_entered >= config.recovery_cooldown_seconds:
                        # Recover one step at a time
                        current_state.recovery_step += 1
                        if current_state.recovery_step >= config.recovery_steps:
                            should_transition = True
                            current_state.recovery_step = 0

            if should_transition:
                logger.warning(
                    f"Degradation level changed: {old_level.value} -> {new_level.value} "
                    f"(queue={queue_depth}, throughput={throughput:.1f}, memory={memory_usage:.1%})"
                )
                current_state.level = new_level
                current_state.entered_at = now

                # Apply degradation actions
                await self._apply_degradation_actions(config, current_state)

            # Update metrics
            current_state.queue_depth = queue_depth
            current_state.throughput = throughput
            current_state.memory_usage = memory_usage
            current_state.polling_interval_multiplier = self._get_polling_multiplier(config, current_state.level)

            # Save state
            await self.save_degradation_state(current_state)

            return current_state

        except Exception as e:
            logger.error(f"Failed to check degradation: {e}")
            return DegradationState()

    def _calculate_degradation_level(
        self,
        config: DegradationConfig,
        queue_depth: int,
        throughput: float,
        memory_usage: float,
    ) -> DegradationLevel:
        """Calculate appropriate degradation level based on metrics."""
        # Check queue depth
        if queue_depth >= config.queue_depth_critical:
            return DegradationLevel.CRITICAL
        if queue_depth >= config.queue_depth_severe:
            return DegradationLevel.SEVERE
        if queue_depth >= config.queue_depth_moderate:
            return DegradationLevel.MODERATE
        if queue_depth >= config.queue_depth_light:
            return DegradationLevel.LIGHT

        # Check throughput
        if config.throughput_target > 0:
            throughput_ratio = throughput / config.throughput_target
            if throughput_ratio < config.throughput_critical:
                return DegradationLevel.SEVERE
            if throughput_ratio < config.throughput_warning:
                return DegradationLevel.MODERATE

        # Check memory
        if memory_usage >= config.memory_critical:
            return DegradationLevel.SEVERE
        if memory_usage >= config.memory_warning:
            return DegradationLevel.MODERATE

        return DegradationLevel.NORMAL

    def _get_polling_multiplier(
        self,
        config: DegradationConfig,
        level: DegradationLevel,
    ) -> float:
        """Get polling interval multiplier for degradation level."""
        multipliers = {
            DegradationLevel.NORMAL: 1.0,
            DegradationLevel.LIGHT: config.polling_interval_light,
            DegradationLevel.MODERATE: config.polling_interval_moderate,
            DegradationLevel.SEVERE: config.polling_interval_severe,
            DegradationLevel.CRITICAL: config.polling_interval_critical,
        }
        return multipliers.get(level, 1.0)

    async def _apply_degradation_actions(
        self,
        config: DegradationConfig,
        state: DegradationState,
    ) -> None:
        """Apply degradation actions based on current level."""
        try:
            level = state.level
            previously_paused = set(state.paused_watch_ids)
            newly_paused = set()

            # Determine which watches to pause based on priority
            pause_threshold = {
                DegradationLevel.NORMAL: -1,  # Don't pause any
                DegradationLevel.LIGHT: -1,  # Don't pause any
                DegradationLevel.MODERATE: config.pause_priority_moderate,
                DegradationLevel.SEVERE: config.pause_priority_severe,
                DegradationLevel.CRITICAL: config.pause_priority_critical,
            }.get(level, -1)

            if pause_threshold >= 0:
                # Get watches with priority at or below threshold
                watches = await self.get_all_watch_folder_configs()
                for watch in watches:
                    effective_priority = watch.calculate_effective_priority()
                    if effective_priority <= pause_threshold and watch.enabled:
                        newly_paused.add(watch.watch_id)

            # Resume watches that no longer need to be paused
            to_resume = previously_paused - newly_paused
            for watch_id in to_resume:
                await self._set_watch_degradation_paused(watch_id, False)
                logger.info(f"Resumed watch {watch_id} after degradation recovery")

            # Pause watches that need to be paused
            to_pause = newly_paused - previously_paused
            for watch_id in to_pause:
                await self._set_watch_degradation_paused(watch_id, True)
                logger.warning(f"Paused watch {watch_id} due to degradation level {level.value}")

            state.paused_watch_ids = list(newly_paused)

        except Exception as e:
            logger.error(f"Failed to apply degradation actions: {e}")

    async def _set_watch_degradation_paused(self, watch_id: str, paused: bool) -> None:
        """Set watch paused state due to degradation."""
        try:
            config = await self.get_watch_folder_config(watch_id)
            if config:
                # Store paused state in metadata
                metadata = config.metadata or {}
                metadata["degradation_paused"] = paused
                metadata["degradation_paused_at"] = datetime.now(timezone.utc).isoformat() if paused else None

                async with self.transaction() as conn:
                    conn.execute(
                        """
                        UPDATE watch_folders
                        SET metadata = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE watch_id = ?
                        """,
                        (self._serialize_json(metadata), watch_id),
                    )

        except Exception as e:
            logger.error(f"Failed to set watch degradation pause: {e}")

    async def _get_total_queue_depth(self) -> int:
        """Get total queue depth across all collections."""
        try:
            with self._lock:
                # Check ingestion_queue
                cursor = self.connection.execute(
                    "SELECT COUNT(*) as count FROM ingestion_queue"
                )
                row = cursor.fetchone()
                ingestion_count = row["count"] if row else 0

                # Check processing_queue
                cursor = self.connection.execute(
                    "SELECT COUNT(*) as count FROM processing_queue"
                )
                row = cursor.fetchone()
                processing_count = row["count"] if row else 0

                return ingestion_count + processing_count

        except Exception as e:
            logger.error(f"Failed to get queue depth: {e}")
            return 0

    async def _get_recent_throughput(self, window_seconds: int = 60) -> float:
        """Get recent processing throughput (items per second)."""
        try:
            now = datetime.now(timezone.utc)
            window_start = now - __import__("datetime").timedelta(seconds=window_seconds)

            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT COUNT(*) as count
                    FROM processing_history
                    WHERE created_at >= ? AND status = 'completed'
                    """,
                    (window_start.isoformat(),),
                )
                row = cursor.fetchone()
                count = row["count"] if row else 0

            return count / window_seconds if window_seconds > 0 else 0.0

        except Exception as e:
            logger.error(f"Failed to get throughput: {e}")
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current memory usage as fraction (0.0-1.0)."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            virtual_memory = psutil.virtual_memory()
            return memory_info.rss / virtual_memory.total

        except ImportError:
            # psutil not available, return safe value
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return 0.0

    async def is_watch_degradation_paused(self, watch_id: str) -> bool:
        """Check if a watch is paused due to degradation (Task 461.19).

        Args:
            watch_id: The watch folder ID

        Returns:
            True if watch is paused due to degradation
        """
        try:
            config = await self.get_watch_folder_config(watch_id)
            if config and config.metadata:
                return config.metadata.get("degradation_paused", False)
            return False

        except Exception as e:
            logger.error(f"Failed to check watch degradation pause: {e}")
            return False

    async def get_adjusted_polling_interval(
        self,
        base_interval_ms: int,
    ) -> int:
        """Get adjusted polling interval based on degradation state (Task 461.19).

        Args:
            base_interval_ms: Base polling interval in milliseconds

        Returns:
            Adjusted polling interval
        """
        try:
            state = await self.get_degradation_state()
            return int(base_interval_ms * state.polling_interval_multiplier)

        except Exception as e:
            logger.error(f"Failed to get adjusted polling interval: {e}")
            return base_interval_ms

    async def get_degradation_summary(self) -> dict[str, Any]:
        """Get summary of current degradation state (Task 461.19).

        Returns:
            Summary dictionary with degradation information
        """
        try:
            state = await self.get_degradation_state()
            config = DegradationConfig()

            return {
                "level": state.level.value,
                "is_degraded": state.is_degraded(),
                "entered_at": state.entered_at.isoformat() if state.entered_at else None,
                "duration_seconds": (datetime.now(timezone.utc) - state.entered_at).total_seconds()
                    if state.entered_at else 0,
                "metrics": {
                    "queue_depth": state.queue_depth,
                    "throughput": state.throughput,
                    "memory_usage": state.memory_usage,
                },
                "thresholds": {
                    "queue_depth_critical": config.queue_depth_critical,
                    "throughput_target": config.throughput_target,
                    "memory_critical": config.memory_critical,
                },
                "actions": {
                    "polling_interval_multiplier": state.polling_interval_multiplier,
                    "paused_watch_count": len(state.paused_watch_ids),
                    "paused_watch_ids": state.paused_watch_ids,
                },
                "recovery": {
                    "step": state.recovery_step,
                    "total_steps": config.recovery_steps,
                    "in_recovery": state.recovery_step > 0,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get degradation summary: {e}")
            return {"error": str(e)}

    async def force_degradation_level(
        self,
        level: DegradationLevel,
        config: DegradationConfig | None = None,
    ) -> DegradationState:
        """Force a specific degradation level (for testing) (Task 461.19).

        Args:
            level: The degradation level to force
            config: Optional configuration

        Returns:
            Updated degradation state
        """
        try:
            config = config or DegradationConfig()
            state = await self.get_degradation_state()

            old_level = state.level
            state.level = level
            state.entered_at = datetime.now(timezone.utc)
            state.recovery_step = 0
            state.polling_interval_multiplier = self._get_polling_multiplier(config, level)

            await self._apply_degradation_actions(config, state)
            await self.save_degradation_state(state)

            logger.warning(f"Forced degradation level: {old_level.value} -> {level.value}")

            return state

        except Exception as e:
            logger.error(f"Failed to force degradation level: {e}")
            return DegradationState()

    async def reset_degradation_state(self) -> bool:
        """Reset degradation state to normal (Task 461.19).

        Resumes all paused watches and resets metrics.

        Returns:
            True if reset successfully
        """
        try:
            state = await self.get_degradation_state()

            # Resume all paused watches
            for watch_id in state.paused_watch_ids:
                await self._set_watch_degradation_paused(watch_id, False)
                logger.info(f"Resumed watch {watch_id} after degradation reset")

            # Reset state
            new_state = DegradationState(
                level=DegradationLevel.NORMAL,
                entered_at=datetime.now(timezone.utc),
                paused_watch_ids=[],
                polling_interval_multiplier=1.0,
                recovery_step=0,
            )

            await self.save_degradation_state(new_state)
            logger.info("Degradation state reset to normal")

            return True

        except Exception as e:
            logger.error(f"Failed to reset degradation state: {e}")
            return False

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

    # =========================================================================
    # Active Projects (Task 36 - code audit round 2)
    # =========================================================================

    async def register_active_project(
        self,
        project_id: str,
        tenant_id: str,
        watch_folder_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActiveProjectState | None:
        """Register or update an active project for fairness scheduling.

        Creates a new active project entry or updates an existing one.
        Called when:
        - Watch folder scanner detects activity in a project
        - Query is executed for a project
        - File is ingested for a project

        Args:
            project_id: Unique project identifier (tenant_id or normalized path)
            tenant_id: Tenant identifier for multi-tenant isolation
            watch_folder_id: Optional reference to watch_folders table
            metadata: Optional JSON metadata for extensibility

        Returns:
            ActiveProjectState on success, None on failure
        """
        try:
            now = datetime.now(timezone.utc)
            async with self.transaction() as conn:
                # Check if project exists
                cursor = conn.execute(
                    "SELECT project_id FROM active_projects WHERE project_id = ?",
                    (project_id,),
                )
                existing = cursor.fetchone()

                if existing:
                    # Update existing project's activity
                    conn.execute(
                        """
                        UPDATE active_projects
                        SET last_activity_at = ?,
                            watch_folder_id = COALESCE(?, watch_folder_id),
                            watch_enabled = CASE WHEN ? IS NOT NULL THEN 1 ELSE watch_enabled END,
                            metadata = COALESCE(?, metadata)
                        WHERE project_id = ?
                        """,
                        (
                            now.isoformat(),
                            watch_folder_id,
                            watch_folder_id,
                            self._serialize_json(metadata) if metadata else None,
                            project_id,
                        ),
                    )
                else:
                    # Insert new project
                    conn.execute(
                        """
                        INSERT INTO active_projects
                        (project_id, tenant_id, last_activity_at, items_processed_count,
                         items_in_queue, watch_enabled, watch_folder_id, created_at,
                         updated_at, metadata)
                        VALUES (?, ?, ?, 0, 0, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id,
                            tenant_id,
                            now.isoformat(),
                            1 if watch_folder_id else 0,
                            watch_folder_id,
                            now.isoformat(),
                            now.isoformat(),
                            self._serialize_json(metadata),
                        ),
                    )

            logger.debug(f"Registered active project: {project_id}")
            return await self.get_active_project(project_id)

        except Exception as e:
            logger.error(f"Failed to register active project {project_id}: {e}")
            return None

    async def get_active_project(self, project_id: str) -> ActiveProjectState | None:
        """Get active project state by project_id.

        Args:
            project_id: Unique project identifier

        Returns:
            ActiveProjectState on success, None if not found
        """
        try:
            with self._lock:
                cursor = self.connection.execute(
                    """
                    SELECT project_id, tenant_id, last_activity_at, items_processed_count,
                           items_in_queue, watch_enabled, watch_folder_id, created_at,
                           updated_at, metadata
                    FROM active_projects
                    WHERE project_id = ?
                    """,
                    (project_id,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return ActiveProjectState(
                    project_id=row["project_id"],
                    tenant_id=row["tenant_id"],
                    last_activity_at=datetime.fromisoformat(
                        row["last_activity_at"].replace("Z", "+00:00")
                    ) if row["last_activity_at"] else None,
                    items_processed_count=row["items_processed_count"] or 0,
                    items_in_queue=row["items_in_queue"] or 0,
                    watch_enabled=bool(row["watch_enabled"]),
                    watch_folder_id=row["watch_folder_id"],
                    created_at=datetime.fromisoformat(
                        row["created_at"].replace("Z", "+00:00")
                    ) if row["created_at"] else None,
                    updated_at=datetime.fromisoformat(
                        row["updated_at"].replace("Z", "+00:00")
                    ) if row["updated_at"] else None,
                    metadata=self._deserialize_json(row["metadata"]),
                )

        except Exception as e:
            logger.error(f"Failed to get active project {project_id}: {e}")
            return None

    async def list_active_projects(
        self,
        tenant_id: str | None = None,
        watch_enabled_only: bool = False,
        order_by_activity: bool = True,
        limit: int | None = None,
    ) -> list[ActiveProjectState]:
        """List active projects with optional filtering.

        Args:
            tenant_id: Optional filter by tenant
            watch_enabled_only: If True, only return projects with watches
            order_by_activity: If True, order by last_activity_at DESC
            limit: Optional limit on number of results

        Returns:
            List of ActiveProjectState objects
        """
        try:
            with self._lock:
                sql = """
                    SELECT project_id, tenant_id, last_activity_at, items_processed_count,
                           items_in_queue, watch_enabled, watch_folder_id, created_at,
                           updated_at, metadata
                    FROM active_projects
                    WHERE 1=1
                """
                params = []

                if tenant_id:
                    sql += " AND tenant_id = ?"
                    params.append(tenant_id)

                if watch_enabled_only:
                    sql += " AND watch_enabled = 1"

                if order_by_activity:
                    sql += " ORDER BY last_activity_at DESC"

                if limit:
                    sql += " LIMIT ?"
                    params.append(limit)

                cursor = self.connection.execute(sql, params)
                rows = cursor.fetchall()

                return [
                    ActiveProjectState(
                        project_id=row["project_id"],
                        tenant_id=row["tenant_id"],
                        last_activity_at=datetime.fromisoformat(
                            row["last_activity_at"].replace("Z", "+00:00")
                        ) if row["last_activity_at"] else None,
                        items_processed_count=row["items_processed_count"] or 0,
                        items_in_queue=row["items_in_queue"] or 0,
                        watch_enabled=bool(row["watch_enabled"]),
                        watch_folder_id=row["watch_folder_id"],
                        created_at=datetime.fromisoformat(
                            row["created_at"].replace("Z", "+00:00")
                        ) if row["created_at"] else None,
                        updated_at=datetime.fromisoformat(
                            row["updated_at"].replace("Z", "+00:00")
                        ) if row["updated_at"] else None,
                        metadata=self._deserialize_json(row["metadata"]),
                    )
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to list active projects: {e}")
            return []

    async def update_active_project_activity(
        self,
        project_id: str,
        items_processed: int = 1,
    ) -> bool:
        """Update project activity after processing items.

        Called by the unified queue processor after successfully processing
        queue items for a project.

        Args:
            project_id: Project identifier
            items_processed: Number of items processed (default: 1)

        Returns:
            True on success, False on failure
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    """
                    UPDATE active_projects
                    SET last_activity_at = datetime('now'),
                        items_processed_count = items_processed_count + ?
                    WHERE project_id = ?
                    """,
                    (items_processed, project_id),
                )

                if cursor.rowcount == 0:
                    logger.warning(f"No active project found for update: {project_id}")
                    return False

            logger.debug(f"Updated activity for project {project_id}: +{items_processed} items")
            return True

        except Exception as e:
            logger.error(f"Failed to update project activity {project_id}: {e}")
            return False

    async def update_active_project_queue_count(
        self,
        project_id: str,
        queue_delta: int,
    ) -> bool:
        """Update the items_in_queue count for a project.

        Called when:
        - Items are added to queue (positive delta)
        - Items are removed from queue (negative delta)

        Args:
            project_id: Project identifier
            queue_delta: Change in queue count (positive or negative)

        Returns:
            True on success, False on failure
        """
        try:
            async with self.transaction() as conn:
                # Use MAX to prevent negative counts
                cursor = conn.execute(
                    """
                    UPDATE active_projects
                    SET items_in_queue = MAX(0, items_in_queue + ?)
                    WHERE project_id = ?
                    """,
                    (queue_delta, project_id),
                )

                if cursor.rowcount == 0:
                    logger.warning(f"No active project found for queue update: {project_id}")
                    return False

            logger.debug(f"Updated queue count for project {project_id}: delta={queue_delta}")
            return True

        except Exception as e:
            logger.error(f"Failed to update project queue count {project_id}: {e}")
            return False

    async def garbage_collect_stale_projects(
        self,
        max_inactive_hours: int = 24,
    ) -> int:
        """Remove projects that have been inactive for too long.

        Projects with no activity for more than max_inactive_hours are
        removed from tracking. This frees up resources and keeps the
        active_projects table focused on truly active projects.

        Args:
            max_inactive_hours: Hours of inactivity before removal (default: 24)

        Returns:
            Number of projects removed
        """
        try:
            async with self.transaction() as conn:
                # First, log what we're about to remove
                cursor = conn.execute(
                    """
                    SELECT project_id, tenant_id, last_activity_at, items_processed_count
                    FROM active_projects
                    WHERE datetime(last_activity_at) < datetime('now', ?)
                    """,
                    (f"-{max_inactive_hours} hours",),
                )
                stale_projects = cursor.fetchall()

                if stale_projects:
                    for row in stale_projects:
                        logger.info(
                            f"Removing stale project: {row['project_id']} "
                            f"(last activity: {row['last_activity_at']}, "
                            f"processed: {row['items_processed_count']} items)"
                        )

                # Delete stale projects
                cursor = conn.execute(
                    """
                    DELETE FROM active_projects
                    WHERE datetime(last_activity_at) < datetime('now', ?)
                    """,
                    (f"-{max_inactive_hours} hours",),
                )

                removed_count = cursor.rowcount

            if removed_count > 0:
                logger.info(f"Garbage collected {removed_count} stale projects (>{max_inactive_hours}h inactive)")

            return removed_count

        except Exception as e:
            logger.error(f"Failed to garbage collect stale projects: {e}")
            return 0

    async def get_active_projects_stats(self) -> dict[str, Any]:
        """Get statistics about active projects.

        Uses the v_active_projects_stats view created by migration 004.

        Returns:
            Dictionary with statistics or empty dict on failure
        """
        try:
            with self._lock:
                cursor = self.connection.execute(
                    "SELECT * FROM v_active_projects_stats"
                )
                row = cursor.fetchone()

                if not row:
                    return {}

                return {
                    "total_projects": row["total_projects"] or 0,
                    "watched_projects": row["watched_projects"] or 0,
                    "total_items_processed": row["total_items_processed"] or 0,
                    "total_items_in_queue": row["total_items_in_queue"] or 0,
                    "most_recent_activity": row["most_recent_activity"],
                    "oldest_activity": row["oldest_activity"],
                    "active_last_hour": row["active_last_hour"] or 0,
                    "active_last_24h": row["active_last_24h"] or 0,
                }

        except Exception as e:
            logger.error(f"Failed to get active projects stats: {e}")
            return {}

    async def get_stale_projects(self) -> list[dict[str, Any]]:
        """Get projects that are candidates for garbage collection.

        Uses the v_stale_projects view created by migration 004.

        Returns:
            List of stale project info dictionaries
        """
        try:
            with self._lock:
                cursor = self.connection.execute(
                    "SELECT * FROM v_stale_projects"
                )
                rows = cursor.fetchall()

                return [
                    {
                        "project_id": row["project_id"],
                        "tenant_id": row["tenant_id"],
                        "last_activity_at": row["last_activity_at"],
                        "items_processed_count": row["items_processed_count"],
                        "created_at": row["created_at"],
                        "days_inactive": row["days_inactive"],
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Failed to get stale projects: {e}")
            return []

    async def remove_active_project(self, project_id: str) -> bool:
        """Remove an active project from tracking.

        Args:
            project_id: Project identifier to remove

        Returns:
            True if removed, False if not found or error
        """
        try:
            async with self.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM active_projects WHERE project_id = ?",
                    (project_id,),
                )

                if cursor.rowcount == 0:
                    logger.warning(f"Active project not found for removal: {project_id}")
                    return False

            logger.info(f"Removed active project: {project_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove active project {project_id}: {e}")
            return False
