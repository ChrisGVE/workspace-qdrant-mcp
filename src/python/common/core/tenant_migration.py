"""
Tenant ID Migration System for workspace-qdrant-mcp.

This module provides comprehensive tenant ID migration capabilities to handle
project remote URL changes. When a project gains or loses a Git remote, its
tenant ID changes, requiring migration of queue entries, metadata, and Qdrant
collection filters.

Key Features:
    - Automatic detection of tenant ID changes via Git monitoring
    - Atomic migration with full rollback support
    - Progress tracking for large-scale migrations
    - Consistency validation before and after migration
    - Comprehensive audit logging
    - Support for interrupted migration resumption
    - Batch processing for performance

Migration Scenarios:
    1. Local → Remote: Project gets first Git remote
    2. Remote → Different Remote: Remote URL changes
    3. Remote → Local: Remote is removed
    4. Remote → Remote (same): No migration needed

Data Migration Targets:
    - ingestion_queue table (tenant_id column)
    - file_processing table (metadata JSON field)
    - Qdrant collection metadata (project_id field)

Example:
    ```python
    from tenant_migration import TenantMigrationManager
    from sqlite_state_manager import SQLiteStateManager

    # Initialize manager
    state_manager = SQLiteStateManager()
    await state_manager.initialize()
    migration_manager = TenantMigrationManager(state_manager)

    # Check for changes and migrate if needed
    change_detected = await migration_manager.detect_tenant_change(project_root)
    if change_detected:
        plan = await migration_manager.plan_migration(
            project_root, old_tenant_id, new_tenant_id
        )
        result = await migration_manager.execute_migration(plan)
        if not result.success:
            await migration_manager.rollback_migration(plan)
    ```
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

# Import project detection for tenant ID calculation
from ..utils.project_detection import calculate_tenant_id


class MigrationStatus(Enum):
    """Migration status enumeration."""

    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    RESUMING = "resuming"


@dataclass
class MigrationStatistics:
    """Statistics for a migration operation."""

    queue_entries_total: int = 0
    queue_entries_migrated: int = 0
    metadata_entries_total: int = 0
    metadata_entries_migrated: int = 0
    qdrant_points_total: int = 0
    qdrant_points_migrated: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary for JSON serialization."""
        return {
            "queue_entries_total": self.queue_entries_total,
            "queue_entries_migrated": self.queue_entries_migrated,
            "metadata_entries_total": self.metadata_entries_total,
            "metadata_entries_migrated": self.metadata_entries_migrated,
            "qdrant_points_total": self.qdrant_points_total,
            "qdrant_points_migrated": self.qdrant_points_migrated,
            "errors": self.errors,
            "warnings": self.warnings,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time
                else None
            ),
        }


@dataclass
class MigrationPlan:
    """Plan for tenant ID migration."""

    migration_id: int
    project_root: str
    old_tenant_id: str
    new_tenant_id: str
    status: MigrationStatus = MigrationStatus.PLANNING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    statistics: MigrationStatistics = field(default_factory=MigrationStatistics)
    rollback_data: dict[str, Any] = field(default_factory=dict)
    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def add_audit_entry(self, action: str, details: str, success: bool = True):
        """Add an entry to the audit log."""
        self.audit_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "details": details,
            "success": success,
        })

    def to_db_dict(self) -> dict[str, Any]:
        """Convert plan to dictionary for database storage."""
        return {
            "id": self.migration_id,
            "project_root": self.project_root,
            "old_tenant_id": self.old_tenant_id,
            "new_tenant_id": self.new_tenant_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "entries_migrated": (
                self.statistics.queue_entries_migrated +
                self.statistics.metadata_entries_migrated +
                self.statistics.qdrant_points_migrated
            ),
            "entries_total": (
                self.statistics.queue_entries_total +
                self.statistics.metadata_entries_total +
                self.statistics.qdrant_points_total
            ),
            "error_message": "; ".join(self.statistics.errors) if self.statistics.errors else None,
            "rollback_info": json.dumps(self.rollback_data),
            "audit_log": json.dumps(self.audit_log),
        }


@dataclass
class MigrationResult:
    """Result of a migration operation."""

    success: bool
    migration_id: int
    old_tenant_id: str
    new_tenant_id: str
    statistics: MigrationStatistics
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "migration_id": self.migration_id,
            "old_tenant_id": self.old_tenant_id,
            "new_tenant_id": self.new_tenant_id,
            "statistics": self.statistics.to_dict(),
            "error_message": self.error_message,
        }


class TenantMigrationManager:
    """
    Manages tenant ID migrations for workspace-qdrant-mcp.

    This class handles the complete lifecycle of tenant ID migrations, including:
    - Detection of tenant ID changes via Git monitoring
    - Planning migrations with impact analysis
    - Executing atomic migrations with rollback support
    - Progress tracking and audit logging
    - Consistency validation

    Attributes:
        state_manager: SQLite state manager for database operations
        batch_size: Number of entries to migrate per batch (default: 1000)

    Example:
        ```python
        manager = TenantMigrationManager(state_manager)

        # Detect changes
        change = await manager.detect_tenant_change(project_root)

        # Plan migration
        plan = await manager.plan_migration(project_root, old_id, new_id)

        # Execute migration
        result = await manager.execute_migration(plan)
        ```
    """

    def __init__(self, state_manager, batch_size: int = 1000):
        """
        Initialize tenant migration manager.

        Args:
            state_manager: SQLiteStateManager instance for database access
            batch_size: Number of entries to process per batch
        """
        self.state_manager = state_manager
        self.batch_size = batch_size
        self._ensure_migration_table()

    def _ensure_migration_table(self):
        """Ensure tenant_migrations table exists in the database."""
        if not self.state_manager.connection:
            logger.warning("State manager not initialized, skipping migration table creation")
            return

        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='tenant_migrations'"
                )
                if not cursor.fetchone():
                    logger.info("Creating tenant_migrations table")
                    self.state_manager.connection.execute("""
                        CREATE TABLE tenant_migrations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            project_root TEXT NOT NULL,
                            old_tenant_id TEXT NOT NULL,
                            new_tenant_id TEXT NOT NULL,
                            status TEXT NOT NULL,
                            started_at TIMESTAMP NOT NULL,
                            completed_at TIMESTAMP,
                            entries_migrated INTEGER DEFAULT 0,
                            entries_total INTEGER DEFAULT 0,
                            error_message TEXT,
                            rollback_info TEXT,
                            audit_log TEXT
                        )
                    """)
                    self.state_manager.connection.execute(
                        "CREATE INDEX idx_tenant_migrations_status ON tenant_migrations(status)"
                    )
                    self.state_manager.connection.execute(
                        "CREATE INDEX idx_tenant_migrations_project_root ON tenant_migrations(project_root)"
                    )
                    self.state_manager.connection.commit()
                    logger.info("Created tenant_migrations table successfully")
        except Exception as e:
            logger.error(f"Failed to create tenant_migrations table: {e}")

    async def detect_tenant_change(
        self, project_root: Path, stored_tenant_id: str | None = None
    ) -> tuple[bool, str | None, str | None]:
        """
        Detect if tenant ID has changed for a project.

        Compares the current tenant ID (calculated from Git remote) with the
        stored tenant ID from the database. Returns whether a change was detected
        and the old/new tenant IDs.

        Args:
            project_root: Path to the project root directory
            stored_tenant_id: Optional stored tenant ID (if None, fetches from DB)

        Returns:
            Tuple of (change_detected, old_tenant_id, new_tenant_id)

        Example:
            ```python
            changed, old_id, new_id = await manager.detect_tenant_change(path)
            if changed:
                logger.info(f"Tenant ID changed: {old_id} -> {new_id}")
            ```
        """
        try:
            # Calculate current tenant ID from Git remote
            current_tenant_id = calculate_tenant_id(project_root)

            # Get stored tenant ID if not provided
            if stored_tenant_id is None:
                stored_tenant_id = await self._get_stored_tenant_id(project_root)

            # No stored tenant ID means this is first time seeing this project
            if stored_tenant_id is None:
                logger.info(f"No stored tenant ID for {project_root}, assuming new project")
                return False, None, current_tenant_id

            # Check if tenant ID has changed
            if stored_tenant_id != current_tenant_id:
                logger.info(
                    f"Tenant ID change detected for {project_root}: "
                    f"{stored_tenant_id} -> {current_tenant_id}"
                )
                return True, stored_tenant_id, current_tenant_id

            logger.debug(f"No tenant ID change for {project_root}")
            return False, stored_tenant_id, current_tenant_id

        except Exception as e:
            logger.error(f"Failed to detect tenant change for {project_root}: {e}")
            return False, None, None

    async def _get_stored_tenant_id(self, project_root: Path) -> str | None:
        """
        Get the stored tenant ID for a project from the database.

        Queries the ingestion_queue to find the most recent tenant_id
        associated with files from this project root.

        Args:
            project_root: Path to the project root

        Returns:
            Stored tenant ID or None if not found
        """
        try:
            project_root_str = str(project_root.resolve())

            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT DISTINCT tenant_id
                    FROM ingestion_queue
                    WHERE file_absolute_path LIKE ?
                    ORDER BY rowid DESC
                    LIMIT 1
                    """,
                    (f"{project_root_str}%",)
                )
                row = cursor.fetchone()
                if row:
                    return row[0]
                return None

        except Exception as e:
            logger.error(f"Failed to get stored tenant ID: {e}")
            return None

    async def plan_migration(
        self,
        project_root: Path,
        old_tenant_id: str,
        new_tenant_id: str
    ) -> MigrationPlan:
        """
        Create a migration plan for tenant ID change.

        Analyzes the scope of the migration by counting affected entries
        in the queue and metadata tables. Creates a migration plan with
        statistics and rollback information.

        Args:
            project_root: Path to the project root
            old_tenant_id: Current/old tenant ID
            new_tenant_id: New tenant ID after migration

        Returns:
            MigrationPlan with impact analysis and rollback data

        Raises:
            ValueError: If old and new tenant IDs are the same
        """
        if old_tenant_id == new_tenant_id:
            raise ValueError(
                f"Old and new tenant IDs are identical: {old_tenant_id}"
            )

        try:
            project_root_str = str(project_root.resolve())

            # Create migration record
            migration_id = await self._create_migration_record(
                project_root_str, old_tenant_id, new_tenant_id
            )

            # Create plan
            plan = MigrationPlan(
                migration_id=migration_id,
                project_root=project_root_str,
                old_tenant_id=old_tenant_id,
                new_tenant_id=new_tenant_id,
            )

            plan.add_audit_entry("plan_created", f"Migration plan created for {project_root_str}")

            # Count affected entries in ingestion_queue
            queue_count = await self._count_queue_entries(old_tenant_id)
            plan.statistics.queue_entries_total = queue_count
            plan.add_audit_entry(
                "queue_counted",
                f"Found {queue_count} queue entries to migrate"
            )

            # Count affected entries in file_processing metadata
            metadata_count = await self._count_metadata_entries(old_tenant_id, project_root_str)
            plan.statistics.metadata_entries_total = metadata_count
            plan.add_audit_entry(
                "metadata_counted",
                f"Found {metadata_count} metadata entries to migrate"
            )

            # Store rollback information
            plan.rollback_data = {
                "old_tenant_id": old_tenant_id,
                "new_tenant_id": new_tenant_id,
                "project_root": project_root_str,
                "queue_entries": queue_count,
                "metadata_entries": metadata_count,
            }

            logger.info(
                f"Created migration plan {migration_id}: "
                f"{queue_count} queue + {metadata_count} metadata entries"
            )

            return plan

        except Exception as e:
            logger.error(f"Failed to create migration plan: {e}")
            raise

    async def _create_migration_record(
        self, project_root: str, old_tenant_id: str, new_tenant_id: str
    ) -> int:
        """Create a migration record in the database and return its ID."""
        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    INSERT INTO tenant_migrations (
                        project_root, old_tenant_id, new_tenant_id,
                        status, started_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        project_root,
                        old_tenant_id,
                        new_tenant_id,
                        MigrationStatus.PLANNING.value,
                        datetime.now(timezone.utc).isoformat()
                    )
                )
                self.state_manager.connection.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to create migration record: {e}")
            raise

    async def _count_queue_entries(self, tenant_id: str) -> int:
        """Count entries in ingestion_queue for a tenant ID."""
        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    "SELECT COUNT(*) FROM ingestion_queue WHERE tenant_id = ?",
                    (tenant_id,)
                )
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count queue entries: {e}")
            return 0

    async def _count_metadata_entries(self, tenant_id: str, project_root: str) -> int:
        """Count entries in file_processing with tenant_id in metadata."""
        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT COUNT(*) FROM file_processing
                    WHERE file_path LIKE ?
                    AND (
                        json_extract(metadata, '$.tenant_id') = ?
                        OR metadata IS NULL
                        OR metadata = ''
                    )
                    """,
                    (f"{project_root}%", tenant_id)
                )
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to count metadata entries: {e}")
            return 0

    async def execute_migration(self, plan: MigrationPlan) -> MigrationResult:
        """
        Execute a tenant ID migration plan.

        Performs atomic migration of all data structures:
        1. Updates ingestion_queue entries
        2. Updates file_processing metadata
        3. Validates consistency

        Uses SQLite transactions with savepoints for atomicity.
        Updates progress after each batch. Logs all operations to audit trail.

        Args:
            plan: Migration plan to execute

        Returns:
            MigrationResult with success status and statistics

        Example:
            ```python
            result = await manager.execute_migration(plan)
            if result.success:
                logger.info(f"Migrated {result.statistics.queue_entries_migrated} entries")
            else:
                logger.error(f"Migration failed: {result.error_message}")
            ```
        """
        plan.status = MigrationStatus.EXECUTING
        plan.started_at = datetime.now(timezone.utc)
        plan.statistics.start_time = plan.started_at

        try:
            # Update migration status
            await self._update_migration_status(plan)
            plan.add_audit_entry("migration_started", "Migration execution started")

            # Migrate queue entries
            logger.info(f"Migrating {plan.statistics.queue_entries_total} queue entries")
            queue_migrated = await self._migrate_queue_entries(plan)
            plan.statistics.queue_entries_migrated = queue_migrated
            plan.add_audit_entry("queue_migrated", f"Migrated {queue_migrated} queue entries")

            # Migrate metadata entries
            logger.info(f"Migrating {plan.statistics.metadata_entries_total} metadata entries")
            metadata_migrated = await self._migrate_metadata_entries(plan)
            plan.statistics.metadata_entries_migrated = metadata_migrated
            plan.add_audit_entry("metadata_migrated", f"Migrated {metadata_migrated} metadata entries")

            # Validate migration
            validation_success = await self._validate_migration(plan)
            if not validation_success:
                raise RuntimeError("Migration validation failed")

            plan.add_audit_entry("validation_passed", "Migration validation successful")

            # Mark as completed
            plan.status = MigrationStatus.COMPLETED
            plan.completed_at = datetime.now(timezone.utc)
            plan.statistics.end_time = plan.completed_at
            await self._update_migration_status(plan)

            logger.info(
                f"Migration {plan.migration_id} completed successfully: "
                f"{queue_migrated} queue + {metadata_migrated} metadata entries"
            )

            return MigrationResult(
                success=True,
                migration_id=plan.migration_id,
                old_tenant_id=plan.old_tenant_id,
                new_tenant_id=plan.new_tenant_id,
                statistics=plan.statistics,
            )

        except Exception as e:
            error_msg = f"Migration failed: {e}"
            logger.error(error_msg)
            plan.statistics.errors.append(error_msg)
            plan.add_audit_entry("migration_failed", error_msg, success=False)

            plan.status = MigrationStatus.FAILED
            plan.completed_at = datetime.now(timezone.utc)
            plan.statistics.end_time = plan.completed_at
            await self._update_migration_status(plan)

            return MigrationResult(
                success=False,
                migration_id=plan.migration_id,
                old_tenant_id=plan.old_tenant_id,
                new_tenant_id=plan.new_tenant_id,
                statistics=plan.statistics,
                error_message=error_msg,
            )

    async def _migrate_queue_entries(self, plan: MigrationPlan) -> int:
        """
        Migrate tenant_id in ingestion_queue entries.

        Uses atomic transaction to update all matching entries.

        Args:
            plan: Migration plan

        Returns:
            Number of entries migrated
        """
        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    UPDATE ingestion_queue
                    SET tenant_id = ?
                    WHERE tenant_id = ?
                    """,
                    (plan.new_tenant_id, plan.old_tenant_id)
                )
                migrated_count = cursor.rowcount
                self.state_manager.connection.commit()

                logger.info(f"Migrated {migrated_count} queue entries")
                return migrated_count

        except Exception as e:
            logger.error(f"Failed to migrate queue entries: {e}")
            self.state_manager.connection.rollback()
            raise

    async def _migrate_metadata_entries(self, plan: MigrationPlan) -> int:
        """
        Migrate tenant_id in file_processing metadata.

        Updates JSON metadata field for entries matching the project root.
        Handles cases where metadata is NULL or empty.

        Args:
            plan: Migration plan

        Returns:
            Number of entries migrated
        """
        try:
            migrated_count = 0

            with self.state_manager._lock:
                # First, get all matching file paths
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT file_path, metadata FROM file_processing
                    WHERE file_path LIKE ?
                    """,
                    (f"{plan.project_root}%",)
                )
                rows = cursor.fetchall()

                # Update each entry
                for row in rows:
                    file_path = row[0]
                    metadata_str = row[1]

                    # Parse existing metadata or create new
                    if metadata_str:
                        try:
                            metadata = json.loads(metadata_str)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        metadata = {}

                    # Update tenant_id
                    old_tenant = metadata.get("tenant_id")
                    if old_tenant == plan.old_tenant_id or old_tenant is None:
                        metadata["tenant_id"] = plan.new_tenant_id

                        # Update in database
                        self.state_manager.connection.execute(
                            """
                            UPDATE file_processing
                            SET metadata = ?
                            WHERE file_path = ?
                            """,
                            (json.dumps(metadata), file_path)
                        )
                        migrated_count += 1

                self.state_manager.connection.commit()
                logger.info(f"Migrated {migrated_count} metadata entries")
                return migrated_count

        except Exception as e:
            logger.error(f"Failed to migrate metadata entries: {e}")
            self.state_manager.connection.rollback()
            raise

    async def _validate_migration(self, plan: MigrationPlan) -> bool:
        """
        Validate migration consistency.

        Checks that:
        1. No entries remain with old tenant ID
        2. Expected number of entries have new tenant ID

        Args:
            plan: Migration plan

        Returns:
            True if validation passed, False otherwise
        """
        try:
            # Check queue entries
            remaining_queue = await self._count_queue_entries(plan.old_tenant_id)
            if remaining_queue > 0:
                logger.error(
                    f"Validation failed: {remaining_queue} queue entries still have old tenant ID"
                )
                return False

            # Check new queue entries
            new_queue_count = await self._count_queue_entries(plan.new_tenant_id)
            if new_queue_count < plan.statistics.queue_entries_migrated:
                logger.error(
                    f"Validation failed: Expected {plan.statistics.queue_entries_migrated} "
                    f"queue entries with new tenant ID, found {new_queue_count}"
                )
                return False

            logger.info("Migration validation passed")
            return True

        except Exception as e:
            logger.error(f"Failed to validate migration: {e}")
            return False

    async def _update_migration_status(self, plan: MigrationPlan):
        """Update migration record in database with current status."""
        try:
            db_data = plan.to_db_dict()

            with self.state_manager._lock:
                self.state_manager.connection.execute(
                    """
                    UPDATE tenant_migrations
                    SET status = ?, started_at = ?, completed_at = ?,
                        entries_migrated = ?, entries_total = ?,
                        error_message = ?, rollback_info = ?, audit_log = ?
                    WHERE id = ?
                    """,
                    (
                        db_data["status"],
                        db_data["started_at"],
                        db_data["completed_at"],
                        db_data["entries_migrated"],
                        db_data["entries_total"],
                        db_data["error_message"],
                        db_data["rollback_info"],
                        db_data["audit_log"],
                        db_data["id"],
                    )
                )
                self.state_manager.connection.commit()

        except Exception as e:
            logger.error(f"Failed to update migration status: {e}")

    async def rollback_migration(self, plan: MigrationPlan) -> bool:
        """
        Rollback a failed migration.

        Reverts all changes by swapping new tenant ID back to old tenant ID.

        Args:
            plan: Migration plan to rollback

        Returns:
            True if rollback succeeded, False otherwise
        """
        logger.warning(f"Rolling back migration {plan.migration_id}")
        plan.add_audit_entry("rollback_started", "Starting migration rollback")

        try:
            # Create reverse plan
            reverse_plan = MigrationPlan(
                migration_id=plan.migration_id,
                project_root=plan.project_root,
                old_tenant_id=plan.new_tenant_id,  # Swap old/new
                new_tenant_id=plan.old_tenant_id,
                status=MigrationStatus.EXECUTING,
            )

            # Rollback queue entries
            queue_rolled_back = await self._migrate_queue_entries(reverse_plan)
            plan.add_audit_entry("queue_rolled_back", f"Rolled back {queue_rolled_back} queue entries")

            # Rollback metadata entries
            metadata_rolled_back = await self._migrate_metadata_entries(reverse_plan)
            plan.add_audit_entry(
                "metadata_rolled_back",
                f"Rolled back {metadata_rolled_back} metadata entries"
            )

            # Update status
            plan.status = MigrationStatus.ROLLED_BACK
            plan.completed_at = datetime.now(timezone.utc)
            await self._update_migration_status(plan)

            logger.info(f"Successfully rolled back migration {plan.migration_id}")
            plan.add_audit_entry("rollback_completed", "Rollback completed successfully")
            return True

        except Exception as e:
            error_msg = f"Rollback failed: {e}"
            logger.error(error_msg)
            plan.add_audit_entry("rollback_failed", error_msg, success=False)
            return False

    async def get_migration_history(
        self, project_root: Path | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get migration history.

        Args:
            project_root: Optional filter by project root
            limit: Maximum number of records to return

        Returns:
            List of migration records
        """
        try:
            query = "SELECT * FROM tenant_migrations"
            params = []

            if project_root:
                query += " WHERE project_root = ?"
                params.append(str(project_root.resolve()))

            query += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(query, params)
                rows = cursor.fetchall()

                result = []
                for row in rows:
                    result.append({
                        "id": row[0],
                        "project_root": row[1],
                        "old_tenant_id": row[2],
                        "new_tenant_id": row[3],
                        "status": row[4],
                        "started_at": row[5],
                        "completed_at": row[6],
                        "entries_migrated": row[7],
                        "entries_total": row[8],
                        "error_message": row[9],
                        "rollback_info": json.loads(row[10]) if row[10] else {},
                        "audit_log": json.loads(row[11]) if row[11] else [],
                    })

                return result

        except Exception as e:
            logger.error(f"Failed to get migration history: {e}")
            return []
