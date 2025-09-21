"""
Component Coordination Schema Migration System.

This module provides schema migration capabilities for the component coordination
system, allowing for safe upgrades from the base SQLite state manager to the
extended four-component coordination schema.

Key Features:
    - Automatic detection of existing schema versions
    - Safe migration from base schema to component coordination schema
    - Rollback capabilities for failed migrations
    - Data preservation during schema updates
    - Migration validation and verification

Example:
    ```python
    from workspace_qdrant_mcp.core.component_migration import ComponentMigrator

    # Initialize migrator
    migrator = ComponentMigrator(db_path="./workspace_state.db")

    # Check if migration is needed
    if await migrator.needs_migration():
        # Perform migration
        success = await migrator.migrate()
        if success:
            print("Migration completed successfully")
        else:
            print("Migration failed, rolling back")
    ```
"""

import asyncio
import json
import sqlite3
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from python.common.core.sqlite_state_manager import SQLiteStateManager


class MigrationError(Exception):
    """Raised when a migration operation fails."""
    pass


class ComponentMigrator:
    """
    Schema migration system for component coordination.

    Handles safe migration from base SQLite state manager schema to
    extended component coordination schema with rollback capabilities.
    """

    BASE_SCHEMA_VERSION = 3  # SQLiteStateManager schema version
    TARGET_SCHEMA_VERSION = 4  # ComponentCoordinator schema version

    def __init__(self, db_path: str):
        """
        Initialize component migrator.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.backup_path = self.db_path.with_suffix('.backup')

    async def needs_migration(self) -> bool:
        """
        Check if database needs migration to component coordination schema.

        Returns:
            True if migration is needed, False otherwise
        """
        if not self.db_path.exists():
            return False

        try:
            connection = sqlite3.connect(str(self.db_path))
            connection.execute("PRAGMA journal_mode=WAL")

            # Check schema version
            cursor = connection.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()

            if not row:
                connection.close()
                return True  # No schema version means needs migration

            current_version = row[0]
            connection.close()

            # Check if component coordination tables exist
            if current_version >= self.TARGET_SCHEMA_VERSION:
                return await self._verify_component_tables_exist()

            return current_version < self.TARGET_SCHEMA_VERSION

        except Exception as e:
            logger.error(f"Error checking migration status: {e}")
            return False

    async def _verify_component_tables_exist(self) -> bool:
        """Verify that all component coordination tables exist."""
        try:
            connection = sqlite3.connect(str(self.db_path))

            expected_tables = [
                'component_registry',
                'component_communication',
                'component_health_metrics',
                'component_processing_queue',
                'component_recovery_log'
            ]

            cursor = connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE 'component_%'
            """)

            existing_tables = [row[0] for row in cursor.fetchall()]
            connection.close()

            missing_tables = set(expected_tables) - set(existing_tables)
            return len(missing_tables) == 0

        except Exception as e:
            logger.error(f"Error verifying component tables: {e}")
            return False

    async def migrate(self) -> bool:
        """
        Perform migration to component coordination schema.

        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info("Starting component coordination schema migration")

            # Create backup
            await self._create_backup()

            # Perform migration
            await self._perform_migration()

            # Verify migration
            if await self._verify_migration():
                logger.info("Component coordination migration completed successfully")
                await self._cleanup_backup()
                return True
            else:
                logger.error("Migration verification failed, rolling back")
                await self._rollback()
                return False

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            await self._rollback()
            return False

    async def _create_backup(self):
        """Create backup of database before migration."""
        if self.backup_path.exists():
            self.backup_path.unlink()

        shutil.copy2(self.db_path, self.backup_path)
        logger.info(f"Created database backup: {self.backup_path}")

    async def _perform_migration(self):
        """Perform the actual schema migration."""
        connection = sqlite3.connect(str(self.db_path))
        connection.execute("PRAGMA journal_mode=WAL")

        try:
            with connection:
                # Add component coordination tables
                self._add_component_registry_table(connection)
                self._add_component_communication_table(connection)
                self._add_component_health_metrics_table(connection)
                self._add_component_processing_queue_table(connection)
                self._add_component_recovery_log_table(connection)

                # Create indexes
                self._create_component_indexes(connection)

                # Update schema version
                connection.execute(
                    "UPDATE schema_version SET version = ?, updated_at = ?",
                    (self.TARGET_SCHEMA_VERSION, datetime.now(timezone.utc).isoformat())
                )

                logger.info("Schema migration completed")

        finally:
            connection.close()

    def _add_component_registry_table(self, connection: sqlite3.Connection):
        """Add component registry table."""
        connection.execute("""
            CREATE TABLE IF NOT EXISTS component_registry (
                component_id TEXT PRIMARY KEY,
                component_type TEXT NOT NULL,
                instance_id TEXT NOT NULL,
                status TEXT NOT NULL,
                health TEXT NOT NULL,
                version TEXT,
                config TEXT,  -- JSON
                endpoints TEXT,  -- JSON
                capabilities TEXT,  -- JSON array
                dependencies TEXT,  -- JSON array
                resources TEXT,  -- JSON
                last_heartbeat TEXT,  -- ISO datetime
                started_at TEXT,  -- ISO datetime
                stopped_at TEXT,  -- ISO datetime
                restart_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                recovery_attempts INTEGER DEFAULT 0,
                metadata TEXT,  -- JSON
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

    def _add_component_communication_table(self, connection: sqlite3.Connection):
        """Add component communication table."""
        connection.execute("""
            CREATE TABLE IF NOT EXISTS component_communication (
                communication_id TEXT PRIMARY KEY,
                source_component TEXT NOT NULL,
                target_component TEXT NOT NULL,
                channel TEXT NOT NULL,
                message_type TEXT NOT NULL,
                status TEXT NOT NULL,
                request_data TEXT,  -- JSON
                response_data TEXT,  -- JSON
                latency_ms REAL,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                timeout_at TEXT,  -- ISO datetime
                sent_at TEXT,  -- ISO datetime
                received_at TEXT,  -- ISO datetime
                metadata TEXT,  -- JSON
                created_at TEXT NOT NULL,
                FOREIGN KEY (source_component) REFERENCES component_registry(component_id),
                FOREIGN KEY (target_component) REFERENCES component_registry(component_id)
            )
        """)

    def _add_component_health_metrics_table(self, connection: sqlite3.Connection):
        """Add component health metrics table."""
        connection.execute("""
            CREATE TABLE IF NOT EXISTS component_health_metrics (
                metric_id TEXT PRIMARY KEY,
                component_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT NOT NULL,
                threshold_warning REAL,
                threshold_critical REAL,
                is_alert BOOLEAN DEFAULT FALSE,
                alert_level TEXT,
                recorded_at TEXT NOT NULL,  -- ISO datetime
                metadata TEXT,  -- JSON
                FOREIGN KEY (component_id) REFERENCES component_registry(component_id)
            )
        """)

    def _add_component_processing_queue_table(self, connection: sqlite3.Connection):
        """Add component processing queue table."""
        connection.execute("""
            CREATE TABLE IF NOT EXISTS component_processing_queue (
                queue_item_id TEXT PRIMARY KEY,
                component_id TEXT NOT NULL,
                queue_type TEXT NOT NULL,
                priority INTEGER NOT NULL,
                payload TEXT NOT NULL,  -- JSON
                status TEXT NOT NULL,
                assigned_worker TEXT,
                processing_started TEXT,  -- ISO datetime
                processing_completed TEXT,  -- ISO datetime
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                error_message TEXT,
                dependencies TEXT,  -- JSON array
                timeout_seconds INTEGER,
                scheduled_at TEXT,  -- ISO datetime
                metadata TEXT,  -- JSON
                created_at TEXT NOT NULL,
                FOREIGN KEY (component_id) REFERENCES component_registry(component_id)
            )
        """)

    def _add_component_recovery_log_table(self, connection: sqlite3.Connection):
        """Add component recovery log table."""
        connection.execute("""
            CREATE TABLE IF NOT EXISTS component_recovery_log (
                recovery_id TEXT PRIMARY KEY,
                component_id TEXT NOT NULL,
                failure_type TEXT NOT NULL,
                failure_details TEXT,  -- JSON
                recovery_action TEXT NOT NULL,
                recovery_result TEXT NOT NULL,
                recovery_duration_ms INTEGER,
                automatic_recovery BOOLEAN DEFAULT TRUE,
                recovery_metadata TEXT,  -- JSON
                failed_at TEXT NOT NULL,  -- ISO datetime
                recovered_at TEXT,  -- ISO datetime
                created_at TEXT NOT NULL,
                FOREIGN KEY (component_id) REFERENCES component_registry(component_id)
            )
        """)

    def _create_component_indexes(self, connection: sqlite3.Connection):
        """Create performance indexes for component tables."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_component_type ON component_registry(component_type)",
            "CREATE INDEX IF NOT EXISTS idx_component_status ON component_registry(status)",
            "CREATE INDEX IF NOT EXISTS idx_component_health ON component_registry(health)",
            "CREATE INDEX IF NOT EXISTS idx_communication_source ON component_communication(source_component)",
            "CREATE INDEX IF NOT EXISTS idx_communication_target ON component_communication(target_component)",
            "CREATE INDEX IF NOT EXISTS idx_communication_status ON component_communication(status)",
            "CREATE INDEX IF NOT EXISTS idx_health_component ON component_health_metrics(component_id)",
            "CREATE INDEX IF NOT EXISTS idx_health_recorded ON component_health_metrics(recorded_at)",
            "CREATE INDEX IF NOT EXISTS idx_queue_component ON component_processing_queue(component_id)",
            "CREATE INDEX IF NOT EXISTS idx_queue_type ON component_processing_queue(queue_type)",
            "CREATE INDEX IF NOT EXISTS idx_queue_status ON component_processing_queue(status)",
            "CREATE INDEX IF NOT EXISTS idx_queue_priority ON component_processing_queue(priority)",
        ]

        for index_sql in indexes:
            connection.execute(index_sql)

    async def _verify_migration(self) -> bool:
        """Verify that migration was successful."""
        try:
            # Check schema version
            connection = sqlite3.connect(str(self.db_path))
            cursor = connection.execute("SELECT version FROM schema_version LIMIT 1")
            row = cursor.fetchone()

            if not row or row[0] != self.TARGET_SCHEMA_VERSION:
                connection.close()
                return False

            # Check that all component tables exist
            expected_tables = [
                'component_registry',
                'component_communication',
                'component_health_metrics',
                'component_processing_queue',
                'component_recovery_log'
            ]

            cursor = connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE 'component_%'
            """)

            existing_tables = [row[0] for row in cursor.fetchall()]
            connection.close()

            missing_tables = set(expected_tables) - set(existing_tables)
            if missing_tables:
                logger.error(f"Missing component tables after migration: {missing_tables}")
                return False

            # Check that all indexes exist
            connection = sqlite3.connect(str(self.db_path))
            cursor = connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name LIKE 'idx_component_%'
                   OR name LIKE 'idx_communication_%'
                   OR name LIKE 'idx_health_%'
                   OR name LIKE 'idx_queue_%'
            """)

            existing_indexes = [row[0] for row in cursor.fetchall()]
            connection.close()

            expected_indexes = [
                'idx_component_type',
                'idx_component_status',
                'idx_component_health',
                'idx_communication_source',
                'idx_communication_target',
                'idx_communication_status',
                'idx_health_component',
                'idx_health_recorded',
                'idx_queue_component',
                'idx_queue_type',
                'idx_queue_status',
                'idx_queue_priority'
            ]

            missing_indexes = set(expected_indexes) - set(existing_indexes)
            if missing_indexes:
                logger.warning(f"Missing indexes after migration: {missing_indexes}")
                # Indexes are not critical for functionality, so don't fail migration

            return True

        except Exception as e:
            logger.error(f"Migration verification failed: {e}")
            return False

    async def _rollback(self):
        """Rollback migration by restoring from backup."""
        try:
            if self.backup_path.exists():
                if self.db_path.exists():
                    self.db_path.unlink()
                shutil.move(self.backup_path, self.db_path)
                logger.info("Migration rolled back successfully")
            else:
                logger.error("No backup found for rollback")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    async def _cleanup_backup(self):
        """Remove backup file after successful migration."""
        try:
            if self.backup_path.exists():
                self.backup_path.unlink()
                logger.debug("Migration backup cleaned up")
        except Exception as e:
            logger.warning(f"Failed to cleanup backup: {e}")

    async def get_migration_status(self) -> Dict[str, Any]:
        """
        Get detailed migration status information.

        Returns:
            Dictionary with migration status details
        """
        try:
            status = {
                "database_exists": self.db_path.exists(),
                "backup_exists": self.backup_path.exists(),
                "needs_migration": False,
                "current_schema_version": None,
                "target_schema_version": self.TARGET_SCHEMA_VERSION,
                "component_tables_exist": False,
                "migration_required": False
            }

            if not self.db_path.exists():
                return status

            # Check current schema version
            connection = sqlite3.connect(str(self.db_path))
            try:
                cursor = connection.execute("SELECT version FROM schema_version LIMIT 1")
                row = cursor.fetchone()
                if row:
                    status["current_schema_version"] = row[0]
            except:
                status["current_schema_version"] = 0  # No schema version table
            finally:
                connection.close()

            # Check if component tables exist
            status["component_tables_exist"] = await self._verify_component_tables_exist()

            # Determine if migration is needed
            status["needs_migration"] = await self.needs_migration()
            status["migration_required"] = status["needs_migration"]

            return status

        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return {"error": str(e)}


async def migrate_to_component_coordination(db_path: str) -> bool:
    """
    Convenience function to migrate database to component coordination schema.

    Args:
        db_path: Path to SQLite database file

    Returns:
        True if migration successful or not needed, False if migration failed
    """
    migrator = ComponentMigrator(db_path)

    if not await migrator.needs_migration():
        logger.info("Database already migrated to component coordination schema")
        return True

    return await migrator.migrate()


async def check_migration_status(db_path: str) -> Dict[str, Any]:
    """
    Check migration status for a database file.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Dictionary with migration status information
    """
    migrator = ComponentMigrator(db_path)
    return await migrator.get_migration_status()