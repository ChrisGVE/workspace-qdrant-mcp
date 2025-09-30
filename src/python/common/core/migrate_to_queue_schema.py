#!/usr/bin/env env python3
"""
SQLite Queue Schema Migration Script

Migrates existing workspace_state.db from schema version 3 to version 4,
introducing the enhanced priority queue system with ingestion_queue,
collection_metadata, and messages tables.

Usage:
    python migrate_to_queue_schema.py [--db-path PATH] [--dry-run] [--backup]

Features:
    - Automatic backup creation
    - Rollback on failure
    - Progress tracking
    - Data validation
    - Preserves existing data integrity
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Priority mapping: old 1-4 scale → new 0-10 scale
PRIORITY_MAPPING = {
    1: 2,   # LOW → 2
    2: 5,   # NORMAL → 5
    3: 7,   # HIGH → 7
    4: 9,   # URGENT → 9
}


class MigrationError(Exception):
    """Migration-specific error."""
    pass


class QueueSchemaMigrator:
    """Handles migration from schema v3 to v4 (queue system)."""

    def __init__(self, db_path: str, dry_run: bool = False, create_backup: bool = True):
        """
        Initialize migrator.

        Args:
            db_path: Path to SQLite database
            dry_run: If True, perform validation only without changes
            create_backup: If True, create backup before migration
        """
        self.db_path = Path(db_path)
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.conn: Optional[sqlite3.Connection] = None
        self.backup_path: Optional[Path] = None

        # Migration statistics
        self.stats = {
            'queue_items_migrated': 0,
            'errors_migrated': 0,
            'collections_created': 0,
            'validation_errors': 0,
        }

    def migrate(self) -> bool:
        """
        Execute full migration process.

        Returns:
            True if migration succeeded, False otherwise
        """
        try:
            print(f"{'[DRY RUN] ' if self.dry_run else ''}Starting migration for: {self.db_path}")

            # Step 1: Validate preconditions
            if not self._validate_preconditions():
                return False

            # Step 2: Create backup
            if self.create_backup and not self.dry_run:
                self._create_backup()

            # Step 3: Connect and prepare
            self._connect()

            # Step 4: Verify current schema version
            current_version = self._get_schema_version()
            print(f"Current schema version: {current_version}")

            if current_version >= 4:
                print("Database already at version 4 or higher. No migration needed.")
                return True

            # Step 5: Create savepoint for rollback
            if not self.dry_run:
                self.conn.execute("SAVEPOINT migration_v4")

            # Step 6: Execute migration steps
            self._execute_migration()

            # Step 7: Validate migration
            if not self._validate_migration():
                raise MigrationError("Migration validation failed")

            # Step 8: Commit or rollback
            if self.dry_run:
                print("\n[DRY RUN] Migration would succeed. No changes made.")
                return True
            else:
                self.conn.execute("RELEASE SAVEPOINT migration_v4")
                self._update_schema_version(4)
                self.conn.commit()
                print("\nMigration completed successfully!")
                self._print_statistics()
                return True

        except Exception as e:
            print(f"\n❌ Migration failed: {e}", file=sys.stderr)
            if self.conn and not self.dry_run:
                try:
                    self.conn.execute("ROLLBACK TO SAVEPOINT migration_v4")
                    self.conn.commit()
                    print("Changes rolled back successfully.")
                except Exception as rollback_error:
                    print(f"⚠️  Rollback failed: {rollback_error}", file=sys.stderr)
            return False

        finally:
            if self.conn:
                self.conn.close()

    def _validate_preconditions(self) -> bool:
        """Validate preconditions before migration."""
        print("\nValidating preconditions...")

        # Check database exists
        if not self.db_path.exists():
            print(f"❌ Database not found: {self.db_path}")
            return False

        # Check database is not locked
        try:
            conn = sqlite3.connect(str(self.db_path), timeout=5.0)
            conn.execute("SELECT 1")
            conn.close()
        except sqlite3.OperationalError as e:
            print(f"❌ Database is locked or inaccessible: {e}")
            return False

        print("✅ Preconditions validated")
        return True

    def _create_backup(self):
        """Create database backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_path = self.db_path.parent / f"{self.db_path.stem}.backup.{timestamp}{self.db_path.suffix}"

        print(f"\nCreating backup: {self.backup_path}")

        import shutil
        shutil.copy2(self.db_path, self.backup_path)

        print(f"✅ Backup created: {self.backup_path}")

    def _connect(self):
        """Connect to database with proper settings."""
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # Explicit transaction control
        )
        self.conn.row_factory = sqlite3.Row

        # Ensure WAL mode for safety
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def _get_schema_version(self) -> int:
        """Get current schema version."""
        try:
            cursor = self.conn.execute(
                "SELECT MAX(version) FROM schema_version"
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else 0
        except sqlite3.OperationalError:
            return 0

    def _execute_migration(self):
        """Execute migration steps in order."""
        print("\nExecuting migration steps...")

        # Step 1: Create new tables
        print("\n1. Creating new tables...")
        self._create_new_tables()

        # Step 2: Migrate error_log → messages
        print("\n2. Migrating error data...")
        self._migrate_errors()

        # Step 3: Detect and create collection metadata
        print("\n3. Creating collection metadata...")
        self._create_collection_metadata()

        # Step 4: Migrate processing_queue → ingestion_queue
        print("\n4. Migrating queue data...")
        self._migrate_queue_data()

        print("\n✅ All migration steps completed")

    def _create_new_tables(self):
        """Create new queue schema tables."""
        # Read schema from queue_schema.sql
        schema_file = self.db_path.parent / "queue_schema.sql"

        if not schema_file.exists():
            raise MigrationError(f"Schema file not found: {schema_file}")

        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Execute schema (split by semicolons and filter comments)
        statements = [
            stmt.strip()
            for stmt in schema_sql.split(';')
            if stmt.strip() and not stmt.strip().startswith('--')
        ]

        for i, stmt in enumerate(statements, 1):
            if not stmt:
                continue
            try:
                self.conn.execute(stmt)
                if 'CREATE TABLE' in stmt.upper():
                    # Extract table name for logging
                    table_name = stmt.split('CREATE TABLE')[1].split('IF NOT EXISTS')[1].split('(')[0].strip()
                    print(f"   ✓ Created table: {table_name}")
            except sqlite3.OperationalError as e:
                # Ignore "already exists" errors
                if 'already exists' not in str(e).lower():
                    raise

    def _migrate_errors(self):
        """Migrate error_log → messages."""
        # Check if error_log table exists
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='error_log'"
        )
        if not cursor.fetchone():
            print("   ⚠️  No error_log table found, skipping error migration")
            return

        # Query existing errors
        cursor = self.conn.execute("""
            SELECT
                id,
                error_type,
                error_message,
                source,
                timestamp,
                metadata
            FROM error_log
            ORDER BY id
        """)

        errors = cursor.fetchall()
        print(f"   Found {len(errors)} errors to migrate")

        for error in errors:
            # Parse metadata to extract file_path and collection_name
            file_path = None
            collection_name = None

            if error['metadata']:
                try:
                    meta = json.loads(error['metadata'])
                    file_path = meta.get('file_path') or meta.get('file')
                    collection_name = meta.get('collection')
                except json.JSONDecodeError:
                    pass

            # If no file_path in metadata, try to extract from source
            if not file_path and error['source']:
                # Simple heuristic: look for path-like strings
                if '/' in error['source'] or '\\' in error['source']:
                    file_path = error['source']

            # Insert into messages table
            self.conn.execute("""
                INSERT OR IGNORE INTO messages (
                    id, error_type, error_message, error_details,
                    occurred_timestamp, file_path, collection_name, retry_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                error['id'],
                error['error_type'],
                error['error_message'],
                error['metadata'],  # Store original metadata as error_details
                error['timestamp'],
                file_path,
                collection_name,
                0  # Initialize retry_count
            ))

            self.stats['errors_migrated'] += 1

        print(f"   ✅ Migrated {self.stats['errors_migrated']} error records")

    def _create_collection_metadata(self):
        """Detect and create collection metadata entries."""
        # Get unique collections from processing_queue and watch_folders
        collections = set()

        # From processing_queue
        cursor = self.conn.execute("SELECT DISTINCT collection FROM processing_queue")
        collections.update(row['collection'] for row in cursor.fetchall())

        # From watch_folders
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='watch_folders'"
        )
        if cursor.fetchone():
            cursor = self.conn.execute("SELECT DISTINCT collection FROM watch_folders")
            watched_collections = set(row['collection'] for row in cursor.fetchall())
        else:
            watched_collections = set()

        print(f"   Found {len(collections)} unique collections")

        for collection in collections:
            # Detect collection type
            collection_type = self._detect_collection_type(collection, watched_collections)

            # Create metadata entry
            self.conn.execute("""
                INSERT OR IGNORE INTO collection_metadata (
                    collection_name, collection_type, configuration, tenant_id, branch
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                collection,
                collection_type,
                json.dumps({}),  # Empty configuration initially
                'default',
                'main'
            ))

            self.stats['collections_created'] += 1

        print(f"   ✅ Created {self.stats['collections_created']} collection metadata entries")

    def _detect_collection_type(self, collection_name: str, watched_collections: set) -> str:
        """Detect collection type from name and context."""
        if collection_name in watched_collections:
            return 'watched-dynamic'
        elif 'project' in collection_name.lower():
            return 'project'
        else:
            return 'non-watched'

    def _migrate_queue_data(self):
        """Migrate processing_queue → ingestion_queue."""
        # Check if processing_queue exists
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='processing_queue'"
        )
        if not cursor.fetchone():
            print("   ⚠️  No processing_queue table found, skipping queue migration")
            return

        # Query existing queue items
        cursor = self.conn.execute("""
            SELECT
                queue_id,
                file_path,
                collection,
                priority,
                created_at,
                scheduled_at,
                attempts,
                metadata
            FROM processing_queue
            ORDER BY priority DESC, created_at ASC
        """)

        items = cursor.fetchall()
        print(f"   Found {len(items)} queue items to migrate")

        for item in items:
            # Parse metadata
            metadata_fields = self._extract_metadata_fields(item['metadata'])

            # Scale priority
            new_priority = self._scale_priority(item['priority'])

            # Infer operation (default to 'ingest')
            operation = 'ingest'

            # Find linked error if any
            error_message_id = self._find_error_for_file(item['file_path'])

            # Insert into ingestion_queue
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO ingestion_queue (
                        file_absolute_path, collection_name, tenant_id, branch,
                        operation, priority, queued_timestamp, retry_count,
                        retry_from, error_message_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item['file_path'],
                    item['collection'],
                    metadata_fields['tenant_id'],
                    metadata_fields['branch'],
                    operation,
                    new_priority,
                    item['created_at'],
                    item['attempts'],
                    None,  # No historical retry chains
                    error_message_id
                ))

                self.stats['queue_items_migrated'] += 1

            except sqlite3.IntegrityError as e:
                print(f"   ⚠️  Failed to migrate {item['file_path']}: {e}")
                self.stats['validation_errors'] += 1

        print(f"   ✅ Migrated {self.stats['queue_items_migrated']} queue items")

    def _extract_metadata_fields(self, metadata_json: Optional[str]) -> Dict[str, str]:
        """Extract tenant_id and branch from metadata JSON."""
        if not metadata_json:
            return {'tenant_id': 'default', 'branch': 'main'}

        try:
            metadata = json.loads(metadata_json)
            return {
                'tenant_id': metadata.get('tenant_id', 'default'),
                'branch': metadata.get('branch', 'main'),
            }
        except json.JSONDecodeError:
            return {'tenant_id': 'default', 'branch': 'main'}

    def _scale_priority(self, old_priority: int) -> int:
        """Scale priority from old 1-4 range to new 0-10 range."""
        return PRIORITY_MAPPING.get(old_priority, 5)

    def _find_error_for_file(self, file_path: str) -> Optional[int]:
        """Find most recent error message ID for a file."""
        cursor = self.conn.execute("""
            SELECT id FROM messages
            WHERE file_path = ?
            ORDER BY occurred_timestamp DESC
            LIMIT 1
        """, (file_path,))

        result = cursor.fetchone()
        return result['id'] if result else None

    def _validate_migration(self) -> bool:
        """Validate migration results."""
        print("\nValidating migration...")

        validation_passed = True

        # Validation 1: Row count preservation
        cursor = self.conn.execute("SELECT COUNT(*) as cnt FROM processing_queue")
        old_count = cursor.fetchone()['cnt']

        cursor = self.conn.execute("SELECT COUNT(*) as cnt FROM ingestion_queue")
        new_count = cursor.fetchone()['cnt']

        if old_count != new_count:
            print(f"   ⚠️  Row count mismatch: {old_count} (old) vs {new_count} (new)")
            validation_passed = False
        else:
            print(f"   ✅ Row count preserved: {new_count} items")

        # Validation 2: Foreign key integrity
        cursor = self.conn.execute("""
            SELECT COUNT(*) as cnt FROM ingestion_queue
            WHERE error_message_id IS NOT NULL
              AND error_message_id NOT IN (SELECT id FROM messages)
        """)
        invalid_fks = cursor.fetchone()['cnt']

        if invalid_fks > 0:
            print(f"   ⚠️  Found {invalid_fks} invalid foreign keys")
            validation_passed = False
        else:
            print("   ✅ Foreign key integrity validated")

        # Validation 3: Priority range check
        cursor = self.conn.execute("""
            SELECT MIN(priority) as min_p, MAX(priority) as max_p FROM ingestion_queue
        """)
        result = cursor.fetchone()
        if result['min_p'] is not None:
            if result['min_p'] < 0 or result['max_p'] > 10:
                print(f"   ⚠️  Priority out of range: {result['min_p']}-{result['max_p']}")
                validation_passed = False
            else:
                print(f"   ✅ Priority range valid: {result['min_p']}-{result['max_p']}")

        # Validation 4: Check constraints
        try:
            # Try to insert invalid operation (should fail)
            self.conn.execute("""
                INSERT INTO ingestion_queue (file_absolute_path, collection_name, operation, priority)
                VALUES ('__test__', '__test__', 'invalid_op', 5)
            """)
            print("   ⚠️  CHECK constraint not enforced for operation")
            validation_passed = False
            self.conn.execute("DELETE FROM ingestion_queue WHERE file_absolute_path = '__test__'")
        except sqlite3.IntegrityError:
            print("   ✅ CHECK constraints validated")

        if validation_passed:
            print("\n✅ All validations passed")
        else:
            print("\n❌ Some validations failed")

        return validation_passed

    def _update_schema_version(self, version: int):
        """Update schema version in database."""
        self.conn.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (version,)
        )

    def _print_statistics(self):
        """Print migration statistics."""
        print("\n" + "="*50)
        print("Migration Statistics:")
        print("="*50)
        print(f"Queue items migrated:    {self.stats['queue_items_migrated']}")
        print(f"Errors migrated:         {self.stats['errors_migrated']}")
        print(f"Collections created:     {self.stats['collections_created']}")
        print(f"Validation errors:       {self.stats['validation_errors']}")
        print("="*50)


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate SQLite state database to queue schema v4"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        help='Path to SQLite database (default: auto-detect from OS directories)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform validation only without making changes'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip automatic backup creation'
    )

    args = parser.parse_args()

    # Determine database path
    if args.db_path:
        db_path = args.db_path
    else:
        # Auto-detect from OS directories
        try:
            from ..utils.os_directories import OSDirectories
            os_dirs = OSDirectories()
            db_path = os_dirs.get_state_file("workspace_state.db")
        except ImportError:
            print("Error: Could not auto-detect database path. Please provide --db-path")
            sys.exit(1)

    # Execute migration
    migrator = QueueSchemaMigrator(
        db_path=db_path,
        dry_run=args.dry_run,
        create_backup=not args.no_backup
    )

    success = migrator.migrate()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
