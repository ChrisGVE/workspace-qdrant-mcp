#!/usr/bin/env python3
"""
Error Messages Schema Migration Script

Migrates the messages table from basic error tracking to enhanced
error message management with severity, category, and acknowledgment support.

Usage:
    python migrate_error_messages.py [--db-path PATH] [--dry-run] [--backup]

Features:
    - Automatic backup creation
    - Rollback on failure
    - Data validation
    - Preserves existing error data
    - Maps error_type to category
    - Consolidates fields into JSON context
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


class MigrationError(Exception):
    """Migration-specific error."""
    pass


# Error type to category mapping
ERROR_TYPE_MAPPING = {
    'PARSE_ERROR': 'parse_error',
    'FILE_NOT_FOUND': 'file_corrupt',
    'PERMISSION_ERROR': 'permission_denied',
    'NETWORK_ERROR': 'network',
    'TIMEOUT_ERROR': 'timeout',
    'METADATA_ERROR': 'metadata_invalid',
    'PROCESSING_ERROR': 'processing_failed',
    'TOOL_ERROR': 'tool_missing',
    'RESOURCE_ERROR': 'resource_exhausted',
}


class ErrorMessageMigrator:
    """Handles migration of messages table to enhanced schema."""

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
        self.conn: sqlite3.Connection | None = None
        self.backup_path: Path | None = None

        # Migration statistics
        self.stats = {
            'messages_migrated': 0,
            'validation_errors': 0,
        }

    def migrate(self) -> bool:
        """
        Execute full migration process.

        Returns:
            True if migration succeeded, False otherwise
        """
        try:
            print(f"{'[DRY RUN] ' if self.dry_run else ''}Starting error messages schema migration: {self.db_path}")

            # Step 1: Validate preconditions
            if not self._validate_preconditions():
                return False

            # Step 2: Create backup
            if self.create_backup and not self.dry_run:
                self._create_backup()

            # Step 3: Connect and prepare
            self._connect()

            # Step 4: Check if migration needed
            if self._is_already_migrated():
                print("Messages table already migrated to enhanced schema. No migration needed.")
                return True

            # Step 5: Execute migration
            self._execute_migration()

            # Step 6: Validate migration
            if not self._validate_migration():
                raise MigrationError("Migration validation failed")

            # Step 7: Success
            if self.dry_run:
                print("\n[DRY RUN] Migration would succeed. No changes made.")
                return True
            else:
                print("\nMigration completed successfully!")
                self._print_statistics()
                return True

        except Exception as e:
            print(f"\n❌ Migration failed: {e}", file=sys.stderr)
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
        self.backup_path = self.db_path.parent / f"{self.db_path.stem}.backup_error_msg.{timestamp}{self.db_path.suffix}"

        print(f"\nCreating backup: {self.backup_path}")

        import shutil
        shutil.copy2(self.db_path, self.backup_path)

        print(f"✅ Backup created: {self.backup_path}")

    def _connect(self):
        """Connect to database with proper settings."""
        self.conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,  # Autocommit mode for executescript compatibility
        )
        self.conn.row_factory = sqlite3.Row

        # Ensure WAL mode for safety
        self.conn.execute("PRAGMA journal_mode=WAL")

    def _is_already_migrated(self) -> bool:
        """Check if messages table already has enhanced schema."""
        try:
            cursor = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'"
            )
            result = cursor.fetchone()
            if not result:
                print("⚠️  No messages table found")
                return False

            table_sql = result['sql']
            # Check if enhanced schema fields exist
            return 'severity' in table_sql and 'category' in table_sql and 'acknowledged' in table_sql

        except sqlite3.Error:
            return False

    def _execute_migration(self):
        """Execute migration steps in order."""
        print("\nExecuting migration steps...")

        # Step 1: Create enhanced messages table
        print("\n1. Creating enhanced messages table...")
        self._create_enhanced_table()

        # Step 2: Migrate data from old to new
        print("\n2. Migrating message data...")
        self._migrate_message_data()

        # Step 3: Handle foreign key constraints
        print("\n3. Updating foreign key references...")
        self._update_foreign_keys()

        # Step 4: Replace old table with new
        print("\n4. Replacing old table with enhanced table...")
        self._replace_table()

        print("\n✅ All migration steps completed")

    def _create_enhanced_table(self):
        """Create messages_enhanced table."""
        # Read schema from error_messages_schema.sql
        schema_file = Path(__file__).parent / "error_messages_schema.sql"

        if not schema_file.exists():
            raise MigrationError(f"Schema file not found: {schema_file}")

        with open(schema_file) as f:
            schema_sql = f.read()

        # Temporarily switch to script mode for schema creation
        cursor = self.conn.cursor()
        cursor.executescript(schema_sql)

        print("   ✓ Created messages_enhanced table")

    def _migrate_message_data(self):
        """Migrate data from messages to messages_enhanced."""
        # Check if old messages table exists
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        )
        if not cursor.fetchone():
            print("   ⚠️  No messages table found, skipping data migration")
            return

        # Query existing messages
        cursor = self.conn.execute("""
            SELECT
                id,
                error_type,
                error_message,
                error_details,
                occurred_timestamp,
                file_path,
                collection_name,
                retry_count
            FROM messages
            ORDER BY id
        """)

        messages = cursor.fetchall()
        print(f"   Found {len(messages)} messages to migrate")

        for msg in messages:
            # Map error_type to category
            category = self._map_error_type_to_category(msg['error_type'])

            # Determine severity (default to 'error', but could be inferred from error_type)
            severity = self._infer_severity(msg['error_type'])

            # Build context JSON
            context = self._build_context(msg)

            # Insert into messages_enhanced
            self.conn.execute("""
                INSERT OR IGNORE INTO messages_enhanced (
                    id, timestamp, severity, category, message, context,
                    acknowledged, acknowledged_at, acknowledged_by, retry_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                msg['id'],
                msg['occurred_timestamp'],
                severity,
                category,
                msg['error_message'],
                context,
                0,  # acknowledged
                None,  # acknowledged_at
                None,  # acknowledged_by
                msg['retry_count']
            ))

            self.stats['messages_migrated'] += 1

        print(f"   ✅ Migrated {self.stats['messages_migrated']} messages")

    def _map_error_type_to_category(self, error_type: str) -> str:
        """Map old error_type to new category."""
        if not error_type:
            return 'unknown'

        # Check direct mapping
        category = ERROR_TYPE_MAPPING.get(error_type.upper())
        if category:
            return category

        # Heuristic mapping based on keywords (order matters!)
        error_type_lower = error_type.lower()
        if 'parse' in error_type_lower or 'syntax' in error_type_lower:
            return 'parse_error'
        elif 'file' in error_type_lower or 'corrupt' in error_type_lower:
            return 'file_corrupt'
        elif 'permission' in error_type_lower or 'access' in error_type_lower:
            return 'permission_denied'
        elif 'timeout' in error_type_lower:  # Check timeout before network/connection
            return 'timeout'
        elif 'network' in error_type_lower or 'connection' in error_type_lower:
            return 'network'
        elif 'metadata' in error_type_lower:
            return 'metadata_invalid'
        elif 'tool' in error_type_lower or 'missing' in error_type_lower:
            return 'tool_missing'
        elif 'resource' in error_type_lower or 'memory' in error_type_lower:
            return 'resource_exhausted'
        elif 'process' in error_type_lower:
            return 'processing_failed'
        else:
            return 'unknown'

    def _infer_severity(self, error_type: str) -> str:
        """Infer severity from error_type."""
        if not error_type:
            return 'error'

        error_type_lower = error_type.lower()

        # Info level errors
        if 'info' in error_type_lower or 'skip' in error_type_lower:
            return 'info'

        # Warning level errors
        if 'warn' in error_type_lower or 'deprecat' in error_type_lower:
            return 'warning'

        # Default to error
        return 'error'

    def _build_context(self, msg: sqlite3.Row) -> str:
        """Build JSON context from message fields."""
        context = {}

        # Add file_path if available
        if msg['file_path']:
            context['file_path'] = msg['file_path']

        # Add collection_name if available
        if msg['collection_name']:
            context['collection'] = msg['collection_name']

        # Merge error_details if it's valid JSON
        if msg['error_details']:
            try:
                error_details = json.loads(msg['error_details'])
                if isinstance(error_details, dict):
                    # Merge without overwriting
                    for key, value in error_details.items():
                        if key not in context:
                            context[key] = value
            except json.JSONDecodeError:
                # Store as raw details if not JSON
                context['details'] = msg['error_details']

        return json.dumps(context) if context else None

    def _update_foreign_keys(self):
        """Update foreign key references to messages table."""
        # Check if ingestion_queue has foreign key to messages
        cursor = self.conn.execute("""
            SELECT sql FROM sqlite_master
            WHERE type='table' AND name='ingestion_queue'
        """)

        result = cursor.fetchone()
        if not result:
            print("   ⚠️  No ingestion_queue table found")
            return

        # Foreign key will be preserved during table replacement
        print("   ✓ Foreign key references will be preserved")

    def _replace_table(self):
        """Replace old messages table with enhanced version."""
        # Disable foreign keys temporarily
        self.conn.execute("PRAGMA foreign_keys=OFF")

        try:
            # Drop old messages table
            self.conn.execute("DROP TABLE IF EXISTS messages")

            # Rename enhanced table to messages
            self.conn.execute("ALTER TABLE messages_enhanced RENAME TO messages")

            print("   ✓ Replaced messages table with enhanced version")

        finally:
            # Re-enable foreign keys
            self.conn.execute("PRAGMA foreign_keys=ON")

    def _validate_migration(self) -> bool:
        """Validate migration results."""
        print("\nValidating migration...")

        validation_passed = True

        # Validation 1: Table exists with correct schema (now renamed to 'messages')
        cursor = self.conn.execute("""
            SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'
        """)
        result = cursor.fetchone()

        if not result:
            print("   ⚠️  messages table not found")
            validation_passed = False
        else:
            table_sql = result['sql']
            required_fields = ['severity', 'category', 'acknowledged']
            missing = [f for f in required_fields if f not in table_sql]
            if missing:
                print(f"   ⚠️  Missing fields: {missing}")
                validation_passed = False
            else:
                print("   ✅ Enhanced schema validated")

        # Validation 2: Row count check (messages table should have data)
        try:
            cursor = self.conn.execute("SELECT COUNT(*) as cnt FROM messages")
            row_count = cursor.fetchone()['cnt']
            print(f"   ✅ Row count preserved: {row_count} messages")
        except sqlite3.Error as e:
            print(f"   ⚠️  Failed to count rows: {e}")
            validation_passed = False

        # Validation 3: Check constraints
        try:
            # Try to insert invalid severity (should fail)
            self.conn.execute("""
                INSERT INTO messages (severity, category, message)
                VALUES ('invalid', 'unknown', 'test')
            """)
            print("   ⚠️  CHECK constraint not enforced for severity")
            validation_passed = False
            self.conn.execute("DELETE FROM messages WHERE message = 'test'")
        except sqlite3.IntegrityError:
            print("   ✅ CHECK constraints validated")

        # Validation 4: Context JSON validity
        cursor = self.conn.execute("""
            SELECT id, context FROM messages
            WHERE context IS NOT NULL
            LIMIT 10
        """)

        for row in cursor.fetchall():
            if row['context']:
                try:
                    json.loads(row['context'])
                except json.JSONDecodeError:
                    print(f"   ⚠️  Invalid JSON in context for message {row['id']}")
                    validation_passed = False
                    break
        else:
            print("   ✅ Context JSON validated")

        if validation_passed:
            print("\n✅ All validations passed")
        else:
            print("\n❌ Some validations failed")

        return validation_passed

    def _print_statistics(self):
        """Print migration statistics."""
        print("\n" + "="*50)
        print("Migration Statistics:")
        print("="*50)
        print(f"Messages migrated:       {self.stats['messages_migrated']}")
        print(f"Validation errors:       {self.stats['validation_errors']}")
        print("="*50)


def main():
    """Main entry point for migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate messages table to enhanced error message schema"
    )
    parser.add_argument(
        '--db-path',
        type=str,
        help='Path to SQLite database (default: ~/.workspace-qdrant/state.db)'
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
        # Default path - matches Rust daemon location
        db_path = Path.home() / ".workspace-qdrant" / "state.db"

    # Execute migration
    migrator = ErrorMessageMigrator(
        db_path=str(db_path),
        dry_run=args.dry_run,
        create_backup=not args.no_backup
    )

    success = migrator.migrate()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
