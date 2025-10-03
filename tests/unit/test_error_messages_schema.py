"""
Unit tests for error messages schema and migration.

Tests cover:
- Schema creation
- Field constraints
- Index creation
- Data migration
- Validation logic
- Edge cases
"""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.python.common.core.migrate_error_messages import (
    ErrorMessageMigrator,
    ERROR_TYPE_MAPPING,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def db_with_old_schema(temp_db):
    """Create a database with the old messages schema."""
    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA foreign_keys=ON")

    # Create old messages table
    conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            error_type TEXT NOT NULL,
            error_message TEXT NOT NULL,
            error_details TEXT,
            occurred_timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            file_path TEXT,
            collection_name TEXT,
            retry_count INTEGER DEFAULT 0
        )
    """)

    # Create ingestion_queue table with foreign key
    conn.execute("""
        CREATE TABLE ingestion_queue (
            file_absolute_path TEXT PRIMARY KEY NOT NULL,
            collection_name TEXT NOT NULL,
            operation TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 5,
            error_message_id INTEGER,
            FOREIGN KEY (error_message_id) REFERENCES messages(id) ON DELETE SET NULL
        )
    """)

    conn.commit()
    conn.close()

    return temp_db


@pytest.fixture
def db_with_sample_data(db_with_old_schema):
    """Populate database with sample error messages."""
    conn = sqlite3.connect(db_with_old_schema)

    # Insert sample messages
    sample_messages = [
        (1, 'PARSE_ERROR', 'Failed to parse Python file', '{"line": 42}', '2024-01-01 10:00:00', '/path/to/file.py', 'my-project', 0),
        (2, 'FILE_NOT_FOUND', 'File does not exist', None, '2024-01-01 11:00:00', '/missing/file.txt', 'my-project', 1),
        (3, 'NETWORK_ERROR', 'Connection timeout', '{"url": "http://example.com"}', '2024-01-01 12:00:00', None, 'library-docs', 2),
        (4, 'PERMISSION_ERROR', 'Access denied', None, '2024-01-01 13:00:00', '/protected/file.py', 'my-project', 0),
        (5, 'UNKNOWN_TYPE', 'Something went wrong', '{"details": "unexpected"}', '2024-01-01 14:00:00', None, None, 3),
    ]

    for msg in sample_messages:
        conn.execute("""
            INSERT INTO messages (id, error_type, error_message, error_details, occurred_timestamp, file_path, collection_name, retry_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, msg)

    # Insert queue item with reference
    conn.execute("""
        INSERT INTO ingestion_queue (file_absolute_path, collection_name, operation, error_message_id)
        VALUES ('/path/to/file.py', 'my-project', 'ingest', 1)
    """)

    conn.commit()
    conn.close()

    return db_with_old_schema


class TestErrorMessagesSchema:
    """Test error messages schema creation and constraints."""

    def test_schema_file_exists(self):
        """Test that schema SQL file exists."""
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        assert schema_file.exists(), "Schema file should exist"

    def test_create_enhanced_table(self, temp_db):
        """Test creating the enhanced messages table."""
        conn = sqlite3.connect(temp_db)

        # Read and execute schema
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Execute schema
        conn.executescript(schema_sql)

        # Verify table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_enhanced'")
        assert cursor.fetchone() is not None, "messages_enhanced table should exist"

        # Verify columns
        cursor = conn.execute("PRAGMA table_info(messages_enhanced)")
        columns = {row[1] for row in cursor.fetchall()}
        expected_columns = {'id', 'timestamp', 'severity', 'category', 'message', 'context', 'acknowledged', 'acknowledged_at', 'acknowledged_by', 'retry_count'}
        assert columns == expected_columns, f"Table should have expected columns: {expected_columns}"

        conn.close()

    def test_severity_constraint(self, temp_db):
        """Test severity field constraint."""
        conn = sqlite3.connect(temp_db)

        # Create table
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        with open(schema_file, 'r') as f:
            conn.executescript(f.read())

        # Test valid severities
        for severity in ['error', 'warning', 'info']:
            conn.execute("""
                INSERT INTO messages_enhanced (severity, category, message)
                VALUES (?, 'unknown', 'test message')
            """, (severity,))

        # Test invalid severity
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO messages_enhanced (severity, category, message)
                VALUES ('invalid', 'unknown', 'test message')
            """)

        conn.close()

    def test_category_constraint(self, temp_db):
        """Test category field constraint."""
        conn = sqlite3.connect(temp_db)

        # Create table
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        with open(schema_file, 'r') as f:
            conn.executescript(f.read())

        # Test valid categories
        valid_categories = ['file_corrupt', 'tool_missing', 'network', 'metadata_invalid', 'processing_failed', 'parse_error', 'permission_denied', 'resource_exhausted', 'timeout', 'unknown']
        for category in valid_categories:
            conn.execute("""
                INSERT INTO messages_enhanced (severity, category, message)
                VALUES ('error', ?, 'test message')
            """, (category,))

        # Test invalid category
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO messages_enhanced (severity, category, message)
                VALUES ('error', 'invalid_category', 'test message')
            """)

        conn.close()

    def test_acknowledged_constraint(self, temp_db):
        """Test acknowledged boolean constraint."""
        conn = sqlite3.connect(temp_db)

        # Create table
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        with open(schema_file, 'r') as f:
            conn.executescript(f.read())

        # Test valid values (0 and 1)
        conn.execute("""
            INSERT INTO messages_enhanced (severity, category, message, acknowledged)
            VALUES ('error', 'unknown', 'test 1', 0)
        """)

        conn.execute("""
            INSERT INTO messages_enhanced (severity, category, message, acknowledged, acknowledged_by)
            VALUES ('error', 'unknown', 'test 2', 1, 'test_user')
        """)

        # Test invalid value
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO messages_enhanced (severity, category, message, acknowledged)
                VALUES ('error', 'unknown', 'test 3', 2)
            """)

        conn.close()

    def test_acknowledged_trigger(self, temp_db):
        """Test trigger for setting acknowledged_at timestamp."""
        conn = sqlite3.connect(temp_db)

        # Create table
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        with open(schema_file, 'r') as f:
            conn.executescript(f.read())

        # Insert message
        conn.execute("""
            INSERT INTO messages_enhanced (severity, category, message, acknowledged)
            VALUES ('error', 'unknown', 'test message', 0)
        """)

        # Update to acknowledged
        conn.execute("""
            UPDATE messages_enhanced
            SET acknowledged = 1, acknowledged_by = 'test_user'
            WHERE id = 1
        """)

        # Check acknowledged_at was set
        cursor = conn.execute("SELECT acknowledged_at FROM messages_enhanced WHERE id = 1")
        result = cursor.fetchone()
        assert result[0] is not None, "acknowledged_at should be set automatically"

        conn.close()

    def test_acknowledged_by_validation_trigger(self, temp_db):
        """Test trigger that validates acknowledged_by is set."""
        conn = sqlite3.connect(temp_db)

        # Create table
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        with open(schema_file, 'r') as f:
            conn.executescript(f.read())

        # Insert message
        conn.execute("""
            INSERT INTO messages_enhanced (severity, category, message, acknowledged)
            VALUES ('error', 'unknown', 'test message', 0)
        """)

        # Try to acknowledge without setting acknowledged_by (should fail)
        with pytest.raises(sqlite3.IntegrityError, match="acknowledged_by must be set"):
            conn.execute("""
                UPDATE messages_enhanced
                SET acknowledged = 1
                WHERE id = 1
            """)

        conn.close()

    def test_indexes_created(self, temp_db):
        """Test that all indexes are created."""
        conn = sqlite3.connect(temp_db)

        # Create table
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        with open(schema_file, 'r') as f:
            conn.executescript(f.read())

        # Check indexes
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = {row[0] for row in cursor.fetchall()}

        expected_indexes = {
            'idx_messages_enhanced_severity',
            'idx_messages_enhanced_category',
            'idx_messages_enhanced_timestamp',
            'idx_messages_enhanced_acknowledged',
            'idx_messages_enhanced_severity_ack',
            'idx_messages_enhanced_retry_count',
        }

        assert expected_indexes.issubset(indexes), f"All expected indexes should be created: {expected_indexes}"

        conn.close()

    def test_views_created(self, temp_db):
        """Test that convenience views are created."""
        conn = sqlite3.connect(temp_db)

        # Create table
        schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
        with open(schema_file, 'r') as f:
            conn.executescript(f.read())

        # Check views
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = {row[0] for row in cursor.fetchall()}

        expected_views = {'unacknowledged_messages', 'message_statistics'}
        assert expected_views.issubset(views), f"All expected views should be created: {expected_views}"

        conn.close()


class TestErrorMessageMigration:
    """Test error message migration logic."""

    def test_migrator_initialization(self, temp_db):
        """Test migrator initialization."""
        migrator = ErrorMessageMigrator(temp_db, dry_run=True, create_backup=False)
        assert migrator.db_path == Path(temp_db)
        assert migrator.dry_run is True
        assert migrator.create_backup is False

    def test_preconditions_validation(self, temp_db):
        """Test preconditions validation."""
        migrator = ErrorMessageMigrator(temp_db, dry_run=True, create_backup=False)
        assert migrator._validate_preconditions() is True

    def test_preconditions_missing_db(self):
        """Test preconditions with missing database."""
        migrator = ErrorMessageMigrator("/nonexistent/db.sqlite", dry_run=True, create_backup=False)
        assert migrator._validate_preconditions() is False

    def test_error_type_mapping(self, temp_db):
        """Test error type to category mapping."""
        migrator = ErrorMessageMigrator(temp_db, dry_run=True, create_backup=False)

        # Test direct mappings
        assert migrator._map_error_type_to_category('PARSE_ERROR') == 'parse_error'
        assert migrator._map_error_type_to_category('FILE_NOT_FOUND') == 'file_corrupt'
        assert migrator._map_error_type_to_category('NETWORK_ERROR') == 'network'

        # Test heuristic mappings
        assert migrator._map_error_type_to_category('SyntaxError') == 'parse_error'
        assert migrator._map_error_type_to_category('PermissionDenied') == 'permission_denied'
        assert migrator._map_error_type_to_category('ConnectionTimeout') == 'timeout'

        # Test unknown
        assert migrator._map_error_type_to_category('UNKNOWN_TYPE') == 'unknown'

    def test_severity_inference(self, temp_db):
        """Test severity inference from error type."""
        migrator = ErrorMessageMigrator(temp_db, dry_run=True, create_backup=False)

        # Test info
        assert migrator._infer_severity('INFO_MESSAGE') == 'info'
        assert migrator._infer_severity('SKIP_FILE') == 'info'

        # Test warning
        assert migrator._infer_severity('WARNING_DEPRECATED') == 'warning'

        # Test error (default)
        assert migrator._infer_severity('FATAL_ERROR') == 'error'
        assert migrator._infer_severity('PARSE_ERROR') == 'error'

    def test_context_building(self, db_with_old_schema):
        """Test context JSON building from old schema fields."""
        conn = sqlite3.connect(db_with_old_schema)
        conn.row_factory = sqlite3.Row

        # Insert test message
        conn.execute("""
            INSERT INTO messages (error_type, error_message, error_details, file_path, collection_name)
            VALUES ('TEST', 'test message', '{"extra": "data"}', '/path/to/file', 'test-collection')
        """)

        cursor = conn.execute("SELECT * FROM messages WHERE id = 1")
        msg = cursor.fetchone()

        migrator = ErrorMessageMigrator(db_with_old_schema, dry_run=True, create_backup=False)
        context_json = migrator._build_context(msg)
        context = json.loads(context_json)

        assert context['file_path'] == '/path/to/file'
        assert context['collection'] == 'test-collection'
        assert context['extra'] == 'data'

        conn.close()

    def test_migration_dry_run(self, db_with_sample_data):
        """Test migration in dry-run mode."""
        migrator = ErrorMessageMigrator(db_with_sample_data, dry_run=True, create_backup=False)
        success = migrator.migrate()
        assert success is True

        # Verify no changes were made
        conn = sqlite3.connect(db_with_sample_data)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages_enhanced'")
        assert cursor.fetchone() is None, "messages_enhanced should not exist in dry-run"
        conn.close()

    def test_migration_full(self, db_with_sample_data):
        """Test full migration."""
        migrator = ErrorMessageMigrator(db_with_sample_data, dry_run=False, create_backup=False)
        success = migrator.migrate()
        assert success is True

        # Verify migration
        conn = sqlite3.connect(db_with_sample_data)

        # Check table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        assert cursor.fetchone() is not None, "messages table should exist"

        # Check all rows migrated
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        assert count == 5, "All 5 messages should be migrated"

        # Check schema has new fields
        cursor = conn.execute("PRAGMA table_info(messages)")
        columns = {row[1] for row in cursor.fetchall()}
        assert 'severity' in columns
        assert 'category' in columns
        assert 'acknowledged' in columns

        # Check specific message
        cursor = conn.execute("SELECT severity, category, message FROM messages WHERE id = 1")
        row = cursor.fetchone()
        assert row[0] == 'error'  # severity
        assert row[1] == 'parse_error'  # category
        assert row[2] == 'Failed to parse Python file'  # message

        conn.close()

    def test_migration_preserves_foreign_keys(self, db_with_sample_data):
        """Test that migration preserves foreign key relationships."""
        migrator = ErrorMessageMigrator(db_with_sample_data, dry_run=False, create_backup=False)
        success = migrator.migrate()
        assert success is True

        conn = sqlite3.connect(db_with_sample_data)
        conn.execute("PRAGMA foreign_keys=ON")

        # Verify foreign key still works
        cursor = conn.execute("""
            SELECT iq.file_absolute_path, m.message
            FROM ingestion_queue iq
            JOIN messages m ON iq.error_message_id = m.id
        """)
        result = cursor.fetchone()
        assert result is not None, "Foreign key join should work"

        conn.close()

    def test_migration_idempotent(self, db_with_sample_data):
        """Test that migration can be run multiple times safely."""
        migrator = ErrorMessageMigrator(db_with_sample_data, dry_run=False, create_backup=False)

        # Run first migration
        success1 = migrator.migrate()
        assert success1 is True

        # Run second migration (should detect already migrated)
        migrator2 = ErrorMessageMigrator(db_with_sample_data, dry_run=False, create_backup=False)
        success2 = migrator2.migrate()
        assert success2 is True

        # Verify data integrity
        conn = sqlite3.connect(db_with_sample_data)
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        assert count == 5, "Should still have 5 messages"
        conn.close()

    def test_context_json_validity(self, db_with_sample_data):
        """Test that all migrated context fields are valid JSON."""
        migrator = ErrorMessageMigrator(db_with_sample_data, dry_run=False, create_backup=False)
        success = migrator.migrate()
        assert success is True

        conn = sqlite3.connect(db_with_sample_data)
        cursor = conn.execute("SELECT id, context FROM messages WHERE context IS NOT NULL")

        for row in cursor.fetchall():
            msg_id, context_json = row
            try:
                context = json.loads(context_json)
                assert isinstance(context, dict), f"Context for message {msg_id} should be a dict"
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON in context for message {msg_id}: {context_json}")

        conn.close()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_database(self, db_with_old_schema):
        """Test migration with empty messages table."""
        migrator = ErrorMessageMigrator(db_with_old_schema, dry_run=False, create_backup=False)
        success = migrator.migrate()
        assert success is True

        conn = sqlite3.connect(db_with_old_schema)
        cursor = conn.execute("SELECT COUNT(*) FROM messages")
        assert cursor.fetchone()[0] == 0, "Should have 0 messages"
        conn.close()

    def test_null_fields(self, db_with_old_schema):
        """Test migration with NULL fields."""
        conn = sqlite3.connect(db_with_old_schema)

        # Insert message with NULL fields
        conn.execute("""
            INSERT INTO messages (error_type, error_message)
            VALUES ('TEST', 'minimal message')
        """)
        conn.commit()
        conn.close()

        migrator = ErrorMessageMigrator(db_with_old_schema, dry_run=False, create_backup=False)
        success = migrator.migrate()
        assert success is True

        conn = sqlite3.connect(db_with_old_schema)
        cursor = conn.execute("SELECT category, context FROM messages WHERE id = 1")
        row = cursor.fetchone()
        assert row[0] == 'unknown', "NULL error_type should map to unknown category"
        # context can be NULL
        conn.close()

    def test_special_characters_in_message(self, db_with_old_schema):
        """Test migration with special characters."""
        conn = sqlite3.connect(db_with_old_schema)

        # Insert message with special characters
        conn.execute("""
            INSERT INTO messages (error_type, error_message, file_path)
            VALUES ('TEST', 'Message with "quotes" and \nnewlines', '/path/with spaces/file.txt')
        """)
        conn.commit()
        conn.close()

        migrator = ErrorMessageMigrator(db_with_old_schema, dry_run=False, create_backup=False)
        success = migrator.migrate()
        assert success is True

        conn = sqlite3.connect(db_with_old_schema)
        cursor = conn.execute("SELECT message, context FROM messages WHERE id = 1")
        row = cursor.fetchone()
        assert 'quotes' in row[0]
        assert 'newlines' in row[0]

        context = json.loads(row[1])
        assert context['file_path'] == '/path/with spaces/file.txt'

        conn.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
