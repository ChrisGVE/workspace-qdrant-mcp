"""
Comprehensive unit tests for language support schema (Schema Version 4).

Tests all 4 new tables with comprehensive coverage:
- languages: Language definitions with LSP and Tree-sitter support
- files_missing_metadata: Files missing LSP or Tree-sitter metadata
- tools: LSP servers and Tree-sitter CLI tools
- language_support_version: YAML configuration version tracking

Coverage includes:
- Table creation and schema verification
- Foreign key constraints and CASCADE behavior
- Index creation and query performance
- CRUD operations
- Constraint violations (UNIQUE, CHECK)
- Schema migration from v3 to v4
- Relationship integrity
"""

import asyncio
import pytest
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.python.common.core.sqlite_state_manager import SQLiteStateManager


class TestLanguageSupportSchemaTables:
    """Test table creation and schema verification."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    async def state_manager(self, temp_db):
        """Create state manager with initialized schema."""
        manager = SQLiteStateManager(db_path=str(temp_db))
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_all_tables_created(self, state_manager):
        """Test that all 4 language support tables are created."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN (
                    'languages', 'files_missing_metadata', 'tools', 'language_support_version'
                )
                ORDER BY name
            """)
            tables = [row[0] for row in cursor.fetchall()]

        assert len(tables) == 4
        assert 'files_missing_metadata' in tables
        assert 'language_support_version' in tables
        assert 'languages' in tables
        assert 'tools' in tables

    @pytest.mark.asyncio
    async def test_languages_table_schema(self, state_manager):
        """Test languages table has correct schema."""
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "PRAGMA table_info(languages)"
            )
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        expected_columns = {
            'id': 'INTEGER',
            'language_name': 'TEXT',
            'file_extensions': 'TEXT',
            'lsp_name': 'TEXT',
            'lsp_executable': 'TEXT',
            'lsp_absolute_path': 'TEXT',
            'lsp_missing': 'BOOLEAN',
            'ts_grammar': 'TEXT',
            'ts_cli_absolute_path': 'TEXT',
            'ts_missing': 'BOOLEAN',
            'created_at': 'TIMESTAMP',
            'updated_at': 'TIMESTAMP',
        }

        for col_name, col_type in expected_columns.items():
            assert col_name in columns, f"Column {col_name} missing"
            assert columns[col_name] == col_type, f"Column {col_name} has wrong type"

    @pytest.mark.asyncio
    async def test_files_missing_metadata_table_schema(self, state_manager):
        """Test files_missing_metadata table has correct schema."""
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "PRAGMA table_info(files_missing_metadata)"
            )
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        expected_columns = {
            'id': 'INTEGER',
            'file_absolute_path': 'TEXT',
            'language_name': 'TEXT',
            'branch': 'TEXT',
            'missing_lsp_metadata': 'BOOLEAN',
            'missing_ts_metadata': 'BOOLEAN',
            'created_at': 'TIMESTAMP',
            'updated_at': 'TIMESTAMP',
        }

        for col_name, col_type in expected_columns.items():
            assert col_name in columns, f"Column {col_name} missing"
            assert columns[col_name] == col_type, f"Column {col_name} has wrong type"

    @pytest.mark.asyncio
    async def test_tools_table_schema(self, state_manager):
        """Test tools table has correct schema."""
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "PRAGMA table_info(tools)"
            )
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        expected_columns = {
            'id': 'INTEGER',
            'tool_name': 'TEXT',
            'tool_type': 'TEXT',
            'absolute_path': 'TEXT',
            'version': 'TEXT',
            'missing': 'BOOLEAN',
            'last_check_at': 'TIMESTAMP',
            'created_at': 'TIMESTAMP',
            'updated_at': 'TIMESTAMP',
        }

        for col_name, col_type in expected_columns.items():
            assert col_name in columns, f"Column {col_name} missing"
            assert columns[col_name] == col_type, f"Column {col_name} has wrong type"

    @pytest.mark.asyncio
    async def test_language_support_version_table_schema(self, state_manager):
        """Test language_support_version table has correct schema."""
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "PRAGMA table_info(language_support_version)"
            )
            columns = {row[1]: row[2] for row in cursor.fetchall()}

        expected_columns = {
            'id': 'INTEGER',
            'yaml_hash': 'TEXT',
            'loaded_at': 'TIMESTAMP',
            'language_count': 'INTEGER',
            'last_checked_at': 'TIMESTAMP',
        }

        for col_name, col_type in expected_columns.items():
            assert col_name in columns, f"Column {col_name} missing"
            assert columns[col_name] == col_type, f"Column {col_name} has wrong type"


class TestLanguageSupportIndexes:
    """Test index creation and query performance."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    async def state_manager(self, temp_db):
        """Create state manager with initialized schema."""
        manager = SQLiteStateManager(db_path=str(temp_db))
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_languages_indexes_created(self, state_manager):
        """Test that all indexes are created for languages table."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='languages'
            """)
            indexes = [row[0] for row in cursor.fetchall()]

        expected_indexes = [
            'idx_languages_language_name',
            'idx_languages_lsp_missing',
            'idx_languages_ts_missing',
        ]

        for index_name in expected_indexes:
            assert index_name in indexes, f"Index {index_name} not found"

    @pytest.mark.asyncio
    async def test_files_missing_metadata_indexes_created(self, state_manager):
        """Test that all indexes are created for files_missing_metadata table."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='files_missing_metadata'
            """)
            indexes = [row[0] for row in cursor.fetchall()]

        expected_indexes = [
            'idx_files_missing_metadata_file_path',
            'idx_files_missing_metadata_language',
            'idx_files_missing_metadata_missing',
        ]

        for index_name in expected_indexes:
            assert index_name in indexes, f"Index {index_name} not found"

    @pytest.mark.asyncio
    async def test_tools_indexes_created(self, state_manager):
        """Test that all indexes are created for tools table."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='tools'
            """)
            indexes = [row[0] for row in cursor.fetchall()]

        expected_indexes = [
            'idx_tools_tool_name',
            'idx_tools_tool_type_missing',
        ]

        for index_name in expected_indexes:
            assert index_name in indexes, f"Index {index_name} not found"

    @pytest.mark.asyncio
    async def test_language_support_version_indexes_created(self, state_manager):
        """Test that all indexes are created for language_support_version table."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND tbl_name='language_support_version'
            """)
            indexes = [row[0] for row in cursor.fetchall()]

        expected_indexes = [
            'idx_language_support_version_yaml_hash',
            'idx_language_support_version_loaded_at',
        ]

        for index_name in expected_indexes:
            assert index_name in indexes, f"Index {index_name} not found"

    @pytest.mark.asyncio
    async def test_index_used_in_query(self, state_manager):
        """Test that indexes are actually used in queries."""
        # Insert test data
        with state_manager._lock:
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_missing)
                VALUES ('python', 0)
            """)

            # Check query plan uses index
            cursor = state_manager.connection.execute("""
                EXPLAIN QUERY PLAN
                SELECT * FROM languages WHERE language_name = 'python'
            """)
            plan_rows = cursor.fetchall()

        # Convert each row to string to check content
        plan_str = ' '.join(' '.join(str(val) for val in row) for row in plan_rows)
        assert 'idx_languages_language_name' in plan_str or 'SEARCH' in plan_str


class TestLanguageSupportCRUD:
    """Test CRUD operations on language support tables."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    async def state_manager(self, temp_db):
        """Create state manager with initialized schema."""
        manager = SQLiteStateManager(db_path=str(temp_db))
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_insert_language(self, state_manager):
        """Test inserting a language record."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                INSERT INTO languages (
                    language_name, file_extensions, lsp_name, lsp_executable,
                    lsp_absolute_path, lsp_missing, ts_grammar, ts_missing
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, ('python', '["py", "pyi"]', 'pyright', 'pyright',
                  '/usr/local/bin/pyright', 0, 'python', 0))

            # Verify insertion
            cursor = state_manager.connection.execute(
                "SELECT * FROM languages WHERE language_name = 'python'"
            )
            row = cursor.fetchone()

        assert row is not None
        assert row['language_name'] == 'python'
        assert row['lsp_name'] == 'pyright'
        assert row['lsp_missing'] == 0
        assert row['ts_missing'] == 0

    @pytest.mark.asyncio
    async def test_update_language(self, state_manager):
        """Test updating a language record."""
        with state_manager._lock:
            # Insert
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_missing)
                VALUES ('python', 1)
            """)

            # Update
            state_manager.connection.execute("""
                UPDATE languages
                SET lsp_missing = 0, lsp_absolute_path = '/usr/local/bin/pyright'
                WHERE language_name = 'python'
            """)

            # Verify
            cursor = state_manager.connection.execute(
                "SELECT lsp_missing, lsp_absolute_path FROM languages WHERE language_name = 'python'"
            )
            row = cursor.fetchone()

        assert row['lsp_missing'] == 0
        assert row['lsp_absolute_path'] == '/usr/local/bin/pyright'

    @pytest.mark.asyncio
    async def test_delete_language(self, state_manager):
        """Test deleting a language record."""
        with state_manager._lock:
            # Insert
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_missing)
                VALUES ('test_lang', 0)
            """)

            # Delete
            cursor = state_manager.connection.execute(
                "DELETE FROM languages WHERE language_name = 'test_lang'"
            )

            # Verify deletion
            cursor = state_manager.connection.execute(
                "SELECT COUNT(*) FROM languages WHERE language_name = 'test_lang'"
            )
            count = cursor.fetchone()[0]

        assert count == 0

    @pytest.mark.asyncio
    async def test_insert_files_missing_metadata(self, state_manager):
        """Test inserting files_missing_metadata record."""
        with state_manager._lock:
            # Insert language first
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_missing)
                VALUES ('python', 0)
            """)

            # Insert file record
            cursor = state_manager.connection.execute("""
                INSERT INTO files_missing_metadata (
                    file_absolute_path, language_name, branch,
                    missing_lsp_metadata, missing_ts_metadata
                )
                VALUES (?, ?, ?, ?, ?)
            """, ('/path/to/file.py', 'python', 'main', 1, 0))

            # Verify
            cursor = state_manager.connection.execute(
                "SELECT * FROM files_missing_metadata WHERE file_absolute_path = '/path/to/file.py'"
            )
            row = cursor.fetchone()

        assert row is not None
        assert row['language_name'] == 'python'
        assert row['missing_lsp_metadata'] == 1
        assert row['missing_ts_metadata'] == 0

    @pytest.mark.asyncio
    async def test_insert_tool(self, state_manager):
        """Test inserting tool record."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                INSERT INTO tools (
                    tool_name, tool_type, absolute_path, version, missing
                )
                VALUES (?, ?, ?, ?, ?)
            """, ('pyright', 'lsp_server', '/usr/local/bin/pyright', '1.1.0', 0))

            # Verify
            cursor = state_manager.connection.execute(
                "SELECT * FROM tools WHERE tool_name = 'pyright'"
            )
            row = cursor.fetchone()

        assert row is not None
        assert row['tool_type'] == 'lsp_server'
        assert row['version'] == '1.1.0'
        assert row['missing'] == 0

    @pytest.mark.asyncio
    async def test_insert_language_support_version(self, state_manager):
        """Test inserting language_support_version record."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("""
                INSERT INTO language_support_version (
                    yaml_hash, language_count
                )
                VALUES (?, ?)
            """, ('abc123def456', 50))

            # Verify
            cursor = state_manager.connection.execute(
                "SELECT * FROM language_support_version WHERE yaml_hash = 'abc123def456'"
            )
            row = cursor.fetchone()

        assert row is not None
        assert row['language_count'] == 50
        assert row['loaded_at'] is not None


class TestLanguageSupportConstraints:
    """Test UNIQUE and CHECK constraints."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    async def state_manager(self, temp_db):
        """Create state manager with initialized schema."""
        manager = SQLiteStateManager(db_path=str(temp_db))
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_language_name_unique_constraint(self, state_manager):
        """Test UNIQUE constraint on language_name."""
        with state_manager._lock:
            # Insert first record
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_missing)
                VALUES ('python', 0)
            """)

            # Attempt to insert duplicate should fail
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
                state_manager.connection.execute("""
                    INSERT INTO languages (language_name, lsp_missing)
                    VALUES ('python', 1)
                """)

    @pytest.mark.asyncio
    async def test_tool_name_unique_constraint(self, state_manager):
        """Test UNIQUE constraint on tool_name."""
        with state_manager._lock:
            # Insert first record
            state_manager.connection.execute("""
                INSERT INTO tools (tool_name, tool_type, missing)
                VALUES ('pyright', 'lsp_server', 0)
            """)

            # Attempt to insert duplicate should fail
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
                state_manager.connection.execute("""
                    INSERT INTO tools (tool_name, tool_type, missing)
                    VALUES ('pyright', 'tree_sitter_cli', 0)
                """)

    @pytest.mark.asyncio
    async def test_yaml_hash_unique_constraint(self, state_manager):
        """Test UNIQUE constraint on yaml_hash."""
        with state_manager._lock:
            # Insert first record
            state_manager.connection.execute("""
                INSERT INTO language_support_version (yaml_hash, language_count)
                VALUES ('abc123', 50)
            """)

            # Attempt to insert duplicate should fail
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
                state_manager.connection.execute("""
                    INSERT INTO language_support_version (yaml_hash, language_count)
                    VALUES ('abc123', 60)
                """)

    @pytest.mark.asyncio
    async def test_file_absolute_path_unique_constraint(self, state_manager):
        """Test UNIQUE constraint on file_absolute_path."""
        with state_manager._lock:
            # Insert language
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_missing)
                VALUES ('python', 0)
            """)

            # Insert first file record
            state_manager.connection.execute("""
                INSERT INTO files_missing_metadata (
                    file_absolute_path, language_name, missing_lsp_metadata, missing_ts_metadata
                )
                VALUES ('/path/to/file.py', 'python', 1, 0)
            """)

            # Attempt to insert duplicate should fail
            with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
                state_manager.connection.execute("""
                    INSERT INTO files_missing_metadata (
                        file_absolute_path, language_name, missing_lsp_metadata, missing_ts_metadata
                    )
                    VALUES ('/path/to/file.py', 'python', 0, 1)
                """)

    @pytest.mark.asyncio
    async def test_tool_type_check_constraint_valid(self, state_manager):
        """Test CHECK constraint on tool_type with valid values."""
        with state_manager._lock:
            # Should succeed with 'lsp_server'
            state_manager.connection.execute("""
                INSERT INTO tools (tool_name, tool_type, missing)
                VALUES ('pyright', 'lsp_server', 0)
            """)

            # Should succeed with 'tree_sitter_cli'
            state_manager.connection.execute("""
                INSERT INTO tools (tool_name, tool_type, missing)
                VALUES ('tree-sitter', 'tree_sitter_cli', 0)
            """)

            cursor = state_manager.connection.execute(
                "SELECT COUNT(*) FROM tools"
            )
            count = cursor.fetchone()[0]

        assert count == 2

    @pytest.mark.asyncio
    async def test_tool_type_check_constraint_invalid(self, state_manager):
        """Test CHECK constraint on tool_type with invalid value."""
        with state_manager._lock:
            # Should fail with invalid tool_type
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
                state_manager.connection.execute("""
                    INSERT INTO tools (tool_name, tool_type, missing)
                    VALUES ('invalid_tool', 'invalid_type', 0)
                """)


class TestLanguageSupportForeignKeys:
    """Test foreign key constraints and CASCADE behavior."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    async def state_manager(self, temp_db):
        """Create state manager with initialized schema."""
        manager = SQLiteStateManager(db_path=str(temp_db))
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_foreign_key_enforcement_enabled(self, state_manager):
        """Test that foreign key enforcement is enabled."""
        with state_manager._lock:
            cursor = state_manager.connection.execute("PRAGMA foreign_keys")
            fk_status = cursor.fetchone()[0]

        assert fk_status == 1, "Foreign keys should be enabled"

    @pytest.mark.asyncio
    async def test_foreign_key_cascade_on_delete_set_null(self, state_manager):
        """Test ON DELETE SET NULL behavior for language_name foreign key."""
        with state_manager._lock:
            # Insert language
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_missing)
                VALUES ('python', 0)
            """)

            # Insert file referencing the language
            state_manager.connection.execute("""
                INSERT INTO files_missing_metadata (
                    file_absolute_path, language_name, missing_lsp_metadata, missing_ts_metadata
                )
                VALUES ('/path/to/file.py', 'python', 1, 0)
            """)

            # Delete the language
            state_manager.connection.execute(
                "DELETE FROM languages WHERE language_name = 'python'"
            )

            # Verify file record still exists but language_name is NULL
            cursor = state_manager.connection.execute(
                "SELECT language_name FROM files_missing_metadata WHERE file_absolute_path = '/path/to/file.py'"
            )
            row = cursor.fetchone()

        assert row is not None
        assert row['language_name'] is None

    @pytest.mark.asyncio
    async def test_foreign_key_invalid_language_reference(self, state_manager):
        """Test that invalid language reference fails."""
        with state_manager._lock:
            # Attempt to insert file with non-existent language should fail
            with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
                state_manager.connection.execute("""
                    INSERT INTO files_missing_metadata (
                        file_absolute_path, language_name, missing_lsp_metadata, missing_ts_metadata
                    )
                    VALUES ('/path/to/file.py', 'nonexistent_lang', 1, 0)
                """)


class TestSchemaMigration:
    """Test schema migration from v3 to v4."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    def _create_v3_schema(self, db_path: Path):
        """Create a v3 schema database for migration testing."""
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA foreign_keys=ON")

        # Create minimal v3 schema
        schema_sql = [
            """
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            "INSERT INTO schema_version (version) VALUES (3)",
            """
            CREATE TABLE file_processing (
                file_path TEXT PRIMARY KEY,
                collection TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 2,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                language_id TEXT,
                lsp_extracted BOOLEAN NOT NULL DEFAULT 0
            )
            """,
        ]

        for sql in schema_sql:
            conn.execute(sql)

        conn.commit()
        conn.close()

    @pytest.mark.asyncio
    async def test_migration_from_v3_to_v4(self, temp_db):
        """Test successful migration from schema v3 to v4."""
        # Create v3 schema
        self._create_v3_schema(temp_db)

        # Initialize state manager (should trigger migration)
        manager = SQLiteStateManager(db_path=str(temp_db))
        await manager.initialize()

        try:
            # Verify schema version is updated
            with manager._lock:
                cursor = manager.connection.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                )
                current_version = cursor.fetchone()[0]

            assert current_version == 4

            # Verify all v4 tables exist
            with manager._lock:
                cursor = manager.connection.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name IN (
                        'languages', 'files_missing_metadata', 'tools', 'language_support_version'
                    )
                """)
                tables = [row[0] for row in cursor.fetchall()]

            assert len(tables) == 4
            assert 'languages' in tables
            assert 'files_missing_metadata' in tables
            assert 'tools' in tables
            assert 'language_support_version' in tables

        finally:
            await manager.close()

    @pytest.mark.asyncio
    async def test_migration_preserves_existing_data(self, temp_db):
        """Test that migration preserves existing data."""
        # Create v3 schema
        self._create_v3_schema(temp_db)

        # Insert test data in v3 schema
        conn = sqlite3.connect(str(temp_db))
        conn.execute("""
            INSERT INTO file_processing (file_path, collection, status)
            VALUES ('/test/file.py', 'test-collection', 'completed')
        """)
        conn.commit()
        conn.close()

        # Migrate to v4
        manager = SQLiteStateManager(db_path=str(temp_db))
        await manager.initialize()

        try:
            # Verify data still exists
            with manager._lock:
                cursor = manager.connection.execute(
                    "SELECT file_path, collection, status FROM file_processing"
                )
                row = cursor.fetchone()

            assert row is not None
            assert row[0] == '/test/file.py'
            assert row[1] == 'test-collection'
            assert row[2] == 'completed'

        finally:
            await manager.close()


class TestRelationshipIntegrity:
    """Test relationship integrity between tables."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_path = Path(temp_file.name)
        temp_file.close()
        yield temp_path
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    async def state_manager(self, temp_db):
        """Create state manager with initialized schema."""
        manager = SQLiteStateManager(db_path=str(temp_db))
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_language_to_files_relationship(self, state_manager):
        """Test relationship between languages and files_missing_metadata."""
        with state_manager._lock:
            # Insert language
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_missing)
                VALUES ('python', 0)
            """)

            # Insert files referencing the language
            state_manager.connection.execute("""
                INSERT INTO files_missing_metadata (
                    file_absolute_path, language_name, missing_lsp_metadata, missing_ts_metadata
                )
                VALUES
                    ('/file1.py', 'python', 1, 0),
                    ('/file2.py', 'python', 0, 1)
            """)

            # Query files for this language
            cursor = state_manager.connection.execute("""
                SELECT f.file_absolute_path, f.missing_lsp_metadata
                FROM files_missing_metadata f
                WHERE f.language_name = 'python'
                ORDER BY f.file_absolute_path
            """)
            files = cursor.fetchall()

        assert len(files) == 2
        assert files[0]['file_absolute_path'] == '/file1.py'
        assert files[0]['missing_lsp_metadata'] == 1

    @pytest.mark.asyncio
    async def test_complete_workflow(self, state_manager):
        """Test complete workflow with all tables."""
        with state_manager._lock:
            # 1. Record language support version
            state_manager.connection.execute("""
                INSERT INTO language_support_version (yaml_hash, language_count)
                VALUES ('abc123', 2)
            """)

            # 2. Add languages
            state_manager.connection.execute("""
                INSERT INTO languages (language_name, lsp_name, lsp_missing, ts_missing)
                VALUES
                    ('python', 'pyright', 0, 0),
                    ('rust', 'rust-analyzer', 1, 0)
            """)

            # 3. Add tools
            state_manager.connection.execute("""
                INSERT INTO tools (tool_name, tool_type, missing)
                VALUES
                    ('pyright', 'lsp_server', 0),
                    ('rust-analyzer', 'lsp_server', 1)
            """)

            # 4. Add files with missing metadata
            state_manager.connection.execute("""
                INSERT INTO files_missing_metadata (
                    file_absolute_path, language_name, missing_lsp_metadata, missing_ts_metadata
                )
                VALUES
                    ('/project/main.py', 'python', 0, 0),
                    ('/project/lib.rs', 'rust', 1, 0)
            """)

            # Query complete data
            cursor = state_manager.connection.execute("""
                SELECT
                    l.language_name,
                    l.lsp_missing,
                    COUNT(f.file_absolute_path) as file_count
                FROM languages l
                LEFT JOIN files_missing_metadata f ON l.language_name = f.language_name
                GROUP BY l.language_name
                ORDER BY l.language_name
            """)
            results = cursor.fetchall()

        assert len(results) == 2
        assert results[0]['language_name'] == 'python'
        assert results[0]['file_count'] == 1
        assert results[1]['language_name'] == 'rust'
        assert results[1]['file_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
