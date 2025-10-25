"""
Unit tests for LanguageSupportManager.

Tests the orchestration of language support YAML parsing, database loading,
version tracking, and language detection functionality.
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.python.common.core.language_support_manager import LanguageSupportManager
from src.python.common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def state_manager():
    """Create a temporary SQLite state manager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        manager = SQLiteStateManager(db_path=str(db_path))
        await manager.initialize()
        yield manager
        await manager.close()


@pytest.fixture
async def lang_manager(state_manager):
    """Create LanguageSupportManager instance."""
    return LanguageSupportManager(state_manager)


@pytest.fixture
async def setup_test_languages(state_manager):
    """Helper fixture to create test language entries in database."""
    async def _setup(*language_names):
        language_data = {
            "python": (json.dumps([".py", ".pyi"]), "ruff-lsp", "ruff-lsp", "python"),
            "javascript": (json.dumps([".js", ".mjs"]), "tsserver", "tsserver", "javascript"),
            "rust": (json.dumps([".rs"]), "rust-analyzer", "rust-analyzer", "rust"),
        }

        async with state_manager.transaction() as conn:
            for lang_name in language_names:
                if lang_name in language_data:
                    extensions, lsp_name, lsp_exec, ts_grammar = language_data[lang_name]
                    conn.execute(
                        """
                        INSERT INTO languages
                        (language_name, file_extensions, lsp_name, lsp_executable,
                         lsp_missing, ts_grammar, ts_missing)
                        VALUES (?, ?, ?, ?, 0, ?, 0)
                        """,
                        (lang_name, extensions, lsp_name, lsp_exec, ts_grammar)
                    )

    return _setup


@pytest.fixture
def sample_yaml_content():
    """Sample YAML content for testing."""
    return b"""
languages:
  python:
    extensions: [".py", ".pyi"]
    lsp: "ruff-lsp"
    tree_sitter: "python"

  javascript:
    extensions: [".js", ".mjs"]
    lsp: "typescript-language-server"
    tree_sitter: "javascript"
"""


@pytest.fixture
def sample_yaml_file(tmp_path, sample_yaml_content):
    """Create a temporary YAML file for testing."""
    yaml_file = tmp_path / "languages_support.yaml"
    yaml_file.write_bytes(sample_yaml_content)
    return yaml_file


class TestLanguageSupportManager:
    """Test suite for LanguageSupportManager class."""

    @pytest.mark.asyncio
    async def test_initialization(self, lang_manager, state_manager):
        """Test LanguageSupportManager initialization."""
        assert lang_manager.state_manager == state_manager

    @pytest.mark.asyncio
    async def test_check_for_updates_no_previous_version(
        self, lang_manager, sample_yaml_file
    ):
        """Test check_for_updates when no previous version exists."""
        needs_update = await lang_manager.check_for_updates(sample_yaml_file)
        assert needs_update is True

    @pytest.mark.asyncio
    async def test_check_for_updates_unchanged(
        self, lang_manager, sample_yaml_file, sample_yaml_content
    ):
        """Test check_for_updates when YAML hasn't changed."""
        # Calculate hash and store in database
        yaml_hash = hashlib.sha256(sample_yaml_content).hexdigest()

        async with lang_manager.state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO language_support_version
                (yaml_hash, language_count)
                VALUES (?, ?)
                """,
                (yaml_hash, 2)
            )

        # Check should return False (no update needed)
        needs_update = await lang_manager.check_for_updates(sample_yaml_file)
        assert needs_update is False

    @pytest.mark.asyncio
    async def test_check_for_updates_changed(
        self, lang_manager, tmp_path, sample_yaml_content
    ):
        """Test check_for_updates when YAML has changed."""
        # Store original hash in database
        original_hash = hashlib.sha256(sample_yaml_content).hexdigest()

        async with lang_manager.state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO language_support_version
                (yaml_hash, language_count)
                VALUES (?, ?)
                """,
                (original_hash, 2)
            )

        # Create a different YAML file
        new_content = b"different content"
        yaml_file = tmp_path / "changed.yaml"
        yaml_file.write_bytes(new_content)

        # Check should return True (update needed)
        needs_update = await lang_manager.check_for_updates(yaml_file)
        assert needs_update is True

    @pytest.mark.asyncio
    async def test_check_for_updates_file_not_found(self, lang_manager):
        """Test check_for_updates with non-existent file."""
        with pytest.raises(FileNotFoundError):
            await lang_manager.check_for_updates(Path("/nonexistent/file.yaml"))

    @pytest.mark.asyncio
    async def test_initialize_from_yaml_placeholder(
        self, lang_manager, sample_yaml_file
    ):
        """Test initialize_from_yaml with placeholder implementation."""
        # Since parser and loader aren't implemented yet, we test the placeholder
        result = await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        assert result["languages_loaded"] == 0  # Placeholder returns 0
        assert result["skipped"] is False
        assert "version" in result

        # Verify version tracking was updated
        version_info = await lang_manager.get_version_info()
        assert version_info is not None
        assert version_info["language_count"] == 0

    @pytest.mark.asyncio
    async def test_initialize_from_yaml_skip_unchanged(
        self, lang_manager, sample_yaml_file, sample_yaml_content
    ):
        """Test initialize_from_yaml skips when YAML unchanged."""
        # Store current hash to simulate already loaded
        yaml_hash = hashlib.sha256(sample_yaml_content).hexdigest()

        async with lang_manager.state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO language_support_version
                (yaml_hash, language_count)
                VALUES (?, ?)
                """,
                (yaml_hash, 2)
            )

        # Should skip loading
        result = await lang_manager.initialize_from_yaml(sample_yaml_file, force=False)

        assert result["skipped"] is True
        assert result["languages_loaded"] == 0
        assert result["version"] == "unchanged"

    @pytest.mark.asyncio
    async def test_initialize_from_yaml_force_reload(
        self, lang_manager, sample_yaml_file, sample_yaml_content
    ):
        """Test initialize_from_yaml with force=True always reloads."""
        # Store current hash
        yaml_hash = hashlib.sha256(sample_yaml_content).hexdigest()

        async with lang_manager.state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO language_support_version
                (yaml_hash, language_count)
                VALUES (?, ?)
                """,
                (yaml_hash, 2)
            )

        # Force reload should not skip
        result = await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        assert result["skipped"] is False

    @pytest.mark.asyncio
    async def test_get_language_for_file_not_found(self, lang_manager):
        """Test get_language_for_file with no matching language."""
        result = await lang_manager.get_language_for_file(Path("test.xyz"))
        assert result is None

    @pytest.mark.asyncio
    async def test_get_language_for_file_no_extension(self, lang_manager):
        """Test get_language_for_file with file without extension."""
        result = await lang_manager.get_language_for_file(Path("Makefile"))
        assert result is None

    @pytest.mark.asyncio
    async def test_get_language_for_file_found(self, lang_manager, state_manager):
        """Test get_language_for_file with matching language."""
        # Insert test language into database
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages
                (language_name, file_extensions, lsp_name, lsp_executable,
                 lsp_missing, ts_grammar, ts_missing)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "python",
                    json.dumps([".py", ".pyi"]),
                    "ruff-lsp",
                    "ruff-lsp",
                    0,
                    "python",
                    0
                )
            )

        # Test detection
        result = await lang_manager.get_language_for_file(Path("test.py"))

        assert result is not None
        assert result["language_name"] == "python"
        assert ".py" in result["file_extensions"]
        assert result["lsp_name"] == "ruff-lsp"
        assert result["lsp_missing"] is False
        assert result["ts_grammar"] == "python"
        assert result["ts_missing"] is False

    @pytest.mark.asyncio
    async def test_get_language_for_file_multiple_matches(
        self, lang_manager, state_manager
    ):
        """Test get_language_for_file when multiple languages match."""
        # Insert two languages with overlapping extensions
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages
                (language_name, file_extensions, lsp_name, lsp_missing, ts_missing)
                VALUES
                ('python', ?, 'ruff-lsp', 0, 0),
                ('cython', ?, 'cython-lsp', 1, 1)
                """,
                (
                    json.dumps([".py", ".pyx"]),
                    json.dumps([".py", ".pyx"])
                )
            )

        # Should return first match
        result = await lang_manager.get_language_for_file(Path("test.py"))

        assert result is not None
        # Will be either python or cython depending on query order
        assert result["language_name"] in ["python", "cython"]

    @pytest.mark.asyncio
    async def test_mark_missing_metadata_lsp(
        self, lang_manager, tmp_path, setup_test_languages
    ):
        """Test marking file as missing LSP metadata."""
        # Create the language entry first
        await setup_test_languages("python")

        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        success = await lang_manager.mark_missing_metadata(
            test_file,
            "python",
            missing_lsp=True,
            missing_ts=False
        )

        assert success is True

        # Verify in database
        with lang_manager.state_manager._lock:
            cursor = lang_manager.state_manager.connection.execute(
                """
                SELECT missing_lsp_metadata, missing_ts_metadata
                FROM files_missing_metadata
                WHERE file_absolute_path = ?
                """,
                (str(test_file.resolve()),)
            )
            row = cursor.fetchone()

        assert row is not None
        assert row["missing_lsp_metadata"] == 1
        assert row["missing_ts_metadata"] == 0

    @pytest.mark.asyncio
    async def test_mark_missing_metadata_both(
        self, lang_manager, tmp_path, setup_test_languages
    ):
        """Test marking file as missing both LSP and Tree-sitter metadata."""
        # Create the language entry first
        await setup_test_languages("javascript")

        test_file = tmp_path / "test.js"
        test_file.write_text("console.log('hello')")

        success = await lang_manager.mark_missing_metadata(
            test_file,
            "javascript",
            missing_lsp=True,
            missing_ts=True
        )

        assert success is True

        # Verify in database
        with lang_manager.state_manager._lock:
            cursor = lang_manager.state_manager.connection.execute(
                """
                SELECT missing_lsp_metadata, missing_ts_metadata
                FROM files_missing_metadata
                WHERE file_absolute_path = ?
                """,
                (str(test_file.resolve()),)
            )
            row = cursor.fetchone()

        assert row is not None
        assert row["missing_lsp_metadata"] == 1
        assert row["missing_ts_metadata"] == 1

    @pytest.mark.asyncio
    async def test_mark_missing_metadata_update(
        self, lang_manager, tmp_path, setup_test_languages
    ):
        """Test updating existing missing metadata record."""
        # Create the language entry first
        await setup_test_languages("python")

        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        # First mark as missing LSP only
        await lang_manager.mark_missing_metadata(
            test_file,
            "python",
            missing_lsp=True,
            missing_ts=False
        )

        # Then mark as missing both (should update)
        success = await lang_manager.mark_missing_metadata(
            test_file,
            "python",
            missing_lsp=True,
            missing_ts=True
        )

        assert success is True

        # Verify updated values
        with lang_manager.state_manager._lock:
            cursor = lang_manager.state_manager.connection.execute(
                """
                SELECT missing_lsp_metadata, missing_ts_metadata
                FROM files_missing_metadata
                WHERE file_absolute_path = ?
                """,
                (str(test_file.resolve()),)
            )
            row = cursor.fetchone()

        assert row is not None
        assert row["missing_lsp_metadata"] == 1
        assert row["missing_ts_metadata"] == 1

    @pytest.mark.asyncio
    async def test_get_files_missing_metadata_empty(self, lang_manager):
        """Test get_files_missing_metadata with no records."""
        results = await lang_manager.get_files_missing_metadata()
        assert results == []

    @pytest.mark.asyncio
    async def test_get_files_missing_metadata_all(
        self, lang_manager, tmp_path, setup_test_languages
    ):
        """Test get_files_missing_metadata returns all records."""
        # Create language entries first
        await setup_test_languages("python", "javascript")

        # Create test files and mark them
        file1 = tmp_path / "test1.py"
        file2 = tmp_path / "test2.js"
        file1.write_text("test")
        file2.write_text("test")

        await lang_manager.mark_missing_metadata(
            file1, "python", missing_lsp=True, missing_ts=False
        )
        await lang_manager.mark_missing_metadata(
            file2, "javascript", missing_lsp=False, missing_ts=True
        )

        # Get all
        results = await lang_manager.get_files_missing_metadata()

        assert len(results) == 2
        assert any(r["language_name"] == "python" for r in results)
        assert any(r["language_name"] == "javascript" for r in results)

    @pytest.mark.asyncio
    async def test_get_files_missing_metadata_filtered_by_language(
        self, lang_manager, tmp_path, setup_test_languages
    ):
        """Test get_files_missing_metadata filtered by language."""
        # Create language entries first
        await setup_test_languages("python", "javascript")

        file1 = tmp_path / "test1.py"
        file2 = tmp_path / "test2.js"
        file1.write_text("test")
        file2.write_text("test")

        await lang_manager.mark_missing_metadata(
            file1, "python", missing_lsp=True, missing_ts=False
        )
        await lang_manager.mark_missing_metadata(
            file2, "javascript", missing_lsp=False, missing_ts=True
        )

        # Filter by language
        results = await lang_manager.get_files_missing_metadata(language_name="python")

        assert len(results) == 1
        assert results[0]["language_name"] == "python"

    @pytest.mark.asyncio
    async def test_get_files_missing_metadata_lsp_only(
        self, lang_manager, tmp_path, setup_test_languages
    ):
        """Test get_files_missing_metadata with LSP filter."""
        # Create language entries first
        await setup_test_languages("python", "javascript")

        file1 = tmp_path / "test1.py"
        file2 = tmp_path / "test2.js"
        file1.write_text("test")
        file2.write_text("test")

        await lang_manager.mark_missing_metadata(
            file1, "python", missing_lsp=True, missing_ts=False
        )
        await lang_manager.mark_missing_metadata(
            file2, "javascript", missing_lsp=False, missing_ts=True
        )

        # Filter by missing LSP only
        results = await lang_manager.get_files_missing_metadata(missing_lsp_only=True)

        assert len(results) == 1
        assert results[0]["missing_lsp_metadata"] is True

    @pytest.mark.asyncio
    async def test_get_supported_languages_empty(self, lang_manager):
        """Test get_supported_languages with no languages."""
        results = await lang_manager.get_supported_languages()
        assert results == []

    @pytest.mark.asyncio
    async def test_get_supported_languages(self, lang_manager, state_manager):
        """Test get_supported_languages returns all languages."""
        # Insert test languages
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages
                (language_name, file_extensions, lsp_name, lsp_missing,
                 ts_grammar, ts_missing)
                VALUES
                ('python', ?, 'ruff-lsp', 0, 'python', 0),
                ('javascript', ?, 'tsserver', 1, 'javascript', 0)
                """,
                (
                    json.dumps([".py"]),
                    json.dumps([".js"])
                )
            )

        results = await lang_manager.get_supported_languages()

        assert len(results) == 2

        python_lang = next((l for l in results if l["language_name"] == "python"), None)
        assert python_lang is not None
        assert python_lang["lsp_available"] is True
        assert python_lang["ts_available"] is True

        js_lang = next((l for l in results if l["language_name"] == "javascript"), None)
        assert js_lang is not None
        assert js_lang["lsp_available"] is False  # lsp_missing = 1
        assert js_lang["ts_available"] is True

    @pytest.mark.asyncio
    async def test_get_version_info_not_initialized(self, lang_manager):
        """Test get_version_info when no version exists."""
        result = await lang_manager.get_version_info()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_version_info(
        self, lang_manager, state_manager, sample_yaml_content
    ):
        """Test get_version_info returns version information."""
        # Insert version
        yaml_hash = hashlib.sha256(sample_yaml_content).hexdigest()

        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO language_support_version
                (yaml_hash, language_count, last_checked_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (yaml_hash, 42)
            )

        result = await lang_manager.get_version_info()

        assert result is not None
        assert result["yaml_hash"] == yaml_hash
        assert result["yaml_hash_short"] == yaml_hash[:16]
        assert result["language_count"] == 42
        assert "loaded_at" in result
        assert "last_checked_at" in result
