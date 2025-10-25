"""
Comprehensive integration tests for Language Support system.

Tests end-to-end workflows including:
- YAML parsing → Database loading → Querying
- Version tracking across multiple loads
- Conflict resolution for user customizations
- File language detection
- Missing metadata tracking and queries
"""

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from src.python.common.core.language_support_loader import LanguageSupportLoader
from src.python.common.core.language_support_manager import LanguageSupportManager
from src.python.common.core.language_support_parser import LanguageSupportParser
from src.python.common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def state_manager(tmp_path):
    """Create a temporary SQLite state manager for testing."""
    db_path = tmp_path / "test_integration.db"
    manager = SQLiteStateManager(db_path=str(db_path))
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def lang_manager(state_manager):
    """Create LanguageSupportManager instance."""
    return LanguageSupportManager(state_manager)


@pytest.fixture
def sample_yaml_file(tmp_path):
    """Create a sample language support YAML file for testing."""
    yaml_content = """
file_extensions:
  .py: python
  .pyi: python
  .rs: rust
  .js: javascript
  .jsx: javascript
  .ts: typescript
  .tsx: typescript

lsp_servers:
  python:
    primary: pylsp
    features:
      - symbols
      - completion
      - hover
    rationale: "Community-favorite Python LSP"
    install_notes: "pip install python-lsp-server"

  rust:
    primary: rust-analyzer
    features:
      - symbols
      - completion
      - diagnostics
    rationale: "Official Rust LSP"
    install_notes: "rustup component add rust-analyzer"

  javascript:
    primary: typescript-language-server
    features:
      - symbols
      - completion
    rationale: "Works for both JS and TS"
    install_notes: "npm install -g typescript-language-server"

tree_sitter_grammars:
  available:
    - python
    - rust
    - javascript
    - typescript
"""
    yaml_file = tmp_path / "languages_support.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


@pytest.fixture
def modified_yaml_file(tmp_path):
    """Create a modified version of the YAML file."""
    yaml_content = """
file_extensions:
  .py: python
  .pyi: python
  .rs: rust
  .js: javascript
  .jsx: javascript
  .ts: typescript
  .tsx: typescript
  .go: go

lsp_servers:
  python:
    primary: pylsp
    features:
      - symbols
      - completion
      - hover
    rationale: "Community-favorite Python LSP"
    install_notes: "pip install python-lsp-server"

  rust:
    primary: rust-analyzer
    features:
      - symbols
      - completion
      - diagnostics
    rationale: "Official Rust LSP"
    install_notes: "rustup component add rust-analyzer"

  go:
    primary: gopls
    features:
      - symbols
      - completion
    rationale: "Official Go LSP"
    install_notes: "go install golang.org/x/tools/gopls@latest"

tree_sitter_grammars:
  available:
    - python
    - rust
    - go
"""
    yaml_file = tmp_path / "languages_support_modified.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


class TestLanguageSupportIntegration:
    """Integration tests for language support system."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, lang_manager, sample_yaml_file):
        """Test complete workflow: Load YAML → Parse → Store → Query."""
        # 1. Load YAML file
        result = await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        assert result["skipped"] is False
        assert result["languages_loaded"] > 0
        assert len(result["version"]) == 16  # Short hash

        # 2. Verify version tracking
        version_info = await lang_manager.get_version_info()
        assert version_info is not None
        assert version_info["language_count"] == result["languages_loaded"]
        assert version_info["yaml_hash_short"] == result["version"]

        # 3. Query back languages
        languages = await lang_manager.get_supported_languages()
        assert len(languages) > 0

        # Verify Python language
        python_lang = next((l for l in languages if l["language_name"] == "python"), None)
        assert python_lang is not None
        assert ".py" in python_lang["file_extensions"]
        assert ".pyi" in python_lang["file_extensions"]
        assert python_lang["lsp_name"] == "pylsp"
        assert python_lang["lsp_available"] is True
        assert python_lang["ts_available"] is True

        # Verify Rust language
        rust_lang = next((l for l in languages if l["language_name"] == "rust"), None)
        assert rust_lang is not None
        assert ".rs" in rust_lang["file_extensions"]
        assert rust_lang["lsp_name"] == "rust-analyzer"

    @pytest.mark.asyncio
    async def test_initial_load(self, lang_manager, sample_yaml_file):
        """Test initial load of language support YAML."""
        # First load should process the file
        result = await lang_manager.initialize_from_yaml(sample_yaml_file)

        assert result["skipped"] is False
        assert result["languages_loaded"] == 4  # python, rust, javascript, typescript
        assert "version" in result

        # Verify database has languages
        languages = await lang_manager.get_supported_languages()
        assert len(languages) == 4

    @pytest.mark.asyncio
    async def test_reload_no_changes(self, lang_manager, sample_yaml_file):
        """Test reload with no changes (should skip)."""
        # First load
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # Second load should skip
        result = await lang_manager.initialize_from_yaml(sample_yaml_file, force=False)

        assert result["skipped"] is True
        assert result["languages_loaded"] == 0
        assert result["version"] == "unchanged"

    @pytest.mark.asyncio
    async def test_reload_with_changes(self, lang_manager, sample_yaml_file, modified_yaml_file):
        """Test reload with changes (should update)."""
        # First load
        result1 = await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)
        assert result1["languages_loaded"] == 4

        # Load modified file (has Go added)
        result2 = await lang_manager.initialize_from_yaml(modified_yaml_file, force=False)

        assert result2["skipped"] is False
        assert result2["languages_loaded"] == 5  # python, rust, go, javascript, typescript
        assert result2["version"] != result1["version"]

        # Verify Go language was added
        languages = await lang_manager.get_supported_languages()
        go_lang = next((l for l in languages if l["language_name"] == "go"), None)
        assert go_lang is not None
        assert ".go" in go_lang["file_extensions"]

    @pytest.mark.asyncio
    async def test_user_customization_preserved(self, lang_manager, sample_yaml_file, state_manager):
        """Test that user customizations are preserved during reload."""
        # Initial load
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # User customizes Python LSP path
        custom_lsp_path = "/custom/path/to/pylsp"
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE languages
                SET lsp_absolute_path = ?
                WHERE language_name = ?
                """,
                (custom_lsp_path, "python")
            )

        # Reload the same file (force=True to bypass hash check)
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # Verify custom path is preserved
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                """
                SELECT lsp_absolute_path FROM languages
                WHERE language_name = ?
                """,
                ("python",)
            )
            row = cursor.fetchone()

        assert row is not None
        assert row["lsp_absolute_path"] == custom_lsp_path

    @pytest.mark.asyncio
    async def test_file_language_detection(self, lang_manager, sample_yaml_file, tmp_path):
        """Test file language detection for various extensions."""
        # Load languages
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # Test Python detection
        py_file = tmp_path / "script.py"
        py_file.write_text("print('hello')")
        py_info = await lang_manager.get_language_for_file(py_file)

        assert py_info is not None
        assert py_info["language_name"] == "python"
        assert ".py" in py_info["file_extensions"]
        assert py_info["lsp_name"] == "pylsp"

        # Test Rust detection
        rs_file = tmp_path / "main.rs"
        rs_file.write_text("fn main() {}")
        rs_info = await lang_manager.get_language_for_file(rs_file)

        assert rs_info is not None
        assert rs_info["language_name"] == "rust"
        assert ".rs" in rs_info["file_extensions"]
        assert rs_info["lsp_name"] == "rust-analyzer"

        # Test TypeScript detection
        ts_file = tmp_path / "app.tsx"
        ts_file.write_text("const x: number = 1;")
        ts_info = await lang_manager.get_language_for_file(ts_file)

        assert ts_info is not None
        assert ts_info["language_name"] == "typescript"
        assert ".tsx" in ts_info["file_extensions"]

        # Test unknown extension
        unknown_file = tmp_path / "file.xyz"
        unknown_file.write_text("unknown")
        unknown_info = await lang_manager.get_language_for_file(unknown_file)

        assert unknown_info is None

    @pytest.mark.asyncio
    async def test_missing_metadata_tracking(self, lang_manager, sample_yaml_file, tmp_path):
        """Test tracking files with missing metadata."""
        # Load languages
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # Create test files
        py_file = tmp_path / "test.py"
        rs_file = tmp_path / "test.rs"
        py_file.write_text("test")
        rs_file.write_text("test")

        # Mark Python file as missing LSP metadata
        success1 = await lang_manager.mark_missing_metadata(
            py_file, "python", missing_lsp=True, missing_ts=False
        )
        assert success1 is True

        # Mark Rust file as missing both
        success2 = await lang_manager.mark_missing_metadata(
            rs_file, "rust", missing_lsp=True, missing_ts=True
        )
        assert success2 is True

        # Query all missing metadata
        all_missing = await lang_manager.get_files_missing_metadata()
        assert len(all_missing) == 2

        # Query only LSP missing
        lsp_missing = await lang_manager.get_files_missing_metadata(missing_lsp_only=True)
        assert len(lsp_missing) == 2  # Both have missing LSP

        # Query only Tree-sitter missing
        ts_missing = await lang_manager.get_files_missing_metadata(missing_ts_only=True)
        assert len(ts_missing) == 1  # Only Rust
        assert ts_missing[0]["language_name"] == "rust"

    @pytest.mark.asyncio
    async def test_missing_metadata_by_language(self, lang_manager, sample_yaml_file, tmp_path):
        """Test querying missing metadata filtered by language."""
        # Load languages
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # Create test files
        py_file1 = tmp_path / "test1.py"
        py_file2 = tmp_path / "test2.py"
        rs_file = tmp_path / "test.rs"
        py_file1.write_text("test")
        py_file2.write_text("test")
        rs_file.write_text("test")

        # Mark files
        await lang_manager.mark_missing_metadata(py_file1, "python", True, False)
        await lang_manager.mark_missing_metadata(py_file2, "python", True, True)
        await lang_manager.mark_missing_metadata(rs_file, "rust", False, True)

        # Query Python files only
        python_missing = await lang_manager.get_files_missing_metadata(language_name="python")
        assert len(python_missing) == 2
        assert all(m["language_name"] == "python" for m in python_missing)

        # Query Rust files only
        rust_missing = await lang_manager.get_files_missing_metadata(language_name="rust")
        assert len(rust_missing) == 1
        assert rust_missing[0]["language_name"] == "rust"
        assert rust_missing[0]["missing_ts_metadata"] is True

    @pytest.mark.asyncio
    async def test_version_tracking_multiple_loads(self, lang_manager, sample_yaml_file, modified_yaml_file):
        """Test version tracking across multiple loads."""
        # Initial load
        result1 = await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)
        version1 = await lang_manager.get_version_info()

        assert version1 is not None
        assert version1["language_count"] == result1["languages_loaded"]

        # Load modified YAML
        result2 = await lang_manager.initialize_from_yaml(modified_yaml_file, force=True)
        version2 = await lang_manager.get_version_info()

        assert version2 is not None
        assert version2["yaml_hash"] != version1["yaml_hash"]
        assert version2["language_count"] == result2["languages_loaded"]

        # Verify version is the latest one
        assert version2["loaded_at"] >= version1["loaded_at"]

    @pytest.mark.asyncio
    async def test_parser_validation_error_handling(self, lang_manager, tmp_path):
        """Test handling of YAML validation errors."""
        # Create invalid YAML file
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content:")

        # Should raise exception
        with pytest.raises(Exception):
            await lang_manager.initialize_from_yaml(invalid_yaml, force=True)

    @pytest.mark.asyncio
    async def test_conflict_resolution_updates_metadata(
        self, lang_manager, sample_yaml_file, state_manager
    ):
        """Test that conflict resolution updates lsp_missing and ts_missing flags."""
        # Initial load
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # Manually mark Python as having missing LSP
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE languages
                SET lsp_missing = 1, lsp_absolute_path = '/custom/path'
                WHERE language_name = ?
                """,
                ("python",)
            )

        # Reload (should preserve custom path but update lsp_missing to 0)
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # Verify
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                """
                SELECT lsp_missing, lsp_absolute_path FROM languages
                WHERE language_name = ?
                """,
                ("python",)
            )
            row = cursor.fetchone()

        assert row is not None
        assert row["lsp_missing"] == 0  # Updated from config
        assert row["lsp_absolute_path"] == "/custom/path"  # Preserved

    @pytest.mark.asyncio
    async def test_real_yaml_file_if_exists(self, lang_manager):
        """Test with the real language_support.yaml file if it exists."""
        real_yaml = Path(__file__).parent.parent.parent / "assets" / "languages_support.yaml"

        if not real_yaml.exists():
            pytest.skip("Real language_support.yaml file not found")

        # Load real YAML
        result = await lang_manager.initialize_from_yaml(real_yaml, force=True)

        assert result["skipped"] is False
        assert result["languages_loaded"] > 0

        # Verify we can query languages
        languages = await lang_manager.get_supported_languages()
        assert len(languages) > 0

        # Verify Python exists in real data
        python_lang = next((l for l in languages if l["language_name"] == "python"), None)
        assert python_lang is not None

    @pytest.mark.asyncio
    async def test_component_integration(self, state_manager, sample_yaml_file):
        """Test integration between Parser → Loader → Manager."""
        # 1. Parser parses YAML
        parser = LanguageSupportParser()
        config = parser.parse_yaml(sample_yaml_file)

        assert len(config.file_extensions) > 0
        assert len(config.lsp_servers) > 0
        assert ".py" in config.file_extensions

        # 2. Loader loads into database
        loader = LanguageSupportLoader(state_manager)
        count = await loader.load_languages(config)

        assert count > 0

        # 3. Manager can query loaded data
        manager = LanguageSupportManager(state_manager)
        languages = await manager.get_supported_languages()

        assert len(languages) == count

        # Verify specific language
        python_lang = next((l for l in languages if l["language_name"] == "python"), None)
        assert python_lang is not None
        assert python_lang["lsp_name"] == "pylsp"

    @pytest.mark.asyncio
    async def test_hash_calculation_consistency(self, lang_manager, sample_yaml_file):
        """Test that hash calculation is consistent across multiple checks."""
        # First check
        needs_update1 = await lang_manager.check_for_updates(sample_yaml_file)
        assert needs_update1 is True  # No previous version

        # Load
        await lang_manager.initialize_from_yaml(sample_yaml_file, force=True)

        # Second check (should be consistent)
        needs_update2 = await lang_manager.check_for_updates(sample_yaml_file)
        assert needs_update2 is False

        # Third check (still consistent)
        needs_update3 = await lang_manager.check_for_updates(sample_yaml_file)
        assert needs_update3 is False
