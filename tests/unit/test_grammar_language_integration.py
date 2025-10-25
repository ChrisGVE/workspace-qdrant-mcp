"""
Unit tests for Grammar-Language Support Integration.

Tests the GrammarLanguageIntegrator class that bridges tree-sitter grammar
management with the language support system.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.python.common.core.grammar_compiler import CompilationResult
from src.python.common.core.grammar_discovery import GrammarInfo
from src.python.common.core.grammar_language_integration import (
    GrammarLanguageIntegrator,
)
from src.python.common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def state_manager():
    """Create SQLiteStateManager with test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    manager = SQLiteStateManager(db_path)
    await manager.initialize()

    yield manager

    # Cleanup
    await manager.close()
    db_path.unlink(missing_ok=True)


@pytest.fixture
async def integrator(state_manager):
    """Create GrammarLanguageIntegrator instance."""
    return GrammarLanguageIntegrator(state_manager)


@pytest.fixture
async def sample_languages(state_manager):
    """Insert sample languages into database."""
    languages = [
        ("python", json.dumps([".py", ".pyi"]), "pyright", 0, "tree-sitter-python", 1),
        ("rust", json.dumps([".rs"]), "rust-analyzer", 0, "tree-sitter-rust", 1),
        ("javascript", json.dumps([".js", ".jsx"]), "typescript-language-server", 0, "tree-sitter-javascript", 1),
    ]

    async with state_manager.transaction() as conn:
        for lang_name, exts, lsp, lsp_missing, ts_grammar, ts_missing in languages:
            conn.execute("""
                INSERT INTO languages
                (language_name, file_extensions, lsp_name, lsp_missing, ts_grammar, ts_missing)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (lang_name, exts, lsp, lsp_missing, ts_grammar, ts_missing))


class TestGrammarLanguageIntegrator:
    """Tests for GrammarLanguageIntegrator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, state_manager):
        """Test integrator initialization."""
        integrator = GrammarLanguageIntegrator(state_manager)

        assert integrator.state_manager == state_manager
        assert integrator.language_support is not None
        assert integrator.grammar_discovery is not None
        assert integrator.config_manager is not None

    @pytest.mark.asyncio
    async def test_sync_grammar_availability_with_grammars(self, integrator, sample_languages):
        """Test syncing grammar availability when grammars are found."""
        # Mock grammar discovery to return grammars
        mock_grammars = {
            "python": GrammarInfo(
                name="python",
                path=Path("/grammars/tree-sitter-python"),
                parser_path=Path("/grammars/tree-sitter-python/build/parser.so")
            ),
            "rust": GrammarInfo(
                name="rust",
                path=Path("/grammars/tree-sitter-rust"),
                parser_path=Path("/grammars/tree-sitter-rust/build/parser.so")
            )
        }

        integrator.grammar_discovery.discover_grammars = Mock(return_value=mock_grammars)

        # Sync grammars
        result = await integrator.sync_grammar_availability()

        assert result["grammars_found"] == 2
        assert result["grammars_synced"] == 2
        assert "python" in result["newly_available"]
        assert "rust" in result["newly_available"]
        assert "javascript" in result["still_missing"]

    @pytest.mark.asyncio
    async def test_sync_grammar_availability_no_grammars(self, integrator, sample_languages):
        """Test syncing when no grammars are available."""
        integrator.grammar_discovery.discover_grammars = Mock(return_value={})

        result = await integrator.sync_grammar_availability()

        assert result["grammars_found"] == 0
        assert len(result["newly_available"]) == 0
        assert len(result["still_missing"]) == 3

    @pytest.mark.asyncio
    async def test_get_grammar_for_language_available(self, integrator, sample_languages):
        """Test getting grammar for an available language."""
        # Mock grammar discovery
        python_grammar = GrammarInfo(
            name="python",
            path=Path("/grammars/tree-sitter-python"),
            parser_path=Path("/grammars/tree-sitter-python/build/parser.so")
        )

        integrator.grammar_discovery.get_grammar = Mock(return_value=python_grammar)

        # Update database to mark python grammar as available
        async with integrator.state_manager.transaction() as conn:
            conn.execute("""
                UPDATE languages
                SET ts_missing = 0, ts_cli_absolute_path = '/usr/bin/tree-sitter'
                WHERE language_name = 'python'
            """)

        grammar = await integrator.get_grammar_for_language("python")

        assert grammar is not None
        assert grammar.name == "python"
        assert grammar.path == Path("/grammars/tree-sitter-python")

    @pytest.mark.asyncio
    async def test_get_grammar_for_language_missing(self, integrator, sample_languages):
        """Test getting grammar for language without grammar."""
        grammar = await integrator.get_grammar_for_language("javascript")

        # Should return None because ts_missing = 1
        assert grammar is None

    @pytest.mark.asyncio
    async def test_get_grammar_for_language_not_found(self, integrator, sample_languages):
        """Test getting grammar for non-existent language."""
        grammar = await integrator.get_grammar_for_language("nonexistent")

        assert grammar is None

    @pytest.mark.asyncio
    async def test_get_grammar_for_file(self, integrator, sample_languages):
        """Test getting grammar for a file based on extension."""
        # Mock grammar discovery
        python_grammar = GrammarInfo(
            name="python",
            path=Path("/grammars/tree-sitter-python"),
            parser_path=Path("/grammars/tree-sitter-python/build/parser.so")
        )

        integrator.grammar_discovery.get_grammar = Mock(return_value=python_grammar)

        # Update database
        async with integrator.state_manager.transaction() as conn:
            conn.execute("""
                UPDATE languages
                SET ts_missing = 0
                WHERE language_name = 'python'
            """)

        grammar = await integrator.get_grammar_for_file(Path("/code/test.py"))

        assert grammar is not None
        assert grammar.name == "python"

    @pytest.mark.asyncio
    async def test_get_grammar_for_file_no_language(self, integrator, sample_languages):
        """Test getting grammar for file with unknown extension."""
        grammar = await integrator.get_grammar_for_file(Path("/code/test.xyz"))

        assert grammar is None

    @pytest.mark.asyncio
    async def test_has_grammar_for_language_true(self, integrator, sample_languages):
        """Test checking grammar availability (available)."""
        # Mark python as having grammar
        async with integrator.state_manager.transaction() as conn:
            conn.execute("""
                UPDATE languages
                SET ts_missing = 0
                WHERE language_name = 'python'
            """)

        has_grammar = await integrator.has_grammar_for_language("python")

        assert has_grammar is True

    @pytest.mark.asyncio
    async def test_has_grammar_for_language_false(self, integrator, sample_languages):
        """Test checking grammar availability (not available)."""
        has_grammar = await integrator.has_grammar_for_language("javascript")

        assert has_grammar is False

    @pytest.mark.asyncio
    async def test_ensure_grammar_compiled_already_compiled(self, integrator, sample_languages):
        """Test ensuring grammar is compiled when already compiled."""
        # Mock grammar with existing parser
        python_grammar = GrammarInfo(
            name="python",
            path=Path("/grammars/tree-sitter-python"),
            parser_path=Path("/grammars/tree-sitter-python/build/parser.so"),
            has_external_scanner=False
        )

        integrator.grammar_discovery.get_grammar = Mock(return_value=python_grammar)

        # Mark as available in DB
        async with integrator.state_manager.transaction() as conn:
            conn.execute("""
                UPDATE languages
                SET ts_missing = 0
                WHERE language_name = 'python'
            """)

        # Mock parser path existence
        with patch.object(Path, 'exists', return_value=True):
            success, message = await integrator.ensure_grammar_compiled("python")

        assert success is True
        assert "already compiled" in message.lower()

    @pytest.mark.asyncio
    async def test_ensure_grammar_compiled_not_installed(self, integrator, sample_languages):
        """Test ensuring compilation when grammar not installed."""
        integrator.grammar_discovery.get_grammar = Mock(return_value=None)

        success, message = await integrator.ensure_grammar_compiled("nonexistent")

        assert success is False
        assert "not installed" in message.lower()

    @pytest.mark.asyncio
    async def test_get_languages_with_grammars(self, integrator, sample_languages):
        """Test getting list of languages with grammars."""
        # Mark some languages as having grammars
        async with integrator.state_manager.transaction() as conn:
            conn.execute("""
                UPDATE languages
                SET ts_missing = 0
                WHERE language_name IN ('python', 'rust')
            """)

        languages = await integrator.get_languages_with_grammars()

        assert len(languages) == 2
        assert "python" in languages
        assert "rust" in languages
        assert "javascript" not in languages

    @pytest.mark.asyncio
    async def test_get_languages_missing_grammars(self, integrator, sample_languages):
        """Test getting list of languages without grammars."""
        # All languages initially have ts_missing = 1
        languages = await integrator.get_languages_missing_grammars()

        assert len(languages) == 3
        assert "python" in languages
        assert "rust" in languages
        assert "javascript" in languages

    @pytest.mark.asyncio
    async def test_get_grammar_stats(self, integrator, sample_languages):
        """Test getting grammar statistics."""
        # Mark python and rust as having grammars
        async with integrator.state_manager.transaction() as conn:
            conn.execute("""
                UPDATE languages
                SET ts_missing = 0
                WHERE language_name IN ('python', 'rust')
            """)

        stats = await integrator.get_grammar_stats()

        assert stats["total_languages"] == 3
        assert stats["with_grammars"] == 2
        assert stats["missing_grammars"] == 1
        assert stats["grammar_coverage"] == pytest.approx(66.67, rel=0.1)
        assert len(stats["languages_with_grammars"]) == 2
        assert len(stats["languages_missing_grammars"]) == 1

    @pytest.mark.asyncio
    async def test_sync_with_force_refresh(self, integrator, sample_languages):
        """Test syncing with forced grammar discovery refresh."""
        mock_grammars = {
            "python": GrammarInfo(
                name="python",
                path=Path("/grammars/tree-sitter-python")
            )
        }

        integrator.grammar_discovery.discover_grammars = Mock(return_value=mock_grammars)

        result = await integrator.sync_grammar_availability(force_refresh=True)

        # Verify force_refresh was passed to discover_grammars
        integrator.grammar_discovery.discover_grammars.assert_called_once_with(True)
        assert result["grammars_found"] == 1

    @pytest.mark.asyncio
    async def test_error_handling_in_sync(self, integrator, sample_languages):
        """Test error handling during sync operation."""
        # Make discover_grammars raise an exception
        integrator.grammar_discovery.discover_grammars = Mock(
            side_effect=RuntimeError("Discovery failed")
        )

        with pytest.raises(RuntimeError, match="Discovery failed"):
            await integrator.sync_grammar_availability()

    @pytest.mark.asyncio
    async def test_get_language_info_helper(self, integrator, sample_languages):
        """Test internal language info helper method."""
        lang_info = await integrator._get_language_info("python")

        assert lang_info is not None
        assert lang_info["language_name"] == "python"
        assert lang_info["lsp_name"] == "pyright"
        assert lang_info["lsp_available"] is True
        assert lang_info["ts_grammar"] == "tree-sitter-python"
        assert lang_info["ts_available"] is False  # ts_missing = 1

    @pytest.mark.asyncio
    async def test_get_language_info_not_found(self, integrator, sample_languages):
        """Test language info helper with non-existent language."""
        lang_info = await integrator._get_language_info("nonexistent")

        assert lang_info is None

    @pytest.mark.asyncio
    async def test_multiple_sync_updates_correctly(self, integrator, sample_languages):
        """Test that multiple syncs update database correctly."""
        # First sync - python available
        integrator.grammar_discovery.discover_grammars = Mock(return_value={
            "python": GrammarInfo(name="python", path=Path("/grammars/python"))
        })

        result1 = await integrator.sync_grammar_availability()
        assert "python" in result1["newly_available"]

        # Second sync - python and rust available
        integrator.grammar_discovery.discover_grammars = Mock(return_value={
            "python": GrammarInfo(name="python", path=Path("/grammars/python")),
            "rust": GrammarInfo(name="rust", path=Path("/grammars/rust"))
        })

        result2 = await integrator.sync_grammar_availability()
        assert "rust" in result2["newly_available"]
        assert "python" not in result2["newly_available"]  # Already available

        # Verify both are now available
        has_python = await integrator.has_grammar_for_language("python")
        has_rust = await integrator.has_grammar_for_language("rust")

        assert has_python is True
        assert has_rust is True


class TestGrammarLanguageIntegrationEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_database(self, state_manager):
        """Test integration with empty database."""
        integrator = GrammarLanguageIntegrator(state_manager)

        stats = await integrator.get_grammar_stats()

        assert stats["total_languages"] == 0
        assert stats["grammar_coverage"] == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_access(self, integrator, sample_languages):
        """Test concurrent grammar sync operations."""
        import asyncio

        integrator.grammar_discovery.discover_grammars = Mock(return_value={
            "python": GrammarInfo(name="python", path=Path("/grammars/python"))
        })

        # Run multiple syncs concurrently
        results = await asyncio.gather(
            integrator.sync_grammar_availability(),
            integrator.sync_grammar_availability(),
            integrator.sync_grammar_availability()
        )

        # All should complete successfully
        assert len(results) == 3
        for result in results:
            assert result["grammars_found"] == 1

    @pytest.mark.asyncio
    async def test_custom_config_path(self, state_manager):
        """Test initialization with custom config path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # ConfigManager treats the path as a directory, not a file
            config_dir = Path(tmpdir) / "config"

            integrator = GrammarLanguageIntegrator(
                state_manager,
                config_path=config_dir
            )

            assert integrator.config_manager is not None
            assert integrator.config_manager.config_dir == config_dir
