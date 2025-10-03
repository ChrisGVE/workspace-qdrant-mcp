"""
Unit tests for grammar discovery module.

Tests the GrammarDiscovery class and related functionality for
discovering and managing tree-sitter grammars.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from common.core.grammar_discovery import GrammarDiscovery, GrammarInfo


class TestGrammarInfo:
    """Test GrammarInfo dataclass."""

    def test_grammar_info_creation(self):
        """Test creating a GrammarInfo object."""
        grammar = GrammarInfo(
            name="python",
            path=Path("/path/to/grammar"),
            parser_path=Path("/path/to/parser.so"),
            version="1.0.0",
            has_external_scanner=True
        )

        assert grammar.name == "python"
        assert grammar.path == Path("/path/to/grammar")
        assert grammar.parser_path == Path("/path/to/parser.so")
        assert grammar.version == "1.0.0"
        assert grammar.has_external_scanner is True

    def test_grammar_info_to_dict(self):
        """Test converting GrammarInfo to dictionary."""
        grammar = GrammarInfo(
            name="javascript",
            path=Path("/grammars/js"),
            parser_path=Path("/grammars/js/parser.so")
        )

        result = grammar.to_dict()

        assert result["name"] == "javascript"
        assert result["path"] == "/grammars/js"
        assert result["parser_path"] == "/grammars/js/parser.so"
        assert result["version"] is None
        assert result["has_external_scanner"] is False


class TestGrammarDiscovery:
    """Test GrammarDiscovery class."""

    def test_initialization(self):
        """Test initializing GrammarDiscovery."""
        discovery = GrammarDiscovery()
        assert discovery.tree_sitter_cli == "tree-sitter"
        assert discovery._grammars_cache is None

        discovery_custom = GrammarDiscovery("/custom/path/tree-sitter")
        assert discovery_custom.tree_sitter_cli == "/custom/path/tree-sitter"

    @patch("subprocess.run")
    def test_discover_grammars_success(self, mock_run):
        """Test successful grammar discovery."""
        # Mock tree-sitter dump-languages output
        mock_output = json.dumps({
            "python": "/home/user/.tree-sitter/grammars/python",
            "javascript": "/home/user/.tree-sitter/grammars/javascript"
        })

        mock_run.return_value = Mock(
            stdout=mock_output,
            returncode=0
        )

        discovery = GrammarDiscovery()

        with patch.object(discovery, '_find_parser_library', return_value=None), \
             patch.object(discovery, '_has_external_scanner', return_value=False):

            grammars = discovery.discover_grammars()

            assert len(grammars) == 2
            assert "python" in grammars
            assert "javascript" in grammars
            assert grammars["python"].name == "python"
            assert str(grammars["python"].path).endswith("python")

    @patch("subprocess.run")
    def test_discover_grammars_cli_not_found(self, mock_run):
        """Test grammar discovery when tree-sitter CLI is not found."""
        mock_run.side_effect = FileNotFoundError()

        discovery = GrammarDiscovery()

        with pytest.raises(RuntimeError, match="tree-sitter CLI not found"):
            discovery.discover_grammars()

    @patch("subprocess.run")
    def test_discover_grammars_timeout(self, mock_run):
        """Test grammar discovery timeout handling."""
        mock_run.side_effect = subprocess.TimeoutExpired("tree-sitter", 10)

        discovery = GrammarDiscovery()

        with pytest.raises(RuntimeError, match="timed out"):
            discovery.discover_grammars()

    @patch("subprocess.run")
    def test_discover_grammars_command_failure(self, mock_run):
        """Test grammar discovery when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "tree-sitter", stderr="Error occurred"
        )

        discovery = GrammarDiscovery()

        with pytest.raises(RuntimeError, match="dump-languages failed"):
            discovery.discover_grammars()

    @patch("subprocess.run")
    def test_discover_grammars_caching(self, mock_run):
        """Test that grammars are cached after first discovery."""
        mock_output = json.dumps({"python": "/path/to/python"})
        mock_run.return_value = Mock(stdout=mock_output, returncode=0)

        discovery = GrammarDiscovery()

        with patch.object(discovery, '_find_parser_library', return_value=None), \
             patch.object(discovery, '_has_external_scanner', return_value=False):

            # First call - should invoke subprocess
            grammars1 = discovery.discover_grammars()
            assert mock_run.call_count == 1

            # Second call - should use cache
            grammars2 = discovery.discover_grammars()
            assert mock_run.call_count == 1  # Still 1, not called again

            assert grammars1 is grammars2  # Same object returned

    @patch("subprocess.run")
    def test_discover_grammars_force_refresh(self, mock_run):
        """Test forcing cache refresh."""
        mock_output = json.dumps({"python": "/path/to/python"})
        mock_run.return_value = Mock(stdout=mock_output, returncode=0)

        discovery = GrammarDiscovery()

        with patch.object(discovery, '_find_parser_library', return_value=None), \
             patch.object(discovery, '_has_external_scanner', return_value=False):

            # First call
            discovery.discover_grammars()
            assert mock_run.call_count == 1

            # Force refresh
            discovery.discover_grammars(force_refresh=True)
            assert mock_run.call_count == 2  # Called again

    def test_parse_dump_languages_output(self):
        """Test parsing dump-languages JSON output."""
        discovery = GrammarDiscovery()

        output = json.dumps({
            "python": "/grammars/python",
            "rust": "/grammars/rust"
        })

        with patch.object(discovery, '_find_parser_library', return_value=Path("/parser.so")), \
             patch.object(discovery, '_has_external_scanner', return_value=True):

            grammars = discovery._parse_dump_languages_output(output)

            assert len(grammars) == 2
            assert grammars["python"].parser_path == Path("/parser.so")
            assert grammars["python"].has_external_scanner is True

    def test_parse_dump_languages_invalid_json(self):
        """Test handling invalid JSON from dump-languages."""
        discovery = GrammarDiscovery()

        invalid_output = "Not valid JSON"

        grammars = discovery._parse_dump_languages_output(invalid_output)

        assert len(grammars) == 0  # Should return empty dict

    def test_find_parser_library(self, tmp_path):
        """Test finding compiled parser library."""
        discovery = GrammarDiscovery()

        # Create test grammar directory structure
        grammar_dir = tmp_path / "grammar"
        build_dir = grammar_dir / "build"
        build_dir.mkdir(parents=True)

        # Create mock parser library
        parser_file = build_dir / "parser.so"
        parser_file.touch()

        result = discovery._find_parser_library(grammar_dir)

        assert result == parser_file

    def test_find_parser_library_not_found(self, tmp_path):
        """Test when parser library is not found."""
        discovery = GrammarDiscovery()

        grammar_dir = tmp_path / "grammar"
        grammar_dir.mkdir()

        result = discovery._find_parser_library(grammar_dir)

        assert result is None

    def test_has_external_scanner(self, tmp_path):
        """Test detecting external scanner."""
        discovery = GrammarDiscovery()

        grammar_dir = tmp_path / "grammar"
        src_dir = grammar_dir / "src"
        src_dir.mkdir(parents=True)

        # Create scanner.c file
        scanner_file = src_dir / "scanner.c"
        scanner_file.touch()

        result = discovery._has_external_scanner(grammar_dir)

        assert result is True

    def test_has_external_scanner_not_found(self, tmp_path):
        """Test when external scanner is not present."""
        discovery = GrammarDiscovery()

        grammar_dir = tmp_path / "grammar"
        grammar_dir.mkdir()

        result = discovery._has_external_scanner(grammar_dir)

        assert result is False

    @patch("subprocess.run")
    def test_get_grammar(self, mock_run):
        """Test retrieving a specific grammar."""
        mock_output = json.dumps({"python": "/path/to/python"})
        mock_run.return_value = Mock(stdout=mock_output, returncode=0)

        discovery = GrammarDiscovery()

        with patch.object(discovery, '_find_parser_library', return_value=None), \
             patch.object(discovery, '_has_external_scanner', return_value=False):

            grammar = discovery.get_grammar("python")

            assert grammar is not None
            assert grammar.name == "python"

            missing_grammar = discovery.get_grammar("nonexistent")
            assert missing_grammar is None

    @patch("subprocess.run")
    def test_list_languages(self, mock_run):
        """Test listing available languages."""
        mock_output = json.dumps({
            "python": "/path/to/python",
            "javascript": "/path/to/javascript",
            "rust": "/path/to/rust"
        })
        mock_run.return_value = Mock(stdout=mock_output, returncode=0)

        discovery = GrammarDiscovery()

        with patch.object(discovery, '_find_parser_library', return_value=None), \
             patch.object(discovery, '_has_external_scanner', return_value=False):

            languages = discovery.list_languages()

            assert languages == ["javascript", "python", "rust"]  # Sorted

    @patch("subprocess.run")
    def test_validate_grammar_success(self, mock_run, tmp_path):
        """Test validating a properly installed grammar."""
        grammar_dir = tmp_path / "grammar"
        grammar_dir.mkdir()

        parser_file = grammar_dir / "parser.so"
        parser_file.touch()

        mock_output = json.dumps({"python": str(grammar_dir)})
        mock_run.return_value = Mock(stdout=mock_output, returncode=0)

        discovery = GrammarDiscovery()

        with patch.object(discovery, '_find_parser_library', return_value=parser_file), \
             patch.object(discovery, '_has_external_scanner', return_value=False):

            is_valid, message = discovery.validate_grammar("python")

            assert is_valid is True
            assert "valid" in message.lower()

    @patch("subprocess.run")
    def test_validate_grammar_not_found(self, mock_run):
        """Test validating a grammar that doesn't exist."""
        mock_output = json.dumps({})
        mock_run.return_value = Mock(stdout=mock_output, returncode=0)

        discovery = GrammarDiscovery()

        is_valid, message = discovery.validate_grammar("nonexistent")

        assert is_valid is False
        assert "not found" in message.lower()

    @patch("subprocess.run")
    def test_validate_grammar_no_parser(self, mock_run, tmp_path):
        """Test validating a grammar without compiled parser."""
        grammar_dir = tmp_path / "grammar"
        grammar_dir.mkdir()

        mock_output = json.dumps({"python": str(grammar_dir)})
        mock_run.return_value = Mock(stdout=mock_output, returncode=0)

        discovery = GrammarDiscovery()

        with patch.object(discovery, '_find_parser_library', return_value=None), \
             patch.object(discovery, '_has_external_scanner', return_value=False):

            is_valid, message = discovery.validate_grammar("python")

            assert is_valid is False
            assert "no compiled parser" in message.lower()

    def test_clear_cache(self):
        """Test clearing the grammars cache."""
        discovery = GrammarDiscovery()
        discovery._grammars_cache = {"python": Mock()}

        assert discovery._grammars_cache is not None

        discovery.clear_cache()

        assert discovery._grammars_cache is None
