"""
Grammar Discovery System for Tree-sitter.

This module provides functionality to discover installed tree-sitter grammars
using the 'tree-sitter dump-languages' command and manage grammar metadata.

Key features:
- Discover installed grammars via tree-sitter CLI
- Parse grammar metadata (name, path, parser version)
- Validate grammar availability
- Cross-platform support (Windows, macOS, Linux)
"""

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GrammarInfo:
    """Information about a discovered tree-sitter grammar."""

    name: str
    """Language name (e.g., 'python', 'javascript')"""

    path: Path
    """Absolute path to grammar directory"""

    parser_path: Path | None = None
    """Path to compiled parser (.so/.dll/.dylib)"""

    version: str | None = None
    """Parser version if available"""

    has_external_scanner: bool = False
    """Whether grammar includes external scanner"""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "path": str(self.path),
            "parser_path": str(self.parser_path) if self.parser_path else None,
            "version": self.version,
            "has_external_scanner": self.has_external_scanner
        }


class GrammarDiscovery:
    """
    Discovers and manages tree-sitter grammar information.

    Uses 'tree-sitter dump-languages' to find installed grammars
    and validates their availability.
    """

    def __init__(self, tree_sitter_cli: str | None = None):
        """
        Initialize grammar discovery.

        Args:
            tree_sitter_cli: Path to tree-sitter CLI executable.
                           If None, searches in PATH.
        """
        self.tree_sitter_cli = tree_sitter_cli or "tree-sitter"
        self._grammars_cache: dict[str, GrammarInfo] | None = None

    def discover_grammars(self, force_refresh: bool = False) -> dict[str, GrammarInfo]:
        """
        Discover all installed tree-sitter grammars.

        Args:
            force_refresh: If True, bypass cache and re-discover grammars

        Returns:
            Dictionary mapping language names to GrammarInfo objects

        Raises:
            RuntimeError: If tree-sitter CLI is not available
            subprocess.CalledProcessError: If command fails
        """
        if not force_refresh and self._grammars_cache is not None:
            return self._grammars_cache

        logger.info("Discovering tree-sitter grammars...")

        try:
            # Run tree-sitter dump-languages command
            result = subprocess.run(
                [self.tree_sitter_cli, "dump-languages"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )

            grammars = self._parse_dump_languages_output(result.stdout)
            self._grammars_cache = grammars

            logger.info(f"Discovered {len(grammars)} tree-sitter grammars")
            return grammars

        except FileNotFoundError:
            raise RuntimeError(
                f"tree-sitter CLI not found at '{self.tree_sitter_cli}'. "
                "Please install tree-sitter CLI: https://tree-sitter.github.io/tree-sitter/#installation"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "tree-sitter dump-languages command timed out after 10 seconds"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"tree-sitter dump-languages failed: {e.stderr}"
            )

    def _parse_dump_languages_output(self, output: str) -> dict[str, GrammarInfo]:
        """
        Parse output from 'tree-sitter dump-languages' command.

        Expected format:
        {
          "python": "/path/to/tree-sitter-python",
          "javascript": "/path/to/tree-sitter-javascript"
        }

        Args:
            output: JSON output from command

        Returns:
            Dictionary of GrammarInfo objects
        """
        grammars = {}

        try:
            languages_data = json.loads(output)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tree-sitter output: {e}")
            return grammars

        for language, grammar_path in languages_data.items():
            try:
                path = Path(grammar_path).resolve()

                # Check for compiled parser
                parser_path = self._find_parser_library(path)

                # Check for external scanner
                has_external_scanner = self._has_external_scanner(path)

                grammars[language] = GrammarInfo(
                    name=language,
                    path=path,
                    parser_path=parser_path,
                    has_external_scanner=has_external_scanner
                )

            except Exception as e:
                logger.warning(f"Failed to process grammar '{language}': {e}")
                continue

        return grammars

    def _find_parser_library(self, grammar_path: Path) -> Path | None:
        """
        Find compiled parser library (.so/.dll/.dylib) for a grammar.

        Args:
            grammar_path: Path to grammar directory

        Returns:
            Path to parser library if found, None otherwise
        """
        # Common locations for compiled parsers
        possible_locations = [
            grammar_path / "build",
            grammar_path / "src",
            grammar_path,
        ]

        # Platform-specific library extensions
        extensions = [".so", ".dll", ".dylib"]

        for location in possible_locations:
            if not location.exists():
                continue

            for ext in extensions:
                parser_files = list(location.glob(f"*{ext}"))
                if parser_files:
                    return parser_files[0]

        return None

    def _has_external_scanner(self, grammar_path: Path) -> bool:
        """
        Check if grammar has external scanner (scanner.c or scanner.cc).

        Args:
            grammar_path: Path to grammar directory

        Returns:
            True if external scanner exists
        """
        src_dir = grammar_path / "src"
        if not src_dir.exists():
            return False

        scanner_files = ["scanner.c", "scanner.cc", "scanner.cpp"]
        return any((src_dir / f).exists() for f in scanner_files)

    def get_grammar(self, language: str) -> GrammarInfo | None:
        """
        Get information about a specific grammar.

        Args:
            language: Language name (e.g., 'python')

        Returns:
            GrammarInfo if found, None otherwise
        """
        grammars = self.discover_grammars()
        return grammars.get(language)

    def list_languages(self) -> list[str]:
        """
        Get list of available language names.

        Returns:
            Sorted list of language names
        """
        grammars = self.discover_grammars()
        return sorted(grammars.keys())

    def validate_grammar(self, language: str) -> tuple[bool, str]:
        """
        Validate that a grammar is properly installed and compiled.

        Args:
            language: Language name to validate

        Returns:
            Tuple of (is_valid, message)
        """
        grammar = self.get_grammar(language)

        if grammar is None:
            return False, f"Grammar '{language}' not found"

        if not grammar.path.exists():
            return False, f"Grammar path does not exist: {grammar.path}"

        if grammar.parser_path is None:
            return False, f"No compiled parser found for '{language}'. Run 'tree-sitter generate' to compile."

        if not grammar.parser_path.exists():
            return False, f"Parser library does not exist: {grammar.parser_path}"

        return True, f"Grammar '{language}' is valid"

    def clear_cache(self):
        """Clear the grammars cache, forcing re-discovery on next access."""
        self._grammars_cache = None


# Export main class and dataclass
__all__ = ["GrammarDiscovery", "GrammarInfo"]
