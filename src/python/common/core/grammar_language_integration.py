"""
Grammar-Language Support Integration Module.

This module bridges the tree-sitter grammar management system with the
language support system, enabling automatic grammar loading and metadata
extraction for detected languages.

Key Features:
- Automatic grammar discovery and registration
- Language-to-grammar mapping
- Grammar availability tracking in language support database
- Integration with LSP and document processing pipelines
- Fallback handling for missing grammars
- Progress tracking for long-running operations

Architecture:
    GrammarLanguageIntegrator coordinates between:
    - GrammarDiscovery: Discovers installed grammars
    - LanguageSupportManager: Manages language metadata
    - SQLiteStateManager: Persists grammar-language mappings

    The integrator ensures that when a file is processed:
    1. Language is detected via file extension
    2. Grammar availability is checked
    3. Grammar is loaded if available
    4. Metadata extraction uses grammar for code analysis

Example:
    ```python
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager
    from workspace_qdrant_mcp.core.grammar_language_integration import GrammarLanguageIntegrator

    # Initialize
    state_manager = SQLiteStateManager()
    await state_manager.initialize()

    integrator = GrammarLanguageIntegrator(state_manager)

    # Sync grammar availability with language support
    stats = await integrator.sync_grammar_availability()
    print(f"Updated {stats['grammars_synced']} languages with grammar info")

    # Get grammar for a language
    grammar = await integrator.get_grammar_for_language("python")
    if grammar:
        print(f"Python grammar: {grammar.name} at {grammar.path}")

    # Check if language has grammar available
    has_grammar = await integrator.has_grammar_for_language("rust")
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .grammar_compiler import CompilerDetector, GrammarCompiler
from .grammar_config import ConfigManager
from .grammar_discovery import GrammarDiscovery, GrammarInfo
from .language_support_manager import LanguageSupportManager
from .sqlite_state_manager import SQLiteStateManager

if TYPE_CHECKING:
    from rich.progress import Progress, TaskID


class GrammarLanguageIntegrator:
    """
    Integrates tree-sitter grammar management with language support system.

    This class coordinates between grammar discovery, compilation, and the
    language support database to ensure grammars are available for detected
    languages and properly integrated into the document processing pipeline.

    Attributes:
        state_manager: SQLiteStateManager instance for database operations
        language_support: LanguageSupportManager instance
        grammar_discovery: GrammarDiscovery instance for finding grammars
        config_manager: ConfigManager for grammar configuration
    """

    def __init__(
        self,
        state_manager: SQLiteStateManager,
        tree_sitter_cli: str | None = None,
        config_path: Path | None = None
    ):
        """
        Initialize grammar-language integrator.

        Args:
            state_manager: Initialized SQLiteStateManager instance
            tree_sitter_cli: Path to tree-sitter CLI (None to search PATH)
            config_path: Path to grammar configuration file
        """
        self.state_manager = state_manager
        self.language_support = LanguageSupportManager(state_manager)
        self.grammar_discovery = GrammarDiscovery(tree_sitter_cli)

        # Initialize config manager
        if config_path:
            self.config_manager = ConfigManager(config_path)
        else:
            self.config_manager = ConfigManager()

        logger.debug("Initialized GrammarLanguageIntegrator")

    async def sync_grammar_availability(
        self,
        force_refresh: bool = False,
        progress: Progress | None = None,
        progress_task: TaskID | None = None
    ) -> dict[str, any]:
        """
        Synchronize grammar availability with language support database.

        This method:
        1. Discovers all installed tree-sitter grammars
        2. Updates language support database with grammar availability
        3. Updates ts_cli_absolute_path for available grammars
        4. Sets ts_missing flag appropriately

        Args:
            force_refresh: Force re-discovery of grammars
            progress: Optional Rich Progress instance for progress tracking
            progress_task: Optional progress task ID to update

        Returns:
            Dictionary containing:
                - grammars_found: Number of grammars discovered
                - grammars_synced: Number of languages updated
                - newly_available: List of newly available languages
                - still_missing: List of languages still missing grammars
        """
        try:
            logger.info("Synchronizing grammar availability with language support...")

            # Update progress
            if progress and progress_task is not None:
                progress.update(progress_task, description="Discovering grammars...")

            # Discover installed grammars
            grammars = self.grammar_discovery.discover_grammars(force_refresh)
            logger.debug(f"Found {len(grammars)} installed grammars")

            # Update progress
            if progress and progress_task is not None:
                progress.update(progress_task, description="Loading languages...")

            # Get all languages from database
            languages = await self.language_support.get_supported_languages()

            newly_available = []
            still_missing = []
            grammars_synced = 0

            # Update progress with total count
            if progress and progress_task is not None:
                progress.update(
                    progress_task,
                    description=f"Syncing {len(languages)} languages...",
                    total=len(languages),
                    completed=0
                )

            # Update each language's grammar availability
            async with self.state_manager.transaction() as conn:
                for idx, lang in enumerate(languages):
                    lang_name = lang["language_name"]

                    # Update progress
                    if progress and progress_task is not None:
                        progress.update(
                            progress_task,
                            description=f"Syncing {lang_name}...",
                            completed=idx
                        )

                    # Check if we have a grammar for this language
                    grammar = grammars.get(lang_name)

                    if grammar:
                        # Grammar available - update database
                        ts_cli_path = str(self.grammar_discovery.tree_sitter_cli)

                        conn.execute(
                            """
                            UPDATE languages
                            SET ts_cli_absolute_path = ?,
                                ts_missing = 0,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE language_name = ?
                            """,
                            (ts_cli_path, lang_name)
                        )

                        if lang.get("ts_available") is False:
                            newly_available.append(lang_name)

                        grammars_synced += 1
                        logger.debug(f"Updated grammar availability for {lang_name}")
                    else:
                        # Grammar not available
                        if lang.get("ts_available") is True:
                            # Was available, now missing
                            conn.execute(
                                """
                                UPDATE languages
                                SET ts_missing = 1,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE language_name = ?
                                """,
                                (lang_name,)
                            )

                        still_missing.append(lang_name)

            # Update progress with completion
            if progress and progress_task is not None:
                progress.update(
                    progress_task,
                    description=f"✓ Synced {grammars_synced} languages",
                    completed=len(languages)
                )

            result = {
                "grammars_found": len(grammars),
                "grammars_synced": grammars_synced,
                "newly_available": newly_available,
                "still_missing": still_missing
            }

            logger.info(
                f"Grammar sync complete: {grammars_synced} synced, "
                f"{len(newly_available)} newly available, "
                f"{len(still_missing)} still missing"
            )

            return result

        except Exception as e:
            logger.error(f"Failed to sync grammar availability: {e}")
            raise

    async def get_grammar_for_language(self, language_name: str) -> GrammarInfo | None:
        """
        Get grammar information for a specific language.

        Args:
            language_name: Programming language name

        Returns:
            GrammarInfo if available, None otherwise
        """
        try:
            # Check language support database first
            lang_info = await self._get_language_info(language_name)

            if not lang_info or lang_info.get("ts_missing"):
                logger.debug(f"No grammar available for {language_name}")
                return None

            # Get grammar from discovery
            grammar = self.grammar_discovery.get_grammar(language_name)

            if grammar:
                logger.debug(f"Found grammar for {language_name}: {grammar.path}")

            return grammar

        except Exception as e:
            logger.error(f"Error getting grammar for {language_name}: {e}")
            return None

    async def get_grammar_for_file(self, file_path: Path) -> GrammarInfo | None:
        """
        Get grammar for a specific file based on its language.

        Args:
            file_path: Path to the file

        Returns:
            GrammarInfo if available, None otherwise
        """
        try:
            # Detect language from file extension
            lang_info = await self.language_support.get_language_for_file(file_path)

            if not lang_info:
                logger.debug(f"No language detected for {file_path}")
                return None

            language_name = lang_info["language_name"]

            # Get grammar for detected language
            return await self.get_grammar_for_language(language_name)

        except Exception as e:
            logger.error(f"Error getting grammar for file {file_path}: {e}")
            return None

    async def has_grammar_for_language(self, language_name: str) -> bool:
        """
        Check if grammar is available for a language.

        Args:
            language_name: Programming language name

        Returns:
            True if grammar is available and compiled
        """
        try:
            lang_info = await self._get_language_info(language_name)

            if not lang_info:
                return False

            return lang_info.get("ts_available", False)

        except Exception as e:
            logger.error(f"Error checking grammar availability for {language_name}: {e}")
            return False

    async def ensure_grammar_compiled(self, language_name: str) -> tuple[bool, str]:
        """
        Ensure grammar is compiled and ready for use.

        If grammar exists but isn't compiled, attempts to compile it.

        Args:
            language_name: Programming language name

        Returns:
            Tuple of (success, message)
        """
        try:
            # Get grammar info
            grammar = await self.get_grammar_for_language(language_name)

            if not grammar:
                return False, f"Grammar not installed for {language_name}"

            # Check if already compiled
            if grammar.parser_path and grammar.parser_path.exists():
                logger.debug(f"Grammar for {language_name} already compiled")
                return True, f"Grammar already compiled: {grammar.parser_path}"

            # Attempt compilation
            logger.info(f"Compiling grammar for {language_name}...")

            compiler_detector = CompilerDetector()
            compiler = GrammarCompiler(compiler_detector)

            result = compiler.compile(grammar.path)

            if result.success:
                # Update grammar discovery cache
                self.grammar_discovery.clear_cache()

                # Sync availability
                await self.sync_grammar_availability(force_refresh=True)

                return True, f"Successfully compiled grammar: {result.output_path}"
            else:
                return False, f"Compilation failed: {result.error_message}"

        except Exception as e:
            logger.error(f"Error ensuring grammar compilation for {language_name}: {e}")
            return False, f"Compilation error: {e}"

    async def get_languages_with_grammars(self) -> list[str]:
        """
        Get list of all languages with available grammars.

        Returns:
            List of language names that have grammars
        """
        try:
            languages = await self.language_support.get_supported_languages()

            return [
                lang["language_name"]
                for lang in languages
                if lang.get("ts_available", False)
            ]

        except Exception as e:
            logger.error(f"Error getting languages with grammars: {e}")
            return []

    async def get_languages_missing_grammars(self) -> list[str]:
        """
        Get list of languages missing grammars.

        Returns:
            List of language names without grammars
        """
        try:
            languages = await self.language_support.get_supported_languages()

            return [
                lang["language_name"]
                for lang in languages
                if not lang.get("ts_available", False)
            ]

        except Exception as e:
            logger.error(f"Error getting languages missing grammars: {e}")
            return []

    async def get_grammar_stats(self) -> dict[str, any]:
        """
        Get statistics about grammar availability across languages.

        Returns:
            Dictionary containing:
                - total_languages: Total number of supported languages
                - with_grammars: Number of languages with grammars
                - missing_grammars: Number of languages without grammars
                - grammar_coverage: Percentage of languages with grammars
                - languages_with_grammars: List of language names
                - languages_missing_grammars: List of language names
        """
        try:
            languages = await self.language_support.get_supported_languages()

            total = len(languages)
            with_grammars = sum(1 for lang in languages if lang.get("ts_available", False))
            missing_grammars = total - with_grammars
            coverage = (with_grammars / total * 100) if total > 0 else 0

            return {
                "total_languages": total,
                "with_grammars": with_grammars,
                "missing_grammars": missing_grammars,
                "grammar_coverage": round(coverage, 2),
                "languages_with_grammars": await self.get_languages_with_grammars(),
                "languages_missing_grammars": await self.get_languages_missing_grammars()
            }

        except Exception as e:
            logger.error(f"Error getting grammar stats: {e}")
            return {
                "total_languages": 0,
                "with_grammars": 0,
                "missing_grammars": 0,
                "grammar_coverage": 0.0,
                "languages_with_grammars": [],
                "languages_missing_grammars": []
            }

    async def _get_language_info(self, language_name: str) -> dict | None:
        """
        Internal helper to get language information from database.

        Args:
            language_name: Programming language name

        Returns:
            Language info dictionary or None
        """
        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT
                        language_name,
                        lsp_name,
                        lsp_missing,
                        ts_grammar,
                        ts_cli_absolute_path,
                        ts_missing
                    FROM languages
                    WHERE language_name = ?
                    """,
                    (language_name,)
                )

                row = cursor.fetchone()

            if not row:
                return None

            return {
                "language_name": row["language_name"],
                "lsp_name": row["lsp_name"],
                "lsp_available": not bool(row["lsp_missing"]),
                "ts_grammar": row["ts_grammar"],
                "ts_cli_absolute_path": row["ts_cli_absolute_path"],
                "ts_available": not bool(row["ts_missing"])
            }

        except Exception as e:
            logger.error(f"Error getting language info for {language_name}: {e}")
            return None


# Export main class
__all__ = ["GrammarLanguageIntegrator"]
