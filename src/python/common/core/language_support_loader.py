"""Language Support Database Loader with Conflict Resolution.

This module provides database loading functionality for language support configuration
data from YAML files. It handles inserting and updating language definitions in the
SQLite state database with intelligent conflict resolution to preserve user customizations.

Key Features:
    - Loads language definitions from LanguageSupportConfig into database
    - Preserves user-customized LSP and Tree-sitter paths using COALESCE
    - Atomic transactions for data consistency
    - Query methods for retrieving language information
    - Updates lsp_missing and ts_missing flags from configuration

Conflict Resolution Strategy:
    When loading languages that already exist in database:
    - COALESCE preserves existing lsp_absolute_path if not NULL
    - COALESCE preserves existing ts_cli_absolute_path if not NULL
    - Updates lsp_missing and ts_missing from new config
    - Updates updated_at timestamp automatically

Example:
    >>> from pathlib import Path
    >>> from language_support_models import LanguageSupportConfig
    >>> from sqlite_state_manager import SQLiteStateManager
    >>>
    >>> # Initialize state manager and loader
    >>> state_manager = SQLiteStateManager(db_path="./workspace_state.db")
    >>> await state_manager.initialize()
    >>> loader = LanguageSupportLoader(state_manager)
    >>>
    >>> # Load language configuration
    >>> config = LanguageSupportConfig.from_yaml(Path("language_support.yaml"))
    >>> count = await loader.load_languages(config)
    >>> print(f"Loaded {count} languages")
    >>>
    >>> # Query specific language
    >>> python_info = await loader.get_language("python")
    >>> if python_info:
    >>>     print(f"Python LSP: {python_info['lsp_name']}")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from .language_support_models import LanguageSupportConfig
from .sqlite_state_manager import SQLiteStateManager


class LanguageSupportLoader:
    """Database loader for language support configuration.

    This class handles loading language support data from LanguageSupportConfig
    objects into the SQLite state database. It implements intelligent conflict
    resolution to preserve user customizations while updating configuration data.

    Attributes:
        state_manager: SQLiteStateManager instance for database operations
    """

    def __init__(self, state_manager: SQLiteStateManager) -> None:
        """Initialize language support loader.

        Args:
            state_manager: SQLiteStateManager instance for database access
        """
        self.state_manager = state_manager

    async def load_languages(self, config: LanguageSupportConfig) -> int:
        """Load languages from configuration into database with conflict resolution.

        This method processes the LanguageSupportConfig and loads language definitions
        into the languages table. It uses INSERT OR REPLACE with COALESCE to preserve
        user-customized paths while updating other fields from the configuration.

        Conflict Resolution:
        - If lsp_absolute_path exists in DB, it's preserved (user customization)
        - If ts_cli_absolute_path exists in DB, it's preserved (user customization)
        - lsp_missing and ts_missing are always updated from config
        - Other fields are updated from config
        - updated_at timestamp is set to current time

        Args:
            config: LanguageSupportConfig containing language definitions

        Returns:
            Number of languages loaded/updated

        Raises:
            Exception: If database transaction fails

        Example:
            >>> config = LanguageSupportConfig.from_yaml(Path("config.yaml"))
            >>> count = await loader.load_languages(config)
            >>> print(f"Loaded {count} languages")
        """
        # Build comprehensive language info by combining multiple config sections
        language_data: Dict[str, Dict[str, Any]] = {}

        # 1. Collect file extensions for each language
        for ext, lang_name in config.file_extensions.items():
            if lang_name not in language_data:
                language_data[lang_name] = {
                    "language_name": lang_name,
                    "file_extensions": [],
                    "lsp_name": None,
                    "lsp_executable": None,
                    "lsp_missing": True,
                    "ts_grammar": None,
                    "ts_missing": True,
                }
            language_data[lang_name]["file_extensions"].append(ext)

        # 2. Add LSP server information
        for lang_name, lsp_config in config.lsp_servers.items():
            if lang_name not in language_data:
                language_data[lang_name] = {
                    "language_name": lang_name,
                    "file_extensions": [],
                    "lsp_name": None,
                    "lsp_executable": None,
                    "lsp_missing": True,
                    "ts_grammar": None,
                    "ts_missing": True,
                }

            language_data[lang_name]["lsp_name"] = lsp_config.primary
            language_data[lang_name]["lsp_executable"] = lsp_config.primary
            language_data[lang_name]["lsp_missing"] = False

        # 3. Add tree-sitter grammar information
        for lang_name in config.tree_sitter_grammars.available:
            if lang_name not in language_data:
                language_data[lang_name] = {
                    "language_name": lang_name,
                    "file_extensions": [],
                    "lsp_name": None,
                    "lsp_executable": None,
                    "lsp_missing": True,
                    "ts_grammar": None,
                    "ts_missing": True,
                }

            language_data[lang_name]["ts_grammar"] = f"tree-sitter-{lang_name}"
            language_data[lang_name]["ts_missing"] = False

        # 4. Insert/update languages in database with transaction
        loaded_count = 0

        def _load_languages_sync() -> int:
            """Synchronous database operation within transaction."""
            count = 0

            with self.state_manager._lock:
                # Start transaction
                cursor = self.state_manager.connection.cursor()

                try:
                    for lang_name, lang_info in language_data.items():
                        # Serialize file extensions as JSON array
                        file_extensions_json = json.dumps(lang_info["file_extensions"])

                        # Use INSERT OR REPLACE with COALESCE to preserve user paths
                        # COALESCE(new_value, old_value) returns new_value if not NULL, else old_value
                        cursor.execute("""
                            INSERT INTO languages (
                                language_name,
                                file_extensions,
                                lsp_name,
                                lsp_executable,
                                lsp_absolute_path,
                                lsp_missing,
                                ts_grammar,
                                ts_cli_absolute_path,
                                ts_missing,
                                updated_at
                            )
                            VALUES (?, ?, ?, ?, NULL, ?, ?, NULL, ?, CURRENT_TIMESTAMP)
                            ON CONFLICT(language_name) DO UPDATE SET
                                file_extensions = excluded.file_extensions,
                                lsp_name = excluded.lsp_name,
                                lsp_executable = excluded.lsp_executable,
                                lsp_absolute_path = COALESCE(languages.lsp_absolute_path, excluded.lsp_absolute_path),
                                lsp_missing = excluded.lsp_missing,
                                ts_grammar = excluded.ts_grammar,
                                ts_cli_absolute_path = COALESCE(languages.ts_cli_absolute_path, excluded.ts_cli_absolute_path),
                                ts_missing = excluded.ts_missing,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            lang_name,
                            file_extensions_json,
                            lang_info["lsp_name"],
                            lang_info["lsp_executable"],
                            1 if lang_info["lsp_missing"] else 0,
                            lang_info["ts_grammar"],
                            1 if lang_info["ts_missing"] else 0,
                        ))

                        count += 1

                    # Commit transaction
                    self.state_manager.connection.commit()

                    logger.info(
                        f"Successfully loaded {count} languages into database"
                    )

                    return count

                except Exception as e:
                    # Rollback on error
                    self.state_manager.connection.rollback()
                    logger.error(
                        f"Failed to load languages into database: {e}"
                    )
                    raise

        # Execute synchronous operation
        loaded_count = _load_languages_sync()

        return loaded_count

    async def get_language(self, language_name: str) -> Optional[Dict[str, Any]]:
        """Query language information by name.

        Retrieves complete language information including LSP server details,
        tree-sitter grammar, file extensions, and user customizations.

        Args:
            language_name: Name of the language to query

        Returns:
            Dictionary containing language information, or None if not found.
            Dictionary keys:
                - id: Database record ID
                - language_name: Language identifier
                - file_extensions: JSON array of file extensions
                - lsp_name: LSP server name
                - lsp_executable: LSP server executable
                - lsp_absolute_path: User-customized LSP path (may be NULL)
                - lsp_missing: Boolean indicating if LSP is missing
                - ts_grammar: Tree-sitter grammar name
                - ts_cli_absolute_path: User-customized tree-sitter CLI path (may be NULL)
                - ts_missing: Boolean indicating if tree-sitter is missing
                - created_at: Record creation timestamp
                - updated_at: Record last update timestamp

        Example:
            >>> info = await loader.get_language("python")
            >>> if info:
            >>>     print(f"Python uses {info['lsp_name']} LSP server")
            >>>     if info['lsp_absolute_path']:
            >>>         print(f"Custom path: {info['lsp_absolute_path']}")
        """
        def _get_language_sync() -> Optional[Dict[str, Any]]:
            """Synchronous database query."""
            with self.state_manager._lock:
                cursor = self.state_manager.connection.cursor()
                cursor.execute("""
                    SELECT
                        id,
                        language_name,
                        file_extensions,
                        lsp_name,
                        lsp_executable,
                        lsp_absolute_path,
                        lsp_missing,
                        ts_grammar,
                        ts_cli_absolute_path,
                        ts_missing,
                        created_at,
                        updated_at
                    FROM languages
                    WHERE language_name = ?
                """, (language_name,))

                row = cursor.fetchone()

                if row is None:
                    return None

                # Convert row to dictionary
                return {
                    "id": row[0],
                    "language_name": row[1],
                    "file_extensions": row[2],
                    "lsp_name": row[3],
                    "lsp_executable": row[4],
                    "lsp_absolute_path": row[5],
                    "lsp_missing": bool(row[6]),
                    "ts_grammar": row[7],
                    "ts_cli_absolute_path": row[8],
                    "ts_missing": bool(row[9]),
                    "created_at": row[10],
                    "updated_at": row[11],
                }

        return _get_language_sync()

    async def list_languages(self) -> List[Dict[str, Any]]:
        """List all languages in database.

        Retrieves all language records ordered by language name.
        Useful for displaying available languages and their configuration status.

        Returns:
            List of dictionaries, each containing complete language information.
            Each dictionary has the same structure as get_language() return value.
            List is ordered alphabetically by language_name.

        Example:
            >>> languages = await loader.list_languages()
            >>> for lang in languages:
            >>>     status = "missing LSP" if lang['lsp_missing'] else "has LSP"
            >>>     print(f"{lang['language_name']}: {status}")
        """
        def _list_languages_sync() -> List[Dict[str, Any]]:
            """Synchronous database query."""
            with self.state_manager._lock:
                cursor = self.state_manager.connection.cursor()
                cursor.execute("""
                    SELECT
                        id,
                        language_name,
                        file_extensions,
                        lsp_name,
                        lsp_executable,
                        lsp_absolute_path,
                        lsp_missing,
                        ts_grammar,
                        ts_cli_absolute_path,
                        ts_missing,
                        created_at,
                        updated_at
                    FROM languages
                    ORDER BY language_name
                """)

                rows = cursor.fetchall()

                # Convert rows to list of dictionaries
                results = []
                for row in rows:
                    results.append({
                        "id": row[0],
                        "language_name": row[1],
                        "file_extensions": row[2],
                        "lsp_name": row[3],
                        "lsp_executable": row[4],
                        "lsp_absolute_path": row[5],
                        "lsp_missing": bool(row[6]),
                        "ts_grammar": row[7],
                        "ts_cli_absolute_path": row[8],
                        "ts_missing": bool(row[9]),
                        "created_at": row[10],
                        "updated_at": row[11],
                    })

                return results

        return _list_languages_sync()
