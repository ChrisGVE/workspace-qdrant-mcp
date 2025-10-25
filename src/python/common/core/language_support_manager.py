"""
Language Support Manager for orchestrating YAML parsing and database loading.

This module provides the LanguageSupportManager class that orchestrates:
- YAML parsing with validation (via LanguageSupportParser)
- Database loading with conflict resolution (via LanguageSupportLoader)
- Version tracking and change detection
- Language detection for files

Key Features:
    - Automatic YAML change detection via hash comparison
    - Orchestrates parser and loader components
    - File extension-based language detection
    - Missing metadata tracking for LSP and Tree-sitter
    - Comprehensive error handling and logging

Example:
    ```python
    from pathlib import Path
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager
    from workspace_qdrant_mcp.core.language_support_manager import LanguageSupportManager

    # Initialize state manager and language support manager
    state_manager = SQLiteStateManager()
    await state_manager.initialize()

    manager = LanguageSupportManager(state_manager)

    # Initialize from YAML (only loads if changed)
    yaml_path = Path("assets/languages_support.yaml")
    summary = await manager.initialize_from_yaml(yaml_path)
    print(f"Loaded {summary['languages_loaded']} languages, version: {summary['version']}")

    # Detect language for a file
    lang_info = await manager.get_language_for_file(Path("script.py"))
    if lang_info:
        print(f"Language: {lang_info['language_name']}, LSP: {lang_info['lsp_name']}")

    # Mark file as missing LSP metadata
    await manager.mark_missing_metadata(
        Path("/path/to/file.py"),
        "python",
        missing_lsp=True,
        missing_ts=False
    )
    ```
"""

import hashlib
import json
from pathlib import Path

from loguru import logger

from .language_support_loader import LanguageSupportLoader
from .language_support_parser import LanguageSupportParser
from .sqlite_state_manager import SQLiteStateManager


class LanguageSupportManager:
    """
    Orchestrates language support YAML parsing, version tracking, and database loading.

    This manager coordinates between:
    - LanguageSupportParser: Parses and validates YAML structure
    - LanguageSupportLoader: Loads language data into SQLite database
    - Version tracking: Detects YAML changes via hash comparison

    Attributes:
        state_manager: SQLiteStateManager instance for database operations
    """

    def __init__(self, state_manager: SQLiteStateManager):
        """
        Initialize LanguageSupportManager with state manager.

        Args:
            state_manager: Initialized SQLiteStateManager instance
        """
        self.state_manager = state_manager
        logger.debug("Initialized LanguageSupportManager")

    async def check_for_updates(self, yaml_path: Path) -> bool:
        """
        Check if YAML file has changed since last load.

        Computes SHA-256 hash of YAML file and compares with stored hash
        in language_support_version table.

        Args:
            yaml_path: Path to language_support.yaml file

        Returns:
            True if YAML has changed or not yet loaded, False if unchanged

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            IOError: If unable to read YAML file
        """
        if not yaml_path.exists():
            raise FileNotFoundError(f"Language support YAML not found: {yaml_path}")

        try:
            # Calculate current YAML hash
            with open(yaml_path, 'rb') as f:
                yaml_content = f.read()
                current_hash = hashlib.sha256(yaml_content).hexdigest()

            # Query database for last loaded hash
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT yaml_hash FROM language_support_version
                    ORDER BY loaded_at DESC LIMIT 1
                    """
                )
                row = cursor.fetchone()

            if not row:
                logger.info("No previous language support version found, update needed")
                return True

            stored_hash = row["yaml_hash"]

            if current_hash != stored_hash:
                logger.info(
                    f"Language support YAML changed: {stored_hash[:8]}... -> {current_hash[:8]}..."
                )
                return True
            else:
                logger.debug("Language support YAML unchanged")
                return False

        except Exception as e:
            logger.error(f"Error checking for language support updates: {e}")
            raise

    async def initialize_from_yaml(
        self,
        yaml_path: Path,
        force: bool = False
    ) -> dict[str, any]:
        """
        Initialize language support from YAML file.

        This method:
        1. Checks if YAML has changed (unless force=True)
        2. Parses YAML using LanguageSupportParser
        3. Loads languages into database using LanguageSupportLoader
        4. Updates version tracking with new hash

        Args:
            yaml_path: Path to language_support.yaml file
            force: If True, reload even if YAML hasn't changed

        Returns:
            Dictionary containing:
                - languages_loaded: Number of languages loaded
                - version: Hash of loaded YAML version
                - skipped: True if skipped due to no changes

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            ValueError: If YAML validation fails
            RuntimeError: If database operations fail
        """
        try:
            # Check if update needed (unless forced)
            if not force:
                needs_update = await self.check_for_updates(yaml_path)
                if not needs_update:
                    logger.info("Language support already up to date, skipping load")
                    return {
                        "languages_loaded": 0,
                        "version": "unchanged",
                        "skipped": True
                    }

            logger.info(f"Loading language support from {yaml_path}")

            # Parse YAML using LanguageSupportParser
            parser = LanguageSupportParser()
            config = parser.parse_yaml(yaml_path)
            logger.debug(f"Parsed YAML configuration with {len(config.file_extensions)} file extensions")

            # Load languages into database using LanguageSupportLoader
            loader = LanguageSupportLoader(self.state_manager)
            languages_loaded = await loader.load_languages(config)
            logger.debug(f"Loaded {languages_loaded} languages into database")

            # Calculate YAML hash for version tracking
            with open(yaml_path, 'rb') as f:
                yaml_content = f.read()
                yaml_hash = hashlib.sha256(yaml_content).hexdigest()

            # Update version tracking in database
            async with self.state_manager.transaction() as conn:
                # Update last_checked_at for existing version if it matches
                cursor = conn.execute(
                    """
                    UPDATE language_support_version
                    SET last_checked_at = CURRENT_TIMESTAMP
                    WHERE yaml_hash = ?
                    """,
                    (yaml_hash,)
                )

                # If no existing version with this hash, insert new record
                if cursor.rowcount == 0:
                    conn.execute(
                        """
                        INSERT INTO language_support_version
                        (yaml_hash, language_count, last_checked_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                        """,
                        (yaml_hash, languages_loaded)
                    )
                    logger.info(
                        f"Created new language support version: {yaml_hash[:8]}... "
                        f"({languages_loaded} languages)"
                    )
                else:
                    logger.debug(f"Updated existing language support version: {yaml_hash[:8]}...")

            logger.info(
                f"Successfully loaded {languages_loaded} languages from {yaml_path.name}"
            )

            return {
                "languages_loaded": languages_loaded,
                "version": yaml_hash[:16],
                "skipped": False
            }

        except FileNotFoundError:
            logger.error(f"Language support YAML file not found: {yaml_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize language support from YAML: {e}")
            raise RuntimeError(f"Language support initialization failed: {e}")

    async def get_language_for_file(self, file_path: Path) -> dict | None:
        """
        Detect programming language for a file based on extension.

        Queries the languages table to find matching language by file extension.

        Args:
            file_path: Path to file to detect language for

        Returns:
            Dictionary with language information if found:
                - language_name: Name of the language
                - file_extensions: List of supported extensions
                - lsp_name: LSP server identifier (if available)
                - lsp_executable: LSP executable name (if available)
                - lsp_missing: True if LSP server not found
                - ts_grammar: Tree-sitter grammar name (if available)
                - ts_missing: True if Tree-sitter CLI not found
            None if no matching language found
        """
        try:
            file_extension = file_path.suffix.lower()

            if not file_extension:
                logger.debug(f"No file extension for {file_path}")
                return None

            # Query languages table for matching extension
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT
                        language_name,
                        file_extensions,
                        lsp_name,
                        lsp_executable,
                        lsp_absolute_path,
                        lsp_missing,
                        ts_grammar,
                        ts_cli_absolute_path,
                        ts_missing
                    FROM languages
                    WHERE file_extensions LIKE ?
                    """,
                    (f'%"{file_extension}"%',)  # JSON array contains extension
                )

                rows = cursor.fetchall()

            if not rows:
                logger.debug(f"No language found for extension {file_extension}")
                return None

            # If multiple matches, take the first one
            # (Could be enhanced with priority/confidence scoring)
            row = rows[0]

            if len(rows) > 1:
                logger.debug(
                    f"Multiple languages found for {file_extension}, using {row['language_name']}"
                )

            # Parse file_extensions from JSON
            try:
                extensions = json.loads(row["file_extensions"]) if row["file_extensions"] else []
            except (json.JSONDecodeError, TypeError):
                extensions = []

            language_info = {
                "language_name": row["language_name"],
                "file_extensions": extensions,
                "lsp_name": row["lsp_name"],
                "lsp_executable": row["lsp_executable"],
                "lsp_absolute_path": row["lsp_absolute_path"],
                "lsp_missing": bool(row["lsp_missing"]),
                "ts_grammar": row["ts_grammar"],
                "ts_cli_absolute_path": row["ts_cli_absolute_path"],
                "ts_missing": bool(row["ts_missing"])
            }

            logger.debug(f"Detected language {language_info['language_name']} for {file_path.name}")
            return language_info

        except Exception as e:
            logger.error(f"Error detecting language for {file_path}: {e}")
            return None

    async def mark_missing_metadata(
        self,
        file_path: Path,
        language_name: str,
        missing_lsp: bool,
        missing_ts: bool
    ) -> bool:
        """
        Mark file as missing LSP or Tree-sitter metadata.

        Inserts or updates files_missing_metadata table to track which files
        need metadata extraction when tools become available.

        Args:
            file_path: Absolute path to the file
            language_name: Programming language name
            missing_lsp: True if LSP metadata extraction failed/unavailable
            missing_ts: True if Tree-sitter metadata extraction failed/unavailable

        Returns:
            True if successfully recorded, False on error
        """
        try:
            file_absolute_path = str(file_path.resolve())

            # Determine current git branch (if in a git repo)
            branch = "main"  # Default
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=file_path.parent,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    branch = result.stdout.strip()
            except Exception:
                pass  # Use default branch

            # Insert or replace record in files_missing_metadata
            async with self.state_manager.transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO files_missing_metadata
                    (file_absolute_path, language_name, branch,
                     missing_lsp_metadata, missing_ts_metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        file_absolute_path,
                        language_name,
                        branch,
                        1 if missing_lsp else 0,
                        1 if missing_ts else 0
                    )
                )

            logger.debug(
                f"Marked {file_path.name} as missing "
                f"{'LSP ' if missing_lsp else ''}"
                f"{'TS ' if missing_ts else ''}metadata"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to mark missing metadata for {file_path}: {e}")
            return False

    async def get_files_missing_metadata(
        self,
        language_name: str | None = None,
        missing_lsp_only: bool = False,
        missing_ts_only: bool = False
    ) -> list[dict]:
        """
        Get list of files missing metadata extraction.

        Args:
            language_name: Filter by specific language (None for all)
            missing_lsp_only: Only return files missing LSP metadata
            missing_ts_only: Only return files missing Tree-sitter metadata

        Returns:
            List of dictionaries containing file information:
                - file_absolute_path: Full path to file
                - language_name: Programming language
                - branch: Git branch
                - missing_lsp_metadata: Boolean
                - missing_ts_metadata: Boolean
        """
        try:
            sql = """
                SELECT
                    file_absolute_path,
                    language_name,
                    branch,
                    missing_lsp_metadata,
                    missing_ts_metadata,
                    created_at,
                    updated_at
                FROM files_missing_metadata
                WHERE 1=1
            """

            params = []

            if language_name:
                sql += " AND language_name = ?"
                params.append(language_name)

            if missing_lsp_only:
                sql += " AND missing_lsp_metadata = 1"

            if missing_ts_only:
                sql += " AND missing_ts_metadata = 1"

            sql += " ORDER BY created_at ASC"

            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(sql, params)
                rows = cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "file_absolute_path": row["file_absolute_path"],
                    "language_name": row["language_name"],
                    "branch": row["branch"],
                    "missing_lsp_metadata": bool(row["missing_lsp_metadata"]),
                    "missing_ts_metadata": bool(row["missing_ts_metadata"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                })

            return results

        except Exception as e:
            logger.error(f"Failed to get files missing metadata: {e}")
            return []

    async def get_supported_languages(self) -> list[dict]:
        """
        Get list of all supported languages from database.

        Returns:
            List of dictionaries containing language information
        """
        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT
                        language_name,
                        file_extensions,
                        lsp_name,
                        lsp_missing,
                        ts_grammar,
                        ts_missing
                    FROM languages
                    ORDER BY language_name
                    """
                )
                rows = cursor.fetchall()

            results = []
            for row in rows:
                try:
                    extensions = json.loads(row["file_extensions"]) if row["file_extensions"] else []
                except (json.JSONDecodeError, TypeError):
                    extensions = []

                results.append({
                    "language_name": row["language_name"],
                    "file_extensions": extensions,
                    "lsp_name": row["lsp_name"],
                    "lsp_available": not bool(row["lsp_missing"]),
                    "ts_grammar": row["ts_grammar"],
                    "ts_available": not bool(row["ts_missing"])
                })

            return results

        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return []

    async def get_version_info(self) -> dict | None:
        """
        Get current language support version information.

        Returns:
            Dictionary containing version info or None if not initialized
        """
        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT yaml_hash, loaded_at, language_count, last_checked_at
                    FROM language_support_version
                    ORDER BY loaded_at DESC
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()

            if not row:
                return None

            return {
                "yaml_hash": row["yaml_hash"],
                "yaml_hash_short": row["yaml_hash"][:16],
                "loaded_at": row["loaded_at"],
                "language_count": row["language_count"],
                "last_checked_at": row["last_checked_at"]
            }

        except Exception as e:
            logger.error(f"Failed to get version info: {e}")
            return None
