"""
Missing Metadata Tracker for Language Support System.

This module provides tracking functionality for files that cannot be processed due to
missing LSP servers or Tree-sitter grammars. It integrates with SQLiteStateManager
to persist tracking information and supports branch-aware filtering.

Key Features:
    - Track files missing LSP metadata
    - Track files missing Tree-sitter metadata
    - Branch-aware tracking for multi-branch workflows
    - Query and filter tracked files by language, metadata type, and branch
    - Automatic duplicate prevention via UPSERT operations
    - Comprehensive statistics collection
    - Tool availability detection for LSP and tree-sitter

Example:
    ```python
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager
    from workspace_qdrant_mcp.core.missing_metadata_tracker import MissingMetadataTracker

    # Initialize state manager
    state_manager = SQLiteStateManager()
    await state_manager.initialize()

    # Create tracker
    tracker = MissingMetadataTracker(state_manager)

    # Track a file with missing metadata
    await tracker.track_missing_metadata(
        file_path="/path/to/file.py",
        language_name="python",
        branch="main",
        missing_lsp=True,
        missing_ts=False
    )

    # Query tracked files
    files = await tracker.get_files_missing_metadata(
        language="python",
        missing_lsp=True,
        branch="main"
    )

    # Check tool availability
    lsp_status = await tracker.check_lsp_available("python")
    if lsp_status["available"]:
        print(f"LSP path: {lsp_status['path']}")

    # Get statistics
    stats = await tracker.get_tracked_file_count()
    print(f"Total tracked files: {stats['total']}")
    ```
"""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .sqlite_state_manager import SQLiteStateManager


class MissingMetadataTracker:
    """
    Tracker for files missing LSP or Tree-sitter metadata.

    This class provides a high-level interface for tracking files that cannot be
    fully processed due to missing language support tools (LSP servers or Tree-sitter
    grammars). It uses the SQLiteStateManager's files_missing_metadata table for
    persistent storage and supports branch-aware tracking.

    Attributes:
        state_manager: SQLiteStateManager instance for database operations
    """

    def __init__(self, state_manager: SQLiteStateManager):
        """
        Initialize the missing metadata tracker.

        Args:
            state_manager: Initialized SQLiteStateManager instance for database access
        """
        self.state_manager = state_manager
        self._cleanup_task: Optional[asyncio.Task] = None

    async def track_missing_metadata(
        self,
        file_path: str,
        language_name: str,
        branch: str,
        missing_lsp: bool = False,
        missing_ts: bool = False,
    ) -> bool:
        """
        Track a file with missing metadata.

        Records a file that cannot be fully processed due to missing LSP server or
        Tree-sitter grammar. Uses INSERT OR REPLACE to prevent duplicates and update
        existing entries. Ensures the language exists in the languages table before
        tracking the file.

        Args:
            file_path: Absolute path to the file
            language_name: Programming language identifier (e.g., "python", "rust")
            branch: Git branch name for branch-aware tracking
            missing_lsp: True if LSP metadata is missing
            missing_ts: True if Tree-sitter metadata is missing

        Returns:
            True if the tracking operation succeeded, False otherwise

        Example:
            ```python
            success = await tracker.track_missing_metadata(
                file_path="/home/user/project/main.rs",
                language_name="rust",
                branch="feature/new-api",
                missing_lsp=True,
                missing_ts=False
            )
            ```
        """
        try:
            # Normalize file path to absolute path
            file_absolute_path = str(Path(file_path).resolve())

            async with self.state_manager.transaction() as conn:
                # Ensure language exists in languages table (satisfy foreign key constraint)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO languages (language_name)
                    VALUES (?)
                    """,
                    (language_name,),
                )

                # Now insert or replace the file tracking record
                conn.execute(
                    """
                    INSERT OR REPLACE INTO files_missing_metadata
                    (file_absolute_path, language_name, branch,
                     missing_lsp_metadata, missing_ts_metadata,
                     updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        file_absolute_path,
                        language_name,
                        branch,
                        1 if missing_lsp else 0,
                        1 if missing_ts else 0,
                    ),
                )

            logger.debug(
                f"Tracked file with missing metadata: {file_absolute_path} "
                f"(language={language_name}, branch={branch}, "
                f"missing_lsp={missing_lsp}, missing_ts={missing_ts})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to track missing metadata for {file_path}: {e}")
            return False

    async def get_files_missing_metadata(
        self,
        language: Optional[str] = None,
        missing_lsp: Optional[bool] = None,
        missing_ts: Optional[bool] = None,
        branch: Optional[str] = None,
    ) -> List[Dict]:
        """
        Query files with missing metadata using optional filters.

        Retrieves files tracked in the files_missing_metadata table with optional
        filtering by language, metadata type, and branch. All filters are applied
        using AND logic.

        Args:
            language: Filter by language name (e.g., "python")
            missing_lsp: Filter by missing LSP metadata (True/False/None for all)
            missing_ts: Filter by missing Tree-sitter metadata (True/False/None for all)
            branch: Filter by git branch name

        Returns:
            List of dictionaries containing file information with keys:
                - file_absolute_path: str
                - language_name: str
                - branch: str
                - missing_lsp_metadata: bool
                - missing_ts_metadata: bool
                - created_at: str (ISO format timestamp)
                - updated_at: str (ISO format timestamp)

        Example:
            ```python
            # Get all Python files missing LSP metadata on main branch
            files = await tracker.get_files_missing_metadata(
                language="python",
                missing_lsp=True,
                branch="main"
            )

            for file in files:
                print(f"File: {file['file_absolute_path']}")
                print(f"Missing LSP: {file['missing_lsp_metadata']}")
            ```
        """
        try:
            # Build query with filters
            query = """
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

            if language is not None:
                query += " AND language_name = ?"
                params.append(language)

            if missing_lsp is not None:
                query += " AND missing_lsp_metadata = ?"
                params.append(1 if missing_lsp else 0)

            if missing_ts is not None:
                query += " AND missing_ts_metadata = ?"
                params.append(1 if missing_ts else 0)

            if branch is not None:
                query += " AND branch = ?"
                params.append(branch)

            query += " ORDER BY updated_at DESC"

            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(query, params)
                rows = cursor.fetchall()

                results = []
                for row in rows:
                    results.append(
                        {
                            "file_absolute_path": row["file_absolute_path"],
                            "language_name": row["language_name"],
                            "branch": row["branch"],
                            "missing_lsp_metadata": bool(row["missing_lsp_metadata"]),
                            "missing_ts_metadata": bool(row["missing_ts_metadata"]),
                            "created_at": row["created_at"],
                            "updated_at": row["updated_at"],
                        }
                    )

                logger.debug(
                    f"Retrieved {len(results)} files with missing metadata "
                    f"(language={language}, missing_lsp={missing_lsp}, "
                    f"missing_ts={missing_ts}, branch={branch})"
                )
                return results

        except Exception as e:
            logger.error(f"Failed to get files with missing metadata: {e}")
            return []

    async def remove_tracked_file(self, file_path: str) -> bool:
        """
        Remove a file from missing metadata tracking.

        Removes the tracking entry for a file, typically called when the missing
        metadata has been resolved (e.g., LSP server or Tree-sitter grammar installed).

        Args:
            file_path: Absolute path to the file to remove from tracking

        Returns:
            True if the file was removed, False if not found or on error

        Example:
            ```python
            # After installing missing LSP server
            success = await tracker.remove_tracked_file("/path/to/file.py")
            if success:
                print("File removed from tracking")
            ```
        """
        try:
            # Normalize file path to absolute path
            file_absolute_path = str(Path(file_path).resolve())

            async with self.state_manager.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM files_missing_metadata WHERE file_absolute_path = ?",
                    (file_absolute_path,),
                )

                deleted = cursor.rowcount > 0

                if deleted:
                    logger.debug(f"Removed tracked file: {file_absolute_path}")
                else:
                    logger.debug(f"File not found in tracking: {file_absolute_path}")

                return deleted

        except Exception as e:
            logger.error(f"Failed to remove tracked file {file_path}: {e}")
            return False

    async def get_tracked_file_count(self) -> Dict[str, int]:
        """
        Get count of tracked files by category.

        Provides statistics on tracked files, breaking down counts by metadata type.
        Useful for monitoring and reporting on missing language support tools.

        Returns:
            Dictionary with keys:
                - total: Total number of tracked files
                - missing_lsp: Files missing LSP metadata
                - missing_ts: Files missing Tree-sitter metadata
                - missing_both: Files missing both LSP and Tree-sitter metadata

        Example:
            ```python
            stats = await tracker.get_tracked_file_count()
            print(f"Total files with missing metadata: {stats['total']}")
            print(f"Missing LSP only: {stats['missing_lsp'] - stats['missing_both']}")
            print(f"Missing Tree-sitter only: {stats['missing_ts'] - stats['missing_both']}")
            print(f"Missing both: {stats['missing_both']}")
            ```
        """
        try:
            with self.state_manager._lock:
                # Get total count
                cursor = self.state_manager.connection.execute(
                    "SELECT COUNT(*) as count FROM files_missing_metadata"
                )
                total = cursor.fetchone()["count"]

                # Get count of files missing LSP metadata
                cursor = self.state_manager.connection.execute(
                    "SELECT COUNT(*) as count FROM files_missing_metadata WHERE missing_lsp_metadata = 1"
                )
                missing_lsp = cursor.fetchone()["count"]

                # Get count of files missing Tree-sitter metadata
                cursor = self.state_manager.connection.execute(
                    "SELECT COUNT(*) as count FROM files_missing_metadata WHERE missing_ts_metadata = 1"
                )
                missing_ts = cursor.fetchone()["count"]

                # Get count of files missing both
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT COUNT(*) as count
                    FROM files_missing_metadata
                    WHERE missing_lsp_metadata = 1 AND missing_ts_metadata = 1
                    """
                )
                missing_both = cursor.fetchone()["count"]

                stats = {
                    "total": total,
                    "missing_lsp": missing_lsp,
                    "missing_ts": missing_ts,
                    "missing_both": missing_both,
                }

                logger.debug(f"Tracked file count: {stats}")
                return stats

        except Exception as e:
            logger.error(f"Failed to get tracked file count: {e}")
            return {"total": 0, "missing_lsp": 0, "missing_ts": 0, "missing_both": 0}


    # Tool Availability Detection Methods

    async def check_lsp_available(self, language_name: str) -> Dict[str, Any]:
        """
        Check if LSP server is available for a language.

        Queries the languages table to determine if an LSP server is available
        for the specified language. Returns availability status and path if found.

        Args:
            language_name: Name of the language to check (e.g., "python", "rust")

        Returns:
            Dictionary with availability information:
            {
                "language": str,
                "available": bool,
                "path": Optional[str]
            }

            Returns {"available": False, "path": None} for non-existent languages.

        Example:
            ```python
            lsp_status = await tracker.check_lsp_available("python")
            if lsp_status["available"]:
                print(f"Python LSP available at: {lsp_status['path']}")
            else:
                print("Python LSP not available")
            ```
        """
        try:
            async with self.state_manager.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT language_name, lsp_absolute_path, lsp_missing
                    FROM languages
                    WHERE language_name = ?
                    """,
                    (language_name,),
                )
                row = cursor.fetchone()

                if not row:
                    logger.debug(f"Language '{language_name}' not found in database")
                    return {
                        "language": language_name,
                        "available": False,
                        "path": None,
                    }

                lsp_path = row["lsp_absolute_path"]
                lsp_missing = bool(row["lsp_missing"])

                # Available if path exists and not marked as missing
                available = lsp_path is not None and not lsp_missing

                logger.debug(
                    f"LSP check for '{language_name}': "
                    f"available={available}, path={lsp_path}"
                )

                return {
                    "language": language_name,
                    "available": available,
                    "path": lsp_path if available else None,
                }

        except Exception as e:
            logger.warning(
                f"Error checking LSP availability for '{language_name}': {e}",
                exc_info=True,
            )
            return {
                "language": language_name,
                "available": False,
                "path": None,
            }

    async def check_tree_sitter_available(self) -> Dict[str, Any]:
        """
        Check if tree-sitter CLI is available.

        Queries the languages table to find any language with a tree-sitter CLI
        path set. Since tree-sitter CLI is global (not language-specific), any
        non-null path indicates availability.

        Returns:
            Dictionary with availability information:
            {
                "available": bool,
                "path": Optional[str]
            }

            Returns first found tree-sitter CLI path, or None if unavailable.

        Example:
            ```python
            ts_status = await tracker.check_tree_sitter_available()
            if ts_status["available"]:
                print(f"Tree-sitter CLI available at: {ts_status['path']}")
            else:
                print("Tree-sitter CLI not available")
            ```
        """
        try:
            async with self.state_manager.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT ts_cli_absolute_path, ts_missing
                    FROM languages
                    WHERE ts_cli_absolute_path IS NOT NULL
                    LIMIT 1
                    """
                )
                row = cursor.fetchone()

                if not row:
                    logger.debug("Tree-sitter CLI not found in database")
                    return {"available": False, "path": None}

                ts_path = row["ts_cli_absolute_path"]
                ts_missing = bool(row["ts_missing"])

                # Available if path exists and not marked as missing
                available = ts_path is not None and not ts_missing

                logger.debug(
                    f"Tree-sitter check: available={available}, path={ts_path}"
                )

                return {
                    "available": available,
                    "path": ts_path if available else None,
                }

        except Exception as e:
            logger.warning(
                f"Error checking tree-sitter availability: {e}", exc_info=True
            )
            return {"available": False, "path": None}

    async def check_tools_available(self, language_name: str) -> Dict[str, Any]:
        """
        Check availability of both LSP and tree-sitter for a language.

        Combines LSP and tree-sitter availability checks into a single call.
        Useful for comprehensive tool availability assessment before processing.

        Args:
            language_name: Name of the language to check

        Returns:
            Dictionary with combined availability information:
            {
                "language": str,
                "lsp": {
                    "available": bool,
                    "path": Optional[str]
                },
                "tree_sitter": {
                    "available": bool,
                    "path": Optional[str]
                }
            }

        Example:
            ```python
            tools_status = await tracker.check_tools_available("rust")
            print(f"LSP available: {tools_status['lsp']['available']}")
            print(f"Tree-sitter available: {tools_status['tree_sitter']['available']}")
            ```
        """
        try:
            # Check LSP availability
            lsp_status = await self.check_lsp_available(language_name)

            # Check tree-sitter availability
            ts_status = await self.check_tree_sitter_available()

            result = {
                "language": language_name,
                "lsp": {
                    "available": lsp_status["available"],
                    "path": lsp_status["path"],
                },
                "tree_sitter": {
                    "available": ts_status["available"],
                    "path": ts_status["path"],
                },
            }

            logger.debug(
                f"Tools check for '{language_name}': "
                f"LSP={result['lsp']['available']}, "
                f"tree-sitter={result['tree_sitter']['available']}"
            )

            return result

        except Exception as e:
            logger.warning(
                f"Error checking tools availability for '{language_name}': {e}",
                exc_info=True,
            )
            return {
                "language": language_name,
                "lsp": {"available": False, "path": None},
                "tree_sitter": {"available": False, "path": None},
            }

    async def get_missing_tools_summary(self) -> Dict[str, List[str]]:
        """
        Get summary of languages grouped by missing tools.

        Queries all languages in the database and groups them by which tools
        are missing. Useful for reporting and diagnostics.

        Returns:
            Dictionary with languages grouped by tool availability:
            {
                "missing_lsp": ["python", "rust"],
                "missing_tree_sitter": ["javascript"],
                "both_available": ["go", "java"]
            }

            Only includes languages that have at least one tool configured
            (either LSP or tree-sitter).

        Example:
            ```python
            summary = await tracker.get_missing_tools_summary()
            print(f"Languages missing LSP: {summary['missing_lsp']}")
            print(f"Languages missing tree-sitter: {summary['missing_tree_sitter']}")
            print(f"Languages with both tools: {summary['both_available']}")
            ```
        """
        try:
            async with self.state_manager.transaction() as conn:
                cursor = conn.execute(
                    """
                    SELECT
                        language_name,
                        lsp_name,
                        lsp_absolute_path,
                        lsp_missing,
                        ts_grammar,
                        ts_cli_absolute_path,
                        ts_missing
                    FROM languages
                    WHERE lsp_name IS NOT NULL OR ts_grammar IS NOT NULL
                    ORDER BY language_name
                    """
                )
                rows = cursor.fetchall()

                missing_lsp: List[str] = []
                missing_tree_sitter: List[str] = []
                both_available: List[str] = []

                for row in rows:
                    language_name = row["language_name"]
                    has_lsp_config = row["lsp_name"] is not None
                    lsp_path = row["lsp_absolute_path"]
                    lsp_missing = bool(row["lsp_missing"])
                    has_ts_config = row["ts_grammar"] is not None
                    ts_path = row["ts_cli_absolute_path"]
                    ts_missing = bool(row["ts_missing"])

                    # Check LSP availability (only if configured)
                    lsp_available = (
                        has_lsp_config and lsp_path is not None and not lsp_missing
                    )

                    # Check tree-sitter availability (only if configured)
                    ts_available = (
                        has_ts_config and ts_path is not None and not ts_missing
                    )

                    # Categorize language based on configured tools
                    has_any_config = has_lsp_config or has_ts_config

                    if not has_any_config:
                        continue  # Skip languages with no tool configuration

                    # Determine which tools are missing
                    lsp_is_missing = has_lsp_config and not lsp_available
                    ts_is_missing = has_ts_config and not ts_available

                    if lsp_is_missing and ts_is_missing:
                        # Both configured but both missing
                        missing_lsp.append(language_name)
                        missing_tree_sitter.append(language_name)
                    elif lsp_is_missing:
                        missing_lsp.append(language_name)
                    elif ts_is_missing:
                        missing_tree_sitter.append(language_name)
                    else:
                        # All configured tools are available
                        both_available.append(language_name)

                logger.debug(
                    f"Missing tools summary: "
                    f"{len(missing_lsp)} missing LSP, "
                    f"{len(missing_tree_sitter)} missing tree-sitter, "
                    f"{len(both_available)} both available"
                )

                return {
                    "missing_lsp": missing_lsp,
                    "missing_tree_sitter": missing_tree_sitter,
                    "both_available": both_available,
                }

        except Exception as e:
            logger.warning(f"Error getting missing tools summary: {e}", exc_info=True)
            return {
                "missing_lsp": [],
                "missing_tree_sitter": [],
                "both_available": [],
            }

    # Tool-Available Requeuing Methods

    async def requeue_when_tools_available(
        self,
        tool_type: str,
        language: Optional[str] = None,
        priority: int = 5,
    ) -> Dict[str, Any]:
        """
        Requeue files when tools become available.

        Checks if specified tools are available and requeues tracked files for
        processing when tools are ready. Supports both LSP and tree-sitter tools.

        For LSP tools, requires language parameter to check language-specific server.
        For tree-sitter, checks global CLI availability and requeues all files.

        Priority calculation (when priority not explicitly provided):
        - Uses calculate_requeue_priority() to determine priority based on context
        - Files in current project: HIGH priority (8)
        - Files on same branch: NORMAL priority (5)
        - Other files: LOW priority (2)

        Args:
            tool_type: Type of tool to check ('lsp' or 'tree_sitter')
            language: Language name (required for 'lsp', ignored for 'tree_sitter')
            current_project_root: Current project root for priority calculation (optional)
            priority: Explicit priority override (0-10), ignores priority calculation if set

        Returns:
            Dictionary with requeuing results:
            {
                "tool_type": str,
                "language": Optional[str],
                "tool_available": bool,
                "files_requeued": int,
                "files_failed": int,
                "files_removed": int,
                "errors": List[str]
            }

        Raises:
            ValueError: If tool_type is invalid, priority out of range, or
                       language not provided for LSP tool type

        Example:
            ```python
            # Requeue Python files when LSP becomes available
            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=5
            )
            print(f"Requeued {result['files_requeued']} files")

            # Requeue all files when tree-sitter becomes available
            result = await tracker.requeue_when_tools_available(
                tool_type="tree_sitter",
                priority=7
            )
            ```
        """
        # Validate inputs
        if tool_type not in ("lsp", "tree_sitter"):
            raise ValueError(f"Invalid tool_type: {tool_type}. Must be 'lsp' or 'tree_sitter'")

        if priority < 0 or priority > 10:
            raise ValueError(f"Priority must be between 0 and 10, got {priority}")

        if tool_type == "lsp" and not language:
            raise ValueError("Language parameter required for LSP tool type")

        # Check tool availability
        tool_available = False
        tool_path = None

        if tool_type == "lsp":
            lsp_status = await self.check_lsp_available(language)
            tool_available = lsp_status["available"]
            tool_path = lsp_status["path"]
        else:  # tree_sitter
            ts_status = await self.check_tree_sitter_available()
            tool_available = ts_status["available"]
            tool_path = ts_status["path"]

        result = {
            "tool_type": tool_type,
            "language": language,
            "tool_available": tool_available,
            "tool_path": tool_path,
            "files_requeued": 0,
            "files_failed": 0,
            "files_removed": 0,
            "errors": [],
        }

        # If tool not available, return early
        if not tool_available:
            logger.info(
                f"{tool_type.upper()} {'for ' + language if language else ''} "
                f"not available, skipping requeue"
            )
            return result

        # Get current branch for priority calculation
        current_branch = None
        if current_project_root and priority is None:
            try:
                current_branch = await self.state_manager.get_current_branch(
                    Path(current_project_root)
                )
            except Exception as e:
                logger.debug(f"Could not get current branch: {e}")

        # Requeue files based on tool type
        try:
            if tool_type == "lsp":
                requeue_result = await self._requeue_for_language_lsp(
                    language,
                    current_project_root=current_project_root,
                    current_branch=current_branch,
                    explicit_priority=priority
                )
            else:  # tree_sitter
                requeue_result = await self._requeue_files_missing_tree_sitter(
                    current_project_root=current_project_root,
                    current_branch=current_branch,
                    explicit_priority=priority
                )

            result.update(requeue_result)

            logger.info(
                f"Requeued {result['files_requeued']} files for {tool_type} "
                f"{'(' + language + ')' if language else ''}, "
                f"{result['files_failed']} failed, "
                f"{result['files_removed']} removed from tracking"
            )

        except Exception as e:
            error_msg = f"Failed to requeue files: {e}"
            logger.error(error_msg, exc_info=True)
            result["errors"].append(error_msg)

        return result

    async def _get_languages_with_missing_lsp(self) -> List[str]:
        """
        Get distinct languages with files missing LSP metadata.

        Returns:
            List of language names that have files missing LSP metadata
        """
        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT DISTINCT language_name
                    FROM files_missing_metadata
                    WHERE missing_lsp_metadata = 1
                    ORDER BY language_name
                    """
                )
                rows = cursor.fetchall()
                languages = [row["language_name"] for row in rows]

                logger.debug(f"Found {len(languages)} languages with files missing LSP")
                return languages

        except Exception as e:
            logger.error(f"Failed to get languages with missing LSP: {e}")
            return []

    async def _requeue_for_language_lsp(
        self,
        language: str,
        current_project_root: Optional[str] = None,
        current_branch: Optional[str] = None,
        explicit_priority: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Requeue files for a specific language when LSP becomes available.

        Args:
            language: Language name
            current_project_root: Current project root for priority calculation (optional)
            current_branch: Current branch for priority calculation (optional)
            explicit_priority: Explicit priority override (0-10), ignores calculation if set

        Returns:
            Dictionary with counts:
            {
                "files_requeued": int,
                "files_failed": int,
                "files_removed": int
            }
        """
        files_requeued = 0
        files_failed = 0
        files_removed = 0

        # Get files missing LSP for this language
        files = await self.get_files_missing_metadata(
            language=language,
            missing_lsp=True
        )

        if not files:
            logger.debug(f"No files found missing LSP for language: {language}")
            return {
                "files_requeued": 0,
                "files_failed": 0,
                "files_removed": 0,
            }

        logger.info(f"Processing {len(files)} files missing LSP for {language}")

        # Process in batches of 100
        batch_size = 100
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]

            for file_info in batch:
                file_path = file_info["file_absolute_path"]
                file_branch = file_info["branch"]

                # Calculate priority if not explicitly provided
                if explicit_priority is not None:
                    priority = explicit_priority
                else:
                    priority = self.calculate_requeue_priority(
                        file_path=file_path,
                        file_branch=file_branch,
                        current_project_root=current_project_root,
                        current_branch=current_branch,
                    )

                try:
                    # Enqueue the file for processing
                    # Note: We need tenant_id and collection for enqueue
                    # Using default values for now, may need to be configurable
                    await self.state_manager.enqueue(
                        file_path=file_path,
                        collection=f"default-{language}",  # Collection name based on language
                        priority=priority,
                        tenant_id="default",  # Default tenant
                        branch=file_branch,
                        metadata={"requeued_for": "lsp", "language": language},
                    )

                    files_requeued += 1
                    logger.debug(
                        f"Requeued file {file_path} with priority {priority} "
                        f"(branch={file_branch}, language={language})"
                    )

                    # Remove from tracking after successful enqueue
                    removed = await self.remove_tracked_file(file_path)
                    if removed:
                        files_removed += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to requeue file {file_path}: {e}",
                        exc_info=False
                    )
                    files_failed += 1

        return {
            "files_requeued": files_requeued,
            "files_failed": files_failed,
            "files_removed": files_removed,
        }

    async def _requeue_files_missing_tree_sitter(
        self,
        current_project_root: Optional[str] = None,
        current_branch: Optional[str] = None,
        explicit_priority: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Requeue all files missing tree-sitter when CLI becomes available.

        Args:
            current_project_root: Current project root for priority calculation (optional)
            current_branch: Current branch for priority calculation (optional)
            explicit_priority: Explicit priority override (0-10), ignores calculation if set

        Returns:
            Dictionary with counts:
            {
                "files_requeued": int,
                "files_failed": int,
                "files_removed": int
            }
        """
        files_requeued = 0
        files_failed = 0
        files_removed = 0

        # Get all files missing tree-sitter
        files = await self.get_files_missing_metadata(missing_ts=True)

        if not files:
            logger.debug("No files found missing tree-sitter")
            return {
                "files_requeued": 0,
                "files_failed": 0,
                "files_removed": 0,
            }

        logger.info(f"Processing {len(files)} files missing tree-sitter")

        # Process in batches of 100
        batch_size = 100
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]

            for file_info in batch:
                file_path = file_info["file_absolute_path"]
                language = file_info["language_name"]
                file_branch = file_info["branch"]

                # Calculate priority if not explicitly provided
                if explicit_priority is not None:
                    priority = explicit_priority
                else:
                    priority = self.calculate_requeue_priority(
                        file_path=file_path,
                        file_branch=file_branch,
                        current_project_root=current_project_root,
                        current_branch=current_branch,
                    )

                try:
                    # Enqueue the file for processing
                    await self.state_manager.enqueue(
                        file_path=file_path,
                        collection=f"default-{language}",  # Collection name based on language
                        priority=priority,
                        tenant_id="default",  # Default tenant
                        branch=file_branch,
                        metadata={"requeued_for": "tree_sitter", "language": language},
                    )

                    files_requeued += 1
                    logger.debug(
                        f"Requeued file {file_path} with priority {priority} "
                        f"(branch={file_branch}, language={language})"
                    )

                    # Remove from tracking after successful enqueue
                    removed = await self.remove_tracked_file(file_path)
                    if removed:
                        files_removed += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to requeue file {file_path}: {e}",
                        exc_info=False
                    )
                    files_failed += 1

        return {
            "files_requeued": files_requeued,
            "files_failed": files_failed,
            "files_removed": files_removed,
        }
