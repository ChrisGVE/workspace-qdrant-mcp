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
