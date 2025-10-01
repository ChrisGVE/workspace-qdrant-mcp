"""Database integration for tool path updates.

This module provides integration between tool discovery and the SQLite state
manager, enabling automatic updates of tool paths in the database as they
are discovered.

Architecture:
    - Updates languages table with LSP and tree-sitter paths
    - Updates tools table with generic tool paths
    - Batch operations with transaction support
    - Preserves user customizations where applicable
    - Graceful error handling (warnings instead of failures)

Example:
    ```python
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager
    from workspace_qdrant_mcp.core.tool_database_integration import ToolDatabaseIntegration

    # Initialize state manager and integration
    state_manager = SQLiteStateManager()
    await state_manager.initialize()

    integration = ToolDatabaseIntegration(state_manager)

    # Update single LSP path
    await integration.update_lsp_path("python", "/usr/bin/pyright-langserver")

    # Batch update LSP paths
    lsp_paths = {
        "python": "/usr/bin/pyright-langserver",
        "rust": "/usr/bin/rust-analyzer",
        "typescript": None,  # Mark as missing
    }
    count = await integration.batch_update_lsp_paths(lsp_paths)
    print(f"Updated {count} LSP paths")

    # Update tree-sitter CLI path for all languages
    await integration.update_tree_sitter_path("/usr/local/bin/tree-sitter")
    ```
"""

from typing import Dict, Optional

from loguru import logger

from .sqlite_state_manager import SQLiteStateManager


class ToolDatabaseIntegration:
    """Integration layer between tool discovery and database state.

    Provides methods to update the SQLite database with discovered tool paths,
    including LSP servers, tree-sitter CLI, and other development tools.
    Handles transaction management, error logging, and user customization
    preservation.

    Attributes:
        state_manager: SQLiteStateManager instance for database operations
    """

    def __init__(self, state_manager: SQLiteStateManager):
        """Initialize tool database integration.

        Args:
            state_manager: Initialized SQLiteStateManager instance
        """
        self.state_manager = state_manager
        logger.debug("ToolDatabaseIntegration initialized")

    async def update_lsp_path(
        self, language_name: str, lsp_absolute_path: Optional[str]
    ) -> bool:
        """Update LSP path for a specific language.

        Updates the languages table with the LSP absolute path. If the path
        is None, marks the LSP as missing. If the path is provided, clears
        the missing flag and stores the absolute path.

        User customizations (if lsp_absolute_path was manually set) are
        preserved by only updating if the current value is NULL or lsp_missing
        is True.

        Args:
            language_name: Name of the language to update
            lsp_absolute_path: Absolute path to LSP executable, or None if not found

        Returns:
            True if update succeeded, False otherwise
        """
        try:
            async with self.state_manager.transaction() as conn:
                if lsp_absolute_path is None:
                    # Mark as missing
                    cursor = conn.execute(
                        """
                        UPDATE languages
                        SET lsp_missing = 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE language_name = ?
                        """,
                        (language_name,),
                    )
                else:
                    # Update path and clear missing flag
                    # Only update if not already customized (NULL or missing=True)
                    cursor = conn.execute(
                        """
                        UPDATE languages
                        SET lsp_absolute_path = ?,
                            lsp_missing = 0,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE language_name = ?
                          AND (lsp_absolute_path IS NULL OR lsp_missing = 1)
                        """,
                        (lsp_absolute_path, language_name),
                    )

                if cursor.rowcount == 0:
                    logger.debug(
                        f"Language '{language_name}' not in database or already customized"
                    )
                    return False

                logger.debug(
                    f"Updated LSP path for '{language_name}': "
                    f"{'<missing>' if lsp_absolute_path is None else lsp_absolute_path}"
                )
                return True

        except Exception as e:
            logger.warning(
                f"Failed to update LSP path for '{language_name}': {e}",
                exc_info=True,
            )
            return False

    async def update_tree_sitter_path(self, ts_cli_absolute_path: str) -> bool:
        """Update tree-sitter CLI path for all languages.

        Updates the languages table with the tree-sitter CLI path for all
        languages that use tree-sitter (ts_grammar IS NOT NULL). Only updates
        languages where the path is NULL to preserve user customizations.

        Args:
            ts_cli_absolute_path: Absolute path to tree-sitter CLI executable

        Returns:
            True if update succeeded (updated at least one row), False otherwise
        """
        try:
            async with self.state_manager.transaction() as conn:
                # Update all languages with tree-sitter grammar that don't have a path set
                cursor = conn.execute(
                    """
                    UPDATE languages
                    SET ts_cli_absolute_path = ?,
                        ts_missing = 0,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE ts_grammar IS NOT NULL
                      AND ts_cli_absolute_path IS NULL
                    """,
                    (ts_cli_absolute_path,),
                )

                updated_count = cursor.rowcount

                if updated_count > 0:
                    logger.info(
                        f"Updated tree-sitter CLI path for {updated_count} languages: "
                        f"{ts_cli_absolute_path}"
                    )
                    return True
                else:
                    logger.debug(
                        "No languages required tree-sitter CLI path update "
                        "(all already set or no tree-sitter languages)"
                    )
                    return False

        except Exception as e:
            logger.warning(
                f"Failed to update tree-sitter CLI path: {e}", exc_info=True
            )
            return False

    async def update_tool_path(
        self, tool_name: str, tool_path: Optional[str], tool_type: str
    ) -> bool:
        """Update or insert tool path in tools table.

        Adds or updates a tool entry in the tools table. Uses INSERT OR REPLACE
        with COALESCE to preserve user customizations when appropriate.

        Args:
            tool_name: Name of the tool (e.g., "pyright-langserver", "tree-sitter")
            tool_path: Absolute path to tool executable, or None if not found
            tool_type: Type of tool, one of:
                - 'lsp_server': LSP server executables
                - 'tree_sitter_cli': Tree-sitter CLI executable

        Returns:
            True if update succeeded, False otherwise

        Raises:
            ValueError: If tool_type is not one of the allowed values
        """
        # Validate tool_type against actual database schema
        allowed_types = {'lsp_server', 'tree_sitter_cli'}
        if tool_type not in allowed_types:
            raise ValueError(
                f"Invalid tool_type '{tool_type}'. Must be one of: {allowed_types}"
            )

        try:
            async with self.state_manager.transaction() as conn:
                # Check if tool already exists
                cursor = conn.execute(
                    "SELECT absolute_path, missing FROM tools WHERE tool_name = ?",
                    (tool_name,),
                )
                existing = cursor.fetchone()

                if existing:
                    # Tool exists - only update if not customized or was missing
                    existing_path = existing["absolute_path"]
                    was_missing = existing["missing"]

                    # Skip if user has customized the path (path set and not missing)
                    if existing_path is not None and not was_missing:
                        logger.debug(
                            f"Tool '{tool_name}' already has customized path, skipping update"
                        )
                        return False

                # Insert or update tool
                if tool_path is None:
                    # Mark as missing
                    conn.execute(
                        """
                        INSERT INTO tools (tool_name, tool_type, absolute_path, missing, last_check_at)
                        VALUES (?, ?, NULL, 1, CURRENT_TIMESTAMP)
                        ON CONFLICT(tool_name) DO UPDATE SET
                            missing = 1,
                            last_check_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (tool_name, tool_type),
                    )
                    logger.debug(f"Marked tool '{tool_name}' as missing")
                else:
                    # Update path and clear missing flag
                    conn.execute(
                        """
                        INSERT INTO tools (tool_name, tool_type, absolute_path, missing, last_check_at)
                        VALUES (?, ?, ?, 0, CURRENT_TIMESTAMP)
                        ON CONFLICT(tool_name) DO UPDATE SET
                            absolute_path = excluded.absolute_path,
                            tool_type = excluded.tool_type,
                            missing = 0,
                            last_check_at = CURRENT_TIMESTAMP,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (tool_name, tool_type, tool_path),
                    )
                    logger.debug(f"Updated tool '{tool_name}': {tool_path}")

                return True

        except Exception as e:
            logger.warning(
                f"Failed to update tool path for '{tool_name}': {e}", exc_info=True
            )
            return False

    async def batch_update_lsp_paths(
        self, lsp_paths: Dict[str, Optional[str]]
    ) -> int:
        """Batch update LSP paths for multiple languages.

        Updates LSP paths for multiple languages in a single transaction.
        Provides atomicity - either all updates succeed or all are rolled back.

        Args:
            lsp_paths: Dictionary mapping language_name to lsp_absolute_path.
                Use None as value to mark LSP as missing.
                Example:
                {
                    "python": "/usr/bin/pyright-langserver",
                    "rust": "/usr/bin/rust-analyzer",
                    "typescript": None,  # Mark as missing
                }

        Returns:
            Count of successfully updated languages
        """
        if not lsp_paths:
            logger.debug("No LSP paths to update")
            return 0

        updated_count = 0

        try:
            async with self.state_manager.transaction() as conn:
                for language_name, lsp_path in lsp_paths.items():
                    try:
                        if lsp_path is None:
                            # Mark as missing
                            cursor = conn.execute(
                                """
                                UPDATE languages
                                SET lsp_missing = 1,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE language_name = ?
                                """,
                                (language_name,),
                            )
                        else:
                            # Update path and clear missing flag
                            cursor = conn.execute(
                                """
                                UPDATE languages
                                SET lsp_absolute_path = ?,
                                    lsp_missing = 0,
                                    updated_at = CURRENT_TIMESTAMP
                                WHERE language_name = ?
                                  AND (lsp_absolute_path IS NULL OR lsp_missing = 1)
                                """,
                                (lsp_path, language_name),
                            )

                        if cursor.rowcount > 0:
                            updated_count += 1
                            logger.debug(
                                f"Batch updated LSP for '{language_name}': "
                                f"{'<missing>' if lsp_path is None else lsp_path}"
                            )

                    except Exception as e:
                        # Log error but continue with other updates
                        logger.warning(
                            f"Error updating LSP path for '{language_name}' in batch: {e}"
                        )

            logger.info(
                f"Batch update complete: {updated_count}/{len(lsp_paths)} LSP paths updated"
            )
            return updated_count

        except Exception as e:
            logger.error(f"Batch LSP path update failed: {e}", exc_info=True)
            return 0

    async def batch_update_tool_paths(
        self, tool_paths: Dict[str, str], tool_type: str
    ) -> int:
        """Batch update tool paths for multiple tools.

        Updates tool paths for multiple tools of the same type in a single
        transaction. Provides atomicity - either all updates succeed or all
        are rolled back.

        Args:
            tool_paths: Dictionary mapping tool_name to tool_path.
                All tools must be of the same type.
                Example:
                {
                    "pyright-langserver": "/usr/bin/pyright-langserver",
                    "rust-analyzer": "/usr/bin/rust-analyzer",
                    "typescript-language-server": "/usr/bin/typescript-language-server",
                }
            tool_type: Type of tools being updated, one of:
                - 'lsp_server'
                - 'tree_sitter_cli'

        Returns:
            Count of successfully updated tools

        Raises:
            ValueError: If tool_type is not one of the allowed values
        """
        # Validate tool_type against actual database schema
        allowed_types = {'lsp_server', 'tree_sitter_cli'}
        if tool_type not in allowed_types:
            raise ValueError(
                f"Invalid tool_type '{tool_type}'. Must be one of: {allowed_types}"
            )

        if not tool_paths:
            logger.debug("No tool paths to update")
            return 0

        updated_count = 0

        try:
            async with self.state_manager.transaction() as conn:
                for tool_name, tool_path in tool_paths.items():
                    try:
                        # Check if tool exists and is customized
                        cursor = conn.execute(
                            "SELECT absolute_path, missing FROM tools WHERE tool_name = ?",
                            (tool_name,),
                        )
                        existing = cursor.fetchone()

                        # Skip if customized
                        if existing:
                            existing_path = existing["absolute_path"]
                            was_missing = existing["missing"]
                            if existing_path is not None and not was_missing:
                                logger.debug(
                                    f"Tool '{tool_name}' has customized path, skipping"
                                )
                                continue

                        # Insert or update tool
                        conn.execute(
                            """
                            INSERT INTO tools (tool_name, tool_type, absolute_path, missing, last_check_at)
                            VALUES (?, ?, ?, 0, CURRENT_TIMESTAMP)
                            ON CONFLICT(tool_name) DO UPDATE SET
                                absolute_path = excluded.absolute_path,
                                tool_type = excluded.tool_type,
                                missing = 0,
                                last_check_at = CURRENT_TIMESTAMP,
                                updated_at = CURRENT_TIMESTAMP
                            """,
                            (tool_name, tool_type, tool_path),
                        )
                        updated_count += 1
                        logger.debug(f"Batch updated tool '{tool_name}': {tool_path}")

                    except Exception as e:
                        # Log error but continue with other updates
                        logger.warning(
                            f"Error updating tool path for '{tool_name}' in batch: {e}"
                        )

            logger.info(
                f"Batch update complete: {updated_count}/{len(tool_paths)} tool paths updated"
            )
            return updated_count

        except Exception as e:
            logger.error(f"Batch tool path update failed: {e}", exc_info=True)
            return 0
