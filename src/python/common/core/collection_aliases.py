"""
Collection alias system for smooth project ID migrations.

This module implements the collection alias system that enables zero-downtime
migrations when a local project gains a git remote. It handles:
- Alias creation and deletion via Qdrant API
- SQLite state tracking for alias mappings
- Transparent alias resolution in queries
- Automatic alias management during project remote updates

Use case:
    Before: _path_abc123def456789a (local path hash)
    After: _github_com_user_repo (git remote URL)
    Need zero-downtime migration via alias

Key Features:
    - Qdrant collection alias operations (create, delete, list)
    - SQLite persistence for alias tracking
    - Alias resolution in search/query operations
    - Integration with project collection naming system
    - Audit trail for alias operations

Example:
    ```python
    from workspace_qdrant_mcp.core.collection_aliases import AliasManager
    from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient

    client = QdrantWorkspaceClient()
    alias_manager = AliasManager(client.client, client.state_manager)

    # Create alias for migration
    await alias_manager.create_alias(
        old_collection="_path_abc123def456",
        new_collection="_github_com_user_repo"
    )

    # Resolve collection name (handles aliases transparently)
    actual_collection = await alias_manager.resolve_collection_name("_path_abc123def456")
    # Returns: "_github_com_user_repo"

    # List all aliases
    aliases = await alias_manager.list_aliases()

    # Remove alias after migration complete
    await alias_manager.delete_alias("_path_abc123def456")
    ```
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
)

# Import SQLiteStateManager for alias persistence
from .sqlite_state_manager import SQLiteStateManager


@dataclass
class CollectionAlias:
    """
    Record for tracking collection aliases.

    Attributes:
        alias_name: The alias name (typically the old collection name)
        collection_name: The actual collection name (typically the new name)
        created_at: When the alias was created
        created_by: Source of alias creation (e.g., "cli", "migration")
        metadata: Optional metadata (e.g., migration details)
    """

    alias_name: str
    collection_name: str
    created_at: datetime = None
    created_by: str = "system"
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class AliasManager:
    """
    Manages collection aliases for smooth project ID migrations.

    This class provides comprehensive alias management including:
    - Creation and deletion of Qdrant collection aliases
    - SQLite persistence for alias state tracking
    - Transparent alias resolution for queries
    - Integration with project collection naming system
    - Audit trail for all alias operations

    The alias system enables zero-downtime migrations when project IDs change,
    such as when a local project gains a git remote URL.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        state_manager: SQLiteStateManager | None = None
    ):
        """
        Initialize the alias manager.

        Args:
            qdrant_client: Qdrant client for alias operations
            state_manager: Optional SQLite state manager for persistence
        """
        self.qdrant_client = qdrant_client
        self.state_manager = state_manager
        self._alias_cache: dict[str, str] = {}  # alias_name -> collection_name
        self._cache_valid = False

    async def initialize(self) -> bool:
        """
        Initialize the alias manager and create database schema.

        Returns:
            True if initialization succeeded
        """
        try:
            # Create aliases table if using state manager
            if self.state_manager and self.state_manager.connection:
                await self._create_aliases_table()

            # Load existing aliases into cache
            await self._refresh_cache()

            logger.info("AliasManager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AliasManager: {e}")
            return False

    async def _create_aliases_table(self):
        """Create the collection_aliases table in SQLite if it doesn't exist."""
        if not self.state_manager or not self.state_manager.connection:
            return

        try:
            with self.state_manager._lock:
                self.state_manager.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS collection_aliases (
                        alias_name TEXT PRIMARY KEY,
                        collection_name TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        created_by TEXT NOT NULL DEFAULT 'system',
                        metadata TEXT,
                        UNIQUE(alias_name)
                    )
                    """
                )

                # Create indexes for faster lookups
                self.state_manager.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_collection_aliases_collection ON collection_aliases(collection_name)"
                )
                self.state_manager.connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_collection_aliases_created_at ON collection_aliases(created_at)"
                )

                self.state_manager.connection.commit()

            logger.debug("Created collection_aliases table")

        except Exception as e:
            logger.error(f"Failed to create aliases table: {e}")
            raise

    async def create_alias(
        self,
        alias_name: str,
        collection_name: str,
        created_by: str = "system",
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Create a collection alias in Qdrant and persist to SQLite.

        This creates an alias that points to an existing collection, allowing
        queries to use either the alias name or the actual collection name.

        Args:
            alias_name: The alias name (e.g., old collection name)
            collection_name: The actual collection name (e.g., new collection name)
            created_by: Source of alias creation (e.g., "cli", "migration")
            metadata: Optional metadata dictionary

        Returns:
            True if alias was created successfully

        Raises:
            Exception: If alias creation fails in Qdrant or SQLite
        """
        try:
            logger.info(f"Creating alias: {alias_name} -> {collection_name}")

            # Create alias in Qdrant
            self.qdrant_client.update_collection_aliases(
                change_aliases_operations=[
                    CreateAliasOperation(
                        create_alias=CreateAlias(
                            collection_name=collection_name,
                            alias_name=alias_name
                        )
                    )
                ]
            )

            logger.debug(f"Created Qdrant alias: {alias_name} -> {collection_name}")

            # Persist to SQLite if state manager available
            if self.state_manager:
                await self._save_alias_to_db(
                    CollectionAlias(
                        alias_name=alias_name,
                        collection_name=collection_name,
                        created_by=created_by,
                        metadata=metadata
                    )
                )

            # Update cache
            self._alias_cache[alias_name] = collection_name
            self._cache_valid = True

            logger.info(f"Successfully created alias: {alias_name} -> {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create alias {alias_name} -> {collection_name}: {e}")
            raise

    async def delete_alias(self, alias_name: str) -> bool:
        """
        Delete a collection alias from Qdrant and SQLite.

        Args:
            alias_name: The alias name to delete

        Returns:
            True if alias was deleted successfully

        Raises:
            Exception: If alias deletion fails
        """
        try:
            logger.info(f"Deleting alias: {alias_name}")

            # Delete alias from Qdrant
            self.qdrant_client.update_collection_aliases(
                change_aliases_operations=[
                    DeleteAliasOperation(
                        delete_alias=DeleteAlias(alias_name=alias_name)
                    )
                ]
            )

            logger.debug(f"Deleted Qdrant alias: {alias_name}")

            # Remove from SQLite if state manager available
            if self.state_manager:
                await self._delete_alias_from_db(alias_name)

            # Update cache
            if alias_name in self._alias_cache:
                del self._alias_cache[alias_name]

            logger.info(f"Successfully deleted alias: {alias_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete alias {alias_name}: {e}")
            raise

    async def get_alias(self, alias_name: str) -> CollectionAlias | None:
        """
        Get alias information from SQLite.

        Args:
            alias_name: The alias name to look up

        Returns:
            CollectionAlias if found, None otherwise
        """
        if not self.state_manager or not self.state_manager.connection:
            return None

        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT alias_name, collection_name, created_at, created_by, metadata
                    FROM collection_aliases
                    WHERE alias_name = ?
                    """,
                    (alias_name,)
                )

                row = cursor.fetchone()
                if not row:
                    return None

                return CollectionAlias(
                    alias_name=row["alias_name"],
                    collection_name=row["collection_name"],
                    created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                    created_by=row["created_by"],
                    metadata=self.state_manager._deserialize_json(row["metadata"])
                )

        except Exception as e:
            logger.error(f"Failed to get alias {alias_name}: {e}")
            return None

    async def list_aliases(self) -> list[CollectionAlias]:
        """
        List all collection aliases from SQLite.

        Returns:
            List of CollectionAlias objects
        """
        if not self.state_manager or not self.state_manager.connection:
            return []

        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT alias_name, collection_name, created_at, created_by, metadata
                    FROM collection_aliases
                    ORDER BY created_at DESC
                    """
                )

                rows = cursor.fetchall()
                aliases = []

                for row in rows:
                    aliases.append(
                        CollectionAlias(
                            alias_name=row["alias_name"],
                            collection_name=row["collection_name"],
                            created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                            created_by=row["created_by"],
                            metadata=self.state_manager._deserialize_json(row["metadata"])
                        )
                    )

                return aliases

        except Exception as e:
            logger.error(f"Failed to list aliases: {e}")
            return []

    async def resolve_collection_name(self, name: str) -> str:
        """
        Resolve a collection name, handling aliases transparently.

        If the name is an alias, returns the actual collection name.
        Otherwise returns the name unchanged.

        Args:
            name: Collection name or alias to resolve

        Returns:
            Actual collection name
        """
        # Refresh cache if invalid
        if not self._cache_valid:
            await self._refresh_cache()

        # Check cache for alias
        if name in self._alias_cache:
            actual_name = self._alias_cache[name]
            logger.debug(f"Resolved alias {name} -> {actual_name}")
            return actual_name

        # Not an alias, return as-is
        return name

    async def get_aliases_for_collection(self, collection_name: str) -> list[str]:
        """
        Get all aliases pointing to a specific collection.

        Args:
            collection_name: The actual collection name

        Returns:
            List of alias names pointing to this collection
        """
        if not self.state_manager or not self.state_manager.connection:
            return []

        try:
            with self.state_manager._lock:
                cursor = self.state_manager.connection.execute(
                    """
                    SELECT alias_name
                    FROM collection_aliases
                    WHERE collection_name = ?
                    ORDER BY created_at DESC
                    """,
                    (collection_name,)
                )

                rows = cursor.fetchall()
                return [row["alias_name"] for row in rows]

        except Exception as e:
            logger.error(f"Failed to get aliases for collection {collection_name}: {e}")
            return []

    async def _save_alias_to_db(self, alias: CollectionAlias) -> bool:
        """
        Save alias information to SQLite.

        Args:
            alias: CollectionAlias object to save

        Returns:
            True if saved successfully
        """
        if not self.state_manager:
            return False

        try:
            async with self.state_manager.transaction() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO collection_aliases
                    (alias_name, collection_name, created_at, created_by, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        alias.alias_name,
                        alias.collection_name,
                        alias.created_at.isoformat(),
                        alias.created_by,
                        self.state_manager._serialize_json(alias.metadata)
                    )
                )

            logger.debug(f"Saved alias to database: {alias.alias_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save alias to database: {e}")
            return False

    async def _delete_alias_from_db(self, alias_name: str) -> bool:
        """
        Delete alias information from SQLite.

        Args:
            alias_name: The alias name to delete

        Returns:
            True if deleted successfully
        """
        if not self.state_manager:
            return False

        try:
            async with self.state_manager.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM collection_aliases WHERE alias_name = ?",
                    (alias_name,)
                )

                deleted = cursor.rowcount > 0
                if deleted:
                    logger.debug(f"Deleted alias from database: {alias_name}")
                else:
                    logger.warning(f"Alias not found in database: {alias_name}")

                return deleted

        except Exception as e:
            logger.error(f"Failed to delete alias from database: {e}")
            return False

    async def _refresh_cache(self):
        """Refresh the in-memory alias cache from SQLite."""
        if not self.state_manager or not self.state_manager.connection:
            self._cache_valid = True  # No state manager, cache is always valid (empty)
            return

        try:
            aliases = await self.list_aliases()
            self._alias_cache = {
                alias.alias_name: alias.collection_name
                for alias in aliases
            }
            self._cache_valid = True
            logger.debug(f"Refreshed alias cache: {len(self._alias_cache)} aliases")

        except Exception as e:
            logger.error(f"Failed to refresh alias cache: {e}")
            self._cache_valid = False
