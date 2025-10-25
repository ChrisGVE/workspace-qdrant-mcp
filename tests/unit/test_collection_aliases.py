"""
Unit tests for collection alias system.

Tests the collection alias management functionality including:
- Alias creation and deletion
- SQLite persistence
- Alias resolution
- Integration with AliasManager
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.python.common.core.collection_aliases import AliasManager, CollectionAlias
from src.python.common.core.sqlite_state_manager import SQLiteStateManager


class TestCollectionAlias:
    """Test CollectionAlias dataclass."""

    def test_collection_alias_creation(self):
        """Test creating a CollectionAlias object."""
        alias = CollectionAlias(
            alias_name="_path_abc123def456",
            collection_name="_github_com_user_repo",
            created_by="cli",
            metadata={"reason": "remote_update"}
        )

        assert alias.alias_name == "_path_abc123def456"
        assert alias.collection_name == "_github_com_user_repo"
        assert alias.created_by == "cli"
        assert alias.metadata["reason"] == "remote_update"
        assert isinstance(alias.created_at, datetime)

    def test_collection_alias_auto_timestamp(self):
        """Test that created_at is auto-generated if not provided."""
        alias = CollectionAlias(
            alias_name="_old",
            collection_name="_new"
        )

        assert alias.created_at is not None
        assert isinstance(alias.created_at, datetime)

    def test_collection_alias_with_explicit_timestamp(self):
        """Test CollectionAlias with explicitly provided timestamp."""
        now = datetime.now(timezone.utc)
        alias = CollectionAlias(
            alias_name="_old",
            collection_name="_new",
            created_at=now
        )

        assert alias.created_at == now


class TestAliasManager:
    """Test AliasManager functionality."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = Mock()
        client.update_collection_aliases = Mock()
        return client

    @pytest.fixture
    def mock_state_manager(self):
        """Create a mock SQLite state manager."""
        state_manager = Mock(spec=SQLiteStateManager)
        state_manager.connection = Mock()
        # Use MagicMock for _lock to support context manager protocol (with statement)
        state_manager._lock = MagicMock()
        state_manager._serialize_json = Mock(side_effect=lambda x: str(x) if x else None)
        state_manager._deserialize_json = Mock(side_effect=lambda x: eval(x) if x else None)
        return state_manager

    @pytest.fixture
    async def alias_manager(self, mock_qdrant_client, mock_state_manager):
        """Create an AliasManager instance with mocked dependencies."""
        manager = AliasManager(mock_qdrant_client, mock_state_manager)
        return manager

    @pytest.mark.asyncio
    async def test_alias_manager_initialization(self, alias_manager):
        """Test AliasManager initialization."""
        assert alias_manager.qdrant_client is not None
        assert alias_manager.state_manager is not None
        assert alias_manager._alias_cache == {}
        assert alias_manager._cache_valid is False

    @pytest.mark.asyncio
    async def test_create_alias_success(self, alias_manager, mock_qdrant_client, mock_state_manager):
        """Test successful alias creation."""
        # Mock transaction context manager
        mock_conn = Mock()
        mock_state_manager.transaction = AsyncMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn)))

        await alias_manager.create_alias(
            alias_name="_old_collection",
            collection_name="_new_collection",
            created_by="test",
            metadata={"reason": "testing"}
        )

        # Verify Qdrant client was called
        mock_qdrant_client.update_collection_aliases.assert_called_once()

        # Verify cache was updated
        assert "_old_collection" in alias_manager._alias_cache
        assert alias_manager._alias_cache["_old_collection"] == "_new_collection"

    @pytest.mark.asyncio
    async def test_delete_alias_success(self, alias_manager, mock_qdrant_client, mock_state_manager):
        """Test successful alias deletion."""
        # Setup: Add alias to cache first
        alias_manager._alias_cache["_old_collection"] = "_new_collection"

        # Mock transaction context manager
        mock_conn = Mock()
        mock_state_manager.transaction = AsyncMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn)))

        await alias_manager.delete_alias("_old_collection")

        # Verify Qdrant client was called
        mock_qdrant_client.update_collection_aliases.assert_called_once()

        # Verify cache was updated
        assert "_old_collection" not in alias_manager._alias_cache

    @pytest.mark.asyncio
    async def test_resolve_collection_name_with_alias(self, alias_manager):
        """Test resolving a collection name that is an alias."""
        # Setup cache with alias
        alias_manager._alias_cache["_old_collection"] = "_new_collection"
        alias_manager._cache_valid = True

        resolved = await alias_manager.resolve_collection_name("_old_collection")
        assert resolved == "_new_collection"

    @pytest.mark.asyncio
    async def test_resolve_collection_name_without_alias(self, alias_manager):
        """Test resolving a collection name that is NOT an alias."""
        alias_manager._cache_valid = True

        resolved = await alias_manager.resolve_collection_name("_direct_collection")
        assert resolved == "_direct_collection"

    @pytest.mark.asyncio
    async def test_list_aliases_empty(self, alias_manager, mock_state_manager):
        """Test listing aliases when none exist."""
        # Mock empty database query
        mock_cursor = Mock()
        mock_cursor.fetchall = Mock(return_value=[])
        mock_state_manager.connection.execute = Mock(return_value=mock_cursor)

        aliases = await alias_manager.list_aliases()
        assert aliases == []

    @pytest.mark.asyncio
    async def test_list_aliases_with_data(self, alias_manager, mock_state_manager):
        """Test listing aliases with existing data."""
        # Mock database query with data
        mock_cursor = Mock()
        mock_row = {
            "alias_name": "_old",
            "collection_name": "_new",
            "created_at": "2025-10-03T10:00:00+00:00",
            "created_by": "cli",
            "metadata": "{}"
        }
        mock_cursor.fetchall = Mock(return_value=[mock_row])
        mock_state_manager.connection.execute = Mock(return_value=mock_cursor)

        aliases = await alias_manager.list_aliases()

        assert len(aliases) == 1
        assert aliases[0].alias_name == "_old"
        assert aliases[0].collection_name == "_new"
        assert aliases[0].created_by == "cli"

    @pytest.mark.asyncio
    async def test_get_alias_exists(self, alias_manager, mock_state_manager):
        """Test getting an alias that exists."""
        # Mock database query
        mock_cursor = Mock()
        mock_row = {
            "alias_name": "_old",
            "collection_name": "_new",
            "created_at": "2025-10-03T10:00:00+00:00",
            "created_by": "test",
            "metadata": "{}"
        }
        mock_cursor.fetchone = Mock(return_value=mock_row)
        mock_state_manager.connection.execute = Mock(return_value=mock_cursor)

        alias = await alias_manager.get_alias("_old")

        assert alias is not None
        assert alias.alias_name == "_old"
        assert alias.collection_name == "_new"

    @pytest.mark.asyncio
    async def test_get_alias_not_exists(self, alias_manager, mock_state_manager):
        """Test getting an alias that doesn't exist."""
        # Mock database query with no results
        mock_cursor = Mock()
        mock_cursor.fetchone = Mock(return_value=None)
        mock_state_manager.connection.execute = Mock(return_value=mock_cursor)

        alias = await alias_manager.get_alias("_nonexistent")
        assert alias is None

    @pytest.mark.asyncio
    async def test_get_aliases_for_collection(self, alias_manager, mock_state_manager):
        """Test getting all aliases pointing to a specific collection."""
        # Mock database query
        mock_cursor = Mock()
        mock_rows = [
            {"alias_name": "_old1"},
            {"alias_name": "_old2"}
        ]
        mock_cursor.fetchall = Mock(return_value=mock_rows)
        mock_state_manager.connection.execute = Mock(return_value=mock_cursor)

        aliases = await alias_manager.get_aliases_for_collection("_new_collection")

        assert len(aliases) == 2
        assert "_old1" in aliases
        assert "_old2" in aliases

    @pytest.mark.asyncio
    async def test_create_alias_error_handling(self, alias_manager, mock_qdrant_client):
        """Test error handling when alias creation fails."""
        # Make Qdrant client raise an exception
        mock_qdrant_client.update_collection_aliases.side_effect = Exception("Qdrant error")

        with pytest.raises(Exception) as exc_info:
            await alias_manager.create_alias("_old", "_new")

        assert "Qdrant error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_alias_error_handling(self, alias_manager, mock_qdrant_client):
        """Test error handling when alias deletion fails."""
        # Make Qdrant client raise an exception
        mock_qdrant_client.update_collection_aliases.side_effect = Exception("Delete error")

        with pytest.raises(Exception) as exc_info:
            await alias_manager.delete_alias("_old")

        assert "Delete error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, alias_manager):
        """Test that cache is refreshed when invalid."""
        alias_manager._cache_valid = False
        alias_manager.list_aliases = AsyncMock(return_value=[
            CollectionAlias("_old1", "_new1"),
            CollectionAlias("_old2", "_new2")
        ])

        # This should trigger cache refresh
        await alias_manager.resolve_collection_name("_old1")

        # Verify list_aliases was called to refresh cache
        alias_manager.list_aliases.assert_called_once()


class TestAliasIntegration:
    """Integration tests for alias system."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_alias_roundtrip(self, tmp_path):
        """Test creating, retrieving, and deleting an alias end-to-end."""
        # This test would require actual Qdrant and SQLite instances
        # Skip in unit tests, implement in integration tests
        pytest.skip("Integration test - requires real Qdrant and SQLite")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_alias_persistence_across_restarts(self, tmp_path):
        """Test that aliases persist across AliasManager restarts."""
        # This test would require actual SQLite database
        # Skip in unit tests, implement in integration tests
        pytest.skip("Integration test - requires real SQLite database")
