"""
Unit tests for Qdrant collection management operations (Task 284).

Tests collection lifecycle, statistics, project detection, naming conventions,
and multi-tenant isolation.

Test Subtasks:
- 284.1: Collection lifecycle tests (create, list, delete)
- 284.2: Collection statistics validation tests
- 284.3: Project-specific collection detection tests
- 284.4: Naming convention and edge case tests
- 284.5: Multi-tenant isolation and concurrency tests

Requirements:
- pytest-asyncio for async test support
- Mock Qdrant client for isolation
- Mock daemon client for write path testing
"""

import asyncio

# Import MCP server functions
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from qdrant_client.models import CollectionStatus, Distance, VectorParams
from workspace_qdrant_mcp.server import manage

# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for isolated testing."""
    client = MagicMock()

    # Mock get_collections response
    mock_collections = MagicMock()
    mock_collections.collections = []
    client.get_collections.return_value = mock_collections

    # Mock get_collection response
    mock_collection_info = MagicMock()
    mock_collection_info.points_count = 100
    mock_collection_info.segments_count = 1
    mock_collection_info.status = CollectionStatus.GREEN
    mock_collection_info.config = MagicMock()
    mock_collection_info.config.params = MagicMock()
    mock_collection_info.config.params.vectors = MagicMock()
    mock_collection_info.config.params.vectors.size = 384
    mock_collection_info.config.params.vectors.distance = Distance.COSINE
    client.get_collection.return_value = mock_collection_info

    # Mock create_collection
    client.create_collection.return_value = None

    # Mock delete_collection
    client.delete_collection.return_value = None

    return client


@pytest.fixture
def mock_daemon_client():
    """Mock daemon client for collection operations."""
    client = AsyncMock()

    # Mock create_collection_v2
    async def mock_create_collection(collection_name, vector_size, distance_metric):
        response = AsyncMock()
        response.success = True
        response.collection_name = collection_name
        return response

    client.create_collection_v2 = mock_create_collection

    return client


@pytest.fixture
def sample_collection_names():
    """Sample collection names for testing."""
    return [
        "test-project-code",
        "test-project-docs",
        "_myproject",
        "_library_requests",
        "user-notes",
        "scratchbook"
    ]


# ============================================================================
# COLLECTION LIFECYCLE TESTS (Task 284.1)
# ============================================================================

class TestCollectionCreate:
    """Test collection creation operations."""

    @pytest.mark.asyncio
    async def test_create_collection_basic(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test basic collection creation."""
        collection_name = "test-collection"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="create_collection",
                name=collection_name
            )

            assert result["success"] is True
            assert result["collection_name"] == collection_name

    @pytest.mark.asyncio
    async def test_create_collection_with_custom_config(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test collection creation with custom vector configuration."""
        collection_name = "custom-vector-collection"
        custom_config = {
            "vector_size": 768,
            "distance": Distance.DOT
        }

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="create_collection",
                name=collection_name,
                config=custom_config
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_collection_fallback_without_daemon(
        self, mock_qdrant_client
    ):
        """Test collection creation falls back to direct Qdrant if daemon unavailable."""
        collection_name = "fallback-collection"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', None), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="create_collection",
                name=collection_name
            )

            # Should fallback to direct Qdrant creation
            assert result["success"] is True
            mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_without_name_fails(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test that creating collection without name returns error."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="create_collection")

            assert result["success"] is False
            assert "error" in result


class TestCollectionList:
    """Test collection listing operations."""

    @pytest.mark.asyncio
    async def test_list_collections_empty(self, mock_qdrant_client):
        """Test listing collections when none exist."""
        mock_qdrant_client.get_collections.return_value.collections = []

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            assert result["total_collections"] == 0
            assert len(result["collections"]) == 0

    @pytest.mark.asyncio
    async def test_list_collections_multiple(
        self, mock_qdrant_client, sample_collection_names
    ):
        """Test listing multiple collections."""
        # Create mock collection objects
        mock_collections = []
        for name in sample_collection_names:
            mock_col = MagicMock()
            mock_col.name = name
            mock_collections.append(mock_col)

        mock_qdrant_client.get_collections.return_value.collections = mock_collections

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            assert result["total_collections"] == len(sample_collection_names)
            assert len(result["collections"]) == len(sample_collection_names)

    @pytest.mark.asyncio
    async def test_list_collections_includes_metadata(self, mock_qdrant_client):
        """Test that listed collections include metadata."""
        mock_col = MagicMock()
        mock_col.name = "test-collection"
        mock_qdrant_client.get_collections.return_value.collections = [mock_col]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            collection = result["collections"][0]
            assert "name" in collection
            assert "points_count" in collection
            assert "status" in collection


class TestCollectionDelete:
    """Test collection deletion operations."""

    @pytest.mark.asyncio
    async def test_delete_collection_basic(self, mock_qdrant_client):
        """Test basic collection deletion."""
        collection_name = "collection-to-delete"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', None), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="delete_collection",
                name=collection_name
            )

            assert result["success"] is True
            mock_qdrant_client.delete_collection.assert_called_once_with(collection_name)

    @pytest.mark.asyncio
    async def test_delete_collection_without_name_fails(self, mock_qdrant_client):
        """Test that deleting without collection name fails."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="delete_collection")

            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection_handling(self, mock_qdrant_client):
        """Test deleting non-existent collection."""
        # Mock raises exception for non-existent collection
        mock_qdrant_client.delete_collection.side_effect = Exception("Collection not found")

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="delete_collection",
                name="nonexistent-collection"
            )

            assert result["success"] is False


# ============================================================================
# COLLECTION STATISTICS TESTS (Task 284.2)
# ============================================================================

class TestCollectionStatistics:
    """Test collection statistics retrieval and validation."""

    @pytest.mark.asyncio
    async def test_collection_info_retrieval(self, mock_qdrant_client):
        """Test retrieving detailed collection information."""
        collection_name = "test-collection"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="collection_info",
                name=collection_name
            )

            assert result["success"] is True
            assert "collection" in result
            assert result["collection"]["name"] == collection_name

    @pytest.mark.asyncio
    async def test_collection_points_count(self, mock_qdrant_client):
        """Test collection points count is included in statistics."""
        mock_col = MagicMock()
        mock_col.name = "stats-collection"
        mock_qdrant_client.get_collections.return_value.collections = [mock_col]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            collection = result["collections"][0]
            assert "points_count" in collection
            assert collection["points_count"] == 100

    @pytest.mark.asyncio
    async def test_collection_vector_configuration(self, mock_qdrant_client):
        """Test vector configuration is included in collection info."""
        mock_col = MagicMock()
        mock_col.name = "vector-config-collection"
        mock_qdrant_client.get_collections.return_value.collections = [mock_col]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            collection = result["collections"][0]
            assert "vector_size" in collection
            assert collection["vector_size"] == 384
            assert "distance" in collection

    @pytest.mark.asyncio
    async def test_collection_status_reporting(self, mock_qdrant_client):
        """Test collection status is reported correctly."""
        mock_col = MagicMock()
        mock_col.name = "status-collection"
        mock_qdrant_client.get_collections.return_value.collections = [mock_col]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            collection = result["collections"][0]
            assert "status" in collection
            # CollectionStatus.GREEN.value should be in result
            assert collection["status"] == "green"


# ============================================================================
# PROJECT-SPECIFIC COLLECTION DETECTION TESTS (Task 284.3)
# ============================================================================

class TestProjectCollectionDetection:
    """Test project-specific collection identification and handling."""

    @pytest.mark.asyncio
    async def test_detect_project_collections(self, mock_qdrant_client):
        """Test identification of project collections (starting with _)."""
        project_collections = [
            "_myproject",
            "_anotherproject",
            "_library_requests"
        ]

        mock_collections = []
        for name in project_collections:
            mock_col = MagicMock()
            mock_col.name = name
            mock_collections.append(mock_col)

        mock_qdrant_client.get_collections.return_value.collections = mock_collections

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            # All collections should start with underscore
            for collection in result["collections"]:
                assert collection["name"].startswith("_")

    @pytest.mark.asyncio
    async def test_detect_user_collections(self, mock_qdrant_client):
        """Test identification of user/non-project collections."""
        user_collections = [
            "user-notes",
            "scratchbook",
            "custom-collection"
        ]

        mock_collections = []
        for name in user_collections:
            mock_col = MagicMock()
            mock_col.name = name
            mock_collections.append(mock_col)

        mock_qdrant_client.get_collections.return_value.collections = mock_collections

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            # No collections should start with underscore
            for collection in result["collections"]:
                assert not collection["name"].startswith("_")

    @pytest.mark.asyncio
    async def test_mixed_project_and_user_collections(
        self, mock_qdrant_client, sample_collection_names
    ):
        """Test handling mixed project and user collections."""
        mock_collections = []
        for name in sample_collection_names:
            mock_col = MagicMock()
            mock_col.name = name
            mock_collections.append(mock_col)

        mock_qdrant_client.get_collections.return_value.collections = mock_collections

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            # Should have both types
            project_cols = [c for c in result["collections"] if c["name"].startswith("_")]
            user_cols = [c for c in result["collections"] if not c["name"].startswith("_")]

            assert len(project_cols) > 0
            assert len(user_cols) > 0


# ============================================================================
# NAMING CONVENTION AND EDGE CASE TESTS (Task 284.4)
# ============================================================================

class TestNamingConventions:
    """Test collection naming conventions and validation."""

    @pytest.mark.asyncio
    async def test_create_collection_with_hyphens(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test creating collection with hyphens in name."""
        collection_name = "my-test-collection"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="create_collection",
                name=collection_name
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_collection_with_underscores(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test creating collection with underscores (project collection)."""
        collection_name = "_my_project_collection"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="create_collection",
                name=collection_name
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_collection_with_numbers(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test creating collection with numbers in name."""
        collection_name = "collection-123-test"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="create_collection",
                name=collection_name
            )

            assert result["success"] is True


class TestEdgeCases:
    """Test edge cases in collection management."""

    @pytest.mark.asyncio
    async def test_very_long_collection_name(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test creating collection with very long name."""
        long_name = "a" * 200  # Very long name

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="create_collection",
                name=long_name
            )

            # Should succeed (Qdrant handles validation)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_collection_with_special_prefixes(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test collections with special prefixes."""
        special_names = [
            "memory",
            "_library_fastapi",
            "_agent_memory"
        ]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            for name in special_names:
                result = await manage(
                    action="create_collection",
                    name=name
                )

                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_collection_info_for_nonexistent(self, mock_qdrant_client):
        """Test getting info for non-existent collection."""
        mock_qdrant_client.get_collection.side_effect = Exception("Collection not found")

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="collection_info",
                name="nonexistent"
            )

            assert result["success"] is False


# ============================================================================
# MULTI-TENANT ISOLATION AND CONCURRENCY TESTS (Task 284.5)
# ============================================================================

class TestMultiTenantIsolation:
    """Test multi-tenant collection isolation."""

    @pytest.mark.asyncio
    async def test_project_collections_isolated(self, mock_qdrant_client):
        """Test that different project collections are isolated."""
        project_collections = [
            "_project_a",
            "_project_b",
            "_project_c"
        ]

        mock_collections = []
        for name in project_collections:
            mock_col = MagicMock()
            mock_col.name = name
            mock_collections.append(mock_col)

        mock_qdrant_client.get_collections.return_value.collections = mock_collections

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True
            # All should be project collections
            assert all(c["name"].startswith("_") for c in result["collections"])
            # All should have different names (isolated)
            names = [c["name"] for c in result["collections"]]
            assert len(names) == len(set(names))  # All unique

    @pytest.mark.asyncio
    async def test_user_and_project_collections_separate(
        self, mock_qdrant_client
    ):
        """Test that user and project collections remain separate."""
        collections = [
            "_project_workspace",  # Project collection
            "user-notes",          # User collection
            "_library_requests",   # Library collection
            "scratchbook"          # User collection
        ]

        mock_collections = []
        for name in collections:
            mock_col = MagicMock()
            mock_col.name = name
            mock_collections.append(mock_col)

        mock_qdrant_client.get_collections.return_value.collections = mock_collections

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True

            project_cols = [c for c in result["collections"] if c["name"].startswith("_")]
            user_cols = [c for c in result["collections"] if not c["name"].startswith("_")]

            # Should have both types, properly separated
            assert len(project_cols) == 2
            assert len(user_cols) == 2


class TestConcurrentOperations:
    """Test concurrent collection operations."""

    @pytest.mark.asyncio
    async def test_concurrent_collection_creation(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test creating multiple collections concurrently."""
        collection_names = [
            f"concurrent-collection-{i}" for i in range(10)
        ]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            tasks = [
                manage(action="create_collection", name=name)
                for name in collection_names
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_list_operations(self, mock_qdrant_client):
        """Test multiple concurrent list operations."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            tasks = [manage(action="list_collections") for _ in range(20)]

            results = await asyncio.gather(*tasks)

            assert len(results) == 20
            assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_create_and_list(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test concurrent create and list operations."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            # Mix of create and list operations
            tasks = []
            for i in range(5):
                tasks.append(manage(action="create_collection", name=f"col-{i}"))
                tasks.append(manage(action="list_collections"))

            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_delete_operations(self, mock_qdrant_client):
        """Test concurrent collection deletion."""
        collection_names = [f"delete-col-{i}" for i in range(5)]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            tasks = [
                manage(action="delete_collection", name=name)
                for name in collection_names
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 5
            assert all(r["success"] for r in results)


# ============================================================================
# WORKSPACE STATUS AND CLEANUP TESTS
# ============================================================================

class TestWorkspaceManagement:
    """Test workspace status and cleanup operations."""

    @pytest.mark.asyncio
    async def test_workspace_status_reporting(self, mock_qdrant_client):
        """Test workspace status action returns system health."""
        mock_col = MagicMock()
        mock_col.name = "test-collection"
        mock_qdrant_client.get_collections.return_value.collections = [mock_col]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="workspace_status")

            assert result["success"] is True
            assert "collections" in result
            assert "health_status" in result

    @pytest.mark.asyncio
    async def test_cleanup_operation(self, mock_qdrant_client):
        """Test cleanup action for removing empty collections."""
        # Mock empty collection
        mock_empty_col = MagicMock()
        mock_empty_col.name = "empty-collection"

        empty_info = MagicMock()
        empty_info.points_count = 0
        mock_qdrant_client.get_collection.return_value = empty_info
        mock_qdrant_client.get_collections.return_value.collections = [mock_empty_col]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="cleanup")

            # Cleanup should identify and potentially remove empty collections
            assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
