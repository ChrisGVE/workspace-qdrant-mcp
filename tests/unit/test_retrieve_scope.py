"""Unit tests for retrieve tool scope parameter (Task 398).

Tests multi-tenant filtering with scope parameter and branch filtering.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from workspace_qdrant_mcp.server import (
    CANONICAL_COLLECTIONS,
)


class TestRetrieveScopeValidation:
    """Tests for retrieve tool scope parameter validation."""

    @pytest.mark.asyncio
    async def test_invalid_scope_returns_error(self):
        """Test that invalid scope returns error response."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool

        # Get the underlying function
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"):
            result = await retrieve_fn(metadata={"test": "value"}, scope="invalid")

        assert result["success"] is False
        assert "Invalid scope" in result["error"]

    @pytest.mark.asyncio
    async def test_valid_scopes_accepted(self):
        """Test that valid scopes are accepted."""
        valid_scopes = ["project", "global", "all"]

        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        for scope in valid_scopes:
            with patch("workspace_qdrant_mcp.server.initialize_components"), \
                 patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
                 patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
                # Mock scroll to return empty results
                mock_client.scroll = AsyncMock(return_value=([], None))

                result = await retrieve_fn(metadata={"test": "value"}, scope=scope)

                # Should get past scope validation
                if not result["success"]:
                    assert "Invalid scope" not in result.get("error", "")
                else:
                    assert result["scope"] == scope

    @pytest.mark.asyncio
    async def test_default_scope_is_project(self):
        """Test that default scope is 'project'."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            # Don't specify scope - should default to "project"
            result = await retrieve_fn(metadata={"test": "value"})

            assert result["success"] is True
            assert result["scope"] == "project"


class TestRetrieveScopeCollections:
    """Tests for retrieve tool collection selection based on scope."""

    @pytest.mark.asyncio
    async def test_project_scope_uses_projects_collection(self):
        """Test that project scope searches only _projects collection."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            result = await retrieve_fn(metadata={"test": "value"}, scope="project")

            assert result["success"] is True
            assert result["collections_searched"] == [CANONICAL_COLLECTIONS["projects"]]
            assert result["filters_applied"]["project_id"] == "abc123def456"

    @pytest.mark.asyncio
    async def test_global_scope_no_project_filter(self):
        """Test that global scope doesn't apply project_id filter."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            result = await retrieve_fn(metadata={"test": "value"}, scope="global")

            assert result["success"] is True
            assert result["collections_searched"] == [CANONICAL_COLLECTIONS["projects"]]
            assert result["filters_applied"]["project_id"] is None

    @pytest.mark.asyncio
    async def test_all_scope_searches_multiple_collections(self):
        """Test that 'all' scope searches projects and libraries."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            result = await retrieve_fn(metadata={"test": "value"}, scope="all")

            assert result["success"] is True
            assert CANONICAL_COLLECTIONS["projects"] in result["collections_searched"]
            assert CANONICAL_COLLECTIONS["libraries"] in result["collections_searched"]
            assert result["filters_applied"]["project_id"] is None

    @pytest.mark.asyncio
    async def test_explicit_collection_overrides_scope(self):
        """Test that explicit collection parameter overrides scope."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            result = await retrieve_fn(
                metadata={"test": "value"},
                collection="custom_collection",
                scope="project"  # Should be ignored
            )

            assert result["success"] is True
            assert result["collections_searched"] == ["custom_collection"]


class TestRetrieveBranchFiltering:
    """Tests for retrieve tool branch filtering."""

    @pytest.mark.asyncio
    async def test_branch_filter_applied_to_metadata_retrieval(self):
        """Test that branch filter is applied to metadata-based retrieval."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"), \
             patch("workspace_qdrant_mcp.server.build_metadata_filters") as mock_build_filters:
            mock_client.scroll = AsyncMock(return_value=([], None))
            mock_build_filters.return_value = None

            result = await retrieve_fn(
                metadata={"author": "test"},
                branch="develop"
            )

            # Verify build_metadata_filters was called with branch
            mock_build_filters.assert_called()
            call_kwargs = mock_build_filters.call_args[1]
            assert call_kwargs.get("branch") == "develop"

    @pytest.mark.asyncio
    async def test_all_branches_retrieval(self):
        """Test retrieval with branch='*' (all branches)."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            result = await retrieve_fn(
                metadata={"test": "value"},
                branch="*"
            )

            assert result["success"] is True
            assert result["filters_applied"]["branch"] == "*"


class TestRetrieveRequiredParams:
    """Tests for retrieve tool required parameters."""

    @pytest.mark.asyncio
    async def test_requires_document_id_or_metadata(self):
        """Test that either document_id or metadata is required."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"):
            result = await retrieve_fn()  # No document_id or metadata

        assert result["success"] is False
        assert "Either document_id or metadata" in result["error"]

    @pytest.mark.asyncio
    async def test_document_id_only_works(self):
        """Test that document_id alone is sufficient."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            # Mock retrieve to return empty (not found)
            mock_client.retrieve = AsyncMock(return_value=[])

            result = await retrieve_fn(document_id="test-uuid-123")

            assert result["query_type"] == "id_lookup"

    @pytest.mark.asyncio
    async def test_metadata_only_works(self):
        """Test that metadata alone is sufficient."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            result = await retrieve_fn(metadata={"author": "test"})

            assert result["query_type"] == "metadata_filter"


class TestRetrieveResponseFormat:
    """Tests for retrieve tool response format."""

    @pytest.mark.asyncio
    async def test_response_includes_scope(self):
        """Test that response includes scope information."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            result = await retrieve_fn(metadata={"test": "value"}, scope="global")

            assert "scope" in result
            assert "collections_searched" in result
            assert "filters_applied" in result
            assert "project_id" in result["filters_applied"]

    @pytest.mark.asyncio
    async def test_response_includes_collections_searched(self):
        """Test that response includes list of collections searched."""
        from workspace_qdrant_mcp.server import retrieve as retrieve_tool
        retrieve_fn = retrieve_tool.fn if hasattr(retrieve_tool, 'fn') else retrieve_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"), \
             patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client, \
             patch("workspace_qdrant_mcp.server.calculate_tenant_id", return_value="abc123def456"):
            mock_client.scroll = AsyncMock(return_value=([], None))

            result = await retrieve_fn(metadata={"test": "value"}, scope="all")

            assert isinstance(result["collections_searched"], list)
            assert len(result["collections_searched"]) >= 2
