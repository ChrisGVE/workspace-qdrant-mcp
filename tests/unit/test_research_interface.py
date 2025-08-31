"""
Unit tests for four-mode research interface.

Tests the implementation of Task 13 from PRD v2.0 specifications.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workspace_qdrant_mcp.tools.research import research_workspace


class TestResearchInterface:
    """Test four-mode research interface functionality."""

    @pytest.mark.asyncio
    async def test_research_project_mode(self):
        """Test project mode research (default)."""
        # Mock workspace client with proper async setup
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.project_info = {
            "main_project": "test-project",
            "subprojects": ["sub1", "sub2"]
        }

        # Mock embedding service and its async methods
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings = AsyncMock(return_value={
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.5, 0.3]}
        })

        mock_client.get_embedding_service.return_value = mock_embedding_service

        # Mock search results
        mock_results = {
            "results": [{"id": "doc1", "score": 0.9, "payload": {"title": "Test"}}],
            "total_results": 1
        }

        with patch("workspace_qdrant_mcp.tools.research.search_workspace", new_callable=AsyncMock, return_value=mock_results) as mock_search:

            result = await research_workspace(
                client=mock_client,
                query="test query",
                mode="project"
            )

        assert "error" not in result
        assert result["research_context"]["mode"] == "project"

        # Check that search was called with project collections
        call_args = mock_search.call_args
        collections = call_args.kwargs["collections"]
        expected_collections = [
            "test-project-scratchbook", "test-project-docs",
            "sub1-scratchbook", "sub1-docs",
            "sub2-scratchbook", "sub2-docs"
        ]
        assert collections == expected_collections

    @pytest.mark.asyncio
    async def test_research_collection_mode(self):
        """Test collection mode research."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_results = {"results": [], "total_results": 0}

        with patch("workspace_qdrant_mcp.tools.research.search_workspace", new_callable=AsyncMock, return_value=mock_results) as mock_search:

            result = await research_workspace(
                client=mock_client,
                query="test query",
                mode="collection",
                target_collection="specific-collection"
            )

        assert "error" not in result
        assert result["research_context"]["mode"] == "collection"
        assert result["research_context"]["target_collection"] == "specific-collection"

        # Check that search was called with target collection only
        call_args = mock_search.call_args
        collections = call_args.kwargs["collections"]
        assert collections == ["specific-collection"]

    @pytest.mark.asyncio
    async def test_research_collection_mode_missing_target(self):
        """Test collection mode fails without target_collection."""
        mock_client = MagicMock()
        mock_client.initialized = True

        result = await research_workspace(
            client=mock_client,
            query="test query",
            mode="collection"
            # Missing target_collection
        )

        assert "error" in result
        assert "target_collection required" in result["error"]

    @pytest.mark.asyncio
    async def test_research_global_mode(self):
        """Test global mode research."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.config.workspace = MagicMock()
        mock_client.config.workspace.global_collections = ["memory", "_technical-books"]
        mock_results = {"results": [], "total_results": 0}

        with patch("workspace_qdrant_mcp.tools.research.search_workspace", new_callable=AsyncMock, return_value=mock_results) as mock_search:

            result = await research_workspace(
                client=mock_client,
                query="test query",
                mode="global"
            )

        assert "error" not in result
        assert result["research_context"]["mode"] == "global"

        # Check that search was called with global collections
        call_args = mock_search.call_args
        collections = call_args.kwargs["collections"]
        assert collections == ["memory", "_technical-books"]

    @pytest.mark.asyncio
    async def test_research_global_mode_default_collections(self):
        """Test global mode with default collections when not configured."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.config.workspace = MagicMock()
        # No global_collections configured - set to None
        mock_client.config.workspace.global_collections = None

        mock_results = {"results": [], "total_results": 0}

        with patch("workspace_qdrant_mcp.tools.research.search_workspace", new_callable=AsyncMock, return_value=mock_results) as mock_search:

            result = await research_workspace(
                client=mock_client,
                query="test query",
                mode="global"
            )

        # Check that default global collections were used
        assert "error" not in result
        call_args = mock_search.call_args
        collections = call_args.kwargs["collections"]
        assert collections == ["memory", "_technical-books", "_standards"]

    @pytest.mark.asyncio
    async def test_research_all_mode(self):
        """Test all collections mode research."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.list_collections = AsyncMock(return_value=["coll1", "coll2", "archived_archive"])
        mock_results = {"results": [], "total_results": 0}

        with patch("workspace_qdrant_mcp.tools.research.search_workspace", new_callable=AsyncMock, return_value=mock_results) as mock_search:

            result = await research_workspace(
                client=mock_client,
                query="test query",
                mode="all"
            )

        assert "error" not in result
        assert result["research_context"]["mode"] == "all"

        # Check that search was called with all collections (archived_archive should be filtered out since it ends with _archive)
        call_args = mock_search.call_args
        collections = call_args.kwargs["collections"]
        assert collections == ["coll1", "coll2"]  # archived_archive filtered out by default (include_archived=False)

    @pytest.mark.asyncio
    async def test_research_all_mode_with_archived(self):
        """Test all collections mode including archived."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.list_collections = AsyncMock(return_value=["coll1", "coll2"])
        mock_results = {"results": [], "total_results": 0}

        with patch("workspace_qdrant_mcp.tools.research.search_workspace", new_callable=AsyncMock, return_value=mock_results) as mock_search:

            result = await research_workspace(
                client=mock_client,
                query="test query",
                mode="all",
                include_archived=True
            )

        # When include_archived=True, it should include all collections (no filtering)
        # In this case none end with _archive so should be same as without filtering
        assert "error" not in result
        call_args = mock_search.call_args
        collections = call_args.kwargs["collections"]
        assert collections == ["coll1", "coll2"]  # No archived collections in the mock

    @pytest.mark.asyncio
    async def test_version_filtering_latest(self):
        """Test version preference filtering for latest only."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.project_info = {"main_project": "test", "subprojects": []}

        mock_results = {
            "results": [
                {"payload": {"is_latest": True, "title": "Latest doc"}},
                {"payload": {"is_latest": False, "title": "Old doc"}},
                {"payload": {"title": "No version info"}}  # Should default to True
            ],
            "total_results": 3
        }

        with patch("workspace_qdrant_mcp.tools.research.search_workspace", new_callable=AsyncMock, return_value=mock_results):

            result = await research_workspace(
                client=mock_client,
                query="test query",
                version_preference="latest"
            )

        # Should filter to only latest versions
        assert "error" not in result
        assert len(result["results"]) == 2  # Latest doc + no version info (defaults to latest)
        assert result["total_results"] == 2

        # Check that non-latest document was filtered out
        titles = [r["payload"]["title"] for r in result["results"]]
        assert "Old doc" not in titles
        assert "Latest doc" in titles
        assert "No version info" in titles

    @pytest.mark.asyncio
    async def test_relationship_information(self):
        """Test inclusion of version relationship information."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.project_info = {"main_project": "test", "subprojects": []}

        mock_results = {
            "results": [
                {
                    "payload": {
                        "version": "2.0.0",
                        "is_latest": True,
                        "supersedes": ["old-doc-id"],
                        "document_type": "book"
                    }
                }
            ]
        }

        with patch("workspace_qdrant_mcp.tools.research.search_workspace", new_callable=AsyncMock, return_value=mock_results):

            result = await research_workspace(
                client=mock_client,
                query="test query",
                include_relationships=True
            )

        # Should include version_info in results
        assert "error" not in result
        assert "version_info" in result["results"][0]
        version_info = result["results"][0]["version_info"]
        assert version_info["version"] == "2.0.0"
        assert version_info["is_latest"] is True
        assert version_info["supersedes"] == ["old-doc-id"]
        assert version_info["document_type"] == "book"

    @pytest.mark.asyncio
    async def test_invalid_mode(self):
        """Test error handling for invalid research mode."""
        mock_client = MagicMock()
        mock_client.initialized = True

        result = await research_workspace(
            client=mock_client,
            query="test query",
            mode="invalid_mode"
        )

        assert "error" in result
        assert "Invalid mode 'invalid_mode'" in result["error"]
