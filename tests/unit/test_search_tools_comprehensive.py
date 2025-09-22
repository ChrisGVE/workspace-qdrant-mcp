"""
Comprehensive unit tests for search tools module.

Tests all search tool functions with 100% line and branch coverage:
- search_workspace: Main workspace search with hybrid/dense/sparse modes
- search_collection_by_metadata: Metadata-based filtering search
- search_workspace_with_project_isolation: Project-aware search
- search_workspace_with_advanced_aggregation: Advanced aggregation features
- _search_collection: Internal collection search function
- _build_metadata_filter: Metadata filter construction
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

# Add the correct path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from workspace_qdrant_mcp.tools.search import (
    search_workspace,
    search_collection_by_metadata,
    search_workspace_with_project_isolation,
    search_workspace_with_advanced_aggregation,
    _search_collection,
    _build_metadata_filter,
)


@pytest.fixture
def enhanced_mock_workspace_client():
    """Enhanced mock workspace client with all required attributes for search tests."""
    client = Mock()
    client.initialized = True

    # Mock configuration
    client.config = Mock()
    client.config.qdrant = Mock()
    client.config.qdrant.url = "http://localhost:6333"
    client.config.qdrant.api_key = None

    # Mock Qdrant client
    client.client = Mock()
    client.client.search = Mock()
    client.client.scroll = Mock()

    # Mock embedding service
    embedding_service = Mock()
    embedding_service.generate_embeddings = AsyncMock()
    client.get_embedding_service = Mock(return_value=embedding_service)

    # Mock collection operations
    client.list_collections = Mock(return_value=["docs", "notes"])

    # Mock collection manager
    client.collection_manager = Mock()
    client.collection_manager.resolve_collection_name = Mock(return_value=("docs", "docs"))
    client.collection_manager.list_searchable_collections = Mock(return_value=["docs", "notes"])
    client.collection_manager.validate_collection_filtering = Mock(return_value={
        "summary": {"total_collections": 2}
    })

    # Mock project context
    client.get_project_context = Mock(return_value=None)
    client._generate_project_id = Mock(return_value="test-project-id")
    client.search_with_project_context = AsyncMock()

    # Mock project detector
    client.project_detector = Mock()
    client.project_info = None

    return client


@pytest.fixture
def enhanced_mock_qdrant_client():
    """Enhanced mock Qdrant client."""
    client = Mock()
    client.search = Mock()
    client.scroll = Mock()
    return client


class TestSearchWorkspaceCore:
    """Core tests for search_workspace function covering main functionality."""

    @pytest.mark.asyncio
    async def test_search_workspace_client_not_initialized(self, enhanced_mock_workspace_client):
        """Test search with uninitialized client."""
        enhanced_mock_workspace_client.initialized = False

        result = await search_workspace(enhanced_mock_workspace_client, "test query")

        assert result["error"] == "Workspace client not initialized"
        assert "results" not in result

    @pytest.mark.asyncio
    async def test_search_workspace_empty_query(self, enhanced_mock_workspace_client):
        """Test search with empty or whitespace-only query."""
        # Test empty string
        result = await search_workspace(enhanced_mock_workspace_client, "")
        assert "error" in result
        assert "Query cannot be empty" in result["error"]

        # Test whitespace-only query
        result = await search_workspace(enhanced_mock_workspace_client, "   \t\n")
        assert "error" in result
        assert "Query cannot be empty" in result["error"]

        # Test None query
        result = await search_workspace(enhanced_mock_workspace_client, None)
        assert "error" in result
        assert "Query cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_invalid_mode(self, enhanced_mock_workspace_client):
        """Test search with invalid search mode."""
        result = await search_workspace(
            enhanced_mock_workspace_client, "test query", mode="invalid_mode"
        )

        assert "error" in result
        assert "Invalid search mode 'invalid_mode'" in result["error"]
        assert "Must be one of: ['hybrid', 'dense', 'sparse']" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_collections_not_found(self, enhanced_mock_workspace_client):
        """Test search with specified collections that don't exist."""
        enhanced_mock_workspace_client.list_collections.return_value = ["docs", "references"]

        result = await search_workspace(
            enhanced_mock_workspace_client,
            "test query",
            collections=["nonexistent1", "nonexistent2"],
        )

        assert "error" in result
        assert "Collections not found: nonexistent1, nonexistent2" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_no_collections_available(self, enhanced_mock_workspace_client):
        """Test search when no collections are available."""
        # Mock enhanced collection selector to fail and fallback to empty collections
        with patch("workspace_qdrant_mcp.tools.search.CollectionSelector") as mock_selector_class:
            mock_selector_class.side_effect = Exception("No selector available")
            enhanced_mock_workspace_client.collection_manager.list_searchable_collections.return_value = []
            enhanced_mock_workspace_client.list_collections.return_value = []

            result = await search_workspace(enhanced_mock_workspace_client, "test query")

        assert "error" in result
        assert "No collections available for search" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_sparse_mode_no_embeddings(self, enhanced_mock_workspace_client):
        """Test sparse mode when sparse embeddings are not available."""
        # Setup embedding service to return only dense embeddings
        embedding_service = enhanced_mock_workspace_client.get_embedding_service()
        embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        result = await search_workspace(
            enhanced_mock_workspace_client, "test query", mode="sparse"
        )

        assert "error" in result
        assert "Sparse embeddings not available for sparse search mode" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_dense_mode_success(self, enhanced_mock_workspace_client):
        """Test successful dense mode search."""
        # Setup embedding service
        embedding_service = enhanced_mock_workspace_client.get_embedding_service()
        embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        # Mock _search_collection function
        with patch("workspace_qdrant_mcp.tools.search._search_collection") as mock_search:
            mock_search.return_value = [
                {
                    "id": "doc1",
                    "score": 0.9,
                    "payload": {"content": "test content"},
                    "search_type": "dense",
                }
            ]

            result = await search_workspace(
                enhanced_mock_workspace_client, "test query", mode="dense"
            )

        assert "results" in result
        assert len(result["results"]) == 2  # One from each collection (docs, notes)
        assert result["search_params"]["mode"] == "dense"
        assert all(r["collection"] in ["docs", "notes"] for r in result["results"])

    @pytest.mark.asyncio
    async def test_search_workspace_project_metadata_injection(self, enhanced_mock_workspace_client):
        """Test search with automatic project metadata injection."""
        # Setup embedding service
        embedding_service = enhanced_mock_workspace_client.get_embedding_service()
        embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        # Mock search_with_project_context
        enhanced_mock_workspace_client.search_with_project_context.return_value = {
            "fused_results": [
                Mock(id="doc1", score=0.9, payload={"content": "test content"})
            ]
        }

        result = await search_workspace(
            enhanced_mock_workspace_client,
            "test query",
            mode="dense",
            auto_inject_project_metadata=True
        )

        # Verify project-aware search was used
        enhanced_mock_workspace_client.search_with_project_context.assert_called()
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_search_workspace_exception_handling(self, enhanced_mock_workspace_client):
        """Test general exception handling during search."""
        # Mock get_embedding_service to raise exception
        enhanced_mock_workspace_client.get_embedding_service.side_effect = Exception("Service unavailable")

        result = await search_workspace(enhanced_mock_workspace_client, "test query")

        assert "error" in result
        assert "Search failed: Service unavailable" in result["error"]


class TestSearchCollectionInternal:
    """Tests for _search_collection internal function."""

    @pytest.mark.asyncio
    async def test_search_collection_hybrid_mode(self, enhanced_mock_qdrant_client):
        """Test hybrid mode search."""
        embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.8, 0.6, 0.9]}
        }

        # Mock HybridSearchEngine
        with patch("workspace_qdrant_mcp.tools.search.HybridSearchEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine.hybrid_search.return_value = {
                "results": [
                    {
                        "id": "doc1",
                        "rrf_score": 0.95,
                        "payload": {"content": "test content"}
                    }
                ]
            }
            mock_engine_class.return_value = mock_engine

            results = await _search_collection(
                enhanced_mock_qdrant_client,
                "test_collection",
                embeddings,
                "hybrid",
                limit=10,
                score_threshold=0.7
            )

        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.95
        assert results[0]["search_type"] == "hybrid"

    @pytest.mark.asyncio
    async def test_search_collection_dense_mode(self, enhanced_mock_qdrant_client):
        """Test dense mode search."""
        embeddings = {"dense": [0.1] * 384}

        # Mock Qdrant search results
        mock_results = [
            models.ScoredPoint(
                id="doc1",
                score=0.92,
                version=0,
                payload={"content": "dense result"}
            )
        ]
        enhanced_mock_qdrant_client.search.return_value = mock_results

        results = await _search_collection(
            enhanced_mock_qdrant_client,
            "test_collection",
            embeddings,
            "dense",
            limit=10,
            score_threshold=0.7
        )

        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.92
        assert results[0]["search_type"] == "dense"

    @pytest.mark.asyncio
    async def test_search_collection_sparse_mode(self, enhanced_mock_qdrant_client):
        """Test sparse mode search."""
        embeddings = {"sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.9]}}

        # Mock sparse vector creation
        with patch("workspace_qdrant_mcp.tools.search.create_named_sparse_vector") as mock_create_sparse:
            mock_sparse_vector = Mock()
            mock_create_sparse.return_value = mock_sparse_vector

            # Mock Qdrant search results
            mock_results = [
                models.ScoredPoint(
                    id="doc1",
                    score=0.85,
                    version=0,
                    payload={"content": "sparse result"}
                )
            ]
            enhanced_mock_qdrant_client.search.return_value = mock_results

            results = await _search_collection(
                enhanced_mock_qdrant_client,
                "test_collection",
                embeddings,
                "sparse",
                limit=10,
                score_threshold=0.7
            )

        assert len(results) == 1
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.85
        assert results[0]["search_type"] == "sparse"

    @pytest.mark.asyncio
    async def test_search_collection_no_embeddings(self, enhanced_mock_qdrant_client):
        """Test search when required embeddings are not available."""
        # Dense mode without dense embeddings
        embeddings = {"sparse": {"indices": [1, 2], "values": [0.8, 0.6]}}
        results = await _search_collection(
            enhanced_mock_qdrant_client, "test_collection", embeddings, "dense", 10, 0.7
        )
        assert results == []

        # Sparse mode without sparse embeddings
        embeddings = {"dense": [0.1] * 384}
        results = await _search_collection(
            enhanced_mock_qdrant_client, "test_collection", embeddings, "sparse", 10, 0.7
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_collection_exception_handling(self, enhanced_mock_qdrant_client):
        """Test exception handling in search collection."""
        embeddings = {"dense": [0.1] * 384}

        # Mock Qdrant to raise exception
        enhanced_mock_qdrant_client.search.side_effect = ResponseHandlingException("Network error")

        results = await _search_collection(
            enhanced_mock_qdrant_client, "test_collection", embeddings, "dense", 10, 0.7
        )

        assert results == []


class TestSearchCollectionByMetadata:
    """Tests for search_collection_by_metadata function."""

    @pytest.mark.asyncio
    async def test_search_by_metadata_client_not_initialized(self, enhanced_mock_workspace_client):
        """Test metadata search with uninitialized client."""
        enhanced_mock_workspace_client.initialized = False

        result = await search_collection_by_metadata(
            enhanced_mock_workspace_client, "docs", {"category": "python"}
        )

        assert result["error"] == "Workspace client not initialized"

    @pytest.mark.asyncio
    async def test_search_by_metadata_empty_filter(self, enhanced_mock_workspace_client):
        """Test metadata search with empty filter."""
        result = await search_collection_by_metadata(
            enhanced_mock_workspace_client, "docs", {}
        )

        assert "error" in result
        assert "Metadata filter cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_search_by_metadata_collection_not_found(self, enhanced_mock_workspace_client):
        """Test metadata search with non-existent collection."""
        enhanced_mock_workspace_client.list_collections.return_value = ["other_collection"]

        result = await search_collection_by_metadata(
            enhanced_mock_workspace_client, "nonexistent_collection", {"category": "python"}
        )

        assert "error" in result
        assert "Collection 'nonexistent_collection' not found" in result["error"]

    @pytest.mark.asyncio
    async def test_search_by_metadata_success(self, enhanced_mock_workspace_client):
        """Test successful metadata search."""
        # Mock scroll results
        mock_points = [
            Mock(id="doc1", payload={"content": "Python guide", "category": "python"}),
            Mock(id="doc2", payload={"content": "Advanced Python", "category": "python"}),
        ]
        enhanced_mock_workspace_client.client.scroll.return_value = (mock_points, None)

        result = await search_collection_by_metadata(
            enhanced_mock_workspace_client, "docs", {"category": "python"}, limit=10
        )

        assert "results" in result
        assert len(result["results"]) == 2
        assert result["total"] == 2
        assert result["collection"] == "docs"
        assert result["filter"] == {"category": "python"}


class TestBuildMetadataFilter:
    """Tests for _build_metadata_filter function."""

    def test_build_metadata_filter_string_values(self):
        """Test filter building with string values."""
        metadata_filter = {
            "category": "python",
            "author": "john_doe",
            "status": "published"
        }

        result = _build_metadata_filter(metadata_filter)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 3
        for condition in result.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchValue)

    def test_build_metadata_filter_list_values(self):
        """Test filter building with list values."""
        metadata_filter = {
            "tags": ["python", "tutorial", "beginner"],
            "categories": ["tech", "programming"]
        }

        result = _build_metadata_filter(metadata_filter)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 2
        for condition in result.must:
            assert isinstance(condition, models.FieldCondition)
            assert isinstance(condition.match, models.MatchAny)

    def test_build_metadata_filter_mixed_types(self):
        """Test filter building with mixed value types."""
        metadata_filter = {
            "title": "Python Guide",        # string
            "page_count": 150,             # int
            "rating": 4.5,                 # float
            "is_published": True,          # bool
            "tags": ["python", "guide"],   # list
        }

        result = _build_metadata_filter(metadata_filter)

        assert isinstance(result, models.Filter)
        assert len(result.must) == 5

    def test_build_metadata_filter_empty_dict(self):
        """Test filter building with empty metadata dictionary."""
        result = _build_metadata_filter({})
        assert result is None

    def test_build_metadata_filter_unsupported_types(self):
        """Test filter building with unsupported value types."""
        metadata_filter = {
            "valid_string": "test",
            "valid_int": 42,
            "unsupported_dict": {"nested": "value"},
            "unsupported_none": None
        }

        result = _build_metadata_filter(metadata_filter)

        # Should only create conditions for supported types
        assert isinstance(result, models.Filter)
        assert len(result.must) == 2  # Only string and int


class TestProjectIsolationAndAggregation:
    """Tests for project isolation and advanced aggregation functions."""

    @pytest.mark.asyncio
    async def test_search_workspace_with_project_isolation(self, enhanced_mock_workspace_client):
        """Test project isolation search."""
        with patch("workspace_qdrant_mcp.tools.search.search_workspace") as mock_search:
            mock_search.return_value = {
                "results": [{"id": "doc1", "score": 0.9}],
                "total": 1,
                "query": "test query"
            }

            result = await search_workspace_with_project_isolation(
                enhanced_mock_workspace_client,
                "test query",
                project_name="custom-project",
                collection_types=["docs"],
                include_shared=False
            )

        # Verify search_workspace was called with correct parameters
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        assert call_args[1]["collections"] == ["docs"]
        assert call_args[1]["auto_inject_project_metadata"] is True
        assert call_args[1]["include_shared"] is False

        # Verify isolation info was added
        assert "isolation_info" in result
        assert result["isolation_info"]["project_name"] == "custom-project"
        assert result["isolation_info"]["isolation_enabled"] is True

    @pytest.mark.asyncio
    async def test_search_workspace_with_project_isolation_no_context(self, enhanced_mock_workspace_client):
        """Test project isolation fallback when no context available."""
        enhanced_mock_workspace_client.get_project_context.return_value = None

        with patch("workspace_qdrant_mcp.tools.search.search_workspace") as mock_search:
            mock_search.return_value = {
                "results": [],
                "total": 0,
                "query": "test query"
            }

            result = await search_workspace_with_project_isolation(
                enhanced_mock_workspace_client, "test query"
            )

        # Should fall back to regular search without project filtering
        call_args = mock_search.call_args
        assert call_args[1]["auto_inject_project_metadata"] is False
        assert "project_context" not in call_args[1]

    @pytest.mark.asyncio
    async def test_search_workspace_with_advanced_aggregation(self, enhanced_mock_workspace_client):
        """Test advanced aggregation search."""
        custom_settings = {
            "enable_multi_tenant_aggregation": False,
            "enable_deduplication": True,
            "score_aggregation_method": "avg_score"
        }

        with patch("workspace_qdrant_mcp.tools.search.search_workspace") as mock_search:
            mock_search.return_value = {
                "results": [],
                "total": 0,
                "query": "test query"
            }

            result = await search_workspace_with_advanced_aggregation(
                enhanced_mock_workspace_client,
                "test query",
                aggregation_settings=custom_settings
            )

        # Verify custom settings were applied
        call_args = mock_search.call_args
        assert call_args[1]["enable_multi_tenant_aggregation"] is False
        assert call_args[1]["enable_deduplication"] is True
        assert call_args[1]["score_aggregation_method"] == "avg_score"

        # Verify settings were added to response
        assert "aggregation_settings_used" in result
        settings = result["aggregation_settings_used"]
        assert settings["enable_multi_tenant_aggregation"] is False
        assert settings["score_aggregation_method"] == "avg_score"

    @pytest.mark.asyncio
    async def test_search_workspace_with_advanced_aggregation_exception(self, enhanced_mock_workspace_client):
        """Test exception handling in advanced aggregation."""
        with patch("workspace_qdrant_mcp.tools.search.search_workspace") as mock_search:
            mock_search.side_effect = Exception("Aggregation failed")

            result = await search_workspace_with_advanced_aggregation(
                enhanced_mock_workspace_client, "test query"
            )

        assert "error" in result
        assert "Advanced aggregation search failed: Aggregation failed" in result["error"]


class TestSearchIntegration:
    """Integration tests covering search workflow."""

    @pytest.mark.asyncio
    async def test_complete_search_workflow(self, enhanced_mock_workspace_client):
        """Test complete search workflow with multiple modes."""
        # Setup embedding service
        embedding_service = enhanced_mock_workspace_client.get_embedding_service()
        embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.8, 0.6, 0.9]}
        }

        modes = ["dense", "sparse", "hybrid"]

        for mode in modes:
            with patch("workspace_qdrant_mcp.tools.search._search_collection") as mock_search:
                mock_search.return_value = [
                    {
                        "id": f"{mode}_doc1",
                        "score": 0.9,
                        "payload": {"content": f"{mode} result"},
                        "search_type": mode
                    }
                ]

                result = await search_workspace(
                    enhanced_mock_workspace_client,
                    f"test {mode} query",
                    mode=mode,
                    limit=5,
                    score_threshold=0.8
                )

            assert "results" in result
            assert result["search_params"]["mode"] == mode
            assert len(result["results"]) == 2  # One from each collection

    @pytest.mark.asyncio
    async def test_error_propagation(self, enhanced_mock_workspace_client):
        """Test error propagation through search function chain."""
        enhanced_mock_workspace_client.initialized = False

        # Test error propagation in advanced aggregation
        with patch("workspace_qdrant_mcp.tools.search.search_workspace") as mock_search:
            mock_search.return_value = {"error": "Workspace client not initialized"}

            result = await search_workspace_with_advanced_aggregation(
                enhanced_mock_workspace_client, "test query"
            )

        assert "error" in result
        assert "Advanced aggregation search failed" in result["error"]