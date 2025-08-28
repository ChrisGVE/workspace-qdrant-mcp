"""
Unit tests for search tools.

Tests workspace search and metadata search functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http import models

from workspace_qdrant_mcp.tools.search import (
    search_collection_by_metadata,
    search_workspace,
)


class TestSearchWorkspace:
    """Test search_workspace function."""

    @pytest.mark.asyncio
    async def test_search_workspace_client_not_initialized(self, mock_workspace_client):
        """Test search with uninitialized client."""
        mock_workspace_client.initialized = False

        result = await search_workspace(mock_workspace_client, "test query")

        assert result["error"] == "Workspace client not initialized"

    @pytest.mark.asyncio
    async def test_search_workspace_empty_query(self, mock_workspace_client):
        """Test search with empty query."""
        mock_workspace_client.initialized = True

        result = await search_workspace(mock_workspace_client, "")

        assert "error" in result
        assert "Query cannot be empty" in result["error"]

        # Test whitespace-only query
        result = await search_workspace(mock_workspace_client, "   ")

        assert "error" in result
        assert "Query cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_no_collections_available(
        self, mock_workspace_client
    ):
        """Test search when no collections are available."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = []

        result = await search_workspace(mock_workspace_client, "test query")

        assert "error" in result
        assert "No collections available for search" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_collections_not_found(self, mock_workspace_client):
        """Test search with specified collections not found."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs", "references"]

        result = await search_workspace(
            mock_workspace_client,
            "test query",
            collections=["nonexistent1", "nonexistent2"],
        )

        assert "error" in result
        assert "Collections not found: nonexistent1, nonexistent2" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_hybrid_success(
        self, mock_workspace_client, mock_embedding_service
    ):
        """Test successful hybrid search."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs", "references"]
        mock_workspace_client.get_embedding_service.return_value = (
            mock_embedding_service
        )

        # Mock embeddings
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.8, 0.6, 0.9]},
        }

        # Mock search results for each collection
        mock_results_docs = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "Python programming"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "Machine learning"}},
        ]
        mock_results_refs = [
            {"id": "ref1", "score": 0.85, "payload": {"content": "API documentation"}}
        ]

        with patch(
            "workspace_qdrant_mcp.tools.search.HybridSearchEngine"
        ) as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.hybrid_search.side_effect = [
                {"results": mock_results_docs, "total": 2, "fusion_method": "rrf"},
                {"results": mock_results_refs, "total": 1, "fusion_method": "rrf"},
            ]
            mock_engine_class.return_value = mock_engine

            result = await search_workspace(
                mock_workspace_client,
                "test query",
                mode="hybrid",
                limit=10,
                score_threshold=0.7,
            )

        assert "results" in result
        assert "total" in result
        assert "collections_searched" in result
        assert "search_params" in result

        # Should have results from both collections
        assert len(result["results"]) == 3
        assert result["total"] == 3
        assert set(result["collections_searched"]) == {"docs", "references"}

        # Verify search was called for both collections
        assert mock_engine.hybrid_search.call_count == 2

        # Verify search parameters
        search_params = result["search_params"]
        assert search_params["mode"] == "hybrid"
        assert search_params["limit"] == 10
        assert search_params["score_threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_search_workspace_specific_collections(
        self, mock_workspace_client, mock_embedding_service
    ):
        """Test search with specific collections specified."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = [
            "docs",
            "references",
            "scratchbook",
        ]
        mock_workspace_client.get_embedding_service.return_value = (
            mock_embedding_service
        )

        mock_embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        with patch(
            "workspace_qdrant_mcp.tools.search.HybridSearchEngine"
        ) as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.hybrid_search.return_value = {
                "results": [{"id": "doc1", "score": 0.9}],
                "total": 1,
            }
            mock_engine_class.return_value = mock_engine

            result = await search_workspace(
                mock_workspace_client,
                "test query",
                collections=["docs", "references"],  # Only these two
            )

        # Should only search specified collections
        assert set(result["collections_searched"]) == {"docs", "references"}
        assert mock_engine.hybrid_search.call_count == 2

    @pytest.mark.asyncio
    async def test_search_workspace_dense_mode(
        self, mock_workspace_client, mock_embedding_service
    ):
        """Test search with dense mode."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]
        mock_workspace_client.get_embedding_service.return_value = (
            mock_embedding_service
        )

        mock_embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.search.return_value = [
            models.ScoredPoint(
                id="doc1", score=0.9, version=0, payload={"content": "Test content"}
            )
        ]

        result = await search_workspace(
            mock_workspace_client, "test query", mode="dense"
        )

        assert "results" in result
        assert len(result["results"]) == 1
        assert result["search_params"]["mode"] == "dense"

        # Verify dense search was called
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert "query_vector" in call_args[1]
        assert "using" in call_args[1]
        assert call_args[1]["using"] == "dense"

    @pytest.mark.asyncio
    async def test_search_workspace_sparse_mode(
        self, mock_workspace_client, mock_embedding_service
    ):
        """Test search with sparse mode."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]
        mock_workspace_client.get_embedding_service.return_value = (
            mock_embedding_service
        )

        mock_embedding_service.generate_embeddings.return_value = {
            "sparse": {"indices": [1, 2], "values": [0.8, 0.6]}
        }

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.search.return_value = [
            models.ScoredPoint(
                id="doc1", score=0.9, version=0, payload={"content": "Test content"}
            )
        ]

        result = await search_workspace(
            mock_workspace_client, "test query", mode="sparse"
        )

        assert "results" in result
        assert result["search_params"]["mode"] == "sparse"

        # Verify sparse search was called
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert "query_vector" in call_args[1]
        assert "using" in call_args[1]
        assert call_args[1]["using"] == "sparse"

    @pytest.mark.asyncio
    async def test_search_workspace_no_sparse_embeddings_sparse_mode(
        self, mock_workspace_client, mock_embedding_service
    ):
        """Test sparse mode when no sparse embeddings available."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]
        mock_workspace_client.get_embedding_service.return_value = (
            mock_embedding_service
        )

        # No sparse embeddings
        mock_embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        result = await search_workspace(
            mock_workspace_client, "test query", mode="sparse"
        )

        assert "error" in result
        assert "Sparse embeddings not available" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_results_filtering_by_threshold(
        self, mock_workspace_client, mock_embedding_service
    ):
        """Test that results are filtered by score threshold."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]
        mock_workspace_client.get_embedding_service.return_value = (
            mock_embedding_service
        )

        mock_embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.search.return_value = [
            models.ScoredPoint(
                id="doc1", score=0.9, version=0, payload={"content": "High score"}
            ),
            models.ScoredPoint(
                id="doc2", score=0.6, version=0, payload={"content": "Low score"}
            ),
            models.ScoredPoint(
                id="doc3", score=0.8, version=0, payload={"content": "Medium score"}
            ),
        ]

        result = await search_workspace(
            mock_workspace_client,
            "test query",
            mode="dense",
            score_threshold=0.7,  # Should filter out doc2 (0.6)
        )

        assert len(result["results"]) == 2  # Only doc1 and doc3
        scores = [r["score"] for r in result["results"]]
        assert all(score >= 0.7 for score in scores)

    @pytest.mark.asyncio
    async def test_search_workspace_results_sorted_by_score(
        self, mock_workspace_client, mock_embedding_service
    ):
        """Test that results are sorted by score in descending order."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]
        mock_workspace_client.get_embedding_service.return_value = (
            mock_embedding_service
        )

        mock_embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.search.return_value = [
            models.ScoredPoint(
                id="doc2", score=0.8, version=0, payload={"content": "Medium"}
            ),
            models.ScoredPoint(
                id="doc1", score=0.9, version=0, payload={"content": "High"}
            ),
            models.ScoredPoint(
                id="doc3", score=0.7, version=0, payload={"content": "Low"}
            ),
        ]

        result = await search_workspace(
            mock_workspace_client, "test query", mode="dense"
        )

        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)  # Should be in descending order

    @pytest.mark.asyncio
    async def test_search_workspace_limit_enforcement(
        self, mock_workspace_client, mock_embedding_service
    ):
        """Test that result limit is enforced."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs", "references"]
        mock_workspace_client.get_embedding_service.return_value = (
            mock_embedding_service
        )

        mock_embedding_service.generate_embeddings.return_value = {"dense": [0.1] * 384}

        # Create more results than the limit
        mock_results = [
            models.ScoredPoint(
                id=f"doc{i}",
                score=0.9 - (i * 0.1),
                version=0,
                payload={"content": f"Doc {i}"},
            )
            for i in range(10)
        ]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.search.return_value = mock_results

        result = await search_workspace(
            mock_workspace_client,
            "test query",
            mode="dense",
            limit=3,  # Lower than available results
        )

        # Should only return 3 results total, not 3 per collection
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_search_workspace_exception_handling(self, mock_workspace_client):
        """Test search exception handling."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.side_effect = Exception(
            "Connection error"
        )

        result = await search_workspace(mock_workspace_client, "test query")

        assert "error" in result
        assert "Search failed" in result["error"]
        assert "Connection error" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_invalid_mode(self, mock_workspace_client):
        """Test search with invalid mode."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        result = await search_workspace(
            mock_workspace_client, "test query", mode="invalid_mode"
        )

        assert "error" in result
        assert "Invalid search mode" in result["error"]


class TestSearchCollectionByMetadata:
    """Test search_collection_by_metadata function."""

    @pytest.mark.asyncio
    async def test_search_by_metadata_client_not_initialized(
        self, mock_workspace_client
    ):
        """Test metadata search with uninitialized client."""
        mock_workspace_client.initialized = False

        result = await search_collection_by_metadata(
            mock_workspace_client, "docs", {"category": "python"}
        )

        assert result["error"] == "Workspace client not initialized"

    @pytest.mark.asyncio
    async def test_search_by_metadata_collection_not_found(self, mock_workspace_client):
        """Test metadata search with non-existent collection."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["other_collection"]

        result = await search_collection_by_metadata(
            mock_workspace_client, "nonexistent_collection", {"category": "python"}
        )

        assert "error" in result
        assert "Collection 'nonexistent_collection' not found" in result["error"]

    @pytest.mark.asyncio
    async def test_search_by_metadata_empty_filter(self, mock_workspace_client):
        """Test metadata search with empty filter."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        result = await search_collection_by_metadata(mock_workspace_client, "docs", {})

        assert "error" in result
        assert "Metadata filter cannot be empty" in result["error"]

    @pytest.mark.asyncio
    async def test_search_by_metadata_success(self, mock_workspace_client):
        """Test successful metadata search."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        # Mock scroll results
        mock_points = [
            MagicMock(
                id="doc1",
                payload={
                    "content": "Python programming guide",
                    "category": "python",
                    "difficulty": "beginner",
                },
            ),
            MagicMock(
                id="doc2",
                payload={
                    "content": "Advanced Python concepts",
                    "category": "python",
                    "difficulty": "advanced",
                },
            ),
        ]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = (mock_points, None)

        result = await search_collection_by_metadata(
            mock_workspace_client, "docs", {"category": "python"}, limit=10
        )

        assert "results" in result
        assert "total" in result
        assert "collection" in result
        assert "filter" in result

        assert len(result["results"]) == 2
        assert result["total"] == 2
        assert result["collection"] == "docs"
        assert result["filter"] == {"category": "python"}

        # Verify scroll was called with correct filter
        mock_client.scroll.assert_called_once()
        call_args = mock_client.scroll.call_args
        assert call_args[1]["collection_name"] == "docs"
        assert call_args[1]["limit"] == 10
        assert call_args[1]["with_payload"] is True

        # Verify filter structure
        scroll_filter = call_args[1]["scroll_filter"]
        assert scroll_filter is not None
        assert hasattr(scroll_filter, "must")

    @pytest.mark.asyncio
    async def test_search_by_metadata_multiple_filters(self, mock_workspace_client):
        """Test metadata search with multiple filter conditions."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([], None)

        await search_collection_by_metadata(
            mock_workspace_client,
            "docs",
            {"category": "python", "difficulty": "beginner", "status": "published"},
        )

        # Verify multiple filter conditions were created
        call_args = mock_client.scroll.call_args
        scroll_filter = call_args[1]["scroll_filter"]

        # Should have created multiple FieldConditions
        assert len(scroll_filter.must) == 3

    @pytest.mark.asyncio
    async def test_search_by_metadata_list_values(self, mock_workspace_client):
        """Test metadata search with list values (any match)."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([], None)

        await search_collection_by_metadata(
            mock_workspace_client, "docs", {"tags": ["python", "tutorial", "beginner"]}
        )

        # Should handle list values for any match
        call_args = mock_client.scroll.call_args
        scroll_filter = call_args[1]["scroll_filter"]
        assert scroll_filter is not None

    @pytest.mark.asyncio
    async def test_search_by_metadata_no_results(self, mock_workspace_client):
        """Test metadata search with no matching results."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([], None)  # No results

        result = await search_collection_by_metadata(
            mock_workspace_client, "docs", {"category": "nonexistent"}
        )

        assert result["results"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_search_by_metadata_limit_enforcement(self, mock_workspace_client):
        """Test that limit is properly enforced."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        # Create more results than the limit
        mock_points = [
            MagicMock(
                id=f"doc{i}", payload={"content": f"Doc {i}", "category": "python"}
            )
            for i in range(10)
        ]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = (mock_points, None)

        result = await search_collection_by_metadata(
            mock_workspace_client, "docs", {"category": "python"}, limit=5
        )

        # Should only return 5 results
        assert len(result["results"]) == 5
        assert result["total"] == 5

    @pytest.mark.asyncio
    async def test_search_by_metadata_exception_handling(self, mock_workspace_client):
        """Test metadata search exception handling."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.side_effect = Exception(
            "Connection error"
        )

        result = await search_collection_by_metadata(
            mock_workspace_client, "docs", {"category": "python"}
        )

        assert "error" in result
        assert "Metadata search failed" in result["error"]
        assert "Connection error" in result["error"]

    @pytest.mark.asyncio
    async def test_search_by_metadata_special_characters(self, mock_workspace_client):
        """Test metadata search with special characters in values."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([], None)

        # Test with special characters
        await search_collection_by_metadata(
            mock_workspace_client,
            "docs",
            {"title": "C++ Programming: Advanced Techniques & Best Practices"},
        )

        # Should handle special characters properly
        mock_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_metadata_numeric_values(self, mock_workspace_client):
        """Test metadata search with numeric values."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([], None)

        await search_collection_by_metadata(
            mock_workspace_client, "docs", {"page_count": 100, "rating": 4.5}
        )

        # Should handle numeric values properly
        mock_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_metadata_boolean_values(self, mock_workspace_client):
        """Test metadata search with boolean values."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["docs"]

        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([], None)

        await search_collection_by_metadata(
            mock_workspace_client, "docs", {"is_published": True, "is_draft": False}
        )

        # Should handle boolean values properly
        mock_client.scroll.assert_called_once()
