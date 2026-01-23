"""
Comprehensive tests for collection-specific hybrid search functionality (Task 322.3).

This test suite validates hybrid search within specific collection types:
- Test hybrid search in USER collections
- Test hybrid search in PROJECT collections
- Test hybrid search in LIBRARY collections
- Test hybrid search in MEMORY collections
- Verify collection filtering with dense and sparse components
- Test collection metadata accuracy
- Validate collection-specific result ranking

Test Coverage:
1. USER collection searches with hybrid search
2. PROJECT collection searches with hybrid search
3. LIBRARY collection searches with hybrid search
4. MEMORY collection searches with hybrid search
5. Collection filtering with both dense and sparse vectors
6. Collection metadata accuracy in results
7. Collection-specific ranking validation
8. Cross-collection isolation
9. Edge cases and error handling
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from qdrant_client.http import models

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.collection_naming import CollectionNameInfo, CollectionType
from common.core.hybrid_search import (
    HybridSearchEngine,
    RRFFusionRanker,
    WeightedSumFusionRanker,
)
from common.core.multitenant_collections import ProjectMetadata

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing with all required methods."""
    client = MagicMock()

    # Core search methods
    client.search = MagicMock(return_value=[])

    # Collection management methods (for MetadataIndexManager)
    client.get_collection = MagicMock(return_value=MagicMock(
        config=MagicMock(
            params=MagicMock(
                vectors=MagicMock(
                    config={"dense": MagicMock(size=384)}
                )
            )
        )
    ))
    client.collection_exists = MagicMock(return_value=True)
    client.create_payload_index = MagicMock(return_value=True)
    client.update_collection = MagicMock(return_value=True)
    client.get_collections = MagicMock(return_value=MagicMock(collections=[]))

    return client


@pytest.fixture
def hybrid_search_engine(mock_qdrant_client):
    """Create a HybridSearchEngine instance for testing."""
    return HybridSearchEngine(
        client=mock_qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=True,
        enable_performance_monitoring=False  # Disable for unit tests
    )


@pytest.fixture
def sample_query_embeddings():
    """Create sample query embeddings with dense and sparse vectors."""
    return {
        "dense": [0.1] * 384,  # 384-dimensional dense vector
        "sparse": {
            "indices": [1, 5, 10, 15, 20],
            "values": [0.8, 0.6, 0.5, 0.4, 0.3]
        }
    }


@pytest.fixture
def sample_query_embeddings_dense_only():
    """Create sample query embeddings with only dense vectors."""
    return {
        "dense": [0.1] * 384
    }


@pytest.fixture
def sample_query_embeddings_sparse_only():
    """Create sample query embeddings with only sparse vectors."""
    return {
        "sparse": {
            "indices": [1, 5, 10, 15, 20],
            "values": [0.8, 0.6, 0.5, 0.4, 0.3]
        }
    }


@pytest.fixture
def mock_sparse_vector_creation():
    """Mock create_named_sparse_vector to handle dict input properly."""
    def create_sparse_vector_from_dict(sparse_dict):
        """Handle both dict and separate indices/values inputs."""
        if isinstance(sparse_dict, dict):
            # Handle dict format with indices and values
            return {
                "sparse": models.SparseVector(
                    indices=sparse_dict["indices"],
                    values=sparse_dict["values"]
                )
            }
        # Handle separate arguments (original function signature)
        return {"sparse": models.SparseVector(indices=sparse_dict, values=[])}

    with patch("common.core.hybrid_search.create_named_sparse_vector", side_effect=create_sparse_vector_from_dict):
        yield


# ============================================================================
# USER Collection Test Fixtures
# ============================================================================


@pytest.fixture
def user_collection_results():
    """Create mock search results for USER collection."""
    return [
        Mock(
            id="user-doc-1",
            score=0.95,
            payload={
                "content": "My personal note about Python testing",
                "collection_type": "user",
                "collection_name": "myapp-notes",
                "project_id": None,  # USER collections are not project-specific
                "user_id": "user123",
                "tags": ["python", "testing"],
                "created_at": "2025-10-05T10:00:00Z"
            }
        ),
        Mock(
            id="user-doc-2",
            score=0.88,
            payload={
                "content": "Bookmark for Python documentation",
                "collection_type": "user",
                "collection_name": "myapp-bookmarks",
                "project_id": None,
                "user_id": "user123",
                "url": "https://docs.python.org",
                "created_at": "2025-10-05T11:00:00Z"
            }
        ),
        Mock(
            id="user-doc-3",
            score=0.82,
            payload={
                "content": "Code snippet for async testing",
                "collection_type": "user",
                "collection_name": "myapp-snippets",
                "project_id": None,
                "user_id": "user123",
                "language": "python",
                "created_at": "2025-10-05T12:00:00Z"
            }
        ),
    ]


# ============================================================================
# PROJECT Collection Test Fixtures
# ============================================================================


@pytest.fixture
def project_collection_results():
    """Create mock search results for PROJECT collection."""
    return [
        Mock(
            id="proj-doc-1",
            score=0.98,
            payload={
                "content": "def test_authentication():",
                "collection_type": "project",
                "collection_name": "_a1b2c3d4e5f6",  # 12-char hex project ID
                "project_id": "a1b2c3d4e5f6",
                "project_name": "my-awesome-project",
                "file_path": "/src/tests/test_auth.py",
                "file_type": "python",
                "branch": "main",
                "symbols": ["test_authentication"]
            }
        ),
        Mock(
            id="proj-doc-2",
            score=0.93,
            payload={
                "content": "class UserAuthenticator:",
                "collection_type": "project",
                "collection_name": "_a1b2c3d4e5f6",
                "project_id": "a1b2c3d4e5f6",
                "project_name": "my-awesome-project",
                "file_path": "/src/auth.py",
                "file_type": "python",
                "branch": "main",
                "symbols": ["UserAuthenticator"]
            }
        ),
        Mock(
            id="proj-doc-3",
            score=0.87,
            payload={
                "content": "# Authentication Module Documentation",
                "collection_type": "project",
                "collection_name": "_a1b2c3d4e5f6",
                "project_id": "a1b2c3d4e5f6",
                "project_name": "my-awesome-project",
                "file_path": "/docs/auth.md",
                "file_type": "markdown",
                "branch": "main",
                "symbols": []
            }
        ),
    ]


# ============================================================================
# LIBRARY Collection Test Fixtures
# ============================================================================


@pytest.fixture
def library_collection_results():
    """Create mock search results for LIBRARY collection."""
    return [
        Mock(
            id="lib-doc-1",
            score=0.96,
            payload={
                "content": "def async_http_request(url: str, method: str = 'GET'):",
                "collection_type": "library",
                "collection_name": "_httpx",  # Library name with underscore prefix
                "library_name": "httpx",
                "module_path": "httpx.client",
                "function_name": "async_http_request",
                "is_async": True,
                "version": "0.27.0"
            }
        ),
        Mock(
            id="lib-doc-2",
            score=0.91,
            payload={
                "content": "class AsyncClient:",
                "collection_type": "library",
                "collection_name": "_httpx",
                "library_name": "httpx",
                "module_path": "httpx.client",
                "class_name": "AsyncClient",
                "is_async": True,
                "version": "0.27.0"
            }
        ),
        Mock(
            id="lib-doc-3",
            score=0.85,
            payload={
                "content": "Async HTTP client for Python",
                "collection_type": "library",
                "collection_name": "_httpx",
                "library_name": "httpx",
                "module_path": "httpx",
                "doc_type": "module_overview",
                "version": "0.27.0"
            }
        ),
    ]


# ============================================================================
# MEMORY Collection Test Fixtures
# ============================================================================


@pytest.fixture
def memory_collection_results():
    """Create mock search results for MEMORY collection."""
    return [
        Mock(
            id="mem-doc-1",
            score=0.94,
            payload={
                "content": "User prefers pytest for testing",
                "collection_type": "memory",
                "collection_name": "memory",
                "memory_type": "preference",
                "category": "testing",
                "importance": "high",
                "created_at": "2025-10-01T10:00:00Z",
                "updated_at": "2025-10-05T10:00:00Z"
            }
        ),
        Mock(
            id="mem-doc-2",
            score=0.89,
            payload={
                "content": "Always use type hints in Python code",
                "collection_type": "memory",
                "collection_name": "memory",
                "memory_type": "rule",
                "category": "coding_style",
                "importance": "high",
                "created_at": "2025-10-02T10:00:00Z"
            }
        ),
        Mock(
            id="mem-doc-3",
            score=0.83,
            payload={
                "content": "User previously worked on authentication module",
                "collection_type": "memory",
                "collection_name": "memory",
                "memory_type": "context",
                "category": "work_history",
                "importance": "medium",
                "created_at": "2025-10-03T10:00:00Z"
            }
        ),
    ]


# ============================================================================
# Test USER Collection Searches
# ============================================================================


class TestUserCollectionHybridSearch:
    """Test hybrid search within USER collections."""

    @pytest.mark.asyncio
    async def test_user_collection_hybrid_search_basic(
        self,
        hybrid_search_engine,
        user_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test basic hybrid search in USER collection returns correct results."""
        mock_qdrant_client.search.return_value = user_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="myapp-notes",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        # Verify search was called
        assert mock_qdrant_client.search.called

        # Verify results are from USER collection
        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        for result in fused_results:
            assert result.payload["collection_type"] == "user"
            assert result.payload["project_id"] is None  # USER collections not project-specific

    @pytest.mark.asyncio
    async def test_user_collection_dense_only_search(
        self,
        hybrid_search_engine,
        user_collection_results,
        sample_query_embeddings_dense_only,
        mock_qdrant_client
    ):
        """Test USER collection search with dense vectors only."""
        mock_qdrant_client.search.return_value = user_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="myapp-notes",
            query_embeddings=sample_query_embeddings_dense_only,
            limit=10,
            fusion_method="rrf"
        )

        # Should still return results with only dense search
        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0
        assert len(results.get("dense_results", [])) > 0
        assert len(results.get("sparse_results", [])) == 0

    @pytest.mark.asyncio
    async def test_user_collection_sparse_only_search(
        self,
        hybrid_search_engine,
        user_collection_results,
        sample_query_embeddings_sparse_only,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test USER collection search with sparse vectors only."""
        mock_qdrant_client.search.return_value = user_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="myapp-bookmarks",
            query_embeddings=sample_query_embeddings_sparse_only,
            limit=10,
            fusion_method="rrf"
        )

        # Should return results with only sparse search
        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0
        assert len(results.get("sparse_results", [])) > 0
        assert len(results.get("dense_results", [])) == 0

    @pytest.mark.asyncio
    async def test_user_collection_metadata_accuracy(
        self,
        hybrid_search_engine,
        user_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test USER collection metadata is accurately preserved in results."""
        mock_qdrant_client.search.return_value = user_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="myapp-notes",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])

        for result in fused_results:
            payload = result.payload
            # Verify USER collection metadata
            assert "collection_type" in payload
            assert payload["collection_type"] == "user"
            assert "user_id" in payload
            assert "created_at" in payload


# ============================================================================
# Test PROJECT Collection Searches
# ============================================================================


class TestProjectCollectionHybridSearch:
    """Test hybrid search within PROJECT collections."""

    @pytest.mark.asyncio
    async def test_project_collection_hybrid_search_basic(
        self,
        hybrid_search_engine,
        project_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test basic hybrid search in PROJECT collection returns correct results."""
        mock_qdrant_client.search.return_value = project_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_a1b2c3d4e5f6",  # PROJECT collection name (12-char hex)
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        # Verify search was called
        assert mock_qdrant_client.search.called

        # Verify results are from PROJECT collection
        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        for result in fused_results:
            assert result.payload["collection_type"] == "project"
            assert result.payload["project_id"] == "a1b2c3d4e5f6"
            assert result.payload["project_name"] == "my-awesome-project"

    @pytest.mark.asyncio
    async def test_project_collection_metadata_accuracy(
        self,
        hybrid_search_engine,
        project_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test PROJECT collection metadata is accurately preserved in results."""
        mock_qdrant_client.search.return_value = project_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_a1b2c3d4e5f6",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])

        for result in fused_results:
            payload = result.payload
            # Verify PROJECT collection metadata
            assert "collection_type" in payload
            assert payload["collection_type"] == "project"
            assert "project_id" in payload
            assert "project_name" in payload
            assert "file_path" in payload
            assert "file_type" in payload
            assert "branch" in payload

    @pytest.mark.asyncio
    async def test_project_collection_with_context_filter(
        self,
        hybrid_search_engine,
        project_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test PROJECT collection search with project context filtering."""
        mock_qdrant_client.search.return_value = project_collection_results

        project_context = {
            "project_name": "my-awesome-project",
            "project_id": "a1b2c3d4e5f6",
            "collection_type": "project"
        }

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_a1b2c3d4e5f6",
            query_embeddings=sample_query_embeddings,
            project_context=project_context,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        # Verify all results match project context
        for result in fused_results:
            assert result.payload["project_id"] == "a1b2c3d4e5f6"
            assert result.payload["project_name"] == "my-awesome-project"

    @pytest.mark.asyncio
    async def test_project_collection_ranking_accuracy(
        self,
        hybrid_search_engine,
        project_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test PROJECT collection results are ranked correctly by score."""
        mock_qdrant_client.search.return_value = project_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_a1b2c3d4e5f6",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        # Verify results are sorted by score (descending)
        scores = [result.score for result in fused_results if hasattr(result, 'score')]
        if len(scores) > 1:
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], "Results should be sorted by score descending"


# ============================================================================
# Test LIBRARY Collection Searches
# ============================================================================


class TestLibraryCollectionHybridSearch:
    """Test hybrid search within LIBRARY collections."""

    @pytest.mark.asyncio
    async def test_library_collection_hybrid_search_basic(
        self,
        hybrid_search_engine,
        library_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test basic hybrid search in LIBRARY collection returns correct results."""
        mock_qdrant_client.search.return_value = library_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_httpx",  # LIBRARY collection name
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        # Verify search was called
        assert mock_qdrant_client.search.called

        # Verify results are from LIBRARY collection
        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        for result in fused_results:
            assert result.payload["collection_type"] == "library"
            assert result.payload["library_name"] == "httpx"

    @pytest.mark.asyncio
    async def test_library_collection_metadata_accuracy(
        self,
        hybrid_search_engine,
        library_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test LIBRARY collection metadata is accurately preserved in results."""
        mock_qdrant_client.search.return_value = library_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_httpx",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])

        for result in fused_results:
            payload = result.payload
            # Verify LIBRARY collection metadata
            assert "collection_type" in payload
            assert payload["collection_type"] == "library"
            assert "library_name" in payload
            assert "module_path" in payload
            assert "version" in payload

    @pytest.mark.asyncio
    async def test_library_collection_weighted_fusion(
        self,
        hybrid_search_engine,
        library_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test LIBRARY collection search with weighted fusion method."""
        mock_qdrant_client.search.return_value = library_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_httpx",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="weighted_sum",
            dense_weight=0.7,
            sparse_weight=0.3
        )

        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        # Verify weighted scores are applied
        for result in fused_results:
            assert hasattr(result, 'payload')
            # Score should exist
            assert hasattr(result, 'score')


# ============================================================================
# Test MEMORY Collection Searches
# ============================================================================


class TestMemoryCollectionHybridSearch:
    """Test hybrid search within MEMORY collections."""

    @pytest.mark.asyncio
    async def test_memory_collection_hybrid_search_basic(
        self,
        hybrid_search_engine,
        memory_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test basic hybrid search in MEMORY collection returns correct results."""
        mock_qdrant_client.search.return_value = memory_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="memory",  # MEMORY collection name
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        # Verify search was called
        assert mock_qdrant_client.search.called

        # Verify results are from MEMORY collection
        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        for result in fused_results:
            assert result.payload["collection_type"] == "memory"

    @pytest.mark.asyncio
    async def test_memory_collection_metadata_accuracy(
        self,
        hybrid_search_engine,
        memory_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test MEMORY collection metadata is accurately preserved in results."""
        mock_qdrant_client.search.return_value = memory_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="memory",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])

        for result in fused_results:
            payload = result.payload
            # Verify MEMORY collection metadata
            assert "collection_type" in payload
            assert payload["collection_type"] == "memory"
            assert "memory_type" in payload
            assert "category" in payload
            assert "importance" in payload
            assert "created_at" in payload

    @pytest.mark.asyncio
    async def test_memory_collection_max_score_fusion(
        self,
        hybrid_search_engine,
        memory_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test MEMORY collection search with max_score fusion method."""
        mock_qdrant_client.search.return_value = memory_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="memory",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="max_score"
        )

        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        # Verify results are ranked correctly
        for result in fused_results:
            assert hasattr(result, 'score')


# ============================================================================
# Test Collection Filtering with Dense and Sparse Components
# ============================================================================


class TestCollectionFilteringWithVectorComponents:
    """Test collection filtering works correctly with both dense and sparse search."""

    @pytest.mark.asyncio
    async def test_user_collection_dense_sparse_filtering(
        self,
        hybrid_search_engine,
        user_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test USER collection filtering works with both dense and sparse vectors."""
        mock_qdrant_client.search.return_value = user_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="myapp-notes",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        # Verify both searches were called
        assert mock_qdrant_client.search.call_count >= 2  # Dense + Sparse

        # Verify results contain data from both searches
        assert len(results.get("dense_results", [])) > 0 or len(results.get("sparse_results", [])) > 0
        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

    @pytest.mark.asyncio
    async def test_project_collection_dense_sparse_filtering(
        self,
        hybrid_search_engine,
        project_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test PROJECT collection filtering works with both dense and sparse vectors."""
        mock_qdrant_client.search.return_value = project_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_a1b2c3d4e5f6",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        # Verify both searches occurred
        assert mock_qdrant_client.search.call_count >= 2

        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        # All results should be from PROJECT collection
        for result in fused_results:
            assert result.payload["collection_type"] == "project"

    @pytest.mark.asyncio
    async def test_library_collection_dense_sparse_filtering(
        self,
        hybrid_search_engine,
        library_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test LIBRARY collection filtering works with both dense and sparse vectors."""
        mock_qdrant_client.search.return_value = library_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_httpx",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        # All results should be from LIBRARY collection
        for result in fused_results:
            assert result.payload["collection_type"] == "library"

    @pytest.mark.asyncio
    async def test_memory_collection_dense_sparse_filtering(
        self,
        hybrid_search_engine,
        memory_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test MEMORY collection filtering works with both dense and sparse vectors."""
        mock_qdrant_client.search.return_value = memory_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="memory",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0

        # All results should be from MEMORY collection
        for result in fused_results:
            assert result.payload["collection_type"] == "memory"


# ============================================================================
# Test Cross-Collection Isolation
# ============================================================================


class TestCrossCollectionIsolation:
    """Test that searches in one collection type don't return results from others."""

    @pytest.mark.asyncio
    async def test_user_collection_does_not_return_project_results(
        self,
        hybrid_search_engine,
        user_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test USER collection search does not return PROJECT collection results."""
        # Mock returns only USER collection results
        mock_qdrant_client.search.return_value = user_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="myapp-notes",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])
        for result in fused_results:
            assert result.payload["collection_type"] != "project"
            assert result.payload["collection_type"] == "user"

    @pytest.mark.asyncio
    async def test_project_collection_does_not_return_library_results(
        self,
        hybrid_search_engine,
        project_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test PROJECT collection search does not return LIBRARY collection results."""
        mock_qdrant_client.search.return_value = project_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_a1b2c3d4e5f6",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])
        for result in fused_results:
            assert result.payload["collection_type"] != "library"
            assert result.payload["collection_type"] == "project"

    @pytest.mark.asyncio
    async def test_library_collection_does_not_return_memory_results(
        self,
        hybrid_search_engine,
        library_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test LIBRARY collection search does not return MEMORY collection results."""
        mock_qdrant_client.search.return_value = library_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="_httpx",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])
        for result in fused_results:
            assert result.payload["collection_type"] != "memory"
            assert result.payload["collection_type"] == "library"

    @pytest.mark.asyncio
    async def test_memory_collection_does_not_return_user_results(
        self,
        hybrid_search_engine,
        memory_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test MEMORY collection search does not return USER collection results."""
        mock_qdrant_client.search.return_value = memory_collection_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="memory",
            query_embeddings=sample_query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        fused_results = results.get("fused_results", [])
        for result in fused_results:
            assert result.payload["collection_type"] != "user"
            assert result.payload["collection_type"] == "memory"


# ============================================================================
# Test Collection Name Filtering and Routing
# ============================================================================


class TestCollectionNameFiltering:
    """Test collection name filtering and routing in hybrid search."""

    @pytest.mark.asyncio
    async def test_collection_name_routing_user(
        self,
        hybrid_search_engine,
        user_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test search correctly routes to USER collection by name."""
        mock_qdrant_client.search.return_value = user_collection_results

        await hybrid_search_engine.hybrid_search(
            collection_name="myapp-notes",  # USER collection pattern
            query_embeddings=sample_query_embeddings,
            limit=10
        )

        # Verify search was called with correct collection name
        search_calls = mock_qdrant_client.search.call_args_list
        for call in search_calls:
            kwargs = call[1]
            assert kwargs["collection_name"] == "myapp-notes"

    @pytest.mark.asyncio
    async def test_collection_name_routing_project(
        self,
        hybrid_search_engine,
        project_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test search correctly routes to PROJECT collection by name."""
        mock_qdrant_client.search.return_value = project_collection_results

        await hybrid_search_engine.hybrid_search(
            collection_name="_a1b2c3d4e5f6",  # PROJECT collection pattern
            query_embeddings=sample_query_embeddings,
            limit=10
        )

        # Verify search was called with correct collection name
        search_calls = mock_qdrant_client.search.call_args_list
        for call in search_calls:
            kwargs = call[1]
            assert kwargs["collection_name"] == "_a1b2c3d4e5f6"

    @pytest.mark.asyncio
    async def test_collection_name_routing_library(
        self,
        hybrid_search_engine,
        library_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test search correctly routes to LIBRARY collection by name."""
        mock_qdrant_client.search.return_value = library_collection_results

        await hybrid_search_engine.hybrid_search(
            collection_name="_httpx",  # LIBRARY collection pattern
            query_embeddings=sample_query_embeddings,
            limit=10
        )

        # Verify search was called with correct collection name
        search_calls = mock_qdrant_client.search.call_args_list
        for call in search_calls:
            kwargs = call[1]
            assert kwargs["collection_name"] == "_httpx"


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================


class TestCollectionSpecificEdgeCases:
    """Test edge cases and error handling for collection-specific search."""

    @pytest.mark.asyncio
    async def test_empty_collection_results(
        self,
        hybrid_search_engine,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test search handles empty results gracefully for all collection types."""
        mock_qdrant_client.search.return_value = []

        # Test each collection type
        collection_names = ["myapp-notes", "_a1b2c3d4e5f6", "_httpx", "memory"]

        for collection_name in collection_names:
            results = await hybrid_search_engine.hybrid_search(
                collection_name=collection_name,
                query_embeddings=sample_query_embeddings,
                limit=10,
                fusion_method="rrf"
            )

            # Should return empty results gracefully
            fused_results = results.get("fused_results", [])
            assert len(fused_results) == 0

    @pytest.mark.asyncio
    async def test_collection_metadata_null_values(
        self,
        hybrid_search_engine,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """Test handling of null/missing metadata in collection results."""
        # Results with missing metadata fields
        incomplete_results = [
            Mock(
                id="incomplete-1",
                score=0.9,
                payload={
                    "content": "Document with minimal metadata",
                    "collection_type": "user"
                    # Missing other expected fields
                }
            )
        ]

        mock_qdrant_client.search.return_value = incomplete_results

        results = await hybrid_search_engine.hybrid_search(
            collection_name="myapp-notes",
            query_embeddings=sample_query_embeddings,
            limit=10
        )

        # Should handle gracefully
        fused_results = results.get("fused_results", [])
        assert len(fused_results) > 0
        assert fused_results[0].payload.get("collection_type") == "user"

    @pytest.mark.asyncio
    async def test_invalid_collection_name(
        self,
        hybrid_search_engine,
        sample_query_embeddings,
        mock_qdrant_client
    ):
        """Test handling of invalid collection names."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        # Mock search to raise exception for invalid collection
        mock_qdrant_client.search.side_effect = ResponseHandlingException("Collection not found")

        with pytest.raises(ResponseHandlingException):
            await hybrid_search_engine.hybrid_search(
                collection_name="invalid-collection-name",
                query_embeddings=sample_query_embeddings,
                limit=10
            )


# ============================================================================
# Integration Test: Multi-Collection Type Scenarios
# ============================================================================


class TestMultiCollectionTypeScenarios:
    """Integration tests for scenarios involving multiple collection types."""

    @pytest.mark.asyncio
    async def test_search_across_different_collection_types(
        self,
        hybrid_search_engine,
        user_collection_results,
        project_collection_results,
        library_collection_results,
        memory_collection_results,
        sample_query_embeddings,
        mock_qdrant_client,
        mock_sparse_vector_creation
    ):
        """
        Test searching across different collection types maintains isolation.

        This simulates a real-world scenario where the same query is used
        across different collection types and verifies proper isolation.
        """
        collection_configs = {
            "myapp-notes": ("user", user_collection_results),
            "_a1b2c3d4e5f6": ("project", project_collection_results),
            "_httpx": ("library", library_collection_results),
            "memory": ("memory", memory_collection_results)
        }

        all_results = {}

        for collection_name, (expected_type, mock_results) in collection_configs.items():
            # Configure mock for this collection
            mock_qdrant_client.search.return_value = mock_results

            # Perform search
            results = await hybrid_search_engine.hybrid_search(
                collection_name=collection_name,
                query_embeddings=sample_query_embeddings,
                limit=10,
                fusion_method="rrf"
            )

            all_results[collection_name] = {
                "expected_type": expected_type,
                "results": results.get("fused_results", [])
            }

        # Verify isolation
        for collection_name, data in all_results.items():
            expected_type = data["expected_type"]
            results = data["results"]

            assert len(results) > 0, f"No results for {collection_name}"

            # All results should match expected collection type
            for result in results:
                assert result.payload["collection_type"] == expected_type, \
                    f"Collection {collection_name} returned wrong type: {result.payload['collection_type']}"

        # Verify no ID overlap across collection types
        all_ids_by_type = {}
        for _collection_name, data in all_results.items():
            expected_type = data["expected_type"]
            result_ids = {r.id for r in data["results"]}
            all_ids_by_type[expected_type] = result_ids

        # Check pairwise disjoint
        types = list(all_ids_by_type.keys())
        for i in range(len(types)):
            for j in range(i + 1, len(types)):
                type_i = types[i]
                type_j = types[j]
                assert all_ids_by_type[type_i].isdisjoint(all_ids_by_type[type_j]), \
                    f"Collection types {type_i} and {type_j} have overlapping IDs"
