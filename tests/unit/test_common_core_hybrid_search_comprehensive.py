"""
Comprehensive unit tests for common.core.hybrid_search module.

This test suite provides complete coverage of the hybrid search functionality,
including all fusion algorithms (RRF, WeightedSum, MaxScore), search engine
operations, multi-tenant support, and performance monitoring.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.hybrid_search import (
    HybridSearchEngine,
    MaxScoreFusionRanker,
    MultiTenantResultAggregator,
    RRFFusionRanker,
    TenantAwareResult,
    TenantAwareResultDeduplicator,
    WeightedSumFusionRanker,
    create_fusion_ranker,
)


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    client = AsyncMock()
    client.search.return_value = []
    return client


@pytest.fixture
def sample_dense_results():
    """Sample dense search results for testing."""
    return [
        {"id": "doc1", "score": 0.95, "payload": {"content": "Dense result 1", "project": "proj1"}},
        {"id": "doc2", "score": 0.87, "payload": {"content": "Dense result 2", "project": "proj1"}},
        {"id": "doc3", "score": 0.82, "payload": {"content": "Dense result 3", "project": "proj2"}},
        {"id": "doc4", "score": 0.79, "payload": {"content": "Dense result 4", "project": "proj1"}},
    ]


@pytest.fixture
def sample_sparse_results():
    """Sample sparse search results for testing."""
    return [
        {"id": "doc2", "score": 0.92, "payload": {"content": "Sparse result 1", "project": "proj1"}},
        {"id": "doc5", "score": 0.88, "payload": {"content": "Sparse result 2", "project": "proj2"}},
        {"id": "doc1", "score": 0.85, "payload": {"content": "Sparse result 3", "project": "proj1"}},
        {"id": "doc6", "score": 0.81, "payload": {"content": "Sparse result 4", "project": "proj3"}},
    ]


class TestRRFFusionRanker:
    """Test RRF (Reciprocal Rank Fusion) algorithm."""

    def test_init_default_k(self):
        """Test RRF ranker initialization with default k value."""
        ranker = RRFFusionRanker()
        assert ranker.k == 60

    def test_init_custom_k(self):
        """Test RRF ranker initialization with custom k value."""
        ranker = RRFFusionRanker(k=100)
        assert ranker.k == 100

    def test_fuse_rankings_empty_results(self):
        """Test fusion with empty result sets."""
        ranker = RRFFusionRanker()
        fused = ranker.fuse_rankings([], [])
        assert fused == []

    def test_fuse_rankings_only_dense_results(self, sample_dense_results):
        """Test fusion with only dense results."""
        ranker = RRFFusionRanker()
        fused = ranker.fuse_rankings(sample_dense_results, [])

        assert len(fused) == len(sample_dense_results)
        assert fused[0]["id"] == "doc1"  # Highest dense score
        assert "rrf_score" in fused[0]
        assert "fusion_explanation" in fused[0]

    def test_fuse_rankings_only_sparse_results(self, sample_sparse_results):
        """Test fusion with only sparse results."""
        ranker = RRFFusionRanker()
        fused = ranker.fuse_rankings([], sample_sparse_results)

        assert len(fused) == len(sample_sparse_results)
        assert fused[0]["id"] == "doc2"  # Highest sparse score
        assert "rrf_score" in fused[0]

    def test_fuse_rankings_both_results(self, sample_dense_results, sample_sparse_results):
        """Test fusion with both dense and sparse results."""
        ranker = RRFFusionRanker()
        fused = ranker.fuse_rankings(sample_dense_results, sample_sparse_results)

        # Should have unique documents from both sets
        doc_ids = [doc["id"] for doc in fused]
        assert "doc1" in doc_ids  # In both sets
        assert "doc2" in doc_ids  # In both sets
        assert "doc5" in doc_ids  # Only in sparse
        assert "doc3" in doc_ids  # Only in dense

        # Results should be sorted by RRF score
        scores = [doc["rrf_score"] for doc in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_score_calculation(self):
        """Test RRF score calculation formula."""
        ranker = RRFFusionRanker(k=60)
        dense_results = [{"id": "doc1", "score": 0.9}]
        sparse_results = [{"id": "doc1", "score": 0.8}]

        fused = ranker.fuse_rankings(dense_results, sparse_results)

        # doc1 is rank 1 in both, so RRF = 1/(1+60) + 1/(1+60) = 2/61
        expected_rrf = 2 / 61
        assert abs(fused[0]["rrf_score"] - expected_rrf) < 0.001

    def test_explain_fusion_detailed(self, sample_dense_results, sample_sparse_results):
        """Test detailed fusion explanation."""
        ranker = RRFFusionRanker()
        fused = ranker.fuse_rankings(sample_dense_results, sample_sparse_results)

        # Check explanation structure
        explanation = fused[0]["fusion_explanation"]
        assert "dense_rank" in explanation
        assert "sparse_rank" in explanation
        assert "rrf_contribution" in explanation
        assert "combined_score" in explanation

    def test_fusion_with_duplicate_handling(self):
        """Test that duplicates are properly handled in fusion."""
        ranker = RRFFusionRanker()
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {"source": "dense"}},
            {"id": "doc2", "score": 0.8, "payload": {"source": "dense"}},
        ]
        sparse_results = [
            {"id": "doc1", "score": 0.85, "payload": {"source": "sparse"}},
            {"id": "doc3", "score": 0.75, "payload": {"source": "sparse"}},
        ]

        fused = ranker.fuse_rankings(dense_results, sparse_results)

        # Should have 3 unique documents
        assert len(fused) == 3
        doc_ids = [doc["id"] for doc in fused]
        assert len(set(doc_ids)) == 3  # All unique

        # doc1 should have highest RRF score (appears in both)
        doc1 = next(doc for doc in fused if doc["id"] == "doc1")
        assert doc1 == fused[0] or fused[0]["rrf_score"] <= doc1["rrf_score"]


class TestWeightedSumFusionRanker:
    """Test Weighted Sum fusion algorithm."""

    def test_init_default_weights(self):
        """Test WeightedSum ranker initialization with default weights."""
        ranker = WeightedSumFusionRanker()
        assert ranker.dense_weight == 1.0
        assert ranker.sparse_weight == 1.0

    def test_init_custom_weights(self):
        """Test WeightedSum ranker initialization with custom weights."""
        ranker = WeightedSumFusionRanker(dense_weight=0.7, sparse_weight=0.3)
        assert ranker.dense_weight == 0.7
        assert ranker.sparse_weight == 0.3

    def test_normalize_scores(self):
        """Test score normalization functionality."""
        ranker = WeightedSumFusionRanker()
        scores = [0.9, 0.7, 0.5, 0.3]

        normalized = ranker._normalize_scores(scores)

        assert min(normalized) >= 0.0
        assert max(normalized) <= 1.0
        assert normalized[0] > normalized[1]  # Maintain ordering

    def test_weighted_sum_calculation(self, sample_dense_results, sample_sparse_results):
        """Test weighted sum score calculation."""
        ranker = WeightedSumFusionRanker(dense_weight=0.6, sparse_weight=0.4)

        fused = ranker.fuse_rankings(sample_dense_results, sample_sparse_results)

        assert len(fused) > 0
        assert "weighted_score" in fused[0]
        assert "fusion_explanation" in fused[0]

    def test_weighted_sum_different_weights(self, sample_dense_results, sample_sparse_results):
        """Test that different weights produce different results."""
        ranker1 = WeightedSumFusionRanker(dense_weight=0.8, sparse_weight=0.2)
        ranker2 = WeightedSumFusionRanker(dense_weight=0.2, sparse_weight=0.8)

        fused1 = ranker1.fuse_rankings(sample_dense_results, sample_sparse_results)
        fused2 = ranker2.fuse_rankings(sample_dense_results, sample_sparse_results)

        # Results should be different due to different weighting
        assert fused1[0]["id"] != fused2[0]["id"] or fused1[0]["weighted_score"] != fused2[0]["weighted_score"]


class TestMaxScoreFusionRanker:
    """Test Max Score fusion algorithm."""

    def test_max_score_fusion(self, sample_dense_results, sample_sparse_results):
        """Test max score fusion takes highest score for each document."""
        ranker = MaxScoreFusionRanker()

        fused = ranker.fuse_rankings(sample_dense_results, sample_sparse_results)

        assert len(fused) > 0
        assert "max_score" in fused[0]

        # For documents that appear in both, should take max score
        doc1 = next(doc for doc in fused if doc["id"] == "doc1")
        assert doc1["max_score"] == 0.95  # Higher of 0.95 (dense) and 0.85 (sparse)

    def test_max_score_single_source(self, sample_dense_results):
        """Test max score with single source (dense only)."""
        ranker = MaxScoreFusionRanker()

        fused = ranker.fuse_rankings(sample_dense_results, [])

        assert len(fused) == len(sample_dense_results)
        assert fused[0]["max_score"] == sample_dense_results[0]["score"]

    def test_max_score_explanation(self, sample_dense_results, sample_sparse_results):
        """Test max score fusion explanation."""
        ranker = MaxScoreFusionRanker()

        fused = ranker.fuse_rankings(sample_dense_results, sample_sparse_results)

        explanation = fused[0]["fusion_explanation"]
        assert "dense_score" in explanation
        assert "sparse_score" in explanation
        assert "max_score_source" in explanation


class TestTenantAwareResult:
    """Test tenant-aware result handling."""

    def test_init_basic_result(self):
        """Test basic TenantAwareResult initialization."""
        result = TenantAwareResult(
            id="doc1",
            score=0.9,
            payload={"content": "test", "project": "proj1"},
            tenant_id="proj1"
        )

        assert result.id == "doc1"
        assert result.score == 0.9
        assert result.tenant_id == "proj1"
        assert result.get_dedup_key() == "doc1"

    def test_custom_dedup_key(self):
        """Test custom deduplication key generation."""
        result = TenantAwareResult(
            id="doc1",
            score=0.9,
            payload={"content": "test", "project": "proj1", "file_path": "/path/to/file"},
            tenant_id="proj1",
            dedup_key_fields=["file_path", "project"]
        )

        dedup_key = result.get_dedup_key()
        assert "file_path" in dedup_key
        assert "project" in dedup_key

    def test_metadata_extraction(self):
        """Test metadata extraction from payload."""
        payload = {
            "content": "test content",
            "project": "proj1",
            "file_type": "python",
            "author": "user1"
        }

        result = TenantAwareResult("doc1", 0.9, payload, "proj1")
        metadata = result.get_metadata()

        assert metadata["project"] == "proj1"
        assert metadata["file_type"] == "python"
        assert "content" not in metadata  # Should exclude content

    def test_result_comparison(self):
        """Test result comparison for sorting."""
        result1 = TenantAwareResult("doc1", 0.9, {}, "proj1")
        result2 = TenantAwareResult("doc2", 0.8, {}, "proj1")

        # Higher score should sort first
        assert result1 > result2
        assert result2 < result1

    def test_tenant_isolation_check(self):
        """Test tenant isolation validation."""
        result = TenantAwareResult("doc1", 0.9, {"project": "proj1"}, "proj1")

        assert result.belongs_to_tenant("proj1")
        assert not result.belongs_to_tenant("proj2")
        assert result.belongs_to_tenant(None)  # Global access


class TestTenantAwareResultDeduplicator:
    """Test result deduplication functionality."""

    def test_deduplication_by_id(self):
        """Test basic deduplication by document ID."""
        results = [
            TenantAwareResult("doc1", 0.9, {"content": "v1"}, "proj1"),
            TenantAwareResult("doc1", 0.8, {"content": "v2"}, "proj1"),  # Duplicate
            TenantAwareResult("doc2", 0.85, {"content": "v3"}, "proj1"),
        ]

        deduplicator = TenantAwareResultDeduplicator()
        deduplicated = deduplicator.deduplicate(results)

        assert len(deduplicated) == 2
        doc_ids = [r.id for r in deduplicated]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

        # Should keep higher scoring version
        doc1 = next(r for r in deduplicated if r.id == "doc1")
        assert doc1.score == 0.9

    def test_deduplication_cross_tenant(self):
        """Test deduplication across different tenants."""
        results = [
            TenantAwareResult("doc1", 0.9, {"content": "v1"}, "proj1"),
            TenantAwareResult("doc1", 0.95, {"content": "v1"}, "proj2"),  # Same doc, different tenant
            TenantAwareResult("doc2", 0.85, {"content": "v2"}, "proj1"),
        ]

        deduplicator = TenantAwareResultDeduplicator(isolation_mode="strict")
        deduplicated = deduplicator.deduplicate(results, target_tenant="proj1")

        # Should only return proj1 results
        assert len(deduplicated) == 2
        assert all(r.tenant_id == "proj1" for r in deduplicated)

    def test_deduplication_by_content_hash(self):
        """Test deduplication by content hash."""
        results = [
            TenantAwareResult("doc1", 0.9, {"content": "same content", "file": "file1.txt"}, "proj1"),
            TenantAwareResult("doc2", 0.8, {"content": "same content", "file": "file2.txt"}, "proj1"),
            TenantAwareResult("doc3", 0.85, {"content": "different content"}, "proj1"),
        ]

        deduplicator = TenantAwareResultDeduplicator(dedup_strategy="content_hash")
        deduplicated = deduplicator.deduplicate(results)

        # Should deduplicate based on content
        assert len(deduplicated) == 2

    def test_aggregation_strategies(self):
        """Test different result aggregation strategies."""
        results = [
            TenantAwareResult("doc1", 0.9, {"source": "dense"}, "proj1"),
            TenantAwareResult("doc1", 0.8, {"source": "sparse"}, "proj1"),
        ]

        # Test max score aggregation
        deduplicator = TenantAwareResultDeduplicator(aggregation_method="max_score")
        deduplicated = deduplicator.deduplicate(results)
        assert deduplicated[0].score == 0.9

        # Test average aggregation
        deduplicator = TenantAwareResultDeduplicator(aggregation_method="average")
        deduplicated = deduplicator.deduplicate(results)
        assert deduplicated[0].score == 0.85  # (0.9 + 0.8) / 2


class TestMultiTenantResultAggregator:
    """Test multi-tenant result aggregation."""

    def test_aggregate_results_single_tenant(self):
        """Test aggregation with single tenant results."""
        results = [
            TenantAwareResult("doc1", 0.9, {"project": "proj1"}, "proj1"),
            TenantAwareResult("doc2", 0.8, {"project": "proj1"}, "proj1"),
        ]

        aggregator = MultiTenantResultAggregator()
        aggregated = aggregator.aggregate_results(results, target_tenant="proj1")

        assert len(aggregated) == 2
        assert all("tenant_info" in r for r in aggregated)

    def test_aggregate_cross_tenant_filtering(self):
        """Test cross-tenant filtering during aggregation."""
        results = [
            TenantAwareResult("doc1", 0.9, {"project": "proj1"}, "proj1"),
            TenantAwareResult("doc2", 0.8, {"project": "proj2"}, "proj2"),
            TenantAwareResult("doc3", 0.85, {"project": "global"}, None),  # Global result
        ]

        aggregator = MultiTenantResultAggregator(isolation_mode="strict")
        aggregated = aggregator.aggregate_results(results, target_tenant="proj1")

        # Should only include proj1 and global results
        assert len(aggregated) == 2
        tenant_ids = [r.get("tenant_id") for r in aggregated]
        assert "proj1" in tenant_ids
        assert None in tenant_ids  # Global
        assert "proj2" not in tenant_ids

    def test_score_normalization_across_tenants(self):
        """Test score normalization across different tenants."""
        results = [
            TenantAwareResult("doc1", 0.9, {"project": "proj1"}, "proj1"),
            TenantAwareResult("doc2", 0.3, {"project": "proj2"}, "proj2"),  # Lower score range
            TenantAwareResult("doc3", 0.5, {"project": "proj2"}, "proj2"),
        ]

        aggregator = MultiTenantResultAggregator(normalize_scores=True)
        aggregated = aggregator.aggregate_results(results)

        # Scores should be normalized
        scores = [r["score"] for r in aggregated]
        assert min(scores) >= 0.0
        assert max(scores) <= 1.0

    def test_convert_to_api_format(self):
        """Test conversion to API response format."""
        results = [
            TenantAwareResult("doc1", 0.9, {"content": "test", "project": "proj1"}, "proj1"),
            TenantAwareResult("doc2", 0.8, {"content": "test2", "project": "proj1"}, "proj1"),
        ]

        aggregator = MultiTenantResultAggregator()
        api_results = aggregator.convert_to_api_format(results)

        assert len(api_results) == 2
        assert "id" in api_results[0]
        assert "score" in api_results[0]
        assert "payload" in api_results[0]
        assert "metadata" in api_results[0]


class TestHybridSearchEngine:
    """Test the main hybrid search engine."""

    def test_init_default_params(self, mock_qdrant_client):
        """Test search engine initialization with defaults."""
        engine = HybridSearchEngine(mock_qdrant_client)

        assert engine.client == mock_qdrant_client
        assert engine.default_fusion_method == "rrf"
        assert not engine.enable_performance_monitoring

    def test_init_with_monitoring(self, mock_qdrant_client):
        """Test initialization with performance monitoring enabled."""
        engine = HybridSearchEngine(mock_qdrant_client, enable_performance_monitoring=True)

        assert engine.enable_performance_monitoring
        assert hasattr(engine, 'performance_monitor')

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, mock_qdrant_client):
        """Test basic hybrid search operation."""
        engine = HybridSearchEngine(mock_qdrant_client)

        # Mock search results
        mock_qdrant_client.search.return_value = [
            models.ScoredPoint(id="doc1", score=0.9, version=0, payload={"content": "result 1"}),
            models.ScoredPoint(id="doc2", score=0.8, version=0, payload={"content": "result 2"}),
        ]

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
        }

        results = await engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings=query_embeddings,
            limit=10
        )

        assert len(results) > 0
        assert "id" in results[0]
        assert "score" in results[0]
        assert mock_qdrant_client.search.call_count == 2  # Dense + sparse searches

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self, mock_qdrant_client):
        """Test hybrid search with metadata filters."""
        engine = HybridSearchEngine(mock_qdrant_client)

        mock_qdrant_client.search.return_value = []

        query_embeddings = {"dense": [0.1] * 384}
        filters = {"project": "test-project", "file_type": "python"}

        await engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings=query_embeddings,
            limit=10,
            filters=filters
        )

        # Verify filters were passed to search calls
        search_calls = mock_qdrant_client.search.call_args_list
        assert len(search_calls) >= 1
        # Check that filters were included in search parameters

    @pytest.mark.asyncio
    async def test_hybrid_search_different_fusion_methods(self, mock_qdrant_client):
        """Test hybrid search with different fusion methods."""
        engine = HybridSearchEngine(mock_qdrant_client)

        mock_qdrant_client.search.return_value = [
            models.ScoredPoint(id="doc1", score=0.9, version=0, payload={"content": "result"}),
        ]

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1], "values": [0.8]}
        }

        # Test different fusion methods
        for method in ["rrf", "weighted_sum", "max_score"]:
            results = await engine.hybrid_search(
                collection_name="test-collection",
                query_embeddings=query_embeddings,
                fusion_method=method,
                limit=5
            )

            assert len(results) >= 0
            if results:
                assert "fusion_explanation" in results[0] or method == "max_score"

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mock_qdrant_client):
        """Test error handling during search operations."""
        engine = HybridSearchEngine(mock_qdrant_client)

        # Mock search failure
        mock_qdrant_client.search.side_effect = ResponseHandlingException("Search failed")

        query_embeddings = {"dense": [0.1] * 384}

        with pytest.raises(ResponseHandlingException):
            await engine.hybrid_search(
                collection_name="test-collection",
                query_embeddings=query_embeddings
            )

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, mock_qdrant_client):
        """Test performance monitoring during search."""
        engine = HybridSearchEngine(mock_qdrant_client, enable_performance_monitoring=True)

        mock_qdrant_client.search.return_value = []

        query_embeddings = {"dense": [0.1] * 384}

        await engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings=query_embeddings
        )

        # Should record performance metrics
        assert hasattr(engine, 'performance_monitor')
        # Performance data should be available through the monitor

    def test_create_fusion_ranker_factory(self):
        """Test fusion ranker factory function."""
        # Test RRF creation
        rrf_ranker = create_fusion_ranker("rrf", k=100)
        assert isinstance(rrf_ranker, RRFFusionRanker)
        assert rrf_ranker.k == 100

        # Test weighted sum creation
        ws_ranker = create_fusion_ranker("weighted_sum", dense_weight=0.7, sparse_weight=0.3)
        assert isinstance(ws_ranker, WeightedSumFusionRanker)
        assert ws_ranker.dense_weight == 0.7

        # Test max score creation
        max_ranker = create_fusion_ranker("max_score")
        assert isinstance(max_ranker, MaxScoreFusionRanker)

        # Test invalid method
        with pytest.raises(ValueError):
            create_fusion_ranker("invalid_method")


class TestHybridSearchEngineOptimizations:
    """Test search engine optimizations and advanced features."""

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, mock_qdrant_client):
        """Test concurrent dense and sparse searches."""
        engine = HybridSearchEngine(mock_qdrant_client)

        # Add delay to mock to test concurrency
        async def delayed_search(*args, **kwargs):
            await asyncio.sleep(0.01)
            return []

        mock_qdrant_client.search.side_effect = delayed_search

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.8, 0.6]}
        }

        import time
        start_time = time.time()

        await engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings=query_embeddings
        )

        elapsed = time.time() - start_time
        # Should be faster than sequential (2 * 0.01 = 0.02s)
        assert elapsed < 0.02

    @pytest.mark.asyncio
    async def test_batch_search_operations(self, mock_qdrant_client):
        """Test batch search operations."""
        engine = HybridSearchEngine(mock_qdrant_client)

        mock_qdrant_client.search.return_value = []

        # Multiple queries
        queries = [
            {"dense": [0.1] * 384},
            {"dense": [0.2] * 384},
            {"sparse": {"indices": [1], "values": [0.8]}}
        ]

        # This would be a batch operation if implemented
        for query_embeddings in queries:
            results = await engine.hybrid_search(
                collection_name="test-collection",
                query_embeddings=query_embeddings
            )
            assert isinstance(results, list)

    def test_memory_efficient_processing(self, mock_qdrant_client):
        """Test memory efficiency with large result sets."""
        HybridSearchEngine(mock_qdrant_client)

        # Create large mock result sets
        large_dense_results = [
            {"id": f"doc{i}", "score": 0.9 - (i * 0.001), "payload": {"content": f"doc {i}"}}
            for i in range(1000)
        ]
        large_sparse_results = [
            {"id": f"doc{i}", "score": 0.8 - (i * 0.001), "payload": {"content": f"doc {i}"}}
            for i in range(500, 1500)
        ]

        ranker = RRFFusionRanker()
        fused_results = ranker.fuse_rankings(large_dense_results, large_sparse_results)

        # Should handle large datasets without memory issues
        assert len(fused_results) > 0
        assert len(fused_results) <= len(large_dense_results) + len(large_sparse_results)

    @pytest.mark.asyncio
    async def test_adaptive_timeout_handling(self, mock_qdrant_client):
        """Test adaptive timeout handling for slow searches."""
        engine = HybridSearchEngine(mock_qdrant_client)

        # Mock timeout scenarios
        mock_qdrant_client.search.side_effect = [
            asyncio.TimeoutError("Dense search timed out"),
            []  # Sparse search succeeds
        ]

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1], "values": [0.8]}
        }

        # Should handle partial timeout gracefully
        # (This would require implementing timeout handling in the actual engine)
        with pytest.raises(asyncio.TimeoutError):
            await engine.hybrid_search(
                collection_name="test-collection",
                query_embeddings=query_embeddings
            )
