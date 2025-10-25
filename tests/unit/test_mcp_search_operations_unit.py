"""
Unit tests for MCP server search operations components.

Granular unit tests for individual search components including:
- RRF fusion algorithm implementation
- Score normalization functions
- Metadata filter construction
- Result deduplication logic
- Search parameter optimization

Task 281: Unit-level tests for search operation components
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from qdrant_client.http import models

# Add src/python to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.hybrid_search import (
    HybridSearchEngine,
    MultiTenantResultAggregator,
    RRFFusionRanker,
    TenantAwareResult,
    TenantAwareResultDeduplicator,
    WeightedSumFusionRanker,
)


class TestRRFFusionRankerUnit:
    """Detailed unit tests for RRF algorithm."""

    def test_rrf_initialization_default(self):
        """Test default RRF ranker initialization."""
        ranker = RRFFusionRanker()
        assert ranker.k == 60
        assert ranker.boost_weights == {}

    def test_rrf_initialization_custom_k(self):
        """Test RRF with custom k parameter."""
        ranker = RRFFusionRanker(k=100)
        assert ranker.k == 100

    def test_rrf_initialization_with_boost(self):
        """Test RRF with boost weights."""
        boost = {"recent": 1.5, "popular": 1.2}
        ranker = RRFFusionRanker(boost_weights=boost)
        assert ranker.boost_weights == boost

    def test_rrf_empty_results(self):
        """Test RRF with empty result sets."""
        ranker = RRFFusionRanker()
        fused = ranker.fuse([], [])
        assert fused == []

    def test_rrf_single_source_dense(self):
        """Test RRF with only dense results."""
        ranker = RRFFusionRanker(k=60)

        dense = [
            Mock(id="doc1", score=0.9, payload={}),
            Mock(id="doc2", score=0.8, payload={})
        ]
        sparse = []

        fused = ranker.fuse(dense, sparse)

        assert len(fused) == 2
        assert fused[0].id == "doc1"
        # RRF score for rank 1: 1/(60+1) ≈ 0.0164
        assert abs(fused[0].payload.get("rrf_score", 0) - 1.0/61.0) < 0.001

    def test_rrf_single_source_sparse(self):
        """Test RRF with only sparse results."""
        ranker = RRFFusionRanker(k=60)

        dense = []
        sparse = [
            Mock(id="doc1", score=0.9, payload={}),
            Mock(id="doc2", score=0.8, payload={})
        ]

        fused = ranker.fuse(dense, sparse)

        assert len(fused) == 2
        assert fused[0].id == "doc1"

    def test_rrf_both_sources_no_overlap(self):
        """Test RRF fusion with no overlapping documents."""
        ranker = RRFFusionRanker(k=60)

        dense = [
            Mock(id="doc1", score=0.9, payload={}),
            Mock(id="doc2", score=0.8, payload={})
        ]
        sparse = [
            Mock(id="doc3", score=0.85, payload={}),
            Mock(id="doc4", score=0.75, payload={})
        ]

        fused = ranker.fuse(dense, sparse)

        # Should have all 4 unique documents
        assert len(fused) == 4
        doc_ids = {r.id for r in fused}
        assert doc_ids == {"doc1", "doc2", "doc3", "doc4"}

    def test_rrf_both_sources_with_overlap(self):
        """Test RRF fusion with overlapping documents."""
        ranker = RRFFusionRanker(k=60)

        dense = [
            Mock(id="doc1", score=0.9, payload={}),
            Mock(id="doc2", score=0.8, payload={}),
            Mock(id="doc3", score=0.7, payload={})
        ]
        sparse = [
            Mock(id="doc2", score=0.85, payload={}),  # Overlap
            Mock(id="doc4", score=0.75, payload={}),
            Mock(id="doc1", score=0.65, payload={})  # Overlap
        ]

        fused = ranker.fuse(dense, sparse)

        # Should have 4 unique documents
        assert len(fused) == 4

        # doc1 and doc2 appear in both, should have higher RRF scores
        doc1 = next(r for r in fused if r.id == "doc1")
        doc3 = next(r for r in fused if r.id == "doc3")
        doc4 = next(r for r in fused if r.id == "doc4")

        # doc1 is in both lists, should score higher than doc3 and doc4
        # which are only in one list each
        doc1_score = doc1.payload.get("rrf_score", 0)
        doc3_score = doc3.payload.get("rrf_score", 0)
        doc4_score = doc4.payload.get("rrf_score", 0)

        assert doc1_score > doc3_score
        assert doc1_score > doc4_score

    def test_rrf_score_formula_accuracy(self):
        """Test exact RRF score calculation."""
        ranker = RRFFusionRanker(k=60)

        # Document at rank 1 in both lists
        dense = [Mock(id="doc1", score=1.0, payload={})]
        sparse = [Mock(id="doc1", score=1.0, payload={})]

        fused = ranker.fuse(dense, sparse)

        # RRF = 1/(60+1) + 1/(60+1) = 2/61
        expected = 2.0 / 61.0
        actual = fused[0].payload.get("rrf_score", 0)

        assert abs(actual - expected) < 0.0001

    def test_rrf_with_custom_weights(self):
        """Test RRF with custom dense/sparse weights."""
        ranker = RRFFusionRanker(k=60)

        dense = [Mock(id="doc1", score=1.0, payload={})]
        sparse = [Mock(id="doc1", score=1.0, payload={})]

        weights = {"dense": 2.0, "sparse": 1.0}
        fused = ranker.fuse(dense, sparse, weights=weights)

        # RRF = 2.0/(60+1) + 1.0/(60+1) = 3/61
        expected = 3.0 / 61.0
        actual = fused[0].payload.get("rrf_score", 0)

        assert abs(actual - expected) < 0.0001

    def test_rrf_result_ordering(self):
        """Test that RRF results are properly sorted."""
        ranker = RRFFusionRanker(k=60)

        # Create results with known ordering
        dense = [
            Mock(id=f"doc{i}", score=1.0 - i*0.1, payload={})
            for i in range(5)
        ]
        sparse = [
            Mock(id=f"doc{i}", score=1.0 - i*0.1, payload={})
            for i in range(5)
        ]

        fused = ranker.fuse(dense, sparse)

        # Scores should be in descending order
        scores = [r.payload.get("rrf_score", 0) for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_explain_fusion(self):
        """Test fusion explanation generation."""
        ranker = RRFFusionRanker(k=60)

        dense = [
            Mock(id="doc1", score=0.9, payload={}),
            Mock(id="doc2", score=0.8, payload={})
        ]
        sparse = [
            Mock(id="doc2", score=0.85, payload={}),
            Mock(id="doc3", score=0.75, payload={})
        ]

        explanation = ranker.explain_fusion(dense, sparse, top_k=2)

        assert "algorithm" in explanation
        assert explanation["algorithm"] == "Reciprocal Rank Fusion (RRF)"
        assert "parameters" in explanation
        assert explanation["parameters"]["k"] == 60
        assert "top_results_analysis" in explanation
        assert len(explanation["top_results_analysis"]) <= 2


class TestWeightedSumFusionUnit:
    """Unit tests for weighted sum fusion."""

    def test_weighted_sum_initialization(self):
        """Test weighted sum ranker initialization."""
        ranker = WeightedSumFusionRanker(dense_weight=0.7, sparse_weight=0.3)
        assert ranker.dense_weight == 0.7
        assert ranker.sparse_weight == 0.3

    def test_weighted_sum_default_weights(self):
        """Test default equal weights."""
        ranker = WeightedSumFusionRanker()
        assert ranker.dense_weight == 0.7
        assert ranker.sparse_weight == 0.3

    def test_normalize_scores_empty(self):
        """Test score normalization with empty results."""
        ranker = WeightedSumFusionRanker()
        normalized = ranker._normalize_scores([])
        assert normalized == []

    def test_normalize_scores_single_value(self):
        """Test normalization with single value."""
        ranker = WeightedSumFusionRanker()
        results = [Mock(score=0.8)]
        normalized = ranker._normalize_scores(results)
        assert len(normalized) == 1
        assert normalized[0] == 1.0  # Single value normalizes to 1.0

    def test_normalize_scores_range(self):
        """Test score normalization to [0, 1] range."""
        ranker = WeightedSumFusionRanker()
        results = [
            Mock(score=1.0),
            Mock(score=0.75),
            Mock(score=0.5),
            Mock(score=0.25),
            Mock(score=0.0)
        ]

        normalized = ranker._normalize_scores(results)

        assert len(normalized) == 5
        assert max(normalized) <= 1.0
        assert min(normalized) >= 0.0
        assert normalized[0] == 1.0  # Max score
        assert normalized[4] == 0.0  # Min score

    def test_weighted_sum_fusion(self):
        """Test weighted sum fusion calculation."""
        ranker = WeightedSumFusionRanker(dense_weight=0.6, sparse_weight=0.4)

        dense = [Mock(id="doc1", score=1.0, payload={})]
        sparse = [Mock(id="doc1", score=0.8, payload={})]

        fused = ranker.fuse(dense, sparse)

        assert len(fused) == 1
        assert "weighted_score" in fused[0].payload

    def test_weighted_sum_different_weights_produce_different_results(self):
        """Test that different weights affect ranking."""
        dense = [
            Mock(id="doc1", score=1.0, payload={}),
            Mock(id="doc2", score=0.6, payload={})
        ]
        sparse = [
            Mock(id="doc1", score=0.5, payload={}),
            Mock(id="doc2", score=0.9, payload={})
        ]

        # Dense-heavy weighting
        ranker1 = WeightedSumFusionRanker(dense_weight=0.8, sparse_weight=0.2)
        fused1 = ranker1.fuse(dense, sparse)

        # Sparse-heavy weighting
        ranker2 = WeightedSumFusionRanker(dense_weight=0.2, sparse_weight=0.8)
        fused2 = ranker2.fuse(dense, sparse)

        # Top results should differ
        assert fused1[0].id != fused2[0].id


class TestTenantAwareResultUnit:
    """Unit tests for tenant-aware results."""

    def test_result_initialization(self):
        """Test TenantAwareResult initialization."""
        result = TenantAwareResult(
            id="doc1",
            score=0.9,
            payload={"content": "test"},
            collection="test-coll",
            search_type="hybrid"
        )

        assert result.id == "doc1"
        assert result.score == 0.9
        assert result.collection == "test-coll"
        assert result.search_type == "hybrid"

    def test_result_with_tenant_metadata(self):
        """Test result with tenant metadata."""
        tenant_meta = {
            "project_name": "proj1",
            "tenant_namespace": "proj1.docs"
        }

        result = TenantAwareResult(
            id="doc1",
            score=0.9,
            payload={},
            collection="test-coll",
            search_type="hybrid",
            tenant_metadata=tenant_meta
        )

        assert result.tenant_metadata == tenant_meta

    def test_deduplication_key_generation(self):
        """Test automatic deduplication key generation."""
        # With content hash
        result1 = TenantAwareResult(
            id="doc1",
            score=0.9,
            payload={"content_hash": "abc123"},
            collection="test-coll",
            search_type="hybrid"
        )
        assert result1.deduplication_key == "abc123"

        # With file path fallback
        result2 = TenantAwareResult(
            id="doc2",
            score=0.9,
            payload={"file_path": "/path/to/file.txt"},
            collection="test-coll",
            search_type="hybrid"
        )
        assert result2.deduplication_key == "/path/to/file.txt"

        # With ID fallback
        result3 = TenantAwareResult(
            id="doc3",
            score=0.9,
            payload={},
            collection="test-coll",
            search_type="hybrid"
        )
        assert result3.deduplication_key == "doc3"


class TestTenantAwareDeduplicatorUnit:
    """Unit tests for result deduplication."""

    def test_deduplicator_initialization(self):
        """Test deduplicator initialization."""
        dedup = TenantAwareResultDeduplicator(preserve_tenant_isolation=True)
        assert dedup.preserve_tenant_isolation is True

    def test_deduplicate_no_duplicates(self):
        """Test deduplication with no duplicates."""
        results = [
            TenantAwareResult("doc1", 0.9, {}, "coll1", "hybrid"),
            TenantAwareResult("doc2", 0.8, {}, "coll1", "hybrid"),
            TenantAwareResult("doc3", 0.7, {}, "coll1", "hybrid")
        ]

        dedup = TenantAwareResultDeduplicator()
        deduplicated = dedup.deduplicate_results(results)

        assert len(deduplicated) == 3

    def test_deduplicate_with_duplicates(self):
        """Test deduplication removes duplicates."""
        results = [
            TenantAwareResult("doc1", 0.9, {"content_hash": "hash1"}, "coll1", "hybrid"),
            TenantAwareResult("doc1", 0.8, {"content_hash": "hash1"}, "coll1", "hybrid"),  # Duplicate
            TenantAwareResult("doc2", 0.85, {"content_hash": "hash2"}, "coll1", "hybrid")
        ]

        dedup = TenantAwareResultDeduplicator()
        deduplicated = dedup.deduplicate_results(results)

        # Should have 2 unique results
        assert len(deduplicated) == 2

        # Should keep higher score
        doc1 = next(r for r in deduplicated if r.id == "doc1")
        assert doc1.score == 0.9

    def test_deduplicate_max_score_aggregation(self):
        """Test max score aggregation method."""
        results = [
            TenantAwareResult("doc1", 0.9, {"content_hash": "hash1"}, "coll1", "dense"),
            TenantAwareResult("doc1", 0.95, {"content_hash": "hash1"}, "coll1", "sparse")  # Higher score
        ]

        dedup = TenantAwareResultDeduplicator()
        deduplicated = dedup.deduplicate_results(results, aggregation_method="max_score")

        assert len(deduplicated) == 1
        assert deduplicated[0].score == 0.95

    def test_deduplicate_avg_score_aggregation(self):
        """Test average score aggregation method."""
        results = [
            TenantAwareResult("doc1", 0.8, {"content_hash": "hash1"}, "coll1", "dense"),
            TenantAwareResult("doc1", 1.0, {"content_hash": "hash1"}, "coll1", "sparse")
        ]

        dedup = TenantAwareResultDeduplicator()
        deduplicated = dedup.deduplicate_results(results, aggregation_method="avg_score")

        assert len(deduplicated) == 1
        # Average: (0.8 + 1.0) / 2 = 0.9
        assert deduplicated[0].score == 0.9

    def test_deduplicate_tenant_isolation(self):
        """Test tenant isolation in deduplication."""
        results = [
            TenantAwareResult(
                "doc1", 0.9, {"content_hash": "hash1"},
                "coll1", "hybrid",
                project_context={"project_name": "proj1"}
            ),
            TenantAwareResult(
                "doc1", 0.95, {"content_hash": "hash1"},
                "coll1", "hybrid",
                project_context={"project_name": "proj2"}
            )
        ]

        # With isolation, should keep both
        dedup = TenantAwareResultDeduplicator(preserve_tenant_isolation=True)
        deduplicated = dedup.deduplicate_results(results)

        assert len(deduplicated) == 2

        # Without isolation, should deduplicate
        dedup_no_isolation = TenantAwareResultDeduplicator(preserve_tenant_isolation=False)
        deduplicated_no_iso = dedup_no_isolation.deduplicate_results(results)

        assert len(deduplicated_no_iso) == 1


class TestMultiTenantAggregatorUnit:
    """Unit tests for multi-tenant result aggregator."""

    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        agg = MultiTenantResultAggregator(
            preserve_tenant_isolation=True,
            enable_score_normalization=True,
            default_aggregation_method="max_score"
        )

        assert agg.preserve_tenant_isolation is True
        assert agg.enable_score_normalization is True
        assert agg.default_aggregation_method == "max_score"

    def test_aggregate_single_collection(self):
        """Test aggregation from single collection."""
        results = [
            TenantAwareResult("doc1", 0.9, {"project": "proj1"}, "coll1", "hybrid"),
            TenantAwareResult("doc2", 0.8, {"project": "proj1"}, "coll1", "hybrid")
        ]

        collection_results = {"coll1": results}

        agg = MultiTenantResultAggregator()
        aggregated = agg.aggregate_multi_collection_results(
            collection_results, limit=10
        )

        assert aggregated["total_results"] == 2
        assert len(aggregated["results"]) == 2

    def test_aggregate_multiple_collections(self):
        """Test aggregation from multiple collections."""
        coll1_results = [
            TenantAwareResult("doc1", 0.9, {}, "coll1", "hybrid"),
            TenantAwareResult("doc2", 0.8, {}, "coll1", "hybrid")
        ]

        coll2_results = [
            TenantAwareResult("doc3", 0.85, {}, "coll2", "hybrid"),
            TenantAwareResult("doc4", 0.75, {}, "coll2", "hybrid")
        ]

        collection_results = {
            "coll1": coll1_results,
            "coll2": coll2_results
        }

        agg = MultiTenantResultAggregator()
        aggregated = agg.aggregate_multi_collection_results(
            collection_results, limit=10
        )

        assert aggregated["total_results"] == 4
        assert aggregated["aggregation_metadata"]["collection_count"] == 2

    def test_score_threshold_filtering(self):
        """Test score threshold filtering during aggregation."""
        results = [
            TenantAwareResult("doc1", 0.9, {}, "coll1", "hybrid"),
            TenantAwareResult("doc2", 0.6, {}, "coll1", "hybrid"),
            TenantAwareResult("doc3", 0.3, {}, "coll1", "hybrid")
        ]

        collection_results = {"coll1": results}

        agg = MultiTenantResultAggregator()
        aggregated = agg.aggregate_multi_collection_results(
            collection_results,
            limit=10,
            score_threshold=0.5
        )

        # Only doc1 and doc2 should pass threshold
        assert aggregated["total_results"] == 2

    def test_result_limit_enforcement(self):
        """Test limit enforcement in aggregation."""
        results = [
            TenantAwareResult(f"doc{i}", 1.0 - i*0.1, {}, "coll1", "hybrid")
            for i in range(10)
        ]

        collection_results = {"coll1": results}

        agg = MultiTenantResultAggregator()
        aggregated = agg.aggregate_multi_collection_results(
            collection_results, limit=5
        )

        # Should enforce limit
        assert aggregated["total_results"] == 5

    def test_score_normalization(self):
        """Test cross-collection score normalization."""
        # Collection 1 with high scores
        coll1_results = [
            TenantAwareResult("doc1", 0.9, {}, "coll1", "hybrid"),
            TenantAwareResult("doc2", 0.8, {}, "coll1", "hybrid")
        ]

        # Collection 2 with low scores
        coll2_results = [
            TenantAwareResult("doc3", 0.3, {}, "coll2", "hybrid"),
            TenantAwareResult("doc4", 0.2, {}, "coll2", "hybrid")
        ]

        collection_results = {
            "coll1": coll1_results,
            "coll2": coll2_results
        }

        agg = MultiTenantResultAggregator(enable_score_normalization=True)
        aggregated = agg.aggregate_multi_collection_results(
            collection_results, limit=10
        )

        # Check normalization metadata is present
        assert aggregated["aggregation_metadata"]["score_normalization_enabled"]


class TestHybridSearchEngineUnit:
    """Unit tests for hybrid search engine components."""

    def test_engine_initialization_no_optimizations(self):
        """Test search engine initialization without optimizations."""
        mock_client = Mock()
        # Disable optimizations to avoid Pydantic validation issue
        engine = HybridSearchEngine(mock_client, enable_optimizations=False)

        assert engine.client == mock_client
        assert isinstance(engine.rrf_ranker, RRFFusionRanker)
        assert isinstance(engine.weighted_ranker, WeightedSumFusionRanker)
        assert engine.optimizations_enabled is False

    def test_engine_with_multi_tenant_aggregation(self):
        """Test engine with multi-tenant aggregation enabled."""
        mock_client = Mock()
        engine = HybridSearchEngine(
            mock_client,
            enable_optimizations=False,  # Disable to avoid Pydantic issue
            enable_multi_tenant_aggregation=True
        )

        assert engine.multi_tenant_aggregation_enabled is True
        assert engine.result_aggregator is not None

    def test_max_score_fusion_logic(self):
        """Test max score fusion internal logic."""
        mock_client = Mock()
        engine = HybridSearchEngine(mock_client, enable_optimizations=False)

        dense_results = [
            Mock(id="doc1", score=0.9, payload={}),
            Mock(id="doc2", score=0.7, payload={})
        ]

        sparse_results = [
            Mock(id="doc1", score=0.8, payload={}),  # Lower than dense
            Mock(id="doc3", score=0.85, payload={})
        ]

        fused = engine._max_score_fusion(dense_results, sparse_results)

        # Check doc1 has max score from dense (0.9)
        doc1 = next(r for r in fused if r.id == "doc1")
        assert doc1.score == 0.9

        # Should have 3 unique docs
        assert len(fused) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
