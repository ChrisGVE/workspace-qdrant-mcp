#!/usr/bin/env python3
"""
Focused test coverage for hybrid_search.py module
Target: 30%+ coverage with essential functionality tests
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from collections import defaultdict

# Import module under test
from src.python.common.core.hybrid_search import (
    TenantAwareResult,
    TenantAwareResultDeduplicator,
    MultiTenantResultAggregator,
    RRFFusionRanker,
    WeightedSumFusionRanker,
    HybridSearchEngine
)

class TestTenantAwareResult:
    """Test TenantAwareResult dataclass"""

    def test_init_basic(self):
        """Test basic initialization"""
        result = TenantAwareResult(
            id="test_id",
            score=0.95,
            payload={"content": "test"},
            collection="test_collection",
            search_type="dense"
        )
        assert result.id == "test_id"
        assert result.score == 0.95
        assert result.search_type == "dense"
        assert result.tenant_metadata == {}
        assert result.project_context == {}
        assert result.deduplication_key == "test_id"

    def test_deduplication_key_generation(self):
        """Test deduplication key generation with different payload types"""
        # With content_hash
        result = TenantAwareResult(
            id="test_id",
            score=0.5,
            payload={"content_hash": "abc123"},
            collection="test",
            search_type="sparse"
        )
        assert result.deduplication_key == "abc123"

        # With file_path fallback
        result2 = TenantAwareResult(
            id="test_id2",
            score=0.6,
            payload={"file_path": "/path/to/file.txt"},
            collection="test",
            search_type="dense"
        )
        assert result2.deduplication_key == "/path/to/file.txt"


class TestTenantAwareResultDeduplicator:
    """Test TenantAwareResultDeduplicator"""

    def test_init(self):
        """Test deduplicator initialization"""
        dedup = TenantAwareResultDeduplicator(preserve_tenant_isolation=False)
        assert not dedup.preserve_tenant_isolation

        dedup2 = TenantAwareResultDeduplicator()
        assert dedup2.preserve_tenant_isolation  # Default True

    def test_deduplicate_empty_results(self):
        """Test deduplication with empty results list"""
        dedup = TenantAwareResultDeduplicator()
        result = dedup.deduplicate_results([])
        assert result == []

    def test_deduplicate_single_result(self):
        """Test deduplication with single result"""
        dedup = TenantAwareResultDeduplicator()
        result = TenantAwareResult(
            id="test1", score=0.8, payload={}, collection="col1", search_type="dense"
        )
        deduplicated = dedup.deduplicate_results([result])
        assert len(deduplicated) == 1
        assert deduplicated[0].id == "test1"

    def test_deduplicate_duplicate_results(self):
        """Test deduplication with actual duplicates"""
        dedup = TenantAwareResultDeduplicator()
        result1 = TenantAwareResult(
            id="test1", score=0.8,
            payload={"content_hash": "same_hash"},
            collection="col1", search_type="dense"
        )
        result2 = TenantAwareResult(
            id="test2", score=0.9,
            payload={"content_hash": "same_hash"},
            collection="col1", search_type="sparse"
        )

        deduplicated = dedup.deduplicate_results([result1, result2])
        assert len(deduplicated) == 1
        assert deduplicated[0].score == 0.9  # Max score by default


class TestRRFFusionRanker:
    """Test RRF (Reciprocal Rank Fusion) ranker"""

    def test_init(self):
        """Test RRF ranker initialization"""
        ranker = RRFFusionRanker(k=60)
        assert ranker.k == 60
        assert ranker.boost_weights == {}

        ranker2 = RRFFusionRanker(k=100, boost_weights={"type1": 1.5})
        assert ranker2.k == 100
        assert ranker2.boost_weights == {"type1": 1.5}

    def test_fuse_empty_results(self):
        """Test fusion with empty result sets"""
        ranker = RRFFusionRanker()
        result = ranker.fuse([], [])
        assert result == []

    def test_fuse_basic(self):
        """Test basic RRF fusion"""
        ranker = RRFFusionRanker(k=60)

        # Create mock results
        dense_results = [Mock(id="doc1"), Mock(id="doc2")]
        sparse_results = [Mock(id="doc2"), Mock(id="doc3")]

        # Mock payload attribute
        for result in dense_results + sparse_results:
            result.payload = {}

        fused = ranker.fuse(dense_results, sparse_results)

        # Should have all unique documents
        assert len(fused) == 3
        # doc2 should be ranked highest (appears in both)
        assert fused[0].id == "doc2"

    def test_explain_fusion(self):
        """Test fusion explanation functionality"""
        ranker = RRFFusionRanker()

        dense_results = [Mock(id="doc1")]
        sparse_results = [Mock(id="doc2")]

        for result in dense_results + sparse_results:
            result.payload = {}

        explanation = ranker.explain_fusion(dense_results, sparse_results, top_k=2)

        assert "algorithm" in explanation
        assert explanation["algorithm"] == "Reciprocal Rank Fusion (RRF)"
        assert "parameters" in explanation
        assert explanation["parameters"]["k"] == 60


class TestWeightedSumFusionRanker:
    """Test WeightedSumFusionRanker"""

    def test_init(self):
        """Test weighted sum ranker initialization"""
        ranker = WeightedSumFusionRanker(dense_weight=0.8, sparse_weight=0.2)
        assert ranker.dense_weight == 0.8
        assert ranker.sparse_weight == 0.2

    def test_normalize_scores_empty(self):
        """Test score normalization with empty results"""
        ranker = WeightedSumFusionRanker()
        normalized = ranker._normalize_scores([])
        assert normalized == []

    def test_normalize_scores_equal(self):
        """Test score normalization with equal scores"""
        ranker = WeightedSumFusionRanker()
        mock_results = [Mock(score=0.5), Mock(score=0.5), Mock(score=0.5)]
        normalized = ranker._normalize_scores(mock_results)
        assert normalized == [1.0, 1.0, 1.0]


class TestHybridSearchEngine:
    """Test HybridSearchEngine main class"""

    def test_init_basic(self):
        """Test basic engine initialization"""
        mock_client = Mock()
        engine = HybridSearchEngine(
            client=mock_client,
            enable_optimizations=False,
            enable_multi_tenant_aggregation=False,
            enable_performance_monitoring=False
        )

        assert engine.client == mock_client
        assert not engine.optimizations_enabled
        assert not engine.multi_tenant_aggregation_enabled
        assert not engine.performance_monitoring_enabled
        assert engine.result_aggregator is None

    def test_init_with_optimizations(self):
        """Test initialization with optimizations enabled"""
        mock_client = Mock()
        engine = HybridSearchEngine(
            client=mock_client,
            enable_optimizations=True,
            enable_multi_tenant_aggregation=True,
            enable_performance_monitoring=True
        )

        assert engine.optimizations_enabled
        assert engine.multi_tenant_aggregation_enabled
        assert engine.performance_monitoring_enabled
        assert engine.result_aggregator is not None

    def test_max_score_fusion(self):
        """Test max score fusion method"""
        mock_client = Mock()
        engine = HybridSearchEngine(client=mock_client, enable_optimizations=False)

        dense_results = [Mock(id="doc1", score=0.8), Mock(id="doc2", score=0.6)]
        sparse_results = [Mock(id="doc2", score=0.9), Mock(id="doc3", score=0.5)]

        fused = engine._max_score_fusion(dense_results, sparse_results)

        # Should have 3 unique documents
        assert len(fused) == 3
        # doc2 should have max score of 0.9 and be ranked first
        assert fused[0].id == "doc2"
        assert fused[0].score == 0.9
        # doc1 should be second with score 0.8
        assert fused[1].id == "doc1"
        assert fused[1].score == 0.8

    def test_build_enhanced_filter_no_context(self):
        """Test filter building with no context"""
        mock_client = Mock()
        engine = HybridSearchEngine(client=mock_client, enable_optimizations=False)

        base_filter = Mock()
        result = engine._build_enhanced_filter(
            base_filter=base_filter,
            project_context=None,
            auto_inject=True
        )

        assert result == base_filter

    def test_build_enhanced_filter_no_auto_inject(self):
        """Test filter building with auto_inject disabled"""
        mock_client = Mock()
        engine = HybridSearchEngine(client=mock_client, enable_optimizations=False)

        base_filter = Mock()
        project_context = {"project_name": "test"}
        result = engine._build_enhanced_filter(
            base_filter=base_filter,
            project_context=project_context,
            auto_inject=False
        )

        assert result == base_filter


if __name__ == "__main__":
    # Run with coverage measurement
    import subprocess
    import sys

    print("Running hybrid_search.py focused tests with coverage...")

    # Run pytest with coverage for this specific module
    cmd = [
        sys.executable, "-m", "pytest",
        __file__,
        "--cov=src/python/common/core/hybrid_search",
        "--cov-report=term-missing",
        "--tb=short",
        "-v"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)