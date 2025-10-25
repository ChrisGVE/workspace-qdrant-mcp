"""
Comprehensive unit tests for Reciprocal Rank Fusion (RRF) algorithm.

This test suite validates the RRF implementation in hybrid_search.py with focus on:
- Mathematical correctness of RRF formula
- Different ranking weights and fusion parameters
- Edge cases (empty, single, duplicate results)
- Validation that fusion improves search quality over individual methods

Task 322.1: Implement comprehensive tests for RRF fusion algorithm

Execution: uv run pytest tests/unit/test_rrf_fusion_algorithm.py -v
"""

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

# Test imports
try:
    from common.core.hybrid_search import RRFFusionRanker
    RRF_AVAILABLE = True
except ImportError as e:
    RRF_AVAILABLE = False
    print(f"RRF import failed: {e}")

pytestmark = pytest.mark.skipif(not RRF_AVAILABLE, reason="RRF module not available")


# Helper functions for creating mock results
def create_mock_result(doc_id: str, score: float, payload: dict = None):
    """Create a mock search result for testing."""
    result = Mock()
    result.id = doc_id
    result.score = score
    result.payload = payload if payload is not None else {}
    return result


def create_dense_results(count: int, score_start: float = 0.9, score_step: float = -0.1):
    """Create mock dense search results with decreasing scores."""
    results = []
    for i in range(count):
        score = score_start + (i * score_step)
        result = create_mock_result(f"dense_doc_{i}", score, {"type": "dense"})
        results.append(result)
    return results


def create_sparse_results(count: int, score_start: float = 0.8, score_step: float = -0.15):
    """Create mock sparse search results with decreasing scores."""
    results = []
    for i in range(count):
        score = score_start + (i * score_step)
        result = create_mock_result(f"sparse_doc_{i}", score, {"type": "sparse"})
        results.append(result)
    return results


def create_overlapping_results(dense_count: int, sparse_count: int, overlap_count: int):
    """Create dense and sparse results with specified overlap."""
    dense_results = []
    sparse_results = []

    # Create overlapping documents
    for i in range(overlap_count):
        doc_id = f"overlap_doc_{i}"
        dense_results.append(create_mock_result(doc_id, 0.9 - i * 0.1, {"type": "both"}))
        sparse_results.append(create_mock_result(doc_id, 0.85 - i * 0.15, {"type": "both"}))

    # Create unique dense documents
    for i in range(dense_count - overlap_count):
        dense_results.append(
            create_mock_result(f"dense_unique_{i}", 0.7 - i * 0.1, {"type": "dense"})
        )

    # Create unique sparse documents
    for i in range(sparse_count - overlap_count):
        sparse_results.append(
            create_mock_result(f"sparse_unique_{i}", 0.6 - i * 0.1, {"type": "sparse"})
        )

    return dense_results, sparse_results


class TestRRFMathematicalCorrectness:
    """Test mathematical correctness of RRF formula: RRF(d) = Σ(1 / (k + r(d)))"""

    def test_rrf_formula_single_document_dense_only(self):
        """Test RRF formula for document appearing only in dense results."""
        ranker = RRFFusionRanker(k=60)

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = []

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 1.0, "sparse": 1.0})

        # Formula: RRF(doc_1) = 1.0 / (60 + 1) = 1/61 ≈ 0.01639
        expected_rrf = 1.0 / (60 + 1)
        actual_rrf = fused[0].payload.get("rrf_score", 0)

        assert abs(actual_rrf - expected_rrf) < 1e-10, f"Expected {expected_rrf}, got {actual_rrf}"

    def test_rrf_formula_single_document_sparse_only(self):
        """Test RRF formula for document appearing only in sparse results."""
        ranker = RRFFusionRanker(k=60)

        dense_results = []
        sparse_results = [create_mock_result("doc_1", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 1.0, "sparse": 1.0})

        # Formula: RRF(doc_1) = 1.0 / (60 + 1) = 1/61 ≈ 0.01639
        expected_rrf = 1.0 / (60 + 1)
        actual_rrf = fused[0].payload.get("rrf_score", 0)

        assert abs(actual_rrf - expected_rrf) < 1e-10, f"Expected {expected_rrf}, got {actual_rrf}"

    def test_rrf_formula_overlapping_document(self):
        """Test RRF formula for document appearing in both dense and sparse results."""
        ranker = RRFFusionRanker(k=60)

        # Same document appears at rank 1 in both dense and sparse
        dense_results = [create_mock_result("doc_overlap", 0.9)]
        sparse_results = [create_mock_result("doc_overlap", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 1.0, "sparse": 1.0})

        # Formula: RRF(doc_overlap) = 1.0/(60+1) + 1.0/(60+1) = 2/61
        expected_rrf = (1.0 / 61) + (1.0 / 61)
        actual_rrf = fused[0].payload.get("rrf_score", 0)

        assert abs(actual_rrf - expected_rrf) < 1e-10, f"Expected {expected_rrf}, got {actual_rrf}"

    def test_rrf_formula_rank_position_impact(self):
        """Test that rank position correctly impacts RRF scores."""
        ranker = RRFFusionRanker(k=60)

        dense_results = [
            create_mock_result("doc_1", 0.9),  # Rank 1
            create_mock_result("doc_2", 0.8),  # Rank 2
            create_mock_result("doc_3", 0.7),  # Rank 3
        ]
        sparse_results = []

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 1.0, "sparse": 1.0})

        # Expected RRF scores: 1/(60+1), 1/(60+2), 1/(60+3)
        expected_scores = [1.0 / (60 + i + 1) for i in range(3)]

        for i, result in enumerate(fused):
            actual_rrf = result.payload.get("rrf_score", 0)
            assert abs(actual_rrf - expected_scores[i]) < 1e-10

    def test_rrf_formula_with_weights(self):
        """Test RRF formula with different dense/sparse weights."""
        ranker = RRFFusionRanker(k=60)

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = [create_mock_result("doc_1", 0.8)]

        # Dense weight = 2.0, sparse weight = 0.5
        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 2.0, "sparse": 0.5})

        # Formula: RRF(doc_1) = 2.0/(60+1) + 0.5/(60+1) = 2.5/61
        expected_rrf = (2.0 / 61) + (0.5 / 61)
        actual_rrf = fused[0].payload.get("rrf_score", 0)

        assert abs(actual_rrf - expected_rrf) < 1e-10, f"Expected {expected_rrf}, got {actual_rrf}"

    def test_rrf_k_parameter_influence(self):
        """Test that k parameter correctly influences RRF scores."""
        # Test with k=10
        ranker_small_k = RRFFusionRanker(k=10)
        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = []

        fused_small = ranker_small_k.fuse(dense_results, sparse_results)
        score_small_k = fused_small[0].payload.get("rrf_score", 0)

        # Test with k=100
        ranker_large_k = RRFFusionRanker(k=100)
        fused_large = ranker_large_k.fuse(dense_results, sparse_results)
        score_large_k = fused_large[0].payload.get("rrf_score", 0)

        # Smaller k should give higher scores: 1/11 > 1/101
        assert score_small_k > score_large_k
        assert abs(score_small_k - 1.0/11) < 1e-10
        assert abs(score_large_k - 1.0/101) < 1e-10


class TestRRFBasicFusion:
    """Test basic fusion scenarios with different result set combinations."""

    def test_both_empty_result_sets(self):
        """Test fusion with both empty result sets."""
        ranker = RRFFusionRanker()
        fused = ranker.fuse([], [])
        assert fused == []

    def test_empty_dense_non_empty_sparse(self):
        """Test fusion with empty dense and non-empty sparse results."""
        ranker = RRFFusionRanker()

        dense_results = []
        sparse_results = create_sparse_results(3)

        fused = ranker.fuse(dense_results, sparse_results)

        assert len(fused) == 3
        # Should contain all sparse results
        fused_ids = [r.id for r in fused]
        assert all(r.id in fused_ids for r in sparse_results)

    def test_empty_sparse_non_empty_dense(self):
        """Test fusion with empty sparse and non-empty dense results."""
        ranker = RRFFusionRanker()

        dense_results = create_dense_results(3)
        sparse_results = []

        fused = ranker.fuse(dense_results, sparse_results)

        assert len(fused) == 3
        # Should contain all dense results
        fused_ids = [r.id for r in fused]
        assert all(r.id in fused_ids for r in dense_results)

    def test_single_result_in_each_set(self):
        """Test fusion with single result in each set (no overlap)."""
        ranker = RRFFusionRanker()

        dense_results = [create_mock_result("doc_dense", 0.9)]
        sparse_results = [create_mock_result("doc_sparse", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results)

        assert len(fused) == 2
        fused_ids = [r.id for r in fused]
        assert "doc_dense" in fused_ids
        assert "doc_sparse" in fused_ids

    def test_no_overlap_multiple_results(self):
        """Test fusion with multiple results and no overlap."""
        ranker = RRFFusionRanker()

        dense_results = create_dense_results(5)
        sparse_results = create_sparse_results(5)

        fused = ranker.fuse(dense_results, sparse_results)

        # Should have all unique documents
        assert len(fused) == 10
        fused_ids = {r.id for r in fused}
        assert len(fused_ids) == 10

    def test_complete_overlap(self):
        """Test fusion with complete overlap (all same documents)."""
        ranker = RRFFusionRanker()

        # Create identical document lists
        dense_results = [
            create_mock_result("doc_1", 0.9),
            create_mock_result("doc_2", 0.8),
            create_mock_result("doc_3", 0.7),
        ]
        sparse_results = [
            create_mock_result("doc_1", 0.85),
            create_mock_result("doc_2", 0.75),
            create_mock_result("doc_3", 0.65),
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Should have only 3 unique documents
        assert len(fused) == 3
        fused_ids = {r.id for r in fused}
        assert fused_ids == {"doc_1", "doc_2", "doc_3"}

    def test_partial_overlap(self):
        """Test fusion with partial overlap between dense and sparse."""
        ranker = RRFFusionRanker()

        dense_results, sparse_results = create_overlapping_results(5, 5, 2)

        fused = ranker.fuse(dense_results, sparse_results)

        # Should have 5 dense + 5 sparse - 2 overlap = 8 unique documents
        fused_ids = {r.id for r in fused}
        assert len(fused_ids) == 8


class TestRRFWeightVariations:
    """Test RRF fusion with different weight configurations."""

    def test_equal_weights(self):
        """Test fusion with equal weights (1.0, 1.0)."""
        ranker = RRFFusionRanker()

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = [create_mock_result("doc_2", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 1.0, "sparse": 1.0})

        assert len(fused) == 2
        # Both should have equal RRF scores since they're rank 1 in their respective lists
        assert fused[0].payload.get("rrf_score", 0) == fused[1].payload.get("rrf_score", 0)

    def test_dense_biased_weights(self):
        """Test fusion with dense-biased weights (2.0, 0.5)."""
        ranker = RRFFusionRanker()

        # Same document in both lists at rank 1
        dense_results = [create_mock_result("doc_overlap", 0.9)]
        sparse_results = [
            create_mock_result("doc_overlap", 0.8),
            create_mock_result("doc_sparse", 0.75),
        ]

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 2.0, "sparse": 0.5})

        # doc_overlap should be first due to higher dense weight
        assert fused[0].id == "doc_overlap"

        # Verify RRF score: 2.0/(60+1) + 0.5/(60+1) = 2.5/61
        expected_overlap_score = (2.0 / 61) + (0.5 / 61)
        actual_overlap_score = fused[0].payload.get("rrf_score", 0)
        assert abs(actual_overlap_score - expected_overlap_score) < 1e-10

    def test_sparse_biased_weights(self):
        """Test fusion with sparse-biased weights (0.5, 2.0)."""
        ranker = RRFFusionRanker()

        # Same document in both lists at rank 1
        dense_results = [create_mock_result("doc_overlap", 0.9)]
        sparse_results = [create_mock_result("doc_overlap", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 0.5, "sparse": 2.0})

        # Verify RRF score: 0.5/(60+1) + 2.0/(60+1) = 2.5/61
        expected_score = (0.5 / 61) + (2.0 / 61)
        actual_score = fused[0].payload.get("rrf_score", 0)
        assert abs(actual_score - expected_score) < 1e-10

    def test_zero_dense_weight(self):
        """Test fusion with zero dense weight (edge case)."""
        ranker = RRFFusionRanker()

        dense_results = [create_mock_result("doc_dense", 0.9)]
        sparse_results = [create_mock_result("doc_sparse", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 0.0, "sparse": 1.0})

        # doc_dense should have RRF score of 0 (0 * 1/61)
        # doc_sparse should have RRF score of 1/61
        sparse_result = [r for r in fused if r.id == "doc_sparse"][0]
        assert sparse_result.payload.get("rrf_score", 0) > 0

    def test_very_large_weights(self):
        """Test fusion with very large weights (stress test)."""
        ranker = RRFFusionRanker()

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = [create_mock_result("doc_1", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results, weights={"dense": 1000.0, "sparse": 500.0})

        # Should still calculate correctly: 1000/(60+1) + 500/(60+1) = 1500/61
        expected_score = (1000.0 / 61) + (500.0 / 61)
        actual_score = fused[0].payload.get("rrf_score", 0)
        assert abs(actual_score - expected_score) < 1e-6  # Slightly larger tolerance for large numbers


class TestRRFKParameterVariations:
    """Test RRF with different k parameter values."""

    def test_default_k_60(self):
        """Test with default k=60."""
        ranker = RRFFusionRanker()  # Default k=60

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = []

        fused = ranker.fuse(dense_results, sparse_results)

        expected_score = 1.0 / 61
        actual_score = fused[0].payload.get("rrf_score", 0)
        assert abs(actual_score - expected_score) < 1e-10

    def test_small_k_10(self):
        """Test with small k=10."""
        ranker = RRFFusionRanker(k=10)

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = []

        fused = ranker.fuse(dense_results, sparse_results)

        expected_score = 1.0 / 11
        actual_score = fused[0].payload.get("rrf_score", 0)
        assert abs(actual_score - expected_score) < 1e-10

    def test_large_k_100(self):
        """Test with large k=100."""
        ranker = RRFFusionRanker(k=100)

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = []

        fused = ranker.fuse(dense_results, sparse_results)

        expected_score = 1.0 / 101
        actual_score = fused[0].payload.get("rrf_score", 0)
        assert abs(actual_score - expected_score) < 1e-10

    def test_k_zero_edge_case(self):
        """Test with k=0 (edge case - should still work without division by zero)."""
        ranker = RRFFusionRanker(k=0)

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = []

        fused = ranker.fuse(dense_results, sparse_results)

        # With k=0, rank 1 gives: 1/(0+1) = 1.0
        expected_score = 1.0 / 1
        actual_score = fused[0].payload.get("rrf_score", 0)
        assert abs(actual_score - expected_score) < 1e-10

    def test_k_parameter_affects_rank_distance(self):
        """Test that k parameter affects the importance of rank distance."""
        dense_results = [
            create_mock_result("doc_1", 0.9),  # Rank 1
            create_mock_result("doc_2", 0.8),  # Rank 2
        ]
        sparse_results = []

        # Small k - rank distance matters more
        ranker_small = RRFFusionRanker(k=1)
        fused_small = ranker_small.fuse(dense_results, sparse_results)
        score_1_small = fused_small[0].payload.get("rrf_score", 0)  # 1/2
        score_2_small = fused_small[1].payload.get("rrf_score", 0)  # 1/3
        ratio_small = score_1_small / score_2_small  # Should be 1.5

        # Large k - rank distance matters less
        ranker_large = RRFFusionRanker(k=1000)
        fused_large = ranker_large.fuse(dense_results, sparse_results)
        score_1_large = fused_large[0].payload.get("rrf_score", 0)  # 1/1001
        score_2_large = fused_large[1].payload.get("rrf_score", 0)  # 1/1002
        ratio_large = score_1_large / score_2_large  # Should be ~1.001

        # Smaller k should have higher ratio (rank matters more)
        assert ratio_small > ratio_large
        assert abs(ratio_small - 1.5) < 1e-6
        assert abs(ratio_large - (1002/1001)) < 1e-6


class TestRRFDuplicateHandling:
    """Test RRF handling of duplicate documents across dense and sparse results."""

    def test_same_document_different_ranks(self):
        """Test document appearing at different ranks in dense and sparse."""
        ranker = RRFFusionRanker(k=60)

        # doc_1 at rank 1 in dense, rank 3 in sparse
        dense_results = [
            create_mock_result("doc_1", 0.9),
            create_mock_result("doc_2", 0.8),
        ]
        sparse_results = [
            create_mock_result("doc_3", 0.85),
            create_mock_result("doc_4", 0.8),
            create_mock_result("doc_1", 0.75),  # Same as dense doc_1
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Find doc_1 in fused results
        doc_1 = [r for r in fused if r.id == "doc_1"][0]

        # RRF score: 1/(60+1) + 1/(60+3) = 1/61 + 1/63
        expected_score = (1.0 / 61) + (1.0 / 63)
        actual_score = doc_1.payload.get("rrf_score", 0)

        assert abs(actual_score - expected_score) < 1e-10

    def test_duplicate_score_aggregation(self):
        """Test correct score aggregation for duplicate documents."""
        ranker = RRFFusionRanker(k=60)

        # Multiple documents with some duplicates
        dense_results = [
            create_mock_result("doc_A", 0.9),
            create_mock_result("doc_B", 0.8),
            create_mock_result("doc_C", 0.7),
        ]
        sparse_results = [
            create_mock_result("doc_B", 0.85),  # Duplicate
            create_mock_result("doc_C", 0.8),   # Duplicate
            create_mock_result("doc_D", 0.75),
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Verify doc_B aggregation: rank 2 in dense + rank 1 in sparse
        doc_B = [r for r in fused if r.id == "doc_B"][0]
        expected_B_score = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
        actual_B_score = doc_B.payload.get("rrf_score", 0)
        assert abs(actual_B_score - expected_B_score) < 1e-10

        # Verify doc_C aggregation: rank 3 in dense + rank 2 in sparse
        doc_C = [r for r in fused if r.id == "doc_C"][0]
        expected_C_score = (1.0 / (60 + 3)) + (1.0 / (60 + 2))
        actual_C_score = doc_C.payload.get("rrf_score", 0)
        assert abs(actual_C_score - expected_C_score) < 1e-10

    def test_all_documents_duplicated(self):
        """Test when all documents appear in both dense and sparse."""
        ranker = RRFFusionRanker(k=60)

        dense_results = [
            create_mock_result("doc_1", 0.9),
            create_mock_result("doc_2", 0.8),
        ]
        sparse_results = [
            create_mock_result("doc_1", 0.85),
            create_mock_result("doc_2", 0.75),
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Should only have 2 unique documents
        assert len(fused) == 2
        fused_ids = {r.id for r in fused}
        assert fused_ids == {"doc_1", "doc_2"}


class TestRRFEdgeCases:
    """Test edge cases in RRF fusion."""

    def test_very_long_result_lists(self):
        """Test fusion with very long result lists (1000+ results)."""
        ranker = RRFFusionRanker()

        dense_results = create_dense_results(1000)
        sparse_results = create_sparse_results(1000)

        fused = ranker.fuse(dense_results, sparse_results)

        # Should handle large lists without errors
        assert len(fused) == 2000

        # Verify RRF scores are correctly calculated
        assert all(r.payload.get("rrf_score", 0) > 0 for r in fused)

        # Verify sorting (scores should be in descending order)
        scores = [r.payload.get("rrf_score", 0) for r in fused]
        assert scores == sorted(scores, reverse=True)

    def test_results_with_identical_scores(self):
        """Test fusion when multiple results have identical original scores."""
        ranker = RRFFusionRanker()

        # All results have same score
        dense_results = [
            create_mock_result("doc_1", 0.5),
            create_mock_result("doc_2", 0.5),
            create_mock_result("doc_3", 0.5),
        ]
        sparse_results = [
            create_mock_result("doc_4", 0.5),
            create_mock_result("doc_5", 0.5),
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # RRF should still differentiate by rank
        assert len(fused) == 5
        scores = [r.payload.get("rrf_score", 0) for r in fused]

        # First result should have highest RRF score
        assert scores[0] == max(scores)

    def test_results_with_none_payload(self):
        """Test fusion with results that have None payload."""
        ranker = RRFFusionRanker()

        # Create result with None payload
        result_none = Mock()
        result_none.id = "doc_none"
        result_none.score = 0.9
        result_none.payload = None

        dense_results = [result_none]
        sparse_results = [create_mock_result("doc_sparse", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results)

        # Should handle None payload gracefully
        assert len(fused) == 2

        # Result with None payload should get initialized payload
        none_result = [r for r in fused if r.id == "doc_none"][0]
        assert none_result.payload is not None
        assert "rrf_score" in none_result.payload

    def test_results_without_payload_attribute(self):
        """Test fusion with results missing payload attribute."""
        ranker = RRFFusionRanker()

        # Create result without payload attribute
        result_no_attr = Mock(spec=['id', 'score'])
        result_no_attr.id = "doc_no_payload"
        result_no_attr.score = 0.9

        dense_results = [result_no_attr]
        sparse_results = [create_mock_result("doc_sparse", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results)

        # Should handle missing payload attribute gracefully
        assert len(fused) == 2


class TestRRFQualityVerification:
    """Test that RRF fusion improves search quality over individual methods."""

    def test_fusion_improves_recall_over_dense_only(self):
        """Verify fusion brings in relevant sparse-only results."""
        ranker = RRFFusionRanker()

        # Dense has 3 results, sparse has 3 different results
        dense_results = create_dense_results(3)
        sparse_results = create_sparse_results(3)

        fused = ranker.fuse(dense_results, sparse_results)

        # Fusion should include results from both sources
        # Recall: number of unique docs / total relevant docs
        # Dense-only recall: 3/6 = 0.5
        # Fusion recall: 6/6 = 1.0

        fused_ids = {r.id for r in fused}
        dense_ids = {r.id for r in dense_results}
        sparse_ids = {r.id for r in sparse_results}

        # Fusion should include all documents
        assert len(fused_ids) > len(dense_ids)
        assert all(doc_id in fused_ids for doc_id in sparse_ids)

    def test_fusion_improves_recall_over_sparse_only(self):
        """Verify fusion brings in relevant dense-only results."""
        ranker = RRFFusionRanker()

        dense_results = create_dense_results(3)
        sparse_results = create_sparse_results(3)

        fused = ranker.fuse(dense_results, sparse_results)

        fused_ids = {r.id for r in fused}
        dense_ids = {r.id for r in dense_results}
        sparse_ids = {r.id for r in sparse_results}

        # Fusion should include all documents
        assert len(fused_ids) > len(sparse_ids)
        assert all(doc_id in fused_ids for doc_id in dense_ids)

    def test_fusion_preserves_top_ranked_results(self):
        """Verify fusion preserves top-ranked results from both sources."""
        ranker = RRFFusionRanker()

        # Top dense result has very high score
        dense_results = [
            create_mock_result("top_dense", 0.95),
            create_mock_result("doc_2", 0.6),
        ]
        # Top sparse result has very high score
        sparse_results = [
            create_mock_result("top_sparse", 0.9),
            create_mock_result("doc_4", 0.55),
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Top 2 fused results should include both top-ranked documents
        top_2_ids = [fused[0].id, fused[1].id]
        assert "top_dense" in top_2_ids
        assert "top_sparse" in top_2_ids

    def test_fusion_brings_diverse_results(self):
        """Verify fusion brings in diverse results from both modalities."""
        ranker = RRFFusionRanker()

        # Create results where dense and sparse find different documents
        dense_results = [
            create_mock_result("dense_1", 0.9),
            create_mock_result("dense_2", 0.8),
            create_mock_result("overlap", 0.7),
        ]
        sparse_results = [
            create_mock_result("sparse_1", 0.85),
            create_mock_result("overlap", 0.75),
            create_mock_result("sparse_2", 0.7),
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Should have diverse results: 3 dense-only + 2 sparse-only + 1 overlap = 5 unique
        fused_ids = {r.id for r in fused}
        assert len(fused_ids) == 5

        # Top 5 should include diverse sources
        top_5_ids = [r.id for r in fused[:5]]
        has_dense_only = any("dense_" in doc_id for doc_id in top_5_ids)
        has_sparse_only = any("sparse_" in doc_id for doc_id in top_5_ids)
        has_overlap = "overlap" in top_5_ids

        assert has_dense_only
        assert has_sparse_only
        assert has_overlap


class TestRRFIntegration:
    """Integration tests for RRF with other features."""

    def test_rrf_with_boost_weights(self):
        """Test RRF fusion with boost weights applied."""
        boost_weights = {"doc_type": 1.5}
        ranker = RRFFusionRanker(k=60, boost_weights=boost_weights)

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = [create_mock_result("doc_2", 0.8)]

        fused = ranker.fuse(dense_results, sparse_results)

        # Should apply boost weights to all documents
        # Note: Current implementation applies boost to all docs equally
        assert len(fused) == 2
        assert all(r.payload.get("rrf_score", 0) > 0 for r in fused)

    def test_rrf_explain_fusion_basic(self):
        """Test RRF explain_fusion functionality."""
        ranker = RRFFusionRanker()

        dense_results = create_dense_results(3)
        sparse_results = create_sparse_results(2)

        explanation = ranker.explain_fusion(dense_results, sparse_results, top_k=3)

        # Verify explanation structure
        assert "algorithm" in explanation
        assert explanation["algorithm"] == "Reciprocal Rank Fusion (RRF)"

        assert "parameters" in explanation
        assert explanation["parameters"]["k"] == 60

        assert "input_stats" in explanation
        assert explanation["input_stats"]["dense_results"] == 3
        assert explanation["input_stats"]["sparse_results"] == 2
        assert explanation["input_stats"]["unique_documents"] == 5

        assert "top_results_analysis" in explanation
        assert len(explanation["top_results_analysis"]) <= 3

    def test_rrf_explain_fusion_with_overlap(self):
        """Test explain_fusion with overlapping documents."""
        ranker = RRFFusionRanker()

        dense_results = [
            create_mock_result("overlap_doc", 0.9),
            create_mock_result("dense_doc", 0.8),
        ]
        sparse_results = [
            create_mock_result("sparse_doc", 0.85),
            create_mock_result("overlap_doc", 0.75),
        ]

        explanation = ranker.explain_fusion(dense_results, sparse_results, top_k=3)

        # Find overlap_doc in analysis
        overlap_analysis = [
            a for a in explanation["top_results_analysis"]
            if a["document_id"] == "overlap_doc"
        ]

        assert len(overlap_analysis) == 1
        analysis = overlap_analysis[0]

        # Should have both dense and sparse ranks
        assert analysis["dense_rank"] is not None
        assert analysis["sparse_rank"] is not None
        assert analysis["dense_contribution"] > 0
        assert analysis["sparse_contribution"] > 0

    def test_rrf_score_metadata_attachment(self):
        """Test that RRF scores are correctly attached to result metadata."""
        ranker = RRFFusionRanker()

        dense_results = [create_mock_result("doc_1", 0.9, {"title": "Document 1"})]
        sparse_results = [create_mock_result("doc_2", 0.8, {"title": "Document 2"})]

        fused = ranker.fuse(dense_results, sparse_results)

        # Verify RRF score is added to payload
        for result in fused:
            assert "rrf_score" in result.payload
            assert result.payload["rrf_score"] > 0

            # Original payload data should be preserved
            if result.id == "doc_1":
                assert result.payload.get("title") == "Document 1"
            elif result.id == "doc_2":
                assert result.payload.get("title") == "Document 2"

    def test_rrf_sorting_correctness(self):
        """Test that RRF correctly sorts results by score (descending)."""
        ranker = RRFFusionRanker()

        # Create results with known ranks
        dense_results = [
            create_mock_result("doc_1", 0.9),
            create_mock_result("doc_2", 0.7),
        ]
        sparse_results = [
            create_mock_result("doc_3", 0.8),
            create_mock_result("doc_1", 0.75),  # Overlap with dense
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Verify descending order
        scores = [r.payload.get("rrf_score", 0) for r in fused]
        assert scores == sorted(scores, reverse=True)

        # doc_1 should be first (appears in both, rank 1 in dense, rank 2 in sparse)
        assert fused[0].id == "doc_1"


class TestRRFRegressionPrevention:
    """Regression tests to prevent bugs in future changes."""

    def test_empty_results_no_exception(self):
        """Ensure empty results don't cause exceptions."""
        ranker = RRFFusionRanker()

        # Should not raise any exceptions
        try:
            ranker.fuse([], [])
            ranker.explain_fusion([], [], top_k=5)
        except Exception as e:
            pytest.fail(f"Empty results caused exception: {e}")

    def test_negative_weights_handled(self):
        """Test that negative weights are handled (shouldn't crash)."""
        ranker = RRFFusionRanker()

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = [create_mock_result("doc_2", 0.8)]

        # Negative weights might produce negative scores, but shouldn't crash
        try:
            fused = ranker.fuse(dense_results, sparse_results, weights={"dense": -1.0, "sparse": 1.0})
            assert len(fused) == 2
        except Exception as e:
            pytest.fail(f"Negative weights caused exception: {e}")

    def test_missing_weight_keys_use_defaults(self):
        """Test that missing weight keys use default values."""
        ranker = RRFFusionRanker()

        dense_results = [create_mock_result("doc_1", 0.9)]
        sparse_results = [create_mock_result("doc_2", 0.8)]

        # Missing 'dense' key - should use default
        fused = ranker.fuse(dense_results, sparse_results, weights={"sparse": 1.0})

        # Should still produce results
        assert len(fused) == 2

    def test_result_id_uniqueness_guaranteed(self):
        """Test that fused results contain each document ID only once."""
        ranker = RRFFusionRanker()

        # Same documents in both lists
        dense_results = [
            create_mock_result("doc_1", 0.9),
            create_mock_result("doc_2", 0.8),
        ]
        sparse_results = [
            create_mock_result("doc_1", 0.85),
            create_mock_result("doc_2", 0.75),
        ]

        fused = ranker.fuse(dense_results, sparse_results)

        # Each ID should appear exactly once
        fused_ids = [r.id for r in fused]
        assert len(fused_ids) == len(set(fused_ids))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
