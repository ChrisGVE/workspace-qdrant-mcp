"""
Unit tests for hybrid search functionality.

Tests RRF fusion, hybrid search engine, and score combination methods.
"""

from unittest.mock import MagicMock, AsyncMock

import pytest
from qdrant_client.http import models

from workspace_qdrant_mcp.core.hybrid_search import RRFFusionRanker, HybridSearchEngine
from workspace_qdrant_mcp.core.sparse_vectors import create_named_sparse_vector


class TestRRFFusionRanker:
    """Test RRFFusionRanker class."""
    
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
    
    def test_fuse_rankings_only_dense_results(self):
        """Test fusion with only dense results."""
        ranker = RRFFusionRanker()
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "doc1"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "doc2"}}
        ]
        
        fused = ranker.fuse_rankings(dense_results, [])
        
        assert len(fused) == 2
        assert fused[0]["id"] == "doc1"  # Higher RRF score (1/61)
        assert fused[1]["id"] == "doc2"  # Lower RRF score (1/62)
        assert fused[0]["search_types"] == ["dense"]
        assert fused[0]["sparse_rank"] is None
    
    def test_fuse_rankings_only_sparse_results(self):
        """Test fusion with only sparse results."""
        ranker = RRFFusionRanker()
        sparse_results = [
            {"id": "doc1", "score": 0.95, "payload": {"content": "doc1"}},
            {"id": "doc2", "score": 0.85, "payload": {"content": "doc2"}}
        ]
        
        fused = ranker.fuse_rankings([], sparse_results)
        
        assert len(fused) == 2
        assert fused[0]["id"] == "doc1"
        assert fused[1]["id"] == "doc2"
        assert fused[0]["search_types"] == ["sparse"]
        assert fused[0]["dense_rank"] is None
    
    def test_fuse_rankings_both_results_no_overlap(self):
        """Test fusion with both result types, no document overlap."""
        ranker = RRFFusionRanker()
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "doc1"}}
        ]
        sparse_results = [
            {"id": "doc2", "score": 0.8, "payload": {"content": "doc2"}}
        ]
        
        fused = ranker.fuse_rankings(dense_results, sparse_results)
        
        assert len(fused) == 2
        # Both should have same RRF score (1/61) so order depends on processing
        doc_ids = {result["id"] for result in fused}
        assert doc_ids == {"doc1", "doc2"}
        
        # Check that each has only one search type
        for result in fused:
            assert len(result["search_types"]) == 1
    
    def test_fuse_rankings_both_results_with_overlap(self):
        """Test fusion with both result types and document overlap."""
        ranker = RRFFusionRanker()
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "doc1"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "doc2"}}
        ]
        sparse_results = [
            {"id": "doc1", "score": 0.85, "payload": {"content": "doc1"}},  # Overlap
            {"id": "doc3", "score": 0.75, "payload": {"content": "doc3"}}
        ]
        
        fused = ranker.fuse_rankings(dense_results, sparse_results)
        
        assert len(fused) == 3
        
        # doc1 should be first (appears in both, highest combined RRF score)
        assert fused[0]["id"] == "doc1"
        assert len(fused[0]["search_types"]) == 2
        assert "dense" in fused[0]["search_types"]
        assert "sparse" in fused[0]["search_types"]
        assert fused[0]["dense_score"] == 0.9
        assert fused[0]["sparse_score"] == 0.85
        assert fused[0]["dense_rank"] == 1
        assert fused[0]["sparse_rank"] == 1
    
    def test_fuse_rankings_with_weights(self):
        """Test fusion with custom weights."""
        ranker = RRFFusionRanker()
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "doc1"}}
        ]
        sparse_results = [
            {"id": "doc1", "score": 0.8, "payload": {"content": "doc1"}}
        ]
        
        # Test with higher sparse weight
        fused = ranker.fuse_rankings(dense_results, sparse_results, dense_weight=0.5, sparse_weight=2.0)
        
        assert len(fused) == 1
        doc = fused[0]
        
        expected_rrf = (0.5 / 61) + (2.0 / 61)  # weighted RRF scores
        assert abs(doc["rrf_score"] - expected_rrf) < 1e-10
    
    def test_explain_fusion(self):
        """Test fusion explanation functionality."""
        ranker = RRFFusionRanker(k=50)
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "doc1"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "doc2"}}
        ]
        sparse_results = [
            {"id": "doc1", "score": 0.85, "payload": {"content": "doc1"}},
            {"id": "doc3", "score": 0.75, "payload": {"content": "doc3"}}
        ]
        
        explanation = ranker.explain_fusion(dense_results, sparse_results, dense_weight=1.5, sparse_weight=0.8)
        
        # Check explanation structure
        assert "fusion_method" in explanation
        assert explanation["fusion_method"] == "Reciprocal Rank Fusion (RRF)"
        assert explanation["k_parameter"] == 50
        
        assert "weights" in explanation
        assert explanation["weights"]["dense"] == 1.5
        assert explanation["weights"]["sparse"] == 0.8
        
        assert "input_stats" in explanation
        assert explanation["input_stats"]["dense_results"] == 2
        assert explanation["input_stats"]["sparse_results"] == 2
        
        assert "fusion_stats" in explanation
        fusion_stats = explanation["fusion_stats"]
        assert fusion_stats["total_fused_results"] == 3
        assert fusion_stats["dense_only"] == 1  # doc2
        assert fusion_stats["sparse_only"] == 1  # doc3
        assert fusion_stats["found_in_both"] == 1  # doc1
        
        assert "fused_results" in explanation
        assert len(explanation["fused_results"]) == 3


class TestHybridSearchEngine:
    """Test HybridSearchEngine class."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client for testing."""
        client = MagicMock()
        return client
    
    @pytest.fixture
    def hybrid_engine(self, mock_qdrant_client):
        """Create hybrid search engine with mock client."""
        return HybridSearchEngine(mock_qdrant_client)
    
    def test_init(self, mock_qdrant_client):
        """Test hybrid search engine initialization."""
        engine = HybridSearchEngine(mock_qdrant_client)
        
        assert engine.client == mock_qdrant_client
        assert isinstance(engine.rrf_ranker, RRFFusionRanker)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_dense_only(self, hybrid_engine):
        """Test hybrid search with only dense embeddings."""
        # Setup mock client responses
        mock_dense_results = [
            models.ScoredPoint(
                id="doc1",
                score=0.9,
                payload={"content": "doc1"}
            ),
            models.ScoredPoint(
                id="doc2",
                score=0.8,
                payload={"content": "doc2"}
            )
        ]
        
        hybrid_engine.client.search.return_value = mock_dense_results
        
        query_embeddings = {
            "dense": [0.1] * 384
        }
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            limit=10
        )
        
        assert "error" not in result
        assert result["collection"] == "test_collection"
        assert result["fusion_method"] == "rrf"
        assert result["dense_results_count"] == 2
        assert result["sparse_results_count"] == 0
        assert len(result["results"]) == 2
        
        # Verify client was called correctly
        hybrid_engine.client.search.assert_called_once()
        call_args = hybrid_engine.client.search.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert call_args[1]["query_vector"] == ("dense", [0.1] * 384)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_sparse_only(self, hybrid_engine):
        """Test hybrid search with only sparse embeddings."""
        # Setup mock client responses
        mock_sparse_results = [
            models.ScoredPoint(
                id="doc1",
                score=0.85,
                payload={"content": "doc1"}
            )
        ]
        
        hybrid_engine.client.search.return_value = mock_sparse_results
        
        query_embeddings = {
            "sparse": {
                "indices": [1, 5, 10],
                "values": [0.8, 0.6, 0.9]
            }
        }
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            limit=10
        )
        
        assert "error" not in result
        assert result["sparse_results_count"] == 1
        assert result["dense_results_count"] == 0
        
        # Verify sparse vector was created correctly
        call_args = hybrid_engine.client.search.call_args
        query_vector = call_args[1]["query_vector"]
        assert hasattr(query_vector, "name")
        assert query_vector.name == "sparse"
    
    @pytest.mark.asyncio
    async def test_hybrid_search_both_embeddings(self, hybrid_engine):
        """Test hybrid search with both dense and sparse embeddings."""
        # Setup mock client responses - different results for each call
        dense_results = [
            models.ScoredPoint(
                id="doc1",
                score=0.9,
                payload={"content": "doc1"}
            )
        ]
        sparse_results = [
            models.ScoredPoint(
                id="doc2",
                score=0.85,
                payload={"content": "doc2"}
            )
        ]
        
        # Mock client to return different results for each call
        hybrid_engine.client.search.side_effect = [dense_results, sparse_results]
        
        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {
                "indices": [1, 5, 10],
                "values": [0.8, 0.6, 0.9]
            }
        }
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            limit=10
        )
        
        assert "error" not in result
        assert result["dense_results_count"] == 1
        assert result["sparse_results_count"] == 1
        assert len(result["results"]) == 2
        
        # Verify both search calls were made
        assert hybrid_engine.client.search.call_count == 2
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_filter(self, hybrid_engine):
        """Test hybrid search with query filter."""
        hybrid_engine.client.search.return_value = []
        
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(value="python")
                )
            ]
        )
        
        query_embeddings = {"dense": [0.1] * 384}
        
        await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            query_filter=query_filter
        )
        
        # Verify filter was passed to client
        call_args = hybrid_engine.client.search.call_args
        assert call_args[1]["query_filter"] == query_filter
    
    @pytest.mark.asyncio
    async def test_hybrid_search_with_weights(self, hybrid_engine):
        """Test hybrid search with custom weights."""
        hybrid_engine.client.search.side_effect = [[], []]  # Empty results for simplicity
        
        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1], "values": [0.8]}
        }
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            dense_weight=1.5,
            sparse_weight=0.8
        )
        
        assert result["weights"]["dense"] == 1.5
        assert result["weights"]["sparse"] == 0.8
    
    @pytest.mark.asyncio
    async def test_hybrid_search_weighted_sum_fusion(self, hybrid_engine):
        """Test hybrid search with weighted sum fusion method."""
        # Setup mock results with different scores
        dense_results = [
            models.ScoredPoint(id="doc1", score=0.9, payload={"content": "doc1"})
        ]
        sparse_results = [
            models.ScoredPoint(id="doc1", score=0.8, payload={"content": "doc1"})
        ]
        
        hybrid_engine.client.search.side_effect = [dense_results, sparse_results]
        
        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1], "values": [0.8]}
        }
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            fusion_method="weighted_sum"
        )
        
        assert result["fusion_method"] == "weighted_sum"
        assert len(result["results"]) == 1
        
        # Should combine scores from both searches
        doc = result["results"][0]
        assert doc["id"] == "doc1"
        assert doc["dense_score"] == 0.9
        assert doc["sparse_score"] == 0.8
        assert doc["score"] == 2.0  # Normalized and weighted (1.0 + 1.0)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_max_fusion(self, hybrid_engine):
        """Test hybrid search with max fusion method."""
        dense_results = [
            models.ScoredPoint(id="doc1", score=0.9, payload={"content": "doc1"})
        ]
        sparse_results = [
            models.ScoredPoint(id="doc1", score=0.8, payload={"content": "doc1"})
        ]
        
        hybrid_engine.client.search.side_effect = [dense_results, sparse_results]
        
        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1], "values": [0.8]}
        }
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            fusion_method="max"
        )
        
        assert result["fusion_method"] == "max"
        doc = result["results"][0]
        assert doc["score"] == 0.9  # Max of 0.9 and 0.8
    
    @pytest.mark.asyncio
    async def test_hybrid_search_unknown_fusion_method(self, hybrid_engine):
        """Test hybrid search with unknown fusion method."""
        query_embeddings = {"dense": [0.1] * 384}
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            fusion_method="unknown_method"
        )
        
        assert "error" in result
        assert "Unknown fusion method" in result["error"]
    
    @pytest.mark.asyncio
    async def test_hybrid_search_client_error(self, hybrid_engine):
        """Test hybrid search when client raises exception."""
        hybrid_engine.client.search.side_effect = Exception("Qdrant error")
        
        query_embeddings = {"dense": [0.1] * 384}
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings
        )
        
        assert "error" in result
        assert "Hybrid search failed" in result["error"]
    
    def test_weighted_sum_fusion_normalization(self, hybrid_engine):
        """Test weighted sum fusion with score normalization."""
        dense_results = [
            {"id": "doc1", "score": 0.8, "payload": {"content": "doc1"}},
            {"id": "doc2", "score": 0.4, "payload": {"content": "doc2"}}
        ]
        sparse_results = [
            {"id": "doc1", "score": 0.6, "payload": {"content": "doc1"}}
        ]
        
        fused = hybrid_engine._weighted_sum_fusion(
            dense_results, sparse_results, dense_weight=1.0, sparse_weight=1.0
        )
        
        # doc1 should be first (appears in both)
        assert fused[0]["id"] == "doc1"
        # Score should be normalized dense (0.8/0.8=1.0) + normalized sparse (0.6/0.6=1.0) = 2.0
        assert fused[0]["score"] == 2.0
        
        # doc2 should be second (only in dense)
        assert fused[1]["id"] == "doc2"
        # Score should be normalized dense (0.4/0.8=0.5)
        assert fused[1]["score"] == 0.5
    
    def test_max_fusion_logic(self, hybrid_engine):
        """Test max fusion logic with overlapping documents."""
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {"content": "doc1"}}
        ]
        sparse_results = [
            {"id": "doc1", "score": 0.7, "payload": {"content": "doc1"}},
            {"id": "doc2", "score": 0.8, "payload": {"content": "doc2"}}
        ]
        
        fused = hybrid_engine._max_fusion(dense_results, sparse_results)
        
        # Should be sorted by score
        assert fused[0]["id"] == "doc1"
        assert fused[0]["score"] == 0.9  # Max of 0.9 and 0.7
        
        assert fused[1]["id"] == "doc2"
        assert fused[1]["score"] == 0.8  # Only sparse score
    
    @pytest.mark.asyncio
    async def test_benchmark_fusion_methods(self, hybrid_engine):
        """Test benchmarking different fusion methods."""
        # Mock search results
        hybrid_engine.client.search.return_value = [
            models.ScoredPoint(id="doc1", score=0.9, payload={"content": "doc1"})
        ]
        
        query_embeddings = {"dense": [0.1] * 384}
        
        benchmark_result = hybrid_engine.benchmark_fusion_methods(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            limit=5
        )
        
        assert "benchmark_results" in benchmark_result
        assert "query_info" in benchmark_result
        
        # Should test all three methods
        methods = benchmark_result["benchmark_results"].keys()
        assert "rrf" in methods
        assert "weighted_sum" in methods
        assert "max" in methods
        
        # Query info should be correct
        query_info = benchmark_result["query_info"]
        assert query_info["has_dense"] is True
        assert query_info["has_sparse"] is False
        assert query_info["limit"] == 5
    
    @pytest.mark.asyncio
    async def test_hybrid_search_limit_application(self, hybrid_engine):
        """Test that final limit is properly applied to results."""
        # Create more results than the limit
        many_results = [
            models.ScoredPoint(
                id=f"doc{i}",
                score=0.9 - (i * 0.1),
                payload={"content": f"doc{i}"}
            )
            for i in range(20)  # 20 results
        ]
        
        hybrid_engine.client.search.return_value = many_results
        
        query_embeddings = {"dense": [0.1] * 384}
        
        result = await hybrid_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            limit=5  # Limit to 5
        )
        
        # Should only return 5 results despite having 20 available
        assert len(result["results"]) == 5
        assert result["dense_results_count"] == 20  # But report full count found
    
    def test_rrf_score_calculation(self):
        """Test RRF score calculation accuracy."""
        ranker = RRFFusionRanker(k=60)
        
        dense_results = [
            {"id": "doc1", "score": 0.9, "payload": {}}
        ]
        sparse_results = [
            {"id": "doc1", "score": 0.8, "payload": {}}
        ]
        
        fused = ranker.fuse_rankings(
            dense_results, sparse_results,
            dense_weight=1.0, sparse_weight=1.0
        )
        
        expected_rrf = (1.0 / (60 + 1)) + (1.0 / (60 + 1))  # Both rank 1
        assert abs(fused[0]["rrf_score"] - expected_rrf) < 1e-10
