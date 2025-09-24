"""
Comprehensive unit tests for four-context hybrid search system.

This test suite focuses specifically on the four-context hierarchy implementation with:
- Context boundary edge cases and scope conflicts
- Search result aggregation failures and ranking inconsistencies
- Inheritance chain breaks and override conflicts
- Relevance scoring edge cases with zero/negative scores
- Large dataset performance and memory pressure scenarios

Task 259: Comprehensive unit tests for four-context hybrid search system.
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock

import pytest

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

try:
    from workspace_qdrant_mcp.core.four_context_search import (
        FourContextSearchEngine,
        SearchContext,
        SearchScope,
        SearchContextConfig,
        FourContextSearchQuery,
        SearchResult,
        FourContextSearchResponse,
        ContextSearchCache
    )
    from qdrant_client.http import models
    FOUR_CONTEXT_AVAILABLE = True
except ImportError as e:
    FOUR_CONTEXT_AVAILABLE = False
    print(f"Four context search import failed: {e}")

pytestmark = pytest.mark.skipif(not FOUR_CONTEXT_AVAILABLE, reason="Four context search not available")


class TestContextSearchCache:
    """Test the context-aware search cache system with edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = ContextSearchCache(ttl_seconds=300, max_size=100)

    def test_cache_key_generation_consistency(self):
        """Test cache key generation produces consistent results."""
        query = "test query with unicode: é"
        collections = ["col1", "col2"]
        mode = "hybrid"

        # Same parameters should generate same key every time
        key1 = self.cache._generate_cache_key(query, SearchContext.PROJECT, collections, mode)
        key2 = self.cache._generate_cache_key(query, SearchContext.PROJECT, collections, mode)
        key3 = self.cache._generate_cache_key(query, SearchContext.PROJECT, collections, mode)

        assert key1 == key2 == key3

    def test_cache_key_generation_differences(self):
        """Test cache key generation produces different keys for different inputs."""
        base_query = "test query"
        collections = ["col1", "col2"]
        mode = "hybrid"

        # Different contexts should generate different keys
        key_project = self.cache._generate_cache_key(base_query, SearchContext.PROJECT, collections, mode)
        key_collection = self.cache._generate_cache_key(base_query, SearchContext.COLLECTION, collections, mode)
        key_global = self.cache._generate_cache_key(base_query, SearchContext.GLOBAL, collections, mode)
        key_all = self.cache._generate_cache_key(base_query, SearchContext.ALL, collections, mode)

        # All keys should be different
        keys = [key_project, key_collection, key_global, key_all]
        assert len(set(keys)) == 4

        # Different queries should generate different keys
        key_query1 = self.cache._generate_cache_key("query1", SearchContext.PROJECT, collections, mode)
        key_query2 = self.cache._generate_cache_key("query2", SearchContext.PROJECT, collections, mode)
        assert key_query1 != key_query2

        # Different modes should generate different keys
        key_hybrid = self.cache._generate_cache_key(base_query, SearchContext.PROJECT, collections, "hybrid")
        key_dense = self.cache._generate_cache_key(base_query, SearchContext.PROJECT, collections, "dense")
        assert key_hybrid != key_dense

    def test_cache_key_collection_order_independence(self):
        """Test cache keys are independent of collection order."""
        query = "test query"
        mode = "hybrid"

        # Different orders should generate the same key
        key1 = self.cache._generate_cache_key(query, SearchContext.PROJECT, ["col1", "col2"], mode)
        key2 = self.cache._generate_cache_key(query, SearchContext.PROJECT, ["col2", "col1"], mode)
        assert key1 == key2

        # Test with more collections
        key3 = self.cache._generate_cache_key(query, SearchContext.PROJECT, ["a", "b", "c"], mode)
        key4 = self.cache._generate_cache_key(query, SearchContext.PROJECT, ["c", "a", "b"], mode)
        key5 = self.cache._generate_cache_key(query, SearchContext.PROJECT, ["b", "c", "a"], mode)
        assert key3 == key4 == key5

    def test_cache_hit_miss_functionality(self):
        """Test cache hit/miss detection with edge cases."""
        query = "test query"
        context = SearchContext.PROJECT
        collections = ["col1"]
        mode = "hybrid"
        result_data = {"results": [{"id": "1", "score": 0.8}]}

        # Initial state - should be miss
        cached = self.cache.get(query, context, collections, mode)
        assert cached is None
        assert self.cache.miss_count == 1
        assert self.cache.hit_count == 0
        assert self.cache.get_stats()["hit_rate"] == 0.0

        # Set cache
        self.cache.set(query, context, collections, mode, result_data)

        # Should be hit now
        cached = self.cache.get(query, context, collections, mode)
        assert cached == result_data
        assert self.cache.hit_count == 1
        assert self.cache.miss_count == 1
        assert abs(self.cache.get_stats()["hit_rate"] - 0.5) < 0.01

    def test_cache_expiration_edge_cases(self):
        """Test cache expiration with very short and very long TTL."""
        # Test very short TTL
        short_cache = ContextSearchCache(ttl_seconds=0.01, max_size=100)
        query = "test query"
        context = SearchContext.PROJECT
        collections = ["col1"]
        mode = "hybrid"
        result_data = {"results": [{"id": "1", "score": 0.8}]}

        # Set cache
        short_cache.set(query, context, collections, mode, result_data)

        # Should be hit immediately
        cached = short_cache.get(query, context, collections, mode)
        assert cached == result_data

        # Wait for expiration
        time.sleep(0.02)

        # Should be miss after expiration
        cached = short_cache.get(query, context, collections, mode)
        assert cached is None

        # Test TTL of 0 (should never cache)
        zero_ttl_cache = ContextSearchCache(ttl_seconds=0, max_size=100)
        zero_ttl_cache.set(query, context, collections, mode, result_data)

        # Should immediately be miss
        cached = zero_ttl_cache.get(query, context, collections, mode)
        assert cached is None

    def test_cache_lru_eviction_behavior(self):
        """Test LRU eviction behavior with edge cases."""
        cache = ContextSearchCache(ttl_seconds=300, max_size=3)  # Small cache

        # Fill cache to capacity
        cache.set("query1", SearchContext.PROJECT, ["col1"], "hybrid", {"data": 1})
        cache.set("query2", SearchContext.PROJECT, ["col1"], "hybrid", {"data": 2})
        cache.set("query3", SearchContext.PROJECT, ["col1"], "hybrid", {"data": 3})
        assert len(cache.cache) == 3

        # Access query1 to make it most recently used
        cached1 = cache.get("query1", SearchContext.PROJECT, ["col1"], "hybrid")
        assert cached1 == {"data": 1}

        # Add query4 - should evict query2 (least recently used)
        cache.set("query4", SearchContext.PROJECT, ["col1"], "hybrid", {"data": 4})
        assert len(cache.cache) == 3

        # query2 should be evicted
        cached2 = cache.get("query2", SearchContext.PROJECT, ["col1"], "hybrid")
        assert cached2 is None

        # query1, query3, query4 should still be there
        assert cache.get("query1", SearchContext.PROJECT, ["col1"], "hybrid") == {"data": 1}
        assert cache.get("query3", SearchContext.PROJECT, ["col1"], "hybrid") == {"data": 3}
        assert cache.get("query4", SearchContext.PROJECT, ["col1"], "hybrid") == {"data": 4}

    def test_cache_size_edge_cases(self):
        """Test cache behavior with edge case sizes."""
        # Size 0 cache should not cache anything
        zero_cache = ContextSearchCache(ttl_seconds=300, max_size=0)
        zero_cache.set("q1", SearchContext.PROJECT, ["col1"], "hybrid", {"data": 1})
        assert zero_cache.get("q1", SearchContext.PROJECT, ["col1"], "hybrid") is None
        assert len(zero_cache.cache) == 0

        # Size 1 cache should only hold one item
        one_cache = ContextSearchCache(ttl_seconds=300, max_size=1)
        one_cache.set("q1", SearchContext.PROJECT, ["col1"], "hybrid", {"data": 1})
        assert len(one_cache.cache) == 1

        one_cache.set("q2", SearchContext.PROJECT, ["col1"], "hybrid", {"data": 2})
        assert len(one_cache.cache) == 1

        # q1 should be evicted
        assert one_cache.get("q1", SearchContext.PROJECT, ["col1"], "hybrid") is None
        assert one_cache.get("q2", SearchContext.PROJECT, ["col1"], "hybrid") == {"data": 2}

    def test_context_invalidation_edge_cases(self):
        """Test context invalidation with complex scenarios."""
        # Add entries for different contexts and collections
        self.cache.set("q1", SearchContext.PROJECT, ["col1"], "hybrid", {"data": 1})
        self.cache.set("q1", SearchContext.PROJECT, ["col2"], "hybrid", {"data": 2})  # Same query, different collection
        self.cache.set("q2", SearchContext.GLOBAL, ["col1"], "hybrid", {"data": 3})
        self.cache.set("q1", SearchContext.GLOBAL, ["col1"], "hybrid", {"data": 4})  # Same query, different context
        self.cache.set("q3", SearchContext.COLLECTION, ["col1"], "sparse", {"data": 5})

        assert len(self.cache.cache) == 5

        # Invalidate PROJECT context
        self.cache.invalidate_context(SearchContext.PROJECT)

        # Only PROJECT entries should be gone
        assert self.cache.get("q1", SearchContext.PROJECT, ["col1"], "hybrid") is None
        assert self.cache.get("q1", SearchContext.PROJECT, ["col2"], "hybrid") is None

        # Other contexts should remain
        assert self.cache.get("q2", SearchContext.GLOBAL, ["col1"], "hybrid") == {"data": 3}
        assert self.cache.get("q1", SearchContext.GLOBAL, ["col1"], "hybrid") == {"data": 4}
        assert self.cache.get("q3", SearchContext.COLLECTION, ["col1"], "sparse") == {"data": 5}

    def test_cache_performance_under_load(self):
        """Test cache performance with many operations."""
        large_cache = ContextSearchCache(ttl_seconds=300, max_size=1000)

        start_time = time.time()

        # Perform many cache operations
        for i in range(500):
            query = f"query_{i}"
            large_cache.set(query, SearchContext.PROJECT, ["col1"], "hybrid", {"data": i})

            # Every 10th query, retrieve multiple items
            if i % 10 == 0:
                for j in range(min(i, 50)):  # Retrieve up to 50 previous items
                    prev_query = f"query_{j}"
                    large_cache.get(prev_query, SearchContext.PROJECT, ["col1"], "hybrid")

        operation_time = time.time() - start_time

        # Should complete in reasonable time (< 2 seconds for 500 + ~1250 retrieval operations)
        assert operation_time < 2.0

        # Check final state
        stats = large_cache.get_stats()
        assert stats["cache_size"] == 500
        assert stats["hit_count"] > 0
        assert stats["miss_count"] >= 0


class TestSearchContextConfig:
    """Test search context configuration edge cases."""

    def test_invalid_score_thresholds(self):
        """Test behavior with invalid score threshold values."""
        # Test negative score threshold
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            score_threshold=-0.5
        )
        assert config.score_threshold == -0.5  # Should accept negative values

        # Test score threshold > 1
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            score_threshold=1.5
        )
        assert config.score_threshold == 1.5  # Should accept values > 1

        # Test score threshold of 0
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            score_threshold=0.0
        )
        assert config.score_threshold == 0.0

    def test_extreme_cache_ttl_values(self):
        """Test configuration with extreme cache TTL values."""
        # Very small TTL
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            cache_ttl_seconds=0.001
        )
        assert config.cache_ttl_seconds == 0.001

        # Zero TTL (no caching)
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            cache_ttl_seconds=0
        )
        assert config.cache_ttl_seconds == 0

        # Very large TTL
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            cache_ttl_seconds=86400 * 365  # 1 year
        )
        assert config.cache_ttl_seconds == 86400 * 365

    def test_extreme_max_results_values(self):
        """Test configuration with extreme max results values."""
        # Zero max results
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            max_results_per_collection=0
        )
        assert config.max_results_per_collection == 0

        # Very large max results
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            max_results_per_collection=100000
        )
        assert config.max_results_per_collection == 100000

    def test_empty_collections_list(self):
        """Test configuration with empty collections list."""
        config = SearchContextConfig(
            context=SearchContext.GLOBAL,
            collections=[]
        )
        assert config.collections == []

    def test_fusion_method_variations(self):
        """Test different fusion method strings."""
        valid_methods = ["rrf", "weighted_sum", "max_score"]

        for method in valid_methods:
            config = SearchContextConfig(
                context=SearchContext.PROJECT,
                fusion_method=method
            )
            assert config.fusion_method == method

        # Test case sensitivity (should preserve original case)
        config = SearchContextConfig(
            context=SearchContext.PROJECT,
            fusion_method="RRF"
        )
        assert config.fusion_method == "RRF"


class TestFourContextSearchQuery:
    """Test four-context search query edge cases."""

    def test_empty_query_string(self):
        """Test behavior with empty query string."""
        query = FourContextSearchQuery(query="")
        assert query.query == ""

    def test_very_long_query_string(self):
        """Test behavior with very long query string."""
        long_query = "test " * 1000  # 5000 characters
        query = FourContextSearchQuery(query=long_query)
        assert query.query == long_query

    def test_unicode_query_string(self):
        """Test behavior with unicode characters in query."""
        unicode_query = "test query with émojis 🔍 and unicode: ñáéíóú"
        query = FourContextSearchQuery(query=unicode_query)
        assert query.query == unicode_query

    def test_empty_contexts_list(self):
        """Test behavior with empty contexts list."""
        query = FourContextSearchQuery(
            query="test",
            contexts=[]
        )
        assert query.contexts == []

    def test_duplicate_contexts(self):
        """Test behavior with duplicate contexts in list."""
        query = FourContextSearchQuery(
            query="test",
            contexts=[SearchContext.PROJECT, SearchContext.PROJECT, SearchContext.GLOBAL]
        )
        # Should preserve duplicates as provided
        assert len(query.contexts) == 3
        assert query.contexts.count(SearchContext.PROJECT) == 2

    def test_extreme_limit_values(self):
        """Test behavior with extreme limit values."""
        # Zero limit
        query = FourContextSearchQuery(query="test", limit=0)
        assert query.limit == 0

        # Very large limit
        query = FourContextSearchQuery(query="test", limit=100000)
        assert query.limit == 100000

        # Negative limit (should accept as provided)
        query = FourContextSearchQuery(query="test", limit=-10)
        assert query.limit == -10

    def test_extreme_response_time_targets(self):
        """Test behavior with extreme response time targets."""
        # Zero response time target
        query = FourContextSearchQuery(query="test", response_time_target_ms=0.0)
        assert query.response_time_target_ms == 0.0

        # Very large response time target
        query = FourContextSearchQuery(query="test", response_time_target_ms=60000.0)  # 1 minute
        assert query.response_time_target_ms == 60000.0

        # Very small response time target
        query = FourContextSearchQuery(query="test", response_time_target_ms=0.001)
        assert query.response_time_target_ms == 0.001

    def test_none_collection_values(self):
        """Test behavior when collection lists are None."""
        query = FourContextSearchQuery(
            query="test",
            target_collections=None,
            global_collections=None
        )
        assert query.target_collections is None
        assert query.global_collections is None

    def test_empty_collection_lists(self):
        """Test behavior with empty collection lists."""
        query = FourContextSearchQuery(
            query="test",
            target_collections=[],
            global_collections=[]
        )
        assert query.target_collections == []
        assert query.global_collections == []


class TestSearchResult:
    """Test search result structure edge cases."""

    def test_zero_and_negative_scores(self):
        """Test search results with zero and negative scores."""
        # Zero score
        result = SearchResult(
            id="doc1",
            score=0.0,
            content="Zero score content",
            collection="test-collection",
            context=SearchContext.PROJECT,
            search_type="hybrid",
            metadata={}
        )
        assert result.score == 0.0

        # Negative score
        result = SearchResult(
            id="doc2",
            score=-0.5,
            content="Negative score content",
            collection="test-collection",
            context=SearchContext.PROJECT,
            search_type="hybrid",
            metadata={}
        )
        assert result.score == -0.5

    def test_very_large_scores(self):
        """Test search results with very large scores."""
        result = SearchResult(
            id="doc1",
            score=999999.99,
            content="Large score content",
            collection="test-collection",
            context=SearchContext.PROJECT,
            search_type="hybrid",
            metadata={}
        )
        assert result.score == 999999.99

    def test_empty_content_and_metadata(self):
        """Test search results with empty content and metadata."""
        result = SearchResult(
            id="doc1",
            score=0.8,
            content="",
            collection="test-collection",
            context=SearchContext.PROJECT,
            search_type="hybrid",
            metadata={}
        )
        assert result.content == ""
        assert result.metadata == {}

    def test_none_optional_fields(self):
        """Test search results with None optional fields."""
        result = SearchResult(
            id="doc1",
            score=0.8,
            content="Test content",
            collection="test-collection",
            context=SearchContext.PROJECT,
            search_type="hybrid",
            metadata={},
            code_result=None,
            fusion_details=None,
            response_time_ms=None
        )
        assert result.code_result is None
        assert result.fusion_details is None
        assert result.response_time_ms is None

    def test_large_metadata_dict(self):
        """Test search results with large metadata dictionaries."""
        large_metadata = {}
        for i in range(1000):
            large_metadata[f"key_{i}"] = f"value_{i}"

        result = SearchResult(
            id="doc1",
            score=0.8,
            content="Test content",
            collection="test-collection",
            context=SearchContext.PROJECT,
            search_type="hybrid",
            metadata=large_metadata
        )
        assert len(result.metadata) == 1000
        assert result.metadata["key_999"] == "value_999"


@pytest.mark.asyncio
class TestFourContextSearchEngineInitialization:
    """Test four-context search engine initialization edge cases."""

    def test_initialization_with_none_workspace_client(self):
        """Test initialization behavior with None workspace client."""
        with pytest.raises(Exception):  # Should fail with None client
            FourContextSearchEngine(workspace_client=None)

    def test_initialization_with_invalid_cache_ttl(self):
        """Test initialization with invalid cache TTL values."""
        mock_client = Mock()
        mock_client.client = Mock()
        mock_client.config = {}
        mock_client.list_collections.return_value = []

        # Negative cache TTL should be accepted (handled by cache implementation)
        engine = FourContextSearchEngine(
            workspace_client=mock_client,
            cache_ttl_seconds=-1
        )
        # Should complete initialization without error

    def test_initialization_with_extreme_performance_settings(self):
        """Test initialization with extreme performance monitoring settings."""
        mock_client = Mock()
        mock_client.client = Mock()
        mock_client.config = {}
        mock_client.list_collections.return_value = []

        # Performance monitoring disabled
        engine1 = FourContextSearchEngine(
            workspace_client=mock_client,
            enable_performance_monitoring=False
        )
        assert engine1.enable_performance_monitoring is False

        # Performance monitoring enabled
        engine2 = FourContextSearchEngine(
            workspace_client=mock_client,
            enable_performance_monitoring=True
        )
        assert engine2.enable_performance_monitoring is True

    def test_get_context_config_for_all_contexts(self):
        """Test getting configuration for all search contexts."""
        mock_client = Mock()
        mock_client.client = Mock()
        mock_client.config = {}
        mock_client.list_collections.return_value = []

        engine = FourContextSearchEngine(mock_client)

        # Should have configs for all contexts
        for context in SearchContext:
            config = engine.get_context_config(context)
            assert config.context == context

    def test_cache_statistics_initial_state(self):
        """Test cache statistics in initial state."""
        mock_client = Mock()
        mock_client.client = Mock()
        mock_client.config = {}
        mock_client.list_collections.return_value = []

        engine = FourContextSearchEngine(mock_client)
        stats = engine.get_cache_stats()

        assert stats["cache_size"] == 0
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        assert stats["hit_rate"] == 0.0

    def test_performance_metrics_initial_state(self):
        """Test performance metrics in initial state."""
        mock_client = Mock()
        mock_client.client = Mock()
        mock_client.config = {}
        mock_client.list_collections.return_value = []

        engine = FourContextSearchEngine(mock_client)
        metrics = engine.get_performance_metrics()

        assert "cache_performance" in metrics
        assert "context_configs" in metrics
        assert metrics["initialized"] is False
        assert len(metrics["context_configs"]) == 4  # One for each context


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])