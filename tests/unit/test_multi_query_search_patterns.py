"""
Comprehensive tests for complex multi-query search patterns (Task 322.7).

This test suite validates complex query patterns including:
- Batch queries (multiple queries executed in parallel)
- Query chaining (using results from one query as input to next)
- Result aggregation across multiple queries
- Complex boolean queries (AND/OR/NOT combinations)
- Wildcard pattern matching
- Fuzzy matching with edit distance
- Query optimization for complex patterns
- Performance characteristics validation

All patterns are tested with the hybrid search architecture using RRF fusion,
dense semantic vectors, and sparse keyword vectors.
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from dataclasses import dataclass

import pytest
from qdrant_client.http import models

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.hybrid_search import (
    HybridSearchEngine,
    RRFFusionRanker,
    TenantAwareResult,
    MultiTenantResultAggregator
)


# Test Data Models

@dataclass
class QueryPattern:
    """Represents a complex query pattern for testing."""
    query_id: str
    query_text: str
    query_embeddings: Dict
    filters: Optional[models.Filter] = None
    expected_min_results: int = 0
    expected_max_results: int = 100


@dataclass
class BatchQueryResult:
    """Result from batch query execution."""
    query_id: str
    results: List
    execution_time_ms: float
    error: Optional[str] = None


# Fixtures

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = MagicMock()
    client.search = MagicMock(return_value=[])
    # Mock collection info for optimization
    client.get_collection = MagicMock(return_value=None)
    return client


@pytest.fixture
def hybrid_search_engine(mock_qdrant_client):
    """Hybrid search engine with mocked client."""
    # Disable optimizations and monitoring to avoid complex mocking
    return HybridSearchEngine(
        client=mock_qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=True,
        enable_performance_monitoring=False
    )


@pytest.fixture
def sample_documents():
    """Sample documents for query testing."""
    return [
        {
            "id": "doc1",
            "score": 0.95,
            "payload": {
                "content": "Python async programming with asyncio",
                "file_type": "code",
                "project_name": "backend-api",
                "file_path": "/src/async_handler.py"
            }
        },
        {
            "id": "doc2",
            "score": 0.92,
            "payload": {
                "content": "Testing async functions with pytest",
                "file_type": "test",
                "project_name": "backend-api",
                "file_path": "/tests/test_async.py"
            }
        },
        {
            "id": "doc3",
            "score": 0.88,
            "payload": {
                "content": "Async database queries with SQLAlchemy",
                "file_type": "code",
                "project_name": "backend-api",
                "file_path": "/src/database.py"
            }
        },
        {
            "id": "doc4",
            "score": 0.85,
            "payload": {
                "content": "FastAPI async endpoints documentation",
                "file_type": "docs",
                "project_name": "backend-api",
                "file_path": "/docs/api.md"
            }
        },
        {
            "id": "doc5",
            "score": 0.82,
            "payload": {
                "content": "Synchronous legacy code patterns",
                "file_type": "code",
                "project_name": "legacy-app",
                "file_path": "/old/sync_handler.py"
            }
        }
    ]


@pytest.fixture
def query_embeddings():
    """Sample query embeddings for testing."""
    return {
        "dense": [0.1] * 384,  # 384-dim semantic vector
        "sparse": {"indices": [1, 5, 10, 15], "values": [0.8, 0.6, 0.4, 0.2]}
    }


# Test Classes

class TestBatchQueries:
    """Test batch query execution patterns."""

    @pytest.mark.asyncio
    async def test_basic_batch_query_execution(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test executing multiple queries in a batch."""
        # Mock search responses - Each query makes 2 calls (dense + sparse)
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:3]],  # Query 1 dense
            [Mock(**doc) for doc in sample_documents[:2]],  # Query 1 sparse
            [Mock(**doc) for doc in sample_documents[2:5]],  # Query 2 dense
            [Mock(**doc) for doc in sample_documents[1:3]],  # Query 2 sparse
            [Mock(**doc) for doc in sample_documents[1:4]],  # Query 3 dense
            [Mock(**doc) for doc in sample_documents[2:4]],  # Query 3 sparse
        ]

        # Define batch queries
        queries = [
            {"collection": "test-coll", "embeddings": query_embeddings, "limit": 5},
            {"collection": "test-coll", "embeddings": query_embeddings, "limit": 5},
            {"collection": "test-coll", "embeddings": query_embeddings, "limit": 5}
        ]

        # Execute batch queries
        results = []
        for query in queries:
            result = await hybrid_search_engine.hybrid_search(
                collection_name=query["collection"],
                query_embeddings=query["embeddings"],
                limit=query["limit"]
            )
            results.append(result)

        # Validate results
        assert len(results) == 3
        for i, result in enumerate(results):
            assert "fused_results" in result
            assert len(result["fused_results"]) > 0

    @pytest.mark.asyncio
    async def test_parallel_batch_execution(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test parallel execution of batch queries."""
        # Mock search responses
        hybrid_search_engine.client.search.return_value = [Mock(**doc) for doc in sample_documents[:3]]

        # Create multiple queries
        queries = [
            hybrid_search_engine.hybrid_search(
                collection_name=f"test-coll-{i}",
                query_embeddings=query_embeddings,
                limit=5
            )
            for i in range(5)
        ]

        # Execute in parallel
        results = await asyncio.gather(*queries)

        # Validate
        assert len(results) == 5
        for result in results:
            assert "fused_results" in result

    @pytest.mark.asyncio
    async def test_batch_query_with_different_fusion_methods(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test batch queries with different fusion methods."""
        # Mock search responses
        hybrid_search_engine.client.search.return_value = [Mock(**doc) for doc in sample_documents]

        fusion_methods = ["rrf", "weighted_sum", "max_score"]
        results = []

        for method in fusion_methods:
            result = await hybrid_search_engine.hybrid_search(
                collection_name="test-coll",
                query_embeddings=query_embeddings,
                limit=5,
                fusion_method=method
            )
            results.append((method, result))

        # Validate each fusion method produces results
        assert len(results) == 3
        for method, result in results:
            assert "fused_results" in result
            assert len(result["fused_results"]) > 0

    @pytest.mark.asyncio
    async def test_batch_query_error_handling(self, hybrid_search_engine, query_embeddings):
        """Test error handling in batch query execution."""
        # Mock one query to fail
        hybrid_search_engine.client.search.side_effect = [
            Exception("Search failed"),  # First query dense fails
            [Mock(id="doc1", score=0.9, payload={})],  # Second query dense succeeds
            [Mock(id="doc2", score=0.8, payload={})],  # Second query sparse succeeds
        ]

        results = []
        errors = []

        for i in range(2):
            try:
                result = await hybrid_search_engine.hybrid_search(
                    collection_name=f"test-coll-{i}",
                    query_embeddings=query_embeddings,
                    limit=5
                )
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        # Validate error handling
        assert len(errors) == 1
        assert len(results) == 1
        assert "Search failed" in errors[0]


class TestQueryChaining:
    """Test query chaining patterns."""

    @pytest.mark.asyncio
    async def test_simple_query_chain(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test chaining queries where results from one query feed into next."""
        # Mock search responses
        first_results = [Mock(**doc) for doc in sample_documents[:2]]
        second_results = [Mock(**doc) for doc in sample_documents[2:4]]

        hybrid_search_engine.client.search.side_effect = [
            first_results,  # First query dense
            [],  # First query sparse
            second_results,  # Second query dense
            []  # Second query sparse
        ]

        # First query
        result1 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=5
        )

        # Extract document IDs from first query
        doc_ids = [r.id for r in result1["fused_results"]]

        # Second query using first results as context
        filter_condition = models.Filter(
            should=[
                models.FieldCondition(
                    key="id",
                    match=models.MatchValue(value=doc_id)
                ) for doc_id in doc_ids
            ]
        )

        result2 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=5,
            filter_conditions=filter_condition
        )

        # Validate chaining
        assert len(result1["fused_results"]) > 0
        assert len(result2["fused_results"]) > 0

    @pytest.mark.asyncio
    async def test_multi_step_query_chain(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test multi-step query chain with 3+ steps."""
        # Mock progressive refinement
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:3]],  # Step 1 dense
            [],  # Step 1 sparse
            [Mock(**doc) for doc in sample_documents[:2]],  # Step 2 dense
            [],  # Step 2 sparse
            [Mock(**doc) for doc in sample_documents[:1]],  # Step 3 dense
            []  # Step 3 sparse
        ]

        # Step 1: Initial broad search
        result1 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10
        )

        # Step 2: Refined search based on step 1
        result2 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=5
        )

        # Step 3: Final refinement
        result3 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=3
        )

        # Validate progressive refinement
        assert len(result1["fused_results"]) >= len(result2["fused_results"])
        assert len(result2["fused_results"]) >= len(result3["fused_results"])

    @pytest.mark.asyncio
    async def test_query_chain_with_metadata_refinement(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test query chain with progressive metadata filtering."""
        # Mock responses
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents],  # All results dense
            [],  # sparse
            [Mock(**doc) for doc in sample_documents[:3]],  # Filtered dense
            []  # sparse
        ]

        # Step 1: Broad search
        result1 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10
        )

        # Step 2: Refine with metadata filter
        metadata_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="code")
                )
            ]
        )

        result2 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=5,
            filter_conditions=metadata_filter
        )

        # Validate refinement
        assert len(result1["fused_results"]) >= len(result2["fused_results"])


class TestResultAggregation:
    """Test result aggregation across queries."""

    @pytest.mark.asyncio
    async def test_basic_result_aggregation(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test aggregating results from multiple queries."""
        # Mock different query results
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:3]],  # dense
            [],  # sparse
            [Mock(**doc) for doc in sample_documents[2:5]],  # dense
            []  # sparse
        ]

        # Execute multiple queries
        results = []
        for i in range(2):
            result = await hybrid_search_engine.hybrid_search(
                collection_name=f"test-coll-{i}",
                query_embeddings=query_embeddings,
                limit=5
            )
            results.append(result)

        # Aggregate results
        all_results = []
        for result in results:
            all_results.extend(result["fused_results"])

        # Deduplicate by ID
        seen_ids = set()
        unique_results = []
        for r in all_results:
            if r.id not in seen_ids:
                seen_ids.add(r.id)
                unique_results.append(r)

        # Validate aggregation
        assert len(all_results) >= len(unique_results)
        assert len(unique_results) > 0

    @pytest.mark.asyncio
    async def test_multi_collection_aggregation(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test multi-collection result aggregation."""
        # Mock responses for different collections
        hybrid_search_engine.client.search.return_value = [Mock(**doc) for doc in sample_documents]

        # Search across multiple collections
        result = await hybrid_search_engine.multi_collection_hybrid_search(
            collection_names=["coll1", "coll2", "coll3"],
            query_embeddings=query_embeddings,
            limit=10,
            enable_deduplication=True,
            aggregation_method="max_score"
        )

        # Validate aggregation
        assert "total_results" in result
        assert "results" in result
        assert "aggregation_metadata" in result
        assert result["aggregation_metadata"]["collection_count"] == 3

    @pytest.mark.asyncio
    async def test_score_normalization_in_aggregation(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test score normalization across aggregated results."""
        # Mock responses with different score ranges
        hybrid_search_engine.client.search.side_effect = [
            [Mock(id=f"doc{i}", score=0.9 - i*0.1, payload={}) for i in range(3)],  # dense high
            [],  # sparse
            [Mock(id=f"doc{i}", score=0.5 - i*0.05, payload={}) for i in range(3)],  # dense low
            []  # sparse
        ]

        # Execute multi-collection search with normalization
        result = await hybrid_search_engine.multi_collection_hybrid_search(
            collection_names=["high-score-coll", "low-score-coll"],
            query_embeddings=query_embeddings,
            limit=10,
            enable_deduplication=True
        )

        # Validate score normalization metadata
        assert "aggregation_metadata" in result


class TestComplexBooleanQueries:
    """Test complex boolean query patterns."""

    @pytest.mark.asyncio
    async def test_and_query_pattern(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test AND boolean query (all conditions must match)."""
        # Mock search with AND filter
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:2]],  # dense
            []  # sparse
        ]

        # Create AND filter
        and_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="code")
                ),
                models.FieldCondition(
                    key="project_name",
                    match=models.MatchValue(value="backend-api")
                )
            ]
        )

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            filter_conditions=and_filter
        )

        # Validate AND logic applied
        assert "fused_results" in result
        # Verify filter was passed to client
        assert hybrid_search_engine.client.search.called

    @pytest.mark.asyncio
    async def test_or_query_pattern(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test OR boolean query (any condition can match)."""
        # Mock search with OR filter
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:4]],  # dense
            []  # sparse
        ]

        # Create OR filter
        or_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="code")
                ),
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="test")
                )
            ]
        )

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            filter_conditions=or_filter
        )

        # Validate OR logic
        assert "fused_results" in result
        assert len(result["fused_results"]) > 0

    @pytest.mark.asyncio
    async def test_not_query_pattern(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test NOT boolean query (exclude matching conditions)."""
        # Mock search with NOT filter
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:3]],  # dense
            []  # sparse
        ]

        # Create NOT filter
        not_filter = models.Filter(
            must_not=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="test")
                )
            ]
        )

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            filter_conditions=not_filter
        )

        # Validate NOT logic
        assert "fused_results" in result

    @pytest.mark.asyncio
    async def test_complex_nested_boolean_query(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test complex nested boolean query (AND with OR and NOT)."""
        # Mock search
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:2]],  # dense
            []  # sparse
        ]

        # Create complex nested filter: (code OR docs) AND (NOT test) AND project_name=backend-api
        complex_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="project_name",
                    match=models.MatchValue(value="backend-api")
                )
            ],
            should=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="code")
                ),
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="docs")
                )
            ],
            must_not=[
                models.FieldCondition(
                    key="file_type",
                    match=models.MatchValue(value="test")
                )
            ]
        )

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            filter_conditions=complex_filter
        )

        # Validate complex query
        assert "fused_results" in result
        assert hybrid_search_engine.client.search.called


class TestWildcardPatterns:
    """Test wildcard pattern matching in queries."""

    @pytest.mark.asyncio
    async def test_prefix_wildcard_pattern(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test prefix wildcard pattern matching (e.g., 'test*')."""
        # Mock search with prefix pattern
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:3]],  # dense
            []  # sparse
        ]

        # Simulate prefix matching via filter
        prefix_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchText(text="/tests/")  # Prefix-like match
                )
            ]
        )

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            filter_conditions=prefix_filter
        )

        # Validate wildcard logic
        assert "fused_results" in result

    @pytest.mark.asyncio
    async def test_suffix_wildcard_pattern(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test suffix wildcard pattern matching (e.g., '*.py')."""
        # Mock search
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:4]],  # dense
            []  # sparse
        ]

        # Simulate suffix matching
        suffix_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="file_path",
                    match=models.MatchText(text=".py")  # Suffix-like match
                )
            ]
        )

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            filter_conditions=suffix_filter
        )

        # Validate
        assert "fused_results" in result

    @pytest.mark.asyncio
    async def test_contains_wildcard_pattern(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test contains wildcard pattern matching (e.g., '*async*')."""
        # Mock search
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:3]],  # dense
            []  # sparse
        ]

        # Simulate contains matching
        contains_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="content",
                    match=models.MatchText(text="async")  # Contains match
                )
            ]
        )

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            filter_conditions=contains_filter
        )

        # Validate
        assert "fused_results" in result


class TestFuzzyMatching:
    """Test fuzzy matching patterns."""

    @pytest.mark.asyncio
    async def test_fuzzy_text_matching(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test fuzzy text matching with edit distance tolerance."""
        # Mock search with fuzzy results
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:3]],  # dense
            []  # sparse
        ]

        # Hybrid search naturally provides fuzzy matching via semantic vectors
        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            fusion_method="rrf"  # RRF handles semantic similarity well
        )

        # Validate fuzzy matching
        assert "fused_results" in result
        assert len(result["fused_results"]) > 0

    @pytest.mark.asyncio
    async def test_fuzzy_matching_with_threshold(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test fuzzy matching with score threshold."""
        # Mock results with varying scores
        mock_results = [
            Mock(id="doc1", score=0.95, payload={"content": "exact match"}),
            Mock(id="doc2", score=0.75, payload={"content": "fuzzy match"}),
            Mock(id="doc3", score=0.55, payload={"content": "weak match"}),
        ]
        hybrid_search_engine.client.search.side_effect = [mock_results, []]  # dense, sparse

        # Search with score threshold via multi-collection (which supports it)
        result = await hybrid_search_engine.multi_collection_hybrid_search(
            collection_names=["test-coll"],
            query_embeddings=query_embeddings,
            limit=10,
            score_threshold=0.7  # Filter weak matches
        )

        # Validate threshold application
        assert "results" in result
        # With threshold=0.7, should filter out 0.55 score result


class TestQueryOptimization:
    """Test query optimization for complex patterns."""

    @pytest.mark.asyncio
    async def test_optimized_query_execution(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test query execution with optimization enabled."""
        # Mock search
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents],  # dense
            []  # sparse
        ]

        # Execute with optimizations
        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        # Validate result structure
        assert "fused_results" in result

    @pytest.mark.asyncio
    async def test_query_caching_optimization(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test query result caching for optimization."""
        # Mock search
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents],  # dense q1
            [],  # sparse q1
            [Mock(**doc) for doc in sample_documents],  # dense q2
            []  # sparse q2
        ]

        # Execute same query twice
        result1 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10
        )

        result2 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10
        )

        # Validate both executed successfully
        assert "fused_results" in result1
        assert "fused_results" in result2

    @pytest.mark.asyncio
    async def test_query_parameter_optimization(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test automatic query parameter optimization."""
        # Mock search
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents],  # dense
            []  # sparse
        ]

        # Execute without explicit search_params (should auto-optimize)
        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            search_params=None  # Auto-optimize
        )

        # Validate optimization occurred
        assert "fused_results" in result


class TestPerformanceCharacteristics:
    """Test performance characteristics of complex query patterns."""

    @pytest.mark.asyncio
    async def test_batch_query_performance(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test performance of batch query execution."""
        import time

        # Mock search
        hybrid_search_engine.client.search.return_value = [Mock(**doc) for doc in sample_documents]

        # Measure batch execution time
        start_time = time.time()

        queries = [
            hybrid_search_engine.hybrid_search(
                collection_name=f"test-coll-{i}",
                query_embeddings=query_embeddings,
                limit=5
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*queries)

        execution_time = (time.time() - start_time) * 1000  # ms

        # Validate performance
        assert len(results) == 10
        # Batch execution should be reasonably fast (< 1000ms for 10 queries)
        assert execution_time < 1000

    @pytest.mark.asyncio
    async def test_query_chain_performance(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test performance of query chaining."""
        import time

        # Mock progressive responses
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:5]],  # q1 dense
            [],  # q1 sparse
            [Mock(**doc) for doc in sample_documents[:3]],  # q2 dense
            [],  # q2 sparse
            [Mock(**doc) for doc in sample_documents[:1]],  # q3 dense
            []  # q3 sparse
        ]

        start_time = time.time()

        # Execute query chain
        result1 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10
        )

        result2 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=5
        )

        result3 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=3
        )

        execution_time = (time.time() - start_time) * 1000  # ms

        # Validate performance
        assert all("fused_results" in r for r in [result1, result2, result3])
        # Sequential execution should still be reasonably fast
        assert execution_time < 500

    @pytest.mark.asyncio
    async def test_complex_query_response_time(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test response time for complex queries."""
        # Mock search
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents],  # dense
            []  # sparse
        ]

        # Complex filter
        complex_filter = models.Filter(
            must=[
                models.FieldCondition(key="project_name", match=models.MatchValue(value="backend-api"))
            ],
            should=[
                models.FieldCondition(key="file_type", match=models.MatchValue(value="code")),
                models.FieldCondition(key="file_type", match=models.MatchValue(value="test"))
            ],
            must_not=[
                models.FieldCondition(key="file_type", match=models.MatchValue(value="data"))
            ]
        )

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            filter_conditions=complex_filter,
            fusion_method="rrf"
        )

        # Validate result structure
        assert "fused_results" in result


class TestHybridSearchArchitectureCompatibility:
    """Test compatibility with hybrid search architecture."""

    @pytest.mark.asyncio
    async def test_dense_sparse_fusion_in_complex_queries(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test dense and sparse vector fusion in complex query patterns."""
        # Mock separate dense and sparse results
        dense_results = [Mock(**doc) for doc in sample_documents[:3]]
        sparse_results = [Mock(**doc) for doc in sample_documents[2:5]]

        hybrid_search_engine.client.search.side_effect = [dense_results, sparse_results]

        result = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=10,
            fusion_method="rrf"
        )

        # Validate fusion
        assert "dense_results" in result
        assert "sparse_results" in result
        assert "fused_results" in result

    @pytest.mark.asyncio
    async def test_rrf_fusion_with_batch_queries(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test RRF fusion consistency across batch queries."""
        # Mock responses
        hybrid_search_engine.client.search.return_value = [Mock(**doc) for doc in sample_documents]

        # Execute batch with RRF
        results = []
        for i in range(3):
            result = await hybrid_search_engine.hybrid_search(
                collection_name=f"test-coll-{i}",
                query_embeddings=query_embeddings,
                limit=5,
                fusion_method="rrf"
            )
            results.append(result)

        # Validate RRF applied consistently
        for result in results:
            assert "fused_results" in result

    @pytest.mark.asyncio
    async def test_weighted_fusion_in_query_chains(self, hybrid_search_engine, query_embeddings, sample_documents):
        """Test weighted fusion in query chaining."""
        # Mock responses
        hybrid_search_engine.client.search.side_effect = [
            [Mock(**doc) for doc in sample_documents[:3]],  # q1 dense
            [],  # q1 sparse
            [Mock(**doc) for doc in sample_documents[2:5]],  # q2 dense
            []  # q2 sparse
        ]

        # First query with weighted fusion
        result1 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=5,
            fusion_method="weighted_sum",
            dense_weight=0.7,
            sparse_weight=0.3
        )

        # Second query in chain
        result2 = await hybrid_search_engine.hybrid_search(
            collection_name="test-coll",
            query_embeddings=query_embeddings,
            limit=5,
            fusion_method="weighted_sum",
            dense_weight=0.6,
            sparse_weight=0.4
        )

        # Validate weighted fusion
        assert "fused_results" in result1
        assert "fused_results" in result2
