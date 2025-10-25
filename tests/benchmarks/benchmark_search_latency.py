"""
Search latency benchmarks with percentile metrics.

Measures performance of search operations across different search types,
query complexities, and result set sizes. Provides detailed latency distributions
including p50, p95, and p99 percentiles.

Run with: uv run pytest tests/benchmarks/benchmark_search_latency.py --benchmark-only
"""

import asyncio
import statistics

import pytest
from common.core.embeddings import EmbeddingService
from common.core.hybrid_search import HybridSearchEngine
from common.core.ssl_config import suppress_qdrant_ssl_warnings
from qdrant_client import QdrantClient
from qdrant_client.http import models


class SearchBenchmarkFixtures:
    """Helper class for setting up search benchmark test data."""

    @staticmethod
    async def create_test_collection(
        client: QdrantClient,
        collection_name: str,
        num_documents: int = 1000,
    ) -> None:
        """
        Create a test collection with sample documents for benchmarking.

        Args:
            client: Qdrant client instance
            collection_name: Name for the test collection
            num_documents: Number of sample documents to create
        """
        # Delete collection if it already exists
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass  # Collection doesn't exist, which is fine

        # Create collection with dense and sparse vectors
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            },
        )

        # Generate sample documents with realistic embeddings
        embedding_service = EmbeddingService()
        await embedding_service.initialize()

        sample_texts = [
            "Machine learning algorithms for natural language processing",
            "Vector database search optimization techniques",
            "Python programming best practices and design patterns",
            "Cloud infrastructure deployment strategies",
            "Data science visualization and analytics",
            "Software testing frameworks and methodologies",
            "API design principles for microservices",
            "Database indexing and query optimization",
            "Security best practices for web applications",
            "Performance monitoring and observability tools",
        ]

        # Create batches of documents
        batch_size = 100
        for i in range(0, num_documents, batch_size):
            points = []
            for j in range(batch_size):
                doc_idx = (i + j) % len(sample_texts)
                text = sample_texts[doc_idx]

                # Generate embeddings
                embeddings = await embedding_service.generate_embeddings(text)

                point = models.PointStruct(
                    id=i + j,
                    vector={
                        "dense": embeddings["dense"],
                        "sparse": models.SparseVector(
                            indices=embeddings["sparse"]["indices"],
                            values=embeddings["sparse"]["values"],
                        ),
                    },
                    payload={
                        "text": text,
                        "index": i + j,
                        "category": f"category_{(i + j) % 5}",
                    },
                )
                points.append(point)

            # Upload batch
            client.upsert(
                collection_name=collection_name,
                points=points,
            )

        await embedding_service.close()

    @staticmethod
    async def generate_query_embeddings(
        query_text: str,
    ) -> dict[str, any]:
        """
        Generate embeddings for a query.

        Args:
            query_text: Text to generate embeddings for

        Returns:
            Dict with 'dense' and 'sparse' embeddings
        """
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings = await embedding_service.generate_embeddings(query_text)
        await embedding_service.close()
        return embeddings


# Fixtures for test infrastructure
@pytest.fixture(scope="module")
def qdrant_client():
    """Create Qdrant client for benchmarking."""
    with suppress_qdrant_ssl_warnings():
        client = QdrantClient(url="http://localhost:6333")
    yield client
    client.close()


@pytest.fixture(scope="module")
def benchmark_collection_name():
    """Provide collection name for benchmarks."""
    return "benchmark_search_latency"


@pytest.fixture(scope="module")
def test_collection(qdrant_client, benchmark_collection_name):
    """Create and populate test collection for benchmarks."""
    # Run async setup
    asyncio.run(
        SearchBenchmarkFixtures.create_test_collection(
            qdrant_client,
            benchmark_collection_name,
            num_documents=1000,
        )
    )
    yield benchmark_collection_name
    # Cleanup
    try:
        qdrant_client.delete_collection(benchmark_collection_name)
    except Exception:
        pass


# Query embedding fixtures for different complexities
@pytest.fixture(scope="module")
def short_query_embeddings():
    """Embeddings for short query (1-3 words)."""
    return asyncio.run(
        SearchBenchmarkFixtures.generate_query_embeddings("machine learning")
    )


@pytest.fixture(scope="module")
def medium_query_embeddings():
    """Embeddings for medium query (5-10 words)."""
    return asyncio.run(
        SearchBenchmarkFixtures.generate_query_embeddings(
            "best practices for database query optimization"
        )
    )


@pytest.fixture(scope="module")
def long_query_embeddings():
    """Embeddings for long query (20+ words)."""
    return asyncio.run(
        SearchBenchmarkFixtures.generate_query_embeddings(
            "comprehensive guide to implementing scalable microservices architecture "
            "with cloud-native technologies and containerization strategies for modern "
            "distributed systems and applications"
        )
    )


# Helper function for percentile calculation
def calculate_percentiles(benchmark_stats) -> dict[str, float]:
    """
    Calculate percentile metrics from benchmark stats.

    Args:
        benchmark_stats: pytest-benchmark stats object

    Returns:
        Dict with p50, p95, p99 metrics in milliseconds
    """
    if hasattr(benchmark_stats, 'stats') and hasattr(benchmark_stats.stats, 'data'):
        # Get raw timing data
        data = benchmark_stats.stats.data
        if data:
            # Convert to milliseconds
            data_ms = [t * 1000 for t in data]
            return {
                "p50_ms": statistics.quantiles(data_ms, n=100)[49],  # 50th percentile
                "p95_ms": statistics.quantiles(data_ms, n=100)[94],  # 95th percentile
                "p99_ms": statistics.quantiles(data_ms, n=100)[98],  # 99th percentile
                "min_ms": min(data_ms),
                "max_ms": max(data_ms),
                "mean_ms": statistics.mean(data_ms),
                "median_ms": statistics.median(data_ms),
            }
    return {}


# Dense search benchmarks
@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_dense_search_short_query_limit_5(
    benchmark, qdrant_client, test_collection, short_query_embeddings
):
    """Benchmark dense search with short query and limit=5."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={"dense": short_query_embeddings["dense"]},
                limit=5,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)

    # Print percentiles for analysis
    if percentiles:
        print(f"\nDense search (short, limit=5) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_dense_search_short_query_limit_10(
    benchmark, qdrant_client, test_collection, short_query_embeddings
):
    """Benchmark dense search with short query and limit=10."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={"dense": short_query_embeddings["dense"]},
                limit=10,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nDense search (short, limit=10) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_dense_search_medium_query_limit_20(
    benchmark, qdrant_client, test_collection, medium_query_embeddings
):
    """Benchmark dense search with medium query and limit=20."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={"dense": medium_query_embeddings["dense"]},
                limit=20,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nDense search (medium, limit=20) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_dense_search_long_query_limit_50(
    benchmark, qdrant_client, test_collection, long_query_embeddings
):
    """Benchmark dense search with long query and limit=50."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={"dense": long_query_embeddings["dense"]},
                limit=50,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nDense search (long, limit=50) percentiles: {percentiles}")

    assert "fused_results" in result


# Sparse search benchmarks
@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_sparse_search_short_query_limit_5(
    benchmark, qdrant_client, test_collection, short_query_embeddings
):
    """Benchmark sparse search with short query and limit=5."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={"sparse": short_query_embeddings["sparse"]},
                limit=5,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nSparse search (short, limit=5) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_sparse_search_medium_query_limit_10(
    benchmark, qdrant_client, test_collection, medium_query_embeddings
):
    """Benchmark sparse search with medium query and limit=10."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={"sparse": medium_query_embeddings["sparse"]},
                limit=10,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nSparse search (medium, limit=10) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_sparse_search_long_query_limit_20(
    benchmark, qdrant_client, test_collection, long_query_embeddings
):
    """Benchmark sparse search with long query and limit=20."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings={"sparse": long_query_embeddings["sparse"]},
                limit=20,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nSparse search (long, limit=20) percentiles: {percentiles}")

    assert "fused_results" in result


# Hybrid search benchmarks - RRF fusion
@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_hybrid_rrf_short_query_limit_10(
    benchmark, qdrant_client, test_collection, short_query_embeddings
):
    """Benchmark hybrid search with RRF fusion, short query, limit=10."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=short_query_embeddings,
                limit=10,
                fusion_method="rrf",
                dense_weight=1.0,
                sparse_weight=1.0,
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nHybrid RRF (short, limit=10) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_hybrid_rrf_medium_query_limit_20(
    benchmark, qdrant_client, test_collection, medium_query_embeddings
):
    """Benchmark hybrid search with RRF fusion, medium query, limit=20."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=medium_query_embeddings,
                limit=20,
                fusion_method="rrf",
                dense_weight=1.0,
                sparse_weight=1.0,
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nHybrid RRF (medium, limit=20) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_hybrid_rrf_long_query_limit_50(
    benchmark, qdrant_client, test_collection, long_query_embeddings
):
    """Benchmark hybrid search with RRF fusion, long query, limit=50."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=long_query_embeddings,
                limit=50,
                fusion_method="rrf",
                dense_weight=1.0,
                sparse_weight=1.0,
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nHybrid RRF (long, limit=50) percentiles: {percentiles}")

    assert "fused_results" in result


# Hybrid search benchmarks - Weighted sum fusion
@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_hybrid_weighted_sum_short_query_limit_10(
    benchmark, qdrant_client, test_collection, short_query_embeddings
):
    """Benchmark hybrid search with weighted sum fusion, short query, limit=10."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=short_query_embeddings,
                limit=10,
                fusion_method="weighted_sum",
                dense_weight=0.7,
                sparse_weight=0.3,
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nHybrid weighted_sum (short, limit=10) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_hybrid_weighted_sum_medium_query_limit_20(
    benchmark, qdrant_client, test_collection, medium_query_embeddings
):
    """Benchmark hybrid search with weighted sum fusion, medium query, limit=20."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=medium_query_embeddings,
                limit=20,
                fusion_method="weighted_sum",
                dense_weight=0.7,
                sparse_weight=0.3,
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nHybrid weighted_sum (medium, limit=20) percentiles: {percentiles}")

    assert "fused_results" in result


# Hybrid search benchmarks - Max score fusion
@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_hybrid_max_score_short_query_limit_10(
    benchmark, qdrant_client, test_collection, short_query_embeddings
):
    """Benchmark hybrid search with max score fusion, short query, limit=10."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=short_query_embeddings,
                limit=10,
                fusion_method="max_score",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nHybrid max_score (short, limit=10) percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_hybrid_max_score_long_query_limit_50(
    benchmark, qdrant_client, test_collection, long_query_embeddings
):
    """Benchmark hybrid search with max score fusion, long query, limit=50."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    def run_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=long_query_embeddings,
                limit=50,
                fusion_method="max_score",
            )
        )

    result = benchmark(run_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nHybrid max_score (long, limit=50) percentiles: {percentiles}")

    assert "fused_results" in result


# Cold start vs warm cache benchmarks
@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_cold_start_search(
    benchmark, qdrant_client, benchmark_collection_name, short_query_embeddings
):
    """Benchmark cold start search (fresh engine instance each time)."""

    def run_cold_search():
        # Create new engine instance each time to simulate cold start
        engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )
        return asyncio.run(
            engine.hybrid_search(
                collection_name=benchmark_collection_name,
                query_embeddings=short_query_embeddings,
                limit=10,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_cold_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nCold start search percentiles: {percentiles}")

    assert "fused_results" in result


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_warm_cache_search(
    benchmark, qdrant_client, test_collection, short_query_embeddings
):
    """Benchmark warm cache search (reused engine instance)."""
    # Create engine once, reuse for all iterations
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    # Warm up the cache with a few searches
    for _ in range(3):
        asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=short_query_embeddings,
                limit=10,
                fusion_method="rrf",
            )
        )

    def run_warm_search():
        return asyncio.run(
            engine.hybrid_search(
                collection_name=test_collection,
                query_embeddings=short_query_embeddings,
                limit=10,
                fusion_method="rrf",
            )
        )

    result = benchmark(run_warm_search)
    percentiles = calculate_percentiles(benchmark)
    if percentiles:
        print(f"\nWarm cache search percentiles: {percentiles}")

    assert "fused_results" in result
