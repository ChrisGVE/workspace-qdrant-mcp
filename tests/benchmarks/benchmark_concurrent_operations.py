"""
Concurrent operation benchmarks with contention analysis.

Measures performance of concurrent operations including multiple simultaneous
file ingestions, concurrent search operations, mixed workloads, and resource
contention scenarios. Tests various concurrency levels (2, 5, 10, 20) to
identify scalability characteristics and contention bottlenecks.

Run with:
    uv run pytest tests/benchmarks/benchmark_concurrent_operations.py --benchmark-only

Run specific concurrency level:
    uv run pytest tests/benchmarks/benchmark_concurrent_operations.py -k "concurrent_2" --benchmark-only

Save baseline for regression detection:
    uv run pytest tests/benchmarks/benchmark_concurrent_operations.py --benchmark-only --benchmark-save=concurrent_baseline

Compare for regressions:
    uv run pytest tests/benchmarks/benchmark_concurrent_operations.py --benchmark-only --benchmark-compare=concurrent_baseline

Metrics Explained:
- Total time: Wall-clock time for all concurrent operations
- Throughput: Operations per second (total ops / total time)
- Mean latency: Average time per operation
- Contention factor: Ratio of concurrent time to sequential time (1.0 = perfect scaling)
"""

import asyncio
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from common.core.embeddings import EmbeddingService
from common.core.hybrid_search import HybridSearchEngine
from common.core.ssl_config import suppress_qdrant_ssl_warnings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from wqm_cli.cli.parsers import CodeParser, MarkdownParser, TextParser


class ConcurrentTestDataGenerator:
    """Generates test data for concurrent operation benchmarks."""

    @staticmethod
    def generate_test_files(
        num_files: int, size_kb: int, extension: str, tmp_dir: Path
    ) -> list[Path]:
        """Create test files for concurrent ingestion."""
        files = []
        content_generators = {
            ".txt": lambda: "Sample text content. " * (size_kb * 50),
            ".py": lambda: f"def func_{i}():\n    return {i}\n\n" * (size_kb * 10),
            ".md": lambda: f"# Heading {i}\n\nContent here.\n\n" * (size_kb * 15),
        }

        generator = content_generators.get(
            extension, lambda: "Default content. " * (size_kb * 50)
        )

        for i in range(num_files):
            file_path = tmp_dir / f"concurrent_test_{i}{extension}"
            content = generator() if extension == ".txt" else content_generators[extension]()
            file_path.write_text(content, encoding="utf-8")
            files.append(file_path)

        return files


class ConcurrentBenchmarkMetrics:
    """Calculate and store concurrent operation metrics."""

    @staticmethod
    def calculate_throughput(num_operations: int, total_time_seconds: float) -> float:
        """Calculate operations per second."""
        return num_operations / total_time_seconds if total_time_seconds > 0 else 0

    @staticmethod
    def calculate_contention_factor(
        concurrent_time: float, sequential_time: float
    ) -> float:
        """
        Calculate contention factor.

        Perfect linear scaling = 1.0
        Higher values indicate more contention/overhead
        """
        expected_time = sequential_time  # Ideally concurrent = sequential for N workers
        return concurrent_time / expected_time if expected_time > 0 else float("inf")

    @staticmethod
    def calculate_percentiles(data: list[float]) -> dict[str, float]:
        """Calculate percentile metrics from timing data."""
        if not data:
            return {}

        # Convert to milliseconds
        data_ms = [t * 1000 for t in data]

        return {
            "p50_ms": statistics.quantiles(data_ms, n=100)[49],
            "p95_ms": statistics.quantiles(data_ms, n=100)[94],
            "p99_ms": statistics.quantiles(data_ms, n=100)[98],
            "min_ms": min(data_ms),
            "max_ms": max(data_ms),
            "mean_ms": statistics.mean(data_ms),
            "median_ms": statistics.median(data_ms),
        }


# Fixtures
@pytest.fixture(scope="module")
def tmp_concurrent_dir():
    """Create temporary directory for concurrent test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="module")
def qdrant_client():
    """Create Qdrant client for concurrent search benchmarks."""
    with suppress_qdrant_ssl_warnings():
        client = QdrantClient(url="http://localhost:6333")
    yield client
    client.close()


@pytest.fixture(scope="module")
def concurrent_search_collection(qdrant_client):
    """Create test collection for concurrent search benchmarks."""
    collection_name = "benchmark_concurrent_search"

    # Delete if exists
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    # Populate with test documents
    async def populate():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()

        sample_texts = [
            f"Document {i}: Machine learning and artificial intelligence content."
            for i in range(100)
        ]

        points = []
        for i, text in enumerate(sample_texts):
            embeddings = await embedding_service.generate_embeddings(text)
            point = models.PointStruct(
                id=i,
                vector={
                    "dense": embeddings["dense"],
                    "sparse": models.SparseVector(
                        indices=embeddings["sparse"]["indices"],
                        values=embeddings["sparse"]["values"],
                    ),
                },
                payload={"text": text, "index": i},
            )
            points.append(point)

        qdrant_client.upsert(collection_name=collection_name, points=points)
        await embedding_service.close()

    asyncio.run(populate())

    yield collection_name

    # Cleanup
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass


# ============================================================================
# Concurrent File Ingestion Benchmarks
# ============================================================================


@pytest.mark.benchmark
def test_concurrent_ingestion_2_files(benchmark, tmp_concurrent_dir):
    """Benchmark concurrent ingestion of 2 small files."""
    files = ConcurrentTestDataGenerator.generate_test_files(
        2, 10, ".txt", tmp_concurrent_dir
    )
    parser = TextParser()

    async def concurrent_parse():
        tasks = [parser.parse(f) for f in files]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_parse())

    results = benchmark(run_benchmark)
    assert len(results) == 2
    assert all(r.content_hash for r in results)


@pytest.mark.benchmark
def test_concurrent_ingestion_5_files(benchmark, tmp_concurrent_dir):
    """Benchmark concurrent ingestion of 5 medium files."""
    files = ConcurrentTestDataGenerator.generate_test_files(
        5, 50, ".py", tmp_concurrent_dir
    )
    parser = CodeParser()

    async def concurrent_parse():
        tasks = [parser.parse(f) for f in files]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_parse())

    results = benchmark(run_benchmark)
    assert len(results) == 5
    print(f"\nConcurrent ingestion (5 files) - Total time: {benchmark.stats.mean:.4f}s")


@pytest.mark.benchmark
def test_concurrent_ingestion_10_files(benchmark, tmp_concurrent_dir):
    """Benchmark concurrent ingestion of 10 files with mixed types."""
    files = []
    files.extend(
        ConcurrentTestDataGenerator.generate_test_files(
            3, 20, ".txt", tmp_concurrent_dir
        )
    )
    files.extend(
        ConcurrentTestDataGenerator.generate_test_files(
            4, 20, ".py", tmp_concurrent_dir
        )
    )
    files.extend(
        ConcurrentTestDataGenerator.generate_test_files(
            3, 20, ".md", tmp_concurrent_dir
        )
    )

    async def concurrent_parse():
        tasks = []
        for file_path in files:
            if file_path.suffix == ".py":
                parser = CodeParser()
            elif file_path.suffix == ".md":
                parser = MarkdownParser()
            else:
                parser = TextParser()
            tasks.append(parser.parse(file_path))

        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_parse())

    results = benchmark(run_benchmark)
    assert len(results) == 10

    throughput = ConcurrentBenchmarkMetrics.calculate_throughput(
        10, benchmark.stats.mean
    )
    print(f"\nConcurrent ingestion (10 files) - Throughput: {throughput:.2f} files/sec")


@pytest.mark.benchmark
def test_concurrent_ingestion_20_files(benchmark, tmp_concurrent_dir):
    """Benchmark concurrent ingestion of 20 large files - stress test."""
    files = ConcurrentTestDataGenerator.generate_test_files(
        20, 100, ".txt", tmp_concurrent_dir
    )
    parser = TextParser()

    async def concurrent_parse():
        tasks = [parser.parse(f) for f in files]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_parse())

    results = benchmark(run_benchmark)
    assert len(results) == 20

    throughput = ConcurrentBenchmarkMetrics.calculate_throughput(
        20, benchmark.stats.mean
    )
    print(f"\nConcurrent ingestion (20 files) - Throughput: {throughput:.2f} files/sec")


# ============================================================================
# Concurrent Search Operations Benchmarks
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_concurrent_search_2_queries(
    benchmark, qdrant_client, concurrent_search_collection
):
    """Benchmark 2 concurrent search operations."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    async def get_embeddings(query: str):
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings = await embedding_service.generate_embeddings(query)
        await embedding_service.close()
        return embeddings

    query_embeddings_1 = asyncio.run(get_embeddings("machine learning algorithms"))
    query_embeddings_2 = asyncio.run(get_embeddings("artificial intelligence research"))

    async def concurrent_search():
        tasks = [
            engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=query_embeddings_1,
                limit=10,
                fusion_method="rrf",
            ),
            engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=query_embeddings_2,
                limit=10,
                fusion_method="rrf",
            ),
        ]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_search())

    results = benchmark(run_benchmark)
    assert len(results) == 2
    assert all("fused_results" in r for r in results)


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_concurrent_search_5_queries(
    benchmark, qdrant_client, concurrent_search_collection
):
    """Benchmark 5 concurrent search operations."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    queries = [
        "machine learning",
        "deep learning",
        "neural networks",
        "natural language processing",
        "computer vision",
    ]

    async def get_all_embeddings():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings_list = []
        for query in queries:
            embeddings = await embedding_service.generate_embeddings(query)
            embeddings_list.append(embeddings)
        await embedding_service.close()
        return embeddings_list

    all_embeddings = asyncio.run(get_all_embeddings())

    async def concurrent_search():
        tasks = [
            engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=embeddings,
                limit=10,
                fusion_method="rrf",
            )
            for embeddings in all_embeddings
        ]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_search())

    results = benchmark(run_benchmark)
    assert len(results) == 5

    throughput = ConcurrentBenchmarkMetrics.calculate_throughput(
        5, benchmark.stats.mean
    )
    print(f"\nConcurrent search (5 queries) - Throughput: {throughput:.2f} searches/sec")


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_concurrent_search_10_queries(
    benchmark, qdrant_client, concurrent_search_collection
):
    """Benchmark 10 concurrent search operations - stress test."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    queries = [f"search query number {i}" for i in range(10)]

    async def get_all_embeddings():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings_list = []
        for query in queries:
            embeddings = await embedding_service.generate_embeddings(query)
            embeddings_list.append(embeddings)
        await embedding_service.close()
        return embeddings_list

    all_embeddings = asyncio.run(get_all_embeddings())

    async def concurrent_search():
        tasks = [
            engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=embeddings,
                limit=10,
                fusion_method="rrf",
            )
            for embeddings in all_embeddings
        ]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_search())

    results = benchmark(run_benchmark)
    assert len(results) == 10

    throughput = ConcurrentBenchmarkMetrics.calculate_throughput(
        10, benchmark.stats.mean
    )
    print(f"\nConcurrent search (10 queries) - Throughput: {throughput:.2f} searches/sec")


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_concurrent_search_20_queries(
    benchmark, qdrant_client, concurrent_search_collection
):
    """Benchmark 20 concurrent search operations - maximum stress test."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    queries = [f"query {i} testing concurrent search" for i in range(20)]

    async def get_all_embeddings():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings_list = []
        for query in queries:
            embeddings = await embedding_service.generate_embeddings(query)
            embeddings_list.append(embeddings)
        await embedding_service.close()
        return embeddings_list

    all_embeddings = asyncio.run(get_all_embeddings())

    async def concurrent_search():
        tasks = [
            engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=embeddings,
                limit=10,
                fusion_method="rrf",
            )
            for embeddings in all_embeddings
        ]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_search())

    results = benchmark(run_benchmark)
    assert len(results) == 20

    throughput = ConcurrentBenchmarkMetrics.calculate_throughput(
        20, benchmark.stats.mean
    )
    print(f"\nConcurrent search (20 queries) - Throughput: {throughput:.2f} searches/sec")


# ============================================================================
# Mixed Workload Benchmarks
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_mixed_workload_5_ingestion_5_search(
    benchmark, tmp_concurrent_dir, qdrant_client, concurrent_search_collection
):
    """Benchmark mixed workload: 5 file ingestions + 5 searches simultaneously."""
    # Prepare ingestion files
    files = ConcurrentTestDataGenerator.generate_test_files(
        5, 20, ".txt", tmp_concurrent_dir
    )
    parser = TextParser()

    # Prepare search engine and queries
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    queries = [f"mixed workload query {i}" for i in range(5)]

    async def get_embeddings_list():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings_list = []
        for query in queries:
            embeddings = await embedding_service.generate_embeddings(query)
            embeddings_list.append(embeddings)
        await embedding_service.close()
        return embeddings_list

    all_embeddings = asyncio.run(get_embeddings_list())

    async def mixed_workload():
        # Create ingestion tasks
        ingestion_tasks = [parser.parse(f) for f in files]

        # Create search tasks
        search_tasks = [
            engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=embeddings,
                limit=10,
                fusion_method="rrf",
            )
            for embeddings in all_embeddings
        ]

        # Run all tasks concurrently
        all_tasks = ingestion_tasks + search_tasks
        results = await asyncio.gather(*all_tasks)
        return results

    def run_benchmark():
        return asyncio.run(mixed_workload())

    results = benchmark(run_benchmark)
    assert len(results) == 10

    throughput = ConcurrentBenchmarkMetrics.calculate_throughput(
        10, benchmark.stats.mean
    )
    print(f"\nMixed workload (5+5) - Throughput: {throughput:.2f} ops/sec")


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_mixed_workload_10_ingestion_10_search(
    benchmark, tmp_concurrent_dir, qdrant_client, concurrent_search_collection
):
    """Benchmark heavy mixed workload: 10 ingestions + 10 searches simultaneously."""
    # Prepare ingestion files
    files = ConcurrentTestDataGenerator.generate_test_files(
        10, 50, ".py", tmp_concurrent_dir
    )
    parser = CodeParser()

    # Prepare search engine and queries
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    queries = [f"heavy workload query {i}" for i in range(10)]

    async def get_embeddings_list():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings_list = []
        for query in queries:
            embeddings = await embedding_service.generate_embeddings(query)
            embeddings_list.append(embeddings)
        await embedding_service.close()
        return embeddings_list

    all_embeddings = asyncio.run(get_embeddings_list())

    async def mixed_workload():
        # Create ingestion tasks
        ingestion_tasks = [parser.parse(f) for f in files]

        # Create search tasks
        search_tasks = [
            engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=embeddings,
                limit=10,
                fusion_method="rrf",
            )
            for embeddings in all_embeddings
        ]

        # Run all tasks concurrently
        all_tasks = ingestion_tasks + search_tasks
        results = await asyncio.gather(*all_tasks)
        return results

    def run_benchmark():
        return asyncio.run(mixed_workload())

    results = benchmark(run_benchmark)
    assert len(results) == 20

    throughput = ConcurrentBenchmarkMetrics.calculate_throughput(
        20, benchmark.stats.mean
    )
    print(f"\nMixed workload (10+10) - Throughput: {throughput:.2f} ops/sec")


# ============================================================================
# Contention Analysis Benchmarks
# ============================================================================


@pytest.mark.benchmark
def test_contention_analysis_sequential_baseline(benchmark, tmp_concurrent_dir):
    """Sequential baseline for contention analysis - 10 files parsed sequentially."""
    files = ConcurrentTestDataGenerator.generate_test_files(
        10, 20, ".txt", tmp_concurrent_dir
    )
    parser = TextParser()

    async def sequential_parse():
        results = []
        for f in files:
            result = await parser.parse(f)
            results.append(result)
        return results

    def run_benchmark():
        return asyncio.run(sequential_parse())

    results = benchmark(run_benchmark)
    assert len(results) == 10
    print(f"\nSequential baseline (10 files) - Time: {benchmark.stats.mean:.4f}s")


@pytest.mark.benchmark
def test_contention_analysis_concurrent_comparison(benchmark, tmp_concurrent_dir):
    """Concurrent comparison for contention analysis - 10 files parsed concurrently."""
    files = ConcurrentTestDataGenerator.generate_test_files(
        10, 20, ".txt", tmp_concurrent_dir
    )
    parser = TextParser()

    async def concurrent_parse():
        tasks = [parser.parse(f) for f in files]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_parse())

    results = benchmark(run_benchmark)
    assert len(results) == 10
    print(f"\nConcurrent (10 files) - Time: {benchmark.stats.mean:.4f}s")
    # Note: Compare this with sequential baseline to calculate contention factor


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_contention_search_sequential_baseline(
    benchmark, qdrant_client, concurrent_search_collection
):
    """Sequential search baseline for contention analysis."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    queries = [f"contention test query {i}" for i in range(10)]

    async def get_embeddings_list():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings_list = []
        for query in queries:
            embeddings = await embedding_service.generate_embeddings(query)
            embeddings_list.append(embeddings)
        await embedding_service.close()
        return embeddings_list

    all_embeddings = asyncio.run(get_embeddings_list())

    async def sequential_search():
        results = []
        for embeddings in all_embeddings:
            result = await engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=embeddings,
                limit=10,
                fusion_method="rrf",
            )
            results.append(result)
        return results

    def run_benchmark():
        return asyncio.run(sequential_search())

    results = benchmark(run_benchmark)
    assert len(results) == 10
    print(f"\nSequential search baseline (10 queries) - Time: {benchmark.stats.mean:.4f}s")


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_contention_search_concurrent_comparison(
    benchmark, qdrant_client, concurrent_search_collection
):
    """Concurrent search comparison for contention analysis."""
    engine = HybridSearchEngine(
        qdrant_client,
        enable_optimizations=False,
        enable_multi_tenant_aggregation=False,
        enable_performance_monitoring=False,
    )

    queries = [f"contention test query {i}" for i in range(10)]

    async def get_embeddings_list():
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
        embeddings_list = []
        for query in queries:
            embeddings = await embedding_service.generate_embeddings(query)
            embeddings_list.append(embeddings)
        await embedding_service.close()
        return embeddings_list

    all_embeddings = asyncio.run(get_embeddings_list())

    async def concurrent_search():
        tasks = [
            engine.hybrid_search(
                collection_name=concurrent_search_collection,
                query_embeddings=embeddings,
                limit=10,
                fusion_method="rrf",
            )
            for embeddings in all_embeddings
        ]
        results = await asyncio.gather(*tasks)
        return results

    def run_benchmark():
        return asyncio.run(concurrent_search())

    results = benchmark(run_benchmark)
    assert len(results) == 10
    print(f"\nConcurrent search (10 queries) - Time: {benchmark.stats.mean:.4f}s")
