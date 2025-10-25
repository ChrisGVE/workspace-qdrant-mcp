"""
Sample performance benchmark tests for workspace-qdrant-mcp.

These tests demonstrate benchmark testing patterns for:
- Vector search performance baselines
- API response time benchmarks
- Memory usage analysis
- Regression detection
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@pytest.mark.benchmark
@pytest.mark.performance
class TestVectorSearchBenchmarks:
    """Benchmark vector search operations for performance baselines."""

    @pytest.fixture(scope="class")
    def benchmark_collection(self, qdrant_client: QdrantClient) -> str:
        """Create a benchmark collection with standardized data."""
        collection_name = "benchmark_collection"

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        # Insert benchmark dataset
        vectors = []
        for i in range(1000):
            vector = np.random.random(384).astype(np.float32)
            point = PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={
                    "category": f"cat_{i % 10}",
                    "subcategory": f"sub_{i % 50}",
                    "index": i,
                    "score": float(i % 100) / 100.0
                }
            )
            vectors.append(point)

        # Insert in batches for better performance
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            qdrant_client.upsert(collection_name=collection_name, points=batch)

        yield collection_name

        # Cleanup
        qdrant_client.delete_collection(collection_name)

    def test_search_performance_baseline(self, benchmark, qdrant_client: QdrantClient, benchmark_collection: str):
        """Benchmark basic vector search performance."""
        search_vector = np.random.random(384).astype(np.float32)

        def search_operation():
            return qdrant_client.search(
                collection_name=benchmark_collection,
                query_vector=search_vector.tolist(),
                limit=10
            )

        # Benchmark the search operation
        result = benchmark(search_operation)

        # Validate results
        assert len(result) == 10
        assert all(isinstance(r.score, float) for r in result)
        assert all(r.score >= 0 for r in result)

        # Performance expectations (baseline)
        stats = benchmark.stats
        assert stats.min < 0.1, f"Search too slow: {stats.min:.3f}s"
        assert stats.mean < 0.05, f"Average search too slow: {stats.mean:.3f}s"

    def test_filtered_search_performance(self, benchmark, qdrant_client: QdrantClient, benchmark_collection: str):
        """Benchmark filtered vector search performance."""
        search_vector = np.random.random(384).astype(np.float32)

        def filtered_search():
            return qdrant_client.search(
                collection_name=benchmark_collection,
                query_vector=search_vector.tolist(),
                query_filter={
                    "must": [
                        {"key": "category", "match": {"value": "cat_1"}},
                        {"key": "score", "range": {"gte": 0.5}}
                    ]
                },
                limit=10
            )

        result = benchmark(filtered_search)

        # Validate filtered results
        for point in result:
            assert point.payload["category"] == "cat_1"
            assert point.payload["score"] >= 0.5

        # Performance expectations (should be reasonable despite filtering)
        stats = benchmark.stats
        assert stats.mean < 0.2, f"Filtered search too slow: {stats.mean:.3f}s"

    def test_batch_insert_performance(self, benchmark, qdrant_client: QdrantClient):
        """Benchmark batch insert performance."""
        collection_name = "batch_benchmark"

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        try:
            # Create batch of vectors
            batch_size = 100
            vectors = []
            for i in range(batch_size):
                vector = np.random.random(384).astype(np.float32)
                point = PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={"index": i}
                )
                vectors.append(point)

            def batch_insert():
                return qdrant_client.upsert(
                    collection_name=collection_name,
                    points=vectors
                )

            result = benchmark(batch_insert)
            assert result.status == "completed"

            # Performance expectations
            stats = benchmark.stats
            vectors_per_second = batch_size / stats.mean
            assert vectors_per_second > 500, f"Insert rate too slow: {vectors_per_second:.1f} vectors/sec"

        finally:
            qdrant_client.delete_collection(collection_name)

    @pytest.mark.parametrize("search_limit", [1, 10, 50, 100])
    def test_search_limit_scaling(self, benchmark, qdrant_client: QdrantClient, benchmark_collection: str, search_limit: int):
        """Benchmark how search performance scales with result limit."""
        search_vector = np.random.random(384).astype(np.float32)

        def search_with_limit():
            return qdrant_client.search(
                collection_name=benchmark_collection,
                query_vector=search_vector.tolist(),
                limit=search_limit
            )

        result = benchmark(search_with_limit)
        assert len(result) == min(search_limit, 1000)  # Collection has 1000 vectors

        # Tag benchmark with limit for comparison
        benchmark.extra_info["search_limit"] = search_limit


@pytest.mark.benchmark
@pytest.mark.api_testing
class TestAPIResponseBenchmarks:
    """Benchmark API response times for various operations."""

    @pytest.fixture(scope="class")
    def http_client(self) -> httpx.AsyncClient:
        """Provide HTTP client for API benchmarking."""
        return httpx.AsyncClient(
            base_url="http://localhost:8000",
            timeout=30.0
        )

    def test_mcp_initialization_benchmark(self, benchmark, http_client: httpx.AsyncClient):
        """Benchmark MCP initialization handshake performance."""
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "benchmark-client", "version": "1.0.0"}
            }
        }

        async def mcp_initialize():
            response = await http_client.post("/", json=initialize_request)
            return response.json()

        # Run async benchmark
        result = benchmark(asyncio.run, mcp_initialize())

        # Validate response
        assert "result" in result
        assert "protocolVersion" in result["result"]

        # Performance expectations
        stats = benchmark.stats
        assert stats.mean < 0.5, f"MCP init too slow: {stats.mean:.3f}s"

    def test_workspace_status_benchmark(self, benchmark, http_client: httpx.AsyncClient):
        """Benchmark workspace status API performance."""
        status_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "workspace_status",
            "params": {}
        }

        async def workspace_status():
            response = await http_client.post("/", json=status_request)
            return response.json()

        result = benchmark(asyncio.run, workspace_status())

        # Validate response structure
        assert "result" in result
        assert "content" in result["result"]

        # Performance expectations
        stats = benchmark.stats
        assert stats.mean < 1.0, f"Workspace status too slow: {stats.mean:.3f}s"

    @pytest.mark.parametrize("query_length", [10, 50, 100, 500])
    def test_search_query_scaling_benchmark(self, benchmark, http_client: httpx.AsyncClient, query_length: int):
        """Benchmark search performance with varying query lengths."""
        # Generate query of specified length
        query = "test search " * (query_length // 12 + 1)
        query = query[:query_length]

        search_request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "search_workspace",
            "params": {
                "query": query,
                "limit": 10
            }
        }

        async def search_operation():
            response = await http_client.post("/", json=search_request)
            return response.json()

        # Note: This might return an error if method not implemented
        # The benchmark will still measure the response time
        benchmark(asyncio.run, search_operation())

        # Tag with query length for analysis
        benchmark.extra_info["query_length"] = query_length

    def test_concurrent_request_benchmark(self, benchmark, http_client: httpx.AsyncClient):
        """Benchmark concurrent API request handling."""
        status_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "workspace_status",
            "params": {}
        }

        async def concurrent_requests():
            # Launch 5 concurrent requests
            tasks = []
            for i in range(5):
                request = {**status_request, "id": i + 1}
                task = http_client.post("/", json=request)
                tasks.append(task)

            responses = await asyncio.gather(*tasks)
            return [r.json() for r in responses]

        results = benchmark(asyncio.run, concurrent_requests())

        # Validate all requests succeeded
        assert len(results) == 5
        for result in results:
            assert "result" in result or "error" in result  # Either success or handled error

        # Performance expectations
        stats = benchmark.stats
        assert stats.mean < 2.0, f"Concurrent requests too slow: {stats.mean:.3f}s"


@pytest.mark.benchmark
@pytest.mark.performance
class TestMemoryUsageBenchmarks:
    """Benchmark memory usage patterns."""

    def test_memory_usage_during_vector_operations(self, benchmark, qdrant_client: QdrantClient):
        """Benchmark memory usage during vector operations."""
        collection_name = "memory_benchmark"

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        try:
            def memory_intensive_operation():
                # Track memory before operation
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss

                # Perform memory-intensive operation
                vectors = []
                for i in range(500):  # Large batch
                    vector = np.random.random(384).astype(np.float32)
                    point = PointStruct(
                        id=i,
                        vector=vector.tolist(),
                        payload={"index": i, "data": "x" * 100}  # Some payload data
                    )
                    vectors.append(point)

                # Insert vectors
                result = qdrant_client.upsert(collection_name=collection_name, points=vectors)

                # Measure memory after
                final_memory = process.memory_info().rss
                memory_delta = final_memory - initial_memory

                return {
                    "operation_result": result,
                    "memory_delta_mb": memory_delta / 1024 / 1024,
                    "initial_memory_mb": initial_memory / 1024 / 1024,
                    "final_memory_mb": final_memory / 1024 / 1024
                }

            result = benchmark(memory_intensive_operation)

            # Validate operation succeeded
            assert result["operation_result"].status == "completed"

            # Memory usage should be reasonable
            assert result["memory_delta_mb"] < 100, f"Excessive memory usage: {result['memory_delta_mb']:.1f}MB"

            # Store memory metrics in benchmark
            benchmark.extra_info.update({
                "memory_delta_mb": result["memory_delta_mb"],
                "initial_memory_mb": result["initial_memory_mb"],
                "final_memory_mb": result["final_memory_mb"]
            })

        finally:
            qdrant_client.delete_collection(collection_name)

    def test_search_memory_efficiency(self, benchmark, qdrant_client: QdrantClient, benchmark_collection: str):
        """Benchmark memory efficiency of search operations."""
        search_vector = np.random.random(384).astype(np.float32)

        def memory_tracked_search():
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Perform multiple searches to stress test
            results = []
            for _ in range(10):
                result = qdrant_client.search(
                    collection_name=benchmark_collection,
                    query_vector=search_vector.tolist(),
                    limit=50
                )
                results.append(result)

            final_memory = process.memory_info().rss
            memory_delta = final_memory - initial_memory

            return {
                "search_results": len(results),
                "total_results": sum(len(r) for r in results),
                "memory_delta_mb": memory_delta / 1024 / 1024
            }

        result = benchmark(memory_tracked_search)

        # Validate searches completed
        assert result["search_results"] == 10
        assert result["total_results"] == 500  # 10 searches × 50 results

        # Memory growth should be minimal for searches
        assert result["memory_delta_mb"] < 10, f"Search memory leak: {result['memory_delta_mb']:.1f}MB"

        benchmark.extra_info["memory_delta_mb"] = result["memory_delta_mb"]


@pytest.mark.benchmark
@pytest.mark.regression
class TestPerformanceRegressionDetection:
    """Tests for detecting performance regressions."""

    def test_baseline_performance_metrics(self, benchmark, qdrant_client: QdrantClient, benchmark_collection: str):
        """Establish baseline performance metrics for regression detection."""
        search_vector = np.random.random(384).astype(np.float32)

        def comprehensive_operation():
            # Search operation
            search_start = time.time()
            search_results = qdrant_client.search(
                collection_name=benchmark_collection,
                query_vector=search_vector.tolist(),
                limit=20
            )
            search_time = time.time() - search_start

            # Count operation
            count_start = time.time()
            count_result = qdrant_client.count(collection_name=benchmark_collection)
            count_time = time.time() - count_start

            # Retrieve operation
            retrieve_start = time.time()
            retrieve_results = qdrant_client.retrieve(
                collection_name=benchmark_collection,
                ids=[0, 1, 2, 3, 4]
            )
            retrieve_time = time.time() - retrieve_start

            return {
                "search_time": search_time,
                "search_results": len(search_results),
                "count_time": count_time,
                "count_result": count_result.count,
                "retrieve_time": retrieve_time,
                "retrieve_results": len(retrieve_results)
            }

        result = benchmark(comprehensive_operation)

        # Validate operations
        assert result["search_results"] == 20
        assert result["count_result"] == 1000
        assert result["retrieve_results"] == 5

        # Store detailed metrics for regression analysis
        benchmark.extra_info.update({
            "search_time": result["search_time"],
            "count_time": result["count_time"],
            "retrieve_time": result["retrieve_time"],
            "total_operation_time": result["search_time"] + result["count_time"] + result["retrieve_time"]
        })

        # Baseline expectations (these become regression thresholds)
        assert result["search_time"] < 0.1, "Search baseline exceeded"
        assert result["count_time"] < 0.05, "Count baseline exceeded"
        assert result["retrieve_time"] < 0.05, "Retrieve baseline exceeded"

    @pytest.mark.slow_functional
    def test_sustained_performance_stability(self, benchmark, qdrant_client: QdrantClient, benchmark_collection: str):
        """Test performance stability over sustained operations."""
        search_vector = np.random.random(384).astype(np.float32)

        def sustained_operations():
            times = []

            # Perform 50 operations and track timing
            for _i in range(50):
                start_time = time.time()

                qdrant_client.search(
                    collection_name=benchmark_collection,
                    query_vector=search_vector.tolist(),
                    limit=10
                )

                operation_time = time.time() - start_time
                times.append(operation_time)

            return {
                "min_time": min(times),
                "max_time": max(times),
                "avg_time": sum(times) / len(times),
                "time_variance": np.var(times),
                "operations": len(times)
            }

        result = benchmark(sustained_operations)

        # Validate sustained performance
        assert result["operations"] == 50
        assert result["max_time"] < result["avg_time"] * 3, "Performance too variable"
        assert result["time_variance"] < 0.01, "Performance too unstable"

        # Store variance metrics
        benchmark.extra_info.update({
            "min_time": result["min_time"],
            "max_time": result["max_time"],
            "avg_time": result["avg_time"],
            "time_variance": result["time_variance"]
        })


# Benchmark configuration and utilities
@pytest.fixture(autouse=True)
def benchmark_config(benchmark):
    """Configure benchmark settings."""
    # Only run 5 rounds to reduce test time
    benchmark.rounds = 5
    benchmark.min_rounds = 3
    benchmark.max_time = 10.0  # Maximum 10 seconds per benchmark
    benchmark.min_time = 0.1   # Minimum 0.1 seconds per benchmark


def pytest_benchmark_generate_json(config, benchmarks):
    """Generate custom benchmark report with performance metrics."""
    report_data = {
        "test_run": {
            "timestamp": time.time(),
            "test_count": len(benchmarks),
            "framework_version": "workspace-qdrant-mcp-benchmarks-1.0"
        },
        "benchmarks": []
    }

    for benchmark in benchmarks:
        benchmark_data = {
            "name": benchmark["name"],
            "stats": benchmark["stats"],
            "extra_info": benchmark.get("extra_info", {})
        }
        report_data["benchmarks"].append(benchmark_data)

    # Save to file for regression analysis
    report_file = Path("benchmark_report.json")
    with open(report_file, "w") as f:
        json.dump(report_data, f, indent=2)

    return report_data
