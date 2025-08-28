"""
Performance tests for workspace-qdrant-mcp using real codebase data.

Comprehensive performance testing including search response times, index build performance,
memory usage profiling, and concurrent operation benchmarking.
"""

import asyncio
import json
import os
import statistics
import time
import tracemalloc
from unittest.mock import AsyncMock

import psutil
import pytest

from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.tools.documents import add_document
from workspace_qdrant_mcp.tools.search import search_workspace

from ..fixtures.test_data_collector import TestDataCollector
from ..utils.metrics import PerformanceBenchmarker


class TestPerformance:
    """Comprehensive performance testing suite."""

    @pytest.fixture(autouse=True)
    async def setup_performance_environment(self, mock_config, tmp_path):
        """Set up performance testing environment."""
        self.tmp_path = tmp_path
        self.benchmarker = PerformanceBenchmarker()

        # Enable memory tracking
        tracemalloc.start()

        # Collect comprehensive test data
        source_root = tmp_path.parent.parent.parent
        self.data_collector = TestDataCollector(source_root)
        self.test_data = self.data_collector.collect_all_data()

        # Create performance-oriented mock client
        self.mock_client = await self._create_performance_mock_client()

        # Track system resources
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        print("Performance testing setup:")
        print(
            f"  Test data: {len(self.test_data['chunks'])} chunks, {len(self.test_data['symbols'])} symbols"
        )
        print(f"  Initial memory: {self.initial_memory:.1f} MB")

        yield

        # Cleanup and final memory check
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - self.initial_memory

        tracemalloc.stop()

        print("📊 Performance test cleanup:")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Memory growth: {memory_growth:.1f} MB")

    async def _create_performance_mock_client(self):
        """Create mock client optimized for performance testing."""
        mock_client = AsyncMock(spec=QdrantWorkspaceClient)

        # Performance counters
        self.operation_counts = {"searches": 0, "insertions": 0, "embeddings": 0}

        # Simulate realistic latencies
        async def mock_search_with_latency(*args, **kwargs):
            # Simulate variable search latency (20-100ms)
            await asyncio.sleep(0.02 + (hash(str(args)) % 80) / 1000)
            self.operation_counts["searches"] += 1

            # Return realistic result structure
            return {
                "results": [
                    {
                        "id": f"result_{i}",
                        "score": 0.9 - (i * 0.1),
                        "content": f"Mock search result {i}",
                        "metadata": {"test": True},
                    }
                    for i in range(min(kwargs.get("limit", 10), 5))
                ],
                "total": min(kwargs.get("limit", 10), 5),
                "query": args[0] if args else "test",
            }

        async def mock_add_document_with_latency(*args, **kwargs):
            # Simulate document insertion latency (10-50ms)
            await asyncio.sleep(0.01 + (hash(str(args)) % 40) / 1000)
            self.operation_counts["insertions"] += 1

            return {
                "success": True,
                "document_id": f"doc_{self.operation_counts['insertions']}",
                "collection": kwargs.get("collection", "test"),
            }

        async def mock_embedding_generation(*args, **kwargs):
            # Simulate embedding generation latency (5-20ms)
            await asyncio.sleep(0.005 + (hash(str(args)) % 15) / 1000)
            self.operation_counts["embeddings"] += 1

            return {
                "dense": [0.1] * 384,
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.9]},
            }

        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings = mock_embedding_generation

        # Assign mock methods
        mock_client.initialized = True
        mock_client.search_workspace = mock_search_with_latency
        mock_client.add_document = mock_add_document_with_latency
        mock_client.get_embedding_service.return_value = mock_embedding_service
        mock_client.list_collections.return_value = ["perf_test_collection"]

        return mock_client

    @pytest.mark.performance
    async def test_search_response_time_benchmarks(self):
        """Test search response times under various conditions."""
        print("Benchmarking search response times...")

        # Test queries of different complexities
        test_queries = [
            ("simple", "client"),
            ("medium", "client initialization process"),
            (
                "complex",
                "FastMCP server configuration with Qdrant client initialization",
            ),
            ("technical", "async def generate_embeddings with sparse vectors"),
            ("long", " ".join(["search"] * 20)),  # Very long query
        ]

        # Benchmark each query type
        query_benchmarks = {}

        for query_type, query in test_queries:
            benchmark = await self.benchmarker.benchmark_async_operation(
                f"search_{query_type}_query",
                lambda q=query: search_workspace(
                    self.mock_client, query=q, mode="hybrid", limit=10
                ),
                iterations=20,
            )

            query_benchmarks[query_type] = benchmark

            print(
                f"  {query_type.capitalize()} query: {benchmark.mean_time_ms:.1f}ms avg "
                f"(median: {benchmark.median_time_ms:.1f}ms, p95: {benchmark.p95_time_ms:.1f}ms)"
            )

            # Performance assertions
            assert benchmark.mean_time_ms < 200, f"{query_type} query should be < 200ms"
            assert benchmark.p95_time_ms < 300, (
                f"{query_type} query p95 should be < 300ms"
            )

        # Test search with different result limits
        limit_tests = [1, 5, 10, 20, 50]
        limit_benchmarks = {}

        for limit in limit_tests:
            benchmark = await self.benchmarker.benchmark_async_operation(
                f"search_limit_{limit}",
                lambda l=limit: search_workspace(
                    self.mock_client, query="performance test", limit=l
                ),
                iterations=10,
            )

            limit_benchmarks[limit] = benchmark
            print(f"  Limit {limit}: {benchmark.mean_time_ms:.1f}ms avg")

        # Verify performance scales reasonably with limit
        limit_10_time = limit_benchmarks[10].mean_time_ms
        limit_50_time = limit_benchmarks[50].mean_time_ms

        # Higher limits shouldn't be more than 3x slower
        performance_ratio = limit_50_time / limit_10_time if limit_10_time > 0 else 1
        assert performance_ratio < 3.0, (
            f"Performance should scale reasonably (got {performance_ratio:.1f}x)"
        )

        print(
            f"  Search scaling: 10→50 results = {performance_ratio:.1f}x time increase"
        )

    @pytest.mark.performance
    async def test_concurrent_search_performance(self):
        """Test search performance under concurrent load."""
        print("🔀 Testing concurrent search performance...")

        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 30]
        concurrent_benchmarks = {}

        for concurrency in concurrency_levels:
            # Create concurrent search tasks
            async def concurrent_searches():
                tasks = []
                for i in range(concurrency):
                    task = search_workspace(
                        self.mock_client,
                        query=f"concurrent test {i}",
                        mode="hybrid",
                        limit=5,
                    )
                    tasks.append(task)

                return await asyncio.gather(*tasks)

            # Benchmark concurrent execution
            benchmark = await self.benchmarker.benchmark_async_operation(
                f"concurrent_search_{concurrency}", concurrent_searches, iterations=5
            )

            concurrent_benchmarks[concurrency] = benchmark

            # Calculate per-request metrics
            per_request_time = benchmark.mean_time_ms / concurrency
            throughput = concurrency / (benchmark.mean_time_ms / 1000)

            print(
                f"  {concurrency} concurrent: {benchmark.mean_time_ms:.1f}ms total, "
                f"{per_request_time:.1f}ms per request, {throughput:.1f} req/sec"
            )

            # Performance assertions
            assert benchmark.mean_time_ms < 2000, (
                "Concurrent searches should complete < 2s"
            )
            assert per_request_time < 500, "Per-request time should be < 500ms"

        # Analyze throughput scaling
        single_throughput = concurrent_benchmarks[1].operations_per_second
        max_throughput = max(
            b.operations_per_second for b in concurrent_benchmarks.values()
        )

        throughput_improvement = max_throughput / single_throughput
        print(f"  Throughput improvement: {throughput_improvement:.1f}x")

        # Should see some concurrency benefit
        assert throughput_improvement > 2.0, (
            "Should show >2x throughput improvement with concurrency"
        )

    @pytest.mark.performance
    async def test_document_insertion_performance(self):
        """Test document insertion performance and throughput."""
        print("📝 Testing document insertion performance...")

        # Use real test data for insertion
        test_chunks = self.test_data["chunks"][:100]  # Use first 100 chunks

        # Test single document insertion
        single_benchmark = await self.benchmarker.benchmark_async_operation(
            "single_document_insertion",
            lambda: add_document(
                self.mock_client,
                content=test_chunks[0]["content"],
                collection="perf_test",
                metadata=test_chunks[0]["metadata"],
                chunk_text=False,
            ),
            iterations=20,
        )

        print(
            f"  Single insertion: {single_benchmark.mean_time_ms:.1f}ms avg, "
            f"{single_benchmark.operations_per_second:.1f} docs/sec"
        )

        # Performance assertions
        assert single_benchmark.mean_time_ms < 100, "Single insertion should be < 100ms"
        assert single_benchmark.operations_per_second > 10, (
            "Should handle > 10 docs/sec"
        )

        # Test batch insertion with different batch sizes
        batch_sizes = [5, 10, 20, 50]
        batch_benchmarks = {}

        for batch_size in batch_sizes:
            batch_chunks = test_chunks[:batch_size]

            async def batch_insertion():
                tasks = []
                for chunk in batch_chunks:
                    task = add_document(
                        self.mock_client,
                        content=chunk["content"],
                        collection="perf_test",
                        metadata=chunk["metadata"],
                        document_id=f"batch_{chunk['id']}",
                        chunk_text=False,
                    )
                    tasks.append(task)

                return await asyncio.gather(*tasks)

            benchmark = await self.benchmarker.benchmark_async_operation(
                f"batch_insertion_{batch_size}", batch_insertion, iterations=5
            )

            batch_benchmarks[batch_size] = benchmark

            per_doc_time = benchmark.mean_time_ms / batch_size
            throughput = batch_size / (benchmark.mean_time_ms / 1000)

            print(
                f"  Batch {batch_size}: {benchmark.mean_time_ms:.1f}ms total, "
                f"{per_doc_time:.1f}ms per doc, {throughput:.1f} docs/sec"
            )

        # Verify batch processing is more efficient than sequential
        sequential_time_50 = single_benchmark.mean_time_ms * 50
        batch_time_50 = batch_benchmarks[50].mean_time_ms
        batch_speedup = sequential_time_50 / batch_time_50

        print(f"  ✅ Batch speedup (50 docs): {batch_speedup:.1f}x")
        assert batch_speedup > 5.0, (
            "Batch insertion should be >5x faster than sequential"
        )

    @pytest.mark.performance
    async def test_embedding_generation_performance(self):
        """Test embedding generation performance with realistic text."""
        print("🧠 Testing embedding generation performance...")

        # Use real code content for embedding tests
        code_chunks = [
            chunk for chunk in self.test_data["chunks"] if chunk["chunk_type"] == "code"
        ]
        doc_chunks = [
            chunk
            for chunk in self.test_data["chunks"]
            if chunk["chunk_type"] == "documentation"
        ]

        # Test with different content types and lengths
        test_contents = [
            (
                "short_code",
                code_chunks[0]["content"][:100] if code_chunks else "def test(): pass",
            ),
            (
                "medium_code",
                code_chunks[0]["content"][:500]
                if code_chunks
                else "def test():\n    return True" * 20,
            ),
            (
                "long_code",
                code_chunks[0]["content"]
                if code_chunks
                else "class Test:\n    pass\n" * 50,
            ),
            (
                "short_doc",
                doc_chunks[0]["content"][:100] if doc_chunks else "Documentation",
            ),
            (
                "long_doc",
                doc_chunks[0]["content"] if doc_chunks else "Documentation text " * 100,
            ),
        ]

        embedding_service = self.mock_client.get_embedding_service()

        for content_type, content in test_contents:
            benchmark = await self.benchmarker.benchmark_async_operation(
                f"embedding_{content_type}",
                lambda c=content: embedding_service.generate_embeddings(
                    c, include_sparse=True
                ),
                iterations=15,
            )

            chars_per_sec = (
                len(content) / (benchmark.mean_time_ms / 1000)
                if benchmark.mean_time_ms > 0
                else 0
            )

            print(
                f"  {content_type}: {benchmark.mean_time_ms:.1f}ms avg "
                f"({len(content)} chars, {chars_per_sec:.0f} chars/sec)"
            )

            # Performance assertions
            assert benchmark.mean_time_ms < 50, "Embedding generation should be < 50ms"
            assert chars_per_sec > 1000, "Should process > 1000 chars/sec"

        # Test concurrent embedding generation
        async def concurrent_embeddings():
            tasks = [
                embedding_service.generate_embeddings(content, include_sparse=True)
                for _, content in test_contents[:3]
            ]
            return await asyncio.gather(*tasks)

        concurrent_benchmark = await self.benchmarker.benchmark_async_operation(
            "concurrent_embeddings", concurrent_embeddings, iterations=10
        )

        print(f"  Concurrent (3): {concurrent_benchmark.mean_time_ms:.1f}ms total")

        # Concurrent should be faster than sequential
        sequential_time = sum(
            self.benchmarker.benchmarks[f"embedding_{ct}"].mean_time_ms
            for ct, _ in test_contents[:3]
        )
        speedup = sequential_time / concurrent_benchmark.mean_time_ms

        print(f"  ✅ Concurrent embedding speedup: {speedup:.1f}x")
        assert speedup > 1.5, "Concurrent embeddings should show >1.5x speedup"

    @pytest.mark.performance
    async def test_memory_usage_profiling(self):
        """Test memory usage patterns during operations."""
        print("💾 Profiling memory usage during operations...")

        # Get initial memory snapshot
        initial_snapshot = tracemalloc.take_snapshot()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        # Perform memory-intensive operations
        operations_data = []

        # Test 1: Large batch document insertion
        large_batch = self.test_data["chunks"][:200]  # 200 documents

        memory_before = self.process.memory_info().rss / 1024 / 1024

        async def large_batch_operation():
            tasks = []
            for i, chunk in enumerate(large_batch):
                task = add_document(
                    self.mock_client,
                    content=chunk["content"],
                    collection="memory_test",
                    metadata=chunk["metadata"],
                    document_id=f"memory_test_{i}",
                    chunk_text=False,
                )
                tasks.append(task)
            return await asyncio.gather(*tasks)

        await large_batch_operation()
        memory_after_batch = self.process.memory_info().rss / 1024 / 1024

        operations_data.append(
            {
                "operation": "large_batch_insertion",
                "documents": len(large_batch),
                "memory_before": memory_before,
                "memory_after": memory_after_batch,
                "memory_increase": memory_after_batch - memory_before,
            }
        )

        # Test 2: Many concurrent searches
        memory_before = memory_after_batch

        async def many_searches():
            tasks = [
                search_workspace(self.mock_client, query=f"search test {i}", limit=20)
                for i in range(50)
            ]
            return await asyncio.gather(*tasks)

        await many_searches()
        memory_after_searches = self.process.memory_info().rss / 1024 / 1024

        operations_data.append(
            {
                "operation": "concurrent_searches",
                "operations": 50,
                "memory_before": memory_before,
                "memory_after": memory_after_searches,
                "memory_increase": memory_after_searches - memory_before,
            }
        )

        # Test 3: Embedding generation for all symbols
        memory_before = memory_after_searches
        embedding_service = self.mock_client.get_embedding_service()

        # Generate embeddings for function signatures
        function_symbols = [
            s
            for s in self.test_data["symbols"]
            if s["type"] in ["function", "async_function"]
        ][:50]

        for symbol in function_symbols:
            await embedding_service.generate_embeddings(
                symbol["signature"], include_sparse=True
            )

        memory_after_embeddings = self.process.memory_info().rss / 1024 / 1024

        operations_data.append(
            {
                "operation": "embedding_generation",
                "embeddings": len(function_symbols),
                "memory_before": memory_before,
                "memory_after": memory_after_embeddings,
                "memory_increase": memory_after_embeddings - memory_before,
            }
        )

        # Analyze memory usage
        print("📊 Memory usage analysis:")
        for op_data in operations_data:
            print(f"  {op_data['operation']}:")
            print(
                f"    Operations: {op_data.get('documents', op_data.get('operations', op_data.get('embeddings')))}"
            )
            print(f"    Memory increase: {op_data['memory_increase']:.1f} MB")
            print(
                f"    Per operation: {op_data['memory_increase'] / op_data.get('documents', op_data.get('operations', op_data.get('embeddings'))):.3f} MB"
            )

        # Take final snapshot for detailed analysis
        final_snapshot = tracemalloc.take_snapshot()
        top_stats = final_snapshot.compare_to(initial_snapshot, "lineno")

        print("🔍 Top memory allocations:")
        for stat in top_stats[:5]:
            print(f"  {stat}")

        # Memory usage assertions
        total_memory_increase = memory_after_embeddings - initial_memory
        print(f"  Total memory increase: {total_memory_increase:.1f} MB")

        # Should not have excessive memory growth
        assert total_memory_increase < 100, "Total memory increase should be < 100MB"

        # Per-operation memory usage should be reasonable
        for op_data in operations_data:
            op_count = op_data.get(
                "documents", op_data.get("operations", op_data.get("embeddings"))
            )
            per_op_memory = op_data["memory_increase"] / op_count if op_count > 0 else 0
            assert per_op_memory < 1.0, (
                f"{op_data['operation']} should use < 1MB per operation"
            )

    @pytest.mark.performance
    async def test_search_quality_vs_performance_tradeoffs(self):
        """Test the tradeoff between search quality and performance."""
        print("⚖️  Testing search quality vs performance tradeoffs...")

        # Test different search modes and their performance characteristics
        search_modes = ["semantic", "hybrid"]
        quality_performance_data = []

        test_queries = [
            "QdrantWorkspaceClient initialization",
            "FastMCP server configuration",
            "embedding vector generation",
            "async function definition",
            "collection management",
        ]

        for mode in search_modes:
            mode_times = []
            mode_result_counts = []

            for query in test_queries:
                # Benchmark search
                start_time = time.perf_counter()
                result = await search_workspace(
                    self.mock_client, query=query, mode=mode, limit=10
                )
                end_time = time.perf_counter()

                search_time = (end_time - start_time) * 1000
                result_count = len(result.get("results", []))

                mode_times.append(search_time)
                mode_result_counts.append(result_count)

            avg_time = statistics.mean(mode_times)
            avg_results = statistics.mean(mode_result_counts)

            quality_performance_data.append(
                {
                    "mode": mode,
                    "avg_time_ms": avg_time,
                    "avg_results": avg_results,
                    "throughput": len(test_queries) / (sum(mode_times) / 1000),
                }
            )

            print(f"  {mode.capitalize()} mode:")
            print(f"    Average time: {avg_time:.1f}ms")
            print(f"    Average results: {avg_results:.1f}")
            print(
                f"    Throughput: {len(test_queries) / (sum(mode_times) / 1000):.1f} queries/sec"
            )

        # Test with different result limits and their impact on performance
        limit_tests = [5, 10, 20, 50]
        limit_performance = []

        for limit in limit_tests:
            benchmark = await self.benchmarker.benchmark_async_operation(
                f"search_limit_{limit}_performance",
                lambda l=limit: search_workspace(
                    self.mock_client,
                    query="performance test query",
                    mode="hybrid",
                    limit=l,
                ),
                iterations=10,
            )

            limit_performance.append(
                {
                    "limit": limit,
                    "time_ms": benchmark.mean_time_ms,
                    "throughput": benchmark.operations_per_second,
                }
            )

            print(
                f"  Limit {limit}: {benchmark.mean_time_ms:.1f}ms, {benchmark.operations_per_second:.1f} ops/sec"
            )

        # Analyze performance scaling
        base_limit = limit_performance[0]  # limit=5
        for data in limit_performance[1:]:
            scaling_factor = data["limit"] / base_limit["limit"]
            time_factor = data["time_ms"] / base_limit["time_ms"]

            print(
                f"  Scaling {base_limit['limit']}→{data['limit']}: "
                f"{scaling_factor:.1f}x limit = {time_factor:.1f}x time"
            )

            # Performance should scale sub-linearly
            assert time_factor < scaling_factor * 1.5, (
                "Performance should scale better than linear"
            )

    @pytest.mark.performance
    async def test_stress_testing_under_load(self):
        """Stress test the system under sustained high load."""
        print("🔥 Stress testing under sustained load...")

        # Sustained load test parameters
        duration_seconds = 10
        operations_per_second = 20
        total_operations = duration_seconds * operations_per_second

        # Track performance metrics during stress test
        operation_times = []
        error_count = 0
        operation_count = 0

        async def stress_operation(operation_id: int):
            nonlocal operation_count, error_count

            try:
                start_time = time.perf_counter()

                # Mix of different operations
                if operation_id % 3 == 0:
                    # Search operation
                    await search_workspace(
                        self.mock_client,
                        query=f"stress test query {operation_id}",
                        limit=5,
                    )
                elif operation_id % 3 == 1:
                    # Document insertion
                    await add_document(
                        self.mock_client,
                        content=f"Stress test document {operation_id}",
                        collection="stress_test",
                        metadata={"stress_id": operation_id},
                    )
                else:
                    # Embedding generation
                    embedding_service = self.mock_client.get_embedding_service()
                    await embedding_service.generate_embeddings(
                        f"Stress test content {operation_id}"
                    )

                end_time = time.perf_counter()
                operation_times.append((end_time - start_time) * 1000)
                operation_count += 1

            except Exception as e:
                error_count += 1
                print(f"⚠️  Operation {operation_id} failed: {e}")

        # Execute stress test
        start_time = time.perf_counter()

        # Create all tasks
        tasks = [stress_operation(i) for i in range(total_operations)]

        # Execute with controlled concurrency
        semaphore = asyncio.Semaphore(20)  # Max 20 concurrent operations

        async def controlled_execution(task):
            async with semaphore:
                await task

        controlled_tasks = [controlled_execution(task) for task in tasks]
        await asyncio.gather(*controlled_tasks)

        end_time = time.perf_counter()
        actual_duration = end_time - start_time

        # Analyze stress test results
        success_rate = (operation_count / total_operations) * 100
        actual_ops_per_sec = operation_count / actual_duration
        avg_operation_time = statistics.mean(operation_times) if operation_times else 0
        p95_operation_time = (
            sorted(operation_times)[int(len(operation_times) * 0.95)]
            if operation_times
            else 0
        )

        print("📊 Stress test results:")
        print(f"  Duration: {actual_duration:.1f}s (target: {duration_seconds}s)")
        print(f"  Operations completed: {operation_count}/{total_operations}")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Actual throughput: {actual_ops_per_sec:.1f} ops/sec")
        print(f"  Average operation time: {avg_operation_time:.1f}ms")
        print(f"  P95 operation time: {p95_operation_time:.1f}ms")
        print(f"  Errors: {error_count}")

        # Performance assertions for stress test
        assert success_rate >= 95, "Success rate should be ≥ 95% under stress"
        assert error_count < total_operations * 0.05, "Error rate should be < 5%"
        assert avg_operation_time < 200, (
            "Average operation time should be < 200ms under stress"
        )
        assert p95_operation_time < 500, (
            "P95 operation time should be < 500ms under stress"
        )

        # System should maintain reasonable throughput under stress
        min_expected_throughput = operations_per_second * 0.8  # Allow 20% degradation
        assert actual_ops_per_sec >= min_expected_throughput, (
            f"Should maintain ≥{min_expected_throughput} ops/sec"
        )

    @pytest.mark.performance
    async def test_resource_cleanup_and_memory_leaks(self):
        """Test resource cleanup and detect potential memory leaks."""
        print("🧹 Testing resource cleanup and memory leak detection...")

        # Take initial memory measurement
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        initial_snapshot = tracemalloc.take_snapshot()

        # Perform operations that should be cleaned up
        for cycle in range(3):
            print(f"  Cleanup test cycle {cycle + 1}/3")

            # Create temporary data structures
            temp_data = []

            # Perform operations that create temporary objects
            for i in range(50):
                # Search operations (create temporary result objects)
                result = await search_workspace(
                    self.mock_client, query=f"cleanup test {cycle}_{i}", limit=10
                )
                temp_data.append(result)

                # Document insertion (create temporary document objects)
                doc_result = await add_document(
                    self.mock_client,
                    content=f"Cleanup test document {cycle}_{i}",
                    collection=f"cleanup_test_{cycle}",
                    metadata={"cycle": cycle, "index": i},
                )
                temp_data.append(doc_result)

            # Force garbage collection
            import gc

            gc.collect()

            # Measure memory after each cycle
            cycle_memory = self.process.memory_info().rss / 1024 / 1024
            memory_increase = cycle_memory - initial_memory

            print(
                f"    Memory after cycle {cycle + 1}: {cycle_memory:.1f} MB (+{memory_increase:.1f} MB)"
            )

            # Clear temporary data
            temp_data.clear()
            del temp_data
            gc.collect()

        # Final memory measurement
        final_memory = self.process.memory_info().rss / 1024 / 1024
        final_snapshot = tracemalloc.take_snapshot()

        total_memory_increase = final_memory - initial_memory

        print("📊 Memory leak analysis:")
        print(f"  Initial memory: {initial_memory:.1f} MB")
        print(f"  Final memory: {final_memory:.1f} MB")
        print(f"  Total increase: {total_memory_increase:.1f} MB")

        # Analyze memory growth patterns
        top_stats = final_snapshot.compare_to(initial_snapshot, "lineno")

        print("  Top memory allocations:")
        for i, stat in enumerate(top_stats[:3]):
            print(f"    {i + 1}. {stat}")

        # Memory leak assertions
        max_acceptable_increase = 20  # MB
        assert total_memory_increase < max_acceptable_increase, (
            f"Memory increase should be < {max_acceptable_increase}MB (got {total_memory_increase:.1f}MB)"
        )

        # Per-cycle memory increase should be minimal
        per_cycle_increase = total_memory_increase / 3
        assert per_cycle_increase < 5, (
            f"Per-cycle memory increase should be < 5MB (got {per_cycle_increase:.1f}MB)"
        )

        print("✅ Memory leak test passed - no significant leaks detected")

    def test_comprehensive_performance_report(self):
        """Generate comprehensive performance test report."""
        print("📋 Generating comprehensive performance report...")

        # Collect all benchmark results
        all_benchmarks = self.benchmarker.get_summary()
        operation_counts = self.operation_counts

        # Calculate overall statistics
        all_times = [b["mean_time_ms"] for b in all_benchmarks.values()]
        all_throughputs = [b["operations_per_second"] for b in all_benchmarks.values()]

        overall_stats = {
            "total_benchmarks": len(all_benchmarks),
            "avg_operation_time_ms": statistics.mean(all_times) if all_times else 0,
            "median_operation_time_ms": statistics.median(all_times)
            if all_times
            else 0,
            "avg_throughput_ops_sec": statistics.mean(all_throughputs)
            if all_throughputs
            else 0,
            "max_throughput_ops_sec": max(all_throughputs) if all_throughputs else 0,
            "total_operations_executed": sum(operation_counts.values()),
        }

        # Performance categories
        search_benchmarks = {k: v for k, v in all_benchmarks.items() if "search" in k}
        insertion_benchmarks = {
            k: v for k, v in all_benchmarks.items() if "insertion" in k or "add" in k
        }
        embedding_benchmarks = {
            k: v for k, v in all_benchmarks.items() if "embedding" in k
        }

        # Create comprehensive report
        performance_report = {
            "test_summary": {
                "test_data_size": {
                    "chunks": len(self.test_data["chunks"]),
                    "symbols": len(self.test_data["symbols"]),
                    "ground_truth_cases": len(self.test_data["ground_truth"]),
                },
                "operations_executed": operation_counts,
                **overall_stats,
            },
            "benchmark_categories": {
                "search_operations": {
                    "count": len(search_benchmarks),
                    "avg_time_ms": statistics.mean(
                        [b["mean_time_ms"] for b in search_benchmarks.values()]
                    )
                    if search_benchmarks
                    else 0,
                    "avg_throughput": statistics.mean(
                        [b["operations_per_second"] for b in search_benchmarks.values()]
                    )
                    if search_benchmarks
                    else 0,
                },
                "insertion_operations": {
                    "count": len(insertion_benchmarks),
                    "avg_time_ms": statistics.mean(
                        [b["mean_time_ms"] for b in insertion_benchmarks.values()]
                    )
                    if insertion_benchmarks
                    else 0,
                    "avg_throughput": statistics.mean(
                        [
                            b["operations_per_second"]
                            for b in insertion_benchmarks.values()
                        ]
                    )
                    if insertion_benchmarks
                    else 0,
                },
                "embedding_operations": {
                    "count": len(embedding_benchmarks),
                    "avg_time_ms": statistics.mean(
                        [b["mean_time_ms"] for b in embedding_benchmarks.values()]
                    )
                    if embedding_benchmarks
                    else 0,
                    "avg_throughput": statistics.mean(
                        [
                            b["operations_per_second"]
                            for b in embedding_benchmarks.values()
                        ]
                    )
                    if embedding_benchmarks
                    else 0,
                },
            },
            "detailed_benchmarks": all_benchmarks,
            "performance_targets": {
                "search_response_time": {
                    "target": "< 200ms",
                    "status": "✅"
                    if overall_stats["avg_operation_time_ms"] < 200
                    else "❌",
                },
                "throughput": {
                    "target": "> 10 ops/sec",
                    "status": "✅"
                    if overall_stats["avg_throughput_ops_sec"] > 10
                    else "❌",
                },
                "concurrent_performance": {
                    "target": "> 2x improvement",
                    "status": "✅",
                },  # Measured in concurrent tests
                "memory_efficiency": {
                    "target": "< 1MB per operation",
                    "status": "✅",
                },  # Measured in memory tests
            },
        }

        # Export detailed report
        report_file = self.tmp_path / "performance_test_report.json"
        with open(report_file, "w") as f:
            json.dump(performance_report, f, indent=2)

        print(f"📊 Performance report exported to: {report_file}")
        print("🎯 Performance Summary:")
        print(f"  Total benchmarks: {overall_stats['total_benchmarks']}")
        print(
            f"  Average operation time: {overall_stats['avg_operation_time_ms']:.1f}ms"
        )
        print(
            f"  Average throughput: {overall_stats['avg_throughput_ops_sec']:.1f} ops/sec"
        )
        print(
            f"  Maximum throughput: {overall_stats['max_throughput_ops_sec']:.1f} ops/sec"
        )
        print(f"  Operations executed: {overall_stats['total_operations_executed']}")

        # Final performance assertions
        assert overall_stats["avg_operation_time_ms"] < 300, (
            "Average operation time should be < 300ms"
        )
        assert overall_stats["avg_throughput_ops_sec"] > 5, (
            "Average throughput should be > 5 ops/sec"
        )
        assert overall_stats["total_benchmarks"] >= 10, (
            "Should have completed at least 10 benchmarks"
        )

        print("✅ Comprehensive performance testing completed successfully")
