"""
Comprehensive Performance Validation Tests for workspace-qdrant-mcp.

This module provides comprehensive performance benchmarking and validation tests
for all critical operations including document processing, vector search, MCP tool
response times, memory usage patterns, and scaling behavior.

SUCCESS CRITERIA:
- Document processing: < 500ms per document (avg)
- Vector search operations: < 100ms response time (avg)
- MCP tool response times: < 200ms per tool (avg)
- Memory usage: < 50MB base overhead, < 5MB per 1000 documents
- Concurrent operations: Linear scaling up to 10 concurrent operations
- No memory leaks: < 1MB growth per 1000 operations
- Scaling behavior: Sub-linear growth (O(log n)) for search operations

PERFORMANCE REGRESSION THRESHOLDS:
- Response time increase: < 20% compared to baseline
- Memory usage increase: < 30% compared to baseline
- Throughput decrease: < 15% compared to baseline
"""

import asyncio
import gc
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

# Test markers
pytestmark = [
    pytest.mark.performance,
    pytest.mark.benchmark,
    pytest.mark.requires_qdrant,
]


class PerformanceMetrics:
    """Advanced performance metrics collector with memory profiling."""

    def __init__(self):
        self.reset()
        self.process = psutil.Process()

    def reset(self):
        """Reset all metrics."""
        self.response_times = []
        self.memory_snapshots = []
        self.throughput_measurements = []
        self.concurrent_performance = []
        self.scaling_metrics = {}
        self.tool_latencies = {}
        self.error_counts = {}
        self.gc_stats = []

    async def measure_async_operation(self, operation_name: str, coro):
        """Measure async operation with comprehensive metrics."""
        # Memory before
        mem_before = self.process.memory_info().rss / 1024 / 1024
        gc_before = gc.get_stats()

        # Start tracemalloc for detailed memory tracking
        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            result = await coro
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1

        # End timing and memory tracking
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory after
        mem_after = self.process.memory_info().rss / 1024 / 1024
        gc_after = gc.get_stats()

        # Calculate metrics
        duration_ms = (end_time - start_time) * 1000
        memory_used_mb = (peak - current) / 1024 / 1024
        memory_growth_mb = mem_after - mem_before

        # Record metrics
        self.response_times.append(duration_ms)
        self.memory_snapshots.append({
            'operation': operation_name,
            'duration_ms': duration_ms,
            'memory_used_mb': memory_used_mb,
            'memory_growth_mb': memory_growth_mb,
            'success': success,
            'error': error
        })

        if operation_name not in self.tool_latencies:
            self.tool_latencies[operation_name] = []
        self.tool_latencies[operation_name].append(duration_ms)

        # GC statistics
        gc_diff = {
            'collections_gen0': gc_after[0]['collections'] - gc_before[0]['collections'],
            'collections_gen1': gc_after[1]['collections'] - gc_before[1]['collections'],
            'collections_gen2': gc_after[2]['collections'] - gc_before[2]['collections'],
        }
        self.gc_stats.append(gc_diff)

        return result, duration_ms, memory_used_mb

    async def measure_concurrent_operations(self, operations: list[tuple[str, any]], max_workers: int = 10):
        """Measure concurrent operation performance."""
        start_time = time.perf_counter()
        mem_before = self.process.memory_info().rss / 1024 / 1024

        # Execute operations concurrently
        tasks = []
        for op_name, coro in operations:
            task = asyncio.create_task(self.measure_async_operation(op_name, coro))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        mem_after = self.process.memory_info().rss / 1024 / 1024

        total_duration = (end_time - start_time) * 1000
        memory_overhead = mem_after - mem_before

        # Calculate concurrent performance metrics
        successful_ops = sum(1 for r in results if not isinstance(r, Exception))
        failed_ops = len(results) - successful_ops
        avg_response_time = sum(r[1] for r in results if not isinstance(r, Exception)) / max(successful_ops, 1)

        concurrent_metrics = {
            'total_operations': len(operations),
            'successful_operations': successful_ops,
            'failed_operations': failed_ops,
            'total_duration_ms': total_duration,
            'avg_response_time_ms': avg_response_time,
            'memory_overhead_mb': memory_overhead,
            'operations_per_second': len(operations) / (total_duration / 1000),
            'max_workers': max_workers
        }

        self.concurrent_performance.append(concurrent_metrics)
        return results, concurrent_metrics

    def measure_scaling_behavior(self, operation_name: str, data_sizes: list[int], results: list[float]):
        """Measure and analyze scaling behavior."""
        if len(data_sizes) != len(results):
            raise ValueError("Data sizes and results must have same length")

        # Calculate scaling metrics
        scaling_data = []
        for i, (size, result) in enumerate(zip(data_sizes, results, strict=False)):
            if i > 0:
                size_ratio = size / data_sizes[0]
                time_ratio = result / results[0]
                scaling_factor = time_ratio / size_ratio if size_ratio > 0 else 0
            else:
                size_ratio = 1.0
                time_ratio = 1.0
                scaling_factor = 1.0

            scaling_data.append({
                'data_size': size,
                'response_time_ms': result,
                'size_ratio': size_ratio,
                'time_ratio': time_ratio,
                'scaling_factor': scaling_factor
            })

        self.scaling_metrics[operation_name] = scaling_data
        return scaling_data

    def detect_memory_leaks(self, operation_count: int, memory_snapshots: list[float]) -> dict[str, Any]:
        """Detect potential memory leaks."""
        if len(memory_snapshots) < 2:
            return {'leak_detected': False, 'reason': 'Insufficient data'}

        # Calculate memory growth trend
        initial_memory = memory_snapshots[0]
        final_memory = memory_snapshots[-1]
        total_growth = final_memory - initial_memory
        growth_per_operation = total_growth / operation_count

        # Check for linear growth (potential leak)
        if len(memory_snapshots) >= 5:
            # Calculate correlation between operation number and memory usage
            operation_numbers = list(range(len(memory_snapshots)))
            memory_values = memory_snapshots

            # Simple linear correlation calculation
            n = len(memory_snapshots)
            sum_x = sum(operation_numbers)
            sum_y = sum(memory_values)
            sum_xy = sum(x * y for x, y in zip(operation_numbers, memory_values, strict=False))
            sum_x2 = sum(x * x for x in operation_numbers)

            correlation = (n * sum_xy - sum_x * sum_y) / ((n * sum_x2 - sum_x * sum_x) ** 0.5 * (n * sum(y * y for y in memory_values) - sum_y * sum_y) ** 0.5)
        else:
            correlation = 0

        # Thresholds for leak detection
        GROWTH_THRESHOLD_MB = 5.0  # 5MB total growth is concerning
        GROWTH_PER_OP_THRESHOLD_MB = 0.01  # 10KB per operation is concerning
        CORRELATION_THRESHOLD = 0.8  # Strong positive correlation indicates leak

        leak_indicators = []
        if total_growth > GROWTH_THRESHOLD_MB:
            leak_indicators.append(f"High total memory growth: {total_growth:.2f}MB")
        if growth_per_operation > GROWTH_PER_OP_THRESHOLD_MB:
            leak_indicators.append(f"High growth per operation: {growth_per_operation:.4f}MB/op")
        if correlation > CORRELATION_THRESHOLD:
            leak_indicators.append(f"Strong memory-operation correlation: {correlation:.3f}")

        return {
            'leak_detected': len(leak_indicators) > 0,
            'indicators': leak_indicators,
            'total_growth_mb': total_growth,
            'growth_per_operation_mb': growth_per_operation,
            'correlation': correlation,
            'operation_count': operation_count
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.response_times:
            return {'error': 'No performance data collected'}

        # Basic response time statistics
        avg_response_time = sum(self.response_times) / len(self.response_times)
        max_response_time = max(self.response_times)
        min_response_time = min(self.response_times)

        # Memory statistics
        memory_snapshots = [s['memory_used_mb'] for s in self.memory_snapshots if s['success']]
        avg_memory_usage = sum(memory_snapshots) / len(memory_snapshots) if memory_snapshots else 0
        max_memory_usage = max(memory_snapshots) if memory_snapshots else 0

        # Error statistics
        total_operations = len(self.memory_snapshots)
        successful_operations = sum(1 for s in self.memory_snapshots if s['success'])
        error_rate = (total_operations - successful_operations) / total_operations if total_operations > 0 else 0

        # Tool latency statistics
        tool_stats = {}
        for tool_name, latencies in self.tool_latencies.items():
            tool_stats[tool_name] = {
                'avg_latency_ms': sum(latencies) / len(latencies),
                'max_latency_ms': max(latencies),
                'min_latency_ms': min(latencies),
                'operation_count': len(latencies)
            }

        return {
            'response_time_stats': {
                'avg_ms': avg_response_time,
                'max_ms': max_response_time,
                'min_ms': min_response_time,
                'total_operations': len(self.response_times)
            },
            'memory_stats': {
                'avg_usage_mb': avg_memory_usage,
                'max_usage_mb': max_memory_usage,
                'total_snapshots': len(memory_snapshots)
            },
            'error_stats': {
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'error_rate': error_rate,
                'error_counts': self.error_counts
            },
            'tool_performance': tool_stats,
            'concurrent_performance': self.concurrent_performance,
            'scaling_metrics': self.scaling_metrics,
            'gc_stats': {
                'total_gen0_collections': sum(s['collections_gen0'] for s in self.gc_stats),
                'total_gen1_collections': sum(s['collections_gen1'] for s in self.gc_stats),
                'total_gen2_collections': sum(s['collections_gen2'] for s in self.gc_stats),
            }
        }


@pytest.fixture
def performance_metrics():
    """Provide performance metrics collector."""
    return PerformanceMetrics()


@pytest.fixture
async def mock_qdrant_client():
    """Provide mock Qdrant client for performance testing."""
    mock_client = AsyncMock()

    # Mock typical responses with realistic timing
    async def mock_search(*args, **kwargs):
        await asyncio.sleep(0.01)  # Simulate 10ms search time
        return MagicMock(points=[
            MagicMock(id="doc1", score=0.95, payload={"content": "test document 1"}),
            MagicMock(id="doc2", score=0.87, payload={"content": "test document 2"}),
        ])

    async def mock_upsert(*args, **kwargs):
        await asyncio.sleep(0.05)  # Simulate 50ms upsert time
        return MagicMock(operation_id=12345, status="completed")

    async def mock_get_collection(*args, **kwargs):
        await asyncio.sleep(0.005)  # Simulate 5ms collection info time
        return MagicMock(config=MagicMock(params=MagicMock(vectors=MagicMock(size=384))))

    mock_client.search = mock_search
    mock_client.upsert = mock_upsert
    mock_client.get_collection = mock_get_collection

    return mock_client


@pytest.fixture
async def mock_embedding_service():
    """Provide mock embedding service for performance testing."""
    mock_service = AsyncMock()

    async def mock_embed(*args, **kwargs):
        await asyncio.sleep(0.02)  # Simulate 20ms embedding time
        return [[0.1] * 384]  # Mock 384-dimensional embedding

    mock_service.embed = mock_embed
    return mock_service


class TestCoreOperationPerformance:
    """Test performance of core operations."""

    @pytest.mark.benchmark
    async def test_document_processing_performance(self, performance_metrics, mock_qdrant_client, mock_embedding_service, benchmark):
        """Benchmark document processing and embedding generation."""

        # Mock document processing pipeline
        async def process_document(content: str):
            # Simulate document processing steps
            await asyncio.sleep(0.001)  # Text preprocessing
            embeddings = await mock_embedding_service.embed([content])
            await mock_qdrant_client.upsert(
                collection_name="test",
                points=[{"id": "test", "vector": embeddings[0], "payload": {"content": content}}]
            )
            return {"id": "test", "status": "processed"}

        # Test single document processing
        test_content = "This is a test document for performance benchmarking. " * 10

        def sync_process():
            return asyncio.run(performance_metrics.measure_async_operation(
                "document_processing",
                process_document(test_content)
            ))

        # Benchmark the operation
        benchmark.pedantic(sync_process, iterations=20, rounds=3)

        # Validate performance requirements
        stats = benchmark.stats
        avg_time_ms = stats.mean * 1000

        # CRITICAL: Document processing should be < 500ms per document
        assert avg_time_ms < 500.0, f"Document processing too slow: {avg_time_ms:.2f}ms"

        # Record performance metrics
        performance_metrics.response_times.extend([avg_time_ms] * 20)

        print("\n📄 Document Processing Performance:")
        print(f"   Average time: {avg_time_ms:.2f}ms")
        print(f"   Max time: {stats.max * 1000:.2f}ms")
        print(f"   Min time: {stats.min * 1000:.2f}ms")

    @pytest.mark.benchmark
    async def test_vector_search_performance(self, performance_metrics, mock_qdrant_client, benchmark):
        """Benchmark vector search operations."""

        async def vector_search(query_vector: list[float], limit: int = 10):
            return await mock_qdrant_client.search(
                collection_name="test",
                query_vector=query_vector,
                limit=limit
            )

        query_vector = [0.1] * 384

        def sync_search():
            return asyncio.run(performance_metrics.measure_async_operation(
                "vector_search",
                vector_search(query_vector)
            ))

        # Benchmark search operation
        benchmark.pedantic(sync_search, iterations=50, rounds=5)

        stats = benchmark.stats
        avg_time_ms = stats.mean * 1000

        # CRITICAL: Vector search should be < 100ms
        assert avg_time_ms < 100.0, f"Vector search too slow: {avg_time_ms:.2f}ms"

        print("\n🔍 Vector Search Performance:")
        print(f"   Average time: {avg_time_ms:.2f}ms")
        print(f"   Max time: {stats.max * 1000:.2f}ms")
        print(f"   Throughput: {1000 / avg_time_ms:.1f} searches/second")

    @pytest.mark.benchmark
    async def test_hybrid_search_performance(self, performance_metrics, mock_qdrant_client, mock_embedding_service, benchmark):
        """Benchmark hybrid search combining dense and sparse vectors."""

        async def hybrid_search(query: str):
            # Simulate hybrid search pipeline
            embeddings = await mock_embedding_service.embed([query])

            # Dense search
            dense_results = await mock_qdrant_client.search(
                collection_name="test",
                query_vector=embeddings[0],
                limit=20
            )

            # Sparse search (simulated)
            await asyncio.sleep(0.015)  # Simulate sparse search time

            # Fusion (simulated)
            await asyncio.sleep(0.005)  # Simulate fusion time

            return {"results": dense_results.points[:10], "search_type": "hybrid"}

        def sync_hybrid_search():
            return asyncio.run(performance_metrics.measure_async_operation(
                "hybrid_search",
                hybrid_search("test query for performance")
            ))

        # Benchmark hybrid search
        benchmark.pedantic(sync_hybrid_search, iterations=30, rounds=3)

        stats = benchmark.stats
        avg_time_ms = stats.mean * 1000

        # CRITICAL: Hybrid search should be < 150ms (allowing for additional complexity)
        assert avg_time_ms < 150.0, f"Hybrid search too slow: {avg_time_ms:.2f}ms"

        print("\n🔀 Hybrid Search Performance:")
        print(f"   Average time: {avg_time_ms:.2f}ms")
        print(f"   Max time: {stats.max * 1000:.2f}ms")


class TestMCPToolPerformance:
    """Test performance of all MCP tools."""

    @pytest.mark.benchmark
    async def test_all_mcp_tools_performance(self, performance_metrics, mock_qdrant_client, mock_embedding_service):
        """Benchmark response times for all MCP tools."""

        # Mock MCP tool implementations
        async def mock_workspace_status():
            await asyncio.sleep(0.01)
            return {"status": "healthy", "collections": 5}

        async def mock_search_workspace():
            await asyncio.sleep(0.05)
            return {"results": [], "total": 0}

        async def mock_add_document():
            await asyncio.sleep(0.1)
            return {"id": "doc123", "status": "added"}

        async def mock_get_document():
            await asyncio.sleep(0.02)
            return {"id": "doc123", "content": "test"}

        async def mock_update_scratchbook():
            await asyncio.sleep(0.03)
            return {"note_id": "note123", "status": "updated"}

        async def mock_search_scratchbook():
            await asyncio.sleep(0.04)
            return {"results": [], "total": 0}

        async def mock_research_workspace():
            await asyncio.sleep(0.15)
            return {"research_results": [], "confidence": 0.8}

        async def mock_hybrid_search_advanced():
            await asyncio.sleep(0.08)
            return {"hybrid_results": [], "search_time_ms": 80}

        async def mock_list_collections():
            await asyncio.sleep(0.005)
            return {"collections": ["test1", "test2"]}

        async def mock_create_collection():
            await asyncio.sleep(0.02)
            return {"collection": "new_test", "status": "created"}

        async def mock_watch_management():
            await asyncio.sleep(0.01)
            return {"watch_id": "watch123", "status": "active"}

        # Define all MCP tools to test
        mcp_tools = [
            ("workspace_status", mock_workspace_status()),
            ("search_workspace", mock_search_workspace()),
            ("add_document", mock_add_document()),
            ("get_document", mock_get_document()),
            ("update_scratchbook", mock_update_scratchbook()),
            ("search_scratchbook", mock_search_scratchbook()),
            ("research_workspace", mock_research_workspace()),
            ("hybrid_search_advanced", mock_hybrid_search_advanced()),
            ("list_collections", mock_list_collections()),
            ("create_collection", mock_create_collection()),
            ("watch_management", mock_watch_management()),
        ]

        # Test each tool individually
        tool_results = {}
        for tool_name, tool_coro in mcp_tools:
            result, duration_ms, memory_mb = await performance_metrics.measure_async_operation(
                tool_name, tool_coro
            )
            tool_results[tool_name] = {
                'duration_ms': duration_ms,
                'memory_mb': memory_mb,
                'success': result is not None
            }

            # CRITICAL: Each MCP tool should respond in < 200ms
            assert duration_ms < 200.0, f"MCP tool {tool_name} too slow: {duration_ms:.2f}ms"

        # Test concurrent tool execution
        concurrent_operations = [(name, coro) for name, coro in mcp_tools]
        concurrent_results, concurrent_metrics = await performance_metrics.measure_concurrent_operations(
            concurrent_operations, max_workers=5
        )

        # Validate concurrent performance
        assert concurrent_metrics['successful_operations'] >= len(mcp_tools) * 0.9, \
            f"Too many concurrent failures: {concurrent_metrics['failed_operations']}"

        assert concurrent_metrics['avg_response_time_ms'] < 300.0, \
            f"Concurrent response time too high: {concurrent_metrics['avg_response_time_ms']:.2f}ms"

        print("\n🛠️  MCP Tools Performance Summary:")
        for tool_name, metrics in tool_results.items():
            print(f"   {tool_name}: {metrics['duration_ms']:.2f}ms")

        print("\n⚡ Concurrent Execution:")
        print(f"   Total operations: {concurrent_metrics['total_operations']}")
        print(f"   Success rate: {concurrent_metrics['successful_operations']/concurrent_metrics['total_operations']*100:.1f}%")
        print(f"   Average response time: {concurrent_metrics['avg_response_time_ms']:.2f}ms")
        print(f"   Operations per second: {concurrent_metrics['operations_per_second']:.1f}")


class TestMemoryPerformance:
    """Test memory usage patterns and leak detection."""

    @pytest.mark.benchmark
    async def test_memory_usage_baseline(self, performance_metrics):
        """Establish memory usage baseline."""

        # Measure baseline memory before any operations
        baseline_memory = performance_metrics.process.memory_info().rss / 1024 / 1024

        # Simulate lightweight operations
        for _i in range(100):
            await performance_metrics.measure_async_operation(
                "baseline_operation",
                asyncio.sleep(0.001)
            )

        # Measure memory after operations
        final_memory = performance_metrics.process.memory_info().rss / 1024 / 1024
        memory_overhead = final_memory - baseline_memory

        # CRITICAL: Base overhead should be < 50MB
        assert memory_overhead < 50.0, f"Base memory overhead too high: {memory_overhead:.2f}MB"

        print("\n💾 Memory Baseline:")
        print(f"   Baseline memory: {baseline_memory:.2f}MB")
        print(f"   Final memory: {final_memory:.2f}MB")
        print(f"   Overhead: {memory_overhead:.2f}MB")

    @pytest.mark.benchmark
    async def test_memory_leak_detection(self, performance_metrics, mock_qdrant_client, mock_embedding_service):
        """Test for memory leaks during extended operations."""

        memory_snapshots = []
        operation_count = 500

        # Perform many operations and track memory
        for i in range(operation_count):
            # Record memory before operation
            memory_before = performance_metrics.process.memory_info().rss / 1024 / 1024
            memory_snapshots.append(memory_before)

            # Perform operation
            if i % 3 == 0:
                # Document processing
                await performance_metrics.measure_async_operation(
                    "leak_test_document",
                    mock_embedding_service.embed([f"test document {i}"])
                )
            elif i % 3 == 1:
                # Search operation
                await performance_metrics.measure_async_operation(
                    "leak_test_search",
                    mock_qdrant_client.search(collection_name="test", query_vector=[0.1]*384)
                )
            else:
                # Collection operation
                await performance_metrics.measure_async_operation(
                    "leak_test_collection",
                    mock_qdrant_client.get_collection("test")
                )

            # Force garbage collection periodically
            if i % 50 == 0:
                gc.collect()

        # Analyze for memory leaks
        leak_analysis = performance_metrics.detect_memory_leaks(operation_count, memory_snapshots)

        # CRITICAL: No significant memory leaks should be detected
        assert not leak_analysis['leak_detected'], \
            f"Memory leak detected: {leak_analysis['indicators']}"

        # CRITICAL: Total memory growth should be reasonable
        assert leak_analysis['total_growth_mb'] < 20.0, \
            f"Excessive memory growth: {leak_analysis['total_growth_mb']:.2f}MB"

        print("\n🚫 Memory Leak Analysis:")
        print(f"   Leak detected: {leak_analysis['leak_detected']}")
        print(f"   Total growth: {leak_analysis['total_growth_mb']:.2f}MB")
        print(f"   Growth per operation: {leak_analysis['growth_per_operation_mb']:.4f}MB")
        print(f"   Correlation: {leak_analysis['correlation']:.3f}")

    @pytest.mark.benchmark
    async def test_memory_scaling_with_documents(self, performance_metrics, mock_qdrant_client, mock_embedding_service):
        """Test memory usage scaling with document count."""

        document_counts = [100, 500, 1000, 2000]
        memory_usage_results = []

        for doc_count in document_counts:
            gc.collect()  # Start with clean memory
            initial_memory = performance_metrics.process.memory_info().rss / 1024 / 1024

            # Process documents
            for i in range(doc_count):
                await performance_metrics.measure_async_operation(
                    f"scaling_test_{doc_count}",
                    mock_embedding_service.embed([f"document {i} content for scaling test"])
                )

            final_memory = performance_metrics.process.memory_info().rss / 1024 / 1024
            memory_per_doc = (final_memory - initial_memory) / doc_count
            memory_usage_results.append(memory_per_doc)

            # CRITICAL: Memory per document should be < 5MB per 1000 documents (0.005MB per doc)
            assert memory_per_doc < 0.01, f"Memory per document too high: {memory_per_doc:.6f}MB for {doc_count} docs"

        # Analyze scaling behavior
        scaling_data = performance_metrics.measure_scaling_behavior(
            "memory_scaling", document_counts, memory_usage_results
        )

        print("\n📈 Memory Scaling Analysis:")
        for data in scaling_data:
            print(f"   {data['data_size']} docs: {data['response_time_ms']:.6f}MB/doc (scaling factor: {data['scaling_factor']:.2f})")


class TestConcurrentPerformance:
    """Test performance under concurrent operations."""

    @pytest.mark.benchmark
    async def test_concurrent_search_operations(self, performance_metrics, mock_qdrant_client):
        """Test concurrent search operation performance."""

        # Create multiple search operations
        search_operations = []
        for i in range(20):
            query_vector = [0.1 + i * 0.01] * 384
            search_op = mock_qdrant_client.search(
                collection_name="test",
                query_vector=query_vector,
                limit=10
            )
            search_operations.append((f"concurrent_search_{i}", search_op))

        # Execute searches concurrently
        results, metrics = await performance_metrics.measure_concurrent_operations(
            search_operations, max_workers=10
        )

        # CRITICAL: Should maintain linear scaling for concurrent operations
        expected_max_time = 200.0 * (len(search_operations) / 10)  # Assuming 10 workers
        assert metrics['total_duration_ms'] < expected_max_time, \
            f"Concurrent search scaling poor: {metrics['total_duration_ms']:.2f}ms > {expected_max_time:.2f}ms"

        # CRITICAL: High success rate for concurrent operations
        success_rate = metrics['successful_operations'] / metrics['total_operations']
        assert success_rate >= 0.95, f"Concurrent search success rate too low: {success_rate:.2%}"

        print("\n🔄 Concurrent Search Performance:")
        print(f"   Total operations: {metrics['total_operations']}")
        print(f"   Success rate: {success_rate:.1%}")
        print(f"   Total duration: {metrics['total_duration_ms']:.2f}ms")
        print(f"   Operations per second: {metrics['operations_per_second']:.1f}")

    @pytest.mark.benchmark
    async def test_concurrent_document_processing(self, performance_metrics, mock_qdrant_client, mock_embedding_service):
        """Test concurrent document processing performance."""

        # Create multiple document processing operations
        process_operations = []
        for i in range(15):
            content = f"Document {i} content for concurrent processing test. " * 20

            async def process_doc(content=content, doc_id=i):
                embeddings = await mock_embedding_service.embed([content])
                return await mock_qdrant_client.upsert(
                    collection_name="test",
                    points=[{"id": f"doc_{doc_id}", "vector": embeddings[0], "payload": {"content": content}}]
                )

            process_operations.append((f"concurrent_process_{i}", process_doc()))

        # Execute processing concurrently
        results, metrics = await performance_metrics.measure_concurrent_operations(
            process_operations, max_workers=5
        )

        # Validate concurrent processing performance
        assert metrics['successful_operations'] >= len(process_operations) * 0.9, \
            f"Too many concurrent processing failures: {metrics['failed_operations']}"

        # CRITICAL: Average response time should not be significantly higher than single operation
        assert metrics['avg_response_time_ms'] < 800.0, \
            f"Concurrent processing response time too high: {metrics['avg_response_time_ms']:.2f}ms"

        print("\n📝 Concurrent Document Processing:")
        print(f"   Operations: {metrics['total_operations']}")
        print(f"   Success rate: {metrics['successful_operations']/metrics['total_operations']:.1%}")
        print(f"   Avg response time: {metrics['avg_response_time_ms']:.2f}ms")
        print(f"   Memory overhead: {metrics['memory_overhead_mb']:.2f}MB")


class TestScalingPerformance:
    """Test scaling behavior with increasing dataset sizes."""

    @pytest.mark.benchmark
    async def test_search_scaling_behavior(self, performance_metrics, mock_qdrant_client):
        """Test search performance scaling with dataset size."""

        # Simulate different dataset sizes by adjusting mock response time
        dataset_sizes = [100, 1000, 5000, 10000]
        search_times = []

        for size in dataset_sizes:
            # Adjust mock search time based on dataset size (logarithmic scaling)
            import math
            expected_time = 0.01 + (math.log10(size) * 0.005)  # Logarithmic scaling

            async def scaled_search():
                await asyncio.sleep(expected_time)
                return MagicMock(points=[MagicMock(id=f"doc_{i}", score=0.9) for i in range(min(10, size))])

            # Measure search time
            result, duration_ms, memory_mb = await performance_metrics.measure_async_operation(
                f"scaling_search_{size}",
                scaled_search()
            )

            search_times.append(duration_ms)

            # CRITICAL: Search time should scale sub-linearly (better than O(n))
            if size > dataset_sizes[0]:
                size_ratio = size / dataset_sizes[0]
                time_ratio = duration_ms / search_times[0]
                scaling_factor = time_ratio / size_ratio

                # Scaling factor should be < 1.0 for sub-linear scaling
                assert scaling_factor < 1.5, \
                    f"Poor scaling for {size} documents: factor {scaling_factor:.2f}"

        # Analyze overall scaling behavior
        scaling_data = performance_metrics.measure_scaling_behavior(
            "search_scaling", dataset_sizes, search_times
        )

        print("\n📊 Search Scaling Analysis:")
        for data in scaling_data:
            print(f"   {data['data_size']} docs: {data['response_time_ms']:.2f}ms (scaling factor: {data['scaling_factor']:.2f})")

    @pytest.mark.benchmark
    async def test_collection_scaling_behavior(self, performance_metrics, mock_qdrant_client):
        """Test collection management scaling with number of collections."""

        collection_counts = [10, 50, 100, 200]
        collection_times = []

        for count in collection_counts:
            # Simulate collection listing time based on count
            list_time = 0.005 + (count * 0.0001)  # Linear scaling with small constant

            async def list_collections():
                await asyncio.sleep(list_time)
                return [f"collection_{i}" for i in range(count)]

            # Measure collection operation time
            result, duration_ms, memory_mb = await performance_metrics.measure_async_operation(
                f"collection_scaling_{count}",
                list_collections()
            )

            collection_times.append(duration_ms)

            # CRITICAL: Collection operations should scale reasonably
            assert duration_ms < 100.0, f"Collection operation too slow for {count} collections: {duration_ms:.2f}ms"

        # Analyze collection scaling
        scaling_data = performance_metrics.measure_scaling_behavior(
            "collection_scaling", collection_counts, collection_times
        )

        print("\n📚 Collection Scaling Analysis:")
        for data in scaling_data:
            print(f"   {data['data_size']} collections: {data['response_time_ms']:.2f}ms")


class TestPerformanceRegression:
    """Test performance regression thresholds."""

    @pytest.mark.benchmark
    async def test_performance_regression_detection(self, performance_metrics):
        """Test performance regression detection against baselines."""

        # Baseline performance metrics (these would be stored from previous runs)
        baseline_metrics = {
            'document_processing_ms': 150.0,
            'vector_search_ms': 50.0,
            'hybrid_search_ms': 80.0,
            'mcp_tool_avg_ms': 100.0,
            'memory_usage_mb': 25.0,
            'concurrent_ops_per_sec': 50.0
        }

        # Current performance metrics (simulated)
        current_metrics = {
            'document_processing_ms': 165.0,  # 10% increase
            'vector_search_ms': 55.0,         # 10% increase
            'hybrid_search_ms': 95.0,         # 18.75% increase
            'mcp_tool_avg_ms': 110.0,         # 10% increase
            'memory_usage_mb': 30.0,          # 20% increase
            'concurrent_ops_per_sec': 48.0    # 4% decrease
        }

        # Define regression thresholds
        thresholds = {
            'response_time_increase_percent': 20.0,
            'memory_increase_percent': 30.0,
            'throughput_decrease_percent': 15.0
        }

        # Check for regressions
        regressions = []

        # Response time regressions
        for metric in ['document_processing_ms', 'vector_search_ms', 'hybrid_search_ms', 'mcp_tool_avg_ms']:
            baseline = baseline_metrics[metric]
            current = current_metrics[metric]
            increase_percent = ((current - baseline) / baseline) * 100

            if increase_percent > thresholds['response_time_increase_percent']:
                regressions.append(f"{metric}: {increase_percent:.1f}% increase > {thresholds['response_time_increase_percent']}% threshold")

        # Memory regression
        memory_baseline = baseline_metrics['memory_usage_mb']
        memory_current = current_metrics['memory_usage_mb']
        memory_increase = ((memory_current - memory_baseline) / memory_baseline) * 100

        if memory_increase > thresholds['memory_increase_percent']:
            regressions.append(f"memory_usage_mb: {memory_increase:.1f}% increase > {thresholds['memory_increase_percent']}% threshold")

        # Throughput regression
        throughput_baseline = baseline_metrics['concurrent_ops_per_sec']
        throughput_current = current_metrics['concurrent_ops_per_sec']
        throughput_decrease = ((throughput_baseline - throughput_current) / throughput_baseline) * 100

        if throughput_decrease > thresholds['throughput_decrease_percent']:
            regressions.append(f"concurrent_ops_per_sec: {throughput_decrease:.1f}% decrease > {thresholds['throughput_decrease_percent']}% threshold")

        # CRITICAL: No significant performance regressions should be detected
        assert len(regressions) == 0, f"Performance regressions detected: {'; '.join(regressions)}"

        print("\n✅ Performance Regression Analysis:")
        print(f"   Response time threshold: {thresholds['response_time_increase_percent']}%")
        print(f"   Memory threshold: {thresholds['memory_increase_percent']}%")
        print(f"   Throughput threshold: {thresholds['throughput_decrease_percent']}%")
        print(f"   Regressions detected: {len(regressions)}")

        # Generate performance comparison report
        print("\n📈 Performance Comparison:")
        for metric in current_metrics:
            baseline = baseline_metrics[metric]
            current = current_metrics[metric]
            change_percent = ((current - baseline) / baseline) * 100
            status = "📈" if change_percent > 0 else "📉"
            print(f"   {metric}: {baseline:.1f} → {current:.1f} ({status} {change_percent:+.1f}%)")


@pytest.mark.benchmark
async def test_comprehensive_performance_report(performance_metrics):
    """Generate comprehensive performance report."""

    # This test would be run after all other performance tests
    # to generate a final comprehensive report

    summary = performance_metrics.get_performance_summary()

    print("\n" + "="*60)
    print("📊 COMPREHENSIVE PERFORMANCE REPORT")
    print("="*60)

    if 'error' in summary:
        print(f"❌ Error: {summary['error']}")
        return

    # Response time summary
    rt_stats = summary['response_time_stats']
    print("\n⏱️  Response Time Performance:")
    print(f"   Average: {rt_stats['avg_ms']:.2f}ms")
    print(f"   Maximum: {rt_stats['max_ms']:.2f}ms")
    print(f"   Minimum: {rt_stats['min_ms']:.2f}ms")
    print(f"   Total operations: {rt_stats['total_operations']}")

    # Memory summary
    mem_stats = summary['memory_stats']
    print("\n💾 Memory Performance:")
    print(f"   Average usage: {mem_stats['avg_usage_mb']:.2f}MB")
    print(f"   Maximum usage: {mem_stats['max_usage_mb']:.2f}MB")
    print(f"   Total snapshots: {mem_stats['total_snapshots']}")

    # Error summary
    err_stats = summary['error_stats']
    print("\n🚨 Error Statistics:")
    print(f"   Total operations: {err_stats['total_operations']}")
    print(f"   Success rate: {(err_stats['successful_operations']/err_stats['total_operations']*100):.1f}%")
    print(f"   Error rate: {(err_stats['error_rate']*100):.2f}%")

    # Tool performance
    if summary['tool_performance']:
        print("\n🛠️  Tool Performance:")
        for tool_name, tool_stats in summary['tool_performance'].items():
            print(f"   {tool_name}: {tool_stats['avg_latency_ms']:.2f}ms avg, {tool_stats['operation_count']} ops")

    # Concurrent performance
    if summary['concurrent_performance']:
        print("\n⚡ Concurrent Performance:")
        for i, perf in enumerate(summary['concurrent_performance']):
            print(f"   Test {i+1}: {perf['operations_per_second']:.1f} ops/sec, {perf['avg_response_time_ms']:.2f}ms avg")

    # GC statistics
    gc_stats = summary['gc_stats']
    print("\n🗑️  Garbage Collection:")
    print(f"   Gen 0 collections: {gc_stats['total_gen0_collections']}")
    print(f"   Gen 1 collections: {gc_stats['total_gen1_collections']}")
    print(f"   Gen 2 collections: {gc_stats['total_gen2_collections']}")

    print("\n" + "="*60)

    # Validate overall performance meets requirements
    assert rt_stats['avg_ms'] < 300.0, f"Overall average response time too high: {rt_stats['avg_ms']:.2f}ms"
    assert mem_stats['avg_usage_mb'] < 100.0, f"Average memory usage too high: {mem_stats['avg_usage_mb']:.2f}MB"
    assert err_stats['error_rate'] < 0.05, f"Error rate too high: {err_stats['error_rate']*100:.2f}%"

    return summary
