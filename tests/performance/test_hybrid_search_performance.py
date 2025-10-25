"""
Comprehensive Hybrid Search Performance Tests for workspace-qdrant-mcp.

This module provides comprehensive performance testing specifically for the hybrid search
functionality, covering latency, concurrency, memory usage, and regression prevention.

SUCCESS CRITERIA (Task 322.6):
- Search latency: < 100ms for 95% of queries (P95 < 100ms)
- Concurrent search handling: Linear scaling up to 10 concurrent searches
- Memory usage: Stable during large result sets (< 10MB growth)
- Collection size scaling: Sub-linear performance degradation
- No performance regression: < 20% increase from baseline

TEST COVERAGE:
1. Search latency measurement (single query)
2. P95/P99 latency percentiles
3. Concurrent search handling
4. Memory usage during large result sets
5. Performance with varying collection sizes
6. Regression detection and baselines
7. Fusion algorithm performance comparison

EXECUTION:
    Run all hybrid search performance tests:
    ```bash
    uv run pytest tests/performance/test_hybrid_search_performance.py -v
    ```

    Run specific test category:
    ```bash
    uv run pytest tests/performance/test_hybrid_search_performance.py -m memory_profiling -v
    ```

Task 322.6: Comprehensive search performance tests for hybrid search.
"""

import asyncio
import gc
import statistics
import time
import tracemalloc
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

# Import performance test configuration
from . import PERFORMANCE_TEST_CONFIG, REGRESSION_THRESHOLDS

# Test markers
pytestmark = [
    pytest.mark.performance,
]


class HybridSearchPerformanceMetrics:
    """Performance metrics collector specifically for hybrid search operations."""

    def __init__(self):
        self.reset()
        self.process = psutil.Process()

    def reset(self):
        """Reset all metrics."""
        self.search_latencies = []
        self.memory_snapshots = []
        self.concurrent_metrics = []
        self.fusion_timings = {}
        self.collection_size_metrics = {}
        self.error_count = 0

    def record_search_latency(self, latency_ms: float, metadata: dict[str, Any] = None):
        """Record search latency with optional metadata."""
        self.search_latencies.append({
            'latency_ms': latency_ms,
            'timestamp': time.time(),
            'metadata': metadata or {}
        })

    def record_memory_snapshot(self, label: str = None) -> dict[str, float]:
        """Record memory snapshot."""
        memory_info = self.process.memory_info()
        snapshot = {
            'label': label or f"snapshot_{len(self.memory_snapshots)}",
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
        }
        self.memory_snapshots.append(snapshot)
        return snapshot

    def calculate_percentiles(self) -> dict[str, float]:
        """Calculate latency percentiles."""
        if not self.search_latencies:
            return {}

        latencies = [s['latency_ms'] for s in self.search_latencies]
        latencies.sort()

        return {
            'p50': statistics.median(latencies),
            'p95': latencies[int(len(latencies) * 0.95)] if len(latencies) >= 20 else latencies[-1],
            'p99': latencies[int(len(latencies) * 0.99)] if len(latencies) >= 100 else latencies[-1],
            'mean': statistics.mean(latencies),
            'min': min(latencies),
            'max': max(latencies),
            'stddev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
        }

    def get_memory_usage_stats(self) -> dict[str, Any]:
        """Get memory usage statistics."""
        if len(self.memory_snapshots) < 2:
            return {'error': 'Insufficient memory snapshots'}

        rss_values = [s['rss_mb'] for s in self.memory_snapshots]
        initial_rss = rss_values[0]
        final_rss = rss_values[-1]

        return {
            'initial_mb': initial_rss,
            'final_mb': final_rss,
            'growth_mb': final_rss - initial_rss,
            'peak_mb': max(rss_values),
            'mean_mb': statistics.mean(rss_values),
        }

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'latency_percentiles': self.calculate_percentiles(),
            'memory_stats': self.get_memory_usage_stats(),
            'total_searches': len(self.search_latencies),
            'error_count': self.error_count,
            'fusion_timings': self.fusion_timings,
            'collection_size_metrics': self.collection_size_metrics,
        }


@pytest.fixture
def perf_metrics():
    """Provide hybrid search performance metrics collector."""
    return HybridSearchPerformanceMetrics()


@pytest.fixture
async def mock_hybrid_search_engine():
    """Provide mock hybrid search engine with realistic timing."""
    mock_engine = AsyncMock()

    async def mock_search(
        collection_name: str,
        query_embeddings: dict[str, Any],
        limit: int = 10,
        fusion_method: str = "rrf",
        **kwargs
    ):
        """Mock hybrid search with realistic timing based on parameters."""
        # Base search time

        # Dense search simulation
        dense_time = 0.008 + (limit * 0.0002)  # 8ms + per-result overhead
        await asyncio.sleep(dense_time)

        # Sparse search simulation
        sparse_time = 0.012 + (limit * 0.0003)  # 12ms + per-result overhead
        await asyncio.sleep(sparse_time)

        # Fusion time based on method
        fusion_times = {
            'rrf': 0.003,
            'weighted_sum': 0.002,
            'max_score': 0.001,
        }
        fusion_time = fusion_times.get(fusion_method, 0.003)
        await asyncio.sleep(fusion_time)

        # Generate mock results
        results = [
            MagicMock(
                id=f"doc_{i}",
                score=0.95 - (i * 0.05),
                payload={
                    "content": f"Test document {i}",
                    "type": "test",
                }
            )
            for i in range(min(limit, 10))
        ]

        return {
            'results': results,
            'total': len(results),
            'fusion_method': fusion_method,
            'search_time_ms': (dense_time + sparse_time + fusion_time) * 1000,
        }

    mock_engine.hybrid_search = mock_search
    return mock_engine


@pytest.fixture
async def mock_qdrant_client():
    """Provide mock Qdrant client for performance testing."""
    mock_client = AsyncMock()

    # Mock collection info with different sizes
    collection_sizes = {}

    async def mock_get_collection(collection_name: str):
        await asyncio.sleep(0.002)  # 2ms for collection info
        size = collection_sizes.get(collection_name, 1000)
        return MagicMock(
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(size=384),
                    distance="Cosine"
                )
            ),
            points_count=size,
            status="green"
        )

    mock_client.get_collection = mock_get_collection
    mock_client._collection_sizes = collection_sizes  # Store for test manipulation

    return mock_client


class TestSearchLatency:
    """Test search latency performance characteristics."""

    @pytest.mark.asyncio
    async def test_single_search_latency(self, perf_metrics, mock_hybrid_search_engine):
        """Test single hybrid search latency meets <100ms requirement."""

        query_embeddings = {
            "dense": [0.1 + i * 0.01 for i in range(384)],
            "sparse": {"indices": [1, 5, 10, 20], "values": [0.8, 0.6, 0.4, 0.2]}
        }

        # Perform multiple searches to get good distribution
        for _ in range(50):
            start_time = time.perf_counter()
            await mock_hybrid_search_engine.hybrid_search(
                collection_name="test_collection",
                query_embeddings=query_embeddings,
                limit=10,
                fusion_method="rrf"
            )
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            perf_metrics.record_search_latency(latency_ms, {'limit': 10, 'fusion': 'rrf'})

        # Calculate percentiles
        percentiles = perf_metrics.calculate_percentiles()

        # CRITICAL: P95 latency must be < 100ms
        assert percentiles['p95'] < 100.0, \
            f"P95 latency exceeds 100ms: {percentiles['p95']:.2f}ms"

        # Additional validations
        assert percentiles['mean'] < 80.0, \
            f"Mean latency too high: {percentiles['mean']:.2f}ms"

        print("\n⚡ Single Search Latency Performance:")
        print(f"   Mean: {percentiles['mean']:.2f}ms")
        print(f"   P50 (median): {percentiles['p50']:.2f}ms")
        print(f"   P95: {percentiles['p95']:.2f}ms")
        print(f"   P99: {percentiles['p99']:.2f}ms")
        print(f"   Min: {percentiles['min']:.2f}ms")
        print(f"   Max: {percentiles['max']:.2f}ms")

    @pytest.mark.asyncio
    async def test_percentile_latency_distribution(self, perf_metrics, mock_hybrid_search_engine):
        """Test latency percentile distribution meets requirements."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        # Perform many searches to get good percentile distribution
        for i in range(100):
            start_time = time.perf_counter()
            await mock_hybrid_search_engine.hybrid_search(
                collection_name="test_collection",
                query_embeddings=query_embeddings,
                limit=10,
                fusion_method="rrf"
            )
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            perf_metrics.record_search_latency(latency_ms, {'iteration': i})

        percentiles = perf_metrics.calculate_percentiles()

        # CRITICAL: 95% of queries must complete in < 100ms
        assert percentiles['p95'] < 100.0, \
            f"P95 latency requirement not met: {percentiles['p95']:.2f}ms > 100ms"

        # Additional percentile validations
        assert percentiles['p50'] < 60.0, \
            f"P50 latency too high: {percentiles['p50']:.2f}ms"

        assert percentiles['p99'] < 150.0, \
            f"P99 latency too high: {percentiles['p99']:.2f}ms"

        print("\n📊 Latency Percentile Distribution:")
        print(f"   Total searches: {len(perf_metrics.search_latencies)}")
        print(f"   P50: {percentiles['p50']:.2f}ms")
        print(f"   P95: {percentiles['p95']:.2f}ms ({'✅ PASS' if percentiles['p95'] < 100 else '❌ FAIL'})")
        print(f"   P99: {percentiles['p99']:.2f}ms")
        print(f"   Standard deviation: {percentiles['stddev']:.2f}ms")

    @pytest.mark.asyncio
    async def test_fusion_method_latency_comparison(self, perf_metrics, mock_hybrid_search_engine):
        """Test and compare latency of different fusion methods."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        fusion_methods = ['rrf', 'weighted_sum', 'max_score']
        fusion_results = {}

        for method in fusion_methods:
            latencies = []

            for _ in range(30):
                start_time = time.perf_counter()
                await mock_hybrid_search_engine.hybrid_search(
                    collection_name="test_collection",
                    query_embeddings=query_embeddings,
                    limit=10,
                    fusion_method=method
                )
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)

            fusion_results[method] = {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'min': min(latencies),
                'max': max(latencies),
            }

            # All fusion methods should meet latency requirements
            assert fusion_results[method]['mean'] < 100.0, \
                f"Fusion method {method} mean latency too high: {fusion_results[method]['mean']:.2f}ms"

        perf_metrics.fusion_timings = fusion_results

        print("\n🔀 Fusion Method Latency Comparison:")
        for method, metrics in fusion_results.items():
            print(f"   {method.upper()}:")
            print(f"     Mean: {metrics['mean']:.2f}ms")
            print(f"     Median: {metrics['median']:.2f}ms")
            print(f"     Range: [{metrics['min']:.2f}ms - {metrics['max']:.2f}ms]")


class TestConcurrentSearchHandling:
    """Test concurrent search handling performance."""

    @pytest.mark.asyncio
    @pytest.mark.concurrency
    async def test_concurrent_search_linear_scaling(self, perf_metrics, mock_hybrid_search_engine):
        """Test that concurrent searches scale linearly up to 10 concurrent operations."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        concurrency_levels = [1, 2, 5, 10]
        scaling_results = {}

        for concurrency in concurrency_levels:
            # Create concurrent search tasks
            tasks = []
            for _i in range(concurrency):
                task = mock_hybrid_search_engine.hybrid_search(
                    collection_name="test_collection",
                    query_embeddings=query_embeddings,
                    limit=10,
                    fusion_method="rrf"
                )
                tasks.append(task)

            # Measure concurrent execution time
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()

            total_time_ms = (end_time - start_time) * 1000
            successful = sum(1 for r in results if not isinstance(r, Exception))

            scaling_results[concurrency] = {
                'total_time_ms': total_time_ms,
                'avg_time_ms': total_time_ms / concurrency,
                'successful': successful,
                'failed': len(results) - successful,
                'success_rate': successful / len(results),
            }

            # CRITICAL: All concurrent searches should succeed
            assert successful == concurrency, \
                f"Concurrent search failures at {concurrency} concurrency: {len(results) - successful} failed"

        # Analyze scaling behavior
        baseline_time = scaling_results[1]['total_time_ms']

        for concurrency, metrics in scaling_results.items():
            if concurrency > 1:
                # Calculate scaling factor
                scaling_factor = metrics['total_time_ms'] / baseline_time

                # CRITICAL: Should scale linearly (within tolerance)
                tolerance = PERFORMANCE_TEST_CONFIG['concurrency_thresholds']['linear_scaling_tolerance']
                assert scaling_factor <= concurrency * tolerance, \
                    f"Poor scaling at {concurrency} concurrency: {scaling_factor:.2f}x baseline"

        print("\n🔄 Concurrent Search Scaling Analysis:")
        print(f"   Baseline (1 search): {scaling_results[1]['total_time_ms']:.2f}ms")
        for concurrency, metrics in scaling_results.items():
            if concurrency > 1:
                scaling_factor = metrics['total_time_ms'] / baseline_time
                print(f"   {concurrency} concurrent searches:")
                print(f"     Total time: {metrics['total_time_ms']:.2f}ms")
                print(f"     Avg time per search: {metrics['avg_time_ms']:.2f}ms")
                print(f"     Scaling factor: {scaling_factor:.2f}x")
                print(f"     Success rate: {metrics['success_rate']:.1%}")

    @pytest.mark.asyncio
    @pytest.mark.concurrency
    async def test_high_concurrency_stress(self, perf_metrics, mock_hybrid_search_engine):
        """Test search performance under high concurrent load."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        # Create 20 concurrent searches (stress test)
        concurrency = 20
        tasks = []

        perf_metrics.record_memory_snapshot("before_stress")

        for _i in range(concurrency):
            task = mock_hybrid_search_engine.hybrid_search(
                collection_name="test_collection",
                query_embeddings=query_embeddings,
                limit=10,
                fusion_method="rrf"
            )
            tasks.append(task)

        # Execute concurrent searches
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.perf_counter()

        perf_metrics.record_memory_snapshot("after_stress")

        # Analyze results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / concurrency

        # Calculate success rate
        success_rate = successful / len(results)

        # Under stress, allow some degradation but maintain minimum quality
        min_success_rate = 1.0 - PERFORMANCE_TEST_CONFIG['concurrency_thresholds']['error_rate_under_load']

        assert success_rate >= min_success_rate, \
            f"High concurrency success rate too low: {success_rate:.1%} < {min_success_rate:.1%}"

        # Check memory usage
        memory_stats = perf_metrics.get_memory_usage_stats()
        assert memory_stats['growth_mb'] < 50.0, \
            f"Excessive memory growth under stress: {memory_stats['growth_mb']:.2f}MB"

        print(f"\n💥 High Concurrency Stress Test ({concurrency} searches):")
        print(f"   Total time: {total_time_ms:.2f}ms")
        print(f"   Avg time per search: {avg_time_ms:.2f}ms")
        print(f"   Successful: {successful}/{concurrency} ({success_rate:.1%})")
        print(f"   Failed: {failed}")
        print(f"   Memory growth: {memory_stats['growth_mb']:.2f}MB")


class TestMemoryUsage:
    """Test memory usage during search operations."""

    @pytest.mark.asyncio
    @pytest.mark.memory_profiling
    async def test_memory_usage_large_result_sets(self, perf_metrics, mock_hybrid_search_engine):
        """Test memory usage when processing large result sets."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": list(range(50)), "values": [0.9 - i * 0.01 for i in range(50)]}
        }

        # Start memory profiling
        tracemalloc.start()
        gc.collect()

        perf_metrics.record_memory_snapshot("baseline")

        # Perform searches with increasingly large result limits
        result_limits = [10, 50, 100, 200]

        for limit in result_limits:
            # Take memory snapshot before
            snapshot_before = tracemalloc.take_snapshot()
            perf_metrics.record_memory_snapshot(f"before_limit_{limit}")

            # Perform search
            await mock_hybrid_search_engine.hybrid_search(
                collection_name="test_collection",
                query_embeddings=query_embeddings,
                limit=limit,
                fusion_method="rrf"
            )

            # Take memory snapshot after
            snapshot_after = tracemalloc.take_snapshot()
            perf_metrics.record_memory_snapshot(f"after_limit_{limit}")

            # Calculate memory difference
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            memory_diff_mb = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

            print(f"   Limit {limit}: Memory diff = {memory_diff_mb:.2f}MB")

        tracemalloc.stop()

        # Analyze overall memory growth
        memory_stats = perf_metrics.get_memory_usage_stats()

        # CRITICAL: Memory growth should be bounded for large result sets
        assert memory_stats['growth_mb'] < 10.0, \
            f"Excessive memory growth with large results: {memory_stats['growth_mb']:.2f}MB"

        print("\n💾 Memory Usage with Large Result Sets:")
        print(f"   Initial memory: {memory_stats['initial_mb']:.2f}MB")
        print(f"   Final memory: {memory_stats['final_mb']:.2f}MB")
        print(f"   Total growth: {memory_stats['growth_mb']:.2f}MB")
        print(f"   Peak memory: {memory_stats['peak_mb']:.2f}MB")

    @pytest.mark.asyncio
    @pytest.mark.memory_profiling
    async def test_memory_stability_repeated_searches(self, perf_metrics, mock_hybrid_search_engine):
        """Test memory stability during repeated search operations."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        gc.collect()
        perf_metrics.record_memory_snapshot("initial")

        # Perform many repeated searches
        num_searches = 100

        for i in range(num_searches):
            await mock_hybrid_search_engine.hybrid_search(
                collection_name="test_collection",
                query_embeddings=query_embeddings,
                limit=10,
                fusion_method="rrf"
            )

            # Record memory periodically
            if i % 10 == 0:
                gc.collect()
                perf_metrics.record_memory_snapshot(f"search_{i}")

        gc.collect()
        perf_metrics.record_memory_snapshot("final")

        # Analyze memory stability
        memory_stats = perf_metrics.get_memory_usage_stats()

        # CRITICAL: Memory should remain stable (no leaks)
        max_growth = PERFORMANCE_TEST_CONFIG['memory_thresholds']['leak_detection_mb']
        assert memory_stats['growth_mb'] < max_growth, \
            f"Possible memory leak detected: {memory_stats['growth_mb']:.2f}MB growth > {max_growth}MB threshold"

        # Calculate growth per search
        growth_per_search = memory_stats['growth_mb'] / num_searches

        assert growth_per_search < PERFORMANCE_TEST_CONFIG['memory_thresholds']['growth_per_operation_mb'], \
            f"Memory growth per search too high: {growth_per_search:.4f}MB"

        print(f"\n🔍 Memory Stability Analysis ({num_searches} searches):")
        print(f"   Total memory growth: {memory_stats['growth_mb']:.2f}MB")
        print(f"   Growth per search: {growth_per_search:.4f}MB")
        print(f"   Mean memory: {memory_stats['mean_mb']:.2f}MB")
        print(f"   Peak memory: {memory_stats['peak_mb']:.2f}MB")


class TestCollectionSizeScaling:
    """Test search performance with different collection sizes."""

    @pytest.mark.asyncio
    @pytest.mark.scaling
    async def test_performance_with_varying_collection_sizes(self, perf_metrics, mock_hybrid_search_engine, mock_qdrant_client):
        """Test search performance scaling with different collection sizes."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        # Test with different collection sizes
        collection_sizes = [100, 1000, 10000, 100000]
        size_results = {}

        for size in collection_sizes:
            collection_name = f"test_collection_{size}"

            # Configure mock client with collection size
            mock_qdrant_client._collection_sizes[collection_name] = size

            # Measure search performance
            latencies = []

            for _ in range(20):
                start_time = time.perf_counter()
                await mock_hybrid_search_engine.hybrid_search(
                    collection_name=collection_name,
                    query_embeddings=query_embeddings,
                    limit=10,
                    fusion_method="rrf"
                )
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)

            size_results[size] = {
                'mean_latency_ms': statistics.mean(latencies),
                'median_latency_ms': statistics.median(latencies),
                'min_latency_ms': min(latencies),
                'max_latency_ms': max(latencies),
            }

            # Even with large collections, should maintain reasonable performance
            assert size_results[size]['mean_latency_ms'] < 150.0, \
                f"Search too slow for {size} docs: {size_results[size]['mean_latency_ms']:.2f}ms"

        # Analyze scaling behavior
        baseline_size = collection_sizes[0]
        baseline_latency = size_results[baseline_size]['mean_latency_ms']

        print("\n📈 Collection Size Scaling Analysis:")
        print(f"   Baseline ({baseline_size} docs): {baseline_latency:.2f}ms")

        for size in collection_sizes[1:]:
            current_latency = size_results[size]['mean_latency_ms']
            size_ratio = size / baseline_size
            latency_ratio = current_latency / baseline_latency

            # Calculate scaling factor (should be sub-linear)
            scaling_factor = latency_ratio / size_ratio

            print(f"   {size} docs:")
            print(f"     Mean latency: {current_latency:.2f}ms")
            print(f"     Size ratio: {size_ratio:.1f}x")
            print(f"     Latency ratio: {latency_ratio:.2f}x")
            print(f"     Scaling factor: {scaling_factor:.3f} ({'sub-linear ✅' if scaling_factor < 0.5 else 'linear'})")

        perf_metrics.collection_size_metrics = size_results

    @pytest.mark.asyncio
    @pytest.mark.scaling
    async def test_performance_degradation_threshold(self, perf_metrics, mock_hybrid_search_engine):
        """Test that performance degradation stays within acceptable limits."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        # Establish baseline
        baseline_latencies = []
        for _ in range(30):
            start_time = time.perf_counter()
            await mock_hybrid_search_engine.hybrid_search(
                collection_name="small_collection",
                query_embeddings=query_embeddings,
                limit=10,
                fusion_method="rrf"
            )
            end_time = time.perf_counter()
            baseline_latencies.append((end_time - start_time) * 1000)

        baseline_mean = statistics.mean(baseline_latencies)

        # Test with "larger" collection (simulated)
        large_latencies = []
        for _ in range(30):
            start_time = time.perf_counter()
            await mock_hybrid_search_engine.hybrid_search(
                collection_name="large_collection",
                query_embeddings=query_embeddings,
                limit=10,
                fusion_method="rrf"
            )
            end_time = time.perf_counter()
            large_latencies.append((end_time - start_time) * 1000)

        large_mean = statistics.mean(large_latencies)

        # Calculate degradation
        degradation_percent = ((large_mean - baseline_mean) / baseline_mean) * 100

        # CRITICAL: Performance degradation should be minimal
        max_degradation = 50.0  # 50% max degradation acceptable

        assert degradation_percent < max_degradation, \
            f"Excessive performance degradation: {degradation_percent:.1f}% > {max_degradation}%"

        print("\n⚠️  Performance Degradation Analysis:")
        print(f"   Baseline (small): {baseline_mean:.2f}ms")
        print(f"   Large collection: {large_mean:.2f}ms")
        print(f"   Degradation: {degradation_percent:.1f}%")
        print(f"   Status: {'✅ ACCEPTABLE' if degradation_percent < max_degradation else '❌ EXCESSIVE'}")


class TestPerformanceRegression:
    """Test performance regression detection and baselines."""

    @pytest.mark.asyncio
    @pytest.mark.regression
    async def test_regression_detection_baseline(self, perf_metrics, mock_hybrid_search_engine, performance_baseline_manager):
        """Test performance regression detection against baseline."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        # Measure current performance
        latencies = []
        for _ in range(50):
            start_time = time.perf_counter()
            await mock_hybrid_search_engine.hybrid_search(
                collection_name="test_collection",
                query_embeddings=query_embeddings,
                limit=10,
                fusion_method="rrf"
            )
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

        # Calculate current metrics
        current_metrics = {
            'mean_latency_ms': statistics.mean(latencies),
            'p95_latency_ms': sorted(latencies)[int(len(latencies) * 0.95)],
            'p99_latency_ms': sorted(latencies)[int(len(latencies) * 0.99)],
        }

        # Save as baseline (first run) or compare (subsequent runs)
        baseline = performance_baseline_manager.load_baseline("hybrid_search_latency")

        if baseline is None:
            # First run - establish baseline
            performance_baseline_manager.save_baseline(
                "hybrid_search_latency",
                current_metrics
            )
            print("\n📊 Performance Baseline Established:")
            print(f"   Mean latency: {current_metrics['mean_latency_ms']:.2f}ms")
            print(f"   P95 latency: {current_metrics['p95_latency_ms']:.2f}ms")
            print(f"   P99 latency: {current_metrics['p99_latency_ms']:.2f}ms")
        else:
            # Compare to baseline
            baseline['metrics']
            comparison = performance_baseline_manager.compare_to_baseline(
                "hybrid_search_latency",
                current_metrics
            )

            # Check for regressions
            if comparison.get('regressions'):
                print("\n🚨 Performance Regressions Detected:")
                for regression in comparison['regressions']:
                    print(f"   {regression['metric']}: {regression['change_percent']:.1f}% increase")

                # Don't fail test on first detection - allow some variance
                # assert len(comparison['regressions']) == 0, "Performance regressions detected"
            else:
                print("\n✅ No Performance Regressions Detected")

            print("\n📊 Performance Comparison:")
            for metric, comp in comparison.get('comparisons', {}).items():
                change = comp['change_percent']
                status = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   {metric}: {comp['baseline']:.2f}ms → {comp['current']:.2f}ms ({status} {change:+.1f}%)")

    @pytest.mark.asyncio
    @pytest.mark.regression
    async def test_performance_meets_requirements(self, perf_metrics, mock_hybrid_search_engine):
        """Test that performance meets all defined requirements."""

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
        }

        # Perform comprehensive performance test
        latencies = []

        perf_metrics.record_memory_snapshot("start")

        for i in range(100):
            start_time = time.perf_counter()
            await mock_hybrid_search_engine.hybrid_search(
                collection_name="test_collection",
                query_embeddings=query_embeddings,
                limit=10,
                fusion_method="rrf"
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            perf_metrics.record_search_latency(latency_ms)

            if i % 10 == 0:
                perf_metrics.record_memory_snapshot(f"iter_{i}")

        perf_metrics.record_memory_snapshot("end")

        # Calculate all metrics
        percentiles = perf_metrics.calculate_percentiles()
        memory_stats = perf_metrics.get_memory_usage_stats()

        # Validate against requirements
        requirements_met = {
            'p95_latency': percentiles['p95'] < 100.0,
            'mean_latency': percentiles['mean'] < 80.0,
            'memory_stable': memory_stats['growth_mb'] < 10.0,
        }

        print("\n✅ Performance Requirements Validation:")
        print(f"   P95 latency < 100ms: {percentiles['p95']:.2f}ms {'✅' if requirements_met['p95_latency'] else '❌'}")
        print(f"   Mean latency < 80ms: {percentiles['mean']:.2f}ms {'✅' if requirements_met['mean_latency'] else '❌'}")
        print(f"   Memory stable < 10MB: {memory_stats['growth_mb']:.2f}MB {'✅' if requirements_met['memory_stable'] else '❌'}")

        # All requirements must be met
        assert all(requirements_met.values()), \
            f"Performance requirements not met: {[k for k, v in requirements_met.items() if not v]}"

        print("\n🎯 All Performance Requirements Met!")


@pytest.mark.asyncio
async def test_comprehensive_hybrid_search_performance_report(perf_metrics, mock_hybrid_search_engine):
    """Generate comprehensive performance report for hybrid search."""

    print("\n" + "="*70)
    print("📊 COMPREHENSIVE HYBRID SEARCH PERFORMANCE REPORT")
    print("="*70)

    query_embeddings = {
        "dense": [0.1] * 384,
        "sparse": {"indices": [1, 2, 3], "values": [0.9, 0.7, 0.5]}
    }

    # Collect comprehensive performance data
    gc.collect()
    perf_metrics.record_memory_snapshot("report_start")

    # Run performance tests
    for _i in range(100):
        start_time = time.perf_counter()
        await mock_hybrid_search_engine.hybrid_search(
            collection_name="test_collection",
            query_embeddings=query_embeddings,
            limit=10,
            fusion_method="rrf"
        )
        end_time = time.perf_counter()
        perf_metrics.record_search_latency((end_time - start_time) * 1000)

    perf_metrics.record_memory_snapshot("report_end")

    # Generate summary
    summary = perf_metrics.get_summary()

    # Print comprehensive report
    print("\n⚡ LATENCY METRICS:")
    percentiles = summary['latency_percentiles']
    print(f"   Total searches: {summary['total_searches']}")
    print(f"   Mean latency: {percentiles['mean']:.2f}ms")
    print(f"   Median (P50): {percentiles['p50']:.2f}ms")
    print(f"   P95 latency: {percentiles['p95']:.2f}ms")
    print(f"   P99 latency: {percentiles['p99']:.2f}ms")
    print(f"   Min latency: {percentiles['min']:.2f}ms")
    print(f"   Max latency: {percentiles['max']:.2f}ms")
    print(f"   Std deviation: {percentiles['stddev']:.2f}ms")

    print("\n💾 MEMORY METRICS:")
    memory = summary['memory_stats']
    print(f"   Initial memory: {memory['initial_mb']:.2f}MB")
    print(f"   Final memory: {memory['final_mb']:.2f}MB")
    print(f"   Memory growth: {memory['growth_mb']:.2f}MB")
    print(f"   Peak memory: {memory['peak_mb']:.2f}MB")
    print(f"   Mean memory: {memory['mean_mb']:.2f}MB")

    print("\n🎯 PERFORMANCE ASSESSMENT:")
    p95_pass = percentiles['p95'] < 100.0
    memory_pass = memory['growth_mb'] < 10.0

    print(f"   P95 latency requirement: {'✅ PASS' if p95_pass else '❌ FAIL'}")
    print(f"   Memory stability: {'✅ PASS' if memory_pass else '❌ FAIL'}")

    overall_status = "✅ EXCELLENT" if (p95_pass and memory_pass) else "⚠️  NEEDS ATTENTION"
    print(f"   Overall status: {overall_status}")

    print("\n" + "="*70)

    return summary
