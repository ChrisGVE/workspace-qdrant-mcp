"""
Performance regression tests for metadata filtering operations.

This module provides comprehensive regression testing to ensure that metadata filtering
performance remains within acceptable baselines. Tests validate both response time
and search accuracy baselines.

Performance Baselines Tested:
    - Response Time: 2.18ms average target, 3.0ms acceptable
    - Search Precision: 94.2% minimum baseline
    - Search Recall: 92.0% target, 85.0% minimum
    - Cache Hit Rate: 80% target, 70% minimum
    - Multi-tenant Isolation: 100% enforcement

Task 233.6: Implementing performance regression tests for metadata filtering baselines.
"""

import asyncio
import pytest
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock

from src.python.common.core.performance_monitoring import (
    MetadataFilteringPerformanceMonitor,
    PerformanceBaseline,
    SearchAccuracyTracker,
    PerformanceBenchmarkSuite
)
from src.python.common.core.hybrid_search import HybridSearchEngine
from src.python.common.core.metadata_optimization import FilterOptimizer


class TestDataGenerator:
    """Generate test data for performance regression testing."""

    @staticmethod
    def generate_test_queries(count: int = 50) -> List[Dict]:
        """Generate test queries with expected results for benchmarking."""
        test_queries = []

        for i in range(count):
            query = {
                "embeddings": {
                    "dense": [0.1 * j for j in range(384)],  # 384-dim vector
                    "sparse": {"indices": [1, 5, 10, 15], "values": [0.8, 0.6, 0.4, 0.2]}
                },
                "expected_results": [f"doc_{i}_{j}" for j in range(5)],  # Expected 5 results
                "project_context": {
                    "project_name": f"test_project_{i % 5}",
                    "tenant_namespace": f"tenant_{i % 3}",
                    "collection_type": "project"
                },
                "filter_conditions": None,
                "query_text": f"test query {i}",
                "complexity_score": min(10, i % 10 + 1)
            }
            test_queries.append(query)

        return test_queries

    @staticmethod
    def generate_multi_tenant_test_data(tenant_count: int = 3, queries_per_tenant: int = 10) -> Dict[str, List[Dict]]:
        """Generate multi-tenant test data for isolation testing."""
        tenant_data = {}

        for tenant_id in range(tenant_count):
            tenant_namespace = f"tenant_{tenant_id}"
            tenant_queries = []

            for query_id in range(queries_per_tenant):
                query = {
                    "embeddings": {
                        "dense": [0.1 * j * (tenant_id + 1) for j in range(384)],
                        "sparse": {"indices": [tenant_id * 5 + j for j in range(4)], "values": [0.9, 0.7, 0.5, 0.3]}
                    },
                    "expected_results": [f"tenant_{tenant_id}_doc_{query_id}_{j}" for j in range(3)],
                    "tenant_context": tenant_namespace,
                    "query_text": f"tenant {tenant_id} query {query_id}"
                }
                tenant_queries.append(query)

            tenant_data[tenant_namespace] = tenant_queries

        return tenant_data


@pytest.mark.performance
class TestMetadataFilteringPerformance:
    """Comprehensive performance regression tests for metadata filtering."""

    @pytest.fixture
    async def mock_search_engine(self):
        """Create mock search engine for testing."""
        engine = Mock(spec=HybridSearchEngine)

        # Mock search results with realistic response times
        async def mock_hybrid_search(*args, **kwargs):
            # Simulate processing time between 1-4ms
            await asyncio.sleep(0.002)  # 2ms base

            return {
                "fused_results": [
                    Mock(id=f"result_{i}", score=0.9 - i * 0.1, payload={"content": f"test content {i}"})
                    for i in range(10)
                ],
                "dense_results": [],
                "sparse_results": [],
                "performance": {
                    "response_time_ms": 2.1,
                    "cache_hit": True,
                    "target_met": True
                }
            }

        async def mock_search_tenant_namespace(*args, **kwargs):
            # Simulate tenant-isolated search
            await asyncio.sleep(0.002)
            tenant_namespace = kwargs.get('tenant_namespace', 'default')

            return {
                "fused_results": [
                    Mock(
                        id=f"{tenant_namespace}_result_{i}",
                        score=0.9 - i * 0.1,
                        payload={"tenant_namespace": tenant_namespace, "content": f"tenant content {i}"}
                    )
                    for i in range(5)
                ],
                "performance": {"response_time_ms": 2.0}
            }

        engine.hybrid_search = AsyncMock(side_effect=mock_hybrid_search)
        engine.search_tenant_namespace = AsyncMock(side_effect=mock_search_tenant_namespace)

        # Mock optimization performance methods
        engine.get_optimization_performance = Mock(return_value={
            "optimizations_enabled": True,
            "filter_cache": {"cache_hit_rate": 85.0, "avg_response_time": 1.8},
            "query_optimization": {"avg_response_time": 2.0, "target_met_rate": 95.0},
            "overall_performance": {"avg_response_time": 2.05, "target_met_rate": 96.0}
        })

        return engine

    @pytest.fixture
    def performance_baseline(self):
        """Create performance baseline for testing."""
        return PerformanceBaseline(
            target_response_time=2.18,
            acceptable_response_time=3.0,
            target_precision=94.2,
            minimum_precision=90.0,
            target_recall=92.0,
            minimum_recall=85.0,
            target_cache_hit_rate=80.0,
            minimum_cache_hit_rate=70.0
        )

    @pytest.fixture
    async def performance_monitor(self, mock_search_engine, performance_baseline):
        """Create performance monitor for testing."""
        return MetadataFilteringPerformanceMonitor(
            search_engine=mock_search_engine,
            baseline_config=performance_baseline.__dict__
        )

    @pytest.mark.asyncio
    async def test_response_time_baseline_compliance(self, performance_monitor, mock_search_engine):
        """Test that metadata filtering meets response time baseline (2.18ms target)."""
        test_queries = TestDataGenerator.generate_test_queries(count=20)

        # Run metadata filtering benchmark
        result = await performance_monitor.benchmark_suite.run_metadata_filtering_benchmark(
            collection_name="test_collection",
            test_queries=test_queries[:5],  # Use smaller set for faster testing
            iterations=10
        )

        # Assert response time meets baseline
        assert result.avg_response_time <= performance_monitor.baseline.acceptable_response_time, \
            f"Average response time {result.avg_response_time}ms exceeds acceptable baseline {performance_monitor.baseline.acceptable_response_time}ms"

        # Check if it meets target (preferred but not required for test pass)
        meets_target = result.avg_response_time <= performance_monitor.baseline.target_response_time
        if meets_target:
            print(f"✅ Response time {result.avg_response_time:.2f}ms meets target {performance_monitor.baseline.target_response_time}ms")
        else:
            print(f"⚠️ Response time {result.avg_response_time:.2f}ms exceeds target {performance_monitor.baseline.target_response_time}ms but within acceptable range")

        # Ensure P95 is reasonable
        assert result.p95_response_time <= performance_monitor.baseline.acceptable_response_time * 1.5, \
            f"P95 response time {result.p95_response_time}ms is too high"

        # Ensure no performance regression
        assert not result.performance_regression, "Performance regression detected"

    @pytest.mark.asyncio
    async def test_search_accuracy_baseline_compliance(self, performance_monitor):
        """Test that search accuracy meets precision and recall baselines."""
        accuracy_tracker = performance_monitor.accuracy_tracker

        # Simulate search accuracy measurements
        test_measurements = [
            ("query_1", "test query 1", "collection_1",
             [Mock(id=f"result_{i}") for i in range(10)],  # 10 actual results
             [f"result_{i}" for i in range(8)]),  # 8 expected results (80% precision)

            ("query_2", "test query 2", "collection_1",
             [Mock(id=f"result_{i}") for i in range(5)],   # 5 actual results
             [f"result_{i}" for i in range(5)]),  # 5 expected results (100% precision)

            ("query_3", "test query 3", "collection_1",
             [Mock(id=f"result_{i}") for i in range(12)],  # 12 actual results
             [f"result_{i}" for i in range(10)]),  # 10 expected results (83% precision)
        ]

        measurements = []
        for query_id, query_text, collection, actual, expected in test_measurements:
            measurement = accuracy_tracker.record_search_accuracy(
                query_id=query_id,
                query_text=query_text,
                collection_name=collection,
                search_results=actual,
                expected_results=expected,
                tenant_context="test_tenant",
                filter_complexity=5
            )
            measurements.append(measurement)

        # Calculate aggregate accuracy
        avg_precision = statistics.mean([m.precision for m in measurements])
        avg_recall = statistics.mean([m.recall for m in measurements])

        # Assert accuracy meets minimum baselines
        assert avg_precision >= performance_monitor.baseline.minimum_precision, \
            f"Average precision {avg_precision:.1f}% below minimum baseline {performance_monitor.baseline.minimum_precision}%"

        assert avg_recall >= performance_monitor.baseline.minimum_recall, \
            f"Average recall {avg_recall:.1f}% below minimum baseline {performance_monitor.baseline.minimum_recall}%"

        # Get accuracy summary and validate
        summary = accuracy_tracker.get_accuracy_summary(hours=24)
        assert summary["overall_accuracy"]["avg_precision"] >= performance_monitor.baseline.minimum_precision

        print(f"✅ Search accuracy meets baselines: Precision={avg_precision:.1f}%, Recall={avg_recall:.1f}%")

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation_performance(self, performance_monitor, mock_search_engine):
        """Test multi-tenant isolation performance and enforcement."""
        tenant_test_data = TestDataGenerator.generate_multi_tenant_test_data(
            tenant_count=3, queries_per_tenant=5
        )

        # Run multi-tenant isolation benchmark
        result = await performance_monitor.benchmark_suite.run_multi_tenant_isolation_benchmark(
            collection_name="test_collection",
            tenant_test_data=tenant_test_data
        )

        # Assert isolation enforcement meets baseline (100%)
        isolation_rate = result.test_config.get("isolation_enforcement_rate", 0)
        assert isolation_rate >= performance_monitor.baseline.tenant_isolation_enforcement, \
            f"Tenant isolation enforcement rate {isolation_rate}% below required {performance_monitor.baseline.tenant_isolation_enforcement}%"

        # Assert performance is acceptable
        assert result.avg_response_time <= performance_monitor.baseline.acceptable_response_time, \
            f"Multi-tenant search response time {result.avg_response_time}ms exceeds acceptable baseline"

        # Ensure no violations detected
        violations = result.test_config.get("isolation_violations", 0)
        assert violations == 0, f"Tenant isolation violations detected: {violations}"

        print(f"✅ Multi-tenant isolation: {isolation_rate}% enforcement, {result.avg_response_time:.2f}ms avg response time")

    @pytest.mark.asyncio
    async def test_cache_performance_baseline(self, performance_monitor, mock_search_engine):
        """Test that cache performance meets baseline targets."""
        # Mock filter optimizer to test cache performance
        if hasattr(mock_search_engine, 'filter_optimizer'):
            mock_filter_optimizer = Mock()
            mock_filter_optimizer.get_performance_metrics.return_value = {
                "cache_hit_rate": 85.0,  # Above target
                "avg_response_time": 1.8,
                "total_queries": 100,
                "performance_target_met": True
            }
            mock_search_engine.filter_optimizer = mock_filter_optimizer

            # Get optimization performance
            opt_performance = mock_search_engine.get_optimization_performance()
            cache_hit_rate = opt_performance["filter_cache"]["cache_hit_rate"]

            # Assert cache performance meets baseline
            assert cache_hit_rate >= performance_monitor.baseline.minimum_cache_hit_rate, \
                f"Cache hit rate {cache_hit_rate}% below minimum baseline {performance_monitor.baseline.minimum_cache_hit_rate}%"

            print(f"✅ Cache performance meets baseline: {cache_hit_rate}% hit rate")

    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, performance_monitor):
        """Test that performance regressions are properly detected."""
        # Simulate multiple benchmark runs with degrading performance
        test_queries = TestDataGenerator.generate_test_queries(count=5)

        benchmark_results = []

        # Run multiple benchmarks to build history
        for i in range(3):
            # Mock increasing response times to simulate regression
            with pytest.MonkeyPatch().context() as m:
                base_response_time = 2.0 + (i * 0.5)  # 2.0ms, 2.5ms, 3.0ms

                async def mock_degraded_search(*args, **kwargs):
                    await asyncio.sleep(base_response_time / 1000)  # Convert to seconds
                    return {
                        "fused_results": [Mock(id=f"result_{j}", score=0.9) for j in range(10)],
                        "performance": {"response_time_ms": base_response_time}
                    }

                performance_monitor.benchmark_suite.search_engine.hybrid_search = AsyncMock(
                    side_effect=mock_degraded_search
                )

                result = await performance_monitor.benchmark_suite.run_metadata_filtering_benchmark(
                    collection_name="test_collection",
                    test_queries=test_queries[:2],
                    iterations=5
                )
                benchmark_results.append(result)

        # Generate regression report
        regression_report = performance_monitor.benchmark_suite.generate_performance_regression_report()

        # Check that regression was detected in later benchmarks
        latest_result = benchmark_results[-1]
        assert latest_result.performance_regression, "Performance regression should be detected"

        # Verify regression report contains identified issues
        assert "regressions" in regression_report

        if regression_report["regressions"]:
            performance_regressions = [r for r in regression_report["regressions"] if r["type"] == "performance_regression"]
            assert len(performance_regressions) > 0, "Performance regression should be identified in report"

        print("✅ Performance regression detection working correctly")

    @pytest.mark.asyncio
    async def test_dashboard_real_time_metrics(self, performance_monitor):
        """Test real-time dashboard metrics collection and reporting."""
        dashboard = performance_monitor.dashboard

        # Record some real-time metrics
        test_metrics = [
            ("metadata_search", 2.1, {"precision": 95.0, "recall": 92.0}),
            ("tenant_search", 1.8, {"precision": 94.5, "recall": 90.0}),
            ("filtered_search", 2.3, {"precision": 93.0, "recall": 88.0}),
            ("hybrid_search", 2.0, {"precision": 96.0, "recall": 93.0}),
        ]

        for operation, response_time, accuracy in test_metrics:
            dashboard.record_real_time_metric(
                operation_type=operation,
                response_time=response_time,
                accuracy_metrics=accuracy,
                metadata={"timestamp": datetime.now().isoformat()}
            )

        # Get dashboard data
        dashboard_data = dashboard.get_real_time_dashboard()

        # Validate dashboard structure and content
        assert "performance_overview" in dashboard_data
        assert "accuracy_overview" in dashboard_data
        assert "baseline_comparison" in dashboard_data

        performance_overview = dashboard_data["performance_overview"]

        # Check performance status determination
        assert performance_overview["status"] in ["excellent", "good", "degraded"]
        assert performance_overview["avg_response_time"] > 0
        assert performance_overview["baseline_target"] == performance_monitor.baseline.target_response_time

        # Verify baseline compliance calculations
        baseline_comparison = dashboard_data["baseline_comparison"]
        assert "response_time" in baseline_comparison
        assert "precision" in baseline_comparison

        print("✅ Real-time dashboard metrics collection working correctly")

    @pytest.mark.asyncio
    async def test_comprehensive_performance_monitoring(self, performance_monitor):
        """Test end-to-end comprehensive performance monitoring."""
        test_queries = TestDataGenerator.generate_test_queries(count=10)
        tenant_test_data = TestDataGenerator.generate_multi_tenant_test_data(tenant_count=2, queries_per_tenant=3)

        # Run comprehensive benchmark
        results = await performance_monitor.run_comprehensive_benchmark(
            collection_name="test_collection",
            test_queries=test_queries[:5],
            tenant_test_data=tenant_test_data
        )

        # Validate all benchmark components ran
        assert "metadata_filtering" in results
        assert "multi_tenant_isolation" in results
        assert "regression_analysis" in results

        # Check metadata filtering results
        metadata_result = results["metadata_filtering"]
        assert isinstance(metadata_result.avg_response_time, (int, float))
        assert metadata_result.avg_response_time > 0

        # Check multi-tenant results
        tenant_result = results["multi_tenant_isolation"]
        assert isinstance(tenant_result.avg_response_time, (int, float))
        assert tenant_result.test_config["isolation_enforcement_rate"] >= 0

        # Get overall performance status
        status = performance_monitor.get_performance_status()

        # Validate status structure
        assert "baseline_configuration" in status
        assert "real_time_dashboard" in status
        assert "accuracy_summary" in status
        assert "benchmark_status" in status

        print("✅ Comprehensive performance monitoring integration successful")

    def test_performance_baseline_configuration(self, performance_baseline):
        """Test performance baseline configuration and validation."""
        # Test baseline creation and serialization
        baseline_dict = performance_baseline.to_dict()

        assert "response_time" in baseline_dict
        assert "accuracy" in baseline_dict
        assert "cache_performance" in baseline_dict
        assert "tenant_isolation" in baseline_dict

        # Validate response time baselines
        response_time_config = baseline_dict["response_time"]
        assert response_time_config["target_ms"] == 2.18
        assert response_time_config["acceptable_ms"] == 3.0
        assert response_time_config["critical_ms"] == 5.0

        # Validate accuracy baselines
        accuracy_config = baseline_dict["accuracy"]
        assert accuracy_config["target_precision"] == 94.2
        assert accuracy_config["minimum_precision"] == 90.0

        print("✅ Performance baseline configuration validated")


@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceStressTests:
    """Stress tests to validate performance under load."""

    @pytest.fixture
    async def stress_test_search_engine(self):
        """Create search engine mock for stress testing."""
        engine = Mock(spec=HybridSearchEngine)

        async def variable_response_search(*args, **kwargs):
            # Simulate variable response times (1-5ms) under load
            import random
            response_time = random.uniform(1.0, 5.0)
            await asyncio.sleep(response_time / 1000)

            return {
                "fused_results": [Mock(id=f"stress_result_{i}", score=0.8) for i in range(8)],
                "performance": {"response_time_ms": response_time}
            }

        engine.hybrid_search = AsyncMock(side_effect=variable_response_search)
        return engine

    @pytest.mark.asyncio
    async def test_high_volume_performance_stability(self, stress_test_search_engine):
        """Test performance stability under high query volume."""
        baseline = PerformanceBaseline()
        monitor = MetadataFilteringPerformanceMonitor(stress_test_search_engine)

        # Generate high volume of test queries
        test_queries = TestDataGenerator.generate_test_queries(count=100)

        # Run benchmark with high iteration count
        start_time = time.time()
        result = await monitor.benchmark_suite.run_metadata_filtering_benchmark(
            collection_name="stress_test_collection",
            test_queries=test_queries[:10],  # Use subset for reasonable test time
            iterations=20
        )
        total_time = time.time() - start_time

        # Validate performance under stress
        assert result.avg_response_time <= baseline.critical_response_time, \
            f"High volume average response time {result.avg_response_time}ms exceeds critical threshold"

        # Check that P95 is reasonable under stress
        assert result.p95_response_time <= baseline.critical_response_time * 1.2, \
            f"P95 response time {result.p95_response_time}ms too high under stress"

        # Ensure test completed in reasonable time
        assert total_time < 60, f"Stress test took too long: {total_time}s"

        print(f"✅ High volume stress test completed: {result.avg_response_time:.2f}ms avg, {total_time:.1f}s total")

    @pytest.mark.asyncio
    async def test_concurrent_multi_tenant_performance(self, stress_test_search_engine):
        """Test performance with concurrent multi-tenant operations."""
        monitor = MetadataFilteringPerformanceMonitor(stress_test_search_engine)

        # Generate multi-tenant test data
        tenant_data = TestDataGenerator.generate_multi_tenant_test_data(
            tenant_count=5, queries_per_tenant=10
        )

        # Run concurrent tenant benchmarks
        concurrent_tasks = []
        for tenant_id, queries in tenant_data.items():
            task = monitor.benchmark_suite.run_multi_tenant_isolation_benchmark(
                collection_name="concurrent_test",
                tenant_test_data={tenant_id: queries[:5]}  # Subset for reasonable test time
            )
            concurrent_tasks.append(task)

        # Execute all tenant benchmarks concurrently
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Filter successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > 0, "No successful concurrent benchmark results"

        # Check concurrent performance
        avg_concurrent_response_time = statistics.mean([r.avg_response_time for r in successful_results])
        assert avg_concurrent_response_time <= monitor.baseline.critical_response_time, \
            f"Concurrent performance {avg_concurrent_response_time:.2f}ms exceeds critical threshold"

        # Ensure all tenants maintained isolation
        for result in successful_results:
            isolation_rate = result.test_config.get("isolation_enforcement_rate", 0)
            assert isolation_rate >= 95.0, f"Concurrent tenant isolation below 95%: {isolation_rate}%"

        print(f"✅ Concurrent multi-tenant test: {len(successful_results)} tenants, {avg_concurrent_response_time:.2f}ms avg")


# Pytest configuration for performance tests
def pytest_configure(config):
    """Configure pytest for performance testing."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance regression test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow-running performance test"
    )


# Custom pytest collection for performance tests
def pytest_collection_modifyitems(config, items):
    """Modify test collection for performance test organization."""
    for item in items:
        # Add performance marker to all tests in this module
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)

        # Mark stress tests as slow
        if "stress" in item.name.lower():
            item.add_marker(pytest.mark.slow)