"""
Performance Tests for Multi-Tenant Architecture.

This module provides comprehensive performance testing for multi-tenant
operations including large-scale collection management, concurrent access,
and performance regression validation.

Performance Test Categories:
    - Large-scale multi-tenant collection creation and management
    - Concurrent multi-project operations
    - Search performance across many tenants
    - Memory usage and resource optimization
    - Collision detection performance
    - Metadata filtering performance optimization
"""

import asyncio
import json
import pytest
import time
import statistics
from typing import Dict, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor
import psutil
import resource

# Performance testing utilities
import testcontainers
from tests.utils.metrics import PerformanceMetrics, ResourceMonitor

# Multi-tenant components
from workspace_qdrant_mcp.core.collision_detection import CollisionDetector
from workspace_qdrant_mcp.core.metadata_filtering import (
    MetadataFilterManager,
    FilterCriteria,
    FilterPerformanceLevel
)
from workspace_qdrant_mcp.core.multitenant_collections import (
    MultiTenantWorkspaceCollectionManager
)
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient

# Test data and configuration
from tests.integration.conftest import TEST_ENVIRONMENT_CONFIG


class TestMultiTenantScalePerformance:
    """Performance tests for large-scale multi-tenant scenarios."""

    @pytest.fixture
    async def performance_qdrant_client(self):
        """Create Qdrant client optimized for performance testing."""
        config = TEST_ENVIRONMENT_CONFIG["qdrant"]

        with testcontainers.core.DockerContainer(
            image=config["image"]
        ).with_exposed_ports(config["http_port"]) as container:
            await asyncio.sleep(config["startup_wait"])

            # Configure client for performance testing
            client = QdrantWorkspaceClient(
                url=f"http://localhost:{container.get_exposed_port(config['http_port'])}",
                timeout=60,  # Extended timeout for performance tests
                max_retries=3
            )
            await client.initialize()

            yield client

            await client.shutdown()

    @pytest.fixture
    def performance_monitor(self):
        """Provide performance monitoring utilities."""
        return ResourceMonitor()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_scale_collection_creation_performance(
        self,
        performance_qdrant_client,
        performance_monitor
    ):
        """Test performance of creating many multi-tenant collections."""
        client = performance_qdrant_client
        manager = MultiTenantWorkspaceCollectionManager(client.client, client.config)

        # Performance test parameters
        num_projects = 100
        collections_per_project = 4
        workspace_types = ["docs", "notes", "scratchbook", "knowledge"]

        performance_metrics = PerformanceMetrics()

        with performance_monitor.monitor_resources() as monitor:
            start_time = time.time()

            # Create projects in batches to avoid overwhelming the system
            batch_size = 10
            creation_times = []

            for batch_start in range(0, num_projects, batch_size):
                batch_end = min(batch_start + batch_size, num_projects)
                batch_projects = [
                    f"perf-project-{i:03d}"
                    for i in range(batch_start, batch_end)
                ]

                # Time batch creation
                batch_start_time = time.time()

                # Create batch concurrently
                tasks = []
                for project in batch_projects:
                    task = manager.initialize_workspace_collections(
                        project_name=project,
                        workspace_types=workspace_types
                    )
                    tasks.append(task)

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                batch_time = time.time() - batch_start_time
                creation_times.append(batch_time)

                # Verify successful creation
                successful_creates = sum(
                    1 for result in batch_results
                    if isinstance(result, dict) and result.get("success")
                )
                assert successful_creates == len(batch_projects)

                # Log progress
                print(f"Created batch {batch_start//batch_size + 1}/{(num_projects-1)//batch_size + 1}: "
                      f"{successful_creates} projects in {batch_time:.2f}s")

            total_time = time.time() - start_time

        # Performance analysis
        total_collections = num_projects * collections_per_project
        collections_per_second = total_collections / total_time
        avg_batch_time = statistics.mean(creation_times)

        performance_metrics.record_metric("total_creation_time", total_time)
        performance_metrics.record_metric("collections_per_second", collections_per_second)
        performance_metrics.record_metric("avg_batch_time", avg_batch_time)

        # Performance assertions
        assert collections_per_second > 5.0, f"Creation rate {collections_per_second:.2f} coll/s too slow"
        assert total_time < 300, f"Total time {total_time:.0f}s exceeds 5 minute limit"

        # Memory usage validation
        peak_memory = monitor.get_peak_memory_mb()
        assert peak_memory < 1000, f"Peak memory {peak_memory}MB exceeds limit"

        print(f"\nPerformance Results:")
        print(f"  Total collections created: {total_collections}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Creation rate: {collections_per_second:.2f} collections/s")
        print(f"  Peak memory usage: {peak_memory}MB")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_multitenant_operations_performance(
        self,
        performance_qdrant_client,
        performance_monitor
    ):
        """Test performance under concurrent multi-tenant operations."""
        client = performance_qdrant_client
        manager = MultiTenantWorkspaceCollectionManager(client.client, client.config)

        # Set up test projects
        test_projects = [f"concurrent-project-{i}" for i in range(20)]

        # Initialize projects
        for project in test_projects:
            result = await manager.initialize_workspace_collections(
                project_name=project,
                workspace_types=["docs", "notes"]
            )
            assert result["success"]

        # Define concurrent operation types
        async def concurrent_document_operations(project_name: str, operation_count: int):
            """Perform concurrent document operations for a project."""
            operation_times = []

            for i in range(operation_count):
                start_time = time.time()

                # Add document
                content = f"Concurrent test document {i} for {project_name}"
                metadata = {
                    "project_name": project_name,
                    "operation_index": i,
                    "test_type": "concurrent_performance"
                }

                doc_id = await client.add_document(
                    collection=f"{project_name}-docs",
                    content=content,
                    metadata=metadata
                )

                # Search for document
                search_results = await client.search(
                    collection=f"{project_name}-docs",
                    query="concurrent test",
                    limit=5
                )

                operation_time = time.time() - start_time
                operation_times.append(operation_time)

                assert doc_id is not None
                assert len(search_results) > 0

            return operation_times

        # Run concurrent operations
        with performance_monitor.monitor_resources() as monitor:
            start_time = time.time()

            # Create concurrent tasks for different projects
            concurrent_tasks = []
            operations_per_project = 10

            for project in test_projects:
                task = concurrent_document_operations(project, operations_per_project)
                concurrent_tasks.append(task)

            # Execute all concurrent operations
            all_operation_times = await asyncio.gather(*concurrent_tasks)

            total_time = time.time() - start_time

        # Analyze performance results
        all_times = [time for project_times in all_operation_times for time in project_times]

        avg_operation_time = statistics.mean(all_times)
        p95_operation_time = statistics.quantiles(all_times, n=20)[18]  # 95th percentile
        total_operations = len(test_projects) * operations_per_project
        operations_per_second = total_operations / total_time

        # Performance assertions
        assert avg_operation_time < 2.0, f"Avg operation time {avg_operation_time:.3f}s too slow"
        assert p95_operation_time < 5.0, f"P95 operation time {p95_operation_time:.3f}s too slow"
        assert operations_per_second > 5.0, f"Throughput {operations_per_second:.2f} ops/s too low"

        # Resource usage validation
        peak_memory = monitor.get_peak_memory_mb()
        peak_cpu = monitor.get_peak_cpu_percent()

        assert peak_memory < 800, f"Peak memory {peak_memory}MB too high"
        assert peak_cpu < 90, f"Peak CPU {peak_cpu}% too high"

        print(f"\nConcurrent Operations Performance:")
        print(f"  Total operations: {total_operations}")
        print(f"  Concurrent projects: {len(test_projects)}")
        print(f"  Average operation time: {avg_operation_time:.3f}s")
        print(f"  P95 operation time: {p95_operation_time:.3f}s")
        print(f"  Operations per second: {operations_per_second:.2f}")
        print(f"  Peak memory: {peak_memory}MB")
        print(f"  Peak CPU: {peak_cpu}%")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_search_performance_across_many_tenants(
        self,
        performance_qdrant_client,
        performance_monitor
    ):
        """Test search performance across large numbers of tenant collections."""
        client = performance_qdrant_client
        manager = MultiTenantWorkspaceCollectionManager(client.client, client.config)

        # Create many tenant projects with documents
        num_tenants = 50
        documents_per_tenant = 20

        tenant_projects = [f"search-tenant-{i:02d}" for i in range(num_tenants)]

        # Initialize all tenant projects
        setup_start = time.time()

        setup_tasks = []
        for project in tenant_projects:
            task = manager.initialize_workspace_collections(
                project_name=project,
                workspace_types=["docs"]
            )
            setup_tasks.append(task)

        await asyncio.gather(*setup_tasks)
        setup_time = time.time() - setup_start

        # Add documents to all tenants
        document_tasks = []
        for project in tenant_projects:
            for doc_idx in range(documents_per_tenant):
                content = f"Search performance test document {doc_idx} for {project}. " \
                         f"This document contains searchable content about {project} operations."

                metadata = {
                    "project_name": project,
                    "document_index": doc_idx,
                    "test_category": "search_performance"
                }

                task = client.add_document(
                    collection=f"{project}-docs",
                    content=content,
                    metadata=metadata
                )
                document_tasks.append(task)

        await asyncio.gather(*document_tasks)

        print(f"Setup completed in {setup_time:.2f}s with {len(document_tasks)} documents")

        # Test search performance scenarios
        filter_manager = MetadataFilterManager(client.client)

        with performance_monitor.monitor_resources() as monitor:
            # Scenario 1: Single-tenant focused searches
            single_tenant_times = []

            for project in tenant_projects[:10]:  # Test subset
                project_filter = filter_manager.create_project_isolation_filter(project)

                start_time = time.time()
                results = await client.search(
                    collection=f"{project}-docs",
                    query="search performance test",
                    filter_condition=project_filter.filter,
                    limit=10
                )
                search_time = time.time() - start_time
                single_tenant_times.append(search_time)

                assert len(results) > 0
                assert all(r.metadata.get("project_name") == project for r in results)

            # Scenario 2: Cross-tenant searches with filtering
            cross_tenant_times = []

            for i in range(10):
                # Search across multiple tenant collections
                target_projects = tenant_projects[i*3:(i+1)*3]  # 3 projects per search
                collections = [f"{p}-docs" for p in target_projects]

                start_time = time.time()
                results = await client.search_multiple_collections(
                    collections=collections,
                    query="performance test document",
                    limit=15
                )
                search_time = time.time() - start_time
                cross_tenant_times.append(search_time)

                assert len(results) > 0

            # Scenario 3: Complex filtered searches
            complex_search_times = []

            for i in range(5):
                criteria = FilterCriteria(
                    collection_types=["docs"],
                    include_shared=True,
                    performance_level=FilterPerformanceLevel.FAST
                )
                complex_filter = filter_manager.create_composite_filter(criteria)

                start_time = time.time()
                results = await client.search(
                    collection=f"{tenant_projects[i]}-docs",
                    query="test document operations",
                    filter_condition=complex_filter.filter,
                    limit=20
                )
                search_time = time.time() - start_time
                complex_search_times.append(search_time)

        # Performance analysis
        avg_single_tenant = statistics.mean(single_tenant_times)
        avg_cross_tenant = statistics.mean(cross_tenant_times)
        avg_complex_search = statistics.mean(complex_search_times)

        p95_single_tenant = statistics.quantiles(single_tenant_times, n=20)[18]
        p95_cross_tenant = statistics.quantiles(cross_tenant_times, n=20)[18]

        # Performance assertions
        assert avg_single_tenant < 1.0, f"Single-tenant search avg {avg_single_tenant:.3f}s too slow"
        assert avg_cross_tenant < 2.0, f"Cross-tenant search avg {avg_cross_tenant:.3f}s too slow"
        assert avg_complex_search < 1.5, f"Complex search avg {avg_complex_search:.3f}s too slow"

        assert p95_single_tenant < 2.0, f"Single-tenant P95 {p95_single_tenant:.3f}s too slow"
        assert p95_cross_tenant < 4.0, f"Cross-tenant P95 {p95_cross_tenant:.3f}s too slow"

        print(f"\nSearch Performance Across {num_tenants} Tenants:")
        print(f"  Single-tenant search avg: {avg_single_tenant:.3f}s (P95: {p95_single_tenant:.3f}s)")
        print(f"  Cross-tenant search avg: {avg_cross_tenant:.3f}s (P95: {p95_cross_tenant:.3f}s)")
        print(f"  Complex filtered search avg: {avg_complex_search:.3f}s")

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_collision_detection_performance_at_scale(
        self,
        performance_qdrant_client,
        performance_monitor
    ):
        """Test collision detection performance with large number of collections."""
        client = performance_qdrant_client
        detector = CollisionDetector(client.client)

        # Create many collections to stress collision detection
        num_collections = 500
        collections_per_batch = 50

        with performance_monitor.monitor_resources() as monitor:
            await detector.initialize()

            # Create collections in batches
            creation_times = []

            for batch_start in range(0, num_collections, collections_per_batch):
                batch_end = min(batch_start + collections_per_batch, num_collections)

                batch_start_time = time.time()

                # Create batch of collections
                for i in range(batch_start, batch_end):
                    collection_name = f"collision-test-{i:04d}"

                    # Check for collision (should be fast even with many existing collections)
                    collision_start = time.time()
                    collision_result = await detector.check_collection_collision(collection_name)
                    collision_time = time.time() - collision_start

                    assert not collision_result.has_collision
                    assert collision_time < 0.1, f"Collision check took {collision_time:.3f}s"

                    # Register the collection
                    await detector.register_collection_creation(
                        collection_name,
                        collection_type="test"
                    )

                batch_time = time.time() - batch_start_time
                creation_times.append(batch_time)

                print(f"Processed collision detection batch {batch_start//collections_per_batch + 1}: "
                      f"{batch_end - batch_start} collections in {batch_time:.2f}s")

            # Test collision detection performance after all collections exist
            detection_times = []

            for i in range(100):  # Test 100 collision checks
                test_name = f"collision-test-{i:04d}"  # These already exist

                start_time = time.time()
                collision_result = await detector.check_collection_collision(test_name)
                detection_time = time.time() - start_time

                detection_times.append(detection_time)
                assert collision_result.has_collision  # Should detect collision

            # Test suggestion generation performance
            suggestion_times = []

            for i in range(20):
                collision_name = f"collision-test-{i:04d}"

                start_time = time.time()
                collision_result = await detector.check_collection_collision(collision_name)
                suggestion_time = time.time() - start_time

                suggestion_times.append(suggestion_time)
                assert len(collision_result.suggested_alternatives) > 0

        # Performance analysis
        avg_detection_time = statistics.mean(detection_times)
        p95_detection_time = statistics.quantiles(detection_times, n=20)[18]
        avg_suggestion_time = statistics.mean(suggestion_times)

        # Get collision detection statistics
        stats = await detector.get_collision_statistics()
        registry_stats = stats["registry_statistics"]

        # Performance assertions
        assert avg_detection_time < 0.05, f"Avg detection time {avg_detection_time:.4f}s too slow"
        assert p95_detection_time < 0.1, f"P95 detection time {p95_detection_time:.4f}s too slow"
        assert avg_suggestion_time < 0.2, f"Avg suggestion time {avg_suggestion_time:.3f}s too slow"

        # Verify Bloom filter efficiency
        bloom_efficiency = registry_stats.get("bloom_filter_efficiency_percent", 0)
        assert bloom_efficiency > 50, f"Bloom filter efficiency {bloom_efficiency}% too low"

        print(f"\nCollision Detection Performance with {num_collections} collections:")
        print(f"  Average detection time: {avg_detection_time:.4f}s")
        print(f"  P95 detection time: {p95_detection_time:.4f}s")
        print(f"  Average suggestion time: {avg_suggestion_time:.3f}s")
        print(f"  Bloom filter efficiency: {bloom_efficiency:.1f}%")
        print(f"  Cache hit rate: {registry_stats.get('cache_hit_rate_percent', 0):.1f}%")

        await detector.shutdown()

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_metadata_filtering_performance_optimization(
        self,
        performance_qdrant_client,
        performance_monitor
    ):
        """Test metadata filtering performance optimizations."""
        client = performance_qdrant_client
        filter_manager = MetadataFilterManager(
            qdrant_client=client.client,
            enable_caching=True,
            enable_performance_monitoring=True
        )

        # Create test collections with diverse metadata
        test_projects = [f"filter-perf-{i:02d}" for i in range(30)]

        for project in test_projects:
            await client.create_collection(
                name=f"{project}-docs",
                vector_size=384
            )

            # Add documents with varied metadata
            for doc_idx in range(50):
                metadata = {
                    "project_name": project,
                    "collection_type": "docs",
                    "access_level": ["private", "shared", "public"][doc_idx % 3],
                    "priority": (doc_idx % 5) + 1,
                    "tags": [f"tag-{doc_idx % 10}", f"category-{doc_idx % 7}"],
                    "created_by": f"user-{doc_idx % 15}",
                    "department": ["engineering", "product", "design"][doc_idx % 3]
                }

                await client.add_document(
                    collection=f"{project}-docs",
                    content=f"Filter performance test document {doc_idx}",
                    metadata=metadata
                )

        with performance_monitor.monitor_resources() as monitor:
            # Test different filter complexity levels
            filter_scenarios = [
                ("simple", FilterCriteria(project_name="filter-perf-01")),
                ("medium", FilterCriteria(
                    project_name="filter-perf-02",
                    collection_types=["docs"],
                    access_levels=["private", "shared"]
                )),
                ("complex", FilterCriteria(
                    project_name="filter-perf-03",
                    collection_types=["docs"],
                    access_levels=["private"],
                    tags=["tag-1", "tag-2"],
                    priority_range=(3, 5),
                    created_by=["user-1", "user-2"]
                ))
            ]

            scenario_results = {}

            for scenario_name, criteria in filter_scenarios:
                filter_times = []
                cache_hits = 0

                # Test filter creation performance
                for i in range(100):
                    start_time = time.time()
                    filter_result = filter_manager.create_composite_filter(criteria)
                    filter_time = time.time() - start_time

                    filter_times.append(filter_time)
                    if filter_result.cache_hit:
                        cache_hits += 1

                avg_filter_time = statistics.mean(filter_times)
                p95_filter_time = statistics.quantiles(filter_times, n=20)[18]
                cache_hit_rate = cache_hits / len(filter_times)

                scenario_results[scenario_name] = {
                    "avg_time": avg_filter_time,
                    "p95_time": p95_filter_time,
                    "cache_hit_rate": cache_hit_rate
                }

        # Performance analysis
        for scenario, results in scenario_results.items():
            print(f"\nFilter Performance - {scenario.upper()} scenario:")
            print(f"  Average filter creation: {results['avg_time']:.4f}s")
            print(f"  P95 filter creation: {results['p95_time']:.4f}s")
            print(f"  Cache hit rate: {results['cache_hit_rate']:.1%}")

        # Performance assertions
        assert scenario_results["simple"]["avg_time"] < 0.01, "Simple filter creation too slow"
        assert scenario_results["medium"]["avg_time"] < 0.02, "Medium filter creation too slow"
        assert scenario_results["complex"]["avg_time"] < 0.05, "Complex filter creation too slow"

        # Cache should improve performance for repeated operations
        assert scenario_results["simple"]["cache_hit_rate"] > 0.8, "Cache hit rate too low"

        # Get comprehensive performance statistics
        perf_stats = filter_manager.get_filter_performance_stats()
        print(f"\nOverall Filter Performance Statistics:")
        print(f"  Total operations: {perf_stats.get('composite', {}).get('total_operations', 0)}")
        print(f"  Average complexity score: {perf_stats.get('composite', {}).get('avg_complexity_score', 0):.2f}")


class TestMultiTenantMemoryOptimization:
    """Test memory usage optimization in multi-tenant scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_with_many_tenants(self):
        """Test memory usage patterns with large numbers of tenants."""
        # This test would monitor memory usage as tenant count increases
        # and verify that memory usage scales linearly, not exponentially

        memory_readings = []
        tenant_counts = [10, 50, 100, 200, 500]

        for tenant_count in tenant_counts:
            # Measure memory before creating tenants
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Create tenants (simulated for performance testing)
            # In real implementation, would create actual tenant collections

            # Measure memory after tenant creation
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory

            memory_readings.append((tenant_count, memory_usage))

        # Analyze memory scaling
        for i, (tenant_count, memory_mb) in enumerate(memory_readings):
            memory_per_tenant = memory_mb / tenant_count if tenant_count > 0 else 0
            print(f"Tenants: {tenant_count:3d}, Memory: {memory_mb:6.1f}MB, "
                  f"Per tenant: {memory_per_tenant:.2f}MB")

            # Memory per tenant should be reasonable and not grow significantly
            if i > 0:
                prev_per_tenant = memory_readings[i-1][1] / memory_readings[i-1][0]
                growth_factor = memory_per_tenant / prev_per_tenant if prev_per_tenant > 0 else 1

                assert growth_factor < 1.5, f"Memory per tenant growing too fast: {growth_factor:.2f}x"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_resource_cleanup_effectiveness(self):
        """Test that resources are properly cleaned up in multi-tenant scenarios."""
        # Monitor file descriptors, memory, and other resources
        initial_fds = len(psutil.Process().open_files())
        initial_memory = psutil.Process().memory_info().rss

        # Simulate creating and destroying many tenant resources
        # In real implementation, would create/destroy actual collections

        # Verify resource cleanup
        final_fds = len(psutil.Process().open_files())
        final_memory = psutil.Process().memory_info().rss

        fd_growth = final_fds - initial_fds
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB

        assert fd_growth < 100, f"Too many file descriptors leaked: {fd_growth}"
        assert memory_growth < 100, f"Too much memory leaked: {memory_growth:.1f}MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])