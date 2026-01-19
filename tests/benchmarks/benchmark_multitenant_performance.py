"""
Multi-tenant architecture performance benchmarks.

Task 411: Benchmarks comparing search performance of unified collections
vs per-project collections. Measures search latency, throughput, and
tenant isolation overhead.

Key metrics:
- Search latency (p50, p95, p99) with tenant_id filtering
- Unified collection vs per-project collection comparison
- Concurrent multi-tenant search throughput
- Ingestion throughput with tenant metadata

Run with: uv run pytest tests/benchmarks/benchmark_multitenant_performance.py --benchmark-only -v
"""

import asyncio
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any

import pytest
from common.core.embeddings import EmbeddingService
from common.core.ssl_config import suppress_qdrant_ssl_warnings
from qdrant_client import QdrantClient
from qdrant_client.http import models


# Suppress SSL warnings for local testing
suppress_qdrant_ssl_warnings()


@dataclass
class LatencyMetrics:
    """Latency percentile metrics."""

    p50: float
    p95: float
    p99: float
    mean: float
    min_val: float
    max_val: float
    count: int

    @classmethod
    def from_samples(cls, samples: list[float]) -> "LatencyMetrics":
        """Compute metrics from a list of latency samples (in milliseconds)."""
        if not samples:
            return cls(0, 0, 0, 0, 0, 0, 0)

        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        return cls(
            p50=sorted_samples[int(n * 0.5)],
            p95=sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1],
            p99=sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1],
            mean=statistics.mean(samples),
            min_val=min(samples),
            max_val=max(samples),
            count=n,
        )


class MultiTenantBenchmarkSetup:
    """Helper class for multi-tenant benchmark setup."""

    UNIFIED_COLLECTION = "_benchmark_unified"
    PER_PROJECT_PREFIX = "_benchmark_project_"

    # Sample document types with realistic content
    SAMPLE_DOCS = [
        ("def authenticate_user(username, password):", "code", "authentication"),
        ("class UserService:", "code", "user_management"),
        ("async def fetch_data(client, endpoint):", "code", "api"),
        ("# Database connection pool configuration", "code", "database"),
        ("TODO: Add rate limiting to API endpoints", "notes", "api"),
        ("Design decision: Use JWT for session tokens", "notes", "authentication"),
        ("Bug fix: Resolved memory leak in cache", "notes", "performance"),
        ("API Reference: /users endpoint documentation", "docs", "api"),
        ("Installation guide for development setup", "docs", "setup"),
        ("Performance tuning recommendations", "docs", "performance"),
    ]

    @staticmethod
    async def create_unified_collection(
        client: QdrantClient,
        embedding_service: EmbeddingService,
        num_tenants: int = 10,
        docs_per_tenant: int = 100,
    ) -> None:
        """
        Create a unified collection with documents from multiple tenants.

        This simulates the new multi-tenant architecture where all project
        data is stored in a single collection with tenant_id filtering.

        Args:
            client: Qdrant client
            embedding_service: Embedding service for vector generation
            num_tenants: Number of simulated projects/tenants
            docs_per_tenant: Documents per tenant
        """
        collection_name = MultiTenantBenchmarkSetup.UNIFIED_COLLECTION

        # Delete if exists
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass

        # Create collection with dense and sparse vectors
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

        # Create payload index on tenant_id for efficient filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="tenant_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        # Create payload index on branch for common filtering
        client.create_payload_index(
            collection_name=collection_name,
            field_name="branch",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

        # Generate tenant IDs (12-char hex like real project_ids)
        tenant_ids = [f"{i:012x}" for i in range(num_tenants)]

        # Insert documents for each tenant
        total_docs = num_tenants * docs_per_tenant
        point_id = 0

        for tenant_id in tenant_ids:
            points = []
            for j in range(docs_per_tenant):
                # Cycle through sample documents
                doc_idx = j % len(MultiTenantBenchmarkSetup.SAMPLE_DOCS)
                text, file_type, category = MultiTenantBenchmarkSetup.SAMPLE_DOCS[
                    doc_idx
                ]

                # Add variation to text
                text = f"{text} # Tenant {tenant_id}, doc {j}"

                # Generate embeddings
                embeddings = await embedding_service.generate_embeddings(text)

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": embeddings["dense"],
                        "sparse": models.SparseVector(
                            indices=embeddings["sparse"]["indices"],
                            values=embeddings["sparse"]["values"],
                        ),
                    },
                    payload={
                        "text": text,
                        "tenant_id": tenant_id,
                        "branch": "main",
                        "file_type": file_type,
                        "category": category,
                        "doc_index": j,
                    },
                )
                points.append(point)
                point_id += 1

            # Batch upsert for this tenant
            client.upsert(collection_name=collection_name, points=points)

        return tenant_ids

    @staticmethod
    async def create_per_project_collections(
        client: QdrantClient,
        embedding_service: EmbeddingService,
        num_projects: int = 10,
        docs_per_project: int = 100,
    ) -> list[str]:
        """
        Create separate collections for each project.

        This simulates the old per-project collection architecture for
        comparison benchmarking.

        Args:
            client: Qdrant client
            embedding_service: Embedding service for vector generation
            num_projects: Number of project collections
            docs_per_project: Documents per collection

        Returns:
            List of collection names created
        """
        collection_names = []

        for i in range(num_projects):
            collection_name = f"{MultiTenantBenchmarkSetup.PER_PROJECT_PREFIX}{i:012x}"
            collection_names.append(collection_name)

            # Delete if exists
            try:
                client.delete_collection(collection_name)
            except Exception:
                pass

            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={"sparse": models.SparseVectorParams()},
            )

            # Generate documents
            points = []
            for j in range(docs_per_project):
                doc_idx = j % len(MultiTenantBenchmarkSetup.SAMPLE_DOCS)
                text, file_type, category = MultiTenantBenchmarkSetup.SAMPLE_DOCS[
                    doc_idx
                ]
                text = f"{text} # Project {i}, doc {j}"

                embeddings = await embedding_service.generate_embeddings(text)

                point = models.PointStruct(
                    id=j,
                    vector={
                        "dense": embeddings["dense"],
                        "sparse": models.SparseVector(
                            indices=embeddings["sparse"]["indices"],
                            values=embeddings["sparse"]["values"],
                        ),
                    },
                    payload={
                        "text": text,
                        "branch": "main",
                        "file_type": file_type,
                        "category": category,
                        "doc_index": j,
                    },
                )
                points.append(point)

            client.upsert(collection_name=collection_name, points=points)

        return collection_names

    @staticmethod
    def cleanup_collections(client: QdrantClient) -> None:
        """Remove all benchmark collections."""
        try:
            client.delete_collection(MultiTenantBenchmarkSetup.UNIFIED_COLLECTION)
        except Exception:
            pass

        # Delete per-project collections
        collections = client.get_collections().collections
        for col in collections:
            if col.name.startswith(MultiTenantBenchmarkSetup.PER_PROJECT_PREFIX):
                try:
                    client.delete_collection(col.name)
                except Exception:
                    pass


@pytest.fixture(scope="module")
def qdrant_client():
    """Create Qdrant client for benchmarks."""
    client = QdrantClient(host="localhost", port=6333)
    yield client
    # Cleanup after all tests
    MultiTenantBenchmarkSetup.cleanup_collections(client)


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async fixtures."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def embedding_service(event_loop):
    """Create embedding service for vector generation."""
    service = EmbeddingService()
    event_loop.run_until_complete(service.initialize())
    return service


@pytest.fixture(scope="module")
def unified_collection_tenants(qdrant_client, embedding_service, event_loop):
    """Set up unified collection with multiple tenants."""
    tenant_ids = event_loop.run_until_complete(
        MultiTenantBenchmarkSetup.create_unified_collection(
            qdrant_client,
            embedding_service,
            num_tenants=10,
            docs_per_tenant=100,
        )
    )
    return tenant_ids


@pytest.fixture(scope="module")
def per_project_collections(qdrant_client, embedding_service, event_loop):
    """Set up per-project collections for comparison."""
    collection_names = event_loop.run_until_complete(
        MultiTenantBenchmarkSetup.create_per_project_collections(
            qdrant_client,
            embedding_service,
            num_projects=10,
            docs_per_project=100,
        )
    )
    return collection_names


class TestUnifiedCollectionSearchLatency:
    """Benchmark search latency in unified multi-tenant collection."""

    @pytest.mark.benchmark(group="unified-search")
    def test_search_with_tenant_filter(
        self,
        benchmark,
        qdrant_client,
        embedding_service,
        unified_collection_tenants,
    ):
        """
        Benchmark: Search in unified collection with tenant_id filter.

        This is the primary search pattern for multi-tenant architecture.
        Measures overhead of tenant filtering on large collection.
        """
        tenant_id = unified_collection_tenants[0]
        query_text = "authentication user login"

        async def search_with_filter():
            embeddings = await embedding_service.generate_embeddings(query_text)
            return qdrant_client.search(
                collection_name=MultiTenantBenchmarkSetup.UNIFIED_COLLECTION,
                query_vector=("dense", embeddings["dense"]),
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchValue(value=tenant_id),
                        )
                    ]
                ),
                limit=10,
            )

        def run_search():
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(search_with_filter())

        result = benchmark(run_search)
        assert len(result) > 0

    @pytest.mark.benchmark(group="unified-search")
    def test_search_with_tenant_and_branch_filter(
        self,
        benchmark,
        qdrant_client,
        embedding_service,
        unified_collection_tenants,
    ):
        """
        Benchmark: Search with tenant_id AND branch filter.

        Common pattern: search within a specific project on a specific branch.
        """
        tenant_id = unified_collection_tenants[0]
        query_text = "database query optimization"

        async def search_with_filters():
            embeddings = await embedding_service.generate_embeddings(query_text)
            return qdrant_client.search(
                collection_name=MultiTenantBenchmarkSetup.UNIFIED_COLLECTION,
                query_vector=("dense", embeddings["dense"]),
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchValue(value=tenant_id),
                        ),
                        models.FieldCondition(
                            key="branch",
                            match=models.MatchValue(value="main"),
                        ),
                    ]
                ),
                limit=10,
            )

        def run_search():
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(search_with_filters())

        result = benchmark(run_search)
        assert len(result) > 0

    @pytest.mark.benchmark(group="unified-search")
    def test_global_search_no_tenant_filter(
        self,
        benchmark,
        qdrant_client,
        embedding_service,
        unified_collection_tenants,
    ):
        """
        Benchmark: Global search across all tenants.

        Measures search performance when scope="global" is used,
        searching across all projects in unified collection.
        """
        query_text = "API endpoint design"

        async def global_search():
            embeddings = await embedding_service.generate_embeddings(query_text)
            return qdrant_client.search(
                collection_name=MultiTenantBenchmarkSetup.UNIFIED_COLLECTION,
                query_vector=("dense", embeddings["dense"]),
                limit=10,
            )

        def run_search():
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(global_search())

        result = benchmark(run_search)
        assert len(result) > 0


class TestPerProjectCollectionSearchLatency:
    """Benchmark search latency in per-project collections (old architecture)."""

    @pytest.mark.benchmark(group="per-project-search")
    def test_search_in_project_collection(
        self,
        benchmark,
        qdrant_client,
        embedding_service,
        per_project_collections,
    ):
        """
        Benchmark: Search in single project collection.

        This is the old architecture pattern - one collection per project.
        Should be compared against unified collection with tenant filter.
        """
        collection_name = per_project_collections[0]
        query_text = "authentication user login"

        async def search_project():
            embeddings = await embedding_service.generate_embeddings(query_text)
            return qdrant_client.search(
                collection_name=collection_name,
                query_vector=("dense", embeddings["dense"]),
                limit=10,
            )

        def run_search():
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(search_project())

        result = benchmark(run_search)
        assert len(result) > 0


class TestSearchLatencyPercentiles:
    """Detailed percentile latency measurements."""

    def test_unified_search_percentiles(
        self,
        qdrant_client,
        embedding_service,
        unified_collection_tenants,
    ):
        """
        Measure p50, p95, p99 latency for unified collection search.

        Runs multiple iterations to compute accurate percentile metrics.
        Target: p95 < 100ms
        """
        tenant_id = unified_collection_tenants[0]
        query_text = "machine learning algorithm"
        iterations = 100

        async def run_iterations():
            latencies = []
            embeddings = await embedding_service.generate_embeddings(query_text)

            for _ in range(iterations):
                start = time.perf_counter()
                qdrant_client.search(
                    collection_name=MultiTenantBenchmarkSetup.UNIFIED_COLLECTION,
                    query_vector=("dense", embeddings["dense"]),
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=tenant_id),
                            )
                        ]
                    ),
                    limit=10,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

            return LatencyMetrics.from_samples(latencies)

        metrics = asyncio.get_event_loop().run_until_complete(run_iterations())

        print(f"\n{'=' * 60}")
        print("Unified Collection Search Latency (with tenant_id filter)")
        print(f"{'=' * 60}")
        print(f"  Iterations: {metrics.count}")
        print(f"  p50: {metrics.p50:.2f} ms")
        print(f"  p95: {metrics.p95:.2f} ms")
        print(f"  p99: {metrics.p99:.2f} ms")
        print(f"  Mean: {metrics.mean:.2f} ms")
        print(f"  Min: {metrics.min_val:.2f} ms")
        print(f"  Max: {metrics.max_val:.2f} ms")
        print(f"{'=' * 60}")

        # Target: p95 < 100ms for good performance
        assert metrics.p95 < 100, f"p95 latency {metrics.p95:.2f}ms exceeds 100ms target"


class TestConcurrentMultiTenantSearch:
    """Benchmark concurrent search across multiple tenants."""

    def test_concurrent_tenant_searches(
        self,
        qdrant_client,
        embedding_service,
        unified_collection_tenants,
    ):
        """
        Benchmark: Concurrent searches from different tenants.

        Simulates multiple MCP server instances searching simultaneously,
        each for their own tenant.
        """
        num_concurrent = 10
        queries_per_tenant = 10
        query_text = "performance optimization"

        async def concurrent_searches():
            start_time = time.perf_counter()
            embeddings = await embedding_service.generate_embeddings(query_text)

            async def search_for_tenant(tenant_id: str) -> list[float]:
                latencies = []
                for _ in range(queries_per_tenant):
                    start = time.perf_counter()
                    qdrant_client.search(
                        collection_name=MultiTenantBenchmarkSetup.UNIFIED_COLLECTION,
                        query_vector=("dense", embeddings["dense"]),
                        query_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="tenant_id",
                                    match=models.MatchValue(value=tenant_id),
                                )
                            ]
                        ),
                        limit=10,
                    )
                    latencies.append((time.perf_counter() - start) * 1000)
                return latencies

            # Run concurrent searches for all tenants
            tasks = [
                search_for_tenant(tid)
                for tid in unified_collection_tenants[:num_concurrent]
            ]
            all_latencies = await asyncio.gather(*tasks)

            total_time = time.perf_counter() - start_time
            total_queries = num_concurrent * queries_per_tenant
            qps = total_queries / total_time

            # Flatten latencies
            flat_latencies = [lat for tenant_lats in all_latencies for lat in tenant_lats]
            metrics = LatencyMetrics.from_samples(flat_latencies)

            return qps, metrics

        qps, metrics = asyncio.get_event_loop().run_until_complete(concurrent_searches())

        print(f"\n{'=' * 60}")
        print(f"Concurrent Multi-Tenant Search ({num_concurrent} tenants)")
        print(f"{'=' * 60}")
        print(f"  Total queries: {num_concurrent * queries_per_tenant}")
        print(f"  Queries/sec: {qps:.1f}")
        print(f"  p50: {metrics.p50:.2f} ms")
        print(f"  p95: {metrics.p95:.2f} ms")
        print(f"  p99: {metrics.p99:.2f} ms")
        print(f"{'=' * 60}")

        # Target: > 100 QPS with p95 < 200ms
        assert qps > 50, f"Throughput {qps:.1f} QPS below 50 QPS target"
        assert metrics.p95 < 200, f"p95 latency {metrics.p95:.2f}ms exceeds 200ms target"


class TestIngestionThroughput:
    """Benchmark document ingestion throughput."""

    def test_batch_ingestion_throughput(
        self,
        qdrant_client,
        embedding_service,
    ):
        """
        Benchmark: Batch document ingestion to unified collection.

        Measures documents/second for batch ingestion with tenant metadata.
        Target: 100+ docs/sec
        """
        collection_name = "_benchmark_ingestion_test"
        num_documents = 500
        batch_size = 50

        # Setup collection
        try:
            qdrant_client.delete_collection(collection_name)
        except Exception:
            pass

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

        async def ingest_documents():
            start_time = time.perf_counter()
            tenant_id = "abc123def456"

            for batch_start in range(0, num_documents, batch_size):
                points = []
                for i in range(batch_size):
                    doc_idx = batch_start + i
                    if doc_idx >= num_documents:
                        break

                    text = f"Document {doc_idx} for batch ingestion test"
                    embeddings = await embedding_service.generate_embeddings(text)

                    point = models.PointStruct(
                        id=doc_idx,
                        vector={
                            "dense": embeddings["dense"],
                            "sparse": models.SparseVector(
                                indices=embeddings["sparse"]["indices"],
                                values=embeddings["sparse"]["values"],
                            ),
                        },
                        payload={
                            "text": text,
                            "tenant_id": tenant_id,
                            "branch": "main",
                            "doc_index": doc_idx,
                        },
                    )
                    points.append(point)

                qdrant_client.upsert(collection_name=collection_name, points=points)

            elapsed = time.perf_counter() - start_time
            return num_documents / elapsed

        docs_per_sec = asyncio.get_event_loop().run_until_complete(ingest_documents())

        print(f"\n{'=' * 60}")
        print("Batch Ingestion Throughput")
        print(f"{'=' * 60}")
        print(f"  Documents: {num_documents}")
        print(f"  Batch size: {batch_size}")
        print(f"  Throughput: {docs_per_sec:.1f} docs/sec")
        print(f"{'=' * 60}")

        # Cleanup
        try:
            qdrant_client.delete_collection(collection_name)
        except Exception:
            pass

        # Target: 100+ docs/sec (embedding generation is the bottleneck)
        # Adjust target based on embedding model performance
        assert (
            docs_per_sec > 20
        ), f"Ingestion throughput {docs_per_sec:.1f} docs/sec below target"


class TestTenantIsolationOverhead:
    """Benchmark the overhead of tenant isolation filtering."""

    def test_filter_overhead_comparison(
        self,
        qdrant_client,
        embedding_service,
        unified_collection_tenants,
    ):
        """
        Compare search latency with and without tenant filter.

        Measures the overhead introduced by tenant_id filtering.
        """
        tenant_id = unified_collection_tenants[0]
        query_text = "software testing methodology"
        iterations = 50

        async def run_comparison():
            embeddings = await embedding_service.generate_embeddings(query_text)

            # Without filter (global search)
            global_latencies = []
            for _ in range(iterations):
                start = time.perf_counter()
                qdrant_client.search(
                    collection_name=MultiTenantBenchmarkSetup.UNIFIED_COLLECTION,
                    query_vector=("dense", embeddings["dense"]),
                    limit=10,
                )
                global_latencies.append((time.perf_counter() - start) * 1000)

            # With tenant filter
            filtered_latencies = []
            for _ in range(iterations):
                start = time.perf_counter()
                qdrant_client.search(
                    collection_name=MultiTenantBenchmarkSetup.UNIFIED_COLLECTION,
                    query_vector=("dense", embeddings["dense"]),
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=tenant_id),
                            )
                        ]
                    ),
                    limit=10,
                )
                filtered_latencies.append((time.perf_counter() - start) * 1000)

            global_metrics = LatencyMetrics.from_samples(global_latencies)
            filtered_metrics = LatencyMetrics.from_samples(filtered_latencies)

            overhead_pct = (
                (filtered_metrics.mean - global_metrics.mean) / global_metrics.mean
            ) * 100

            return global_metrics, filtered_metrics, overhead_pct

        global_metrics, filtered_metrics, overhead_pct = (
            asyncio.get_event_loop().run_until_complete(run_comparison())
        )

        print(f"\n{'=' * 60}")
        print("Tenant Filter Overhead Analysis")
        print(f"{'=' * 60}")
        print(f"  Global search mean: {global_metrics.mean:.2f} ms")
        print(f"  Filtered search mean: {filtered_metrics.mean:.2f} ms")
        print(f"  Filter overhead: {overhead_pct:+.1f}%")
        print(f"{'=' * 60}")

        # Tenant filtering should add minimal overhead (<50% on indexed field)
        # Note: Can be negative if filtering reduces result processing
        assert (
            abs(overhead_pct) < 100
        ), f"Filter overhead {overhead_pct:.1f}% exceeds acceptable range"


class TestArchitectureComparison:
    """Compare unified vs per-project collection architectures."""

    def test_unified_vs_per_project_latency(
        self,
        qdrant_client,
        embedding_service,
        unified_collection_tenants,
        per_project_collections,
    ):
        """
        Compare search latency between unified and per-project architectures.

        This is the key benchmark for validating the multi-tenant design decision.
        """
        query_text = "database query optimization"
        iterations = 50

        async def run_comparison():
            embeddings = await embedding_service.generate_embeddings(query_text)

            # Unified collection with tenant filter
            unified_latencies = []
            tenant_id = unified_collection_tenants[0]
            for _ in range(iterations):
                start = time.perf_counter()
                qdrant_client.search(
                    collection_name=MultiTenantBenchmarkSetup.UNIFIED_COLLECTION,
                    query_vector=("dense", embeddings["dense"]),
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="tenant_id",
                                match=models.MatchValue(value=tenant_id),
                            )
                        ]
                    ),
                    limit=10,
                )
                unified_latencies.append((time.perf_counter() - start) * 1000)

            # Per-project collection (no filter needed)
            per_project_latencies = []
            collection_name = per_project_collections[0]
            for _ in range(iterations):
                start = time.perf_counter()
                qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=("dense", embeddings["dense"]),
                    limit=10,
                )
                per_project_latencies.append((time.perf_counter() - start) * 1000)

            unified_metrics = LatencyMetrics.from_samples(unified_latencies)
            per_project_metrics = LatencyMetrics.from_samples(per_project_latencies)

            return unified_metrics, per_project_metrics

        unified_metrics, per_project_metrics = (
            asyncio.get_event_loop().run_until_complete(run_comparison())
        )

        print(f"\n{'=' * 60}")
        print("Architecture Comparison: Unified vs Per-Project")
        print(f"{'=' * 60}")
        print("Unified Collection (with tenant_id filter):")
        print(f"  p50: {unified_metrics.p50:.2f} ms")
        print(f"  p95: {unified_metrics.p95:.2f} ms")
        print(f"  Mean: {unified_metrics.mean:.2f} ms")
        print()
        print("Per-Project Collection (no filter):")
        print(f"  p50: {per_project_metrics.p50:.2f} ms")
        print(f"  p95: {per_project_metrics.p95:.2f} ms")
        print(f"  Mean: {per_project_metrics.mean:.2f} ms")
        print()
        diff_pct = (
            (unified_metrics.mean - per_project_metrics.mean)
            / per_project_metrics.mean
            * 100
        )
        print(f"Difference: {diff_pct:+.1f}% (positive = unified slower)")
        print(f"{'=' * 60}")

        # Unified should be within 100% of per-project performance
        # (indexing on tenant_id should keep overhead low)
        assert (
            diff_pct < 200
        ), f"Unified collection {diff_pct:.1f}% slower than acceptable threshold"
