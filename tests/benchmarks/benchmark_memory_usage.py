"""
Memory usage profiling benchmarks.

Measures memory consumption and detects memory leaks across various workload
scenarios including document ingestion, search operations, connection pooling,
and embedding model usage.

The benchmarks use Python's built-in tracemalloc for detailed allocation
tracking and psutil for process-level memory measurements (RSS).

Run with: uv run pytest tests/benchmarks/benchmark_memory_usage.py --benchmark-only

For verbose output with memory statistics:
    uv run pytest tests/benchmarks/benchmark_memory_usage.py --benchmark-only -v

For leak detection tests (longer running):
    uv run pytest tests/benchmarks/benchmark_memory_usage.py -k leak --benchmark-only

Memory Metrics Explained:
- peak_rss_mb: Peak resident set size (physical memory) in MB
- current_rss_mb: Current RSS after operation
- rss_delta_mb: Change in RSS during operation
- tracemalloc_peak_mb: Peak memory allocations tracked by tracemalloc
- tracemalloc_current_mb: Current allocations tracked by tracemalloc
- allocation_count: Number of memory allocations

Interpreting Results:
- Low rss_delta_mb: Operation is memory-efficient
- High peak_rss_mb: Operation requires significant memory
- Growing memory in leak tests: Potential memory leak
- Stable memory in leak tests: No leak detected
"""

import asyncio
import gc
import tracemalloc
from typing import Any

import psutil
import pytest
from common.core.embeddings import EmbeddingService
from common.core.hybrid_search import HybridSearchEngine
from common.core.ssl_config import suppress_qdrant_ssl_warnings
from qdrant_client import QdrantClient
from qdrant_client.http import models


class MemoryProfiler:
    """
    Memory profiling utility combining tracemalloc and psutil.

    Uses tracemalloc for Python allocation tracking and psutil for
    process-level memory measurements.
    """

    def __init__(self):
        self.process = psutil.Process()
        self.start_rss = 0
        self.peak_rss = 0

    def start(self):
        """Start memory profiling."""
        # Force garbage collection for clean baseline
        gc.collect()

        # Start tracemalloc
        tracemalloc.start()

        # Record starting RSS
        self.start_rss = self.process.memory_info().rss
        self.peak_rss = self.start_rss

    def snapshot(self) -> dict[str, float]:
        """Take a memory snapshot and return metrics."""
        current_rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, current_rss)

        # Get tracemalloc stats
        current_tracemalloc, peak_tracemalloc = tracemalloc.get_traced_memory()

        # Get top allocations for analysis
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        return {
            'current_rss_mb': current_rss / 1024 / 1024,
            'peak_rss_mb': self.peak_rss / 1024 / 1024,
            'rss_delta_mb': (current_rss - self.start_rss) / 1024 / 1024,
            'tracemalloc_current_mb': current_tracemalloc / 1024 / 1024,
            'tracemalloc_peak_mb': peak_tracemalloc / 1024 / 1024,
            'allocation_count': len(top_stats),
        }

    def stop(self) -> dict[str, float]:
        """Stop profiling and return final metrics."""
        metrics = self.snapshot()
        tracemalloc.stop()

        # Force cleanup
        gc.collect()

        return metrics


class TestDataGenerator:
    """Generates test documents for memory profiling."""

    @staticmethod
    def generate_text_document(size_kb: int, doc_id: int) -> dict[str, Any]:
        """Generate a text document of specified size."""
        lines = []
        target_bytes = size_kb * 1024
        current_bytes = 0

        line_template = f"Document {doc_id}: This is line {{}} of test content for memory profiling. "

        line_num = 0
        while current_bytes < target_bytes:
            line = line_template.format(line_num) + "\n"
            lines.append(line)
            current_bytes += len(line.encode('utf-8'))
            line_num += 1

        content = "".join(lines)

        return {
            "content": content,
            "metadata": {
                "doc_id": f"test_doc_{doc_id}",
                "size_kb": size_kb,
                "line_count": line_num,
            }
        }

    @staticmethod
    def generate_batch_documents(count: int, size_kb: int = 10) -> list[dict[str, Any]]:
        """Generate a batch of documents."""
        return [
            TestDataGenerator.generate_text_document(size_kb, i)
            for i in range(count)
        ]


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def qdrant_client():
    """Fixture providing a Qdrant client for testing."""
    suppress_qdrant_ssl_warnings()

    client = QdrantClient(host="localhost", port=6333)

    collection_name = "memory_test_collection"

    # Clean up if exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

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

    yield client

    # Cleanup
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass


@pytest.fixture
async def embedding_service():
    """Fixture providing an embedding service for testing."""
    service = EmbeddingService()
    await service.initialize()
    return service


# ============================================================================
# Helper Functions
# ============================================================================

def ingest_document_with_embeddings(
    client: QdrantClient,
    embedding_service: EmbeddingService,
    doc: dict[str, Any],
    point_id: int,
    collection_name: str = "memory_test_collection"
):
    """Helper to ingest a document with embeddings synchronously."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        dense_vector, sparse_vector = loop.run_until_complete(
            embedding_service.generate_embeddings(doc["content"])
        )
    finally:
        loop.close()

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector={"dense": dense_vector, "sparse": sparse_vector},
                payload={**doc["metadata"], "content": doc["content"]},
            )
        ],
    )


# ============================================================================
# Single Document Ingestion Memory Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_single_document_ingestion_memory_small(
    benchmark, qdrant_client, embedding_service
):
    """Benchmark memory usage for single small document (1KB) ingestion."""

    def ingest_doc():
        profiler = MemoryProfiler()
        profiler.start()

        doc = TestDataGenerator.generate_text_document(size_kb=1, doc_id=1)
        ingest_document_with_embeddings(qdrant_client, embedding_service, doc, 1)

        return profiler.stop()

    result = benchmark(ingest_doc)
    assert result['rss_delta_mb'] < 100, f"Memory usage too high: {result['rss_delta_mb']:.2f}MB"


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_single_document_ingestion_memory_medium(
    benchmark, qdrant_client, embedding_service
):
    """Benchmark memory usage for single medium document (100KB) ingestion."""

    def ingest_doc():
        profiler = MemoryProfiler()
        profiler.start()

        doc = TestDataGenerator.generate_text_document(size_kb=100, doc_id=2)
        ingest_document_with_embeddings(qdrant_client, embedding_service, doc, 2)

        return profiler.stop()

    result = benchmark(ingest_doc)
    assert result['rss_delta_mb'] < 150, f"Memory usage too high: {result['rss_delta_mb']:.2f}MB"


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_single_document_ingestion_memory_large(
    benchmark, qdrant_client, embedding_service
):
    """Benchmark memory usage for single large document (1MB) ingestion."""

    def ingest_doc():
        profiler = MemoryProfiler()
        profiler.start()

        doc = TestDataGenerator.generate_text_document(size_kb=1024, doc_id=3)
        ingest_document_with_embeddings(qdrant_client, embedding_service, doc, 3)

        return profiler.stop()

    result = benchmark(ingest_doc)
    assert result['rss_delta_mb'] < 250, f"Memory usage too high: {result['rss_delta_mb']:.2f}MB"


# ============================================================================
# Batch Document Ingestion Memory Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_batch_ingestion_memory_10_docs(benchmark, qdrant_client, embedding_service):
    """Benchmark memory usage for batch ingestion of 10 documents."""

    def ingest_batch():
        profiler = MemoryProfiler()
        profiler.start()

        docs = TestDataGenerator.generate_batch_documents(count=10, size_kb=10)

        for i, doc in enumerate(docs):
            ingest_document_with_embeddings(qdrant_client, embedding_service, doc, i + 100)

        return profiler.stop()

    result = benchmark(ingest_batch)
    assert result['rss_delta_mb'] < 200, f"Memory usage too high: {result['rss_delta_mb']:.2f}MB"


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_batch_ingestion_memory_100_docs(benchmark, qdrant_client, embedding_service):
    """Benchmark memory usage for batch ingestion of 100 documents."""

    def ingest_batch():
        profiler = MemoryProfiler()
        profiler.start()

        docs = TestDataGenerator.generate_batch_documents(count=100, size_kb=10)

        for i, doc in enumerate(docs):
            ingest_document_with_embeddings(qdrant_client, embedding_service, doc, i + 1000)
            if i % 20 == 0:
                gc.collect()

        return profiler.stop()

    result = benchmark(ingest_batch)
    assert result['rss_delta_mb'] < 600, f"Memory usage too high: {result['rss_delta_mb']:.2f}MB"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.requires_qdrant
def test_batch_ingestion_memory_1000_docs(benchmark, qdrant_client, embedding_service):
    """Benchmark memory usage for batch ingestion of 1000 documents."""

    def ingest_batch():
        profiler = MemoryProfiler()
        profiler.start()

        docs = TestDataGenerator.generate_batch_documents(count=1000, size_kb=5)

        for i, doc in enumerate(docs):
            ingest_document_with_embeddings(qdrant_client, embedding_service, doc, i + 10000)
            if i % 50 == 0:
                gc.collect()

        return profiler.stop()

    result = benchmark(ingest_batch)
    assert result['rss_delta_mb'] < 1536, f"Memory usage too high: {result['rss_delta_mb']:.2f}MB"


# ============================================================================
# Search Operation Memory Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_search_operation_memory(benchmark, qdrant_client, embedding_service):
    """Benchmark memory usage for search operations."""

    # Populate collection first
    docs = TestDataGenerator.generate_batch_documents(count=50, size_kb=10)
    for i, doc in enumerate(docs):
        ingest_document_with_embeddings(qdrant_client, embedding_service, doc, i + 20000)

    def perform_searches():
        profiler = MemoryProfiler()
        profiler.start()

        # Perform multiple searches
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for i in range(10):
                query_vector, _ = loop.run_until_complete(
                    embedding_service.generate_embeddings(f"test content {i}")
                )

                qdrant_client.search(
                    collection_name="memory_test_collection",
                    query_vector=("dense", query_vector),
                    limit=10,
                )
        finally:
            loop.close()

        return profiler.stop()

    result = benchmark(perform_searches)
    assert result['rss_delta_mb'] < 100, f"Search memory usage too high: {result['rss_delta_mb']:.2f}MB"


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_hybrid_search_memory(benchmark, qdrant_client, embedding_service):
    """Benchmark memory usage for hybrid search operations."""

    # Populate collection
    docs = TestDataGenerator.generate_batch_documents(count=50, size_kb=10)
    for i, doc in enumerate(docs):
        ingest_document_with_embeddings(qdrant_client, embedding_service, doc, i + 30000)

    # Create hybrid search engine
    search_engine = HybridSearchEngine(client=qdrant_client)

    def perform_hybrid_searches():
        profiler = MemoryProfiler()
        profiler.start()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for i in range(10):
                loop.run_until_complete(
                    search_engine.hybrid_search(
                        collection_name="memory_test_collection",
                        query_text=f"test content {i}",
                        limit=10,
                    )
                )
        finally:
            loop.close()

        return profiler.stop()

    result = benchmark(perform_hybrid_searches)
    assert result['rss_delta_mb'] < 150, f"Hybrid search memory too high: {result['rss_delta_mb']:.2f}MB"


# ============================================================================
# Connection Pooling Memory Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_connection_pooling_memory(benchmark):
    """Benchmark memory overhead of connection pooling."""

    def create_connections():
        profiler = MemoryProfiler()
        profiler.start()

        suppress_qdrant_ssl_warnings()

        # Create multiple client instances
        clients = []
        for _i in range(10):
            client = QdrantClient(host="localhost", port=6333)
            clients.append(client)

        profiler.snapshot()

        # Cleanup
        del clients
        gc.collect()

        return profiler.stop()

    result = benchmark(create_connections)
    assert result['rss_delta_mb'] < 200, f"Connection pooling memory too high: {result['rss_delta_mb']:.2f}MB"


# ============================================================================
# Embedding Model Memory Tests
# ============================================================================

@pytest.mark.benchmark
def test_embedding_model_memory_single(benchmark, embedding_service):
    """Benchmark memory usage of embedding model for single text."""

    def embed_text():
        profiler = MemoryProfiler()
        profiler.start()

        text = "This is a test document for embedding memory profiling. " * 50

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                embedding_service.generate_embeddings(text)
            )
        finally:
            loop.close()

        return profiler.stop()

    result = benchmark(embed_text)
    assert result['rss_delta_mb'] < 150, f"Embedding memory too high: {result['rss_delta_mb']:.2f}MB"


@pytest.mark.benchmark
def test_embedding_model_memory_batch(benchmark, embedding_service):
    """Benchmark memory usage of embedding model for batch of texts."""

    def embed_batch():
        profiler = MemoryProfiler()
        profiler.start()

        texts = [
            f"This is test document {i} for embedding memory profiling. " * 20
            for i in range(20)
        ]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for text in texts:
                loop.run_until_complete(
                    embedding_service.generate_embeddings(text)
                )
        finally:
            loop.close()

        return profiler.stop()

    result = benchmark(embed_batch)
    assert result['rss_delta_mb'] < 300, f"Batch embedding memory too high: {result['rss_delta_mb']:.2f}MB"


# ============================================================================
# Memory Leak Detection Tests
# ============================================================================

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.requires_qdrant
def test_long_running_ingestion_leak_detection(benchmark, qdrant_client, embedding_service):
    """
    Detect memory leaks in long-running ingestion operations.

    Runs ingestion in a loop and monitors memory growth. Memory should
    stabilize after initial allocation, not grow continuously.
    """

    def long_running_ingestion():
        profiler = MemoryProfiler()
        profiler.start()

        memory_snapshots = []
        iterations = 50  # Reduced for benchmark performance

        for i in range(iterations):
            doc = TestDataGenerator.generate_text_document(size_kb=5, doc_id=i)
            ingest_document_with_embeddings(qdrant_client, embedding_service, doc, i + 40000)

            # Take memory snapshot every 10 iterations
            if i % 10 == 0:
                snapshot = profiler.snapshot()
                memory_snapshots.append(snapshot['current_rss_mb'])
                gc.collect()

        final_metrics = profiler.stop()

        # Check for memory leak: memory should stabilize
        if len(memory_snapshots) >= 3:
            early_avg = sum(memory_snapshots[:2]) / 2
            late_avg = sum(memory_snapshots[-2:]) / 2
            growth_rate = (late_avg - early_avg) / early_avg

            # Memory growth should be less than 50%
            assert growth_rate < 0.5, f"Potential memory leak detected: {growth_rate*100:.1f}% growth"

        return final_metrics

    result = benchmark(long_running_ingestion)
    assert result['rss_delta_mb'] < 400, f"Long-running memory usage too high: {result['rss_delta_mb']:.2f}MB"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.requires_qdrant
def test_long_running_search_leak_detection(benchmark, qdrant_client, embedding_service):
    """
    Detect memory leaks in long-running search operations.

    Performs repeated searches and monitors memory growth.
    """

    # Populate collection
    docs = TestDataGenerator.generate_batch_documents(count=100, size_kb=5)
    for i, doc in enumerate(docs):
        ingest_document_with_embeddings(qdrant_client, embedding_service, doc, i + 50000)

    def long_running_search():
        profiler = MemoryProfiler()
        profiler.start()

        memory_snapshots = []
        iterations = 100

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            for i in range(iterations):
                query_vector, _ = loop.run_until_complete(
                    embedding_service.generate_embeddings(f"test content {i % 20}")
                )

                qdrant_client.search(
                    collection_name="memory_test_collection",
                    query_vector=("dense", query_vector),
                    limit=10,
                )

                # Take memory snapshot every 20 iterations
                if i % 20 == 0:
                    snapshot = profiler.snapshot()
                    memory_snapshots.append(snapshot['current_rss_mb'])
                    gc.collect()
        finally:
            loop.close()

        final_metrics = profiler.stop()

        # Check for memory leak
        if len(memory_snapshots) >= 3:
            early_avg = sum(memory_snapshots[:2]) / 2
            late_avg = sum(memory_snapshots[-2:]) / 2
            growth_rate = (late_avg - early_avg) / early_avg

            # Search should have minimal memory growth
            assert growth_rate < 0.3, f"Potential search leak detected: {growth_rate*100:.1f}% growth"

        return final_metrics

    result = benchmark(long_running_search)
    assert result['rss_delta_mb'] < 100, f"Search memory delta too high: {result['rss_delta_mb']:.2f}MB"


# ============================================================================
# Documentation: Rust Daemon Memory Profiling
# ============================================================================

"""
Future Rust daemon memory profiling approach:

The Rust daemon has strong memory safety guarantees, but long-running processes
should still be profiled for memory leaks and growth patterns.

Recommended tools and approaches:

1. **heaptrack** - Linux heap profiler with GUI
   ```bash
   heaptrack target/release/workspace-qdrant-daemon
   heaptrack_gui heaptrack.workspace-qdrant-daemon.*.gz
   ```

2. **Valgrind Massif** - Memory profiler
   ```bash
   valgrind --tool=massif target/release/workspace-qdrant-daemon
   ms_print massif.out.*
   ```

3. **cargo-flamegraph** - Flame graph generation
   ```bash
   cargo flamegraph --bin workspace-qdrant-daemon
   ```

4. **/proc monitoring** - Track RSS, VmSize, VmRSS
   ```bash
   watch -n 1 'cat /proc/$(pidof workspace-qdrant-daemon)/status | grep Vm'
   ```

5. **jemalloc** - Alternative allocator with profiling
   ```rust
   // In Cargo.toml:
   [dependencies]
   jemalloc-ctl = "0.5"
   jemallocator = "0.5"

   // In main.rs:
   #[global_allocator]
   static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
   ```

Test scenarios for Rust daemon:
- Long-running file watching (24+ hours)
- Processing large file batches
- gRPC connection handling under load
- Queue management with backpressure
- Crash recovery and restart cycles

Monitor for:
- RSS growth over time
- Heap fragmentation
- Thread-local storage leaks
- gRPC connection leaks
- File descriptor leaks
"""
