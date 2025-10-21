"""
gRPC communication overhead benchmarks.

Measures gRPC communication overhead between Python client and Rust daemon,
including connection establishment, request/response latency, batch throughput,
and serialization overhead across various payload sizes.

Prerequisites:
    - Rust daemon must be running: wqm service start
    - Qdrant server must be running on localhost:6333

Run with:
    uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only

    # Run specific benchmarks
    uv run pytest tests/benchmarks/benchmark_grpc_overhead.py::test_connection_establishment --benchmark-only
    uv run pytest tests/benchmarks/benchmark_grpc_overhead.py -k "1kb" --benchmark-only

    # Save results for comparison
    uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-save=grpc_baseline

    # Compare with previous results
    uv run pytest tests/benchmarks/benchmark_grpc_overhead.py --benchmark-only --benchmark-compare=grpc_baseline
"""

import asyncio
import time
from typing import Dict, List
import pytest

from common.grpc.daemon_client import DaemonClient
from common.grpc.generated import workspace_daemon_pb2 as pb2
from google.protobuf.empty_pb2 import Empty


# Payload size configurations
PAYLOAD_SIZES = {
    "small": 1 * 1024,      # 1KB
    "medium": 100 * 1024,   # 100KB
    "large": 1 * 1024 * 1024  # 1MB
}


class PayloadGenerator:
    """Generate test payloads of various sizes for gRPC benchmarks."""

    @staticmethod
    def generate_text(size_bytes: int) -> str:
        """
        Generate text content of specified size in bytes.

        Args:
            size_bytes: Target size in bytes

        Returns:
            Generated text content
        """
        # Use repeating pattern for consistency
        pattern = "This is a test payload for gRPC overhead benchmarking. " * 100
        repetitions = (size_bytes // len(pattern.encode('utf-8'))) + 1
        content = pattern * repetitions
        # Truncate to exact size
        return content[:size_bytes].ljust(size_bytes, ' ')

    @staticmethod
    def generate_metadata(num_fields: int = 10) -> Dict[str, str]:
        """
        Generate metadata dictionary with specified number of fields.

        Args:
            num_fields: Number of metadata fields to generate

        Returns:
            Metadata dictionary
        """
        return {
            f"field_{i}": f"value_{i}_" + "x" * 50
            for i in range(num_fields)
        }


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def daemon_client():
    """
    Fixture providing a connected daemon client.

    The daemon must be running before tests execute.
    """
    client = DaemonClient(host="localhost", port=50051)

    try:
        await client.start()
        # Verify daemon is actually reachable
        await client.health_check(timeout=5.0)
        yield client
    except Exception as e:
        pytest.skip(f"Daemon not available: {e}")
    finally:
        await client.stop()


@pytest.fixture
def payload_generator():
    """Fixture providing payload generator."""
    return PayloadGenerator()


# =============================================================================
# Connection Establishment Benchmarks
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_connection_establishment(benchmark):
    """
    Benchmark gRPC connection establishment time.

    Measures time to establish a new gRPC channel and verify it's ready.
    This is a one-time overhead when the client starts.
    """
    async def establish_connection():
        client = DaemonClient(host="localhost", port=50051)
        await client.start()
        await client.stop()
        return client

    # Run synchronously for benchmark compatibility
    def sync_wrapper():
        return asyncio.run(establish_connection())

    result = benchmark(sync_wrapper)
    assert result is not None


# =============================================================================
# Unary RPC Latency Benchmarks
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_health_check_latency(benchmark, daemon_client):
    """
    Benchmark health check RPC latency (minimal payload).

    Measures round-trip time for the simplest RPC call with no payload.
    This represents baseline gRPC overhead.
    """
    async def health_check():
        return await daemon_client.health_check(timeout=5.0)

    def sync_wrapper():
        return asyncio.run(health_check())

    result = benchmark(sync_wrapper)
    assert result.status in [pb2.SERVICE_STATUS_HEALTHY, pb2.SERVICE_STATUS_DEGRADED]


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_get_status_latency(benchmark, daemon_client):
    """
    Benchmark system status RPC latency (moderate response payload).

    Measures latency for RPCs with structured response data.
    """
    async def get_status():
        return await daemon_client.get_status(timeout=10.0)

    def sync_wrapper():
        return asyncio.run(get_status())

    result = benchmark(sync_wrapper)
    assert result.status in [pb2.SERVICE_STATUS_HEALTHY, pb2.SERVICE_STATUS_DEGRADED]


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_get_metrics_latency(benchmark, daemon_client):
    """
    Benchmark metrics RPC latency (moderate response payload with arrays).

    Measures latency for RPCs returning array/list data.
    """
    async def get_metrics():
        return await daemon_client.get_metrics(timeout=10.0)

    def sync_wrapper():
        return asyncio.run(get_metrics())

    result = benchmark(sync_wrapper)
    assert result is not None


# =============================================================================
# Text Ingestion Latency by Payload Size
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_ingest_text_1kb_latency(benchmark, daemon_client, payload_generator):
    """
    Benchmark text ingestion with 1KB payload.

    Measures end-to-end latency including serialization, transmission,
    processing, and response for small payloads.
    """
    content = payload_generator.generate_text(PAYLOAD_SIZES["small"])
    metadata = payload_generator.generate_metadata(5)

    async def ingest():
        return await daemon_client.ingest_text(
            content=content,
            collection_basename="bench-test",
            tenant_id="test_project",
            metadata=metadata,
            chunk_text=True,
            timeout=60.0
        )

    def sync_wrapper():
        return asyncio.run(ingest())

    result = benchmark(sync_wrapper)
    assert result.success is True
    assert result.chunks_created > 0


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_ingest_text_100kb_latency(benchmark, daemon_client, payload_generator):
    """
    Benchmark text ingestion with 100KB payload.

    Measures latency for medium-sized payloads to identify scaling behavior.
    """
    content = payload_generator.generate_text(PAYLOAD_SIZES["medium"])
    metadata = payload_generator.generate_metadata(10)

    async def ingest():
        return await daemon_client.ingest_text(
            content=content,
            collection_basename="bench-test",
            tenant_id="test_project",
            metadata=metadata,
            chunk_text=True,
            timeout=60.0
        )

    def sync_wrapper():
        return asyncio.run(ingest())

    result = benchmark(sync_wrapper)
    assert result.success is True
    assert result.chunks_created > 0


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_ingest_text_1mb_latency(benchmark, daemon_client, payload_generator):
    """
    Benchmark text ingestion with 1MB payload.

    Measures latency for large payloads to identify throughput limits
    and serialization overhead.
    """
    content = payload_generator.generate_text(PAYLOAD_SIZES["large"])
    metadata = payload_generator.generate_metadata(20)

    async def ingest():
        return await daemon_client.ingest_text(
            content=content,
            collection_basename="bench-test",
            tenant_id="test_project",
            metadata=metadata,
            chunk_text=True,
            timeout=60.0
        )

    def sync_wrapper():
        return asyncio.run(ingest())

    result = benchmark(sync_wrapper)
    assert result.success is True
    assert result.chunks_created > 0


# =============================================================================
# Batch Request Throughput Benchmarks
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_sequential_health_checks_throughput(benchmark, daemon_client):
    """
    Benchmark throughput of sequential health check RPCs.

    Measures requests per second for sequential unary calls.
    Useful for understanding single-threaded client throughput.
    """
    async def sequential_checks():
        results = []
        for _ in range(10):
            result = await daemon_client.health_check(timeout=5.0)
            results.append(result)
        return results

    def sync_wrapper():
        return asyncio.run(sequential_checks())

    results = benchmark(sync_wrapper)
    assert len(results) == 10
    assert all(r.status in [pb2.SERVICE_STATUS_HEALTHY, pb2.SERVICE_STATUS_DEGRADED] for r in results)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_concurrent_health_checks_throughput(benchmark, daemon_client):
    """
    Benchmark throughput of concurrent health check RPCs.

    Measures concurrent request handling capability.
    Shows the benefit of async I/O and connection pooling.
    """
    async def concurrent_checks():
        tasks = [
            daemon_client.health_check(timeout=5.0)
            for _ in range(10)
        ]
        return await asyncio.gather(*tasks)

    def sync_wrapper():
        return asyncio.run(concurrent_checks())

    results = benchmark(sync_wrapper)
    assert len(results) == 10
    assert all(r.status in [pb2.SERVICE_STATUS_HEALTHY, pb2.SERVICE_STATUS_DEGRADED] for r in results)


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_batch_ingest_small_documents_throughput(benchmark, daemon_client, payload_generator):
    """
    Benchmark throughput for batch ingestion of small documents.

    Measures documents per second for realistic batch ingestion workload.
    """
    documents = [
        (payload_generator.generate_text(1024), payload_generator.generate_metadata(3))
        for _ in range(5)
    ]

    async def batch_ingest():
        tasks = []
        for idx, (content, metadata) in enumerate(documents):
            task = daemon_client.ingest_text(
                content=content,
                collection_basename="bench-test",
                tenant_id="test_project",
                document_id=f"batch_doc_{idx}",
                metadata=metadata,
                chunk_text=True,
                timeout=60.0
            )
            tasks.append(task)
        return await asyncio.gather(*tasks)

    def sync_wrapper():
        return asyncio.run(batch_ingest())

    results = benchmark(sync_wrapper)
    assert len(results) == 5
    assert all(r.success for r in results)


# =============================================================================
# Collection Management Latency
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_create_collection_latency(benchmark, daemon_client):
    """
    Benchmark collection creation RPC latency.

    Measures latency for collection management operations.
    """
    collection_counter = 0

    async def create_collection():
        nonlocal collection_counter
        collection_counter += 1
        return await daemon_client.create_collection(
            collection_name=f"bench-collection-{collection_counter}",
            project_id="test_project",
            timeout=30.0
        )

    def sync_wrapper():
        return asyncio.run(create_collection())

    result = benchmark(sync_wrapper)
    assert result.success is True


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_alias_operations_latency(benchmark, daemon_client):
    """
    Benchmark alias creation/deletion latency.

    Measures latency for lightweight metadata operations.
    """
    alias_counter = 0

    async def alias_operations():
        nonlocal alias_counter
        alias_counter += 1
        alias_name = f"bench-alias-{alias_counter}"

        # Create collection first
        await daemon_client.create_collection(
            collection_name=f"bench-target-{alias_counter}",
            project_id="test_project",
            timeout=30.0
        )

        # Create alias
        await daemon_client.create_collection_alias(
            alias_name=alias_name,
            collection_name=f"bench-target-{alias_counter}",
            timeout=10.0
        )

        # Delete alias
        await daemon_client.delete_collection_alias(
            alias_name=alias_name,
            timeout=10.0
        )

        return True

    def sync_wrapper():
        return asyncio.run(alias_operations())

    result = benchmark(sync_wrapper)
    assert result is True


# =============================================================================
# Serialization Overhead Benchmarks
# =============================================================================

@pytest.mark.benchmark
def test_protobuf_serialization_1kb(benchmark, payload_generator):
    """
    Benchmark protobuf serialization overhead for 1KB payload.

    Measures pure serialization time without network I/O.
    """
    content = payload_generator.generate_text(PAYLOAD_SIZES["small"])
    metadata = payload_generator.generate_metadata(5)

    def serialize():
        request = pb2.IngestTextRequest(
            content=content,
            collection_basename="bench-test",
            tenant_id="test_project",
            metadata=metadata,
            chunk_text=True,
        )
        # Serialize to bytes
        serialized = request.SerializeToString()
        # Deserialize back
        deserialized = pb2.IngestTextRequest()
        deserialized.ParseFromString(serialized)
        return deserialized

    result = benchmark(serialize)
    assert result.content == content


@pytest.mark.benchmark
def test_protobuf_serialization_100kb(benchmark, payload_generator):
    """
    Benchmark protobuf serialization overhead for 100KB payload.

    Measures serialization scaling with payload size.
    """
    content = payload_generator.generate_text(PAYLOAD_SIZES["medium"])
    metadata = payload_generator.generate_metadata(10)

    def serialize():
        request = pb2.IngestTextRequest(
            content=content,
            collection_basename="bench-test",
            tenant_id="test_project",
            metadata=metadata,
            chunk_text=True,
        )
        serialized = request.SerializeToString()
        deserialized = pb2.IngestTextRequest()
        deserialized.ParseFromString(serialized)
        return deserialized

    result = benchmark(serialize)
    assert result.content == content


@pytest.mark.benchmark
def test_protobuf_serialization_1mb(benchmark, payload_generator):
    """
    Benchmark protobuf serialization overhead for 1MB payload.

    Measures serialization performance limits for large messages.
    """
    content = payload_generator.generate_text(PAYLOAD_SIZES["large"])
    metadata = payload_generator.generate_metadata(20)

    def serialize():
        request = pb2.IngestTextRequest(
            content=content,
            collection_basename="bench-test",
            tenant_id="test_project",
            metadata=metadata,
            chunk_text=True,
        )
        serialized = request.SerializeToString()
        deserialized = pb2.IngestTextRequest()
        deserialized.ParseFromString(serialized)
        return deserialized

    result = benchmark(serialize)
    assert result.content == content


# =============================================================================
# Retry and Circuit Breaker Overhead
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_successful_request_with_retry_enabled(benchmark, daemon_client):
    """
    Benchmark overhead of retry logic on successful requests.

    Measures the performance cost of retry/circuit breaker infrastructure
    when requests succeed on first attempt.
    """
    async def request_with_retry():
        return await daemon_client.health_check(timeout=5.0)

    def sync_wrapper():
        return asyncio.run(request_with_retry())

    result = benchmark(sync_wrapper)
    assert result.status in [pb2.SERVICE_STATUS_HEALTHY, pb2.SERVICE_STATUS_DEGRADED]


# =============================================================================
# Message Size Impact Analysis
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_metadata_size_impact_small(benchmark, daemon_client, payload_generator):
    """
    Benchmark impact of small metadata (5 fields) on latency.

    Measures overhead of metadata serialization and transmission.
    """
    content = payload_generator.generate_text(5 * 1024)  # 5KB content
    metadata = payload_generator.generate_metadata(5)

    async def ingest():
        return await daemon_client.ingest_text(
            content=content,
            collection_basename="bench-test",
            tenant_id="test_project",
            metadata=metadata,
            chunk_text=True,
            timeout=60.0
        )

    def sync_wrapper():
        return asyncio.run(ingest())

    result = benchmark(sync_wrapper)
    assert result.success is True


@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_metadata_size_impact_large(benchmark, daemon_client, payload_generator):
    """
    Benchmark impact of large metadata (50 fields) on latency.

    Compares with small metadata to quantify metadata overhead.
    """
    content = payload_generator.generate_text(5 * 1024)  # 5KB content
    metadata = payload_generator.generate_metadata(50)

    async def ingest():
        return await daemon_client.ingest_text(
            content=content,
            collection_basename="bench-test",
            tenant_id="test_project",
            metadata=metadata,
            chunk_text=True,
            timeout=60.0
        )

    def sync_wrapper():
        return asyncio.run(ingest())

    result = benchmark(sync_wrapper)
    assert result.success is True
