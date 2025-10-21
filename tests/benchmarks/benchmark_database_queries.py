"""
Database Query Performance Benchmarks.

Measures query performance for SQLite state manager and Qdrant vector database operations.
Tests include various data volumes, batch operations, and query optimization scenarios.

Run with: uv run pytest tests/benchmarks/benchmark_database_queries.py --benchmark-only
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any

from src.python.common.core.sqlite_state_manager import (
    SQLiteStateManager,
    WatchFolderConfig,
    FileProcessingStatus,
    ProcessingPriority,
)


# Test data volume configurations
SMALL_DATASET = 10
MEDIUM_DATASET = 100
LARGE_DATASET = 1000


def run_async(coro):
    """Helper to run async code in benchmarks (creates new event loop each time)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def create_sqlite_manager():
    """Create and initialize a temporary SQLite manager for benchmarking."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    manager = SQLiteStateManager(db_path=db_path)
    run_async(manager.initialize())
    return manager, db_path


def cleanup_sqlite_manager(manager, db_path):
    """Clean up SQLite manager and database file."""
    run_async(manager.close())
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def watch_folder_configs_small() -> List[WatchFolderConfig]:
    """Generate small dataset of watch folder configs."""
    return [
        WatchFolderConfig(
            watch_id=f"watch-{i}",
            path=f"/test/path/{i}",
            collection=f"collection-{i}",
            patterns=["*.py", "*.md"],
            ignore_patterns=["*.pyc", "__pycache__/*"],
            auto_ingest=True,
            recursive=True,
        )
        for i in range(SMALL_DATASET)
    ]


@pytest.fixture
def watch_folder_configs_medium() -> List[WatchFolderConfig]:
    """Generate medium dataset of watch folder configs."""
    return [
        WatchFolderConfig(
            watch_id=f"watch-{i}",
            path=f"/test/path/{i}",
            collection=f"collection-{i}",
            patterns=["*.py", "*.md"],
            ignore_patterns=["*.pyc", "__pycache__/*"],
            auto_ingest=True,
            recursive=True,
        )
        for i in range(MEDIUM_DATASET)
    ]


@pytest.fixture
def watch_folder_configs_large() -> List[WatchFolderConfig]:
    """Generate large dataset of watch folder configs."""
    return [
        WatchFolderConfig(
            watch_id=f"watch-{i}",
            path=f"/test/path/{i}",
            collection=f"collection-{i}",
            patterns=["*.py", "*.md"],
            ignore_patterns=["*.pyc", "__pycache__/*"],
            auto_ingest=True,
            recursive=True,
        )
        for i in range(LARGE_DATASET)
    ]


# ============================================================================
# SQLite State Manager - Watch Folder Operations
# ============================================================================


@pytest.mark.benchmark
def test_watch_folder_save_single_small(benchmark, watch_folder_configs_small):
    """Benchmark single watch folder config save (small dataset)."""
    manager, db_path = create_sqlite_manager()
    config = watch_folder_configs_small[0]

    def save_config():
        return run_async(manager.save_watch_folder_config(config))

    result = benchmark(save_config)
    cleanup_sqlite_manager(manager, db_path)
    assert result is True


@pytest.mark.benchmark
def test_watch_folder_save_batch_small(benchmark, sqlite_manager, watch_folder_configs_small):
    """Benchmark batch watch folder config saves (small dataset)."""

    async def save_batch():
        results = []
        for config in watch_folder_configs_small:
            result = await sqlite_manager.save_watch_folder_config(config)
            results.append(result)
        return results

    results = benchmark(asyncio.run, save_batch())
    assert all(results)


@pytest.mark.benchmark
def test_watch_folder_save_batch_medium(benchmark, sqlite_manager, watch_folder_configs_medium):
    """Benchmark batch watch folder config saves (medium dataset)."""

    async def save_batch():
        results = []
        for config in watch_folder_configs_medium:
            result = await sqlite_manager.save_watch_folder_config(config)
            results.append(result)
        return results

    results = benchmark(asyncio.run, save_batch())
    assert all(results)


@pytest.mark.benchmark
def test_watch_folder_get_single(benchmark, sqlite_manager, watch_folder_configs_medium):
    """Benchmark single watch folder config retrieval."""
    # Pre-populate database
    async def setup():
        for config in watch_folder_configs_medium:
            await sqlite_manager.save_watch_folder_config(config)

    run_async(setup())

    async def get_config():
        return await sqlite_manager.get_watch_folder_config("watch-50")

    result = benchmark(asyncio.run, get_config())
    assert result is not None
    assert result.watch_id == "watch-50"


@pytest.mark.benchmark
def test_watch_folder_list_all_small(benchmark, sqlite_manager, watch_folder_configs_small):
    """Benchmark listing all watch folders (small dataset)."""
    # Pre-populate database
    async def setup():
        for config in watch_folder_configs_small:
            await sqlite_manager.save_watch_folder_config(config)

    run_async(setup())

    async def list_all():
        return await sqlite_manager.get_all_watch_folder_configs()

    results = benchmark(asyncio.run, list_all())
    assert len(results) == SMALL_DATASET


@pytest.mark.benchmark
def test_watch_folder_list_all_medium(benchmark, sqlite_manager, watch_folder_configs_medium):
    """Benchmark listing all watch folders (medium dataset)."""
    # Pre-populate database
    async def setup():
        for config in watch_folder_configs_medium:
            await sqlite_manager.save_watch_folder_config(config)

    run_async(setup())

    async def list_all():
        return await sqlite_manager.get_all_watch_folder_configs()

    results = benchmark(asyncio.run, list_all())
    assert len(results) == MEDIUM_DATASET


@pytest.mark.benchmark
def test_watch_folder_list_all_large(benchmark, sqlite_manager, watch_folder_configs_large):
    """Benchmark listing all watch folders (large dataset)."""
    # Pre-populate database
    async def setup():
        for config in watch_folder_configs_large:
            await sqlite_manager.save_watch_folder_config(config)

    run_async(setup())

    async def list_all():
        return await sqlite_manager.get_all_watch_folder_configs()

    results = benchmark(asyncio.run, list_all())
    assert len(results) == LARGE_DATASET


@pytest.mark.benchmark
def test_watch_folder_remove(benchmark, sqlite_manager, watch_folder_configs_medium):
    """Benchmark watch folder config removal."""
    # Pre-populate database
    async def setup():
        for config in watch_folder_configs_medium:
            await sqlite_manager.save_watch_folder_config(config)

    run_async(setup())

    async def remove_config():
        return await sqlite_manager.remove_watch_folder_config("watch-75")

    result = benchmark(asyncio.run, remove_config())
    assert result is True


# ============================================================================
# SQLite State Manager - Ingestion Queue Operations
# ============================================================================


@pytest.mark.benchmark
def test_ingestion_queue_enqueue_single(benchmark, sqlite_manager):
    """Benchmark single file enqueue operation."""

    async def enqueue_file():
        return await sqlite_manager.enqueue(
            file_path="/test/file.py",
            collection="test-collection",
            priority=5,
            tenant_id="test-tenant",
            branch="main",
        )

    result = benchmark(asyncio.run, enqueue_file())
    assert result is not None


@pytest.mark.benchmark
def test_ingestion_queue_enqueue_batch_small(benchmark, sqlite_manager):
    """Benchmark batch file enqueue (small dataset)."""

    async def enqueue_batch():
        results = []
        for i in range(SMALL_DATASET):
            result = await sqlite_manager.enqueue(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=5,
                tenant_id="test-tenant",
                branch="main",
            )
            results.append(result)
        return results

    results = benchmark(asyncio.run, enqueue_batch())
    assert len(results) == SMALL_DATASET


@pytest.mark.benchmark
def test_ingestion_queue_enqueue_batch_medium(benchmark, sqlite_manager):
    """Benchmark batch file enqueue (medium dataset)."""

    async def enqueue_batch():
        results = []
        for i in range(MEDIUM_DATASET):
            result = await sqlite_manager.enqueue(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=5,
                tenant_id="test-tenant",
                branch="main",
            )
            results.append(result)
        return results

    results = benchmark(asyncio.run, enqueue_batch())
    assert len(results) == MEDIUM_DATASET


@pytest.mark.benchmark
def test_ingestion_queue_dequeue_small(benchmark, sqlite_manager):
    """Benchmark dequeue operation (small dataset)."""
    # Pre-populate queue
    async def setup():
        for i in range(SMALL_DATASET):
            await sqlite_manager.enqueue(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=5,
                tenant_id="test-tenant",
                branch="main",
            )

    run_async(setup())

    async def dequeue_items():
        return await sqlite_manager.dequeue(batch_size=10)

    results = benchmark(asyncio.run, dequeue_items())
    assert len(results) == 10


@pytest.mark.benchmark
def test_ingestion_queue_dequeue_medium(benchmark, sqlite_manager):
    """Benchmark dequeue operation (medium dataset)."""
    # Pre-populate queue
    async def setup():
        for i in range(MEDIUM_DATASET):
            await sqlite_manager.enqueue(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=5,
                tenant_id="test-tenant",
                branch="main",
            )

    run_async(setup())

    async def dequeue_items():
        return await sqlite_manager.dequeue(batch_size=50)

    results = benchmark(asyncio.run, dequeue_items())
    assert len(results) == 50


@pytest.mark.benchmark
def test_ingestion_queue_get_depth(benchmark, sqlite_manager):
    """Benchmark queue depth query."""
    # Pre-populate queue
    async def setup():
        for i in range(MEDIUM_DATASET):
            await sqlite_manager.enqueue(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=5,
                tenant_id="test-tenant",
                branch="main",
            )

    run_async(setup())

    async def get_depth():
        return await sqlite_manager.get_queue_depth()

    result = benchmark(asyncio.run, get_depth())
    assert result == MEDIUM_DATASET


@pytest.mark.benchmark
def test_ingestion_queue_remove_single(benchmark, sqlite_manager):
    """Benchmark single item removal from queue."""
    # Pre-populate queue
    async def setup():
        queue_ids = []
        for i in range(MEDIUM_DATASET):
            queue_id = await sqlite_manager.enqueue(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=5,
                tenant_id="test-tenant",
                branch="main",
            )
            queue_ids.append(queue_id)
        return queue_ids

    queue_ids = run_async(setup())

    async def remove_item():
        return await sqlite_manager.remove_from_queue(queue_ids[50])

    result = benchmark(asyncio.run, remove_item())
    assert result is True


# ============================================================================
# SQLite State Manager - File Processing State Operations
# ============================================================================


@pytest.mark.benchmark
def test_file_processing_start_single(benchmark, sqlite_manager):
    """Benchmark starting file processing for single file."""

    async def start_processing():
        return await sqlite_manager.start_file_processing(
            file_path="/test/file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            file_size=1024,
        )

    result = benchmark(asyncio.run, start_processing())
    assert result is True


@pytest.mark.benchmark
def test_file_processing_start_batch(benchmark, sqlite_manager):
    """Benchmark starting file processing for batch of files."""

    async def start_batch():
        results = []
        for i in range(SMALL_DATASET):
            result = await sqlite_manager.start_file_processing(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=ProcessingPriority.NORMAL,
                file_size=1024,
            )
            results.append(result)
        return results

    results = benchmark(asyncio.run, start_batch())
    assert all(results)


@pytest.mark.benchmark
def test_file_processing_complete_single(benchmark, sqlite_manager):
    """Benchmark completing file processing for single file."""
    # Pre-populate with processing file
    async def setup():
        await sqlite_manager.start_file_processing(
            file_path="/test/file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            file_size=1024,
        )

    run_async(setup())

    async def complete_processing():
        return await sqlite_manager.complete_file_processing(
            file_path="/test/file.py",
            success=True,
            processing_time_ms=100,
        )

    result = benchmark(asyncio.run, complete_processing())
    assert result is True


@pytest.mark.benchmark
def test_file_processing_get_status(benchmark, sqlite_manager):
    """Benchmark getting file processing status."""
    # Pre-populate with files
    async def setup():
        for i in range(MEDIUM_DATASET):
            await sqlite_manager.start_file_processing(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=ProcessingPriority.NORMAL,
                file_size=1024,
            )

    run_async(setup())

    async def get_status():
        return await sqlite_manager.get_file_processing_status("/test/file-50.py")

    result = benchmark(asyncio.run, get_status())
    assert result is not None


@pytest.mark.benchmark
def test_file_processing_get_by_status(benchmark, sqlite_manager):
    """Benchmark querying files by status."""
    # Pre-populate with files in different states
    async def setup():
        for i in range(MEDIUM_DATASET):
            await sqlite_manager.start_file_processing(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=ProcessingPriority.NORMAL,
                file_size=1024,
            )
            # Complete some files
            if i < 50:
                await sqlite_manager.complete_file_processing(
                    file_path=f"/test/file-{i}.py",
                    success=True,
                    processing_time_ms=100,
                )

    run_async(setup())

    async def get_by_status():
        return await sqlite_manager.get_files_by_status(
            FileProcessingStatus.PROCESSING,
            limit=100,
        )

    results = benchmark(asyncio.run, get_by_status())
    assert len(results) > 0


# ============================================================================
# SQLite State Manager - Multi-Component Operations
# ============================================================================


@pytest.mark.benchmark
def test_record_event(benchmark, sqlite_manager):
    """Benchmark event recording."""
    event = {
        "type": "file_processed",
        "file_path": "/test/file.py",
        "component": "ingestion",
        "timestamp": datetime.now(timezone.utc).timestamp(),
    }

    async def record_event():
        return await sqlite_manager.record_event(event)

    result = benchmark(asyncio.run, record_event())
    assert result is True


@pytest.mark.benchmark
def test_get_events(benchmark, sqlite_manager):
    """Benchmark event retrieval with filtering."""
    # Pre-populate events
    async def setup():
        for i in range(MEDIUM_DATASET):
            event = {
                "type": "file_processed",
                "file_path": f"/test/file-{i}.py",
                "component": "ingestion",
                "timestamp": datetime.now(timezone.utc).timestamp(),
            }
            await sqlite_manager.record_event(event)

    run_async(setup())

    async def get_events():
        return await sqlite_manager.get_events(
            filter_params={"event_type": "file_processed"}
        )

    results = benchmark(asyncio.run, get_events())
    assert len(results) == MEDIUM_DATASET


@pytest.mark.benchmark
def test_record_search_operation(benchmark, sqlite_manager):
    """Benchmark search operation recording."""

    async def record_search():
        return await sqlite_manager.record_search_operation(
            query="test query",
            results_count=10,
            source="benchmark",
            metadata={"response_time_ms": 50},
        )

    result = benchmark(asyncio.run, record_search())
    assert result is True


@pytest.mark.benchmark
def test_get_search_history(benchmark, sqlite_manager):
    """Benchmark search history retrieval."""
    # Pre-populate search history
    async def setup():
        for i in range(MEDIUM_DATASET):
            await sqlite_manager.record_search_operation(
                query=f"test query {i}",
                results_count=10,
                source="benchmark",
                metadata={"response_time_ms": 50},
            )

    run_async(setup())

    async def get_history():
        return await sqlite_manager.get_search_history(limit=50)

    results = benchmark(asyncio.run, get_history())
    assert len(results) == 50


# ============================================================================
# Comparison Benchmarks - Batch vs Single Operations
# ============================================================================


@pytest.mark.benchmark
def test_comparison_enqueue_single_vs_batch(benchmark, sqlite_manager):
    """Compare single enqueue vs batch enqueue performance."""

    async def single_operations():
        for i in range(SMALL_DATASET):
            await sqlite_manager.enqueue(
                file_path=f"/test/single-{i}.py",
                collection="test-collection",
                priority=5,
                tenant_id="test-tenant",
                branch="main",
            )

    benchmark(asyncio.run, single_operations())


@pytest.mark.benchmark
def test_comparison_watch_folder_indexed_lookup(benchmark, sqlite_manager, watch_folder_configs_large):
    """Test indexed lookup performance on large dataset."""
    # Pre-populate database
    async def setup():
        for config in watch_folder_configs_large:
            await sqlite_manager.save_watch_folder_config(config)

    run_async(setup())

    # Lookup using indexed column (watch_id is primary key)
    async def indexed_lookup():
        return await sqlite_manager.get_watch_folder_config("watch-500")

    result = benchmark(asyncio.run, indexed_lookup())
    assert result is not None


@pytest.mark.benchmark
def test_comparison_queue_priority_ordering(benchmark, sqlite_manager):
    """Test priority-based queue ordering performance."""
    # Pre-populate queue with mixed priorities
    async def setup():
        for i in range(MEDIUM_DATASET):
            priority = (i % 10) + 1  # Priorities 1-10
            await sqlite_manager.enqueue(
                file_path=f"/test/file-{i}.py",
                collection="test-collection",
                priority=priority,
                tenant_id="test-tenant",
                branch="main",
            )

    run_async(setup())

    # Dequeue should respect priority ordering (uses indexed ORDER BY)
    async def dequeue_prioritized():
        return await sqlite_manager.dequeue(batch_size=20)

    results = benchmark(asyncio.run, dequeue_prioritized())
    assert len(results) == 20


# ============================================================================
# Qdrant Vector Database - Collection Operations
# ============================================================================


@pytest.fixture
async def qdrant_client(isolated_qdrant_container):
    """Create Qdrant client for benchmarking."""
    from qdrant_client import QdrantClient

    url = isolated_qdrant_container.get_url()
    client = QdrantClient(url=url)

    yield client

    # Cleanup: delete all collections
    collections = client.get_collections().collections
    for collection in collections:
        client.delete_collection(collection.name)
    client.close()


@pytest.fixture
def vector_dimensions():
    """Standard vector dimensions for testing."""
    return 384  # sentence-transformers/all-MiniLM-L6-v2


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_collection_create(benchmark, qdrant_client, vector_dimensions):
    """Benchmark Qdrant collection creation."""
    from qdrant_client.http import models

    def create_collection():
        collection_name = f"benchmark-{benchmark.stats.stats.total}"
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_dimensions,
                distance=models.Distance.COSINE,
            ),
        )
        return collection_name

    collection_name = benchmark(create_collection)
    assert qdrant_client.collection_exists(collection_name)


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_collection_list(benchmark, qdrant_client, vector_dimensions):
    """Benchmark Qdrant collection listing."""
    from qdrant_client.http import models

    # Pre-create collections
    for i in range(20):
        qdrant_client.create_collection(
            collection_name=f"benchmark-col-{i}",
            vectors_config=models.VectorParams(
                size=vector_dimensions,
                distance=models.Distance.COSINE,
            ),
        )

    def list_collections():
        return qdrant_client.get_collections()

    result = benchmark(list_collections)
    assert len(result.collections) >= 20


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_collection_info(benchmark, qdrant_client, vector_dimensions):
    """Benchmark getting collection info."""
    from qdrant_client.http import models

    # Create collection with some points
    collection_name = "benchmark-info"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    def get_collection_info():
        return qdrant_client.get_collection(collection_name)

    result = benchmark(get_collection_info)
    assert result.config.params.vectors.size == vector_dimensions


# ============================================================================
# Qdrant Vector Database - Point Insertion Operations
# ============================================================================


@pytest.fixture
def generate_test_vectors(vector_dimensions):
    """Helper to generate test vectors."""
    import numpy as np

    def generator(count: int):
        vectors = []
        for i in range(count):
            vector = np.random.rand(vector_dimensions).tolist()
            vectors.append({
                "id": i,
                "vector": vector,
                "payload": {
                    "file_path": f"/test/file-{i}.py",
                    "content": f"Test content {i}",
                    "index": i,
                },
            })
        return vectors

    return generator


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_point_insert_single(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark single point insertion."""
    from qdrant_client.http import models

    # Create collection
    collection_name = "benchmark-insert-single"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points = generate_test_vectors(1)
    point = points[0]

    def insert_point():
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point["id"],
                    vector=point["vector"],
                    payload=point["payload"],
                )
            ],
        )

    benchmark(insert_point)

    # Verify insertion
    count = qdrant_client.count(collection_name)
    assert count.count > 0


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_point_insert_batch_small(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark batch point insertion (small dataset)."""
    from qdrant_client.http import models

    # Create collection
    collection_name = "benchmark-insert-batch-small"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(SMALL_DATASET)

    def insert_batch():
        points = [
            models.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in points_data
        ]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )

    benchmark(insert_batch)

    # Verify insertion
    count = qdrant_client.count(collection_name)
    assert count.count == SMALL_DATASET


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_point_insert_batch_medium(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark batch point insertion (medium dataset)."""
    from qdrant_client.http import models

    # Create collection
    collection_name = "benchmark-insert-batch-medium"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(MEDIUM_DATASET)

    def insert_batch():
        points = [
            models.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"],
            )
            for p in points_data
        ]
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
        )

    benchmark(insert_batch)

    # Verify insertion
    count = qdrant_client.count(collection_name)
    assert count.count == MEDIUM_DATASET


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_point_insert_batch_large(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark batch point insertion (large dataset)."""
    from qdrant_client.http import models

    # Create collection
    collection_name = "benchmark-insert-batch-large"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(LARGE_DATASET)

    def insert_batch():
        # Insert in chunks to avoid timeout
        chunk_size = 100
        for i in range(0, len(points_data), chunk_size):
            chunk = points_data[i:i + chunk_size]
            points = [
                models.PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p["payload"],
                )
                for p in chunk
            ]
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            )

    benchmark(insert_batch)

    # Verify insertion
    count = qdrant_client.count(collection_name)
    assert count.count == LARGE_DATASET


# ============================================================================
# Qdrant Vector Database - Search Operations
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_search_dense_only_small(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark dense vector search (small dataset)."""
    from qdrant_client.http import models
    import numpy as np

    # Create and populate collection
    collection_name = "benchmark-search-small"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(SMALL_DATASET)
    points = [
        models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points_data
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    # Generate query vector
    query_vector = np.random.rand(vector_dimensions).tolist()

    def search():
        return qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10,
        )

    results = benchmark(search)
    assert len(results) > 0


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_search_dense_only_medium(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark dense vector search (medium dataset)."""
    from qdrant_client.http import models
    import numpy as np

    # Create and populate collection
    collection_name = "benchmark-search-medium"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(MEDIUM_DATASET)
    points = [
        models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points_data
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    # Generate query vector
    query_vector = np.random.rand(vector_dimensions).tolist()

    def search():
        return qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=10,
        )

    results = benchmark(search)
    assert len(results) > 0


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_search_with_filter(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark search with metadata filtering."""
    from qdrant_client.http import models
    import numpy as np

    # Create and populate collection
    collection_name = "benchmark-search-filter"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(MEDIUM_DATASET)
    # Add category to payload for filtering
    for i, p in enumerate(points_data):
        p["payload"]["category"] = f"cat-{i % 5}"

    points = [
        models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points_data
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    # Generate query vector
    query_vector = np.random.rand(vector_dimensions).tolist()

    def search_with_filter():
        return qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value="cat-2"),
                    )
                ]
            ),
            limit=10,
        )

    results = benchmark(search_with_filter)
    assert len(results) > 0


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_search_large_limit(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark search with large result limit."""
    from qdrant_client.http import models
    import numpy as np

    # Create and populate collection
    collection_name = "benchmark-search-large-limit"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(MEDIUM_DATASET)
    points = [
        models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points_data
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    # Generate query vector
    query_vector = np.random.rand(vector_dimensions).tolist()

    def search():
        return qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=50,
        )

    results = benchmark(search)
    assert len(results) == 50


# ============================================================================
# Qdrant Vector Database - Point Retrieval Operations
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_retrieve_single_point(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark single point retrieval by ID."""
    from qdrant_client.http import models

    # Create and populate collection
    collection_name = "benchmark-retrieve"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(MEDIUM_DATASET)
    points = [
        models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points_data
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    def retrieve_point():
        return qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[50],
        )

    results = benchmark(retrieve_point)
    assert len(results) == 1


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_retrieve_batch_points(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark batch point retrieval by IDs."""
    from qdrant_client.http import models

    # Create and populate collection
    collection_name = "benchmark-retrieve-batch"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(MEDIUM_DATASET)
    points = [
        models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points_data
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    # Retrieve 20 points
    ids_to_retrieve = list(range(10, 30))

    def retrieve_batch():
        return qdrant_client.retrieve(
            collection_name=collection_name,
            ids=ids_to_retrieve,
        )

    results = benchmark(retrieve_batch)
    assert len(results) == 20


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_qdrant_scroll_points(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Benchmark scrolling through points."""
    from qdrant_client.http import models

    # Create and populate collection
    collection_name = "benchmark-scroll"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(MEDIUM_DATASET)
    points = [
        models.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
        for p in points_data
    ]
    qdrant_client.upsert(collection_name=collection_name, points=points)

    def scroll():
        return qdrant_client.scroll(
            collection_name=collection_name,
            limit=50,
        )

    results, _ = benchmark(scroll)
    assert len(results) == 50


# ============================================================================
# Comparison Benchmarks - Qdrant Single vs Batch Operations
# ============================================================================


@pytest.mark.benchmark
@pytest.mark.requires_qdrant
def test_comparison_qdrant_single_vs_batch_insert(benchmark, qdrant_client, vector_dimensions, generate_test_vectors):
    """Compare single insert vs batch insert performance."""
    from qdrant_client.http import models

    # Create collection
    collection_name = "benchmark-comparison-insert"
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_dimensions,
            distance=models.Distance.COSINE,
        ),
    )

    points_data = generate_test_vectors(SMALL_DATASET)

    def single_inserts():
        for p in points_data:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=p["id"],
                        vector=p["vector"],
                        payload=p["payload"],
                    )
                ],
            )

    benchmark(single_inserts)

    # Verify
    count = qdrant_client.count(collection_name)
    assert count.count == SMALL_DATASET
