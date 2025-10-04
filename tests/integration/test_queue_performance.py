"""
Performance and Load Tests for Queue System.

Tests queue system performance under various load conditions:
- High volume ingestion
- Concurrent processing
- Resource utilization
- Throughput and latency benchmarks
- Stress testing
- Memory and CPU profiling

Test Coverage:
    - High volume file ingestion (100+ files)
    - Concurrent worker simulation
    - Queue depth scaling
    - Priority queue performance
    - Error handling under load
    - Resource leak detection
    - Performance degradation tests
"""

import asyncio
import gc
import psutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from unittest.mock import patch

import pytest

from src.python.common.core.queue_client import (
    SQLiteQueueClient,
    QueueOperation,
)
from src.python.common.core.priority_queue_manager import (
    PriorityQueueManager,
    MCPActivityLevel,
    ProcessingMode,
)
from src.python.common.core.sqlite_state_manager import SQLiteStateManager
from src.python.common.core.queue_connection import ConnectionConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
async def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    try:
        Path(db_path).unlink()
        Path(f"{db_path}-wal").unlink(missing_ok=True)
        Path(f"{db_path}-shm").unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
async def initialized_db(temp_db):
    """Initialize database with all required schemas."""
    conn = sqlite3.connect(temp_db)
    conn.execute("PRAGMA journal_mode=WAL")

    # Load queue schema
    queue_schema_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "python"
        / "common"
        / "core"
        / "queue_schema.sql"
    )
    with open(queue_schema_path, "r") as f:
        conn.executescript(f.read())

    # Load error messages schema
    error_schema_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "python"
        / "common"
        / "core"
        / "error_messages_schema.sql"
    )
    with open(error_schema_path, "r") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()

    yield temp_db


@pytest.fixture
async def queue_client(initialized_db):
    """Initialize SQLiteQueueClient with optimized connection pool."""
    client = SQLiteQueueClient(
        db_path=initialized_db,
        connection_config=ConnectionConfig(
            max_connections=10,
            timeout=30.0,
            check_same_thread=False,
        ),
    )
    await client.initialize()
    yield client
    await client.close()


@pytest.fixture
def large_file_set(tmp_path):
    """Create large set of test files for performance testing."""
    files = []
    for i in range(100):
        test_file = tmp_path / f"perf_test_file_{i:03d}.py"
        content = f"# Performance test file {i}\n" + ("x = 1\n" * 50)
        test_file.write_text(content)
        files.append(str(test_file))
    return files


@pytest.fixture
def memory_tracker():
    """Track memory usage during tests."""
    process = psutil.Process()

    class MemoryTracker:
        def __init__(self):
            self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.initial_memory
            self.samples = []

        def sample(self):
            current = process.memory_info().rss / 1024 / 1024  # MB
            self.samples.append(current)
            self.peak_memory = max(self.peak_memory, current)
            return current

        def get_stats(self):
            return {
                "initial_mb": self.initial_memory,
                "peak_mb": self.peak_memory,
                "final_mb": self.samples[-1] if self.samples else self.initial_memory,
                "delta_mb": self.peak_memory - self.initial_memory,
            }

    tracker = MemoryTracker()
    yield tracker

    # Force garbage collection
    gc.collect()


# =============================================================================
# PERFORMANCE BENCHMARKS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_high_volume_enqueue_performance(queue_client, large_file_set):
    """Test enqueueing large number of files."""
    start_time = time.time()

    # Enqueue 100 files
    for file_path in large_file_set:
        await queue_client.enqueue_file(
            file_path=file_path,
            collection="perf-test",
            priority=5,
        )

    elapsed = time.time() - start_time

    # Verify all enqueued
    depth = await queue_client.get_queue_depth()
    assert depth == len(large_file_set)

    # Performance assertions
    throughput = len(large_file_set) / elapsed
    assert throughput > 10.0, f"Enqueue throughput too low: {throughput:.2f} files/sec"

    print(f"\nEnqueue Performance:")
    print(f"  Total files: {len(large_file_set)}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} files/sec")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_batch_enqueue_performance(queue_client, large_file_set):
    """Test batch enqueue performance."""
    items = [
        {"file_path": fp, "collection": "perf-test", "priority": 5}
        for fp in large_file_set
    ]

    start_time = time.time()

    successful, failed = await queue_client.enqueue_batch(items=items)

    elapsed = time.time() - start_time

    assert successful == len(large_file_set)
    assert len(failed) == 0

    # Performance assertions
    throughput = successful / elapsed
    assert throughput > 20.0, f"Batch enqueue throughput too low: {throughput:.2f} files/sec"

    print(f"\nBatch Enqueue Performance:")
    print(f"  Total files: {successful}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} files/sec")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_dequeue_performance(queue_client, large_file_set):
    """Test dequeue performance under load."""
    # Enqueue files
    items = [
        {"file_path": fp, "collection": "perf-test", "priority": i % 10}
        for i, fp in enumerate(large_file_set)
    ]
    await queue_client.enqueue_batch(items=items)

    # Dequeue in batches
    start_time = time.time()
    total_dequeued = 0

    while total_dequeued < len(large_file_set):
        batch = await queue_client.dequeue_batch(batch_size=20)
        total_dequeued += len(batch)

        if not batch:
            break

    elapsed = time.time() - start_time

    assert total_dequeued == len(large_file_set)

    # Performance assertions
    throughput = total_dequeued / elapsed
    assert throughput > 50.0, f"Dequeue throughput too low: {throughput:.2f} files/sec"

    print(f"\nDequeue Performance:")
    print(f"  Total files: {total_dequeued}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} files/sec")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_priority_queue_sorting_performance(queue_client, large_file_set):
    """Test priority queue sorting with many items."""
    # Enqueue with random priorities
    import random

    items = [
        {"file_path": fp, "collection": "perf-test", "priority": random.randint(1, 9)}
        for fp in large_file_set
    ]
    await queue_client.enqueue_batch(items=items)

    # Dequeue all and verify ordering
    start_time = time.time()

    all_items = await queue_client.dequeue_batch(batch_size=len(large_file_set))

    elapsed = time.time() - start_time

    # Verify priority ordering (DESC)
    priorities = [item.priority for item in all_items]
    assert priorities == sorted(priorities, reverse=True)

    print(f"\nPriority Sorting Performance:")
    print(f"  Total files: {len(all_items)}")
    print(f"  Sort time: {elapsed * 1000:.2f}ms")


# =============================================================================
# CONCURRENT PROCESSING TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_worker_simulation(queue_client, large_file_set):
    """Simulate multiple workers processing queue concurrently."""
    # Enqueue files
    items = [
        {"file_path": fp, "collection": "perf-test", "priority": 5}
        for fp in large_file_set
    ]
    await queue_client.enqueue_batch(items=items)

    # Simulate workers
    num_workers = 5
    processed_counts = []

    async def worker(worker_id):
        count = 0
        while True:
            items = await queue_client.dequeue_batch(batch_size=5)
            if not items:
                break

            for item in items:
                # Simulate processing
                await asyncio.sleep(0.001)
                await queue_client.mark_complete(
                    file_path=item.file_absolute_path,
                    processing_time_ms=1.0,
                )
                count += 1

        return count

    start_time = time.time()

    # Run workers concurrently
    results = await asyncio.gather(*[worker(i) for i in range(num_workers)])

    elapsed = time.time() - start_time

    total_processed = sum(results)
    assert total_processed == len(large_file_set)

    # Verify queue empty
    depth = await queue_client.get_queue_depth()
    assert depth == 0

    throughput = total_processed / elapsed
    print(f"\nConcurrent Workers Performance:")
    print(f"  Workers: {num_workers}")
    print(f"  Total processed: {total_processed}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} files/sec")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_concurrent_enqueue_and_process(queue_client, tmp_path):
    """Test concurrent enqueue and process operations."""
    files_to_create = 50

    async def enqueue_worker():
        for i in range(files_to_create):
            test_file = tmp_path / f"concurrent_{i}.py"
            test_file.write_text(f"# File {i}\n")

            await queue_client.enqueue_file(
                file_path=str(test_file),
                collection="concurrent-test",
                priority=5,
            )
            await asyncio.sleep(0.01)  # Simulate file generation delay

    async def process_worker():
        processed = 0
        while processed < files_to_create:
            items = await queue_client.dequeue_batch(batch_size=5)

            for item in items:
                await queue_client.mark_complete(
                    file_path=item.file_absolute_path,
                    processing_time_ms=5.0,
                )
                processed += 1

            if not items:
                await asyncio.sleep(0.05)  # Wait for more items

        return processed

    start_time = time.time()

    # Run enqueue and process concurrently
    enqueue_task = asyncio.create_task(enqueue_worker())
    process_task = asyncio.create_task(process_worker())

    await enqueue_task
    processed_count = await process_task

    elapsed = time.time() - start_time

    assert processed_count == files_to_create

    print(f"\nConcurrent Enqueue/Process:")
    print(f"  Files: {files_to_create}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {processed_count / elapsed:.2f} files/sec")


# =============================================================================
# RESOURCE UTILIZATION TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_memory_usage_under_load(queue_client, large_file_set, memory_tracker):
    """Test memory usage with large queue."""
    memory_tracker.sample()  # Initial

    # Enqueue all files
    items = [
        {"file_path": fp, "collection": "perf-test", "priority": 5}
        for fp in large_file_set
    ]
    await queue_client.enqueue_batch(items=items)

    memory_tracker.sample()  # After enqueue

    # Process all files
    while True:
        items = await queue_client.dequeue_batch(batch_size=20)
        if not items:
            break

        for item in items:
            await queue_client.mark_complete(
                file_path=item.file_absolute_path,
                processing_time_ms=1.0,
            )

        memory_tracker.sample()  # During processing

    memory_tracker.sample()  # Final

    stats = memory_tracker.get_stats()

    # Memory should not grow excessively
    assert stats["delta_mb"] < 100, f"Memory delta too high: {stats['delta_mb']:.2f}MB"

    print(f"\nMemory Usage:")
    print(f"  Initial: {stats['initial_mb']:.2f}MB")
    print(f"  Peak: {stats['peak_mb']:.2f}MB")
    print(f"  Final: {stats['final_mb']:.2f}MB")
    print(f"  Delta: {stats['delta_mb']:.2f}MB")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_connection_pool_under_load(queue_client, large_file_set):
    """Test connection pool behavior under concurrent load."""

    async def concurrent_operation(file_path):
        # Each operation does enqueue → dequeue → complete
        queue_id = await queue_client.enqueue_file(
            file_path=file_path,
            collection="pool-test",
            priority=5,
        )

        items = await queue_client.dequeue_batch(batch_size=1)

        if items:
            await queue_client.mark_complete(
                file_path=items[0].file_absolute_path,
                processing_time_ms=1.0,
            )

    start_time = time.time()

    # Run many concurrent operations
    await asyncio.gather(*[concurrent_operation(fp) for fp in large_file_set[:50]])

    elapsed = time.time() - start_time

    # Verify all processed
    depth = await queue_client.get_queue_depth()
    assert depth == 0

    print(f"\nConnection Pool Performance:")
    print(f"  Operations: 50")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Avg latency: {elapsed * 1000 / 50:.2f}ms")


# =============================================================================
# STRESS TESTS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_rapid_priority_updates(queue_client, large_file_set):
    """Test rapid priority update operations."""
    # Enqueue files
    items = [
        {"file_path": fp, "collection": "stress-test", "priority": 5}
        for fp in large_file_set[:50]
    ]
    await queue_client.enqueue_batch(items=items)

    # Rapidly update priorities
    start_time = time.time()

    for i in range(10):  # 10 iterations of priority updates
        for file_path in large_file_set[:50]:
            new_priority = (i % 9) + 1
            await queue_client.update_priority(
                file_path=file_path,
                new_priority=new_priority,
            )

    elapsed = time.time() - start_time

    updates_count = 50 * 10
    throughput = updates_count / elapsed

    print(f"\nPriority Update Stress Test:")
    print(f"  Total updates: {updates_count}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.2f} updates/sec")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_error_handling_under_load(queue_client, large_file_set):
    """Test error handling with high volume of failures."""
    # Enqueue files
    items = [
        {"file_path": fp, "collection": "error-stress", "priority": 5}
        for fp in large_file_set
    ]
    await queue_client.enqueue_batch(items=items)

    start_time = time.time()

    # Process with errors
    error_count = 0
    success_count = 0

    while True:
        items = await queue_client.dequeue_batch(batch_size=10)
        if not items:
            break

        for item in items:
            # Simulate 50% failure rate
            if hash(item.file_absolute_path) % 2 == 0:
                await queue_client.mark_error(
                    file_path=item.file_absolute_path,
                    exception=RuntimeError("Simulated error"),
                    max_retries=1,
                )
                error_count += 1
            else:
                await queue_client.mark_complete(
                    file_path=item.file_absolute_path,
                    processing_time_ms=1.0,
                )
                success_count += 1

    elapsed = time.time() - start_time

    print(f"\nError Handling Stress Test:")
    print(f"  Total items: {len(large_file_set)}")
    print(f"  Errors: {error_count}")
    print(f"  Successes: {success_count}")
    print(f"  Time: {elapsed:.2f}s")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_queue_depth_scaling(queue_client, tmp_path):
    """Test queue behavior as depth scales."""
    depths_to_test = [10, 50, 100, 200]
    results = []

    for target_depth in depths_to_test:
        # Clear queue
        await queue_client.clear_queue()

        # Create and enqueue files
        files = []
        for i in range(target_depth):
            test_file = tmp_path / f"scale_test_{i}.py"
            test_file.write_text(f"# File {i}\n")
            files.append(str(test_file))

        start_time = time.time()

        items = [
            {"file_path": fp, "collection": "scale-test", "priority": 5}
            for fp in files
        ]
        await queue_client.enqueue_batch(items=items)

        enqueue_time = time.time() - start_time

        # Dequeue all
        start_time = time.time()
        await queue_client.dequeue_batch(batch_size=target_depth)
        dequeue_time = time.time() - start_time

        results.append({
            "depth": target_depth,
            "enqueue_ms": enqueue_time * 1000,
            "dequeue_ms": dequeue_time * 1000,
        })

    print(f"\nQueue Depth Scaling:")
    for result in results:
        print(
            f"  Depth {result['depth']:3d}: "
            f"enqueue {result['enqueue_ms']:6.2f}ms, "
            f"dequeue {result['dequeue_ms']:6.2f}ms"
        )

    # Verify performance doesn't degrade significantly
    # Dequeue time should scale roughly linearly
    for i in range(1, len(results)):
        depth_ratio = results[i]["depth"] / results[i - 1]["depth"]
        time_ratio = results[i]["dequeue_ms"] / results[i - 1]["dequeue_ms"]

        # Time ratio should not be more than 2x the depth ratio
        assert time_ratio < depth_ratio * 2, (
            f"Performance degradation detected: "
            f"depth ratio {depth_ratio:.2f}, time ratio {time_ratio:.2f}"
        )


# =============================================================================
# THROUGHPUT BENCHMARKS
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.slow
async def test_end_to_end_throughput(queue_client, large_file_set):
    """Test complete end-to-end throughput."""
    start_time = time.time()

    # Enqueue all
    items = [
        {"file_path": fp, "collection": "throughput-test", "priority": 5}
        for fp in large_file_set
    ]
    await queue_client.enqueue_batch(items=items)

    enqueue_time = time.time()

    # Process all
    processed = 0
    while processed < len(large_file_set):
        items = await queue_client.dequeue_batch(batch_size=20)

        for item in items:
            await queue_client.mark_complete(
                file_path=item.file_absolute_path,
                processing_time_ms=1.0,
            )
            processed += 1

    complete_time = time.time()

    total_time = complete_time - start_time
    enqueue_elapsed = enqueue_time - start_time
    process_elapsed = complete_time - enqueue_time

    throughput = len(large_file_set) / total_time

    print(f"\nEnd-to-End Throughput:")
    print(f"  Total files: {len(large_file_set)}")
    print(f"  Enqueue time: {enqueue_elapsed:.2f}s")
    print(f"  Process time: {process_elapsed:.2f}s")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.2f} files/sec")

    # Performance baseline
    assert throughput > 15.0, f"Throughput below baseline: {throughput:.2f} files/sec"
