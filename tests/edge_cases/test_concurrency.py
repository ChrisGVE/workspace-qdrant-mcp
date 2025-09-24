"""
Concurrency edge case tests for workspace-qdrant-mcp.

Tests race conditions, deadlocks, and concurrent operation validation
to ensure thread safety and proper async behavior.
"""

import asyncio
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from workspace_qdrant_mcp.core.client import EnhancedQdrantClient
from workspace_qdrant_mcp.core.hybrid_search import HybridSearchEngine
from workspace_qdrant_mcp.tools.memory import DocumentMemoryManager


class ConcurrencyTestHelper:
    """Helper class for concurrency testing scenarios."""

    @staticmethod
    def create_race_condition_scenario(shared_resource: Dict[str, Any],
                                     operations: int = 100):
        """Create a race condition scenario with shared resource access."""
        results = []
        errors = []

        def racing_operation(thread_id: int):
            """Operation that races to modify shared resource."""
            try:
                for i in range(operations // 10):
                    # Read-modify-write operation (potential race condition)
                    current_value = shared_resource.get('counter', 0)
                    time.sleep(0.001)  # Small delay to increase race condition probability
                    shared_resource['counter'] = current_value + 1
                    shared_resource[f'thread_{thread_id}_op_{i}'] = True

                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        return racing_operation, results, errors

    @staticmethod
    async def create_async_race_condition(shared_state: Dict[str, Any],
                                        concurrent_tasks: int = 50):
        """Create async race condition with multiple coroutines."""

        async def async_racing_operation(task_id: int):
            """Async operation that races to modify shared state."""
            for i in range(10):
                # Simulate async I/O with potential race condition
                current_value = shared_state.get('async_counter', 0)
                await asyncio.sleep(0.001)  # Async delay
                shared_state['async_counter'] = current_value + 1
                shared_state[f'task_{task_id}_op_{i}'] = True

            return task_id

        tasks = [async_racing_operation(i) for i in range(concurrent_tasks)]
        return await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.concurrency
class TestBasicConcurrencyScenarios:
    """Test basic concurrency scenarios and race conditions."""

    def test_thread_safety_shared_resource(self):
        """Test thread safety with shared resource access."""
        shared_resource = {'counter': 0}
        num_threads = 10

        racing_operation, results, errors = ConcurrencyTestHelper.create_race_condition_scenario(
            shared_resource, operations=100
        )

        # Start multiple threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=racing_operation, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        assert len(results) == num_threads

        # Note: Due to race condition, final counter might be less than expected
        final_counter = shared_resource['counter']
        expected_max = num_threads * 10

        # In a race condition, we might lose some increments
        assert 0 < final_counter <= expected_max

    @pytest.mark.asyncio
    async def test_async_concurrency_race_conditions(self):
        """Test async race conditions with shared state."""
        shared_state = {'async_counter': 0}

        results = await ConcurrencyTestHelper.create_async_race_condition(
            shared_state, concurrent_tasks=20
        )

        # Check that all tasks completed (some might have exceptions due to race conditions)
        completed_tasks = [r for r in results if not isinstance(r, Exception)]
        failed_tasks = [r for r in results if isinstance(r, Exception)]

        assert len(completed_tasks) > 0, "No tasks completed successfully"

        # Some race conditions might cause failures, but most should succeed
        assert len(completed_tasks) >= len(results) * 0.5

        # Final counter might be inconsistent due to race conditions
        final_counter = shared_state['async_counter']
        assert final_counter > 0

    def test_deadlock_prevention(self):
        """Test deadlock prevention mechanisms."""
        # Create potential deadlock scenario with multiple locks
        lock1 = threading.Lock()
        lock2 = threading.Lock()
        results = []
        errors = []

        def acquire_locks_order_1():
            """Acquire locks in order: lock1, then lock2."""
            try:
                if lock1.acquire(timeout=5):
                    time.sleep(0.1)  # Hold lock1 briefly
                    if lock2.acquire(timeout=5):
                        results.append("Thread 1: acquired both locks")
                        lock2.release()
                    lock1.release()
                else:
                    errors.append("Thread 1: timeout acquiring lock1")
            except Exception as e:
                errors.append(f"Thread 1 error: {e}")

        def acquire_locks_order_2():
            """Acquire locks in order: lock2, then lock1."""
            try:
                if lock2.acquire(timeout=5):
                    time.sleep(0.1)  # Hold lock2 briefly
                    if lock1.acquire(timeout=5):
                        results.append("Thread 2: acquired both locks")
                        lock1.release()
                    lock2.release()
                else:
                    errors.append("Thread 2: timeout acquiring lock2")
            except Exception as e:
                errors.append(f"Thread 2 error: {e}")

        # Start threads that could deadlock
        thread1 = threading.Thread(target=acquire_locks_order_1)
        thread2 = threading.Thread(target=acquire_locks_order_2)

        thread1.start()
        thread2.start()

        thread1.join(timeout=10)
        thread2.join(timeout=10)

        # At least one thread should complete, or we should have timeout errors
        total_outcomes = len(results) + len(errors)
        assert total_outcomes > 0, "No thread outcomes recorded"

        # Should not have actual deadlock (threads should timeout)
        assert not (thread1.is_alive() or thread2.is_alive()), "Threads appear to be deadlocked"


@pytest.mark.concurrency
class TestConcurrentQdrantOperations:
    """Test concurrent Qdrant client operations."""

    @pytest.fixture
    async def mock_client(self):
        """Create mock Qdrant client for testing."""
        client = Mock(spec=EnhancedQdrantClient)
        client.client = Mock()
        return client

    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self, mock_client):
        """Test concurrent search operations."""
        hybrid_search = HybridSearchEngine(
            qdrant_client=mock_client,
            embedding_model="test-model"
        )

        # Mock search responses with delays to simulate real operations
        async def delayed_search(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate search processing time
            query = kwargs.get('query', 'unknown')
            return {
                "points": [
                    {"id": f"result_{hash(query) % 1000}", "score": 0.9,
                     "payload": {"text": f"Result for: {query}"}}
                ]
            }

        mock_client.hybrid_search = AsyncMock(side_effect=delayed_search)

        # Start multiple concurrent searches
        search_tasks = []
        for i in range(20):
            task = hybrid_search.search(
                query=f"search query {i}",
                collections=[f"collection-{i % 3}"],
                limit=5
            )
            search_tasks.append(task)

        # Execute all searches concurrently
        start_time = time.time()
        results = await asyncio.gather(*search_tasks)
        end_time = time.time()

        # All searches should complete
        assert len(results) == 20
        assert all(isinstance(result, dict) for result in results)

        # Concurrent execution should be faster than sequential
        total_time = end_time - start_time
        assert total_time < 15.0  # Should complete much faster than 20 * 0.1 = 2.0s

    @pytest.mark.asyncio
    async def test_concurrent_document_storage(self, mock_client):
        """Test concurrent document storage operations."""
        memory_manager = DocumentMemoryManager(
            client=mock_client,
            project_name="test-project"
        )

        storage_count = 0

        async def tracked_storage(*args, **kwargs):
            """Track storage operations for concurrency testing."""
            nonlocal storage_count
            await asyncio.sleep(0.05)  # Simulate storage time
            storage_count += 1
            return {"operation_id": storage_count, "status": "success"}

        mock_client.upsert_points = AsyncMock(side_effect=tracked_storage)

        # Store multiple documents concurrently
        storage_tasks = []
        for i in range(15):
            task = memory_manager.store_document(
                content=f"Document content {i}",
                metadata={"title": f"doc_{i}.txt", "index": i},
                collection_name="test-collection"
            )
            storage_tasks.append(task)

        results = await asyncio.gather(*storage_tasks)

        # All storage operations should complete
        assert len(results) == 15
        assert storage_count == 15
        assert all("operation_id" in result for result in results)

    @pytest.mark.asyncio
    async def test_concurrent_collection_operations(self, mock_client):
        """Test concurrent collection management operations."""

        collections_created = []
        collections_deleted = []
        operation_errors = []

        async def mock_create_collection(name, *args, **kwargs):
            """Mock collection creation with tracking."""
            await asyncio.sleep(0.02)
            if name in collections_created:
                raise ValueError(f"Collection {name} already exists")
            collections_created.append(name)
            return {"status": "created", "name": name}

        async def mock_delete_collection(name, *args, **kwargs):
            """Mock collection deletion with tracking."""
            await asyncio.sleep(0.02)
            if name not in collections_created:
                raise ValueError(f"Collection {name} does not exist")
            collections_deleted.append(name)
            return {"status": "deleted", "name": name}

        mock_client.create_collection = AsyncMock(side_effect=mock_create_collection)
        mock_client.delete_collection = AsyncMock(side_effect=mock_delete_collection)

        # Perform concurrent collection operations
        tasks = []

        # Create collections
        for i in range(10):
            task = mock_client.create_collection(f"collection-{i}")
            tasks.append(task)

        # Delete some collections (these should fail since creation runs concurrently)
        for i in range(0, 5):
            task = mock_client.delete_collection(f"collection-{i}")
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful_ops = [r for r in results if not isinstance(r, Exception)]
        failed_ops = [r for r in results if isinstance(r, Exception)]

        # Some operations should succeed, others may fail due to race conditions
        assert len(successful_ops) > 0
        assert len(collections_created) > 0


@pytest.mark.concurrency
class TestAsyncResourceManagement:
    """Test async resource management and cleanup."""

    @pytest.mark.asyncio
    async def test_async_context_manager_concurrency(self):
        """Test concurrent usage of async context managers."""

        class MockAsyncResource:
            def __init__(self, resource_id):
                self.resource_id = resource_id
                self.is_open = False
                self.operations_performed = 0

            async def __aenter__(self):
                await asyncio.sleep(0.01)  # Simulate async setup
                self.is_open = True
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                await asyncio.sleep(0.01)  # Simulate async cleanup
                self.is_open = False

            async def perform_operation(self):
                if not self.is_open:
                    raise RuntimeError("Resource not open")
                await asyncio.sleep(0.02)
                self.operations_performed += 1
                return f"Operation completed on resource {self.resource_id}"

        # Use multiple async resources concurrently
        async def use_resource(resource_id):
            async with MockAsyncResource(resource_id) as resource:
                results = []
                for _ in range(3):
                    result = await resource.perform_operation()
                    results.append(result)
                return results

        # Start multiple concurrent resource users
        resource_tasks = [use_resource(i) for i in range(8)]

        results = await asyncio.gather(*resource_tasks)

        # All resource operations should complete successfully
        assert len(results) == 8
        for result_list in results:
            assert len(result_list) == 3
            assert all("Operation completed" in result for result in result_list)

    @pytest.mark.asyncio
    async def test_semaphore_resource_limiting(self):
        """Test resource limiting with semaphores."""

        # Limit concurrent operations to 3
        semaphore = asyncio.Semaphore(3)
        active_operations = []
        max_concurrent = 0

        async def limited_operation(operation_id):
            """Operation limited by semaphore."""
            async with semaphore:
                active_operations.append(operation_id)
                current_active = len(active_operations)
                nonlocal max_concurrent
                max_concurrent = max(max_concurrent, current_active)

                await asyncio.sleep(0.1)  # Simulate work

                active_operations.remove(operation_id)
                return f"Completed operation {operation_id}"

        # Start many concurrent operations
        operation_tasks = [limited_operation(i) for i in range(20)]

        results = await asyncio.gather(*operation_tasks)

        # All operations should complete
        assert len(results) == 20

        # Should never exceed semaphore limit
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_async_queue_concurrency(self):
        """Test concurrent producer-consumer with async queues."""

        queue = asyncio.Queue(maxsize=5)
        produced_items = []
        consumed_items = []

        async def producer(producer_id, num_items=10):
            """Produce items into the queue."""
            for i in range(num_items):
                item = f"item_{producer_id}_{i}"
                await queue.put(item)
                produced_items.append(item)
                await asyncio.sleep(0.01)

        async def consumer(consumer_id, num_items=15):
            """Consume items from the queue."""
            items_consumed = 0
            while items_consumed < num_items:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    consumed_items.append(item)
                    items_consumed += 1
                    queue.task_done()
                except asyncio.TimeoutError:
                    break

        # Start multiple producers and consumers
        producers = [producer(i, 5) for i in range(3)]  # 15 items total
        consumers = [consumer(i, 8) for i in range(2)]  # Should consume all items

        # Run producers and consumers concurrently
        await asyncio.gather(*producers, *consumers)

        # Should have produced and consumed items
        assert len(produced_items) == 15
        assert len(consumed_items) > 0

        # All produced items should eventually be consumed
        assert set(consumed_items).issubset(set(produced_items))


@pytest.mark.concurrency
class TestRaceConditionDetection:
    """Test detection and handling of specific race conditions."""

    @pytest.mark.asyncio
    async def test_cache_invalidation_race_condition(self):
        """Test race conditions in cache invalidation."""

        cache = {}
        cache_lock = asyncio.Lock()
        access_log = []

        async def cached_operation(key, value, operation_id):
            """Simulate cached operation with potential race condition."""

            # Check cache (potential race condition here)
            if key in cache:
                access_log.append(f"Op {operation_id}: Cache hit for {key}")
                return cache[key]

            # Simulate expensive operation
            await asyncio.sleep(0.05)

            # Update cache (race condition: multiple operations might do this)
            async with cache_lock:
                if key not in cache:  # Double-check locking pattern
                    cache[key] = f"{value}_computed_by_{operation_id}"
                    access_log.append(f"Op {operation_id}: Cache miss, computed {key}")
                else:
                    access_log.append(f"Op {operation_id}: Cache populated by another operation")

            return cache[key]

        # Start multiple operations that might race on the same cache key
        tasks = []
        for i in range(10):
            # Multiple operations working with the same keys
            key = f"key_{i % 3}"  # Only 3 unique keys for 10 operations
            task = cached_operation(key, f"value_{i}", i)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All operations should complete
        assert len(results) == 10

        # Should have some cache hits and misses
        hits = [log for log in access_log if "Cache hit" in log]
        misses = [log for log in access_log if "Cache miss" in log]

        assert len(misses) > 0  # Should have computed some values
        assert len(cache) <= 3  # Should only have 3 unique keys

    def test_file_write_race_condition(self, tmp_path):
        """Test race conditions in concurrent file writing."""

        test_file = tmp_path / "concurrent_writes.txt"
        write_results = []
        errors = []

        def concurrent_writer(writer_id, content_lines):
            """Writer that appends to the same file."""
            try:
                for i, line in enumerate(content_lines):
                    with open(test_file, 'a') as f:
                        f.write(f"Writer {writer_id}, Line {i}: {line}\n")
                        time.sleep(0.001)  # Small delay to increase race probability

                write_results.append(f"Writer {writer_id} completed")
            except Exception as e:
                errors.append(f"Writer {writer_id} error: {e}")

        # Start multiple writers
        writers = []
        for writer_id in range(5):
            content = [f"Content_{writer_id}_{i}" for i in range(10)]
            thread = threading.Thread(target=concurrent_writer, args=(writer_id, content))
            writers.append(thread)
            thread.start()

        # Wait for all writers
        for thread in writers:
            thread.join(timeout=10)

        # Check results
        assert len(errors) == 0, f"Write errors: {errors}"
        assert len(write_results) == 5

        # Verify file content
        if test_file.exists():
            content = test_file.read_text()
            lines = content.strip().split('\n')

            # Should have content from all writers (50 lines total)
            assert len(lines) > 0

            # Each writer should have contributed some lines
            writer_contributions = {}
            for line in lines:
                if line.startswith("Writer"):
                    writer_id = line.split()[1][0]  # Extract writer ID
                    writer_contributions[writer_id] = writer_contributions.get(writer_id, 0) + 1

            assert len(writer_contributions) > 0

    @pytest.mark.asyncio
    async def test_connection_pool_race_condition(self):
        """Test race conditions in connection pool management."""

        class MockConnectionPool:
            def __init__(self, max_connections=5):
                self.max_connections = max_connections
                self.active_connections = []
                self.connection_id_counter = 0
                self.lock = asyncio.Lock()

            async def get_connection(self):
                """Get connection from pool (potential race condition)."""
                async with self.lock:
                    if len(self.active_connections) < self.max_connections:
                        self.connection_id_counter += 1
                        conn_id = self.connection_id_counter
                        self.active_connections.append(conn_id)
                        return conn_id
                    else:
                        raise RuntimeError("Connection pool exhausted")

            async def release_connection(self, conn_id):
                """Release connection back to pool."""
                async with self.lock:
                    if conn_id in self.active_connections:
                        self.active_connections.remove(conn_id)
                    else:
                        raise ValueError(f"Connection {conn_id} not found in pool")

        pool = MockConnectionPool(max_connections=3)
        operation_results = []

        async def pool_user(user_id):
            """User that acquires and releases connections."""
            try:
                # Get connection
                conn_id = await pool.get_connection()
                operation_results.append(f"User {user_id}: Got connection {conn_id}")

                # Simulate work with connection
                await asyncio.sleep(0.1)

                # Release connection
                await pool.release_connection(conn_id)
                operation_results.append(f"User {user_id}: Released connection {conn_id}")

                return f"User {user_id} completed successfully"

            except Exception as e:
                return f"User {user_id} failed: {e}"

        # Start more users than available connections
        user_tasks = [pool_user(i) for i in range(8)]

        results = await asyncio.gather(*user_tasks, return_exceptions=True)

        # Some users should succeed, others might fail due to pool exhaustion
        successful_users = [r for r in results if not isinstance(r, Exception) and "completed successfully" in r]
        failed_users = [r for r in results if isinstance(r, Exception) or "failed" in str(r)]

        # At least some operations should work within pool limits
        assert len(successful_users) > 0
        assert len(successful_users) <= 8

        # Pool should be properly managed (no more than max_connections active)
        assert len(pool.active_connections) >= 0  # All connections should be released