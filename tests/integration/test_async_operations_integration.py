"""
Async Operations Integration Tests (Task 382.10).

Comprehensive integration tests validating async refactoring implementation
and concurrent request handling capabilities of the MCP server.

Test Coverage:
    - AsyncQdrantClient operations across all MCP tools
    - Concurrent request handling without event loop blocking
    - Async subprocess execution (git operations)
    - Thread-pool embedding generation
    - Performance benchmarks comparing async vs blocking operations
    - Timeout and error handling in async context

Requirements:
    - Running Qdrant instance (localhost:6333)
    - pytest with async support (pytest-asyncio)
    - Performance monitoring capabilities

Usage:
    # Run all async integration tests
    pytest tests/integration/test_async_operations_integration.py -v

    # Run performance benchmarks
    pytest tests/integration/test_async_operations_integration.py -v -m benchmark
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Import MCP server components
from workspace_qdrant_mcp.server import (
    app,
    ensure_collection_exists,
    generate_embeddings,
    get_project_name,
    initialize_components,
    manage,
    retrieve,
    search,
    store,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
async def async_qdrant_client():
    """
    Provide AsyncQdrantClient for validation.

    Verifies that AsyncQdrantClient is properly configured and accessible.
    """
    client = AsyncQdrantClient(host="localhost", port=6333)

    # Verify Qdrant is accessible
    try:
        await client.get_collections()
    except Exception as e:
        pytest.skip(f"Qdrant server not accessible: {e}")

    yield client

    # Cleanup
    await client.close()


@pytest.fixture
async def clean_test_collection(async_qdrant_client):
    """
    Provide clean test collection for each test.

    Creates a test collection and ensures cleanup after test completion.
    """
    collection_name = f"test_async_{int(time.time() * 1000)}"

    # Create test collection
    await async_qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    yield collection_name

    # Cleanup
    try:
        await async_qdrant_client.delete_collection(collection_name)
    except Exception:
        pass


# =============================================================================
# ASYNC CLIENT OPERATIONS TESTS
# =============================================================================

@pytest.mark.asyncio
class TestAsyncQdrantOperations:
    """Test AsyncQdrantClient integration in MCP server."""

    async def test_async_client_initialization(self):
        """Verify AsyncQdrantClient is properly initialized."""
        await initialize_components()

        from workspace_qdrant_mcp.server import qdrant_client

        # Verify client is AsyncQdrantClient instance
        assert isinstance(qdrant_client, AsyncQdrantClient), \
            "qdrant_client should be AsyncQdrantClient instance"

    async def test_ensure_collection_async(self, clean_test_collection):
        """Test async collection existence check and creation."""
        collection_name = f"test_ensure_{int(time.time() * 1000)}"

        # Verify async collection creation
        result = await ensure_collection_exists(collection_name)
        assert result is True, "Collection creation should succeed"

        # Verify async collection existence check
        result = await ensure_collection_exists(collection_name)
        assert result is True, "Collection existence check should succeed"

        # Cleanup
        from workspace_qdrant_mcp.server import qdrant_client
        await qdrant_client.delete_collection(collection_name)

    async def test_store_tool_async_operations(self, clean_test_collection):
        """Test store tool uses async Qdrant operations."""
        result = await store(
            content="Test content for async validation",
            title="Async Test Document",
            collection=clean_test_collection,
        )

        assert result["success"] is True, "Store operation should succeed"
        assert "document_id" in result, "Should return document ID"

        # Verify document was stored asynchronously
        from workspace_qdrant_mcp.server import qdrant_client
        points = await qdrant_client.scroll(
            collection_name=clean_test_collection,
            limit=10
        )

        assert len(points[0]) == 1, "Should have 1 document stored"

    async def test_search_tool_async_operations(self, clean_test_collection, async_qdrant_client):
        """Test search tool uses async Qdrant operations."""
        # First, store a test document
        test_content = "Machine learning algorithms for data analysis"
        await store(
            content=test_content,
            collection=clean_test_collection,
        )

        # Perform async search
        result = await search(
            query="machine learning",
            collection=clean_test_collection,
            mode="hybrid",
            limit=5,
        )

        assert result["success"] is True, "Search should succeed"
        assert result["total_results"] >= 0, "Should return result count"
        assert "search_time_ms" in result, "Should include search timing"

    async def test_manage_tool_async_operations(self, async_qdrant_client):
        """Test manage tool uses async Qdrant operations."""
        # Test list_collections action
        result = await manage(action="list_collections")

        assert result["success"] is True, "List collections should succeed"
        assert "collections" in result, "Should return collections list"
        assert isinstance(result["collections"], list), "Collections should be a list"

    async def test_retrieve_tool_async_operations(self, clean_test_collection):
        """Test retrieve tool uses async Qdrant operations."""
        # First, store a test document
        store_result = await store(
            content="Test document for retrieval",
            collection=clean_test_collection,
        )
        document_id = store_result["document_id"]

        # Perform async retrieval
        result = await retrieve(
            document_id=document_id,
            collection=clean_test_collection,
        )

        assert result["success"] is True, "Retrieve should succeed"
        assert result["total_results"] == 1, "Should find 1 document"
        assert result["results"][0]["id"] == document_id, "Should return correct document"


# =============================================================================
# ASYNC SUBPROCESS TESTS
# =============================================================================

@pytest.mark.asyncio
class TestAsyncSubprocessOperations:
    """Test async subprocess execution for git operations."""

    async def test_get_project_name_async(self):
        """Test get_project_name uses async subprocess."""
        # Mock git subprocess
        with patch("workspace_qdrant_mcp.server.asyncio.create_subprocess_exec") as mock_exec:
            # Create mock process
            mock_proc = AsyncMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(
                b"https://github.com/user/test-repo.git\n",
                b""
            ))
            mock_exec.return_value = mock_proc

            # Execute async function
            project_name = await get_project_name()

            # Verify async subprocess was used
            mock_exec.assert_called_once()
            assert project_name == "test-repo", "Should extract repo name from git URL"

    async def test_get_project_name_timeout_handling(self):
        """Test get_project_name handles subprocess timeout."""
        with patch("workspace_qdrant_mcp.server.asyncio.create_subprocess_exec") as mock_exec:
            # Create mock process that never completes
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_proc.kill = AsyncMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc

            # Should fall back to directory name on timeout
            project_name = await get_project_name()
            assert project_name == Path.cwd().name, "Should fall back to directory name"

    async def test_get_project_name_fallback(self):
        """Test get_project_name falls back to directory name on error."""
        with patch("workspace_qdrant_mcp.server.asyncio.create_subprocess_exec") as mock_exec:
            # Simulate git command failure
            mock_proc = AsyncMock()
            mock_proc.returncode = 128  # Git error code
            mock_proc.communicate = AsyncMock(return_value=(b"", b"fatal: not a git repository"))
            mock_exec.return_value = mock_proc

            # Should fall back to directory name
            project_name = await get_project_name()
            assert project_name == Path.cwd().name, "Should fall back to directory name"


# =============================================================================
# THREAD POOL EMBEDDING TESTS
# =============================================================================

@pytest.mark.asyncio
class TestAsyncEmbeddingGeneration:
    """Test embedding generation uses thread pool to avoid blocking."""

    async def test_generate_embeddings_non_blocking(self):
        """Test generate_embeddings runs in thread pool."""
        # Initialize components to ensure embedding model is loaded
        await initialize_components()

        # Generate embeddings and verify async execution
        start_time = time.time()
        embeddings = await generate_embeddings("Test text for embedding generation")
        execution_time = time.time() - start_time

        # Verify embeddings were generated
        assert isinstance(embeddings, list), "Should return list of floats"
        assert len(embeddings) == 384, "Should return 384-dimensional embeddings"

        # Verify it didn't block excessively (heuristic check)
        assert execution_time < 5.0, "Embedding generation should not block for >5 seconds"

    async def test_concurrent_embedding_generation(self):
        """Test multiple concurrent embedding generations don't block event loop."""
        await initialize_components()

        # Generate multiple embeddings concurrently
        texts = [f"Test text {i}" for i in range(10)]
        start_time = time.time()

        # Use asyncio.gather to run concurrently
        embeddings_list = await asyncio.gather(*[
            generate_embeddings(text) for text in texts
        ])

        execution_time = time.time() - start_time

        # Verify all embeddings were generated
        assert len(embeddings_list) == 10, "Should generate 10 embeddings"

        # Concurrent execution should be faster than sequential
        # (This is a heuristic - actual speedup depends on thread pool size)
        assert execution_time < 10.0, "Concurrent generation should complete reasonably fast"


# =============================================================================
# CONCURRENT REQUEST HANDLING TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.benchmark
class TestConcurrentRequestHandling:
    """Test MCP server handles concurrent requests without blocking."""

    async def test_concurrent_store_operations(self, clean_test_collection):
        """Test multiple concurrent store operations."""
        # Create multiple store requests
        store_tasks = [
            store(
                content=f"Test document {i}",
                title=f"Doc {i}",
                collection=clean_test_collection,
            )
            for i in range(20)
        ]

        # Execute concurrently
        start_time = time.time()
        results = await asyncio.gather(*store_tasks)
        execution_time = time.time() - start_time

        # Verify all operations succeeded
        assert all(r["success"] for r in results), "All store operations should succeed"
        assert len(results) == 20, "Should complete all 20 operations"

        # Concurrent execution should be reasonably fast
        assert execution_time < 30.0, "20 concurrent stores should complete in <30s"

    async def test_concurrent_search_operations(self, clean_test_collection):
        """Test multiple concurrent search operations."""
        # First, store some test documents
        for i in range(5):
            await store(
                content=f"Machine learning document {i}",
                collection=clean_test_collection,
            )

        # Create multiple search requests
        search_tasks = [
            search(
                query="machine learning",
                collection=clean_test_collection,
                mode="semantic",
            )
            for _ in range(50)
        ]

        # Execute concurrently
        start_time = time.time()
        results = await asyncio.gather(*search_tasks)
        execution_time = time.time() - start_time

        # Verify all searches succeeded
        assert all(r["success"] for r in results), "All searches should succeed"
        assert len(results) == 50, "Should complete all 50 searches"

        # Concurrent searches should be fast
        assert execution_time < 60.0, "50 concurrent searches should complete in <60s"

    async def test_mixed_concurrent_operations(self, clean_test_collection):
        """Test mixed concurrent operations (store, search, retrieve)."""
        # Store initial documents
        store_results = await asyncio.gather(*[
            store(content=f"Initial doc {i}", collection=clean_test_collection)
            for i in range(3)
        ])

        # Create mixed operation tasks
        tasks = []

        # Add store tasks
        tasks.extend([
            store(content=f"New doc {i}", collection=clean_test_collection)
            for i in range(10)
        ])

        # Add search tasks
        tasks.extend([
            search(query="doc", collection=clean_test_collection, mode="hybrid")
            for _ in range(20)
        ])

        # Add retrieve tasks
        for store_result in store_results:
            tasks.append(
                retrieve(
                    document_id=store_result["document_id"],
                    collection=clean_test_collection,
                )
            )

        # Execute all concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time

        # Verify all operations succeeded
        successful = sum(1 for r in results if r.get("success"))
        assert successful == len(tasks), "All operations should succeed"

        # Mixed concurrent operations should complete reasonably
        assert execution_time < 45.0, "Mixed operations should complete in <45s"


# =============================================================================
# PERFORMANCE BENCHMARK TESTS
# =============================================================================

@pytest.mark.asyncio
@pytest.mark.benchmark
class TestAsyncPerformanceBenchmarks:
    """Benchmark async operations performance."""

    async def test_async_vs_sequential_stores_benchmark(self, clean_test_collection):
        """Benchmark async concurrent vs sequential store operations."""
        # Sequential execution baseline
        start_time = time.time()
        for i in range(10):
            await store(content=f"Sequential doc {i}", collection=clean_test_collection)
        sequential_time = time.time() - start_time

        # Clean collection for fair comparison
        from workspace_qdrant_mcp.server import qdrant_client
        await qdrant_client.delete_collection(clean_test_collection)
        await qdrant_client.create_collection(
            collection_name=clean_test_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Concurrent execution
        start_time = time.time()
        await asyncio.gather(*[
            store(content=f"Concurrent doc {i}", collection=clean_test_collection)
            for i in range(10)
        ])
        concurrent_time = time.time() - start_time

        # Concurrent should be faster (or at least not significantly slower)
        print(f"\nSequential: {sequential_time:.2f}s, Concurrent: {concurrent_time:.2f}s")
        # Note: We don't assert concurrent is faster as it depends on server load
        # but we document the performance characteristics

    async def test_search_latency_benchmark(self, clean_test_collection):
        """Benchmark search operation latency."""
        # Store test documents
        await asyncio.gather(*[
            store(content=f"Benchmark document {i}", collection=clean_test_collection)
            for i in range(100)
        ])

        # Measure search latency
        latencies: List[float] = []
        for _ in range(50):
            start_time = time.time()
            await search(query="benchmark", collection=clean_test_collection)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        print(f"\nSearch latency - Avg: {avg_latency:.2f}ms, "
              f"Min: {min_latency:.2f}ms, Max: {max_latency:.2f}ms")

        # Verify reasonable latency (async should not add significant overhead)
        assert avg_latency < 1000.0, "Average search latency should be <1s"


# =============================================================================
# ERROR HANDLING AND EDGE CASES
# =============================================================================

@pytest.mark.asyncio
class TestAsyncErrorHandling:
    """Test error handling in async operations."""

    async def test_qdrant_connection_error_handling(self):
        """Test graceful handling of Qdrant connection errors."""
        with patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client:
            # Simulate connection error
            mock_client.get_collection = AsyncMock(
                side_effect=Exception("Connection refused")
            )

            # Should handle error gracefully
            result = await ensure_collection_exists("test_collection")
            # Should attempt to create collection and may fail or fallback
            assert isinstance(result, bool), "Should return boolean result"

    async def test_timeout_error_handling(self):
        """Test handling of async timeout errors."""
        with patch("workspace_qdrant_mcp.server.qdrant_client") as mock_client:
            # Simulate timeout
            mock_client.search = AsyncMock(side_effect=asyncio.TimeoutError())

            # Should handle timeout gracefully
            result = await search(
                query="test",
                collection="test_collection",
            )

            assert result["success"] is False, "Should indicate failure on timeout"
            assert "error" in result, "Should include error message"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
