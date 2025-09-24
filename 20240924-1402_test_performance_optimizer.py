"""
Comprehensive unit tests for the high-throughput performance optimizer.

Tests cover all optimization components including edge cases, error conditions,
and performance benchmarks to ensure reliable operation under load.
"""

import asyncio
import pytest
import time
import threading
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

# Import the performance optimizer
from performance_optimizer_20240924 import (
    PerformanceMetrics,
    OptimizationConfig,
    ConnectionPool,
    BatchProcessor,
    QueryCache,
    PerformanceOptimizer
)


class TestPerformanceMetrics:
    """Test performance metrics tracking functionality."""

    def test_metrics_initialization(self):
        """Test proper initialization of performance metrics."""
        metrics = PerformanceMetrics("test_operation")

        assert metrics.operation_name == "test_operation"
        assert metrics.end_time is None
        assert metrics.duration is None
        assert metrics.throughput is None
        assert metrics.items_processed == 0
        assert metrics.error_count == 0
        assert metrics.start_time > 0

    def test_metrics_finish_calculation(self):
        """Test metrics calculation when operation finishes."""
        metrics = PerformanceMetrics("test_operation")
        metrics.items_processed = 100

        # Simulate some processing time
        time.sleep(0.01)
        metrics.finish()

        assert metrics.end_time is not None
        assert metrics.duration is not None
        assert metrics.duration > 0
        assert metrics.throughput is not None
        assert metrics.throughput > 0
        assert metrics.memory_usage_mb >= 0
        assert metrics.cpu_usage_percent >= 0

    def test_metrics_zero_items(self):
        """Test metrics with zero items processed."""
        metrics = PerformanceMetrics("test_operation")
        metrics.items_processed = 0

        metrics.finish()

        assert metrics.throughput is None  # Can't calculate throughput with 0 items


class TestOptimizationConfig:
    """Test optimization configuration with various settings."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.batch_size == 100
        assert config.max_batch_size == 1000
        assert config.parallel_workers is not None
        assert config.parallel_workers > 0
        assert config.max_connections == 10
        assert config.enable_query_cache is True
        assert config.cache_ttl_seconds == 300

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            batch_size=50,
            max_connections=5,
            enable_query_cache=False
        )

        assert config.batch_size == 50
        assert config.max_connections == 5
        assert config.enable_query_cache is False

    def test_post_init_defaults(self):
        """Test that post-init sets appropriate defaults."""
        config = OptimizationConfig()

        # These should be set based on system capabilities
        assert config.parallel_workers is not None
        assert config.compute_thread_pool_size is not None
        assert config.parallel_workers <= 8  # Capped at 8


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock()
    client.close.return_value = None
    return client


@pytest.fixture
def optimization_config():
    """Test optimization configuration."""
    return OptimizationConfig(
        batch_size=10,
        max_connections=3,
        connection_timeout=5.0,
        cache_ttl_seconds=60,
        max_cache_size=100
    )


@pytest.fixture
def qdrant_config():
    """Mock Qdrant configuration."""
    return {
        "url": "http://localhost:6333",
        "timeout": 30
    }


class TestConnectionPool:
    """Test connection pool functionality."""

    @pytest.mark.asyncio
    async def test_pool_initialization(self, optimization_config, qdrant_config):
        """Test connection pool initialization."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            pool = ConnectionPool(optimization_config, qdrant_config)
            await pool.initialize()

            stats = pool.get_pool_stats()
            assert stats["total_connections"] >= 1
            assert stats["available_connections"] >= 1
            assert stats["total_requests"] == 0

            await pool.close()

    @pytest.mark.asyncio
    async def test_connection_acquisition(self, optimization_config, qdrant_config):
        """Test connection acquisition and return."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            pool = ConnectionPool(optimization_config, qdrant_config)
            await pool.initialize()

            # Test connection context manager
            async with pool.get_connection() as conn:
                assert conn is not None
                assert conn == mock_client

            # Check stats after use
            stats = pool.get_pool_stats()
            assert stats["total_requests"] == 1
            assert stats["pool_hit_rate"] > 0

            await pool.close()

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, optimization_config, qdrant_config):
        """Test behavior when connection pool is exhausted."""
        # Set very small pool size
        optimization_config.max_connections = 1
        optimization_config.connection_timeout = 0.1

        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            pool = ConnectionPool(optimization_config, qdrant_config)
            await pool.initialize()

            # Hold one connection
            async with pool.get_connection():
                # Try to get another connection (should create new one)
                async with pool.get_connection() as conn2:
                    assert conn2 is not None

            await pool.close()

    @pytest.mark.asyncio
    async def test_connection_recycling(self, optimization_config, qdrant_config):
        """Test connection recycling based on age."""
        # Set very short recycle time for testing
        optimization_config.pool_recycle_time = 0.01

        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client.close.return_value = None
            mock_client_class.return_value = mock_client

            pool = ConnectionPool(optimization_config, qdrant_config)
            await pool.initialize()

            # Wait for connections to age
            await asyncio.sleep(0.02)

            # Get connection (should trigger recycling)
            async with pool.get_connection() as conn:
                assert conn is not None

            await pool.close()

    @pytest.mark.asyncio
    async def test_connection_creation_failure(self, optimization_config, qdrant_config):
        """Test handling of connection creation failures."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client_class.side_effect = Exception("Connection failed")

            pool = ConnectionPool(optimization_config, qdrant_config)

            with pytest.raises(Exception, match="Connection failed"):
                await pool.initialize()


class TestBatchProcessor:
    """Test batch processing functionality."""

    def test_batch_processor_initialization(self, optimization_config):
        """Test batch processor initialization."""
        processor = BatchProcessor(optimization_config)

        stats = processor.get_processing_stats()
        assert stats["total_processed"] == 0
        assert stats["total_batches"] == 0

    def test_optimal_batch_size_calculation(self, optimization_config):
        """Test optimal batch size calculation."""
        processor = BatchProcessor(optimization_config)

        # Test with small documents
        small_docs = [{"content": "small"} for _ in range(100)]
        batch_size = processor._calculate_optimal_batch_size(small_docs)
        assert batch_size > 0
        assert batch_size <= optimization_config.max_batch_size

        # Test with empty list
        empty_batch_size = processor._calculate_optimal_batch_size([])
        assert empty_batch_size == optimization_config.batch_size

    @pytest.mark.asyncio
    async def test_batch_document_processing(self, optimization_config):
        """Test batch document processing."""
        processor = BatchProcessor(optimization_config)

        # Mock processor function
        async def mock_processor(batch):
            await asyncio.sleep(0.01)  # Simulate processing time
            return [{"processed": True, **doc} for doc in batch]

        # Test documents
        documents = [{"id": i, "content": f"doc_{i}"} for i in range(25)]

        # Process documents
        results = await processor.process_documents_batch(
            documents, mock_processor
        )

        assert len(results) == len(documents)
        assert all(doc["processed"] for doc in results)

        # Check stats
        stats = processor.get_processing_stats()
        assert stats["total_processed"] == len(documents)
        assert stats["total_batches"] > 0

        await processor.close()

    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, optimization_config):
        """Test batch processing with some batches failing."""
        processor = BatchProcessor(optimization_config)

        # Mock processor that fails on certain batches
        async def failing_processor(batch):
            if batch[0]["id"] % 10 == 0:  # Fail on every 10th batch
                raise ValueError("Simulated batch failure")
            return [{"processed": True, **doc} for doc in batch]

        documents = [{"id": i, "content": f"doc_{i}"} for i in range(25)]

        results = await processor.process_documents_batch(
            documents, failing_processor
        )

        # Should still return results, but some may have errors
        assert len(results) > 0

        await processor.close()

    @pytest.mark.asyncio
    async def test_batch_processing_progress_callback(self, optimization_config):
        """Test batch processing with progress callback."""
        processor = BatchProcessor(optimization_config)

        progress_updates = []

        async def progress_callback(progress, current, total):
            progress_updates.append((progress, current, total))

        async def mock_processor(batch):
            await asyncio.sleep(0.01)
            return [{"processed": True, **doc} for doc in batch]

        documents = [{"id": i, "content": f"doc_{i}"} for i in range(20)]

        await processor.process_documents_batch(
            documents, mock_processor, progress_callback
        )

        # Should have received progress updates
        assert len(progress_updates) > 0
        assert all(0 <= progress <= 1 for progress, _, _ in progress_updates)

        await processor.close()


class TestQueryCache:
    """Test query cache functionality."""

    @pytest.mark.asyncio
    async def test_cache_basic_operations(self, optimization_config):
        """Test basic cache get/set operations."""
        cache = QueryCache(optimization_config)

        # Test cache miss
        result = await cache.get("test_key")
        assert result is None

        # Test cache set and hit
        test_value = {"data": "test_value"}
        await cache.set("test_key", test_value)

        result = await cache.get("test_key")
        assert result == test_value

        # Check stats
        stats = cache.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, optimization_config):
        """Test cache TTL expiration."""
        # Set very short TTL for testing
        optimization_config.cache_ttl_seconds = 0.01
        cache = QueryCache(optimization_config)

        # Set a value
        await cache.set("expire_key", "expire_value")

        # Should be available immediately
        result = await cache.get("expire_key")
        assert result == "expire_value"

        # Wait for expiration
        await asyncio.sleep(0.02)

        # Should be expired now
        result = await cache.get("expire_key")
        assert result is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, optimization_config):
        """Test LRU eviction when cache is full."""
        # Set small cache size
        optimization_config.max_cache_size = 2
        cache = QueryCache(optimization_config)

        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Access key1 to make it more recently used
        await cache.get("key1")

        # Add another key (should evict key2)
        await cache.set("key3", "value3")

        # key1 and key3 should be available, key2 should be evicted
        assert await cache.get("key1") == "value1"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key2") is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_disabled(self, optimization_config):
        """Test cache behavior when disabled."""
        optimization_config.enable_query_cache = False
        cache = QueryCache(optimization_config)

        # Set should do nothing
        await cache.set("disabled_key", "disabled_value")

        # Get should return None
        result = await cache.get("disabled_key")
        assert result is None

        await cache.close()

    @pytest.mark.asyncio
    async def test_cache_clear(self, optimization_config):
        """Test cache clear operation."""
        cache = QueryCache(optimization_config)

        # Add some entries
        await cache.set("clear_key1", "value1")
        await cache.set("clear_key2", "value2")

        # Verify entries exist
        assert await cache.get("clear_key1") == "value1"
        assert await cache.get("clear_key2") == "value2"

        # Clear cache
        await cache.clear()

        # Entries should be gone
        assert await cache.get("clear_key1") is None
        assert await cache.get("clear_key2") is None

        await cache.close()


class TestPerformanceOptimizer:
    """Test main performance optimizer integration."""

    @pytest.mark.asyncio
    async def test_optimizer_initialization(self, optimization_config, qdrant_config):
        """Test performance optimizer initialization."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)
            await optimizer.initialize()

            stats = optimizer.get_comprehensive_stats()

            assert "connection_pool" in stats
            assert "batch_processor" in stats
            assert "query_cache" in stats
            assert "config" in stats

            await optimizer.close()

    @pytest.mark.asyncio
    async def test_optimized_connection_context(self, optimization_config, qdrant_config):
        """Test optimized connection context manager."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)

            async with optimizer.get_optimized_connection() as conn:
                assert conn is not None
                assert conn == mock_client

            await optimizer.close()

    @pytest.mark.asyncio
    async def test_cached_query_execution(self, optimization_config, qdrant_config):
        """Test cached query execution."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)
            await optimizer.initialize()

            # Mock query function
            query_call_count = 0
            async def mock_query(param):
                nonlocal query_call_count
                query_call_count += 1
                return f"result_for_{param}"

            # First call should execute query
            result1 = await optimizer.cached_query("test_cache_key", mock_query, "param1")
            assert result1 == "result_for_param1"
            assert query_call_count == 1

            # Second call should use cache
            result2 = await optimizer.cached_query("test_cache_key", mock_query, "param1")
            assert result2 == "result_for_param1"
            assert query_call_count == 1  # Should not increment

            await optimizer.close()

    @pytest.mark.asyncio
    async def test_optimized_document_processing(self, optimization_config, qdrant_config):
        """Test optimized document processing."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)

            # Mock processor function
            async def mock_processor(batch):
                await asyncio.sleep(0.001)  # Simulate processing
                return [{"processed": True, **doc} for doc in batch]

            documents = [{"id": i, "content": f"doc_{i}"} for i in range(30)]

            results = await optimizer.process_documents_optimized(
                documents, mock_processor
            )

            assert len(results) == len(documents)
            assert all(doc["processed"] for doc in results)

            await optimizer.close()

    @pytest.mark.asyncio
    async def test_performance_metrics_recording(self, optimization_config, qdrant_config):
        """Test performance metrics recording."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)

            # Create and record metrics
            metrics = PerformanceMetrics("test_operation")
            metrics.items_processed = 100
            metrics.finish()

            optimizer.record_performance_metrics(metrics)

            stats = optimizer.get_comprehensive_stats()
            assert stats["recent_metrics"] == 1

            await optimizer.close()

    @pytest.mark.asyncio
    async def test_comprehensive_stats(self, optimization_config, qdrant_config):
        """Test comprehensive statistics gathering."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)
            await optimizer.initialize()

            stats = optimizer.get_comprehensive_stats()

            # Verify all expected sections are present
            expected_sections = [
                "connection_pool", "batch_processor", "query_cache",
                "recent_metrics", "config"
            ]

            for section in expected_sections:
                assert section in stats, f"Missing section: {section}"

            # Verify config section has expected values
            config_stats = stats["config"]
            assert config_stats["batch_size"] == optimization_config.batch_size
            assert config_stats["max_connections"] == optimization_config.max_connections
            assert config_stats["cache_enabled"] == optimization_config.enable_query_cache

            await optimizer.close()


class TestIntegrationScenarios:
    """Test complex integration scenarios and edge cases."""

    @pytest.mark.asyncio
    async def test_concurrent_connection_usage(self, optimization_config, qdrant_config):
        """Test concurrent connection usage under load."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)
            await optimizer.initialize()

            # Create multiple concurrent tasks using connections
            async def use_connection(task_id):
                async with optimizer.get_optimized_connection() as conn:
                    await asyncio.sleep(0.01)  # Simulate work
                    return f"task_{task_id}_completed"

            # Run multiple tasks concurrently
            tasks = [use_connection(i) for i in range(20)]
            results = await asyncio.gather(*tasks)

            assert len(results) == 20
            assert all("completed" in result for result in results)

            await optimizer.close()

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, optimization_config, qdrant_config):
        """Test behavior under memory pressure."""
        # Set low memory limits
        optimization_config.max_memory_mb = 100
        optimization_config.stream_threshold_mb = 10

        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            mock_client = MagicMock()
            mock_client.get_collections.return_value = MagicMock()
            mock_client_class.return_value = mock_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)

            # Create large documents to test memory handling
            large_documents = [
                {"id": i, "content": "x" * 1000} for i in range(100)
            ]

            async def mock_processor(batch):
                return [{"processed": True, **doc} for doc in batch]

            # Should handle large document set gracefully
            results = await optimizer.process_documents_optimized(
                large_documents, mock_processor
            )

            assert len(results) == len(large_documents)

            await optimizer.close()

    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, optimization_config, qdrant_config):
        """Test error recovery and system resilience."""
        with patch('performance_optimizer_20240924.QdrantClient') as mock_client_class:
            # Create a client that sometimes fails
            failure_count = 0
            def create_failing_client():
                nonlocal failure_count
                if failure_count < 2:
                    failure_count += 1
                    raise Exception("Temporary connection failure")

                mock_client = MagicMock()
                mock_client.get_collections.return_value = MagicMock()
                return mock_client

            mock_client_class.side_effect = create_failing_client

            optimizer = PerformanceOptimizer(optimization_config, qdrant_config)

            # Should eventually initialize successfully after failures
            await optimizer.initialize()

            # Should be able to use connections normally
            async with optimizer.get_optimized_connection() as conn:
                assert conn is not None

            await optimizer.close()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])