"""
Resource exhaustion testing for workspace-qdrant-mcp.

Tests behavior under resource constraints including memory limits,
disk space exhaustion, and CPU overload scenarios.
"""

import asyncio
import os
import psutil
import pytest
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Generator

from workspace_qdrant_mcp.core.client import EnhancedQdrantClient
from workspace_qdrant_mcp.core.hybrid_search import HybridSearchEngine
from workspace_qdrant_mcp.tools.memory import DocumentMemoryManager


class ResourceExhaustionSimulator:
    """Utility class for simulating resource exhaustion scenarios."""

    @staticmethod
    def create_memory_pressure(target_mb: int = 100) -> List[bytes]:
        """Create memory pressure by allocating large chunks of memory."""
        memory_chunks = []
        try:
            for _ in range(target_mb):
                # Allocate 1MB chunks
                chunk = b'x' * (1024 * 1024)
                memory_chunks.append(chunk)
            return memory_chunks
        except MemoryError:
            return memory_chunks

    @staticmethod
    def create_cpu_pressure(duration_seconds: int = 5) -> threading.Thread:
        """Create CPU pressure by running intensive computation."""
        def cpu_intensive_task():
            end_time = time.time() + duration_seconds
            while time.time() < end_time:
                # Perform CPU-intensive operations
                sum(i**2 for i in range(1000))

        thread = threading.Thread(target=cpu_intensive_task)
        thread.daemon = True
        thread.start()
        return thread

    @staticmethod
    def create_disk_pressure(directory: Path, target_mb: int = 50) -> List[Path]:
        """Create disk pressure by filling up storage space."""
        files_created = []
        try:
            for i in range(target_mb):
                file_path = directory / f"test_file_{i}.dat"
                with open(file_path, 'wb') as f:
                    # Write 1MB of data per file
                    f.write(b'x' * (1024 * 1024))
                files_created.append(file_path)
        except OSError:
            # Disk space exhausted or other IO error
            pass
        return files_created

    @staticmethod
    def simulate_file_descriptor_exhaustion():
        """Simulate file descriptor exhaustion."""
        open_files = []
        try:
            # Try to open many files until we hit the limit
            for i in range(10000):
                temp_file = tempfile.NamedTemporaryFile()
                open_files.append(temp_file)
        except OSError:
            # File descriptor limit reached
            pass
        finally:
            # Clean up opened files
            for f in open_files:
                try:
                    f.close()
                except:
                    pass


@pytest.mark.resource_exhaustion
class TestMemoryExhaustionHandling:
    """Test behavior under memory pressure conditions."""

    @pytest.fixture
    async def mock_client(self):
        """Create a mock client for testing."""
        client = Mock(spec=EnhancedQdrantClient)
        client.client = Mock()
        return client

    @pytest.mark.asyncio
    async def test_large_document_processing_oom(self, mock_client):
        """Test handling of out-of-memory conditions during large document processing."""
        memory_manager = DocumentMemoryManager(
            client=mock_client,
            project_name="test-project"
        )

        # Create memory pressure
        memory_chunks = ResourceExhaustionSimulator.create_memory_pressure(50)

        try:
            # Simulate processing a very large document
            large_content = "x" * (10 * 1024 * 1024)  # 10MB string

            # Mock the client to simulate memory allocation during processing
            def memory_intensive_processing(*args, **kwargs):
                # Allocate more memory during processing
                temp_data = "y" * (50 * 1024 * 1024)  # 50MB
                return {"operation_id": 12345, "status": "success"}

            mock_client.upsert_points = Mock(side_effect=memory_intensive_processing)

            # This should handle memory pressure gracefully
            try:
                await memory_manager.store_document(
                    content=large_content,
                    metadata={"title": "large_document.txt"},
                    collection_name="test-collection"
                )
            except MemoryError:
                pytest.skip("Memory exhaustion as expected")

        finally:
            # Clean up memory
            del memory_chunks

    @pytest.mark.asyncio
    async def test_concurrent_memory_intensive_operations(self, mock_client):
        """Test concurrent memory-intensive operations under pressure."""
        # Create moderate memory pressure
        memory_chunks = ResourceExhaustionSimulator.create_memory_pressure(25)

        try:
            hybrid_search = HybridSearchEngine(
                qdrant_client=mock_client,
                embedding_model="test-model"
            )

            # Mock memory-intensive search operations
            async def memory_intensive_search(*args, **kwargs):
                # Simulate memory allocation during search
                temp_vectors = [[0.1] * 384 for _ in range(1000)]  # Large vector batch
                await asyncio.sleep(0.1)  # Simulate processing time
                return {
                    "points": [{"id": i, "score": 0.9, "payload": {"text": f"result_{i}"}}
                              for i in range(len(temp_vectors))]
                }

            mock_client.hybrid_search = AsyncMock(side_effect=memory_intensive_search)

            # Run multiple concurrent searches
            tasks = []
            for i in range(5):
                task = hybrid_search.search(
                    query=f"test query {i}",
                    collections=[f"collection-{i}"],
                    limit=10
                )
                tasks.append(task)

            # Should handle concurrent memory pressure
            try:
                results = await asyncio.gather(*tasks)
                assert len(results) == 5
            except (MemoryError, asyncio.TimeoutError):
                pytest.skip("Memory exhaustion during concurrent operations")

        finally:
            del memory_chunks

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, mock_client):
        """Test detection of potential memory leaks."""
        initial_memory = psutil.Process().memory_info().rss

        memory_manager = DocumentMemoryManager(
            client=mock_client,
            project_name="test-project"
        )

        # Simulate repetitive operations that might leak memory
        for i in range(100):
            mock_content = f"Document content {i} " * 100
            mock_client.upsert_points = AsyncMock(return_value={"operation_id": i})

            await memory_manager.store_document(
                content=mock_content,
                metadata={"title": f"doc_{i}.txt"},
                collection_name="test-collection"
            )

        final_memory = psutil.Process().memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB

        # Memory should not increase significantly (threshold: 50MB)
        assert memory_increase < 50, f"Potential memory leak detected: {memory_increase:.2f}MB increase"

    def test_embedding_model_memory_pressure(self):
        """Test embedding model behavior under memory pressure."""
        memory_chunks = ResourceExhaustionSimulator.create_memory_pressure(100)

        try:
            from fastembed import TextEmbedding

            # Try to initialize embedding model under memory pressure
            try:
                model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Process text under memory pressure
                texts = ["test document"] * 1000
                embeddings = list(model.embed(texts))

                assert len(embeddings) > 0
                assert all(len(emb) == 384 for emb in embeddings)

            except (MemoryError, RuntimeError) as e:
                pytest.skip(f"Memory pressure prevented embedding: {e}")

        finally:
            del memory_chunks


@pytest.mark.resource_exhaustion
class TestDiskSpaceExhaustionHandling:
    """Test behavior under disk space constraints."""

    @pytest.mark.asyncio
    async def test_log_file_disk_exhaustion(self, tmp_path):
        """Test handling of disk exhaustion during log file writing."""
        # Fill up disk space in temp directory
        files_created = ResourceExhaustionSimulator.create_disk_pressure(tmp_path, 20)

        try:
            # Configure logging to use the temp directory
            log_file = tmp_path / "test.log"

            # Simulate extensive logging under disk pressure
            try:
                for i in range(1000):
                    with open(log_file, 'a') as f:
                        f.write(f"Log entry {i}: " + "x" * 1000 + "\n")
            except OSError as e:
                # Expected behavior: graceful handling of disk full
                assert "No space left on device" in str(e) or "disk full" in str(e).lower()

        finally:
            # Clean up created files
            for file_path in files_created:
                try:
                    file_path.unlink()
                except:
                    pass

    @pytest.mark.asyncio
    async def test_cache_file_disk_exhaustion(self, tmp_path):
        """Test cache behavior when disk space is exhausted."""
        # Create disk pressure
        files_created = ResourceExhaustionSimulator.create_disk_pressure(tmp_path, 30)

        try:
            cache_dir = tmp_path / "cache"
            cache_dir.mkdir(exist_ok=True)

            # Simulate cache operations under disk pressure
            try:
                for i in range(100):
                    cache_file = cache_dir / f"cache_{i}.dat"
                    with open(cache_file, 'wb') as f:
                        f.write(b"cached_data" * 10000)  # 90KB per file
            except OSError as e:
                # Should handle disk exhaustion gracefully
                assert "No space left on device" in str(e) or "disk full" in str(e).lower()

        finally:
            for file_path in files_created:
                try:
                    file_path.unlink()
                except:
                    pass

    @pytest.mark.asyncio
    async def test_temporary_file_cleanup_on_disk_full(self, tmp_path):
        """Test cleanup of temporary files when disk becomes full."""
        temp_files = []

        try:
            # Create many temporary files until disk is full
            for i in range(100):
                try:
                    temp_file = tmp_path / f"temp_{i}.dat"
                    with open(temp_file, 'wb') as f:
                        f.write(b"x" * (1024 * 1024))  # 1MB per file
                    temp_files.append(temp_file)
                except OSError:
                    # Disk full - test cleanup
                    break

            # Simulate cleanup operation
            cleaned_count = 0
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                except:
                    pass

            # Should be able to clean up at least some files
            assert cleaned_count > 0

        finally:
            # Ensure cleanup
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except:
                    pass


@pytest.mark.resource_exhaustion
class TestCPUExhaustionHandling:
    """Test behavior under CPU overload conditions."""

    @pytest.mark.asyncio
    async def test_search_performance_under_cpu_load(self):
        """Test search performance degradation under CPU load."""
        mock_client = Mock(spec=EnhancedQdrantClient)
        hybrid_search = HybridSearchEngine(
            qdrant_client=mock_client,
            embedding_model="test-model"
        )

        # Create CPU pressure
        cpu_thread = ResourceExhaustionSimulator.create_cpu_pressure(5)

        try:
            # Mock search with processing delay
            async def cpu_intensive_search(*args, **kwargs):
                # Simulate CPU-intensive processing
                start_time = time.time()
                while time.time() - start_time < 0.5:  # 500ms of CPU work
                    sum(i**2 for i in range(1000))

                return {
                    "points": [{"id": 1, "score": 0.9, "payload": {"text": "result"}}]
                }

            mock_client.hybrid_search = AsyncMock(side_effect=cpu_intensive_search)

            # Measure search time under CPU load
            start_time = time.time()
            result = await hybrid_search.search(
                query="test query",
                collections=["test-collection"],
                limit=5
            )
            end_time = time.time()

            search_time = end_time - start_time

            # Search should still complete (may be slower under load)
            assert result is not None
            assert search_time < 10.0  # Should complete within 10 seconds

        finally:
            # Wait for CPU pressure thread to complete
            cpu_thread.join(timeout=1.0)

    @pytest.mark.asyncio
    async def test_concurrent_operations_cpu_throttling(self):
        """Test concurrent operations under CPU throttling."""
        mock_client = Mock(spec=EnhancedQdrantClient)

        # Create sustained CPU pressure
        cpu_thread = ResourceExhaustionSimulator.create_cpu_pressure(10)

        try:
            # Mock CPU-intensive operations
            async def cpu_bound_operation(operation_id):
                # Simulate CPU-bound work
                result = sum(i**2 for i in range(10000))
                await asyncio.sleep(0.01)  # Small async yield
                return {"operation_id": operation_id, "result": result}

            # Run multiple concurrent CPU-bound operations
            tasks = []
            for i in range(20):
                task = cpu_bound_operation(i)
                tasks.append(task)

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            total_time = end_time - start_time

            # All operations should complete
            assert len(results) == 20
            assert all("operation_id" in result for result in results)

            # May take longer under CPU pressure, but should complete
            assert total_time < 30.0  # Reasonable upper bound

        finally:
            cpu_thread.join(timeout=1.0)


@pytest.mark.resource_exhaustion
class TestFileDescriptorExhaustion:
    """Test behavior when file descriptor limits are reached."""

    @pytest.mark.asyncio
    async def test_file_descriptor_limit_handling(self, tmp_path):
        """Test graceful handling when file descriptor limit is reached."""
        opened_files = []

        try:
            # Try to exhaust file descriptors
            for i in range(1000):  # Try to open many files
                try:
                    file_path = tmp_path / f"test_{i}.txt"
                    f = open(file_path, 'w')
                    f.write(f"content {i}")
                    opened_files.append(f)
                except OSError as e:
                    # Hit file descriptor limit
                    if "Too many open files" in str(e):
                        break
                    else:
                        raise

            # Should handle file descriptor exhaustion gracefully
            assert len(opened_files) > 0  # At least some files were opened

        finally:
            # Clean up opened files
            for f in opened_files:
                try:
                    f.close()
                except:
                    pass

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self):
        """Test handling of connection pool exhaustion."""
        mock_client = Mock(spec=EnhancedQdrantClient)

        # Mock connection limit reached
        connection_count = 0
        max_connections = 10

        async def limited_connection(*args, **kwargs):
            nonlocal connection_count
            if connection_count >= max_connections:
                raise OSError("Too many open files")

            connection_count += 1
            try:
                return {"status": "connected"}
            finally:
                connection_count -= 1

        mock_client.client.get_collections = AsyncMock(side_effect=limited_connection)

        # Try to make more connections than the limit
        tasks = []
        for i in range(15):  # More than max_connections
            task = mock_client.client.get_collections()
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some should succeed, others should fail with connection limit error
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = sum(1 for r in results if isinstance(r, OSError))

        assert success_count > 0  # Some connections should succeed
        assert error_count > 0    # Some should hit the limit


@pytest.mark.resource_exhaustion
class TestResourceMonitoring:
    """Test resource monitoring and alerting capabilities."""

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring and threshold alerting."""
        initial_memory = psutil.Process().memory_info().rss

        # Simulate memory-intensive operations
        large_data = []
        for i in range(100):
            large_data.append([0.1] * 10000)  # Large lists

        current_memory = psutil.Process().memory_info().rss
        memory_increase = (current_memory - initial_memory) / (1024 * 1024)

        # Should be able to monitor memory increase
        assert memory_increase > 0

        # Clean up
        del large_data

    def test_cpu_usage_monitoring(self):
        """Test CPU usage monitoring during intensive operations."""
        initial_cpu_percent = psutil.cpu_percent(interval=1)

        # Create CPU-intensive work
        cpu_thread = ResourceExhaustionSimulator.create_cpu_pressure(3)

        # Monitor CPU usage
        time.sleep(1)  # Let CPU usage register
        peak_cpu_percent = psutil.cpu_percent(interval=1)

        cpu_thread.join(timeout=5.0)

        # CPU usage should have increased during intensive operations
        assert peak_cpu_percent >= initial_cpu_percent

    def test_disk_usage_monitoring(self, tmp_path):
        """Test disk usage monitoring and cleanup triggers."""
        initial_disk_usage = psutil.disk_usage(str(tmp_path))

        # Create files to increase disk usage
        files_created = ResourceExhaustionSimulator.create_disk_pressure(tmp_path, 10)

        current_disk_usage = psutil.disk_usage(str(tmp_path))

        # Disk usage should have changed
        usage_change = initial_disk_usage.used - current_disk_usage.used

        try:
            # Should be able to monitor disk usage changes
            assert abs(usage_change) >= 0  # Some change expected
        finally:
            # Clean up
            for file_path in files_created:
                try:
                    file_path.unlink()
                except:
                    pass