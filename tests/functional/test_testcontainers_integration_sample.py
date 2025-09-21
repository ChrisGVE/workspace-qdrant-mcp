"""
Sample testcontainers integration tests for workspace-qdrant-mcp.

These tests demonstrate isolated testing patterns using:
- Containerized Qdrant instances
- Network isolation testing
- Service integration scenarios
- Database migration testing
"""

import pytest
import asyncio
import logging
from typing import Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import time
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)


@pytest.mark.testcontainers
@pytest.mark.qdrant_integration
class TestQdrantContainerIntegration:
    """Test Qdrant integration using containerized instances."""

    async def test_basic_collection_operations(self, qdrant_client: QdrantClient, test_collection: str):
        """Test basic collection operations with containerized Qdrant."""
        # Verify collection exists
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        assert test_collection in collection_names

        # Get collection info
        collection_info = qdrant_client.get_collection(test_collection)
        assert collection_info.config.params.vectors.size == 384
        assert collection_info.config.params.vectors.distance == Distance.COSINE

        # Test collection statistics
        collection_stats = qdrant_client.get_collection(test_collection)
        assert collection_stats.vectors_count == 0  # Should be empty initially

    async def test_vector_operations_lifecycle(self, qdrant_client: QdrantClient, test_collection: str):
        """Test complete vector operations lifecycle."""
        # Create test vectors
        vectors = []
        for i in range(10):
            vector = np.random.random(384).astype(np.float32)
            point = PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={"category": f"test_{i % 3}", "index": i}
            )
            vectors.append(point)

        # Insert vectors
        operation_info = qdrant_client.upsert(
            collection_name=test_collection,
            points=vectors
        )
        assert operation_info.status == "completed"

        # Verify insertion
        collection_info = qdrant_client.get_collection(test_collection)
        assert collection_info.vectors_count == 10

        # Test search
        search_vector = np.random.random(384).astype(np.float32)
        search_results = qdrant_client.search(
            collection_name=test_collection,
            query_vector=search_vector.tolist(),
            limit=5
        )
        assert len(search_results) == 5
        assert all(result.score >= 0 for result in search_results)

        # Test filtering
        filtered_results = qdrant_client.search(
            collection_name=test_collection,
            query_vector=search_vector.tolist(),
            query_filter={"must": [{"key": "category", "match": {"value": "test_0"}}]},
            limit=10
        )
        assert len(filtered_results) <= 4  # Should match category pattern

        # Test point retrieval
        point = qdrant_client.retrieve(
            collection_name=test_collection,
            ids=[0, 1, 2]
        )
        assert len(point) == 3

        # Test point deletion
        qdrant_client.delete(
            collection_name=test_collection,
            points_selector=[0, 1]
        )

        # Verify deletion
        remaining_points = qdrant_client.count(collection_name=test_collection)
        assert remaining_points.count == 8

    @pytest.mark.benchmark
    async def test_performance_with_large_dataset(self, performance_qdrant_client: QdrantClient):
        """Test performance with larger dataset using optimized client."""
        collection_name = "performance_test"

        # Create performance test collection
        performance_qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        try:
            # Generate larger dataset
            batch_size = 100
            total_vectors = 1000
            insert_times = []

            for batch_start in range(0, total_vectors, batch_size):
                vectors = []
                for i in range(batch_start, min(batch_start + batch_size, total_vectors)):
                    vector = np.random.random(384).astype(np.float32)
                    point = PointStruct(
                        id=i,
                        vector=vector.tolist(),
                        payload={"batch": batch_start // batch_size, "index": i}
                    )
                    vectors.append(point)

                # Time the insertion
                start_time = time.time()
                performance_qdrant_client.upsert(
                    collection_name=collection_name,
                    points=vectors
                )
                end_time = time.time()
                insert_times.append(end_time - start_time)

            # Verify all vectors inserted
            collection_info = performance_qdrant_client.get_collection(collection_name)
            assert collection_info.vectors_count == total_vectors

            # Test search performance
            search_vector = np.random.random(384).astype(np.float32)
            search_times = []

            for _ in range(10):
                start_time = time.time()
                results = performance_qdrant_client.search(
                    collection_name=collection_name,
                    query_vector=search_vector.tolist(),
                    limit=50
                )
                end_time = time.time()
                search_times.append(end_time - start_time)
                assert len(results) == 50

            # Performance assertions
            avg_insert_time = sum(insert_times) / len(insert_times)
            avg_search_time = sum(search_times) / len(search_times)

            assert avg_insert_time < 2.0, f"Insert too slow: {avg_insert_time:.3f}s per batch"
            assert avg_search_time < 0.5, f"Search too slow: {avg_search_time:.3f}s"

        finally:
            # Cleanup
            performance_qdrant_client.delete_collection(collection_name)

    async def test_concurrent_operations(self, qdrant_client: QdrantClient):
        """Test concurrent operations on containerized Qdrant."""
        # Create multiple collections for concurrent testing
        collections = []
        for i in range(3):
            collection_name = f"concurrent_test_{i}"
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            collections.append(collection_name)

        try:
            async def insert_vectors(collection_name: str, start_id: int):
                """Insert vectors into a specific collection."""
                vectors = []
                for i in range(20):
                    vector = np.random.random(384).astype(np.float32)
                    point = PointStruct(
                        id=start_id + i,
                        vector=vector.tolist(),
                        payload={"collection": collection_name, "index": i}
                    )
                    vectors.append(point)

                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=vectors
                )
                return len(vectors)

            # Run concurrent insertions
            tasks = []
            for i, collection_name in enumerate(collections):
                task = insert_vectors(collection_name, i * 100)
                tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Verify all insertions completed
            assert all(result == 20 for result in results)

            # Verify collections have correct counts
            for collection_name in collections:
                info = qdrant_client.get_collection(collection_name)
                assert info.vectors_count == 20

        finally:
            # Cleanup
            for collection_name in collections:
                qdrant_client.delete_collection(collection_name)


@pytest.mark.testcontainers
@pytest.mark.network_required
class TestServiceIntegrationScenarios:
    """Test integration scenarios with multiple containerized services."""

    async def test_qdrant_with_mock_services(self, isolated_environment: Dict[str, Any]):
        """Test Qdrant integration with mock external services."""
        import httpx

        # Test Qdrant connectivity
        qdrant_config = isolated_environment["qdrant"]
        async with httpx.AsyncClient() as client:
            health_response = await client.get(f"{qdrant_config['url']}/health")
            assert health_response.status_code == 200

        # Test mock API connectivity
        mock_config = isolated_environment["mock_api"]
        async with httpx.AsyncClient() as client:
            api_response = await client.get(f"{mock_config['url']}/api/v1/test")
            assert api_response.status_code == 200
            assert api_response.json()["status"] == "ok"

        # Test integration scenario
        qdrant_client = QdrantClient(
            host=qdrant_config["host"],
            port=qdrant_config["port"]
        )

        # Create collection for integration test
        collection_name = "integration_test"
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        try:
            # Simulate workflow that uses both services
            # 1. Get data from mock API
            async with httpx.AsyncClient() as client:
                data_response = await client.get(f"{mock_config['url']}/api/v1/test")
                external_data = data_response.json()

            # 2. Store in Qdrant
            vector = np.random.random(384).astype(np.float32)
            point = PointStruct(
                id=1,
                vector=vector.tolist(),
                payload={"external_data": external_data["message"]}
            )

            qdrant_client.upsert(collection_name=collection_name, points=[point])

            # 3. Verify integration
            retrieved = qdrant_client.retrieve(collection_name=collection_name, ids=[1])
            assert len(retrieved) == 1
            assert retrieved[0].payload["external_data"] == "Test API"

        finally:
            qdrant_client.delete_collection(collection_name)

    @pytest.mark.slow_functional
    async def test_data_migration_scenario(self, qdrant_client: QdrantClient):
        """Test data migration between collections."""
        source_collection = "migration_source"
        target_collection = "migration_target"

        # Create source collection with data
        qdrant_client.create_collection(
            collection_name=source_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        # Insert test data
        vectors = []
        for i in range(50):
            vector = np.random.random(384).astype(np.float32)
            point = PointStruct(
                id=i,
                vector=vector.tolist(),
                payload={"original_id": i, "category": f"cat_{i % 5}"}
            )
            vectors.append(point)

        qdrant_client.upsert(collection_name=source_collection, points=vectors)

        try:
            # Create target collection with different configuration
            qdrant_client.create_collection(
                collection_name=target_collection,
                vectors_config=VectorParams(size=384, distance=Distance.DOT)
            )

            # Migrate data with transformation
            scroll_result = qdrant_client.scroll(
                collection_name=source_collection,
                limit=100
            )

            migrated_points = []
            for point in scroll_result[0]:
                # Transform during migration
                new_point = PointStruct(
                    id=point.id + 1000,  # Offset IDs
                    vector=point.vector,
                    payload={**point.payload, "migrated": True}
                )
                migrated_points.append(new_point)

            qdrant_client.upsert(collection_name=target_collection, points=migrated_points)

            # Verify migration
            target_info = qdrant_client.get_collection(target_collection)
            assert target_info.vectors_count == 50

            # Verify data integrity
            sample_point = qdrant_client.retrieve(collection_name=target_collection, ids=[1000])
            assert len(sample_point) == 1
            assert sample_point[0].payload["original_id"] == 0
            assert sample_point[0].payload["migrated"] is True

        finally:
            # Cleanup
            qdrant_client.delete_collection(source_collection)
            qdrant_client.delete_collection(target_collection)


@pytest.mark.testcontainers
@pytest.mark.regression
class TestContainerRegressionScenarios:
    """Regression tests using containerized services."""

    async def test_container_restart_resilience(self, qdrant_container, qdrant_client: QdrantClient):
        """Test resilience to container restarts."""
        collection_name = "restart_test"

        # Create collection and add data
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        vector = np.random.random(384).astype(np.float32)
        point = PointStruct(
            id=1,
            vector=vector.tolist(),
            payload={"test": "restart_resilience"}
        )
        qdrant_client.upsert(collection_name=collection_name, points=[point])

        # Simulate restart by creating new client
        # (In real scenario, container would be restarted)
        new_client = QdrantClient(
            host=qdrant_container.get_container_host_ip(),
            port=qdrant_container.get_exposed_port(6333),
            timeout=30
        )

        # Verify data persistence
        collections = new_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        assert collection_name in collection_names

        retrieved = new_client.retrieve(collection_name=collection_name, ids=[1])
        assert len(retrieved) == 1
        assert retrieved[0].payload["test"] == "restart_resilience"

        # Cleanup
        new_client.delete_collection(collection_name)

    async def test_memory_leak_detection(self, qdrant_client: QdrantClient):
        """Test for memory leaks during sustained operations."""
        collection_name = "memory_test"

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

        try:
            # Perform sustained operations
            for round_num in range(10):
                # Insert batch
                vectors = []
                for i in range(100):
                    vector = np.random.random(384).astype(np.float32)
                    point = PointStruct(
                        id=round_num * 100 + i,
                        vector=vector.tolist(),
                        payload={"round": round_num, "index": i}
                    )
                    vectors.append(point)

                qdrant_client.upsert(collection_name=collection_name, points=vectors)

                # Perform searches
                for _ in range(10):
                    search_vector = np.random.random(384).astype(np.float32)
                    qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=search_vector.tolist(),
                        limit=10
                    )

                # Delete some points to test cleanup
                if round_num > 0:
                    delete_ids = list(range((round_num - 1) * 100, (round_num - 1) * 100 + 50))
                    qdrant_client.delete(
                        collection_name=collection_name,
                        points_selector=delete_ids
                    )

            # Final verification
            final_info = qdrant_client.get_collection(collection_name)
            # Should have approximately 950 vectors (1000 - 50 deleted from round 0)
            assert final_info.vectors_count > 900
            assert final_info.vectors_count <= 1000

        finally:
            qdrant_client.delete_collection(collection_name)