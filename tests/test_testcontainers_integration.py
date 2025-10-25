"""
Integration tests for testcontainers Qdrant setup.

Validates that the testcontainers infrastructure provides clean, isolated
Qdrant instances for testing and prevents data contamination between tests.
"""

import asyncio

import pytest
from qdrant_client.http import models

from tests.utils.testcontainers_qdrant import (
    IsolatedQdrantContainer,
    isolated_qdrant_instance,
)


@pytest.mark.requires_docker
@pytest.mark.isolated_container
class TestIsolatedQdrantContainer:
    """Test isolated Qdrant container functionality."""

    def test_container_lifecycle(self):
        """Test container start/stop lifecycle."""
        container = IsolatedQdrantContainer()

        # Container should not be started initially
        assert not container._is_started

        # Start container
        container.start()
        assert container._is_started
        assert container.client is not None
        assert container.http_url.startswith("http://")
        assert container.grpc_url.startswith("http://")

        # Stop container
        container.stop()
        assert not container._is_started

    def test_container_context_manager(self):
        """Test container as context manager."""
        with IsolatedQdrantContainer() as container:
            assert container._is_started
            assert container.client is not None

            # Test basic operation
            collections = container.client.get_collections()
            assert isinstance(collections, models.CollectionsResponse)

    def test_container_reset(self):
        """Test container state reset functionality."""
        with IsolatedQdrantContainer() as container:
            client = container.client

            # Create test collection
            test_collection = "test_reset_collection"
            client.create_collection(
                collection_name=test_collection,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

            # Verify collection exists
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            assert test_collection in collection_names

            # Reset container
            container.reset()

            # Verify collection is gone
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            assert test_collection not in collection_names

    def test_multiple_containers_isolated(self):
        """Test that multiple containers are truly isolated."""
        container1 = IsolatedQdrantContainer()
        container2 = IsolatedQdrantContainer()

        try:
            container1.start()
            container2.start()

            # Create collection in first container
            test_collection = "isolation_test"
            container1.client.create_collection(
                collection_name=test_collection,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )

            # Verify collection exists in first container
            collections1 = container1.client.get_collections()
            names1 = [c.name for c in collections1.collections]
            assert test_collection in names1

            # Verify collection does NOT exist in second container
            collections2 = container2.client.get_collections()
            names2 = [c.name for c in collections2.collections]
            assert test_collection not in names2

        finally:
            container1.stop()
            container2.stop()


@pytest.mark.requires_docker
@pytest.mark.isolated_container
class TestTestcontainersFixtures:
    """Test pytest fixtures for testcontainers."""

    def test_isolated_qdrant_container_fixture(self, isolated_qdrant_container):
        """Test isolated container fixture."""
        assert isolated_qdrant_container._is_started
        assert isolated_qdrant_container.client is not None

        # Test basic operations
        collections = isolated_qdrant_container.client.get_collections()
        assert isinstance(collections, models.CollectionsResponse)

    async def test_isolated_qdrant_client_fixture(self, isolated_qdrant_client):
        """Test isolated client fixture."""
        assert isolated_qdrant_client is not None

        # Test basic operations
        collections = isolated_qdrant_client.get_collections()
        assert isinstance(collections, models.CollectionsResponse)

    def test_test_config_fixture(self, test_config):
        """Test configuration fixture."""
        assert test_config is not None
        assert test_config.qdrant.url.startswith("http://")
        assert test_config.workspace.github_user == "testuser"

    async def test_test_workspace_client_fixture(self, test_workspace_client):
        """Test workspace client fixture."""
        assert test_workspace_client is not None
        assert test_workspace_client.initialized

    def test_data_isolation_between_tests_1(self, isolated_qdrant_client):
        """First test to verify data isolation."""
        # Create a collection
        test_collection = "isolation_test_1"
        isolated_qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

        # Verify it exists
        collections = isolated_qdrant_client.get_collections()
        names = [c.name for c in collections.collections]
        assert test_collection in names

    def test_data_isolation_between_tests_2(self, isolated_qdrant_client):
        """Second test to verify data isolation."""
        # This test should not see collections from previous test
        collections = isolated_qdrant_client.get_collections()
        names = [c.name for c in collections.collections]
        assert "isolation_test_1" not in names

        # Create own collection
        test_collection = "isolation_test_2"
        isolated_qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

        # Verify it exists
        collections = isolated_qdrant_client.get_collections()
        names = [c.name for c in collections.collections]
        assert test_collection in names


@pytest.mark.requires_docker
@pytest.mark.shared_container
class TestSharedContainer:
    """Test shared container functionality."""

    def test_shared_container_reset_1(self, shared_qdrant_client):
        """First test using shared container."""
        # Create a collection
        test_collection = "shared_test_1"
        shared_qdrant_client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

        # Verify it exists
        collections = shared_qdrant_client.get_collections()
        names = [c.name for c in collections.collections]
        assert test_collection in names

    def test_shared_container_reset_2(self, shared_qdrant_client):
        """Second test using shared container - should be reset."""
        # Should not see collections from previous test due to reset
        collections = shared_qdrant_client.get_collections()
        names = [c.name for c in collections.collections]
        assert "shared_test_1" not in names


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_async_context_manager():
    """Test async context manager for isolated instances."""
    async with isolated_qdrant_instance() as (container, client):
        assert container._is_started
        assert client is not None

        # Test basic operations
        collections = client.get_collections()
        assert isinstance(collections, models.CollectionsResponse)

        # Create test collection
        test_collection = "async_context_test"
        client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

        # Verify collection exists
        collections = client.get_collections()
        names = [c.name for c in collections.collections]
        assert test_collection in names


@pytest.mark.requires_docker
@pytest.mark.performance
def test_container_startup_performance():
    """Test container startup performance."""
    import time

    start_time = time.time()
    with IsolatedQdrantContainer() as container:
        startup_time = time.time() - start_time

        # Container should start reasonably quickly
        # This is a performance test, so we allow more time
        assert startup_time < 30, f"Container startup took {startup_time:.2f}s, expected < 30s"

        # Test that it's functional
        collections = container.client.get_collections()
        assert isinstance(collections, models.CollectionsResponse)


@pytest.mark.requires_docker
def test_container_health_validation():
    """Test container health check and validation."""
    container = IsolatedQdrantContainer()

    try:
        container.start()

        # Health check should have passed
        assert container._is_started

        # Test that all expected operations work
        client = container.client

        # Basic operations
        collections = client.get_collections()
        assert isinstance(collections, models.CollectionsResponse)

        # Collection operations
        test_collection = "health_test"
        client.create_collection(
            collection_name=test_collection,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

        # Verify collection creation
        info = client.get_collection(test_collection)
        assert info.status == models.CollectionStatus.GREEN

        # Cleanup
        client.delete_collection(test_collection)

    finally:
        container.stop()
