"""
Testcontainers integration for isolated Qdrant testing.

Provides clean, isolated Qdrant container instances for testing that prevent
data contamination between tests and ensure reliable test results.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, AsyncGenerator, Tuple

import pytest
from testcontainers.qdrant import QdrantContainer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests

from workspace_qdrant_mcp.core.config import Config

logger = logging.getLogger(__name__)


class IsolatedQdrantContainer:
    """
    Enhanced Qdrant container with lifecycle management and health checks.

    Provides isolated Qdrant instances for testing with automatic cleanup
    and proper startup validation.
    """

    def __init__(
        self,
        image: str = "qdrant/qdrant:v1.7.4",
        http_port: int = 6333,
        grpc_port: int = 6334,
        startup_timeout: int = 60,
        health_check_interval: float = 1.0
    ):
        self.image = image
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.startup_timeout = startup_timeout
        self.health_check_interval = health_check_interval

        self._container: Optional[QdrantContainer] = None
        self._client: Optional[QdrantClient] = None
        self._is_started = False

    @property
    def container(self) -> QdrantContainer:
        """Get the underlying container instance."""
        if self._container is None:
            raise RuntimeError("Container not initialized. Call start() first.")
        return self._container

    @property
    def client(self) -> QdrantClient:
        """Get Qdrant client connected to the container."""
        if self._client is None:
            raise RuntimeError("Container not started. Call start() first.")
        return self._client

    @property
    def http_url(self) -> str:
        """Get HTTP URL for the Qdrant instance."""
        return f"http://{self.container.get_container_host_ip()}:{self.container.get_exposed_port(self.http_port)}"

    @property
    def grpc_url(self) -> str:
        """Get gRPC URL for the Qdrant instance."""
        return f"http://{self.container.get_container_host_ip()}:{self.container.get_exposed_port(self.grpc_port)}"

    def start(self) -> "IsolatedQdrantContainer":
        """Start the Qdrant container and wait for it to be ready."""
        if self._is_started:
            return self

        logger.info(f"Starting isolated Qdrant container with image {self.image}")

        # Create and start container
        self._container = QdrantContainer(self.image)
        self._container.with_exposed_ports(self.http_port, self.grpc_port)
        self._container.start()

        # Wait for health check
        self._wait_for_health()

        # Create client
        self._client = QdrantClient(
            url=self.http_url,
            timeout=30.0
        )

        # Validate client connection
        self._validate_connection()

        self._is_started = True
        logger.info(f"Qdrant container ready at {self.http_url}")
        return self

    def stop(self) -> None:
        """Stop and cleanup the container."""
        if not self._is_started:
            return

        logger.info("Stopping isolated Qdrant container")

        # Close client connection
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Qdrant client: {e}")
            finally:
                self._client = None

        # Stop container
        if self._container:
            try:
                self._container.stop()
            except Exception as e:
                logger.warning(f"Error stopping container: {e}")
            finally:
                self._container = None

        self._is_started = False
        logger.info("Qdrant container stopped")

    def _wait_for_health(self) -> None:
        """Wait for Qdrant container to be healthy."""
        start_time = time.time()
        health_url = f"{self.http_url}/health"

        while time.time() - start_time < self.startup_timeout:
            try:
                response = requests.get(health_url, timeout=5.0)
                if response.status_code == 200:
                    logger.debug("Qdrant health check passed")
                    return
            except Exception as e:
                logger.debug(f"Health check failed: {e}")

            time.sleep(self.health_check_interval)

        raise TimeoutError(
            f"Qdrant container failed to become healthy within {self.startup_timeout} seconds"
        )

    def _validate_connection(self) -> None:
        """Validate that the Qdrant client can connect and perform basic operations."""
        try:
            # Test basic connection
            collections = self._client.get_collections()
            logger.debug(f"Connected to Qdrant, found {len(collections.collections)} collections")

            # Test collection creation and deletion
            test_collection = "test_connection"
            try:
                self._client.create_collection(
                    collection_name=test_collection,
                    vectors_config=models.VectorParams(
                        size=384,
                        distance=models.Distance.COSINE
                    )
                )
                self._client.delete_collection(test_collection)
                logger.debug("Qdrant connection validation successful")
            except Exception as e:
                logger.warning(f"Collection operation test failed: {e}")

        except Exception as e:
            raise ConnectionError(f"Failed to validate Qdrant connection: {e}")

    def reset(self) -> None:
        """Reset container state by deleting all collections."""
        if not self._is_started or not self._client:
            return

        try:
            collections = self._client.get_collections()
            for collection in collections.collections:
                try:
                    self._client.delete_collection(collection.name)
                    logger.debug(f"Deleted collection: {collection.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete collection {collection.name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to reset container state: {e}")

    def __enter__(self) -> "IsolatedQdrantContainer":
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class QdrantContainerManager:
    """
    Manager for Qdrant container lifecycle across different test scopes.

    Provides optimized container management for different testing scenarios:
    - Session-scoped containers for integration test suites
    - Function-scoped containers for isolated unit tests
    - Shared containers with reset capability for performance
    """

    def __init__(self):
        self._containers: Dict[str, IsolatedQdrantContainer] = {}
        self._session_container: Optional[IsolatedQdrantContainer] = None

    def get_session_container(self) -> IsolatedQdrantContainer:
        """Get or create session-scoped container."""
        if self._session_container is None:
            self._session_container = IsolatedQdrantContainer()
            self._session_container.start()
        return self._session_container

    def get_isolated_container(self, test_id: str) -> IsolatedQdrantContainer:
        """Get isolated container for specific test."""
        if test_id not in self._containers:
            self._containers[test_id] = IsolatedQdrantContainer()
            self._containers[test_id].start()
        return self._containers[test_id]

    def cleanup_container(self, test_id: str) -> None:
        """Cleanup specific container."""
        if test_id in self._containers:
            self._containers[test_id].stop()
            del self._containers[test_id]

    def cleanup_session(self) -> None:
        """Cleanup session container."""
        if self._session_container:
            self._session_container.stop()
            self._session_container = None

    def cleanup_all(self) -> None:
        """Cleanup all containers."""
        for container in self._containers.values():
            container.stop()
        self._containers.clear()
        self.cleanup_session()


# Global container manager instance
_container_manager = QdrantContainerManager()


@asynccontextmanager
async def isolated_qdrant_instance() -> AsyncGenerator[Tuple[IsolatedQdrantContainer, QdrantClient], None]:
    """
    Async context manager for isolated Qdrant instances.

    Returns:
        Tuple of (container, client) for the isolated instance
    """
    container = IsolatedQdrantContainer()
    try:
        container.start()
        yield container, container.client
    finally:
        container.stop()


def create_test_config(container: IsolatedQdrantContainer) -> Config:
    """
    Create test configuration that uses the isolated container.

    Args:
        container: The isolated Qdrant container

    Returns:
        Config object pointing to the container
    """
    import os
    from workspace_qdrant_mcp.core.config import Config

    # Override environment variables to use container URL
    original_url = os.environ.get("WORKSPACE_QDRANT_QDRANT__URL")
    os.environ["WORKSPACE_QDRANT_QDRANT__URL"] = container.http_url

    try:
        # Create config with container URL
        config = Config()
        # Ensure the container URL is used
        config.qdrant.url = container.http_url
        return config
    finally:
        # Restore original environment
        if original_url is not None:
            os.environ["WORKSPACE_QDRANT_QDRANT__URL"] = original_url
        else:
            os.environ.pop("WORKSPACE_QDRANT_QDRANT__URL", None)


async def create_test_workspace_client(container: IsolatedQdrantContainer) -> "QdrantWorkspaceClient":
    """
    Create workspace client connected to isolated container.

    Args:
        container: The isolated Qdrant container

    Returns:
        Initialized workspace client
    """
    from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient

    config = create_test_config(container)
    client = QdrantWorkspaceClient(config)
    await client.initialize()
    return client


# Pytest fixtures for different scopes

def get_container_manager() -> QdrantContainerManager:
    """Get the global container manager."""
    return _container_manager


# Test markers for container requirements
requires_qdrant_container = pytest.mark.requires_qdrant_container
isolated_container = pytest.mark.isolated_container
shared_container = pytest.mark.shared_container