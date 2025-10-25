"""
Testcontainers utilities for Qdrant isolation.

Provides utilities for managing isolated Qdrant containers in tests,
including lifecycle management, health checks, and cleanup.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs


class IsolatedQdrantContainer:
    """
    Manage an isolated Qdrant container for testing.

    Provides a clean Qdrant instance that is started, health-checked,
    and automatically cleaned up after use.
    """

    def __init__(
        self,
        image: str = "qdrant/qdrant:v1.7.4",
        http_port: int = 6333,
        grpc_port: int = 6334,
    ):
        """
        Initialize Qdrant container configuration.

        Args:
            image: Docker image to use
            http_port: HTTP API port (will be mapped to random host port)
            grpc_port: gRPC port (will be mapped to random host port)
        """
        self.image = image
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.container: DockerContainer | None = None
        self._http_url: str | None = None

    def start(self) -> "IsolatedQdrantContainer":
        """
        Start the Qdrant container.

        Returns:
            Self for chaining

        Raises:
            RuntimeError: If container fails to start
        """
        self.container = DockerContainer(self.image)
        self.container.with_exposed_ports(self.http_port, self.grpc_port)

        # Start container
        self.container.start()

        # Get mapped ports
        host = self.container.get_container_host_ip()
        http_mapped_port = self.container.get_exposed_port(self.http_port)
        self._http_url = f"http://{host}:{http_mapped_port}"

        # Wait for Qdrant to be ready
        self._wait_for_health()

        return self

    def stop(self):
        """Stop and remove the container."""
        if self.container:
            self.container.stop()
            self.container = None
            self._http_url = None

    def get_http_url(self) -> str:
        """
        Get the HTTP URL for the Qdrant instance.

        Returns:
            HTTP URL (e.g., "http://localhost:12345")

        Raises:
            RuntimeError: If container not started
        """
        if not self._http_url:
            raise RuntimeError("Container not started")
        return self._http_url

    def get_grpc_port(self) -> int:
        """
        Get the mapped gRPC port.

        Returns:
            Mapped gRPC port number

        Raises:
            RuntimeError: If container not started
        """
        if not self.container:
            raise RuntimeError("Container not started")
        return int(self.container.get_exposed_port(self.grpc_port))

    def _wait_for_health(self, timeout: int = 30):
        """
        Wait for Qdrant to be healthy.

        Args:
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If health check times out
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = httpx.get(
                    f"{self._http_url}/health", timeout=2.0, follow_redirects=True
                )
                if response.status_code == 200:
                    return
            except (httpx.RequestError, httpx.HTTPError):
                pass

            time.sleep(0.5)

        raise RuntimeError(f"Qdrant failed to become healthy within {timeout}s")

    async def reset(self):
        """
        Reset container state by deleting all collections.

        Useful for reusing a container across tests while maintaining isolation.
        """
        if not self._http_url:
            raise RuntimeError("Container not started")

        async with httpx.AsyncClient() as client:
            # Get all collections
            response = await client.get(f"{self._http_url}/collections")
            response.raise_for_status()
            collections = response.json().get("result", {}).get("collections", [])

            # Delete each collection
            for collection in collections:
                collection_name = collection["name"]
                await client.delete(f"{self._http_url}/collections/{collection_name}")

    def __enter__(self):
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


@asynccontextmanager
async def isolated_qdrant_instance(
    image: str = "qdrant/qdrant:v1.7.4",
):
    """
    Async context manager for isolated Qdrant instance.

    Args:
        image: Docker image to use

    Yields:
        tuple: (container, http_url)

    Example:
        async with isolated_qdrant_instance() as (container, url):
            # Use url for Qdrant client
            client = QdrantClient(url=url)
            # ... run tests ...
    """
    container = IsolatedQdrantContainer(image=image)
    container.start()

    try:
        yield container, container.get_http_url()
    finally:
        container.stop()


class QdrantContainerManager:
    """
    Manage Qdrant container lifecycle across test scopes.

    Supports session-scoped containers with reset capability
    for better test performance.
    """

    def __init__(self):
        self._container: IsolatedQdrantContainer | None = None
        self._reference_count = 0

    async def get_or_create(
        self, image: str = "qdrant/qdrant:v1.7.4"
    ) -> IsolatedQdrantContainer:
        """
        Get existing container or create new one.

        Args:
            image: Docker image to use

        Returns:
            Qdrant container instance
        """
        if self._container is None:
            self._container = IsolatedQdrantContainer(image=image)
            self._container.start()

        self._reference_count += 1
        return self._container

    async def release(self, reset: bool = True):
        """
        Release reference to container.

        Args:
            reset: Whether to reset container state (delete collections)
        """
        if self._container is None:
            return

        self._reference_count -= 1

        if reset:
            await self._container.reset()

        # Stop container when no more references
        if self._reference_count <= 0:
            self._container.stop()
            self._container = None
            self._reference_count = 0

    async def cleanup(self):
        """Force cleanup of container regardless of reference count."""
        if self._container:
            self._container.stop()
            self._container = None
            self._reference_count = 0


# Global manager for session-scoped containers
_global_container_manager = QdrantContainerManager()


async def get_shared_qdrant_container(
    image: str = "qdrant/qdrant:v1.7.4",
) -> IsolatedQdrantContainer:
    """
    Get shared Qdrant container for test session.

    This is useful for integration tests where container startup
    overhead should be minimized. Container state is reset between tests.

    Args:
        image: Docker image to use

    Returns:
        Shared Qdrant container instance
    """
    return await _global_container_manager.get_or_create(image)


async def release_shared_qdrant_container(reset: bool = True):
    """
    Release reference to shared container.

    Args:
        reset: Whether to reset container state
    """
    await _global_container_manager.release(reset)


async def cleanup_shared_containers():
    """Cleanup all shared containers (called at session end)."""
    await _global_container_manager.cleanup()
