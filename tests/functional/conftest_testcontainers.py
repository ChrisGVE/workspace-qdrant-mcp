"""
Testcontainers configuration for workspace-qdrant-mcp functional testing.

This module provides containerized services for isolated testing including:
- Qdrant vector database instances
- Mock external services
- Network isolation testing scenarios
"""

import asyncio
import json
import logging
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from testcontainers.compose import DockerCompose
from testcontainers.qdrant import QdrantContainer

logger = logging.getLogger(__name__)

# Container configuration
QDRANT_VERSION = "v1.7.4"
QDRANT_PORT = 6333
QDRANT_GRPC_PORT = 6334


@pytest.fixture(scope="session")
def docker_available() -> bool:
    """Check if Docker is available for testing."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception as e:
        pytest.skip(f"Docker not available: {e}")
        return False


@pytest.fixture(scope="session")
def qdrant_container(docker_available) -> Generator[QdrantContainer, None, None]:
    """
    Provide a Qdrant container for testing.

    Yields:
        QdrantContainer: Configured Qdrant container instance
    """
    if not docker_available:
        pytest.skip("Docker not available")

    with QdrantContainer(f"qdrant/qdrant:{QDRANT_VERSION}") as qdrant:
        # Wait for container to be ready
        qdrant.start()

        # Verify container health
        client = QdrantClient(
            host=qdrant.get_container_host_ip(),
            port=qdrant.get_exposed_port(QDRANT_PORT),
            timeout=10
        )

        # Wait for Qdrant to be ready
        max_retries = 30
        for i in range(max_retries):
            try:
                client.get_collections()
                logger.info(f"Qdrant container ready after {i} retries")
                break
            except Exception as e:
                if i == max_retries - 1:
                    raise RuntimeError(f"Qdrant container failed to start: {e}")
                asyncio.sleep(1)

        yield qdrant


@pytest.fixture
def qdrant_client(qdrant_container: QdrantContainer) -> Generator[QdrantClient, None, None]:
    """
    Provide a Qdrant client connected to the test container.

    Args:
        qdrant_container: Qdrant container fixture

    Yields:
        QdrantClient: Connected client instance
    """
    client = QdrantClient(
        host=qdrant_container.get_container_host_ip(),
        port=qdrant_container.get_exposed_port(QDRANT_PORT),
        timeout=30
    )

    yield client

    # Cleanup: remove all collections after test
    try:
        collections = client.get_collections()
        for collection in collections.collections:
            client.delete_collection(collection.name)
    except Exception as e:
        logger.warning(f"Failed to cleanup collections: {e}")


@pytest.fixture
def test_collection(qdrant_client: QdrantClient) -> Generator[str, None, None]:
    """
    Create a test collection with standard configuration.

    Args:
        qdrant_client: Connected Qdrant client

    Yields:
        str: Collection name
    """
    collection_name = "test_collection"

    # Create collection with standard vector configuration
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=384,  # Standard embedding size
            distance=Distance.COSINE
        )
    )

    yield collection_name

    # Cleanup
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception as e:
        logger.warning(f"Failed to cleanup collection {collection_name}: {e}")


@pytest.fixture(scope="session")
def docker_compose_environment() -> Generator[DockerCompose, None, None]:
    """
    Provide a Docker Compose environment for complex testing scenarios.

    This fixture starts multiple services:
    - Qdrant database
    - Mock external APIs
    - Network isolation testing
    """
    compose_file_content = """
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
      - "6334:6334"
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    volumes:
      - ./qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  mock_api:
    image: wiremock/wiremock:2.35.0
    ports:
      - "8080:8080"
    volumes:
      - ./mock_api_mappings:/home/wiremock/mappings
    command: ["--port", "8080", "--verbose"]
"""

    # Create temporary directory for compose environment
    with tempfile.TemporaryDirectory() as temp_dir:
        compose_path = Path(temp_dir)

        # Write compose file
        with open(compose_path / "docker-compose.yml", "w") as f:
            f.write(compose_file_content)

        # Create directories
        (compose_path / "qdrant_data").mkdir()
        (compose_path / "mock_api_mappings").mkdir()

        # Create mock API mapping
        mock_mapping = {
            "request": {
                "method": "GET",
                "url": "/api/v1/test"
            },
            "response": {
                "status": 200,
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": json.dumps({"status": "ok", "message": "Test API"})
            }
        }

        with open(compose_path / "mock_api_mappings" / "test.json", "w") as f:
            json.dump(mock_mapping, f)

        # Start compose environment
        with DockerCompose(str(compose_path)) as compose:
            yield compose


@pytest.fixture
def isolated_environment(docker_compose_environment) -> dict[str, Any]:
    """
    Provide connection details for isolated testing environment.

    Returns:
        Dict containing service endpoints and connection details
    """
    return {
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "url": "http://localhost:6333"
        },
        "mock_api": {
            "host": "localhost",
            "port": 8080,
            "url": "http://localhost:8080"
        }
    }


@pytest.fixture
def performance_qdrant_client(qdrant_container: QdrantContainer) -> Generator[QdrantClient, None, None]:
    """
    Provide a Qdrant client configured for performance testing.

    Args:
        qdrant_container: Qdrant container fixture

    Yields:
        QdrantClient: Performance-optimized client
    """
    client = QdrantClient(
        host=qdrant_container.get_container_host_ip(),
        port=qdrant_container.get_exposed_port(QDRANT_PORT),
        timeout=60,  # Longer timeout for performance tests
        prefer_grpc=True  # Use gRPC for better performance
    )

    yield client


# Utility functions for container management
def wait_for_container_health(container, port: int, max_retries: int = 30) -> bool:
    """Wait for container to become healthy."""
    import requests

    for _i in range(max_retries):
        try:
            response = requests.get(
                f"http://{container.get_container_host_ip()}:{container.get_exposed_port(port)}/health",
                timeout=5
            )
            if response.status_code == 200:
                return True
        except Exception:
            pass
        asyncio.sleep(1)

    return False


def get_container_logs(container) -> str:
    """Get container logs for debugging."""
    try:
        return container.get_logs()[0].decode('utf-8')
    except Exception as e:
        return f"Failed to get logs: {e}"
