"""
Common test fixtures shared across all test domains.

Provides reusable fixtures for testcontainers, test data,
and common testing utilities.
"""

import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, List

from tests.shared.testcontainers_utils import (
    IsolatedQdrantContainer,
    get_shared_qdrant_container,
    release_shared_qdrant_container,
    cleanup_shared_containers,
)
from tests.shared.test_data import TestDataGenerator, SampleData


@pytest.fixture(scope="session")
def event_loop():
    """
    Create an event loop for the test session.

    This ensures async tests can run properly across the session.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def isolated_qdrant_container():
    """
    Provide isolated Qdrant container for individual test.

    Each test gets a fresh container that is torn down after the test.
    Use this for tests that need complete isolation.
    """
    container = IsolatedQdrantContainer()
    container.start()

    yield container

    container.stop()


@pytest.fixture(scope="session")
async def shared_qdrant_container():
    """
    Provide shared Qdrant container for test session.

    Container is shared across tests and reset between tests.
    Use this for integration tests where startup overhead matters.
    """
    container = await get_shared_qdrant_container()

    yield container

    await release_shared_qdrant_container(reset=True)


@pytest.fixture(scope="session", autouse=True)
async def cleanup_test_containers(request):
    """Automatically cleanup all test containers at session end."""

    def cleanup():
        asyncio.run(cleanup_shared_containers())

    request.addfinalizer(cleanup)


@pytest.fixture
def test_data_generator():
    """Provide test data generator instance."""
    return TestDataGenerator()


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return SampleData.get_sample_documents()


@pytest.fixture
def sample_queries():
    """Provide sample search queries for testing."""
    return SampleData.get_sample_queries()


@pytest.fixture
def edge_case_documents():
    """Provide edge case documents for testing."""
    return SampleData.get_edge_case_documents()


@pytest.fixture
def test_collection_name(test_data_generator):
    """Generate unique test collection name."""
    return test_data_generator.generate_collection_name("pytest")


@pytest.fixture
async def temp_test_workspace(tmp_path: Path) -> Path:
    """
    Create temporary workspace directory with realistic structure.

    Returns:
        Path to temporary workspace
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create standard project structure
    directories = [
        "src",
        "tests",
        "docs",
        "config",
        ".git",
    ]

    for directory in directories:
        (workspace / directory).mkdir()

    # Create sample files
    (workspace / "README.md").write_text("# Test Project\n\nTest workspace for testing.")
    (workspace / "pyproject.toml").write_text("[tool.pytest]\ntestpaths = ['tests']")
    (workspace / "src" / "__init__.py").write_text("# Package init")
    (workspace / "docs" / "guide.md").write_text("# User Guide\n\nDocumentation.")

    return workspace


@pytest.fixture
def embedding_dimensions() -> int:
    """Standard embedding dimensions for testing."""
    return 384  # sentence-transformers/all-MiniLM-L6-v2 dimensions


@pytest.fixture
def test_embeddings(test_data_generator, embedding_dimensions):
    """Generate test embedding vectors."""
    return [
        test_data_generator.generate_embedding(embedding_dimensions) for _ in range(5)
    ]


@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """
    Provide performance threshold expectations.

    Returns:
        Dict of operation -> max duration in milliseconds
    """
    return {
        "document_storage": 500.0,  # ms per document
        "search_query": 1000.0,  # ms per search
        "collection_create": 1000.0,  # ms to create collection
        "batch_insert": 5000.0,  # ms for 100 documents
        "tool_call": 100.0,  # ms for MCP tool call
    }


@pytest.fixture
def test_timeout_config() -> Dict[str, int]:
    """
    Provide timeout configuration for tests.

    Returns:
        Dict of operation -> timeout in seconds
    """
    return {
        "short": 5,  # Quick operations
        "medium": 30,  # Most operations
        "long": 60,  # Slow operations like full ingestion
        "stress": 300,  # Stress test operations
    }


@pytest.fixture
async def cleanup_collections(isolated_qdrant_container):
    """
    Track and cleanup test collections.

    Yields a function to register collections for cleanup.
    """
    collections_to_cleanup = []

    def register(collection_name: str):
        """Register collection for cleanup."""
        collections_to_cleanup.append(collection_name)

    yield register

    # Cleanup registered collections
    import httpx

    url = isolated_qdrant_container.get_http_url()
    async with httpx.AsyncClient() as client:
        for collection_name in collections_to_cleanup:
            try:
                await client.delete(f"{url}/collections/{collection_name}")
            except Exception:
                pass  # Ignore cleanup errors


@pytest.fixture
def mock_git_repo(tmp_path: Path) -> Path:
    """
    Create mock Git repository structure.

    Returns:
        Path to mock Git repository
    """
    repo_path = tmp_path / "mock_repo"
    repo_path.mkdir()

    # Create .git directory
    git_dir = repo_path / ".git"
    git_dir.mkdir()

    # Create Git config file
    (git_dir / "config").write_text(
        "[core]\n" "    repositoryformatversion = 0\n" "    filemode = true\n"
    )

    # Create some project files
    (repo_path / "README.md").write_text("# Mock Repository")
    (repo_path / "src").mkdir()
    (repo_path / "src" / "main.py").write_text("def main():\n    pass\n")

    return repo_path


@pytest.fixture
def environment_vars(tmp_path: Path) -> Dict[str, str]:
    """
    Provide test environment variables.

    Returns:
        Dict of environment variables for testing
    """
    return {
        "QDRANT_URL": "http://localhost:6333",
        "WQM_TEST_MODE": "true",
        "WQM_DATA_DIR": str(tmp_path / "data"),
        "WQM_CONFIG_DIR": str(tmp_path / "config"),
        "FASTEMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    }


@pytest.fixture
def skip_if_no_docker(request):
    """
    Skip test if Docker is not available.

    Use as decorator: @pytest.mark.usefixtures("skip_if_no_docker")
    """
    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception:
        pytest.skip("Docker not available")


@pytest.fixture
def skip_if_no_qdrant(request):
    """
    Skip test if Qdrant server is not available.

    Use as decorator: @pytest.mark.usefixtures("skip_if_no_qdrant")
    """
    try:
        import httpx

        response = httpx.get("http://localhost:6333/health", timeout=2.0)
        if response.status_code != 200:
            pytest.skip("Qdrant server not available")
    except Exception:
        pytest.skip("Qdrant server not available")