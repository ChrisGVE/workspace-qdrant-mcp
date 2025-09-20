"""
Pytest configuration and fixtures for workspace-qdrant-mcp tests.

This module provides common test fixtures and configuration
for all test modules in the project.
"""

import asyncio
import logging
import os
import tempfile
import shutil
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from common.core.config import Config
from tests.utils.fastmcp_test_infrastructure import (
    FastMCPTestServer,
    FastMCPTestClient,
    MCPProtocolTester,
    fastmcp_test_environment
)
from tests.utils.pytest_mcp_framework import (
    AITestEvaluator,
    MCPToolEvaluator,
    IntelligentTestRunner,
    ai_powered_mcp_testing
)
from tests.utils.testcontainers_qdrant import (
    IsolatedQdrantContainer,
    QdrantContainerManager,
    get_container_manager,
    create_test_config,
    create_test_workspace_client,
    isolated_qdrant_instance
)


# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def temp_directory() -> AsyncGenerator[Path, None]:
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="wqm_test_")
    temp_path = Path(temp_dir)
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def environment_variables():
    """Set up test environment variables."""
    test_env = {
        "WORKSPACE_QDRANT_HOST": "127.0.0.1",
        "WORKSPACE_QDRANT_PORT": "8000",
        "WORKSPACE_QDRANT_DEBUG": "true",
        "WORKSPACE_QDRANT_QDRANT__URL": "http://localhost:6333",
        "WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER": "testuser"
    }
    
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield test_env
    
    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=Config)
    config.host = "127.0.0.1"
    config.port = 8000
    config.debug = True
    config.qdrant.url = "http://localhost:6333"
    config.qdrant.api_key = None
    config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding.chunk_size = 500
    config.embedding.chunk_overlap = 50
    config.workspace.github_user = "testuser"
    config.workspace.collections = ["project"]
    config.workspace.global_collections = ["docs"]
    return config


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = Mock(spec=QdrantClient)
    
    # Mock basic operations
    client.search = AsyncMock(return_value=[])
    client.upsert = AsyncMock(return_value=models.UpdateResult(
        operation_id=123, 
        status=models.UpdateStatus.COMPLETED
    ))
    client.delete = AsyncMock(return_value=models.UpdateResult(
        operation_id=124,
        status=models.UpdateStatus.COMPLETED
    ))
    client.retrieve = AsyncMock(return_value=[])
    client.scroll = AsyncMock(return_value=([], None))
    client.count = AsyncMock(return_value=models.CountResult(count=0))
    
    # Mock collection operations
    client.get_collection = AsyncMock(return_value=models.CollectionInfo(
        status=models.CollectionStatus.GREEN,
        points_count=0,
        segments_count=1,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        config=models.CollectionConfig()
    ))
    client.create_collection = AsyncMock(return_value=True)
    client.delete_collection = AsyncMock(return_value=True)
    client.collection_exists = AsyncMock(return_value=True)
    client.get_collections = AsyncMock(return_value=models.CollectionsResponse(
        collections=[]
    ))
    
    # Mock client lifecycle
    client.close = Mock()
    
    return client


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    service = Mock()
    service.initialize = AsyncMock()
    service.close = AsyncMock()
    service.generate_embeddings = AsyncMock(return_value={
        "dense": [0.1] * 384,  # Standard 384-dim embedding
        "sparse": {
            "indices": [1, 5, 10, 20, 35],
            "values": [0.8, 0.6, 0.9, 0.7, 0.5]
        }
    })
    return service


@pytest.fixture
async def mock_workspace_client(mock_config, mock_qdrant_client, mock_embedding_service):
    """Mock workspace client for testing."""
    from common.core.client import QdrantWorkspaceClient
    
    client = Mock(spec=QdrantWorkspaceClient)
    client.initialized = True
    client.config = mock_config
    client.client = mock_qdrant_client
    
    # Mock basic methods
    client.initialize = AsyncMock()
    client.close = AsyncMock()
    client.get_status = AsyncMock(return_value={
        "connected": True,
        "current_project": "test-project",
        "collections": ["test-project_docs", "test-project_scratchbook"]
    })
    client.list_collections = AsyncMock(return_value=[
        "test-project_docs", 
        "test-project_scratchbook"
    ])
    
    # Mock embedding service
    client.get_embedding_service = Mock(return_value=mock_embedding_service)
    client.embedding_service = mock_embedding_service
    
    # Mock project detection
    client.project_detector = Mock()
    client.project_detector.get_project_info = Mock(return_value={
        "main_project": "test-project",
        "subprojects": [],
        "git_root": "/tmp/test-project",
        "is_git_repo": True
    })
    
    # Mock collection manager
    client.collection_manager = Mock()
    client.collection_manager.get_collection_name = Mock(return_value="test_collection")
    client.collection_manager.ensure_collection_exists = AsyncMock()
    client.collection_manager.validate_mcp_write_access = Mock()
    client.collection_manager.resolve_collection_name = Mock(return_value=("test", "test"))
    client.collection_manager.list_workspace_collections = AsyncMock(return_value=[
        "test-project_docs",
        "test-project_scratchbook"
    ])
    client.collection_manager.initialize_workspace_collections = AsyncMock()
    
    # Mock search operations
    client.search = AsyncMock(return_value={
        "results": [],
        "total": 0,
        "query": "test"
    })
    
    yield client


@pytest.fixture
def temp_git_repo_with_submodules(temp_directory: Path):
    """Create a temporary git repository with mock submodules for testing."""
    git_dir = temp_directory / "test_git_repo"
    git_dir.mkdir()
    
    # Create .git directory
    (git_dir / ".git").mkdir()
    (git_dir / ".git" / "config").write_text("[core]\n    repositoryformatversion = 0")
    
    # Create some source files
    src_dir = git_dir / "src"
    src_dir.mkdir()
    (src_dir / "main.py").write_text("def main():\n    pass")
    (src_dir / "utils.py").write_text("def helper():\n    return True")
    
    # Create README
    (git_dir / "README.md").write_text("# Test Project\n\nThis is a test project.")
    
    return git_dir


@pytest.fixture
async def sample_documents(temp_directory: Path) -> AsyncGenerator[list[Path], None]:
    """Create sample documents for testing."""
    documents = []
    
    # Create various file types
    txt_file = temp_directory / "sample.txt"
    txt_file.write_text("This is a sample text document for testing.")
    documents.append(txt_file)
    
    md_file = temp_directory / "sample.md"
    md_file.write_text("# Sample Markdown\n\nThis is a markdown document.")
    documents.append(md_file)
    
    py_file = temp_directory / "sample.py"
    py_file.write_text('''
"""Sample Python module."""

def sample_function(x, y):
    """Add two numbers together."""
    return x + y

class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, value):
        self.value = value
        
    def get_value(self):
        """Get the stored value."""
        return self.value
''')
    documents.append(py_file)
    
    pdf_placeholder = temp_directory / "sample.pdf" 
    pdf_placeholder.write_bytes(b"PDF placeholder content")
    documents.append(pdf_placeholder)
    
    yield documents


@pytest.fixture
def mock_ingestion_callback():
    """Mock ingestion callback for file watching tests."""
    return AsyncMock()


@pytest.fixture  
def mock_event_callback():
    """Mock event callback for file watching tests."""
    return Mock()


# Test configuration
pytest_plugins = ["pytest_asyncio"]


# Pytest markers configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "performance: Performance and benchmark tests")
    config.addinivalue_line("markers", "requires_qdrant: Tests requiring Qdrant server")
    config.addinivalue_line("markers", "requires_git: Tests requiring Git repository")
    config.addinivalue_line("markers", "regression: Regression tests for bug fixes")
    config.addinivalue_line("markers", "smoke: Smoke tests for basic functionality")
    config.addinivalue_line("markers", "fastmcp: FastMCP protocol and infrastructure tests")
    config.addinivalue_line("markers", "mcp_tools: MCP tool functionality tests")
    config.addinivalue_line("markers", "ai_evaluation: AI-powered evaluation tests")
    config.addinivalue_line("markers", "intelligent_testing: Intelligent test runner tests")
    config.addinivalue_line("markers", "pytest_mcp: pytest-mcp framework tests")
    config.addinivalue_line("markers", "requires_qdrant_container: Tests requiring isolated Qdrant container")
    config.addinivalue_line("markers", "isolated_container: Tests using isolated containers")
    config.addinivalue_line("markers", "shared_container: Tests using shared containers")
    config.addinivalue_line("markers", "requires_docker: Tests requiring Docker daemon")


@pytest.fixture
async def fastmcp_test_server():
    """FastMCP test server fixture for in-memory testing."""
    # Import here to avoid circular imports
    from workspace_qdrant_mcp.server import app

    async with FastMCPTestServer(app, "pytest-server") as server:
        yield server


@pytest.fixture
async def fastmcp_test_client(fastmcp_test_server):
    """FastMCP test client fixture connected to test server."""
    client = await fastmcp_test_server.create_test_client()
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
async def mcp_protocol_tester(fastmcp_test_server):
    """MCP protocol tester fixture for comprehensive compliance testing."""
    tester = MCPProtocolTester(fastmcp_test_server)
    yield tester


@pytest.fixture
async def fastmcp_test_environment():
    """Complete FastMCP testing environment with server and client."""
    from workspace_qdrant_mcp.server import app

    async with fastmcp_test_environment(app, "pytest-environment") as (server, client):
        yield server, client


@pytest.fixture
async def ai_test_evaluator():
    """AI test evaluator fixture for intelligent MCP evaluation."""
    evaluator = AITestEvaluator()
    yield evaluator


@pytest.fixture
async def mcp_tool_evaluator(ai_test_evaluator):
    """MCP tool evaluator fixture for comprehensive tool testing."""
    evaluator = MCPToolEvaluator(ai_test_evaluator)
    yield evaluator


@pytest.fixture
async def intelligent_test_runner(ai_test_evaluator):
    """Intelligent test runner fixture for AI-powered test execution."""
    runner = IntelligentTestRunner(ai_test_evaluator)
    yield runner


@pytest.fixture
async def ai_powered_test_environment():
    """AI-powered MCP testing environment with intelligent evaluation."""
    from workspace_qdrant_mcp.server import app

    async with ai_powered_mcp_testing(app, "pytest-ai-environment") as runner:
        yield runner


# Testcontainers fixtures for isolated Qdrant testing

@pytest.fixture(scope="session")
def qdrant_container_manager():
    """Session-scoped container manager for Qdrant instances."""
    manager = get_container_manager()
    yield manager
    # Cleanup all containers at end of session
    manager.cleanup_all()


@pytest.fixture(scope="session")
def session_qdrant_container(qdrant_container_manager):
    """Session-scoped Qdrant container for integration tests."""
    container = qdrant_container_manager.get_session_container()
    yield container
    # Reset container state between test modules
    container.reset()


@pytest.fixture
def isolated_qdrant_container(request, qdrant_container_manager):
    """Function-scoped isolated Qdrant container for unit tests."""
    test_id = f"{request.module.__name__}::{request.function.__name__}"
    container = qdrant_container_manager.get_isolated_container(test_id)

    yield container

    # Cleanup after test
    qdrant_container_manager.cleanup_container(test_id)


@pytest.fixture
def shared_qdrant_container(session_qdrant_container):
    """Shared Qdrant container that resets state between tests."""
    # Reset state before each test
    session_qdrant_container.reset()
    yield session_qdrant_container


@pytest.fixture
async def isolated_qdrant_client(isolated_qdrant_container):
    """Qdrant client connected to isolated container."""
    yield isolated_qdrant_container.client


@pytest.fixture
async def shared_qdrant_client(shared_qdrant_container):
    """Qdrant client connected to shared container."""
    yield shared_qdrant_container.client


@pytest.fixture
async def test_workspace_client(isolated_qdrant_container):
    """Workspace client connected to isolated Qdrant container."""
    client = await create_test_workspace_client(isolated_qdrant_container)
    yield client
    await client.close()


@pytest.fixture
def test_config(isolated_qdrant_container):
    """Test configuration using isolated Qdrant container."""
    yield create_test_config(isolated_qdrant_container)


@pytest.fixture
async def containerized_qdrant_instance():
    """Async context manager for isolated Qdrant instance."""
    async with isolated_qdrant_instance() as (container, client):
        yield container, client


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "functional" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath) or "test_performance" in item.name:
            item.add_marker(pytest.mark.performance)
        elif "memory" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)