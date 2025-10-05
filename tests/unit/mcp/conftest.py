"""
MCP tool call validation test fixtures.

Provides FastMCP test infrastructure for validating tool call handling,
parameter validation, and MCP protocol compliance.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from tests.utils.fastmcp_test_infrastructure import (
    FastMCPTestServer,
    FastMCPTestClient,
    MCPProtocolTester,
)


@pytest.fixture
async def fastmcp_test_server():
    """
    Provide FastMCP test server with mocked dependencies.

    Uses the actual workspace-qdrant-mcp FastMCP app but mocks
    external dependencies (Qdrant, daemon, filesystem).
    """
    from workspace_qdrant_mcp.server import app

    # Create and initialize test server
    server = FastMCPTestServer(app, name="mcp-validation-server")
    await server.initialize()

    yield server

    await server.cleanup()


@pytest.fixture
async def fastmcp_test_client(fastmcp_test_server):
    """
    Provide FastMCP test client connected to test server.

    The client can directly invoke MCP tools and validate responses.
    """
    client = await fastmcp_test_server.create_test_client()

    yield client

    await client.close()


@pytest.fixture
async def mcp_protocol_tester(fastmcp_test_server):
    """
    Provide MCP protocol compliance tester.

    Runs comprehensive protocol validation tests on the FastMCP server.
    """
    return MCPProtocolTester(fastmcp_test_server)


@pytest.fixture
def mock_qdrant_collections():
    """
    Provide mock Qdrant collections for testing.

    Returns a pre-configured set of collections with realistic structure.
    """
    return [
        Mock(
            name="_test_project_id",
            points_count=100,
            segments_count=1,
            status="green",
        ),
        Mock(
            name="_memory",
            points_count=50,
            segments_count=1,
            status="green",
        ),
        Mock(
            name="custom-notes",
            points_count=25,
            segments_count=1,
            status="green",
        ),
    ]


@pytest.fixture
def mock_qdrant_collection_info():
    """
    Provide mock collection info for detailed collection queries.
    """
    mock_info = Mock()
    mock_info.points_count = 100
    mock_info.segments_count = 1
    mock_info.status.value = "green"
    mock_info.config.params.vectors.size = 384
    mock_info.config.params.vectors.distance.value = "Cosine"
    mock_info.indexed_vectors_count = 100
    mock_info.optimizer_status = "ok"

    return mock_info


@pytest.fixture
def mock_daemon_response_success():
    """
    Provide mock successful daemon response.
    """
    response = Mock()
    response.success = True
    response.document_id = "test-doc-123"
    response.chunks_created = 1
    response.error_message = None

    return response


@pytest.fixture
def mock_daemon_response_failure():
    """
    Provide mock failed daemon response.
    """
    response = Mock()
    response.success = False
    response.document_id = None
    response.chunks_created = 0
    response.error_message = "Daemon connection failed"

    return response


@pytest.fixture
def sample_document_metadata():
    """
    Provide sample document metadata for testing.
    """
    return {
        "title": "Test Document",
        "source": "test",
        "document_type": "text",
        "file_type": "code",
        "branch": "main",
        "project": "test-project",
    }


@pytest.fixture
def sample_search_filters():
    """
    Provide sample search filters for testing.
    """
    return {
        "branch": "main",
        "file_type": "code",
        "source": "test",
    }
