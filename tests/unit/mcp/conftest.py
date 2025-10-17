"""
MCP tool call validation test fixtures using official fastmcp.Client SDK.

Provides official MCP SDK test infrastructure for validating tool call handling,
parameter validation, and MCP protocol compliance.

Migration from custom FastMCPTestServer to official SDK (Task 325.4).
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from fastmcp import Client
from fastmcp.client.client import CallToolResult


@pytest.fixture
async def mcp_client():
    """
    Official SDK fixture using fastmcp.Client with in-memory transport.

    Replaces custom FastMCPTestServer/FastMCPTestClient infrastructure
    with the official MCP SDK pattern recommended by the MCP team.

    Applies mocks for Qdrant and daemon dependencies before Client initialization.
    """
    from workspace_qdrant_mcp.server import app

    # Set up mocks for external dependencies
    mock_qdrant = Mock()
    mock_qdrant.get_collections.return_value.collections = [
        Mock(name="_test_project_id", points_count=100),
        Mock(name="_memory", points_count=50),
    ]
    mock_qdrant.get_collection.return_value = Mock(
        points_count=100,
        segments_count=1,
        status=Mock(value="green"),
        config=Mock(
            params=Mock(
                vectors=Mock(
                    size=384,
                    distance=Mock(value="Cosine")
                )
            )
        ),
        indexed_vectors_count=100,
        optimizer_status="ok"
    )
    mock_qdrant.search.return_value = []
    mock_qdrant.scroll.return_value = ([], None)

    mock_daemon = AsyncMock()
    mock_daemon.ping.return_value = {"status": "ok", "connected": True}
    mock_daemon.ingest_text.return_value = Mock(
        success=True,
        document_id="test-doc-123",
        chunks_created=1,
        error_message=None
    )
    mock_daemon.ingest_file.return_value = Mock(
        success=True,
        document_id="test-doc-456",
        chunks_created=1,
        error_message=None
    )

    # Mock utility functions to prevent business logic from running
    # Use path hash format: "path_" + 16-char hex
    mock_calculate_tenant_id = Mock(return_value="path_abc123def456789a")
    mock_get_current_branch = Mock(return_value="main")

    # Apply patches before creating Client
    with patch("workspace_qdrant_mcp.server.qdrant_client", mock_qdrant), \
         patch("workspace_qdrant_mcp.server.daemon_client", mock_daemon), \
         patch("workspace_qdrant_mcp.server.calculate_tenant_id", mock_calculate_tenant_id), \
         patch("workspace_qdrant_mcp.server.get_current_branch", mock_get_current_branch):

        # Use async context manager for automatic initialization and cleanup
        async with Client(app) as client:
            # Context manager handles initialization automatically
            yield client
            # Cleanup is automatic when context exits


# Keep legacy fixtures for backward compatibility during migration
# These will be removed once all tests are migrated


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
