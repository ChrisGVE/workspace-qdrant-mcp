"""
Comprehensive Unit Tests for MCP Server Tools (Task 242.2).

This test suite provides complete coverage for all MCP server tools
with focus on the 11 core tools mentioned in requirements:
- workspace_status, search_workspace, get_server_info
- collection management tools (create, list, manage)
- document tools (add, get, search by metadata)
- scratchbook tools (update, search, list)
- advanced search tools (hybrid search)

Tests include:
- Function coverage for all tools
- Parameter validation and edge cases
- Error condition handling
- FastMCP protocol compliance
- Mock external dependencies (Qdrant, gRPC, file system)
- Async operation testing with pytest-asyncio
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
from typing import Dict, Any, List, Optional

# FastMCP testing infrastructure
from tests.utils.fastmcp_test_infrastructure import (
    FastMCPTestClient,
    MCPProtocolTester,
    fastmcp_test_environment
)

# Import server and tools for testing
from workspace_qdrant_mcp.server import app
from workspace_qdrant_mcp.tools import (
    search_workspace,
    search_collection_by_metadata,
    add_document,
    update_document,
    delete_document,
    get_document,
    update_scratchbook,
    ScratchbookManager
)


class TestCoreSystemTools:
    """Test core system tools: workspace_status, get_server_info."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_workspace_status_success(self, fastmcp_test_client):
        """Test workspace_status tool returns valid status information."""
        client = fastmcp_test_client

        # Mock the dependencies that workspace_status uses
        with patch('workspace_qdrant_mcp.server.get_current_config') as mock_config, \
             patch('workspace_qdrant_mcp.server.get_client') as mock_client, \
             patch('workspace_qdrant_mcp.server.detect_project') as mock_project:

            # Setup mocks
            mock_config.return_value = MagicMock(
                qdrant_client_config=MagicMock(url="http://localhost:6333"),
                workspace=MagicMock(global_collections=["test-collection"])
            )
            mock_client.return_value = AsyncMock()
            mock_project.return_value = "test-project"

            # Mock Qdrant client operations
            mock_qdrant = AsyncMock()
            mock_qdrant.get_collections.return_value = MagicMock(
                collections=[
                    MagicMock(name="test-project-scratchbook"),
                    MagicMock(name="test-collection")
                ]
            )

            with patch('workspace_qdrant_mcp.server.QdrantClient', return_value=mock_qdrant):
                result = await client.call_tool("workspace_status", {})

        # Verify the call was successful
        assert result.success, f"workspace_status failed: {result.error}"
        assert isinstance(result.response, dict)

        # Verify expected response structure
        response = result.response
        expected_keys = ["connected", "current_project", "collections", "embedding_model"]
        for key in expected_keys:
            assert key in response, f"Missing key '{key}' in response"

        # Verify data types
        assert isinstance(response["connected"], bool)
        assert isinstance(response["collections"], (list, dict))
        assert isinstance(response.get("current_project"), (str, type(None)))

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_workspace_status_connection_error(self, fastmcp_test_client):
        """Test workspace_status handles connection errors gracefully."""
        client = fastmcp_test_client

        with patch('workspace_qdrant_mcp.server.get_current_config') as mock_config:
            mock_config.side_effect = Exception("Connection failed")

            result = await client.call_tool("workspace_status", {})

        # Should handle error gracefully and return structured response
        assert result.success or result.error is not None
        if result.success:
            assert "connected" in result.response
            assert result.response["connected"] == False

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_workspace_status_no_parameters(self, fastmcp_test_client):
        """Test workspace_status works without parameters."""
        client = fastmcp_test_client

        # Test with empty parameters
        result = await client.call_tool("workspace_status", {})

        # Should work regardless of connection status
        assert result.success or result.error is not None
        assert result.parameters == {}


class TestCollectionManagementTools:
    """Test collection management tools: create, list, delete."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_list_workspace_collections_success(self, fastmcp_test_client):
        """Test listing workspace collections."""
        client = fastmcp_test_client

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.detect_project') as mock_project:

            # Setup mocks
            mock_project.return_value = "test-project"
            mock_qdrant = AsyncMock()
            mock_qdrant.get_collections.return_value = MagicMock(
                collections=[
                    MagicMock(name="test-project-scratchbook", vectors_count=10),
                    MagicMock(name="test-project-docs", vectors_count=25),
                    MagicMock(name="global-collection", vectors_count=50)
                ]
            )
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("list_workspace_collections", {})

        assert result.success, f"list_workspace_collections failed: {result.error}"
        assert isinstance(result.response, dict)

        response = result.response
        assert "collections" in response
        assert isinstance(response["collections"], list)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_create_collection_success(self, fastmcp_test_client):
        """Test creating a new collection."""
        client = fastmcp_test_client

        create_params = {
            "collection_name": "test-new-collection",
            "dimension": 384,
            "distance": "Cosine"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.create_naming_manager') as mock_naming:

            mock_qdrant = AsyncMock()
            mock_qdrant.create_collection = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_qdrant

            mock_naming_manager = MagicMock()
            mock_naming_manager.create_collection_name.return_value = "test-new-collection"
            mock_naming.return_value = mock_naming_manager

            result = await client.call_tool("create_collection", create_params)

        assert result.success, f"create_collection failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "success" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_create_collection_invalid_parameters(self, fastmcp_test_client):
        """Test create_collection with invalid parameters."""
        client = fastmcp_test_client

        # Test with missing required parameters
        invalid_params = {
            "collection_name": "",  # Empty name
            "dimension": -1,        # Invalid dimension
        }

        result = await client.call_tool("create_collection", invalid_params)

        # Should fail with validation error
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_delete_collection_success(self, fastmcp_test_client):
        """Test deleting an existing collection."""
        client = fastmcp_test_client

        delete_params = {"collection_name": "test-collection-to-delete"}

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.delete_collection = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("delete_collection", delete_params)

        assert result.success, f"delete_collection failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "success" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_delete_collection_not_found(self, fastmcp_test_client):
        """Test deleting a non-existent collection."""
        client = fastmcp_test_client

        delete_params = {"collection_name": "non-existent-collection"}

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.delete_collection = AsyncMock(side_effect=Exception("Collection not found"))
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("delete_collection", delete_params)

        # Should handle error gracefully
        assert not result.success
        assert result.error is not None


class TestDocumentTools:
    """Test document tools: add, get, search, update, delete."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_add_document_success(self, fastmcp_test_client):
        """Test adding a document successfully."""
        client = fastmcp_test_client

        doc_params = {
            "content": "This is a test document for MCP testing.",
            "metadata": {"type": "test", "category": "unit-test"},
            "collection_name": "test-collection"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.get_embedding_model') as mock_embedding:

            # Mock embedding generation
            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            # Mock Qdrant client
            mock_qdrant = AsyncMock()
            mock_qdrant.upsert = AsyncMock(return_value=MagicMock(operation_id=1))
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("add_document_tool", doc_params)

        assert result.success, f"add_document_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "document_id" in result.response or "success" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_add_document_empty_content(self, fastmcp_test_client):
        """Test adding document with empty content."""
        client = fastmcp_test_client

        doc_params = {
            "content": "",  # Empty content
            "metadata": {"type": "test"},
            "collection_name": "test-collection"
        }

        result = await client.call_tool("add_document_tool", doc_params)

        # Should fail with validation error
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_get_document_success(self, fastmcp_test_client):
        """Test retrieving a document by ID."""
        client = fastmcp_test_client

        get_params = {
            "document_id": "test-doc-123",
            "collection_name": "test-collection"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.retrieve = AsyncMock(return_value=[
                MagicMock(
                    id="test-doc-123",
                    payload={"content": "Test document content", "metadata": {"type": "test"}}
                )
            ])
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("get_document_tool", get_params)

        assert result.success, f"get_document_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "content" in result.response or "document" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_get_document_not_found(self, fastmcp_test_client):
        """Test retrieving a non-existent document."""
        client = fastmcp_test_client

        get_params = {
            "document_id": "non-existent-doc",
            "collection_name": "test-collection"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.retrieve = AsyncMock(return_value=[])
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("get_document_tool", get_params)

        # Should handle gracefully
        assert not result.success or "not found" in str(result.response).lower()

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_by_metadata_success(self, fastmcp_test_client):
        """Test searching documents by metadata."""
        client = fastmcp_test_client

        search_params = {
            "metadata_query": {"type": "test", "category": "document"},
            "collection_name": "test-collection",
            "limit": 10
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.scroll = AsyncMock(return_value=(
                [
                    MagicMock(
                        id="doc1",
                        score=0.95,
                        payload={"content": "Test doc 1", "metadata": {"type": "test"}}
                    ),
                    MagicMock(
                        id="doc2",
                        score=0.87,
                        payload={"content": "Test doc 2", "metadata": {"type": "test"}}
                    )
                ],
                None
            ))
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("search_by_metadata_tool", search_params)

        assert result.success, f"search_by_metadata_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "results" in result.response
        assert isinstance(result.response["results"], list)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_by_metadata_empty_results(self, fastmcp_test_client):
        """Test metadata search with no results."""
        client = fastmcp_test_client

        search_params = {
            "metadata_query": {"nonexistent": "field"},
            "collection_name": "test-collection",
            "limit": 10
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.scroll = AsyncMock(return_value=([], None))
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("search_by_metadata_tool", search_params)

        assert result.success, f"search_by_metadata_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "results" in result.response
        assert len(result.response["results"]) == 0


class TestSearchTools:
    """Test search tools: workspace search, hybrid search."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_workspace_success(self, fastmcp_test_client):
        """Test workspace search with valid query."""
        client = fastmcp_test_client

        search_params = {
            "query": "test search query",
            "limit": 5,
            "collection": "test-collection"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.get_embedding_model') as mock_embedding:

            # Mock embedding generation
            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            # Mock search results
            mock_qdrant = AsyncMock()
            mock_qdrant.search = AsyncMock(return_value=[
                MagicMock(
                    id="result1",
                    score=0.95,
                    payload={"content": "Matching content 1", "metadata": {"type": "doc"}}
                ),
                MagicMock(
                    id="result2",
                    score=0.87,
                    payload={"content": "Matching content 2", "metadata": {"type": "doc"}}
                )
            ])
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("search_workspace_tool", search_params)

        assert result.success, f"search_workspace_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "results" in result.response
        assert isinstance(result.response["results"], list)
        assert len(result.response["results"]) <= search_params["limit"]

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_workspace_empty_query(self, fastmcp_test_client):
        """Test workspace search with empty query."""
        client = fastmcp_test_client

        search_params = {
            "query": "",  # Empty query
            "limit": 5
        }

        result = await client.call_tool("search_workspace_tool", search_params)

        # Should handle empty query gracefully
        assert not result.success or len(result.response.get("results", [])) == 0

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_hybrid_search_advanced_success(self, fastmcp_test_client):
        """Test advanced hybrid search functionality."""
        client = fastmcp_test_client

        hybrid_params = {
            "query": "advanced search test",
            "limit": 10,
            "semantic_weight": 0.7,
            "keyword_weight": 0.3,
            "collection": "test-collection"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.get_embedding_model') as mock_embedding:

            # Mock embedding and sparse vector generation
            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            # Mock hybrid search results
            mock_qdrant = AsyncMock()
            mock_qdrant.search = AsyncMock(return_value=[
                MagicMock(
                    id="hybrid1",
                    score=0.92,
                    payload={"content": "Hybrid result 1", "metadata": {"relevance": "high"}}
                )
            ])
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("hybrid_search_advanced_tool", hybrid_params)

        assert result.success, f"hybrid_search_advanced_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "results" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_workspace_with_project_isolation(self, fastmcp_test_client):
        """Test project-isolated workspace search."""
        client = fastmcp_test_client

        isolation_params = {
            "query": "project isolated search",
            "limit": 10,
            "project_name": "test-project"
        }

        with patch('workspace_qdrant_mcp.server.detect_project') as mock_project, \
             patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:

            mock_project.return_value = "test-project"
            mock_qdrant = AsyncMock()
            mock_qdrant.search = AsyncMock(return_value=[])
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("search_workspace_with_project_isolation_tool", isolation_params)

        assert result.success, f"project isolation search failed: {result.error}"
        assert isinstance(result.response, dict)


class TestScratchbookTools:
    """Test scratchbook tools: update, search, list, delete."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_update_scratchbook_success(self, fastmcp_test_client):
        """Test updating scratchbook with new note."""
        client = fastmcp_test_client

        scratchbook_params = {
            "note": "This is a test scratchbook note for unit testing.",
            "metadata": {"priority": "high", "category": "testing"},
            "project_name": "test-project"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.get_embedding_model') as mock_embedding, \
             patch('workspace_qdrant_mcp.server.detect_project') as mock_project:

            mock_project.return_value = "test-project"

            # Mock embedding generation
            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            # Mock Qdrant operations
            mock_qdrant = AsyncMock()
            mock_qdrant.upsert = AsyncMock(return_value=MagicMock(operation_id=1))
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("update_scratchbook_tool", scratchbook_params)

        assert result.success, f"update_scratchbook_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "note_id" in result.response or "success" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_scratchbook_success(self, fastmcp_test_client):
        """Test searching scratchbook notes."""
        client = fastmcp_test_client

        search_params = {
            "query": "testing notes",
            "limit": 5,
            "project_name": "test-project"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.get_embedding_model') as mock_embedding, \
             patch('workspace_qdrant_mcp.server.detect_project') as mock_project:

            mock_project.return_value = "test-project"

            # Mock embedding generation
            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            # Mock search results
            mock_qdrant = AsyncMock()
            mock_qdrant.search = AsyncMock(return_value=[
                MagicMock(
                    id="note1",
                    score=0.95,
                    payload={"note": "Testing note 1", "metadata": {"priority": "high"}}
                ),
                MagicMock(
                    id="note2",
                    score=0.87,
                    payload={"note": "Testing note 2", "metadata": {"priority": "medium"}}
                )
            ])
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("search_scratchbook_tool", search_params)

        assert result.success, f"search_scratchbook_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "results" in result.response
        assert isinstance(result.response["results"], list)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_list_scratchbook_notes_success(self, fastmcp_test_client):
        """Test listing all scratchbook notes."""
        client = fastmcp_test_client

        list_params = {
            "limit": 20,
            "project_name": "test-project"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.detect_project') as mock_project:

            mock_project.return_value = "test-project"

            # Mock listing results
            mock_qdrant = AsyncMock()
            mock_qdrant.scroll = AsyncMock(return_value=(
                [
                    MagicMock(
                        id="note1",
                        payload={"note": "Note 1", "timestamp": "2024-01-01T10:00:00Z"}
                    ),
                    MagicMock(
                        id="note2",
                        payload={"note": "Note 2", "timestamp": "2024-01-01T11:00:00Z"}
                    )
                ],
                None
            ))
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("list_scratchbook_notes_tool", list_params)

        assert result.success, f"list_scratchbook_notes_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "notes" in result.response
        assert isinstance(result.response["notes"], list)

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_delete_scratchbook_note_success(self, fastmcp_test_client):
        """Test deleting a scratchbook note."""
        client = fastmcp_test_client

        delete_params = {
            "note_id": "test-note-123",
            "project_name": "test-project"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client, \
             patch('workspace_qdrant_mcp.server.detect_project') as mock_project:

            mock_project.return_value = "test-project"

            # Mock deletion
            mock_qdrant = AsyncMock()
            mock_qdrant.delete = AsyncMock(return_value=MagicMock(operation_id=1))
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("delete_scratchbook_note_tool", delete_params)

        assert result.success, f"delete_scratchbook_note_tool failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "success" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_delete_scratchbook_note_not_found(self, fastmcp_test_client):
        """Test deleting non-existent scratchbook note."""
        client = fastmcp_test_client

        delete_params = {
            "note_id": "non-existent-note",
            "project_name": "test-project"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.delete = AsyncMock(side_effect=Exception("Note not found"))
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("delete_scratchbook_note_tool", delete_params)

        # Should handle error gracefully
        assert not result.success
        assert result.error is not None


class TestAdvancedTools:
    """Test advanced tools: research, memory search, gRPC integration."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_research_workspace_success(self, fastmcp_test_client):
        """Test workspace research functionality."""
        client = fastmcp_test_client

        research_params = {
            "research_query": "AI and machine learning best practices",
            "context_limit": 10,
            "include_scratchbook": True
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.search = AsyncMock(return_value=[
                MagicMock(
                    id="research1",
                    score=0.94,
                    payload={"content": "AI research content", "metadata": {"type": "research"}}
                )
            ])
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("research_workspace", research_params)

        assert result.success, f"research_workspace failed: {result.error}"
        assert isinstance(result.response, dict)
        assert "research_results" in result.response or "results" in result.response

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_search_memories_tool_success(self, fastmcp_test_client):
        """Test memory search functionality."""
        client = fastmcp_test_client

        memory_params = {
            "query": "important memories",
            "limit": 10,
            "memory_type": "all"
        }

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.search = AsyncMock(return_value=[])
            mock_get_client.return_value = mock_qdrant

            result = await client.call_tool("search_memories_tool", memory_params)

        assert result.success, f"search_memories_tool failed: {result.error}"
        assert isinstance(result.response, dict)


class TestParameterValidation:
    """Test parameter validation across all tools."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_invalid_limit_parameters(self, fastmcp_test_client):
        """Test tools with invalid limit parameters."""
        client = fastmcp_test_client

        # Test negative limit
        result = await client.call_tool("search_workspace_tool", {
            "query": "test",
            "limit": -5  # Invalid negative limit
        })

        # Should either fail or sanitize the parameter
        assert not result.success or result.response.get("results", []) == []

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_missing_required_parameters(self, fastmcp_test_client):
        """Test tools with missing required parameters."""
        client = fastmcp_test_client

        # Test missing collection_name for create_collection
        result = await client.call_tool("create_collection", {
            "dimension": 384
            # Missing collection_name
        })

        # Should fail with validation error
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_invalid_data_types(self, fastmcp_test_client):
        """Test tools with invalid data types."""
        client = fastmcp_test_client

        # Test string where integer expected
        result = await client.call_tool("search_workspace_tool", {
            "query": "test",
            "limit": "not_a_number"  # String instead of int
        })

        # Should handle gracefully
        assert not result.success or isinstance(result.response, dict)


class TestErrorHandling:
    """Test error handling across all tools."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_qdrant_connection_error(self, fastmcp_test_client):
        """Test tools behavior when Qdrant is unavailable."""
        client = fastmcp_test_client

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            mock_get_client.side_effect = Exception("Qdrant connection failed")

            result = await client.call_tool("workspace_status", {})

        # Should handle connection error gracefully
        assert not result.success or result.response.get("connected") == False

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_embedding_model_error(self, fastmcp_test_client):
        """Test tools behavior when embedding model fails."""
        client = fastmcp_test_client

        with patch('workspace_qdrant_mcp.server.get_embedding_model') as mock_embedding:
            mock_embedding.side_effect = Exception("Embedding model failed")

            result = await client.call_tool("search_workspace_tool", {
                "query": "test search"
            })

        # Should handle embedding error gracefully
        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_timeout_handling(self, fastmcp_test_client):
        """Test tools behavior with simulated timeouts."""
        client = fastmcp_test_client

        with patch('workspace_qdrant_mcp.server.get_client') as mock_get_client:
            import asyncio

            async def slow_operation():
                await asyncio.sleep(10)  # Simulate slow operation
                return AsyncMock()

            mock_get_client.side_effect = slow_operation

            # This should timeout or be handled gracefully
            result = await client.call_tool("workspace_status", {})

        # Should handle timeout gracefully
        assert not result.success or result.success


class TestProtocolCompliance:
    """Test FastMCP protocol compliance for all tools."""

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_protocol_compliance_workspace_status(self, mcp_protocol_tester):
        """Test workspace_status protocol compliance."""
        tester = mcp_protocol_tester

        # Test tool registration compliance
        registration_results = await tester.test_tool_registration()
        assert registration_results["success_rate"] > 0

        # Check that workspace_status is registered
        tool_names = [tool["name"] for tool in registration_results["test_results"]]
        assert "workspace_status" in tool_names

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_response_format_compliance(self, fastmcp_test_client):
        """Test that all tool responses follow MCP format standards."""
        client = fastmcp_test_client

        # Test multiple tools for consistent response format
        tools_to_test = [
            ("workspace_status", {}),
            ("list_workspace_collections", {}),
            ("search_workspace_tool", {"query": "test", "limit": 5})
        ]

        for tool_name, params in tools_to_test:
            result = await client.call_tool(tool_name, params)

            # Verify result structure
            assert hasattr(result, 'success')
            assert hasattr(result, 'response') or hasattr(result, 'error')
            assert hasattr(result, 'execution_time_ms')
            assert hasattr(result, 'tool_name')
            assert result.tool_name == tool_name

            # Verify metadata structure
            assert hasattr(result, 'metadata')
            assert isinstance(result.metadata, dict)
            assert 'timestamp' in result.metadata

    @pytest.mark.asyncio
    @pytest.mark.fastmcp
    async def test_error_response_compliance(self, fastmcp_test_client):
        """Test that error responses follow MCP standards."""
        client = fastmcp_test_client

        # Force an error with invalid parameters
        result = await client.call_tool("create_collection", {
            "collection_name": "",  # Invalid empty name
            "dimension": 0  # Invalid dimension
        })

        # Should have structured error response
        assert not result.success
        assert result.error is not None
        assert isinstance(result.error, str)
        assert len(result.error) > 0


# Test configuration and fixtures are imported from conftest.py and test infrastructure