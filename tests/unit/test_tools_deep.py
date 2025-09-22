"""
Deep coverage tests for tools modules focusing on method-level execution.

This test suite targets memory tools, search tools, and other MCP tools
to achieve comprehensive method-level coverage.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call

import pytest

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

# Test imports
try:
    from workspace_qdrant_mcp.tools.memory import (
        add_document, get_document, search_workspace,
        update_scratchbook, search_scratchbook
    )
    from workspace_qdrant_mcp.tools.state_management import (
        get_server_info, workspace_status,
        list_collections, create_workspace_collection
    )
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    print(f"Tools import failed: {e}")

pytestmark = pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools modules not available")


class TestMemoryToolsDeep:
    """Deep coverage tests for memory tools."""

    @pytest.fixture
    def mock_client(self):
        """Mock workspace client."""
        client = AsyncMock()
        client.is_initialized = True
        client.collection_manager = AsyncMock()
        client.embedding_service = AsyncMock()
        return client

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = MagicMock()
        config.workspace.global_collections = ["scratchbook"]
        config.workspace.project_collections = ["notes", "docs"]
        return config

    @pytest.mark.asyncio
    async def test_add_document_success(self, mock_client, mock_config):
        """Test successful document addition."""
        mock_client.collection_manager.upsert_document.return_value = True
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await add_document(
                    content="test content",
                    collection_name="test-collection",
                    document_id="doc1",
                    metadata={"type": "test"}
                )

                assert result["success"] is True
                assert result["document_id"] == "doc1"
                mock_client.collection_manager.upsert_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_without_id(self, mock_client, mock_config):
        """Test document addition without explicit ID."""
        mock_client.collection_manager.upsert_document.return_value = True
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await add_document(
                    content="test content",
                    collection_name="test-collection"
                )

                assert result["success"] is True
                assert "document_id" in result
                # Should generate a UUID
                assert len(result["document_id"]) > 10

    @pytest.mark.asyncio
    async def test_add_document_client_not_initialized(self, mock_config):
        """Test document addition when client is not initialized."""
        mock_client = AsyncMock()
        mock_client.is_initialized = False

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await add_document(
                    content="test content",
                    collection_name="test-collection"
                )

                assert result["success"] is False
                assert "not initialized" in result["error"]

    @pytest.mark.asyncio
    async def test_add_document_error_handling(self, mock_client, mock_config):
        """Test document addition error handling."""
        mock_client.collection_manager.upsert_document.side_effect = Exception("Upsert failed")
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await add_document(
                    content="test content",
                    collection_name="test-collection"
                )

                assert result["success"] is False
                assert "Upsert failed" in result["error"]

    @pytest.mark.asyncio
    async def test_get_document_success(self, mock_client, mock_config):
        """Test successful document retrieval."""
        mock_document = {
            "id": "doc1",
            "content": "test content",
            "metadata": {"type": "test"}
        }
        mock_client.collection_manager.get_document.return_value = mock_document

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await get_document(
                    document_id="doc1",
                    collection_name="test-collection"
                )

                assert result["success"] is True
                assert result["document"]["id"] == "doc1"
                assert result["document"]["content"] == "test content"

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mock_client, mock_config):
        """Test document retrieval when document not found."""
        mock_client.collection_manager.get_document.return_value = None

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await get_document(
                    document_id="nonexistent",
                    collection_name="test-collection"
                )

                assert result["success"] is False
                assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_search_workspace_success(self, mock_client, mock_config):
        """Test successful workspace search."""
        mock_results = [
            MagicMock(id="doc1", score=0.9, payload={"content": "test1"}),
            MagicMock(id="doc2", score=0.8, payload={"content": "test2"})
        ]
        mock_client.collection_manager.search_documents.return_value = mock_results
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await search_workspace(
                    query="test query",
                    collection_name="test-collection",
                    limit=10
                )

                assert result["success"] is True
                assert len(result["results"]) == 2
                assert result["results"][0]["id"] == "doc1"

    @pytest.mark.asyncio
    async def test_search_workspace_empty_query(self, mock_client, mock_config):
        """Test workspace search with empty query."""
        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await search_workspace(
                    query="",
                    collection_name="test-collection"
                )

                assert result["success"] is False
                assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_workspace_no_results(self, mock_client, mock_config):
        """Test workspace search with no results."""
        mock_client.collection_manager.search_documents.return_value = []
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await search_workspace(
                    query="nonexistent query",
                    collection_name="test-collection"
                )

                assert result["success"] is True
                assert len(result["results"]) == 0

    @pytest.mark.asyncio
    async def test_update_scratchbook_success(self, mock_client, mock_config):
        """Test successful scratchbook update."""
        mock_client.collection_manager.upsert_document.return_value = True
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await update_scratchbook(
                    content="scratchbook content",
                    note_id="note1",
                    tags=["tag1", "tag2"]
                )

                assert result["success"] is True
                assert result["note_id"] == "note1"

    @pytest.mark.asyncio
    async def test_update_scratchbook_without_id(self, mock_client, mock_config):
        """Test scratchbook update without note ID."""
        mock_client.collection_manager.upsert_document.return_value = True
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await update_scratchbook(
                    content="scratchbook content"
                )

                assert result["success"] is True
                assert "note_id" in result
                # Should generate a UUID
                assert len(result["note_id"]) > 10

    @pytest.mark.asyncio
    async def test_search_scratchbook_success(self, mock_client, mock_config):
        """Test successful scratchbook search."""
        mock_results = [
            MagicMock(id="note1", score=0.9, payload={"content": "note1 content"}),
            MagicMock(id="note2", score=0.8, payload={"content": "note2 content"})
        ]
        mock_client.collection_manager.search_documents.return_value = mock_results
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await search_scratchbook(
                    query="test query",
                    limit=5
                )

                assert result["success"] is True
                assert len(result["notes"]) == 2
                assert result["notes"][0]["id"] == "note1"

    @pytest.mark.asyncio
    async def test_search_scratchbook_by_tags(self, mock_client, mock_config):
        """Test scratchbook search by tags."""
        mock_results = [MagicMock(id="note1", score=0.9, payload={"content": "tagged note"})]
        mock_client.collection_manager.search_documents.return_value = mock_results
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                result = await search_scratchbook(
                    query="test query",
                    tags=["important", "work"]
                )

                assert result["success"] is True
                # Should apply tag filtering
                mock_client.collection_manager.search_documents.assert_called_once()


class TestStateManagementToolsDeep:
    """Deep coverage tests for state management tools."""

    @pytest.fixture
    def mock_client(self):
        """Mock workspace client."""
        client = AsyncMock()
        client.is_initialized = True
        client.list_collections.return_value = ["collection1", "collection2"]
        client.get_status.return_value = {
            "client_initialized": True,
            "collections_count": 2,
            "qdrant_url": "http://localhost:6333"
        }
        return client

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = MagicMock()
        config.version = "1.0.0"
        config.workspace.global_collections = ["scratchbook"]
        config.workspace.project_collections = ["notes", "docs"]
        return config

    @pytest.mark.asyncio
    async def test_get_server_info_success(self, mock_client, mock_config):
        """Test successful server info retrieval."""
        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await get_server_info()

                assert result["success"] is True
                assert result["server_info"]["version"] == "1.0.0"
                assert "workspace_qdrant_mcp" in result["server_info"]["name"]

    @pytest.mark.asyncio
    async def test_get_server_info_client_not_initialized(self, mock_config):
        """Test server info when client is not initialized."""
        mock_client = AsyncMock()
        mock_client.is_initialized = False

        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await get_server_info()

                assert result["success"] is True
                assert result["server_info"]["client_status"] == "not_initialized"

    @pytest.mark.asyncio
    async def test_workspace_status_success(self, mock_client, mock_config):
        """Test successful workspace status retrieval."""
        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await workspace_status()

                assert result["success"] is True
                assert result["status"]["client_initialized"] is True
                assert result["status"]["collections_count"] == 2

    @pytest.mark.asyncio
    async def test_workspace_status_error_handling(self, mock_config):
        """Test workspace status error handling."""
        mock_client = AsyncMock()
        mock_client.get_status.side_effect = Exception("Status retrieval failed")

        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await workspace_status()

                assert result["success"] is False
                assert "Status retrieval failed" in result["error"]

    @pytest.mark.asyncio
    async def test_list_collections_success(self, mock_client, mock_config):
        """Test successful collection listing."""
        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await list_collections()

                assert result["success"] is True
                assert len(result["collections"]) == 2
                assert "collection1" in result["collections"]
                assert "collection2" in result["collections"]

    @pytest.mark.asyncio
    async def test_list_collections_empty(self, mock_client, mock_config):
        """Test collection listing when no collections exist."""
        mock_client.list_collections.return_value = []

        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await list_collections()

                assert result["success"] is True
                assert len(result["collections"]) == 0

    @pytest.mark.asyncio
    async def test_create_workspace_collection_success(self, mock_client, mock_config):
        """Test successful workspace collection creation."""
        mock_client.create_collection.return_value = True

        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await create_workspace_collection(
                    collection_name="new-collection",
                    description="Test collection"
                )

                assert result["success"] is True
                assert result["collection_name"] == "new-collection"
                mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_workspace_collection_already_exists(self, mock_client, mock_config):
        """Test workspace collection creation when collection already exists."""
        mock_client.create_collection.return_value = False
        mock_client.collection_exists.return_value = True

        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await create_workspace_collection(
                    collection_name="existing-collection"
                )

                assert result["success"] is True
                assert "already exists" in result["message"]

    @pytest.mark.asyncio
    async def test_create_workspace_collection_error(self, mock_client, mock_config):
        """Test workspace collection creation error handling."""
        mock_client.create_collection.side_effect = Exception("Creation failed")

        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await create_workspace_collection(
                    collection_name="failed-collection"
                )

                assert result["success"] is False
                assert "Creation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_create_workspace_collection_invalid_name(self, mock_client, mock_config):
        """Test workspace collection creation with invalid name."""
        with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                result = await create_workspace_collection(
                    collection_name=""  # Empty name
                )

                assert result["success"] is False
                assert "invalid" in result["error"].lower()

    def test_tools_module_imports(self):
        """Test that all required tools can be imported."""
        # Test memory tools
        assert add_document is not None
        assert get_document is not None
        assert search_workspace is not None
        assert update_scratchbook is not None
        assert search_scratchbook is not None

        # Test state management tools
        assert get_server_info is not None
        assert workspace_status is not None
        assert list_collections is not None
        assert create_workspace_collection is not None

    @pytest.mark.asyncio
    async def test_concurrent_tool_operations(self, mock_client, mock_config):
        """Test concurrent tool operations."""
        mock_client.collection_manager.search_documents.return_value = []
        mock_client.embedding_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]

        with patch('workspace_qdrant_mcp.tools.memory.get_client', return_value=mock_client):
            with patch('workspace_qdrant_mcp.tools.memory.get_config', return_value=mock_config):
                with patch('workspace_qdrant_mcp.tools.state_management.get_client', return_value=mock_client):
                    with patch('workspace_qdrant_mcp.tools.state_management.get_config', return_value=mock_config):
                        # Run multiple operations concurrently
                        tasks = [
                            search_workspace("query1", "collection1"),
                            search_workspace("query2", "collection2"),
                            workspace_status(),
                            list_collections()
                        ]

                        results = await asyncio.gather(*tasks)
                        assert len(results) == 4
                        assert all(result["success"] for result in results)


if __name__ == "__main__":
    pytest.main([__file__])