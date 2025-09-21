"""
Tests for readonly collection access control enforcement in MCP tools.

This module tests that MCP tools properly enforce access control for readonly collections
(collections with '_' prefix) and prevent write operations while allowing read operations.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock

from common.core.collection_naming import CollectionPermissionError
from workspace_qdrant_mcp.tools.documents import add_document, delete_document
from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager


class TestDocumentsAccessControl:
    """Test access control enforcement in documents.py tools."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client with collection manager."""
        client = Mock()
        client.initialized = True
        client.collection_manager = Mock()
        return client

    @pytest.mark.asyncio
    async def test_add_document_blocks_readonly_collections(self, mock_client):
        """Test that add_document blocks write operations on readonly collections."""
        # Configure mock to raise permission error for readonly collection
        mock_client.collection_manager.validate_mcp_write_access.side_effect = (
            CollectionPermissionError("Library collection '_test' is readonly from MCP server")
        )
        mock_client.list_collections.return_value = ["_test"]
        
        result = await add_document(
            client=mock_client,
            content="test content",
            collection="_test",
            metadata={"test": "value"}
        )
        
        # Should return error instead of proceeding with write
        assert "error" in result
        assert "readonly from MCP server" in result["error"]
        mock_client.collection_manager.validate_mcp_write_access.assert_called_once_with("_test")

    @pytest.mark.asyncio
    async def test_add_document_allows_writable_collections(self, mock_client):
        """Test that add_document allows write operations on writable collections."""
        # Configure mock to allow writes (no exception)
        mock_client.collection_manager.validate_mcp_write_access.return_value = None
        mock_client.collection_manager.resolve_collection_name.return_value = ("test-docs", False)
        mock_client.list_collections.return_value = ["test-docs"]
        
        # Mock the embedding service and other dependencies
        mock_embedding_service = Mock()
        mock_embedding_service.config.embedding.chunk_size = 1000
        mock_embedding_service.embed_text_async.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.5, 0.3]}
        }
        mock_client.get_embedding_service.return_value = mock_embedding_service
        mock_client.client.upsert.return_value = None
        
        with patch('workspace_qdrant_mcp.tools.documents.uuid.uuid4', return_value=Mock()), \
             patch('workspace_qdrant_mcp.tools.documents.create_qdrant_sparse_vector'), \
             patch('workspace_qdrant_mcp.tools.documents.models.PointStruct'):
            
            result = await add_document(
                client=mock_client,
                content="test content",
                collection="test-docs",
                metadata={"test": "value"}
            )
        
        # Should succeed for writable collections
        mock_client.collection_manager.validate_mcp_write_access.assert_called_once_with("test-docs")
        # Note: We can't easily test full success without mocking more components

    @pytest.mark.asyncio
    async def test_delete_document_blocks_readonly_collections(self, mock_client):
        """Test that delete_document blocks delete operations on readonly collections."""
        # Configure mock to raise permission error for readonly collection
        mock_client.collection_manager.validate_mcp_write_access.side_effect = (
            CollectionPermissionError("Library collection '_test' is readonly from MCP server")
        )
        mock_client.list_collections.return_value = ["_test"]
        
        result = await delete_document(
            client=mock_client,
            document_id="test-doc",
            collection="_test"
        )
        
        # Should return error instead of proceeding with delete
        assert "error" in result
        assert "readonly from MCP server" in result["error"]
        mock_client.collection_manager.validate_mcp_write_access.assert_called_once_with("_test")

    @pytest.mark.asyncio
    async def test_delete_document_allows_writable_collections(self, mock_client):
        """Test that delete_document allows delete operations on writable collections."""
        # Configure mock to allow writes (no exception)
        mock_client.collection_manager.validate_mcp_write_access.return_value = None
        mock_client.collection_manager.resolve_collection_name.return_value = ("test-docs", False)
        mock_client.list_collections.return_value = ["test-docs"]
        
        # Mock successful delete operation
        mock_result = Mock()
        mock_result.operation_id = "test-op-id"
        mock_client.client.delete.return_value = mock_result
        
        with patch('workspace_qdrant_mcp.tools.documents.models.FilterSelector'), \
             patch('workspace_qdrant_mcp.tools.documents.models.Filter'), \
             patch('workspace_qdrant_mcp.tools.documents.models.FieldCondition'), \
             patch('workspace_qdrant_mcp.tools.documents.models.MatchValue'):
            
            result = await delete_document(
                client=mock_client,
                document_id="test-doc",
                collection="test-docs"
            )
        
        # Should succeed for writable collections
        mock_client.collection_manager.validate_mcp_write_access.assert_called_once_with("test-docs")
        assert result["status"] == "success"


class TestScratchbookAccessControl:
    """Test access control enforcement in scratchbook tools."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock client with collection manager."""
        client = Mock()
        client.initialized = True
        client.collection_manager = Mock()
        return client

    @pytest.fixture
    def scratchbook_manager(self, mock_client):
        """Create a scratchbook manager with mock client."""
        with patch('workspace_qdrant_mcp.tools.scratchbook.QdrantWorkspaceClient'):
            manager = ScratchbookManager(mock_client)
            return manager

    @pytest.mark.asyncio
    async def test_add_note_blocks_readonly_collections(self, scratchbook_manager, mock_client):
        """Test that add_note blocks write operations on readonly collections."""
        # Configure mock to raise permission error for readonly collection
        mock_client.collection_manager.validate_mcp_write_access.side_effect = (
            CollectionPermissionError("Library collection '_notes' is readonly from MCP server")
        )
        
        with patch.object(scratchbook_manager, '_get_scratchbook_collection_name', return_value="_notes"):
            result = await scratchbook_manager.add_note(
                content="test note content",
                title="Test Note"
            )
        
        # Should return error instead of proceeding with write
        assert "error" in result
        assert "readonly from MCP server" in result["error"]
        mock_client.collection_manager.validate_mcp_write_access.assert_called_once_with("_notes")

    @pytest.mark.asyncio
    async def test_add_note_allows_writable_collections(self, scratchbook_manager, mock_client):
        """Test that add_note allows write operations on writable collections."""
        # Configure mock to allow writes (no exception)
        mock_client.collection_manager.validate_mcp_write_access.return_value = None
        mock_client.ensure_collection_exists = AsyncMock()
        
        # Mock embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.embed_text_async.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.5, 0.3]}
        }
        mock_client.get_embedding_service.return_value = mock_embedding_service
        mock_client.client.upsert.return_value = None
        
        with patch.object(scratchbook_manager, '_get_scratchbook_collection_name', return_value="test-scratchbook"), \
             patch('workspace_qdrant_mcp.tools.scratchbook.uuid.uuid4', return_value=Mock()), \
             patch('workspace_qdrant_mcp.tools.scratchbook.create_qdrant_sparse_vector'), \
             patch('workspace_qdrant_mcp.tools.scratchbook.models.PointStruct'):
            
            result = await scratchbook_manager.add_note(
                content="test note content",
                title="Test Note"
            )
        
        # Should succeed for writable collections
        mock_client.collection_manager.validate_mcp_write_access.assert_called_once_with("test-scratchbook")

    @pytest.mark.asyncio
    async def test_update_note_blocks_readonly_collections(self, scratchbook_manager, mock_client):
        """Test that update_note blocks write operations on readonly collections."""
        # Configure mock to raise permission error for readonly collection
        mock_client.collection_manager.validate_mcp_write_access.side_effect = (
            CollectionPermissionError("Library collection '_notes' is readonly from MCP server")
        )
        
        with patch.object(scratchbook_manager, '_get_scratchbook_collection_name', return_value="_notes"):
            result = await scratchbook_manager.update_note(
                note_id="test-note-id",
                content="updated content"
            )
        
        # Should return error instead of proceeding with write
        assert "error" in result
        assert "readonly from MCP server" in result["error"]
        mock_client.collection_manager.validate_mcp_write_access.assert_called_once_with("_notes")

    @pytest.mark.asyncio
    async def test_update_note_allows_writable_collections(self, scratchbook_manager, mock_client):
        """Test that update_note allows write operations on writable collections."""
        # Configure mock to allow writes (no exception)
        mock_client.collection_manager.validate_mcp_write_access.return_value = None
        
        # Mock existing note lookup
        mock_existing_point = Mock()
        mock_existing_point.payload = {
            "content": "original content",
            "title": "Original Title",
            "version": 1,
            "created_at": "2024-01-01T00:00:00Z"
        }
        mock_client.client.scroll.return_value = ([[mock_existing_point]], None)
        
        # Mock embedding service
        mock_embedding_service = Mock()
        mock_embedding_service.embed_text_async.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.5, 0.3]}
        }
        mock_client.get_embedding_service.return_value = mock_embedding_service
        mock_client.client.upsert.return_value = None
        
        with patch.object(scratchbook_manager, '_get_scratchbook_collection_name', return_value="test-scratchbook"), \
             patch('workspace_qdrant_mcp.tools.scratchbook.create_qdrant_sparse_vector'), \
             patch('workspace_qdrant_mcp.tools.scratchbook.models.Filter'), \
             patch('workspace_qdrant_mcp.tools.scratchbook.models.FieldCondition'), \
             patch('workspace_qdrant_mcp.tools.scratchbook.models.MatchValue'), \
             patch('workspace_qdrant_mcp.tools.scratchbook.models.PointStruct'):
            
            result = await scratchbook_manager.update_note(
                note_id="test-note-id",
                content="updated content"
            )
        
        # Should succeed for writable collections
        mock_client.collection_manager.validate_mcp_write_access.assert_called_once_with("test-scratchbook")


class TestCollectionManagerAccessControl:
    """Test access control enforcement in collection manager."""

    def test_validate_mcp_write_access_blocks_library_collections(self):
        """Test that validate_mcp_write_access blocks library collections."""
        from common.core.collections import WorkspaceCollectionManager
        from common.core.collection_naming import CollectionType, CollectionNameInfo
        
        # Create a mock collection manager
        mock_config = Mock()
        with patch.object(WorkspaceCollectionManager, '__init__', return_value=None):
            manager = WorkspaceCollectionManager.__new__(WorkspaceCollectionManager)
        
        # Mock the naming manager to return a library collection
        mock_info = CollectionNameInfo(
            name="_test",
            display_name="test",
            collection_type=CollectionType.LIBRARY,
            is_readonly_from_mcp=True,
            library_name="test"
        )
        manager.naming_manager = Mock()
        manager.naming_manager.get_collection_info.return_value = mock_info
        
        # Mock resolve_collection_name to return readonly flag
        with patch.object(manager, 'resolve_collection_name', return_value=("_test", True)):
            with pytest.raises(CollectionPermissionError) as exc_info:
                manager.validate_mcp_write_access("test")
            
            assert "readonly from MCP server" in str(exc_info.value)
            assert "CLI/Rust engine" in str(exc_info.value)

    def test_validate_mcp_write_access_allows_writable_collections(self):
        """Test that validate_mcp_write_access allows writable collections."""
        from common.core.collections import WorkspaceCollectionManager
        
        # Create a mock collection manager
        mock_config = Mock()
        with patch.object(WorkspaceCollectionManager, '__init__', return_value=None):
            manager = WorkspaceCollectionManager.__new__(WorkspaceCollectionManager)
        
        # Mock resolve_collection_name to return writable flag
        with patch.object(manager, 'resolve_collection_name', return_value=("test-docs", False)):
            # Should not raise any exception
            manager.validate_mcp_write_access("test-docs")


if __name__ == "__main__":
    pytest.main([__file__])