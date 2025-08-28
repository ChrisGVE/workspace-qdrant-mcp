"""
Unit tests for document management tools.

Tests document add, update, delete, and get operations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime

from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from workspace_qdrant_mcp.tools.documents import (
    add_document,
    update_document,
    delete_document,
    get_document,
    _add_single_document
)
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient


class TestAddDocument:
    """Test add_document function."""
    
    @pytest.mark.asyncio
    async def test_add_document_client_not_initialized(self, mock_workspace_client):
        """Test add_document with uninitialized client."""
        mock_workspace_client.initialized = False
        
        result = await add_document(
            mock_workspace_client,
            "Test content",
            "test_collection"
        )
        
        assert result["error"] == "Workspace client not initialized"
    
    @pytest.mark.asyncio
    async def test_add_document_empty_content(self, mock_workspace_client):
        """Test add_document with empty content."""
        mock_workspace_client.initialized = True
        
        # Test empty string
        result = await add_document(
            mock_workspace_client,
            "",
            "test_collection"
        )
        assert result["error"] == "Content cannot be empty"
        
        # Test whitespace-only string
        result = await add_document(
            mock_workspace_client,
            "   ",
            "test_collection"
        )
        assert result["error"] == "Content cannot be empty"
    
    @pytest.mark.asyncio
    async def test_add_document_collection_not_found(self, mock_workspace_client):
        """Test add_document with non-existent collection."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["other_collection"]
        
        result = await add_document(
            mock_workspace_client,
            "Test content",
            "nonexistent_collection"
        )
        
        assert "error" in result
        assert "Collection 'nonexistent_collection' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_add_document_single_success(self, mock_workspace_client, mock_embedding_service):
        """Test successful single document addition."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        mock_workspace_client.get_embedding_service.return_value = mock_embedding_service
        
        # Mock embedding service
        mock_embedding_service.config.embedding.chunk_size = 2000
        
        # Mock _add_single_document
        with patch('workspace_qdrant_mcp.tools.documents._add_single_document', return_value=True) as mock_add_single:
            result = await add_document(
                mock_workspace_client,
                "Test content",
                "test_collection",
                metadata={"source": "test"},
                document_id="doc123",
                chunk_text=False
            )
        
        assert result["document_id"] == "doc123"
        assert result["collection"] == "test_collection"
        assert result["points_added"] == 1
        assert result["content_length"] == 12
        assert not result["chunked"]
        assert "source" in result["metadata"]
        mock_add_single.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_document_generate_id(self, mock_workspace_client, mock_embedding_service):
        """Test document addition with auto-generated ID."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        mock_workspace_client.get_embedding_service.return_value = mock_embedding_service
        
        mock_embedding_service.config.embedding.chunk_size = 2000
        
        with patch('workspace_qdrant_mcp.tools.documents._add_single_document', return_value=True):
            result = await add_document(
                mock_workspace_client,
                "Test content",
                "test_collection"
            )
        
        # Should have generated a UUID
        assert "document_id" in result
        assert len(result["document_id"]) == 36  # UUID length
    
    @pytest.mark.asyncio
    async def test_add_document_chunked_success(self, mock_workspace_client, mock_embedding_service):
        """Test successful chunked document addition."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        mock_workspace_client.get_embedding_service.return_value = mock_embedding_service
        
        # Configure for chunking
        mock_embedding_service.config.embedding.chunk_size = 10  # Small chunk size
        mock_embedding_service.chunk_text.return_value = ["chunk1", "chunk2", "chunk3"]
        
        with patch('workspace_qdrant_mcp.tools.documents._add_single_document', return_value=True) as mock_add_single:
            result = await add_document(
                mock_workspace_client,
                "This is a long document that will be chunked",
                "test_collection",
                document_id="doc123",
                chunk_text=True
            )
        
        assert result["document_id"] == "doc123"
        assert result["points_added"] == 3
        assert result["chunked"] is True
        assert mock_add_single.call_count == 3
        
        # Verify chunk metadata
        calls = mock_add_single.call_args_list
        for i, call in enumerate(calls):
            chunk_metadata = call[0][3]  # Fourth argument is metadata
            assert chunk_metadata["chunk_index"] == i
            assert chunk_metadata["chunk_count"] == 3
            assert chunk_metadata["is_chunk"] is True
    
    @pytest.mark.asyncio
    async def test_add_document_chunked_partial_failure(self, mock_workspace_client, mock_embedding_service):
        """Test chunked addition with partial failures."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        mock_workspace_client.get_embedding_service.return_value = mock_embedding_service
        
        mock_embedding_service.config.embedding.chunk_size = 10
        mock_embedding_service.chunk_text.return_value = ["chunk1", "chunk2", "chunk3"]
        
        # Mock partial failure: first two succeed, third fails
        with patch('workspace_qdrant_mcp.tools.documents._add_single_document', side_effect=[True, True, False]):
            result = await add_document(
                mock_workspace_client,
                "Long document content",
                "test_collection",
                chunk_text=True
            )
        
        assert result["points_added"] == 2  # Only 2 out of 3 chunks succeeded
    
    @pytest.mark.asyncio
    async def test_add_document_exception_handling(self, mock_workspace_client):
        """Test add_document exception handling."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.side_effect = Exception("Connection error")
        
        result = await add_document(
            mock_workspace_client,
            "Test content",
            "test_collection"
        )
        
        assert "error" in result
        assert "Failed to add document" in result["error"]
        assert "Connection error" in result["error"]


class TestAddSingleDocument:
    """Test _add_single_document helper function."""
    
    @pytest.mark.asyncio
    async def test_add_single_document_success(self, mock_workspace_client, mock_embedding_service):
        """Test successful single document addition."""
        mock_workspace_client.get_embedding_service.return_value = mock_embedding_service
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.8, 0.6]}
        }
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        
        result = await _add_single_document(
            mock_workspace_client,
            "Test content",
            "test_collection",
            {"source": "test"},
            "doc123"
        )
        
        assert result is True
        mock_client.upsert.assert_called_once()
        
        # Verify upsert was called with correct parameters
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        assert len(call_args[1]["points"]) == 1
        
        point = call_args[1]["points"][0]
        assert point.id == "doc123"
        assert "dense" in point.vector
        assert "sparse" in point.vector
        assert point.payload["content"] == "Test content"
        assert point.payload["source"] == "test"
    
    @pytest.mark.asyncio
    async def test_add_single_document_no_sparse(self, mock_workspace_client, mock_embedding_service):
        """Test single document addition without sparse vectors."""
        mock_workspace_client.get_embedding_service.return_value = mock_embedding_service
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384
            # No sparse vectors
        }
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        
        result = await _add_single_document(
            mock_workspace_client,
            "Test content",
            "test_collection",
            {"source": "test"},
            "doc123"
        )
        
        assert result is True
        
        # Verify vector structure
        call_args = mock_client.upsert.call_args
        point = call_args[1]["points"][0]
        assert "dense" in point.vector
        assert "sparse" not in point.vector
    
    @pytest.mark.asyncio
    async def test_add_single_document_exception(self, mock_workspace_client, mock_embedding_service):
        """Test single document addition with exception."""
        mock_workspace_client.get_embedding_service.return_value = mock_embedding_service
        mock_embedding_service.generate_embeddings.side_effect = Exception("Embedding error")
        
        result = await _add_single_document(
            mock_workspace_client,
            "Test content",
            "test_collection",
            {"source": "test"},
            "doc123"
        )
        
        assert result is False


class TestUpdateDocument:
    """Test update_document function."""
    
    @pytest.mark.asyncio
    async def test_update_document_client_not_initialized(self, mock_workspace_client):
        """Test update with uninitialized client."""
        mock_workspace_client.initialized = False
        
        result = await update_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert result["error"] == "Workspace client not initialized"
    
    @pytest.mark.asyncio
    async def test_update_document_collection_not_found(self, mock_workspace_client):
        """Test update with non-existent collection."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["other_collection"]
        
        result = await update_document(
            mock_workspace_client,
            "doc123",
            "nonexistent_collection"
        )
        
        assert "error" in result
        assert "Collection 'nonexistent_collection' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_update_document_not_found(self, mock_workspace_client):
        """Test update of non-existent document."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([], None)  # Empty results
        
        result = await update_document(
            mock_workspace_client,
            "nonexistent_doc",
            "test_collection"
        )
        
        assert "error" in result
        assert "Document 'nonexistent_doc' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_update_document_metadata_only(self, mock_workspace_client):
        """Test updating document metadata only."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        # Mock existing document
        mock_point = MagicMock()
        mock_point.id = "doc123"
        mock_point.payload = {"content": "original content", "source": "old"}
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([mock_point], None)
        
        result = await update_document(
            mock_workspace_client,
            "doc123",
            "test_collection",
            metadata={"source": "new", "category": "updated"}
        )
        
        assert result["document_id"] == "doc123"
        assert result["points_updated"] == 1
        assert result["metadata_updated"] is True
        assert result["content_updated"] is False
        
        # Verify upsert was called
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        point = call_args[1]["points"][0]
        assert point.payload["source"] == "new"
        assert point.payload["category"] == "updated"
    
    @pytest.mark.asyncio
    async def test_update_document_content_and_metadata(self, mock_workspace_client, mock_embedding_service):
        """Test updating both content and metadata."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        mock_workspace_client.get_embedding_service.return_value = mock_embedding_service
        
        # Mock embeddings
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.2] * 384,
            "sparse": {"indices": [3, 4], "values": [0.9, 0.7]}
        }
        
        # Mock existing document
        mock_point = MagicMock()
        mock_point.id = "doc123"
        mock_point.payload = {"content": "original content", "source": "old"}
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([mock_point], None)
        
        result = await update_document(
            mock_workspace_client,
            "doc123",
            "test_collection",
            content="new content",
            metadata={"source": "updated"}
        )
        
        assert result["document_id"] == "doc123"
        assert result["points_updated"] == 1
        assert result["content_updated"] is True
        assert result["metadata_updated"] is True
        
        # Verify upsert with new vectors
        call_args = mock_client.upsert.call_args
        point = call_args[1]["points"][0]
        assert point.payload["content"] == "new content"
        assert point.payload["source"] == "updated"
        assert "updated_at" in point.payload
        assert "dense" in point.vector
        assert "sparse" in point.vector
    
    @pytest.mark.asyncio
    async def test_update_document_multiple_points(self, mock_workspace_client):
        """Test updating document with multiple points (chunks)."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        # Mock multiple points (chunks)
        mock_points = []
        for i in range(3):
            point = MagicMock()
            point.id = f"doc123_chunk_{i}"
            point.payload = {"content": f"chunk {i}", "chunk_index": i}
            mock_points.append(point)
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = (mock_points, None)
        
        result = await update_document(
            mock_workspace_client,
            "doc123",
            "test_collection",
            metadata={"category": "updated"}
        )
        
        assert result["points_updated"] == 3
        assert mock_client.upsert.call_count == 3
    
    @pytest.mark.asyncio
    async def test_update_document_partial_failure(self, mock_workspace_client):
        """Test update with partial point failures."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        # Mock points
        mock_points = [MagicMock() for _ in range(2)]
        for i, point in enumerate(mock_points):
            point.id = f"doc123_chunk_{i}"
            point.payload = {"content": f"chunk {i}"}
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = (mock_points, None)
        
        # First upsert succeeds, second fails
        mock_client.upsert.side_effect = [None, Exception("Upsert error")]
        
        result = await update_document(
            mock_workspace_client,
            "doc123",
            "test_collection",
            metadata={"category": "updated"}
        )
        
        assert result["points_updated"] == 1  # Only one succeeded
    
    @pytest.mark.asyncio
    async def test_update_document_exception_handling(self, mock_workspace_client):
        """Test update exception handling."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.side_effect = Exception("Connection error")
        
        result = await update_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert "error" in result
        assert "Failed to update document" in result["error"]


class TestDeleteDocument:
    """Test delete_document function."""
    
    @pytest.mark.asyncio
    async def test_delete_document_client_not_initialized(self, mock_workspace_client):
        """Test delete with uninitialized client."""
        mock_workspace_client.initialized = False
        
        result = await delete_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert result["error"] == "Workspace client not initialized"
    
    @pytest.mark.asyncio
    async def test_delete_document_collection_not_found(self, mock_workspace_client):
        """Test delete with non-existent collection."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["other_collection"]
        
        result = await delete_document(
            mock_workspace_client,
            "doc123",
            "nonexistent_collection"
        )
        
        assert "error" in result
        assert "Collection 'nonexistent_collection' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_delete_document_success(self, mock_workspace_client):
        """Test successful document deletion."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        
        # Mock successful delete result
        mock_result = MagicMock()
        mock_result.operation_id = 12345
        mock_client.delete.return_value = mock_result
        
        result = await delete_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert result["document_id"] == "doc123"
        assert result["collection"] == "test_collection"
        assert result["points_deleted"] is True
        assert result["status"] == "success"
        
        # Verify delete was called with correct filter
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "test_collection"
        
        # Check filter structure
        filter_selector = call_args[1]["points_selector"]
        assert hasattr(filter_selector, "filter")
    
    @pytest.mark.asyncio
    async def test_delete_document_no_operation_id(self, mock_workspace_client):
        """Test delete result with no operation ID."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        
        # Mock result with no operation_id
        mock_result = MagicMock()
        mock_result.operation_id = None
        mock_client.delete.return_value = mock_result
        
        result = await delete_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert result["points_deleted"] is False
    
    @pytest.mark.asyncio
    async def test_delete_document_exception_handling(self, mock_workspace_client):
        """Test delete exception handling."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.side_effect = Exception("Connection error")
        
        result = await delete_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert "error" in result
        assert "Failed to delete document" in result["error"]


class TestGetDocument:
    """Test get_document function."""
    
    @pytest.mark.asyncio
    async def test_get_document_client_not_initialized(self, mock_workspace_client):
        """Test get with uninitialized client."""
        mock_workspace_client.initialized = False
        
        result = await get_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert result["error"] == "Workspace client not initialized"
    
    @pytest.mark.asyncio
    async def test_get_document_collection_not_found(self, mock_workspace_client):
        """Test get with non-existent collection."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["other_collection"]
        
        result = await get_document(
            mock_workspace_client,
            "doc123",
            "nonexistent_collection"
        )
        
        assert "error" in result
        assert "Collection 'nonexistent_collection' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mock_workspace_client):
        """Test get of non-existent document."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([], None)  # Empty results
        
        result = await get_document(
            mock_workspace_client,
            "nonexistent_doc",
            "test_collection"
        )
        
        assert "error" in result
        assert "Document 'nonexistent_doc' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_get_document_single_success(self, mock_workspace_client):
        """Test successful single document retrieval."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        # Mock document point
        mock_point = MagicMock()
        mock_point.id = "doc123"
        mock_point.payload = {
            "content": "Test document content",
            "source": "test",
            "document_id": "doc123"
        }
        mock_point.vector = None
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([mock_point], None)
        
        result = await get_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert result["document_id"] == "doc123"
        assert result["collection"] == "test_collection"
        assert result["total_points"] == 1
        assert result["is_chunked"] is False
        assert len(result["points"]) == 1
        
        point_data = result["points"][0]
        assert point_data["id"] == "doc123"
        assert point_data["payload"]["content"] == "Test document content"
        assert "vectors" not in point_data
    
    @pytest.mark.asyncio
    async def test_get_document_with_vectors(self, mock_workspace_client):
        """Test document retrieval with vectors."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        # Mock document point with vectors
        mock_point = MagicMock()
        mock_point.id = "doc123"
        mock_point.payload = {"content": "Test content"}
        mock_point.vector = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.8, 0.6]}
        }
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = ([mock_point], None)
        
        result = await get_document(
            mock_workspace_client,
            "doc123",
            "test_collection",
            include_vectors=True
        )
        
        point_data = result["points"][0]
        assert "vectors" in point_data
        assert "dense" in point_data["vectors"]
        assert "sparse" in point_data["vectors"]
    
    @pytest.mark.asyncio
    async def test_get_document_chunked(self, mock_workspace_client):
        """Test retrieval of chunked document."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.return_value = ["test_collection"]
        
        # Mock multiple chunks
        mock_points = []
        for i in range(3):
            point = MagicMock()
            point.id = f"doc123_chunk_{i}"
            point.payload = {
                "content": f"Chunk {i} content",
                "chunk_index": i,
                "chunk_count": 3,
                "document_id": "doc123"
            }
            point.vector = None
            mock_points.append(point)
        
        # Return chunks in wrong order to test sorting
        mock_points_shuffled = [mock_points[2], mock_points[0], mock_points[1]]
        
        mock_client = MagicMock()
        mock_workspace_client.client = mock_client
        mock_client.scroll.return_value = (mock_points_shuffled, None)
        
        result = await get_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert result["total_points"] == 3
        assert result["is_chunked"] is True
        
        # Verify chunks are sorted by index
        points = result["points"]
        for i, point in enumerate(points):
            assert point["payload"]["chunk_index"] == i
    
    @pytest.mark.asyncio
    async def test_get_document_exception_handling(self, mock_workspace_client):
        """Test get exception handling."""
        mock_workspace_client.initialized = True
        mock_workspace_client.list_collections.side_effect = Exception("Connection error")
        
        result = await get_document(
            mock_workspace_client,
            "doc123",
            "test_collection"
        )
        
        assert "error" in result
        assert "Failed to get document" in result["error"]
