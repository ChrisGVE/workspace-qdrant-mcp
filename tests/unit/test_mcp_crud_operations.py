"""
Comprehensive Basic CRUD Operation Tests for MCP Document Management (Task 323.1).

This test suite validates the core CRUD operations for document management using
the 4-tool MCP server: store, search, manage, retrieve.

CRUD Operations Tested:
    - CREATE: store tool - Add documents with various types
    - READ: retrieve tool - Get documents by ID and metadata
    - UPDATE: store tool with update - Update content and metadata
    - DELETE: manage tool - Remove documents

Document Types:
    - Text documents (plain content)
    - Code documents (source code)
    - Binary metadata (structured data)

Test Focus:
    - Response format validation
    - Status code verification
    - Data integrity checks
    - Error handling
    - FastMCP tool access patterns
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/python'))

# Import server module for testing
import workspace_qdrant_mcp.server as server_module
from workspace_qdrant_mcp.server import app


class TestStoreCRUDCreate:
    """Test CREATE operations using the store() tool."""

    @pytest.mark.asyncio
    async def test_store_text_document_basic(self):
        """Test storing a basic text document."""
        content = "This is a test document for CRUD testing."
        title = "Test Document 1"
        metadata = {"type": "text", "category": "test"}

        # Access the underlying function from the tool
        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            result = await store_fn(
                content=content,
                title=title,
                metadata=metadata
            )

            # Verify response structure
            assert isinstance(result, dict)
            assert "success" in result
            assert "document_id" in result
            assert "collection" in result
            assert result["success"] is True
            assert result["title"] == title
            assert result["content_length"] == len(content)

            # Verify upsert was called
            mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_code_document(self):
        """Test storing a code document."""
        code_content = '''
def hello_world():
    """Print hello world."""
    print("Hello, World!")
    return True
'''
        title = "hello_world.py"
        metadata = {"type": "code", "language": "python", "file_type": "code"}

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            result = await store_fn(
                content=code_content,
                title=title,
                metadata=metadata,
                source="file",
                document_type="code"
            )

            # Verify response
            assert result["success"] is True
            assert "document_id" in result
            assert result["title"] == title

            # Verify code was stored with proper metadata
            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            assert len(points) == 1
            payload = points[0].payload
            assert payload["content"] == code_content
            assert payload["document_type"] == "code"

    @pytest.mark.asyncio
    async def test_store_binary_metadata_document(self):
        """Test storing document with binary/structured metadata."""
        content = "Complex document with binary metadata"
        title = "Binary Metadata Doc"
        binary_metadata = {
            "type": "binary",
            "version": 2,
            "flags": [True, False, True],
            "config": {
                "enabled": True,
                "threshold": 0.85,
                "nested": {"key": "value"}
            },
            "tags": ["binary", "metadata", "test"]
        }

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            result = await store_fn(
                content=content,
                title=title,
                metadata=binary_metadata
            )

            # Verify binary metadata is preserved
            assert result["success"] is True

            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            payload = points[0].payload

            # Check nested metadata structure
            assert payload["config"]["enabled"] is True
            assert payload["config"]["threshold"] == 0.85
            assert payload["tags"] == ["binary", "metadata", "test"]

    @pytest.mark.asyncio
    async def test_store_with_file_path(self):
        """Test storing document with file path metadata."""
        content = "File content"
        file_path = "/path/to/test/document.py"

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            result = await store_fn(
                content=content,
                file_path=file_path
            )

            # Verify file metadata
            assert result["success"] is True

            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            payload = points[0].payload

            assert payload["file_path"] == file_path
            assert payload["file_name"] == "document.py"

    @pytest.mark.asyncio
    async def test_store_with_url_metadata(self):
        """Test storing document with URL metadata."""
        content = "Web content"
        url = "https://example.com/docs/api-reference"

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            result = await store_fn(
                content=content,
                url=url,
                source="web"
            )

            # Verify URL metadata
            assert result["success"] is True

            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            payload = points[0].payload

            assert payload["url"] == url
            assert payload["domain"] == "example.com"
            assert payload["source"] == "web"


class TestRetrieveCRUDRead:
    """Test READ operations using the retrieve() tool."""

    @pytest.mark.asyncio
    async def test_retrieve_by_document_id(self):
        """Test retrieving document by ID."""
        document_id = str(uuid.uuid4())
        test_content = "Test document content for retrieval"

        retrieve_fn = app._tool_manager._tools['retrieve'].fn

        # Mock point with branch metadata
        mock_point = MagicMock()
        mock_point.id = document_id
        mock_point.payload = {
            "content": test_content,
            "title": "Test Doc",
            "branch": "main",
            "file_type": "text"
        }

        with patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'get_current_branch', return_value="main"):

            mock_qdrant.retrieve = Mock(return_value=[mock_point])

            result = await retrieve_fn(document_id=document_id)

            # Verify response
            assert result["success"] is True
            assert result["total_results"] == 1
            assert len(result["results"]) == 1

            doc = result["results"][0]
            assert doc["id"] == document_id
            assert doc["content"] == test_content
            assert doc["title"] == "Test Doc"

    @pytest.mark.asyncio
    async def test_retrieve_by_metadata(self):
        """Test retrieving documents by metadata filter."""
        metadata_filter = {"type": "code", "language": "python"}

        retrieve_fn = app._tool_manager._tools['retrieve'].fn

        # Mock scroll results
        mock_points = [
            MagicMock(
                id=str(uuid.uuid4()),
                payload={
                    "content": "def test(): pass",
                    "title": "test.py",
                    "type": "code",
                    "language": "python",
                    "branch": "main"
                }
            ),
            MagicMock(
                id=str(uuid.uuid4()),
                payload={
                    "content": "class TestClass: pass",
                    "title": "class.py",
                    "type": "code",
                    "language": "python",
                    "branch": "main"
                }
            )
        ]

        with patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'get_current_branch', return_value="main"):

            mock_qdrant.scroll = Mock(return_value=(mock_points, None))

            result = await retrieve_fn(
                metadata=metadata_filter,
                limit=10
            )

            # Verify results
            assert result["success"] is True
            assert result["total_results"] == 2
            assert result["query_type"] == "metadata_filter"

            # Verify all results match filter
            for doc in result["results"]:
                assert doc["metadata"]["type"] == "code"
                assert doc["metadata"]["language"] == "python"

    @pytest.mark.asyncio
    async def test_retrieve_by_id_not_found(self):
        """Test retrieving non-existent document by ID."""
        document_id = str(uuid.uuid4())

        retrieve_fn = app._tool_manager._tools['retrieve'].fn

        with patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'get_current_branch', return_value="main"):

            mock_qdrant.retrieve = Mock(return_value=[])

            result = await retrieve_fn(document_id=document_id)

            # Should succeed but with no results
            assert result["success"] is True
            assert result["total_results"] == 0
            assert len(result["results"]) == 0

    @pytest.mark.asyncio
    async def test_retrieve_with_branch_filter(self):
        """Test retrieving documents with branch filter."""
        document_id = str(uuid.uuid4())

        retrieve_fn = app._tool_manager._tools['retrieve'].fn

        # Mock point on develop branch
        mock_point = MagicMock()
        mock_point.id = document_id
        mock_point.payload = {
            "content": "Branch-specific content",
            "branch": "develop",
            "file_type": "code"
        }

        with patch.object(server_module, 'qdrant_client') as mock_qdrant:

            mock_qdrant.retrieve = Mock(return_value=[mock_point])

            result = await retrieve_fn(
                document_id=document_id,
                branch="develop"
            )

            # Verify branch-specific retrieval
            assert result["success"] is True
            assert result["total_results"] == 1
            assert result["filters_applied"]["branch"] == "develop"

    @pytest.mark.asyncio
    async def test_retrieve_with_file_type_filter(self):
        """Test retrieving documents with file_type filter."""
        metadata_filter = {"category": "test"}

        retrieve_fn = app._tool_manager._tools['retrieve'].fn

        mock_points = [
            MagicMock(
                id=str(uuid.uuid4()),
                payload={
                    "content": "Test content",
                    "file_type": "test",
                    "category": "test",
                    "branch": "main"
                }
            )
        ]

        with patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'get_current_branch', return_value="main"):

            mock_qdrant.scroll = Mock(return_value=(mock_points, None))

            result = await retrieve_fn(
                metadata=metadata_filter,
                file_type="test"
            )

            # Verify file_type filter was applied
            assert result["success"] is True
            assert result["filters_applied"]["file_type"] == "test"


class TestStoreCRUDUpdate:
    """Test UPDATE operations by re-storing documents."""

    @pytest.mark.asyncio
    async def test_update_document_content(self):
        """Test updating document content by storing with same ID."""
        document_id = str(uuid.uuid4())
        updated_content = "Updated content with new information"

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            # Update by storing with new content
            result = await store_fn(
                content=updated_content,
                title="Updated Document",
                metadata={"document_id": document_id, "updated": True}
            )

            # Verify update succeeded
            assert result["success"] is True
            assert result["content_length"] == len(updated_content)

            # Verify upsert was called (upsert handles update)
            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            assert points[0].payload["content"] == updated_content

    @pytest.mark.asyncio
    async def test_update_document_metadata(self):
        """Test updating document metadata."""
        content = "Same content"

        updated_metadata = {"version": 2, "status": "published", "reviewed": True}

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            # Update metadata
            result = await store_fn(
                content=content,
                metadata=updated_metadata
            )

            # Verify metadata update
            assert result["success"] is True

            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            payload = points[0].payload

            assert payload["version"] == 2
            assert payload["status"] == "published"
            assert payload["reviewed"] is True

    @pytest.mark.asyncio
    async def test_update_partial_metadata(self):
        """Test updating only specific metadata fields."""
        content = "Document content"

        # Update only field2
        update_meta = {"field1": "value1", "field2": "updated_value2", "field3": "value3"}

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            await store_fn(
                content=content,
                metadata=update_meta
            )

            # Verify partial update
            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            payload = points[0].payload

            assert payload["field1"] == "value1"
            assert payload["field2"] == "updated_value2"
            assert payload["field3"] == "value3"


class TestManageCRUDDelete:
    """Test DELETE operations using the manage() tool."""

    @pytest.mark.asyncio
    async def test_delete_collection(self):
        """Test deleting a collection (which deletes all documents)."""
        collection_name = "test-collection-to-delete"

        manage_fn = app._tool_manager._tools['manage'].fn

        with patch.object(server_module, 'daemon_client') as mock_daemon, \
             patch.object(server_module, 'qdrant_client'):

            # Mock daemon delete
            mock_daemon.delete_collection_v2 = AsyncMock()

            result = await manage_fn(
                action="delete_collection",
                name=collection_name
            )

            # Verify deletion
            assert result["success"] is True
            assert result["action"] == "delete_collection"
            assert collection_name in result["message"]

            # Verify daemon was called
            mock_daemon.delete_collection_v2.assert_called_once_with(
                collection_name=collection_name
            )

    @pytest.mark.asyncio
    async def test_delete_collection_fallback_mode(self):
        """Test deleting collection falls back when daemon unavailable."""
        collection_name = "test-collection"

        manage_fn = app._tool_manager._tools['manage'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant:

            mock_qdrant.delete_collection = Mock()

            result = await manage_fn(
                action="delete_collection",
                name=collection_name
            )

            # Verify fallback deletion
            assert result["success"] is True
            assert "direct write" in result["message"].lower()

            # Verify direct qdrant call
            mock_qdrant.delete_collection.assert_called_once_with(collection_name)

    @pytest.mark.asyncio
    async def test_cleanup_empty_collections(self):
        """Test cleanup action deletes empty collections."""
        manage_fn = app._tool_manager._tools['manage'].fn

        # Mock collections response
        mock_collections = [
            MagicMock(name="empty-collection-1"),
            MagicMock(name="collection-with-data"),
            MagicMock(name="empty-collection-2")
        ]

        mock_col_info_1 = MagicMock(points_count=0)
        mock_col_info_2 = MagicMock(points_count=100)
        mock_col_info_3 = MagicMock(points_count=0)

        with patch.object(server_module, 'daemon_client') as mock_daemon, \
             patch.object(server_module, 'qdrant_client') as mock_qdrant:

            mock_collections_response = MagicMock()
            mock_collections_response.collections = mock_collections

            mock_qdrant.get_collections = Mock(return_value=mock_collections_response)
            mock_qdrant.get_collection = Mock(side_effect=[
                mock_col_info_1, mock_col_info_2, mock_col_info_3
            ])

            mock_daemon.delete_collection_v2 = AsyncMock()

            result = await manage_fn(action="cleanup")

            # Verify cleanup deleted empty collections
            assert result["success"] is True
            assert result["action"] == "cleanup"
            assert len(result["cleaned_collections"]) == 2
            assert "empty-collection-1" in result["cleaned_collections"]
            assert "empty-collection-2" in result["cleaned_collections"]


class TestCRUDResponseFormats:
    """Test CRUD operation response format consistency."""

    @pytest.mark.asyncio
    async def test_store_response_format(self):
        """Test store response has consistent format."""
        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            result = await store_fn(
                content="Test content",
                title="Test"
            )

            # Verify response format
            assert isinstance(result, dict)
            required_fields = ["success", "document_id", "collection", "title", "content_length"]
            for field in required_fields:
                assert field in result, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_retrieve_response_format(self):
        """Test retrieve response has consistent format."""
        document_id = str(uuid.uuid4())

        retrieve_fn = app._tool_manager._tools['retrieve'].fn

        mock_point = MagicMock()
        mock_point.id = document_id
        mock_point.payload = {"content": "Test", "branch": "main"}

        with patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'get_current_branch', return_value="main"):

            mock_qdrant.retrieve = Mock(return_value=[mock_point])

            result = await retrieve_fn(document_id=document_id)

            # Verify response format
            assert isinstance(result, dict)
            required_fields = ["success", "total_results", "results", "query_type", "filters_applied"]
            for field in required_fields:
                assert field in result, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_manage_response_format(self):
        """Test manage response has consistent format."""
        manage_fn = app._tool_manager._tools['manage'].fn

        with patch.object(server_module, 'qdrant_client') as mock_qdrant:

            mock_collections_response = MagicMock()
            mock_collections_response.collections = []
            mock_qdrant.get_collections = Mock(return_value=mock_collections_response)

            result = await manage_fn(action="list_collections")

            # Verify response format
            assert isinstance(result, dict)
            required_fields = ["success", "action", "collections", "total_collections"]
            for field in required_fields:
                assert field in result, f"Missing field: {field}"


class TestCRUDDataIntegrity:
    """Test data integrity across CRUD operations."""

    @pytest.mark.asyncio
    async def test_content_preservation_text(self):
        """Test text content is preserved exactly through store/retrieve."""
        original_content = "This is test content with special chars: @#$%^&*()"

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            await store_fn(content=original_content)

            # Verify content in upsert call
            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            stored_content = points[0].payload["content"]

            assert stored_content == original_content

    @pytest.mark.asyncio
    async def test_content_preservation_code(self):
        """Test code content is preserved with indentation and formatting."""
        code_content = '''
def complex_function(x, y):
    """
    A complex function with indentation.
    """
    if x > y:
        return x * 2
    else:
        return y / 2
'''

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            await store_fn(
                content=code_content,
                document_type="code"
            )

            # Verify code formatting preserved
            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            stored_content = points[0].payload["content"]

            assert stored_content == code_content
            assert "    if x > y:" in stored_content  # Indentation preserved

    @pytest.mark.asyncio
    async def test_metadata_integrity_nested(self):
        """Test nested metadata structure is preserved."""
        nested_metadata = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep_value",
                        "list": [1, 2, 3]
                    }
                }
            }
        }

        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'qdrant_client') as mock_qdrant, \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', return_value=[0.1] * 384):

            mock_qdrant.upsert = Mock()

            await store_fn(
                content="Test",
                metadata=nested_metadata
            )

            # Verify nested structure preserved
            call_args = mock_qdrant.upsert.call_args
            points = call_args[1]["points"]
            payload = points[0].payload

            assert payload["level1"]["level2"]["level3"]["value"] == "deep_value"
            assert payload["level1"]["level2"]["level3"]["list"] == [1, 2, 3]


class TestCRUDErrorHandling:
    """Test error handling in CRUD operations."""

    @pytest.mark.asyncio
    async def test_store_collection_creation_failure(self):
        """Test store handles collection creation failure."""
        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'ensure_collection_exists', return_value=False):

            result = await store_fn(content="Test")

            # Verify error response
            assert result["success"] is False
            assert "error" in result
            assert "collection" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_retrieve_missing_parameters(self):
        """Test retrieve handles missing required parameters."""
        retrieve_fn = app._tool_manager._tools['retrieve'].fn

        result = await retrieve_fn()

        # Should return error for missing parameters
        assert result["success"] is False
        assert "error" in result
        assert "document_id" in result["error"] or "metadata" in result["error"]

    @pytest.mark.asyncio
    async def test_manage_invalid_action(self):
        """Test manage handles invalid action."""
        manage_fn = app._tool_manager._tools['manage'].fn

        with patch.object(server_module, 'qdrant_client'):
            result = await manage_fn(action="invalid_action")

            # Verify error response
            assert result["success"] is False
            assert "error" in result
            assert "unknown action" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_store_embedding_generation_failure(self):
        """Test store handles embedding generation failure."""
        store_fn = app._tool_manager._tools['store'].fn

        with patch.object(server_module, 'daemon_client', None), \
             patch.object(server_module, 'ensure_collection_exists', return_value=True), \
             patch.object(server_module, 'generate_embeddings', side_effect=Exception("Embedding failed")):

            result = await store_fn(content="Test")

            # Should handle error gracefully
            assert result["success"] is False
            assert "error" in result
