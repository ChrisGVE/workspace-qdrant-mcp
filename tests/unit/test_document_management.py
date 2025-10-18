"""
Unit tests for document management operations (Task 282).

Tests CRUD operations on documents in the MCP server including:
- Basic CRUD operations (create, read, update, delete)
- Batch operations
- Metadata updates
- Duplicate detection
- Error handling for edge cases
- Async operation validation

Test Subtasks:
- 282.1: Test fixtures and infrastructure (CURRENT)
- 282.2: Basic CRUD operation tests
- 282.3: Batch operation validation tests
- 282.4: Metadata update and duplicate detection tests
- 282.5: Error handling and edge case tests
- 282.6: Async operation testing with pytest-asyncio

Requirements:
- pytest-asyncio for async test support
- Mock Qdrant client responses for isolation
- Mock daemon client for write path testing
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

# Import MCP server functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from workspace_qdrant_mcp.server import store, manage, retrieve
from qdrant_client.models import Distance, PointStruct, VectorParams


# ============================================================================
# TEST FIXTURES (Task 282.1)
# ============================================================================

@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for isolated testing."""
    client = MagicMock()
    client.get_collections = MagicMock()
    client.get_collection = MagicMock()
    client.create_collection = MagicMock()
    client.delete_collection = MagicMock()
    client.upsert = MagicMock()
    client.retrieve = MagicMock()
    client.scroll = MagicMock()
    client.delete = MagicMock()
    return client


@pytest.fixture
def mock_daemon_client():
    """Mock daemon client for write path testing."""
    client = AsyncMock()

    # Mock ingest_text response
    async def mock_ingest_text(content, collection_basename, tenant_id, metadata, chunk_text=True):
        response = AsyncMock()
        response.document_id = str(uuid.uuid4())
        response.chunks_created = 1
        response.success = True
        return response

    client.ingest_text = mock_ingest_text

    # Mock create_collection_v2 response
    async def mock_create_collection(collection_name, vector_size, distance_metric):
        response = AsyncMock()
        response.success = True
        response.collection_name = collection_name
        return response

    client.create_collection_v2 = mock_create_collection

    return client


@pytest.fixture
def sample_document_metadata():
    """Sample document metadata for testing."""
    return {
        "title": "Test Document",
        "author": "Test User",
        "tags": ["test", "sample"],
        "category": "testing",
        "created_at": datetime.now(timezone.utc).isoformat()
    }


@pytest.fixture
def sample_documents():
    """Sample documents for batch operation testing."""
    return [
        {
            "content": "First test document content",
            "title": "Document 1",
            "metadata": {"index": 1, "category": "test"}
        },
        {
            "content": "Second test document content",
            "title": "Document 2",
            "metadata": {"index": 2, "category": "test"}
        },
        {
            "content": "Third test document content",
            "title": "Document 3",
            "metadata": {"index": 3, "category": "test"}
        }
    ]


@pytest.fixture
def mock_embeddings():
    """Mock embeddings for document vectors."""
    return [0.1] * 384  # Standard embedding size


@pytest.fixture
async def cleanup_test_collections(mock_qdrant_client):
    """Cleanup fixture for test collections."""
    yield
    # Cleanup code would go here in real environment
    # For unit tests with mocks, no actual cleanup needed


# ============================================================================
# BASIC CRUD OPERATION TESTS (Task 282.2)
# ============================================================================

class TestDocumentCreate:
    """Test document creation operations."""

    @pytest.mark.asyncio
    async def test_create_document_basic(
        self, mock_qdrant_client, mock_daemon_client, sample_document_metadata
    ):
        """Test basic document creation with minimal parameters."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(
                content="Test document content",
                title="Test Document",
                metadata=sample_document_metadata
            )

            assert result["success"] is True
            assert "document_id" in result
            assert result["title"] == "Test Document"
            assert result["content_length"] > 0

    @pytest.mark.asyncio
    async def test_create_document_with_full_metadata(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test document creation with comprehensive metadata."""
        full_metadata = {
            "title": "Comprehensive Test Document",
            "author": "Test User",
            "tags": ["test", "comprehensive", "metadata"],
            "category": "testing",
            "version": "1.0.0",
            "language": "en",
            "custom_field": "custom_value"
        }

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(
                content="Test document with full metadata",
                title="Comprehensive Document",
                metadata=full_metadata,
                source="user_input",
                document_type="text"
            )

            assert result["success"] is True
            assert "metadata" in result
            assert result["metadata"]["author"] == "Test User"

    @pytest.mark.asyncio
    async def test_create_document_different_sources(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test document creation from different sources."""
        sources = ["user_input", "scratchbook", "file", "web", "api"]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            for source in sources:
                result = await store(
                    content=f"Document from {source}",
                    title=f"Document - {source}",
                    source=source
                )

                assert result["success"] is True
                assert result["metadata"]["source"] == source

    @pytest.mark.asyncio
    async def test_create_document_with_file_path(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test document creation with file path metadata."""
        test_file_path = "/test/path/to/document.py"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(
                content="def test_function():\n    pass",
                title="Python File",
                file_path=test_file_path,
                source="file",
                document_type="code"
            )

            assert result["success"] is True
            assert result["metadata"]["file_path"] == test_file_path
            assert result["metadata"]["file_name"] == "document.py"


class TestDocumentRead:
    """Test document retrieval operations."""

    @pytest.mark.asyncio
    async def test_retrieve_document_by_id(self, mock_qdrant_client):
        """Test retrieving a document by its ID."""
        test_doc_id = str(uuid.uuid4())

        # Mock retrieve response
        mock_point = MagicMock()
        mock_point.id = test_doc_id
        mock_point.payload = {
            "content": "Test content",
            "title": "Test Document",
            "branch": "main"
        }
        mock_qdrant_client.retrieve.return_value = [mock_point]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock), \
             patch('workspace_qdrant_mcp.server.get_project_collection', return_value="test-collection"):

            result = await retrieve(document_id=test_doc_id)

            assert result["success"] is True
            assert "results" in result

    @pytest.mark.asyncio
    async def test_retrieve_documents_by_metadata(self, mock_qdrant_client):
        """Test retrieving documents using metadata filters."""
        # Mock scroll response
        mock_point = MagicMock()
        mock_point.id = str(uuid.uuid4())
        mock_point.payload = {
            "content": "Filtered content",
            "title": "Filtered Document",
            "category": "test"
        }
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock), \
             patch('workspace_qdrant_mcp.server.get_project_collection', return_value="test-collection"):

            result = await retrieve(metadata={"category": "test"})

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent_document(self, mock_qdrant_client):
        """Test retrieval of non-existent document returns empty result."""
        mock_qdrant_client.retrieve.return_value = []

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock), \
             patch('workspace_qdrant_mcp.server.get_project_collection', return_value="test-collection"):

            result = await retrieve(document_id=str(uuid.uuid4()))

            assert result["success"] is True
            assert result["total_results"] == 0


class TestDocumentUpdate:
    """Test document update operations."""

    @pytest.mark.asyncio
    async def test_update_document_content(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test updating document content creates new version."""
        # In current architecture, updates create new points
        test_doc_id = str(uuid.uuid4())

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            # Store updated version
            result = await store(
                content="Updated document content",
                title="Updated Document",
                metadata={"original_id": test_doc_id, "version": 2}
            )

            assert result["success"] is True
            assert "document_id" in result

    @pytest.mark.asyncio
    async def test_update_document_metadata(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test updating document metadata."""
        original_metadata = {"title": "Original", "version": 1}
        updated_metadata = {"title": "Updated", "version": 2, "updated": True}

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(
                content="Document content",
                title="Updated Document",
                metadata=updated_metadata
            )

            assert result["success"] is True
            assert result["metadata"]["version"] == 2


class TestDocumentDelete:
    """Test document deletion operations."""

    @pytest.mark.asyncio
    async def test_delete_collection(self, mock_qdrant_client):
        """Test deleting an entire collection."""
        test_collection = "test-collection"
        mock_qdrant_client.delete_collection.return_value = None

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', None), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(
                action="delete_collection",
                name=test_collection
            )

            assert result["success"] is True
            mock_qdrant_client.delete_collection.assert_called_once_with(test_collection)


# ============================================================================
# BATCH OPERATION TESTS (Task 282.3)
# ============================================================================

class TestBatchOperations:
    """Test batch document operations."""

    @pytest.mark.asyncio
    async def test_batch_document_creation(
        self, mock_qdrant_client, mock_daemon_client, sample_documents
    ):
        """Test creating multiple documents in batch."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            results = []
            for doc in sample_documents:
                result = await store(
                    content=doc["content"],
                    title=doc["title"],
                    metadata=doc["metadata"]
                )
                results.append(result)

            assert len(results) == len(sample_documents)
            assert all(r["success"] for r in results)

    @pytest.mark.asyncio
    async def test_batch_retrieval_by_category(self, mock_qdrant_client, sample_documents):
        """Test retrieving multiple documents by category."""
        # Mock scroll response with multiple documents
        mock_points = []
        for i, doc in enumerate(sample_documents):
            mock_point = MagicMock()
            mock_point.id = str(uuid.uuid4())
            mock_point.payload = {
                "content": doc["content"],
                "title": doc["title"],
                **doc["metadata"]
            }
            mock_points.append(mock_point)

        mock_qdrant_client.scroll.return_value = (mock_points, None)

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock), \
             patch('workspace_qdrant_mcp.server.get_project_collection', return_value="test-collection"):

            result = await retrieve(
                metadata={"category": "test"},
                limit=10
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_batch_operation_performance(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test batch operation completes within reasonable time."""
        import time

        batch_size = 50

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            start_time = time.time()

            tasks = []
            for i in range(batch_size):
                task = store(
                    content=f"Document {i} content",
                    title=f"Document {i}",
                    metadata={"index": i}
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start_time

            assert len(results) == batch_size
            assert all(r["success"] for r in results)
            # With mocks, should be very fast
            assert elapsed < 5.0  # 5 seconds for 50 documents with mocks


# ============================================================================
# METADATA AND DUPLICATE DETECTION TESTS (Task 282.4)
# ============================================================================

class TestMetadataOperations:
    """Test metadata update and management operations."""

    @pytest.mark.asyncio
    async def test_metadata_enrichment(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test automatic metadata enrichment during document creation."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(
                content="Test content",
                title="Test Document",
                metadata={"custom_field": "custom_value"}
            )

            assert result["success"] is True
            # Server adds created_at, source, document_type automatically
            assert "created_at" in result["metadata"]
            assert "source" in result["metadata"]
            assert result["metadata"]["custom_field"] == "custom_value"

    @pytest.mark.asyncio
    async def test_metadata_filtering(self, mock_qdrant_client):
        """Test filtering documents by metadata fields."""
        mock_point = MagicMock()
        mock_point.id = str(uuid.uuid4())
        mock_point.payload = {
            "content": "Test content",
            "title": "Filtered Document",
            "status": "published",
            "priority": "high"
        }
        mock_qdrant_client.scroll.return_value = ([mock_point], None)

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock), \
             patch('workspace_qdrant_mcp.server.get_project_collection', return_value="test-collection"):

            result = await retrieve(
                metadata={"status": "published", "priority": "high"}
            )

            assert result["success"] is True


class TestDuplicateDetection:
    """Test duplicate document detection and handling."""

    @pytest.mark.asyncio
    async def test_duplicate_content_creates_new_document(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test that duplicate content creates a new document (upsert behavior)."""
        duplicate_content = "This content appears twice"

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            # Create first document
            result1 = await store(
                content=duplicate_content,
                title="Original Document"
            )

            # Create second document with same content
            result2 = await store(
                content=duplicate_content,
                title="Duplicate Document"
            )

            assert result1["success"] is True
            assert result2["success"] is True
            # Different document IDs (upsert creates new points)
            assert result1["document_id"] != result2["document_id"]


# ============================================================================
# ERROR HANDLING AND EDGE CASE TESTS (Task 282.5)
# ============================================================================

class TestErrorHandling:
    """Test error handling for document operations."""

    @pytest.mark.asyncio
    async def test_empty_content_handling(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test handling of empty content."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(
                content="",
                title="Empty Document"
            )

            # Empty content should still be accepted (validation at daemon level)
            assert result["success"] is True
            assert result["content_length"] == 0

    @pytest.mark.asyncio
    async def test_missing_metadata_creates_defaults(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test that missing metadata gets default values."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(content="Test content")

            assert result["success"] is True
            # Default title is generated if not provided
            assert "title" in result["metadata"]

    @pytest.mark.asyncio
    async def test_retrieve_without_parameters_fails(self, mock_qdrant_client):
        """Test that retrieve without document_id or metadata returns error."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await retrieve()

            assert result["success"] is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_daemon_unavailable_fallback(self, mock_qdrant_client):
        """Test fallback to direct Qdrant when daemon unavailable."""
        # Prepare mock for fallback path
        mock_qdrant_client.upsert.return_value = None

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', None), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock), \
             patch('workspace_qdrant_mcp.server.ensure_collection_exists', return_value=True), \
             patch('workspace_qdrant_mcp.server.generate_embeddings', return_value=[0.1]*384):

            result = await store(
                content="Test content",
                title="Fallback Test"
            )

            # Fallback mode should still succeed
            assert result["success"] is True
            assert "fallback_mode" in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_large_content(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test handling of very large content."""
        large_content = "x" * 100000  # 100KB content

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(
                content=large_content,
                title="Large Document"
            )

            assert result["success"] is True
            assert result["content_length"] == 100000

    @pytest.mark.asyncio
    async def test_special_characters_in_metadata(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test metadata with special characters."""
        special_metadata = {
            "title": "Test with 特殊字符 and émojis 🎉",
            "tags": ["test@tag", "tag#1", "tag$2"],
            "description": "Line 1\nLine 2\tTabbed"
        }

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await store(
                content="Test content",
                metadata=special_metadata
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_document_operations(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test concurrent document creation and retrieval."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            # Create multiple documents concurrently
            tasks = [
                store(content=f"Doc {i}", title=f"Concurrent {i}")
                for i in range(10)
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 10
            assert all(r["success"] for r in results)


# ============================================================================
# ASYNC OPERATION VALIDATION TESTS (Task 282.6)
# ============================================================================

class TestAsyncOperations:
    """Test async operation behavior with pytest-asyncio."""

    @pytest.mark.asyncio
    async def test_store_is_async(self, mock_qdrant_client, mock_daemon_client):
        """Test that store operation is properly async."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            # Verify await is required (test passes if async)
            result = await store(content="Async test", title="Async Document")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_retrieve_is_async(self, mock_qdrant_client):
        """Test that retrieve operation is properly async."""
        mock_point = MagicMock()
        mock_point.id = str(uuid.uuid4())
        mock_point.payload = {"content": "test", "branch": "main"}
        mock_qdrant_client.retrieve.return_value = [mock_point]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock), \
             patch('workspace_qdrant_mcp.server.get_project_collection', return_value="test-collection"):

            result = await retrieve(document_id=str(uuid.uuid4()))

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_manage_is_async(self, mock_qdrant_client):
        """Test that manage operation is properly async."""
        mock_qdrant_client.get_collections.return_value.collections = []

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            result = await manage(action="list_collections")

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_concurrent_async_operations(
        self, mock_qdrant_client, mock_daemon_client
    ):
        """Test multiple async operations running concurrently."""
        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon_client), \
             patch('workspace_qdrant_mcp.server.initialize_components', new_callable=AsyncMock):

            # Run 20 operations concurrently
            tasks = [
                store(content=f"Concurrent {i}", title=f"Doc {i}")
                for i in range(20)
            ]

            results = await asyncio.gather(*tasks)

            assert len(results) == 20
            assert all(r["success"] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
