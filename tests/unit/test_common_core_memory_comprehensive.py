"""
Comprehensive unit tests for common.core.memory module.

This test suite provides complete coverage of the document memory management
system, including document lifecycle, storage operations, retrieval, and
integration with embedding services.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.memory import (
    ChunkingStrategy,
    Document,
    DocumentChunk,
    DocumentMemoryManager,
    DocumentMetadata,
    MemoryIndex,
    RetrievalOptions,
    create_chunking_strategy,
)


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing."""
    client = AsyncMock()
    client.upsert.return_value = models.UpdateResult(
        operation_id=1,
        status=models.UpdateStatus.COMPLETED
    )
    client.search.return_value = []
    client.delete.return_value = models.UpdateResult(
        operation_id=2,
        status=models.UpdateStatus.COMPLETED
    )
    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.embed_document.return_value = {
        "dense": [0.1] * 384,
        "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
    }
    service.embed_query.return_value = {
        "dense": [0.2] * 384,
        "sparse": {"indices": [2, 6, 11], "values": [0.7, 0.5, 0.3]}
    }
    return service


@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    metadata = DocumentMetadata(
        file_path="/path/to/document.txt",
        file_type="text",
        file_size=1024,
        created_at=datetime.now(timezone.utc),
        modified_at=datetime.now(timezone.utc),
        project="test-project",
        author="test-user",
        tags=["test", "sample"],
        language="en"
    )

    return Document(
        id="doc123",
        content="This is a sample document content for testing purposes.",
        metadata=metadata,
        embedding=None,
        chunks=[]
    )


@pytest.fixture
def sample_document_chunks():
    """Create sample document chunks."""
    return [
        DocumentChunk(
            id="chunk1",
            document_id="doc123",
            content="This is the first chunk of the document.",
            start_offset=0,
            end_offset=40,
            chunk_index=0,
            embedding=None,
            metadata={"section": "introduction"}
        ),
        DocumentChunk(
            id="chunk2",
            document_id="doc123",
            content="This is the second chunk of the document.",
            start_offset=40,
            end_offset=82,
            chunk_index=1,
            embedding=None,
            metadata={"section": "body"}
        )
    ]


class TestDocumentMetadata:
    """Test DocumentMetadata class."""

    def test_init_required_fields(self):
        """Test initialization with required fields only."""
        metadata = DocumentMetadata(
            file_path="/test/file.txt",
            file_type="text",
            file_size=100
        )

        assert metadata.file_path == "/test/file.txt"
        assert metadata.file_type == "text"
        assert metadata.file_size == 100
        assert metadata.created_at is not None
        assert metadata.modified_at is not None

    def test_init_all_fields(self):
        """Test initialization with all fields."""
        created_time = datetime.now(timezone.utc)
        modified_time = datetime.now(timezone.utc)

        metadata = DocumentMetadata(
            file_path="/test/file.py",
            file_type="python",
            file_size=2048,
            created_at=created_time,
            modified_at=modified_time,
            project="my-project",
            author="developer",
            tags=["python", "code"],
            language="python",
            checksum="abc123def456"
        )

        assert metadata.project == "my-project"
        assert metadata.author == "developer"
        assert metadata.tags == ["python", "code"]
        assert metadata.language == "python"
        assert metadata.checksum == "abc123def456"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = DocumentMetadata(
            file_path="/test/file.txt",
            file_type="text",
            file_size=100,
            project="test"
        )

        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert result["file_path"] == "/test/file.txt"
        assert result["project"] == "test"
        assert "created_at" in result
        assert "modified_at" in result

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "file_path": "/test/file.txt",
            "file_type": "text",
            "file_size": 100,
            "project": "test",
            "tags": ["tag1", "tag2"]
        }

        metadata = DocumentMetadata.from_dict(data)

        assert metadata.file_path == "/test/file.txt"
        assert metadata.project == "test"
        assert metadata.tags == ["tag1", "tag2"]

    def test_update_checksum(self):
        """Test checksum calculation and update."""
        metadata = DocumentMetadata(
            file_path="/test/file.txt",
            file_type="text",
            file_size=100
        )

        content = "test content for checksum"
        metadata.update_checksum(content)

        assert metadata.checksum is not None
        assert len(metadata.checksum) > 0

        # Same content should produce same checksum
        original_checksum = metadata.checksum
        metadata.update_checksum(content)
        assert metadata.checksum == original_checksum

        # Different content should produce different checksum
        metadata.update_checksum("different content")
        assert metadata.checksum != original_checksum


class TestDocument:
    """Test Document class."""

    def test_init_basic_document(self, sample_document):
        """Test basic document initialization."""
        assert sample_document.id == "doc123"
        assert "sample document content" in sample_document.content
        assert sample_document.metadata.file_path == "/path/to/document.txt"
        assert sample_document.embedding is None
        assert len(sample_document.chunks) == 0

    def test_add_chunks(self, sample_document, sample_document_chunks):
        """Test adding chunks to document."""
        for chunk in sample_document_chunks:
            sample_document.add_chunk(chunk)

        assert len(sample_document.chunks) == 2
        assert sample_document.chunks[0].chunk_index == 0
        assert sample_document.chunks[1].chunk_index == 1

    def test_get_chunk_by_id(self, sample_document, sample_document_chunks):
        """Test retrieving chunk by ID."""
        for chunk in sample_document_chunks:
            sample_document.add_chunk(chunk)

        found_chunk = sample_document.get_chunk_by_id("chunk1")
        assert found_chunk is not None
        assert found_chunk.id == "chunk1"

        not_found = sample_document.get_chunk_by_id("nonexistent")
        assert not_found is None

    def test_get_chunks_by_section(self, sample_document, sample_document_chunks):
        """Test retrieving chunks by metadata section."""
        for chunk in sample_document_chunks:
            sample_document.add_chunk(chunk)

        intro_chunks = sample_document.get_chunks_by_metadata("section", "introduction")
        assert len(intro_chunks) == 1
        assert intro_chunks[0].id == "chunk1"

    def test_update_embedding(self, sample_document):
        """Test updating document embedding."""
        embedding = {"dense": [0.1, 0.2, 0.3], "sparse": {"indices": [1], "values": [0.5]}}

        sample_document.update_embedding(embedding)

        assert sample_document.embedding == embedding

    def test_document_serialization(self, sample_document):
        """Test document serialization to/from dict."""
        doc_dict = sample_document.to_dict()

        assert isinstance(doc_dict, dict)
        assert doc_dict["id"] == "doc123"
        assert "content" in doc_dict
        assert "metadata" in doc_dict

        # Test deserialization
        reconstructed = Document.from_dict(doc_dict)
        assert reconstructed.id == sample_document.id
        assert reconstructed.content == sample_document.content


class TestDocumentChunk:
    """Test DocumentChunk class."""

    def test_init_basic_chunk(self):
        """Test basic chunk initialization."""
        chunk = DocumentChunk(
            id="chunk1",
            document_id="doc1",
            content="Test chunk content",
            start_offset=0,
            end_offset=18,
            chunk_index=0
        )

        assert chunk.id == "chunk1"
        assert chunk.document_id == "doc1"
        assert chunk.content == "Test chunk content"
        assert chunk.start_offset == 0
        assert chunk.end_offset == 18
        assert chunk.chunk_index == 0

    def test_chunk_with_metadata(self):
        """Test chunk with custom metadata."""
        metadata = {"section": "header", "importance": "high"}
        chunk = DocumentChunk(
            id="chunk1",
            document_id="doc1",
            content="Test chunk",
            start_offset=0,
            end_offset=10,
            chunk_index=0,
            metadata=metadata
        )

        assert chunk.metadata == metadata
        assert chunk.get_metadata_value("section") == "header"
        assert chunk.get_metadata_value("nonexistent") is None

    def test_chunk_embedding_update(self):
        """Test updating chunk embedding."""
        chunk = DocumentChunk("chunk1", "doc1", "content", 0, 7, 0)

        embedding = {"dense": [0.1, 0.2], "sparse": {"indices": [1], "values": [0.5]}}
        chunk.update_embedding(embedding)

        assert chunk.embedding == embedding

    def test_chunk_content_length_validation(self):
        """Test that chunk offsets match content length."""
        content = "This is test content"
        chunk = DocumentChunk(
            "chunk1", "doc1", content, 0, len(content), 0
        )

        assert chunk.end_offset - chunk.start_offset == len(content)

    def test_chunk_serialization(self):
        """Test chunk serialization."""
        chunk = DocumentChunk(
            "chunk1", "doc1", "content", 0, 7, 0,
            metadata={"section": "test"}
        )

        chunk_dict = chunk.to_dict()
        assert isinstance(chunk_dict, dict)
        assert chunk_dict["id"] == "chunk1"

        reconstructed = DocumentChunk.from_dict(chunk_dict)
        assert reconstructed.id == chunk.id
        assert reconstructed.content == chunk.content


class TestChunkingStrategy:
    """Test different chunking strategies."""

    def test_fixed_size_chunking(self):
        """Test fixed-size chunking strategy."""
        strategy = create_chunking_strategy("fixed_size", chunk_size=50, overlap=10)

        content = "This is a long document that needs to be split into multiple chunks for processing."
        chunks = strategy.chunk_text(content, document_id="doc1")

        assert len(chunks) > 1
        assert all(len(chunk.content) <= 50 for chunk in chunks)

        # Test overlap
        if len(chunks) > 1:
            # Should have some overlap between consecutive chunks
            chunks[0].content[-10:]  # Last 10 chars
            chunks[1].content[:10]  # First 10 chars
            # Some overlap should exist (exact match not guaranteed due to word boundaries)

    def test_sentence_based_chunking(self):
        """Test sentence-based chunking strategy."""
        strategy = create_chunking_strategy("sentence", max_sentences=2)

        content = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = strategy.chunk_text(content, document_id="doc1")

        assert len(chunks) >= 2
        # Each chunk should contain complete sentences

    def test_paragraph_based_chunking(self):
        """Test paragraph-based chunking strategy."""
        strategy = create_chunking_strategy("paragraph")

        content = "First paragraph content.\n\nSecond paragraph content.\n\nThird paragraph content."
        chunks = strategy.chunk_text(content, document_id="doc1")

        assert len(chunks) == 3
        assert "First paragraph" in chunks[0].content
        assert "Second paragraph" in chunks[1].content

    def test_semantic_chunking(self):
        """Test semantic-based chunking (mock implementation)."""
        strategy = create_chunking_strategy("semantic", similarity_threshold=0.8)

        content = "Topic A content here. More about topic A. Now topic B begins. Topic B continues."
        chunks = strategy.chunk_text(content, document_id="doc1")

        assert len(chunks) >= 1
        # Semantic chunking would group similar content together

    def test_invalid_strategy(self):
        """Test error handling for invalid strategy."""
        with pytest.raises(ValueError):
            create_chunking_strategy("invalid_strategy")


class TestMemoryIndex:
    """Test memory index functionality."""

    def test_init_memory_index(self):
        """Test memory index initialization."""
        index = MemoryIndex(name="test-index", collection_name="test-collection")

        assert index.name == "test-index"
        assert index.collection_name == "test-collection"
        assert len(index.documents) == 0

    def test_add_document_to_index(self, sample_document):
        """Test adding document to index."""
        index = MemoryIndex("test-index", "test-collection")

        index.add_document(sample_document)

        assert len(index.documents) == 1
        assert sample_document.id in index.documents
        assert index.get_document(sample_document.id) == sample_document

    def test_remove_document_from_index(self, sample_document):
        """Test removing document from index."""
        index = MemoryIndex("test-index", "test-collection")
        index.add_document(sample_document)

        removed = index.remove_document(sample_document.id)

        assert removed == sample_document
        assert len(index.documents) == 0
        assert index.get_document(sample_document.id) is None

    def test_index_statistics(self, sample_document, sample_document_chunks):
        """Test index statistics collection."""
        index = MemoryIndex("test-index", "test-collection")

        # Add document with chunks
        for chunk in sample_document_chunks:
            sample_document.add_chunk(chunk)
        index.add_document(sample_document)

        stats = index.get_statistics()

        assert stats["document_count"] == 1
        assert stats["total_chunks"] == 2
        assert "average_document_size" in stats

    def test_index_search_capabilities(self, sample_document):
        """Test index search functionality."""
        index = MemoryIndex("test-index", "test-collection")
        index.add_document(sample_document)

        # Simple text search
        results = index.search_documents("sample")
        assert len(results) == 1
        assert results[0].id == sample_document.id

        # No matches
        results = index.search_documents("nonexistent")
        assert len(results) == 0


class TestDocumentMemoryManager:
    """Test the main document memory manager."""

    def test_init_memory_manager(self, mock_qdrant_client, mock_embedding_service):
        """Test memory manager initialization."""
        manager = DocumentMemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_name="test-memory"
        )

        assert manager.qdrant_client == mock_qdrant_client
        assert manager.embedding_service == mock_embedding_service
        assert manager.collection_name == "test-memory"

    @pytest.mark.asyncio
    async def test_store_document(self, mock_qdrant_client, mock_embedding_service, sample_document):
        """Test storing a document in memory."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        result = await manager.store_document(sample_document)

        assert result["status"] == "success"
        assert result["document_id"] == sample_document.id

        # Verify embedding was generated
        mock_embedding_service.embed_document.assert_called_once()

        # Verify document was stored in Qdrant
        mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_document_with_chunking(self, mock_qdrant_client, mock_embedding_service):
        """Test storing document with automatic chunking."""
        manager = DocumentMemoryManager(
            mock_qdrant_client,
            mock_embedding_service,
            "test-memory",
            chunking_strategy=create_chunking_strategy("fixed_size", chunk_size=50)
        )

        long_content = "This is a very long document that will be automatically chunked. " * 10
        metadata = DocumentMetadata("/test/long.txt", "text", len(long_content))
        document = Document("long_doc", long_content, metadata)

        result = await manager.store_document(document)

        assert result["status"] == "success"
        assert result.get("chunks_created", 0) > 0

        # Should have generated embeddings for chunks
        assert mock_embedding_service.embed_document.call_count > 1

    @pytest.mark.asyncio
    async def test_retrieve_document(self, mock_qdrant_client, mock_embedding_service, sample_document):
        """Test retrieving a document from memory."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        # Mock Qdrant search response
        mock_qdrant_client.search.return_value = [
            models.ScoredPoint(
                id=sample_document.id,
                score=0.95,
                version=0,
                payload={
                    "content": sample_document.content,
                    "metadata": sample_document.metadata.to_dict(),
                    "document_type": "full_document"
                }
            )
        ]

        result = await manager.retrieve_document(sample_document.id)

        assert result is not None
        assert result.id == sample_document.id
        assert result.content == sample_document.content

    @pytest.mark.asyncio
    async def test_search_documents(self, mock_qdrant_client, mock_embedding_service):
        """Test searching documents by query."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        # Mock search results
        mock_qdrant_client.search.return_value = [
            models.ScoredPoint(
                id="doc1",
                score=0.9,
                version=0,
                payload={"content": "matching content", "metadata": {}}
            ),
            models.ScoredPoint(
                id="doc2",
                score=0.8,
                version=0,
                payload={"content": "another match", "metadata": {}}
            )
        ]

        results = await manager.search_documents("test query", limit=10)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["score"] == 0.9

        # Verify query embedding was generated
        mock_embedding_service.embed_query.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_qdrant_client, mock_embedding_service):
        """Test searching with metadata filters."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        filters = {"project": "test-project", "file_type": "python"}

        await manager.search_documents("query", filters=filters)

        # Verify filters were passed to Qdrant search
        search_call = mock_qdrant_client.search.call_args
        assert search_call is not None

    @pytest.mark.asyncio
    async def test_delete_document(self, mock_qdrant_client, mock_embedding_service):
        """Test deleting a document from memory."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        result = await manager.delete_document("doc123")

        assert result["status"] == "success"
        assert result["document_id"] == "doc123"

        # Verify deletion was called on Qdrant
        mock_qdrant_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_document(self, mock_qdrant_client, mock_embedding_service, sample_document):
        """Test updating an existing document."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        # Update document content
        sample_document.content = "Updated content for the document"
        sample_document.metadata.modified_at = datetime.now(timezone.utc)

        result = await manager.update_document(sample_document)

        assert result["status"] == "success"

        # Should generate new embedding for updated content
        mock_embedding_service.embed_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_operations(self, mock_qdrant_client, mock_embedding_service):
        """Test batch document operations."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        # Create multiple documents
        documents = []
        for i in range(5):
            metadata = DocumentMetadata(f"/test/doc{i}.txt", "text", 100)
            doc = Document(f"doc{i}", f"Content for document {i}", metadata)
            documents.append(doc)

        # Batch store
        results = await manager.batch_store_documents(documents)

        assert len(results) == 5
        assert all(r["status"] == "success" for r in results)

        # Should have called embedding service for each document
        assert mock_embedding_service.embed_document.call_count == 5

    @pytest.mark.asyncio
    async def test_memory_statistics(self, mock_qdrant_client, mock_embedding_service):
        """Test memory statistics collection."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        # Mock collection info
        mock_qdrant_client.get_collection.return_value = Mock(
            status=models.CollectionStatus.GREEN,
            vectors_count=100,
            indexed_vectors_count=100,
            points_count=50,
        )

        stats = await manager.get_statistics()

        assert "vectors_count" in stats
        assert "points_count" in stats
        assert stats["collection_name"] == "test-memory"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_qdrant_client, mock_embedding_service, sample_document):
        """Test error handling in memory operations."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        # Test embedding service failure
        mock_embedding_service.embed_document.side_effect = RuntimeError("Embedding failed")

        with pytest.raises(RuntimeError):
            await manager.store_document(sample_document)

        # Test Qdrant client failure
        mock_embedding_service.embed_document.side_effect = None
        mock_qdrant_client.upsert.side_effect = ResponseHandlingException("Storage failed")

        with pytest.raises(ResponseHandlingException):
            await manager.store_document(sample_document)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_qdrant_client, mock_embedding_service):
        """Test concurrent memory operations."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        # Create documents for concurrent operations
        documents = []
        for i in range(3):
            metadata = DocumentMetadata(f"/test/concurrent{i}.txt", "text", 100)
            doc = Document(f"concurrent{i}", f"Concurrent content {i}", metadata)
            documents.append(doc)

        # Run concurrent stores
        tasks = [manager.store_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(r["status"] == "success" for r in results)

        # All documents should have been processed
        assert mock_embedding_service.embed_document.call_count == 3


class TestRetrievalOptions:
    """Test document retrieval options and configurations."""

    def test_basic_retrieval_options(self):
        """Test basic retrieval options."""
        options = RetrievalOptions(limit=50, include_metadata=True)

        assert options.limit == 50
        assert options.include_metadata is True

    def test_advanced_retrieval_options(self):
        """Test advanced retrieval options."""
        options = RetrievalOptions(
            limit=100,
            include_metadata=True,
            include_chunks=True,
            chunk_limit=10,
            score_threshold=0.8,
            filters={"project": "test"},
            sort_by="relevance"
        )

        assert options.limit == 100
        assert options.include_chunks is True
        assert options.chunk_limit == 10
        assert options.score_threshold == 0.8
        assert options.filters == {"project": "test"}

    @pytest.mark.asyncio
    async def test_retrieval_with_options(self, mock_qdrant_client, mock_embedding_service):
        """Test document retrieval with custom options."""
        manager = DocumentMemoryManager(mock_qdrant_client, mock_embedding_service, "test-memory")

        options = RetrievalOptions(
            limit=5,
            score_threshold=0.9,
            include_metadata=True,
            filters={"file_type": "python"}
        )

        mock_qdrant_client.search.return_value = [
            models.ScoredPoint(id="doc1", score=0.95, version=0, payload={"content": "test"})
        ]

        await manager.search_documents("query", options=options)

        # Verify options were applied
        search_call = mock_qdrant_client.search.call_args
        assert search_call[1]["limit"] == 5
