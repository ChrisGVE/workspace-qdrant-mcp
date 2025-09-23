"""
Comprehensive unit tests for src/python/common/core/embeddings.py module.

Tests cover the EmbeddingService class including initialization, dense and sparse
embedding generation, text chunking, document processing, and error handling.
"""

import asyncio
import hashlib
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import numpy as np

# Import the module under test
from python.common.core.embeddings import EmbeddingService
from python.common.core.config import Config, EmbeddingConfig


class TestEmbeddingService:
    """Test suite for EmbeddingService class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        config = Mock(spec=Config)
        config.embedding = Mock(spec=EmbeddingConfig)
        config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.enable_sparse_vectors = True
        config.embedding.batch_size = 10
        config.embedding.chunk_size = 1000
        config.embedding.chunk_overlap = 100
        return config

    @pytest.fixture
    def mock_config_no_sparse(self):
        """Create a mock configuration without sparse vectors."""
        config = Mock(spec=Config)
        config.embedding = Mock(spec=EmbeddingConfig)
        config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.enable_sparse_vectors = False
        config.embedding.batch_size = 10
        config.embedding.chunk_size = 1000
        config.embedding.chunk_overlap = 100
        return config

    @pytest.fixture
    def embedding_service(self, mock_config):
        """Create an EmbeddingService instance for testing."""
        return EmbeddingService(mock_config)

    def test_initialization(self, mock_config):
        """Test EmbeddingService initialization."""
        service = EmbeddingService(mock_config)

        assert service.config == mock_config
        assert service.dense_model is None
        assert service.sparse_model is None
        assert service.bm25_encoder is None
        assert service.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, embedding_service, mock_config):
        """Test successful initialization of embedding models."""
        mock_dense_model = Mock()
        mock_bm25_encoder = Mock()
        mock_bm25_encoder.initialize = AsyncMock()

        with patch('python.common.core.embeddings.TextEmbedding', return_value=mock_dense_model), \
             patch('python.common.core.embeddings.BM25SparseEncoder', return_value=mock_bm25_encoder), \
             patch('asyncio.get_event_loop') as mock_loop:

            # Mock executor to return the dense model directly
            mock_executor = AsyncMock(return_value=mock_dense_model)
            mock_loop.return_value.run_in_executor = mock_executor

            await embedding_service.initialize()

            assert embedding_service.initialized is True
            assert embedding_service.dense_model == mock_dense_model
            assert embedding_service.bm25_encoder == mock_bm25_encoder
            mock_bm25_encoder.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_no_sparse(self, mock_config_no_sparse):
        """Test initialization without sparse vectors."""
        service = EmbeddingService(mock_config_no_sparse)
        mock_dense_model = Mock()

        with patch('python.common.core.embeddings.TextEmbedding', return_value=mock_dense_model), \
             patch('asyncio.get_event_loop') as mock_loop:

            mock_executor = AsyncMock(return_value=mock_dense_model)
            mock_loop.return_value.run_in_executor = mock_executor

            await service.initialize()

            assert service.initialized is True
            assert service.dense_model == mock_dense_model
            assert service.bm25_encoder is None

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, embedding_service):
        """Test initialization when already initialized."""
        embedding_service.initialized = True

        with patch('python.common.core.embeddings.TextEmbedding') as mock_text_embedding:
            await embedding_service.initialize()
            mock_text_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, embedding_service):
        """Test initialization failure handling."""
        with patch('python.common.core.embeddings.TextEmbedding', side_effect=Exception("Model load failed")):
            with pytest.raises(RuntimeError, match="Failed to initialize embedding models"):
                await embedding_service.initialize()

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized(self, embedding_service):
        """Test generate_embeddings when service not initialized."""
        with pytest.raises(RuntimeError, match="EmbeddingService must be initialized first"):
            await embedding_service.generate_embeddings("test text")

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_text(self, embedding_service):
        """Test generate_embeddings with empty text."""
        embedding_service.initialized = True

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.generate_embeddings("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.generate_embeddings("   ")

    @pytest.mark.asyncio
    async def test_generate_embeddings_dense_only(self, embedding_service, mock_config_no_sparse):
        """Test generating dense embeddings only."""
        service = EmbeddingService(mock_config_no_sparse)
        service.initialized = True

        dense_vector = [0.1, 0.2, 0.3]
        service._generate_dense_embeddings = AsyncMock(return_value=[dense_vector])

        result = await service.generate_embeddings("test text")

        assert result == {"dense": dense_vector}
        service._generate_dense_embeddings.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_sparse(self, embedding_service):
        """Test generating both dense and sparse embeddings."""
        embedding_service.initialized = True
        embedding_service.bm25_encoder = Mock()

        dense_vector = [0.1, 0.2, 0.3]
        sparse_vector = {"indices": [1, 2], "values": [0.5, 0.8]}

        embedding_service._generate_dense_embeddings = AsyncMock(return_value=[dense_vector])
        embedding_service._generate_sparse_embeddings = AsyncMock(return_value=[sparse_vector])

        result = await embedding_service.generate_embeddings("test text", include_sparse=True)

        expected = {
            "dense": dense_vector,
            "sparse": sparse_vector
        }
        assert result == expected

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_empty(self, embedding_service):
        """Test batch generation with empty list."""
        embedding_service.initialized = True

        result = await embedding_service.generate_embeddings_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_not_initialized(self, embedding_service):
        """Test batch generation when not initialized."""
        with pytest.raises(RuntimeError, match="EmbeddingService must be initialized first"):
            await embedding_service.generate_embeddings_batch(["text1", "text2"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_success(self, embedding_service):
        """Test successful batch embeddings generation."""
        embedding_service.initialized = True
        embedding_service.bm25_encoder = Mock()

        texts = ["text1", "text2"]
        dense_vectors = [[0.1, 0.2], [0.3, 0.4]]
        sparse_vectors = [{"indices": [1], "values": [0.5]}, {"indices": [2], "values": [0.8]}]

        embedding_service._generate_dense_embeddings = AsyncMock(return_value=dense_vectors)
        embedding_service._generate_sparse_embeddings = AsyncMock(return_value=sparse_vectors)

        result = await embedding_service.generate_embeddings_batch(texts, include_sparse=True)

        expected = [
            {"dense": [0.1, 0.2], "sparse": {"indices": [1], "values": [0.5]}},
            {"dense": [0.3, 0.4], "sparse": {"indices": [2], "values": [0.8]}}
        ]
        assert result == expected

    @pytest.mark.asyncio
    async def test_generate_dense_embeddings(self, embedding_service):
        """Test dense embeddings generation."""
        mock_model = Mock()
        embedding_service.dense_model = mock_model

        # Mock numpy array response
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed.return_value = [mock_embedding]

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(return_value=[mock_embedding])
            mock_loop.return_value.run_in_executor = mock_executor

            result = await embedding_service._generate_dense_embeddings(["test text"])

            assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    async def test_generate_dense_embeddings_plain_list(self, embedding_service):
        """Test dense embeddings generation with plain list (no tolist method)."""
        mock_model = Mock()
        embedding_service.dense_model = mock_model

        plain_embedding = [0.1, 0.2, 0.3]
        mock_model.embed.return_value = [plain_embedding]

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(return_value=[plain_embedding])
            mock_loop.return_value.run_in_executor = mock_executor

            result = await embedding_service._generate_dense_embeddings(["test text"])

            assert result == [[0.1, 0.2, 0.3]]

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_single(self, embedding_service):
        """Test sparse embeddings generation for single text."""
        mock_encoder = Mock()
        embedding_service.bm25_encoder = mock_encoder

        sparse_vector = {"indices": [1, 2], "values": [0.5, 0.8]}
        mock_encoder.encode.return_value = sparse_vector

        result = await embedding_service._generate_sparse_embeddings(["test text"])

        assert result == [sparse_vector]
        mock_encoder.encode.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_batch(self, embedding_service):
        """Test sparse embeddings generation for multiple texts."""
        mock_encoder = Mock()
        embedding_service.bm25_encoder = mock_encoder

        texts = ["text1", "text2"]
        sparse_vectors = [{"indices": [1], "values": [0.5]}, {"indices": [2], "values": [0.8]}]
        mock_encoder.encode.side_effect = sparse_vectors

        result = await embedding_service._generate_sparse_embeddings(texts)

        assert result == sparse_vectors
        assert mock_encoder.encode.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_documents_empty(self, embedding_service):
        """Test embedding empty document list."""
        result = await embedding_service.embed_documents([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_documents_success(self, embedding_service, mock_config):
        """Test successful document embedding."""
        documents = [
            {"content": "doc1", "id": 1},
            {"content": "doc2", "id": 2}
        ]

        embeddings_response = {
            "dense": [[0.1, 0.2], [0.3, 0.4]],
            "sparse": [{"indices": [1], "values": [0.5]}, {"indices": [2], "values": [0.8]}]
        }

        embedding_service.generate_embeddings = AsyncMock(return_value=embeddings_response)

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.time.return_value = 123456.789

            result = await embedding_service.embed_documents(documents)

            assert len(result) == 2
            for i, doc in enumerate(result):
                assert doc["id"] == documents[i]["id"]
                assert doc["dense_vector"] == embeddings_response["dense"][i]
                assert doc["sparse_vector"] == embeddings_response["sparse"][i]
                assert doc["embedding_model"] == mock_config.embedding.model
                assert doc["embedding_timestamp"] == 123456.789
                assert "content_hash" in doc

    def test_chunk_text_short_text(self, embedding_service):
        """Test chunking text that's shorter than chunk size."""
        text = "Short text"
        chunks = embedding_service.chunk_text(text, chunk_size=1000)

        assert chunks == ["Short text"]

    def test_chunk_text_long_text(self, embedding_service):
        """Test chunking long text."""
        text = "This is a very long text. " * 50  # Creates ~1300 characters
        chunks = embedding_service.chunk_text(text, chunk_size=500, chunk_overlap=50)

        assert len(chunks) > 1
        assert all(len(chunk) <= 500 for chunk in chunks)
        assert all(chunk.strip() for chunk in chunks)  # No empty chunks

    def test_chunk_text_with_separators(self, embedding_service):
        """Test chunking with specific separators."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = embedding_service.chunk_text(text, chunk_size=30, separators=["\n\n", ". "])

        assert len(chunks) >= 2
        # Should break at paragraph boundaries when possible

    def test_chunk_text_force_break(self, embedding_service):
        """Test chunking when forced to break mid-word."""
        text = "verylongwordwithoutspacesorpunctuation" * 10
        chunks = embedding_service.chunk_text(text, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1
        assert all(len(chunk) <= 50 for chunk in chunks)

    def test_hash_content(self, embedding_service):
        """Test content hashing."""
        content = "test content"
        hash1 = embedding_service._hash_content(content)
        hash2 = embedding_service._hash_content(content)
        hash3 = embedding_service._hash_content("different content")

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA256 hex length

    def test_generate_cache_key(self, embedding_service):
        """Test cache key generation."""
        key1 = embedding_service._generate_cache_key("test", True)
        key2 = embedding_service._generate_cache_key("test", True)
        key3 = embedding_service._generate_cache_key("test", False)

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 64  # SHA256 hex length

    def test_preprocess_text(self, embedding_service):
        """Test text preprocessing."""
        # Test whitespace normalization
        assert embedding_service._preprocess_text("  hello   world  ") == "hello world"

        # Test empty text
        assert embedding_service._preprocess_text("") == ""
        assert embedding_service._preprocess_text("   ") == ""

        # Test unicode whitespace removal
        text_with_unicode = "hello\u00a0world\u2000test"
        result = embedding_service._preprocess_text(text_with_unicode)
        assert result == "hello world test"

    def test_get_vector_size_no_model(self, embedding_service):
        """Test getting vector size when no model is loaded."""
        assert embedding_service._get_vector_size() is None

    def test_get_vector_size_with_model(self, embedding_service):
        """Test getting vector size with loaded model."""
        mock_model = Mock()
        mock_embedding = [0.1, 0.2, 0.3, 0.4]
        mock_model.embed.return_value = [mock_embedding]
        embedding_service.dense_model = mock_model

        size = embedding_service._get_vector_size()
        assert size == 4

    def test_get_vector_size_exception(self, embedding_service):
        """Test getting vector size when model throws exception."""
        mock_model = Mock()
        mock_model.embed.side_effect = Exception("Model error")
        embedding_service.dense_model = mock_model

        size = embedding_service._get_vector_size()
        assert size is None

    def test_get_model_info_basic(self, embedding_service, mock_config):
        """Test getting model information."""
        embedding_service.initialized = True
        embedding_service.dense_model = Mock()

        info = embedding_service.get_model_info()

        assert info["model_name"] == mock_config.embedding.model
        assert info["initialized"] is True
        assert info["sparse_enabled"] == mock_config.embedding.enable_sparse_vectors
        assert "dense_model" in info
        assert "sparse_model" in info
        assert "config" in info

    def test_get_model_info_with_bm25(self, embedding_service, mock_config):
        """Test getting model information with BM25 encoder."""
        embedding_service.initialized = True
        embedding_service.dense_model = Mock()

        mock_bm25 = Mock()
        mock_bm25.get_model_info.return_value = {"vocabulary_size": 10000}
        embedding_service.bm25_encoder = mock_bm25

        info = embedding_service.get_model_info()

        assert "vocabulary_size" in info["sparse_model"]
        mock_bm25.get_model_info.assert_called_once()

    def test_get_model_info_dimensions(self, embedding_service, mock_config):
        """Test model dimension detection for different models."""
        test_cases = [
            ("sentence-transformers/all-MiniLM-L6-v2", 384),
            ("BAAI/bge-base-en-v1.5", 768),
            ("BAAI/bge-large-en-v1.5", 1024),
            ("unknown-model", 384)  # default
        ]

        for model_name, expected_dim in test_cases:
            mock_config.embedding.model = model_name
            service = EmbeddingService(mock_config)
            service.initialized = True
            service.dense_model = Mock()

            info = service.get_model_info()
            assert info["dense_model"]["dimensions"] == expected_dim

    @pytest.mark.asyncio
    async def test_close(self, embedding_service):
        """Test cleanup method."""
        embedding_service.dense_model = Mock()
        embedding_service.sparse_model = Mock()
        embedding_service.bm25_encoder = Mock()
        embedding_service.initialized = True

        await embedding_service.close()

        assert embedding_service.dense_model is None
        assert embedding_service.sparse_model is None
        assert embedding_service.bm25_encoder is None
        assert embedding_service.initialized is False

    @pytest.mark.asyncio
    async def test_async_context_manager(self, embedding_service):
        """Test async context manager functionality."""
        embedding_service.initialize = AsyncMock()
        embedding_service.close = AsyncMock()

        async with embedding_service as service:
            assert service == embedding_service
            embedding_service.initialize.assert_called_once()

        embedding_service.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_exception_handling(self, embedding_service):
        """Test exception handling in generate_embeddings."""
        embedding_service.initialized = True
        embedding_service._generate_dense_embeddings = AsyncMock(side_effect=Exception("Dense error"))

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            await embedding_service.generate_embeddings("test text")

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_exception_handling(self, embedding_service):
        """Test exception handling in batch generation."""
        embedding_service.initialized = True
        embedding_service._generate_dense_embeddings = AsyncMock(side_effect=Exception("Batch error"))

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            await embedding_service.generate_embeddings_batch(["text1", "text2"])

    @pytest.mark.asyncio
    async def test_generate_dense_embeddings_exception(self, embedding_service):
        """Test exception handling in dense embeddings generation."""
        mock_model = Mock()
        mock_model.embed.side_effect = Exception("Model error")
        embedding_service.dense_model = mock_model

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock(side_effect=Exception("Model error"))
            mock_loop.return_value.run_in_executor = mock_executor

            with pytest.raises(Exception, match="Model error"):
                await embedding_service._generate_dense_embeddings(["test"])

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_exception(self, embedding_service):
        """Test exception handling in sparse embeddings generation."""
        mock_encoder = Mock()
        mock_encoder.encode.side_effect = Exception("Encoder error")
        embedding_service.bm25_encoder = mock_encoder

        with pytest.raises(Exception, match="Encoder error"):
            await embedding_service._generate_sparse_embeddings(["test"])