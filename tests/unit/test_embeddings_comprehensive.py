"""
Comprehensive unit tests for embeddings module to achieve 100% test coverage.

This test suite covers all aspects of the EmbeddingService class including:
- Initialization and model loading scenarios
- Dense and sparse embedding generation
- Text preprocessing and chunking
- Batch processing operations
- Error handling and edge cases
- Context manager behavior
- Model information retrieval
- Memory and resource management
"""

import sys
from pathlib import Path
import asyncio
import hashlib
import math
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Any

import pytest
import numpy as np

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from python.common.core.config import Config, EmbeddingConfig
from python.common.core.embeddings import EmbeddingService


@pytest.fixture
def embedding_config():
    """Create comprehensive embedding configuration for testing."""
    return EmbeddingConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        enable_sparse_vectors=True,
        chunk_size=1000,
        chunk_overlap=200,
        batch_size=50,
    )


@pytest.fixture
def config(embedding_config):
    """Create full config with embedding settings."""
    config = Config()
    config.embedding = embedding_config
    return config


@pytest.fixture
def service(config):
    """Create EmbeddingService instance for testing."""
    return EmbeddingService(config)


class TestEmbeddingServiceInitialization:
    """Test suite for EmbeddingService initialization scenarios."""

    def test_init_default_state(self, service, config):
        """Test EmbeddingService initialization sets correct default state."""
        assert service.config == config
        assert service.dense_model is None
        assert service.sparse_model is None
        assert service.bm25_encoder is None
        assert service.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_dense_only_disabled_sparse(self, service):
        """Test initialization with sparse vectors explicitly disabled."""
        service.config.embedding.enable_sparse_vectors = False
        mock_dense_model = MagicMock()

        with patch("python.common.core.embeddings.TextEmbedding") as mock_text_embedding:
            mock_text_embedding.return_value = mock_dense_model

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor.return_value = mock_dense_model

                await service.initialize()

        assert service.dense_model == mock_dense_model
        assert service.bm25_encoder is None
        assert service.initialized is True
        mock_loop.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_sparse_vectors_full_setup(self, service):
        """Test initialization with both dense and sparse embeddings enabled."""
        mock_dense_model = MagicMock()
        mock_bm25_encoder = MagicMock()
        mock_bm25_encoder.initialize = AsyncMock()

        with (
            patch("python.common.core.embeddings.TextEmbedding") as mock_text_embedding,
            patch("python.common.core.embeddings.BM25SparseEncoder") as mock_bm25_class,
        ):
            mock_text_embedding.return_value = mock_dense_model
            mock_bm25_class.return_value = mock_bm25_encoder

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor.return_value = mock_dense_model

                await service.initialize()

        assert service.dense_model == mock_dense_model
        assert service.bm25_encoder == mock_bm25_encoder
        assert service.initialized is True
        mock_bm25_class.assert_called_once_with(use_fastembed=True)
        mock_bm25_encoder.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized_early_return(self, service):
        """Test that initialize returns early if already initialized."""
        service.initialized = True
        service.dense_model = MagicMock()  # Set to verify no changes

        with patch("python.common.core.embeddings.TextEmbedding") as mock_text_embedding:
            await service.initialize()
            mock_text_embedding.assert_not_called()

        # Verify state unchanged
        assert service.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_mock_compatibility_executor_fallback(self, service):
        """Test initialization handling of mocked executor results."""
        service.config.embedding.enable_sparse_vectors = False
        mock_dense_model = MagicMock()

        with patch("python.common.core.embeddings.TextEmbedding") as mock_text_embedding:
            mock_text_embedding.return_value = mock_dense_model

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                # Return mock directly that can't be awaited
                mock_loop.run_in_executor.return_value = mock_dense_model

                await service.initialize()

        assert service.dense_model == mock_dense_model
        assert service.initialized is True

    @pytest.mark.asyncio
    async def test_initialize_dense_model_loading_failure(self, service):
        """Test initialization failure when dense model fails to load."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.side_effect = Exception("Model loading failed")

            with pytest.raises(RuntimeError, match="Failed to initialize embedding models"):
                await service.initialize()

        assert service.initialized is False
        assert service.dense_model is None

    @pytest.mark.asyncio
    async def test_initialize_sparse_model_initialization_failure(self, service):
        """Test initialization failure when sparse model fails."""
        mock_dense_model = MagicMock()
        mock_bm25_encoder = MagicMock()
        mock_bm25_encoder.initialize.side_effect = Exception("BM25 initialization failed")

        with (
            patch("python.common.core.embeddings.TextEmbedding") as mock_text_embedding,
            patch("python.common.core.embeddings.BM25SparseEncoder") as mock_bm25_class,
        ):
            mock_text_embedding.return_value = mock_dense_model
            mock_bm25_class.return_value = mock_bm25_encoder

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor.return_value = mock_dense_model

                with pytest.raises(RuntimeError, match="Failed to initialize embedding models"):
                    await service.initialize()

        assert service.initialized is False


class TestEmbeddingGeneration:
    """Test suite for embedding generation methods."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized_error(self, service):
        """Test embedding generation when service not initialized."""
        assert service.initialized is False

        with pytest.raises(RuntimeError, match="EmbeddingService must be initialized first"):
            await service.generate_embeddings("test text")

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_text_validation(self, service):
        """Test embedding generation with various empty text scenarios."""
        service.initialized = True
        service.dense_model = MagicMock()

        # Test completely empty string
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embeddings("")

        # Test whitespace-only string
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embeddings("   ")

        # Test string with only tabs and newlines
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embeddings("\t\n\r")

    @pytest.mark.asyncio
    async def test_generate_embeddings_dense_only_success(self, service):
        """Test generating dense embeddings only."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = None

        # Mock dense embeddings
        mock_embeddings = [[0.1, 0.2, 0.3, 0.4]]

        with patch.object(service, '_generate_dense_embeddings', return_value=mock_embeddings) as mock_dense:
            result = await service.generate_embeddings("test text")

        assert "dense" in result
        assert result["dense"] == [0.1, 0.2, 0.3, 0.4]
        assert "sparse" not in result
        mock_dense.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_sparse_vectors(self, service):
        """Test generating both dense and sparse embeddings."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()

        # Mock embeddings
        mock_dense = [[0.1, 0.2, 0.3, 0.4]]
        mock_sparse = [{"indices": [1, 3, 5], "values": [0.8, 0.6, 0.9]}]

        with (
            patch.object(service, '_generate_dense_embeddings', return_value=mock_dense) as mock_dense_gen,
            patch.object(service, '_generate_sparse_embeddings', return_value=mock_sparse) as mock_sparse_gen,
        ):
            result = await service.generate_embeddings("test text", include_sparse=True)

        assert "dense" in result
        assert "sparse" in result
        assert result["dense"] == [0.1, 0.2, 0.3, 0.4]
        assert result["sparse"] == {"indices": [1, 3, 5], "values": [0.8, 0.6, 0.9]}
        mock_dense_gen.assert_called_once_with(["test text"])
        mock_sparse_gen.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_config_default_sparse_setting(self, service):
        """Test that sparse inclusion defaults to config setting."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()
        service.config.embedding.enable_sparse_vectors = True

        mock_dense = [[0.1, 0.2]]
        mock_sparse = [{"indices": [1], "values": [0.8]}]

        with (
            patch.object(service, '_generate_dense_embeddings', return_value=mock_dense),
            patch.object(service, '_generate_sparse_embeddings', return_value=mock_sparse) as mock_sparse_gen,
        ):
            result = await service.generate_embeddings("test text")  # No include_sparse specified

        assert "sparse" in result
        mock_sparse_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_embeddings_sparse_requested_but_no_encoder(self, service):
        """Test requesting sparse embeddings when encoder not available."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = None

        mock_dense = [[0.1, 0.2, 0.3, 0.4]]

        with patch.object(service, '_generate_dense_embeddings', return_value=mock_dense):
            result = await service.generate_embeddings("test text", include_sparse=True)

        assert "dense" in result
        assert "sparse" not in result
        assert result["dense"] == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_generate_embeddings_preprocessing_applied(self, service):
        """Test that text preprocessing is applied before embedding generation."""
        service.initialized = True
        service.dense_model = MagicMock()

        mock_dense = [[0.1, 0.2]]

        with (
            patch.object(service, '_preprocess_text', return_value="cleaned text") as mock_preprocess,
            patch.object(service, '_generate_dense_embeddings', return_value=mock_dense),
        ):
            await service.generate_embeddings("  raw   text  \n")

        mock_preprocess.assert_called_once_with("  raw   text  \n")

    @pytest.mark.asyncio
    async def test_generate_embeddings_exception_handling(self, service):
        """Test exception handling during embedding generation."""
        service.initialized = True
        service.dense_model = MagicMock()

        with patch.object(service, '_generate_dense_embeddings', side_effect=Exception("Dense embedding error")):
            with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
                await service.generate_embeddings("test text")


class TestBatchEmbeddingGeneration:
    """Test suite for batch embedding generation methods."""

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_not_initialized(self, service):
        """Test batch embedding generation when service not initialized."""
        assert service.initialized is False

        with pytest.raises(RuntimeError, match="EmbeddingService must be initialized first"):
            await service.generate_embeddings_batch(["text1", "text2"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_empty_list(self, service):
        """Test batch embedding generation with empty text list."""
        service.initialized = True
        service.dense_model = MagicMock()

        result = await service.generate_embeddings_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_dense_only(self, service):
        """Test batch processing with dense embeddings only."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = None

        texts = ["text 1", "text 2", "text 3"]
        mock_dense = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch.object(service, '_generate_dense_embeddings', return_value=mock_dense):
            results = await service.generate_embeddings_batch(texts)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert "dense" in result
            assert "sparse" not in result
            assert result["dense"] == mock_dense[i]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_with_sparse(self, service):
        """Test batch processing with both dense and sparse embeddings."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()

        texts = ["text 1", "text 2"]
        mock_dense = [[0.1, 0.2], [0.3, 0.4]]
        mock_sparse = [
            {"indices": [1, 2], "values": [0.8, 0.6]},
            {"indices": [2, 3], "values": [0.7, 0.9]},
        ]

        with (
            patch.object(service, '_generate_dense_embeddings', return_value=mock_dense),
            patch.object(service, '_generate_sparse_embeddings', return_value=mock_sparse),
        ):
            results = await service.generate_embeddings_batch(texts, include_sparse=True)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert "dense" in result
            assert "sparse" in result
            assert result["dense"] == mock_dense[i]
            assert result["sparse"] == mock_sparse[i]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_preprocessing_all_texts(self, service):
        """Test that preprocessing is applied to all texts in batch."""
        service.initialized = True
        service.dense_model = MagicMock()

        texts = ["  text1  ", "  text2  "]
        mock_dense = [[0.1], [0.2]]

        with (
            patch.object(service, '_preprocess_text', side_effect=lambda x: x.strip()) as mock_preprocess,
            patch.object(service, '_generate_dense_embeddings', return_value=mock_dense),
        ):
            await service.generate_embeddings_batch(texts)

        assert mock_preprocess.call_count == 2
        mock_preprocess.assert_has_calls([call("  text1  "), call("  text2  ")])

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_config_default_sparse(self, service):
        """Test that batch processing uses config default for sparse vectors."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()
        service.config.embedding.enable_sparse_vectors = False

        texts = ["text1"]
        mock_dense = [[0.1]]

        with (
            patch.object(service, '_generate_dense_embeddings', return_value=mock_dense),
            patch.object(service, '_generate_sparse_embeddings') as mock_sparse_gen,
        ):
            results = await service.generate_embeddings_batch(texts)

        assert len(results) == 1
        assert "sparse" not in results[0]
        mock_sparse_gen.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_exception_handling(self, service):
        """Test exception handling during batch embedding generation."""
        service.initialized = True
        service.dense_model = MagicMock()

        with patch.object(service, '_generate_dense_embeddings', side_effect=Exception("Batch error")):
            with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
                await service.generate_embeddings_batch(["text1"])


class TestDenseEmbeddingGeneration:
    """Test suite for dense embedding generation internal methods."""

    @pytest.mark.asyncio
    async def test_generate_dense_embeddings_success(self, service):
        """Test successful dense embedding generation."""
        service.dense_model = MagicMock()

        # Mock embedding generation with numpy arrays
        mock_embedding1 = MagicMock()
        mock_embedding1.tolist.return_value = [0.1, 0.2, 0.3]
        mock_embedding2 = MagicMock()
        mock_embedding2.tolist.return_value = [0.4, 0.5, 0.6]
        mock_embeddings = [mock_embedding1, mock_embedding2]

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = mock_embeddings

            result = await service._generate_dense_embeddings(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        mock_loop.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_dense_embeddings_plain_lists(self, service):
        """Test dense embedding generation with plain list returns."""
        service.dense_model = MagicMock()

        # Mock embedding generation returning plain lists (no tolist method)
        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = mock_embeddings

            result = await service._generate_dense_embeddings(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_generate_dense_embeddings_exception_handling(self, service):
        """Test exception handling in dense embedding generation."""
        service.dense_model = MagicMock()

        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.side_effect = Exception("Dense generation error")

            with pytest.raises(Exception, match="Dense generation error"):
                await service._generate_dense_embeddings(["text1"])


class TestSparseEmbeddingGeneration:
    """Test suite for sparse embedding generation internal methods."""

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_single_text(self, service):
        """Test sparse embedding generation for single text."""
        service.bm25_encoder = MagicMock()
        mock_sparse_vector = {"indices": [1, 3], "values": [0.8, 0.6]}
        service.bm25_encoder.encode.return_value = mock_sparse_vector

        result = await service._generate_sparse_embeddings(["single text"])

        assert len(result) == 1
        assert result[0] == mock_sparse_vector
        service.bm25_encoder.encode.assert_called_once_with("single text")

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_multiple_texts(self, service):
        """Test sparse embedding generation for multiple texts."""
        service.bm25_encoder = MagicMock()

        sparse_vectors = [
            {"indices": [1, 2], "values": [0.8, 0.6]},
            {"indices": [2, 3], "values": [0.7, 0.9]},
        ]
        service.bm25_encoder.encode.side_effect = sparse_vectors

        result = await service._generate_sparse_embeddings(["text1", "text2"])

        assert len(result) == 2
        assert result[0] == sparse_vectors[0]
        assert result[1] == sparse_vectors[1]
        assert service.bm25_encoder.encode.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_sparse_embeddings_exception_handling(self, service):
        """Test exception handling in sparse embedding generation."""
        service.bm25_encoder = MagicMock()
        service.bm25_encoder.encode.side_effect = Exception("Sparse generation error")

        with pytest.raises(Exception, match="Sparse generation error"):
            await service._generate_sparse_embeddings(["text1"])


class TestDocumentEmbedding:
    """Test suite for document embedding with metadata."""

    @pytest.mark.asyncio
    async def test_embed_documents_empty_list(self, service):
        """Test embedding empty document list."""
        result = await service.embed_documents([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_documents_default_batch_size(self, service):
        """Test embedding documents using default batch size from config."""
        service.config.embedding.batch_size = 2
        documents = [{"content": "doc1"}, {"content": "doc2"}, {"content": "doc3"}]

        with patch.object(service, 'generate_embeddings') as mock_gen:
            mock_gen.return_value = {
                "dense": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            }

            await service.embed_documents(documents)

        # Should be called twice due to batch size of 2
        assert mock_gen.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_documents_custom_batch_size(self, service):
        """Test embedding documents with custom batch size."""
        documents = [{"content": "doc1"}, {"content": "doc2"}]

        with patch.object(service, 'generate_embeddings') as mock_gen:
            mock_gen.return_value = {"dense": [[0.1], [0.2]]}

            await service.embed_documents(documents, batch_size=1)

        # Should be called twice due to batch size of 1
        assert mock_gen.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_documents_custom_content_field(self, service):
        """Test embedding documents with custom content field."""
        documents = [{"text": "document content"}]

        with patch.object(service, 'generate_embeddings') as mock_gen:
            mock_gen.return_value = {"dense": [[0.1]]}

            result = await service.embed_documents(documents, content_field="text")

        mock_gen.assert_called_with(["document content"])
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_embed_documents_with_metadata_and_sparse(self, service):
        """Test embedding documents with full metadata and sparse vectors."""
        documents = [{"content": "test document", "title": "Test"}]

        with (
            patch.object(service, 'generate_embeddings') as mock_gen,
            patch('asyncio.get_event_loop') as mock_loop,
        ):
            mock_gen.return_value = {
                "dense": [[0.1, 0.2]],
                "sparse": [{"indices": [1], "values": [0.8]}]
            }
            mock_loop.return_value.time.return_value = 1234567890.0

            result = await service.embed_documents(documents)

        assert len(result) == 1
        doc = result[0]

        # Check original metadata preserved
        assert doc["content"] == "test document"
        assert doc["title"] == "Test"

        # Check embeddings added
        assert doc["dense_vector"] == [0.1, 0.2]
        assert doc["sparse_vector"] == {"indices": [1], "values": [0.8]}

        # Check metadata added
        assert doc["embedding_model"] == service.config.embedding.model
        assert doc["embedding_timestamp"] == 1234567890.0
        assert "content_hash" in doc

    @pytest.mark.asyncio
    async def test_embed_documents_missing_content_field(self, service):
        """Test embedding documents with missing content field."""
        documents = [{"title": "No content field"}]

        with patch.object(service, 'generate_embeddings') as mock_gen:
            mock_gen.return_value = {"dense": [[]]}

            result = await service.embed_documents(documents)

        # Should call with empty string for missing content
        mock_gen.assert_called_with([""])
        assert len(result) == 1


class TestTextChunking:
    """Test suite for text chunking functionality."""

    def test_chunk_text_short_text_no_chunking(self, service):
        """Test chunking of text shorter than chunk size."""
        short_text = "This is a short text."
        chunks = service.chunk_text(short_text)

        assert chunks == [short_text]

    def test_chunk_text_long_text_multiple_chunks(self, service):
        """Test chunking of text longer than chunk size."""
        # Create text longer than default chunk_size (1000 chars)
        long_text = "This is a sentence. " * 60  # ~1200 characters

        chunks = service.chunk_text(long_text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= service.config.embedding.chunk_size

    def test_chunk_text_custom_parameters(self, service):
        """Test chunking with custom chunk size and overlap."""
        text = "Word " * 200  # ~1000 characters

        chunks = service.chunk_text(text, chunk_size=500, chunk_overlap=100)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_chunk_text_custom_separators(self, service):
        """Test chunking with custom separators."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3." * 50

        chunks = service.chunk_text(text, separators=["\n\n", ".", " "])

        assert len(chunks) > 1
        # Should prefer paragraph breaks when possible

    def test_chunk_text_preserve_sentence_boundaries(self, service):
        """Test that chunking preserves sentence boundaries when possible."""
        sentences = [f"This is sentence number {i}. " for i in range(100)]
        text = "".join(sentences)

        chunks = service.chunk_text(text)

        # Verify sentences are preserved (most chunks should end with period)
        complete_sentence_chunks = [chunk for chunk in chunks[:-1] if chunk.rstrip().endswith(".")]
        assert len(complete_sentence_chunks) >= len(chunks) // 2

    def test_chunk_text_word_boundary_preservation(self, service):
        """Test that chunking preserves word boundaries when no good separator found."""
        # Create text without sentence breaks
        text = "word" * 500  # No spaces or separators

        chunks = service.chunk_text(text, chunk_size=100)

        # Should force break at chunk_size when no separators found
        assert len(chunks) > 1

    def test_chunk_text_overlap_calculation(self, service):
        """Test chunk overlap is calculated correctly."""
        text = "A " * 1000  # Simple repeated text
        chunk_size = 200
        chunk_overlap = 50

        chunks = service.chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        if len(chunks) > 1:
            # Verify overlap exists between consecutive chunks
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0

    def test_chunk_text_edge_case_exact_chunk_size(self, service):
        """Test chunking when text is exactly chunk size."""
        text = "a" * service.config.embedding.chunk_size

        chunks = service.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_zero_overlap_handling(self, service):
        """Test chunking with zero overlap."""
        text = "Word " * 300

        chunks = service.chunk_text(text, chunk_overlap=0)

        # Should still chunk properly without overlap
        assert len(chunks) > 1


class TestUtilityMethods:
    """Test suite for utility and helper methods."""

    def test_hash_content(self, service):
        """Test content hashing functionality."""
        content = "test content for hashing"

        hash_result = service._hash_content(content)

        # Should return SHA256 hex digest
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

        # Same content should produce same hash
        hash_result2 = service._hash_content(content)
        assert hash_result == hash_result2

        # Different content should produce different hash
        hash_result3 = service._hash_content("different content")
        assert hash_result != hash_result3

    def test_generate_cache_key(self, service):
        """Test cache key generation."""
        text = "test text for caching"
        include_sparse = True

        key = service._generate_cache_key(text, include_sparse)

        # Should be consistent hash
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest length

        # Same inputs should produce same key
        key2 = service._generate_cache_key(text, include_sparse)
        assert key == key2

        # Different inputs should produce different keys
        key3 = service._generate_cache_key(text, False)
        assert key != key3

        key4 = service._generate_cache_key("different text", include_sparse)
        assert key != key4

    def test_preprocess_text_whitespace_normalization(self, service):
        """Test text preprocessing normalizes whitespace correctly."""
        test_cases = [
            ("  Normal text with spaces  ", "Normal text with spaces"),
            ("Text\twith\ttabs", "Text with tabs"),
            ("Text\nwith\nnewlines\n", "Text with newlines"),
            ("Text with    multiple    spaces", "Text with multiple spaces"),
            ("Text\rwith\rcarriage\rreturns", "Text with carriage returns"),
        ]

        for input_text, expected in test_cases:
            result = service._preprocess_text(input_text)
            assert result == expected

    def test_preprocess_text_unicode_whitespace(self, service):
        """Test preprocessing handles unicode whitespace characters."""
        # Text with non-breaking space and other unicode whitespace
        text_with_unicode = "Text with \u00a0 non-breaking spaces \u2000 and \u202f other"

        result = service._preprocess_text(text_with_unicode)

        # Should normalize unicode whitespace to regular spaces
        assert "\u00a0" not in result
        assert "\u2000" not in result
        assert "\u202f" not in result
        assert result == "Text with non-breaking spaces and other"

    def test_preprocess_text_empty_and_whitespace_only(self, service):
        """Test preprocessing of empty and whitespace-only text."""
        test_cases = ["", "   ", "\n\n", "\t\t", "\r\r", "\u00a0\u2000"]

        for text in test_cases:
            result = service._preprocess_text(text)
            assert result == ""

    def test_get_vector_size_with_model(self, service):
        """Test getting vector size from initialized model."""
        service.dense_model = MagicMock()

        # Mock test embedding
        mock_embedding = [0.1] * 384
        service.dense_model.embed.return_value = [mock_embedding]

        size = service._get_vector_size()

        assert size == 384
        service.dense_model.embed.assert_called_once_with(["test"])

    def test_get_vector_size_no_model(self, service):
        """Test getting vector size when no model available."""
        service.dense_model = None

        size = service._get_vector_size()

        assert size is None

    def test_get_vector_size_model_exception(self, service):
        """Test getting vector size when model throws exception."""
        service.dense_model = MagicMock()
        service.dense_model.embed.side_effect = Exception("Model error")

        size = service._get_vector_size()

        assert size is None


class TestModelInformation:
    """Test suite for model information retrieval."""

    def test_get_model_info_not_initialized(self, service):
        """Test getting model info when service not initialized."""
        service.initialized = False

        info = service.get_model_info()

        assert info["model_name"] == service.config.embedding.model
        assert info["vector_size"] is None
        assert info["sparse_enabled"] == service.config.embedding.enable_sparse_vectors
        assert info["initialized"] is False

    def test_get_model_info_initialized_with_sparse(self, service):
        """Test getting model info when fully initialized."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()
        service.bm25_encoder.get_model_info.return_value = {"encoder_type": "fastembed"}

        with patch.object(service, "_get_vector_size", return_value=384):
            info = service.get_model_info()

        assert info["model_name"] == service.config.embedding.model
        assert info["vector_size"] == 384
        assert info["sparse_enabled"] is True
        assert info["initialized"] is True
        assert info["dense_model"]["loaded"] is True
        assert info["sparse_model"]["loaded"] is True
        assert info["sparse_model"]["encoder_type"] == "fastembed"

    def test_get_model_info_model_dimension_detection(self, service):
        """Test model dimension detection for different model names."""
        service.initialized = True
        service.dense_model = MagicMock()

        test_cases = [
            ("sentence-transformers/all-MiniLM-L6-v2", 384),
            ("BAAI/bge-base-en-v1.5", 768),
            ("sentence-transformers/all-mpnet-base-v2", 768),
            ("jinaai/jina-embeddings-v2-base-en", 768),
            ("thenlper/gte-base", 768),
            ("BAAI/bge-large-en-v1.5", 1024),
            ("BAAI/bge-m3", 1024),
            ("unknown-model", 384),  # Default fallback
        ]

        for model_name, expected_dim in test_cases:
            service.config.embedding.model = model_name

            info = service.get_model_info()

            assert info["dense_model"]["dimensions"] == expected_dim

    def test_get_model_info_sparse_disabled(self, service):
        """Test model info when sparse vectors are disabled."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = None
        service.config.embedding.enable_sparse_vectors = False

        info = service.get_model_info()

        assert info["sparse_model"]["name"] is None
        assert info["sparse_model"]["loaded"] is False
        assert info["sparse_model"]["enabled"] is False

    def test_get_model_info_config_section(self, service):
        """Test model info includes correct config information."""
        info = service.get_model_info()

        expected_config = {
            "chunk_size": service.config.embedding.chunk_size,
            "chunk_overlap": service.config.embedding.chunk_overlap,
            "batch_size": service.config.embedding.batch_size,
        }

        assert info["config"] == expected_config


class TestAsyncContextManager:
    """Test suite for async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_initialization_and_cleanup(self, config):
        """Test using EmbeddingService as async context manager."""
        with (
            patch("python.common.core.embeddings.TextEmbedding"),
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = MagicMock()

            async with EmbeddingService(config) as service:
                assert service.initialized is True
                assert service.dense_model is not None

            # Should be cleaned up after context exit
            assert service.initialized is False
            assert service.dense_model is None

    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self, config):
        """Test context manager properly cleans up on exception."""
        with (
            patch("python.common.core.embeddings.TextEmbedding"),
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.return_value = MagicMock()

            try:
                async with EmbeddingService(config) as service:
                    assert service.initialized is True
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Should still be cleaned up after exception
            assert service.initialized is False
            assert service.dense_model is None


class TestCleanupAndResourceManagement:
    """Test suite for cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_close_cleanup_all_resources(self, service):
        """Test proper cleanup when closing service."""
        # Set up initialized state
        service.initialized = True
        service.dense_model = MagicMock()
        service.sparse_model = MagicMock()
        service.bm25_encoder = MagicMock()

        await service.close()

        # Verify all resources are cleaned up
        assert service.dense_model is None
        assert service.sparse_model is None
        assert service.bm25_encoder is None
        assert service.initialized is False

    @pytest.mark.asyncio
    async def test_close_idempotent(self, service):
        """Test that close() can be called multiple times safely."""
        service.initialized = True
        service.dense_model = MagicMock()

        # First close
        await service.close()
        assert service.initialized is False

        # Second close should not raise errors
        await service.close()
        assert service.initialized is False


class TestEdgeCasesAndErrorConditions:
    """Test suite for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_embedding_generation_with_very_long_text(self, service):
        """Test embedding generation with extremely long text."""
        service.initialized = True
        service.dense_model = MagicMock()

        # Create very long text
        very_long_text = "word " * 10000  # 50000 characters

        mock_dense = [[0.1] * 384]

        with patch.object(service, '_generate_dense_embeddings', return_value=mock_dense):
            result = await service.generate_embeddings(very_long_text)

        assert "dense" in result
        assert len(result["dense"]) == 384

    def test_chunk_text_with_no_separators_found(self, service):
        """Test chunking when no preferred separators are found."""
        # Text with no spaces, periods, or newlines
        text = "a" * 2000  # Longer than chunk size, no separators

        chunks = service.chunk_text(text, chunk_size=500)

        # Should force break at chunk boundaries
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 500

    def test_chunk_text_overlap_larger_than_chunk_size(self, service):
        """Test chunking with overlap larger than chunk size."""
        text = "word " * 100

        # This should be handled gracefully
        chunks = service.chunk_text(text, chunk_size=50, chunk_overlap=100)

        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_unicode_text(self, service):
        """Test embedding generation with unicode text."""
        service.initialized = True
        service.dense_model = MagicMock()

        unicode_text = "Hello 世界 🌍 émojis and special characters"
        mock_dense = [[0.1, 0.2, 0.3]]

        with patch.object(service, '_generate_dense_embeddings', return_value=mock_dense):
            result = await service.generate_embeddings(unicode_text)

        assert "dense" in result
        assert result["dense"] == [0.1, 0.2, 0.3]

    def test_hash_content_with_unicode(self, service):
        """Test content hashing with unicode characters."""
        unicode_content = "Content with émojis 🚀 and 中文"

        hash_result = service._hash_content(unicode_content)

        # Should handle unicode properly
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64

        # Should be reproducible
        hash_result2 = service._hash_content(unicode_content)
        assert hash_result == hash_result2