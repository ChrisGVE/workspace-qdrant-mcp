"""
Unit tests for embedding service.

Tests FastEmbed integration, dense/sparse embeddings, and text processing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.core.config import Config, EmbeddingConfig
from common.core.embeddings import EmbeddingService


class TestEmbeddingService:
    """Test EmbeddingService class."""

    @pytest.fixture
    def embedding_config(self):
        """Create embedding configuration for testing."""
        return EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            enable_sparse_vectors=True,
            chunk_size=1000,
            chunk_overlap=200,
            batch_size=50,
        )

    @pytest.fixture
    def config(self, embedding_config):
        """Create full config with embedding settings."""
        config = Config()
        config.embedding = embedding_config
        return config

    @pytest.fixture
    def service(self, config):
        """Create EmbeddingService instance for testing."""
        return EmbeddingService(config)

    def test_init(self, service, config):
        """Test EmbeddingService initialization."""
        assert service.config == config
        assert service.dense_model is None
        assert service.sparse_model is None
        assert service.bm25_encoder is None
        assert service.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_dense_only(self, service):
        """Test initialization with dense embeddings only."""
        # Disable sparse vectors
        service.config.embedding.enable_sparse_vectors = False

        mock_dense_model = MagicMock()

        with patch(
            "workspace_qdrant_mcp.core.embeddings.TextEmbedding"
        ) as mock_text_embedding:
            mock_text_embedding.return_value = mock_dense_model

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor.return_value = mock_dense_model

                await service.initialize()

        assert service.dense_model == mock_dense_model
        assert service.bm25_encoder is None
        assert service.initialized is True

        # Verify TextEmbedding was called with correct parameters
        mock_loop.run_in_executor.assert_called_once()
        executor_args = mock_loop.run_in_executor.call_args[0]
        assert executor_args[0] is None  # No executor specified

    @pytest.mark.asyncio
    async def test_initialize_with_sparse_vectors(self, service):
        """Test initialization with both dense and sparse embeddings."""
        mock_dense_model = MagicMock()
        mock_bm25_encoder = MagicMock()
        mock_bm25_encoder.initialize = AsyncMock()

        with (
            patch(
                "workspace_qdrant_mcp.core.embeddings.TextEmbedding"
            ) as mock_text_embedding,
            patch(
                "workspace_qdrant_mcp.core.embeddings.BM25SparseEncoder"
            ) as mock_bm25_class,
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

        # Verify BM25 encoder initialization
        mock_bm25_class.assert_called_once_with(use_fastembed=True)
        mock_bm25_encoder.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, service):
        """Test that initialize returns early if already initialized."""
        service.initialized = True

        with patch(
            "workspace_qdrant_mcp.core.embeddings.TextEmbedding"
        ) as mock_text_embedding:
            await service.initialize()

            # Should not attempt to initialize models
            mock_text_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_dense_model_failure(self, service):
        """Test initialization failure when dense model fails to load."""
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            mock_loop.run_in_executor.side_effect = Exception("Model loading failed")

            with pytest.raises(
                RuntimeError, match="Failed to initialize embedding models"
            ):
                await service.initialize()

            assert service.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_sparse_model_failure(self, service):
        """Test initialization failure when sparse model fails."""
        mock_dense_model = MagicMock()
        mock_bm25_encoder = MagicMock()
        mock_bm25_encoder.initialize.side_effect = Exception(
            "BM25 initialization failed"
        )

        with (
            patch(
                "workspace_qdrant_mcp.core.embeddings.TextEmbedding"
            ) as mock_text_embedding,
            patch(
                "workspace_qdrant_mcp.core.embeddings.BM25SparseEncoder"
            ) as mock_bm25_class,
        ):
            mock_text_embedding.return_value = mock_dense_model
            mock_bm25_class.return_value = mock_bm25_encoder

            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor.return_value = mock_dense_model

                with pytest.raises(
                    RuntimeError, match="Failed to initialize embedding models"
                ):
                    await service.initialize()

            assert service.initialized is False

    @pytest.mark.asyncio
    async def test_generate_embeddings_not_initialized(self, service):
        """Test embedding generation when service not initialized."""
        service.initialized = False

        with pytest.raises(RuntimeError, match="EmbeddingService must be initialized"):
            await service.generate_embeddings("test text")

    @pytest.mark.asyncio
    async def test_generate_embeddings_dense_only(self, service):
        """Test generating dense embeddings only."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = None  # No sparse vectors

        # Mock dense embeddings
        mock_embeddings = [[0.1, 0.2, 0.3, 0.4]]
        service.dense_model.embed.return_value = mock_embeddings

        result = await service.generate_embeddings("test text")

        assert "dense" in result
        assert result["dense"] == [0.1, 0.2, 0.3, 0.4]
        assert "sparse" not in result

        service.dense_model.embed.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_with_sparse(self, service):
        """Test generating both dense and sparse embeddings."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()

        # Mock dense embeddings
        mock_dense = [[0.1, 0.2, 0.3, 0.4]]
        service.dense_model.embed.return_value = mock_dense

        # Mock sparse embeddings
        mock_sparse = {"indices": [1, 3, 5], "values": [0.8, 0.6, 0.9]}
        service.bm25_encoder.encode.return_value = mock_sparse

        result = await service.generate_embeddings("test text", include_sparse=True)

        assert "dense" in result
        assert "sparse" in result
        assert result["dense"] == [0.1, 0.2, 0.3, 0.4]
        assert result["sparse"] == mock_sparse

        service.dense_model.embed.assert_called_once_with(["test text"])
        service.bm25_encoder.encode.assert_called_once_with("test text")

    @pytest.mark.asyncio
    async def test_generate_embeddings_sparse_requested_but_not_available(
        self, service
    ):
        """Test requesting sparse embeddings when not available."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = None  # No sparse encoder

        mock_dense = [[0.1, 0.2, 0.3, 0.4]]
        service.dense_model.embed.return_value = mock_dense

        result = await service.generate_embeddings("test text", include_sparse=True)

        # Should still return dense embeddings, just no sparse
        assert "dense" in result
        assert "sparse" not in result
        assert result["dense"] == [0.1, 0.2, 0.3, 0.4]

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_processing(self, service):
        """Test batch processing of multiple texts."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()

        texts = ["text 1", "text 2", "text 3"]

        # Mock batch embeddings
        mock_dense = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        service.dense_model.embed.return_value = mock_dense

        # Mock sparse embeddings for each text
        sparse_results = [
            {"indices": [1, 2], "values": [0.8, 0.6]},
            {"indices": [2, 3], "values": [0.7, 0.9]},
            {"indices": [1, 3], "values": [0.5, 0.8]},
        ]
        service.bm25_encoder.encode.side_effect = sparse_results

        results = await service.generate_embeddings_batch(texts, include_sparse=True)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert "dense" in result
            assert "sparse" in result
            assert result["dense"] == mock_dense[i]
            assert result["sparse"] == sparse_results[i]

        service.dense_model.embed.assert_called_once_with(texts)
        assert service.bm25_encoder.encode.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_embeddings_empty_text(self, service):
        """Test embedding generation with empty text."""
        service.initialized = True
        service.dense_model = MagicMock()

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embeddings("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await service.generate_embeddings("   ")  # Only whitespace

    @pytest.mark.asyncio
    async def test_generate_embeddings_exception_handling(self, service):
        """Test exception handling during embedding generation."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.dense_model.embed.side_effect = Exception("Embedding error")

        with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
            await service.generate_embeddings("test text")

    def test_chunk_text_short_text(self, service):
        """Test chunking of short text (no chunking needed)."""
        short_text = "This is a short text that doesn't need chunking."
        chunks = service.chunk_text(short_text)

        assert chunks == [short_text]

    def test_chunk_text_long_text(self, service):
        """Test chunking of long text."""
        # Create text longer than chunk_size (1000 chars)
        long_text = "This is a sentence. " * 60  # ~1200 characters

        chunks = service.chunk_text(long_text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= service.config.embedding.chunk_size

        # Verify overlap between chunks
        if len(chunks) > 1:
            overlap_size = service.config.embedding.chunk_overlap
            chunk1_end = chunks[0][-overlap_size:]
            chunk2_start = chunks[1][:overlap_size]
            # Some overlap should exist (may not be exact due to sentence boundaries)
            assert len(chunk1_end.strip()) > 0
            assert len(chunk2_start.strip()) > 0

    def test_chunk_text_custom_separators(self, service):
        """Test chunking with custom separators."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3." * 100

        chunks = service.chunk_text(text, separators=["\n\n", ".", " "])

        assert len(chunks) > 1
        # Should prefer paragraph breaks when possible
        for chunk in chunks:
            assert len(chunk) <= service.config.embedding.chunk_size

    def test_chunk_text_preserve_context(self, service):
        """Test that chunking preserves important context."""
        # Text with clear sentence boundaries
        sentences = [f"This is sentence number {i}. " for i in range(100)]
        text = "".join(sentences)

        chunks = service.chunk_text(text)

        # Verify no sentences are cut in the middle
        for chunk in chunks:
            # Should end with complete sentences (ending with period and space, or end of text)
            assert chunk.rstrip().endswith(".") or chunk == chunks[-1]

    def test_get_model_info_not_initialized(self, service):
        """Test getting model info when not initialized."""
        service.initialized = False

        info = service.get_model_info()

        assert info["model_name"] == service.config.embedding.model
        assert info["vector_size"] is None
        assert info["sparse_enabled"] == service.config.embedding.enable_sparse_vectors
        assert info["initialized"] is False

    def test_get_model_info_initialized(self, service):
        """Test getting model info when initialized."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()

        # Mock model dimensions
        with patch.object(service, "_get_vector_size", return_value=384):
            info = service.get_model_info()

        assert info["model_name"] == service.config.embedding.model
        assert info["vector_size"] == 384
        assert info["sparse_enabled"] is True
        assert info["initialized"] is True

    def test_get_vector_size_with_model(self, service):
        """Test getting vector size from model."""
        service.dense_model = MagicMock()

        # Mock embedding to get vector size
        mock_embedding = [[0.1] * 384]
        service.dense_model.embed.return_value = mock_embedding

        size = service._get_vector_size()

        assert size == 384
        service.dense_model.embed.assert_called_once_with(["test"])

    def test_get_vector_size_no_model(self, service):
        """Test getting vector size when no model available."""
        service.dense_model = None

        size = service._get_vector_size()

        assert size is None

    def test_get_vector_size_exception(self, service):
        """Test getting vector size when model fails."""
        service.dense_model = MagicMock()
        service.dense_model.embed.side_effect = Exception("Model error")

        size = service._get_vector_size()

        assert size is None

    @pytest.mark.asyncio
    async def test_generate_cache_key(self, service):
        """Test cache key generation."""
        text = "test text for caching"
        include_sparse = True

        key = service._generate_cache_key(text, include_sparse)

        # Should be a consistent hash
        assert isinstance(key, str)
        assert len(key) == 64  # SHA256 hex digest length

        # Same input should produce same key
        key2 = service._generate_cache_key(text, include_sparse)
        assert key == key2

        # Different input should produce different key
        key3 = service._generate_cache_key(text, False)
        assert key != key3

    def test_text_preprocessing(self, service):
        """Test text preprocessing functionality."""
        # Test with various text formats
        texts = [
            "  Normal text with spaces  ",
            "Text\twith\ttabs",
            "Text\nwith\nnewlines\n",
            "Text with    multiple    spaces",
            "Text with \u00a0 non-breaking spaces",
        ]

        for text in texts:
            processed = service._preprocess_text(text)

            # Should normalize whitespace
            assert "\t" not in processed
            assert "\n" not in processed
            assert not processed.startswith(" ")
            assert not processed.endswith(" ")
            assert "  " not in processed  # No double spaces

    def test_text_preprocessing_empty(self, service):
        """Test preprocessing of empty or whitespace text."""
        empty_texts = ["", "   ", "\n\n", "\t\t"]

        for text in empty_texts:
            processed = service._preprocess_text(text)
            assert processed == ""

    @pytest.mark.asyncio
    async def test_close_cleanup(self, service):
        """Test proper cleanup when closing service."""
        service.initialized = True
        service.dense_model = MagicMock()
        service.bm25_encoder = MagicMock()

        await service.close()

        assert service.dense_model is None
        assert service.bm25_encoder is None
        assert service.initialized is False

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, config):
        """Test using EmbeddingService as async context manager."""
        with patch("workspace_qdrant_mcp.core.embeddings.TextEmbedding"):
            with patch("asyncio.get_event_loop") as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                mock_loop.run_in_executor.return_value = MagicMock()

                async with EmbeddingService(config) as service:
                    assert service.initialized is True

                # Should be cleaned up after context exit
                assert service.initialized is False
