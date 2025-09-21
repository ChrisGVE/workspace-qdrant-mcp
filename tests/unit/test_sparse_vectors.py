"""
Unit tests for sparse vector utilities.

Tests BM25 sparse encoding and Qdrant sparse vector creation.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from qdrant_client.http import models

from common.core.sparse_vectors import (
    BM25SparseEncoder,
    create_named_sparse_vector,
    create_qdrant_sparse_vector,
)


class TestBM25SparseEncoder:
    """Test BM25SparseEncoder class."""

    @pytest.fixture
    def encoder_basic(self):
        """Create basic BM25SparseEncoder for testing."""
        return BM25SparseEncoder(use_fastembed=False)

    @pytest.fixture
    def encoder_fastembed(self):
        """Create BM25SparseEncoder with FastEmbed for testing."""
        return BM25SparseEncoder(use_fastembed=True)

    def test_init_basic(self, encoder_basic):
        """Test basic BM25SparseEncoder initialization."""
        assert encoder_basic.use_fastembed is False
        assert encoder_basic.sparse_model is None
        assert encoder_basic.vectorizer is None
        assert encoder_basic.bm25_model is None
        assert encoder_basic.vocabulary is None
        assert encoder_basic.initialized is False

    def test_init_fastembed(self, encoder_fastembed):
        """Test BM25SparseEncoder with FastEmbed initialization."""
        assert encoder_fastembed.use_fastembed is True
        assert encoder_fastembed.sparse_model is None
        assert encoder_fastembed.vectorizer is None
        assert encoder_fastembed.bm25_model is None
        assert encoder_fastembed.vocabulary is None
        assert encoder_fastembed.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_fastembed_success(self, encoder_fastembed):
        """Test successful FastEmbed initialization."""
        mock_model = MagicMock()
        mock_model.embed.return_value = [
            (np.array([1, 2, 3]), np.array([0.8, 0.6, 0.9]))
        ]

        with patch(
            "workspace_qdrant_mcp.core.sparse_vectors.SparseTextEmbedding"
        ) as mock_sparse_class:
            mock_sparse_class.return_value = mock_model

            await encoder_fastembed.initialize()

        assert encoder_fastembed.sparse_model == mock_model
        assert encoder_fastembed.initialized is True

        # Verify model was created with correct parameters
        mock_sparse_class.assert_called_once_with(
            model_name="Qdrant/bm25", max_length=512
        )

    @pytest.mark.asyncio
    async def test_initialize_fastembed_failure(self, encoder_fastembed):
        """Test FastEmbed initialization failure."""
        with patch(
            "workspace_qdrant_mcp.core.sparse_vectors.SparseTextEmbedding"
        ) as mock_sparse_class:
            mock_sparse_class.side_effect = Exception("FastEmbed initialization failed")

            with pytest.raises(
                RuntimeError, match="Failed to initialize BM25 sparse encoder"
            ):
                await encoder_fastembed.initialize()

        assert encoder_fastembed.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_basic_with_corpus(self, encoder_basic):
        """Test basic initialization with training corpus."""
        corpus = [
            "This is the first document.",
            "This document is the second one.",
            "And this is the third document.",
        ]

        with (
            patch(
                "sklearn.feature_extraction.text.TfidfVectorizer"
            ) as mock_vectorizer_class,
            patch("rank_bm25.BM25Okapi") as mock_bm25_class,
        ):
            mock_vectorizer = MagicMock()
            mock_vectorizer.fit_transform.return_value = MagicMock()
            mock_vectorizer.get_feature_names_out.return_value = [
                "word1",
                "word2",
                "word3",
            ]
            mock_vectorizer_class.return_value = mock_vectorizer

            mock_bm25 = MagicMock()
            mock_bm25_class.return_value = mock_bm25

            await encoder_basic.initialize(training_corpus=corpus)

        assert encoder_basic.vectorizer == mock_vectorizer
        assert encoder_basic.bm25_model == mock_bm25
        assert encoder_basic.vocabulary == ["word1", "word2", "word3"]
        assert encoder_basic.initialized is True

        # Verify training was called
        mock_vectorizer.fit_transform.assert_called_once_with(corpus)

    @pytest.mark.asyncio
    async def test_initialize_basic_no_corpus(self, encoder_basic):
        """Test basic initialization without training corpus."""
        await encoder_basic.initialize()

        # Should initialize with minimal setup
        assert encoder_basic.initialized is True
        assert encoder_basic.vectorizer is None
        assert encoder_basic.bm25_model is None

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, encoder_basic):
        """Test that initialize returns early if already initialized."""
        encoder_basic.initialized = True

        with patch(
            "sklearn.feature_extraction.text.TfidfVectorizer"
        ) as mock_vectorizer_class:
            await encoder_basic.initialize()

            # Should not attempt to initialize again
            mock_vectorizer_class.assert_not_called()

    def test_encode_not_initialized(self, encoder_basic):
        """Test encoding when not initialized."""
        encoder_basic.initialized = False

        with pytest.raises(RuntimeError, match="BM25SparseEncoder must be initialized"):
            encoder_basic.encode("test text")

    def test_encode_fastembed_success(self, encoder_fastembed):
        """Test successful encoding with FastEmbed."""
        encoder_fastembed.initialized = True
        encoder_fastembed.sparse_model = MagicMock()

        # Mock FastEmbed response
        mock_indices = np.array([1, 5, 10])
        mock_values = np.array([0.8, 0.6, 0.9])
        encoder_fastembed.sparse_model.embed.return_value = [
            (mock_indices, mock_values)
        ]

        result = encoder_fastembed.encode("test text")

        assert "indices" in result
        assert "values" in result
        assert result["indices"] == [1, 5, 10]
        assert result["values"] == [0.8, 0.6, 0.9]

        encoder_fastembed.sparse_model.embed.assert_called_once_with(["test text"])

    def test_encode_fastembed_empty_result(self, encoder_fastembed):
        """Test encoding with FastEmbed returning empty result."""
        encoder_fastembed.initialized = True
        encoder_fastembed.sparse_model = MagicMock()
        encoder_fastembed.sparse_model.embed.return_value = []

        result = encoder_fastembed.encode("test text")

        assert result["indices"] == []
        assert result["values"] == []

    def test_encode_basic_with_bm25(self, encoder_basic):
        """Test encoding with basic BM25 model."""
        encoder_basic.initialized = True
        encoder_basic.bm25_model = MagicMock()
        encoder_basic.vocabulary = ["word1", "word2", "word3"]

        # Mock BM25 scores
        mock_scores = [0.8, 0.0, 0.6]  # word2 has zero score
        encoder_basic.bm25_model.get_scores.return_value = mock_scores

        with patch(
            "workspace_qdrant_mcp.core.sparse_vectors.word_tokenize"
        ) as mock_tokenize:
            mock_tokenize.return_value = ["word1", "word3"]  # Only some words

            result = encoder_basic.encode("test text")

        assert "indices" in result
        assert "values" in result
        # Should only include non-zero scores
        assert 1 not in result["indices"]  # word2 index excluded (zero score)
        assert len(result["indices"]) == len(result["values"])

    def test_encode_basic_no_bm25(self, encoder_basic):
        """Test encoding with basic encoder but no BM25 model."""
        encoder_basic.initialized = True
        encoder_basic.bm25_model = None

        with patch(
            "workspace_qdrant_mcp.core.sparse_vectors.word_tokenize"
        ) as mock_tokenize:
            mock_tokenize.return_value = ["word1", "word2"]

            result = encoder_basic.encode("test text")

        # Should use simple term frequency
        assert "indices" in result
        assert "values" in result
        assert len(result["indices"]) > 0

    def test_encode_empty_text(self, encoder_fastembed):
        """Test encoding empty text."""
        encoder_fastembed.initialized = True
        encoder_fastembed.sparse_model = MagicMock()
        encoder_fastembed.sparse_model.embed.return_value = []

        result = encoder_fastembed.encode("")

        assert result["indices"] == []
        assert result["values"] == []

    def test_encode_exception_handling(self, encoder_fastembed):
        """Test encoding exception handling."""
        encoder_fastembed.initialized = True
        encoder_fastembed.sparse_model = MagicMock()
        encoder_fastembed.sparse_model.embed.side_effect = Exception("Encoding error")

        with pytest.raises(RuntimeError, match="Failed to encode text"):
            encoder_fastembed.encode("test text")

    def test_get_vocabulary_fastembed(self, encoder_fastembed):
        """Test vocabulary retrieval with FastEmbed."""
        encoder_fastembed.initialized = True
        encoder_fastembed.use_fastembed = True

        vocab = encoder_fastembed.get_vocabulary()

        # FastEmbed doesn't expose vocabulary
        assert vocab is None

    def test_get_vocabulary_basic(self, encoder_basic):
        """Test vocabulary retrieval with basic encoder."""
        encoder_basic.initialized = True
        encoder_basic.vocabulary = ["word1", "word2", "word3"]

        vocab = encoder_basic.get_vocabulary()

        assert vocab == ["word1", "word2", "word3"]

    def test_get_vocabulary_not_initialized(self, encoder_basic):
        """Test vocabulary retrieval when not initialized."""
        encoder_basic.initialized = False

        vocab = encoder_basic.get_vocabulary()

        assert vocab is None


class TestSparseVectorUtilities:
    """Test sparse vector utility functions."""

    def test_create_qdrant_sparse_vector(self):
        """Test creating Qdrant sparse vector."""
        indices = [1, 5, 10]
        values = [0.8, 0.6, 0.9]

        result = create_qdrant_sparse_vector(indices, values)

        assert isinstance(result, models.SparseVector)
        assert result.indices == indices
        assert result.values == values

    def test_create_qdrant_sparse_vector_empty(self):
        """Test creating Qdrant sparse vector with empty data."""
        result = create_qdrant_sparse_vector([], [])

        assert isinstance(result, models.SparseVector)
        assert result.indices == []
        assert result.values == []

    def test_create_qdrant_sparse_vector_mismatched_lengths(self):
        """Test creating sparse vector with mismatched indices and values."""
        indices = [1, 2, 3]
        values = [0.8, 0.6]  # One less value

        with pytest.raises(
            ValueError, match="Indices and values must have the same length"
        ):
            create_qdrant_sparse_vector(indices, values)

    def test_create_named_sparse_vector(self):
        """Test creating named sparse vector."""
        indices = [1, 5, 10]
        values = [0.8, 0.6, 0.9]
        name = "sparse_vector"

        result = create_named_sparse_vector(indices, values, name)

        assert isinstance(result, dict)
        assert name in result
        assert isinstance(result[name], models.SparseVector)
        assert result[name].indices == indices
        assert result[name].values == values

    def test_create_named_sparse_vector_default_name(self):
        """Test creating named sparse vector with default name."""
        indices = [1, 2, 3]
        values = [0.5, 0.7, 0.9]

        result = create_named_sparse_vector(indices, values)

        assert "sparse" in result
        assert isinstance(result["sparse"], models.SparseVector)
