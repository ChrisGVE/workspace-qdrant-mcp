"""
Unit tests for sparse vector utilities.

Tests BM25 sparse encoding and Qdrant sparse vector creation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from qdrant_client.http import models

from workspace_qdrant_mcp.core.sparse_vectors import (
    BM25SparseEncoder,
    create_qdrant_sparse_vector,
    create_named_sparse_vector,
    normalize_sparse_vector
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
        mock_model.embed.return_value = [(np.array([1, 2, 3]), np.array([0.8, 0.6, 0.9]))]
        
        with patch('workspace_qdrant_mcp.core.sparse_vectors.SparseTextEmbedding') as mock_sparse_class:
            mock_sparse_class.return_value = mock_model
            
            await encoder_fastembed.initialize()
        
        assert encoder_fastembed.sparse_model == mock_model
        assert encoder_fastembed.initialized is True
        
        # Verify model was created with correct parameters
        mock_sparse_class.assert_called_once_with(
            model_name="Qdrant/bm25",
            max_length=512
        )
    
    @pytest.mark.asyncio
    async def test_initialize_fastembed_failure(self, encoder_fastembed):
        """Test FastEmbed initialization failure."""
        with patch('workspace_qdrant_mcp.core.sparse_vectors.SparseTextEmbedding') as mock_sparse_class:
            mock_sparse_class.side_effect = Exception("FastEmbed initialization failed")
            
            with pytest.raises(RuntimeError, match="Failed to initialize BM25 sparse encoder"):
                await encoder_fastembed.initialize()
        
        assert encoder_fastembed.initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_basic_with_corpus(self, encoder_basic):
        """Test basic initialization with training corpus."""
        corpus = [
            "This is the first document.",
            "This document is the second one.",
            "And this is the third document."
        ]
        
        with patch('sklearn.feature_extraction.text.TfidfVectorizer') as mock_vectorizer_class, \
             patch('rank_bm25.BM25Okapi') as mock_bm25_class:
            
            mock_vectorizer = MagicMock()
            mock_vectorizer.fit_transform.return_value = MagicMock()
            mock_vectorizer.get_feature_names_out.return_value = ["word1", "word2", "word3"]
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
        
        with patch('sklearn.feature_extraction.text.TfidfVectorizer') as mock_vectorizer_class:
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
        encoder_fastembed.sparse_model.embed.return_value = [(mock_indices, mock_values)]
        
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
        
        with patch('workspace_qdrant_mcp.core.sparse_vectors.word_tokenize') as mock_tokenize:
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
        
        with patch('workspace_qdrant_mcp.core.sparse_vectors.word_tokenize') as mock_tokenize:
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
        
        with pytest.raises(ValueError, match="Indices and values must have the same length"):
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
    
    def test_normalize_sparse_vector_l2(self):
        """Test L2 normalization of sparse vector."""
        indices = [0, 2, 4]
        values = [3.0, 4.0, 0.0]  # Will normalize to [0.6, 0.8, 0.0]
        
        norm_indices, norm_values = normalize_sparse_vector(indices, values, method="l2")
        
        assert norm_indices == [0, 2]  # Zero values removed
        assert len(norm_values) == 2
        assert abs(norm_values[0] - 0.6) < 1e-6
        assert abs(norm_values[1] - 0.8) < 1e-6
    
    def test_normalize_sparse_vector_l1(self):
        """Test L1 normalization of sparse vector."""
        indices = [0, 2, 4]
        values = [2.0, 6.0, 2.0]  # Sum = 10, will normalize to [0.2, 0.6, 0.2]
        
        norm_indices, norm_values = normalize_sparse_vector(indices, values, method="l1")
        
        assert norm_indices == indices  # All values non-zero
        assert len(norm_values) == 3
        assert abs(norm_values[0] - 0.2) < 1e-6
        assert abs(norm_values[1] - 0.6) < 1e-6
        assert abs(norm_values[2] - 0.2) < 1e-6
        
        # Verify L1 norm is 1
        assert abs(sum(norm_values) - 1.0) < 1e-6
    
    def test_normalize_sparse_vector_max(self):
        """Test max normalization of sparse vector."""
        indices = [1, 3, 5]
        values = [2.0, 8.0, 4.0]  # Max = 8, will normalize to [0.25, 1.0, 0.5]
        
        norm_indices, norm_values = normalize_sparse_vector(indices, values, method="max")
        
        assert norm_indices == indices
        assert len(norm_values) == 3
        assert abs(norm_values[0] - 0.25) < 1e-6
        assert abs(norm_values[1] - 1.0) < 1e-6
        assert abs(norm_values[2] - 0.5) < 1e-6
        
        # Verify max value is 1
        assert abs(max(norm_values) - 1.0) < 1e-6
    
    def test_normalize_sparse_vector_zero_values(self):
        """Test normalization with zero values."""
        indices = [0, 1, 2, 3]
        values = [0.0, 3.0, 0.0, 4.0]
        
        norm_indices, norm_values = normalize_sparse_vector(indices, values, method="l2")
        
        # Should remove zero values
        assert norm_indices == [1, 3]
        assert len(norm_values) == 2
        assert abs(norm_values[0] - 0.6) < 1e-6  # 3/5
        assert abs(norm_values[1] - 0.8) < 1e-6  # 4/5
    
    def test_normalize_sparse_vector_all_zeros(self):
        """Test normalization with all zero values."""
        indices = [0, 1, 2]
        values = [0.0, 0.0, 0.0]
        
        norm_indices, norm_values = normalize_sparse_vector(indices, values)
        
        # Should return empty arrays
        assert norm_indices == []
        assert norm_values == []
    
    def test_normalize_sparse_vector_single_value(self):
        """Test normalization with single non-zero value."""
        indices = [5]
        values = [7.0]
        
        norm_indices, norm_values = normalize_sparse_vector(indices, values, method="l2")
        
        assert norm_indices == [5]
        assert len(norm_values) == 1
        assert abs(norm_values[0] - 1.0) < 1e-6  # Single value normalized to 1
    
    def test_normalize_sparse_vector_invalid_method(self):
        """Test normalization with invalid method."""
        indices = [0, 1]
        values = [1.0, 2.0]
        
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_sparse_vector(indices, values, method="invalid")
    
    def test_normalize_sparse_vector_empty(self):
        """Test normalization with empty vectors."""
        norm_indices, norm_values = normalize_sparse_vector([], [])
        
        assert norm_indices == []
        assert norm_values == []
    
    def test_normalize_sparse_vector_mismatched_lengths(self):
        """Test normalization with mismatched indices and values."""
        indices = [0, 1, 2]
        values = [1.0, 2.0]  # One less value
        
        with pytest.raises(ValueError, match="Indices and values must have the same length"):
            normalize_sparse_vector(indices, values)
    
    def test_normalize_sparse_vector_preserve_order(self):
        """Test that normalization preserves index order."""
        indices = [5, 1, 10, 3]  # Not in sorted order
        values = [2.0, 1.0, 4.0, 3.0]
        
        norm_indices, norm_values = normalize_sparse_vector(indices, values, method="l1")
        
        # Should maintain the same order
        assert norm_indices == [5, 1, 10, 3]
        assert len(norm_values) == 4
        
        # Values should be normalized but maintain correspondence
        total = sum(values)
        expected_values = [v / total for v in values]
        for actual, expected in zip(norm_values, expected_values):
            assert abs(actual - expected) < 1e-6
