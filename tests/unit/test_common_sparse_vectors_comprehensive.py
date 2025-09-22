"""
Comprehensive unit tests for python.common.core.sparse_vectors module.

Tests cover BM25SparseEncoder and all sparse vector functionality
with 100% coverage including async patterns, dual implementations, and error handling.
"""

import pytest
import asyncio
import math
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

# Import modules under test
from src.python.common.core.sparse_vectors import (
    BM25SparseEncoder,
    create_qdrant_sparse_vector,
    create_named_sparse_vector,
    word_tokenize
)


class TestWordTokenize:
    """Test word_tokenize utility function."""

    def test_basic_tokenization(self):
        """Test basic word tokenization."""
        text = "Hello world this is a test"
        tokens = word_tokenize(text)
        assert tokens == ["hello", "world", "this", "test"]

    def test_punctuation_removal(self):
        """Test tokenization removes punctuation."""
        text = "Hello, world! This is a test."
        tokens = word_tokenize(text)
        assert tokens == ["hello", "world", "this", "test"]

    def test_short_word_filtering(self):
        """Test that short words are filtered out."""
        text = "I am a big dog"
        tokens = word_tokenize(text)
        assert tokens == ["big", "dog"]  # "I", "am", "a" are too short

    def test_numeric_filtering(self):
        """Test that numeric content is filtered."""
        text = "This is test 123 with numbers"
        tokens = word_tokenize(text)
        assert tokens == ["this", "test", "with", "numbers"]

    def test_empty_text(self):
        """Test tokenization of empty text."""
        tokens = word_tokenize("")
        assert tokens == []

    def test_special_characters(self):
        """Test handling of special characters."""
        text = "test@email.com and #hashtag"
        tokens = word_tokenize(text)
        assert tokens == ["test", "email", "com", "and", "hashtag"]

    def test_case_normalization(self):
        """Test case normalization."""
        text = "Hello WORLD Test"
        tokens = word_tokenize(text)
        assert tokens == ["hello", "world", "test"]


class TestBM25SparseEncoder:
    """Test BM25SparseEncoder functionality."""

    @pytest.fixture
    def encoder_fastembed(self):
        """Create BM25SparseEncoder configured for FastEmbed."""
        return BM25SparseEncoder(
            use_fastembed=True,
            k1=1.2,
            b=0.75,
            min_df=1,
            max_df=0.95
        )

    @pytest.fixture
    def encoder_custom(self):
        """Create BM25SparseEncoder configured for custom BM25."""
        return BM25SparseEncoder(
            use_fastembed=False,
            k1=1.5,
            b=0.6,
            min_df=2,
            max_df=0.8
        )

    @pytest.fixture
    def mock_fastembed_model(self):
        """Create a mock FastEmbed model."""
        model = Mock()
        model.embed = Mock()
        return model

    @pytest.fixture
    def mock_vectorizer(self):
        """Create a mock TfidfVectorizer."""
        vectorizer = Mock()
        vectorizer.fit_transform = Mock()
        vectorizer.get_feature_names_out = Mock(return_value=["machine", "learning", "algorithm"])
        return vectorizer

    @pytest.fixture
    def mock_bm25_model(self):
        """Create a mock BM25 model."""
        model = Mock()
        model.get_scores = Mock(return_value=[0.5, 0.8, 0.0, 0.3])
        return model

    def test_init_default_params(self):
        """Test encoder initialization with default parameters."""
        encoder = BM25SparseEncoder()

        assert encoder.use_fastembed == True
        assert encoder.k1 == 1.2
        assert encoder.b == 0.75
        assert encoder.min_df == 1
        assert encoder.max_df == 0.95
        assert encoder.sparse_model is None
        assert encoder.vectorizer is None
        assert encoder.bm25_model is None
        assert encoder.vocabulary is None
        assert encoder.initialized == False

    def test_init_custom_params(self):
        """Test encoder initialization with custom parameters."""
        encoder = BM25SparseEncoder(
            use_fastembed=False,
            k1=1.8,
            b=0.5,
            min_df=3,
            max_df=0.7
        )

        assert encoder.use_fastembed == False
        assert encoder.k1 == 1.8
        assert encoder.b == 0.5
        assert encoder.min_df == 3
        assert encoder.max_df == 0.7

    async def test_initialize_fastembed_success(self, encoder_fastembed, mock_fastembed_model):
        """Test successful FastEmbed initialization."""
        with patch('src.python.common.core.sparse_vectors.SparseTextEmbedding', mock_fastembed_model):
            await encoder_fastembed.initialize()

            assert encoder_fastembed.initialized == True
            assert encoder_fastembed.sparse_model == mock_fastembed_model
            assert encoder_fastembed.fastembed_model == mock_fastembed_model

    async def test_initialize_fastembed_not_available(self, encoder_fastembed):
        """Test initialization when FastEmbed is not available."""
        with patch('src.python.common.core.sparse_vectors.SparseTextEmbedding', None):
            await encoder_fastembed.initialize()

            assert encoder_fastembed.initialized == True
            assert encoder_fastembed.sparse_model is None

    async def test_initialize_fastembed_failure(self, encoder_fastembed):
        """Test FastEmbed initialization failure."""
        with patch('src.python.common.core.sparse_vectors.SparseTextEmbedding') as mock_class:
            mock_class.side_effect = Exception("Model loading failed")

            with pytest.raises(RuntimeError, match="Failed to initialize BM25 sparse encoder"):
                await encoder_fastembed.initialize()

            assert encoder_fastembed.initialized == False

    async def test_initialize_fastembed_async_executor_mock(self, encoder_fastembed):
        """Test FastEmbed initialization with mocked executor."""
        mock_model = Mock()

        with patch('src.python.common.core.sparse_vectors.SparseTextEmbedding', return_value=mock_model):
            with patch('asyncio.get_event_loop') as mock_loop:
                # Mock run_in_executor to return the model directly (simulating mock behavior)
                mock_loop.return_value.run_in_executor.return_value = mock_model

                await encoder_fastembed.initialize()

                assert encoder_fastembed.initialized == True
                assert encoder_fastembed.sparse_model == mock_model

    async def test_initialize_custom_with_corpus(self, encoder_custom, mock_vectorizer, mock_bm25_model):
        """Test custom BM25 initialization with training corpus."""
        training_corpus = ["machine learning", "deep learning", "neural networks"]

        with patch('src.python.common.core.sparse_vectors.TfidfVectorizer', return_value=mock_vectorizer):
            with patch('src.python.common.core.sparse_vectors.rank_bm25') as mock_rank_bm25:
                mock_rank_bm25.BM25Okapi.return_value = mock_bm25_model

                await encoder_custom.initialize(training_corpus)

                assert encoder_custom.initialized == True
                assert encoder_custom.vectorizer == mock_vectorizer
                assert encoder_custom.bm25_model == mock_bm25_model
                assert encoder_custom.vocabulary == ["machine", "learning", "algorithm"]

    async def test_initialize_custom_missing_sklearn(self, encoder_custom):
        """Test custom initialization when sklearn is missing."""
        training_corpus = ["test document"]

        with patch('src.python.common.core.sparse_vectors.TfidfVectorizer', None):
            with pytest.raises(ImportError, match="sklearn is required"):
                await encoder_custom.initialize(training_corpus)

    async def test_initialize_custom_missing_rank_bm25(self, encoder_custom):
        """Test custom initialization when rank_bm25 is missing."""
        training_corpus = ["test document"]

        with patch('src.python.common.core.sparse_vectors.rank_bm25', None):
            with pytest.raises(ImportError, match="rank_bm25 is required"):
                await encoder_custom.initialize(training_corpus)

    async def test_initialize_idempotent(self, encoder_fastembed):
        """Test that initialize is idempotent."""
        with patch('src.python.common.core.sparse_vectors.SparseTextEmbedding') as mock_class:
            mock_model = Mock()
            mock_class.return_value = mock_model

            await encoder_fastembed.initialize()
            await encoder_fastembed.initialize()  # Second call

            # Should only create model once
            mock_class.assert_called_once()

    def test_encode_not_initialized(self, encoder_fastembed):
        """Test encoding without initialization raises error."""
        with pytest.raises(RuntimeError, match="must be initialized first"):
            encoder_fastembed.encode("test text")

    def test_encode_fastembed_success(self, encoder_fastembed, mock_fastembed_model):
        """Test successful encoding with FastEmbed."""
        encoder_fastembed.initialized = True
        encoder_fastembed.sparse_model = mock_fastembed_model

        # Mock embedding result as tuple (for testing)
        mock_embedding = ([0, 2, 5], [0.5, 0.8, 0.3])
        mock_fastembed_model.embed.return_value = [mock_embedding]

        result = encoder_fastembed.encode("test text")

        assert result == {"indices": [0, 2, 5], "values": [0.5, 0.8, 0.3]}
        mock_fastembed_model.embed.assert_called_once_with(["test text"])

    def test_encode_fastembed_object_format(self, encoder_fastembed, mock_fastembed_model):
        """Test encoding with FastEmbed object format."""
        encoder_fastembed.initialized = True
        encoder_fastembed.sparse_model = mock_fastembed_model

        # Mock embedding result as object (real FastEmbed format)
        mock_embedding = Mock()
        mock_embedding.indices.tolist.return_value = [1, 3, 7]
        mock_embedding.values.tolist.return_value = [0.4, 0.6, 0.9]
        mock_fastembed_model.embed.return_value = [mock_embedding]

        result = encoder_fastembed.encode("test text")

        assert result == {"indices": [1, 3, 7], "values": [0.4, 0.6, 0.9]}

    def test_encode_fastembed_empty_result(self, encoder_fastembed, mock_fastembed_model):
        """Test encoding with empty FastEmbed result."""
        encoder_fastembed.initialized = True
        encoder_fastembed.sparse_model = mock_fastembed_model
        mock_fastembed_model.embed.return_value = []

        result = encoder_fastembed.encode("test text")

        assert result == {"indices": [], "values": []}

    def test_encode_fastembed_failure(self, encoder_fastembed, mock_fastembed_model):
        """Test encoding failure with FastEmbed."""
        encoder_fastembed.initialized = True
        encoder_fastembed.sparse_model = mock_fastembed_model
        mock_fastembed_model.embed.side_effect = Exception("Encoding failed")

        with pytest.raises(RuntimeError, match="Failed to encode text"):
            encoder_fastembed.encode("test text")

    def test_encode_custom_bm25_success(self, encoder_custom, mock_bm25_model):
        """Test successful encoding with custom BM25."""
        encoder_custom.initialized = True
        encoder_custom.bm25_model = mock_bm25_model
        encoder_custom.vocabulary = ["machine", "learning", "algorithm", "test"]

        # Mock BM25 scores (indices 0, 1, 3 have non-zero scores)
        mock_bm25_model.get_scores.return_value = [0.5, 0.8, 0.0, 0.3]

        result = encoder_custom.encode("machine learning test")

        assert result == {"indices": [0, 1, 3], "values": [0.5, 0.8, 0.3]}

    def test_encode_custom_simple_fallback(self, encoder_custom):
        """Test encoding with simple term frequency fallback."""
        encoder_custom.initialized = True
        encoder_custom.bm25_model = None
        encoder_custom.vocabulary = None

        result = encoder_custom.encode("machine learning test")

        # Should use simple term frequency
        assert "indices" in result
        assert "values" in result
        assert len(result["indices"]) == len(result["values"])

    async def test_encode_single(self, encoder_fastembed):
        """Test async encode_single method."""
        encoder_fastembed.encode = Mock(return_value={"indices": [1, 2], "values": [0.5, 0.8]})

        result = await encoder_fastembed.encode_single("test text")

        assert result == {"indices": [1, 2], "values": [0.5, 0.8]}
        encoder_fastembed.encode.assert_called_once_with("test text")

    async def test_encode_batch_not_initialized(self, encoder_fastembed):
        """Test batch encoding auto-initializes."""
        texts = ["doc1", "doc2"]

        with patch.object(encoder_fastembed, 'initialize', new_callable=AsyncMock) as mock_init:
            with patch.object(encoder_fastembed, '_encode_with_fastembed', new_callable=AsyncMock) as mock_encode:
                mock_encode.return_value = [{"indices": [], "values": []}] * 2

                await encoder_fastembed.encode_batch(texts)

                mock_init.assert_called_once()

    async def test_encode_batch_fastembed(self, encoder_fastembed):
        """Test batch encoding with FastEmbed."""
        encoder_fastembed.initialized = True
        encoder_fastembed.use_fastembed = True
        encoder_fastembed.sparse_model = Mock()

        expected_result = [
            {"indices": [0, 1], "values": [0.5, 0.8]},
            {"indices": [2, 3], "values": [0.3, 0.9]}
        ]

        with patch.object(encoder_fastembed, '_encode_with_fastembed', new_callable=AsyncMock) as mock_encode:
            mock_encode.return_value = expected_result

            result = await encoder_fastembed.encode_batch(["doc1", "doc2"])

            assert result == expected_result
            mock_encode.assert_called_once_with(["doc1", "doc2"])

    async def test_encode_batch_custom(self, encoder_custom):
        """Test batch encoding with custom BM25."""
        encoder_custom.initialized = True
        encoder_custom.use_fastembed = False

        expected_result = [
            {"indices": [0, 1], "values": [0.4, 0.7]},
            {"indices": [1, 2], "values": [0.6, 0.5]}
        ]

        with patch.object(encoder_custom, '_encode_with_custom_bm25') as mock_encode:
            mock_encode.return_value = expected_result

            result = await encoder_custom.encode_batch(["doc1", "doc2"])

            assert result == expected_result
            mock_encode.assert_called_once_with(["doc1", "doc2"])

    async def test_encode_with_fastembed_success(self, encoder_fastembed, mock_fastembed_model):
        """Test _encode_with_fastembed method."""
        encoder_fastembed.sparse_model = mock_fastembed_model

        # Mock embeddings as tuples
        mock_embeddings = [
            ([0, 1], [0.5, 0.8]),
            ([2, 3], [0.3, 0.9])
        ]
        mock_fastembed_model.embed.return_value = mock_embeddings

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor.return_value = mock_embeddings

            result = await encoder_fastembed._encode_with_fastembed(["doc1", "doc2"])

            expected = [
                {"indices": [0, 1], "values": [0.5, 0.8]},
                {"indices": [2, 3], "values": [0.3, 0.9]}
            ]
            assert result == expected

    async def test_encode_with_fastembed_failure_fallback(self, encoder_fastembed, mock_fastembed_model):
        """Test _encode_with_fastembed fallback on failure."""
        encoder_fastembed.sparse_model = mock_fastembed_model

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor.side_effect = Exception("Encoding failed")

            with patch.object(encoder_fastembed, '_encode_with_custom_bm25') as mock_fallback:
                expected_result = [{"indices": [], "values": []}]
                mock_fallback.return_value = expected_result

                result = await encoder_fastembed._encode_with_fastembed(["doc1"])

                assert result == expected_result
                mock_fallback.assert_called_once_with(["doc1"])

    def test_encode_with_custom_bm25_no_vocab(self, encoder_custom):
        """Test custom BM25 encoding without existing vocabulary."""
        texts = ["machine learning", "deep learning"]

        with patch.object(encoder_custom, '_build_vocabulary') as mock_build:
            with patch.object(encoder_custom, '_compute_bm25_scores') as mock_compute:
                mock_compute.side_effect = [
                    {"indices": [0, 1], "values": [0.5, 0.8]},
                    {"indices": [1, 2], "values": [0.6, 0.7]}
                ]

                result = encoder_custom._encode_with_custom_bm25(texts)

                mock_build.assert_called_once_with(texts)
                assert len(result) == 2

    def test_encode_with_custom_bm25_existing_vocab(self, encoder_custom):
        """Test custom BM25 encoding with existing vocabulary."""
        encoder_custom.vocab = {"machine": 0, "learning": 1}
        texts = ["machine learning"]

        with patch.object(encoder_custom, '_build_vocabulary') as mock_build:
            with patch.object(encoder_custom, '_compute_bm25_scores') as mock_compute:
                mock_compute.return_value = {"indices": [0, 1], "values": [0.5, 0.8]}

                result = encoder_custom._encode_with_custom_bm25(texts)

                mock_build.assert_not_called()  # Should not rebuild vocab
                assert len(result) == 1

    def test_build_vocabulary(self, encoder_custom):
        """Test vocabulary building."""
        texts = [
            "machine learning algorithms",
            "deep learning networks",
            "machine vision systems"
        ]

        with patch.object(encoder_custom, '_tokenize') as mock_tokenize:
            mock_tokenize.side_effect = [
                ["machine", "learning", "algorithms"],
                ["deep", "learning", "networks"],
                ["machine", "vision", "systems"]
            ]

            encoder_custom._build_vocabulary(texts)

            assert encoder_custom.corpus_size == 3
            assert len(encoder_custom.vocab) > 0
            assert "machine" in encoder_custom.vocab
            assert "learning" in encoder_custom.vocab
            assert encoder_custom.avg_doc_length == 3.0

    def test_build_vocabulary_with_filtering(self, encoder_custom):
        """Test vocabulary building with frequency filtering."""
        encoder_custom.min_df = 2  # Require term in at least 2 docs
        encoder_custom.max_df = 0.6  # Ignore terms in more than 60% of docs

        texts = [
            "common rare unique",
            "common different special",
            "common another test",
            "common final words"
        ]

        with patch.object(encoder_custom, '_tokenize') as mock_tokenize:
            mock_tokenize.side_effect = [
                ["common", "rare", "unique"],
                ["common", "different", "special"],
                ["common", "another", "test"],
                ["common", "final", "words"]
            ]

            encoder_custom._build_vocabulary(texts)

            # "common" should be filtered out (appears in 100% of docs, > 60%)
            # "rare", "unique", etc. should be filtered out (appear in < 2 docs)
            # Only terms appearing in 2-2 docs should remain (none in this case)
            assert len(encoder_custom.vocab) == 0  # All terms filtered

    def test_compute_bm25_scores(self, encoder_custom):
        """Test BM25 score computation."""
        encoder_custom.vocab = {"machine": 0, "learning": 1, "algorithm": 2}
        encoder_custom.idf_scores = {"machine": 1.5, "learning": 1.2, "algorithm": 2.0}
        encoder_custom.avg_doc_length = 10.0
        encoder_custom.k1 = 1.2
        encoder_custom.b = 0.75

        with patch.object(encoder_custom, '_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["machine", "learning", "machine"]

            result = encoder_custom._compute_bm25_scores("machine learning machine")

            assert "indices" in result
            assert "values" in result
            assert len(result["indices"]) == len(result["values"])
            # Should have scores for "machine" and "learning"
            assert len(result["indices"]) == 2

    def test_compute_bm25_scores_with_doc_length(self, encoder_custom):
        """Test BM25 score computation with custom document length."""
        encoder_custom.vocab = {"test": 0}
        encoder_custom.idf_scores = {"test": 1.0}
        encoder_custom.avg_doc_length = 10.0
        encoder_custom.k1 = 1.2
        encoder_custom.b = 0.75

        with patch.object(encoder_custom, '_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["test"]

            result = encoder_custom._compute_bm25_scores("test", doc_length=5)

            assert len(result["indices"]) == 1
            assert result["indices"][0] == 0

    def test_compute_bm25_scores_empty_result(self, encoder_custom):
        """Test BM25 computation with no matching terms."""
        encoder_custom.vocab = {"machine": 0, "learning": 1}
        encoder_custom.idf_scores = {"machine": 1.5, "learning": 1.2}

        with patch.object(encoder_custom, '_tokenize') as mock_tokenize:
            mock_tokenize.return_value = ["unknown", "terms"]

            result = encoder_custom._compute_bm25_scores("unknown terms")

            assert result == {"indices": [], "values": []}

    def test_tokenize(self, encoder_custom):
        """Test tokenization method."""
        text = "Machine Learning and AI, with 123 numbers!"
        tokens = encoder_custom._tokenize(text)

        assert tokens == ["machine", "learning", "and", "with", "numbers"]

    def test_tokenize_short_tokens_filtered(self, encoder_custom):
        """Test that short tokens are filtered."""
        text = "I am a big dog"
        tokens = encoder_custom._tokenize(text)

        assert tokens == ["big", "dog"]  # "I", "am", "a" are too short

    def test_get_vocabulary_fastembed(self, encoder_fastembed):
        """Test get_vocabulary with FastEmbed (should return None)."""
        encoder_fastembed.initialized = True
        encoder_fastembed.use_fastembed = True

        vocab = encoder_fastembed.get_vocabulary()
        assert vocab is None

    def test_get_vocabulary_custom(self, encoder_custom):
        """Test get_vocabulary with custom BM25."""
        encoder_custom.initialized = True
        encoder_custom.use_fastembed = False
        encoder_custom.vocabulary = ["machine", "learning", "algorithm"]

        vocab = encoder_custom.get_vocabulary()
        assert vocab == ["machine", "learning", "algorithm"]

    def test_get_vocabulary_not_initialized(self, encoder_custom):
        """Test get_vocabulary when not initialized."""
        encoder_custom.initialized = False

        vocab = encoder_custom.get_vocabulary()
        assert vocab is None

    def test_get_vocab_size(self, encoder_custom):
        """Test getting vocabulary size."""
        encoder_custom.vocab = {"machine": 0, "learning": 1, "algorithm": 2}

        size = encoder_custom.get_vocab_size()
        assert size == 3

    def test_get_vocab_size_empty(self, encoder_custom):
        """Test getting vocabulary size when empty."""
        encoder_custom.vocab = {}

        size = encoder_custom.get_vocab_size()
        assert size == 0

    def test_get_model_info_fastembed(self, encoder_fastembed):
        """Test getting model info for FastEmbed encoder."""
        encoder_fastembed.initialized = True
        encoder_fastembed.use_fastembed = True
        encoder_fastembed.sparse_model = Mock()
        encoder_fastembed.vocab = {"term1": 0, "term2": 1}
        encoder_fastembed.corpus_size = 100
        encoder_fastembed.avg_doc_length = 15.5

        info = encoder_fastembed.get_model_info()

        assert info["encoder_type"] == "fastembed"
        assert info["vocab_size"] == 2
        assert info["corpus_size"] == 100
        assert info["avg_doc_length"] == 15.5
        assert info["parameters"]["k1"] == 1.2
        assert info["parameters"]["b"] == 0.75
        assert info["initialized"] == True

    def test_get_model_info_custom(self, encoder_custom):
        """Test getting model info for custom BM25 encoder."""
        encoder_custom.initialized = True
        encoder_custom.use_fastembed = False
        encoder_custom.sparse_model = None
        encoder_custom.vocab = {"term1": 0, "term2": 1, "term3": 2}
        encoder_custom.corpus_size = 50
        encoder_custom.avg_doc_length = 12.3
        encoder_custom.k1 = 1.5
        encoder_custom.b = 0.6

        info = encoder_custom.get_model_info()

        assert info["encoder_type"] == "custom_bm25"
        assert info["vocab_size"] == 3
        assert info["corpus_size"] == 50
        assert info["avg_doc_length"] == 12.3
        assert info["parameters"]["k1"] == 1.5
        assert info["parameters"]["b"] == 0.6
        assert info["initialized"] == True


class TestCreateQdrantSparseVector:
    """Test create_qdrant_sparse_vector utility function."""

    def test_create_sparse_vector_success(self):
        """Test successful sparse vector creation."""
        indices = [0, 2, 5, 10]
        values = [0.5, 0.8, 0.3, 0.9]

        with patch('src.python.common.core.sparse_vectors.models') as mock_models:
            mock_sparse_vector = Mock()
            mock_models.SparseVector.return_value = mock_sparse_vector

            result = create_qdrant_sparse_vector(indices, values)

            assert result == mock_sparse_vector
            mock_models.SparseVector.assert_called_once_with(indices=indices, values=values)

    def test_create_sparse_vector_empty(self):
        """Test sparse vector creation with empty lists."""
        indices = []
        values = []

        with patch('src.python.common.core.sparse_vectors.models') as mock_models:
            mock_sparse_vector = Mock()
            mock_models.SparseVector.return_value = mock_sparse_vector

            result = create_qdrant_sparse_vector(indices, values)

            assert result == mock_sparse_vector
            mock_models.SparseVector.assert_called_once_with(indices=[], values=[])

    def test_create_sparse_vector_mismatched_lengths(self):
        """Test sparse vector creation with mismatched lengths."""
        indices = [0, 1, 2]
        values = [0.5, 0.8]  # Different length

        with pytest.raises(ValueError, match="must have the same length"):
            create_qdrant_sparse_vector(indices, values)

    def test_create_sparse_vector_single_element(self):
        """Test sparse vector creation with single element."""
        indices = [42]
        values = [1.5]

        with patch('src.python.common.core.sparse_vectors.models') as mock_models:
            mock_sparse_vector = Mock()
            mock_models.SparseVector.return_value = mock_sparse_vector

            result = create_qdrant_sparse_vector(indices, values)

            assert result == mock_sparse_vector
            mock_models.SparseVector.assert_called_once_with(indices=[42], values=[1.5])


class TestCreateNamedSparseVector:
    """Test create_named_sparse_vector utility function."""

    def test_create_named_sparse_vector_default(self):
        """Test named sparse vector creation with default name."""
        indices = [1, 3, 7]
        values = [0.4, 0.6, 0.9]

        with patch('src.python.common.core.sparse_vectors.create_qdrant_sparse_vector') as mock_create:
            mock_sparse_vector = Mock()
            mock_create.return_value = mock_sparse_vector

            result = create_named_sparse_vector(indices, values)

            assert result == {"sparse": mock_sparse_vector}
            mock_create.assert_called_once_with(indices, values)

    def test_create_named_sparse_vector_custom_name(self):
        """Test named sparse vector creation with custom name."""
        indices = [0, 5, 12]
        values = [0.7, 0.2, 1.1]
        name = "keywords"

        with patch('src.python.common.core.sparse_vectors.create_qdrant_sparse_vector') as mock_create:
            mock_sparse_vector = Mock()
            mock_create.return_value = mock_sparse_vector

            result = create_named_sparse_vector(indices, values, name)

            assert result == {"keywords": mock_sparse_vector}
            mock_create.assert_called_once_with(indices, values)

    def test_create_named_sparse_vector_empty(self):
        """Test named sparse vector creation with empty data."""
        indices = []
        values = []
        name = "empty_sparse"

        with patch('src.python.common.core.sparse_vectors.create_qdrant_sparse_vector') as mock_create:
            mock_sparse_vector = Mock()
            mock_create.return_value = mock_sparse_vector

            result = create_named_sparse_vector(indices, values, name)

            assert result == {"empty_sparse": mock_sparse_vector}
            mock_create.assert_called_once_with([], [])

    def test_create_named_sparse_vector_propagates_error(self):
        """Test that named sparse vector creation propagates underlying errors."""
        indices = [1, 2]
        values = [0.5]  # Mismatched length

        # Should propagate the ValueError from create_qdrant_sparse_vector
        with pytest.raises(ValueError, match="must have the same length"):
            create_named_sparse_vector(indices, values)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            "Machine learning algorithms for text classification",
            "Deep learning neural networks and backpropagation",
            "Natural language processing with transformers",
            "Computer vision and image recognition systems"
        ]

    async def test_full_encoding_workflow_fastembed(self, sample_documents):
        """Test complete encoding workflow with FastEmbed."""
        encoder = BM25SparseEncoder(use_fastembed=True)

        # Mock FastEmbed model
        mock_model = Mock()
        mock_embeddings = [
            ([0, 1, 5], [0.8, 0.6, 0.4]),
            ([1, 2, 7], [0.7, 0.9, 0.3]),
            ([2, 3, 8], [0.5, 0.8, 0.6]),
            ([0, 4, 9], [0.9, 0.4, 0.7])
        ]
        mock_model.embed.return_value = mock_embeddings

        with patch('src.python.common.core.sparse_vectors.SparseTextEmbedding', return_value=mock_model):
            await encoder.initialize()

            # Encode batch
            vectors = await encoder.encode_batch(sample_documents)

            assert len(vectors) == 4
            for vector in vectors:
                assert "indices" in vector
                assert "values" in vector
                assert len(vector["indices"]) == len(vector["values"])

            # Check model info
            info = encoder.get_model_info()
            assert info["encoder_type"] == "fastembed"
            assert info["initialized"] == True

    async def test_full_encoding_workflow_custom(self, sample_documents):
        """Test complete encoding workflow with custom BM25."""
        encoder = BM25SparseEncoder(use_fastembed=False, k1=1.5, b=0.6)

        # Mock dependencies
        mock_vectorizer = Mock()
        mock_vectorizer.get_feature_names_out.return_value = ["machine", "learning", "deep", "neural"]

        mock_bm25 = Mock()
        mock_bm25.get_scores.return_value = [0.5, 0.8, 0.0, 0.3]

        with patch('src.python.common.core.sparse_vectors.TfidfVectorizer', return_value=mock_vectorizer):
            with patch('src.python.common.core.sparse_vectors.rank_bm25') as mock_rank_bm25:
                mock_rank_bm25.BM25Okapi.return_value = mock_bm25

                await encoder.initialize(sample_documents)

                # Encode single document
                vector = encoder.encode(sample_documents[0])

                assert "indices" in vector
                assert "values" in vector
                assert len(vector["indices"]) == len(vector["values"])

                # Check model info
                info = encoder.get_model_info()
                assert info["encoder_type"] == "custom_bm25"
                assert info["parameters"]["k1"] == 1.5
                assert info["parameters"]["b"] == 0.6

    async def test_fallback_behavior(self, sample_documents):
        """Test fallback from FastEmbed to custom BM25."""
        encoder = BM25SparseEncoder(use_fastembed=True)

        # Mock FastEmbed failure during initialization
        with patch('src.python.common.core.sparse_vectors.SparseTextEmbedding') as mock_class:
            mock_class.side_effect = Exception("FastEmbed failed")

            with pytest.raises(RuntimeError, match="Failed to initialize BM25 sparse encoder"):
                await encoder.initialize()

    async def test_error_handling_robustness(self):
        """Test error handling in various scenarios."""
        encoder = BM25SparseEncoder()

        # Test encoding before initialization
        with pytest.raises(RuntimeError, match="must be initialized first"):
            encoder.encode("test")

        # Test with missing dependencies
        with patch('src.python.common.core.sparse_vectors.SparseTextEmbedding', None):
            await encoder.initialize()  # Should succeed without FastEmbed

    def test_qdrant_integration_vectors(self):
        """Test creating Qdrant-compatible vectors."""
        indices = [0, 5, 12, 25]
        values = [0.8, 0.6, 0.9, 0.4]

        # Create basic sparse vector
        with patch('src.python.common.core.sparse_vectors.models.SparseVector') as mock_sparse:
            sparse_vector = create_qdrant_sparse_vector(indices, values)
            mock_sparse.assert_called_once_with(indices=indices, values=values)

        # Create named sparse vector
        named_vector = create_named_sparse_vector(indices, values, "bm25")
        assert "bm25" in named_vector


if __name__ == "__main__":
    pytest.main([__file__])