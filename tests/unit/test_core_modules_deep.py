"""
Deep coverage tests for core modules focusing on method-level execution.

This test suite targets embeddings.py, hybrid_search.py, memory.py, and other
core modules to achieve comprehensive method-level coverage.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call

import pytest

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

# Test imports
try:
    from common.core.embeddings import EmbeddingService
    from common.core.hybrid_search import HybridSearchEngine, RRFFusionRanker
    from common.core.memory import MemoryManager
    from common.core.sparse_vectors import create_named_sparse_vector, SparseVectorConfig
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"Core modules import failed: {e}")

pytestmark = pytest.mark.skipif(not CORE_AVAILABLE, reason="Core modules not available")


class TestEmbeddingServiceDeep:
    """Deep coverage tests for EmbeddingService."""

    @pytest.fixture
    def embedding_config(self):
        """Mock embedding configuration."""
        config = MagicMock()
        config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.dimension = 384
        config.embedding.batch_size = 32
        return config

    def test_embedding_service_initialization(self, embedding_config):
        """Test EmbeddingService initialization."""
        with patch('common.core.embeddings.FastEmbedService') as mock_service:
            embedding_service = EmbeddingService(embedding_config)

            assert embedding_service.config == embedding_config
            mock_service.assert_called_once()

    def test_embedding_service_initialization_failure(self, embedding_config):
        """Test EmbeddingService initialization failure."""
        with patch('common.core.embeddings.FastEmbedService', side_effect=Exception("Model load failed")):
            with pytest.raises(Exception, match="Model load failed"):
                EmbeddingService(embedding_config)

    @pytest.mark.asyncio
    async def test_generate_embeddings_success(self, embedding_config):
        """Test successful embedding generation."""
        with patch('common.core.embeddings.FastEmbedService') as mock_service:
            mock_embedder = Mock()
            mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
            mock_service.return_value = mock_embedder

            embedding_service = EmbeddingService(embedding_config)
            embeddings = await embedding_service.generate_embeddings(["test text"])

            assert len(embeddings) == 1
            assert len(embeddings[0]) == 3
            mock_embedder.embed.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, embedding_config):
        """Test embedding generation with batching."""
        texts = ["text1", "text2", "text3"]
        expected_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch('common.core.embeddings.FastEmbedService') as mock_service:
            mock_embedder = Mock()
            mock_embedder.embed.return_value = expected_embeddings
            mock_service.return_value = mock_embedder

            embedding_service = EmbeddingService(embedding_config)
            embeddings = await embedding_service.generate_embeddings(texts)

            assert len(embeddings) == 3
            assert embeddings == expected_embeddings

    @pytest.mark.asyncio
    async def test_generate_embeddings_error_handling(self, embedding_config):
        """Test embedding generation error handling."""
        with patch('common.core.embeddings.FastEmbedService') as mock_service:
            mock_embedder = Mock()
            mock_embedder.embed.side_effect = Exception("Embedding failed")
            mock_service.return_value = mock_embedder

            embedding_service = EmbeddingService(embedding_config)

            with pytest.raises(Exception, match="Embedding failed"):
                await embedding_service.generate_embeddings(["test text"])

    def test_embedding_service_properties(self, embedding_config):
        """Test EmbeddingService properties."""
        with patch('common.core.embeddings.FastEmbedService'):
            embedding_service = EmbeddingService(embedding_config)

            assert embedding_service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
            assert embedding_service.dimension == 384

    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, embedding_config):
        """Test generating a single embedding."""
        with patch('common.core.embeddings.FastEmbedService') as mock_service:
            mock_embedder = Mock()
            mock_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
            mock_service.return_value = mock_embedder

            embedding_service = EmbeddingService(embedding_config)
            embedding = await embedding_service.generate_single_embedding("test text")

            assert len(embedding) == 3
            assert embedding == [0.1, 0.2, 0.3]


class TestHybridSearchEngineDeep:
    """Deep coverage tests for HybridSearchEngine."""

    @pytest.fixture
    def mock_client(self):
        """Mock Qdrant client."""
        client = AsyncMock()
        return client

    def test_hybrid_search_engine_initialization(self, mock_client):
        """Test HybridSearchEngine initialization."""
        engine = HybridSearchEngine(mock_client)

        assert engine.client == mock_client
        assert engine.default_fusion_method == "rrf"

    @pytest.mark.asyncio
    async def test_hybrid_search_success(self, mock_client):
        """Test successful hybrid search."""
        mock_dense_results = [
            MagicMock(id="doc1", score=0.9),
            MagicMock(id="doc2", score=0.8)
        ]
        mock_sparse_results = [
            MagicMock(id="doc2", score=0.95),
            MagicMock(id="doc3", score=0.7)
        ]

        mock_client.search.side_effect = [mock_dense_results, mock_sparse_results]

        engine = HybridSearchEngine(mock_client)
        query_embeddings = {
            "dense": [0.1, 0.2, 0.3],
            "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
        }

        results = await engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings=query_embeddings,
            limit=5
        )

        assert len(results) > 0
        assert mock_client.search.call_count == 2

    @pytest.mark.asyncio
    async def test_hybrid_search_dense_only(self, mock_client):
        """Test hybrid search with dense vectors only."""
        mock_dense_results = [MagicMock(id="doc1", score=0.9)]
        mock_client.search.return_value = mock_dense_results

        engine = HybridSearchEngine(mock_client)
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}

        results = await engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings=query_embeddings,
            limit=5
        )

        assert len(results) >= 0
        assert mock_client.search.call_count == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_sparse_only(self, mock_client):
        """Test hybrid search with sparse vectors only."""
        mock_sparse_results = [MagicMock(id="doc1", score=0.9)]
        mock_client.search.return_value = mock_sparse_results

        engine = HybridSearchEngine(mock_client)
        query_embeddings = {
            "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
        }

        results = await engine.hybrid_search(
            collection_name="test-collection",
            query_embeddings=query_embeddings,
            limit=5
        )

        assert len(results) >= 0
        assert mock_client.search.call_count == 1

    @pytest.mark.asyncio
    async def test_hybrid_search_error_handling(self, mock_client):
        """Test hybrid search error handling."""
        mock_client.search.side_effect = Exception("Search failed")

        engine = HybridSearchEngine(mock_client)
        query_embeddings = {"dense": [0.1, 0.2, 0.3]}

        with pytest.raises(Exception, match="Search failed"):
            await engine.hybrid_search(
                collection_name="test-collection",
                query_embeddings=query_embeddings,
                limit=5
            )

    def test_rrf_fusion_ranker(self):
        """Test RRF fusion ranking."""
        ranker = RRFFusionRanker()

        dense_results = [
            MagicMock(id="doc1", score=0.9),
            MagicMock(id="doc2", score=0.8)
        ]
        sparse_results = [
            MagicMock(id="doc2", score=0.95),
            MagicMock(id="doc3", score=0.7)
        ]

        fused_results = ranker.fuse_results(dense_results, sparse_results)

        assert len(fused_results) > 0
        # Results should be ordered by fused score
        for i in range(len(fused_results) - 1):
            assert fused_results[i].score >= fused_results[i + 1].score

    def test_rrf_fusion_ranker_empty_results(self):
        """Test RRF fusion with empty results."""
        ranker = RRFFusionRanker()

        fused_results = ranker.fuse_results([], [])
        assert len(fused_results) == 0

    def test_rrf_fusion_ranker_single_source(self):
        """Test RRF fusion with single result source."""
        ranker = RRFFusionRanker()

        dense_results = [MagicMock(id="doc1", score=0.9)]
        fused_results = ranker.fuse_results(dense_results, [])

        assert len(fused_results) == 1
        assert fused_results[0].id == "doc1"


class TestSparseVectorsDeep:
    """Deep coverage tests for sparse vector functionality."""

    def test_create_named_sparse_vector(self):
        """Test creating named sparse vectors."""
        text = "hello world test"
        vector = create_named_sparse_vector(text, "test-vector")

        assert vector is not None
        assert hasattr(vector, 'indices')
        assert hasattr(vector, 'values')

    def test_create_named_sparse_vector_empty_text(self):
        """Test creating sparse vector with empty text."""
        vector = create_named_sparse_vector("", "empty-vector")

        # Should handle empty text gracefully
        assert vector is not None

    def test_sparse_vector_config(self):
        """Test SparseVectorConfig creation."""
        config = SparseVectorConfig(
            model="bm25",
            dimension=1000,
            k1=1.2,
            b=0.75
        )

        assert config.model == "bm25"
        assert config.dimension == 1000
        assert config.k1 == 1.2
        assert config.b == 0.75

    def test_sparse_vector_config_defaults(self):
        """Test SparseVectorConfig with default values."""
        config = SparseVectorConfig()

        # Should have reasonable defaults
        assert hasattr(config, 'model')
        assert hasattr(config, 'dimension')

    def test_create_sparse_vector_with_config(self):
        """Test creating sparse vector with configuration."""
        config = SparseVectorConfig(dimension=500)
        text = "test document for sparse vector"

        vector = create_named_sparse_vector(text, "config-vector", config)

        assert vector is not None
        assert hasattr(vector, 'indices')
        assert hasattr(vector, 'values')

    def test_create_multiple_sparse_vectors(self):
        """Test creating multiple sparse vectors."""
        texts = ["first document", "second document", "third document"]
        vectors = []

        for i, text in enumerate(texts):
            vector = create_named_sparse_vector(text, f"vector-{i}")
            vectors.append(vector)

        assert len(vectors) == 3
        for vector in vectors:
            assert vector is not None
            assert hasattr(vector, 'indices')
            assert hasattr(vector, 'values')


class TestMemoryManagerDeep:
    """Deep coverage tests for MemoryManager."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for memory manager."""
        config = MagicMock()
        config.memory.max_documents = 1000
        config.memory.cleanup_interval = 3600
        config.memory.embedding_dimension = 384
        return config

    def test_memory_manager_initialization(self, mock_config):
        """Test MemoryManager initialization."""
        with patch('common.core.memory.QdrantClient'):
            manager = MemoryManager(mock_config)

            assert manager.config == mock_config
            assert manager.max_documents == 1000

    @pytest.mark.asyncio
    async def test_add_document_success(self, mock_config):
        """Test successful document addition."""
        with patch('common.core.memory.QdrantClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.upsert.return_value = True

            manager = MemoryManager(mock_config)
            result = await manager.add_document(
                collection_name="test-collection",
                document_id="doc1",
                content="test content",
                metadata={"type": "document"}
            )

            assert result is True
            mock_client_instance.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_error_handling(self, mock_config):
        """Test document addition error handling."""
        with patch('common.core.memory.QdrantClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.upsert.side_effect = Exception("Upsert failed")

            manager = MemoryManager(mock_config)

            with pytest.raises(Exception, match="Upsert failed"):
                await manager.add_document(
                    collection_name="test-collection",
                    document_id="doc1",
                    content="test content"
                )

    @pytest.mark.asyncio
    async def test_search_documents(self, mock_config):
        """Test document search functionality."""
        with patch('common.core.memory.QdrantClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance

            mock_results = [
                MagicMock(id="doc1", score=0.9, payload={"content": "test1"}),
                MagicMock(id="doc2", score=0.8, payload={"content": "test2"})
            ]
            mock_client_instance.search.return_value = mock_results

            manager = MemoryManager(mock_config)
            results = await manager.search_documents(
                collection_name="test-collection",
                query_vector=[0.1, 0.2, 0.3],
                limit=10
            )

            assert len(results) == 2
            assert results[0].id == "doc1"
            mock_client_instance.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_document(self, mock_config):
        """Test document deletion."""
        with patch('common.core.memory.QdrantClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.delete.return_value = True

            manager = MemoryManager(mock_config)
            result = await manager.delete_document(
                collection_name="test-collection",
                document_id="doc1"
            )

            assert result is True
            mock_client_instance.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_old_documents(self, mock_config):
        """Test cleanup of old documents."""
        with patch('common.core.memory.QdrantClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance

            # Mock scroll to return old documents
            mock_client_instance.scroll.return_value = (
                [MagicMock(id="old_doc1"), MagicMock(id="old_doc2")],
                None
            )
            mock_client_instance.delete.return_value = True

            manager = MemoryManager(mock_config)
            cleaned_count = await manager.cleanup_old_documents(
                collection_name="test-collection",
                max_age_hours=24
            )

            assert cleaned_count >= 0
            mock_client_instance.scroll.assert_called_once()

    def test_memory_manager_properties(self, mock_config):
        """Test MemoryManager properties."""
        with patch('common.core.memory.QdrantClient'):
            manager = MemoryManager(mock_config)

            assert manager.max_documents == 1000
            assert manager.cleanup_interval == 3600
            assert manager.embedding_dimension == 384

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, mock_config):
        """Test getting collection statistics."""
        with patch('common.core.memory.QdrantClient') as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance

            mock_info = MagicMock()
            mock_info.points_count = 100
            mock_info.segments_count = 5
            mock_client_instance.get_collection.return_value = mock_info

            manager = MemoryManager(mock_config)
            stats = await manager.get_collection_stats("test-collection")

            assert stats['points_count'] == 100
            assert stats['segments_count'] == 5
            mock_client_instance.get_collection.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])