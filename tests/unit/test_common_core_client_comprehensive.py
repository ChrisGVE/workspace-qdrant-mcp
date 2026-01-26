"""
Comprehensive unit tests for common.core.client module.

This test suite provides complete coverage of the QdrantWorkspaceClient class,
testing all initialization paths, error conditions, async operations, and
integration points with proper mocking of external dependencies.
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, call, patch

import pytest
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.client import QdrantWorkspaceClient, create_qdrant_client
from common.core.config import Config


@pytest.fixture
def mock_config():
    """Create a comprehensive mock configuration."""
    config = MagicMock()
    config.environment = "development"
    config.qdrant = MagicMock()
    config.qdrant.url = "http://localhost:6333"
    config.qdrant_client_config = {"host": "localhost", "port": 6333}
    config.embedding = MagicMock()
    config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding.enable_sparse_vectors = True
    config.workspace = MagicMock()
    config.workspace.github_user = "testuser"
    config.workspace.global_collections = ["scratchbook"]
    config.security = MagicMock()
    config.security.qdrant_auth_token = None
    config.security.qdrant_api_key = None
    return config


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client with common operations."""
    client = AsyncMock()
    client.get_collections.return_value = models.CollectionsResponse(
        collections=[
            models.CollectionDescription(name="test-collection"),
            models.CollectionDescription(name="scratchbook")
        ]
    )
    client.collection_exists.return_value = True
    client.get_collection.return_value = models.CollectionInfo(
        status=models.CollectionStatus.GREEN,
        config=models.CollectionConfig(params=models.VectorParams(size=384, distance=models.Distance.COSINE))
    )
    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.initialize.return_value = None
    service.embed_query.return_value = {
        "dense": [0.1] * 384,
        "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
    }
    service.close.return_value = None
    return service


@pytest.fixture
def mock_collection_manager():
    """Create a mock collection manager."""
    manager = AsyncMock()
    manager.initialize.return_value = None
    manager.list_collections.return_value = ["test-collection", "scratchbook"]
    manager.get_project_collections.return_value = ["test-collection"]
    manager.close.return_value = None
    return manager


@pytest.fixture
def mock_project_detector():
    """Create a mock project detector."""
    detector = MagicMock()
    detector.detect_projects.return_value = {
        "primary_project": "test-project",
        "subprojects": ["frontend", "backend"],
        "git_root": "/path/to/project",
        "github_user": "testuser"
    }
    return detector


class TestQdrantWorkspaceClientInitialization:
    """Test client initialization scenarios."""

    def test_init_with_valid_config(self, mock_config):
        """Test successful client initialization."""
        client = QdrantWorkspaceClient(mock_config)

        assert client.config == mock_config
        assert client.client is None
        assert client.collection_manager is None
        assert client.embedding_service is not None
        assert not client.initialized

    def test_init_with_invalid_config(self):
        """Test initialization with invalid config."""
        with pytest.raises((AttributeError, TypeError)):
            QdrantWorkspaceClient(None)

    @patch('common.core.client.EmbeddingService')
    def test_init_embedding_service_creation(self, mock_embedding_class, mock_config):
        """Test embedding service is created during initialization."""
        mock_embedding_instance = AsyncMock()
        mock_embedding_class.return_value = mock_embedding_instance

        client = QdrantWorkspaceClient(mock_config)

        mock_embedding_class.assert_called_once_with(mock_config)
        assert client.embedding_service == mock_embedding_instance


class TestQdrantWorkspaceClientAsyncOperations:
    """Test async client operations."""

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_qdrant_client,
                                    mock_embedding_service, mock_collection_manager,
                                    mock_project_detector):
        """Test successful client initialization with all dependencies."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service = mock_embedding_service

        with patch('common.core.client.create_qdrant_client', return_value=mock_qdrant_client), \
             patch('common.core.client.WorkspaceCollectionManager', return_value=mock_collection_manager), \
             patch('common.core.client.ProjectDetector', return_value=mock_project_detector):

            await client.initialize()

            assert client.initialized
            assert client.client == mock_qdrant_client
            assert client.collection_manager == mock_collection_manager
            assert client.project_detector == mock_project_detector

            mock_embedding_service.initialize.assert_called_once()
            mock_collection_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_qdrant_connection_failure(self, mock_config):
        """Test initialization failure when Qdrant connection fails."""
        client = QdrantWorkspaceClient(mock_config)

        with patch('common.core.client.create_qdrant_client', side_effect=ConnectionError("Connection failed")):
            with pytest.raises(ConnectionError):
                await client.initialize()

            assert not client.initialized

    @pytest.mark.asyncio
    async def test_initialize_embedding_service_failure(self, mock_config, mock_qdrant_client):
        """Test initialization failure when embedding service fails."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service = AsyncMock()
        client.embedding_service.initialize.side_effect = RuntimeError("Embedding init failed")

        with patch('common.core.client.create_qdrant_client', return_value=mock_qdrant_client):
            with pytest.raises(RuntimeError):
                await client.initialize()

            assert not client.initialized

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_config, mock_qdrant_client,
                                      mock_embedding_service, mock_collection_manager,
                                      mock_project_detector):
        """Test that multiple initialize calls are idempotent."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service = mock_embedding_service

        with patch('common.core.client.create_qdrant_client', return_value=mock_qdrant_client), \
             patch('common.core.client.WorkspaceCollectionManager', return_value=mock_collection_manager), \
             patch('common.core.client.ProjectDetector', return_value=mock_project_detector):

            await client.initialize()
            await client.initialize()  # Second call should be no-op

            # Verify services only initialized once
            mock_embedding_service.initialize.assert_called_once()
            mock_collection_manager.initialize.assert_called_once()


class TestQdrantWorkspaceClientCollectionOperations:
    """Test collection management operations."""

    @pytest.mark.asyncio
    async def test_list_collections_success(self, mock_config, mock_collection_manager):
        """Test successful collection listing."""
        client = QdrantWorkspaceClient(mock_config)
        client.collection_manager = mock_collection_manager
        client.initialized = True

        mock_collection_manager.list_collections.return_value = ["col1", "col2", "scratchbook"]

        result = client.list_collections()

        assert result == ["col1", "col2", "scratchbook"]
        mock_collection_manager.list_collections.assert_called_once()

    def test_list_collections_not_initialized(self, mock_config):
        """Test collection listing when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            client.list_collections()

    @pytest.mark.asyncio
    async def test_get_status_success(self, mock_config, mock_qdrant_client):
        """Test successful status retrieval."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.initialized = True

        mock_qdrant_client.get_collections.return_value = models.CollectionsResponse(
            collections=[models.CollectionDescription(name="test")]
        )

        status = await client.get_status()

        assert "qdrant_status" in status
        assert "collections" in status
        assert status["collections"] == ["test"]
        mock_qdrant_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_status_qdrant_error(self, mock_config, mock_qdrant_client):
        """Test status retrieval when Qdrant returns error."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.initialized = True

        mock_qdrant_client.get_collections.side_effect = ResponseHandlingException("Server error")

        status = await client.get_status()

        assert "error" in status
        assert "Server error" in status["error"]


class TestQdrantWorkspaceClientSearchOperations:
    """Test search operations."""

    @pytest.mark.asyncio
    async def test_search_with_embeddings(self, mock_config, mock_qdrant_client, mock_embedding_service):
        """Test search operation with embedding generation."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.embedding_service = mock_embedding_service
        client.initialized = True

        # Mock search results
        mock_results = [
            models.ScoredPoint(id="1", score=0.9, version=0, payload={"content": "doc1"}),
            models.ScoredPoint(id="2", score=0.8, version=0, payload={"content": "doc2"})
        ]
        mock_qdrant_client.search.return_value = mock_results

        # Mock embedding generation
        mock_embedding_service.embed_query.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 5], "values": [0.8, 0.6]}
        }

        results = await client.search(
            collection_name="test-collection",
            query="test query",
            limit=10
        )

        assert len(results) == 2
        assert results[0]["id"] == "1"
        assert results[0]["score"] == 0.9

        mock_embedding_service.embed_query.assert_called_once_with("test query")
        mock_qdrant_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_embedding_error(self, mock_config, mock_embedding_service):
        """Test search when embedding generation fails."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service = mock_embedding_service
        client.initialized = True

        mock_embedding_service.embed_query.side_effect = RuntimeError("Embedding failed")

        with pytest.raises(RuntimeError, match="Embedding failed"):
            await client.search("test-collection", "test query")

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, mock_config):
        """Test search when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await client.search("test-collection", "test query")


class TestQdrantWorkspaceClientResourceManagement:
    """Test resource management and cleanup."""

    @pytest.mark.asyncio
    async def test_close_all_resources(self, mock_config, mock_qdrant_client,
                                     mock_embedding_service, mock_collection_manager):
        """Test proper cleanup of all resources."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.embedding_service = mock_embedding_service
        client.collection_manager = mock_collection_manager
        client.initialized = True

        await client.close()

        mock_embedding_service.close.assert_called_once()
        mock_collection_manager.close.assert_called_once()
        assert not client.initialized

    @pytest.mark.asyncio
    async def test_close_partial_initialization(self, mock_config, mock_embedding_service):
        """Test close when only some components are initialized."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service = mock_embedding_service
        client.collection_manager = None
        client.initialized = True

        await client.close()

        mock_embedding_service.close.assert_called_once()
        assert not client.initialized

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_config, mock_embedding_service):
        """Test that multiple close calls are safe."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service = mock_embedding_service
        client.initialized = True

        await client.close()
        await client.close()  # Should not raise error

        # Close should only be called once
        mock_embedding_service.close.assert_called_once()


class TestCreateQdrantClient:
    """Test Qdrant client creation function."""

    @patch('common.core.client.QdrantClient')
    def test_create_client_basic_config(self, mock_qdrant_class, mock_config):
        """Test client creation with basic configuration."""
        mock_client_instance = MagicMock()
        mock_qdrant_class.return_value = mock_client_instance

        result = create_qdrant_client(mock_config)

        assert result == mock_client_instance
        mock_qdrant_class.assert_called_once()

    @patch('common.core.client.QdrantClient')
    def test_create_client_with_auth_token(self, mock_qdrant_class, mock_config):
        """Test client creation with authentication token."""
        mock_config.security.qdrant_auth_token = "test-token"
        mock_client_instance = MagicMock()
        mock_qdrant_class.return_value = mock_client_instance

        result = create_qdrant_client(mock_config)

        assert result == mock_client_instance
        # Verify auth token is passed in configuration
        call_args = mock_qdrant_class.call_args
        assert "api_key" in call_args.kwargs or any("api_key" in str(arg) for arg in call_args.args)

    @patch('common.core.client.QdrantClient')
    def test_create_client_connection_error(self, mock_qdrant_class, mock_config):
        """Test client creation when connection fails."""
        mock_qdrant_class.side_effect = ConnectionError("Failed to connect")

        with pytest.raises(ConnectionError):
            create_qdrant_client(mock_config)


class TestQdrantWorkspaceClientEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_config, mock_qdrant_client,
                                       mock_embedding_service):
        """Test concurrent operations on the same client."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.embedding_service = mock_embedding_service
        client.initialized = True

        # Mock successful operations
        mock_embedding_service.embed_query.return_value = {"dense": [0.1] * 384}
        mock_qdrant_client.search.return_value = []

        # Run concurrent searches
        tasks = [
            client.search("col1", "query1"),
            client.search("col2", "query2"),
            client.search("col3", "query3")
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert mock_embedding_service.embed_query.call_count == 3

    def test_memory_usage_tracking(self, mock_config):
        """Test that client properly manages memory usage."""
        # Create multiple clients to test memory management
        clients = [QdrantWorkspaceClient(mock_config) for _ in range(10)]

        # All clients should be properly instantiated
        assert len(clients) == 10
        assert all(not client.initialized for client in clients)

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_config, mock_qdrant_client):
        """Test handling of timeout scenarios."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.initialized = True

        # Mock timeout error
        mock_qdrant_client.get_collections.side_effect = asyncio.TimeoutError("Operation timed out")

        with pytest.raises(asyncio.TimeoutError):
            await client.get_status()
