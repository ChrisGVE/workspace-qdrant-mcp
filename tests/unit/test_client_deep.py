"""
Deep coverage tests for client.py module focusing on method-level execution.

This test suite targets the QdrantWorkspaceClient class and its methods to achieve
comprehensive coverage of client functionality, connection handling, and error scenarios.
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
    from common.core.client import QdrantWorkspaceClient
    from common.core.config import Config
    CLIENT_AVAILABLE = True
except ImportError as e:
    CLIENT_AVAILABLE = False
    print(f"Client import failed: {e}")

pytestmark = pytest.mark.skipif(not CLIENT_AVAILABLE, reason="Client module not available")


class TestClientDeepCoverage:
    """Deep coverage tests for QdrantWorkspaceClient."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for client testing."""
        config = MagicMock(spec=Config)
        config.qdrant_client_config.url = "http://localhost:6333"
        config.qdrant_client_config.api_key = None
        config.qdrant_client_config.timeout = 30
        config.qdrant_client_config.prefer_grpc = False
        config.workspace.global_collections = ["global", "shared"]
        config.workspace.project_collections = ["notes", "docs", "scratchbook"]
        config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.dimension = 384
        config.project_detection.enabled = True
        config.project_detection.github_user = "testuser"
        return config

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client."""
        client = AsyncMock()
        client.get_collections.return_value = MagicMock()
        client.collection_exists.return_value = False
        client.create_collection.return_value = True
        client.close.return_value = None
        return client

    def test_client_initialization(self, mock_config):
        """Test QdrantWorkspaceClient initialization."""
        client = QdrantWorkspaceClient(mock_config)

        assert client.config == mock_config
        assert client.client is None
        assert client.collection_manager is None
        assert client.embedding_service is not None

    def test_client_initialization_without_config(self):
        """Test client initialization without config."""
        client = QdrantWorkspaceClient(None)

        assert client.config is None
        assert client.client is None
        assert client.collection_manager is None

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_qdrant_client):
        """Test successful client initialization."""
        with patch('common.core.client.QdrantClient', return_value=mock_qdrant_client):
            with patch('common.core.client.WorkspaceCollectionManager') as mock_collection_manager:
                with patch('common.core.client.EmbeddingService') as mock_embedding_service:
                    mock_collection_manager.return_value = AsyncMock()
                    mock_embedding_service.return_value = AsyncMock()

                    client = QdrantWorkspaceClient(mock_config)
                    await client.initialize()

                    assert client.client == mock_qdrant_client
                    assert client.collection_manager is not None
                    mock_collection_manager.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_api_key(self, mock_config, mock_qdrant_client):
        """Test client initialization with API key."""
        mock_config.qdrant_client_config.api_key = "test-api-key"

        with patch('common.core.client.QdrantClient', return_value=mock_qdrant_client) as mock_client_class:
            with patch('common.core.client.WorkspaceCollectionManager'):
                with patch('common.core.client.EmbeddingService'):
                    client = QdrantWorkspaceClient(mock_config)
                    await client.initialize()

                    # Verify QdrantClient was called with API key
                    mock_client_class.assert_called_once()
                    call_kwargs = mock_client_class.call_args[1]
                    assert call_kwargs['api_key'] == "test-api-key"

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, mock_config):
        """Test client initialization with connection failure."""
        with patch('common.core.client.QdrantClient', side_effect=Exception("Connection failed")):
            client = QdrantWorkspaceClient(mock_config)

            with pytest.raises(Exception, match="Connection failed"):
                await client.initialize()

    @pytest.mark.asyncio
    async def test_close_client(self, mock_config, mock_qdrant_client):
        """Test client close functionality."""
        with patch('common.core.client.QdrantClient', return_value=mock_qdrant_client):
            with patch('common.core.client.WorkspaceCollectionManager'):
                with patch('common.core.client.EmbeddingService'):
                    client = QdrantWorkspaceClient(mock_config)
                    await client.initialize()

                    await client.close()

                    mock_qdrant_client.close.assert_called_once()
                    assert client.client is None

    @pytest.mark.asyncio
    async def test_close_without_client(self, mock_config):
        """Test close when no client is initialized."""
        client = QdrantWorkspaceClient(mock_config)

        # Should not raise exception
        await client.close()

    def test_list_collections_without_initialization(self, mock_config):
        """Test list_collections before initialization."""
        client = QdrantWorkspaceClient(mock_config)

        collections = client.list_collections()
        assert collections == []

    def test_list_collections_with_client(self, mock_config, mock_qdrant_client):
        """Test list_collections with initialized client."""
        mock_collections = MagicMock()
        mock_collections.collections = [
            MagicMock(name="collection1"),
            MagicMock(name="collection2")
        ]
        mock_qdrant_client.get_collections.return_value = mock_collections

        with patch('common.core.client.QdrantClient', return_value=mock_qdrant_client):
            with patch('common.core.client.WorkspaceCollectionManager'):
                with patch('common.core.client.EmbeddingService'):
                    client = QdrantWorkspaceClient(mock_config)
                    client.client = mock_qdrant_client

                    collections = client.list_collections()

                    assert len(collections) == 2
                    mock_qdrant_client.get_collections.assert_called_once()

    def test_list_collections_error_handling(self, mock_config, mock_qdrant_client):
        """Test list_collections error handling."""
        mock_qdrant_client.get_collections.side_effect = Exception("Qdrant error")

        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client

        collections = client.list_collections()
        assert collections == []

    @pytest.mark.asyncio
    async def test_get_status_without_client(self, mock_config):
        """Test get_status without initialized client."""
        client = QdrantWorkspaceClient(mock_config)

        status = await client.get_status()

        assert status['client_initialized'] is False
        assert status['collections_count'] == 0

    @pytest.mark.asyncio
    async def test_get_status_with_client(self, mock_config, mock_qdrant_client):
        """Test get_status with initialized client."""
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(name="test-collection")]
        mock_qdrant_client.get_collections.return_value = mock_collections

        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client

        status = await client.get_status()

        assert status['client_initialized'] is True
        assert status['collections_count'] == 1
        assert 'qdrant_url' in status

    @pytest.mark.asyncio
    async def test_get_status_with_project_info(self, mock_config, mock_qdrant_client):
        """Test get_status with project information."""
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_qdrant_client.get_collections.return_value = mock_collections

        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.project_info = {
            'name': 'test-project',
            'root': '/path/to/project'
        }

        status = await client.get_status()

        assert status['project_name'] == 'test-project'
        assert status['project_root'] == '/path/to/project'

    def test_create_collection_without_client(self, mock_config):
        """Test create_collection without initialized client."""
        client = QdrantWorkspaceClient(mock_config)

        # Should not raise exception but return False
        result = client.create_collection("test-collection")
        assert result is False

    def test_create_collection_with_client(self, mock_config, mock_qdrant_client):
        """Test create_collection with initialized client."""
        mock_qdrant_client.collection_exists.return_value = False
        mock_qdrant_client.create_collection.return_value = True

        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client

        result = client.create_collection("test-collection")

        assert result is True
        mock_qdrant_client.collection_exists.assert_called_once_with("test-collection")
        mock_qdrant_client.create_collection.assert_called_once()

    def test_create_collection_already_exists(self, mock_config, mock_qdrant_client):
        """Test create_collection when collection already exists."""
        mock_qdrant_client.collection_exists.return_value = True

        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client

        result = client.create_collection("test-collection")

        assert result is True  # Already exists is considered success
        mock_qdrant_client.collection_exists.assert_called_once_with("test-collection")
        mock_qdrant_client.create_collection.assert_not_called()

    def test_create_collection_error_handling(self, mock_config, mock_qdrant_client):
        """Test create_collection error handling."""
        mock_qdrant_client.collection_exists.return_value = False
        mock_qdrant_client.create_collection.side_effect = Exception("Creation failed")

        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client

        result = client.create_collection("test-collection")

        assert result is False

    def test_collection_exists_without_client(self, mock_config):
        """Test collection_exists without initialized client."""
        client = QdrantWorkspaceClient(mock_config)

        result = client.collection_exists("test-collection")
        assert result is False

    def test_collection_exists_with_client(self, mock_config, mock_qdrant_client):
        """Test collection_exists with initialized client."""
        mock_qdrant_client.collection_exists.return_value = True

        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client

        result = client.collection_exists("test-collection")

        assert result is True
        mock_qdrant_client.collection_exists.assert_called_once_with("test-collection")

    def test_collection_exists_error_handling(self, mock_config, mock_qdrant_client):
        """Test collection_exists error handling."""
        mock_qdrant_client.collection_exists.side_effect = Exception("Check failed")

        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client

        result = client.collection_exists("test-collection")
        assert result is False

    @pytest.mark.asyncio
    async def test_initialize_with_ssl_config(self, mock_config):
        """Test initialization with SSL configuration."""
        mock_config.qdrant_client_config.use_ssl = True
        mock_config.qdrant_client_config.ssl_cert_path = "/path/to/cert"

        with patch('common.core.client.create_secure_qdrant_config') as mock_ssl_config:
            with patch('common.core.client.QdrantClient') as mock_client_class:
                with patch('common.core.client.WorkspaceCollectionManager'):
                    with patch('common.core.client.EmbeddingService'):
                        mock_ssl_config.return_value = {"ssl": "config"}
                        mock_client = AsyncMock()
                        mock_client_class.return_value = mock_client

                        client = QdrantWorkspaceClient(mock_config)
                        await client.initialize()

                        mock_ssl_config.assert_called_once_with(mock_config)

    @pytest.mark.asyncio
    async def test_initialize_embedding_service_failure(self, mock_config):
        """Test initialization when embedding service fails."""
        with patch('common.core.client.QdrantClient') as mock_client_class:
            with patch('common.core.client.WorkspaceCollectionManager'):
                with patch('common.core.client.EmbeddingService', side_effect=Exception("Embedding service failed")):
                    mock_client_class.return_value = AsyncMock()

                    client = QdrantWorkspaceClient(mock_config)

                    with pytest.raises(Exception, match="Embedding service failed"):
                        await client.initialize()

    def test_client_properties_access(self, mock_config):
        """Test accessing client properties."""
        client = QdrantWorkspaceClient(mock_config)

        # Test property access without initialization
        assert client.is_initialized is False
        assert client.embedding_model == mock_config.embedding.model_name
        assert client.qdrant_url == mock_config.qdrant_client_config.url

    @pytest.mark.asyncio
    async def test_context_manager_usage(self, mock_config, mock_qdrant_client):
        """Test client as context manager."""
        with patch('common.core.client.QdrantClient', return_value=mock_qdrant_client):
            with patch('common.core.client.WorkspaceCollectionManager'):
                with patch('common.core.client.EmbeddingService'):
                    async with QdrantWorkspaceClient(mock_config) as client:
                        assert client.client is not None

                    # Should be closed after context exit
                    mock_qdrant_client.close.assert_called_once()

    def test_repr_and_str_methods(self, mock_config):
        """Test string representations of client."""
        client = QdrantWorkspaceClient(mock_config)

        repr_str = repr(client)
        str_str = str(client)

        assert "QdrantWorkspaceClient" in repr_str
        assert "QdrantWorkspaceClient" in str_str
        assert mock_config.qdrant_client_config.url in repr_str

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_config, mock_qdrant_client):
        """Test concurrent client operations."""
        with patch('common.core.client.QdrantClient', return_value=mock_qdrant_client):
            with patch('common.core.client.WorkspaceCollectionManager'):
                with patch('common.core.client.EmbeddingService'):
                    client = QdrantWorkspaceClient(mock_config)
                    await client.initialize()

                    # Run multiple operations concurrently
                    tasks = [
                        client.get_status(),
                        client.get_status(),
                        client.get_status()
                    ]

                    results = await asyncio.gather(*tasks)
                    assert len(results) == 3
                    assert all(result['client_initialized'] for result in results)

    def test_client_with_minimal_config(self):
        """Test client with minimal configuration."""
        minimal_config = MagicMock()
        minimal_config.qdrant_client_config.url = "http://localhost:6333"
        minimal_config.qdrant_client_config.api_key = None
        minimal_config.workspace = None
        minimal_config.embedding = None

        client = QdrantWorkspaceClient(minimal_config)
        assert client.config == minimal_config


if __name__ == "__main__":
    pytest.main([__file__])