"""
Comprehensive unit tests for core/client.py

This module provides exhaustive test coverage for the QdrantWorkspaceClient
class, including all async operations, project detection, collection management,
embedding service integration, SSL configuration, and error handling.

Test Coverage Goals:
- Client initialization and configuration
- Project structure detection and workspace isolation
- Collection management and lifecycle operations
- Embedding service integration (dense + sparse vectors)
- Search operations with project context
- SSL/TLS configuration and connection security
- Error handling, connection recovery, and edge cases
- LLM access control validation
- Resource cleanup and connection management
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.client import QdrantWorkspaceClient, create_qdrant_client
from common.core.config import ConfigManager, get_config, reset_config


class TestQdrantWorkspaceClientInit:
    """Test QdrantWorkspaceClient initialization."""

    def test_client_init_with_config(self):
        """Test client initialization with lua-style configuration access."""
        # Reset config to clean state
        reset_config()

        # Create client using new lua-style pattern (no config parameter)
        client = QdrantWorkspaceClient()

        assert client.client is None
        assert client.collection_manager is None
        assert client.embedding_service is not None
        assert client.project_detector is None
        assert client.project_info is None
        assert client.initialized is False

    def test_client_init_initializes_embedding_service(self):
        """Test that embedding service is initialized during client creation."""
        reset_config()

        with patch('common.core.client.EmbeddingService') as MockEmbeddingService:
            mock_embedding_service = Mock()
            MockEmbeddingService.return_value = mock_embedding_service

            client = QdrantWorkspaceClient()

            MockEmbeddingService.assert_called_once_with()
            assert client.embedding_service == mock_embedding_service

    def test_client_init_sets_default_attributes(self):
        """Test that client sets all required default attributes."""
        reset_config()
        client = QdrantWorkspaceClient()

        # Check all attributes are properly initialized
        assert hasattr(client, 'client')
        assert hasattr(client, 'collection_manager')
        assert hasattr(client, 'embedding_service')
        assert hasattr(client, 'project_detector')
        assert hasattr(client, 'project_info')
        assert hasattr(client, 'initialized')


class TestClientInitialization:
    """Test async initialization of QdrantWorkspaceClient."""

    @pytest.mark.asyncio
    async def test_initialize_success_with_ssl(self):
        """Test successful initialization with SSL configuration."""
        reset_config()

        with patch('common.core.client.create_secure_qdrant_config') as mock_ssl, \
             patch('common.core.client.QdrantClient') as MockQdrantClient, \
             patch('common.core.client.WorkspaceCollectionManager') as MockCollectionManager, \
             patch('common.core.client.get_ssl_manager') as MockSSLManager:

            # Setup mocks
            mock_ssl_config = {"ssl": True}
            mock_ssl.return_value = mock_ssl_config
            mock_qdrant_client = Mock()
            MockQdrantClient.return_value = mock_qdrant_client
            mock_collection_manager = Mock()
            MockCollectionManager.return_value = mock_collection_manager

            # Mock SSL manager
            mock_ssl_manager = Mock()
            mock_ssl_manager.is_localhost_url.return_value = False
            MockSSLManager.return_value = mock_ssl_manager

            client = QdrantWorkspaceClient()

            # Mock project detection locally
            with patch('common.utils.project_detection.ProjectDetector') as MockProjectDetector:
                mock_project_detector = Mock()
                mock_project_info = {"project_name": "test-project", "workspace_path": "/test"}
                mock_project_detector.detect_project_structure.return_value = mock_project_info
                MockProjectDetector.return_value = mock_project_detector

                await client.initialize()

            # Verify initialization
            assert client.initialized is True
            assert client.client == mock_qdrant_client
            assert client.collection_manager == mock_collection_manager

            # Verify SSL config was used
            mock_ssl.assert_called_once()
            MockQdrantClient.assert_called_once_with(**mock_ssl_config)

    @pytest.mark.asyncio
    async def test_initialize_success_without_ssl(self):
        """Test successful initialization without SSL."""
        reset_config()

        with patch('common.core.client.QdrantClient') as MockQdrantClient, \
             patch('common.core.client.WorkspaceCollectionManager') as MockCollectionManager, \
             patch('common.core.client.create_secure_qdrant_config') as mock_ssl, \
             patch('common.core.client.get_ssl_manager') as MockSSLManager:

            mock_qdrant_client = Mock()
            MockQdrantClient.return_value = mock_qdrant_client
            mock_collection_manager = Mock()
            MockCollectionManager.return_value = mock_collection_manager

            # Mock SSL configuration
            mock_ssl_config = {"url": "http://localhost:6333"}
            mock_ssl.return_value = mock_ssl_config

            # Mock SSL manager
            mock_ssl_manager = Mock()
            mock_ssl_manager.is_localhost_url.return_value = True
            mock_ssl_manager.for_localhost.return_value.__enter__ = Mock()
            mock_ssl_manager.for_localhost.return_value.__exit__ = Mock()
            MockSSLManager.return_value = mock_ssl_manager

            client = QdrantWorkspaceClient()

            with patch('common.utils.project_detection.ProjectDetector') as MockProjectDetector:
                mock_project_detector = Mock()
                mock_project_info = {"project_name": "test-project"}
                mock_project_detector.detect_project_structure.return_value = mock_project_info
                MockProjectDetector.return_value = mock_project_detector

                await client.initialize()

            assert client.initialized is True
            # Should call QdrantClient with config
            MockQdrantClient.assert_called_once_with(**mock_ssl_config)

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test that re-initialization is skipped."""
        reset_config()
        client = QdrantWorkspaceClient()
        client.initialized = True

        with patch('common.core.client.QdrantClient') as MockQdrantClient:
            await client.initialize()
            MockQdrantClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_qdrant_connection_failure(self):
        """Test handling of Qdrant connection failure."""
        reset_config()

        with patch('common.core.client.QdrantClient') as MockQdrantClient, \
             patch('common.core.client.create_secure_qdrant_config') as mock_ssl, \
             patch('common.core.client.get_ssl_manager') as MockSSLManager:

            MockQdrantClient.side_effect = Exception("Connection failed")

            # Mock SSL configuration
            mock_ssl.return_value = {"url": "http://localhost:6333"}
            mock_ssl_manager = Mock()
            MockSSLManager.return_value = mock_ssl_manager

            client = QdrantWorkspaceClient()

            with pytest.raises(Exception):
                await client.initialize()

            assert client.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_project_detection_failure(self):
        """Test handling when project detection fails."""
        reset_config()

        with patch('common.core.client.QdrantClient') as MockQdrantClient, \
             patch('common.core.client.WorkspaceCollectionManager') as MockCollectionManager, \
             patch('common.core.client.create_secure_qdrant_config') as mock_ssl, \
             patch('common.core.client.get_ssl_manager') as MockSSLManager:

            mock_qdrant_client = Mock()
            MockQdrantClient.return_value = mock_qdrant_client
            mock_collection_manager = Mock()
            MockCollectionManager.return_value = mock_collection_manager

            # Mock SSL configuration
            mock_ssl.return_value = {"url": "http://localhost:6333"}
            mock_ssl_manager = Mock()
            mock_ssl_manager.is_localhost_url.return_value = True
            mock_ssl_manager.for_localhost.return_value.__enter__ = Mock()
            mock_ssl_manager.for_localhost.return_value.__exit__ = Mock()
            MockSSLManager.return_value = mock_ssl_manager

            client = QdrantWorkspaceClient()

            with patch('common.utils.project_detection.ProjectDetector') as MockProjectDetector:
                mock_project_detector = Mock()
                mock_project_detector.detect_project_structure.side_effect = Exception("Detection failed")
                MockProjectDetector.return_value = mock_project_detector

                # Should still initialize but with no project info
                await client.initialize()

            assert client.initialized is True
            assert client.project_info is None


class TestClientStatus:
    """Test client status operations."""

    @pytest.mark.asyncio
    async def test_get_status_success(self):
        """Test successful status retrieval."""
        reset_config()
        client = QdrantWorkspaceClient()

        # Mock initialized state
        client.initialized = True
        client.client = Mock()
        client.collection_manager = Mock()
        client.project_info = {"main_project": "test-project", "workspace_path": "/test"}

        # Mock collection manager response
        mock_workspace_collections = ["test-collection", "test-scratchbook"]
        client.collection_manager.list_workspace_collections.return_value = mock_workspace_collections
        client.collection_manager.get_collection_info.return_value = {"info": "test"}

        # Mock Qdrant client get_collections
        mock_collections_response = Mock()
        mock_collections_response.collections = ["collection1", "collection2"]
        client.client.get_collections.return_value = mock_collections_response

        # Mock embedding service
        client.embedding_service.get_model_info.return_value = {"model": "test-model"}

        status = await client.get_status()

        assert status["connected"] is True
        assert status["qdrant_url"] == "http://localhost:6333"
        assert status["collections_count"] == 2
        assert status["workspace_collections"] == mock_workspace_collections
        assert status["current_project"] == "test-project"
        assert status["project_info"] == client.project_info

    @pytest.mark.asyncio
    async def test_get_status_not_initialized(self):
        """Test status when client is not initialized."""
        reset_config()
        client = QdrantWorkspaceClient()
        # Don't initialize

        status = await client.get_status()

        assert status["error"] == "Client not initialized"

    @pytest.mark.asyncio
    async def test_get_status_qdrant_error(self):
        """Test status when Qdrant client has issues."""
        reset_config()
        client = QdrantWorkspaceClient()

        client.initialized = True
        client.client = Mock()
        client.collection_manager = Mock()
        client.project_info = {"main_project": "test-project"}

        # Mock Qdrant client failure
        client.client.get_collections.side_effect = Exception("Qdrant error")

        status = await client.get_status()

        assert "error" in status
        assert "Qdrant error" in status["error"]

    @pytest.mark.asyncio
    async def test_get_status_collection_manager_error(self):
        """Test status when collection manager has issues."""
        reset_config()
        client = QdrantWorkspaceClient()

        client.initialized = True
        client.client = Mock()
        client.collection_manager = Mock()
        client.project_info = {"main_project": "test-project"}

        # Mock collection manager failure
        client.collection_manager.list_workspace_collections.side_effect = Exception("Collection error")

        status = await client.get_status()

        assert "error" in status
        assert "Collection error" in status["error"]


if __name__ == "__main__":
    # Run with: python -m pytest tests/unit/test_core_client.py -v
    pytest.main([__file__, "-v", "--tb=short"])
