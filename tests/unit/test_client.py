"""
Unit tests for QdrantWorkspaceClient.

Tests client initialization, project detection, and workspace operations.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models

from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.config import Config


class TestQdrantWorkspaceClient:
    """Test QdrantWorkspaceClient class."""
    
    def test_init(self, mock_config):
        """Test client initialization."""
        client = QdrantWorkspaceClient(mock_config)
        
        assert client.config == mock_config
        assert client.client is None
        assert client.collection_manager is None
        assert client.embedding_service is not None
        assert client.project_detector is not None
        assert client.project_detector.github_user == "testuser"
        assert client.project_info is None
        assert client.initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config, mock_qdrant_client):
        """Test successful client initialization."""
        client = QdrantWorkspaceClient(mock_config)
        
        # Mock dependencies
        with patch('workspace_qdrant_mcp.core.client.QdrantClient') as mock_qdrant_class, \
             patch('workspace_qdrant_mcp.core.client.WorkspaceCollectionManager') as mock_collection_manager_class, \
             patch.object(client.embedding_service, 'initialize') as mock_embed_init, \
             patch.object(client.project_detector, 'get_project_info') as mock_project_info:
            
            mock_qdrant_class.return_value = mock_qdrant_client
            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager
            
            mock_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": ["subproject1"]
            }
            
            # Mock asyncio.get_event_loop().run_in_executor
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                
                # Create a future and set its result
                import asyncio
                future = asyncio.Future()
                future.set_result(None)
                mock_loop.run_in_executor.return_value = future
                
                await client.initialize()
            
            assert client.initialized is True
            assert client.client == mock_qdrant_client
            assert client.collection_manager == mock_collection_manager
            assert client.project_info["main_project"] == "test-project"
            
            # Verify initialization sequence
            mock_qdrant_class.assert_called_once_with(**mock_config.qdrant_client_config)
            mock_embed_init.assert_called_once()
            mock_collection_manager.initialize_workspace_collections.assert_called_once_with(
                project_name="test-project",
                subprojects=["subproject1"]
            )
    
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_config):
        """Test initialization when already initialized."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        
        # Should return early without doing anything
        await client.initialize()
        
        assert client.client is None  # Should not be set
    
    @pytest.mark.asyncio
    async def test_initialize_qdrant_connection_failure(self, mock_config):
        """Test initialization with Qdrant connection failure."""
        client = QdrantWorkspaceClient(mock_config)
        
        with patch('workspace_qdrant_mcp.core.client.QdrantClient') as mock_qdrant_class:
            mock_qdrant_class.side_effect = Exception("Connection failed")
            
            with pytest.raises(Exception, match="Connection failed"):
                await client.initialize()
            
            assert client.initialized is False
    
    @pytest.mark.asyncio
    async def test_get_status_not_initialized(self, mock_config):
        """Test get_status when client is not initialized."""
        client = QdrantWorkspaceClient(mock_config)
        
        status = await client.get_status()
        
        assert "error" in status
        assert status["error"] == "Client not initialized"
    
    @pytest.mark.asyncio
    async def test_get_status_success(self, mock_config, mock_qdrant_client):
        """Test successful status retrieval."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client
        client.project_info = {
            "main_project": "test-project",
            "subprojects": []
        }
        
        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = AsyncMock(return_value=["collection1", "collection2"])
        mock_collection_manager.get_collection_info = AsyncMock(return_value={"info": "test"})
        client.collection_manager = mock_collection_manager
        
        # Mock embedding service
        client.embedding_service.get_model_info.return_value = {"model": "test-model"}
        
        # Mock asyncio executor for get_collections
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            
            collections_response = models.CollectionsResponse(collections=[
                models.CollectionDescription(
                    name="test_collection",
                    status=models.CollectionStatus.GREEN,
                    vectors_count=100,
                    indexed_vectors_count=100,
                    points_count=50,
                    segments_count=1,
                    config=models.CollectionConfig(
                        params=models.CollectionParams(
                            vectors=models.VectorParams(
                                size=384,
                                distance=models.Distance.COSINE
                            )
                        )
                    )
                )
            ])
            
            future = asyncio.create_future()
            future.set_result(collections_response)
            mock_loop.run_in_executor.return_value = future
            
            status = await client.get_status()
        
        assert "error" not in status
        assert status["connected"] is True
        assert status["qdrant_url"] == mock_config.qdrant.url
        assert status["collections_count"] == 1
        assert status["workspace_collections"] == ["collection1", "collection2"]
        assert status["current_project"] == "test-project"
        assert "project_info" in status
        assert "collection_info" in status
        assert "embedding_info" in status
        assert "config" in status
    
    @pytest.mark.asyncio
    async def test_get_status_qdrant_error(self, mock_config, mock_qdrant_client):
        """Test get_status when Qdrant operations fail."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client
        
        # Mock collection manager to raise exception
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = AsyncMock(side_effect=Exception("Qdrant error"))
        client.collection_manager = mock_collection_manager
        
        status = await client.get_status()
        
        assert "error" in status
        assert "Failed to get status" in status["error"]
    
    @pytest.mark.asyncio
    async def test_list_collections_not_initialized(self, mock_config):
        """Test list_collections when client is not initialized."""
        client = QdrantWorkspaceClient(mock_config)
        
        collections = await client.list_collections()
        
        assert collections == []
    
    @pytest.mark.asyncio
    async def test_list_collections_success(self, mock_config):
        """Test successful collections listing."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        
        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = AsyncMock(
            return_value=["collection1", "collection2", "collection3"]
        )
        client.collection_manager = mock_collection_manager
        
        collections = await client.list_collections()
        
        assert collections == ["collection1", "collection2", "collection3"]
        mock_collection_manager.list_workspace_collections.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_collections_error(self, mock_config):
        """Test list_collections when operation fails."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        
        # Mock collection manager to raise exception
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = AsyncMock(
            side_effect=Exception("Collection error")
        )
        client.collection_manager = mock_collection_manager
        
        collections = await client.list_collections()
        
        assert collections == []
    
    def test_get_project_info(self, mock_config):
        """Test getting project information."""
        client = QdrantWorkspaceClient(mock_config)
        
        # Initially None
        assert client.get_project_info() is None
        
        # Set project info
        test_project_info = {"main_project": "test", "subprojects": []}
        client.project_info = test_project_info
        
        assert client.get_project_info() == test_project_info
    
    def test_refresh_project_detection(self, mock_config):
        """Test refreshing project detection."""
        client = QdrantWorkspaceClient(mock_config)
        
        new_project_info = {
            "main_project": "new-project",
            "subprojects": ["new-sub"]
        }
        
        with patch.object(client.project_detector, 'get_project_info', return_value=new_project_info):
            result = client.refresh_project_detection()
            
            assert result == new_project_info
            assert client.project_info == new_project_info
    
    def test_get_embedding_service(self, mock_config):
        """Test getting embedding service."""
        client = QdrantWorkspaceClient(mock_config)
        
        embedding_service = client.get_embedding_service()
        
        assert embedding_service == client.embedding_service
    
    @pytest.mark.asyncio
    async def test_close(self, mock_config, mock_qdrant_client):
        """Test client cleanup."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.initialized = True
        
        # Mock embedding service close
        client.embedding_service.close = AsyncMock()
        
        await client.close()
        
        # Verify cleanup
        client.embedding_service.close.assert_called_once()
        mock_qdrant_client.close.assert_called_once()
        assert client.client is None
        assert client.initialized is False
    
    @pytest.mark.asyncio
    async def test_close_no_client(self, mock_config):
        """Test close when no client is set."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service.close = AsyncMock()
        
        # Should not raise exception
        await client.close()
        
        client.embedding_service.close.assert_called_once()
    
    def test_project_detector_github_user_configuration(self, mock_config):
        """Test that project detector is configured with GitHub user."""
        # Test with GitHub user
        mock_config.workspace.github_user = "testuser"
        client = QdrantWorkspaceClient(mock_config)
        
        assert client.project_detector.github_user == "testuser"
        
        # Test without GitHub user
        mock_config.workspace.github_user = None
        client2 = QdrantWorkspaceClient(mock_config)
        
        assert client2.project_detector.github_user is None
    
    @pytest.mark.asyncio
    async def test_initialize_collection_manager_configuration(self, mock_config, mock_qdrant_client):
        """Test that collection manager is properly configured during initialization."""
        client = QdrantWorkspaceClient(mock_config)
        
        with patch('workspace_qdrant_mcp.core.client.QdrantClient', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.core.client.WorkspaceCollectionManager') as mock_collection_manager_class, \
             patch.object(client.embedding_service, 'initialize'), \
             patch.object(client.project_detector, 'get_project_info', return_value={"main_project": "test", "subprojects": []}), \
             patch('asyncio.get_event_loop') as mock_get_loop:
            
            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager
            
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            future = asyncio.create_future()
            future.set_result(None)
            mock_loop.run_in_executor.return_value = future
            
            await client.initialize()
            
            # Verify collection manager was created with correct parameters
            mock_collection_manager_class.assert_called_once_with(mock_qdrant_client, mock_config)
    
    def test_embedding_service_configuration(self, mock_config):
        """Test that embedding service is configured correctly."""
        client = QdrantWorkspaceClient(mock_config)
        
        # Embedding service should be created with the config
        assert client.embedding_service is not None
        # The actual EmbeddingService would be configured with mock_config
        # but since we're testing the client, we just verify it's created
    
    @pytest.mark.asyncio
    async def test_initialize_project_detection_and_collection_setup(self, mock_config, mock_qdrant_client):
        """Test the complete initialization flow with project detection."""
        client = QdrantWorkspaceClient(mock_config)
        
        expected_project_info = {
            "main_project": "workspace-qdrant-mcp",
            "subprojects": ["submodule1", "submodule2"],
            "git_root": "/path/to/project",
            "remote_url": "https://github.com/testuser/workspace-qdrant-mcp.git",
            "is_git_repo": True,
            "belongs_to_user": True
        }
        
        with patch('workspace_qdrant_mcp.core.client.QdrantClient', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.core.client.WorkspaceCollectionManager') as mock_collection_manager_class, \
             patch.object(client.embedding_service, 'initialize'), \
             patch.object(client.project_detector, 'get_project_info', return_value=expected_project_info), \
             patch('asyncio.get_event_loop') as mock_get_loop:
            
            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager
            
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            future = asyncio.create_future()
            future.set_result(None)
            mock_loop.run_in_executor.return_value = future
            
            await client.initialize()
            
            # Verify project info was detected and stored
            assert client.project_info == expected_project_info
            
            # Verify collection manager was called with detected project info
            mock_collection_manager.initialize_workspace_collections.assert_called_once_with(
                project_name="workspace-qdrant-mcp",
                subprojects=["submodule1", "submodule2"]
            )
    
    @pytest.mark.asyncio
    async def test_get_status_comprehensive_response(self, mock_config, mock_qdrant_client):
        """Test that get_status returns comprehensive status information."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client
        client.project_info = {
            "main_project": "test-project",
            "subprojects": ["sub1", "sub2"],
            "git_root": "/path/to/project"
        }
        
        # Mock all dependencies
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = AsyncMock(
            return_value=["test_docs", "test_scratchbook"]
        )
        mock_collection_manager.get_collection_info = AsyncMock(
            return_value={"total_points": 150, "collections": 2}
        )
        client.collection_manager = mock_collection_manager
        
        client.embedding_service.get_model_info.return_value = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_size": 384
        }
        
        # Mock Qdrant collections response
        collections_response = models.CollectionsResponse(
            collections=[
                models.CollectionDescription(
                    name="test_collection",
                    status=models.CollectionStatus.GREEN,
                    vectors_count=100,
                    indexed_vectors_count=100,
                    points_count=50,
                    segments_count=1,
                    config=models.CollectionConfig(
                        params=models.CollectionParams(
                            vectors=models.VectorParams(
                                size=384,
                                distance=models.Distance.COSINE
                            )
                        )
                    )
                )
            ]
        )
        
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            future = asyncio.create_future()
            future.set_result(collections_response)
            mock_loop.run_in_executor.return_value = future
            
            status = await client.get_status()
        
        # Verify comprehensive status information
        expected_fields = [
            "connected", "qdrant_url", "collections_count", "workspace_collections",
            "current_project", "project_info", "collection_info", "embedding_info", "config"
        ]
        
        for field in expected_fields:
            assert field in status, f"Missing field: {field}"
        
        # Verify config information
        config_info = status["config"]
        assert "embedding_model" in config_info
        assert "sparse_vectors_enabled" in config_info
        assert "global_collections" in config_info
        
        assert config_info["embedding_model"] == mock_config.embedding.model
        assert config_info["sparse_vectors_enabled"] == mock_config.embedding.enable_sparse_vectors
        assert config_info["global_collections"] == mock_config.workspace.global_collections
