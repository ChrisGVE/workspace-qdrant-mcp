"""
Enhanced tests for QdrantWorkspaceClient to achieve 100% coverage.

This test file focuses on the uncovered areas identified in the coverage analysis,
including exception handling, edge cases, and complex conditional logic.
"""

import asyncio
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock
from typing import Dict, Any, Optional

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from common.core.client import QdrantWorkspaceClient, create_qdrant_client
from common.core.config import Config


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self):
        self.environment = "development"
        self.qdrant = MagicMock()
        self.qdrant.url = "http://localhost:6333"
        self.qdrant_client_config = {"host": "localhost", "port": 6333}
        self.embedding = MagicMock()
        self.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding.enable_sparse_vectors = True
        self.workspace = MagicMock()
        self.workspace.github_user = "testuser"
        self.workspace.global_collections = ["scratchbook"]
        self.security = MagicMock()
        self.security.qdrant_auth_token = None
        self.security.qdrant_api_key = None


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self):
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def close(self):
        self.initialized = False

    def get_model_info(self):
        return {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 384,
            "sparse_enabled": True
        }


class MockQdrantClient:
    """Mock Qdrant client for testing."""

    def __init__(self, **kwargs):
        self._collections = []

    def get_collections(self):
        return MagicMock(collections=[
            MagicMock(name="test-project"),
            MagicMock(name="scratchbook")
        ])

    def close(self):
        pass


class MockCollectionManager:
    """Mock collection manager for testing."""

    def __init__(self, client, config):
        self.client = client
        self.config = config

    async def initialize_workspace_collections(self, project_name, subprojects):
        pass

    def list_workspace_collections(self):
        return ["test-project", "scratchbook"]

    def list_collections_for_project(self, project_name):
        if project_name:
            return [f"{project_name}", f"{project_name}-docs"]
        return []

    def get_collection_info(self):
        return {"total_collections": 2, "workspace_collections": ["test-project"]}

    def _get_vector_size(self):
        return 384

    def _ensure_collection_exists(self, collection_config):
        pass


class MockMemoryCollectionManager:
    """Mock memory collection manager for testing."""

    def __init__(self, client, config):
        self.client = client
        self.config = config

    async def ensure_memory_collections_exist(self, project):
        return {"created": ["test-project-memory"], "existing": []}


class TestQdrantWorkspaceClientEnhanced:
    """Enhanced tests for QdrantWorkspaceClient targeting uncovered areas."""

    @pytest.mark.asyncio
    async def test_initialize_ssl_localhost_handling(self):
        """Test initialize with SSL localhost handling."""
        config = MockConfig()
        config.environment = "development"
        config.qdrant.url = "http://localhost:6333"

        client = QdrantWorkspaceClient(config)

        with patch('qdrant_client.QdrantClient', return_value=MockQdrantClient()), \
             patch('common.core.collections.WorkspaceCollectionManager', MockCollectionManager), \
             patch('common.core.collections.MemoryCollectionManager', MockMemoryCollectionManager), \
             patch.object(client.embedding_service, 'initialize', new_callable=AsyncMock), \
             patch('common.utils.project_detection.ProjectDetector') as mock_detector, \
             patch('common.core.ssl_config.get_ssl_manager') as mock_ssl, \
             patch('common.core.ssl_config.create_secure_qdrant_config') as mock_config_creator, \
             patch('common.core.ssl_config.suppress_qdrant_ssl_warnings') as mock_ssl_suppress, \
             patch('common.core.llm_access_control.validate_llm_collection_access', return_value=None):

            # Setup SSL manager mock
            mock_ssl_manager = MagicMock()
            mock_ssl_manager.is_localhost_url.return_value = True
            mock_ssl_context = MagicMock()
            mock_ssl_context.__enter__ = Mock(return_value=None)
            mock_ssl_context.__exit__ = Mock(return_value=None)
            mock_ssl_manager.for_localhost.return_value = mock_ssl_context
            mock_ssl.return_value = mock_ssl_manager

            mock_config_creator.return_value = {"host": "localhost", "port": 6333}
            mock_ssl_suppress.return_value.__enter__ = Mock()
            mock_ssl_suppress.return_value.__exit__ = Mock(return_value=None)

            # Setup project detector
            mock_detector_instance = MagicMock()
            mock_detector_instance.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": ["frontend"],
                "git_info": {"branch": "main"},
                "directory_structure": {}
            }
            mock_detector.return_value = mock_detector_instance

            await client.initialize()

            assert client.initialized is True
            mock_ssl_manager.is_localhost_url.assert_called_once_with("http://localhost:6333")
            mock_ssl_manager.for_localhost.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_authentication_tokens(self):
        """Test initialize with authentication tokens."""
        config = MockConfig()
        config.security.qdrant_auth_token = "test-auth-token"
        config.security.qdrant_api_key = "test-api-key"

        client = QdrantWorkspaceClient(config)

        with patch('qdrant_client.QdrantClient', return_value=MockQdrantClient()), \
             patch('common.core.collections.WorkspaceCollectionManager', MockCollectionManager), \
             patch('common.core.collections.MemoryCollectionManager', MockMemoryCollectionManager), \
             patch.object(client.embedding_service, 'initialize', new_callable=AsyncMock), \
             patch('common.utils.project_detection.ProjectDetector') as mock_detector, \
             patch('common.core.ssl_config.get_ssl_manager') as mock_ssl, \
             patch('common.core.ssl_config.create_secure_qdrant_config') as mock_config_creator, \
             patch('common.core.ssl_config.suppress_qdrant_ssl_warnings'), \
             patch('common.core.llm_access_control.validate_llm_collection_access', return_value=None):

            mock_ssl.return_value.is_localhost_url.return_value = False
            mock_config_creator.return_value = {"host": "remote", "port": 6333}

            # Setup project detector
            mock_detector_instance = MagicMock()
            mock_detector_instance.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "git_info": {},
                "directory_structure": {}
            }
            mock_detector.return_value = mock_detector_instance

            await client.initialize()

            # Verify auth tokens were passed to config creator
            mock_config_creator.assert_called_once()
            call_args = mock_config_creator.call_args
            assert call_args[1]['auth_token'] == "test-auth-token"
            assert call_args[1]['api_key'] == "test-api-key"

    @pytest.mark.asyncio
    async def test_initialize_production_environment(self):
        """Test initialize in production environment."""
        config = MockConfig()
        config.environment = "production"
        config.qdrant.url = "https://production.qdrant.com"

        client = QdrantWorkspaceClient(config)

        with patch('qdrant_client.QdrantClient', return_value=MockQdrantClient()), \
             patch('common.core.collections.WorkspaceCollectionManager', MockCollectionManager), \
             patch('common.core.collections.MemoryCollectionManager', MockMemoryCollectionManager), \
             patch.object(client.embedding_service, 'initialize', new_callable=AsyncMock), \
             patch('common.utils.project_detection.ProjectDetector') as mock_detector, \
             patch('common.core.ssl_config.get_ssl_manager') as mock_ssl, \
             patch('common.core.ssl_config.create_secure_qdrant_config') as mock_config_creator, \
             patch('common.core.ssl_config.suppress_qdrant_ssl_warnings'), \
             patch('common.core.llm_access_control.validate_llm_collection_access', return_value=None):

            mock_ssl.return_value.is_localhost_url.return_value = False
            mock_config_creator.return_value = {"host": "production.qdrant.com", "port": 443, "https": True}

            # Setup project detector
            mock_detector_instance = MagicMock()
            mock_detector_instance.get_project_info.return_value = {
                "main_project": "prod-project",
                "subprojects": [],
                "git_info": {},
                "directory_structure": {}
            }
            mock_detector.return_value = mock_detector_instance

            await client.initialize()

            assert client.initialized is True
            # Should not use localhost SSL context in production
            mock_ssl.return_value.for_localhost.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_status_with_collection_manager_exception(self):
        """Test get_status when collection manager raises exception."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.client = MockQdrantClient()

        # Mock collection manager that raises exception
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections.side_effect = Exception("Collection error")
        client.collection_manager = mock_collection_manager
        client.embedding_service = MockEmbeddingService()
        client.project_info = {"main_project": "test-project"}

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.run_in_executor = AsyncMock(return_value=MagicMock(collections=[]))
            mock_loop.return_value = mock_executor

            status = await client.get_status()

            assert "error" in status
            assert "Failed to get status" in status["error"]

    @pytest.mark.asyncio
    async def test_get_status_with_async_executor_exception(self):
        """Test get_status when async executor raises exception."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.client = MockQdrantClient()
        client.collection_manager = MockCollectionManager(None, config)
        client.embedding_service = MockEmbeddingService()
        client.project_info = {"main_project": "test-project"}

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.run_in_executor = AsyncMock(side_effect=Exception("Executor error"))
            mock_loop.return_value = mock_executor

            status = await client.get_status()

            assert "error" in status
            assert "Executor error" in status["error"]

    def test_list_collections_with_project_context(self):
        """Test list_collections with project context and enhanced manager."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True

        # Mock collection manager with enhanced functionality
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_collections_for_project.return_value = ["project-main", "project-docs"]
        client.collection_manager = mock_collection_manager
        client.project_info = {"main_project": "project"}

        collections = client.list_collections()

        assert "project-main" in collections
        assert "project-docs" in collections

    def test_list_collections_fallback_to_workspace(self):
        """Test list_collections fallback when enhanced functionality not available."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True

        # Mock collection manager without list_collections_for_project
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections.return_value = ["workspace-1", "workspace-2"]
        # Explicitly remove the enhanced method
        if hasattr(mock_collection_manager, 'list_collections_for_project'):
            delattr(mock_collection_manager, 'list_collections_for_project')
        client.collection_manager = mock_collection_manager
        client.project_info = {"main_project": "project"}

        collections = client.list_collections()

        assert "workspace-1" in collections
        assert "workspace-2" in collections

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_with_import_error_fallback(self):
        """Test ensure_collection_exists with ImportError fallback."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.collection_manager = MockCollectionManager(None, config)
        client.project_info = {"main_project": "test-project"}

        with patch('common.core.collections.CollectionConfig') as mock_config_class, \
             patch('common.core.collection_naming_validation.CollectionNamingValidator', side_effect=ImportError), \
             patch('common.core.llm_access_control.validate_llm_collection_access', return_value=None):

            mock_config_instance = MagicMock()
            mock_config_class.return_value = mock_config_instance

            await client.ensure_collection_exists("test-collection")

            # Should have fallen back to basic validation
            mock_config_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_with_validation_failure(self):
        """Test ensure_collection_exists with validation failure."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.collection_manager = MockCollectionManager(None, config)
        client.project_info = {"main_project": "test-project"}

        with patch('common.core.collections.CollectionConfig') as mock_config_class, \
             patch('common.core.collection_naming_validation.CollectionNamingValidator') as mock_validator_class, \
             patch('common.core.llm_access_control.validate_llm_collection_access', return_value=None):

            # Mock validation to fail
            mock_validator = MagicMock()
            mock_validation_result = MagicMock()
            mock_validation_result.is_valid = False
            mock_validation_result.error_message = "Invalid collection name"
            mock_validator.validate_name.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            with pytest.raises(ValueError, match="Invalid collection name"):
                await client.ensure_collection_exists("invalid-collection")

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_with_manager_exception(self):
        """Test ensure_collection_exists when collection manager raises exception."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True

        # Mock collection manager that raises exception
        mock_collection_manager = MagicMock()
        mock_collection_manager._ensure_collection_exists.side_effect = Exception("Manager error")
        mock_collection_manager._get_vector_size.return_value = 384
        client.collection_manager = mock_collection_manager
        client.project_info = {"main_project": "test-project"}

        with patch('common.core.collections.CollectionConfig') as mock_config_class, \
             patch('common.core.llm_access_control.validate_llm_collection_access', return_value=None):

            mock_config_class.return_value = MagicMock()

            with pytest.raises(RuntimeError, match="Failed to ensure collection"):
                await client.ensure_collection_exists("test-collection")

    @pytest.mark.asyncio
    async def test_search_with_project_context_with_filters(self):
        """Test search_with_project_context with different filter types."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.client = MockQdrantClient()
        client.project_info = {"main_project": "test-project"}

        # Mock hybrid search engine
        mock_search_engine = MagicMock()
        mock_search_results = {
            "fused_results": [{"id": "doc1", "score": 0.9}],
            "search_metadata": {}
        }
        mock_search_engine.hybrid_search = AsyncMock(return_value=mock_search_results)

        with patch('common.core.hybrid_search.HybridSearchEngine', return_value=mock_search_engine):
            # Test with various filter types
            additional_filters = {
                "string_field": "test_value",
                "int_field": 42,
                "float_field": 3.14,
                "list_field": ["value1", "value2"]
            }

            result = await client.search_with_project_context(
                "test-collection",
                {"dense": [0.1] * 384},
                additional_filters=additional_filters,
                include_shared=True
            )

            assert "fused_results" in result
            assert result["include_shared"] is True
            mock_search_engine.hybrid_search.assert_called_once()

            # Check that filters were built correctly
            call_args = mock_search_engine.hybrid_search.call_args
            assert call_args[1]['filter_conditions'] is not None

    @pytest.mark.asyncio
    async def test_search_with_project_context_engine_failure(self):
        """Test search_with_project_context when HybridSearchEngine fails."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.client = MockQdrantClient()

        with patch('common.core.hybrid_search.HybridSearchEngine', side_effect=Exception("Engine error")):
            result = await client.search_with_project_context(
                "test-collection",
                {"dense": [0.1] * 384}
            )

            assert "error" in result
            assert "Engine error" in result["error"]

    def test_get_enhanced_collection_selector_with_project_detector_init(self):
        """Test get_enhanced_collection_selector when project_detector is None."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.client = MockQdrantClient()
        client.project_detector = None

        with patch('common.utils.project_detection.ProjectDetector') as mock_detector, \
             patch('common.core.collections.CollectionSelector') as mock_selector:

            mock_detector_instance = MagicMock()
            mock_detector.return_value = mock_detector_instance
            mock_selector_instance = MagicMock()
            mock_selector.return_value = mock_selector_instance

            selector = client.get_enhanced_collection_selector()

            assert selector is mock_selector_instance
            mock_detector.assert_called_once_with(github_user="testuser")
            assert client.project_detector is mock_detector_instance

    def test_select_collections_by_type_with_exception(self):
        """Test select_collections_by_type when selector raises exception."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True

        with patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Selector error")):
            result = client.select_collections_by_type("memory_collection")

            # Should return empty collections dict on exception
            expected_keys = ['memory_collections', 'code_collections', 'shared_collections',
                            'project_collections', 'fallback_collections']
            for key in expected_keys:
                assert key in result
                assert result[key] == []

    def test_get_searchable_collections_with_exception_fallback(self):
        """Test get_searchable_collections exception fallback to list_collections."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.collection_manager = MockCollectionManager(None, config)

        with patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Selector error")):
            result = client.get_searchable_collections()

            # Should fallback to list_collections
            assert "test-project" in result
            assert "scratchbook" in result

    def test_validate_collection_access_with_exception(self):
        """Test validate_collection_access when selector raises exception."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True

        with patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Validation error")):
            is_allowed, reason = client.validate_collection_access("test", "read")

            assert is_allowed is False
            assert "Validation error" in reason

    @pytest.mark.asyncio
    async def test_create_collection_multitenant_import_error_fallback(self):
        """Test create_collection fallback when multitenant components unavailable."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.collection_manager = MockCollectionManager(None, config)
        client.project_info = {"main_project": "test-project"}

        with patch('common.core.multitenant_collections.MultiTenantWorkspaceCollectionManager', side_effect=ImportError), \
             patch('common.core.collection_naming_validation.CollectionNamingValidator', side_effect=ImportError), \
             patch('common.core.collections.CollectionConfig') as mock_config_class:

            mock_config_class.return_value = MagicMock()

            result = await client.create_collection("test-collection", "docs")

            assert result["success"] is True
            assert result["method"] == "legacy"

    @pytest.mark.asyncio
    async def test_create_collection_validation_failure(self):
        """Test create_collection with validation failure."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True

        with patch('common.core.multitenant_collections.MultiTenantWorkspaceCollectionManager') as mock_mt, \
             patch('common.core.collection_naming_validation.CollectionNamingValidator') as mock_validator_class:

            # Mock validation to fail
            mock_validator = MagicMock()
            mock_validation_result = MagicMock()
            mock_validation_result.is_valid = False
            mock_validation_result.error_message = "Invalid name"
            mock_validation_result.suggested_names = ["suggestion1"]
            mock_validator.validate_name.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            result = await client.create_collection("invalid-name")

            assert result["success"] is False
            assert "Invalid collection name" in result["error"]
            assert "suggestions" in result

    @pytest.mark.asyncio
    async def test_create_collection_no_project_context(self):
        """Test create_collection when no project context is available."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)
        client.initialized = True
        client.project_info = None  # No project info

        with patch('common.core.multitenant_collections.MultiTenantWorkspaceCollectionManager') as mock_mt, \
             patch('common.core.collection_naming_validation.CollectionNamingValidator') as mock_validator_class:

            # Mock validation to pass
            mock_validator = MagicMock()
            mock_validation_result = MagicMock()
            mock_validation_result.is_valid = True
            mock_validator.validate_name.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            result = await client.create_collection("test-collection")

            assert result["success"] is False
            assert "No project context available" in result["error"]

    @pytest.mark.asyncio
    async def test_close_with_services(self):
        """Test close with actual services."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)

        # Set up services to close
        mock_embedding_service = AsyncMock()
        mock_qdrant_client = MagicMock()

        client.embedding_service = mock_embedding_service
        client.client = mock_qdrant_client
        client.initialized = True

        await client.close()

        mock_embedding_service.close.assert_called_once()
        mock_qdrant_client.close.assert_called_once()
        assert client.client is None
        assert client.initialized is False

    def test_refresh_project_detection_with_existing_detector(self):
        """Test refresh_project_detection when project_detector already exists."""
        config = MockConfig()
        client = QdrantWorkspaceClient(config)

        # Set up existing project detector
        mock_existing_detector = MagicMock()
        mock_existing_detector.get_project_info.return_value = {
            "main_project": "existing-project",
            "subprojects": ["sub1"],
            "git_info": {},
            "directory_structure": {}
        }
        client.project_detector = mock_existing_detector

        result = client.refresh_project_detection()

        assert result["main_project"] == "existing-project"
        assert client.project_info == result
        # Should not create new detector
        assert client.project_detector is mock_existing_detector


if __name__ == "__main__":
    pytest.main([__file__, "-v"])