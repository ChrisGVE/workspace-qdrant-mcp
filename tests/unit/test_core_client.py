"""
Unit tests for QdrantWorkspaceClient.

Comprehensive testing of the QdrantWorkspaceClient class using lightweight mocking patterns
to test all method-level functionality including initialization, project detection,
collection management, search capabilities, and error scenarios.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock
from typing import Dict, Any, Optional

import pytest
from qdrant_client.http import models

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

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


class MockProjectDetector:
    """Mock project detector for testing."""

    def __init__(self, github_user=None):
        self.github_user = github_user
        self._project_info = {
            "main_project": "test-project",
            "subprojects": ["frontend", "backend"],
            "git_info": {"branch": "main", "remote": "origin"},
            "directory_structure": {"src": True, "tests": True}
        }

    def get_project_info(self):
        return self._project_info


class MockWorkspaceCollectionManager:
    """Mock workspace collection manager for testing."""

    def __init__(self, client, config):
        self.client = client
        self.config = config
        self._collections = ["test-project", "test-project-frontend"]

    async def initialize_workspace_collections(self, project_name, subprojects):
        pass

    def list_workspace_collections(self):
        return self._collections

    def list_collections_for_project(self, project_name):
        if project_name:
            return [col for col in self._collections if col.startswith(project_name)]
        return self._collections


    def get_collection_info(self):
        return {
            "total_collections": len(self._collections),
            "workspace_collections": self._collections
        }

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


@pytest.fixture
def mock_config():
    """Provide mock configuration."""
    return MockConfig()


@pytest.fixture
def mock_embedding_service():
    """Provide mock embedding service."""
    return MockEmbeddingService()


@pytest.fixture
def mock_project_detector():
    """Provide mock project detector."""
    return MockProjectDetector()


@pytest.fixture
def mock_collection_manager():
    """Provide mock collection manager."""
    def factory(client, config):
        return MockWorkspaceCollectionManager(client, config)
    return factory


@pytest.fixture
def mock_memory_collection_manager():
    """Provide mock memory collection manager."""
    def factory(client, config):
        return MockMemoryCollectionManager(client, config)
    return factory


@pytest.fixture
def mock_qdrant_client():
    """Provide mock Qdrant client."""
    return MockQdrantClient()




class TestQdrantWorkspaceClient:
    """Test QdrantWorkspaceClient class."""

    def test_init(self, mock_config):
        """Test client initialization sets correct attributes."""
        client = QdrantWorkspaceClient(mock_config)

        assert client.config is mock_config
        assert client.client is None
        assert client.collection_manager is None
        assert client.memory_collection_manager is None
        assert client.embedding_service is not None
        assert client.project_detector is None
        assert client.project_info is None
        assert client.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_config):
        """Test successful client initialization."""
        client = QdrantWorkspaceClient(mock_config)

        with patch('qdrant_client.QdrantClient', return_value=MockQdrantClient()), \
             patch('common.core.collections.WorkspaceCollectionManager', MockWorkspaceCollectionManager), \
             patch('common.core.collections.MemoryCollectionManager', MockMemoryCollectionManager), \
             patch.object(client.embedding_service, 'initialize', new_callable=AsyncMock), \
             patch('common.utils.project_detection.ProjectDetector', MockProjectDetector), \
             patch('common.core.ssl_config.get_ssl_manager') as mock_ssl, \
             patch('common.core.ssl_config.create_secure_qdrant_config') as mock_config_creator, \
             patch('common.core.ssl_config.suppress_qdrant_ssl_warnings'), \
             patch('common.core.llm_access_control.validate_llm_collection_access') as mock_llm_control:

            mock_ssl.return_value.is_localhost_url.return_value = True
            mock_ssl.return_value.for_localhost.return_value.__enter__ = Mock()
            mock_ssl.return_value.for_localhost.return_value.__exit__ = Mock(return_value=None)
            mock_config_creator.return_value = {"host": "localhost", "port": 6333}
            mock_llm_control.return_value = None  # No access control blocking

            await client.initialize()

            assert client.initialized is True
            assert client.client is not None
            assert client.collection_manager is not None
            assert client.memory_collection_manager is not None
            assert client.project_info is not None
            assert client.project_info["main_project"] == "test-project"

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_config):
        """Test initialization is idempotent."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        await client.initialize()

        # Should return early without doing anything
        assert client.initialized is True
        assert client.client is None

    @pytest.mark.asyncio
    async def test_initialize_connection_error(self, mock_config):
        """Test initialization with connection error."""
        client = QdrantWorkspaceClient(mock_config)

        with patch('qdrant_client.QdrantClient') as mock_qdrant_class, \
             patch('common.core.ssl_config.get_ssl_manager'), \
             patch('common.core.ssl_config.create_secure_qdrant_config') as mock_config_creator, \
             patch('common.core.llm_access_control.validate_llm_collection_access') as mock_llm_control:

            mock_config_creator.return_value = {"host": "localhost", "port": 6333}
            mock_llm_control.return_value = None  # No access control blocking
            mock_qdrant_class.side_effect = ConnectionError("Failed to connect")

            with pytest.raises(ConnectionError):
                await client.initialize()

            assert client.initialized is False

    @pytest.mark.asyncio
    async def test_get_status_not_initialized(self, mock_config):
        """Test get_status when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        status = await client.get_status()

        assert "error" in status
        assert status["error"] == "Client not initialized"

    @pytest.mark.asyncio
    async def test_get_status_success(self, mock_config):
        """Test successful status retrieval."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = MockQdrantClient()
        client.collection_manager = MockWorkspaceCollectionManager(client.client, mock_config)
        client.embedding_service = MockEmbeddingService()
        client.project_info = {"main_project": "test-project"}

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.run_in_executor = AsyncMock(return_value=MagicMock(collections=[]))
            mock_loop.return_value = mock_executor

            status = await client.get_status()

            assert status["connected"] is True
            assert status["qdrant_url"] == "http://localhost:6333"
            assert status["current_project"] == "test-project"
            assert "collection_info" in status
            assert "embedding_info" in status

    @pytest.mark.asyncio
    async def test_get_status_exception(self, mock_config):
        """Test get_status with exception."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = Mock()
        client.client.get_collections.side_effect = Exception("Connection lost")

        with patch('asyncio.get_event_loop') as mock_loop:
            mock_executor = AsyncMock()
            mock_executor.run_in_executor = AsyncMock(side_effect=Exception("Connection lost"))
            mock_loop.return_value = mock_executor

            status = await client.get_status()

            assert "error" in status
            assert "Connection lost" in status["error"]

    def test_list_collections_not_initialized(self, mock_config):
        """Test list_collections when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        collections = client.list_collections()

        assert collections == []

    def test_list_collections_success(self, mock_config):
        """Test successful collection listing."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.collection_manager = MockWorkspaceCollectionManager(None, mock_config)
        client.project_info = {"main_project": "test-project"}

        collections = client.list_collections()

        assert len(collections) == 2
        assert "test-project" in collections
        assert "test-project-frontend" in collections

    def test_list_collections_exception(self, mock_config):
        """Test list_collections with exception."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.collection_manager = Mock()
        client.collection_manager.list_workspace_collections.side_effect = Exception("Error")

        collections = client.list_collections()

        assert collections == []

    def test_get_project_info(self, mock_config):
        """Test getting project information."""
        client = QdrantWorkspaceClient(mock_config)
        project_info = {"main_project": "test", "subprojects": []}
        client.project_info = project_info

        result = client.get_project_info()

        assert result is project_info

    def test_get_project_context_no_project(self, mock_config):
        """Test get_project_context with no project detected."""
        client = QdrantWorkspaceClient(mock_config)
        client.project_info = None

        context = client.get_project_context()

        assert context is None

    def test_get_project_context_success(self, mock_config):
        """Test successful project context generation."""
        client = QdrantWorkspaceClient(mock_config)
        client.project_info = {"main_project": "test-project"}

        context = client.get_project_context("docs")

        assert context["project_name"] == "test-project"
        assert context["collection_type"] == "docs"
        assert context["workspace_scope"] == "project"
        assert "project_id" in context
        assert context["tenant_namespace"] == "test-project.docs"

    def test_generate_project_id(self, mock_config):
        """Test project ID generation."""
        client = QdrantWorkspaceClient(mock_config)

        project_id = client._generate_project_id("test-project")

        assert isinstance(project_id, str)
        assert len(project_id) == 12

        # Same input should generate same ID
        project_id2 = client._generate_project_id("test-project")
        assert project_id == project_id2

    def test_refresh_project_detection(self, mock_config):
        """Test refreshing project detection."""
        client = QdrantWorkspaceClient(mock_config)

        with patch('common.utils.project_detection.ProjectDetector', MockProjectDetector):
            result = client.refresh_project_detection()

            assert result["main_project"] == "test-project"
            assert client.project_info is not None

    def test_get_embedding_service(self, mock_config):
        """Test getting embedding service."""
        client = QdrantWorkspaceClient(mock_config)

        service = client.get_embedding_service()

        assert service is client.embedding_service

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_not_initialized(self, mock_config):
        """Test ensure_collection_exists when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        with pytest.raises(RuntimeError, match="Client not initialized"):
            await client.ensure_collection_exists("test-collection")

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_empty_name(self, mock_config):
        """Test ensure_collection_exists with empty collection name."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        with pytest.raises(ValueError, match="Collection name cannot be empty"):
            await client.ensure_collection_exists("")

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_success(self, mock_config):
        """Test successful collection creation."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.collection_manager = MockWorkspaceCollectionManager(None, mock_config)
        client.project_info = {"main_project": "test-project"}

        # Mock the CollectionConfig class locally in the test
        mock_config_instance = Mock()
        with patch('common.core.collections.CollectionConfig', return_value=mock_config_instance), \
             patch('common.core.llm_access_control.validate_llm_collection_access') as mock_llm_control:
            mock_llm_control.return_value = None  # No access control blocking
            await client.ensure_collection_exists("test-collection", "docs")

    @pytest.mark.asyncio
    async def test_search_with_project_context_not_initialized(self, mock_config):
        """Test search with project context when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        result = await client.search_with_project_context(
            "test-collection",
            {"dense": [0.1] * 384}
        )

        assert "error" in result
        assert result["error"] == "Workspace client not initialized"

    @pytest.mark.asyncio
    async def test_search_with_project_context_success(self, mock_config):
        """Test successful search with project context."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = MockQdrantClient()
        client.project_info = {"main_project": "test-project"}

        mock_search_engine = Mock()
        mock_search_results = {
            "fused_results": [{"id": "doc1", "score": 0.9}],
            "search_metadata": {}
        }
        mock_search_engine.hybrid_search = AsyncMock(return_value=mock_search_results)

        with patch('common.core.hybrid_search.HybridSearchEngine', return_value=mock_search_engine):
            result = await client.search_with_project_context(
                "test-collection",
                {"dense": [0.1] * 384},
                collection_type="docs"
            )

            assert "fused_results" in result
            assert result["project_context"]["project_name"] == "test-project"
            assert result["collection_type"] == "docs"
            mock_search_engine.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_project_context_exception(self, mock_config):
        """Test search with project context exception handling."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = MockQdrantClient()

        with patch('common.core.hybrid_search.HybridSearchEngine') as mock_search_class:
            mock_search_class.side_effect = Exception("Search failed")

            result = await client.search_with_project_context(
                "test-collection",
                {"dense": [0.1] * 384}
            )

            assert "error" in result
            assert "Search failed" in result["error"]

    def test_get_enhanced_collection_selector_not_initialized(self, mock_config):
        """Test enhanced collection selector when not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        with pytest.raises(RuntimeError, match="Client must be initialized"):
            client.get_enhanced_collection_selector()

    def test_get_enhanced_collection_selector_success(self, mock_config):
        """Test getting enhanced collection selector."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = MockQdrantClient()

        mock_selector = Mock()
        with patch('common.utils.project_detection.ProjectDetector', MockProjectDetector), \
             patch('common.core.collections.CollectionSelector', return_value=mock_selector):

            selector = client.get_enhanced_collection_selector()

            assert selector is mock_selector

    def test_select_collections_by_type_not_initialized(self, mock_config):
        """Test select collections by type when not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        result = client.select_collections_by_type("memory_collection")

        expected_keys = ['memory_collections', 'code_collections', 'shared_collections',
                        'project_collections', 'fallback_collections']
        for key in expected_keys:
            assert key in result
            assert result[key] == []

    def test_select_collections_by_type_success(self, mock_config):
        """Test successful collection selection by type."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = MockQdrantClient()

        mock_selector = Mock()
        mock_selector.select_collections_by_type.return_value = {
            'memory_collections': ['test-memory'],
            'code_collections': ['test-code'],
            'shared_collections': ['scratchbook'],
            'project_collections': ['test-project'],
            'fallback_collections': []
        }

        with patch.object(client, 'get_enhanced_collection_selector', return_value=mock_selector):
            result = client.select_collections_by_type("memory_collection")

            assert result['memory_collections'] == ['test-memory']
            assert result['shared_collections'] == ['scratchbook']

    def test_get_searchable_collections_not_initialized(self, mock_config):
        """Test get searchable collections when not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        result = client.get_searchable_collections()

        assert result == []

    def test_get_searchable_collections_success(self, mock_config):
        """Test successful searchable collections retrieval."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = MockQdrantClient()

        mock_selector = Mock()
        mock_selector.get_searchable_collections.return_value = ['test-project', 'scratchbook']

        with patch.object(client, 'get_enhanced_collection_selector', return_value=mock_selector):
            result = client.get_searchable_collections()

            assert result == ['test-project', 'scratchbook']

    def test_validate_collection_access_not_initialized(self, mock_config):
        """Test collection access validation when not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        is_allowed, reason = client.validate_collection_access("test", "read")

        assert is_allowed is False
        assert reason == "Client not initialized"

    def test_validate_collection_access_success(self, mock_config):
        """Test successful collection access validation."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = MockQdrantClient()

        mock_selector = Mock()
        mock_selector.validate_collection_access.return_value = (True, "Access allowed")

        with patch.object(client, 'get_enhanced_collection_selector', return_value=mock_selector):
            is_allowed, reason = client.validate_collection_access("test", "read")

            assert is_allowed is True
            assert reason == "Access allowed"

    @pytest.mark.asyncio
    async def test_create_collection_not_initialized(self, mock_config):
        """Test create collection when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        result = await client.create_collection("test-collection")

        assert "error" in result
        assert result["error"] == "Client not initialized"

    @pytest.mark.asyncio
    async def test_create_collection_legacy_mode(self, mock_config):
        """Test collection creation in legacy mode."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.collection_manager = MockWorkspaceCollectionManager(None, mock_config)
        client.project_info = {"main_project": "test-project"}

        mock_config_instance = Mock()
        with patch('common.core.collections.CollectionConfig', return_value=mock_config_instance), \
             patch('common.core.multitenant_collections.MultiTenantWorkspaceCollectionManager', side_effect=ImportError), \
             patch('common.core.collection_naming_validation.CollectionNamingValidator', side_effect=ImportError):
            result = await client.create_collection("test-collection", "docs")

            assert result["success"] is True
            assert result["method"] == "legacy"

    @pytest.mark.asyncio
    async def test_close(self, mock_config):
        """Test client cleanup."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service = MockEmbeddingService()
        client.client = MockQdrantClient()
        client.initialized = True

        await client.close()

        assert client.embedding_service.initialized is False
        assert client.client is None
        assert client.initialized is False

    @pytest.mark.asyncio
    async def test_close_with_none_services(self, mock_config):
        """Test client cleanup with None services."""
        client = QdrantWorkspaceClient(mock_config)
        client.embedding_service = None
        client.client = None

        # Should not raise exception
        await client.close()

        assert client.initialized is False


class TestCreateQdrantClient:
    """Test create_qdrant_client factory function."""

    def test_create_qdrant_client(self):
        """Test factory function creates client with config."""
        config_data = {"host": "localhost", "port": 6333}

        client = create_qdrant_client(config_data)

        assert isinstance(client, QdrantWorkspaceClient)
        assert isinstance(client.config, Config)


# Edge cases and error scenarios
class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_exception(self, mock_config):
        """Test ensure_collection_exists with internal exception."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.collection_manager = Mock()
        client.collection_manager._ensure_collection_exists.side_effect = Exception("Internal error")

        mock_config_instance = Mock()
        with patch('common.core.collections.CollectionConfig', return_value=mock_config_instance), \
             patch('common.core.llm_access_control.validate_llm_collection_access') as mock_llm_control:
            mock_llm_control.return_value = None  # No access control blocking
            with pytest.raises(RuntimeError, match="Failed to ensure collection"):
                await client.ensure_collection_exists("test-collection")

    def test_get_project_context_empty_project(self, mock_config):
        """Test get_project_context with empty project name."""
        client = QdrantWorkspaceClient(mock_config)
        client.project_info = {"main_project": ""}

        context = client.get_project_context()

        assert context is None

    def test_select_collections_by_type_exception(self, mock_config):
        """Test select_collections_by_type with exception."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        with patch.object(client, 'get_enhanced_collection_selector') as mock_selector:
            mock_selector.side_effect = Exception("Selector error")

            result = client.select_collections_by_type("memory_collection")

            # Should return empty collections dict
            expected_keys = ['memory_collections', 'code_collections', 'shared_collections',
                            'project_collections', 'fallback_collections']
            for key in expected_keys:
                assert key in result
                assert result[key] == []

    def test_get_searchable_collections_exception_fallback(self, mock_config):
        """Test get_searchable_collections exception fallback."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.collection_manager = MockWorkspaceCollectionManager(None, mock_config)

        with patch.object(client, 'get_enhanced_collection_selector') as mock_selector:
            mock_selector.side_effect = Exception("Selector error")

            result = client.get_searchable_collections()

            # Should fallback to list_collections
            assert len(result) == 2
            assert "test-project" in result

    def test_validate_collection_access_exception(self, mock_config):
        """Test validate_collection_access with exception."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        with patch.object(client, 'get_enhanced_collection_selector') as mock_selector:
            mock_selector.side_effect = Exception("Validation error")

            is_allowed, reason = client.validate_collection_access("test", "read")

            assert is_allowed is False
            assert "Validation error" in reason