"""
Comprehensive unit tests for QdrantWorkspaceClient to achieve 100% test coverage.

This test suite covers all methods, branches, and edge cases in the client module
including error handling, async operations, and complex initialization flows.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import warnings
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from unittest.mock import call

import pytest
import urllib3
from qdrant_client.http import models

from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient, create_qdrant_client


class TestQdrantWorkspaceClientComprehensive:
    """Comprehensive test suite for QdrantWorkspaceClient achieving 100% coverage."""

    def test_init_comprehensive(self, mock_config):
        """Test comprehensive client initialization with all attributes."""
        client = QdrantWorkspaceClient(mock_config)

        # Verify all attributes are properly initialized
        assert client.config == mock_config
        assert client.client is None
        assert client.collection_manager is None
        assert client.memory_collection_manager is None
        assert client.embedding_service is not None
        assert client.project_detector is None  # Lazy initialized
        assert client.project_info is None
        assert client.initialized is False

    @pytest.mark.asyncio
    async def test_initialize_comprehensive_ssl_config(self, mock_config):
        """Test initialization with comprehensive SSL configuration."""
        client = QdrantWorkspaceClient(mock_config)

        # Mock security configuration
        mock_security = MagicMock()
        mock_security.qdrant_auth_token = "test_token"
        mock_security.qdrant_api_key = "test_api_key"
        mock_config.security = mock_security

        with (
            patch("workspace_qdrant_mcp.core.client.get_ssl_manager") as mock_ssl_manager,
            patch("workspace_qdrant_mcp.core.client.create_secure_qdrant_config") as mock_secure_config,
            patch("workspace_qdrant_mcp.core.client.QdrantClient") as mock_qdrant_class,
            patch("workspace_qdrant_mcp.core.client.WorkspaceCollectionManager") as mock_collection_manager_class,
            patch("workspace_qdrant_mcp.core.client.MemoryCollectionManager") as mock_memory_manager_class,
            patch.object(client.embedding_service, "initialize"),
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks
            mock_ssl_mgr = MagicMock()
            mock_ssl_mgr.is_localhost_url.return_value = True
            mock_ssl_mgr.for_localhost.return_value.__enter__ = MagicMock()
            mock_ssl_mgr.for_localhost.return_value.__exit__ = MagicMock()
            mock_ssl_manager.return_value = mock_ssl_mgr

            mock_secure_config.return_value = {"url": "http://localhost:6333"}

            mock_qdrant_client = MagicMock()
            mock_qdrant_class.return_value = mock_qdrant_client

            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager

            mock_memory_manager = MagicMock()
            mock_memory_manager.ensure_memory_collections_exist = AsyncMock(
                return_value={"created": ["test-collection"]}
            )
            mock_memory_manager_class.return_value = mock_memory_manager

            # Mock project detector
            from workspace_qdrant_mcp.utils.project_detection import ProjectDetector
            with patch("workspace_qdrant_mcp.core.client.ProjectDetector") as mock_project_detector_class:
                mock_project_detector = MagicMock()
                mock_project_detector.get_project_info.return_value = {
                    "main_project": "test-project",
                    "subprojects": ["sub1"]
                }
                mock_project_detector_class.return_value = mock_project_detector

                # Mock asyncio loop
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                future = asyncio.Future()
                future.set_result(None)
                mock_loop.run_in_executor.return_value = future

                await client.initialize()

            # Verify SSL configuration was called with correct parameters
            mock_secure_config.assert_called_once_with(
                base_config=mock_config.qdrant_client_config,
                url=mock_config.qdrant.url,
                environment="development",
                auth_token="test_token",
                api_key="test_api_key",
            )

            # Verify components were initialized
            assert client.initialized is True
            assert client.client == mock_qdrant_client
            assert client.collection_manager == mock_collection_manager
            assert client.memory_collection_manager == mock_memory_manager

    @pytest.mark.asyncio
    async def test_initialize_no_security_config(self, mock_config):
        """Test initialization without security configuration."""
        client = QdrantWorkspaceClient(mock_config)
        # Ensure no security attribute exists
        if hasattr(mock_config, 'security'):
            delattr(mock_config, 'security')

        with (
            patch("workspace_qdrant_mcp.core.client.get_ssl_manager") as mock_ssl_manager,
            patch("workspace_qdrant_mcp.core.client.create_secure_qdrant_config") as mock_secure_config,
            patch("workspace_qdrant_mcp.core.client.QdrantClient") as mock_qdrant_class,
            patch("workspace_qdrant_mcp.core.client.WorkspaceCollectionManager") as mock_collection_manager_class,
            patch("workspace_qdrant_mcp.core.client.MemoryCollectionManager") as mock_memory_manager_class,
            patch.object(client.embedding_service, "initialize"),
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks
            mock_ssl_mgr = MagicMock()
            mock_ssl_mgr.is_localhost_url.return_value = False
            mock_ssl_manager.return_value = mock_ssl_mgr

            mock_secure_config.return_value = {"url": "http://localhost:6333"}

            mock_qdrant_client = MagicMock()
            mock_qdrant_class.return_value = mock_qdrant_client

            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager

            mock_memory_manager = MagicMock()
            mock_memory_manager.ensure_memory_collections_exist = AsyncMock(
                return_value={"created": []}
            )
            mock_memory_manager_class.return_value = mock_memory_manager

            # Mock project detector
            with patch("workspace_qdrant_mcp.core.client.ProjectDetector") as mock_project_detector_class:
                mock_project_detector = MagicMock()
                mock_project_detector.get_project_info.return_value = {
                    "main_project": "test-project",
                    "subprojects": []
                }
                mock_project_detector_class.return_value = mock_project_detector

                # Mock asyncio loop
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                future = asyncio.Future()
                future.set_result(None)
                mock_loop.run_in_executor.return_value = future

                await client.initialize()

            # Verify secure config was called with None auth parameters
            mock_secure_config.assert_called_once_with(
                base_config=mock_config.qdrant_client_config,
                url=mock_config.qdrant.url,
                environment="development",
                auth_token=None,
                api_key=None,
            )

    @pytest.mark.asyncio
    async def test_initialize_no_main_project(self, mock_config):
        """Test initialization when no main project is detected."""
        client = QdrantWorkspaceClient(mock_config)

        with (
            patch("workspace_qdrant_mcp.core.client.get_ssl_manager") as mock_ssl_manager,
            patch("workspace_qdrant_mcp.core.client.create_secure_qdrant_config") as mock_secure_config,
            patch("workspace_qdrant_mcp.core.client.QdrantClient") as mock_qdrant_class,
            patch("workspace_qdrant_mcp.core.client.WorkspaceCollectionManager") as mock_collection_manager_class,
            patch("workspace_qdrant_mcp.core.client.MemoryCollectionManager") as mock_memory_manager_class,
            patch.object(client.embedding_service, "initialize"),
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks
            mock_ssl_mgr = MagicMock()
            mock_ssl_mgr.is_localhost_url.return_value = False
            mock_ssl_manager.return_value = mock_ssl_mgr

            mock_secure_config.return_value = {"url": "http://localhost:6333"}

            mock_qdrant_client = MagicMock()
            mock_qdrant_class.return_value = mock_qdrant_client

            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager

            mock_memory_manager = MagicMock()
            mock_memory_manager.ensure_memory_collections_exist = AsyncMock()
            mock_memory_manager_class.return_value = mock_memory_manager

            # Mock project detector to return no main project
            with patch("workspace_qdrant_mcp.core.client.ProjectDetector") as mock_project_detector_class:
                mock_project_detector = MagicMock()
                mock_project_detector.get_project_info.return_value = {
                    "main_project": None,
                    "subprojects": []
                }
                mock_project_detector_class.return_value = mock_project_detector

                # Mock asyncio loop
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                future = asyncio.Future()
                future.set_result(None)
                mock_loop.run_in_executor.return_value = future

                await client.initialize()

            # Verify memory collections were not initialized due to no project
            mock_memory_manager.ensure_memory_collections_exist.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_with_custom_environment(self, mock_config):
        """Test initialization with custom environment setting."""
        client = QdrantWorkspaceClient(mock_config)
        mock_config.environment = "production"

        with (
            patch("workspace_qdrant_mcp.core.client.get_ssl_manager") as mock_ssl_manager,
            patch("workspace_qdrant_mcp.core.client.create_secure_qdrant_config") as mock_secure_config,
            patch("workspace_qdrant_mcp.core.client.QdrantClient") as mock_qdrant_class,
            patch("workspace_qdrant_mcp.core.client.WorkspaceCollectionManager") as mock_collection_manager_class,
            patch("workspace_qdrant_mcp.core.client.MemoryCollectionManager") as mock_memory_manager_class,
            patch.object(client.embedding_service, "initialize"),
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks
            mock_ssl_mgr = MagicMock()
            mock_ssl_mgr.is_localhost_url.return_value = False
            mock_ssl_manager.return_value = mock_ssl_mgr

            mock_secure_config.return_value = {"url": "https://prod.qdrant.com"}

            mock_qdrant_client = MagicMock()
            mock_qdrant_class.return_value = mock_qdrant_client

            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager

            mock_memory_manager = MagicMock()
            mock_memory_manager.ensure_memory_collections_exist = AsyncMock(
                return_value={"created": []}
            )
            mock_memory_manager_class.return_value = mock_memory_manager

            # Mock project detector
            with patch("workspace_qdrant_mcp.core.client.ProjectDetector") as mock_project_detector_class:
                mock_project_detector = MagicMock()
                mock_project_detector.get_project_info.return_value = {
                    "main_project": "prod-project",
                    "subprojects": []
                }
                mock_project_detector_class.return_value = mock_project_detector

                # Mock asyncio loop
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                future = asyncio.Future()
                future.set_result(None)
                mock_loop.run_in_executor.return_value = future

                await client.initialize()

            # Verify environment was passed correctly
            mock_secure_config.assert_called_once_with(
                base_config=mock_config.qdrant_client_config,
                url=mock_config.qdrant.url,
                environment="production",
                auth_token=None,
                api_key=None,
            )

    @pytest.mark.asyncio
    async def test_initialize_ssl_warnings_suppression(self, mock_config):
        """Test that SSL warnings are properly suppressed during initialization."""
        client = QdrantWorkspaceClient(mock_config)

        with (
            patch("workspace_qdrant_mcp.core.client.get_ssl_manager") as mock_ssl_manager,
            patch("workspace_qdrant_mcp.core.client.create_secure_qdrant_config") as mock_secure_config,
            patch("workspace_qdrant_mcp.core.client.QdrantClient") as mock_qdrant_class,
            patch("workspace_qdrant_mcp.core.client.WorkspaceCollectionManager") as mock_collection_manager_class,
            patch("workspace_qdrant_mcp.core.client.MemoryCollectionManager") as mock_memory_manager_class,
            patch.object(client.embedding_service, "initialize"),
            patch("workspace_qdrant_mcp.core.client.suppress_qdrant_ssl_warnings") as mock_suppress_ssl,
            patch("warnings.catch_warnings") as mock_catch_warnings,
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup mocks
            mock_ssl_mgr = MagicMock()
            mock_ssl_mgr.is_localhost_url.return_value = True
            mock_ssl_mgr.for_localhost.return_value.__enter__ = MagicMock()
            mock_ssl_mgr.for_localhost.return_value.__exit__ = MagicMock()
            mock_ssl_manager.return_value = mock_ssl_mgr

            mock_secure_config.return_value = {"url": "http://localhost:6333"}

            mock_qdrant_client = MagicMock()
            mock_qdrant_class.return_value = mock_qdrant_client

            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager

            mock_memory_manager = MagicMock()
            mock_memory_manager.ensure_memory_collections_exist = AsyncMock(
                return_value={"created": []}
            )
            mock_memory_manager_class.return_value = mock_memory_manager

            mock_suppress_context = MagicMock()
            mock_suppress_context.__enter__ = MagicMock()
            mock_suppress_context.__exit__ = MagicMock()
            mock_suppress_ssl.return_value = mock_suppress_context

            # Mock project detector
            with patch("workspace_qdrant_mcp.core.client.ProjectDetector") as mock_project_detector_class:
                mock_project_detector = MagicMock()
                mock_project_detector.get_project_info.return_value = {
                    "main_project": "test-project",
                    "subprojects": []
                }
                mock_project_detector_class.return_value = mock_project_detector

                # Mock asyncio loop
                mock_loop = MagicMock()
                mock_get_loop.return_value = mock_loop
                future = asyncio.Future()
                future.set_result(None)
                mock_loop.run_in_executor.return_value = future

                await client.initialize()

            # Verify SSL warning suppression was used
            mock_suppress_ssl.assert_called_once()
            mock_catch_warnings.assert_called()

    def test_get_project_context_no_project_info(self, mock_config):
        """Test get_project_context when no project info is available."""
        client = QdrantWorkspaceClient(mock_config)
        client.project_info = None

        result = client.get_project_context()
        assert result is None

    def test_get_project_context_no_main_project(self, mock_config):
        """Test get_project_context when main project is None."""
        client = QdrantWorkspaceClient(mock_config)
        client.project_info = {"main_project": None}

        result = client.get_project_context()
        assert result is None

    def test_get_project_context_empty_main_project(self, mock_config):
        """Test get_project_context when main project is empty string."""
        client = QdrantWorkspaceClient(mock_config)
        client.project_info = {"main_project": ""}

        result = client.get_project_context()
        assert result is None

    def test_get_project_context_with_custom_collection_type(self, mock_config):
        """Test get_project_context with custom collection type."""
        client = QdrantWorkspaceClient(mock_config)
        client.project_info = {"main_project": "test-project"}

        result = client.get_project_context("docs")

        assert result is not None
        assert result["project_name"] == "test-project"
        assert result["collection_type"] == "docs"
        assert result["tenant_namespace"] == "test-project.docs"
        assert result["workspace_scope"] == "project"
        assert "project_id" in result

    def test_generate_project_id(self, mock_config):
        """Test project ID generation."""
        client = QdrantWorkspaceClient(mock_config)

        # Test that same project name generates same ID
        project_id1 = client._generate_project_id("test-project")
        project_id2 = client._generate_project_id("test-project")
        assert project_id1 == project_id2
        assert len(project_id1) == 12

        # Test that different project names generate different IDs
        project_id3 = client._generate_project_id("different-project")
        assert project_id1 != project_id3

    def test_refresh_project_detection_lazy_initialization(self, mock_config):
        """Test refresh_project_detection when project_detector is None."""
        client = QdrantWorkspaceClient(mock_config)
        client.project_detector = None

        with patch("workspace_qdrant_mcp.core.client.ProjectDetector") as mock_project_detector_class:
            mock_project_detector = MagicMock()
            mock_project_detector.get_project_info.return_value = {
                "main_project": "new-project",
                "subprojects": []
            }
            mock_project_detector_class.return_value = mock_project_detector

            result = client.refresh_project_detection()

            # Verify project detector was created and used
            mock_project_detector_class.assert_called_once_with(
                github_user=mock_config.workspace.github_user
            )
            assert result == {"main_project": "new-project", "subprojects": []}
            assert client.project_info == result

    def test_list_collections_with_project_context(self, mock_config):
        """Test list_collections when project context is available."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.project_info = {"main_project": "test-project"}

        # Mock collection manager with project filtering
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_collections_for_project = MagicMock(
            return_value=["test-project-docs", "test-project-scratchbook"]
        )
        client.collection_manager = mock_collection_manager

        with patch.object(client, 'get_project_context', return_value={"project_name": "test-project"}):
            collections = client.list_collections()

            assert collections == ["test-project-docs", "test-project-scratchbook"]
            mock_collection_manager.list_collections_for_project.assert_called_once_with("test-project")

    def test_list_collections_fallback_to_workspace_collections(self, mock_config):
        """Test list_collections fallback when project filtering is not available."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.project_info = {"main_project": "test-project"}

        # Mock collection manager without project filtering method
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = MagicMock(
            return_value=["all-collections"]
        )
        # Remove the list_collections_for_project method
        if hasattr(mock_collection_manager, 'list_collections_for_project'):
            delattr(mock_collection_manager, 'list_collections_for_project')
        client.collection_manager = mock_collection_manager

        with patch.object(client, 'get_project_context', return_value={"project_name": "test-project"}):
            collections = client.list_collections()

            assert collections == ["all-collections"]
            mock_collection_manager.list_workspace_collections.assert_called_once()

    def test_list_collections_no_project_context(self, mock_config):
        """Test list_collections when no project context is available."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = MagicMock(
            return_value=["workspace-collections"]
        )
        client.collection_manager = mock_collection_manager

        with patch.object(client, 'get_project_context', return_value=None):
            collections = client.list_collections()

            assert collections == ["workspace-collections"]
            mock_collection_manager.list_workspace_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_with_validation_components(self, mock_config):
        """Test ensure_collection_exists with validation components available."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager._ensure_collection_exists = MagicMock()
        mock_collection_manager._get_vector_size = MagicMock(return_value=384)
        client.collection_manager = mock_collection_manager

        with (
            patch("workspace_qdrant_mcp.core.client.CollectionNamingValidator") as mock_validator_class,
            patch("workspace_qdrant_mcp.core.client.MultiTenantMetadataSchema"),
            patch.object(client, 'list_collections', return_value=["existing-collection"]),
            patch.object(client, 'get_project_context', return_value={"project_name": "test-project"}),
        ):
            # Mock validator
            mock_validator = MagicMock()
            mock_validation_result = MagicMock()
            mock_validation_result.is_valid = True
            mock_validator.validate_name.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            await client.ensure_collection_exists("new-collection", "docs")

            # Verify validation was called
            mock_validator.validate_name.assert_called_once_with(
                "new-collection",
                existing_collections=["existing-collection"]
            )

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_validation_failure(self, mock_config):
        """Test ensure_collection_exists when validation fails."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        with (
            patch("workspace_qdrant_mcp.core.client.CollectionNamingValidator") as mock_validator_class,
            patch("workspace_qdrant_mcp.core.client.MultiTenantMetadataSchema"),
            patch.object(client, 'list_collections', return_value=[]),
        ):
            # Mock validator with validation failure
            mock_validator = MagicMock()
            mock_validation_result = MagicMock()
            mock_validation_result.is_valid = False
            mock_validation_result.error_message = "Invalid collection name format"
            mock_validator.validate_name.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            with pytest.raises(ValueError, match="Invalid collection name 'invalid-name': Invalid collection name format"):
                await client.ensure_collection_exists("invalid-name")

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_fallback_validation(self, mock_config):
        """Test ensure_collection_exists fallback when validation components are not available."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager._ensure_collection_exists = MagicMock()
        mock_collection_manager._get_vector_size = MagicMock(return_value=384)
        client.collection_manager = mock_collection_manager

        with (
            patch("workspace_qdrant_mcp.core.client.CollectionNamingValidator", side_effect=ImportError),
            patch.object(client, 'get_project_context', return_value={"project_name": "test-project"}),
        ):
            # Should not raise exception and use fallback validation
            await client.ensure_collection_exists("test-collection")

            # Verify collection was still created
            mock_collection_manager._ensure_collection_exists.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_project_context_comprehensive(self, mock_config, mock_qdrant_client):
        """Test search_with_project_context with comprehensive scenarios."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client

        query_embeddings = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.9]}
        }

        with (
            patch("workspace_qdrant_mcp.core.client.HybridSearchEngine") as mock_search_engine_class,
            patch.object(client, 'get_project_context') as mock_get_context,
        ):
            # Mock search engine
            mock_search_engine = MagicMock()
            mock_search_engine.hybrid_search = AsyncMock(return_value={
                "fused_results": [{"id": 1, "score": 0.95}],
                "dense_results": [{"id": 1, "score": 0.9}],
                "sparse_results": [{"id": 1, "score": 0.8}]
            })
            mock_search_engine_class.return_value = mock_search_engine

            # Mock project context
            mock_get_context.return_value = {
                "project_name": "test-project",
                "project_id": "abc123",
                "tenant_namespace": "test-project.docs"
            }

            # Test search with additional filters
            additional_filters = {
                "category": "documentation",
                "priority": 1,
                "tags": ["important", "recent"]
            }

            result = await client.search_with_project_context(
                collection_name="test-docs",
                query_embeddings=query_embeddings,
                collection_type="docs",
                limit=5,
                fusion_method="rrf",
                dense_weight=0.7,
                sparse_weight=0.3,
                additional_filters=additional_filters,
                include_shared=True
            )

            # Verify search was called with correct parameters
            mock_search_engine.hybrid_search.assert_called_once()
            call_args = mock_search_engine.hybrid_search.call_args

            assert call_args[1]["collection_name"] == "test-docs"
            assert call_args[1]["query_embeddings"] == query_embeddings
            assert call_args[1]["limit"] == 5
            assert call_args[1]["fusion_method"] == "rrf"
            assert call_args[1]["dense_weight"] == 0.7
            assert call_args[1]["sparse_weight"] == 0.3
            assert call_args[1]["auto_inject_metadata"] is True

            # Verify result enrichment
            assert "project_context" in result
            assert "collection_type" in result
            assert "include_shared" in result
            assert result["collection_type"] == "docs"
            assert result["include_shared"] is True

    @pytest.mark.asyncio
    async def test_search_with_project_context_not_initialized(self, mock_config):
        """Test search_with_project_context when client is not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        result = await client.search_with_project_context(
            collection_name="test",
            query_embeddings={"dense": [0.1] * 384}
        )

        assert "error" in result
        assert result["error"] == "Workspace client not initialized"

    @pytest.mark.asyncio
    async def test_search_with_project_context_search_failure(self, mock_config, mock_qdrant_client):
        """Test search_with_project_context when search fails."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client

        with (
            patch("workspace_qdrant_mcp.core.client.HybridSearchEngine") as mock_search_engine_class,
            patch.object(client, 'get_project_context', return_value={"project_name": "test"}),
        ):
            # Mock search engine to raise exception
            mock_search_engine = MagicMock()
            mock_search_engine.hybrid_search = AsyncMock(side_effect=Exception("Search failed"))
            mock_search_engine_class.return_value = mock_search_engine

            result = await client.search_with_project_context(
                collection_name="test",
                query_embeddings={"dense": [0.1] * 384}
            )

            assert "error" in result
            assert "Project context search failed" in result["error"]

    def test_get_enhanced_collection_selector_not_initialized(self, mock_config):
        """Test get_enhanced_collection_selector when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        with pytest.raises(RuntimeError, match="Client must be initialized before using collection selector"):
            client.get_enhanced_collection_selector()

    def test_get_enhanced_collection_selector_lazy_project_detector(self, mock_config, mock_qdrant_client):
        """Test get_enhanced_collection_selector with lazy project detector initialization."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client
        client.project_detector = None

        with (
            patch("workspace_qdrant_mcp.core.client.ProjectDetector") as mock_project_detector_class,
            patch("workspace_qdrant_mcp.core.client.CollectionSelector") as mock_selector_class,
        ):
            mock_project_detector = MagicMock()
            mock_project_detector_class.return_value = mock_project_detector

            mock_selector = MagicMock()
            mock_selector_class.return_value = mock_selector

            result = client.get_enhanced_collection_selector()

            # Verify project detector was created
            mock_project_detector_class.assert_called_once_with(
                github_user=mock_config.workspace.github_user
            )
            # Verify selector was created with correct parameters
            mock_selector_class.assert_called_once_with(
                mock_qdrant_client, mock_config, mock_project_detector
            )
            assert result == mock_selector

    def test_select_collections_by_type_not_initialized(self, mock_config):
        """Test select_collections_by_type when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        result = client.select_collections_by_type("memory_collection")

        expected_empty_result = {
            'memory_collections': [],
            'code_collections': [],
            'shared_collections': [],
            'project_collections': [],
            'fallback_collections': []
        }
        assert result == expected_empty_result

    def test_select_collections_by_type_with_exception(self, mock_config, mock_qdrant_client):
        """Test select_collections_by_type when selector raises exception."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client

        with patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Selector error")):
            result = client.select_collections_by_type("memory_collection")

            expected_empty_result = {
                'memory_collections': [],
                'code_collections': [],
                'shared_collections': [],
                'project_collections': [],
                'fallback_collections': []
            }
            assert result == expected_empty_result

    def test_get_searchable_collections_not_initialized(self, mock_config):
        """Test get_searchable_collections when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        result = client.get_searchable_collections()
        assert result == []

    def test_get_searchable_collections_with_exception(self, mock_config, mock_qdrant_client):
        """Test get_searchable_collections when selector raises exception."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client

        with (
            patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Selector error")),
            patch.object(client, 'list_collections', return_value=["fallback-collection"]),
        ):
            result = client.get_searchable_collections()
            assert result == ["fallback-collection"]

    def test_validate_collection_access_not_initialized(self, mock_config):
        """Test validate_collection_access when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        is_allowed, reason = client.validate_collection_access("test", "read")
        assert is_allowed is False
        assert reason == "Client not initialized"

    def test_validate_collection_access_with_exception(self, mock_config, mock_qdrant_client):
        """Test validate_collection_access when selector raises exception."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client

        with patch.object(client, 'get_enhanced_collection_selector', side_effect=Exception("Validation error")):
            is_allowed, reason = client.validate_collection_access("test", "read")

            assert is_allowed is False
            assert "Validation error: Validation error" in reason

    @pytest.mark.asyncio
    async def test_create_collection_not_initialized(self, mock_config):
        """Test create_collection when client not initialized."""
        client = QdrantWorkspaceClient(mock_config)

        result = await client.create_collection("test-collection")
        assert "error" in result
        assert result["error"] == "Client not initialized"

    @pytest.mark.asyncio
    async def test_create_collection_with_multitenant_components(self, mock_config, mock_qdrant_client):
        """Test create_collection with multi-tenant components available."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client
        client.project_info = {"main_project": "test-project"}

        with (
            patch("workspace_qdrant_mcp.core.client.MultiTenantWorkspaceCollectionManager") as mock_mt_manager_class,
            patch("workspace_qdrant_mcp.core.client.CollectionNamingValidator") as mock_validator_class,
            patch.object(client, 'list_collections', return_value=[]),
        ):
            # Mock validator
            mock_validator = MagicMock()
            mock_validation_result = MagicMock()
            mock_validation_result.is_valid = True
            mock_validation_result.proposed_metadata = MagicMock()
            mock_validation_result.proposed_metadata.to_dict.return_value = {"metadata": "test"}
            mock_validator.validate_name.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            # Mock multi-tenant manager
            mock_mt_manager = MagicMock()
            mock_mt_manager.create_workspace_collection = AsyncMock(return_value={
                "success": True,
                "collection_name": "test-project-docs"
            })
            mock_mt_manager_class.return_value = mock_mt_manager

            result = await client.create_collection(
                collection_name="test-project-docs",
                collection_type="docs"
            )

            assert result["success"] is True
            assert "metadata_schema" in result
            mock_mt_manager.create_workspace_collection.assert_called_once_with(
                project_name="test-project",
                collection_type="docs",
                enable_metadata_indexing=True
            )

    @pytest.mark.asyncio
    async def test_create_collection_validation_failure(self, mock_config, mock_qdrant_client):
        """Test create_collection when name validation fails."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client

        with (
            patch("workspace_qdrant_mcp.core.client.CollectionNamingValidator") as mock_validator_class,
            patch.object(client, 'list_collections', return_value=[]),
        ):
            # Mock validator with validation failure
            mock_validator = MagicMock()
            mock_validation_result = MagicMock()
            mock_validation_result.is_valid = False
            mock_validation_result.error_message = "Invalid name format"
            mock_validation_result.suggested_names = ["suggested-name"]
            mock_validator.validate_name.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            result = await client.create_collection("invalid-name")

            assert result["success"] is False
            assert "Invalid collection name" in result["error"]
            assert result["suggestions"] == ["suggested-name"]

    @pytest.mark.asyncio
    async def test_create_collection_no_project_context(self, mock_config, mock_qdrant_client):
        """Test create_collection when no project context is available."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client
        client.project_info = None

        with (
            patch("workspace_qdrant_mcp.core.client.MultiTenantWorkspaceCollectionManager"),
            patch("workspace_qdrant_mcp.core.client.CollectionNamingValidator") as mock_validator_class,
            patch.object(client, 'list_collections', return_value=[]),
            patch.object(client, 'get_project_context', return_value=None),
        ):
            # Mock validator
            mock_validator = MagicMock()
            mock_validation_result = MagicMock()
            mock_validation_result.is_valid = True
            mock_validator.validate_name.return_value = mock_validation_result
            mock_validator_class.return_value = mock_validator

            result = await client.create_collection("test-collection")

            assert result["success"] is False
            assert "No project context available" in result["error"]

    @pytest.mark.asyncio
    async def test_create_collection_legacy_fallback(self, mock_config, mock_qdrant_client):
        """Test create_collection fallback to legacy creation."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client

        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager._ensure_collection_exists = MagicMock()
        mock_collection_manager._get_vector_size = MagicMock(return_value=384)
        client.collection_manager = mock_collection_manager

        with (
            patch("workspace_qdrant_mcp.core.client.MultiTenantWorkspaceCollectionManager", side_effect=ImportError),
            patch.object(client, 'get_project_context', return_value={"project_name": "test-project"}),
        ):
            result = await client.create_collection(
                collection_name="test-collection",
                collection_type="docs",
                project_metadata={"project_name": "test-project"}
            )

            assert result["success"] is True
            assert result["method"] == "legacy"
            mock_collection_manager._ensure_collection_exists.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_exception_handling(self, mock_config, mock_qdrant_client):
        """Test create_collection exception handling."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client

        with patch.object(client, 'get_project_context', side_effect=Exception("Context error")):
            result = await client.create_collection("test-collection")

            assert result["success"] is False
            assert "Collection creation failed" in result["error"]

    @pytest.mark.asyncio
    async def test_close_comprehensive(self, mock_config, mock_qdrant_client):
        """Test comprehensive client cleanup."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.initialized = True

        # Mock embedding service
        mock_embedding_service = MagicMock()
        mock_embedding_service.close = AsyncMock()
        client.embedding_service = mock_embedding_service

        await client.close()

        # Verify cleanup sequence
        mock_embedding_service.close.assert_called_once()
        mock_qdrant_client.close.assert_called_once()
        assert client.client is None
        assert client.initialized is False

    @pytest.mark.asyncio
    async def test_close_no_embedding_service(self, mock_config, mock_qdrant_client):
        """Test close when embedding service is None."""
        client = QdrantWorkspaceClient(mock_config)
        client.client = mock_qdrant_client
        client.initialized = True
        client.embedding_service = None

        # Should not raise exception
        await client.close()

        mock_qdrant_client.close.assert_called_once()
        assert client.client is None
        assert client.initialized is False


class TestCreateQdrantClient:
    """Test the create_qdrant_client factory function."""

    def test_create_qdrant_client_basic(self):
        """Test basic client creation from config data."""
        config_data = {"url": "http://localhost:6333"}

        with patch("workspace_qdrant_mcp.core.client.Config") as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            client = create_qdrant_client(config_data)

            assert isinstance(client, QdrantWorkspaceClient)
            assert client.config == mock_config
            mock_config_class.assert_called_once()

    def test_create_qdrant_client_with_config_integration(self):
        """Test client creation integrates properly with Config class."""
        config_data = {"url": "http://test:6333", "api_key": "test_key"}

        # Use real Config import path but mock the Config class
        with patch("workspace_qdrant_mcp.core.config.Config") as mock_config_class:
            mock_config = MagicMock()
            mock_config_class.return_value = mock_config

            client = create_qdrant_client(config_data)

            # Verify client was created with mocked config
            assert client.config == mock_config
            assert isinstance(client, QdrantWorkspaceClient)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_get_status_with_invalid_collections_response(self, mock_config, mock_qdrant_client):
        """Test get_status when Qdrant returns invalid collections response."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True
        client.client = mock_qdrant_client
        client.project_info = {"main_project": "test-project"}

        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = MagicMock(return_value=[])
        mock_collection_manager.get_collection_info = MagicMock(return_value={})
        client.collection_manager = mock_collection_manager

        # Mock embedding service
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_model_info.return_value = {}
        client.embedding_service = mock_embedding_service

        # Mock async executor to raise exception
        with patch("asyncio.get_event_loop") as mock_get_loop:
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            future = asyncio.Future()
            future.set_exception(Exception("Qdrant connection error"))
            mock_loop.run_in_executor.return_value = future

            status = await client.get_status()

            assert "error" in status
            assert "Failed to get status" in status["error"]

    @pytest.mark.asyncio
    async def test_initialize_project_detector_import_error(self, mock_config):
        """Test initialization when ProjectDetector import fails."""
        client = QdrantWorkspaceClient(mock_config)

        with (
            patch("workspace_qdrant_mcp.core.client.get_ssl_manager") as mock_ssl_manager,
            patch("workspace_qdrant_mcp.core.client.create_secure_qdrant_config") as mock_secure_config,
            patch("workspace_qdrant_mcp.core.client.QdrantClient") as mock_qdrant_class,
            patch("workspace_qdrant_mcp.core.client.WorkspaceCollectionManager") as mock_collection_manager_class,
            patch("workspace_qdrant_mcp.core.client.MemoryCollectionManager") as mock_memory_manager_class,
            patch.object(client.embedding_service, "initialize"),
            patch("workspace_qdrant_mcp.core.client.ProjectDetector", side_effect=ImportError("Cannot import ProjectDetector")),
            patch("asyncio.get_event_loop") as mock_get_loop,
        ):
            # Setup basic mocks
            mock_ssl_mgr = MagicMock()
            mock_ssl_mgr.is_localhost_url.return_value = False
            mock_ssl_manager.return_value = mock_ssl_mgr

            mock_secure_config.return_value = {"url": "http://localhost:6333"}

            mock_qdrant_client = MagicMock()
            mock_qdrant_class.return_value = mock_qdrant_client

            mock_collection_manager = MagicMock()
            mock_collection_manager.initialize_workspace_collections = AsyncMock()
            mock_collection_manager_class.return_value = mock_collection_manager

            mock_memory_manager = MagicMock()
            mock_memory_manager.ensure_memory_collections_exist = AsyncMock()
            mock_memory_manager_class.return_value = mock_memory_manager

            # Mock asyncio loop
            mock_loop = MagicMock()
            mock_get_loop.return_value = mock_loop
            future = asyncio.Future()
            future.set_result(None)
            mock_loop.run_in_executor.return_value = future

            # Should raise the import error
            with pytest.raises(ImportError, match="Cannot import ProjectDetector"):
                await client.initialize()

    def test_list_collections_sync_method(self, mock_config):
        """Test that list_collections is a synchronous method."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager.list_workspace_collections = MagicMock(return_value=["test"])
        client.collection_manager = mock_collection_manager

        # Should be callable without await
        collections = client.list_collections()
        assert collections == ["test"]

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_complex_project_context(self, mock_config):
        """Test ensure_collection_exists with complex project context."""
        client = QdrantWorkspaceClient(mock_config)
        client.initialized = True

        # Mock collection manager
        mock_collection_manager = MagicMock()
        mock_collection_manager._ensure_collection_exists = MagicMock()
        mock_collection_manager._get_vector_size = MagicMock(return_value=768)
        client.collection_manager = mock_collection_manager

        # Mock project context with complex data
        complex_project_context = {
            "project_name": "complex-project",
            "project_id": "complex123",
            "tenant_namespace": "complex-project.multilevel.docs",
            "collection_type": "docs",
            "workspace_scope": "project",
            "additional_metadata": {
                "department": "engineering",
                "priority": "high"
            }
        }

        with patch.object(client, 'get_project_context', return_value=complex_project_context):
            await client.ensure_collection_exists("complex-collection", "multilevel-docs")

            # Verify collection config includes project context
            call_args = mock_collection_manager._ensure_collection_exists.call_args[0][0]
            assert call_args.project_name == "complex-project"
            assert call_args.collection_type == "multilevel-docs"
            assert call_args.vector_size == 768