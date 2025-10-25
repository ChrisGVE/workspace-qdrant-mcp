"""
Comprehensive unit tests for python.common.core.collections module.

Tests cover WorkspaceCollectionManager, MemoryCollectionManager, CollectionSelector,
and all related collection management functionality with 100% coverage.
"""

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from src.python.common.core.collection_naming import CollectionNamingManager
from src.python.common.core.collection_types import CollectionTypeClassifier

# Import modules under test
from src.python.common.core.collections import (
    CollectionConfig,
    CollectionSelector,
    MemoryCollectionManager,
    WorkspaceCollectionManager,
)
from src.python.common.core.config import Config


class TestCollectionConfig:
    """Test CollectionConfig dataclass."""

    def test_collection_config_creation(self):
        """Test creating CollectionConfig with all parameters."""
        config = CollectionConfig(
            name="test-collection",
            description="Test collection",
            collection_type="docs",
            project_name="test-project",
            vector_size=768,
            distance_metric="Euclidean",
            enable_sparse_vectors=False
        )

        assert config.name == "test-collection"
        assert config.description == "Test collection"
        assert config.collection_type == "docs"
        assert config.project_name == "test-project"
        assert config.vector_size == 768
        assert config.distance_metric == "Euclidean"
        assert not config.enable_sparse_vectors

    def test_collection_config_defaults(self):
        """Test CollectionConfig with default values."""
        config = CollectionConfig(
            name="minimal-collection",
            description="Minimal config",
            collection_type="notes"
        )

        assert config.project_name is None
        assert config.vector_size == 384
        assert config.distance_metric == "Cosine"
        assert config.enable_sparse_vectors


class TestWorkspaceCollectionManager:
    """Test WorkspaceCollectionManager functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock QdrantClient."""
        client = Mock(spec=QdrantClient)
        return client

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config."""
        config = Mock(spec=Config)
        config.workspace = Mock()
        config.workspace.global_collections = ["global1", "global2"]
        config.workspace.effective_collection_types = ["docs", "notes"]
        config.workspace.auto_create_collections = True
        config.embedding = Mock()
        config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.enable_sparse_vectors = True
        return config

    @pytest.fixture
    def manager(self, mock_client, mock_config):
        """Create WorkspaceCollectionManager instance."""
        return WorkspaceCollectionManager(mock_client, mock_config)

    def test_initialization(self, manager, mock_client, mock_config):
        """Test manager initialization."""
        assert manager.client == mock_client
        assert manager.config == mock_config
        assert manager._collections_cache is None
        assert manager._project_info is None
        assert isinstance(manager.type_classifier, CollectionTypeClassifier)
        assert isinstance(manager.naming_manager, CollectionNamingManager)

    @patch('git.Repo')
    def test_get_current_project_name_from_git(self, mock_repo, manager):
        """Test project name detection from git."""
        # Setup git repository mock
        repo_instance = Mock()
        mock_repo.return_value = repo_instance

        # Mock remote with URL
        remote = Mock()
        remote.url = "git@github.com:user/test-project.git"
        repo_instance.remotes = [remote]

        project_name = manager._get_current_project_name()
        assert project_name == "test-project"

    @patch('git.Repo')
    @patch('pathlib.Path.cwd')
    def test_get_current_project_name_from_directory(self, mock_cwd, mock_repo, manager):
        """Test project name detection from directory."""
        # Git repository detection fails
        mock_repo.side_effect = Exception("Not a git repo")

        # Mock current directory
        mock_path = Mock()
        mock_path.name = "my-project"
        mock_cwd.return_value = mock_path

        project_name = manager._get_current_project_name()
        assert project_name == "my-project"

    @patch('git.Repo')
    @patch('pathlib.Path.cwd')
    def test_get_current_project_name_skip_common_dirs(self, mock_cwd, mock_repo, manager):
        """Test skipping common subdirectory names."""
        mock_repo.side_effect = Exception("Not a git repo")

        # Mock current directory as common subdir
        mock_path = Mock()
        mock_path.name = "src"
        mock_parent = Mock()
        mock_parent.name = "actual-project"
        mock_path.parent = mock_parent
        mock_cwd.return_value = mock_path

        project_name = manager._get_current_project_name()
        assert project_name == "actual-project"

    def test_get_all_project_names_with_stored_info(self, manager):
        """Test getting project names from stored project info."""
        manager._project_info = {
            "main_project": "main-project",
            "subprojects": ["sub1", "sub2"]
        }

        project_names = manager._get_all_project_names()
        assert "main-project" in project_names
        assert "sub1" in project_names
        assert "sub2" in project_names
        assert len(project_names) == 3

    @patch.object(WorkspaceCollectionManager, '_get_current_project_name')
    def test_get_all_project_names_fallback(self, mock_get_current, manager):
        """Test fallback to current project detection."""
        manager._project_info = None
        mock_get_current.return_value = "fallback-project"

        project_names = manager._get_all_project_names()
        assert project_names == ["fallback-project"]

    def test_validate_collection_filtering(self, manager, mock_client):
        """Test collection filtering validation."""
        # Mock collections response
        collection_mock = Mock()
        collection_mock.name = "test-docs"
        collections_response = Mock()
        collections_response.collections = [collection_mock]
        mock_client.get_collections.return_value = collections_response

        # Setup project info
        manager._project_info = {"main_project": "test", "subprojects": []}

        result = manager.validate_collection_filtering()

        assert "project_info" in result
        assert "project_names" in result
        assert "config_info" in result
        assert "all_collections" in result
        assert "workspace_collections" in result
        assert "filtering_results" in result
        assert "summary" in result

    def test_get_filtering_reason_memexd_exclusion(self, manager):
        """Test filtering reason for memexd daemon collections."""
        reason = manager._get_filtering_reason("project-code")
        assert "Excluded: memexd daemon collection" in reason

    def test_get_filtering_reason_global_inclusion(self, manager):
        """Test filtering reason for global collections."""
        reason = manager._get_filtering_reason("global1")
        assert "Included: global collection" in reason

    def test_get_filtering_reason_suffix_inclusion(self, manager):
        """Test filtering reason for suffix-based inclusion."""
        reason = manager._get_filtering_reason("project-docs")
        assert "Included: ends with configured suffix 'docs'" in reason

    async def test_initialize_workspace_collections_auto_create(self, manager, mock_client):
        """Test workspace collection initialization with auto-create enabled."""
        # Mock existing collections check
        collections_response = Mock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response

        # Test initialization
        await manager.initialize_workspace_collections("test-project", ["sub1"])

        # Verify project info is stored
        assert manager._project_info["main_project"] == "test-project"
        assert manager._project_info["subprojects"] == ["sub1"]

    async def test_initialize_workspace_collections_no_auto_create(self, manager, mock_client, mock_config):
        """Test workspace collection initialization with auto-create disabled."""
        mock_config.workspace.auto_create_collections = False

        await manager.initialize_workspace_collections("test-project")

        # No collections should be created when auto_create_collections=False
        mock_client.create_collection.assert_not_called()

    def test_ensure_collection_exists_already_exists(self, manager, mock_client):
        """Test _ensure_collection_exists when collection already exists."""
        # Mock existing collection
        collection_mock = Mock()
        collection_mock.name = "existing-collection"
        collections_response = Mock()
        collections_response.collections = [collection_mock]
        mock_client.get_collections.return_value = collections_response

        config = CollectionConfig(
            name="existing-collection",
            description="Test",
            collection_type="docs"
        )

        # Should not raise exception and not create collection
        manager._ensure_collection_exists(config)
        mock_client.create_collection.assert_not_called()

    @patch('src.python.common.core.collections.validate_llm_collection_access')
    def test_ensure_collection_exists_llm_access_blocked(self, mock_validate, manager, mock_client):
        """Test _ensure_collection_exists when LLM access is blocked."""
        from src.python.common.core.collections import LLMAccessControlError

        mock_validate.side_effect = LLMAccessControlError("Access blocked")

        # Mock no existing collections
        collections_response = Mock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response

        config = CollectionConfig(
            name="blocked-collection",
            description="Test",
            collection_type="docs"
        )

        with pytest.raises(ValueError, match="Collection creation blocked"):
            manager._ensure_collection_exists(config)

    @patch('src.python.common.core.collections.validate_llm_collection_access')
    def test_ensure_collection_exists_with_sparse_vectors(self, mock_validate, manager, mock_client):
        """Test creating collection with sparse vectors."""
        mock_validate.return_value = None

        # Mock no existing collections
        collections_response = Mock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response

        config = CollectionConfig(
            name="sparse-collection",
            description="Test",
            collection_type="docs",
            enable_sparse_vectors=True
        )

        manager._ensure_collection_exists(config)

        # Verify collection creation with sparse vectors
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert "sparse_vectors_config" in call_args.kwargs

    @patch('src.python.common.core.collections.validate_llm_collection_access')
    def test_ensure_collection_exists_dense_only(self, mock_validate, manager, mock_client):
        """Test creating collection with dense vectors only."""
        mock_validate.return_value = None

        collections_response = Mock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response

        config = CollectionConfig(
            name="dense-collection",
            description="Test",
            collection_type="docs",
            enable_sparse_vectors=False
        )

        manager._ensure_collection_exists(config)

        # Verify collection creation without sparse vectors
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert "sparse_vectors_config" not in call_args.kwargs

    def test_ensure_collection_exists_response_handling_exception(self, manager, mock_client):
        """Test handling ResponseHandlingException during collection creation."""
        collections_response = Mock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response

        mock_client.create_collection.side_effect = ResponseHandlingException("Qdrant error")

        config = CollectionConfig(name="fail-collection", description="Test", collection_type="docs")

        with pytest.raises(ResponseHandlingException):
            manager._ensure_collection_exists(config)

    def test_list_workspace_collections(self, manager, mock_client):
        """Test listing workspace collections."""
        # Mock collections
        collection1 = Mock()
        collection1.name = "project-docs"
        collection2 = Mock()
        collection2.name = "global1"
        collection3 = Mock()
        collection3.name = "external-code"  # Should be excluded

        collections_response = Mock()
        collections_response.collections = [collection1, collection2, collection3]
        mock_client.get_collections.return_value = collections_response

        # Mock type classifier
        manager.type_classifier = Mock()

        def mock_get_collection_info(name):
            from src.python.common.core.collection_types import (
                CollectionInfo,
                CollectionType,
            )
            if name == "external-code":
                return CollectionInfo(collection_type=CollectionType.UNKNOWN)
            return CollectionInfo(collection_type=CollectionType.PROJECT)

        manager.type_classifier.get_collection_info.side_effect = mock_get_collection_info
        manager.type_classifier.get_display_name.side_effect = lambda x: x

        collections = manager.list_workspace_collections()

        assert "project-docs" in collections
        assert "global1" in collections
        assert "external-code" not in collections

    def test_list_workspace_collections_exception(self, manager, mock_client):
        """Test listing workspace collections with exception."""
        mock_client.get_collections.side_effect = Exception("Connection error")

        collections = manager.list_workspace_collections()
        assert collections == []

    def test_get_collection_info(self, manager, mock_client):
        """Test getting collection information."""
        # Mock collections list
        collection_mock = Mock()
        collection_mock.name = "test-collection"
        collections_response = Mock()
        collections_response.collections = [collection_mock]
        mock_client.get_collections.return_value = collections_response

        # Mock workspace collections
        manager.list_workspace_collections = Mock(return_value=["test-collection"])

        # Mock individual collection info
        collection_info = Mock()
        collection_info.vectors_count = 100
        collection_info.points_count = 50
        collection_info.status = "green"
        collection_info.optimizer_status = {"indexing": "complete"}

        # Mock vector config (new API format)
        vector_params = Mock()
        vector_params.distance = "Cosine"
        vector_params.size = 384
        collection_info.config.params.vectors = {"dense": vector_params}

        mock_client.get_collection.return_value = collection_info

        result = manager.get_collection_info()

        assert result["total_collections"] == 1
        assert "test-collection" in result["collections"]
        assert result["collections"]["test-collection"]["vectors_count"] == 100
        assert result["collections"]["test-collection"]["config"]["distance"] == "Cosine"

    def test_get_collection_info_legacy_api(self, manager, mock_client):
        """Test getting collection info with legacy API format."""
        manager.list_workspace_collections = Mock(return_value=["test-collection"])

        collection_info = Mock()
        collection_info.vectors_count = 100
        collection_info.points_count = 50
        collection_info.status = "green"
        collection_info.optimizer_status = {"indexing": "complete"}

        # Mock legacy vector config (direct VectorParams)
        vector_params = Mock()
        vector_params.distance = "Euclidean"
        vector_params.size = 768
        collection_info.config.params.vectors = vector_params

        mock_client.get_collection.return_value = collection_info

        result = manager.get_collection_info()

        assert result["collections"]["test-collection"]["config"]["distance"] == "Euclidean"
        assert result["collections"]["test-collection"]["config"]["vector_size"] == 768

    def test_get_collection_info_individual_error(self, manager, mock_client):
        """Test getting collection info with individual collection error."""
        manager.list_workspace_collections = Mock(return_value=["error-collection"])
        mock_client.get_collection.side_effect = Exception("Collection error")

        result = manager.get_collection_info()

        assert "error-collection" in result["collections"]
        assert "error" in result["collections"]["error-collection"]

    def test_is_workspace_collection(self, manager):
        """Test workspace collection classification."""
        # Mock naming manager
        manager.naming_manager = Mock()

        def mock_get_collection_info(name):
            from src.python.common.core.collection_types import (
                CollectionInfo,
                CollectionType,
            )
            if name == "memory":
                return CollectionInfo(collection_type=CollectionType.MEMORY)
            elif name.startswith("_"):
                return CollectionInfo(collection_type=CollectionType.LIBRARY)
            elif name.startswith("project-"):
                return CollectionInfo(collection_type=CollectionType.PROJECT)
            elif name == "legacy-collection":
                return CollectionInfo(collection_type=CollectionType.LEGACY)
            else:
                return CollectionInfo(collection_type=CollectionType.UNKNOWN)

        manager.naming_manager.get_collection_info.side_effect = mock_get_collection_info

        # Test memory collection
        assert manager._is_workspace_collection("memory")

        # Test library collection
        assert manager._is_workspace_collection("_library")

        # Test project collection
        assert manager._is_workspace_collection("project-docs")

        # Test unknown collection
        assert not manager._is_workspace_collection("unknown")

    def test_is_workspace_collection_legacy_filtering(self, manager):
        """Test legacy workspace collection filtering."""
        manager.naming_manager = Mock()

        def mock_get_collection_info(name):
            from src.python.common.core.collection_types import (
                CollectionInfo,
                CollectionType,
            )
            return CollectionInfo(collection_type=CollectionType.LEGACY)

        manager.naming_manager.get_collection_info.side_effect = mock_get_collection_info

        # Test memexd exclusion
        assert not manager._is_workspace_collection("memexd-code")

        # Test global collection inclusion
        assert manager._is_workspace_collection("global1")

        # Test suffix-based inclusion
        assert manager._is_workspace_collection("project-docs")

    def test_is_workspace_collection_no_config_project_patterns(self, manager, mock_config):
        """Test workspace collection filtering with project patterns."""
        mock_config.workspace.effective_collection_types = []
        mock_config.workspace.global_collections = []

        manager._project_info = {"main_project": "myproject", "subprojects": []}
        manager.naming_manager = Mock()

        def mock_get_collection_info(name):
            from src.python.common.core.collection_types import (
                CollectionInfo,
                CollectionType,
            )
            return CollectionInfo(collection_type=CollectionType.LEGACY)

        manager.naming_manager.get_collection_info.side_effect = mock_get_collection_info

        # Test project pattern matching
        assert manager._is_workspace_collection("myproject-docs")
        assert manager._is_workspace_collection("myproject")
        assert manager._is_workspace_collection("notes")  # Common standalone
        assert not manager._is_workspace_collection("unrelated")

    def test_resolve_collection_name(self, manager, mock_client):
        """Test resolving display names to actual collection names."""
        # Mock collections
        collection1 = Mock()
        collection1.name = "__system_collection"
        collection2 = Mock()
        collection2.name = "_library_collection"
        collection3 = Mock()
        collection3.name = "regular_collection"

        collections_response = Mock()
        collections_response.collections = [collection1, collection2, collection3]
        mock_client.get_collections.return_value = collections_response

        # Mock type classifier
        manager.type_classifier = Mock()

        def mock_get_collection_info(name):
            from src.python.common.core.collection_types import CollectionInfo
            if name.startswith("__"):
                return CollectionInfo(is_readonly=False)
            elif name.startswith("_"):
                return CollectionInfo(is_readonly=True)
            else:
                return CollectionInfo(is_readonly=False)

        manager.type_classifier.get_collection_info.side_effect = mock_get_collection_info

        # Test system collection resolution
        actual, readonly = manager.resolve_collection_name("system_collection")
        assert actual == "__system_collection"
        assert not readonly

        # Test library collection resolution
        actual, readonly = manager.resolve_collection_name("library_collection")
        assert actual == "_library_collection"
        assert readonly

        # Test regular collection resolution
        actual, readonly = manager.resolve_collection_name("regular_collection")
        assert actual == "regular_collection"
        assert not readonly

    def test_resolve_collection_name_not_found(self, manager, mock_client):
        """Test resolving non-existent collection name."""
        collections_response = Mock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response

        manager.type_classifier = Mock()

        actual, readonly = manager.resolve_collection_name("non_existent")
        assert actual == "non_existent"
        assert not readonly

    def test_resolve_collection_name_fallback(self, manager, mock_client):
        """Test fallback behavior when type classifier unavailable."""
        manager.type_classifier = None

        # Mock collections
        collection1 = Mock()
        collection1.name = "_library_collection"
        collections_response = Mock()
        collections_response.collections = [collection1]
        mock_client.get_collections.return_value = collections_response

        # Mock naming manager
        manager.naming_manager = Mock()
        collection_info = Mock()
        collection_info.is_readonly_from_mcp = True
        manager.naming_manager.get_collection_info.return_value = collection_info

        actual, readonly = manager.resolve_collection_name("library_collection")
        assert actual == "_library_collection"
        assert readonly

    def test_validate_mcp_write_access_allowed(self, manager):
        """Test MCP write access validation - allowed case."""
        manager.resolve_collection_name = Mock(return_value=("test-collection", False))

        # Should not raise exception
        manager.validate_mcp_write_access("test-collection")

    def test_validate_mcp_write_access_blocked_library(self, manager):
        """Test MCP write access validation - blocked library collection."""
        from src.python.common.core.collection_types import (
            CollectionInfo,
            CollectionType,
        )
        from src.python.common.core.collections import CollectionPermissionError

        manager.resolve_collection_name = Mock(return_value=("_library", True))

        # Mock naming manager
        manager.naming_manager = Mock()
        collection_info = CollectionInfo(collection_type=CollectionType.LIBRARY)
        manager.naming_manager.get_collection_info.return_value = collection_info

        with pytest.raises(CollectionPermissionError, match="Library collection.*readonly from MCP"):
            manager.validate_mcp_write_access("library")

    def test_validate_mcp_write_access_blocked_other(self, manager):
        """Test MCP write access validation - blocked other collection."""
        from src.python.common.core.collection_types import (
            CollectionInfo,
            CollectionType,
        )
        from src.python.common.core.collections import CollectionPermissionError

        manager.resolve_collection_name = Mock(return_value=("readonly-collection", True))

        # Mock naming manager
        manager.naming_manager = Mock()
        collection_info = CollectionInfo(collection_type=CollectionType.SYSTEM)
        manager.naming_manager.get_collection_info.return_value = collection_info

        with pytest.raises(CollectionPermissionError, match="readonly from MCP server"):
            manager.validate_mcp_write_access("readonly")

    def test_get_naming_manager(self, manager):
        """Test getting naming manager."""
        naming_manager = manager.get_naming_manager()
        assert isinstance(naming_manager, CollectionNamingManager)

    def test_list_searchable_collections(self, manager, mock_client):
        """Test listing searchable collections."""
        # Mock collections
        collection1 = Mock()
        collection1.name = "__system_collection"  # Not searchable
        collection2 = Mock()
        collection2.name = "searchable_collection"  # Searchable

        collections_response = Mock()
        collections_response.collections = [collection1, collection2]
        mock_client.get_collections.return_value = collections_response

        # Mock type classifier
        manager.type_classifier = Mock()

        def mock_get_collection_info(name):
            from src.python.common.core.collection_types import CollectionInfo
            if name.startswith("__"):
                return CollectionInfo(is_searchable=False)
            else:
                return CollectionInfo(is_searchable=True)

        manager.type_classifier.get_collection_info.side_effect = mock_get_collection_info
        manager.type_classifier.get_display_name.side_effect = lambda x: x.lstrip("_")

        searchable = manager.list_searchable_collections()

        assert "system_collection" not in searchable  # System collections excluded
        assert "searchable_collection" in searchable

    def test_list_collections_for_project(self, manager):
        """Test listing collections for specific project."""
        # Mock metadata filtering import
        with patch('src.python.common.core.collections.MetadataFilterManager'):
            with patch('src.python.common.core.collections.FilterCriteria'):
                # Mock collections
                collection1 = Mock()
                collection1.name = "project1-docs"
                collection2 = Mock()
                collection2.name = "project2-docs"

                collections_response = Mock()
                collections_response.collections = [collection1, collection2]
                manager.client.get_collections.return_value = collections_response

                # Mock collection belonging
                manager._collection_belongs_to_project = Mock(side_effect=lambda name, project: name.startswith(project))
                manager.type_classifier = Mock()
                manager.type_classifier.get_display_name.side_effect = lambda x: x

                result = manager.list_collections_for_project("project1")

                assert "project1-docs" in result
                assert "project2-docs" not in result

    def test_list_collections_for_project_no_metadata_filtering(self, manager):
        """Test listing collections for project without metadata filtering."""
        with patch('src.python.common.core.collections.MetadataFilterManager', side_effect=ImportError):
            manager._list_collections_for_project_legacy = Mock(return_value=["legacy-collection"])

            result = manager.list_collections_for_project("test-project")

            assert result == ["legacy-collection"]

    def test_collection_belongs_to_project(self, manager, mock_config):
        """Test checking if collection belongs to project."""
        # Test project prefix pattern
        assert manager._collection_belongs_to_project("myproject-docs", "myproject")

        # Test global collection
        assert manager._collection_belongs_to_project("global1", "myproject")

        # Test system collections
        assert manager._collection_belongs_to_project("__system", "myproject")
        assert manager._collection_belongs_to_project("_library", "myproject")

        # Test exact project match
        assert manager._collection_belongs_to_project("myproject", "myproject")

        # Test unrelated collection
        assert not manager._collection_belongs_to_project("other-docs", "myproject")

    def test_list_collections_for_project_legacy(self, manager, mock_client):
        """Test legacy project filtering method."""
        # Mock collections
        collection1 = Mock()
        collection1.name = "project1-docs"
        collection2 = Mock()
        collection2.name = "unrelated"

        collections_response = Mock()
        collections_response.collections = [collection1, collection2]
        mock_client.get_collections.return_value = collections_response

        # Mock collection belonging
        manager._collection_belongs_to_project = Mock(side_effect=lambda name, project: name.startswith(project))

        result = manager._list_collections_for_project_legacy("project1")

        assert "project1-docs" in result
        assert "unrelated" not in result

    def test_validate_collection_operation(self, manager):
        """Test collection operation validation."""
        manager.resolve_collection_name = Mock(return_value=("test-collection", False))

        # Mock collection type validation
        with patch('src.python.common.core.collections.validate_collection_operation') as mock_validate:
            with patch('src.python.common.core.collections.COLLECTION_TYPES_AVAILABLE', True):
                mock_validate.return_value = (True, "Operation allowed")

                manager.type_classifier = Mock()

                is_valid, reason = manager.validate_collection_operation("test", "read")

                assert is_valid
                assert reason == "Operation allowed"

    def test_validate_collection_operation_fallback(self, manager):
        """Test collection operation validation fallback."""
        manager.resolve_collection_name = Mock(return_value=("test-collection", False))

        # Mock no collection types available
        with patch('src.python.common.core.collections.COLLECTION_TYPES_AVAILABLE', False):
            manager.type_classifier = None

            # Test valid operation
            is_valid, reason = manager.validate_collection_operation("test", "read")
            assert is_valid

            # Test invalid operation
            is_valid, reason = manager.validate_collection_operation("test", "invalid")
            assert not is_valid

            # Test readonly collection
            manager.resolve_collection_name = Mock(return_value=("readonly", True))
            is_valid, reason = manager.validate_collection_operation("readonly", "write")
            assert not is_valid

    def test_get_vector_size(self, manager):
        """Test getting vector size for different models."""
        # Test default model
        size = manager._get_vector_size()
        assert size == 384

        # Test specific model
        manager.config.embedding.model = "BAAI/bge-large-en-v1.5"
        size = manager._get_vector_size()
        assert size == 1024

        # Test unknown model (should default to 384)
        manager.config.embedding.model = "unknown-model"
        size = manager._get_vector_size()
        assert size == 384

    async def test_optimize_metadata_indexing(self, manager, mock_client):
        """Test metadata indexing optimization."""
        collections = [
            CollectionConfig("test1", "Test 1", "docs"),
            CollectionConfig("test2", "Test 2", "library"),  # Should be skipped
        ]

        # Mock executor for async operations
        with patch('asyncio.get_event_loop') as mock_loop:
            Mock()
            mock_loop.return_value.run_in_executor = AsyncMock()

            await manager._optimize_metadata_indexing(collections)

            # Should be called for non-library collections
            assert mock_loop.return_value.run_in_executor.call_count > 0

    async def test_create_collection_with_metadata_support(self, manager):
        """Test creating collection with metadata support."""
        config = CollectionConfig("test", "Test", "docs")
        project_context = {"project": "test-project"}

        # Mock methods
        manager._ensure_collection_exists = Mock()
        manager._optimize_metadata_indexing = AsyncMock()

        await manager._create_collection_with_metadata_support(config, project_context)

        manager._ensure_collection_exists.assert_called_once_with(config)
        manager._optimize_metadata_indexing.assert_called_once_with([config])


class TestMemoryCollectionManager:
    """Test MemoryCollectionManager functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock QdrantClient."""
        return Mock(spec=QdrantClient)

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config."""
        config = Mock()
        config.workspace = Mock()
        config.workspace.global_collections = []
        config.workspace.effective_collection_types = []
        config.embedding = Mock()
        config.embedding.model = "sentence-transformers/all-MiniLM-L6-v2"
        config.embedding.enable_sparse_vectors = True
        return config

    @pytest.fixture
    def memory_manager(self, mock_client, mock_config):
        """Create MemoryCollectionManager instance."""
        return MemoryCollectionManager(mock_client, mock_config)

    def test_memory_manager_initialization(self, memory_manager, mock_client, mock_config):
        """Test MemoryCollectionManager initialization."""
        assert memory_manager.workspace_client == mock_client
        assert memory_manager.config == mock_config
        assert isinstance(memory_manager.naming_manager, CollectionNamingManager)
        assert isinstance(memory_manager.type_classifier, CollectionTypeClassifier)
        assert memory_manager.memory_collection == "memory"

    def test_memory_manager_custom_memory_collection(self, mock_client, mock_config):
        """Test MemoryCollectionManager with custom memory collection name."""
        mock_config.memory_collection = "custom_memory"

        manager = MemoryCollectionManager(mock_client, mock_config)
        assert manager.memory_collection == "custom_memory"

    async def test_ensure_memory_collections_exist_both_missing(self, memory_manager, mock_client):
        """Test ensuring memory collections exist when both are missing."""
        # Mock no existing collections
        collections_response = Mock()
        collections_response.collections = []
        mock_client.get_collections.return_value = collections_response

        # Mock collection creation
        memory_manager.create_system_memory_collection = Mock(return_value={"status": "created"})
        memory_manager.create_project_memory_collection = Mock(return_value={"status": "created"})

        result = await memory_manager.ensure_memory_collections_exist("test-project")

        assert "system_memory" in result
        assert "project_memory" in result
        assert len(result["created"]) == 2
        assert "__memory" in result["created"]
        assert "test-project-memory" in result["created"]

    async def test_ensure_memory_collections_exist_already_exist(self, memory_manager, mock_client):
        """Test ensuring memory collections exist when they already exist."""
        # Mock existing collections
        collection1 = Mock()
        collection1.name = "__memory"
        collection2 = Mock()
        collection2.name = "test-project-memory"

        collections_response = Mock()
        collections_response.collections = [collection1, collection2]
        mock_client.get_collections.return_value = collections_response

        result = await memory_manager.ensure_memory_collections_exist("test-project")

        assert len(result["existing"]) == 2
        assert len(result["created"]) == 0

    def test_collection_exists(self, memory_manager, mock_client):
        """Test checking if collection exists."""
        collection1 = Mock()
        collection1.name = "existing-collection"

        collections_response = Mock()
        collections_response.collections = [collection1]
        mock_client.get_collections.return_value = collections_response

        assert memory_manager.collection_exists("existing-collection")
        assert not memory_manager.collection_exists("non-existent")

    def test_collection_exists_error(self, memory_manager, mock_client):
        """Test collection exists check with error."""
        mock_client.get_collections.side_effect = Exception("Connection error")

        result = memory_manager.collection_exists("any-collection")
        assert not result

    @patch('src.python.common.core.collections.validate_llm_collection_access')
    def test_create_system_memory_collection(self, mock_validate, memory_manager, mock_client):
        """Test creating system memory collection."""
        from src.python.common.core.collections import LLMAccessControlError

        # LLM access should be blocked for system collections
        mock_validate.side_effect = LLMAccessControlError("Access blocked")

        # Mock _create_memory_collection method
        memory_manager._create_memory_collection = Mock()

        result = memory_manager.create_system_memory_collection("__test_memory")

        assert result["collection_name"] == "__test_memory"
        assert result["type"] == "system_memory"
        assert result["access_control"]["cli_writable"]
        assert not result["access_control"]["llm_writable"]
        assert result["status"] == "created"

    def test_create_system_memory_collection_invalid_name(self, memory_manager):
        """Test creating system memory collection with invalid name."""
        with pytest.raises(ValueError, match="System memory collection must start with '__'"):
            memory_manager.create_system_memory_collection("invalid_name")

    def test_create_project_memory_collection(self, memory_manager):
        """Test creating project memory collection."""
        # Mock _create_memory_collection method
        memory_manager._create_memory_collection = Mock()

        result = memory_manager.create_project_memory_collection("test-project-memory")

        assert result["collection_name"] == "test-project-memory"
        assert result["type"] == "project_memory"
        assert result["project_name"] == "test-project"
        assert result["access_control"]["mcp_writable"]
        assert result["status"] == "created"

    def test_create_project_memory_collection_invalid_name(self, memory_manager):
        """Test creating project memory collection with invalid name."""
        with pytest.raises(ValueError, match="Project memory collection must end with '-memory'"):
            memory_manager.create_project_memory_collection("invalid_name")

    def test_create_memory_collection_with_sparse_vectors(self, memory_manager, mock_client):
        """Test creating memory collection with sparse vectors."""
        config = CollectionConfig(
            name="test-memory",
            description="Test",
            collection_type="memory",
            enable_sparse_vectors=True
        )

        memory_manager._create_memory_collection(config)

        # Verify collection creation with sparse vectors
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert "sparse_vectors_config" in call_args.kwargs

    def test_create_memory_collection_dense_only(self, memory_manager, mock_client):
        """Test creating memory collection with dense vectors only."""
        config = CollectionConfig(
            name="test-memory",
            description="Test",
            collection_type="memory",
            enable_sparse_vectors=False
        )

        memory_manager._create_memory_collection(config)

        # Verify collection creation without sparse vectors
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        assert "sparse_vectors_config" not in call_args.kwargs

    def test_get_memory_collections(self, memory_manager, mock_client):
        """Test getting memory collections information."""
        # Mock existing collections
        collection1 = Mock()
        collection1.name = "__memory"
        collection2 = Mock()
        collection2.name = "test-project-memory"

        collections_response = Mock()
        collections_response.collections = [collection1, collection2]
        mock_client.get_collections.return_value = collections_response

        result = memory_manager.get_memory_collections("test-project")

        assert result["system_memory"]["name"] == "__memory"
        assert result["system_memory"]["display_name"] == "memory"
        assert result["system_memory"]["exists"]

        assert result["project_memory"]["name"] == "test-project-memory"
        assert result["project_memory"]["exists"]
        assert result["project_memory"]["project_name"] == "test-project"

    def test_get_memory_collections_error(self, memory_manager, mock_client):
        """Test getting memory collections with error."""
        mock_client.get_collections.side_effect = Exception("Connection error")

        result = memory_manager.get_memory_collections("test-project")

        assert "error" in result["system_memory"]
        assert "error" in result["project_memory"]

    def test_validate_memory_collection_access_delete_blocked(self, memory_manager):
        """Test memory collection access validation - delete blocked."""
        memory_manager.type_classifier = Mock()

        is_allowed, reason = memory_manager.validate_memory_collection_access("test-memory", "delete")

        assert not is_allowed
        assert "cannot be deleted by LLM" in reason

    def test_validate_memory_collection_access_system_write_blocked(self, memory_manager):
        """Test system memory collection write access blocked."""
        from src.python.common.core.collection_types import (
            CollectionInfo,
            CollectionType,
        )

        memory_manager.type_classifier = Mock()
        collection_info = Mock()
        collection_info.type = CollectionType.SYSTEM
        memory_manager.type_classifier.get_collection_info.return_value = collection_info

        is_allowed, reason = memory_manager.validate_memory_collection_access("__system_memory", "write")

        assert not is_allowed
        assert "CLI-writable only" in reason

    def test_validate_memory_collection_access_system_read_allowed(self, memory_manager):
        """Test system memory collection read access allowed."""
        from src.python.common.core.collection_types import (
            CollectionInfo,
            CollectionType,
        )

        memory_manager.type_classifier = Mock()
        collection_info = Mock()
        collection_info.type = CollectionType.SYSTEM
        memory_manager.type_classifier.get_collection_info.return_value = collection_info

        is_allowed, reason = memory_manager.validate_memory_collection_access("__system_memory", "read")

        assert is_allowed
        assert "Read access allowed" in reason

    def test_validate_memory_collection_access_project_allowed(self, memory_manager):
        """Test project memory collection access allowed."""
        from src.python.common.core.collection_types import (
            CollectionInfo,
            CollectionType,
        )

        memory_manager.type_classifier = Mock()
        collection_info = Mock()
        collection_info.type = CollectionType.PROJECT
        memory_manager.type_classifier.get_collection_info.return_value = collection_info

        is_allowed, reason = memory_manager.validate_memory_collection_access("project-memory", "write")

        assert is_allowed
        assert "Write access allowed" in reason

    def test_get_vector_size_memory_manager(self, memory_manager):
        """Test getting vector size in memory manager."""
        # Test default model
        size = memory_manager._get_vector_size()
        assert size == 384

        # Test specific model
        memory_manager.config.embedding.model = "BAAI/bge-m3"
        size = memory_manager._get_vector_size()
        assert size == 1024


class TestCollectionSelector:
    """Test CollectionSelector functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock QdrantClient."""
        return Mock(spec=QdrantClient)

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config."""
        config = Mock()
        config.workspace = Mock()
        config.embedding = Mock()
        return config

    @pytest.fixture
    def mock_project_detector(self):
        """Create a mock project detector."""
        detector = Mock()
        detector.get_project_info.return_value = {"main_project": "test-project"}
        return detector

    @pytest.fixture
    def selector(self, mock_client, mock_config, mock_project_detector):
        """Create CollectionSelector instance."""
        with patch('src.python.common.core.collections.WorkspaceCollectionRegistry'):
            with patch('src.python.common.core.collections.ProjectIsolationManager'):
                return CollectionSelector(mock_client, mock_config, mock_project_detector)

    def test_collection_selector_initialization(self, selector, mock_client, mock_config, mock_project_detector):
        """Test CollectionSelector initialization."""
        assert selector.client == mock_client
        assert selector.config == mock_config
        assert selector.project_detector == mock_project_detector
        assert hasattr(selector, 'workspace_manager')
        assert hasattr(selector, 'memory_manager')
        assert hasattr(selector, 'type_classifier')
        assert hasattr(selector, 'naming_manager')

    def test_select_collections_by_type_memory(self, selector):
        """Test selecting memory collections."""
        # Mock methods
        selector._get_all_collections_with_metadata = Mock(return_value=[
            {"name": "__system_memory", "info": {}},
            {"name": "project-memory", "info": {"project_name": "test-project"}},
            {"name": "memory", "info": {}},
        ])
        selector._select_memory_collections = Mock(return_value=["__system_memory", "memory"])
        selector._is_result_empty = Mock(return_value=False)

        result = selector.select_collections_by_type("memory_collection", "test-project")

        assert "memory_collections" in result
        selector._select_memory_collections.assert_called_once()

    def test_select_collections_by_type_code(self, selector):
        """Test selecting code collections."""
        selector._get_all_collections_with_metadata = Mock(return_value=[
            {"name": "project-docs", "info": {}},
            {"name": "_library", "info": {}},
        ])
        selector._select_code_collections = Mock(return_value=["project-docs", "_library"])
        selector._is_result_empty = Mock(return_value=False)

        result = selector.select_collections_by_type("code_collection", "project")

        assert "code_collections" in result
        selector._select_code_collections.assert_called_once()

    def test_select_collections_by_type_mixed(self, selector):
        """Test selecting mixed collection types."""
        selector._get_all_collections_with_metadata = Mock(return_value=[])
        selector._select_memory_collections = Mock(return_value=["memory"])
        selector._select_code_collections = Mock(return_value=["docs"])
        selector._is_result_empty = Mock(return_value=False)

        result = selector.select_collections_by_type("mixed", "project")

        assert "memory_collections" in result
        assert "code_collections" in result

    def test_select_collections_by_type_with_fallback(self, selector):
        """Test selecting collections with fallback."""
        selector._get_all_collections_with_metadata = Mock(return_value=[])
        selector._select_memory_collections = Mock(return_value=[])
        selector._select_code_collections = Mock(return_value=[])
        selector._select_shared_collections = Mock(return_value=[])
        selector._is_result_empty = Mock(return_value=True)
        selector._apply_fallback_selection = Mock(return_value=["fallback-collection"])

        result = selector.select_collections_by_type("memory_collection", "project")

        assert result["fallback_collections"] == ["fallback-collection"]

    def test_select_collections_by_type_auto_detect_project(self, selector):
        """Test selecting collections with auto-detected project."""
        selector._get_all_collections_with_metadata = Mock(return_value=[])
        selector._select_memory_collections = Mock(return_value=[])
        selector._is_result_empty = Mock(return_value=False)

        # Call without project name - should auto-detect
        selector.select_collections_by_type("memory_collection")

        # Verify project detector was called
        selector.project_detector.get_project_info.assert_called_once()

    def test_select_memory_collections(self, selector):
        """Test selecting memory collections."""
        collections = [
            {"name": "__system_memory", "info": {}},
            {"name": "project1-memory", "info": {"project_name": "project1"}},
            {"name": "memory", "info": {}},
            {"name": "other-docs", "info": {}},
        ]

        selector._is_memory_collection = Mock(side_effect=lambda name, info: "memory" in name)

        result = selector._select_memory_collections(collections, "project1", True)

        assert "__system_memory" in result  # System memory (shared)
        assert "project1-memory" in result  # Project-specific
        assert "memory" in result  # Legacy memory
        assert "other-docs" not in result  # Not memory collection

    def test_select_code_collections(self, selector):
        """Test selecting code collections."""
        collections = [
            {"name": "project1-docs", "info": {}},
            {"name": "_library", "info": {}},
            {"name": "memory", "info": {}},  # Should be skipped
            {"name": "legacy-collection", "info": {}},
        ]

        selector.registry = Mock()
        selector.registry.get_workspace_types.return_value = ["docs", "notes"]
        selector._is_memory_collection = Mock(side_effect=lambda name, info: "memory" in name)
        selector._is_legacy_code_collection = Mock(return_value=True)

        result = selector._select_code_collections(collections, "project1", ["docs"], True)

        assert "project1-docs" in result
        assert "_library" in result
        assert "memory" not in result
        assert "legacy-collection" in result

    def test_select_shared_collections(self, selector):
        """Test selecting shared collections."""
        collections = [
            {"name": "scratchbook", "info": {}},
            {"name": "shared-docs", "info": {"workspace_scope": "shared"}},
            {"name": "private-docs", "info": {}},
        ]

        selector.registry = Mock()
        selector.registry.get_workspace_types.return_value = ["scratchbook", "docs"]

        result = selector._select_shared_collections(collections, ["scratchbook"])

        assert "scratchbook" in result
        assert "shared-docs" in result
        assert "private-docs" not in result

    def test_is_memory_collection(self, selector):
        """Test memory collection identification."""
        # System memory
        assert selector._is_memory_collection("__system_memory", {})

        # Legacy memory
        assert selector._is_memory_collection("memory", {})

        # Metadata-based memory
        assert selector._is_memory_collection("custom", {"collection_type": "memory"})

        # Not memory
        assert not selector._is_memory_collection("docs", {})

    def test_is_legacy_code_collection(self, selector):
        """Test legacy code collection identification."""
        selector.naming_manager = Mock()
        selector.naming_manager.RESERVED_NAMES = ["reserved"]
        selector.registry = Mock()
        selector.registry.get_workspace_types.return_value = ["docs", "notes"]

        # Library collection (excluded)
        assert not selector._is_legacy_code_collection("_library", {})

        # Reserved name (excluded)
        assert not selector._is_legacy_code_collection("reserved", {})

        # Memory collection (excluded)
        assert not selector._is_legacy_code_collection("memory", {})

        # Known workspace type
        assert selector._is_legacy_code_collection("docs", {})

        # Other legacy collection
        assert selector._is_legacy_code_collection("other", {"collection_type": "legacy"})

        # System/memory type (excluded)
        assert not selector._is_legacy_code_collection("other", {"collection_type": "memory"})

    def test_apply_fallback_selection_memory(self, selector):
        """Test fallback selection for memory collections."""
        selector._get_basic_collection_list = Mock(return_value=[
            "__system_memory",
            "user_memory",
            "docs",
        ])
        selector._is_memory_collection = Mock(side_effect=lambda name, info: "memory" in name)

        result = selector._apply_fallback_selection("memory_collection", "project")

        assert "__system_memory" in result
        assert "user_memory" in result
        assert "docs" not in result

    def test_apply_fallback_selection_code(self, selector):
        """Test fallback selection for code collections."""
        selector._get_basic_collection_list = Mock(return_value=[
            "memory",
            "docs",
            "notes",
        ])
        selector._is_memory_collection = Mock(side_effect=lambda name, info: name == "memory")

        result = selector._apply_fallback_selection("code_collection", "project")

        assert "memory" not in result
        assert "docs" in result
        assert "notes" in result

    def test_apply_fallback_selection_mixed(self, selector):
        """Test fallback selection for mixed types."""
        selector._get_basic_collection_list = Mock(return_value=[
            "memory",
            "docs",
            "reserved",
        ])
        selector.naming_manager = Mock()
        selector.naming_manager.RESERVED_NAMES = ["reserved"]

        result = selector._apply_fallback_selection("mixed", "project")

        assert "memory" in result
        assert "docs" in result
        assert "reserved" not in result

    def test_get_all_collections_with_metadata(self, selector):
        """Test getting all collections with metadata."""
        selector._get_basic_collection_list = Mock(return_value=["collection1", "collection2"])

        # Mock collection info
        collection_info = Mock()
        collection_info.config.params = {"metadata": "test"}
        selector.client.get_collection.return_value = collection_info

        result = selector._get_all_collections_with_metadata()

        assert len(result) == 2
        assert result[0]["name"] == "collection1"
        assert result[0]["info"] == {"metadata": "test"}

    def test_get_all_collections_with_metadata_error(self, selector):
        """Test getting collections with metadata when individual collection fails."""
        selector._get_basic_collection_list = Mock(return_value=["error_collection"])
        selector.client.get_collection.side_effect = Exception("Collection error")

        result = selector._get_all_collections_with_metadata()

        assert len(result) == 1
        assert result[0]["name"] == "error_collection"
        assert result[0]["info"] == {}

    def test_get_basic_collection_list(self, selector):
        """Test getting basic collection list."""
        collection1 = Mock()
        collection1.name = "collection1"
        collection2 = Mock()
        collection2.name = "collection2"

        collections_response = Mock()
        collections_response.collections = [collection1, collection2]
        selector.client.get_collections.return_value = collections_response

        result = selector._get_basic_collection_list()

        assert result == ["collection1", "collection2"]

    def test_get_basic_collection_list_error(self, selector):
        """Test getting basic collection list with error."""
        selector.client.get_collections.side_effect = Exception("Connection error")

        result = selector._get_basic_collection_list()
        assert result == []

    def test_is_result_empty(self, selector):
        """Test checking if result is empty."""
        empty_result = {
            "memory_collections": [],
            "code_collections": [],
            "shared_collections": [],
        }
        assert selector._is_result_empty(empty_result)

        non_empty_result = {
            "memory_collections": ["memory"],
            "code_collections": [],
            "shared_collections": [],
        }
        assert not selector._is_result_empty(non_empty_result)

    def test_get_empty_result_with_fallback(self, selector):
        """Test getting empty result with fallback."""
        selector._apply_fallback_selection = Mock(return_value=["fallback"])

        result = selector._get_empty_result_with_fallback("memory_collection")

        assert result["fallback_collections"] == ["fallback"]
        assert result["memory_collections"] == []

    def test_get_searchable_collections(self, selector):
        """Test getting searchable collections."""
        # Mock selection methods
        selector.select_collections_by_type = Mock(side_effect=[
            {"code_collections": ["docs"], "shared_collections": ["shared"]},
            {"memory_collections": ["memory"]},
        ])
        selector._is_memory_collection_searchable = Mock(return_value=True)

        result = selector.get_searchable_collections("project", include_memory=True)

        assert "docs" in result
        assert "shared" in result
        assert "memory" in result

    def test_get_searchable_collections_no_memory(self, selector):
        """Test getting searchable collections without memory."""
        selector.select_collections_by_type = Mock(return_value={
            "code_collections": ["docs"],
            "shared_collections": ["shared"]
        })

        result = selector.get_searchable_collections("project", include_memory=False)

        assert "docs" in result
        assert "shared" in result
        assert len(selector.select_collections_by_type.call_args_list) == 1  # Only called once

    def test_get_searchable_collections_with_fallback(self, selector):
        """Test getting searchable collections with fallback."""
        selector.select_collections_by_type = Mock(side_effect=[
            {"code_collections": [], "shared_collections": []},  # Empty result
            {"fallback_collections": ["fallback"]},  # Fallback result
        ])

        result = selector.get_searchable_collections("project")

        assert "fallback" in result

    def test_is_memory_collection_searchable(self, selector):
        """Test memory collection searchability."""
        selector.registry = Mock()
        selector.registry.is_searchable.return_value = True

        # System collections not searchable
        assert not selector._is_memory_collection_searchable("__system")

        # Legacy memory searchable
        assert selector._is_memory_collection_searchable("memory")

        # Registry-based searchability
        assert selector._is_memory_collection_searchable("custom_memory")

    def test_validate_collection_access_memory(self, selector):
        """Test collection access validation for memory collections."""
        selector._is_memory_collection = Mock(return_value=True)
        selector.memory_manager = Mock()
        selector.memory_manager.validate_memory_collection_access.return_value = (True, "Allowed")

        is_allowed, reason = selector.validate_collection_access("memory", "read")

        assert is_allowed
        assert reason == "Allowed"

    def test_validate_collection_access_code(self, selector):
        """Test collection access validation for code collections."""
        selector._is_memory_collection = Mock(return_value=False)
        selector.workspace_manager = Mock()
        selector.workspace_manager.validate_collection_operation.return_value = (True, "Allowed")

        is_allowed, reason = selector.validate_collection_access("docs", "write")

        assert is_allowed
        assert reason == "Allowed"

    def test_validate_collection_access_error(self, selector):
        """Test collection access validation with error."""
        selector._is_memory_collection = Mock(side_effect=Exception("Validation error"))

        is_allowed, reason = selector.validate_collection_access("collection", "read")

        assert not is_allowed
        assert "Validation error" in reason


if __name__ == "__main__":
    pytest.main([__file__])
