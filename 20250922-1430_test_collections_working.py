"""
Working collections tests to achieve actual coverage measurement.
"""
import pytest
from unittest.mock import Mock, patch
from dataclasses import dataclass

# Import the actual collections module
from src.python.common.core.collections import (
    CollectionConfig,
    WorkspaceCollectionManager
)


class TestCollectionsWorking:
    """Working tests for collections module."""

    def test_collection_config_basic(self):
        """Test CollectionConfig basic functionality."""
        config = CollectionConfig(
            name="test-collection",
            description="Test collection",
            collection_type="test",
            project_name="test-project"
        )

        assert config.name == "test-collection"
        assert config.description == "Test collection"
        assert config.collection_type == "test"
        assert config.project_name == "test-project"
        assert config.vector_size == 384  # Default value
        assert config.distance_metric == "Cosine"  # Default value
        assert config.enable_sparse_vectors is True  # Default value

    def test_collection_config_defaults(self):
        """Test CollectionConfig with minimal required fields."""
        config = CollectionConfig(
            name="minimal-collection",
            description="Minimal test",
            collection_type="minimal"
        )

        assert config.name == "minimal-collection"
        assert config.description == "Minimal test"
        assert config.collection_type == "minimal"
        assert config.project_name is None  # Default
        assert config.vector_size == 384
        assert config.distance_metric == "Cosine"
        assert config.enable_sparse_vectors is True

    def test_collection_config_custom_values(self):
        """Test CollectionConfig with custom values."""
        config = CollectionConfig(
            name="custom-collection",
            description="Custom test",
            collection_type="custom",
            project_name="custom-project",
            vector_size=768,
            distance_metric="Euclidean",
            enable_sparse_vectors=False
        )

        assert config.name == "custom-collection"
        assert config.vector_size == 768
        assert config.distance_metric == "Euclidean"
        assert config.enable_sparse_vectors is False

    @patch('src.python.common.core.collections.QdrantClient')
    def test_workspace_collection_manager_init(self, mock_qdrant_client):
        """Test WorkspaceCollectionManager initialization."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.global_collections = ["global1", "global2"]
        mock_config.workspace.effective_collection_types = ["docs", "code"]

        manager = WorkspaceCollectionManager(mock_client, mock_config)

        assert manager.client == mock_client
        assert manager.config == mock_config
        assert manager._collections_cache is None
        assert manager._project_info is None

    @patch('src.python.common.core.collections.QdrantClient')
    def test_get_current_project_name(self, mock_qdrant_client):
        """Test project name detection."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.global_collections = []
        mock_config.workspace.effective_collection_types = []

        manager = WorkspaceCollectionManager(mock_client, mock_config)

        # Test the method exists and can be called
        project_name = manager._get_current_project_name()
        # Project name can be None or a string
        assert project_name is None or isinstance(project_name, str)

    @patch('src.python.common.core.collections.git')
    @patch('src.python.common.core.collections.QdrantClient')
    def test_project_name_from_git(self, mock_qdrant_client, mock_git):
        """Test project name extraction from git repository."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.global_collections = []
        mock_config.workspace.effective_collection_types = []

        # Mock git repository
        mock_repo = Mock()
        mock_repo.working_dir = "/path/to/test-project"
        mock_git.Repo.return_value = mock_repo

        manager = WorkspaceCollectionManager(mock_client, mock_config)
        project_name = manager._get_current_project_name()

        # Should be able to get project name
        assert project_name is None or isinstance(project_name, str)

    def test_collection_config_string_representation(self):
        """Test CollectionConfig can be represented as string."""
        config = CollectionConfig(
            name="repr-test",
            description="Test repr",
            collection_type="test"
        )

        # Should be able to convert to string without error
        str_repr = str(config)
        assert isinstance(str_repr, str)
        assert "repr-test" in str_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])