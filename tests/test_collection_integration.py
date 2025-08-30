"""
Integration tests for collection naming system with collection manager.

This module tests the integration between CollectionNamingManager and 
WorkspaceCollectionManager to ensure the reserved naming system works 
correctly with collection management operations.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

from workspace_qdrant_mcp.core.collections import WorkspaceCollectionManager
from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.core.collection_naming import CollectionPermissionError


class TestCollectionManagerIntegration:
    """Test integration between collection naming and collection manager."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock QdrantClient
        self.mock_client = Mock()
        self.mock_client.get_collections.return_value = Mock(collections=[])
        
        # Mock Config
        self.mock_config = Mock(spec=Config)
        self.mock_config.workspace = Mock()
        self.mock_config.workspace.global_collections = ["docs", "references"]
        self.mock_config.workspace.collections = ["project"]
        self.mock_config.embedding = Mock()
        self.mock_config.embedding.enable_sparse_vectors = True
        
        # Create collection manager
        self.manager = WorkspaceCollectionManager(self.mock_client, self.mock_config)

    def test_naming_manager_initialization(self):
        """Test that naming manager is properly initialized."""
        assert self.manager.naming_manager is not None
        assert self.manager.naming_manager.global_collections == {"docs", "references"}

    def test_resolve_collection_name_library(self):
        """Test resolving library collection names."""
        # Mock existing collections including a library collection
        collections = [Mock(name="_mylib"), Mock(name="project-docs")]
        self.mock_client.get_collections.return_value = Mock(collections=collections)
        
        # Test library collection resolution
        actual_name, is_readonly = self.manager.resolve_collection_name("mylib")
        assert actual_name == "_mylib"
        assert is_readonly == True

    def test_resolve_collection_name_regular(self):
        """Test resolving regular collection names."""
        # Mock existing collections  
        collections = [Mock(name="memory"), Mock(name="project-docs")]
        self.mock_client.get_collections.return_value = Mock(collections=collections)
        
        # Test regular collection resolution
        actual_name, is_readonly = self.manager.resolve_collection_name("memory")
        assert actual_name == "memory"
        assert is_readonly == False

    def test_validate_mcp_write_access_allowed(self):
        """Test MCP write access validation for allowed collections."""
        # Mock existing regular collection
        collections = [Mock(name="memory"), Mock(name="project-docs")]
        self.mock_client.get_collections.return_value = Mock(collections=collections)
        
        # Should not raise exception for regular collection
        try:
            self.manager.validate_mcp_write_access("memory")
        except CollectionPermissionError:
            pytest.fail("Should not raise exception for regular collection")

    def test_validate_mcp_write_access_readonly(self):
        """Test MCP write access validation for readonly collections."""
        # Mock existing library collection
        collections = [Mock(name="_mylib")]
        self.mock_client.get_collections.return_value = Mock(collections=collections)
        
        # Should raise exception for library collection
        with pytest.raises(CollectionPermissionError) as exc_info:
            self.manager.validate_mcp_write_access("mylib")
        
        assert "readonly from MCP server" in str(exc_info.value)
        assert "Use the CLI/Rust engine" in str(exc_info.value)

    def test_workspace_collections_display_names(self):
        """Test that workspace collections return display names."""
        # Mock collections with library collection
        collections = [
            Mock(name="memory"),
            Mock(name="_mylib"), 
            Mock(name="project-docs"),
            Mock(name="memexd-project-code")  # Should be excluded
        ]
        self.mock_client.get_collections.return_value = Mock(collections=collections)
        
        workspace_collections = self.manager.list_workspace_collections()
        
        # Should include display names (library without underscore)
        expected = ["memory", "mylib", "project-docs"]  # Note: mylib not _mylib
        assert sorted(workspace_collections) == sorted(expected)
        # Should not include daemon collection
        assert "memexd-project-code" not in workspace_collections

    def test_naming_manager_access(self):
        """Test direct access to naming manager."""
        naming_manager = self.manager.get_naming_manager()
        assert naming_manager is not None
        assert naming_manager == self.manager.naming_manager

    def test_collection_creation_with_validation(self):
        """Test collection creation includes naming validation."""
        # This would be tested with actual async methods in a full integration test
        # For now, just verify the naming manager is properly configured
        
        # Test memory collection validation
        result = self.manager.naming_manager.validate_collection_name("memory")
        assert result.is_valid
        
        # Test invalid collection validation
        result = self.manager.naming_manager.validate_collection_name("_invalid!")
        assert not result.is_valid


class TestCollectionNamingManagerWithRealConfig:
    """Test naming manager with real config objects."""

    def test_with_real_config(self):
        """Test naming manager works with real Config object."""
        # Create real config
        config = Config()
        
        # Mock Qdrant client
        mock_client = Mock()
        mock_client.get_collections.return_value = Mock(collections=[])
        
        # Create collection manager
        manager = WorkspaceCollectionManager(mock_client, config)
        
        # Verify naming manager is created correctly
        assert manager.naming_manager is not None
        assert isinstance(manager.naming_manager.global_collections, set)
        
        # Test basic functionality
        result = manager.naming_manager.validate_collection_name("memory")
        assert result.is_valid
        
        result = manager.naming_manager.validate_collection_name("_mylib")
        assert result.is_valid
        assert result.collection_info.is_readonly_from_mcp == True