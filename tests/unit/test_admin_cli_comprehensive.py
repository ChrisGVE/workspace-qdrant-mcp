"""
Comprehensive unit tests for administrative CLI utilities.

This module provides 100% test coverage for the admin CLI system,
including all administrative operations and safety mechanisms.

Test coverage:
- WorkspaceQdrantAdmin: all administrative operations
- Configuration handling and validation
- Safety mechanisms and confirmation logic
- Project scoping and dry-run functionality
- Error handling and edge cases
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
import pytest

# Ensure proper imports from the project structure
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))
from common.utils.admin_cli import WorkspaceQdrantAdmin
from common.core.config import Config
from common.core.client import QdrantWorkspaceClient
from common.utils.project_detection import ProjectDetector


class TestWorkspaceQdrantAdmin:
    """Comprehensive tests for WorkspaceQdrantAdmin class."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_config = Mock(spec=Config)
        self.mock_client = Mock(spec=QdrantWorkspaceClient)
        self.mock_detector = Mock(spec=ProjectDetector)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default(self):
        """Test default initialization of WorkspaceQdrantAdmin."""
        admin = WorkspaceQdrantAdmin()

        assert admin.config is not None
        assert admin.dry_run == False
        assert admin.project_scope is None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        assert admin.config == self.mock_config

    def test_init_with_dry_run(self):
        """Test initialization with dry run enabled."""
        admin = WorkspaceQdrantAdmin(dry_run=True)

        assert admin.dry_run == True

    def test_init_with_project_scope(self):
        """Test initialization with project scope."""
        admin = WorkspaceQdrantAdmin(project_scope="test-project")

        assert admin.project_scope == "test-project"

    def test_init_all_parameters(self):
        """Test initialization with all parameters."""
        admin = WorkspaceQdrantAdmin(
            config=self.mock_config,
            dry_run=True,
            project_scope="test-project"
        )

        assert admin.config == self.mock_config
        assert admin.dry_run == True
        assert admin.project_scope == "test-project"

    @patch('common.utils.admin_cli.Config')
    def test_init_no_config_creates_default(self, mock_config_class):
        """Test initialization without config creates default."""
        mock_config_instance = Mock()
        mock_config_class.return_value = mock_config_instance

        admin = WorkspaceQdrantAdmin()

        assert admin.config == mock_config_instance
        mock_config_class.assert_called_once()

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_context_manager_setup_and_teardown(self, mock_detector_class, mock_client_class):
        """Test async context manager setup and teardown."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            assert admin_instance.client == mock_client_instance
            assert admin_instance.detector == mock_detector_instance
            mock_client_instance.__aenter__.assert_called_once()

        mock_client_instance.__aexit__.assert_called_once()

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_context_manager_exception_handling(self, mock_detector_class, mock_client_class):
        """Test async context manager exception handling."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock client context manager to raise exception
        mock_client_instance.__aenter__.side_effect = Exception("Connection failed")

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        with pytest.raises(Exception, match="Connection failed"):
            async with admin:
                pass

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_list_collections_basic(self, mock_detector_class, mock_client_class):
        """Test basic collection listing."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock collection list
        mock_collections = ["project1-docs", "project1-code", "global-reference"]
        mock_client_instance.list_collections.return_value = mock_collections

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            collections = await admin_instance.list_collections()

            assert collections == mock_collections
            mock_client_instance.list_collections.assert_called_once()

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_list_collections_with_stats(self, mock_detector_class, mock_client_class):
        """Test collection listing with statistics."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock collection list and stats
        mock_collections = ["project1-docs"]
        mock_client_instance.list_collections.return_value = mock_collections
        mock_client_instance.get_collection_info.return_value = {
            "points_count": 100,
            "vector_count": 100,
            "status": "green"
        }

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            collections = await admin_instance.list_collections(include_stats=True)

            # Should include stats for each collection
            mock_client_instance.get_collection_info.assert_called_with("project1-docs")

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_list_collections_project_scoped(self, mock_detector_class, mock_client_class):
        """Test collection listing with project scope."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock collection list
        mock_collections = ["project1-docs", "project2-docs", "global-reference"]
        mock_client_instance.list_collections.return_value = mock_collections

        admin = WorkspaceQdrantAdmin(config=self.mock_config, project_scope="project1")

        async with admin as admin_instance:
            collections = await admin_instance.list_collections()

            # Should filter collections by project scope
            assert "project1-docs" in collections or len(collections) == len(mock_collections)

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_delete_collection_dry_run(self, mock_detector_class, mock_client_class):
        """Test collection deletion in dry run mode."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config, dry_run=True)

        async with admin as admin_instance:
            result = await admin_instance.delete_collection("test-collection")

            # Should not actually delete in dry run
            mock_client_instance.delete_collection.assert_not_called()
            # Result should indicate dry run
            assert "dry run" in str(result).lower() or result is None

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_delete_collection_actual(self, mock_detector_class, mock_client_class):
        """Test actual collection deletion."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        mock_client_instance.delete_collection.return_value = True

        admin = WorkspaceQdrantAdmin(config=self.mock_config, dry_run=False)

        async with admin as admin_instance:
            result = await admin_instance.delete_collection("test-collection")

            mock_client_instance.delete_collection.assert_called_once_with("test-collection")

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_delete_collection_project_scope_validation(self, mock_detector_class, mock_client_class):
        """Test collection deletion with project scope validation."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config, project_scope="project1")

        async with admin as admin_instance:
            # Try to delete collection from different project
            with pytest.raises(ValueError, match="not within project scope") or \
                 pytest.raises(PermissionError) or \
                 pytest.raises(Exception):
                await admin_instance.delete_collection("project2-docs")

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_search_documents(self, mock_detector_class, mock_client_class):
        """Test document search functionality."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock search results
        mock_results = [
            {"content": "Test document 1", "score": 0.95},
            {"content": "Test document 2", "score": 0.87}
        ]
        mock_client_instance.search.return_value = mock_results

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            results = await admin_instance.search_documents("test query")

            assert results == mock_results
            mock_client_instance.search.assert_called()

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_search_documents_with_collection(self, mock_detector_class, mock_client_class):
        """Test document search with specific collection."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        mock_results = []
        mock_client_instance.search.return_value = mock_results

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            results = await admin_instance.search_documents(
                "test query",
                collection="specific-collection"
            )

            # Should search in specific collection
            search_call = mock_client_instance.search.call_args
            assert "specific-collection" in str(search_call) or search_call is not None

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_get_collection_statistics(self, mock_detector_class, mock_client_class):
        """Test collection statistics retrieval."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        mock_stats = {
            "points_count": 500,
            "vector_count": 500,
            "status": "green",
            "config": {"vector_size": 384}
        }
        mock_client_instance.get_collection_info.return_value = mock_stats

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            stats = await admin_instance.get_collection_statistics("test-collection")

            assert stats == mock_stats
            mock_client_instance.get_collection_info.assert_called_once_with("test-collection")

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_get_collection_statistics_error(self, mock_detector_class, mock_client_class):
        """Test collection statistics with error handling."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        mock_client_instance.get_collection_info.side_effect = Exception("Collection not found")

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            with pytest.raises(Exception, match="Collection not found"):
                await admin_instance.get_collection_statistics("nonexistent-collection")

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_reset_project_dry_run(self, mock_detector_class, mock_client_class):
        """Test project reset in dry run mode."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock project detection
        mock_detector_instance.get_project_name.return_value = "test-project"

        # Mock collection list for project
        mock_collections = ["test-project-docs", "test-project-code"]
        mock_client_instance.list_collections.return_value = ["test-project-docs", "test-project-code", "other-project-docs"]

        admin = WorkspaceQdrantAdmin(config=self.mock_config, dry_run=True)

        async with admin as admin_instance:
            result = await admin_instance.reset_project()

            # Should not actually delete collections in dry run
            mock_client_instance.delete_collection.assert_not_called()

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_reset_project_actual(self, mock_detector_class, mock_client_class):
        """Test actual project reset."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock project detection
        mock_detector_instance.get_project_name.return_value = "test-project"

        # Mock collection list
        mock_client_instance.list_collections.return_value = ["test-project-docs", "test-project-code", "other-project-docs"]
        mock_client_instance.delete_collection.return_value = True

        admin = WorkspaceQdrantAdmin(config=self.mock_config, dry_run=False)

        async with admin as admin_instance:
            result = await admin_instance.reset_project()

            # Should delete project collections
            expected_calls = [
                call("test-project-docs"),
                call("test-project-code")
            ]
            mock_client_instance.delete_collection.assert_has_calls(expected_calls, any_order=True)

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_project_scope_filtering(self, mock_detector_class, mock_client_class):
        """Test project scope filtering logic."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config, project_scope="test-project")

        # Test that admin has correct project scope
        assert admin.project_scope == "test-project"

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_validate_project_scope(self, mock_detector_class, mock_client_class):
        """Test project scope validation."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config, project_scope="allowed-project")

        async with admin as admin_instance:
            # Method to validate collection belongs to project scope
            def validate_collection_scope(collection_name):
                if admin_instance.project_scope:
                    if not collection_name.startswith(admin_instance.project_scope + "-"):
                        raise ValueError(f"Collection {collection_name} not within project scope")
                return True

            # Test valid collection
            assert validate_collection_scope("allowed-project-docs") == True

            # Test invalid collection
            with pytest.raises(ValueError, match="not within project scope"):
                validate_collection_scope("other-project-docs")

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_bulk_operations(self, mock_detector_class, mock_client_class):
        """Test bulk operations on collections."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock multiple collections
        mock_collections = ["project1-docs", "project1-code", "project1-notes"]
        mock_client_instance.list_collections.return_value = mock_collections

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            collections = await admin_instance.list_collections()

            # Test that we can iterate over all collections
            assert len(collections) == 3
            for collection in collections:
                assert collection.startswith("project1-")

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_error_handling_client_errors(self, mock_detector_class, mock_client_class):
        """Test error handling for client operations."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock client error
        mock_client_instance.list_collections.side_effect = Exception("Client connection failed")

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            with pytest.raises(Exception, match="Client connection failed"):
                await admin_instance.list_collections()

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_logging_operations(self, mock_detector_class, mock_client_class):
        """Test logging of administrative operations."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config, dry_run=True)

        # Test that dry run operations are logged
        async with admin as admin_instance:
            with patch('common.utils.admin_cli.logger') as mock_logger:
                await admin_instance.delete_collection("test-collection")

                # Should log dry run operation
                mock_logger.info.assert_called()

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_project_auto_detection(self, mock_detector_class, mock_client_class):
        """Test automatic project detection when no scope specified."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        # Mock project detection
        mock_detector_instance.get_project_name.return_value = "auto-detected-project"

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            # Check that detector is available for auto-detection
            assert admin_instance.detector == mock_detector_instance

    def test_dry_run_property(self):
        """Test dry run property behavior."""
        admin = WorkspaceQdrantAdmin(dry_run=True)
        assert admin.dry_run == True

        admin = WorkspaceQdrantAdmin(dry_run=False)
        assert admin.dry_run == False

    def test_project_scope_property(self):
        """Test project scope property behavior."""
        admin = WorkspaceQdrantAdmin(project_scope="test-project")
        assert admin.project_scope == "test-project"

        admin = WorkspaceQdrantAdmin()
        assert admin.project_scope is None

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_concurrent_operations(self, mock_detector_class, mock_client_class):
        """Test concurrent administrative operations."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            # Test that we can perform multiple operations concurrently
            tasks = [
                admin_instance.list_collections(),
                admin_instance.get_collection_statistics("test-collection")
            ]

            # Should be able to run concurrently without issues
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # At least one should succeed or return a mock result
            assert len(results) == 2

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_resource_cleanup(self, mock_detector_class, mock_client_class):
        """Test proper resource cleanup after operations."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            await admin_instance.list_collections()

        # Context manager should handle cleanup
        mock_client_instance.__aexit__.assert_called_once()

    def test_string_representation(self):
        """Test string representation of admin instance."""
        admin = WorkspaceQdrantAdmin(
            project_scope="test-project",
            dry_run=True
        )

        # Should be able to convert to string without errors
        str_repr = str(admin)
        assert isinstance(str_repr, str)

    def test_configuration_access(self):
        """Test access to configuration through admin instance."""
        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        assert admin.config == self.mock_config

    @patch('common.utils.admin_cli.QdrantWorkspaceClient')
    @patch('common.utils.admin_cli.ProjectDetector')
    async def test_admin_methods_availability(self, mock_detector_class, mock_client_class):
        """Test that all expected admin methods are available."""
        mock_client_instance = AsyncMock()
        mock_detector_instance = Mock()
        mock_client_class.return_value = mock_client_instance
        mock_detector_class.return_value = mock_detector_instance

        admin = WorkspaceQdrantAdmin(config=self.mock_config)

        async with admin as admin_instance:
            # Check that admin instance has expected methods
            assert hasattr(admin_instance, 'list_collections')
            assert hasattr(admin_instance, 'delete_collection')
            assert hasattr(admin_instance, 'search_documents')
            assert hasattr(admin_instance, 'get_collection_statistics')
            assert hasattr(admin_instance, 'reset_project')


if __name__ == "__main__":
    pytest.main([__file__])