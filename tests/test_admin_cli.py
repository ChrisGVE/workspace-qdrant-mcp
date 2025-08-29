"""
Tests for workspace-qdrant administrative CLI.

Tests the safety features, project scoping, and collection management.
"""

from unittest.mock import Mock, patch

import pytest

from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.utils.admin_cli import WorkspaceQdrantAdmin


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=Config)
    config.workspace = Mock()
    config.workspace.global_collections = ["global-docs", "shared"]
    config.workspace.github_user = "testuser"
    config.qdrant_client_config = {"url": "http://localhost:6333"}
    return config


@pytest.fixture
def mock_client():
    """Mock Qdrant client for testing."""
    client = Mock()

    # Mock collections response
    collections_response = Mock()
    mock_collections = [
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
    ]
    mock_collections[0].name = "test-project-scratchbook"
    mock_collections[1].name = "test-project-docs"
    mock_collections[2].name = "subproject-docs"
    mock_collections[3].name = "global-docs"
    mock_collections[4].name = "memexd-main-code"  # Protected collection
    mock_collections[5].name = "external-collection"  # Not project-scoped

    collections_response.collections = mock_collections
    client.get_collections.return_value = collections_response

    # Mock collection info response
    collection_info = Mock()
    collection_info.points_count = 100
    collection_info.vectors_count = 100
    collection_info.status = "green"
    collection_info.optimizer_status = Mock()
    collection_info.config = Mock()
    collection_info.config.params = Mock()
    collection_info.config.params.vectors = Mock()
    collection_info.config.params.vectors.distance = Mock()
    collection_info.config.params.vectors.distance.value = "Cosine"
    collection_info.config.params.vectors.size = 384
    client.get_collection.return_value = collection_info

    return client


@pytest.fixture
def mock_project_detector():
    """Mock project detector for testing."""
    detector = Mock()
    detector.get_project_info.return_value = {
        "main_project": "test-project",
        "subprojects": ["subproject"],
        "is_git_repo": True,
        "remote_url": "https://github.com/testuser/test-project.git",
    }
    return detector


class TestWorkspaceQdrantAdmin:
    """Test cases for WorkspaceQdrantAdmin class."""

    def test_init(self, mock_config):
        """Test admin initialization."""
        with (
            patch("workspace_qdrant_mcp.utils.admin_cli.QdrantClient") as mock_qdrant,
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config)

            assert admin.config == mock_config
            assert admin.current_project == "test-project"
            mock_qdrant.assert_called_once_with(**mock_config.qdrant_client_config)

    def test_get_protected_collections(self, mock_config, mock_client):
        """Test identification of protected collections."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config)
            protected = admin.get_protected_collections()

            assert "memexd-main-code" in protected
            assert len(protected) == 1

    def test_is_project_scoped_collection(self, mock_config, mock_client):
        """Test project scoping logic."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": ["subproject"],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config)

            # Test project collections
            assert admin.is_project_scoped_collection("test-project-scratchbook")
            assert admin.is_project_scoped_collection("subproject-docs")

            # Test global collections
            assert admin.is_project_scoped_collection("global-docs")

            # Test external collections
            assert not admin.is_project_scoped_collection("external-collection")

    def test_validate_collection_for_deletion_success(self, mock_config, mock_client):
        """Test successful collection validation."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config)
            can_delete, reason = admin.validate_collection_for_deletion(
                "test-project-scratchbook"
            )

            assert can_delete
            assert "can be safely deleted" in reason

    def test_validate_collection_for_deletion_protected(self, mock_config, mock_client):
        """Test validation fails for protected collections."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config)
            can_delete, reason = admin.validate_collection_for_deletion(
                "memexd-main-code"
            )

            assert not can_delete
            assert "protected" in reason

    def test_validate_collection_for_deletion_out_of_scope(
        self, mock_config, mock_client
    ):
        """Test validation fails for out-of-scope collections."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config)
            can_delete, reason = admin.validate_collection_for_deletion(
                "external-collection"
            )

            assert not can_delete
            assert "outside current project scope" in reason

    def test_delete_collection_dry_run(self, mock_config, mock_client):
        """Test dry-run collection deletion."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config, dry_run=True)
            success = admin.delete_collection("test-project-scratchbook")

            assert success
            mock_client.delete_collection.assert_not_called()

    def test_delete_collection_force(self, mock_config, mock_client):
        """Test forced collection deletion."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config)
            success = admin.delete_collection("test-project-scratchbook", force=True)

            assert success
            mock_client.delete_collection.assert_called_once_with(
                "test-project-scratchbook"
            )

    def test_list_collections(self, mock_config, mock_client):
        """Test collection listing."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"
            ) as mock_manager,
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            # Mock the _is_workspace_collection method
            mock_manager.return_value._is_workspace_collection.side_effect = (
                lambda name: not name.endswith("-code")
            )

            # Mock client to return the proper collections
            mock_client.get_collections.return_value = collections_response

            admin = WorkspaceQdrantAdmin(config=mock_config)
            collections = admin.list_collections(show_all=False)

            expected_collections = [
                "external-collection",
                "global-docs",
                "subproject-docs",
                "test-project-docs",
                "test-project-scratchbook",
            ]
            assert collections == expected_collections

    def test_get_collection_info_single(self, mock_config, mock_client):
        """Test getting info for a single collection."""
        with (
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.QdrantClient",
                return_value=mock_client,
            ),
            patch("workspace_qdrant_mcp.utils.admin_cli.WorkspaceCollectionManager"),
            patch(
                "workspace_qdrant_mcp.utils.admin_cli.ProjectDetector"
            ) as mock_detector,
        ):
            mock_detector.return_value.get_project_info.return_value = {
                "main_project": "test-project",
                "subprojects": [],
                "is_git_repo": True,
            }

            admin = WorkspaceQdrantAdmin(config=mock_config)
            info = admin.get_collection_info("test-project-scratchbook")

            assert "test-project-scratchbook" in info
            collection_data = info["test-project-scratchbook"]
            assert collection_data["points_count"] == 100
            assert collection_data["project_scoped"] is True
            assert collection_data["protected"] is False


# CLI tests temporarily disabled - CLI implementation uses argparse not Typer
# TODO: Update CLI tests to work with argparse-based implementation
# class TestCLICommands:
#     """Test cases for CLI commands."""
#
#     def setup_method(self):
#         """Set up test runner."""
#         pass
#
#     def test_list_collections_command(self):
#         """Test list collections CLI command."""
#         pytest.skip("CLI tests need updating for argparse implementation")
#
#     def test_delete_collection_command_dry_run(self):
#         """Test delete collection CLI command with dry-run."""
#         pytest.skip("CLI tests need updating for argparse implementation")
#
#     def test_collection_info_command(self):
#         """Test collection info CLI command."""
#         pytest.skip("CLI tests need updating for argparse implementation")


@pytest.mark.integration
class TestAdminCLIIntegration:
    """Integration tests for admin CLI with real Qdrant."""

    @pytest.mark.requires_qdrant
    def test_list_collections_integration(self):
        """Integration test for listing collections."""
        # This would connect to a real Qdrant instance
        # Skip if QDRANT_URL not available
        pytest.skip("Integration test - requires running Qdrant instance")

    @pytest.mark.requires_qdrant
    def test_collection_operations_integration(self):
        """Integration test for collection operations."""
        # This would test real collection creation/deletion
        pytest.skip("Integration test - requires running Qdrant instance")
