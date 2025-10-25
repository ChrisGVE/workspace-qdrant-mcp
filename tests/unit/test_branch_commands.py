"""
Unit tests for branch management CLI commands.

This module tests the branch management functionality including:
- Deleting documents by branch
- Renaming branches across documents
- Listing branches with document counts
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    Record,
)
from typer.testing import CliRunner


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_collections():
    """Create mock collections response."""
    # Simple mock object with collections attribute
    mock_response = MagicMock()
    mock_coll1 = MagicMock()
    mock_coll1.name = "_path_abc123def456789a"
    mock_coll2 = MagicMock()
    mock_coll2.name = "_test123"
    mock_response.collections = [mock_coll1, mock_coll2]
    return mock_response


@pytest.fixture
def mock_empty_collections():
    """Create mock empty collections response."""
    mock_response = MagicMock()
    mock_response.collections = []
    return mock_response


@pytest.fixture
def mock_records_main():
    """Create mock records for main branch."""
    return [
        Record(id="1", payload={"branch": "main"}, vector=None),
        Record(id="2", payload={"branch": "main"}, vector=None),
        Record(id="3", payload={"branch": "main"}, vector=None),
    ]


@pytest.fixture
def mock_records_multiple_branches():
    """Create mock records with multiple branches."""
    return [
        Record(id="1", payload={"branch": "main"}, vector=None),
        Record(id="2", payload={"branch": "main"}, vector=None),
        Record(id="3", payload={"branch": "feature"}, vector=None),
        Record(id="4", payload={"branch": "develop"}, vector=None),
        Record(id="5", payload={"branch": "develop"}, vector=None),
    ]


class TestBranchDeleteCommand:
    """Test branch delete command."""

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_delete_branch_dry_run(
        self, mock_get_config, mock_client_class, runner, mock_collections, mock_records_main
    ):
        """Test delete command in dry-run mode."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = (mock_records_main, None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute dry-run
        result = runner.invoke(
            branch_app,
            ["delete", "--project", "path_abc123def456789a", "--branch", "main", "--dry-run"],
        )

        # Verify
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        assert "3 documents would be deleted" in result.stdout
        # Verify delete was NOT called
        mock_qdrant_client.delete.assert_not_called()

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_delete_branch_force(
        self, mock_get_config, mock_client_class, runner, mock_collections, mock_records_main
    ):
        """Test delete command with --force flag."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = (mock_records_main, None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute with force
        result = runner.invoke(
            branch_app,
            ["delete", "--project", "path_abc123def456789a", "--branch", "main", "--force"],
        )

        # Verify
        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout
        assert "3 documents" in result.stdout
        # Verify delete was called with FilterSelector
        mock_qdrant_client.delete.assert_called_once()
        call_args = mock_qdrant_client.delete.call_args
        assert call_args.kwargs["collection_name"] == "_path_abc123def456789a"
        assert isinstance(call_args.kwargs["points_selector"], FilterSelector)

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_delete_branch_empty_collection(
        self, mock_get_config, mock_client_class, runner, mock_collections
    ):
        """Test delete command with no matching documents."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = ([], None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute
        result = runner.invoke(
            branch_app,
            ["delete", "--project", "path_abc123def456789a", "--branch", "nonexistent"],
        )

        # Verify
        assert result.exit_code == 0
        assert "No documents found" in result.stdout

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_delete_branch_nonexistent_collection(
        self, mock_get_config, mock_client_class, runner, mock_empty_collections
    ):
        """Test delete command with nonexistent collection."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_empty_collections
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute
        result = runner.invoke(
            branch_app,
            ["delete", "--project", "path_abc123def456789a", "--branch", "main"],
        )

        # Verify
        assert result.exit_code == 1
        # Error message is written to stderr, check output (combines stdout and stderr)
        assert "does not exist" in (result.stdout + (result.stderr or ""))


class TestBranchRenameCommand:
    """Test branch rename command."""

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_rename_branch_dry_run(
        self, mock_get_config, mock_client_class, runner, mock_collections, mock_records_main
    ):
        """Test rename command in dry-run mode."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = (mock_records_main, None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute dry-run
        result = runner.invoke(
            branch_app,
            [
                "rename",
                "--project",
                "path_abc123def456789a",
                "--from",
                "main",
                "--to",
                "master",
                "--dry-run",
            ],
        )

        # Verify
        assert result.exit_code == 0
        assert "DRY RUN" in result.stdout
        assert "3 documents would have branch renamed" in result.stdout
        # Verify set_payload was NOT called
        mock_qdrant_client.set_payload.assert_not_called()

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_rename_branch_force(
        self, mock_get_config, mock_client_class, runner, mock_collections, mock_records_main
    ):
        """Test rename command with --force flag."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = (mock_records_main, None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute with force
        result = runner.invoke(
            branch_app,
            [
                "rename",
                "--project",
                "path_abc123def456789a",
                "--from",
                "old-name",
                "--to",
                "new-name",
                "--force",
            ],
        )

        # Verify
        assert result.exit_code == 0
        assert "Successfully renamed" in result.stdout
        # Verify set_payload was called with correct payload
        mock_qdrant_client.set_payload.assert_called()
        call_args = mock_qdrant_client.set_payload.call_args
        assert call_args.kwargs["payload"] == {"branch": "new-name"}


class TestBranchListCommand:
    """Test branch list command."""

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_list_branches_table_format(
        self,
        mock_get_config,
        mock_client_class,
        runner,
        mock_collections,
        mock_records_multiple_branches,
    ):
        """Test list command with table format."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = (mock_records_multiple_branches, None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute
        result = runner.invoke(
            branch_app,
            ["list", "--project", "path_abc123def456789a", "--format", "table"],
        )

        # Verify
        assert result.exit_code == 0
        assert "Branch" in result.stdout
        assert "Documents" in result.stdout
        assert "main" in result.stdout
        assert "feature" in result.stdout
        assert "develop" in result.stdout
        assert "3 branches" in result.stdout
        assert "5 documents" in result.stdout

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_list_branches_json_format(
        self,
        mock_get_config,
        mock_client_class,
        runner,
        mock_collections,
        mock_records_multiple_branches,
    ):
        """Test list command with JSON format."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = (mock_records_multiple_branches, None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute
        result = runner.invoke(
            branch_app,
            ["list", "--project", "path_abc123def456789a", "--format", "json"],
        )

        # Verify
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["collection"] == "_path_abc123def456789a"
        assert data["project_id"] == "path_abc123def456789a"
        assert data["total_branches"] == 3
        assert len(data["branches"]) == 3

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_list_branches_simple_format(
        self,
        mock_get_config,
        mock_client_class,
        runner,
        mock_collections,
        mock_records_multiple_branches,
    ):
        """Test list command with simple format."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = (mock_records_multiple_branches, None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute
        result = runner.invoke(
            branch_app,
            ["list", "--project", "path_abc123def456789a", "--format", "simple"],
        )

        # Verify
        assert result.exit_code == 0
        assert "main: 2" in result.stdout
        assert "feature: 1" in result.stdout
        assert "develop: 2" in result.stdout

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    @patch("wqm_cli.cli.commands.branch.get_current_branch")
    def test_list_branches_with_current(
        self,
        mock_get_branch,
        mock_get_config,
        mock_client_class,
        runner,
        mock_collections,
        mock_records_multiple_branches,
    ):
        """Test list command with --show-current flag."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_get_branch.return_value = "main"
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = (mock_records_multiple_branches, None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute
        result = runner.invoke(
            branch_app,
            [
                "list",
                "--project",
                "path_abc123def456789a",
                "--format",
                "table",
                "--show-current",
            ],
        )

        # Verify
        assert result.exit_code == 0
        assert "Current" in result.stdout
        assert "Current Git branch: main" in result.stdout

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_list_branches_empty_collection(
        self, mock_get_config, mock_client_class, runner, mock_collections
    ):
        """Test list command with empty collection."""
        from wqm_cli.cli.commands.branch import branch_app

        # Setup mocks
        mock_client = AsyncMock()
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.get_collections.return_value = mock_collections
        mock_qdrant_client.scroll.return_value = ([], None)
        mock_client.client = mock_qdrant_client
        mock_client.initialize = AsyncMock()
        mock_client.close = AsyncMock()
        mock_client_class.return_value = mock_client

        # Execute
        result = runner.invoke(
            branch_app,
            ["list", "--project", "path_abc123def456789a"],
        )

        # Verify
        assert result.exit_code == 0
        assert "No documents found" in result.stdout


class TestBranchCommandHelp:
    """Test branch command help and usage."""

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_branch_command_help(self, mock_get_config, mock_client_class, runner):
        """Test branch command help output."""
        from wqm_cli.cli.commands.branch import branch_app

        result = runner.invoke(branch_app, ["--help"])

        assert result.exit_code == 0
        assert "Branch management" in result.stdout
        assert "delete" in result.stdout
        assert "rename" in result.stdout
        assert "list" in result.stdout

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_delete_command_help(self, mock_get_config, mock_client_class, runner):
        """Test delete command help output."""
        from wqm_cli.cli.commands.branch import branch_app

        result = runner.invoke(branch_app, ["delete", "--help"])

        assert result.exit_code == 0
        assert "Delete all documents with specified branch" in result.stdout
        assert "--project" in result.stdout
        assert "--branch" in result.stdout
        assert "--force" in result.stdout
        assert "--dry-run" in result.stdout

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_rename_command_help(self, mock_get_config, mock_client_class, runner):
        """Test rename command help output."""
        from wqm_cli.cli.commands.branch import branch_app

        result = runner.invoke(branch_app, ["rename", "--help"])

        assert result.exit_code == 0
        assert "Rename branch metadata" in result.stdout
        assert "--project" in result.stdout
        assert "--from" in result.stdout
        assert "--to" in result.stdout

    @patch("wqm_cli.cli.commands.branch.QdrantWorkspaceClient")
    @patch("wqm_cli.cli.commands.branch.get_config")
    def test_list_command_help(self, mock_get_config, mock_client_class, runner):
        """Test list command help output."""
        from wqm_cli.cli.commands.branch import branch_app

        result = runner.invoke(branch_app, ["list", "--help"])

        assert result.exit_code == 0
        assert "List all branches" in result.stdout
        assert "--project" in result.stdout
        assert "--format" in result.stdout
        assert "--show-current" in result.stdout
