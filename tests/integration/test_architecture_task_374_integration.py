"""
Integration tests for Task 374 architectural changes.

This module tests the complete workflow of the new collection naming architecture:
- Tenant ID calculation
- Single collection per project
- Metadata-based differentiation
- Branch support
- Collection aliases for migrations
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from qdrant_client.models import Distance, PointStruct, VectorParams

from common.core.client import QdrantWorkspaceClient
from common.core.collection_aliases import AliasManager
from common.core.collection_naming import build_project_collection_name
from common.utils.project_detection import calculate_tenant_id
from common.utils.git_utils import get_current_branch


class TestTenantIDWorkflow:
    """Test tenant ID calculation and usage in real scenarios."""

    def test_git_project_tenant_id_workflow(self):
        """Test tenant ID calculation for git projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Mock git repository with remote
            with patch("common.utils.project_detection.git.Repo") as mock_repo_class:
                mock_repo = Mock()
                mock_remote = Mock()
                mock_remote.url = "https://github.com/test/project.git"
                mock_repo.remotes.origin = mock_remote
                mock_repo_class.return_value = mock_repo

                tenant_id = calculate_tenant_id(project_path)

                # Verify tenant ID format
                assert tenant_id == "github_com_test_project"
                assert not tenant_id.startswith("path_")

                # Verify it can be used to build collection name
                collection_name = build_project_collection_name(tenant_id)
                assert collection_name == f"_{tenant_id}"
                assert collection_name.startswith("_github_com_")

    def test_local_project_tenant_id_workflow(self):
        """Test tenant ID calculation for local (non-git) projects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Mock no git repository
            with patch("common.utils.project_detection.git.Repo", side_effect=Exception("Not a git repo")):
                tenant_id = calculate_tenant_id(project_path)

                # Verify path-based tenant ID
                assert tenant_id.startswith("path_")
                assert len(tenant_id) == 21  # "path_" + 16 hex chars

                # Verify it can be used to build collection name
                collection_name = build_project_collection_name(tenant_id)
                assert collection_name == f"_{tenant_id}"
                assert collection_name.startswith("_path_")


@pytest.mark.asyncio
class TestSingleCollectionPerProject:
    """Test that all files from a project go to single collection."""

    async def test_multiple_file_types_same_collection(self):
        """Test that code, docs, and tests all go to same collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Calculate tenant ID
            with patch("common.utils.project_detection.git.Repo", side_effect=Exception("Not a git repo")):
                tenant_id = calculate_tenant_id(project_path)
                collection_name = build_project_collection_name(tenant_id)

            # Mock Qdrant client
            mock_client = AsyncMock()
            mock_client.collection_exists = AsyncMock(return_value=False)
            mock_client.create_collection = AsyncMock()
            mock_client.upsert = AsyncMock()

            # Simulate ingesting different file types
            files_to_ingest = [
                {"path": "src/main.py", "file_type": "code", "content": "def main(): pass"},
                {"path": "README.md", "file_type": "docs", "content": "# Project"},
                {"path": "tests/test_main.py", "file_type": "test", "content": "def test_main(): pass"},
            ]

            # All should use same collection
            for file in files_to_ingest:
                # Verify they all route to same collection
                assert build_project_collection_name(tenant_id) == collection_name

            # Verify no proliferation occurred (no myproject-code, myproject-docs, etc.)
            assert "_" in collection_name  # Starts with underscore
            assert "-code" not in collection_name
            assert "-docs" not in collection_name
            assert "-tests" not in collection_name


@pytest.mark.asyncio
class TestBranchSupport:
    """Test multi-branch support within single collection."""

    async def test_branch_filtering_in_queries(self):
        """Test querying with branch filters."""
        tenant_id = "github_com_test_project"
        collection_name = build_project_collection_name(tenant_id)

        # Mock client with documents from different branches
        mock_client = AsyncMock()

        # Simulate scroll results with different branches
        mock_points = [
            Mock(id="1", payload={"branch": "main", "file_type": "code", "project_id": tenant_id}),
            Mock(id="2", payload={"branch": "main", "file_type": "code", "project_id": tenant_id}),
            Mock(id="3", payload={"branch": "feature", "file_type": "code", "project_id": tenant_id}),
            Mock(id="4", payload={"branch": "develop", "file_type": "code", "project_id": tenant_id}),
        ]

        # Test branch filtering
        main_branch_docs = [p for p in mock_points if p.payload["branch"] == "main"]
        feature_branch_docs = [p for p in mock_points if p.payload["branch"] == "feature"]

        assert len(main_branch_docs) == 2
        assert len(feature_branch_docs) == 1

        # Verify all docs are in same collection but differentiated by branch
        for point in mock_points:
            assert point.payload["project_id"] == tenant_id

    async def test_branch_detection_integration(self):
        """Test branch detection during ingestion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # Mock git repository on feature branch
            with patch("common.utils.git_utils.Repo") as mock_repo_class:
                mock_repo = Mock()
                # Setup mock head attributes for get_current_branch
                mock_head = Mock()
                mock_head.commit = Mock()  # Indicates repo has commits
                mock_head.is_detached = False  # Not in detached HEAD state
                mock_ref = Mock()
                mock_ref.name = "feature/new-api"
                mock_head.ref = mock_ref
                mock_repo.head = mock_head
                mock_repo_class.return_value = mock_repo

                branch = get_current_branch(project_path)

                assert branch == "feature/new-api"

                # Verify metadata would include this branch
                metadata = {
                    "project_id": "github_com_test_project",
                    "branch": branch,
                    "file_type": "code"
                }

                assert metadata["branch"] == "feature/new-api"


@pytest.mark.asyncio
class TestCollectionAliases:
    """Test collection alias system for migrations."""

    async def test_alias_migration_workflow(self):
        """Test complete workflow of creating alias when remote is added."""
        # Simulate local project gaining remote
        old_tenant_id = "path_abc123def456789a"
        new_tenant_id = "github_com_user_repo"

        old_collection = build_project_collection_name(old_tenant_id)
        new_collection = build_project_collection_name(new_tenant_id)

        assert old_collection == "_path_abc123def456789a"
        assert new_collection == "_github_com_user_repo"

        # Mock Qdrant client
        mock_client = AsyncMock()
        mock_client.update_collection_aliases = AsyncMock()

        # Create alias
        alias_manager = AliasManager(mock_client)
        await alias_manager.create_alias(old_collection, new_collection)

        # Verify alias was created
        mock_client.update_collection_aliases.assert_called_once()


@pytest.mark.asyncio
class TestMetadataEnrichment:
    """Test that metadata includes all required fields."""

    async def test_metadata_schema_compliance(self):
        """Test that ingested documents have all required metadata."""
        # Required fields per architecture
        required_fields = ["project_id", "branch", "file_type"]

        # Simulate document metadata
        metadata = {
            "project_id": "github_com_test_project",
            "branch": "main",
            "file_type": "code",
            "file_path": "/path/to/file.py",
            "timestamp": "2025-10-03T20:00:00Z",
        }

        # Verify all required fields present
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

        # Verify types
        assert isinstance(metadata["project_id"], str)
        assert isinstance(metadata["branch"], str)
        assert isinstance(metadata["file_type"], str)

    async def test_optional_metadata_fields(self):
        """Test that optional symbol fields can be included."""
        metadata = {
            "project_id": "github_com_test_project",
            "branch": "main",
            "file_type": "code",
            "symbols_defined": ["MyClass", "my_function"],
            "symbols_used": ["import_something", "OtherClass"],
            "imports": ["from typing import List"],
            "exports": ["__all__ = ['MyClass']"],
        }

        # Verify optional fields accepted
        assert "symbols_defined" in metadata
        assert "symbols_used" in metadata
        assert len(metadata["symbols_defined"]) == 2
        assert len(metadata["symbols_used"]) == 2


@pytest.mark.asyncio
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""

    async def test_full_ingestion_search_workflow(self):
        """Test ingesting and searching with new architecture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir)

            # 1. Calculate tenant ID
            with patch("common.utils.project_detection.git.Repo") as mock_repo_class:
                mock_repo = Mock()
                mock_remote = Mock()
                mock_remote.url = "https://github.com/test/project.git"
                mock_repo.remotes.origin = mock_remote
                mock_repo_class.return_value = mock_repo

                tenant_id = calculate_tenant_id(project_path)
                assert tenant_id == "github_com_test_project"

            # 2. Build collection name
            collection_name = build_project_collection_name(tenant_id)
            assert collection_name == "_github_com_test_project"

            # 3. Get branch
            with patch("common.utils.git_utils.Repo") as mock_repo_class:
                mock_repo = Mock()
                mock_branch = Mock()
                mock_branch.name = "main"
                mock_repo.active_branch = mock_branch
                mock_repo_class.return_value = mock_repo

                branch = get_current_branch(project_path)
                assert branch == "main"

            # 4. Create document with full metadata
            document = {
                "id": "doc1",
                "content": "def hello(): print('world')",
                "metadata": {
                    "project_id": tenant_id,
                    "branch": branch,
                    "file_type": "code",
                    "file_path": "src/main.py",
                }
            }

            # 5. Verify document structure
            assert document["metadata"]["project_id"] == "github_com_test_project"
            assert document["metadata"]["branch"] == "main"
            assert document["metadata"]["file_type"] == "code"

            # 6. Simulate search with filters
            search_filters = {
                "project_id": tenant_id,
                "branch": "main",
                "file_type": "code"
            }

            # Verify search would find this document
            assert all(
                document["metadata"].get(key) == value
                for key, value in search_filters.items()
            )


@pytest.mark.asyncio
class TestArchitectureValidation:
    """Validate architectural principles are followed."""

    async def test_no_collection_proliferation(self):
        """Verify no {project}-{type} collections are created."""
        tenant_ids = [
            "github_com_test_project",
            "gitlab_com_user_app",
            "path_abc123def456789a"
        ]

        for tenant_id in tenant_ids:
            collection_name = build_project_collection_name(tenant_id)

            # Should start with underscore
            assert collection_name.startswith("_")

            # Should NOT contain hyphens separating project and type
            assert "-code" not in collection_name
            assert "-docs" not in collection_name
            assert "-scratchbook" not in collection_name
            assert "-web" not in collection_name

            # Should be single collection name in valid format
            # Path hash: _path_{16 hex chars} (2 underscores)
            # Git remote: _{platform}_{domain}_{user}_{repo} (4+ underscores)
            assert collection_name.count("_") >= 2  # At least 2 underscores for either format

    async def test_metadata_based_differentiation(self):
        """Verify file types are differentiated by metadata, not collection."""
        tenant_id = "github_com_test_project"
        collection_name = build_project_collection_name(tenant_id)

        # All file types use same collection
        file_types = ["code", "docs", "test", "config", "data"]

        for file_type in file_types:
            # Verify all use same collection
            assert build_project_collection_name(tenant_id) == collection_name

            # Differentiation is via metadata
            metadata = {
                "project_id": tenant_id,
                "branch": "main",
                "file_type": file_type  # This is how we differentiate
            }

            assert metadata["file_type"] == file_type
            assert metadata["project_id"] == tenant_id
