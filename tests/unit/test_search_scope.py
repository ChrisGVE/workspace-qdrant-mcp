"""Unit tests for search tool scope parameter (Task 396).

Tests multi-tenant filtering with scope and include_libraries parameters.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from workspace_qdrant_mcp.server import (
    build_metadata_filters,
    _merge_with_rrf,
    CANONICAL_COLLECTIONS,
)
from qdrant_client.models import FieldCondition, MatchValue


class TestBuildMetadataFilters:
    """Tests for build_metadata_filters function."""

    def test_project_id_filter_added(self):
        """Test that project_id filter is added when provided."""
        result = build_metadata_filters(project_id="abc123def456")

        assert result is not None
        assert len(result.must) >= 1

        # Find the project_id condition
        project_id_conditions = [
            c for c in result.must
            if isinstance(c, FieldCondition) and c.key == "project_id"
        ]
        assert len(project_id_conditions) == 1
        assert project_id_conditions[0].match.value == "abc123def456"

    def test_no_project_id_filter_when_none(self):
        """Test that project_id filter is not added when None."""
        with patch("workspace_qdrant_mcp.server.get_current_branch", return_value="main"):
            result = build_metadata_filters(project_id=None, branch="*")

        # Should return None or empty filter when no conditions
        if result is not None:
            # Check no project_id condition
            project_id_conditions = [
                c for c in result.must
                if isinstance(c, FieldCondition) and c.key == "project_id"
            ]
            assert len(project_id_conditions) == 0

    def test_all_filters_combined(self):
        """Test that all filters are combined correctly."""
        with patch("workspace_qdrant_mcp.server.get_current_branch", return_value="develop"):
            result = build_metadata_filters(
                project_id="test_project_id",
                branch="main",
                file_type="code",
                filters={"author": "john"}
            )

        assert result is not None
        assert len(result.must) == 4  # project_id, branch, file_type, author

        # Check all conditions exist
        keys = [c.key for c in result.must if isinstance(c, FieldCondition)]
        assert "project_id" in keys
        assert "branch" in keys
        assert "file_type" in keys
        assert "author" in keys


class TestMergeWithRRF:
    """Tests for Reciprocal Rank Fusion merging."""

    def test_single_list_preserved(self):
        """Test that single result list is preserved."""
        results = [
            [
                {"id": "1", "score": 0.9, "content": "first"},
                {"id": "2", "score": 0.8, "content": "second"},
            ]
        ]

        merged = _merge_with_rrf(results)

        assert len(merged) == 2
        assert merged[0]["id"] == "1"
        assert merged[1]["id"] == "2"

    def test_multiple_lists_merged(self):
        """Test that multiple result lists are merged with RRF."""
        results = [
            [
                {"id": "1", "score": 0.9, "content": "first"},
                {"id": "2", "score": 0.8, "content": "second"},
            ],
            [
                {"id": "2", "score": 0.95, "content": "second"},
                {"id": "3", "score": 0.7, "content": "third"},
            ],
        ]

        merged = _merge_with_rrf(results)

        assert len(merged) == 3
        # Item 2 appears in both lists, should have highest RRF score
        ids = [r["id"] for r in merged]
        assert "2" in ids
        assert "1" in ids
        assert "3" in ids

    def test_rrf_score_calculation(self):
        """Test that RRF scores are calculated correctly."""
        results = [
            [{"id": "1", "score": 0.9, "content": "first"}],
            [{"id": "1", "score": 0.8, "content": "first"}],
        ]

        merged = _merge_with_rrf(results, k=60)

        # RRF score = 1/(60+1) + 1/(60+1) = 2/61
        expected_rrf = 2 / 61
        assert len(merged) == 1
        assert abs(merged[0]["rrf_score"] - expected_rrf) < 0.001

    def test_empty_lists_handled(self):
        """Test that empty result lists are handled."""
        results = [[], []]

        merged = _merge_with_rrf(results)

        assert len(merged) == 0


class TestUnifiedCollections:
    """Tests for CANONICAL_COLLECTIONS constant."""

    def test_collections_defined(self):
        """Test that all unified collections are defined."""
        assert "projects" in CANONICAL_COLLECTIONS
        assert "libraries" in CANONICAL_COLLECTIONS
        assert "memory" in CANONICAL_COLLECTIONS

    def test_collection_names_correct(self):
        """Test that collection names follow naming convention."""
        assert CANONICAL_COLLECTIONS["projects"] == "projects"
        assert CANONICAL_COLLECTIONS["libraries"] == "libraries"
        assert CANONICAL_COLLECTIONS["memory"] == "memory"


class TestScopeValidation:
    """Tests for scope parameter validation."""

    @pytest.mark.asyncio
    async def test_invalid_scope_returns_error(self):
        """Test that invalid scope returns error response."""
        # Import the decorated search function and get the underlying function
        from workspace_qdrant_mcp.server import search as search_tool

        # The FunctionTool has a .fn attribute that holds the actual function
        search_fn = search_tool.fn if hasattr(search_tool, 'fn') else search_tool

        with patch("workspace_qdrant_mcp.server.initialize_components"):
            result = await search_fn(query="test", scope="invalid")

        assert result["success"] is False
        assert "Invalid scope" in result["error"]

    @pytest.mark.asyncio
    async def test_valid_scopes_accepted(self):
        """Test that valid scopes are accepted."""
        valid_scopes = ["project", "global", "all"]

        from workspace_qdrant_mcp.server import search as search_tool
        search_fn = search_tool.fn if hasattr(search_tool, 'fn') else search_tool

        for scope in valid_scopes:
            # Just verify the scope validation passes (not full search)
            # We mock everything to avoid actual Qdrant calls
            with patch("workspace_qdrant_mcp.server.initialize_components"), \
                 patch("workspace_qdrant_mcp.server.ensure_collection_exists", return_value=False):
                result = await search_fn(query="test", scope=scope)

                # Should get past scope validation
                # May fail for other reasons but not scope validation
                if not result["success"]:
                    assert "Invalid scope" not in result.get("error", "")
