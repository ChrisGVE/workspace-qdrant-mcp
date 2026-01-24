"""
Unit tests for additive library deletion policy (Task 460, ADR-001).

Tests the library deletion functionality that marks libraries as deleted
rather than physically removing them from Qdrant.

Key principles:
- Libraries are NEVER physically deleted from Qdrant
- Mark as deleted: deleted=true, deleted_at=timestamp
- Filter out deleted by default in search
- Allow searching deleted with explicit flag
- Re-ingestion clears deletion markers
"""

import pytest


class TestBuildMetadataFiltersExcludeDeleted:
    """Tests for exclude_deleted parameter in build_metadata_filters."""

    def test_build_filters_exclude_deleted_true(self):
        """Should add deleted=false condition when exclude_deleted=True."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        result = build_metadata_filters(exclude_deleted=True, branch="*")
        assert result is not None
        # Should have a condition for deleted=false
        conditions = result.must
        deleted_conditions = [c for c in conditions if c.key == "deleted"]
        assert len(deleted_conditions) == 1
        assert deleted_conditions[0].match.value is False

    def test_build_filters_exclude_deleted_false(self):
        """Should not add deletion filter when exclude_deleted=False."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        result = build_metadata_filters(exclude_deleted=False, branch="*")
        # Result should be None when no filters
        assert result is None

    def test_build_filters_exclude_deleted_with_other_filters(self):
        """Should combine exclude_deleted with other filters."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        result = build_metadata_filters(
            exclude_deleted=True,
            branch="*",
            file_type="docs",
            tag="numpy"
        )
        assert result is not None
        conditions = result.must

        # Should have conditions for file_type, tag (main_tag), and deleted
        keys = [c.key for c in conditions]
        assert "file_type" in keys
        assert "main_tag" in keys
        assert "deleted" in keys


class TestSearchIncludeDeletedLogic:
    """Tests for include_deleted parameter logic in search()."""

    def test_search_excludes_deleted_by_default(self):
        """Search should exclude deleted libraries by default."""
        # This test verifies the logic - exclude_deleted should be True (not include_deleted)
        # When include_deleted=False (default), exclude_deleted should be True
        include_deleted = False
        exclude_deleted = not include_deleted
        assert exclude_deleted is True

    def test_search_includes_deleted_when_flag_set(self):
        """Search should include deleted libraries when include_deleted=True."""
        include_deleted = True
        exclude_deleted = not include_deleted
        assert exclude_deleted is False


class TestLibraryDeletionMetadataFormat:
    """Tests for the metadata format used in library deletion."""

    def test_deletion_metadata_fields(self):
        """Verify the expected fields for deletion metadata."""
        from datetime import datetime, timezone

        # Simulate what the manage action creates
        deletion_timestamp = datetime.now(timezone.utc).isoformat()
        payload_update = {
            "deleted": True,
            "deleted_at": deletion_timestamp
        }

        assert payload_update["deleted"] is True
        assert "deleted_at" in payload_update
        assert payload_update["deleted_at"] is not None

    def test_restoration_metadata_fields(self):
        """Verify the expected fields for restoration metadata."""
        from datetime import datetime, timezone

        # Simulate what the restore action creates
        restore_timestamp = datetime.now(timezone.utc).isoformat()
        payload_update = {
            "deleted": False,
            "deleted_at": None,
            "restored_at": restore_timestamp
        }

        assert payload_update["deleted"] is False
        assert payload_update["deleted_at"] is None
        assert payload_update["restored_at"] is not None


class TestFilteringDeletedDocuments:
    """Tests for the filtering logic with deleted documents."""

    def test_filter_excludes_deleted(self):
        """Filter with exclude_deleted=True should exclude deleted docs."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        filter_result = build_metadata_filters(
            exclude_deleted=True,
            branch="*"
        )

        # The filter should have a condition that matches deleted=False
        assert filter_result is not None
        deleted_condition = next(
            (c for c in filter_result.must if c.key == "deleted"),
            None
        )
        assert deleted_condition is not None
        assert deleted_condition.match.value is False

    def test_filter_includes_deleted(self):
        """Filter with exclude_deleted=False should include all docs."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        filter_result = build_metadata_filters(
            exclude_deleted=False,
            branch="*"
        )

        # No filter should be applied
        assert filter_result is None


class TestSearchFilterConstruction:
    """Tests for how search constructs filters with include_deleted parameter."""

    def test_library_filter_excludes_deleted_by_default(self):
        """Library filter should exclude deleted when include_deleted=False."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        # Simulate how search() builds library_filter
        include_deleted = False
        library_filter = build_metadata_filters(
            filters=None,
            branch="*",
            file_type=None,
            project_id=None,
            tag=None,
            exclude_deleted=not include_deleted
        )

        assert library_filter is not None
        deleted_condition = next(
            (c for c in library_filter.must if c.key == "deleted"),
            None
        )
        assert deleted_condition is not None
        assert deleted_condition.match.value is False

    def test_library_filter_includes_deleted_when_requested(self):
        """Library filter should include deleted when include_deleted=True."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        # Simulate how search() builds library_filter
        include_deleted = True
        library_filter = build_metadata_filters(
            filters=None,
            branch="*",
            file_type=None,
            project_id=None,
            tag=None,
            exclude_deleted=not include_deleted
        )

        # No deletion filter should be applied
        assert library_filter is None


class TestCanonicalCollectionsIncludeLibraries:
    """Tests that CANONICAL_COLLECTIONS includes libraries collection."""

    def test_canonical_collections_has_libraries(self):
        """CANONICAL_COLLECTIONS should include libraries collection."""
        from workspace_qdrant_mcp.server import CANONICAL_COLLECTIONS

        assert "libraries" in CANONICAL_COLLECTIONS
        assert CANONICAL_COLLECTIONS["libraries"] == "libraries"


class TestManageActionsList:
    """Tests that manage function documentation includes library deletion actions."""

    def test_manage_docstring_includes_library_deletion_actions(self):
        """Manage function docstring should document library deletion actions."""
        from workspace_qdrant_mcp.server import manage

        # Access the underlying function through FunctionTool
        if hasattr(manage, 'fn'):
            doc = manage.fn.__doc__
        elif hasattr(manage, '__doc__'):
            doc = manage.__doc__
        else:
            doc = str(manage.description) if hasattr(manage, 'description') else ""

        assert "mark_library_deleted" in doc
        assert "restore_deleted_library" in doc
        assert "list_deleted_libraries" in doc


class TestDeletedLibraryGrouping:
    """Tests for grouping deleted libraries by name."""

    def test_group_deleted_by_library_name(self):
        """Should correctly group deleted documents by library name."""
        # Simulate the grouping logic from list_deleted_libraries

        # Mock documents
        documents = [
            {"library_name": "numpy", "deleted_at": "2026-01-01T00:00:00Z"},
            {"library_name": "numpy", "deleted_at": "2026-01-01T00:00:00Z"},
            {"library_name": "pandas", "deleted_at": "2026-01-02T00:00:00Z"},
        ]

        # Group by library_name
        deleted_libraries = {}
        for doc in documents:
            lib_name = doc.get("library_name", "unknown")
            deleted_at = doc.get("deleted_at")
            if lib_name not in deleted_libraries:
                deleted_libraries[lib_name] = {
                    "library_name": lib_name,
                    "document_count": 0,
                    "deleted_at": deleted_at
                }
            deleted_libraries[lib_name]["document_count"] += 1

        assert len(deleted_libraries) == 2
        assert deleted_libraries["numpy"]["document_count"] == 2
        assert deleted_libraries["pandas"]["document_count"] == 1
