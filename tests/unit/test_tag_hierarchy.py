"""
Unit tests for dot-separated tag hierarchy (Task 459, ADR-001).

Tests the tag generation, parsing, and validation functions implemented
for hierarchical organization within collections.

Tag format: main_tag.sub_tag
- Projects: project_id.branch (e.g., workspace-qdrant-mcp.feature-auth)
- Libraries: library_name.version (e.g., numpy.1.24.0)
"""

import pytest


class TestGenerateMainTag:
    """Tests for generate_main_tag function."""

    def test_generate_main_tag_with_project_id(self):
        """Should return project_id as main_tag."""
        from workspace_qdrant_mcp.server import generate_main_tag

        result = generate_main_tag(project_id="a1b2c3d4e5f6")
        assert result == "a1b2c3d4e5f6"

    def test_generate_main_tag_with_library_name(self):
        """Should return library_name as main_tag."""
        from workspace_qdrant_mcp.server import generate_main_tag

        result = generate_main_tag(library_name="numpy")
        assert result == "numpy"

    def test_generate_main_tag_project_takes_precedence(self):
        """Project_id should take precedence over library_name."""
        from workspace_qdrant_mcp.server import generate_main_tag

        result = generate_main_tag(project_id="myproject", library_name="numpy")
        assert result == "myproject"

    def test_generate_main_tag_none_when_no_input(self):
        """Should return None when neither project_id nor library_name provided."""
        from workspace_qdrant_mcp.server import generate_main_tag

        result = generate_main_tag()
        assert result is None


class TestGenerateFullTag:
    """Tests for generate_full_tag function."""

    def test_generate_full_tag_with_sub_tag(self):
        """Should join main_tag and sub_tag with dot separator."""
        from workspace_qdrant_mcp.server import generate_full_tag

        result = generate_full_tag("a1b2c3d4e5f6", "feature-auth")
        assert result == "a1b2c3d4e5f6.feature-auth"

    def test_generate_full_tag_without_sub_tag(self):
        """Should return main_tag alone when no sub_tag."""
        from workspace_qdrant_mcp.server import generate_full_tag

        result = generate_full_tag("a1b2c3d4e5f6", None)
        assert result == "a1b2c3d4e5f6"

    def test_generate_full_tag_empty_sub_tag(self):
        """Should return main_tag alone when sub_tag is empty string."""
        from workspace_qdrant_mcp.server import generate_full_tag

        result = generate_full_tag("numpy", "")
        assert result == "numpy"

    def test_generate_full_tag_library_with_version(self):
        """Should generate library.version format."""
        from workspace_qdrant_mcp.server import generate_full_tag

        result = generate_full_tag("numpy", "1.24.0")
        assert result == "numpy.1.24.0"


class TestParseTag:
    """Tests for parse_tag function."""

    def test_parse_tag_with_dot_separator(self):
        """Should split on first dot."""
        from workspace_qdrant_mcp.server import parse_tag

        main, sub = parse_tag("myproject.feature-auth")
        assert main == "myproject"
        assert sub == "feature-auth"

    def test_parse_tag_no_separator(self):
        """Should return tag as main_tag with None sub_tag."""
        from workspace_qdrant_mcp.server import parse_tag

        main, sub = parse_tag("myproject")
        assert main == "myproject"
        assert sub is None

    def test_parse_tag_multiple_dots_library_version(self):
        """Should handle multiple dots (library versions)."""
        from workspace_qdrant_mcp.server import parse_tag

        main, sub = parse_tag("numpy.1.24.0")
        assert main == "numpy"
        assert sub == "1.24.0"

    def test_parse_tag_branch_with_slashes(self):
        """Should handle branch names with slashes."""
        from workspace_qdrant_mcp.server import parse_tag

        main, sub = parse_tag("myproject.feature/my-feature")
        assert main == "myproject"
        assert sub == "feature/my-feature"


class TestValidateTag:
    """Tests for validate_tag function."""

    def test_validate_tag_valid_simple(self):
        """Should accept simple alphanumeric tag."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("myproject")
        assert is_valid is True
        assert error is None

    def test_validate_tag_valid_with_hyphen(self):
        """Should accept tag with hyphens."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("my-project")
        assert is_valid is True
        assert error is None

    def test_validate_tag_valid_with_underscore(self):
        """Should accept tag with underscores."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("my_project")
        assert is_valid is True
        assert error is None

    def test_validate_tag_valid_full_tag(self):
        """Should accept valid full tag with dot."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("myproject.feature-auth")
        assert is_valid is True
        assert error is None

    def test_validate_tag_valid_library_version(self):
        """Should accept library version format."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("numpy.1.24.0")
        assert is_valid is True
        assert error is None

    def test_validate_tag_empty(self):
        """Should reject empty tag."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_tag_starts_with_dot(self):
        """Should reject tag starting with dot."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag(".myproject")
        assert is_valid is False
        assert "start with" in error.lower()

    def test_validate_tag_ends_with_dot(self):
        """Should reject tag ending with dot."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("myproject.")
        assert is_valid is False
        assert "end with" in error.lower()

    def test_validate_tag_consecutive_dots(self):
        """Should reject tag with consecutive dots."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("myproject..branch")
        assert is_valid is False
        assert "consecutive" in error.lower()

    def test_validate_tag_invalid_main_tag_chars(self):
        """Should reject invalid characters in main_tag."""
        from workspace_qdrant_mcp.server import validate_tag

        is_valid, error = validate_tag("my@project.branch")
        assert is_valid is False
        assert "main_tag" in error.lower()


class TestMatchesTagPrefix:
    """Tests for matches_tag_prefix function."""

    def test_matches_tag_prefix_exact_match(self):
        """Should match when tags are equal."""
        from workspace_qdrant_mcp.server import matches_tag_prefix

        assert matches_tag_prefix("myproject", "myproject") is True

    def test_matches_tag_prefix_prefix_match(self):
        """Should match when full_tag starts with prefix."""
        from workspace_qdrant_mcp.server import matches_tag_prefix

        assert matches_tag_prefix("myproject.feature", "myproject") is True

    def test_matches_tag_prefix_full_tag_match(self):
        """Should match when prefix equals full_tag."""
        from workspace_qdrant_mcp.server import matches_tag_prefix

        assert matches_tag_prefix("myproject.feature", "myproject.feature") is True

    def test_matches_tag_prefix_no_match_different(self):
        """Should not match different tags."""
        from workspace_qdrant_mcp.server import matches_tag_prefix

        assert matches_tag_prefix("otherproject.feature", "myproject") is False

    def test_matches_tag_prefix_no_match_partial_word(self):
        """Should not match if prefix is partial word (not at dot boundary)."""
        from workspace_qdrant_mcp.server import matches_tag_prefix

        # "myproj" is not a valid prefix of "myproject.feature"
        # because it doesn't end at a dot boundary
        assert matches_tag_prefix("myproject.feature", "myproj") is False

    def test_matches_tag_prefix_library_version(self):
        """Should match library name as prefix for versioned library."""
        from workspace_qdrant_mcp.server import matches_tag_prefix

        assert matches_tag_prefix("numpy.1.24.0", "numpy") is True


class TestBuildMetadataFiltersWithTag:
    """Tests for tag filtering in build_metadata_filters."""

    def test_build_filters_with_main_tag(self):
        """Should filter by main_tag when tag has no dot."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        result = build_metadata_filters(tag="myproject", branch="*")
        assert result is not None
        # Should have a condition for main_tag
        conditions = result.must
        main_tag_conditions = [c for c in conditions if c.key == "main_tag"]
        assert len(main_tag_conditions) == 1
        assert main_tag_conditions[0].match.value == "myproject"

    def test_build_filters_with_full_tag(self):
        """Should filter by full_tag when tag has dot."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        result = build_metadata_filters(tag="myproject.feature", branch="*")
        assert result is not None
        # Should have a condition for full_tag
        conditions = result.must
        full_tag_conditions = [c for c in conditions if c.key == "full_tag"]
        assert len(full_tag_conditions) == 1
        assert full_tag_conditions[0].match.value == "myproject.feature"

    def test_build_filters_no_tag(self):
        """Should not add tag conditions when tag is None."""
        from workspace_qdrant_mcp.server import build_metadata_filters

        result = build_metadata_filters(tag=None, branch="*")
        # Result should be None when no filters (branch="*" means no branch filter)
        assert result is None


class TestStoreWithTagHierarchy:
    """Integration tests for store() with tag hierarchy."""

    @pytest.mark.asyncio
    async def test_store_generates_tags_in_metadata(self):
        """Store should generate main_tag and full_tag in metadata."""
        # This would require mocking daemon_client, but we can at least
        # verify the tag generation logic is consistent
        from workspace_qdrant_mcp.server import generate_main_tag, generate_full_tag

        project_id = "a1b2c3d4e5f6"
        branch = "feature-auth"

        main_tag = generate_main_tag(project_id=project_id)
        full_tag = generate_full_tag(main_tag, branch)

        assert main_tag == "a1b2c3d4e5f6"
        assert full_tag == "a1b2c3d4e5f6.feature-auth"
