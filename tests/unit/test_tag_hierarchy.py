"""
Unit tests for Task 431: Dot-separated tag hierarchy functionality.
Tests tag indexing, filtering, and hierarchy management.
"""

import pytest
import tempfile
from pathlib import Path

from common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def state_manager():
    """Create a test SQLiteStateManager with temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_state.db"
        manager = SQLiteStateManager(db_path=str(db_path))
        await manager.initialize()
        yield manager
        await manager.close()


class TestTagIndexing:
    """Tests for tag index operations."""

    @pytest.mark.asyncio
    async def test_index_tag_simple(self, state_manager):
        """Test indexing a simple tag (parent_tag and depth are auto-computed)."""
        await state_manager.index_tag(
            tag_value="myproject.main",
            collection="_projects",
            tag_type="project"
        )

        # Verify tag was indexed
        tag = await state_manager.get_tag("myproject.main", "_projects")
        assert tag is not None
        assert tag["tag_value"] == "myproject.main"
        assert tag["collection"] == "_projects"
        assert tag["tag_type"] == "project"
        # parent_tag and depth are auto-computed by the method
        assert tag["parent_tag"] == "myproject"  # auto-computed
        assert tag["depth"] == 2  # auto-computed
        assert tag["watched"] is False
        assert tag["document_count"] == 0

    @pytest.mark.asyncio
    async def test_index_tag_root_level(self, state_manager):
        """Test indexing a root-level tag (no parent)."""
        await state_manager.index_tag(
            tag_value="myproject",
            collection="_projects",
            tag_type="project"
        )

        tag = await state_manager.get_tag("myproject", "_projects")
        assert tag is not None
        assert tag["parent_tag"] is None  # No parent for root tag
        assert tag["depth"] == 1

    @pytest.mark.asyncio
    async def test_index_tag_library_hierarchy(self, state_manager):
        """Test indexing library tags with folder hierarchy."""
        # Index library tags at different depths - parent_tag auto-computed
        await state_manager.index_tag("numpy", "_libraries", "library")
        await state_manager.index_tag("numpy.linalg", "_libraries", "library")
        await state_manager.index_tag("numpy.linalg.norm", "_libraries", "library")

        # Verify all tags (plus auto-created parent tags)
        tags = await state_manager.list_tags(collection="_libraries")
        tag_values = [t["tag_value"] for t in tags]
        assert "numpy" in tag_values
        assert "numpy.linalg" in tag_values
        assert "numpy.linalg.norm" in tag_values

        # Verify hierarchy
        deep_tag = await state_manager.get_tag("numpy.linalg.norm", "_libraries")
        assert deep_tag["parent_tag"] == "numpy.linalg"
        assert deep_tag["depth"] == 3

    @pytest.mark.asyncio
    async def test_increment_document_count(self, state_manager):
        """Test incrementing document count for a tag."""
        await state_manager.index_tag(
            tag_value="myproject.main",
            collection="_projects",
            tag_type="project"
        )

        # Increment count
        await state_manager.increment_tag_document_count("myproject.main", "_projects")
        await state_manager.increment_tag_document_count("myproject.main", "_projects")

        tag = await state_manager.get_tag("myproject.main", "_projects")
        assert tag["document_count"] == 2


class TestTagFiltering:
    """Tests for tag filtering operations."""

    @pytest.mark.asyncio
    async def test_list_tags_by_collection(self, state_manager):
        """Test listing tags filtered by collection."""
        # Index tags in different collections
        await state_manager.index_tag("proj1.main", "_projects", "project")
        await state_manager.index_tag("lib1.folder", "_libraries", "library")

        # Filter by collection (note: parent tags are auto-created)
        project_tags = await state_manager.list_tags(collection="_projects")
        # Should have proj1 (auto-created parent) and proj1.main
        assert len(project_tags) == 2
        tag_values = [t["tag_value"] for t in project_tags]
        assert "proj1.main" in tag_values
        assert "proj1" in tag_values  # auto-created parent

        library_tags = await state_manager.list_tags(collection="_libraries")
        # Should have lib1 (auto-created parent) and lib1.folder
        assert len(library_tags) == 2
        tag_values = [t["tag_value"] for t in library_tags]
        assert "lib1.folder" in tag_values

    @pytest.mark.asyncio
    async def test_list_tags_by_type(self, state_manager):
        """Test listing tags filtered by tag_type."""
        await state_manager.index_tag("proj1.main", "_projects", "project")
        await state_manager.index_tag("lib1.folder", "_libraries", "library")
        await state_manager.index_tag("mem1", "_memory", "memory")

        project_tags = await state_manager.list_tags(tag_type="project")
        # All project tags have tag_type="project"
        for tag in project_tags:
            assert tag["tag_type"] == "project"

        memory_tags = await state_manager.list_tags(tag_type="memory")
        assert len(memory_tags) == 1
        assert memory_tags[0]["tag_type"] == "memory"

    @pytest.mark.asyncio
    async def test_list_tags_by_parent(self, state_manager):
        """Test listing tags filtered by parent tag."""
        await state_manager.index_tag("numpy", "_libraries", "library")
        await state_manager.index_tag("numpy.linalg", "_libraries", "library")
        await state_manager.index_tag("numpy.random", "_libraries", "library")
        await state_manager.index_tag("pandas", "_libraries", "library")

        # Get children of numpy
        numpy_children = await state_manager.list_tags(parent_tag="numpy")
        assert len(numpy_children) == 2
        for tag in numpy_children:
            assert tag["parent_tag"] == "numpy"

    @pytest.mark.asyncio
    async def test_list_tags_with_hierarchy(self, state_manager):
        """Test listing tags with hierarchy inclusion."""
        await state_manager.index_tag("numpy", "_libraries", "library")
        await state_manager.index_tag("numpy.linalg", "_libraries", "library")
        await state_manager.index_tag("numpy.linalg.norm", "_libraries", "library")

        # Get hierarchy starting from numpy.linalg
        tags = await state_manager.list_tags(
            parent_tag="numpy.linalg",
            include_hierarchy=True
        )
        # Should include numpy.linalg.norm (child of numpy.linalg)
        tag_values = [t["tag_value"] for t in tags]
        assert "numpy.linalg.norm" in tag_values


class TestWatchedTags:
    """Tests for watched tag functionality."""

    @pytest.mark.asyncio
    async def test_set_tag_watched(self, state_manager):
        """Test marking a tag as watched."""
        await state_manager.index_tag("myproject.main", "_projects", "project")

        # Mark as watched
        await state_manager.set_tag_watched("myproject.main", "_projects", True)

        tag = await state_manager.get_tag("myproject.main", "_projects")
        assert tag["watched"] is True

        # Unmark
        await state_manager.set_tag_watched("myproject.main", "_projects", False)
        tag = await state_manager.get_tag("myproject.main", "_projects")
        assert tag["watched"] is False

    @pytest.mark.asyncio
    async def test_get_watched_tags(self, state_manager):
        """Test retrieving watched tags."""
        await state_manager.index_tag("proj1.main", "_projects", "project")
        await state_manager.index_tag("proj1.dev", "_projects", "project")
        await state_manager.index_tag("proj2.main", "_projects", "project")

        # Mark some as watched
        await state_manager.set_tag_watched("proj1.main", "_projects", True)
        await state_manager.set_tag_watched("proj2.main", "_projects", True)

        watched = await state_manager.get_watched_tags()
        assert len(watched) == 2
        watched_values = [t["tag_value"] for t in watched]
        assert "proj1.main" in watched_values
        assert "proj2.main" in watched_values
        assert "proj1.dev" not in watched_values

    @pytest.mark.asyncio
    async def test_list_tags_watched_only(self, state_manager):
        """Test listing only watched tags."""
        await state_manager.index_tag("proj1.main", "_projects", "project")
        await state_manager.index_tag("proj1.dev", "_projects", "project")

        await state_manager.set_tag_watched("proj1.main", "_projects", True)

        tags = await state_manager.list_tags(watched_only=True)
        assert len(tags) == 1
        assert tags[0]["tag_value"] == "proj1.main"


class TestTagDeletion:
    """Tests for tag deletion operations."""

    @pytest.mark.asyncio
    async def test_delete_tag(self, state_manager):
        """Test deleting a tag."""
        await state_manager.index_tag("myproject.main", "_projects", "project")

        # Verify exists
        tag = await state_manager.get_tag("myproject.main", "_projects")
        assert tag is not None

        # Delete
        await state_manager.delete_tag("myproject.main", "_projects")

        # Verify gone
        tag = await state_manager.get_tag("myproject.main", "_projects")
        assert tag is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_tag(self, state_manager):
        """Test deleting a tag that doesn't exist (should not error)."""
        # Should not raise
        await state_manager.delete_tag("nonexistent.tag", "_projects")


class TestTagAutoParentCreation:
    """Tests for automatic parent tag creation."""

    @pytest.mark.asyncio
    async def test_deep_tag_creates_parents(self, state_manager):
        """Test that indexing a deep tag auto-creates parent tags."""
        # Index a deep tag
        await state_manager.index_tag("a.b.c.d", "_libraries", "library")

        # Verify all parent tags were created
        tags = await state_manager.list_tags(collection="_libraries")
        tag_values = [t["tag_value"] for t in tags]

        # Should have a, a.b, a.b.c, a.b.c.d
        assert "a" in tag_values
        assert "a.b" in tag_values
        assert "a.b.c" in tag_values
        assert "a.b.c.d" in tag_values

        # Verify depths
        a = await state_manager.get_tag("a", "_libraries")
        assert a["depth"] == 1
        assert a["parent_tag"] is None

        ab = await state_manager.get_tag("a.b", "_libraries")
        assert ab["depth"] == 2
        assert ab["parent_tag"] == "a"

        abcd = await state_manager.get_tag("a.b.c.d", "_libraries")
        assert abcd["depth"] == 4
        assert abcd["parent_tag"] == "a.b.c"
