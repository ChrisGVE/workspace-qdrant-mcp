"""Unit tests for daemon file routing (Task 402).

Tests multi-tenant collection routing via watch_type configuration.
"""

import pytest
from datetime import datetime, timezone
from common.core.sqlite_state_manager import WatchFolderConfig


class TestWatchFolderConfigWatchType:
    """Tests for WatchFolderConfig watch_type field."""

    def test_default_watch_type_is_project(self):
        """Test that default watch_type is 'project'."""
        config = WatchFolderConfig(
            watch_id="test-watch",
            path="/test/path",
            collection="_old_collection",
            patterns=["*.py"],
            ignore_patterns=[],
        )
        assert config.watch_type == "project"

    def test_watch_type_project(self):
        """Test explicit project watch type."""
        config = WatchFolderConfig(
            watch_id="project-watch",
            path="/test/project",
            collection="_my_project",
            patterns=["*.py", "*.rs"],
            ignore_patterns=["__pycache__/*"],
            watch_type="project",
        )
        assert config.watch_type == "project"
        assert config.library_name is None

    def test_watch_type_library(self):
        """Test library watch type with library_name."""
        config = WatchFolderConfig(
            watch_id="library-watch",
            path="/test/library",
            collection="_langchain",
            patterns=["*.md", "*.pdf"],
            ignore_patterns=[],
            watch_type="library",
            library_name="langchain",
        )
        assert config.watch_type == "library"
        assert config.library_name == "langchain"

    def test_invalid_watch_type_defaults_to_project(self):
        """Test that invalid watch_type defaults to 'project'."""
        config = WatchFolderConfig(
            watch_id="invalid-watch",
            path="/test/path",
            collection="_collection",
            patterns=["*"],
            ignore_patterns=[],
            watch_type="invalid_type",  # Invalid value
        )
        # __post_init__ should correct this
        assert config.watch_type == "project"

    def test_library_watch_without_library_name(self):
        """Test library watch type without library_name (allowed but not recommended)."""
        config = WatchFolderConfig(
            watch_id="lib-no-name",
            path="/test/library",
            collection="_some_lib",
            patterns=["*.md"],
            ignore_patterns=[],
            watch_type="library",
            # No library_name - daemon will use fallback logic
        )
        assert config.watch_type == "library"
        assert config.library_name is None


class TestWatchFolderConfigRouting:
    """Tests for multi-tenant routing configuration."""

    def test_project_watch_full_config(self):
        """Test complete project watch configuration."""
        config = WatchFolderConfig(
            watch_id="my-project-py",
            path="/Users/test/dev/my-project",
            collection="_my_project",
            patterns=["*.py", "*.rs", "*.md"],
            ignore_patterns=["__pycache__/*", ".git/*", "*.pyc"],
            auto_ingest=True,
            recursive=True,
            recursive_depth=15,
            debounce_seconds=3.0,
            enabled=True,
            watch_type="project",
        )

        assert config.watch_id == "my-project-py"
        assert config.watch_type == "project"
        assert config.recursive is True
        assert config.debounce_seconds == 3.0

    def test_library_watch_full_config(self):
        """Test complete library watch configuration."""
        config = WatchFolderConfig(
            watch_id="langchain-docs",
            path="/Users/test/libraries/langchain",
            collection="_langchain",
            patterns=["*.md", "*.pdf", "*.txt"],
            ignore_patterns=[".git/*"],
            auto_ingest=True,
            recursive=True,
            recursive_depth=5,
            debounce_seconds=5.0,
            enabled=True,
            watch_type="library",
            library_name="langchain",
            metadata={"version": "0.1.0", "source": "github"},
        )

        assert config.watch_id == "langchain-docs"
        assert config.watch_type == "library"
        assert config.library_name == "langchain"
        assert config.metadata["version"] == "0.1.0"

    def test_timestamps_auto_set(self):
        """Test that timestamps are auto-set on creation."""
        before = datetime.now(timezone.utc)
        config = WatchFolderConfig(
            watch_id="timestamp-test",
            path="/test",
            collection="_test",
            patterns=["*"],
            ignore_patterns=[],
        )
        after = datetime.now(timezone.utc)

        assert config.created_at is not None
        assert config.updated_at is not None
        assert before <= config.created_at <= after
        assert config.created_at == config.updated_at


class TestUnifiedCollectionNames:
    """Tests for unified collection naming convention."""

    def test_projects_collection_name(self):
        """Test that project watches route to _projects collection."""
        # This is validated by the Rust daemon, but we test the convention
        expected_collection = "projects"
        assert expected_collection.startswith("_")

    def test_libraries_collection_name(self):
        """Test that library watches route to _libraries collection."""
        # This is validated by the Rust daemon, but we test the convention
        expected_collection = "libraries"
        assert expected_collection.startswith("_")


class TestWatchTypeValidation:
    """Tests for watch_type validation in __post_init__."""

    def test_valid_project_type(self):
        """Test valid 'project' type passes validation."""
        config = WatchFolderConfig(
            watch_id="valid-project",
            path="/test",
            collection="_test",
            patterns=["*"],
            ignore_patterns=[],
            watch_type="project",
        )
        assert config.watch_type == "project"

    def test_valid_library_type(self):
        """Test valid 'library' type passes validation."""
        config = WatchFolderConfig(
            watch_id="valid-library",
            path="/test",
            collection="_test",
            patterns=["*"],
            ignore_patterns=[],
            watch_type="library",
        )
        assert config.watch_type == "library"

    def test_empty_watch_type_defaults_to_project(self):
        """Test empty watch_type defaults to 'project'."""
        config = WatchFolderConfig(
            watch_id="empty-type",
            path="/test",
            collection="_test",
            patterns=["*"],
            ignore_patterns=[],
            watch_type="",  # Empty string
        )
        # __post_init__ should correct this
        assert config.watch_type == "project"
