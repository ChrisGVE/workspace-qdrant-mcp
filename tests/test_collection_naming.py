"""
Tests for the collection naming system.

This module tests the comprehensive collection naming architecture including
validation, conflict prevention, display name mapping, and MCP permission enforcement.
"""

from unittest.mock import Mock

import pytest

from workspace_qdrant_mcp.core.collection_naming import (
    CollectionNamingManager,
    CollectionPermissionError,
    CollectionType,
    NamingValidationResult,
)


class TestCollectionNamingManager:
    """Test the CollectionNamingManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.naming_manager = CollectionNamingManager(
            global_collections=["docs", "references", "standards"]
        )

    def test_memory_collection_validation(self):
        """Test memory collection naming validation."""
        # Valid memory collection
        result = self.naming_manager.validate_collection_name("memory")
        assert result.is_valid
        assert result.collection_info.collection_type == CollectionType.MEMORY
        assert not result.collection_info.is_readonly_from_mcp

        # Other names starting with memory are allowed (they become legacy collections)
        result = self.naming_manager.validate_collection_name("memory2")
        assert result.is_valid  # This should be allowed as a legacy collection
        assert result.collection_info.collection_type == CollectionType.LEGACY

    def test_library_collection_validation(self):
        """Test library collection naming validation."""
        # Valid library collections
        test_cases = [
            "_mylib",
            "_auth-utils",
            "_data_processing",
            "_core"
        ]

        for name in test_cases:
            result = self.naming_manager.validate_collection_name(name)
            assert result.is_valid, f"Failed for {name}: {result.error_message}"
            assert result.collection_info.collection_type == CollectionType.LIBRARY
            assert result.collection_info.is_readonly_from_mcp
            assert result.collection_info.display_name == name[1:]  # Without underscore

        # Invalid library collections
        invalid_cases = [
            "_",           # Empty after underscore
            "_123",        # Starts with number
            "_test-",      # Ends with hyphen
            "_test_",      # Ends with underscore
            "_Test",       # Contains uppercase
            "_test.lib",   # Contains period
        ]

        for name in invalid_cases:
            result = self.naming_manager.validate_collection_name(name)
            assert not result.is_valid, f"Should be invalid: {name}"

    def test_project_collection_validation(self):
        """Test project collection naming validation."""
        # Valid project collections
        valid_cases = [
            "my-project-docs",
            "my-project-scratchbook",
            "frontend-docs",
            "api-gateway-scratchbook"
        ]

        for name in valid_cases:
            result = self.naming_manager.validate_collection_name(name)
            assert result.is_valid, f"Failed for {name}: {result.error_message}"
            assert result.collection_info.collection_type == CollectionType.PROJECT
            assert not result.collection_info.is_readonly_from_mcp

        # Invalid project collections
        invalid_cases = [
            "my-project-code",       # Invalid suffix
            "my-project-invalid",    # Invalid suffix
            "project-",             # Missing suffix
            "-docs",                # Missing project name
        ]

        for name in invalid_cases:
            result = self.naming_manager.validate_collection_name(name)
            assert not result.is_valid, f"Should be invalid: {name}"

    def test_reserved_names_validation(self):
        """Test reserved name validation."""
        reserved_names = [
            "memory",
            "_memory",
            "system",
            "_system",
            "admin",
            "_admin"
        ]

        for name in reserved_names[1:]:  # Skip 'memory' since it's valid
            result = self.naming_manager.validate_collection_name(name)
            if name != "memory":
                assert not result.is_valid, f"Should be invalid: {name}"

    def test_naming_conflicts(self):
        """Test naming conflict detection."""
        existing_collections = ["docs", "_mylib", "project-docs"]

        # Test library/display conflicts
        result = self.naming_manager.check_naming_conflicts("mylib", existing_collections)
        assert not result.is_valid
        assert "library collection '_mylib' already exists" in result.error_message

        result = self.naming_manager.check_naming_conflicts("_docs", existing_collections)
        assert not result.is_valid
        assert "collection 'docs' already exists" in result.error_message

        # Test direct conflicts
        result = self.naming_manager.check_naming_conflicts("docs", existing_collections)
        assert not result.is_valid
        assert "already exists" in result.error_message

        # Test valid new collection
        result = self.naming_manager.check_naming_conflicts("newlib", existing_collections)
        assert result.is_valid

    def test_display_name_mapping(self):
        """Test display name mapping for user interfaces."""
        test_cases = [
            ("_mylib", "mylib"),           # Library collection
            ("memory", "memory"),           # Memory collection
            ("project-docs", "project-docs"),  # Project collection
            ("docs", "docs"),              # Legacy global collection
        ]

        for actual_name, expected_display in test_cases:
            display_name = self.naming_manager.get_display_name(actual_name)
            assert display_name == expected_display

    def test_actual_name_resolution(self):
        """Test resolving display names to actual collection names."""
        test_cases = [
            ("mylib", CollectionType.LIBRARY, "_mylib"),
            ("memory", CollectionType.MEMORY, "memory"),
            ("project-docs", CollectionType.PROJECT, "project-docs"),
        ]

        for display_name, collection_type, expected_actual in test_cases:
            actual_name = self.naming_manager.get_actual_name(display_name, collection_type)
            assert actual_name == expected_actual

    def test_mcp_readonly_check(self):
        """Test MCP readonly permission checking."""
        test_cases = [
            ("_mylib", True),          # Library collection - readonly
            ("memory", False),          # Memory collection - read/write
            ("project-docs", False),    # Project collection - read/write
            ("docs", False),           # Legacy global - read/write
        ]

        for collection_name, expected_readonly in test_cases:
            is_readonly = self.naming_manager.is_mcp_readonly(collection_name)
            assert is_readonly == expected_readonly

    def test_workspace_collection_filtering(self):
        """Test filtering workspace collections from all collections."""
        all_collections = [
            "memory",               # Include - memory collection
            "_mylib",              # Include - library collection
            "project-docs",        # Include - project collection
            "project-scratchbook", # Include - project collection
            "docs",                # Include - legacy global collection
            "memexd-project-code", # Exclude - daemon collection
            "unknown-collection",  # Exclude - doesn't match patterns
        ]

        workspace_collections = self.naming_manager.filter_workspace_collections(all_collections)
        expected = ["docs", "memory", "_mylib", "project-docs", "project-scratchbook"]
        assert sorted(workspace_collections) == sorted(expected)

    def test_project_collection_generation(self):
        """Test generating collection names for projects."""
        project_name = "my-project"
        collections = self.naming_manager.generate_project_collection_names(project_name)
        expected = ["my-project-scratchbook", "my-project-docs"]
        assert sorted(collections) == sorted(expected)

    def test_collection_classification(self):
        """Test collection name classification."""
        test_cases = [
            ("memory", CollectionType.MEMORY),
            ("_mylib", CollectionType.LIBRARY),
            ("project-docs", CollectionType.PROJECT),
            ("random-name", CollectionType.LEGACY),
        ]

        for name, expected_type in test_cases:
            info = self.naming_manager.get_collection_info(name)
            assert info.collection_type == expected_type
            assert info.name == name

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty name
        result = self.naming_manager.validate_collection_name("")
        assert not result.is_valid
        assert "cannot be empty" in result.error_message

        # Too long name
        long_name = "x" * 101
        result = self.naming_manager.validate_collection_name(long_name)
        assert not result.is_valid
        assert "cannot exceed 100 characters" in result.error_message

        # Whitespace handling
        result = self.naming_manager.validate_collection_name("  ")
        assert not result.is_valid


class TestCollectionPermissionError:
    """Test the CollectionPermissionError exception."""

    def test_permission_error_creation(self):
        """Test creating permission errors."""
        error = CollectionPermissionError("Library collection is readonly")
        assert str(error) == "Library collection is readonly"
        assert isinstance(error, Exception)


def test_create_naming_manager_factory():
    """Test the factory function for creating naming managers."""
    global_collections = ["custom1", "custom2"]
    manager = CollectionNamingManager(global_collections)
    assert manager.global_collections == set(global_collections)
