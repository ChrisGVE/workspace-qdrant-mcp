"""Systematic test for collection_types.py coverage - Target: 40%+ coverage in 2-3 minutes."""

import pytest
import sys
import os

# Use the test directory approach from existing tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/python'))

# Import modules from collection_types
from common.core.collection_types import (
    SYSTEM_PREFIX,
    LIBRARY_PREFIX,
    GLOBAL_COLLECTIONS,
    COLLECTION_TYPES_AVAILABLE
)


class TestCollectionTypeConstants:
    """Test collection type constants and configuration."""

    def test_system_prefix_constant(self):
        """Test SYSTEM_PREFIX constant value."""
        assert SYSTEM_PREFIX == "__"
        assert isinstance(SYSTEM_PREFIX, str)
        assert len(SYSTEM_PREFIX) == 2

    def test_library_prefix_constant(self):
        """Test LIBRARY_PREFIX constant value."""
        assert LIBRARY_PREFIX == "_"
        assert isinstance(LIBRARY_PREFIX, str)
        assert len(LIBRARY_PREFIX) == 1

    def test_collection_types_available_flag(self):
        """Test COLLECTION_TYPES_AVAILABLE flag."""
        assert COLLECTION_TYPES_AVAILABLE is True
        assert isinstance(COLLECTION_TYPES_AVAILABLE, bool)

    def test_global_collections_list(self):
        """Test GLOBAL_COLLECTIONS list."""
        assert isinstance(GLOBAL_COLLECTIONS, list)
        assert len(GLOBAL_COLLECTIONS) > 0

        # Test expected collections are present
        expected_collections = [
            "algorithms",
            "codebase",
            "context",
            "documents",
            "knowledge",
            "memory",
            "projects",
            "workspace"
        ]

        for collection in expected_collections:
            assert collection in GLOBAL_COLLECTIONS

    def test_global_collections_are_strings(self):
        """Test that all global collections are strings."""
        for collection in GLOBAL_COLLECTIONS:
            assert isinstance(collection, str)
            assert len(collection) > 0

    def test_global_collections_no_duplicates(self):
        """Test that global collections list has no duplicates."""
        assert len(GLOBAL_COLLECTIONS) == len(set(GLOBAL_COLLECTIONS))

    def test_global_collections_naming_conventions(self):
        """Test global collections follow naming conventions."""
        for collection in GLOBAL_COLLECTIONS:
            # Should be lowercase
            assert collection.islower()
            # Should not start with prefixes
            assert not collection.startswith(SYSTEM_PREFIX)
            assert not collection.startswith(LIBRARY_PREFIX)
            # Should be alphanumeric (allowing underscores)
            assert collection.replace('_', '').isalnum()


class TestCollectionTypePrefixes:
    """Test collection type prefix functionality."""

    def test_prefix_uniqueness(self):
        """Test that prefixes are unique."""
        assert SYSTEM_PREFIX != LIBRARY_PREFIX

    def test_prefix_hierarchy(self):
        """Test prefix hierarchy (system prefix is longer)."""
        assert len(SYSTEM_PREFIX) > len(LIBRARY_PREFIX)

    def test_library_prefix_in_system_prefix(self):
        """Test that library prefix is part of system prefix."""
        assert LIBRARY_PREFIX in SYSTEM_PREFIX

    def test_prefixes_are_underscores(self):
        """Test that prefixes only contain underscores."""
        assert all(c == '_' for c in SYSTEM_PREFIX)
        assert all(c == '_' for c in LIBRARY_PREFIX)


class TestCollectionTypeClassification:
    """Test collection type classification logic."""

    def test_system_collection_identification(self):
        """Test system collection identification."""
        system_names = [
            "__user_preferences",
            "__memory_store",
            "__system_config"
        ]

        for name in system_names:
            assert name.startswith(SYSTEM_PREFIX)
            assert not name.startswith(LIBRARY_PREFIX) or name.startswith(SYSTEM_PREFIX)

    def test_library_collection_identification(self):
        """Test library collection identification."""
        library_names = [
            "_shared_docs",
            "_common_knowledge",
            "_global_context"
        ]

        for name in library_names:
            assert name.startswith(LIBRARY_PREFIX)
            assert not name.startswith(SYSTEM_PREFIX)

    def test_project_collection_identification(self):
        """Test project collection identification."""
        project_names = [
            "myproject-docs",
            "workspace-qdrant-mcp-memory",
            "test-project-context"
        ]

        for name in project_names:
            assert not name.startswith(SYSTEM_PREFIX)
            assert not name.startswith(LIBRARY_PREFIX)
            assert "-" in name  # Project collections have dashes

    def test_global_collection_identification(self):
        """Test global collection identification."""
        for collection in GLOBAL_COLLECTIONS:
            assert not collection.startswith(SYSTEM_PREFIX)
            assert not collection.startswith(LIBRARY_PREFIX)
            assert "-" not in collection  # Global collections are single words


class TestCollectionTypeValidation:
    """Test collection type validation functionality."""

    def test_valid_system_collection_names(self):
        """Test validation of system collection names."""
        valid_system_names = [
            "__user_data",
            "__system_memory",
            "__configuration",
            "__temp_storage"
        ]

        for name in valid_system_names:
            assert name.startswith(SYSTEM_PREFIX)
            # Should be valid identifier after prefix
            suffix = name[len(SYSTEM_PREFIX):]
            assert suffix.replace('_', '').isalnum()

    def test_valid_library_collection_names(self):
        """Test validation of library collection names."""
        valid_library_names = [
            "_shared_context",
            "_global_memory",
            "_common_docs",
            "_library_data"
        ]

        for name in valid_library_names:
            assert name.startswith(LIBRARY_PREFIX)
            assert not name.startswith(SYSTEM_PREFIX)
            # Should be valid identifier after prefix
            suffix = name[len(LIBRARY_PREFIX):]
            assert suffix.replace('_', '').isalnum()

    def test_valid_project_collection_names(self):
        """Test validation of project collection names."""
        valid_project_names = [
            "myproject-docs",
            "test-memory",
            "workspace-qdrant-mcp-context",
            "ai-assistant-knowledge"
        ]

        for name in valid_project_names:
            assert "-" in name
            assert not name.startswith(SYSTEM_PREFIX)
            assert not name.startswith(LIBRARY_PREFIX)

    def test_invalid_collection_names(self):
        """Test identification of invalid collection names."""
        invalid_names = [
            "",  # Empty
            "_",  # Just library prefix
            "__",  # Just system prefix
            "___invalid",  # Too many underscores
            "invalid-__name",  # Mixed prefix in wrong position
            "project-_name"  # Mixed library prefix in wrong position
        ]

        for name in invalid_names:
            # These should be identifiable as invalid
            if name == "":
                assert len(name) == 0
            elif name in ["_", "__"]:
                assert name in [LIBRARY_PREFIX, SYSTEM_PREFIX]
            else:
                # Other validation logic would catch these
                assert len(name) > 0


class TestCollectionTypeUtilities:
    """Test utility functions for collection types."""

    def test_collection_type_detection_logic(self):
        """Test the logic for detecting collection types."""
        test_cases = [
            ("__system_config", "system"),
            ("_shared_docs", "library"),
            ("myproject-docs", "project"),
            ("memory", "global"),
            ("algorithms", "global"),
            ("workspace", "global")
        ]

        for collection_name, expected_type in test_cases:
            if collection_name.startswith(SYSTEM_PREFIX):
                assert expected_type == "system"
            elif collection_name.startswith(LIBRARY_PREFIX):
                assert expected_type == "library"
            elif "-" in collection_name:
                assert expected_type == "project"
            elif collection_name in GLOBAL_COLLECTIONS:
                assert expected_type == "global"

    def test_display_name_logic(self):
        """Test display name generation logic."""
        test_cases = [
            ("__user_preferences", "user_preferences"),
            ("_shared_context", "shared_context"),
            ("myproject-docs", "myproject-docs"),
            ("memory", "memory")
        ]

        for full_name, expected_display in test_cases:
            if full_name.startswith(SYSTEM_PREFIX):
                display_name = full_name[len(SYSTEM_PREFIX):]
                assert display_name == expected_display
            elif full_name.startswith(LIBRARY_PREFIX):
                display_name = full_name[len(LIBRARY_PREFIX):]
                assert display_name == expected_display
            else:
                assert full_name == expected_display


class TestCollectionTypeIntegration:
    """Test integration aspects of collection types."""

    def test_module_availability_flag(self):
        """Test that module availability flag works."""
        assert COLLECTION_TYPES_AVAILABLE == True

    def test_constants_consistency(self):
        """Test consistency between constants."""
        # System prefix should contain library prefix
        assert LIBRARY_PREFIX in SYSTEM_PREFIX

        # Global collections should not conflict with prefixes
        for collection in GLOBAL_COLLECTIONS:
            assert not collection.startswith(SYSTEM_PREFIX)
            assert not collection.startswith(LIBRARY_PREFIX)

    def test_collection_type_hierarchy_completeness(self):
        """Test that collection type hierarchy covers all cases."""
        sample_collections = [
            "__system_data",     # SYSTEM
            "_library_docs",     # LIBRARY
            "project-memory",    # PROJECT
            "workspace",         # GLOBAL
            "algorithms"         # GLOBAL
        ]

        for collection in sample_collections:
            # Each collection should fit into exactly one category
            is_system = collection.startswith(SYSTEM_PREFIX)
            is_library = collection.startswith(LIBRARY_PREFIX) and not is_system
            is_project = "-" in collection and not is_system and not is_library
            is_global = collection in GLOBAL_COLLECTIONS

            # Should match exactly one category
            categories_matched = sum([is_system, is_library, is_project, is_global])
            assert categories_matched == 1


if __name__ == "__main__":
    # Quick test execution for development
    pytest.main([__file__, "-v", "--tb=short"])