"""
Comprehensive unit tests for the collection type hierarchy system.

This test suite provides complete coverage of the core.collection_types module,
testing all classes, functions, and edge cases with high precision and maintainability.

Test Categories:
- CollectionTypeClassifier methods
- Collection type classification for all types
- Display name mapping and prefix removal
- Collection type validation for operations
- Memory collection pattern detection
- Integration with collection_naming module
- Edge cases and error handling
"""

import pytest
from typing import List, Tuple, Optional
from unittest.mock import patch, MagicMock

# Import the module under test
try:
    from core.collection_types import (
        # Classes
        CollectionType,
        CollectionInfo,
        CollectionTypeClassifier,
        
        # Constants
        SYSTEM_PREFIX,
        LIBRARY_PREFIX,
        GLOBAL_COLLECTIONS,
        PROJECT_PATTERN,
        SYSTEM_MEMORY_PATTERN,
        PROJECT_MEMORY_PATTERN,
        
        # Utility functions
        validate_collection_operation,
        get_collections_by_type,
        get_searchable_collections,
        validate_collection_name_with_type,
        build_collection_name_for_type
    )
    from core import collection_naming
except ImportError:
    # For direct execution when not used as a package
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from core.collection_types import (
        CollectionType,
        CollectionInfo,
        CollectionTypeClassifier,
        SYSTEM_PREFIX,
        LIBRARY_PREFIX,
        GLOBAL_COLLECTIONS,
        PROJECT_PATTERN,
        SYSTEM_MEMORY_PATTERN,
        PROJECT_MEMORY_PATTERN,
        validate_collection_operation,
        get_collections_by_type,
        get_searchable_collections,
        validate_collection_name_with_type,
        build_collection_name_for_type
    )
    from core import collection_naming


class TestCollectionTypeConstants:
    """Test collection type constants and patterns."""
    
    def test_system_prefix_constant(self):
        """Test that SYSTEM_PREFIX is correct."""
        assert SYSTEM_PREFIX == "__"
    
    def test_library_prefix_constant(self):
        """Test that LIBRARY_PREFIX is correct."""
        assert LIBRARY_PREFIX == "_"
    
    def test_global_collections_list(self):
        """Test that GLOBAL_COLLECTIONS contains expected values."""
        expected_globals = [
            "algorithms",
            "codebase", 
            "context",
            "documents",
            "knowledge",
            "memory",
            "projects",
            "workspace"
        ]
        assert GLOBAL_COLLECTIONS == expected_globals
    
    def test_pattern_constants(self):
        """Test pattern constants are valid regex patterns."""
        import re
        
        # Test that patterns compile without errors
        assert re.compile(PROJECT_PATTERN)
        assert re.compile(SYSTEM_MEMORY_PATTERN)
        assert re.compile(PROJECT_MEMORY_PATTERN)


class TestCollectionTypeEnum:
    """Test CollectionType enumeration."""
    
    def test_collection_type_values(self):
        """Test that CollectionType has correct values."""
        assert CollectionType.SYSTEM.value == "system"
        assert CollectionType.LIBRARY.value == "library"
        assert CollectionType.PROJECT.value == "project"
        assert CollectionType.GLOBAL.value == "global"
        assert CollectionType.UNKNOWN.value == "unknown"
    
    def test_collection_type_members(self):
        """Test that all expected enum members exist."""
        expected_members = {"SYSTEM", "LIBRARY", "PROJECT", "GLOBAL", "UNKNOWN"}
        actual_members = {member.name for member in CollectionType}
        assert actual_members == expected_members


class TestCollectionInfo:
    """Test CollectionInfo dataclass."""
    
    def test_collection_info_creation(self):
        """Test creating CollectionInfo instances."""
        info = CollectionInfo(
            name="test_collection",
            type=CollectionType.PROJECT,
            display_name="test_collection",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False,
            project_name="test",
            suffix="collection"
        )
        
        assert info.name == "test_collection"
        assert info.type == CollectionType.PROJECT
        assert info.display_name == "test_collection"
        assert info.is_searchable is True
        assert info.is_readonly is False
        assert info.is_memory_collection is False
        assert info.project_name == "test"
        assert info.suffix == "collection"
    
    def test_collection_info_defaults(self):
        """Test CollectionInfo with default values."""
        info = CollectionInfo(
            name="test",
            type=CollectionType.GLOBAL,
            display_name="test",
            is_searchable=True,
            is_readonly=False,
            is_memory_collection=False
        )
        
        assert info.project_name is None
        assert info.suffix is None


class TestCollectionTypeClassifier:
    """Test CollectionTypeClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Fixture providing a CollectionTypeClassifier instance."""
        return CollectionTypeClassifier()
    
    # Classification Tests
    
    @pytest.mark.parametrize("collection_name,expected_type", [
        # System collections
        ("__user_preferences", CollectionType.SYSTEM),
        ("__system_config", CollectionType.SYSTEM),
        ("__memory_collection", CollectionType.SYSTEM),
        ("__a", CollectionType.SYSTEM),
        
        # Library collections  
        ("_library_docs", CollectionType.LIBRARY),
        ("_utils", CollectionType.LIBRARY),
        ("_test_data", CollectionType.LIBRARY),
        ("_a", CollectionType.LIBRARY),
        
        # Global collections
        ("algorithms", CollectionType.GLOBAL),
        ("codebase", CollectionType.GLOBAL),
        ("context", CollectionType.GLOBAL),
        ("documents", CollectionType.GLOBAL),
        ("knowledge", CollectionType.GLOBAL),
        ("memory", CollectionType.GLOBAL),
        ("projects", CollectionType.GLOBAL),
        ("workspace", CollectionType.GLOBAL),
        
        # Project collections
        ("project-documents", CollectionType.PROJECT),
        ("my_project-source_code", CollectionType.PROJECT),
        ("test-data", CollectionType.PROJECT),
        ("workspace-lsp_metadata", CollectionType.PROJECT),
        ("a-b", CollectionType.PROJECT),
        
        # Unknown collections
        ("invalid_name", CollectionType.UNKNOWN),
        ("no-dash-but-multiple-words", CollectionType.UNKNOWN),
        ("", CollectionType.UNKNOWN),
        ("project-", CollectionType.UNKNOWN),
        ("-suffix", CollectionType.UNKNOWN),
        ("project--double-dash", CollectionType.UNKNOWN),
    ])
    def test_classify_collection_type(self, classifier, collection_name, expected_type):
        """Test collection type classification for various names."""
        assert classifier.classify_collection_type(collection_name) == expected_type
    
    def test_classify_collection_type_invalid_input(self, classifier):
        """Test classification with invalid inputs."""
        assert classifier.classify_collection_type(None) == CollectionType.UNKNOWN
        assert classifier.classify_collection_type(123) == CollectionType.UNKNOWN
        assert classifier.classify_collection_type([]) == CollectionType.UNKNOWN
    
    # Type checking methods
    
    @pytest.mark.parametrize("collection_name,expected", [
        ("__system_config", True),
        ("__user_prefs", True),
        ("_library", False),
        ("project-docs", False),
        ("algorithms", False),
        ("", False),
    ])
    def test_is_system_collection(self, classifier, collection_name, expected):
        """Test system collection identification."""
        assert classifier.is_system_collection(collection_name) == expected
    
    @pytest.mark.parametrize("collection_name,expected", [
        ("_library_docs", True),
        ("_utils", True),
        ("__system", False),
        ("project-docs", False),
        ("algorithms", False),
        ("", False),
    ])
    def test_is_library_collection(self, classifier, collection_name, expected):
        """Test library collection identification."""
        assert classifier.is_library_collection(collection_name) == expected
    
    @pytest.mark.parametrize("collection_name,expected", [
        ("project-documents", True),
        ("my_project-source", True),
        ("__system", False),
        ("_library", False),
        ("algorithms", False),
        ("invalid", False),
        ("", False),
    ])
    def test_is_project_collection(self, classifier, collection_name, expected):
        """Test project collection identification."""
        assert classifier.is_project_collection(collection_name) == expected
    
    @pytest.mark.parametrize("collection_name,expected", [
        ("algorithms", True),
        ("codebase", True),
        ("workspace", True),
        ("__system", False),
        ("_library", False),
        ("project-docs", False),
        ("invalid", False),
        ("", False),
    ])
    def test_is_global_collection(self, classifier, collection_name, expected):
        """Test global collection identification."""
        assert classifier.is_global_collection(collection_name) == expected
    
    # Memory collection detection
    
    @pytest.mark.parametrize("collection_name,expected", [
        # System memory collections
        ("__memory_collection", True),
        ("__user_memory", True),
        ("__system_state", True),
        
        # Project memory collections
        ("project-memory", True),
        ("my_project-memory", True),
        ("workspace-memory", True),
        
        # Non-memory collections
        ("__system_config", False),
        ("_library_docs", False),
        ("project-documents", False),
        ("algorithms", False),
        ("memory", False),  # Global collection, not memory pattern
        ("project-memory-backup", False),  # Not exact memory pattern
        ("", False),
    ])
    def test_is_memory_collection(self, classifier, collection_name, expected):
        """Test memory collection pattern detection."""
        assert classifier.is_memory_collection(collection_name) == expected
    
    # Display name tests
    
    @pytest.mark.parametrize("collection_name,expected_display", [
        # System collections - remove __ prefix
        ("__user_preferences", "user_preferences"),
        ("__system_config", "system_config"),
        ("__a", "a"),
        
        # Library collections - remove _ prefix
        ("_library_docs", "library_docs"),
        ("_utils", "utils"),
        ("_a", "a"),
        
        # Other collections - no change
        ("project-documents", "project-documents"),
        ("algorithms", "algorithms"),
        ("workspace", "workspace"),
        ("invalid_name", "invalid_name"),
        ("", ""),
    ])
    def test_get_display_name(self, classifier, collection_name, expected_display):
        """Test display name generation with prefix removal."""
        assert classifier.get_display_name(collection_name) == expected_display
    
    # Comprehensive collection info tests
    
    def test_get_collection_info_system(self, classifier):
        """Test CollectionInfo for system collection."""
        info = classifier.get_collection_info("__user_preferences")
        
        assert info.name == "__user_preferences"
        assert info.type == CollectionType.SYSTEM
        assert info.display_name == "user_preferences"
        assert info.is_searchable is False  # Not globally searchable
        assert info.is_readonly is False    # CLI-writable
        assert info.is_memory_collection is False
        assert info.project_name is None
        assert info.suffix is None
    
    def test_get_collection_info_system_memory(self, classifier):
        """Test CollectionInfo for system memory collection."""
        info = classifier.get_collection_info("__memory_collection")
        
        assert info.name == "__memory_collection"
        assert info.type == CollectionType.SYSTEM
        assert info.display_name == "memory_collection"
        assert info.is_searchable is False
        assert info.is_readonly is False
        assert info.is_memory_collection is True
        assert info.project_name is None
        assert info.suffix is None
    
    def test_get_collection_info_library(self, classifier):
        """Test CollectionInfo for library collection."""
        info = classifier.get_collection_info("_library_docs")
        
        assert info.name == "_library_docs"
        assert info.type == CollectionType.LIBRARY
        assert info.display_name == "library_docs"
        assert info.is_searchable is True   # Globally searchable
        assert info.is_readonly is True     # MCP-readonly
        assert info.is_memory_collection is False
        assert info.project_name is None
        assert info.suffix is None
    
    def test_get_collection_info_project(self, classifier):
        """Test CollectionInfo for project collection."""
        info = classifier.get_collection_info("my_project-documents")
        
        assert info.name == "my_project-documents"
        assert info.type == CollectionType.PROJECT
        assert info.display_name == "my_project-documents"
        assert info.is_searchable is True
        assert info.is_readonly is False
        assert info.is_memory_collection is False
        assert info.project_name == "my_project"
        assert info.suffix == "documents"
    
    def test_get_collection_info_project_memory(self, classifier):
        """Test CollectionInfo for project memory collection."""
        info = classifier.get_collection_info("workspace-memory")
        
        assert info.name == "workspace-memory"
        assert info.type == CollectionType.PROJECT
        assert info.display_name == "workspace-memory"
        assert info.is_searchable is True
        assert info.is_readonly is False
        assert info.is_memory_collection is True
        assert info.project_name == "workspace"
        assert info.suffix == "memory"
    
    def test_get_collection_info_global(self, classifier):
        """Test CollectionInfo for global collection."""
        info = classifier.get_collection_info("algorithms")
        
        assert info.name == "algorithms"
        assert info.type == CollectionType.GLOBAL
        assert info.display_name == "algorithms"
        assert info.is_searchable is True
        assert info.is_readonly is False
        assert info.is_memory_collection is False
        assert info.project_name is None
        assert info.suffix is None
    
    def test_get_collection_info_unknown(self, classifier):
        """Test CollectionInfo for unknown collection."""
        info = classifier.get_collection_info("invalid_collection")
        
        assert info.name == "invalid_collection"
        assert info.type == CollectionType.UNKNOWN
        assert info.display_name == "invalid_collection"
        assert info.is_searchable is False  # Default to restricted
        assert info.is_readonly is True     # Default to readonly
        assert info.is_memory_collection is False
        assert info.project_name is None
        assert info.suffix is None
    
    # Project component extraction tests
    
    @pytest.mark.parametrize("collection_name,expected", [
        ("my_project-documents", ("my_project", "documents")),
        ("workspace-lsp_metadata", ("workspace", "lsp_metadata")),
        ("test-data", ("test", "data")),
        ("a-b", ("a", "b")),
        
        # Non-project collections
        ("__system_config", None),
        ("_library_docs", None),
        ("algorithms", None),
        ("invalid_name", None),
        ("", None),
        
        # Edge cases
        ("project-", None),  # Invalid project pattern
        ("-suffix", None),   # Invalid project pattern
    ])
    def test_extract_project_components(self, classifier, collection_name, expected):
        """Test project component extraction."""
        result = classifier.extract_project_components(collection_name)
        assert result == expected


class TestValidationFunctions:
    """Test module-level validation functions."""
    
    # Collection operation validation tests
    
    @pytest.mark.parametrize("collection_name,operation,expected_valid,expected_reason_contains", [
        # System collections - CLI-writable
        ("__user_prefs", "read", True, "Read operations are generally allowed"),
        ("__user_prefs", "write", True, "System collections are CLI-writable"),
        ("__user_prefs", "delete", True, "System collections are CLI-writable"),
        ("__user_prefs", "create", True, "System collections are CLI-writable"),
        
        # Library collections - MCP-readonly
        ("_library_docs", "read", True, "Read operations are generally allowed"),
        ("_library_docs", "write", False, "Library collections are read-only via MCP"),
        ("_library_docs", "delete", False, "Library collections are read-only via MCP"),
        ("_library_docs", "create", False, "Library collections are read-only via MCP"),
        
        # Project collections - user-writable
        ("project-docs", "read", True, "Read operations are generally allowed"),
        ("project-docs", "write", True, "Operation 'write' is allowed on project collection"),
        ("project-docs", "delete", True, "Operation 'delete' is allowed on project collection"),
        ("project-docs", "create", True, "Operation 'create' is allowed on project collection"),
        
        # Global collections - generally writable
        ("algorithms", "read", True, "Read operations are generally allowed"),
        ("algorithms", "write", True, "Operation 'write' is allowed on global collection"),
        ("algorithms", "delete", True, "Operation 'delete' is allowed on global collection"),
        ("algorithms", "create", True, "Operation 'create' is allowed on global collection"),
        
        # Unknown collections - default to readonly
        ("invalid", "read", True, "Read operations are generally allowed"),
        ("invalid", "write", False, "Collection 'invalid' is read-only"),
        ("invalid", "delete", False, "Collection 'invalid' is read-only"),
        ("invalid", "create", False, "Collection 'invalid' is read-only"),
        
        # Invalid operations
        ("__system", "invalid_op", False, "Invalid operation 'invalid_op'"),
        ("project-docs", "destroy", False, "Invalid operation 'destroy'"),
    ])
    def test_validate_collection_operation(self, collection_name, operation, expected_valid, expected_reason_contains):
        """Test collection operation validation."""
        is_valid, reason = validate_collection_operation(collection_name, operation)
        
        assert is_valid == expected_valid
        assert expected_reason_contains.lower() in reason.lower()
    
    # Collection filtering tests
    
    def test_get_collections_by_type(self):
        """Test filtering collections by type."""
        collections = [
            "__system_config",
            "_library_docs",
            "project-documents",
            "algorithms",
            "workspace-memory",
            "invalid_collection"
        ]
        
        system_collections = get_collections_by_type(collections, CollectionType.SYSTEM)
        assert system_collections == ["__system_config"]
        
        library_collections = get_collections_by_type(collections, CollectionType.LIBRARY)
        assert library_collections == ["_library_docs"]
        
        project_collections = get_collections_by_type(collections, CollectionType.PROJECT)
        assert set(project_collections) == {"project-documents", "workspace-memory"}
        
        global_collections = get_collections_by_type(collections, CollectionType.GLOBAL)
        assert global_collections == ["algorithms"]
        
        unknown_collections = get_collections_by_type(collections, CollectionType.UNKNOWN)
        assert unknown_collections == ["invalid_collection"]
    
    def test_get_collections_by_type_empty_list(self):
        """Test filtering with empty collection list."""
        result = get_collections_by_type([], CollectionType.SYSTEM)
        assert result == []
    
    def test_get_searchable_collections(self):
        """Test filtering for searchable collections."""
        collections = [
            "__system_config",      # Not searchable
            "_library_docs",        # Searchable
            "project-documents",    # Searchable
            "algorithms",           # Searchable
            "invalid_collection"    # Not searchable (unknown)
        ]
        
        searchable = get_searchable_collections(collections)
        expected_searchable = {"_library_docs", "project-documents", "algorithms"}
        
        assert set(searchable) == expected_searchable
    
    def test_get_searchable_collections_empty_list(self):
        """Test filtering with empty collection list."""
        result = get_searchable_collections([])
        assert result == []
    
    # Collection name validation with type
    
    @pytest.mark.parametrize("collection_name,expected_type,expected_valid", [
        # Matching types
        ("__system_config", CollectionType.SYSTEM, True),
        ("_library_docs", CollectionType.LIBRARY, True),
        ("project-docs", CollectionType.PROJECT, True),
        ("algorithms", CollectionType.GLOBAL, True),
        ("invalid", CollectionType.UNKNOWN, True),
        
        # Non-matching types
        ("__system_config", CollectionType.LIBRARY, False),
        ("_library_docs", CollectionType.SYSTEM, False),
        ("project-docs", CollectionType.GLOBAL, False),
        ("algorithms", CollectionType.PROJECT, False),
        ("invalid", CollectionType.SYSTEM, False),
    ])
    def test_validate_collection_name_with_type(self, collection_name, expected_type, expected_valid):
        """Test collection name validation against expected types."""
        result = validate_collection_name_with_type(collection_name, expected_type)
        assert result == expected_valid


class TestBuildCollectionNameForType:
    """Test collection name building function."""
    
    def test_build_system_collection_name(self):
        """Test building system collection names."""
        result = build_collection_name_for_type("user_prefs", CollectionType.SYSTEM)
        assert result == "__user_prefs"
        
        # Test with normalization
        result = build_collection_name_for_type("user-prefs", CollectionType.SYSTEM)
        assert result == "__user_prefs"
    
    def test_build_library_collection_name(self):
        """Test building library collection names."""
        result = build_collection_name_for_type("library_docs", CollectionType.LIBRARY)
        assert result == "_library_docs"
        
        # Test with normalization
        result = build_collection_name_for_type("library-docs", CollectionType.LIBRARY)
        assert result == "_library_docs"
    
    def test_build_project_collection_name(self):
        """Test building project collection names."""
        result = build_collection_name_for_type("documents", CollectionType.PROJECT, "myproject")
        assert result == "myproject-documents"
        
        # Test with normalization
        result = build_collection_name_for_type("lsp metadata", CollectionType.PROJECT, "my-project")
        assert result == "my_project-lsp_metadata"
    
    def test_build_project_collection_name_missing_project(self):
        """Test building project collection without project name."""
        with pytest.raises(ValueError, match="project_name is required for project collections"):
            build_collection_name_for_type("documents", CollectionType.PROJECT)
    
    def test_build_global_collection_name_valid(self):
        """Test building valid global collection names."""
        for global_name in GLOBAL_COLLECTIONS:
            result = build_collection_name_for_type(global_name, CollectionType.GLOBAL)
            assert result == global_name
    
    def test_build_global_collection_name_invalid(self):
        """Test building invalid global collection names."""
        with pytest.raises(ValueError, match="'invalid' is not a valid global collection"):
            build_collection_name_for_type("invalid", CollectionType.GLOBAL)
    
    def test_build_unknown_collection_type(self):
        """Test building collection name for unknown type."""
        with pytest.raises(ValueError, match="Cannot build collection name for type"):
            build_collection_name_for_type("test", CollectionType.UNKNOWN)


class TestIntegrationWithCollectionNaming:
    """Test integration with collection_naming module."""
    
    def test_build_system_collection_integration(self):
        """Test integration with collection_naming for system collections."""
        # Mock collection_naming functions to verify they're called
        with patch.object(collection_naming, 'build_system_memory_collection_name') as mock_build:
            mock_build.return_value = "__test_memory"
            
            result = build_collection_name_for_type("test_memory", CollectionType.SYSTEM)
            
            mock_build.assert_called_once_with("test_memory")
            assert result == "__test_memory"
    
    def test_build_project_collection_integration(self):
        """Test integration with collection_naming for project collections."""
        with patch.object(collection_naming, 'build_project_collection_name') as mock_build:
            mock_build.return_value = "project-documents"
            
            result = build_collection_name_for_type("documents", CollectionType.PROJECT, "project")
            
            mock_build.assert_called_once_with("project", "documents")
            assert result == "project-documents"
    
    def test_build_library_collection_integration(self):
        """Test integration with collection_naming for library collections."""
        with patch.object(collection_naming, 'normalize_collection_name_component') as mock_normalize:
            mock_normalize.return_value = "library_docs"
            
            result = build_collection_name_for_type("library-docs", CollectionType.LIBRARY)
            
            mock_normalize.assert_called_once_with("library-docs")
            assert result == "_library_docs"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""
    
    @pytest.fixture
    def classifier(self):
        """Fixture providing a CollectionTypeClassifier instance."""
        return CollectionTypeClassifier()
    
    def test_empty_string_handling(self, classifier):
        """Test handling of empty strings."""
        assert classifier.classify_collection_type("") == CollectionType.UNKNOWN
        assert classifier.is_system_collection("") is False
        assert classifier.is_memory_collection("") is False
        assert classifier.get_display_name("") == ""
        
        info = classifier.get_collection_info("")
        assert info.type == CollectionType.UNKNOWN
        assert info.is_readonly is True
        assert info.is_searchable is False
    
    def test_whitespace_handling(self, classifier):
        """Test handling of whitespace-only strings."""
        whitespace_names = ["   ", "\t", "\n", " \t\n "]
        
        for name in whitespace_names:
            assert classifier.classify_collection_type(name) == CollectionType.UNKNOWN
            assert classifier.is_system_collection(name) is False
            assert classifier.get_display_name(name) == name
    
    def test_none_and_non_string_inputs(self, classifier):
        """Test handling of None and non-string inputs."""
        invalid_inputs = [None, 123, [], {}, object()]
        
        for invalid_input in invalid_inputs:
            assert classifier.classify_collection_type(invalid_input) == CollectionType.UNKNOWN
            assert classifier.is_system_collection(invalid_input) is False
            assert classifier.is_library_collection(invalid_input) is False
            assert classifier.is_project_collection(invalid_input) is False
            assert classifier.is_global_collection(invalid_input) is False
            assert classifier.is_memory_collection(invalid_input) is False
    
    def test_very_long_collection_names(self, classifier):
        """Test handling of very long collection names."""
        long_system = "__" + "a" * 1000
        long_library = "_" + "b" * 1000  
        long_project = "c" * 500 + "-" + "d" * 500
        
        assert classifier.classify_collection_type(long_system) == CollectionType.SYSTEM
        assert classifier.classify_collection_type(long_library) == CollectionType.LIBRARY
        assert classifier.classify_collection_type(long_project) == CollectionType.PROJECT
    
    def test_special_characters_in_names(self, classifier):
        """Test handling of special characters in collection names."""
        special_names = [
            "__system@config",
            "_library#docs",
            "project$-documents%",
            "test collection with spaces",
            "collection.with.dots",
            "collection/with/slashes"
        ]
        
        # Most special characters should result in classification based on prefix
        assert classifier.classify_collection_type("__system@config") == CollectionType.SYSTEM
        assert classifier.classify_collection_type("_library#docs") == CollectionType.LIBRARY
        
        # Invalid project patterns should be UNKNOWN
        assert classifier.classify_collection_type("test collection with spaces") == CollectionType.UNKNOWN
    
    def test_case_sensitivity(self, classifier):
        """Test that classification is case-sensitive."""
        # These should not match system/library patterns due to case
        assert classifier.classify_collection_type("__System_Config") == CollectionType.SYSTEM  # Still system due to __
        assert classifier.classify_collection_type("_Library_Docs") == CollectionType.LIBRARY   # Still library due to _
        
        # Global collections are case-sensitive
        assert classifier.classify_collection_type("Algorithms") == CollectionType.UNKNOWN
        assert classifier.classify_collection_type("ALGORITHMS") == CollectionType.UNKNOWN
    
    def test_boundary_conditions_for_patterns(self, classifier):
        """Test boundary conditions for pattern matching."""
        # Minimal valid names
        assert classifier.classify_collection_type("__a") == CollectionType.SYSTEM
        assert classifier.classify_collection_type("_a") == CollectionType.LIBRARY
        assert classifier.classify_collection_type("a-b") == CollectionType.PROJECT
        
        # Edge cases for project pattern
        assert classifier.classify_collection_type("a-") == CollectionType.UNKNOWN
        assert classifier.classify_collection_type("-b") == CollectionType.UNKNOWN
        assert classifier.classify_collection_type("-") == CollectionType.UNKNOWN
        assert classifier.classify_collection_type("--") == CollectionType.UNKNOWN
        
        # Multiple dashes (not valid project pattern according to regex)
        assert classifier.classify_collection_type("a-b-c") == CollectionType.UNKNOWN
    
    def test_prefix_edge_cases(self, classifier):
        """Test edge cases with prefixes."""
        # Just prefixes
        assert classifier.classify_collection_type("__") == CollectionType.UNKNOWN
        assert classifier.classify_collection_type("_") == CollectionType.UNKNOWN
        
        # Triple underscore (should still be system)
        assert classifier.classify_collection_type("___system") == CollectionType.SYSTEM
        
        # Prefix within name (not at start)
        assert classifier.classify_collection_type("system__config") == CollectionType.UNKNOWN
        assert classifier.classify_collection_type("lib_rary") == CollectionType.UNKNOWN
    
    def test_memory_pattern_edge_cases(self, classifier):
        """Test edge cases for memory pattern detection."""
        # Valid memory patterns
        assert classifier.is_memory_collection("__memory") is True
        assert classifier.is_memory_collection("__a") is True  # Any system collection matches pattern
        assert classifier.is_memory_collection("project-memory") is True
        assert classifier.is_memory_collection("a-memory") is True
        
        # Invalid memory patterns
        assert classifier.is_memory_collection("__memory_extra") is True  # Still matches system pattern
        assert classifier.is_memory_collection("project-memory-extra") is False  # Doesn't match exact project pattern
        assert classifier.is_memory_collection("project_memory") is False  # No dash
        assert classifier.is_memory_collection("memory") is False  # Global collection, not memory pattern
    
    def test_validation_with_edge_case_operations(self):
        """Test operation validation with edge case operations."""
        # Test case sensitivity in operations
        is_valid, reason = validate_collection_operation("__system", "READ")
        assert is_valid is False
        assert "invalid operation" in reason.lower()
        
        # Test with empty operation
        is_valid, reason = validate_collection_operation("__system", "")
        assert is_valid is False
        assert "invalid operation" in reason.lower()


class TestPerformanceAndScalability:
    """Test performance characteristics and scalability."""
    
    @pytest.fixture
    def classifier(self):
        """Fixture providing a CollectionTypeClassifier instance."""
        return CollectionTypeClassifier()
    
    def test_classifier_regex_compilation(self, classifier):
        """Test that regex patterns are compiled during initialization."""
        # Verify that patterns are compiled (have pattern attribute)
        assert hasattr(classifier._project_pattern, 'pattern')
        assert hasattr(classifier._system_memory_pattern, 'pattern')
        assert hasattr(classifier._project_memory_pattern, 'pattern')
    
    def test_global_collections_set_performance(self, classifier):
        """Test that global collections use set for O(1) lookup."""
        # Verify that global collections are stored as a set
        assert isinstance(classifier._global_collections_set, set)
        assert len(classifier._global_collections_set) == len(GLOBAL_COLLECTIONS)
    
    def test_repeated_classification_consistency(self, classifier):
        """Test that repeated classifications return consistent results."""
        test_names = [
            "__system_config",
            "_library_docs", 
            "project-documents",
            "algorithms",
            "invalid_name"
        ]
        
        # Test multiple iterations for consistency
        first_results = [classifier.classify_collection_type(name) for name in test_names]
        
        for _ in range(10):
            current_results = [classifier.classify_collection_type(name) for name in test_names]
            assert current_results == first_results


if __name__ == "__main__":
    # Run tests with verbose output when executed directly
    pytest.main([__file__, "-v", "--tb=short"])