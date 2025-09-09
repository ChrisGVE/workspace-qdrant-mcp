"""
Comprehensive unit tests for the collection naming framework.

This module tests all functions in core/collection_naming.py including:
- normalize_collection_name_component()
- build_project_collection_name()
- build_system_memory_collection_name()
- validate_collection_name()

Tests cover normal operation, edge cases, and error handling scenarios.
"""

import pytest
from typing import Any

from core.collection_naming import (
    normalize_collection_name_component,
    build_project_collection_name,
    build_system_memory_collection_name,
    validate_collection_name,
    PROJECT_COLLECTION_PATTERN,
    SYSTEM_COLLECTION_PATTERN,
    SINGLE_COMPONENT_PATTERN
)


class TestNormalizeCollectionNameComponent:
    """Test normalize_collection_name_component() function."""
    
    def test_basic_dash_replacement(self):
        """Test basic dash replacement functionality."""
        result = normalize_collection_name_component("my-project")
        assert result == "my_project"
        
    def test_basic_space_replacement(self):
        """Test basic space replacement functionality."""
        result = normalize_collection_name_component("user notes")
        assert result == "user_notes"
        
    def test_combined_dashes_and_spaces(self):
        """Test handling of combined dashes and spaces."""
        result = normalize_collection_name_component("my-awesome project")
        assert result == "my_awesome_project"
        
    def test_multiple_consecutive_separators(self):
        """Test handling of multiple consecutive separators."""
        test_cases = [
            ("my--project", "my_project"),
            ("my  project", "my_project"),
            ("my- -project", "my_project"),
            ("my  --  project", "my_project")
        ]
        
        for input_name, expected in test_cases:
            result = normalize_collection_name_component(input_name)
            assert result == expected, f"Failed for input: {input_name}"
            
    def test_leading_trailing_whitespace(self):
        """Test handling of leading and trailing whitespace."""
        test_cases = [
            ("  my-project  ", "my_project"),
            ("\tmy project\t", "my_project"),
            ("\n my-project \n", "my_project")
        ]
        
        for input_name, expected in test_cases:
            result = normalize_collection_name_component(input_name)
            assert result == expected, f"Failed for input: {repr(input_name)}"
            
    def test_leading_trailing_separators_removal(self):
        """Test removal of leading and trailing underscores after normalization."""
        test_cases = [
            ("-my-project-", "my_project"),
            (" my project ", "my_project"),
            ("--my--project--", "my_project"),
            ("  my  project  ", "my_project")
        ]
        
        for input_name, expected in test_cases:
            result = normalize_collection_name_component(input_name)
            assert result == expected, f"Failed for input: {input_name}"

    @pytest.mark.parametrize("input_name,expected", [
        ("simple", "simple"),
        ("already_normalized", "already_normalized"),
        ("123numbers", "123numbers"),
        ("CamelCase", "CamelCase"),
        ("mixed123_test", "mixed123_test")
    ])
    def test_already_normalized_names(self, input_name: str, expected: str):
        """Test names that are already normalized."""
        result = normalize_collection_name_component(input_name)
        assert result == expected
        
    def test_numbers_and_special_chars(self):
        """Test handling of numbers and other characters."""
        test_cases = [
            ("project123", "project123"),
            ("123project", "123project"),
            ("test_v1.2", "test_v1.2"),  # Dots and other chars preserved
            ("my@project", "my@project")  # @ preserved
        ]
        
        for input_name, expected in test_cases:
            result = normalize_collection_name_component(input_name)
            assert result == expected, f"Failed for input: {input_name}"
            
    def test_error_handling_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(TypeError, match="Expected string, got NoneType"):
            normalize_collection_name_component(None)
            
    def test_error_handling_non_string_input(self):
        """Test error handling for non-string inputs."""
        test_cases = [123, [], {}, 45.67, True]
        
        for invalid_input in test_cases:
            with pytest.raises(TypeError, match=f"Expected string, got {type(invalid_input).__name__}"):
                normalize_collection_name_component(invalid_input)
                
    def test_error_handling_empty_string(self):
        """Test error handling for empty strings."""
        with pytest.raises(ValueError, match="Name cannot be empty or contain only whitespace"):
            normalize_collection_name_component("")
            
    def test_error_handling_whitespace_only(self):
        """Test error handling for whitespace-only strings."""
        whitespace_cases = ["   ", "\t\t", "\n\n", " \t\n "]
        
        for whitespace_str in whitespace_cases:
            with pytest.raises(ValueError, match="Name cannot be empty or contain only whitespace"):
                normalize_collection_name_component(whitespace_str)
                
    def test_error_handling_empty_after_normalization(self):
        """Test error handling when normalization results in empty string."""
        edge_cases = ["---", "   ", "- -", "--  --"]
        
        for edge_case in edge_cases:
            with pytest.raises(ValueError, match="Name resulted in empty string after normalization"):
                normalize_collection_name_component(edge_case)


class TestBuildProjectCollectionName:
    """Test build_project_collection_name() function."""
    
    def test_basic_project_suffix_combination(self):
        """Test basic project and suffix combination."""
        result = build_project_collection_name("myproject", "documents")
        assert result == "myproject-documents"
        
    def test_project_and_suffix_normalization(self):
        """Test that both project and suffix are normalized."""
        result = build_project_collection_name("my project", "source code")
        assert result == "my_project-source_code"
        
    def test_edge_case_my_awesome_project(self):
        """Test specific edge case: 'my-awesome-project' → 'my_awesome_project-notes'."""
        result = build_project_collection_name("my-awesome-project", "notes")
        assert result == "my_awesome_project-notes"
        
    def test_complex_project_names(self):
        """Test various complex project names."""
        test_cases = [
            ("frontend-app", "docs", "frontend_app-docs"),
            ("api gateway", "source-code", "api_gateway-source_code"),
            ("my--complex   project", "test-data", "my_complex_project-test_data"),
            ("workspace", "lsp metadata", "workspace-lsp_metadata")
        ]
        
        for project, suffix, expected in test_cases:
            result = build_project_collection_name(project, suffix)
            assert result == expected, f"Failed for project='{project}', suffix='{suffix}'"
            
    def test_schema_delimiter_preservation(self):
        """Test that dash delimiter is preserved between project and suffix."""
        # Even if both components have underscores, delimiter should be dash
        result = build_project_collection_name("my_project", "test_data")
        assert result == "my_project-test_data"
        assert "-" in result, "Schema delimiter (dash) should be preserved"
        
    def test_error_handling_none_project(self):
        """Test error handling for None project."""
        with pytest.raises(TypeError, match="Project must be string, got NoneType"):
            build_project_collection_name(None, "suffix")
            
    def test_error_handling_none_suffix(self):
        """Test error handling for None suffix."""
        with pytest.raises(TypeError, match="Suffix must be string, got NoneType"):
            build_project_collection_name("project", None)
            
    def test_error_handling_non_string_project(self):
        """Test error handling for non-string project."""
        with pytest.raises(TypeError, match="Project must be string, got int"):
            build_project_collection_name(123, "suffix")
            
    def test_error_handling_non_string_suffix(self):
        """Test error handling for non-string suffix."""
        with pytest.raises(TypeError, match="Suffix must be string, got list"):
            build_project_collection_name("project", [])
            
    def test_error_handling_empty_project(self):
        """Test error handling for empty project after normalization."""
        with pytest.raises(ValueError, match="Name cannot be empty or contain only whitespace"):
            build_project_collection_name("", "suffix")
            
    def test_error_handling_empty_suffix(self):
        """Test error handling for empty suffix after normalization."""
        with pytest.raises(ValueError, match="Name cannot be empty or contain only whitespace"):
            build_project_collection_name("project", "   ")
            
    def test_error_propagation_from_normalization(self):
        """Test that normalization errors are properly propagated."""
        # This should trigger the "empty after normalization" error
        with pytest.raises(ValueError, match="Name resulted in empty string after normalization"):
            build_project_collection_name("---", "suffix")


class TestBuildSystemMemoryCollectionName:
    """Test build_system_memory_collection_name() function."""
    
    def test_basic_system_collection_creation(self):
        """Test basic system collection name creation."""
        result = build_system_memory_collection_name("user_preferences")
        assert result == "__user_preferences"
        
    def test_memory_collection_normalization(self):
        """Test that memory collection names are normalized."""
        test_cases = [
            ("user-preferences", "__user_preferences"),
            ("system config", "__system_config"),
            ("global-state", "__global_state"),
            ("cache data", "__cache_data")
        ]
        
        for memory_name, expected in test_cases:
            result = build_system_memory_collection_name(memory_name)
            assert result == expected, f"Failed for memory_name: {memory_name}"
            
    def test_double_underscore_prefix(self):
        """Test that double underscore prefix is always added."""
        result = build_system_memory_collection_name("memory")
        assert result.startswith("__")
        assert result.count("_") >= 2  # At least the prefix
        
    def test_complex_memory_names(self):
        """Test various complex memory collection names."""
        test_cases = [
            ("user--session-data", "__user_session_data"),
            ("application   state", "__application_state"),
            ("multi-part-config-data", "__multi_part_config_data")
        ]
        
        for memory_name, expected in test_cases:
            result = build_system_memory_collection_name(memory_name)
            assert result == expected, f"Failed for memory_name: {memory_name}"
            
    def test_error_handling_none_input(self):
        """Test error handling for None input."""
        with pytest.raises(TypeError, match="Memory collection must be string, got NoneType"):
            build_system_memory_collection_name(None)
            
    def test_error_handling_non_string_input(self):
        """Test error handling for non-string inputs."""
        test_cases = [123, [], {}, 45.67]
        
        for invalid_input in test_cases:
            expected_type = type(invalid_input).__name__
            with pytest.raises(TypeError, match=f"Memory collection must be string, got {expected_type}"):
                build_system_memory_collection_name(invalid_input)
                
    def test_error_handling_empty_memory_collection(self):
        """Test error handling for empty memory collection name."""
        with pytest.raises(ValueError, match="Name cannot be empty or contain only whitespace"):
            build_system_memory_collection_name("")
            
    def test_error_handling_whitespace_only(self):
        """Test error handling for whitespace-only memory collection name."""
        with pytest.raises(ValueError, match="Name cannot be empty or contain only whitespace"):
            build_system_memory_collection_name("   ")
            
    def test_error_propagation_from_normalization(self):
        """Test that normalization errors are properly propagated."""
        with pytest.raises(ValueError, match="Name resulted in empty string after normalization"):
            build_system_memory_collection_name("---")


class TestValidateCollectionName:
    """Test validate_collection_name() function."""
    
    def test_valid_project_pattern_names(self):
        """Test validation of valid project pattern names."""
        valid_names = [
            "project_name-documents",
            "my_project-source_code", 
            "frontend-api_data",
            "test123-docs456",
            "simple-test"
        ]
        
        for name in valid_names:
            result = validate_collection_name(name)
            assert result is True, f"Should be valid: {name}"
            
    def test_valid_system_pattern_names(self):
        """Test validation of valid system pattern names."""
        valid_names = [
            "__system_config",
            "__user_preferences",
            "__cache123",
            "__simple",
            "__data_store_config"
        ]
        
        for name in valid_names:
            result = validate_collection_name(name)
            assert result is True, f"Should be valid: {name}"
            
    def test_valid_single_component_names(self):
        """Test validation of valid single component names."""
        valid_names = [
            "documents",
            "test123",
            "simple_name",
            "camelCase",
            "under_score_name"
        ]
        
        for name in valid_names:
            result = validate_collection_name(name)
            assert result is True, f"Should be valid: {name}"
            
    def test_invalid_project_patterns(self):
        """Test validation of invalid project patterns."""
        invalid_names = [
            "invalid--name",        # Double dash
            "project-",             # Trailing dash
            "-documents",           # Leading dash
            "project--docs",        # Double dash
            "a-b-c",               # Too many dashes
            "project-docs-extra"    # Too many parts
        ]
        
        for name in invalid_names:
            result = validate_collection_name(name)
            assert result is False, f"Should be invalid: {name}"
            
    def test_invalid_system_patterns(self):
        """Test validation of invalid system patterns."""
        invalid_names = [
            "_single_underscore",   # Single underscore
            "___triple_underscore", # Triple underscore
            "__",                   # Just double underscore
            "__invalid-dash",       # Contains dash
            "__invalid space",      # Contains space
            "__invalid.dot"         # Contains dot
        ]
        
        for name in invalid_names:
            result = validate_collection_name(name)
            assert result is False, f"Should be invalid: {name}"
            
    def test_invalid_single_component_patterns(self):
        """Test validation of invalid single component patterns."""
        invalid_names = [
            "invalid-dash",         # Contains dash (should be project pattern)
            "invalid space",        # Contains space
            "invalid.dot",          # Contains dot
            "invalid@symbol",       # Contains symbol
            "invalid/slash"         # Contains slash
        ]
        
        for name in invalid_names:
            result = validate_collection_name(name)
            assert result is False, f"Should be invalid: {name}"
            
    def test_error_handling_none_input(self):
        """Test error handling for None input."""
        result = validate_collection_name(None)
        assert result is False
        
    def test_error_handling_empty_string(self):
        """Test error handling for empty string."""
        result = validate_collection_name("")
        assert result is False
        
    def test_error_handling_non_string_input(self):
        """Test error handling for non-string inputs."""
        test_cases = [123, [], {}, 45.67, True]
        
        for invalid_input in test_cases:
            result = validate_collection_name(invalid_input)
            assert result is False, f"Should be False for input: {invalid_input}"
            
    def test_edge_case_patterns(self):
        """Test edge case patterns."""
        edge_cases = [
            ("_", False),           # Single underscore
            ("__a", True),          # Minimal valid system
            ("a", True),            # Minimal valid single
            ("a-b", True),          # Minimal valid project
            ("a-b-c", False),       # Too many dashes
        ]
        
        for name, expected in edge_cases:
            result = validate_collection_name(name)
            assert result == expected, f"Failed for edge case: {name}"


class TestModuleConstants:
    """Test module-level constants."""
    
    def test_pattern_constants_exist(self):
        """Test that pattern constants are defined."""
        assert PROJECT_COLLECTION_PATTERN is not None
        assert SYSTEM_COLLECTION_PATTERN is not None
        assert SINGLE_COMPONENT_PATTERN is not None
        
    def test_pattern_constants_are_strings(self):
        """Test that pattern constants are regex strings."""
        assert isinstance(PROJECT_COLLECTION_PATTERN, str)
        assert isinstance(SYSTEM_COLLECTION_PATTERN, str)
        assert isinstance(SINGLE_COMPONENT_PATTERN, str)
        
    def test_project_pattern_examples(self):
        """Test project pattern with regex examples."""
        import re
        pattern = re.compile(PROJECT_COLLECTION_PATTERN)
        
        valid_examples = ["test_name-docs", "project123-data_store"]
        invalid_examples = ["single", "__system", "too-many-parts"]
        
        for example in valid_examples:
            assert pattern.match(example), f"Project pattern should match: {example}"
            
        for example in invalid_examples:
            assert not pattern.match(example), f"Project pattern should not match: {example}"
            
    def test_system_pattern_examples(self):
        """Test system pattern with regex examples."""
        import re
        pattern = re.compile(SYSTEM_COLLECTION_PATTERN)
        
        valid_examples = ["__config", "__user_data123"]
        invalid_examples = ["_single", "no_prefix", "project-docs"]
        
        for example in valid_examples:
            assert pattern.match(example), f"System pattern should match: {example}"
            
        for example in invalid_examples:
            assert not pattern.match(example), f"System pattern should not match: {example}"
            
    def test_single_component_pattern_examples(self):
        """Test single component pattern with regex examples."""
        import re
        pattern = re.compile(SINGLE_COMPONENT_PATTERN)
        
        valid_examples = ["simple", "test123", "under_score"]
        invalid_examples = ["with-dash", "__system", "with space"]
        
        for example in valid_examples:
            assert pattern.match(example), f"Single component pattern should match: {example}"
            
        for example in invalid_examples:
            assert not pattern.match(example), f"Single component pattern should not match: {example}"


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple functions."""
    
    def test_end_to_end_project_workflow(self):
        """Test end-to-end workflow for project collections."""
        # Start with raw project and suffix names
        raw_project = "my-awesome-project"
        raw_suffix = "user notes"
        
        # Build project collection name
        collection_name = build_project_collection_name(raw_project, raw_suffix)
        assert collection_name == "my_awesome_project-user_notes"
        
        # Validate the resulting collection name
        is_valid = validate_collection_name(collection_name)
        assert is_valid is True
        
    def test_end_to_end_system_workflow(self):
        """Test end-to-end workflow for system collections."""
        # Start with raw memory collection name
        raw_memory_name = "user-session data"
        
        # Build system memory collection name
        collection_name = build_system_memory_collection_name(raw_memory_name)
        assert collection_name == "__user_session_data"
        
        # Validate the resulting collection name
        is_valid = validate_collection_name(collection_name)
        assert is_valid is True
        
    def test_normalization_consistency(self):
        """Test that normalization is consistent across functions."""
        test_input = "complex-name with spaces"
        
        # Normalize directly
        direct_result = normalize_collection_name_component(test_input)
        
        # Normalize through project collection building
        project_result = build_project_collection_name(test_input, "suffix")
        project_normalized = project_result.split("-")[0]
        
        # Normalize through system collection building
        system_result = build_system_memory_collection_name(test_input)
        system_normalized = system_result[2:]  # Remove __ prefix
        
        # All should produce the same normalization
        assert direct_result == project_normalized == system_normalized
        assert direct_result == "complex_name_with_spaces"


# Parametrized test fixtures for comprehensive edge case testing
@pytest.mark.parametrize("input_str,expected", [
    # Basic normalization cases
    ("simple", "simple"),
    ("with-dash", "with_dash"),
    ("with space", "with_space"),
    ("with-dash and space", "with_dash_and_space"),
    
    # Multiple separator cases
    ("multiple--dashes", "multiple_dashes"),
    ("multiple  spaces", "multiple_spaces"),
    ("mixed--  separators", "mixed_separators"),
    
    # Leading/trailing cases
    ("-leading-dash", "leading_dash"),
    ("trailing-dash-", "trailing_dash"),
    (" leading-space", "leading_space"),
    ("trailing-space ", "trailing_space"),
    ("  multiple  leading  ", "multiple_leading"),
])
def test_normalize_collection_name_component_parametrized(input_str: str, expected: str):
    """Parametrized test for normalize_collection_name_component."""
    result = normalize_collection_name_component(input_str)
    assert result == expected


@pytest.mark.parametrize("collection_name,expected", [
    # Valid project patterns
    ("project-docs", True),
    ("my_project-source_code", True),
    ("test123-data456", True),
    
    # Valid system patterns  
    ("__system_config", True),
    ("__user123", True),
    ("__complex_system_name", True),
    
    # Valid single patterns
    ("simple", True),
    ("test123", True),
    ("under_score", True),
    
    # Invalid patterns
    ("invalid--double", False),
    ("invalid-", False),
    ("-invalid", False),
    ("__invalid-dash", False),
    ("invalid space", False),
    ("", False),
])
def test_validate_collection_name_parametrized(collection_name: str, expected: bool):
    """Parametrized test for validate_collection_name."""
    result = validate_collection_name(collection_name)
    assert result == expected