"""
Unit tests for the collection naming validation system.

This test suite provides comprehensive validation of the naming validation
implementation including pattern validation, conflict detection, metadata
integration, and error handling.

Test Categories:
    - Pattern validation for each collection category
    - Conflict detection across collection types
    - Configuration validation and custom settings
    - Error messages and suggestion generation
    - Integration with metadata schema
    - Edge cases and boundary conditions
"""

import pytest
from typing import List, Optional

from workspace_qdrant_mcp.core.collection_naming_validation import (
    CollectionNamingValidator,
    ValidationResult,
    NamingConfiguration,
    PatternValidator,
    ConflictDetector,
    ValidationSeverity,
    ConflictType,
    create_naming_validator,
    validate_collection_name_with_metadata,
    check_collection_conflicts
)

from workspace_qdrant_mcp.core.metadata_schema import (
    CollectionCategory,
    WorkspaceScope,
    AccessLevel
)


class TestPatternValidator:
    """Test suite for PatternValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = NamingConfiguration()
        self.validator = PatternValidator(self.config)

    def test_validate_memory_collection_success(self):
        """Test successful memory collection validation."""
        result = self.validator.validate_memory_collection("memory")

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.SUCCESS
        assert result.detected_category == CollectionCategory.SYSTEM
        assert result.detected_pattern == "memory_collection"

    def test_validate_memory_collection_with_custom_name(self):
        """Test memory collection validation with custom configuration."""
        config = NamingConfiguration(
            memory_collection_name="user_memory",
            allow_custom_memory_names=True
        )
        validator = PatternValidator(config)

        result = validator.validate_memory_collection("user_memory")
        assert result.is_valid is True
        assert result.detected_pattern == "memory_collection"

    def test_validate_memory_collection_invalid_prefix(self):
        """Test memory collection validation with invalid prefixes."""
        result = self.validator.validate_memory_collection("__memory")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.PATTERN_VIOLATION
        assert "cannot use system or library prefixes" in result.error_message
        assert "memory" in result.suggested_names

    def test_validate_memory_collection_custom_disabled(self):
        """Test memory collection validation when custom names are disabled."""
        result = self.validator.validate_memory_collection("custom_memory")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.MEMORY_CONFLICT
        assert "Only 'memory' is allowed" in result.error_message

    def test_validate_system_collection_success(self):
        """Test successful system collection validation."""
        result = self.validator.validate_system_collection("__user_config")

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.SUCCESS
        assert result.detected_category == CollectionCategory.SYSTEM
        assert result.detected_pattern == "system_prefix"

    def test_validate_system_collection_missing_prefix(self):
        """Test system collection validation without prefix."""
        result = self.validator.validate_system_collection("user_config")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.PATTERN_VIOLATION
        assert "must start with '__'" in result.error_message

    def test_validate_system_collection_empty_base(self):
        """Test system collection validation with empty base name."""
        result = self.validator.validate_system_collection("__")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert "must have content after '__' prefix" in result.error_message

    def test_validate_system_collection_invalid_base(self):
        """Test system collection validation with invalid base name."""
        result = self.validator.validate_system_collection("__123invalid")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert "has invalid format" in result.error_message

    def test_validate_library_collection_success(self):
        """Test successful library collection validation."""
        result = self.validator.validate_library_collection("_mylibrary")

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.SUCCESS
        assert result.detected_category == CollectionCategory.LIBRARY
        assert result.detected_pattern == "library_prefix"

    def test_validate_library_collection_wrong_prefix(self):
        """Test library collection validation with wrong prefix."""
        result = self.validator.validate_library_collection("__mylibrary")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.PATTERN_VIOLATION
        assert "start with '_' but not '__'" in result.error_message

    def test_validate_library_collection_no_prefix(self):
        """Test library collection validation without prefix."""
        result = self.validator.validate_library_collection("mylibrary")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert "start with '_' but not '__'" in result.error_message

    def test_validate_project_collection_success(self):
        """Test successful project collection validation."""
        config = NamingConfiguration(valid_project_suffixes={"docs", "notes", "scratchbook"})
        validator = PatternValidator(config)

        result = validator.validate_project_collection("my-project-docs")

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.SUCCESS
        assert result.detected_category == CollectionCategory.PROJECT
        assert result.detected_pattern == "project_pattern"

    def test_validate_project_collection_invalid_format(self):
        """Test project collection validation with invalid format."""
        result = self.validator.validate_project_collection("myproject")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.PROJECT_FORMAT_ERROR
        assert "must use format 'project-name-suffix'" in result.error_message

    def test_validate_project_collection_invalid_suffix(self):
        """Test project collection validation with invalid suffix."""
        config = NamingConfiguration(valid_project_suffixes={"docs", "notes"})
        validator = PatternValidator(config)

        result = validator.validate_project_collection("my-project-invalid")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.PROJECT_FORMAT_ERROR
        assert "not valid" in result.error_message
        assert len(result.suggested_names) > 0

    def test_validate_project_collection_reserved_prefix(self):
        """Test project collection validation with reserved prefixes."""
        result = self.validator.validate_project_collection("__system-docs")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert "cannot use reserved prefixes" in result.error_message

    def test_validate_global_collection_success(self):
        """Test successful global collection validation."""
        result = self.validator.validate_global_collection("algorithms")

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.SUCCESS
        assert result.detected_category == CollectionCategory.GLOBAL
        assert result.detected_pattern == "global_collection"

    def test_validate_global_collection_reserved_prefix(self):
        """Test global collection validation with reserved prefix."""
        result = self.validator.validate_global_collection("__global")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert "cannot use reserved prefixes" in result.error_message

    def test_validate_global_collection_project_like(self):
        """Test global collection validation with project-like name."""
        config = NamingConfiguration(valid_project_suffixes={"docs", "notes"})
        validator = PatternValidator(config)

        result = validator.validate_global_collection("something-docs")

        assert result.is_valid is True  # Still valid but with warning
        assert result.severity == ValidationSeverity.WARNING
        assert "resembles project collection format" in result.warning_message


class TestConflictDetector:
    """Test suite for ConflictDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = NamingConfiguration()
        self.pattern_validator = PatternValidator(self.config)
        self.detector = ConflictDetector(self.config, self.pattern_validator)

    def test_detect_direct_duplicate(self):
        """Test detection of direct duplicate names."""
        existing = ["existing_collection", "another_one"]

        result = self.detector.detect_conflicts("existing_collection", existing)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.DIRECT_DUPLICATE
        assert "already exists" in result.error_message
        assert "existing_collection" in result.conflicting_collections

    def test_detect_reserved_name(self):
        """Test detection of reserved name conflicts."""
        result = self.detector.detect_conflicts("system", [])

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.RESERVED_NAME
        assert "reserved collection name" in result.error_message

    def test_detect_memory_conflicts(self):
        """Test detection of memory collection conflicts."""
        existing = ["memory"]

        result = self.detector.detect_conflicts("memory", existing)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.MEMORY_CONFLICT
        assert "already exists" in result.error_message

    def test_detect_category_conflicts_library_vs_regular(self):
        """Test detection of conflicts between library and regular collections."""
        existing = ["_mylibrary"]

        result = self.detector.detect_conflicts("mylibrary", existing)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.CATEGORY_CONFLICT
        assert "_mylibrary" in result.conflicting_collections

    def test_detect_category_conflicts_regular_vs_library(self):
        """Test detection of conflicts between regular and library collections."""
        existing = ["mylibrary"]

        result = self.detector.detect_conflicts("_mylibrary", existing)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.CATEGORY_CONFLICT
        assert "mylibrary" in result.conflicting_collections

    def test_detect_prefix_abuse_system(self):
        """Test detection of system prefix abuse."""
        result = self.detector.detect_conflicts("__invalid", [], CollectionCategory.LIBRARY)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.SYSTEM_PREFIX_ABUSE
        assert "Only system collections can use '__'" in result.error_message

    def test_detect_prefix_abuse_library(self):
        """Test detection of library prefix abuse."""
        result = self.detector.detect_conflicts("_invalid", [], CollectionCategory.GLOBAL)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert result.conflict_type == ConflictType.LIBRARY_PREFIX_ABUSE
        assert "Only library collections can use '_'" in result.error_message

    def test_no_conflicts_valid_name(self):
        """Test no conflicts detected for valid unique name."""
        existing = ["other_collection", "_library_one"]

        result = self.detector.detect_conflicts("my_collection", existing)

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.SUCCESS


class TestCollectionNamingValidator:
    """Test suite for CollectionNamingValidator main class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CollectionNamingValidator()

    def test_validate_name_empty(self):
        """Test validation of empty name."""
        result = self.validator.validate_name("")

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert "cannot be empty" in result.error_message

    def test_validate_name_too_long(self):
        """Test validation of overly long name."""
        long_name = "a" * 101
        result = self.validator.validate_name(long_name)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert "cannot exceed 100 characters" in result.error_message

    def test_validate_name_memory_success(self):
        """Test successful memory collection validation."""
        result = self.validator.validate_name("memory", CollectionCategory.SYSTEM)

        assert result.is_valid is True
        assert result.severity == ValidationSeverity.SUCCESS
        assert result.detected_category == CollectionCategory.SYSTEM
        assert result.proposed_metadata is not None
        assert result.proposed_metadata.collection_category == CollectionCategory.SYSTEM

    def test_validate_name_auto_detect_system(self):
        """Test auto-detection of system collection."""
        result = self.validator.validate_name("__config")

        assert result.is_valid is True
        assert result.detected_category == CollectionCategory.SYSTEM
        assert result.proposed_metadata is not None

    def test_validate_name_auto_detect_library(self):
        """Test auto-detection of library collection."""
        result = self.validator.validate_name("_tools")

        assert result.is_valid is True
        assert result.detected_category == CollectionCategory.LIBRARY
        assert result.proposed_metadata is not None

    def test_validate_name_auto_detect_project(self):
        """Test auto-detection of project collection."""
        # Configure with known project suffixes
        config = NamingConfiguration(valid_project_suffixes={"docs", "notes"})
        validator = CollectionNamingValidator(config)

        result = validator.validate_name("my-project-docs")

        assert result.is_valid is True
        assert result.detected_category == CollectionCategory.PROJECT
        assert result.proposed_metadata is not None

    def test_validate_name_auto_detect_global(self):
        """Test auto-detection of global collection."""
        result = self.validator.validate_name("algorithms")

        assert result.is_valid is True
        assert result.detected_category == CollectionCategory.GLOBAL
        assert result.proposed_metadata is not None

    def test_validate_name_with_conflicts(self):
        """Test validation with existing collection conflicts."""
        existing = ["memory", "_tools", "my-project-docs"]

        result = self.validator.validate_name("memory", existing_collections=existing)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.ERROR
        assert "already exists" in result.error_message

    def test_check_conflicts_method(self):
        """Test the check_conflicts method specifically."""
        existing = ["existing_collection"]

        result = self.validator.check_conflicts("existing_collection", existing)

        assert result.is_valid is False
        assert result.conflict_type == ConflictType.DIRECT_DUPLICATE

    def test_get_name_suggestions_system(self):
        """Test name suggestion generation for system collections."""
        suggestions = self.validator.get_name_suggestions("config", CollectionCategory.SYSTEM)

        assert len(suggestions) > 0
        assert any(s.startswith("__") for s in suggestions)

    def test_get_name_suggestions_library(self):
        """Test name suggestion generation for library collections."""
        suggestions = self.validator.get_name_suggestions("tools", CollectionCategory.LIBRARY)

        assert len(suggestions) > 0
        assert any(s.startswith("_") for s in suggestions)

    def test_get_name_suggestions_project(self):
        """Test name suggestion generation for project collections."""
        config = NamingConfiguration(valid_project_suffixes={"docs", "notes"})
        validator = CollectionNamingValidator(config)

        suggestions = validator.get_name_suggestions("myproject", CollectionCategory.PROJECT)

        assert len(suggestions) > 0
        assert any("-docs" in s for s in suggestions)

    def test_get_name_suggestions_disabled(self):
        """Test that suggestion generation can be disabled."""
        config = NamingConfiguration(generate_suggestions=False)
        validator = CollectionNamingValidator(config)

        suggestions = validator.get_name_suggestions("invalid", CollectionCategory.SYSTEM)

        assert len(suggestions) == 0

    @pytest.mark.parametrize("category,expected_pattern", [
        (CollectionCategory.SYSTEM, "system_prefix"),
        (CollectionCategory.LIBRARY, "library_prefix"),
        (CollectionCategory.PROJECT, "project_pattern"),
        (CollectionCategory.GLOBAL, "global_collection"),
    ])
    def test_validate_by_category_patterns(self, category, expected_pattern):
        """Test validation patterns for each category."""
        # Set up valid names for each category
        valid_names = {
            CollectionCategory.SYSTEM: "__config",
            CollectionCategory.LIBRARY: "_tools",
            CollectionCategory.PROJECT: "test-docs",
            CollectionCategory.GLOBAL: "algorithms"
        }

        # Configure for project validation
        if category == CollectionCategory.PROJECT:
            config = NamingConfiguration(valid_project_suffixes={"docs"})
            validator = CollectionNamingValidator(config)
        else:
            validator = self.validator

        name = valid_names[category]
        result = validator._validate_by_category(name, category)

        assert result.is_valid is True
        assert result.detected_pattern == expected_pattern


class TestNamingConfiguration:
    """Test suite for NamingConfiguration class."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = NamingConfiguration()

        assert config.memory_collection_name == "memory"
        assert config.allow_custom_memory_names is False
        assert config.system_prefix == "__"
        assert config.library_prefix == "_"
        assert config.strict_validation is True
        assert config.generate_suggestions is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = NamingConfiguration(
            memory_collection_name="user_memory",
            allow_custom_memory_names=True,
            valid_project_suffixes={"docs", "notes", "code"},
            strict_validation=False,
            additional_reserved_names={"forbidden"}
        )

        assert config.memory_collection_name == "user_memory"
        assert config.allow_custom_memory_names is True
        assert "docs" in config.valid_project_suffixes
        assert config.strict_validation is False
        assert "forbidden" in config.additional_reserved_names


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_create_naming_validator(self):
        """Test the create_naming_validator convenience function."""
        validator = create_naming_validator(memory_collection_name="custom_memory")

        assert validator.config.memory_collection_name == "custom_memory"
        assert isinstance(validator, CollectionNamingValidator)

    def test_validate_collection_name_with_metadata(self):
        """Test the validate_collection_name_with_metadata function."""
        result = validate_collection_name_with_metadata("memory", CollectionCategory.SYSTEM)

        assert result.is_valid is True
        assert result.proposed_metadata is not None
        assert result.proposed_metadata.collection_category == CollectionCategory.SYSTEM

    def test_check_collection_conflicts_function(self):
        """Test the check_collection_conflicts function."""
        existing = ["existing"]
        result = check_collection_conflicts("existing", existing)

        assert result.is_valid is False
        assert result.conflict_type == ConflictType.DIRECT_DUPLICATE


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CollectionNamingValidator()

    def test_whitespace_handling(self):
        """Test proper handling of whitespace in names."""
        result = self.validator.validate_name("  memory  ", CollectionCategory.SYSTEM)

        assert result.is_valid is True
        # Name should be normalized

    def test_case_normalization(self):
        """Test case normalization in validation."""
        result = self.validator.validate_name("MEMORY", CollectionCategory.SYSTEM)

        assert result.is_valid is True
        # Should be normalized to lowercase

    def test_special_characters_in_name(self):
        """Test handling of special characters."""
        result = self.validator.validate_name("test@collection", CollectionCategory.GLOBAL)

        assert result.is_valid is False
        # Special characters should be rejected

    def test_boundary_length_validation(self):
        """Test boundary conditions for name length."""
        # Test exactly 100 characters
        exactly_100 = "a" * 100
        result = self.validator.validate_name(exactly_100, CollectionCategory.GLOBAL)
        assert result.is_valid is True

        # Test 101 characters
        too_long = "a" * 101
        result = self.validator.validate_name(too_long, CollectionCategory.GLOBAL)
        assert result.is_valid is False

    def test_single_character_names(self):
        """Test validation of single character names."""
        result = self.validator.validate_name("a", CollectionCategory.GLOBAL)
        assert result.is_valid is True

    def test_hyphen_edge_cases(self):
        """Test hyphen handling in various contexts."""
        # Test name starting with hyphen
        result = self.validator.validate_name("-invalid", CollectionCategory.GLOBAL)
        assert result.is_valid is False

        # Test name ending with hyphen
        result = self.validator.validate_name("invalid-", CollectionCategory.GLOBAL)
        assert result.is_valid is False

    def test_memory_collection_edge_cases(self):
        """Test edge cases specific to memory collections."""
        # Test memory with different cases
        result = self.validator.validate_name("Memory", CollectionCategory.SYSTEM)
        assert result.is_valid is True  # Should normalize to lowercase

        # Test memory-like names
        result = self.validator.validate_name("memories", CollectionCategory.GLOBAL)
        assert result.is_valid is True  # Different enough to be allowed

    def test_complex_project_names(self):
        """Test complex project collection names."""
        config = NamingConfiguration(valid_project_suffixes={"docs", "notes"})
        validator = CollectionNamingValidator(config)

        # Test multi-hyphen project name
        result = validator.validate_name("my-complex-project-name-docs", CollectionCategory.PROJECT)
        assert result.is_valid is True

        # Test single component project name
        result = validator.validate_name("simple-docs", CollectionCategory.PROJECT)
        assert result.is_valid is True

    def test_metadata_generation_edge_cases(self):
        """Test metadata generation for edge cases."""
        # Test memory collection metadata
        result = self.validator.validate_name("memory", CollectionCategory.SYSTEM)
        assert result.proposed_metadata is not None
        assert result.proposed_metadata.original_name_pattern is not None

        # Test library collection metadata
        result = self.validator.validate_name("_tools", CollectionCategory.LIBRARY)
        assert result.proposed_metadata is not None
        assert result.proposed_metadata.mcp_readonly is True

    def test_concurrent_validation(self):
        """Test that validation is thread-safe."""
        import threading

        results = []

        def validate_name():
            result = self.validator.validate_name("memory", CollectionCategory.SYSTEM)
            results.append(result.is_valid)

        threads = [threading.Thread(target=validate_name) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert all(results)
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])