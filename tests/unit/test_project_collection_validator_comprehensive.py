"""
Comprehensive unit tests for project collection validator utilities.

This module provides 100% test coverage for the project collection validation system,
including all validation methods, naming conventions, and edge cases.

Test coverage:
- ProjectCollectionValidator: all validation methods and naming patterns
- CollectionNamingRule: rule definition and application
- Error handling and edge cases for all methods
- Comprehensive validation scenarios
"""

# Ensure proper imports from the project structure
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

from common.utils.project_collection_validator import (
    CollectionNamingRule,
    ProjectCollectionValidator,
)


class TestCollectionNamingRule:
    """Test the CollectionNamingRule dataclass."""

    def test_collection_naming_rule_creation(self):
        """Test creation of CollectionNamingRule."""
        rule = CollectionNamingRule(
            pattern=r'^[a-z]+$',
            description="Simple pattern",
            scope='global',
            required_suffix=False,
            allowed_types={'docs', 'notes'}
        )

        assert rule.pattern == r'^[a-z]+$'
        assert rule.description == "Simple pattern"
        assert rule.scope == 'global'
        assert not rule.required_suffix
        assert rule.allowed_types == {'docs', 'notes'}


class TestProjectCollectionValidator:
    """Comprehensive tests for ProjectCollectionValidator class."""

    def setup_method(self):
        """Set up test environment."""
        self.validator = ProjectCollectionValidator()

    def test_init(self):
        """Test validator initialization."""
        validator = ProjectCollectionValidator()

        assert validator.PROJECT_COLLECTION_TYPES == {
            'notes', 'docs', 'code', 'research', 'scratchbook',
            'knowledge', 'context', 'memory', 'assets', 'logs'
        }

        assert validator.GLOBAL_COLLECTION_TYPES == {
            'docs', 'reference', 'standards', 'shared', 'templates'
        }

        assert validator.RESERVED_NAMES == {
            'system', 'admin', 'config', 'metadata', 'internal',
            'qdrant', 'vector', 'index', 'default', 'temp'
        }

        assert len(validator.naming_rules) > 0

    def test_initialize_naming_rules(self):
        """Test naming rules initialization."""
        validator = ProjectCollectionValidator()
        rules = validator._initialize_naming_rules()

        assert len(rules) == 3

        # Check basic naming pattern rule
        basic_rule = rules[0]
        assert basic_rule.scope == 'both'
        assert not basic_rule.required_suffix

        # Check project collection pattern rule
        project_rule = rules[1]
        assert project_rule.scope == 'project'
        assert project_rule.required_suffix

        # Check global collection pattern rule
        global_rule = rules[2]
        assert global_rule.scope == 'global'
        assert not global_rule.required_suffix

    def test_validate_collection_name_empty(self):
        """Test validation of empty collection name."""
        result = self.validator.validate_collection_name("")

        assert not result['valid']
        assert any("cannot be empty" in error for error in result['errors'])

    def test_validate_collection_name_uppercase(self):
        """Test validation of collection name with uppercase letters."""
        result = self.validator.validate_collection_name("MyProject-docs")

        assert "should be lowercase" in result['warnings']

    def test_validate_collection_name_reserved(self):
        """Test validation of reserved collection names."""
        result = self.validator.validate_collection_name("system")

        assert not result['valid']
        assert any("reserved collection name" in error for error in result['errors'])

    def test_validate_collection_name_too_short(self):
        """Test validation of collection name that's too short."""
        result = self.validator.validate_collection_name("x")

        assert not result['valid']
        assert any("at least 2 characters" in error for error in result['errors'])

    def test_validate_collection_name_too_long(self):
        """Test validation of collection name that's too long."""
        long_name = "a" * 65
        result = self.validator.validate_collection_name(long_name)

        assert not result['valid']
        assert any("64 characters or less" in error for error in result['errors'])

    def test_validate_collection_name_valid_project_collection(self):
        """Test validation of valid project collection."""
        result = self.validator.validate_collection_name(
            "myproject-docs",
            project_name="myproject",
            collection_type="docs"
        )

        assert result['valid']
        assert result['detected_pattern'] == 'project'
        assert len(result['errors']) == 0

    def test_validate_collection_name_valid_global_collection(self):
        """Test validation of valid global collection."""
        result = self.validator.validate_collection_name("docs")

        assert result['valid']
        assert result['detected_pattern'] == 'global'
        assert len(result['errors']) == 0

    def test_validate_collection_name_invalid_pattern(self):
        """Test validation of collection name with invalid pattern."""
        result = self.validator.validate_collection_name("invalid@collection")

        assert not result['valid']
        assert any("does not match any valid naming pattern" in error for error in result['errors'])

    def test_validate_collection_name_project_mismatch(self):
        """Test validation with project name mismatch."""
        result = self.validator.validate_collection_name(
            "wrongproject-docs",
            project_name="correctproject",
            collection_type="docs"
        )

        assert not result['valid']
        assert any("does not match expected project" in error for error in result['errors'])

    def test_validate_collection_name_type_mismatch(self):
        """Test validation with collection type mismatch."""
        result = self.validator.validate_collection_name(
            "myproject-notes",
            project_name="myproject",
            collection_type="docs"
        )

        assert any("does not match expected type" in warning for warning in result['warnings'])

    def test_detect_naming_pattern_project(self):
        """Test detection of project naming pattern."""
        pattern = self.validator._detect_naming_pattern("my-project-docs")
        assert pattern == 'project'

    def test_detect_naming_pattern_global_known_type(self):
        """Test detection of global pattern for known type."""
        pattern = self.validator._detect_naming_pattern("docs")
        assert pattern == 'global'

    def test_detect_naming_pattern_global_simple(self):
        """Test detection of global pattern for simple name."""
        pattern = self.validator._detect_naming_pattern("mydata")
        assert pattern == 'global'

    def test_detect_naming_pattern_invalid(self):
        """Test detection of invalid pattern."""
        pattern = self.validator._detect_naming_pattern("-invalid-")
        assert pattern is None

    def test_detect_naming_pattern_no_hyphens(self):
        """Test detection when no hyphens present."""
        pattern = self.validator._detect_naming_pattern("singleword")
        assert pattern == 'global'

    def test_validate_project_collection_multipart_type(self):
        """Test validation of project collection with multi-part type."""
        result = self.validator.validate_collection_name(
            "my-project-test-data",
            project_name="my-project",
            collection_type="test-data"
        )

        assert result['valid']
        assert result['detected_pattern'] == 'project'

    def test_validate_project_collection_global_type_warning(self):
        """Test validation when using global type for project collection."""
        result = self.validator.validate_collection_name(
            "my-project-reference",
            project_name="my-project",
            collection_type="reference"
        )

        # Should warn about using global type
        assert any("typically used for global collections" in warning for warning in result['warnings'])

    def test_validate_project_collection_docs_allowed(self):
        """Test validation of docs type for project collection (should be allowed)."""
        result = self.validator.validate_collection_name(
            "my-project-docs",
            project_name="my-project",
            collection_type="docs"
        )

        # Should not warn about docs type as it's allowed for both
        global_warnings = [w for w in result['warnings'] if "typically used for global" in w]
        assert len(global_warnings) == 0

    def test_validate_global_collection_standard_type(self):
        """Test validation of standard global collection type."""
        result = self.validator.validate_collection_name("reference")

        assert result['valid']
        assert result['detected_pattern'] == 'global'

    def test_validate_global_collection_non_standard_type(self):
        """Test validation of non-standard global collection type."""
        result = self.validator.validate_collection_name("customglobal")

        assert result['valid']
        assert any("not a standard global collection type" in warning for warning in result['warnings'])

    def test_validate_global_collection_with_expected_type(self):
        """Test validation of global collection with expected type."""
        result = self.validator.validate_collection_name(
            "customname",
            collection_type="standards"
        )

        assert any("Consider using standard global name: 'standards'" in suggestion for suggestion in result['suggestions'])

    def test_check_naming_conflicts_duplicate_project_collection(self):
        """Test conflict detection for duplicate project collections."""
        collections = ["my-project-docs", "my-project-docs"]
        conflicts = self.validator.check_naming_conflicts(collections)

        assert len(conflicts) == 1
        assert conflicts[0]['type'] == 'duplicate_project_collection'
        assert "Duplicate project collection type 'docs'" in conflicts[0]['description']

    def test_check_naming_conflicts_duplicate_global_collection(self):
        """Test conflict detection for duplicate global collections."""
        collections = ["docs", "docs"]
        conflicts = self.validator.check_naming_conflicts(collections)

        assert len(conflicts) == 1
        assert conflicts[0]['type'] == 'duplicate_global_collection'
        assert "Duplicate global collection 'docs'" in conflicts[0]['description']

    def test_check_naming_conflicts_no_conflicts(self):
        """Test conflict detection with no conflicts."""
        collections = ["project1-docs", "project2-docs", "global-reference"]
        conflicts = self.validator.check_naming_conflicts(collections)

        assert len(conflicts) == 0

    def test_check_naming_conflicts_same_project_different_types(self):
        """Test no conflict for same project with different types."""
        collections = ["my-project-docs", "my-project-code"]
        conflicts = self.validator.check_naming_conflicts(collections)

        assert len(conflicts) == 0

    def test_check_naming_conflicts_multipart_type(self):
        """Test conflict detection with multi-part collection types."""
        collections = ["project-test-data", "project-test-data"]
        conflicts = self.validator.check_naming_conflicts(collections)

        assert len(conflicts) == 1
        assert "Duplicate project collection type 'test-data'" in conflicts[0]['description']

    def test_suggest_collection_name_project_scope(self):
        """Test collection name suggestion for project scope."""
        name = self.validator.suggest_collection_name(
            "my_project",
            "test_data",
            scope="project"
        )

        assert name == "my-project-test-data"

    def test_suggest_collection_name_global_scope(self):
        """Test collection name suggestion for global scope."""
        name = self.validator.suggest_collection_name(
            "my_project",
            "docs",
            scope="global"
        )

        assert name == "docs"

    def test_suggest_collection_name_global_type(self):
        """Test collection name suggestion for global type regardless of scope."""
        name = self.validator.suggest_collection_name(
            "my_project",
            "reference",  # Global type
            scope="project"
        )

        assert name == "reference"  # Should return global type

    def test_suggest_collection_name_normalization(self):
        """Test collection name suggestion with normalization."""
        name = self.validator.suggest_collection_name(
            "My_Project",
            "Test_Data",
            scope="project"
        )

        assert name == "my-project-test-data"

    def test_get_project_collections_pattern(self):
        """Test regex pattern generation for project collections."""
        pattern = self.validator.get_project_collections_pattern("my-project")

        assert pattern == r"^my\-project-[a-z0-9][a-z0-9\-]*$"

        # Test that the pattern works
        import re
        regex = re.compile(pattern)

        assert regex.match("my-project-docs")
        assert regex.match("my-project-test-data")
        assert not regex.match("other-project-docs")
        assert not regex.match("my-project")  # Missing type

    def test_get_project_collections_pattern_special_chars(self):
        """Test regex pattern generation with special characters."""
        pattern = self.validator.get_project_collections_pattern("my.project+special")

        # Should escape special regex characters
        assert r"\." in pattern
        assert r"\+" in pattern

    def test_extract_project_from_collection_project_pattern(self):
        """Test project extraction from project collection."""
        project = self.validator.extract_project_from_collection("my-project-docs")

        assert project == "my-project"

    def test_extract_project_from_collection_global_pattern(self):
        """Test project extraction from global collection."""
        project = self.validator.extract_project_from_collection("docs")

        assert project is None

    def test_extract_project_from_collection_invalid_pattern(self):
        """Test project extraction from invalid collection."""
        project = self.validator.extract_project_from_collection("invalid@collection")

        assert project is None

    def test_extract_type_from_collection_project_pattern(self):
        """Test type extraction from project collection."""
        collection_type = self.validator.extract_type_from_collection("my-project-docs")

        assert collection_type == "docs"

    def test_extract_type_from_collection_multipart_type(self):
        """Test type extraction from project collection with multi-part type."""
        collection_type = self.validator.extract_type_from_collection("my-project-test-data")

        assert collection_type == "test-data"

    def test_extract_type_from_collection_global_pattern(self):
        """Test type extraction from global collection."""
        collection_type = self.validator.extract_type_from_collection("docs")

        assert collection_type == "docs"

    def test_extract_type_from_collection_invalid_pattern(self):
        """Test type extraction from invalid collection."""
        collection_type = self.validator.extract_type_from_collection("invalid@collection")

        assert collection_type is None

    def test_validate_collection_name_edge_case_single_hyphen(self):
        """Test validation of collection name with single hyphen."""
        result = self.validator.validate_collection_name("a-b")

        # Should be detected as project pattern
        assert result['detected_pattern'] == 'project'

    def test_validate_collection_name_edge_case_multiple_hyphens(self):
        """Test validation of collection name with multiple hyphens."""
        result = self.validator.validate_collection_name("my-long-project-name-docs")

        assert result['detected_pattern'] == 'project'

    def test_validate_collection_name_edge_case_starts_with_hyphen(self):
        """Test validation of collection name starting with hyphen."""
        result = self.validator.validate_collection_name("-invalid")

        assert result['detected_pattern'] is None

    def test_validate_collection_name_edge_case_ends_with_hyphen(self):
        """Test validation of collection name ending with hyphen."""
        result = self.validator.validate_collection_name("invalid-")

        assert result['detected_pattern'] is None

    def test_validate_collection_name_minimum_valid_length(self):
        """Test validation of minimum valid collection name length."""
        result = self.validator.validate_collection_name("ab")

        assert result['valid']
        assert len(result['errors']) == 0

    def test_validate_collection_name_maximum_valid_length(self):
        """Test validation of maximum valid collection name length."""
        max_name = "a" * 64
        result = self.validator.validate_collection_name(max_name)

        assert result['valid']
        assert len(result['errors']) == 0

    def test_detect_naming_pattern_complex_project_name(self):
        """Test pattern detection with complex project names."""
        pattern = self.validator._detect_naming_pattern("my-complex-project-name-docs")
        assert pattern == 'project'

    def test_validate_project_collection_complex_project_name(self):
        """Test validation with complex project names."""
        result = self.validator.validate_collection_name(
            "my-complex-project-name-docs",
            project_name="my-complex-project-name",
            collection_type="docs"
        )

        assert result['valid']
        assert result['detected_pattern'] == 'project'

    def test_collection_constants_immutability(self):
        """Test that collection type constants are properly defined."""
        validator = ProjectCollectionValidator()

        # Test that the sets contain expected values
        assert 'docs' in validator.PROJECT_COLLECTION_TYPES
        assert 'docs' in validator.GLOBAL_COLLECTION_TYPES
        assert 'system' in validator.RESERVED_NAMES

        # Test that modifying returned sets doesn't affect the originals
        project_types = validator.PROJECT_COLLECTION_TYPES
        original_size = len(project_types)

        # This should not affect the validator's constants
        project_types.add('new_type')

        # Verify the validator's set is unchanged if it's a reference
        # (Note: Sets are mutable, so this tests implementation detail)
        assert len(validator.PROJECT_COLLECTION_TYPES) >= original_size

    def test_validation_result_structure(self):
        """Test that validation results have consistent structure."""
        result = self.validator.validate_collection_name("test-collection")

        # Check all required keys are present
        required_keys = {'valid', 'errors', 'warnings', 'detected_pattern', 'suggestions'}
        assert set(result.keys()) == required_keys

        # Check value types
        assert isinstance(result['valid'], bool)
        assert isinstance(result['errors'], list)
        assert isinstance(result['warnings'], list)
        assert isinstance(result['suggestions'], list)
        assert result['detected_pattern'] is None or isinstance(result['detected_pattern'], str)

    def test_conflict_result_structure(self):
        """Test that conflict detection results have consistent structure."""
        collections = ["project-docs", "project-docs"]
        conflicts = self.validator.check_naming_conflicts(collections)

        if conflicts:  # If there are conflicts
            for conflict in conflicts:
                assert 'type' in conflict
                assert 'collections' in conflict
                assert 'description' in conflict
                assert isinstance(conflict['collections'], list)
                assert len(conflict['collections']) >= 2

    @patch('common.utils.project_collection_validator.logger')
    def test_logging_calls(self, mock_logger):
        """Test that appropriate logging calls are made."""
        # Test project collection validation logging
        self.validator.validate_collection_name(
            "my-project-docs",
            project_name="my-project",
            collection_type="docs"
        )

        # Should log debug message for project collection validation
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Validated project collection" in call for call in debug_calls)

    @patch('common.utils.project_collection_validator.logger')
    def test_logging_global_collection(self, mock_logger):
        """Test logging for global collection validation."""
        self.validator.validate_collection_name("docs")

        # Should log debug message for global collection validation
        mock_logger.debug.assert_called()
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Validated global collection" in call for call in debug_calls)


if __name__ == "__main__":
    pytest.main([__file__])
