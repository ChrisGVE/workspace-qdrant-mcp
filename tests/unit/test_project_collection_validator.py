"""
Unit tests for project collection naming validator.

Tests the ProjectCollectionValidator that ensures proper collection naming
conventions for multi-tenant environments.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from unittest.mock import patch

import pytest
from workspace_qdrant_mcp.utils.project_collection_validator import (
    CollectionNamingRule,
    ProjectCollectionValidator,
)


class TestProjectCollectionValidator:
    """Test ProjectCollectionValidator class."""

    def setup_method(self):
        """Set up test instance."""
        self.validator = ProjectCollectionValidator()

    def test_validate_project_collection_valid(self):
        """Test validation of valid project collection names."""
        test_cases = [
            ("myproject-notes", "myproject", "notes"),
            ("workspace-qdrant-mcp-docs", "workspace-qdrant-mcp", "docs"),
            ("test-project-123-research", "test-project-123", "research"),
            ("simple-scratchbook", "simple", "scratchbook"),
        ]

        for collection_name, project_name, collection_type in test_cases:
            result = self.validator.validate_collection_name(
                collection_name, project_name, collection_type
            )
            assert result['valid'], f"Expected {collection_name} to be valid: {result['errors']}"
            assert result['detected_pattern'] == 'project'

    def test_validate_global_collection_valid(self):
        """Test validation of valid global collection names."""
        test_cases = [
            ("docs", None, "docs"),
            ("reference", None, "reference"),
            ("standards", None, "standards"),
            ("shared", None, "shared"),
            ("templates", None, "templates"),
        ]

        for collection_name, project_name, collection_type in test_cases:
            result = self.validator.validate_collection_name(
                collection_name, project_name, collection_type
            )
            assert result['valid'], f"Expected {collection_name} to be valid: {result['errors']}"
            assert result['detected_pattern'] == 'global'

    def test_validate_collection_invalid_reserved_names(self):
        """Test validation rejects reserved collection names."""
        reserved_names = ["system", "admin", "config", "qdrant", "internal"]

        for name in reserved_names:
            result = self.validator.validate_collection_name(name)
            assert not result['valid'], f"Expected {name} to be invalid"
            assert any("reserved" in error.lower() for error in result['errors'])

    def test_validate_collection_invalid_patterns(self):
        """Test validation of invalid collection name patterns."""
        invalid_cases = [
            ("", "Empty name"),
            ("a", "Too short"),
            ("-project-notes", "Starts with hyphen"),
            ("project-notes-", "Ends with hyphen"),
            ("Project-Notes", "Mixed case"),
            ("project..notes", "Invalid characters"),
        ]

        for collection_name, description in invalid_cases:
            result = self.validator.validate_collection_name(collection_name)
            assert not result['valid'], f"Expected '{collection_name}' to be invalid ({description}): {result}"

    def test_validate_project_mismatch(self):
        """Test validation when project name doesn't match collection prefix."""
        result = self.validator.validate_collection_name(
            "project-a-notes", "project-b", "notes"
        )
        assert not result['valid']
        assert any("project" in error.lower() for error in result['errors'])

    def test_check_naming_conflicts(self):
        """Test detection of naming conflicts."""
        # Test duplicate project collections
        collections = [
            "myproject-notes",
            "myproject-notes",  # Duplicate
            "otherproject-docs",
            "docs",  # Global
            "docs",  # Duplicate global
        ]

        conflicts = self.validator.check_naming_conflicts(collections)

        # Should find 2 conflicts: duplicate project collection and duplicate global
        assert len(conflicts) == 2

        conflict_types = [c['type'] for c in conflicts]
        assert 'duplicate_project_collection' in conflict_types
        assert 'duplicate_global_collection' in conflict_types

    def test_suggest_collection_name(self):
        """Test collection name suggestions."""
        # Test project scope
        suggested = self.validator.suggest_collection_name("my_project", "notes", "project")
        assert suggested == "my-project-notes"

        # Test global scope
        suggested = self.validator.suggest_collection_name("any_project", "docs", "global")
        assert suggested == "docs"

        # Test global type forced by type
        suggested = self.validator.suggest_collection_name("myproject", "reference", "project")
        assert suggested == "reference"

    def test_extract_project_from_collection(self):
        """Test extraction of project name from collection name."""
        test_cases = [
            ("myproject-notes", "myproject"),
            ("workspace-qdrant-mcp-docs", "workspace-qdrant-mcp"),
            ("simple-research", "simple"),
            ("docs", None),  # Global collection
            ("reference", None),  # Global collection
        ]

        for collection_name, expected_project in test_cases:
            result = self.validator.extract_project_from_collection(collection_name)
            assert result == expected_project, f"Expected project '{expected_project}' from '{collection_name}', got '{result}'"

    def test_extract_type_from_collection(self):
        """Test extraction of collection type from collection name."""
        test_cases = [
            ("myproject-notes", "notes"),
            ("workspace-qdrant-mcp-docs", "docs"),
            ("complex-project-multi-word-type", "multi-word-type"),
            ("docs", "docs"),  # Global collection
            ("reference", "reference"),  # Global collection
        ]

        for collection_name, expected_type in test_cases:
            result = self.validator.extract_type_from_collection(collection_name)
            assert result == expected_type, f"Expected type '{expected_type}' from '{collection_name}', got '{result}'"

    def test_get_project_collections_pattern(self):
        """Test generation of regex pattern for project collections."""
        pattern = self.validator.get_project_collections_pattern("myproject")

        import re
        regex = re.compile(pattern)

        # Should match project collections
        assert regex.match("myproject-notes")
        assert regex.match("myproject-docs")
        assert regex.match("myproject-multi-word")

        # Should not match other collections
        assert not regex.match("otherproject-notes")
        assert not regex.match("docs")
        assert not regex.match("myproject")
        assert not regex.match("myproject-")

    def test_integration_with_project_detection(self):
        """Test integration with project detection system."""
        # This test validates that the validator works with real project names
        # from the project detection system

        # Simulate project detection results
        project_name = "workspace-qdrant-mcp"
        collection_types = ["notes", "docs", "scratchbook", "research"]

        for collection_type in collection_types:
            suggested_name = self.validator.suggest_collection_name(
                project_name, collection_type, "project"
            )

            # Validate the suggested name
            result = self.validator.validate_collection_name(
                suggested_name, project_name, collection_type
            )

            assert result['valid'], f"Suggested name '{suggested_name}' should be valid: {result}"
            assert result['detected_pattern'] == 'project'

    def test_multi_tenant_isolation_validation(self):
        """Test that the validator supports multi-tenant isolation requirements."""
        # Simulate multiple tenants with their own projects
        tenants = [
            ("user1", "project-alpha", ["notes", "docs"]),
            ("user2", "project-beta", ["notes", "research"]),
            ("shared", None, ["docs", "reference", "standards"])  # Global collections
        ]

        all_collections = []

        for tenant, project, types in tenants:
            for collection_type in types:
                if project:
                    # Project-scoped collection
                    collection_name = f"{project}-{collection_type}"
                else:
                    # Global collection
                    collection_name = collection_type

                all_collections.append(collection_name)

                # Validate each collection
                result = self.validator.validate_collection_name(
                    collection_name, project, collection_type
                )
                assert result['valid'], f"Collection '{collection_name}' should be valid for tenant '{tenant}'"

        # Check for conflicts
        conflicts = self.validator.check_naming_conflicts(all_collections)

        # Should have no conflicts because each tenant has different project names
        assert len(conflicts) == 0, f"Expected no conflicts, but found: {conflicts}"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very long collection names
        long_name = "a" * 65  # Too long
        result = self.validator.validate_collection_name(long_name)
        assert not result['valid']
        assert any("64 characters" in error for error in result['errors'])

        # Maximum length (should be valid)
        max_name = "a" * 64
        result = self.validator.validate_collection_name(max_name)
        assert result['valid']

        # Special characters
        special_cases = ["project@notes", "project#notes", "project notes", "project/notes"]
        for case in special_cases:
            result = self.validator.validate_collection_name(case)
            assert not result['valid'], f"Expected '{case}' to be invalid"

    def test_case_sensitivity_warnings(self):
        """Test that mixed case generates appropriate warnings."""
        result = self.validator.validate_collection_name("MyProject-Notes")
        assert len(result['warnings']) > 0
        assert any("lowercase" in warning.lower() for warning in result['warnings'])

    def test_collection_type_validation(self):
        """Test validation of collection types against allowed types."""
        # Valid project collection types
        valid_types = ["notes", "docs", "code", "research", "scratchbook"]

        for collection_type in valid_types:
            collection_name = f"myproject-{collection_type}"
            result = self.validator.validate_collection_name(
                collection_name, "myproject", collection_type
            )
            assert result['valid'], f"Collection type '{collection_type}' should be valid"

        # Global types used in project context should generate warnings
        global_types = ["reference", "standards", "templates"]

        for collection_type in global_types:
            if collection_type not in {"docs"}:  # docs is allowed in both contexts
                collection_name = f"myproject-{collection_type}"
                result = self.validator.validate_collection_name(
                    collection_name, "myproject", collection_type
                )
                # Should be valid but with warnings
                assert result['valid'], f"Collection '{collection_name}' should be valid"
                assert len(result['warnings']) > 0, "Should have warnings for global type in project context"
