"""
Unit tests for the multi-tenant metadata schema.

This test suite provides comprehensive validation of the metadata schema
implementation including field validation, factory methods, serialization,
and integration with existing collection systems.

Test Categories:
    - Schema creation and validation
    - Factory methods for different collection types
    - Serialization and deserialization
    - Field validation and constraints
    - Enum value handling
    - Backward compatibility
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from src.python.common.core.metadata_schema import (
    MultiTenantMetadataSchema,
    CollectionCategory,
    WorkspaceScope,
    AccessLevel,
    METADATA_SCHEMA_VERSION
)


class TestMultiTenantMetadataSchema:
    """Test suite for MultiTenantMetadataSchema class."""

    def test_create_for_project_basic(self):
        """Test basic project collection metadata creation."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="test_project",
            collection_type="docs"
        )

        assert metadata.project_name == "test_project"
        assert metadata.collection_type == "docs"
        assert metadata.tenant_namespace == "test_project.docs"
        assert metadata.collection_category == CollectionCategory.PROJECT
        assert metadata.workspace_scope == WorkspaceScope.PROJECT
        assert metadata.access_level == AccessLevel.PRIVATE
        assert metadata.mcp_readonly is False
        assert metadata.cli_writable is True
        assert metadata.created_by == "user"
        assert len(metadata.project_id) == 12
        assert all(c in "0123456789abcdef" for c in metadata.project_id)

    def test_create_for_project_with_options(self):
        """Test project collection creation with custom options."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="advanced_project",
            collection_type="notes",
            created_by="admin",
            access_level=AccessLevel.SHARED,
            tags=["important", "docs"],
            category="documentation",
            priority=5
        )

        assert metadata.created_by == "admin"
        assert metadata.access_level == AccessLevel.SHARED
        assert metadata.tags == ["important", "docs"]
        assert metadata.category == "documentation"
        assert metadata.priority == 5

    def test_create_for_system(self):
        """Test system collection metadata creation."""
        metadata = MultiTenantMetadataSchema.create_for_system(
            collection_name="__user_preferences",
            collection_type="memory_collection"
        )

        assert metadata.project_name == "system"
        assert metadata.collection_type == "memory_collection"
        assert metadata.tenant_namespace == "system.memory_collection"
        assert metadata.collection_category == CollectionCategory.SYSTEM
        assert metadata.workspace_scope == WorkspaceScope.GLOBAL
        assert metadata.access_level == AccessLevel.PRIVATE
        assert metadata.mcp_readonly is False
        assert metadata.cli_writable is True
        assert metadata.is_reserved_name is True
        assert metadata.naming_pattern == "system_prefix"
        assert metadata.original_name_pattern == "__user_preferences"
        assert metadata.legacy_collection_name == "__user_preferences"

    def test_create_for_system_invalid_name(self):
        """Test system collection creation with invalid name."""
        with pytest.raises(ValueError, match="System collection names must start with '__'"):
            MultiTenantMetadataSchema.create_for_system(
                collection_name="invalid_name",
                collection_type="memory_collection"
            )

    def test_create_for_library(self):
        """Test library collection metadata creation."""
        metadata = MultiTenantMetadataSchema.create_for_library(
            collection_name="_code_refs",
            collection_type="code_collection"
        )

        assert metadata.project_name == "library"
        assert metadata.collection_type == "code_collection"
        assert metadata.tenant_namespace == "library.code_collection"
        assert metadata.collection_category == CollectionCategory.LIBRARY
        assert metadata.workspace_scope == WorkspaceScope.LIBRARY
        assert metadata.access_level == AccessLevel.SHARED
        assert metadata.mcp_readonly is True
        assert metadata.cli_writable is True
        assert metadata.is_reserved_name is True
        assert metadata.naming_pattern == "library_prefix"
        assert metadata.original_name_pattern == "_code_refs"

    def test_create_for_library_invalid_name(self):
        """Test library collection creation with invalid names."""
        # Missing prefix
        with pytest.raises(ValueError, match="Library collection names must start with '_' but not '__'"):
            MultiTenantMetadataSchema.create_for_library(
                collection_name="invalid_name",
                collection_type="code_collection"
            )

        # System prefix
        with pytest.raises(ValueError, match="Library collection names must start with '_' but not '__'"):
            MultiTenantMetadataSchema.create_for_library(
                collection_name="__system_name",
                collection_type="code_collection"
            )

    def test_create_for_global(self):
        """Test global collection metadata creation."""
        metadata = MultiTenantMetadataSchema.create_for_global(
            collection_name="algorithms",
            collection_type="global"
        )

        assert metadata.project_name == "global"
        assert metadata.collection_type == "global"
        assert metadata.tenant_namespace == "global.global"
        assert metadata.collection_category == CollectionCategory.GLOBAL
        assert metadata.workspace_scope == WorkspaceScope.GLOBAL
        assert metadata.access_level == AccessLevel.PUBLIC
        assert metadata.mcp_readonly is False
        assert metadata.cli_writable is True
        assert metadata.is_reserved_name is False
        assert metadata.naming_pattern == "global_collection"

    def test_project_id_generation_stable(self):
        """Test that project ID generation is stable."""
        metadata1 = MultiTenantMetadataSchema.create_for_project("test_project", "docs")
        metadata2 = MultiTenantMetadataSchema.create_for_project("test_project", "notes")

        # Same project should generate same project ID
        assert metadata1.project_id == metadata2.project_id

        # Different projects should generate different IDs
        metadata3 = MultiTenantMetadataSchema.create_for_project("other_project", "docs")
        assert metadata1.project_id != metadata3.project_id

    def test_field_normalization(self):
        """Test automatic field normalization."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="My-Project",  # Should be normalized
            collection_type="My-Docs",  # Should be normalized
            created_by="User Name"     # Should be normalized
        )

        assert metadata.project_name == "my_project"
        assert metadata.collection_type == "my_docs"
        assert metadata.tenant_namespace == "my_project.my_docs"
        assert metadata.created_by == "user name"

    def test_field_constraints_validation(self):
        """Test field constraint validation."""
        # Test project name length constraint
        with pytest.raises(ValueError, match="project_name exceeds maximum length"):
            MultiTenantMetadataSchema(
                project_id="a1b2c3d4e5f6",
                project_name="x" * 150,  # Too long
                tenant_namespace="test.docs",
                collection_type="docs",
                collection_category=CollectionCategory.PROJECT,
                workspace_scope=WorkspaceScope.PROJECT
            )

        # Test invalid project ID format
        with pytest.raises(ValueError, match="project_id must be exactly 12 hexadecimal characters"):
            MultiTenantMetadataSchema(
                project_id="invalid",
                project_name="test",
                tenant_namespace="test.docs",
                collection_type="docs",
                collection_category=CollectionCategory.PROJECT,
                workspace_scope=WorkspaceScope.PROJECT
            )

        # Test invalid priority range
        with pytest.raises(ValueError, match="priority must be between 1 and 5"):
            MultiTenantMetadataSchema(
                project_id="a1b2c3d4e5f6",
                project_name="test",
                tenant_namespace="test.docs",
                collection_type="docs",
                collection_category=CollectionCategory.PROJECT,
                workspace_scope=WorkspaceScope.PROJECT,
                priority=10
            )

    def test_to_qdrant_payload(self):
        """Test conversion to Qdrant payload format."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="test_project",
            collection_type="docs",
            tags=["test", "docs"],
            priority=4
        )

        payload = metadata.to_qdrant_payload()

        # Check all required fields are present
        required_fields = [
            "project_id", "project_name", "tenant_namespace",
            "collection_type", "collection_category", "workspace_scope",
            "access_level", "mcp_readonly", "cli_writable", "created_by"
        ]

        for field in required_fields:
            assert field in payload

        # Check enum values are converted to strings
        assert payload["collection_category"] == "project"
        assert payload["workspace_scope"] == "project"
        assert payload["access_level"] == "private"

        # Check data types
        assert isinstance(payload["mcp_readonly"], bool)
        assert isinstance(payload["cli_writable"], bool)
        assert isinstance(payload["is_reserved_name"], bool)
        assert isinstance(payload["tags"], list)
        assert isinstance(payload["priority"], int)
        assert isinstance(payload["version"], int)

    def test_from_qdrant_payload(self):
        """Test creation from Qdrant payload."""
        original_metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="test_project",
            collection_type="docs",
            tags=["test", "docs"],
            priority=4
        )

        payload = original_metadata.to_qdrant_payload()
        reconstructed_metadata = MultiTenantMetadataSchema.from_qdrant_payload(payload)

        # Check core fields match
        assert reconstructed_metadata.project_id == original_metadata.project_id
        assert reconstructed_metadata.project_name == original_metadata.project_name
        assert reconstructed_metadata.tenant_namespace == original_metadata.tenant_namespace
        assert reconstructed_metadata.collection_type == original_metadata.collection_type
        assert reconstructed_metadata.collection_category == original_metadata.collection_category
        assert reconstructed_metadata.workspace_scope == original_metadata.workspace_scope
        assert reconstructed_metadata.access_level == original_metadata.access_level
        assert reconstructed_metadata.tags == original_metadata.tags
        assert reconstructed_metadata.priority == original_metadata.priority

    def test_get_indexed_fields(self):
        """Test retrieval of indexed fields list."""
        metadata = MultiTenantMetadataSchema.create_for_project("test", "docs")
        indexed_fields = metadata.get_indexed_fields()

        expected_indexed_fields = [
            "project_id", "project_name", "tenant_namespace",
            "collection_type", "collection_category", "workspace_scope",
            "access_level", "mcp_readonly", "cli_writable", "created_by",
            "is_reserved_name", "naming_pattern"
        ]

        assert set(indexed_fields) == set(expected_indexed_fields)

    def test_filtering_methods(self):
        """Test metadata filtering methods."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="test_project",
            collection_type="docs"
        )

        # Test project filter matching
        assert metadata.matches_project_filter(metadata.project_id) is True
        assert metadata.matches_project_filter("different_id") is False

        # Test collection type filter matching
        assert metadata.matches_collection_type_filter(["docs", "notes"]) is True
        assert metadata.matches_collection_type_filter(["notes", "scratchbook"]) is False
        assert metadata.matches_collection_type_filter([]) is False

    def test_access_control_methods(self):
        """Test access control check methods."""
        # Test project collection
        project_metadata = MultiTenantMetadataSchema.create_for_project("test", "docs")
        assert project_metadata.is_accessible_by_mcp() is True
        assert project_metadata.is_writable_by_mcp() is True
        assert project_metadata.is_globally_searchable() is False  # Private by default

        # Test system collection
        system_metadata = MultiTenantMetadataSchema.create_for_system("__user_prefs", "memory")
        assert system_metadata.is_accessible_by_mcp() is True
        assert system_metadata.is_writable_by_mcp() is True  # CLI writable but not necessarily MCP
        assert system_metadata.is_globally_searchable() is False  # System collections not searchable

        # Test library collection
        library_metadata = MultiTenantMetadataSchema.create_for_library("_code_refs", "code")
        assert library_metadata.is_accessible_by_mcp() is True
        assert library_metadata.is_writable_by_mcp() is False  # MCP readonly
        assert library_metadata.is_globally_searchable() is True  # Library collections searchable

        # Test global collection
        global_metadata = MultiTenantMetadataSchema.create_for_global("algorithms", "global")
        assert global_metadata.is_accessible_by_mcp() is True
        assert global_metadata.is_writable_by_mcp() is True
        assert global_metadata.is_globally_searchable() is True  # Global collections searchable

    def test_shared_project_collection_searchable(self):
        """Test that shared project collections are globally searchable."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="test_project",
            collection_type="docs",
            access_level=AccessLevel.SHARED
        )
        assert metadata.is_globally_searchable() is True

    def test_update_timestamp(self):
        """Test timestamp update functionality."""
        metadata = MultiTenantMetadataSchema.create_for_project("test", "docs")
        original_updated_at = metadata.updated_at
        original_version = metadata.version

        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.01)

        metadata.update_timestamp()

        assert metadata.updated_at != original_updated_at
        assert metadata.version == original_version + 1

    def test_tenant_namespace_consistency(self):
        """Test that tenant namespace is always consistent with project and collection type."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="workspace_qdrant_mcp",
            collection_type="documentation"
        )

        assert metadata.tenant_namespace == "workspace_qdrant_mcp.documentation"

        # Test with system collection
        system_metadata = MultiTenantMetadataSchema.create_for_system(
            "__user_config",
            "memory_collection"
        )
        assert system_metadata.tenant_namespace == "system.memory_collection"

    def test_migration_fields(self):
        """Test migration-related fields are properly set."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="migrated_project",
            collection_type="docs",
            created_by="migration"
        )

        assert metadata.migration_source == "metadata_based"
        assert metadata.compatibility_version == METADATA_SCHEMA_VERSION
        assert metadata.legacy_collection_name is None  # Not set for new collections

        # Test with system collection that has legacy name
        system_metadata = MultiTenantMetadataSchema.create_for_system(
            "__legacy_config",
            "memory_collection"
        )
        assert system_metadata.legacy_collection_name == "__legacy_config"

    def test_default_values(self):
        """Test that default values are properly set."""
        metadata = MultiTenantMetadataSchema.create_for_project("test", "docs")

        assert metadata.naming_pattern == "project_pattern"
        assert metadata.is_reserved_name is False
        assert metadata.migration_source == "metadata_based"
        assert metadata.category == "general"
        assert metadata.priority == 3
        assert metadata.version == 1
        assert metadata.tags == []
        assert metadata.compatibility_version == METADATA_SCHEMA_VERSION

    def test_enum_handling(self):
        """Test that enum values are handled correctly."""
        metadata = MultiTenantMetadataSchema.create_for_project("test", "docs")

        # Test enum comparisons
        assert metadata.collection_category == CollectionCategory.PROJECT
        assert metadata.workspace_scope == WorkspaceScope.PROJECT
        assert metadata.access_level == AccessLevel.PRIVATE

        # Test enum serialization
        payload = metadata.to_qdrant_payload()
        assert payload["collection_category"] == "project"
        assert payload["workspace_scope"] == "project"
        assert payload["access_level"] == "private"

        # Test enum deserialization
        reconstructed = MultiTenantMetadataSchema.from_qdrant_payload(payload)
        assert reconstructed.collection_category == CollectionCategory.PROJECT
        assert isinstance(reconstructed.collection_category, CollectionCategory)

    @pytest.mark.parametrize("project_name,collection_type,expected_id_length", [
        ("short", "docs", 12),
        ("very_long_project_name_with_many_chars", "documentation", 12),
        ("project-with-hyphens", "notes", 12),
        ("project_with_underscores", "scratchbook", 12),
    ])
    def test_project_id_generation_various_names(self, project_name, collection_type, expected_id_length):
        """Test project ID generation with various project names."""
        metadata = MultiTenantMetadataSchema.create_for_project(project_name, collection_type)
        assert len(metadata.project_id) == expected_id_length
        assert all(c in "0123456789abcdef" for c in metadata.project_id)

    def test_required_field_validation(self):
        """Test that required fields are validated during creation."""
        # Test missing project_id
        with pytest.raises(ValueError, match="Required field 'project_id' is missing"):
            MultiTenantMetadataSchema(
                project_id="",
                project_name="test",
                tenant_namespace="test.docs",
                collection_type="docs",
                collection_category=CollectionCategory.PROJECT,
                workspace_scope=WorkspaceScope.PROJECT
            )

        # Test missing project_name
        with pytest.raises(ValueError, match="Required field 'project_name' is missing"):
            MultiTenantMetadataSchema(
                project_id="a1b2c3d4e5f6",
                project_name="",
                tenant_namespace="test.docs",
                collection_type="docs",
                collection_category=CollectionCategory.PROJECT,
                workspace_scope=WorkspaceScope.PROJECT
            )

    def test_string_field_normalization_edge_cases(self):
        """Test string field normalization with edge cases."""
        metadata = MultiTenantMetadataSchema.create_for_project(
            project_name="  UPPER-case_Project  ",
            collection_type="  Mixed-CASE_Type  ",
            created_by="  Admin User  "
        )

        assert metadata.project_name == "upper_case_project"
        assert metadata.collection_type == "mixed_case_type"
        assert metadata.created_by == "admin user"
        assert metadata.tenant_namespace == "upper_case_project.mixed_case_type"

    def test_timestamp_fields(self):
        """Test timestamp field handling."""
        metadata = MultiTenantMetadataSchema.create_for_project("test", "docs")

        # Check that timestamps are set
        assert metadata.created_at is not None
        assert metadata.updated_at is not None

        # Check timestamp format (should be ISO format)
        from datetime import datetime
        try:
            datetime.fromisoformat(metadata.created_at.replace('Z', '+00:00'))
            datetime.fromisoformat(metadata.updated_at.replace('Z', '+00:00'))
        except ValueError:
            pytest.fail("Timestamps are not in valid ISO format")

        # Initially, created_at and updated_at should be very close
        created_dt = datetime.fromisoformat(metadata.created_at.replace('Z', '+00:00'))
        updated_dt = datetime.fromisoformat(metadata.updated_at.replace('Z', '+00:00'))
        time_diff = abs((updated_dt - created_dt).total_seconds())
        assert time_diff < 1.0  # Should be within 1 second