"""
Unit tests for configuration migration functionality.

Tests the ConfigMigrator.migrate_collection_config() method and related functionality.
"""

import pytest
import sys
import os
from pathlib import Path

# Add both source paths to sys.path
test_dir = Path(__file__).parent.parent
sys.path.insert(0, str(test_dir.parent / "src" / "python"))

from workspace_qdrant_mcp.utils.migration import ConfigMigrator, ConfigVersion

class TestCollectionConfigMigration:
    """Test collection configuration migration functionality."""

    @pytest.fixture
    def migrator(self):
        """Create a ConfigMigrator instance for testing."""
        return ConfigMigrator()

    def test_collection_suffixes_to_types_migration(self, migrator):
        """Test migration from collection_suffixes to collection_types."""
        config_data = {
            "workspace": {
                "collection_suffixes": ["docs", "notes", "scratchbook"],
                "github_user": "testuser"
            },
            "qdrant": {
                "url": "http://localhost:6333"
            }
        }

        migrated = migrator.migrate_collection_config(config_data)

        # Verify migration
        workspace = migrated["workspace"]
        assert "collection_suffixes" not in workspace, "collection_suffixes should be removed"
        assert "collection_types" in workspace, "collection_types should be added"
        assert workspace["collection_types"] == ["docs", "notes", "scratchbook"], "Values should be preserved"

    def test_collection_prefix_migration(self, migrator):
        """Test migration of collection_prefix settings."""
        config_data = {
            "workspace": {
                "collection_prefix": "proj_",
                "collection_types": ["proj_docs", "proj_notes"],
                "github_user": "testuser"
            }
        }

        migrated = migrator.migrate_collection_config(config_data)

        # Verify migration
        workspace = migrated["workspace"]
        assert "collection_prefix" not in workspace, "collection_prefix should be removed"
        assert workspace["collection_types"] == ["docs", "notes"], "Prefix should be stripped"

    def test_both_fields_present_uses_collection_types(self, migrator):
        """Test that collection_types takes precedence when both fields are present."""
        config_data = {
            "workspace": {
                "collection_suffixes": ["old1", "old2"],
                "collection_types": ["new1", "new2"],
                "github_user": "testuser"
            }
        }

        migrated = migrator.migrate_collection_config(config_data)

        # Verify migration
        workspace = migrated["workspace"]
        assert "collection_suffixes" not in workspace, "collection_suffixes should be removed"
        assert workspace["collection_types"] == ["new1", "new2"], "Should keep collection_types values"

    def test_max_collections_deprecation(self, migrator):
        """Test removal of deprecated max_collections field."""
        config_data = {
            "workspace": {
                "collection_types": ["docs", "notes"],
                "max_collections": 50,
                "github_user": "testuser"
            }
        }

        migrated = migrator.migrate_collection_config(config_data)

        # Verify migration
        workspace = migrated["workspace"]
        assert "max_collections" not in workspace, "max_collections should be removed"
        assert workspace["collection_types"] == ["docs", "notes"], "collection_types should be preserved"

    def test_empty_collections_get_default(self, migrator):
        """Test that empty collections get default values when migration is applied."""
        config_data = {
            "workspace": {
                "collection_prefix": "old_",
                "github_user": "testuser"
            }
        }

        migrated = migrator.migrate_collection_config(config_data)

        # Verify migration
        workspace = migrated["workspace"]
        assert "collection_prefix" not in workspace, "collection_prefix should be removed"
        assert "collection_types" in workspace, "collection_types should be added"
        assert workspace["collection_types"] == ["scratchbook"], "Should have default types"

    def test_no_migration_needed_leaves_config_unchanged(self, migrator):
        """Test that config without deprecated fields is left unchanged."""
        config_data = {
            "workspace": {
                "collection_types": ["docs", "notes"],
                "github_user": "testuser"
            },
            "qdrant": {
                "url": "http://localhost:6333"
            }
        }

        migrated = migrator.migrate_collection_config(config_data)

        # Should be unchanged
        assert migrated == config_data, "Config should be unchanged when no migration needed"

    def test_validate_migrated_collection_types_valid(self, migrator):
        """Test validation of valid collection types."""
        result = migrator._validate_migrated_collection_types(["docs", "notes", "scratchbook"])

        assert result["is_valid"], "Should be valid"
        assert result["warning"] is None, "Should have no warnings"

    def test_validate_migrated_collection_types_reserved_patterns(self, migrator):
        """Test validation warnings for reserved patterns."""
        result = migrator._validate_migrated_collection_types(["docs", "_reserved", "__system"])

        assert not result["is_valid"], "Should be invalid due to reserved patterns"
        assert "reserved prefix pattern" in result["warning"], "Should warn about prefix patterns"

    def test_validate_migrated_collection_types_reserved_names(self, migrator):
        """Test validation warnings for reserved names."""
        result = migrator._validate_migrated_collection_types(["docs", "memory", "system"])

        assert not result["is_valid"], "Should be invalid due to reserved names"
        assert "conflicts with reserved names" in result["warning"], "Should warn about reserved names"

    def test_map_prefixed_collections_with_underscore(self, migrator):
        """Test mapping of prefixed collections with underscore."""
        result = migrator._map_prefixed_collections(["proj_docs", "proj_notes"], "proj_")

        assert result == ["docs", "notes"], "Should remove prefix with underscore"

    def test_map_prefixed_collections_without_underscore(self, migrator):
        """Test mapping of prefixed collections without underscore."""
        result = migrator._map_prefixed_collections(["projdocs", "projnotes"], "proj")

        assert result == ["docs", "notes"], "Should remove prefix without underscore"

    def test_map_prefixed_collections_no_prefix_match(self, migrator):
        """Test mapping when collection names don't match prefix."""
        result = migrator._map_prefixed_collections(["docs", "notes"], "proj_")

        assert result == ["docs", "notes"], "Should keep names unchanged when no prefix match"

    def test_migrate_invalid_config_data(self, migrator):
        """Test migration with invalid config data."""
        # Test with non-dict config
        result = migrator.migrate_collection_config("invalid")
        assert result == "invalid", "Should return unchanged for invalid config"

        # Test with None
        result = migrator.migrate_collection_config(None)
        assert result is None, "Should return unchanged for None config"