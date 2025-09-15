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


class TestPatternConfigMigration:
    """Test pattern configuration migration functionality."""

    @pytest.fixture
    def migrator(self):
        """Create a ConfigMigrator instance for testing."""
        return ConfigMigrator()

    def test_include_patterns_migration(self, migrator):
        """Test migration from include_patterns to custom_include_patterns."""
        config_data = {
            "patterns": {
                "include_patterns": ["*.py", "*.js", "**/*.md"]
            },
            "qdrant": {
                "url": "http://localhost:6333"
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        assert "include_patterns" not in patterns, "include_patterns should be removed"
        assert "custom_include_patterns" in patterns, "custom_include_patterns should be added"
        assert patterns["custom_include_patterns"] == ["*.py", "*.js", "**/*.md"], "Values should be preserved"

    def test_exclude_patterns_migration(self, migrator):
        """Test migration from exclude_patterns to custom_exclude_patterns."""
        config_data = {
            "patterns": {
                "exclude_patterns": ["*.tmp", "build/**", ".git/**"]
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        assert "exclude_patterns" not in patterns, "exclude_patterns should be removed"
        assert "custom_exclude_patterns" in patterns, "custom_exclude_patterns should be added"
        assert patterns["custom_exclude_patterns"] == ["*.tmp", "build/**", ".git/**"], "Values should be preserved"

    def test_file_patterns_alias_migration(self, migrator):
        """Test migration from file_patterns alias to custom_include_patterns."""
        config_data = {
            "workspace": {
                "file_patterns": ["*.txt", "docs/**"]
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        assert "custom_include_patterns" in patterns, "custom_include_patterns should be added"
        assert patterns["custom_include_patterns"] == ["*.txt", "docs/**"], "Values should be preserved"

        # Workspace should still exist but without file_patterns
        workspace = migrated["workspace"]
        assert "file_patterns" not in workspace, "file_patterns should be removed from workspace"

    def test_ignore_patterns_alias_migration(self, migrator):
        """Test migration from ignore_patterns alias to custom_exclude_patterns."""
        config_data = {
            "workspace": {
                "ignore_patterns": ["*.log", "temp/**"]
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        assert "custom_exclude_patterns" in patterns, "custom_exclude_patterns should be added"
        assert patterns["custom_exclude_patterns"] == ["*.log", "temp/**"], "Values should be preserved"

    def test_file_types_to_patterns_migration(self, migrator):
        """Test migration from file_types to include patterns."""
        config_data = {
            "patterns": {
                "file_types": [".py", ".js", "md"]  # Mixed format with and without dots
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        assert "file_types" not in patterns, "file_types should be removed"
        assert "custom_include_patterns" in patterns, "custom_include_patterns should be added"

        expected_patterns = ["**/*.py", "**/*.js", "**/*.md"]
        assert patterns["custom_include_patterns"] == expected_patterns, "Extensions should be converted to patterns"

    def test_supported_extensions_migration(self, migrator):
        """Test migration from supported_extensions to include patterns."""
        config_data = {
            "workspace": {
                "supported_extensions": ["cpp", ".h", ".hpp"]
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        assert "custom_include_patterns" in patterns, "custom_include_patterns should be added"

        expected_patterns = ["**/*.cpp", "**/*.h", "**/*.hpp"]
        assert patterns["custom_include_patterns"] == expected_patterns, "Extensions should be converted to patterns"

    def test_custom_ecosystems_migration(self, migrator):
        """Test migration from custom_ecosystems to custom_project_indicators."""
        config_data = {
            "patterns": {
                "custom_ecosystems": {
                    "my_framework": {
                        "files": ["framework.config", "*.framework"],
                        "optional_files": ["docs/framework.md"],
                        "min_optional": 1
                    },
                    "custom_lang": {
                        "required_files": ["lang.toml", "src/**/*.lang"]
                    }
                }
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        assert "custom_ecosystems" not in patterns, "custom_ecosystems should be removed"
        assert "custom_project_indicators" in patterns, "custom_project_indicators should be added"

        indicators = patterns["custom_project_indicators"]
        assert "my_framework" in indicators, "Framework ecosystem should be migrated"
        assert "custom_lang" in indicators, "Custom language ecosystem should be migrated"

        # Check specific mappings
        framework_indicator = indicators["my_framework"]
        assert framework_indicator["required_files"] == ["framework.config", "*.framework"]
        assert framework_indicator["optional_files"] == ["docs/framework.md"]
        assert framework_indicator["min_optional_files"] == 1

    def test_pattern_priorities_migration(self, migrator):
        """Test migration of pattern priorities with ordering."""
        config_data = {
            "patterns": {
                "include_patterns": ["*.py", "*.js", "*.md"],
                "exclude_patterns": ["*.tmp", "*.log"],
                "pattern_priorities": {
                    "include": {
                        "*.py": 10,
                        "*.js": 5,
                        "*.md": 1
                    },
                    "exclude": {
                        "*.tmp": 8
                    }
                }
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        assert "pattern_priorities" not in patterns, "pattern_priorities should be removed"

        # Check that patterns are ordered by priority (higher first)
        include_patterns = patterns["custom_include_patterns"]
        expected_order = ["*.py", "*.js", "*.md"]  # Ordered by priority 10, 5, 1
        assert include_patterns == expected_order, "Include patterns should be ordered by priority"

    def test_structured_patterns_migration(self, migrator):
        """Test migration of structured pattern objects to simple strings."""
        config_data = {
            "patterns": {
                "include_patterns": [
                    "*.py",  # Simple string
                    {"pattern": "*.js", "description": "JavaScript files"},  # Structured object
                    {"pattern": "**/*.md", "priority": 5}  # Structured with priority
                ]
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration
        patterns = migrated["patterns"]
        include_patterns = patterns["custom_include_patterns"]
        expected_patterns = ["*.py", "*.js", "**/*.md"]
        assert include_patterns == expected_patterns, "Structured patterns should be converted to strings"

    def test_invalid_patterns_filtered(self, migrator):
        """Test that invalid patterns are filtered out during migration."""
        config_data = {
            "patterns": {
                "include_patterns": [
                    "*.py",  # Valid
                    "",      # Invalid - empty
                    "/etc/passwd",  # Invalid - absolute path
                    "*.js",  # Valid
                    None,    # Invalid - not string
                    ".",     # Invalid - dangerous
                    {"pattern": "*.md"},  # Valid structured
                    {"pattern": ""},      # Invalid structured
                ]
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify migration - only valid patterns should remain
        patterns = migrated["patterns"]
        include_patterns = patterns["custom_include_patterns"]
        expected_patterns = ["*.py", "*.js", "*.md"]
        assert include_patterns == expected_patterns, "Only valid patterns should be migrated"

    def test_multiple_sources_combined(self, migrator):
        """Test migration combining patterns from multiple configuration sections."""
        config_data = {
            "patterns": {
                "include_patterns": ["*.py", "*.rs"]
            },
            "workspace": {
                "file_patterns": ["*.js", "*.ts"],
                "supported_extensions": ["cpp", ".h"]
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Verify all sources are combined
        patterns = migrated["patterns"]
        include_patterns = patterns["custom_include_patterns"]

        # Should contain patterns from all sources
        expected_patterns = ["*.py", "*.rs", "*.js", "*.ts", "**/*.cpp", "**/*.h"]
        assert set(include_patterns) == set(expected_patterns), "Patterns from all sources should be combined"

    def test_no_migration_needed_leaves_config_unchanged(self, migrator):
        """Test that config without deprecated pattern fields is left unchanged."""
        config_data = {
            "patterns": {
                "custom_include_patterns": ["*.py", "*.js"],
                "custom_exclude_patterns": ["*.tmp"]
            },
            "qdrant": {
                "url": "http://localhost:6333"
            }
        }

        migrated = migrator.migrate_pattern_config(config_data)

        # Should be unchanged
        assert migrated == config_data, "Config should be unchanged when no migration needed"

    def test_migrate_invalid_config_data(self, migrator):
        """Test migration with invalid config data."""
        # Test with non-dict config
        result = migrator.migrate_pattern_config("invalid")
        assert result == "invalid", "Should return unchanged for invalid config"

        # Test with None
        result = migrator.migrate_pattern_config(None)
        assert result is None, "Should return unchanged for None config"

    def test_validate_and_convert_patterns(self, migrator):
        """Test pattern validation and conversion helper method."""
        patterns = [
            "*.py",  # Valid string
            {"pattern": "*.js", "desc": "JS files"},  # Valid structured
            "",  # Invalid - empty string
            {"pattern": "/etc/passwd"},  # Invalid - absolute path
            None,  # Invalid - not string/dict
            "docs/**/*.md",  # Valid complex pattern
        ]

        validated = migrator._validate_and_convert_patterns(patterns, "include")
        expected = ["*.py", "*.js", "docs/**/*.md"]
        assert validated == expected, "Should return only valid patterns as strings"

    def test_convert_extensions_to_patterns(self, migrator):
        """Test file extension to pattern conversion helper method."""
        extensions = [".py", "js", ".md", "", None, "cpp"]

        patterns = migrator._convert_extensions_to_patterns(extensions)
        expected = ["**/*.py", "**/*.js", "**/*.md", "**/*.cpp"]
        assert patterns == expected, "Should convert extensions to glob patterns"

    def test_convert_ecosystems_to_indicators(self, migrator):
        """Test ecosystem to indicator conversion helper method."""
        ecosystems = {
            "framework": {
                "files": ["config.yml"],
                "optional_files": ["docs.md"],
                "min_optional": 1
            },
            "invalid": "not_a_dict",
            "lang": {
                "required_files": ["lang.toml"]
            }
        }

        indicators = migrator._convert_ecosystems_to_indicators(ecosystems)

        assert "framework" in indicators, "Valid framework should be converted"
        assert "lang" in indicators, "Valid language should be converted"
        assert "invalid" not in indicators, "Invalid ecosystem should be skipped"

        framework = indicators["framework"]
        assert framework["required_files"] == ["config.yml"]
        assert framework["optional_files"] == ["docs.md"]
        assert framework["min_optional_files"] == 1