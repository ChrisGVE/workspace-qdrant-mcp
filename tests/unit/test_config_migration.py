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

from common.utils.migration import ConfigMigrator, ConfigVersion

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


class TestDeprecatedFieldRemoval:
    """Test deprecated field removal functionality."""

    @pytest.fixture
    def migrator(self):
        """Create a ConfigMigrator instance for testing."""
        return ConfigMigrator()

    def test_remove_collection_deprecated_fields(self, migrator):
        """Test removal of collection-related deprecated fields."""
        config_data = {
            "workspace": {
                "collection_prefix": "proj_",
                "max_collections": 100,
                "github_user": "testuser"
            },
            "qdrant": {
                "url": "http://localhost:6333"
            },
            "collections": {
                "project_suffixes": ["docs", "notes"]
            }
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Verify removal
        workspace = cleaned.get("workspace", {})
        assert "collection_prefix" not in workspace, "collection_prefix should be removed"
        assert "max_collections" not in workspace, "max_collections should be removed"
        assert workspace.get("github_user") == "testuser", "Non-deprecated fields should be preserved"

        # Other sections should be preserved
        assert "qdrant" in cleaned, "qdrant section should be preserved"
        assert "collections" in cleaned, "collections section should be preserved"

    def test_remove_ingestion_deprecated_fields(self, migrator):
        """Test removal of ingestion-related deprecated fields."""
        config_data = {
            "workspace": {
                "recursive_depth": 5,
                "github_user": "testuser"
            },
            "qdrant": {
                "url": "http://localhost:6333"
            }
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Verify removal and fallback
        workspace = cleaned.get("workspace", {})
        assert "recursive_depth" not in workspace, "recursive_depth should be removed"

        # Should apply fallback
        ingestion = cleaned.get("ingestion", {})
        assert ingestion.get("max_depth") == 10, "Should apply fallback max_depth"

    def test_remove_pattern_deprecated_fields(self, migrator):
        """Test removal of pattern-related deprecated fields."""
        config_data = {
            "patterns": {
                "include_patterns": ["*.py", "*.js"],
                "exclude_patterns": ["*.tmp"],
                "file_types": [".md", ".txt"],
                "custom_ecosystems": {"test": {"files": ["test.yml"]}},
                "pattern_priorities": {"include": {"*.py": 10}}
            },
            "workspace": {
                "file_patterns": ["*.doc"],
                "ignore_patterns": ["*.log"],
                "supported_extensions": [".cpp", ".h"]
            }
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Verify removal from patterns section
        patterns = cleaned.get("patterns", {})
        deprecated_pattern_fields = [
            "include_patterns", "exclude_patterns", "file_types",
            "custom_ecosystems", "pattern_priorities"
        ]
        for field in deprecated_pattern_fields:
            assert field not in patterns, f"{field} should be removed from patterns"

        # Verify removal from workspace section
        workspace = cleaned.get("workspace", {})
        deprecated_workspace_fields = ["file_patterns", "ignore_patterns", "supported_extensions"]
        for field in deprecated_workspace_fields:
            assert field not in workspace, f"{field} should be removed from workspace"

    def test_remove_legacy_mode_field(self, migrator):
        """Test removal of enable_legacy_mode field."""
        config_data = {
            "workspace": {
                "enable_legacy_mode": True,
                "github_user": "testuser"
            }
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Verify removal
        workspace = cleaned.get("workspace", {})
        assert "enable_legacy_mode" not in workspace, "enable_legacy_mode should be removed"
        assert workspace.get("github_user") == "testuser", "Other fields should be preserved"

    def test_nested_deprecated_field_removal(self, migrator):
        """Test removal of deprecated fields in nested structures."""
        config_data = {
            "workspace": {
                "settings": {
                    "collection_prefix": "nested_",
                    "valid_field": "keep_this"
                },
                "github_user": "testuser"
            },
            "patterns": {
                "advanced": {
                    "include_patterns": ["*.py"],
                    "valid_pattern": "keep_this"
                }
            }
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Verify nested removal
        settings = cleaned.get("workspace", {}).get("settings", {})
        assert "collection_prefix" not in settings, "Nested collection_prefix should be removed"
        assert settings.get("valid_field") == "keep_this", "Valid nested field should be preserved"

        advanced_patterns = cleaned.get("patterns", {}).get("advanced", {})
        assert "include_patterns" not in advanced_patterns, "Nested include_patterns should be removed"
        assert advanced_patterns.get("valid_pattern") == "keep_this", "Valid nested pattern should be preserved"

    def test_fallback_handling_for_recursive_depth(self, migrator):
        """Test fallback handling when recursive_depth is removed without replacement."""
        config_data = {
            "workspace": {
                "recursive_depth": 8,
                "github_user": "testuser"
            },
            "qdrant": {
                "url": "http://localhost:6333"
            }
            # No ingestion section
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Verify fallback was applied
        assert "ingestion" in cleaned, "ingestion section should be created"
        assert cleaned["ingestion"]["max_depth"] == 10, "Should apply fallback max_depth"

    def test_fallback_handling_for_collection_settings(self, migrator):
        """Test fallback handling when collection settings are removed without replacement."""
        config_data = {
            "workspace": {
                "collection_prefix": "test_",
                "max_collections": 50,
                "github_user": "testuser"
            },
            "qdrant": {
                "url": "http://localhost:6333"
            }
            # No collections section
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Verify fallback was applied
        assert "collections" in cleaned, "collections section should be created"
        assert cleaned["collections"]["project_suffixes"] == ["scratchbook"], "Should apply fallback project_suffixes"

    def test_no_fallback_when_replacement_exists(self, migrator):
        """Test that fallback is not applied when replacement already exists."""
        config_data = {
            "workspace": {
                "recursive_depth": 8,
                "github_user": "testuser"
            },
            "ingestion": {
                "max_depth": 15  # Existing replacement
            }
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Verify no fallback was applied
        assert cleaned["ingestion"]["max_depth"] == 15, "Existing max_depth should be preserved"

    def test_validation_error_for_broken_config(self, migrator):
        """Test validation error when removing fields would break functionality."""
        # Create a config that would become invalid after field removal
        config_data = {
            "workspace": {
                "recursive_depth": 5
            }
            # No essential sections like qdrant, embeddings
        }

        with pytest.raises(ValueError, match="Removing deprecated fields would break functionality"):
            migrator.remove_deprecated_fields(config_data)

    def test_no_changes_for_clean_config(self, migrator):
        """Test that clean configuration without deprecated fields is unchanged."""
        config_data = {
            "qdrant": {
                "url": "http://localhost:6333"
            },
            "embeddings": {
                "model": "all-MiniLM-L6-v2"
            },
            "collections": {
                "project_suffixes": ["docs", "notes"]
            },
            "patterns": {
                "custom_include_patterns": ["*.py", "*.js"]
            }
        }

        cleaned = migrator.remove_deprecated_fields(config_data)

        # Should be unchanged
        assert cleaned == config_data, "Clean config should be unchanged"

    def test_generate_field_specific_warning(self, migrator):
        """Test generation of field-specific warning messages."""
        # Test collection_prefix warning
        warning = migrator._generate_field_specific_warning("collection_prefix", "test_", "workspace.collection_prefix")
        assert "collection_prefix" in warning
        assert "CollectionType system" in warning
        assert "collections.project_suffixes" in warning

        # Test recursive_depth warning
        warning = migrator._generate_field_specific_warning("recursive_depth", 5, "workspace.recursive_depth")
        assert "recursive_depth" in warning
        assert "ingestion.max_depth" in warning
        assert "5" in warning  # Should include the original value

        # Test unknown field warning
        warning = migrator._generate_field_specific_warning("unknown_field", "value", "section.unknown_field")
        assert "unknown_field" in warning
        assert "documentation" in warning

    def test_validate_functionality_preservation(self, migrator):
        """Test functionality preservation validation."""
        original_config = {
            "workspace": {
                "collection_prefix": "test_",
                "recursive_depth": 5
            }
        }

        cleaned_config = {
            "qdrant": {"url": "http://localhost:6333"}
        }

        removed_fields = ["workspace.collection_prefix", "workspace.recursive_depth"]

        result = migrator._validate_functionality_preservation(original_config, cleaned_config, removed_fields)

        assert result["is_valid"] is True
        assert len(result["warnings"]) > 0, "Should have warnings about missing replacements"
        assert len(result["recommendations"]) > 0, "Should have recommendations"

    def test_validate_cleaned_config(self, migrator):
        """Test validation of cleaned configuration."""
        # Test valid config
        valid_config = {
            "qdrant": {"url": "http://localhost:6333"},
            "embeddings": {"model": "test"}
        }
        result = migrator._validate_cleaned_config(valid_config)
        assert result["is_valid"] is True
        assert result["error"] is None

        # Test empty config
        result = migrator._validate_cleaned_config({})
        assert result["is_valid"] is False
        assert "empty" in result["error"]

        # Test non-dict config
        result = migrator._validate_cleaned_config("invalid")
        assert result["is_valid"] is False
        assert "dictionary" in result["error"]

        # Test config missing essential sections
        incomplete_config = {"other": "value"}
        result = migrator._validate_cleaned_config(incomplete_config)
        assert result["is_valid"] is False
        assert "essential sections" in result["error"]

    def test_has_replacement_config_methods(self, migrator):
        """Test helper methods for checking replacement configurations."""
        # Test collection replacement detection
        config_with_collections = {
            "collections": {"project_suffixes": ["docs"]}
        }
        assert migrator._has_collection_config_replacement(config_with_collections)

        config_with_workspace = {
            "workspace": {"collection_types": ["notes"]}
        }
        assert migrator._has_collection_config_replacement(config_with_workspace)

        config_without_collections = {"other": "value"}
        assert not migrator._has_collection_config_replacement(config_without_collections)

        # Test ingestion depth replacement detection
        config_with_depth = {
            "ingestion": {"max_depth": 10}
        }
        assert migrator._has_ingestion_depth_config(config_with_depth)

        config_without_depth = {"ingestion": {"other": "value"}}
        assert not migrator._has_ingestion_depth_config(config_without_depth)

        # Test pattern replacement detection
        config_with_patterns = {
            "patterns": {"custom_include_patterns": ["*.py"]}
        }
        assert migrator._has_pattern_config_replacement(config_with_patterns)

        config_without_patterns = {"patterns": {"other": "value"}}
        assert not migrator._has_pattern_config_replacement(config_without_patterns)

    def test_invalid_config_data_handling(self, migrator):
        """Test handling of invalid configuration data types."""
        # Test non-dict input
        result = migrator.remove_deprecated_fields("invalid")
        assert result == "invalid", "Should return unchanged for invalid input"

        # Test None input
        result = migrator.remove_deprecated_fields(None)
        assert result is None, "Should return unchanged for None input"

        # Test list input
        result = migrator.remove_deprecated_fields([])
        assert result == [], "Should return unchanged for list input"