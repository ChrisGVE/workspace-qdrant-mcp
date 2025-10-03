"""
Unit tests for collection_type_config module.

Tests comprehensive coverage of:
- Metadata field validation
- Type-specific configurations
- Deletion mode handling
- Performance settings
- Migration settings
- Validation rules
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from src.python.common.core.collection_type_config import (
    DeletionMode,
    MetadataFieldType,
    MetadataFieldSpec,
    PerformanceSettings,
    MigrationSettings,
    CollectionTypeConfig,
    get_type_config,
    get_all_type_configs,
    validate_metadata_for_type,
    get_deletion_mode,
    should_use_cumulative_deletion,
    should_use_dynamic_deletion,
)
from src.python.common.core.collection_types import CollectionType


class TestMetadataFieldSpec:
    """Tests for MetadataFieldSpec validation."""

    def test_string_field_validation_success(self):
        """Test successful string field validation."""
        spec = MetadataFieldSpec(
            name="test_field",
            field_type=MetadataFieldType.STRING,
            required=True,
            min_length=3,
            max_length=10
        )

        is_valid, error = spec.validate("hello")
        assert is_valid
        assert error is None

    def test_string_field_validation_too_short(self):
        """Test string field validation with value too short."""
        spec = MetadataFieldSpec(
            name="test_field",
            field_type=MetadataFieldType.STRING,
            min_length=5
        )

        is_valid, error = spec.validate("hi")
        assert not is_valid
        assert "at least 5 characters" in error

    def test_string_field_validation_too_long(self):
        """Test string field validation with value too long."""
        spec = MetadataFieldSpec(
            name="test_field",
            field_type=MetadataFieldType.STRING,
            max_length=5
        )

        is_valid, error = spec.validate("too long string")
        assert not is_valid
        assert "at most 5 characters" in error

    def test_string_field_pattern_validation(self):
        """Test string field pattern validation."""
        spec = MetadataFieldSpec(
            name="test_field",
            field_type=MetadataFieldType.STRING,
            pattern=r"^__[a-zA-Z0-9_-]+$"
        )

        # Valid pattern
        is_valid, error = spec.validate("__system_config")
        assert is_valid
        assert error is None

        # Invalid pattern
        is_valid, error = spec.validate("no_prefix")
        assert not is_valid
        assert "does not match required pattern" in error

    def test_integer_field_validation_success(self):
        """Test successful integer field validation."""
        spec = MetadataFieldSpec(
            name="priority",
            field_type=MetadataFieldType.INTEGER,
            min_value=1,
            max_value=5
        )

        is_valid, error = spec.validate(3)
        assert is_valid
        assert error is None

    def test_integer_field_validation_wrong_type(self):
        """Test integer field validation with wrong type."""
        spec = MetadataFieldSpec(
            name="priority",
            field_type=MetadataFieldType.INTEGER
        )

        is_valid, error = spec.validate("not an integer")
        assert not is_valid
        assert "must be an integer" in error

    def test_integer_field_validation_below_min(self):
        """Test integer field validation below minimum."""
        spec = MetadataFieldSpec(
            name="priority",
            field_type=MetadataFieldType.INTEGER,
            min_value=1
        )

        is_valid, error = spec.validate(0)
        assert not is_valid
        assert "at least 1" in error

    def test_integer_field_validation_above_max(self):
        """Test integer field validation above maximum."""
        spec = MetadataFieldSpec(
            name="priority",
            field_type=MetadataFieldType.INTEGER,
            max_value=5
        )

        is_valid, error = spec.validate(10)
        assert not is_valid
        assert "at most 5" in error

    def test_float_field_validation(self):
        """Test float field validation."""
        spec = MetadataFieldSpec(
            name="score",
            field_type=MetadataFieldType.FLOAT,
            min_value=0.0,
            max_value=1.0
        )

        # Valid float
        is_valid, error = spec.validate(0.5)
        assert is_valid

        # Valid integer (coerced to float)
        is_valid, error = spec.validate(1)
        assert is_valid

        # Below minimum
        is_valid, error = spec.validate(-0.1)
        assert not is_valid

        # Above maximum
        is_valid, error = spec.validate(1.5)
        assert not is_valid

    def test_boolean_field_validation(self):
        """Test boolean field validation."""
        spec = MetadataFieldSpec(
            name="enabled",
            field_type=MetadataFieldType.BOOLEAN
        )

        is_valid, error = spec.validate(True)
        assert is_valid

        is_valid, error = spec.validate(False)
        assert is_valid

        is_valid, error = spec.validate("true")
        assert not is_valid
        assert "must be a boolean" in error

    def test_list_field_validation(self):
        """Test list field validation."""
        spec = MetadataFieldSpec(
            name="tags",
            field_type=MetadataFieldType.LIST,
            min_length=1,
            max_length=5
        )

        # Valid list
        is_valid, error = spec.validate(["tag1", "tag2"])
        assert is_valid

        # Empty list (below min)
        is_valid, error = spec.validate([])
        assert not is_valid
        assert "at least 1 items" in error

        # Too many items
        is_valid, error = spec.validate(["a", "b", "c", "d", "e", "f"])
        assert not is_valid
        assert "at most 5 items" in error

    def test_dict_field_validation(self):
        """Test dict field validation."""
        spec = MetadataFieldSpec(
            name="metadata",
            field_type=MetadataFieldType.DICT
        )

        is_valid, error = spec.validate({"key": "value"})
        assert is_valid

        is_valid, error = spec.validate("not a dict")
        assert not is_valid
        assert "must be a dictionary" in error

    def test_allowed_values_validation(self):
        """Test allowed values validation."""
        spec = MetadataFieldSpec(
            name="status",
            field_type=MetadataFieldType.STRING,
            allowed_values={"active", "inactive", "pending"}
        )

        # Valid value
        is_valid, error = spec.validate("active")
        assert is_valid

        # Invalid value
        is_valid, error = spec.validate("unknown")
        assert not is_valid
        assert "must be one of" in error

    def test_required_field_validation(self):
        """Test required field validation with None value."""
        spec = MetadataFieldSpec(
            name="required_field",
            field_type=MetadataFieldType.STRING,
            required=True
        )

        is_valid, error = spec.validate(None)
        assert not is_valid
        assert "is required" in error

    def test_optional_field_with_none(self):
        """Test optional field with None value."""
        spec = MetadataFieldSpec(
            name="optional_field",
            field_type=MetadataFieldType.STRING,
            required=False
        )

        is_valid, error = spec.validate(None)
        assert is_valid

    def test_field_with_default_value(self):
        """Test field with default value."""
        spec = MetadataFieldSpec(
            name="priority",
            field_type=MetadataFieldType.INTEGER,
            required=False,
            default=3
        )

        # None is valid since there's a default
        is_valid, error = spec.validate(None)
        assert is_valid


class TestCollectionTypeConfig:
    """Tests for CollectionTypeConfig."""

    def test_validate_metadata_all_required_fields(self):
        """Test metadata validation with all required fields present."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.SYSTEM,
            deletion_mode=DeletionMode.CUMULATIVE,
            required_metadata_fields=[
                MetadataFieldSpec("name", MetadataFieldType.STRING, required=True),
                MetadataFieldSpec("created_at", MetadataFieldType.STRING, required=True),
            ]
        )

        metadata = {
            "name": "test_collection",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid
        assert len(errors) == 0

    def test_validate_metadata_missing_required_field(self):
        """Test metadata validation with missing required field."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.SYSTEM,
            deletion_mode=DeletionMode.CUMULATIVE,
            required_metadata_fields=[
                MetadataFieldSpec("name", MetadataFieldType.STRING, required=True),
                MetadataFieldSpec("created_at", MetadataFieldType.STRING, required=True),
            ]
        )

        metadata = {
            "name": "test_collection"
            # Missing created_at
        }

        is_valid, errors = config.validate_metadata(metadata)
        assert not is_valid
        assert len(errors) > 0
        assert any("created_at" in error for error in errors)

    def test_validate_metadata_with_optional_fields(self):
        """Test metadata validation with optional fields."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.SYSTEM,
            deletion_mode=DeletionMode.CUMULATIVE,
            required_metadata_fields=[
                MetadataFieldSpec("name", MetadataFieldType.STRING, required=True),
            ],
            optional_metadata_fields=[
                MetadataFieldSpec("description", MetadataFieldType.STRING, required=False),
            ]
        )

        # Without optional field
        metadata = {"name": "test"}
        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid

        # With optional field
        metadata = {"name": "test", "description": "A test collection"}
        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid

    def test_validate_metadata_with_custom_rule(self):
        """Test metadata validation with custom validation rules."""
        def custom_rule(metadata: Dict[str, Any]) -> tuple:
            if "name" in metadata and "test" not in metadata["name"]:
                return False, "Name must contain 'test'"
            return True, None

        config = CollectionTypeConfig(
            collection_type=CollectionType.SYSTEM,
            deletion_mode=DeletionMode.CUMULATIVE,
            required_metadata_fields=[
                MetadataFieldSpec("name", MetadataFieldType.STRING, required=True),
            ],
            validation_rules=[custom_rule]
        )

        # Valid metadata
        metadata = {"name": "test_collection"}
        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid

        # Invalid metadata
        metadata = {"name": "invalid_collection"}
        is_valid, errors = config.validate_metadata(metadata)
        assert not is_valid
        assert any("must contain 'test'" in error for error in errors)

    def test_get_required_field_names(self):
        """Test getting required field names."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.SYSTEM,
            deletion_mode=DeletionMode.CUMULATIVE,
            required_metadata_fields=[
                MetadataFieldSpec("name", MetadataFieldType.STRING),
                MetadataFieldSpec("created_at", MetadataFieldType.STRING),
            ],
            optional_metadata_fields=[
                MetadataFieldSpec("description", MetadataFieldType.STRING),
            ]
        )

        required_fields = config.get_required_field_names()
        assert required_fields == {"name", "created_at"}

    def test_get_all_field_names(self):
        """Test getting all field names."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.SYSTEM,
            deletion_mode=DeletionMode.CUMULATIVE,
            required_metadata_fields=[
                MetadataFieldSpec("name", MetadataFieldType.STRING),
                MetadataFieldSpec("created_at", MetadataFieldType.STRING),
            ],
            optional_metadata_fields=[
                MetadataFieldSpec("description", MetadataFieldType.STRING),
            ]
        )

        all_fields = config.get_all_field_names()
        assert all_fields == {"name", "created_at", "description"}

    def test_get_field_spec(self):
        """Test getting field specification."""
        spec = MetadataFieldSpec("name", MetadataFieldType.STRING)
        config = CollectionTypeConfig(
            collection_type=CollectionType.SYSTEM,
            deletion_mode=DeletionMode.CUMULATIVE,
            required_metadata_fields=[spec]
        )

        retrieved_spec = config.get_field_spec("name")
        assert retrieved_spec == spec

        missing_spec = config.get_field_spec("nonexistent")
        assert missing_spec is None

    def test_get_default_metadata(self):
        """Test getting default metadata values."""
        config = CollectionTypeConfig(
            collection_type=CollectionType.SYSTEM,
            deletion_mode=DeletionMode.CUMULATIVE,
            required_metadata_fields=[
                MetadataFieldSpec("name", MetadataFieldType.STRING),
            ],
            optional_metadata_fields=[
                MetadataFieldSpec("priority", MetadataFieldType.INTEGER, default=3),
                MetadataFieldSpec("enabled", MetadataFieldType.BOOLEAN, default=True),
            ]
        )

        defaults = config.get_default_metadata()
        assert defaults == {"priority": 3, "enabled": True}


class TestTypeSpecificConfigs:
    """Tests for type-specific configurations."""

    def test_system_config(self):
        """Test SYSTEM collection configuration."""
        config = get_type_config(CollectionType.SYSTEM)

        assert config.collection_type == CollectionType.SYSTEM
        assert config.deletion_mode == DeletionMode.CUMULATIVE
        assert len(config.required_metadata_fields) > 0
        assert config.performance_settings.batch_size == 50
        assert config.migration_settings.supports_legacy_format is True

        # Test validation with valid metadata
        metadata = {
            "collection_name": "__test_system",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection_category": "system",
        }
        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid, f"Validation errors: {errors}"

    def test_library_config(self):
        """Test LIBRARY collection configuration."""
        config = get_type_config(CollectionType.LIBRARY)

        assert config.collection_type == CollectionType.LIBRARY
        assert config.deletion_mode == DeletionMode.CUMULATIVE
        assert len(config.required_metadata_fields) > 0
        assert config.performance_settings.batch_size == 100
        assert config.migration_settings.supports_legacy_format is True

        # Test validation with valid metadata
        metadata = {
            "collection_name": "_test_library",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection_category": "library",
            "language": "python",
        }
        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid, f"Validation errors: {errors}"

    def test_project_config(self):
        """Test PROJECT collection configuration."""
        config = get_type_config(CollectionType.PROJECT)

        assert config.collection_type == CollectionType.PROJECT
        assert config.deletion_mode == DeletionMode.DYNAMIC
        assert len(config.required_metadata_fields) > 0
        assert config.performance_settings.batch_size == 150
        assert config.migration_settings.migration_batch_size == 100

        # Test validation with valid metadata
        metadata = {
            "project_name": "test_project",
            "project_id": "abc123def456",
            "collection_type": "documents",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid, f"Validation errors: {errors}"

    def test_global_config(self):
        """Test GLOBAL collection configuration."""
        config = get_type_config(CollectionType.GLOBAL)

        assert config.collection_type == CollectionType.GLOBAL
        assert config.deletion_mode == DeletionMode.DYNAMIC
        assert len(config.required_metadata_fields) > 0
        assert config.performance_settings.batch_size == 200
        assert config.migration_settings.supports_legacy_format is False

        # Test validation with valid metadata
        metadata = {
            "collection_name": "algorithms",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid, f"Validation errors: {errors}"


class TestConfigRegistry:
    """Tests for configuration registry functions."""

    def test_get_type_config(self):
        """Test getting type configuration."""
        config = get_type_config(CollectionType.SYSTEM)
        assert isinstance(config, CollectionTypeConfig)
        assert config.collection_type == CollectionType.SYSTEM

    def test_get_type_config_invalid_type(self):
        """Test getting configuration for invalid type."""
        with pytest.raises(ValueError, match="No configuration available"):
            get_type_config(CollectionType.UNKNOWN)

    def test_get_all_type_configs(self):
        """Test getting all type configurations."""
        all_configs = get_all_type_configs()

        assert len(all_configs) == 4
        assert CollectionType.SYSTEM in all_configs
        assert CollectionType.LIBRARY in all_configs
        assert CollectionType.PROJECT in all_configs
        assert CollectionType.GLOBAL in all_configs

        # Verify each is a valid config
        for collection_type, config in all_configs.items():
            assert isinstance(config, CollectionTypeConfig)
            assert config.collection_type == collection_type

    def test_validate_metadata_for_type(self):
        """Test validating metadata for a specific type."""
        metadata = {
            "collection_name": "__test",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection_category": "system",
        }

        is_valid, errors = validate_metadata_for_type(CollectionType.SYSTEM, metadata)
        assert is_valid


class TestDeletionModes:
    """Tests for deletion mode functions."""

    def test_get_deletion_mode(self):
        """Test getting deletion mode for types."""
        assert get_deletion_mode(CollectionType.SYSTEM) == DeletionMode.CUMULATIVE
        assert get_deletion_mode(CollectionType.LIBRARY) == DeletionMode.CUMULATIVE
        assert get_deletion_mode(CollectionType.PROJECT) == DeletionMode.DYNAMIC
        assert get_deletion_mode(CollectionType.GLOBAL) == DeletionMode.DYNAMIC

    def test_should_use_cumulative_deletion(self):
        """Test cumulative deletion check."""
        assert should_use_cumulative_deletion(CollectionType.SYSTEM) is True
        assert should_use_cumulative_deletion(CollectionType.LIBRARY) is True
        assert should_use_cumulative_deletion(CollectionType.PROJECT) is False
        assert should_use_cumulative_deletion(CollectionType.GLOBAL) is False

    def test_should_use_dynamic_deletion(self):
        """Test dynamic deletion check."""
        assert should_use_dynamic_deletion(CollectionType.SYSTEM) is False
        assert should_use_dynamic_deletion(CollectionType.LIBRARY) is False
        assert should_use_dynamic_deletion(CollectionType.PROJECT) is True
        assert should_use_dynamic_deletion(CollectionType.GLOBAL) is True


class TestPerformanceSettings:
    """Tests for performance settings."""

    def test_system_performance_settings(self):
        """Test SYSTEM collection performance settings."""
        config = get_type_config(CollectionType.SYSTEM)
        perf = config.performance_settings

        assert perf.batch_size == 50
        assert perf.max_concurrent_operations == 3
        assert perf.priority_weight == 4
        assert perf.cache_ttl_seconds == 600

    def test_library_performance_settings(self):
        """Test LIBRARY collection performance settings."""
        config = get_type_config(CollectionType.LIBRARY)
        perf = config.performance_settings

        assert perf.batch_size == 100
        assert perf.max_concurrent_operations == 5
        assert perf.priority_weight == 3
        assert perf.enable_caching is True

    def test_project_performance_settings(self):
        """Test PROJECT collection performance settings."""
        config = get_type_config(CollectionType.PROJECT)
        perf = config.performance_settings

        assert perf.batch_size == 150
        assert perf.max_concurrent_operations == 10
        assert perf.priority_weight == 2
        assert perf.enable_batch_processing is True

    def test_global_performance_settings(self):
        """Test GLOBAL collection performance settings."""
        config = get_type_config(CollectionType.GLOBAL)
        perf = config.performance_settings

        assert perf.batch_size == 200
        assert perf.max_concurrent_operations == 8
        assert perf.priority_weight == 5


class TestMigrationSettings:
    """Tests for migration settings."""

    def test_system_migration_settings(self):
        """Test SYSTEM collection migration settings."""
        config = get_type_config(CollectionType.SYSTEM)
        migration = config.migration_settings

        assert migration.supports_legacy_format is True
        assert migration.auto_detect_legacy is True
        assert len(migration.legacy_collection_patterns) > 0

    def test_library_migration_settings(self):
        """Test LIBRARY collection migration settings."""
        config = get_type_config(CollectionType.LIBRARY)
        migration = config.migration_settings

        assert migration.supports_legacy_format is True
        assert migration.auto_detect_legacy is True
        assert migration.preserve_legacy_metadata is True

    def test_project_migration_settings(self):
        """Test PROJECT collection migration settings."""
        config = get_type_config(CollectionType.PROJECT)
        migration = config.migration_settings

        assert migration.supports_legacy_format is True
        assert migration.auto_detect_legacy is False
        assert migration.migration_batch_size == 100

    def test_global_migration_settings(self):
        """Test GLOBAL collection migration settings."""
        config = get_type_config(CollectionType.GLOBAL)
        migration = config.migration_settings

        assert migration.supports_legacy_format is False
        assert migration.auto_detect_legacy is False


class TestMetadataValidationIntegration:
    """Integration tests for metadata validation."""

    def test_system_collection_invalid_name_pattern(self):
        """Test SYSTEM collection with invalid name pattern."""
        config = get_type_config(CollectionType.SYSTEM)

        metadata = {
            "collection_name": "no_prefix",  # Missing __ prefix
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection_category": "system",
        }

        is_valid, errors = config.validate_metadata(metadata)
        assert not is_valid
        assert any("does not match required pattern" in error for error in errors)

    def test_library_collection_missing_language(self):
        """Test LIBRARY collection with missing required language field."""
        config = get_type_config(CollectionType.LIBRARY)

        metadata = {
            "collection_name": "_test_lib",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "collection_category": "library",
            # Missing language field
        }

        is_valid, errors = config.validate_metadata(metadata)
        assert not is_valid
        assert any("language" in error.lower() for error in errors)

    def test_project_collection_invalid_project_id_length(self):
        """Test PROJECT collection with invalid project_id length."""
        config = get_type_config(CollectionType.PROJECT)

        metadata = {
            "project_name": "test",
            "project_id": "short",  # Should be exactly 12 characters
            "collection_type": "documents",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        is_valid, errors = config.validate_metadata(metadata)
        assert not is_valid

    def test_global_collection_invalid_name(self):
        """Test GLOBAL collection with invalid name."""
        config = get_type_config(CollectionType.GLOBAL)

        metadata = {
            "collection_name": "invalid_global_name",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        is_valid, errors = config.validate_metadata(metadata)
        assert not is_valid
        assert any("must be one of" in error for error in errors)

    def test_project_collection_with_optional_priority(self):
        """Test PROJECT collection with optional priority field."""
        config = get_type_config(CollectionType.PROJECT)

        # Valid priority
        metadata = {
            "project_name": "test",
            "project_id": "abc123def456",
            "collection_type": "documents",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "priority": 4,
        }

        is_valid, errors = config.validate_metadata(metadata)
        assert is_valid

        # Invalid priority (out of range)
        metadata["priority"] = 10
        is_valid, errors = config.validate_metadata(metadata)
        assert not is_valid
