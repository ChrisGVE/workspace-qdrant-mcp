"""Unit tests for schema-based configuration validation (Task 450).

Tests the config_validator module in common/core covering:
- FieldType and FieldSchema definitions
- ConfigValidator type checking
- Range and pattern validation
- Cross-field dependency validation
- Error message formatting
- Schema documentation generation
"""

import pytest

from common.core.config_validator import (
    FieldType,
    FieldSchema,
    ValidationError,
    DependencyRule,
    ConfigValidator,
    CONFIG_SCHEMAS,
    DEPENDENCY_RULES,
    validate_config,
    validate_config_at_startup,
    get_schema_documentation,
    _validate_url,
    _validate_port,
    _validate_log_level,
    _validate_percentage,
    _validate_checksum_algorithm,
    _validate_deduplication_strategy,
)


class TestFieldType:
    """Tests for FieldType enum."""

    def test_string_type(self):
        """Test STRING field type value."""
        assert FieldType.STRING.value == "string"

    def test_integer_type(self):
        """Test INTEGER field type value."""
        assert FieldType.INTEGER.value == "integer"

    def test_float_type(self):
        """Test FLOAT field type value."""
        assert FieldType.FLOAT.value == "float"

    def test_boolean_type(self):
        """Test BOOLEAN field type value."""
        assert FieldType.BOOLEAN.value == "boolean"

    def test_list_type(self):
        """Test LIST field type value."""
        assert FieldType.LIST.value == "list"

    def test_dict_type(self):
        """Test DICT field type value."""
        assert FieldType.DICT.value == "dict"

    def test_any_type(self):
        """Test ANY field type value."""
        assert FieldType.ANY.value == "any"


class TestFieldSchema:
    """Tests for FieldSchema dataclass."""

    def test_default_values(self):
        """Test FieldSchema default values."""
        schema = FieldSchema(field_type=FieldType.STRING)
        assert schema.required is False
        assert schema.default is None
        assert schema.min_value is None
        assert schema.max_value is None
        assert schema.min_length is None
        assert schema.max_length is None
        assert schema.pattern is None
        assert schema.allowed_values is None

    def test_custom_values(self):
        """Test FieldSchema with custom values."""
        schema = FieldSchema(
            field_type=FieldType.INTEGER,
            required=True,
            default=100,
            min_value=1,
            max_value=1000,
            description="Test field",
        )
        assert schema.field_type == FieldType.INTEGER
        assert schema.required is True
        assert schema.default == 100
        assert schema.min_value == 1
        assert schema.max_value == 1000
        assert schema.description == "Test field"


class TestValidationError:
    """Tests for ValidationError dataclass."""

    def test_basic_error(self):
        """Test basic validation error."""
        error = ValidationError(
            path="qdrant.url",
            message="Invalid URL format",
        )
        assert error.path == "qdrant.url"
        assert error.message == "Invalid URL format"

    def test_error_with_value(self):
        """Test validation error with value."""
        error = ValidationError(
            path="server.port",
            message="Port out of range",
            value=99999,
        )
        assert error.value == 99999

    def test_error_with_suggestion(self):
        """Test validation error with suggestion."""
        error = ValidationError(
            path="embedding.model",
            message="Unknown model",
            suggestion="Check available models at huggingface.co",
        )
        assert error.suggestion == "Check available models at huggingface.co"

    def test_error_str_format(self):
        """Test validation error string formatting."""
        error = ValidationError(
            path="qdrant.url",
            message="Invalid URL",
            value="not-a-url",
            suggestion="Use format http://host:port",
        )
        result = str(error)
        assert "[qdrant.url]" in result
        assert "Invalid URL" in result
        assert "not-a-url" in result
        assert "Use format http://host:port" in result


class TestCustomValidators:
    """Tests for custom validation functions."""

    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        assert _validate_url("http://localhost:6333") is None
        assert _validate_url("https://qdrant.example.com") is None
        assert _validate_url("http://192.168.1.1:8080") is None

    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        assert _validate_url("not-a-url") is not None
        assert _validate_url(123) is not None
        assert _validate_url("ftp://server.com") is not None

    def test_validate_port_valid(self):
        """Test port validation with valid ports."""
        assert _validate_port(80) is None
        assert _validate_port(8080) is None
        assert _validate_port(65535) is None
        assert _validate_port(1) is None

    def test_validate_port_invalid(self):
        """Test port validation with invalid ports."""
        assert _validate_port(0) is not None
        assert _validate_port(65536) is not None
        assert _validate_port(-1) is not None
        assert _validate_port("8080") is not None

    def test_validate_log_level_valid(self):
        """Test log level validation with valid levels."""
        assert _validate_log_level("debug") is None
        assert _validate_log_level("INFO") is None
        assert _validate_log_level("Warning") is None
        assert _validate_log_level("error") is None

    def test_validate_log_level_invalid(self):
        """Test log level validation with invalid levels."""
        assert _validate_log_level("verbose") is not None
        assert _validate_log_level(123) is not None

    def test_validate_percentage_valid(self):
        """Test percentage validation with valid values."""
        assert _validate_percentage(0) is None
        assert _validate_percentage(50) is None
        assert _validate_percentage(100) is None
        assert _validate_percentage(75.5) is None

    def test_validate_percentage_invalid(self):
        """Test percentage validation with invalid values."""
        assert _validate_percentage(-1) is not None
        assert _validate_percentage(101) is not None
        assert _validate_percentage("50%") is not None

    def test_validate_checksum_algorithm_valid(self):
        """Test checksum algorithm validation."""
        assert _validate_checksum_algorithm("xxhash64") is None
        assert _validate_checksum_algorithm("SHA256") is None
        assert _validate_checksum_algorithm("none") is None

    def test_validate_checksum_algorithm_invalid(self):
        """Test checksum algorithm validation with invalid values."""
        assert _validate_checksum_algorithm("invalid") is not None
        assert _validate_checksum_algorithm(123) is not None

    def test_validate_deduplication_strategy_valid(self):
        """Test deduplication strategy validation."""
        assert _validate_deduplication_strategy("path") is None
        assert _validate_deduplication_strategy("content_hash") is None
        assert _validate_deduplication_strategy("mtime") is None

    def test_validate_deduplication_strategy_invalid(self):
        """Test deduplication strategy validation with invalid values."""
        assert _validate_deduplication_strategy("invalid") is not None
        assert _validate_deduplication_strategy(123) is not None


class TestConfigValidator:
    """Tests for ConfigValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ConfigValidator()

    def test_validate_empty_config(self, validator):
        """Test validation of empty configuration."""
        errors = validator.validate({})
        # Should have errors for required fields
        required_errors = [e for e in errors if "Required" in e.message]
        assert len(required_errors) > 0

    def test_validate_valid_qdrant_section(self, validator):
        """Test validation of valid Qdrant configuration."""
        config = {
            "qdrant": {
                "url": "http://localhost:6333",
                "timeout": 30000,
                "max_retries": 3,
            }
        }
        errors = validator.validate_section("qdrant", config["qdrant"])
        assert len(errors) == 0

    def test_validate_invalid_qdrant_url(self, validator):
        """Test validation of invalid Qdrant URL."""
        config = {
            "qdrant": {
                "url": "not-a-valid-url",
            }
        }
        errors = validator.validate_section("qdrant", config["qdrant"])
        url_errors = [e for e in errors if "url" in e.path]
        assert len(url_errors) > 0

    def test_validate_invalid_port(self, validator):
        """Test validation of invalid port number."""
        config = {
            "server": {
                "host": "127.0.0.1",
                "port": 99999,
            }
        }
        errors = validator.validate_section("server", config["server"])
        port_errors = [e for e in errors if "port" in e.path]
        assert len(port_errors) > 0

    def test_validate_integer_out_of_range(self, validator):
        """Test validation of integer out of range."""
        config = {
            "embedding": {
                "chunk_size": 50000,  # Max is 10000
                "batch_size": 100,
                "model": "test-model",
            }
        }
        errors = validator.validate_section("embedding", config["embedding"])
        chunk_errors = [e for e in errors if "chunk_size" in e.path]
        assert len(chunk_errors) > 0
        assert "maximum" in chunk_errors[0].message.lower()

    def test_validate_string_pattern(self, validator):
        """Test validation of string pattern."""
        config = {
            "auto_ingestion": {
                "target_collection_suffix": "invalid@name!",  # Contains invalid chars
            }
        }
        errors = validator.validate_section("auto_ingestion", config["auto_ingestion"])
        pattern_errors = [e for e in errors if "pattern" in e.message.lower()]
        assert len(pattern_errors) > 0

    def test_validate_allowed_values(self, validator):
        """Test validation of allowed values."""
        config = {
            "qdrant": {
                "url": "http://localhost:6333",
                "transport": "invalid_transport",  # Should be http or grpc
            }
        }
        errors = validator.validate_section("qdrant", config["qdrant"])
        transport_errors = [e for e in errors if "transport" in e.path]
        assert len(transport_errors) > 0
        assert "allowed" in transport_errors[0].message.lower()

    def test_validate_list_type(self, validator):
        """Test validation of list type."""
        config = {
            "workspace": {
                "collection_types": "not-a-list",  # Should be a list
            }
        }
        errors = validator.validate_section("workspace", config["workspace"])
        list_errors = [e for e in errors if "collection_types" in e.path]
        assert len(list_errors) > 0

    def test_validate_list_item_type(self, validator):
        """Test validation of list item types."""
        config = {
            "workspace": {
                "collection_types": ["valid", 123, "also-valid"],  # 123 is not a string
            }
        }
        errors = validator.validate_section("workspace", config["workspace"])
        item_errors = [e for e in errors if "[1]" in e.path]  # Second item
        assert len(item_errors) > 0

    def test_validate_list_max_length(self, validator):
        """Test validation of list max length."""
        config = {
            "workspace": {
                "collection_types": [f"type_{i}" for i in range(25)],  # Max is 20
            }
        }
        errors = validator.validate_section("workspace", config["workspace"])
        length_errors = [e for e in errors if "maximum" in e.message.lower()]
        assert len(length_errors) > 0

    def test_validate_boolean_type(self, validator):
        """Test validation of boolean type."""
        config = {
            "grpc": {
                "enabled": "yes",  # Should be a boolean
            }
        }
        errors = validator.validate_section("grpc", config["grpc"])
        bool_errors = [e for e in errors if "enabled" in e.path]
        assert len(bool_errors) > 0

    def test_validate_float_type(self, validator):
        """Test validation of float type."""
        config = {
            "auto_ingestion": {
                "batch_delay_seconds": "not-a-number",
            }
        }
        errors = validator.validate_section("auto_ingestion", config["auto_ingestion"])
        float_errors = [e for e in errors if "batch_delay_seconds" in e.path]
        assert len(float_errors) > 0

    def test_validate_unknown_section(self, validator):
        """Test validation of unknown section."""
        errors = validator.validate_section("unknown_section", {"key": "value"})
        assert len(errors) == 1
        assert "Unknown" in errors[0].message

    def test_validate_with_custom_validator(self, validator):
        """Test validation with custom validator function."""
        config = {
            "logging": {
                "level": "invalid_level",
            }
        }
        errors = validator.validate_section("logging", config["logging"])
        level_errors = [e for e in errors if "level" in e.path]
        assert len(level_errors) > 0


class TestDependencyValidation:
    """Tests for cross-field dependency validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ConfigValidator()

    def test_chunk_overlap_less_than_size(self, validator):
        """Test chunk_overlap must be less than chunk_size."""
        config = {
            "embedding": {
                "model": "test-model",
                "chunk_size": 500,
                "chunk_overlap": 600,  # Greater than chunk_size
                "batch_size": 32,
            }
        }
        errors = validator.validate(config)
        dep_errors = [e for e in errors if "chunk_overlap" in e.message and "chunk_size" in e.message]
        assert len(dep_errors) > 0

    def test_chunk_overlap_valid(self, validator):
        """Test valid chunk_overlap less than chunk_size."""
        config = {
            "embedding": {
                "model": "test-model",
                "chunk_size": 500,
                "chunk_overlap": 100,
                "batch_size": 32,
            }
        }
        errors = validator.validate(config)
        dep_errors = [e for e in errors if "chunk_overlap" in e.message and "chunk_size" in e.message]
        assert len(dep_errors) == 0

    def test_grpc_port_required_when_enabled(self, validator):
        """Test gRPC port must be set when enabled."""
        config = {
            "grpc": {
                "enabled": True,
                "port": None,  # Port should be set
            }
        }
        errors = validator.validate(config)
        grpc_errors = [e for e in errors if "grpc.port" in e.path or "grpc.port" in e.message]
        assert len(grpc_errors) > 0

    def test_file_logging_path_required(self, validator):
        """Test file_path should be set when file logging is enabled."""
        config = {
            "logging": {
                "use_file_logging": True,
                "file_path": None,  # Should be set
            }
        }
        errors = validator.validate(config)
        path_errors = [e for e in errors if "file_path" in e.message]
        assert len(path_errors) > 0

    def test_timeout_dependencies(self, validator):
        """Test timeout value dependencies."""
        config = {
            "optimized_ingestion": {
                "file_timeout_seconds": 100.0,
                "batch_timeout_seconds": 50.0,  # Should be >= file_timeout
                "operation_timeout_seconds": 200.0,
            }
        }
        errors = validator.validate(config)
        timeout_errors = [e for e in errors if "batch_timeout" in e.message]
        assert len(timeout_errors) > 0


class TestValidateConfigFunction:
    """Tests for validate_config convenience function."""

    def test_validate_config_returns_strings(self):
        """Test validate_config returns list of strings."""
        config = {
            "qdrant": {
                "url": "invalid",
            }
        }
        errors = validate_config(config)
        assert isinstance(errors, list)
        assert all(isinstance(e, str) for e in errors)

    def test_validate_config_empty_on_valid(self):
        """Test validate_config returns empty list for valid config."""
        config = {
            "qdrant": {
                "url": "http://localhost:6333",
                "timeout": 30000,
            },
            "embedding": {
                "model": "test-model",
                "chunk_size": 800,
                "chunk_overlap": 100,
                "batch_size": 32,
            },
            "server": {
                "host": "127.0.0.1",
                "port": 8000,
            },
        }
        errors = validate_config(config)
        # Filter out only critical errors (not warnings about optional fields)
        critical_errors = [e for e in errors if "Required" in e]
        assert len(critical_errors) == 0


class TestValidateConfigAtStartup:
    """Tests for validate_config_at_startup function."""

    def test_returns_errors_without_raising(self):
        """Test function returns errors without raising by default."""
        config = {"qdrant": {"url": "invalid"}}
        errors = validate_config_at_startup(config, raise_on_error=False)
        assert isinstance(errors, list)
        assert len(errors) > 0

    def test_raises_on_error_when_requested(self):
        """Test function raises ValueError when requested."""
        config = {"qdrant": {"url": "invalid"}}
        with pytest.raises(ValueError) as exc_info:
            validate_config_at_startup(config, raise_on_error=True)
        assert "validation failed" in str(exc_info.value).lower()

    def test_no_raise_on_valid_config(self):
        """Test function does not raise on valid config."""
        config = {
            "qdrant": {"url": "http://localhost:6333"},
            "embedding": {
                "model": "test-model",
                "chunk_size": 800,
                "chunk_overlap": 100,
                "batch_size": 32,
            },
            "server": {"host": "127.0.0.1", "port": 8000},
        }
        # Should not raise
        errors = validate_config_at_startup(config, raise_on_error=True)
        # May have warnings but no critical required field errors
        critical = [e for e in errors if "Required" in str(e)]
        assert len(critical) == 0


class TestSchemaDocumentation:
    """Tests for schema documentation generation."""

    def test_documentation_generated(self):
        """Test documentation is generated."""
        docs = get_schema_documentation()
        assert isinstance(docs, str)
        assert len(docs) > 0

    def test_documentation_contains_sections(self):
        """Test documentation contains all sections."""
        docs = get_schema_documentation()
        for section_name in CONFIG_SCHEMAS:
            assert f"[{section_name}]" in docs

    def test_documentation_contains_field_types(self):
        """Test documentation contains field type information."""
        docs = get_schema_documentation()
        assert "string" in docs.lower()
        assert "integer" in docs.lower()
        assert "boolean" in docs.lower()

    def test_documentation_contains_required_marker(self):
        """Test documentation marks required fields."""
        docs = get_schema_documentation()
        assert "(required)" in docs.lower()


class TestConfigSchemas:
    """Tests for CONFIG_SCHEMAS definitions."""

    def test_qdrant_schema_exists(self):
        """Test Qdrant schema is defined."""
        assert "qdrant" in CONFIG_SCHEMAS
        assert "url" in CONFIG_SCHEMAS["qdrant"]

    def test_server_schema_exists(self):
        """Test server schema is defined."""
        assert "server" in CONFIG_SCHEMAS
        assert "host" in CONFIG_SCHEMAS["server"]
        assert "port" in CONFIG_SCHEMAS["server"]

    def test_embedding_schema_exists(self):
        """Test embedding schema is defined."""
        assert "embedding" in CONFIG_SCHEMAS
        assert "model" in CONFIG_SCHEMAS["embedding"]
        assert "chunk_size" in CONFIG_SCHEMAS["embedding"]

    def test_grpc_schema_exists(self):
        """Test gRPC schema is defined."""
        assert "grpc" in CONFIG_SCHEMAS
        assert "enabled" in CONFIG_SCHEMAS["grpc"]
        assert "port" in CONFIG_SCHEMAS["grpc"]

    def test_auto_ingestion_schema_exists(self):
        """Test auto_ingestion schema is defined."""
        assert "auto_ingestion" in CONFIG_SCHEMAS
        assert "enabled" in CONFIG_SCHEMAS["auto_ingestion"]

    def test_optimized_ingestion_schema_exists(self):
        """Test optimized_ingestion schema is defined."""
        assert "optimized_ingestion" in CONFIG_SCHEMAS
        assert "max_concurrent_files" in CONFIG_SCHEMAS["optimized_ingestion"]

    def test_queue_processor_schema_exists(self):
        """Test queue_processor schema is defined."""
        assert "queue_processor" in CONFIG_SCHEMAS
        assert "batch_size" in CONFIG_SCHEMAS["queue_processor"]

    def test_logging_schema_exists(self):
        """Test logging schema is defined."""
        assert "logging" in CONFIG_SCHEMAS
        assert "level" in CONFIG_SCHEMAS["logging"]

    def test_backup_schema_exists(self):
        """Test backup schema is defined."""
        assert "backup" in CONFIG_SCHEMAS
        assert "enabled" in CONFIG_SCHEMAS["backup"]


class TestDependencyRules:
    """Tests for DEPENDENCY_RULES definitions."""

    def test_rules_exist(self):
        """Test dependency rules are defined."""
        assert isinstance(DEPENDENCY_RULES, list)
        assert len(DEPENDENCY_RULES) > 0

    def test_chunk_overlap_rule_exists(self):
        """Test chunk_overlap dependency rule exists."""
        rule = next(
            (r for r in DEPENDENCY_RULES if "chunk_overlap" in r.target_path),
            None
        )
        assert rule is not None

    def test_grpc_port_rule_exists(self):
        """Test gRPC port dependency rule exists."""
        rule = next(
            (r for r in DEPENDENCY_RULES if "grpc" in r.source_path),
            None
        )
        assert rule is not None

    def test_rules_have_messages(self):
        """Test all rules have error messages."""
        for rule in DEPENDENCY_RULES:
            assert rule.message is not None
            assert len(rule.message) > 0


class TestTypeValidation:
    """Tests for type validation edge cases."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ConfigValidator()

    def test_boolean_not_confused_with_int(self, validator):
        """Test boolean values are not confused with integers."""
        # In Python, bool is subclass of int, so True == 1
        # But for config validation, we want to distinguish them
        config = {
            "grpc": {
                "enabled": 1,  # Should be boolean, not int
            }
        }
        errors = validator.validate_section("grpc", config["grpc"])
        type_errors = [e for e in errors if "enabled" in e.path and "type" in e.message.lower()]
        assert len(type_errors) > 0

    def test_float_accepts_int(self, validator):
        """Test float fields accept integer values."""
        config = {
            "auto_ingestion": {
                "batch_delay_seconds": 2,  # Int instead of float
            }
        }
        errors = validator.validate_section("auto_ingestion", config["auto_ingestion"])
        type_errors = [e for e in errors if "batch_delay_seconds" in e.path and "type" in e.message.lower()]
        assert len(type_errors) == 0

    def test_any_type_accepts_all(self, validator):
        """Test ANY type accepts all values."""
        # Create custom schema with ANY type
        custom_schema = {
            "test_section": {
                "any_field": FieldSchema(
                    field_type=FieldType.ANY,
                    description="Accepts any type",
                ),
            }
        }
        custom_validator = ConfigValidator(schemas=custom_schema, dependency_rules=[])

        # Test with different types
        for value in ["string", 123, 45.6, True, ["list"], {"dict": "value"}]:
            errors = custom_validator.validate_section(
                "test_section",
                {"any_field": value}
            )
            type_errors = [e for e in errors if "type" in e.message.lower()]
            assert len(type_errors) == 0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ConfigValidator()

    def test_empty_string_validation(self, validator):
        """Test validation of empty strings."""
        config = {
            "qdrant": {
                "url": "",  # Empty URL
            }
        }
        errors = validator.validate_section("qdrant", config["qdrant"])
        url_errors = [e for e in errors if "url" in e.path]
        assert len(url_errors) > 0

    def test_negative_values(self, validator):
        """Test validation of negative values where positive is expected."""
        config = {
            "embedding": {
                "model": "test-model",
                "chunk_size": -100,  # Negative
                "batch_size": 32,
            }
        }
        errors = validator.validate_section("embedding", config["embedding"])
        negative_errors = [e for e in errors if "minimum" in e.message.lower()]
        assert len(negative_errors) > 0

    def test_boundary_values(self, validator):
        """Test validation at boundary values."""
        # Test at exact minimum
        config = {
            "server": {
                "host": "127.0.0.1",
                "port": 1,  # Minimum valid port
            }
        }
        errors = validator.validate_section("server", config["server"])
        port_errors = [e for e in errors if "port" in e.path]
        assert len(port_errors) == 0

        # Test at exact maximum
        config["server"]["port"] = 65535  # Maximum valid port
        errors = validator.validate_section("server", config["server"])
        port_errors = [e for e in errors if "port" in e.path]
        assert len(port_errors) == 0

    def test_none_values(self, validator):
        """Test validation of None values."""
        config = {
            "qdrant": {
                "url": None,  # Required field is None
            }
        }
        errors = validator.validate_section("qdrant", config["qdrant"])
        required_errors = [e for e in errors if "Required" in e.message]
        assert len(required_errors) > 0

    def test_nested_dict_missing_key(self, validator):
        """Test validation when nested dict key is missing."""
        config = {
            "qdrant": {
                # url is missing
                "timeout": 30000,
            }
        }
        errors = validator.validate_section("qdrant", config["qdrant"])
        url_errors = [e for e in errors if "url" in e.path and "Required" in e.message]
        assert len(url_errors) > 0

    def test_special_characters_in_patterns(self, validator):
        """Test pattern validation with special characters."""
        config = {
            "server": {
                "host": "my-server.local",  # Valid hostname with dash and dot
                "port": 8080,
            }
        }
        errors = validator.validate_section("server", config["server"])
        host_errors = [e for e in errors if "host" in e.path and "pattern" in e.message.lower()]
        assert len(host_errors) == 0

    def test_empty_list(self, validator):
        """Test validation of empty lists."""
        config = {
            "workspace": {
                "collection_types": [],  # Empty list is valid
            }
        }
        errors = validator.validate_section("workspace", config["workspace"])
        list_errors = [e for e in errors if "collection_types" in e.path]
        assert len(list_errors) == 0

    def test_full_valid_configuration(self, validator):
        """Test validation of a complete valid configuration."""
        config = {
            "deployment": {"develop": True},
            "server": {"host": "127.0.0.1", "port": 8000, "debug": False},
            "qdrant": {"url": "http://localhost:6333", "timeout": 30000, "max_retries": 3},
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 800,
                "chunk_overlap": 100,
                "batch_size": 32,
            },
            "workspace": {"collection_types": ["code", "notes"], "global_collections": []},
            "grpc": {"enabled": True, "host": "127.0.0.1", "port": 50051},
            "auto_ingestion": {"enabled": True, "max_files_per_batch": 5},
            "optimized_ingestion": {
                "max_concurrent_files": 10,
                "batch_size": 50,
                "file_timeout_seconds": 60.0,
                "batch_timeout_seconds": 300.0,
                "operation_timeout_seconds": 3600.0,
            },
            "queue_processor": {"batch_size": 10, "poll_interval_ms": 500},
            "logging": {"level": "info", "use_file_logging": False},
            "performance": {"max_concurrent_tasks": 4, "default_timeout_ms": 30000},
            "backup": {"enabled": True, "retention_days": 30},
        }
        errors = validator.validate(config)
        # Should have no errors
        assert len(errors) == 0


class TestNestedValueAccess:
    """Tests for nested value access in dependency validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return ConfigValidator()

    def test_get_nested_value_success(self, validator):
        """Test successful nested value access."""
        config = {
            "embedding": {
                "chunk_size": 800,
            }
        }
        value = validator._get_nested_value(config, "embedding.chunk_size")
        assert value == 800

    def test_get_nested_value_missing(self, validator):
        """Test nested value access for missing path."""
        config = {"embedding": {}}
        value = validator._get_nested_value(config, "embedding.missing_field")
        assert value is None

    def test_get_nested_value_partial_path(self, validator):
        """Test nested value access with partial path."""
        config = {}
        value = validator._get_nested_value(config, "missing.nested.path")
        assert value is None


class TestCustomSchemaValidator:
    """Tests for custom schema validation."""

    def test_custom_schema(self):
        """Test validation with custom schema."""
        custom_schema = {
            "custom": {
                "field1": FieldSchema(
                    field_type=FieldType.STRING,
                    required=True,
                    description="Custom field",
                ),
                "field2": FieldSchema(
                    field_type=FieldType.INTEGER,
                    min_value=0,
                    max_value=100,
                    description="Another field",
                ),
            }
        }
        validator = ConfigValidator(schemas=custom_schema, dependency_rules=[])

        # Valid config
        config = {"custom": {"field1": "value", "field2": 50}}
        errors = validator.validate(config)
        assert len(errors) == 0

        # Invalid config - missing required
        config = {"custom": {"field2": 50}}
        errors = validator.validate(config)
        assert any("Required" in str(e) for e in errors)

        # Invalid config - out of range
        config = {"custom": {"field1": "value", "field2": 150}}
        errors = validator.validate(config)
        assert any("maximum" in str(e).lower() for e in errors)

    def test_custom_dependency_rules(self):
        """Test validation with custom dependency rules."""
        custom_schema = {
            "test": {
                "min": FieldSchema(field_type=FieldType.INTEGER),
                "max": FieldSchema(field_type=FieldType.INTEGER),
            }
        }
        custom_rules = [
            DependencyRule(
                source_path="test.min",
                target_path="test.max",
                condition=lambda _: True,
                requirement=lambda min_val, max_val: max_val >= min_val,
                message="test.max must be >= test.min",
            )
        ]
        validator = ConfigValidator(schemas=custom_schema, dependency_rules=custom_rules)

        # Valid config
        config = {"test": {"min": 10, "max": 20}}
        errors = validator.validate(config)
        dep_errors = [e for e in errors if "must be" in str(e)]
        assert len(dep_errors) == 0

        # Invalid config
        config = {"test": {"min": 20, "max": 10}}
        errors = validator.validate(config)
        dep_errors = [e for e in errors if "must be" in str(e)]
        assert len(dep_errors) > 0
