"""Comprehensive configuration validation for workspace-qdrant-mcp.

This module provides schema-based validation for all configuration values at startup.
It ensures type checking, range validation, dependency validation, and provides
clear error messages for invalid configurations.

Features:
    - Schema-based validation using declarative field definitions
    - Type checking with support for complex types (Optional, List, Dict)
    - Range validation for numeric fields
    - Pattern validation for string fields (URLs, paths, identifiers)
    - Cross-field dependency validation
    - Clear, actionable error messages
    - Integration with ConfigManager

Example:
    ```python
    from common.core.config_validator import ConfigValidator, validate_config

    # Validate entire configuration
    errors = validate_config(config_dict)
    if errors:
        for error in errors:
            print(f"Config error: {error}")

    # Validate specific section
    validator = ConfigValidator()
    errors = validator.validate_section("qdrant", qdrant_config)
    ```
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class FieldType(Enum):
    """Supported field types for configuration validation."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    ANY = "any"


@dataclass
class FieldSchema:
    """Schema definition for a configuration field.

    Attributes:
        field_type: Expected type of the field
        required: Whether the field is required
        default: Default value if not provided
        min_value: Minimum value for numeric fields
        max_value: Maximum value for numeric fields
        min_length: Minimum length for string/list fields
        max_length: Maximum length for string/list fields
        pattern: Regex pattern for string validation
        allowed_values: List of allowed values (enum-like)
        item_type: Type of items in a list
        nested_schema: Schema for nested dict fields
        description: Human-readable description
        custom_validator: Custom validation function
    """

    field_type: FieldType
    required: bool = False
    default: Any = None
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    allowed_values: list[Any] | None = None
    item_type: FieldType | None = None
    nested_schema: dict[str, "FieldSchema"] | None = None
    description: str = ""
    custom_validator: Callable[[Any], str | None] | None = None


@dataclass
class ValidationError:
    """Represents a configuration validation error.

    Attributes:
        path: Dot-separated path to the invalid field
        message: Human-readable error message
        value: The invalid value that was found
        suggestion: Suggested fix for the error
    """

    path: str
    message: str
    value: Any = None
    suggestion: str = ""

    def __str__(self) -> str:
        """Format error as string."""
        result = f"[{self.path}] {self.message}"
        if self.value is not None:
            result += f" (got: {self.value!r})"
        if self.suggestion:
            result += f". {self.suggestion}"
        return result


@dataclass
class DependencyRule:
    """Rule for cross-field dependency validation.

    Attributes:
        source_path: Path to the source field
        target_path: Path to the dependent field
        condition: Condition function (source_value) -> bool
        requirement: What the target field must satisfy
        message: Error message if validation fails
    """

    source_path: str
    target_path: str
    condition: Callable[[Any], bool]
    requirement: Callable[[Any, Any], bool]
    message: str


# URL pattern for validation
URL_PATTERN = r"^https?://[^\s/$.?#].[^\s]*$"
# Host pattern (IP or hostname)
HOST_PATTERN = r"^[a-zA-Z0-9][a-zA-Z0-9\-\.]*$"
# Collection name pattern
COLLECTION_NAME_PATTERN = r"^[a-zA-Z_][a-zA-Z0-9_\-]*$"
# Path pattern
PATH_PATTERN = r"^[^<>:\"|?*]+$"


def _validate_url(value: Any) -> str | None:
    """Custom validator for URLs."""
    if not isinstance(value, str):
        return "URL must be a string"
    if not value.startswith(("http://", "https://")):
        return "URL must start with http:// or https://"
    return None


def _validate_port(value: Any) -> str | None:
    """Custom validator for port numbers."""
    if not isinstance(value, int):
        return "Port must be an integer"
    if value < 1 or value > 65535:
        return "Port must be between 1 and 65535"
    return None


def _validate_log_level(value: Any) -> str | None:
    """Custom validator for log levels."""
    if not isinstance(value, str):
        return "Log level must be a string"
    valid_levels = ["trace", "debug", "info", "warn", "warning", "error", "critical"]
    if value.lower() not in valid_levels:
        return f"Log level must be one of: {', '.join(valid_levels)}"
    return None


def _validate_percentage(value: Any) -> str | None:
    """Custom validator for percentage values (0-100)."""
    if not isinstance(value, (int, float)):
        return "Percentage must be a number"
    if value < 0 or value > 100:
        return "Percentage must be between 0 and 100"
    return None


def _validate_checksum_algorithm(value: Any) -> str | None:
    """Custom validator for checksum algorithms."""
    if not isinstance(value, str):
        return "Checksum algorithm must be a string"
    valid_algorithms = ["xxhash64", "sha256", "sha1", "md5", "none"]
    if value.lower() not in valid_algorithms:
        return f"Checksum algorithm must be one of: {', '.join(valid_algorithms)}"
    return None


def _validate_deduplication_strategy(value: Any) -> str | None:
    """Custom validator for deduplication strategies."""
    if not isinstance(value, str):
        return "Deduplication strategy must be a string"
    valid_strategies = ["path", "content_hash", "mtime"]
    if value.lower() not in valid_strategies:
        return f"Deduplication strategy must be one of: {', '.join(valid_strategies)}"
    return None


# Configuration schemas for all sections
CONFIG_SCHEMAS: dict[str, dict[str, FieldSchema]] = {
    "deployment": {
        "develop": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=False,
            description="Enable development mode",
        ),
        "base_path": FieldSchema(
            field_type=FieldType.STRING,
            description="Base path for assets and configuration",
        ),
    },
    "server": {
        "host": FieldSchema(
            field_type=FieldType.STRING,
            required=True,
            default="127.0.0.1",
            pattern=HOST_PATTERN,
            description="Server host address",
        ),
        "port": FieldSchema(
            field_type=FieldType.INTEGER,
            required=True,
            default=8000,
            custom_validator=_validate_port,
            description="Server port number",
        ),
        "debug": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=False,
            description="Enable debug mode",
        ),
    },
    "qdrant": {
        "url": FieldSchema(
            field_type=FieldType.STRING,
            required=True,
            default="http://localhost:6333",
            custom_validator=_validate_url,
            description="Qdrant server URL",
        ),
        "api_key": FieldSchema(
            field_type=FieldType.STRING,
            description="Qdrant API key for authentication",
        ),
        "timeout": FieldSchema(
            field_type=FieldType.INTEGER,
            default=30000,
            min_value=1000,
            max_value=300000,
            description="Connection timeout in milliseconds",
        ),
        "transport": FieldSchema(
            field_type=FieldType.STRING,
            default="http",
            allowed_values=["http", "grpc"],
            description="Transport protocol for Qdrant communication",
        ),
        "max_retries": FieldSchema(
            field_type=FieldType.INTEGER,
            default=3,
            min_value=0,
            max_value=20,
            description="Maximum retry attempts for failed operations",
        ),
        "check_compatibility": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Check Qdrant version compatibility at startup",
        ),
    },
    "embedding": {
        "model": FieldSchema(
            field_type=FieldType.STRING,
            required=True,
            default="sentence-transformers/all-MiniLM-L6-v2",
            description="Embedding model name or path",
        ),
        "chunk_size": FieldSchema(
            field_type=FieldType.INTEGER,
            required=True,
            default=800,
            min_value=100,
            max_value=10000,
            description="Chunk size for text splitting",
        ),
        "chunk_overlap": FieldSchema(
            field_type=FieldType.INTEGER,
            default=100,
            min_value=0,
            max_value=5000,
            description="Overlap between chunks",
        ),
        "batch_size": FieldSchema(
            field_type=FieldType.INTEGER,
            default=32,
            min_value=1,
            max_value=1000,
            description="Batch size for embedding generation",
        ),
        "max_tokens": FieldSchema(
            field_type=FieldType.INTEGER,
            default=512,
            min_value=64,
            max_value=8192,
            description="Maximum tokens per embedding",
        ),
        "normalize_embeddings": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Normalize embedding vectors",
        ),
    },
    "workspace": {
        "collection_types": FieldSchema(
            field_type=FieldType.LIST,
            default=[],
            item_type=FieldType.STRING,
            max_length=20,
            description="List of supported collection types",
        ),
        "global_collections": FieldSchema(
            field_type=FieldType.LIST,
            default=[],
            item_type=FieldType.STRING,
            max_length=50,
            description="List of global collection names",
        ),
        "project_root_markers": FieldSchema(
            field_type=FieldType.LIST,
            default=[".git", "package.json", "pyproject.toml", "Cargo.toml"],
            item_type=FieldType.STRING,
            description="Files that indicate project root",
        ),
    },
    "grpc": {
        "enabled": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable gRPC service",
        ),
        "host": FieldSchema(
            field_type=FieldType.STRING,
            default="127.0.0.1",
            pattern=HOST_PATTERN,
            description="gRPC server host",
        ),
        "port": FieldSchema(
            field_type=FieldType.INTEGER,
            default=50051,
            custom_validator=_validate_port,
            description="gRPC server port",
        ),
        "max_message_size": FieldSchema(
            field_type=FieldType.INTEGER,
            default=104857600,  # 100MB
            min_value=1048576,  # 1MB
            max_value=1073741824,  # 1GB
            description="Maximum gRPC message size in bytes",
        ),
        "keepalive_time_ms": FieldSchema(
            field_type=FieldType.INTEGER,
            default=60000,
            min_value=1000,
            max_value=3600000,
            description="Keepalive ping interval in milliseconds",
        ),
    },
    "auto_ingestion": {
        "enabled": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable automatic file ingestion",
        ),
        "auto_create_watches": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Automatically create watch folders for projects",
        ),
        "include_common_files": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Include common files like README, LICENSE",
        ),
        "include_source_files": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Include source code files",
        ),
        "target_collection_suffix": FieldSchema(
            field_type=FieldType.STRING,
            default="scratchbook",
            pattern=COLLECTION_NAME_PATTERN,
            description="Suffix for auto-created collections",
        ),
        "max_files_per_batch": FieldSchema(
            field_type=FieldType.INTEGER,
            default=5,
            min_value=1,
            max_value=100,
            description="Maximum files to process per batch",
        ),
        "batch_delay_seconds": FieldSchema(
            field_type=FieldType.FLOAT,
            default=2.0,
            min_value=0.1,
            max_value=60.0,
            description="Delay between batches in seconds",
        ),
        "max_file_size_mb": FieldSchema(
            field_type=FieldType.INTEGER,
            default=50,
            min_value=1,
            max_value=1000,
            description="Maximum file size in MB",
        ),
        "recursive_depth": FieldSchema(
            field_type=FieldType.INTEGER,
            default=5,
            min_value=1,
            max_value=50,
            description="Maximum directory recursion depth",
        ),
        "debounce_seconds": FieldSchema(
            field_type=FieldType.INTEGER,
            default=10,
            min_value=1,
            max_value=300,
            description="Debounce time for file changes",
        ),
    },
    "optimized_ingestion": {
        "max_concurrent_files": FieldSchema(
            field_type=FieldType.INTEGER,
            default=10,
            min_value=1,
            max_value=50,
            description="Maximum concurrent file processing",
        ),
        "batch_size": FieldSchema(
            field_type=FieldType.INTEGER,
            default=50,
            min_value=1,
            max_value=500,
            description="Batch processing size",
        ),
        "adaptive_batch_sizing": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable adaptive batch sizing",
        ),
        "enable_deduplication": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable file deduplication",
        ),
        "deduplication_strategy": FieldSchema(
            field_type=FieldType.STRING,
            default="content_hash",
            custom_validator=_validate_deduplication_strategy,
            description="Deduplication strategy",
        ),
        "progress_interval_seconds": FieldSchema(
            field_type=FieldType.FLOAT,
            default=1.0,
            min_value=0.1,
            max_value=60.0,
            description="Progress report interval",
        ),
        "enable_progress_callbacks": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable progress callbacks",
        ),
        "max_memory_usage_mb": FieldSchema(
            field_type=FieldType.FLOAT,
            default=512.0,
            min_value=100.0,
            max_value=2048.0,
            description="Maximum memory usage in MB",
        ),
        "max_cpu_percent": FieldSchema(
            field_type=FieldType.FLOAT,
            default=80.0,
            min_value=10.0,
            max_value=100.0,
            description="Maximum CPU usage percent",
        ),
        "file_timeout_seconds": FieldSchema(
            field_type=FieldType.FLOAT,
            default=60.0,
            min_value=5.0,
            max_value=600.0,
            description="Per-file processing timeout",
        ),
        "batch_timeout_seconds": FieldSchema(
            field_type=FieldType.FLOAT,
            default=300.0,
            min_value=30.0,
            max_value=3600.0,
            description="Batch processing timeout",
        ),
        "operation_timeout_seconds": FieldSchema(
            field_type=FieldType.FLOAT,
            default=3600.0,
            min_value=60.0,
            max_value=86400.0,
            description="Total operation timeout",
        ),
        "max_retries": FieldSchema(
            field_type=FieldType.INTEGER,
            default=3,
            min_value=0,
            max_value=10,
            description="Maximum retry attempts",
        ),
        "retry_delay_seconds": FieldSchema(
            field_type=FieldType.FLOAT,
            default=1.0,
            min_value=0.1,
            max_value=60.0,
            description="Delay between retries",
        ),
    },
    "queue_processor": {
        "batch_size": FieldSchema(
            field_type=FieldType.INTEGER,
            default=10,
            min_value=1,
            max_value=1000,
            description="Items to dequeue per batch",
        ),
        "poll_interval_ms": FieldSchema(
            field_type=FieldType.INTEGER,
            default=500,
            min_value=1,
            max_value=60000,
            description="Poll interval in milliseconds",
        ),
        "max_retries": FieldSchema(
            field_type=FieldType.INTEGER,
            default=5,
            min_value=0,
            max_value=20,
            description="Maximum retry attempts",
        ),
        "retry_delays_seconds": FieldSchema(
            field_type=FieldType.LIST,
            default=[60, 300, 900, 3600],
            item_type=FieldType.INTEGER,
            max_length=10,
            description="Retry delay progression in seconds",
        ),
        "target_throughput": FieldSchema(
            field_type=FieldType.INTEGER,
            default=1000,
            min_value=1,
            max_value=100000,
            description="Target docs per minute",
        ),
        "enable_metrics": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable metrics logging",
        ),
    },
    "logging": {
        "level": FieldSchema(
            field_type=FieldType.STRING,
            default="info",
            custom_validator=_validate_log_level,
            description="Default log level",
        ),
        "use_file_logging": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=False,
            description="Enable file logging",
        ),
        "file_path": FieldSchema(
            field_type=FieldType.STRING,
            description="Log file path",
        ),
        "max_file_size_mb": FieldSchema(
            field_type=FieldType.INTEGER,
            default=10,
            min_value=1,
            max_value=1000,
            description="Maximum log file size in MB",
        ),
        "backup_count": FieldSchema(
            field_type=FieldType.INTEGER,
            default=5,
            min_value=0,
            max_value=100,
            description="Number of log file backups to keep",
        ),
        "info_includes_connection_events": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Include connection events in info logs",
        ),
        "info_includes_transport_details": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Include transport details in info logs",
        ),
        "error_includes_stack_trace": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Include stack trace in error logs",
        ),
    },
    "performance": {
        "max_concurrent_tasks": FieldSchema(
            field_type=FieldType.INTEGER,
            default=4,
            min_value=1,
            max_value=64,
            description="Maximum concurrent tasks",
        ),
        "default_timeout_ms": FieldSchema(
            field_type=FieldType.INTEGER,
            default=30000,
            min_value=1000,
            max_value=600000,
            description="Default operation timeout in milliseconds",
        ),
        "enable_preemption": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable task preemption",
        ),
    },
    "backup": {
        "enabled": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable backup functionality",
        ),
        "auto_backup_before_restore": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Auto backup before restore operations",
        ),
        "default_backup_directory": FieldSchema(
            field_type=FieldType.STRING,
            description="Default backup directory path",
        ),
        "retention_days": FieldSchema(
            field_type=FieldType.INTEGER,
            default=30,
            min_value=0,
            max_value=365,
            description="Backup retention period in days",
        ),
        "compression": FieldSchema(
            field_type=FieldType.BOOLEAN,
            default=True,
            description="Enable backup compression",
        ),
    },
}

# Dependency rules for cross-field validation
DEPENDENCY_RULES: list[DependencyRule] = [
    # chunk_overlap must be less than chunk_size
    DependencyRule(
        source_path="embedding.chunk_size",
        target_path="embedding.chunk_overlap",
        condition=lambda _: True,  # Always check
        requirement=lambda chunk_size, overlap: overlap < chunk_size,
        message="embedding.chunk_overlap must be less than embedding.chunk_size",
    ),
    # If gRPC is enabled, port must be set
    DependencyRule(
        source_path="grpc.enabled",
        target_path="grpc.port",
        condition=lambda enabled: enabled is True,
        requirement=lambda _, port: port is not None and isinstance(port, int),
        message="grpc.port must be set when grpc.enabled is true",
    ),
    # If file logging is enabled, file_path should be set
    DependencyRule(
        source_path="logging.use_file_logging",
        target_path="logging.file_path",
        condition=lambda enabled: enabled is True,
        requirement=lambda _, path: path is not None and isinstance(path, str) and len(path) > 0,
        message="logging.file_path should be set when logging.use_file_logging is true",
    ),
    # batch_timeout should be greater than file_timeout * batch_size
    DependencyRule(
        source_path="optimized_ingestion.file_timeout_seconds",
        target_path="optimized_ingestion.batch_timeout_seconds",
        condition=lambda _: True,
        requirement=lambda file_timeout, batch_timeout: batch_timeout >= file_timeout,
        message="optimized_ingestion.batch_timeout_seconds should be >= file_timeout_seconds",
    ),
    # operation_timeout should be greater than batch_timeout
    DependencyRule(
        source_path="optimized_ingestion.batch_timeout_seconds",
        target_path="optimized_ingestion.operation_timeout_seconds",
        condition=lambda _: True,
        requirement=lambda batch_timeout, op_timeout: op_timeout >= batch_timeout,
        message="optimized_ingestion.operation_timeout_seconds should be >= batch_timeout_seconds",
    ),
]


class ConfigValidator:
    """Configuration validator with schema-based validation.

    This class validates configuration dictionaries against predefined schemas,
    checking types, ranges, patterns, and cross-field dependencies.
    """

    def __init__(
        self,
        schemas: dict[str, dict[str, FieldSchema]] | None = None,
        dependency_rules: list[DependencyRule] | None = None,
    ):
        """Initialize the validator.

        Args:
            schemas: Custom schemas (uses CONFIG_SCHEMAS if not provided)
            dependency_rules: Custom dependency rules (uses DEPENDENCY_RULES if not provided)
        """
        self.schemas = schemas or CONFIG_SCHEMAS
        self.dependency_rules = dependency_rules or DEPENDENCY_RULES

    def validate(self, config: dict[str, Any]) -> list[ValidationError]:
        """Validate entire configuration dictionary.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[ValidationError] = []

        # Validate each section
        for section_name, section_schema in self.schemas.items():
            section_config = config.get(section_name, {})
            if not isinstance(section_config, dict):
                section_config = {}

            section_errors = self._validate_section(
                section_name, section_config, section_schema
            )
            errors.extend(section_errors)

        # Validate cross-field dependencies
        dependency_errors = self._validate_dependencies(config)
        errors.extend(dependency_errors)

        return errors

    def validate_section(
        self, section_name: str, section_config: dict[str, Any]
    ) -> list[ValidationError]:
        """Validate a specific configuration section.

        Args:
            section_name: Name of the section to validate
            section_config: Section configuration dictionary

        Returns:
            List of validation errors for this section
        """
        if section_name not in self.schemas:
            return [
                ValidationError(
                    path=section_name,
                    message=f"Unknown configuration section",
                    suggestion=f"Valid sections: {', '.join(self.schemas.keys())}",
                )
            ]

        return self._validate_section(
            section_name, section_config, self.schemas[section_name]
        )

    def _validate_section(
        self,
        section_name: str,
        section_config: dict[str, Any],
        section_schema: dict[str, FieldSchema],
    ) -> list[ValidationError]:
        """Validate a configuration section against its schema.

        Args:
            section_name: Name of the section
            section_config: Section configuration dictionary
            section_schema: Schema for the section

        Returns:
            List of validation errors
        """
        errors: list[ValidationError] = []

        for field_name, field_schema in section_schema.items():
            path = f"{section_name}.{field_name}"
            value = section_config.get(field_name)

            # Check required fields
            if field_schema.required and value is None:
                errors.append(
                    ValidationError(
                        path=path,
                        message="Required field is missing",
                        suggestion=f"Provide a value for {path}",
                    )
                )
                continue

            # Skip validation if field is not present and not required
            if value is None:
                continue

            # Validate the field value
            field_errors = self._validate_field(path, value, field_schema)
            errors.extend(field_errors)

        return errors

    def _validate_field(
        self, path: str, value: Any, schema: FieldSchema
    ) -> list[ValidationError]:
        """Validate a single field value against its schema.

        Args:
            path: Dot-separated path to the field
            value: Field value to validate
            schema: Field schema

        Returns:
            List of validation errors
        """
        errors: list[ValidationError] = []

        # Type validation
        type_error = self._validate_type(value, schema.field_type)
        if type_error:
            errors.append(
                ValidationError(
                    path=path,
                    message=type_error,
                    value=value,
                    suggestion=f"Expected type: {schema.field_type.value}",
                )
            )
            return errors  # Don't continue validation if type is wrong

        # Range validation for numeric fields
        if schema.field_type in (FieldType.INTEGER, FieldType.FLOAT):
            if schema.min_value is not None and value < schema.min_value:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Value below minimum ({schema.min_value})",
                        value=value,
                        suggestion=f"Use a value >= {schema.min_value}",
                    )
                )
            if schema.max_value is not None and value > schema.max_value:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Value above maximum ({schema.max_value})",
                        value=value,
                        suggestion=f"Use a value <= {schema.max_value}",
                    )
                )

        # Length validation for string/list fields
        if schema.field_type in (FieldType.STRING, FieldType.LIST):
            length = len(value)
            if schema.min_length is not None and length < schema.min_length:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Length below minimum ({schema.min_length})",
                        value=value,
                        suggestion=f"Provide at least {schema.min_length} items/characters",
                    )
                )
            if schema.max_length is not None and length > schema.max_length:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Length above maximum ({schema.max_length})",
                        value=value,
                        suggestion=f"Reduce to at most {schema.max_length} items/characters",
                    )
                )

        # Pattern validation for string fields
        if schema.field_type == FieldType.STRING and schema.pattern:
            if not re.match(schema.pattern, value):
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Value does not match required pattern",
                        value=value,
                        suggestion=f"Must match pattern: {schema.pattern}",
                    )
                )

        # Allowed values validation
        if schema.allowed_values is not None:
            if value not in schema.allowed_values:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Value not in allowed list",
                        value=value,
                        suggestion=f"Allowed values: {', '.join(str(v) for v in schema.allowed_values)}",
                    )
                )

        # List item type validation
        if schema.field_type == FieldType.LIST and schema.item_type:
            for i, item in enumerate(value):
                item_type_error = self._validate_type(item, schema.item_type)
                if item_type_error:
                    errors.append(
                        ValidationError(
                            path=f"{path}[{i}]",
                            message=item_type_error,
                            value=item,
                            suggestion=f"List items must be of type: {schema.item_type.value}",
                        )
                    )

        # Custom validator
        if schema.custom_validator:
            custom_error = schema.custom_validator(value)
            if custom_error:
                errors.append(
                    ValidationError(
                        path=path,
                        message=custom_error,
                        value=value,
                    )
                )

        return errors

    def _validate_type(self, value: Any, expected_type: FieldType) -> str | None:
        """Validate value type against expected type.

        Args:
            value: Value to check
            expected_type: Expected field type

        Returns:
            Error message if type is wrong, None otherwise
        """
        if expected_type == FieldType.ANY:
            return None

        type_checks = {
            FieldType.STRING: lambda v: isinstance(v, str),
            FieldType.INTEGER: lambda v: isinstance(v, int) and not isinstance(v, bool),
            FieldType.FLOAT: lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            FieldType.BOOLEAN: lambda v: isinstance(v, bool),
            FieldType.LIST: lambda v: isinstance(v, list),
            FieldType.DICT: lambda v: isinstance(v, dict),
        }

        check = type_checks.get(expected_type)
        if check and not check(value):
            return f"Invalid type: expected {expected_type.value}, got {type(value).__name__}"

        return None

    def _validate_dependencies(
        self, config: dict[str, Any]
    ) -> list[ValidationError]:
        """Validate cross-field dependencies.

        Args:
            config: Full configuration dictionary

        Returns:
            List of dependency validation errors
        """
        errors: list[ValidationError] = []

        for rule in self.dependency_rules:
            source_value = self._get_nested_value(config, rule.source_path)
            target_value = self._get_nested_value(config, rule.target_path)

            # Check if condition applies
            if source_value is not None and rule.condition(source_value):
                # Check if requirement is satisfied
                if not rule.requirement(source_value, target_value):
                    errors.append(
                        ValidationError(
                            path=rule.target_path,
                            message=rule.message,
                            value=target_value,
                        )
                    )

        return errors

    def _get_nested_value(
        self, config: dict[str, Any], path: str
    ) -> Any:
        """Get nested value from config using dot notation path.

        Args:
            config: Configuration dictionary
            path: Dot-separated path

        Returns:
            Value at path, or None if not found
        """
        keys = path.split(".")
        current = config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate configuration and return list of error messages.

    This is a convenience function for quick validation.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of error messages (empty if valid)
    """
    validator = ConfigValidator()
    errors = validator.validate(config)
    return [str(error) for error in errors]


def validate_config_at_startup(config: dict[str, Any], raise_on_error: bool = False) -> list[ValidationError]:
    """Validate configuration at startup with optional exception raising.

    This function is designed to be called during application initialization.
    It validates the configuration and can optionally raise an exception if
    validation fails.

    Args:
        config: Configuration dictionary to validate
        raise_on_error: If True, raise ValueError on validation failure

    Returns:
        List of ValidationError objects

    Raises:
        ValueError: If raise_on_error is True and validation fails
    """
    validator = ConfigValidator()
    errors = validator.validate(config)

    if errors and raise_on_error:
        error_messages = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(
            f"Configuration validation failed with {len(errors)} error(s):\n{error_messages}"
        )

    return errors


def get_schema_documentation() -> str:
    """Generate human-readable documentation for all configuration options.

    Returns:
        Formatted documentation string
    """
    lines = [
        "Configuration Schema Documentation",
        "=" * 50,
        "",
    ]

    for section_name, section_schema in CONFIG_SCHEMAS.items():
        lines.append(f"\n[{section_name}]")
        lines.append("-" * 40)

        for field_name, field_schema in section_schema.items():
            required = " (required)" if field_schema.required else ""
            default = f" [default: {field_schema.default}]" if field_schema.default is not None else ""

            lines.append(f"  {field_name}: {field_schema.field_type.value}{required}{default}")
            if field_schema.description:
                lines.append(f"    {field_schema.description}")

            if field_schema.min_value is not None or field_schema.max_value is not None:
                range_str = f"    Range: [{field_schema.min_value or ''}, {field_schema.max_value or ''}]"
                lines.append(range_str)

            if field_schema.allowed_values:
                lines.append(f"    Allowed: {', '.join(str(v) for v in field_schema.allowed_values)}")

            lines.append("")

    return "\n".join(lines)
