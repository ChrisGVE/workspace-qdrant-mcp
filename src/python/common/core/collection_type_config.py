"""
Collection type configuration manager for workspace-qdrant-mcp.

This module defines type-specific behaviors, constraints, and configurations for
different collection types (SYSTEM, LIBRARY, PROJECT, GLOBAL). It provides a
centralized configuration system for managing deletion handling, metadata
requirements, validation rules, and performance optimization settings.

Key Features:
    - Deletion handling modes (dynamic vs cumulative)
    - Type-specific metadata field requirements
    - Validation rules and constraints
    - Performance optimization settings
    - Migration compatibility settings

Deletion Handling Modes:
    - DYNAMIC: Immediate deletion (for PROJECT, GLOBAL types)
    - CUMULATIVE: Mark deleted + batch cleanup (for SYSTEM, LIBRARY types)

Usage:
    ```python
    from collection_type_config import CollectionTypeConfig, get_type_config

    # Get configuration for a collection type
    config = get_type_config(CollectionType.SYSTEM)

    # Check deletion handling mode
    if config.deletion_mode == DeletionMode.CUMULATIVE:
        # Mark as deleted, batch cleanup later
        pass

    # Validate metadata
    is_valid, errors = config.validate_metadata(metadata_dict)

    # Get required metadata fields
    required_fields = config.required_metadata_fields
    ```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from loguru import logger

try:
    from .collection_types import CollectionType, CollectionInfo
except ImportError:
    # Fallback for development/testing
    logger.warning("Collection types not available, using fallback definitions")
    from enum import Enum

    class CollectionType(Enum):
        SYSTEM = "system"
        LIBRARY = "library"
        PROJECT = "project"
        GLOBAL = "global"
        UNKNOWN = "unknown"


class DeletionMode(Enum):
    """Deletion handling modes for collection types."""

    DYNAMIC = "dynamic"         # Immediate deletion
    CUMULATIVE = "cumulative"   # Mark deleted + batch cleanup


class MetadataFieldType(Enum):
    """Types of metadata fields for validation."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    DATETIME = "datetime"
    ENUM = "enum"


@dataclass
class MetadataFieldSpec:
    """Specification for a metadata field."""

    name: str
    field_type: MetadataFieldType
    required: bool = True
    default: Any = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None
    pattern: Optional[str] = None
    description: str = ""

    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate a value against this field specification.

        Args:
            value: The value to validate

        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        # Check None values for required fields
        if value is None:
            if self.required and self.default is None:
                return False, f"Field '{self.name}' is required"
            return True, None

        # Type validation
        if self.field_type == MetadataFieldType.STRING:
            if not isinstance(value, str):
                return False, f"Field '{self.name}' must be a string"
            if self.min_length and len(value) < self.min_length:
                return False, f"Field '{self.name}' must be at least {self.min_length} characters"
            if self.max_length and len(value) > self.max_length:
                return False, f"Field '{self.name}' must be at most {self.max_length} characters"
            if self.pattern:
                import re
                if not re.match(self.pattern, value):
                    return False, f"Field '{self.name}' does not match required pattern"

        elif self.field_type == MetadataFieldType.INTEGER:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Field '{self.name}' must be an integer"
            if self.min_value is not None and value < self.min_value:
                return False, f"Field '{self.name}' must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Field '{self.name}' must be at most {self.max_value}"

        elif self.field_type == MetadataFieldType.FLOAT:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Field '{self.name}' must be a number"
            if self.min_value is not None and value < self.min_value:
                return False, f"Field '{self.name}' must be at least {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"Field '{self.name}' must be at most {self.max_value}"

        elif self.field_type == MetadataFieldType.BOOLEAN:
            if not isinstance(value, bool):
                return False, f"Field '{self.name}' must be a boolean"

        elif self.field_type == MetadataFieldType.LIST:
            if not isinstance(value, list):
                return False, f"Field '{self.name}' must be a list"
            if self.min_length is not None and len(value) < self.min_length:
                return False, f"Field '{self.name}' must have at least {self.min_length} items"
            if self.max_length is not None and len(value) > self.max_length:
                return False, f"Field '{self.name}' must have at most {self.max_length} items"

        elif self.field_type == MetadataFieldType.DICT:
            if not isinstance(value, dict):
                return False, f"Field '{self.name}' must be a dictionary"

        # Allowed values validation
        if self.allowed_values and value not in self.allowed_values:
            return False, f"Field '{self.name}' must be one of: {self.allowed_values}"

        return True, None


@dataclass
class PerformanceSettings:
    """Performance optimization settings for a collection type."""

    batch_size: int = 100
    max_concurrent_operations: int = 5
    priority_weight: int = 1  # 1=lowest, 5=highest
    cache_ttl_seconds: int = 300
    enable_batch_processing: bool = True
    enable_caching: bool = True
    max_retries: int = 3
    retry_delay_seconds: int = 1


@dataclass
class MigrationSettings:
    """Migration compatibility settings for a collection type."""

    supports_legacy_format: bool = False
    legacy_collection_patterns: List[str] = field(default_factory=list)
    migration_batch_size: int = 50
    auto_detect_legacy: bool = False
    preserve_legacy_metadata: bool = True


@dataclass
class CollectionTypeConfig:
    """
    Configuration for a specific collection type.

    This class defines all type-specific behaviors, constraints, and settings
    for a collection type. It provides a comprehensive configuration system
    for managing collections with different requirements and characteristics.

    Attributes:
        collection_type: The collection type this config applies to
        deletion_mode: How deletions are handled (dynamic or cumulative)
        required_metadata_fields: List of required metadata field specifications
        optional_metadata_fields: List of optional metadata field specifications
        performance_settings: Performance optimization settings
        migration_settings: Migration compatibility settings
        validation_rules: Custom validation functions
        description: Human-readable description of the type
    """

    collection_type: CollectionType
    deletion_mode: DeletionMode
    required_metadata_fields: List[MetadataFieldSpec] = field(default_factory=list)
    optional_metadata_fields: List[MetadataFieldSpec] = field(default_factory=list)
    performance_settings: PerformanceSettings = field(default_factory=PerformanceSettings)
    migration_settings: MigrationSettings = field(default_factory=MigrationSettings)
    validation_rules: List[Callable[[Dict[str, Any]], Tuple[bool, Optional[str]]]] = field(default_factory=list)
    description: str = ""

    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate metadata against this configuration.

        Args:
            metadata: The metadata dictionary to validate

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []

        # Validate required fields
        for field_spec in self.required_metadata_fields:
            value = metadata.get(field_spec.name)
            is_valid, error_msg = field_spec.validate(value)
            if not is_valid:
                errors.append(error_msg)

        # Validate optional fields if present
        for field_spec in self.optional_metadata_fields:
            if field_spec.name in metadata:
                value = metadata[field_spec.name]
                is_valid, error_msg = field_spec.validate(value)
                if not is_valid:
                    errors.append(error_msg)

        # Run custom validation rules
        for validation_rule in self.validation_rules:
            is_valid, error_msg = validation_rule(metadata)
            if not is_valid and error_msg:
                errors.append(error_msg)

        return len(errors) == 0, errors

    def get_required_field_names(self) -> Set[str]:
        """Get the names of all required metadata fields."""
        return {field.name for field in self.required_metadata_fields}

    def get_all_field_names(self) -> Set[str]:
        """Get the names of all metadata fields (required and optional)."""
        return {field.name for field in self.required_metadata_fields + self.optional_metadata_fields}

    def get_field_spec(self, field_name: str) -> Optional[MetadataFieldSpec]:
        """Get the specification for a specific field."""
        for field_spec in self.required_metadata_fields + self.optional_metadata_fields:
            if field_spec.name == field_name:
                return field_spec
        return None

    def get_default_metadata(self) -> Dict[str, Any]:
        """Get a dictionary of default values for all fields with defaults."""
        defaults = {}
        for field_spec in self.required_metadata_fields + self.optional_metadata_fields:
            if field_spec.default is not None:
                defaults[field_spec.name] = field_spec.default
        return defaults


# Type-specific configurations

def _create_system_config() -> CollectionTypeConfig:
    """Create configuration for SYSTEM collection type."""
    return CollectionTypeConfig(
        collection_type=CollectionType.SYSTEM,
        deletion_mode=DeletionMode.CUMULATIVE,
        required_metadata_fields=[
            MetadataFieldSpec(
                name="collection_name",
                field_type=MetadataFieldType.STRING,
                required=True,
                min_length=3,
                max_length=128,
                pattern=r"^__[a-zA-Z0-9_-]+$",
                description="System collection name with __ prefix"
            ),
            MetadataFieldSpec(
                name="created_at",
                field_type=MetadataFieldType.STRING,
                required=True,
                description="ISO timestamp of creation"
            ),
            MetadataFieldSpec(
                name="collection_category",
                field_type=MetadataFieldType.STRING,
                required=True,
                allowed_values={"system"},
                description="Collection category (must be 'system')"
            ),
        ],
        optional_metadata_fields=[
            MetadataFieldSpec(
                name="updated_at",
                field_type=MetadataFieldType.STRING,
                required=False,
                description="ISO timestamp of last update"
            ),
            MetadataFieldSpec(
                name="description",
                field_type=MetadataFieldType.STRING,
                required=False,
                max_length=1024,
                description="Human-readable description"
            ),
            MetadataFieldSpec(
                name="cli_writable",
                field_type=MetadataFieldType.BOOLEAN,
                required=False,
                default=True,
                description="Whether CLI can write to this collection"
            ),
        ],
        performance_settings=PerformanceSettings(
            batch_size=50,
            max_concurrent_operations=3,
            priority_weight=4,
            cache_ttl_seconds=600,
        ),
        migration_settings=MigrationSettings(
            supports_legacy_format=True,
            legacy_collection_patterns=[r"^__[a-zA-Z0-9_-]+$"],
            auto_detect_legacy=True,
        ),
        description="System collections (__ prefix) - CLI-writable, LLM-readable, not globally searchable"
    )


def _create_library_config() -> CollectionTypeConfig:
    """Create configuration for LIBRARY collection type."""
    return CollectionTypeConfig(
        collection_type=CollectionType.LIBRARY,
        deletion_mode=DeletionMode.CUMULATIVE,
        required_metadata_fields=[
            MetadataFieldSpec(
                name="collection_name",
                field_type=MetadataFieldType.STRING,
                required=True,
                min_length=2,
                max_length=128,
                pattern=r"^_[a-zA-Z0-9_-]+$",
                description="Library collection name with _ prefix"
            ),
            MetadataFieldSpec(
                name="created_at",
                field_type=MetadataFieldType.STRING,
                required=True,
                description="ISO timestamp of creation"
            ),
            MetadataFieldSpec(
                name="collection_category",
                field_type=MetadataFieldType.STRING,
                required=True,
                allowed_values={"library"},
                description="Collection category (must be 'library')"
            ),
            MetadataFieldSpec(
                name="language",
                field_type=MetadataFieldType.STRING,
                required=True,
                description="Primary programming language"
            ),
        ],
        optional_metadata_fields=[
            MetadataFieldSpec(
                name="updated_at",
                field_type=MetadataFieldType.STRING,
                required=False,
                description="ISO timestamp of last update"
            ),
            MetadataFieldSpec(
                name="description",
                field_type=MetadataFieldType.STRING,
                required=False,
                max_length=1024,
                description="Human-readable description"
            ),
            MetadataFieldSpec(
                name="symbols",
                field_type=MetadataFieldType.LIST,
                required=False,
                description="List of exported symbols"
            ),
            MetadataFieldSpec(
                name="dependencies",
                field_type=MetadataFieldType.LIST,
                required=False,
                description="List of library dependencies"
            ),
            MetadataFieldSpec(
                name="version",
                field_type=MetadataFieldType.STRING,
                required=False,
                description="Library version"
            ),
            MetadataFieldSpec(
                name="mcp_readonly",
                field_type=MetadataFieldType.BOOLEAN,
                required=False,
                default=True,
                description="Whether MCP access is read-only"
            ),
        ],
        performance_settings=PerformanceSettings(
            batch_size=100,
            max_concurrent_operations=5,
            priority_weight=3,
            cache_ttl_seconds=900,
            enable_caching=True,
        ),
        migration_settings=MigrationSettings(
            supports_legacy_format=True,
            legacy_collection_patterns=[r"^_[a-zA-Z0-9_-]+$"],
            auto_detect_legacy=True,
            preserve_legacy_metadata=True,
        ),
        description="Library collections (_ prefix) - CLI-managed, MCP-readonly, globally searchable"
    )


def _create_project_config() -> CollectionTypeConfig:
    """Create configuration for PROJECT collection type."""
    return CollectionTypeConfig(
        collection_type=CollectionType.PROJECT,
        deletion_mode=DeletionMode.DYNAMIC,
        required_metadata_fields=[
            MetadataFieldSpec(
                name="project_name",
                field_type=MetadataFieldType.STRING,
                required=True,
                min_length=1,
                max_length=128,
                description="Project name"
            ),
            MetadataFieldSpec(
                name="project_id",
                field_type=MetadataFieldType.STRING,
                required=True,
                min_length=12,
                max_length=12,
                description="12-character project identifier"
            ),
            MetadataFieldSpec(
                name="collection_type",
                field_type=MetadataFieldType.STRING,
                required=True,
                description="Collection type (docs, notes, memory, etc.)"
            ),
            MetadataFieldSpec(
                name="created_at",
                field_type=MetadataFieldType.STRING,
                required=True,
                description="ISO timestamp of creation"
            ),
        ],
        optional_metadata_fields=[
            MetadataFieldSpec(
                name="updated_at",
                field_type=MetadataFieldType.STRING,
                required=False,
                description="ISO timestamp of last update"
            ),
            MetadataFieldSpec(
                name="description",
                field_type=MetadataFieldType.STRING,
                required=False,
                max_length=1024,
                description="Human-readable description"
            ),
            MetadataFieldSpec(
                name="tenant_namespace",
                field_type=MetadataFieldType.STRING,
                required=False,
                description="Tenant namespace for isolation"
            ),
            MetadataFieldSpec(
                name="tags",
                field_type=MetadataFieldType.LIST,
                required=False,
                description="Organizational tags"
            ),
            MetadataFieldSpec(
                name="priority",
                field_type=MetadataFieldType.INTEGER,
                required=False,
                min_value=1,
                max_value=5,
                default=3,
                description="Priority level (1=lowest, 5=highest)"
            ),
        ],
        performance_settings=PerformanceSettings(
            batch_size=150,
            max_concurrent_operations=10,
            priority_weight=2,
            cache_ttl_seconds=300,
            enable_batch_processing=True,
        ),
        migration_settings=MigrationSettings(
            supports_legacy_format=True,
            legacy_collection_patterns=[r"^[a-zA-Z0-9_-]+-[a-zA-Z0-9_-]+$"],
            auto_detect_legacy=False,
            migration_batch_size=100,
        ),
        description="Project collections ({project}-{suffix}) - user-created, project-scoped"
    )


def _create_global_config() -> CollectionTypeConfig:
    """Create configuration for GLOBAL collection type."""
    return CollectionTypeConfig(
        collection_type=CollectionType.GLOBAL,
        deletion_mode=DeletionMode.DYNAMIC,
        required_metadata_fields=[
            MetadataFieldSpec(
                name="collection_name",
                field_type=MetadataFieldType.STRING,
                required=True,
                min_length=1,
                max_length=128,
                allowed_values={
                    "algorithms", "codebase", "context", "documents",
                    "knowledge", "memory", "projects", "workspace"
                },
                description="Global collection name"
            ),
            MetadataFieldSpec(
                name="created_at",
                field_type=MetadataFieldType.STRING,
                required=True,
                description="ISO timestamp of creation"
            ),
        ],
        optional_metadata_fields=[
            MetadataFieldSpec(
                name="updated_at",
                field_type=MetadataFieldType.STRING,
                required=False,
                description="ISO timestamp of last update"
            ),
            MetadataFieldSpec(
                name="description",
                field_type=MetadataFieldType.STRING,
                required=False,
                max_length=1024,
                description="Human-readable description"
            ),
            MetadataFieldSpec(
                name="workspace_scope",
                field_type=MetadataFieldType.STRING,
                required=False,
                allowed_values={"global"},
                default="global",
                description="Workspace scope (must be 'global')"
            ),
        ],
        performance_settings=PerformanceSettings(
            batch_size=200,
            max_concurrent_operations=8,
            priority_weight=5,
            cache_ttl_seconds=1800,
            enable_caching=True,
        ),
        migration_settings=MigrationSettings(
            supports_legacy_format=False,
            auto_detect_legacy=False,
        ),
        description="Global collections - system-wide, always available"
    )


# Configuration registry
_TYPE_CONFIGS: Dict[CollectionType, CollectionTypeConfig] = {
    CollectionType.SYSTEM: _create_system_config(),
    CollectionType.LIBRARY: _create_library_config(),
    CollectionType.PROJECT: _create_project_config(),
    CollectionType.GLOBAL: _create_global_config(),
}


def get_type_config(collection_type: CollectionType) -> CollectionTypeConfig:
    """
    Get the configuration for a specific collection type.

    Args:
        collection_type: The collection type to get configuration for

    Returns:
        CollectionTypeConfig: The configuration for the specified type

    Raises:
        ValueError: If the collection type is not supported
    """
    if collection_type not in _TYPE_CONFIGS:
        raise ValueError(f"No configuration available for collection type: {collection_type}")
    return _TYPE_CONFIGS[collection_type]


def get_all_type_configs() -> Dict[CollectionType, CollectionTypeConfig]:
    """
    Get all collection type configurations.

    Returns:
        Dict[CollectionType, CollectionTypeConfig]: Dictionary mapping types to configs
    """
    return _TYPE_CONFIGS.copy()


def validate_metadata_for_type(
    collection_type: CollectionType,
    metadata: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    Validate metadata against a specific collection type's configuration.

    Args:
        collection_type: The collection type to validate against
        metadata: The metadata dictionary to validate

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_errors)
    """
    config = get_type_config(collection_type)
    return config.validate_metadata(metadata)


def get_deletion_mode(collection_type: CollectionType) -> DeletionMode:
    """
    Get the deletion handling mode for a collection type.

    Args:
        collection_type: The collection type to query

    Returns:
        DeletionMode: The deletion mode for this type
    """
    config = get_type_config(collection_type)
    return config.deletion_mode


def should_use_cumulative_deletion(collection_type: CollectionType) -> bool:
    """
    Check if a collection type should use cumulative deletion (mark + batch cleanup).

    Args:
        collection_type: The collection type to check

    Returns:
        bool: True if cumulative deletion should be used
    """
    return get_deletion_mode(collection_type) == DeletionMode.CUMULATIVE


def should_use_dynamic_deletion(collection_type: CollectionType) -> bool:
    """
    Check if a collection type should use dynamic deletion (immediate).

    Args:
        collection_type: The collection type to check

    Returns:
        bool: True if dynamic deletion should be used
    """
    return get_deletion_mode(collection_type) == DeletionMode.DYNAMIC


# Export all public classes and functions
__all__ = [
    # Enums
    'DeletionMode',
    'MetadataFieldType',

    # Data classes
    'MetadataFieldSpec',
    'PerformanceSettings',
    'MigrationSettings',
    'CollectionTypeConfig',

    # Functions
    'get_type_config',
    'get_all_type_configs',
    'validate_metadata_for_type',
    'get_deletion_mode',
    'should_use_cumulative_deletion',
    'should_use_dynamic_deletion',
]
