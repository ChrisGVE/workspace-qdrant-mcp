"""
Metadata validation system for multi-tenant workspace collections.

This module provides comprehensive validation rules and constraint checking for
the multi-tenant metadata schema. It ensures data integrity, validates naming
patterns, enforces access control rules, and provides detailed error reporting.

Key Features:
    - Comprehensive field validation with detailed error messages
    - Reserved naming pattern validation for system and library collections
    - Project ID format and uniqueness validation
    - Access control rule enforcement
    - Tenant namespace format validation
    - Backward compatibility validation for migrated collections

Validation Categories:
    - **Format Validation**: Field formats, lengths, and patterns
    - **Business Rule Validation**: Access control, naming conventions
    - **Consistency Validation**: Cross-field dependencies and constraints
    - **Migration Validation**: Backward compatibility and migration rules

Example:
    ```python
    from metadata_validator import MetadataValidator, ValidationResult
    from metadata_schema import MultiTenantMetadataSchema

    validator = MetadataValidator()
    metadata = MultiTenantMetadataSchema.create_for_project("myproject", "docs")

    # Validate metadata
    result = validator.validate_metadata(metadata)
    if result.is_valid:
        print("Metadata is valid")
    else:
        for error in result.errors:
            print(f"Error: {error}")
    ```
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum

from loguru import logger

try:
    from .metadata_schema import (
        MultiTenantMetadataSchema,
        CollectionCategory,
        WorkspaceScope,
        AccessLevel,
        METADATA_SCHEMA_VERSION,
        MAX_PROJECT_NAME_LENGTH,
        MAX_COLLECTION_TYPE_LENGTH,
        MAX_TENANT_NAMESPACE_LENGTH,
        MAX_CREATED_BY_LENGTH,
        MAX_ACCESS_LEVEL_LENGTH,
        MAX_BRANCH_LENGTH
    )
except ImportError:
    logger.error("Cannot import metadata_schema module")
    raise


class ValidationSeverity(Enum):
    """Severity levels for validation errors."""

    ERROR = "error"           # Critical errors that prevent operation
    WARNING = "warning"       # Issues that should be addressed but not critical
    INFO = "info"            # Informational messages
    DEPRECATED = "deprecated" # Usage of deprecated features


@dataclass
class ValidationError:
    """Individual validation error with context and suggestions."""

    field_name: str
    severity: ValidationSeverity
    message: str
    code: str
    value: Any = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of metadata validation with errors and warnings."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    info_messages: List[ValidationError] = field(default_factory=list)

    def add_error(self, field_name: str, message: str, code: str,
                  value: Any = None, suggestion: str = None):
        """Add a validation error."""
        self.errors.append(ValidationError(
            field_name=field_name,
            severity=ValidationSeverity.ERROR,
            message=message,
            code=code,
            value=value,
            suggestion=suggestion
        ))
        self.is_valid = False

    def add_warning(self, field_name: str, message: str, code: str,
                   value: Any = None, suggestion: str = None):
        """Add a validation warning."""
        self.warnings.append(ValidationError(
            field_name=field_name,
            severity=ValidationSeverity.WARNING,
            message=message,
            code=code,
            value=value,
            suggestion=suggestion
        ))

    def add_info(self, field_name: str, message: str, code: str,
                value: Any = None, suggestion: str = None):
        """Add an informational message."""
        self.info_messages.append(ValidationError(
            field_name=field_name,
            severity=ValidationSeverity.INFO,
            message=message,
            code=code,
            value=value,
            suggestion=suggestion
        ))

    def get_all_issues(self) -> List[ValidationError]:
        """Get all validation issues (errors + warnings + info)."""
        return self.errors + self.warnings + self.info_messages

    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if there are any validation warnings."""
        return len(self.warnings) > 0


class MetadataValidator:
    """
    Comprehensive validator for multi-tenant metadata schemas.

    This class provides complete validation of metadata schemas including
    format validation, business rule enforcement, and constraint checking.
    It supports both strict validation for new collections and lenient
    validation for migrated collections.
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize the metadata validator.

        Args:
            strict_mode: If True, enforces strict validation rules.
                        If False, provides more lenient validation for migrations.
        """
        self.strict_mode = strict_mode

        # Compile regex patterns for performance
        self._project_id_pattern = re.compile(r'^[a-f0-9]{12}$')
        self._project_name_pattern = re.compile(r'^[a-zA-Z0-9_-]+$')
        self._collection_type_pattern = re.compile(r'^[a-zA-Z0-9_]+$')
        self._system_collection_pattern = re.compile(r'^__[a-zA-Z0-9_-]+$')
        self._library_collection_pattern = re.compile(r'^_[a-zA-Z0-9_-]+$')
        self._project_collection_pattern = re.compile(r'^[a-zA-Z0-9_-]+-[a-zA-Z0-9_-]+$')
        self._tenant_namespace_pattern = re.compile(r'^[a-zA-Z0-9_-]+\.[a-zA-Z0-9_]+$')

        # Valid collection types
        self._valid_collection_types = {
            'docs', 'notes', 'scratchbook', 'knowledge', 'context', 'memory',
            'memory_collection', 'code_collection', 'algorithms', 'codebase',
            'documents', 'projects', 'workspace'
        }

        # Valid global collections
        self._valid_global_collections = {
            'algorithms', 'codebase', 'context', 'documents',
            'knowledge', 'memory', 'projects', 'workspace'
        }

        # Valid naming patterns
        self._valid_naming_patterns = {
            'metadata_based', 'system_prefix', 'library_prefix',
            'project_pattern', 'global_collection'
        }

        # Valid migration sources
        self._valid_migration_sources = {
            'metadata_based', 'suffix_based', 'manual', 'auto_create', 'cli', 'migration'
        }

        # Valid created_by values
        self._valid_created_by = {
            'system', 'user', 'cli', 'migration', 'admin'
        }

    def validate_metadata(self, metadata: MultiTenantMetadataSchema) -> ValidationResult:
        """
        Perform comprehensive validation of metadata schema.

        Args:
            metadata: The metadata schema to validate

        Returns:
            ValidationResult with errors, warnings, and info messages
        """
        result = ValidationResult(is_valid=True)

        logger.debug(f"Validating metadata for project '{metadata.project_name}', "
                    f"collection '{metadata.collection_type}'")

        # Core field validation
        self._validate_core_fields(metadata, result)

        # Collection classification validation
        self._validate_collection_classification(metadata, result)

        # Access control validation
        self._validate_access_control(metadata, result)

        # Naming pattern validation
        self._validate_naming_patterns(metadata, result)

        # Consistency validation
        self._validate_consistency(metadata, result)

        # Migration and compatibility validation
        self._validate_migration_compatibility(metadata, result)

        # Format and constraint validation
        self._validate_formats_and_constraints(metadata, result)

        # Code analysis metadata validation
        self._validate_code_analysis_fields(metadata, result)

        logger.debug(f"Validation completed: valid={result.is_valid}, "
                    f"errors={len(result.errors)}, warnings={len(result.warnings)}")

        return result

    def _validate_core_fields(self, metadata: MultiTenantMetadataSchema, result: ValidationResult):
        """Validate core tenant isolation fields."""

        # Project ID validation
        if not metadata.project_id:
            result.add_error("project_id", "Project ID is required", "MISSING_PROJECT_ID")
        elif not self._project_id_pattern.match(metadata.project_id):
            result.add_error(
                "project_id",
                "Project ID must be exactly 12 hexadecimal characters",
                "INVALID_PROJECT_ID_FORMAT",
                metadata.project_id,
                "Use a 12-character hex string (e.g., 'a1b2c3d4e5f6')"
            )

        # Project name validation
        if not metadata.project_name:
            result.add_error("project_name", "Project name is required", "MISSING_PROJECT_NAME")
        elif len(metadata.project_name) > MAX_PROJECT_NAME_LENGTH:
            result.add_error(
                "project_name",
                f"Project name exceeds maximum length of {MAX_PROJECT_NAME_LENGTH}",
                "PROJECT_NAME_TOO_LONG",
                metadata.project_name
            )
        elif not self._project_name_pattern.match(metadata.project_name):
            result.add_error(
                "project_name",
                "Project name contains invalid characters (use letters, numbers, underscore, hyphen)",
                "INVALID_PROJECT_NAME_CHARS",
                metadata.project_name
            )

        # Branch validation
        if not metadata.branch:
            result.add_error("branch", "Branch is required", "MISSING_BRANCH")
        elif not metadata.branch.strip():
            result.add_error(
                "branch",
                "Branch cannot be empty or whitespace only",
                "EMPTY_BRANCH",
                metadata.branch
            )
        elif len(metadata.branch) > MAX_BRANCH_LENGTH:
            result.add_error(
                "branch",
                f"Branch exceeds maximum length of {MAX_BRANCH_LENGTH}",
                "BRANCH_TOO_LONG",
                metadata.branch
            )

        # Tenant namespace validation
        if not metadata.tenant_namespace:
            result.add_error("tenant_namespace", "Tenant namespace is required", "MISSING_TENANT_NAMESPACE")
        elif not self._tenant_namespace_pattern.match(metadata.tenant_namespace):
            result.add_error(
                "tenant_namespace",
                "Tenant namespace must follow format 'project.collection_type'",
                "INVALID_TENANT_NAMESPACE_FORMAT",
                metadata.tenant_namespace,
                f"Use format '{metadata.project_name}.{metadata.collection_type}'"
            )
        elif metadata.tenant_namespace != f"{metadata.project_name}.{metadata.collection_type}":
            result.add_error(
                "tenant_namespace",
                "Tenant namespace must match project_name.collection_type",
                "INCONSISTENT_TENANT_NAMESPACE",
                metadata.tenant_namespace,
                f"Should be '{metadata.project_name}.{metadata.collection_type}'"
            )

    def _validate_collection_classification(self, metadata: MultiTenantMetadataSchema, result: ValidationResult):
        """Validate collection type and classification fields."""

        # Collection type validation
        if not metadata.collection_type:
            result.add_error("collection_type", "Collection type is required", "MISSING_COLLECTION_TYPE")
        elif len(metadata.collection_type) > MAX_COLLECTION_TYPE_LENGTH:
            result.add_error(
                "collection_type",
                f"Collection type exceeds maximum length of {MAX_COLLECTION_TYPE_LENGTH}",
                "COLLECTION_TYPE_TOO_LONG",
                metadata.collection_type
            )
        elif not self._collection_type_pattern.match(metadata.collection_type):
            result.add_error(
                "collection_type",
                "Collection type contains invalid characters (use letters, numbers, underscore)",
                "INVALID_COLLECTION_TYPE_CHARS",
                metadata.collection_type
            )
        elif metadata.collection_type not in self._valid_collection_types:
            result.add_warning(
                "collection_type",
                f"Collection type '{metadata.collection_type}' is not a standard type",
                "NON_STANDARD_COLLECTION_TYPE",
                metadata.collection_type,
                f"Consider using one of: {', '.join(sorted(self._valid_collection_types))}"
            )

        # Collection category validation
        if not isinstance(metadata.collection_category, CollectionCategory):
            result.add_error(
                "collection_category",
                "Collection category must be a CollectionCategory enum value",
                "INVALID_COLLECTION_CATEGORY_TYPE",
                metadata.collection_category
            )

        # Workspace scope validation
        if not isinstance(metadata.workspace_scope, WorkspaceScope):
            result.add_error(
                "workspace_scope",
                "Workspace scope must be a WorkspaceScope enum value",
                "INVALID_WORKSPACE_SCOPE_TYPE",
                metadata.workspace_scope
            )

    def _validate_access_control(self, metadata: MultiTenantMetadataSchema, result: ValidationResult):
        """Validate access control and permissions fields."""

        # Access level validation
        if not isinstance(metadata.access_level, AccessLevel):
            result.add_error(
                "access_level",
                "Access level must be an AccessLevel enum value",
                "INVALID_ACCESS_LEVEL_TYPE",
                metadata.access_level
            )

        # Created by validation
        if not metadata.created_by:
            result.add_error("created_by", "Created by field is required", "MISSING_CREATED_BY")
        elif len(metadata.created_by) > MAX_CREATED_BY_LENGTH:
            result.add_error(
                "created_by",
                f"Created by exceeds maximum length of {MAX_CREATED_BY_LENGTH}",
                "CREATED_BY_TOO_LONG",
                metadata.created_by
            )
        elif metadata.created_by not in self._valid_created_by:
            result.add_warning(
                "created_by",
                f"Created by '{metadata.created_by}' is not a standard value",
                "NON_STANDARD_CREATED_BY",
                metadata.created_by,
                f"Consider using one of: {', '.join(sorted(self._valid_created_by))}"
            )

        # Boolean field validation
        if not isinstance(metadata.mcp_readonly, bool):
            result.add_error(
                "mcp_readonly",
                "MCP readonly must be a boolean value",
                "INVALID_MCP_READONLY_TYPE",
                metadata.mcp_readonly
            )

        if not isinstance(metadata.cli_writable, bool):
            result.add_error(
                "cli_writable",
                "CLI writable must be a boolean value",
                "INVALID_CLI_WRITABLE_TYPE",
                metadata.cli_writable
            )

        if not isinstance(metadata.is_reserved_name, bool):
            result.add_error(
                "is_reserved_name",
                "Is reserved name must be a boolean value",
                "INVALID_IS_RESERVED_NAME_TYPE",
                metadata.is_reserved_name
            )

    def _validate_naming_patterns(self, metadata: MultiTenantMetadataSchema, result: ValidationResult):
        """Validate naming patterns and reserved name constraints."""

        # Naming pattern validation
        if metadata.naming_pattern not in self._valid_naming_patterns:
            result.add_error(
                "naming_pattern",
                f"Invalid naming pattern '{metadata.naming_pattern}'",
                "INVALID_NAMING_PATTERN",
                metadata.naming_pattern,
                f"Must be one of: {', '.join(sorted(self._valid_naming_patterns))}"
            )

        # Validate consistency between naming pattern and collection category
        expected_patterns = {
            CollectionCategory.SYSTEM: "system_prefix",
            CollectionCategory.LIBRARY: "library_prefix",
            CollectionCategory.PROJECT: "project_pattern",
            CollectionCategory.GLOBAL: "global_collection"
        }

        expected_pattern = expected_patterns.get(metadata.collection_category)
        if expected_pattern and metadata.naming_pattern != expected_pattern:
            result.add_warning(
                "naming_pattern",
                f"Naming pattern '{metadata.naming_pattern}' doesn't match collection category "
                f"'{metadata.collection_category.value}' (expected '{expected_pattern}')",
                "INCONSISTENT_NAMING_PATTERN",
                metadata.naming_pattern
            )

        # Validate reserved name consistency
        should_be_reserved = metadata.collection_category in [CollectionCategory.SYSTEM, CollectionCategory.LIBRARY]
        if metadata.is_reserved_name != should_be_reserved:
            result.add_error(
                "is_reserved_name",
                f"Reserved name flag ({metadata.is_reserved_name}) doesn't match collection category "
                f"({metadata.collection_category.value})",
                "INCONSISTENT_RESERVED_NAME_FLAG",
                metadata.is_reserved_name
            )

        # Validate original name pattern for reserved collections
        if metadata.is_reserved_name and not metadata.original_name_pattern:
            result.add_warning(
                "original_name_pattern",
                "Reserved collections should have original_name_pattern set",
                "MISSING_ORIGINAL_NAME_PATTERN",
                suggestion="Set original_name_pattern to the actual collection name"
            )

    def _validate_consistency(self, metadata: MultiTenantMetadataSchema, result: ValidationResult):
        """Validate consistency between related fields."""

        # System collection consistency
        if metadata.collection_category == CollectionCategory.SYSTEM:
            if metadata.workspace_scope != WorkspaceScope.GLOBAL:
                result.add_warning(
                    "workspace_scope",
                    "System collections typically have global workspace scope",
                    "SYSTEM_COLLECTION_SCOPE_MISMATCH",
                    metadata.workspace_scope.value
                )

            if metadata.access_level != AccessLevel.PRIVATE:
                result.add_warning(
                    "access_level",
                    "System collections typically have private access level",
                    "SYSTEM_COLLECTION_ACCESS_MISMATCH",
                    metadata.access_level.value
                )

        # Library collection consistency
        if metadata.collection_category == CollectionCategory.LIBRARY:
            if not metadata.mcp_readonly:
                result.add_error(
                    "mcp_readonly",
                    "Library collections must be MCP readonly",
                    "LIBRARY_COLLECTION_MCP_READONLY_REQUIRED",
                    metadata.mcp_readonly
                )

            if metadata.workspace_scope != WorkspaceScope.LIBRARY:
                result.add_warning(
                    "workspace_scope",
                    "Library collections should have library workspace scope",
                    "LIBRARY_COLLECTION_SCOPE_MISMATCH",
                    metadata.workspace_scope.value
                )

        # Project collection consistency
        if metadata.collection_category == CollectionCategory.PROJECT:
            if metadata.workspace_scope not in [WorkspaceScope.PROJECT, WorkspaceScope.SHARED]:
                result.add_warning(
                    "workspace_scope",
                    "Project collections should have project or shared workspace scope",
                    "PROJECT_COLLECTION_SCOPE_MISMATCH",
                    metadata.workspace_scope.value
                )

        # Global collection consistency
        if metadata.collection_category == CollectionCategory.GLOBAL:
            if metadata.collection_type not in self._valid_global_collections:
                result.add_warning(
                    "collection_type",
                    f"Collection type '{metadata.collection_type}' is not a standard global collection",
                    "NON_STANDARD_GLOBAL_COLLECTION",
                    metadata.collection_type,
                    f"Standard global collections: {', '.join(sorted(self._valid_global_collections))}"
                )

            if metadata.workspace_scope != WorkspaceScope.GLOBAL:
                result.add_warning(
                    "workspace_scope",
                    "Global collections should have global workspace scope",
                    "GLOBAL_COLLECTION_SCOPE_MISMATCH",
                    metadata.workspace_scope.value
                )

    def _validate_migration_compatibility(self, metadata: MultiTenantMetadataSchema, result: ValidationResult):
        """Validate migration and compatibility fields."""

        # Migration source validation
        if metadata.migration_source not in self._valid_migration_sources:
            result.add_warning(
                "migration_source",
                f"Migration source '{metadata.migration_source}' is not a standard value",
                "NON_STANDARD_MIGRATION_SOURCE",
                metadata.migration_source,
                f"Standard values: {', '.join(sorted(self._valid_migration_sources))}"
            )

        # Compatibility version validation
        if not metadata.compatibility_version:
            result.add_error(
                "compatibility_version",
                "Compatibility version is required",
                "MISSING_COMPATIBILITY_VERSION"
            )
        elif metadata.compatibility_version != METADATA_SCHEMA_VERSION:
            result.add_info(
                "compatibility_version",
                f"Compatibility version '{metadata.compatibility_version}' differs from current version "
                f"'{METADATA_SCHEMA_VERSION}'",
                "VERSION_MISMATCH",
                metadata.compatibility_version
            )

        # Legacy collection name validation for migrated collections
        if metadata.migration_source in ['suffix_based', 'manual'] and not metadata.legacy_collection_name:
            result.add_warning(
                "legacy_collection_name",
                "Migrated collections should have legacy_collection_name set",
                "MISSING_LEGACY_COLLECTION_NAME",
                suggestion="Set legacy_collection_name to preserve migration history"
            )

    def _validate_code_analysis_fields(self, metadata: MultiTenantMetadataSchema, result: ValidationResult):
        """Validate code analysis metadata fields (symbols, imports, exports)."""

        # Validate symbols_defined
        if not isinstance(metadata.symbols_defined, list):
            result.add_error(
                "symbols_defined",
                "symbols_defined must be a list",
                "INVALID_SYMBOLS_DEFINED_TYPE",
                type(metadata.symbols_defined).__name__
            )
        elif any(not isinstance(item, str) for item in metadata.symbols_defined):
            result.add_error(
                "symbols_defined",
                "All items in symbols_defined must be strings",
                "INVALID_SYMBOLS_DEFINED_ITEM_TYPE",
                metadata.symbols_defined
            )

        # Validate symbols_used
        if not isinstance(metadata.symbols_used, list):
            result.add_error(
                "symbols_used",
                "symbols_used must be a list",
                "INVALID_SYMBOLS_USED_TYPE",
                type(metadata.symbols_used).__name__
            )
        elif any(not isinstance(item, str) for item in metadata.symbols_used):
            result.add_error(
                "symbols_used",
                "All items in symbols_used must be strings",
                "INVALID_SYMBOLS_USED_ITEM_TYPE",
                metadata.symbols_used
            )

        # Validate imports
        if not isinstance(metadata.imports, list):
            result.add_error(
                "imports",
                "imports must be a list",
                "INVALID_IMPORTS_TYPE",
                type(metadata.imports).__name__
            )
        elif any(not isinstance(item, str) for item in metadata.imports):
            result.add_error(
                "imports",
                "All items in imports must be strings",
                "INVALID_IMPORTS_ITEM_TYPE",
                metadata.imports
            )

        # Validate exports
        if not isinstance(metadata.exports, list):
            result.add_error(
                "exports",
                "exports must be a list",
                "INVALID_EXPORTS_TYPE",
                type(metadata.exports).__name__
            )
        elif any(not isinstance(item, str) for item in metadata.exports):
            result.add_error(
                "exports",
                "All items in exports must be strings",
                "INVALID_EXPORTS_ITEM_TYPE",
                metadata.exports
            )

    def _validate_formats_and_constraints(self, metadata: MultiTenantMetadataSchema, result: ValidationResult):
        """Validate field formats and constraints."""

        # Priority validation
        if not isinstance(metadata.priority, int):
            result.add_error(
                "priority",
                "Priority must be an integer",
                "INVALID_PRIORITY_TYPE",
                metadata.priority
            )
        elif not 1 <= metadata.priority <= 5:
            result.add_error(
                "priority",
                "Priority must be between 1 and 5",
                "INVALID_PRIORITY_RANGE",
                metadata.priority
            )

        # Version validation
        if not isinstance(metadata.version, int):
            result.add_error(
                "version",
                "Version must be an integer",
                "INVALID_VERSION_TYPE",
                metadata.version
            )
        elif metadata.version < 1:
            result.add_error(
                "version",
                "Version must be at least 1",
                "INVALID_VERSION_VALUE",
                metadata.version
            )

        # Tags validation
        if not isinstance(metadata.tags, list):
            result.add_error(
                "tags",
                "Tags must be a list",
                "INVALID_TAGS_TYPE",
                metadata.tags
            )
        elif any(not isinstance(tag, str) for tag in metadata.tags):
            result.add_error(
                "tags",
                "All tags must be strings",
                "INVALID_TAG_TYPE",
                metadata.tags
            )

        # Category validation
        if not metadata.category:
            result.add_warning(
                "category",
                "Category should be specified",
                "MISSING_CATEGORY",
                suggestion="Set category to help organize collections"
            )
        elif not isinstance(metadata.category, str):
            result.add_error(
                "category",
                "Category must be a string",
                "INVALID_CATEGORY_TYPE",
                metadata.category
            )

        # Timestamp validation (basic format check)
        for field_name in ['created_at', 'updated_at']:
            timestamp = getattr(metadata, field_name)
            if not timestamp:
                result.add_error(
                    field_name,
                    f"{field_name.replace('_', ' ').title()} timestamp is required",
                    f"MISSING_{field_name.upper()}"
                )
            elif not isinstance(timestamp, str):
                result.add_error(
                    field_name,
                    f"{field_name.replace('_', ' ').title()} must be a string",
                    f"INVALID_{field_name.upper()}_TYPE",
                    timestamp
                )

    def validate_collection_name_pattern(self, collection_name: str,
                                       expected_category: CollectionCategory) -> ValidationResult:
        """
        Validate that a collection name matches expected naming patterns.

        Args:
            collection_name: The collection name to validate
            expected_category: The expected collection category

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult(is_valid=True)

        if expected_category == CollectionCategory.SYSTEM:
            if not self._system_collection_pattern.match(collection_name):
                result.add_error(
                    "collection_name",
                    "System collections must start with '__'",
                    "INVALID_SYSTEM_COLLECTION_NAME",
                    collection_name,
                    "Use format '__collection_name'"
                )

        elif expected_category == CollectionCategory.LIBRARY:
            if not self._library_collection_pattern.match(collection_name):
                result.add_error(
                    "collection_name",
                    "Library collections must start with '_' but not '__'",
                    "INVALID_LIBRARY_COLLECTION_NAME",
                    collection_name,
                    "Use format '_collection_name'"
                )

        elif expected_category == CollectionCategory.PROJECT:
            if not self._project_collection_pattern.match(collection_name):
                result.add_error(
                    "collection_name",
                    "Project collections must follow 'project-suffix' pattern",
                    "INVALID_PROJECT_COLLECTION_NAME",
                    collection_name,
                    "Use format 'project_name-collection_type'"
                )

        elif expected_category == CollectionCategory.GLOBAL:
            if collection_name not in self._valid_global_collections:
                result.add_warning(
                    "collection_name",
                    f"'{collection_name}' is not a standard global collection",
                    "NON_STANDARD_GLOBAL_COLLECTION_NAME",
                    collection_name,
                    f"Standard global collections: {', '.join(sorted(self._valid_global_collections))}"
                )

        return result

    def suggest_fixes(self, metadata: MultiTenantMetadataSchema) -> Dict[str, str]:
        """
        Generate suggestions for fixing common metadata issues.

        Args:
            metadata: The metadata to analyze

        Returns:
            Dictionary of field_name -> suggestion mappings
        """
        suggestions = {}

        # Project ID fix
        if not self._project_id_pattern.match(metadata.project_id):
            import hashlib
            correct_id = hashlib.sha256(metadata.project_name.encode()).hexdigest()[:12]
            suggestions['project_id'] = f"Use '{correct_id}' (generated from project name)"

        # Tenant namespace fix
        expected_namespace = f"{metadata.project_name}.{metadata.collection_type}"
        if metadata.tenant_namespace != expected_namespace:
            suggestions['tenant_namespace'] = f"Use '{expected_namespace}'"

        # Branch fix
        if not metadata.branch or not metadata.branch.strip():
            suggestions['branch'] = "Use 'main' as default branch"

        # Collection category consistency fixes
        if metadata.collection_category == CollectionCategory.LIBRARY and not metadata.mcp_readonly:
            suggestions['mcp_readonly'] = "Set to True for library collections"

        if metadata.collection_category == CollectionCategory.SYSTEM and metadata.access_level != AccessLevel.PRIVATE:
            suggestions['access_level'] = "Set to 'private' for system collections"

        return suggestions


# Export all public classes and functions
__all__ = [
    'MetadataValidator',
    'ValidationResult',
    'ValidationError',
    'ValidationSeverity'
]
