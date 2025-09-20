"""
Collection Naming Validation System for workspace-qdrant-mcp.

This module implements comprehensive collection naming validation that builds upon
the multi-tenant metadata schema from subtask 249.1. It provides:

1. Reserved naming pattern validation (memory collection, _prefix for libraries, root_name-type for projects)
2. Conflict prevention between collection types
3. Naming convention enforcement and constraints
4. Clear error messages for invalid names with helpful suggestions
5. Configurable memory collection naming support
6. Integration with existing collection management and metadata schema

Key Features:
    - Integrates with MultiTenantMetadataSchema for consistent validation
    - Prevents naming conflicts between different collection categories
    - Supports configurable memory collection naming (not just hardcoded 'memory')
    - Provides detailed error messages with suggestions for valid alternatives
    - Validates reserved patterns for system (__prefix) and library (_prefix) collections
    - Enforces project collection naming patterns (root_name-type format)
    - Thread-safe validation with comprehensive logging

Architecture:
    - CollectionNamingValidator: Main validation class with comprehensive rules
    - ValidationResult: Detailed validation outcome with errors and suggestions
    - ConfigurableMemoryNaming: Support for user-defined memory collection names
    - ConflictDetector: Advanced conflict detection across collection categories
    - PatternValidator: Specific validation for each naming pattern type

Example Usage:
    ```python
    validator = CollectionNamingValidator(memory_collection_name="user_memory")

    # Validate project collection
    result = validator.validate_name("my_project-docs", CollectionCategory.PROJECT)

    # Check for conflicts with existing collections
    conflicts = validator.check_conflicts("_mylibrary", existing_collections)

    # Get suggestions for invalid names
    suggestions = validator.get_name_suggestions("invalid-name")
    ```

Integration with Metadata Schema:
    This module uses the MultiTenantMetadataSchema and related enums from the
    metadata_schema module to ensure consistent classification and validation
    across the entire system.
"""

import re
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from loguru import logger

# Import metadata schema components from subtask 249.1
from .metadata_schema import (
    MultiTenantMetadataSchema,
    CollectionCategory,
    WorkspaceScope,
    AccessLevel,
    METADATA_SCHEMA_VERSION
)

# Import existing collection naming components for integration
from .collection_naming import (
    CollectionNamingManager,
    CollectionNameInfo,
    NamingValidationResult as LegacyNamingValidationResult,
    CollectionType as LegacyCollectionType,
    CollectionNameError,
    normalize_collection_name_component
)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"           # Hard validation failure - operation must be blocked
    WARNING = "warning"       # Soft validation issue - operation allowed with warning
    INFO = "info"            # Informational notice - operation allowed
    SUCCESS = "success"      # Validation passed completely


class ConflictType(Enum):
    """Types of naming conflicts that can occur."""

    DIRECT_DUPLICATE = "direct_duplicate"         # Exact same name already exists
    RESERVED_NAME = "reserved_name"               # Name is in reserved set
    CATEGORY_CONFLICT = "category_conflict"       # Name conflicts across categories
    PATTERN_VIOLATION = "pattern_violation"       # Name violates naming pattern rules
    MEMORY_CONFLICT = "memory_conflict"           # Conflicts with memory collection
    SYSTEM_PREFIX_ABUSE = "system_prefix_abuse"   # Misuse of __ prefix
    LIBRARY_PREFIX_ABUSE = "library_prefix_abuse" # Misuse of _ prefix
    PROJECT_FORMAT_ERROR = "project_format_error" # Invalid project collection format


@dataclass
class ValidationResult:
    """
    Comprehensive validation result with detailed feedback.

    This class provides complete information about validation outcomes including
    error details, suggestions for fixes, and metadata about the proposed name.
    """

    is_valid: bool
    severity: ValidationSeverity

    # Error and message details
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    info_message: Optional[str] = None

    # Helpful suggestions and alternatives
    suggested_names: Optional[List[str]] = None
    suggested_category: Optional[CollectionCategory] = None
    correction_hint: Optional[str] = None

    # Technical details about the conflict/issue
    conflict_type: Optional[ConflictType] = None
    conflicting_collections: Optional[List[str]] = None
    violated_rules: Optional[List[str]] = None

    # Metadata about the proposed name
    detected_category: Optional[CollectionCategory] = None
    detected_pattern: Optional[str] = None
    proposed_metadata: Optional[MultiTenantMetadataSchema] = None


@dataclass
class NamingConfiguration:
    """
    Configuration for the naming validation system.

    This allows customization of the validation behavior including memory
    collection names, project suffixes, and validation strictness.
    """

    # Memory collection configuration
    memory_collection_name: str = "memory"
    allow_custom_memory_names: bool = False
    custom_memory_patterns: Optional[List[str]] = None

    # Project collection configuration
    valid_project_suffixes: Optional[Set[str]] = None
    enforce_project_pattern: bool = True

    # System and library configuration
    system_prefix: str = "__"
    library_prefix: str = "_"

    # Validation behavior
    strict_validation: bool = True
    allow_legacy_patterns: bool = False
    generate_suggestions: bool = True
    max_suggestions: int = 5

    # Reserved names (additional to built-in ones)
    additional_reserved_names: Optional[Set[str]] = None


class PatternValidator:
    """
    Specialized validator for different naming patterns.

    This class handles the specific validation logic for each collection category
    and naming pattern, providing detailed feedback about what makes a name valid
    or invalid within each pattern.
    """

    def __init__(self, config: NamingConfiguration):
        self.config = config

        # Compile regex patterns for performance
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient validation."""

        # Project name component pattern (for project-suffix format)
        self.project_name_pattern = re.compile(r'^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$')

        # Library name pattern (after removing _ prefix)
        self.library_name_pattern = re.compile(r'^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$')

        # System memory name pattern (after removing __ prefix)
        self.system_name_pattern = re.compile(r'^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$')

        # General collection name pattern
        self.general_name_pattern = re.compile(r'^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$')

        # Memory collection patterns (if custom patterns are configured)
        if self.config.custom_memory_patterns:
            self.memory_patterns = [re.compile(pattern) for pattern in self.config.custom_memory_patterns]
        else:
            self.memory_patterns = []

    def validate_memory_collection(self, name: str) -> ValidationResult:
        """Validate memory collection name."""

        # Check against configured memory collection name
        if name == self.config.memory_collection_name:
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.SUCCESS,
                detected_category=CollectionCategory.SYSTEM,
                detected_pattern="memory_collection"
            )

        # Check custom memory patterns if enabled
        if self.config.allow_custom_memory_names and self.memory_patterns:
            for pattern in self.memory_patterns:
                if pattern.match(name):
                    return ValidationResult(
                        is_valid=True,
                        severity=ValidationSeverity.SUCCESS,
                        detected_category=CollectionCategory.SYSTEM,
                        detected_pattern="custom_memory_pattern"
                    )

        # Name doesn't match memory collection expectations
        if name.startswith(self.config.system_prefix) or name.startswith(self.config.library_prefix):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Memory collection name '{name}' cannot use system or library prefixes",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                suggested_names=[self.config.memory_collection_name],
                correction_hint="Use the configured memory collection name or a simple name without prefixes"
            )

        if not self.config.allow_custom_memory_names:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Only '{self.config.memory_collection_name}' is allowed as memory collection name",
                conflict_type=ConflictType.MEMORY_CONFLICT,
                suggested_names=[self.config.memory_collection_name],
                correction_hint="Use the configured memory collection name"
            )

        # Validate custom memory name format
        if not self.general_name_pattern.match(name):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Memory collection name '{name}' has invalid format",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint="Memory collection names must start with a letter, contain only lowercase letters, numbers, hyphens, and underscores, and end with a letter or number"
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.SUCCESS,
            detected_category=CollectionCategory.SYSTEM,
            detected_pattern="custom_memory_collection"
        )

    def validate_system_collection(self, name: str) -> ValidationResult:
        """Validate system collection name with __ prefix."""

        if not name.startswith(self.config.system_prefix):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"System collection names must start with '{self.config.system_prefix}'",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint=f"Add '{self.config.system_prefix}' prefix to make this a system collection"
            )

        if len(name) <= len(self.config.system_prefix):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"System collection name must have content after '{self.config.system_prefix}' prefix",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint=f"Add a descriptive name after the '{self.config.system_prefix}' prefix"
            )

        # Extract and validate the base name
        base_name = name[len(self.config.system_prefix):]
        if not self.system_name_pattern.match(base_name):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"System collection base name '{base_name}' has invalid format",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint="System collection names must start with a letter, contain only lowercase letters, numbers, hyphens, and underscores, and end with a letter or number"
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.SUCCESS,
            detected_category=CollectionCategory.SYSTEM,
            detected_pattern="system_prefix"
        )

    def validate_library_collection(self, name: str) -> ValidationResult:
        """Validate library collection name with _ prefix."""

        if not name.startswith(self.config.library_prefix) or name.startswith(self.config.system_prefix):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Library collection names must start with '{self.config.library_prefix}' but not '{self.config.system_prefix}'",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint=f"Use single underscore '{self.config.library_prefix}' prefix for library collections"
            )

        if len(name) <= len(self.config.library_prefix):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Library collection name must have content after '{self.config.library_prefix}' prefix",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint=f"Add a descriptive name after the '{self.config.library_prefix}' prefix"
            )

        # Extract and validate the base name
        base_name = name[len(self.config.library_prefix):]
        if not self.library_name_pattern.match(base_name):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Library collection base name '{base_name}' has invalid format",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint="Library collection names must start with a letter, contain only lowercase letters, numbers, hyphens, and underscores, and end with a letter or number"
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.SUCCESS,
            detected_category=CollectionCategory.LIBRARY,
            detected_pattern="library_prefix"
        )

    def validate_project_collection(self, name: str) -> ValidationResult:
        """Validate project collection name with root_name-type format."""

        # Check if it looks like a system or library collection
        if name.startswith(self.config.system_prefix) or name.startswith(self.config.library_prefix):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Project collection names cannot use reserved prefixes '{self.config.system_prefix}' or '{self.config.library_prefix}'",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint="Use format 'project-name-suffix' without prefixes for project collections"
            )

        # Parse project collection format: project-name-suffix
        parts = name.split('-')
        if len(parts) < 2:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Project collection name '{name}' must use format 'project-name-suffix'",
                conflict_type=ConflictType.PROJECT_FORMAT_ERROR,
                correction_hint="Use hyphen-separated format: 'project-name-suffix'"
            )

        suffix = parts[-1]
        project_name = '-'.join(parts[:-1])

        # Validate project name component
        if not self.project_name_pattern.match(project_name):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Project name component '{project_name}' has invalid format",
                conflict_type=ConflictType.PROJECT_FORMAT_ERROR,
                correction_hint="Project names must start with a letter, contain only lowercase letters, numbers, hyphens, and underscores, and end with a letter or number"
            )

        # Validate suffix if project suffixes are configured
        if self.config.valid_project_suffixes and suffix not in self.config.valid_project_suffixes:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Project collection suffix '{suffix}' is not valid",
                conflict_type=ConflictType.PROJECT_FORMAT_ERROR,
                suggested_names=[f"{project_name}-{s}" for s in list(self.config.valid_project_suffixes)[:3]],
                correction_hint=f"Valid suffixes: {', '.join(sorted(self.config.valid_project_suffixes))}"
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.SUCCESS,
            detected_category=CollectionCategory.PROJECT,
            detected_pattern="project_pattern"
        )

    def validate_global_collection(self, name: str) -> ValidationResult:
        """Validate global collection name."""

        # Check if it looks like other collection types
        if name.startswith(self.config.system_prefix) or name.startswith(self.config.library_prefix):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Global collection names cannot use reserved prefixes '{self.config.system_prefix}' or '{self.config.library_prefix}'",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint="Use simple names without prefixes for global collections"
            )

        # Check if it looks like a project collection
        if '-' in name and len(name.split('-')) >= 2:
            parts = name.split('-')
            if self.config.valid_project_suffixes and parts[-1] in self.config.valid_project_suffixes:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    warning_message=f"Global collection name '{name}' resembles project collection format",
                    detected_category=CollectionCategory.GLOBAL,
                    detected_pattern="global_collection",
                    correction_hint="Consider using a simpler name to avoid confusion with project collections"
                )

        # Validate basic format
        if not self.general_name_pattern.match(name):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Global collection name '{name}' has invalid format",
                conflict_type=ConflictType.PATTERN_VIOLATION,
                correction_hint="Global collection names must start with a letter, contain only lowercase letters, numbers, hyphens, and underscores, and end with a letter or number"
            )

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.SUCCESS,
            detected_category=CollectionCategory.GLOBAL,
            detected_pattern="global_collection"
        )


class ConflictDetector:
    """
    Advanced conflict detection across collection categories.

    This class identifies potential naming conflicts that could cause confusion
    or operational issues, including cross-category conflicts and reserved name
    violations.
    """

    def __init__(self, config: NamingConfiguration, pattern_validator: PatternValidator):
        self.config = config
        self.pattern_validator = pattern_validator

        # Built-in reserved names that cannot be used
        self.built_in_reserved = {
            "system", "admin", "config", "settings", "cache", "temp", "tmp",
            "log", "logs", "debug", "test", "tests", "backup", "restore",
            "index", "metadata", "schema", "migration", "version"
        }

        # Combine with user-configured reserved names
        self.reserved_names = self.built_in_reserved.copy()
        if config.additional_reserved_names:
            self.reserved_names.update(config.additional_reserved_names)

    def detect_conflicts(self, name: str, existing_collections: List[str], intended_category: Optional[CollectionCategory] = None) -> ValidationResult:
        """
        Detect naming conflicts with existing collections and reserved names.

        Args:
            name: The proposed collection name
            existing_collections: List of existing collection names
            intended_category: The intended category for the new collection

        Returns:
            ValidationResult with conflict details
        """

        # Check for direct duplicates
        if name in existing_collections:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Collection '{name}' already exists",
                conflict_type=ConflictType.DIRECT_DUPLICATE,
                conflicting_collections=[name]
            )

        # Check reserved names
        if name in self.reserved_names:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"'{name}' is a reserved collection name",
                conflict_type=ConflictType.RESERVED_NAME,
                correction_hint="Choose a different name that doesn't conflict with reserved system names"
            )

        # Check memory collection conflicts
        memory_conflict = self._check_memory_conflicts(name, existing_collections)
        if memory_conflict:
            return memory_conflict

        # Check cross-category conflicts
        category_conflict = self._check_category_conflicts(name, existing_collections, intended_category)
        if category_conflict:
            return category_conflict

        # Check for prefix abuse
        prefix_conflict = self._check_prefix_abuse(name, intended_category)
        if prefix_conflict:
            return prefix_conflict

        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.SUCCESS
        )

    def _check_memory_conflicts(self, name: str, existing_collections: List[str]) -> Optional[ValidationResult]:
        """Check for conflicts with memory collection naming."""

        # If proposing the configured memory collection name
        if name == self.config.memory_collection_name:
            # Check if memory collection already exists
            if self.config.memory_collection_name in existing_collections:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    error_message=f"Memory collection '{self.config.memory_collection_name}' already exists",
                    conflict_type=ConflictType.MEMORY_CONFLICT,
                    conflicting_collections=[self.config.memory_collection_name]
                )

        # Check for similar memory-related names that could cause confusion
        memory_like_names = []
        for existing in existing_collections:
            if "memory" in existing.lower() and existing != name:
                memory_like_names.append(existing)

        if memory_like_names and "memory" in name.lower():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                warning_message=f"Name '{name}' is similar to existing memory-related collections",
                conflict_type=ConflictType.MEMORY_CONFLICT,
                conflicting_collections=memory_like_names,
                correction_hint="Consider using a more distinctive name to avoid confusion"
            )

        return None

    def _check_category_conflicts(self, name: str, existing_collections: List[str], intended_category: Optional[CollectionCategory]) -> Optional[ValidationResult]:
        """Check for conflicts across collection categories."""

        conflicts = []

        # Check if a library version exists when proposing non-library
        if not name.startswith(self.config.library_prefix):
            library_name = f"{self.config.library_prefix}{name}"
            if library_name in existing_collections:
                conflicts.append(library_name)

        # Check if a non-library version exists when proposing library
        if name.startswith(self.config.library_prefix):
            display_name = name[len(self.config.library_prefix):]
            if display_name in existing_collections:
                conflicts.append(display_name)

        # Check if a system version exists
        if not name.startswith(self.config.system_prefix):
            system_name = f"{self.config.system_prefix}{name}"
            if system_name in existing_collections:
                conflicts.append(system_name)

        if conflicts:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Name '{name}' conflicts with existing collections: {', '.join(conflicts)}",
                conflict_type=ConflictType.CATEGORY_CONFLICT,
                conflicting_collections=conflicts,
                correction_hint="Choose a different base name to avoid conflicts across collection categories"
            )

        return None

    def _check_prefix_abuse(self, name: str, intended_category: Optional[CollectionCategory]) -> Optional[ValidationResult]:
        """Check for misuse of reserved prefixes."""

        # Check system prefix abuse
        if name.startswith(self.config.system_prefix):
            if intended_category and intended_category != CollectionCategory.SYSTEM:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    error_message=f"Only system collections can use '{self.config.system_prefix}' prefix",
                    conflict_type=ConflictType.SYSTEM_PREFIX_ABUSE,
                    correction_hint=f"Remove '{self.config.system_prefix}' prefix or create as system collection"
                )

        # Check library prefix abuse
        if name.startswith(self.config.library_prefix) and not name.startswith(self.config.system_prefix):
            if intended_category and intended_category != CollectionCategory.LIBRARY:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    error_message=f"Only library collections can use '{self.config.library_prefix}' prefix",
                    conflict_type=ConflictType.LIBRARY_PREFIX_ABUSE,
                    correction_hint=f"Remove '{self.config.library_prefix}' prefix or create as library collection"
                )

        return None


class CollectionNamingValidator:
    """
    Main collection naming validation system.

    This is the primary interface for validating collection names according to
    the comprehensive naming rules, including integration with the metadata schema
    and conflict detection across all collection categories.
    """

    def __init__(self, config: Optional[NamingConfiguration] = None):
        """
        Initialize the collection naming validator.

        Args:
            config: Optional configuration object. If None, uses default configuration.
        """
        self.config = config or NamingConfiguration()
        self.pattern_validator = PatternValidator(self.config)
        self.conflict_detector = ConflictDetector(self.config, self.pattern_validator)

        # Integration with existing naming manager for compatibility
        self.legacy_naming_manager = CollectionNamingManager()

        logger.info(f"CollectionNamingValidator initialized with memory collection: {self.config.memory_collection_name}")

    def validate_name(self, name: str, intended_category: Optional[CollectionCategory] = None, existing_collections: Optional[List[str]] = None) -> ValidationResult:
        """
        Validate a collection name comprehensively.

        Args:
            name: The proposed collection name
            intended_category: The intended category for the collection
            existing_collections: List of existing collection names to check conflicts

        Returns:
            ValidationResult with detailed validation outcome
        """
        logger.debug(f"Validating collection name: {name} (intended category: {intended_category})")

        # Basic validation
        if not name or not name.strip():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message="Collection name cannot be empty",
                conflict_type=ConflictType.PATTERN_VIOLATION
            )

        name = name.strip().lower()

        # Length validation
        if len(name) > 100:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message="Collection name cannot exceed 100 characters",
                conflict_type=ConflictType.PATTERN_VIOLATION
            )

        # Check reserved names first (always, regardless of existing collections)
        if name in self.conflict_detector.reserved_names:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"'{name}' is a reserved collection name",
                conflict_type=ConflictType.RESERVED_NAME,
                correction_hint="Choose a different name that doesn't conflict with reserved system names"
            )

        # Auto-detect category if not provided
        if intended_category is None:
            intended_category = self._detect_category(name)

        # Validate based on detected/intended category
        pattern_result = self._validate_by_category(name, intended_category)
        if not pattern_result.is_valid:
            return pattern_result

        # Check for conflicts with existing collections
        if existing_collections:
            conflict_result = self.conflict_detector.detect_conflicts(name, existing_collections, intended_category)
            if not conflict_result.is_valid:
                return conflict_result

        # Generate metadata schema for successful validation
        try:
            metadata = self._generate_metadata_schema(name, intended_category)
            pattern_result.proposed_metadata = metadata
        except Exception as e:
            logger.warning(f"Failed to generate metadata schema for {name}: {e}")

        return pattern_result

    def check_conflicts(self, name: str, existing_collections: List[str], intended_category: Optional[CollectionCategory] = None) -> ValidationResult:
        """
        Check for naming conflicts with existing collections.

        Args:
            name: The proposed collection name
            existing_collections: List of existing collection names
            intended_category: The intended category for the collection

        Returns:
            ValidationResult focused on conflict detection
        """
        return self.conflict_detector.detect_conflicts(name, existing_collections, intended_category)

    def get_name_suggestions(self, invalid_name: str, intended_category: Optional[CollectionCategory] = None, count: int = 5) -> List[str]:
        """
        Get suggested valid names based on an invalid name.

        Args:
            invalid_name: The invalid name to base suggestions on
            intended_category: The intended category for suggestions
            count: Maximum number of suggestions to return

        Returns:
            List of suggested valid collection names
        """
        if not self.config.generate_suggestions:
            return []

        suggestions = []
        base_name = invalid_name.strip().lower()

        # Remove problematic characters and normalize
        base_name = normalize_collection_name_component(base_name)

        # Generate suggestions based on category
        if intended_category == CollectionCategory.SYSTEM:
            if not base_name.startswith(self.config.system_prefix):
                suggestions.append(f"{self.config.system_prefix}{base_name}")
            if base_name != "memory":
                suggestions.append(f"{self.config.system_prefix}memory")

        elif intended_category == CollectionCategory.LIBRARY:
            if not base_name.startswith(self.config.library_prefix):
                suggestions.append(f"{self.config.library_prefix}{base_name}")
            suggestions.extend([f"{self.config.library_prefix}tools", f"{self.config.library_prefix}utils", f"{self.config.library_prefix}helpers"])

        elif intended_category == CollectionCategory.PROJECT:
            if self.config.valid_project_suffixes:
                for suffix in list(self.config.valid_project_suffixes)[:3]:
                    suggestions.append(f"{base_name}-{suffix}")
            else:
                suggestions.extend([f"{base_name}-docs", f"{base_name}-notes", f"{base_name}-scratchbook"])

        elif intended_category == CollectionCategory.GLOBAL:
            suggestions.extend([base_name, f"global_{base_name}", f"{base_name}_global"])

        else:
            # Generic suggestions
            suggestions.extend([
                base_name,
                f"{base_name}_collection",
                f"my_{base_name}",
                self.config.memory_collection_name
            ])

        # Remove duplicates and invalid suggestions, limit count
        valid_suggestions = []
        for suggestion in suggestions:
            if suggestion != invalid_name:
                result = self.validate_name(suggestion, intended_category)
                if result.is_valid:
                    valid_suggestions.append(suggestion)
                    if len(valid_suggestions) >= count:
                        break

        return valid_suggestions

    def _detect_category(self, name: str) -> CollectionCategory:
        """Detect the most likely category for a collection name."""

        # Memory collection
        if name == self.config.memory_collection_name:
            return CollectionCategory.SYSTEM

        # System collections
        if name.startswith(self.config.system_prefix):
            return CollectionCategory.SYSTEM

        # Library collections
        if name.startswith(self.config.library_prefix) and not name.startswith(self.config.system_prefix):
            return CollectionCategory.LIBRARY

        # Project collections (contains hyphens and valid suffix)
        if '-' in name and len(name.split('-')) >= 2:
            parts = name.split('-')
            suffix = parts[-1]
            if self.config.valid_project_suffixes and suffix in self.config.valid_project_suffixes:
                return CollectionCategory.PROJECT

        # Default to global
        return CollectionCategory.GLOBAL

    def _validate_by_category(self, name: str, category: CollectionCategory) -> ValidationResult:
        """Validate a name according to its category-specific rules."""

        if category == CollectionCategory.SYSTEM:
            # Check if it's a memory collection
            if name == self.config.memory_collection_name or (self.config.allow_custom_memory_names and not name.startswith(self.config.system_prefix)):
                return self.pattern_validator.validate_memory_collection(name)
            else:
                return self.pattern_validator.validate_system_collection(name)

        elif category == CollectionCategory.LIBRARY:
            return self.pattern_validator.validate_library_collection(name)

        elif category == CollectionCategory.PROJECT:
            return self.pattern_validator.validate_project_collection(name)

        elif category == CollectionCategory.GLOBAL:
            return self.pattern_validator.validate_global_collection(name)

        else:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                error_message=f"Unknown collection category: {category}",
                conflict_type=ConflictType.PATTERN_VIOLATION
            )

    def _generate_metadata_schema(self, name: str, category: CollectionCategory) -> MultiTenantMetadataSchema:
        """Generate metadata schema for a validated collection name."""

        if category == CollectionCategory.SYSTEM:
            if name == self.config.memory_collection_name:
                return MultiTenantMetadataSchema.create_for_system(
                    collection_name=f"__{name}",
                    collection_type="memory_collection"
                )
            else:
                return MultiTenantMetadataSchema.create_for_system(
                    collection_name=name,
                    collection_type="system_collection"
                )

        elif category == CollectionCategory.LIBRARY:
            return MultiTenantMetadataSchema.create_for_library(
                collection_name=name,
                collection_type="library_collection"
            )

        elif category == CollectionCategory.PROJECT:
            parts = name.split('-')
            project_name = '-'.join(parts[:-1])
            collection_type = parts[-1]
            return MultiTenantMetadataSchema.create_for_project(
                project_name=project_name,
                collection_type=collection_type
            )

        elif category == CollectionCategory.GLOBAL:
            return MultiTenantMetadataSchema.create_for_global(
                collection_name=name,
                collection_type="global"
            )

        else:
            raise ValueError(f"Cannot generate metadata for unknown category: {category}")


# Convenience functions for integration with existing systems

def create_naming_validator(memory_collection_name: str = "memory", **kwargs) -> CollectionNamingValidator:
    """
    Create a collection naming validator with custom configuration.

    Args:
        memory_collection_name: Name of the memory collection
        **kwargs: Additional configuration options

    Returns:
        Configured CollectionNamingValidator instance
    """
    config = NamingConfiguration(memory_collection_name=memory_collection_name, **kwargs)
    return CollectionNamingValidator(config)


def validate_collection_name_with_metadata(name: str, category: Optional[CollectionCategory] = None, existing_collections: Optional[List[str]] = None) -> ValidationResult:
    """
    Validate a collection name and return metadata-enhanced result.

    Args:
        name: The collection name to validate
        category: Optional intended category
        existing_collections: Optional list of existing collections

    Returns:
        ValidationResult with metadata schema if valid
    """
    validator = CollectionNamingValidator()
    return validator.validate_name(name, category, existing_collections)


def check_collection_conflicts(name: str, existing_collections: List[str]) -> ValidationResult:
    """
    Check for collection naming conflicts.

    Args:
        name: The proposed collection name
        existing_collections: List of existing collection names

    Returns:
        ValidationResult focused on conflict detection
    """
    validator = CollectionNamingValidator()
    return validator.check_conflicts(name, existing_collections)


# Export all public classes and functions
__all__ = [
    'CollectionNamingValidator',
    'ValidationResult',
    'NamingConfiguration',
    'PatternValidator',
    'ConflictDetector',
    'ValidationSeverity',
    'ConflictType',
    'create_naming_validator',
    'validate_collection_name_with_metadata',
    'check_collection_conflicts'
]