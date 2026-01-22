"""
Collection naming system for workspace-qdrant-mcp.

CANONICAL ARCHITECTURE (ADR-001):
    This module enforces the canonical 3-collection multi-tenant architecture.

    Canonical Collections:
    1. 'memory' - Global behavioral rules and instructions
    2. 'projects' - All project code/documents (tenant isolation via project_id)
    3. 'libraries' - All library documentation (tenant isolation via library_name)

    DEPRECATED Patterns (rejected with helpful error):
    - _projects, _libraries, _memory, _agent_memory (underscore-prefixed)
    - __memory, __system (double-underscore system)
    - _{project_id} (per-project collections)
    - {project}-scratchbook, {project}-docs, {project}-code (suffixed)

Key features:
- Validates collection names against canonical architecture
- Rejects deprecated patterns with migration guidance
- Provides helpful error messages pointing to ADR-001
- Maintains backward compatibility for read operations during migration

See: docs/decisions/ADR-001-multi-tenant-architecture.md
"""

import re
from dataclasses import dataclass
from enum import Enum

from loguru import logger

# logger imported from loguru


class CollectionType(Enum):
    """
    Types of collections in the canonical naming system (ADR-001).

    Canonical types:
    - MEMORY: 'memory' collection for behavioral rules
    - PROJECTS: 'projects' collection for all project data
    - LIBRARIES: 'libraries' collection for documentation

    Deprecated types (kept for migration/compatibility):
    - LEGACY: Collections using deprecated naming patterns
    """

    # Canonical collection types (ADR-001)
    MEMORY = "memory"          # 'memory' collection - behavioral rules
    PROJECTS = "projects"      # 'projects' collection - multi-tenant project data
    LIBRARIES = "libraries"    # 'libraries' collection - documentation

    # Deprecated - kept for backward compatibility during migration
    LEGACY = "legacy"          # Deprecated collection patterns


@dataclass
class CollectionNameInfo:
    """Information about a collection name and its classification (ADR-001)."""

    name: str  # Actual collection name in Qdrant
    display_name: str  # Name shown to users
    collection_type: CollectionType
    is_canonical: bool  # Whether this is a canonical collection (memory/projects/libraries)
    is_deprecated: bool = False  # Whether this uses deprecated naming patterns
    deprecation_message: str | None = None  # Migration guidance for deprecated collections


@dataclass
class NamingValidationResult:
    """Result of collection name validation."""

    is_valid: bool
    error_message: str | None = None
    warning_message: str | None = None
    collection_info: CollectionNameInfo | None = None


class CollectionNamingManager:
    """
    Manages the canonical collection naming system (ADR-001).

    CANONICAL COLLECTIONS:
    - 'memory' - Global behavioral rules
    - 'projects' - Multi-tenant project data
    - 'libraries' - Multi-tenant library documentation

    DEPRECATED PATTERNS (rejected with migration guidance):
    - Underscore-prefixed: _projects, _libraries, _memory, _{project_id}
    - Double-underscore: __memory, __system
    - Per-project suffixed: {name}-scratchbook, {name}-docs, {name}-code

    See: docs/decisions/ADR-001-multi-tenant-architecture.md
    """

    # Canonical collection names (ADR-001)
    CANONICAL_COLLECTIONS = {"memory", "projects", "libraries"}

    # Deprecated patterns that will be rejected with helpful messages
    DEPRECATED_PATTERNS = {
        "_projects", "_libraries", "_memory", "_agent_memory",
        "__memory", "__system",
    }

    # Deprecated prefixes (any collection starting with these is deprecated)
    DEPRECATED_PREFIXES = ("_", "__")

    # Deprecated suffixes from old per-project naming
    DEPRECATED_SUFFIXES = ("-scratchbook", "-docs", "-code", "-notes", "-memory")

    def __init__(self):
        """Initialize the canonical collection naming manager (ADR-001)."""
        pass

    def validate_collection_name(
        self, name: str, intended_type: CollectionType | None = None
    ) -> NamingValidationResult:
        """
        Validate a collection name against canonical architecture (ADR-001).

        Only canonical collection names are accepted:
        - 'memory' - Global behavioral rules
        - 'projects' - Multi-tenant project data
        - 'libraries' - Multi-tenant library documentation

        Deprecated patterns are rejected with helpful migration guidance.

        Args:
            name: The proposed collection name
            intended_type: Optional hint about intended collection type (ignored)

        Returns:
            NamingValidationResult with validation status and details
        """
        # Basic name validation
        if not name or not name.strip():
            return NamingValidationResult(
                is_valid=False, error_message="Collection name cannot be empty"
            )

        name = name.strip()

        if len(name) > 100:
            return NamingValidationResult(
                is_valid=False,
                error_message="Collection name cannot exceed 100 characters",
            )

        # Classify the collection
        collection_info = self._classify_collection_name(name)

        # Reject deprecated patterns with helpful guidance
        if collection_info.is_deprecated:
            return NamingValidationResult(
                is_valid=False,
                error_message=collection_info.deprecation_message,
                collection_info=collection_info,
            )

        # Only canonical collections are valid for new operations
        if not collection_info.is_canonical:
            return NamingValidationResult(
                is_valid=False,
                error_message=(
                    f"Collection '{name}' is not a canonical collection. "
                    f"Only these collections are supported: {', '.join(sorted(self.CANONICAL_COLLECTIONS))}. "
                    f"See ADR-001 for architecture details."
                ),
                collection_info=collection_info,
            )

        return NamingValidationResult(is_valid=True, collection_info=collection_info)

    def check_naming_conflicts(
        self, proposed_name: str, existing_collections: list[str]
    ) -> NamingValidationResult:
        """
        Check for naming conflicts with existing collections (ADR-001).

        In the canonical architecture, only memory/projects/libraries are valid.
        This method validates against canonical names and checks existence.

        Args:
            proposed_name: The name being proposed for creation
            existing_collections: List of existing collection names

        Returns:
            NamingValidationResult indicating if conflicts exist
        """
        # First validate the proposed name against canonical architecture
        validation = self.validate_collection_name(proposed_name)
        if not validation.is_valid:
            return validation

        existing_set = set(existing_collections)

        # Check if collection already exists
        if proposed_name in existing_set:
            return NamingValidationResult(
                is_valid=False,
                error_message=f"Collection '{proposed_name}' already exists",
            )

        return NamingValidationResult(
            is_valid=True, collection_info=validation.collection_info
        )

    def get_collection_info(self, name: str) -> CollectionNameInfo:
        """
        Get detailed information about a collection name (ADR-001).

        Args:
            name: The collection name to analyze

        Returns:
            CollectionNameInfo with canonical/deprecated classification
        """
        return self._classify_collection_name(name)

    def get_display_name(self, collection_name: str) -> str:
        """
        Get the user-facing display name for a collection.

        In the canonical architecture, display names equal collection names.

        Args:
            collection_name: The actual collection name in Qdrant

        Returns:
            The display name (same as collection name in ADR-001)
        """
        info = self._classify_collection_name(collection_name)
        return info.display_name

    def is_canonical_collection(self, collection_name: str) -> bool:
        """
        Check if a collection is a canonical collection (ADR-001).

        Canonical collections: memory, projects, libraries

        Args:
            collection_name: The collection name to check

        Returns:
            True if canonical, False otherwise
        """
        return collection_name in self.CANONICAL_COLLECTIONS

    def is_deprecated_collection(self, collection_name: str) -> bool:
        """
        Check if a collection uses deprecated naming patterns.

        Args:
            collection_name: The collection name to check

        Returns:
            True if deprecated, False otherwise
        """
        info = self._classify_collection_name(collection_name)
        return info.is_deprecated

    def filter_canonical_collections(self, all_collections: list[str]) -> list[str]:
        """
        Filter a list to include only canonical collections (ADR-001).

        Args:
            all_collections: List of all collection names from Qdrant

        Returns:
            List of canonical collection names (memory, projects, libraries)
        """
        return [c for c in all_collections if c in self.CANONICAL_COLLECTIONS]

    def get_canonical_collection_name(self, collection_type: CollectionType) -> str:
        """
        Get the canonical collection name for a type (ADR-001).

        Args:
            collection_type: MEMORY, PROJECTS, or LIBRARIES

        Returns:
            Canonical collection name

        Raises:
            ValueError: If type is not a canonical type
        """
        type_to_name = {
            CollectionType.MEMORY: "memory",
            CollectionType.PROJECTS: "projects",
            CollectionType.LIBRARIES: "libraries",
        }
        if collection_type not in type_to_name:
            raise ValueError(
                f"Invalid collection type: {collection_type}. "
                f"Canonical types are: MEMORY, PROJECTS, LIBRARIES"
            )
        return type_to_name[collection_type]

    def _classify_collection_name(self, name: str) -> CollectionNameInfo:
        """
        Classify a collection name according to canonical architecture (ADR-001).

        Args:
            name: The collection name to classify

        Returns:
            CollectionNameInfo with canonical/deprecated classification
        """
        # Canonical collections (ADR-001)
        if name == "memory":
            return CollectionNameInfo(
                name=name,
                display_name=name,
                collection_type=CollectionType.MEMORY,
                is_canonical=True,
            )

        if name == "projects":
            return CollectionNameInfo(
                name=name,
                display_name=name,
                collection_type=CollectionType.PROJECTS,
                is_canonical=True,
            )

        if name == "libraries":
            return CollectionNameInfo(
                name=name,
                display_name=name,
                collection_type=CollectionType.LIBRARIES,
                is_canonical=True,
            )

        # Check for deprecated patterns
        if name in self.DEPRECATED_PATTERNS:
            return CollectionNameInfo(
                name=name,
                display_name=name,
                collection_type=CollectionType.LEGACY,
                is_canonical=False,
                is_deprecated=True,
                deprecation_message=(
                    f"Collection '{name}' uses deprecated naming. "
                    f"Use canonical collections: memory, projects, libraries. "
                    f"See ADR-001 for migration guidance."
                ),
            )

        # Check for underscore-prefixed (deprecated)
        if name.startswith("__"):
            return CollectionNameInfo(
                name=name,
                display_name=name,
                collection_type=CollectionType.LEGACY,
                is_canonical=False,
                is_deprecated=True,
                deprecation_message=(
                    f"Double-underscore collection '{name}' is deprecated. "
                    f"Use canonical collections: memory, projects, libraries. "
                    f"See ADR-001 for migration guidance."
                ),
            )

        if name.startswith("_"):
            return CollectionNameInfo(
                name=name,
                display_name=name,
                collection_type=CollectionType.LEGACY,
                is_canonical=False,
                is_deprecated=True,
                deprecation_message=(
                    f"Underscore-prefixed collection '{name}' is deprecated. "
                    f"Use canonical collections: memory, projects, libraries. "
                    f"See ADR-001 for migration guidance."
                ),
            )

        # Check for deprecated suffixes (old per-project naming)
        for suffix in self.DEPRECATED_SUFFIXES:
            if name.endswith(suffix):
                return CollectionNameInfo(
                    name=name,
                    display_name=name,
                    collection_type=CollectionType.LEGACY,
                    is_canonical=False,
                    is_deprecated=True,
                    deprecation_message=(
                        f"Collection '{name}' uses deprecated per-project naming. "
                        f"Use canonical 'projects' collection with project_id metadata filtering. "
                        f"See ADR-001 for migration guidance."
                    ),
                )

        # Unknown collection - not canonical but not explicitly deprecated
        return CollectionNameInfo(
            name=name,
            display_name=name,
            collection_type=CollectionType.LEGACY,
            is_canonical=False,
            is_deprecated=False,
        )


class CollectionNameError(Exception):
    """Exception raised when a collection name is invalid."""

    pass


class CollectionPermissionError(Exception):
    """Exception raised when attempting prohibited collection operations."""

    pass


def validate_collection_name(name: str, allow_library: bool = False) -> None:
    """
    Validate a collection name against canonical architecture (ADR-001).

    Only canonical collection names are valid:
    - 'memory' - Global behavioral rules
    - 'projects' - Multi-tenant project data
    - 'libraries' - Multi-tenant library documentation

    Args:
        name: The collection name to validate
        allow_library: Ignored in canonical architecture (all deprecated prefixes rejected)

    Raises:
        CollectionNameError: If the collection name is not canonical
    """
    manager = CollectionNamingManager()
    result = manager.validate_collection_name(name)

    if not result.is_valid:
        raise CollectionNameError(f"Invalid collection name: {result.error_message}")


def normalize_collection_name_component(name: str) -> str:
    """
    Normalize a collection name component to be valid in Qdrant.

    Args:
        name: The name component to normalize

    Returns:
        Normalized name component (lowercase, alphanumeric with hyphens)
    """
    # Convert to lowercase and replace invalid characters with hyphens
    normalized = re.sub(r'[^a-z0-9-]', '-', name.lower())
    # Remove multiple consecutive hyphens
    normalized = re.sub(r'-+', '-', normalized)
    # Remove leading/trailing hyphens
    return normalized.strip('-')


def build_project_collection_name(project_id: str) -> str:
    """
    DEPRECATED: Build a project collection name from tenant ID.

    This function is deprecated per ADR-001. Use the canonical 'projects'
    collection with project_id metadata filtering instead.

    Args:
        project_id: The tenant ID (ignored in canonical architecture)

    Returns:
        Always returns 'projects' (the canonical collection name)

    .. deprecated::
        Use canonical 'projects' collection with project_id metadata filtering.
        See ADR-001 for migration guidance.
    """
    logger.warning(
        f"DEPRECATION: build_project_collection_name() is deprecated. "
        f"Use canonical 'projects' collection with project_id='{project_id}' metadata filtering. "
        f"See ADR-001 for migration guidance."
    )
    # Return canonical collection name instead of per-project collection
    return "projects"


def build_user_collection_name(name: str, suffix: str) -> str:
    """
    DEPRECATED: Build a user collection name from name and suffix.

    This function is deprecated per ADR-001. User notes should be stored
    in the canonical 'projects' collection with appropriate metadata.

    Args:
        name: The collection name (ignored)
        suffix: The collection type suffix (ignored)

    Returns:
        Always returns 'projects' (the canonical collection name)

    .. deprecated::
        Store user content in 'projects' collection with metadata.
        See ADR-001 for migration guidance.
    """
    logger.warning(
        f"DEPRECATION: build_user_collection_name() is deprecated. "
        f"Store user content in canonical 'projects' collection with appropriate metadata. "
        f"See ADR-001 for migration guidance."
    )
    return "projects"


def build_system_memory_collection_name(memory_name: str) -> str:
    """
    DEPRECATED: Build a system memory collection name with proper prefix.

    This function is deprecated per ADR-001. Use the canonical 'memory'
    collection with tags for categorization.

    Args:
        memory_name: The memory collection name (ignored)

    Returns:
        Always returns 'memory' (the canonical collection name)

    .. deprecated::
        Use canonical 'memory' collection with tags.
        See ADR-001 for migration guidance.
    """
    logger.warning(
        f"DEPRECATION: build_system_memory_collection_name() is deprecated. "
        f"Use canonical 'memory' collection with tags for categorization. "
        f"See ADR-001 for migration guidance."
    )
    return "memory"


def create_naming_manager() -> CollectionNamingManager:
    """
    Create a collection naming manager instance (ADR-001).

    Returns:
        Configured CollectionNamingManager instance
    """
    return CollectionNamingManager()


# Import Task 181 Collection Rules Enforcement System
class ValidationSource(Enum):
    """Source of the collection operation request."""
    LLM = "llm"           # Request from LLM via MCP
    CLI = "cli"           # Request from CLI/Rust engine
    MCP_INTERNAL = "mcp"  # Internal MCP operations
    SYSTEM = "system"     # System/admin operations


class OperationType(Enum):
    """Types of collection operations that need validation."""
    CREATE = "create"
    DELETE = "delete"
    WRITE = "write"
    READ = "read"
    LIST = "list"


@dataclass
class ValidationResult:
    """Result of collection operation validation."""
    is_valid: bool
    error_message: str | None = None
    warning_message: str | None = None
    suggested_alternatives: list[str] | None = None
    violation_type: str | None = None


class CollectionRulesEnforcementError(Exception):
    """Exception raised when collection rules enforcement is violated."""

    def __init__(self, validation_result: ValidationResult):
        self.validation_result = validation_result
        super().__init__(validation_result.error_message)


class CollectionRulesEnforcer:
    """
    Collection management rules enforcer for canonical architecture (ADR-001).

    This class provides unified validation for all collection operations,
    ensuring only canonical collections (memory, projects, libraries) are used.

    Key Features:
    - Validates all operations against canonical architecture
    - Rejects deprecated collection patterns with helpful messages
    - Source-aware validation (LLM vs CLI vs MCP internal)
    - Comprehensive logging and audit trail

    Security Boundaries (ADR-001):
    - Only canonical collections can be created/modified
    - Deprecated patterns are rejected with migration guidance
    - All writes route through daemon (First Principle 10)
    """

    def __init__(self, config=None):
        """
        Initialize the collection rules enforcer.

        Args:
            config: Optional configuration object for context
        """
        self.config = config

        # Initialize subsystems
        try:
            from .collection_types import CollectionType as TypesCollectionType
            from .collection_types import CollectionTypeClassifier
            from .llm_access_control import LLMAccessControlError, LLMAccessController

            self.llm_access_controller = LLMAccessController(config)
            self.type_classifier = CollectionTypeClassifier()
        except ImportError:
            # Handle case where dependencies are not available
            logger.warning("Some enforcement dependencies not available - using basic validation only")
            self.llm_access_controller = None
            self.type_classifier = None

        self.naming_manager = CollectionNamingManager()

        # Track existing collections for validation
        self._existing_collections: set[str] = set()

        logger.info("CollectionRulesEnforcer initialized")

    def set_existing_collections(self, collections: list[str]) -> None:
        """
        Update the set of existing collections for validation.

        Args:
            collections: List of currently existing collection names
        """
        self._existing_collections = set(collections)
        if self.llm_access_controller:
            self.llm_access_controller.set_existing_collections(collections)
        logger.debug(f"Updated existing collections: {len(collections)}")

    def validate_collection_creation(self, name: str, source: ValidationSource) -> ValidationResult:
        """
        Validate collection creation request with comprehensive rules enforcement.

        Args:
            name: The collection name to create
            source: Source of the creation request

        Returns:
            ValidationResult with validation status and details
        """
        logger.debug(f"Validating collection creation: {name} from {source.value}")

        # Basic parameter validation
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Collection name must be a non-empty string",
                violation_type="invalid_collection_name"
            )

        name = name.strip()

        # Check for existing collection
        if name in self._existing_collections:
            return ValidationResult(
                is_valid=False,
                error_message=f"Collection '{name}' already exists",
                violation_type="collection_already_exists"
            )

        # Validate collection name format
        # Skip naming manager validation for system collections as it doesn't handle __ prefix correctly
        if not name.startswith("__"):
            naming_result = self.naming_manager.validate_collection_name(name)
            if not naming_result.is_valid:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Invalid collection name format: {naming_result.error_message}",
                    violation_type="invalid_collection_name"
                )
        else:
            # Basic validation for system collections
            if len(name) < 3 or not name[2:]:  # Must have content after __
                return ValidationResult(
                    is_valid=False,
                    error_message=f"System collection name '{name}' must have content after '__' prefix",
                    violation_type="invalid_collection_name"
                )

        # Source-specific validation
        if source == ValidationSource.LLM and self.llm_access_controller:
            # LLM operations must go through LLM access control
            try:
                self.llm_access_controller.validate_llm_collection_access("create", name)
            except Exception as e:  # LLMAccessControlError
                return ValidationResult(
                    is_valid=False,
                    error_message=str(e),
                    violation_type="llm_access_denied"
                )

        logger.debug(f"Collection creation validation passed: {name} from {source.value}")
        return ValidationResult(is_valid=True)

    def validate_collection_deletion(self, name: str, source: ValidationSource) -> ValidationResult:
        """
        Validate collection deletion request with comprehensive rules enforcement.

        Args:
            name: The collection name to delete
            source: Source of the deletion request

        Returns:
            ValidationResult with validation status and details
        """
        logger.debug(f"Validating collection deletion: {name} from {source.value}")

        # Basic parameter validation
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Collection name must be a non-empty string",
                violation_type="invalid_collection_name"
            )

        name = name.strip()

        # Check if collection exists
        if name not in self._existing_collections:
            return ValidationResult(
                is_valid=False,
                error_message=f"Collection '{name}' does not exist",
                violation_type="collection_not_found"
            )

        # Source-specific validation
        if source == ValidationSource.LLM and self.llm_access_controller:
            # LLM operations must go through LLM access control
            try:
                self.llm_access_controller.validate_llm_collection_access("delete", name)
            except Exception as e:  # LLMAccessControlError
                return ValidationResult(
                    is_valid=False,
                    error_message=str(e),
                    violation_type="llm_access_denied"
                )

        elif source == ValidationSource.CLI:
            # CLI can delete most collections but protect critical system ones
            if self.type_classifier:
                collection_info = self.type_classifier.get_collection_info(name)
                if hasattr(collection_info, 'type') and hasattr(collection_info.type, 'name'):
                    if collection_info.type.name == 'SYSTEM':
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"System collection '{name}' requires explicit admin confirmation",
                            warning_message="Use --force flag if you're certain",
                            violation_type="forbidden_system_deletion"
                        )

        elif source == ValidationSource.MCP_INTERNAL:
            # MCP internal operations cannot delete system collections for safety
            if name.startswith("__"):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"MCP cannot delete system collection '{name}' - use CLI with admin privileges",
                    violation_type="forbidden_system_deletion"
                )

        logger.debug(f"Collection deletion validation passed: {name} from {source.value}")
        return ValidationResult(is_valid=True)

    def validate_collection_write(self, name: str, source: ValidationSource) -> ValidationResult:
        """
        Validate collection write access with comprehensive rules enforcement.

        Args:
            name: The collection name to write to
            source: Source of the write request

        Returns:
            ValidationResult with validation status and details
        """
        logger.debug(f"Validating collection write access: {name} from {source.value}")

        # Basic parameter validation
        if not isinstance(name, str) or not name.strip():
            return ValidationResult(
                is_valid=False,
                error_message="Collection name must be a non-empty string",
                violation_type="invalid_collection_name"
            )

        name = name.strip()

        # Check if collection exists (for write operations)
        if name not in self._existing_collections:
            return ValidationResult(
                is_valid=False,
                error_message=f"Collection '{name}' does not exist",
                violation_type="collection_not_found"
            )

        # Source-specific validation
        if source == ValidationSource.LLM and self.llm_access_controller:
            # LLM operations must go through LLM access control
            try:
                self.llm_access_controller.validate_llm_collection_access("write", name)
            except Exception as e:  # LLMAccessControlError
                return ValidationResult(
                    is_valid=False,
                    error_message=str(e),
                    violation_type="llm_access_denied"
                )

        elif source == ValidationSource.MCP_INTERNAL:
            # MCP internal operations respect read-only boundaries
            # System memory collections are read-only from MCP
            if name.startswith("__") and ("memory" in name.lower()):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"System memory collection '{name}' is read-only from MCP",
                    violation_type="forbidden_system_write"
                )

            # Library collections are read-only from MCP
            if name.startswith("_") and not name.startswith("__"):
                # Check if it's a project collection (_{project_id}) or library collection
                name_without_prefix = name[1:]
                if not re.match(r"^[a-f0-9]{12}$", name_without_prefix):
                    # It's a library collection, not a project collection
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Library collection '{name}' is read-only from MCP - use CLI to modify",
                        violation_type="forbidden_library_write"
                    )

        logger.debug(f"Collection write validation passed: {name} from {source.value}")
        return ValidationResult(is_valid=True)

    def validate_operation(self, operation: OperationType, name: str, source: ValidationSource) -> ValidationResult:
        """
        Unified validation method for any collection operation.

        Args:
            operation: The type of operation being performed
            name: The collection name
            source: Source of the operation request

        Returns:
            ValidationResult with validation status and details
        """
        logger.debug(f"Validating collection operation: {operation.value} {name} from {source.value}")

        if operation == OperationType.CREATE:
            return self.validate_collection_creation(name, source)
        elif operation == OperationType.DELETE:
            return self.validate_collection_deletion(name, source)
        elif operation == OperationType.WRITE:
            return self.validate_collection_write(name, source)
        elif operation == OperationType.READ:
            # Read access is generally allowed for all sources to all existing collections
            if name not in self._existing_collections:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Collection '{name}' does not exist",
                    violation_type="collection_not_found"
                )
            return ValidationResult(is_valid=True)
        elif operation == OperationType.LIST:
            # List operations don't need collection-specific validation
            return ValidationResult(is_valid=True)
        else:
            return ValidationResult(
                is_valid=False,
                error_message=f"Unknown operation type: {operation}",
                violation_type="invalid_operation"
            )

    def enforce_operation(self, operation: OperationType, name: str, source: ValidationSource) -> None:
        """
        Enforce collection operation rules by validating and raising exception on failure.

        Args:
            operation: The type of operation being performed
            name: The collection name
            source: Source of the operation request

        Raises:
            CollectionRulesEnforcementError: If the operation violates rules
        """
        result = self.validate_operation(operation, name, source)
        if not result.is_valid:
            logger.warning(f"Collection operation blocked: {operation.value} {name} from {source.value} - {result.error_message}")
            raise CollectionRulesEnforcementError(result)

        logger.debug(f"Collection operation allowed: {operation.value} {name} from {source.value}")


def create_rules_enforcer(config=None) -> CollectionRulesEnforcer:
    """
    Create a collection rules enforcer instance.

    Args:
        config: Optional configuration object

    Returns:
        Configured CollectionRulesEnforcer instance
    """
    return CollectionRulesEnforcer(config)
