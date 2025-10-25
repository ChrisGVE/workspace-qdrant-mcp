"""
Collection naming system for workspace-qdrant-mcp.

This module implements the reserved collection naming architecture including:
- Memory collection for user preferences and LLM behavioral rules
- Underscore-prefixed library collections (_name pattern)
- Underscore-prefixed project collections (_{project_id} pattern)
- Collection name validation and conflict prevention
- Display name mapping for user-facing interfaces

The naming system enforces these patterns:
1. Memory Collection: 'memory' (reserved)
2. Library Collections: '_{name}' (readonly from MCP, user-defined via CLI)
3. Project Collections: '_{project_id}' (12-char hex hash, auto-created, read/write via MCP)
4. User Collections: '{name}-{suffix}' (user-created collections)

Key features:
- Hard error prevention for naming conflicts
- Display name mapping (library/project collections show without underscore)
- MCP readonly enforcement for library collections
- Comprehensive validation with detailed error messages
"""

import re
from dataclasses import dataclass
from enum import Enum

from loguru import logger

# logger imported from loguru


class CollectionType(Enum):
    """Types of collections in the naming system."""

    MEMORY = "memory"
    SYSTEM_MEMORY = "system_memory"  # System memory collections with '__' prefix
    LIBRARY = "library"
    PROJECT = "project"  # Project collections with '_{project_id}' pattern
    USER = "user"  # User collections with '{name}-{suffix}' pattern
    LEGACY = "legacy"  # For existing collections that don't match patterns


@dataclass
class CollectionNameInfo:
    """Information about a collection name and its classification."""

    name: str  # Actual collection name in Qdrant
    display_name: str  # Name shown to users (without underscore prefix)
    collection_type: CollectionType
    is_readonly_from_mcp: bool  # Whether MCP server can modify this collection
    project_id: str | None = None  # For project collections (12-char hex)
    user_collection_suffix: str | None = None  # For user collections (suffix part)
    library_name: str | None = None  # For library collections (without underscore)
    system_memory_name: str | None = None  # For system memory collections (without __ prefix)


@dataclass
class NamingValidationResult:
    """Result of collection name validation."""

    is_valid: bool
    error_message: str | None = None
    warning_message: str | None = None
    collection_info: CollectionNameInfo | None = None


class CollectionNamingManager:
    """
    Manages the reserved collection naming system.

    This class implements the comprehensive naming architecture including:
    - Reserved name validation and conflict prevention
    - Collection type classification and permissions
    - Display name mapping for user interfaces
    - Integration with existing collection management

    The naming system supports:
    1. Memory collection: 'memory' (global, read/write via MCP and CLI)
    2. Library collections: '_{name}' (user libraries, readonly from MCP)
    3. Project collections: '_{project_id}' (auto-created, read/write via MCP, single collection per project)
    4. User collections: '{name}-{suffix}' (user-created, multi-purpose)
    5. Legacy collections: Existing collections that don't match new patterns
    """

    # Reserved collection names that cannot be used for user collections (excluding 'memory')
    RESERVED_NAMES = {
        "_memory",  # Prevent confusion with memory
        "__memory",  # Prevent confusion with system memory
        "system",  # Reserved for future system use
        "_system",  # Reserved for future system use
        "__system",  # Reserved for future system use
        "admin",  # Reserved for future admin use
        "_admin",  # Reserved for future admin use
        "__admin",  # Reserved for future admin use
    }

    # Valid suffixes for user collections - can be overridden by configuration
    DEFAULT_USER_SUFFIXES = {
        "notes",  # User notes
        "bookmarks",  # User bookmarks
        "snippets",  # Code snippets
        "resources",  # General resources
    }

    # Pattern for valid library names (after underscore) - must be at least 2 chars
    LIBRARY_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$")

    # Pattern for valid project IDs (12-char hex hash)
    PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{12}$")

    # Pattern for user collection names
    USER_COLLECTION_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$")

    def __init__(self, global_collections: list[str] = None, valid_user_suffixes: list[str] = None):
        """
        Initialize the collection naming manager.

        Args:
            global_collections: Legacy global collection names to preserve compatibility
            valid_user_suffixes: Configured user collection suffixes (overrides defaults)
        """
        self.global_collections = set(global_collections or [])
        self.valid_user_suffixes = set(valid_user_suffixes) if valid_user_suffixes else self.DEFAULT_USER_SUFFIXES

    def validate_collection_name(
        self, name: str, intended_type: CollectionType | None = None
    ) -> NamingValidationResult:
        """
        Validate a collection name according to the naming system rules.

        Args:
            name: The proposed collection name
            intended_type: Optional hint about intended collection type

        Returns:
            NamingValidationResult with validation status and details
        """
        # Basic name validation
        if not name or not name.strip():
            return NamingValidationResult(
                is_valid=False, error_message="Collection name cannot be empty"
            )

        # Normalize name (strip whitespace)
        name = name.strip()

        if len(name) > 100:
            return NamingValidationResult(
                is_valid=False,
                error_message="Collection name cannot exceed 100 characters",
            )

        # Check for reserved names (but allow system memory collections like '__memory')
        if name in self.RESERVED_NAMES:
            # Allow system memory collections even if they're in reserved names
            # This covers cases like '__memory' which is both reserved and a valid system collection
            if not (name.startswith("__") and len(name) > 2):
                return NamingValidationResult(
                    is_valid=False, error_message=f"'{name}' is a reserved collection name"
                )

        # Classify the collection type
        collection_info = self._classify_collection_name(name)

        # Validate based on type
        if collection_info.collection_type == CollectionType.MEMORY:
            if name != "memory":
                return NamingValidationResult(
                    is_valid=False,
                    error_message="Only 'memory' is allowed as a memory collection name",
                )

        elif collection_info.collection_type == CollectionType.SYSTEM_MEMORY:
            # System memory classification handles validation internally
            if not collection_info.system_memory_name:  # Invalid system memory name
                system_memory_name = name[2:]  # Remove __ prefix
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"System memory name '{system_memory_name}' must start with a letter, "
                    f"contain only lowercase letters, numbers, hyphens, and underscores, "
                    f"and end with a letter or number (or be a single letter)",
                )

        elif collection_info.collection_type == CollectionType.LIBRARY:
            # Library classification handles validation internally
            if not collection_info.library_name:  # Invalid library name
                library_name = name[1:]  # Remove single underscore prefix
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"Library name '{library_name}' must start with a letter, "
                    f"contain only lowercase letters, numbers, hyphens, and underscores, "
                    f"and end with a letter or number (or be a single letter)",
                )

        elif collection_info.collection_type == CollectionType.PROJECT:
            # Project classification handles validation internally
            if not collection_info.project_id:  # Invalid project ID
                project_id_part = name[1:]  # Remove underscore prefix
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"Project ID '{project_id_part}' must be exactly 12 hexadecimal characters "
                    f"(a-f, 0-9)",
                )

        elif collection_info.collection_type == CollectionType.USER:
            # User collection validation
            if not collection_info.user_collection_suffix:
                # Check if it's a user pattern but invalid suffix
                parts = name.split("-")
                if len(parts) >= 2:
                    potential_suffix = parts[-1]
                    return NamingValidationResult(
                        is_valid=False,
                        error_message=f"Invalid user collection suffix '{potential_suffix}'. "
                        f"Valid suffixes: {', '.join(self.valid_user_suffixes)}",
                    )
                else:
                    return NamingValidationResult(
                        is_valid=False,
                        error_message=f"Invalid user collection format: '{name}'. "
                        f"Expected: 'name-suffix'",
                    )

        elif collection_info.collection_type == CollectionType.LEGACY:
            # For legacy collections, check if they look like invalid patterns
            if name.startswith("__"):
                # This is an invalid system memory name
                system_memory_name = name[2:]  # Remove __ prefix
                if not system_memory_name:  # Just double underscore
                    return NamingValidationResult(
                        is_valid=False,
                        error_message="System memory name cannot be empty after '__' prefix",
                    )
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"System memory name '{system_memory_name}' must start with a letter, "
                    f"contain only lowercase letters, numbers, hyphens, and underscores, "
                    f"and end with a letter or number (or be a single letter)",
                )
            elif name.startswith("_"):
                # Check if it's a project ID pattern or invalid library name
                name_without_underscore = name[1:]
                if self.PROJECT_ID_PATTERN.match(name_without_underscore):
                    # Valid project ID pattern but not classified - this is actually valid
                    return NamingValidationResult(is_valid=True, collection_info=collection_info)
                else:
                    # Invalid library name
                    return NamingValidationResult(
                        is_valid=False,
                        error_message=f"Library name '{name_without_underscore}' must start with a letter, "
                        f"contain only lowercase letters, numbers, hyphens, and underscores, "
                        f"and end with a letter or number (or be a single letter)",
                    )

            # Check for malformed patterns
            if name.endswith("-"):
                return NamingValidationResult(
                    is_valid=False,
                    error_message="Collection names cannot end with a hyphen",
                )

            if name.startswith("-"):
                return NamingValidationResult(
                    is_valid=False,
                    error_message="Collection names cannot start with a hyphen",
                )

            # Check if it looks like invalid user collection
            parts = name.split("-")
            if len(parts) >= 2:
                potential_suffix = parts[-1]
                if potential_suffix not in self.valid_user_suffixes:
                    # Only flag as error if it looks like it's trying to be a user collection
                    suspicious_suffixes = {
                        "code",
                        "test",
                        "tests",
                        "build",
                        "dist",
                        "lib",
                        "src",
                        "invalid",
                        "temp",
                        "tmp",
                        "cache",
                        "config",
                        "data",
                        "docs",  # Removed from project suffixes
                        "scratchbook",  # Removed from project suffixes
                    }
                    if potential_suffix in suspicious_suffixes:
                        return NamingValidationResult(
                            is_valid=False,
                            error_message=f"Invalid user collection suffix '{potential_suffix}'. "
                            f"Valid suffixes: {', '.join(self.valid_user_suffixes)}",
                        )

        # Check type consistency if intended type was provided
        if intended_type and intended_type != collection_info.collection_type:
            return NamingValidationResult(
                is_valid=False,
                error_message=f"Collection name '{name}' suggests type {collection_info.collection_type.value} "
                f"but intended type is {intended_type.value}",
            )

        return NamingValidationResult(is_valid=True, collection_info=collection_info)

    def check_naming_conflicts(
        self, proposed_name: str, existing_collections: list[str]
    ) -> NamingValidationResult:
        """
        Check for naming conflicts with existing collections.

        This prevents conflicts like:
        - Creating 'library' when '_library' exists
        - Creating '_library' when 'library' exists
        - Creating any collection with reserved names

        Args:
            proposed_name: The name being proposed for creation
            existing_collections: List of existing collection names

        Returns:
            NamingValidationResult indicating if conflicts exist
        """
        # First validate the proposed name
        validation = self.validate_collection_name(proposed_name)
        if not validation.is_valid:
            return validation

        existing_set = set(existing_collections)

        # Check direct conflicts
        if proposed_name in existing_set:
            return NamingValidationResult(
                is_valid=False,
                error_message=f"Collection '{proposed_name}' already exists",
            )

        # Check library/display name conflicts
        if proposed_name.startswith("_") and not proposed_name.startswith("__"):
            # Proposing a library or project collection, check if display name exists
            display_name = proposed_name[1:]
            if display_name in existing_set:
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"Cannot create collection '{proposed_name}' because "
                    f"collection '{display_name}' already exists. This would create "
                    f"a naming conflict.",
                )
        else:
            # Proposing a non-library/non-project collection, check if library/project version exists
            prefixed_name = f"_{proposed_name}"
            if prefixed_name in existing_set:
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"Cannot create collection '{proposed_name}' because "
                    f"collection '{prefixed_name}' already exists. This would "
                    f"create a naming conflict.",
                )

        return NamingValidationResult(
            is_valid=True, collection_info=validation.collection_info
        )

    def get_collection_info(self, name: str) -> CollectionNameInfo:
        """
        Get detailed information about a collection name.

        Args:
            name: The collection name to analyze

        Returns:
            CollectionNameInfo with classification and metadata
        """
        return self._classify_collection_name(name)

    def get_display_name(self, collection_name: str) -> str:
        """
        Get the user-facing display name for a collection.

        For library and project collections, this removes the underscore prefix.
        For other collections, this returns the name unchanged.

        Args:
            collection_name: The actual collection name in Qdrant

        Returns:
            The display name to show to users
        """
        info = self._classify_collection_name(collection_name)
        return info.display_name

    def get_actual_name(
        self, display_name: str, collection_type: CollectionType
    ) -> str:
        """
        Get the actual collection name in Qdrant from a display name.

        For library and project collections, this adds the underscore prefix.
        For system memory collections, this adds the double underscore prefix.
        For other collections, this returns the name unchanged.

        Args:
            display_name: The name shown to users
            collection_type: The type of collection

        Returns:
            The actual collection name to use in Qdrant
        """
        if collection_type == CollectionType.LIBRARY:
            return f"_{display_name}"
        elif collection_type == CollectionType.PROJECT:
            return f"_{display_name}"
        elif collection_type == CollectionType.SYSTEM_MEMORY:
            return f"__{display_name}"
        return display_name

    def is_mcp_readonly(self, collection_name: str) -> bool:
        """
        Check if a collection is readonly from the MCP server perspective.

        Library collections and system memory collections are readonly from MCP -
        they can only be modified via the CLI/Rust engine.

        Args:
            collection_name: The collection name to check

        Returns:
            True if the collection is readonly from MCP, False otherwise
        """
        info = self._classify_collection_name(collection_name)
        return info.is_readonly_from_mcp

    def is_system_memory_collection(self, collection_name: str) -> bool:
        """
        Check if a collection is a system memory collection (double underscore prefix).

        Args:
            collection_name: The collection name to check

        Returns:
            True if the collection is a system memory collection, False otherwise
        """
        info = self._classify_collection_name(collection_name)
        return info.collection_type == CollectionType.SYSTEM_MEMORY

    def filter_workspace_collections(self, all_collections: list[str]) -> list[str]:
        """
        Filter a list of collections to include only workspace collections.

        This excludes external collections like memexd daemon collections
        (ending with '-code') while including all workspace collections.

        Args:
            all_collections: List of all collection names from Qdrant

        Returns:
            List of workspace collection names
        """
        workspace_collections = []

        for collection in all_collections:
            # Exclude memexd daemon collections
            if collection.endswith("-code"):
                continue

            info = self._classify_collection_name(collection)

            # Include all workspace collection types
            if info.collection_type in [
                CollectionType.MEMORY,
                CollectionType.SYSTEM_MEMORY,
                CollectionType.LIBRARY,
                CollectionType.PROJECT,
                CollectionType.USER,
            ]:
                workspace_collections.append(collection)
            elif info.collection_type == CollectionType.LEGACY:
                # Include legacy global collections
                if collection in self.global_collections:
                    workspace_collections.append(collection)

        return sorted(workspace_collections)

    def generate_project_collection_name(self, project_id: str) -> str:
        """
        Generate collection name for a project based on project ID.

        Args:
            project_id: The 12-character hex project ID

        Returns:
            Single collection name for the project (_{project_id})
        """
        # Validate project ID format
        if not self.PROJECT_ID_PATTERN.match(project_id):
            raise ValueError(f"Invalid project ID format: '{project_id}'. Must be 12 hexadecimal characters.")

        return f"_{project_id}"

    def _classify_collection_name(self, name: str) -> CollectionNameInfo:
        """
        Classify a collection name and extract metadata.

        Args:
            name: The collection name to classify

        Returns:
            CollectionNameInfo with type classification and metadata
        """
        # Memory collection
        if name == "memory":
            return CollectionNameInfo(
                name=name,
                display_name=name,
                collection_type=CollectionType.MEMORY,
                is_readonly_from_mcp=False,
            )

        # System memory collections (double underscore prefix)
        if name.startswith("__") and len(name) > 2:
            system_memory_name = name[2:]  # Remove __ prefix
            # Validate system memory name format (similar to library validation)
            if self.LIBRARY_NAME_PATTERN.match(system_memory_name):
                return CollectionNameInfo(
                    name=name,
                    display_name=system_memory_name,
                    collection_type=CollectionType.SYSTEM_MEMORY,
                    is_readonly_from_mcp=True,  # System memory is readonly from MCP
                    system_memory_name=system_memory_name,
                )

        # Project collections (_{project_id} where project_id is 12-char hex)
        if name.startswith("_") and not name.startswith("__") and len(name) == 13:  # _ + 12 chars
            potential_project_id = name[1:]
            if self.PROJECT_ID_PATTERN.match(potential_project_id):
                return CollectionNameInfo(
                    name=name,
                    display_name=potential_project_id,
                    collection_type=CollectionType.PROJECT,
                    is_readonly_from_mcp=False,
                    project_id=potential_project_id,
                )

        # Library collections (single underscore-prefixed, not double, not project ID)
        if name.startswith("_") and not name.startswith("__") and len(name) > 1:
            library_name = name[1:]
            # Validate library name format (not a project ID)
            if self.LIBRARY_NAME_PATTERN.match(library_name) and not self.PROJECT_ID_PATTERN.match(library_name):
                return CollectionNameInfo(
                    name=name,
                    display_name=library_name,
                    collection_type=CollectionType.LIBRARY,
                    is_readonly_from_mcp=True,
                    library_name=library_name,
                )

        # User collections (name-suffix)
        parts = name.split("-")
        if len(parts) >= 2:
            potential_suffix = parts[-1]
            if potential_suffix in self.valid_user_suffixes:
                user_name = "-".join(parts[:-1])
                # Validate user name format
                if self.USER_COLLECTION_PATTERN.match(user_name):
                    return CollectionNameInfo(
                        name=name,
                        display_name=name,
                        collection_type=CollectionType.USER,
                        is_readonly_from_mcp=False,
                        user_collection_suffix=potential_suffix,
                    )

        # Legacy/unknown collections
        return CollectionNameInfo(
            name=name,
            display_name=name,
            collection_type=CollectionType.LEGACY,
            is_readonly_from_mcp=False,
        )


class CollectionNameError(Exception):
    """Exception raised when a collection name is invalid."""

    pass


class CollectionPermissionError(Exception):
    """Exception raised when attempting prohibited collection operations."""

    pass


def validate_collection_name(name: str, allow_library: bool = False) -> None:
    """
    Validate a collection name and raise exception if invalid.

    This is a convenience function that creates a naming manager and
    validates the name, raising CollectionNameError if invalid.

    Args:
        name: The collection name to validate
        allow_library: Whether to allow library collection names (starting with '_')

    Raises:
        CollectionNameError: If the collection name is invalid
    """
    manager = CollectionNamingManager()

    # Determine intended type based on name pattern
    intended_type = None
    if name == "memory":
        intended_type = CollectionType.MEMORY
    elif name.startswith("__"):
        intended_type = CollectionType.SYSTEM_MEMORY
    elif name.startswith("_"):
        if not allow_library:
            raise CollectionNameError(
                f"Library/project collection names (starting with '_') not allowed here: {name}"
            )
        # Could be library or project - let validation determine
        name_without_prefix = name[1:]
        if len(name_without_prefix) == 12 and re.match(r"^[a-f0-9]{12}$", name_without_prefix):
            intended_type = CollectionType.PROJECT
        else:
            intended_type = CollectionType.LIBRARY
    else:
        # Could be user collection or legacy
        if "-" in name:
            intended_type = CollectionType.USER
        else:
            intended_type = CollectionType.LEGACY

    result = manager.validate_collection_name(name, intended_type)

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
    Build a project collection name from tenant ID.

    Args:
        project_id: The tenant ID (either git remote format like 'github_com_user_repo'
                    or path hash format like 'path_abc123def456789a')

    Returns:
        Formatted collection name (_{project_id})
    """
    # Validate tenant ID format
    # Allow git remote format (alphanumeric + underscores) or path hash format (path_ + 16 hex chars)
    if not re.match(r"^(?:path_[a-f0-9]{16}|[a-z]+_[a-z]+_[a-z0-9_]+)$", project_id):
        raise ValueError(
            f"Invalid tenant ID format: '{project_id}'. "
            "Must be either git remote format (e.g., 'github_com_user_repo') "
            "or path hash format (e.g., 'path_abc123def456789a')"
        )

    return f"_{project_id}"


def build_user_collection_name(name: str, suffix: str) -> str:
    """
    Build a user collection name from name and suffix.

    Args:
        name: The collection name
        suffix: The collection type suffix

    Returns:
        Formatted collection name
    """
    # Use the normalize function for consistency
    name_clean = normalize_collection_name_component(name)
    suffix_clean = normalize_collection_name_component(suffix)
    return f"{name_clean}-{suffix_clean}"


def build_system_memory_collection_name(memory_name: str) -> str:
    """
    Build a system memory collection name with proper prefix.

    Args:
        memory_name: The memory collection name

    Returns:
        System memory collection name with __ prefix
    """
    normalized = normalize_collection_name_component(memory_name)
    return f"__{normalized}"


def create_naming_manager(
    global_collections: list[str] = None,
) -> CollectionNamingManager:
    """
    Create a collection naming manager instance.

    Args:
        global_collections: Legacy global collection names for compatibility

    Returns:
        Configured CollectionNamingManager instance
    """
    return CollectionNamingManager(global_collections)


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
    Comprehensive collection management rules enforcer for Task 181.

    This class provides unified validation for all collection operations across
    MCP tools, preventing rule bypass and ensuring security boundaries are maintained.
    It integrates with existing LLM access control and collection type systems.

    Key Features:
    - Source-aware validation (LLM vs CLI vs MCP internal)
    - Integration with LLM access control from Task 173
    - Prevention of rule bypass through parameter manipulation
    - System collection protection enforcement
    - Clear error messages with suggested alternatives
    - Comprehensive logging and audit trail

    Security Boundaries:
    - LLM cannot create/delete system collections (__*)
    - LLM cannot create/delete library collections (_*)
    - LLM cannot delete memory collections (*-memory, __*memory*)
    - System memory collections are read-only from MCP
    - All operations go through validation - no bypass paths
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
