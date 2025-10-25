"""
Multi-tenant metadata schema for workspace-qdrant-mcp.

This module defines a comprehensive metadata schema for supporting multi-tenant
project isolation with existing collection management systems. The schema enables
efficient metadata-based filtering across project boundaries while preserving
existing naming conventions and access patterns.

Key Features:
    - Project isolation via project_id metadata field
    - Collection type identification (memory, library, project, global)
    - Reserved naming pattern validation for system and library collections
    - Support for existing suffix-based collection naming
    - Performance-optimized metadata fields with appropriate indexing
    - Comprehensive validation and constraint enforcement

Collection Categories:
    - SYSTEM: "__" prefix collections (CLI-writable, LLM-readable)
    - LIBRARY: "_" prefix collections (CLI-managed, MCP-readonly)
    - PROJECT: "{project}-{suffix}" collections (user-created, project-scoped)
    - GLOBAL: Predefined global collections (system-wide, always available)

Multi-tenant Architecture:
    - Single shared collections per type (notes, docs, scratchbook, etc.)
    - Project isolation through metadata filtering
    - Efficient resource utilization and centralized management
    - Maintains workspace boundaries through tenant_namespace metadata

Example:
    ```python
    from metadata_schema import MultiTenantMetadataSchema, CollectionCategory

    # Create metadata for project collection
    metadata = MultiTenantMetadataSchema.create_for_project(
        project_name="workspace-qdrant-mcp",
        collection_type="docs",
        created_by="user"
    )

    # Create metadata for system collection
    system_metadata = MultiTenantMetadataSchema.create_for_system(
        collection_name="__user_preferences",
        collection_type="memory_collection"
    )

    # Validate and convert to Qdrant format
    qdrant_metadata = metadata.to_qdrant_payload()
    ```
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from loguru import logger

# Import existing collection types for integration
try:
    from .collection_naming import CollectionNamingManager
    from .collection_types import CollectionType, CollectionTypeClassifier
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


# Schema version for future migrations
METADATA_SCHEMA_VERSION = "1.0.0"

# Maximum lengths for string fields (performance optimization)
MAX_PROJECT_NAME_LENGTH = 128
MAX_COLLECTION_TYPE_LENGTH = 64
MAX_TENANT_NAMESPACE_LENGTH = 192
MAX_CREATED_BY_LENGTH = 64
MAX_ACCESS_LEVEL_LENGTH = 32
MAX_BRANCH_LENGTH = 256


class CollectionCategory(Enum):
    """Collection category classification for metadata-based filtering."""

    SYSTEM = "system"           # __ prefix collections
    LIBRARY = "library"         # _ prefix collections
    PROJECT = "project"         # {project}-{suffix} collections
    GLOBAL = "global"          # Predefined global collections
    LEGACY = "legacy"          # Collections without metadata
    UNKNOWN = "unknown"        # Unclassified collections


class WorkspaceScope(Enum):
    """Workspace scope for collection accessibility."""

    PROJECT = "project"        # Project-specific access
    SHARED = "shared"          # Shared across projects
    GLOBAL = "global"          # Global system-wide access
    LIBRARY = "library"        # Library collection scope


class AccessLevel(Enum):
    """Access control levels for collections."""

    PUBLIC = "public"          # Publicly accessible
    PRIVATE = "private"        # Private to creator/project
    SHARED = "shared"          # Shared with team/organization
    READONLY = "readonly"      # Read-only access


@dataclass
class MultiTenantMetadataSchema:
    """
    Comprehensive metadata schema for multi-tenant workspace collections.

    This class defines the complete metadata structure needed for project isolation,
    collection classification, access control, and backward compatibility. All fields
    are designed for efficient indexing and filtering in Qdrant.

    Attributes:
        # Core tenant isolation (required for project isolation)
        project_id: Stable 12-character identifier for project filtering
        project_name: Human-readable project name
        tenant_namespace: Hierarchical namespace for tenant isolation
        branch: Git branch name, defaults to "main"

        # Collection classification (required for type-based operations)
        collection_type: Workspace collection type (docs, notes, memory, etc.)
        collection_category: System/library/project/global classification
        workspace_scope: Project/shared/global/library accessibility scope

        # Code Analysis Metadata (optional)
        symbols_defined: List of symbols defined in file
        symbols_used: List of symbols imported/used
        imports: List of import statements
        exports: List of export statements

        # Reserved naming and compatibility (for migration support)
        naming_pattern: Original naming convention used
        is_reserved_name: Flag for system/library collections
        original_name_pattern: Stores original collection name pattern

        # Access control and permissions (for security)
        access_level: Public/private/shared/readonly access level
        mcp_readonly: Flag for MCP server read-only access
        cli_writable: Flag for CLI write access
        created_by: Origin of collection creation

        # Migration tracking
        migration_source: How collection was created/migrated
        legacy_collection_name: Original name for reference
        compatibility_version: Schema version for future migrations

        # Temporal and organizational metadata
        created_at: ISO timestamp of creation
        updated_at: ISO timestamp of last update
        version: Metadata version number
        tags: Organizational tags
        category: General category classification
        priority: Priority level (1-5 scale)
    """

    # === Core Tenant Isolation Fields (Required) ===
    project_id: str                              # 12-char hash for filtering
    project_name: str                            # Human-readable project name
    tenant_namespace: str                        # {project}.{collection_type}

    # === Collection Classification Fields (Required) ===
    collection_type: str                         # docs, notes, memory, etc.
    collection_category: CollectionCategory     # system/library/project/global
    workspace_scope: WorkspaceScope             # project/shared/global/library
    branch: str = "main"                         # Git branch name, defaults to "main"

    # === Code Analysis Metadata (Optional) ===
    symbols_defined: list[str] = field(default_factory=list)  # Symbols defined in file
    symbols_used: list[str] = field(default_factory=list)     # Symbols imported/used
    imports: list[str] = field(default_factory=list)          # Import statements
    exports: list[str] = field(default_factory=list)          # Export statements

    # === Reserved Naming and Compatibility ===
    naming_pattern: str = "metadata_based"      # metadata_based, system_prefix, library_prefix, project_pattern
    is_reserved_name: bool = False               # True for system/library collections
    original_name_pattern: str | None = None # Original naming convention

    # === Access Control and Permissions ===
    access_level: AccessLevel = AccessLevel.PRIVATE
    mcp_readonly: bool = False                   # True for library collections
    cli_writable: bool = True                    # False only for some special cases
    created_by: str = "system"                   # system, user, cli, migration

    # === Migration Tracking Fields ===
    migration_source: str = "metadata_based"    # metadata_based, suffix_based, manual, auto_create
    legacy_collection_name: str | None = None # Original collection name
    compatibility_version: str = METADATA_SCHEMA_VERSION

    # === Temporal and Organizational Metadata ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: int = 1
    tags: list[str] = field(default_factory=list)
    category: str = "general"
    priority: int = 3                            # 1=lowest, 5=highest

    def __post_init__(self):
        """Validate and normalize metadata fields after initialization."""
        # Apply default branch if not provided or empty
        if not self.branch or not self.branch.strip():
            self.branch = "main"

        # Validate required fields
        self._validate_required_fields()

        # Normalize string fields
        self._normalize_string_fields()

        # Validate field constraints
        self._validate_field_constraints()

        # Update timestamp
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def _validate_required_fields(self):
        """Validate that all required fields are present and non-empty."""
        required_fields = [
            'project_id', 'project_name', 'tenant_namespace', 'branch',
            'collection_type', 'collection_category', 'workspace_scope'
        ]

        for field_name in required_fields:
            value = getattr(self, field_name)
            if not value or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"Required field '{field_name}' is missing or empty")

    def _normalize_string_fields(self):
        """Normalize string fields for consistency."""
        # Normalize project_name and collection_type to lowercase with underscores
        self.project_name = self.project_name.strip().lower().replace('-', '_')
        self.collection_type = self.collection_type.strip().lower().replace('-', '_')

        # Normalize tenant_namespace
        self.tenant_namespace = f"{self.project_name}.{self.collection_type}"

        # Normalize created_by
        self.created_by = self.created_by.strip().lower()

        # Normalize category
        self.category = self.category.strip().lower()

        # Normalize branch (trim whitespace but preserve case)
        self.branch = self.branch.strip()

    def _validate_field_constraints(self):
        """Validate field length and format constraints."""
        # Length constraints
        if len(self.project_name) > MAX_PROJECT_NAME_LENGTH:
            raise ValueError(f"project_name exceeds maximum length of {MAX_PROJECT_NAME_LENGTH}")

        if len(self.collection_type) > MAX_COLLECTION_TYPE_LENGTH:
            raise ValueError(f"collection_type exceeds maximum length of {MAX_COLLECTION_TYPE_LENGTH}")

        if len(self.tenant_namespace) > MAX_TENANT_NAMESPACE_LENGTH:
            raise ValueError(f"tenant_namespace exceeds maximum length of {MAX_TENANT_NAMESPACE_LENGTH}")

        if len(self.created_by) > MAX_CREATED_BY_LENGTH:
            raise ValueError(f"created_by exceeds maximum length of {MAX_CREATED_BY_LENGTH}")

        if len(self.branch) > MAX_BRANCH_LENGTH:
            raise ValueError(f"branch exceeds maximum length of {MAX_BRANCH_LENGTH}")

        # Project ID format validation (12 alphanumeric characters)
        if not re.match(r'^[a-f0-9]{12}$', self.project_id):
            raise ValueError("project_id must be exactly 12 hexadecimal characters")

        # Priority range validation
        if not 1 <= self.priority <= 5:
            raise ValueError("priority must be between 1 and 5")

        # Symbol fields validation (must be lists of strings if present)
        for field_name in ['symbols_defined', 'symbols_used', 'imports', 'exports']:
            value = getattr(self, field_name)
            if not isinstance(value, list):
                raise ValueError(f"{field_name} must be a list")
            if any(not isinstance(item, str) for item in value):
                raise ValueError(f"All items in {field_name} must be strings")

    def to_qdrant_payload(self) -> dict[str, Any]:
        """
        Convert metadata schema to Qdrant payload format.

        Returns:
            Dict optimized for Qdrant metadata storage and indexing
        """
        return {
            # Core tenant isolation (indexed)
            "project_id": self.project_id,
            "project_name": self.project_name,
            "tenant_namespace": self.tenant_namespace,
            "branch": self.branch,

            # Collection classification (indexed)
            "collection_type": self.collection_type,
            "collection_category": self.collection_category.value,
            "workspace_scope": self.workspace_scope.value,

            # Code analysis metadata
            "symbols_defined": self.symbols_defined,
            "symbols_used": self.symbols_used,
            "imports": self.imports,
            "exports": self.exports,

            # Reserved naming and compatibility
            "naming_pattern": self.naming_pattern,
            "is_reserved_name": self.is_reserved_name,
            "original_name_pattern": self.original_name_pattern,

            # Access control (indexed)
            "access_level": self.access_level.value,
            "mcp_readonly": self.mcp_readonly,
            "cli_writable": self.cli_writable,
            "created_by": self.created_by,

            # Migration tracking
            "migration_source": self.migration_source,
            "legacy_collection_name": self.legacy_collection_name,
            "compatibility_version": self.compatibility_version,

            # Temporal and organizational
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "tags": self.tags,
            "category": self.category,
            "priority": self.priority
        }

    @classmethod
    def from_qdrant_payload(cls, payload: dict[str, Any]) -> "MultiTenantMetadataSchema":
        """
        Create metadata schema from Qdrant payload.

        Args:
            payload: Qdrant metadata payload dictionary

        Returns:
            MultiTenantMetadataSchema instance
        """
        # Convert enum values back to enum instances
        collection_category = CollectionCategory(payload.get("collection_category", "unknown"))
        workspace_scope = WorkspaceScope(payload.get("workspace_scope", "project"))
        access_level = AccessLevel(payload.get("access_level", "private"))

        return cls(
            # Core tenant isolation
            project_id=payload["project_id"],
            project_name=payload["project_name"],
            tenant_namespace=payload["tenant_namespace"],
            branch=payload.get("branch", "main"),  # Default to "main" for backwards compatibility

            # Collection classification
            collection_type=payload["collection_type"],
            collection_category=collection_category,
            workspace_scope=workspace_scope,

            # Code analysis metadata (optional, default to empty lists)
            symbols_defined=payload.get("symbols_defined", []),
            symbols_used=payload.get("symbols_used", []),
            imports=payload.get("imports", []),
            exports=payload.get("exports", []),

            # Reserved naming and compatibility
            naming_pattern=payload.get("naming_pattern", "metadata_based"),
            is_reserved_name=payload.get("is_reserved_name", False),
            original_name_pattern=payload.get("original_name_pattern"),

            # Access control
            access_level=access_level,
            mcp_readonly=payload.get("mcp_readonly", False),
            cli_writable=payload.get("cli_writable", True),
            created_by=payload.get("created_by", "system"),

            # Migration tracking
            migration_source=payload.get("migration_source", "metadata_based"),
            legacy_collection_name=payload.get("legacy_collection_name"),
            compatibility_version=payload.get("compatibility_version", METADATA_SCHEMA_VERSION),

            # Temporal and organizational
            created_at=payload.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=payload.get("updated_at", datetime.now(timezone.utc).isoformat()),
            version=payload.get("version", 1),
            tags=payload.get("tags", []),
            category=payload.get("category", "general"),
            priority=payload.get("priority", 3)
        )

    @classmethod
    def create_for_project(
        cls,
        project_name: str,
        collection_type: str,
        branch: str = "main",
        created_by: str = "user",
        access_level: AccessLevel = AccessLevel.PRIVATE,
        tags: list[str] | None = None,
        category: str = "general",
        priority: int = 3,
        symbols_defined: list[str] | None = None,
        symbols_used: list[str] | None = None,
        imports: list[str] | None = None,
        exports: list[str] | None = None
    ) -> "MultiTenantMetadataSchema":
        """
        Factory method for creating project collection metadata.

        Args:
            project_name: Name of the project
            collection_type: Type of collection (docs, notes, etc.)
            branch: Git branch name, defaults to "main"
            created_by: Who created the collection
            access_level: Access control level
            tags: Organizational tags
            category: General category
            priority: Priority level (1-5)
            symbols_defined: Symbols defined in file
            symbols_used: Symbols imported/used
            imports: Import statements
            exports: Export statements

        Returns:
            MultiTenantMetadataSchema configured for project collection
        """
        # Generate stable project ID
        project_id = cls._generate_project_id(project_name)

        return cls(
            # Core tenant isolation
            project_id=project_id,
            project_name=project_name,
            tenant_namespace=f"{project_name}.{collection_type}",
            branch=branch,

            # Collection classification
            collection_type=collection_type,
            collection_category=CollectionCategory.PROJECT,
            workspace_scope=WorkspaceScope.PROJECT,

            # Code analysis metadata
            symbols_defined=symbols_defined or [],
            symbols_used=symbols_used or [],
            imports=imports or [],
            exports=exports or [],

            # Reserved naming
            naming_pattern="project_pattern",
            is_reserved_name=False,

            # Access control
            access_level=access_level,
            mcp_readonly=False,
            cli_writable=True,
            created_by=created_by,

            # Migration tracking
            migration_source="metadata_based",

            # Organizational
            tags=tags or [],
            category=category,
            priority=priority
        )

    @classmethod
    def create_for_system(
        cls,
        collection_name: str,
        collection_type: str = "memory_collection",
        branch: str = "main",
        created_by: str = "system"
    ) -> "MultiTenantMetadataSchema":
        """
        Factory method for creating system collection metadata.

        Args:
            collection_name: Full system collection name (with __ prefix)
            collection_type: Type of collection
            branch: Git branch name, defaults to "main"
            created_by: Who created the collection

        Returns:
            MultiTenantMetadataSchema configured for system collection
        """
        if not collection_name.startswith("__"):
            raise ValueError("System collection names must start with '__'")

        # Extract base name for project context
        collection_name[2:]  # Remove __ prefix
        project_id = cls._generate_project_id("system")

        return cls(
            # Core tenant isolation
            project_id=project_id,
            project_name="system",
            tenant_namespace=f"system.{collection_type}",
            branch=branch,

            # Collection classification
            collection_type=collection_type,
            collection_category=CollectionCategory.SYSTEM,
            workspace_scope=WorkspaceScope.GLOBAL,

            # Reserved naming
            naming_pattern="system_prefix",
            is_reserved_name=True,
            original_name_pattern=collection_name,

            # Access control
            access_level=AccessLevel.PRIVATE,
            mcp_readonly=False,  # CLI can write, but MCP typically cannot
            cli_writable=True,
            created_by=created_by,

            # Migration tracking
            migration_source="metadata_based",
            legacy_collection_name=collection_name,

            # Organizational
            category="system",
            priority=4
        )

    @classmethod
    def create_for_library(
        cls,
        collection_name: str,
        collection_type: str = "code_collection",
        branch: str = "main",
        created_by: str = "cli"
    ) -> "MultiTenantMetadataSchema":
        """
        Factory method for creating library collection metadata.

        Args:
            collection_name: Full library collection name (with _ prefix)
            collection_type: Type of collection
            branch: Git branch name, defaults to "main"
            created_by: Who created the collection

        Returns:
            MultiTenantMetadataSchema configured for library collection
        """
        if not collection_name.startswith("_") or collection_name.startswith("__"):
            raise ValueError("Library collection names must start with '_' but not '__'")

        # Extract base name for context
        collection_name[1:]  # Remove _ prefix
        project_id = cls._generate_project_id("library")

        return cls(
            # Core tenant isolation
            project_id=project_id,
            project_name="library",
            tenant_namespace=f"library.{collection_type}",
            branch=branch,

            # Collection classification
            collection_type=collection_type,
            collection_category=CollectionCategory.LIBRARY,
            workspace_scope=WorkspaceScope.LIBRARY,

            # Reserved naming
            naming_pattern="library_prefix",
            is_reserved_name=True,
            original_name_pattern=collection_name,

            # Access control
            access_level=AccessLevel.SHARED,
            mcp_readonly=True,   # Library collections are read-only from MCP
            cli_writable=True,   # CLI can write
            created_by=created_by,

            # Migration tracking
            migration_source="metadata_based",
            legacy_collection_name=collection_name,

            # Organizational
            category="library",
            priority=4
        )

    @classmethod
    def create_for_global(
        cls,
        collection_name: str,
        collection_type: str = "global",
        branch: str = "main",
        created_by: str = "system"
    ) -> "MultiTenantMetadataSchema":
        """
        Factory method for creating global collection metadata.

        Args:
            collection_name: Global collection name
            collection_type: Type of collection
            branch: Git branch name, defaults to "main"
            created_by: Who created the collection

        Returns:
            MultiTenantMetadataSchema configured for global collection
        """
        # Predefined global collections
        valid_global_collections = {
            "algorithms", "codebase", "context", "documents",
            "knowledge", "memory", "projects", "workspace"
        }

        if collection_name not in valid_global_collections:
            logger.warning(f"Collection '{collection_name}' is not a predefined global collection")

        project_id = cls._generate_project_id("global")

        return cls(
            # Core tenant isolation
            project_id=project_id,
            project_name="global",
            tenant_namespace=f"global.{collection_type}",
            branch=branch,

            # Collection classification
            collection_type=collection_type,
            collection_category=CollectionCategory.GLOBAL,
            workspace_scope=WorkspaceScope.GLOBAL,

            # Reserved naming
            naming_pattern="global_collection",
            is_reserved_name=False,

            # Access control
            access_level=AccessLevel.PUBLIC,
            mcp_readonly=False,
            cli_writable=True,
            created_by=created_by,

            # Migration tracking
            migration_source="metadata_based",

            # Organizational
            category="global",
            priority=5
        )

    @staticmethod
    def _generate_project_id(project_name: str) -> str:
        """
        Generate stable 12-character project ID from project name.

        Args:
            project_name: Project name to hash

        Returns:
            12-character hexadecimal project ID
        """
        return hashlib.sha256(project_name.encode()).hexdigest()[:12]

    def get_indexed_fields(self) -> list[str]:
        """
        Get list of metadata fields that should be indexed for performance.

        Returns:
            List of field names that need Qdrant payload indexes
        """
        return [
            "project_id",
            "project_name",
            "tenant_namespace",
            "branch",
            "collection_type",
            "collection_category",
            "workspace_scope",
            "access_level",
            "mcp_readonly",
            "cli_writable",
            "created_by",
            "is_reserved_name",
            "naming_pattern"
        ]

    def matches_project_filter(self, target_project_id: str) -> bool:
        """
        Check if this metadata matches a project filter.

        Args:
            target_project_id: Project ID to match against

        Returns:
            True if metadata matches project filter
        """
        return self.project_id == target_project_id

    def matches_collection_type_filter(self, target_types: list[str]) -> bool:
        """
        Check if this metadata matches collection type filters.

        Args:
            target_types: List of collection types to match

        Returns:
            True if metadata matches any of the target types
        """
        return self.collection_type in target_types

    def is_accessible_by_mcp(self) -> bool:
        """
        Check if collection is accessible by MCP server.

        Returns:
            True if MCP can access this collection
        """
        # MCP can read all collections, but write access depends on mcp_readonly flag
        return True

    def is_writable_by_mcp(self) -> bool:
        """
        Check if collection is writable by MCP server.

        Returns:
            True if MCP can write to this collection
        """
        return not self.mcp_readonly

    def is_globally_searchable(self) -> bool:
        """
        Check if collection should be included in global searches.

        Returns:
            True if collection should appear in global search results
        """
        # System collections are not globally searchable by default
        if self.collection_category == CollectionCategory.SYSTEM:
            return False

        # Library and global collections are searchable
        if self.collection_category in [CollectionCategory.LIBRARY, CollectionCategory.GLOBAL]:
            return True

        # Project collections are searchable if not private
        if self.collection_category == CollectionCategory.PROJECT:
            return self.access_level != AccessLevel.PRIVATE

        return False

    def update_timestamp(self):
        """Update the updated_at timestamp and increment version."""
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.version += 1


# Export all public classes and constants
__all__ = [
    'MultiTenantMetadataSchema',
    'CollectionCategory',
    'WorkspaceScope',
    'AccessLevel',
    'METADATA_SCHEMA_VERSION',
    'MAX_PROJECT_NAME_LENGTH',
    'MAX_COLLECTION_TYPE_LENGTH',
    'MAX_TENANT_NAMESPACE_LENGTH',
    'MAX_CREATED_BY_LENGTH',
    'MAX_ACCESS_LEVEL_LENGTH',
    'MAX_BRANCH_LENGTH'
]
