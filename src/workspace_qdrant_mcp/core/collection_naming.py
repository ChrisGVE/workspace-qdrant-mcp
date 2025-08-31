"""
Collection naming system for workspace-qdrant-mcp.

This module implements the reserved collection naming architecture including:
- Memory collection for user preferences and LLM behavioral rules
- Underscore-prefixed library collections (_name pattern)
- Project collections with automatic creation
- Collection name validation and conflict prevention
- Display name mapping for user-facing interfaces

The naming system enforces these patterns:
1. Memory Collection: 'memory' (reserved)
2. Library Collections: '_{name}' (readonly from MCP, user-defined via CLI)
3. Project Collections: '{project-name}-{type}' (auto-created based on detection)

Key features:
- Hard error prevention for naming conflicts
- Display name mapping (library collections show as 'name' not '_name')
- MCP readonly enforcement for library collections
- Comprehensive validation with detailed error messages
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Types of collections in the naming system."""

    MEMORY = "memory"
    LIBRARY = "library"
    PROJECT = "project"
    LEGACY = "legacy"  # For existing collections that don't match patterns


@dataclass
class CollectionNameInfo:
    """Information about a collection name and its classification."""

    name: str                    # Actual collection name in Qdrant
    display_name: str           # Name shown to users (libraries without underscore)
    collection_type: CollectionType
    is_readonly_from_mcp: bool  # Whether MCP server can modify this collection
    project_name: str | None = None    # For project collections
    collection_suffix: str | None = None  # For project collections (docs, scratchbook, etc.)
    library_name: str | None = None    # For library collections (without underscore)


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
    3. Project collections: '{project}-{type}' (auto-created, read/write via MCP)
    4. Legacy collections: Existing collections that don't match new patterns
    """

    # Reserved collection names that cannot be used for user collections (excluding 'memory')
    RESERVED_NAMES = {
        "_memory",          # Prevent confusion with memory
        "system",           # Reserved for future system use
        "_system",          # Reserved for future system use
        "admin",            # Reserved for future admin use
        "_admin",           # Reserved for future admin use
    }

    # Valid project collection suffixes
    VALID_PROJECT_SUFFIXES = {
        "scratchbook",      # Interactive notes and context
        "docs",             # Documentation (not code - reserved for memexd)
    }

    # Pattern for valid library names (after underscore) - must be at least 2 chars
    LIBRARY_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$')

    # Pattern for valid project names - must be at least 2 chars
    PROJECT_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_-]*[a-z0-9]$|^[a-z]$')

    def __init__(self, global_collections: list[str] = None):
        """
        Initialize the collection naming manager.

        Args:
            global_collections: Legacy global collection names to preserve compatibility
        """
        self.global_collections = set(global_collections or [])

    def validate_collection_name(self, name: str,
                                intended_type: CollectionType | None = None) -> NamingValidationResult:
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
                is_valid=False,
                error_message="Collection name cannot be empty"
            )

        # Normalize name (strip whitespace)
        name = name.strip()

        if len(name) > 100:
            return NamingValidationResult(
                is_valid=False,
                error_message="Collection name cannot exceed 100 characters"
            )

        # Check for reserved names
        if name in self.RESERVED_NAMES:
            return NamingValidationResult(
                is_valid=False,
                error_message=f"'{name}' is a reserved collection name"
            )

        # Classify the collection type
        collection_info = self._classify_collection_name(name)

        # Validate based on type
        if collection_info.collection_type == CollectionType.MEMORY:
            if name != "memory":
                return NamingValidationResult(
                    is_valid=False,
                    error_message="Only 'memory' is allowed as a memory collection name"
                )

        elif collection_info.collection_type == CollectionType.LIBRARY:
            # Library classification handles validation internally
            if not collection_info.library_name:  # Invalid library name
                library_name = name[1:]  # Remove underscore prefix
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"Library name '{library_name}' must start with a letter, "
                                  f"contain only lowercase letters, numbers, hyphens, and underscores, "
                                  f"and end with a letter or number (or be a single letter)"
                )

        elif collection_info.collection_type == CollectionType.PROJECT:
            # Project classification handles validation internally
            if not collection_info.project_name or not collection_info.collection_suffix:
                # Check if it's a project pattern but invalid suffix
                parts = name.split('-')
                if len(parts) >= 2:
                    potential_suffix = parts[-1]
                    return NamingValidationResult(
                        is_valid=False,
                        error_message=f"Invalid project collection suffix '{potential_suffix}'. "
                                      f"Valid suffixes: {', '.join(self.VALID_PROJECT_SUFFIXES)}"
                    )
                else:
                    return NamingValidationResult(
                        is_valid=False,
                        error_message=f"Invalid project collection format: '{name}'. "
                                      f"Expected: 'project-name-suffix'"
                    )

        elif collection_info.collection_type == CollectionType.LEGACY:
            # For legacy collections, check if they look like invalid patterns
            if name.startswith('_'):
                # This is an invalid library name
                library_name = name[1:]  # Remove underscore prefix
                if not library_name:  # Just underscore
                    return NamingValidationResult(
                        is_valid=False,
                        error_message="Library name cannot be empty after underscore"
                    )
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"Library name '{library_name}' must start with a letter, "
                                  f"contain only lowercase letters, numbers, hyphens, and underscores, "
                                  f"and end with a letter or number (or be a single letter)"
                )

            # Check for malformed patterns
            if name.endswith('-'):
                return NamingValidationResult(
                    is_valid=False,
                    error_message="Collection names cannot end with a hyphen"
                )

            if name.startswith('-'):
                return NamingValidationResult(
                    is_valid=False,
                    error_message="Collection names cannot start with a hyphen"
                )

            # Check if it looks like invalid project collection
            parts = name.split('-')
            if len(parts) >= 2:
                potential_suffix = parts[-1]
                if potential_suffix not in self.VALID_PROJECT_SUFFIXES:
                    # Only flag as error if it looks like it's trying to be a project collection
                    suspicious_suffixes = {
                        'code', 'test', 'tests', 'build', 'dist', 'lib', 'src',
                        'invalid', 'temp', 'tmp', 'cache', 'config', 'data'
                    }
                    if potential_suffix in suspicious_suffixes:
                        return NamingValidationResult(
                            is_valid=False,
                            error_message=f"Invalid project collection suffix '{potential_suffix}'. "
                                          f"Valid suffixes: {', '.join(self.VALID_PROJECT_SUFFIXES)}"
                        )

        # Check type consistency if intended type was provided
        if intended_type and intended_type != collection_info.collection_type:
            return NamingValidationResult(
                is_valid=False,
                error_message=f"Collection name '{name}' suggests type {collection_info.collection_type.value} "
                              f"but intended type is {intended_type.value}"
            )

        return NamingValidationResult(
            is_valid=True,
            collection_info=collection_info
        )

    def check_naming_conflicts(self, proposed_name: str,
                             existing_collections: list[str]) -> NamingValidationResult:
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
                error_message=f"Collection '{proposed_name}' already exists"
            )

        # Check library/display name conflicts
        if proposed_name.startswith('_'):
            # Proposing a library collection, check if display name exists
            display_name = proposed_name[1:]
            if display_name in existing_set:
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"Cannot create library collection '{proposed_name}' because "
                                  f"collection '{display_name}' already exists. This would create "
                                  f"a naming conflict."
                )
        else:
            # Proposing a non-library collection, check if library version exists
            library_name = f'_{proposed_name}'
            if library_name in existing_set:
                return NamingValidationResult(
                    is_valid=False,
                    error_message=f"Cannot create collection '{proposed_name}' because "
                                  f"library collection '{library_name}' already exists. This would "
                                  f"create a naming conflict."
                )

        return NamingValidationResult(
            is_valid=True,
            collection_info=validation.collection_info
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

        For library collections, this removes the underscore prefix.
        For other collections, this returns the name unchanged.

        Args:
            collection_name: The actual collection name in Qdrant

        Returns:
            The display name to show to users
        """
        info = self._classify_collection_name(collection_name)
        return info.display_name

    def get_actual_name(self, display_name: str, collection_type: CollectionType) -> str:
        """
        Get the actual collection name in Qdrant from a display name.

        For library collections, this adds the underscore prefix.
        For other collections, this returns the name unchanged.

        Args:
            display_name: The name shown to users
            collection_type: The type of collection

        Returns:
            The actual collection name to use in Qdrant
        """
        if collection_type == CollectionType.LIBRARY:
            return f"_{display_name}"
        return display_name

    def is_mcp_readonly(self, collection_name: str) -> bool:
        """
        Check if a collection is readonly from the MCP server perspective.

        Library collections are readonly from MCP - they can only be modified
        via the CLI/Rust engine.

        Args:
            collection_name: The collection name to check

        Returns:
            True if the collection is readonly from MCP, False otherwise
        """
        info = self._classify_collection_name(collection_name)
        return info.is_readonly_from_mcp

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
            if collection.endswith('-code'):
                continue

            info = self._classify_collection_name(collection)

            # Include all workspace collection types
            if info.collection_type in [CollectionType.MEMORY, CollectionType.LIBRARY,
                                       CollectionType.PROJECT]:
                workspace_collections.append(collection)
            elif info.collection_type == CollectionType.LEGACY:
                # Include legacy global collections
                if collection in self.global_collections:
                    workspace_collections.append(collection)

        return sorted(workspace_collections)

    def generate_project_collection_names(self, project_name: str) -> list[str]:
        """
        Generate collection names for a project based on configured suffixes.

        Args:
            project_name: The project name

        Returns:
            List of collection names that should be created for the project
        """
        collections = []
        for suffix in self.VALID_PROJECT_SUFFIXES:
            collections.append(f"{project_name}-{suffix}")
        return collections

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
                is_readonly_from_mcp=False
            )

        # Library collections (underscore-prefixed)
        if name.startswith('_') and len(name) > 1:
            library_name = name[1:]
            # Validate library name format
            if self.LIBRARY_NAME_PATTERN.match(library_name):
                return CollectionNameInfo(
                    name=name,
                    display_name=library_name,
                    collection_type=CollectionType.LIBRARY,
                    is_readonly_from_mcp=True,
                    library_name=library_name
                )

        # Project collections (project-name-suffix)
        parts = name.split('-')
        if len(parts) >= 2:
            potential_suffix = parts[-1]
            if potential_suffix in self.VALID_PROJECT_SUFFIXES:
                project_name = '-'.join(parts[:-1])
                # Validate project name format
                if self.PROJECT_NAME_PATTERN.match(project_name):
                    return CollectionNameInfo(
                        name=name,
                        display_name=name,
                        collection_type=CollectionType.PROJECT,
                        is_readonly_from_mcp=False,
                        project_name=project_name,
                        collection_suffix=potential_suffix
                    )

        # Legacy/unknown collections
        return CollectionNameInfo(
            name=name,
            display_name=name,
            collection_type=CollectionType.LEGACY,
            is_readonly_from_mcp=False
        )


class CollectionPermissionError(Exception):
    """Exception raised when attempting prohibited collection operations."""
    pass


def create_naming_manager(global_collections: list[str] = None) -> CollectionNamingManager:
    """
    Create a collection naming manager instance.

    Args:
        global_collections: Legacy global collection names for compatibility

    Returns:
        Configured CollectionNamingManager instance
    """
    return CollectionNamingManager(global_collections)
