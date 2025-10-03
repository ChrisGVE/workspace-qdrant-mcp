"""
Collection type hierarchy system for workspace-qdrant-mcp.

This module builds on the collection naming framework to provide a comprehensive
system for classifying and managing different types of collections in the Qdrant
vector database. It defines collection type constants, patterns, and provides
classification and validation utilities.

Collection Type Hierarchy:
- SYSTEM: "__" prefix (CLI-writable, LLM-readable, not globally searchable)
- LIBRARY: "_" prefix (CLI-managed, MCP-readonly, globally searchable)
- PROJECT: "_{project_id}" format (12-char hex hash, single collection per project)
- GLOBAL: Predefined global collections (system-wide, always available)

Key classes and functions:
- CollectionTypeClassifier: Main classification and utility class
- Collection type constants and patterns
- Display name mapping (removes prefixes for system collections)
- Validation functions for collection operations
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass

try:
    from . import collection_naming
except ImportError:
    # For direct imports when not used as a package
    import collection_naming

# Flag to indicate collection types module is available
COLLECTION_TYPES_AVAILABLE = True


# Collection Type Constants
SYSTEM_PREFIX = "__"  # CLI-writable, LLM-readable, not globally searchable
LIBRARY_PREFIX = "_"  # CLI-managed, MCP-readonly, globally searchable

# Global Collections (system-wide, always available)
GLOBAL_COLLECTIONS = [
    "algorithms",
    "codebase",
    "context",
    "documents",
    "knowledge",
    "memory",
    "projects",
    "workspace"
]

# Pattern formats for different collection types
PROJECT_PATTERN = r"^_[a-f0-9]{12}$"  # _{project_id} where project_id is 12-char hex
SYSTEM_MEMORY_PATTERN = r"^__[a-zA-Z0-9_-]+$"      # __{memory_name}
PROJECT_MEMORY_PATTERN = r"^[a-zA-Z0-9_-]+-memory$" # {project}-memory (DEPRECATED)


class CollectionType(Enum):
    """Enumeration of collection types."""

    SYSTEM = "system"       # __ prefix
    LIBRARY = "library"     # _ prefix
    PROJECT = "project"     # _{project_id} format (12-char hex)
    GLOBAL = "global"       # Predefined global collections
    UNKNOWN = "unknown"     # Unrecognized pattern


@dataclass
class CollectionInfo:
    """Information about a collection's type and properties."""

    name: str
    type: CollectionType
    display_name: str
    is_searchable: bool
    is_readonly: bool
    is_memory_collection: bool
    project_id: Optional[str] = None  # For project collections (12-char hex)
    suffix: Optional[str] = None  # DEPRECATED - for backward compatibility only


class CollectionTypeClassifier:
    """
    Classifier for determining collection types and properties.

    This class provides comprehensive methods for classifying collections,
    determining their properties, and managing display names. It integrates
    with the collection_naming module to provide a complete collection
    management system.
    """

    def __init__(self):
        """Initialize the classifier with compiled regex patterns."""
        self._project_pattern = re.compile(PROJECT_PATTERN)
        self._system_memory_pattern = re.compile(SYSTEM_MEMORY_PATTERN)
        self._project_memory_pattern = re.compile(PROJECT_MEMORY_PATTERN)
        self._global_collections_set = set(GLOBAL_COLLECTIONS)

    def classify_collection_type(self, collection_name: str) -> CollectionType:
        """
        Classify a collection name into its type category.

        Args:
            collection_name: The collection name to classify

        Returns:
            CollectionType: The classified type of the collection

        Examples:
            >>> classifier = CollectionTypeClassifier()
            >>> classifier.classify_collection_type("__user_preferences")
            CollectionType.SYSTEM
            >>> classifier.classify_collection_type("_library_docs")
            CollectionType.LIBRARY
            >>> classifier.classify_collection_type("_a1b2c3d4e5f6")
            CollectionType.PROJECT
            >>> classifier.classify_collection_type("algorithms")
            CollectionType.GLOBAL
        """
        if not isinstance(collection_name, str) or not collection_name:
            return CollectionType.UNKNOWN

        # Check for system collections (__ prefix)
        if collection_name.startswith(SYSTEM_PREFIX):
            return CollectionType.SYSTEM

        # Check for project collections (_{project_id} where project_id is 12-char hex)
        if self._project_pattern.match(collection_name):
            return CollectionType.PROJECT

        # Check for library collections (_ prefix, but not __ and not project)
        if collection_name.startswith(LIBRARY_PREFIX) and not collection_name.startswith(SYSTEM_PREFIX):
            return CollectionType.LIBRARY

        # Check for global collections
        if collection_name in self._global_collections_set:
            return CollectionType.GLOBAL

        return CollectionType.UNKNOWN

    def is_system_collection(self, collection_name: str) -> bool:
        """
        Check if a collection is a system collection.

        System collections use the __ prefix and are CLI-writable,
        LLM-readable, but not globally searchable.

        Args:
            collection_name: The collection name to check

        Returns:
            bool: True if the collection is a system collection
        """
        return self.classify_collection_type(collection_name) == CollectionType.SYSTEM

    def is_library_collection(self, collection_name: str) -> bool:
        """
        Check if a collection is a library collection.

        Library collections use the _ prefix and are CLI-managed,
        MCP-readonly, and globally searchable.

        Args:
            collection_name: The collection name to check

        Returns:
            bool: True if the collection is a library collection
        """
        return self.classify_collection_type(collection_name) == CollectionType.LIBRARY

    def is_project_collection(self, collection_name: str) -> bool:
        """
        Check if a collection is a project collection.

        Project collections use the _{project_id} format (12-char hex hash).

        Args:
            collection_name: The collection name to check

        Returns:
            bool: True if the collection is a project collection
        """
        return self.classify_collection_type(collection_name) == CollectionType.PROJECT

    def is_global_collection(self, collection_name: str) -> bool:
        """
        Check if a collection is a global collection.

        Global collections are predefined system-wide collections
        that are always available.

        Args:
            collection_name: The collection name to check

        Returns:
            bool: True if the collection is a global collection
        """
        return self.classify_collection_type(collection_name) == CollectionType.GLOBAL

    def is_memory_collection(self, collection_name: str) -> bool:
        """
        Check if a collection is a memory collection.

        Memory collections use system memory pattern: __{memory_name}

        Args:
            collection_name: The collection name to check

        Returns:
            bool: True if the collection is a memory collection
        """
        return self._system_memory_pattern.match(collection_name) is not None

    def get_display_name(self, collection_name: str) -> str:
        """
        Get the display name for a collection (removes prefixes for system/library/project collections).

        For system, library, and project collections, this removes the prefix to create
        a cleaner display name. For other collections, returns the name as-is.

        Args:
            collection_name: The collection name to process

        Returns:
            str: The display name for the collection

        Examples:
            >>> classifier = CollectionTypeClassifier()
            >>> classifier.get_display_name("__user_preferences")
            'user_preferences'
            >>> classifier.get_display_name("_library_docs")
            'library_docs'
            >>> classifier.get_display_name("_a1b2c3d4e5f6")
            'a1b2c3d4e5f6'
        """
        collection_type = self.classify_collection_type(collection_name)

        if collection_type == CollectionType.SYSTEM:
            return collection_name[len(SYSTEM_PREFIX):]
        elif collection_type in (CollectionType.LIBRARY, CollectionType.PROJECT):
            return collection_name[len(LIBRARY_PREFIX):]
        else:
            return collection_name

    def get_collection_info(self, collection_name: str) -> CollectionInfo:
        """
        Get comprehensive information about a collection.

        Args:
            collection_name: The collection name to analyze

        Returns:
            CollectionInfo: Complete information about the collection
        """
        collection_type = self.classify_collection_type(collection_name)
        display_name = self.get_display_name(collection_name)
        is_memory = self.is_memory_collection(collection_name)

        # Determine properties based on type
        if collection_type == CollectionType.SYSTEM:
            is_searchable = False  # Not globally searchable
            is_readonly = True     # CLI-writable only, read-only from MCP
        elif collection_type == CollectionType.LIBRARY:
            is_searchable = True   # Globally searchable
            is_readonly = True     # MCP-readonly
        elif collection_type == CollectionType.PROJECT:
            is_searchable = True   # Project collections are searchable
            is_readonly = False    # User-writable via MCP
        elif collection_type == CollectionType.GLOBAL:
            is_searchable = True   # System-wide availability
            is_readonly = False    # Generally writable
        else:
            is_searchable = False  # Unknown collections default to restricted
            is_readonly = True     # Unknown collections default to readonly

        # Extract project ID for project collections
        project_id = None
        if collection_type == CollectionType.PROJECT and collection_name.startswith('_'):
            project_id = collection_name[1:]  # Remove underscore prefix

        return CollectionInfo(
            name=collection_name,
            type=collection_type,
            display_name=display_name,
            is_searchable=is_searchable,
            is_readonly=is_readonly,
            is_memory_collection=is_memory,
            project_id=project_id,
            suffix=None  # DEPRECATED
        )

    def extract_project_id(self, collection_name: str) -> Optional[str]:
        """
        Extract project ID from project collection names.

        Args:
            collection_name: The collection name to parse

        Returns:
            Optional[str]: Project ID (12-char hex) if project collection, None otherwise

        Examples:
            >>> classifier = CollectionTypeClassifier()
            >>> classifier.extract_project_id("_a1b2c3d4e5f6")
            'a1b2c3d4e5f6'
            >>> classifier.extract_project_id("__system_config")
            None
        """
        if self.is_project_collection(collection_name) and collection_name.startswith('_'):
            return collection_name[1:]
        return None


# Module-level utility functions

def validate_collection_operation(collection_name: str, operation: str) -> Tuple[bool, str]:
    """
    Validate if an operation is allowed on a collection based on its type.

    Args:
        collection_name: The collection name to validate
        operation: The operation to validate ('read', 'write', 'delete', 'create')

    Returns:
        Tuple[bool, str]: (is_valid, reason) where reason explains why if invalid

    Examples:
        >>> validate_collection_operation("_library_docs", "write")
        (False, "Library collections are read-only via MCP")
        >>> validate_collection_operation("__user_prefs", "write")
        (True, "System collections are CLI-writable")
    """
    classifier = CollectionTypeClassifier()
    collection_info = classifier.get_collection_info(collection_name)

    valid_operations = {'read', 'write', 'delete', 'create'}
    if operation not in valid_operations:
        return False, f"Invalid operation '{operation}'. Must be one of: {valid_operations}"

    # Read operations are generally allowed
    if operation == 'read':
        return True, "Read operations are generally allowed"

    # Check write/delete/create permissions based on collection type
    if collection_info.is_readonly:
        if collection_info.type == CollectionType.LIBRARY:
            return False, "Library collections are read-only via MCP"
        else:
            return False, f"Collection '{collection_name}' is read-only"

    # Special validation for system collections
    if collection_info.type == CollectionType.SYSTEM:
        return True, "System collections are CLI-writable"

    # Other collections follow standard permissions
    return True, f"Operation '{operation}' is allowed on {collection_info.type.value} collection"


def get_collections_by_type(collections: List[str],
                          collection_type: CollectionType) -> List[str]:
    """
    Filter collections by their type.

    Args:
        collections: List of collection names to filter
        collection_type: The type to filter for

    Returns:
        List[str]: Collections matching the specified type

    Examples:
        >>> collections = ["__system", "_lib", "_a1b2c3d4e5f6", "algorithms"]
        >>> get_collections_by_type(collections, CollectionType.SYSTEM)
        ['__system']
    """
    classifier = CollectionTypeClassifier()
    return [name for name in collections
            if classifier.classify_collection_type(name) == collection_type]


def get_searchable_collections(collections: List[str]) -> List[str]:
    """
    Get collections that are globally searchable.

    Args:
        collections: List of collection names to filter

    Returns:
        List[str]: Collections that are globally searchable
    """
    classifier = CollectionTypeClassifier()
    return [name for name in collections
            if classifier.get_collection_info(name).is_searchable]


def validate_collection_name_with_type(collection_name: str,
                                     expected_type: CollectionType) -> bool:
    """
    Validate that a collection name matches an expected type.

    Args:
        collection_name: The collection name to validate
        expected_type: The expected collection type

    Returns:
        bool: True if the collection name matches the expected type

    Examples:
        >>> validate_collection_name_with_type("__system_config", CollectionType.SYSTEM)
        True
        >>> validate_collection_name_with_type("_a1b2c3d4e5f6", CollectionType.PROJECT)
        True
    """
    classifier = CollectionTypeClassifier()
    actual_type = classifier.classify_collection_type(collection_name)
    return actual_type == expected_type


def build_collection_name_for_type(base_name: str,
                                 collection_type: CollectionType,
                                 project_id: Optional[str] = None) -> str:
    """
    Build a collection name for a specific type using the existing naming framework.

    Args:
        base_name: The base name to use (for system/library/global types)
        collection_type: The type of collection to create
        project_id: 12-character hex project ID (required for project collections)

    Returns:
        str: The properly formatted collection name

    Raises:
        ValueError: If required parameters are missing or invalid

    Examples:
        >>> build_collection_name_for_type("user_prefs", CollectionType.SYSTEM)
        '__user_prefs'
        >>> build_collection_name_for_type("", CollectionType.PROJECT, project_id="a1b2c3d4e5f6")
        '_a1b2c3d4e5f6'
    """
    if collection_type == CollectionType.SYSTEM:
        return collection_naming.build_system_memory_collection_name(base_name)
    elif collection_type == CollectionType.LIBRARY:
        normalized = collection_naming.normalize_collection_name_component(base_name)
        return f"{LIBRARY_PREFIX}{normalized}"
    elif collection_type == CollectionType.PROJECT:
        if not project_id:
            raise ValueError("project_id is required for project collections")
        return collection_naming.build_project_collection_name(project_id)
    elif collection_type == CollectionType.GLOBAL:
        # Global collections use their name as-is
        if base_name not in GLOBAL_COLLECTIONS:
            raise ValueError(f"'{base_name}' is not a valid global collection. "
                           f"Must be one of: {GLOBAL_COLLECTIONS}")
        return base_name
    else:
        raise ValueError(f"Cannot build collection name for type: {collection_type}")


# Export all public classes and functions
__all__ = [
    # Constants
    'SYSTEM_PREFIX',
    'LIBRARY_PREFIX',
    'GLOBAL_COLLECTIONS',
    'PROJECT_PATTERN',
    'SYSTEM_MEMORY_PATTERN',
    'PROJECT_MEMORY_PATTERN',

    # Classes
    'CollectionType',
    'CollectionInfo',
    'CollectionTypeClassifier',

    # Utility functions
    'validate_collection_operation',
    'get_collections_by_type',
    'get_searchable_collections',
    'validate_collection_name_with_type',
    'build_collection_name_for_type'
]
