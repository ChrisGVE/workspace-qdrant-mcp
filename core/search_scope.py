"""
Search scope architecture system for qdrant_find operations.

This module provides comprehensive search scope resolution functionality that converts
scope strings into collection lists for the qdrant_find tool. It integrates with the
existing collection type system and configuration to provide flexible search operations
across different scopes: single collections, projects, workspaces, all accessible 
collections, and memory collections.

Key Features:
- Resolves search scope strings to collection lists
- Supports all required scopes: "collection", "project", "workspace", "all", "memory"
- Integrates with existing collection type system and configuration
- Handles memory collections (both system and project-scoped)
- Comprehensive error handling for invalid scope/collection combinations
- Type-safe implementation with complete docstrings

Search Scopes:
- collection: Single specified collection
- project: Collections belonging to the current project 
- workspace: Project + global collections (excludes system)
- all: All accessible collections (includes readonly, excludes system)
- memory: Both system and project memory collections

Example:
    ```python
    from core.search_scope import resolve_search_scope, SearchScopeError
    
    # Resolve scope to collection list
    collections = resolve_search_scope(
        scope="project",
        collection="my-project", 
        client=workspace_client,
        config=config
    )
    
    # Validate scope/collection combination
    validate_search_scope("collection", "my-collection")
    ```
"""

import re
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    from .collection_types import (
        CollectionTypeClassifier, 
        CollectionType, 
        GLOBAL_COLLECTIONS,
        SYSTEM_MEMORY_PATTERN,
        PROJECT_MEMORY_PATTERN
    )
    from .config import Config
except ImportError:
    # Fallback for direct imports when not used as a package
    from collection_types import (
        CollectionTypeClassifier,
        CollectionType,
        GLOBAL_COLLECTIONS,
        SYSTEM_MEMORY_PATTERN,
        PROJECT_MEMORY_PATTERN
    )
    from config import Config


class SearchScope(Enum):
    """Enumeration of supported search scopes."""
    
    COLLECTION = "collection"  # Single specified collection
    PROJECT = "project"        # Collections belonging to current project
    WORKSPACE = "workspace"    # Project + global collections (excludes system)
    ALL = "all"               # All accessible collections (includes readonly, excludes system)
    MEMORY = "memory"         # Both system and project memory collections


class SearchScopeError(Exception):
    """Exception raised for search scope related errors."""
    pass


class ScopeValidationError(SearchScopeError):
    """Exception raised when scope/collection combination is invalid."""
    pass


class CollectionNotFoundError(SearchScopeError):
    """Exception raised when specified collection is not found."""
    pass


# Search scope constants
VALID_SEARCH_SCOPES = {scope.value for scope in SearchScope}

# Scope descriptions for error messages
SCOPE_DESCRIPTIONS = {
    SearchScope.COLLECTION.value: "Single specified collection",
    SearchScope.PROJECT.value: "Collections belonging to the current project",
    SearchScope.WORKSPACE.value: "Project + global collections (excludes system)",
    SearchScope.ALL.value: "All accessible collections (includes readonly, excludes system)",
    SearchScope.MEMORY.value: "Both system and project memory collections"
}


@dataclass
class ScopeResolutionResult:
    """Result of search scope resolution."""
    
    scope: str
    collections: List[str]
    collection_count: int
    excluded_collections: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        """Initialize computed fields."""
        if self.excluded_collections is None:
            self.excluded_collections = []
        if self.warnings is None:
            self.warnings = []
        self.collection_count = len(self.collections)


def validate_search_scope(scope: str, collection: str) -> None:
    """
    Validate that a scope/collection combination is valid.
    
    Performs validation checks to ensure the scope is supported and that
    the collection parameter is appropriate for the given scope.
    
    Args:
        scope: The search scope string to validate
        collection: The collection parameter to validate
        
    Raises:
        ScopeValidationError: If the scope/collection combination is invalid
        
    Examples:
        >>> validate_search_scope("collection", "my-collection")  # Valid
        >>> validate_search_scope("project", "")  # Valid - empty collection for project scope
        >>> validate_search_scope("invalid", "test")  # Raises ScopeValidationError
    """
    if not scope:
        raise ScopeValidationError("Search scope cannot be empty")
    
    if scope not in VALID_SEARCH_SCOPES:
        raise ScopeValidationError(
            f"Invalid search scope '{scope}'. Must be one of: {', '.join(VALID_SEARCH_SCOPES)}"
        )
    
    # Validate collection parameter based on scope
    if scope == SearchScope.COLLECTION.value:
        if not collection:
            raise ScopeValidationError(
                "Collection name is required when using 'collection' scope"
            )
    elif scope in [SearchScope.PROJECT.value, SearchScope.WORKSPACE.value, 
                   SearchScope.ALL.value, SearchScope.MEMORY.value]:
        # These scopes determine collections automatically, collection param is optional/ignored
        pass


def resolve_search_scope(
    scope: str, 
    collection: str, 
    client: Any,  # QdrantWorkspaceClient type hint would create circular import
    config: Config
) -> List[str]:
    """
    Main function to resolve search scope strings to collection lists.
    
    Converts scope strings into lists of collection names based on the current
    workspace context, project detection, and collection availability. This is
    the primary entry point for scope resolution in the qdrant_find tool.
    
    Args:
        scope: The search scope ('collection', 'project', 'workspace', 'all', 'memory')
        collection: The specific collection name (required for 'collection' scope)
        client: The QdrantWorkspaceClient instance for accessing collections
        config: Configuration object containing workspace settings
        
    Returns:
        List[str]: List of collection names to search within the specified scope
        
    Raises:
        ScopeValidationError: If scope/collection combination is invalid
        CollectionNotFoundError: If specified collection doesn't exist
        SearchScopeError: If scope resolution fails due to client/config issues
        
    Examples:
        >>> collections = resolve_search_scope("collection", "my-docs", client, config)
        ['my-docs']
        >>> collections = resolve_search_scope("project", "", client, config)
        ['my-project-docs', 'my-project-code', 'my-project-memory']
        >>> collections = resolve_search_scope("memory", "", client, config)
        ['__user_memory', '__system_memory', 'my-project-memory']
    """
    # Validate the scope/collection combination
    validate_search_scope(scope, collection)
    
    if not client or not config:
        raise SearchScopeError("Client and config are required for scope resolution")
    
    if not client.initialized:
        raise SearchScopeError("Client must be initialized before resolving search scope")
    
    try:
        if scope == SearchScope.COLLECTION.value:
            return _resolve_single_collection(collection, client, config)
        elif scope == SearchScope.PROJECT.value:
            return get_project_collections(client, config)
        elif scope == SearchScope.WORKSPACE.value:
            return get_workspace_collections(client, config)
        elif scope == SearchScope.ALL.value:
            return get_all_collections(client, config)
        elif scope == SearchScope.MEMORY.value:
            return get_memory_collections(client, config)
        else:
            # Should not reach here due to validation, but defensive programming
            raise ScopeValidationError(f"Unsupported scope: {scope}")
            
    except Exception as e:
        if isinstance(e, (ScopeValidationError, CollectionNotFoundError)):
            raise
        raise SearchScopeError(f"Failed to resolve search scope '{scope}': {e}") from e


def _resolve_single_collection(
    collection: str, 
    client: Any, 
    config: Config
) -> List[str]:
    """
    Resolve a single collection name to a collection list.
    
    Validates that the specified collection exists and is accessible.
    
    Args:
        collection: The collection name to resolve
        client: The QdrantWorkspaceClient instance
        config: Configuration object
        
    Returns:
        List[str]: Single-item list containing the validated collection name
        
    Raises:
        CollectionNotFoundError: If the collection doesn't exist
    """
    available_collections = client.list_collections()
    
    if collection not in available_collections:
        raise CollectionNotFoundError(
            f"Collection '{collection}' not found. "
            f"Available collections: {', '.join(available_collections)}"
        )
    
    return [collection]


def get_project_collections(client: Any, config: Config) -> List[str]:
    """
    Get collections belonging to the current project.
    
    Returns all collections that are associated with the current project,
    including the main project collection, subproject collections, and
    project-specific memory collections.
    
    Args:
        client: The QdrantWorkspaceClient instance with project detection
        config: Configuration object containing project settings
        
    Returns:
        List[str]: Project-scoped collection names
        
    Examples:
        >>> get_project_collections(client, config)
        ['my-project-docs', 'my-project-frontend', 'my-project-backend', 'my-project-memory']
    """
    project_collections = []
    classifier = CollectionTypeClassifier()
    
    # Get all available collections
    all_collections = client.list_collections()
    
    # Get current project info
    project_info = client.get_project_info()
    if not project_info or not project_info.get("main_project"):
        return project_collections
    
    current_project = project_info["main_project"]
    
    # Filter for project collections
    for collection_name in all_collections:
        collection_info = classifier.get_collection_info(collection_name)
        
        # Include collections that belong to the current project
        if (collection_info.type == CollectionType.PROJECT and 
            collection_info.project_name == current_project):
            project_collections.append(collection_name)
    
    return sorted(project_collections)


def get_workspace_collections(client: Any, config: Config) -> List[str]:
    """
    Get workspace collections (project + global, excludes system).
    
    Returns collections that are accessible within the workspace context,
    including project-specific collections and global collections, but
    excluding system collections that are not globally searchable.
    
    Args:
        client: The QdrantWorkspaceClient instance
        config: Configuration object containing workspace settings
        
    Returns:
        List[str]: Workspace-accessible collection names
        
    Examples:
        >>> get_workspace_collections(client, config)
        ['algorithms', 'documents', 'knowledge', 'my-project-docs', 'my-project-memory']
    """
    workspace_collections = []
    classifier = CollectionTypeClassifier()
    
    # Get all available collections
    all_collections = client.list_collections()
    
    # Include project collections
    project_collections = get_project_collections(client, config)
    workspace_collections.extend(project_collections)
    
    # Include global collections
    for collection_name in all_collections:
        collection_info = classifier.get_collection_info(collection_name)
        
        # Include global collections and searchable library collections
        if (collection_info.type == CollectionType.GLOBAL or 
            (collection_info.type == CollectionType.LIBRARY and collection_info.is_searchable)):
            if collection_name not in workspace_collections:
                workspace_collections.append(collection_name)
    
    return sorted(workspace_collections)


def get_all_collections(client: Any, config: Config) -> List[str]:
    """
    Get all accessible collections (includes readonly, excludes system).
    
    Returns all collections that are accessible to the current user,
    including readonly collections, but excluding system collections
    that are not intended for general search operations.
    
    Args:
        client: The QdrantWorkspaceClient instance
        config: Configuration object
        
    Returns:
        List[str]: All accessible collection names
        
    Examples:
        >>> get_all_collections(client, config)
        ['algorithms', 'documents', '_library_docs', 'my-project-docs', 'other-project-docs']
    """
    accessible_collections = []
    classifier = CollectionTypeClassifier()
    
    # Get all available collections
    all_collections = client.list_collections()
    
    for collection_name in all_collections:
        collection_info = classifier.get_collection_info(collection_name)
        
        # Exclude only system collections (not globally searchable)
        if collection_info.type != CollectionType.SYSTEM:
            accessible_collections.append(collection_name)
    
    return sorted(accessible_collections)


def get_memory_collections(client: Any, config: Config) -> List[str]:
    """
    Get both system and project memory collections.
    
    Returns all collections that are designated as memory collections,
    including both system-level memory collections (with __ prefix) and
    project-specific memory collections (with -memory suffix).
    
    Args:
        client: The QdrantWorkspaceClient instance
        config: Configuration object
        
    Returns:
        List[str]: Memory collection names (both system and project)
        
    Examples:
        >>> get_memory_collections(client, config)
        ['__user_memory', '__system_memory', 'my-project-memory', 'other-project-memory']
    """
    memory_collections = []
    classifier = CollectionTypeClassifier()
    
    # Get all available collections
    all_collections = client.list_collections()
    
    # Compile patterns for memory collections
    system_memory_pattern = re.compile(SYSTEM_MEMORY_PATTERN)
    project_memory_pattern = re.compile(PROJECT_MEMORY_PATTERN)
    
    for collection_name in all_collections:
        # Check if collection matches memory patterns
        if (system_memory_pattern.match(collection_name) or 
            project_memory_pattern.match(collection_name)):
            memory_collections.append(collection_name)
        
        # Also check using the classifier's memory detection
        elif classifier.is_memory_collection(collection_name):
            if collection_name not in memory_collections:
                memory_collections.append(collection_name)
    
    return sorted(memory_collections)


def get_scope_description(scope: str) -> str:
    """
    Get a human-readable description of a search scope.
    
    Args:
        scope: The search scope string
        
    Returns:
        str: Description of what the scope includes
        
    Examples:
        >>> get_scope_description("project")
        'Collections belonging to the current project'
        >>> get_scope_description("invalid")
        'Unknown scope: invalid'
    """
    return SCOPE_DESCRIPTIONS.get(scope, f"Unknown scope: {scope}")


def get_available_scopes() -> List[str]:
    """
    Get list of all available search scopes.
    
    Returns:
        List[str]: Available search scope names
        
    Examples:
        >>> get_available_scopes()
        ['collection', 'project', 'workspace', 'all', 'memory']
    """
    return sorted(VALID_SEARCH_SCOPES)


def create_scope_resolution_result(
    scope: str,
    collections: List[str],
    excluded_collections: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None
) -> ScopeResolutionResult:
    """
    Create a structured scope resolution result.
    
    Args:
        scope: The resolved scope string
        collections: List of resolved collection names
        excluded_collections: Collections that were excluded from the scope
        warnings: Any warnings generated during resolution
        
    Returns:
        ScopeResolutionResult: Structured result object
    """
    return ScopeResolutionResult(
        scope=scope,
        collections=collections,
        collection_count=len(collections),
        excluded_collections=excluded_collections or [],
        warnings=warnings or []
    )


# Export all public classes and functions
__all__ = [
    # Enums
    'SearchScope',
    
    # Exceptions
    'SearchScopeError',
    'ScopeValidationError', 
    'CollectionNotFoundError',
    
    # Data classes
    'ScopeResolutionResult',
    
    # Constants
    'VALID_SEARCH_SCOPES',
    'SCOPE_DESCRIPTIONS',
    
    # Core functions
    'resolve_search_scope',
    'validate_search_scope',
    
    # Collection resolution functions
    'get_project_collections',
    'get_workspace_collections', 
    'get_all_collections',
    'get_memory_collections',
    
    # Utility functions
    'get_scope_description',
    'get_available_scopes',
    'create_scope_resolution_result'
]