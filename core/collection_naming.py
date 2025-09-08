"""
Collection name normalization framework for workspace-qdrant-mcp.

This module provides utilities for normalizing collection names to ensure
consistency and compatibility with Qdrant collection naming conventions.
The framework handles external names (project names, memory collection names)
and converts them into valid, normalized collection identifiers.

Key functions:
- normalize_collection_name_component(): Normalizes individual name components
- build_project_collection_name(): Builds project-scoped collection names
- build_system_memory_collection_name(): Builds system memory collection names
"""

import re
# Type alias for clarity
String = str


def normalize_collection_name_component(name: String) -> String:
    """
    Normalize external names for use in collection patterns.
    
    Converts dashes (-) and spaces to underscores (_) to create valid
    collection name components that are consistent and compatible with
    Qdrant collection naming conventions.
    
    Args:
        name: The raw name component to normalize (e.g., project name, suffix)
        
    Returns:
        str: The normalized name with dashes and spaces converted to underscores
        
    Raises:
        ValueError: If name is empty or contains only whitespace
        TypeError: If name is not a string
        
    Examples:
        >>> normalize_collection_name_component("my-project")
        'my_project'
        >>> normalize_collection_name_component("project name")  
        'project_name'
        >>> normalize_collection_name_component("mixed-name test")
        'mixed_name_test'
    """
    if not isinstance(name, str):
        raise TypeError(f"Expected string, got {type(name).__name__}")
    
    if not name.strip():
        raise ValueError("Name cannot be empty or contain only whitespace")
    
    # Convert dashes and spaces to underscores
    # First replace multiple consecutive spaces/dashes with single underscore
    normalized = re.sub(r'[-\s]+', '_', name.strip())
    
    # Remove leading/trailing underscores that might result from normalization
    normalized = normalized.strip('_')
    
    if not normalized:
        raise ValueError("Name resulted in empty string after normalization")
    
    return normalized


def build_project_collection_name(project: String, suffix: String) -> String:
    """
    Build project collection name with schema delimiter preservation.
    
    Creates a collection name for project-scoped collections using the format:
    {normalized_project}-{normalized_suffix}
    
    The dash (-) delimiter is preserved between project and suffix to maintain
    the schema distinction, while individual components are normalized.
    
    Args:
        project: The project name to normalize and use as prefix
        suffix: The collection suffix to normalize and append
        
    Returns:
        str: Project collection name in format: {project}-{suffix}
        
    Raises:
        ValueError: If project or suffix is empty after normalization
        TypeError: If project or suffix is not a string
        
    Examples:
        >>> build_project_collection_name("my-project", "documents")
        'my_project-documents'
        >>> build_project_collection_name("test project", "source-code")
        'test_project-source_code'
        >>> build_project_collection_name("workspace", "lsp metadata")
        'workspace-lsp_metadata'
    """
    if not isinstance(project, str):
        raise TypeError(f"Project must be string, got {type(project).__name__}")
        
    if not isinstance(suffix, str):
        raise TypeError(f"Suffix must be string, got {type(suffix).__name__}")
    
    # Normalize both components
    normalized_project = normalize_collection_name_component(project)
    normalized_suffix = normalize_collection_name_component(suffix)
    
    # Build with schema delimiter (dash) preserved
    collection_name = f"{normalized_project}-{normalized_suffix}"
    
    return collection_name


def build_system_memory_collection_name(memory_collection: String) -> String:
    """
    Build system memory collection name with double underscore prefix.
    
    Creates a collection name for system memory collections using the format:
    __{normalized_memory}
    
    The double underscore prefix (__) indicates system-level collections
    that are distinct from project-scoped collections.
    
    Args:
        memory_collection: The memory collection name to normalize
        
    Returns:
        str: System memory collection name in format: __{memory_collection}
        
    Raises:
        ValueError: If memory_collection is empty after normalization
        TypeError: If memory_collection is not a string
        
    Examples:
        >>> build_system_memory_collection_name("user-preferences")
        '__user_preferences'
        >>> build_system_memory_collection_name("system config")
        '__system_config'
        >>> build_system_memory_collection_name("global-state")
        '__global_state'
    """
    if not isinstance(memory_collection, str):
        raise TypeError(f"Memory collection must be string, got {type(memory_collection).__name__}")
    
    # Normalize the memory collection name
    normalized_memory = normalize_collection_name_component(memory_collection)
    
    # Build with system prefix (double underscore)
    collection_name = f"__{normalized_memory}"
    
    return collection_name


def validate_collection_name(name: String) -> bool:
    """
    Validate that a collection name follows expected patterns.
    
    Checks if a collection name matches either:
    - Project pattern: {component}-{component}
    - System pattern: __{component}
    - Single component pattern: {component}
    
    Args:
        name: The collection name to validate
        
    Returns:
        bool: True if the name follows expected patterns, False otherwise
        
    Examples:
        >>> validate_collection_name("project_name-documents")
        True
        >>> validate_collection_name("__system_config")  
        True
        >>> validate_collection_name("invalid--name")
        False
    """
    if not isinstance(name, str) or not name:
        return False
    
    # System memory pattern: __component
    if name.startswith('__'):
        component = name[2:]
        return bool(component and re.match(r'^[a-zA-Z0-9_]+$', component))
    
    # Project pattern: component-component or single component
    if '-' in name:
        parts = name.split('-')
        if len(parts) != 2:
            return False
        return all(part and re.match(r'^[a-zA-Z0-9_]+$', part) for part in parts)
    else:
        # Single component
        return bool(re.match(r'^[a-zA-Z0-9_]+$', name))


# Module-level constants for common patterns
PROJECT_COLLECTION_PATTERN = r'^[a-zA-Z0-9_]+-[a-zA-Z0-9_]+$'
SYSTEM_COLLECTION_PATTERN = r'^__[a-zA-Z0-9_]+$'
SINGLE_COMPONENT_PATTERN = r'^[a-zA-Z0-9_]+$'

# Export all public functions
__all__ = [
    'normalize_collection_name_component',
    'build_project_collection_name', 
    'build_system_memory_collection_name',
    'validate_collection_name',
    'PROJECT_COLLECTION_PATTERN',
    'SYSTEM_COLLECTION_PATTERN', 
    'SINGLE_COMPONENT_PATTERN'
]