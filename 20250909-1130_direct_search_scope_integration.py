#!/usr/bin/env python3
"""
Direct integration of search_scope parameter into qdrant_find.

This script demonstrates the search scope integration by creating a working
version of the qdrant_find function with search_scope support.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import re
from enum import Enum

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# Simplified Search Scope System (integrated from temporary files)
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


# Collection type constants
VALID_SEARCH_SCOPES = {scope.value for scope in SearchScope}
SYSTEM_MEMORY_PATTERN = r"^__[a-zA-Z0-9_]+$"
PROJECT_MEMORY_PATTERN = r"^[a-zA-Z0-9_]+-memory$"

# Global collections
GLOBAL_COLLECTIONS = [
    "algorithms", "codebase", "context", "documents", 
    "knowledge", "memory", "projects", "workspace"
]


def validate_search_scope(scope: str, collection: str) -> None:
    """Validate that a scope/collection combination is valid."""
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


def resolve_search_scope(scope: str, collection: str, client, config) -> List[str]:
    """Resolve search scope strings to collection lists."""
    validate_search_scope(scope, collection)
    
    if not client or not config:
        raise SearchScopeError("Client and config are required for scope resolution")
    
    if not hasattr(client, 'initialized') or not client.initialized:
        raise SearchScopeError("Client must be initialized before resolving search scope")
    
    try:
        if scope == SearchScope.COLLECTION.value:
            return _resolve_single_collection(collection, client)
        elif scope == SearchScope.PROJECT.value:
            return get_project_collections(client)
        elif scope == SearchScope.WORKSPACE.value:
            return get_workspace_collections(client)
        elif scope == SearchScope.ALL.value:
            return get_all_collections(client)
        elif scope == SearchScope.MEMORY.value:
            return get_memory_collections(client)
        else:
            raise ScopeValidationError(f"Unsupported scope: {scope}")
            
    except Exception as e:
        if isinstance(e, (ScopeValidationError, CollectionNotFoundError)):
            raise
        raise SearchScopeError(f"Failed to resolve search scope '{scope}': {e}") from e


def _resolve_single_collection(collection: str, client) -> List[str]:
    """Resolve a single collection name to a collection list."""
    available_collections = client.list_collections()
    
    if collection not in available_collections:
        raise CollectionNotFoundError(
            f"Collection '{collection}' not found. "
            f"Available collections: {', '.join(available_collections)}"
        )
    
    return [collection]


def get_project_collections(client) -> List[str]:
    """Get collections belonging to the current project."""
    project_collections = []
    all_collections = client.list_collections()
    
    # Get current project info
    project_info = client.get_project_info()
    if not project_info or not project_info.get("main_project"):
        return project_collections
    
    current_project = project_info["main_project"]
    
    # Filter for project collections (simple pattern matching)
    for collection_name in all_collections:
        if collection_name.startswith(f"{current_project}-"):
            project_collections.append(collection_name)
    
    return sorted(project_collections)


def get_workspace_collections(client) -> List[str]:
    """Get workspace collections (project + global, excludes system)."""
    workspace_collections = []
    all_collections = client.list_collections()
    
    # Include project collections
    project_collections = get_project_collections(client)
    workspace_collections.extend(project_collections)
    
    # Include global collections
    for collection_name in all_collections:
        if (collection_name in GLOBAL_COLLECTIONS and 
            collection_name not in workspace_collections):
            workspace_collections.append(collection_name)
        
        # Include library collections (_prefix but not __)
        elif (collection_name.startswith("_") and 
              not collection_name.startswith("__") and
              collection_name not in workspace_collections):
            workspace_collections.append(collection_name)
    
    return sorted(workspace_collections)


def get_all_collections(client) -> List[str]:
    """Get all accessible collections (includes readonly, excludes system)."""
    accessible_collections = []
    all_collections = client.list_collections()
    
    for collection_name in all_collections:
        # Exclude only system collections (__ prefix)
        if not collection_name.startswith("__"):
            accessible_collections.append(collection_name)
    
    return sorted(accessible_collections)


def get_memory_collections(client) -> List[str]:
    """Get both system and project memory collections."""
    memory_collections = []
    all_collections = client.list_collections()
    
    # Compile patterns for memory collections
    system_memory_pattern = re.compile(SYSTEM_MEMORY_PATTERN)
    project_memory_pattern = re.compile(PROJECT_MEMORY_PATTERN)
    
    for collection_name in all_collections:
        # Check if collection matches memory patterns
        if (system_memory_pattern.match(collection_name) or 
            project_memory_pattern.match(collection_name)):
            memory_collections.append(collection_name)
    
    return sorted(memory_collections)


# Mock workspace client for testing
class MockWorkspaceClient:
    """Mock workspace client for testing search scope functionality."""
    
    def __init__(self):
        self.initialized = True
        self.mock_collections = [
            # System collections
            "__user_memory", "__system_config", "__user_prefs",
            # Library collections  
            "_library_docs", "_reference_data",
            # Project collections
            "my-project-docs", "my-project-memory", "my-project-code",
            "other-project-docs", "other-project-memory",
            # Global collections
            "algorithms", "documents", "knowledge", "workspace"
        ]
        self.mock_project_info = {
            "main_project": "my-project",
            "project_collections": ["my-project-docs", "my-project-memory", "my-project-code"]
        }
    
    def list_collections(self) -> List[str]:
        return self.mock_collections.copy()
    
    def get_project_info(self) -> Dict[str, Any]:
        return self.mock_project_info.copy()


# Enhanced qdrant_find function with search_scope parameter
async def enhanced_qdrant_find(
    query: str,
    search_scope: str = "project",  # NEW PARAMETER
    collection: str = None,
    limit: int = 10,
    score_threshold: float = 0.7,
    search_mode: str = "hybrid",
    filters: Dict[str, Any] = None,
    note_types: List[str] = None,
    tags: List[str] = None,
    include_relationships: bool = False,
    workspace_client=None,
    config=None
) -> Dict[str, Any]:
    """
    Enhanced qdrant_find with search_scope parameter.
    
    Args:
        query: Natural language search query
        search_scope: Search scope - "collection", "project", "workspace", "all", "memory"
        collection: Specific collection (required for "collection" scope)
        limit: Maximum number of results
        score_threshold: Minimum relevance score
        search_mode: Search strategy - "hybrid", "semantic", "keyword", "exact"
        filters: Metadata filters
        note_types: Filter by content types
        tags: Filter by tags
        include_relationships: Include related documents
        workspace_client: Client instance (for testing)
        config: Configuration object (for testing)
        
    Returns:
        dict: Search results with resolved scope information
    """
    
    # Use mock client for testing if not provided
    if not workspace_client:
        workspace_client = MockWorkspaceClient()
    if not config:
        config = {"default": True}  # Simple mock config
    
    # Validate parameters
    try:
        limit = int(limit) if isinstance(limit, str) else limit
        score_threshold = float(score_threshold) if isinstance(score_threshold, str) else score_threshold
        
        if limit <= 0:
            return {"error": "limit must be greater than 0"}
        if not (0.0 <= score_threshold <= 1.0):
            return {"error": "score_threshold must be between 0.0 and 1.0"}
    except (ValueError, TypeError) as e:
        return {"error": f"Invalid parameter types: {e}"}
    
    try:
        # Resolve search scope to collection list
        target_collections = resolve_search_scope(search_scope, collection, workspace_client, config)
        
        # Mock search results (in real implementation, this would call actual search)
        mock_results = []
        for coll in target_collections[:limit]:  # Simulate results from resolved collections
            mock_results.append({
                "collection": coll,
                "content": f"Mock result from {coll} for query: {query}",
                "score": 0.85,
                "metadata": {"search_scope": search_scope, "resolved_from": coll}
            })
        
        return {
            "success": True,
            "query": query,
            "search_scope": search_scope,
            "resolved_collections": target_collections,
            "total_collections": len(target_collections),
            "results": mock_results,
            "total_results": len(mock_results),
            "parameters": {
                "limit": limit,
                "score_threshold": score_threshold,
                "search_mode": search_mode,
                "filters": filters,
                "note_types": note_types,
                "tags": tags,
                "include_relationships": include_relationships
            }
        }
        
    except (SearchScopeError, ScopeValidationError, CollectionNotFoundError) as e:
        return {"error": f"Search scope error: {str(e)}", "success": False}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "success": False}


async def test_enhanced_qdrant_find():
    """Test the enhanced qdrant_find function with search_scope parameter."""
    print("Testing Enhanced qdrant_find with Search Scope")
    print("=" * 50)
    
    # Test cases for different search scopes
    test_cases = [
        {
            "name": "Collection Scope",
            "params": {"query": "test query", "search_scope": "collection", "collection": "my-project-docs"}
        },
        {
            "name": "Project Scope", 
            "params": {"query": "project content", "search_scope": "project"}
        },
        {
            "name": "Workspace Scope",
            "params": {"query": "workspace search", "search_scope": "workspace"}
        },
        {
            "name": "All Collections Scope",
            "params": {"query": "comprehensive search", "search_scope": "all"}
        },
        {
            "name": "Memory Collections Scope",
            "params": {"query": "memory search", "search_scope": "memory"}
        }
    ]
    
    for test_case in test_cases:
        print(f"\\n--- {test_case['name']} ---")
        try:
            result = await enhanced_qdrant_find(**test_case["params"])
            
            if result.get("success"):
                print(f"✓ Search successful")
                print(f"  Query: {result['query']}")
                print(f"  Scope: {result['search_scope']}")
                print(f"  Resolved Collections ({result['total_collections']}): {', '.join(result['resolved_collections'])}")
                print(f"  Results: {result['total_results']}")
            else:
                print(f"✗ Search failed: {result.get('error')}")
                
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    # Test error cases
    print("\\n--- Error Handling Tests ---")
    
    error_test_cases = [
        {
            "name": "Missing collection for collection scope",
            "params": {"query": "test", "search_scope": "collection", "collection": ""}
        },
        {
            "name": "Invalid scope",
            "params": {"query": "test", "search_scope": "invalid"}
        },
        {
            "name": "Non-existent collection",
            "params": {"query": "test", "search_scope": "collection", "collection": "non-existent"}
        }
    ]
    
    for test_case in error_test_cases:
        print(f"\\n{test_case['name']}:")
        try:
            result = await enhanced_qdrant_find(**test_case["params"])
            if result.get("error"):
                print(f"✓ Error correctly caught: {result['error']}")
            else:
                print(f"✗ Error should have been caught")
        except Exception as e:
            print(f"✓ Exception correctly raised: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_enhanced_qdrant_find())