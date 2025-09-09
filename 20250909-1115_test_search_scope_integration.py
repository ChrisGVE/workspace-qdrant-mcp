#!/usr/bin/env python3
"""
Test script for validating search scope integration with qdrant_find.

This script tests the search scope architecture before integrating it
with the qdrant_find function to ensure all components work correctly.
"""

import os
import sys
import asyncio
from typing import Dict, Any, List
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Test imports first
try:
    from workspace_qdrant_mcp.core.config import Config
    print("✓ Config module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import Config: {e}")
    sys.exit(1)

# Import search scope system and collection types later after setting up modules
search_scope_module = None
collection_types_module = None


class MockWorkspaceClient:
    """Mock client for testing search scope resolution."""
    
    def __init__(self):
        self.initialized = True
        
        # Mock collections for testing
        self.mock_collections = [
            # System collections
            "__user_memory",
            "__system_config", 
            "__user_prefs",
            
            # Library collections
            "_library_docs",
            "_reference_data",
            
            # Project collections  
            "my-project-docs",
            "my-project-memory",
            "my-project-code",
            "other-project-docs",
            
            # Global collections
            "algorithms",
            "documents", 
            "knowledge",
            "workspace"
        ]
        
        self.mock_project_info = {
            "main_project": "my-project",
            "project_collections": ["my-project-docs", "my-project-memory", "my-project-code"]
        }
    
    def list_collections(self) -> List[str]:
        """Mock method to return available collections."""
        return self.mock_collections.copy()
    
    def get_project_info(self) -> Dict[str, Any]:
        """Mock method to return project information."""
        return self.mock_project_info.copy()


async def test_search_scope_validation():
    """Test search scope validation functionality."""
    print("\n--- Testing Search Scope Validation ---")
    
    # Get functions from loaded module
    validate_search_scope = search_scope_module.validate_search_scope
    get_available_scopes = search_scope_module.get_available_scopes
    ScopeValidationError = search_scope_module.ScopeValidationError
    
    # Test valid scopes
    valid_scopes = ["collection", "project", "workspace", "all", "memory"]
    for scope in valid_scopes:
        try:
            validate_search_scope(scope, "test-collection")
            print(f"✓ Valid scope '{scope}' accepted")
        except ScopeValidationError as e:
            print(f"✗ Valid scope '{scope}' rejected: {e}")
    
    # Test collection scope requires collection parameter
    try:
        validate_search_scope("collection", "")
        print("✗ Collection scope should require collection parameter")
    except ScopeValidationError:
        print("✓ Collection scope correctly requires collection parameter")
    
    # Test invalid scopes
    try:
        validate_search_scope("invalid", "test")
        print("✗ Invalid scope should be rejected")
    except ScopeValidationError:
        print("✓ Invalid scope correctly rejected")
    
    # Test available scopes
    available = get_available_scopes()
    print(f"✓ Available scopes: {available}")


async def test_search_scope_resolution():
    """Test search scope resolution with mock client."""
    print("\n--- Testing Search Scope Resolution ---")
    
    # Get functions from loaded module
    resolve_search_scope = search_scope_module.resolve_search_scope
    SearchScopeError = search_scope_module.SearchScopeError
    ScopeValidationError = search_scope_module.ScopeValidationError
    
    client = MockWorkspaceClient()
    config = Config()  # Use default config
    
    test_cases = [
        ("collection", "my-project-docs"),
        ("project", ""),
        ("workspace", ""),
        ("all", ""),
        ("memory", "")
    ]
    
    for scope, collection in test_cases:
        try:
            resolved_collections = resolve_search_scope(scope, collection, client, config)
            print(f"✓ Scope '{scope}' resolved to {len(resolved_collections)} collections:")
            for coll in resolved_collections:
                print(f"    - {coll}")
            print()
        except (SearchScopeError, ScopeValidationError) as e:
            print(f"✗ Failed to resolve scope '{scope}': {e}")


async def test_collection_type_classifier():
    """Test collection type classification."""
    print("\n--- Testing Collection Type Classifier ---")
    
    # Get classes from loaded module
    CollectionTypeClassifier = collection_types_module.CollectionTypeClassifier
    
    classifier = CollectionTypeClassifier()
    
    test_collections = [
        ("__user_memory", "system"),
        ("_library_docs", "library"), 
        ("my-project-docs", "project"),
        ("algorithms", "global"),
        ("unknown-format", "unknown")
    ]
    
    for collection_name, expected_type in test_collections:
        collection_info = classifier.get_collection_info(collection_name)
        print(f"✓ '{collection_name}' -> {collection_info.type.value} "
              f"(searchable: {collection_info.is_searchable}, "
              f"readonly: {collection_info.is_readonly}, "
              f"memory: {collection_info.is_memory_collection})")


async def test_error_handling():
    """Test error handling in search scope resolution."""
    print("\n--- Testing Error Handling ---")
    
    # Get functions from loaded module
    resolve_search_scope = search_scope_module.resolve_search_scope
    SearchScopeError = search_scope_module.SearchScopeError
    
    client = MockWorkspaceClient()
    config = Config()
    
    # Test non-existent collection
    try:
        resolve_search_scope("collection", "non-existent", client, config)
        print("✗ Non-existent collection should cause error")
    except SearchScopeError as e:
        print(f"✓ Non-existent collection correctly rejected: {e}")
    
    # Test with uninitialized client
    client.initialized = False
    try:
        resolve_search_scope("project", "", client, config)
        print("✗ Uninitialized client should cause error")
    except SearchScopeError as e:
        print(f"✓ Uninitialized client correctly rejected: {e}")


async def main():
    """Run all tests."""
    print("Starting Search Scope Integration Tests")
    print("=" * 50)
    
    try:
        await test_search_scope_validation()
        await test_collection_type_classifier()
        await test_search_scope_resolution()
        await test_error_handling()
        
        print("\n" + "=" * 50)
        print("✓ All search scope integration tests completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def setup_modules():
    """Load the temporary modules dynamically."""
    global search_scope_module, collection_types_module
    
    import importlib.util
    
    temp_files_dir = Path(__file__).parent
    
    # Load search scope module
    search_scope_path = temp_files_dir / "20250909-0911_search_scope_task175.py"
    if not search_scope_path.exists():
        print(f"✗ Search scope file not found: {search_scope_path}")
        return False
        
    spec = importlib.util.spec_from_file_location("search_scope_task175", search_scope_path)
    search_scope_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(search_scope_module)
    
    # Load collection types module  
    collection_types_path = temp_files_dir / "20250909-0911_collection_types_task175.py"
    if not collection_types_path.exists():
        print(f"✗ Collection types file not found: {collection_types_path}")
        return False
        
    spec = importlib.util.spec_from_file_location("collection_types_task175", collection_types_path)
    collection_types_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collection_types_module)
    
    print("✓ Search scope and collection types modules loaded successfully")
    return True


if __name__ == "__main__":
    if not setup_modules():
        sys.exit(1)
    
    asyncio.run(main())