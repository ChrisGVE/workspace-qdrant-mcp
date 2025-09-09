#!/usr/bin/env python3
"""
Simple test for search scope implementation validation.
"""

import sys
from pathlib import Path
import inspect

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_imports():
    """Test that we can import our search scope components."""
    try:
        # Import search scope components
        from workspace_qdrant_mcp.tools.simplified_interface import (
            SearchScope,
            validate_search_scope,
            resolve_search_scope,
            SimplifiedToolsRouter
        )
        print("✓ Successfully imported search scope components")
        return True
    except ImportError as e:
        print(f"✗ Failed to import components: {e}")
        return False

def test_function_signature():
    """Test that qdrant_find has the correct signature with search_scope parameter."""
    try:
        from workspace_qdrant_mcp.tools.simplified_interface import SimplifiedToolsRouter
        
        # Get the qdrant_find method signature
        qdrant_find_method = SimplifiedToolsRouter.qdrant_find
        sig = inspect.signature(qdrant_find_method)
        
        # Check if search_scope parameter exists
        if 'search_scope' not in sig.parameters:
            print("✗ search_scope parameter not found in qdrant_find signature")
            return False
        
        # Check the default value
        search_scope_param = sig.parameters['search_scope']
        if search_scope_param.default != "project":
            print(f"✗ search_scope default value should be 'project', got '{search_scope_param.default}'")
            return False
        
        print("✓ qdrant_find signature correctly includes search_scope parameter with default 'project'")
        print(f"  Parameters: {list(sig.parameters.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking function signature: {e}")
        return False

def test_search_scope_enum():
    """Test that SearchScope enum has correct values."""
    try:
        from workspace_qdrant_mcp.tools.simplified_interface import SearchScope
        
        expected_scopes = ["collection", "project", "workspace", "all", "memory"]
        actual_scopes = [scope.value for scope in SearchScope]
        
        if set(actual_scopes) != set(expected_scopes):
            print(f"✗ SearchScope enum values mismatch. Expected: {expected_scopes}, Got: {actual_scopes}")
            return False
        
        print("✓ SearchScope enum has correct values")
        print(f"  Available scopes: {actual_scopes}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking SearchScope enum: {e}")
        return False

def test_validation_function():
    """Test basic validation function behavior."""
    try:
        from workspace_qdrant_mcp.tools.simplified_interface import (
            validate_search_scope, 
            ScopeValidationError
        )
        
        # Valid scope should not raise exception
        validate_search_scope("project", "")
        validate_search_scope("collection", "test-collection")
        print("✓ validate_search_scope accepts valid inputs")
        
        # Invalid scope should raise exception
        try:
            validate_search_scope("invalid", "test")
            print("✗ validate_search_scope should reject invalid scope")
            return False
        except ScopeValidationError:
            print("✓ validate_search_scope correctly rejects invalid scope")
        
        # Collection scope without collection should raise exception
        try:
            validate_search_scope("collection", "")
            print("✗ validate_search_scope should require collection for 'collection' scope")
            return False
        except ScopeValidationError:
            print("✓ validate_search_scope correctly requires collection for 'collection' scope")
        
        return True
    except Exception as e:
        print(f"✗ Error testing validation function: {e}")
        return False

def test_file_modification():
    """Test that our changes are present in the source file."""
    try:
        file_path = src_path / "workspace_qdrant_mcp" / "tools" / "simplified_interface.py"
        
        if not file_path.exists():
            print(f"✗ Source file not found: {file_path}")
            return False
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check for key additions
        checks = [
            ("SearchScope enum", "class SearchScope(Enum):"),
            ("validate_search_scope function", "def validate_search_scope("),
            ("resolve_search_scope function", "def resolve_search_scope("),
            ("search_scope parameter", "search_scope: str = \"project\""),
            ("scope resolution logic", "resolve_search_scope("),
        ]
        
        all_found = True
        for name, pattern in checks:
            if pattern in content:
                print(f"✓ Found {name}")
            else:
                print(f"✗ Missing {name}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"✗ Error checking file modifications: {e}")
        return False

def main():
    """Run all simple tests."""
    print("Simple Search Scope Implementation Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Function Signature Test", test_function_signature), 
        ("SearchScope Enum Test", test_search_scope_enum),
        ("Validation Function Test", test_validation_function),
        ("File Modification Test", test_file_modification)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    
    if all(results):
        print("✓ ALL TESTS PASSED!")
        print("✓ Search scope implementation appears to be working correctly")
        print("✓ Task 175: Search Scope Architecture for qdrant_find - IMPLEMENTED")
        return True
    else:
        failed_count = len([r for r in results if not r])
        print(f"✗ {failed_count} test(s) failed")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)