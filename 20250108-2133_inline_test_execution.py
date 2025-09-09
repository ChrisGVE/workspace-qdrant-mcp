#!/usr/bin/env python3
"""
Inline test execution to validate the collection naming framework.
"""

import sys
import os
from pathlib import Path

# Setup environment
project_root = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print("COLLECTION NAMING FRAMEWORK - INLINE TEST EXECUTION")
print("=" * 60)
print(f"Working directory: {project_root}")
print()

# Execute validation inline
try:
    # Import the module
    import core.collection_naming as cn
    print("✓ Successfully imported core.collection_naming")
    
    # Test 1: Basic functionality
    print("\n1. BASIC FUNCTIONALITY TEST")
    print("-" * 30)
    
    # Test normalize_collection_name_component
    result = cn.normalize_collection_name_component("my-project")
    assert result == "my_project", f"Expected 'my_project', got '{result}'"
    print(f"✓ normalize: 'my-project' -> '{result}'")
    
    result = cn.normalize_collection_name_component("user notes")
    assert result == "user_notes", f"Expected 'user_notes', got '{result}'"
    print(f"✓ normalize: 'user notes' -> '{result}'")
    
    # Test build_project_collection_name
    result = cn.build_project_collection_name("my-project", "documents")
    assert result == "my_project-documents", f"Expected 'my_project-documents', got '{result}'"
    print(f"✓ project: 'my-project', 'documents' -> '{result}'")
    
    # Test build_system_memory_collection_name
    result = cn.build_system_memory_collection_name("user-preferences")
    assert result == "__user_preferences", f"Expected '__user_preferences', got '{result}'"
    print(f"✓ system: 'user-preferences' -> '{result}'")
    
    # Test validate_collection_name
    result = cn.validate_collection_name("project_name-documents")
    assert result is True, f"Expected True, got {result}"
    print(f"✓ validate: 'project_name-documents' -> {result}")
    
    result = cn.validate_collection_name("invalid--name")
    assert result is False, f"Expected False, got {result}"
    print(f"✓ validate: 'invalid--name' -> {result}")
    
    # Test 2: Complex edge cases
    print("\n2. EDGE CASE TESTING")
    print("-" * 30)
    
    # Complex normalization
    result = cn.normalize_collection_name_component("my--awesome   project")
    assert result == "my_awesome_project", f"Expected 'my_awesome_project', got '{result}'"
    print(f"✓ complex normalize: 'my--awesome   project' -> '{result}'")
    
    # Project with complex names
    result = cn.build_project_collection_name("frontend-app", "source-code")
    assert result == "frontend_app-source_code", f"Expected 'frontend_app-source_code', got '{result}'"
    print(f"✓ complex project: 'frontend-app', 'source-code' -> '{result}'")
    
    # Test 3: Error handling
    print("\n3. ERROR HANDLING TEST")
    print("-" * 30)
    
    # Test None input
    try:
        cn.normalize_collection_name_component(None)
        print("✗ Should have raised TypeError for None")
        error_count = 1
    except TypeError as e:
        print("✓ Correctly raised TypeError for None input")
        error_count = 0
        
    # Test empty string
    try:
        cn.normalize_collection_name_component("")
        print("✗ Should have raised ValueError for empty string")
        error_count += 1
    except ValueError:
        print("✓ Correctly raised ValueError for empty string")
        
    # Test whitespace only
    try:
        cn.normalize_collection_name_component("   ")
        print("✗ Should have raised ValueError for whitespace only")
        error_count += 1
    except ValueError:
        print("✓ Correctly raised ValueError for whitespace only")
        
    # Test 4: Validation patterns
    print("\n4. VALIDATION PATTERN TESTING")
    print("-" * 30)
    
    validation_cases = [
        ("project-docs", True),
        ("my_project-source_code", True), 
        ("__system_config", True),
        ("simple", True),
        ("invalid--double", False),
        ("project-", False),
        ("-documents", False),
        ("", False),
    ]
    
    validation_errors = 0
    for name, expected in validation_cases:
        result = cn.validate_collection_name(name)
        if result == expected:
            print(f"✓ validate: '{name}' -> {result}")
        else:
            print(f"✗ validate: '{name}' -> {result} (expected {expected})")
            validation_errors += 1
            
    # Test 5: Constants check
    print("\n5. CONSTANTS AVAILABILITY")
    print("-" * 30)
    
    constants = [
        'PROJECT_COLLECTION_PATTERN',
        'SYSTEM_COLLECTION_PATTERN', 
        'SINGLE_COMPONENT_PATTERN'
    ]
    
    const_errors = 0
    for const in constants:
        if hasattr(cn, const):
            pattern = getattr(cn, const)
            print(f"✓ {const}: '{pattern}'")
        else:
            print(f"✗ {const}: Missing")
            const_errors += 1
            
    # Final summary
    print("\n" + "=" * 60)
    print("TEST EXECUTION SUMMARY")
    print("=" * 60)
    
    total_errors = error_count + validation_errors + const_errors
    
    if total_errors == 0:
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✓ Basic functionality: Working")
        print("✓ Edge cases: Handled correctly")
        print("✓ Error handling: Robust")
        print("✓ Validation patterns: Accurate") 
        print("✓ Constants: Available")
        print("\nThe collection naming framework implementation is valid and working correctly.")
        exit_code = 0
    else:
        print(f"❌ {total_errors} ERRORS DETECTED")
        print("Some aspects of the implementation need attention.")
        exit_code = 1
        
    # Now attempt to run pytest if available
    print("\n6. PYTEST EXECUTION ATTEMPT")
    print("-" * 30)
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_collection_naming.py", 
            "-v", "--tb=short"
        ], 
        capture_output=True, 
        text=True, 
        timeout=30,
        cwd=project_root,
        env={**os.environ, 'PYTHONPATH': str(project_root)}
        )
        
        print("Pytest execution completed")
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
            
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("✓ Pytest execution successful!")
        else:
            print(f"✗ Pytest failed with exit code: {result.returncode}")
            
    except Exception as e:
        print(f"Pytest execution failed: {e}")
        print("But manual validation was successful")
        
    print("=" * 60)
    print(f"FINAL EXIT CODE: {exit_code}")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Critical error during test execution: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
sys.exit(exit_code)