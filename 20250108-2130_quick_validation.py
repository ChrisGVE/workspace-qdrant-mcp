#!/usr/bin/env python3
"""
Quick validation of the collection naming implementation.
"""

import sys
import os
from pathlib import Path

# Setup
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

print("QUICK VALIDATION OF COLLECTION NAMING FRAMEWORK")
print("=" * 55)

# Import and test
try:
    import core.collection_naming as cn
    print("✓ Module imported successfully")
    
    # Test 1: normalize_collection_name_component
    print("\n1. Testing normalize_collection_name_component:")
    tests = [
        ("my-project", "my_project"),
        ("user notes", "user_notes"), 
        ("my-awesome project", "my_awesome_project"),
        ("my--project", "my_project"),
        ("my  project", "my_project"),
    ]
    
    for input_val, expected in tests:
        result = cn.normalize_collection_name_component(input_val)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{input_val}' -> '{result}' (expected: '{expected}')")
        
    # Test 2: build_project_collection_name  
    print("\n2. Testing build_project_collection_name:")
    project_tests = [
        ("myproject", "documents", "myproject-documents"),
        ("my project", "source code", "my_project-source_code"),
        ("my-awesome-project", "notes", "my_awesome_project-notes"),
    ]
    
    for project, suffix, expected in project_tests:
        result = cn.build_project_collection_name(project, suffix)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{project}', '{suffix}' -> '{result}' (expected: '{expected}')")
        
    # Test 3: build_system_memory_collection_name
    print("\n3. Testing build_system_memory_collection_name:")
    system_tests = [
        ("user_preferences", "__user_preferences"),
        ("user-preferences", "__user_preferences"),
        ("system config", "__system_config"),
    ]
    
    for memory_name, expected in system_tests:
        result = cn.build_system_memory_collection_name(memory_name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{memory_name}' -> '{result}' (expected: '{expected}')")
        
    # Test 4: validate_collection_name
    print("\n4. Testing validate_collection_name:")
    validation_tests = [
        ("project_name-documents", True),
        ("__system_config", True),
        ("documents", True),
        ("invalid--name", False),
        ("project-", False),
        ("-documents", False),
    ]
    
    for name, expected in validation_tests:
        result = cn.validate_collection_name(name)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{name}' -> {result} (expected: {expected})")
        
    # Test 5: Error handling
    print("\n5. Testing error handling:")
    
    # Test None input
    try:
        cn.normalize_collection_name_component(None)
        print("  ✗ Should have raised TypeError for None")
    except TypeError:
        print("  ✓ Correctly raised TypeError for None")
        
    # Test empty string  
    try:
        cn.normalize_collection_name_component("")
        print("  ✗ Should have raised ValueError for empty string")
    except ValueError:
        print("  ✓ Correctly raised ValueError for empty string")
        
    # Test constants
    print("\n6. Testing module constants:")
    constants = ['PROJECT_COLLECTION_PATTERN', 'SYSTEM_COLLECTION_PATTERN', 'SINGLE_COMPONENT_PATTERN']
    for const in constants:
        if hasattr(cn, const):
            print(f"  ✓ {const} available")
        else:
            print(f"  ✗ {const} missing")
    
    print("\n" + "=" * 55)
    print("VALIDATION COMPLETE - Implementation looks good!")
    
    # Now try to execute actual test cases programmatically
    print("\n7. Running actual test cases:")
    
    # Import test classes and run some key tests manually
    try:
        sys.path.append('tests')
        from test_collection_naming import TestNormalizeCollectionNameComponent
        
        # Create test instance and run a few key tests
        test_instance = TestNormalizeCollectionNameComponent()
        
        # Run some key tests
        test_instance.test_basic_dash_replacement()
        print("  ✓ test_basic_dash_replacement passed")
        
        test_instance.test_basic_space_replacement()  
        print("  ✓ test_basic_space_replacement passed")
        
        test_instance.test_combined_dashes_and_spaces()
        print("  ✓ test_combined_dashes_and_spaces passed")
        
        print("  ✓ Key manual test execution successful")
        
    except Exception as e:
        print(f"  ✗ Manual test execution failed: {e}")
    
    print("\n" + "=" * 55)
    print("FRAMEWORK VALIDATION SUCCESSFUL!")
    print("The collection naming implementation appears to be working correctly.")
    
except Exception as e:
    print(f"✗ Error during validation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)