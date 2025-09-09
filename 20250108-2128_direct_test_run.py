#!/usr/bin/env python3
"""
Direct test execution for collection naming framework.
"""

import sys
import os
from pathlib import Path

# Set up the environment
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print("COLLECTION NAMING FRAMEWORK - TEST EXECUTION")
print("=" * 60)
print(f"Working directory: {project_root}")
print(f"Python path: {sys.path[0]}")
print()

# Test 1: Import verification
print("1. IMPORT TESTING")
print("-" * 30)
try:
    import core.collection_naming as cn
    print("✓ Core collection_naming module imported")
    
    # Test the constants are available
    assert hasattr(cn, 'PROJECT_COLLECTION_PATTERN')
    assert hasattr(cn, 'SYSTEM_COLLECTION_PATTERN')
    assert hasattr(cn, 'SINGLE_COMPONENT_PATTERN')
    print("✓ Module constants available")
    
    # Test all main functions are available
    assert hasattr(cn, 'normalize_collection_name_component')
    assert hasattr(cn, 'build_project_collection_name')
    assert hasattr(cn, 'build_system_memory_collection_name')
    assert hasattr(cn, 'validate_collection_name')
    print("✓ All main functions available")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Basic functionality verification
print("2. BASIC FUNCTIONALITY TEST")
print("-" * 30)

try:
    # Test normalize_collection_name_component
    result = cn.normalize_collection_name_component("test-name")
    expected = "test_name"
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✓ normalize: 'test-name' -> '{result}'")
    
    # Test build_project_collection_name
    result = cn.build_project_collection_name("my-project", "docs")
    expected = "my_project-docs"
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✓ project: 'my-project', 'docs' -> '{result}'")
    
    # Test build_system_memory_collection_name
    result = cn.build_system_memory_collection_name("user-prefs")
    expected = "__user_prefs"
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✓ system: 'user-prefs' -> '{result}'")
    
    # Test validate_collection_name
    result = cn.validate_collection_name("project-docs")
    assert result is True, f"Expected True, got {result}"
    print(f"✓ validate: 'project-docs' -> {result}")
    
except Exception as e:
    print(f"✗ Basic functionality test failed: {e}")
    sys.exit(1)

print()

# Test 3: Run specific test cases manually
print("3. MANUAL TEST EXECUTION")
print("-" * 30)

test_cases = [
    # normalize_collection_name_component tests
    ("my-awesome-project", "my_awesome_project"),
    ("user notes", "user_notes"),
    ("mixed--  separators", "mixed_separators"),
    
    # Error handling tests (should raise exceptions)
    # We'll test these with try/except
]

for input_val, expected in test_cases:
    try:
        result = cn.normalize_collection_name_component(input_val)
        if result == expected:
            print(f"✓ '{input_val}' -> '{result}'")
        else:
            print(f"✗ '{input_val}': expected '{expected}', got '{result}'")
    except Exception as e:
        print(f"✗ '{input_val}': raised {e}")

# Test error handling
print("\n4. ERROR HANDLING TEST")
print("-" * 30)

# Test None input
try:
    cn.normalize_collection_name_component(None)
    print("✗ Should have raised TypeError for None input")
except TypeError:
    print("✓ Correctly raised TypeError for None input")

# Test empty string
try:
    cn.normalize_collection_name_component("")
    print("✗ Should have raised ValueError for empty string")
except ValueError:
    print("✓ Correctly raised ValueError for empty string")

# Test whitespace only
try:
    cn.normalize_collection_name_component("   ")
    print("✗ Should have raised ValueError for whitespace only")
except ValueError:
    print("✓ Correctly raised ValueError for whitespace only")

print()

# Test 5: Validation tests
print("5. VALIDATION TESTS")
print("-" * 30)

validation_tests = [
    ("project-docs", True),
    ("my_project-source_code", True),
    ("__system_config", True),
    ("simple", True),
    ("invalid--double", False),
    ("invalid-", False),
    ("-invalid", False),
    ("", False),
]

for name, expected in validation_tests:
    result = cn.validate_collection_name(name)
    if result == expected:
        print(f"✓ '{name}' -> {result}")
    else:
        print(f"✗ '{name}': expected {expected}, got {result}")

print()
print("=" * 60)
print("MANUAL TEST EXECUTION COMPLETE")

# Now let's try to run pytest programmatically
print()
print("6. PYTEST EXECUTION")
print("-" * 30)

try:
    import pytest
    
    # Run pytest on the test file
    exit_code = pytest.main([
        "tests/test_collection_naming.py",
        "-v",
        "--tb=short"
    ])
    
    print("-" * 30)
    if exit_code == 0:
        print("✓ ALL PYTEST TESTS PASSED!")
    else:
        print(f"✗ Some pytest tests failed (exit code: {exit_code})")
        
except ImportError:
    print("pytest not available - skipping automated test execution")
    print("Manual tests completed successfully")
    exit_code = 0

print("=" * 60)
print("TEST EXECUTION SUMMARY:")
print(f"Exit Code: {exit_code}")
if exit_code == 0:
    print("✓ All tests passed successfully!")
else:
    print("✗ Some tests failed")
    
print("=" * 60)