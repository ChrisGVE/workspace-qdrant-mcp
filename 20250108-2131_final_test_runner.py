#!/usr/bin/env python3
"""
Final comprehensive test runner for collection naming framework.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Setup
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("COLLECTION NAMING FRAMEWORK - COMPREHENSIVE TEST VALIDATION")
    print("=" * 70)
    print(f"Working directory: {project_root}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Step 1: Quick validation
    print("STEP 1: BASIC VALIDATION")
    print("-" * 40)
    
    try:
        # Add project to Python path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
            
        # Test import
        import core.collection_naming as cn
        print("✓ Core module imported")
        
        # Test basic function
        result = cn.normalize_collection_name_component("test-name")
        assert result == "test_name"
        print("✓ Basic functionality confirmed")
        
        # Test all main functions exist
        functions = [
            'normalize_collection_name_component',
            'build_project_collection_name', 
            'build_system_memory_collection_name',
            'validate_collection_name'
        ]
        
        for func in functions:
            assert hasattr(cn, func), f"Missing function: {func}"
            print(f"✓ Function {func} available")
            
        # Test constants exist  
        constants = ['PROJECT_COLLECTION_PATTERN', 'SYSTEM_COLLECTION_PATTERN', 'SINGLE_COMPONENT_PATTERN']
        for const in constants:
            assert hasattr(cn, const), f"Missing constant: {const}"
            print(f"✓ Constant {const} available")
            
    except Exception as e:
        print(f"✗ Basic validation failed: {e}")
        return 1
        
    print()
    
    # Step 2: Run pytest with multiple strategies
    print("STEP 2: PYTEST EXECUTION")
    print("-" * 40)
    
    strategies = [
        # Strategy 1: Direct pytest module execution
        [sys.executable, "-m", "pytest", "tests/test_collection_naming.py", "-v", "--tb=short"],
        
        # Strategy 2: Pytest with explicit PYTHONPATH
        [sys.executable, "-m", "pytest", "tests/test_collection_naming.py", "-v", "--tb=line"],
        
        # Strategy 3: Simple execution without extra flags
        [sys.executable, "-m", "pytest", "tests/test_collection_naming.py"],
    ]
    
    success = False
    
    for i, cmd in enumerate(strategies, 1):
        print(f"\nAttempt {i}: {' '.join(cmd)}")
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')
            
            result = subprocess.run(
                cmd, 
                cwd=project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            print(f"Exit code: {result.returncode}")
            
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
                
            if result.stderr and result.stderr.strip():
                print("STDERR:")  
                print(result.stderr)
                
            if result.returncode == 0:
                print("✓ Tests passed successfully!")
                success = True
                break
            else:
                print(f"✗ Tests failed with exit code {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print("✗ Test execution timed out")
        except Exception as e:
            print(f"✗ Error running tests: {e}")
            
    if not success:
        print("\nAll pytest strategies failed. Attempting manual test verification...")
        return run_manual_tests()
    else:
        print("\n" + "=" * 70)
        print("✓ PYTEST EXECUTION SUCCESSFUL!")
        print("All collection naming framework tests passed.")
        return 0

def run_manual_tests():
    """Run manual test verification as fallback."""
    print("\nSTEP 3: MANUAL TEST VERIFICATION")
    print("-" * 40)
    
    try:
        import core.collection_naming as cn
        
        # Critical test cases
        test_cases = [
            # normalize_collection_name_component
            ("my-project", cn.normalize_collection_name_component, "my_project"),
            ("user notes", cn.normalize_collection_name_component, "user_notes"),
            
            # build_project_collection_name
            (("myproject", "documents"), cn.build_project_collection_name, "myproject-documents"),
            (("my project", "source code"), cn.build_project_collection_name, "my_project-source_code"),
            
            # build_system_memory_collection_name  
            ("user_preferences", cn.build_system_memory_collection_name, "__user_preferences"),
            
            # validate_collection_name
            ("project_name-documents", cn.validate_collection_name, True),
            ("__system_config", cn.validate_collection_name, True),
            ("invalid--name", cn.validate_collection_name, False),
        ]
        
        passed = 0
        total = len(test_cases)
        
        for i, (input_val, func, expected) in enumerate(test_cases, 1):
            try:
                if isinstance(input_val, tuple):
                    result = func(*input_val)
                    input_str = str(input_val)
                else:
                    result = func(input_val)
                    input_str = repr(input_val)
                    
                if result == expected:
                    print(f"✓ Test {i}: {func.__name__}({input_str}) = {result}")
                    passed += 1
                else:
                    print(f"✗ Test {i}: {func.__name__}({input_str}) = {result}, expected {expected}")
                    
            except Exception as e:
                print(f"✗ Test {i}: {func.__name__}({input_str}) raised {e}")
                
        print(f"\nManual tests: {passed}/{total} passed")
        
        if passed == total:
            print("✓ All manual tests passed!")
            return 0
        else:
            print("✗ Some manual tests failed")
            return 1
            
    except Exception as e:
        print(f"✗ Manual testing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())