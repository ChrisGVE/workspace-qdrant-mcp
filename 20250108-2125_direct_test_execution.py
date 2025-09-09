#!/usr/bin/env python3
"""
Direct test execution for collection naming framework.
"""

import sys
import os
from pathlib import Path

# Set working directory 
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print(f"Working directory: {project_root}")
print(f"Python executable: {sys.executable}")
print("Starting tests...")
print("=" * 50)

try:
    # Import the test module to verify imports work
    from tests.test_collection_naming import *
    print("✓ Test imports successful")
    
    # Import the module being tested
    import core.collection_naming as cn
    print("✓ Core module import successful")
    
    # Run a quick smoke test
    result = cn.normalize_collection_name_component("test-name")
    print(f"✓ Smoke test successful: 'test-name' -> '{result}'")
    
    print("=" * 50)
    print("All imports and basic functionality verified.")
    print("Running full test suite with pytest...")
    print("=" * 50)
    
    # Now run pytest
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        "tests/test_collection_naming.py", 
        "-v", "--tb=short", "--no-header"
    ], cwd=project_root)
    
    print("=" * 50)
    print(f"Tests completed with return code: {result.returncode}")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)