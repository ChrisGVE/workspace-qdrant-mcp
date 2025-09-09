#!/usr/bin/env python3

import os
import sys
from pathlib import Path

# Setup environment
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

print("Testing collection naming framework...")

# Test imports first
try:
    import core.collection_naming as cn
    print("✓ Core module imported")
    
    # Test basic functionality
    result = cn.normalize_collection_name_component("test-name")
    print(f"✓ Basic test: 'test-name' -> '{result}'")
    
    # Try to run pytest
    import subprocess
    result = subprocess.run([
        sys.executable, "-c", 
        """
import pytest
import sys
import os
from pathlib import Path
os.chdir(Path.cwd())
sys.path.insert(0, '.')
exit_code = pytest.main(['tests/test_collection_naming.py', '-v'])
sys.exit(exit_code)
        """
    ], capture_output=True, text=True)
    
    print("Pytest output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)
    print(f"Exit code: {result.returncode}")
    
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"Error: {e}")