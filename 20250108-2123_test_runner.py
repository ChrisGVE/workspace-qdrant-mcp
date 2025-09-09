#!/usr/bin/env python3
"""
Execute tests for the collection naming framework.
"""

import os
import sys
from pathlib import Path

# Change to project root
os.chdir(Path(__file__).parent)

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Now run the actual test file
if __name__ == "__main__":
    # Import and run tests manually to see output
    import pytest
    
    print("Running collection naming framework tests...")
    print("=" * 60)
    
    # Run pytest with verbose output
    exit_code = pytest.main([
        "tests/test_collection_naming.py",
        "-v",
        "--tb=short",
        "--no-header"
    ])
    
    print("=" * 60)
    print(f"Tests completed with exit code: {exit_code}")
    sys.exit(exit_code)