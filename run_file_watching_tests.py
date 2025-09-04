#!/usr/bin/env python3
"""
Quick test runner to execute the file watching tests and validate functionality.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_tests():
    """Run the file watching tests."""
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Set PYTHONPATH to include src directory
    env = os.environ.copy()
    src_path = str(project_dir / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path
    
    # Run the comprehensive tests
    print("Running comprehensive file watching tests for Task 87...")
    print("=" * 60)
    
    test_file = "tests/test_file_watching_comprehensive.py"
    
    # Check if pytest is available
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", test_file,
            "-v", "--tb=short", "--maxfail=3"
        ], env=env, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nTest Result: {'PASSED' if result.returncode == 0 else 'FAILED'}")
        print(f"Return Code: {result.returncode}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)