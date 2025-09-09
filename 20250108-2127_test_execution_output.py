#!/usr/bin/env python3
"""
Capture test execution output for collection naming framework.
"""

import subprocess
import sys
from pathlib import Path
import os

# Set up execution environment
project_root = Path(__file__).parent
os.chdir(project_root)

# Execute the test validation script and capture output
print("Executing collection naming framework test validation...")
print("=" * 70)

try:
    result = subprocess.run([
        sys.executable, 
        "20250108-2126_run_test_validation.py"
    ], 
    capture_output=True, 
    text=True,
    cwd=project_root
    )
    
    print("STDOUT:")
    print("-" * 40)
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print("-" * 40)
        print(result.stderr)
    
    print("=" * 70)
    print(f"Process exit code: {result.returncode}")
    
    # Also write results to a file for reference
    with open("20250108-2127_test_results.txt", "w") as f:
        f.write("Collection Naming Framework Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Exit Code: {result.returncode}\n\n")
        f.write("STDOUT:\n")
        f.write("-" * 20 + "\n")
        f.write(result.stdout)
        if result.stderr:
            f.write("\nSTDERR:\n")
            f.write("-" * 20 + "\n")
            f.write(result.stderr)
    
    print("Test results saved to: 20250108-2127_test_results.txt")
    
except Exception as e:
    print(f"Error executing tests: {e}")
    sys.exit(1)

sys.exit(result.returncode)