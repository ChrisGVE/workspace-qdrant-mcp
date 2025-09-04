#!/usr/bin/env python3
"""
Execute Task 91: Integration Test Suite Execution
Comprehensive integration test execution with detailed reporting and analysis.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Execute comprehensive integration test suite."""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🎯 Task 91: Integration Test Suite Execution")
    print("=" * 80)
    print(f"Project Root: {project_root}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Run the comprehensive integration test runner
    try:
        result = subprocess.run([
            sys.executable, "run_integration_tests.py"
        ], cwd=project_root)
        
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"❌ Failed to execute integration tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()