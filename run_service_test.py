#!/usr/bin/env python3
"""Quick service test runner to execute our comprehensive tests."""
import subprocess
import sys
from pathlib import Path

def main():
    test_script = Path("20250111-1643_service_testing_plan.py")
    
    if not test_script.exists():
        print("❌ Test script not found")
        return 1
        
    print("🚀 Running comprehensive service tests...")
    result = subprocess.run([sys.executable, str(test_script)], cwd=".")
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())