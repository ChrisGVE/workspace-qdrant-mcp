#!/usr/bin/env python3
"""
Test script to validate that CI workflow configuration issues are resolved.

This script tests:
1. pytest can collect tests without import errors
2. Basic unit tests can run successfully
3. No more "No module named 'tests'" or "No module named 'src'" errors
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return success status."""
    print(f"\n🧪 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if result.stdout:
                print("Output:", result.stdout[:500])
        else:
            print(f"❌ FAILED: {description} (exit code: {result.returncode})")
            if result.stderr:
                print("Error:", result.stderr[:500])
                
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False


def main():
    """Run the validation tests."""
    print("🚀 Testing CI Workflow Configuration Fixes")
    print("=" * 80)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    import os
    os.chdir(project_dir)
    
    # Test cases
    tests = [
        {
            "cmd": ["python", "-m", "pytest", "--collect-only", "-q"],
            "description": "Test collection without import errors"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/unit/test_cli_parsers.py", "-v", "--tb=short", "--no-cov"],
            "description": "Run basic unit tests that should always pass"
        },
        {
            "cmd": ["python", "-c", "import tests.fixtures.test_data_collector; import src.workspace_qdrant_mcp.server; print('✅ Import test passed')"],
            "description": "Test critical imports work correctly"
        },
        {
            "cmd": ["curl", "-f", "http://httpbin.org/status/200"],  # Test curl works
            "description": "Test curl command availability (for health checks)"
        }
    ]
    
    results = []
    for test in tests:
        success = run_command(test["cmd"], test["description"])
        results.append(success)
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY:")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    for i, test in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} - {test['description']}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed! CI fixes are working.")
        return 0
    else:
        print(f"\n⚠️  Some issues remain. Check the failed tests above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())