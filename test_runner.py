#!/usr/bin/env python3
"""
Test runner for multi-component communication integration tests.
"""
import asyncio
import subprocess
import sys
from pathlib import Path

async def run_multi_component_tests():
    """Run the multi-component communication tests."""
    test_file = Path(__file__).parent / "tests/integration/test_multi_component_communication.py"
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
        
    print("🧪 Running multi-component communication integration tests...")
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            str(test_file), 
            "-v", "--tb=short", "-s"
        ], cwd=Path(__file__).parent, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Multi-component communication tests passed!")
            print("\n=== Test Output ===")
            print(result.stdout)
            return True
        else:
            print("❌ Multi-component communication tests failed!")
            print("\n=== STDOUT ===")
            print(result.stdout)
            print("\n=== STDERR ===")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_multi_component_tests())
    sys.exit(0 if success else 1)