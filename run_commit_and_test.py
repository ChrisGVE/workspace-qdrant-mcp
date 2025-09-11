#!/usr/bin/env python3
"""Run commit and test."""
import subprocess
import sys

def main():
    print("🔧 Committing fix...")
    subprocess.run([sys.executable, "commit_fix.py"])
    
    print("\n🧪 Testing fixed service...")
    result = subprocess.run([sys.executable, "20250111-1705_test_fixed_service.py"])
    
    print(f"\n✅ Complete! Exit code: {result.returncode}")
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())