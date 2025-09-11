#!/usr/bin/env python3
"""Quick test of service commands to understand current failures."""
import subprocess
import sys
import time
from pathlib import Path

def run_cmd(cmd):
    """Run a command and return result."""
    print(f"🔍 Testing: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=15
        )
        print(f"   Return code: {result.returncode}")
        if result.stdout:
            print(f"   STDOUT: {result.stdout[:200]}")
        if result.stderr:
            print(f"   STDERR: {result.stderr[:200]}")
        print()
        return result
    except Exception as e:
        print(f"   ERROR: {e}")
        print()
        return None

def main():
    wqm = "/Users/chris/.local/bin/wqm"
    
    if not Path(wqm).exists():
        print(f"❌ {wqm} not found")
        return 1
        
    print("🔬 Quick service command testing...")
    print("="*50)
    
    # Test basic commands
    commands = [
        [wqm, "service", "status"],
        [wqm, "service", "uninstall"],
        [wqm, "service", "install"],
        [wqm, "service", "status"],
        [wqm, "service", "start"],
        [wqm, "service", "status"],
    ]
    
    for cmd in commands:
        run_cmd(cmd)
        time.sleep(2)
    
    print("✅ Quick test complete")

if __name__ == "__main__":
    main()