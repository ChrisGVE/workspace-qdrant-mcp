#!/usr/bin/env python3

import subprocess
import os

os.chdir("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")

print("Quick WQM Service Check")
print("=" * 30)

# Test if wqm command works at all
try:
    result = subprocess.run(["uv", "run", "wqm", "--help"], 
                          capture_output=True, text=True, timeout=10)
    print("WQM Command Test:")
    print(f"  Exit code: {result.returncode}")
    if result.returncode == 0:
        print("  ✓ wqm command available")
    else:
        print("  ✗ wqm command failed")
        print(f"  Error: {result.stderr}")
except Exception as e:
    print(f"  ✗ Exception: {e}")

print()

# Test service subcommand
try:
    result = subprocess.run(["uv", "run", "wqm", "service", "--help"], 
                          capture_output=True, text=True, timeout=10)
    print("WQM Service Subcommand Test:")
    print(f"  Exit code: {result.returncode}")
    if result.returncode == 0:
        print("  ✓ service subcommand available")
        print("  Available commands:")
        for line in result.stdout.split('\n'):
            if 'install' in line or 'start' in line or 'stop' in line or 'status' in line:
                print(f"    {line.strip()}")
    else:
        print("  ✗ service subcommand failed")  
        print(f"  Error: {result.stderr}")
except Exception as e:
    print(f"  ✗ Exception: {e}")

print()

# Quick status test
print("Quick Status Test:")
try:
    result = subprocess.run(["uv", "run", "wqm", "service", "status"], 
                          capture_output=True, text=True, timeout=15)
    print(f"  Exit code: {result.returncode}")
    print(f"  Output: {result.stdout[:200]}")
    if result.stderr:
        print(f"  Error: {result.stderr[:200]}")
except Exception as e:
    print(f"  Exception: {e}")

print("\nQuick check complete.")