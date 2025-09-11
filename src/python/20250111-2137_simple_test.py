#!/usr/bin/env python3
import subprocess
import os

# Change to project root
os.chdir("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")

print("Testing wqm service commands...")

# Test 1: Basic status check
print("\n=== TEST 1: Status Check ===")
try:
    result = subprocess.run(["uv", "run", "wqm", "service", "status"], 
                          capture_output=True, text=True, timeout=10)
    print(f"Exit code: {result.returncode}")
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 2: Install 
print("\n=== TEST 2: Install ===")
try:
    result = subprocess.run(["uv", "run", "wqm", "service", "install"], 
                          capture_output=True, text=True, timeout=15)
    print(f"Exit code: {result.returncode}")
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 3: Status after install
print("\n=== TEST 3: Status After Install ===")
try:
    result = subprocess.run(["uv", "run", "wqm", "service", "status"], 
                          capture_output=True, text=True, timeout=10)
    print(f"Exit code: {result.returncode}")
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 4: Start service
print("\n=== TEST 4: Start Service ===")
try:
    result = subprocess.run(["uv", "run", "wqm", "service", "start"], 
                          capture_output=True, text=True, timeout=15)
    print(f"Exit code: {result.returncode}")
    print(f"Output: {result.stdout}")
    print(f"Error: {result.stderr}")
except Exception as e:
    print(f"Failed: {e}")

# Test 5: Check processes
print("\n=== TEST 5: Process Check ===")
try:
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    memexd_lines = [line for line in result.stdout.split('\n') if 'memexd' in line and 'grep' not in line]
    if memexd_lines:
        print(f"Found {len(memexd_lines)} memexd processes:")
        for line in memexd_lines:
            print(f"  {line}")
    else:
        print("No memexd processes found")
except Exception as e:
    print(f"Process check failed: {e}")

print("\nBasic testing complete.")