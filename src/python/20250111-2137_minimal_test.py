#!/usr/bin/env python3

import subprocess
import os

# Change to project root
os.chdir("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")

print("MINIMAL WQM SERVICE TEST")
print("Date:", subprocess.run(["date"], capture_output=True, text=True).stdout.strip())
print("Working dir:", os.getcwd())
print()

def run_test(cmd_list, name):
    print(f"--- {name} ---")
    print(f"Command: {' '.join(cmd_list)}")
    try:
        result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=20)
        print(f"Exit code: {result.returncode}")
        
        if result.stdout:
            print(f"Output: {result.stdout[:200]}")
        if result.stderr:
            print(f"Error: {result.stderr[:200]}")
            
        return result.returncode == 0
    except Exception as e:
        print(f"Failed: {e}")
        return False

# Test sequence
tests = [
    (["uv", "run", "wqm", "--help"], "WQM Help"),
    (["uv", "run", "wqm", "service", "--help"], "Service Help"),  
    (["uv", "run", "wqm", "service", "status"], "Initial Status"),
    (["uv", "run", "wqm", "service", "install"], "Install"),
    (["uv", "run", "wqm", "service", "status"], "Status After Install"),
    (["uv", "run", "wqm", "service", "start"], "Start"),
    (["uv", "run", "wqm", "service", "status"], "Status After Start"),
    (["uv", "run", "wqm", "service", "stop"], "Stop"),
    (["uv", "run", "wqm", "service", "uninstall"], "Uninstall")
]

results = []
for cmd, name in tests:
    success = run_test(cmd, name)
    results.append((name, success))
    print()

print("SUMMARY:")
for name, success in results:
    status = "✓" if success else "✗"
    print(f"{status} {name}")

success_count = sum(1 for _, success in results if success)
print(f"\nTotal: {len(results)}, Success: {success_count}, Failed: {len(results) - success_count}")

# Save results
with open("src/python/20250111-2137_minimal_test_results.txt", "w") as f:
    f.write("WQM Service Test Results\n")
    f.write("=" * 30 + "\n\n")
    for name, success in results:
        f.write(f"{'PASS' if success else 'FAIL'}: {name}\n")
    f.write(f"\nSummary: {success_count}/{len(results)} tests passed\n")

print("Results saved to: src/python/20250111-2137_minimal_test_results.txt")