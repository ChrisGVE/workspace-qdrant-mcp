#!/usr/bin/env python3
"""
Direct execution of WQM service tests with real-time results
"""
import subprocess
import sys
import os
from datetime import datetime

# Set working directory
project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
os.chdir(project_root)

print("=== WQM SERVICE COMMAND TESTING ===")
print(f"Time: {datetime.now()}")
print(f"Directory: {os.getcwd()}")
print("=" * 50)

# Test results storage
test_results = []

def execute_test(cmd_parts, description):
    """Execute a single test and capture results"""
    print(f"\n[TEST] {description}")
    print(f"[CMD]  {' '.join(cmd_parts)}")
    print("-" * 40)
    
    try:
        # Run command
        result = subprocess.run(
            cmd_parts,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Display results immediately
        print(f"EXIT CODE: {result.returncode}")
        
        if result.stdout.strip():
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr.strip():
            print("STDERR:")
            print(result.stderr)
        
        # Check for memexd processes
        ps_result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        memexd_count = len([line for line in ps_result.stdout.split('\n') 
                           if 'memexd' in line and 'grep' not in line])
        print(f"MEMEXD PROCESSES: {memexd_count}")
        
        # Store result
        test_results.append({
            'command': ' '.join(cmd_parts),
            'description': description,
            'exit_code': result.returncode,
            'stdout': result.stdout.strip(),
            'stderr': result.stderr.strip(),
            'success': result.returncode == 0,
            'memexd_count': memexd_count
        })
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT: Command exceeded 30 seconds")
        test_results.append({
            'command': ' '.join(cmd_parts),
            'description': description,
            'exit_code': -1,
            'stdout': '',
            'stderr': 'Timeout after 30 seconds',
            'success': False,
            'memexd_count': 0
        })
        return False
        
    except Exception as e:
        print(f"EXCEPTION: {e}")
        test_results.append({
            'command': ' '.join(cmd_parts),
            'description': description,
            'exit_code': -2,
            'stdout': '',
            'stderr': f'Exception: {e}',
            'success': False,
            'memexd_count': 0
        })
        return False

# Define test sequence
test_commands = [
    # Basic functionality tests
    (["uv", "run", "wqm", "--help"], "WQM base command help"),
    (["uv", "run", "wqm", "service", "--help"], "Service subcommand help"),
    
    # Initial state
    (["uv", "run", "wqm", "service", "status"], "Initial service status"),
    
    # Clean slate setup
    (["uv", "run", "wqm", "service", "stop"], "Stop existing service (cleanup)"),
    (["uv", "run", "wqm", "service", "uninstall"], "Uninstall existing service (cleanup)"),
    (["uv", "run", "wqm", "service", "status"], "Status after cleanup"),
    
    # Fresh install workflow
    (["uv", "run", "wqm", "service", "install"], "Fresh service install"),
    (["uv", "run", "wqm", "service", "status"], "Status after install"),
    (["uv", "run", "wqm", "service", "start"], "Start installed service"),
    (["uv", "run", "wqm", "service", "status"], "Status after start"),
    
    # Service control operations
    (["uv", "run", "wqm", "service", "stop"], "Stop running service"),
    (["uv", "run", "wqm", "service", "status"], "Status after stop"),
    (["uv", "run", "wqm", "service", "restart"], "Restart service"),
    (["uv", "run", "wqm", "service", "status"], "Status after restart"),
    
    # Error condition tests
    (["uv", "run", "wqm", "service", "start"], "Start already running service"),
    (["uv", "run", "wqm", "service", "install"], "Install already installed service"),
    
    # Final cleanup and edge cases
    (["uv", "run", "wqm", "service", "stop"], "Stop before uninstall"),
    (["uv", "run", "wqm", "service", "stop"], "Double stop test"),
    (["uv", "run", "wqm", "service", "uninstall"], "Uninstall service"),
    (["uv", "run", "wqm", "service", "uninstall"], "Double uninstall test"),
    (["uv", "run", "wqm", "service", "start"], "Start non-installed service"),
    (["uv", "run", "wqm", "service", "status"], "Final status check")
]

# Execute all tests
print("Executing test sequence...")

for cmd_parts, description in test_commands:
    success = execute_test(cmd_parts, description)
    # Brief pause between tests
    import time
    time.sleep(1)

# Generate summary
print(f"\n{'='*60}")
print("TESTING COMPLETED - SUMMARY")
print(f"{'='*60}")

total_tests = len(test_results)
successful_tests = sum(1 for r in test_results if r['success'])
failed_tests = total_tests - successful_tests

print(f"Total tests executed: {total_tests}")
print(f"Successful tests: {successful_tests}")
print(f"Failed tests: {failed_tests}")
print(f"Success rate: {(successful_tests/total_tests*100):.1f}%")

if failed_tests > 0:
    print(f"\nFAILED TESTS:")
    for result in test_results:
        if not result['success']:
            print(f"  ✗ {result['description']} (exit code {result['exit_code']})")
            if result['stderr']:
                print(f"    Error: {result['stderr'][:80]}...")

# Save detailed results to file
report_file = "src/python/20250111-2137_wqm_service_test_report_data.txt"
with open(report_file, 'w') as f:
    f.write("WQM SERVICE TESTING - DETAILED RESULTS\n")
    f.write("=" * 60 + "\n")
    f.write(f"Generated: {datetime.now()}\n")
    f.write(f"Total Tests: {total_tests}\n")
    f.write(f"Successful: {successful_tests}\n")
    f.write(f"Failed: {failed_tests}\n")
    f.write(f"Success Rate: {(successful_tests/total_tests*100):.1f}%\n\n")
    
    for i, result in enumerate(test_results, 1):
        f.write(f"TEST {i}: {result['description']}\n")
        f.write(f"Command: {result['command']}\n")
        f.write(f"Exit Code: {result['exit_code']}\n")
        f.write(f"Success: {'YES' if result['success'] else 'NO'}\n")
        f.write(f"Memexd Processes: {result['memexd_count']}\n")
        
        if result['stdout']:
            f.write(f"STDOUT:\n{result['stdout']}\n")
        
        if result['stderr']:
            f.write(f"STDERR:\n{result['stderr']}\n")
        
        f.write("-" * 40 + "\n\n")

print(f"\nDetailed results saved to: {report_file}")
print(f"SUCCESS: Generated comprehensive WQM service test report")

# Return non-zero exit code if any tests failed
if failed_tests > 0:
    print(f"\nWARNING: {failed_tests} tests failed - see report for details")
    sys.exit(1)
else:
    print(f"\nSUCCESS: All {total_tests} tests passed")
    sys.exit(0)