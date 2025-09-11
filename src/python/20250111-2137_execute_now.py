#!/usr/bin/env python3

# Simple execution wrapper 
import subprocess
import os

os.chdir("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")

print("Executing WQM service tests...")
print("This may take several minutes to complete all test scenarios.")
print("-" * 50)

try:
    # Execute the comprehensive test
    result = subprocess.run([
        "python3", 
        "src/python/20250111-2137_direct_execution.py"
    ], 
    stdout=subprocess.PIPE, 
    stderr=subprocess.STDOUT, 
    text=True,
    timeout=300  # 5 minute timeout
    )
    
    # Print all output
    print(result.stdout)
    
    if result.returncode == 0:
        print("✓ Tests completed successfully")
    else:
        print(f"⚠ Tests completed with issues (exit code: {result.returncode})")
    
    # Check if report file was created
    report_files = [
        "src/python/20250111-2137_wqm_service_test_report_data.txt"
    ]
    
    for report_file in report_files:
        if os.path.exists(report_file):
            print(f"✓ Report generated: {report_file}")
            break
    else:
        print("⚠ Report file not found")
        
except subprocess.TimeoutExpired:
    print("✗ Test execution timed out after 5 minutes")
except Exception as e:
    print(f"✗ Test execution failed: {e}")

print("\nTest execution wrapper complete.")