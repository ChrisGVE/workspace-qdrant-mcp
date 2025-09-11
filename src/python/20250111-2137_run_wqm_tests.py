#!/usr/bin/env python3
"""
Execute WQM Service Tests and capture results
"""
import subprocess
import sys
import os

def main():
    print("Starting WQM Service Command Testing...")
    print("This will test all service command combinations systematically.")
    print("-" * 60)
    
    # Change to project root directory for testing
    project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Execute the test script
    try:
        result = subprocess.run([
            "python3", 
            "src/python/20250111-2137_wqm_service_testing.py"
        ], cwd=project_root, capture_output=False, text=True)
        
        print(f"\nTest script completed with exit code: {result.returncode}")
        
        # Check for result files
        result_files = [
            "src/python/20250111-2137_wqm_service_detailed_results.txt",
            "20250111-2137_wqm_service_detailed_results.txt"
        ]
        
        for file_path in result_files:
            if os.path.exists(file_path):
                print(f"Results saved to: {file_path}")
                break
        else:
            print("Warning: Could not find results file")
            
        return result.returncode
        
    except Exception as e:
        print(f"Error executing test script: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)