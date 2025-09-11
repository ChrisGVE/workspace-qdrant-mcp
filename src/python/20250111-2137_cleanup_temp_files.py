#!/usr/bin/env python3
"""
Clean up temporary test files created during WQM service testing
"""
import os
import glob

def cleanup_test_files():
    """Remove temporary test files"""
    project_root = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp"
    
    # Files to clean up (keep the main report)
    temp_files = [
        "src/python/20250111-2137_wqm_service_testing.py",
        "src/python/20250111-2137_run_wqm_tests.py", 
        "src/python/20250111-2137_manual_wqm_test.py",
        "src/python/20250111-2137_simple_test.py",
        "src/python/20250111-2137_execute_direct_test.py",
        "src/python/20250111-2137_direct_execution.py",
        "src/python/20250111-2137_execute_now.py",
        "src/python/20250111-2137_quick_wqm_check.py",
        "src/python/20250111-2137_individual_test.py",
        "src/python/20250111-2137_find_cli.py",
        "src/python/20250111-2137_ls_project.py",
        "src/python/20250111-2137_exec_and_test.py",
        "src/python/20250111-2137_minimal_test.py",
        "src/python/20250111-2137_run_minimal.sh",
        "src/python/20250111-2137_run_tests.sh",
        "20250111-2137_execute_tests.sh",
        "src/python/20250111-2137_cleanup_temp_files.py"  # This file itself
    ]
    
    # Files to keep 
    keep_files = [
        "src/python/20250111-2137_wqm_service_test_report.md"  # Main report
    ]
    
    os.chdir(project_root)
    
    print("Cleaning up temporary WQM test files...")
    print(f"Working directory: {os.getcwd()}")
    
    cleaned_count = 0
    
    for file_path in temp_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"✓ Removed: {file_path}")
                cleaned_count += 1
            except Exception as e:
                print(f"✗ Failed to remove {file_path}: {e}")
        else:
            print(f"- Not found: {file_path}")
    
    # Check for any other temp files matching the pattern
    pattern_files = glob.glob("src/python/20250111-2137_*")
    for file_path in pattern_files:
        if file_path not in keep_files and os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"✓ Pattern cleanup: {file_path}")
                cleaned_count += 1
            except Exception as e:
                print(f"✗ Pattern cleanup failed {file_path}: {e}")
    
    print(f"\nCleanup complete: {cleaned_count} files removed")
    print("Files preserved:")
    for keep_file in keep_files:
        if os.path.exists(keep_file):
            print(f"✓ Kept: {keep_file}")

if __name__ == "__main__":
    cleanup_test_files()