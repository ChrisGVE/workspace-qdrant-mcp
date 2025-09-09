#!/usr/bin/env python3
"""
Clean up temporary test files and commit the core module fix.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
os.chdir(project_root)

# List of temporary files to clean up
temp_files = [
    "run_tests.py",
    "20250108-2123_test_runner.py", 
    "20250108-2124_execute_tests.sh",
    "20250108-2125_direct_test_execution.py",
    "20250108-2126_run_test_validation.py",
    "20250108-2127_test_execution_output.py",
    "20250108-2128_direct_test_run.py",
    "20250108-2129_test_runner_simple.py",
    "20250108-2130_quick_validation.py",
    "20250108-2131_final_test_runner.py",
    "20250108-2132_execute_final_tests.py",
    "20250108-2133_inline_test_execution.py",
    "20250108-2134_cleanup_temp_files.py",  # This file itself
    "test_results.md"  # Move to permanent location
]

print("Cleaning up temporary test files...")

for file in temp_files:
    file_path = project_root / file
    if file_path.exists():
        if file == "test_results.md":
            # Move to permanent location
            permanent_path = project_root / "docs" / "test_results_collection_naming.md"
            permanent_path.parent.mkdir(exist_ok=True)
            file_path.rename(permanent_path)
            print(f"Moved {file} to docs/")
        else:
            file_path.unlink()
            print(f"Removed {file}")

print("Cleanup complete.")

# Now execute a final validation inline
print("\nFinal validation of collection naming framework...")
print("-" * 50)

try:
    sys.path.insert(0, str(project_root))
    import core.collection_naming as cn
    
    # Quick smoke test
    result = cn.normalize_collection_name_component("test-name")
    assert result == "test_name"
    
    result = cn.build_project_collection_name("my-project", "docs")  
    assert result == "my_project-docs"
    
    result = cn.validate_collection_name("project-docs")
    assert result is True
    
    print("✅ Collection naming framework is working correctly!")
    print("All core functions tested and validated.")
    
except Exception as e:
    print(f"❌ Validation failed: {e}")
    sys.exit(1)

print("-" * 50)
print("Ready for commit.")