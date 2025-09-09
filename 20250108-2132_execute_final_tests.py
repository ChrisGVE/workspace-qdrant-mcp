#!/usr/bin/env python3

# Execute the comprehensive test validation
import sys
import os
from pathlib import Path

# Change to project directory
project_root = Path("/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp")
os.chdir(project_root)
sys.path.insert(0, str(project_root))

print("Executing collection naming framework test validation...")
print("Working from:", project_root)

# Import the test runner and execute
try:
    # Import our test runner module
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_runner", project_root / "20250108-2131_final_test_runner.py")
    test_runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_runner)
    
    # Execute the main function
    exit_code = test_runner.main()
    print(f"\nFinal exit code: {exit_code}")
    
    if exit_code == 0:
        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
        
except Exception as e:
    print(f"Error executing test runner: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback - try manual execution
    print("\nFalling back to manual execution...")
    try:
        exit_code = test_runner.run_manual_tests()
        print(f"Manual test exit code: {exit_code}")
    except:
        print("Manual execution also failed")
        sys.exit(1)