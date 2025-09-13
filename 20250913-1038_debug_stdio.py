#!/usr/bin/env python3
"""
Debug script to test stdio mode detection and silencing.
"""

import os
import sys

print("Starting debug test...")
print(f"Command line args: {sys.argv}")

# Set up test environment
sys.argv = ['test', '--transport', 'stdio']
os.environ['WQM_STDIO_MODE'] = 'true'

print("About to import server module...")

try:
    # Test the import
    from workspace_qdrant_mcp.server import _STDIO_MODE, _ORIGINAL_STDOUT, run_server
    print(f"Import successful!")
    print(f"Stdio mode detected: {_STDIO_MODE}")
    print(f"Original stdout available: {_ORIGINAL_STDOUT is not None}")

    # Test if we can access the original stdout
    if _ORIGINAL_STDOUT:
        _ORIGINAL_STDOUT.write("This should be visible on original stdout\n")
        _ORIGINAL_STDOUT.flush()

    print("Testing complete - if you see this, stdio mode is NOT working")

except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()