#!/usr/bin/env python3
"""
Test simple server import.
"""

import os
import sys

# Set up test environment
sys.argv = ['test', '--transport', 'stdio']
os.environ['WQM_STDIO_MODE'] = 'true'

try:
    print("Importing server...")
    from workspace_qdrant_mcp.server import _STDIO_MODE, app, run_server
    print(f"✓ Server imported successfully, stdio mode: {_STDIO_MODE}")
    print(f"✓ App available: {type(app)}")
    print(f"✓ run_server function available: {callable(run_server)}")
    print("Success!")

except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()