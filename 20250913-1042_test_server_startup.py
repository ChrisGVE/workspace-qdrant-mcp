#!/usr/bin/env python3
"""
Test server startup process step by step.
"""

import os
import sys
import asyncio

# Set up test environment
sys.argv = ['test', '--transport', 'stdio']
os.environ['WQM_STDIO_MODE'] = 'true'

print("Testing server startup process...")

try:
    print("Step 1: Import server module...")
    from workspace_qdrant_mcp.server import _STDIO_MODE, app
    print(f"✓ Server imported, stdio mode: {_STDIO_MODE}")

    print("Step 2: Test app object...")
    print(f"✓ App type: {type(app)}")
    print(f"✓ App has run method: {hasattr(app, 'run')}")

    print("Step 3: Test workspace client import...")
    from workspace_qdrant_mcp.server import workspace_client
    print(f"✓ Workspace client imported: {workspace_client}")

    print("Step 4: Test async initialization...")
    async def test_init():
        try:
            # Import the initialization function
            from workspace_qdrant_mcp.server import initialize_workspace
            print("✓ Initialize workspace function imported")

            # Try to run it
            await initialize_workspace(None)
            print("✓ Workspace initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Workspace initialization failed: {e}")
            return False

    # Run the async test
    result = asyncio.run(test_init())

    if result:
        print("All tests passed! Server should work.")
    else:
        print("Workspace initialization failed.")

except Exception as e:
    print(f"Test failed at import: {e}")
    import traceback
    traceback.print_exc()