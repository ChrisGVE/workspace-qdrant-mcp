#!/usr/bin/env python3
"""
Test the improved stdio implementation.
"""

import os
import sys
import json

# Set up test environment
sys.argv = ['test', '--transport', 'stdio']

print("Starting improved stdio test...")

try:
    from workspace_qdrant_mcp.server import _STDIO_MODE, run_server
    print(f"Stdio mode detected: {_STDIO_MODE}")

    # Test stdout wrapper behavior
    print("Testing print statement (should be filtered out)")

    # Test JSON-RPC output (should pass through)
    json_response = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"protocolVersion": "2024-11-05", "capabilities": {}}
    }
    print(json.dumps(json_response))

    print("Test complete")

except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()