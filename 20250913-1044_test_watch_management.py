#!/usr/bin/env python3
"""
Test watch_management import specifically.
"""

import os
import sys

# Set up test environment
sys.argv = ['test', '--transport', 'stdio']
os.environ['WQM_STDIO_MODE'] = 'true'

print("Testing watch_management import...")

try:
    print("Step 1: Import client...")
    from common.core.client import QdrantWorkspaceClient
    print("✓ client imported")

    print("Step 2: Import watch_management...")
    from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager
    print("✓ watch_management imported")

    print("Step 3: Import auto_ingestion...")
    from common.core.auto_ingestion import AutoIngestionManager
    print("✓ auto_ingestion imported")

    print("All imports successful!")

except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()