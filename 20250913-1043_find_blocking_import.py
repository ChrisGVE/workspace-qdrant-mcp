#!/usr/bin/env python3
"""
Find which import is causing the hang.
"""

import os
import sys

# Set up test environment
sys.argv = ['test', '--transport', 'stdio']
os.environ['WQM_STDIO_MODE'] = 'true'

print("Testing imports one by one...")

try:
    print("Testing basic Python modules...")
    import asyncio
    print("✓ asyncio")
    import atexit
    print("✓ atexit")
    import logging
    print("✓ logging")
    import os
    print("✓ os")
    import signal
    print("✓ signal")
    import datetime
    print("✓ datetime")
    from typing import List, Optional
    print("✓ typing")

    print("Testing third-party modules...")
    import typer
    print("✓ typer")

    print("Testing fastmcp...")
    from fastmcp import FastMCP
    print("✓ fastmcp")

    print("Testing pydantic...")
    from pydantic import BaseModel
    print("✓ pydantic")

    print("Testing workspace_qdrant_mcp common modules...")

    try:
        from common.core.advanced_watch_config import AdvancedConfigValidator
        print("✓ advanced_watch_config")
    except Exception as e:
        print(f"✗ advanced_watch_config: {e}")

    try:
        from common.core.auto_ingestion import AutoIngestionManager
        print("✓ auto_ingestion")
    except Exception as e:
        print(f"✗ auto_ingestion: {e}")

    try:
        from common.core.client import QdrantWorkspaceClient
        print("✓ client")
    except Exception as e:
        print(f"✗ client: {e}")

    print("All imports completed successfully!")

except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()