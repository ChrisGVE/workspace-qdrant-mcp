#!/usr/bin/env python3
"""
Final validation script for loguru migration cleanup.
Quick tests to confirm the migration is successful.
"""

import os
import sys
import warnings
from pathlib import Path

def main():
    """Run final validation tests."""
    print("=== FINAL LOGURU MIGRATION VALIDATION ===\n")

    # Set up environment
    sys.path.insert(0, "src/python")
    os.environ["WQM_STDIO_MODE"] = "true"
    warnings.filterwarnings('ignore')  # Suppress deprecation warnings during testing

    success_count = 0
    total_tests = 5

    # Test 1: Basic loguru import
    try:
        from common.logging.loguru_config import get_logger
        logger = get_logger('test')
        logger.info("Test log message")
        print("✓ Test 1: Basic loguru import and logging - SUCCESS")
        success_count += 1
    except Exception as e:
        print(f"✗ Test 1: Basic loguru import - FAILED: {e}")

    # Test 2: stdio_server import
    try:
        from workspace_qdrant_mcp import stdio_server
        print("✓ Test 2: stdio_server import - SUCCESS")
        success_count += 1
    except Exception as e:
        print(f"✗ Test 2: stdio_server import - FAILED: {e}")

    # Test 3: MCP tools import
    try:
        from workspace_qdrant_mcp.tools import memory
        print("✓ Test 3: MCP tools import - SUCCESS")
        success_count += 1
    except Exception as e:
        print(f"✗ Test 3: MCP tools import - FAILED: {e}")

    # Test 4: Core client import
    try:
        from common.core.client import QdrantWorkspaceClient
        print("✓ Test 4: Core client import - SUCCESS")
        success_count += 1
    except Exception as e:
        print(f"✗ Test 4: Core client import - FAILED: {e}")

    # Test 5: Verify old files removed
    old_files = [
        "src/python/common/logging/config.py",
        "src/python/common/logging/formatters.py",
        "src/python/common/logging/handlers.py",
        "src/python/common/logging/core.py",
        "src/python/common/logging/migration.py",
        "src/python/common/observability/logger.py",
    ]

    still_exist = [f for f in old_files if Path(f).exists()]
    if not still_exist:
        print("✓ Test 5: Old logging files properly removed - SUCCESS")
        success_count += 1
    else:
        print(f"✗ Test 5: Old files still exist: {still_exist}")

    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {success_count}/{total_tests} tests")

    if success_count == total_tests:
        print("🎉 LOGURU MIGRATION CLEANUP SUCCESSFUL!")
        print("\nAll key functionality is working:")
        print("- Loguru logging system active")
        print("- MCP server components import correctly")
        print("- Old logging infrastructure removed")
        print("- No import conflicts detected")
        return 0
    else:
        print("❌ Some issues remain - see test results above")
        return 1

if __name__ == "__main__":
    sys.exit(main())