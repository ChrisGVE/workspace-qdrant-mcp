#!/usr/bin/env python3
"""
Test observability and tool imports.
"""

import os
import sys

# Set up test environment
sys.argv = ['test', '--transport', 'stdio']
os.environ['WQM_STDIO_MODE'] = 'true'

print("Testing observability imports...")

try:
    print("Testing common.observability...")
    from common.observability import (
        health_checker_instance,
        metrics_instance,
        monitor_async,
        record_operation,
    )
    print("✓ observability imported")

    print("Testing observability.endpoints...")
    from common.observability.endpoints import (
        add_observability_routes,
        setup_observability_middleware,
    )
    print("✓ observability.endpoints imported")

    print("Testing tools.documents...")
    from workspace_qdrant_mcp.tools.documents import (
        add_document,
        get_document,
    )
    print("✓ tools.documents imported")

    print("Testing tools.grpc_tools...")
    from workspace_qdrant_mcp.tools.grpc_tools import (
        get_grpc_engine_stats,
        process_document_via_grpc,
        search_via_grpc,
        test_grpc_connection,
    )
    print("✓ tools.grpc_tools imported")

    print("All imports successful!")

except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()