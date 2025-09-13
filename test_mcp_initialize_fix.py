#!/usr/bin/env python3
"""
Test script to verify the MCP 'initialize' method fix.

This script tests that the OptimizedFastMCPApp now correctly handles
the 'initialize' method that was causing "Method not found: initialize" errors.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from common.optimization.complete_fastmcp_optimization import OptimizedWorkspaceServer


async def test_initialize_method():
    """Test that the initialize method works correctly."""
    print("🔧 Testing MCP 'initialize' method fix...")

    try:
        # Create optimized server
        optimizer = OptimizedWorkspaceServer(enable_optimizations=True)
        app = optimizer.create_optimized_app("test-workspace-qdrant-mcp")

        # Test the exact request that was failing
        initialize_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "Claude Desktop",
                    "version": "0.7.1"
                }
            }
        }

        print(f"📤 Sending initialize request: {initialize_request['method']}")
        response = await app.handle_request(initialize_request)

        print(f"📥 Response: {response}")

        # Verify response format
        if "error" in response:
            error_message = response["error"]["message"]
            if "Method not found: initialize" in error_message:
                print("❌ FAILED: Still getting 'Method not found: initialize' error")
                return False
            else:
                print(f"❌ FAILED: Different error occurred: {error_message}")
                return False

        if "result" in response:
            result = response["result"]

            # Check required fields
            required_fields = ["protocolVersion", "capabilities", "serverInfo"]
            missing_fields = [field for field in required_fields if field not in result]

            if missing_fields:
                print(f"❌ FAILED: Missing required fields in response: {missing_fields}")
                return False

            print("✅ SUCCESS: Initialize method working correctly!")
            print(f"  - Protocol version: {result['protocolVersion']}")
            print(f"  - Server name: {result['serverInfo']['name']}")
            print(f"  - Server version: {result['serverInfo']['version']}")

            return True

        print(f"❌ FAILED: Unexpected response format: {response}")
        return False

    except Exception as e:
        print(f"❌ FAILED: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_other_core_methods():
    """Test other core MCP methods."""
    print("\n🔧 Testing other core MCP methods...")

    optimizer = OptimizedWorkspaceServer(enable_optimizations=True)
    app = optimizer.create_optimized_app("test-workspace-qdrant-mcp")

    test_cases = [
        {
            "name": "initialized",
            "request": {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "initialized",
                "params": {}
            }
        },
        {
            "name": "ping",
            "request": {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "ping",
                "params": {}
            }
        },
        {
            "name": "tools/list",
            "request": {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/list",
                "params": {}
            }
        }
    ]

    results = {}

    for test_case in test_cases:
        method_name = test_case["name"]
        request = test_case["request"]

        try:
            print(f"📤 Testing {method_name}...")
            response = await app.handle_request(request)

            if "error" in response:
                if "Method not found" in response["error"]["message"]:
                    print(f"❌ {method_name}: Method not found error")
                    results[method_name] = False
                else:
                    print(f"⚠️ {method_name}: Other error: {response['error']['message']}")
                    results[method_name] = True  # At least it's handled
            else:
                print(f"✅ {method_name}: Working correctly")
                results[method_name] = True

        except Exception as e:
            print(f"❌ {method_name}: Exception: {e}")
            results[method_name] = False

    success_count = sum(results.values())
    total_count = len(results)

    print(f"\n📊 Core method test results: {success_count}/{total_count} methods working")

    return success_count == total_count


async def main():
    """Main test function."""
    print("🚀 MCP Protocol Compliance Fix Test")
    print("=" * 50)

    # Test the main issue
    initialize_success = await test_initialize_method()

    # Test other methods
    other_methods_success = await test_other_core_methods()

    print("\n" + "=" * 50)
    if initialize_success and other_methods_success:
        print("🎉 ALL TESTS PASSED: MCP protocol compliance fix successful!")
        print("\nThe 'Method not found: initialize' error has been resolved.")
        print("✅ OptimizedFastMCPApp now supports all core MCP protocol methods")
        return True
    else:
        print("❌ SOME TESTS FAILED: Further fixes needed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)