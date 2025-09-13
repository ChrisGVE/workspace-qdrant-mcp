#!/usr/bin/env python3
"""
Test a complete MCP session with the isolated stdio server.
"""

import subprocess
import json
import sys

def test_mcp_session():
    """Test a complete MCP protocol session."""

    server_path = "/Users/chris/Dropbox/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp/20250913-1049_isolated_stdio_server.py"

    # Start the server process
    process = subprocess.Popen(
        ["uv", "run", "python", server_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    try:
        # Send initialize request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test_client",
                    "version": "1.0.0"
                }
            }
        }

        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        # Read response
        response_line = process.stdout.readline()
        if response_line:
            print("✓ Initialize response:", response_line.strip())
            init_response = json.loads(response_line.strip())

            if init_response.get("id") == 1 and "result" in init_response:
                print("✓ Initialize successful")

                # Send tools/list request
                tools_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                }

                process.stdin.write(json.dumps(tools_request) + "\n")
                process.stdin.flush()

                # Read tools response
                tools_response_line = process.stdout.readline()
                if tools_response_line:
                    print("✓ Tools/list response:", tools_response_line.strip())
                    tools_response = json.loads(tools_response_line.strip())

                    if tools_response.get("id") == 2 and "result" in tools_response:
                        tools = tools_response["result"].get("tools", [])
                        print(f"✓ Found {len(tools)} tools")
                        for tool in tools:
                            print(f"  - {tool.get('name', 'unnamed')}")

                        # Test tool call
                        if tools:
                            tool_call_request = {
                                "jsonrpc": "2.0",
                                "id": 3,
                                "method": "tools/call",
                                "params": {
                                    "name": "workspace_status",
                                    "arguments": {}
                                }
                            }

                            process.stdin.write(json.dumps(tool_call_request) + "\n")
                            process.stdin.flush()

                            # Read tool call response
                            call_response_line = process.stdout.readline()
                            if call_response_line:
                                print("✓ Tool call response:", call_response_line.strip())
                                call_response = json.loads(call_response_line.strip())

                                if call_response.get("id") == 3 and "result" in call_response:
                                    result = call_response["result"]["content"][0]["text"]
                                    print("✓ Tool call successful:", result)
                                    return True

        return False

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

    finally:
        process.terminate()
        process.wait(timeout=5)

if __name__ == "__main__":
    success = test_mcp_session()
    sys.exit(0 if success else 1)