#!/usr/bin/env python3
"""
Completely standalone MCP stdio server for Claude Code integration.

This module is designed to work independently without importing any
problematic dependencies from the main server or common modules.
"""

import json
import os
import sys
import warnings

# Suppress all warnings immediately
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Set up stdio mode environment
os.environ["WQM_STDIO_MODE"] = "true"
os.environ["MCP_QUIET_MODE"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Redirect stderr to null to ensure complete silence
_null_device = open(os.devnull, 'w')
sys.stderr = _null_device

# Import FastMCP for the server
try:
    from fastmcp import FastMCP
except ImportError:
    # If FastMCP is not available, provide a minimal JSON-RPC server
    class MinimalStdioServer:
        def __init__(self, name):
            self.name = name

        def tool(self):
            def decorator(func):
                return func
            return decorator

        def run(self, transport="stdio"):
            self._run_jsonrpc_loop()

        def _run_jsonrpc_loop(self):
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue

                try:
                    request = json.loads(line)
                    response = self._handle_request(request)
                    print(json.dumps(response), flush=True)
                except Exception:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32603, "message": "Internal error"}
                    }
                    print(json.dumps(error_response), flush=True)

        def _handle_request(self, request):
            method = request.get("method")
            request_id = request.get("id")

            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": True}},
                        "serverInfo": {"name": self.name, "version": "1.0.0-standalone"}
                    }
                }
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"tools": [
                        {"name": "workspace_status", "description": "Get workspace status"},
                        {"name": "echo_test", "description": "Echo test message"}
                    ]}
                }
            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                return self._handle_tool_call(request_id, tool_name, arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"}
                }

        def _handle_tool_call(self, request_id, tool_name, arguments):
            if tool_name == "workspace_status":
                result = {
                    "status": "active",
                    "mode": "standalone-stdio",
                    "workspace": os.getcwd(),
                    "message": "Standalone MCP server running in stdio mode"
                }
            elif tool_name == "echo_test":
                message = arguments.get("message", "test")
                result = f"Echo: {message}"
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": str(result)}]}
            }

    FastMCP = MinimalStdioServer


def create_standalone_server():
    """Create a standalone MCP server with basic tools."""
    app = FastMCP("workspace-qdrant-mcp-standalone")

    @app.tool()
    async def workspace_status() -> dict:
        """Get basic workspace status."""
        return {
            "status": "active",
            "mode": "standalone-stdio",
            "workspace": os.getcwd(),
            "message": "Standalone MCP server with basic functionality",
            "tools": ["workspace_status", "echo_test", "search_placeholder"]
        }

    @app.tool()
    async def echo_test(message: str) -> str:
        """Echo test message."""
        return f"Echo: {message}"

    @app.tool()
    async def search_placeholder(query: str, limit: int = 10) -> dict:
        """Placeholder search function."""
        return {
            "query": query,
            "results": [],
            "message": "Search requires full server mode with Qdrant connection",
            "limit": limit,
            "total": 0
        }

    return app


def main():
    """Main entry point for the standalone stdio server."""
    try:
        app = create_standalone_server()
        app.run(transport="stdio")
    except Exception as e:
        # Even in case of error, try to provide some basic response
        # But don't use any logging that might interfere with stdio
        minimal_server = FastMCP("workspace-qdrant-mcp-fallback")
        minimal_server._run_jsonrpc_loop() if hasattr(minimal_server, '_run_jsonrpc_loop') else sys.exit(1)


if __name__ == "__main__":
    main()