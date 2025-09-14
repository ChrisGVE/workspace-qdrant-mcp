#!/usr/bin/env python3
"""
Completely isolated MCP stdio server with zero dependencies on the main server.

This server is designed to provide complete console silence for MCP protocol
compliance by avoiding any imports that could trigger logging or server initialization.
"""

import os
import sys
import json
import asyncio
from typing import Dict, Any, List, Optional

# CRITICAL: Complete stdio suppression setup immediately
os.environ["WQM_STDIO_MODE"] = "true"
os.environ["MCP_QUIET_MODE"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Redirect all output to null immediately
_null_device = open(os.devnull, 'w')
sys.stderr = _null_device
sys.stdout = sys.stdout  # Keep stdout for MCP protocol

# Disable all warnings
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


class MinimalMCPServer:
    """Minimal MCP server implementation with zero external dependencies."""

    def __init__(self, name: str):
        self.name = name
        self.tools = {}

    def tool(self, name: Optional[str] = None):
        """Register a tool function."""
        def decorator(func):
            tool_name = name or func.__name__
            self.tools[tool_name] = func
            return func
        return decorator

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single MCP request."""

        if request.get("method") == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "experimental": {},
                        "prompts": {"listChanged": False},
                        "resources": {"subscribe": False, "listChanged": False}
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": "1.0.0-stdio"
                    }
                }
            }

        elif request.get("method") == "tools/list":
            tools_list = []
            for tool_name, tool_func in self.tools.items():
                tools_list.append({
                    "name": tool_name,
                    "description": getattr(tool_func, '__doc__', '') or f"{tool_name} tool"
                })

            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "result": {"tools": tools_list}
            }

        elif request.get("method") == "tools/call":
            tool_name = request.get("params", {}).get("name")
            arguments = request.get("params", {}).get("arguments", {})

            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](**arguments)
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "result": {"content": [{"type": "text", "text": str(result)}]}
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": request.get("id"),
                        "error": {"code": -32603, "message": f"Tool execution error: {str(e)}"}
                    }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32601, "message": "Method not found"}
            }

    def run_stdio(self):
        """Run the server in stdio mode."""

        # Read from stdin line by line
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.handle_request(request)
                # Write response to stdout (this is the only allowed output)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                # Invalid JSON, send error response
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"}
                }
                print(json.dumps(error_response), flush=True)
            except Exception:
                # Any other error, send generic error
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id") if 'request' in locals() else None,
                    "error": {"code": -32603, "message": "Internal error"}
                }
                print(json.dumps(error_response), flush=True)


def create_isolated_server():
    """Create and configure the isolated MCP server."""

    app = MinimalMCPServer("workspace-qdrant-mcp-stdio")

    @app.tool("workspace_status")
    def workspace_status():
        """Get basic workspace status information."""
        return {
            "status": "active",
            "mode": "stdio-isolated",
            "workspace": os.getcwd(),
            "message": "Isolated MCP server running in stdio mode"
        }

    @app.tool("echo_test")
    def echo_test(message: str = "test"):
        """Echo test message for MCP protocol validation."""
        return f"Echo: {message}"

    @app.tool("server_info")
    def server_info():
        """Get server information and capabilities."""
        return {
            "name": "workspace-qdrant-mcp-stdio",
            "version": "1.0.0-isolated",
            "mode": "stdio_isolated",
            "description": "Completely isolated MCP server for stdio mode",
            "capabilities": ["workspace_status", "echo_test", "server_info"]
        }

    return app


def main():
    """Main entry point for isolated stdio server."""

    # Ensure we're in stdio mode
    if not (os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
            (hasattr(sys.stdin, 'isatty') and not sys.stdin.isatty())):
        # Not in stdio mode, just exit silently
        sys.exit(0)

    # Create and run the isolated server
    app = create_isolated_server()
    app.run_stdio()


if __name__ == "__main__":
    main()