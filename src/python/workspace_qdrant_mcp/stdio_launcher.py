#!/usr/bin/env python3
"""
Dedicated stdio launcher that completely bypasses all server imports.

This launcher is designed to execute only in stdio mode without triggering
any imports that could cause server loading or logging initialization.
"""

import os
import sys
import json
from typing import Dict, Any


def detect_stdio_mode() -> bool:
    """Detect if we should use stdio mode."""
    # Check command line arguments
    if "--transport" in sys.argv:
        try:
            transport_idx = sys.argv.index("--transport")
            if transport_idx + 1 < len(sys.argv):
                return sys.argv[transport_idx + 1] == "stdio"
        except (ValueError, IndexError):
            pass

    # Check environment variables
    if os.getenv("WQM_STDIO_MODE", "").lower() == "true":
        return True
    if os.getenv("MCP_TRANSPORT") == "stdio":
        return True

    # Default to stdio if no transport specified (common for MCP clients)
    if "--transport" not in sys.argv and len(sys.argv) == 1:
        return True

    # Check if stdin is piped (typical MCP scenario)
    if hasattr(sys.stdin, 'isatty') and not sys.stdin.isatty():
        return True

    # Check if stdout is piped (typical MCP scenario)
    if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
        return True

    return False


class StdioMCPServer:
    """Minimal MCP server for stdio mode with zero external dependencies."""

    def __init__(self, name: str):
        self.name = name
        self.tools = {
            "workspace_status": self._workspace_status,
            "echo_test": self._echo_test,
            "server_info": self._server_info,
        }

    def _workspace_status(self) -> Dict[str, Any]:
        """Get basic workspace status."""
        return {
            "status": "active",
            "mode": "stdio-isolated-direct",
            "workspace": os.getcwd(),
            "message": "Direct stdio MCP server - complete isolation achieved"
        }

    def _echo_test(self, message: str = "test") -> str:
        """Echo test for MCP protocol validation."""
        return f"Echo: {message}"

    def _server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return {
            "name": "workspace-qdrant-mcp-stdio",
            "version": "1.0.0-direct",
            "mode": "stdio_isolated_direct",
            "description": "Direct stdio MCP server bypassing all imports",
            "capabilities": ["workspace_status", "echo_test", "server_info"]
        }

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request."""
        method = request.get("method")
        request_id = request.get("id")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
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
                        "version": "1.0.0-direct"
                    }
                }
            }

        elif method == "tools/list":
            tools_list = []
            for tool_name in self.tools.keys():
                tools_list.append({
                    "name": tool_name,
                    "description": f"{tool_name} tool"
                })
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools_list}
            }

        elif method == "tools/call":
            tool_name = request.get("params", {}).get("name")
            arguments = request.get("params", {}).get("arguments", {})

            if tool_name in self.tools:
                try:
                    result = self.tools[tool_name](**arguments)
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {"content": [{"type": "text", "text": str(result)}]}
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": f"Tool execution error: {str(e)}"}
                    }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"}
            }

    def run(self):
        """Run the server in stdio mode."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
                response = self.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"}
                }
                print(json.dumps(error_response), flush=True)
            except Exception:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id") if 'request' in locals() else None,
                    "error": {"code": -32603, "message": "Internal error"}
                }
                print(json.dumps(error_response), flush=True)


def main():
    """Main entry point for stdio launcher."""

    # Set up complete stdio suppression immediately
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress all warnings
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Suppress all logging
    import logging
    logging.getLogger().disabled = True
    logging.getLogger().handlers.clear()

    # Redirect stderr to null
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        pass

    # Create and run stdio server
    server = StdioMCPServer("workspace-qdrant-mcp-stdio")
    server.run()


if __name__ == "__main__":
    main()