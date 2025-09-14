#!/usr/bin/env python3
"""
Entry point that detects execution mode and routes accordingly.

This entry point is designed to avoid any imports that could trigger
server loading before we can determine the execution mode.
"""

import os
import sys
import subprocess


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


def run_stdio_mode():
    """Run in stdio mode with complete isolation."""
    import json
    from typing import Dict, Any

    # Set up complete suppression immediately
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress warnings and logging
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.getLogger().disabled = True
    logging.getLogger().handlers.clear()

    # Redirect stderr
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        pass

    class StdioMCPServer:
        """Minimal stdio MCP server."""

        def __init__(self):
            self.name = "workspace-qdrant-mcp-stdio"

        def handle_initialize(self, request_id):
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
                        "version": "1.0.0-entry-point"
                    }
                }
            }

        def handle_tools_list(self, request_id):
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {"name": "workspace_status", "description": "Get workspace status"},
                        {"name": "echo_test", "description": "Echo test message"}
                    ]
                }
            }

        def handle_tool_call(self, request_id, tool_name, arguments):
            if tool_name == "workspace_status":
                result = {
                    "status": "active",
                    "mode": "stdio-entry-point",
                    "workspace": os.getcwd(),
                    "message": "Entry point stdio server active"
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

        def handle_request(self, request):
            method = request.get("method")
            request_id = request.get("id")

            if method == "initialize":
                return self.handle_initialize(request_id)
            elif method == "tools/list":
                return self.handle_tools_list(request_id)
            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                return self.handle_tool_call(request_id, tool_name, arguments)
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"}
                }

        def run(self):
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                try:
                    request = json.loads(line)
                    response = self.handle_request(request)
                    print(json.dumps(response), flush=True)
                except Exception:
                    error = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32603, "message": "Internal error"}
                    }
                    print(json.dumps(error), flush=True)

    server = StdioMCPServer()
    server.run()


def run_full_mode():
    """Run in full server mode."""
    # Set environment to prevent stdio mode check in package init
    os.environ["WQM_STDIO_MODE"] = "false"

    # Now safe to import the full server
    from workspace_qdrant_mcp.server import main as server_main
    server_main()


def main():
    """Main entry point that routes to appropriate mode."""

    if detect_stdio_mode():
        run_stdio_mode()
    else:
        run_full_mode()


if __name__ == "__main__":
    main()