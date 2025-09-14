"""
Smart launcher that chooses between full server and stdio-optimized server.

This launcher detects the transport mode and runtime environment to select
the appropriate server implementation, avoiding import hangs in stdio mode.
"""

import os
import sys

# CRITICAL: Set up stdio mode detection and environment IMMEDIATELY at module load time
# This must happen before ANY other imports that could trigger server loading

def _detect_stdio_at_import() -> bool:
    """Detect stdio mode at module import time."""
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

# Set up stdio mode environment variables IMMEDIATELY if detected
if _detect_stdio_at_import():
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_TRACE"] = ""

    # Redirect stderr to null immediately
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        pass

    # Suppress warnings immediately
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

# CRITICAL: Set up stdio suppression immediately if needed
def _setup_stdio_suppression():
    """Set up stdio suppression before any server imports that might log."""
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GRPC_VERBOSITY"] = "NONE"
    os.environ["GRPC_TRACE"] = ""

    # Redirect stderr to null immediately
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        pass

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")


def detect_stdio_mode() -> bool:
    """Detect if we should use stdio mode based on arguments and environment."""
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

    # Check if stdin is piped (typical MCP scenario where JSON-RPC comes via stdin)
    if hasattr(sys.stdin, 'isatty') and not sys.stdin.isatty():
        return True

    # Check if stdout is piped (typical MCP scenario)
    if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
        return True

    return False


def main():
    """Main launcher entry point."""

    is_stdio = detect_stdio_mode()

    if is_stdio:
        # CRITICAL: Use dedicated stdio launcher that bypasses all imports
        # This completely avoids the package import chain that triggers server loading
        import subprocess
        import sys

        # Execute the dedicated stdio launcher directly
        stdio_launcher_path = __file__.replace('launcher.py', 'stdio_launcher.py')

        # Execute as subprocess to completely isolate from current import context
        result = subprocess.run([sys.executable, stdio_launcher_path],
                               stdin=sys.stdin,
                               stdout=sys.stdout,
                               stderr=sys.stderr)
        sys.exit(result.returncode)
    else:
        # Use full server implementation for non-stdio modes
        from .server import main as server_main
        server_main()


def _run_isolated_stdio_server():
    """Run the isolated stdio server directly without imports."""
    import os
    import sys
    import json
    from typing import Dict, Any

    # Set up complete stdio suppression immediately
    os.environ["WQM_STDIO_MODE"] = "true"
    os.environ["MCP_QUIET_MODE"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Suppress all warnings
    import warnings
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")

    # Disable all logging
    import logging
    logging.getLogger().disabled = True
    logging.getLogger().handlers.clear()

    # Redirect stderr to null
    try:
        sys.stderr = open(os.devnull, 'w')
    except:
        pass

    # Inline minimal MCP server implementation to avoid any imports
    class InlineMCPServer:
        def __init__(self, name: str):
            self.name = name
            self.tools = {
                "workspace_status": lambda: {
                    "status": "active",
                    "mode": "stdio-isolated-inline",
                    "workspace": os.getcwd(),
                    "message": "Inline MCP server running in stdio mode"
                },
                "echo_test": lambda message="test": f"Echo: {message}",
                "server_info": lambda: {
                    "name": "workspace-qdrant-mcp-stdio",
                    "version": "1.0.0-inline",
                    "mode": "stdio_isolated_inline",
                    "description": "Inline MCP server for complete stdio isolation",
                    "capabilities": ["workspace_status", "echo_test", "server_info"]
                }
            }

        def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
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
                            "version": "1.0.0-inline"
                        }
                    }
                }
            elif request.get("method") == "tools/list":
                tools_list = []
                for tool_name in self.tools.keys():
                    tools_list.append({
                        "name": tool_name,
                        "description": f"{tool_name} tool"
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
            # Read from stdin and handle MCP requests
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

    # Create and run the inline server
    app = InlineMCPServer("workspace-qdrant-mcp-stdio")
    app.run_stdio()


if __name__ == "__main__":
    main()