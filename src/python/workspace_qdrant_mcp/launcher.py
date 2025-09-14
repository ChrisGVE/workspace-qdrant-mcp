"""
Smart launcher that chooses between full server and stdio-optimized server.

This launcher detects the transport mode and runtime environment to select
the appropriate server implementation, avoiding import hangs in stdio mode.
"""

import os
import sys

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
        # Set up stdio suppression BEFORE importing any server modules
        _setup_stdio_suppression()

        # Use lightweight stdio server to avoid console output
        from .stdio_server import main as stdio_server_main
        stdio_server_main()
    else:
        # Use full server implementation for non-stdio modes
        from .server import main as server_main
        server_main()


if __name__ == "__main__":
    main()