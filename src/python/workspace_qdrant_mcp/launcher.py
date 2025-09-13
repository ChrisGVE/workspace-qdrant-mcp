"""
Smart launcher that chooses between full server and stdio-optimized server.

This launcher detects the transport mode and runtime environment to select
the appropriate server implementation, avoiding import hangs in stdio mode.
"""

import os
import sys


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

    # Check if stdout is piped (typical MCP scenario)
    if hasattr(sys.stdout, 'isatty') and not sys.stdout.isatty():
        if os.getenv("TERM") is None:
            return True

    return False


def main():
    """Main launcher entry point."""

    if detect_stdio_mode():
        # Use lightweight stdio server to avoid import hangs
        from .stdio_server import main as stdio_main
        stdio_main()
    else:
        # Use full server for HTTP and other transports
        from .server import main as server_main
        server_main()


if __name__ == "__main__":
    main()