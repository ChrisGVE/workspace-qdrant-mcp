"""
Main entry point for the Workspace Qdrant MCP server.
Delegates to the main server implementation.
"""

from .server import main

if __name__ == "__main__":
    main()