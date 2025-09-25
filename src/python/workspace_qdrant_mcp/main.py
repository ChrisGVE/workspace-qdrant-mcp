"""
Main entry point for the Workspace Qdrant MCP server.

Simple, clean implementation with exactly 4 tools:
1. store - Store any content (text, files, web URLs)
2. search - Search with hybrid semantic + keyword matching
3. manage - Manage collections, documents, and system
4. retrieve - Retrieve documents without search ranking

No complexity, no modes, no backward compatibility.
The content you pass determines what happens.
"""

import asyncio
import logging
import sys
from pathlib import Path

from .simple_server import create_simple_server


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stderr)
        ]
    )


def main():
    """Main entry point for the MCP server."""
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting Workspace Qdrant MCP Server with 4 tools")

    # Create the simple server
    app = create_simple_server()

    # Run the server
    try:
        app.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()