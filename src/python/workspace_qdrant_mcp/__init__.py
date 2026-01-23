"""
workspace-qdrant-mcp: MCP Server for project-scoped Qdrant operations.

A Model Context Protocol (MCP) server that provides intelligent
vector database operations with automatic project detection, hybrid search
capabilities, and cross-project scratchbook functionality.

Key Features:
    **Project-Aware Collections**: Automatically detects and manages project structure
    **Hybrid Search**: Combines dense semantic + sparse keyword search with RRF fusion
    **Universal Scratchbook**: Cross-project note-taking and knowledge management
    **High Performance**: Evidence-based 100% precision for exact matches
    **Production Ready**: Comprehensive error handling, validation, and logging

MCP Tools Available:
    - workspace_status: Get comprehensive workspace diagnostics
    - search_workspace: Hybrid search across all collections
    - add_document: Add documents with intelligent chunking
    - get_document: Retrieve documents with metadata
    - update_scratchbook: Manage cross-project notes
    - search_scratchbook: Find notes across projects
    - hybrid_search_advanced: Advanced search with custom parameters
    - And more...
"""

__version__ = "0.4.0"
__author__ = "Chris"
__email__ = "chris@example.com"
__description__ = "Advanced project-scoped Qdrant MCP server with hybrid search"
__url__ = "https://github.com/your-org/workspace-qdrant-mcp"

import os

# Core module import
try:
    from common import core
except ImportError:
    core = None

# Only import the main server if not in stdio mode to prevent import hangs
if os.getenv("WQM_STDIO_MODE", "").lower() != "true":
    try:
        from .server import app
        __all__ = ["app", "core"] if core else ["app"]
    except ImportError:
        __all__ = ["core"] if core else []
else:
    # In stdio mode, don't import anything to avoid hangs
    __all__ = ["core"] if core else []
