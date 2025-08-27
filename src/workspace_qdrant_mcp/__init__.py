"""
workspace-qdrant-mcp: Project-scoped Qdrant MCP server with scratchbook functionality.

A Python port of claude-qdrant-mcp with FastEmbed integration and project-aware 
collection management.
"""

__version__ = "0.1.0"
__author__ = "Chris"
__email__ = "chris@example.com"

from .server import app

__all__ = ["app"]