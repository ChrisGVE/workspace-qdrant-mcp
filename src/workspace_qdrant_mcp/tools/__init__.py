"""MCP tools for workspace operations."""

from .search import search_workspace, search_collection_by_metadata
from .documents import add_document, update_document, delete_document, get_document
from .scratchbook import update_scratchbook, ScratchbookManager

__all__ = [
    "search_workspace",
    "search_collection_by_metadata", 
    "add_document",
    "update_document",
    "delete_document",
    "get_document",
    "update_scratchbook",
    "ScratchbookManager",
]