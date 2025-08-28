"""MCP tools for workspace operations."""

from .documents import add_document, delete_document, get_document, update_document
from .scratchbook import ScratchbookManager, update_scratchbook
from .search import search_collection_by_metadata, search_workspace

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
