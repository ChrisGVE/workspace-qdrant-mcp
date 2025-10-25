"""
Conflict detector facade for workspace-qdrant-mcp.

This module re-exports ConflictDetector from common.memory.
"""

from common.memory.conflict_detector import ConflictDetector

__all__ = ["ConflictDetector"]
