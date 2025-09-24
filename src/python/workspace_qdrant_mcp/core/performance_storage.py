"""
Performance Storage System for Workspace Qdrant MCP.

This module provides persistent storage capabilities for performance metrics,
operation traces, and analysis reports. This is a simplified wrapper around
the comprehensive storage module in the common core directory.

Task 265: Performance storage interface for the workspace_qdrant_mcp namespace.
"""

# Import all classes from the comprehensive storage module
import sys
from pathlib import Path

# Add the common module path
sys.path.append(str(Path(__file__).parent.parent.parent / "common"))

from core.performance_storage import (
    PerformanceStorage,
    get_performance_storage
)

# Re-export all classes
__all__ = [
    "PerformanceStorage",
    "get_performance_storage"
]