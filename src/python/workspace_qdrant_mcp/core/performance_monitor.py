"""
Integrated Performance Monitoring System for Workspace Qdrant MCP.

This module provides the main performance monitoring coordinator that integrates
metrics collection, analytics, storage, and optimization recommendations into
a unified system for daemon performance management. This is a simplified wrapper
around the comprehensive monitor module in the common core directory.

Task 265: Performance monitoring interface for the workspace_qdrant_mcp namespace.
"""

# Import all classes from the comprehensive monitor module
import sys
from pathlib import Path

# Add the common module path
sys.path.append(str(Path(__file__).parent.parent.parent / "common"))

from core.performance_monitor import (
    PerformanceAlert,
    PerformanceMonitor,
    get_performance_monitor,
    stop_performance_monitor,
    get_all_performance_summaries,
    cleanup_all_performance_monitors
)

# Re-export all classes
__all__ = [
    "PerformanceAlert",
    "PerformanceMonitor",
    "get_performance_monitor",
    "stop_performance_monitor",
    "get_all_performance_summaries",
    "cleanup_all_performance_monitors"
]