"""
Performance Analytics and Optimization Engine for Workspace Qdrant MCP.

This module provides intelligent analysis of performance metrics and generates
optimization recommendations based on usage patterns, resource utilization,
and performance trends. This is a simplified wrapper around the comprehensive
analytics module in the common core directory.

Task 265: Performance analytics interface for the workspace_qdrant_mcp namespace.
"""

# Import all classes from the comprehensive analytics module
import sys
from pathlib import Path

# Add the common module path
sys.path.append(str(Path(__file__).parent.parent.parent / "common"))

from core.performance_analytics import (
    OptimizationType,
    Priority,
    OptimizationRecommendation,
    PerformanceInsight,
    PerformanceReport,
    PerformanceAnalyzer,
    OptimizationEngine
)

# Re-export all classes
__all__ = [
    "OptimizationType",
    "Priority",
    "OptimizationRecommendation",
    "PerformanceInsight",
    "PerformanceReport",
    "PerformanceAnalyzer",
    "OptimizationEngine"
]