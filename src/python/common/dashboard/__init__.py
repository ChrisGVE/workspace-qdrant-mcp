"""
Dashboard module for performance monitoring and visualization.

This module provides web-based dashboards and visualization components
for monitoring search performance, accuracy metrics, and baseline compliance.

Task 233.6: Performance monitoring and benchmarking for metadata filtering.
"""

from .performance_dashboard import PerformanceDashboardServer, create_dashboard_server

__all__ = [
    "PerformanceDashboardServer",
    "create_dashboard_server"
]
