"""
Observability module for workspace-qdrant-mcp.

This module provides structured logging, metrics collection, and monitoring
capabilities for production deployments.

Key Components:
    - Structured logging with JSON format and proper log levels
    - Metrics collection compatible with Prometheus
    - Health check endpoints for monitoring systems
    - Performance monitoring for key operations
    - Error tracking and alerting support

Example:
    ```python
    from workspace_qdrant_mcp.observability import get_logger, MetricsCollector
    
    logger = get_logger(__name__)
    metrics = MetricsCollector()
    
    logger.info("Operation started", operation="search", query="example")
    with metrics.timer("search_duration"):
        # Perform search operation
        pass
    ```
"""

from .logger import get_logger, configure_logging, LogContext, PerformanceLogger
from .metrics import MetricsCollector, metrics_instance, record_operation
from .health import HealthChecker, health_checker_instance, HealthStatus
from .monitoring import (
    OperationMonitor, 
    monitor_async, 
    monitor_sync, 
    monitor_performance,
    BatchOperationMonitor,
    monitor_batch_operation,
    async_operation_monitor
)

__all__ = [
    "get_logger",
    "configure_logging", 
    "LogContext",
    "PerformanceLogger",
    "MetricsCollector",
    "metrics_instance",
    "record_operation",
    "HealthChecker",
    "health_checker_instance",
    "HealthStatus",
    "OperationMonitor",
    "monitor_async",
    "monitor_sync",
    "monitor_performance",
    "BatchOperationMonitor",
    "monitor_batch_operation",
    "async_operation_monitor",
]