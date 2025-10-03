"""Error Monitoring System

Provides pluggable monitoring backends, metrics collection, and health checks
for error management with support for Prometheus, CloudWatch, webhooks, and logging.

Features:
    - MonitoringHook abstract base class for pluggable backends
    - Built-in hooks: Logging, Webhook, Prometheus (optional), CloudWatch (optional)
    - ErrorMetricsCollector for aggregating and emitting metrics
    - HealthCheckManager for error-based health status
    - Integration with ErrorMessageManager

Example:
    ```python
    from workspace_qdrant_mcp.core.error_monitoring import (
        ErrorMetricsCollector,
        HealthCheckManager,
        LoggingHook,
        WebhookHook
    )

    # Initialize metrics collector
    collector = ErrorMetricsCollector()

    # Register hooks
    collector.register_hook(LoggingHook())
    collector.register_hook(WebhookHook("https://example.com/webhook"))

    # Emit metrics
    collector.emit_error_metric(error_message)
    collector.emit_error_rate(5.2, "1m")

    # Health checks
    health_mgr = HealthCheckManager(error_manager)
    await health_mgr.initialize()
    status = await health_mgr.get_health_status()
    ```
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

# Import type for ErrorMessage
from .error_message_manager import ErrorMessage, ErrorMessageManager


class HealthStatus(Enum):
    """Health status levels for monitoring."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

    def to_numeric(self) -> int:
        """Convert health status to numeric value for metrics."""
        return {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 1,
            HealthStatus.UNHEALTHY: 2
        }[self]


@dataclass
class OverallHealthStatus:
    """Overall health status with details."""

    status: HealthStatus
    checks: Dict[str, HealthStatus] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "checks": {name: status.value for name, status in self.checks.items()},
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class MonitoringHook(ABC):
    """Abstract base class for monitoring backends.

    Implement this interface to create custom monitoring hooks that
    can emit errors, metrics, and health status to external systems.
    """

    @abstractmethod
    async def emit_error(self, error: ErrorMessage) -> None:
        """Emit an error message to the monitoring backend.

        Args:
            error: ErrorMessage instance
        """
        pass

    @abstractmethod
    async def emit_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Emit a metric to the monitoring backend.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional metric labels/tags
        """
        pass

    @abstractmethod
    async def emit_health_status(self, status: OverallHealthStatus) -> None:
        """Emit health status to the monitoring backend.

        Args:
            status: Overall health status
        """
        pass

    async def close(self) -> None:
        """Close and cleanup resources. Override if needed."""
        pass


class LoggingHook(MonitoringHook):
    """Monitoring hook that logs to loguru.

    Always available, provides basic monitoring via structured logging.
    """

    async def emit_error(self, error: ErrorMessage) -> None:
        """Log error message."""
        logger.info(
            "Error metric",
            error_id=error.id,
            severity=error.severity.value,
            category=error.category.value,
            message=error.message[:100]
        )

    async def emit_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Log metric."""
        logger.debug(
            f"Metric: {name}",
            metric_name=name,
            value=value,
            labels=labels or {}
        )

    async def emit_health_status(self, status: OverallHealthStatus) -> None:
        """Log health status."""
        log_level = {
            HealthStatus.HEALTHY: "info",
            HealthStatus.DEGRADED: "warning",
            HealthStatus.UNHEALTHY: "error"
        }[status.status]

        logger.log(
            log_level.upper(),
            f"Health status: {status.status.value}",
            checks=status.checks,
            details=status.details
        )


class WebhookHook(MonitoringHook):
    """Monitoring hook that sends JSON payloads to HTTP endpoints.

    Sends error events, metrics, and health status to a configured webhook URL.
    """

    def __init__(self, webhook_url: str, timeout: float = 5.0):
        """Initialize webhook hook.

        Args:
            webhook_url: HTTP endpoint URL
            timeout: Request timeout in seconds
        """
        self.webhook_url = webhook_url
        self.timeout = timeout
        self._client: Optional[Any] = None

    async def _get_client(self):
        """Get or create httpx async client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(timeout=self.timeout)
            except ImportError:
                logger.warning("httpx not available, webhook monitoring disabled")
                return None
        return self._client

    async def emit_error(self, error: ErrorMessage) -> None:
        """Send error to webhook."""
        client = await self._get_client()
        if not client:
            return

        payload = {
            "event": "error",
            "data": error.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        try:
            await client.post(self.webhook_url, json=payload)
            logger.debug(f"Sent error to webhook: {error.id}")
        except Exception as e:
            logger.error(f"Failed to send error to webhook: {e}")

    async def emit_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Send metric to webhook."""
        client = await self._get_client()
        if not client:
            return

        payload = {
            "event": "metric",
            "data": {
                "name": name,
                "value": value,
                "labels": labels or {}
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        try:
            await client.post(self.webhook_url, json=payload)
            logger.debug(f"Sent metric to webhook: {name}={value}")
        except Exception as e:
            logger.error(f"Failed to send metric to webhook: {e}")

    async def emit_health_status(self, status: OverallHealthStatus) -> None:
        """Send health status to webhook."""
        client = await self._get_client()
        if not client:
            return

        payload = {
            "event": "health_status",
            "data": status.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        try:
            await client.post(self.webhook_url, json=payload)
            logger.debug(f"Sent health status to webhook: {status.status.value}")
        except Exception as e:
            logger.error(f"Failed to send health status to webhook: {e}")

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class PrometheusHook(MonitoringHook):
    """Monitoring hook for Prometheus metrics.

    Requires prometheus_client package (optional dependency).
    """

    def __init__(self):
        """Initialize Prometheus hook."""
        self._available = False
        self._metrics: Dict[str, Any] = {}

        try:
            from prometheus_client import Counter, Gauge, Histogram
            self._Counter = Counter
            self._Gauge = Gauge
            self._Histogram = Histogram
            self._available = True

            # Initialize standard metrics
            self._metrics["error_total"] = Counter(
                "wqm_error_total",
                "Total error count",
                ["severity", "category"]
            )
            self._metrics["error_rate"] = Gauge(
                "wqm_error_rate",
                "Error rate per minute"
            )
            self._metrics["acknowledgment_time"] = Histogram(
                "wqm_acknowledgment_time_seconds",
                "Time to acknowledge errors"
            )
            self._metrics["unacknowledged_errors"] = Gauge(
                "wqm_unacknowledged_errors",
                "Number of unacknowledged errors",
                ["severity"]
            )
            self._metrics["health_status"] = Gauge(
                "wqm_error_health_status",
                "Error system health status"
            )

            logger.info("Prometheus monitoring hook initialized")
        except ImportError:
            logger.debug("prometheus_client not available, Prometheus monitoring disabled")

    async def emit_error(self, error: ErrorMessage) -> None:
        """Increment error counter."""
        if not self._available:
            return

        self._metrics["error_total"].labels(
            severity=error.severity.value,
            category=error.category.value
        ).inc()

    async def emit_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Set Prometheus metric."""
        if not self._available:
            return

        # Map common metric names to Prometheus metrics
        if name == "error_rate" and "error_rate" in self._metrics:
            self._metrics["error_rate"].set(value)
        elif name == "acknowledgment_time_seconds" and "acknowledgment_time" in self._metrics:
            self._metrics["acknowledgment_time"].observe(value)
        elif name.startswith("unacknowledged_errors_") and "unacknowledged_errors" in self._metrics:
            severity = name.replace("unacknowledged_errors_", "")
            self._metrics["unacknowledged_errors"].labels(severity=severity).set(value)

    async def emit_health_status(self, status: OverallHealthStatus) -> None:
        """Set health status metric."""
        if not self._available:
            return

        self._metrics["health_status"].set(status.status.to_numeric())


class CloudWatchHook(MonitoringHook):
    """Monitoring hook for AWS CloudWatch.

    Requires boto3 package (optional dependency).
    """

    def __init__(self, namespace: str = "WorkspaceQdrant"):
        """Initialize CloudWatch hook.

        Args:
            namespace: CloudWatch namespace for metrics
        """
        self.namespace = namespace
        self._available = False
        self._client: Optional[Any] = None

        try:
            import boto3
            self._client = boto3.client('cloudwatch')
            self._available = True
            logger.info("CloudWatch monitoring hook initialized")
        except ImportError:
            logger.debug("boto3 not available, CloudWatch monitoring disabled")

    async def emit_error(self, error: ErrorMessage) -> None:
        """Send error metric to CloudWatch."""
        if not self._available or not self._client:
            return

        try:
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{
                    'MetricName': 'ErrorCount',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Severity', 'Value': error.severity.value},
                        {'Name': 'Category', 'Value': error.category.value}
                    ],
                    'Timestamp': error.timestamp
                }]
            )
        except Exception as e:
            logger.error(f"Failed to send error metric to CloudWatch: {e}")

    async def emit_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Send metric to CloudWatch."""
        if not self._available or not self._client:
            return

        try:
            metric_data = {
                'MetricName': name,
                'Value': value,
                'Unit': 'None',
                'Timestamp': datetime.now(timezone.utc)
            }

            if labels:
                metric_data['Dimensions'] = [
                    {'Name': k, 'Value': v} for k, v in labels.items()
                ]

            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[metric_data]
            )
        except Exception as e:
            logger.error(f"Failed to send metric to CloudWatch: {e}")

    async def emit_health_status(self, status: OverallHealthStatus) -> None:
        """Send health status to CloudWatch."""
        if not self._available or not self._client:
            return

        try:
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=[{
                    'MetricName': 'HealthStatus',
                    'Value': status.status.to_numeric(),
                    'Unit': 'None',
                    'Timestamp': status.timestamp
                }]
            )
        except Exception as e:
            logger.error(f"Failed to send health status to CloudWatch: {e}")


class ErrorMetricsCollector:
    """Collects and emits error metrics to registered hooks.

    Aggregates error metrics and distributes them to all registered
    monitoring hooks (Prometheus, CloudWatch, webhooks, logging, etc.).
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.hooks: List[MonitoringHook] = []
        self._metrics: Dict[str, Any] = {
            "error_total": {},
            "error_rate": 0.0,
            "acknowledgment_times": [],
            "unacknowledged_errors": {}
        }
        self._rate_task: Optional[asyncio.Task] = None

    def register_hook(self, hook: MonitoringHook) -> bool:
        """Register a monitoring hook.

        Args:
            hook: MonitoringHook instance

        Returns:
            True if registered successfully
        """
        if hook not in self.hooks:
            self.hooks.append(hook)
            logger.info(f"Registered monitoring hook: {hook.__class__.__name__}")
            return True
        return False

    async def emit_error_metric(self, error: ErrorMessage) -> None:
        """Emit error metric to all hooks.

        Args:
            error: ErrorMessage instance
        """
        # Update internal metrics
        key = f"{error.severity.value}_{error.category.value}"
        self._metrics["error_total"][key] = self._metrics["error_total"].get(key, 0) + 1

        # Emit to all hooks
        for hook in self.hooks:
            try:
                await hook.emit_error(error)
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__} failed to emit error: {e}")

    async def emit_error_rate(self, rate: float, window: str) -> None:
        """Emit error rate metric.

        Args:
            rate: Error rate (errors per minute)
            window: Time window (e.g., "1m", "5m")
        """
        self._metrics["error_rate"] = rate

        for hook in self.hooks:
            try:
                await hook.emit_metric("error_rate", rate, {"window": window})
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__} failed to emit rate: {e}")

    async def emit_acknowledgment_metric(self, ack_time_seconds: float) -> None:
        """Emit acknowledgment time metric.

        Args:
            ack_time_seconds: Time to acknowledge in seconds
        """
        self._metrics["acknowledgment_times"].append(ack_time_seconds)

        for hook in self.hooks:
            try:
                await hook.emit_metric("acknowledgment_time_seconds", ack_time_seconds)
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__} failed to emit ack time: {e}")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot.

        Returns:
            Dictionary with current metrics
        """
        return {
            "error_total": dict(self._metrics["error_total"]),
            "error_rate": self._metrics["error_rate"],
            "acknowledgment_times": {
                "count": len(self._metrics["acknowledgment_times"]),
                "avg": sum(self._metrics["acknowledgment_times"]) / len(self._metrics["acknowledgment_times"])
                    if self._metrics["acknowledgment_times"] else 0,
                "min": min(self._metrics["acknowledgment_times"])
                    if self._metrics["acknowledgment_times"] else 0,
                "max": max(self._metrics["acknowledgment_times"])
                    if self._metrics["acknowledgment_times"] else 0
            },
            "unacknowledged_errors": dict(self._metrics["unacknowledged_errors"])
        }

    async def close(self) -> None:
        """Close all hooks and cleanup."""
        if self._rate_task:
            self._rate_task.cancel()
            try:
                await self._rate_task
            except asyncio.CancelledError:
                pass

        for hook in self.hooks:
            try:
                await hook.close()
            except Exception as e:
                logger.error(f"Hook {hook.__class__.__name__} failed to close: {e}")


class HealthCheckManager:
    """Manages health checks based on error metrics.

    Performs health checks based on error thresholds, acknowledgment rates,
    and custom health check functions.
    """

    def __init__(self, error_manager: ErrorMessageManager):
        """Initialize health check manager.

        Args:
            error_manager: ErrorMessageManager instance
        """
        self.error_manager = error_manager
        self._custom_checks: Dict[str, Callable[[], HealthStatus]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize health check manager."""
        if not self._initialized:
            await self.error_manager.initialize()
            self._initialized = True

    async def check_error_threshold(
        self,
        severity: str,
        threshold: int,
        window_minutes: int
    ) -> HealthStatus:
        """Check if error count exceeds threshold.

        Args:
            severity: Error severity to check
            threshold: Maximum allowed errors
            window_minutes: Time window in minutes

        Returns:
            HealthStatus based on threshold
        """
        start_date = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        errors = await self.error_manager.get_errors(
            severity=severity,
            start_date=start_date,
            limit=threshold + 1
        )

        count = len(errors)

        if count >= threshold:
            return HealthStatus.UNHEALTHY
        elif count >= threshold * 0.8:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def check_acknowledgment_rate(
        self,
        min_rate: float,
        window_minutes: int
    ) -> HealthStatus:
        """Check acknowledgment rate.

        Args:
            min_rate: Minimum acceptable acknowledgment rate (0.0-1.0)
            window_minutes: Time window in minutes

        Returns:
            HealthStatus based on acknowledgment rate
        """
        start_date = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

        # Get all errors in window
        all_errors = await self.error_manager.get_errors(
            start_date=start_date,
            limit=1000
        )

        if not all_errors:
            return HealthStatus.HEALTHY

        # Calculate acknowledgment rate
        acknowledged = sum(1 for e in all_errors if e.acknowledged)
        rate = acknowledged / len(all_errors)

        if rate >= min_rate:
            return HealthStatus.HEALTHY
        elif rate >= min_rate * 0.7:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    async def get_health_status(self) -> OverallHealthStatus:
        """Get overall health status.

        Returns:
            OverallHealthStatus with all check results
        """
        checks: Dict[str, HealthStatus] = {}
        details: Dict[str, Any] = {}

        # Standard checks
        try:
            # Check error threshold (last 5 minutes)
            error_check = await self.check_error_threshold("error", 10, 5)
            checks["error_threshold"] = error_check

            # Check acknowledgment rate (last 60 minutes)
            ack_check = await self.check_acknowledgment_rate(0.8, 60)
            checks["acknowledgment_rate"] = ack_check

            # Get error stats
            stats = await self.error_manager.get_error_stats(
                start_date=datetime.now(timezone.utc) - timedelta(hours=1)
            )
            details["error_stats"] = stats.to_dict()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            checks["system"] = HealthStatus.UNHEALTHY
            details["error"] = str(e)

        # Run custom checks
        for name, check_func in self._custom_checks.items():
            try:
                checks[name] = check_func()
            except Exception as e:
                logger.error(f"Custom check '{name}' failed: {e}")
                checks[name] = HealthStatus.UNHEALTHY

        # Determine overall status
        if any(status == HealthStatus.UNHEALTHY for status in checks.values()):
            overall_status = HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in checks.values()):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return OverallHealthStatus(
            status=overall_status,
            checks=checks,
            details=details
        )

    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], HealthStatus]
    ) -> bool:
        """Register custom health check.

        Args:
            name: Check name
            check_func: Function that returns HealthStatus

        Returns:
            True if registered successfully
        """
        if name not in self._custom_checks:
            self._custom_checks[name] = check_func
            logger.info(f"Registered custom health check: {name}")
            return True
        return False

    async def close(self) -> None:
        """Close health check manager."""
        if self._initialized:
            await self.error_manager.close()
            self._initialized = False
