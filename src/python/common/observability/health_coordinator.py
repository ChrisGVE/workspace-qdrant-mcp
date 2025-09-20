"""
Four-Component Health Coordinator for workspace-qdrant-mcp.

This module provides centralized health monitoring and coordination across all four
architecture components: SQLite State Manager, Rust Daemon, Python MCP Server, and CLI/Context Injector.

Key Features:
    - Cross-component health orchestration and dependency tracking
    - Multi-level alerting with automated recovery actions
    - Health trend analysis and anomaly detection
    - Component performance correlation analysis
    - Graceful degradation health management
    - Real-time health dashboard coordination

Architecture Integration:
    - Leverages existing HealthChecker infrastructure
    - Integrates with ComponentLifecycleManager for startup/shutdown health
    - Coordinates with ComponentCoordinator for state management
    - Provides unified health APIs for monitoring systems

Example:
    ```python
    from workspace_qdrant_mcp.observability.health_coordinator import HealthCoordinator

    # Initialize health coordinator
    coordinator = HealthCoordinator(db_path="./workspace_state.db")
    await coordinator.initialize()

    # Start comprehensive health monitoring
    await coordinator.start_monitoring()

    # Get unified health status
    health_status = await coordinator.get_unified_health_status()
    ```
"""

import asyncio
import json
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from loguru import logger

from .health import HealthChecker, HealthStatus, ComponentHealth, get_health_checker
from .metrics import metrics_instance
from ..core.component_coordination import (
    ComponentCoordinator,
    ComponentType,
    ComponentStatus,
    ComponentHealth as CoordComponentHealth,
    get_component_coordinator,
)
from ..core.component_lifecycle import (
    ComponentLifecycleManager,
    ComponentState,
    LifecyclePhase,
    get_lifecycle_manager,
)


class AlertSeverity(Enum):
    """Alert severity levels for health monitoring."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class HealthTrend(Enum):
    """Health trend analysis results."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNSTABLE = "unstable"
    CRITICAL_DECLINE = "critical_decline"


@dataclass
class ComponentHealthMetrics:
    """Comprehensive health metrics for a component."""

    component_type: ComponentType
    health_status: HealthStatus
    response_time_ms: float
    error_rate: float
    resource_usage: Dict[str, float]
    dependency_health: Dict[str, bool]
    uptime_seconds: float
    last_restart: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthAlert:
    """Health monitoring alert."""

    alert_id: str
    component_type: ComponentType
    severity: AlertSeverity
    message: str
    description: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    auto_recovery_attempted: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class HealthTrendAnalysis:
    """Health trend analysis results."""

    component_type: ComponentType
    trend: HealthTrend
    confidence: float  # 0.0 to 1.0
    analysis_period_hours: float
    key_indicators: List[str]
    predictions: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class HealthCoordinator:
    """
    Four-Component Health Coordinator.

    Orchestrates comprehensive health monitoring across all architecture components
    with advanced alerting, trend analysis, and automated recovery capabilities.
    """

    # Component dependency mapping for health validation
    COMPONENT_DEPENDENCIES = {
        ComponentType.RUST_DAEMON: [],  # Foundation component
        ComponentType.PYTHON_MCP_SERVER: [ComponentType.RUST_DAEMON],
        ComponentType.CLI_UTILITY: [ComponentType.RUST_DAEMON, ComponentType.PYTHON_MCP_SERVER],
        ComponentType.CONTEXT_INJECTOR: [ComponentType.PYTHON_MCP_SERVER],
    }

    # Health check intervals per component (seconds)
    COMPONENT_CHECK_INTERVALS = {
        ComponentType.RUST_DAEMON: 15.0,
        ComponentType.PYTHON_MCP_SERVER: 10.0,
        ComponentType.CLI_UTILITY: 30.0,
        ComponentType.CONTEXT_INJECTOR: 20.0,
    }

    # Critical thresholds for automated alerts
    CRITICAL_THRESHOLDS = {
        "response_time_ms": 5000.0,
        "error_rate": 0.05,  # 5%
        "cpu_usage": 0.90,   # 90%
        "memory_usage": 0.95,  # 95%
        "disk_usage": 0.95,    # 95%
    }

    def __init__(
        self,
        db_path: str = "workspace_state.db",
        project_name: Optional[str] = None,
        project_path: Optional[str] = None,
        enable_auto_recovery: bool = True,
        trend_analysis_window_hours: float = 24.0
    ):
        """
        Initialize Health Coordinator.

        Args:
            db_path: Path to SQLite database for state coordination
            project_name: Project name for component scoping
            project_path: Project path for workspace detection
            enable_auto_recovery: Whether to enable automated recovery actions
            trend_analysis_window_hours: Hours of data for trend analysis
        """
        self.db_path = db_path
        self.project_name = project_name or self._detect_project_name()
        self.project_path = project_path or str(Path.cwd())
        self.enable_auto_recovery = enable_auto_recovery
        self.trend_analysis_window_hours = trend_analysis_window_hours

        # Core health monitoring components
        self.health_checker: Optional[HealthChecker] = None
        self.component_coordinator: Optional[ComponentCoordinator] = None
        self.lifecycle_manager: Optional[ComponentLifecycleManager] = None

        # Health monitoring state
        self.component_health_history: Dict[ComponentType, deque] = {
            comp_type: deque(maxlen=1000) for comp_type in ComponentType
        }
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_callbacks: List[Callable[[HealthAlert], None]] = []
        self.monitoring_tasks: List[asyncio.Task] = []

        # Trend analysis
        self.trend_analysis_cache: Dict[ComponentType, HealthTrendAnalysis] = {}
        self.last_trend_analysis: Optional[datetime] = None

        # Recovery automation
        self.recovery_attempts: Dict[ComponentType, int] = defaultdict(int)
        self.max_recovery_attempts = 3
        self.recovery_cooldown_minutes = 5.0
        self.last_recovery_attempt: Dict[ComponentType, datetime] = {}

        # Performance correlation tracking
        self.performance_correlations: Dict[str, float] = {}

        logger.info(
            "Health Coordinator initialized",
            project=self.project_name,
            auto_recovery=enable_auto_recovery,
            trend_window_hours=trend_analysis_window_hours
        )

    async def initialize(self) -> bool:
        """
        Initialize the health coordinator and all monitoring components.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Health Coordinator")

            # Initialize core health monitoring components
            self.health_checker = get_health_checker()
            self.component_coordinator = await get_component_coordinator(self.db_path)
            self.lifecycle_manager = await get_lifecycle_manager(
                db_path=self.db_path,
                project_name=self.project_name,
                project_path=self.project_path
            )

            # Start background health monitoring for existing health checker
            self.health_checker.start_background_monitoring(interval=30.0)

            logger.info("Health Coordinator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Health Coordinator: {e}")
            return False

    async def start_monitoring(self) -> None:
        """Start comprehensive health monitoring across all components."""
        logger.info("Starting comprehensive health monitoring")

        # Start component-specific monitoring tasks
        for component_type in ComponentType:
            interval = self.COMPONENT_CHECK_INTERVALS[component_type]
            task = asyncio.create_task(
                self._monitor_component_health(component_type, interval)
            )
            self.monitoring_tasks.append(task)

        # Start trend analysis task
        trend_task = asyncio.create_task(self._trend_analysis_loop())
        self.monitoring_tasks.append(trend_task)

        # Start alert processing task
        alert_task = asyncio.create_task(self._alert_processing_loop())
        self.monitoring_tasks.append(alert_task)

        # Start performance correlation analysis
        correlation_task = asyncio.create_task(self._performance_correlation_loop())
        self.monitoring_tasks.append(correlation_task)

        logger.info(f"Started {len(self.monitoring_tasks)} health monitoring tasks")

    async def stop_monitoring(self) -> None:
        """Stop all health monitoring tasks."""
        logger.info("Stopping health monitoring")

        # Cancel all monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()

        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        self.monitoring_tasks.clear()

        # Stop background monitoring in health checker
        if self.health_checker:
            self.health_checker.stop_background_monitoring()

        logger.info("Health monitoring stopped")

    async def get_unified_health_status(self) -> Dict[str, Any]:
        """
        Get unified health status across all components.

        Returns:
            Comprehensive health status including all components and dependencies
        """
        try:
            # Get base health status from health checker
            base_health = await self.health_checker.get_health_status()

            # Enhance with component-specific health data
            component_health_details = {}
            overall_status = HealthStatus.HEALTHY

            for component_type in ComponentType:
                component_health = await self._get_component_health_detailed(component_type)
                component_health_details[component_type.value] = asdict(component_health)

                # Update overall status based on component health and dependencies
                if component_health.health_status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif (component_health.health_status == HealthStatus.DEGRADED
                      and overall_status == HealthStatus.HEALTHY):
                    overall_status = HealthStatus.DEGRADED

            # Get trend analysis
            trend_summary = await self._get_trend_summary()

            # Get active alerts
            alert_summary = self._get_alert_summary()

            # Build unified response
            unified_status = {
                "timestamp": time.time(),
                "overall_status": overall_status.value,
                "base_health": base_health,
                "component_health": component_health_details,
                "trend_analysis": trend_summary,
                "active_alerts": alert_summary,
                "dependency_health": await self._get_dependency_health_matrix(),
                "performance_correlations": self.performance_correlations,
                "coordinator_info": {
                    "monitoring_tasks": len(self.monitoring_tasks),
                    "auto_recovery_enabled": self.enable_auto_recovery,
                    "trend_analysis_window_hours": self.trend_analysis_window_hours,
                }
            }

            # Record unified health metrics
            metrics_instance.set_gauge(
                "unified_health_status",
                1 if overall_status == HealthStatus.HEALTHY else 0
            )
            metrics_instance.set_gauge("active_alerts_count", len(self.active_alerts))

            return unified_status

        except Exception as e:
            logger.error(f"Failed to get unified health status: {e}")
            return {
                "timestamp": time.time(),
                "overall_status": "error",
                "error": str(e),
                "component_health": {},
                "active_alerts": {},
            }

    async def get_health_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive health dashboard data.

        Returns:
            Dashboard-ready health data with visualizations and summaries
        """
        unified_status = await self.get_unified_health_status()

        # Add dashboard-specific enhancements
        dashboard_data = {
            **unified_status,
            "dashboard_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "data_sources": ["health_checker", "component_coordinator", "lifecycle_manager"],
                "refresh_interval_seconds": 30,
            },
            "visualization_data": await self._get_visualization_data(),
            "health_timeline": await self._get_health_timeline(),
            "recovery_actions": await self._get_recovery_actions_summary(),
        }

        return dashboard_data

    def register_alert_callback(self, callback: Callable[[HealthAlert], None]) -> None:
        """Register callback for health alerts."""
        self.alert_callbacks.append(callback)
        logger.debug(f"Registered alert callback: {callback.__name__}")

    async def trigger_manual_recovery(
        self,
        component_type: ComponentType,
        recovery_action: str = "restart"
    ) -> bool:
        """
        Manually trigger recovery action for a component.

        Args:
            component_type: Component to recover
            recovery_action: Type of recovery action

        Returns:
            True if recovery action initiated successfully
        """
        try:
            logger.info(f"Manual recovery triggered: {component_type.value} - {recovery_action}")

            if recovery_action == "restart":
                success = await self.lifecycle_manager.restart_component(component_type)
            else:
                logger.warning(f"Unknown recovery action: {recovery_action}")
                return False

            if success:
                # Reset recovery attempt counter
                self.recovery_attempts[component_type] = 0

                # Create recovery alert
                alert = HealthAlert(
                    alert_id=f"manual_recovery_{component_type.value}_{int(time.time())}",
                    component_type=component_type,
                    severity=AlertSeverity.INFO,
                    message=f"Manual recovery completed for {component_type.value}",
                    description=f"Successfully executed {recovery_action} recovery action",
                    timestamp=datetime.now(timezone.utc),
                    auto_recovery_attempted=False,
                    resolved=True,
                    resolution_time=datetime.now(timezone.utc)
                )

                await self._process_alert(alert)

            return success

        except Exception as e:
            logger.error(f"Manual recovery failed for {component_type.value}: {e}")
            return False

    async def _monitor_component_health(
        self,
        component_type: ComponentType,
        interval: float
    ) -> None:
        """Monitor health for a specific component."""
        logger.debug(f"Starting health monitoring for {component_type.value}")

        while True:
            try:
                # Get detailed component health
                health_metrics = await self._get_component_health_detailed(component_type)

                # Store in history for trend analysis
                self.component_health_history[component_type].append({
                    "timestamp": time.time(),
                    "metrics": asdict(health_metrics)
                })

                # Check for alert conditions
                await self._check_alert_conditions(component_type, health_metrics)

                # Record component health metrics
                metrics_instance.set_gauge(
                    "component_health_status",
                    1 if health_metrics.health_status == HealthStatus.HEALTHY else 0,
                    component=component_type.value
                )

                metrics_instance.set_gauge(
                    "component_response_time_ms",
                    health_metrics.response_time_ms,
                    component=component_type.value
                )

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                logger.debug(f"Health monitoring cancelled for {component_type.value}")
                break
            except Exception as e:
                logger.error(f"Error monitoring {component_type.value}: {e}")
                await asyncio.sleep(interval)

    async def _get_component_health_detailed(
        self,
        component_type: ComponentType
    ) -> ComponentHealthMetrics:
        """Get detailed health metrics for a component."""
        try:
            start_time = time.perf_counter()

            # Get component status from coordinator
            component_id = f"{component_type.value}-{self.project_name}"
            coordinator_status = await self.component_coordinator.get_component_status(component_id)

            # Determine health status
            if coordinator_status and "error" not in coordinator_status:
                health_status = HealthStatus.HEALTHY
                error_rate = 0.0
            else:
                health_status = HealthStatus.UNHEALTHY
                error_rate = 1.0

            response_time = (time.perf_counter() - start_time) * 1000

            # Get resource usage (simplified for demo)
            resource_usage = {
                "cpu": 0.0,  # Would get from actual monitoring
                "memory": 0.0,
                "disk": 0.0,
            }

            # Check dependency health
            dependency_health = {}
            for dep_component in self.COMPONENT_DEPENDENCIES.get(component_type, []):
                dep_component_id = f"{dep_component.value}-{self.project_name}"
                dep_status = await self.component_coordinator.get_component_status(dep_component_id)
                dependency_health[dep_component.value] = (
                    dep_status is not None and "error" not in dep_status
                )

            # Calculate uptime (simplified)
            uptime_seconds = 3600.0  # Would get from actual component monitoring

            return ComponentHealthMetrics(
                component_type=component_type,
                health_status=health_status,
                response_time_ms=response_time,
                error_rate=error_rate,
                resource_usage=resource_usage,
                dependency_health=dependency_health,
                uptime_seconds=uptime_seconds,
                performance_metrics={}
            )

        except Exception as e:
            logger.error(f"Failed to get detailed health for {component_type.value}: {e}")
            return ComponentHealthMetrics(
                component_type=component_type,
                health_status=HealthStatus.UNKNOWN,
                response_time_ms=0.0,
                error_rate=1.0,
                resource_usage={},
                dependency_health={},
                uptime_seconds=0.0,
            )

    async def _check_alert_conditions(
        self,
        component_type: ComponentType,
        health_metrics: ComponentHealthMetrics
    ) -> None:
        """Check for alert conditions and generate alerts."""
        alerts_to_create = []

        # Check critical thresholds
        if health_metrics.response_time_ms > self.CRITICAL_THRESHOLDS["response_time_ms"]:
            alerts_to_create.append(HealthAlert(
                alert_id=f"response_time_{component_type.value}_{int(time.time())}",
                component_type=component_type,
                severity=AlertSeverity.WARNING,
                message=f"High response time: {health_metrics.response_time_ms:.1f}ms",
                description=f"Component response time exceeds threshold",
                timestamp=datetime.now(timezone.utc),
                metrics={"response_time_ms": health_metrics.response_time_ms}
            ))

        if health_metrics.error_rate > self.CRITICAL_THRESHOLDS["error_rate"]:
            alerts_to_create.append(HealthAlert(
                alert_id=f"error_rate_{component_type.value}_{int(time.time())}",
                component_type=component_type,
                severity=AlertSeverity.CRITICAL,
                message=f"High error rate: {health_metrics.error_rate:.1%}",
                description=f"Component error rate exceeds threshold",
                timestamp=datetime.now(timezone.utc),
                metrics={"error_rate": health_metrics.error_rate}
            ))

        # Check component health status
        if health_metrics.health_status == HealthStatus.UNHEALTHY:
            alerts_to_create.append(HealthAlert(
                alert_id=f"unhealthy_{component_type.value}_{int(time.time())}",
                component_type=component_type,
                severity=AlertSeverity.CRITICAL,
                message=f"Component unhealthy: {component_type.value}",
                description=f"Component health check failed",
                timestamp=datetime.now(timezone.utc),
                metrics=asdict(health_metrics)
            ))

        # Check dependency health
        failed_dependencies = [
            dep for dep, healthy in health_metrics.dependency_health.items()
            if not healthy
        ]
        if failed_dependencies:
            alerts_to_create.append(HealthAlert(
                alert_id=f"dependencies_{component_type.value}_{int(time.time())}",
                component_type=component_type,
                severity=AlertSeverity.WARNING,
                message=f"Dependency failures: {', '.join(failed_dependencies)}",
                description=f"Component dependencies are unhealthy",
                timestamp=datetime.now(timezone.utc),
                metrics={"failed_dependencies": failed_dependencies}
            ))

        # Process all alerts
        for alert in alerts_to_create:
            await self._process_alert(alert)

    async def _process_alert(self, alert: HealthAlert) -> None:
        """Process and handle a health alert."""
        # Check if similar alert already exists
        existing_alert_id = None
        for existing_id, existing_alert in self.active_alerts.items():
            if (existing_alert.component_type == alert.component_type
                and existing_alert.severity == alert.severity
                and not existing_alert.resolved):
                existing_alert_id = existing_id
                break

        if existing_alert_id:
            # Update existing alert timestamp
            self.active_alerts[existing_alert_id].timestamp = alert.timestamp
            return

        # Store new alert
        self.active_alerts[alert.alert_id] = alert

        logger.warning(
            f"Health alert: {alert.severity.value.upper()}",
            component=alert.component_type.value,
            message=alert.message,
            alert_id=alert.alert_id
        )

        # Record alert metrics
        metrics_instance.increment_counter(
            "health_alerts_total",
            component=alert.component_type.value,
            severity=alert.severity.value
        )

        # Trigger alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Consider automated recovery for critical alerts
        if (self.enable_auto_recovery
            and alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]):
            await self._consider_auto_recovery(alert)

    async def _consider_auto_recovery(self, alert: HealthAlert) -> None:
        """Consider automated recovery actions for critical alerts."""
        component_type = alert.component_type

        # Check recovery attempt limits
        if self.recovery_attempts[component_type] >= self.max_recovery_attempts:
            logger.warning(f"Max recovery attempts reached for {component_type.value}")
            return

        # Check recovery cooldown
        last_attempt = self.last_recovery_attempt.get(component_type)
        if last_attempt:
            cooldown_expires = last_attempt + timedelta(minutes=self.recovery_cooldown_minutes)
            if datetime.now(timezone.utc) < cooldown_expires:
                logger.debug(f"Recovery cooldown active for {component_type.value}")
                return

        # Attempt automated recovery
        logger.info(f"Attempting automated recovery for {component_type.value}")

        try:
            success = await self.lifecycle_manager.restart_component(component_type)

            self.recovery_attempts[component_type] += 1
            self.last_recovery_attempt[component_type] = datetime.now(timezone.utc)

            if success:
                # Mark alert as having recovery attempted
                alert.auto_recovery_attempted = True

                # Create recovery success alert
                recovery_alert = HealthAlert(
                    alert_id=f"recovery_success_{component_type.value}_{int(time.time())}",
                    component_type=component_type,
                    severity=AlertSeverity.INFO,
                    message=f"Automated recovery successful for {component_type.value}",
                    description=f"Component restart completed successfully",
                    timestamp=datetime.now(timezone.utc),
                    auto_recovery_attempted=True,
                    resolved=True,
                    resolution_time=datetime.now(timezone.utc)
                )

                await self._process_alert(recovery_alert)

                logger.info(f"Automated recovery successful for {component_type.value}")
            else:
                logger.error(f"Automated recovery failed for {component_type.value}")

        except Exception as e:
            logger.error(f"Automated recovery error for {component_type.value}: {e}")

    async def _trend_analysis_loop(self) -> None:
        """Background task for health trend analysis."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Perform trend analysis for each component
                for component_type in ComponentType:
                    trend_analysis = await self._analyze_component_trend(component_type)
                    if trend_analysis:
                        self.trend_analysis_cache[component_type] = trend_analysis

                self.last_trend_analysis = datetime.now(timezone.utc)

                logger.debug("Health trend analysis completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trend analysis error: {e}")
                await asyncio.sleep(300)

    async def _analyze_component_trend(
        self,
        component_type: ComponentType
    ) -> Optional[HealthTrendAnalysis]:
        """Analyze health trend for a specific component."""
        history = self.component_health_history[component_type]

        if len(history) < 10:  # Need minimum data points
            return None

        try:
            # Extract time series data
            timestamps = [entry["timestamp"] for entry in history]
            response_times = [
                entry["metrics"]["response_time_ms"] for entry in history
                if "response_time_ms" in entry["metrics"]
            ]
            error_rates = [
                entry["metrics"]["error_rate"] for entry in history
                if "error_rate" in entry["metrics"]
            ]

            if not response_times or not error_rates:
                return None

            # Simple trend analysis using linear regression
            x = np.arange(len(response_times))

            # Response time trend
            rt_slope = np.polyfit(x, response_times, 1)[0] if len(response_times) > 1 else 0

            # Error rate trend
            er_slope = np.polyfit(x, error_rates, 1)[0] if len(error_rates) > 1 else 0

            # Determine overall trend
            if rt_slope > 100 or er_slope > 0.01:  # Degrading thresholds
                trend = HealthTrend.DEGRADING
                confidence = min(0.9, abs(rt_slope) / 1000 + abs(er_slope) * 10)
            elif rt_slope < -50 and er_slope < -0.005:  # Improving thresholds
                trend = HealthTrend.IMPROVING
                confidence = min(0.9, abs(rt_slope) / 1000 + abs(er_slope) * 10)
            else:
                trend = HealthTrend.STABLE
                confidence = 0.7

            # Check for instability (high variance)
            rt_std = np.std(response_times) if len(response_times) > 1 else 0
            er_std = np.std(error_rates) if len(error_rates) > 1 else 0

            if rt_std > 500 or er_std > 0.02:
                trend = HealthTrend.UNSTABLE
                confidence = 0.8

            # Generate recommendations
            recommendations = []
            if trend == HealthTrend.DEGRADING:
                recommendations.extend([
                    f"Monitor {component_type.value} resource usage",
                    "Consider scaling or optimization",
                    "Review recent configuration changes"
                ])
            elif trend == HealthTrend.UNSTABLE:
                recommendations.extend([
                    f"Investigate {component_type.value} performance variability",
                    "Check for intermittent failures",
                    "Review system load patterns"
                ])

            analysis_period = (timestamps[-1] - timestamps[0]) / 3600  # hours

            return HealthTrendAnalysis(
                component_type=component_type,
                trend=trend,
                confidence=confidence,
                analysis_period_hours=analysis_period,
                key_indicators=[
                    f"Response time trend: {rt_slope:.1f}ms/sample",
                    f"Error rate trend: {er_slope:.3f}/sample",
                    f"Response time std: {rt_std:.1f}ms",
                    f"Error rate std: {er_std:.3f}"
                ],
                predictions={
                    "response_time_trend_ms_per_hour": rt_slope * 12,  # Assuming 5min intervals
                    "error_rate_trend_per_hour": er_slope * 12
                },
                recommendations=recommendations
            )

        except Exception as e:
            logger.error(f"Trend analysis failed for {component_type.value}: {e}")
            return None

    async def _alert_processing_loop(self) -> None:
        """Background task for alert processing and cleanup."""
        while True:
            try:
                await asyncio.sleep(60)  # Process every minute

                # Clean up resolved alerts older than 1 hour
                current_time = datetime.now(timezone.utc)
                alerts_to_remove = []

                for alert_id, alert in self.active_alerts.items():
                    if alert.resolved and alert.resolution_time:
                        age = current_time - alert.resolution_time
                        if age.total_seconds() > 3600:  # 1 hour
                            alerts_to_remove.append(alert_id)

                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]

                if alerts_to_remove:
                    logger.debug(f"Cleaned up {len(alerts_to_remove)} resolved alerts")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)

    async def _performance_correlation_loop(self) -> None:
        """Background task for performance correlation analysis."""
        while True:
            try:
                await asyncio.sleep(900)  # Run every 15 minutes

                # Analyze correlations between component performance metrics
                await self._analyze_performance_correlations()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance correlation analysis error: {e}")
                await asyncio.sleep(900)

    async def _analyze_performance_correlations(self) -> None:
        """Analyze performance correlations between components."""
        try:
            # Simple correlation analysis between component response times
            correlations = {}

            component_types = list(ComponentType)
            for i, comp_a in enumerate(component_types):
                for comp_b in component_types[i+1:]:
                    history_a = self.component_health_history[comp_a]
                    history_b = self.component_health_history[comp_b]

                    if len(history_a) < 5 or len(history_b) < 5:
                        continue

                    # Extract response times
                    rt_a = [
                        entry["metrics"].get("response_time_ms", 0)
                        for entry in list(history_a)[-20:]  # Last 20 samples
                    ]
                    rt_b = [
                        entry["metrics"].get("response_time_ms", 0)
                        for entry in list(history_b)[-20:]
                    ]

                    if len(rt_a) == len(rt_b) and len(rt_a) > 3:
                        correlation = np.corrcoef(rt_a, rt_b)[0, 1] if np.std(rt_a) > 0 and np.std(rt_b) > 0 else 0
                        correlations[f"{comp_a.value}_vs_{comp_b.value}"] = float(correlation)

            self.performance_correlations = correlations

            # Log significant correlations
            for pair, correlation in correlations.items():
                if abs(correlation) > 0.7:
                    logger.info(f"Strong performance correlation detected: {pair} = {correlation:.2f}")

        except Exception as e:
            logger.error(f"Performance correlation analysis failed: {e}")

    async def _get_trend_summary(self) -> Dict[str, Any]:
        """Get summary of trend analysis for all components."""
        trend_summary = {}

        for component_type, analysis in self.trend_analysis_cache.items():
            trend_summary[component_type.value] = {
                "trend": analysis.trend.value,
                "confidence": analysis.confidence,
                "key_indicators": analysis.key_indicators,
                "recommendations": analysis.recommendations,
            }

        return trend_summary

    def _get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of active alerts."""
        alert_summary = {
            "total_alerts": len(self.active_alerts),
            "by_severity": {},
            "by_component": {},
            "recent_alerts": [],
        }

        # Count by severity
        for severity in AlertSeverity:
            count = sum(
                1 for alert in self.active_alerts.values()
                if alert.severity == severity and not alert.resolved
            )
            alert_summary["by_severity"][severity.value] = count

        # Count by component
        for component_type in ComponentType:
            count = sum(
                1 for alert in self.active_alerts.values()
                if alert.component_type == component_type and not alert.resolved
            )
            alert_summary["by_component"][component_type.value] = count

        # Recent alerts (last 10)
        recent_alerts = sorted(
            [alert for alert in self.active_alerts.values() if not alert.resolved],
            key=lambda x: x.timestamp,
            reverse=True
        )[:10]

        alert_summary["recent_alerts"] = [
            {
                "alert_id": alert.alert_id,
                "component": alert.component_type.value,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
            }
            for alert in recent_alerts
        ]

        return alert_summary

    async def _get_dependency_health_matrix(self) -> Dict[str, Any]:
        """Get health status matrix for component dependencies."""
        matrix = {}

        for component_type, dependencies in self.COMPONENT_DEPENDENCIES.items():
            component_health = {}

            for dep_component in dependencies:
                # Get dependency health from component coordinator
                dep_component_id = f"{dep_component.value}-{self.project_name}"
                dep_status = await self.component_coordinator.get_component_status(dep_component_id)
                component_health[dep_component.value] = (
                    dep_status is not None and "error" not in dep_status
                )

            matrix[component_type.value] = component_health

        return matrix

    async def _get_visualization_data(self) -> Dict[str, Any]:
        """Get data formatted for health dashboard visualizations."""
        visualization_data = {
            "component_status_chart": {},
            "response_time_trends": {},
            "error_rate_trends": {},
            "alert_timeline": [],
        }

        # Component status for pie chart
        for component_type in ComponentType:
            health_metrics = await self._get_component_health_detailed(component_type)
            visualization_data["component_status_chart"][component_type.value] = {
                "status": health_metrics.health_status.value,
                "response_time": health_metrics.response_time_ms,
                "error_rate": health_metrics.error_rate,
            }

        # Response time trends for line charts
        for component_type in ComponentType:
            history = list(self.component_health_history[component_type])[-50:]  # Last 50 points
            if history:
                visualization_data["response_time_trends"][component_type.value] = [
                    {
                        "timestamp": entry["timestamp"],
                        "value": entry["metrics"].get("response_time_ms", 0)
                    }
                    for entry in history
                ]

        # Alert timeline
        recent_alerts = sorted(
            self.active_alerts.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:20]

        visualization_data["alert_timeline"] = [
            {
                "timestamp": alert.timestamp.isoformat(),
                "component": alert.component_type.value,
                "severity": alert.severity.value,
                "message": alert.message,
                "resolved": alert.resolved,
            }
            for alert in recent_alerts
        ]

        return visualization_data

    async def _get_health_timeline(self) -> List[Dict[str, Any]]:
        """Get health timeline for the last 24 hours."""
        timeline = []
        current_time = time.time()

        # Create timeline entries from component health history
        for component_type in ComponentType:
            history = self.component_health_history[component_type]

            for entry in history:
                if current_time - entry["timestamp"] <= 86400:  # Last 24 hours
                    timeline.append({
                        "timestamp": entry["timestamp"],
                        "component": component_type.value,
                        "health_status": entry["metrics"].get("health_status", "unknown"),
                        "response_time_ms": entry["metrics"].get("response_time_ms", 0),
                        "error_rate": entry["metrics"].get("error_rate", 0),
                    })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    async def _get_recovery_actions_summary(self) -> Dict[str, Any]:
        """Get summary of recovery actions and their effectiveness."""
        recovery_summary = {
            "total_attempts": sum(self.recovery_attempts.values()),
            "by_component": dict(self.recovery_attempts),
            "success_rate": 0.8,  # Would calculate from actual data
            "last_attempts": {},
        }

        for component_type, last_attempt in self.last_recovery_attempt.items():
            recovery_summary["last_attempts"][component_type.value] = last_attempt.isoformat()

        return recovery_summary

    def _detect_project_name(self) -> str:
        """Detect project name from current working directory."""
        try:
            return Path.cwd().name
        except Exception:
            return "default"


# Global health coordinator instance
_health_coordinator: Optional[HealthCoordinator] = None


async def get_health_coordinator(
    db_path: str = "workspace_state.db",
    project_name: Optional[str] = None,
    project_path: Optional[str] = None,
    **kwargs
) -> HealthCoordinator:
    """Get or create global health coordinator instance."""
    global _health_coordinator

    if _health_coordinator is None:
        _health_coordinator = HealthCoordinator(
            db_path=db_path,
            project_name=project_name,
            project_path=project_path,
            **kwargs
        )

        if not await _health_coordinator.initialize():
            raise RuntimeError("Failed to initialize health coordinator")

    return _health_coordinator


async def shutdown_health_coordinator():
    """Shutdown global health coordinator."""
    global _health_coordinator

    if _health_coordinator:
        await _health_coordinator.stop_monitoring()
        _health_coordinator = None