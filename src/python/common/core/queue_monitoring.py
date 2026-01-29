"""
Unified Queue Monitoring System

Provides comprehensive monitoring integration combining queue statistics, performance
metrics, error tracking, health assessment, alerting, and dashboard capabilities into
a single unified interface.

Features:
    - Unified initialization and lifecycle management
    - Combined health assessment (queue + error metrics)
    - Integrated system status reporting
    - Coordinated monitoring across all subsystems
    - Background monitoring task management
    - Context manager support for clean resource handling

Components:
    - Queue statistics collection and analysis
    - Performance metrics and bottleneck detection
    - Backpressure detection and mitigation
    - Health calculation with composite scoring
    - Error message management and metrics
    - Alerting system with multiple channels
    - Dashboard data provisioning

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_monitoring import QueueMonitoringSystem

    # Initialize with default configuration
    monitoring = QueueMonitoringSystem()
    await monitoring.initialize()

    # Get comprehensive system status
    status = await monitoring.get_system_status()
    print(f"Queue size: {status['queue_stats']['queue_size']}")
    print(f"Error count: {status['error_stats']['total_errors']}")

    # Get unified health assessment
    health = await monitoring.get_comprehensive_health()
    print(f"Overall health: {health['overall_status']}")
    print(f"Health score: {health['score']}/100")

    # Start background monitoring
    await monitoring.start_background_monitoring()

    # Get monitoring summary
    summary = await monitoring.get_monitoring_summary()

    # Stop and cleanup
    await monitoring.stop_background_monitoring()
    await monitoring.close()

    # Or use as async context manager
    async with QueueMonitoringSystem() as monitoring:
        status = await monitoring.get_system_status()
    ```
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from .error_message_manager import ErrorMessageManager
from .error_monitoring import (
    ErrorMetricsCollector,
    LoggingHook,
)
from .error_monitoring import (
    HealthCheckManager as ErrorHealthCheckManager,
)
from .queue_alerting import QueueAlertingSystem
from .queue_backpressure import BackpressureDetector
from .queue_dashboard_data import QueueDashboardDataProvider
from .queue_health import QueueHealthCalculator
from .queue_performance_metrics import QueuePerformanceCollector
from .queue_statistics import QueueStatisticsCollector


@dataclass
class SystemStatus:
    """
    Comprehensive system status combining queue and error metrics.

    Attributes:
        queue_stats: Queue statistics snapshot
        error_stats: Error statistics snapshot
        health_summary: Combined health assessment
        timestamp: When status was captured
    """

    queue_stats: dict[str, Any]
    error_stats: dict[str, Any]
    health_summary: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "queue_stats": self.queue_stats,
            "error_stats": self.error_stats,
            "health_summary": self.health_summary,
            "timestamp": self.timestamp.isoformat()
        }


class QueueMonitoringSystem:
    """
    Unified monitoring system integrating queue and error monitoring.

    Provides single entry point for all monitoring operations including statistics
    collection, health assessment, error tracking, alerting, and dashboard data.
    Manages lifecycle of all monitoring components with proper initialization order.
    """

    def __init__(
        self,
        db_path: str | None = None,
        enable_alerting: bool = False,
        enable_dashboard: bool = False,
        enable_error_monitoring: bool = True,
        enable_performance_tracking: bool = True,
        enable_backpressure_detection: bool = True
    ):
        """
        Initialize queue monitoring system.

        Args:
            db_path: Optional custom database path (defaults to ~/.workspace-qdrant/state.db)
            enable_alerting: Whether to initialize alerting system
            enable_dashboard: Whether to initialize dashboard data provider
            enable_error_monitoring: Whether to track error messages
            enable_performance_tracking: Whether to track performance metrics
            enable_backpressure_detection: Whether to detect backpressure
        """
        if db_path is None:
            db_path = str(Path.home() / ".workspace-qdrant" / "state.db")

        self.db_path = db_path
        self.enable_alerting = enable_alerting
        self.enable_dashboard = enable_dashboard
        self.enable_error_monitoring = enable_error_monitoring
        self.enable_performance_tracking = enable_performance_tracking
        self.enable_backpressure_detection = enable_backpressure_detection

        # Core monitoring components
        self.error_manager: ErrorMessageManager | None = None
        self.stats_collector: QueueStatisticsCollector | None = None
        self.performance_collector: QueuePerformanceCollector | None = None
        self.backpressure_detector: BackpressureDetector | None = None
        self.health_calculator: QueueHealthCalculator | None = None

        # Error monitoring components
        self.error_metrics_collector: ErrorMetricsCollector | None = None
        self.error_health_checker: ErrorHealthCheckManager | None = None

        # Optional high-level components
        self.alerting_system: QueueAlertingSystem | None = None
        self.dashboard_provider: QueueDashboardDataProvider | None = None

        # Lifecycle management
        self._initialized = False
        self._background_monitoring_active = False
        self._lock = asyncio.Lock()

    async def initialize(self):
        """
        Initialize all monitoring components in dependency order.

        Initialization order ensures components have their dependencies ready:
        1. Error manager (no dependencies)
        2. Stats collector (uses error manager)
        3. Performance collector (independent)
        4. Backpressure detector (uses stats collector)
        5. Health calculator (uses stats, performance, backpressure)
        6. Error metrics & health (uses error manager)
        7. Alerting system (uses all collectors)
        8. Dashboard provider (uses all collectors)

        Raises:
            RuntimeError: If already initialized
        """
        if self._initialized:
            logger.warning("Queue monitoring system already initialized")
            return

        logger.info("Initializing queue monitoring system...")

        # 1. Initialize error manager
        if self.enable_error_monitoring:
            logger.debug("Initializing error manager...")
            self.error_manager = ErrorMessageManager(db_path=self.db_path)
            await self.error_manager.initialize()

        # 2. Initialize queue statistics collector
        logger.debug("Initializing queue statistics collector...")
        self.stats_collector = QueueStatisticsCollector(db_path=self.db_path)
        await self.stats_collector.initialize()

        # 3. Initialize performance collector
        if self.enable_performance_tracking:
            logger.debug("Initializing performance collector...")
            self.performance_collector = QueuePerformanceCollector(
                db_path=self.db_path,
                enable_resource_tracking=True
            )
            await self.performance_collector.initialize()

        # 4. Initialize backpressure detector
        if self.enable_backpressure_detection:
            logger.debug("Initializing backpressure detector...")
            self.backpressure_detector = BackpressureDetector(
                stats_collector=self.stats_collector
            )
            await self.backpressure_detector.initialize()

        # 5. Initialize queue health calculator
        logger.debug("Initializing queue health calculator...")
        self.health_calculator = QueueHealthCalculator(
            stats_collector=self.stats_collector,
            backpressure_detector=self.backpressure_detector,
            performance_collector=self.performance_collector
        )
        await self.health_calculator.initialize()

        # 6. Initialize error monitoring components
        if self.enable_error_monitoring and self.error_manager:
            logger.debug("Initializing error metrics collector...")
            self.error_metrics_collector = ErrorMetricsCollector()
            # Register default logging hook
            self.error_metrics_collector.register_hook(LoggingHook())

            logger.debug("Initializing error health checker...")
            self.error_health_checker = ErrorHealthCheckManager(self.error_manager)
            await self.error_health_checker.initialize()

        # 7. Initialize alerting system
        if self.enable_alerting:
            logger.debug("Initializing alerting system...")
            self.alerting_system = QueueAlertingSystem(
                db_path=self.db_path,
                stats_collector=self.stats_collector,
                performance_collector=self.performance_collector,
                health_calculator=self.health_calculator,
                backpressure_detector=self.backpressure_detector
            )
            await self.alerting_system.initialize()

        # 8. Initialize dashboard provider
        if self.enable_dashboard:
            logger.debug("Initializing dashboard provider...")
            self.dashboard_provider = QueueDashboardDataProvider(
                db_path=self.db_path,
                enable_trend_analysis=True,
                enable_bottleneck_detection=True
            )
            await self.dashboard_provider.initialize()

        self._initialized = True
        logger.info("Queue monitoring system initialized successfully")

    async def close(self):
        """
        Close all monitoring components and cleanup resources.

        Closes components in reverse order of initialization to ensure
        proper dependency cleanup.
        """
        if not self._initialized:
            return

        logger.info("Closing queue monitoring system...")

        # Stop background monitoring if active
        if self._background_monitoring_active:
            await self.stop_background_monitoring()

        # Close in reverse order of initialization
        if self.dashboard_provider:
            await self.dashboard_provider.close()

        if self.alerting_system:
            await self.alerting_system.close()

        if self.error_health_checker:
            await self.error_health_checker.close()

        if self.error_metrics_collector:
            await self.error_metrics_collector.close()

        if self.health_calculator:
            await self.health_calculator.close()

        if self.backpressure_detector:
            await self.backpressure_detector.close()

        if self.performance_collector:
            await self.performance_collector.close()

        if self.stats_collector:
            await self.stats_collector.close()

        if self.error_manager:
            await self.error_manager.close()

        self._initialized = False
        logger.info("Queue monitoring system closed successfully")

    async def get_system_status(
        self,
        queue_type: str = "ingestion_queue"
    ) -> SystemStatus:
        """
        Get comprehensive system status combining queue and error metrics.

        Args:
            queue_type: Queue type to get statistics for

        Returns:
            SystemStatus with queue stats, error stats, and health summary

        Raises:
            RuntimeError: If system not initialized
        """
        self._ensure_initialized()

        # Get queue statistics
        queue_stats = await self.stats_collector.get_current_statistics(queue_type=queue_type)

        # Get error statistics
        error_stats = {}
        if self.error_manager:
            raw_error_stats = await self.error_manager.get_error_stats()
            error_stats = {
                "total_errors": raw_error_stats.get("total_count", 0),
                "by_severity": raw_error_stats.get("by_severity", {}),
                "by_category": raw_error_stats.get("by_category", {}),
                "unacknowledged_count": raw_error_stats.get("unacknowledged_count", 0)
            }

        # Get health summary
        health_summary = await self._get_health_summary(queue_type=queue_type)

        return SystemStatus(
            queue_stats=queue_stats.to_dict(),
            error_stats=error_stats,
            health_summary=health_summary
        )

    async def get_comprehensive_health(
        self,
        queue_type: str = "ingestion_queue"
    ) -> dict[str, Any]:
        """
        Get comprehensive health assessment combining queue and error health.

        Merges health indicators from both queue health calculator and error
        health checker to provide unified health view.

        Args:
            queue_type: Queue type to assess

        Returns:
            Dictionary with combined health assessment

        Raises:
            RuntimeError: If system not initialized
        """
        self._ensure_initialized()

        # Get queue health
        queue_health = await self.health_calculator.calculate_health(queue_type=queue_type)

        # Get error health
        error_health = None
        if self.error_health_checker:
            error_health = await self.error_health_checker.get_health_status()

        # Combine health assessments
        combined = queue_health.to_dict()

        if error_health:
            # Add error health indicators
            combined["error_health"] = error_health.to_dict()

            # Adjust overall status if error health is worse
            if error_health.status.value == "unhealthy":
                combined["overall_status"] = "unhealthy"
            elif error_health.status.value == "degraded" and combined["overall_status"] == "healthy":
                combined["overall_status"] = "degraded"

            # Adjust score based on error health
            # Weight: 70% queue health, 30% error health
            queue_weight = 0.7
            error_weight = 0.3

            # Convert error health status to score
            error_score_map = {
                "healthy": 100.0,
                "degraded": 70.0,
                "unhealthy": 40.0
            }
            error_score = error_score_map.get(error_health.status.value, 50.0)

            combined["score"] = (
                combined["score"] * queue_weight +
                error_score * error_weight
            )

        return combined

    async def get_monitoring_summary(self) -> dict[str, Any]:
        """
        Get high-level monitoring summary.

        Provides quick overview of system state suitable for health checks
        and status endpoints.

        Returns:
            Dictionary with monitoring summary

        Raises:
            RuntimeError: If system not initialized
        """
        self._ensure_initialized()

        stats = await self.stats_collector.get_current_statistics()
        health = await self.get_comprehensive_health()

        summary = {
            "status": health["overall_status"],
            "health_score": round(health["score"], 1),
            "queue_size": stats.queue_size,
            "processing_rate": round(stats.processing_rate, 2),
            "error_rate": round(stats.failure_rate, 2),
            "success_rate": round(stats.success_rate, 2),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Add error count if error monitoring enabled
        if self.error_manager:
            error_stats = await self.error_manager.get_error_stats()
            summary["total_errors"] = error_stats.get("total_count", 0)
            summary["unacknowledged_errors"] = error_stats.get("unacknowledged_count", 0)

        return summary

    async def start_background_monitoring(
        self,
        stats_interval: int = 5,
        performance_interval: int = 10
    ) -> bool:
        """
        Start background monitoring tasks.

        Starts collection tasks for statistics and performance metrics.

        Args:
            stats_interval: Statistics collection interval in seconds
            performance_interval: Performance metrics collection interval in seconds

        Returns:
            True if started successfully, False if already running

        Raises:
            RuntimeError: If system not initialized
        """
        self._ensure_initialized()

        if self._background_monitoring_active:
            logger.warning("Background monitoring already active")
            return False

        # Start statistics collection
        await self.stats_collector.start_collection(interval_seconds=stats_interval)

        # Start performance collection if enabled
        if self.performance_collector:
            await self.performance_collector.start_collection(interval_seconds=performance_interval)

        self._background_monitoring_active = True
        logger.info("Background monitoring started")
        return True

    async def stop_background_monitoring(self) -> bool:
        """
        Stop background monitoring tasks.

        Returns:
            True if stopped successfully, False if not running
        """
        if not self._background_monitoring_active:
            logger.warning("Background monitoring not active")
            return False

        # Stop statistics collection
        if self.stats_collector:
            await self.stats_collector.stop_collection()

        # Stop performance collection
        if self.performance_collector:
            await self.performance_collector.stop_collection()

        self._background_monitoring_active = False
        logger.info("Background monitoring stopped")
        return True

    async def _get_health_summary(self, queue_type: str) -> dict[str, Any]:
        """Get health summary from both queue and error health."""
        summary = {}

        # Queue health
        queue_health = await self.health_calculator.calculate_health(queue_type=queue_type)
        summary["queue_health"] = {
            "status": queue_health.overall_status.value,
            "score": round(queue_health.score, 1),
            "indicators": len(queue_health.indicators)
        }

        # Error health
        if self.error_health_checker:
            error_health = await self.error_health_checker.get_health_status()
            summary["error_health"] = {
                "status": error_health.status.value,
                "checks": len(error_health.checks)
            }

        return summary

    def _ensure_initialized(self):
        """Ensure system is initialized."""
        if not self._initialized:
            raise RuntimeError(
                "Queue monitoring system not initialized. Call initialize() first."
            )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
