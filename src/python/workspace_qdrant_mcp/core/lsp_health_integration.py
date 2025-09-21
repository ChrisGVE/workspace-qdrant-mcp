"""
LSP Health Monitoring Integration

This module provides integration between the LSP health monitoring system and the
ecosystem-aware LSP detection and symbol extraction manager. It enhances the existing
health monitor with project context awareness and ecosystem-specific recovery strategies.

Key Features:
    - Ecosystem-aware health monitoring for project-specific LSP servers
    - Health-based LSP server selection and routing
    - Automatic recovery with project context consideration
    - Health metrics aggregation for ecosystem-specific LSPs
    - Health dashboard integration for debugging purposes
    - Performance correlation between health status and extraction success

Architecture:
    - EcosystemHealthMonitor: Enhanced health monitor with project awareness
    - HealthAwareExtraction: Health-guided extraction routing
    - HealthMetricsAggregator: Ecosystem-specific health metrics
    - RecoveryContextManager: Project-aware recovery strategies

Example:
    ```python
    from workspace_qdrant_mcp.core.lsp_health_integration import EcosystemHealthMonitor

    # Initialize with project context
    health_monitor = EcosystemHealthMonitor(
        project_path="/path/to/project",
        lsp_detector=detector
    )

    # Register LSP servers with ecosystem context
    await health_monitor.register_ecosystem_lsp(
        "rust-analyzer",
        client,
        ecosystems=["rust"]
    )

    # Monitor health with ecosystem awareness
    health_status = await health_monitor.get_ecosystem_health("rust")
    ```
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

from loguru import logger

# Import existing components
try:
    from python.common.core.lsp_health_monitor import (
        LspHealthMonitor, HealthStatus, RecoveryStrategy, NotificationLevel,
        HealthCheckConfig, ServerHealthInfo, UserNotification
    )
    from python.common.core.lsp_detector import LSPDetector, LSPServerInfo
    from python.common.core.lsp_client import AsyncioLspClient, ConnectionState
    from python.common.core.pattern_manager import PatternManager
except ImportError as e:
    logger.warning(f"Failed to import health monitoring components: {e}")
    # Fallback for development
    LspHealthMonitor = None
    HealthStatus = None
    LSPDetector = None
    PatternManager = None


@dataclass
class EcosystemHealthInfo:
    """Health information for an ecosystem and its associated LSP servers."""

    ecosystem_name: str
    associated_lsps: List[str] = field(default_factory=list)
    overall_health: HealthStatus = HealthStatus.UNKNOWN if HealthStatus else None
    health_score: float = 0.0  # 0.0-1.0 health score
    last_health_check: float = 0.0
    failure_count: int = 0
    recovery_attempts: int = 0
    recommended_fallback: Optional[str] = None


@dataclass
class HealthMetrics:
    """Aggregated health metrics for monitoring and debugging."""

    total_lsp_servers: int = 0
    healthy_servers: int = 0
    degraded_servers: int = 0
    unhealthy_servers: int = 0
    disconnected_servers: int = 0
    failed_servers: int = 0

    ecosystem_health_scores: Dict[str, float] = field(default_factory=dict)
    average_response_times: Dict[str, float] = field(default_factory=dict)
    error_rates: Dict[str, float] = field(default_factory=dict)
    recovery_success_rates: Dict[str, float] = field(default_factory=dict)

    uptime_percentage: float = 0.0
    total_notifications_sent: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0

    def get_overall_health_score(self) -> float:
        """Calculate overall health score across all ecosystems."""
        if not self.ecosystem_health_scores:
            return 0.0
        return sum(self.ecosystem_health_scores.values()) / len(self.ecosystem_health_scores)


class EcosystemHealthMonitor:
    """
    Enhanced LSP health monitor with ecosystem awareness and project context.

    Extends the existing LspHealthMonitor with:
    - Ecosystem-specific health tracking
    - Project-aware recovery strategies
    - Health-based LSP selection routing
    - Ecosystem health metrics aggregation
    """

    def __init__(
        self,
        project_path: Union[str, Path],
        lsp_detector: Optional[LSPDetector] = None,
        base_health_monitor: Optional[LspHealthMonitor] = None,
        config: Optional[HealthCheckConfig] = None
    ):
        """
        Initialize ecosystem-aware health monitor.

        Args:
            project_path: Path to the project root
            lsp_detector: LSP detector for ecosystem awareness
            base_health_monitor: Existing health monitor to extend
            config: Health check configuration
        """
        self.project_path = Path(project_path)
        self.lsp_detector = lsp_detector

        # Initialize base health monitor
        if base_health_monitor:
            self.base_monitor = base_health_monitor
        elif LspHealthMonitor:
            self.base_monitor = LspHealthMonitor(config=config)
        else:
            logger.error("LspHealthMonitor not available")
            raise RuntimeError("Health monitoring system required")

        # Ecosystem tracking
        self.ecosystem_health: Dict[str, EcosystemHealthInfo] = {}
        self.lsp_to_ecosystems: Dict[str, List[str]] = {}
        self.ecosystem_fallbacks: Dict[str, List[str]] = {}

        # Health metrics
        self.metrics = HealthMetrics()
        self.health_history: List[Tuple[float, HealthMetrics]] = []
        self.max_history_entries = 1000

        # Event handlers
        self.health_change_callbacks: List[Callable[[str, HealthStatus, Optional[Exception]], None]] = []
        self.ecosystem_health_callbacks: List[Callable[[str, EcosystemHealthInfo], None]] = []

        # State
        self.initialized = False
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info("Ecosystem health monitor initialized",
                   project_path=str(self.project_path))

    async def initialize(self) -> None:
        """Initialize the ecosystem health monitor."""
        if self.initialized:
            return

        # Detect project ecosystems
        if self.lsp_detector:
            detection_result = self.lsp_detector.get_ecosystem_aware_lsps(self.project_path)

            # Initialize ecosystem health tracking
            for ecosystem in detection_result.detected_ecosystems:
                self.ecosystem_health[ecosystem] = EcosystemHealthInfo(
                    ecosystem_name=ecosystem,
                    overall_health=HealthStatus.UNKNOWN if HealthStatus else None
                )

            # Map LSPs to ecosystems
            for ecosystem, lsp_names in detection_result.ecosystem_lsp_recommendations.items():
                for lsp_name in lsp_names:
                    if lsp_name not in self.lsp_to_ecosystems:
                        self.lsp_to_ecosystems[lsp_name] = []
                    self.lsp_to_ecosystems[lsp_name].append(ecosystem)

                    # Track LSPs for this ecosystem
                    self.ecosystem_health[ecosystem].associated_lsps.append(lsp_name)

        # Setup ecosystem fallback strategies
        self._setup_ecosystem_fallbacks()

        # Start ecosystem health monitoring
        self.monitoring_task = asyncio.create_task(self._ecosystem_health_monitor_loop())

        self.initialized = True
        logger.info("Ecosystem health monitor ready",
                   ecosystems=len(self.ecosystem_health),
                   lsp_mappings=len(self.lsp_to_ecosystems))

    async def register_ecosystem_lsp(
        self,
        lsp_name: str,
        client: AsyncioLspClient,
        ecosystems: List[str],
        auto_start_monitoring: bool = True
    ) -> None:
        """
        Register an LSP server with ecosystem context.

        Args:
            lsp_name: Name of the LSP server
            client: LSP client instance
            ecosystems: List of ecosystems this LSP serves
            auto_start_monitoring: Whether to start monitoring immediately
        """
        # Register with base health monitor
        self.base_monitor.register_server(lsp_name, client, auto_start_monitoring)

        # Track ecosystem associations
        self.lsp_to_ecosystems[lsp_name] = ecosystems

        # Update ecosystem health info
        for ecosystem in ecosystems:
            if ecosystem not in self.ecosystem_health:
                self.ecosystem_health[ecosystem] = EcosystemHealthInfo(ecosystem_name=ecosystem)

            if lsp_name not in self.ecosystem_health[ecosystem].associated_lsps:
                self.ecosystem_health[ecosystem].associated_lsps.append(lsp_name)

        # Register for health change notifications
        self.base_monitor.add_notification_handler(
            lambda notif: self._handle_lsp_health_notification(lsp_name, notif)
        )

        logger.info(f"Registered LSP {lsp_name} for ecosystems: {ecosystems}")

    async def get_ecosystem_health(self, ecosystem_name: str) -> Optional[EcosystemHealthInfo]:
        """Get health information for a specific ecosystem."""
        if ecosystem_name not in self.ecosystem_health:
            return None

        ecosystem_info = self.ecosystem_health[ecosystem_name]

        # Update health score based on associated LSPs
        await self._update_ecosystem_health(ecosystem_name)

        return ecosystem_info

    async def get_healthy_lsps_for_ecosystem(self, ecosystem_name: str) -> List[str]:
        """Get list of healthy LSP servers for an ecosystem."""
        if ecosystem_name not in self.ecosystem_health:
            return []

        ecosystem_info = self.ecosystem_health[ecosystem_name]
        healthy_lsps = []

        for lsp_name in ecosystem_info.associated_lsps:
            server_health = self.base_monitor.get_server_health(lsp_name)
            if server_health and server_health.health_status == HealthStatus.HEALTHY:
                healthy_lsps.append(lsp_name)

        return healthy_lsps

    async def select_best_lsp_for_ecosystem(
        self,
        ecosystem_name: str,
        file_extension: Optional[str] = None
    ) -> Optional[str]:
        """
        Select the best available LSP server for an ecosystem.

        Args:
            ecosystem_name: Target ecosystem
            file_extension: Optional file extension for additional filtering

        Returns:
            Name of the best LSP server, or None if none available
        """
        healthy_lsps = await self.get_healthy_lsps_for_ecosystem(ecosystem_name)

        if not healthy_lsps:
            logger.warning(f"No healthy LSPs available for ecosystem: {ecosystem_name}")
            return None

        # If we have a file extension, filter by support
        if file_extension and self.lsp_detector:
            supported_lsps = []
            for lsp_name in healthy_lsps:
                lsp_info = self.lsp_detector.get_lsp_for_extension(file_extension)
                if lsp_info and lsp_info.name == lsp_name:
                    supported_lsps.append(lsp_name)

            if supported_lsps:
                healthy_lsps = supported_lsps

        # Select LSP with best health metrics
        best_lsp = None
        best_score = -1.0

        for lsp_name in healthy_lsps:
            server_health = self.base_monitor.get_server_health(lsp_name)
            if server_health:
                # Calculate health score based on multiple factors
                score = self._calculate_lsp_health_score(server_health)
                if score > best_score:
                    best_score = score
                    best_lsp = lsp_name

        logger.debug(f"Selected LSP {best_lsp} for ecosystem {ecosystem_name} (score: {best_score:.2f})")
        return best_lsp

    def add_health_change_callback(
        self,
        callback: Callable[[str, HealthStatus, Optional[Exception]], None]
    ) -> None:
        """Add a callback for LSP health status changes."""
        self.health_change_callbacks.append(callback)

    def add_ecosystem_health_callback(
        self,
        callback: Callable[[str, EcosystemHealthInfo], None]
    ) -> None:
        """Add a callback for ecosystem health updates."""
        self.ecosystem_health_callbacks.append(callback)

    async def get_health_metrics(self) -> HealthMetrics:
        """Get comprehensive health metrics."""
        await self._update_metrics()
        return self.metrics

    async def get_health_dashboard_data(self) -> Dict[str, Any]:
        """Get health data for dashboard/debugging purposes."""
        metrics = await self.get_health_metrics()

        dashboard_data = {
            "timestamp": time.time(),
            "project_path": str(self.project_path),
            "overall_health_score": metrics.get_overall_health_score(),
            "server_counts": {
                "total": metrics.total_lsp_servers,
                "healthy": metrics.healthy_servers,
                "degraded": metrics.degraded_servers,
                "unhealthy": metrics.unhealthy_servers,
                "disconnected": metrics.disconnected_servers,
                "failed": metrics.failed_servers
            },
            "ecosystem_health": {
                name: {
                    "health_score": info.health_score,
                    "overall_health": info.overall_health.value if info.overall_health else "unknown",
                    "associated_lsps": info.associated_lsps,
                    "failure_count": info.failure_count,
                    "recovery_attempts": info.recovery_attempts
                }
                for name, info in self.ecosystem_health.items()
            },
            "performance_metrics": {
                "average_response_times": metrics.average_response_times,
                "error_rates": metrics.error_rates,
                "recovery_success_rates": metrics.recovery_success_rates,
                "uptime_percentage": metrics.uptime_percentage
            },
            "recent_history": self.health_history[-10:]  # Last 10 entries
        }

        return dashboard_data

    def _setup_ecosystem_fallbacks(self) -> None:
        """Setup fallback strategies for ecosystems."""
        # Define fallback chains for common ecosystems
        fallback_chains = {
            "python": ["ruff", "pyright", "pylsp"],
            "rust": ["rust-analyzer"],
            "typescript": ["typescript-language-server"],
            "javascript": ["typescript-language-server"],
            "go": ["gopls"],
            "java": ["java-language-server"],
            "cpp": ["clangd"],
            "c": ["clangd"]
        }

        for ecosystem, fallback_lsps in fallback_chains.items():
            if ecosystem in self.ecosystem_health:
                self.ecosystem_fallbacks[ecosystem] = fallback_lsps

        logger.debug("Ecosystem fallback strategies configured")

    async def _ecosystem_health_monitor_loop(self) -> None:
        """Main ecosystem health monitoring loop."""
        while True:
            try:
                # Update ecosystem health for all ecosystems
                for ecosystem_name in self.ecosystem_health.keys():
                    await self._update_ecosystem_health(ecosystem_name)

                # Update overall metrics
                await self._update_metrics()

                # Store health snapshot
                self._store_health_snapshot()

                # Sleep until next check
                await asyncio.sleep(30.0)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in ecosystem health monitoring loop: {e}")
                await asyncio.sleep(5.0)  # Brief pause before retrying

    async def _update_ecosystem_health(self, ecosystem_name: str) -> None:
        """Update health information for a specific ecosystem."""
        if ecosystem_name not in self.ecosystem_health:
            return

        ecosystem_info = self.ecosystem_health[ecosystem_name]
        associated_lsps = ecosystem_info.associated_lsps

        if not associated_lsps:
            ecosystem_info.overall_health = HealthStatus.UNKNOWN if HealthStatus else None
            ecosystem_info.health_score = 0.0
            return

        # Calculate health based on associated LSPs
        health_scores = []
        health_statuses = []

        for lsp_name in associated_lsps:
            server_health = self.base_monitor.get_server_health(lsp_name)
            if server_health:
                health_scores.append(self._calculate_lsp_health_score(server_health))
                health_statuses.append(server_health.health_status)

        if health_scores:
            # Average health score
            ecosystem_info.health_score = sum(health_scores) / len(health_scores)

            # Determine overall health status
            if all(status == HealthStatus.HEALTHY for status in health_statuses):
                ecosystem_info.overall_health = HealthStatus.HEALTHY
            elif any(status == HealthStatus.FAILED for status in health_statuses):
                ecosystem_info.overall_health = HealthStatus.FAILED
            elif any(status == HealthStatus.UNHEALTHY for status in health_statuses):
                ecosystem_info.overall_health = HealthStatus.UNHEALTHY
            elif any(status == HealthStatus.DEGRADED for status in health_statuses):
                ecosystem_info.overall_health = HealthStatus.DEGRADED
            else:
                ecosystem_info.overall_health = HealthStatus.DISCONNECTED

        ecosystem_info.last_health_check = time.time()

        # Notify ecosystem health callbacks
        for callback in self.ecosystem_health_callbacks:
            try:
                callback(ecosystem_name, ecosystem_info)
            except Exception as e:
                logger.error(f"Error in ecosystem health callback: {e}")

    def _calculate_lsp_health_score(self, server_health: ServerHealthInfo) -> float:
        """Calculate a health score for an LSP server (0.0-1.0)."""
        base_score = 0.0

        # Health status contributes 60% of score
        status_scores = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.7,
            HealthStatus.UNHEALTHY: 0.3,
            HealthStatus.DISCONNECTED: 0.1,
            HealthStatus.FAILED: 0.0,
            HealthStatus.UNKNOWN: 0.0
        }

        if HealthStatus:
            base_score += status_scores.get(server_health.health_status, 0.0) * 0.6

        # Response time contributes 20% (faster is better)
        if hasattr(server_health, 'average_response_time') and server_health.average_response_time > 0:
            # Good response time: < 1s, Poor: > 10s
            response_score = max(0.0, min(1.0, (10.0 - server_health.average_response_time) / 9.0))
            base_score += response_score * 0.2

        # Consecutive successes contribute 20%
        if hasattr(server_health, 'consecutive_successes'):
            success_score = min(1.0, server_health.consecutive_successes / 5.0)
            base_score += success_score * 0.2

        return min(1.0, base_score)

    async def _update_metrics(self) -> None:
        """Update overall health metrics."""
        # Get all server healths from base monitor
        all_servers = self.base_monitor.get_all_server_health()

        # Reset counters
        self.metrics.total_lsp_servers = len(all_servers)
        self.metrics.healthy_servers = 0
        self.metrics.degraded_servers = 0
        self.metrics.unhealthy_servers = 0
        self.metrics.disconnected_servers = 0
        self.metrics.failed_servers = 0

        # Count by status
        for server_health in all_servers.values():
            if server_health.health_status == HealthStatus.HEALTHY:
                self.metrics.healthy_servers += 1
            elif server_health.health_status == HealthStatus.DEGRADED:
                self.metrics.degraded_servers += 1
            elif server_health.health_status == HealthStatus.UNHEALTHY:
                self.metrics.unhealthy_servers += 1
            elif server_health.health_status == HealthStatus.DISCONNECTED:
                self.metrics.disconnected_servers += 1
            elif server_health.health_status == HealthStatus.FAILED:
                self.metrics.failed_servers += 1

        # Update ecosystem health scores
        self.metrics.ecosystem_health_scores = {
            name: info.health_score
            for name, info in self.ecosystem_health.items()
        }

        # Calculate uptime percentage
        if self.metrics.total_lsp_servers > 0:
            healthy_and_degraded = self.metrics.healthy_servers + self.metrics.degraded_servers
            self.metrics.uptime_percentage = healthy_and_degraded / self.metrics.total_lsp_servers
        else:
            self.metrics.uptime_percentage = 0.0

    def _store_health_snapshot(self) -> None:
        """Store a snapshot of current health metrics."""
        snapshot = (time.time(), self.metrics)
        self.health_history.append(snapshot)

        # Trim history if too long
        if len(self.health_history) > self.max_history_entries:
            self.health_history = self.health_history[-self.max_history_entries:]

    def _handle_lsp_health_notification(self, lsp_name: str, notification: UserNotification) -> None:
        """Handle health notifications from the base monitor."""
        # Update ecosystem health tracking
        ecosystems = self.lsp_to_ecosystems.get(lsp_name, [])

        for ecosystem in ecosystems:
            if ecosystem in self.ecosystem_health:
                ecosystem_info = self.ecosystem_health[ecosystem]

                if notification.level in [NotificationLevel.ERROR, NotificationLevel.CRITICAL]:
                    ecosystem_info.failure_count += 1

                if notification.auto_recovery_attempted:
                    ecosystem_info.recovery_attempts += 1

        # Notify health change callbacks
        health_status = self._notification_level_to_health_status(notification.level)
        error = Exception(notification.message) if notification.level in [NotificationLevel.ERROR, NotificationLevel.CRITICAL] else None

        for callback in self.health_change_callbacks:
            try:
                callback(lsp_name, health_status, error)
            except Exception as e:
                logger.error(f"Error in health change callback: {e}")

    def _notification_level_to_health_status(self, level: NotificationLevel) -> HealthStatus:
        """Convert notification level to health status."""
        if not HealthStatus:
            return None

        mapping = {
            NotificationLevel.INFO: HealthStatus.HEALTHY,
            NotificationLevel.WARNING: HealthStatus.DEGRADED,
            NotificationLevel.ERROR: HealthStatus.UNHEALTHY,
            NotificationLevel.CRITICAL: HealthStatus.FAILED
        }
        return mapping.get(level, HealthStatus.UNKNOWN)

    async def shutdown(self) -> None:
        """Shutdown the ecosystem health monitor."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        # Shutdown base monitor if we own it
        if hasattr(self.base_monitor, 'shutdown'):
            await self.base_monitor.shutdown()

        logger.info("Ecosystem health monitor shutdown complete")


# Convenience functions for common operations
async def create_ecosystem_health_monitor(
    project_path: Union[str, Path],
    lsp_detector: Optional[LSPDetector] = None
) -> EcosystemHealthMonitor:
    """Create and initialize an ecosystem health monitor."""
    monitor = EcosystemHealthMonitor(project_path, lsp_detector)
    await monitor.initialize()
    return monitor


async def get_project_health_status(
    project_path: Union[str, Path],
    lsp_detector: Optional[LSPDetector] = None
) -> Dict[str, Any]:
    """Get health status for a project's LSP ecosystem."""
    monitor = await create_ecosystem_health_monitor(project_path, lsp_detector)
    try:
        return await monitor.get_health_dashboard_data()
    finally:
        await monitor.shutdown()