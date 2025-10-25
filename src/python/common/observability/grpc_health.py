"""
gRPC Health Check Implementation for workspace-qdrant-mcp.

Provides comprehensive gRPC health checking services for all four components
following the standard gRPC health checking protocol with enhanced monitoring
and diagnostic capabilities.

Key Features:
    - Standard gRPC health checking protocol implementation
    - Component-specific health service integration
    - Health status streaming and watching
    - Enhanced diagnostic information in health responses
    - Integration with existing health monitoring infrastructure
    - Automatic service registration and discovery

gRPC Health Protocol:
    - Implements the grpc.health.v1.Health service
    - Supports per-service health checking
    - Provides streaming health updates
    - Compatible with Kubernetes health probes
    - Supports health check middleware integration

Example:
    ```python
    from workspace_qdrant_mcp.observability.grpc_health import GrpcHealthService

    # Create health service
    health_service = GrpcHealthService()
    await health_service.initialize()

    # Register service health
    await health_service.set_service_health("workspace-qdrant-mcp", True)

    # Add to gRPC server
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)
    ```
"""

import asyncio
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from loguru import logger

from .health import get_health_checker
from .health_coordinator import get_health_coordinator
from .metrics import metrics_instance


class GrpcHealthStatus(Enum):
    """gRPC health status enumeration matching the protobuf definition."""

    UNKNOWN = health_pb2.HealthCheckResponse.UNKNOWN
    SERVING = health_pb2.HealthCheckResponse.SERVING
    NOT_SERVING = health_pb2.HealthCheckResponse.NOT_SERVING
    SERVICE_UNKNOWN = health_pb2.HealthCheckResponse.SERVICE_UNKNOWN


@dataclass
class ServiceHealth:
    """Health status for a specific service."""

    service_name: str
    status: GrpcHealthStatus
    last_check: float
    details: dict[str, Any] = field(default_factory=dict)
    watchers: set[Any] = field(default_factory=set)


class GrpcHealthService(health_pb2_grpc.HealthServicer):
    """
    gRPC Health Service implementation.

    Provides comprehensive health checking for all workspace-qdrant-mcp components
    following the standard gRPC health checking protocol with enhanced monitoring.
    """

    # Standard gRPC health service names for workspace-qdrant-mcp components
    SERVICE_NAMES = {
        "workspace-qdrant-mcp": "Primary MCP service",
        "workspace-qdrant-rust-daemon": "Rust processing daemon",
        "workspace-qdrant-cli": "CLI utility service",
        "workspace-qdrant-context": "Context injection service",
    }

    def __init__(
        self,
        enable_detailed_diagnostics: bool = True,
        health_check_interval: float = 30.0,
        max_watchers_per_service: int = 100
    ):
        """
        Initialize gRPC health service.

        Args:
            enable_detailed_diagnostics: Include detailed diagnostic info in responses
            health_check_interval: Interval for background health checks
            max_watchers_per_service: Maximum watchers per service to prevent resource exhaustion
        """
        self.enable_detailed_diagnostics = enable_detailed_diagnostics
        self.health_check_interval = health_check_interval
        self.max_watchers_per_service = max_watchers_per_service

        # Service health tracking
        self.service_health: dict[str, ServiceHealth] = {}
        self.global_health_status = GrpcHealthStatus.UNKNOWN

        # Background tasks
        self.background_tasks: list[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        # Health monitoring integration
        self.health_checker = None
        self.health_coordinator = None

        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="grpc_health")

        logger.info(
            "gRPC Health Service initialized",
            detailed_diagnostics=enable_detailed_diagnostics,
            check_interval=health_check_interval
        )

    async def initialize(self) -> None:
        """Initialize the gRPC health service and start background monitoring."""
        try:
            # Initialize health monitoring components
            self.health_checker = get_health_checker()
            self.health_coordinator = await get_health_coordinator()

            # Initialize service health statuses
            for service_name in self.SERVICE_NAMES:
                self.service_health[service_name] = ServiceHealth(
                    service_name=service_name,
                    status=GrpcHealthStatus.UNKNOWN,
                    last_check=time.time()
                )

            # Start background health monitoring
            self.background_tasks.append(
                asyncio.create_task(self._background_health_monitor())
            )

            # Set initial global health
            await self._update_global_health()

            logger.info("gRPC Health Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize gRPC Health Service: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the gRPC health service and cleanup resources."""
        logger.info("Shutting down gRPC Health Service")

        # Signal shutdown to background tasks
        self.shutdown_event.set()

        # Cancel all background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Notify all watchers that services are not serving
        for service_health in self.service_health.values():
            for watcher in service_health.watchers.copy():
                try:
                    await watcher.put(health_pb2.HealthCheckResponse(
                        status=health_pb2.HealthCheckResponse.NOT_SERVING
                    ))
                except Exception as e:
                    logger.debug(f"Failed to notify watcher during shutdown: {e}")

        logger.info("gRPC Health Service shutdown complete")

    async def Check(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext
    ) -> health_pb2.HealthCheckResponse:
        """
        Perform a health check for a specific service.

        Args:
            request: Health check request containing service name
            context: gRPC service context

        Returns:
            Health check response with service status
        """
        service_name = request.service or ""
        start_time = time.perf_counter()

        try:
            logger.debug(f"Health check requested for service: '{service_name}'")

            # Handle empty service name (global health check)
            if not service_name:
                status = await self._get_global_health_status()
                response_time = (time.perf_counter() - start_time) * 1000

                # Record metrics
                metrics_instance.increment_counter(
                    "grpc_health_checks_total",
                    service="global",
                    status=status.name.lower()
                )
                metrics_instance.record_histogram(
                    "grpc_health_check_duration_seconds",
                    response_time / 1000,
                    service="global"
                )

                return health_pb2.HealthCheckResponse(status=status.value)

            # Check if service is known
            if service_name not in self.service_health:
                # Record unknown service metric
                metrics_instance.increment_counter(
                    "grpc_health_checks_total",
                    service=service_name,
                    status="service_unknown"
                )

                return health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
                )

            # Get service health
            service_health = self.service_health[service_name]
            status = service_health.status

            # Update last check time
            service_health.last_check = time.time()

            response_time = (time.perf_counter() - start_time) * 1000

            # Record metrics
            metrics_instance.increment_counter(
                "grpc_health_checks_total",
                service=service_name,
                status=status.name.lower()
            )
            metrics_instance.record_histogram(
                "grpc_health_check_duration_seconds",
                response_time / 1000,
                service=service_name
            )

            logger.debug(
                f"Health check completed for '{service_name}': {status.name}",
                response_time_ms=response_time
            )

            return health_pb2.HealthCheckResponse(status=status.value)

        except Exception as e:
            logger.error(f"Health check failed for service '{service_name}': {e}")

            # Record error metric
            metrics_instance.increment_counter(
                "grpc_health_checks_total",
                service=service_name,
                status="error"
            )

            # Return unknown status on error
            return health_pb2.HealthCheckResponse(
                status=health_pb2.HealthCheckResponse.UNKNOWN
            )

    async def Watch(
        self,
        request: health_pb2.HealthCheckRequest,
        context: grpc.aio.ServicerContext
    ) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        """
        Watch health status changes for a service.

        Args:
            request: Health watch request containing service name
            context: gRPC service context

        Yields:
            Health check responses when status changes
        """
        service_name = request.service or ""

        try:
            logger.debug(f"Health watch started for service: '{service_name}'")

            # Handle empty service name (global health watch)
            if not service_name:
                async for response in self._watch_global_health():
                    yield response
                return

            # Check if service is known
            if service_name not in self.service_health:
                yield health_pb2.HealthCheckResponse(
                    status=health_pb2.HealthCheckResponse.SERVICE_UNKNOWN
                )
                return

            service_health = self.service_health[service_name]

            # Check watcher limit
            if len(service_health.watchers) >= self.max_watchers_per_service:
                logger.warning(f"Maximum watchers reached for service '{service_name}'")
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "Too many watchers")
                return

            # Create watcher queue
            watcher_queue = asyncio.Queue(maxsize=100)
            service_health.watchers.add(watcher_queue)

            try:
                # Send initial status
                yield health_pb2.HealthCheckResponse(status=service_health.status.value)

                # Record watcher metric
                metrics_instance.increment_counter(
                    "grpc_health_watchers_total",
                    service=service_name
                )

                # Stream status updates
                while not self.shutdown_event.is_set() and not context.cancelled():
                    try:
                        # Wait for status update with timeout
                        response = await asyncio.wait_for(watcher_queue.get(), timeout=5.0)
                        yield response

                    except asyncio.TimeoutError:
                        # Send periodic heartbeat to keep connection alive
                        yield health_pb2.HealthCheckResponse(status=service_health.status.value)

                    except asyncio.CancelledError:
                        logger.debug(f"Health watch cancelled for service '{service_name}'")
                        break

            finally:
                # Remove watcher from service
                service_health.watchers.discard(watcher_queue)

                logger.debug(f"Health watch ended for service: '{service_name}'")

        except Exception as e:
            logger.error(f"Health watch failed for service '{service_name}': {e}")

    async def set_service_health(
        self,
        service_name: str,
        healthy: bool,
        details: dict[str, Any] | None = None
    ) -> None:
        """
        Set health status for a specific service.

        Args:
            service_name: Name of the service
            healthy: Whether the service is healthy
            details: Optional diagnostic details
        """
        if service_name not in self.service_health:
            # Create new service health entry
            self.service_health[service_name] = ServiceHealth(
                service_name=service_name,
                status=GrpcHealthStatus.UNKNOWN,
                last_check=time.time()
            )

        service_health = self.service_health[service_name]
        old_status = service_health.status

        # Update status
        new_status = GrpcHealthStatus.SERVING if healthy else GrpcHealthStatus.NOT_SERVING
        service_health.status = new_status
        service_health.last_check = time.time()

        if details:
            service_health.details.update(details)

        # Notify watchers if status changed
        if old_status != new_status:
            logger.info(
                f"Service health changed: {service_name}",
                old_status=old_status.name,
                new_status=new_status.name
            )

            # Record status change metric
            metrics_instance.increment_counter(
                "grpc_health_status_changes_total",
                service=service_name,
                from_status=old_status.name.lower(),
                to_status=new_status.name.lower()
            )

            # Notify all watchers
            response = health_pb2.HealthCheckResponse(status=new_status.value)
            await self._notify_watchers(service_health, response)

        # Update global health
        await self._update_global_health()

    async def get_service_health_status(self, service_name: str) -> GrpcHealthStatus | None:
        """Get current health status for a service."""
        service_health = self.service_health.get(service_name)
        return service_health.status if service_health else None

    async def get_all_service_statuses(self) -> dict[str, dict[str, Any]]:
        """Get health status for all services."""
        statuses = {}

        for service_name, service_health in self.service_health.items():
            statuses[service_name] = {
                "status": service_health.status.name,
                "last_check": service_health.last_check,
                "watchers_count": len(service_health.watchers),
                "details": service_health.details if self.enable_detailed_diagnostics else {},
            }

        return statuses

    async def _background_health_monitor(self) -> None:
        """Background task to monitor and update service health."""
        logger.debug("Starting background health monitoring")

        while not self.shutdown_event.is_set():
            try:
                # Update health for all known services
                await self._update_all_service_health()

                # Wait for next check interval
                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                logger.debug("Background health monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Background health monitoring error: {e}")
                await asyncio.sleep(5.0)  # Short delay on error

    async def _update_all_service_health(self) -> None:
        """Update health status for all services."""
        try:
            # Get unified health status from health coordinator
            if self.health_coordinator:
                unified_status = await self.health_coordinator.get_unified_health_status()

                # Update service health based on unified status
                overall_healthy = unified_status.get("overall_status") == "healthy"

                # Map component health to service health
                component_health = unified_status.get("component_health", {})

                service_mapping = {
                    "workspace-qdrant-mcp": ["python_mcp_server"],
                    "workspace-qdrant-rust-daemon": ["rust_daemon"],
                    "workspace-qdrant-cli": ["cli_utility"],
                    "workspace-qdrant-context": ["context_injector"],
                }

                for service_name, component_names in service_mapping.items():
                    service_healthy = True
                    service_details = {}

                    for comp_name in component_names:
                        comp_health = component_health.get(comp_name, {})
                        comp_status = comp_health.get("health_status", "unknown")

                        if comp_status not in ["healthy", "degraded"]:
                            service_healthy = False

                        service_details[f"{comp_name}_status"] = comp_status
                        service_details[f"{comp_name}_response_time"] = comp_health.get("response_time_ms", 0)

                    await self.set_service_health(service_name, service_healthy, service_details)

            else:
                # Fallback to basic health checker
                if self.health_checker:
                    base_health = await self.health_checker.get_health_status()
                    overall_healthy = base_health.get("status") == "healthy"

                    # Set all services to same status
                    for service_name in self.SERVICE_NAMES:
                        await self.set_service_health(service_name, overall_healthy)

        except Exception as e:
            logger.error(f"Failed to update service health: {e}")

    async def _get_global_health_status(self) -> GrpcHealthStatus:
        """Get global health status across all services."""
        if not self.service_health:
            return GrpcHealthStatus.UNKNOWN

        # Global health is serving only if all services are serving
        all_serving = all(
            service_health.status == GrpcHealthStatus.SERVING
            for service_health in self.service_health.values()
        )

        if all_serving:
            return GrpcHealthStatus.SERVING

        # Check if any services are serving
        any_serving = any(
            service_health.status == GrpcHealthStatus.SERVING
            for service_health in self.service_health.values()
        )

        if any_serving:
            return GrpcHealthStatus.NOT_SERVING  # Partial availability
        else:
            return GrpcHealthStatus.NOT_SERVING

    async def _update_global_health(self) -> None:
        """Update global health status and record metrics."""
        old_status = self.global_health_status
        new_status = await self._get_global_health_status()

        if old_status != new_status:
            self.global_health_status = new_status

            logger.info(
                "Global health status changed",
                old_status=old_status.name,
                new_status=new_status.name
            )

            # Record global health metric
            metrics_instance.set_gauge(
                "grpc_global_health_status",
                1 if new_status == GrpcHealthStatus.SERVING else 0
            )

    async def _watch_global_health(self) -> AsyncIterator[health_pb2.HealthCheckResponse]:
        """Watch global health status changes."""
        # Create global watcher queue
        asyncio.Queue(maxsize=100)

        # Store reference for notifications (simplified)
        current_status = self.global_health_status

        try:
            # Send initial status
            yield health_pb2.HealthCheckResponse(status=current_status.value)

            # Monitor for changes
            while not self.shutdown_event.is_set():
                new_status = await self._get_global_health_status()

                if new_status != current_status:
                    current_status = new_status
                    yield health_pb2.HealthCheckResponse(status=current_status.value)

                await asyncio.sleep(1.0)  # Check every second

        except asyncio.CancelledError:
            logger.debug("Global health watch cancelled")

    async def _notify_watchers(
        self,
        service_health: ServiceHealth,
        response: health_pb2.HealthCheckResponse
    ) -> None:
        """Notify all watchers of a service about status change."""
        if not service_health.watchers:
            return

        # Notify all watchers
        failed_watchers = []

        for watcher in service_health.watchers:
            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(watcher.put(response), timeout=1.0)
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Failed to notify watcher: {e}")
                failed_watchers.append(watcher)

        # Remove failed watchers
        for failed_watcher in failed_watchers:
            service_health.watchers.discard(failed_watcher)

        if failed_watchers:
            logger.debug(f"Removed {len(failed_watchers)} failed watchers")


# Global gRPC health service instance
_grpc_health_service: GrpcHealthService | None = None


async def get_grpc_health_service(**kwargs) -> GrpcHealthService:
    """Get or create global gRPC health service instance."""
    global _grpc_health_service

    if _grpc_health_service is None:
        _grpc_health_service = GrpcHealthService(**kwargs)
        await _grpc_health_service.initialize()

    return _grpc_health_service


async def shutdown_grpc_health_service():
    """Shutdown global gRPC health service."""
    global _grpc_health_service

    if _grpc_health_service:
        await _grpc_health_service.shutdown()
        _grpc_health_service = None


def add_grpc_health_service_to_server(
    server: grpc.aio.Server,
    health_service: GrpcHealthService | None = None
) -> GrpcHealthService:
    """
    Add gRPC health service to a gRPC server.

    Args:
        server: gRPC server instance
        health_service: Optional existing health service instance

    Returns:
        Health service instance added to server
    """
    if health_service is None:
        # Create new health service (will be initialized later)
        health_service = GrpcHealthService()

    # Add to server
    health_pb2_grpc.add_HealthServicer_to_server(health_service, server)

    logger.info("gRPC Health Service added to server")
    return health_service
