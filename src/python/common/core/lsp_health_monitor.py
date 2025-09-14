"""
LSP Server Health Monitoring and Recovery System

This module provides comprehensive health monitoring for LSP servers with:
- Periodic health checks and server capability validation
- Automatic recovery with progressive backoff strategies
- Fallback mechanisms when LSP servers are unavailable
- User notification system for issues and troubleshooting
- LSP capability detection and graceful degradation
- Integration with AsyncioLspClient and PriorityQueueManager

Features:
    - Real-time health monitoring with configurable check intervals
    - Circuit breaker patterns for failed LSP server connections
    - Progressive backoff for recovery attempts
    - User-friendly notifications with actionable troubleshooting steps
    - Capability validation and graceful degradation strategies
    - Comprehensive metrics and health status reporting
    - Integration with priority queue for efficient processing
"""

import asyncio
import json
from common.logging.loguru_config import get_logger
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from weakref import WeakSet


from .error_handling import WorkspaceError, ErrorCategory, ErrorSeverity
from .lsp_client import (
    AsyncioLspClient,
    ConnectionState,
    CircuitBreakerState,
    LspError,
    LspTimeoutError,
    ServerCapabilities,
)
from .priority_queue_manager import PriorityQueueManager, MCPActivityLevel

logger = get_logger(__name__)


class HealthStatus(Enum):
    """LSP server health status indicators"""
    
    HEALTHY = "healthy"           # Server is working normally
    DEGRADED = "degraded"         # Server has issues but is functional
    UNHEALTHY = "unhealthy"       # Server has significant problems
    DISCONNECTED = "disconnected" # Server is not connected
    FAILED = "failed"             # Server has failed completely
    UNKNOWN = "unknown"           # Health status cannot be determined


class RecoveryStrategy(Enum):
    """Recovery strategies for failed LSP servers"""
    
    IMMEDIATE = "immediate"       # Attempt immediate reconnection
    BACKOFF = "backoff"          # Use progressive backoff
    CIRCUIT_BREAKER = "circuit_breaker"  # Use circuit breaker pattern
    FALLBACK = "fallback"        # Switch to fallback mode
    DISABLE = "disable"          # Disable the server


class NotificationLevel(Enum):
    """Notification levels for user communication"""
    
    INFO = "info"               # Informational notifications
    WARNING = "warning"         # Warning notifications
    ERROR = "error"            # Error notifications
    CRITICAL = "critical"      # Critical issues requiring attention


@dataclass
class HealthCheckConfig:
    """Configuration for health check behavior"""
    
    # Health check intervals
    check_interval: float = 30.0            # Regular health check interval
    fast_check_interval: float = 5.0        # Fast check when issues detected
    capability_check_interval: float = 300.0  # Capability validation interval
    
    # Timeout settings
    health_check_timeout: float = 10.0      # Health check request timeout
    recovery_timeout: float = 60.0          # Recovery attempt timeout
    
    # Failure thresholds
    consecutive_failures_threshold: int = 3  # Failures before marking unhealthy
    recovery_success_threshold: int = 2      # Successes needed to mark healthy
    
    # Recovery settings
    max_recovery_attempts: int = 5          # Maximum recovery attempts
    base_backoff_delay: float = 1.0        # Base delay for backoff
    max_backoff_delay: float = 300.0       # Maximum backoff delay
    backoff_multiplier: float = 2.0        # Backoff multiplier
    
    # Feature flags
    enable_capability_validation: bool = True
    enable_auto_recovery: bool = True
    enable_fallback_mode: bool = True
    enable_user_notifications: bool = True


@dataclass
class HealthCheckResult:
    """Result of a health check operation"""
    
    timestamp: float
    status: HealthStatus
    response_time_ms: float
    capabilities_valid: bool = True
    error_message: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging and storage"""
        return {
            "timestamp": self.timestamp,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "capabilities_valid": self.capabilities_valid,
            "error_message": self.error_message,
            "error_details": self.error_details,
        }


@dataclass
class ServerHealthInfo:
    """Health information for an LSP server"""
    
    server_name: str
    client: AsyncioLspClient
    last_check: Optional[HealthCheckResult] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    total_failures: int = 0
    last_recovery_attempt: float = 0.0
    recovery_attempts: int = 0
    current_backoff_delay: float = 1.0
    
    # Capability tracking
    last_capabilities_check: float = 0.0
    capabilities_valid: bool = True
    supported_features: Set[str] = field(default_factory=set)
    
    # Statistics
    average_response_time: float = 0.0
    uptime_percentage: float = 100.0
    health_history: List[HealthCheckResult] = field(default_factory=list)
    
    @property
    def current_status(self) -> HealthStatus:
        """Get current health status"""
        if self.last_check is None:
            return HealthStatus.UNKNOWN
        return self.last_check.status
    
    @property
    def is_healthy(self) -> bool:
        """Check if server is considered healthy"""
        return self.current_status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    
    def add_check_result(self, result: HealthCheckResult) -> None:
        """Add a health check result and update statistics"""
        self.last_check = result
        self.total_checks += 1
        
        # Update consecutive counters
        if result.status == HealthStatus.HEALTHY:
            self.consecutive_successes += 1
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.total_failures += 1
        
        # Update response time average
        if self.average_response_time == 0.0:
            self.average_response_time = result.response_time_ms
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_checks - 1) + result.response_time_ms)
                / self.total_checks
            )
        
        # Update uptime percentage
        self.uptime_percentage = ((self.total_checks - self.total_failures) / self.total_checks) * 100
        
        # Maintain health history (keep last 100 results)
        self.health_history.append(result)
        if len(self.health_history) > 100:
            self.health_history.pop(0)


@dataclass
class UserNotification:
    """User notification for LSP health issues"""
    
    timestamp: float
    level: NotificationLevel
    title: str
    message: str
    server_name: str
    troubleshooting_steps: List[str] = field(default_factory=list)
    auto_recovery_attempted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external systems"""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "title": self.title,
            "message": self.message,
            "server_name": self.server_name,
            "troubleshooting_steps": self.troubleshooting_steps,
            "auto_recovery_attempted": self.auto_recovery_attempted,
        }


class LspHealthMonitor:
    """
    Comprehensive LSP server health monitoring and recovery system.
    
    This monitor provides:
    - Periodic health checks with configurable intervals
    - Automatic recovery with progressive backoff
    - Fallback strategies when servers are unavailable
    - User notifications with troubleshooting guidance
    - Capability detection and graceful degradation
    """
    
    def __init__(
        self,
        config: Optional[HealthCheckConfig] = None,
        priority_queue_manager: Optional[PriorityQueueManager] = None,
    ):
        self._config = config or HealthCheckConfig()
        self._priority_queue_manager = priority_queue_manager
        
        # Server tracking
        self._servers: Dict[str, ServerHealthInfo] = {}
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        
        # Notification handlers
        self._notification_handlers: WeakSet[Callable[[UserNotification], None]] = WeakSet()
        
        # Fallback capabilities
        self._fallback_modes: Dict[str, bool] = {}
        self._degraded_features: Set[str] = set()
        
        # Statistics
        self._start_time = time.time()
        self._total_notifications_sent = 0
        self._successful_recoveries = 0
        self._failed_recoveries = 0
        
        logger.info(
            "LSP health monitor initialized",
            check_interval=self._config.check_interval,
            auto_recovery=self._config.enable_auto_recovery,
            fallback_mode=self._config.enable_fallback_mode,
        )
    
    def register_server(
        self,
        server_name: str,
        client: AsyncioLspClient,
        auto_start_monitoring: bool = True,
    ) -> None:
        """
        Register an LSP server for health monitoring.
        
        Args:
            server_name: Unique name for the server
            client: AsyncioLspClient instance
            auto_start_monitoring: Whether to start monitoring immediately
        """
        if server_name in self._servers:
            logger.warning(
                "LSP server already registered, updating",
                server_name=server_name,
            )
        
        server_info = ServerHealthInfo(
            server_name=server_name,
            client=client,
        )
        
        self._servers[server_name] = server_info
        self._fallback_modes[server_name] = False
        
        logger.info(
            "LSP server registered for health monitoring",
            server_name=server_name,
            auto_start=auto_start_monitoring,
        )
        
        if auto_start_monitoring:
            self.start_monitoring(server_name)
    
    def unregister_server(self, server_name: str) -> None:
        """
        Unregister an LSP server from health monitoring.
        
        Args:
            server_name: Name of the server to unregister
        """
        # Stop monitoring task
        if server_name in self._monitoring_tasks:
            self._monitoring_tasks[server_name].cancel()
            del self._monitoring_tasks[server_name]
        
        # Remove server info
        if server_name in self._servers:
            del self._servers[server_name]
        
        if server_name in self._fallback_modes:
            del self._fallback_modes[server_name]
        
        logger.info("LSP server unregistered", server_name=server_name)
    
    def start_monitoring(self, server_name: str) -> None:
        """
        Start health monitoring for a specific server.
        
        Args:
            server_name: Name of the server to monitor
        """
        if server_name not in self._servers:
            raise ValueError(f"Server '{server_name}' not registered")
        
        if server_name in self._monitoring_tasks:
            logger.warning("Monitoring already started", server_name=server_name)
            return
        
        task = asyncio.create_task(
            self._monitoring_loop(server_name),
            name=f"lsp_monitor_{server_name}",
        )
        self._monitoring_tasks[server_name] = task
        
        logger.info("Started LSP health monitoring", server_name=server_name)
    
    def stop_monitoring(self, server_name: str) -> None:
        """
        Stop health monitoring for a specific server.
        
        Args:
            server_name: Name of the server to stop monitoring
        """
        if server_name in self._monitoring_tasks:
            self._monitoring_tasks[server_name].cancel()
            del self._monitoring_tasks[server_name]
            logger.info("Stopped LSP health monitoring", server_name=server_name)
    
    async def shutdown(self) -> None:
        """Shutdown the health monitor and all monitoring tasks"""
        logger.info("Shutting down LSP health monitor")
        
        self._shutdown_event.set()
        
        # Cancel all monitoring tasks
        tasks = list(self._monitoring_tasks.values())
        for task in tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._monitoring_tasks.clear()
        logger.info("LSP health monitor shutdown complete")
    
    def register_notification_handler(
        self,
        handler: Callable[[UserNotification], None],
    ) -> None:
        """
        Register a handler for user notifications.
        
        Args:
            handler: Function to handle notifications
        """
        self._notification_handlers.add(handler)
        logger.debug("Notification handler registered")
    
    async def perform_health_check(
        self,
        server_name: str,
        include_capabilities: bool = False,
    ) -> HealthCheckResult:
        """
        Perform a health check on a specific LSP server.
        
        Args:
            server_name: Name of the server to check
            include_capabilities: Whether to validate capabilities
            
        Returns:
            HealthCheckResult with check details
        """
        if server_name not in self._servers:
            raise ValueError(f"Server '{server_name}' not registered")
        
        server_info = self._servers[server_name]
        client = server_info.client
        start_time = time.time()
        
        try:
            # Check connection state first
            if not client.is_connected():
                return HealthCheckResult(
                    timestamp=start_time,
                    status=HealthStatus.DISCONNECTED,
                    response_time_ms=0.0,
                    capabilities_valid=False,
                    error_message="LSP client not connected",
                )
            
            # Check circuit breaker state
            if client.is_circuit_open:
                return HealthCheckResult(
                    timestamp=start_time,
                    status=HealthStatus.FAILED,
                    response_time_ms=0.0,
                    capabilities_valid=False,
                    error_message="LSP client circuit breaker is open",
                )
            
            # Perform a lightweight health check using workspace/symbol
            # This is a good test as it exercises the server without being too heavy
            try:
                await asyncio.wait_for(
                    client.workspace_symbol("__health_check__"),
                    timeout=self._config.health_check_timeout,
                )
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                
                # Validate capabilities if requested
                capabilities_valid = True
                if include_capabilities:
                    capabilities_valid = await self._validate_capabilities(server_name)
                
                # Determine status based on response time and capabilities
                if response_time > 5000:  # 5 seconds is very slow
                    status = HealthStatus.DEGRADED
                elif not capabilities_valid:
                    status = HealthStatus.DEGRADED
                else:
                    status = HealthStatus.HEALTHY
                
                return HealthCheckResult(
                    timestamp=start_time,
                    status=status,
                    response_time_ms=response_time,
                    capabilities_valid=capabilities_valid,
                )
                
            except asyncio.TimeoutError:
                return HealthCheckResult(
                    timestamp=start_time,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=self._config.health_check_timeout * 1000,
                    capabilities_valid=False,
                    error_message="Health check timed out",
                )
            
        except LspError as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                timestamp=start_time,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                capabilities_valid=False,
                error_message=str(e),
                error_details={"category": e.category.value, "retryable": e.retryable},
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                timestamp=start_time,
                status=HealthStatus.FAILED,
                response_time_ms=response_time,
                capabilities_valid=False,
                error_message=f"Unexpected error during health check: {e}",
                error_details={"exception_type": type(e).__name__},
            )
    
    async def _validate_capabilities(self, server_name: str) -> bool:
        """
        Validate that the LSP server capabilities are as expected.
        
        Args:
            server_name: Name of the server to validate
            
        Returns:
            True if capabilities are valid, False otherwise
        """
        server_info = self._servers[server_name]
        client = server_info.client
        
        if not client.server_capabilities:
            return False
        
        try:
            capabilities = client.server_capabilities
            
            # Check essential capabilities
            essential_features = {
                "hover": capabilities.supports_hover(),
                "definition": capabilities.supports_definition(),
                "references": capabilities.supports_references(),
                "document_symbol": capabilities.supports_document_symbol(),
                "workspace_symbol": capabilities.supports_workspace_symbol(),
            }
            
            # Update supported features
            server_info.supported_features = {
                feature for feature, supported in essential_features.items() 
                if supported
            }
            
            # At least hover and definition should be supported for basic functionality
            basic_functionality = essential_features["hover"] or essential_features["definition"]
            
            server_info.last_capabilities_check = time.time()
            server_info.capabilities_valid = basic_functionality
            
            return basic_functionality
            
        except Exception as e:
            logger.warning(
                "Capability validation failed",
                server_name=server_name,
                error=str(e),
            )
            return False
    
    async def attempt_recovery(
        self,
        server_name: str,
        strategy: Optional[RecoveryStrategy] = None,
    ) -> bool:
        """
        Attempt to recover a failed LSP server.
        
        Args:
            server_name: Name of the server to recover
            strategy: Recovery strategy to use (auto-selected if None)
            
        Returns:
            True if recovery was successful, False otherwise
        """
        if server_name not in self._servers:
            raise ValueError(f"Server '{server_name}' not registered")
        
        server_info = self._servers[server_name]
        
        # Don't attempt recovery if auto-recovery is disabled
        if not self._config.enable_auto_recovery:
            logger.info(
                "Auto-recovery disabled, skipping recovery attempt",
                server_name=server_name,
            )
            return False
        
        # Check if we've exceeded max recovery attempts
        if server_info.recovery_attempts >= self._config.max_recovery_attempts:
            logger.warning(
                "Maximum recovery attempts reached",
                server_name=server_name,
                attempts=server_info.recovery_attempts,
            )
            await self._enable_fallback_mode(server_name)
            return False
        
        # Determine recovery strategy
        if strategy is None:
            strategy = self._determine_recovery_strategy(server_name)
        
        logger.info(
            "Attempting LSP server recovery",
            server_name=server_name,
            strategy=strategy.value,
            attempt=server_info.recovery_attempts + 1,
        )
        
        server_info.recovery_attempts += 1
        server_info.last_recovery_attempt = time.time()
        
        try:
            success = False
            
            if strategy == RecoveryStrategy.IMMEDIATE:
                success = await self._immediate_recovery(server_name)
            elif strategy == RecoveryStrategy.BACKOFF:
                success = await self._backoff_recovery(server_name)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                success = await self._circuit_breaker_recovery(server_name)
            elif strategy == RecoveryStrategy.FALLBACK:
                success = await self._enable_fallback_mode(server_name)
            
            if success:
                self._successful_recoveries += 1
                server_info.recovery_attempts = 0
                server_info.current_backoff_delay = self._config.base_backoff_delay
                
                # Send success notification
                await self._send_notification(UserNotification(
                    timestamp=time.time(),
                    level=NotificationLevel.INFO,
                    title="LSP Server Recovered",
                    message=f"Successfully recovered LSP server '{server_name}'",
                    server_name=server_name,
                    auto_recovery_attempted=True,
                ))
                
                logger.info(
                    "LSP server recovery successful",
                    server_name=server_name,
                    strategy=strategy.value,
                )
            else:
                self._failed_recoveries += 1
                
                # Calculate next backoff delay
                server_info.current_backoff_delay = min(
                    server_info.current_backoff_delay * self._config.backoff_multiplier,
                    self._config.max_backoff_delay,
                )
                
                logger.warning(
                    "LSP server recovery failed",
                    server_name=server_name,
                    strategy=strategy.value,
                    next_backoff=server_info.current_backoff_delay,
                )
            
            return success
            
        except Exception as e:
            self._failed_recoveries += 1
            logger.error(
                "Exception during LSP server recovery",
                server_name=server_name,
                strategy=strategy.value,
                error=str(e),
            )
            return False
    
    def _determine_recovery_strategy(self, server_name: str) -> RecoveryStrategy:
        """
        Determine the best recovery strategy for a server based on its state.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Recommended recovery strategy
        """
        server_info = self._servers[server_name]
        client = server_info.client
        
        # If circuit breaker is open, use circuit breaker recovery
        if client.is_circuit_open:
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        # If we've had multiple failures, use backoff
        if server_info.consecutive_failures >= 3:
            return RecoveryStrategy.BACKOFF
        
        # If this is the first recovery attempt, try immediate
        if server_info.recovery_attempts == 0:
            return RecoveryStrategy.IMMEDIATE
        
        # Default to backoff for subsequent attempts
        return RecoveryStrategy.BACKOFF
    
    async def _immediate_recovery(self, server_name: str) -> bool:
        """
        Attempt immediate recovery by reconnecting the LSP server.
        
        Args:
            server_name: Name of the server to recover
            
        Returns:
            True if recovery was successful
        """
        server_info = self._servers[server_name]
        client = server_info.client
        
        try:
            # Disconnect if still connected
            if client.is_connected():
                await client.disconnect()
            
            # Wait a moment before reconnecting
            await asyncio.sleep(1.0)
            
            # Attempt to reconnect using the last known connection method
            # This is a simplified approach - in practice, you'd need to store
            # the connection parameters and use the appropriate method
            if client.communication_mode.name == "STDIO":
                # For stdio connections, we'd need to restart the process
                logger.warning(
                    "Cannot perform immediate recovery for STDIO connections",
                    server_name=server_name,
                )
                return False
            
            # For TCP connections, attempt reconnection
            # This is simplified - you'd need actual host/port info
            # await client.connect_tcp()
            
            # For now, return False as we need more connection info
            return False
            
        except Exception as e:
            logger.error(
                "Immediate recovery failed",
                server_name=server_name,
                error=str(e),
            )
            return False
    
    async def _backoff_recovery(self, server_name: str) -> bool:
        """
        Attempt recovery with progressive backoff delay.
        
        Args:
            server_name: Name of the server to recover
            
        Returns:
            True if recovery was successful
        """
        server_info = self._servers[server_name]
        
        # Wait for backoff delay
        logger.info(
            "Waiting for backoff delay before recovery",
            server_name=server_name,
            delay=server_info.current_backoff_delay,
        )
        
        await asyncio.sleep(server_info.current_backoff_delay)
        
        # Then attempt immediate recovery
        return await self._immediate_recovery(server_name)
    
    async def _circuit_breaker_recovery(self, server_name: str) -> bool:
        """
        Attempt recovery by waiting for circuit breaker to transition to half-open.
        
        Args:
            server_name: Name of the server to recover
            
        Returns:
            True if recovery was successful
        """
        server_info = self._servers[server_name]
        client = server_info.client
        
        # Wait for circuit breaker timeout
        # The client will automatically transition to half-open
        max_wait = self._config.recovery_timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if client.circuit_breaker_state != CircuitBreakerState.OPEN:
                # Circuit breaker is no longer open, attempt a health check
                result = await self.perform_health_check(server_name)
                return result.status == HealthStatus.HEALTHY
            
            await asyncio.sleep(5.0)  # Check every 5 seconds
        
        logger.warning(
            "Circuit breaker recovery timed out",
            server_name=server_name,
            timeout=max_wait,
        )
        return False
    
    async def _enable_fallback_mode(self, server_name: str) -> bool:
        """
        Enable fallback mode for a server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            Always returns True as fallback is always available
        """
        if not self._config.enable_fallback_mode:
            return False
        
        self._fallback_modes[server_name] = True
        
        # Determine which features to disable
        server_info = self._servers[server_name]
        lost_features = server_info.supported_features.copy()
        self._degraded_features.update(lost_features)
        
        logger.warning(
            "Enabled fallback mode for LSP server",
            server_name=server_name,
            lost_features=list(lost_features),
        )
        
        # Send fallback notification
        await self._send_notification(UserNotification(
            timestamp=time.time(),
            level=NotificationLevel.WARNING,
            title="LSP Server in Fallback Mode",
            message=f"LSP server '{server_name}' is now in fallback mode. "
                   f"Some features may be unavailable: {', '.join(lost_features)}",
            server_name=server_name,
            troubleshooting_steps=[
                "Check if the LSP server process is running",
                "Verify server configuration and paths",
                "Check network connectivity (for remote servers)",
                "Review server logs for error messages",
                "Try restarting the LSP server manually",
            ],
        ))
        
        return True
    
    async def _monitoring_loop(self, server_name: str) -> None:
        """
        Main monitoring loop for a specific LSP server.
        
        Args:
            server_name: Name of the server to monitor
        """
        logger.info("Starting LSP monitoring loop", server_name=server_name)
        server_info = self._servers[server_name]
        
        # Initial capability check
        if self._config.enable_capability_validation:
            await self._validate_capabilities(server_name)
        
        check_interval = self._config.check_interval
        last_capability_check = time.time()
        
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Determine if we need to check capabilities
                    include_capabilities = (
                        self._config.enable_capability_validation and
                        time.time() - last_capability_check >= self._config.capability_check_interval
                    )
                    
                    if include_capabilities:
                        last_capability_check = time.time()
                    
                    # Perform health check
                    result = await self.perform_health_check(
                        server_name,
                        include_capabilities=include_capabilities,
                    )
                    
                    # Update server info
                    server_info.add_check_result(result)
                    
                    # Log health status
                    logger.debug(
                        "LSP health check completed",
                        server_name=server_name,
                        status=result.status.value,
                        response_time_ms=result.response_time_ms,
                        capabilities_valid=result.capabilities_valid,
                    )
                    
                    # Handle health status changes
                    await self._handle_health_status_change(server_name, result)
                    
                    # Adjust check interval based on health status
                    if result.status in (HealthStatus.UNHEALTHY, HealthStatus.FAILED):
                        check_interval = self._config.fast_check_interval
                    else:
                        check_interval = self._config.check_interval
                    
                    # Wait for next check
                    await asyncio.sleep(check_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        "Error in LSP monitoring loop",
                        server_name=server_name,
                        error=str(e),
                    )
                    # Use fast check interval after errors
                    await asyncio.sleep(self._config.fast_check_interval)
                    
        except asyncio.CancelledError:
            pass
        finally:
            logger.info("LSP monitoring loop ended", server_name=server_name)
    
    async def _handle_health_status_change(
        self,
        server_name: str,
        result: HealthCheckResult,
    ) -> None:
        """
        Handle health status changes and trigger appropriate actions.
        
        Args:
            server_name: Name of the server
            result: Latest health check result
        """
        server_info = self._servers[server_name]
        current_status = result.status
        
        # Handle unhealthy states
        if current_status in (HealthStatus.UNHEALTHY, HealthStatus.FAILED, HealthStatus.DISCONNECTED):
            # Check if we need to attempt recovery
            if server_info.consecutive_failures >= self._config.consecutive_failures_threshold:
                # Send warning notification
                await self._send_notification(UserNotification(
                    timestamp=time.time(),
                    level=NotificationLevel.WARNING,
                    title="LSP Server Health Issues",
                    message=f"LSP server '{server_name}' has failed {server_info.consecutive_failures} "
                           f"consecutive health checks. Status: {current_status.value}",
                    server_name=server_name,
                    troubleshooting_steps=self._get_troubleshooting_steps(server_name, result),
                ))
                
                # Attempt recovery
                if self._config.enable_auto_recovery:
                    recovery_successful = await self.attempt_recovery(server_name)
                    if not recovery_successful and server_info.recovery_attempts >= self._config.max_recovery_attempts:
                        # Enable fallback mode after max attempts
                        await self._enable_fallback_mode(server_name)
        
        # Handle recovery to healthy state
        elif current_status == HealthStatus.HEALTHY:
            if server_info.consecutive_successes >= self._config.recovery_success_threshold:
                # Disable fallback mode if it was enabled
                if self._fallback_modes.get(server_name, False):
                    self._fallback_modes[server_name] = False
                    
                    # Remove degraded features
                    self._degraded_features -= server_info.supported_features
                    
                    # Send recovery notification
                    await self._send_notification(UserNotification(
                        timestamp=time.time(),
                        level=NotificationLevel.INFO,
                        title="LSP Server Fully Recovered",
                        message=f"LSP server '{server_name}' is now fully operational. "
                               f"All features have been restored.",
                        server_name=server_name,
                    ))
        
        # Update priority queue manager if available
        if self._priority_queue_manager:
            await self._update_priority_queue_health(server_name, current_status)
    
    async def _update_priority_queue_health(
        self,
        server_name: str,
        status: HealthStatus,
    ) -> None:
        """
        Update priority queue manager with LSP server health information.
        
        Args:
            server_name: Name of the LSP server
            status: Current health status
        """
        try:
            # Determine MCP activity level based on LSP health
            if status == HealthStatus.HEALTHY:
                activity_level = MCPActivityLevel.MODERATE
            elif status == HealthStatus.DEGRADED:
                activity_level = MCPActivityLevel.LOW
            else:
                activity_level = MCPActivityLevel.INACTIVE
            
            # This would be implemented if PriorityQueueManager had a method to update LSP health
            # await self._priority_queue_manager.update_lsp_health(server_name, status, activity_level)
            
        except Exception as e:
            logger.warning(
                "Failed to update priority queue with LSP health",
                server_name=server_name,
                error=str(e),
            )
    
    def _get_troubleshooting_steps(
        self,
        server_name: str,
        result: HealthCheckResult,
    ) -> List[str]:
        """
        Generate troubleshooting steps based on the health check result.
        
        Args:
            server_name: Name of the server
            result: Health check result
            
        Returns:
            List of troubleshooting steps
        """
        steps = []
        
        if result.status == HealthStatus.DISCONNECTED:
            steps.extend([
                f"Check if the LSP server '{server_name}' process is running",
                "Verify server startup command and arguments",
                "Check if the server executable is available in PATH",
                "Review server configuration files",
            ])
        elif result.status == HealthStatus.UNHEALTHY:
            steps.extend([
                f"Check LSP server '{server_name}' logs for error messages",
                "Verify server has sufficient system resources (CPU, memory)",
                "Check if project workspace is accessible to the server",
                "Try restarting the LSP server process",
            ])
        elif result.status == HealthStatus.FAILED:
            steps.extend([
                f"LSP server '{server_name}' has completely failed",
                "Check system logs for crash information",
                "Verify server installation and dependencies",
                "Consider updating or reinstalling the LSP server",
                "Check for conflicts with other language servers",
            ])
        
        # Add capability-specific troubleshooting
        if not result.capabilities_valid:
            steps.extend([
                "Verify server supports required LSP features",
                "Check server version compatibility",
                "Review server initialization parameters",
            ])
        
        # Add performance-related troubleshooting
        if result.response_time_ms > 1000:
            steps.extend([
                "Check system CPU and memory usage",
                "Verify project size is not overwhelming the server",
                "Consider increasing server timeout settings",
                "Check for I/O bottlenecks in workspace directory",
            ])
        
        return steps
    
    async def _send_notification(self, notification: UserNotification) -> None:
        """
        Send a user notification through all registered handlers.
        
        Args:
            notification: Notification to send
        """
        if not self._config.enable_user_notifications:
            return
        
        self._total_notifications_sent += 1
        
        # Log the notification
        log_level = logging.WARNING if notification.level in (NotificationLevel.WARNING, NotificationLevel.ERROR) else logging.INFO
        logger.log(
            log_level,
            "LSP health notification",
            title=notification.title,
            message=notification.message,
            server_name=notification.server_name,
            notification_level=notification.level.value,
        )
        
        # Send to registered handlers
        for handler in list(self._notification_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception as e:
                logger.error(
                    "Error in notification handler",
                    handler=str(handler),
                    error=str(e),
                )
    
    def get_server_health(self, server_name: str) -> Optional[ServerHealthInfo]:
        """
        Get health information for a specific server.
        
        Args:
            server_name: Name of the server
            
        Returns:
            ServerHealthInfo if server exists, None otherwise
        """
        return self._servers.get(server_name)
    
    def get_all_servers_health(self) -> Dict[str, ServerHealthInfo]:
        """
        Get health information for all monitored servers.
        
        Returns:
            Dictionary mapping server names to health info
        """
        return self._servers.copy()
    
    def is_feature_available(self, feature: str) -> bool:
        """
        Check if a specific LSP feature is available across all servers.
        
        Args:
            feature: Feature name to check
            
        Returns:
            True if feature is available on at least one healthy server
        """
        if feature in self._degraded_features:
            return False
        
        for server_info in self._servers.values():
            if server_info.is_healthy and feature in server_info.supported_features:
                return True
        
        return False
    
    def get_available_features(self) -> Set[str]:
        """
        Get all LSP features currently available.
        
        Returns:
            Set of available feature names
        """
        available_features = set()
        
        for server_info in self._servers.values():
            if server_info.is_healthy:
                available_features.update(server_info.supported_features)
        
        # Remove degraded features
        available_features -= self._degraded_features
        
        return available_features
    
    def get_health_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive health statistics for the monitor.
        
        Returns:
            Dictionary with health statistics
        """
        total_servers = len(self._servers)
        healthy_servers = sum(1 for info in self._servers.values() if info.is_healthy)
        
        uptime = time.time() - self._start_time
        
        return {
            "monitor_uptime_seconds": uptime,
            "total_servers": total_servers,
            "healthy_servers": healthy_servers,
            "unhealthy_servers": total_servers - healthy_servers,
            "servers_in_fallback": sum(1 for enabled in self._fallback_modes.values() if enabled),
            "degraded_features": list(self._degraded_features),
            "available_features": list(self.get_available_features()),
            "total_notifications_sent": self._total_notifications_sent,
            "successful_recoveries": self._successful_recoveries,
            "failed_recoveries": self._failed_recoveries,
            "recovery_success_rate": (
                self._successful_recoveries / max(1, self._successful_recoveries + self._failed_recoveries)
            ),
            "servers": {
                name: {
                    "status": info.current_status.value,
                    "uptime_percentage": info.uptime_percentage,
                    "average_response_time_ms": info.average_response_time,
                    "consecutive_failures": info.consecutive_failures,
                    "total_checks": info.total_checks,
                    "total_failures": info.total_failures,
                    "recovery_attempts": info.recovery_attempts,
                    "supported_features": list(info.supported_features),
                    "in_fallback_mode": self._fallback_modes.get(name, False),
                }
                for name, info in self._servers.items()
            },
        }
    
    @asynccontextmanager
    async def monitoring_context(self):
        """
        Async context manager for health monitoring lifecycle.
        
        Usage:
            async with monitor.monitoring_context():
                # Health monitoring is active
                await do_work()
            # Monitoring is automatically stopped
        """
        # Start monitoring for all registered servers
        for server_name in self._servers:
            if server_name not in self._monitoring_tasks:
                self.start_monitoring(server_name)
        
        try:
            yield self
        finally:
            await self.shutdown()