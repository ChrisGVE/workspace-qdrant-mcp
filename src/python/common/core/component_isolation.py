"""
Component Isolation and Failure Handling System for Four-Component Architecture.

This module provides comprehensive component isolation mechanisms to prevent cascading failures
and ensure that component failures don't bring down the entire system. It implements error
boundaries, process separation, resource limits, and timeout handling.

Key Features:
    - Error boundaries to prevent failure propagation between components
    - Process-level isolation with resource limits and monitoring
    - Comprehensive timeout handling for all inter-component communications
    - Detailed error logging and failure analysis
    - Integration with existing graceful degradation and recovery systems
    - Resource isolation and quota management per component
    - Failure impact analysis and containment strategies

Isolation Strategies:
    - PROCESS_SEPARATION: Each component runs in isolated process with resource limits
    - RESOURCE_ISOLATION: Memory, CPU, and disk quotas per component
    - COMMUNICATION_BOUNDARIES: Timeouts and error handling for all inter-component calls
    - FAILURE_CONTAINMENT: Prevent failures from propagating across component boundaries
    - STATE_ISOLATION: Independent state management with transaction boundaries

Example:
    ```python
    from workspace_qdrant_mcp.core.component_isolation import IsolationManager

    # Initialize isolation manager
    isolation = IsolationManager(
        lifecycle_manager=lifecycle_manager,
        coordinator=coordinator,
        degradation_manager=degradation_manager,
        recovery_manager=recovery_manager
    )
    await isolation.initialize()

    # Set resource limits for a component
    await isolation.set_component_resource_limits(
        ComponentType.RUST_DAEMON,
        cpu_limit_percent=50.0,
        memory_limit_mb=512,
        disk_limit_mb=1024
    )

    # Execute component operation with isolation
    async with isolation.component_boundary(ComponentType.PYTHON_MCP_SERVER):
        result = await some_component_operation()
    ```
"""

import asyncio
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, EnumMeta
from typing import Any
from weakref import WeakSet

import psutil
from loguru import logger

from .automatic_recovery import RecoveryManager
from .component_coordination import (
    ComponentCoordinator,
    ComponentHealth,
    ComponentType,
)
from .component_lifecycle import (
    ComponentLifecycleManager,
)
from .graceful_degradation import (
    DegradationManager,
)
from .lsp_health_monitor import (
    LspHealthMonitor,
    NotificationLevel,
    UserNotification,
)


class DefaultEnumMeta(EnumMeta):
    """Enum metaclass that allows no-arg instantiation for tests."""

    def __call__(cls, *args, **kwargs):
        if not args and not kwargs:
            return next(iter(cls))
        return super().__call__(*args, **kwargs)


class DefaultEnum(Enum, metaclass=DefaultEnumMeta):
    """Enum base with a default value for no-arg construction."""


class IsolationStrategy(DefaultEnum):
    """Component isolation strategies."""

    PROCESS_SEPARATION = "process_separation"     # Each component in separate process
    RESOURCE_ISOLATION = "resource_isolation"     # CPU, memory, disk limits
    COMMUNICATION_BOUNDARIES = "communication_boundaries"  # Timeouts and error handling
    FAILURE_CONTAINMENT = "failure_containment"   # Prevent failure propagation
    STATE_ISOLATION = "state_isolation"           # Independent state management


class BoundaryType(DefaultEnum):
    """Types of component boundaries."""

    SYNCHRONOUS = "synchronous"       # Sync call with timeout
    ASYNCHRONOUS = "asynchronous"     # Async call with callback
    MESSAGE_QUEUE = "message_queue"   # Queue-based communication
    EVENT_STREAM = "event_stream"     # Event-driven communication
    SHARED_STATE = "shared_state"     # Shared state access


class FailureImpact(DefaultEnum):
    """Failure impact levels for isolation decisions."""

    ISOLATED = "isolated"           # Failure contained within component
    LIMITED = "limited"            # Failure affects dependent components only
    CASCADING = "cascading"        # Failure spreads to multiple components
    SYSTEM_WIDE = "system_wide"    # Failure affects entire system


class ResourceType(DefaultEnum):
    """Types of resources for isolation management."""

    CPU = "cpu"
    MEMORY = "memory"
    DISK_SPACE = "disk_space"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    FILE_DESCRIPTORS = "file_descriptors"
    PROCESSES = "processes"
    THREADS = "threads"


@dataclass
class ResourceLimits:
    """Resource limits for component isolation."""

    cpu_limit_percent: float | None = None      # CPU usage limit (0-100%)
    memory_limit_mb: int | None = None          # Memory limit in MB
    disk_limit_mb: int | None = None            # Disk usage limit in MB
    disk_io_limit_mbps: float | None = None     # Disk I/O limit in MB/s
    network_io_limit_mbps: float | None = None  # Network I/O limit in MB/s
    max_file_descriptors: int | None = None     # Max open file descriptors
    max_processes: int | None = None            # Max child processes
    max_threads: int | None = None              # Max threads per component

    # Enforcement settings
    enforce_cpu: bool = True
    enforce_memory: bool = True
    enforce_disk: bool = True
    kill_on_violation: bool = False                # Kill process on resource violation
    warn_threshold: float = 0.8                    # Warning threshold (80% of limit)


@dataclass
class ComponentBoundary:
    """Configuration for component boundary protection."""

    component_type: ComponentType
    boundary_type: BoundaryType
    timeout_seconds: float = 30.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_enabled: bool = True
    fallback_strategy: str | None = None
    isolation_level: IsolationStrategy = IsolationStrategy.COMMUNICATION_BOUNDARIES

    # Error handling
    allowed_exceptions: set[str] = field(default_factory=set)
    critical_exceptions: set[str] = field(default_factory=set)
    max_error_rate: float = 0.1  # 10% error rate threshold
    error_window_seconds: int = 300  # 5-minute error window

    # Resource protection
    max_concurrent_calls: int = 100
    queue_timeout_seconds: float = 10.0
    memory_threshold_mb: int | None = None


@dataclass
class IsolationEvent:
    """Event record for isolation actions."""

    event_id: str
    component_id: str
    isolation_strategy: IsolationStrategy
    boundary_type: BoundaryType
    trigger_reason: str
    action_taken: str
    impact_level: FailureImpact
    resource_usage: dict[str, float]
    error_details: dict[str, Any] | None = None
    recovery_triggered: bool = False
    timestamp: datetime = None
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessInfo:
    """Information about isolated component process."""

    component_type: ComponentType
    process_id: int
    parent_process_id: int | None
    command_line: list[str]
    working_directory: str
    environment: dict[str, str]
    resource_limits: ResourceLimits
    start_time: datetime

    # Runtime information
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    file_descriptor_count: int = 0
    thread_count: int = 0
    child_process_count: int = 0

    # Status tracking
    is_running: bool = True
    exit_code: int | None = None
    termination_reason: str | None = None
    last_health_check: datetime | None = None
    resource_violations: list[str] = field(default_factory=list)


class ComponentIsolationManager:
    """
    Comprehensive component isolation and failure handling manager.

    Provides error boundaries, process separation, resource limits, and timeout handling
    to prevent cascading failures and ensure system stability.
    """

    # Default resource limits for each component type
    DEFAULT_RESOURCE_LIMITS = {
        ComponentType.RUST_DAEMON: ResourceLimits(
            cpu_limit_percent=50.0,
            memory_limit_mb=512,
            disk_limit_mb=1024,
            max_file_descriptors=1000,
            max_processes=10,
            max_threads=50
        ),
        ComponentType.PYTHON_MCP_SERVER: ResourceLimits(
            cpu_limit_percent=30.0,
            memory_limit_mb=256,
            disk_limit_mb=512,
            max_file_descriptors=500,
            max_processes=5,
            max_threads=20
        ),
        ComponentType.CLI_UTILITY: ResourceLimits(
            cpu_limit_percent=10.0,
            memory_limit_mb=128,
            disk_limit_mb=256,
            max_file_descriptors=100,
            max_processes=3,
            max_threads=10
        ),
        ComponentType.CONTEXT_INJECTOR: ResourceLimits(
            cpu_limit_percent=20.0,
            memory_limit_mb=192,
            disk_limit_mb=384,
            max_file_descriptors=200,
            max_processes=5,
            max_threads=15
        ),
    }

    # Default boundary configurations
    DEFAULT_BOUNDARIES = {
        ComponentType.RUST_DAEMON: ComponentBoundary(
            component_type=ComponentType.RUST_DAEMON,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=30.0,
            retry_count=3,
            max_concurrent_calls=50,
            memory_threshold_mb=400
        ),
        ComponentType.PYTHON_MCP_SERVER: ComponentBoundary(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            boundary_type=BoundaryType.MESSAGE_QUEUE,
            timeout_seconds=60.0,
            retry_count=2,
            max_concurrent_calls=100,
            memory_threshold_mb=200
        ),
        ComponentType.CLI_UTILITY: ComponentBoundary(
            component_type=ComponentType.CLI_UTILITY,
            boundary_type=BoundaryType.SYNCHRONOUS,
            timeout_seconds=10.0,
            retry_count=1,
            max_concurrent_calls=10,
            memory_threshold_mb=100
        ),
        ComponentType.CONTEXT_INJECTOR: ComponentBoundary(
            component_type=ComponentType.CONTEXT_INJECTOR,
            boundary_type=BoundaryType.EVENT_STREAM,
            timeout_seconds=45.0,
            retry_count=2,
            max_concurrent_calls=20,
            memory_threshold_mb=150
        ),
    }

    def __init__(
        self,
        lifecycle_manager: ComponentLifecycleManager | None = None,
        coordinator: ComponentCoordinator | None = None,
        degradation_manager: DegradationManager | None = None,
        recovery_manager: RecoveryManager | None = None,
        health_monitor: LspHealthMonitor | None = None,
        config: dict[str, Any] | None = None
    ):
        """
        Initialize component isolation manager.

        Args:
            lifecycle_manager: Component lifecycle manager instance
            coordinator: Component coordinator instance
            degradation_manager: Graceful degradation manager instance
            recovery_manager: Automatic recovery manager instance
            health_monitor: LSP health monitor instance
            config: Configuration dictionary
        """
        self.lifecycle_manager = lifecycle_manager
        self.coordinator = coordinator
        self.degradation_manager = degradation_manager
        self.recovery_manager = recovery_manager
        self.health_monitor = health_monitor
        self.config = config or {}

        # Isolation state
        self.component_processes: dict[str, ProcessInfo] = {}
        self.resource_limits: dict[ComponentType, ResourceLimits] = {}
        self.component_boundaries: dict[ComponentType, ComponentBoundary] = {}
        self.active_boundaries: dict[str, list[str]] = {}  # component_id -> list of active boundary calls

        # Error tracking and boundaries
        self.error_counts: dict[str, list[datetime]] = {}
        self.isolation_events: list[IsolationEvent] = []
        self.boundary_violations: dict[str, int] = {}

        # Resource monitoring
        self.resource_usage_history: dict[str, list[tuple[datetime, dict[str, float]]]] = {}

        # Background tasks
        self.monitoring_tasks: list[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        # Timeout and concurrency management
        self.active_calls: dict[str, set[str]] = {}  # component_id -> set of call_ids
        self.call_timeouts: dict[str, asyncio.Handle] = {}  # call_id -> timeout handle
        self.semaphores: dict[ComponentType, asyncio.Semaphore] = {}

        # Notification handlers
        self.notification_handlers: WeakSet[Callable[[UserNotification], None]] = WeakSet()

        # Statistics
        self.start_time = time.time()
        self.isolation_count = 0
        self.boundary_violations_count = 0
        self.resource_violations_count = 0
        self.process_restarts_count = 0

        # Initialize defaults
        self.resource_limits = self.DEFAULT_RESOURCE_LIMITS.copy()
        self.component_boundaries = self.DEFAULT_BOUNDARIES.copy()

        # Create semaphores for each component
        for component_type in ComponentType:
            boundary = self.component_boundaries.get(component_type)
            if boundary:
                self.semaphores[component_type] = asyncio.Semaphore(boundary.max_concurrent_calls)
            else:
                self.semaphores[component_type] = asyncio.Semaphore(10)  # Default limit

        logger.info("Component isolation manager initialized")

    async def initialize(self) -> bool:
        """
        Initialize the component isolation manager.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize resource monitoring for existing processes
            await self._discover_existing_processes()

            # Start monitoring tasks
            await self._start_monitoring_tasks()

            # Register with other managers for notifications
            if self.degradation_manager:
                self.degradation_manager.register_notification_handler(
                    self._handle_degradation_notification
                )

            if self.recovery_manager:
                self.recovery_manager.register_notification_handler(
                    self._handle_recovery_notification
                )

            if self.health_monitor:
                self.health_monitor.register_notification_handler(
                    self._handle_health_notification
                )

            logger.info("Component isolation manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize component isolation manager: {e}")
            return False

    async def _discover_existing_processes(self):
        """Discover and register existing component processes."""
        try:
            # Get current system processes
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
                try:
                    proc_info = proc.info
                    cmdline = proc_info.get('cmdline', [])

                    # Try to identify component processes
                    component_type = self._identify_component_from_cmdline(cmdline)
                    if component_type:
                        process_info = ProcessInfo(
                            component_type=component_type,
                            process_id=proc_info['pid'],
                            parent_process_id=proc.ppid(),
                            command_line=cmdline,
                            working_directory=proc.cwd() if proc.is_running() else "",
                            environment={},  # Can't easily get env for existing process
                            resource_limits=self.resource_limits.get(component_type, ResourceLimits()),
                            start_time=datetime.fromtimestamp(proc_info['create_time'], timezone.utc)
                        )

                        component_id = f"{component_type.value}-{proc_info['pid']}"
                        self.component_processes[component_id] = process_info

                        logger.info(f"Discovered existing process: {component_id}")

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

        except Exception as e:
            logger.error(f"Failed to discover existing processes: {e}")

    def _identify_component_from_cmdline(self, cmdline: list[str]) -> ComponentType | None:
        """Identify component type from command line."""
        if not cmdline:
            return None

        cmdline_str = " ".join(cmdline).lower()

        if "rust" in cmdline_str and "daemon" in cmdline_str:
            return ComponentType.RUST_DAEMON
        elif "mcp" in cmdline_str and "server" in cmdline_str:
            return ComponentType.PYTHON_MCP_SERVER
        elif "cli" in cmdline_str and ("workspace" in cmdline_str or "qdrant" in cmdline_str):
            return ComponentType.CLI_UTILITY
        elif "context" in cmdline_str and "injector" in cmdline_str:
            return ComponentType.CONTEXT_INJECTOR

        return None

    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        self.monitoring_tasks = [
            asyncio.create_task(self._resource_monitoring_loop()),
            asyncio.create_task(self._boundary_monitoring_loop()),
            asyncio.create_task(self._process_health_monitoring_loop()),
            asyncio.create_task(self._timeout_cleanup_loop()),
            asyncio.create_task(self._isolation_metrics_loop()),
        ]
        logger.info("Started component isolation monitoring tasks")

    async def _resource_monitoring_loop(self):
        """Monitor resource usage for all component processes."""
        while not self.shutdown_event.is_set():
            try:
                await self._check_resource_usage()
                await asyncio.sleep(10.0)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _boundary_monitoring_loop(self):
        """Monitor component boundary violations and error rates."""
        while not self.shutdown_event.is_set():
            try:
                await self._check_boundary_violations()
                await asyncio.sleep(30.0)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in boundary monitoring loop: {e}")
                await asyncio.sleep(10.0)

    async def _process_health_monitoring_loop(self):
        """Monitor health of isolated component processes."""
        while not self.shutdown_event.is_set():
            try:
                await self._check_process_health()
                await asyncio.sleep(15.0)  # Check every 15 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process health monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _timeout_cleanup_loop(self):
        """Clean up expired timeouts and stale calls."""
        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_expired_timeouts()
                await asyncio.sleep(60.0)  # Clean every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in timeout cleanup loop: {e}")
                await asyncio.sleep(30.0)

    async def _isolation_metrics_loop(self):
        """Update isolation metrics and statistics."""
        while not self.shutdown_event.is_set():
            try:
                await self._update_isolation_metrics()
                await asyncio.sleep(300.0)  # Update every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in isolation metrics loop: {e}")
                await asyncio.sleep(60.0)

    async def _check_resource_usage(self):
        """Check resource usage for all component processes."""
        current_time = datetime.now(timezone.utc)

        for component_id, process_info in list(self.component_processes.items()):
            try:
                if not process_info.is_running:
                    continue

                # Get process handle
                try:
                    proc = psutil.Process(process_info.process_id)
                    if not proc.is_running():
                        process_info.is_running = False
                        continue

                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    process_info.is_running = False
                    continue

                # Get resource usage
                cpu_percent = proc.cpu_percent()
                memory_info = proc.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

                # Get file descriptor count (Unix only)
                fd_count = 0
                try:
                    fd_count = proc.num_fds() if hasattr(proc, 'num_fds') else 0
                except (psutil.AccessDenied, AttributeError):
                    pass

                # Get thread count
                thread_count = proc.num_threads()

                # Update process info
                process_info.cpu_usage_percent = cpu_percent
                process_info.memory_usage_mb = memory_mb
                process_info.file_descriptor_count = fd_count
                process_info.thread_count = thread_count
                process_info.last_health_check = current_time

                # Store usage history
                if component_id not in self.resource_usage_history:
                    self.resource_usage_history[component_id] = []

                usage_data = {
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_mb,
                    'fd_count': fd_count,
                    'thread_count': thread_count
                }

                self.resource_usage_history[component_id].append((current_time, usage_data))

                # Keep only last hour of data
                cutoff_time = current_time - timedelta(hours=1)
                self.resource_usage_history[component_id] = [
                    (ts, data) for ts, data in self.resource_usage_history[component_id]
                    if ts > cutoff_time
                ]

                # Check for resource violations
                await self._check_resource_violations(component_id, process_info)

            except Exception as e:
                logger.error(f"Error checking resource usage for {component_id}: {e}")

    async def _check_resource_violations(self, component_id: str, process_info: ProcessInfo):
        """Check for resource limit violations."""
        limits = process_info.resource_limits
        violations = []

        # Check CPU limit
        if limits.cpu_limit_percent and limits.enforce_cpu:
            if process_info.cpu_usage_percent > limits.cpu_limit_percent:
                violations.append(f"CPU usage {process_info.cpu_usage_percent:.1f}% exceeds limit {limits.cpu_limit_percent:.1f}%")

        # Check memory limit
        if limits.memory_limit_mb and limits.enforce_memory:
            if process_info.memory_usage_mb > limits.memory_limit_mb:
                violations.append(f"Memory usage {process_info.memory_usage_mb:.1f}MB exceeds limit {limits.memory_limit_mb}MB")

        # Check file descriptor limit
        if limits.max_file_descriptors:
            if process_info.file_descriptor_count > limits.max_file_descriptors:
                violations.append(f"File descriptors {process_info.file_descriptor_count} exceeds limit {limits.max_file_descriptors}")

        # Check thread limit
        if limits.max_threads:
            if process_info.thread_count > limits.max_threads:
                violations.append(f"Thread count {process_info.thread_count} exceeds limit {limits.max_threads}")

        if violations:
            process_info.resource_violations.extend(violations)
            self.resource_violations_count += len(violations)

            # Log violation event
            event = IsolationEvent(
                event_id=str(uuid.uuid4()),
                component_id=component_id,
                isolation_strategy=IsolationStrategy.RESOURCE_ISOLATION,
                boundary_type=BoundaryType.SYNCHRONOUS,
                trigger_reason="Resource limit violations",
                action_taken="Logged violations and monitoring",
                impact_level=FailureImpact.LIMITED,
                resource_usage={
                    'cpu_percent': process_info.cpu_usage_percent,
                    'memory_mb': process_info.memory_usage_mb,
                    'fd_count': process_info.file_descriptor_count,
                    'thread_count': process_info.thread_count
                },
                error_details={'violations': violations}
            )

            self.isolation_events.append(event)

            # Send notification
            await self._send_resource_violation_notification(component_id, violations)

            # Take action based on configuration
            if limits.kill_on_violation:
                await self._terminate_violating_process(component_id, violations)

            logger.warning(f"Resource violations detected for {component_id}: {violations}")

    async def _check_boundary_violations(self):
        """Check for component boundary violations."""
        current_time = datetime.now(timezone.utc)

        for component_type, boundary in self.component_boundaries.items():
            # Check error rates
            error_key = f"{component_type.value}_errors"
            if error_key in self.error_counts:
                # Remove old errors outside the window
                window_start = current_time - timedelta(seconds=boundary.error_window_seconds)
                self.error_counts[error_key] = [
                    ts for ts in self.error_counts[error_key] if ts > window_start
                ]

                # Check if error rate exceeds threshold
                error_count = len(self.error_counts[error_key])
                error_rate = error_count / boundary.error_window_seconds

                if error_rate > boundary.max_error_rate:
                    self.boundary_violations_count += 1

                    # Create isolation event
                    event = IsolationEvent(
                        event_id=str(uuid.uuid4()),
                        component_id=f"{component_type.value}-boundary",
                        isolation_strategy=IsolationStrategy.FAILURE_CONTAINMENT,
                        boundary_type=boundary.boundary_type,
                        trigger_reason=f"Error rate {error_rate:.3f} exceeds threshold {boundary.max_error_rate}",
                        action_taken="Triggered boundary protection",
                        impact_level=FailureImpact.CASCADING,
                        resource_usage={},
                        error_details={'error_rate': error_rate, 'error_count': error_count}
                    )

                    self.isolation_events.append(event)

                    # Trigger degradation or recovery if managers available
                    if self.degradation_manager:
                        component_id = f"{component_type.value}-default"
                        await self.degradation_manager.record_component_failure(component_id)

                    logger.warning(f"Boundary violation: {component_type.value} error rate {error_rate:.3f} exceeds {boundary.max_error_rate}")

    async def _check_process_health(self):
        """Check health of all component processes."""
        for component_id, process_info in list(self.component_processes.items()):
            try:
                if not process_info.is_running:
                    continue

                # Check if process is still alive
                try:
                    proc = psutil.Process(process_info.process_id)
                    if not proc.is_running():
                        process_info.is_running = False
                        process_info.termination_reason = "Process no longer running"

                        # Log process termination
                        await self._handle_process_termination(component_id, process_info)
                        continue

                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    process_info.is_running = False
                    process_info.termination_reason = "Process not found"
                    await self._handle_process_termination(component_id, process_info)
                    continue

                # Check for health indicators
                current_time = datetime.now(timezone.utc)
                if process_info.last_health_check:
                    time_since_check = (current_time - process_info.last_health_check).total_seconds()

                    # If no health check for too long, consider unhealthy
                    if time_since_check > 300:  # 5 minutes
                        logger.warning(f"Process {component_id} hasn't been health-checked for {time_since_check:.1f} seconds")

                # Update coordinator if available
                if self.coordinator:
                    health_status = ComponentHealth.HEALTHY
                    if process_info.resource_violations:
                        health_status = ComponentHealth.WARNING if len(process_info.resource_violations) < 3 else ComponentHealth.CRITICAL

                    await self.coordinator.update_component_health(
                        component_id,
                        health_status,
                        {
                            'cpu_usage': process_info.cpu_usage_percent,
                            'memory_mb': process_info.memory_usage_mb,
                            'fd_count': float(process_info.file_descriptor_count),
                            'thread_count': float(process_info.thread_count)
                        }
                    )

            except Exception as e:
                logger.error(f"Error checking process health for {component_id}: {e}")

    async def _handle_process_termination(self, component_id: str, process_info: ProcessInfo):
        """Handle termination of a component process."""
        logger.warning(f"Process terminated: {component_id} - {process_info.termination_reason}")

        # Create isolation event
        event = IsolationEvent(
            event_id=str(uuid.uuid4()),
            component_id=component_id,
            isolation_strategy=IsolationStrategy.PROCESS_SEPARATION,
            boundary_type=BoundaryType.SYNCHRONOUS,
            trigger_reason=process_info.termination_reason or "Process termination",
            action_taken="Process isolation and cleanup",
            impact_level=FailureImpact.LIMITED,
            resource_usage={
                'cpu_percent': process_info.cpu_usage_percent,
                'memory_mb': process_info.memory_usage_mb
            }
        )

        self.isolation_events.append(event)

        # Trigger recovery if available
        if self.recovery_manager:
            try:
                await self.recovery_manager.trigger_component_recovery(
                    process_info.component_type,
                    reason=f"Process termination: {process_info.termination_reason}"
                )
            except Exception as e:
                logger.error(f"Failed to trigger recovery for {component_id}: {e}")

        # Clean up process info
        del self.component_processes[component_id]
        self.process_restarts_count += 1

    async def _cleanup_expired_timeouts(self):
        """Clean up expired timeout handles and stale calls."""
        time.time()
        expired_calls = []

        for call_id, timeout_handle in list(self.call_timeouts.items()):
            # Check if handle is cancelled or has fired
            if timeout_handle.cancelled():
                expired_calls.append(call_id)

        for call_id in expired_calls:
            del self.call_timeouts[call_id]

            # Remove from active calls
            for _component_id, call_set in self.active_calls.items():
                call_set.discard(call_id)

        if expired_calls:
            logger.debug(f"Cleaned up {len(expired_calls)} expired timeout calls")

    async def _update_isolation_metrics(self):
        """Update isolation metrics and statistics."""
        try:
            # Calculate uptime
            uptime = time.time() - self.start_time

            # Count active processes
            active_processes = sum(1 for p in self.component_processes.values() if p.is_running)

            # Count recent violations
            current_time = datetime.now(timezone.utc)
            recent_cutoff = current_time - timedelta(hours=1)
            recent_events = [
                e for e in self.isolation_events
                if e.timestamp > recent_cutoff
            ]

            metrics = {
                'uptime_seconds': uptime,
                'active_processes': active_processes,
                'total_processes': len(self.component_processes),
                'isolation_events': len(self.isolation_events),
                'recent_events': len(recent_events),
                'boundary_violations': self.boundary_violations_count,
                'resource_violations': self.resource_violations_count,
                'process_restarts': self.process_restarts_count
            }

            logger.debug(f"Isolation metrics: {metrics}")

        except Exception as e:
            logger.error(f"Error updating isolation metrics: {e}")

    @asynccontextmanager
    async def component_boundary(
        self,
        component_type: ComponentType,
        operation_name: str = "unknown",
        timeout_override: float | None = None,
        allow_degraded: bool = False
    ):
        """
        Async context manager for executing operations within component boundaries.

        Provides timeout handling, error isolation, and resource protection.

        Args:
            component_type: Target component type
            operation_name: Name of the operation for logging
            timeout_override: Override default timeout
            allow_degraded: Allow operation in degraded mode

        Yields:
            Context for protected operation execution

        Raises:
            TimeoutError: If operation exceeds timeout
            ResourceError: If resource limits would be exceeded
            ComponentUnavailableError: If component is not available
        """
        boundary = self.component_boundaries.get(component_type, self.DEFAULT_BOUNDARIES[component_type])
        timeout = timeout_override or boundary.timeout_seconds
        call_id = str(uuid.uuid4())
        component_id = f"{component_type.value}-default"

        # Check if component is available
        if not allow_degraded and self.degradation_manager:
            if not self.degradation_manager.is_component_available(component_id):
                raise ComponentUnavailableError(f"Component {component_type.value} is not available")

        # Acquire semaphore for concurrency control
        semaphore = self.semaphores.get(component_type, asyncio.Semaphore(10))

        try:
            # Wait for semaphore with timeout
            await asyncio.wait_for(semaphore.acquire(), timeout=boundary.queue_timeout_seconds)

            # Track active call
            if component_id not in self.active_calls:
                self.active_calls[component_id] = set()
            self.active_calls[component_id].add(call_id)

            start_time = time.time()

            try:
                # Create boundary context and execute with timeout
                async with asyncio.timeout(timeout):
                    # Create boundary context
                    context = ComponentBoundaryContext(
                        component_type=component_type,
                        component_id=component_id,
                        call_id=call_id,
                        operation_name=operation_name,
                        start_time=start_time,
                        isolation_manager=self
                    )

                    yield context

                # Record successful operation
                await self._record_successful_operation(component_type, operation_name, time.time() - start_time)

            except asyncio.TimeoutError:
                # Operation timeout
                await self._record_operation_error(component_type, operation_name, asyncio.TimeoutError("Operation timeout"))
                raise

            except Exception as e:
                # Record error
                await self._record_operation_error(component_type, operation_name, e)

                # Check if error should trigger boundary protection
                if not self._is_allowed_exception(boundary, e):
                    if self._is_critical_exception(boundary, e):
                        # Critical error - trigger immediate isolation
                        await self._trigger_component_isolation(component_type, str(e))

                    # Re-raise the exception
                    raise

        except asyncio.TimeoutError:
            # Queue timeout
            await self._record_operation_error(component_type, operation_name, TimeoutError("Queue timeout"))
            raise TimeoutError(f"Queue timeout for {component_type.value}")

        finally:
            # Clean up active calls
            if component_id in self.active_calls:
                self.active_calls[component_id].discard(call_id)

            try:
                semaphore.release()
            except ValueError:
                pass  # Semaphore might have been released already

    def _handle_call_timeout(self, call_id: str, component_id: str, operation_name: str):
        """Handle call timeout."""
        logger.warning(f"Call timeout: {operation_name} for {component_id} (call_id: {call_id})")

        # Remove from active calls
        if component_id in self.active_calls:
            self.active_calls[component_id].discard(call_id)

        # Create timeout event
        event = IsolationEvent(
            event_id=str(uuid.uuid4()),
            component_id=component_id,
            isolation_strategy=IsolationStrategy.COMMUNICATION_BOUNDARIES,
            boundary_type=BoundaryType.SYNCHRONOUS,
            trigger_reason=f"Call timeout for {operation_name}",
            action_taken="Call cancelled due to timeout",
            impact_level=FailureImpact.LIMITED,
            resource_usage={}
        )

        self.isolation_events.append(event)

    def _is_allowed_exception(self, boundary: ComponentBoundary, exception: Exception) -> bool:
        """Check if exception is in allowed list."""
        exception_name = type(exception).__name__
        return exception_name in boundary.allowed_exceptions

    def _is_critical_exception(self, boundary: ComponentBoundary, exception: Exception) -> bool:
        """Check if exception is critical."""
        exception_name = type(exception).__name__
        return exception_name in boundary.critical_exceptions

    async def _record_successful_operation(self, component_type: ComponentType, operation_name: str, duration: float):
        """Record successful operation."""
        logger.debug(f"Successful operation: {component_type.value}.{operation_name} ({duration:.3f}s)")

        # Record in degradation manager if available
        if self.degradation_manager:
            component_id = f"{component_type.value}-default"
            await self.degradation_manager.record_component_success(component_id)

    async def _record_operation_error(self, component_type: ComponentType, operation_name: str, error: Exception):
        """Record operation error."""
        logger.warning(f"Operation error: {component_type.value}.{operation_name} - {error}")

        # Track error for rate limiting
        error_key = f"{component_type.value}_errors"
        if error_key not in self.error_counts:
            self.error_counts[error_key] = []

        self.error_counts[error_key].append(datetime.now(timezone.utc))

        # Record in degradation manager if available
        if self.degradation_manager:
            component_id = f"{component_type.value}-default"
            await self.degradation_manager.record_component_failure(component_id)

    async def _trigger_component_isolation(self, component_type: ComponentType, reason: str):
        """Trigger component isolation due to critical error."""
        logger.error(f"Triggering component isolation: {component_type.value} - {reason}")

        self.isolation_count += 1

        # Create isolation event
        event = IsolationEvent(
            event_id=str(uuid.uuid4()),
            component_id=f"{component_type.value}-isolated",
            isolation_strategy=IsolationStrategy.FAILURE_CONTAINMENT,
            boundary_type=BoundaryType.SYNCHRONOUS,
            trigger_reason=reason,
            action_taken="Component isolated due to critical error",
            impact_level=FailureImpact.CASCADING,
            resource_usage={},
            recovery_triggered=True
        )

        self.isolation_events.append(event)

        # Trigger recovery if available
        if self.recovery_manager:
            try:
                await self.recovery_manager.trigger_component_recovery(
                    component_type,
                    reason=f"Component isolation: {reason}"
                )
            except Exception as e:
                logger.error(f"Failed to trigger recovery for {component_type.value}: {e}")

        # Send notification
        await self._send_isolation_notification(component_type, reason)

    async def set_component_resource_limits(
        self,
        component_type: ComponentType,
        resource_limits: ResourceLimits
    ):
        """
        Set resource limits for a component.

        Args:
            component_type: Component type
            resource_limits: New resource limits
        """
        self.resource_limits[component_type] = resource_limits

        # Update existing processes
        for _component_id, process_info in self.component_processes.items():
            if process_info.component_type == component_type:
                process_info.resource_limits = resource_limits

        logger.info(f"Updated resource limits for {component_type.value}")

    async def set_component_boundary(
        self,
        component_type: ComponentType,
        boundary: ComponentBoundary
    ):
        """
        Set component boundary configuration.

        Args:
            component_type: Component type
            boundary: New boundary configuration
        """
        self.component_boundaries[component_type] = boundary

        # Update semaphore
        self.semaphores[component_type] = asyncio.Semaphore(boundary.max_concurrent_calls)

        logger.info(f"Updated boundary configuration for {component_type.value}")

    async def get_isolation_status(self) -> dict[str, Any]:
        """
        Get comprehensive isolation status.

        Returns:
            Isolation status information
        """
        current_time = datetime.now(timezone.utc)
        uptime = time.time() - self.start_time

        # Process information
        process_info = []
        for component_id, process in self.component_processes.items():
            process_info.append({
                'component_id': component_id,
                'component_type': process.component_type.value,
                'process_id': process.process_id,
                'is_running': process.is_running,
                'cpu_usage_percent': process.cpu_usage_percent,
                'memory_usage_mb': process.memory_usage_mb,
                'file_descriptor_count': process.file_descriptor_count,
                'thread_count': process.thread_count,
                'resource_violations': process.resource_violations,
                'start_time': process.start_time.isoformat(),
                'last_health_check': process.last_health_check.isoformat() if process.last_health_check else None
            })

        # Active calls information
        active_calls_info = {}
        for component_id, call_set in self.active_calls.items():
            active_calls_info[component_id] = len(call_set)

        # Recent events (last hour)
        recent_cutoff = current_time - timedelta(hours=1)
        recent_events = [
            {
                'event_id': event.event_id,
                'component_id': event.component_id,
                'isolation_strategy': event.isolation_strategy.value,
                'trigger_reason': event.trigger_reason,
                'action_taken': event.action_taken,
                'impact_level': event.impact_level.value,
                'timestamp': event.timestamp.isoformat()
            }
            for event in self.isolation_events
            if event.timestamp > recent_cutoff
        ]

        return {
            'uptime_seconds': uptime,
            'isolation_count': self.isolation_count,
            'boundary_violations_count': self.boundary_violations_count,
            'resource_violations_count': self.resource_violations_count,
            'process_restarts_count': self.process_restarts_count,
            'active_processes': sum(1 for p in self.component_processes.values() if p.is_running),
            'total_processes': len(self.component_processes),
            'active_calls': active_calls_info,
            'processes': process_info,
            'recent_events': recent_events,
            'component_boundaries': {
                ct.value: {
                    'timeout_seconds': boundary.timeout_seconds,
                    'max_concurrent_calls': boundary.max_concurrent_calls,
                    'retry_count': boundary.retry_count,
                    'isolation_level': boundary.isolation_level.value
                }
                for ct, boundary in self.component_boundaries.items()
            },
            'resource_limits': {
                ct.value: {
                    'cpu_limit_percent': limits.cpu_limit_percent,
                    'memory_limit_mb': limits.memory_limit_mb,
                    'max_file_descriptors': limits.max_file_descriptors,
                    'max_threads': limits.max_threads
                }
                for ct, limits in self.resource_limits.items()
            }
        }

    async def force_component_isolation(
        self,
        component_type: ComponentType,
        reason: str = "Manual isolation"
    ):
        """
        Force isolation of a component.

        Args:
            component_type: Component to isolate
            reason: Reason for isolation
        """
        await self._trigger_component_isolation(component_type, reason)
        logger.warning(f"Component manually isolated: {component_type.value}")

    async def _terminate_violating_process(self, component_id: str, violations: list[str]):
        """Terminate a process that violates resource limits."""
        logger.warning(f"Terminating process {component_id} due to resource violations: {violations}")

        process_info = self.component_processes.get(component_id)
        if not process_info or not process_info.is_running:
            return

        try:
            # Try graceful termination first
            proc = psutil.Process(process_info.process_id)
            proc.terminate()

            # Wait for graceful shutdown
            try:
                proc.wait(timeout=10)
            except psutil.TimeoutExpired:
                # Force kill if graceful termination fails
                proc.kill()
                logger.warning(f"Force killed process {component_id}")

            process_info.is_running = False
            process_info.termination_reason = f"Resource violations: {', '.join(violations)}"

            # Create termination event
            event = IsolationEvent(
                event_id=str(uuid.uuid4()),
                component_id=component_id,
                isolation_strategy=IsolationStrategy.RESOURCE_ISOLATION,
                boundary_type=BoundaryType.SYNCHRONOUS,
                trigger_reason=f"Resource violations: {', '.join(violations)}",
                action_taken="Process terminated",
                impact_level=FailureImpact.LIMITED,
                resource_usage={},
                recovery_triggered=True
            )

            self.isolation_events.append(event)

        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Failed to terminate process {component_id}: {e}")

    async def _send_resource_violation_notification(self, component_id: str, violations: list[str]):
        """Send notification about resource violations."""
        title = f"Resource Violation: {component_id}"
        message = f"Component {component_id} has exceeded resource limits: {', '.join(violations)}"

        notification = UserNotification(
            timestamp=time.time(),
            level=NotificationLevel.WARNING,
            title=title,
            message=message,
            server_name="workspace-qdrant-mcp",
            troubleshooting_steps=[
                "Check component resource usage and optimization opportunities",
                "Consider increasing resource limits if appropriate",
                "Review component code for resource leaks",
                "Monitor component behavior for patterns"
            ],
            auto_recovery_attempted=False
        )

        # Send to all registered handlers
        for handler in list(self.notification_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception as e:
                logger.error(f"Error in resource violation notification handler: {e}")

    async def _send_isolation_notification(self, component_type: ComponentType, reason: str):
        """Send notification about component isolation."""
        title = f"Component Isolated: {component_type.value}"
        message = f"Component {component_type.value} has been isolated due to: {reason}"

        notification = UserNotification(
            timestamp=time.time(),
            level=NotificationLevel.ERROR,
            title=title,
            message=message,
            server_name="workspace-qdrant-mcp",
            troubleshooting_steps=[
                "Check component logs for detailed error information",
                "Verify component health and resource status",
                "Review recent component changes or updates",
                "Consider manual intervention if automatic recovery fails"
            ],
            auto_recovery_attempted=True
        )

        # Send to all registered handlers
        for handler in list(self.notification_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception as e:
                logger.error(f"Error in isolation notification handler: {e}")

    async def _handle_degradation_notification(self, notification: UserNotification):
        """Handle degradation notifications."""
        if notification.level in {NotificationLevel.CRITICAL, NotificationLevel.ERROR}:
            # Extract component information and trigger isolation if needed
            for component_type in ComponentType:
                if component_type.value.lower() in notification.message.lower():
                    await self._trigger_component_isolation(
                        component_type,
                        f"Degradation notification: {notification.title}"
                    )
                    break

    async def _handle_recovery_notification(self, notification: UserNotification):
        """Handle recovery notifications."""
        # Log recovery events for isolation tracking
        logger.info(f"Recovery notification received: {notification.title}")

    async def _handle_health_notification(self, notification: UserNotification):
        """Handle health notifications."""
        if notification.level == NotificationLevel.CRITICAL:
            # Trigger isolation for critical health issues
            for component_type in ComponentType:
                if component_type.value.lower() in notification.message.lower():
                    await self._trigger_component_isolation(
                        component_type,
                        f"Health alert: {notification.title}"
                    )
                    break

    def register_notification_handler(
        self,
        handler: Callable[[UserNotification], None]
    ):
        """Register a handler for isolation notifications."""
        self.notification_handlers.add(handler)

    async def shutdown(self):
        """Shutdown the component isolation manager."""
        logger.info("Shutting down component isolation manager")

        self.shutdown_event.set()

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        self.monitoring_tasks.clear()

        # Cancel all active timeouts
        for timeout_handle in self.call_timeouts.values():
            timeout_handle.cancel()
        self.call_timeouts.clear()

        # Clear active calls
        self.active_calls.clear()

        logger.info("Component isolation manager shutdown complete")


class ComponentBoundaryContext:
    """Context object for operations within component boundaries."""

    def __init__(
        self,
        component_type: ComponentType,
        component_id: str,
        call_id: str,
        operation_name: str,
        start_time: float,
        isolation_manager: ComponentIsolationManager
    ):
        self.component_type = component_type
        self.component_id = component_id
        self.call_id = call_id
        self.operation_name = operation_name
        self.start_time = start_time
        self.isolation_manager = isolation_manager

    def get_elapsed_time(self) -> float:
        """Get elapsed time since operation start."""
        return time.time() - self.start_time

    async def check_timeout(self, remaining_time_threshold: float = 5.0):
        """
        Check if operation is approaching timeout.

        Args:
            remaining_time_threshold: Threshold in seconds to warn about approaching timeout

        Raises:
            TimeoutWarning: If approaching timeout
        """
        boundary = self.isolation_manager.component_boundaries.get(self.component_type)
        if boundary:
            elapsed = self.get_elapsed_time()
            remaining = boundary.timeout_seconds - elapsed

            if remaining <= remaining_time_threshold:
                logger.warning(f"Operation {self.operation_name} approaching timeout: {remaining:.1f}s remaining")


class ComponentUnavailableError(Exception):
    """Raised when a component is not available for operations."""
    pass


class ResourceExhaustedError(Exception):
    """Raised when component resources are exhausted."""
    pass


class TimeoutWarning(Warning):
    """Warning for operations approaching timeout."""
    pass


# Global isolation manager instance
_isolation_manager: ComponentIsolationManager | None = None


async def get_isolation_manager(
    lifecycle_manager: ComponentLifecycleManager | None = None,
    coordinator: ComponentCoordinator | None = None,
    degradation_manager: DegradationManager | None = None,
    recovery_manager: RecoveryManager | None = None,
    health_monitor: LspHealthMonitor | None = None,
    config: dict[str, Any] | None = None
) -> ComponentIsolationManager:
    """Get or create global component isolation manager instance."""
    global _isolation_manager

    if _isolation_manager is None:
        _isolation_manager = ComponentIsolationManager(
            lifecycle_manager, coordinator, degradation_manager, recovery_manager, health_monitor, config
        )

        if not await _isolation_manager.initialize():
            raise RuntimeError("Failed to initialize component isolation manager")

    return _isolation_manager


async def shutdown_isolation_manager():
    """Shutdown global component isolation manager."""
    global _isolation_manager

    if _isolation_manager:
        await _isolation_manager.shutdown()
        _isolation_manager = None
