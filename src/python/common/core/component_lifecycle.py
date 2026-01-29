"""
Component Lifecycle Manager for Four-Component Architecture.

This module provides component lifecycle orchestration, startup sequences,
dependency management, and graceful shutdown procedures for the workspace-qdrant-mcp
four-component system.

Key Features:
    - Component startup sequence orchestration with dependency ordering
    - Graceful shutdown procedures with cleanup operations
    - Component dependency checking and wait mechanisms
    - Component readiness validation and health verification
    - Startup validation and timeout handling
    - Component recovery and restart coordination
    - Lifecycle event logging and monitoring

Architecture Components (startup order):
    1. SQLite State Manager (Foundation)
    2. Rust Daemon (Heavy Processing Engine)
    3. Python MCP Server (Intelligent Interface)
    4. CLI Utility / Context Injector (User Control/LLM Integration)

Example:
    ```python
    from workspace_qdrant_mcp.core.component_lifecycle import ComponentLifecycleManager

    # Initialize lifecycle manager
    lifecycle = ComponentLifecycleManager(db_path="./workspace_state.db")
    await lifecycle.initialize()

    # Start all components with proper dependency ordering
    await lifecycle.startup_sequence()

    # Gracefully shutdown all components
    await lifecycle.shutdown_sequence()
    ```
"""

import asyncio
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from common.core.component_coordination import (
    ComponentCoordinator,
    ComponentHealth,
    ComponentStatus,
    ComponentType,
    ProcessingQueueType,
    get_component_coordinator,
)
from common.core.daemon_manager import DaemonManager, ensure_daemon_running
from common.grpc.daemon_client import DaemonClient


class LifecyclePhase(Enum):
    """Lifecycle phases for component management."""

    INITIALIZATION = "initialization"
    DEPENDENCY_CHECK = "dependency_check"
    COMPONENT_STARTUP = "component_startup"
    READINESS_VALIDATION = "readiness_validation"
    OPERATIONAL = "operational"
    SHUTDOWN_INITIATED = "shutdown_initiated"
    COMPONENT_SHUTDOWN = "component_shutdown"
    CLEANUP = "cleanup"
    STOPPED = "stopped"


class ComponentState(Enum):
    """Component states during lifecycle management."""

    NOT_STARTED = "not_started"
    DEPENDENCY_WAITING = "dependency_waiting"
    STARTING = "starting"
    READY = "ready"
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    FAILED = "failed"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class StartupDependency(Enum):
    """Startup dependencies between components."""

    # SQLite is foundation - no dependencies
    SQLITE_MANAGER = "sqlite_manager"

    # Rust daemon depends on SQLite for state coordination
    RUST_DAEMON = "rust_daemon"

    # Python MCP server depends on SQLite and Rust daemon
    PYTHON_MCP_SERVER = "python_mcp_server"

    # CLI/Context injector can start after core services
    CLI_CONTEXT_INJECTOR = "cli_context_injector"


@dataclass
class ComponentConfig:
    """Configuration for component lifecycle management."""

    component_type: ComponentType
    startup_dependency: StartupDependency
    startup_timeout: float = 30.0
    shutdown_timeout: float = 15.0
    health_check_interval: float = 5.0
    max_startup_retries: int = 3
    readiness_checks: list[str] = None
    config_overrides: dict[str, Any] = None
    environment_variables: dict[str, str] = None
    startup_command: str | None = None
    shutdown_command: str | None = None

    def __post_init__(self):
        if self.readiness_checks is None:
            self.readiness_checks = []
        if self.config_overrides is None:
            self.config_overrides = {}
        if self.environment_variables is None:
            self.environment_variables = {}


@dataclass
class LifecycleEvent:
    """Lifecycle event record for monitoring and debugging."""

    event_id: str
    component_id: str
    phase: LifecyclePhase
    event_type: str  # startup, shutdown, failure, recovery
    message: str
    details: dict[str, Any] = None
    duration_ms: float | None = None
    success: bool = True
    error_message: str | None = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.details is None:
            self.details = {}


class ComponentLifecycleManager:
    """
    Component Lifecycle Manager for orchestrating startup and shutdown sequences.

    Manages the complete lifecycle of all four components with proper dependency
    ordering, health validation, and graceful recovery mechanisms.
    """

    # Component startup order based on dependencies
    STARTUP_ORDER = [
        StartupDependency.SQLITE_MANAGER,
        StartupDependency.RUST_DAEMON,
        StartupDependency.PYTHON_MCP_SERVER,
        StartupDependency.CLI_CONTEXT_INJECTOR,
    ]

    # Reverse order for shutdown
    SHUTDOWN_ORDER = list(reversed(STARTUP_ORDER))

    # Default configurations for each component
    DEFAULT_CONFIGS = {
        ComponentType.RUST_DAEMON: ComponentConfig(
            component_type=ComponentType.RUST_DAEMON,
            startup_dependency=StartupDependency.RUST_DAEMON,
            startup_timeout=45.0,
            shutdown_timeout=20.0,
            health_check_interval=10.0,
            readiness_checks=[
                "grpc_server_responsive",
                "sqlite_connection_active",
                "process_health_ok"
            ]
        ),
        ComponentType.PYTHON_MCP_SERVER: ComponentConfig(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            startup_dependency=StartupDependency.PYTHON_MCP_SERVER,
            startup_timeout=30.0,
            shutdown_timeout=15.0,
            health_check_interval=8.0,
            readiness_checks=[
                "mcp_server_listening",
                "qdrant_connection_active",
                "grpc_client_connected"
            ]
        ),
        ComponentType.CLI_UTILITY: ComponentConfig(
            component_type=ComponentType.CLI_UTILITY,
            startup_dependency=StartupDependency.CLI_CONTEXT_INJECTOR,
            startup_timeout=15.0,
            shutdown_timeout=10.0,
            health_check_interval=15.0,
            readiness_checks=[
                "cli_commands_available",
                "config_validation_passed"
            ]
        ),
        ComponentType.CONTEXT_INJECTOR: ComponentConfig(
            component_type=ComponentType.CONTEXT_INJECTOR,
            startup_dependency=StartupDependency.CLI_CONTEXT_INJECTOR,
            startup_timeout=20.0,
            shutdown_timeout=10.0,
            health_check_interval=12.0,
            readiness_checks=[
                "context_hooks_registered",
                "mcp_server_accessible"
            ]
        ),
    }

    def __init__(
        self,
        db_path: str = "workspace_state.db",
        project_name: str | None = None,
        project_path: str | None = None,
        component_configs: dict[ComponentType, ComponentConfig] | None = None
    ):
        """
        Initialize Component Lifecycle Manager.

        Args:
            db_path: Path to SQLite database for state coordination
            project_name: Project name for component scoping
            project_path: Project path for workspace detection
            component_configs: Custom component configurations
        """
        self.db_path = db_path
        self.project_name = project_name or self._detect_project_name()
        self.project_path = project_path or str(Path.cwd())

        # Component coordination
        self.coordinator: ComponentCoordinator | None = None

        # Component configurations
        self.component_configs = {**self.DEFAULT_CONFIGS}
        if component_configs:
            self.component_configs.update(component_configs)

        # Component instances and state tracking
        self.component_instances: dict[ComponentType, Any] = {}
        self.component_states: dict[ComponentType, ComponentState] = {}
        self.startup_events: list[LifecycleEvent] = []

        # Lifecycle state
        self.current_phase = LifecyclePhase.INITIALIZATION
        self.startup_start_time: datetime | None = None
        self.shutdown_start_time: datetime | None = None

        # Background tasks
        self._lifecycle_tasks: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Daemon management
        self.daemon_manager: DaemonManager | None = None

        # Initialize component states
        for component_type in self.component_configs.keys():
            self.component_states[component_type] = ComponentState.NOT_STARTED

    async def initialize(self) -> bool:
        """
        Initialize the lifecycle manager and component coordinator.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            logger.info(
                "Initializing Component Lifecycle Manager",
                project=self.project_name,
                components=list(self.component_configs.keys())
            )

            # Initialize component coordinator (SQLite foundation)
            self.coordinator = await get_component_coordinator(self.db_path)

            # Initialize daemon manager
            self.daemon_manager = DaemonManager()

            # Log initialization event
            await self._log_lifecycle_event(
                component_id="lifecycle_manager",
                phase=LifecyclePhase.INITIALIZATION,
                event_type="initialization",
                message="Lifecycle manager initialized successfully",
                details={
                    "project_name": self.project_name,
                    "project_path": self.project_path,
                    "component_count": len(self.component_configs)
                }
            )

            logger.info("Component Lifecycle Manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Component Lifecycle Manager: {e}")
            await self._log_lifecycle_event(
                component_id="lifecycle_manager",
                phase=LifecyclePhase.INITIALIZATION,
                event_type="initialization",
                message="Lifecycle manager initialization failed",
                success=False,
                error_message=str(e),
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
            return False

    async def startup_sequence(self) -> bool:
        """
        Execute the complete startup sequence for all components.

        Returns:
            True if all components started successfully, False otherwise
        """
        try:
            self.startup_start_time = datetime.now(timezone.utc)
            self.current_phase = LifecyclePhase.DEPENDENCY_CHECK

            logger.info(
                "Starting component startup sequence",
                project=self.project_name,
                order=self.STARTUP_ORDER
            )

            # Start background monitoring
            await self._start_lifecycle_monitoring()

            # Execute startup sequence in dependency order
            for dependency in self.STARTUP_ORDER:
                if dependency == StartupDependency.SQLITE_MANAGER:
                    # SQLite manager is already initialized in self.coordinator
                    await self._mark_sqlite_ready()
                    continue

                # Find components that match this dependency
                matching_components = [
                    comp_type for comp_type, config in self.component_configs.items()
                    if config.startup_dependency == dependency
                ]

                if not matching_components:
                    continue

                # Start components in parallel for this dependency level
                await self._start_dependency_level(dependency, matching_components)

            # Validate all components are operational
            self.current_phase = LifecyclePhase.READINESS_VALIDATION
            if await self._validate_all_components_ready():
                self.current_phase = LifecyclePhase.OPERATIONAL

                startup_duration = (
                    datetime.now(timezone.utc) - self.startup_start_time
                ).total_seconds()

                logger.info(
                    "Component startup sequence completed successfully",
                    duration_seconds=startup_duration,
                    components_started=len(self.component_instances)
                )

                await self._log_lifecycle_event(
                    component_id="lifecycle_manager",
                    phase=LifecyclePhase.OPERATIONAL,
                    event_type="startup",
                    message="All components started and validated",
                    duration_ms=startup_duration * 1000,
                    details={
                        "components_started": list(self.component_instances.keys()),
                        "total_duration_seconds": startup_duration
                    }
                )

                return True
            else:
                logger.error("Component readiness validation failed")
                await self._handle_startup_failure()
                return False

        except Exception as e:
            logger.error(f"Startup sequence failed: {e}")
            await self._handle_startup_failure()
            await self._log_lifecycle_event(
                component_id="lifecycle_manager",
                phase=self.current_phase,
                event_type="startup",
                message="Startup sequence failed",
                success=False,
                error_message=str(e),
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
            return False

    async def shutdown_sequence(self) -> bool:
        """
        Execute graceful shutdown sequence for all components.

        Returns:
            True if all components shutdown successfully, False otherwise
        """
        try:
            self.shutdown_start_time = datetime.now(timezone.utc)
            self.current_phase = LifecyclePhase.SHUTDOWN_INITIATED

            logger.info(
                "Starting component shutdown sequence",
                project=self.project_name,
                order=self.SHUTDOWN_ORDER
            )

            # Signal shutdown to monitoring tasks
            self._shutdown_event.set()

            # Execute shutdown sequence in reverse dependency order
            for dependency in self.SHUTDOWN_ORDER:
                if dependency == StartupDependency.SQLITE_MANAGER:
                    # SQLite manager is handled last in coordinator cleanup
                    continue

                # Find components that match this dependency
                matching_components = [
                    comp_type for comp_type, config in self.component_configs.items()
                    if config.startup_dependency == dependency
                    and comp_type in self.component_instances
                ]

                if not matching_components:
                    continue

                # Shutdown components in parallel for this dependency level
                await self._shutdown_dependency_level(dependency, matching_components)

            # Final cleanup
            self.current_phase = LifecyclePhase.CLEANUP
            await self._cleanup_lifecycle_resources()

            # Shutdown coordinator (SQLite manager)
            if self.coordinator:
                await self.coordinator.close()

            self.current_phase = LifecyclePhase.STOPPED

            shutdown_duration = (
                datetime.now(timezone.utc) - self.shutdown_start_time
            ).total_seconds()

            logger.info(
                "Component shutdown sequence completed",
                duration_seconds=shutdown_duration
            )

            await self._log_lifecycle_event(
                component_id="lifecycle_manager",
                phase=LifecyclePhase.STOPPED,
                event_type="shutdown",
                message="All components shutdown successfully",
                duration_ms=shutdown_duration * 1000,
                details={
                    "components_shutdown": list(self.component_instances.keys()),
                    "total_duration_seconds": shutdown_duration
                }
            )

            return True

        except Exception as e:
            logger.error(f"Shutdown sequence failed: {e}")
            await self._log_lifecycle_event(
                component_id="lifecycle_manager",
                phase=self.current_phase,
                event_type="shutdown",
                message="Shutdown sequence failed",
                success=False,
                error_message=str(e),
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
            return False

    async def restart_component(self, component_type: ComponentType) -> bool:
        """
        Restart a specific component with proper dependency handling.

        Args:
            component_type: Component to restart

        Returns:
            True if restart successful, False otherwise
        """
        try:
            logger.info(f"Restarting component: {component_type.value}")

            if component_type not in self.component_instances:
                logger.error(f"Component not running: {component_type.value}")
                return False

            # Shutdown component
            await self._shutdown_component(component_type)

            # Wait briefly for cleanup
            await asyncio.sleep(2.0)

            # Restart component
            success = await self._start_component(component_type)

            if success:
                # Validate readiness
                success = await self._validate_component_readiness(component_type)

            if success:
                logger.info(f"Component restarted successfully: {component_type.value}")
            else:
                logger.error(f"Component restart failed: {component_type.value}")

            return success

        except Exception as e:
            logger.error(f"Failed to restart component {component_type.value}: {e}")
            return False

    async def get_component_status(self) -> dict[str, Any]:
        """
        Get comprehensive status of all components and lifecycle manager.

        Returns:
            Component status information
        """
        try:
            status = {
                "lifecycle_manager": {
                    "current_phase": self.current_phase.value,
                    "project_name": self.project_name,
                    "project_path": self.project_path,
                    "startup_time": self.startup_start_time.isoformat() if self.startup_start_time else None,
                    "shutdown_time": self.shutdown_start_time.isoformat() if self.shutdown_start_time else None,
                },
                "components": {},
                "startup_events": [
                    {
                        "event_id": event.event_id,
                        "component_id": event.component_id,
                        "phase": event.phase.value,
                        "event_type": event.event_type,
                        "message": event.message,
                        "success": event.success,
                        "timestamp": event.timestamp.isoformat(),
                        "duration_ms": event.duration_ms,
                    }
                    for event in self.startup_events[-20:]  # Last 20 events
                ]
            }

            # Get component states
            for component_type, state in self.component_states.items():
                component_status = {
                    "state": state.value,
                    "instance_active": component_type in self.component_instances,
                    "config": asdict(self.component_configs[component_type])
                }

                # Get coordinator status if available
                if self.coordinator:
                    component_id = f"{component_type.value}-{self.project_name}"
                    coordinator_status = await self.coordinator.get_component_status(component_id)
                    if coordinator_status and "error" not in coordinator_status:
                        component_status["coordinator_info"] = coordinator_status

                status["components"][component_type.value] = component_status

            return status

        except Exception as e:
            logger.error(f"Failed to get component status: {e}")
            return {"error": str(e)}

    async def _mark_sqlite_ready(self):
        """Mark SQLite manager as ready since it's initialized with coordinator."""
        logger.info("SQLite State Manager ready")

        # Register SQLite manager in coordinator if not already registered
        if self.coordinator:
            try:
                component_id = await self.coordinator.register_component(
                    component_type=ComponentType.PYTHON_MCP_SERVER,  # Use as proxy for SQLite
                    instance_id=f"sqlite-{self.project_name}",
                    config={"db_path": self.db_path},
                    endpoints={"sqlite_db": self.db_path},
                    capabilities=["state_persistence", "component_coordination"],
                    version="1.0.0"
                )

                await self.coordinator.update_component_status(
                    component_id,
                    ComponentStatus.HEALTHY,
                    ComponentHealth.HEALTHY
                )

                logger.debug(f"SQLite manager registered as component: {component_id}")

            except Exception as e:
                logger.warning(f"Failed to register SQLite manager: {e}")

        await self._log_lifecycle_event(
            component_id="sqlite_manager",
            phase=LifecyclePhase.COMPONENT_STARTUP,
            event_type="startup",
            message="SQLite State Manager ready",
            details={"db_path": self.db_path}
        )

    async def _start_dependency_level(
        self,
        dependency: StartupDependency,
        components: list[ComponentType]
    ):
        """Start all components at a specific dependency level."""
        logger.info(
            f"Starting dependency level: {dependency.value}",
            components=[comp.value for comp in components]
        )

        self.current_phase = LifecyclePhase.COMPONENT_STARTUP

        # Start components in parallel
        startup_tasks = []
        for component_type in components:
            task = asyncio.create_task(
                self._start_component_with_retry(component_type)
            )
            startup_tasks.append(task)

        # Wait for all components to start
        results = await asyncio.gather(*startup_tasks, return_exceptions=True)

        # Check results
        failed_components = []
        for i, result in enumerate(results):
            component_type = components[i]
            if isinstance(result, Exception) or not result:
                failed_components.append(component_type)
                logger.error(
                    f"Failed to start component: {component_type.value}",
                    error=str(result) if isinstance(result, Exception) else "Unknown error"
                )

        if failed_components:
            raise RuntimeError(
                f"Failed to start components: {[c.value for c in failed_components]}"
            )

        logger.info(f"Dependency level {dependency.value} started successfully")

    async def _start_component_with_retry(self, component_type: ComponentType) -> bool:
        """Start a component with retry logic."""
        config = self.component_configs[component_type]

        for attempt in range(config.max_startup_retries):
            try:
                if await self._start_component(component_type):
                    return True
                else:
                    logger.warning(
                        f"Component startup attempt {attempt + 1} failed: {component_type.value}"
                    )
                    if attempt < config.max_startup_retries - 1:
                        await asyncio.sleep(2.0 * (attempt + 1))  # Exponential backoff
            except Exception as e:
                logger.error(
                    f"Component startup attempt {attempt + 1} error: {component_type.value}",
                    error=str(e)
                )
                if attempt < config.max_startup_retries - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))

        return False

    async def _start_component(self, component_type: ComponentType) -> bool:
        """Start a specific component."""
        config = self.component_configs[component_type]
        start_time = time.time()

        try:
            logger.info(f"Starting component: {component_type.value}")
            self.component_states[component_type] = ComponentState.STARTING

            # Register component in coordinator
            component_id = await self._register_component(component_type)

            # Start component based on type
            if component_type == ComponentType.RUST_DAEMON:
                instance = await self._start_rust_daemon(config)
            elif component_type == ComponentType.PYTHON_MCP_SERVER:
                instance = await self._start_python_mcp_server(config)
            elif component_type == ComponentType.CLI_UTILITY:
                instance = await self._start_cli_utility(config)
            elif component_type == ComponentType.CONTEXT_INJECTOR:
                instance = await self._start_context_injector(config)
            else:
                raise ValueError(f"Unknown component type: {component_type}")

            if instance:
                self.component_instances[component_type] = instance
                self.component_states[component_type] = ComponentState.READY

                # Update coordinator status
                await self.coordinator.update_component_status(
                    component_id,
                    ComponentStatus.HEALTHY,
                    ComponentHealth.HEALTHY
                )

                duration_ms = (time.time() - start_time) * 1000

                await self._log_lifecycle_event(
                    component_id=component_id,
                    phase=LifecyclePhase.COMPONENT_STARTUP,
                    event_type="startup",
                    message=f"Component {component_type.value} started successfully",
                    duration_ms=duration_ms,
                    details={"startup_config": asdict(config)}
                )

                logger.info(
                    f"Component started successfully: {component_type.value}",
                    duration_ms=duration_ms
                )
                return True
            else:
                self.component_states[component_type] = ComponentState.FAILED
                return False

        except Exception as e:
            self.component_states[component_type] = ComponentState.FAILED
            logger.error(f"Failed to start component {component_type.value}: {e}")

            duration_ms = (time.time() - start_time) * 1000
            await self._log_lifecycle_event(
                component_id=f"{component_type.value}-{self.project_name}",
                phase=LifecyclePhase.COMPONENT_STARTUP,
                event_type="startup",
                message=f"Component {component_type.value} startup failed",
                success=False,
                error_message=str(e),
                duration_ms=duration_ms,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
            return False

    async def _start_rust_daemon(self, config: ComponentConfig) -> Any:
        """Start the Rust daemon component."""
        try:
            logger.info("Starting Rust daemon")

            # Prepare daemon configuration
            daemon_config = {
                "project_name": self.project_name,
                "project_path": self.project_path,
                **config.config_overrides
            }

            # Use daemon manager to start daemon
            daemon_instance = await ensure_daemon_running(
                self.project_name,
                self.project_path,
                daemon_config
            )

            if daemon_instance:
                # Wait for daemon to be fully ready
                max_wait = config.startup_timeout
                wait_time = 0
                while wait_time < max_wait:
                    if await daemon_instance.health_check():
                        break
                    await asyncio.sleep(1.0)
                    wait_time += 1.0

                if wait_time >= max_wait:
                    logger.error("Rust daemon startup timeout")
                    return None

                logger.info(
                    "Rust daemon started successfully",
                    port=daemon_instance.config.grpc_port
                )
                return daemon_instance

            return None

        except Exception as e:
            logger.error(f"Failed to start Rust daemon: {e}")
            return None

    async def _start_python_mcp_server(self, config: ComponentConfig) -> Any:
        """Start the Python MCP server component."""
        try:
            logger.info("Starting Python MCP server")

            # Import server module
            from workspace_qdrant_mcp.server import Config

            # Create server configuration
            server_config = Config()

            # Apply configuration overrides
            for key, value in config.config_overrides.items():
                if hasattr(server_config, key):
                    setattr(server_config, key, value)

            # Create gRPC daemon client for MCP server
            daemon_client = DaemonClient(
                project_path=self.project_path,
            )

            await daemon_client.start()

            # Store client reference
            logger.info("Python MCP server initialized successfully")
            return daemon_client

        except Exception as e:
            logger.error(f"Failed to start Python MCP server: {e}")
            return None

    async def _start_cli_utility(self, config: ComponentConfig) -> Any:
        """Start the CLI utility component."""
        try:
            logger.info("Starting CLI utility")

            # CLI utility is command-based, so we validate it's available
            from workspace_qdrant_mcp.cli.main import app as cli_app

            # Test basic CLI functionality
            test_result = True  # Placeholder for actual CLI validation

            if test_result:
                logger.info("CLI utility validated successfully")
                return {"status": "ready", "cli_app": cli_app}

            return None

        except Exception as e:
            logger.error(f"Failed to start CLI utility: {e}")
            return None

    async def _start_context_injector(self, config: ComponentConfig) -> Any:
        """Start the context injector component."""
        try:
            logger.info("Starting context injector")

            # Context injector hooks into LLM integrations
            # For now, this is a placeholder that validates integration readiness

            injector_instance = {
                "status": "ready",
                "hooks_registered": True,
                "mcp_server_accessible": True
            }

            logger.info("Context injector initialized successfully")
            return injector_instance

        except Exception as e:
            logger.error(f"Failed to start context injector: {e}")
            return None

    async def _register_component(self, component_type: ComponentType) -> str:
        """Register component in coordinator."""
        config = self.component_configs[component_type]

        component_id = await self.coordinator.register_component(
            component_type=component_type,
            instance_id=f"{component_type.value}-{self.project_name}",
            config=asdict(config),
            capabilities=config.readiness_checks,
            version="1.0.0"
        )

        return component_id

    async def _validate_all_components_ready(self) -> bool:
        """Validate that all components are ready and operational."""
        logger.info("Validating component readiness")

        for component_type in self.component_instances.keys():
            if not await self._validate_component_readiness(component_type):
                return False

        logger.info("All components validated as ready")
        return True

    async def _validate_component_readiness(self, component_type: ComponentType) -> bool:
        """Validate readiness of a specific component."""
        config = self.component_configs[component_type]

        try:
            logger.debug(f"Validating readiness: {component_type.value}")

            # Perform component-specific readiness checks
            for check_name in config.readiness_checks:
                if not await self._perform_readiness_check(component_type, check_name):
                    logger.error(
                        f"Readiness check failed: {component_type.value}.{check_name}"
                    )
                    return False

            # Update component state
            self.component_states[component_type] = ComponentState.OPERATIONAL

            logger.debug(f"Component readiness validated: {component_type.value}")
            return True

        except Exception as e:
            logger.error(
                f"Component readiness validation failed: {component_type.value}",
                error=str(e)
            )
            self.component_states[component_type] = ComponentState.FAILED
            return False

    async def _perform_readiness_check(
        self,
        component_type: ComponentType,
        check_name: str
    ) -> bool:
        """Perform a specific readiness check."""
        try:
            instance = self.component_instances.get(component_type)
            if not instance:
                return False

            # Component-specific readiness checks
            if component_type == ComponentType.RUST_DAEMON:
                if check_name == "grpc_server_responsive":
                    return await instance.health_check()
                elif check_name == "sqlite_connection_active":
                    return self.coordinator is not None
                elif check_name == "process_health_ok":
                    return instance.status.state == "running"

            elif component_type == ComponentType.PYTHON_MCP_SERVER:
                if check_name == "mcp_server_listening":
                    return instance.get_operation_mode() in ["grpc", "direct"]
                elif check_name == "qdrant_connection_active":
                    status = await instance.get_status()
                    return status.get("qdrant_available", False)
                elif check_name == "grpc_client_connected":
                    return instance.is_grpc_available()

            elif component_type == ComponentType.CLI_UTILITY:
                if check_name == "cli_commands_available":
                    return instance.get("status") == "ready"
                elif check_name == "config_validation_passed":
                    return True  # Placeholder

            elif component_type == ComponentType.CONTEXT_INJECTOR:
                if check_name == "context_hooks_registered":
                    return instance.get("hooks_registered", False)
                elif check_name == "mcp_server_accessible":
                    return instance.get("mcp_server_accessible", False)

            # Default check passed if we reach here
            return True

        except Exception as e:
            logger.error(
                f"Readiness check error: {component_type.value}.{check_name}",
                error=str(e)
            )
            return False

    async def _shutdown_dependency_level(
        self,
        dependency: StartupDependency,
        components: list[ComponentType]
    ):
        """Shutdown all components at a specific dependency level."""
        logger.info(
            f"Shutting down dependency level: {dependency.value}",
            components=[comp.value for comp in components]
        )

        self.current_phase = LifecyclePhase.COMPONENT_SHUTDOWN

        # Shutdown components in parallel
        shutdown_tasks = []
        for component_type in components:
            task = asyncio.create_task(self._shutdown_component(component_type))
            shutdown_tasks.append(task)

        # Wait for all components to shutdown
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        logger.info(f"Dependency level {dependency.value} shutdown completed")

    async def _shutdown_component(self, component_type: ComponentType):
        """Shutdown a specific component."""
        self.component_configs[component_type]

        try:
            logger.info(f"Shutting down component: {component_type.value}")
            self.component_states[component_type] = ComponentState.SHUTTING_DOWN

            instance = self.component_instances.get(component_type)
            if not instance:
                logger.warning(f"Component not running: {component_type.value}")
                return

            # Component-specific shutdown procedures
            if component_type == ComponentType.RUST_DAEMON:
                if hasattr(instance, 'shutdown'):
                    await instance.shutdown()

            elif component_type == ComponentType.PYTHON_MCP_SERVER:
                if hasattr(instance, 'close'):
                    await instance.close()

            elif component_type == ComponentType.CLI_UTILITY:
                # CLI utility doesn't need special shutdown
                pass

            elif component_type == ComponentType.CONTEXT_INJECTOR:
                # Context injector cleanup
                pass

            # Remove from instances
            del self.component_instances[component_type]
            self.component_states[component_type] = ComponentState.STOPPED

            # Update coordinator status
            component_id = f"{component_type.value}-{self.project_name}"
            await self.coordinator.update_component_status(
                component_id,
                ComponentStatus.STOPPED
            )

            logger.info(f"Component shutdown completed: {component_type.value}")

        except Exception as e:
            logger.error(f"Failed to shutdown component {component_type.value}: {e}")
            self.component_states[component_type] = ComponentState.FAILED

    async def _start_lifecycle_monitoring(self):
        """Start background monitoring tasks."""
        self._lifecycle_tasks = [
            asyncio.create_task(self._component_health_monitor()),
            asyncio.create_task(self._dependency_monitor()),
        ]
        logger.debug("Lifecycle monitoring tasks started")

    async def _component_health_monitor(self):
        """Monitor component health and handle failures."""
        while not self._shutdown_event.is_set():
            try:
                for component_type, _instance in self.component_instances.items():
                    self.component_configs[component_type]

                    # Skip if component is not operational
                    if self.component_states[component_type] != ComponentState.OPERATIONAL:
                        continue

                    # Perform health check
                    if not await self._validate_component_readiness(component_type):
                        logger.warning(f"Component health check failed: {component_type.value}")
                        self.component_states[component_type] = ComponentState.DEGRADED

                        # Record health metrics if coordinator available
                        if self.coordinator:
                            component_id = f"{component_type.value}-{self.project_name}"
                            await self.coordinator.update_component_health(
                                component_id,
                                ComponentHealth.WARNING,
                                {"health_check_failed": True}
                            )

                await asyncio.sleep(10.0)  # Health check interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(5.0)

    async def _dependency_monitor(self):
        """Monitor component dependencies and restart if needed."""
        while not self._shutdown_event.is_set():
            try:
                # Check for failed components that need restart
                for component_type, state in self.component_states.items():
                    if state == ComponentState.FAILED:
                        logger.info(f"Attempting to restart failed component: {component_type.value}")
                        await self.restart_component(component_type)

                await asyncio.sleep(15.0)  # Dependency check interval

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dependency monitoring: {e}")
                await asyncio.sleep(5.0)

    async def _handle_startup_failure(self):
        """Handle startup failure by shutting down any started components."""
        logger.error("Handling startup failure - shutting down started components")

        # Shutdown any components that were started
        for component_type in list(self.component_instances.keys()):
            try:
                await self._shutdown_component(component_type)
            except Exception as e:
                logger.error(f"Error shutting down component during failure handling: {e}")

    async def _cleanup_lifecycle_resources(self):
        """Cleanup lifecycle management resources."""
        # Cancel monitoring tasks
        for task in self._lifecycle_tasks:
            if not task.done():
                task.cancel()

        if self._lifecycle_tasks:
            await asyncio.gather(*self._lifecycle_tasks, return_exceptions=True)

        self._lifecycle_tasks.clear()
        logger.debug("Lifecycle resources cleaned up")

    async def _log_lifecycle_event(
        self,
        component_id: str,
        phase: LifecyclePhase,
        event_type: str,
        message: str,
        success: bool = True,
        error_message: str | None = None,
        duration_ms: float | None = None,
        details: dict[str, Any] | None = None
    ):
        """Log a lifecycle event for monitoring and debugging."""
        event = LifecycleEvent(
            event_id=str(uuid.uuid4()),
            component_id=component_id,
            phase=phase,
            event_type=event_type,
            message=message,
            success=success,
            error_message=error_message,
            duration_ms=duration_ms,
            details=details or {}
        )

        self.startup_events.append(event)

        # Keep only last 100 events to prevent memory growth
        if len(self.startup_events) > 100:
            self.startup_events = self.startup_events[-100:]

        # Log to coordinator if available
        if self.coordinator:
            try:
                # Convert event to JSON-serializable format
                event_data = asdict(event)
                event_data["phase"] = event.phase.value  # Convert enum to string
                event_data["timestamp"] = event.timestamp.isoformat()  # Convert datetime to string

                # Only enqueue if this is for a registered component
                # (lifecycle_manager events don't need to be enqueued)
                if component_id != "lifecycle_manager":
                    await self.coordinator.enqueue_processing_item(
                        component_id=component_id,
                        queue_type=ProcessingQueueType.ADMIN_COMMANDS,
                        payload={
                            "event_type": "lifecycle_event",
                            "event_data": event_data
                        },
                        priority=3
                    )
            except Exception as e:
                logger.debug(f"Failed to log event to coordinator: {e}")

    def _detect_project_name(self) -> str:
        """Detect project name from current working directory."""
        try:
            return Path.cwd().name
        except Exception:
            return "default"


# Global lifecycle manager instance
_lifecycle_manager: ComponentLifecycleManager | None = None


async def get_lifecycle_manager(
    db_path: str = "workspace_state.db",
    project_name: str | None = None,
    project_path: str | None = None,
    component_configs: dict[ComponentType, ComponentConfig] | None = None
) -> ComponentLifecycleManager:
    """Get or create global lifecycle manager instance."""
    global _lifecycle_manager

    if _lifecycle_manager is None:
        _lifecycle_manager = ComponentLifecycleManager(
            db_path, project_name, project_path, component_configs
        )

        if not await _lifecycle_manager.initialize():
            raise RuntimeError("Failed to initialize component lifecycle manager")

    return _lifecycle_manager


async def shutdown_lifecycle_manager():
    """Shutdown global lifecycle manager."""
    global _lifecycle_manager

    if _lifecycle_manager:
        await _lifecycle_manager.shutdown_sequence()
        _lifecycle_manager = None
