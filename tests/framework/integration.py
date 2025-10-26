"""
Integration Testing Coordinator for Cross-Component Testing

This module provides sophisticated coordination of integration tests that span
multiple system components, including Python, Rust, and external services.
It manages component isolation, service orchestration, and environment setup.

Features:
- Cross-component test coordination (Python ↔ Rust ↔ Services)
- Service dependency management and orchestration
- Environment isolation and cleanup coordination
- Test fixture sharing across components
- Component lifecycle management
- Service health monitoring during tests
- Resource conflict resolution between components
- Integrated logging and debugging support
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
    contextmanager,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Optional,
    Union,
)

import psutil

import docker

from .discovery import ResourceRequirement, TestCategory, TestComplexity, TestMetadata
from .execution import ExecutionResult, ExecutionStatus


class ComponentType(Enum):
    """Types of system components."""
    PYTHON_SERVICE = auto()    # Python services and applications
    RUST_SERVICE = auto()      # Rust services and daemons
    DATABASE = auto()          # Database services (PostgreSQL, Redis, etc.)
    MESSAGE_QUEUE = auto()     # Message queues (RabbitMQ, Kafka, etc.)
    EXTERNAL_API = auto()      # External HTTP APIs
    DOCKER_CONTAINER = auto()  # Docker containers
    PROCESS = auto()          # System processes


class ComponentState(Enum):
    """Component lifecycle states."""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    FAILED = auto()
    UNKNOWN = auto()


class IsolationLevel(Enum):
    """Test isolation levels."""
    NONE = auto()             # No isolation
    PROCESS = auto()          # Process-level isolation
    CONTAINER = auto()        # Container-level isolation
    NETWORK = auto()          # Network namespace isolation
    FULL = auto()             # Complete system isolation


@dataclass
class ComponentConfig:
    """Configuration for a system component."""
    name: str
    component_type: ComponentType
    start_command: list[str] | None = None
    stop_command: list[str] | None = None
    health_check_url: str | None = None
    health_check_command: list[str] | None = None
    environment_vars: dict[str, str] = field(default_factory=dict)
    working_directory: Path | None = None
    startup_timeout: float = 30.0
    shutdown_timeout: float = 10.0
    depends_on: set[str] = field(default_factory=set)
    ports: list[int] = field(default_factory=list)
    volumes: list[tuple[str, str]] = field(default_factory=list)  # host:container
    networks: list[str] = field(default_factory=list)
    docker_image: str | None = None
    docker_command: list[str] | None = None


@dataclass
class ComponentInstance:
    """Runtime instance of a system component."""
    config: ComponentConfig
    state: ComponentState = ComponentState.STOPPED
    process: subprocess.Popen | None = None
    container: Any | None = None  # Docker container object
    pid: int | None = None
    start_time: float | None = None
    stop_time: float | None = None
    health_check_failures: int = 0
    logs: deque = field(default_factory=lambda: deque(maxlen=1000))


class ComponentController(ABC):
    """Abstract base class for component controllers."""

    @abstractmethod
    async def start(self, instance: ComponentInstance) -> bool:
        """Start the component instance."""
        pass

    @abstractmethod
    async def stop(self, instance: ComponentInstance) -> bool:
        """Stop the component instance."""
        pass

    @abstractmethod
    async def health_check(self, instance: ComponentInstance) -> bool:
        """Check if the component is healthy."""
        pass

    @abstractmethod
    def get_logs(self, instance: ComponentInstance, lines: int = 100) -> list[str]:
        """Get component logs."""
        pass


class ProcessController(ComponentController):
    """Controller for process-based components."""

    async def start(self, instance: ComponentInstance) -> bool:
        """Start a process-based component."""
        config = instance.config

        if not config.start_command:
            logging.error(f"No start command configured for {config.name}")
            return False

        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(config.environment_vars)

            # Start process
            instance.process = subprocess.Popen(
                config.start_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=config.working_directory,
                universal_newlines=True,
                bufsize=1
            )

            instance.pid = instance.process.pid
            instance.start_time = time.time()
            instance.state = ComponentState.STARTING

            # Wait for startup with timeout
            startup_deadline = time.time() + config.startup_timeout

            while time.time() < startup_deadline:
                if instance.process.poll() is not None:
                    # Process exited
                    instance.state = ComponentState.FAILED
                    return False

                # Check health if health check is configured
                if await self.health_check(instance):
                    instance.state = ComponentState.RUNNING
                    return True

                await asyncio.sleep(0.5)

            # Startup timeout
            instance.state = ComponentState.FAILED
            await self.stop(instance)
            return False

        except Exception as e:
            logging.error(f"Failed to start {config.name}: {e}")
            instance.state = ComponentState.FAILED
            return False

    async def stop(self, instance: ComponentInstance) -> bool:
        """Stop a process-based component."""
        if instance.state == ComponentState.STOPPED:
            return True

        instance.state = ComponentState.STOPPING

        try:
            if instance.process and instance.process.poll() is None:
                # Try graceful shutdown first
                if instance.config.stop_command:
                    try:
                        subprocess.run(
                            instance.config.stop_command,
                            timeout=instance.config.shutdown_timeout,
                            check=True
                        )
                    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                        pass

                # Try SIGTERM
                try:
                    instance.process.terminate()
                    instance.process.wait(timeout=5)
                except (subprocess.TimeoutExpired, OSError):
                    # Force kill
                    try:
                        instance.process.kill()
                        instance.process.wait(timeout=5)
                    except (subprocess.TimeoutExpired, OSError):
                        pass

            instance.state = ComponentState.STOPPED
            instance.stop_time = time.time()
            return True

        except Exception as e:
            logging.error(f"Error stopping {instance.config.name}: {e}")
            instance.state = ComponentState.FAILED
            return False

    async def health_check(self, instance: ComponentInstance) -> bool:
        """Check process health."""
        config = instance.config

        # Check if process is still running
        if instance.process and instance.process.poll() is not None:
            return False

        # Run health check command if configured
        if config.health_check_command:
            try:
                result = subprocess.run(
                    config.health_check_command,
                    timeout=5,
                    capture_output=True,
                    check=True
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                instance.health_check_failures += 1
                return False

        # Basic process existence check
        return instance.process and instance.process.poll() is None

    def get_logs(self, instance: ComponentInstance, lines: int = 100) -> list[str]:
        """Get process logs."""
        if instance.process and instance.process.stdout:
            try:
                # Read available output
                output = []
                while True:
                    line = instance.process.stdout.readline()
                    if not line:
                        break
                    output.append(line.strip())
                    instance.logs.append(line.strip())

                # Return last N lines
                return list(instance.logs)[-lines:]
            except Exception:
                pass

        return list(instance.logs)[-lines:]


class DockerController(ComponentController):
    """Controller for Docker container components."""

    def __init__(self):
        try:
            self.client = docker.from_env()
        except Exception as e:
            logging.warning(f"Docker not available: {e}")
            self.client = None

    async def start(self, instance: ComponentInstance) -> bool:
        """Start a Docker container component."""
        if not self.client:
            logging.error("Docker client not available")
            return False

        config = instance.config

        if not config.docker_image:
            logging.error(f"No docker image configured for {config.name}")
            return False

        try:
            # Prepare container configuration
            container_config = {
                'image': config.docker_image,
                'name': f"test_{config.name}_{int(time.time())}",
                'detach': True,
                'environment': config.environment_vars,
                'remove': True,  # Auto-remove when stopped
            }

            # Add ports
            if config.ports:
                container_config['ports'] = {f"{port}/tcp": port for port in config.ports}

            # Add volumes
            if config.volumes:
                container_config['volumes'] = {
                    host_path: {'bind': container_path, 'mode': 'rw'}
                    for host_path, container_path in config.volumes
                }

            # Add networks
            if config.networks:
                container_config['network'] = config.networks[0]  # Primary network

            # Add command
            if config.docker_command:
                container_config['command'] = config.docker_command

            # Start container
            instance.container = self.client.containers.run(**container_config)
            instance.start_time = time.time()
            instance.state = ComponentState.STARTING

            # Wait for container to be ready
            startup_deadline = time.time() + config.startup_timeout

            while time.time() < startup_deadline:
                instance.container.reload()
                if instance.container.status == 'exited':
                    instance.state = ComponentState.FAILED
                    return False

                if await self.health_check(instance):
                    instance.state = ComponentState.RUNNING
                    return True

                await asyncio.sleep(0.5)

            # Startup timeout
            instance.state = ComponentState.FAILED
            await self.stop(instance)
            return False

        except Exception as e:
            logging.error(f"Failed to start Docker container {config.name}: {e}")
            instance.state = ComponentState.FAILED
            return False

    async def stop(self, instance: ComponentInstance) -> bool:
        """Stop a Docker container component."""
        if instance.state == ComponentState.STOPPED:
            return True

        instance.state = ComponentState.STOPPING

        try:
            if instance.container:
                try:
                    instance.container.stop(timeout=instance.config.shutdown_timeout)
                except Exception:
                    # Force kill if graceful stop fails
                    try:
                        instance.container.kill()
                    except Exception:
                        pass

            instance.state = ComponentState.STOPPED
            instance.stop_time = time.time()
            return True

        except Exception as e:
            logging.error(f"Error stopping Docker container {instance.config.name}: {e}")
            instance.state = ComponentState.FAILED
            return False

    async def health_check(self, instance: ComponentInstance) -> bool:
        """Check Docker container health."""
        if not instance.container:
            return False

        try:
            instance.container.reload()

            # Check container status
            if instance.container.status != 'running':
                return False

            config = instance.config

            # Run health check command inside container
            if config.health_check_command:
                try:
                    exit_code, output = instance.container.exec_run(
                        config.health_check_command,
                        timeout=5
                    )
                    return exit_code == 0
                except Exception:
                    instance.health_check_failures += 1
                    return False

            # Basic container running check
            return instance.container.status == 'running'

        except Exception:
            return False

    def get_logs(self, instance: ComponentInstance, lines: int = 100) -> list[str]:
        """Get Docker container logs."""
        if not instance.container:
            return []

        try:
            logs = instance.container.logs(tail=lines).decode('utf-8')
            return logs.splitlines()
        except Exception:
            return []


class EnvironmentManager:
    """Manages test environment setup and teardown."""

    def __init__(self, isolation_level: IsolationLevel = IsolationLevel.PROCESS):
        self.isolation_level = isolation_level
        self.temp_directories: list[Path] = []
        self.environment_vars: dict[str, str] = {}
        self.cleanup_callbacks: list[Callable] = []

    @contextmanager
    def isolated_environment(self):
        """Create an isolated test environment."""
        original_env = os.environ.copy()

        try:
            # Set up isolation
            self._setup_isolation()

            # Apply environment variables
            os.environ.update(self.environment_vars)

            yield

        finally:
            # Restore original environment (preserve pytest internal variables)
            current_keys = set(os.environ.keys())
            original_keys = set(original_env.keys())

            # Remove keys added during test (except pytest internal variables)
            for key in current_keys - original_keys:
                if not key.startswith('PYTEST_'):
                    os.environ.pop(key, None)

            # Restore original values
            for key in original_keys:
                if key in original_env:
                    os.environ[key] = original_env[key]
                else:
                    os.environ.pop(key, None)

            # Clean up
            self._cleanup()

    def _setup_isolation(self):
        """Set up environment isolation based on isolation level."""
        if self.isolation_level in [IsolationLevel.PROCESS, IsolationLevel.FULL]:
            # Create temporary directories
            temp_dir = Path(tempfile.mkdtemp(prefix="test_env_"))
            self.temp_directories.append(temp_dir)

            # Set environment variables for isolation
            self.environment_vars.update({
                'TMPDIR': str(temp_dir),
                'HOME': str(temp_dir / "home"),
                'XDG_CONFIG_HOME': str(temp_dir / "config"),
                'XDG_DATA_HOME': str(temp_dir / "data"),
                'XDG_CACHE_HOME': str(temp_dir / "cache"),
            })

            # Create directory structure
            for env_var in ['HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME']:
                Path(self.environment_vars[env_var]).mkdir(parents=True, exist_ok=True)

    def _cleanup(self):
        """Clean up test environment."""
        # Run cleanup callbacks
        for callback in reversed(self.cleanup_callbacks):
            try:
                callback()
            except Exception as e:
                logging.warning(f"Cleanup callback failed: {e}")

        # Remove temporary directories
        for temp_dir in self.temp_directories:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logging.warning(f"Failed to remove temp directory {temp_dir}: {e}")

        # Reset state
        self.temp_directories.clear()
        self.environment_vars.clear()
        self.cleanup_callbacks.clear()

    def add_cleanup_callback(self, callback: Callable):
        """Add a cleanup callback."""
        self.cleanup_callbacks.append(callback)

    def set_environment_variable(self, key: str, value: str):
        """Set an environment variable for the test."""
        self.environment_vars[key] = value


class ServiceFixture:
    """Fixture for sharing services across tests."""

    def __init__(self, name: str, component_instance: ComponentInstance):
        self.name = name
        self.instance = component_instance
        self.ref_count = 0
        self.lock = threading.RLock()

    def acquire(self) -> ComponentInstance:
        """Acquire a reference to the service."""
        with self.lock:
            self.ref_count += 1
            return self.instance

    def release(self):
        """Release a reference to the service."""
        with self.lock:
            self.ref_count = max(0, self.ref_count - 1)

    def is_in_use(self) -> bool:
        """Check if the service is currently in use."""
        with self.lock:
            return self.ref_count > 0


class IntegrationTestCoordinator:
    """Coordinates integration tests across multiple components."""

    def __init__(self, isolation_level: IsolationLevel = IsolationLevel.PROCESS):
        self.components: dict[str, ComponentConfig] = {}
        self.instances: dict[str, ComponentInstance] = {}
        self.controllers: dict[ComponentType, ComponentController] = {
            ComponentType.PYTHON_SERVICE: ProcessController(),
            ComponentType.RUST_SERVICE: ProcessController(),
            ComponentType.DATABASE: ProcessController(),
            ComponentType.MESSAGE_QUEUE: ProcessController(),
            ComponentType.EXTERNAL_API: ProcessController(),
            ComponentType.DOCKER_CONTAINER: DockerController(),
            ComponentType.PROCESS: ProcessController(),
        }

        self.environment_manager = EnvironmentManager(isolation_level)
        self.service_fixtures: dict[str, ServiceFixture] = {}
        self.coordination_lock = asyncio.Lock()

    def register_component(self, config: ComponentConfig):
        """Register a component configuration."""
        self.components[config.name] = config

    def register_components(self, configs: list[ComponentConfig]):
        """Register multiple component configurations."""
        for config in configs:
            self.register_component(config)

    @asynccontextmanager
    async def managed_components(self, component_names: list[str]):
        """Context manager for managing component lifecycle."""
        async with self.coordination_lock:
            started_components = []

            try:
                # Start components in dependency order
                start_order = self._resolve_startup_order(component_names)

                for name in start_order:
                    if await self._start_component(name):
                        started_components.append(name)
                    else:
                        raise RuntimeError(f"Failed to start component: {name}")

                yield {name: self.instances[name] for name in started_components}

            finally:
                # Stop components in reverse order
                for name in reversed(started_components):
                    await self._stop_component(name)

    @asynccontextmanager
    async def test_environment(self,
                              component_names: list[str],
                              environment_vars: dict[str, str] | None = None):
        """Complete test environment with components and isolation."""
        # Set up environment variables
        if environment_vars:
            for key, value in environment_vars.items():
                self.environment_manager.set_environment_variable(key, value)

        with self.environment_manager.isolated_environment():
            async with self.managed_components(component_names) as components:
                yield components

    def get_service_fixture(self, component_name: str) -> ServiceFixture | None:
        """Get a shared service fixture."""
        if component_name in self.service_fixtures:
            return self.service_fixtures[component_name]
        return None

    def create_service_fixture(self, component_name: str) -> ServiceFixture:
        """Create a shared service fixture."""
        if component_name not in self.instances:
            raise ValueError(f"Component not running: {component_name}")

        fixture = ServiceFixture(component_name, self.instances[component_name])
        self.service_fixtures[component_name] = fixture
        return fixture

    async def _start_component(self, name: str) -> bool:
        """Start a single component."""
        if name not in self.components:
            logging.error(f"Component not registered: {name}")
            return False

        config = self.components[name]
        instance = ComponentInstance(config=config)

        controller = self.controllers.get(config.component_type)
        if not controller:
            logging.error(f"No controller for component type: {config.component_type}")
            return False

        success = await controller.start(instance)
        if success:
            self.instances[name] = instance
            logging.info(f"Started component: {name}")
        else:
            logging.error(f"Failed to start component: {name}")

        return success

    async def _stop_component(self, name: str) -> bool:
        """Stop a single component."""
        if name not in self.instances:
            return True

        instance = self.instances[name]
        controller = self.controllers.get(instance.config.component_type)

        if controller:
            success = await controller.stop(instance)
            if success:
                logging.info(f"Stopped component: {name}")
            else:
                logging.error(f"Failed to stop component: {name}")

        # Remove from instances regardless of success
        del self.instances[name]

        # Clean up fixture if exists
        if name in self.service_fixtures:
            del self.service_fixtures[name]

        return True

    def _resolve_startup_order(self, component_names: list[str]) -> list[str]:
        """Resolve component startup order based on dependencies."""
        # Simple topological sort
        visited = set()
        temp_visited = set()
        result = []

        def visit(name: str):
            if name in temp_visited:
                raise RuntimeError(f"Circular dependency detected involving: {name}")

            if name in visited:
                return

            temp_visited.add(name)

            # Visit dependencies first
            if name in self.components:
                for dep in self.components[name].depends_on:
                    if dep in component_names:
                        visit(dep)

            temp_visited.remove(name)
            visited.add(name)
            result.append(name)

        for name in component_names:
            visit(name)

        return result

    async def health_check_all(self) -> dict[str, bool]:
        """Perform health check on all running components."""
        results = {}

        for name, instance in self.instances.items():
            controller = self.controllers.get(instance.config.component_type)
            if controller:
                try:
                    results[name] = await controller.health_check(instance)
                except Exception as e:
                    logging.error(f"Health check failed for {name}: {e}")
                    results[name] = False
            else:
                results[name] = False

        return results

    def get_component_logs(self, component_name: str, lines: int = 100) -> list[str]:
        """Get logs from a specific component."""
        if component_name not in self.instances:
            return []

        instance = self.instances[component_name]
        controller = self.controllers.get(instance.config.component_type)

        if controller:
            return controller.get_logs(instance, lines)

        return []

    def get_all_logs(self) -> dict[str, list[str]]:
        """Get logs from all running components."""
        return {
            name: self.get_component_logs(name)
            for name in self.instances.keys()
        }

    async def execute_integration_test(self,
                                     test_metadata: TestMetadata,
                                     component_names: list[str],
                                     test_function: Callable,
                                     environment_vars: dict[str, str] | None = None) -> ExecutionResult:
        """Execute an integration test with full component coordination."""
        start_time = time.time()

        try:
            async with self.test_environment(component_names, environment_vars) as components:
                # Wait for all components to be healthy
                health_results = await self.health_check_all()
                unhealthy_components = [name for name, healthy in health_results.items() if not healthy]

                if unhealthy_components:
                    raise RuntimeError(f"Unhealthy components: {unhealthy_components}")

                # Execute the test
                if asyncio.iscoroutinefunction(test_function):
                    await test_function(components)
                else:
                    test_function(components)

                end_time = time.time()

                return ExecutionResult(
                    test_name=test_metadata.name,
                    status=ExecutionStatus.COMPLETED,
                    duration=end_time - start_time,
                    start_time=start_time,
                    end_time=end_time,
                    stdout=f"Integration test completed successfully with components: {component_names}",
                    stderr="",
                    return_code=0
                )

        except Exception as e:
            end_time = time.time()

            # Collect logs from all components for debugging
            logs = self.get_all_logs()
            error_info = f"Integration test failed: {str(e)}\n\nComponent logs:\n"
            for name, component_logs in logs.items():
                error_info += f"\n--- {name} ---\n"
                error_info += "\n".join(component_logs[-10:])  # Last 10 lines

            return ExecutionResult(
                test_name=test_metadata.name,
                status=ExecutionStatus.FAILED,
                duration=end_time - start_time,
                start_time=start_time,
                end_time=end_time,
                stdout="",
                stderr=error_info,
                return_code=1,
                error_message=str(e)
            )

    async def cleanup_all(self):
        """Clean up all resources."""
        # Stop all components
        component_names = list(self.instances.keys())
        for name in reversed(component_names):
            await self._stop_component(name)

        # Clear fixtures
        self.service_fixtures.clear()

        # Clean up environment
        self.environment_manager._cleanup()
