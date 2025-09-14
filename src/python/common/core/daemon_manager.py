"""
Daemon Process Management for Workspace Qdrant MCP.

This module provides comprehensive daemon lifecycle management for the Rust
ingestion engine, including:
- Automatic daemon startup on first MCP tool usage
- Health monitoring with periodic heartbeats (30s intervals)
- Graceful shutdown on MCP server exit
- Configuration synchronization between Python and Rust components
- Support for multiple daemon instances per project
- Logging integration to capture daemon stdout/stderr

The daemon manager ensures reliable operation of the Rust gRPC server
that provides high-performance document ingestion and processing capabilities.
"""

import asyncio
import atexit
import hashlib
import json
from common.logging.loguru_config import get_logger
import os
import platform
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..utils.project_detection import DaemonIdentifier, ProjectDetector
from .resource_manager import ResourceManager, ResourceLimits, get_resource_manager
from .project_config_manager import (
    ProjectConfigManager, 
    DaemonProjectConfig, 
    ConfigScope,
    get_project_config_manager
)

logger = get_logger(__name__)


@dataclass
class DaemonConfig:
    """Configuration for a daemon instance."""

    project_name: str
    project_path: str
    project_id: Optional[str] = None  # Unique project identifier for multi-instance support
    grpc_host: str = "127.0.0.1"
    grpc_port: int = 50051
    qdrant_url: str = "http://localhost:6333"
    log_level: str = "info"
    max_concurrent_jobs: int = 4
    health_check_interval: float = 30.0
    startup_timeout: float = 30.0
    shutdown_timeout: float = 10.0
    restart_on_failure: bool = True
    max_restart_attempts: int = 3
    restart_backoff_base: float = 2.0
    
    # Resource limits
    resource_limits: Optional[ResourceLimits] = None
    enable_resource_monitoring: bool = True
    
    @classmethod
    def from_project_config(cls, project_config: DaemonProjectConfig) -> "DaemonConfig":
        """Create DaemonConfig from DaemonProjectConfig."""
        return cls(
            project_name=project_config.project_name,
            project_path=project_config.project_path,
            project_id=project_config.project_id,
            grpc_host=project_config.grpc_host,
            grpc_port=project_config.grpc_port or 50051,
            qdrant_url=project_config.qdrant_url,
            log_level=project_config.log_level,
            max_concurrent_jobs=project_config.max_concurrent_jobs,
            health_check_interval=project_config.health_check_interval,
            startup_timeout=project_config.startup_timeout,
            shutdown_timeout=project_config.shutdown_timeout,
            restart_on_failure=project_config.restart_on_failure,
            max_restart_attempts=project_config.max_restart_attempts,
            resource_limits=ResourceLimits(
                max_memory_mb=project_config.max_memory_mb,
                max_cpu_percent=project_config.max_cpu_percent,
                max_open_files=project_config.max_open_files,
                processing_timeout=project_config.startup_timeout,
                connection_timeout=project_config.startup_timeout
            ),
            enable_resource_monitoring=project_config.enable_resource_monitoring
        )


@dataclass
class DaemonStatus:
    """Status information for a daemon instance."""

    pid: Optional[int] = None
    state: str = "stopped"  # stopped, starting, running, stopping, failed
    start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"  # healthy, unhealthy, unknown
    restart_count: int = 0
    last_error: Optional[str] = None
    grpc_available: bool = False


class PortManager:
    """
    Intelligent port allocation system for multiple daemon instances.
    
    This class manages port allocation across multiple daemon instances with:
    - Conflict detection and automatic port selection
    - Port range configuration and scanning
    - Registry persistence across daemon restarts
    - Health checks on allocated ports before assignment
    """
    
    # Class-level registry to track allocated ports
    _allocated_ports: Set[int] = set()
    _port_registry: Dict[int, Dict[str, Any]] = {}
    _registry_file: Optional[Path] = None
    
    def __init__(self, port_range: tuple[int, int] = (50051, 51051)):
        """Initialize port manager with configurable port range.
        
        Args:
            port_range: Tuple of (start_port, end_port) for allocation range
        """
        self.start_port, self.end_port = port_range
        
        # Initialize registry file in temp directory
        if not self._registry_file:
            temp_dir = Path(tempfile.gettempdir())
            self._registry_file = temp_dir / "wqm_port_registry.json"
            self._load_registry()
    
    def allocate_port(self, project_id: str, preferred_port: Optional[int] = None) -> int:
        """Allocate an available port for a project.
        
        Args:
            project_id: Unique project identifier
            preferred_port: Optional preferred port number
            
        Returns:
            Allocated port number
            
        Raises:
            RuntimeError: If no available ports found in range
        """
        # Check if we already have a port allocated for this project
        existing_port = self._get_project_port(project_id)
        if existing_port and self._is_port_available(existing_port):
            logger.debug("Reusing existing port", project_id=project_id, port=existing_port)
            return existing_port
        
        # Try preferred port first if specified
        if preferred_port and self._is_port_usable(preferred_port):
            self._register_port(preferred_port, project_id)
            return preferred_port
        
        # Scan for available ports in range
        for port in range(self.start_port, self.end_port + 1):
            if self._is_port_usable(port):
                self._register_port(port, project_id)
                return port
        
        # If no ports available, try to reclaim stale allocations
        self._cleanup_stale_allocations()
        
        # Try again after cleanup
        for port in range(self.start_port, self.end_port + 1):
            if self._is_port_usable(port):
                self._register_port(port, project_id)
                return port
        
        raise RuntimeError(
            f"No available ports in range {self.start_port}-{self.end_port} "
            f"for project {project_id}"
        )
    
    def release_port(self, port: int, project_id: str) -> bool:
        """Release a port allocation.
        
        Args:
            port: Port number to release
            project_id: Project identifier that allocated the port
            
        Returns:
            True if port was released, False if not allocated to this project
        """
        port_info = self._port_registry.get(port)
        if port_info and port_info.get('project_id') == project_id:
            self._allocated_ports.discard(port)
            del self._port_registry[port]
            self._save_registry()
            
            logger.debug("Released port", port=port, project_id=project_id)
            return True
        
        return False
    
    def get_allocated_ports(self) -> Dict[int, Dict[str, Any]]:
        """Get all currently allocated ports and their information.
        
        Returns:
            Dictionary mapping port numbers to allocation information
        """
        return self._port_registry.copy()
    
    def is_port_allocated(self, port: int) -> bool:
        """Check if a port is currently allocated.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is allocated, False otherwise
        """
        return port in self._allocated_ports
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available for binding.
        
        Args:
            port: Port number to check
            
        Returns:
            True if port is available, False if in use
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('127.0.0.1', port))
                return True
        except (socket.error, OSError):
            return False
    
    def _is_port_usable(self, port: int) -> bool:
        """Check if a port is usable (available and not allocated).
        
        Args:
            port: Port number to check
            
        Returns:
            True if port can be used, False otherwise
        """
        return (
            port not in self._allocated_ports and
            self._is_port_available(port) and
            self.start_port <= port <= self.end_port
        )
    
    def _get_project_port(self, project_id: str) -> Optional[int]:
        """Get the currently allocated port for a project.
        
        Args:
            project_id: Project identifier
            
        Returns:
            Port number if allocated, None otherwise
        """
        for port, info in self._port_registry.items():
            if info.get('project_id') == project_id:
                return port
        return None
    
    def _register_port(self, port: int, project_id: str) -> None:
        """Register a port allocation.
        
        Args:
            port: Port number to register
            project_id: Project identifier
        """
        self._allocated_ports.add(port)
        self._port_registry[port] = {
            'project_id': project_id,
            'allocated_at': datetime.now().isoformat(),
            'pid': os.getpid(),
            'host': '127.0.0.1',
        }
        self._save_registry()
        
        logger.debug("Registered port", port=port, project_id=project_id)
    
    def _cleanup_stale_allocations(self) -> None:
        """Clean up stale port allocations from dead processes."""
        stale_ports = []
        
        for port, info in self._port_registry.items():
            # Check if the process that allocated this port is still running
            pid = info.get('pid')
            if pid:
                try:
                    # Check if process exists (doesn't kill it)
                    os.kill(pid, 0)
                except (OSError, ProcessLookupError):
                    # Process doesn't exist, mark as stale
                    stale_ports.append(port)
            
            # Also check if port is actually in use
            if not self._is_port_available(port):
                # Port is in use by something else, but not necessarily stale
                continue
        
        # Remove stale allocations
        for port in stale_ports:
            project_id = self._port_registry[port].get('project_id', 'unknown')
            logger.info("Cleaning up stale port allocation", port=port, project_id=project_id)
            
            self._allocated_ports.discard(port)
            del self._port_registry[port]
        
        if stale_ports:
            self._save_registry()
    
    def _load_registry(self) -> None:
        """Load port registry from persistent storage."""
        if self._registry_file and self._registry_file.exists():
            try:
                with open(self._registry_file, 'r') as f:
                    data = json.load(f)
                    
                # Convert string keys back to integers
                for port_str, info in data.items():
                    port = int(port_str)
                    self._allocated_ports.add(port)
                    self._port_registry[port] = info
                    
                logger.debug("Loaded port registry", registry_file=str(self._registry_file))
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning("Failed to load port registry", error=str(e))
                # Start with clean registry if loading fails
                self._allocated_ports.clear()
                self._port_registry.clear()
    
    def _save_registry(self) -> None:
        """Save port registry to persistent storage."""
        if self._registry_file:
            try:
                # Convert integer keys to strings for JSON serialization
                data = {str(port): info for port, info in self._port_registry.items()}
                
                with open(self._registry_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
                logger.debug("Saved port registry", registry_file=str(self._registry_file))
                
            except (OSError, json.JSONEncodeError) as e:
                logger.warning("Failed to save port registry", error=str(e))
    
    @classmethod
    def get_instance(cls) -> "PortManager":
        """Get singleton instance of port manager."""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance


class DaemonInstance:
    """Manages a single daemon process instance."""

    def __init__(self, config: DaemonConfig):
        self.config = config
        self.status = DaemonStatus()
        self.process: Optional[asyncio.subprocess.Process] = None
        self.health_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.log_handlers: List[Callable[[str], None]] = []
        self.port_manager = PortManager.get_instance()
        self.resource_monitor = None  # Will be set during start()
        self.config_manager: Optional[ProjectConfigManager] = None
        self.project_config: Optional[DaemonProjectConfig] = None
        
        # Generate or use provided project identifier
        if not config.project_id:
            detector = ProjectDetector()
            identifier = detector.create_daemon_identifier(config.project_path)
            config.project_id = identifier.generate_identifier()

        # Create project-specific temp directory using project identifier
        temp_prefix = f"daemon_{config.project_id}_"
        self.temp_dir = Path(tempfile.mkdtemp(prefix=temp_prefix))
        self.config_file = self.temp_dir / "daemon_config.json"
        
        # Store project-specific PID file path
        self.pid_file = self.temp_dir / f"{config.project_id}.pid"

        logger.info(
            "Created daemon instance",
            project=config.project_name,
            project_id=config.project_id,
            port=config.grpc_port,
            temp_dir=str(self.temp_dir),
        )

    async def start(self) -> bool:
        """Start the daemon process."""
        if self.status.state in ["starting", "running"]:
            logger.warning(
                "Daemon already starting/running", project=self.config.project_name
            )
            return True

        logger.info(
            "Starting daemon",
            project=self.config.project_name,
            port=self.config.grpc_port,
        )

        try:
            self.status.state = "starting"
            self.status.start_time = datetime.now()
            self.status.last_error = None

            # Load project-specific configuration
            try:
                self.config_manager = get_project_config_manager(self.config.project_path)
                self.project_config = self.config_manager.load_config(self.config.project_id)
                
                # Start configuration watching for hot-reload
                if self.project_config.config_hot_reload:
                    self.config_manager.add_change_callback(self._on_config_change)
                    self.config_manager.start_watching()
                
                logger.info(
                    "Project configuration loaded",
                    project=self.config.project_name,
                    project_id=self.config.project_id,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load project configuration, using defaults",
                    project=self.config.project_name,
                    error=str(e),
                )

            # Write configuration file for Rust daemon
            await self._write_config_file()

            # Find and start the daemon binary
            daemon_path = await self._find_daemon_binary()
            if not daemon_path:
                raise RuntimeError("Daemon binary not found")

            # Prepare environment
            env = os.environ.copy()
            env.update(
                {
                    "RUST_LOG": self.config.log_level,
                    "GRPC_HOST": self.config.grpc_host,
                    "GRPC_PORT": str(self.config.grpc_port),
                    "CONFIG_FILE": str(self.config_file),
                }
            )

            # Start the daemon process with project-specific parameters
            daemon_args = [
                str(daemon_path),
                "--config", str(self.config_file),
                "--port", str(self.config.grpc_port),
                "--host", self.config.grpc_host,
                "--pid-file", str(self.pid_file),
            ]
            
            # Add project-id if available for multi-instance support
            if self.config.project_id:
                daemon_args.extend(["--project-id", self.config.project_id])
                
            self.process = await asyncio.create_subprocess_exec(
                *daemon_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.project_path,
            )

            self.status.pid = self.process.pid
            logger.info(
                "Daemon process started",
                project=self.config.project_name,
                pid=self.status.pid,
            )

            # Start output monitoring
            asyncio.create_task(self._monitor_output())

            # Wait for daemon to be ready
            if await self._wait_for_startup():
                self.status.state = "running"
                self.status.grpc_available = True

                # Start resource monitoring if enabled
                if self.config.enable_resource_monitoring and self.config.project_id:
                    try:
                        resource_manager = await get_resource_manager()
                        self.resource_monitor = await resource_manager.register_project(
                            self.config.project_id,
                            self.config.resource_limits
                        )
                        logger.info(
                            "Resource monitoring started",
                            project=self.config.project_name,
                            project_id=self.config.project_id,
                        )
                        
                        # Start performance monitoring
                        try:
                            from .performance_monitor import get_performance_monitor
                            performance_monitor = await get_performance_monitor(self.config.project_id)
                            logger.info(
                                "Performance monitoring started",
                                project=self.config.project_name,
                                project_id=self.config.project_id,
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to start performance monitoring",
                                project=self.config.project_name,
                                error=str(e),
                            )
                        
                    except Exception as e:
                        logger.warning(
                            "Failed to start resource monitoring",
                            project=self.config.project_name,
                            error=str(e),
                        )

                # Start health monitoring
                self.health_task = asyncio.create_task(self._health_monitor_loop())

                logger.info(
                    "Daemon started successfully",
                    project=self.config.project_name,
                    port=self.config.grpc_port,
                )
                return True
            else:
                await self.stop()
                self.status.state = "failed"
                self.status.last_error = "Daemon failed to start within timeout"
                return False

        except Exception as e:
            self.status.state = "failed"
            self.status.last_error = str(e)
            logger.error(
                "Failed to start daemon", project=self.config.project_name, error=str(e)
            )
            await self.stop()
            return False

    async def stop(self, timeout: Optional[float] = None) -> bool:
        """Stop the daemon process gracefully."""
        if self.status.state == "stopped":
            return True

        timeout = timeout or self.config.shutdown_timeout
        logger.info(
            "Stopping daemon", project=self.config.project_name, timeout=timeout
        )

        try:
            self.status.state = "stopping"
            self.shutdown_event.set()

            # Stop health monitoring
            if self.health_task and not self.health_task.done():
                self.health_task.cancel()
                try:
                    await asyncio.wait_for(self.health_task, timeout=1.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

            # Graceful shutdown of daemon process
            if self.process and self.process.returncode is None:
                try:
                    # Send SIGTERM for graceful shutdown
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=timeout)
                    logger.info(
                        "Daemon stopped gracefully", project=self.config.project_name
                    )
                except asyncio.TimeoutError:
                    # Force kill if graceful shutdown fails
                    logger.warning(
                        "Daemon graceful shutdown timed out, force killing",
                        project=self.config.project_name,
                    )
                    self.process.kill()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.error(
                            "Failed to kill daemon process",
                            project=self.config.project_name,
                        )

            # Cleanup resource monitoring
            if self.resource_monitor and self.config.project_id:
                try:
                    resource_manager = await get_resource_manager()
                    await resource_manager.unregister_project(self.config.project_id)
                    logger.info(
                        "Resource monitoring stopped",
                        project=self.config.project_name,
                        project_id=self.config.project_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to stop resource monitoring",
                        project=self.config.project_name,
                        error=str(e),
                    )
                
                # Stop performance monitoring
                try:
                    from .performance_monitor import stop_performance_monitor
                    await stop_performance_monitor(self.config.project_id)
                    logger.info(
                        "Performance monitoring stopped",
                        project=self.config.project_name,
                        project_id=self.config.project_id,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to stop performance monitoring",
                        project=self.config.project_name,
                        error=str(e),
                    )

            # Stop configuration watching
            if self.config_manager:
                try:
                    self.config_manager.stop_watching()
                    logger.info(
                        "Configuration watching stopped",
                        project=self.config.project_name,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to stop configuration watching",
                        project=self.config.project_name,
                        error=str(e),
                    )

            self.status.state = "stopped"
            self.status.pid = None
            self.status.grpc_available = False

            return True

        except Exception as e:
            logger.error(
                "Error stopping daemon", project=self.config.project_name, error=str(e)
            )
            return False
        finally:
            self._cleanup()

    async def restart(self) -> bool:
        """Restart the daemon process."""
        logger.info("Restarting daemon", project=self.config.project_name)

        # Stop gracefully first
        await self.stop()

        # Apply backoff delay
        if self.status.restart_count > 0:
            delay = min(
                self.config.restart_backoff_base**self.status.restart_count,
                30.0,  # Max 30 second delay
            )
            logger.info(
                "Applying restart backoff",
                project=self.config.project_name,
                delay=delay,
                restart_count=self.status.restart_count,
            )
            await asyncio.sleep(delay)

        # Attempt restart
        self.status.restart_count += 1
        return await self.start()

    async def health_check(self) -> bool:
        """Perform a health check on the daemon."""
        if self.status.state != "running" or not self.process:
            return False

        try:
            # Check if process is still running
            if self.process.returncode is not None:
                logger.warning(
                    "Daemon process has exited",
                    project=self.config.project_name,
                    return_code=self.process.returncode,
                )
                return False

            # Try to connect to gRPC endpoint
            from ..grpc.client import AsyncIngestClient
            from ..grpc.connection_manager import ConnectionConfig

            config = ConnectionConfig(
                host=self.config.grpc_host,
                port=self.config.grpc_port,
                connection_timeout=5.0,
            )

            client = AsyncIngestClient(connection_config=config)

            try:
                await client.start()
                is_healthy = await client.test_connection()
                await client.stop()

                self.status.last_health_check = datetime.now()
                self.status.health_status = "healthy" if is_healthy else "unhealthy"
                self.status.grpc_available = is_healthy

                # Check resource health if monitoring is enabled
                if is_healthy and self.resource_monitor:
                    try:
                        resource_manager = await get_resource_manager()
                        resource_usage = await resource_manager.get_project_usage(self.config.project_id)
                        
                        if resource_usage and not resource_usage.is_healthy:
                            logger.warning(
                                "Resource health check failed",
                                project=self.config.project_name,
                                warnings=resource_usage.warnings,
                                errors=resource_usage.errors,
                            )
                            # Consider resource issues as health degradation but not failure
                            is_healthy = len(resource_usage.errors) == 0
                    except Exception as e:
                        logger.debug(
                            "Resource health check error",
                            project=self.config.project_name,
                            error=str(e),
                        )

                return is_healthy

            except Exception as e:
                logger.debug(
                    "Health check gRPC connection failed",
                    project=self.config.project_name,
                    error=str(e),
                )
                self.status.health_status = "unhealthy"
                self.status.grpc_available = False
                return False

        except Exception as e:
            logger.warning(
                "Health check failed", project=self.config.project_name, error=str(e)
            )
            self.status.health_status = "unhealthy"
            return False

    def add_log_handler(self, handler: Callable[[str], None]):
        """Add a log handler for daemon output."""
        self.log_handlers.append(handler)

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information including resource usage."""
        status_dict = {
            "config": asdict(self.config),
            "status": asdict(self.status),
            "process_info": {
                "pid": self.status.pid,
                "running": self.process and self.process.returncode is None,
                "return_code": self.process.returncode if self.process else None,
            },
        }
        
        # Add resource information if monitoring is enabled
        if self.resource_monitor and self.config.project_id:
            try:
                resource_manager = await get_resource_manager()
                resource_usage = await resource_manager.get_project_usage(self.config.project_id)
                
                if resource_usage:
                    status_dict["resource_usage"] = asdict(resource_usage)
            except Exception as e:
                status_dict["resource_error"] = str(e)
        
        return status_dict

    def _on_config_change(self, new_config: DaemonProjectConfig) -> None:
        """Handle configuration changes from hot-reload."""
        logger.info(
            "Configuration change detected",
            project=self.config.project_name,
            project_id=self.config.project_id,
        )
        
        # Update stored project config
        self.project_config = new_config
        
        # Apply configuration changes that can be updated at runtime
        try:
            # Update resource limits if monitoring is active
            if self.resource_monitor:
                asyncio.create_task(self._update_resource_limits(new_config))
            
            # Update health check interval if changed
            if new_config.health_check_interval != self.config.health_check_interval:
                self.config.health_check_interval = new_config.health_check_interval
                logger.info(
                    "Updated health check interval",
                    new_interval=new_config.health_check_interval,
                )
            
            logger.info("Configuration updated successfully")
            
        except Exception as e:
            logger.error(
                "Error applying configuration changes",
                project=self.config.project_name,
                error=str(e),
            )

    async def _update_resource_limits(self, new_config: DaemonProjectConfig) -> None:
        """Update resource limits from new configuration."""
        try:
            resource_manager = await get_resource_manager()
            
            # Unregister old limits
            await resource_manager.unregister_project(self.config.project_id)
            
            # Register with new limits
            new_limits = ResourceLimits(
                max_memory_mb=new_config.max_memory_mb,
                max_cpu_percent=new_config.max_cpu_percent,
                max_open_files=new_config.max_open_files,
                processing_timeout=new_config.startup_timeout,
                connection_timeout=new_config.startup_timeout
            )
            
            self.resource_monitor = await resource_manager.register_project(
                self.config.project_id,
                new_limits
            )
            
            logger.info(
                "Resource limits updated",
                project=self.config.project_name,
                memory_mb=new_config.max_memory_mb,
                cpu_percent=new_config.max_cpu_percent,
            )
            
        except Exception as e:
            logger.error(
                "Failed to update resource limits",
                project=self.config.project_name,
                error=str(e),
            )

    async def _write_config_file(self):
        """Write configuration file for the Rust daemon with project-specific settings."""
        # Create project-specific log file path
        log_file_path = self.temp_dir / f"{self.config.project_id}.log"
        
        # Use project config if available, otherwise use daemon config
        config_source = self.project_config if self.project_config else self.config
        
        config_data = {
            "project_name": config_source.project_name,
            "project_path": config_source.project_path,
            "project_id": config_source.project_id,
            "grpc": {
                "host": getattr(config_source, 'grpc_host', self.config.grpc_host),
                "port": getattr(config_source, 'grpc_port', self.config.grpc_port)
            },
            "qdrant": {
                "url": getattr(config_source, 'qdrant_url', self.config.qdrant_url),
                "api_key": getattr(config_source, 'qdrant_api_key', None)
            },
            "processing": {
                "max_concurrent_jobs": getattr(config_source, 'max_concurrent_jobs', self.config.max_concurrent_jobs)
            },
            "logging": {
                "level": getattr(config_source, 'log_level', self.config.log_level),
                "file": str(log_file_path),
                "project_scoped": True,
            },
            "daemon": {
                "pid_file": str(self.pid_file),
                "temp_directory": str(self.temp_dir),
            },
        }
        
        # Add project-specific settings if available
        if self.project_config:
            config_data.update({
                "ingestion": {
                    "default_collection": self.project_config.default_collection,
                    "auto_create_collections": self.project_config.auto_create_collections,
                    "chunk_size": self.project_config.chunk_size,
                    "chunk_overlap": self.project_config.chunk_overlap,
                },
                "watching": {
                    "enabled": self.project_config.enable_file_watching,
                    "patterns": self.project_config.watch_patterns,
                    "ignore_patterns": self.project_config.ignore_patterns,
                },
                "resource_limits": {
                    "max_memory_mb": self.project_config.max_memory_mb,
                    "max_cpu_percent": self.project_config.max_cpu_percent,
                    "max_open_files": self.project_config.max_open_files,
                },
                "monitoring": {
                    "enabled": self.project_config.enable_resource_monitoring,
                    "metrics_collection": self.project_config.enable_metrics_collection,
                },
                "custom_settings": self.project_config.custom_settings,
            })

        with open(self.config_file, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.debug(
            "Wrote daemon config file",
            project=self.config.project_name,
            config_file=str(self.config_file),
        )

    async def _find_daemon_binary(self) -> Optional[Path]:
        """Find the daemon binary, building if necessary."""
        # Look for pre-built binary first
        project_root = Path(self.config.project_path).parent
        rust_engine_path = project_root / "rust-engine"

        # Check for built binary in target directory
        target_dirs = [
            rust_engine_path / "target" / "release",
            rust_engine_path / "target" / "debug",
        ]

        binary_name = "memexd"
        if platform.system() == "Windows":
            binary_name += ".exe"

        for target_dir in target_dirs:
            binary_path = target_dir / binary_name
            if binary_path.exists():
                logger.info("Found daemon binary", path=str(binary_path))
                return binary_path

        # Try to build if source exists
        if rust_engine_path.exists() and (rust_engine_path / "Cargo.toml").exists():
            logger.info("Building daemon binary", rust_path=str(rust_engine_path))

            try:
                # Use cargo to build the gRPC service
                result = await asyncio.create_subprocess_exec(
                    "cargo",
                    "build",
                    "--release",
                    "--bin",
                    "memexd",
                    cwd=str(rust_engine_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await result.communicate()

                if result.returncode == 0:
                    binary_path = rust_engine_path / "target" / "release" / binary_name
                    if binary_path.exists():
                        logger.info(
                            "Successfully built daemon binary", path=str(binary_path)
                        )
                        return binary_path
                else:
                    logger.error(
                        "Failed to build daemon binary", stderr=stderr.decode()
                    )

            except Exception as e:
                logger.error("Error building daemon binary", error=str(e))

        logger.error("Daemon binary not found and could not be built")
        return None

    async def _wait_for_startup(self) -> bool:
        """Wait for daemon to be ready for connections."""
        start_time = time.time()

        while time.time() - start_time < self.config.startup_timeout:
            if self.process and self.process.returncode is not None:
                logger.error(
                    "Daemon process exited during startup",
                    return_code=self.process.returncode,
                )
                return False

            # Try health check
            if await self.health_check():
                return True

            await asyncio.sleep(1.0)

        logger.error(
            "Daemon startup timeout exceeded", timeout=self.config.startup_timeout
        )
        return False

    async def _health_monitor_loop(self):
        """Background health monitoring loop."""
        logger.info(
            "Starting health monitor",
            project=self.config.project_name,
            interval=self.config.health_check_interval,
        )

        try:
            while not self.shutdown_event.is_set():
                try:
                    if not await self.health_check():
                        if self.config.restart_on_failure:
                            if (
                                self.status.restart_count
                                < self.config.max_restart_attempts
                            ):
                                logger.warning(
                                    "Health check failed, restarting daemon",
                                    project=self.config.project_name,
                                    restart_count=self.status.restart_count,
                                )
                                if not await self.restart():
                                    logger.error(
                                        "Failed to restart daemon after health check failure",
                                        project=self.config.project_name,
                                    )
                                    break
                            else:
                                logger.error(
                                    "Max restart attempts exceeded",
                                    project=self.config.project_name,
                                    max_attempts=self.config.max_restart_attempts,
                                )
                                self.status.state = "failed"
                                break

                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=self.config.health_check_interval,
                    )

                except asyncio.TimeoutError:
                    # Normal timeout, continue monitoring
                    continue
                except Exception as e:
                    logger.error(
                        "Error in health monitor loop",
                        project=self.config.project_name,
                        error=str(e),
                    )
                    await asyncio.sleep(5.0)  # Brief pause before retrying

        except asyncio.CancelledError:
            logger.info("Health monitor cancelled", project=self.config.project_name)
        except Exception as e:
            logger.error(
                "Health monitor loop failed",
                project=self.config.project_name,
                error=str(e),
            )

    async def _monitor_output(self):
        """Monitor daemon stdout and stderr."""
        if not self.process:
            return

        async def read_stream(stream, stream_name):
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break

                    line_str = line.decode().rstrip()
                    if line_str:
                        # Log daemon output
                        log_msg = (
                            f"[{self.config.project_name}:{stream_name}] {line_str}"
                        )
                        logger.info(log_msg)

                        # Notify handlers
                        for handler in self.log_handlers:
                            try:
                                handler(line_str)
                            except Exception as e:
                                logger.warning("Log handler failed", error=str(e))

            except Exception as e:
                logger.error(
                    "Error monitoring daemon output", stream=stream_name, error=str(e)
                )

        # Start monitoring both stdout and stderr
        if self.process.stdout:
            asyncio.create_task(read_stream(self.process.stdout, "stdout"))
        if self.process.stderr:
            asyncio.create_task(read_stream(self.process.stderr, "stderr"))

    def _cleanup(self):
        """Clean up temporary resources."""
        try:
            # Release allocated port
            if self.config.project_id and hasattr(self, 'port_manager'):
                self.port_manager.release_port(self.config.grpc_port, self.config.project_id)
            
            if self.temp_dir.exists():
                shutil.rmtree(str(self.temp_dir))
                logger.debug("Cleaned up temp directory", path=str(self.temp_dir))
        except Exception as e:
            logger.warning(
                "Failed to cleanup temp directory",
                path=str(self.temp_dir),
                error=str(e),
            )


class DaemonManager:
    """Manages multiple daemon instances for different projects."""

    _instance: Optional["DaemonManager"] = None
    _lock = asyncio.Lock()

    def __init__(self):
        self.daemons: Dict[str, DaemonInstance] = {}
        self.shutdown_handlers: List[Callable[[], None]] = []
        self._setup_signal_handlers()
        atexit.register(self._sync_shutdown)

    @classmethod
    async def get_instance(cls) -> "DaemonManager":
        """Get singleton instance of daemon manager."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info("Received shutdown signal", signal=signum)
            asyncio.create_task(self.shutdown_all())

        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, signal_handler)
        if hasattr(signal, "SIGINT"):
            signal.signal(signal.SIGINT, signal_handler)

    def _sync_shutdown(self):
        """Synchronous shutdown for atexit handler."""
        logger.info("Performing synchronous daemon shutdown")
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create task for cleanup
                asyncio.create_task(self.shutdown_all())
            else:
                # Run cleanup directly
                asyncio.run(self.shutdown_all())
        except Exception as e:
            logger.error("Error in synchronous shutdown", error=str(e))

    async def get_or_create_daemon(
        self,
        project_name: str,
        project_path: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> DaemonInstance:
        """Get existing daemon or create a new one for the project."""
        daemon_key = self._get_daemon_key(project_name, project_path)

        if daemon_key not in self.daemons:
            logger.info(
                "Creating new daemon instance",
                project=project_name,
                daemon_key=daemon_key,
            )

            # Create configuration with project identifier
            config = DaemonConfig(
                project_name=project_name,
                project_path=project_path,
                project_id=daemon_key,  # Use the daemon key as project_id
                grpc_port=self._get_available_port(daemon_key, project_path),
            )

            # Apply any overrides
            if config_overrides:
                for key, value in config_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

            self.daemons[daemon_key] = DaemonInstance(config)

        return self.daemons[daemon_key]

    async def start_daemon(
        self,
        project_name: str,
        project_path: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Start a daemon for the specified project."""
        daemon = await self.get_or_create_daemon(
            project_name, project_path, config_overrides
        )

        if daemon.status.state == "running":
            logger.info("Daemon already running", project=project_name)
            return True

        logger.info("Starting daemon for project", project=project_name)
        return await daemon.start()

    async def stop_daemon(self, project_name: str, project_path: str) -> bool:
        """Stop a daemon for the specified project."""
        daemon_key = self._get_daemon_key(project_name, project_path)

        if daemon_key not in self.daemons:
            logger.info("Daemon not found", daemon_key=daemon_key)
            return True

        daemon = self.daemons[daemon_key]
        result = await daemon.stop()

        # Remove from active daemons
        del self.daemons[daemon_key]

        return result

    async def get_daemon_status(
        self, project_name: str, project_path: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of a daemon."""
        daemon_key = self._get_daemon_key(project_name, project_path)

        if daemon_key not in self.daemons:
            return None

        return await self.daemons[daemon_key].get_status()

    async def list_daemons(self) -> Dict[str, Dict[str, Any]]:
        """List all active daemons with resource information."""
        results = {}
        for key, daemon in self.daemons.items():
            results[key] = await daemon.get_status()
        return results

    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all daemons."""
        results = {}

        for daemon_key, daemon in self.daemons.items():
            try:
                results[daemon_key] = await daemon.health_check()
            except Exception as e:
                logger.error("Health check failed", daemon_key=daemon_key, error=str(e))
                results[daemon_key] = False

        return results

    async def shutdown_all(self):
        """Shutdown all daemon instances."""
        logger.info("Shutting down all daemons", count=len(self.daemons))

        # Run shutdown handlers
        for handler in self.shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error("Shutdown handler failed", error=str(e))

        # Stop all daemons concurrently
        shutdown_tasks = [daemon.stop() for daemon in self.daemons.values()]

        if shutdown_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*shutdown_tasks, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Daemon shutdown timeout exceeded")

        # Cleanup global resource manager
        try:
            resource_manager = await get_resource_manager()
            await resource_manager.cleanup_all()
            logger.info("Resource manager cleaned up")
        except Exception as e:
            logger.error("Error cleaning up resource manager", error=str(e))
        
        # Cleanup performance monitoring
        try:
            from .performance_monitor import cleanup_all_performance_monitors
            await cleanup_all_performance_monitors()
            logger.info("Performance monitoring cleaned up")
        except Exception as e:
            logger.error("Error cleaning up performance monitoring", error=str(e))

        self.daemons.clear()
        logger.info("All daemons shut down")

    def add_shutdown_handler(self, handler: Callable[[], None]):
        """Add a shutdown handler."""
        self.shutdown_handlers.append(handler)

    async def get_system_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive system resource status across all daemons."""
        try:
            resource_manager = await get_resource_manager()
            return await resource_manager.get_system_status()
        except Exception as e:
            logger.error("Failed to get system resource status", error=str(e))
            return {
                "error": str(e),
                "total_projects": len(self.daemons),
                "daemons": list(self.daemons.keys())
            }

    def _get_daemon_key(self, project_name: str, project_path: str) -> str:
        """Generate a unique key for daemon identification using enhanced system."""
        # Use the new DaemonIdentifier for consistent, collision-resistant identification
        detector = ProjectDetector()
        identifier = detector.create_daemon_identifier(project_path)
        return identifier.generate_identifier()

    def _get_available_port(self, project_id: str, project_path: str = ".", base_port: int = 50051) -> int:
        """Find an available port for a new daemon using intelligent allocation."""
        
        # Use project ID hash to get preferred port
        id_hash = hashlib.md5(project_id.encode()).hexdigest()
        port_offset = int(id_hash[:4], 16) % 1000  # 0-999 offset
        preferred_port = base_port + port_offset
        
        # Use PortManager for intelligent allocation
        port_manager = PortManager.get_instance()
        try:
            return port_manager.allocate_port(project_id, preferred_port)
        except RuntimeError as e:
            logger.error("Failed to allocate port", project_id=project_id, error=str(e))
            # Fallback to original simple method as last resort
            return preferred_port


# Module-level convenience functions
_daemon_manager: Optional[DaemonManager] = None


async def get_daemon_manager() -> DaemonManager:
    """Get the global daemon manager instance."""
    global _daemon_manager
    if _daemon_manager is None:
        _daemon_manager = await DaemonManager.get_instance()
    return _daemon_manager


async def ensure_daemon_running(
    project_name: str,
    project_path: str,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> DaemonInstance:
    """Ensure a daemon is running for the specified project."""
    manager = await get_daemon_manager()
    daemon = await manager.get_or_create_daemon(
        project_name, project_path, config_overrides
    )

    if daemon.status.state != "running":
        success = await daemon.start()
        if not success:
            raise RuntimeError(f"Failed to start daemon for project {project_name}")

    return daemon


async def get_daemon_for_project(
    project_name: str, project_path: str
) -> Optional[DaemonInstance]:
    """Get daemon instance for a project, if it exists."""
    manager = await get_daemon_manager()
    daemon_key = manager._get_daemon_key(project_name, project_path)
    return manager.daemons.get(daemon_key)


async def shutdown_all_daemons():
    """Shutdown all daemon instances."""
    if _daemon_manager:
        await _daemon_manager.shutdown_all()


# Initialize daemon manager on module import
async def _initialize_daemon_manager():
    """Initialize the daemon manager."""
    await get_daemon_manager()


# Auto-initialize on import (run in background)
if hasattr(asyncio, "_get_running_loop") and asyncio._get_running_loop() is not None:
    asyncio.create_task(_initialize_daemon_manager())
