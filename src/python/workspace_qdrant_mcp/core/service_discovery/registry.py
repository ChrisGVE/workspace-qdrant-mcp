"""
File-based Service Registry

This module implements a file-based service registry using JSON for storing
service information. It provides atomic operations for registration,
deregistration, and discovery with file locking for concurrent access.
"""

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union
import fcntl
from loguru import logger

from .exceptions import RegistryError

# logger imported from loguru


class ServiceStatus(Enum):
    """Service status enumeration"""
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"


@dataclass
class ServiceInfo:
    """Information about a registered service"""
    host: str
    port: int
    pid: int
    startup_time: str
    auth_token: Optional[str] = None
    health_endpoint: str = "/health"
    additional_ports: Dict[str, int] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.STARTING
    last_health_check: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(cls, host: str, port: int, **kwargs) -> 'ServiceInfo':
        """Create a new ServiceInfo instance"""
        return cls(
            host=host,
            port=port,
            pid=os.getpid(),
            startup_time=datetime.now(timezone.utc).isoformat(),
            **kwargs
        )

    def with_auth_token(self, token: str) -> 'ServiceInfo':
        """Set authentication token"""
        self.auth_token = token
        return self

    def with_health_endpoint(self, endpoint: str) -> 'ServiceInfo':
        """Set health endpoint"""
        self.health_endpoint = endpoint
        return self

    def with_additional_port(self, name: str, port: int) -> 'ServiceInfo':
        """Add additional port"""
        self.additional_ports[name] = port
        return self

    def with_metadata(self, key: str, value: str) -> 'ServiceInfo':
        """Add metadata"""
        self.metadata[key] = value
        return self

    @staticmethod
    def generate_auth_token() -> str:
        """Generate a secure authentication token"""
        return str(uuid.uuid4())


@dataclass
class RegistryFile:
    """Service registry file format"""
    version: str = "1.0.0"
    services: Dict[str, ServiceInfo] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ServiceRegistry:
    """File-based service registry implementation"""

    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        """Initialize service registry"""
        if registry_path is None:
            self.registry_path = self._default_registry_path()
        else:
            self.registry_path = Path(registry_path)

        # Ensure parent directory exists
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Service registry initialized at: {self.registry_path}")

    @staticmethod
    def _default_registry_path() -> Path:
        """Get default registry path (~/.workspace-qdrant/services.json)"""
        home_dir = Path.home()
        return home_dir / ".workspace-qdrant" / "services.json"

    async def register_service(self, service_name: str, service_info: ServiceInfo) -> None:
        """Register a service in the registry"""
        logger.debug(f"Registering service: {service_name} at {service_info.host}:{service_info.port}")

        # Validate process is running
        if not self._is_process_running(service_info.pid):
            raise RegistryError(f"Process {service_info.pid} is not running")

        registry = await self._load_or_create_registry()

        # Check if service already exists with different PID
        if service_name in registry.services:
            existing = registry.services[service_name]
            if existing.pid != service_info.pid:
                logger.warning(f"Service {service_name} already registered with different PID. Updating.")

        registry.services[service_name] = service_info
        registry.last_updated = datetime.now(timezone.utc).isoformat()

        await self._save_registry(registry)

        logger.info(f"Service {service_name} registered successfully")

    async def deregister_service(self, service_name: str) -> bool:
        """Deregister a service from the registry"""
        logger.debug(f"Deregistering service: {service_name}")

        registry = await self._load_or_create_registry()
        existed = registry.services.pop(service_name, None) is not None

        if existed:
            registry.last_updated = datetime.now(timezone.utc).isoformat()
            await self._save_registry(registry)
            logger.info(f"Service {service_name} deregistered successfully")
        else:
            logger.debug(f"Service {service_name} was not registered")

        return existed

    async def discover_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover a specific service by name"""
        registry = await self._load_registry()

        if registry and service_name in registry.services:
            service_info = registry.services[service_name]

            # Validate the service process is still running
            if self._is_process_running(service_info.pid):
                logger.debug(f"Service {service_name} discovered at {service_info.host}:{service_info.port}")
                return service_info
            else:
                logger.warning(f"Service {service_name} found but process {service_info.pid} is not running")
                # Clean up stale entry
                await self.deregister_service(service_name)
                return None
        else:
            logger.debug(f"Service {service_name} not found in registry")
            return None

    async def list_services(self) -> List[tuple[str, ServiceInfo]]:
        """List all registered services"""
        registry = await self._load_registry()
        if registry:
            services = list(registry.services.items())
            logger.debug(f"Listed {len(services)} registered services")
            return services
        return []

    async def update_service_status(self, service_name: str, status: ServiceStatus) -> bool:
        """Update service status"""
        registry = await self._load_or_create_registry()

        if service_name in registry.services:
            registry.services[service_name].status = status
            registry.services[service_name].last_health_check = datetime.now(timezone.utc).isoformat()
            registry.last_updated = datetime.now(timezone.utc).isoformat()

            await self._save_registry(registry)
            logger.debug(f"Updated status for service {service_name} to {status}")
            return True
        else:
            logger.debug(f"Service {service_name} not found for status update")
            return False

    async def cleanup_stale_entries(self) -> List[str]:
        """Clean up stale registry entries"""
        registry = await self._load_or_create_registry()
        removed_services = []

        for service_name, service_info in list(registry.services.items()):
            if not self._is_process_running(service_info.pid):
                logger.warning(f"Removing stale service {service_name} (PID {service_info.pid} not running)")
                registry.services.pop(service_name)
                removed_services.append(service_name)

        if removed_services:
            registry.last_updated = datetime.now(timezone.utc).isoformat()
            await self._save_registry(registry)
            logger.info(f"Cleaned up {len(removed_services)} stale service entries")

        return removed_services

    def exists(self) -> bool:
        """Check if registry file exists"""
        return self.registry_path.exists()

    async def _load_or_create_registry(self) -> RegistryFile:
        """Load the registry file or create a new one"""
        registry = await self._load_registry()
        return registry if registry else RegistryFile()

    async def _load_registry(self) -> Optional[RegistryFile]:
        """Load the registry file"""
        if not self.registry_path.exists():
            return None

        try:
            with open(self.registry_path, 'r') as f:
                # Use file locking for concurrent access
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Convert dict services to ServiceInfo objects
            services = {}
            for name, service_data in data.get("services", {}).items():
                if isinstance(service_data, dict):
                    # Convert status string back to enum
                    if "status" in service_data:
                        service_data["status"] = ServiceStatus(service_data["status"])
                    services[name] = ServiceInfo(**service_data)
                else:
                    services[name] = service_data

            return RegistryFile(
                version=data.get("version", "1.0.0"),
                services=services,
                last_updated=data.get("last_updated", datetime.now(timezone.utc).isoformat())
            )

        except (json.JSONDecodeError, KeyError, ValueError, OSError) as e:
            raise RegistryError(f"Invalid registry format: {e}")

    async def _save_registry(self, registry: RegistryFile) -> None:
        """Save the registry file atomically"""
        # Create temporary file for atomic write
        temp_path = self.registry_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'w') as f:
                # Use file locking for concurrent access
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    # Convert to dict for JSON serialization
                    data = {
                        "version": registry.version,
                        "services": {
                            name: {
                                **asdict(service_info),
                                "status": service_info.status.value
                            }
                            for name, service_info in registry.services.items()
                        },
                        "last_updated": registry.last_updated
                    }
                    json.dump(data, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Atomically replace the original file
            temp_path.replace(self.registry_path)

            logger.debug(f"Registry saved to {self.registry_path}")

        except OSError as e:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise RegistryError(f"Failed to save registry: {e}")

    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if a process is running by PID"""
        try:
            # Send signal 0 to check if process exists
            os.kill(pid, 0)
            return True
        except OSError:
            return False