"""
Python Service Discovery Client for Multi-Instance Daemon Communication.

This module provides client-side service discovery functionality that integrates
with the Rust service discovery system to automatically locate and connect to
the correct daemon instance for each project context.
"""

import asyncio
import json
from loguru import logger
import socket
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from contextlib import asynccontextmanager

from common.utils.project_detection import ProjectDetector
from common.core.yaml_config import WorkspaceConfig

# logger imported from loguru


@dataclass
class ServiceEndpoint:
    """Represents a discovered service endpoint."""
    
    host: str
    port: int
    project_id: str
    service_name: str
    protocol: str = "grpc"
    health_status: str = "unknown"
    last_seen: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_seen == 0.0:
            self.last_seen = time.time()
    
    @property
    def address(self) -> str:
        """Get the full address for this endpoint."""
        return f"{self.host}:{self.port}"
    
    @property
    def is_stale(self, max_age: float = 300.0) -> bool:
        """Check if this endpoint information is stale."""
        return time.time() - self.last_seen > max_age


@dataclass
class DiscoveryConfig:
    """Configuration for service discovery."""
    
    registry_path: Optional[Path] = None
    network_discovery_enabled: bool = True
    multicast_address: str = "224.0.0.1"
    multicast_port: int = 8765
    discovery_timeout: float = 10.0
    health_check_timeout: float = 5.0
    max_endpoints_cache: int = 100
    endpoint_ttl: float = 300.0  # 5 minutes


class ServiceDiscoveryClient:
    """
    Client for discovering and connecting to daemon instances.
    
    Implements a multi-tier discovery strategy:
    1. Registry-based discovery (file-based persistence)
    2. Network multicast discovery
    3. Configuration fallback
    4. Default endpoint attempts
    """
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        """Initialize service discovery client."""
        self.config = config or DiscoveryConfig()
        self.project_detector = ProjectDetector()
        self.endpoints_cache: Dict[str, ServiceEndpoint] = {}
        self._discovery_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Set up registry path if not provided
        if not self.config.registry_path:
            import tempfile
            temp_dir = Path(tempfile.gettempdir())
            self.config.registry_path = temp_dir / "wqm_service_registry.json"
    
    async def start_discovery(self) -> None:
        """Start background service discovery."""
        if self._running:
            return
        
        self._running = True
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        logger.info("Started service discovery client")
    
    async def stop_discovery(self) -> None:
        """Stop background service discovery."""
        self._running = False
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped service discovery client")
    
    @asynccontextmanager
    async def discovery_context(self):
        """Context manager for service discovery lifecycle."""
        await self.start_discovery()
        try:
            yield self
        finally:
            await self.stop_discovery()
    
    async def discover_daemon_for_project(
        self, 
        project_path: str,
        preferred_endpoint: Optional[Tuple[str, int]] = None
    ) -> Optional[ServiceEndpoint]:
        """
        Discover daemon instance for a specific project.
        
        Args:
            project_path: Path to the project directory
            preferred_endpoint: Optional preferred (host, port) tuple
            
        Returns:
            ServiceEndpoint if found, None otherwise
        """
        # Generate project identifier
        identifier = self.project_detector.create_daemon_identifier(project_path)
        project_id = identifier.generate_identifier()
        
        logger.debug(f"Discovering daemon for project {project_id} at {project_path}")
        
        # Strategy 1: Check registry
        endpoint = await self._discover_from_registry(project_id)
        if endpoint and await self._verify_endpoint_health(endpoint):
            logger.info(f"Found daemon via registry: {endpoint.address} for project {project_id}")
            return endpoint
        
        # Strategy 2: Network discovery
        endpoint = await self._discover_via_network(project_id)
        if endpoint and await self._verify_endpoint_health(endpoint):
            logger.info(f"Found daemon via network discovery: {endpoint.address} for project {project_id}")
            await self._register_endpoint(endpoint)
            return endpoint
        
        # Strategy 3: Configuration fallback
        if preferred_endpoint:
            endpoint = ServiceEndpoint(
                host=preferred_endpoint[0],
                port=preferred_endpoint[1],
                project_id=project_id,
                service_name="workspace-qdrant-daemon"
            )
            if await self._verify_endpoint_health(endpoint):
                logger.info(f"Found daemon via configuration: {endpoint.address} for project {project_id}")
                await self._register_endpoint(endpoint)
                return endpoint
        
        # Strategy 4: Default endpoints
        for port in range(50051, 50061):  # Try common port range
            endpoint = ServiceEndpoint(
                host="127.0.0.1",
                port=port,
                project_id=project_id,
                service_name="workspace-qdrant-daemon"
            )
            if await self._verify_endpoint_health(endpoint):
                logger.info(f"Found daemon via default scan: {endpoint.address} for project {project_id}")
                await self._register_endpoint(endpoint)
                return endpoint
        
        logger.warning(f"No healthy daemon found for project {project_id}")
        return None
    
    async def list_available_daemons(self) -> List[ServiceEndpoint]:
        """List all available daemon endpoints."""
        # Load from registry
        await self._load_registry()
        
        # Filter out stale endpoints
        healthy_endpoints = []
        for endpoint in self.endpoints_cache.values():
            if not endpoint.is_stale and await self._verify_endpoint_health(endpoint):
                healthy_endpoints.append(endpoint)
        
        return healthy_endpoints
    
    async def register_daemon(self, endpoint: ServiceEndpoint) -> None:
        """Register a daemon endpoint."""
        await self._register_endpoint(endpoint)
    
    async def unregister_daemon(self, project_id: str) -> None:
        """Unregister a daemon endpoint."""
        if project_id in self.endpoints_cache:
            del self.endpoints_cache[project_id]
            await self._save_registry()
            logger.info(f"Unregistered daemon {project_id}")
    
    async def _discovery_loop(self) -> None:
        """Background discovery loop."""
        try:
            while self._running:
                # Cleanup stale entries
                await self._cleanup_stale_endpoints()
                
                # Perform network discovery
                if self.config.network_discovery_enabled:
                    await self._network_discovery_scan()
                
                # Wait before next scan
                await asyncio.sleep(30.0)  # Scan every 30 seconds
                
        except asyncio.CancelledError:
            logger.debug("Discovery loop cancelled")
        except Exception as e:
            logger.error(f"Error in discovery loop: {str(e)}")
    
    async def _discover_from_registry(self, project_id: str) -> Optional[ServiceEndpoint]:
        """Discover service from registry file."""
        await self._load_registry()
        return self.endpoints_cache.get(project_id)
    
    async def _discover_via_network(self, project_id: str) -> Optional[ServiceEndpoint]:
        """Discover service via network multicast."""
        if not self.config.network_discovery_enabled:
            return None
        
        try:
            # Create multicast socket for discovery
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Enable broadcasting
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            # Send discovery request
            discovery_msg = {
                "type": "discovery_request",
                "target_project_id": project_id,
                "sender": "python_client",
                "timestamp": time.time()
            }
            
            message = json.dumps(discovery_msg).encode('utf-8')
            sock.sendto(message, (self.config.multicast_address, self.config.multicast_port))
            
            # Set timeout for response
            sock.settimeout(self.config.discovery_timeout)
            
            try:
                data, addr = sock.recvfrom(1024)
                response = json.loads(data.decode('utf-8'))
                
                if (response.get("type") == "discovery_response" and 
                    response.get("project_id") == project_id):
                    
                    endpoint = ServiceEndpoint(
                        host=response.get("host", addr[0]),
                        port=response.get("port", 50051),
                        project_id=project_id,
                        service_name=response.get("service_name", "workspace-qdrant-daemon"),
                        metadata=response.get("metadata", {})
                    )
                    
                    return endpoint
                    
            except socket.timeout:
                logger.debug(f"Network discovery timeout for project {project_id}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid discovery response {error=str(e}"))
            
            sock.close()
            
        except Exception as e:
            logger.warning(f"Network discovery failed {project_id=project_id, error=str(e}"))
        
        return None
    
    async def _network_discovery_scan(self) -> None:
        """Perform a general network discovery scan."""
        try:
            # Send general discovery broadcast
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            discovery_msg = {
                "type": "general_discovery",
                "sender": "python_client",
                "timestamp": time.time()
            }
            
            message = json.dumps(discovery_msg).encode('utf-8')
            sock.sendto(message, (self.config.multicast_address, self.config.multicast_port))
            sock.close()
            
        except Exception as e:
            logger.debug(f"Network discovery scan failed {error=str(e}"))
    
    async def _verify_endpoint_health(self, endpoint: ServiceEndpoint) -> bool:
        """Verify that an endpoint is healthy and reachable."""
        try:
            # Try to establish a TCP connection
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(endpoint.host, endpoint.port),
                timeout=self.config.health_check_timeout
            )
            
            # Close the connection
            writer.close()
            await writer.wait_closed()
            
            endpoint.health_status = "healthy"
            endpoint.last_seen = time.time()
            return True
            
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
            logger.debug(f"Health check failed {endpoint=endpoint.address, error=str(e}"))
            endpoint.health_status = "unhealthy"
            return False
    
    async def _register_endpoint(self, endpoint: ServiceEndpoint) -> None:
        """Register an endpoint in the cache and persistent registry."""
        self.endpoints_cache[endpoint.project_id] = endpoint
        await self._save_registry()
        logger.debug(f"Registered endpoint {endpoint.address} for project {endpoint.project_id}")
    
    async def _load_registry(self) -> None:
        """Load endpoints from persistent registry."""
        if not self.config.registry_path or not self.config.registry_path.exists():
            return
        
        try:
            with open(self.config.registry_path, 'r') as f:
                data = json.load(f)
            
            for project_id, endpoint_data in data.items():
                endpoint = ServiceEndpoint(
                    host=endpoint_data["host"],
                    port=endpoint_data["port"],
                    project_id=endpoint_data["project_id"],
                    service_name=endpoint_data["service_name"],
                    health_status=endpoint_data.get("health_status", "unknown"),
                    last_seen=endpoint_data.get("last_seen", 0.0),
                    metadata=endpoint_data.get("metadata", {})
                )
                self.endpoints_cache[project_id] = endpoint
            
            logger.debug(f"Loaded registry {count=len(self.endpoints_cache}"))
            
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Failed to load service registry {error=str(e}"))
    
    async def _save_registry(self) -> None:
        """Save endpoints to persistent registry."""
        if not self.config.registry_path:
            return
        
        try:
            # Ensure directory exists
            self.config.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            data = {}
            for project_id, endpoint in self.endpoints_cache.items():
                data[project_id] = {
                    "host": endpoint.host,
                    "port": endpoint.port,
                    "project_id": endpoint.project_id,
                    "service_name": endpoint.service_name,
                    "health_status": endpoint.health_status,
                    "last_seen": endpoint.last_seen,
                    "metadata": endpoint.metadata
                }
            
            # Write atomically
            temp_path = self.config.registry_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_path.replace(self.config.registry_path)
            logger.debug(f"Saved registry {count=len(data}"))
            
        except Exception as e:
            logger.warning(f"Failed to save service registry {error=str(e}"))
    
    async def _cleanup_stale_endpoints(self) -> None:
        """Remove stale endpoints from cache."""
        stale_projects = [
            project_id for project_id, endpoint in self.endpoints_cache.items()
            if endpoint.is_stale
        ]
        
        for project_id in stale_projects:
            del self.endpoints_cache[project_id]
            logger.debug(f"Removed stale endpoint for project {project_id}")
        
        if stale_projects:
            await self._save_registry()


# Global service discovery client instance
_discovery_client: Optional[ServiceDiscoveryClient] = None


async def get_discovery_client() -> ServiceDiscoveryClient:
    """Get or create the global service discovery client."""
    global _discovery_client
    if _discovery_client is None:
        _discovery_client = ServiceDiscoveryClient()
    return _discovery_client


async def discover_daemon_endpoint(
    project_path: str,
    preferred_endpoint: Optional[Tuple[str, int]] = None
) -> Optional[ServiceEndpoint]:
    """
    Convenience function to discover daemon endpoint for a project.
    
    Args:
        project_path: Path to the project directory
        preferred_endpoint: Optional preferred (host, port) tuple
        
    Returns:
        ServiceEndpoint if found, None otherwise
    """
    client = await get_discovery_client()
    return await client.discover_daemon_for_project(project_path, preferred_endpoint)


async def list_available_daemons() -> List[ServiceEndpoint]:
    """List all available daemon endpoints."""
    client = await get_discovery_client()
    return await client.list_available_daemons()