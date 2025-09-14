"""
Service Discovery Manager

This module provides the main orchestrator for service discovery, combining
file-based registry, network discovery, health checking, and configuration
fallback mechanisms into a unified discovery system.
"""

import asyncio
from common.logging import get_logger
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Union
from pathlib import Path

from .exceptions import DiscoveryError, ServiceNotFoundError
from .registry import ServiceRegistry, ServiceInfo, ServiceStatus
from .network import NetworkDiscovery, DiscoveryEvent
from .health import HealthChecker, HealthConfig, HealthStatus, HealthCheckResult

logger = get_logger(__name__)


class DiscoveryStrategy(Enum):
    """Discovery strategy enumeration"""
    REGISTRY = "registry"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    DEFAULTS = "defaults"


@dataclass
class DiscoveryConfig:
    """Service discovery configuration"""
    registry_path: Optional[Path] = None
    multicast_address: str = "239.255.42.42"
    multicast_port: int = 9999
    discovery_timeout: float = 10.0
    health_check_interval: float = 30.0
    cleanup_interval: float = 60.0
    enable_network_discovery: bool = True
    enable_authentication: bool = True
    auth_token: Optional[str] = None


@dataclass 
class ServiceDiscoveryEvent:
    """Service discovery events"""
    event_type: str
    service_name: str
    service_info: Optional[ServiceInfo] = None
    strategy: Optional[DiscoveryStrategy] = None
    old_status: Optional[HealthStatus] = None
    new_status: Optional[HealthStatus] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class DiscoveryManager:
    """Discovery manager orchestrating all discovery mechanisms"""

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        """Initialize discovery manager"""
        self.config = config or DiscoveryConfig()
        
        # Initialize components
        self.registry = ServiceRegistry(self.config.registry_path)
        
        self.network_discovery: Optional[NetworkDiscovery] = None
        if self.config.enable_network_discovery:
            auth_token = None
            if self.config.enable_authentication:
                auth_token = self.config.auth_token or ServiceInfo.generate_auth_token()
            
            self.network_discovery = NetworkDiscovery(
                self.config.multicast_address,
                self.config.multicast_port,
                auth_token
            )
        
        health_config = HealthConfig(
            request_timeout=self.config.discovery_timeout,
            check_interval=self.config.health_check_interval,
            max_failures=3,
            validate_process=True
        )
        self.health_checker = HealthChecker(health_config)
        
        # Event management
        self.event_callbacks: List[Callable[[ServiceDiscoveryEvent], None]] = []
        self.known_services: Dict[str, tuple[ServiceInfo, DiscoveryStrategy]] = {}
        self.monitoring_tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info("Service discovery manager initialized")

    async def start(self) -> None:
        """Start the discovery manager"""
        logger.info("Starting service discovery manager")
        
        self.running = True
        
        # Start health checker
        await self.health_checker.start()
        
        # Start network discovery if enabled
        if self.network_discovery:
            await self.network_discovery.start()
            self._setup_network_event_handling()
        
        # Start periodic tasks
        self._start_periodic_cleanup()
        
        logger.info("Service discovery manager started successfully")

    async def stop(self) -> None:
        """Stop the discovery manager"""
        logger.info("Stopping service discovery manager")
        
        self.running = False
        
        # Stop monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        self.monitoring_tasks.clear()
        
        # Stop components
        await self.health_checker.stop()
        
        if self.network_discovery:
            await self.network_discovery.stop()
        
        logger.info("Service discovery manager stopped")

    async def register_service(self, service_name: str, service_info: ServiceInfo) -> None:
        """Register this service in the discovery system"""
        logger.info(f"Registering service: {service_name} at {service_info.host}:{service_info.port}")
        
        try:
            # Register in file-based registry
            await self.registry.register_service(service_name, service_info)
            
            # Announce via network discovery if enabled
            if self.network_discovery:
                await self.network_discovery.announce_service(service_name, service_info)
            
            # Add to known services
            self.known_services[service_name] = (service_info, DiscoveryStrategy.REGISTRY)
            
            # Send discovery event
            event = ServiceDiscoveryEvent(
                event_type="service_discovered",
                service_name=service_name,
                service_info=service_info,
                strategy=DiscoveryStrategy.REGISTRY
            )
            self._broadcast_event(event)
            
            logger.info(f"Service {service_name} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register service {service_name}: {e}")
            raise DiscoveryError(f"Service registration failed: {e}")

    async def deregister_service(self, service_name: str) -> None:
        """Deregister this service from the discovery system"""
        logger.info(f"Deregistering service: {service_name}")
        
        try:
            # Deregister from file-based registry
            await self.registry.deregister_service(service_name)
            
            # Announce shutdown via network discovery if enabled
            if self.network_discovery:
                await self.network_discovery.announce_shutdown(service_name)
            
            # Remove from known services
            self.known_services.pop(service_name, None)
            
            # Stop health monitoring for this service
            await self.health_checker.stop_monitoring(service_name)
            
            # Send lost event
            event = ServiceDiscoveryEvent(
                event_type="service_lost",
                service_name=service_name,
                reason="Deregistered"
            )
            self._broadcast_event(event)
            
            logger.info(f"Service {service_name} deregistered successfully")
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_name}: {e}")
            raise DiscoveryError(f"Service deregistration failed: {e}")

    async def discover_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Discover a specific service using all available strategies"""
        logger.info(f"Discovering service: {service_name}")
        
        # Strategy 1: Check file-based registry
        service_info = await self._try_registry_discovery(service_name)
        if service_info:
            return service_info
        
        # Strategy 2: Network discovery
        if self.network_discovery:
            service_info = await self._try_network_discovery(service_name)
            if service_info:
                return service_info
        
        # Strategy 3: Configuration fallback
        service_info = await self._try_configuration_discovery(service_name)
        if service_info:
            return service_info
        
        # Strategy 4: Default endpoints
        service_info = await self._try_default_discovery(service_name)
        if service_info:
            return service_info
        
        logger.warning(f"Service {service_name} not found using any discovery strategy")
        return None

    async def discover_all_services(self) -> Dict[str, ServiceInfo]:
        """Discover all available services"""
        logger.info("Discovering all available services")
        
        discovered = {}
        
        # Get services from registry
        try:
            registry_services = await self.registry.list_services()
            for name, info in registry_services:
                discovered[name] = info
        except Exception as e:
            logger.warning(f"Registry discovery failed: {e}")
        
        # Get services from network discovery
        if self.network_discovery:
            try:
                network_services = await self.network_discovery.discover_services(
                    [], self.config.discovery_timeout
                )
                discovered.update(network_services)
            except Exception as e:
                logger.warning(f"Network discovery failed: {e}")
        
        logger.info(f"Discovered {len(discovered)} services total")
        return discovered

    async def get_known_services(self) -> Dict[str, ServiceInfo]:
        """Get all currently known services"""
        return {
            name: service_info 
            for name, (service_info, _) in self.known_services.items()
        }

    async def check_service_health(self, service_name: str) -> Optional[HealthStatus]:
        """Check health of a specific service"""
        if service_name in self.known_services:
            service_info, _ = self.known_services[service_name]
            try:
                result = await self.health_checker.check_service_health(service_name, service_info)
                return result.status
            except Exception as e:
                logger.warning(f"Health check failed for {service_name}: {e}")
                return HealthStatus.UNREACHABLE
        return None

    async def start_health_monitoring(self, services: Optional[Dict[str, ServiceInfo]] = None) -> None:
        """Start health monitoring for services"""
        if services is None:
            services = await self.get_known_services()
        
        await self.health_checker.start_monitoring(services)
        logger.info(f"Started health monitoring for {len(services)} services")

    def subscribe_events(self, callback: Callable[[ServiceDiscoveryEvent], None]) -> None:
        """Subscribe to discovery events"""
        self.event_callbacks.append(callback)

    def unsubscribe_events(self, callback: Callable[[ServiceDiscoveryEvent], None]) -> None:
        """Unsubscribe from discovery events"""
        if callback in self.event_callbacks:
            self.event_callbacks.remove(callback)

    async def _try_registry_discovery(self, service_name: str) -> Optional[ServiceInfo]:
        """Try discovery using file-based registry"""
        logger.debug(f"Trying registry discovery for: {service_name}")
        
        try:
            service_info = await self.registry.discover_service(service_name)
            if service_info:
                logger.debug(f"Found {service_name} via registry discovery")
                self._update_known_service(service_name, service_info, DiscoveryStrategy.REGISTRY)
                
                event = ServiceDiscoveryEvent(
                    event_type="service_discovered",
                    service_name=service_name,
                    service_info=service_info,
                    strategy=DiscoveryStrategy.REGISTRY
                )
                self._broadcast_event(event)
                
                return service_info
            else:
                logger.debug(f"Service {service_name} not found in registry")
                
        except Exception as e:
            logger.warning(f"Registry discovery failed for {service_name}: {e}")
            
            event = ServiceDiscoveryEvent(
                event_type="strategy_failed",
                service_name=service_name,
                strategy=DiscoveryStrategy.REGISTRY,
                error=str(e)
            )
            self._broadcast_event(event)
        
        return None

    async def _try_network_discovery(self, service_name: str) -> Optional[ServiceInfo]:
        """Try discovery using network multicast"""
        logger.debug(f"Trying network discovery for: {service_name}")
        
        try:
            discovered = await self.network_discovery.discover_services(
                [service_name], self.config.discovery_timeout
            )
            
            if service_name in discovered:
                service_info = discovered[service_name]
                logger.debug(f"Found {service_name} via network discovery")
                self._update_known_service(service_name, service_info, DiscoveryStrategy.NETWORK)
                
                event = ServiceDiscoveryEvent(
                    event_type="service_discovered",
                    service_name=service_name,
                    service_info=service_info,
                    strategy=DiscoveryStrategy.NETWORK
                )
                self._broadcast_event(event)
                
                return service_info
            else:
                logger.debug(f"Service {service_name} not found via network discovery")
                
        except Exception as e:
            logger.warning(f"Network discovery failed for {service_name}: {e}")
            
            event = ServiceDiscoveryEvent(
                event_type="strategy_failed",
                service_name=service_name,
                strategy=DiscoveryStrategy.NETWORK,
                error=str(e)
            )
            self._broadcast_event(event)
        
        return None

    async def _try_configuration_discovery(self, service_name: str) -> Optional[ServiceInfo]:
        """Try discovery using configuration files"""
        logger.debug(f"Trying configuration discovery for: {service_name}")
        
        # TODO: Implement configuration-based discovery
        # This would read from unified configuration files
        
        logger.warning("Configuration discovery not yet implemented")
        
        event = ServiceDiscoveryEvent(
            event_type="strategy_failed",
            service_name=service_name,
            strategy=DiscoveryStrategy.CONFIGURATION,
            error="Not implemented"
        )
        self._broadcast_event(event)
        
        return None

    async def _try_default_discovery(self, service_name: str) -> Optional[ServiceInfo]:
        """Try discovery using default endpoints"""
        logger.debug(f"Trying default endpoint discovery for: {service_name}")
        
        # Define default service endpoints
        default_services = {
            "rust-daemon": ServiceInfo.create("127.0.0.1", 8080),
            "python-mcp": ServiceInfo.create("127.0.0.1", 8000)
        }
        
        if service_name in default_services:
            service_info = default_services[service_name]
            
            # Verify the service is actually reachable
            try:
                result = await self.health_checker.check_service_health(service_name, service_info)
                if result.is_reachable():
                    logger.debug(f"Found {service_name} via default endpoints")
                    self._update_known_service(service_name, service_info, DiscoveryStrategy.DEFAULTS)
                    
                    event = ServiceDiscoveryEvent(
                        event_type="service_discovered",
                        service_name=service_name,
                        service_info=service_info,
                        strategy=DiscoveryStrategy.DEFAULTS
                    )
                    self._broadcast_event(event)
                    
                    return service_info
                else:
                    logger.debug(f"Default endpoint for {service_name} not reachable")
                    
            except Exception as e:
                logger.debug(f"Health check failed for default {service_name}: {e}")
                
            event = ServiceDiscoveryEvent(
                event_type="strategy_failed",
                service_name=service_name,
                strategy=DiscoveryStrategy.DEFAULTS,
                error="Service not reachable"
            )
            self._broadcast_event(event)
        else:
            logger.debug(f"No default endpoint defined for service: {service_name}")
        
        return None

    def _update_known_service(self, service_name: str, service_info: ServiceInfo, strategy: DiscoveryStrategy) -> None:
        """Update known service information"""
        self.known_services[service_name] = (service_info, strategy)

    def _setup_network_event_handling(self) -> None:
        """Setup network discovery event handling"""
        if self.network_discovery:
            def handle_network_event(event: DiscoveryEvent):
                if event.event_type == "service_discovered":
                    self._update_known_service(
                        event.service_name, 
                        event.service_info, 
                        DiscoveryStrategy.NETWORK
                    )
                    
                    discovery_event = ServiceDiscoveryEvent(
                        event_type="service_discovered",
                        service_name=event.service_name,
                        service_info=event.service_info,
                        strategy=DiscoveryStrategy.NETWORK
                    )
                    self._broadcast_event(discovery_event)
                    
                elif event.event_type == "service_lost":
                    self.known_services.pop(event.service_name, None)
                    
                    discovery_event = ServiceDiscoveryEvent(
                        event_type="service_lost",
                        service_name=event.service_name,
                        reason="Network discovery lost"
                    )
                    self._broadcast_event(discovery_event)
            
            self.network_discovery.subscribe_events(handle_network_event)

    def _start_periodic_cleanup(self) -> None:
        """Start periodic cleanup tasks"""
        task = asyncio.create_task(self._cleanup_loop())
        self.monitoring_tasks.append(task)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale entries"""
        while self.running:
            try:
                removed_services = await self.registry.cleanup_stale_entries()
                for service_name in removed_services:
                    self.known_services.pop(service_name, None)
                    
                    event = ServiceDiscoveryEvent(
                        event_type="service_lost",
                        service_name=service_name,
                        reason="Stale registry entry"
                    )
                    self._broadcast_event(event)
                    
                await asyncio.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    def _broadcast_event(self, event: ServiceDiscoveryEvent) -> None:
        """Broadcast event to all subscribers"""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")


# Service name constants
class ServiceNames:
    """Known service names in the discovery system"""
    RUST_DAEMON = "rust-daemon"
    PYTHON_MCP = "python-mcp"