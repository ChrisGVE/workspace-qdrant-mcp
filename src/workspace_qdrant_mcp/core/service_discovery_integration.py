"""
Service Discovery Integration for Python MCP Server

This module provides integration between the service discovery system
and the Python MCP server, enabling automatic communication with the
Rust daemon.
"""

import asyncio
import logging
import os
import signal
from typing import Optional

from .service_discovery import (
    DiscoveryManager, DiscoveryConfig, ServiceInfo, ServiceStatus,
    ServiceDiscoveryEvent, ServiceNames
)

logger = logging.getLogger(__name__)


class MCPServerWithDiscovery:
    """MCP Server with integrated service discovery"""
    
    def __init__(self, 
                 host: str = "127.0.0.1",
                 port: int = 8000,
                 discovery_config: Optional[DiscoveryConfig] = None):
        """Initialize MCP server with service discovery"""
        self.host = host
        self.port = port
        self.discovery_manager = DiscoveryManager(discovery_config)
        self.running = False
        
        # Setup discovery event handling
        self.discovery_manager.subscribe_events(self._handle_discovery_event)
        
    async def start(self) -> None:
        """Start the MCP server with service discovery"""
        logger.info("Starting MCP server with service discovery")
        
        self.running = True
        
        # Start discovery system
        await self.discovery_manager.start()
        
        # Create our service info
        our_service = ServiceInfo.create(
            host=self.host,
            port=self.port
        ).with_auth_token(
            ServiceInfo.generate_auth_token()
        ).with_health_endpoint(
            "/health"
        ).with_metadata(
            "version", "0.2.1"
        ).with_metadata(
            "component", "python-mcp"
        ).with_metadata(
            "transport", "stdio"
        )
        
        # Register ourselves in the discovery system
        await self.discovery_manager.register_service(
            ServiceNames.PYTHON_MCP, 
            our_service
        )
        logger.info("Python MCP server registered for service discovery")
        
        # Try to discover Rust daemon
        await self._discover_rust_daemon()
        
        # Start health monitoring
        await self.discovery_manager.start_health_monitoring()
        
        # Setup health check endpoint (simplified HTTP server)
        asyncio.create_task(self._run_health_server())
        
        logger.info(f"MCP server with discovery running on {self.host}:{self.port}")
        
    async def stop(self) -> None:
        """Stop the MCP server and discovery system"""
        logger.info("Stopping MCP server with service discovery")
        
        self.running = False
        
        # Deregister from discovery system
        await self.discovery_manager.deregister_service(ServiceNames.PYTHON_MCP)
        
        # Stop discovery system
        await self.discovery_manager.stop()
        
        logger.info("MCP server with discovery stopped")
        
    async def discover_rust_daemon(self) -> Optional[ServiceInfo]:
        """Discover the Rust daemon service"""
        return await self.discovery_manager.discover_service(ServiceNames.RUST_DAEMON)
        
    async def get_all_services(self) -> dict:
        """Get all discovered services"""
        return await self.discovery_manager.get_known_services()
        
    def _handle_discovery_event(self, event: ServiceDiscoveryEvent) -> None:
        """Handle discovery events"""
        if event.event_type == "service_discovered":
            logger.info(
                f"Discovered service {event.service_name} via {event.strategy} "
                f"at {event.service_info.host}:{event.service_info.port}"
            )
            
            # If it's the Rust daemon, establish communication
            if event.service_name == ServiceNames.RUST_DAEMON:
                asyncio.create_task(self._establish_daemon_communication(event.service_info))
                
        elif event.event_type == "service_lost":
            logger.warning(f"Lost service {event.service_name}: {event.reason}")
            
        elif event.event_type == "health_changed":
            logger.info(
                f"Health changed for {event.service_name}: "
                f"{event.old_status} -> {event.new_status}"
            )
            
    async def _discover_rust_daemon(self) -> None:
        """Try to discover the Rust daemon"""
        rust_daemon = await self.discover_rust_daemon()
        
        if rust_daemon:
            logger.info(f"Found Rust daemon at {rust_daemon.host}:{rust_daemon.port}")
            await self._establish_daemon_communication(rust_daemon)
        else:
            logger.warning("Rust daemon not found. Will continue checking...")
            
    async def _establish_daemon_communication(self, daemon_info: ServiceInfo) -> None:
        """Establish communication with the Rust daemon"""
        logger.info(f"Establishing communication with Rust daemon at {daemon_info.host}:{daemon_info.port}")
        
        # TODO: Implement actual communication setup
        # This could involve:
        # - Setting up gRPC clients
        # - Establishing IPC channels
        # - Configuring data exchange protocols
        # - Setting up event streaming
        
        try:
            # Check daemon health
            health_status = await self.discovery_manager.check_service_health(ServiceNames.RUST_DAEMON)
            logger.info(f"Rust daemon health: {health_status}")
            
            # For now, just log that we would establish communication
            logger.info("Communication with Rust daemon would be established here")
            
        except Exception as e:
            logger.error(f"Failed to establish communication with Rust daemon: {e}")
    
    async def _run_health_server(self) -> None:
        """Run a simple HTTP server for health checks"""
        try:
            from aiohttp import web
        except ImportError:
            logger.warning("aiohttp not available, health server disabled")
            return
        
        async def health_handler(request):
            """Health check endpoint handler"""
            health_data = {
                "status": "healthy",
                "timestamp": asyncio.get_event_loop().time(),
                "service": ServiceNames.PYTHON_MCP,
                "pid": os.getpid(),
                "version": "0.2.1"
            }
            return web.json_response(health_data)
        
        # Create simple web app
        app = web.Application()
        app.router.add_get('/health', health_handler)
        
        # Start server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Health server started on http://{self.host}:{self.port}/health")
        
        # Keep running while server is active
        while self.running:
            await asyncio.sleep(1)
            
        # Cleanup
        await runner.cleanup()


async def main():
    """Example main function demonstrating service discovery integration"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure service discovery
    config = DiscoveryConfig(
        enable_network_discovery=True,
        enable_authentication=True,
        discovery_timeout=10.0,
        health_check_interval=30.0
    )
    
    # Create and start server
    server = MCPServerWithDiscovery(
        host="127.0.0.1",
        port=8000,
        discovery_config=config
    )
    
    try:
        await server.start()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler():
            logger.info("Received shutdown signal")
            asyncio.create_task(server.stop())
        
        # Register signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, signal_handler)
        
        # Keep running
        logger.info("MCP server running. Press Ctrl+C to shutdown.")
        while server.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())