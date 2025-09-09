#!/usr/bin/env python3
"""
Service Discovery Integration Tests

This script tests the service discovery system to ensure both the Rust daemon
and Python MCP server can discover and communicate with each other.
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the src directory to Python path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workspace_qdrant_mcp.core.service_discovery import (
    DiscoveryManager, DiscoveryConfig, ServiceRegistry, ServiceInfo, ServiceStatus,
    NetworkDiscovery, HealthChecker, HealthConfig, ServiceNames
)


class ServiceDiscoveryTestSuite:
    """Comprehensive test suite for service discovery system"""
    
    def __init__(self):
        self.temp_dir = None
        self.registry_path = None
        self.test_results = []
        
    async def setup(self):
        """Setup test environment"""
        logger.info("Setting up service discovery test environment")
        
        # Create temporary directory for registry
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "test_services.json"
        
        logger.info(f"Test registry path: {self.registry_path}")
        
    async def cleanup(self):
        """Clean up test environment"""
        logger.info("Cleaning up test environment")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            
    def log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "PASSED" if passed else "FAILED"
        logger.info(f"TEST {status}: {test_name} - {details}")
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "details": details
        })
        
    async def test_file_registry_operations(self):
        """Test file-based registry operations"""
        logger.info("Testing file-based registry operations")
        
        try:
            # Create registry
            registry = ServiceRegistry(self.registry_path)
            
            # Test service registration
            service_info = ServiceInfo.create("127.0.0.1", 8080)
            service_info.with_metadata("test", "value")
            
            await registry.register_service("test-service", service_info)
            
            # Test service discovery
            discovered = await registry.discover_service("test-service")
            assert discovered is not None
            assert discovered.host == "127.0.0.1"
            assert discovered.port == 8080
            
            # Test service listing
            services = await registry.list_services()
            assert len(services) == 1
            assert services[0][0] == "test-service"
            
            # Test service deregistration
            result = await registry.deregister_service("test-service")
            assert result == True
            
            # Verify service is gone
            discovered = await registry.discover_service("test-service")
            assert discovered is None
            
            self.log_test_result("file_registry_operations", True, "All registry operations successful")
            
        except Exception as e:
            self.log_test_result("file_registry_operations", False, f"Error: {e}")
            
    async def test_network_discovery(self):
        """Test network discovery functionality"""
        logger.info("Testing network discovery")
        
        try:
            # Create two network discovery instances
            discovery1 = NetworkDiscovery("239.255.42.42", 19999, "test-token")
            discovery2 = NetworkDiscovery("239.255.42.42", 19999, "test-token")
            
            # Track discovered services
            discovered_services = []
            
            def event_handler(event):
                if event.event_type == "service_discovered":
                    discovered_services.append(event.service_name)
                    logger.info(f"Discovered: {event.service_name}")
            
            discovery2.subscribe_events(event_handler)
            
            # Start both discovery services
            await discovery1.start()
            await discovery2.start()
            
            # Wait a moment for startup
            await asyncio.sleep(1)
            
            # Announce a service from discovery1
            service_info = ServiceInfo.create("127.0.0.1", 8080)
            await discovery1.announce_service("test-network-service", service_info)
            
            # Wait for discovery
            await asyncio.sleep(2)
            
            # Check if discovery2 received the announcement
            cached_services = discovery2.get_cached_services()
            
            # Stop discovery services
            await discovery1.stop()
            await discovery2.stop()
            
            # Verify discovery worked
            if "test-network-service" in cached_services:
                self.log_test_result("network_discovery", True, "Network discovery successful")
            else:
                self.log_test_result("network_discovery", False, "Service not discovered via network")
                
        except Exception as e:
            self.log_test_result("network_discovery", False, f"Error: {e}")
            
    async def test_health_checking(self):
        """Test health checking functionality"""
        logger.info("Testing health checking")
        
        try:
            # Start a simple HTTP server for health checks
            from aiohttp import web
            
            async def health_handler(request):
                return web.json_response({"status": "healthy", "timestamp": time.time()})
            
            app = web.Application()
            app.router.add_get('/health', health_handler)
            
            runner = web.AppRunner(app)
            await runner.setup()
            
            site = web.TCPSite(runner, '127.0.0.1', 18080)
            await site.start()
            
            # Create health checker
            config = HealthConfig(request_timeout=5.0)
            health_checker = HealthChecker(config)
            await health_checker.start()
            
            # Test health check
            service_info = ServiceInfo.create("127.0.0.1", 18080)
            result = await health_checker.check_service_health("test-health-service", service_info)
            
            # Stop servers
            await runner.cleanup()
            await health_checker.stop()
            
            # Verify results
            if result.is_healthy():
                self.log_test_result("health_checking", True, f"Health check successful: {result.status}")
            else:
                self.log_test_result("health_checking", False, f"Health check failed: {result.status}")
                
        except Exception as e:
            self.log_test_result("health_checking", False, f"Error: {e}")
            
    async def test_discovery_manager_integration(self):
        """Test the integrated discovery manager"""
        logger.info("Testing discovery manager integration")
        
        try:
            config = DiscoveryConfig(
                registry_path=self.registry_path,
                multicast_address="239.255.42.42",
                multicast_port=19998,
                discovery_timeout=5.0,
                enable_network_discovery=True,
                enable_authentication=False  # Disable for testing
            )
            
            # Create discovery manager
            manager = DiscoveryManager(config)
            await manager.start()
            
            # Register a service
            service_info = ServiceInfo.create("127.0.0.1", 8080)
            await manager.register_service("test-integrated-service", service_info)
            
            # Discover the service
            discovered = await manager.discover_service("test-integrated-service")
            
            # Check known services
            known_services = await manager.get_known_services()
            
            # Deregister service
            await manager.deregister_service("test-integrated-service")
            
            # Stop manager
            await manager.stop()
            
            # Verify results
            if discovered and "test-integrated-service" in known_services:
                self.log_test_result("discovery_manager_integration", True, "Discovery manager working correctly")
            else:
                self.log_test_result("discovery_manager_integration", False, "Discovery manager failed")
                
        except Exception as e:
            self.log_test_result("discovery_manager_integration", False, f"Error: {e}")
            
    async def test_service_discovery_scenarios(self):
        """Test realistic service discovery scenarios"""
        logger.info("Testing realistic service discovery scenarios")
        
        try:
            # Scenario 1: Rust daemon starts first, then Python MCP
            config = DiscoveryConfig(
                registry_path=self.registry_path,
                enable_network_discovery=False  # Use registry only for deterministic testing
            )
            
            # Simulate Rust daemon
            daemon_manager = DiscoveryManager(config)
            await daemon_manager.start()
            
            daemon_service = ServiceInfo.create("127.0.0.1", 8080)
            daemon_service.with_additional_port("grpc", 50051)
            await daemon_manager.register_service(ServiceNames.RUST_DAEMON, daemon_service)
            
            # Simulate Python MCP starting later
            await asyncio.sleep(1)
            
            mcp_manager = DiscoveryManager(config)
            await mcp_manager.start()
            
            # Python MCP should discover Rust daemon
            rust_daemon = await mcp_manager.discover_service(ServiceNames.RUST_DAEMON)
            
            # Register Python MCP
            mcp_service = ServiceInfo.create("127.0.0.1", 8000)
            await mcp_manager.register_service(ServiceNames.PYTHON_MCP, mcp_service)
            
            # Rust daemon should discover Python MCP
            python_mcp = await daemon_manager.discover_service(ServiceNames.PYTHON_MCP)
            
            # Cleanup
            await daemon_manager.stop()
            await mcp_manager.stop()
            
            # Verify both services found each other
            if rust_daemon and python_mcp:
                self.log_test_result("service_discovery_scenarios", True, "Both services discovered each other")
            else:
                self.log_test_result("service_discovery_scenarios", False, 
                                   f"Discovery failed - rust_daemon: {bool(rust_daemon)}, python_mcp: {bool(python_mcp)}")
                
        except Exception as e:
            self.log_test_result("service_discovery_scenarios", False, f"Error: {e}")
            
    async def run_all_tests(self):
        """Run all tests in the suite"""
        logger.info("Starting comprehensive service discovery test suite")
        
        await self.setup()
        
        try:
            # Run individual tests
            await self.test_file_registry_operations()
            await self.test_network_discovery() 
            await self.test_health_checking()
            await self.test_discovery_manager_integration()
            await self.test_service_discovery_scenarios()
            
        finally:
            await self.cleanup()
            
        # Report results
        self.report_results()
        
    def report_results(self):
        """Report test results summary"""
        logger.info("=" * 60)
        logger.info("SERVICE DISCOVERY TEST RESULTS")
        logger.info("=" * 60)
        
        passed = sum(1 for result in self.test_results if result["passed"])
        total = len(self.test_results)
        
        for result in self.test_results:
            status = "✓" if result["passed"] else "✗"
            logger.info(f"{status} {result['test']}: {result['details']}")
            
        logger.info("=" * 60)
        logger.info(f"SUMMARY: {passed}/{total} tests passed")
        logger.info("=" * 60)
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! Service discovery system is working correctly.")
            return True
        else:
            logger.error("❌ Some tests failed. Please check the implementation.")
            return False


async def main():
    """Main test execution"""
    test_suite = ServiceDiscoveryTestSuite()
    success = await test_suite.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())