"""
Service Discovery System for workspace-qdrant-mcp

This module provides automatic service discovery between the Python MCP server
and Rust daemon components. It supports multiple discovery strategies with
fallback mechanisms for robust component communication.

Features:
    - File-based service registry management
    - Network discovery via UDP multicast
    - Health checking and monitoring
    - Process lifecycle tracking
    - Authentication and security
    - Cross-platform compatibility
"""

from .manager import DiscoveryManager
from .registry import ServiceRegistry, ServiceInfo, ServiceStatus
from .network import NetworkDiscovery, DiscoveryMessage, DiscoveryMessageType
from .health import HealthChecker, HealthStatus, HealthConfig
from .exceptions import DiscoveryError, RegistryError, NetworkError, HealthError
from .client import ServiceDiscoveryClient, ServiceEndpoint, DiscoveryConfig, discover_daemon_endpoint, list_available_daemons

__all__ = [
    # Main components
    'DiscoveryManager',
    'ServiceRegistry', 
    'NetworkDiscovery',
    'HealthChecker',
    'ServiceDiscoveryClient',
    
    # Data structures
    'ServiceInfo',
    'ServiceStatus', 
    'ServiceEndpoint',
    'DiscoveryMessage',
    'DiscoveryMessageType',
    'HealthStatus',
    'HealthConfig',
    'DiscoveryConfig',
    
    # Convenience functions
    'discover_daemon_endpoint',
    'list_available_daemons',
    
    # Exceptions
    'DiscoveryError',
    'RegistryError', 
    'NetworkError',
    'HealthError',
]