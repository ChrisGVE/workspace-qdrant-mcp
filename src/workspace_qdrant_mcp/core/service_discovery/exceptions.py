"""
Service Discovery Exceptions

Custom exception classes for service discovery operations.
"""

class DiscoveryError(Exception):
    """Base exception for service discovery errors"""
    pass


class RegistryError(DiscoveryError):
    """Registry-specific errors"""
    pass


class NetworkError(DiscoveryError):
    """Network discovery errors"""
    pass


class HealthError(DiscoveryError):
    """Health checking errors"""
    pass


class ServiceNotFoundError(DiscoveryError):
    """Service not found during discovery"""
    pass


class AuthenticationError(DiscoveryError):
    """Authentication failed during discovery"""
    pass


class TimeoutError(DiscoveryError):
    """Timeout during discovery operation"""
    pass