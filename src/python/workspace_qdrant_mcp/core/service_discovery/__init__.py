"""
Service discovery module proxy for backward compatibility.
"""

# Import all service discovery modules from common.core.service_discovery
try:
    from common.core.service_discovery import *
except ImportError:
    pass
