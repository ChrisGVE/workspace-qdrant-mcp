"""
gRPC client module for Python-Rust communication.

This module provides async gRPC client wrappers for communicating with
the Rust-based daemon using the workspace_daemon protocol.
"""

from .connection_manager import ConnectionConfig
from .daemon_client import (
    DaemonClient,
    DaemonConnectionError,
    DaemonUnavailableError,
)

__all__ = [
    "ConnectionConfig",
    "DaemonClient",
    "DaemonConnectionError",
    "DaemonUnavailableError",
]
