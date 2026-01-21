"""gRPC tools for CLI interaction with the daemon.

This module provides wrapper functions for the CLI to communicate with the
memexd daemon via gRPC. Functions are designed to be called from the CLI's
status and admin commands.

Task 422: Re-enable CLI gRPC integration
"""

import asyncio
from typing import Any, AsyncIterator

from common.grpc.daemon_client import DaemonClient

# Default configuration
DEFAULT_GRPC_HOST = "localhost"
DEFAULT_GRPC_PORT = 50051


async def test_grpc_connection(
    host: str = DEFAULT_GRPC_HOST,
    port: int = DEFAULT_GRPC_PORT,
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Test connection to the gRPC daemon.

    Args:
        host: Daemon host address
        port: Daemon gRPC port
        timeout: Connection timeout in seconds

    Returns:
        dict with 'connected' bool and optional 'error' message
    """
    client = DaemonClient(host=host, port=port)
    try:
        await client.start()
        health = await client.health_check(timeout=timeout)
        await client.stop()
        return {
            "connected": True,
            "status": str(health.status) if hasattr(health, 'status') else "healthy",
            "components": [
                {
                    "name": c.component_name,
                    "status": str(c.status),
                    "message": c.message,
                }
                for c in health.components
            ] if hasattr(health, 'components') else [],
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
        }
    finally:
        try:
            await client.stop()
        except Exception:
            pass


async def get_engine_stats(
    host: str = DEFAULT_GRPC_HOST,
    port: int = DEFAULT_GRPC_PORT,
) -> dict[str, Any]:
    """Get engine statistics from the daemon.

    Uses the new SystemService.GetStatus RPC (not legacy IngestService).

    Args:
        host: Daemon host address
        port: Daemon gRPC port

    Returns:
        dict with engine statistics
    """
    client = DaemonClient(host=host, port=port)
    try:
        await client.start()
        # Use new protocol's get_status() instead of legacy get_stats()
        status = await client.get_status()
        await client.stop()

        # Calculate uptime from timestamp
        uptime_seconds = 0
        if hasattr(status, 'uptime_since') and status.uptime_since.seconds:
            import time
            uptime_seconds = int(time.time() - status.uptime_since.seconds)

        return {
            "success": True,
            "stats": {
                "total_documents": getattr(status, 'total_documents', 0),
                "total_collections": getattr(status, 'total_collections', 0),
                "active_projects": getattr(status, 'active_projects', 0),
                "active_connections": getattr(status.metrics, 'active_connections', 0) if hasattr(status, 'metrics') else 0,
                "uptime_seconds": uptime_seconds,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        try:
            await client.stop()
        except Exception:
            pass


async def get_grpc_engine_stats(
    host: str = DEFAULT_GRPC_HOST,
    port: int = DEFAULT_GRPC_PORT,
) -> dict[str, Any]:
    """Alias for get_engine_stats for backwards compatibility."""
    return await get_engine_stats(host, port)


async def stream_processing_status_grpc(
    host: str = DEFAULT_GRPC_HOST,
    port: int = DEFAULT_GRPC_PORT,
) -> dict[str, Any]:
    """Get current processing status from the daemon.

    Uses the new SystemService.GetStatus RPC (not legacy IngestService).
    Note: This is a single-shot query, not a streaming operation.
    The name is retained for backwards compatibility with CLI code.

    Args:
        host: Daemon host address
        port: Daemon gRPC port

    Returns:
        dict with processing status
    """
    client = DaemonClient(host=host, port=port)
    try:
        await client.start()
        # Use new protocol's get_status() instead of legacy get_processing_status()
        status = await client.get_status()
        await client.stop()

        # Calculate uptime from timestamp
        uptime_seconds = 0
        if hasattr(status, 'uptime_since') and status.uptime_since.seconds:
            import time
            uptime_seconds = int(time.time() - status.uptime_since.seconds)

        return {
            "success": True,
            "status": {
                "total_documents": getattr(status, 'total_documents', 0),
                "total_collections": getattr(status, 'total_collections', 0),
                "active_projects": getattr(status, 'active_projects', 0),
                "active_connections": getattr(status.metrics, 'active_connections', 0) if hasattr(status, 'metrics') else 0,
                "uptime_seconds": uptime_seconds,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        try:
            await client.stop()
        except Exception:
            pass


async def stream_system_metrics_grpc(
    host: str = DEFAULT_GRPC_HOST,
    port: int = DEFAULT_GRPC_PORT,
) -> dict[str, Any]:
    """Get system metrics from the daemon.

    Note: This is a single-shot query, not a streaming operation.

    Args:
        host: Daemon host address
        port: Daemon gRPC port

    Returns:
        dict with system metrics
    """
    client = DaemonClient(host=host, port=port)
    try:
        await client.start()
        metrics = await client.get_metrics()
        await client.stop()
        return {
            "success": True,
            "metrics": {
                "cpu_usage": getattr(metrics, 'cpu_usage', 0.0),
                "memory_usage": getattr(metrics, 'memory_usage', 0.0),
                "disk_usage": getattr(metrics, 'disk_usage', 0.0),
                "uptime_seconds": getattr(metrics, 'uptime_seconds', 0),
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        try:
            await client.stop()
        except Exception:
            pass


async def stream_queue_status_grpc(
    host: str = DEFAULT_GRPC_HOST,
    port: int = DEFAULT_GRPC_PORT,
) -> dict[str, Any]:
    """Get queue status from the daemon.

    Uses the new SystemService.GetStatus RPC (not legacy IngestService).
    Note: This is a single-shot query, not a streaming operation.

    Args:
        host: Daemon host address
        port: Daemon gRPC port

    Returns:
        dict with queue status
    """
    client = DaemonClient(host=host, port=port)
    try:
        await client.start()
        # Use new protocol's get_status() instead of legacy get_processing_status()
        status = await client.get_status()
        await client.stop()
        return {
            "success": True,
            "queue": {
                "total_queued": 0,  # Not yet implemented in daemon
                "active_projects": getattr(status, 'active_projects', 0),
                "active_connections": getattr(status.metrics, 'active_connections', 0) if hasattr(status, 'metrics') else 0,
            },
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
    finally:
        try:
            await client.stop()
        except Exception:
            pass
