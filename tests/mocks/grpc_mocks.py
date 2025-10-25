"""
gRPC communication mocking for testing daemon/client interactions.

Provides comprehensive mocking for gRPC communications between the Python MCP server
and Rust daemon components, including error scenarios and realistic behavior simulation.
"""

import asyncio
import random
from collections.abc import AsyncIterator
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, Mock

import grpc
from grpc import StatusCode

from .error_injection import ErrorInjector, FailureScenarios


class GRPCErrorInjector(ErrorInjector):
    """Specialized error injector for gRPC operations."""

    def __init__(self):
        super().__init__()
        self.failure_modes = {
            "unavailable": {"probability": 0.0, "status_code": StatusCode.UNAVAILABLE},
            "deadline_exceeded": {"probability": 0.0, "status_code": StatusCode.DEADLINE_EXCEEDED},
            "unauthenticated": {"probability": 0.0, "status_code": StatusCode.UNAUTHENTICATED},
            "permission_denied": {"probability": 0.0, "status_code": StatusCode.PERMISSION_DENIED},
            "resource_exhausted": {"probability": 0.0, "status_code": StatusCode.RESOURCE_EXHAUSTED},
            "failed_precondition": {"probability": 0.0, "status_code": StatusCode.FAILED_PRECONDITION},
            "aborted": {"probability": 0.0, "status_code": StatusCode.ABORTED},
            "internal": {"probability": 0.0, "status_code": StatusCode.INTERNAL},
            "data_loss": {"probability": 0.0, "status_code": StatusCode.DATA_LOSS},
            "cancelled": {"probability": 0.0, "status_code": StatusCode.CANCELLED},
        }

    def configure_connection_issues(self, probability: float = 0.1):
        """Configure connection-related gRPC failures."""
        self.failure_modes["unavailable"]["probability"] = probability
        self.failure_modes["deadline_exceeded"]["probability"] = probability / 2

    def configure_auth_issues(self, probability: float = 0.1):
        """Configure authentication-related gRPC failures."""
        self.failure_modes["unauthenticated"]["probability"] = probability
        self.failure_modes["permission_denied"]["probability"] = probability / 2

    def configure_resource_issues(self, probability: float = 0.1):
        """Configure resource-related gRPC failures."""
        self.failure_modes["resource_exhausted"]["probability"] = probability
        self.failure_modes["failed_precondition"]["probability"] = probability / 2


class MockGRPCException(grpc.RpcError):
    """Mock gRPC exception for testing."""

    def __init__(self, status_code: StatusCode, details: str = ""):
        self._status_code = status_code
        self._details = details

    def code(self) -> StatusCode:
        return self._status_code

    def details(self) -> str:
        return self._details

    def __str__(self) -> str:
        return f"gRPC error: {self._status_code.name} - {self._details}"


class GRPCClientMock:
    """Mock gRPC client for daemon communication."""

    def __init__(self, error_injector: GRPCErrorInjector | None = None):
        self.error_injector = error_injector or GRPCErrorInjector()
        self.operation_history: list[dict[str, Any]] = []
        self.connected = False
        self.channel = None
        self.performance_delays = {
            "ingestion": 0.1,
            "search": 0.05,
            "status": 0.01,
            "collection_ops": 0.03,
        }

        # Setup method mocks
        self._setup_client_methods()

    def _setup_client_methods(self):
        """Setup gRPC client method mocks."""
        self.connect = AsyncMock(side_effect=self._mock_connect)
        self.disconnect = AsyncMock(side_effect=self._mock_disconnect)
        self.ingest_document = AsyncMock(side_effect=self._mock_ingest_document)
        self.search_documents = AsyncMock(side_effect=self._mock_search_documents)
        self.get_daemon_status = AsyncMock(side_effect=self._mock_get_daemon_status)
        self.create_collection = AsyncMock(side_effect=self._mock_create_collection)
        self.delete_collection = AsyncMock(side_effect=self._mock_delete_collection)
        self.watch_files = AsyncMock(side_effect=self._mock_watch_files)
        self.get_metadata = AsyncMock(side_effect=self._mock_get_metadata)

    async def _inject_grpc_error(self, operation: str) -> None:
        """Inject gRPC errors based on configuration."""
        # Add realistic operation delay
        if operation in self.performance_delays:
            await asyncio.sleep(self.performance_delays[operation])

        if self.error_injector.should_inject_error():
            error_type = self.error_injector.get_random_error()
            await self._raise_grpc_error(error_type)

    async def _raise_grpc_error(self, error_type: str) -> None:
        """Raise appropriate gRPC error based on error type."""
        error_config = self.error_injector.failure_modes.get(error_type, {})
        status_code = error_config.get("status_code", StatusCode.INTERNAL)

        error_messages = {
            "unavailable": "gRPC service unavailable",
            "deadline_exceeded": "gRPC deadline exceeded",
            "unauthenticated": "gRPC authentication failed",
            "permission_denied": "gRPC permission denied",
            "resource_exhausted": "gRPC resource exhausted",
            "failed_precondition": "gRPC precondition failed",
            "aborted": "gRPC operation aborted",
            "internal": "gRPC internal error",
            "data_loss": "gRPC data loss detected",
            "cancelled": "gRPC operation cancelled",
        }

        message = error_messages.get(error_type, "Unknown gRPC error")
        raise MockGRPCException(status_code, message)

    async def _mock_connect(self, daemon_address: str = "localhost:50051") -> None:
        """Mock gRPC connection to daemon."""
        await self._inject_grpc_error("connect")

        self.connected = True
        self.channel = f"mock_channel_{daemon_address}"

        self.operation_history.append({
            "operation": "connect",
            "daemon_address": daemon_address,
            "success": True
        })

    async def _mock_disconnect(self) -> None:
        """Mock gRPC disconnection."""
        self.connected = False
        self.channel = None

        self.operation_history.append({
            "operation": "disconnect",
            "success": True
        })

    async def _mock_ingest_document(self,
                                   collection_name: str,
                                   document_path: str,
                                   content: str,
                                   metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Mock document ingestion through gRPC."""
        await self._inject_grpc_error("ingestion")

        if not self.connected:
            raise MockGRPCException(StatusCode.UNAVAILABLE, "Not connected to daemon")

        self.operation_history.append({
            "operation": "ingest_document",
            "collection_name": collection_name,
            "document_path": document_path,
            "content_length": len(content),
            "metadata_keys": list(metadata.keys()) if metadata else []
        })

        return {
            "document_id": f"doc_{random.randint(1000, 9999)}",
            "collection": collection_name,
            "status": "ingested",
            "processing_time_ms": random.randint(50, 200),
            "chunk_count": max(1, len(content) // 500)
        }

    async def _mock_search_documents(self,
                                   collection_name: str,
                                   query: str,
                                   limit: int = 10,
                                   filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Mock document search through gRPC."""
        await self._inject_grpc_error("search")

        if not self.connected:
            raise MockGRPCException(StatusCode.UNAVAILABLE, "Not connected to daemon")

        self.operation_history.append({
            "operation": "search_documents",
            "collection_name": collection_name,
            "query": query,
            "limit": limit,
            "filters": filters
        })

        # Generate realistic search results
        results = []
        for i in range(min(limit, random.randint(0, 5))):
            results.append({
                "document_id": f"result_{i}",
                "score": 0.95 - (i * 0.1) + random.uniform(-0.05, 0.05),
                "content": f"Mock search result {i} for query: {query}",
                "metadata": {
                    "source": f"document_{i}.txt",
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            })

        return {
            "results": results,
            "total_count": len(results),
            "query": query,
            "collection": collection_name,
            "search_time_ms": random.randint(10, 100)
        }

    async def _mock_get_daemon_status(self) -> dict[str, Any]:
        """Mock daemon status retrieval."""
        await self._inject_grpc_error("status")

        self.operation_history.append({
            "operation": "get_daemon_status"
        })

        return {
            "status": "running" if self.connected else "disconnected",
            "version": "0.3.0dev0",
            "uptime_seconds": random.randint(100, 10000),
            "active_collections": random.randint(1, 5),
            "pending_operations": random.randint(0, 10),
            "memory_usage_mb": random.randint(50, 200),
            "cpu_usage_percent": random.randint(5, 30)
        }

    async def _mock_create_collection(self,
                                     collection_name: str,
                                     config: dict[str, Any]) -> dict[str, Any]:
        """Mock collection creation through gRPC."""
        await self._inject_grpc_error("collection_ops")

        if not self.connected:
            raise MockGRPCException(StatusCode.UNAVAILABLE, "Not connected to daemon")

        self.operation_history.append({
            "operation": "create_collection",
            "collection_name": collection_name,
            "config": config
        })

        return {
            "collection_name": collection_name,
            "status": "created",
            "vector_size": config.get("vector_size", 384),
            "distance_metric": config.get("distance_metric", "cosine")
        }

    async def _mock_delete_collection(self, collection_name: str) -> dict[str, Any]:
        """Mock collection deletion through gRPC."""
        await self._inject_grpc_error("collection_ops")

        if not self.connected:
            raise MockGRPCException(StatusCode.UNAVAILABLE, "Not connected to daemon")

        self.operation_history.append({
            "operation": "delete_collection",
            "collection_name": collection_name
        })

        return {
            "collection_name": collection_name,
            "status": "deleted"
        }

    async def _mock_watch_files(self,
                               paths: list[str],
                               event_types: list[str]) -> AsyncIterator[dict[str, Any]]:
        """Mock file watching through gRPC streaming."""
        await self._inject_grpc_error("watch_files")

        if not self.connected:
            raise MockGRPCException(StatusCode.UNAVAILABLE, "Not connected to daemon")

        self.operation_history.append({
            "operation": "watch_files",
            "paths": paths,
            "event_types": event_types
        })

        # Simulate file events
        for i in range(3):  # Generate a few mock events
            await asyncio.sleep(0.1)
            yield {
                "event_type": random.choice(event_types),
                "file_path": random.choice(paths),
                "timestamp": "2024-01-01T12:00:00Z",
                "event_id": f"event_{i}"
            }

    async def _mock_get_metadata(self,
                                document_id: str,
                                collection_name: str) -> dict[str, Any]:
        """Mock metadata retrieval through gRPC."""
        await self._inject_grpc_error("search")

        if not self.connected:
            raise MockGRPCException(StatusCode.UNAVAILABLE, "Not connected to daemon")

        self.operation_history.append({
            "operation": "get_metadata",
            "document_id": document_id,
            "collection_name": collection_name
        })

        return {
            "document_id": document_id,
            "collection": collection_name,
            "metadata": {
                "source": f"document_{document_id}.txt",
                "size": random.randint(1000, 10000),
                "created_at": "2024-01-01T12:00:00Z",
                "language": "en",
                "content_type": "text/plain"
            }
        }

    def get_operation_history(self) -> list[dict[str, Any]]:
        """Get history of gRPC operations."""
        return self.operation_history.copy()

    def reset_state(self) -> None:
        """Reset mock state."""
        self.operation_history.clear()
        self.connected = False
        self.channel = None
        self.error_injector.reset()


class GRPCServerMock:
    """Mock gRPC server for testing server-side functionality."""

    def __init__(self, error_injector: GRPCErrorInjector | None = None):
        self.error_injector = error_injector or GRPCErrorInjector()
        self.operation_history: list[dict[str, Any]] = []
        self.running = False
        self.port = None

        # Setup method mocks
        self._setup_server_methods()

    def _setup_server_methods(self):
        """Setup gRPC server method mocks."""
        self.start = AsyncMock(side_effect=self._mock_start)
        self.stop = AsyncMock(side_effect=self._mock_stop)
        self.serve = AsyncMock(side_effect=self._mock_serve)

    async def _mock_start(self, port: int = 50051) -> None:
        """Mock gRPC server start."""
        if self.error_injector.should_inject_error():
            raise OSError(f"Failed to start gRPC server on port {port}")

        self.running = True
        self.port = port

        self.operation_history.append({
            "operation": "start",
            "port": port,
            "success": True
        })

    async def _mock_stop(self, grace_period: float = 5.0) -> None:
        """Mock gRPC server stop."""
        self.running = False
        self.port = None

        self.operation_history.append({
            "operation": "stop",
            "grace_period": grace_period,
            "success": True
        })

    async def _mock_serve(self) -> None:
        """Mock gRPC server serving."""
        if not self.running:
            raise RuntimeError("Server not started")

        self.operation_history.append({
            "operation": "serve",
            "port": self.port
        })

        # Simulate serving (would run indefinitely in real server)
        await asyncio.sleep(0.1)

    def reset_state(self) -> None:
        """Reset server state."""
        self.operation_history.clear()
        self.running = False
        self.port = None
        self.error_injector.reset()


class DaemonCommunicationMock:
    """High-level mock for daemon communication patterns."""

    def __init__(self, error_injector: GRPCErrorInjector | None = None):
        self.error_injector = error_injector or GRPCErrorInjector()
        self.client = GRPCClientMock(error_injector)
        self.operation_history: list[dict[str, Any]] = []

        # Setup high-level method mocks
        self._setup_communication_methods()

    def _setup_communication_methods(self):
        """Setup high-level communication method mocks."""
        self.initialize_daemon = AsyncMock(side_effect=self._mock_initialize_daemon)
        self.shutdown_daemon = AsyncMock(side_effect=self._mock_shutdown_daemon)
        self.health_check = AsyncMock(side_effect=self._mock_health_check)
        self.bulk_ingest = AsyncMock(side_effect=self._mock_bulk_ingest)
        self.stream_search_results = AsyncMock(side_effect=self._mock_stream_search_results)

    async def _mock_initialize_daemon(self, config: dict[str, Any]) -> dict[str, Any]:
        """Mock daemon initialization."""
        await self.client.connect()

        self.operation_history.append({
            "operation": "initialize_daemon",
            "config": config
        })

        return {
            "status": "initialized",
            "daemon_version": "0.3.0dev0",
            "supported_features": ["ingestion", "search", "watching"]
        }

    async def _mock_shutdown_daemon(self) -> dict[str, Any]:
        """Mock daemon shutdown."""
        await self.client.disconnect()

        self.operation_history.append({
            "operation": "shutdown_daemon"
        })

        return {
            "status": "shutdown",
            "final_operations_count": len(self.client.operation_history)
        }

    async def _mock_health_check(self) -> dict[str, Any]:
        """Mock daemon health check."""
        status = await self.client.get_daemon_status()

        self.operation_history.append({
            "operation": "health_check",
            "daemon_status": status["status"]
        })

        return {
            "healthy": status["status"] == "running",
            "last_check": "2024-01-01T12:00:00Z",
            "response_time_ms": random.randint(1, 10)
        }

    async def _mock_bulk_ingest(self,
                               documents: list[dict[str, Any]],
                               collection_name: str) -> dict[str, Any]:
        """Mock bulk document ingestion."""
        results = []

        for doc in documents:
            try:
                result = await self.client.ingest_document(
                    collection_name=collection_name,
                    document_path=doc.get("path", "unknown"),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {})
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "document": doc.get("path", "unknown")})

        self.operation_history.append({
            "operation": "bulk_ingest",
            "collection_name": collection_name,
            "document_count": len(documents),
            "success_count": len([r for r in results if "error" not in r])
        })

        return {
            "total_documents": len(documents),
            "successful": len([r for r in results if "error" not in r]),
            "failed": len([r for r in results if "error" in r]),
            "results": results
        }

    async def _mock_stream_search_results(self,
                                         query: str,
                                         collections: list[str]) -> AsyncIterator[dict[str, Any]]:
        """Mock streaming search across multiple collections."""
        for collection in collections:
            try:
                results = await self.client.search_documents(
                    collection_name=collection,
                    query=query,
                    limit=5
                )

                yield {
                    "collection": collection,
                    "results": results["results"],
                    "status": "completed"
                }
            except Exception as e:
                yield {
                    "collection": collection,
                    "error": str(e),
                    "status": "failed"
                }

            await asyncio.sleep(0.05)  # Simulate streaming delay

    def get_operation_history(self) -> list[dict[str, Any]]:
        """Get combined operation history."""
        return self.operation_history + self.client.get_operation_history()

    def reset_state(self) -> None:
        """Reset all state."""
        self.operation_history.clear()
        self.client.reset_state()
        self.error_injector.reset()


def create_grpc_mock(
    component: str = "client",
    with_error_injection: bool = False,
    error_probability: float = 0.1
) -> GRPCClientMock | GRPCServerMock | DaemonCommunicationMock:
    """
    Create a gRPC mock component with optional error injection.

    Args:
        component: Type of mock to create ("client", "server", "daemon")
        with_error_injection: Enable error injection
        error_probability: Probability of errors (0.0 to 1.0)

    Returns:
        Configured gRPC mock instance
    """
    error_injector = None
    if with_error_injection:
        error_injector = GRPCErrorInjector()
        error_injector.configure_connection_issues(error_probability)
        error_injector.configure_auth_issues(error_probability)
        error_injector.configure_resource_issues(error_probability)

    if component == "client":
        return GRPCClientMock(error_injector)
    elif component == "server":
        return GRPCServerMock(error_injector)
    elif component == "daemon":
        return DaemonCommunicationMock(error_injector)
    else:
        raise ValueError(f"Unknown component type: {component}")


# Convenience functions for common scenarios
def create_basic_grpc_client() -> GRPCClientMock:
    """Create basic gRPC client mock without error injection."""
    return create_grpc_mock("client")


def create_failing_grpc_client(error_rate: float = 0.5) -> GRPCClientMock:
    """Create gRPC client mock with high failure rate."""
    return create_grpc_mock("client", with_error_injection=True, error_probability=error_rate)


def create_realistic_daemon_communication() -> DaemonCommunicationMock:
    """Create realistic daemon communication mock with occasional errors."""
    return create_grpc_mock("daemon", with_error_injection=True, error_probability=0.02)
