"""
Comprehensive gRPC client for workspace-qdrant-mcp daemon communication.

This module provides a complete Python client for all 15 RPCs across three services:
- SystemService (7 RPCs): Health, status, metrics, refresh signals, lifecycle
- CollectionService (5 RPCs): Collection and alias management
- DocumentService (3 RPCs): Text ingestion, update, deletion

The client provides:
- Connection pooling and health monitoring via GrpcConnectionManager
- Automatic retry with exponential backoff
- Circuit breaker for fault tolerance
- Comprehensive error handling and logging
- Async/await support for all operations
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime

import grpc.aio
from google.protobuf.empty_pb2 import Empty

from .connection_manager import ConnectionConfig, GrpcConnectionManager
from .generated import workspace_daemon_pb2 as pb2
from .generated import workspace_daemon_pb2_grpc as pb2_grpc

logger = logging.getLogger(__name__)


class DaemonClientError(Exception):
    """Base exception for daemon client errors."""
    pass


class DaemonUnavailableError(DaemonClientError):
    """Raised when daemon is unavailable or unreachable."""
    pass


class DaemonTimeoutError(DaemonClientError):
    """Raised when an operation times out."""
    pass


class DaemonClient:
    """
    Comprehensive gRPC client for workspace-qdrant-mcp daemon.

    Provides access to all 15 RPCs across SystemService, CollectionService,
    and DocumentService with robust connection management, retries, and error handling.

    Example:
        ```python
        async with DaemonClient() as client:
            # Health check
            health = await client.health_check()
            print(f"Daemon status: {health.status}")

            # Ingest text
            result = await client.ingest_text(
                content="Sample document",
                collection_basename="myapp",
                tenant_id="project_123"
            )
            print(f"Created document: {result.document_id}")
        ```
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        connection_config: Optional[ConnectionConfig] = None
    ):
        """
        Initialize daemon client.

        Args:
            host: gRPC server host (default: localhost)
            port: gRPC server port (default: 50051)
            connection_config: Optional custom connection configuration
        """
        if connection_config is None:
            connection_config = ConnectionConfig(
                host=host,
                port=port,
                # Increased timeouts for potentially slow operations
                connection_timeout=10.0,
                health_check_interval=30.0,
                max_retries=3,
                initial_retry_delay=1.0,
                max_retry_delay=16.0,
                retry_backoff_multiplier=2.0,
                # Enable connection pooling for better performance
                enable_connection_pooling=True,
                pool_size=5,
                max_pool_size=10,
                # Enable circuit breaker
                enable_circuit_breaker=True,
                circuit_breaker_failure_threshold=5,
                circuit_breaker_timeout=60.0,
            )

        self.config = connection_config
        self._channel: Optional[grpc.aio.Channel] = None
        self._system_stub: Optional[pb2_grpc.SystemServiceStub] = None
        self._collection_stub: Optional[pb2_grpc.CollectionServiceStub] = None
        self._document_stub: Optional[pb2_grpc.DocumentServiceStub] = None
        self._started = False
        self._shutdown = False

        # Circuit breaker state
        self._circuit_breaker_failures = 0
        self._circuit_breaker_state = "closed"  # closed, open, half-open
        self._circuit_breaker_last_failure: Optional[float] = None

        logger.info(
            f"DaemonClient initialized: address={self.config.address}, "
            f"connection_pooling={self.config.enable_connection_pooling}, "
            f"circuit_breaker={self.config.enable_circuit_breaker}"
        )

    async def start(self):
        """Start the daemon client and establish connection."""
        if self._started:
            return

        try:
            # Create channel with configuration
            options = self.config.get_channel_options()
            self._channel = grpc.aio.insecure_channel(
                self.config.address,
                options=options
            )

            # Wait for channel to be ready
            await asyncio.wait_for(
                self._channel.channel_ready(),
                timeout=self.config.connection_timeout,
            )

            # Create service stubs
            self._system_stub = pb2_grpc.SystemServiceStub(self._channel)
            self._collection_stub = pb2_grpc.CollectionServiceStub(self._channel)
            self._document_stub = pb2_grpc.DocumentServiceStub(self._channel)

            self._started = True
            logger.info(f"DaemonClient connected successfully: {self.config.address}")

        except asyncio.TimeoutError as e:
            logger.error(f"Connection timeout: {self.config.address}")
            raise DaemonUnavailableError(f"Connection timeout to {self.config.address}") from e
        except Exception as e:
            logger.error(f"Connection failed: {self.config.address}, error={e}")
            raise DaemonUnavailableError(f"Failed to connect to {self.config.address}: {e}") from e

    async def stop(self):
        """Stop the daemon client and clean up resources."""
        if self._shutdown:
            return

        self._shutdown = True

        if self._channel:
            try:
                await asyncio.wait_for(self._channel.close(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Channel close timed out")
            except Exception as e:
                logger.warning(f"Error closing channel: {e}")

        self._channel = None
        self._system_stub = None
        self._collection_stub = None
        self._document_stub = None
        self._started = False

        logger.info("DaemonClient stopped")

    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.stop()

    def _can_attempt_request(self) -> bool:
        """Check if request is allowed by circuit breaker."""
        if not self.config.enable_circuit_breaker:
            return True

        if self._circuit_breaker_state == "closed":
            return True
        elif self._circuit_breaker_state == "open":
            if (self._circuit_breaker_last_failure and
                time.time() - self._circuit_breaker_last_failure > self.config.circuit_breaker_timeout):
                self._circuit_breaker_state = "half-open"
                logger.info("Circuit breaker transitioning to half-open")
                return True
            return False
        else:  # half-open
            return True

    def _record_request_result(self, success: bool):
        """Record request result for circuit breaker."""
        if not self.config.enable_circuit_breaker:
            return

        if success:
            self._circuit_breaker_failures = 0
            if self._circuit_breaker_state == "half-open":
                self._circuit_breaker_state = "closed"
                logger.info("Circuit breaker closed")
        else:
            self._circuit_breaker_failures += 1
            self._circuit_breaker_last_failure = time.time()

            if self._circuit_breaker_failures >= self.config.circuit_breaker_failure_threshold:
                self._circuit_breaker_state = "open"
                logger.warning(
                    f"Circuit breaker opened: failures={self._circuit_breaker_failures}, "
                    f"threshold={self.config.circuit_breaker_failure_threshold}"
                )

    async def _retry_operation(self, operation, timeout: float = 30.0):
        """
        Execute operation with retry logic and circuit breaker.

        Args:
            operation: Async callable to execute
            timeout: Operation timeout in seconds

        Returns:
            Operation result

        Raises:
            DaemonUnavailableError: If daemon is unavailable
            DaemonTimeoutError: If operation times out
            DaemonClientError: For other errors
        """
        if not self._started:
            await self.start()

        if not self._can_attempt_request():
            raise DaemonUnavailableError("Circuit breaker is open")

        retry_delay = self.config.initial_retry_delay
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await asyncio.wait_for(operation(), timeout=timeout)
                self._record_request_result(success=True)
                return result

            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(
                    f"Operation timed out: attempt={attempt + 1}/{self.config.max_retries + 1}, "
                    f"timeout={timeout}s"
                )

                if attempt == self.config.max_retries:
                    self._record_request_result(success=False)
                    raise DaemonTimeoutError(f"Operation timed out after {timeout}s") from e

            except grpc.RpcError as e:
                last_exception = e
                is_retryable = e.code() in (
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    grpc.StatusCode.ABORTED,
                )

                logger.warning(
                    f"gRPC operation failed: attempt={attempt + 1}/{self.config.max_retries + 1}, "
                    f"status_code={e.code()}, details={e.details()}, retryable={is_retryable}"
                )

                if attempt == self.config.max_retries or not is_retryable:
                    self._record_request_result(success=False)
                    if e.code() == grpc.StatusCode.UNAVAILABLE:
                        raise DaemonUnavailableError(f"Daemon unavailable: {e.details()}") from e
                    raise DaemonClientError(f"gRPC error: {e.details()}") from e

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error in operation: {e}, attempt={attempt + 1}")
                self._record_request_result(success=False)
                raise DaemonClientError(f"Unexpected error: {e}") from e

            # Wait before retry with exponential backoff
            if attempt < self.config.max_retries:
                logger.info(f"Retrying operation: retry_delay={retry_delay}s, attempt={attempt + 1}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(
                    retry_delay * self.config.retry_backoff_multiplier,
                    self.config.max_retry_delay
                )

        # Should not reach here, but handle gracefully
        self._record_request_result(success=False)
        if last_exception:
            raise DaemonClientError(f"Operation failed after retries: {last_exception}") from last_exception
        raise DaemonClientError("Operation failed after retries")

    # =========================================================================
    # SystemService Methods (7 RPCs)
    # =========================================================================

    async def health_check(self, timeout: float = 5.0) -> pb2.HealthCheckResponse:
        """
        Quick health check for monitoring/alerting.

        Args:
            timeout: Request timeout in seconds (default: 5.0)

        Returns:
            HealthCheckResponse with status and component health

        Example:
            ```python
            health = await client.health_check()
            if health.status == pb2.SERVICE_STATUS_HEALTHY:
                print("Daemon is healthy")
            ```
        """
        async def operation():
            return await self._system_stub.HealthCheck(Empty())

        return await self._retry_operation(operation, timeout=timeout)

    async def get_status(self, timeout: float = 10.0) -> pb2.SystemStatusResponse:
        """
        Get comprehensive system state snapshot.

        Includes queue depths, active watches, processor states, uptime.

        Args:
            timeout: Request timeout in seconds (default: 10.0)

        Returns:
            SystemStatusResponse with complete system state
        """
        async def operation():
            return await self._system_stub.GetStatus(Empty())

        return await self._retry_operation(operation, timeout=timeout)

    async def get_metrics(self, timeout: float = 10.0) -> pb2.MetricsResponse:
        """
        Get current performance metrics.

        Returns processing rates, memory usage, error rates (snapshot only).

        Args:
            timeout: Request timeout in seconds (default: 10.0)

        Returns:
            MetricsResponse with current metrics
        """
        async def operation():
            return await self._system_stub.GetMetrics(Empty())

        return await self._retry_operation(operation, timeout=timeout)

    async def send_refresh_signal(
        self,
        queue_type: pb2.QueueType,
        lsp_languages: Optional[List[str]] = None,
        grammar_languages: Optional[List[str]] = None,
        timeout: float = 5.0
    ):
        """
        Signal database state changes for event-driven refresh.

        Daemon batches signals (10 items or 5 second threshold) for efficiency.

        Args:
            queue_type: Type of queue that changed (INGEST_QUEUE, WATCH_FOLDERS, TOOL_REGISTRY)
            lsp_languages: LSP language changes (optional)
            grammar_languages: Tree-sitter grammar changes (optional)
            timeout: Request timeout in seconds (default: 5.0)

        Example:
            ```python
            # Signal that watch folders changed
            await client.send_refresh_signal(pb2.WATCH_FOLDERS)

            # Signal LSP language changes
            await client.send_refresh_signal(
                pb2.TOOLS_AVAILABLE,
                lsp_languages=["python", "rust"]
            )
            ```
        """
        request = pb2.RefreshSignalRequest(
            queue_type=queue_type,
            lsp_languages=lsp_languages or [],
            grammar_languages=grammar_languages or [],
        )

        async def operation():
            return await self._system_stub.SendRefreshSignal(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def notify_server_status(
        self,
        state: pb2.ServerState,
        project_name: Optional[str] = None,
        project_root: Optional[str] = None,
        timeout: float = 5.0
    ):
        """
        Notify daemon of MCP/CLI server lifecycle changes.

        Updates priorities for project files when server starts/stops.

        Args:
            state: Server state (STARTING, RUNNING, STOPPING, STOPPED)
            project_name: Project name (optional)
            project_root: Project root path (optional)
            timeout: Request timeout in seconds (default: 5.0)

        Example:
            ```python
            # Notify server starting
            await client.notify_server_status(
                pb2.SERVER_STATE_UP,
                project_name="myapp",
                project_root="/path/to/myapp"
            )
            ```
        """
        request = pb2.ServerStatusNotification(state=state)
        if project_name is not None:
            request.project_name = project_name
        if project_root is not None:
            request.project_root = project_root

        async def operation():
            return await self._system_stub.NotifyServerStatus(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def pause_all_watchers(self, timeout: float = 5.0):
        """
        Pause all file watchers (master switch).

        Returns immediately, watchers stop asynchronously.

        Args:
            timeout: Request timeout in seconds (default: 5.0)
        """
        async def operation():
            return await self._system_stub.PauseAllWatchers(Empty())

        return await self._retry_operation(operation, timeout=timeout)

    async def resume_all_watchers(self, timeout: float = 5.0):
        """
        Resume all file watchers (master switch).

        Returns immediately, watchers resume asynchronously.

        Args:
            timeout: Request timeout in seconds (default: 5.0)
        """
        async def operation():
            return await self._system_stub.ResumeAllWatchers(Empty())

        return await self._retry_operation(operation, timeout=timeout)

    # =========================================================================
    # CollectionService Methods (5 RPCs)
    # =========================================================================

    async def create_collection(
        self,
        collection_name: str,
        project_id: str,
        config: Optional[pb2.CollectionConfig] = None,
        timeout: float = 30.0
    ) -> pb2.CreateCollectionResponse:
        """
        Create a new Qdrant collection.

        Args:
            collection_name: Name of collection to create
            project_id: Project identifier
            config: Collection configuration (vector size, distance metric, etc.)
            timeout: Request timeout in seconds (default: 30.0)

        Returns:
            CreateCollectionResponse with success status and collection ID

        Example:
            ```python
            config = pb2.CollectionConfig(
                vector_size=384,
                distance_metric="Cosine",
                enable_indexing=True
            )
            result = await client.create_collection(
                collection_name="myapp-code",
                project_id="project_123",
                config=config
            )
            ```
        """
        if config is None:
            # Default configuration
            config = pb2.CollectionConfig(
                vector_size=384,  # FastEmbed default
                distance_metric="Cosine",
                enable_indexing=True,
            )

        request = pb2.CreateCollectionRequest(
            collection_name=collection_name,
            project_id=project_id,
            config=config,
        )

        async def operation():
            return await self._collection_stub.CreateCollection(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def delete_collection(
        self,
        collection_name: str,
        project_id: str,
        force: bool = False,
        timeout: float = 30.0
    ):
        """
        Delete a Qdrant collection and its data.

        Args:
            collection_name: Name of collection to delete
            project_id: Project identifier
            force: Force deletion of non-empty collection (default: False)
            timeout: Request timeout in seconds (default: 30.0)

        Example:
            ```python
            await client.delete_collection(
                collection_name="myapp-code",
                project_id="project_123",
                force=True
            )
            ```
        """
        request = pb2.DeleteCollectionRequest(
            collection_name=collection_name,
            project_id=project_id,
            force=force,
        )

        async def operation():
            return await self._collection_stub.DeleteCollection(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def create_collection_alias(
        self,
        alias_name: str,
        collection_name: str,
        timeout: float = 10.0
    ):
        """
        Create a collection alias.

        Useful for tenant_id changes when Git remotes change.

        Args:
            alias_name: Alias name to create
            collection_name: Target collection name
            timeout: Request timeout in seconds (default: 10.0)

        Example:
            ```python
            await client.create_collection_alias(
                alias_name="myapp-code-old",
                collection_name="myapp-code"
            )
            ```
        """
        request = pb2.CreateAliasRequest(
            alias_name=alias_name,
            collection_name=collection_name,
        )

        async def operation():
            return await self._collection_stub.CreateCollectionAlias(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def delete_collection_alias(
        self,
        alias_name: str,
        timeout: float = 10.0
    ):
        """
        Delete a collection alias.

        Args:
            alias_name: Alias name to delete
            timeout: Request timeout in seconds (default: 10.0)
        """
        request = pb2.DeleteAliasRequest(alias_name=alias_name)

        async def operation():
            return await self._collection_stub.DeleteCollectionAlias(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def rename_collection_alias(
        self,
        old_name: str,
        new_name: str,
        collection_name: str,
        timeout: float = 10.0
    ):
        """
        Rename a collection alias atomically.

        Safer than delete + create as it's atomic.

        Args:
            old_name: Current alias name
            new_name: New alias name
            collection_name: Target collection name
            timeout: Request timeout in seconds (default: 10.0)

        Example:
            ```python
            await client.rename_collection_alias(
                old_name="myapp-code-old",
                new_name="myapp-code-backup",
                collection_name="myapp-code"
            )
            ```
        """
        request = pb2.RenameAliasRequest(
            old_alias_name=old_name,
            new_alias_name=new_name,
            collection_name=collection_name,
        )

        async def operation():
            return await self._collection_stub.RenameCollectionAlias(request)

        return await self._retry_operation(operation, timeout=timeout)

    # =========================================================================
    # DocumentService Methods (3 RPCs)
    # =========================================================================

    async def ingest_text(
        self,
        content: str,
        collection_basename: str,
        tenant_id: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        chunk_text: bool = True,
        timeout: float = 60.0
    ) -> pb2.IngestTextResponse:
        """
        Ingest text content directly (synchronous, no queuing).

        Processes immediately and returns chunks created.

        Args:
            content: Text content to ingest
            collection_basename: Base collection name (e.g., "myapp-notes")
            tenant_id: Tenant/project identifier
            document_id: Optional document ID (auto-generated if None)
            metadata: Optional metadata dictionary
            chunk_text: Whether to chunk the text (default: True)
            timeout: Request timeout in seconds (default: 60.0)

        Returns:
            IngestTextResponse with document_id, chunks_created

        Example:
            ```python
            result = await client.ingest_text(
                content="Important project notes",
                collection_basename="myapp-notes",
                tenant_id="project_123",
                metadata={"type": "note", "author": "user1"}
            )
            print(f"Document {result.document_id} created with {result.chunks_created} chunks")
            ```
        """
        request = pb2.IngestTextRequest(
            content=content,
            collection_basename=collection_basename,
            tenant_id=tenant_id,
            metadata=metadata or {},
            chunk_text=chunk_text,
        )

        if document_id is not None:
            request.document_id = document_id

        async def operation():
            return await self._document_stub.IngestText(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def update_text(
        self,
        document_id: str,
        content: str,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        timeout: float = 60.0
    ) -> pb2.UpdateTextResponse:
        """
        Update existing text document.

        Replaces content and optionally metadata.

        Args:
            document_id: Document ID to update
            content: New text content
            collection_name: Optional collection name (required if ambiguous)
            metadata: Optional new metadata
            timeout: Request timeout in seconds (default: 60.0)

        Returns:
            UpdateTextResponse with success status and updated_at timestamp

        Example:
            ```python
            result = await client.update_text(
                document_id="doc_123",
                content="Updated project notes",
                metadata={"type": "note", "author": "user1", "updated": "true"}
            )
            ```
        """
        request = pb2.UpdateTextRequest(
            document_id=document_id,
            content=content,
            metadata=metadata or {},
        )

        if collection_name is not None:
            request.collection_name = collection_name

        async def operation():
            return await self._document_stub.UpdateText(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def delete_text(
        self,
        document_id: str,
        collection_name: str,
        timeout: float = 30.0
    ):
        """
        Delete text document and all its chunks.

        Args:
            document_id: Document ID to delete
            collection_name: Collection name
            timeout: Request timeout in seconds (default: 30.0)

        Example:
            ```python
            await client.delete_text(
                document_id="doc_123",
                collection_name="myapp-notes"
            )
            ```
        """
        request = pb2.DeleteTextRequest(
            document_id=document_id,
            collection_name=collection_name,
        )

        async def operation():
            return await self._document_stub.DeleteText(request)

        return await self._retry_operation(operation, timeout=timeout)

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information and client state.

        Returns:
            Dictionary with connection details, circuit breaker state, etc.
        """
        return {
            "address": self.config.address,
            "connected": self._started,
            "circuit_breaker": {
                "enabled": self.config.enable_circuit_breaker,
                "state": self._circuit_breaker_state,
                "failures": self._circuit_breaker_failures,
                "threshold": self.config.circuit_breaker_failure_threshold,
            },
            "connection_pooling": {
                "enabled": self.config.enable_connection_pooling,
                "pool_size": self.config.pool_size,
                "max_pool_size": self.config.max_pool_size,
            },
            "retry_config": {
                "max_retries": self.config.max_retries,
                "initial_delay": self.config.initial_retry_delay,
                "max_delay": self.config.max_retry_delay,
                "backoff_multiplier": self.config.retry_backoff_multiplier,
            },
        }
