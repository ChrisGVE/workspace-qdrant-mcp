"""
Unified gRPC client for workspace-qdrant-mcp daemon communication.

This module provides a complete Python client for all daemon operations:
- NEW PROTOCOL (workspace_daemon):
  - SystemService (7 RPCs): Health, status, metrics, refresh signals, lifecycle
  - CollectionService (5 RPCs): Collection and alias management
  - DocumentService (3 RPCs): Text ingestion, update, deletion
- LEGACY PROTOCOL (IngestService):
  - Document processing, folder watching, search, configuration, memory operations

The client provides:
- Connection pooling and health monitoring
- Automatic retry with exponential backoff
- Circuit breaker for fault tolerance
- Service discovery for multi-instance daemons
- LLM access control validation
- Comprehensive error handling and logging
- Async/await support for all operations
"""

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import grpc.aio
from google.protobuf.empty_pb2 import Empty

from .connection_manager import ConnectionConfig
from .generated import workspace_daemon_pb2 as pb2
from .generated import workspace_daemon_pb2_grpc as pb2_grpc

# Legacy IngestService protocol imports
from .ingestion_pb2 import (
    # Memory operations
    AddMemoryRuleRequest,
    AddMemoryRuleResponse,
    CollectionInfo,
    ConfigureWatchRequest,
    ConfigureWatchResponse,
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    DeleteMemoryRuleRequest,
    DeleteMemoryRuleResponse,
    # Search operations
    ExecuteQueryRequest,
    ExecuteQueryResponse,
    GetCollectionInfoRequest,
    GetDocumentRequest,
    GetDocumentResponse,
    GetProcessingStatusRequest,
    # Status and monitoring
    GetStatsRequest,
    GetStatsResponse,
    HealthResponse,
    ListCollectionsRequest,
    ListCollectionsResponse,
    # Document management
    ListDocumentsRequest,
    ListDocumentsResponse,
    ListMemoryRulesRequest,
    ListMemoryRulesResponse,
    ListWatchesRequest,
    ListWatchesResponse,
    # Configuration management
    LoadConfigurationRequest,
    LoadConfigurationResponse,
    # Document processing
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ProcessFolderProgress,
    ProcessFolderRequest,
    ProcessingStatusResponse,
    SaveConfigurationRequest,
    SaveConfigurationResponse,
    SearchMemoryRulesRequest,
    SearchMemoryRulesResponse,
    SearchMode,
    # File watching
    StartWatchingRequest,
    StopWatchingRequest,
    StopWatchingResponse,
    SystemStatusResponse,
    ValidateConfigurationRequest,
    ValidateConfigurationResponse,
    WatchingUpdate,
    WatchStatus,
)
from .ingestion_pb2 import (
    CreateCollectionRequest as LegacyCreateCollectionRequest,
)
from .ingestion_pb2 import (
    CreateCollectionResponse as LegacyCreateCollectionResponse,
)
from .ingestion_pb2 import (
    DeleteCollectionRequest as LegacyDeleteCollectionRequest,
)
from .ingestion_pb2 import (
    DeleteCollectionResponse as LegacyDeleteCollectionResponse,
)
from .ingestion_pb2_grpc import IngestServiceStub

# LLM access control system
try:
    from ..core.llm_access_control import (
        LLMAccessControlError,
        validate_llm_collection_access,
    )
except ImportError:
    # Fallback for direct imports when not used as a package
    try:
        from llm_access_control import (
            LLMAccessControlError,
            validate_llm_collection_access,
        )
    except ImportError:
        # Define no-op fallbacks if LLM access control not available
        def validate_llm_collection_access(operation, collection, config):
            pass
        class LLMAccessControlError(Exception):
            pass

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


# Backward compatibility alias for legacy code
DaemonConnectionError = DaemonUnavailableError


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
        connection_config: ConnectionConfig | None = None,
        project_path: str | None = None,
        config_manager: Any | None = None
    ):
        """
        Initialize daemon client.

        Args:
            host: gRPC server host (default: localhost)
            port: gRPC server port (default: 50051)
            connection_config: Optional custom connection configuration
            project_path: Optional project path for service discovery context
            config_manager: Optional configuration manager for LLM access control
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
        self.project_path = project_path or os.getcwd()
        self.config_manager = config_manager

        self._channel: grpc.aio.Channel | None = None

        # New protocol stubs
        self._system_stub: pb2_grpc.SystemServiceStub | None = None
        self._collection_stub: pb2_grpc.CollectionServiceStub | None = None
        self._document_stub: pb2_grpc.DocumentServiceStub | None = None
        self._project_stub: pb2_grpc.ProjectServiceStub | None = None

        # Legacy protocol stub
        self._ingest_stub: IngestServiceStub | None = None

        self._started = False
        self._shutdown = False
        self._connected = False

        # Circuit breaker state
        self._circuit_breaker_failures = 0
        self._circuit_breaker_state = "closed"  # closed, open, half-open
        self._circuit_breaker_last_failure: float | None = None

        # Service discovery state
        self._discovered_endpoint: str | None = None

        logger.info(
            f"DaemonClient initialized: address={self.config.address}, "
            f"project_path={self.project_path}, "
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

            # Create service stubs (new protocol)
            self._system_stub = pb2_grpc.SystemServiceStub(self._channel)
            self._collection_stub = pb2_grpc.CollectionServiceStub(self._channel)
            self._document_stub = pb2_grpc.DocumentServiceStub(self._channel)
            self._project_stub = pb2_grpc.ProjectServiceStub(self._channel)

            # Create legacy protocol stub
            self._ingest_stub = IngestServiceStub(self._channel)

            self._started = True
            self._connected = True
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
        self._project_stub = None
        self._ingest_stub = None
        self._started = False
        self._connected = False

        logger.info("DaemonClient stopped")

    # Compatibility aliases for legacy code
    async def connect(self) -> None:
        """Alias for start() - for backward compatibility."""
        await self.start()

    async def disconnect(self) -> None:
        """Alias for stop() - for backward compatibility."""
        await self.stop()

    @asynccontextmanager
    async def connection(self):
        """Context manager for daemon connection - for backward compatibility."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

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
        lsp_languages: list[str] | None = None,
        grammar_languages: list[str] | None = None,
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
        project_name: str | None = None,
        project_root: str | None = None,
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
        config: pb2.CollectionConfig | None = None,
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

    async def create_collection_v2(
        self,
        collection_name: str,
        project_id: str,
        config: pb2.CollectionConfig | None = None,
        timeout: float = 30.0
    ) -> pb2.CreateCollectionResponse:
        """
        Alias for create_collection to maintain compatibility with server code.

        This is the same as create_collection() but with a version suffix
        to distinguish from the legacy IngestService method.
        """
        return await self.create_collection(
            collection_name=collection_name,
            project_id=project_id,
            config=config,
            timeout=timeout
        )

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

    async def delete_collection_v2(
        self,
        collection_name: str,
        project_id: str,
        force: bool = False,
        timeout: float = 30.0
    ):
        """
        Alias for delete_collection to maintain compatibility with server code.

        This is the same as delete_collection() but with a version suffix
        to distinguish from the legacy IngestService method.
        """
        return await self.delete_collection(
            collection_name=collection_name,
            project_id=project_id,
            force=force,
            timeout=timeout
        )

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
        document_id: str | None = None,
        metadata: dict[str, str] | None = None,
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
        collection_name: str | None = None,
        metadata: dict[str, str] | None = None,
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

    # =========================================================================
    # ProjectService Methods (5 RPCs)
    # =========================================================================

    async def register_project(
        self,
        path: str,
        project_id: str,
        name: str | None = None,
        git_remote: str | None = None,
        timeout: float = 10.0
    ) -> pb2.RegisterProjectResponse:
        """
        Register a project for high-priority processing.

        Called when MCP server starts for a project. Increments active session
        count and sets priority to HIGH.

        Args:
            path: Absolute path to project root
            project_id: 12-char hex identifier (from calculate_tenant_id)
            name: Human-readable project name (optional)
            git_remote: Git remote URL for normalization (optional)
            timeout: Request timeout in seconds (default: 10.0)

        Returns:
            RegisterProjectResponse with:
            - created: True if new project, False if existing
            - project_id: Confirmed project ID
            - priority: Current priority ("high", "normal", "low")
            - active_sessions: Number of active sessions

        Example:
            ```python
            result = await client.register_project(
                path="/path/to/myproject",
                project_id="abc123def456",
                name="My Project",
                git_remote="git@github.com:user/myproject.git"
            )
            print(f"Project registered: {result.project_id}, priority={result.priority}")
            ```
        """
        request = pb2.RegisterProjectRequest(
            path=path,
            project_id=project_id,
        )
        if name is not None:
            request.name = name
        if git_remote is not None:
            request.git_remote = git_remote

        async def operation():
            return await self._project_stub.RegisterProject(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def deprioritize_project(
        self,
        project_id: str,
        timeout: float = 10.0
    ) -> pb2.DeprioritizeProjectResponse:
        """
        Deprioritize a project (decrement session count).

        Called when MCP server stops. Decrements active session count.
        When count reaches 0, priority may be lowered.

        Args:
            project_id: 12-char hex identifier
            timeout: Request timeout in seconds (default: 10.0)

        Returns:
            DeprioritizeProjectResponse with:
            - success: Whether operation succeeded
            - remaining_sessions: Sessions after decrement
            - new_priority: Priority after demotion

        Example:
            ```python
            result = await client.deprioritize_project(project_id="abc123def456")
            print(f"Remaining sessions: {result.remaining_sessions}")
            ```
        """
        request = pb2.DeprioritizeProjectRequest(project_id=project_id)

        async def operation():
            return await self._project_stub.DeprioritizeProject(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def get_project_status(
        self,
        project_id: str,
        timeout: float = 10.0
    ) -> pb2.GetProjectStatusResponse:
        """
        Get current status of a project.

        Args:
            project_id: 12-char hex identifier
            timeout: Request timeout in seconds (default: 10.0)

        Returns:
            GetProjectStatusResponse with project details including:
            - found: Whether project exists
            - project_id, project_name, project_root
            - priority: Current priority level
            - active_sessions: Number of active sessions
            - last_active, registered_at timestamps
            - git_remote (optional)

        Example:
            ```python
            status = await client.get_project_status(project_id="abc123def456")
            if status.found:
                print(f"Project: {status.project_name}, priority={status.priority}")
            ```
        """
        request = pb2.GetProjectStatusRequest(project_id=project_id)

        async def operation():
            return await self._project_stub.GetProjectStatus(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def list_projects(
        self,
        priority_filter: str | None = None,
        active_only: bool = False,
        timeout: float = 10.0
    ) -> pb2.ListProjectsResponse:
        """
        List all registered projects with their status.

        Args:
            priority_filter: Filter by priority ("high", "normal", "low", or None for all)
            active_only: Only return projects with active_sessions > 0
            timeout: Request timeout in seconds (default: 10.0)

        Returns:
            ListProjectsResponse with:
            - projects: List of ProjectInfo objects
            - total_count: Total number of matching projects

        Example:
            ```python
            # List all high-priority projects
            result = await client.list_projects(priority_filter="high")
            for project in result.projects:
                print(f"{project.project_name}: {project.active_sessions} sessions")

            # List only active projects
            active = await client.list_projects(active_only=True)
            ```
        """
        request = pb2.ListProjectsRequest(active_only=active_only)
        if priority_filter is not None:
            request.priority_filter = priority_filter

        async def operation():
            return await self._project_stub.ListProjects(request)

        return await self._retry_operation(operation, timeout=timeout)

    async def heartbeat(
        self,
        project_id: str,
        timeout: float = 5.0
    ) -> pb2.HeartbeatResponse:
        """
        Send heartbeat to keep session alive.

        Called periodically by MCP servers to prevent session timeout.
        Default timeout is 60 seconds - heartbeats should be sent more frequently.

        Args:
            project_id: 12-char hex identifier
            timeout: Request timeout in seconds (default: 5.0)

        Returns:
            HeartbeatResponse with:
            - acknowledged: True if heartbeat was accepted
            - next_heartbeat_by: Deadline for next heartbeat

        Example:
            ```python
            result = await client.heartbeat(project_id="abc123def456")
            if result.acknowledged:
                # Schedule next heartbeat before next_heartbeat_by
                deadline = result.next_heartbeat_by.ToDatetime()
            ```
        """
        request = pb2.HeartbeatRequest(project_id=project_id)

        async def operation():
            return await self._project_stub.Heartbeat(request)

        return await self._retry_operation(operation, timeout=timeout)

    # =========================================================================
    # DEPRECATED: Legacy IngestService Methods
    # =========================================================================
    #
    # These methods use the deprecated IngestService protocol (ingestion.proto).
    # They will be REMOVED in the next major version.
    #
    # Migration Guide:
    # - process_document() → Use ingest_text() with DocumentService
    # - process_folder() → Use folder processing via SQLite watch configuration
    # - execute_query() → Use search() method with new protocol
    # - Collection operations → Use CollectionService methods
    # - Memory operations → Store in canonical 'memory' collection via ingest_text()
    #
    # See: docs/decisions/ADR-001-multi-tenant-architecture.md
    # =========================================================================

    def _warn_deprecated(self, method_name: str, replacement: str | None = None) -> None:
        """Log deprecation warning for legacy methods."""
        msg = f"DEPRECATED: {method_name}() uses legacy IngestService protocol. "
        if replacement:
            msg += f"Use {replacement} instead. "
        msg += "Will be removed in next major version."
        logger.warning(msg)

    async def process_document(
        self,
        file_path: str,
        collection: str,
        metadata: dict[str, str] | None = None,
        document_id: str | None = None,
        chunk_text: bool = True,
    ) -> ProcessDocumentResponse:
        """
        DEPRECATED: Process a single document via legacy IngestService.

        Use ingest_text() instead for the new protocol.
        """
        self._warn_deprecated("process_document", "ingest_text()")
        self._ensure_connected()

        # Apply LLM access control validation for collection writes
        if self.config_manager:
            try:
                validate_llm_collection_access('write', collection, self.config_manager)
            except LLMAccessControlError as e:
                logger.warning("LLM access control blocked collection write: %s", str(e))
                raise DaemonUnavailableError(f"Collection write blocked: {str(e)}") from e

        request = ProcessDocumentRequest(
            file_path=file_path,
            collection=collection,
            metadata=metadata or {},
            document_id=document_id,
            chunk_text=chunk_text,
        )

        return await self._ingest_stub.ProcessDocument(request)

    async def process_folder(
        self,
        folder_path: str,
        collection: str,
        include_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        recursive: bool = True,
        max_depth: int = 5,
        dry_run: bool = False,
        metadata: dict[str, str] | None = None,
    ) -> AsyncIterator[ProcessFolderProgress]:
        """
        DEPRECATED: Process all documents in a folder via legacy IngestService.

        Use SQLite watch folder configuration instead.
        """
        self._warn_deprecated("process_folder", "SQLite watch configuration")
        self._ensure_connected()

        # Apply LLM access control validation for collection writes (unless dry run)
        if not dry_run and self.config_manager:
            try:
                validate_llm_collection_access('write', collection, self.config_manager)
            except LLMAccessControlError as e:
                logger.warning("LLM access control blocked collection write: %s", str(e))
                raise DaemonUnavailableError(f"Collection write blocked: {str(e)}") from e

        request = ProcessFolderRequest(
            folder_path=folder_path,
            collection=collection,
            include_patterns=include_patterns or [],
            ignore_patterns=ignore_patterns or [],
            recursive=recursive,
            max_depth=max_depth,
            dry_run=dry_run,
            metadata=metadata or {},
        )

        async for progress in self._ingest_stub.ProcessFolder(request):
            yield progress

    async def start_watching(
        self,
        path: str,
        collection: str,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        auto_ingest: bool = True,
        recursive: bool = True,
        recursive_depth: int = -1,
        debounce_seconds: int = 5,
        update_frequency_ms: int = 1000,
        watch_id: str | None = None,
    ) -> AsyncIterator[WatchingUpdate]:
        """
        DEPRECATED: Start watching a folder for changes via legacy IngestService.

        Use SQLiteStateManager.save_watch_folder_config() instead.
        """
        self._warn_deprecated("start_watching", "SQLiteStateManager.save_watch_folder_config()")
        self._ensure_connected()

        request = StartWatchingRequest(
            path=path,
            collection=collection,
            patterns=patterns or [],
            ignore_patterns=ignore_patterns or [],
            auto_ingest=auto_ingest,
            recursive=recursive,
            recursive_depth=recursive_depth,
            debounce_seconds=debounce_seconds,
            update_frequency_ms=update_frequency_ms,
            watch_id=watch_id,
        )

        async for update in self._ingest_stub.StartWatching(request):
            yield update

    async def stop_watching(self, watch_id: str) -> StopWatchingResponse:
        """DEPRECATED: Stop watching a folder via legacy IngestService."""
        self._warn_deprecated("stop_watching", "SQLiteStateManager.remove_watch_folder_config()")
        self._ensure_connected()
        request = StopWatchingRequest(watch_id=watch_id)
        return await self._ingest_stub.StopWatching(request)

    async def list_watches(self, active_only: bool = False) -> ListWatchesResponse:
        """DEPRECATED: List all folder watches via legacy IngestService."""
        self._warn_deprecated("list_watches", "SQLiteStateManager.list_watch_folders()")
        self._ensure_connected()
        request = ListWatchesRequest(active_only=active_only)
        return await self._ingest_stub.ListWatches(request)

    async def configure_watch(
        self,
        watch_id: str,
        status: WatchStatus | None = None,
        patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
        auto_ingest: bool | None = None,
        recursive: bool | None = None,
        recursive_depth: int | None = None,
        debounce_seconds: int | None = None,
    ) -> ConfigureWatchResponse:
        """DEPRECATED: Configure an existing watch via legacy IngestService."""
        self._warn_deprecated("configure_watch", "SQLiteStateManager.save_watch_folder_config()")
        self._ensure_connected()

        request = ConfigureWatchRequest(
            watch_id=watch_id,
            status=status,
            patterns=patterns or [],
            ignore_patterns=ignore_patterns or [],
            auto_ingest=auto_ingest,
            recursive=recursive,
            recursive_depth=recursive_depth,
            debounce_seconds=debounce_seconds,
        )

        return await self._ingest_stub.ConfigureWatch(request)

    async def execute_query(
        self,
        query: str,
        collections: list[str] | None = None,
        mode: SearchMode = SearchMode.SEARCH_MODE_HYBRID,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> ExecuteQueryResponse:
        """DEPRECATED: Execute a search query via legacy IngestService."""
        self._warn_deprecated("execute_query", "search() method with new protocol")
        self._ensure_connected()

        request = ExecuteQueryRequest(
            query=query,
            collections=collections or [],
            mode=mode,
            limit=limit,
            score_threshold=score_threshold,
        )

        return await self._ingest_stub.ExecuteQuery(request)

    async def list_collections(
        self, include_stats: bool = False
    ) -> ListCollectionsResponse:
        """DEPRECATED: List all collections via legacy IngestService."""
        self._warn_deprecated("list_collections", "CollectionService.ListCollections()")
        self._ensure_connected()
        request = ListCollectionsRequest(include_stats=include_stats)
        return await self._ingest_stub.ListCollections(request)

    async def get_collection_info(
        self, collection_name: str, include_sample_documents: bool = False
    ) -> CollectionInfo:
        """DEPRECATED: Get information about a specific collection via legacy IngestService."""
        self._warn_deprecated("get_collection_info", "CollectionService.GetCollectionInfo()")
        self._ensure_connected()

        request = GetCollectionInfoRequest(
            collection_name=collection_name,
            include_sample_documents=include_sample_documents,
        )

        return await self._ingest_stub.GetCollectionInfo(request)

    async def create_collection_legacy(
        self,
        collection_name: str,
        description: str = "",
        metadata: dict[str, str] | None = None,
    ) -> LegacyCreateCollectionResponse:
        """DEPRECATED: Create a new collection via legacy IngestService."""
        self._warn_deprecated("create_collection_legacy", "CollectionService.CreateCollection()")
        self._ensure_connected()

        # Apply LLM access control validation for collection creation
        if self.config_manager:
            try:
                validate_llm_collection_access('create', collection_name, self.config_manager)
            except LLMAccessControlError as e:
                logger.warning("LLM access control blocked collection creation: %s", str(e))
                raise DaemonUnavailableError(f"Collection creation blocked: {str(e)}") from e

        request = LegacyCreateCollectionRequest(
            collection_name=collection_name,
            description=description,
            metadata=metadata or {},
        )

        return await self._ingest_stub.CreateCollection(request)

    async def delete_collection_legacy(
        self, collection_name: str, confirm: bool = False
    ) -> LegacyDeleteCollectionResponse:
        """DEPRECATED: Delete a collection via legacy IngestService."""
        self._warn_deprecated("delete_collection_legacy", "CollectionService.DeleteCollection()")
        self._ensure_connected()

        # Apply LLM access control validation for collection deletion
        if self.config_manager:
            try:
                validate_llm_collection_access('delete', collection_name, self.config_manager)
            except LLMAccessControlError as e:
                logger.warning("LLM access control blocked collection deletion: %s", str(e))
                raise DaemonUnavailableError(f"Collection deletion blocked: {str(e)}") from e

        request = LegacyDeleteCollectionRequest(
            collection_name=collection_name, confirm=confirm
        )

        return await self._ingest_stub.DeleteCollection(request)

    async def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """DEPRECATED: Check if a collection exists via legacy IngestService."""
        self._warn_deprecated("collection_exists", "CollectionService.GetCollectionInfo()")
        self._ensure_connected()

        try:
            response = await self.list_collections(include_stats=False)
            return any(c.name == collection_name for c in response.collections)
        except Exception as e:
            logger.warning(
                "Failed to check collection existence",
                collection_name=collection_name,
                error=str(e),
            )
            return False

    async def list_documents(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int = 0,
        filter_pattern: str = "",
    ) -> ListDocumentsResponse:
        """DEPRECATED: List documents in a collection via legacy IngestService."""
        self._warn_deprecated("list_documents", "DocumentService.ListDocuments()")
        self._ensure_connected()

        request = ListDocumentsRequest(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            filter_pattern=filter_pattern,
        )

        return await self._ingest_stub.ListDocuments(request)

    async def get_document(
        self, document_id: str, collection_name: str, include_content: bool = False
    ) -> GetDocumentResponse:
        """DEPRECATED: Get a specific document via legacy IngestService."""
        self._warn_deprecated("get_document", "DocumentService.GetDocument()")
        self._ensure_connected()

        request = GetDocumentRequest(
            document_id=document_id,
            collection_name=collection_name,
            include_content=include_content,
        )

        return await self._ingest_stub.GetDocument(request)

    async def delete_document(
        self, document_id: str, collection_name: str
    ) -> DeleteDocumentResponse:
        """DEPRECATED: Delete a document via legacy IngestService."""
        self._warn_deprecated("delete_document", "DocumentService.DeleteDocument()")
        self._ensure_connected()

        request = DeleteDocumentRequest(
            document_id=document_id, collection_name=collection_name
        )

        return await self._ingest_stub.DeleteDocument(request)

    async def load_configuration(
        self, config_path: str | None = None
    ) -> LoadConfigurationResponse:
        """DEPRECATED: Load configuration from daemon via legacy IngestService."""
        self._warn_deprecated("load_configuration", "SystemService.GetConfig()")
        self._ensure_connected()
        request = LoadConfigurationRequest(config_path=config_path)
        return await self._ingest_stub.LoadConfiguration(request)

    async def save_configuration(
        self, config_yaml: str, target_path: str
    ) -> SaveConfigurationResponse:
        """DEPRECATED: Save configuration to daemon via legacy IngestService."""
        self._warn_deprecated("save_configuration", "SystemService.SetConfig()")
        self._ensure_connected()

        request = SaveConfigurationRequest(
            config_yaml=config_yaml, target_path=target_path
        )

        return await self._ingest_stub.SaveConfiguration(request)

    async def validate_configuration(
        self, config_yaml: str
    ) -> ValidateConfigurationResponse:
        """DEPRECATED: Validate configuration via legacy IngestService."""
        self._warn_deprecated("validate_configuration")
        self._ensure_connected()
        request = ValidateConfigurationRequest(config_yaml=config_yaml)
        return await self._ingest_stub.ValidateConfiguration(request)

    async def add_memory_rule(
        self,
        category: str,
        name: str,
        rule_text: str,
        authority: str | None = None,
        scope: list[str] | None = None,
        source: str | None = None,
    ) -> AddMemoryRuleResponse:
        """DEPRECATED: Add a memory rule via legacy IngestService."""
        self._warn_deprecated("add_memory_rule", "ingest_text() with collection='memory'")
        self._ensure_connected()

        request = AddMemoryRuleRequest(
            category=category,
            name=name,
            rule_text=rule_text,
            authority=authority,
            scope=scope or [],
            source=source,
        )

        return await self._ingest_stub.AddMemoryRule(request)

    async def list_memory_rules(
        self,
        category: str | None = None,
        authority: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ListMemoryRulesResponse:
        """DEPRECATED: List memory rules via legacy IngestService."""
        self._warn_deprecated("list_memory_rules", "search on 'memory' collection")
        self._ensure_connected()

        request = ListMemoryRulesRequest(
            category=category, authority=authority, limit=limit, offset=offset
        )

        return await self._ingest_stub.ListMemoryRules(request)

    async def delete_memory_rule(self, rule_id: str) -> DeleteMemoryRuleResponse:
        """DEPRECATED: Delete a memory rule via legacy IngestService."""
        self._warn_deprecated("delete_memory_rule", "DocumentService.DeleteDocument() on 'memory'")
        self._ensure_connected()
        request = DeleteMemoryRuleRequest(rule_id=rule_id)
        return await self._ingest_stub.DeleteMemoryRule(request)

    async def search_memory_rules(
        self,
        query: str,
        category: str | None = None,
        authority: str | None = None,
        limit: int = 10,
    ) -> SearchMemoryRulesResponse:
        """DEPRECATED: Search memory rules via legacy IngestService."""
        self._warn_deprecated("search_memory_rules", "search() on 'memory' collection")
        self._ensure_connected()

        request = SearchMemoryRulesRequest(
            query=query, category=category, authority=authority, limit=limit
        )

        return await self._ingest_stub.SearchMemoryRules(request)

    async def get_stats(
        self, include_collection_stats: bool = True, include_watch_stats: bool = True
    ) -> GetStatsResponse:
        """DEPRECATED: Get daemon statistics via legacy IngestService."""
        self._warn_deprecated("get_stats", "SystemService.GetStatus()")
        self._ensure_connected()

        request = GetStatsRequest(
            include_collection_stats=include_collection_stats,
            include_watch_stats=include_watch_stats,
        )

        return await self._ingest_stub.GetStats(request)

    async def get_processing_status(
        self, include_history: bool = False, history_limit: int = 50
    ) -> ProcessingStatusResponse:
        """DEPRECATED: Get processing status via legacy IngestService."""
        self._warn_deprecated("get_processing_status", "SystemService.GetStatus()")
        self._ensure_connected()

        request = GetProcessingStatusRequest(
            include_history=include_history, history_limit=history_limit
        )

        return await self._ingest_stub.GetProcessingStatus(request)

    async def get_system_status(self) -> SystemStatusResponse:
        """DEPRECATED: Get system status via legacy IngestService."""
        self._warn_deprecated("get_system_status", "SystemService.GetStatus()")
        self._ensure_connected()
        return await self._ingest_stub.GetSystemStatus(Empty())

    async def health_check_legacy(self) -> HealthResponse:
        """DEPRECATED: Perform health check via legacy IngestService."""
        self._warn_deprecated("health_check_legacy", "health_check()")
        self._ensure_connected()
        return await self._ingest_stub.HealthCheck(Empty())

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _ensure_connected(self) -> None:
        """Ensure client is connected to daemon - for legacy compatibility."""
        if not self._connected or not self._ingest_stub:
            raise DaemonUnavailableError(
                "Cannot perform operation: daemon not connected.\n"
                "\n"
                "To resolve this issue:\n"
                "1. Start the daemon: wqm admin start-daemon\n"
                "2. Check daemon status: wqm admin status\n"
                "3. Verify Qdrant is running at localhost:6333\n"
                "\n"
                "For more help: wqm admin diagnostics"
            )

    def get_connection_info(self) -> dict[str, Any]:
        """
        Get connection information and client state.

        Returns:
            Dictionary with connection details, circuit breaker state, service discovery info, etc.
        """
        info = {
            "address": self.config.address,
            "connected": self._started,
            "project_path": self.project_path,
            "discovery_used": self._discovered_endpoint is not None,
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
            "discovery_strategy": "configuration"
        }

        return info


# =========================================================================
# Global client instance and helper functions (backward compatibility)
# =========================================================================

_daemon_client: DaemonClient | None = None


def get_daemon_client(
    config_manager=None,
    project_path: str | None = None
) -> DaemonClient:
    """
    Get the global daemon client instance with optional project context.

    Args:
        config_manager: Optional configuration manager
        project_path: Optional project path for service discovery

    Returns:
        DaemonClient instance
    """
    global _daemon_client

    if _daemon_client is None or config_manager is not None or project_path is not None:
        _daemon_client = DaemonClient(
            config_manager=config_manager,
            project_path=project_path
        )

    return _daemon_client


async def with_daemon_client(
    operation,
    config_manager=None,
    project_path: str | None = None
):
    """
    Execute an operation with a connected daemon client.

    Args:
        operation: Async callable that takes a DaemonClient as argument
        config_manager: Optional configuration manager
        project_path: Optional project path for service discovery

    Returns:
        Result of the operation

    Example:
        ```python
        async def my_operation(client: DaemonClient):
            return await client.health_check()

        result = await with_daemon_client(my_operation)
        ```
    """
    client = get_daemon_client(config_manager, project_path)
    async with client.connection():
        return await operation(client)


def create_project_client(project_path: str, config_manager=None) -> DaemonClient:
    """
    Create a new daemon client for a specific project path.

    Args:
        project_path: Project root path
        config_manager: Optional configuration manager

    Returns:
        DaemonClient instance configured for the project
    """
    return DaemonClient(config_manager=config_manager, project_path=project_path)
