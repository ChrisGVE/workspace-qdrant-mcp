"""
Unified gRPC client for workspace-qdrant-mcp daemon communication.

This module provides a single interface for all components (CLI, MCP server, web UI)
to communicate with the daemon, eliminating direct Qdrant client usage and code duplication.
Includes automatic service discovery for multi-instance daemon support.

Enhanced with DocumentService and CollectionService support for Task 375.1.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import grpc
from google.protobuf.empty_pb2 import Empty
from google.protobuf.timestamp_pb2 import Timestamp

# Legacy IngestService imports (to be deprecated)
from ..grpc.ingestion_pb2 import (
    # Memory operations
    AddMemoryRuleRequest,
    AddMemoryRuleResponse,
    CollectionInfo,
    ConfigureWatchRequest,
    ConfigureWatchResponse,
    CreateCollectionRequest,
    CreateCollectionResponse,
    DeleteCollectionRequest,
    DeleteCollectionResponse,
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
    HealthStatus,
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
    # Enums
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
from ..grpc.ingestion_pb2_grpc import IngestServiceStub

# New workspace_daemon protocol imports
from ..grpc.generated.workspace_daemon_pb2 import (
    CreateCollectionRequest as NewCreateCollectionRequest,
    CreateCollectionResponse as NewCreateCollectionResponse,
    DeleteCollectionRequest as NewDeleteCollectionRequest,
    IngestTextRequest,
    IngestTextResponse,
    UpdateTextRequest,
    UpdateTextResponse,
    DeleteTextRequest,
    CollectionConfig,
    CreateAliasRequest,
    DeleteAliasRequest,
    RenameAliasRequest,
)
from ..grpc.generated.workspace_daemon_pb2_grpc import (
    DocumentServiceStub,
    CollectionServiceStub,
)

from loguru import logger
from .config import get_config_manager
# TEMP: Comment out service_discovery import to fix syntax errors
# from .service_discovery import discover_daemon_endpoint, ServiceEndpoint

# Import LLM access control system
try:
    from .llm_access_control import validate_llm_collection_access, LLMAccessControlError
except ImportError:
    # Fallback for direct imports when not used as a package
    from llm_access_control import validate_llm_collection_access, LLMAccessControlError

# logger imported from loguru


class DaemonConnectionError(Exception):
    """Raised when daemon connection fails."""

    pass


class DaemonClient:
    """
    Unified gRPC client for daemon communication with automatic service discovery.

    Provides a single interface for all workspace-qdrant-mcp operations,
    replacing direct Qdrant client usage throughout the codebase.
    Automatically discovers the correct daemon instance for the current project context.

    Enhanced with DocumentService and CollectionService support (Task 375.1).
    """

    def __init__(self, config_manager=None, project_path: Optional[str] = None):
        """Initialize daemon client with configuration and optional project context."""
        self.config = config_manager or get_config_manager()
        self.project_path = project_path or os.getcwd()
        self.channel: Optional[grpc.aio.Channel] = None

        # Legacy stub (to be deprecated)
        self.stub: Optional[IngestServiceStub] = None

        # New service stubs
        self.document_service: Optional[DocumentServiceStub] = None
        self.collection_service: Optional[CollectionServiceStub] = None

        self._connected = False
        self._discovered_endpoint: Optional[Any] = None  # ServiceEndpoint type when available

    async def connect(self) -> None:
        """Establish connection to daemon using service discovery."""
        if self._connected:
            return

        # First, attempt service discovery for project-specific daemon
        grpc_host = self.config.get("grpc.host", "localhost")
        grpc_port = self.config.get("grpc.port", 50051)
        preferred_endpoint = (grpc_host, grpc_port)

        logger.info("Attempting service discovery for daemon connection",
                   project_path=self.project_path)

        # Try to discover daemon endpoint for this project
        # TEMP: Service discovery disabled until implementation complete
        # discovered_endpoint = await discover_daemon_endpoint(
        #     self.project_path,
        #     preferred_endpoint
        # )
        discovered_endpoint = None

        if discovered_endpoint:
            address = discovered_endpoint.address
            self._discovered_endpoint = discovered_endpoint
            logger.info("Using discovered daemon endpoint",
                       address=address,
                       project_id=discovered_endpoint.project_id)
        else:
            # Fallback to configured endpoint
            address = f"{grpc_host}:{grpc_port}"
            logger.warning("Service discovery failed, using configured endpoint",
                          address=address)

        try:
            # Get gRPC configuration with proper defaults
            grpc_config_dict = self.config.get("grpc", {})
            max_message_size_mb = grpc_config_dict.get("max_message_size_mb", 100)

            # Create channel with appropriate options
            options = [
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 5000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
                ("grpc.http2.min_time_between_pings_ms", 10000),
                ("grpc.http2.min_ping_interval_without_data_ms", 300000),
                (
                    "grpc.max_receive_message_length",
                    max_message_size_mb * 1024 * 1024,
                ),
                (
                    "grpc.max_send_message_length",
                    max_message_size_mb * 1024 * 1024,
                ),
            ]

            self.channel = grpc.aio.insecure_channel(address, options=options)

            # Initialize all service stubs
            self.stub = IngestServiceStub(self.channel)
            self.document_service = DocumentServiceStub(self.channel)
            self.collection_service = CollectionServiceStub(self.channel)

            # Test connection with health check
            await self.health_check()
            self._connected = True
            logger.info(f"Connected to daemon at {address}")

        except Exception as e:
            error_msg = (
                f"Failed to connect to daemon at {address}: {e}\n"
                "\n"
                "Troubleshooting steps:\n"
                "1. Check if daemon is running: wqm admin status\n"
                "2. Start daemon if needed: wqm admin start-daemon\n"
                "3. Verify Qdrant is accessible at localhost:6333\n"
                "4. Check firewall/network settings\n"
            )
            logger.error(error_msg)
            if self.channel:
                await self.channel.close()
                self.channel = None
                self.stub = None
                self.document_service = None
                self.collection_service = None
            raise DaemonConnectionError(f"Failed to connect to daemon: {e}")

    async def disconnect(self) -> None:
        """Close connection to daemon."""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
            self.document_service = None
            self.collection_service = None
            self._connected = False
            self._discovered_endpoint = None
            logger.info("Disconnected from daemon")

    @asynccontextmanager
    async def connection(self):
        """Context manager for daemon connection."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    def _ensure_connected(self) -> None:
        """Ensure client is connected to daemon."""
        if not self._connected or not self.stub:
            raise DaemonConnectionError(
                "Cannot perform operation: daemon not connected.\n"
                "\n"
                "To resolve this issue:\n"
                "1. Start the daemon: wqm admin start-daemon\n"
                "2. Check daemon status: wqm admin status\n"
                "3. Verify Qdrant is running at localhost:6333\n"
                "\n"
                "For more help: wqm admin diagnostics"
            )

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the current daemon connection."""
        info = {
            "connected": self._connected,
            "project_path": self.project_path,
            "discovery_used": self._discovered_endpoint is not None,
        }

        if self._discovered_endpoint:
            info.update({
                "endpoint": self._discovered_endpoint.address,
                "project_id": self._discovered_endpoint.project_id,
                "service_name": self._discovered_endpoint.service_name,
                "health_status": self._discovered_endpoint.health_status,
                "discovery_strategy": "service_discovery"
            })
        else:
            grpc_host = self.config.get("grpc.host", "localhost")
            grpc_port = self.config.get("grpc.port", 50051)
            info.update({
                "endpoint": f"{grpc_host}:{grpc_port}",
                "discovery_strategy": "configuration"
            })

        return info

    # =========================================================================
    # NEW: DocumentService methods (Task 375.1)
    # =========================================================================

    async def ingest_text(
        self,
        content: str,
        collection_basename: str,
        tenant_id: str,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        chunk_text: bool = True,
    ) -> IngestTextResponse:
        """
        Ingest text content directly via DocumentService.

        This method provides synchronous text ingestion without file-based processing.
        Use cases: user input, chat snippets, scraped web content, manual notes.

        Args:
            content: The text content to ingest
            collection_basename: Base collection name (e.g., "memory", "scratchbook")
            tenant_id: Multi-tenant identifier for collection scoping
            document_id: Optional custom document ID (generated if omitted)
            metadata: Additional metadata to attach to the document
            chunk_text: Whether to chunk the text (default: True)

        Returns:
            IngestTextResponse with document_id, success status, and chunks_created count

        Raises:
            DaemonConnectionError: If daemon is not connected or ingestion fails
        """
        self._ensure_connected()

        if not self.document_service:
            raise DaemonConnectionError(
                "DocumentService not available. Daemon may not support new protocol."
            )

        # Apply LLM access control validation
        full_collection_name = f"{collection_basename}_{tenant_id}"
        try:
            validate_llm_collection_access('write', full_collection_name, self.config)
        except LLMAccessControlError as e:
            logger.warning("LLM access control blocked text ingestion: %s", str(e))
            raise DaemonConnectionError(f"Text ingestion blocked: {str(e)}") from e

        request = IngestTextRequest(
            content=content,
            collection_basename=collection_basename,
            tenant_id=tenant_id,
            document_id=document_id or "",
            metadata=metadata or {},
            chunk_text=chunk_text,
        )

        try:
            response = await self.document_service.IngestText(request)
            logger.info(
                "Text ingested successfully",
                document_id=response.document_id,
                chunks_created=response.chunks_created,
                collection=full_collection_name,
            )
            return response
        except grpc.RpcError as e:
            error_msg = f"Failed to ingest text: {e.code()}: {e.details()}"
            logger.error(error_msg)
            raise DaemonConnectionError(error_msg) from e

    async def update_text(
        self,
        document_id: str,
        content: str,
        collection_name: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> UpdateTextResponse:
        """
        Update previously ingested text document via DocumentService.

        Args:
            document_id: The document ID returned from ingest_text()
            content: New text content
            collection_name: Optional new collection name (for moving documents)
            metadata: Updated metadata

        Returns:
            UpdateTextResponse with success status and updated_at timestamp

        Raises:
            DaemonConnectionError: If daemon is not connected or update fails
        """
        self._ensure_connected()

        if not self.document_service:
            raise DaemonConnectionError(
                "DocumentService not available. Daemon may not support new protocol."
            )

        request = UpdateTextRequest(
            document_id=document_id,
            content=content,
            collection_name=collection_name or "",
            metadata=metadata or {},
        )

        try:
            response = await self.document_service.UpdateText(request)
            logger.info(
                "Text updated successfully",
                document_id=document_id,
                collection=collection_name,
            )
            return response
        except grpc.RpcError as e:
            error_msg = f"Failed to update text: {e.code()}: {e.details()}"
            logger.error(error_msg)
            raise DaemonConnectionError(error_msg) from e

    async def delete_text(
        self,
        document_id: str,
        collection_name: str,
    ) -> None:
        """
        Delete ingested text document via DocumentService.

        Args:
            document_id: The document ID to delete
            collection_name: Collection containing the document

        Raises:
            DaemonConnectionError: If daemon is not connected or deletion fails
        """
        self._ensure_connected()

        if not self.document_service:
            raise DaemonConnectionError(
                "DocumentService not available. Daemon may not support new protocol."
            )

        request = DeleteTextRequest(
            document_id=document_id,
            collection_name=collection_name,
        )

        try:
            await self.document_service.DeleteText(request)
            logger.info(
                "Text deleted successfully",
                document_id=document_id,
                collection=collection_name,
            )
        except grpc.RpcError as e:
            error_msg = f"Failed to delete text: {e.code()}: {e.details()}"
            logger.error(error_msg)
            raise DaemonConnectionError(error_msg) from e

    # =========================================================================
    # NEW: CollectionService methods (Task 375.1)
    # =========================================================================

    async def create_collection_v2(
        self,
        collection_name: str,
        project_id: str = "",
        vector_size: int = 384,
        distance_metric: str = "Cosine",
        enable_indexing: bool = True,
        metadata_schema: Optional[Dict[str, str]] = None,
    ) -> NewCreateCollectionResponse:
        """
        Create a collection via CollectionService (new protocol).

        This method uses the new workspace_daemon protocol for collection creation.
        For legacy compatibility, use create_collection() method.

        Args:
            collection_name: Name of the collection to create
            project_id: Optional project association
            vector_size: Embedding vector dimension (default: 384 for all-MiniLM-L6-v2)
            distance_metric: "Cosine", "Euclidean", or "Dot" (default: "Cosine")
            enable_indexing: Enable HNSW indexing (default: True)
            metadata_schema: Expected metadata fields

        Returns:
            NewCreateCollectionResponse with success status and collection_id

        Raises:
            DaemonConnectionError: If daemon is not connected or creation fails
        """
        self._ensure_connected()

        if not self.collection_service:
            raise DaemonConnectionError(
                "CollectionService not available. Daemon may not support new protocol."
            )

        # Apply LLM access control validation
        try:
            validate_llm_collection_access('create', collection_name, self.config)
        except LLMAccessControlError as e:
            logger.warning("LLM access control blocked collection creation: %s", str(e))
            raise DaemonConnectionError(f"Collection creation blocked: {str(e)}") from e

        config = CollectionConfig(
            vector_size=vector_size,
            distance_metric=distance_metric,
            enable_indexing=enable_indexing,
            metadata_schema=metadata_schema or {},
        )

        request = NewCreateCollectionRequest(
            collection_name=collection_name,
            project_id=project_id,
            config=config,
        )

        try:
            response = await self.collection_service.CreateCollection(request)
            if response.success:
                logger.info(
                    "Collection created successfully",
                    collection_name=collection_name,
                    collection_id=response.collection_id,
                )
            else:
                logger.error(
                    "Collection creation failed",
                    collection_name=collection_name,
                    error=response.error_message,
                )
            return response
        except grpc.RpcError as e:
            error_msg = f"Failed to create collection: {e.code()}: {e.details()}"
            logger.error(error_msg)
            raise DaemonConnectionError(error_msg) from e

    async def delete_collection_v2(
        self,
        collection_name: str,
        project_id: str = "",
        force: bool = False,
    ) -> None:
        """
        Delete a collection via CollectionService (new protocol).

        This method uses the new workspace_daemon protocol for collection deletion.
        For legacy compatibility, use delete_collection() method.

        Args:
            collection_name: Name of the collection to delete
            project_id: Project ID for validation
            force: Skip confirmation checks (default: False)

        Raises:
            DaemonConnectionError: If daemon is not connected or deletion fails
        """
        self._ensure_connected()

        if not self.collection_service:
            raise DaemonConnectionError(
                "CollectionService not available. Daemon may not support new protocol."
            )

        # Apply LLM access control validation
        try:
            validate_llm_collection_access('delete', collection_name, self.config)
        except LLMAccessControlError as e:
            logger.warning("LLM access control blocked collection deletion: %s", str(e))
            raise DaemonConnectionError(f"Collection deletion blocked: {str(e)}") from e

        request = NewDeleteCollectionRequest(
            collection_name=collection_name,
            project_id=project_id,
            force=force,
        )

        try:
            await self.collection_service.DeleteCollection(request)
            logger.info(
                "Collection deleted successfully",
                collection_name=collection_name,
            )
        except grpc.RpcError as e:
            error_msg = f"Failed to delete collection: {e.code()}: {e.details()}"
            logger.error(error_msg)
            raise DaemonConnectionError(error_msg) from e

    async def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        """
        Check if a collection exists.

        This is a convenience method that queries Qdrant directly via the legacy
        ListCollections RPC. For the new protocol, MCP server queries Qdrant directly.

        Args:
            collection_name: Name of the collection to check

        Returns:
            True if collection exists, False otherwise

        Raises:
            DaemonConnectionError: If daemon is not connected or query fails
        """
        self._ensure_connected()

        try:
            # Use legacy method since new protocol expects MCP to query Qdrant directly
            response = await self.list_collections(include_stats=False)
            return any(c.name == collection_name for c in response.collections)
        except Exception as e:
            logger.warning(
                "Failed to check collection existence",
                collection_name=collection_name,
                error=str(e),
            )
            return False

    async def create_collection_alias(
        self,
        alias_name: str,
        collection_name: str,
    ) -> None:
        """
        Create a collection alias via CollectionService.

        Use case: Tenant ID changes - create alias to maintain compatibility.

        Args:
            alias_name: The new alias name
            collection_name: The collection this alias points to

        Raises:
            DaemonConnectionError: If daemon is not connected or creation fails
        """
        self._ensure_connected()

        if not self.collection_service:
            raise DaemonConnectionError(
                "CollectionService not available. Daemon may not support new protocol."
            )

        request = CreateAliasRequest(
            alias_name=alias_name,
            collection_name=collection_name,
        )

        try:
            await self.collection_service.CreateCollectionAlias(request)
            logger.info(
                "Collection alias created successfully",
                alias_name=alias_name,
                collection_name=collection_name,
            )
        except grpc.RpcError as e:
            error_msg = f"Failed to create collection alias: {e.code()}: {e.details()}"
            logger.error(error_msg)
            raise DaemonConnectionError(error_msg) from e

    async def delete_collection_alias(
        self,
        alias_name: str,
    ) -> None:
        """
        Delete a collection alias via CollectionService.

        Args:
            alias_name: The alias name to delete

        Raises:
            DaemonConnectionError: If daemon is not connected or deletion fails
        """
        self._ensure_connected()

        if not self.collection_service:
            raise DaemonConnectionError(
                "CollectionService not available. Daemon may not support new protocol."
            )

        request = DeleteAliasRequest(alias_name=alias_name)

        try:
            await self.collection_service.DeleteCollectionAlias(request)
            logger.info(
                "Collection alias deleted successfully",
                alias_name=alias_name,
            )
        except grpc.RpcError as e:
            error_msg = f"Failed to delete collection alias: {e.code()}: {e.details()}"
            logger.error(error_msg)
            raise DaemonConnectionError(error_msg) from e

    async def rename_collection_alias(
        self,
        old_alias_name: str,
        new_alias_name: str,
        collection_name: str,
    ) -> None:
        """
        Atomically rename a collection alias via CollectionService.

        This is safer than delete + create as it's atomic.

        Args:
            old_alias_name: Current alias name
            new_alias_name: New alias name
            collection_name: The collection it points to

        Raises:
            DaemonConnectionError: If daemon is not connected or rename fails
        """
        self._ensure_connected()

        if not self.collection_service:
            raise DaemonConnectionError(
                "CollectionService not available. Daemon may not support new protocol."
            )

        request = RenameAliasRequest(
            old_alias_name=old_alias_name,
            new_alias_name=new_alias_name,
            collection_name=collection_name,
        )

        try:
            await self.collection_service.RenameCollectionAlias(request)
            logger.info(
                "Collection alias renamed successfully",
                old_alias=old_alias_name,
                new_alias=new_alias_name,
                collection_name=collection_name,
            )
        except grpc.RpcError as e:
            error_msg = f"Failed to rename collection alias: {e.code()}: {e.details()}"
            logger.error(error_msg)
            raise DaemonConnectionError(error_msg) from e

    # =========================================================================
    # Legacy IngestService methods (maintained for backward compatibility)
    # =========================================================================

    # Document processing operations

    async def process_document(
        self,
        file_path: str,
        collection: str,
        metadata: Optional[Dict[str, str]] = None,
        document_id: Optional[str] = None,
        chunk_text: bool = True,
    ) -> ProcessDocumentResponse:
        """Process a single document."""
        self._ensure_connected()

        # Apply LLM access control validation for collection writes
        try:
            validate_llm_collection_access('write', collection, self.config)
        except LLMAccessControlError as e:
            logger.warning("LLM access control blocked collection write: %s", str(e))
            raise DaemonConnectionError(f"Collection write blocked: {str(e)}") from e

        request = ProcessDocumentRequest(
            file_path=file_path,
            collection=collection,
            metadata=metadata or {},
            document_id=document_id,
            chunk_text=chunk_text,
        )

        return await self.stub.ProcessDocument(request)

    async def process_folder(
        self,
        folder_path: str,
        collection: str,
        include_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        max_depth: int = 5,
        dry_run: bool = False,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Iterator[ProcessFolderProgress]:
        """Process all documents in a folder."""
        self._ensure_connected()

        # Apply LLM access control validation for collection writes (unless dry run)
        if not dry_run:
            try:
                validate_llm_collection_access('write', collection, self.config)
            except LLMAccessControlError as e:
                logger.warning("LLM access control blocked collection write: %s", str(e))
                raise DaemonConnectionError(f"Collection write blocked: {str(e)}") from e

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

        async for progress in self.stub.ProcessFolder(request):
            yield progress

    # File watching operations

    async def start_watching(
        self,
        path: str,
        collection: str,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        auto_ingest: bool = True,
        recursive: bool = True,
        recursive_depth: int = -1,
        debounce_seconds: int = 5,
        update_frequency_ms: int = 1000,
        watch_id: Optional[str] = None,
    ) -> Iterator[WatchingUpdate]:
        """Start watching a folder for changes."""
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

        async for update in self.stub.StartWatching(request):
            yield update

    async def stop_watching(self, watch_id: str) -> StopWatchingResponse:
        """Stop watching a folder."""
        self._ensure_connected()

        request = StopWatchingRequest(watch_id=watch_id)
        return await self.stub.StopWatching(request)

    async def list_watches(self, active_only: bool = False) -> ListWatchesResponse:
        """List all folder watches."""
        self._ensure_connected()

        request = ListWatchesRequest(active_only=active_only)
        return await self.stub.ListWatches(request)

    async def configure_watch(
        self,
        watch_id: str,
        status: Optional[WatchStatus] = None,
        patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None,
        auto_ingest: Optional[bool] = None,
        recursive: Optional[bool] = None,
        recursive_depth: Optional[int] = None,
        debounce_seconds: Optional[int] = None,
    ) -> ConfigureWatchResponse:
        """Configure an existing watch."""
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

        return await self.stub.ConfigureWatch(request)

    # Search operations

    async def execute_query(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        mode: SearchMode = SearchMode.SEARCH_MODE_HYBRID,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> ExecuteQueryResponse:
        """Execute a search query."""
        self._ensure_connected()

        request = ExecuteQueryRequest(
            query=query,
            collections=collections or [],
            mode=mode,
            limit=limit,
            score_threshold=score_threshold,
        )

        return await self.stub.ExecuteQuery(request)

    async def list_collections(
        self, include_stats: bool = False
    ) -> ListCollectionsResponse:
        """List all collections."""
        self._ensure_connected()

        request = ListCollectionsRequest(include_stats=include_stats)
        return await self.stub.ListCollections(request)

    async def get_collection_info(
        self, collection_name: str, include_sample_documents: bool = False
    ) -> CollectionInfo:
        """Get information about a specific collection."""
        self._ensure_connected()

        request = GetCollectionInfoRequest(
            collection_name=collection_name,
            include_sample_documents=include_sample_documents,
        )

        return await self.stub.GetCollectionInfo(request)

    async def create_collection(
        self,
        collection_name: str,
        description: str = "",
        metadata: Optional[Dict[str, str]] = None,
    ) -> CreateCollectionResponse:
        """Create a new collection (legacy method)."""
        self._ensure_connected()

        # Apply LLM access control validation for collection creation
        try:
            validate_llm_collection_access('create', collection_name, self.config)
        except LLMAccessControlError as e:
            logger.warning("LLM access control blocked collection creation: %s", str(e))
            raise DaemonConnectionError(f"Collection creation blocked: {str(e)}") from e

        request = CreateCollectionRequest(
            collection_name=collection_name,
            description=description,
            metadata=metadata or {},
        )

        return await self.stub.CreateCollection(request)

    async def delete_collection(
        self, collection_name: str, confirm: bool = False
    ) -> DeleteCollectionResponse:
        """Delete a collection (legacy method)."""
        self._ensure_connected()

        # Apply LLM access control validation for collection deletion
        try:
            validate_llm_collection_access('delete', collection_name, self.config)
        except LLMAccessControlError as e:
            logger.warning("LLM access control blocked collection deletion: %s", str(e))
            raise DaemonConnectionError(f"Collection deletion blocked: {str(e)}") from e

        request = DeleteCollectionRequest(
            collection_name=collection_name, confirm=confirm
        )

        return await self.stub.DeleteCollection(request)

    # Document management

    async def list_documents(
        self,
        collection_name: str,
        limit: int = 100,
        offset: int = 0,
        filter_pattern: str = "",
    ) -> ListDocumentsResponse:
        """List documents in a collection."""
        self._ensure_connected()

        request = ListDocumentsRequest(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            filter_pattern=filter_pattern,
        )

        return await self.stub.ListDocuments(request)

    async def get_document(
        self, document_id: str, collection_name: str, include_content: bool = False
    ) -> GetDocumentResponse:
        """Get a specific document."""
        self._ensure_connected()

        request = GetDocumentRequest(
            document_id=document_id,
            collection_name=collection_name,
            include_content=include_content,
        )

        return await self.stub.GetDocument(request)

    async def delete_document(
        self, document_id: str, collection_name: str
    ) -> DeleteDocumentResponse:
        """Delete a document."""
        self._ensure_connected()

        request = DeleteDocumentRequest(
            document_id=document_id, collection_name=collection_name
        )

        return await self.stub.DeleteDocument(request)

    # Configuration management

    async def load_configuration(
        self, config_path: Optional[str] = None
    ) -> LoadConfigurationResponse:
        """Load configuration from daemon."""
        self._ensure_connected()

        request = LoadConfigurationRequest(config_path=config_path)
        return await self.stub.LoadConfiguration(request)

    async def save_configuration(
        self, config_yaml: str, target_path: str
    ) -> SaveConfigurationResponse:
        """Save configuration to daemon."""
        self._ensure_connected()

        request = SaveConfigurationRequest(
            config_yaml=config_yaml, target_path=target_path
        )

        return await self.stub.SaveConfiguration(request)

    async def validate_configuration(
        self, config_yaml: str
    ) -> ValidateConfigurationResponse:
        """Validate configuration."""
        self._ensure_connected()

        request = ValidateConfigurationRequest(config_yaml=config_yaml)
        return await self.stub.ValidateConfiguration(request)

    # Memory operations

    async def add_memory_rule(
        self,
        category: str,
        name: str,
        rule_text: str,
        authority: Optional[str] = None,
        scope: Optional[List[str]] = None,
        source: Optional[str] = None,
    ) -> AddMemoryRuleResponse:
        """Add a memory rule."""
        self._ensure_connected()

        request = AddMemoryRuleRequest(
            category=category,
            name=name,
            rule_text=rule_text,
            authority=authority,
            scope=scope or [],
            source=source,
        )

        return await self.stub.AddMemoryRule(request)

    async def list_memory_rules(
        self,
        category: Optional[str] = None,
        authority: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> ListMemoryRulesResponse:
        """List memory rules."""
        self._ensure_connected()

        request = ListMemoryRulesRequest(
            category=category, authority=authority, limit=limit, offset=offset
        )

        return await self.stub.ListMemoryRules(request)

    async def delete_memory_rule(self, rule_id: str) -> DeleteMemoryRuleResponse:
        """Delete a memory rule."""
        self._ensure_connected()

        request = DeleteMemoryRuleRequest(rule_id=rule_id)
        return await self.stub.DeleteMemoryRule(request)

    async def search_memory_rules(
        self,
        query: str,
        category: Optional[str] = None,
        authority: Optional[str] = None,
        limit: int = 10,
    ) -> SearchMemoryRulesResponse:
        """Search memory rules."""
        self._ensure_connected()

        request = SearchMemoryRulesRequest(
            query=query, category=category, authority=authority, limit=limit
        )

        return await self.stub.SearchMemoryRules(request)

    # Status and monitoring

    async def get_stats(
        self, include_collection_stats: bool = True, include_watch_stats: bool = True
    ) -> GetStatsResponse:
        """Get daemon statistics."""
        self._ensure_connected()

        request = GetStatsRequest(
            include_collection_stats=include_collection_stats,
            include_watch_stats=include_watch_stats,
        )

        return await self.stub.GetStats(request)

    async def get_processing_status(
        self, include_history: bool = False, history_limit: int = 50
    ) -> ProcessingStatusResponse:
        """Get processing status."""
        self._ensure_connected()

        request = GetProcessingStatusRequest(
            include_history=include_history, history_limit=history_limit
        )

        return await self.stub.GetProcessingStatus(request)

    async def get_system_status(self) -> SystemStatusResponse:
        """Get system status."""
        self._ensure_connected()

        return await self.stub.GetSystemStatus(Empty())

    async def health_check(self) -> HealthResponse:
        """Perform health check."""
        self._ensure_connected()

        return await self.stub.HealthCheck(Empty())


# Global client instance for convenience
_daemon_client: Optional[DaemonClient] = None


def get_daemon_client(
    config_manager=None,
    project_path: Optional[str] = None
) -> DaemonClient:
    """Get the global daemon client instance with optional project context."""
    global _daemon_client

    if _daemon_client is None or config_manager is not None or project_path is not None:
        _daemon_client = DaemonClient(config_manager, project_path)

    return _daemon_client


async def with_daemon_client(
    operation,
    config_manager=None,
    project_path: Optional[str] = None
):
    """Execute an operation with a connected daemon client."""
    client = get_daemon_client(config_manager, project_path)
    async with client.connection():
        return await operation(client)


def create_project_client(project_path: str, config_manager=None) -> DaemonClient:
    """Create a new daemon client for a specific project path."""
    return DaemonClient(config_manager, project_path)
