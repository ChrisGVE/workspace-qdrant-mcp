"""
Unified gRPC client for workspace-qdrant-mcp daemon communication.

This module provides a single interface for all components (CLI, MCP server, web UI)
to communicate with the daemon, eliminating direct Qdrant client usage and code duplication.
Includes automatic service discovery for multi-instance daemon support.
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
from loguru import logger
from .yaml_config import WorkspaceConfig, load_config
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
    """

    def __init__(self, config: Optional[WorkspaceConfig] = None, project_path: Optional[str] = None):
        """Initialize daemon client with configuration and optional project context."""
        self.config = config or load_config()
        self.project_path = project_path or os.getcwd()
        self.channel: Optional[grpc.aio.Channel] = None
        self.stub: Optional[IngestServiceStub] = None
        self._connected = False
        self._discovered_endpoint: Optional[ServiceEndpoint] = None

    async def connect(self) -> None:
        """Establish connection to daemon using service discovery."""
        if self._connected:
            return

        # First, attempt service discovery for project-specific daemon
        grpc_config = self.config.daemon.grpc
        preferred_endpoint = (grpc_config.host, grpc_config.port)
        
        logger.info("Attempting service discovery for daemon connection", 
                   project_path=self.project_path)
        
        # Try to discover daemon endpoint for this project
        discovered_endpoint = await discover_daemon_endpoint(
            self.project_path, 
            preferred_endpoint
        )
        
        if discovered_endpoint:
            address = discovered_endpoint.address
            self._discovered_endpoint = discovered_endpoint
            logger.info("Using discovered daemon endpoint", 
                       address=address, 
                       project_id=discovered_endpoint.project_id)
        else:
            # Fallback to configured endpoint
            address = f"{grpc_config.host}:{grpc_config.port}"
            logger.warning("Service discovery failed, using configured endpoint", 
                          address=address)

        try:
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
                    grpc_config.max_message_size_mb * 1024 * 1024,
                ),
                (
                    "grpc.max_send_message_length",
                    grpc_config.max_message_size_mb * 1024 * 1024,
                ),
            ]

            self.channel = grpc.aio.insecure_channel(address, options=options)
            self.stub = IngestServiceStub(self.channel)

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
            raise DaemonConnectionError(f"Failed to connect to daemon: {e}")

    async def disconnect(self) -> None:
        """Close connection to daemon."""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
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
            grpc_config = self.config.daemon.grpc
            info.update({
                "endpoint": f"{grpc_config.host}:{grpc_config.port}",
                "discovery_strategy": "configuration"
            })
        
        return info

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
        """Create a new collection."""
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
        """Delete a collection."""
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
    config: Optional[WorkspaceConfig] = None, 
    project_path: Optional[str] = None
) -> DaemonClient:
    """Get the global daemon client instance with optional project context."""
    global _daemon_client

    if _daemon_client is None or config is not None or project_path is not None:
        _daemon_client = DaemonClient(config, project_path)

    return _daemon_client


async def with_daemon_client(
    operation, 
    config: Optional[WorkspaceConfig] = None,
    project_path: Optional[str] = None
):
    """Execute an operation with a connected daemon client."""
    client = get_daemon_client(config, project_path)
    async with client.connection():
        return await operation(client)


def create_project_client(project_path: str, config: Optional[WorkspaceConfig] = None) -> DaemonClient:
    """Create a new daemon client for a specific project path."""
    return DaemonClient(config, project_path)
