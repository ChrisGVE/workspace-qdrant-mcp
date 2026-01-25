"""
gRPC-enabled workspace client that integrates with the Rust ingestion engine.

This module provides a hybrid client that can operate in two modes:
1. Direct mode: Uses the Python Qdrant client directly (existing behavior)
2. gRPC mode: Routes operations through the Rust ingestion engine

The client automatically falls back to direct mode if the gRPC server is unavailable.
"""

from pathlib import Path
from typing import Any

from loguru import logger

from ..grpc.client import AsyncIngestClient
from ..grpc.connection_manager import ConnectionConfig
from .client import QdrantWorkspaceClient
from .config import ConfigManager
from .daemon_manager import ensure_daemon_running, get_daemon_for_project

# logger imported from loguru


class GrpcWorkspaceClient:
    """
    Hybrid workspace client that can use either direct Qdrant access or gRPC routing.

    This client provides a unified interface that abstracts whether operations
    are performed directly against Qdrant or routed through the Rust engine.
    """

    def __init__(
        self,
        config: ConfigManager,
        grpc_enabled: bool = True,
        grpc_host: str = "127.0.0.1",
        grpc_port: int | None = None,
        fallback_to_direct: bool = True,
        auto_start_daemon: bool = True,
        project_name: str | None = None,
        project_path: str | None = None,
    ):
        """Initialize the gRPC-enabled workspace client.

        Args:
            config: Workspace configuration
            grpc_enabled: Whether to attempt gRPC connections
            grpc_host: gRPC server host
            grpc_port: gRPC server port (auto-assigned if None)
            fallback_to_direct: Fall back to direct mode if gRPC fails
            auto_start_daemon: Automatically start daemon if needed
            project_name: Project name for daemon management
            project_path: Project path for daemon management
        """
        self.config = config
        self.grpc_enabled = grpc_enabled
        self.fallback_to_direct = fallback_to_direct
        self.auto_start_daemon = auto_start_daemon

        # Project identification for daemon management
        self.project_name = project_name or self._detect_project_name()
        self.project_path = project_path or str(Path.cwd())

        # Initialize direct client (always available as fallback)
        # QdrantWorkspaceClient uses get_config() internally, no args needed
        self.direct_client = QdrantWorkspaceClient()

        # gRPC client and daemon info - will be initialized during startup
        self.grpc_client: AsyncIngestClient | None = None
        self.grpc_available = False
        self.grpc_host = grpc_host
        self.grpc_port = grpc_port  # Will be determined by daemon manager if None
        self.daemon_instance = None

        self._mode = "unknown"  # Will be determined during initialization

    async def initialize(self):
        """Initialize the workspace client and determine operation mode."""
        logger.info(
            "Initializing GrpcWorkspaceClient with daemon management",
            project=self.project_name,
            auto_start=self.auto_start_daemon,
        )

        # Always initialize the direct client
        await self.direct_client.initialize()
        logger.info("Direct Qdrant client initialized successfully")

        # Handle gRPC initialization with daemon management
        if self.grpc_enabled:
            success = await self._initialize_grpc_with_daemon()
            if not success and not self.fallback_to_direct:
                raise RuntimeError("gRPC mode required but daemon unavailable")

        # Set operation mode
        if self.grpc_available:
            self._mode = "grpc"
            logger.info(
                "Operating in gRPC mode with Rust daemon",
                port=self.grpc_port,
                project=self.project_name,
            )
        elif self.fallback_to_direct:
            self._mode = "direct"
            logger.info("Operating in direct mode (Qdrant only)")
        else:
            raise RuntimeError("No operation mode available")

        logger.info("GrpcWorkspaceClient initialization completed", mode=self._mode)

    async def close(self):
        """Close the client and clean up resources."""
        logger.info("Closing GrpcWorkspaceClient", project=self.project_name)

        if self.grpc_client:
            try:
                await self.grpc_client.stop()
            except Exception as e:
                logger.warning("Error stopping gRPC client", error=str(e))

        if self.direct_client:
            try:
                await self.direct_client.close()
            except Exception as e:
                logger.warning("Error closing direct client", error=str(e))

        # Note: We don't stop the daemon here as it may be shared with other clients
        # The daemon manager handles cleanup on process exit

        logger.info("GrpcWorkspaceClient closed")

    def get_operation_mode(self) -> str:
        """Get the current operation mode ('grpc', 'direct', or 'unknown')."""
        return self._mode

    def is_grpc_available(self) -> bool:
        """Check if gRPC mode is available."""
        return self.grpc_available

    async def get_status(self) -> dict[str, Any]:
        """Get comprehensive workspace status including gRPC information."""
        # Get base status from direct client
        status = await self.direct_client.get_status()

        # Add gRPC-specific information
        status.update(
            {
                "grpc_enabled": self.grpc_enabled,
                "grpc_available": self.grpc_available,
                "operation_mode": self._mode,
                "fallback_enabled": self.fallback_to_direct,
            }
        )

        # Add gRPC connection info if available
        if self.grpc_client and self.grpc_available:
            try:
                grpc_info = self.grpc_client.get_connection_info()
                status["grpc_connection"] = grpc_info
            except Exception as e:
                logger.warning("Failed to get gRPC connection info", error=str(e))
                status["grpc_connection"] = {"error": str(e)}

        return status

    async def list_collections(self) -> list[str]:
        """List available collections."""
        # This always uses direct client as it's a metadata operation
        return self.direct_client.list_collections()

    async def add_document(
        self,
        content: str,
        collection: str,
        metadata: dict[str, str] | None = None,
        document_id: str | None = None,
        chunk_text: bool = True,
    ) -> dict[str, Any]:
        """Add a document to a collection, using gRPC if available."""

        # For file-based operations, prefer gRPC if available
        if self.grpc_available and self.grpc_client:
            try:
                # Note: This assumes content represents a file path for gRPC
                # In practice, you might need to write content to a temp file
                # or modify the gRPC interface to accept content directly

                logger.debug("Using gRPC mode for document addition")

                # For now, fall back to direct mode for content-based addition
                # TODO: Extend gRPC interface to support direct content or handle temp files
                logger.debug(
                    "Falling back to direct mode for content-based document addition"
                )
                return await self.direct_client.add_document(
                    content, collection, metadata, document_id, chunk_text
                )

            except Exception as e:
                logger.warning(
                    "gRPC document addition failed, falling back", error=str(e)
                )
                if not self.fallback_to_direct:
                    raise

        # Use direct client
        return await self.direct_client.add_document(
            content, collection, metadata, document_id, chunk_text
        )

    async def process_document_file(
        self,
        file_path: str,
        collection: str,
        metadata: dict[str, str] | None = None,
        document_id: str | None = None,
        chunk_text: bool = True,
    ) -> dict[str, Any]:
        """Process a document file, preferring gRPC for file-based operations."""

        if self.grpc_available and self.grpc_client:
            try:
                logger.debug("Using gRPC mode for file processing", file_path=file_path)

                response = await self.grpc_client.process_document(
                    file_path=file_path,
                    collection=collection,
                    metadata=metadata,
                    document_id=document_id,
                    chunk_text=chunk_text,
                )

                # Convert gRPC response to expected format
                return {
                    "success": response.success,
                    "message": response.message,
                    "document_id": response.document_id,
                    "chunks_added": response.chunks_added,
                    "collection": collection,
                    "metadata": response.applied_metadata or metadata or {},
                    "processing_mode": "grpc",
                }

            except Exception as e:
                logger.warning(
                    "gRPC file processing failed, falling back",
                    file_path=file_path,
                    error=str(e),
                )
                if not self.fallback_to_direct:
                    raise

        # Fall back to direct client
        # Read file content and use direct addition
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            result = await self.direct_client.add_document(
                content, collection, metadata, document_id, chunk_text
            )
            result["processing_mode"] = "direct"
            return result

        except Exception as e:
            logger.error(
                "File processing failed in both modes",
                file_path=file_path,
                error=str(e),
            )
            raise

    async def search_workspace(
        self,
        query: str,
        collections: list[str] | None = None,
        mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> dict[str, Any]:
        """Execute a search query, using gRPC if available for better performance."""

        if self.grpc_available and self.grpc_client:
            try:
                logger.debug("Using gRPC mode for search", query=query[:50])

                response = await self.grpc_client.execute_query(
                    query=query,
                    collections=collections,
                    mode=mode,
                    limit=limit,
                    score_threshold=score_threshold,
                )

                # Convert gRPC response to expected format
                results = []
                for result in response.results:
                    results.append(
                        {
                            "id": result.id,
                            "score": result.score,
                            "payload": result.payload,
                            "collection": result.collection,
                            "search_type": result.search_type,
                        }
                    )

                return {
                    "query": response.query,
                    "mode": response.mode,
                    "collections_searched": response.collections_searched,
                    "total_results": response.total_results,
                    "results": results,
                    "processing_mode": "grpc",
                }

            except Exception as e:
                logger.warning("gRPC search failed, falling back", error=str(e))
                if not self.fallback_to_direct:
                    raise

        # Fall back to direct client search
        from ..tools.search import search_workspace

        result = await search_workspace(
            self.direct_client, query, collections, mode, limit, score_threshold
        )
        result["processing_mode"] = "direct"
        return result

    async def get_document(
        self, document_id: str, collection: str, include_vectors: bool = False
    ) -> dict[str, Any]:
        """Get a document by ID (always uses direct client for metadata operations)."""
        return await self.direct_client.get_document(
            document_id, collection, include_vectors
        )

    async def get_grpc_stats(self) -> dict[str, Any] | None:
        """Get statistics from the gRPC server if available."""
        if self.grpc_available and self.grpc_client:
            try:
                return await self.grpc_client.get_stats()
            except Exception as e:
                logger.warning("Failed to get gRPC stats", error=str(e))
                return None
        return None

    async def start_file_watching(
        self, path: str, collection: str, patterns: list[str] | None = None, **kwargs
    ):
        """Start file watching using gRPC if available."""
        if self.grpc_available and self.grpc_client:
            try:
                logger.info(
                    "Starting gRPC file watching", path=path, collection=collection
                )

                async for event in self.grpc_client.start_watching(
                    path=path, collection=collection, patterns=patterns, **kwargs
                ):
                    yield event

            except Exception as e:
                logger.error("gRPC file watching failed", path=path, error=str(e))
                if not self.fallback_to_direct:
                    raise
        else:
            logger.warning("File watching requested but gRPC not available")
            if not self.fallback_to_direct:
                raise RuntimeError("File watching requires gRPC mode")

        # Note: Direct mode file watching would require implementing
        # Python-based file watching, which is not currently available
        logger.warning("File watching fallback not implemented in direct mode")

    async def _initialize_grpc_with_daemon(self) -> bool:
        """Initialize gRPC connection, starting daemon if needed."""
        try:
            # Check if daemon is already running
            existing_daemon = await get_daemon_for_project(
                self.project_name, self.project_path
            )

            if existing_daemon and existing_daemon.status.state == "running":
                logger.info(
                    "Found existing daemon",
                    project=self.project_name,
                    port=existing_daemon.config.grpc_port,
                )
                self.daemon_instance = existing_daemon
                self.grpc_port = existing_daemon.config.grpc_port
            elif self.auto_start_daemon:
                logger.info(
                    "Starting new daemon for project", project=self.project_name
                )

                # Prepare daemon configuration overrides
                daemon_config = {}
                if self.grpc_port is not None:
                    daemon_config["grpc_port"] = self.grpc_port
                if hasattr(self.config, "qdrant_url"):
                    daemon_config["qdrant_url"] = self.config.qdrant_url

                # Start daemon
                self.daemon_instance = await ensure_daemon_running(
                    self.project_name, self.project_path, daemon_config
                )
                self.grpc_port = self.daemon_instance.config.grpc_port
                logger.info(
                    "Daemon started successfully",
                    project=self.project_name,
                    port=self.grpc_port,
                )
            else:
                logger.warning("No daemon available and auto-start disabled")
                return False

            # Now initialize gRPC client with daemon connection
            if self.daemon_instance:
                connection_config = ConnectionConfig(
                    host=self.grpc_host,
                    port=self.grpc_port,
                    connection_timeout=5.0,
                )

                self.grpc_client = AsyncIngestClient(
                    connection_config=connection_config
                )
                await self.grpc_client.start()

                # Test connection
                is_available = await self.grpc_client.test_connection()
                if is_available:
                    self.grpc_available = True
                    logger.info(
                        "gRPC connection established with daemon",
                        host=self.grpc_host,
                        port=self.grpc_port,
                    )
                    return True
                else:
                    logger.warning("gRPC health check failed despite daemon running")
                    return False

            return False

        except Exception as e:
            logger.error(
                "Failed to initialize gRPC with daemon",
                project=self.project_name,
                error=str(e),
            )
            return False

    def _detect_project_name(self) -> str:
        """Detect project name from current working directory."""
        try:
            return Path.cwd().name
        except Exception:
            return "default"

    async def ensure_daemon_available(self) -> bool:
        """Ensure daemon is available, starting if necessary."""
        if self.grpc_available and self.daemon_instance:
            # Check daemon health
            if await self.daemon_instance.health_check():
                return True
            else:
                logger.warning("Daemon health check failed", project=self.project_name)

        if self.auto_start_daemon:
            logger.info("Attempting to restart daemon", project=self.project_name)
            return await self._initialize_grpc_with_daemon()

        return False

    async def get_daemon_status(self) -> dict[str, Any] | None:
        """Get status of the associated daemon."""
        if self.daemon_instance:
            return self.daemon_instance.get_status()
        return None

    # Delegate other methods to direct client
    def get_embedding_service(self):
        """Get the embedding service (always from direct client)."""
        return self.direct_client.get_embedding_service()

    @property
    def client(self):
        """Get the underlying Qdrant client (direct mode)."""
        return self.direct_client.client
