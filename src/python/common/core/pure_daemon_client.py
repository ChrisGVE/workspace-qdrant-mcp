"""Pure daemon client for memexd communication.

This module provides a clean interface to communicate with the pure daemon
architecture of memexd, removing all hybrid mode logic and on-demand
process spawning for a cleaner, more reliable architecture.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger
from .client import QdrantWorkspaceClient
from .daemon_client import DaemonClient
from ..grpc.daemon_client import DaemonClientError as DaemonError

# logger imported from loguru


class PureDaemonWorkspaceClient:
    """Pure daemon workspace client with priority-based communication.

    This client communicates exclusively with a running memexd daemon process,
    providing priority-based task scheduling and resource management without
    any fallback to direct operations.
    """

    def __init__(
        self,
        daemon_host: str = "127.0.0.1",
        daemon_port: int = 50051,
        qdrant_url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        project_name: str = "default",
        project_path: str = "/tmp",
        timeout_seconds: float = 30.0,
    ):
        self.daemon_host = daemon_host
        self.daemon_port = daemon_port
        self.qdrant_url = qdrant_url
        self.api_key = api_key
        self.project_name = project_name
        self.project_path = project_path
        self.timeout_seconds = timeout_seconds

        # Pure daemon client for all operations
        self.daemon_client = DaemonClient(
            host=daemon_host, port=daemon_port, timeout=timeout_seconds
        )

        # Track MCP server activity for priority management
        self.mcp_active = False
        self._operation_count = 0

        logger.info(
            "Initializing pure daemon workspace client",
            daemon_address=f"{daemon_host}:{daemon_port}",
            project_name=project_name,
            project_path=project_path,
        )

    async def initialize(self) -> None:
        """Initialize the pure daemon client connection."""
        logger.debug("Initializing pure daemon client connection")

        try:
            # Test daemon connection and ensure it's running
            await self.daemon_client.connect()

            # Set MCP activity status to enable high-priority processing
            await self._set_mcp_active(True)

            # Initialize workspace collections via daemon
            await self._initialize_workspace_collections()

            logger.info("Pure daemon client initialized successfully")

        except Exception as e:
            logger.error(
                "Failed to initialize pure daemon client", error=str(e), exc_info=True
            )
            raise RuntimeError(f"Pure daemon client initialization failed: {e}") from e

    async def close(self) -> None:
        """Clean up daemon client connection."""
        logger.debug("Closing pure daemon client connection")

        try:
            # Set MCP inactive to allow low-priority tasks to resume
            await self._set_mcp_active(False)

            # Close daemon connection
            await self.daemon_client.close()

            logger.info("Pure daemon client closed successfully")

        except Exception as e:
            logger.error("Error during pure daemon client cleanup", error=str(e))

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive workspace status from daemon."""
        logger.debug("Requesting workspace status from daemon")

        try:
            # Mark this as high-priority MCP operation
            await self._mark_mcp_operation()

            # Request status from daemon
            status_data = await self.daemon_client.call_method(
                "get_workspace_status",
                {"project_name": self.project_name, "project_path": self.project_path},
            )

            # Add client-specific information
            status_data.update(
                {
                    "daemon_address": f"{self.daemon_host}:{self.daemon_port}",
                    "operation_mode": "pure_daemon",
                    "mcp_active": self.mcp_active,
                    "operation_count": self._operation_count,
                    "client_type": "PureDaemonWorkspaceClient",
                }
            )

            logger.debug("Workspace status retrieved from daemon", status=status_data)
            return status_data

        except DaemonError as e:
            logger.error("Daemon error while getting status", error=str(e))
            return {
                "connected": False,
                "error": f"Daemon communication failed: {e}",
                "operation_mode": "pure_daemon_error",
            }
        except Exception as e:
            logger.error(
                "Unexpected error while getting status", error=str(e), exc_info=True
            )
            return {
                "connected": False,
                "error": f"Unexpected error: {e}",
                "operation_mode": "pure_daemon_error",
            }

    async def list_collections(self) -> List[str]:
        """List all available collections via daemon."""
        logger.debug("Requesting collection list from daemon")

        try:
            await self._mark_mcp_operation()

            collections = await self.daemon_client.call_method(
                "list_collections", {"project_name": self.project_name}
            )

            logger.debug("Collections listed via daemon", collections=collections)
            return collections

        except DaemonError as e:
            logger.error("Daemon error while listing collections", error=str(e))
            return []
        except Exception as e:
            logger.error(
                "Unexpected error while listing collections",
                error=str(e),
                exc_info=True,
            )
            return []

    async def add_document(
        self,
        content: str,
        collection: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        chunk_text: bool = True,
    ) -> Dict[str, Any]:
        """Add document via daemon with high priority."""
        logger.debug(
            "Adding document via daemon",
            collection=collection,
            content_length=len(content),
            document_id=document_id,
            chunk_text=chunk_text,
        )

        try:
            await self._mark_mcp_operation()

            result = await self.daemon_client.call_method(
                "add_document",
                {
                    "content": content,
                    "collection": collection,
                    "metadata": metadata or {},
                    "document_id": document_id,
                    "chunk_text": chunk_text,
                    "priority": "high",
                    "source": "mcp_server",
                },
            )

            logger.info(
                "Document added via daemon",
                success=result.get("success", False),
                document_id=result.get("document_id"),
                collection=collection,
            )

            return result

        except DaemonError as e:
            logger.error("Daemon error while adding document", error=str(e))
            return {"success": False, "error": f"Daemon communication failed: {e}"}
        except Exception as e:
            logger.error(
                "Unexpected error while adding document", error=str(e), exc_info=True
            )
            return {"success": False, "error": f"Unexpected error: {e}"}

    async def get_document(
        self, document_id: str, collection: str, include_vectors: bool = False
    ) -> Dict[str, Any]:
        """Get document via daemon with high priority."""
        logger.debug(
            "Getting document via daemon",
            document_id=document_id,
            collection=collection,
            include_vectors=include_vectors,
        )

        try:
            await self._mark_mcp_operation()

            result = await self.daemon_client.call_method(
                "get_document",
                {
                    "document_id": document_id,
                    "collection": collection,
                    "include_vectors": include_vectors,
                    "priority": "high",
                    "source": "mcp_server",
                },
            )

            logger.debug("Document retrieved via daemon", document_id=document_id)
            return result

        except DaemonError as e:
            logger.error("Daemon error while getting document", error=str(e))
            return {"error": f"Daemon communication failed: {e}"}
        except Exception as e:
            logger.error(
                "Unexpected error while getting document", error=str(e), exc_info=True
            )
            return {"error": f"Unexpected error: {e}"}

    async def search(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Search via daemon with high priority."""
        logger.debug(
            "Searching via daemon",
            query_length=len(query),
            collections=collections,
            mode=mode,
            limit=limit,
            score_threshold=score_threshold,
        )

        try:
            await self._mark_mcp_operation()

            result = await self.daemon_client.call_method(
                "search",
                {
                    "query": query,
                    "collections": collections or [],
                    "mode": mode,
                    "limit": limit,
                    "score_threshold": score_threshold,
                    "priority": "high",
                    "source": "mcp_server",
                },
            )

            logger.info(
                "Search completed via daemon",
                results_count=result.get("total_results", 0),
                collections_searched=len(result.get("collections_searched", [])),
            )

            return result

        except DaemonError as e:
            logger.error("Daemon error while searching", error=str(e))
            return {
                "error": f"Daemon communication failed: {e}",
                "total_results": 0,
                "results": [],
            }
        except Exception as e:
            logger.error(
                "Unexpected error while searching", error=str(e), exc_info=True
            )
            return {
                "error": f"Unexpected error: {e}",
                "total_results": 0,
                "results": [],
            }

    async def search_by_metadata(
        self, collection: str, metadata_filter: Dict[str, Any], limit: int = 10
    ) -> Dict[str, Any]:
        """Search by metadata via daemon with high priority."""
        logger.debug(
            "Searching by metadata via daemon",
            collection=collection,
            filter_keys=list(metadata_filter.keys()),
            limit=limit,
        )

        try:
            await self._mark_mcp_operation()

            result = await self.daemon_client.call_method(
                "search_by_metadata",
                {
                    "collection": collection,
                    "metadata_filter": metadata_filter,
                    "limit": limit,
                    "priority": "high",
                    "source": "mcp_server",
                },
            )

            logger.debug("Metadata search completed via daemon", collection=collection)
            return result

        except DaemonError as e:
            logger.error("Daemon error while searching by metadata", error=str(e))
            return {"error": f"Daemon communication failed: {e}", "results": []}
        except Exception as e:
            logger.error(
                "Unexpected error while searching by metadata",
                error=str(e),
                exc_info=True,
            )
            return {"error": f"Unexpected error: {e}", "results": []}

    async def get_resource_stats(self) -> Dict[str, Any]:
        """Get resource management statistics from daemon."""
        logger.debug("Requesting resource statistics from daemon")

        try:
            stats = await self.daemon_client.call_method(
                "get_resource_stats", {"include_queues": True}
            )

            logger.debug("Resource statistics retrieved", stats=stats)
            return stats

        except DaemonError as e:
            logger.error("Daemon error while getting resource stats", error=str(e))
            return {"error": f"Daemon communication failed: {e}"}
        except Exception as e:
            logger.error(
                "Unexpected error while getting resource stats",
                error=str(e),
                exc_info=True,
            )
            return {"error": f"Unexpected error: {e}"}

    async def _initialize_workspace_collections(self) -> None:
        """Initialize workspace collections via daemon."""
        logger.debug("Initializing workspace collections via daemon")

        try:
            result = await self.daemon_client.call_method(
                "initialize_workspace",
                {
                    "project_name": self.project_name,
                    "project_path": self.project_path,
                    "qdrant_url": self.qdrant_url,
                    "api_key": self.api_key,
                },
            )

            if result.get("success", False):
                logger.info("Workspace collections initialized via daemon")
            else:
                logger.warning(
                    "Workspace initialization via daemon failed",
                    error=result.get("error"),
                )

        except Exception as e:
            logger.error(
                "Failed to initialize workspace collections via daemon", error=str(e)
            )
            raise

    async def _mark_mcp_operation(self) -> None:
        """Mark the start of an MCP operation for priority tracking."""
        self._operation_count += 1

        # Set MCP active status every 10 operations to keep daemon informed
        if self._operation_count % 10 == 0:
            await self._set_mcp_active(True)

    async def _set_mcp_active(self, active: bool) -> None:
        """Set MCP activity status in daemon for priority management."""
        if self.mcp_active != active:
            self.mcp_active = active

            try:
                await self.daemon_client.call_method(
                    "set_mcp_active",
                    {
                        "active": active,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

                logger.debug("MCP activity status updated in daemon", active=active)

            except Exception as e:
                logger.warning("Failed to update MCP activity status", error=str(e))

    def get_operation_mode(self) -> str:
        """Get the current operation mode."""
        return "pure_daemon"

    def is_daemon_mode(self) -> bool:
        """Check if client is in daemon mode."""
        return True

    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        return {
            "mode": "pure_daemon",
            "daemon_host": self.daemon_host,
            "daemon_port": self.daemon_port,
            "mcp_active": self.mcp_active,
            "operation_count": self._operation_count,
            "project_name": self.project_name,
        }
