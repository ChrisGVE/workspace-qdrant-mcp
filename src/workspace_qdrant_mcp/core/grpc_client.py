"""
gRPC-enabled workspace client that integrates with the Rust ingestion engine.

This module provides a hybrid client that can operate in two modes:
1. Direct mode: Uses the Python Qdrant client directly (existing behavior)
2. gRPC mode: Routes operations through the Rust ingestion engine

The client automatically falls back to direct mode if the gRPC server is unavailable.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..grpc.client import AsyncIngestClient
from ..grpc.connection_manager import ConnectionConfig
from .client import QdrantWorkspaceClient
from .config import Config

logger = logging.getLogger(__name__)


class GrpcWorkspaceClient:
    """
    Hybrid workspace client that can use either direct Qdrant access or gRPC routing.
    
    This client provides a unified interface that abstracts whether operations
    are performed directly against Qdrant or routed through the Rust engine.
    """
    
    def __init__(
        self,
        config: Config,
        grpc_enabled: bool = True,
        grpc_host: str = "127.0.0.1",
        grpc_port: int = 50051,
        fallback_to_direct: bool = True
    ):
        """Initialize the gRPC-enabled workspace client.
        
        Args:
            config: Workspace configuration
            grpc_enabled: Whether to attempt gRPC connections
            grpc_host: gRPC server host
            grpc_port: gRPC server port
            fallback_to_direct: Fall back to direct mode if gRPC fails
        """
        self.config = config
        self.grpc_enabled = grpc_enabled
        self.fallback_to_direct = fallback_to_direct
        
        # Initialize direct client (always available as fallback)
        self.direct_client = QdrantWorkspaceClient(config)
        
        # Initialize gRPC client if enabled
        self.grpc_client: Optional[AsyncIngestClient] = None
        self.grpc_available = False
        
        if grpc_enabled:
            try:
                connection_config = ConnectionConfig(
                    host=grpc_host,
                    port=grpc_port,
                    connection_timeout=5.0,  # Quick timeout for availability check
                )
                self.grpc_client = AsyncIngestClient(
                    connection_config=connection_config
                )
                logger.info("gRPC client initialized", host=grpc_host, port=grpc_port)
            except Exception as e:
                logger.warning("Failed to initialize gRPC client", error=str(e))
                if not fallback_to_direct:
                    raise
        
        self._mode = "unknown"  # Will be determined during initialization
    
    async def initialize(self):
        """Initialize the workspace client and determine operation mode."""
        logger.info("Initializing GrpcWorkspaceClient")
        
        # Always initialize the direct client
        await self.direct_client.initialize()
        logger.info("Direct Qdrant client initialized successfully")
        
        # Test gRPC availability if enabled
        if self.grpc_enabled and self.grpc_client:
            try:
                await self.grpc_client.start()
                
                # Test connection
                is_available = await self.grpc_client.test_connection()
                if is_available:
                    self.grpc_available = True
                    self._mode = "grpc"
                    logger.info("gRPC mode enabled - Rust engine available")
                else:
                    raise ConnectionError("gRPC health check failed")
                    
            except Exception as e:
                logger.warning("gRPC server not available", error=str(e))
                self.grpc_available = False
                
                if not self.fallback_to_direct:
                    raise RuntimeError(f"gRPC mode required but server unavailable: {e}")
        
        # Set operation mode
        if self.grpc_available:
            self._mode = "grpc"
        elif self.fallback_to_direct:
            self._mode = "direct"
            logger.info("Operating in direct mode (Qdrant only)")
        else:
            raise RuntimeError("No operation mode available")
        
        logger.info("GrpcWorkspaceClient initialization completed", mode=self._mode)
    
    async def close(self):
        """Close the client and clean up resources."""
        logger.info("Closing GrpcWorkspaceClient")
        
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
        
        logger.info("GrpcWorkspaceClient closed")
    
    def get_operation_mode(self) -> str:
        """Get the current operation mode ('grpc', 'direct', or 'unknown')."""
        return self._mode
    
    def is_grpc_available(self) -> bool:
        """Check if gRPC mode is available."""
        return self.grpc_available
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive workspace status including gRPC information."""
        # Get base status from direct client
        status = await self.direct_client.get_status()
        
        # Add gRPC-specific information
        status.update({
            "grpc_enabled": self.grpc_enabled,
            "grpc_available": self.grpc_available,
            "operation_mode": self._mode,
            "fallback_enabled": self.fallback_to_direct,
        })
        
        # Add gRPC connection info if available
        if self.grpc_client and self.grpc_available:
            try:
                grpc_info = self.grpc_client.get_connection_info()
                status["grpc_connection"] = grpc_info
            except Exception as e:
                logger.warning("Failed to get gRPC connection info", error=str(e))
                status["grpc_connection"] = {"error": str(e)}
        
        return status
    
    async def list_collections(self) -> List[str]:
        """List available collections."""
        # This always uses direct client as it's a metadata operation
        return await self.direct_client.list_collections()
    
    async def add_document(
        self,
        content: str,
        collection: str,
        metadata: Optional[Dict[str, str]] = None,
        document_id: Optional[str] = None,
        chunk_text: bool = True
    ) -> Dict[str, Any]:
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
                logger.debug("Falling back to direct mode for content-based document addition")
                return await self.direct_client.add_document(
                    content, collection, metadata, document_id, chunk_text
                )
                
            except Exception as e:
                logger.warning("gRPC document addition failed, falling back", error=str(e))
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
        metadata: Optional[Dict[str, str]] = None,
        document_id: Optional[str] = None,
        chunk_text: bool = True
    ) -> Dict[str, Any]:
        """Process a document file, preferring gRPC for file-based operations."""
        
        if self.grpc_available and self.grpc_client:
            try:
                logger.debug("Using gRPC mode for file processing", file_path=file_path)
                
                response = await self.grpc_client.process_document(
                    file_path=file_path,
                    collection=collection,
                    metadata=metadata,
                    document_id=document_id,
                    chunk_text=chunk_text
                )
                
                # Convert gRPC response to expected format
                return {
                    "success": response.success,
                    "message": response.message,
                    "document_id": response.document_id,
                    "chunks_added": response.chunks_added,
                    "collection": collection,
                    "metadata": response.applied_metadata or metadata or {},
                    "processing_mode": "grpc"
                }
                
            except Exception as e:
                logger.warning("gRPC file processing failed, falling back", 
                              file_path=file_path, error=str(e))
                if not self.fallback_to_direct:
                    raise
        
        # Fall back to direct client
        # Read file content and use direct addition
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = await self.direct_client.add_document(
                content, collection, metadata, document_id, chunk_text
            )
            result["processing_mode"] = "direct"
            return result
            
        except Exception as e:
            logger.error("File processing failed in both modes", file_path=file_path, error=str(e))
            raise
    
    async def search_workspace(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Execute a search query, using gRPC if available for better performance."""
        
        if self.grpc_available and self.grpc_client:
            try:
                logger.debug("Using gRPC mode for search", query=query[:50])
                
                response = await self.grpc_client.execute_query(
                    query=query,
                    collections=collections,
                    mode=mode,
                    limit=limit,
                    score_threshold=score_threshold
                )
                
                # Convert gRPC response to expected format
                results = []
                for result in response.results:
                    results.append({
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload,
                        "collection": result.collection,
                        "search_type": result.search_type
                    })
                
                return {
                    "query": response.query,
                    "mode": response.mode,
                    "collections_searched": response.collections_searched,
                    "total_results": response.total_results,
                    "results": results,
                    "processing_mode": "grpc"
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
        self,
        document_id: str,
        collection: str,
        include_vectors: bool = False
    ) -> Dict[str, Any]:
        """Get a document by ID (always uses direct client for metadata operations)."""
        return await self.direct_client.get_document(document_id, collection, include_vectors)
    
    async def get_grpc_stats(self) -> Optional[Dict[str, Any]]:
        """Get statistics from the gRPC server if available."""
        if self.grpc_available and self.grpc_client:
            try:
                return await self.grpc_client.get_stats()
            except Exception as e:
                logger.warning("Failed to get gRPC stats", error=str(e))
                return None
        return None
    
    async def start_file_watching(
        self,
        path: str,
        collection: str,
        patterns: Optional[List[str]] = None,
        **kwargs
    ):
        """Start file watching using gRPC if available."""
        if self.grpc_available and self.grpc_client:
            try:
                logger.info("Starting gRPC file watching", path=path, collection=collection)
                
                async for event in self.grpc_client.start_watching(
                    path=path,
                    collection=collection,
                    patterns=patterns,
                    **kwargs
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
    
    # Delegate other methods to direct client
    def get_embedding_service(self):
        """Get the embedding service (always from direct client)."""
        return self.direct_client.get_embedding_service()
    
    @property
    def client(self):
        """Get the underlying Qdrant client (direct mode)."""
        return self.direct_client.client