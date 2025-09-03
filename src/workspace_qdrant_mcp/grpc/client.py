"""
Async gRPC client for communicating with the Rust ingestion engine.

This module provides a high-level async Python client that wraps the 
generated gRPC stubs with connection management, error handling, and
integration with the MCP server's async architecture.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from contextlib import asynccontextmanager

import grpc
from google.protobuf.empty_pb2 import Empty

from .connection_manager import GrpcConnectionManager, ConnectionConfig
from .types import (
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ExecuteQueryRequest,
    ExecuteQueryResponse,
    HealthCheckRequest,
    HealthCheckResponse,
)
from .ingestion_pb2_grpc import IngestServiceStub
from .ingestion_pb2 import (
    StartWatchingRequest,
    StopWatchingRequest,
    GetStatsRequest,
)

logger = logging.getLogger(__name__)


class AsyncIngestClient:
    """
    Async gRPC client for the Rust ingestion engine.
    
    Provides high-level Python async methods for document processing,
    search queries, file watching, and health monitoring.
    """
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50051,
        connection_config: Optional[ConnectionConfig] = None
    ):
        """Initialize the async gRPC client.
        
        Args:
            host: gRPC server host address
            port: gRPC server port  
            connection_config: Optional connection configuration
        """
        if connection_config:
            self.connection_config = connection_config
        else:
            self.connection_config = ConnectionConfig(host=host, port=port)
        
        self.connection_manager = GrpcConnectionManager(self.connection_config)
        self._started = False
        
        logger.info("AsyncIngestClient initialized",
                   host=host, port=port)
    
    async def start(self):
        """Start the client and connection management."""
        if not self._started:
            await self.connection_manager.start()
            self._started = True
            logger.info("AsyncIngestClient started")
    
    async def stop(self):
        """Stop the client and clean up resources."""
        if self._started:
            await self.connection_manager.stop()
            self._started = False
            logger.info("AsyncIngestClient stopped")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def process_document(
        self,
        file_path: str,
        collection: str,
        metadata: Optional[Dict[str, str]] = None,
        document_id: Optional[str] = None,
        chunk_text: bool = True,
        timeout: float = 30.0
    ) -> ProcessDocumentResponse:
        """
        Process a document for ingestion into Qdrant.
        
        Args:
            file_path: Path to the document file
            collection: Target collection name
            metadata: Optional metadata dictionary
            document_id: Optional custom document ID
            chunk_text: Whether to chunk large documents
            timeout: Request timeout in seconds
            
        Returns:
            ProcessDocumentResponse with processing results
            
        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()
        
        request = ProcessDocumentRequest(
            file_path=file_path,
            collection=collection,
            metadata=metadata,
            document_id=document_id,
            chunk_text=chunk_text
        )
        
        async def _process_doc(stub: IngestServiceStub):
            pb_request = request.to_pb()
            pb_response = await asyncio.wait_for(
                stub.ProcessDocument(pb_request),
                timeout=timeout
            )
            return ProcessDocumentResponse.from_pb(pb_response)
        
        return await self.connection_manager.with_retry(_process_doc)
    
    async def execute_query(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.7,
        timeout: float = 15.0
    ) -> ExecuteQueryResponse:
        """
        Execute a search query against indexed documents.
        
        Args:
            query: Search query text
            collections: Optional list of collections to search
            mode: Search mode ("hybrid", "dense", "sparse")  
            limit: Maximum number of results
            score_threshold: Minimum relevance score
            timeout: Request timeout in seconds
            
        Returns:
            ExecuteQueryResponse with search results
            
        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()
        
        request = ExecuteQueryRequest(
            query=query,
            collections=collections,
            mode=mode,
            limit=limit,
            score_threshold=score_threshold
        )
        
        async def _execute_query(stub: IngestServiceStub):
            pb_request = request.to_pb()
            pb_response = await asyncio.wait_for(
                stub.ExecuteQuery(pb_request),
                timeout=timeout
            )
            return ExecuteQueryResponse.from_pb(pb_response)
        
        return await self.connection_manager.with_retry(_execute_query)
    
    async def health_check(self, timeout: float = 5.0) -> HealthCheckResponse:
        """
        Perform a health check on the ingestion service.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            HealthCheckResponse with service health status
            
        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()
        
        async def _health_check(stub: IngestServiceStub):
            pb_response = await asyncio.wait_for(
                stub.HealthCheck(Empty()),
                timeout=timeout
            )
            return HealthCheckResponse.from_pb(pb_response)
        
        return await self.connection_manager.with_retry(_health_check)
    
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
        watch_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Start watching a directory for file changes.
        
        This returns an async iterator that yields watch events.
        
        Args:
            path: Directory path to watch
            collection: Target collection for ingested files
            patterns: File patterns to include
            ignore_patterns: File patterns to ignore
            auto_ingest: Enable automatic ingestion
            recursive: Watch subdirectories
            recursive_depth: Maximum recursion depth (-1 for unlimited)
            debounce_seconds: Debounce delay before processing
            update_frequency_ms: File system check frequency
            watch_id: Optional custom watch identifier
            
        Yields:
            Dict containing watch event information
            
        Raises:
            grpc.RpcError: If the request fails
        """
        if not self._started:
            await self.start()
        
        request = StartWatchingRequest()
        request.path = path
        request.collection = collection
        request.auto_ingest = auto_ingest
        request.recursive = recursive
        request.recursive_depth = recursive_depth
        request.debounce_seconds = debounce_seconds
        request.update_frequency_ms = update_frequency_ms
        
        if patterns:
            request.patterns.extend(patterns)
        if ignore_patterns:
            request.ignore_patterns.extend(ignore_patterns)
        if watch_id:
            request.watch_id = watch_id
        
        async with self.connection_manager.get_stub() as stub:
            async for update in stub.StartWatching(request):
                # Convert protobuf response to dict
                event_data = {
                    "watch_id": update.watch_id,
                    "event_type": update.event_type,
                    "file_path": update.file_path,
                    "timestamp": update.timestamp.ToDatetime(),
                    "status": update.status,
                }
                
                if update.HasField('error_message'):
                    event_data["error_message"] = update.error_message
                
                yield event_data
    
    async def stop_watching(
        self,
        watch_id: str,
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Stop watching a specific watch configuration.
        
        Args:
            watch_id: Watch identifier to stop
            timeout: Request timeout in seconds
            
        Returns:
            Dict with stop operation results
            
        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()
        
        async def _stop_watching(stub: IngestServiceStub):
            request = StopWatchingRequest()
            request.watch_id = watch_id
            
            response = await asyncio.wait_for(
                stub.StopWatching(request),
                timeout=timeout
            )
            
            return {
                "success": response.success,
                "message": response.message
            }
        
        return await self.connection_manager.with_retry(_stop_watching)
    
    async def get_stats(
        self,
        include_collection_stats: bool = True,
        include_watch_stats: bool = True,
        timeout: float = 10.0
    ) -> Dict[str, Any]:
        """
        Get statistics and health information about the ingestion engine.
        
        Args:
            include_collection_stats: Include collection statistics
            include_watch_stats: Include watch statistics  
            timeout: Request timeout in seconds
            
        Returns:
            Dict with engine statistics
            
        Raises:
            grpc.RpcError: If the request fails
            asyncio.TimeoutError: If request times out
        """
        if not self._started:
            await self.start()
        
        async def _get_stats(stub: IngestServiceStub):
            request = GetStatsRequest()
            request.include_collection_stats = include_collection_stats
            request.include_watch_stats = include_watch_stats
            
            response = await asyncio.wait_for(
                stub.GetStats(request),
                timeout=timeout
            )
            
            # Convert protobuf response to dict
            stats = {
                "engine_stats": {
                    "started_at": response.engine_stats.started_at.ToDatetime(),
                    "uptime_seconds": response.engine_stats.uptime.total_seconds(),
                    "total_documents_processed": response.engine_stats.total_documents_processed,
                    "total_documents_indexed": response.engine_stats.total_documents_indexed,
                    "active_watches": response.engine_stats.active_watches,
                    "version": response.engine_stats.version,
                }
            }
            
            if include_collection_stats:
                stats["collection_stats"] = []
                for col_stat in response.collection_stats:
                    stats["collection_stats"].append({
                        "name": col_stat.name,
                        "document_count": col_stat.document_count,
                        "total_size_bytes": col_stat.total_size_bytes,
                        "last_updated": col_stat.last_updated.ToDatetime(),
                    })
            
            if include_watch_stats:
                stats["watch_stats"] = []
                for watch_stat in response.watch_stats:
                    stats["watch_stats"].append({
                        "watch_id": watch_stat.watch_id,
                        "path": watch_stat.path,
                        "collection": watch_stat.collection,
                        "status": watch_stat.status,
                        "files_processed": watch_stat.files_processed,
                        "files_failed": watch_stat.files_failed,
                        "created_at": watch_stat.created_at.ToDatetime(),
                        "last_activity": watch_stat.last_activity.ToDatetime(),
                    })
            
            return stats
        
        return await self.connection_manager.with_retry(_get_stats)
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about the current gRPC connection."""
        return self.connection_manager.get_connection_info()
    
    async def test_connection(self) -> bool:
        """
        Test if we can connect to the gRPC server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            await self.health_check(timeout=5.0)
            return True
        except Exception as e:
            logger.warning("Connection test failed", error=str(e))
            return False