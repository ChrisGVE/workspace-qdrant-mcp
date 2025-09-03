"""
gRPC-specific tools for testing and monitoring Rust engine integration.

This module provides MCP tools for testing gRPC connectivity, getting
engine statistics, and managing the hybrid client mode.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from ..grpc.client import AsyncIngestClient
from ..grpc.connection_manager import ConnectionConfig
from ..core.grpc_client import GrpcWorkspaceClient

logger = logging.getLogger(__name__)


async def test_grpc_connection(
    host: str = "127.0.0.1",
    port: int = 50051,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Test gRPC connection to the Rust ingestion engine.
    
    Args:
        host: gRPC server host
        port: gRPC server port  
        timeout: Connection timeout in seconds
        
    Returns:
        Dict with connection test results
    """
    result = {
        "host": host,
        "port": port,
        "address": f"{host}:{port}",
        "connected": False,
        "healthy": False,
        "error": None,
        "response_time_ms": None,
        "engine_info": None
    }
    
    client = None
    try:
        # Create client with test configuration
        config = ConnectionConfig(
            host=host,
            port=port,
            connection_timeout=timeout
        )
        client = AsyncIngestClient(connection_config=config)
        
        # Measure connection time
        import time
        start_time = time.time()
        
        # Test connection
        await client.start()
        connected = await client.test_connection()
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        
        result.update({
            "connected": connected,
            "response_time_ms": round(response_time_ms, 2)
        })
        
        if connected:
            try:
                # Get health check
                health_response = await client.health_check(timeout=5.0)
                result["healthy"] = health_response.status == "healthy"
                result["engine_info"] = {
                    "status": health_response.status,
                    "message": health_response.message,
                    "services": health_response.services
                }
                
                # Get basic stats if available
                stats = await client.get_stats(timeout=5.0)
                if stats:
                    result["engine_stats"] = {
                        "uptime_seconds": stats["engine_stats"]["uptime_seconds"],
                        "total_documents_processed": stats["engine_stats"]["total_documents_processed"],
                        "active_watches": stats["engine_stats"]["active_watches"],
                        "version": stats["engine_stats"]["version"]
                    }
                
            except Exception as health_error:
                logger.warning("Health check failed", error=str(health_error))
                result["health_error"] = str(health_error)
        
    except Exception as e:
        logger.warning("gRPC connection test failed", 
                      host=host, port=port, error=str(e))
        result["error"] = str(e)
        
    finally:
        if client:
            try:
                await client.stop()
            except Exception as cleanup_error:
                logger.warning("Error during client cleanup", error=str(cleanup_error))
    
    return result


async def get_grpc_engine_stats(
    host: str = "127.0.0.1",
    port: int = 50051,
    include_collections: bool = True,
    include_watches: bool = True,
    timeout: float = 15.0
) -> Dict[str, Any]:
    """
    Get comprehensive statistics from the Rust ingestion engine.
    
    Args:
        host: gRPC server host
        port: gRPC server port
        include_collections: Include collection statistics
        include_watches: Include file watch statistics  
        timeout: Request timeout in seconds
        
    Returns:
        Dict with engine statistics or error information
    """
    result = {
        "success": False,
        "error": None,
        "stats": None
    }
    
    client = None
    try:
        # Create and start client
        config = ConnectionConfig(host=host, port=port)
        client = AsyncIngestClient(connection_config=config)
        await client.start()
        
        # Get comprehensive stats
        stats = await client.get_stats(
            include_collection_stats=include_collections,
            include_watch_stats=include_watches,
            timeout=timeout
        )
        
        result.update({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        logger.error("Failed to get gRPC engine stats", error=str(e))
        result["error"] = str(e)
        
    finally:
        if client:
            try:
                await client.stop()
            except Exception as cleanup_error:
                logger.warning("Error during stats client cleanup", error=str(cleanup_error))
    
    return result


async def process_document_via_grpc(
    file_path: str,
    collection: str,
    host: str = "127.0.0.1",
    port: int = 50051,
    metadata: Optional[Dict[str, str]] = None,
    document_id: Optional[str] = None,
    chunk_text: bool = True,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """
    Process a document directly via gRPC (bypassing hybrid client).
    
    This is useful for testing gRPC functionality or when you specifically
    want to use the Rust engine for document processing.
    
    Args:
        file_path: Path to document file
        collection: Target collection name
        host: gRPC server host
        port: gRPC server port
        metadata: Optional document metadata
        document_id: Optional custom document ID
        chunk_text: Whether to chunk large documents
        timeout: Processing timeout in seconds
        
    Returns:
        Dict with processing results
    """
    result = {
        "success": False,
        "error": None,
        "processing_mode": "grpc_direct",
        "file_path": file_path,
        "collection": collection
    }
    
    client = None
    try:
        # Create and start client
        config = ConnectionConfig(host=host, port=port)
        client = AsyncIngestClient(connection_config=config)
        await client.start()
        
        # Process document
        response = await client.process_document(
            file_path=file_path,
            collection=collection,
            metadata=metadata,
            document_id=document_id,
            chunk_text=chunk_text,
            timeout=timeout
        )
        
        result.update({
            "success": response.success,
            "message": response.message,
            "document_id": response.document_id,
            "chunks_added": response.chunks_added,
            "applied_metadata": response.applied_metadata
        })
        
        if not response.success:
            result["error"] = response.message
        
    except Exception as e:
        logger.error("gRPC document processing failed", 
                    file_path=file_path, error=str(e))
        result["error"] = str(e)
        
    finally:
        if client:
            try:
                await client.stop()
            except Exception as cleanup_error:
                logger.warning("Error during processing client cleanup", error=str(cleanup_error))
    
    return result


async def search_via_grpc(
    query: str,
    collections: Optional[list] = None,
    host: str = "127.0.0.1", 
    port: int = 50051,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.7,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """
    Execute search directly via gRPC (bypassing hybrid client).
    
    Args:
        query: Search query text
        collections: Optional list of collections to search
        host: gRPC server host
        port: gRPC server port
        mode: Search mode ("hybrid", "dense", "sparse")
        limit: Maximum results to return
        score_threshold: Minimum relevance score
        timeout: Search timeout in seconds
        
    Returns:
        Dict with search results
    """
    result = {
        "success": False,
        "error": None,
        "processing_mode": "grpc_direct",
        "query": query
    }
    
    client = None
    try:
        # Create and start client  
        config = ConnectionConfig(host=host, port=port)
        client = AsyncIngestClient(connection_config=config)
        await client.start()
        
        # Execute search
        response = await client.execute_query(
            query=query,
            collections=collections,
            mode=mode,
            limit=limit,
            score_threshold=score_threshold,
            timeout=timeout
        )
        
        # Convert to standard format
        results = []
        for search_result in response.results:
            results.append({
                "id": search_result.id,
                "score": search_result.score,
                "payload": search_result.payload,
                "collection": search_result.collection,
                "search_type": search_result.search_type
            })
        
        result.update({
            "success": True,
            "query": response.query,
            "mode": response.mode,
            "collections_searched": response.collections_searched,
            "total_results": response.total_results,
            "results": results
        })
        
    except Exception as e:
        logger.error("gRPC search failed", query=query, error=str(e))
        result["error"] = str(e)
        
    finally:
        if client:
            try:
                await client.stop()
            except Exception as cleanup_error:
                logger.warning("Error during search client cleanup", error=str(cleanup_error))
    
    return result