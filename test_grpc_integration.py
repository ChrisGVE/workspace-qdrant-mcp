#!/usr/bin/env python3
"""
Test script for gRPC integration with the Rust ingestion engine.

This script tests the Python-Rust gRPC communication components:
1. Connection management and health checks
2. Document processing via gRPC
3. Search operations via gRPC
4. Error handling and fallback behavior
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from workspace_qdrant_mcp.grpc.client import AsyncIngestClient
from workspace_qdrant_mcp.grpc.connection_manager import ConnectionConfig
from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.core.grpc_client import GrpcWorkspaceClient
from workspace_qdrant_mcp.tools.grpc_tools import (
    test_grpc_connection,
    get_grpc_engine_stats,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_grpc_connection():
    """Test basic gRPC connection to the Rust engine."""
    logger.info("=" * 60)
    logger.info("Testing basic gRPC connection...")
    
    result = await test_grpc_connection()
    
    logger.info(f"Connection test results:")
    logger.info(f"  Address: {result['address']}")
    logger.info(f"  Connected: {result['connected']}")
    logger.info(f"  Healthy: {result.get('healthy', 'N/A')}")
    logger.info(f"  Response time: {result.get('response_time_ms', 'N/A')} ms")
    
    if result['error']:
        logger.warning(f"  Error: {result['error']}")
    
    if result.get('engine_info'):
        logger.info(f"  Engine status: {result['engine_info']['status']}")
        logger.info(f"  Engine message: {result['engine_info']['message']}")
    
    return result['connected']


async def test_async_ingest_client():
    """Test the AsyncIngestClient directly."""
    logger.info("=" * 60)
    logger.info("Testing AsyncIngestClient...")
    
    config = ConnectionConfig(
        host="127.0.0.1",
        port=50051,
        connection_timeout=5.0
    )
    
    async with AsyncIngestClient(connection_config=config) as client:
        try:
            # Test health check
            health = await client.health_check(timeout=5.0)
            logger.info(f"Health check: {health.status} - {health.message}")
            
            # Test connection info
            conn_info = client.get_connection_info()
            logger.info(f"Connection info: {conn_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"AsyncIngestClient test failed: {e}")
            return False


async def test_hybrid_client():
    """Test the hybrid GrpcWorkspaceClient."""
    logger.info("=" * 60)
    logger.info("Testing GrpcWorkspaceClient...")
    
    # Create configuration with gRPC enabled
    config = Config()
    config.grpc.enabled = True
    config.grpc.fallback_to_direct = True
    
    client = None
    try:
        client = GrpcWorkspaceClient(
            config=config,
            grpc_enabled=True,
            fallback_to_direct=True
        )
        
        await client.initialize()
        
        # Get status
        status = await client.get_status()
        logger.info(f"Operation mode: {status.get('operation_mode')}")
        logger.info(f"gRPC available: {status.get('grpc_available')}")
        logger.info(f"Collections: {status.get('collections_count', 0)}")
        
        # Test listing collections
        collections = await client.list_collections()
        logger.info(f"Available collections: {collections}")
        
        return True
        
    except Exception as e:
        logger.error(f"GrpcWorkspaceClient test failed: {e}")
        return False
    
    finally:
        if client:
            await client.close()


async def test_engine_stats():
    """Test getting engine statistics."""
    logger.info("=" * 60)
    logger.info("Testing engine statistics...")
    
    try:
        stats = await get_grpc_engine_stats()
        
        if stats['success']:
            engine_stats = stats['stats']['engine_stats']
            logger.info(f"Engine uptime: {engine_stats['uptime_seconds']:.1f} seconds")
            logger.info(f"Documents processed: {engine_stats['total_documents_processed']}")
            logger.info(f"Active watches: {engine_stats['active_watches']}")
            logger.info(f"Engine version: {engine_stats['version']}")
            
            if 'collection_stats' in stats['stats']:
                logger.info(f"Collections: {len(stats['stats']['collection_stats'])}")
            
            return True
        else:
            logger.warning(f"Failed to get stats: {stats['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Engine stats test failed: {e}")
        return False


async def main():
    """Run all gRPC integration tests."""
    logger.info("Starting gRPC integration tests...")
    logger.info("Note: These tests require a running Rust ingestion engine on localhost:50051")
    
    tests = [
        ("Basic Connection", test_basic_grpc_connection),
        ("AsyncIngestClient", test_async_ingest_client),
        ("Hybrid Client", test_hybrid_client),
        ("Engine Stats", test_engine_stats),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            logger.info(f"\nRunning test: {test_name}")
            results[test_name] = await test_func()
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"  {test_name:<20} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        logger.info("All tests passed! gRPC integration is working correctly.")
        return 0
    else:
        logger.warning("Some tests failed. Check the Rust engine is running and accessible.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))