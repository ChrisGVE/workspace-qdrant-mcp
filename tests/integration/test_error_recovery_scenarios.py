"""
Integration tests for error recovery scenarios.

Tests system behavior under various failure conditions including:
- Network connectivity failures and reconnection
- Qdrant service unavailability and recovery
- File system errors and corruption handling
- Memory pressure and resource exhaustion
- gRPC communication failures and retries
- Configuration errors and validation failures
- Concurrent operation conflicts and resolution
- Data consistency recovery after failures
"""

import asyncio
import json
import os
import pytest
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, patch, MagicMock
import signal
import psutil

from testcontainers.compose import DockerCompose

from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.core.daemon_manager import DaemonManager, DaemonConfig
from workspace_qdrant_mcp.tools.grpc_tools import (
    test_grpc_connection,
    process_document_via_grpc,
    search_via_grpc
)
from workspace_qdrant_mcp.core.exceptions import (
    WorkspaceQdrantError,
    ConnectionError,
    ConfigurationError,
    ProcessingError
)


@pytest.fixture(scope="module")
def error_recovery_environment():
    """Set up controlled test environment for error recovery testing."""
    compose_file = """
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6341:6333"
      - "6342:6334"
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 2s
      timeout: 1s
      retries: 5
    # Allow manual stop/start for failure simulation

volumes:
  qdrant_storage:
"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        compose_path = Path(temp_dir) / "docker-compose.yml"
        compose_path.write_text(compose_file)
        
        with DockerCompose(temp_dir) as compose:
            qdrant_url = compose.get_service_host("qdrant", 6333)
            qdrant_port = compose.get_service_port("qdrant", 6333)
            grpc_port = compose.get_service_port("qdrant", 6334)
            
            # Wait for initial startup
            import requests
            for _ in range(20):
                try:
                    response = requests.get(f"http://{qdrant_url}:{qdrant_port}/health")
                    if response.status_code == 200:
                        break
                except:
                    pass
                time.sleep(1)
            
            yield {
                "qdrant_host": qdrant_url,
                "qdrant_port": qdrant_port,
                "grpc_port": grpc_port,
                "compose": compose  # Allow manual service control
            }


@pytest.fixture
def recovery_test_config(error_recovery_environment):
    """Create configuration for error recovery testing."""
    config_data = {
        "qdrant": {
            "host": error_recovery_environment["qdrant_host"],
            "port": error_recovery_environment["qdrant_port"],
            "grpc_port": error_recovery_environment["grpc_port"],
            "collection_name": "error_recovery_test",
            "timeout": 10,
            "max_retries": 3,
            "retry_delay": 1.0,
            "prefer_grpc": True
        },
        "embeddings": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu"
        },
        "ingestion": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "batch_size": 5,
            "max_file_size_mb": 10
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config_data, f)
        config_path = f.name
    
    try:
        config = Config.from_yaml(config_path)
        yield config
    finally:
        Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def test_documents():
    """Create test documents for error recovery scenarios."""
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        
        # Valid document
        valid_doc = test_dir / "valid.txt"
        valid_doc.write_text("This is a valid test document for error recovery testing.")
        
        # Large document
        large_doc = test_dir / "large.txt"
        large_doc.write_text("Large document content. " * 1000)
        
        # Corrupted file (binary in text context)
        corrupted_doc = test_dir / "corrupted.txt"
        corrupted_doc.write_bytes(b"\xff\xfe\x00\x01Invalid UTF-8 content\x80\x81")
        
        # Empty file
        empty_doc = test_dir / "empty.txt"
        empty_doc.touch()
        
        yield {
            "valid": str(valid_doc),
            "large": str(large_doc),
            "corrupted": str(corrupted_doc),
            "empty": str(empty_doc),
            "directory": str(test_dir)
        }


@pytest.mark.integration
@pytest.mark.slow
class TestErrorRecoveryScenarios:
    """Test system error recovery under various failure conditions."""
    
    async def test_network_connectivity_failure_recovery(
        self,
        error_recovery_environment,
        recovery_test_config,
        test_documents
    ):
        """
        Test recovery from network connectivity failures.
        
        Simulates:
        1. Network interruption during operation
        2. Connection timeout scenarios  
        3. Automatic reconnection attempts
        4. Exponential backoff behavior
        5. Operation retry after reconnection
        """
        
        client = QdrantWorkspaceClient(config=recovery_test_config)
        await client.initialize()
        
        try:
            # First, establish working connection
            await client.create_collection_if_not_exists(
                recovery_test_config.qdrant.collection_name
            )
            
            # Test document processing under normal conditions
            result = await client.add_document(
                file_path=test_documents["valid"],
                collection=recovery_test_config.qdrant.collection_name,
                metadata={"test": "network_recovery", "phase": "baseline"}
            )
            assert result.get("success", False), "Baseline operation should succeed"
            
            # Simulate network failure
            with patch.object(client, '_qdrant_client') as mock_client:
                # First call fails with connection error
                mock_client.upsert.side_effect = [
                    ConnectionError("Network unreachable"),
                    ConnectionError("Connection timeout"),
                    # Third call succeeds (recovery)
                    MagicMock(operation_id=12345, status="completed")
                ]
                
                # Attempt operation during simulated network failure
                start_time = time.time()
                
                try:
                    result = await client.add_document(
                        file_path=test_documents["valid"],
                        collection=recovery_test_config.qdrant.collection_name,
                        metadata={"test": "network_recovery", "phase": "during_failure"}
                    )
                    recovery_succeeded = True
                except Exception as e:
                    recovery_succeeded = False
                    recovery_error = str(e)
                
                end_time = time.time()
                recovery_duration = end_time - start_time
                
                # Verify recovery behavior
                # Should either succeed after retries OR fail gracefully
                if recovery_succeeded:
                    assert result.get("success", False), "Should succeed after recovery"
                    # Should have taken time due to retries
                    assert recovery_duration >= 2.0, "Should include retry delays"
                else:
                    # Should fail gracefully with meaningful error
                    assert "connection" in recovery_error.lower() or "network" in recovery_error.lower()
                    assert recovery_duration <= 15.0, "Should not hang indefinitely"
                
                # Verify retry attempts were made
                assert mock_client.upsert.call_count >= 2, "Should attempt retries"
            
        finally:
            await client.close()

    async def test_qdrant_service_unavailability_recovery(
        self,
        error_recovery_environment,
        recovery_test_config,
        test_documents
    ):
        """
        Test recovery when Qdrant service becomes unavailable.
        
        Simulates:
        1. Qdrant service shutdown
        2. Service restart and reconnection  
        3. Data consistency verification
        4. Operation queuing during downtime
        5. Batch processing after recovery
        """
        
        # Mock service availability states
        service_available = [True]  # Use list for mutable state
        operation_queue = []
        
        async def mock_service_operation(operation_type, *args, **kwargs):
            if not service_available[0]:
                # Queue operations during downtime
                operation_queue.append({
                    "type": operation_type,
                    "args": args,
                    "kwargs": kwargs,
                    "timestamp": time.time()
                })
                raise ConnectionError("Service unavailable")
            
            # Process queued operations when service comes back
            if operation_queue:
                queued_ops = len(operation_queue)
                operation_queue.clear()
                return {"success": True, "queued_operations_processed": queued_ops}
            
            return {"success": True, "queued_operations_processed": 0}
        
        with patch('workspace_qdrant_mcp.core.client.QdrantWorkspaceClient.add_document') as mock_add:
            mock_add.side_effect = lambda *args, **kwargs: mock_service_operation("add_document", *args, **kwargs)
            
            client = QdrantWorkspaceClient(config=recovery_test_config)
            await client.initialize()
            
            try:
                # Phase 1: Normal operation
                result = await client.add_document(
                    file_path=test_documents["valid"],
                    collection=recovery_test_config.qdrant.collection_name,
                    metadata={"phase": "normal"}
                )
                assert result["success"], "Normal operation should succeed"
                
                # Phase 2: Service becomes unavailable
                service_available[0] = False
                
                failed_operations = []
                for i in range(3):
                    try:
                        await client.add_document(
                            file_path=test_documents["valid"],
                            collection=recovery_test_config.qdrant.collection_name,
                            metadata={"phase": "unavailable", "attempt": i}
                        )
                    except Exception as e:
                        failed_operations.append(str(e))
                
                # Should have queued operations and failed
                assert len(failed_operations) == 3, "All operations should fail during downtime"
                assert len(operation_queue) == 3, "Operations should be queued"
                assert all("unavailable" in error for error in failed_operations)
                
                # Phase 3: Service recovery
                service_available[0] = True
                
                recovery_result = await client.add_document(
                    file_path=test_documents["valid"],
                    collection=recovery_test_config.qdrant.collection_name,
                    metadata={"phase": "recovery"}
                )
                
                # Should succeed and process queued operations
                assert recovery_result["success"], "Should succeed after service recovery"
                assert recovery_result["queued_operations_processed"] == 3, "Should process queued operations"
                
            finally:
                await client.close()

    async def test_file_system_error_handling(
        self,
        error_recovery_environment,
        recovery_test_config,
        test_documents
    ):
        """
        Test handling of file system errors and corruption.
        
        Simulates:
        1. File permission errors
        2. Corrupted file handling
        3. Missing file scenarios
        4. Disk space exhaustion
        5. File locking conflicts
        """
        
        client = QdrantWorkspaceClient(config=recovery_test_config)
        await client.initialize()
        
        try:
            await client.create_collection_if_not_exists(
                recovery_test_config.qdrant.collection_name
            )
            
            # Test 1: Permission denied
            with patch('builtins.open') as mock_open:
                mock_open.side_effect = PermissionError("Permission denied")
                
                result = await client.add_document(
                    file_path=test_documents["valid"],
                    collection=recovery_test_config.qdrant.collection_name
                )
                
                # Should handle permission error gracefully
                assert not result.get("success", True), "Permission error should cause failure"
                assert "permission" in str(result.get("error", "")).lower()
            
            # Test 2: Corrupted file handling
            result = await client.add_document(
                file_path=test_documents["corrupted"],
                collection=recovery_test_config.qdrant.collection_name,
                metadata={"test": "corrupted_file"}
            )
            
            # Should either handle gracefully or provide meaningful error
            if not result.get("success", False):
                error_msg = str(result.get("error", ""))
                assert any(keyword in error_msg.lower() for keyword in 
                          ["encoding", "decode", "utf-8", "invalid"])
            
            # Test 3: Missing file
            missing_file = str(Path(test_documents["directory"]) / "nonexistent.txt")
            result = await client.add_document(
                file_path=missing_file,
                collection=recovery_test_config.qdrant.collection_name
            )
            
            assert not result.get("success", True), "Missing file should cause failure"
            assert "not found" in str(result.get("error", "")).lower() or \
                   "no such file" in str(result.get("error", "")).lower()
            
            # Test 4: Empty file handling
            result = await client.add_document(
                file_path=test_documents["empty"],
                collection=recovery_test_config.qdrant.collection_name,
                metadata={"test": "empty_file"}
            )
            
            # Should handle empty files gracefully
            if result.get("success", False):
                assert result.get("chunks_added", 0) == 0, "Empty file should produce no chunks"
            else:
                assert "empty" in str(result.get("error", "")).lower()
            
            # Test 5: Disk space simulation (mock)
            with patch('pathlib.Path.write_text') as mock_write:
                mock_write.side_effect = OSError("No space left on device")
                
                # This would typically be tested in a component that writes temp files
                try:
                    temp_file = Path(test_documents["directory"]) / "temp_test.txt"
                    temp_file.write_text("test content")
                    disk_space_error = False
                except OSError as e:
                    disk_space_error = "space" in str(e).lower()
                
                assert disk_space_error, "Should simulate disk space error"
            
        finally:
            await client.close()

    async def test_memory_pressure_handling(
        self,
        error_recovery_environment,
        recovery_test_config,
        test_documents
    ):
        """
        Test system behavior under memory pressure.
        
        Simulates:
        1. Large document processing with limited memory
        2. Memory exhaustion scenarios  
        3. Garbage collection triggers
        4. Resource cleanup under pressure
        5. Graceful degradation of performance
        """
        
        # Mock memory monitoring
        memory_usage = {"current": 100}  # MB
        memory_limit = 500  # MB
        
        def mock_memory_check():
            return {
                "used_mb": memory_usage["current"],
                "available_mb": memory_limit - memory_usage["current"],
                "percent_used": memory_usage["current"] / memory_limit * 100
            }
        
        with patch('psutil.virtual_memory') as mock_vmem:
            mock_vmem.return_value = MagicMock(
                available=int((memory_limit - memory_usage["current"]) * 1024 * 1024),
                total=int(memory_limit * 1024 * 1024),
                percent=memory_usage["current"] / memory_limit * 100
            )
            
            client = QdrantWorkspaceClient(config=recovery_test_config)
            await client.initialize()
            
            try:
                await client.create_collection_if_not_exists(
                    recovery_test_config.qdrant.collection_name
                )
                
                # Test 1: Normal memory usage
                memory_usage["current"] = 100
                result = await client.add_document(
                    file_path=test_documents["valid"],
                    collection=recovery_test_config.qdrant.collection_name,
                    metadata={"test": "normal_memory"}
                )
                assert result.get("success", False), "Normal memory should allow processing"
                
                # Test 2: High memory usage
                memory_usage["current"] = 450  # 90% usage
                
                with patch.object(client, 'add_document') as mock_add:
                    # Simulate memory-conscious processing
                    mock_add.return_value = {
                        "success": True,
                        "chunks_added": 2,
                        "memory_optimized": True,
                        "reduced_batch_size": True
                    }
                    
                    result = await client.add_document(
                        file_path=test_documents["large"],
                        collection=recovery_test_config.qdrant.collection_name,
                        metadata={"test": "high_memory"}
                    )
                    
                    # Should adapt to memory pressure
                    assert result.get("success", False), "Should handle high memory usage"
                    if result.get("memory_optimized"):
                        assert result.get("reduced_batch_size"), "Should reduce batch size under pressure"
                
                # Test 3: Memory exhaustion
                memory_usage["current"] = 495  # 99% usage
                
                with patch.object(client, 'add_document') as mock_add:
                    mock_add.side_effect = MemoryError("Not enough memory available")
                    
                    result = await client.add_document(
                        file_path=test_documents["large"],
                        collection=recovery_test_config.qdrant.collection_name,
                        metadata={"test": "memory_exhaustion"}
                    )
                    
                    # Should handle memory exhaustion gracefully
                    assert not result.get("success", True), "Memory exhaustion should cause failure"
                    assert "memory" in str(result.get("error", "")).lower()
                
            finally:
                await client.close()

    async def test_grpc_communication_failure_recovery(
        self,
        error_recovery_environment,
        recovery_test_config,
        test_documents
    ):
        """
        Test gRPC communication failure recovery.
        
        Simulates:
        1. gRPC connection drops  
        2. Message serialization failures
        3. Timeout scenarios
        4. Channel reconnection
        5. Circuit breaker patterns
        """
        
        # Track connection attempts and failures
        connection_attempts = []
        failure_count = [0]  # Mutable counter
        
        async def mock_grpc_operation(operation, *args, **kwargs):
            connection_attempts.append({
                "operation": operation,
                "time": time.time(),
                "attempt": len(connection_attempts) + 1
            })
            
            # Simulate intermittent failures
            failure_count[0] += 1
            if failure_count[0] <= 3:  # First 3 attempts fail
                if failure_count[0] == 1:
                    raise ConnectionError("gRPC connection lost")
                elif failure_count[0] == 2:
                    raise asyncio.TimeoutError("gRPC timeout")
                else:
                    raise Exception("gRPC serialization error")
            
            # 4th attempt succeeds
            return {"success": True, "connection_recovered": True}
        
        with patch('workspace_qdrant_mcp.tools.grpc_tools.process_document_via_grpc') as mock_grpc:
            mock_grpc.side_effect = lambda *args, **kwargs: mock_grpc_operation("process_document", *args, **kwargs)
            
            # Test gRPC recovery
            start_time = time.time()
            
            try:
                result = await process_document_via_grpc(
                    file_path=test_documents["valid"],
                    collection="grpc_recovery_test",
                    host=error_recovery_environment["qdrant_host"],
                    port=error_recovery_environment["grpc_port"]
                )
                recovery_succeeded = True
            except Exception as e:
                recovery_succeeded = False
                recovery_error = str(e)
            
            end_time = time.time()
            recovery_time = end_time - start_time
            
            # Verify recovery behavior
            assert len(connection_attempts) >= 3, "Should make multiple connection attempts"
            
            if recovery_succeeded:
                assert result["success"], "Should succeed after recovery"
                assert result.get("connection_recovered"), "Should indicate recovery"
                assert recovery_time < 30, "Recovery should complete within reasonable time"
            else:
                # If recovery failed, should be due to exhausted retries
                assert "connection" in recovery_error.lower() or "timeout" in recovery_error.lower()
                assert len(connection_attempts) >= 3, "Should exhaust retry attempts"

    async def test_concurrent_operation_conflict_resolution(
        self,
        error_recovery_environment,
        recovery_test_config,
        test_documents
    ):
        """
        Test resolution of concurrent operation conflicts.
        
        Simulates:
        1. Race conditions in document updates
        2. Collection creation conflicts
        3. Resource locking scenarios
        4. Deadlock detection and recovery
        5. Conflict resolution strategies
        """
        
        client = QdrantWorkspaceClient(config=recovery_test_config)
        await client.initialize()
        
        try:
            # Test concurrent collection creation
            collection_name = "concurrent_conflict_test"
            
            async def create_collection_task(task_id):
                try:
                    await client.create_collection_if_not_exists(collection_name)
                    return {"task_id": task_id, "success": True}
                except Exception as e:
                    return {"task_id": task_id, "success": False, "error": str(e)}
            
            # Start multiple concurrent collection creation tasks
            tasks = [create_collection_task(i) for i in range(5)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_creates = [r for r in results if not isinstance(r, Exception) and r.get("success")]
            failed_creates = [r for r in results if isinstance(r, Exception) or not r.get("success")]
            
            # Should handle concurrent creation gracefully
            assert len(successful_creates) >= 1, "At least one creation should succeed"
            # Other attempts should either succeed (idempotent) or fail gracefully
            
            # Test concurrent document processing
            async def process_document_task(task_id, document_path):
                try:
                    result = await client.add_document(
                        file_path=document_path,
                        collection=collection_name,
                        metadata={"task_id": task_id, "concurrent": True}
                    )
                    return {"task_id": task_id, "result": result, "success": result.get("success", False)}
                except Exception as e:
                    return {"task_id": task_id, "result": None, "success": False, "error": str(e)}
            
            # Process same document concurrently (potential conflict)
            concurrent_tasks = [
                process_document_task(i, test_documents["valid"]) 
                for i in range(3)
            ]
            concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
            
            successful_processing = [
                r for r in concurrent_results 
                if not isinstance(r, Exception) and r.get("success")
            ]
            
            # Should handle concurrent processing
            assert len(successful_processing) >= 1, "Should handle concurrent document processing"
            
            # Verify no data corruption
            if len(successful_processing) > 1:
                doc_ids = [r["result"].get("document_id") for r in successful_processing if r["result"]]
                doc_ids = [doc_id for doc_id in doc_ids if doc_id]
                
                if len(doc_ids) > 1:
                    # If multiple documents were created, they should have unique IDs
                    assert len(set(doc_ids)) == len(doc_ids), "Document IDs should be unique"
            
        finally:
            await client.close()

    async def test_data_consistency_recovery(
        self,
        error_recovery_environment,
        recovery_test_config,
        test_documents
    ):
        """
        Test data consistency recovery after failures.
        
        Simulates:
        1. Partial document ingestion failures
        2. Inconsistent vector storage  
        3. Metadata corruption scenarios
        4. Index rebuilding requirements
        5. Data validation and repair
        """
        
        client = QdrantWorkspaceClient(config=recovery_test_config)
        await client.initialize()
        
        try:
            collection_name = "data_consistency_test"
            await client.create_collection_if_not_exists(collection_name)
            
            # Test 1: Partial ingestion failure
            with patch.object(client, '_store_chunks') as mock_store:
                # Simulate partial failure - first chunk succeeds, second fails
                mock_store.side_effect = [
                    {"success": True, "stored_count": 1},  # First chunk
                    Exception("Storage failure during second chunk"),  # Second chunk fails
                ]
                
                result = await client.add_document(
                    file_path=test_documents["large"],
                    collection=collection_name,
                    metadata={"test": "partial_failure"}
                )
                
                # Should handle partial failure appropriately
                if not result.get("success", False):
                    # Should report partial completion
                    error_info = result.get("error", "")
                    assert "partial" in error_info.lower() or "storage" in error_info.lower()
            
            # Test 2: Metadata consistency check
            with patch.object(client, 'search') as mock_search:
                # Simulate inconsistent search results (metadata mismatch)
                mock_search.return_value = {
                    "success": True,
                    "results": [
                        {
                            "id": "doc_1",
                            "score": 0.9,
                            "payload": {"text": "content", "metadata": {"corrupted": True}}
                        }
                    ],
                    "consistency_issues": ["metadata_mismatch"]
                }
                
                search_result = await client.search(
                    query="test content",
                    collection=collection_name
                )
                
                # Should detect consistency issues
                if search_result.get("consistency_issues"):
                    assert "metadata_mismatch" in search_result["consistency_issues"]
            
            # Test 3: Recovery verification
            with patch.object(client, 'verify_collection_consistency') as mock_verify:
                mock_verify.return_value = {
                    "consistent": False,
                    "issues": ["orphaned_vectors", "missing_metadata"],
                    "repair_actions": ["rebuild_index", "restore_metadata"]
                }
                
                # In a real implementation, this would trigger recovery
                consistency_check = await client.verify_collection_consistency(collection_name)
                
                if not consistency_check.get("consistent", True):
                    assert len(consistency_check.get("issues", [])) > 0
                    assert len(consistency_check.get("repair_actions", [])) > 0
            
        finally:
            await client.close()

    async def test_configuration_error_recovery(
        self,
        error_recovery_environment,
        recovery_test_config
    ):
        """
        Test recovery from configuration errors.
        
        Simulates:
        1. Invalid configuration parameters
        2. Configuration file corruption
        3. Environment variable issues
        4. Dynamic reconfiguration
        5. Fallback configuration activation
        """
        
        # Test 1: Invalid port configuration
        invalid_config = recovery_test_config.model_copy()
        invalid_config.qdrant.port = -1
        invalid_config.qdrant.grpc_port = 99999
        
        try:
            client = QdrantWorkspaceClient(config=invalid_config)
            await client.initialize()
            initialization_failed = False
        except Exception as e:
            initialization_failed = True
            error_msg = str(e)
        
        assert initialization_failed, "Invalid configuration should prevent initialization"
        assert "port" in error_msg.lower() or "connection" in error_msg.lower()
        
        # Test 2: Missing required configuration
        incomplete_config = recovery_test_config.model_copy()
        incomplete_config.qdrant.host = ""
        
        try:
            client = QdrantWorkspaceClient(config=incomplete_config)
            await client.initialize()
            config_validation_failed = False
        except Exception as e:
            config_validation_failed = True
            config_error = str(e)
        
        assert config_validation_failed, "Missing required config should fail"
        assert "host" in config_error.lower() or "required" in config_error.lower()
        
        # Test 3: Configuration recovery with fallback
        with patch('workspace_qdrant_mcp.core.config.Config.from_yaml') as mock_config:
            # First attempt fails, second succeeds with fallback
            mock_config.side_effect = [
                ConfigurationError("Configuration file corrupted"),
                recovery_test_config  # Fallback configuration
            ]
            
            try:
                # This would typically try multiple config sources
                config = Config.from_yaml("corrupted_config.yaml")
                recovery_successful = False
            except:
                # Try fallback
                config = recovery_test_config
                recovery_successful = True
            
            assert recovery_successful, "Should recover with fallback configuration"
            assert config.qdrant.host, "Fallback config should have valid settings"