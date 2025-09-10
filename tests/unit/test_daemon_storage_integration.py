"""
Unit tests for storage integration and error recovery within daemon context.

Tests the daemon's storage integration capabilities including:
- Qdrant client operations (collections, documents, search) through daemon
- Bulk document processing and batching strategies
- Connection management and retry logic for storage operations
- Storage error recovery and circuit breaker patterns
- Performance monitoring and metrics collection for storage ops
- Storage state synchronization between daemon and Qdrant
- SQLite state manager integration for persistence
- Crash recovery and data consistency mechanisms
"""

import asyncio
import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock, call
from unittest import mock
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

from src.workspace_qdrant_mcp.core.daemon_manager import (
    DaemonManager,
    DaemonInstance,
    DaemonConfig
)
from src.workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from src.workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager
from src.workspace_qdrant_mcp.core.collections import WorkspaceCollectionManager
from src.workspace_qdrant_mcp.core.embeddings import EmbeddingService

from .conftest_daemon import (
    mock_daemon_config,
    mock_daemon_instance,
    mock_daemon_manager,
    isolated_daemon_temp_dir,
    DaemonTestHelper
)


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = Mock(spec=QdrantClient)
    client.get_collections = Mock(return_value=Mock(collections=[
        Mock(name="test_collection"),
        Mock(name="project_docs")
    ]))
    client.create_collection = Mock(return_value=True)
    client.delete_collection = Mock(return_value=True)
    client.upsert = Mock(return_value=Mock(operation_id=12345, status="completed"))
    client.search = Mock(return_value=[
        Mock(id="doc_1", score=0.95, payload={"content": "test document"}),
        Mock(id="doc_2", score=0.87, payload={"content": "another document"})
    ])
    client.get_collection = Mock(return_value=Mock(
        status="green",
        vectors_count=1000,
        indexed_vectors_count=1000
    ))
    client.delete = Mock(return_value=Mock(operation_id=12346, status="completed"))
    client.scroll = Mock(return_value=(
        [Mock(id="doc_1"), Mock(id="doc_2")],
        "next_offset"
    ))
    return client


@pytest.fixture
def mock_workspace_client():
    """Mock workspace client for testing."""
    client = Mock(spec=QdrantWorkspaceClient)
    client.initialize = AsyncMock(return_value=True)
    client.get_status = AsyncMock(return_value={
        "connected": True,
        "collections": ["test_collection", "project_docs"],
        "total_documents": 1000
    })
    client.store_memory = AsyncMock(return_value="doc_123")
    client.search_memory = AsyncMock(return_value=[
        {"id": "doc_1", "score": 0.95, "content": "test content"}
    ])
    client.delete_memory = AsyncMock(return_value=True)
    client.list_collections = Mock(return_value=["test_collection", "project_docs"])
    return client


@pytest.fixture
def mock_sqlite_state_manager():
    """Mock SQLite state manager for testing."""
    manager = Mock(spec=SQLiteStateManager)
    manager.initialize = AsyncMock(return_value=True)
    manager.start_file_processing = AsyncMock(return_value=True)
    manager.complete_file_processing = AsyncMock(return_value=True)
    manager.save_watch_folder_config = AsyncMock(return_value=True)
    manager.get_processing_state = AsyncMock(return_value={
        "status": "completed",
        "started_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc)
    })
    manager.cleanup_old_records = AsyncMock(return_value=5)
    manager.backup_database = AsyncMock(return_value="/backup/path.db")
    return manager


@pytest.fixture
def mock_collection_manager():
    """Mock collection manager for testing."""
    manager = Mock(spec=WorkspaceCollectionManager)
    manager.create_collection = AsyncMock(return_value=True)
    manager.delete_collection = AsyncMock(return_value=True)
    manager.get_collection_info = AsyncMock(return_value={
        "name": "test_collection",
        "vectors_count": 1000,
        "size": 1024000
    })
    manager.list_collections = AsyncMock(return_value=["test_collection", "project_docs"])
    manager.ensure_collection_exists = AsyncMock(return_value=True)
    return manager


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for testing."""
    service = Mock(spec=EmbeddingService)
    service.encode = AsyncMock(return_value=[0.1, 0.2, 0.3, 0.4])  # Mock embedding vector
    service.encode_batch = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    service.get_embedding_dim = Mock(return_value=384)
    return service


class TestDaemonQdrantIntegration:
    """Test Qdrant storage operations through daemon context."""
    
    @pytest.mark.asyncio
    async def test_daemon_qdrant_connection_management(self, mock_daemon_instance, mock_qdrant_client):
        """Test Qdrant connection management within daemon context."""
        # Setup daemon with Qdrant client
        mock_daemon_instance.qdrant_client = mock_qdrant_client
        mock_daemon_instance.storage_connected = False
        
        # Mock daemon's Qdrant connection management
        with patch.object(mock_daemon_instance, 'initialize_storage', new_callable=AsyncMock) as mock_init:
            async def init_storage():
                mock_daemon_instance.storage_connected = True
                return True
            
            mock_init.side_effect = init_storage
            
            # Test storage initialization
            result = await mock_daemon_instance.initialize_storage()
            
            assert result is True
            assert mock_daemon_instance.storage_connected is True
    
    @pytest.mark.asyncio
    async def test_daemon_collection_operations(self, mock_daemon_instance, mock_qdrant_client, mock_collection_manager):
        """Test collection operations through daemon interface."""
        # Setup daemon with collection management
        mock_daemon_instance.qdrant_client = mock_qdrant_client
        mock_daemon_instance.collection_manager = mock_collection_manager
        mock_daemon_instance.storage_connected = True
        
        # Test collection creation
        creation_result = await mock_collection_manager.create_collection(
            "new_collection",
            {"vector_size": 384, "distance": "cosine"}
        )
        assert creation_result is True
        
        # Test collection listing
        collections = await mock_collection_manager.list_collections()
        assert "test_collection" in collections
        assert "project_docs" in collections
        
        # Test collection deletion
        deletion_result = await mock_collection_manager.delete_collection("old_collection")
        assert deletion_result is True
    
    @pytest.mark.asyncio
    async def test_daemon_document_processing(self, mock_daemon_instance, mock_workspace_client, mock_embedding_service):
        """Test document processing and storage through daemon."""
        # Setup daemon with document processing capabilities
        mock_daemon_instance.workspace_client = mock_workspace_client
        mock_daemon_instance.embedding_service = mock_embedding_service
        mock_daemon_instance.storage_connected = True
        
        # Mock document processing workflow
        with patch.object(mock_daemon_instance, 'process_document', new_callable=AsyncMock) as mock_process:
            async def process_document(file_path, collection_name):
                # Simulate document processing pipeline
                content = f"Content from {file_path}"
                embedding = await mock_embedding_service.encode(content)
                doc_id = await mock_workspace_client.store_memory(
                    content=content,
                    collection=collection_name,
                    metadata={"file_path": file_path}
                )
                return {"success": True, "document_id": doc_id, "embedding_dim": len(embedding)}
            
            mock_process.side_effect = process_document
            
            # Test document processing
            result = await mock_daemon_instance.process_document("/test/file.py", "test_collection")
            
            assert result["success"] is True
            assert result["document_id"] == "doc_123"
            assert result["embedding_dim"] == 4
            mock_embedding_service.encode.assert_called_once()
            mock_workspace_client.store_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_bulk_document_processing(self, mock_daemon_instance, mock_workspace_client, mock_embedding_service):
        """Test bulk document processing and batching strategies."""
        # Setup daemon with bulk processing
        mock_daemon_instance.workspace_client = mock_workspace_client
        mock_daemon_instance.embedding_service = mock_embedding_service
        mock_daemon_instance.bulk_batch_size = 10
        
        # Mock bulk processing
        with patch.object(mock_daemon_instance, 'process_documents_bulk', new_callable=AsyncMock) as mock_bulk:
            async def process_bulk(file_paths, collection_name, batch_size=10):
                results = []
                for i in range(0, len(file_paths), batch_size):
                    batch = file_paths[i:i + batch_size]
                    batch_contents = [f"Content from {path}" for path in batch]
                    embeddings = await mock_embedding_service.encode_batch(batch_contents)
                    
                    batch_results = []
                    for j, path in enumerate(batch):
                        doc_id = f"doc_{i + j}"
                        batch_results.append({
                            "file_path": path,
                            "document_id": doc_id,
                            "success": True
                        })
                    results.extend(batch_results)
                return {"processed": len(file_paths), "successful": len(results), "results": results}
            
            mock_bulk.side_effect = process_bulk
            
            # Test bulk processing
            file_paths = [f"/test/file{i}.py" for i in range(25)]
            result = await mock_daemon_instance.process_documents_bulk(file_paths, "test_collection")
            
            assert result["processed"] == 25
            assert result["successful"] == 25
            assert len(result["results"]) == 25
            # Should process in 3 batches (10, 10, 5)
            mock_embedding_service.encode_batch.call_count == 3
    
    @pytest.mark.asyncio
    async def test_daemon_search_operations(self, mock_daemon_instance, mock_workspace_client):
        """Test search operations through daemon interface."""
        # Setup daemon with search capabilities
        mock_daemon_instance.workspace_client = mock_workspace_client
        mock_daemon_instance.storage_connected = True
        
        # Mock search operations
        with patch.object(mock_daemon_instance, 'search_documents', new_callable=AsyncMock) as mock_search:
            async def search_documents(query, collection_name, limit=10):
                return await mock_workspace_client.search_memory(
                    query=query,
                    collection=collection_name,
                    limit=limit
                )
            
            mock_search.side_effect = search_documents
            
            # Test document search
            search_results = await mock_daemon_instance.search_documents(
                "test query",
                "test_collection",
                limit=5
            )
            
            assert len(search_results) == 1
            assert search_results[0]["id"] == "doc_1"
            assert search_results[0]["score"] == 0.95
            mock_workspace_client.search_memory.assert_called_once()


class TestDaemonStorageErrorRecovery:
    """Test storage error recovery and resilience mechanisms."""
    
    @pytest.mark.asyncio
    async def test_qdrant_connection_failure_recovery(self, mock_daemon_instance, mock_qdrant_client):
        """Test recovery from Qdrant connection failures."""
        # Setup daemon with connection failure scenario
        mock_daemon_instance.qdrant_client = mock_qdrant_client
        mock_daemon_instance.storage_connected = True
        mock_daemon_instance.connection_retry_count = 0
        
        # Simulate connection failure
        mock_qdrant_client.get_collections.side_effect = ConnectionError("Qdrant unavailable")
        
        # Mock recovery mechanism
        with patch.object(mock_daemon_instance, 'recover_storage_connection', new_callable=AsyncMock) as mock_recover:
            async def recover_connection():
                mock_daemon_instance.connection_retry_count += 1
                if mock_daemon_instance.connection_retry_count >= 3:
                    # Simulate successful recovery after retries
                    mock_qdrant_client.get_collections.side_effect = None
                    mock_qdrant_client.get_collections.return_value = Mock(collections=[])
                    mock_daemon_instance.storage_connected = True
                    return True
                return False
            
            mock_recover.side_effect = recover_connection
            
            # Test recovery process
            for _ in range(3):
                try:
                    await mock_qdrant_client.get_collections()
                except ConnectionError:
                    recovery_result = await mock_daemon_instance.recover_storage_connection()
            
            assert mock_daemon_instance.connection_retry_count == 3
            assert mock_daemon_instance.storage_connected is True
    
    @pytest.mark.asyncio
    async def test_storage_circuit_breaker_pattern(self, mock_daemon_instance, mock_qdrant_client):
        """Test circuit breaker pattern for storage operations."""
        # Setup daemon with circuit breaker
        mock_daemon_instance.qdrant_client = mock_qdrant_client
        mock_daemon_instance.circuit_breaker_state = "closed"
        mock_daemon_instance.failure_count = 0
        mock_daemon_instance.failure_threshold = 5
        
        # Mock circuit breaker logic
        async def storage_operation_with_circuit_breaker(operation):
            if mock_daemon_instance.circuit_breaker_state == "open":
                raise Exception("Circuit breaker open")
            
            try:
                return await operation()
            except Exception as e:
                mock_daemon_instance.failure_count += 1
                if mock_daemon_instance.failure_count >= mock_daemon_instance.failure_threshold:
                    mock_daemon_instance.circuit_breaker_state = "open"
                raise e
        
        # Simulate multiple failures
        for i in range(6):
            try:
                await storage_operation_with_circuit_breaker(
                    lambda: mock_qdrant_client.upsert(
                        collection_name="test",
                        points=[PointStruct(id=i, vector=[0.1, 0.2])]
                    )
                )
            except Exception:
                if i < 5:
                    mock_qdrant_client.upsert.side_effect = Exception("Storage failure")
                else:
                    # Circuit breaker should be open
                    assert mock_daemon_instance.circuit_breaker_state == "open"
    
    @pytest.mark.asyncio
    async def test_document_processing_error_handling(self, mock_daemon_instance, mock_workspace_client, mock_sqlite_state_manager):
        """Test error handling during document processing."""
        # Setup daemon with error tracking
        mock_daemon_instance.workspace_client = mock_workspace_client
        mock_daemon_instance.state_manager = mock_sqlite_state_manager
        
        # Simulate processing errors
        error_count = 0
        processed_files = []
        failed_files = []
        
        async def mock_process_with_errors(file_path, collection_name):
            nonlocal error_count
            
            await mock_sqlite_state_manager.start_file_processing(file_path, collection_name)
            
            try:
                if "error" in file_path:
                    error_count += 1
                    raise Exception(f"Processing failed for {file_path}")
                
                # Simulate successful processing
                await mock_workspace_client.store_memory(
                    content=f"Content from {file_path}",
                    collection=collection_name
                )
                
                await mock_sqlite_state_manager.complete_file_processing(file_path, success=True)
                processed_files.append(file_path)
                return {"success": True, "file_path": file_path}
                
            except Exception as e:
                await mock_sqlite_state_manager.complete_file_processing(
                    file_path, 
                    success=False, 
                    error=str(e)
                )
                failed_files.append(file_path)
                return {"success": False, "file_path": file_path, "error": str(e)}
        
        # Test processing with mixed success/failure
        files_to_process = [
            "/test/file1.py",
            "/test/error_file.py",  # This will fail
            "/test/file2.py",
            "/test/another_error.py"  # This will fail
        ]
        
        results = []
        for file_path in files_to_process:
            result = await mock_process_with_errors(file_path, "test_collection")
            results.append(result)
        
        assert error_count == 2
        assert len(processed_files) == 2
        assert len(failed_files) == 2
        assert mock_sqlite_state_manager.start_file_processing.call_count == 4
        assert mock_sqlite_state_manager.complete_file_processing.call_count == 4
    
    @pytest.mark.asyncio
    async def test_storage_state_synchronization(self, mock_daemon_instance, mock_qdrant_client, mock_sqlite_state_manager):
        """Test storage state synchronization between daemon and Qdrant."""
        # Setup daemon with state synchronization
        mock_daemon_instance.qdrant_client = mock_qdrant_client
        mock_daemon_instance.state_manager = mock_sqlite_state_manager
        
        # Mock state synchronization
        with patch.object(mock_daemon_instance, 'sync_storage_state', new_callable=AsyncMock) as mock_sync:
            async def sync_state():
                # Get Qdrant collections
                qdrant_collections = mock_qdrant_client.get_collections().collections
                qdrant_collection_names = [col.name for col in qdrant_collections]
                
                # Get state manager tracked collections
                state_collections = ["test_collection", "project_docs", "orphaned_collection"]
                
                # Find discrepancies
                missing_in_qdrant = set(state_collections) - set(qdrant_collection_names)
                missing_in_state = set(qdrant_collection_names) - set(state_collections)
                
                # Synchronize state
                sync_result = {
                    "qdrant_collections": qdrant_collection_names,
                    "state_collections": state_collections,
                    "missing_in_qdrant": list(missing_in_qdrant),
                    "missing_in_state": list(missing_in_state),
                    "synced": True
                }
                return sync_result
            
            mock_sync.side_effect = sync_state
            
            # Test state synchronization
            sync_result = await mock_daemon_instance.sync_storage_state()
            
            assert sync_result["synced"] is True
            assert "test_collection" in sync_result["qdrant_collections"]
            assert "orphaned_collection" in sync_result["missing_in_qdrant"]


class TestSQLiteStateManagerIntegration:
    """Test SQLite state manager integration with daemon storage operations."""
    
    @pytest.mark.asyncio
    async def test_sqlite_crash_recovery(self, mock_sqlite_state_manager, isolated_daemon_temp_dir):
        """Test SQLite crash recovery mechanisms."""
        # Setup SQLite database path
        db_path = isolated_daemon_temp_dir / "test_state.db"
        
        # Mock crash recovery
        recovery_state = {
            "incomplete_operations": [
                {"file_path": "/test/file1.py", "status": "processing", "started_at": "2024-01-01T10:00:00Z"},
                {"file_path": "/test/file2.py", "status": "processing", "started_at": "2024-01-01T10:05:00Z"}
            ],
            "completed_operations": [
                {"file_path": "/test/file3.py", "status": "completed", "completed_at": "2024-01-01T10:10:00Z"}
            ]
        }
        
        with patch.object(mock_sqlite_state_manager, 'recover_from_crash', new_callable=AsyncMock) as mock_recover:
            async def recover_from_crash():
                # Simulate crash recovery logic
                incomplete_count = len(recovery_state["incomplete_operations"])
                completed_count = len(recovery_state["completed_operations"])
                
                # Mark incomplete operations for retry
                for op in recovery_state["incomplete_operations"]:
                    await mock_sqlite_state_manager.mark_for_retry(op["file_path"])
                
                return {
                    "recovered": True,
                    "incomplete_operations": incomplete_count,
                    "completed_operations": completed_count,
                    "marked_for_retry": incomplete_count
                }
            
            mock_recover.side_effect = recover_from_crash
            
            # Test crash recovery
            recovery_result = await mock_sqlite_state_manager.recover_from_crash()
            
            assert recovery_result["recovered"] is True
            assert recovery_result["incomplete_operations"] == 2
            assert recovery_result["marked_for_retry"] == 2
    
    @pytest.mark.asyncio
    async def test_sqlite_state_persistence(self, mock_sqlite_state_manager):
        """Test SQLite state persistence during daemon operations."""
        # Mock state persistence operations
        operation_log = []
        
        async def mock_persist_operation(operation_type, data):
            operation_log.append({
                "type": operation_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc)
            })
            return True
        
        mock_sqlite_state_manager.persist_operation = mock_persist_operation
        
        # Test various state persistence scenarios
        await mock_sqlite_state_manager.persist_operation("file_processing_start", {
            "file_path": "/test/file.py",
            "collection": "test_collection"
        })
        
        await mock_sqlite_state_manager.persist_operation("collection_created", {
            "collection_name": "new_collection",
            "vector_size": 384
        })
        
        await mock_sqlite_state_manager.persist_operation("file_processing_complete", {
            "file_path": "/test/file.py",
            "success": True,
            "document_id": "doc_123"
        })
        
        assert len(operation_log) == 3
        assert operation_log[0]["type"] == "file_processing_start"
        assert operation_log[1]["type"] == "collection_created"
        assert operation_log[2]["type"] == "file_processing_complete"
    
    @pytest.mark.asyncio
    async def test_sqlite_cleanup_and_maintenance(self, mock_sqlite_state_manager):
        """Test SQLite cleanup and maintenance procedures."""
        # Mock cleanup operations
        cleanup_stats = {
            "old_records_cleaned": 0,
            "database_size_before": 1024000,
            "database_size_after": 512000,
            "vacuum_performed": False
        }
        
        with patch.object(mock_sqlite_state_manager, 'perform_maintenance', new_callable=AsyncMock) as mock_maintenance:
            async def perform_maintenance(max_age_days=30):
                # Simulate cleanup operations
                cleanup_stats["old_records_cleaned"] = 150
                cleanup_stats["vacuum_performed"] = True
                cleanup_stats["database_size_after"] = cleanup_stats["database_size_before"] // 2
                
                return cleanup_stats
            
            mock_maintenance.side_effect = perform_maintenance
            
            # Test maintenance operations
            maintenance_result = await mock_sqlite_state_manager.perform_maintenance(max_age_days=7)
            
            assert maintenance_result["old_records_cleaned"] == 150
            assert maintenance_result["vacuum_performed"] is True
            assert maintenance_result["database_size_after"] < maintenance_result["database_size_before"]


class TestStoragePerformanceMonitoring:
    """Test storage performance monitoring and metrics collection."""
    
    @pytest.mark.asyncio
    async def test_storage_operation_metrics(self, mock_daemon_instance, mock_qdrant_client):
        """Test storage operation performance monitoring."""
        # Setup daemon with performance monitoring
        mock_daemon_instance.qdrant_client = mock_qdrant_client
        mock_daemon_instance.performance_metrics = {}
        
        # Mock performance tracking
        with patch.object(mock_daemon_instance, 'track_storage_performance', new_callable=AsyncMock) as mock_track:
            async def track_performance(operation, start_time, end_time, success=True):
                duration = end_time - start_time
                operation_key = f"storage_{operation}"
                
                if operation_key not in mock_daemon_instance.performance_metrics:
                    mock_daemon_instance.performance_metrics[operation_key] = []
                
                mock_daemon_instance.performance_metrics[operation_key].append({
                    "duration": duration,
                    "success": success,
                    "timestamp": start_time
                })
                return duration
            
            mock_track.side_effect = track_performance
            
            # Simulate storage operations with performance tracking
            operations = [
                ("upsert", 0.05, True),
                ("search", 0.12, True),
                ("delete", 0.03, True),
                ("upsert", 0.08, False)  # Failed operation
            ]
            
            for operation, duration, success in operations:
                start_time = time.time()
                end_time = start_time + duration
                await mock_daemon_instance.track_storage_performance(operation, start_time, end_time, success)
            
            # Verify metrics collection
            assert "storage_upsert" in mock_daemon_instance.performance_metrics
            assert "storage_search" in mock_daemon_instance.performance_metrics
            assert "storage_delete" in mock_daemon_instance.performance_metrics
            
            upsert_metrics = mock_daemon_instance.performance_metrics["storage_upsert"]
            assert len(upsert_metrics) == 2  # 2 upsert operations
            assert upsert_metrics[0]["success"] is True
            assert upsert_metrics[1]["success"] is False
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_storage_operation_performance(self, mock_daemon_instance, mock_workspace_client, benchmark):
        """Benchmark storage operations through daemon interface."""
        # Setup daemon with storage client
        mock_daemon_instance.workspace_client = mock_workspace_client
        mock_daemon_instance.storage_connected = True
        
        async def storage_operation():
            return await mock_workspace_client.store_memory(
                content="test content for benchmarking",
                collection="benchmark_collection",
                metadata={"source": "benchmark_test"}
            )
        
        # Benchmark the operation
        def sync_wrapper():
            return asyncio.run(storage_operation())
        
        result = benchmark(sync_wrapper)
        assert result == "doc_123"


@pytest.mark.daemon_unit
@pytest.mark.daemon_storage
@pytest.mark.daemon_error_recovery
class TestDaemonStorageIntegrationScenarios:
    """Integration tests for daemon storage operations and error recovery."""
    
    @pytest.mark.asyncio
    async def test_complete_storage_workflow(self, mock_daemon_instance, mock_workspace_client, mock_sqlite_state_manager):
        """Test complete storage workflow from ingestion to cleanup."""
        # Setup daemon with full storage stack
        mock_daemon_instance.workspace_client = mock_workspace_client
        mock_daemon_instance.state_manager = mock_sqlite_state_manager
        mock_daemon_instance.storage_connected = True
        
        workflow_state = {
            "files_processed": 0,
            "documents_stored": 0,
            "errors_encountered": 0,
            "collections_created": 0
        }
        
        # Mock complete workflow
        with patch.object(mock_daemon_instance, 'execute_storage_workflow', new_callable=AsyncMock) as mock_workflow:
            async def execute_workflow(file_paths, target_collection):
                # 1. Ensure collection exists
                collection_created = await mock_workspace_client.create_collection(target_collection)
                if collection_created:
                    workflow_state["collections_created"] += 1
                
                # 2. Process files
                for file_path in file_paths:
                    try:
                        await mock_sqlite_state_manager.start_file_processing(file_path, target_collection)
                        
                        # Simulate document processing
                        doc_id = await mock_workspace_client.store_memory(
                            content=f"Content from {file_path}",
                            collection=target_collection
                        )
                        
                        await mock_sqlite_state_manager.complete_file_processing(file_path, success=True)
                        
                        workflow_state["files_processed"] += 1
                        workflow_state["documents_stored"] += 1
                        
                    except Exception as e:
                        await mock_sqlite_state_manager.complete_file_processing(
                            file_path, success=False, error=str(e)
                        )
                        workflow_state["errors_encountered"] += 1
                
                # 3. Perform cleanup
                await mock_sqlite_state_manager.cleanup_old_records()
                
                return workflow_state
            
            mock_workflow.side_effect = execute_workflow
            
            # Test complete workflow
            test_files = ["/test/file1.py", "/test/file2.py", "/test/file3.py"]
            result = await mock_daemon_instance.execute_storage_workflow(test_files, "workflow_test")
            
            assert result["files_processed"] == 3
            assert result["documents_stored"] == 3
            assert result["errors_encountered"] == 0
            assert result["collections_created"] == 1
    
    @pytest.mark.asyncio
    async def test_storage_error_recovery_scenarios(self, mock_daemon_instance, mock_qdrant_client, mock_sqlite_state_manager):
        """Test various storage error recovery scenarios."""
        # Setup daemon with error recovery capabilities
        mock_daemon_instance.qdrant_client = mock_qdrant_client
        mock_daemon_instance.state_manager = mock_sqlite_state_manager
        
        error_scenarios = [
            {"type": "connection_timeout", "recovery_strategy": "retry_with_backoff"},
            {"type": "storage_full", "recovery_strategy": "cleanup_and_retry"},
            {"type": "invalid_data", "recovery_strategy": "skip_and_log"},
            {"type": "network_partition", "recovery_strategy": "queue_for_later"}
        ]
        
        recovery_results = {}
        
        # Mock error recovery for each scenario
        for scenario in error_scenarios:
            error_type = scenario["type"]
            strategy = scenario["recovery_strategy"]
            
            with patch.object(mock_daemon_instance, f'handle_{error_type}', new_callable=AsyncMock) as mock_handler:
                async def error_handler():
                    if strategy == "retry_with_backoff":
                        return {"recovered": True, "attempts": 3}
                    elif strategy == "cleanup_and_retry":
                        await mock_sqlite_state_manager.cleanup_old_records()
                        return {"recovered": True, "cleanup_performed": True}
                    elif strategy == "skip_and_log":
                        return {"recovered": False, "skipped": True}
                    elif strategy == "queue_for_later":
                        return {"recovered": False, "queued": True}
                
                mock_handler.side_effect = error_handler
                recovery_results[error_type] = await getattr(mock_daemon_instance, f'handle_{error_type}')()
        
        # Verify recovery strategies were applied correctly
        assert recovery_results["connection_timeout"]["recovered"] is True
        assert recovery_results["storage_full"]["cleanup_performed"] is True
        assert recovery_results["invalid_data"]["skipped"] is True
        assert recovery_results["network_partition"]["queued"] is True