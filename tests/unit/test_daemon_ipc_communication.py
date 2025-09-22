"""
Unit tests for IPC communication and service discovery between daemon components.

Tests the inter-process communication capabilities including:
- gRPC client-server communication between Python and Rust components
- Service discovery and registration mechanisms
- Network discovery protocols and daemon endpoint detection
- Health checking and heartbeat mechanisms across components
- Connection pooling and retry logic for reliable communication
- Message serialization/deserialization for gRPC protocols
- Timeout and error handling in distributed communication
- Service registry operations for multi-daemon coordination
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock, call
from unittest import mock
import pytest
import grpc
from grpc import StatusCode
from google.protobuf.empty_pb2 import Empty

from workspace_qdrant_mcp.core.daemon_client import (
    DaemonClient,
    DaemonConnectionError
)
from workspace_qdrant_mcp.core.grpc_client import GrpcWorkspaceClient
from workspace_qdrant_mcp.core.daemon_manager import (
    DaemonManager,
    DaemonInstance,
    DaemonConfig
)

from .conftest_daemon import (
    mock_daemon_config,
    mock_daemon_instance,
    mock_daemon_manager,
    isolated_daemon_temp_dir,
    DaemonTestHelper
)


@pytest.fixture
def mock_grpc_stub():
    """Mock gRPC stub for testing."""
    stub = Mock()
    stub.Health = AsyncMock(return_value=Mock(status="SERVING"))
    stub.GetSystemStatus = AsyncMock(return_value=Mock(
        status="healthy",
        version="1.0.0",
        uptime_seconds=3600
    ))
    stub.ProcessDocument = AsyncMock(return_value=Mock(
        success=True,
        document_id="doc_123",
        collection_name="test_collection"
    ))
    stub.CreateCollection = AsyncMock(return_value=Mock(
        success=True,
        collection_name="test_collection"
    ))
    stub.ListCollections = AsyncMock(return_value=Mock(
        collections=["collection1", "collection2"]
    ))
    return stub


@pytest.fixture
def mock_grpc_channel():
    """Mock gRPC channel for testing."""
    channel = Mock()
    channel.get_state = Mock(return_value=grpc.ChannelConnectivity.READY)
    channel.subscribe_to_connectivity_state_changes = Mock()
    channel.unsubscribe_from_connectivity_state_changes = Mock()
    channel.close = Mock()
    return channel


@pytest.fixture
def mock_service_endpoint():
    """Mock service endpoint for testing."""
    return {
        "host": "127.0.0.1",
        "port": 50051,
        "project_name": "test-project",
        "daemon_id": "daemon_123",
        "last_seen": datetime.now(timezone.utc),
        "health_status": "healthy"
    }


@pytest.fixture
def mock_service_registry():
    """Mock service registry for testing."""
    registry = Mock()
    registry.register_service = AsyncMock(return_value=True)
    registry.unregister_service = AsyncMock(return_value=True)
    registry.discover_services = AsyncMock(return_value=[
        {"host": "127.0.0.1", "port": 50051, "project": "project1"},
        {"host": "127.0.0.1", "port": 50052, "project": "project2"}
    ])
    registry.get_service_health = AsyncMock(return_value={"healthy": True})
    registry.update_service_heartbeat = AsyncMock(return_value=True)
    return registry


class TestDaemonGrpcCommunication:
    """Test gRPC communication between Python and Rust daemon components."""
    
    @pytest.mark.asyncio
    async def test_grpc_client_server_connection(self, mock_grpc_channel, mock_grpc_stub):
        """Test basic gRPC client-server connection establishment."""
        with patch('grpc.aio.insecure_channel') as mock_channel_factory:
            mock_channel_factory.return_value = mock_grpc_channel
            
            with patch('src.workspace_qdrant_mcp.grpc.ingestion_pb2_grpc.IngestServiceStub') as mock_stub_class:
                mock_stub_class.return_value = mock_grpc_stub
                
                client = DaemonClient()
                
                # Test connection establishment
                await client.connect("127.0.0.1", 50051)
                
                assert client.connected is True
                mock_channel_factory.assert_called_once_with("127.0.0.1:50051")
                mock_stub_class.assert_called_once_with(mock_grpc_channel)
    
    @pytest.mark.asyncio
    async def test_grpc_request_response_handling(self, mock_grpc_stub):
        """Test gRPC request/response message handling."""
        client = DaemonClient()
        client._stub = mock_grpc_stub
        client._connected = True
        
        # Test health check request
        health_response = await client.health_check()
        
        assert health_response.status == "SERVING"
        mock_grpc_stub.Health.assert_called_once()
        
        # Test document processing request
        process_response = await client.process_document(
            file_path="/test/file.py",
            collection_name="test_collection"
        )
        
        assert process_response.success is True
        assert process_response.document_id == "doc_123"
        mock_grpc_stub.ProcessDocument.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_grpc_streaming_operations(self, mock_grpc_stub):
        """Test gRPC streaming operations for real-time updates."""
        client = DaemonClient()
        client._stub = mock_grpc_stub
        client._connected = True
        
        # Mock streaming response
        async def mock_streaming_response():
            for i in range(3):
                yield Mock(
                    progress=i * 33.3,
                    status=f"Processing file {i+1}",
                    current_file=f"/test/file{i+1}.py"
                )
        
        mock_grpc_stub.ProcessFolderStream = AsyncMock(return_value=mock_streaming_response())
        
        # Test streaming folder processing
        progress_updates = []
        async for update in mock_grpc_stub.ProcessFolderStream():
            progress_updates.append(update)
        
        assert len(progress_updates) == 3
        assert progress_updates[0].progress == 0.0
        assert progress_updates[2].progress == 66.6
    
    @pytest.mark.asyncio
    async def test_grpc_connection_retry_logic(self, mock_grpc_channel, mock_grpc_stub):
        """Test gRPC connection retry logic and resilience."""
        connection_attempts = 0
        
        def mock_connect_with_retries(*args, **kwargs):
            nonlocal connection_attempts
            connection_attempts += 1
            if connection_attempts < 3:
                raise grpc.RpcError("Connection failed")
            return mock_grpc_channel
        
        with patch('grpc.aio.insecure_channel', side_effect=mock_connect_with_retries):
            client = DaemonClient(max_retry_attempts=3, retry_delay=0.1)
            
            # Test connection with retries
            await client.connect_with_retry("127.0.0.1", 50051)
            
            assert connection_attempts == 3
            assert client.connected is True
    
    @pytest.mark.asyncio
    async def test_grpc_connection_pool_management(self, mock_grpc_channel, mock_grpc_stub):
        """Test gRPC connection pooling for multiple endpoints."""
        pool = {}
        
        async def mock_get_connection(endpoint):
            if endpoint not in pool:
                pool[endpoint] = {
                    "channel": mock_grpc_channel,
                    "stub": mock_grpc_stub,
                    "created": time.time(),
                    "last_used": time.time()
                }
            return pool[endpoint]
        
        client = DaemonClient()
        client.get_connection = mock_get_connection
        
        # Test connection pooling
        conn1 = await client.get_connection("127.0.0.1:50051")
        conn2 = await client.get_connection("127.0.0.1:50052")
        conn3 = await client.get_connection("127.0.0.1:50051")  # Reuse existing
        
        assert len(pool) == 2  # Only two unique endpoints
        assert conn1 is conn3  # Same connection reused
        assert conn1 is not conn2  # Different connections for different endpoints
    
    @pytest.mark.asyncio
    async def test_grpc_message_serialization(self, mock_grpc_stub):
        """Test gRPC message serialization and deserialization."""
        client = DaemonClient()
        client._stub = mock_grpc_stub
        client._connected = True
        
        # Test complex message serialization
        document_metadata = {
            "file_path": "/test/complex_file.py",
            "content_type": "text/python",
            "size": 1024,
            "modified_time": datetime.now(timezone.utc).isoformat(),
            "tags": ["python", "test", "unit"],
            "custom_fields": {
                "author": "test_user",
                "project": "test_project",
                "complexity": 0.75
            }
        }
        
        # Mock serialization/deserialization
        with patch.object(client, '_serialize_document_metadata') as mock_serialize:
            with patch.object(client, '_deserialize_response') as mock_deserialize:
                mock_serialize.return_value = json.dumps(document_metadata)
                mock_deserialize.return_value = {"success": True, "id": "doc_456"}
                
                result = await client.process_document_with_metadata(document_metadata)
                
                assert result["success"] is True
                mock_serialize.assert_called_once_with(document_metadata)
                mock_deserialize.assert_called_once()


class TestServiceDiscovery:
    """Test service discovery and registration mechanisms."""
    
    @pytest.mark.asyncio
    async def test_daemon_service_registration(self, mock_service_registry, mock_service_endpoint):
        """Test daemon service registration with service registry."""
        daemon_info = {
            "daemon_id": "daemon_123",
            "project_name": "test-project",
            "host": "127.0.0.1",
            "port": 50051,
            "capabilities": ["document_processing", "search", "collection_management"],
            "health_endpoint": "/health",
            "metrics_endpoint": "/metrics"
        }
        
        # Test service registration
        result = await mock_service_registry.register_service(daemon_info)
        
        assert result is True
        mock_service_registry.register_service.assert_called_once_with(daemon_info)
    
    @pytest.mark.asyncio
    async def test_service_discovery_protocol(self, mock_service_registry):
        """Test network discovery protocols for daemon endpoints."""
        # Mock network discovery
        with patch('src.workspace_qdrant_mcp.core.service_discovery.NetworkDiscovery') as mock_discovery:
            discovery_instance = Mock()
            discovery_instance.scan_network = AsyncMock(return_value=[
                {"ip": "127.0.0.1", "port": 50051, "service_type": "qdrant-daemon"},
                {"ip": "127.0.0.1", "port": 50052, "service_type": "qdrant-daemon"},
                {"ip": "127.0.0.1", "port": 50053, "service_type": "other-service"}
            ])
            mock_discovery.return_value = discovery_instance
            
            # Test network scanning
            discovered_services = await discovery_instance.scan_network("127.0.0.0/24")
            
            assert len(discovered_services) == 3
            daemon_services = [s for s in discovered_services if s["service_type"] == "qdrant-daemon"]
            assert len(daemon_services) == 2
    
    @pytest.mark.asyncio
    async def test_service_health_checking(self, mock_service_registry, mock_grpc_stub):
        """Test health checking mechanisms across service endpoints."""
        endpoints = [
            {"host": "127.0.0.1", "port": 50051, "daemon_id": "daemon_1"},
            {"host": "127.0.0.1", "port": 50052, "daemon_id": "daemon_2"},
            {"host": "127.0.0.1", "port": 50053, "daemon_id": "daemon_3"}
        ]
        
        # Mock health check responses
        health_responses = {
            "daemon_1": {"healthy": True, "response_time": 0.05},
            "daemon_2": {"healthy": False, "error": "connection_timeout"},
            "daemon_3": {"healthy": True, "response_time": 0.12}
        }
        
        async def mock_health_check(endpoint):
            daemon_id = endpoint["daemon_id"]
            return health_responses.get(daemon_id, {"healthy": False, "error": "unknown"})
        
        # Test health checking across all endpoints
        health_results = {}
        for endpoint in endpoints:
            health_results[endpoint["daemon_id"]] = await mock_health_check(endpoint)
        
        assert health_results["daemon_1"]["healthy"] is True
        assert health_results["daemon_2"]["healthy"] is False
        assert health_results["daemon_3"]["healthy"] is True
        assert len([r for r in health_results.values() if r["healthy"]]) == 2
    
    @pytest.mark.asyncio
    async def test_service_heartbeat_mechanism(self, mock_service_registry):
        """Test heartbeat mechanisms for service availability tracking."""
        daemon_id = "daemon_123"
        heartbeat_interval = 0.1  # 100ms for testing
        
        # Mock heartbeat tracking
        heartbeat_count = 0
        last_heartbeat = None
        
        async def mock_send_heartbeat():
            nonlocal heartbeat_count, last_heartbeat
            heartbeat_count += 1
            last_heartbeat = datetime.now(timezone.utc)
            await mock_service_registry.update_service_heartbeat(daemon_id, last_heartbeat)
            return True
        
        # Test heartbeat sending
        for _ in range(3):
            await mock_send_heartbeat()
            await asyncio.sleep(heartbeat_interval)
        
        assert heartbeat_count == 3
        assert last_heartbeat is not None
        assert mock_service_registry.update_service_heartbeat.call_count == 3
    
    @pytest.mark.asyncio
    async def test_multi_daemon_coordination(self, mock_service_registry):
        """Test coordination between multiple daemon instances."""
        daemon_configs = [
            {"id": "daemon_1", "project": "project_a", "port": 50051},
            {"id": "daemon_2", "project": "project_b", "port": 50052},
            {"id": "daemon_3", "project": "project_c", "port": 50053}
        ]
        
        # Mock coordination protocol
        coordination_messages = []
        
        async def mock_broadcast_message(sender_id, message_type, payload):
            message = {
                "sender": sender_id,
                "type": message_type,
                "payload": payload,
                "timestamp": datetime.now(timezone.utc)
            }
            coordination_messages.append(message)
            return len(daemon_configs) - 1  # Number of recipients
        
        # Test coordination scenarios
        # 1. Configuration update broadcast
        recipients = await mock_broadcast_message(
            "daemon_1",
            "config_update",
            {"updated_fields": ["log_level"], "new_values": {"log_level": "debug"}}
        )
        assert recipients == 2
        
        # 2. Resource sharing notification
        recipients = await mock_broadcast_message(
            "daemon_2",
            "resource_sharing",
            {"available_collections": ["shared_docs"], "access_mode": "read_only"}
        )
        assert recipients == 2
        
        assert len(coordination_messages) == 2
        assert coordination_messages[0]["type"] == "config_update"
        assert coordination_messages[1]["type"] == "resource_sharing"


class TestIPCErrorHandling:
    """Test error handling in IPC communication scenarios."""
    
    @pytest.mark.asyncio
    async def test_network_partition_handling(self, mock_grpc_channel, mock_grpc_stub):
        """Test handling of network partitions and connectivity issues."""
        client = DaemonClient()
        client._channel = mock_grpc_channel
        client._stub = mock_grpc_stub
        client._connected = True
        
        # Simulate network partition
        mock_grpc_stub.Health.side_effect = grpc.RpcError("Network unreachable")
        
        # Test graceful handling of network partition
        with pytest.raises(DaemonConnectionError):
            await client.health_check()
        
        # Test automatic reconnection attempt
        with patch.object(client, 'reconnect', new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = True
            mock_grpc_stub.Health.side_effect = None  # Clear the error
            mock_grpc_stub.Health.return_value = Mock(status="SERVING")
            
            await client.health_check_with_reconnect()
            
            mock_reconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_service_timeout_handling(self, mock_grpc_stub):
        """Test timeout handling for slow or unresponsive services."""
        client = DaemonClient(request_timeout=0.1)  # 100ms timeout
        client._stub = mock_grpc_stub
        client._connected = True
        
        # Simulate slow service response
        async def slow_response():
            await asyncio.sleep(0.2)  # Slower than timeout
            return Mock(status="SERVING")
        
        mock_grpc_stub.Health.side_effect = slow_response
        
        # Test timeout handling
        with pytest.raises(asyncio.TimeoutError):
            await client.health_check()
    
    @pytest.mark.asyncio
    async def test_service_registration_failures(self, mock_service_registry):
        """Test handling of service registration and deregistration failures."""
        daemon_info = {
            "daemon_id": "daemon_456",
            "project_name": "test-project",
            "host": "127.0.0.1",
            "port": 50051
        }
        
        # Simulate registration failure
        mock_service_registry.register_service.side_effect = Exception("Registry unavailable")
        
        # Test graceful handling of registration failure
        with pytest.raises(Exception, match="Registry unavailable"):
            await mock_service_registry.register_service(daemon_info)
        
        # Test retry mechanism
        retry_count = 0
        async def mock_register_with_retry(info):
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise Exception("Temporary failure")
            return True
        
        mock_service_registry.register_service.side_effect = mock_register_with_retry
        
        # Should succeed after retries
        result = await mock_service_registry.register_service(daemon_info)
        assert result is True
        assert retry_count == 3
    
    @pytest.mark.asyncio
    async def test_connection_pool_cleanup(self, mock_grpc_channel):
        """Test connection pool cleanup during errors and shutdowns."""
        client = DaemonClient()
        client._connection_pool = {
            "endpoint1": {"channel": mock_grpc_channel, "last_used": time.time()},
            "endpoint2": {"channel": mock_grpc_channel, "last_used": time.time() - 3600}  # Old connection
        }
        
        # Mock connection cleanup
        with patch.object(client, 'cleanup_stale_connections', new_callable=AsyncMock) as mock_cleanup:
            async def cleanup_connections(max_age=1800):
                current_time = time.time()
                stale_endpoints = [
                    endpoint for endpoint, conn in client._connection_pool.items()
                    if current_time - conn["last_used"] > max_age
                ]
                for endpoint in stale_endpoints:
                    conn = client._connection_pool.pop(endpoint)
                    conn["channel"].close()
                return len(stale_endpoints)
            
            mock_cleanup.side_effect = cleanup_connections
            
            # Test cleanup of stale connections
            cleaned_count = await client.cleanup_stale_connections(max_age=1800)
            
            assert cleaned_count == 1
            assert "endpoint2" not in client._connection_pool
            assert "endpoint1" in client._connection_pool


class TestIPCPerformanceAndScaling:
    """Test performance and scaling aspects of IPC communication."""
    
    @pytest.mark.asyncio
    async def test_concurrent_grpc_requests(self, mock_grpc_stub):
        """Test handling of concurrent gRPC requests."""
        client = DaemonClient()
        client._stub = mock_grpc_stub
        client._connected = True
        
        # Mock concurrent request handling
        request_count = 0
        async def mock_process_request(*args, **kwargs):
            nonlocal request_count
            request_count += 1
            await asyncio.sleep(0.01)  # Simulate processing time
            return Mock(success=True, request_id=request_count)
        
        mock_grpc_stub.ProcessDocument.side_effect = mock_process_request
        
        # Test concurrent requests
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                client.process_document(f"/test/file{i}.py", "test_collection")
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 10
        assert all(result.success for result in results)
        assert request_count == 10
    
    @pytest.mark.asyncio
    async def test_load_balancing_across_daemons(self, mock_service_registry):
        """Test load balancing requests across multiple daemon instances."""
        available_daemons = [
            {"id": "daemon_1", "host": "127.0.0.1", "port": 50051, "load": 0.2},
            {"id": "daemon_2", "host": "127.0.0.1", "port": 50052, "load": 0.8},
            {"id": "daemon_3", "host": "127.0.0.1", "port": 50053, "load": 0.1}
        ]
        
        # Mock load balancer
        request_distribution = {"daemon_1": 0, "daemon_2": 0, "daemon_3": 0}
        
        def select_daemon_by_load():
            # Select daemon with lowest load
            return min(available_daemons, key=lambda d: d["load"])
        
        # Test load-based distribution
        for _ in range(100):
            selected_daemon = select_daemon_by_load()
            request_distribution[selected_daemon["id"]] += 1
            # Simulate load increase
            selected_daemon["load"] += 0.01
        
        # Verify requests were distributed (daemon_3 should get most requests initially)
        assert request_distribution["daemon_3"] > request_distribution["daemon_2"]
        assert request_distribution["daemon_1"] > request_distribution["daemon_2"]
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_grpc_communication_performance(self, mock_grpc_stub, benchmark):
        """Benchmark gRPC communication performance."""
        client = DaemonClient()
        client._stub = mock_grpc_stub
        client._connected = True
        
        async def perform_grpc_operation():
            return await client.health_check()
        
        # Benchmark the operation
        def sync_wrapper():
            return asyncio.run(perform_grpc_operation())
        
        result = benchmark(sync_wrapper)
        assert result.status == "SERVING"


@pytest.mark.daemon_unit
@pytest.mark.daemon_ipc
@pytest.mark.daemon_service_discovery
class TestDaemonIPCIntegration:
    """Integration tests for daemon IPC and service discovery."""
    
    @pytest.mark.asyncio
    async def test_full_service_lifecycle(self, mock_service_registry, mock_grpc_stub):
        """Test complete service lifecycle from registration to deregistration."""
        daemon_info = {
            "daemon_id": "integration_daemon",
            "project_name": "integration-test",
            "host": "127.0.0.1",
            "port": 50051
        }
        
        # Test service registration
        registration_result = await mock_service_registry.register_service(daemon_info)
        assert registration_result is True
        
        # Test service discovery
        discovered_services = await mock_service_registry.discover_services()
        assert len(discovered_services) >= 1
        
        # Test service health monitoring
        health_status = await mock_service_registry.get_service_health(daemon_info["daemon_id"])
        assert health_status["healthy"] is True
        
        # Test service deregistration
        deregistration_result = await mock_service_registry.unregister_service(daemon_info["daemon_id"])
        assert deregistration_result is True
    
    @pytest.mark.asyncio
    async def test_multi_component_coordination(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test coordination between multiple system components."""
        projects = [
            ("project-ipc-1", str(isolated_daemon_temp_dir / "proj1")),
            ("project-ipc-2", str(isolated_daemon_temp_dir / "proj2"))
        ]
        
        # Create project directories
        for _, path in projects:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Mock multi-component coordination
        coordination_state = {"active_daemons": [], "shared_resources": []}
        
        with patch.object(mock_daemon_manager, 'coordinate_daemon_operations', new_callable=AsyncMock) as mock_coordinate:
            async def coordinate_operations(operation, params):
                if operation == "startup_coordination":
                    coordination_state["active_daemons"].append(params["daemon_id"])
                    return {"coordinated": True}
                elif operation == "resource_sharing":
                    coordination_state["shared_resources"].extend(params["resources"])
                    return {"shared": True}
                return {"operation": operation, "status": "unknown"}
            
            mock_coordinate.side_effect = coordinate_operations
            
            # Test coordination scenarios
            startup_result = await mock_daemon_manager.coordinate_daemon_operations(
                "startup_coordination", {"daemon_id": "daemon_test_1"}
            )
            resource_result = await mock_daemon_manager.coordinate_daemon_operations(
                "resource_sharing", {"resources": ["collection_a", "collection_b"]}
            )
            
            assert startup_result["coordinated"] is True
            assert resource_result["shared"] is True
            assert len(coordination_state["active_daemons"]) == 1
            assert len(coordination_state["shared_resources"]) == 2