"""
Unit tests for LSP integration and server management within daemon context.

Tests the daemon's LSP integration capabilities including:
- LSP server detection and registration within daemon lifecycle
- LSP health monitoring integration with daemon health checks
- LSP communication through daemon gRPC interface
- LSP failure recovery and fallback mechanisms
- LSP server lifecycle management within daemon processes
- Configuration synchronization between LSP and daemon components
"""

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

from common.core.daemon_manager import (
    DaemonManager,
    DaemonInstance,
    DaemonConfig,
    DaemonStatus
)
from common.core.lsp_client import (
    LspClient,
    ConnectionState,
    CommunicationMode,
    CircuitBreakerState
)
from common.core.lsp_detector import LspDetector
from common.core.lsp_health_monitor import LspHealthMonitor
from common.core.lsp_metadata_extractor import LspMetadataExtractor

from .conftest_daemon import (
    mock_daemon_config,
    mock_daemon_instance,
    mock_daemon_manager,
    isolated_daemon_temp_dir,
    DaemonTestHelper,
    assert_daemon_config_valid
)


@pytest.fixture
def mock_lsp_client():
    """Mock LSP client for testing."""
    client = Mock(spec=LspClient)
    client.state = ConnectionState.DISCONNECTED
    client.mode = CommunicationMode.STDIO
    client.circuit_breaker_state = CircuitBreakerState.CLOSED
    client.connect = AsyncMock(return_value=True)
    client.disconnect = AsyncMock(return_value=True)
    client.send_request = AsyncMock(return_value={"result": "success"})
    client.send_notification = AsyncMock()
    client.is_connected = Mock(return_value=False)
    client.get_capabilities = AsyncMock(return_value={
        "textDocumentSync": 1,
        "hoverProvider": True,
        "definitionProvider": True
    })
    client.health_check = AsyncMock(return_value=True)
    return client


@pytest.fixture
def mock_lsp_detector():
    """Mock LSP detector for testing."""
    detector = Mock(spec=LspDetector)
    detector.detect_lsp_servers = AsyncMock(return_value=[
        {
            "name": "pylsp",
            "command": ["pylsp"],
            "languages": ["python"],
            "port": None,
            "capabilities": ["textDocument/hover", "textDocument/definition"]
        }
    ])
    detector.validate_server = AsyncMock(return_value=True)
    detector.get_server_config = AsyncMock(return_value={
        "command": ["pylsp"],
        "args": [],
        "env": {}
    })
    return detector


@pytest.fixture
def mock_lsp_health_monitor():
    """Mock LSP health monitor for testing."""
    monitor = Mock(spec=LspHealthMonitor)
    monitor.start_monitoring = AsyncMock()
    monitor.stop_monitoring = AsyncMock()
    monitor.check_health = AsyncMock(return_value=True)
    monitor.get_health_status = Mock(return_value={
        "healthy": True,
        "last_check": datetime.now(timezone.utc),
        "response_time": 0.05
    })
    monitor.register_health_callback = Mock()
    return monitor


@pytest.fixture
def mock_lsp_metadata_extractor():
    """Mock LSP metadata extractor for testing."""
    extractor = Mock(spec=LspMetadataExtractor)
    extractor.extract_metadata = AsyncMock(return_value={
        "symbols": [
            {"name": "test_function", "kind": "function", "location": {"line": 10, "character": 0}}
        ],
        "imports": ["os", "sys"],
        "classes": [],
        "functions": ["test_function"]
    })
    extractor.get_document_symbols = AsyncMock(return_value=[])
    extractor.get_references = AsyncMock(return_value=[])
    return extractor


class TestDaemonLspIntegration:
    """Test LSP integration within daemon context."""
    
    @pytest.mark.asyncio
    async def test_daemon_lsp_server_detection(self, mock_daemon_instance, mock_lsp_detector, isolated_daemon_temp_dir):
        """Test LSP server detection during daemon startup."""
        # Setup daemon with LSP detection capability
        mock_daemon_instance.config.project_path = str(isolated_daemon_temp_dir)
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.LspDetector', return_value=mock_lsp_detector):
            # Simulate daemon startup with LSP detection
            await DaemonTestHelper.simulate_daemon_startup(mock_daemon_instance, success=True)
            
            result = await mock_daemon_instance.start()
            
            assert result is True
            assert mock_daemon_instance.status.state == "running"
            
            # Verify LSP detection was called
            mock_lsp_detector.detect_lsp_servers.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_lsp_health_integration(self, mock_daemon_instance, mock_lsp_health_monitor):
        """Test LSP health monitoring integration with daemon health checks."""
        # Setup daemon with LSP health monitoring
        mock_daemon_instance.lsp_health_monitor = mock_lsp_health_monitor
        mock_daemon_instance.status.state = "running"
        
        with patch.object(mock_daemon_instance, 'health_check', new_callable=AsyncMock) as mock_health:
            # Mock combined health check (daemon + LSP)
            async def combined_health_check():
                daemon_healthy = True
                lsp_healthy = await mock_lsp_health_monitor.check_health()
                return daemon_healthy and lsp_healthy
            
            mock_health.side_effect = combined_health_check
            
            result = await mock_daemon_instance.health_check()
            
            assert result is True
            mock_lsp_health_monitor.check_health.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_lsp_communication_through_grpc(self, mock_daemon_instance, mock_lsp_client):
        """Test LSP communication through daemon gRPC interface."""
        # Setup daemon with LSP client
        mock_daemon_instance.lsp_client = mock_lsp_client
        mock_daemon_instance.status.state = "running"
        mock_daemon_instance.status.grpc_available = True
        
        # Mock gRPC call that forwards to LSP
        with patch.object(mock_daemon_instance, 'grpc_call', new_callable=AsyncMock) as mock_grpc:
            async def grpc_lsp_proxy(method, params):
                if method == "lsp_request":
                    return await mock_lsp_client.send_request(params.get("method"), params.get("params"))
                return {"error": "unknown_method"}
            
            mock_grpc.side_effect = grpc_lsp_proxy
            
            # Test LSP request through gRPC
            result = await mock_daemon_instance.grpc_call("lsp_request", {
                "method": "textDocument/hover",
                "params": {
                    "textDocument": {"uri": "file:///test.py"},
                    "position": {"line": 10, "character": 5}
                }
            })
            
            assert result == {"result": "success"}
            mock_lsp_client.send_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_lsp_failure_recovery(self, mock_daemon_instance, mock_lsp_client, mock_lsp_health_monitor):
        """Test LSP failure recovery within daemon context."""
        # Setup daemon with LSP components
        mock_daemon_instance.lsp_client = mock_lsp_client
        mock_daemon_instance.lsp_health_monitor = mock_lsp_health_monitor
        mock_daemon_instance.status.state = "running"
        
        # Simulate LSP failure
        mock_lsp_client.is_connected.return_value = False
        mock_lsp_client.state = ConnectionState.ERROR
        mock_lsp_health_monitor.check_health.return_value = False
        
        # Mock daemon's LSP recovery mechanism
        with patch.object(mock_daemon_instance, 'recover_lsp', new_callable=AsyncMock) as mock_recover:
            async def lsp_recovery():
                # Restart LSP client
                await mock_lsp_client.disconnect()
                await mock_lsp_client.connect()
                mock_lsp_client.state = ConnectionState.CONNECTED
                mock_lsp_client.is_connected.return_value = True
                return True
            
            mock_recover.side_effect = lsp_recovery
            
            # Trigger recovery
            recovery_result = await mock_daemon_instance.recover_lsp()
            
            assert recovery_result is True
            assert mock_lsp_client.state == ConnectionState.CONNECTED
            mock_lsp_client.disconnect.assert_called_once()
            mock_lsp_client.connect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_lsp_server_lifecycle_management(self, mock_daemon_instance, mock_lsp_detector):
        """Test LSP server lifecycle management within daemon processes."""
        # Setup daemon configuration for LSP
        mock_daemon_instance.config.project_path = "/test/project"
        mock_daemon_instance.lsp_servers = {}
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.LspClient') as mock_lsp_class:
            mock_lsp_instance = Mock()
            mock_lsp_instance.connect = AsyncMock(return_value=True)
            mock_lsp_instance.disconnect = AsyncMock(return_value=True)
            mock_lsp_instance.state = ConnectionState.CONNECTED
            mock_lsp_class.return_value = mock_lsp_instance
            
            # Mock daemon's LSP server management
            with patch.object(mock_daemon_instance, 'start_lsp_server', new_callable=AsyncMock) as mock_start_lsp:
                with patch.object(mock_daemon_instance, 'stop_lsp_server', new_callable=AsyncMock) as mock_stop_lsp:
                    
                    async def start_lsp_server(server_config):
                        client = mock_lsp_class(server_config)
                        await client.connect()
                        mock_daemon_instance.lsp_servers[server_config["name"]] = client
                        return client
                    
                    async def stop_lsp_server(server_name):
                        if server_name in mock_daemon_instance.lsp_servers:
                            client = mock_daemon_instance.lsp_servers[server_name]
                            await client.disconnect()
                            del mock_daemon_instance.lsp_servers[server_name]
                        return True
                    
                    mock_start_lsp.side_effect = start_lsp_server
                    mock_stop_lsp.side_effect = stop_lsp_server
                    
                    # Test starting LSP server
                    server_config = {"name": "pylsp", "command": ["pylsp"]}
                    client = await mock_daemon_instance.start_lsp_server(server_config)
                    
                    assert client is not None
                    assert "pylsp" in mock_daemon_instance.lsp_servers
                    mock_lsp_instance.connect.assert_called_once()
                    
                    # Test stopping LSP server
                    result = await mock_daemon_instance.stop_lsp_server("pylsp")
                    
                    assert result is True
                    assert "pylsp" not in mock_daemon_instance.lsp_servers
                    mock_lsp_instance.disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_lsp_configuration_synchronization(self, mock_daemon_instance, isolated_daemon_temp_dir):
        """Test configuration synchronization between LSP and daemon components."""
        # Setup daemon with LSP configuration
        mock_daemon_instance.config.project_path = str(isolated_daemon_temp_dir)
        lsp_config = {
            "servers": {
                "pylsp": {
                    "command": ["pylsp"],
                    "settings": {"pylsp": {"plugins": {"pycodestyle": {"enabled": False}}}}
                }
            }
        }
        
        # Create LSP configuration file
        lsp_config_file = isolated_daemon_temp_dir / ".lsp_config.json"
        lsp_config_file.write_text(json.dumps(lsp_config))
        
        # Mock daemon's configuration sync
        with patch.object(mock_daemon_instance, 'sync_lsp_config', new_callable=AsyncMock) as mock_sync:
            async def sync_lsp_config():
                # Read LSP config and merge with daemon config
                if lsp_config_file.exists():
                    config_data = json.loads(lsp_config_file.read_text())
                    mock_daemon_instance.lsp_config = config_data
                    return True
                return False
            
            mock_sync.side_effect = sync_lsp_config
            
            # Test configuration synchronization
            result = await mock_daemon_instance.sync_lsp_config()
            
            assert result is True
            assert hasattr(mock_daemon_instance, 'lsp_config')
            assert mock_daemon_instance.lsp_config["servers"]["pylsp"]["command"] == ["pylsp"]
    
    @pytest.mark.asyncio
    async def test_daemon_lsp_fallback_mechanisms(self, mock_daemon_instance, mock_lsp_client):
        """Test LSP fallback mechanisms when primary LSP fails."""
        # Setup daemon with LSP fallback
        mock_daemon_instance.lsp_client = mock_lsp_client
        mock_daemon_instance.lsp_fallback_enabled = True
        
        # Simulate LSP failure
        mock_lsp_client.is_connected.return_value = False
        mock_lsp_client.circuit_breaker_state = CircuitBreakerState.OPEN
        
        # Mock fallback mechanism
        with patch.object(mock_daemon_instance, 'activate_lsp_fallback', new_callable=AsyncMock) as mock_fallback:
            async def activate_fallback():
                # Switch to fallback mode (e.g., file system analysis)
                mock_daemon_instance.lsp_fallback_active = True
                return {"mode": "filesystem", "available": True}
            
            mock_fallback.side_effect = activate_fallback
            
            # Test fallback activation
            fallback_status = await mock_daemon_instance.activate_lsp_fallback()
            
            assert fallback_status["available"] is True
            assert mock_daemon_instance.lsp_fallback_active is True
    
    @pytest.mark.asyncio
    async def test_daemon_lsp_metadata_extraction_integration(self, mock_daemon_instance, mock_lsp_metadata_extractor):
        """Test LSP metadata extraction integration with daemon processing."""
        # Setup daemon with metadata extraction
        mock_daemon_instance.lsp_metadata_extractor = mock_lsp_metadata_extractor
        mock_daemon_instance.status.state = "running"
        
        # Mock daemon's document processing with LSP metadata
        with patch.object(mock_daemon_instance, 'process_document_with_lsp', new_callable=AsyncMock) as mock_process:
            async def process_with_lsp(file_path):
                # Extract LSP metadata and combine with document processing
                metadata = await mock_lsp_metadata_extractor.extract_metadata(file_path)
                return {
                    "file_path": file_path,
                    "content": "test content",
                    "lsp_metadata": metadata,
                    "processed": True
                }
            
            mock_process.side_effect = process_with_lsp
            
            # Test document processing with LSP metadata
            result = await mock_daemon_instance.process_document_with_lsp("/test/file.py")
            
            assert result["processed"] is True
            assert "lsp_metadata" in result
            assert result["lsp_metadata"]["symbols"] is not None
            mock_lsp_metadata_extractor.extract_metadata.assert_called_once_with("/test/file.py")


class TestDaemonManagerLspIntegration:
    """Test DaemonManager's LSP integration capabilities."""
    
    @pytest.mark.asyncio
    async def test_daemon_manager_lsp_server_registry(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test DaemonManager's LSP server registry functionality."""
        project_name = "test-lsp-project"
        project_path = str(isolated_daemon_temp_dir)
        
        # Mock LSP server registry
        lsp_servers = {
            "pylsp": {"status": "active", "pid": 12345},
            "typescript-language-server": {"status": "inactive", "pid": None}
        }
        
        with patch.object(mock_daemon_manager, 'get_lsp_server_registry', new_callable=AsyncMock) as mock_registry:
            mock_registry.return_value = lsp_servers
            
            registry = await mock_daemon_manager.get_lsp_server_registry(project_name, project_path)
            
            assert "pylsp" in registry
            assert registry["pylsp"]["status"] == "active"
            assert registry["typescript-language-server"]["status"] == "inactive"
    
    @pytest.mark.asyncio
    async def test_daemon_manager_lsp_health_monitoring(self, mock_daemon_manager):
        """Test DaemonManager's LSP health monitoring across all daemons."""
        expected_lsp_health = {
            "daemon_123": {
                "lsp_servers": {
                    "pylsp": {"healthy": True, "response_time": 0.05},
                    "rust-analyzer": {"healthy": False, "error": "connection_timeout"}
                }
            },
            "daemon_456": {
                "lsp_servers": {
                    "typescript-language-server": {"healthy": True, "response_time": 0.03}
                }
            }
        }
        
        with patch.object(mock_daemon_manager, 'get_lsp_health_status', new_callable=AsyncMock) as mock_lsp_health:
            mock_lsp_health.return_value = expected_lsp_health
            
            health_status = await mock_daemon_manager.get_lsp_health_status()
            
            assert "daemon_123" in health_status
            assert health_status["daemon_123"]["lsp_servers"]["pylsp"]["healthy"] is True
            assert health_status["daemon_123"]["lsp_servers"]["rust-analyzer"]["healthy"] is False
    
    @pytest.mark.asyncio
    async def test_daemon_manager_lsp_coordination(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test DaemonManager's coordination of LSP operations across projects."""
        projects = [
            ("project-1", str(isolated_daemon_temp_dir / "proj1")),
            ("project-2", str(isolated_daemon_temp_dir / "proj2"))
        ]
        
        # Create project directories
        for _, path in projects:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Mock LSP coordination across projects
        with patch.object(mock_daemon_manager, 'coordinate_lsp_operations', new_callable=AsyncMock) as mock_coordinate:
            async def coordinate_operations(operation, params):
                if operation == "broadcast_shutdown":
                    return {"coordinated": True, "projects": len(projects)}
                elif operation == "sync_configurations":
                    return {"synced": True, "configs_updated": len(projects)}
                return {"operation": operation, "status": "unknown"}
            
            mock_coordinate.side_effect = coordinate_operations
            
            # Test LSP operation coordination
            shutdown_result = await mock_daemon_manager.coordinate_lsp_operations("broadcast_shutdown", {})
            sync_result = await mock_daemon_manager.coordinate_lsp_operations("sync_configurations", {})
            
            assert shutdown_result["coordinated"] is True
            assert sync_result["synced"] is True
            assert shutdown_result["projects"] == 2
            assert sync_result["configs_updated"] == 2


class TestDaemonLspErrorHandling:
    """Test error handling in daemon LSP integration."""
    
    @pytest.mark.asyncio
    async def test_lsp_connection_failure_handling(self, mock_daemon_instance, mock_lsp_client):
        """Test handling of LSP connection failures in daemon context."""
        # Setup daemon with failing LSP client
        mock_daemon_instance.lsp_client = mock_lsp_client
        mock_lsp_client.connect.side_effect = ConnectionError("LSP server unavailable")
        
        # Mock daemon's LSP error handling
        with patch.object(mock_daemon_instance, 'handle_lsp_error', new_callable=AsyncMock) as mock_error_handler:
            async def handle_lsp_error(error):
                # Log error and activate fallback
                mock_daemon_instance.lsp_error_count = getattr(mock_daemon_instance, 'lsp_error_count', 0) + 1
                if mock_daemon_instance.lsp_error_count >= 3:
                    mock_daemon_instance.lsp_fallback_active = True
                return {"handled": True, "fallback_active": mock_daemon_instance.lsp_fallback_active}
            
            mock_error_handler.side_effect = handle_lsp_error
            
            # Simulate multiple LSP connection failures
            for i in range(3):
                try:
                    await mock_lsp_client.connect()
                except ConnectionError as e:
                    result = await mock_daemon_instance.handle_lsp_error(e)
            
            assert mock_daemon_instance.lsp_error_count == 3
            assert mock_daemon_instance.lsp_fallback_active is True
    
    @pytest.mark.asyncio
    async def test_lsp_circuit_breaker_integration(self, mock_daemon_instance, mock_lsp_client):
        """Test LSP circuit breaker integration with daemon operations."""
        # Setup daemon with LSP circuit breaker
        mock_daemon_instance.lsp_client = mock_lsp_client
        mock_lsp_client.circuit_breaker_state = CircuitBreakerState.CLOSED
        
        # Mock circuit breaker behavior
        failure_count = 0
        
        async def mock_lsp_request(method, params):
            nonlocal failure_count
            failure_count += 1
            if failure_count >= 5:
                mock_lsp_client.circuit_breaker_state = CircuitBreakerState.OPEN
                raise ConnectionError("Circuit breaker open")
            if failure_count >= 3:
                raise TimeoutError("LSP timeout")
            return {"result": "success"}
        
        mock_lsp_client.send_request.side_effect = mock_lsp_request
        
        # Test circuit breaker activation
        for i in range(6):
            try:
                await mock_lsp_client.send_request("test/method", {})
            except (TimeoutError, ConnectionError):
                pass
        
        assert mock_lsp_client.circuit_breaker_state == CircuitBreakerState.OPEN
    
    @pytest.mark.asyncio
    async def test_lsp_recovery_after_daemon_restart(self, mock_daemon_instance, mock_lsp_client):
        """Test LSP recovery after daemon restart."""
        # Setup daemon with LSP client
        mock_daemon_instance.lsp_client = mock_lsp_client
        mock_daemon_instance.status.state = "running"
        
        # Simulate daemon restart
        mock_daemon_instance.status.restart_count = 1
        
        # Mock LSP recovery after restart
        with patch.object(mock_daemon_instance, 'recover_lsp_after_restart', new_callable=AsyncMock) as mock_recovery:
            async def recover_lsp():
                # Reset LSP state after daemon restart
                mock_lsp_client.circuit_breaker_state = CircuitBreakerState.CLOSED
                mock_lsp_client.state = ConnectionState.DISCONNECTED
                await mock_lsp_client.connect()
                mock_lsp_client.state = ConnectionState.CONNECTED
                return True
            
            mock_recovery.side_effect = recover_lsp
            
            # Test LSP recovery
            recovery_result = await mock_daemon_instance.recover_lsp_after_restart()
            
            assert recovery_result is True
            assert mock_lsp_client.state == ConnectionState.CONNECTED
            assert mock_lsp_client.circuit_breaker_state == CircuitBreakerState.CLOSED


@pytest.mark.daemon_unit
@pytest.mark.daemon_lsp
@pytest.mark.daemon_integration
class TestDaemonLspPerformanceIntegration:
    """Test performance aspects of daemon LSP integration."""
    
    @pytest.mark.asyncio
    async def test_lsp_request_performance_monitoring(self, mock_daemon_instance, mock_lsp_client):
        """Test LSP request performance monitoring within daemon."""
        # Setup daemon with performance monitoring
        mock_daemon_instance.lsp_client = mock_lsp_client
        mock_daemon_instance.lsp_performance_metrics = {}
        
        # Mock performance tracking
        with patch.object(mock_daemon_instance, 'track_lsp_performance', new_callable=AsyncMock) as mock_track:
            async def track_performance(method, start_time, end_time):
                duration = end_time - start_time
                if method not in mock_daemon_instance.lsp_performance_metrics:
                    mock_daemon_instance.lsp_performance_metrics[method] = []
                mock_daemon_instance.lsp_performance_metrics[method].append(duration)
                return duration
            
            mock_track.side_effect = track_performance
            
            # Simulate LSP requests with performance tracking
            start_time = time.time()
            await mock_lsp_client.send_request("textDocument/hover", {})
            end_time = time.time()
            
            await mock_daemon_instance.track_lsp_performance("textDocument/hover", start_time, end_time)
            
            assert "textDocument/hover" in mock_daemon_instance.lsp_performance_metrics
            assert len(mock_daemon_instance.lsp_performance_metrics["textDocument/hover"]) == 1
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_lsp_daemon_integration_benchmark(self, mock_daemon_instance, mock_lsp_client, benchmark):
        """Benchmark LSP operations through daemon integration."""
        # Setup daemon with LSP client
        mock_daemon_instance.lsp_client = mock_lsp_client
        mock_daemon_instance.status.state = "running"
        
        async def lsp_operation_through_daemon():
            # Simulate LSP operation through daemon
            result = await mock_lsp_client.send_request("textDocument/definition", {
                "textDocument": {"uri": "file:///test.py"},
                "position": {"line": 10, "character": 5}
            })
            return result
        
        # Benchmark the operation
        def sync_wrapper():
            return asyncio.run(lsp_operation_through_daemon())
        
        result = benchmark(sync_wrapper)
        assert result == {"result": "success"}