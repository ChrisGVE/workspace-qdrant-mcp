"""
Integration tests for daemon management with gRPC client.

This test module verifies the end-to-end integration between the daemon manager
and the gRPC-enabled workspace client, ensuring automatic daemon startup,
health monitoring, and graceful shutdown work correctly together.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from common.core.daemon_manager import DaemonManager, ensure_daemon_running
from common.core.grpc_client import GrpcWorkspaceClient
from common.core.config import Config


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Mock(spec=Config)
    config.qdrant = Mock()
    config.qdrant.url = "http://localhost:6333"
    config.grpc = Mock()
    config.grpc.enabled = True
    config.grpc.host = "127.0.0.1"
    config.grpc.port = 50053  # Test port
    config.grpc.fallback_to_direct = True
    return config


@pytest.fixture
def mock_project_detector():
    """Mock project detection for testing."""
    detector = Mock()
    detector.detect_project_structure = AsyncMock(return_value={
        "main_project": "test-integration-project",
        "subprojects": [],
        "workspace_collections": ["test-integration-project"]
    })
    return detector


class TestDaemonGrpcIntegration:
    """Test integration between daemon manager and gRPC client."""
    
    @pytest.mark.asyncio
    async def test_grpc_client_daemon_auto_start(self, mock_config, temp_project_dir):
        """Test that gRPC client automatically starts daemon on initialization."""
        project_name = "auto-start-test"
        
        # Create gRPC client with auto-start enabled
        client = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            grpc_host="127.0.0.1",
            grpc_port=50054,
            auto_start_daemon=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        # Mock daemon startup sequence
        with patch('workspace_qdrant_mcp.core.grpc_client.ensure_daemon_running') as mock_ensure:
            with patch('workspace_qdrant_mcp.core.grpc_client.get_daemon_for_project', return_value=None):
                mock_daemon = Mock()
                mock_daemon.status.state = "running"
                mock_daemon.config.grpc_port = 50054
                mock_ensure.return_value = mock_daemon
                
                # Mock gRPC client connection
                with patch('workspace_qdrant_mcp.grpc.client.AsyncIngestClient') as mock_grpc_client:
                    mock_grpc_instance = AsyncMock()
                    mock_grpc_instance.test_connection.return_value = True
                    mock_grpc_client.return_value = mock_grpc_instance
                    
                    # Mock direct client initialization
                    with patch.object(client.direct_client, 'initialize'):
                        await client.initialize()
                        
                        # Verify daemon was started
                        mock_ensure.assert_called_once_with(
                            project_name,
                            temp_project_dir,
                            {'grpc_port': 50054}
                        )
                        
                        # Verify gRPC mode is active
                        assert client.get_operation_mode() == "grpc"
                        assert client.is_grpc_available() is True
        
        # Cleanup
        from common.core.daemon_manager import shutdown_all_daemons
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_grpc_client_fallback_to_direct(self, mock_config, temp_project_dir):
        """Test gRPC client fallback to direct mode when daemon fails."""
        project_name = "fallback-test"
        
        client = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            auto_start_daemon=True,
            fallback_to_direct=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        # Mock daemon startup failure
        with patch('workspace_qdrant_mcp.core.grpc_client.ensure_daemon_running') as mock_ensure:
            mock_ensure.side_effect = RuntimeError("Failed to start daemon")
            
            # Mock direct client initialization
            with patch.object(client.direct_client, 'initialize'):
                await client.initialize()
                
                # Verify fallback to direct mode
                assert client.get_operation_mode() == "direct"
                assert client.is_grpc_available() is False
    
    @pytest.mark.asyncio
    async def test_daemon_health_monitoring_integration(self, mock_config, temp_project_dir):
        """Test daemon health monitoring integration with gRPC client."""
        project_name = "health-monitor-test"
        
        client = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            auto_start_daemon=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        # Mock daemon with health check capability
        mock_daemon = Mock()
        mock_daemon.status.state = "running"
        mock_daemon.config.grpc_port = 50055
        mock_daemon.health_check = AsyncMock(return_value=True)
        
        with patch('workspace_qdrant_mcp.core.grpc_client.ensure_daemon_running', return_value=mock_daemon):
            with patch('workspace_qdrant_mcp.grpc.client.AsyncIngestClient') as mock_grpc_client:
                mock_grpc_instance = AsyncMock()
                mock_grpc_instance.test_connection.return_value = True
                mock_grpc_client.return_value = mock_grpc_instance
                
                with patch.object(client.direct_client, 'initialize'):
                    await client.initialize()
                    
                    # Test daemon health check through client
                    is_available = await client.ensure_daemon_available()
                    assert is_available is True
                    
                    # Verify health check was called
                    mock_daemon.health_check.assert_called()
        
        # Cleanup
        from common.core.daemon_manager import shutdown_all_daemons
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_daemon_restart_on_failure(self, mock_config, temp_project_dir):
        """Test daemon restart when health check fails."""
        project_name = "restart-test"
        
        client = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            auto_start_daemon=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        # Mock daemon that initially fails health check
        mock_daemon = Mock()
        mock_daemon.status.state = "running"
        mock_daemon.config.grpc_port = 50056
        mock_daemon.health_check = AsyncMock(side_effect=[False, True])  # Fail then succeed
        
        with patch('workspace_qdrant_mcp.core.grpc_client.ensure_daemon_running', return_value=mock_daemon):
            with patch('workspace_qdrant_mcp.core.grpc_client.get_daemon_for_project', return_value=mock_daemon):
                with patch.object(client, '_initialize_grpc_with_daemon', side_effect=[True, True]):
                    with patch.object(client.direct_client, 'initialize'):
                        await client.initialize()
                        
                        # First call should fail health check
                        is_available = await client.ensure_daemon_available()
                        
                        # Should attempt restart and succeed
                        assert mock_daemon.health_check.call_count >= 1
    
    @pytest.mark.asyncio 
    async def test_daemon_status_reporting(self, mock_config, temp_project_dir):
        """Test daemon status reporting through gRPC client."""
        project_name = "status-test"
        
        client = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            auto_start_daemon=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        # Mock daemon with status
        mock_daemon = Mock()
        mock_daemon.get_status.return_value = {
            "config": {
                "project_name": project_name,
                "grpc_port": 50057
            },
            "status": {
                "state": "running",
                "health_status": "healthy",
                "pid": 12345
            },
            "process_info": {
                "pid": 12345,
                "running": True
            }
        }
        
        client.daemon_instance = mock_daemon
        
        status = await client.get_daemon_status()
        
        assert status is not None
        assert status["config"]["project_name"] == project_name
        assert status["status"]["state"] == "running"
        assert status["process_info"]["pid"] == 12345
    
    @pytest.mark.asyncio
    async def test_multiple_clients_same_daemon(self, mock_config, temp_project_dir):
        """Test multiple gRPC clients sharing the same daemon."""
        project_name = "shared-daemon-test"
        
        # Create two clients for the same project
        client1 = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            auto_start_daemon=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        client2 = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            auto_start_daemon=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        # Mock daemon manager to return same daemon
        mock_daemon = Mock()
        mock_daemon.status.state = "running"
        mock_daemon.config.grpc_port = 50058
        
        with patch('workspace_qdrant_mcp.core.grpc_client.get_daemon_for_project', return_value=mock_daemon):
            with patch('workspace_qdrant_mcp.grpc.client.AsyncIngestClient') as mock_grpc_client:
                mock_grpc_instance = AsyncMock()
                mock_grpc_instance.test_connection.return_value = True
                mock_grpc_client.return_value = mock_grpc_instance
                
                with patch.object(client1.direct_client, 'initialize'):
                    with patch.object(client2.direct_client, 'initialize'):
                        await client1.initialize()
                        await client2.initialize()
                        
                        # Both clients should reference the same daemon
                        assert client1.daemon_instance is mock_daemon
                        assert client2.daemon_instance is mock_daemon
                        
                        # Both should be in gRPC mode
                        assert client1.get_operation_mode() == "grpc"
                        assert client2.get_operation_mode() == "grpc"
        
        # Cleanup
        from common.core.daemon_manager import shutdown_all_daemons
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_daemon_cleanup_on_client_close(self, mock_config, temp_project_dir):
        """Test daemon cleanup behavior when client is closed."""
        project_name = "cleanup-test"
        
        client = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            auto_start_daemon=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        # Mock daemon
        mock_daemon = Mock()
        mock_daemon.status.state = "running"
        mock_daemon.config.grpc_port = 50059
        client.daemon_instance = mock_daemon
        
        # Mock gRPC client
        client.grpc_client = AsyncMock()
        
        # Mock direct client
        client.direct_client = AsyncMock()
        
        await client.close()
        
        # Verify gRPC client was stopped
        client.grpc_client.stop.assert_called_once()
        
        # Verify direct client was closed
        client.direct_client.close.assert_called_once()
        
        # Note: Daemon should NOT be stopped (shared resource)
        # This is handled by the daemon manager's signal handlers
        
        # Cleanup
        from common.core.daemon_manager import shutdown_all_daemons
        await shutdown_all_daemons()


class TestDaemonManagerIntegration:
    """Test daemon manager integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_daemon_manager_signal_handling(self):
        """Test daemon manager signal handling setup."""
        manager = DaemonManager()
        
        # Verify signal handlers are set up
        import signal
        assert signal.signal(signal.SIGINT, None) is not None
        assert signal.signal(signal.SIGTERM, None) is not None
        
        # Test shutdown handler registration
        handler_count_before = len(manager.shutdown_handlers)
        manager.add_shutdown_handler(lambda: None)
        assert len(manager.shutdown_handlers) == handler_count_before + 1
    
    @pytest.mark.asyncio
    async def test_daemon_manager_atexit_handling(self, temp_project_dir):
        """Test daemon manager atexit handling."""
        import atexit
        
        # Count existing handlers
        handlers_before = len(atexit._exithandlers)
        
        # Create daemon manager (should register atexit handler)
        manager = DaemonManager()
        
        # Should have registered an atexit handler
        assert len(atexit._exithandlers) > handlers_before
        
        # Create a daemon to test cleanup
        await manager.get_or_create_daemon("atexit-test", temp_project_dir)
        
        # Manual cleanup for test
        await manager.shutdown_all()
    
    @pytest.mark.asyncio
    async def test_configuration_synchronization(self, temp_project_dir):
        """Test configuration synchronization between Python and daemon."""
        project_name = "config-sync-test"
        
        # Custom configuration
        config_overrides = {
            "grpc_port": 50060,
            "qdrant_url": "http://custom-qdrant:6333",
            "max_concurrent_jobs": 8,
            "log_level": "debug"
        }
        
        daemon = await ensure_daemon_running(
            project_name,
            temp_project_dir,
            config_overrides
        )
        
        # Verify configuration was applied
        assert daemon.config.grpc_port == 50060
        assert daemon.config.qdrant_url == "http://custom-qdrant:6333"
        assert daemon.config.max_concurrent_jobs == 8
        assert daemon.config.log_level == "debug"
        
        # Test config file generation
        await daemon._write_config_file()
        
        assert daemon.config_file.exists()
        
        import json
        with open(daemon.config_file) as f:
            config_data = json.load(f)
        
        assert config_data["grpc"]["port"] == 50060
        assert config_data["qdrant"]["url"] == "http://custom-qdrant:6333"
        assert config_data["processing"]["max_concurrent_jobs"] == 8
        assert config_data["logging"]["level"] == "debug"
        
        # Cleanup
        daemon._cleanup()
        from common.core.daemon_manager import shutdown_all_daemons
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_project_isolation(self, temp_project_dir):
        """Test that different projects get isolated daemon instances."""
        projects = ["project-alpha", "project-beta", "project-gamma"]
        daemons = []
        
        # Create daemons for different projects
        for project in projects:
            daemon = await ensure_daemon_running(project, temp_project_dir)
            daemons.append(daemon)
        
        # Verify each daemon has unique configuration
        ports = [daemon.config.grpc_port for daemon in daemons]
        temp_dirs = [daemon.temp_dir for daemon in daemons]
        config_files = [daemon.config_file for daemon in daemons]
        
        # All should be unique
        assert len(set(ports)) == len(ports)
        assert len(set(str(td) for td in temp_dirs)) == len(temp_dirs)
        assert len(set(str(cf) for cf in config_files)) == len(config_files)
        
        # Each should have correct project name
        for i, project in enumerate(projects):
            assert daemons[i].config.project_name == project
            assert project in str(daemons[i].temp_dir)
        
        # Cleanup
        from common.core.daemon_manager import shutdown_all_daemons
        await shutdown_all_daemons()


# End-to-end integration test
class TestEndToEndIntegration:
    """Test complete end-to-end daemon management integration."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, mock_config, temp_project_dir):
        """Test complete workflow from MCP server startup to shutdown."""
        project_name = "e2e-workflow-test"
        
        # Phase 1: Initialize gRPC client (simulating MCP server startup)
        client = GrpcWorkspaceClient(
            config=mock_config,
            grpc_enabled=True,
            auto_start_daemon=True,
            project_name=project_name,
            project_path=temp_project_dir
        )
        
        # Mock the daemon startup sequence
        mock_daemon = Mock()
        mock_daemon.status.state = "running"
        mock_daemon.config.grpc_port = 50061
        mock_daemon.health_check = AsyncMock(return_value=True)
        mock_daemon.get_status.return_value = {
            "status": {"state": "running", "health_status": "healthy"}
        }
        
        with patch('workspace_qdrant_mcp.core.grpc_client.ensure_daemon_running', return_value=mock_daemon):
            with patch('workspace_qdrant_mcp.grpc.client.AsyncIngestClient') as mock_grpc_client:
                mock_grpc_instance = AsyncMock()
                mock_grpc_instance.test_connection.return_value = True
                mock_grpc_client.return_value = mock_grpc_instance
                
                with patch.object(client.direct_client, 'initialize'):
                    # Phase 2: Client initialization (daemon auto-start)
                    await client.initialize()
                    
                    assert client.get_operation_mode() == "grpc"
                    assert client.daemon_instance is mock_daemon
                    
                    # Phase 3: Health monitoring
                    is_available = await client.ensure_daemon_available()
                    assert is_available is True
                    
                    # Phase 4: Status reporting
                    status = await client.get_daemon_status()
                    assert status["status"]["state"] == "running"
                    
                    # Phase 5: Client shutdown (simulating MCP server shutdown)
                    await client.close()
                    
                    # Verify cleanup was performed
                    mock_grpc_instance.stop.assert_called()
        
        # Phase 6: Global daemon cleanup (simulating process exit)
        from common.core.daemon_manager import shutdown_all_daemons
        await shutdown_all_daemons()
        
        # Verify final cleanup
        from common.core.daemon_manager import get_daemon_for_project
        final_daemon = await get_daemon_for_project(project_name, temp_project_dir)
        assert final_daemon is None