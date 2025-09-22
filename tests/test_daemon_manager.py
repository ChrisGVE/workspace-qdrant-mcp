"""
Tests for daemon process management functionality.

This test module verifies the daemon lifecycle management capabilities including:
- Daemon startup and shutdown automation  
- Health monitoring with periodic heartbeats
- Configuration synchronization between Python and Rust
- Multiple daemon instance management per project
- Graceful shutdown handling and cleanup
- Error recovery and restart logic
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from workspace_qdrant_mcp.core.daemon_manager import (
    DaemonManager, 
    DaemonInstance, 
    DaemonConfig, 
    DaemonStatus,
    ensure_daemon_running,
    get_daemon_for_project,
    shutdown_all_daemons
)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture  
def daemon_config(temp_project_dir):
    """Create a daemon configuration for testing."""
    return DaemonConfig(
        project_name="test-project",
        project_path=temp_project_dir,
        grpc_host="127.0.0.1",
        grpc_port=50052,  # Use different port for tests
        health_check_interval=1.0,  # Faster for testing
        startup_timeout=5.0,
        shutdown_timeout=2.0,
        restart_on_failure=False  # Disable auto-restart for tests
    )


@pytest.fixture
def mock_process():
    """Mock asyncio subprocess for testing."""
    process = Mock()
    process.pid = 12345
    process.returncode = None
    process.stdout = AsyncMock()
    process.stderr = AsyncMock()
    process.terminate = Mock()
    process.kill = Mock()
    process.wait = AsyncMock(return_value=0)
    return process


class TestDaemonConfig:
    """Test daemon configuration data class."""
    
    def test_default_config_values(self):
        """Test default configuration values."""
        config = DaemonConfig(
            project_name="test",
            project_path="/tmp/test"
        )
        
        assert config.project_name == "test"
        assert config.project_path == "/tmp/test"
        assert config.grpc_host == "127.0.0.1"
        assert config.grpc_port == 50051
        assert config.health_check_interval == 30.0
        assert config.restart_on_failure is True
    
    def test_custom_config_values(self):
        """Test custom configuration values."""
        config = DaemonConfig(
            project_name="custom-project",
            project_path="/custom/path",
            grpc_port=60000,
            health_check_interval=10.0,
            max_restart_attempts=5
        )
        
        assert config.project_name == "custom-project"
        assert config.grpc_port == 60000
        assert config.health_check_interval == 10.0
        assert config.max_restart_attempts == 5


class TestDaemonStatus:
    """Test daemon status tracking."""
    
    def test_default_status(self):
        """Test default status values."""
        status = DaemonStatus()
        
        assert status.pid is None
        assert status.state == "stopped"
        assert status.start_time is None
        assert status.health_status == "unknown"
        assert status.restart_count == 0
        assert status.grpc_available is False
    
    def test_status_updates(self):
        """Test status field updates."""
        status = DaemonStatus()
        
        # Update status fields
        status.pid = 12345
        status.state = "running"
        status.start_time = datetime.now()
        status.health_status = "healthy"
        status.grpc_available = True
        
        assert status.pid == 12345
        assert status.state == "running"
        assert status.start_time is not None
        assert status.health_status == "healthy"
        assert status.grpc_available is True


class TestDaemonInstance:
    """Test individual daemon instance management."""
    
    @pytest.mark.asyncio
    async def test_daemon_instance_creation(self, daemon_config):
        """Test daemon instance creation."""
        daemon = DaemonInstance(daemon_config)
        
        assert daemon.config == daemon_config
        assert daemon.status.state == "stopped"
        assert daemon.process is None
        assert daemon.temp_dir.exists()
        
        # Cleanup
        daemon._cleanup()
    
    @pytest.mark.asyncio
    async def test_config_file_writing(self, daemon_config):
        """Test daemon configuration file writing."""
        daemon = DaemonInstance(daemon_config)
        
        await daemon._write_config_file()
        
        assert daemon.config_file.exists()
        
        # Verify config content
        import json
        with open(daemon.config_file) as f:
            config_data = json.load(f)
        
        assert config_data["project_name"] == "test-project"
        assert config_data["grpc"]["port"] == 50052
        assert config_data["qdrant"]["url"] == "http://localhost:6333"
        
        # Cleanup
        daemon._cleanup()
    
    @pytest.mark.asyncio
    async def test_daemon_startup_no_binary(self, daemon_config):
        """Test daemon startup when binary is not available."""
        daemon = DaemonInstance(daemon_config)
        
        with patch.object(daemon, '_find_daemon_binary', return_value=None):
            success = await daemon.start()
            assert success is False
            assert daemon.status.state == "failed"
            assert "Daemon binary not found" in daemon.status.last_error
        
        # Cleanup
        daemon._cleanup()
    
    @pytest.mark.asyncio
    async def test_daemon_startup_process_creation(self, daemon_config, mock_process):
        """Test daemon startup with process creation."""
        daemon = DaemonInstance(daemon_config)
        
        with patch.object(daemon, '_find_daemon_binary', return_value=Path("/fake/binary")):
            with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                with patch.object(daemon, '_wait_for_startup', return_value=True):
                    success = await daemon.start()
                    
                    assert success is True
                    assert daemon.status.state == "running"
                    assert daemon.status.pid == 12345
                    assert daemon.process == mock_process
        
        # Cleanup
        daemon._cleanup()
    
    @pytest.mark.asyncio
    async def test_daemon_shutdown_graceful(self, daemon_config, mock_process):
        """Test graceful daemon shutdown."""
        daemon = DaemonInstance(daemon_config)
        daemon.process = mock_process
        daemon.status.state = "running"
        daemon.status.pid = 12345
        
        success = await daemon.stop()
        
        assert success is True
        assert daemon.status.state == "stopped"
        assert daemon.status.pid is None
        mock_process.terminate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_shutdown_forced(self, daemon_config, mock_process):
        """Test forced daemon shutdown when graceful fails."""
        daemon = DaemonInstance(daemon_config)
        daemon.process = mock_process
        daemon.status.state = "running"
        daemon.status.pid = 12345
        
        # Make wait() timeout to trigger force kill
        mock_process.wait.side_effect = asyncio.TimeoutError()
        
        success = await daemon.stop()
        
        assert success is True
        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_process_exited(self, daemon_config, mock_process):
        """Test health check when process has exited."""
        daemon = DaemonInstance(daemon_config)
        daemon.process = mock_process
        daemon.status.state = "running"
        
        # Simulate process exit
        mock_process.returncode = 1
        
        is_healthy = await daemon.health_check()
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_health_check_grpc_connection(self, daemon_config, mock_process):
        """Test health check with gRPC connection test."""
        daemon = DaemonInstance(daemon_config)
        daemon.process = mock_process
        daemon.status.state = "running"
        
        # Mock successful gRPC connection
        mock_client = AsyncMock()
        mock_client.test_connection.return_value = True
        
        with patch('workspace_qdrant_mcp.grpc.client.AsyncIngestClient', return_value=mock_client):
            is_healthy = await daemon.health_check()
            
            assert is_healthy is True
            assert daemon.status.health_status == "healthy"
            assert daemon.status.grpc_available is True
            mock_client.start.assert_called_once()
            mock_client.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_restart_with_backoff(self, daemon_config):
        """Test daemon restart with exponential backoff."""
        daemon = DaemonInstance(daemon_config)
        daemon.status.restart_count = 2
        
        start_time = time.time()
        
        with patch.object(daemon, 'stop'):
            with patch.object(daemon, 'start', return_value=True):
                success = await daemon.restart()
                
                elapsed = time.time() - start_time
                expected_delay = min(daemon.config.restart_backoff_base ** 2, 30.0)
                
                assert success is True
                assert daemon.status.restart_count == 3
                assert elapsed >= expected_delay * 0.9  # Allow some tolerance
    
    def test_daemon_status_reporting(self, daemon_config):
        """Test comprehensive status reporting."""
        daemon = DaemonInstance(daemon_config)
        daemon.status.pid = 12345
        daemon.status.state = "running"
        daemon.status.health_status = "healthy"
        
        status = daemon.get_status()
        
        assert status["config"]["project_name"] == "test-project"
        assert status["status"]["pid"] == 12345
        assert status["status"]["state"] == "running"
        assert status["status"]["health_status"] == "healthy"
        assert "process_info" in status
        
        # Cleanup
        daemon._cleanup()


class TestDaemonManager:
    """Test daemon manager orchestration."""
    
    @pytest.mark.asyncio
    async def test_singleton_instance(self):
        """Test daemon manager singleton pattern."""
        manager1 = await DaemonManager.get_instance()
        manager2 = await DaemonManager.get_instance()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_get_or_create_daemon(self, temp_project_dir):
        """Test daemon creation and retrieval."""
        manager = DaemonManager()
        
        # Create daemon
        daemon1 = await manager.get_or_create_daemon("test-proj", temp_project_dir)
        assert daemon1.config.project_name == "test-proj"
        assert daemon1.config.project_path == temp_project_dir
        
        # Get same daemon
        daemon2 = await manager.get_or_create_daemon("test-proj", temp_project_dir)
        assert daemon1 is daemon2
        
        # Cleanup
        await manager.shutdown_all()
    
    @pytest.mark.asyncio
    async def test_daemon_port_assignment(self, temp_project_dir):
        """Test unique port assignment for different projects."""
        manager = DaemonManager()
        
        daemon1 = await manager.get_or_create_daemon("project-a", temp_project_dir)
        daemon2 = await manager.get_or_create_daemon("project-b", temp_project_dir)
        
        # Should have different ports
        assert daemon1.config.grpc_port != daemon2.config.grpc_port
        
        # Same project should get same port
        daemon3 = await manager.get_or_create_daemon("project-a", temp_project_dir)
        assert daemon1.config.grpc_port == daemon3.config.grpc_port
        
        # Cleanup
        await manager.shutdown_all()
    
    @pytest.mark.asyncio
    async def test_start_daemon_success(self, temp_project_dir):
        """Test successful daemon startup."""
        manager = DaemonManager()
        
        with patch.object(DaemonInstance, 'start', return_value=True):
            success = await manager.start_daemon("test-proj", temp_project_dir)
            assert success is True
        
        # Cleanup
        await manager.shutdown_all()
    
    @pytest.mark.asyncio
    async def test_stop_daemon(self, temp_project_dir):
        """Test daemon stopping."""
        manager = DaemonManager()
        daemon = await manager.get_or_create_daemon("test-proj", temp_project_dir)
        
        with patch.object(daemon, 'stop', return_value=True) as mock_stop:
            success = await manager.stop_daemon("test-proj", temp_project_dir)
            assert success is True
            mock_stop.assert_called_once()
        
        # Daemon should be removed from active daemons
        status = await manager.get_daemon_status("test-proj", temp_project_dir)
        assert status is None
    
    @pytest.mark.asyncio
    async def test_health_check_all(self, temp_project_dir):
        """Test health checking all daemons."""
        manager = DaemonManager()
        
        # Create multiple daemons
        daemon1 = await manager.get_or_create_daemon("proj-1", temp_project_dir)
        daemon2 = await manager.get_or_create_daemon("proj-2", temp_project_dir)
        
        with patch.object(daemon1, 'health_check', return_value=True):
            with patch.object(daemon2, 'health_check', return_value=False):
                results = await manager.health_check_all()
                
                assert len(results) == 2
                assert any(result is True for result in results.values())
                assert any(result is False for result in results.values())
        
        # Cleanup
        await manager.shutdown_all()
    
    @pytest.mark.asyncio
    async def test_shutdown_all_daemons(self, temp_project_dir):
        """Test shutting down all daemons."""
        manager = DaemonManager()
        
        # Create multiple daemons
        daemon1 = await manager.get_or_create_daemon("proj-1", temp_project_dir)
        daemon2 = await manager.get_or_create_daemon("proj-2", temp_project_dir)
        
        with patch.object(daemon1, 'stop', return_value=True) as mock_stop1:
            with patch.object(daemon2, 'stop', return_value=True) as mock_stop2:
                await manager.shutdown_all()
                
                mock_stop1.assert_called_once()
                mock_stop2.assert_called_once()
                assert len(manager.daemons) == 0
    
    def test_daemon_key_generation(self, temp_project_dir):
        """Test daemon key generation for unique identification."""
        manager = DaemonManager()
        
        key1 = manager._get_daemon_key("project-a", "/path/to/project-a")
        key2 = manager._get_daemon_key("project-b", "/path/to/project-b")
        key3 = manager._get_daemon_key("project-a", "/path/to/project-a")  # Same as key1
        
        # Different projects should have different keys
        assert key1 != key2
        
        # Same project should have same key
        assert key1 == key3
        
        # Keys should include project name
        assert "project-a" in key1
        assert "project-b" in key2
    
    def test_port_assignment_consistency(self):
        """Test consistent port assignment for same project."""
        manager = DaemonManager()
        
        port1 = manager._get_available_port("project-name")
        port2 = manager._get_available_port("project-name")
        port3 = manager._get_available_port("different-project")
        
        # Same project should get same port
        assert port1 == port2
        
        # Different project should get different port  
        assert port1 != port3
        
        # All ports should be in valid range
        assert 50051 <= port1 <= 51050
        assert 50051 <= port3 <= 51050


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""
    
    @pytest.mark.asyncio
    async def test_ensure_daemon_running_new(self, temp_project_dir):
        """Test ensuring daemon is running for new project."""
        with patch.object(DaemonInstance, 'start', return_value=True):
            daemon = await ensure_daemon_running("test-proj", temp_project_dir)
            
            assert daemon.config.project_name == "test-proj"
            assert daemon.config.project_path == temp_project_dir
        
        # Cleanup
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_ensure_daemon_running_existing(self, temp_project_dir):
        """Test ensuring daemon is running for existing project."""
        # Start daemon first
        daemon1 = await ensure_daemon_running("test-proj", temp_project_dir)
        daemon1.status.state = "running"
        
        # Should return same daemon without restarting
        daemon2 = await ensure_daemon_running("test-proj", temp_project_dir)
        assert daemon1 is daemon2
        
        # Cleanup
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_ensure_daemon_running_failure(self, temp_project_dir):
        """Test daemon startup failure handling."""
        with patch.object(DaemonInstance, 'start', return_value=False):
            with pytest.raises(RuntimeError, match="Failed to start daemon"):
                await ensure_daemon_running("test-proj", temp_project_dir)
        
        # Cleanup
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_get_daemon_for_project_exists(self, temp_project_dir):
        """Test getting daemon when it exists."""
        # Create daemon first
        await ensure_daemon_running("test-proj", temp_project_dir)
        
        daemon = await get_daemon_for_project("test-proj", temp_project_dir)
        assert daemon is not None
        assert daemon.config.project_name == "test-proj"
        
        # Cleanup
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_get_daemon_for_project_not_exists(self, temp_project_dir):
        """Test getting daemon when it doesn't exist."""
        daemon = await get_daemon_for_project("nonexistent", temp_project_dir)
        assert daemon is None
    
    @pytest.mark.asyncio
    async def test_shutdown_all_daemons_function(self, temp_project_dir):
        """Test module-level shutdown function."""
        # Create some daemons
        await ensure_daemon_running("proj-1", temp_project_dir)
        await ensure_daemon_running("proj-2", temp_project_dir)
        
        # Should shutdown all daemons
        await shutdown_all_daemons()
        
        # Verify daemons are gone
        daemon1 = await get_daemon_for_project("proj-1", temp_project_dir)
        daemon2 = await get_daemon_for_project("proj-2", temp_project_dir)
        assert daemon1 is None
        assert daemon2 is None


class TestIntegrationScenarios:
    """Test comprehensive integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_daemon_lifecycle(self, temp_project_dir):
        """Test complete daemon lifecycle from start to finish."""
        project_name = "integration-test"
        
        # Phase 1: Start daemon
        daemon = await ensure_daemon_running(project_name, temp_project_dir)
        assert daemon.config.project_name == project_name
        assert daemon.status.state == "stopped"  # Not actually started due to no binary
        
        # Phase 2: Simulate startup
        with patch.object(daemon, '_find_daemon_binary', return_value=Path("/fake/binary")):
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = Mock()
                mock_process.pid = 99999
                mock_process.returncode = None
                mock_subprocess.return_value = mock_process
                
                with patch.object(daemon, '_wait_for_startup', return_value=True):
                    success = await daemon.start()
                    assert success is True
                    assert daemon.status.state == "running"
        
        # Phase 3: Health monitoring
        with patch.object(daemon, 'health_check', return_value=True):
            is_healthy = await daemon.health_check()
            assert is_healthy is True
            assert daemon.status.health_status == "healthy"
        
        # Phase 4: Graceful shutdown
        with patch.object(daemon.process, 'terminate'):
            with patch.object(daemon.process, 'wait'):
                success = await daemon.stop()
                assert success is True
                assert daemon.status.state == "stopped"
        
        # Phase 5: Cleanup
        await shutdown_all_daemons()
    
    @pytest.mark.asyncio
    async def test_multiple_project_management(self, temp_project_dir):
        """Test managing multiple project daemons simultaneously."""
        projects = ["project-alpha", "project-beta", "project-gamma"]
        daemons = []
        
        # Start multiple daemons
        for project in projects:
            daemon = await ensure_daemon_running(project, temp_project_dir)
            daemons.append(daemon)
        
        # Verify each has unique port
        ports = [daemon.config.grpc_port for daemon in daemons]
        assert len(set(ports)) == len(ports)  # All unique
        
        # Test health check all
        manager = await DaemonManager.get_instance()
        with patch.object(DaemonInstance, 'health_check', return_value=True):
            health_results = await manager.health_check_all()
            assert len(health_results) == len(projects)
            assert all(health_results.values())
        
        # Cleanup
        await shutdown_all_daemons()
        
        # Verify all cleaned up
        for project in projects:
            daemon = await get_daemon_for_project(project, temp_project_dir)
            assert daemon is None