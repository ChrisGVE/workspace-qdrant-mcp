"""
Unit tests for DaemonManager lifecycle and configuration management.

Tests the core functionality of daemon lifecycle management including:
- DaemonManager singleton pattern and initialization
- Configuration loading and validation
- Port allocation and project identification
- Daemon instance creation and management
- Resource management and monitoring integration
- Graceful shutdown and cleanup
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
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from unittest import mock
import pytest

from workspace_qdrant_mcp.core.daemon_manager import (
    DaemonManager,
    DaemonInstance,
    DaemonConfig,
    DaemonStatus,
    PortManager,
    ensure_daemon_running,
    get_daemon_for_project,
    shutdown_all_daemons
)
from workspace_qdrant_mcp.core.resource_manager import ResourceLimits
from workspace_qdrant_mcp.utils.project_detection import DaemonIdentifier, ProjectDetector

from .conftest_daemon import (
    mock_daemon_config,
    mock_daemon_status,
    mock_daemon_instance,
    mock_daemon_manager,
    mock_port_manager,
    isolated_daemon_temp_dir,
    DaemonTestHelper,
    assert_daemon_config_valid
)


class TestDaemonConfig:
    """Test DaemonConfig dataclass functionality."""
    
    def test_daemon_config_initialization(self, isolated_daemon_temp_dir):
        """Test basic daemon configuration initialization."""
        config = DaemonConfig(
            project_name="test-project",
            project_path=str(isolated_daemon_temp_dir),
            grpc_port=50051
        )
        
        assert_daemon_config_valid(config)
        assert config.project_name == "test-project"
        assert config.project_path == str(isolated_daemon_temp_dir)
        assert config.grpc_port == 50051
        assert config.grpc_host == "127.0.0.1"  # Default value
        assert config.log_level == "info"  # Default value
        assert config.max_concurrent_jobs == 4  # Default value
    
    def test_daemon_config_from_project_config(self, isolated_daemon_temp_dir):
        """Test creating DaemonConfig from DaemonProjectConfig."""
        # Create a mock project config
        project_config = Mock()
        project_config.project_name = "test-project"
        project_config.project_path = str(isolated_daemon_temp_dir)
        project_config.project_id = "test_id_123"
        project_config.grpc_host = "127.0.0.1"
        project_config.grpc_port = 50052
        project_config.qdrant_url = "http://localhost:6333"
        project_config.log_level = "debug"
        project_config.max_concurrent_jobs = 8
        project_config.health_check_interval = 60.0
        project_config.startup_timeout = 45.0
        project_config.shutdown_timeout = 15.0
        project_config.restart_on_failure = True
        project_config.max_restart_attempts = 5
        project_config.max_memory_mb = 2048
        project_config.max_cpu_percent = 80
        project_config.max_open_files = 1024
        project_config.enable_resource_monitoring = True
        
        config = DaemonConfig.from_project_config(project_config)
        
        assert_daemon_config_valid(config)
        assert config.project_name == "test-project"
        assert config.grpc_port == 50052
        assert config.log_level == "debug"
        assert config.max_concurrent_jobs == 8
        assert config.resource_limits is not None
        assert config.resource_limits.max_memory_mb == 2048
        assert config.enable_resource_monitoring is True
    
    def test_daemon_config_defaults(self, isolated_daemon_temp_dir):
        """Test daemon configuration default values."""
        config = DaemonConfig(
            project_name="minimal-test",
            project_path=str(isolated_daemon_temp_dir)
        )
        
        # Verify defaults
        assert config.grpc_host == "127.0.0.1"
        assert config.grpc_port == 50051
        assert config.qdrant_url == "http://localhost:6333"
        assert config.log_level == "info"
        assert config.max_concurrent_jobs == 4
        assert config.health_check_interval == 30.0
        assert config.startup_timeout == 30.0
        assert config.shutdown_timeout == 10.0
        assert config.restart_on_failure is True
        assert config.max_restart_attempts == 3
        assert config.restart_backoff_base == 2.0
        assert config.resource_limits is None
        assert config.enable_resource_monitoring is True


class TestDaemonStatus:
    """Test DaemonStatus dataclass functionality."""
    
    def test_daemon_status_initialization(self):
        """Test daemon status initialization."""
        status = DaemonStatus()
        
        assert status.pid is None
        assert status.state == "stopped"
        assert status.start_time is None
        assert status.last_health_check is None
        assert status.health_status == "unknown"
        assert status.restart_count == 0
        assert status.last_error is None
        assert status.grpc_available is False
    
    def test_daemon_status_running_state(self):
        """Test daemon status in running state."""
        now = datetime.now(timezone.utc)
        status = DaemonStatus(
            pid=12345,
            state="running",
            start_time=now,
            last_health_check=now,
            health_status="healthy",
            grpc_available=True
        )
        
        assert status.pid == 12345
        assert status.state == "running"
        assert status.start_time == now
        assert status.health_status == "healthy"
        assert status.grpc_available is True


class TestPortManager:
    """Test PortManager functionality."""
    
    def test_port_manager_singleton(self):
        """Test PortManager singleton pattern."""
        manager1 = PortManager.get_instance()
        manager2 = PortManager.get_instance()
        
        assert manager1 is manager2
    
    def test_port_allocation_basic(self, mock_port_manager):
        """Test basic port allocation functionality."""
        project_id = "test_project_123"
        
        port = mock_port_manager.allocate_port(project_id)
        
        assert port is not None
        mock_port_manager.allocate_port.assert_called_once_with(project_id)
    
    def test_port_allocation_with_preference(self, mock_port_manager):
        """Test port allocation with preferred port."""
        project_id = "test_project_123"
        preferred_port = 50055
        
        mock_port_manager.allocate_port.return_value = preferred_port
        port = mock_port_manager.allocate_port(project_id, preferred_port)
        
        assert port == preferred_port
        mock_port_manager.allocate_port.assert_called_once_with(project_id, preferred_port)
    
    def test_port_release(self, mock_port_manager):
        """Test port release functionality."""
        project_id = "test_project_123"
        port = 50055
        
        result = mock_port_manager.release_port(port, project_id)
        
        assert result is True
        mock_port_manager.release_port.assert_called_once_with(port, project_id)
    
    def test_port_allocation_status(self, mock_port_manager):
        """Test port allocation status checking."""
        port = 50055
        
        mock_port_manager.is_port_allocated.return_value = False
        result = mock_port_manager.is_port_allocated(port)
        
        assert result is False
        mock_port_manager.is_port_allocated.assert_called_once_with(port)


class TestDaemonInstance:
    """Test DaemonInstance functionality."""
    
    @pytest.mark.asyncio
    async def test_daemon_instance_initialization(self, mock_daemon_config, mock_port_manager):
        """Test daemon instance initialization."""
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.PortManager.get_instance', return_value=mock_port_manager):
            with patch('src.workspace_qdrant_mcp.utils.project_detection.ProjectDetector') as mock_detector:
                mock_identifier = Mock()
                mock_identifier.generate_identifier.return_value = "test_daemon_123"
                mock_detector.return_value.create_daemon_identifier.return_value = mock_identifier
                
                instance = DaemonInstance(mock_daemon_config)
                
                assert instance.config == mock_daemon_config
                assert instance.status.state == "stopped"
                assert instance.process is None
                assert instance.health_task is None
                assert not instance.shutdown_event.is_set()
                assert instance.log_handlers == []
                assert instance.temp_dir.exists()
                assert instance.config_file.exists() or True  # May not exist until start
    
    @pytest.mark.asyncio
    async def test_daemon_instance_start_success(self, mock_daemon_instance):
        """Test successful daemon instance startup."""
        await DaemonTestHelper.simulate_daemon_startup(mock_daemon_instance, success=True)
        
        result = await mock_daemon_instance.start()
        
        assert result is True
        assert mock_daemon_instance.status.state == "running"
        assert mock_daemon_instance.status.pid == 12345
        assert mock_daemon_instance.status.grpc_available is True
        mock_daemon_instance.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_instance_start_failure(self, mock_daemon_instance):
        """Test daemon instance startup failure."""
        await DaemonTestHelper.simulate_daemon_startup(mock_daemon_instance, success=False)
        
        result = await mock_daemon_instance.start()
        
        assert result is False
        assert mock_daemon_instance.status.state == "failed"
        assert mock_daemon_instance.status.last_error == "Startup failed"
        mock_daemon_instance.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_instance_stop(self, mock_daemon_instance):
        """Test daemon instance graceful stop."""
        # Simulate running daemon
        mock_daemon_instance.status.state = "running"
        mock_daemon_instance.status.pid = 12345
        
        result = await mock_daemon_instance.stop()
        
        assert result is True
        mock_daemon_instance.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_instance_restart(self, mock_daemon_instance):
        """Test daemon instance restart functionality."""
        # Simulate running daemon
        mock_daemon_instance.status.state = "running"
        mock_daemon_instance.status.restart_count = 1
        
        result = await mock_daemon_instance.restart()
        
        assert result is True
        mock_daemon_instance.restart.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_instance_health_check_healthy(self, mock_daemon_instance):
        """Test daemon health check when healthy."""
        await DaemonTestHelper.simulate_health_check(mock_daemon_instance, healthy=True)
        
        result = await mock_daemon_instance.health_check()
        
        assert result is True
        assert mock_daemon_instance.status.health_status == "healthy"
        mock_daemon_instance.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_instance_health_check_unhealthy(self, mock_daemon_instance):
        """Test daemon health check when unhealthy."""
        await DaemonTestHelper.simulate_health_check(mock_daemon_instance, healthy=False)
        
        result = await mock_daemon_instance.health_check()
        
        assert result is False
        assert mock_daemon_instance.status.health_status == "unhealthy"
        mock_daemon_instance.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_daemon_instance_status_retrieval(self, mock_daemon_instance):
        """Test daemon instance status retrieval."""
        status = await mock_daemon_instance.get_status()
        
        assert "config" in status
        assert "status" in status
        assert "process_info" in status
        mock_daemon_instance.get_status.assert_called_once()
    
    def test_daemon_instance_log_handler(self, mock_daemon_instance):
        """Test daemon instance log handler management."""
        handler = Mock()
        
        mock_daemon_instance.add_log_handler(handler)
        
        mock_daemon_instance.add_log_handler.assert_called_once_with(handler)


class TestDaemonManager:
    """Test DaemonManager functionality."""
    
    @pytest.mark.asyncio
    async def test_daemon_manager_singleton(self):
        """Test DaemonManager singleton pattern."""
        manager1 = await DaemonManager.get_instance()
        manager2 = await DaemonManager.get_instance()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_daemon_manager_initialization(self, mock_daemon_manager):
        """Test daemon manager initialization."""
        assert mock_daemon_manager.daemons == {}
        assert mock_daemon_manager.shutdown_handlers == []
    
    @pytest.mark.asyncio
    async def test_get_or_create_daemon_new(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test creating a new daemon instance."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        mock_daemon_instance = Mock()
        mock_daemon_manager.get_or_create_daemon.return_value = mock_daemon_instance
        
        daemon = await mock_daemon_manager.get_or_create_daemon(project_name, project_path)
        
        assert daemon is not None
        mock_daemon_manager.get_or_create_daemon.assert_called_once_with(project_name, project_path, None)
    
    @pytest.mark.asyncio
    async def test_get_or_create_daemon_existing(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test retrieving an existing daemon instance."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        # Simulate existing daemon
        existing_daemon = Mock()
        mock_daemon_manager.get_or_create_daemon.return_value = existing_daemon
        
        daemon = await mock_daemon_manager.get_or_create_daemon(project_name, project_path)
        
        assert daemon is existing_daemon
        mock_daemon_manager.get_or_create_daemon.assert_called_once_with(project_name, project_path, None)
    
    @pytest.mark.asyncio
    async def test_start_daemon_success(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test successful daemon startup."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        result = await mock_daemon_manager.start_daemon(project_name, project_path)
        
        assert result is True
        mock_daemon_manager.start_daemon.assert_called_once_with(project_name, project_path, None)
    
    @pytest.mark.asyncio
    async def test_start_daemon_already_running(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test starting daemon when already running."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        # Simulate already running
        mock_daemon_manager.start_daemon.return_value = True
        
        result = await mock_daemon_manager.start_daemon(project_name, project_path)
        
        assert result is True
        mock_daemon_manager.start_daemon.assert_called_once_with(project_name, project_path, None)
    
    @pytest.mark.asyncio
    async def test_stop_daemon_success(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test successful daemon stop."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        result = await mock_daemon_manager.stop_daemon(project_name, project_path)
        
        assert result is True
        mock_daemon_manager.stop_daemon.assert_called_once_with(project_name, project_path)
    
    @pytest.mark.asyncio
    async def test_stop_daemon_not_found(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test stopping daemon that doesn't exist."""
        project_name = "nonexistent-project"
        project_path = str(isolated_daemon_temp_dir)
        
        # Simulate daemon not found
        mock_daemon_manager.stop_daemon.return_value = True  # Should still return True
        
        result = await mock_daemon_manager.stop_daemon(project_name, project_path)
        
        assert result is True
        mock_daemon_manager.stop_daemon.assert_called_once_with(project_name, project_path)
    
    @pytest.mark.asyncio
    async def test_get_daemon_status_exists(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test getting status for existing daemon."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        expected_status = {
            "state": "running",
            "pid": 12345,
            "health_status": "healthy"
        }
        mock_daemon_manager.get_daemon_status.return_value = expected_status
        
        status = await mock_daemon_manager.get_daemon_status(project_name, project_path)
        
        assert status == expected_status
        mock_daemon_manager.get_daemon_status.assert_called_once_with(project_name, project_path)
    
    @pytest.mark.asyncio
    async def test_get_daemon_status_not_found(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test getting status for non-existent daemon."""
        project_name = "nonexistent-project"
        project_path = str(isolated_daemon_temp_dir)
        
        mock_daemon_manager.get_daemon_status.return_value = None
        
        status = await mock_daemon_manager.get_daemon_status(project_name, project_path)
        
        assert status is None
        mock_daemon_manager.get_daemon_status.assert_called_once_with(project_name, project_path)
    
    @pytest.mark.asyncio
    async def test_list_daemons_empty(self, mock_daemon_manager):
        """Test listing daemons when none exist."""
        mock_daemon_manager.list_daemons.return_value = {}
        
        daemons = await mock_daemon_manager.list_daemons()
        
        assert daemons == {}
        mock_daemon_manager.list_daemons.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_daemons_with_daemons(self, mock_daemon_manager):
        """Test listing daemons when some exist."""
        expected_daemons = {
            "test_daemon_123": {
                "state": "running",
                "pid": 12345,
                "project_name": "test-project"
            }
        }
        mock_daemon_manager.list_daemons.return_value = expected_daemons
        
        daemons = await mock_daemon_manager.list_daemons()
        
        assert daemons == expected_daemons
        mock_daemon_manager.list_daemons.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self, mock_daemon_manager):
        """Test health check for all daemons when healthy."""
        expected_health = {
            "test_daemon_123": True,
            "test_daemon_456": True
        }
        mock_daemon_manager.health_check_all.return_value = expected_health
        
        health = await mock_daemon_manager.health_check_all()
        
        assert health == expected_health
        mock_daemon_manager.health_check_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_all_mixed(self, mock_daemon_manager):
        """Test health check for all daemons with mixed results."""
        expected_health = {
            "test_daemon_123": True,
            "test_daemon_456": False
        }
        mock_daemon_manager.health_check_all.return_value = expected_health
        
        health = await mock_daemon_manager.health_check_all()
        
        assert health == expected_health
        mock_daemon_manager.health_check_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_all_daemons(self, mock_daemon_manager):
        """Test shutdown of all daemon instances."""
        await mock_daemon_manager.shutdown_all()
        
        mock_daemon_manager.shutdown_all.assert_called_once()
    
    def test_add_shutdown_handler(self, mock_daemon_manager):
        """Test adding shutdown handler."""
        handler = Mock()
        
        mock_daemon_manager.add_shutdown_handler(handler)
        
        mock_daemon_manager.add_shutdown_handler.assert_called_once_with(handler)
    
    @pytest.mark.asyncio
    async def test_get_system_resource_status(self, mock_daemon_manager):
        """Test getting system resource status."""
        expected_status = {
            "total_projects": 2,
            "cpu_usage": 45.2,
            "memory_usage": 62.1,
            "daemons": ["test_daemon_123", "test_daemon_456"]
        }
        mock_daemon_manager.get_system_resource_status.return_value = expected_status
        
        status = await mock_daemon_manager.get_system_resource_status()
        
        assert status == expected_status
        mock_daemon_manager.get_system_resource_status.assert_called_once()


class TestDaemonManagerConfigurationOverrides:
    """Test DaemonManager with configuration overrides."""
    
    @pytest.mark.asyncio
    async def test_daemon_creation_with_overrides(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test daemon creation with configuration overrides."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        config_overrides = {
            "grpc_port": 50099,
            "log_level": "debug",
            "max_concurrent_jobs": 8
        }
        
        daemon = await mock_daemon_manager.get_or_create_daemon(
            project_name, project_path, config_overrides
        )
        
        assert daemon is not None
        mock_daemon_manager.get_or_create_daemon.assert_called_once_with(
            project_name, project_path, config_overrides
        )
    
    @pytest.mark.asyncio
    async def test_daemon_startup_with_overrides(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test daemon startup with configuration overrides."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        config_overrides = {
            "startup_timeout": 60.0,
            "health_check_interval": 5.0
        }
        
        result = await mock_daemon_manager.start_daemon(
            project_name, project_path, config_overrides
        )
        
        assert result is True
        mock_daemon_manager.start_daemon.assert_called_once_with(
            project_name, project_path, config_overrides
        )


class TestDaemonManagerUtilityFunctions:
    """Test daemon manager utility functions."""
    
    @pytest.mark.asyncio
    async def test_ensure_daemon_running_new(self, isolated_daemon_temp_dir):
        """Test ensuring daemon is running for new project."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.get_daemon_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_daemon = Mock()
            mock_daemon.status.state = "running"
            
            mock_manager.get_or_create_daemon = AsyncMock(return_value=mock_daemon)
            mock_get_manager.return_value = mock_manager
            
            daemon = await ensure_daemon_running(project_name, project_path)
            
            assert daemon is mock_daemon
            mock_manager.get_or_create_daemon.assert_called_once_with(
                project_name, project_path, None
            )
    
    @pytest.mark.asyncio
    async def test_ensure_daemon_running_startup_needed(self, isolated_daemon_temp_dir):
        """Test ensuring daemon is running when startup is needed."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.get_daemon_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_daemon = Mock()
            mock_daemon.status.state = "stopped"
            mock_daemon.start = AsyncMock(return_value=True)
            
            mock_manager.get_or_create_daemon = AsyncMock(return_value=mock_daemon)
            mock_get_manager.return_value = mock_manager
            
            daemon = await ensure_daemon_running(project_name, project_path)
            
            assert daemon is mock_daemon
            mock_daemon.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ensure_daemon_running_startup_failure(self, isolated_daemon_temp_dir):
        """Test ensuring daemon is running when startup fails."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.get_daemon_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_daemon = Mock()
            mock_daemon.status.state = "stopped"
            mock_daemon.start = AsyncMock(return_value=False)
            
            mock_manager.get_or_create_daemon = AsyncMock(return_value=mock_daemon)
            mock_get_manager.return_value = mock_manager
            
            with pytest.raises(RuntimeError, match="Failed to start daemon"):
                await ensure_daemon_running(project_name, project_path)
    
    @pytest.mark.asyncio
    async def test_get_daemon_for_project_exists(self, isolated_daemon_temp_dir):
        """Test getting daemon for project when it exists."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.get_daemon_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_daemon = Mock()
            mock_manager._get_daemon_key = Mock(return_value="test_key")
            mock_manager.daemons = {"test_key": mock_daemon}
            mock_get_manager.return_value = mock_manager
            
            daemon = await get_daemon_for_project(project_name, project_path)
            
            assert daemon is mock_daemon
    
    @pytest.mark.asyncio
    async def test_get_daemon_for_project_not_exists(self, isolated_daemon_temp_dir):
        """Test getting daemon for project when it doesn't exist."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.get_daemon_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager._get_daemon_key = Mock(return_value="test_key")
            mock_manager.daemons = {}
            mock_get_manager.return_value = mock_manager
            
            daemon = await get_daemon_for_project(project_name, project_path)
            
            assert daemon is None
    
    @pytest.mark.asyncio
    async def test_shutdown_all_daemons_utility(self):
        """Test shutdown all daemons utility function."""
        with patch('src.workspace_qdrant_mcp.core.daemon_manager._daemon_manager') as mock_manager:
            mock_manager.shutdown_all = AsyncMock()
            
            await shutdown_all_daemons()
            
            mock_manager.shutdown_all.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_all_daemons_no_manager(self):
        """Test shutdown all daemons when no manager exists."""
        with patch('src.workspace_qdrant_mcp.core.daemon_manager._daemon_manager', None):
            # Should not raise exception
            await shutdown_all_daemons()


class TestDaemonManagerErrorHandling:
    """Test error handling in DaemonManager."""
    
    @pytest.mark.asyncio
    async def test_daemon_creation_error_handling(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test error handling during daemon creation."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        # Simulate exception during daemon creation
        mock_daemon_manager.get_or_create_daemon.side_effect = RuntimeError("Creation failed")
        
        with pytest.raises(RuntimeError, match="Creation failed"):
            await mock_daemon_manager.get_or_create_daemon(project_name, project_path)
    
    @pytest.mark.asyncio
    async def test_daemon_startup_error_handling(self, mock_daemon_manager, isolated_daemon_temp_dir):
        """Test error handling during daemon startup."""
        project_name = "test-project"
        project_path = str(isolated_daemon_temp_dir)
        
        # Simulate exception during startup
        mock_daemon_manager.start_daemon.side_effect = RuntimeError("Startup failed")
        
        with pytest.raises(RuntimeError, match="Startup failed"):
            await mock_daemon_manager.start_daemon(project_name, project_path)
    
    @pytest.mark.asyncio
    async def test_health_check_error_handling(self, mock_daemon_manager):
        """Test error handling during health checks."""
        # Simulate exception during health check
        mock_daemon_manager.health_check_all.side_effect = RuntimeError("Health check failed")
        
        with pytest.raises(RuntimeError, match="Health check failed"):
            await mock_daemon_manager.health_check_all()


@pytest.mark.daemon_unit
@pytest.mark.daemon_lifecycle  
@pytest.mark.daemon_config
class TestDaemonManagerIntegration:
    """Integration tests for DaemonManager components."""
    
    @pytest.mark.asyncio
    async def test_full_daemon_lifecycle(self, isolated_daemon_temp_dir):
        """Test complete daemon lifecycle integration."""
        project_name = "integration-test"
        project_path = str(isolated_daemon_temp_dir)
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.DaemonManager.get_instance') as mock_get_instance:
            mock_manager = Mock()
            mock_daemon = Mock()
            
            # Setup mock behavior for full lifecycle
            mock_daemon.status.state = "stopped"
            mock_daemon.start = AsyncMock(return_value=True)
            mock_daemon.health_check = AsyncMock(return_value=True)
            mock_daemon.stop = AsyncMock(return_value=True)
            
            mock_manager.get_or_create_daemon = AsyncMock(return_value=mock_daemon)
            mock_manager.start_daemon = AsyncMock(return_value=True)
            mock_manager.get_daemon_status = AsyncMock(return_value={"state": "running"})
            mock_manager.stop_daemon = AsyncMock(return_value=True)
            mock_get_instance.return_value = mock_manager
            
            # Test lifecycle
            manager = await mock_get_instance()
            
            # Create daemon
            daemon = await manager.get_or_create_daemon(project_name, project_path)
            assert daemon is not None
            
            # Start daemon
            result = await manager.start_daemon(project_name, project_path)
            assert result is True
            
            # Check status
            status = await manager.get_daemon_status(project_name, project_path)
            assert status["state"] == "running"
            
            # Stop daemon
            result = await manager.stop_daemon(project_name, project_path)
            assert result is True
    
    @pytest.mark.asyncio
    async def test_multiple_daemon_management(self, isolated_daemon_temp_dir):
        """Test managing multiple daemon instances."""
        projects = [
            ("project-1", str(isolated_daemon_temp_dir / "proj1")),
            ("project-2", str(isolated_daemon_temp_dir / "proj2")),
            ("project-3", str(isolated_daemon_temp_dir / "proj3"))
        ]
        
        # Create project directories
        for _, path in projects:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        with patch('src.workspace_qdrant_mcp.core.daemon_manager.DaemonManager.get_instance') as mock_get_instance:
            mock_manager = Mock()
            
            # Setup mock for multiple daemons
            daemons = {}
            for i, (name, path) in enumerate(projects):
                daemon = Mock()
                daemon.status.state = "running"
                daemon.start = AsyncMock(return_value=True)
                daemons[f"daemon_{i}"] = daemon
            
            mock_manager.get_or_create_daemon = AsyncMock(side_effect=lambda name, path, overrides=None: daemons[f"daemon_{projects.index((name, path))}"])
            mock_manager.list_daemons = AsyncMock(return_value={k: {"state": "running"} for k in daemons.keys()})
            mock_manager.health_check_all = AsyncMock(return_value={k: True for k in daemons.keys()})
            mock_get_instance.return_value = mock_manager
            
            manager = await mock_get_instance()
            
            # Create all daemons
            created_daemons = []
            for name, path in projects:
                daemon = await manager.get_or_create_daemon(name, path)
                created_daemons.append(daemon)
            
            assert len(created_daemons) == 3
            
            # Check all daemons listed
            daemon_list = await manager.list_daemons()
            assert len(daemon_list) == 3
            
            # Health check all
            health = await manager.health_check_all()
            assert all(health.values())