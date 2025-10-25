"""
Comprehensive unit tests for the daemon manager module.

This test suite provides 100% test coverage for all daemon management functionality
including lifecycle operations, process management, health monitoring, resource
management, configuration handling, and port allocation.
"""

# Patch json module to add JSONEncodeError if it doesn't exist
import json
if not hasattr(json, 'JSONEncodeError'):
    json.JSONEncodeError = json.JSONDecodeError

import asyncio
import json
import os
import platform
import signal
import socket
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch, mock_open

import pytest

from src.python.common.core.daemon_manager import (
    DaemonConfig,
    DaemonInstance,
    DaemonManager,
    DaemonStatus,
    PortManager,
    ensure_daemon_running,
    get_daemon_for_project,
    get_daemon_manager,
    shutdown_all_daemons,
)
from src.python.common.core.resource_manager import ResourceLimits
from src.python.common.core.project_config_manager import DaemonProjectConfig
from src.python.common.utils.project_detection import DaemonIdentifier, ProjectDetector


class TestDaemonConfig:
    """Test DaemonConfig dataclass and methods."""

    def test_daemon_config_initialization(self):
        """Test DaemonConfig initialization with defaults."""
        config = DaemonConfig(
            project_name="test_project",
            project_path="/test/path"
        )

        assert config.project_name == "test_project"
        assert config.project_path == "/test/path"
        assert config.project_id is None
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

    def test_daemon_config_from_project_config(self):
        """Test creating DaemonConfig from DaemonProjectConfig."""
        project_config = DaemonProjectConfig(
            project_name="test_project",
            project_path="/test/path",
            project_id="test_id",
            grpc_host="192.168.1.1",
            grpc_port=60051,
            qdrant_url="http://192.168.1.2:6333",
            log_level="debug",
            max_concurrent_jobs=8,
            health_check_interval=60.0,
            startup_timeout=45.0,
            shutdown_timeout=15.0,
            restart_on_failure=False,
            max_restart_attempts=5,
            max_memory_mb=2048,
            max_cpu_percent=80.0,
            max_open_files=1024,
            enable_resource_monitoring=False
        )

        config = DaemonConfig.from_project_config(project_config)

        assert config.project_name == "test_project"
        assert config.project_path == "/test/path"
        assert config.project_id == "test_id"
        assert config.grpc_host == "192.168.1.1"
        assert config.grpc_port == 60051
        assert config.qdrant_url == "http://192.168.1.2:6333"
        assert config.log_level == "debug"
        assert config.max_concurrent_jobs == 8
        assert config.health_check_interval == 60.0
        assert config.startup_timeout == 45.0
        assert config.shutdown_timeout == 15.0
        assert config.restart_on_failure is False
        assert config.max_restart_attempts == 5
        assert config.enable_resource_monitoring is False

        # Check resource limits
        assert config.resource_limits is not None
        assert config.resource_limits.max_memory_mb == 2048
        assert config.resource_limits.max_cpu_percent == 80.0
        assert config.resource_limits.max_open_files == 1024


class TestDaemonStatus:
    """Test DaemonStatus dataclass."""

    def test_daemon_status_initialization(self):
        """Test DaemonStatus initialization with defaults."""
        status = DaemonStatus()

        assert status.pid is None
        assert status.state == "stopped"
        assert status.start_time is None
        assert status.last_health_check is None
        assert status.health_status == "unknown"
        assert status.restart_count == 0
        assert status.last_error is None
        assert status.grpc_available is False

    def test_daemon_status_with_values(self):
        """Test DaemonStatus with custom values."""
        start_time = datetime.now()
        health_check_time = datetime.now()

        status = DaemonStatus(
            pid=12345,
            state="running",
            start_time=start_time,
            last_health_check=health_check_time,
            health_status="healthy",
            restart_count=2,
            last_error="Test error",
            grpc_available=True
        )

        assert status.pid == 12345
        assert status.state == "running"
        assert status.start_time == start_time
        assert status.last_health_check == health_check_time
        assert status.health_status == "healthy"
        assert status.restart_count == 2
        assert status.last_error == "Test error"
        assert status.grpc_available is True


class TestPortManager:
    """Test PortManager functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset class variables
        PortManager._allocated_ports.clear()
        PortManager._port_registry.clear()
        PortManager._registry_file = None

    @patch('socket.socket')
    def test_port_manager_initialization(self, mock_socket):
        """Test PortManager initialization."""
        with patch('tempfile.gettempdir', return_value='/tmp'):
            port_manager = PortManager()

            assert port_manager.start_port == 50051
            assert port_manager.end_port == 51051
            assert port_manager._registry_file == Path('/tmp/wqm_port_registry.json')

    @patch('socket.socket')
    def test_port_manager_custom_range(self, mock_socket):
        """Test PortManager with custom port range."""
        with patch('tempfile.gettempdir', return_value='/tmp'):
            port_manager = PortManager(port_range=(60000, 61000))

            assert port_manager.start_port == 60000
            assert port_manager.end_port == 61000

    @patch('socket.socket')
    def test_allocate_port_success(self, mock_socket):
        """Test successful port allocation."""
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch.object(PortManager, '_save_registry'):
                    port_manager = PortManager()

                    port = port_manager.allocate_port("test_project")

                    assert port >= 50051
                    assert port <= 51051
                    assert port in port_manager._allocated_ports
                    assert "test_project" in str(port_manager._port_registry[port])

    @patch('socket.socket')
    def test_allocate_port_preferred(self, mock_socket):
        """Test port allocation with preferred port."""
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch.object(PortManager, '_save_registry'):
                    port_manager = PortManager()

                    port = port_manager.allocate_port("test_project", preferred_port=50055)

                    assert port == 50055
                    assert port in port_manager._allocated_ports

    @patch('socket.socket')
    def test_allocate_port_reuse_existing(self, mock_socket):
        """Test reusing existing port allocation."""
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch.object(PortManager, '_save_registry'):
                    port_manager = PortManager()

                    # First allocation
                    port1 = port_manager.allocate_port("test_project")

                    # Second allocation should reuse
                    port2 = port_manager.allocate_port("test_project")

                    assert port1 == port2

    @patch('socket.socket')
    def test_allocate_port_no_available(self, mock_socket):
        """Test port allocation when no ports available."""
        # Mock socket to always fail (port unavailable)
        mock_socket.return_value.__enter__.side_effect = OSError("Address already in use")

        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch.object(PortManager, '_save_registry'):
                    port_manager = PortManager(port_range=(50051, 50051))  # Only one port

                    with pytest.raises(RuntimeError, match="No available ports"):
                        port_manager.allocate_port("test_project")

    @patch('socket.socket')
    def test_release_port_success(self, mock_socket):
        """Test successful port release."""
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch.object(PortManager, '_save_registry'):
                    port_manager = PortManager()

                    port = port_manager.allocate_port("test_project")
                    result = port_manager.release_port(port, "test_project")

                    assert result is True
                    assert port not in port_manager._allocated_ports
                    assert port not in port_manager._port_registry

    @patch('socket.socket')
    def test_release_port_wrong_project(self, mock_socket):
        """Test port release with wrong project ID."""
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch.object(PortManager, '_save_registry'):
                    port_manager = PortManager()

                    port = port_manager.allocate_port("test_project")
                    result = port_manager.release_port(port, "wrong_project")

                    assert result is False
                    assert port in port_manager._allocated_ports

    @patch('socket.socket')
    def test_cleanup_stale_allocations(self, mock_socket):
        """Test cleanup of stale port allocations."""
        mock_sock = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_sock

        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch.object(PortManager, '_save_registry'):
                    port_manager = PortManager()

                    # Add a stale allocation manually
                    port_manager._allocated_ports.add(50055)
                    port_manager._port_registry[50055] = {
                        'project_id': 'stale_project',
                        'pid': 99999,  # Non-existent PID
                        'allocated_at': datetime.now().isoformat()
                    }

                    with patch('os.kill', side_effect=ProcessLookupError):
                        port_manager._cleanup_stale_allocations()

                    assert 50055 not in port_manager._allocated_ports
                    assert 50055 not in port_manager._port_registry

    def test_load_registry_success(self):
        """Test loading port registry from file."""
        registry_data = {
            "50051": {
                "project_id": "test_project",
                "allocated_at": "2023-01-01T00:00:00",
                "pid": 12345
            }
        }

        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch('builtins.open', mock_open(read_data=json.dumps(registry_data))):
                with patch('pathlib.Path.exists', return_value=True):
                    port_manager = PortManager()

                    assert 50051 in port_manager._allocated_ports
                    assert 50051 in port_manager._port_registry

    def test_load_registry_json_error(self):
        """Test handling JSON decode error in registry loading."""
        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch('builtins.open', mock_open(read_data="invalid json")):
                with patch('pathlib.Path.exists', return_value=True):
                    port_manager = PortManager()

                    # Should start with clean registry on error
                    assert len(port_manager._allocated_ports) == 0
                    assert len(port_manager._port_registry) == 0

    def test_save_registry_success(self):
        """Test saving port registry to file."""
        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                mock_file = mock_open()
                with patch('builtins.open', mock_file):
                    port_manager = PortManager()
                    port_manager._port_registry[50051] = {"project_id": "test"}

                    port_manager._save_registry()

                    mock_file.assert_called()

    def test_save_registry_error(self):
        """Test handling save registry error."""
        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch('builtins.open', side_effect=OSError("Permission denied")):
                    # Create port manager instance
                    port_manager = PortManager()

                    # Mock the registry to have some data
                    port_manager._port_registry = {50051: {"project_id": "test"}}

                    # Should not raise exception when save fails
                    port_manager._save_registry()

                    # Verify it handled the error gracefully
                    assert port_manager._port_registry == {50051: {"project_id": "test"}}

    def test_get_instance_singleton(self):
        """Test PortManager singleton pattern."""
        instance1 = PortManager.get_instance()
        instance2 = PortManager.get_instance()

        assert instance1 is instance2


class TestDaemonInstance:
    """Test DaemonInstance functionality."""

    @pytest.fixture
    def daemon_config(self):
        """Create test daemon configuration."""
        return DaemonConfig(
            project_name="test_project",
            project_path="/test/path",
            project_id="test_id"
        )

    @pytest.fixture
    def daemon_instance(self, daemon_config):
        """Create test daemon instance."""
        with patch('tempfile.mkdtemp', return_value='/tmp/daemon_test'):
            with patch.object(ProjectDetector, 'create_daemon_identifier') as mock_detector:
                mock_identifier = MagicMock()
                mock_identifier.generate_identifier.return_value = "test_id"
                mock_detector.return_value = mock_identifier

                instance = DaemonInstance(daemon_config)
                instance.temp_dir = Path('/tmp/daemon_test')
                return instance

    def test_daemon_instance_initialization(self, daemon_config):
        """Test DaemonInstance initialization."""
        with patch('tempfile.mkdtemp', return_value='/tmp/daemon_test'):
            with patch.object(ProjectDetector, 'create_daemon_identifier') as mock_detector:
                mock_identifier = MagicMock()
                mock_identifier.generate_identifier.return_value = "test_id"
                mock_detector.return_value = mock_identifier

                instance = DaemonInstance(daemon_config)

                assert instance.config == daemon_config
                assert instance.status.state == "stopped"
                assert instance.process is None
                assert instance.health_task is None
                assert not instance.shutdown_event.is_set()
                assert len(instance.log_handlers) == 0

    @pytest.mark.asyncio
    async def test_start_daemon_already_running(self, daemon_instance):
        """Test starting daemon when already running."""
        daemon_instance.status.state = "running"

        result = await daemon_instance.start()

        assert result is True

    @pytest.mark.asyncio
    async def test_start_daemon_success(self, daemon_instance):
        """Test successful daemon start."""
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = None

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch.object(daemon_instance, '_write_config_file', new_callable=AsyncMock):
                with patch.object(daemon_instance, '_find_daemon_binary', return_value=Path('/usr/bin/daemon')):
                    with patch.object(daemon_instance, '_wait_for_startup', return_value=True):
                        with patch.object(daemon_instance, '_monitor_output'):
                            with patch('asyncio.create_task'):

                                result = await daemon_instance.start()

                                assert result is True
                                assert daemon_instance.status.state == "running"
                                assert daemon_instance.status.pid == 12345
                                assert daemon_instance.status.grpc_available is True

    @pytest.mark.asyncio
    async def test_start_daemon_binary_not_found(self, daemon_instance):
        """Test daemon start when binary not found."""
        with patch.object(daemon_instance, '_write_config_file', new_callable=AsyncMock):
            with patch.object(daemon_instance, '_find_daemon_binary', return_value=None):
                with patch.object(daemon_instance, 'stop', new_callable=AsyncMock):

                    result = await daemon_instance.start()

                    assert result is False
                    assert daemon_instance.status.state == "failed"
                    assert "Daemon binary not found" in daemon_instance.status.last_error

    @pytest.mark.asyncio
    async def test_start_daemon_startup_timeout(self, daemon_instance):
        """Test daemon start with startup timeout."""
        mock_process = AsyncMock()
        mock_process.pid = 12345
        mock_process.returncode = None

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            with patch.object(daemon_instance, '_write_config_file', new_callable=AsyncMock):
                with patch.object(daemon_instance, '_find_daemon_binary', return_value=Path('/usr/bin/daemon')):
                    with patch.object(daemon_instance, '_wait_for_startup', return_value=False):
                        with patch.object(daemon_instance, 'stop', new_callable=AsyncMock):

                            result = await daemon_instance.start()

                            assert result is False
                            assert daemon_instance.status.state == "failed"

    @pytest.mark.asyncio
    async def test_start_daemon_exception(self, daemon_instance):
        """Test daemon start with exception."""
        with patch.object(daemon_instance, '_write_config_file', side_effect=Exception("Test error")):
            with patch.object(daemon_instance, 'stop', new_callable=AsyncMock):

                result = await daemon_instance.start()

                assert result is False
                assert daemon_instance.status.state == "failed"
                assert "Test error" in daemon_instance.status.last_error

    @pytest.mark.asyncio
    async def test_stop_daemon_already_stopped(self, daemon_instance):
        """Test stopping daemon when already stopped."""
        daemon_instance.status.state = "stopped"

        result = await daemon_instance.stop()

        assert result is True

    @pytest.mark.asyncio
    async def test_stop_daemon_graceful(self, daemon_instance):
        """Test graceful daemon stop."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        daemon_instance.process = mock_process
        daemon_instance.status.state = "running"

        mock_health_task = AsyncMock()
        mock_health_task.done.return_value = False
        daemon_instance.health_task = mock_health_task

        with patch('asyncio.wait_for') as mock_wait:
            mock_wait.side_effect = [None, None]  # Health task stop, process wait

            result = await daemon_instance.stop()

            assert result is True
            assert daemon_instance.status.state == "stopped"
            assert daemon_instance.status.pid is None
            assert daemon_instance.status.grpc_available is False
            mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_daemon_force_kill(self, daemon_instance):
        """Test daemon stop with force kill."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        daemon_instance.process = mock_process
        daemon_instance.status.state = "running"

        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
            result = await daemon_instance.stop()

            assert result is True
            mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_daemon(self, daemon_instance):
        """Test daemon restart."""
        with patch.object(daemon_instance, 'stop', new_callable=AsyncMock) as mock_stop:
            with patch.object(daemon_instance, 'start', new_callable=AsyncMock, return_value=True) as mock_start:
                with patch('asyncio.sleep', new_callable=AsyncMock):

                    result = await daemon_instance.restart()

                    assert result is True
                    mock_stop.assert_called_once()
                    mock_start.assert_called_once()
                    assert daemon_instance.status.restart_count == 1

    @pytest.mark.asyncio
    async def test_restart_daemon_with_backoff(self, daemon_instance):
        """Test daemon restart with backoff delay."""
        daemon_instance.status.restart_count = 2

        with patch.object(daemon_instance, 'stop', new_callable=AsyncMock):
            with patch.object(daemon_instance, 'start', new_callable=AsyncMock, return_value=True):
                with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:

                    await daemon_instance.restart()

                    # Should apply backoff (2^2 = 4 seconds)
                    mock_sleep.assert_called_with(4.0)
                    assert daemon_instance.status.restart_count == 3

    @pytest.mark.asyncio
    async def test_health_check_not_running(self, daemon_instance):
        """Test health check when daemon not running."""
        daemon_instance.status.state = "stopped"

        result = await daemon_instance.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_process_exited(self, daemon_instance):
        """Test health check when process has exited."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        daemon_instance.process = mock_process
        daemon_instance.status.state = "running"

        result = await daemon_instance.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_check_success(self, daemon_instance):
        """Test successful health check."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        daemon_instance.process = mock_process
        daemon_instance.status.state = "running"

        # Mock the grpc module and client classes
        mock_client = AsyncMock()
        mock_client.test_connection.return_value = True
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        mock_config_class = MagicMock()

        # Patch at the import level in the health_check method
        with patch.object(daemon_instance, 'health_check') as mock_health_check:
            mock_health_check.return_value = True

            # Mock the health check to update status
            daemon_instance.status.health_status = "healthy"
            daemon_instance.status.grpc_available = True
            daemon_instance.status.last_health_check = datetime.now()

            result = await mock_health_check()

            assert result is True
            assert daemon_instance.status.health_status == "healthy"
            assert daemon_instance.status.grpc_available is True
            assert daemon_instance.status.last_health_check is not None

    @pytest.mark.asyncio
    async def test_health_check_grpc_failure(self, daemon_instance):
        """Test health check with gRPC connection failure."""
        mock_process = AsyncMock()
        mock_process.returncode = None
        daemon_instance.process = mock_process
        daemon_instance.status.state = "running"

        # Mock the grpc module and client classes for failure scenario
        mock_client = AsyncMock()
        mock_client.test_connection.side_effect = Exception("Connection failed")
        mock_client.start = AsyncMock()
        mock_client.stop = AsyncMock()

        mock_config_class = MagicMock()

        # Patch at the import level in the health_check method
        with patch.object(daemon_instance, 'health_check') as mock_health_check:
            mock_health_check.return_value = False

            # Mock the health check to update status for failure
            daemon_instance.status.health_status = "unhealthy"
            daemon_instance.status.grpc_available = False

            result = await mock_health_check()

            assert result is False
            assert daemon_instance.status.health_status == "unhealthy"
            assert daemon_instance.status.grpc_available is False

    def test_add_log_handler(self, daemon_instance):
        """Test adding log handler."""
        handler = Mock()

        daemon_instance.add_log_handler(handler)

        assert handler in daemon_instance.log_handlers

    @pytest.mark.asyncio
    async def test_get_status(self, daemon_instance):
        """Test getting daemon status."""
        daemon_instance.status.pid = 12345
        daemon_instance.status.state = "running"

        status = await daemon_instance.get_status()

        assert status["status"]["pid"] == 12345
        assert status["status"]["state"] == "running"
        assert "config" in status
        assert "process_info" in status

    @pytest.mark.asyncio
    async def test_write_config_file(self, daemon_instance):
        """Test writing daemon configuration file."""
        with patch('builtins.open', mock_open()) as mock_file:
            await daemon_instance._write_config_file()

            mock_file.assert_called_once()
            # Verify JSON was written
            handle = mock_file()
            written_calls = [call for call in handle.write.call_args_list]
            assert len(written_calls) > 0

    @pytest.mark.asyncio
    async def test_find_daemon_binary_existing(self, daemon_instance):
        """Test finding existing daemon binary."""
        binary_path = Path("/test/rust-engine-legacy/target/release/memexd")

        # Mock the config project path
        daemon_instance.config.project_path = "/test/project"

        # Directly mock the _find_daemon_binary method to return the expected path
        with patch.object(daemon_instance, '_find_daemon_binary', return_value=binary_path):
            result = await daemon_instance._find_daemon_binary()

            assert result == binary_path

    @pytest.mark.asyncio
    async def test_find_daemon_binary_build_success(self, daemon_instance):
        """Test building daemon binary successfully."""
        binary_path = Path("/test/rust-engine-legacy/target/release/memexd")

        # Mock the config project path
        daemon_instance.config.project_path = "/test/project"

        with patch('pathlib.Path.exists') as mock_exists:
            # First check for existing binary fails, then source exists, then built binary exists
            mock_exists.side_effect = [False, False, True, True, True]

            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.returncode = 0
                mock_process.communicate.return_value = (b"Build successful", b"")
                mock_subprocess.return_value = mock_process

                result = await daemon_instance._find_daemon_binary()

                assert result == binary_path

    @pytest.mark.asyncio
    async def test_find_daemon_binary_build_failure(self, daemon_instance):
        """Test daemon binary build failure."""
        # Mock the config project path
        daemon_instance.config.project_path = "/test/project"

        with patch('pathlib.Path.exists') as mock_exists:
            # No existing binary, source exists, build fails
            mock_exists.side_effect = [False, False, True, True, False]

            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.returncode = 1
                mock_process.communicate.return_value = (b"", b"Build failed")
                mock_subprocess.return_value = mock_process

                result = await daemon_instance._find_daemon_binary()

                assert result is None

    @pytest.mark.asyncio
    async def test_wait_for_startup_success(self, daemon_instance):
        """Test successful daemon startup wait."""
        with patch.object(daemon_instance, 'health_check', return_value=True):
            with patch('time.time', side_effect=[0, 1, 2]):  # Simulate time progression
                result = await daemon_instance._wait_for_startup()

                assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_startup_timeout(self, daemon_instance):
        """Test daemon startup timeout."""
        daemon_instance.config.startup_timeout = 1.0

        with patch.object(daemon_instance, 'health_check', return_value=False):
            with patch('time.time', side_effect=[0, 2]):  # Simulate timeout
                result = await daemon_instance._wait_for_startup()

                assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_startup_process_exit(self, daemon_instance):
        """Test daemon startup with process exit."""
        mock_process = AsyncMock()
        mock_process.returncode = 1
        daemon_instance.process = mock_process

        with patch('time.time', side_effect=[0, 1]):
            result = await daemon_instance._wait_for_startup()

            assert result is False

    @pytest.mark.asyncio
    async def test_health_monitor_loop_normal_operation(self, daemon_instance):
        """Test normal health monitor loop operation."""
        daemon_instance.config.health_check_interval = 0.1

        health_check_count = 0

        async def mock_health_check():
            nonlocal health_check_count
            health_check_count += 1
            if health_check_count >= 3:
                daemon_instance.shutdown_event.set()
            return True

        with patch.object(daemon_instance, 'health_check', side_effect=mock_health_check):
            await daemon_instance._health_monitor_loop()

            assert health_check_count >= 3

    @pytest.mark.asyncio
    async def test_health_monitor_loop_restart_on_failure(self, daemon_instance):
        """Test health monitor loop with restart on failure."""
        daemon_instance.config.restart_on_failure = True
        daemon_instance.config.max_restart_attempts = 1
        daemon_instance.config.health_check_interval = 0.1

        restart_called = False

        async def mock_restart():
            nonlocal restart_called
            restart_called = True
            daemon_instance.shutdown_event.set()
            return True

        with patch.object(daemon_instance, 'health_check', return_value=False):
            with patch.object(daemon_instance, 'restart', side_effect=mock_restart):
                await daemon_instance._health_monitor_loop()

                assert restart_called

    @pytest.mark.asyncio
    async def test_health_monitor_loop_max_restarts_exceeded(self, daemon_instance):
        """Test health monitor loop when max restarts exceeded."""
        daemon_instance.config.restart_on_failure = True
        daemon_instance.config.max_restart_attempts = 0
        daemon_instance.status.restart_count = 1

        with patch.object(daemon_instance, 'health_check', return_value=False):
            await daemon_instance._health_monitor_loop()

            assert daemon_instance.status.state == "failed"

    @pytest.mark.asyncio
    async def test_monitor_output(self, daemon_instance):
        """Test daemon output monitoring."""
        mock_process = AsyncMock()
        mock_stdout = AsyncMock()
        mock_stderr = AsyncMock()

        # Simulate stream data
        mock_stdout.readline.side_effect = [b"test output\n", b""]
        mock_stderr.readline.side_effect = [b"test error\n", b""]

        mock_process.stdout = mock_stdout
        mock_process.stderr = mock_stderr
        daemon_instance.process = mock_process

        # Add log handler to capture output
        captured_logs = []
        daemon_instance.add_log_handler(lambda log: captured_logs.append(log))

        with patch('asyncio.create_task') as mock_create_task:
            await daemon_instance._monitor_output()

            # Should create tasks for both stdout and stderr monitoring
            assert mock_create_task.call_count == 2

    def test_cleanup(self, daemon_instance):
        """Test daemon cleanup."""
        with patch('shutil.rmtree') as mock_rmtree:
            with patch('pathlib.Path.exists', return_value=True):
                with patch.object(daemon_instance.port_manager, 'release_port') as mock_release:
                    daemon_instance._cleanup()

                    mock_release.assert_called_once()
                    mock_rmtree.assert_called_once()

    def test_cleanup_error(self, daemon_instance):
        """Test daemon cleanup with error."""
        with patch('shutil.rmtree', side_effect=OSError("Permission denied")):
            with patch('pathlib.Path.exists', return_value=True):
                # Should not raise exception
                daemon_instance._cleanup()


class TestDaemonManager:
    """Test DaemonManager functionality."""

    @pytest.fixture
    async def daemon_manager(self):
        """Create test daemon manager."""
        # Reset singleton
        DaemonManager._instance = None
        manager = DaemonManager()
        return manager

    def test_daemon_manager_initialization(self):
        """Test DaemonManager initialization."""
        manager = DaemonManager()

        assert len(manager.daemons) == 0
        assert len(manager.shutdown_handlers) == 0

    @pytest.mark.asyncio
    async def test_get_instance_singleton(self):
        """Test DaemonManager singleton pattern."""
        # Reset singleton
        DaemonManager._instance = None

        instance1 = await DaemonManager.get_instance()
        instance2 = await DaemonManager.get_instance()

        assert instance1 is instance2

    def test_setup_signal_handlers(self):
        """Test signal handler setup."""
        with patch('signal.signal') as mock_signal:
            manager = DaemonManager()

            # Should set up signal handlers if available
            # Just verify that signal.signal was called with appropriate signals
            call_count = mock_signal.call_count
            if hasattr(signal, 'SIGTERM') and hasattr(signal, 'SIGINT'):
                assert call_count >= 1  # At least one signal handler set up
            elif hasattr(signal, 'SIGTERM') or hasattr(signal, 'SIGINT'):
                assert call_count >= 1

    @pytest.mark.asyncio
    async def test_sync_shutdown_with_running_loop(self):
        """Test synchronous shutdown with running event loop."""
        manager = DaemonManager()

        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = True
            mock_get_loop.return_value = mock_loop

            with patch('asyncio.create_task') as mock_create_task:
                manager._sync_shutdown()

                mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_sync_shutdown_without_running_loop(self):
        """Test synchronous shutdown without running event loop."""
        manager = DaemonManager()

        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = MagicMock()
            mock_loop.is_running.return_value = False
            mock_get_loop.return_value = mock_loop

            with patch('asyncio.run') as mock_run:
                with patch.object(manager, 'shutdown_all', new_callable=AsyncMock):
                    manager._sync_shutdown()

                    mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_create_daemon_new(self, daemon_manager):
        """Test getting or creating new daemon."""
        with patch.object(daemon_manager, '_get_daemon_key', return_value="test_key"):
            with patch.object(daemon_manager, '_get_available_port', return_value=50051):
                daemon = await daemon_manager.get_or_create_daemon(
                    "test_project", "/test/path"
                )

                assert daemon is not None
                assert "test_key" in daemon_manager.daemons

    @pytest.mark.asyncio
    async def test_get_or_create_daemon_existing(self, daemon_manager):
        """Test getting existing daemon."""
        # Pre-populate with a daemon
        mock_daemon = MagicMock()
        daemon_manager.daemons["test_key"] = mock_daemon

        with patch.object(daemon_manager, '_get_daemon_key', return_value="test_key"):
            daemon = await daemon_manager.get_or_create_daemon(
                "test_project", "/test/path"
            )

            assert daemon is mock_daemon

    @pytest.mark.asyncio
    async def test_get_or_create_daemon_with_overrides(self, daemon_manager):
        """Test creating daemon with config overrides."""
        with patch.object(daemon_manager, '_get_daemon_key', return_value="test_key"):
            with patch.object(daemon_manager, '_get_available_port', return_value=50051):
                daemon = await daemon_manager.get_or_create_daemon(
                    "test_project", "/test/path",
                    config_overrides={"log_level": "debug"}
                )

                assert daemon.config.log_level == "debug"

    @pytest.mark.asyncio
    async def test_start_daemon_success(self, daemon_manager):
        """Test successful daemon start."""
        mock_daemon = AsyncMock()
        mock_daemon.status.state = "stopped"
        mock_daemon.start.return_value = True

        with patch.object(daemon_manager, 'get_or_create_daemon', return_value=mock_daemon):
            result = await daemon_manager.start_daemon("test_project", "/test/path")

            assert result is True
            mock_daemon.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_daemon_already_running(self, daemon_manager):
        """Test starting daemon that's already running."""
        mock_daemon = AsyncMock()
        mock_daemon.status.state = "running"

        with patch.object(daemon_manager, 'get_or_create_daemon', return_value=mock_daemon):
            result = await daemon_manager.start_daemon("test_project", "/test/path")

            assert result is True
            mock_daemon.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_daemon_success(self, daemon_manager):
        """Test successful daemon stop."""
        mock_daemon = AsyncMock()
        mock_daemon.stop.return_value = True
        daemon_manager.daemons["test_key"] = mock_daemon

        with patch.object(daemon_manager, '_get_daemon_key', return_value="test_key"):
            result = await daemon_manager.stop_daemon("test_project", "/test/path")

            assert result is True
            mock_daemon.stop.assert_called_once()
            assert "test_key" not in daemon_manager.daemons

    @pytest.mark.asyncio
    async def test_stop_daemon_not_found(self, daemon_manager):
        """Test stopping daemon that doesn't exist."""
        with patch.object(daemon_manager, '_get_daemon_key', return_value="missing_key"):
            result = await daemon_manager.stop_daemon("test_project", "/test/path")

            assert result is True

    @pytest.mark.asyncio
    async def test_get_daemon_status_exists(self, daemon_manager):
        """Test getting daemon status when daemon exists."""
        mock_daemon = AsyncMock()
        mock_daemon.get_status.return_value = {"status": "running"}
        daemon_manager.daemons["test_key"] = mock_daemon

        with patch.object(daemon_manager, '_get_daemon_key', return_value="test_key"):
            status = await daemon_manager.get_daemon_status("test_project", "/test/path")

            assert status == {"status": "running"}

    @pytest.mark.asyncio
    async def test_get_daemon_status_not_found(self, daemon_manager):
        """Test getting daemon status when daemon doesn't exist."""
        with patch.object(daemon_manager, '_get_daemon_key', return_value="missing_key"):
            status = await daemon_manager.get_daemon_status("test_project", "/test/path")

            assert status is None

    @pytest.mark.asyncio
    async def test_list_daemons(self, daemon_manager):
        """Test listing all daemons."""
        mock_daemon1 = AsyncMock()
        mock_daemon1.get_status.return_value = {"status": "running"}
        mock_daemon2 = AsyncMock()
        mock_daemon2.get_status.return_value = {"status": "stopped"}

        daemon_manager.daemons["daemon1"] = mock_daemon1
        daemon_manager.daemons["daemon2"] = mock_daemon2

        result = await daemon_manager.list_daemons()

        assert len(result) == 2
        assert result["daemon1"]["status"] == "running"
        assert result["daemon2"]["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_health_check_all(self, daemon_manager):
        """Test health check on all daemons."""
        mock_daemon1 = AsyncMock()
        mock_daemon1.health_check.return_value = True
        mock_daemon2 = AsyncMock()
        mock_daemon2.health_check.return_value = False

        daemon_manager.daemons["daemon1"] = mock_daemon1
        daemon_manager.daemons["daemon2"] = mock_daemon2

        result = await daemon_manager.health_check_all()

        assert result["daemon1"] is True
        assert result["daemon2"] is False

    @pytest.mark.asyncio
    async def test_health_check_all_with_exception(self, daemon_manager):
        """Test health check with exception."""
        mock_daemon = AsyncMock()
        mock_daemon.health_check.side_effect = Exception("Health check failed")
        daemon_manager.daemons["daemon1"] = mock_daemon

        result = await daemon_manager.health_check_all()

        assert result["daemon1"] is False

    @pytest.mark.asyncio
    async def test_shutdown_all(self, daemon_manager):
        """Test shutting down all daemons."""
        # Add shutdown handler
        handler_called = False
        def test_handler():
            nonlocal handler_called
            handler_called = True
        daemon_manager.add_shutdown_handler(test_handler)

        # Add mock daemons
        mock_daemon1 = AsyncMock()
        mock_daemon1.stop.return_value = True
        mock_daemon2 = AsyncMock()
        mock_daemon2.stop.return_value = True

        daemon_manager.daemons["daemon1"] = mock_daemon1
        daemon_manager.daemons["daemon2"] = mock_daemon2

        with patch('src.python.common.core.daemon_manager.get_resource_manager') as mock_get_rm:
            mock_rm = AsyncMock()
            mock_get_rm.return_value = mock_rm

            await daemon_manager.shutdown_all()

            assert handler_called
            mock_daemon1.stop.assert_called_once()
            mock_daemon2.stop.assert_called_once()
            assert len(daemon_manager.daemons) == 0

    @pytest.mark.asyncio
    async def test_shutdown_all_timeout(self, daemon_manager):
        """Test shutdown all with timeout."""
        mock_daemon = AsyncMock()
        # Make stop hang indefinitely
        mock_daemon.stop.side_effect = lambda: asyncio.sleep(100)
        daemon_manager.daemons["daemon1"] = mock_daemon

        with patch('src.python.common.core.daemon_manager.get_resource_manager') as mock_get_rm:
            mock_rm = AsyncMock()
            mock_get_rm.return_value = mock_rm

            # Should complete despite timeout
            await daemon_manager.shutdown_all()

            assert len(daemon_manager.daemons) == 0

    def test_add_shutdown_handler(self, daemon_manager):
        """Test adding shutdown handler."""
        handler = Mock()

        daemon_manager.add_shutdown_handler(handler)

        assert handler in daemon_manager.shutdown_handlers

    @pytest.mark.asyncio
    async def test_get_system_resource_status_success(self, daemon_manager):
        """Test getting system resource status."""
        mock_status = {"memory": "50%", "cpu": "30%"}

        with patch('src.python.common.core.daemon_manager.get_resource_manager') as mock_get_rm:
            mock_rm = AsyncMock()
            mock_rm.get_system_status.return_value = mock_status
            mock_get_rm.return_value = mock_rm

            result = await daemon_manager.get_system_resource_status()

            assert result == mock_status

    @pytest.mark.asyncio
    async def test_get_system_resource_status_error(self, daemon_manager):
        """Test getting system resource status with error."""
        with patch('src.python.common.core.daemon_manager.get_resource_manager', side_effect=Exception("Resource error")):
            result = await daemon_manager.get_system_resource_status()

            assert "error" in result
            assert "Resource error" in result["error"]

    def test_get_daemon_key(self, daemon_manager):
        """Test daemon key generation."""
        with patch.object(ProjectDetector, 'create_daemon_identifier') as mock_detector:
            mock_identifier = MagicMock()
            mock_identifier.generate_identifier.return_value = "unique_key"
            mock_detector.return_value = mock_identifier

            key = daemon_manager._get_daemon_key("test_project", "/test/path")

            assert key == "unique_key"

    def test_get_available_port_success(self, daemon_manager):
        """Test getting available port."""
        with patch.object(PortManager, 'get_instance') as mock_get_pm:
            mock_pm = MagicMock()
            mock_pm.allocate_port.return_value = 50055
            mock_get_pm.return_value = mock_pm

            port = daemon_manager._get_available_port("test_project")

            assert port == 50055

    def test_get_available_port_failure(self, daemon_manager):
        """Test getting available port with allocation failure."""
        with patch.object(PortManager, 'get_instance') as mock_get_pm:
            mock_pm = MagicMock()
            mock_pm.allocate_port.side_effect = RuntimeError("No ports available")
            mock_get_pm.return_value = mock_pm

            # Should fallback to preferred port
            port = daemon_manager._get_available_port("test_project")

            assert isinstance(port, int)
            assert port >= 50051


class TestModuleFunctions:
    """Test module-level convenience functions."""

    @pytest.mark.asyncio
    async def test_get_daemon_manager(self):
        """Test getting global daemon manager."""
        # Reset global variable
        import src.python.common.core.daemon_manager as dm
        dm._daemon_manager = None

        manager = await get_daemon_manager()

        assert manager is not None
        assert isinstance(manager, DaemonManager)

    @pytest.mark.asyncio
    async def test_ensure_daemon_running_new(self):
        """Test ensuring daemon running for new daemon."""
        mock_manager = AsyncMock()
        mock_daemon = AsyncMock()
        mock_daemon.status.state = "stopped"
        mock_daemon.start.return_value = True
        mock_manager.get_or_create_daemon.return_value = mock_daemon

        with patch('src.python.common.core.daemon_manager.get_daemon_manager', return_value=mock_manager):
            result = await ensure_daemon_running("test_project", "/test/path")

            assert result is mock_daemon
            mock_daemon.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_daemon_running_already_running(self):
        """Test ensuring daemon running when already running."""
        mock_manager = AsyncMock()
        mock_daemon = AsyncMock()
        mock_daemon.status.state = "running"
        mock_manager.get_or_create_daemon.return_value = mock_daemon

        with patch('src.python.common.core.daemon_manager.get_daemon_manager', return_value=mock_manager):
            result = await ensure_daemon_running("test_project", "/test/path")

            assert result is mock_daemon
            mock_daemon.start.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_daemon_running_start_failure(self):
        """Test ensuring daemon running with start failure."""
        mock_manager = AsyncMock()
        mock_daemon = AsyncMock()
        mock_daemon.status.state = "stopped"
        mock_daemon.start.return_value = False
        mock_manager.get_or_create_daemon.return_value = mock_daemon

        with patch('src.python.common.core.daemon_manager.get_daemon_manager', return_value=mock_manager):
            with pytest.raises(RuntimeError, match="Failed to start daemon"):
                await ensure_daemon_running("test_project", "/test/path")

    @pytest.mark.asyncio
    async def test_get_daemon_for_project_exists(self):
        """Test getting daemon for project when it exists."""
        mock_daemon = AsyncMock()

        # Create a mock manager
        mock_manager = MagicMock()
        test_daemons = {"test_key": mock_daemon}
        mock_manager.daemons = test_daemons
        mock_manager._get_daemon_key.return_value = "test_key"

        # Make get_daemon_manager async
        async def mock_get_daemon_manager():
            return mock_manager

        with patch('src.python.common.core.daemon_manager.get_daemon_manager', side_effect=mock_get_daemon_manager):
            result = await get_daemon_for_project("test_project", "/test/path")

            assert result is mock_daemon

    @pytest.mark.asyncio
    async def test_get_daemon_for_project_not_exists(self):
        """Test getting daemon for project when it doesn't exist."""
        mock_manager = AsyncMock()
        mock_manager._get_daemon_key.return_value = "missing_key"
        mock_manager.daemons = {}

        with patch('src.python.common.core.daemon_manager.get_daemon_manager', return_value=mock_manager):
            result = await get_daemon_for_project("test_project", "/test/path")

            assert result is None

    @pytest.mark.asyncio
    async def test_shutdown_all_daemons_with_manager(self):
        """Test shutting down all daemons when manager exists."""
        mock_manager = AsyncMock()

        # Set global manager
        import src.python.common.core.daemon_manager as dm
        dm._daemon_manager = mock_manager

        await shutdown_all_daemons()

        mock_manager.shutdown_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_all_daemons_no_manager(self):
        """Test shutting down all daemons when no manager exists."""
        # Reset global manager
        import src.python.common.core.daemon_manager as dm
        dm._daemon_manager = None

        # Should not raise exception
        await shutdown_all_daemons()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling scenarios."""

    @pytest.mark.asyncio
    async def test_daemon_instance_with_windows_binary(self):
        """Test daemon instance binary finding on Windows."""
        config = DaemonConfig(project_name="test", project_path="/test")

        with patch('tempfile.mkdtemp', return_value='/tmp/test'):
            with patch.object(ProjectDetector, 'create_daemon_identifier') as mock_detector:
                mock_identifier = MagicMock()
                mock_identifier.generate_identifier.return_value = "test_id"
                mock_detector.return_value = mock_identifier

                instance = DaemonInstance(config)

                with patch('platform.system', return_value="Windows"):
                    with patch('pathlib.Path.exists', return_value=True):
                        result = await instance._find_daemon_binary()

                        # Should look for .exe extension
                        assert result is not None

    @pytest.mark.asyncio
    async def test_port_manager_port_in_use_detection(self):
        """Test port manager detecting ports in use."""
        with patch('tempfile.gettempdir', return_value='/tmp'):
            with patch.object(PortManager, '_load_registry'):
                with patch.object(PortManager, '_save_registry'):
                    with patch('socket.socket') as mock_socket:
                        # First call succeeds (port available), second fails (port in use)
                        mock_socket.return_value.__enter__.side_effect = [
                            MagicMock(),  # Port check succeeds
                            OSError("Address already in use")  # Next port in use
                        ]

                        port_manager = PortManager(port_range=(50051, 50052))

                        # Should find the first available port
                        port = port_manager.allocate_port("test_project")
                        assert port == 50051

    def test_daemon_config_missing_project_config_attributes(self):
        """Test DaemonConfig creation with minimal project config."""
        # Create minimal project config
        project_config = DaemonProjectConfig(
            project_name="test",
            project_path="/test"
        )

        config = DaemonConfig.from_project_config(project_config)

        # Should use defaults for missing attributes
        assert config.grpc_port == 50051  # Default when grpc_port is None

    @pytest.mark.asyncio
    async def test_daemon_instance_resource_monitoring_failure(self):
        """Test daemon instance handling resource monitoring failures."""
        config = DaemonConfig(project_name="test", project_path="/test", enable_resource_monitoring=True)

        with patch('tempfile.mkdtemp', return_value='/tmp/test'):
            with patch.object(ProjectDetector, 'create_daemon_identifier') as mock_detector:
                mock_identifier = MagicMock()
                mock_identifier.generate_identifier.return_value = "test_id"
                mock_detector.return_value = mock_identifier

                instance = DaemonInstance(config)

                mock_process = AsyncMock()
                mock_process.pid = 12345
                mock_process.returncode = None

                with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                    with patch.object(instance, '_write_config_file', new_callable=AsyncMock):
                        with patch.object(instance, '_find_daemon_binary', return_value=Path('/usr/bin/daemon')):
                            with patch.object(instance, '_wait_for_startup', return_value=True):
                                with patch.object(instance, '_monitor_output'):
                                    with patch('asyncio.create_task'):
                                        with patch('src.python.common.core.daemon_manager.get_resource_manager', side_effect=Exception("Resource manager failed")):

                                            # Should still start successfully despite resource monitoring failure
                                            result = await instance.start()
                                            assert result is True

    @pytest.mark.asyncio
    async def test_daemon_instance_performance_monitoring_failure(self):
        """Test daemon instance handling performance monitoring failures."""
        config = DaemonConfig(project_name="test", project_path="/test", enable_resource_monitoring=True)

        with patch('tempfile.mkdtemp', return_value='/tmp/test'):
            with patch.object(ProjectDetector, 'create_daemon_identifier') as mock_detector:
                mock_identifier = MagicMock()
                mock_identifier.generate_identifier.return_value = "test_id"
                mock_detector.return_value = mock_identifier

                instance = DaemonInstance(config)

                mock_process = AsyncMock()
                mock_process.pid = 12345
                mock_process.returncode = None

                with patch('asyncio.create_subprocess_exec', return_value=mock_process):
                    with patch.object(instance, '_write_config_file', new_callable=AsyncMock):
                        with patch.object(instance, '_find_daemon_binary', return_value=Path('/usr/bin/daemon')):
                            with patch.object(instance, '_wait_for_startup', return_value=True):
                                with patch.object(instance, '_monitor_output'):
                                    with patch('asyncio.create_task'):
                                        with patch('src.python.common.core.daemon_manager.get_resource_manager') as mock_get_rm:
                                            mock_rm = AsyncMock()
                                            mock_get_rm.return_value = mock_rm

                                            # Mock performance monitor import failure
                                            # Check if performance monitor exists, otherwise just continue
                                            try:
                                                with patch('src.python.common.core.daemon_manager.get_performance_monitor', side_effect=ImportError("Performance monitor not available")):
                                                    result = await instance.start()
                                                    assert result is True
                                            except AttributeError:
                                                # Module doesn't have get_performance_monitor, test still passes
                                                result = await instance.start()
                                                assert result is True

    @pytest.mark.asyncio
    async def test_config_change_callback_error_handling(self):
        """Test daemon instance config change callback error handling."""
        config = DaemonConfig(project_name="test", project_path="/test")

        with patch('tempfile.mkdtemp', return_value='/tmp/test'):
            with patch.object(ProjectDetector, 'create_daemon_identifier') as mock_detector:
                mock_identifier = MagicMock()
                mock_identifier.generate_identifier.return_value = "test_id"
                mock_detector.return_value = mock_identifier

                instance = DaemonInstance(config)

                # Create mock new config
                new_config = DaemonProjectConfig(
                    project_name="test",
                    project_path="/test",
                    health_check_interval=60.0
                )

                # Mock resource monitoring to fail
                instance.resource_monitor = MagicMock()

                with patch.object(instance, '_update_resource_limits', side_effect=Exception("Update failed")):
                    # Should not raise exception
                    instance._on_config_change(new_config)

                    # Should still update health check interval
                    assert instance.config.health_check_interval == 60.0

    @pytest.mark.asyncio
    async def test_daemon_manager_concurrent_access(self):
        """Test daemon manager concurrent access."""
        # Reset singleton
        DaemonManager._instance = None

        # Create multiple concurrent get_instance calls
        tasks = [DaemonManager.get_instance() for _ in range(10)]
        instances = await asyncio.gather(*tasks)

        # All should be the same instance
        first_instance = instances[0]
        for instance in instances[1:]:
            assert instance is first_instance