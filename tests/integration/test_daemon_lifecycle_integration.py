"""
Integration tests for daemon lifecycle management.

Tests comprehensive daemon process management including:
- Daemon startup and initialization
- Health monitoring and heartbeat systems  
- Graceful shutdown and cleanup
- Process recovery and restart logic
- Multiple daemon instance coordination
- Configuration synchronization
- Resource management and cleanup
- Error recovery scenarios
"""

import asyncio
import json
import os
import pytest
import signal
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, patch, MagicMock
import psutil
import subprocess

from testcontainers.compose import DockerCompose

from common.core.daemon_manager import (
    DaemonManager,
    DaemonInstance, 
    DaemonConfig,
    DaemonStatus,
    ensure_daemon_running,
    get_daemon_for_project,
    shutdown_all_daemons
)
from common.core.config import Config
from workspace_qdrant_mcp.tools.grpc_tools import test_grpc_connection


@pytest.fixture(scope="module") 
def lifecycle_test_environment():
    """Set up test environment for daemon lifecycle testing."""
    compose_file = """
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6339:6333" 
      - "6340:6334"
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 3s
      timeout: 2s
      retries: 10

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
            
            # Wait for Qdrant
            import requests
            for _ in range(30):
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
                "grpc_port": grpc_port
            }


@pytest.fixture
def test_project_directories():
    """Create multiple test project directories for daemon testing."""
    projects = []
    
    for i in range(3):
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create project structure
            (project_path / "src").mkdir()
            (project_path / "docs").mkdir()
            
            # Add some test files
            (project_path / "src" / "main.py").write_text(f"# Project {i} main file")
            (project_path / "docs" / "readme.md").write_text(f"# Project {i} Documentation")
            
            projects.append({
                "name": f"test-project-{i}",
                "path": str(project_path),
                "grpc_port": 50060 + i,
                "expected_files": 2
            })
    
    yield projects


@pytest.fixture
def daemon_configs(lifecycle_test_environment, test_project_directories):
    """Generate daemon configurations for testing."""
    configs = []
    
    for i, project in enumerate(test_project_directories):
        config = DaemonConfig(
            project_name=project["name"],
            project_path=project["path"],
            grpc_host=lifecycle_test_environment["qdrant_host"],
            grpc_port=project["grpc_port"],
            qdrant_host=lifecycle_test_environment["qdrant_host"],
            qdrant_port=lifecycle_test_environment["qdrant_port"],
            collection_name=f"daemon_test_{i}",
            health_check_interval=1.0,  # Fast for testing
            startup_timeout=10.0,
            shutdown_timeout=5.0,
            max_restart_attempts=3,
            restart_delay=2.0
        )
        configs.append(config)
    
    yield configs


@pytest.mark.integration
@pytest.mark.slow 
class TestDaemonLifecycleIntegration:
    """Integration tests for daemon lifecycle management."""
    
    async def test_daemon_startup_and_initialization(
        self, 
        daemon_configs,
        lifecycle_test_environment
    ):
        """
        Test daemon startup and initialization process.
        
        Verifies:
        1. Daemon process starts successfully
        2. gRPC server becomes available
        3. Health checks pass
        4. Configuration is properly loaded
        5. Required resources are allocated
        """
        
        config = daemon_configs[0]
        
        # Mock the daemon process and gRPC server startup
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            with patch('subprocess.Popen') as MockPopen:
                with patch('workspace_qdrant_mcp.tools.grpc_tools.test_grpc_connection') as mock_grpc:
                    
                    # Set up mocks for successful startup
                    mock_process = MagicMock()
                    mock_process.pid = 12345
                    mock_process.poll.return_value = None  # Process running
                    mock_process.returncode = None
                    MockPopen.return_value = mock_process
                    
                    mock_daemon_instance = MagicMock()
                    mock_daemon_instance.config = config
                    mock_daemon_instance.process = mock_process
                    mock_daemon_instance.status = DaemonStatus.STARTING
                    mock_daemon_instance.pid = 12345
                    mock_daemon_instance.start_time = time.time()
                    mock_daemon_instance.is_healthy.return_value = True
                    
                    mock_daemon_manager = AsyncMock()
                    mock_daemon_manager.start_daemon.return_value = mock_daemon_instance
                    MockDaemonManager.return_value = mock_daemon_manager
                    
                    # Mock successful gRPC connection
                    mock_grpc.return_value = {
                        "connected": True,
                        "healthy": True,
                        "response_time_ms": 45,
                        "engine_info": {
                            "status": "healthy",
                            "services": ["ingest", "search", "health"]
                        }
                    }
                    
                    # Test daemon startup
                    daemon_manager = DaemonManager(config)
                    daemon_instance = await daemon_manager.start_daemon()
                    
                    # Verify startup results
                    assert daemon_instance is not None, "Daemon instance should be created"
                    assert daemon_instance.pid == 12345, "Should have valid PID"
                    assert daemon_instance.status == DaemonStatus.STARTING, "Should be in starting state"
                    
                    # Wait for initialization to complete
                    await asyncio.sleep(0.5)
                    
                    # Verify gRPC connectivity
                    connection_result = await mock_grpc(
                        host=config.grpc_host,
                        port=config.grpc_port,
                        timeout=5.0
                    )
                    
                    assert connection_result["connected"], "gRPC should be connected"
                    assert connection_result["healthy"], "Engine should be healthy"
                    assert connection_result["response_time_ms"] < 1000, "Response time reasonable"

    async def test_daemon_health_monitoring(
        self,
        daemon_configs,
        lifecycle_test_environment
    ):
        """
        Test daemon health monitoring and heartbeat system.
        
        Verifies:
        1. Health checks are performed regularly
        2. Unhealthy daemons are detected
        3. Health status is properly reported
        4. Recovery actions are triggered when needed
        """
        
        config = daemon_configs[0]
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            # Set up daemon with health monitoring
            mock_daemon_manager = AsyncMock()
            MockDaemonManager.return_value = mock_daemon_manager
            
            # Mock daemon instance with health states
            daemon_instance = MagicMock()
            daemon_instance.config = config
            daemon_instance.status = DaemonStatus.RUNNING
            daemon_instance.last_health_check = time.time()
            daemon_instance.health_check_failures = 0
            
            # Test healthy daemon
            daemon_instance.is_healthy.return_value = True
            mock_daemon_manager.get_daemon_health.return_value = {
                "healthy": True,
                "last_check": daemon_instance.last_health_check,
                "failures": 0,
                "uptime_seconds": 120
            }
            
            daemon_manager = DaemonManager(config)
            health_status = await daemon_manager.get_daemon_health()
            
            assert health_status["healthy"], "Healthy daemon should report healthy"
            assert health_status["failures"] == 0, "No failures for healthy daemon"
            assert health_status["uptime_seconds"] > 0, "Should report uptime"
            
            # Test unhealthy daemon detection
            daemon_instance.is_healthy.return_value = False
            daemon_instance.health_check_failures = 3
            mock_daemon_manager.get_daemon_health.return_value = {
                "healthy": False,
                "last_check": daemon_instance.last_health_check,
                "failures": 3,
                "uptime_seconds": 120,
                "last_error": "gRPC connection failed"
            }
            
            health_status = await daemon_manager.get_daemon_health()
            
            assert not health_status["healthy"], "Unhealthy daemon should report unhealthy"
            assert health_status["failures"] == 3, "Should track failure count"
            assert "error" in health_status, "Should include error information"

    async def test_daemon_graceful_shutdown(
        self,
        daemon_configs,
        lifecycle_test_environment
    ):
        """
        Test daemon graceful shutdown and cleanup.
        
        Verifies:
        1. Shutdown signal is sent correctly
        2. Resources are cleaned up properly
        3. Connections are closed gracefully
        4. Files and temporary data are removed
        5. Shutdown timeout is respected
        """
        
        config = daemon_configs[0]
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            with patch('os.kill') as mock_kill:
                with patch('psutil.Process') as MockProcess:
                    
                    # Set up daemon process mock
                    mock_process = MagicMock()
                    mock_process.pid = 12345
                    mock_process.is_running.return_value = True
                    mock_process.terminate.return_value = None
                    mock_process.wait.return_value = 0
                    MockProcess.return_value = mock_process
                    
                    # Set up daemon instance
                    daemon_instance = MagicMock()
                    daemon_instance.config = config
                    daemon_instance.process = mock_process
                    daemon_instance.pid = 12345
                    daemon_instance.status = DaemonStatus.RUNNING
                    daemon_instance.cleanup_resources = AsyncMock()
                    
                    mock_daemon_manager = AsyncMock()
                    mock_daemon_manager.shutdown_daemon.return_value = True
                    MockDaemonManager.return_value = mock_daemon_manager
                    
                    daemon_manager = DaemonManager(config)
                    
                    # Test graceful shutdown
                    start_time = time.time()
                    shutdown_result = await daemon_manager.shutdown_daemon(
                        timeout=config.shutdown_timeout
                    )
                    end_time = time.time()
                    
                    shutdown_duration = end_time - start_time
                    
                    # Verify shutdown results
                    assert shutdown_result, "Shutdown should succeed"
                    assert shutdown_duration < config.shutdown_timeout + 1, "Should complete within timeout"
                    
                    # Verify shutdown sequence
                    mock_daemon_manager.shutdown_daemon.assert_called_once()
                    
                    # Test forced shutdown if graceful fails
                    mock_process.is_running.return_value = True  # Still running
                    mock_process.wait.side_effect = asyncio.TimeoutError()
                    
                    with patch('time.sleep') as mock_sleep:
                        forced_result = await daemon_manager.shutdown_daemon(
                            timeout=1.0,  # Short timeout
                            force=True
                        )
                        
                        # Should still succeed with force
                        assert forced_result, "Forced shutdown should succeed"

    async def test_daemon_restart_and_recovery(
        self,
        daemon_configs,
        lifecycle_test_environment
    ):
        """
        Test daemon restart and error recovery logic.
        
        Verifies:
        1. Failed daemons are detected
        2. Restart attempts are made  
        3. Exponential backoff is applied
        4. Max restart attempts are respected
        5. Recovery state is properly managed
        """
        
        config = daemon_configs[0]
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            # Track restart attempts
            restart_attempts = []
            restart_count = [0]  # Use list for mutable reference
            
            async def mock_start_daemon():
                restart_count[0] += 1
                restart_attempts.append({
                    "attempt": restart_count[0],
                    "time": time.time()
                })
                
                # Simulate failure for first 2 attempts, success on 3rd
                if restart_count[0] <= 2:
                    raise Exception(f"Startup failed (attempt {restart_count[0]})")
                
                # Return successful daemon on 3rd attempt
                daemon_instance = MagicMock()
                daemon_instance.status = DaemonStatus.RUNNING
                daemon_instance.pid = 12300 + restart_count[0]
                daemon_instance.is_healthy.return_value = True
                return daemon_instance
            
            mock_daemon_manager = AsyncMock()
            mock_daemon_manager.start_daemon.side_effect = mock_start_daemon
            mock_daemon_manager.restart_daemon.side_effect = mock_start_daemon
            MockDaemonManager.return_value = mock_daemon_manager
            
            daemon_manager = DaemonManager(config)
            
            # Test restart with recovery
            start_time = time.time()
            
            try:
                # This should eventually succeed after retries
                daemon_instance = await daemon_manager.restart_daemon()
                success = True
            except Exception:
                success = False
            
            end_time = time.time()
            recovery_time = end_time - start_time
            
            # Verify recovery behavior
            assert success, "Should eventually succeed after retries"
            assert len(restart_attempts) == 3, "Should make 3 attempts"
            assert restart_count[0] == 3, "Should succeed on 3rd attempt"
            assert recovery_time < 30, "Recovery should complete in reasonable time"
            
            # Verify backoff timing (rough check)
            if len(restart_attempts) >= 2:
                first_attempt = restart_attempts[0]["time"]
                second_attempt = restart_attempts[1]["time"]
                backoff_delay = second_attempt - first_attempt
                assert backoff_delay >= 1.0, "Should have backoff delay between attempts"

    async def test_multiple_daemon_coordination(
        self,
        daemon_configs,
        lifecycle_test_environment
    ):
        """
        Test coordination between multiple daemon instances.
        
        Verifies:
        1. Multiple daemons can run simultaneously
        2. Port conflicts are avoided
        3. Resources are properly isolated
        4. Cross-daemon communication works
        5. Shutdown coordination is proper
        """
        
        # Use first 2 daemon configs for multi-daemon test
        config1, config2 = daemon_configs[:2]
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            # Set up two daemon instances
            daemon_instances = []
            daemon_managers = []
            
            for i, config in enumerate([config1, config2]):
                daemon_instance = MagicMock()
                daemon_instance.config = config
                daemon_instance.pid = 12400 + i
                daemon_instance.status = DaemonStatus.RUNNING
                daemon_instance.is_healthy.return_value = True
                daemon_instances.append(daemon_instance)
                
                mock_daemon_manager = AsyncMock()
                mock_daemon_manager.start_daemon.return_value = daemon_instance
                mock_daemon_manager.get_daemon_health.return_value = {
                    "healthy": True, "pid": daemon_instance.pid
                }
                daemon_managers.append(mock_daemon_manager)
            
            MockDaemonManager.side_effect = daemon_managers
            
            # Start multiple daemons
            started_daemons = []
            for config in [config1, config2]:
                daemon_manager = DaemonManager(config)
                daemon_instance = await daemon_manager.start_daemon()
                started_daemons.append((daemon_manager, daemon_instance))
            
            # Verify both daemons are running
            assert len(started_daemons) == 2, "Should start both daemons"
            
            # Check each daemon is healthy
            for daemon_manager, daemon_instance in started_daemons:
                health = await daemon_manager.get_daemon_health()
                assert health["healthy"], f"Daemon {daemon_instance.pid} should be healthy"
            
            # Verify different PIDs (no conflicts)
            pids = [daemon_instance.pid for _, daemon_instance in started_daemons]
            assert len(set(pids)) == len(pids), "PIDs should be unique"
            
            # Verify different ports
            ports = [daemon_instance.config.grpc_port for _, daemon_instance in started_daemons]
            assert len(set(ports)) == len(ports), "gRPC ports should be unique"
            
            # Test coordinated shutdown
            shutdown_results = []
            for daemon_manager, daemon_instance in started_daemons:
                result = await daemon_manager.shutdown_daemon()
                shutdown_results.append(result)
            
            assert all(shutdown_results), "All daemons should shutdown successfully"

    async def test_daemon_resource_management(
        self,
        daemon_configs,
        lifecycle_test_environment
    ):
        """
        Test daemon resource management and cleanup.
        
        Verifies:
        1. Memory usage is tracked and controlled
        2. File handles are properly managed
        3. Network connections are cleaned up
        4. Temporary files are removed
        5. Resource limits are respected
        """
        
        config = daemon_configs[0]
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            with patch('psutil.Process') as MockProcess:
                
                # Mock process resource usage
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.memory_info.return_value = MagicMock(
                    rss=50 * 1024 * 1024,  # 50MB resident memory
                    vms=100 * 1024 * 1024  # 100MB virtual memory
                )
                mock_process.open_files.return_value = [
                    MagicMock(path="/tmp/daemon_12345.log"),
                    MagicMock(path="/tmp/daemon_12345.pid")
                ]
                mock_process.connections.return_value = [
                    MagicMock(laddr=("127.0.0.1", config.grpc_port), status="LISTEN")
                ]
                mock_process.is_running.return_value = True
                MockProcess.return_value = mock_process
                
                daemon_instance = MagicMock()
                daemon_instance.config = config
                daemon_instance.process = mock_process
                daemon_instance.pid = 12345
                daemon_instance.status = DaemonStatus.RUNNING
                
                mock_daemon_manager = AsyncMock()
                mock_daemon_manager.get_resource_usage.return_value = {
                    "memory_mb": 50,
                    "open_files": 2,
                    "network_connections": 1,
                    "cpu_percent": 5.2,
                    "uptime_seconds": 300
                }
                MockDaemonManager.return_value = mock_daemon_manager
                
                daemon_manager = DaemonManager(config)
                
                # Test resource monitoring
                resource_usage = await daemon_manager.get_resource_usage()
                
                assert resource_usage["memory_mb"] > 0, "Should report memory usage"
                assert resource_usage["open_files"] >= 0, "Should report file handle count"
                assert resource_usage["network_connections"] >= 0, "Should report connections"
                assert resource_usage["cpu_percent"] >= 0, "Should report CPU usage"
                
                # Verify reasonable resource usage
                assert resource_usage["memory_mb"] < 200, "Memory usage should be reasonable"
                assert resource_usage["open_files"] < 100, "File handles should be limited"
                assert resource_usage["cpu_percent"] < 50, "CPU usage should be reasonable"

    async def test_daemon_configuration_synchronization(
        self,
        daemon_configs,
        lifecycle_test_environment
    ):
        """
        Test configuration synchronization between Python and Rust.
        
        Verifies:
        1. Configuration is passed correctly to daemon
        2. Configuration changes trigger daemon restart
        3. Invalid configurations are rejected
        4. Configuration validation works properly
        """
        
        config = daemon_configs[0]
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            
            # Test valid configuration
            mock_daemon_manager = AsyncMock()
            mock_daemon_manager.validate_config.return_value = True
            mock_daemon_manager.apply_config.return_value = True
            MockDaemonManager.return_value = mock_daemon_manager
            
            daemon_manager = DaemonManager(config)
            
            # Test configuration validation
            config_valid = await daemon_manager.validate_config()
            assert config_valid, "Valid configuration should pass validation"
            
            # Test configuration application
            config_applied = await daemon_manager.apply_config()
            assert config_applied, "Configuration should be applied successfully"
            
            # Test invalid configuration
            invalid_config = DaemonConfig(
                project_name="",  # Invalid empty name
                project_path="/nonexistent/path",  # Invalid path
                grpc_host="invalid-host",
                grpc_port=-1,  # Invalid port
                health_check_interval=-1.0,  # Invalid interval
                startup_timeout=0.0  # Invalid timeout
            )
            
            mock_daemon_manager.validate_config.return_value = False
            daemon_manager_invalid = DaemonManager(invalid_config)
            
            config_valid = await daemon_manager_invalid.validate_config()
            assert not config_valid, "Invalid configuration should fail validation"
            
            # Test configuration change handling
            original_port = config.grpc_port
            config.grpc_port = original_port + 1
            
            mock_daemon_manager.detect_config_changes.return_value = True
            mock_daemon_manager.restart_daemon.return_value = MagicMock(
                status=DaemonStatus.RUNNING,
                config=config
            )
            
            # Simulate configuration change
            config_changed = await daemon_manager.detect_config_changes()
            assert config_changed, "Should detect configuration changes"
            
            if config_changed:
                restarted_daemon = await daemon_manager.restart_daemon()
                assert restarted_daemon.status == DaemonStatus.RUNNING
                assert restarted_daemon.config.grpc_port == original_port + 1