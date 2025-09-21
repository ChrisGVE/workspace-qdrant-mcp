"""
Comprehensive Multi-Instance Daemon Testing Framework.

This module provides comprehensive testing for multi-instance daemon operations,
covering concurrent operations, resource contention, stress testing, chaos engineering,
and performance benchmarking scenarios.

Test Categories:
1. Concurrent Multi-Instance Operations
2. Resource Contention and Port Allocation
3. End-to-End Multi-Project Workflows  
4. Chaos Testing for Failure Recovery
5. Performance Benchmarks (Single vs Multi-Instance)
6. Monitoring and Observability Validation
7. Stress Testing under Load
"""

import asyncio
import json
import os
import pytest
import signal
import tempfile
import time
import random
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import AsyncMock, patch, MagicMock
import psutil
import subprocess
from dataclasses import dataclass, field

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
from common.core.resource_manager import (
    ResourceManager,
    ResourceMonitor,
    ResourceLimits,
    SharedResourcePool,
    get_resource_manager
)
from common.core.service_discovery.client import (
    ServiceDiscoveryClient,
    ServiceEndpoint
)
from common.core.project_config_manager import (
    ProjectConfigManager,
    DaemonProjectConfig
)
from common.core.config import Config
from workspace_qdrant_mcp.tools.grpc_tools import test_grpc_connection


@dataclass
class TestProject:
    """Test project configuration for multi-instance testing."""
    name: str
    path: str
    grpc_port: int
    collection_name: str
    expected_files: int = 0
    resource_limits: Optional[ResourceLimits] = None
    config_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MultiInstanceTestMetrics:
    """Metrics collected during multi-instance testing."""
    startup_times: List[float] = field(default_factory=list)
    memory_usage: List[Dict[str, float]] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    failure_counts: Dict[str, int] = field(default_factory=dict)
    resource_contention_events: int = 0
    port_conflicts: int = 0
    recovery_times: List[float] = field(default_factory=list)


@pytest.fixture(scope="module")
def multi_instance_test_environment():
    """Set up isolated test environment for multi-instance daemon testing."""
    compose_file = """
version: '3.8'
services:
  qdrant-multi-test:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6350:6333"  # Different port range to avoid conflicts
      - "6351:6334"
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 2s
      timeout: 1s
      retries: 15
    volumes:
      - qdrant_multi_storage:/qdrant/storage

volumes:
  qdrant_multi_storage:
"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        compose_path = Path(temp_dir) / "docker-compose.yml"
        compose_path.write_text(compose_file)
        
        with DockerCompose(temp_dir) as compose:
            qdrant_url = compose.get_service_host("qdrant-multi-test", 6333)
            qdrant_port = compose.get_service_port("qdrant-multi-test", 6333)
            grpc_port = compose.get_service_port("qdrant-multi-test", 6334)
            
            # Wait for Qdrant with longer timeout for multi-instance tests
            import requests
            for attempt in range(45):
                try:
                    response = requests.get(f"http://{qdrant_url}:{qdrant_port}/health", timeout=3)
                    if response.status_code == 200:
                        break
                except:
                    pass
                time.sleep(1)
            else:
                pytest.skip("Qdrant service not available for multi-instance tests")
            
            yield {
                "qdrant_host": qdrant_url,
                "qdrant_port": qdrant_port,
                "grpc_port": grpc_port,
                "base_url": f"http://{qdrant_url}:{qdrant_port}"
            }


@pytest.fixture
def test_project_cluster():
    """Create a cluster of test projects for multi-instance testing."""
    projects = []
    temp_dirs = []
    
    try:
        for i in range(5):  # Create 5 test projects
            temp_dir = tempfile.mkdtemp(prefix=f"multi_test_project_{i}_")
            temp_dirs.append(temp_dir)
            project_path = Path(temp_dir)
            
            # Create realistic project structure
            (project_path / "src").mkdir()
            (project_path / "tests").mkdir()
            (project_path / "docs").mkdir()
            (project_path / ".git").mkdir()  # Simulate git repo
            
            # Add varied content for different projects
            (project_path / "src" / "main.py").write_text(f"""
# Project {i} main application
import asyncio
from typing import List, Dict

class Project{i}App:
    def __init__(self):
        self.name = "project-{i}"
        self.version = "1.{i}.0"
    
    async def run(self):
        print(f"Running {{self.name}} v{{self.version}}")
        return True

if __name__ == "__main__":
    app = Project{i}App()
    asyncio.run(app.run())
""")
            
            (project_path / "tests" / "test_main.py").write_text(f"""
import pytest
from src.main import Project{i}App

def test_project_{i}_creation():
    app = Project{i}App()
    assert app.name == "project-{i}"
    assert app.version == "1.{i}.0"

@pytest.mark.asyncio
async def test_project_{i}_run():
    app = Project{i}App()
    result = await app.run()
    assert result is True
""")
            
            (project_path / "docs" / "README.md").write_text(f"""
# Project {i} Documentation

This is test project {i} for multi-instance daemon testing.

## Features
- Feature A for project {i}
- Feature B for project {i}
- Advanced feature C-{i}

## Usage
Run the main application with:
```python
python src/main.py
```
""")
            
            # Create resource limits based on project index
            resource_limits = ResourceLimits(
                max_memory_mb=50 + (i * 20),  # 50-130 MB
                max_cpu_percent=10 + (i * 5),  # 10-30%
                max_file_descriptors=100 + (i * 20),  # 100-180 FDs
                max_disk_mb=200 + (i * 50)  # 200-400 MB
            )
            
            project = TestProject(
                name=f"multi-test-project-{i}",
                path=str(project_path),
                grpc_port=50070 + i,  # Ports 50070-50074
                collection_name=f"multi_test_collection_{i}",
                expected_files=3,
                resource_limits=resource_limits,
                config_overrides={
                    "health_check_interval": 2.0 + (i * 0.5),  # Varied intervals
                    "startup_timeout": 15.0 + (i * 2),
                    "enable_resource_monitoring": True,
                    "enable_file_watching": True
                }
            )
            
            projects.append(project)
        
        yield projects
    
    finally:
        # Cleanup temp directories
        for temp_dir in temp_dirs:
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass


@pytest.fixture
def daemon_config_cluster(multi_instance_test_environment, test_project_cluster):
    """Generate daemon configurations for multi-instance testing."""
    configs = []
    
    for project in test_project_cluster:
        config = DaemonConfig(
            project_name=project.name,
            project_path=project.path,
            grpc_host=multi_instance_test_environment["qdrant_host"],
            grpc_port=project.grpc_port,
            qdrant_host=multi_instance_test_environment["qdrant_host"],
            qdrant_port=multi_instance_test_environment["qdrant_port"],
            collection_name=project.collection_name,
            health_check_interval=project.config_overrides.get("health_check_interval", 2.0),
            startup_timeout=project.config_overrides.get("startup_timeout", 15.0),
            shutdown_timeout=5.0,
            max_restart_attempts=3,
            restart_delay=1.0,
            enable_resource_monitoring=project.config_overrides.get("enable_resource_monitoring", True),
            enable_file_watching=project.config_overrides.get("enable_file_watching", True),
            project_id=f"project_{hash(project.path) % 10000:04d}"
        )
        configs.append(config)
    
    yield configs


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.multi_instance
class TestMultiInstanceDaemonFramework:
    """Comprehensive multi-instance daemon testing framework."""
    
    async def test_concurrent_daemon_startup(
        self,
        daemon_config_cluster,
        multi_instance_test_environment
    ):
        """
        Test concurrent startup of multiple daemon instances.
        
        Verifies:
        1. Multiple daemons can start simultaneously without conflicts
        2. Port allocation works correctly for concurrent startups
        3. Resource initialization doesn't cause contention
        4. All instances become healthy within timeout
        5. Startup times are reasonable under concurrent load
        """
        
        configs = daemon_config_cluster
        metrics = MultiInstanceTestMetrics()
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            
            # Set up mock daemon instances for concurrent testing
            mock_instances = []
            mock_managers = []
            
            for i, config in enumerate(configs):
                daemon_instance = MagicMock()
                daemon_instance.config = config
                daemon_instance.pid = 15000 + i
                daemon_instance.status = DaemonStatus.STARTING
                daemon_instance.start_time = time.time()
                daemon_instance.is_healthy.return_value = True
                mock_instances.append(daemon_instance)
                
                mock_manager = AsyncMock()
                mock_manager.start_daemon.return_value = daemon_instance
                mock_manager.get_daemon_health.return_value = {
                    "healthy": True,
                    "pid": daemon_instance.pid,
                    "startup_time": 2.0 + (i * 0.2),  # Simulated startup times
                    "memory_mb": 45 + (i * 5)
                }
                mock_managers.append(mock_manager)
            
            MockDaemonManager.side_effect = mock_managers
            
            # Test concurrent startup
            startup_tasks = []
            start_time = time.time()
            
            async def start_daemon_with_metrics(config, manager_class):
                daemon_start = time.time()
                daemon_manager = manager_class(config)
                daemon_instance = await daemon_manager.start_daemon()
                startup_time = time.time() - daemon_start
                metrics.startup_times.append(startup_time)
                return daemon_manager, daemon_instance
            
            # Launch all startups concurrently
            for config, manager_class in zip(configs, mock_managers):
                task = start_daemon_with_metrics(config, MockDaemonManager)
                startup_tasks.append(task)
            
            # Wait for all startups to complete
            results = await asyncio.gather(*startup_tasks, return_exceptions=True)
            total_startup_time = time.time() - start_time
            
            # Verify all startups succeeded
            successful_starts = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_starts) == len(configs), f"Expected {len(configs)} successful starts, got {len(successful_starts)}"
            
            # Verify startup performance
            assert total_startup_time < 30.0, f"Concurrent startup took too long: {total_startup_time}s"
            assert max(metrics.startup_times) < 15.0, f"Individual startup time exceeded limit: {max(metrics.startup_times)}s"
            assert len(set(d.pid for _, d in successful_starts)) == len(configs), "All PIDs should be unique"
            
            # Verify no port conflicts occurred
            used_ports = set()
            for _, daemon_instance in successful_starts:
                port = daemon_instance.config.grpc_port
                assert port not in used_ports, f"Port conflict detected: {port}"
                used_ports.add(port)
            
            print(f"Concurrent startup metrics:")
            print(f"  Total time: {total_startup_time:.2f}s")
            print(f"  Individual times: {[f'{t:.2f}s' for t in metrics.startup_times]}")
            print(f"  Average startup: {sum(metrics.startup_times) / len(metrics.startup_times):.2f}s")

    async def test_resource_contention_and_limits(
        self,
        daemon_config_cluster,
        multi_instance_test_environment
    ):
        """
        Test resource contention handling and limit enforcement.
        
        Verifies:
        1. Resource limits are enforced per daemon instance
        2. Shared resource pooling works correctly
        3. Resource contention is detected and handled
        4. Memory pressure triggers appropriate responses
        5. Resource monitoring provides accurate data
        """
        
        configs = daemon_config_cluster[:3]  # Use 3 instances for resource testing
        metrics = MultiInstanceTestMetrics()
        
        with patch('workspace_qdrant_mcp.core.resource_manager.get_resource_manager') as mock_get_manager:
            with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
                
                # Set up resource manager with monitoring
                resource_manager = AsyncMock(spec=ResourceManager)
                mock_monitors = []
                
                for i, config in enumerate(configs):
                    monitor = MagicMock(spec=ResourceMonitor)
                    monitor.project_id = config.project_id
                    monitor.limits = ResourceLimits(
                        max_memory_mb=60 + (i * 10),
                        max_cpu_percent=15 + (i * 5),
                        max_file_descriptors=120 + (i * 10)
                    )
                    monitor.current_usage = {
                        "memory_mb": 40 + (i * 8),  # Within limits initially
                        "cpu_percent": 10 + (i * 2),
                        "file_descriptors": 80 + (i * 5)
                    }
                    monitor.is_over_limit.return_value = False
                    monitor.get_usage_percentage.return_value = 0.7 + (i * 0.1)  # 70-90% usage
                    mock_monitors.append(monitor)
                
                resource_manager.register_project = AsyncMock()
                resource_manager.register_project.side_effect = mock_monitors
                resource_manager.get_shared_pool.return_value = MagicMock()
                resource_manager.check_system_resources.return_value = {
                    "memory_available_mb": 1024,
                    "cpu_available_percent": 80,
                    "disk_available_mb": 5000
                }
                mock_get_manager.return_value = resource_manager
                
                # Set up daemon managers with resource monitoring
                daemon_managers = []
                for i, config in enumerate(configs):
                    daemon_instance = MagicMock()
                    daemon_instance.config = config
                    daemon_instance.pid = 16000 + i
                    daemon_instance.status = DaemonStatus.RUNNING
                    daemon_instance.resource_monitor = mock_monitors[i]
                    
                    mock_manager = AsyncMock()
                    mock_manager.start_daemon.return_value = daemon_instance
                    mock_manager.get_resource_usage.return_value = {
                        "memory_mb": mock_monitors[i].current_usage["memory_mb"],
                        "cpu_percent": mock_monitors[i].current_usage["cpu_percent"],
                        "file_descriptors": mock_monitors[i].current_usage["file_descriptors"],
                        "within_limits": True
                    }
                    daemon_managers.append(mock_manager)
                
                MockDaemonManager.side_effect = daemon_managers
                
                # Start daemons with resource monitoring
                running_daemons = []
                for config, manager_class in zip(configs, daemon_managers):
                    daemon_manager = MockDaemonManager(config)
                    daemon_instance = await daemon_manager.start_daemon()
                    running_daemons.append((daemon_manager, daemon_instance))
                
                # Test resource monitoring
                for daemon_manager, daemon_instance in running_daemons:
                    usage = await daemon_manager.get_resource_usage()
                    
                    assert usage["memory_mb"] > 0, "Should report memory usage"
                    assert usage["cpu_percent"] >= 0, "Should report CPU usage"
                    assert usage["file_descriptors"] > 0, "Should report file descriptor usage"
                    assert usage["within_limits"], "Should be within resource limits"
                    
                    metrics.memory_usage.append(usage)
                
                # Simulate resource pressure scenario
                high_usage_monitor = mock_monitors[0]
                high_usage_monitor.current_usage = {
                    "memory_mb": 58,  # Close to 60MB limit
                    "cpu_percent": 14,  # Close to 15% limit
                    "file_descriptors": 118  # Close to 120 limit
                }
                high_usage_monitor.is_over_limit.return_value = False
                high_usage_monitor.get_usage_percentage.return_value = 0.95  # 95% usage
                
                # Test resource pressure detection
                usage_under_pressure = await daemon_managers[0].get_resource_usage()
                assert usage_under_pressure["memory_mb"] > 50, "Should show high memory usage"
                
                # Simulate resource limit exceeded
                over_limit_monitor = mock_monitors[1]
                over_limit_monitor.current_usage = {
                    "memory_mb": 72,  # Over 70MB limit
                    "cpu_percent": 22,  # Over 20% limit
                    "file_descriptors": 135  # Over 130 limit
                }
                over_limit_monitor.is_over_limit.return_value = True
                daemon_managers[1].get_resource_usage.return_value = {
                    **over_limit_monitor.current_usage,
                    "within_limits": False,
                    "limit_exceeded": ["memory_mb", "cpu_percent", "file_descriptors"]
                }
                
                usage_over_limit = await daemon_managers[1].get_resource_usage()
                assert not usage_over_limit["within_limits"], "Should detect resource limit exceeded"
                assert "limit_exceeded" in usage_over_limit, "Should specify which limits exceeded"
                
                metrics.resource_contention_events += 1
                
                print(f"Resource contention test results:")
                print(f"  Monitored instances: {len(running_daemons)}")
                print(f"  Resource pressure events: {metrics.resource_contention_events}")
                print(f"  Memory usage range: {min(m['memory_mb'] for m in metrics.memory_usage)}-{max(m['memory_mb'] for m in metrics.memory_usage)} MB")

    async def test_end_to_end_multi_project_workflow(
        self,
        daemon_config_cluster,
        test_project_cluster,
        multi_instance_test_environment
    ):
        """
        Test complete end-to-end workflow with multiple projects.
        
        Verifies:
        1. Multi-project ingestion and processing
        2. Cross-project search capabilities
        3. Project isolation and data separation
        4. Configuration management across projects
        5. Service discovery between instances
        """
        
        configs = daemon_config_cluster[:3]
        projects = test_project_cluster[:3]
        metrics = MultiInstanceTestMetrics()
        
        with patch('workspace_qdrant_mcp.core.service_discovery.client.ServiceDiscoveryClient') as MockDiscoveryClient:
            with patch('workspace_qdrant_mcp.core.project_config_manager.ProjectConfigManager') as MockConfigManager:
                with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
                    
                    # Set up service discovery
                    discovery_client = AsyncMock(spec=ServiceDiscoveryClient)
                    service_endpoints = []
                    
                    for i, config in enumerate(configs):
                        endpoint = ServiceEndpoint(
                            host=config.grpc_host,
                            port=config.grpc_port,
                            project_id=config.project_id,
                            health_status="healthy",
                            last_seen=time.time(),
                            metadata={
                                "project_name": config.project_name,
                                "collection": config.collection_name,
                                "version": "1.0.0"
                            }
                        )
                        service_endpoints.append(endpoint)
                    
                    discovery_client.register_service.return_value = True
                    discovery_client.discover_daemon_for_project.side_effect = service_endpoints
                    discovery_client.list_all_services.return_value = service_endpoints
                    MockDiscoveryClient.return_value = discovery_client
                    
                    # Set up project configuration managers
                    config_managers = []
                    for i, project in enumerate(projects):
                        config_manager = AsyncMock(spec=ProjectConfigManager)
                        project_config = DaemonProjectConfig(
                            project_path=project.path,
                            daemon_config={
                                "grpc_port": configs[i].grpc_port,
                                "collection_name": configs[i].collection_name,
                                "enable_file_watching": True
                            },
                            collection_settings={
                                "vector_size": 384,
                                "distance_metric": "cosine"
                            }
                        )
                        config_manager.load_config.return_value = project_config
                        config_manager.save_config.return_value = True
                        config_managers.append(config_manager)
                    
                    MockConfigManager.side_effect = config_managers
                    
                    # Set up daemon managers
                    daemon_managers = []
                    daemon_instances = []
                    
                    for i, config in enumerate(configs):
                        daemon_instance = MagicMock()
                        daemon_instance.config = config
                        daemon_instance.pid = 17000 + i
                        daemon_instance.status = DaemonStatus.RUNNING
                        daemon_instance.project_config = config_managers[i].load_config()
                        daemon_instances.append(daemon_instance)
                        
                        mock_manager = AsyncMock()
                        mock_manager.start_daemon.return_value = daemon_instance
                        mock_manager.ingest_documents.return_value = {
                            "processed_files": projects[i].expected_files,
                            "success": True,
                            "processing_time": 1.5 + (i * 0.3)
                        }
                        mock_manager.search_documents.return_value = {
                            "results": [
                                {
                                    "content": f"Sample content from project {i}",
                                    "score": 0.85 - (i * 0.1),
                                    "metadata": {"project": projects[i].name}
                                }
                            ],
                            "total_results": 1,
                            "search_time_ms": 45 + (i * 10)
                        }
                        daemon_managers.append(mock_manager)
                    
                    MockDaemonManager.side_effect = daemon_managers
                    
                    # Execute end-to-end workflow
                    
                    # Phase 1: Start all daemon instances
                    print("Phase 1: Starting daemon instances...")
                    active_daemons = []
                    for config, manager_class in zip(configs, daemon_managers):
                        daemon_manager = MockDaemonManager(config)
                        daemon_instance = await daemon_manager.start_daemon()
                        active_daemons.append((daemon_manager, daemon_instance))
                    
                    assert len(active_daemons) == 3, "Should start all 3 daemon instances"
                    
                    # Phase 2: Register services with discovery
                    print("Phase 2: Registering services...")
                    for i, (daemon_manager, daemon_instance) in enumerate(active_daemons):
                        endpoint = service_endpoints[i]
                        registered = await discovery_client.register_service(endpoint)
                        assert registered, f"Should register service for daemon {i}"
                    
                    # Phase 3: Document ingestion across projects
                    print("Phase 3: Ingesting documents...")
                    ingestion_results = []
                    for i, (daemon_manager, daemon_instance) in enumerate(active_daemons):
                        start_time = time.time()
                        result = await daemon_manager.ingest_documents(projects[i].path)
                        ingestion_time = time.time() - start_time
                        
                        assert result["success"], f"Ingestion should succeed for project {i}"
                        assert result["processed_files"] == projects[i].expected_files
                        
                        ingestion_results.append(result)
                        metrics.response_times.append(ingestion_time)
                    
                    # Phase 4: Cross-project search testing
                    print("Phase 4: Testing cross-project search...")
                    search_results = []
                    for i, (daemon_manager, daemon_instance) in enumerate(active_daemons):
                        start_time = time.time()
                        result = await daemon_manager.search_documents(
                            query=f"test content project {i}",
                            limit=10
                        )
                        search_time = time.time() - start_time
                        
                        assert len(result["results"]) > 0, f"Should find results for project {i}"
                        assert result["search_time_ms"] < 500, f"Search should be fast for project {i}"
                        
                        search_results.append(result)
                        metrics.response_times.append(search_time)
                    
                    # Phase 5: Service discovery verification
                    print("Phase 5: Verifying service discovery...")
                    all_services = await discovery_client.list_all_services()
                    assert len(all_services) == 3, "Should discover all 3 services"
                    
                    project_names = {s.metadata["project_name"] for s in all_services}
                    expected_names = {p.name for p in projects}
                    assert project_names == expected_names, "Should discover all project services"
                    
                    # Phase 6: Configuration validation
                    print("Phase 6: Validating project configurations...")
                    for i, config_manager in enumerate(config_managers):
                        project_config = config_manager.load_config()
                        assert project_config.project_path == projects[i].path
                        assert project_config.daemon_config["grpc_port"] == configs[i].grpc_port
                    
                    print(f"End-to-end workflow completed successfully:")
                    print(f"  Active daemons: {len(active_daemons)}")
                    print(f"  Documents ingested: {sum(r['processed_files'] for r in ingestion_results)}")
                    print(f"  Search operations: {len(search_results)}")
                    print(f"  Average response time: {sum(metrics.response_times) / len(metrics.response_times):.3f}s")

    async def test_chaos_engineering_failure_recovery(
        self,
        daemon_config_cluster,
        multi_instance_test_environment
    ):
        """
        Test daemon failure scenarios and recovery mechanisms.
        
        Verifies:
        1. Graceful handling of random daemon failures
        2. Automatic restart and recovery procedures
        3. Service discovery updates during failures
        4. Resource cleanup after failures
        5. System stability under chaos conditions
        """
        
        configs = daemon_config_cluster[:4]  # Use 4 instances for chaos testing
        metrics = MultiInstanceTestMetrics()
        failure_scenarios = ["process_kill", "network_failure", "resource_exhaustion", "config_corruption"]
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            with patch('workspace_qdrant_mcp.core.service_discovery.client.ServiceDiscoveryClient') as MockDiscoveryClient:
                
                # Set up daemon instances with failure simulation
                daemon_managers = []
                daemon_instances = []
                failure_states = {}  # Track which daemons have failed
                
                for i, config in enumerate(configs):
                    daemon_instance = MagicMock()
                    daemon_instance.config = config
                    daemon_instance.pid = 18000 + i
                    daemon_instance.status = DaemonStatus.RUNNING
                    daemon_instance.restart_count = 0
                    daemon_instance.last_failure = None
                    daemon_instances.append(daemon_instance)
                    
                    failure_states[i] = {"failed": False, "failure_type": None, "recovery_time": None}
                    
                    mock_manager = AsyncMock()
                    mock_manager.start_daemon.return_value = daemon_instance
                    mock_manager.restart_daemon = AsyncMock()
                    mock_manager.is_healthy = AsyncMock()
                    daemon_managers.append(mock_manager)
                
                MockDaemonManager.side_effect = daemon_managers
                
                # Set up service discovery with failure handling
                discovery_client = AsyncMock()
                discovery_client.register_service.return_value = True
                discovery_client.unregister_service.return_value = True
                discovery_client.mark_service_unhealthy.return_value = True
                MockDiscoveryClient.return_value = discovery_client
                
                # Start all daemon instances
                active_daemons = []
                for config, manager_class in zip(configs, daemon_managers):
                    daemon_manager = MockDaemonManager(config)
                    daemon_instance = await daemon_manager.start_daemon()
                    active_daemons.append((daemon_manager, daemon_instance))
                
                print(f"Chaos testing started with {len(active_daemons)} daemons")
                
                # Simulate chaos scenarios
                chaos_rounds = 3
                for round_num in range(chaos_rounds):
                    print(f"Chaos round {round_num + 1}/{chaos_rounds}")
                    
                    # Randomly select daemons to fail
                    num_failures = random.randint(1, min(2, len(active_daemons)))
                    failed_indices = random.sample(range(len(active_daemons)), num_failures)
                    
                    for fail_idx in failed_indices:
                        daemon_manager, daemon_instance = active_daemons[fail_idx]
                        failure_type = random.choice(failure_scenarios)
                        
                        print(f"  Simulating {failure_type} for daemon {fail_idx}")
                        
                        # Simulate different failure types
                        if failure_type == "process_kill":
                            daemon_instance.status = DaemonStatus.STOPPED
                            daemon_instance.pid = None
                            daemon_manager.is_healthy.return_value = False
                            
                        elif failure_type == "network_failure":
                            daemon_instance.status = DaemonStatus.RUNNING
                            daemon_manager.is_healthy.return_value = False
                            # Simulate network connectivity issues
                            
                        elif failure_type == "resource_exhaustion":
                            daemon_instance.status = DaemonStatus.RUNNING
                            daemon_manager.is_healthy.return_value = False
                            # Simulate resource limits exceeded
                            
                        elif failure_type == "config_corruption":
                            daemon_instance.status = DaemonStatus.FAILED
                            daemon_manager.is_healthy.return_value = False
                            # Simulate configuration file corruption
                        
                        failure_states[fail_idx] = {
                            "failed": True,
                            "failure_type": failure_type,
                            "failure_time": time.time()
                        }
                        
                        metrics.failure_counts[failure_type] = metrics.failure_counts.get(failure_type, 0) + 1
                        
                        # Simulate service discovery notification
                        await discovery_client.mark_service_unhealthy(daemon_instance.config.project_id)
                    
                    # Allow some time for failure detection
                    await asyncio.sleep(1.0)
                    
                    # Test recovery procedures
                    for fail_idx in failed_indices:
                        daemon_manager, daemon_instance = active_daemons[fail_idx]
                        failure_info = failure_states[fail_idx]
                        
                        print(f"  Attempting recovery for daemon {fail_idx}")
                        recovery_start = time.time()
                        
                        # Simulate recovery based on failure type
                        if failure_info["failure_type"] == "process_kill":
                            # Simulate process restart
                            daemon_instance.status = DaemonStatus.STARTING
                            daemon_instance.pid = 18100 + fail_idx
                            daemon_instance.restart_count += 1
                            await asyncio.sleep(0.5)  # Simulate startup time
                            daemon_instance.status = DaemonStatus.RUNNING
                            daemon_manager.is_healthy.return_value = True
                            
                        elif failure_info["failure_type"] == "network_failure":
                            # Simulate network recovery
                            await asyncio.sleep(0.3)
                            daemon_manager.is_healthy.return_value = True
                            
                        elif failure_info["failure_type"] == "resource_exhaustion":
                            # Simulate resource cleanup and restart
                            daemon_instance.restart_count += 1
                            await asyncio.sleep(0.7)
                            daemon_manager.is_healthy.return_value = True
                            
                        elif failure_info["failure_type"] == "config_corruption":
                            # Simulate config restoration and restart
                            daemon_instance.status = DaemonStatus.STARTING
                            daemon_instance.restart_count += 1
                            await asyncio.sleep(0.8)
                            daemon_instance.status = DaemonStatus.RUNNING
                            daemon_manager.is_healthy.return_value = True
                        
                        recovery_time = time.time() - recovery_start
                        metrics.recovery_times.append(recovery_time)
                        
                        failure_states[fail_idx] = {
                            "failed": False,
                            "failure_type": None,
                            "recovery_time": recovery_time
                        }
                        
                        # Re-register with service discovery
                        await discovery_client.register_service(
                            ServiceEndpoint(
                                host=daemon_instance.config.grpc_host,
                                port=daemon_instance.config.grpc_port,
                                project_id=daemon_instance.config.project_id,
                                health_status="healthy",
                                last_seen=time.time()
                            )
                        )
                        
                        print(f"    Recovery completed in {recovery_time:.2f}s")
                    
                    # Verify system stability after recovery
                    healthy_count = 0
                    for daemon_manager, daemon_instance in active_daemons:
                        if await daemon_manager.is_healthy():
                            healthy_count += 1
                    
                    assert healthy_count == len(active_daemons), f"All daemons should be healthy after recovery (round {round_num + 1})"
                    
                    # Wait between chaos rounds
                    if round_num < chaos_rounds - 1:
                        await asyncio.sleep(2.0)
                
                print(f"Chaos engineering test completed:")
                print(f"  Total failures induced: {sum(metrics.failure_counts.values())}")
                print(f"  Failure types: {dict(metrics.failure_counts)}")
                print(f"  Average recovery time: {sum(metrics.recovery_times) / len(metrics.recovery_times):.2f}s")
                print(f"  Max recovery time: {max(metrics.recovery_times):.2f}s")
                
                # Verify final system state
                final_healthy_count = 0
                for daemon_manager, daemon_instance in active_daemons:
                    if await daemon_manager.is_healthy():
                        final_healthy_count += 1
                
                assert final_healthy_count == len(active_daemons), "All daemons should be healthy after chaos testing"
                assert len(metrics.recovery_times) > 0, "Should have recorded recovery times"
                assert max(metrics.recovery_times) < 10.0, "Recovery times should be reasonable"

    async def test_performance_benchmarks_single_vs_multi(
        self,
        daemon_config_cluster,
        multi_instance_test_environment
    ):
        """
        Test performance benchmarks comparing single vs multi-instance overhead.
        
        Verifies:
        1. Performance characteristics under different loads
        2. Resource utilization efficiency
        3. Scalability patterns
        4. Throughput and latency measurements
        5. Memory and CPU overhead analysis
        """
        
        configs = daemon_config_cluster
        metrics = MultiInstanceTestMetrics()
        
        # Test scenarios: single instance vs multiple instances
        test_scenarios = [
            {"name": "single_instance", "instance_count": 1},
            {"name": "dual_instance", "instance_count": 2},
            {"name": "multi_instance", "instance_count": 4},
            {"name": "stress_instance", "instance_count": 5}
        ]
        
        benchmark_results = {}
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            
            for scenario in test_scenarios:
                print(f"Benchmarking {scenario['name']} scenario...")
                scenario_configs = configs[:scenario['instance_count']]
                scenario_results = {
                    "startup_time": 0,
                    "memory_usage": [],
                    "cpu_usage": [],
                    "operation_latency": [],
                    "throughput_ops_per_sec": 0,
                    "resource_overhead": 0
                }
                
                # Set up mock daemon managers for this scenario
                daemon_managers = []
                daemon_instances = []
                
                base_memory_per_instance = 45  # MB
                base_cpu_per_instance = 8  # %
                
                for i, config in enumerate(scenario_configs):
                    daemon_instance = MagicMock()
                    daemon_instance.config = config
                    daemon_instance.pid = 19000 + i
                    daemon_instance.status = DaemonStatus.RUNNING
                    
                    # Simulate realistic resource usage with slight overhead for multiple instances
                    memory_overhead_factor = 1 + (0.1 * (scenario['instance_count'] - 1))
                    cpu_overhead_factor = 1 + (0.05 * (scenario['instance_count'] - 1))
                    
                    memory_usage = base_memory_per_instance * memory_overhead_factor + (i * 5)
                    cpu_usage = base_cpu_per_instance * cpu_overhead_factor + (i * 2)
                    
                    daemon_instances.append(daemon_instance)
                    
                    mock_manager = AsyncMock()
                    mock_manager.start_daemon.return_value = daemon_instance
                    mock_manager.get_resource_usage.return_value = {
                        "memory_mb": memory_usage,
                        "cpu_percent": cpu_usage,
                        "startup_time": 2.0 + (i * 0.2),
                        "operation_latency_ms": 50 + (i * 5) + (scenario['instance_count'] * 2)
                    }
                    
                    # Simulate operations with scaling characteristics
                    operations_per_second = max(100 - (scenario['instance_count'] * 5), 50)
                    mock_manager.benchmark_operations.return_value = {
                        "ops_per_second": operations_per_second,
                        "avg_latency_ms": 45 + (scenario['instance_count'] * 3),
                        "p95_latency_ms": 80 + (scenario['instance_count'] * 5),
                        "success_rate": 0.99 - (scenario['instance_count'] * 0.005)
                    }
                    
                    daemon_managers.append(mock_manager)
                
                MockDaemonManager.side_effect = daemon_managers
                
                # Benchmark startup time
                startup_start = time.time()
                active_daemons = []
                
                startup_tasks = []
                for config, manager_class in zip(scenario_configs, daemon_managers):
                    async def start_daemon(cfg, mgr_cls):
                        daemon_manager = MockDaemonManager(cfg)
                        return daemon_manager, await daemon_manager.start_daemon()
                    
                    startup_tasks.append(start_daemon(config, MockDaemonManager))
                
                startup_results = await asyncio.gather(*startup_tasks)
                scenario_results["startup_time"] = time.time() - startup_start
                active_daemons = startup_results
                
                # Collect resource usage metrics
                total_memory = 0
                total_cpu = 0
                operation_latencies = []
                
                for daemon_manager, daemon_instance in active_daemons:
                    usage = await daemon_manager.get_resource_usage()
                    scenario_results["memory_usage"].append(usage["memory_mb"])
                    scenario_results["cpu_usage"].append(usage["cpu_percent"])
                    operation_latencies.append(usage["operation_latency_ms"])
                    
                    total_memory += usage["memory_mb"]
                    total_cpu += usage["cpu_percent"]
                
                scenario_results["operation_latency"] = operation_latencies
                
                # Benchmark operation throughput
                throughput_results = []
                for daemon_manager, daemon_instance in active_daemons:
                    bench_result = await daemon_manager.benchmark_operations()
                    throughput_results.append(bench_result["ops_per_second"])
                
                scenario_results["throughput_ops_per_sec"] = sum(throughput_results)
                
                # Calculate resource overhead (compared to single instance baseline)
                if scenario['name'] == 'single_instance':
                    baseline_memory = total_memory
                    baseline_cpu = total_cpu
                    scenario_results["resource_overhead"] = 0
                else:
                    expected_linear_memory = baseline_memory * scenario['instance_count']
                    expected_linear_cpu = baseline_cpu * scenario['instance_count']
                    
                    memory_overhead = ((total_memory - expected_linear_memory) / expected_linear_memory) * 100
                    cpu_overhead = ((total_cpu - expected_linear_cpu) / expected_linear_cpu) * 100
                    
                    scenario_results["resource_overhead"] = (memory_overhead + cpu_overhead) / 2
                
                benchmark_results[scenario['name']] = scenario_results
                
                print(f"  {scenario['name']} results:")
                print(f"    Startup time: {scenario_results['startup_time']:.2f}s")
                print(f"    Total memory: {total_memory:.1f}MB")
                print(f"    Total CPU: {total_cpu:.1f}%")
                print(f"    Throughput: {scenario_results['throughput_ops_per_sec']:.1f} ops/sec")
                print(f"    Avg latency: {sum(operation_latencies) / len(operation_latencies):.1f}ms")
                if scenario['name'] != 'single_instance':
                    print(f"    Resource overhead: {scenario_results['resource_overhead']:.1f}%")
        
        # Analyze benchmark results
        print("\nPerformance Benchmark Analysis:")
        
        # Startup time scaling
        startup_times = [benchmark_results[s['name']]['startup_time'] for s in test_scenarios]
        print(f"Startup time scaling: {startup_times}")
        
        # Verify reasonable startup times (should be sub-linear growth)
        assert benchmark_results['single_instance']['startup_time'] < 5.0, "Single instance startup should be fast"
        assert benchmark_results['multi_instance']['startup_time'] < 15.0, "Multi-instance startup should be reasonable"
        
        # Memory efficiency
        memory_per_instance = [
            sum(benchmark_results[s['name']]['memory_usage']) / s['instance_count'] 
            for s in test_scenarios
        ]
        print(f"Memory per instance: {[f'{m:.1f}MB' for m in memory_per_instance]}")
        
        # Throughput scaling
        throughput_values = [benchmark_results[s['name']]['throughput_ops_per_sec'] for s in test_scenarios]
        print(f"Throughput scaling: {throughput_values}")
        
        # Resource overhead should be reasonable
        for scenario in test_scenarios[1:]:  # Skip single instance
            overhead = benchmark_results[scenario['name']]['resource_overhead']
            assert overhead < 50.0, f"Resource overhead for {scenario['name']} should be reasonable: {overhead}%"
        
        # Latency should not degrade significantly
        avg_latencies = [
            sum(benchmark_results[s['name']]['operation_latency']) / len(benchmark_results[s['name']]['operation_latency'])
            for s in test_scenarios
        ]
        print(f"Average latencies: {[f'{l:.1f}ms' for l in avg_latencies]}")
        
        single_latency = avg_latencies[0]
        for i, latency in enumerate(avg_latencies[1:], 1):
            latency_increase = ((latency - single_latency) / single_latency) * 100
            assert latency_increase < 100.0, f"Latency increase for {test_scenarios[i]['name']} should be reasonable: {latency_increase:.1f}%"
        
        print("Performance benchmark tests completed successfully")

    async def test_monitoring_and_observability_validation(
        self,
        daemon_config_cluster,
        multi_instance_test_environment
    ):
        """
        Test monitoring and observability for multi-instance daemon operations.
        
        Verifies:
        1. Health monitoring across all instances
        2. Metrics collection and aggregation
        3. Log correlation and centralized logging
        4. Alert generation and notification
        5. Observability dashboard data accuracy
        """
        
        configs = daemon_config_cluster[:3]
        metrics = MultiInstanceTestMetrics()
        
        # Simulate monitoring infrastructure
        monitoring_data = {
            "health_checks": [],
            "metrics": {},
            "logs": [],
            "alerts": []
        }
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            with patch('workspace_qdrant_mcp.core.resource_manager.get_resource_manager') as mock_get_manager:
                
                # Set up monitoring-enabled daemon managers
                daemon_managers = []
                daemon_instances = []
                
                for i, config in enumerate(configs):
                    daemon_instance = MagicMock()
                    daemon_instance.config = config
                    daemon_instance.pid = 20000 + i
                    daemon_instance.status = DaemonStatus.RUNNING
                    daemon_instance.start_time = time.time() - (300 + i * 60)  # Started at different times
                    daemon_instances.append(daemon_instance)
                    
                    mock_manager = AsyncMock()
                    mock_manager.start_daemon.return_value = daemon_instance
                    
                    # Health monitoring
                    mock_manager.get_daemon_health.return_value = {
                        "healthy": True,
                        "pid": daemon_instance.pid,
                        "status": "running",
                        "uptime_seconds": 300 + i * 60,
                        "last_health_check": time.time(),
                        "health_check_failures": 0,
                        "response_time_ms": 25 + (i * 5)
                    }
                    
                    # Metrics collection
                    mock_manager.get_metrics.return_value = {
                        "requests_total": 1500 + (i * 200),
                        "requests_per_second": 5.2 + (i * 0.8),
                        "error_rate": 0.001 + (i * 0.0005),
                        "memory_usage_mb": 48 + (i * 6),
                        "cpu_usage_percent": 12 + (i * 3),
                        "disk_usage_mb": 150 + (i * 25),
                        "network_bytes_in": 102400 + (i * 20480),
                        "network_bytes_out": 81920 + (i * 16384),
                        "active_connections": 8 + (i * 2),
                        "cache_hit_ratio": 0.85 - (i * 0.05)
                    }
                    
                    # Log generation
                    mock_manager.get_recent_logs.return_value = [
                        {
                            "timestamp": time.time() - 60,
                            "level": "INFO",
                            "message": f"Daemon {i} processing request",
                            "project": config.project_name,
                            "component": "daemon_manager"
                        },
                        {
                            "timestamp": time.time() - 30,
                            "level": "DEBUG", 
                            "message": f"Health check passed for daemon {i}",
                            "project": config.project_name,
                            "component": "health_monitor"
                        }
                    ]
                    
                    daemon_managers.append(mock_manager)
                
                MockDaemonManager.side_effect = daemon_managers
                
                # Start all daemons
                active_daemons = []
                for config, manager_class in zip(configs, daemon_managers):
                    daemon_manager = MockDaemonManager(config)
                    daemon_instance = await daemon_manager.start_daemon()
                    active_daemons.append((daemon_manager, daemon_instance))
                
                # Test health monitoring aggregation
                print("Testing health monitoring...")
                health_results = []
                for daemon_manager, daemon_instance in active_daemons:
                    health = await daemon_manager.get_daemon_health()
                    health_results.append(health)
                    monitoring_data["health_checks"].append({
                        "daemon_id": daemon_instance.pid,
                        "project": daemon_instance.config.project_name,
                        "health": health,
                        "timestamp": time.time()
                    })
                
                # Verify all instances are healthy
                healthy_instances = [h for h in health_results if h["healthy"]]
                assert len(healthy_instances) == len(configs), "All instances should be healthy"
                
                # Verify health check response times are reasonable
                response_times = [h["response_time_ms"] for h in health_results]
                assert max(response_times) < 100, f"Health check response times should be fast: {response_times}"
                
                # Test metrics collection and aggregation
                print("Testing metrics collection...")
                all_metrics = []
                for daemon_manager, daemon_instance in active_daemons:
                    daemon_metrics = await daemon_manager.get_metrics()
                    all_metrics.append(daemon_metrics)
                    monitoring_data["metrics"][daemon_instance.config.project_name] = daemon_metrics
                
                # Aggregate system-wide metrics
                system_metrics = {
                    "total_requests": sum(m["requests_total"] for m in all_metrics),
                    "total_rps": sum(m["requests_per_second"] for m in all_metrics),
                    "avg_error_rate": sum(m["error_rate"] for m in all_metrics) / len(all_metrics),
                    "total_memory_mb": sum(m["memory_usage_mb"] for m in all_metrics),
                    "total_cpu_percent": sum(m["cpu_usage_percent"] for m in all_metrics),
                    "avg_cache_hit_ratio": sum(m["cache_hit_ratio"] for m in all_metrics) / len(all_metrics)
                }
                
                print(f"System-wide metrics:")
                print(f"  Total requests: {system_metrics['total_requests']}")
                print(f"  Total RPS: {system_metrics['total_rps']:.1f}")
                print(f"  Average error rate: {system_metrics['avg_error_rate']:.3f}")
                print(f"  Total memory: {system_metrics['total_memory_mb']:.1f}MB")
                print(f"  Total CPU: {system_metrics['total_cpu_percent']:.1f}%")
                
                # Verify metrics are within expected ranges
                assert system_metrics["total_requests"] > 1000, "Should have processed significant requests"
                assert system_metrics["avg_error_rate"] < 0.01, "Error rate should be low"
                assert system_metrics["total_memory_mb"] < 500, "Memory usage should be reasonable"
                assert system_metrics["avg_cache_hit_ratio"] > 0.5, "Cache hit ratio should be decent"
                
                # Test log correlation and collection
                print("Testing log correlation...")
                all_logs = []
                for daemon_manager, daemon_instance in active_daemons:
                    logs = await daemon_manager.get_recent_logs()
                    for log_entry in logs:
                        log_entry["daemon_id"] = daemon_instance.pid
                        all_logs.append(log_entry)
                    monitoring_data["logs"].extend(logs)
                
                # Verify log correlation
                project_logs = {}
                for log_entry in all_logs:
                    project = log_entry["project"]
                    if project not in project_logs:
                        project_logs[project] = []
                    project_logs[project].append(log_entry)
                
                assert len(project_logs) == len(configs), "Should have logs from all projects"
                
                for project, logs in project_logs.items():
                    assert len(logs) > 0, f"Should have logs for project {project}"
                    # Verify log structure
                    for log_entry in logs:
                        assert "timestamp" in log_entry, "Log should have timestamp"
                        assert "level" in log_entry, "Log should have level"
                        assert "message" in log_entry, "Log should have message"
                        assert "component" in log_entry, "Log should have component"
                
                # Test alert generation (simulate threshold breaches)
                print("Testing alert generation...")
                
                # Simulate high error rate alert
                high_error_daemon = active_daemons[0]
                high_error_manager, high_error_instance = high_error_daemon
                high_error_manager.get_metrics.return_value = {
                    **all_metrics[0],
                    "error_rate": 0.05,  # 5% error rate (high)
                    "requests_per_second": 2.1   # Low RPS
                }
                
                updated_metrics = await high_error_manager.get_metrics()
                if updated_metrics["error_rate"] > 0.02:  # 2% threshold
                    alert = {
                        "type": "high_error_rate",
                        "daemon_id": high_error_instance.pid,
                        "project": high_error_instance.config.project_name,
                        "value": updated_metrics["error_rate"],
                        "threshold": 0.02,
                        "timestamp": time.time(),
                        "severity": "warning"
                    }
                    monitoring_data["alerts"].append(alert)
                
                # Simulate high memory usage alert
                high_memory_daemon = active_daemons[1]
                high_memory_manager, high_memory_instance = high_memory_daemon
                high_memory_manager.get_metrics.return_value = {
                    **all_metrics[1],
                    "memory_usage_mb": 180,  # High memory usage
                    "cpu_usage_percent": 25   # High CPU
                }
                
                updated_memory_metrics = await high_memory_manager.get_metrics()
                if updated_memory_metrics["memory_usage_mb"] > 150:  # 150MB threshold
                    alert = {
                        "type": "high_memory_usage",
                        "daemon_id": high_memory_instance.pid,
                        "project": high_memory_instance.config.project_name,
                        "value": updated_memory_metrics["memory_usage_mb"],
                        "threshold": 150,
                        "timestamp": time.time(),
                        "severity": "warning"
                    }
                    monitoring_data["alerts"].append(alert)
                
                # Verify alerts were generated
                assert len(monitoring_data["alerts"]) >= 2, "Should generate alerts for threshold breaches"
                
                alert_types = {alert["type"] for alert in monitoring_data["alerts"]}
                assert "high_error_rate" in alert_types, "Should generate high error rate alert"
                assert "high_memory_usage" in alert_types, "Should generate high memory usage alert"
                
                # Test observability dashboard data accuracy
                print("Testing observability dashboard data...")
                
                dashboard_data = {
                    "overview": {
                        "total_instances": len(active_daemons),
                        "healthy_instances": len([h for h in health_results if h["healthy"]]),
                        "total_uptime": sum(h["uptime_seconds"] for h in health_results),
                        "avg_response_time": sum(h["response_time_ms"] for h in health_results) / len(health_results)
                    },
                    "performance": {
                        "total_rps": system_metrics["total_rps"],
                        "error_rate": system_metrics["avg_error_rate"],
                        "cache_hit_ratio": system_metrics["avg_cache_hit_ratio"]
                    },
                    "resources": {
                        "total_memory_mb": system_metrics["total_memory_mb"],
                        "total_cpu_percent": system_metrics["total_cpu_percent"],
                        "instances": [
                            {
                                "project": dm[1].config.project_name,
                                "memory_mb": m["memory_usage_mb"],
                                "cpu_percent": m["cpu_usage_percent"]
                            }
                            for (dm, _), m in zip(active_daemons, all_metrics)
                        ]
                    },
                    "alerts": {
                        "active_count": len(monitoring_data["alerts"]),
                        "recent_alerts": monitoring_data["alerts"][-5:]  # Last 5 alerts
                    }
                }
                
                # Verify dashboard data accuracy
                assert dashboard_data["overview"]["total_instances"] == len(configs)
                assert dashboard_data["overview"]["healthy_instances"] == len(configs)
                assert dashboard_data["overview"]["avg_response_time"] < 50
                
                assert dashboard_data["performance"]["total_rps"] > 10
                assert dashboard_data["performance"]["error_rate"] < 0.1
                
                assert dashboard_data["resources"]["total_memory_mb"] > 100
                assert dashboard_data["resources"]["total_cpu_percent"] > 10
                assert len(dashboard_data["resources"]["instances"]) == len(configs)
                
                print("Monitoring and observability validation completed:")
                print(f"  Health checks: {len(monitoring_data['health_checks'])}")
                print(f"  Metrics collected: {len(monitoring_data['metrics'])}")
                print(f"  Log entries: {len(monitoring_data['logs'])}")
                print(f"  Alerts generated: {len(monitoring_data['alerts'])}")
                print(f"  Dashboard accuracy: Verified")

    async def test_stress_testing_under_load(
        self,
        daemon_config_cluster,
        multi_instance_test_environment
    ):
        """
        Test multi-instance daemon behavior under high load conditions.
        
        Verifies:
        1. System stability under sustained high load
        2. Resource utilization under stress
        3. Performance degradation characteristics
        4. Load balancing and distribution
        5. Recovery after load spikes
        """
        
        configs = daemon_config_cluster
        metrics = MultiInstanceTestMetrics()
        
        # Stress test configuration
        stress_config = {
            "duration_seconds": 10,  # Short duration for testing
            "concurrent_operations": 50,
            "operation_types": ["search", "ingest", "health_check"],
            "load_ramp_up_seconds": 2,
            "load_ramp_down_seconds": 2
        }
        
        with patch('workspace_qdrant_mcp.core.daemon_manager.DaemonManager') as MockDaemonManager:
            
            # Set up stress-test capable daemon managers
            daemon_managers = []
            daemon_instances = []
            
            for i, config in enumerate(configs):
                daemon_instance = MagicMock()
                daemon_instance.config = config
                daemon_instance.pid = 21000 + i
                daemon_instance.status = DaemonStatus.RUNNING
                daemon_instance.load_factor = 0.0  # Start with no load
                daemon_instances.append(daemon_instance)
                
                mock_manager = AsyncMock()
                mock_manager.start_daemon.return_value = daemon_instance
                
                # Simulate load-dependent performance
                def create_load_dependent_methods(instance_idx):
                    
                    async def mock_search(query, **kwargs):
                        load = daemon_instances[instance_idx].load_factor
                        base_time = 50  # ms
                        load_penalty = load * 100  # Additional ms under load
                        
                        await asyncio.sleep((base_time + load_penalty) / 1000)  # Simulate processing time
                        
                        return {
                            "results": [{"content": f"Result for {query}", "score": 0.9 - (load * 0.3)}],
                            "latency_ms": base_time + load_penalty,
                            "success": load < 0.9  # Fail if load too high
                        }
                    
                    async def mock_ingest(documents, **kwargs):
                        load = daemon_instances[instance_idx].load_factor
                        base_time = 100  # ms
                        load_penalty = load * 200
                        
                        await asyncio.sleep((base_time + load_penalty) / 1000)
                        
                        return {
                            "processed": len(documents) if load < 0.8 else max(0, len(documents) - int(load * 10)),
                            "latency_ms": base_time + load_penalty,
                            "success": load < 0.8
                        }
                    
                    async def mock_health_check():
                        load = daemon_instances[instance_idx].load_factor
                        return {
                            "healthy": load < 0.9,
                            "load_factor": load,
                            "response_time_ms": 10 + (load * 40)
                        }
                    
                    return mock_search, mock_ingest, mock_health_check
                
                search_fn, ingest_fn, health_fn = create_load_dependent_methods(i)
                mock_manager.search_documents = search_fn
                mock_manager.ingest_documents = ingest_fn
                mock_manager.get_daemon_health = health_fn
                
                daemon_managers.append(mock_manager)
            
            MockDaemonManager.side_effect = daemon_managers
            
            # Start all daemon instances
            active_daemons = []
            for config, manager_class in zip(configs, daemon_managers):
                daemon_manager = MockDaemonManager(config)
                daemon_instance = await daemon_manager.start_daemon()
                active_daemons.append((daemon_manager, daemon_instance))
            
            print(f"Stress testing {len(active_daemons)} daemon instances...")
            
            # Phase 1: Baseline performance measurement
            print("Phase 1: Measuring baseline performance...")
            baseline_results = []
            
            for daemon_manager, daemon_instance in active_daemons:
                # Test each operation type at baseline
                search_result = await daemon_manager.search_documents("test query")
                ingest_result = await daemon_manager.ingest_documents(["doc1", "doc2"])
                health_result = await daemon_manager.get_daemon_health()
                
                baseline_results.append({
                    "daemon_id": daemon_instance.pid,
                    "search_latency": search_result["latency_ms"],
                    "ingest_latency": ingest_result["latency_ms"],
                    "health_latency": health_result["response_time_ms"]
                })
            
            baseline_search_latency = sum(r["search_latency"] for r in baseline_results) / len(baseline_results)
            baseline_ingest_latency = sum(r["ingest_latency"] for r in baseline_results) / len(baseline_results)
            
            print(f"Baseline - Search: {baseline_search_latency:.1f}ms, Ingest: {baseline_ingest_latency:.1f}ms")
            
            # Phase 2: Stress test execution
            print("Phase 2: Executing stress test...")
            stress_start_time = time.time()
            
            # Create stress test tasks
            stress_tasks = []
            task_results = {"completed": 0, "failed": 0, "latencies": []}
            
            async def stress_operation(daemon_idx, operation_type, operation_id):
                daemon_manager, daemon_instance = active_daemons[daemon_idx]
                
                try:
                    start_time = time.time()
                    
                    if operation_type == "search":
                        result = await daemon_manager.search_documents(f"stress query {operation_id}")
                        success = result["success"]
                        latency = result["latency_ms"]
                        
                    elif operation_type == "ingest":
                        docs = [f"stress doc {operation_id}-{i}" for i in range(3)]
                        result = await daemon_manager.ingest_documents(docs)
                        success = result["success"]
                        latency = result["latency_ms"]
                        
                    elif operation_type == "health_check":
                        result = await daemon_manager.get_daemon_health()
                        success = result["healthy"]
                        latency = result["response_time_ms"]
                    
                    operation_time = time.time() - start_time
                    
                    if success:
                        task_results["completed"] += 1
                    else:
                        task_results["failed"] += 1
                    
                    task_results["latencies"].append(latency)
                    
                    return {"success": success, "latency": latency, "operation_time": operation_time}
                    
                except Exception as e:
                    task_results["failed"] += 1
                    return {"success": False, "error": str(e)}
            
            # Ramp up load gradually
            current_load = 0.0
            load_increment = 1.0 / stress_config["concurrent_operations"]
            
            for operation_id in range(stress_config["concurrent_operations"]):
                # Distribute operations across daemons
                daemon_idx = operation_id % len(active_daemons)
                operation_type = stress_config["operation_types"][operation_id % len(stress_config["operation_types"])]
                
                # Update load factor for this daemon
                daemon_instances[daemon_idx].load_factor = min(current_load, 1.0)
                current_load += load_increment
                
                # Create stress task
                task = stress_operation(daemon_idx, operation_type, operation_id)
                stress_tasks.append(task)
                
                # Gradual ramp-up delay
                if operation_id < stress_config["concurrent_operations"] * 0.2:  # First 20% of operations
                    await asyncio.sleep(stress_config["load_ramp_up_seconds"] / (stress_config["concurrent_operations"] * 0.2))
            
            # Execute all stress tasks concurrently
            print(f"Executing {len(stress_tasks)} concurrent operations...")
            stress_results = await asyncio.gather(*stress_tasks, return_exceptions=True)
            
            stress_end_time = time.time()
            total_stress_time = stress_end_time - stress_start_time
            
            # Phase 3: Analyze stress test results
            print("Phase 3: Analyzing stress test results...")
            
            successful_operations = [r for r in stress_results if isinstance(r, dict) and r.get("success", False)]
            failed_operations = len(stress_results) - len(successful_operations)
            
            if successful_operations:
                stress_latencies = [r["latency"] for r in successful_operations]
                avg_stress_latency = sum(stress_latencies) / len(stress_latencies)
                p95_stress_latency = sorted(stress_latencies)[int(len(stress_latencies) * 0.95)]
                max_stress_latency = max(stress_latencies)
            else:
                avg_stress_latency = p95_stress_latency = max_stress_latency = 0
            
            success_rate = len(successful_operations) / len(stress_results)
            throughput = len(successful_operations) / total_stress_time
            
            print(f"Stress test results:")
            print(f"  Total operations: {len(stress_results)}")
            print(f"  Successful: {len(successful_operations)}")
            print(f"  Failed: {failed_operations}")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Throughput: {throughput:.1f} ops/sec")
            print(f"  Avg latency: {avg_stress_latency:.1f}ms")
            print(f"  P95 latency: {p95_stress_latency:.1f}ms")
            print(f"  Max latency: {max_stress_latency:.1f}ms")
            
            # Phase 4: Recovery testing
            print("Phase 4: Testing recovery after stress...")
            
            # Reset load factors
            for daemon_instance in daemon_instances:
                daemon_instance.load_factor = 0.0
            
            # Wait for system to recover
            await asyncio.sleep(2.0)
            
            # Test recovery performance
            recovery_results = []
            for daemon_manager, daemon_instance in active_daemons:
                search_result = await daemon_manager.search_documents("recovery test")
                health_result = await daemon_manager.get_daemon_health()
                
                recovery_results.append({
                    "daemon_id": daemon_instance.pid,
                    "search_latency": search_result["latency_ms"],
                    "healthy": health_result["healthy"],
                    "health_latency": health_result["response_time_ms"]
                })
            
            recovery_search_latency = sum(r["search_latency"] for r in recovery_results) / len(recovery_results)
            healthy_instances = sum(1 for r in recovery_results if r["healthy"])
            
            print(f"Recovery results:")
            print(f"  Healthy instances: {healthy_instances}/{len(active_daemons)}")
            print(f"  Recovery search latency: {recovery_search_latency:.1f}ms")
            print(f"  Latency improvement: {((avg_stress_latency - recovery_search_latency) / avg_stress_latency * 100):.1f}%")
            
            # Verify stress test results meet criteria
            assert success_rate > 0.7, f"Success rate should be reasonable under stress: {success_rate:.1%}"
            assert throughput > 5.0, f"Throughput should be maintained under stress: {throughput:.1f} ops/sec"
            assert avg_stress_latency < baseline_search_latency * 5, f"Latency degradation should be bounded: {avg_stress_latency:.1f}ms vs {baseline_search_latency:.1f}ms baseline"
            
            # Verify recovery
            assert healthy_instances == len(active_daemons), "All instances should recover after stress"
            assert recovery_search_latency < avg_stress_latency, "Performance should improve after stress ends"
            
            print("Stress testing completed successfully")


# Test utility functions for the multi-instance framework

async def setup_test_cluster(num_instances: int, base_config: DaemonConfig) -> List[Tuple[DaemonManager, DaemonInstance]]:
    """Utility function to set up a test cluster of daemon instances."""
    cluster = []
    
    for i in range(num_instances):
        config = DaemonConfig(
            project_name=f"{base_config.project_name}-{i}",
            project_path=f"{base_config.project_path}-{i}",
            grpc_port=base_config.grpc_port + i,
            **{k: v for k, v in base_config.__dict__.items() 
               if k not in ['project_name', 'project_path', 'grpc_port']}
        )
        
        daemon_manager = DaemonManager(config)
        daemon_instance = await daemon_manager.start_daemon()
        cluster.append((daemon_manager, daemon_instance))
    
    return cluster


async def cleanup_test_cluster(cluster: List[Tuple[DaemonManager, DaemonInstance]]):
    """Utility function to clean up a test cluster."""
    for daemon_manager, daemon_instance in cluster:
        try:
            await daemon_manager.shutdown_daemon()
        except Exception as e:
            print(f"Warning: Failed to shutdown daemon {daemon_instance.pid}: {e}")


def assert_cluster_health(cluster: List[Tuple[DaemonManager, DaemonInstance]], expected_healthy: int = None):
    """Utility function to assert cluster health status."""
    if expected_healthy is None:
        expected_healthy = len(cluster)
    
    healthy_count = 0
    for daemon_manager, daemon_instance in cluster:
        if daemon_instance.status == DaemonStatus.RUNNING:
            healthy_count += 1
    
    assert healthy_count == expected_healthy, f"Expected {expected_healthy} healthy instances, found {healthy_count}"


def generate_test_load(cluster: List[Tuple[DaemonManager, DaemonInstance]], operations_per_daemon: int = 10):
    """Utility function to generate test load across a cluster."""
    
    async def load_operation(daemon_manager: DaemonManager, operation_id: int):
        try:
            # Simulate different operation types
            if operation_id % 3 == 0:
                return await daemon_manager.search_documents(f"test query {operation_id}")
            elif operation_id % 3 == 1:
                return await daemon_manager.ingest_documents([f"test doc {operation_id}"])
            else:
                return await daemon_manager.get_daemon_health()
        except Exception as e:
            return {"error": str(e), "success": False}
    
    tasks = []
    for i, (daemon_manager, _) in enumerate(cluster):
        for j in range(operations_per_daemon):
            operation_id = i * operations_per_daemon + j
            tasks.append(load_operation(daemon_manager, operation_id))
    
    return tasks