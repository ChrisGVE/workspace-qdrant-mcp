"""
Daemon-specific test fixtures and utilities for unit testing.

This module provides specialized fixtures for testing daemon core functionality
including DaemonManager, PriorityQueueManager, ProcessingEngine, and file watching.
"""

import asyncio
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest

from common.core.daemon_manager import (
    DaemonConfig, 
    DaemonStatus, 
    DaemonInstance, 
    DaemonManager,
    PortManager
)
from common.core.priority_queue_manager import (
    PriorityQueueManager,
    ResourceConfiguration,
    MCPActivityMetrics,
    MCPActivityLevel,
    ProcessingMode,
    QueueStatistics,
    ProcessingJob,
    PriorityCalculationContext
)
from common.core.sqlite_state_manager import (
    SQLiteStateManager,
    ProcessingPriority,
    FileProcessingStatus
)


@pytest.fixture
async def isolated_daemon_temp_dir() -> AsyncGenerator[Path, None]:
    """Create isolated temporary directory for daemon testing."""
    temp_dir = tempfile.mkdtemp(prefix="daemon_test_")
    temp_path = Path(temp_dir)
    try:
        # Create project structure
        (temp_path / "src").mkdir()
        (temp_path / "tests").mkdir()
        (temp_path / ".git").mkdir()
        (temp_path / "README.md").write_text("Test project")
        
        yield temp_path
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_daemon_config(isolated_daemon_temp_dir: Path) -> DaemonConfig:
    """Create mock daemon configuration for testing."""
    return DaemonConfig(
        project_name="test-daemon-project",
        project_path=str(isolated_daemon_temp_dir),
        project_id="test_daemon_123",
        grpc_host="127.0.0.1",
        grpc_port=50099,  # High port for testing
        qdrant_url="http://localhost:6333",
        log_level="debug",
        max_concurrent_jobs=2,
        health_check_interval=0.5,  # Fast for testing
        startup_timeout=5.0,
        shutdown_timeout=2.0,
        restart_on_failure=False,  # Disable auto-restart in tests
        max_restart_attempts=1
    )


@pytest.fixture
def mock_daemon_status() -> DaemonStatus:
    """Create mock daemon status for testing."""
    return DaemonStatus(
        pid=12345,
        state="running",
        start_time=datetime.now(timezone.utc),
        last_health_check=datetime.now(timezone.utc),
        health_status="healthy",
        restart_count=0,
        last_error=None,
        grpc_available=True
    )


@pytest.fixture
def mock_port_manager():
    """Create mock port manager for testing."""
    port_manager = Mock(spec=PortManager)
    port_manager.allocate_port = Mock(return_value=50099)
    port_manager.release_port = Mock(return_value=True)
    port_manager.is_port_allocated = Mock(return_value=False)
    port_manager.get_allocated_ports = Mock(return_value={})
    return port_manager


@pytest.fixture
async def mock_daemon_instance(mock_daemon_config, mock_daemon_status, mock_port_manager):
    """Create mock daemon instance for testing."""
    with patch('src.workspace_qdrant_mcp.core.daemon_manager.PortManager.get_instance', return_value=mock_port_manager):
        instance = Mock(spec=DaemonInstance)
        instance.config = mock_daemon_config
        instance.status = mock_daemon_status
        instance.process = None
        instance.health_task = None
        instance.shutdown_event = asyncio.Event()
        instance.log_handlers = []
        instance.temp_dir = Path(tempfile.mkdtemp(prefix="daemon_test_"))
        instance.config_file = instance.temp_dir / "daemon_config.json"
        instance.pid_file = instance.temp_dir / f"{mock_daemon_config.project_id}.pid"
        
        # Mock methods
        instance.start = AsyncMock(return_value=True)
        instance.stop = AsyncMock(return_value=True)
        instance.restart = AsyncMock(return_value=True)
        instance.health_check = AsyncMock(return_value=True)
        instance.get_status = AsyncMock(return_value={
            "config": mock_daemon_config.__dict__,
            "status": mock_daemon_status.__dict__,
            "process_info": {
                "pid": mock_daemon_status.pid,
                "running": True,
                "return_code": None
            }
        })
        instance.add_log_handler = Mock()
        
        yield instance
        
        # Cleanup
        if instance.temp_dir.exists():
            import shutil
            shutil.rmtree(instance.temp_dir, ignore_errors=True)


@pytest.fixture
async def mock_daemon_manager():
    """Create mock daemon manager for testing."""
    manager = Mock(spec=DaemonManager)
    manager.daemons = {}
    manager.shutdown_handlers = []
    
    # Mock methods
    manager.get_or_create_daemon = AsyncMock()
    manager.start_daemon = AsyncMock(return_value=True)
    manager.stop_daemon = AsyncMock(return_value=True)
    manager.get_daemon_status = AsyncMock(return_value=None)
    manager.list_daemons = AsyncMock(return_value={})
    manager.health_check_all = AsyncMock(return_value={})
    manager.shutdown_all = AsyncMock()
    manager.add_shutdown_handler = Mock()
    manager.get_system_resource_status = AsyncMock(return_value={})
    
    return manager


@pytest.fixture
def mock_resource_configuration() -> ResourceConfiguration:
    """Create mock resource configuration for testing."""
    return ResourceConfiguration(
        max_concurrent_jobs=2,
        max_memory_mb=512,
        max_cpu_percent=70,
        conservative_concurrent_jobs=1,
        conservative_memory_mb=128,
        balanced_concurrent_jobs=2,
        balanced_memory_mb=256,
        aggressive_concurrent_jobs=4,
        aggressive_memory_mb=1024,
        burst_concurrent_jobs=6,
        burst_memory_mb=2048,
        burst_duration_seconds=60,
        backpressure_threshold=0.8,
        health_check_interval=1
    )


@pytest.fixture
def mock_mcp_activity_metrics() -> MCPActivityMetrics:
    """Create mock MCP activity metrics for testing."""
    metrics = MCPActivityMetrics()
    metrics.requests_per_minute = 5.0
    metrics.active_sessions = 1
    metrics.last_request_time = datetime.now(timezone.utc)
    metrics.activity_level = MCPActivityLevel.MODERATE
    metrics.burst_detected = False
    metrics.session_start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
    metrics.total_requests = 50
    metrics.average_request_duration = 0.5
    return metrics


@pytest.fixture
async def mock_sqlite_state_manager(isolated_daemon_temp_dir: Path):
    """Create mock SQLite state manager for testing."""
    db_path = isolated_daemon_temp_dir / "test_state.db"
    
    manager = Mock(spec=SQLiteStateManager)
    manager.db_path = str(db_path)
    manager._initialized = True
    
    # Mock database operations
    manager.initialize = AsyncMock(return_value=True)
    manager.close = AsyncMock()
    manager.add_to_processing_queue = AsyncMock(return_value="queue_123")
    manager.get_next_queue_item = AsyncMock(return_value=None)
    manager.remove_from_processing_queue = AsyncMock()
    manager.mark_queue_item_processing = AsyncMock()
    manager.reschedule_queue_item = AsyncMock()
    manager.get_queue_stats = AsyncMock(return_value={
        ProcessingPriority.LOW.value: 2,
        ProcessingPriority.NORMAL.value: 3,
        ProcessingPriority.HIGH.value: 1,
        ProcessingPriority.URGENT.value: 0
    })
    manager.start_file_processing = AsyncMock()
    manager.complete_file_processing = AsyncMock()
    manager.get_file_processing_record = AsyncMock(return_value=None)
    manager.clear_queue = AsyncMock(return_value=0)
    
    return manager


@pytest.fixture
async def mock_priority_queue_manager(
    mock_sqlite_state_manager, 
    mock_resource_configuration,
    mock_mcp_activity_metrics
):
    """Create mock priority queue manager for testing."""
    manager = Mock(spec=PriorityQueueManager)
    manager.state_manager = mock_sqlite_state_manager
    manager.incremental_processor = None
    manager.resource_config = mock_resource_configuration
    manager.mcp_activity = mock_mcp_activity_metrics
    manager.current_project_root = None
    manager.recent_files = set()
    manager.processing_mode = ProcessingMode.BALANCED
    manager.executor = None
    manager.active_jobs = {}
    manager.job_semaphore = None
    manager.statistics = QueueStatistics()
    manager.processing_history = []
    manager.health_metrics = {}
    manager._initialized = True
    manager._shutdown_event = asyncio.Event()
    manager._monitoring_task = None
    manager._activity_detection_task = None
    manager.priority_calculation_hooks = []
    manager.processing_hooks = []
    manager.monitoring_hooks = []
    
    # Mock methods
    manager.initialize = AsyncMock(return_value=True)
    manager.shutdown = AsyncMock()
    manager.enqueue_file = AsyncMock(return_value="queue_123")
    manager.process_next_batch = AsyncMock(return_value=[])
    manager.get_queue_status = AsyncMock(return_value={
        "initialized": True,
        "processing_mode": "balanced",
        "statistics": {},
        "active_jobs": 0
    })
    manager.get_processing_context = Mock()
    manager.add_priority_calculation_hook = Mock()
    manager.add_processing_hook = Mock()
    manager.add_monitoring_hook = Mock()
    manager.set_current_project_root = Mock()
    manager.clear_queue = AsyncMock(return_value=0)
    manager.get_health_status = AsyncMock(return_value={
        "health_status": "healthy",
        "system_metrics": {},
        "queue_statistics": {},
        "resource_status": {},
        "mcp_activity": {}
    })
    
    return manager


@pytest.fixture
def mock_processing_job() -> ProcessingJob:
    """Create mock processing job for testing."""
    return ProcessingJob(
        queue_id="test_queue_123",
        file_path="/tmp/test_file.py",
        collection="test-collection",
        priority=ProcessingPriority.NORMAL,
        calculated_score=45.0,
        created_at=datetime.now(timezone.utc),
        scheduled_at=None,
        attempts=0,
        max_attempts=3,
        timeout_seconds=300,
        metadata={"test": True},
        processing_context=None
    )


@pytest.fixture
def mock_priority_calculation_context(isolated_daemon_temp_dir: Path) -> PriorityCalculationContext:
    """Create mock priority calculation context for testing."""
    test_file = isolated_daemon_temp_dir / "test.py"
    test_file.write_text("def test(): pass")
    
    return PriorityCalculationContext(
        file_path=str(test_file),
        collection="test-collection",
        mcp_activity=MCPActivityMetrics(),
        current_project_root=str(isolated_daemon_temp_dir),
        file_modification_time=datetime.now(timezone.utc),
        file_size=17,  # Size of "def test(): pass"
        is_user_triggered=False,
        is_current_project=True,
        is_recently_modified=True,
        has_dependencies=False,
        processing_history={}
    )


@pytest.fixture
def mock_file_watcher():
    """Create mock file watcher for testing."""
    watcher = Mock()
    watcher.start = AsyncMock()
    watcher.stop = AsyncMock()
    watcher.add_watch = Mock()
    watcher.remove_watch = Mock()
    watcher.get_watched_paths = Mock(return_value=[])
    watcher.is_watching = Mock(return_value=False)
    
    # Mock event handling
    watcher.on_created = AsyncMock()
    watcher.on_modified = AsyncMock()
    watcher.on_deleted = AsyncMock()
    watcher.on_moved = AsyncMock()
    
    return watcher


@pytest.fixture
async def mock_file_event():
    """Create mock file system event for testing."""
    event = Mock()
    event.src_path = "/tmp/test_file.py"
    event.event_type = "modified"
    event.is_directory = False
    event.timestamp = time.time()
    return event


@pytest.fixture
def mock_processing_engine():
    """Create mock processing engine for testing."""
    engine = Mock()
    engine.start = AsyncMock()
    engine.stop = AsyncMock()
    engine.process_file = AsyncMock()
    engine.process_batch = AsyncMock(return_value=[])
    engine.get_status = AsyncMock(return_value={
        "running": True,
        "active_jobs": 0,
        "processed_files": 0
    })
    
    return engine


# Helper functions for daemon testing

def create_test_file_tree(base_path: Path, structure: Dict[str, any]) -> Dict[str, Path]:
    """
    Create a test file tree structure.
    
    Args:
        base_path: Base directory path
        structure: Dictionary describing file structure
                  {"file.txt": "content", "dir": {"subfile.py": "content"}}
    
    Returns:
        Dictionary mapping relative paths to absolute Path objects
    """
    created_files = {}
    
    def _create_recursive(current_path: Path, items: Dict[str, any], prefix: str = ""):
        for name, content in items.items():
            item_path = current_path / name
            relative_path = f"{prefix}/{name}" if prefix else name
            
            if isinstance(content, dict):
                # Directory
                item_path.mkdir(exist_ok=True)
                _create_recursive(item_path, content, relative_path)
            else:
                # File
                item_path.write_text(str(content))
                created_files[relative_path] = item_path
    
    _create_recursive(base_path, structure)
    return created_files


async def wait_for_condition(condition_func, timeout=5.0, interval=0.1):
    """
    Wait for a condition to become true.
    
    Args:
        condition_func: Callable that returns True when condition is met
        timeout: Maximum time to wait in seconds
        interval: Check interval in seconds
    
    Raises:
        asyncio.TimeoutError: If condition not met within timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return
        await asyncio.sleep(interval)
    
    raise asyncio.TimeoutError(f"Condition not met within {timeout} seconds")


def assert_daemon_config_valid(config: DaemonConfig):
    """Assert that daemon configuration is valid."""
    assert config.project_name
    assert config.project_path
    assert config.grpc_host
    assert 1024 <= config.grpc_port <= 65535
    assert config.health_check_interval > 0
    assert config.startup_timeout > 0
    assert config.shutdown_timeout > 0
    assert config.max_restart_attempts >= 0


def assert_processing_job_valid(job: ProcessingJob):
    """Assert that processing job is valid."""
    assert job.queue_id
    assert job.file_path
    assert job.collection
    assert job.priority in ProcessingPriority
    assert job.calculated_score >= 0
    assert job.attempts >= 0
    assert job.max_attempts > 0
    assert job.timeout_seconds > 0


class MockAsyncContextManager:
    """Helper class for creating async context managers in tests."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.entered = False
        self.exited = False
        self.exception_info = None
    
    async def __aenter__(self):
        self.entered = True
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        self.exception_info = (exc_type, exc_val, exc_tb)
        return False


class DaemonTestHelper:
    """Helper class for daemon testing operations."""
    
    @staticmethod
    def create_daemon_binary_mock(temp_dir: Path) -> Path:
        """Create a mock daemon binary for testing."""
        binary_dir = temp_dir / "target" / "release"
        binary_dir.mkdir(parents=True)
        binary_path = binary_dir / "memexd"
        binary_path.write_text("#!/bin/bash\necho 'Mock daemon'\n")
        binary_path.chmod(0o755)
        return binary_path
    
    @staticmethod
    async def simulate_daemon_startup(daemon_instance: Mock, success: bool = True):
        """Simulate daemon startup process."""
        if success:
            daemon_instance.status.state = "running"
            daemon_instance.status.pid = 12345
            daemon_instance.status.grpc_available = True
            daemon_instance.start.return_value = True
        else:
            daemon_instance.status.state = "failed"
            daemon_instance.status.last_error = "Startup failed"
            daemon_instance.start.return_value = False
    
    @staticmethod
    async def simulate_health_check(daemon_instance: Mock, healthy: bool = True):
        """Simulate daemon health check."""
        if healthy:
            daemon_instance.status.health_status = "healthy"
            daemon_instance.status.last_health_check = datetime.now(timezone.utc)
            daemon_instance.health_check.return_value = True
        else:
            daemon_instance.status.health_status = "unhealthy"
            daemon_instance.health_check.return_value = False


# Pytest markers for daemon tests
pytest_daemon_markers = [
    "daemon_unit: Daemon unit tests",
    "daemon_lifecycle: Daemon lifecycle tests", 
    "daemon_config: Daemon configuration tests",
    "daemon_monitoring: Daemon monitoring tests",
    "queue_unit: Priority queue unit tests",
    "queue_processing: Queue processing tests",
    "queue_priority: Priority calculation tests",
    "file_watching: File watching tests",
    "processing_engine: Processing engine tests"
]