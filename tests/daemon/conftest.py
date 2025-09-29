"""
Daemon test fixtures and configuration.

Provides fixtures specific to daemon component testing including
gRPC mocking, Rust process management, and daemon lifecycle control.
"""

import pytest
import asyncio
from typing import Optional, Dict, Any


@pytest.fixture
def daemon_config() -> Dict[str, Any]:
    """Provide test configuration for daemon."""
    return {
        "grpc": {
            "host": "127.0.0.1",
            "port": 50051,
            "max_message_size": 4 * 1024 * 1024,  # 4MB
        },
        "processing": {
            "batch_size": 100,
            "worker_threads": 2,
            "queue_capacity": 1000,
        },
        "watching": {
            "debounce_ms": 500,
            "recursive": True,
            "ignore_patterns": ["*.tmp", ".git/*", "__pycache__/*"],
        },
    }


@pytest.fixture
async def mock_daemon_client():
    """Provide mock daemon client for testing."""

    class MockDaemonClient:
        """Mock daemon client for testing without actual Rust daemon."""

        def __init__(self):
            self.connected = False
            self.processed_files = []
            self.status = {"health": "ok", "uptime_seconds": 0}

        async def connect(self):
            """Mock connection to daemon."""
            await asyncio.sleep(0.01)  # Simulate connection delay
            self.connected = True
            return True

        async def disconnect(self):
            """Mock disconnection from daemon."""
            await asyncio.sleep(0.01)
            self.connected = False

        async def process_file(self, file_path: str) -> Dict[str, Any]:
            """Mock file processing."""
            self.processed_files.append(file_path)
            return {
                "success": True,
                "file_path": file_path,
                "chunks_created": 5,
                "processing_time_ms": 100,
            }

        async def get_status(self) -> Dict[str, Any]:
            """Mock status retrieval."""
            return self.status

        async def health_check(self) -> bool:
            """Mock health check."""
            return self.connected

    client = MockDaemonClient()
    yield client

    if client.connected:
        await client.disconnect()


@pytest.fixture
def daemon_process_manager():
    """Provide daemon process lifecycle management."""

    class DaemonProcessManager:
        """Manages daemon process lifecycle for testing."""

        def __init__(self):
            self.process: Optional[Any] = None
            self.is_running = False

        async def start(self, config: Optional[Dict[str, Any]] = None):
            """Start daemon process (mocked for now)."""
            # TODO: Implement actual daemon process start when Rust daemon is ready
            await asyncio.sleep(0.1)  # Simulate startup
            self.is_running = True
            return True

        async def stop(self):
            """Stop daemon process."""
            if self.is_running:
                await asyncio.sleep(0.1)  # Simulate shutdown
                self.is_running = False

        def get_port(self) -> int:
            """Get daemon gRPC port."""
            return 50051

        def get_pid(self) -> Optional[int]:
            """Get daemon process ID."""
            return None if not self.is_running else 12345

    manager = DaemonProcessManager()
    yield manager

    if manager.is_running:
        asyncio.create_task(manager.stop())


@pytest.fixture
def grpc_channel_config() -> Dict[str, Any]:
    """Provide gRPC channel configuration for testing."""
    return {
        "target": "localhost:50051",
        "options": [
            ("grpc.max_send_message_length", 4 * 1024 * 1024),
            ("grpc.max_receive_message_length", 4 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
        ],
        "compression": "gzip",
    }


@pytest.fixture
def sample_watch_events():
    """Provide sample file watch events for testing."""
    return [
        {
            "event_type": "created",
            "path": "/test/project/src/new_file.py",
            "timestamp": "2024-01-01T00:00:00Z",
        },
        {
            "event_type": "modified",
            "path": "/test/project/docs/README.md",
            "timestamp": "2024-01-01T00:01:00Z",
        },
        {
            "event_type": "deleted",
            "path": "/test/project/old_file.txt",
            "timestamp": "2024-01-01T00:02:00Z",
        },
    ]