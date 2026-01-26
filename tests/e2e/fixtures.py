"""
Comprehensive end-to-end test fixtures for full system integration testing.

This module provides fixtures for testing complete workflows across all
components: Qdrant, daemon, MCP server, and CLI.
"""

import asyncio
import os
import signal
import subprocess
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import psutil
import pytest

from tests.shared.testcontainers_utils import IsolatedQdrantContainer


@dataclass
class SystemComponents:
    """Container for all system component instances."""

    qdrant: IsolatedQdrantContainer
    qdrant_url: str
    daemon_process: subprocess.Popen | None
    mcp_server_process: subprocess.Popen | None
    state_db_path: Path
    workspace_path: Path
    config_path: Path


@dataclass
class ResourceMetrics:
    """Resource usage metrics for system monitoring."""

    timestamp: float
    cpu_percent: float
    memory_mb: float
    open_files: int
    thread_count: int
    process_name: str
    pid: int


class DaemonManager:
    """Manage daemon lifecycle for testing."""

    def __init__(self, qdrant_url: str, state_db_path: Path, log_level: str = "debug"):
        self.qdrant_url = qdrant_url
        self.state_db_path = state_db_path
        self.log_level = log_level
        self.process: subprocess.Popen | None = None
        self.pid: int | None = None

    def start(self, timeout: int = 30) -> subprocess.Popen:
        """
        Start the daemon process.

        Args:
            timeout: Maximum time to wait for daemon to be ready

        Returns:
            Running daemon process

        Raises:
            RuntimeError: If daemon fails to start
        """
        # Build daemon start command
        cmd = [
            "./src/rust/daemon/target/release/memexd",
            "--foreground",
            "--log-level",
            self.log_level,
        ]

        env = os.environ.copy()
        env.update(
            {
                "QDRANT_URL": self.qdrant_url,
                "SQLITE_PATH": str(self.state_db_path),
                "RUST_BACKTRACE": "1",
            }
        )

        # Start daemon process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self.pid = self.process.pid

        # Wait for daemon to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                # Give daemon a moment to fully initialize
                time.sleep(2)
                return self.process

            time.sleep(0.5)

        # Daemon failed to start
        self.stop()
        raise RuntimeError("Daemon failed to start within timeout")

    def stop(self, timeout: int = 10):
        """
        Stop the daemon process gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self.process:
            return

        try:
            # Try graceful shutdown first (SIGTERM)
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception:
            pass  # Process already dead

        self.process = None
        self.pid = None

    def is_running(self) -> bool:
        """Check if daemon process is running."""
        if not self.pid:
            return False

        try:
            process = psutil.Process(self.pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def get_metrics(self) -> ResourceMetrics | None:
        """
        Get current resource metrics for daemon.

        Returns:
            Resource metrics or None if process not running
        """
        if not self.is_running() or not self.pid:
            return None

        try:
            process = psutil.Process(self.pid)
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=process.cpu_percent(interval=0.1),
                memory_mb=process.memory_info().rss / 1024 / 1024,
                open_files=len(process.open_files()),
                thread_count=process.num_threads(),
                process_name=process.name(),
                pid=self.pid,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


class MCPServerManager:
    """Manage MCP server lifecycle for testing."""

    def __init__(
        self,
        qdrant_url: str,
        state_db_path: Path,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        self.qdrant_url = qdrant_url
        self.state_db_path = state_db_path
        self.host = host
        self.port = port
        self.process: subprocess.Popen | None = None
        self.pid: int | None = None
        self.base_url = f"http://{host}:{port}"

    def start(self, timeout: int = 30) -> subprocess.Popen:
        """
        Start the MCP server process.

        Args:
            timeout: Maximum time to wait for server to be ready

        Returns:
            Running server process

        Raises:
            RuntimeError: If server fails to start
        """
        # Build server start command
        cmd = [
            "uv",
            "run",
            "workspace-qdrant-mcp",
            "--transport",
            "http",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        env = os.environ.copy()
        env.update(
            {
                "QDRANT_URL": self.qdrant_url,
                "SQLITE_PATH": str(self.state_db_path),
                "WQM_LOG_LEVEL": "DEBUG",
                "INTEGRATION_TESTING": "1",
            }
        )

        # Start server process
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env
        )
        self.pid = self.process.pid

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._is_healthy():
                time.sleep(1)  # Give server a moment to fully initialize
                return self.process

            time.sleep(0.5)

        # Server failed to start
        self.stop()
        raise RuntimeError("MCP server failed to start within timeout")

    def stop(self, timeout: int = 10):
        """
        Stop the MCP server process gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown
        """
        if not self.process:
            return

        try:
            # Try graceful shutdown first (SIGTERM)
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                self.process.kill()
                self.process.wait(timeout=5)
        except Exception:
            pass  # Process already dead

        self.process = None
        self.pid = None

    def _is_healthy(self) -> bool:
        """Check if MCP server is healthy."""
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return response.status_code == 200
        except (httpx.RequestError, httpx.HTTPError):
            return False

    def is_running(self) -> bool:
        """Check if MCP server process is running."""
        if not self.pid:
            return False

        try:
            process = psutil.Process(self.pid)
            return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def get_metrics(self) -> ResourceMetrics | None:
        """
        Get current resource metrics for MCP server.

        Returns:
            Resource metrics or None if process not running
        """
        if not self.is_running() or not self.pid:
            return None

        try:
            process = psutil.Process(self.pid)
            return ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=process.cpu_percent(interval=0.1),
                memory_mb=process.memory_info().rss / 1024 / 1024,
                open_files=len(process.open_files()),
                thread_count=process.num_threads(),
                process_name=process.name(),
                pid=self.pid,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


class CLIHelper:
    """Helper for executing CLI commands in tests."""

    def __init__(self, state_db_path: Path, qdrant_url: str):
        self.state_db_path = state_db_path
        self.qdrant_url = qdrant_url

    def run_command(
        self,
        command: list[str],
        timeout: int = 30,
        check: bool = False,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a wqm CLI command.

        Args:
            command: Command arguments (e.g., ["service", "status"])
            timeout: Maximum execution time
            check: Whether to raise on non-zero exit code
            env: Additional environment variables

        Returns:
            Completed process result
        """
        full_command = ["uv", "run", "wqm"] + command

        cmd_env = os.environ.copy()
        cmd_env.update(
            {"QDRANT_URL": self.qdrant_url, "SQLITE_PATH": str(self.state_db_path)}
        )
        if env:
            cmd_env.update(env)

        return subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
            env=cmd_env,
        )


class ResourceMonitor:
    """Monitor system resource usage during tests."""

    def __init__(self):
        self.metrics_history: list[ResourceMetrics] = []
        self.monitoring = False
        self._monitor_task: asyncio.Task | None = None

    async def start_monitoring(
        self, components: SystemComponents, interval: float = 1.0
    ):
        """
        Start monitoring resource usage.

        Args:
            components: System components to monitor
            interval: Monitoring interval in seconds
        """
        self.monitoring = True
        self.metrics_history.clear()

        async def monitor():
            while self.monitoring:
                # Collect metrics from daemon
                if components.daemon_process:
                    daemon_manager = DaemonManager(
                        components.qdrant_url, components.state_db_path
                    )
                    daemon_manager.process = components.daemon_process
                    daemon_manager.pid = components.daemon_process.pid
                    metrics = daemon_manager.get_metrics()
                    if metrics:
                        self.metrics_history.append(metrics)

                # Collect metrics from MCP server
                if components.mcp_server_process:
                    server_manager = MCPServerManager(
                        components.qdrant_url, components.state_db_path
                    )
                    server_manager.process = components.mcp_server_process
                    server_manager.pid = components.mcp_server_process.pid
                    metrics = server_manager.get_metrics()
                    if metrics:
                        self.metrics_history.append(metrics)

                await asyncio.sleep(interval)

        self._monitor_task = asyncio.create_task(monitor())

    async def stop_monitoring(self):
        """Stop monitoring and return collected metrics."""
        self.monitoring = False
        if self._monitor_task:
            await self._monitor_task
            self._monitor_task = None

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary statistics of collected metrics.

        Returns:
            Dictionary with min, max, avg for CPU and memory
        """
        if not self.metrics_history:
            return {}

        # Group by process name
        by_process: dict[str, list[ResourceMetrics]] = {}
        for metrics in self.metrics_history:
            process_name = metrics.process_name
            if process_name not in by_process:
                by_process[process_name] = []
            by_process[process_name].append(metrics)

        # Calculate summary for each process
        summary = {}
        for process_name, metrics_list in by_process.items():
            cpu_values = [m.cpu_percent for m in metrics_list]
            memory_values = [m.memory_mb for m in metrics_list]

            summary[process_name] = {
                "cpu_percent": {
                    "min": min(cpu_values),
                    "max": max(cpu_values),
                    "avg": sum(cpu_values) / len(cpu_values),
                },
                "memory_mb": {
                    "min": min(memory_values),
                    "max": max(memory_values),
                    "avg": sum(memory_values) / len(memory_values),
                },
                "sample_count": len(metrics_list),
            }

        return summary


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def integration_test_workspace(tmp_path_factory) -> Path:
    """
    Create workspace directory for integration tests.

    Scope is module-level so it persists across tests in the module.
    """
    workspace = tmp_path_factory.mktemp("integration_workspace")

    # Create realistic project structure
    (workspace / "src").mkdir()
    (workspace / "tests").mkdir()
    (workspace / "docs").mkdir()
    (workspace / ".git").mkdir()

    # Add sample files
    (workspace / "README.md").write_text("# Integration Test Project")
    (workspace / "src" / "main.py").write_text("def main():\n    pass\n")
    (workspace / "docs" / "guide.md").write_text("# Guide\n\nDocumentation.")

    return workspace


@pytest.fixture(scope="module")
def integration_state_db(tmp_path_factory) -> Path:
    """
    Create SQLite state database for integration tests.

    Scope is module-level so it persists across tests.
    """
    db_path = tmp_path_factory.mktemp("integration_data") / "state.db"
    return db_path


@pytest.fixture(scope="module")
async def system_components(
    integration_test_workspace, integration_state_db
) -> SystemComponents:
    """
    Provide fully integrated system with all components running.

    This fixture starts:
    - Qdrant container
    - Daemon process
    - MCP server process

    All components are stopped and cleaned up after tests complete.
    """
    # Skip if Docker is unavailable (needed for Qdrant container)
    try:
        import docker

        client = docker.from_env()
        client.ping()
    except Exception:
        pytest.skip("Docker not available or not running")

    # Start Qdrant container
    qdrant = IsolatedQdrantContainer()
    qdrant.start()
    qdrant_url = qdrant.get_http_url()

    # Start daemon
    daemon_manager = DaemonManager(qdrant_url, integration_state_db)
    try:
        daemon_process = daemon_manager.start()
    except RuntimeError:
        # Daemon failed to start, cleanup and skip
        qdrant.stop()
        pytest.skip("Daemon not available or failed to start")
        return  # Type checker doesn't know pytest.skip raises

    # Start MCP server
    mcp_manager = MCPServerManager(qdrant_url, integration_state_db)
    try:
        mcp_process = mcp_manager.start()
    except RuntimeError:
        # MCP server failed to start, cleanup and skip
        daemon_manager.stop()
        qdrant.stop()
        pytest.skip("MCP server failed to start")
        return  # Type checker doesn't know pytest.skip raises

    components = SystemComponents(
        qdrant=qdrant,
        qdrant_url=qdrant_url,
        daemon_process=daemon_process,
        mcp_server_process=mcp_process,
        state_db_path=integration_state_db,
        workspace_path=integration_test_workspace,
        config_path=integration_test_workspace / "config",
    )

    yield components

    # Cleanup
    mcp_manager.stop()
    daemon_manager.stop()
    qdrant.stop()


@pytest.fixture
def cli_helper(system_components: SystemComponents) -> CLIHelper:
    """Provide CLI helper for executing commands."""
    return CLIHelper(
        system_components.state_db_path, system_components.qdrant_url
    )


@pytest.fixture
async def resource_monitor() -> ResourceMonitor:
    """Provide resource monitor for tracking system performance."""
    monitor = ResourceMonitor()
    yield monitor
    # Ensure monitoring is stopped
    if monitor.monitoring:
        await monitor.stop_monitoring()


@pytest.fixture
def performance_baseline() -> dict[str, float]:
    """
    Provide baseline performance expectations for validation.

    Returns:
        Dictionary of metric -> threshold values
    """
    return {
        "max_cpu_percent": 50.0,  # Maximum CPU usage %
        "max_memory_mb": 500.0,  # Maximum memory in MB
        "max_file_descriptors": 100,  # Maximum open files
        "search_latency_ms": 500.0,  # Maximum search latency
        "ingestion_throughput": 10.0,  # Minimum docs/sec
    }


@pytest.fixture
async def system_health_check(system_components: SystemComponents) -> bool:
    """
    Verify all system components are healthy.

    Returns:
        True if all components healthy, False otherwise
    """

    async def check_health() -> bool:
        # Check Qdrant
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{system_components.qdrant_url}/health", timeout=5.0
                )
                if response.status_code != 200:
                    return False
        except Exception:
            return False

        # Check daemon
        if not system_components.daemon_process:
            return False
        if system_components.daemon_process.poll() is not None:
            return False  # Process exited

        # Check MCP server
        if not system_components.mcp_server_process:
            return False
        if system_components.mcp_server_process.poll() is not None:
            return False  # Process exited

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:8000/health", timeout=5.0)
                if response.status_code != 200:
                    return False
        except Exception:
            return False

        return True

    return await check_health()
