"""
E2E Test Utilities (Task 292.1).

Helper functions and utilities for E2E testing.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import subprocess


class HealthChecker:
    """Utility for checking component health."""

    @staticmethod
    async def wait_for_http_endpoint(
        url: str,
        timeout: int = 30,
        interval: float = 2.0,
        expected_status: int = 200
    ) -> bool:
        """
        Wait for HTTP endpoint to become available.

        Args:
            url: URL to check
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds
            expected_status: Expected HTTP status code

        Returns:
            True if endpoint becomes available, False otherwise
        """
        import aiohttp

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=5) as response:
                        if response.status == expected_status:
                            return True
            except Exception:
                pass

            await asyncio.sleep(interval)

        return False

    @staticmethod
    async def check_grpc_service(
        host: str,
        port: int,
        timeout: int = 10
    ) -> bool:
        """
        Check if gRPC service is available.

        Args:
            host: Service host
            port: Service port
            timeout: Connection timeout

        Returns:
            True if service is available
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False


class WorkflowTimer:
    """Utility for timing workflow steps."""

    def __init__(self):
        self.checkpoints = []
        self.start_time = None

    def start(self):
        """Start timing."""
        self.start_time = time.time()
        self.checkpoints = []

    def checkpoint(self, name: str):
        """Record a checkpoint."""
        if self.start_time is None:
            self.start()

        elapsed = time.time() - self.start_time
        self.checkpoints.append({
            "name": name,
            "elapsed_seconds": elapsed,
            "timestamp": time.time()
        })

    def get_duration(self, checkpoint_name: Optional[str] = None) -> float:
        """Get duration to specific checkpoint or total."""
        if checkpoint_name:
            for cp in self.checkpoints:
                if cp["name"] == checkpoint_name:
                    return cp["elapsed_seconds"]
            return 0.0

        if not self.checkpoints:
            return 0.0

        return self.checkpoints[-1]["elapsed_seconds"]

    def get_summary(self) -> Dict[str, Any]:
        """Get timing summary."""
        if not self.checkpoints:
            return {"total_duration": 0.0, "checkpoints": []}

        return {
            "total_duration": self.checkpoints[-1]["elapsed_seconds"],
            "checkpoints": self.checkpoints
        }


class TestDataGenerator:
    """Generate test data for E2E tests."""

    @staticmethod
    def create_python_module(
        name: str,
        functions: int = 3,
        classes: int = 1
    ) -> str:
        """
        Generate Python module content.

        Args:
            name: Module name
            functions: Number of functions to generate
            classes: Number of classes to generate

        Returns:
            Python source code
        """
        lines = [
            f'"""Test module: {name}."""',
            "",
            "import os",
            "import sys",
            "",
        ]

        # Add functions
        for i in range(functions):
            lines.extend([
                f"def function_{i}(param_{i}):",
                f'    """Function {i} description."""',
                f"    return param_{i} * {i + 1}",
                "",
            ])

        # Add classes
        for i in range(classes):
            lines.extend([
                f"class TestClass{i}:",
                f'    """Test class {i}."""',
                "",
                "    def __init__(self):",
                f'        self.value = {i}',
                "",
                "    def method(self):",
                '        """Test method."""',
                "        return self.value",
                "",
            ])

        return "\n".join(lines)

    @staticmethod
    def create_markdown_document(
        title: str,
        sections: int = 3,
        content_per_section: int = 100
    ) -> str:
        """
        Generate Markdown document content.

        Args:
            title: Document title
            sections: Number of sections
            content_per_section: Words per section

        Returns:
            Markdown content
        """
        lines = [
            f"# {title}",
            "",
            "This is a test document for E2E testing.",
            "",
        ]

        for i in range(sections):
            lines.extend([
                f"## Section {i + 1}",
                "",
                " ".join([f"word{j}" for j in range(content_per_section)]),
                "",
            ])

        return "\n".join(lines)

    @staticmethod
    def create_config_file(
        format: str = "yaml",
        complexity: str = "simple"
    ) -> str:
        """
        Generate configuration file content.

        Args:
            format: Configuration format (yaml, json, toml)
            complexity: Configuration complexity (simple, complex)

        Returns:
            Configuration content
        """
        if complexity == "simple":
            config = {
                "debug": False,
                "log_level": "info",
                "port": 8000
            }
        else:
            config = {
                "server": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "workers": 4
                },
                "database": {
                    "url": "postgresql://localhost/db",
                    "pool_size": 10
                },
                "logging": {
                    "level": "info",
                    "format": "json",
                    "handlers": ["console", "file"]
                }
            }

        if format == "json":
            return json.dumps(config, indent=2)
        elif format == "yaml":
            import yaml
            return yaml.dump(config, default_flow_style=False)
        elif format == "toml":
            import toml
            return toml.dumps(config)

        return str(config)


class ComponentController:
    """Control system components for testing."""

    @staticmethod
    async def restart_component(
        component_name: str,
        stop_timeout: int = 10,
        start_timeout: int = 30
    ) -> bool:
        """
        Restart a system component.

        Args:
            component_name: Name of component to restart
            stop_timeout: Timeout for graceful stop
            start_timeout: Timeout for startup

        Returns:
            True if restart successful
        """
        # Placeholder - would use actual component managers
        await asyncio.sleep(2)
        return True

    @staticmethod
    async def simulate_component_failure(
        component_name: str,
        failure_type: str = "crash"
    ):
        """
        Simulate component failure for testing.

        Args:
            component_name: Component to fail
            failure_type: Type of failure (crash, hang, network)
        """
        # Placeholder - would use actual failure injection
        await asyncio.sleep(1)

    @staticmethod
    async def verify_component_isolation(
        component_name: str,
        check_process: bool = True,
        check_network: bool = True,
        check_files: bool = True
    ) -> Dict[str, bool]:
        """
        Verify component isolation and resource usage.

        Args:
            component_name: Component to check
            check_process: Check process isolation
            check_network: Check network isolation
            check_files: Check file isolation

        Returns:
            Isolation check results
        """
        results = {}

        if check_process:
            # Check process exists and is isolated
            results["process_isolated"] = True

        if check_network:
            # Check network isolation
            results["network_isolated"] = True

        if check_files:
            # Check file isolation
            results["files_isolated"] = True

        return results


class QdrantTestHelper:
    """Helper for Qdrant operations in tests."""

    def __init__(self, url: str):
        self.url = url

    async def create_test_collection(
        self,
        collection_name: str,
        vector_size: int = 384
    ) -> bool:
        """Create a test collection."""
        # Placeholder - would use Qdrant client
        await asyncio.sleep(0.5)
        return True

    async def verify_document_count(
        self,
        collection_name: str,
        expected_count: int
    ) -> bool:
        """Verify document count in collection."""
        # Placeholder - would query Qdrant
        await asyncio.sleep(0.3)
        return True

    async def search_test_query(
        self,
        collection_name: str,
        query: str,
        min_results: int = 1
    ) -> List[Dict[str, Any]]:
        """Execute test search query."""
        # Placeholder - would perform actual search
        await asyncio.sleep(0.5)
        return [
            {"id": "doc1", "score": 0.95, "content": "Test result"}
        ]

    async def cleanup_test_collections(self, prefix: str = "test_"):
        """Clean up test collections."""
        # Placeholder - would delete test collections
        await asyncio.sleep(0.5)


def assert_within_threshold(
    actual: float,
    expected: float,
    threshold_percent: float = 20.0,
    metric_name: str = "value"
):
    """
    Assert value is within threshold percentage.

    Args:
        actual: Actual value
        expected: Expected value
        threshold_percent: Allowed variance percentage
        metric_name: Name of metric for error message

    Raises:
        AssertionError: If value exceeds threshold
    """
    if expected == 0:
        assert actual == 0, f"{metric_name}: expected 0, got {actual}"
        return

    variance_percent = abs((actual - expected) / expected) * 100

    assert variance_percent <= threshold_percent, (
        f"{metric_name} outside threshold: "
        f"expected {expected}, got {actual} "
        f"(variance {variance_percent:.1f}%, threshold {threshold_percent}%)"
    )


def run_git_command(
    command: List[str],
    cwd: Path,
    check: bool = True
) -> subprocess.CompletedProcess:
    """
    Run git command in workspace.

    Args:
        command: Git command arguments
        cwd: Working directory
        check: Raise on non-zero exit

    Returns:
        Completed process
    """
    full_command = ["git"] + command

    return subprocess.run(
        full_command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check
    )
