"""
Deployment testing utilities and helpers for workspace-qdrant-mcp.

Provides utility functions and fixtures for testing production deployment
scenarios including Docker containerization, service management, and
monitoring validation.
"""

import asyncio
import json
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests


class DeploymentTestHelper:
    """Helper class for production deployment testing."""

    def __init__(self):
        self.temp_dirs: list[Path] = []
        self.mock_processes: list[AsyncMock] = []

    def create_temp_dir(self) -> Path:
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def cleanup(self):
        """Clean up temporary directories and resources."""
        import shutil
        for temp_dir in self.temp_dirs:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        self.temp_dirs.clear()
        self.mock_processes.clear()

    def create_mock_binary(self, binary_dir: Path, binary_name: str) -> Path:
        """Create a mock binary file for testing.

        Note: This creates a mock binary file but does not set executable permissions.
        Use mocking for os.access(path, os.X_OK) in tests that need to verify executability.
        """
        binary_dir.mkdir(parents=True, exist_ok=True)
        binary_path = binary_dir / binary_name

        # Create a simple shell script that acts as the binary
        binary_content = """#!/bin/bash
echo "Mock binary: $0"
echo "Args: $@"
exit 0
"""
        binary_path.write_text(binary_content)

        return binary_path

    def create_mock_config(self, config_dir: Path, config_name: str = "config.json") -> Path:
        """Create a mock configuration file."""
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / config_name

        config_data = {
            "qdrant_url": "http://localhost:6333",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "log_level": "INFO",
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 9090,
                "health_check_interval": 30
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        return config_path

    async def mock_subprocess_exec(self, *args, stdout=None, stderr=None, **kwargs):
        """Mock subprocess execution for testing."""
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b'', b''))
        mock_process.returncode = 0

        # Handle different commands
        if args and len(args) > 0:
            command = args[0]

            if command == "systemctl":
                if len(args) > 2 and args[2] == "is-active":
                    mock_process.communicate = AsyncMock(return_value=(b'active\n', b''))
                elif len(args) > 2 and args[2] == "is-enabled":
                    mock_process.communicate = AsyncMock(return_value=(b'enabled\n', b''))
                elif len(args) > 1 and args[1] == "status":
                    mock_status = "● test.service - Test Service\n   Loaded: loaded\n   Active: active (running)\n   Main PID: 12345\n"
                    mock_process.communicate = AsyncMock(return_value=(mock_status.encode(), b''))

            elif command == "launchctl":
                if len(args) > 1 and args[1] == "list":
                    mock_list = "PID\tStatus\tLabel\n12345\t0\tcom.test.service\n"
                    mock_process.communicate = AsyncMock(return_value=(mock_list.encode(), b''))

            elif command == "journalctl":
                mock_logs = "Jan 01 12:00:00 host test[12345]: Test log entry 1\nJan 01 12:00:01 host test[12345]: Test log entry 2\n"
                mock_process.communicate = AsyncMock(return_value=(mock_logs.encode(), b''))

        self.mock_processes.append(mock_process)
        return mock_process


class DockerTestHelper:
    """Helper for Docker-related deployment testing."""

    def __init__(self):
        self.containers: list[str] = []

    def is_docker_available(self) -> bool:
        """Check if Docker is available for testing."""
        try:
            result = subprocess.run(['docker', '--version'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def is_compose_available(self) -> bool:
        """Check if Docker Compose is available."""
        try:
            result = subprocess.run(['docker', 'compose', 'version'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def build_test_image(self, dockerfile_path: Path, tag: str) -> bool:
        """Build a test Docker image."""
        if not self.is_docker_available():
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                'docker', 'build', '-t', tag, '-f', str(dockerfile_path), '.',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()
            return process.returncode == 0

        except Exception:
            return False

    async def run_container(self, image: str, ports: dict[int, int] | None = None,
                          env_vars: dict[str, str] | None = None,
                          detach: bool = True) -> str | None:
        """Run a test container and return container ID."""
        if not self.is_docker_available():
            return None

        cmd = ['docker', 'run']

        if detach:
            cmd.append('-d')

        if ports:
            for host_port, container_port in ports.items():
                cmd.extend(['-p', f"{host_port}:{container_port}"])

        if env_vars:
            for key, value in env_vars.items():
                cmd.extend(['-e', f"{key}={value}"])

        cmd.append(image)

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                container_id = stdout.decode().strip()
                self.containers.append(container_id)
                return container_id

            return None

        except Exception:
            return None

    async def stop_container(self, container_id: str) -> bool:
        """Stop a running container."""
        try:
            process = await asyncio.create_subprocess_exec(
                'docker', 'stop', container_id,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await process.communicate()
            return process.returncode == 0

        except Exception:
            return False

    async def cleanup_containers(self):
        """Clean up all test containers."""
        for container_id in self.containers:
            await self.stop_container(container_id)

            # Remove container
            try:
                await asyncio.create_subprocess_exec(
                    'docker', 'rm', container_id,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
            except Exception:
                pass

        self.containers.clear()


class MonitoringTestHelper:
    """Helper for monitoring and observability testing."""

    @staticmethod
    async def wait_for_endpoint(url: str, timeout: float = 30.0,
                              expected_status: int = 200) -> bool:
        """Wait for an endpoint to become available."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == expected_status:
                    return True
            except requests.RequestException:
                pass

            await asyncio.sleep(1)

        return False

    @staticmethod
    async def test_prometheus_endpoint(url: str) -> dict[str, Any]:
        """Test a Prometheus metrics endpoint."""
        try:
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response": response.text[:500] if response.text else ""
                }

            content = response.text

            # Basic validation of Prometheus format
            has_help = "# HELP" in content
            has_type = "# TYPE" in content
            has_metrics = any(line and not line.startswith("#")
                            for line in content.split('\n'))

            return {
                "success": True,
                "content_length": len(content),
                "has_help": has_help,
                "has_type": has_type,
                "has_metrics": has_metrics,
                "metric_lines": len([line for line in content.split('\n')
                                   if line and not line.startswith('#')])
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    async def test_health_endpoint(url: str) -> dict[str, Any]:
        """Test a health check endpoint."""
        try:
            response = requests.get(url, timeout=10)

            if response.status_code not in [200, 503]:  # 503 for unhealthy but responsive
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "response": response.text[:500] if response.text else ""
                }

            try:
                data = response.json()
            except json.JSONDecodeError:
                return {
                    "success": False,
                    "error": "Invalid JSON response",
                    "response": response.text[:500]
                }

            # Validate health check format
            required_fields = ["status", "timestamp"]
            missing_fields = [field for field in required_fields
                            if field not in data]

            valid_statuses = ["healthy", "degraded", "unhealthy"]
            status_valid = data.get("status") in valid_statuses

            return {
                "success": True,
                "status": data.get("status"),
                "status_valid": status_valid,
                "missing_fields": missing_fields,
                "has_components": "components" in data,
                "components_count": len(data.get("components", {}))
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class LoadTestHelper:
    """Helper for load testing production deployments."""

    @staticmethod
    async def concurrent_requests(url: str, num_requests: int = 10,
                                concurrency: int = 5) -> dict[str, Any]:
        """Send concurrent requests to test endpoint performance."""
        import aiohttp

        async def make_request(session: aiohttp.ClientSession, request_id: int):
            start_time = time.time()
            try:
                async with session.get(url, timeout=10) as response:
                    end_time = time.time()
                    return {
                        "request_id": request_id,
                        "status_code": response.status,
                        "duration": end_time - start_time,
                        "success": True
                    }
            except Exception as e:
                end_time = time.time()
                return {
                    "request_id": request_id,
                    "status_code": 0,
                    "duration": end_time - start_time,
                    "success": False,
                    "error": str(e)
                }

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request(session: aiohttp.ClientSession, request_id: int):
            async with semaphore:
                return await make_request(session, request_id)

        # Execute requests
        async with aiohttp.ClientSession() as session:
            tasks = [
                limited_request(session, i)
                for i in range(num_requests)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed_requests = [r for r in results if isinstance(r, dict) and not r.get("success")]
        exceptions = [r for r in results if not isinstance(r, dict)]

        if successful_requests:
            durations = [r["duration"] for r in successful_requests]
            avg_duration = sum(durations) / len(durations)
            min_duration = min(durations)
            max_duration = max(durations)
        else:
            avg_duration = min_duration = max_duration = 0

        return {
            "total_requests": num_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "exceptions": len(exceptions),
            "success_rate": len(successful_requests) / num_requests * 100,
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "requests_per_second": len(successful_requests) / max(avg_duration * concurrency, 0.001)
        }


@asynccontextmanager
async def deployment_test_environment():
    """Context manager for setting up and tearing down deployment test environment."""
    helper = DeploymentTestHelper()
    docker_helper = DockerTestHelper()

    try:
        yield {
            "deployment": helper,
            "docker": docker_helper,
            "monitoring": MonitoringTestHelper(),
            "load_test": LoadTestHelper()
        }
    finally:
        helper.cleanup()
        await docker_helper.cleanup_containers()


# Pytest fixtures for deployment testing

@pytest.fixture
def deployment_helper():
    """Fixture providing deployment test helper."""
    helper = DeploymentTestHelper()
    yield helper
    helper.cleanup()


@pytest.fixture
def docker_helper():
    """Fixture providing Docker test helper."""
    helper = DockerTestHelper()
    yield helper
    asyncio.run(helper.cleanup_containers())


@pytest.fixture
def monitoring_helper():
    """Fixture providing monitoring test helper."""
    return MonitoringTestHelper()


@pytest.fixture
def load_test_helper():
    """Fixture providing load test helper."""
    return LoadTestHelper()


# Skip markers for conditional tests

def skip_without_docker():
    """Skip test if Docker is not available."""
    return pytest.mark.skipif(
        not DockerTestHelper().is_docker_available(),
        reason="Docker not available"
    )


def skip_without_compose():
    """Skip test if Docker Compose is not available."""
    return pytest.mark.skipif(
        not DockerTestHelper().is_compose_available(),
        reason="Docker Compose not available"
    )


def skip_on_ci():
    """Skip test in CI environments."""
    return pytest.mark.skipif(
        os.getenv("CI") is not None,
        reason="Skip in CI environment"
    )


# Test data generators

def generate_test_metrics():
    """Generate test metrics data."""
    return {
        "counters": {
            "requests_total": 100,
            "errors_total": 5,
            "documents_processed": 250
        },
        "gauges": {
            "active_connections": 15,
            "memory_usage_bytes": 500 * 1024 * 1024,
            "cpu_usage_percent": 45.2
        },
        "histograms": {
            "request_duration_seconds": {
                "count": 100,
                "sum": 45.5,
                "buckets": {
                    "0.1": 20,
                    "0.5": 70,
                    "1.0": 95,
                    "inf": 100
                }
            }
        }
    }


def generate_test_health_status():
    """Generate test health status data."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "All systems operational",
        "components": {
            "database": {
                "status": "healthy",
                "message": "Connected to Qdrant",
                "response_time": 0.015
            },
            "embedding_service": {
                "status": "healthy",
                "message": "Model loaded successfully",
                "response_time": 0.234
            },
            "daemon": {
                "status": "healthy",
                "message": "Rust daemon running",
                "response_time": 0.001
            }
        }
    }


if __name__ == "__main__":
    # Run basic tests to verify helpers work
    import asyncio

    async def test_helpers():
        async with deployment_test_environment() as env:
            print("Deployment test environment created successfully")

            # Test helper creation
            temp_dir = env["deployment"].create_temp_dir()
            print(f"Created temp dir: {temp_dir}")

            # Test mock config creation
            config_path = env["deployment"].create_mock_config(temp_dir)
            print(f"Created mock config: {config_path}")

            print("All helpers working correctly")

    asyncio.run(test_helpers())
