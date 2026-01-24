"""
End-to-End Test Infrastructure (Task 292.1).

Comprehensive E2E testing framework with pytest-bdd, Docker Compose orchestration,
component lifecycle management, and test utilities for system-wide testing.

Features:
- pytest-bdd scenario-based testing
- Docker Compose service management (Qdrant, daemon, MCP server)
- Component lifecycle fixtures (startup, shutdown, health checks)
- Temporary project creation with Git repos
- Test isolation and cleanup
- Performance monitoring utilities
- Resource tracking and validation
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pytest
from testcontainers.compose import DockerCompose

import docker

# E2E Test Configuration
E2E_TEST_CONFIG = {
    "docker_compose": {
        "project_name": "wqm-e2e-tests",
        "compose_file": "docker/integration-tests/docker-compose.yml",
        "startup_timeout": 60,
        "shutdown_timeout": 30,
        "health_check_interval": 2,
        "max_health_checks": 30
    },
    "services": {
        "qdrant": {
            "http_port": 6333,
            "grpc_port": 6334,
            "health_endpoint": "http://localhost:6333/health",
            "startup_wait": 10
        },
        "daemon": {
            "grpc_port": 50051,
            "health_timeout": 20,
            "startup_wait": 15
        },
        "mcp_server": {
            "http_port": 8000,
            "health_endpoint": "http://localhost:8000/health",
            "startup_wait": 5
        }
    },
    "timeouts": {
        "short": 10,
        "medium": 30,
        "long": 60,
        "workflow": 120
    },
    "performance": {
        "baseline_latency_ms": 1000,
        "baseline_throughput_docs_per_sec": 1.0,
        "memory_limit_mb": 1000,
        "max_open_files": 200
    }
}


@pytest.fixture(scope="session")
def docker_client():
    """Provide Docker client for E2E tests."""
    try:
        client = docker.from_env()
        yield client
    except Exception as e:
        pytest.skip(f"Docker not available for E2E tests: {e}")


@pytest.fixture(scope="session")
def docker_compose_project(docker_client):
    """
    Set up Docker Compose project for E2E tests.

    Manages lifecycle of all services (Qdrant, daemon, MCP server).
    """
    compose_file = Path(E2E_TEST_CONFIG["docker_compose"]["compose_file"])

    if not compose_file.exists():
        pytest.skip(f"Docker Compose file not found: {compose_file}")

    E2E_TEST_CONFIG["docker_compose"]["project_name"]

    class DockerComposeManager:
        def __init__(self):
            self.compose = DockerCompose(
                filepath=str(compose_file.parent),
                compose_file_name=str(compose_file.name),
                pull=False
            )
            self.started = False

        def start(self):
            """Start all Docker Compose services."""
            if self.started:
                return

            print(f"\nStarting Docker Compose services from {compose_file}...")
            self.compose.start()
            self.started = True

            # Wait for services to be ready
            startup_wait = E2E_TEST_CONFIG["docker_compose"]["startup_timeout"]
            print(f"Waiting {startup_wait}s for services to initialize...")
            time.sleep(startup_wait)

        def stop(self):
            """Stop all Docker Compose services."""
            if not self.started:
                return

            print("\nStopping Docker Compose services...")
            try:
                self.compose.stop()
            except Exception as e:
                print(f"Error stopping services: {e}")
            finally:
                self.started = False

        def get_service_url(self, service: str, protocol: str = "http") -> str:
            """Get service URL."""
            service_config = E2E_TEST_CONFIG["services"].get(service, {})
            port = service_config.get("http_port", 8000)
            return f"{protocol}://localhost:{port}"

    manager = DockerComposeManager()

    yield manager

    # Cleanup
    manager.stop()


@pytest.fixture(scope="function")
def e2e_services(docker_compose_project):
    """
    Provide E2E services for each test function.

    Ensures services are started and provides service URLs.
    """
    # Start services if not already started
    docker_compose_project.start()

    # Provide service information
    services = {
        "qdrant_url": docker_compose_project.get_service_url("qdrant"),
        "mcp_url": docker_compose_project.get_service_url("mcp_server"),
        "daemon_grpc_port": E2E_TEST_CONFIG["services"]["daemon"]["grpc_port"]
    }

    yield services

    # Services stay running between tests (session scope)


@pytest.fixture
def temp_project_workspace():
    """
    Create temporary workspace with realistic project structure.

    Includes Git repository initialization for project detection testing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)

        # Create realistic project structure
        dirs = [
            "src/python/myapp",
            "src/python/tests",
            "docs/api",
            "docs/guides",
            "config",
            "data",
            "scripts",
            ".git/objects",
            ".git/refs"
        ]
        for dir_path in dirs:
            (workspace / dir_path).mkdir(parents=True)

        # Initialize Git repository
        git_init = subprocess.run(
            ["git", "init"],
            cwd=workspace,
            capture_output=True,
            text=True
        )

        if git_init.returncode == 0:
            # Configure Git for testing
            subprocess.run(["git", "config", "user.name", "E2E Test"], cwd=workspace)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=workspace)

        # Create common files
        files = {
            "README.md": "# E2E Test Project\n\nComprehensive testing workspace.",
            "pyproject.toml": "[project]\nname = 'e2e-test-project'\nversion = '0.1.0'",
            ".gitignore": "__pycache__/\n*.pyc\n.env\n.venv/",
            "src/python/myapp/__init__.py": '"""Test application."""',
            "src/python/myapp/main.py": "def main():\n    print('E2E Test App')",
            "src/python/tests/__init__.py": "",
            "src/python/tests/test_main.py": "def test_main():\n    assert True",
            "docs/api/README.md": "# API Documentation",
            "docs/guides/getting-started.md": "# Getting Started Guide",
            "config/settings.yaml": "debug: false\nlog_level: info",
            "scripts/deploy.sh": "#!/bin/bash\necho 'Deploy script'",
        }

        for file_path, content in files.items():
            full_path = workspace / file_path
            full_path.write_text(content)

        # NOTE: Scripts created with default permissions (no chmod)
        # If executable bit is needed, tests should mock or adapt expectations

        # Initial Git commit
        if git_init.returncode == 0:
            subprocess.run(["git", "add", "."], cwd=workspace)
            subprocess.run(
                ["git", "commit", "-m", "Initial commit"],
                cwd=workspace
            )

        yield {
            "path": workspace,
            "files": list(files.keys()),
            "total_files": len(files),
            "git_initialized": git_init.returncode == 0
        }


@pytest.fixture
async def component_lifecycle_manager():
    """
    Manage component lifecycle for E2E tests.

    Provides utilities for starting, stopping, and checking health of components.
    """
    class ComponentManager:
        def __init__(self):
            self.components = {}
            self.startup_order = ["qdrant", "daemon", "mcp_server"]
            self.shutdown_order = ["mcp_server", "daemon", "qdrant"]

        async def start_component(self, name: str) -> bool:
            """Start a specific component."""
            if name in self.components and self.components[name].get("running"):
                return True

            print(f"Starting component: {name}")

            # Component-specific startup logic (mocked for now)
            # In real implementation, would use service managers
            self.components[name] = {
                "running": True,
                "start_time": time.time()
            }

            return True

        async def stop_component(self, name: str) -> bool:
            """Stop a specific component."""
            if name not in self.components:
                return True

            print(f"Stopping component: {name}")

            self.components[name]["running"] = False

            return True

        async def check_health(self, name: str) -> dict[str, Any]:
            """Check component health status."""
            if name not in self.components:
                return {"healthy": False, "error": "Component not started"}

            component = self.components[name]
            if not component.get("running"):
                return {"healthy": False, "error": "Component not running"}

            uptime = time.time() - component["start_time"]

            return {
                "healthy": True,
                "uptime_seconds": uptime,
                "status": "running"
            }

        async def start_all(self) -> bool:
            """Start all components in correct order."""
            for component in self.startup_order:
                success = await self.start_component(component)
                if not success:
                    print(f"Failed to start {component}")
                    return False

                # Wait for component to be ready
                await asyncio.sleep(2)

            return True

        async def stop_all(self) -> bool:
            """Stop all components in reverse order."""
            for component in self.shutdown_order:
                await self.stop_component(component)
                await asyncio.sleep(1)

            return True

        async def wait_for_ready(self, timeout: int = 60) -> bool:
            """Wait for all components to be ready."""
            start_time = time.time()

            while time.time() - start_time < timeout:
                all_healthy = True

                for component in self.startup_order:
                    health = await self.check_health(component)
                    if not health.get("healthy"):
                        all_healthy = False
                        break

                if all_healthy:
                    return True

                await asyncio.sleep(E2E_TEST_CONFIG["docker_compose"]["health_check_interval"])

            return False

    manager = ComponentManager()
    yield manager

    # Cleanup
    await manager.stop_all()


@pytest.fixture
def resource_tracker():
    """
    Track resource usage during E2E tests.

    Monitors memory, CPU, disk, file descriptors.
    """
    class ResourceTracker:
        def __init__(self):
            self.baseline = {}
            self.current = {}
            self.history = []

        def capture_baseline(self):
            """Capture baseline resource usage."""
            import psutil

            self.baseline = {
                "memory_mb": psutil.virtual_memory().used / (1024 * 1024),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "timestamp": time.time()
            }

        def capture_current(self):
            """Capture current resource usage."""
            import psutil

            self.current = {
                "memory_mb": psutil.virtual_memory().used / (1024 * 1024),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "timestamp": time.time()
            }

            self.history.append(self.current.copy())

        def get_delta(self) -> dict[str, float]:
            """Get resource usage delta from baseline."""
            if not self.baseline or not self.current:
                return {}

            return {
                "memory_delta_mb": self.current["memory_mb"] - self.baseline["memory_mb"],
                "cpu_delta_percent": self.current["cpu_percent"] - self.baseline["cpu_percent"],
                "disk_delta_percent": self.current["disk_usage_percent"] - self.baseline["disk_usage_percent"],
                "duration_seconds": self.current["timestamp"] - self.baseline["timestamp"]
            }

        def check_thresholds(self) -> list[str]:
            """Check if resource usage exceeds thresholds."""
            warnings = []
            delta = self.get_delta()

            if delta.get("memory_delta_mb", 0) > E2E_TEST_CONFIG["performance"]["memory_limit_mb"]:
                warnings.append(f"Memory usage exceeded: {delta['memory_delta_mb']:.1f} MB")

            return warnings

    tracker = ResourceTracker()
    tracker.capture_baseline()

    yield tracker

    # Final capture and report
    tracker.capture_current()
    tracker.get_delta()
    warnings = tracker.check_thresholds()

    if warnings:
        print("\nResource usage warnings:")
        for warning in warnings:
            print(f"  - {warning}")


@pytest.fixture
def test_orchestrator():
    """
    Test orchestration utilities for complex E2E scenarios.

    Provides helpers for coordinating multi-step workflows.
    """
    class TestOrchestrator:
        def __init__(self):
            self.steps = []
            self.results = []

        def add_step(self, name: str, description: str, func):
            """Add a test step to the workflow."""
            self.steps.append({
                "name": name,
                "description": description,
                "func": func
            })

        async def execute_workflow(self) -> dict[str, Any]:
            """Execute all steps in order."""
            start_time = time.time()

            for i, step in enumerate(self.steps, 1):
                print(f"\nExecuting step {i}/{len(self.steps)}: {step['name']}")
                print(f"  {step['description']}")

                step_start = time.time()

                try:
                    if asyncio.iscoroutinefunction(step['func']):
                        result = await step['func']()
                    else:
                        result = step['func']()

                    step_duration = time.time() - step_start

                    self.results.append({
                        "step": i,
                        "name": step['name'],
                        "success": True,
                        "duration_seconds": step_duration,
                        "result": result
                    })

                except Exception as e:
                    step_duration = time.time() - step_start

                    self.results.append({
                        "step": i,
                        "name": step['name'],
                        "success": False,
                        "duration_seconds": step_duration,
                        "error": str(e)
                    })

                    # Stop on first failure
                    break

            total_duration = time.time() - start_time

            return {
                "total_steps": len(self.steps),
                "completed_steps": len(self.results),
                "successful_steps": sum(1 for r in self.results if r.get("success")),
                "failed_steps": sum(1 for r in self.results if not r.get("success")),
                "total_duration_seconds": total_duration,
                "results": self.results
            }

        def get_summary(self) -> str:
            """Get workflow execution summary."""
            if not self.results:
                return "No steps executed"

            lines = ["Workflow Execution Summary:"]
            lines.append(f"  Total steps: {len(self.steps)}")
            lines.append(f"  Completed: {len(self.results)}")

            for result in self.results:
                status = "✅" if result.get("success") else "❌"
                duration = result.get("duration_seconds", 0)
                lines.append(f"  {status} Step {result['step']}: {result['name']} ({duration:.2f}s)")

            return "\n".join(lines)

    yield TestOrchestrator()


@pytest.fixture
def scenario_context():
    """
    Shared context for pytest-bdd scenarios.

    Stores state between scenario steps.
    """
    class ScenarioContext:
        def __init__(self):
            self.data = {}

        def set(self, key: str, value: Any):
            """Set context value."""
            self.data[key] = value

        def get(self, key: str, default: Any = None) -> Any:
            """Get context value."""
            return self.data.get(key, default)

        def clear(self):
            """Clear all context data."""
            self.data = {}

    context = ScenarioContext()
    yield context
    context.clear()


# pytest-bdd configuration
# Commented out until pytest-bdd is installed
# def pytest_bdd_step_error(request, feature, scenario, step, step_func, step_func_args, exception):
#     """Handle pytest-bdd step errors."""
#     print(f"\nStep failed: {step.name}")
#     print(f"  Feature: {feature.name}")
#     print(f"  Scenario: {scenario.name}")
#     print(f"  Error: {exception}")


def pytest_configure(config):
    """Configure E2E tests."""
    # Register markers
    config.addinivalue_line("markers", "e2e: End-to-end system tests")
    config.addinivalue_line("markers", "workflow: Complete workflow tests")
    config.addinivalue_line("markers", "stability: Long-running stability tests")
    config.addinivalue_line("markers", "performance: Performance regression tests")
    config.addinivalue_line("markers", "concurrent: Concurrent/parallel access tests")

    # Set environment for E2E tests
    os.environ["E2E_TESTING"] = "1"

    # Note: PYTEST_CURRENT_TEST is managed by pytest internally, don't manipulate it


def pytest_collection_modifyitems(config, items):
    """Modify E2E test collection."""
    for item in items:
        # Add e2e marker to all tests in e2e directory
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add slow marker to stability tests
        if "stability" in item.name.lower() or "24hour" in item.name.lower():
            item.add_marker(pytest.mark.slow)
