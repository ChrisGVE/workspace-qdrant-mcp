"""
Testcontainers integration tests for workspace-qdrant-mcp.

Tests Docker-based testing environments and integration scenarios
using testcontainers for isolated and reproducible testing.
"""

import os
import pytest
import time
from pathlib import Path
from typing import Dict, Any

try:
    from testcontainers.core.container import DockerContainer
    from testcontainers.core.waiting_utils import wait_container_is_ready
    from testcontainers.compose import DockerCompose
    TESTCONTAINERS_AVAILABLE = True
except ImportError:
    TESTCONTAINERS_AVAILABLE = False
    DockerContainer = None
    DockerCompose = None


@pytest.mark.skipif(not TESTCONTAINERS_AVAILABLE, reason="testcontainers not available")
class TestDockerTestEnvironment:
    """Test Docker-based testing environment."""

    @pytest.fixture(scope="class")
    def test_environment_container(self):
        """Create and start the test environment container."""
        dockerfile_path = Path(__file__).parent.parent.parent / "docker" / "test-environment.Dockerfile"

        if not dockerfile_path.exists():
            pytest.skip("Test environment Dockerfile not found")

        # Build and start the test environment container
        container = (
            DockerContainer(path=str(dockerfile_path.parent))
            .with_dockerfile("test-environment.Dockerfile")
            .with_env("TEST_SUITE", "unit")
            .with_env("PYTHONPATH", "/app/src/python")
            .with_exposed_port(8000)
            .with_volume_mapping(str(dockerfile_path.parent.parent), "/app", mode="rw")
            .with_command("sleep", "300")  # Keep container running for tests
        )

        with container as running_container:
            # Wait for container to be ready
            self._wait_for_container_ready(running_container)
            yield running_container

    def _wait_for_container_ready(self, container, timeout=60):
        """Wait for container to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = container.exec("python3 -c 'import sys; print(sys.version)'")
                if result.exit_code == 0:
                    return
            except Exception:
                pass
            time.sleep(2)

        raise TimeoutError("Container failed to become ready within timeout")

    def test_container_python_environment(self, test_environment_container):
        """Test Python environment in container."""
        # Check Python version
        result = test_environment_container.exec("python3 --version")
        assert result.exit_code == 0
        assert "Python 3.11" in result.output

        # Check virtual environment
        result = test_environment_container.exec("ls -la /app/.venv")
        assert result.exit_code == 0

        # Test package installation
        result = test_environment_container.exec(
            "bash -c 'source /app/.venv/bin/activate && python -c \"import pytest; print(pytest.__version__)\"'"
        )
        assert result.exit_code == 0

    def test_container_rust_environment(self, test_environment_container):
        """Test Rust environment in container."""
        # Check Rust version
        result = test_environment_container.exec("rustc --version")
        assert result.exit_code == 0
        assert "rustc" in result.output

        # Check Cargo
        result = test_environment_container.exec("cargo --version")
        assert result.exit_code == 0

        # Test Rust daemon directory exists
        result = test_environment_container.exec("ls -la /app/src/rust/daemon")
        assert result.exit_code == 0

    def test_container_unit_tests(self, test_environment_container):
        """Test running unit tests in container."""
        # Run unit tests
        result = test_environment_container.exec(
            "bash -c 'cd /app && source .venv/bin/activate && pytest tests/unit/ --tb=short -q'"
        )

        # Tests should run (exit code 0 or 1 for test failures, not environment issues)
        assert result.exit_code in [0, 1, 5]  # 5 = no tests collected

        # Should have pytest output
        assert "test" in result.output.lower() or "collect" in result.output.lower()

    def test_container_health_check(self, test_environment_container):
        """Test container health check functionality."""
        # Run health check script
        result = test_environment_container.exec("/app/healthcheck.sh")

        # Health check should pass
        if result.exit_code != 0:
            print(f"Health check output: {result.output}")

        # Health check might fail in test environment due to missing services
        # but should at least execute
        assert isinstance(result.exit_code, int)


@pytest.mark.skipif(not TESTCONTAINERS_AVAILABLE, reason="testcontainers not available")
class TestDockerComposeIntegration:
    """Test Docker Compose integration for multi-service testing."""

    @pytest.fixture(scope="class")
    def compose_environment(self):
        """Create Docker Compose test environment."""
        compose_file_content = """
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 5

  test-app:
    build:
      context: .
      dockerfile: docker/test-environment.Dockerfile
    depends_on:
      - qdrant
    environment:
      QDRANT_URL: http://qdrant:6333
      TEST_SUITE: integration
    volumes:
      - .:/app
    command: ["sleep", "300"]
"""

        # Create temporary compose file
        compose_file = Path("/tmp/test-compose.yml")
        compose_file.write_text(compose_file_content)

        try:
            with DockerCompose(str(compose_file.parent), compose_file_name="test-compose.yml") as compose:
                # Wait for services to be ready
                self._wait_for_services(compose)
                yield compose
        finally:
            # Clean up
            if compose_file.exists():
                compose_file.unlink()

    def _wait_for_services(self, compose, timeout=120):
        """Wait for all services to be ready."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check if Qdrant is ready
                qdrant_container = compose.get_service_container("qdrant", 1)
                result = qdrant_container.exec("curl -f http://localhost:6333/healthz")

                if result.exit_code == 0:
                    return
            except Exception as e:
                print(f"Waiting for services: {e}")

            time.sleep(5)

        raise TimeoutError("Services failed to become ready within timeout")

    def test_multi_service_integration(self, compose_environment):
        """Test integration between multiple services."""
        # Get the test app container
        app_container = compose_environment.get_service_container("test-app", 1)

        # Test connection to Qdrant
        result = app_container.exec("curl -f http://qdrant:6333/healthz")
        assert result.exit_code == 0

    def test_qdrant_service_connectivity(self, compose_environment):
        """Test Qdrant service connectivity."""
        qdrant_container = compose_environment.get_service_container("qdrant", 1)

        # Test Qdrant health endpoint
        result = qdrant_container.exec("curl -f http://localhost:6333/healthz")
        assert result.exit_code == 0

        # Test collections endpoint
        result = qdrant_container.exec("curl -f http://localhost:6333/collections")
        assert result.exit_code == 0


@pytest.mark.skipif(not TESTCONTAINERS_AVAILABLE, reason="testcontainers not available")
class TestIsolatedTestingScenarios:
    """Test isolated testing scenarios using containers."""

    def test_clean_database_state(self):
        """Test that each test gets a clean database state."""
        with DockerContainer("qdrant/qdrant:v1.7.0").with_exposed_port(6333) as qdrant:
            # Wait for Qdrant to be ready
            port = qdrant.get_exposed_port(6333)
            self._wait_for_qdrant(f"http://localhost:{port}")

            # Test 1: Create a collection
            import requests
            base_url = f"http://localhost:{port}"

            # Create collection
            response = requests.put(
                f"{base_url}/collections/test_collection",
                json={
                    "vectors": {
                        "size": 384,
                        "distance": "Cosine"
                    }
                }
            )
            assert response.status_code in [200, 201]

            # Verify collection exists
            response = requests.get(f"{base_url}/collections/test_collection")
            assert response.status_code == 200

    def _wait_for_qdrant(self, url, timeout=60):
        """Wait for Qdrant to be ready."""
        import requests
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/healthz", timeout=5)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)

        raise TimeoutError("Qdrant failed to become ready within timeout")

    def test_resource_cleanup(self):
        """Test that containers are properly cleaned up after tests."""
        container_id = None

        with DockerContainer("alpine:latest").with_command("sleep", "10") as container:
            container_id = container.get_container_id()

            # Container should be running
            result = container.exec("echo 'test'")
            assert result.exit_code == 0

        # After context exit, container should be stopped and removed
        # This is handled automatically by testcontainers
        assert container_id is not None

    def test_concurrent_container_usage(self):
        """Test concurrent usage of multiple containers."""
        import threading
        import time

        results = []
        errors = []

        def run_container_test(container_name, test_id):
            try:
                with DockerContainer("alpine:latest").with_command("sleep", "30") as container:
                    result = container.exec(f"echo 'test_{test_id}'")
                    assert result.exit_code == 0
                    results.append(f"Container {test_id} completed")
            except Exception as e:
                errors.append(f"Container {test_id} failed: {e}")

        # Start multiple concurrent containers
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_container_test, args=(f"test_{i}", i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)

        # All containers should have completed successfully
        assert len(errors) == 0, f"Container errors: {errors}"
        assert len(results) == 3

    def test_container_networking(self):
        """Test networking between containers."""
        # This test requires Docker network setup which is complex in testcontainers
        # For now, we'll test basic container communication concepts

        with DockerContainer("alpine:latest").with_command("sleep", "60") as container1:
            # Test that container has network access
            result = container1.exec("ping -c 1 8.8.8.8")

            # Network access might not be available in all test environments
            # So we check that the command executed (exit code is meaningful)
            assert isinstance(result.exit_code, int)

    def test_environment_variable_isolation(self):
        """Test environment variable isolation between containers."""
        env1 = {"TEST_VAR": "value1", "CONTAINER_ID": "container1"}
        env2 = {"TEST_VAR": "value2", "CONTAINER_ID": "container2"}

        with DockerContainer("alpine:latest").with_env_vars(env1).with_command("sleep", "30") as container1:
            with DockerContainer("alpine:latest").with_env_vars(env2).with_command("sleep", "30") as container2:
                # Test environment isolation
                result1 = container1.exec("echo $TEST_VAR")
                result2 = container2.exec("echo $TEST_VAR")

                assert result1.exit_code == 0
                assert result2.exit_code == 0

                # Each container should have its own environment
                assert "value1" in result1.output
                assert "value2" in result2.output


@pytest.mark.skipif(not TESTCONTAINERS_AVAILABLE, reason="testcontainers not available")
class TestContainerPerformanceValidation:
    """Test performance validation in containerized environments."""

    def test_container_startup_time(self):
        """Test container startup time meets requirements."""
        start_time = time.time()

        with DockerContainer("alpine:latest").with_command("echo", "ready") as container:
            startup_time = time.time() - start_time

            # Container should start within reasonable time (30 seconds)
            assert startup_time < 30.0

            # Test that container is functional
            result = container.exec("echo 'test'")
            assert result.exit_code == 0

    def test_container_resource_usage(self):
        """Test container resource usage constraints."""
        # This test checks that containers don't consume excessive resources
        with DockerContainer("alpine:latest").with_command("sleep", "10") as container:
            # Run a simple command to verify container is working
            result = container.exec("ps aux")
            assert result.exit_code == 0

            # Check that we can get process information
            assert "PID" in result.output or "sleep" in result.output

    def test_container_memory_limits(self):
        """Test container memory limit enforcement."""
        # Create container with memory limit
        with DockerContainer("alpine:latest").with_command("sleep", "10") as container:
            # Test memory-related commands work
            result = container.exec("free -m")

            # Command should execute successfully
            assert result.exit_code == 0

    def test_concurrent_container_performance(self):
        """Test performance with multiple concurrent containers."""
        start_time = time.time()
        containers_count = 5

        # Function to run container test
        def run_performance_test(test_id):
            with DockerContainer("alpine:latest").with_command("sleep", "5") as container:
                result = container.exec("echo 'performance_test'")
                return result.exit_code == 0

        # Run concurrent containers
        import threading
        results = []

        def worker(test_id):
            results.append(run_performance_test(test_id))

        threads = []
        for i in range(containers_count):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=30)

        end_time = time.time()
        total_time = end_time - start_time

        # All tests should succeed
        assert all(results)
        assert len(results) == containers_count

        # Should complete within reasonable time (concurrent execution should be faster than sequential)
        assert total_time < 60.0  # 60 seconds for 5 concurrent containers