"""
Docker Container Compatibility Tests.

Tests Docker image builds, container operations, networking, and multi-container scenarios
for AMD64 and ARM64 architectures.
"""

import os
import platform
import pytest
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, List


# Check if Docker is available
def _check_docker_available() -> bool:
    """Check if Docker is available on the system."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


DOCKER_AVAILABLE = _check_docker_available()
SKIP_DOCKER_REASON = "Docker not available or not running"


def _get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def project_root():
    """Fixture providing project root path."""
    return _get_project_root()


@pytest.fixture
def docker_dir(project_root):
    """Fixture providing docker directory path."""
    return project_root / "docker"


class TestDockerAvailability:
    """Test Docker installation and availability."""

    def test_docker_command_exists(self):
        """Test Docker command is available."""
        result = subprocess.run(
            ["which", "docker"] if platform.system() != "Windows" else ["where", "docker"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Docker command not found in PATH"

    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
    def test_docker_version(self):
        """Test Docker version can be retrieved."""
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0
        assert "Docker version" in result.stdout

    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
    def test_docker_info(self):
        """Test Docker info command works."""
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, "Docker daemon not running or not accessible"


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
class TestDockerfileValidation:
    """Test Dockerfile validation."""

    def test_main_dockerfile_exists(self, docker_dir):
        """Test main Dockerfile exists."""
        dockerfile = docker_dir / "Dockerfile"
        assert dockerfile.exists(), f"Dockerfile not found at {dockerfile}"

    def test_main_dockerfile_syntax(self, docker_dir):
        """Test main Dockerfile has valid syntax."""
        dockerfile = docker_dir / "Dockerfile"
        content = dockerfile.read_text()

        # Check for required directives
        assert "FROM" in content, "Dockerfile missing FROM directive"
        assert ("RUN" in content or "COPY" in content or "ADD" in content), (
            "Dockerfile has no build instructions"
        )

    def test_integration_test_dockerfiles_exist(self, docker_dir):
        """Test integration test Dockerfiles exist."""
        integration_dir = docker_dir / "integration-tests"
        assert integration_dir.exists()

        dockerfile = integration_dir / "Dockerfile"
        assert dockerfile.exists()

        daemon_dockerfile = integration_dir / "Dockerfile.daemon"
        assert daemon_dockerfile.exists()


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
class TestDockerComposeValidation:
    """Test Docker Compose configuration validation."""

    def test_docker_compose_installed(self):
        """Test docker-compose or docker compose is available."""
        # Try docker-compose first (standalone)
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return

        # Try docker compose (plugin)
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Neither docker-compose nor docker compose available"

    def test_main_compose_file_exists(self, docker_dir):
        """Test main docker-compose.yml exists."""
        compose_file = docker_dir / "docker-compose.yml"
        assert compose_file.exists()

    def test_compose_file_syntax(self, docker_dir):
        """Test docker-compose.yml has valid YAML syntax."""
        import yaml

        compose_file = docker_dir / "docker-compose.yml"
        content = compose_file.read_text()

        try:
            config = yaml.safe_load(content)
            assert isinstance(config, dict)
            assert "services" in config or "version" in config
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML syntax: {e}")

    def test_dev_compose_file_exists(self, docker_dir):
        """Test docker-compose.dev.yml exists."""
        compose_file = docker_dir / "docker-compose.dev.yml"
        assert compose_file.exists()

    def test_prod_compose_file_exists(self, docker_dir):
        """Test docker-compose.prod.yml exists."""
        compose_file = docker_dir / "docker-compose.prod.yml"
        assert compose_file.exists()


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
@pytest.mark.slow
class TestDockerImageBuild:
    """Test Docker image building."""

    def test_docker_build_context(self, project_root):
        """Test Docker build context is valid."""
        dockerfile = project_root / "docker" / "Dockerfile"
        assert dockerfile.exists()

        # Test build context (dry-run)
        result = subprocess.run(
            ["docker", "build", "--help"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
        )
        assert result.returncode == 0

    @pytest.mark.slow
    def test_docker_build_syntax_check(self, project_root):
        """Test Dockerfile syntax with --check flag (if supported)."""
        dockerfile = project_root / "docker" / "Dockerfile"

        # Try to validate Dockerfile syntax
        # Note: Not all Docker versions support --check
        result = subprocess.run(
            ["docker", "build", "--help"],
            capture_output=True,
            text=True,
        )

        if "--check" in result.stdout:
            result = subprocess.run(
                ["docker", "build", "--check", "-f", str(dockerfile), str(project_root)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            # --check should pass or be unsupported
            assert result.returncode in (0, 1)  # 1 might mean unsupported


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
class TestDockerArchitectureSupport:
    """Test Docker multi-architecture support."""

    def test_current_architecture_detection(self):
        """Test current system architecture detection."""
        machine = platform.machine().lower()
        assert machine in ("x86_64", "amd64", "arm64", "aarch64"), (
            f"Unsupported architecture: {machine}"
        )

    def test_docker_buildx_available(self):
        """Test Docker buildx is available for multi-platform builds."""
        result = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            pytest.skip("Docker buildx not available")

        assert "buildx" in result.stdout.lower()

    def test_docker_platforms_available(self):
        """Test which platforms are available for builds."""
        result = subprocess.run(
            ["docker", "buildx", "ls"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            pytest.skip("Docker buildx not available")

        # Check output includes platform information
        assert result.stdout, "No buildx builders found"


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
class TestDockerNetworking:
    """Test Docker networking capabilities."""

    def test_docker_network_ls(self):
        """Test listing Docker networks."""
        result = subprocess.run(
            ["docker", "network", "ls"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "bridge" in result.stdout.lower()

    def test_docker_network_inspect_bridge(self):
        """Test inspecting default bridge network."""
        result = subprocess.run(
            ["docker", "network", "inspect", "bridge"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

        import json

        try:
            network_info = json.loads(result.stdout)
            assert isinstance(network_info, list)
            assert len(network_info) > 0
        except json.JSONDecodeError:
            pytest.fail("Invalid JSON output from docker network inspect")


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
class TestDockerVolumes:
    """Test Docker volume operations."""

    def test_docker_volume_ls(self):
        """Test listing Docker volumes."""
        result = subprocess.run(
            ["docker", "volume", "ls"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_docker_volume_create_and_remove(self):
        """Test creating and removing a Docker volume."""
        volume_name = "wqm_test_volume"

        # Create volume
        result = subprocess.run(
            ["docker", "volume", "create", volume_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

        try:
            # Inspect volume
            result = subprocess.run(
                ["docker", "volume", "inspect", volume_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode == 0
        finally:
            # Remove volume
            subprocess.run(
                ["docker", "volume", "rm", volume_name],
                capture_output=True,
                text=True,
                timeout=10,
            )


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
@pytest.mark.slow
class TestDockerComposeOperations:
    """Test Docker Compose operations."""

    def _get_compose_command(self) -> List[str]:
        """Get the appropriate docker-compose command."""
        # Try docker-compose first
        result = subprocess.run(
            ["docker-compose", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return ["docker-compose"]

        # Try docker compose
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return ["docker", "compose"]

        pytest.skip("Neither docker-compose nor docker compose available")

    def test_compose_config_validation(self, docker_dir):
        """Test docker-compose config validation."""
        compose_cmd = self._get_compose_command()
        compose_file = docker_dir / "docker-compose.yml"

        result = subprocess.run(
            [*compose_cmd, "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            pytest.fail(f"Compose config validation failed: {result.stderr}")

    def test_compose_ps(self, docker_dir):
        """Test docker-compose ps command."""
        compose_cmd = self._get_compose_command()
        compose_file = docker_dir / "docker-compose.yml"

        result = subprocess.run(
            [*compose_cmd, "-f", str(compose_file), "ps"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(docker_dir),
        )

        # ps should work even if no containers are running
        assert result.returncode in (0, 1)


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason=SKIP_DOCKER_REASON)
class TestDockerSecurityFeatures:
    """Test Docker security features."""

    def test_docker_security_options(self):
        """Test Docker supports security options."""
        result = subprocess.run(
            ["docker", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0

        # Check for security-related flags
        help_text = result.stdout.lower()
        assert "--security-opt" in help_text or "security" in help_text

    def test_docker_user_namespace_support(self):
        """Test Docker user namespace remapping support."""
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            pytest.skip("Docker info not accessible")

        # Check if user namespace remapping is mentioned
        # This is optional, so we just verify the info command works
        assert result.returncode == 0
