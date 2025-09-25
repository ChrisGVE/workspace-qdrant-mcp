#!/usr/bin/env python3
"""
Comprehensive unit tests for the Infrastructure Provisioning system.

Tests cover:
- Environment registration and validation
- Multi-provider infrastructure provisioning
- Configuration drift detection and correction
- Health checks and monitoring
- Resource management and lifecycle
- Error handling and edge cases
- Provider-specific implementations
- Security and compliance validation
"""

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml

# Import the module under test
try:
    from infrastructure_provisioning import (
        EnvironmentConfig,
        EnvironmentType,
        HealthCheckResult,
        InfrastructureProvider,
        InfrastructureProvisioner,
        InfrastructureState,
        InfrastructureStatus,
        ResourceConfiguration,
        create_example_environment,
    )
except ImportError:
    # For when running as standalone module
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from infrastructure_provisioning import (
        EnvironmentConfig,
        EnvironmentType,
        HealthCheckResult,
        InfrastructureProvider,
        InfrastructureProvisioner,
        InfrastructureState,
        InfrastructureStatus,
        ResourceConfiguration,
        create_example_environment,
    )


class TestResourceConfiguration:
    """Test ResourceConfiguration class."""

    def test_resource_configuration_creation(self):
        """Test basic resource configuration creation."""
        resource = ResourceConfiguration(
            resource_type="vm",
            name="web-server",
            properties={"instance_type": "t3.micro", "ami": "ami-12345"},
            dependencies=["security-group"],
            tags={"Environment": "test"},
            health_checks=[{"type": "http", "url": "http://localhost:80"}]
        )

        assert resource.resource_type == "vm"
        assert resource.name == "web-server"
        assert resource.properties["instance_type"] == "t3.micro"
        assert "security-group" in resource.dependencies
        assert resource.tags["Environment"] == "test"
        assert len(resource.health_checks) == 1

    def test_resource_configuration_defaults(self):
        """Test resource configuration with default values."""
        resource = ResourceConfiguration(
            resource_type="database",
            name="main-db",
            properties={"engine": "postgresql"}
        )

        assert resource.dependencies == []
        assert resource.tags == {}
        assert resource.health_checks == []

    def test_resource_configuration_complex_properties(self):
        """Test resource configuration with complex properties."""
        complex_properties = {
            "networking": {
                "vpc_id": "vpc-12345",
                "subnets": ["subnet-1", "subnet-2"],
                "security_groups": ["sg-1", "sg-2"]
            },
            "scaling": {
                "min_size": 1,
                "max_size": 10,
                "desired_capacity": 3
            }
        }

        resource = ResourceConfiguration(
            resource_type="autoscaling_group",
            name="web-asg",
            properties=complex_properties
        )

        assert resource.properties["networking"]["vpc_id"] == "vpc-12345"
        assert resource.properties["scaling"]["min_size"] == 1


class TestEnvironmentConfig:
    """Test EnvironmentConfig class."""

    def test_environment_config_creation(self):
        """Test environment configuration creation."""
        resources = [
            ResourceConfiguration(
                resource_type="vm",
                name="web-server",
                properties={"instance_type": "t3.micro"}
            )
        ]

        env_config = EnvironmentConfig(
            name="test-env",
            environment_type=EnvironmentType.TEST,
            provider=InfrastructureProvider.AWS,
            region="us-west-2",
            resources=resources,
            variables={"APP_ENV": "test"},
            secrets=["API_KEY"],
            monitoring={"enabled": True},
            security_settings={"encryption": True}
        )

        assert env_config.name == "test-env"
        assert env_config.environment_type == EnvironmentType.TEST
        assert env_config.provider == InfrastructureProvider.AWS
        assert len(env_config.resources) == 1
        assert env_config.variables["APP_ENV"] == "test"
        assert "API_KEY" in env_config.secrets

    def test_environment_config_defaults(self):
        """Test environment configuration with default values."""
        resources = [
            ResourceConfiguration(
                resource_type="vm",
                name="test-vm",
                properties={}
            )
        ]

        env_config = EnvironmentConfig(
            name="minimal-env",
            environment_type=EnvironmentType.DEVELOPMENT,
            provider=InfrastructureProvider.DOCKER,
            region="local",
            resources=resources
        )

        assert env_config.variables == {}
        assert env_config.secrets == []
        assert env_config.monitoring == {}
        assert env_config.security_settings == {}


class TestInfrastructureProvisioner:
    """Test InfrastructureProvisioner class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def provisioner(self, temp_dir):
        """Create a test provisioner instance."""
        return InfrastructureProvisioner(
            config_dir=temp_dir / "config",
            state_dir=temp_dir / "state",
            monitoring_enabled=True
        )

    @pytest.fixture
    def sample_environment(self):
        """Create a sample environment configuration."""
        resources = [
            ResourceConfiguration(
                resource_type="vm",
                name="test-vm",
                properties={"instance_type": "t3.micro"},
                health_checks=[{"type": "tcp", "port": 22}]
            ),
            ResourceConfiguration(
                resource_type="database",
                name="test-db",
                properties={"engine": "postgresql"},
                dependencies=["test-vm"],
                health_checks=[{"type": "tcp", "port": 5432}]
            )
        ]

        return EnvironmentConfig(
            name="test-env",
            environment_type=EnvironmentType.TEST,
            provider=InfrastructureProvider.TERRAFORM,
            region="us-west-2",
            resources=resources,
            variables={"DATABASE_URL": "postgresql://localhost:5432/test"},
            secrets=["DB_PASSWORD"]
        )

    def test_provisioner_initialization(self, temp_dir):
        """Test provisioner initialization."""
        provisioner = InfrastructureProvisioner(
            config_dir=temp_dir / "config",
            state_dir=temp_dir / "state"
        )

        assert provisioner.config_dir == temp_dir / "config"
        assert provisioner.state_dir == temp_dir / "state"
        assert provisioner.config_dir.exists()
        assert provisioner.state_dir.exists()

    @patch('subprocess.run')
    def test_check_dependencies(self, mock_run, provisioner):
        """Test dependency checking."""
        # Simulate terraform available, kubectl not available
        mock_run.side_effect = [
            MagicMock(returncode=0),  # terraform --version
            subprocess.CalledProcessError(1, 'kubectl'),  # kubectl --version
            MagicMock(returncode=0),  # helm --version
        ]

        provisioner._check_dependencies()

        assert mock_run.call_count == 3

    def test_is_tool_available(self, provisioner):
        """Test tool availability checking."""
        with patch('subprocess.run') as mock_run:
            # Tool available
            mock_run.return_value = MagicMock(returncode=0)
            assert provisioner._is_tool_available("terraform")

            # Tool not available
            mock_run.side_effect = subprocess.CalledProcessError(1, 'missing-tool')
            assert not provisioner._is_tool_available("missing-tool")

            # Tool not found
            mock_run.side_effect = FileNotFoundError()
            assert not provisioner._is_tool_available("not-found")

    async def test_register_environment_success(self, provisioner, sample_environment):
        """Test successful environment registration."""
        await provisioner.register_environment(sample_environment)

        assert "test-env" in provisioner.environments
        assert "test-env" in provisioner.infrastructure_status

        registered_config = provisioner.environments["test-env"]
        assert registered_config.name == "test-env"
        assert len(registered_config.resources) == 2

        status = provisioner.infrastructure_status["test-env"]
        assert status.environment_name == "test-env"
        assert status.state == InfrastructureState.PLANNING

    async def test_register_environment_validation_errors(self, provisioner):
        """Test environment registration validation errors."""
        # Empty name
        with pytest.raises(ValueError, match="Environment name cannot be empty"):
            await provisioner.register_environment(
                EnvironmentConfig(
                    name="",
                    environment_type=EnvironmentType.TEST,
                    provider=InfrastructureProvider.DOCKER,
                    region="local",
                    resources=[]
                )
            )

        # No resources
        with pytest.raises(ValueError, match="Environment must have at least one resource"):
            await provisioner.register_environment(
                EnvironmentConfig(
                    name="empty-env",
                    environment_type=EnvironmentType.TEST,
                    provider=InfrastructureProvider.DOCKER,
                    region="local",
                    resources=[]
                )
            )

        # Invalid dependency
        invalid_resources = [
            ResourceConfiguration(
                resource_type="vm",
                name="vm1",
                properties={},
                dependencies=["non-existent-resource"]
            )
        ]

        with pytest.raises(ValueError, match="has invalid dependency"):
            await provisioner.register_environment(
                EnvironmentConfig(
                    name="invalid-deps",
                    environment_type=EnvironmentType.TEST,
                    provider=InfrastructureProvider.DOCKER,
                    region="local",
                    resources=invalid_resources
                )
            )

    async def test_generate_terraform_config(self, provisioner, sample_environment, temp_dir):
        """Test Terraform configuration generation."""
        provisioner.config_dir = temp_dir
        await provisioner._generate_terraform_config(sample_environment)

        # Check generated files
        env_dir = temp_dir / "terraform" / "test-env"
        assert env_dir.exists()
        assert (env_dir / "main.tf").exists()
        assert (env_dir / "variables.tf").exists()
        assert (env_dir / "outputs.tf").exists()

        # Validate main.tf content
        with open(env_dir / "main.tf") as f:
            main_config = json.load(f)

        assert "terraform" in main_config
        assert "provider" in main_config
        assert "resource" in main_config

        # Validate variables.tf content
        with open(env_dir / "variables.tf") as f:
            vars_config = json.load(f)

        assert "variable" in vars_config
        assert "DATABASE_URL" in vars_config["variable"]
        assert "DB_PASSWORD" in vars_config["variable"]

        # Check sensitive flag for secrets
        assert vars_config["variable"]["DB_PASSWORD"]["sensitive"]
        assert not vars_config["variable"]["DATABASE_URL"]["sensitive"]

    async def test_generate_kubernetes_config(self, provisioner, temp_dir):
        """Test Kubernetes configuration generation."""
        provisioner.config_dir = temp_dir

        k8s_resources = [
            ResourceConfiguration(
                resource_type="deployment",
                name="web-app",
                properties={
                    "image": "nginx:latest",
                    "replicas": 3,
                    "ports": [{"containerPort": 80}]
                }
            ),
            ResourceConfiguration(
                resource_type="service",
                name="web-service",
                properties={
                    "app": "web-app",
                    "ports": [{"port": 80, "targetPort": 80}],
                    "type": "LoadBalancer"
                }
            )
        ]

        k8s_env = EnvironmentConfig(
            name="k8s-test",
            environment_type=EnvironmentType.TEST,
            provider=InfrastructureProvider.KUBERNETES,
            region="us-west-2",
            resources=k8s_resources,
            variables={"APP_ENV": "test"}
        )

        await provisioner._generate_kubernetes_config(k8s_env)

        # Check generated files
        env_dir = temp_dir / "kubernetes" / "k8s-test"
        assert env_dir.exists()
        assert (env_dir / "namespace.yaml").exists()
        assert (env_dir / "web-app.yaml").exists()
        assert (env_dir / "web-service.yaml").exists()

        # Validate namespace manifest
        with open(env_dir / "namespace.yaml") as f:
            namespace = yaml.safe_load(f)

        assert namespace["kind"] == "Namespace"
        assert namespace["metadata"]["name"] == "k8s-test"

        # Validate deployment manifest
        with open(env_dir / "web-app.yaml") as f:
            deployment = yaml.safe_load(f)

        assert deployment["kind"] == "Deployment"
        assert deployment["metadata"]["name"] == "web-app"
        assert deployment["spec"]["replicas"] == 3

    async def test_generate_docker_config(self, provisioner, temp_dir):
        """Test Docker Compose configuration generation."""
        provisioner.config_dir = temp_dir

        docker_resources = [
            ResourceConfiguration(
                resource_type="docker_container",
                name="web",
                properties={
                    "image": "nginx:latest",
                    "ports": ["80:80"],
                    "restart": "unless-stopped"
                }
            ),
            ResourceConfiguration(
                resource_type="docker_container",
                name="db",
                properties={
                    "image": "postgres:13",
                    "ports": ["5432:5432"],
                    "volumes": ["db_data:/var/lib/postgresql/data"]
                }
            )
        ]

        docker_env = EnvironmentConfig(
            name="docker-test",
            environment_type=EnvironmentType.DEVELOPMENT,
            provider=InfrastructureProvider.DOCKER,
            region="local",
            resources=docker_resources,
            variables={"POSTGRES_DB": "testdb", "POSTGRES_USER": "user"},
            secrets=["POSTGRES_PASSWORD"]
        )

        await provisioner._generate_docker_config(docker_env)

        # Check generated files
        env_dir = temp_dir / "docker" / "docker-test"
        assert env_dir.exists()
        assert (env_dir / "docker-compose.yml").exists()
        assert (env_dir / ".env").exists()

        # Validate docker-compose.yml
        with open(env_dir / "docker-compose.yml") as f:
            compose_config = yaml.safe_load(f)

        assert "services" in compose_config
        assert "web" in compose_config["services"]
        assert "db" in compose_config["services"]

        web_service = compose_config["services"]["web"]
        assert web_service["image"] == "nginx:latest"
        assert "80:80" in web_service["ports"]

        # Validate .env file (should not contain secrets)
        with open(env_dir / ".env") as f:
            env_content = f.read()

        assert "POSTGRES_DB=testdb" in env_content
        assert "POSTGRES_USER=user" in env_content
        assert "POSTGRES_PASSWORD" not in env_content

    def test_map_resource_to_terraform(self, provisioner):
        """Test resource type mapping to Terraform resources."""
        assert provisioner._map_resource_to_terraform("vm") == "aws_instance"
        assert provisioner._map_resource_to_terraform("load_balancer") == "aws_lb"
        assert provisioner._map_resource_to_terraform("database") == "aws_db_instance"
        assert provisioner._map_resource_to_terraform("unknown") is None

    def test_infer_terraform_type(self, provisioner):
        """Test Terraform type inference from Python values."""
        assert provisioner._infer_terraform_type(True) == "bool"
        assert provisioner._infer_terraform_type(42) == "number"
        assert provisioner._infer_terraform_type(["a", "b"]) == "list(string)"
        assert provisioner._infer_terraform_type({"key": "value"}) == "map(string)"
        assert provisioner._infer_terraform_type("string") == "string"

    @patch('asyncio.create_subprocess_exec')
    async def test_provision_terraform_success(self, mock_subprocess, provisioner, sample_environment):
        """Test successful Terraform provisioning."""
        # Mock successful subprocess calls
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Success", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        await provisioner.register_environment(sample_environment)
        status = await provisioner.provision_environment("test-env", dry_run=True)

        assert status.state == InfrastructureState.PLANNING  # Dry run doesn't change state
        assert mock_subprocess.call_count >= 2  # init + plan

    @patch('asyncio.create_subprocess_exec')
    async def test_provision_terraform_failure(self, mock_subprocess, provisioner, sample_environment):
        """Test Terraform provisioning failure."""
        # Mock failed subprocess call
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"Error occurred")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process

        await provisioner.register_environment(sample_environment)

        with pytest.raises(subprocess.CalledProcessError):
            await provisioner.provision_environment("test-env", dry_run=True)

        status = provisioner.infrastructure_status["test-env"]
        assert status.state == InfrastructureState.FAILED

    async def test_provision_unregistered_environment(self, provisioner):
        """Test provisioning of unregistered environment."""
        with pytest.raises(ValueError, match="Environment .* not registered"):
            await provisioner.provision_environment("non-existent")

    @patch('asyncio.create_subprocess_exec')
    async def test_run_command_success(self, mock_subprocess, provisioner):
        """Test successful command execution."""
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        result = await provisioner._run_command(["echo", "hello"])

        assert result.returncode == 0
        assert result.stdout == "output"
        assert result.stderr == ""

    @patch('asyncio.create_subprocess_exec')
    async def test_run_command_failure(self, mock_subprocess, provisioner):
        """Test failed command execution."""
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"error")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process

        with pytest.raises(subprocess.CalledProcessError):
            await provisioner._run_command(["failing-command"])

    @patch('aiohttp.ClientSession.get')
    async def test_http_health_check_success(self, mock_get, provisioner):
        """Test successful HTTP health check."""
        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_get.return_value.__aenter__.return_value = mock_response

        resource = ResourceConfiguration(
            resource_type="web_server",
            name="test-server",
            properties={},
            health_checks=[{"type": "http", "url": "http://test-server:80"}]
        )

        health_check = {"type": "http", "url": "http://test-server:80", "expected_status": 200}
        result = await provisioner._execute_health_check(resource, health_check)

        assert result.status == "passed"
        assert result.resource_name == "test-server"
        assert result.check_type == "http"

    @patch('aiohttp.ClientSession.get')
    async def test_http_health_check_failure(self, mock_get, provisioner):
        """Test failed HTTP health check."""
        # Mock HTTP error
        mock_get.side_effect = Exception("Connection refused")

        resource = ResourceConfiguration(
            resource_type="web_server",
            name="test-server",
            properties={}
        )

        health_check = {"type": "http", "url": "http://test-server:80"}
        result = await provisioner._execute_health_check(resource, health_check)

        assert result.status == "failed"
        assert result.resource_name == "test-server"
        assert result.check_type == "http"
        assert "Connection refused" in result.message

    @patch('aiohttp.ClientSession.get')
    async def test_http_health_check_wrong_status(self, mock_get, provisioner):
        """Test HTTP health check with unexpected status code."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_get.return_value.__aenter__.return_value = mock_response

        resource = ResourceConfiguration(
            resource_type="web_server",
            name="test-server",
            properties={}
        )

        health_check = {"type": "http", "url": "http://test-server:80", "expected_status": 200}
        result = await provisioner._execute_health_check(resource, health_check)

        assert result.status == "failed"
        assert "500 (expected 200)" in result.message

    @patch('asyncio.open_connection')
    async def test_tcp_health_check_success(self, mock_open_connection, provisioner):
        """Test successful TCP health check."""
        # Mock successful TCP connection
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_writer.wait_closed = AsyncMock()
        mock_open_connection.return_value = (mock_reader, mock_writer)

        resource = ResourceConfiguration(
            resource_type="database",
            name="test-db",
            properties={}
        )

        health_check = {"type": "tcp", "host": "test-db", "port": 5432}
        result = await provisioner._execute_health_check(resource, health_check)

        assert result.status == "passed"
        assert result.resource_name == "test-db"
        assert result.check_type == "tcp"
        mock_writer.close.assert_called_once()

    @patch('asyncio.open_connection')
    async def test_tcp_health_check_failure(self, mock_open_connection, provisioner):
        """Test failed TCP health check."""
        # Mock TCP connection failure
        mock_open_connection.side_effect = ConnectionRefusedError("Connection refused")

        resource = ResourceConfiguration(
            resource_type="database",
            name="test-db",
            properties={}
        )

        health_check = {"type": "tcp", "host": "test-db", "port": 5432}
        result = await provisioner._execute_health_check(resource, health_check)

        assert result.status == "failed"
        assert result.resource_name == "test-db"
        assert result.check_type == "tcp"
        assert "Connection refused" in result.message

    @patch('asyncio.wait_for')
    async def test_tcp_health_check_timeout(self, mock_wait_for, provisioner):
        """Test TCP health check timeout."""
        # Mock timeout
        mock_wait_for.side_effect = asyncio.TimeoutError()

        resource = ResourceConfiguration(
            resource_type="database",
            name="test-db",
            properties={}
        )

        health_check = {"type": "tcp", "host": "test-db", "port": 5432, "timeout": 1}
        result = await provisioner._execute_health_check(resource, health_check)

        assert result.status == "failed"
        assert "timeout" in result.message.lower()

    @patch('asyncio.wait_for')
    async def test_command_health_check_success(self, mock_wait_for, provisioner):
        """Test successful command health check."""
        # Mock successful command result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "OK"
        mock_result.stderr = ""
        mock_wait_for.return_value = mock_result

        resource = ResourceConfiguration(
            resource_type="service",
            name="test-service",
            properties={}
        )

        health_check = {"type": "command", "command": ["curl", "-f", "http://localhost/health"]}

        with patch.object(provisioner, '_run_command', return_value=mock_result):
            result = await provisioner._execute_health_check(resource, health_check)

        assert result.status == "passed"
        assert result.resource_name == "test-service"
        assert result.check_type == "command"

    @patch('asyncio.wait_for')
    async def test_command_health_check_wrong_exit_code(self, mock_wait_for, provisioner):
        """Test command health check with unexpected exit code."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error"
        mock_wait_for.return_value = mock_result

        resource = ResourceConfiguration(
            resource_type="service",
            name="test-service",
            properties={}
        )

        health_check = {
            "type": "command",
            "command": ["false"],
            "expected_exit_code": 0
        }

        with patch.object(provisioner, '_run_command', return_value=mock_result):
            result = await provisioner._execute_health_check(resource, health_check)

        assert result.status == "failed"
        assert "exited with code 1 (expected 0)" in result.message

    async def test_execute_health_check_unknown_type(self, provisioner):
        """Test health check with unknown type."""
        resource = ResourceConfiguration(
            resource_type="service",
            name="test-service",
            properties={}
        )

        health_check = {"type": "unknown_type"}

        with pytest.raises(ValueError, match="Unknown health check type"):
            await provisioner._execute_health_check(resource, health_check)

    @patch('asyncio.create_subprocess_exec')
    async def test_detect_terraform_drift_no_changes(self, mock_subprocess, provisioner, sample_environment):
        """Test Terraform drift detection with no changes."""
        # Mock terraform plan with no changes (exit code 0)
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"No changes", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        await provisioner.register_environment(sample_environment)
        drift_detected = await provisioner.detect_drift("test-env")

        assert not drift_detected
        status = provisioner.infrastructure_status["test-env"]
        assert not status.drift_detected
        assert status.drift_details == []

    @patch('asyncio.create_subprocess_exec')
    async def test_detect_terraform_drift_with_changes(self, mock_subprocess, provisioner, sample_environment):
        """Test Terraform drift detection with changes."""
        # Mock terraform plan with changes (exit code 2)
        mock_plan_process = AsyncMock()
        mock_plan_process.communicate.return_value = (b"Changes detected", b"")
        mock_plan_process.returncode = 2

        # Mock terraform show for plan details
        mock_show_process = AsyncMock()
        plan_data = {
            "resource_changes": [
                {
                    "address": "aws_instance.web",
                    "change": {
                        "actions": ["update"],
                        "before": {"instance_type": "t3.micro"},
                        "after": {"instance_type": "t3.small"}
                    }
                }
            ]
        }
        mock_show_process.communicate.return_value = (json.dumps(plan_data).encode(), b"")
        mock_show_process.returncode = 0

        mock_subprocess.side_effect = [mock_plan_process, mock_show_process]

        await provisioner.register_environment(sample_environment)
        drift_detected = await provisioner.detect_drift("test-env")

        assert drift_detected
        status = provisioner.infrastructure_status["test-env"]
        assert status.drift_detected
        assert len(status.drift_details) == 1
        assert status.state == InfrastructureState.DRIFTED

    @patch('asyncio.create_subprocess_exec')
    async def test_detect_kubernetes_drift_with_missing_resource(self, mock_subprocess, provisioner, temp_dir):
        """Test Kubernetes drift detection with missing resource."""
        provisioner.config_dir = temp_dir

        k8s_resources = [
            ResourceConfiguration(
                resource_type="deployment",
                name="web-app",
                properties={"image": "nginx:latest", "replicas": 2}
            )
        ]

        k8s_env = EnvironmentConfig(
            name="k8s-drift-test",
            environment_type=EnvironmentType.TEST,
            provider=InfrastructureProvider.KUBERNETES,
            region="us-west-2",
            resources=k8s_resources
        )

        await provisioner.register_environment(k8s_env)

        # Mock kubectl get failure (resource doesn't exist)
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"", b"resource not found")
        mock_process.returncode = 1
        mock_subprocess.return_value = mock_process

        drift_detected = await provisioner.detect_drift("k8s-drift-test")

        assert drift_detected
        status = provisioner.infrastructure_status["k8s-drift-test"]
        assert status.drift_detected
        assert len(status.drift_details) == 1
        assert status.drift_details[0]["status"] == "missing"

    @patch('asyncio.create_subprocess_exec')
    async def test_detect_docker_drift_with_stopped_service(self, mock_subprocess, provisioner, temp_dir):
        """Test Docker drift detection with stopped service."""
        provisioner.config_dir = temp_dir

        docker_resources = [
            ResourceConfiguration(
                resource_type="docker_container",
                name="web",
                properties={"image": "nginx:latest"}
            )
        ]

        docker_env = EnvironmentConfig(
            name="docker-drift-test",
            environment_type=EnvironmentType.TEST,
            provider=InfrastructureProvider.DOCKER,
            region="local",
            resources=docker_resources
        )

        await provisioner.register_environment(docker_env)

        # Mock docker-compose ps showing stopped service
        mock_process = AsyncMock()
        container_data = json.dumps({
            "Name": "web_1",
            "Service": "web",
            "State": "exited",
            "Image": "nginx:latest"
        })
        mock_process.communicate.return_value = (container_data.encode(), b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        drift_detected = await provisioner.detect_drift("docker-drift-test")

        assert drift_detected
        status = provisioner.infrastructure_status["docker-drift-test"]
        assert status.drift_detected
        assert len(status.drift_details) == 1
        assert status.drift_details[0]["issue"] == "not_running"

    async def test_detect_drift_unregistered_environment(self, provisioner):
        """Test drift detection for unregistered environment."""
        with pytest.raises(ValueError, match="Environment .* not registered"):
            await provisioner.detect_drift("non-existent")

    @patch('asyncio.create_subprocess_exec')
    async def test_correct_terraform_drift_auto_approve(self, mock_subprocess, provisioner, sample_environment):
        """Test correcting Terraform drift with auto-approve."""
        # Setup environment with drift
        await provisioner.register_environment(sample_environment)
        status = provisioner.infrastructure_status["test-env"]
        status.drift_detected = True
        status.drift_details = [{"resource": "aws_instance.web", "action": ["update"]}]
        status.state = InfrastructureState.DRIFTED

        # Mock successful terraform apply
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Apply complete!", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        # Mock drift detection after correction (no drift)
        with patch.object(provisioner, 'detect_drift', return_value=False):
            corrected_status = await provisioner.correct_drift("test-env", auto_approve=True)

        assert corrected_status.state == InfrastructureState.RUNNING
        assert not corrected_status.drift_detected

    @patch('asyncio.create_subprocess_exec')
    async def test_correct_kubernetes_drift(self, mock_subprocess, provisioner, temp_dir):
        """Test correcting Kubernetes drift."""
        provisioner.config_dir = temp_dir

        k8s_resources = [
            ResourceConfiguration(
                resource_type="deployment",
                name="web-app",
                properties={"image": "nginx:latest"}
            )
        ]

        k8s_env = EnvironmentConfig(
            name="k8s-correct-test",
            environment_type=EnvironmentType.TEST,
            provider=InfrastructureProvider.KUBERNETES,
            region="us-west-2",
            resources=k8s_resources
        )

        await provisioner.register_environment(k8s_env)

        # Setup drift state
        status = provisioner.infrastructure_status["k8s-correct-test"]
        status.drift_detected = True
        status.drift_details = [{"resource": "deployment/web-app", "status": "missing"}]
        status.state = InfrastructureState.DRIFTED

        # Mock successful kubectl apply
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"deployment configured", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        # Mock no drift after correction
        with patch.object(provisioner, 'detect_drift', return_value=False):
            corrected_status = await provisioner.correct_drift("k8s-correct-test", auto_approve=True)

        assert corrected_status.state == InfrastructureState.RUNNING
        assert not corrected_status.drift_detected

    async def test_correct_drift_no_drift_detected(self, provisioner, sample_environment):
        """Test correcting drift when no drift is detected."""
        await provisioner.register_environment(sample_environment)

        # No drift initially
        status = provisioner.infrastructure_status["test-env"]
        assert not status.drift_detected

        corrected_status = await provisioner.correct_drift("test-env")
        assert corrected_status == status  # Should return unchanged status

    @patch('asyncio.create_subprocess_exec')
    async def test_destroy_terraform_environment(self, mock_subprocess, provisioner, sample_environment):
        """Test destroying Terraform environment."""
        await provisioner.register_environment(sample_environment)

        # Mock successful terraform destroy
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"Destroy complete!", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        await provisioner.destroy_environment("test-env")

        # Environment should be removed
        assert "test-env" not in provisioner.environments
        assert "test-env" not in provisioner.infrastructure_status

    @patch('asyncio.create_subprocess_exec')
    async def test_destroy_kubernetes_environment(self, mock_subprocess, provisioner, temp_dir):
        """Test destroying Kubernetes environment."""
        provisioner.config_dir = temp_dir

        k8s_resources = [
            ResourceConfiguration(
                resource_type="deployment",
                name="web-app",
                properties={"image": "nginx:latest"}
            )
        ]

        k8s_env = EnvironmentConfig(
            name="k8s-destroy-test",
            environment_type=EnvironmentType.TEST,
            provider=InfrastructureProvider.KUBERNETES,
            region="us-west-2",
            resources=k8s_resources
        )

        await provisioner.register_environment(k8s_env)

        # Mock successful kubectl delete namespace
        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"namespace deleted", b"")
        mock_process.returncode = 0
        mock_subprocess.return_value = mock_process

        await provisioner.destroy_environment("k8s-destroy-test", force=True)

        # Environment should be removed
        assert "k8s-destroy-test" not in provisioner.environments
        assert "k8s-destroy-test" not in provisioner.infrastructure_status

    async def test_destroy_unregistered_environment(self, provisioner):
        """Test destroying unregistered environment."""
        with pytest.raises(ValueError, match="Environment .* not registered"):
            await provisioner.destroy_environment("non-existent")

    def test_get_environment_status(self, provisioner):
        """Test getting environment status."""
        # Non-existent environment
        assert provisioner.get_environment_status("non-existent") is None

        # Add a test status
        test_status = InfrastructureStatus(
            environment_name="test-env",
            state=InfrastructureState.RUNNING,
            resources={},
            health_checks=[],
            last_update=time.time()
        )
        provisioner.infrastructure_status["test-env"] = test_status

        retrieved_status = provisioner.get_environment_status("test-env")
        assert retrieved_status == test_status

    async def test_list_environments(self, provisioner, sample_environment):
        """Test listing registered environments."""
        # Initially empty
        assert provisioner.list_environments() == []

        # After registration
        await provisioner.register_environment(sample_environment)
        assert provisioner.list_environments() == ["test-env"]

    async def test_generate_status_report(self, provisioner, sample_environment):
        """Test generating status report."""
        await provisioner.register_environment(sample_environment)

        # Update status to have some data
        status = provisioner.infrastructure_status["test-env"]
        status.state = InfrastructureState.RUNNING
        status.resources = {"vm1": {"status": "running"}, "db1": {"status": "running"}}
        status.health_checks = [
            HealthCheckResult(
                resource_name="vm1",
                check_type="tcp",
                status="passed",
                message="OK",
                timestamp=time.time()
            ),
            HealthCheckResult(
                resource_name="db1",
                check_type="tcp",
                status="failed",
                message="Connection refused",
                timestamp=time.time()
            )
        ]
        status.drift_detected = True
        status.drift_details = [{"resource": "vm1", "change": "instance_type"}]

        report = provisioner.generate_status_report()

        assert "timestamp" in report
        assert "environments" in report
        assert "summary" in report

        # Check environment data
        env_data = report["environments"]["test-env"]
        assert env_data["state"] == "running"
        assert env_data["resources_count"] == 2
        assert env_data["health_checks_passed"] == 1
        assert env_data["health_checks_total"] == 2
        assert env_data["drift_detected"] is True
        assert env_data["drift_issues"] == 1

        # Check summary
        summary = report["summary"]
        assert summary["total_environments"] == 1
        assert summary["running"] == 1
        assert summary["failed"] == 0

    def test_print_status_table(self, provisioner, sample_environment, capsys):
        """Test printing status table."""
        # This is a simple test since Rich output is complex to test
        provisioner.print_status_table()  # Should not crash with empty data

        # With data would require more complex Rich testing setup


class TestComplexScenarios:
    """Test complex scenarios and edge cases."""

    @pytest.fixture
    def complex_environment(self):
        """Create a complex environment with multiple dependencies."""
        resources = [
            # Network infrastructure
            ResourceConfiguration(
                resource_type="network",
                name="main-vpc",
                properties={"cidr": "10.0.0.0/16"},
                tags={"Environment": "production", "Tier": "network"}
            ),
            ResourceConfiguration(
                resource_type="security_group",
                name="web-sg",
                properties={"ingress_rules": [{"port": 80, "protocol": "tcp"}]},
                dependencies=["main-vpc"],
                tags={"Environment": "production", "Tier": "security"}
            ),
            # Application tier
            ResourceConfiguration(
                resource_type="vm",
                name="web-server-1",
                properties={"instance_type": "t3.medium", "ami": "ami-12345"},
                dependencies=["web-sg"],
                health_checks=[
                    {"type": "http", "url": "http://web-server-1/health", "timeout": 30},
                    {"type": "tcp", "port": 22, "timeout": 5}
                ],
                tags={"Environment": "production", "Tier": "web", "Instance": "1"}
            ),
            ResourceConfiguration(
                resource_type="vm",
                name="web-server-2",
                properties={"instance_type": "t3.medium", "ami": "ami-12345"},
                dependencies=["web-sg"],
                health_checks=[
                    {"type": "http", "url": "http://web-server-2/health", "timeout": 30},
                    {"type": "tcp", "port": 22, "timeout": 5}
                ],
                tags={"Environment": "production", "Tier": "web", "Instance": "2"}
            ),
            # Load balancer
            ResourceConfiguration(
                resource_type="load_balancer",
                name="web-lb",
                properties={
                    "type": "application",
                    "targets": ["web-server-1", "web-server-2"]
                },
                dependencies=["web-server-1", "web-server-2"],
                health_checks=[
                    {"type": "http", "url": "http://web-lb/health", "timeout": 10}
                ],
                tags={"Environment": "production", "Tier": "load-balancer"}
            ),
            # Database tier
            ResourceConfiguration(
                resource_type="database",
                name="main-db",
                properties={"engine": "postgresql", "instance_class": "db.t3.large"},
                dependencies=["main-vpc"],
                health_checks=[
                    {"type": "tcp", "port": 5432, "timeout": 10},
                    {
                        "type": "command",
                        "command": ["pg_isready", "-h", "main-db", "-p", "5432"],
                        "timeout": 15
                    }
                ],
                tags={"Environment": "production", "Tier": "database"}
            ),
            # Storage
            ResourceConfiguration(
                resource_type="storage",
                name="app-storage",
                properties={"type": "s3", "bucket_name": "app-production-storage"},
                health_checks=[
                    {
                        "type": "command",
                        "command": ["aws", "s3", "ls", "s3://app-production-storage"],
                        "timeout": 20
                    }
                ],
                tags={"Environment": "production", "Tier": "storage"}
            )
        ]

        return EnvironmentConfig(
            name="production",
            environment_type=EnvironmentType.PRODUCTION,
            provider=InfrastructureProvider.TERRAFORM,
            region="us-west-2",
            resources=resources,
            variables={
                "vpc_cidr": "10.0.0.0/16",
                "instance_type": "t3.medium",
                "db_instance_class": "db.t3.large",
                "environment": "production"
            },
            secrets=["db_password", "api_keys", "ssl_certificates"],
            monitoring={
                "enabled": True,
                "metrics_interval": 60,
                "log_level": "INFO",
                "alerts": [
                    "high_cpu_usage",
                    "high_memory_usage",
                    "disk_space_low",
                    "database_connections_high",
                    "response_time_slow"
                ]
            },
            security_settings={
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "network_segmentation": True,
                "access_logging": True,
                "vulnerability_scanning": True,
                "compliance_checks": ["SOC2", "PCI-DSS"]
            }
        )

    async def test_complex_environment_registration(self, complex_environment):
        """Test registering a complex environment with dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(
                config_dir=Path(tmpdir) / "config",
                state_dir=Path(tmpdir) / "state"
            )

            await provisioner.register_environment(complex_environment)

            # Verify registration
            assert "production" in provisioner.environments
            registered_config = provisioner.environments["production"]
            assert len(registered_config.resources) == 6

            # Verify dependency resolution
            resource_names = {r.name for r in registered_config.resources}
            for resource in registered_config.resources:
                for dep in resource.dependencies:
                    assert dep in resource_names, f"Dependency {dep} not found for {resource.name}"

    async def test_complex_terraform_config_generation(self, complex_environment):
        """Test generating Terraform config for complex environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(config_dir=Path(tmpdir))
            await provisioner._generate_terraform_config(complex_environment)

            # Check generated files
            env_dir = Path(tmpdir) / "terraform" / "production"
            assert (env_dir / "main.tf").exists()
            assert (env_dir / "variables.tf").exists()
            assert (env_dir / "outputs.tf").exists()

            # Validate main.tf has all resources
            with open(env_dir / "main.tf") as f:
                main_config = json.load(f)

            # Check provider configuration
            assert "provider" in main_config
            assert "aws" in main_config["provider"]

            # Check resource generation
            assert "resource" in main_config
            resources = main_config["resource"]

            # Verify specific resource mappings
            if "aws_vpc" in resources:
                assert "main-vpc" in resources["aws_vpc"]

            if "aws_instance" in resources:
                instances = resources["aws_instance"]
                assert "web-server-1" in instances
                assert "web-server-2" in instances

    @patch('aiohttp.ClientSession.get')
    async def test_complex_health_checks(self, mock_get, complex_environment):
        """Test health checks for complex environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(config_dir=Path(tmpdir))
            await provisioner.register_environment(complex_environment)

            # Mock successful HTTP responses
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response

            # Mock successful TCP connections
            with patch('asyncio.open_connection') as mock_tcp:
                mock_reader = AsyncMock()
                mock_writer = AsyncMock()
                mock_writer.wait_closed = AsyncMock()
                mock_tcp.return_value = (mock_reader, mock_writer)

                # Mock successful commands
                with patch.object(provisioner, '_run_command') as mock_cmd:
                    mock_result = MagicMock()
                    mock_result.returncode = 0
                    mock_result.stdout = "accepting connections"
                    mock_result.stderr = ""
                    mock_cmd.return_value = mock_result

                    await provisioner._run_health_checks("production")

            status = provisioner.infrastructure_status["production"]
            health_checks = status.health_checks

            # Should have multiple health checks
            assert len(health_checks) > 0

            # Verify health check types are present
            check_types = {hc.check_type for hc in health_checks}
            assert "http" in check_types
            assert "tcp" in check_types
            assert "command" in check_types

    @patch('asyncio.create_subprocess_exec')
    async def test_complex_drift_detection(self, mock_subprocess, complex_environment):
        """Test drift detection for complex environment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(config_dir=Path(tmpdir))
            await provisioner.register_environment(complex_environment)

            # Mock terraform plan showing multiple resource changes
            mock_plan_process = AsyncMock()
            mock_plan_process.communicate.return_value = (b"Changes detected", b"")
            mock_plan_process.returncode = 2

            # Mock terraform show with complex changes
            mock_show_process = AsyncMock()
            plan_data = {
                "resource_changes": [
                    {
                        "address": "aws_instance.web-server-1",
                        "change": {
                            "actions": ["update"],
                            "before": {"instance_type": "t3.medium"},
                            "after": {"instance_type": "t3.large"}
                        }
                    },
                    {
                        "address": "aws_lb.web-lb",
                        "change": {
                            "actions": ["create"],
                            "before": None,
                            "after": {"type": "application"}
                        }
                    },
                    {
                        "address": "aws_db_instance.main-db",
                        "change": {
                            "actions": ["update"],
                            "before": {"instance_class": "db.t3.large"},
                            "after": {"instance_class": "db.t3.xlarge"}
                        }
                    }
                ]
            }
            mock_show_process.communicate.return_value = (json.dumps(plan_data).encode(), b"")
            mock_show_process.returncode = 0

            mock_subprocess.side_effect = [mock_plan_process, mock_show_process]

            drift_detected = await provisioner.detect_drift("production")

            assert drift_detected
            status = provisioner.infrastructure_status["production"]
            assert status.drift_detected
            assert len(status.drift_details) == 3
            assert status.state == InfrastructureState.DRIFTED

            # Verify specific drift details
            drift_resources = {detail["resource"] for detail in status.drift_details}
            assert "aws_instance.web-server-1" in drift_resources
            assert "aws_lb.web-lb" in drift_resources
            assert "aws_db_instance.main-db" in drift_resources

    async def test_partial_health_check_failures(self, complex_environment):
        """Test handling of partial health check failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(config_dir=Path(tmpdir))
            await provisioner.register_environment(complex_environment)

            # Mock mixed success/failure responses
            with patch('aiohttp.ClientSession.get') as mock_get:
                # HTTP checks: some pass, some fail
                responses = [
                    (200, "OK"),  # web-server-1 health
                    (500, "Error"),  # web-server-2 health (fail)
                    (200, "OK"),  # load balancer health
                ]

                async def mock_get_side_effect(*args, **kwargs):
                    response = AsyncMock()
                    status_code, text = responses.pop(0) if responses else (200, "OK")
                    response.status = status_code
                    response.text = AsyncMock(return_value=text)
                    return response

                mock_get.return_value.__aenter__ = mock_get_side_effect

                # Mock TCP connections: some succeed, some fail
                with patch('asyncio.open_connection') as mock_tcp:
                    call_count = [0]

                    async def mock_tcp_side_effect(host, port):
                        call_count[0] += 1
                        if call_count[0] % 3 == 0:  # Every third call fails
                            raise ConnectionRefusedError("Connection refused")
                        mock_reader = AsyncMock()
                        mock_writer = AsyncMock()
                        mock_writer.wait_closed = AsyncMock()
                        return mock_reader, mock_writer

                    mock_tcp.side_effect = mock_tcp_side_effect

                    # Mock commands: mixed results
                    with patch.object(provisioner, '_run_command') as mock_cmd:
                        call_count_cmd = [0]

                        async def mock_cmd_side_effect(cmd):
                            call_count_cmd[0] += 1
                            result = MagicMock()
                            if "pg_isready" in cmd:
                                result.returncode = 0
                                result.stdout = "accepting connections"
                            elif "aws" in cmd:
                                result.returncode = 1  # S3 check fails
                                result.stderr = "Access denied"
                            else:
                                result.returncode = 0
                                result.stdout = "OK"
                            result.stderr = ""
                            return result

                        mock_cmd.side_effect = mock_cmd_side_effect

                        await provisioner._run_health_checks("production")

            status = provisioner.infrastructure_status["production"]
            health_checks = status.health_checks

            # Should have health checks with mixed results
            passed_checks = [hc for hc in health_checks if hc.status == "passed"]
            failed_checks = [hc for hc in health_checks if hc.status == "failed"]

            assert len(passed_checks) > 0
            assert len(failed_checks) > 0
            assert len(passed_checks) + len(failed_checks) == len(health_checks)

    async def test_environment_lifecycle_complete(self, complex_environment):
        """Test complete environment lifecycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(config_dir=Path(tmpdir))

            # 1. Register environment
            await provisioner.register_environment(complex_environment)
            assert "production" in provisioner.environments

            # 2. Provision (dry run)
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b"Plan complete", b"")
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process

                status = await provisioner.provision_environment("production", dry_run=True)
                assert status.state == InfrastructureState.PLANNING

            # 3. Detect drift (simulate changes)
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_plan_process = AsyncMock()
                mock_plan_process.communicate.return_value = (b"Changes", b"")
                mock_plan_process.returncode = 2

                mock_show_process = AsyncMock()
                plan_data = {"resource_changes": []}
                mock_show_process.communicate.return_value = (json.dumps(plan_data).encode(), b"")
                mock_show_process.returncode = 0

                mock_subprocess.side_effect = [mock_plan_process, mock_show_process]

                drift_detected = await provisioner.detect_drift("production")
                assert not drift_detected  # No actual changes in mock

            # 4. Generate status report
            report = provisioner.generate_status_report()
            assert report["environments"]["production"]["state"] == "planning"

            # 5. Clean up
            with patch('asyncio.create_subprocess_exec') as mock_subprocess:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b"Destroy complete", b"")
                mock_process.returncode = 0
                mock_subprocess.return_value = mock_process

                await provisioner.destroy_environment("production")

            assert "production" not in provisioner.environments


# Test data for edge cases
EDGE_CASE_CONFIGS = [
    # Empty properties
    ResourceConfiguration(
        resource_type="minimal",
        name="empty-resource",
        properties={}
    ),
    # Maximum complexity
    ResourceConfiguration(
        resource_type="complex",
        name="complex-resource",
        properties={
            "nested": {
                "deeply": {
                    "nested": {
                        "values": ["a", "b", "c"],
                        "mapping": {"key1": "value1", "key2": "value2"}
                    }
                }
            },
            "arrays": [
                {"item1": "value1"},
                {"item2": "value2"}
            ],
            "booleans": [True, False],
            "numbers": [1, 2.5, -3]
        },
        dependencies=["dep1", "dep2", "dep3"],
        tags={
            "env": "test",
            "version": "1.0",
            "critical": "true"
        },
        health_checks=[
            {"type": "http", "url": "http://example.com", "timeout": 5},
            {"type": "tcp", "host": "localhost", "port": 8080},
            {"type": "command", "command": ["echo", "test"]},
        ]
    )
]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_empty_resource_properties(self):
        """Test resource with empty properties."""
        resource = EDGE_CASE_CONFIGS[0]
        assert resource.properties == {}
        assert resource.dependencies == []
        assert resource.health_checks == []

    async def test_complex_nested_properties(self):
        """Test resource with deeply nested properties."""
        resource = EDGE_CASE_CONFIGS[1]
        assert resource.properties["nested"]["deeply"]["nested"]["values"] == ["a", "b", "c"]
        assert len(resource.health_checks) == 3
        assert len(resource.dependencies) == 3

    @patch('asyncio.create_subprocess_exec')
    async def test_command_timeout_handling(self, mock_subprocess):
        """Test handling of command timeouts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(config_dir=Path(tmpdir))

            # Mock process that hangs
            mock_subprocess.side_effect = asyncio.TimeoutError("Command timed out")

            with pytest.raises(asyncio.TimeoutError):
                await provisioner._run_command(["sleep", "300"])

    async def test_invalid_yaml_config_handling(self):
        """Test handling of invalid YAML configurations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(config_dir=Path(tmpdir))

            # Create invalid YAML file
            invalid_yaml_path = Path(tmpdir) / "invalid.yaml"
            with open(invalid_yaml_path, "w") as f:
                f.write("invalid: yaml: content: [unclosed")

            # Should handle gracefully when parsing
            try:
                with open(invalid_yaml_path) as f:
                    yaml.safe_load(f)
            except yaml.YAMLError:
                # Expected behavior
                pass

    async def test_concurrent_operations(self):
        """Test concurrent infrastructure operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provisioner = InfrastructureProvisioner(config_dir=Path(tmpdir))

            # Create multiple environments
            envs = []
            for i in range(3):
                resources = [
                    ResourceConfiguration(
                        resource_type="vm",
                        name=f"vm-{i}",
                        properties={"instance_type": "t3.micro"}
                    )
                ]
                env = EnvironmentConfig(
                    name=f"env-{i}",
                    environment_type=EnvironmentType.TEST,
                    provider=InfrastructureProvider.DOCKER,
                    region="local",
                    resources=resources
                )
                envs.append(env)

            # Register all environments concurrently
            await asyncio.gather(*[
                provisioner.register_environment(env) for env in envs
            ])

            # Verify all were registered
            assert len(provisioner.environments) == 3
            assert all(f"env-{i}" in provisioner.environments for i in range(3))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])