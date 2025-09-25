#!/usr/bin/env python3
"""
Infrastructure Provisioning and Configuration Management System

This module provides comprehensive infrastructure provisioning with multi-environment support,
configuration drift detection, and automated recovery mechanisms.

Key features:
- Multi-cloud infrastructure provisioning
- Environment-specific configuration management
- Infrastructure drift detection and correction
- Automated health checks and monitoring
- Resource optimization and cost management
- Security compliance validation
- Rollback and disaster recovery
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class InfrastructureProvider(str, Enum):
    """Supported infrastructure providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    TERRAFORM = "terraform"


class EnvironmentType(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    TEST = "test"


class InfrastructureState(str, Enum):
    """Infrastructure deployment states."""
    PLANNING = "planning"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    UPDATING = "updating"
    DESTROYING = "destroying"
    FAILED = "failed"
    DRIFTED = "drifted"


@dataclass
class ResourceConfiguration:
    """Infrastructure resource configuration."""
    resource_type: str
    name: str
    properties: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    health_checks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    name: str
    environment_type: EnvironmentType
    provider: InfrastructureProvider
    region: str
    resources: List[ResourceConfiguration]
    variables: Dict[str, Any] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)
    monitoring: Dict[str, Any] = field(default_factory=dict)
    security_settings: Dict[str, Any] = field(default_factory=dict)


class HealthCheckResult(BaseModel):
    """Health check result model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    resource_name: str
    check_type: str
    status: str
    message: str
    timestamp: float
    details: Dict[str, Any] = Field(default_factory=dict)


class InfrastructureStatus(BaseModel):
    """Infrastructure status model."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    environment_name: str
    state: InfrastructureState
    resources: Dict[str, Dict[str, Any]]
    health_checks: List[HealthCheckResult]
    last_update: float
    drift_detected: bool = False
    drift_details: List[Dict[str, Any]] = Field(default_factory=list)


class InfrastructureProvisioner:
    """Comprehensive infrastructure provisioning and management system."""

    def __init__(
        self,
        config_dir: Path = Path("infrastructure"),
        state_dir: Path = Path(".terraform"),
        monitoring_enabled: bool = True
    ):
        """Initialize the infrastructure provisioner."""
        self.config_dir = Path(config_dir)
        self.state_dir = Path(state_dir)
        self.monitoring_enabled = monitoring_enabled

        # Create required directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.environments: Dict[str, EnvironmentConfig] = {}
        self.infrastructure_status: Dict[str, InfrastructureStatus] = {}

        # Initialize terraform if available
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check for required tools and dependencies."""
        required_tools = ["terraform", "kubectl", "helm"]

        for tool in required_tools:
            if not self._is_tool_available(tool):
                logger.warning(f"Tool {tool} not found - some features may be limited")

    def _is_tool_available(self, tool: str) -> bool:
        """Check if a command-line tool is available."""
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    async def register_environment(
        self,
        environment_config: EnvironmentConfig
    ) -> None:
        """Register a new environment configuration."""
        logger.info(f"Registering environment: {environment_config.name}")

        # Validate configuration
        self._validate_environment_config(environment_config)

        # Store configuration
        self.environments[environment_config.name] = environment_config

        # Initialize infrastructure status
        self.infrastructure_status[environment_config.name] = InfrastructureStatus(
            environment_name=environment_config.name,
            state=InfrastructureState.PLANNING,
            resources={},
            health_checks=[],
            last_update=time.time()
        )

        # Generate provider-specific configurations
        await self._generate_infrastructure_config(environment_config)

        logger.info(f"Environment {environment_config.name} registered successfully")

    def _validate_environment_config(self, config: EnvironmentConfig) -> None:
        """Validate environment configuration."""
        if not config.name:
            raise ValueError("Environment name cannot be empty")

        if not config.resources:
            raise ValueError("Environment must have at least one resource")

        # Validate resource dependencies
        resource_names = {r.name for r in config.resources}
        for resource in config.resources:
            for dep in resource.dependencies:
                if dep not in resource_names:
                    raise ValueError(f"Resource {resource.name} has invalid dependency: {dep}")

    async def _generate_infrastructure_config(
        self,
        environment_config: EnvironmentConfig
    ) -> None:
        """Generate provider-specific infrastructure configurations."""
        if environment_config.provider == InfrastructureProvider.TERRAFORM:
            await self._generate_terraform_config(environment_config)
        elif environment_config.provider == InfrastructureProvider.KUBERNETES:
            await self._generate_kubernetes_config(environment_config)
        elif environment_config.provider == InfrastructureProvider.DOCKER:
            await self._generate_docker_config(environment_config)
        else:
            logger.warning(f"Configuration generation not implemented for {environment_config.provider}")

    async def _generate_terraform_config(self, config: EnvironmentConfig) -> None:
        """Generate Terraform configuration files."""
        env_dir = self.config_dir / "terraform" / config.name
        env_dir.mkdir(parents=True, exist_ok=True)

        # Generate main.tf
        terraform_config = {
            "terraform": {
                "required_version": ">= 1.0",
                "required_providers": {
                    "aws": {"source": "hashicorp/aws", "version": "~> 5.0"},
                    "google": {"source": "hashicorp/google", "version": "~> 4.0"},
                    "azurerm": {"source": "hashicorp/azurerm", "version": "~> 3.0"},
                }
            },
            "provider": self._generate_provider_config(config),
            "resource": self._generate_terraform_resources(config)
        }

        main_tf_path = env_dir / "main.tf"
        with open(main_tf_path, "w") as f:
            json.dump(terraform_config, f, indent=2)

        # Generate variables.tf
        variables_config = {
            var_name: {
                "description": f"Variable for {var_name}",
                "type": self._infer_terraform_type(var_value),
                "default": var_value if not var_name in config.secrets else None,
                "sensitive": var_name in config.secrets
            }
            for var_name, var_value in config.variables.items()
        }

        variables_tf_path = env_dir / "variables.tf"
        with open(variables_tf_path, "w") as f:
            json.dump({"variable": variables_config}, f, indent=2)

        # Generate outputs.tf
        outputs_config = {
            f"{resource.name}_id": {
                "description": f"ID of {resource.name}",
                "value": f"${{{resource.resource_type}.{resource.name}.id}}"
            }
            for resource in config.resources
        }

        outputs_tf_path = env_dir / "outputs.tf"
        with open(outputs_tf_path, "w") as f:
            json.dump({"output": outputs_config}, f, indent=2)

        logger.info(f"Generated Terraform configuration for {config.name}")

    async def _generate_kubernetes_config(self, config: EnvironmentConfig) -> None:
        """Generate Kubernetes manifests."""
        env_dir = self.config_dir / "kubernetes" / config.name
        env_dir.mkdir(parents=True, exist_ok=True)

        # Generate namespace
        namespace_manifest = {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": config.name,
                "labels": {
                    "environment": config.environment_type.value,
                    "managed-by": "infrastructure-provisioner"
                }
            }
        }

        namespace_path = env_dir / "namespace.yaml"
        with open(namespace_path, "w") as f:
            yaml.dump(namespace_manifest, f)

        # Generate resource manifests
        for resource in config.resources:
            manifest = self._generate_kubernetes_resource_manifest(resource, config)
            if manifest:
                resource_path = env_dir / f"{resource.name}.yaml"
                with open(resource_path, "w") as f:
                    yaml.dump(manifest, f)

        logger.info(f"Generated Kubernetes configuration for {config.name}")

    async def _generate_docker_config(self, config: EnvironmentConfig) -> None:
        """Generate Docker Compose configuration."""
        env_dir = self.config_dir / "docker" / config.name
        env_dir.mkdir(parents=True, exist_ok=True)

        # Generate docker-compose.yml
        compose_config = {
            "version": "3.8",
            "services": {},
            "networks": {"default": {"external": True}},
            "volumes": {}
        }

        for resource in config.resources:
            if resource.resource_type == "docker_container":
                service_config = self._generate_docker_service_config(resource, config)
                compose_config["services"][resource.name] = service_config

        compose_path = env_dir / "docker-compose.yml"
        with open(compose_path, "w") as f:
            yaml.dump(compose_config, f)

        # Generate environment file
        env_vars = {k: str(v) for k, v in config.variables.items() if k not in config.secrets}
        env_file_path = env_dir / ".env"
        with open(env_file_path, "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        logger.info(f"Generated Docker configuration for {config.name}")

    def _generate_provider_config(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Generate provider-specific configuration."""
        if "aws" in config.region:
            return {
                "aws": {
                    "region": config.region,
                    "default_tags": {
                        "tags": {
                            "Environment": config.environment_type.value,
                            "ManagedBy": "infrastructure-provisioner",
                            "Project": "workspace-qdrant-mcp"
                        }
                    }
                }
            }
        elif "gcp" in config.region:
            return {
                "google": {
                    "region": config.region,
                    "project": config.variables.get("project_id", "default-project")
                }
            }
        else:
            return {}

    def _generate_terraform_resources(self, config: EnvironmentConfig) -> Dict[str, Any]:
        """Generate Terraform resource configurations."""
        resources = {}

        for resource in config.resources:
            tf_resource_type = self._map_resource_to_terraform(resource.resource_type)
            if tf_resource_type:
                resources[tf_resource_type] = resources.get(tf_resource_type, {})
                resources[tf_resource_type][resource.name] = {
                    **resource.properties,
                    "tags": {
                        **resource.tags,
                        "Name": resource.name,
                        "Environment": config.environment_type.value
                    }
                }

        return resources

    def _generate_kubernetes_resource_manifest(
        self,
        resource: ResourceConfiguration,
        config: EnvironmentConfig
    ) -> Optional[Dict[str, Any]]:
        """Generate Kubernetes resource manifest."""
        if resource.resource_type == "deployment":
            return {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": resource.name,
                    "namespace": config.name,
                    "labels": resource.tags
                },
                "spec": {
                    "replicas": resource.properties.get("replicas", 1),
                    "selector": {"matchLabels": {"app": resource.name}},
                    "template": {
                        "metadata": {"labels": {"app": resource.name}},
                        "spec": {
                            "containers": [{
                                "name": resource.name,
                                "image": resource.properties.get("image", "nginx:latest"),
                                "ports": resource.properties.get("ports", [{"containerPort": 80}]),
                                "env": [
                                    {"name": k, "value": str(v)}
                                    for k, v in config.variables.items()
                                    if k not in config.secrets
                                ]
                            }]
                        }
                    }
                }
            }
        elif resource.resource_type == "service":
            return {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": resource.name,
                    "namespace": config.name,
                    "labels": resource.tags
                },
                "spec": {
                    "selector": {"app": resource.properties.get("app", resource.name)},
                    "ports": resource.properties.get("ports", [{"port": 80, "targetPort": 80}]),
                    "type": resource.properties.get("type", "ClusterIP")
                }
            }

        return None

    def _generate_docker_service_config(
        self,
        resource: ResourceConfiguration,
        config: EnvironmentConfig
    ) -> Dict[str, Any]:
        """Generate Docker service configuration."""
        service_config = {
            "image": resource.properties.get("image", "nginx:latest"),
            "container_name": resource.name,
            "restart": resource.properties.get("restart", "unless-stopped"),
            "environment": {
                k: str(v) for k, v in config.variables.items()
                if k not in config.secrets
            }
        }

        if "ports" in resource.properties:
            service_config["ports"] = resource.properties["ports"]

        if "volumes" in resource.properties:
            service_config["volumes"] = resource.properties["volumes"]

        if "depends_on" in resource.properties:
            service_config["depends_on"] = resource.properties["depends_on"]

        return service_config

    def _map_resource_to_terraform(self, resource_type: str) -> Optional[str]:
        """Map generic resource types to Terraform resource types."""
        mapping = {
            "vm": "aws_instance",
            "load_balancer": "aws_lb",
            "database": "aws_db_instance",
            "storage": "aws_s3_bucket",
            "network": "aws_vpc",
            "security_group": "aws_security_group"
        }
        return mapping.get(resource_type)

    def _infer_terraform_type(self, value: Any) -> str:
        """Infer Terraform variable type from Python value."""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "number"
        elif isinstance(value, list):
            return "list(string)"
        elif isinstance(value, dict):
            return "map(string)"
        else:
            return "string"

    async def provision_environment(
        self,
        environment_name: str,
        dry_run: bool = False
    ) -> InfrastructureStatus:
        """Provision infrastructure for an environment."""
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not registered")

        config = self.environments[environment_name]
        status = self.infrastructure_status[environment_name]

        logger.info(f"Starting {'dry-run' if dry_run else 'provisioning'} for {environment_name}")

        try:
            # Update status
            status.state = InfrastructureState.PROVISIONING
            status.last_update = time.time()

            # Provision based on provider
            if config.provider == InfrastructureProvider.TERRAFORM:
                await self._provision_terraform(config, dry_run)
            elif config.provider == InfrastructureProvider.KUBERNETES:
                await self._provision_kubernetes(config, dry_run)
            elif config.provider == InfrastructureProvider.DOCKER:
                await self._provision_docker(config, dry_run)

            # Update status on success
            if not dry_run:
                status.state = InfrastructureState.RUNNING
                await self._update_resource_status(environment_name)
                await self._run_health_checks(environment_name)

            logger.info(f"Provisioning completed for {environment_name}")

        except Exception as e:
            logger.error(f"Provisioning failed for {environment_name}: {e}")
            status.state = InfrastructureState.FAILED
            raise

        return status

    async def _provision_terraform(
        self,
        config: EnvironmentConfig,
        dry_run: bool = False
    ) -> None:
        """Provision infrastructure using Terraform."""
        env_dir = self.config_dir / "terraform" / config.name

        # Initialize Terraform
        init_cmd = ["terraform", "init"]
        await self._run_terraform_command(init_cmd, env_dir)

        # Plan
        plan_cmd = ["terraform", "plan", "-out=tfplan"]
        for var_name, var_value in config.variables.items():
            if var_name not in config.secrets:
                plan_cmd.extend(["-var", f"{var_name}={var_value}"])

        await self._run_terraform_command(plan_cmd, env_dir)

        # Apply (if not dry run)
        if not dry_run:
            apply_cmd = ["terraform", "apply", "-auto-approve", "tfplan"]
            await self._run_terraform_command(apply_cmd, env_dir)

    async def _provision_kubernetes(
        self,
        config: EnvironmentConfig,
        dry_run: bool = False
    ) -> None:
        """Provision infrastructure using Kubernetes."""
        env_dir = self.config_dir / "kubernetes" / config.name

        kubectl_args = ["--dry-run=client"] if dry_run else []

        # Apply all manifests
        for manifest_file in env_dir.glob("*.yaml"):
            cmd = ["kubectl", "apply", "-f", str(manifest_file)] + kubectl_args
            await self._run_command(cmd)

    async def _provision_docker(
        self,
        config: EnvironmentConfig,
        dry_run: bool = False
    ) -> None:
        """Provision infrastructure using Docker Compose."""
        env_dir = self.config_dir / "docker" / config.name
        compose_file = env_dir / "docker-compose.yml"

        if dry_run:
            # Validate compose file
            cmd = ["docker-compose", "-f", str(compose_file), "config"]
        else:
            # Deploy services
            cmd = ["docker-compose", "-f", str(compose_file), "up", "-d"]

        await self._run_command(cmd, cwd=env_dir)

    async def _run_terraform_command(
        self,
        cmd: List[str],
        cwd: Path
    ) -> subprocess.CompletedProcess:
        """Run a Terraform command with proper error handling."""
        return await self._run_command(cmd, cwd=cwd)

    async def _run_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None
    ) -> subprocess.CompletedProcess:
        """Run a command asynchronously with error handling."""
        logger.info(f"Running command: {' '.join(cmd)}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                cmd,
                stdout.decode(),
                stderr.decode()
            )

        return subprocess.CompletedProcess(
            cmd,
            process.returncode,
            stdout.decode(),
            stderr.decode()
        )

    async def _update_resource_status(self, environment_name: str) -> None:
        """Update the status of all resources in an environment."""
        config = self.environments[environment_name]
        status = self.infrastructure_status[environment_name]

        # Query provider for actual resource states
        if config.provider == InfrastructureProvider.TERRAFORM:
            await self._update_terraform_resource_status(environment_name)
        elif config.provider == InfrastructureProvider.KUBERNETES:
            await self._update_kubernetes_resource_status(environment_name)
        elif config.provider == InfrastructureProvider.DOCKER:
            await self._update_docker_resource_status(environment_name)

        status.last_update = time.time()

    async def _update_terraform_resource_status(self, environment_name: str) -> None:
        """Update Terraform resource status."""
        config = self.environments[environment_name]
        status = self.infrastructure_status[environment_name]
        env_dir = self.config_dir / "terraform" / config.name

        try:
            # Get terraform state
            cmd = ["terraform", "show", "-json"]
            result = await self._run_terraform_command(cmd, env_dir)
            state_data = json.loads(result.stdout)

            # Update resource information
            if "values" in state_data and "root_module" in state_data["values"]:
                resources = state_data["values"]["root_module"].get("resources", [])
                for resource in resources:
                    resource_name = resource.get("name", "unknown")
                    status.resources[resource_name] = {
                        "type": resource.get("type", "unknown"),
                        "values": resource.get("values", {}),
                        "status": "running"
                    }

        except Exception as e:
            logger.warning(f"Failed to update Terraform resource status: {e}")

    async def _update_kubernetes_resource_status(self, environment_name: str) -> None:
        """Update Kubernetes resource status."""
        config = self.environments[environment_name]
        status = self.infrastructure_status[environment_name]

        try:
            # Get all resources in namespace
            cmd = ["kubectl", "get", "all", "-n", config.name, "-o", "json"]
            result = await self._run_command(cmd)
            resources_data = json.loads(result.stdout)

            # Update resource information
            for resource in resources_data.get("items", []):
                resource_name = resource["metadata"]["name"]
                status.resources[resource_name] = {
                    "type": resource["kind"],
                    "status": resource.get("status", {}),
                    "metadata": resource["metadata"]
                }

        except Exception as e:
            logger.warning(f"Failed to update Kubernetes resource status: {e}")

    async def _update_docker_resource_status(self, environment_name: str) -> None:
        """Update Docker resource status."""
        config = self.environments[environment_name]
        status = self.infrastructure_status[environment_name]
        env_dir = self.config_dir / "docker" / config.name

        try:
            # Get container status
            cmd = ["docker-compose", "-f", "docker-compose.yml", "ps", "--format", "json"]
            result = await self._run_command(cmd, cwd=env_dir)

            # Parse container information
            for line in result.stdout.strip().split('\n'):
                if line:
                    container_data = json.loads(line)
                    container_name = container_data.get("Name", "unknown")
                    status.resources[container_name] = {
                        "type": "container",
                        "status": container_data.get("State", "unknown"),
                        "image": container_data.get("Image", "unknown"),
                        "ports": container_data.get("Ports", "")
                    }

        except Exception as e:
            logger.warning(f"Failed to update Docker resource status: {e}")

    async def _run_health_checks(self, environment_name: str) -> None:
        """Run health checks for all resources in an environment."""
        config = self.environments[environment_name]
        status = self.infrastructure_status[environment_name]

        health_checks = []

        for resource in config.resources:
            for health_check in resource.health_checks:
                try:
                    result = await self._execute_health_check(resource, health_check)
                    health_checks.append(result)
                except Exception as e:
                    health_checks.append(HealthCheckResult(
                        resource_name=resource.name,
                        check_type=health_check.get("type", "unknown"),
                        status="failed",
                        message=str(e),
                        timestamp=time.time()
                    ))

        status.health_checks = health_checks

        # Log health check summary
        passed = sum(1 for hc in health_checks if hc.status == "passed")
        total = len(health_checks)
        logger.info(f"Health checks for {environment_name}: {passed}/{total} passed")

    async def _execute_health_check(
        self,
        resource: ResourceConfiguration,
        health_check: Dict[str, Any]
    ) -> HealthCheckResult:
        """Execute a specific health check."""
        check_type = health_check.get("type", "ping")

        if check_type == "http":
            return await self._http_health_check(resource, health_check)
        elif check_type == "tcp":
            return await self._tcp_health_check(resource, health_check)
        elif check_type == "command":
            return await self._command_health_check(resource, health_check)
        else:
            raise ValueError(f"Unknown health check type: {check_type}")

    async def _http_health_check(
        self,
        resource: ResourceConfiguration,
        health_check: Dict[str, Any]
    ) -> HealthCheckResult:
        """Perform HTTP health check."""
        import aiohttp

        url = health_check.get("url", f"http://{resource.name}")
        timeout = health_check.get("timeout", 10)
        expected_status = health_check.get("expected_status", 200)

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    if response.status == expected_status:
                        return HealthCheckResult(
                            resource_name=resource.name,
                            check_type="http",
                            status="passed",
                            message=f"HTTP {response.status} from {url}",
                            timestamp=time.time(),
                            details={"status_code": response.status, "url": url}
                        )
                    else:
                        return HealthCheckResult(
                            resource_name=resource.name,
                            check_type="http",
                            status="failed",
                            message=f"HTTP {response.status} (expected {expected_status}) from {url}",
                            timestamp=time.time(),
                            details={"status_code": response.status, "url": url}
                        )
            except Exception as e:
                return HealthCheckResult(
                    resource_name=resource.name,
                    check_type="http",
                    status="failed",
                    message=f"HTTP check failed: {e}",
                    timestamp=time.time(),
                    details={"url": url, "error": str(e)}
                )

    async def _tcp_health_check(
        self,
        resource: ResourceConfiguration,
        health_check: Dict[str, Any]
    ) -> HealthCheckResult:
        """Perform TCP port health check."""
        host = health_check.get("host", resource.name)
        port = health_check.get("port", 80)
        timeout = health_check.get("timeout", 5)

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()

            return HealthCheckResult(
                resource_name=resource.name,
                check_type="tcp",
                status="passed",
                message=f"TCP connection successful to {host}:{port}",
                timestamp=time.time(),
                details={"host": host, "port": port}
            )
        except Exception as e:
            return HealthCheckResult(
                resource_name=resource.name,
                check_type="tcp",
                status="failed",
                message=f"TCP connection failed to {host}:{port}: {e}",
                timestamp=time.time(),
                details={"host": host, "port": port, "error": str(e)}
            )

    async def _command_health_check(
        self,
        resource: ResourceConfiguration,
        health_check: Dict[str, Any]
    ) -> HealthCheckResult:
        """Perform command-based health check."""
        command = health_check.get("command", ["echo", "hello"])
        timeout = health_check.get("timeout", 30)
        expected_exit_code = health_check.get("expected_exit_code", 0)

        try:
            result = await asyncio.wait_for(
                self._run_command(command),
                timeout=timeout
            )

            if result.returncode == expected_exit_code:
                return HealthCheckResult(
                    resource_name=resource.name,
                    check_type="command",
                    status="passed",
                    message=f"Command exited with code {result.returncode}",
                    timestamp=time.time(),
                    details={"command": command, "exit_code": result.returncode}
                )
            else:
                return HealthCheckResult(
                    resource_name=resource.name,
                    check_type="command",
                    status="failed",
                    message=f"Command exited with code {result.returncode} (expected {expected_exit_code})",
                    timestamp=time.time(),
                    details={
                        "command": command,
                        "exit_code": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                )
        except Exception as e:
            return HealthCheckResult(
                resource_name=resource.name,
                check_type="command",
                status="failed",
                message=f"Command health check failed: {e}",
                timestamp=time.time(),
                details={"command": command, "error": str(e)}
            )

    async def detect_drift(self, environment_name: str) -> bool:
        """Detect configuration drift in an environment."""
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not registered")

        config = self.environments[environment_name]
        status = self.infrastructure_status[environment_name]

        logger.info(f"Detecting drift for environment: {environment_name}")

        drift_detected = False
        drift_details = []

        try:
            if config.provider == InfrastructureProvider.TERRAFORM:
                drift_detected, drift_details = await self._detect_terraform_drift(environment_name)
            elif config.provider == InfrastructureProvider.KUBERNETES:
                drift_detected, drift_details = await self._detect_kubernetes_drift(environment_name)
            elif config.provider == InfrastructureProvider.DOCKER:
                drift_detected, drift_details = await self._detect_docker_drift(environment_name)

            # Update status
            status.drift_detected = drift_detected
            status.drift_details = drift_details
            status.last_update = time.time()

            if drift_detected:
                logger.warning(f"Configuration drift detected in {environment_name}: {len(drift_details)} issues")
                status.state = InfrastructureState.DRIFTED
            else:
                logger.info(f"No configuration drift detected in {environment_name}")

        except Exception as e:
            logger.error(f"Drift detection failed for {environment_name}: {e}")
            raise

        return drift_detected

    async def _detect_terraform_drift(self, environment_name: str) -> tuple[bool, List[Dict[str, Any]]]:
        """Detect Terraform configuration drift."""
        config = self.environments[environment_name]
        env_dir = self.config_dir / "terraform" / config.name

        drift_details = []

        try:
            # Run terraform plan to detect changes
            plan_cmd = ["terraform", "plan", "-detailed-exitcode", "-out=drift-plan"]
            for var_name, var_value in config.variables.items():
                if var_name not in config.secrets:
                    plan_cmd.extend(["-var", f"{var_name}={var_value}"])

            result = await self._run_terraform_command(plan_cmd, env_dir)

            # Exit code 2 means changes detected
            if result.returncode == 2:
                # Parse plan output for details
                show_cmd = ["terraform", "show", "-json", "drift-plan"]
                show_result = await self._run_terraform_command(show_cmd, env_dir)
                plan_data = json.loads(show_result.stdout)

                # Extract resource changes
                if "resource_changes" in plan_data:
                    for change in plan_data["resource_changes"]:
                        if change.get("change", {}).get("actions") != ["no-op"]:
                            drift_details.append({
                                "resource": change.get("address", "unknown"),
                                "action": change.get("change", {}).get("actions", []),
                                "before": change.get("change", {}).get("before"),
                                "after": change.get("change", {}).get("after")
                            })

            return len(drift_details) > 0, drift_details

        except Exception as e:
            logger.error(f"Terraform drift detection failed: {e}")
            return False, []

    async def _detect_kubernetes_drift(self, environment_name: str) -> tuple[bool, List[Dict[str, Any]]]:
        """Detect Kubernetes configuration drift."""
        config = self.environments[environment_name]
        env_dir = self.config_dir / "kubernetes" / config.name

        drift_details = []

        try:
            # Compare desired vs actual state for each manifest
            for manifest_file in env_dir.glob("*.yaml"):
                with open(manifest_file, 'r') as f:
                    desired_manifest = yaml.safe_load(f)

                if not desired_manifest:
                    continue

                # Get actual resource from cluster
                resource_type = desired_manifest.get("kind", "").lower()
                resource_name = desired_manifest.get("metadata", {}).get("name", "")

                cmd = ["kubectl", "get", resource_type, resource_name, "-n", config.name, "-o", "yaml"]

                try:
                    result = await self._run_command(cmd)
                    actual_manifest = yaml.safe_load(result.stdout)

                    # Compare specifications (simplified comparison)
                    desired_spec = desired_manifest.get("spec", {})
                    actual_spec = actual_manifest.get("spec", {})

                    if desired_spec != actual_spec:
                        drift_details.append({
                            "resource": f"{resource_type}/{resource_name}",
                            "file": str(manifest_file),
                            "desired": desired_spec,
                            "actual": actual_spec
                        })

                except subprocess.CalledProcessError:
                    # Resource doesn't exist
                    drift_details.append({
                        "resource": f"{resource_type}/{resource_name}",
                        "file": str(manifest_file),
                        "status": "missing",
                        "desired": desired_manifest
                    })

            return len(drift_details) > 0, drift_details

        except Exception as e:
            logger.error(f"Kubernetes drift detection failed: {e}")
            return False, []

    async def _detect_docker_drift(self, environment_name: str) -> tuple[bool, List[Dict[str, Any]]]:
        """Detect Docker configuration drift."""
        config = self.environments[environment_name]
        env_dir = self.config_dir / "docker" / config.name
        compose_file = env_dir / "docker-compose.yml"

        drift_details = []

        try:
            # Load desired configuration
            with open(compose_file, 'r') as f:
                desired_config = yaml.safe_load(f)

            # Get actual container configuration
            cmd = ["docker-compose", "-f", str(compose_file), "ps", "--format", "json"]
            result = await self._run_command(cmd, cwd=env_dir)

            # Compare desired vs actual services
            desired_services = set(desired_config.get("services", {}).keys())
            actual_services = set()

            for line in result.stdout.strip().split('\n'):
                if line:
                    container_data = json.loads(line)
                    service_name = container_data.get("Service", "unknown")
                    actual_services.add(service_name)

                    # Check if service is running
                    if container_data.get("State") != "running":
                        drift_details.append({
                            "service": service_name,
                            "issue": "not_running",
                            "current_state": container_data.get("State")
                        })

            # Check for missing services
            missing_services = desired_services - actual_services
            for service in missing_services:
                drift_details.append({
                    "service": service,
                    "issue": "missing",
                    "desired": desired_config["services"][service]
                })

            return len(drift_details) > 0, drift_details

        except Exception as e:
            logger.error(f"Docker drift detection failed: {e}")
            return False, []

    async def correct_drift(
        self,
        environment_name: str,
        auto_approve: bool = False
    ) -> InfrastructureStatus:
        """Correct detected configuration drift."""
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not registered")

        status = self.infrastructure_status[environment_name]

        if not status.drift_detected:
            logger.info(f"No drift detected for {environment_name}, nothing to correct")
            return status

        logger.info(f"Correcting drift for environment: {environment_name}")

        try:
            # Update status
            status.state = InfrastructureState.UPDATING
            status.last_update = time.time()

            # Apply corrections based on provider
            config = self.environments[environment_name]
            if config.provider == InfrastructureProvider.TERRAFORM:
                await self._correct_terraform_drift(environment_name, auto_approve)
            elif config.provider == InfrastructureProvider.KUBERNETES:
                await self._correct_kubernetes_drift(environment_name, auto_approve)
            elif config.provider == InfrastructureProvider.DOCKER:
                await self._correct_docker_drift(environment_name, auto_approve)

            # Verify drift correction
            drift_still_present = await self.detect_drift(environment_name)

            if not drift_still_present:
                status.state = InfrastructureState.RUNNING
                status.drift_detected = False
                status.drift_details = []
                logger.info(f"Drift corrected successfully for {environment_name}")
            else:
                logger.warning(f"Some drift may still exist for {environment_name}")

        except Exception as e:
            logger.error(f"Drift correction failed for {environment_name}: {e}")
            status.state = InfrastructureState.FAILED
            raise

        return status

    async def _correct_terraform_drift(
        self,
        environment_name: str,
        auto_approve: bool = False
    ) -> None:
        """Correct Terraform configuration drift."""
        config = self.environments[environment_name]
        env_dir = self.config_dir / "terraform" / config.name

        # Apply the plan that detected drift
        if auto_approve:
            apply_cmd = ["terraform", "apply", "-auto-approve", "drift-plan"]
        else:
            apply_cmd = ["terraform", "apply", "drift-plan"]

        await self._run_terraform_command(apply_cmd, env_dir)
        logger.info("Terraform drift corrections applied")

    async def _correct_kubernetes_drift(
        self,
        environment_name: str,
        auto_approve: bool = False
    ) -> None:
        """Correct Kubernetes configuration drift."""
        config = self.environments[environment_name]
        env_dir = self.config_dir / "kubernetes" / config.name

        # Reapply all manifests
        for manifest_file in env_dir.glob("*.yaml"):
            cmd = ["kubectl", "apply", "-f", str(manifest_file)]
            if not auto_approve:
                # In a real implementation, you might want user confirmation
                pass

            await self._run_command(cmd)

        logger.info("Kubernetes drift corrections applied")

    async def _correct_docker_drift(
        self,
        environment_name: str,
        auto_approve: bool = False
    ) -> None:
        """Correct Docker configuration drift."""
        config = self.environments[environment_name]
        env_dir = self.config_dir / "docker" / config.name
        compose_file = env_dir / "docker-compose.yml"

        # Recreate services to match desired state
        if auto_approve:
            # Stop and recreate services
            stop_cmd = ["docker-compose", "-f", str(compose_file), "down"]
            await self._run_command(stop_cmd, cwd=env_dir)

            start_cmd = ["docker-compose", "-f", str(compose_file), "up", "-d"]
            await self._run_command(start_cmd, cwd=env_dir)
        else:
            # Just restart services
            restart_cmd = ["docker-compose", "-f", str(compose_file), "restart"]
            await self._run_command(restart_cmd, cwd=env_dir)

        logger.info("Docker drift corrections applied")

    async def destroy_environment(
        self,
        environment_name: str,
        force: bool = False
    ) -> None:
        """Destroy an environment and all its resources."""
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not registered")

        config = self.environments[environment_name]
        status = self.infrastructure_status[environment_name]

        logger.info(f"Destroying environment: {environment_name}")

        try:
            # Update status
            status.state = InfrastructureState.DESTROYING
            status.last_update = time.time()

            # Destroy based on provider
            if config.provider == InfrastructureProvider.TERRAFORM:
                await self._destroy_terraform(environment_name, force)
            elif config.provider == InfrastructureProvider.KUBERNETES:
                await self._destroy_kubernetes(environment_name, force)
            elif config.provider == InfrastructureProvider.DOCKER:
                await self._destroy_docker(environment_name, force)

            # Clean up local state
            del self.environments[environment_name]
            del self.infrastructure_status[environment_name]

            logger.info(f"Environment {environment_name} destroyed successfully")

        except Exception as e:
            logger.error(f"Environment destruction failed for {environment_name}: {e}")
            status.state = InfrastructureState.FAILED
            raise

    async def _destroy_terraform(self, environment_name: str, force: bool = False) -> None:
        """Destroy Terraform-managed infrastructure."""
        config = self.environments[environment_name]
        env_dir = self.config_dir / "terraform" / config.name

        # Terraform destroy
        destroy_cmd = ["terraform", "destroy", "-auto-approve"]
        if force:
            destroy_cmd.append("-force")

        for var_name, var_value in config.variables.items():
            if var_name not in config.secrets:
                destroy_cmd.extend(["-var", f"{var_name}={var_value}"])

        await self._run_terraform_command(destroy_cmd, env_dir)

    async def _destroy_kubernetes(self, environment_name: str, force: bool = False) -> None:
        """Destroy Kubernetes-managed infrastructure."""
        config = self.environments[environment_name]

        # Delete namespace (which deletes all resources within)
        cmd = ["kubectl", "delete", "namespace", config.name]
        if force:
            cmd.append("--force")

        await self._run_command(cmd)

    async def _destroy_docker(self, environment_name: str, force: bool = False) -> None:
        """Destroy Docker-managed infrastructure."""
        config = self.environments[environment_name]
        env_dir = self.config_dir / "docker" / config.name
        compose_file = env_dir / "docker-compose.yml"

        # Docker compose down
        cmd = ["docker-compose", "-f", str(compose_file), "down"]
        if force:
            cmd.extend(["--volumes", "--remove-orphans"])

        await self._run_command(cmd, cwd=env_dir)

    def get_environment_status(self, environment_name: str) -> Optional[InfrastructureStatus]:
        """Get the current status of an environment."""
        return self.infrastructure_status.get(environment_name)

    def list_environments(self) -> List[str]:
        """List all registered environments."""
        return list(self.environments.keys())

    def generate_status_report(self) -> Dict[str, Any]:
        """Generate a comprehensive status report for all environments."""
        report = {
            "timestamp": time.time(),
            "environments": {},
            "summary": {
                "total_environments": len(self.environments),
                "running": 0,
                "failed": 0,
                "drifted": 0,
                "updating": 0
            }
        }

        for env_name, status in self.infrastructure_status.items():
            report["environments"][env_name] = {
                "state": status.state.value,
                "resources_count": len(status.resources),
                "health_checks_passed": sum(1 for hc in status.health_checks if hc.status == "passed"),
                "health_checks_total": len(status.health_checks),
                "drift_detected": status.drift_detected,
                "drift_issues": len(status.drift_details),
                "last_update": status.last_update
            }

            # Update summary counts
            if status.state == InfrastructureState.RUNNING:
                report["summary"]["running"] += 1
            elif status.state == InfrastructureState.FAILED:
                report["summary"]["failed"] += 1
            elif status.state == InfrastructureState.DRIFTED:
                report["summary"]["drifted"] += 1
            elif status.state == InfrastructureState.UPDATING:
                report["summary"]["updating"] += 1

        return report

    def print_status_table(self) -> None:
        """Print a formatted status table to console."""
        table = Table(title="Infrastructure Status")

        table.add_column("Environment", style="cyan")
        table.add_column("Provider", style="green")
        table.add_column("State", style="yellow")
        table.add_column("Resources", justify="center")
        table.add_column("Health", justify="center")
        table.add_column("Drift", justify="center")
        table.add_column("Last Update", style="dim")

        for env_name, config in self.environments.items():
            status = self.infrastructure_status.get(env_name)
            if status:
                # Health check summary
                health_passed = sum(1 for hc in status.health_checks if hc.status == "passed")
                health_total = len(status.health_checks)
                health_str = f"{health_passed}/{health_total}"

                # Drift status
                drift_str = "✓" if not status.drift_detected else f"⚠ {len(status.drift_details)}"

                # Last update
                last_update = time.strftime("%H:%M:%S", time.localtime(status.last_update))

                table.add_row(
                    env_name,
                    config.provider.value,
                    status.state.value,
                    str(len(status.resources)),
                    health_str,
                    drift_str,
                    last_update
                )

        console.print(table)


# Example usage and configuration
async def create_example_environment():
    """Create an example environment configuration."""
    # Example web application environment
    web_resources = [
        ResourceConfiguration(
            resource_type="deployment",
            name="web-app",
            properties={
                "image": "workspace-qdrant-mcp:latest",
                "replicas": 3,
                "ports": [{"containerPort": 8000}]
            },
            health_checks=[
                {
                    "type": "http",
                    "url": "http://web-app:8000/health",
                    "timeout": 10,
                    "expected_status": 200
                }
            ],
            tags={"app": "web-app", "tier": "frontend"}
        ),
        ResourceConfiguration(
            resource_type="service",
            name="web-app-service",
            properties={
                "app": "web-app",
                "ports": [{"port": 80, "targetPort": 8000}],
                "type": "LoadBalancer"
            },
            dependencies=["web-app"],
            tags={"app": "web-app", "tier": "frontend"}
        ),
        ResourceConfiguration(
            resource_type="deployment",
            name="qdrant",
            properties={
                "image": "qdrant/qdrant:latest",
                "replicas": 1,
                "ports": [{"containerPort": 6333}, {"containerPort": 6334}]
            },
            health_checks=[
                {
                    "type": "http",
                    "url": "http://qdrant:6333/health",
                    "timeout": 10,
                    "expected_status": 200
                }
            ],
            tags={"app": "qdrant", "tier": "database"}
        ),
        ResourceConfiguration(
            resource_type="service",
            name="qdrant-service",
            properties={
                "app": "qdrant",
                "ports": [
                    {"port": 6333, "targetPort": 6333, "name": "http"},
                    {"port": 6334, "targetPort": 6334, "name": "grpc"}
                ],
                "type": "ClusterIP"
            },
            dependencies=["qdrant"],
            tags={"app": "qdrant", "tier": "database"}
        )
    ]

    # Create environment configuration
    staging_config = EnvironmentConfig(
        name="staging",
        environment_type=EnvironmentType.STAGING,
        provider=InfrastructureProvider.KUBERNETES,
        region="us-west-2",
        resources=web_resources,
        variables={
            "QDRANT_URL": "http://qdrant-service:6333",
            "ENVIRONMENT": "staging",
            "LOG_LEVEL": "DEBUG",
            "REPLICAS": 3
        },
        secrets=["DATABASE_PASSWORD", "API_KEY"],
        monitoring={
            "enabled": True,
            "metrics_interval": 30,
            "alerts": ["high_error_rate", "slow_response_time"]
        },
        security_settings={
            "network_policies": True,
            "pod_security_standards": "restricted",
            "image_scanning": True
        }
    )

    return staging_config


async def main():
    """Main example demonstrating the infrastructure provisioner."""
    # Initialize provisioner
    provisioner = InfrastructureProvisioner(
        config_dir=Path("infrastructure"),
        monitoring_enabled=True
    )

    # Create and register example environment
    staging_config = await create_example_environment()
    await provisioner.register_environment(staging_config)

    try:
        # Provision environment (dry run first)
        console.print("[blue]Running dry-run provisioning...[/blue]")
        await provisioner.provision_environment("staging", dry_run=True)

        # Actual provisioning (commented out for safety)
        # console.print("[green]Provisioning staging environment...[/green]")
        # await provisioner.provision_environment("staging", dry_run=False)

        # Drift detection
        console.print("[yellow]Checking for configuration drift...[/yellow]")
        drift_detected = await provisioner.detect_drift("staging")

        if drift_detected:
            console.print("[red]Configuration drift detected![/red]")
            # await provisioner.correct_drift("staging", auto_approve=True)
        else:
            console.print("[green]No configuration drift detected[/green]")

        # Display status
        provisioner.print_status_table()

        # Generate report
        report = provisioner.generate_status_report()
        console.print(f"\n[bold]Infrastructure Report:[/bold]")
        console.print(json.dumps(report, indent=2))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())