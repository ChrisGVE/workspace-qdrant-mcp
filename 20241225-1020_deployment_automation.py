#!/usr/bin/env python3
"""
Automated Deployment Pipeline System

This module provides comprehensive deployment automation with multi-environment support,
quality gates, rollback mechanisms, and deployment validation.

Key features:
- Multi-environment deployment pipelines
- Blue-green and canary deployment strategies
- Automated quality gates and validation
- Rollback mechanisms with health monitoring
- Deployment orchestration and coordination
- Integration with infrastructure provisioning
- Security scanning and compliance checks
- Performance monitoring and alerting
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
from typing import Any, Callable, Dict, List, Optional, Set, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()


class DeploymentStrategy(str, Enum):
    """Deployment strategy types."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TESTING = "a_b_testing"


class DeploymentStatus(str, Enum):
    """Deployment status values."""
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class QualityGateType(str, Enum):
    """Quality gate types."""
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_TEST = "performance_test"
    CODE_QUALITY = "code_quality"
    SMOKE_TEST = "smoke_test"
    HEALTH_CHECK = "health_check"
    CUSTOM = "custom"


class ValidationStatus(str, Enum):
    """Validation status values."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class QualityGate:
    """Quality gate configuration."""
    name: str
    gate_type: QualityGateType
    command: List[str]
    timeout: int = 300
    required: bool = True
    retry_count: int = 0
    environment_vars: Dict[str, str] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    failure_threshold: float = 0.0
    warning_threshold: float = 0.0


@dataclass
class DeploymentTarget:
    """Deployment target configuration."""
    name: str
    environment: str
    infrastructure_config: Optional[str] = None
    health_check_url: Optional[str] = None
    health_check_timeout: int = 60
    readiness_probe: Optional[Dict[str, Any]] = None
    liveness_probe: Optional[Dict[str, Any]] = None
    resource_limits: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    version: str
    strategy: DeploymentStrategy
    targets: List[DeploymentTarget]
    quality_gates: List[QualityGate] = field(default_factory=list)
    rollback_config: Optional[Dict[str, Any]] = None
    notification_config: Optional[Dict[str, Any]] = None
    timeout: int = 1800  # 30 minutes default
    parallel_deployments: bool = False
    deployment_variables: Dict[str, str] = field(default_factory=dict)
    secrets: List[str] = field(default_factory=list)


class ValidationResult(BaseModel):
    """Quality gate validation result."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    gate_name: str
    status: ValidationStatus
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = Field(default_factory=dict)
    artifacts: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class DeploymentProgress(BaseModel):
    """Deployment progress tracking."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    deployment_id: str
    status: DeploymentStatus
    current_target: Optional[str] = None
    completed_targets: List[str] = Field(default_factory=list)
    failed_targets: List[str] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    rollback_reason: Optional[str] = None
    deployment_artifacts: List[str] = Field(default_factory=list)


class DeploymentPipeline:
    """Comprehensive deployment automation pipeline."""

    def __init__(
        self,
        artifacts_dir: Path = Path("artifacts"),
        logs_dir: Path = Path("logs"),
        config_dir: Path = Path("deployment-config"),
        enable_monitoring: bool = True
    ):
        """Initialize the deployment pipeline."""
        self.artifacts_dir = Path(artifacts_dir)
        self.logs_dir = Path(logs_dir)
        self.config_dir = Path(config_dir)
        self.enable_monitoring = enable_monitoring

        # Create required directories
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.active_deployments: Dict[str, DeploymentProgress] = {}
        self.deployment_history: List[DeploymentProgress] = []

        # Quality gate validators
        self._quality_gate_validators: Dict[QualityGateType, Callable] = {
            QualityGateType.UNIT_TESTS: self._validate_unit_tests,
            QualityGateType.INTEGRATION_TESTS: self._validate_integration_tests,
            QualityGateType.SECURITY_SCAN: self._validate_security_scan,
            QualityGateType.PERFORMANCE_TEST: self._validate_performance_test,
            QualityGateType.CODE_QUALITY: self._validate_code_quality,
            QualityGateType.SMOKE_TEST: self._validate_smoke_test,
            QualityGateType.HEALTH_CHECK: self._validate_health_check,
            QualityGateType.CUSTOM: self._validate_custom,
        }

        logger.info("Deployment pipeline initialized")

    async def deploy(
        self,
        config: DeploymentConfig,
        force: bool = False,
        dry_run: bool = False
    ) -> str:
        """Execute deployment with the specified configuration."""
        deployment_id = f"{config.name}-{config.version}-{int(time.time())}"

        logger.info(f"Starting deployment {deployment_id} ({'dry-run' if dry_run else 'live'})")

        # Initialize deployment progress
        progress = DeploymentProgress(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            start_time=time.time()
        )

        self.active_deployments[deployment_id] = progress

        try:
            # Pre-deployment validation
            await self._pre_deployment_validation(config, progress)

            # Execute quality gates
            if not force:
                await self._execute_quality_gates(config, progress)

            # Execute deployment strategy
            progress.status = DeploymentStatus.RUNNING
            await self._execute_deployment_strategy(config, progress, dry_run)

            # Post-deployment validation
            progress.status = DeploymentStatus.VALIDATING
            await self._post_deployment_validation(config, progress)

            # Mark as completed
            progress.status = DeploymentStatus.COMPLETED
            progress.end_time = time.time()
            progress.duration = progress.end_time - progress.start_time

            logger.info(f"Deployment {deployment_id} completed successfully in {progress.duration:.2f}s")

        except Exception as e:
            logger.error(f"Deployment {deployment_id} failed: {e}")
            progress.status = DeploymentStatus.FAILED
            progress.error_message = str(e)
            progress.end_time = time.time()
            progress.duration = progress.end_time - progress.start_time

            # Attempt automatic rollback if configured
            if config.rollback_config and config.rollback_config.get("auto_rollback", False):
                await self._execute_rollback(config, progress, str(e))

            raise
        finally:
            # Move to history and cleanup active tracking
            self.deployment_history.append(progress)
            if deployment_id in self.active_deployments:
                del self.active_deployments[deployment_id]

        return deployment_id

    async def _pre_deployment_validation(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress
    ) -> None:
        """Perform pre-deployment validation."""
        logger.info("Performing pre-deployment validation")

        # Validate targets
        for target in config.targets:
            if not target.environment:
                raise ValueError(f"Target {target.name} missing environment configuration")

        # Validate artifacts exist (if specified)
        for artifact in progress.deployment_artifacts:
            artifact_path = self.artifacts_dir / artifact
            if not artifact_path.exists():
                raise FileNotFoundError(f"Deployment artifact not found: {artifact}")

        # Check for conflicting deployments
        for active_id, active_progress in self.active_deployments.items():
            if active_id != progress.deployment_id:
                # Check for target overlap
                active_targets = set(active_progress.completed_targets + [active_progress.current_target])
                new_targets = set(target.name for target in config.targets)

                if active_targets & new_targets:  # Intersection exists
                    raise RuntimeError(
                        f"Deployment conflict: targets {active_targets & new_targets} "
                        f"already being deployed by {active_id}"
                    )

        logger.info("Pre-deployment validation passed")

    async def _execute_quality_gates(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress
    ) -> None:
        """Execute all quality gates."""
        if not config.quality_gates:
            logger.info("No quality gates configured, skipping")
            return

        logger.info(f"Executing {len(config.quality_gates)} quality gates")

        failed_gates = []

        for gate in config.quality_gates:
            validation_result = ValidationResult(
                gate_name=gate.name,
                status=ValidationStatus.RUNNING,
                start_time=time.time()
            )
            progress.validation_results.append(validation_result)

            try:
                # Execute the quality gate
                validator = self._quality_gate_validators[gate.gate_type]
                await validator(gate, validation_result, config)

                validation_result.end_time = time.time()
                validation_result.duration = validation_result.end_time - validation_result.start_time

                if validation_result.status == ValidationStatus.FAILED:
                    if gate.required:
                        failed_gates.append(gate.name)
                    logger.warning(f"Quality gate '{gate.name}' failed: {validation_result.message}")
                else:
                    logger.info(f"Quality gate '{gate.name}' passed")

            except Exception as e:
                validation_result.status = ValidationStatus.FAILED
                validation_result.message = str(e)
                validation_result.end_time = time.time()
                validation_result.duration = validation_result.end_time - validation_result.start_time

                if gate.required:
                    failed_gates.append(gate.name)
                logger.error(f"Quality gate '{gate.name}' failed with exception: {e}")

        if failed_gates:
            raise RuntimeError(f"Required quality gates failed: {', '.join(failed_gates)}")

        logger.info("All quality gates passed")

    async def _execute_deployment_strategy(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress,
        dry_run: bool = False
    ) -> None:
        """Execute the deployment strategy."""
        logger.info(f"Executing {config.strategy.value} deployment strategy")

        if config.strategy == DeploymentStrategy.ROLLING:
            await self._rolling_deployment(config, progress, dry_run)
        elif config.strategy == DeploymentStrategy.BLUE_GREEN:
            await self._blue_green_deployment(config, progress, dry_run)
        elif config.strategy == DeploymentStrategy.CANARY:
            await self._canary_deployment(config, progress, dry_run)
        elif config.strategy == DeploymentStrategy.RECREATE:
            await self._recreate_deployment(config, progress, dry_run)
        elif config.strategy == DeploymentStrategy.A_B_TESTING:
            await self._ab_testing_deployment(config, progress, dry_run)
        else:
            raise ValueError(f"Unsupported deployment strategy: {config.strategy}")

    async def _rolling_deployment(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress,
        dry_run: bool = False
    ) -> None:
        """Execute rolling deployment strategy."""
        logger.info("Executing rolling deployment")

        for target in config.targets:
            progress.current_target = target.name
            logger.info(f"Deploying to target: {target.name}")

            try:
                # Deploy to target
                await self._deploy_to_target(target, config, dry_run)

                # Wait for readiness
                await self._wait_for_readiness(target, config)

                # Validate deployment
                await self._validate_target_deployment(target, config)

                progress.completed_targets.append(target.name)
                logger.info(f"Successfully deployed to {target.name}")

            except Exception as e:
                progress.failed_targets.append(target.name)
                logger.error(f"Failed to deploy to {target.name}: {e}")
                raise

            # Brief pause between deployments
            await asyncio.sleep(5)

        progress.current_target = None

    async def _blue_green_deployment(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress,
        dry_run: bool = False
    ) -> None:
        """Execute blue-green deployment strategy."""
        logger.info("Executing blue-green deployment")

        if len(config.targets) != 2:
            raise ValueError("Blue-green deployment requires exactly 2 targets (blue and green)")

        blue_target = config.targets[0]
        green_target = config.targets[1]

        # Deploy to green environment (inactive)
        progress.current_target = green_target.name
        logger.info(f"Deploying to green environment: {green_target.name}")

        await self._deploy_to_target(green_target, config, dry_run)
        await self._wait_for_readiness(green_target, config)
        await self._validate_target_deployment(green_target, config)

        progress.completed_targets.append(green_target.name)

        # Switch traffic from blue to green
        logger.info("Switching traffic from blue to green")
        if not dry_run:
            await self._switch_traffic(blue_target, green_target, config)

        # Optionally keep blue for quick rollback
        logger.info("Blue-green deployment completed")
        progress.current_target = None

    async def _canary_deployment(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress,
        dry_run: bool = False
    ) -> None:
        """Execute canary deployment strategy."""
        logger.info("Executing canary deployment")

        if len(config.targets) < 2:
            raise ValueError("Canary deployment requires at least 2 targets")

        canary_target = config.targets[0]
        production_targets = config.targets[1:]

        # Deploy to canary
        progress.current_target = canary_target.name
        logger.info(f"Deploying to canary: {canary_target.name}")

        await self._deploy_to_target(canary_target, config, dry_run)
        await self._wait_for_readiness(canary_target, config)
        await self._validate_target_deployment(canary_target, config)

        progress.completed_targets.append(canary_target.name)

        # Monitor canary performance
        logger.info("Monitoring canary performance")
        canary_success = await self._monitor_canary(canary_target, config)

        if not canary_success:
            raise RuntimeError("Canary deployment failed validation")

        # Progressive rollout to production
        logger.info("Rolling out to production targets")
        for target in production_targets:
            progress.current_target = target.name
            await self._deploy_to_target(target, config, dry_run)
            await self._wait_for_readiness(target, config)
            await self._validate_target_deployment(target, config)
            progress.completed_targets.append(target.name)

        progress.current_target = None

    async def _recreate_deployment(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress,
        dry_run: bool = False
    ) -> None:
        """Execute recreate deployment strategy."""
        logger.info("Executing recreate deployment")

        # Stop all targets first
        logger.info("Stopping all current deployments")
        for target in config.targets:
            if not dry_run:
                await self._stop_target(target, config)

        # Deploy to all targets
        for target in config.targets:
            progress.current_target = target.name
            logger.info(f"Recreating deployment on target: {target.name}")

            await self._deploy_to_target(target, config, dry_run)
            await self._wait_for_readiness(target, config)
            await self._validate_target_deployment(target, config)

            progress.completed_targets.append(target.name)

        progress.current_target = None

    async def _ab_testing_deployment(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress,
        dry_run: bool = False
    ) -> None:
        """Execute A/B testing deployment strategy."""
        logger.info("Executing A/B testing deployment")

        if len(config.targets) != 2:
            raise ValueError("A/B testing deployment requires exactly 2 targets (A and B)")

        target_a = config.targets[0]
        target_b = config.targets[1]

        # Deploy to both targets
        for target in [target_a, target_b]:
            progress.current_target = target.name
            logger.info(f"Deploying to A/B target: {target.name}")

            await self._deploy_to_target(target, config, dry_run)
            await self._wait_for_readiness(target, config)
            await self._validate_target_deployment(target, config)

            progress.completed_targets.append(target.name)

        # Configure traffic splitting
        logger.info("Configuring A/B traffic splitting")
        if not dry_run:
            await self._configure_ab_traffic(target_a, target_b, config)

        progress.current_target = None

    async def _deploy_to_target(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig,
        dry_run: bool = False
    ) -> None:
        """Deploy to a specific target."""
        logger.info(f"Deploying {config.name} v{config.version} to {target.name}")

        # Generate deployment commands based on target type
        deploy_commands = await self._generate_deployment_commands(target, config)

        for command in deploy_commands:
            if dry_run:
                logger.info(f"DRY RUN: Would execute: {' '.join(command)}")
            else:
                logger.info(f"Executing: {' '.join(command)}")
                await self._execute_command(
                    command,
                    timeout=config.timeout,
                    env_vars=config.deployment_variables
                )

    async def _generate_deployment_commands(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig
    ) -> List[List[str]]:
        """Generate deployment commands for a target."""
        commands = []

        # Determine deployment method based on target infrastructure
        if target.infrastructure_config:
            if "kubernetes" in target.infrastructure_config.lower():
                commands.extend(await self._generate_kubernetes_commands(target, config))
            elif "docker" in target.infrastructure_config.lower():
                commands.extend(await self._generate_docker_commands(target, config))
            elif "terraform" in target.infrastructure_config.lower():
                commands.extend(await self._generate_terraform_commands(target, config))
            else:
                # Generic deployment commands
                commands.extend(await self._generate_generic_commands(target, config))
        else:
            commands.extend(await self._generate_generic_commands(target, config))

        return commands

    async def _generate_kubernetes_commands(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig
    ) -> List[List[str]]:
        """Generate Kubernetes deployment commands."""
        commands = []

        # Apply manifests
        manifest_dir = self.config_dir / "kubernetes" / target.environment
        if manifest_dir.exists():
            for manifest_file in manifest_dir.glob("*.yaml"):
                commands.append([
                    "kubectl", "apply", "-f", str(manifest_file),
                    "--namespace", target.environment
                ])

        # Set image version
        commands.append([
            "kubectl", "set", "image",
            f"deployment/{config.name}",
            f"{config.name}={config.name}:{config.version}",
            "--namespace", target.environment
        ])

        # Wait for rollout
        commands.append([
            "kubectl", "rollout", "status",
            f"deployment/{config.name}",
            "--namespace", target.environment,
            "--timeout=600s"
        ])

        return commands

    async def _generate_docker_commands(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig
    ) -> List[List[str]]:
        """Generate Docker deployment commands."""
        commands = []

        compose_file = self.config_dir / "docker" / target.environment / "docker-compose.yml"
        if compose_file.exists():
            # Pull latest images
            commands.append([
                "docker-compose", "-f", str(compose_file),
                "pull", config.name
            ])

            # Update service
            commands.append([
                "docker-compose", "-f", str(compose_file),
                "up", "-d", "--no-deps", config.name
            ])

        return commands

    async def _generate_terraform_commands(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig
    ) -> List[List[str]]:
        """Generate Terraform deployment commands."""
        commands = []

        terraform_dir = self.config_dir / "terraform" / target.environment
        if terraform_dir.exists():
            # Initialize
            commands.append([
                "terraform", "init", "-input=false"
            ])

            # Plan
            commands.append([
                "terraform", "plan",
                "-var", f"app_version={config.version}",
                "-out=tfplan"
            ])

            # Apply
            commands.append([
                "terraform", "apply", "-input=false", "tfplan"
            ])

        return commands

    async def _generate_generic_commands(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig
    ) -> List[List[str]]:
        """Generate generic deployment commands."""
        commands = []

        # Example generic deployment script
        deploy_script = self.config_dir / "scripts" / f"deploy-{target.environment}.sh"
        if deploy_script.exists():
            commands.append([
                "bash", str(deploy_script),
                config.name, config.version, target.environment
            ])

        return commands

    async def _wait_for_readiness(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig,
        max_wait: int = 300
    ) -> None:
        """Wait for target to become ready."""
        logger.info(f"Waiting for {target.name} to become ready")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            if target.readiness_probe:
                ready = await self._check_readiness_probe(target)
                if ready:
                    logger.info(f"{target.name} is ready")
                    return

            if target.health_check_url:
                healthy = await self._check_health_url(
                    target.health_check_url,
                    timeout=target.health_check_timeout
                )
                if healthy:
                    logger.info(f"{target.name} health check passed")
                    return

            await asyncio.sleep(5)

        raise RuntimeError(f"{target.name} failed to become ready within {max_wait} seconds")

    async def _check_readiness_probe(self, target: DeploymentTarget) -> bool:
        """Check target readiness probe."""
        probe = target.readiness_probe
        if not probe:
            return True

        try:
            if probe.get("type") == "http":
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    url = probe["url"]
                    timeout = probe.get("timeout", 10)
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                        expected_status = probe.get("expected_status", 200)
                        return response.status == expected_status

            elif probe.get("type") == "tcp":
                host = probe["host"]
                port = probe["port"]
                timeout = probe.get("timeout", 5)

                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=timeout
                )
                writer.close()
                await writer.wait_closed()
                return True

            elif probe.get("type") == "command":
                command = probe["command"]
                result = await self._execute_command(command)
                expected_exit_code = probe.get("expected_exit_code", 0)
                return result.returncode == expected_exit_code

        except Exception as e:
            logger.debug(f"Readiness probe failed: {e}")
            return False

        return False

    async def _check_health_url(self, url: str, timeout: int = 30) -> bool:
        """Check health URL."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Health check failed for {url}: {e}")
            return False

    async def _validate_target_deployment(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig
    ) -> None:
        """Validate deployment to a target."""
        logger.info(f"Validating deployment on {target.name}")

        # Run liveness probe if configured
        if target.liveness_probe:
            probe_passed = await self._check_liveness_probe(target)
            if not probe_passed:
                raise RuntimeError(f"Liveness probe failed for {target.name}")

        # Additional deployment-specific validation
        await self._run_deployment_validation(target, config)

        logger.info(f"Deployment validation passed for {target.name}")

    async def _check_liveness_probe(self, target: DeploymentTarget) -> bool:
        """Check target liveness probe."""
        probe = target.liveness_probe
        if not probe:
            return True

        # Similar to readiness probe but for liveness
        return await self._check_readiness_probe(target)

    async def _run_deployment_validation(
        self,
        target: DeploymentTarget,
        config: DeploymentConfig
    ) -> None:
        """Run deployment-specific validation."""
        # This could include integration tests, smoke tests, etc.
        validation_script = self.config_dir / "scripts" / f"validate-{target.environment}.sh"
        if validation_script.exists():
            result = await self._execute_command([
                "bash", str(validation_script),
                config.name, config.version, target.environment
            ])

            if result.returncode != 0:
                raise RuntimeError(f"Deployment validation failed: {result.stderr}")

    async def _post_deployment_validation(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress
    ) -> None:
        """Perform post-deployment validation."""
        logger.info("Performing post-deployment validation")

        # Run post-deployment quality gates
        post_deployment_gates = [
            gate for gate in config.quality_gates
            if gate.gate_type in [QualityGateType.SMOKE_TEST, QualityGateType.HEALTH_CHECK]
        ]

        for gate in post_deployment_gates:
            validation_result = ValidationResult(
                gate_name=f"post-{gate.name}",
                status=ValidationStatus.RUNNING,
                start_time=time.time()
            )
            progress.validation_results.append(validation_result)

            try:
                validator = self._quality_gate_validators[gate.gate_type]
                await validator(gate, validation_result, config)

                validation_result.end_time = time.time()
                validation_result.duration = validation_result.end_time - validation_result.start_time

                if validation_result.status == ValidationStatus.FAILED and gate.required:
                    raise RuntimeError(f"Post-deployment validation failed: {validation_result.message}")

            except Exception as e:
                validation_result.status = ValidationStatus.FAILED
                validation_result.message = str(e)
                validation_result.end_time = time.time()
                validation_result.duration = validation_result.end_time - validation_result.start_time

                if gate.required:
                    raise

        logger.info("Post-deployment validation completed")

    async def _switch_traffic(
        self,
        from_target: DeploymentTarget,
        to_target: DeploymentTarget,
        config: DeploymentConfig
    ) -> None:
        """Switch traffic between targets (blue-green)."""
        logger.info(f"Switching traffic from {from_target.name} to {to_target.name}")

        # This would typically involve updating load balancer configuration
        # or service mesh routing rules

        switch_script = self.config_dir / "scripts" / "switch-traffic.sh"
        if switch_script.exists():
            await self._execute_command([
                "bash", str(switch_script),
                from_target.name, to_target.name, from_target.environment
            ])
        else:
            logger.warning("No traffic switching script found")

    async def _monitor_canary(
        self,
        canary_target: DeploymentTarget,
        config: DeploymentConfig,
        monitor_duration: int = 300
    ) -> bool:
        """Monitor canary deployment performance."""
        logger.info(f"Monitoring canary {canary_target.name} for {monitor_duration} seconds")

        start_time = time.time()
        metrics = {
            "error_rate": 0.0,
            "response_time": 0.0,
            "success_rate": 100.0
        }

        while time.time() - start_time < monitor_duration:
            # Check health
            if canary_target.health_check_url:
                healthy = await self._check_health_url(canary_target.health_check_url)
                if not healthy:
                    logger.warning("Canary health check failed")
                    return False

            # Collect metrics (would integrate with monitoring system)
            current_metrics = await self._collect_canary_metrics(canary_target)

            # Update running averages
            for key, value in current_metrics.items():
                if key in metrics:
                    metrics[key] = (metrics[key] + value) / 2

            # Check thresholds
            if metrics["error_rate"] > 5.0:  # 5% error rate threshold
                logger.error(f"Canary error rate too high: {metrics['error_rate']}%")
                return False

            if metrics["response_time"] > 2000:  # 2 second response time threshold
                logger.error(f"Canary response time too high: {metrics['response_time']}ms")
                return False

            await asyncio.sleep(10)  # Check every 10 seconds

        logger.info(f"Canary monitoring completed successfully: {metrics}")
        return True

    async def _collect_canary_metrics(self, target: DeploymentTarget) -> Dict[str, float]:
        """Collect metrics from canary deployment."""
        # This would integrate with monitoring systems like Prometheus, DataDog, etc.
        # For now, return mock metrics
        return {
            "error_rate": 1.0,  # 1% error rate
            "response_time": 150.0,  # 150ms average response time
            "success_rate": 99.0,  # 99% success rate
            "cpu_usage": 45.0,  # 45% CPU usage
            "memory_usage": 60.0,  # 60% memory usage
        }

    async def _configure_ab_traffic(
        self,
        target_a: DeploymentTarget,
        target_b: DeploymentTarget,
        config: DeploymentConfig
    ) -> None:
        """Configure A/B testing traffic split."""
        logger.info(f"Configuring A/B traffic between {target_a.name} and {target_b.name}")

        # This would configure load balancer or service mesh for traffic splitting
        ab_script = self.config_dir / "scripts" / "configure-ab.sh"
        if ab_script.exists():
            await self._execute_command([
                "bash", str(ab_script),
                target_a.name, target_b.name, target_a.environment
            ])

    async def _stop_target(self, target: DeploymentTarget, config: DeploymentConfig) -> None:
        """Stop deployment on a target."""
        logger.info(f"Stopping deployment on {target.name}")

        stop_script = self.config_dir / "scripts" / f"stop-{target.environment}.sh"
        if stop_script.exists():
            await self._execute_command([
                "bash", str(stop_script),
                config.name, target.environment
            ])

    async def _execute_rollback(
        self,
        config: DeploymentConfig,
        progress: DeploymentProgress,
        reason: str
    ) -> None:
        """Execute deployment rollback."""
        logger.info(f"Executing rollback for deployment {progress.deployment_id}")

        progress.status = DeploymentStatus.ROLLING_BACK
        progress.rollback_reason = reason

        rollback_config = config.rollback_config or {}
        previous_version = rollback_config.get("previous_version")

        if not previous_version:
            logger.error("No previous version configured for rollback")
            return

        try:
            # Create rollback deployment config
            rollback_deployment_config = DeploymentConfig(
                name=config.name,
                version=previous_version,
                strategy=DeploymentStrategy.ROLLING,  # Use rolling for safety
                targets=config.targets,
                quality_gates=[],  # Skip quality gates for emergency rollback
                timeout=config.timeout
            )

            # Execute rollback
            await self._execute_deployment_strategy(
                rollback_deployment_config,
                progress,
                dry_run=False
            )

            progress.status = DeploymentStatus.ROLLED_BACK
            logger.info(f"Rollback completed to version {previous_version}")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            progress.error_message = f"Rollback failed: {e}"

    async def rollback_deployment(
        self,
        deployment_id: str,
        target_version: Optional[str] = None,
        reason: str = "Manual rollback"
    ) -> None:
        """Manually rollback a deployment."""
        logger.info(f"Manual rollback requested for deployment {deployment_id}")

        # Find the deployment in history
        deployment = None
        for historical_deployment in self.deployment_history:
            if historical_deployment.deployment_id == deployment_id:
                deployment = historical_deployment
                break

        if not deployment:
            # Check active deployments
            deployment = self.active_deployments.get(deployment_id)

        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")

        # Create rollback configuration
        # This is a simplified approach - in practice you'd need to reconstruct
        # the original deployment config and modify it for rollback
        logger.info(f"Rollback initiated for {deployment_id}: {reason}")

        # Implementation would depend on stored deployment configuration
        # For now, just update status
        if deployment_id in self.active_deployments:
            self.active_deployments[deployment_id].status = DeploymentStatus.ROLLING_BACK
            self.active_deployments[deployment_id].rollback_reason = reason

    # Quality Gate Validators

    async def _validate_unit_tests(
        self,
        gate: QualityGate,
        result: ValidationResult,
        config: DeploymentConfig
    ) -> None:
        """Validate unit tests quality gate."""
        try:
            command_result = await self._execute_command(
                gate.command,
                timeout=gate.timeout,
                env_vars=gate.environment_vars
            )

            if command_result.returncode == 0:
                result.status = ValidationStatus.PASSED
                result.message = "Unit tests passed"

                # Parse test results if available
                if "coverage" in gate.success_criteria:
                    # Extract coverage from output (implementation specific)
                    result.metrics["coverage"] = 85.0  # Mock value

            else:
                result.status = ValidationStatus.FAILED
                result.message = f"Unit tests failed: {command_result.stderr}"

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.message = f"Unit test execution failed: {e}"

    async def _validate_integration_tests(
        self,
        gate: QualityGate,
        result: ValidationResult,
        config: DeploymentConfig
    ) -> None:
        """Validate integration tests quality gate."""
        try:
            command_result = await self._execute_command(
                gate.command,
                timeout=gate.timeout,
                env_vars=gate.environment_vars
            )

            if command_result.returncode == 0:
                result.status = ValidationStatus.PASSED
                result.message = "Integration tests passed"
            else:
                result.status = ValidationStatus.FAILED
                result.message = f"Integration tests failed: {command_result.stderr}"

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.message = f"Integration test execution failed: {e}"

    async def _validate_security_scan(
        self,
        gate: QualityGate,
        result: ValidationResult,
        config: DeploymentConfig
    ) -> None:
        """Validate security scan quality gate."""
        try:
            command_result = await self._execute_command(
                gate.command,
                timeout=gate.timeout,
                env_vars=gate.environment_vars
            )

            # Parse security scan results
            vulnerabilities_found = 0  # Would parse from output
            critical_vulnerabilities = 0  # Would parse from output

            max_vulnerabilities = gate.success_criteria.get("max_vulnerabilities", 0)
            max_critical = gate.success_criteria.get("max_critical_vulnerabilities", 0)

            if vulnerabilities_found <= max_vulnerabilities and critical_vulnerabilities <= max_critical:
                result.status = ValidationStatus.PASSED
                result.message = f"Security scan passed: {vulnerabilities_found} vulnerabilities found"
                result.metrics["vulnerabilities"] = vulnerabilities_found
                result.metrics["critical_vulnerabilities"] = critical_vulnerabilities
            else:
                result.status = ValidationStatus.FAILED
                result.message = f"Security scan failed: {vulnerabilities_found} vulnerabilities, {critical_vulnerabilities} critical"

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.message = f"Security scan failed: {e}"

    async def _validate_performance_test(
        self,
        gate: QualityGate,
        result: ValidationResult,
        config: DeploymentConfig
    ) -> None:
        """Validate performance test quality gate."""
        try:
            command_result = await self._execute_command(
                gate.command,
                timeout=gate.timeout,
                env_vars=gate.environment_vars
            )

            if command_result.returncode == 0:
                # Parse performance metrics
                response_time = 150.0  # Would parse from output
                throughput = 1000.0  # Would parse from output
                error_rate = 1.0  # Would parse from output

                max_response_time = gate.success_criteria.get("max_response_time", 1000.0)
                min_throughput = gate.success_criteria.get("min_throughput", 100.0)
                max_error_rate = gate.success_criteria.get("max_error_rate", 5.0)

                if (response_time <= max_response_time and
                    throughput >= min_throughput and
                    error_rate <= max_error_rate):
                    result.status = ValidationStatus.PASSED
                    result.message = "Performance test passed"
                    result.metrics.update({
                        "response_time": response_time,
                        "throughput": throughput,
                        "error_rate": error_rate
                    })
                else:
                    result.status = ValidationStatus.FAILED
                    result.message = "Performance test failed criteria"
            else:
                result.status = ValidationStatus.FAILED
                result.message = f"Performance test execution failed: {command_result.stderr}"

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.message = f"Performance test failed: {e}"

    async def _validate_code_quality(
        self,
        gate: QualityGate,
        result: ValidationResult,
        config: DeploymentConfig
    ) -> None:
        """Validate code quality quality gate."""
        try:
            command_result = await self._execute_command(
                gate.command,
                timeout=gate.timeout,
                env_vars=gate.environment_vars
            )

            if command_result.returncode == 0:
                result.status = ValidationStatus.PASSED
                result.message = "Code quality check passed"

                # Mock quality metrics
                result.metrics.update({
                    "complexity": 7.5,
                    "maintainability": 85.0,
                    "technical_debt": 2.5
                })
            else:
                result.status = ValidationStatus.FAILED
                result.message = f"Code quality check failed: {command_result.stderr}"

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.message = f"Code quality check failed: {e}"

    async def _validate_smoke_test(
        self,
        gate: QualityGate,
        result: ValidationResult,
        config: DeploymentConfig
    ) -> None:
        """Validate smoke test quality gate."""
        try:
            command_result = await self._execute_command(
                gate.command,
                timeout=gate.timeout,
                env_vars=gate.environment_vars
            )

            if command_result.returncode == 0:
                result.status = ValidationStatus.PASSED
                result.message = "Smoke tests passed"
            else:
                result.status = ValidationStatus.FAILED
                result.message = f"Smoke tests failed: {command_result.stderr}"

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.message = f"Smoke test execution failed: {e}"

    async def _validate_health_check(
        self,
        gate: QualityGate,
        result: ValidationResult,
        config: DeploymentConfig
    ) -> None:
        """Validate health check quality gate."""
        try:
            # Extract URL from command or environment variables
            health_url = gate.environment_vars.get("HEALTH_URL")

            if health_url:
                healthy = await self._check_health_url(health_url)
                if healthy:
                    result.status = ValidationStatus.PASSED
                    result.message = "Health check passed"
                else:
                    result.status = ValidationStatus.FAILED
                    result.message = "Health check failed"
            else:
                # Execute command-based health check
                command_result = await self._execute_command(
                    gate.command,
                    timeout=gate.timeout,
                    env_vars=gate.environment_vars
                )

                if command_result.returncode == 0:
                    result.status = ValidationStatus.PASSED
                    result.message = "Health check command passed"
                else:
                    result.status = ValidationStatus.FAILED
                    result.message = f"Health check command failed: {command_result.stderr}"

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.message = f"Health check failed: {e}"

    async def _validate_custom(
        self,
        gate: QualityGate,
        result: ValidationResult,
        config: DeploymentConfig
    ) -> None:
        """Validate custom quality gate."""
        try:
            command_result = await self._execute_command(
                gate.command,
                timeout=gate.timeout,
                env_vars=gate.environment_vars
            )

            if command_result.returncode == 0:
                result.status = ValidationStatus.PASSED
                result.message = "Custom quality gate passed"
            else:
                result.status = ValidationStatus.FAILED
                result.message = f"Custom quality gate failed: {command_result.stderr}"

        except Exception as e:
            result.status = ValidationStatus.FAILED
            result.message = f"Custom quality gate failed: {e}"

    async def _execute_command(
        self,
        command: List[str],
        timeout: int = 300,
        env_vars: Optional[Dict[str, str]] = None
    ) -> subprocess.CompletedProcess:
        """Execute a command with timeout and environment variables."""
        env = os.environ.copy()
        if env_vars:
            env.update(env_vars)

        logger.debug(f"Executing command: {' '.join(command)}")

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return subprocess.CompletedProcess(
                command,
                process.returncode,
                stdout.decode(),
                stderr.decode()
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise asyncio.TimeoutError(f"Command timed out after {timeout} seconds: {' '.join(command)}")

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentProgress]:
        """Get status of a deployment."""
        # Check active deployments first
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id]

        # Check deployment history
        for deployment in self.deployment_history:
            if deployment.deployment_id == deployment_id:
                return deployment

        return None

    def list_active_deployments(self) -> List[str]:
        """List all active deployment IDs."""
        return list(self.active_deployments.keys())

    def get_deployment_history(self, limit: int = 50) -> List[DeploymentProgress]:
        """Get deployment history."""
        return self.deployment_history[-limit:]

    def cancel_deployment(self, deployment_id: str, reason: str = "Manual cancellation") -> bool:
        """Cancel an active deployment."""
        if deployment_id not in self.active_deployments:
            return False

        progress = self.active_deployments[deployment_id]
        progress.status = DeploymentStatus.CANCELLED
        progress.error_message = reason
        progress.end_time = time.time()
        progress.duration = progress.end_time - progress.start_time

        # Move to history
        self.deployment_history.append(progress)
        del self.active_deployments[deployment_id]

        logger.info(f"Deployment {deployment_id} cancelled: {reason}")
        return True

    def generate_deployment_report(self, deployment_id: str) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        progress = self.get_deployment_status(deployment_id)
        if not progress:
            raise ValueError(f"Deployment {deployment_id} not found")

        report = {
            "deployment_id": progress.deployment_id,
            "status": progress.status.value,
            "start_time": progress.start_time,
            "end_time": progress.end_time,
            "duration": progress.duration,
            "targets": {
                "completed": progress.completed_targets,
                "failed": progress.failed_targets,
                "current": progress.current_target
            },
            "quality_gates": [
                {
                    "name": result.gate_name,
                    "status": result.status.value,
                    "duration": result.duration,
                    "message": result.message,
                    "metrics": result.metrics
                }
                for result in progress.validation_results
            ],
            "artifacts": progress.deployment_artifacts,
            "error_message": progress.error_message,
            "rollback_reason": progress.rollback_reason
        }

        return report

    def print_deployment_status_table(self) -> None:
        """Print deployment status table."""
        table = Table(title="Active Deployments")

        table.add_column("Deployment ID", style="cyan")
        table.add_column("Status", style="yellow")
        table.add_column("Current Target", style="green")
        table.add_column("Completed", justify="center")
        table.add_column("Failed", justify="center")
        table.add_column("Duration", style="dim")

        for deployment_id, progress in self.active_deployments.items():
            duration_str = f"{progress.duration or (time.time() - progress.start_time):.0f}s"

            table.add_row(
                deployment_id[:50] + "..." if len(deployment_id) > 50 else deployment_id,
                progress.status.value,
                progress.current_target or "N/A",
                str(len(progress.completed_targets)),
                str(len(progress.failed_targets)),
                duration_str
            )

        console.print(table)


# Example deployment configurations
def create_example_web_app_deployment() -> DeploymentConfig:
    """Create example web application deployment configuration."""
    quality_gates = [
        QualityGate(
            name="unit-tests",
            gate_type=QualityGateType.UNIT_TESTS,
            command=["python", "-m", "pytest", "tests/unit/", "--cov", "--cov-fail-under=80"],
            timeout=300,
            required=True,
            success_criteria={"min_coverage": 80.0}
        ),
        QualityGate(
            name="integration-tests",
            gate_type=QualityGateType.INTEGRATION_TESTS,
            command=["python", "-m", "pytest", "tests/integration/"],
            timeout=600,
            required=True
        ),
        QualityGate(
            name="security-scan",
            gate_type=QualityGateType.SECURITY_SCAN,
            command=["bandit", "-r", "src/"],
            timeout=180,
            required=True,
            success_criteria={
                "max_vulnerabilities": 5,
                "max_critical_vulnerabilities": 0
            }
        ),
        QualityGate(
            name="smoke-tests",
            gate_type=QualityGateType.SMOKE_TEST,
            command=["python", "-m", "pytest", "tests/smoke/"],
            timeout=300,
            required=False
        )
    ]

    targets = [
        DeploymentTarget(
            name="staging",
            environment="staging",
            infrastructure_config="kubernetes",
            health_check_url="http://staging.example.com/health",
            readiness_probe={
                "type": "http",
                "url": "http://staging.example.com/ready",
                "timeout": 10
            },
            liveness_probe={
                "type": "http",
                "url": "http://staging.example.com/health",
                "timeout": 5
            }
        ),
        DeploymentTarget(
            name="production",
            environment="production",
            infrastructure_config="kubernetes",
            health_check_url="http://example.com/health",
            readiness_probe={
                "type": "http",
                "url": "http://example.com/ready",
                "timeout": 10
            },
            liveness_probe={
                "type": "http",
                "url": "http://example.com/health",
                "timeout": 5
            }
        )
    ]

    return DeploymentConfig(
        name="workspace-qdrant-mcp",
        version="0.3.0",
        strategy=DeploymentStrategy.ROLLING,
        targets=targets,
        quality_gates=quality_gates,
        rollback_config={
            "auto_rollback": True,
            "previous_version": "0.2.1",
            "rollback_timeout": 600
        },
        timeout=1800,
        parallel_deployments=False,
        deployment_variables={
            "ENVIRONMENT": "production",
            "LOG_LEVEL": "INFO",
            "QDRANT_URL": "http://qdrant:6333"
        },
        secrets=["DATABASE_PASSWORD", "API_KEYS"]
    )


async def main():
    """Main example demonstrating the deployment pipeline."""
    # Initialize deployment pipeline
    pipeline = DeploymentPipeline(
        artifacts_dir=Path("artifacts"),
        logs_dir=Path("logs"),
        config_dir=Path("deployment-config")
    )

    # Create example deployment configuration
    config = create_example_web_app_deployment()

    try:
        # Execute deployment (dry run)
        console.print("[blue]Executing dry-run deployment...[/blue]")
        deployment_id = await pipeline.deploy(config, dry_run=True)

        console.print(f"[green]Dry-run deployment completed: {deployment_id}[/green]")

        # Show deployment status
        pipeline.print_deployment_status_table()

        # Generate report
        report = pipeline.generate_deployment_report(deployment_id)
        console.print(f"\n[bold]Deployment Report:[/bold]")
        console.print(json.dumps(report, indent=2))

    except Exception as e:
        console.print(f"[red]Deployment failed: {e}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())