"""
Advanced ML Deployment Manager for automated model deployment and serving.

This module provides comprehensive deployment automation including staging,
production deployments, rollback mechanisms, health monitoring, and
A/B testing capabilities for machine learning models.
"""

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ..config.ml_config import MLConfig
from .model_registry import ModelRegistry


class DeploymentStage(Enum):
    """Deployment stages."""
    STAGING = "staging"
    CANARY = "canary"
    PRODUCTION = "production"
    ROLLBACK = "rollback"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    canary_percentage: float = 10.0  # Percentage of traffic for canary
    health_check_timeout: int = 300  # Seconds to wait for health checks
    rollback_threshold: float = 0.95  # Minimum success rate to prevent rollback
    monitoring_duration: int = 3600  # Seconds to monitor after deployment
    max_replicas: int = 3
    min_replicas: int = 1
    resource_limits: dict[str, str] = None

    def __post_init__(self):
        if self.resource_limits is None:
            self.resource_limits = {"cpu": "500m", "memory": "1Gi"}


@dataclass
class DeploymentMetrics:
    """Metrics for a deployment."""
    success_rate: float = 0.0
    response_time_p99: float = 0.0  # 99th percentile response time in ms
    error_rate: float = 0.0
    throughput: float = 0.0  # requests per second
    resource_usage: dict[str, float] = None  # CPU/Memory usage

    def __post_init__(self):
        if self.resource_usage is None:
            self.resource_usage = {"cpu_percent": 0.0, "memory_percent": 0.0}


@dataclass
class DeploymentRecord:
    """Record of a model deployment."""
    deployment_id: str
    model_id: str
    model_version: str
    stage: DeploymentStage
    status: DeploymentStatus
    strategy: DeploymentStrategy
    config: DeploymentConfig
    metrics: DeploymentMetrics
    created_at: datetime
    updated_at: datetime
    deployed_at: datetime | None = None
    rolled_back_at: datetime | None = None
    logs: list[str] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.deployed_at:
            data['deployed_at'] = self.deployed_at.isoformat()
        if self.rolled_back_at:
            data['rolled_back_at'] = self.rolled_back_at.isoformat()
        return data


class DeploymentError(Exception):
    """Base exception for deployment errors."""
    pass


class HealthCheckError(DeploymentError):
    """Exception for health check failures."""
    pass


class RollbackError(DeploymentError):
    """Exception for rollback failures."""
    pass


class DeploymentManager:
    """
    Advanced ML deployment manager with comprehensive deployment automation.

    Features:
    - Multiple deployment strategies (blue-green, canary, rolling, shadow)
    - Automated health checks and monitoring
    - Intelligent rollback mechanisms
    - A/B testing support
    - Resource management and scaling
    - Comprehensive logging and metrics collection
    """

    def __init__(self, config: MLConfig, model_registry: ModelRegistry):
        """
        Initialize deployment manager.

        Args:
            config: ML configuration
            model_registry: Model registry instance
        """
        self.config = config
        self.model_registry = model_registry
        self.deployments: dict[str, DeploymentRecord] = {}
        self.active_deployments: dict[str, str] = {}  # stage -> deployment_id
        self.health_check_callbacks: list[Callable[[str], bool]] = []
        self.metrics_callbacks: list[Callable[[str], DeploymentMetrics]] = []

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Create deployment directory
        self.deployment_dir = config.model_directory / "deployments"
        self.deployment_dir.mkdir(parents=True, exist_ok=True)

        # Load existing deployments
        self._load_deployments()

    def deploy_model(
        self,
        model_id: str,
        stage: DeploymentStage,
        config: DeploymentConfig | None = None,
        version: str | None = None
    ) -> str:
        """
        Deploy a model to specified stage.

        Args:
            model_id: Model ID to deploy
            stage: Deployment stage
            config: Deployment configuration (uses default if None)
            version: Specific version to deploy (uses latest if None)

        Returns:
            Deployment ID

        Raises:
            DeploymentError: If deployment fails
        """
        try:
            # Get model metadata
            if version:
                model = self.model_registry.get_model_by_name(model_id, version=version)
            else:
                model = self.model_registry.get_model_by_name(model_id)

            if not model:
                raise DeploymentError(f"Model {model_id} not found")

            # Use default config if none provided
            if config is None:
                config = DeploymentConfig()

            # Generate deployment ID
            deployment_id = f"{model_id}_{stage.value}_{int(time.time())}"

            # Create deployment record
            deployment = DeploymentRecord(
                deployment_id=deployment_id,
                model_id=model_id,
                model_version=model.version,
                stage=stage,
                status=DeploymentStatus.PENDING,
                strategy=config.strategy,
                config=config,
                metrics=DeploymentMetrics(),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )

            self.deployments[deployment_id] = deployment
            deployment.logs.append(f"Deployment {deployment_id} created")

            # Start deployment process
            threading.Thread(
                target=self._deploy_async,
                args=(deployment_id,),
                daemon=True
            ).start()

            self.logger.info(f"Started deployment {deployment_id} for model {model_id}")
            return deployment_id

        except Exception as e:
            error_msg = f"Failed to deploy model {model_id}: {str(e)}"
            self.logger.error(error_msg)
            raise DeploymentError(error_msg)

    def get_deployment_status(self, deployment_id: str) -> DeploymentRecord | None:
        """Get deployment status and details."""
        return self.deployments.get(deployment_id)

    def list_deployments(
        self,
        stage: DeploymentStage | None = None,
        status: DeploymentStatus | None = None
    ) -> list[DeploymentRecord]:
        """
        List deployments with optional filtering.

        Args:
            stage: Filter by deployment stage
            status: Filter by deployment status

        Returns:
            List of deployment records
        """
        deployments = list(self.deployments.values())

        if stage:
            deployments = [d for d in deployments if d.stage == stage]
        if status:
            deployments = [d for d in deployments if d.status == status]

        return sorted(deployments, key=lambda x: x.created_at, reverse=True)

    def rollback_deployment(self, deployment_id: str) -> bool:
        """
        Rollback a deployment.

        Args:
            deployment_id: Deployment to rollback

        Returns:
            True if rollback successful, False otherwise
        """
        try:
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                self.logger.error(f"Deployment {deployment_id} not found")
                return False

            if deployment.status != DeploymentStatus.DEPLOYED:
                self.logger.error(f"Cannot rollback deployment {deployment_id} with status {deployment.status}")
                return False

            deployment.status = DeploymentStatus.ROLLING_BACK
            deployment.logs.append(f"Starting rollback for deployment {deployment_id}")

            # Perform rollback based on strategy
            success = self._perform_rollback(deployment)

            if success:
                deployment.status = DeploymentStatus.ROLLED_BACK
                deployment.rolled_back_at = datetime.now()
                deployment.logs.append("Rollback completed successfully")
                self.logger.info(f"Rollback completed for deployment {deployment_id}")
            else:
                deployment.logs.append("Rollback failed")
                self.logger.error(f"Rollback failed for deployment {deployment_id}")

            deployment.updated_at = datetime.now()
            self._save_deployments()

            return success

        except Exception as e:
            self.logger.error(f"Error during rollback: {str(e)}")
            return False

    def promote_deployment(self, deployment_id: str, target_stage: DeploymentStage) -> bool:
        """
        Promote a deployment to a higher stage.

        Args:
            deployment_id: Deployment to promote
            target_stage: Target deployment stage

        Returns:
            True if promotion successful
        """
        try:
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                self.logger.error(f"Deployment {deployment_id} not found")
                return False

            if deployment.status != DeploymentStatus.DEPLOYED:
                self.logger.error(f"Cannot promote deployment {deployment_id} with status {deployment.status}")
                return False

            # Create new deployment for target stage
            new_deployment_id = self.deploy_model(
                deployment.model_id,
                target_stage,
                deployment.config,
                deployment.model_version
            )

            deployment.logs.append(f"Promoted to {target_stage.value} as deployment {new_deployment_id}")
            deployment.updated_at = datetime.now()
            self._save_deployments()

            self.logger.info(f"Promoted deployment {deployment_id} to {target_stage.value}")
            return True

        except Exception as e:
            self.logger.error(f"Error promoting deployment: {str(e)}")
            return False

    def add_health_check(self, callback: Callable[[str], bool]):
        """
        Add a health check callback.

        Args:
            callback: Function that takes deployment_id and returns True if healthy
        """
        self.health_check_callbacks.append(callback)

    def add_metrics_collector(self, callback: Callable[[str], DeploymentMetrics]):
        """
        Add a metrics collection callback.

        Args:
            callback: Function that takes deployment_id and returns metrics
        """
        self.metrics_callbacks.append(callback)

    def get_deployment_metrics(self, deployment_id: str) -> DeploymentMetrics | None:
        """Get current metrics for a deployment."""
        deployment = self.deployments.get(deployment_id)
        if not deployment or deployment.status != DeploymentStatus.DEPLOYED:
            return None

        # Collect metrics from callbacks
        if self.metrics_callbacks:
            try:
                metrics = self.metrics_callbacks[0](deployment_id)
                deployment.metrics = metrics
                deployment.updated_at = datetime.now()
                return metrics
            except Exception as e:
                self.logger.error(f"Error collecting metrics for {deployment_id}: {str(e)}")

        return deployment.metrics

    def _deploy_async(self, deployment_id: str):
        """Asynchronous deployment process."""
        try:
            deployment = self.deployments[deployment_id]

            # Update status
            deployment.status = DeploymentStatus.IN_PROGRESS
            deployment.updated_at = datetime.now()
            deployment.logs.append("Starting deployment process")

            # Load model
            model = self.model_registry.load_model(deployment.model_id, deployment.model_version)
            if not model:
                raise DeploymentError("Failed to load model")

            # Deploy based on strategy
            if deployment.config.strategy == DeploymentStrategy.BLUE_GREEN:
                success = self._deploy_blue_green(deployment, model)
            elif deployment.config.strategy == DeploymentStrategy.CANARY:
                success = self._deploy_canary(deployment, model)
            elif deployment.config.strategy == DeploymentStrategy.ROLLING:
                success = self._deploy_rolling(deployment, model)
            elif deployment.config.strategy == DeploymentStrategy.SHADOW:
                success = self._deploy_shadow(deployment, model)
            else:
                raise DeploymentError(f"Unknown deployment strategy: {deployment.config.strategy}")

            if success:
                # Perform health checks
                if self._perform_health_checks(deployment):
                    deployment.status = DeploymentStatus.DEPLOYED
                    deployment.deployed_at = datetime.now()
                    deployment.logs.append("Deployment successful")

                    # Update active deployment tracking
                    self.active_deployments[deployment.stage.value] = deployment_id

                    # Start monitoring
                    self._start_monitoring(deployment_id)
                else:
                    deployment.status = DeploymentStatus.FAILED
                    deployment.logs.append("Health checks failed")
                    self._perform_rollback(deployment)
            else:
                deployment.status = DeploymentStatus.FAILED
                deployment.logs.append("Deployment process failed")

            deployment.updated_at = datetime.now()
            self._save_deployments()

        except Exception as e:
            deployment = self.deployments.get(deployment_id)
            if deployment:
                deployment.status = DeploymentStatus.FAILED
                deployment.logs.append(f"Deployment failed: {str(e)}")
                deployment.updated_at = datetime.now()
                self._save_deployments()

            self.logger.error(f"Deployment {deployment_id} failed: {str(e)}")

    def _deploy_blue_green(self, deployment: DeploymentRecord, model: Any) -> bool:
        """Implement blue-green deployment strategy."""
        try:
            deployment.logs.append("Implementing blue-green deployment")

            # Create new environment (green)
            green_env = self._create_environment(deployment, model)
            if not green_env:
                return False

            # Test green environment
            if self._test_environment(green_env):
                # Switch traffic to green
                self._switch_traffic(green_env)
                # Cleanup old environment (blue)
                self._cleanup_old_environment(deployment.stage)
                deployment.logs.append("Blue-green deployment completed")
                return True
            else:
                self._cleanup_environment(green_env)
                return False

        except Exception as e:
            deployment.logs.append(f"Blue-green deployment error: {str(e)}")
            return False

    def _deploy_canary(self, deployment: DeploymentRecord, model: Any) -> bool:
        """Implement canary deployment strategy."""
        try:
            deployment.logs.append(f"Implementing canary deployment ({deployment.config.canary_percentage}%)")

            # Deploy canary version
            canary_env = self._create_environment(deployment, model)
            if not canary_env:
                return False

            # Route percentage of traffic to canary
            self._route_traffic_percentage(canary_env, deployment.config.canary_percentage)

            # Monitor canary performance
            if self._monitor_canary(deployment, canary_env):
                # Gradually increase traffic
                self._gradual_rollout(canary_env, deployment)
                deployment.logs.append("Canary deployment completed")
                return True
            else:
                self._cleanup_environment(canary_env)
                return False

        except Exception as e:
            deployment.logs.append(f"Canary deployment error: {str(e)}")
            return False

    def _deploy_rolling(self, deployment: DeploymentRecord, model: Any) -> bool:
        """Implement rolling deployment strategy."""
        try:
            deployment.logs.append("Implementing rolling deployment")

            # Get current instances
            current_instances = self._get_current_instances(deployment.stage)

            # Replace instances one by one
            for i, instance in enumerate(current_instances):
                new_instance = self._create_instance(deployment, model)
                if not new_instance:
                    return False

                # Health check new instance
                if self._health_check_instance(new_instance):
                    # Replace old instance
                    self._replace_instance(instance, new_instance)
                    deployment.logs.append(f"Replaced instance {i+1}/{len(current_instances)}")
                else:
                    self._cleanup_instance(new_instance)
                    return False

            deployment.logs.append("Rolling deployment completed")
            return True

        except Exception as e:
            deployment.logs.append(f"Rolling deployment error: {str(e)}")
            return False

    def _deploy_shadow(self, deployment: DeploymentRecord, model: Any) -> bool:
        """Implement shadow deployment strategy."""
        try:
            deployment.logs.append("Implementing shadow deployment")

            # Create shadow environment
            shadow_env = self._create_environment(deployment, model)
            if not shadow_env:
                return False

            # Mirror traffic to shadow without affecting responses
            self._mirror_traffic(shadow_env)

            # Monitor shadow performance
            if self._monitor_shadow(deployment, shadow_env):
                deployment.logs.append("Shadow deployment completed successfully")
                return True
            else:
                self._cleanup_environment(shadow_env)
                return False

        except Exception as e:
            deployment.logs.append(f"Shadow deployment error: {str(e)}")
            return False

    def _perform_health_checks(self, deployment: DeploymentRecord) -> bool:
        """Perform health checks on deployment."""
        if not self.health_check_callbacks:
            deployment.logs.append("No health check callbacks configured, skipping")
            return True

        deployment.logs.append("Starting health checks")
        timeout = deployment.config.health_check_timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            all_healthy = True
            for callback in self.health_check_callbacks:
                try:
                    if not callback(deployment.deployment_id):
                        all_healthy = False
                        break
                except Exception as e:
                    deployment.logs.append(f"Health check error: {str(e)}")
                    all_healthy = False
                    break

            if all_healthy:
                deployment.logs.append("All health checks passed")
                return True

            time.sleep(10)  # Wait before retry

        deployment.logs.append("Health checks timed out")
        return False

    def _perform_rollback(self, deployment: DeploymentRecord) -> bool:
        """Perform rollback for deployment."""
        try:
            deployment.logs.append("Performing rollback")

            # Find previous successful deployment for this stage
            previous_deployment = self._get_previous_deployment(deployment.stage, deployment.deployment_id)
            if not previous_deployment:
                deployment.logs.append("No previous deployment found for rollback")
                return False

            # Restore previous deployment
            success = self._restore_deployment(previous_deployment)
            if success:
                deployment.logs.append(f"Rolled back to deployment {previous_deployment.deployment_id}")

            return success

        except Exception as e:
            deployment.logs.append(f"Rollback error: {str(e)}")
            return False

    def _start_monitoring(self, deployment_id: str):
        """Start monitoring for deployed model."""
        def monitor():
            deployment = self.deployments.get(deployment_id)
            if not deployment:
                return

            end_time = time.time() + deployment.config.monitoring_duration
            while time.time() < end_time and deployment.status == DeploymentStatus.DEPLOYED:
                try:
                    # Collect metrics
                    metrics = self.get_deployment_metrics(deployment_id)
                    if metrics:
                        # Check for rollback conditions
                        if metrics.success_rate < deployment.config.rollback_threshold:
                            deployment.logs.append(f"Success rate {metrics.success_rate} below threshold, initiating rollback")
                            self.rollback_deployment(deployment_id)
                            break

                    time.sleep(60)  # Check every minute

                except Exception as e:
                    deployment.logs.append(f"Monitoring error: {str(e)}")
                    break

            deployment.logs.append("Monitoring period completed")

        threading.Thread(target=monitor, daemon=True).start()

    # Helper methods for deployment strategies (simplified implementations)

    def _create_environment(self, deployment: DeploymentRecord, model: Any) -> str | None:
        """Create deployment environment."""
        try:
            env_id = f"{deployment.deployment_id}_env"
            deployment.logs.append(f"Creating environment {env_id}")
            # Simulate environment creation
            time.sleep(1)
            return env_id
        except:
            return None

    def _test_environment(self, env_id: str) -> bool:
        """Test environment."""
        time.sleep(0.5)
        return True

    def _switch_traffic(self, env_id: str):
        """Switch traffic to new environment."""
        pass

    def _cleanup_old_environment(self, stage: DeploymentStage):
        """Cleanup old environment."""
        pass

    def _cleanup_environment(self, env_id: str):
        """Cleanup environment."""
        pass

    def _route_traffic_percentage(self, env_id: str, percentage: float):
        """Route percentage of traffic to environment."""
        pass

    def _monitor_canary(self, deployment: DeploymentRecord, env_id: str) -> bool:
        """Monitor canary deployment."""
        time.sleep(2)
        return True

    def _gradual_rollout(self, env_id: str, deployment: DeploymentRecord):
        """Gradually increase traffic to canary."""
        pass

    def _get_current_instances(self, stage: DeploymentStage) -> list[str]:
        """Get current instances for stage."""
        return [f"instance_{i}" for i in range(2)]

    def _create_instance(self, deployment: DeploymentRecord, model: Any) -> str | None:
        """Create new instance."""
        return f"instance_{int(time.time())}"

    def _health_check_instance(self, instance_id: str) -> bool:
        """Health check instance."""
        return True

    def _replace_instance(self, old_instance: str, new_instance: str):
        """Replace old instance with new one."""
        pass

    def _cleanup_instance(self, instance_id: str):
        """Cleanup instance."""
        pass

    def _mirror_traffic(self, env_id: str):
        """Mirror traffic to shadow environment."""
        pass

    def _monitor_shadow(self, deployment: DeploymentRecord, env_id: str) -> bool:
        """Monitor shadow deployment."""
        time.sleep(1)
        return True

    def _get_previous_deployment(self, stage: DeploymentStage, exclude_id: str) -> DeploymentRecord | None:
        """Get previous successful deployment for stage."""
        deployments = [
            d for d in self.deployments.values()
            if d.stage == stage and d.status == DeploymentStatus.DEPLOYED
            and d.deployment_id != exclude_id
        ]
        return max(deployments, key=lambda x: x.deployed_at) if deployments else None

    def _restore_deployment(self, deployment: DeploymentRecord) -> bool:
        """Restore previous deployment."""
        try:
            # Simulate restoration
            time.sleep(1)
            return True
        except:
            return False

    def _save_deployments(self):
        """Save deployments to disk."""
        try:
            deployments_file = self.deployment_dir / "deployments.json"
            data = {
                deployment_id: deployment.to_dict()
                for deployment_id, deployment in self.deployments.items()
            }
            with open(deployments_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save deployments: {str(e)}")

    def _load_deployments(self):
        """Load deployments from disk."""
        try:
            deployments_file = self.deployment_dir / "deployments.json"
            if deployments_file.exists():
                with open(deployments_file) as f:
                    data = json.load(f)

                for deployment_id, deployment_data in data.items():
                    # Parse datetime fields
                    deployment_data['created_at'] = datetime.fromisoformat(deployment_data['created_at'])
                    deployment_data['updated_at'] = datetime.fromisoformat(deployment_data['updated_at'])
                    if deployment_data.get('deployed_at'):
                        deployment_data['deployed_at'] = datetime.fromisoformat(deployment_data['deployed_at'])
                    if deployment_data.get('rolled_back_at'):
                        deployment_data['rolled_back_at'] = datetime.fromisoformat(deployment_data['rolled_back_at'])

                    # Convert enums
                    deployment_data['stage'] = DeploymentStage(deployment_data['stage'])
                    deployment_data['status'] = DeploymentStatus(deployment_data['status'])
                    deployment_data['strategy'] = DeploymentStrategy(deployment_data['strategy'])

                    # Create objects
                    deployment_data['config'] = DeploymentConfig(**deployment_data['config'])
                    deployment_data['metrics'] = DeploymentMetrics(**deployment_data['metrics'])

                    self.deployments[deployment_id] = DeploymentRecord(**deployment_data)

        except Exception as e:
            self.logger.error(f"Failed to load deployments: {str(e)}")
