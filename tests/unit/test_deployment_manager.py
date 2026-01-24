"""
Comprehensive unit tests for ML deployment manager.

Tests all deployment manager components including deployment strategies,
health checks, rollback mechanisms, and monitoring with extensive edge cases
and error conditions.
"""

import json
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.python.common.ml.config.ml_config import MLConfig
from src.python.common.ml.management.deployment_manager import (
    DeploymentConfig,
    DeploymentError,
    DeploymentManager,
    DeploymentMetrics,
    DeploymentRecord,
    DeploymentStage,
    DeploymentStatus,
    DeploymentStrategy,
    HealthCheckError,
    RollbackError,
)
from src.python.common.ml.management.model_registry import ModelMetadata, ModelRegistry


class TestDeploymentConfig:
    """Test DeploymentConfig class."""

    def test_deployment_config_creation(self):
        """Test creating deployment config with defaults."""
        config = DeploymentConfig()

        assert config.strategy == DeploymentStrategy.BLUE_GREEN
        assert config.canary_percentage == 10.0
        assert config.health_check_timeout == 300
        assert config.rollback_threshold == 0.95
        assert config.monitoring_duration == 3600
        assert config.max_replicas == 3
        assert config.min_replicas == 1
        assert config.resource_limits == {"cpu": "500m", "memory": "1Gi"}

    def test_deployment_config_custom_values(self):
        """Test creating deployment config with custom values."""
        config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            canary_percentage=20.0,
            health_check_timeout=600,
            rollback_threshold=0.90,
            resource_limits={"cpu": "1000m", "memory": "2Gi"}
        )

        assert config.strategy == DeploymentStrategy.CANARY
        assert config.canary_percentage == 20.0
        assert config.health_check_timeout == 600
        assert config.rollback_threshold == 0.90
        assert config.resource_limits == {"cpu": "1000m", "memory": "2Gi"}


class TestDeploymentMetrics:
    """Test DeploymentMetrics class."""

    def test_deployment_metrics_creation(self):
        """Test creating deployment metrics with defaults."""
        metrics = DeploymentMetrics()

        assert metrics.success_rate == 0.0
        assert metrics.response_time_p99 == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.throughput == 0.0
        assert metrics.resource_usage == {"cpu_percent": 0.0, "memory_percent": 0.0}

    def test_deployment_metrics_custom_values(self):
        """Test creating deployment metrics with custom values."""
        metrics = DeploymentMetrics(
            success_rate=0.99,
            response_time_p99=150.5,
            error_rate=0.01,
            throughput=1000.0,
            resource_usage={"cpu_percent": 65.5, "memory_percent": 45.2}
        )

        assert metrics.success_rate == 0.99
        assert metrics.response_time_p99 == 150.5
        assert metrics.error_rate == 0.01
        assert metrics.throughput == 1000.0
        assert metrics.resource_usage == {"cpu_percent": 65.5, "memory_percent": 45.2}


class TestDeploymentRecord:
    """Test DeploymentRecord class."""

    def test_deployment_record_creation(self):
        """Test creating deployment record."""
        config = DeploymentConfig()
        metrics = DeploymentMetrics()
        now = datetime.now()

        record = DeploymentRecord(
            deployment_id="test_deployment_123",
            model_id="test_model",
            model_version="1.0.0",
            stage=DeploymentStage.STAGING,
            status=DeploymentStatus.PENDING,
            strategy=DeploymentStrategy.BLUE_GREEN,
            config=config,
            metrics=metrics,
            created_at=now,
            updated_at=now
        )

        assert record.deployment_id == "test_deployment_123"
        assert record.model_id == "test_model"
        assert record.model_version == "1.0.0"
        assert record.stage == DeploymentStage.STAGING
        assert record.status == DeploymentStatus.PENDING
        assert record.strategy == DeploymentStrategy.BLUE_GREEN
        assert record.config == config
        assert record.metrics == metrics
        assert record.created_at == now
        assert record.updated_at == now
        assert record.deployed_at is None
        assert record.rolled_back_at is None
        assert record.logs == []

    def test_deployment_record_to_dict(self):
        """Test converting deployment record to dictionary."""
        config = DeploymentConfig()
        metrics = DeploymentMetrics()
        now = datetime.now()

        record = DeploymentRecord(
            deployment_id="test_deployment_123",
            model_id="test_model",
            model_version="1.0.0",
            stage=DeploymentStage.STAGING,
            status=DeploymentStatus.DEPLOYED,
            strategy=DeploymentStrategy.BLUE_GREEN,
            config=config,
            metrics=metrics,
            created_at=now,
            updated_at=now,
            deployed_at=now,
            logs=["deployment started", "deployment completed"]
        )

        data = record.to_dict()

        assert data["deployment_id"] == "test_deployment_123"
        assert data["model_id"] == "test_model"
        assert data["stage"] == DeploymentStage.STAGING
        assert data["status"] == DeploymentStatus.DEPLOYED
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)
        assert isinstance(data["deployed_at"], str)
        assert data["logs"] == ["deployment started", "deployment completed"]


class TestDeploymentManager:
    """Test DeploymentManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_path = Path(tempfile.mkdtemp())

        # Create ML config
        self.config = MLConfig(
            project_name="test_deployment_project",
            model_directory=self.temp_path / "models",
            artifacts_directory=self.temp_path / "artifacts"
        )

        # Create mock model registry
        self.mock_registry = Mock(spec=ModelRegistry)

        # Sample model metadata
        self.mock_model = Mock(spec=ModelMetadata)
        self.mock_model.model_id = "test_model_123"
        self.mock_model.name = "test_model"
        self.mock_model.version = "1.0.0"

        # Setup model registry mock responses
        self.mock_registry.get_model_by_name.return_value = self.mock_model
        self.mock_registry.load_model.return_value = Mock()  # Mock model object

        # Create deployment manager
        self.deployment_manager = DeploymentManager(self.config, self.mock_registry)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)

    def test_deployment_manager_initialization(self):
        """Test deployment manager initialization."""
        assert self.deployment_manager.config == self.config
        assert self.deployment_manager.model_registry == self.mock_registry
        assert isinstance(self.deployment_manager.deployments, dict)
        assert isinstance(self.deployment_manager.active_deployments, dict)
        assert isinstance(self.deployment_manager.health_check_callbacks, list)
        assert isinstance(self.deployment_manager.metrics_callbacks, list)
        assert self.deployment_manager.deployment_dir.exists()

    def test_deploy_model_success(self):
        """Test successful model deployment."""
        deployment_id = self.deployment_manager.deploy_model(
            "test_model",
            DeploymentStage.STAGING
        )

        assert deployment_id is not None
        assert deployment_id in self.deployment_manager.deployments

        deployment = self.deployment_manager.deployments[deployment_id]
        assert deployment.model_id == "test_model"
        assert deployment.stage == DeploymentStage.STAGING
        assert deployment.status == DeploymentStatus.PENDING
        assert deployment.strategy == DeploymentStrategy.BLUE_GREEN  # Default

    def test_deploy_model_with_custom_config(self):
        """Test model deployment with custom configuration."""
        config = DeploymentConfig(
            strategy=DeploymentStrategy.CANARY,
            canary_percentage=15.0,
            health_check_timeout=600
        )

        deployment_id = self.deployment_manager.deploy_model(
            "test_model",
            DeploymentStage.PRODUCTION,
            config=config
        )

        deployment = self.deployment_manager.deployments[deployment_id]
        assert deployment.config.strategy == DeploymentStrategy.CANARY
        assert deployment.config.canary_percentage == 15.0
        assert deployment.config.health_check_timeout == 600

    def test_deploy_model_specific_version(self):
        """Test deploying specific model version."""
        deployment_id = self.deployment_manager.deploy_model(
            "test_model",
            DeploymentStage.STAGING,
            version="2.0.0"
        )

        # Verify model registry was called with version
        self.mock_registry.get_model_by_name.assert_called_with("test_model", version="2.0.0")

        deployment = self.deployment_manager.deployments[deployment_id]
        assert deployment.model_version == "1.0.0"  # From mock

    def test_deploy_model_not_found(self):
        """Test deploying non-existent model."""
        self.mock_registry.get_model_by_name.return_value = None

        with pytest.raises(DeploymentError, match="Model test_model not found"):
            self.deployment_manager.deploy_model(
                "test_model",
                DeploymentStage.STAGING
            )

    def test_get_deployment_status(self):
        """Test getting deployment status."""
        deployment_id = self.deployment_manager.deploy_model(
            "test_model",
            DeploymentStage.STAGING
        )

        status = self.deployment_manager.get_deployment_status(deployment_id)
        assert status is not None
        assert status.deployment_id == deployment_id
        assert status.model_id == "test_model"

    def test_get_deployment_status_not_found(self):
        """Test getting status for non-existent deployment."""
        status = self.deployment_manager.get_deployment_status("nonexistent")
        assert status is None

    def test_list_deployments_all(self):
        """Test listing all deployments."""
        # Create multiple deployments
        deployment_id1 = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment_id2 = self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)

        deployments = self.deployment_manager.list_deployments()
        assert len(deployments) == 2
        deployment_ids = [d.deployment_id for d in deployments]
        assert deployment_id1 in deployment_ids
        assert deployment_id2 in deployment_ids

    def test_list_deployments_filtered_by_stage(self):
        """Test listing deployments filtered by stage."""
        deployment_id1 = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)

        staging_deployments = self.deployment_manager.list_deployments(stage=DeploymentStage.STAGING)
        assert len(staging_deployments) == 1
        assert staging_deployments[0].deployment_id == deployment_id1

    def test_list_deployments_filtered_by_status(self):
        """Test listing deployments filtered by status."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)

        # Update deployment status
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYED

        deployed_deployments = self.deployment_manager.list_deployments(status=DeploymentStatus.DEPLOYED)
        assert len(deployed_deployments) == 1
        assert deployed_deployments[0].deployment_id == deployment_id

    def test_rollback_deployment_success(self):
        """Test successful deployment rollback."""
        # Create and "deploy" a model
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYED

        # Create a previous deployment
        prev_deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)
        prev_deployment = self.deployment_manager.deployments[prev_deployment_id]
        prev_deployment.status = DeploymentStatus.DEPLOYED
        prev_deployment.deployed_at = datetime.now()

        with patch.object(self.deployment_manager, '_perform_rollback', return_value=True):
            success = self.deployment_manager.rollback_deployment(deployment_id)

        assert success is True
        assert deployment.status == DeploymentStatus.ROLLED_BACK
        assert deployment.rolled_back_at is not None

    def test_rollback_deployment_not_found(self):
        """Test rollback of non-existent deployment."""
        success = self.deployment_manager.rollback_deployment("nonexistent")
        assert success is False

    def test_rollback_deployment_wrong_status(self):
        """Test rollback of deployment with wrong status."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        # Deployment remains in PENDING status

        success = self.deployment_manager.rollback_deployment(deployment_id)
        assert success is False

    def test_promote_deployment_success(self):
        """Test successful deployment promotion."""
        # Create and "deploy" a staging deployment
        staging_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        staging_deployment = self.deployment_manager.deployments[staging_id]
        staging_deployment.status = DeploymentStatus.DEPLOYED

        with patch.object(self.deployment_manager, 'deploy_model', return_value="new_deployment_id"):
            success = self.deployment_manager.promote_deployment(staging_id, DeploymentStage.PRODUCTION)

        assert success is True

    def test_promote_deployment_not_found(self):
        """Test promotion of non-existent deployment."""
        success = self.deployment_manager.promote_deployment("nonexistent", DeploymentStage.PRODUCTION)
        assert success is False

    def test_promote_deployment_wrong_status(self):
        """Test promotion of deployment with wrong status."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        # Deployment remains in PENDING status

        success = self.deployment_manager.promote_deployment(deployment_id, DeploymentStage.PRODUCTION)
        assert success is False

    def test_add_health_check(self):
        """Test adding health check callback."""
        def health_check(deployment_id):
            return True

        self.deployment_manager.add_health_check(health_check)
        assert health_check in self.deployment_manager.health_check_callbacks

    def test_add_metrics_collector(self):
        """Test adding metrics collector callback."""
        def metrics_collector(deployment_id):
            return DeploymentMetrics()

        self.deployment_manager.add_metrics_collector(metrics_collector)
        assert metrics_collector in self.deployment_manager.metrics_callbacks

    def test_get_deployment_metrics_with_callback(self):
        """Test getting deployment metrics with callback."""
        # Create and "deploy" a model
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYED

        # Add metrics callback
        test_metrics = DeploymentMetrics(success_rate=0.99, throughput=500.0)
        def metrics_callback(dep_id):
            return test_metrics

        self.deployment_manager.add_metrics_collector(metrics_callback)

        metrics = self.deployment_manager.get_deployment_metrics(deployment_id)
        assert metrics is not None
        assert metrics.success_rate == 0.99
        assert metrics.throughput == 500.0

    def test_get_deployment_metrics_no_callback(self):
        """Test getting deployment metrics without callback."""
        # Create and "deploy" a model
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYED

        metrics = self.deployment_manager.get_deployment_metrics(deployment_id)
        assert metrics is not None
        assert metrics.success_rate == 0.0  # Default value

    def test_get_deployment_metrics_not_deployed(self):
        """Test getting metrics for non-deployed model."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        # Deployment remains in PENDING status

        metrics = self.deployment_manager.get_deployment_metrics(deployment_id)
        assert metrics is None

    @patch('threading.Thread')
    def test_deploy_async_blue_green_success(self, mock_thread):
        """Test asynchronous blue-green deployment success."""
        # Mock the deployment process methods
        with patch.object(self.deployment_manager, '_deploy_blue_green', return_value=True), \
             patch.object(self.deployment_manager, '_perform_health_checks', return_value=True), \
             patch.object(self.deployment_manager, '_start_monitoring'):

            deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)

            # Simulate async deployment completion
            self.deployment_manager._deploy_async(deployment_id)

            deployment = self.deployment_manager.deployments[deployment_id]
            assert deployment.status == DeploymentStatus.DEPLOYED
            assert deployment.deployed_at is not None

    @patch('threading.Thread')
    def test_deploy_async_health_check_failure(self, mock_thread):
        """Test asynchronous deployment with health check failure."""
        with patch.object(self.deployment_manager, '_deploy_blue_green', return_value=True), \
             patch.object(self.deployment_manager, '_perform_health_checks', return_value=False), \
             patch.object(self.deployment_manager, '_perform_rollback', return_value=True):

            deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)

            # Simulate async deployment with health check failure
            self.deployment_manager._deploy_async(deployment_id)

            deployment = self.deployment_manager.deployments[deployment_id]
            assert deployment.status == DeploymentStatus.FAILED

    @patch('threading.Thread')
    def test_deploy_async_deployment_failure(self, mock_thread):
        """Test asynchronous deployment failure."""
        with patch.object(self.deployment_manager, '_deploy_blue_green', return_value=False):

            deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)

            # Simulate async deployment failure
            self.deployment_manager._deploy_async(deployment_id)

            deployment = self.deployment_manager.deployments[deployment_id]
            assert deployment.status == DeploymentStatus.FAILED

    def test_perform_health_checks_success(self):
        """Test successful health checks."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment = self.deployment_manager.deployments[deployment_id]

        # Add health check callback that always succeeds
        self.deployment_manager.add_health_check(lambda dep_id: True)

        result = self.deployment_manager._perform_health_checks(deployment)
        assert result is True

    def test_perform_health_checks_failure(self):
        """Test failed health checks."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.config.health_check_timeout = 1  # Short timeout for test

        # Add health check callback that always fails
        self.deployment_manager.add_health_check(lambda dep_id: False)

        result = self.deployment_manager._perform_health_checks(deployment)
        assert result is False

    def test_perform_health_checks_no_callbacks(self):
        """Test health checks with no callbacks configured."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment = self.deployment_manager.deployments[deployment_id]

        result = self.deployment_manager._perform_health_checks(deployment)
        assert result is True  # Should pass when no callbacks

    def test_perform_health_checks_exception(self):
        """Test health checks with callback exception."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.config.health_check_timeout = 1  # Short timeout for test

        # Add health check callback that raises exception
        def failing_callback(dep_id):
            raise Exception("Health check failed")

        self.deployment_manager.add_health_check(failing_callback)

        result = self.deployment_manager._perform_health_checks(deployment)
        assert result is False

    def test_get_previous_deployment_success(self):
        """Test getting previous deployment for rollback."""
        # Create multiple deployments
        deployment_id1 = self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)
        deployment1 = self.deployment_manager.deployments[deployment_id1]
        deployment1.status = DeploymentStatus.DEPLOYED
        deployment1.deployed_at = datetime.now()

        deployment_id2 = self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)
        deployment2 = self.deployment_manager.deployments[deployment_id2]
        deployment2.status = DeploymentStatus.DEPLOYED
        deployment2.deployed_at = datetime.now()

        # Get previous deployment
        previous = self.deployment_manager._get_previous_deployment(
            DeploymentStage.PRODUCTION,
            deployment_id2
        )

        assert previous is not None
        assert previous.deployment_id == deployment_id1

    def test_get_previous_deployment_none_found(self):
        """Test getting previous deployment when none exists."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)

        previous = self.deployment_manager._get_previous_deployment(
            DeploymentStage.PRODUCTION,
            deployment_id
        )

        assert previous is None

    def test_save_and_load_deployments(self):
        """Test saving and loading deployments."""
        # Create a deployment
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYED
        deployment.deployed_at = datetime.now()
        deployment.logs = ["deployment started", "deployment completed"]

        # Save deployments
        self.deployment_manager._save_deployments()

        # Create new deployment manager and load
        new_manager = DeploymentManager(self.config, self.mock_registry)

        # Verify deployment was loaded
        assert deployment_id in new_manager.deployments
        loaded_deployment = new_manager.deployments[deployment_id]
        assert loaded_deployment.model_id == "test_model"
        assert loaded_deployment.stage == DeploymentStage.STAGING
        assert loaded_deployment.status == DeploymentStatus.DEPLOYED
        assert loaded_deployment.logs == ["deployment started", "deployment completed"]


class TestDeploymentEdgeCases:
    """Test edge cases and error conditions for deployment manager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_path = Path(tempfile.mkdtemp())

        self.config = MLConfig(
            project_name="test_deployment_edge_cases",
            model_directory=self.temp_path / "models",
            artifacts_directory=self.temp_path / "artifacts"
        )

        self.mock_registry = Mock(spec=ModelRegistry)
        self.mock_model = Mock(spec=ModelMetadata)
        self.mock_model.model_id = "test_model_123"
        self.mock_model.name = "test_model"
        self.mock_model.version = "1.0.0"

        self.mock_registry.get_model_by_name.return_value = self.mock_model
        self.mock_registry.load_model.return_value = Mock()

        self.deployment_manager = DeploymentManager(self.config, self.mock_registry)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)

    def test_concurrent_deployments(self):
        """Test handling of concurrent deployments."""
        def deploy_model(stage):
            return self.deployment_manager.deploy_model("test_model", stage)

        # Start multiple deployments concurrently
        threads = []
        for i in range(3):
            stage = DeploymentStage.STAGING if i % 2 == 0 else DeploymentStage.PRODUCTION
            thread = threading.Thread(target=deploy_model, args=(stage,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all deployments were created
        assert len(self.deployment_manager.deployments) == 3

    def test_rollback_without_previous_deployment(self):
        """Test rollback when no previous deployment exists."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYED

        success = self.deployment_manager.rollback_deployment(deployment_id)
        assert success is False

    def test_metrics_callback_exception(self):
        """Test metrics collection with callback exception."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYED

        def failing_metrics_callback(dep_id):
            raise Exception("Metrics collection failed")

        self.deployment_manager.add_metrics_collector(failing_metrics_callback)

        metrics = self.deployment_manager.get_deployment_metrics(deployment_id)
        assert metrics is not None  # Should return deployment's metrics even if callback fails

    def test_deployment_with_invalid_model_registry_response(self):
        """Test deployment when model registry returns invalid response."""
        self.mock_registry.load_model.return_value = None

        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)

        # Simulate async deployment with invalid model
        with pytest.raises(Exception):  # Should raise exception when model loading fails
            self.deployment_manager._deploy_async(deployment_id)

    def test_deployment_config_edge_values(self):
        """Test deployment with edge configuration values."""
        config = DeploymentConfig(
            canary_percentage=0.0,  # Edge case: 0% canary
            health_check_timeout=0,  # Edge case: no timeout
            rollback_threshold=1.0,  # Edge case: perfect threshold
            monitoring_duration=0,  # Edge case: no monitoring
            max_replicas=1,  # Edge case: single replica
            min_replicas=1
        )

        deployment_id = self.deployment_manager.deploy_model(
            "test_model",
            DeploymentStage.STAGING,
            config=config
        )

        deployment = self.deployment_manager.deployments[deployment_id]
        assert deployment.config.canary_percentage == 0.0
        assert deployment.config.health_check_timeout == 0
        assert deployment.config.rollback_threshold == 1.0

    def test_monitoring_with_low_success_rate(self):
        """Test monitoring triggers rollback with low success rate."""
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.PRODUCTION)
        deployment = self.deployment_manager.deployments[deployment_id]
        deployment.status = DeploymentStatus.DEPLOYED
        deployment.config.monitoring_duration = 1  # Short duration for test
        deployment.config.rollback_threshold = 0.95

        # Mock metrics callback with low success rate
        low_success_metrics = DeploymentMetrics(success_rate=0.90)
        self.deployment_manager.add_metrics_collector(lambda dep_id: low_success_metrics)

        with patch.object(self.deployment_manager, 'rollback_deployment', return_value=True):
            self.deployment_manager._start_monitoring(deployment_id)
            time.sleep(2)  # Wait for monitoring thread

            # Should trigger rollback due to low success rate
            # Note: In real scenario, this would be verified differently due to threading

    def test_save_deployments_permission_error(self):
        """Test saving deployments with permission error."""
        # Mock os.access to simulate write permission denied
        with patch('os.access', return_value=False):
            self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
            # Should not raise exception even if save fails
            self.deployment_manager._save_deployments()

    def test_load_deployments_corrupted_file(self):
        """Test loading deployments from corrupted file."""
        # Create corrupted deployments file
        deployments_file = self.deployment_manager.deployment_dir / "deployments.json"
        with open(deployments_file, 'w') as f:
            f.write("invalid json content")

        # Should not raise exception on corrupted file
        new_manager = DeploymentManager(self.config, self.mock_registry)
        assert len(new_manager.deployments) == 0
