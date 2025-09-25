"""
Integration tests for complete ML system.

Tests the integration of all ML components including config, training pipeline,
model registry, deployment manager, and monitoring system working together.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import shutil

from src.python.common.ml import (
    MLConfig,
    TrainingPipeline,
    ModelRegistry,
    DeploymentManager,
    ModelMonitor
)
from src.python.common.ml.config.ml_config import MLModelConfig, ModelType, MLTaskType
from src.python.common.ml.monitoring.model_monitor import PerformanceMetrics
from src.python.common.ml.management.deployment_manager import DeploymentStage


class TestMLSystemIntegration:
    """Test integration of complete ML system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_path = Path(tempfile.mkdtemp())

        # Create comprehensive ML config
        self.config = MLConfig(
            project_name="ml_integration_test",
            model_directory=self.temp_path / "models",
            artifacts_directory=self.temp_path / "artifacts"
        )

        # Initialize all ML components
        self.model_registry = ModelRegistry(self.config)
        self.deployment_manager = DeploymentManager(self.config, self.model_registry)

        # Patch threading for monitoring system
        with patch('threading.Thread'):
            self.monitor = ModelMonitor(self.config)
            self.monitor.monitoring_active = False

    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
        if self.temp_path.exists():
            shutil.rmtree(self.temp_path)

    def test_ml_system_components_initialization(self):
        """Test that all ML system components initialize properly."""
        # Verify config is shared across components
        assert self.model_registry.config == self.config
        assert self.deployment_manager.config == self.config
        assert self.monitor.config == self.config

        # Verify directories are created
        assert self.config.model_directory.exists()
        assert self.config.artifacts_directory.exists()
        assert self.monitor.monitoring_dir.exists()

    def test_complete_ml_workflow_simulation(self):
        """Test a complete ML workflow from training to monitoring."""
        # 1. Create training data simulation
        X_train = np.random.rand(100, 4)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.rand(20, 4)
        y_test = np.random.randint(0, 2, 20)

        # 2. Create model config for training
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"n_estimators": 10, "max_depth": 3}
        )

        # 3. Initialize training pipeline
        training_pipeline = TrainingPipeline(self.config)

        # 4. Simulate model training (mock the actual training)
        with patch.object(training_pipeline, '_train_model') as mock_train:
            # Mock successful training result
            from sklearn.ensemble import RandomForestClassifier
            mock_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
            mock_model.fit(X_train, y_train)

            from src.python.common.ml.pipeline.training_pipeline import TrainingResult
            from src.python.common.ml.config.ml_config import MLPerformanceMetrics

            mock_metrics = MLPerformanceMetrics(
                accuracy=0.95,
                precision=0.93,
                recall=0.97,
                f1_score=0.95
            )

            mock_result = TrainingResult(
                model=mock_model,
                metrics=mock_metrics,
                model_config=model_config,
                feature_names=["feature1", "feature2", "feature3", "feature4"],
                best_params={"n_estimators": 10, "max_depth": 3}
            )
            mock_train.return_value = mock_result

            # Train the model
            results = training_pipeline.fit(X_train, y_train)
            assert len(results) > 0
            result = results[0]

        # 5. Register model in registry
        model_id = self.model_registry.register_model(
            result,
            name="integration_test_model",
            description="Model for integration testing"
        )
        assert model_id is not None

        # 6. Set up monitoring baseline
        features = {
            "feature1": X_train[:, 0],
            "feature2": X_train[:, 1],
            "feature3": X_train[:, 2],
            "feature4": X_train[:, 3]
        }

        baseline_metrics = PerformanceMetrics(
            model_id=model_id,
            timestamp=datetime.now(),
            accuracy=0.95,
            f1_score=0.95
        )

        self.monitor.set_baseline_profile(model_id, features, baseline_metrics)

        # 7. Deploy model to staging
        deployment_id = self.deployment_manager.deploy_model(
            model_id,
            DeploymentStage.STAGING
        )
        assert deployment_id is not None

        # 8. Simulate monitoring with new data (no drift)
        new_features = {
            "feature1": X_test[:, 0],
            "feature2": X_test[:, 1],
            "feature3": X_test[:, 2],
            "feature4": X_test[:, 3]
        }

        drift_scores = self.monitor.monitor_data_drift(model_id, new_features)
        assert len(drift_scores) == 4

        # Should have low drift scores for similar data
        for score in drift_scores.values():
            assert score < 0.5

        # 9. Check model health
        health = self.monitor.get_model_health(model_id)
        assert health["model_id"] == model_id
        assert health["status"] in ["healthy", "warning", "degraded", "critical"]

        # 10. Verify deployment status
        deployment_status = self.deployment_manager.get_deployment_status(deployment_id)
        assert deployment_status is not None
        assert deployment_status.model_id == model_id

    def test_ml_component_interactions(self):
        """Test interactions between ML components."""
        # Test that deployment manager can work with model registry
        mock_model = Mock()
        mock_model.model_id = "test_model_123"
        mock_model.name = "test_model"
        mock_model.version = "1.0.0"

        self.model_registry.get_model_by_name = Mock(return_value=mock_model)
        self.model_registry.load_model = Mock(return_value=Mock())

        # Deploy should work with registry integration
        deployment_id = self.deployment_manager.deploy_model("test_model", DeploymentStage.STAGING)
        assert deployment_id is not None

        # Verify model registry was called
        self.model_registry.get_model_by_name.assert_called_once_with("test_model")

    def test_error_handling_across_components(self):
        """Test error handling across ML system components."""
        # Test registry error handling
        with pytest.raises(Exception):
            self.model_registry.get_model("nonexistent_model")

        # Test deployment error handling
        self.model_registry.get_model_by_name = Mock(return_value=None)

        with pytest.raises(Exception):
            self.deployment_manager.deploy_model("nonexistent_model", DeploymentStage.STAGING)

        # Test monitoring error handling
        with pytest.raises(Exception):
            # Should fail when no baseline is set
            self.monitor.monitor_data_drift("nonexistent_model", {"feature1": np.array([1, 2, 3])})

    def test_configuration_consistency(self):
        """Test that configuration is consistent across all components."""
        # All components should use the same directories
        assert str(self.model_registry.config.model_directory) == str(self.config.model_directory)
        assert str(self.deployment_manager.config.model_directory) == str(self.config.model_directory)
        assert str(self.monitor.config.artifacts_directory) == str(self.config.artifacts_directory)

        # Directory structure should be correct
        expected_dirs = [
            self.config.model_directory,
            self.config.artifacts_directory,
            self.deployment_manager.deployment_dir,
            self.monitor.monitoring_dir
        ]

        for dir_path in expected_dirs:
            assert dir_path.exists(), f"Directory {dir_path} should exist"

    def test_data_flow_between_components(self):
        """Test data flow between different ML components."""
        from sklearn.ensemble import RandomForestClassifier
        from src.python.common.ml.pipeline.training_pipeline import TrainingResult
        from src.python.common.ml.config.ml_config import MLPerformanceMetrics

        # Create a training result
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X_dummy = np.random.rand(50, 3)
        y_dummy = np.random.randint(0, 2, 50)
        model.fit(X_dummy, y_dummy)

        metrics = MLPerformanceMetrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85
        )

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION
        )

        training_result = TrainingResult(
            model=model,
            metrics=metrics,
            model_config=model_config,
            feature_names=["f1", "f2", "f3"]
        )

        # 1. Register model
        model_id = self.model_registry.register_model(training_result, "dataflow_test")

        # 2. Set monitoring baseline
        features = {"f1": X_dummy[:, 0], "f2": X_dummy[:, 1], "f3": X_dummy[:, 2]}
        monitor_metrics = PerformanceMetrics(
            model_id=model_id,
            timestamp=datetime.now(),
            accuracy=0.85
        )
        self.monitor.set_baseline_profile(model_id, features, monitor_metrics)

        # 3. Deploy model
        deployment_id = self.deployment_manager.deploy_model(model_id, DeploymentStage.STAGING)

        # 4. Verify data consistency across components
        # Model should be retrievable from registry
        stored_model = self.model_registry.get_model(model_id)
        assert stored_model.model_id == model_id

        # Deployment should reference correct model
        deployment = self.deployment_manager.get_deployment_status(deployment_id)
        assert deployment.model_id == model_id

        # Monitoring should have baseline for this model
        assert model_id in self.monitor.model_profiles
        assert model_id in self.monitor.baseline_metrics

    def test_system_scalability_simulation(self):
        """Test system behavior with multiple models and deployments."""
        # Create multiple models simulation
        model_ids = []

        for i in range(3):
            # Mock training result
            mock_result = Mock()
            mock_result.model = Mock()
            mock_result.metrics = Mock()
            mock_result.metrics.to_dict.return_value = {"accuracy": 0.9 + i * 0.01}
            mock_result.model_config = MLModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                task_type=MLTaskType.CLASSIFICATION
            )
            mock_result.feature_names = ["feature1", "feature2"]
            mock_result.best_params = {}

            # Register model
            model_id = self.model_registry.register_model(
                mock_result,
                f"scale_test_model_{i}",
                description=f"Scale test model {i}"
            )
            model_ids.append(model_id)

        # Deploy all models
        deployment_ids = []
        for model_id in model_ids:
            deployment_id = self.deployment_manager.deploy_model(model_id, DeploymentStage.STAGING)
            deployment_ids.append(deployment_id)

        # Set up monitoring for all models
        for model_id in model_ids:
            features = {
                "feature1": np.random.rand(20),
                "feature2": np.random.rand(20)
            }
            self.monitor.set_baseline_profile(model_id, features)

        # Verify all models are tracked
        all_deployments = self.deployment_manager.list_deployments()
        assert len(all_deployments) == 3

        assert len(self.monitor.model_profiles) == 3

        # Verify system can handle queries for all models
        for model_id in model_ids:
            health = self.monitor.get_model_health(model_id)
            assert health["model_id"] == model_id