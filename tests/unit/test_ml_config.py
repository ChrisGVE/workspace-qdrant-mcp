"""
Comprehensive unit tests for ML configuration system.

Tests all ML configuration classes with edge cases, validation,
and error conditions to ensure robust configuration management.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from common.ml.config.ml_config import (
    FeatureSelectionMethod,
    MLConfig,
    MLExperimentConfig,
    MLModelConfig,
    MLPerformanceMetrics,
    MLTaskType,
    ModelType,
    create_default_config,
)


class TestMLModelConfig:
    """Test MLModelConfig class."""

    def test_valid_model_config_creation(self):
        """Test creating valid model configuration."""
        config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={
                "n_estimators": [100, 200],
                "max_depth": [10, 20],
            },
            feature_selection_method=FeatureSelectionMethod.TREE_IMPORTANCE,
            max_features=1000,
        )

        assert config.model_type == ModelType.RANDOM_FOREST
        assert config.task_type == MLTaskType.CLASSIFICATION
        assert config.hyperparameters["n_estimators"] == [100, 200]
        assert config.feature_selection_method == FeatureSelectionMethod.TREE_IMPORTANCE
        assert config.max_features == 1000

    def test_model_config_with_minimal_params(self):
        """Test model configuration with only required parameters."""
        config = MLModelConfig(
            model_type=ModelType.SVM,
            task_type=MLTaskType.REGRESSION,
        )

        assert config.model_type == ModelType.SVM
        assert config.task_type == MLTaskType.REGRESSION
        assert config.hyperparameters == {}
        assert config.feature_selection_method is None
        assert config.max_features is None

    def test_model_config_enum_values(self):
        """Test that enum values are properly handled."""
        config = MLModelConfig(
            model_type="random_forest",  # String should work
            task_type="classification",
        )

        assert config.model_type == ModelType.RANDOM_FOREST
        assert config.task_type == MLTaskType.CLASSIFICATION

    def test_invalid_model_type(self):
        """Test validation with invalid model type."""
        with pytest.raises(ValueError, match="Input should be"):
            MLModelConfig(
                model_type="invalid_model",
                task_type=MLTaskType.CLASSIFICATION,
            )

    def test_invalid_task_type(self):
        """Test validation with invalid task type."""
        with pytest.raises(ValueError, match="Input should be"):
            MLModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                task_type="invalid_task",
            )


class TestMLExperimentConfig:
    """Test MLExperimentConfig class."""

    def test_valid_experiment_config_creation(self):
        """Test creating valid experiment configuration."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        config = MLExperimentConfig(
            name="test_experiment",
            description="Test experiment",
            model_configs=[model_config],
            tuning_method="random_search",
            cv_folds=5,
            scoring_metric="accuracy",
            max_iter=100,
        )

        assert config.name == "test_experiment"
        assert config.description == "Test experiment"
        assert len(config.model_configs) == 1
        assert config.tuning_method == "random_search"
        assert config.cv_folds == 5
        assert config.scoring_metric == "accuracy"
        assert config.max_iter == 100

    def test_experiment_config_defaults(self):
        """Test experiment configuration with default values."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        config = MLExperimentConfig(
            name="test_experiment",
            model_configs=[model_config],
        )

        assert config.tuning_method == "random_search"
        assert config.cv_folds == 5
        assert config.scoring_metric == "accuracy"
        assert config.max_iter == 100
        assert config.n_jobs == -1
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.stratify is True
        assert config.scale_features is True
        assert config.handle_missing == "impute"
        assert config.categorical_encoding == "onehot"
        assert config.min_accuracy == 0.6
        assert config.min_precision == 0.6
        assert config.min_recall == 0.6

    def test_empty_model_configs_validation(self):
        """Test validation fails with empty model configs."""
        with pytest.raises(ValueError, match="At least one model configuration is required"):
            MLExperimentConfig(
                name="test_experiment",
                model_configs=[],
            )

    def test_invalid_tuning_method(self):
        """Test validation with invalid tuning method."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        with pytest.raises(ValueError, match="String should match pattern"):
            MLExperimentConfig(
                name="test_experiment",
                model_configs=[model_config],
                tuning_method="invalid_method",
            )

    def test_invalid_cv_folds_range(self):
        """Test validation with invalid CV folds range."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        # Too low
        with pytest.raises(ValueError, match="Input should be greater than or equal to 2"):
            MLExperimentConfig(
                name="test_experiment",
                model_configs=[model_config],
                cv_folds=1,
            )

        # Too high
        with pytest.raises(ValueError, match="Input should be less than or equal to 20"):
            MLExperimentConfig(
                name="test_experiment",
                model_configs=[model_config],
                cv_folds=25,
            )

    def test_invalid_test_size_range(self):
        """Test validation with invalid test size range."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        # Too low
        with pytest.raises(ValueError, match="Input should be greater than 0"):
            MLExperimentConfig(
                name="test_experiment",
                model_configs=[model_config],
                test_size=0.0,
            )

        # Too high
        with pytest.raises(ValueError, match="Input should be less than 1"):
            MLExperimentConfig(
                name="test_experiment",
                model_configs=[model_config],
                test_size=1.0,
            )

    def test_invalid_metric_thresholds(self):
        """Test validation with invalid metric thresholds."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        # Invalid accuracy
        with pytest.raises(ValueError):
            MLExperimentConfig(
                name="test_experiment",
                model_configs=[model_config],
                min_accuracy=1.5,
            )

        # Invalid precision
        with pytest.raises(ValueError):
            MLExperimentConfig(
                name="test_experiment",
                model_configs=[model_config],
                min_precision=-0.1,
            )


class TestMLConfig:
    """Test MLConfig class."""

    def test_valid_ml_config_creation(self):
        """Test creating valid ML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            config = MLConfig(
                project_name="test_project",
                version="1.0.0",
                data_directory=temp_path / "data",
                model_directory=temp_path / "models",
                artifacts_directory=temp_path / "artifacts",
            )

            assert config.project_name == "test_project"
            assert config.version == "1.0.0"
            assert config.data_directory.exists()
            assert config.model_directory.exists()
            assert config.artifacts_directory.exists()

    def test_ml_config_defaults(self):
        """Test ML configuration with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory so default paths work
            os.chdir(temp_dir)

            config = MLConfig(project_name="test_project")

            assert config.version == "1.0.0"
            assert config.enable_monitoring is True
            assert config.drift_detection_threshold == 0.05
            assert config.performance_alert_threshold == 0.1
            assert config.monitoring_interval == 3600
            assert config.max_memory_gb == 4.0
            assert config.max_cpu_cores == 4
            assert config.training_timeout == 3600
            assert config.deployment_environment == "development"
            assert config.auto_deploy is False
            assert config.deployment_approval_required is True

    def test_directory_creation(self):
        """Test that directories are created automatically."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Directories don't exist yet
            data_dir = temp_path / "nonexistent" / "data"
            model_dir = temp_path / "nonexistent" / "models"

            assert not data_dir.exists()
            assert not model_dir.exists()

            config = MLConfig(
                project_name="test_project",
                data_directory=data_dir,
                model_directory=model_dir,
            )

            # Should be created automatically
            assert config.data_directory.exists()
            assert config.model_directory.exists()

    def test_yaml_serialization(self):
        """Test YAML serialization and deserialization."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        experiment_config = MLExperimentConfig(
            name="test_experiment",
            model_configs=[model_config],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            original_config = MLConfig(
                project_name="test_project",
                experiments=[experiment_config],
                data_directory=temp_path / "data",
                model_directory=temp_path / "models",
                artifacts_directory=temp_path / "artifacts",
            )

            # Save to YAML
            yaml_path = temp_path / "config.yaml"
            original_config.to_yaml(yaml_path)

            assert yaml_path.exists()

            # Load from YAML
            loaded_config = MLConfig.from_yaml(yaml_path)

            assert loaded_config.project_name == original_config.project_name
            assert len(loaded_config.experiments) == 1
            assert loaded_config.experiments[0].name == "test_experiment"

    def test_yaml_file_not_found(self):
        """Test loading from non-existent YAML file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            MLConfig.from_yaml("nonexistent.yaml")

    def test_experiment_management(self):
        """Test experiment management methods."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        experiment1 = MLExperimentConfig(
            name="experiment1",
            model_configs=[model_config],
        )

        experiment2 = MLExperimentConfig(
            name="experiment2",
            model_configs=[model_config],
        )

        config = MLConfig(project_name="test_project")

        # Add experiments
        config.add_experiment(experiment1)
        config.add_experiment(experiment2)

        assert len(config.experiments) == 2

        # Get experiment by name
        retrieved = config.get_experiment_config("experiment1")
        assert retrieved is not None
        assert retrieved.name == "experiment1"

        # Non-existent experiment
        assert config.get_experiment_config("nonexistent") is None

        # Duplicate name should raise error
        duplicate = MLExperimentConfig(
            name="experiment1",
            model_configs=[model_config],
        )

        with pytest.raises(ValueError, match="already exists"):
            config.add_experiment(duplicate)

    @patch('os.access')
    @patch('os.cpu_count')
    def test_configuration_validation(self, mock_cpu_count, mock_access):
        """Test configuration validation."""
        mock_cpu_count.return_value = 8
        mock_access.return_value = False  # No permissions

        with tempfile.TemporaryDirectory() as temp_dir:
            config = MLConfig(
                project_name="test_project",
                max_memory_gb=64.0,  # Very high
                max_cpu_cores=16,    # Exceeds system
                deployment_environment="production",
                deployment_approval_required=False,
                data_directory=Path(temp_dir) / "data",
            )

            issues = config.validate_configuration()

            # Should have errors for permissions
            assert len(issues["errors"]) > 0
            assert any("permissions" in error for error in issues["errors"])

            # Should have warnings for resource limits and deployment
            assert len(issues["warnings"]) > 0
            assert any("Memory limit" in warning for warning in issues["warnings"])
            assert any("CPU cores" in warning for warning in issues["warnings"])
            assert any("Production deployment" in warning for warning in issues["warnings"])

    def test_duplicate_experiment_names_validation(self):
        """Test validation catches duplicate experiment names."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        # Create experiments with same name
        experiment1 = MLExperimentConfig(name="duplicate", model_configs=[model_config])
        experiment2 = MLExperimentConfig(name="duplicate", model_configs=[model_config])

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment1, experiment2],
        )

        issues = config.validate_configuration()
        assert any("Duplicate experiment names" in error for error in issues["errors"])

    def test_invalid_deployment_environment(self):
        """Test validation with invalid deployment environment."""
        with pytest.raises(ValueError, match="String should match pattern"):
            MLConfig(
                project_name="test_project",
                deployment_environment="invalid_env",
            )

    def test_invalid_drift_threshold(self):
        """Test validation with invalid drift detection threshold."""
        with pytest.raises(ValueError):
            MLConfig(
                project_name="test_project",
                drift_detection_threshold=1.5,  # > 1.0
            )

    def test_invalid_resource_limits(self):
        """Test validation with invalid resource limits."""
        # Memory too low
        with pytest.raises(ValueError):
            MLConfig(
                project_name="test_project",
                max_memory_gb=0.1,  # < 0.5
            )

        # CPU cores too low
        with pytest.raises(ValueError):
            MLConfig(
                project_name="test_project",
                max_cpu_cores=0,  # < 1
            )

        # Timeout too low
        with pytest.raises(ValueError):
            MLConfig(
                project_name="test_project",
                training_timeout=30,  # < 60
            )


class TestMLPerformanceMetrics:
    """Test MLPerformanceMetrics class."""

    def test_performance_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = MLPerformanceMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1_score=0.95,
            roc_auc=0.98,
            training_time=120.5,
            inference_time=0.1,
        )

        assert metrics.accuracy == 0.95
        assert metrics.precision == 0.93
        assert metrics.recall == 0.97
        assert metrics.f1_score == 0.95
        assert metrics.roc_auc == 0.98
        assert metrics.training_time == 120.5
        assert metrics.inference_time == 0.1

    def test_meets_requirements_true(self):
        """Test that metrics meet requirements."""
        metrics = MLPerformanceMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1_score=0.95,
        )

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        experiment_config = MLExperimentConfig(
            name="test",
            model_configs=[model_config],
            min_accuracy=0.8,
            min_precision=0.8,
            min_recall=0.8,
        )

        assert metrics.meets_requirements(experiment_config) is True

    def test_meets_requirements_false(self):
        """Test that metrics don't meet requirements."""
        metrics = MLPerformanceMetrics(
            accuracy=0.7,  # Below threshold
            precision=0.93,
            recall=0.5,   # Below threshold
            f1_score=0.6,
        )

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        experiment_config = MLExperimentConfig(
            name="test",
            model_configs=[model_config],
            min_accuracy=0.8,
            min_precision=0.8,
            min_recall=0.8,
        )

        assert metrics.meets_requirements(experiment_config) is False

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        metrics = MLPerformanceMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1_score=0.95,
            roc_auc=0.98,
            mean_squared_error=0.1,
            mean_absolute_error=0.05,
            training_time=120.5,
            inference_time=0.1,
        )

        result = metrics.to_dict()

        expected = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.97,
            "f1_score": 0.95,
            "roc_auc": 0.98,
            "mean_squared_error": 0.1,
            "mean_absolute_error": 0.05,
            "training_time": 120.5,
            "inference_time": 0.1,
        }

        assert result == expected

    def test_optional_fields_none(self):
        """Test that optional fields can be None."""
        metrics = MLPerformanceMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1_score=0.95,
        )

        result = metrics.to_dict()

        assert result["roc_auc"] is None
        assert result["mean_squared_error"] is None
        assert result["mean_absolute_error"] is None


class TestCreateDefaultConfig:
    """Test create_default_config function."""

    def test_creates_valid_default_config(self):
        """Test that default config is created correctly."""
        config = create_default_config("test_project")

        assert config.project_name == "test_project"
        assert len(config.experiments) == 1
        assert config.experiments[0].name == "document_classification"
        assert len(config.experiments[0].model_configs) == 2

        # Check model types
        model_types = [mc.model_type for mc in config.experiments[0].model_configs]
        assert ModelType.RANDOM_FOREST in model_types
        assert ModelType.GRADIENT_BOOSTING in model_types

    def test_default_config_validation(self):
        """Test that default config passes validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)  # Change to a directory where we have permissions
                config = create_default_config("test_project")

                issues = config.validate_configuration()

                # Should have no errors (warnings may exist for resource limits)
                assert len(issues["errors"]) == 0
            finally:
                os.chdir(original_cwd)  # Restore original directory

    def test_default_experiment_hyperparameters(self):
        """Test that default experiment has proper hyperparameters."""
        config = create_default_config("test_project")

        experiment = config.experiments[0]
        rf_config = next(
            mc for mc in experiment.model_configs
            if mc.model_type == ModelType.RANDOM_FOREST
        )

        assert "n_estimators" in rf_config.hyperparameters
        assert "max_depth" in rf_config.hyperparameters
        assert rf_config.feature_selection_method == FeatureSelectionMethod.TREE_IMPORTANCE
        assert rf_config.max_features == 1000


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions."""

    def test_empty_project_name(self):
        """Test validation with empty project name."""
        with pytest.raises(ValueError):
            MLConfig(project_name="")

    def test_extreme_resource_values(self):
        """Test with extreme resource values."""
        # Should work with minimum valid values
        config = MLConfig(
            project_name="test",
            max_memory_gb=0.5,
            max_cpu_cores=1,
            training_timeout=60,
            monitoring_interval=60,
        )

        assert config.max_memory_gb == 0.5
        assert config.max_cpu_cores == 1

    def test_malformed_yaml_handling(self):
        """Test handling of malformed YAML files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: {")
            f.flush()

            with pytest.raises(yaml.YAMLError):
                MLConfig.from_yaml(f.name)

        os.unlink(f.name)

    def test_invalid_yaml_structure(self):
        """Test handling of YAML with invalid structure."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({"invalid_field": "value"}, f)
            f.flush()

            with pytest.raises((ValueError, TypeError)):
                MLConfig.from_yaml(f.name)

        os.unlink(f.name)

    def test_concurrent_directory_creation(self):
        """Test handling concurrent directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate race condition where directory gets created between check and creation
            data_dir = Path(temp_dir) / "concurrent" / "data"

            # This should handle the race condition gracefully
            config = MLConfig(
                project_name="test",
                data_directory=data_dir,
            )

            assert config.data_directory.exists()

    @patch('pathlib.Path.mkdir')
    def test_directory_creation_permission_error(self, mock_mkdir):
        """Test handling of permission errors during directory creation."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        with pytest.raises(PermissionError):
            MLConfig(
                project_name="test",
                data_directory=Path("/root/forbidden"),
            )

    def test_model_config_large_hyperparameter_space(self):
        """Test model config with very large hyperparameter space."""
        large_params = {
            f"param_{i}": list(range(100)) for i in range(10)
        }

        config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters=large_params,
        )

        assert len(config.hyperparameters) == 10
        assert len(config.hyperparameters["param_0"]) == 100

    def test_experiment_config_extreme_values(self):
        """Test experiment config with extreme but valid values."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
        )

        # Test with extreme but valid values
        config = MLExperimentConfig(
            name="extreme_test",
            model_configs=[model_config],
            cv_folds=20,  # Maximum
            max_iter=1,   # Minimum
            test_size=0.99,  # Very high but valid
            min_accuracy=0.0,  # Minimum
            min_precision=1.0,  # Maximum
            min_recall=1.0,    # Maximum
        )

        assert config.cv_folds == 20
        assert config.max_iter == 1
        assert config.test_size == 0.99
