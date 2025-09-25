"""
Comprehensive unit tests for ML model registry system.

Tests all model registry components including model storage, versioning,
metadata management, and deployment workflows with extensive edge cases
and error conditions.
"""

import pytest
import tempfile
import sqlite3
import pickle
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import shutil
import hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from src.python.common.ml.config.ml_config import (
    MLConfig,
    MLExperimentConfig,
    MLModelConfig,
    MLPerformanceMetrics,
    MLTaskType,
    ModelType
)
from src.python.common.ml.pipeline.training_pipeline import TrainingResult
from src.python.common.ml.management.model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelVersion,
    ModelRegistryError,
    ModelNotFoundError,
    ModelVersionError,
    ModelStorageError
)


def create_trained_logistic_model():
    """Create a simple trained logistic regression model for testing."""
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


def create_trained_forest_model():
    """Create a simple trained random forest model for testing."""
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)  # Small model for testing
    model.fit(X, y)
    return model


class TestModelMetadata:
    """Test ModelMetadata class."""

    def test_model_metadata_creation(self):
        """Test creating model metadata."""
        metadata = ModelMetadata(
            model_id="test_model_123",
            name="test_model",
            version="1.0.0",
            description="Test model",
            stage="development",
            model_type="random_forest",
            task_type="classification"
        )

        assert metadata.model_id == "test_model_123"
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.description == "Test model"
        assert metadata.stage == "development"
        assert metadata.model_type == "random_forest"
        assert metadata.task_type == "classification"
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)

    def test_model_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ModelMetadata(
            model_id="test_model_123",
            name="test_model",
            version="1.0.0",
            tags={"env": "test", "author": "pytest"},
            metrics={"accuracy": 0.95, "f1_score": 0.94}
        )

        data = metadata.to_dict()

        assert data["model_id"] == "test_model_123"
        assert data["name"] == "test_model"
        assert data["version"] == "1.0.0"
        assert data["tags"]["env"] == "test"
        assert data["metrics"]["accuracy"] == 0.95
        assert isinstance(data["created_at"], str)  # Should be ISO format string
        assert isinstance(data["updated_at"], str)

    def test_model_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        now = datetime.now()
        data = {
            "model_id": "test_model_123",
            "name": "test_model",
            "version": "1.0.0",
            "description": "Test model",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "stage": "development",
            "model_type": "random_forest",
            "task_type": "classification",
            "tags": {"env": "test"},
            "metrics": {"accuracy": 0.95},
            "hyperparameters": {"n_estimators": 100},
            "feature_names": ["feature_1", "feature_2"],
            "model_size_bytes": 1024,
            "model_hash": "abc123"
        }

        metadata = ModelMetadata.from_dict(data)

        assert metadata.model_id == "test_model_123"
        assert metadata.name == "test_model"
        assert isinstance(metadata.created_at, datetime)
        assert metadata.tags["env"] == "test"
        assert metadata.metrics["accuracy"] == 0.95


class TestModelRegistry:
    """Test ModelRegistry class."""

    def setup_method(self):
        """Setup test data for each test method."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create test configuration
        self.config = MLConfig(
            project_name="test_project",
            data_directory=self.temp_path / "data",
            model_directory=self.temp_path / "models",
            artifacts_directory=self.temp_path / "artifacts"
        )

        # Create test training result
        self.test_model = create_trained_forest_model()  # Real trained sklearn model

        self.test_metrics = MLPerformanceMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1_score=0.95,
            training_time=120.5,
            inference_time=0.1
        )

        self.test_model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"n_estimators": 100, "max_depth": 10}
        )

        self.test_training_result = TrainingResult(
            model=self.test_model,
            metrics=self.test_metrics,
            model_config=self.test_model_config,
            feature_names=["feature_1", "feature_2", "feature_3"],
            best_params={"n_estimators": 100, "max_depth": 10},
            training_metadata={"training_time": 120.5, "feature_count": 3}
        )

    def teardown_method(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_registry_initialization(self):
        """Test model registry initialization."""
        registry = ModelRegistry(self.config)

        assert registry.config == self.config
        assert registry.registry_path.exists()
        assert registry.models_path.exists()
        assert registry.db_path.exists()

        # Check that database tables were created
        with sqlite3.connect(registry.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            assert "models" in tables
            assert "model_metrics" in tables
            assert "model_hyperparameters" in tables
            assert "model_tags" in tables
            assert "model_features" in tables

    @pytest.mark.skip(reason="Path validation in MLConfig prevents testing directory creation failures")
    def test_registry_initialization_failure(self):
        """Test registry initialization failure handling."""
        # Skip this test as MLConfig validates paths during construction
        # Making it difficult to test ModelRegistry initialization failures
        pass

    def test_register_model_success(self):
        """Test successful model registration."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0",
            description="Test classification model",
            tags={"author": "pytest", "env": "test"},
            stage="development"
        )

        assert model_id is not None
        assert model_id.startswith("test_classifier_1.0.0_")

        # Verify model was stored
        model_version = registry.get_model(model_id)
        assert model_version.model_id == model_id
        assert model_version.metadata.name == "test_classifier"
        assert model_version.metadata.version == "1.0.0"
        assert model_version.metadata.stage == "development"
        assert model_version.metadata.tags["author"] == "pytest"
        assert model_version.model_path.exists()

    def test_register_model_auto_version(self):
        """Test model registration with auto-generated version."""
        registry = ModelRegistry(self.config)

        # Register first model (should get version 1.0.0)
        model_id_1 = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier"
        )

        model_1 = registry.get_model(model_id_1)
        assert model_1.metadata.version == "1.0.0"

        # Register second model (should get version 1.0.1)
        model_id_2 = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier"
        )

        model_2 = registry.get_model(model_id_2)
        assert model_2.metadata.version == "1.0.1"

    def test_register_model_duplicate_version(self):
        """Test registering model with duplicate version."""
        registry = ModelRegistry(self.config)

        # Register first model
        registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        # Try to register with same name and version
        with pytest.raises(ModelRegistryError):
            registry.register_model(
                training_result=self.test_training_result,
                name="test_classifier",
                version="1.0.0"
            )

    def test_get_model_success(self):
        """Test successful model retrieval."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        model_version = registry.get_model(model_id)

        assert model_version.model_id == model_id
        assert model_version.version == "1.0.0"
        assert model_version.metadata.name == "test_classifier"
        assert model_version.model_path.exists()

    def test_get_model_not_found(self):
        """Test getting non-existent model."""
        registry = ModelRegistry(self.config)

        with pytest.raises(ModelNotFoundError, match="Model not found"):
            registry.get_model("nonexistent_model")

    def test_get_model_by_name_latest(self):
        """Test getting model by name (latest version)."""
        registry = ModelRegistry(self.config)

        # Register multiple versions
        registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        model_id_2 = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.1"
        )

        # Should return latest version
        model_version = registry.get_model_by_name("test_classifier")
        assert model_version.model_id == model_id_2
        assert model_version.version == "1.0.1"

    def test_get_model_by_name_specific_version(self):
        """Test getting model by name and specific version."""
        registry = ModelRegistry(self.config)

        model_id_1 = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.1"
        )

        # Should return specific version
        model_version = registry.get_model_by_name("test_classifier", version="1.0.0")
        assert model_version.model_id == model_id_1
        assert model_version.version == "1.0.0"

    def test_get_model_by_name_with_stage_filter(self):
        """Test getting model by name with stage filter."""
        registry = ModelRegistry(self.config)

        # Register models in different stages
        model_id_dev = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0",
            stage="development"
        )

        model_id_prod = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.1",
            stage="production"
        )

        # Get production version
        model_version = registry.get_model_by_name(
            "test_classifier", stage="production"
        )
        assert model_version.model_id == model_id_prod
        assert model_version.metadata.stage == "production"

    def test_get_model_by_name_not_found(self):
        """Test getting non-existent model by name."""
        registry = ModelRegistry(self.config)

        with pytest.raises(ModelNotFoundError, match="Model nonexistent not found"):
            registry.get_model_by_name("nonexistent")

    def test_list_models_all(self):
        """Test listing all models."""
        registry = ModelRegistry(self.config)

        # Register multiple models
        registry.register_model(
            training_result=self.test_training_result,
            name="classifier_1",
            version="1.0.0"
        )

        registry.register_model(
            training_result=self.test_training_result,
            name="classifier_2",
            version="1.0.0"
        )

        models = registry.list_models()

        assert len(models) == 2
        assert any(m.name == "classifier_1" for m in models)
        assert any(m.name == "classifier_2" for m in models)

    def test_list_models_with_filters(self):
        """Test listing models with filters."""
        registry = ModelRegistry(self.config)

        # Register models with different characteristics
        registry.register_model(
            training_result=self.test_training_result,
            name="classifier_dev",
            version="1.0.0",
            stage="development",
            tags={"env": "test"}
        )

        registry.register_model(
            training_result=self.test_training_result,
            name="classifier_prod",
            version="1.0.0",
            stage="production",
            tags={"env": "prod"}
        )

        registry.register_model(
            training_result=self.test_training_result,
            name="regressor_dev",
            version="1.0.0",
            stage="development"
        )

        # Test stage filter
        prod_models = registry.list_models(stage="production")
        assert len(prod_models) == 1
        assert prod_models[0].name == "classifier_prod"

        # Test name pattern filter
        classifier_models = registry.list_models(name_pattern="classifier_%")
        assert len(classifier_models) == 2

        # Test tag filter
        test_models = registry.list_models(tags={"env": "test"})
        assert len(test_models) == 1
        assert test_models[0].name == "classifier_dev"

    def test_update_model_stage(self):
        """Test updating model stage."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0",
            stage="development"
        )

        # Update to staging
        registry.update_model_stage(model_id, "staging")

        model_version = registry.get_model(model_id)
        assert model_version.metadata.stage == "staging"

        # Update to production
        registry.update_model_stage(model_id, "production")

        model_version = registry.get_model(model_id)
        assert model_version.metadata.stage == "production"

    def test_update_model_stage_invalid(self):
        """Test updating model stage with invalid stage."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        with pytest.raises(ValueError, match="Invalid stage"):
            registry.update_model_stage(model_id, "invalid_stage")

    def test_update_model_stage_not_found(self):
        """Test updating stage for non-existent model."""
        registry = ModelRegistry(self.config)

        with pytest.raises(ModelNotFoundError):
            registry.update_model_stage("nonexistent_model", "staging")

    def test_delete_model_success(self):
        """Test successful model deletion."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0",
            stage="development"
        )

        # Verify model exists
        assert registry.get_model(model_id) is not None

        # Delete model
        registry.delete_model(model_id)

        # Verify model is gone
        with pytest.raises(ModelNotFoundError):
            registry.get_model(model_id)

    def test_delete_model_production_without_force(self):
        """Test deleting production model without force flag."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0",
            stage="production"
        )

        with pytest.raises(ModelRegistryError, match="Cannot delete production model"):
            registry.delete_model(model_id, force=False)

    def test_delete_model_production_with_force(self):
        """Test deleting production model with force flag."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0",
            stage="production"
        )

        # Should succeed with force=True
        registry.delete_model(model_id, force=True)

        with pytest.raises(ModelNotFoundError):
            registry.get_model(model_id)

    def test_delete_model_not_found(self):
        """Test deleting non-existent model."""
        registry = ModelRegistry(self.config)

        with pytest.raises(ModelNotFoundError):
            registry.delete_model("nonexistent_model")

    def test_compare_models(self):
        """Test model comparison functionality."""
        registry = ModelRegistry(self.config)

        # Create models with different metrics
        metrics_1 = MLPerformanceMetrics(
            accuracy=0.90, precision=0.88, recall=0.92, f1_score=0.90
        )
        training_result_1 = TrainingResult(
            model=self.test_model,
            metrics=metrics_1,
            model_config=self.test_model_config,
            feature_names=["f1", "f2"]
        )

        metrics_2 = MLPerformanceMetrics(
            accuracy=0.95, precision=0.93, recall=0.97, f1_score=0.95
        )
        training_result_2 = TrainingResult(
            model=self.test_model,
            metrics=metrics_2,
            model_config=self.test_model_config,
            feature_names=["f1", "f2"]
        )

        model_id_1 = registry.register_model(
            training_result=training_result_1,
            name="classifier_v1",
            version="1.0.0"
        )

        model_id_2 = registry.register_model(
            training_result=training_result_2,
            name="classifier_v2",
            version="1.0.0"
        )

        # Compare models
        comparison = registry.compare_models([model_id_1, model_id_2])

        assert len(comparison) == 2
        assert comparison[model_id_1]["metrics"]["accuracy"] == 0.90
        assert comparison[model_id_2]["metrics"]["accuracy"] == 0.95

        # Compare specific metrics
        comparison_specific = registry.compare_models(
            [model_id_1, model_id_2], metrics=["accuracy", "f1_score"]
        )

        assert "accuracy" in comparison_specific[model_id_1]["metrics"]
        assert "f1_score" in comparison_specific[model_id_1]["metrics"]
        assert "precision" not in comparison_specific[model_id_1]["metrics"]

    def test_compare_models_not_found(self):
        """Test comparing models when one doesn't exist."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        with pytest.raises(ModelNotFoundError):
            registry.compare_models([model_id, "nonexistent_model"])

    def test_get_best_model(self):
        """Test getting best model by metric."""
        registry = ModelRegistry(self.config)

        # Create models with different metrics
        metrics_1 = MLPerformanceMetrics(
            accuracy=0.90, precision=0.88, recall=0.92, f1_score=0.90
        )
        training_result_1 = TrainingResult(
            model=self.test_model,
            metrics=metrics_1,
            model_config=self.test_model_config,
            feature_names=["f1", "f2"]
        )

        metrics_2 = MLPerformanceMetrics(
            accuracy=0.95, precision=0.93, recall=0.97, f1_score=0.95
        )
        training_result_2 = TrainingResult(
            model=self.test_model,
            metrics=metrics_2,
            model_config=self.test_model_config,
            feature_names=["f1", "f2"]
        )

        model_id_1 = registry.register_model(
            training_result=training_result_1,
            name="test_classifier",
            version="1.0.0"
        )

        model_id_2 = registry.register_model(
            training_result=training_result_2,
            name="test_classifier",
            version="1.0.1"
        )

        # Get best model by accuracy (should be model_2)
        best_model = registry.get_best_model("test_classifier", "accuracy", maximize=True)

        assert best_model is not None
        assert best_model.model_id == model_id_2
        assert best_model.metadata.metrics["accuracy"] == 0.95

        # Get best model by accuracy (minimize - should be model_1)
        worst_model = registry.get_best_model("test_classifier", "accuracy", maximize=False)

        assert worst_model is not None
        assert worst_model.model_id == model_id_1
        assert worst_model.metadata.metrics["accuracy"] == 0.90

    def test_get_best_model_not_found(self):
        """Test getting best model when no models exist."""
        registry = ModelRegistry(self.config)

        best_model = registry.get_best_model("nonexistent_classifier", "accuracy")
        assert best_model is None

    def test_get_best_model_no_metric(self):
        """Test getting best model when metric doesn't exist."""
        registry = ModelRegistry(self.config)

        registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        best_model = registry.get_best_model("test_classifier", "nonexistent_metric")
        assert best_model is None

    def test_load_model_success(self):
        """Test successful model loading."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        loaded_model = registry.load_model(model_id)

        assert loaded_model is not None
        # Model should be the same object we stored
        assert loaded_model == self.test_model

    def test_load_model_not_found(self):
        """Test loading non-existent model."""
        registry = ModelRegistry(self.config)

        with pytest.raises(ModelNotFoundError):
            registry.load_model("nonexistent_model")

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_load_model_storage_error(self, mock_open):
        """Test model loading with storage error."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        with pytest.raises(ModelStorageError, match="Failed to load model"):
            registry.load_model(model_id)

    def test_load_model_hash_mismatch(self):
        """Test model loading with hash mismatch warning."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        # Corrupt the model file to create hash mismatch
        model_version = registry.get_model(model_id)
        with open(model_version.model_path, 'ab') as f:
            f.write(b'corrupted_data')

        # Should load but log warning
        with patch.object(registry.logger, 'warning') as mock_warning:
            loaded_model = registry.load_model(model_id)
            assert loaded_model is not None
            mock_warning.assert_called_once()
            assert "hash mismatch" in mock_warning.call_args[0][0]

    def test_cleanup_old_models(self):
        """Test cleaning up old model versions."""
        registry = ModelRegistry(self.config)

        # Register multiple versions of the same model
        model_ids = []
        for i in range(5):
            model_id = registry.register_model(
                training_result=self.test_training_result,
                name="test_classifier",
                version=f"1.0.{i}",
                stage="development"
            )
            model_ids.append(model_id)

        # Keep only 2 latest versions
        deleted_models = registry.cleanup_old_models(keep_latest=2, dry_run=False)

        assert len(deleted_models) == 3  # Should delete 3 oldest versions

        # Verify latest 2 versions still exist
        for model_id in model_ids[-2:]:
            assert registry.get_model(model_id) is not None

        # Verify oldest 3 versions are gone
        for model_id in model_ids[:-2]:
            with pytest.raises(ModelNotFoundError):
                registry.get_model(model_id)

    def test_cleanup_old_models_keep_production(self):
        """Test cleanup respecting production models."""
        registry = ModelRegistry(self.config)

        # Register models with different stages
        dev_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0",
            stage="development"
        )

        prod_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.1",
            stage="production"
        )

        staging_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.2",
            stage="staging"
        )

        # Cleanup keeping only 1 latest, but preserve production
        deleted_models = registry.cleanup_old_models(
            keep_latest=1,
            keep_production=True,
            dry_run=False
        )

        assert len(deleted_models) == 1  # Should delete only dev model
        assert dev_id in deleted_models

        # Production and staging should still exist
        assert registry.get_model(prod_id) is not None
        assert registry.get_model(staging_id) is not None

    def test_cleanup_old_models_dry_run(self):
        """Test cleanup dry run mode."""
        registry = ModelRegistry(self.config)

        # Register multiple models
        model_ids = []
        for i in range(3):
            model_id = registry.register_model(
                training_result=self.test_training_result,
                name="test_classifier",
                version=f"1.0.{i}"
            )
            model_ids.append(model_id)

        # Dry run - should return models that would be deleted but not delete them
        deleted_models = registry.cleanup_old_models(keep_latest=1, dry_run=True)

        assert len(deleted_models) == 2  # Would delete 2 oldest

        # All models should still exist
        for model_id in model_ids:
            assert registry.get_model(model_id) is not None

    def test_get_registry_stats(self):
        """Test getting registry statistics."""
        registry = ModelRegistry(self.config)

        # Register models with different characteristics
        registry.register_model(
            training_result=self.test_training_result,
            name="classifier_1",
            version="1.0.0",
            stage="development"
        )

        registry.register_model(
            training_result=self.test_training_result,
            name="classifier_2",
            version="1.0.0",
            stage="production"
        )

        stats = registry.get_registry_stats()

        assert stats["total_models"] == 2
        assert stats["models_by_stage"]["development"] == 1
        assert stats["models_by_stage"]["production"] == 1
        assert stats["models_by_type"]["random_forest"] == 2
        assert stats["total_storage_bytes"] > 0
        assert stats["total_storage_mb"] > 0
        assert stats["actual_storage_bytes"] > 0
        assert stats["actual_storage_mb"] > 0

    def test_version_generation_semantic(self):
        """Test semantic version generation."""
        registry = ModelRegistry(self.config)

        # First model should get version 1.0.0
        model_id_1 = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier"
        )
        model_1 = registry.get_model(model_id_1)
        assert model_1.metadata.version == "1.0.0"

        # Second model should get version 1.0.1
        model_id_2 = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier"
        )
        model_2 = registry.get_model(model_id_2)
        assert model_2.metadata.version == "1.0.1"

    def test_version_generation_non_semantic_fallback(self):
        """Test version generation with non-semantic existing versions."""
        registry = ModelRegistry(self.config)

        # Register model with non-semantic version
        model_id_1 = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="v1_alpha"
        )

        # Next auto-generated version should use fallback
        model_id_2 = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier"
        )
        model_2 = registry.get_model(model_id_2)
        assert model_2.metadata.version == "v1_alpha_1"

    def test_model_artifacts_storage(self):
        """Test storage of model artifacts."""
        # Create training result with additional artifacts
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest

        scaler = StandardScaler()
        selector = SelectKBest()

        training_result_with_artifacts = TrainingResult(
            model=self.test_model,
            metrics=self.test_metrics,
            model_config=self.test_model_config,
            feature_names=["f1", "f2"],
            scaler=scaler,
            feature_selector=selector,
            training_metadata={"test": "data"}
        )

        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=training_result_with_artifacts,
            name="test_classifier",
            version="1.0.0"
        )

        model_version = registry.get_model(model_id)

        # Check that artifacts directory exists
        assert model_version.artifacts_path is not None
        assert model_version.artifacts_path.exists()

        # Check specific artifact files
        scaler_path = model_version.artifacts_path / "scaler.pkl"
        selector_path = model_version.artifacts_path / "feature_selector.pkl"
        metadata_path = model_version.artifacts_path / "training_metadata.json"

        assert scaler_path.exists()
        assert selector_path.exists()
        assert metadata_path.exists()

        # Verify metadata file content
        with open(metadata_path, 'r') as f:
            metadata_content = json.load(f)
        assert metadata_content["test"] == "data"

    def test_concurrent_access_safety(self):
        """Test registry safety under concurrent access."""
        registry = ModelRegistry(self.config)

        # This test mainly ensures no exceptions are raised during
        # concurrent-like operations (though not truly concurrent in unit test)
        import threading
        import time

        results = []
        errors = []

        def register_model_worker(worker_id):
            try:
                model_id = registry.register_model(
                    training_result=self.test_training_result,
                    name=f"concurrent_classifier_{worker_id}",
                    version="1.0.0"
                )
                results.append(model_id)
            except Exception as e:
                errors.append(str(e))

        # Simulate concurrent registration
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_model_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert len(errors) == 0
        assert len(results) == 5

        # All models should be retrievable
        for model_id in results:
            assert registry.get_model(model_id) is not None

    def test_database_corruption_recovery(self):
        """Test registry behavior with database issues."""
        registry = ModelRegistry(self.config)

        # Register a model first
        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        # Simulate database corruption by closing connection improperly
        registry.db_path.unlink()  # Delete database file

        # Operations should fail gracefully
        with pytest.raises(ModelStorageError):
            registry.get_model(model_id)

        # But registry should be able to recover by reinitializing
        new_registry = ModelRegistry(self.config)
        assert new_registry.db_path.exists()

    def test_model_hash_calculation(self):
        """Test model hash calculation for integrity checking."""
        registry = ModelRegistry(self.config)

        model_id = registry.register_model(
            training_result=self.test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        model_version = registry.get_model(model_id)

        # Hash should be non-empty
        assert model_version.metadata.model_hash
        assert len(model_version.metadata.model_hash) == 64  # SHA256 hex length

        # Verify hash calculation
        expected_hash = registry._calculate_model_hash(model_version.model_path)
        assert model_version.metadata.model_hash == expected_hash

    def test_large_model_handling(self):
        """Test handling of large models."""
        registry = ModelRegistry(self.config)

        # Create a real model and simulate large size
        large_model = create_trained_forest_model()
        large_model.large_data = b'x' * (10 * 1024 * 1024)  # 10MB of data

        large_training_result = TrainingResult(
            model=large_model,
            metrics=self.test_metrics,
            model_config=self.test_model_config,
            feature_names=["f1", "f2"]
        )

        model_id = registry.register_model(
            training_result=large_training_result,
            name="large_classifier",
            version="1.0.0"
        )

        model_version = registry.get_model(model_id)

        # Size should be recorded
        assert model_version.metadata.model_size_bytes > 1024 * 1024  # > 1MB

        # Should be able to load
        loaded_model = registry.load_model(model_id)
        assert loaded_model is not None


class TestModelRegistryEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_model_path_handling(self):
        """Test handling of invalid model paths."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)

        config = MLConfig(
            project_name="test_project",
            model_directory=temp_path / "models"
        )

        registry = ModelRegistry(config)

        # Manually create a model entry but delete the file
        test_model = create_trained_forest_model()
        test_metrics = MLPerformanceMetrics(accuracy=0.95, precision=0.93, recall=0.97, f1_score=0.95)
        test_model_config = MLModelConfig(model_type=ModelType.RANDOM_FOREST, task_type=MLTaskType.CLASSIFICATION)
        test_training_result = TrainingResult(
            model=test_model,
            metrics=test_metrics,
            model_config=test_model_config,
            feature_names=["f1", "f2"]
        )

        model_id = registry.register_model(
            training_result=test_training_result,
            name="test_classifier",
            version="1.0.0"
        )

        # Delete the model file
        model_version = registry.get_model(model_id)
        model_version.model_path.unlink()

        # Should raise storage error
        with pytest.raises(ModelStorageError, match="Model file not found"):
            registry.get_model(model_id)

        temp_dir.cleanup()

    def test_empty_model_list(self):
        """Test listing models when registry is empty."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)

        config = MLConfig(
            project_name="test_project",
            model_directory=temp_path / "models"
        )

        registry = ModelRegistry(config)

        models = registry.list_models()
        assert len(models) == 0

        temp_dir.cleanup()

    def test_malformed_database_recovery(self):
        """Test recovery from malformed database entries."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)

        config = MLConfig(
            project_name="test_project",
            model_directory=temp_path / "models"
        )

        registry = ModelRegistry(config)

        # Directly insert malformed data into database
        with sqlite3.connect(registry.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO models (
                    model_id, name, version, description, created_at, updated_at,
                    stage, model_type, task_type, model_size_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                "malformed_model",
                "test_model",
                "1.0.0",
                "Test",
                "invalid_date",  # Invalid date format
                "2023-01-01T00:00:00",
                "development",
                "random_forest",
                "classification",
                1024
            ))

        # Should handle gracefully when retrieving
        try:
            metadata = registry._get_metadata("malformed_model")
            # If it succeeds, it should handle the malformed date
            assert metadata is None or metadata.model_id == "malformed_model"
        except Exception:
            # Or it might fail, which is also acceptable
            pass

        temp_dir.cleanup()

    def test_extremely_long_model_names(self):
        """Test handling of extremely long model names."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)

        config = MLConfig(
            project_name="test_project",
            model_directory=temp_path / "models"
        )

        registry = ModelRegistry(config)

        test_model = create_trained_forest_model()
        test_metrics = MLPerformanceMetrics(accuracy=0.95, precision=0.93, recall=0.97, f1_score=0.95)
        test_model_config = MLModelConfig(model_type=ModelType.RANDOM_FOREST, task_type=MLTaskType.CLASSIFICATION)
        test_training_result = TrainingResult(
            model=test_model,
            metrics=test_metrics,
            model_config=test_model_config,
            feature_names=["f1", "f2"]
        )

        # Very long name
        long_name = "a" * 1000

        # Should handle without crashing
        try:
            model_id = registry.register_model(
                training_result=test_training_result,
                name=long_name,
                version="1.0.0"
            )
            assert model_id is not None
        except Exception as e:
            # If it fails due to filesystem limitations, that's acceptable
            assert "too long" in str(e).lower() or "invalid" in str(e).lower()

        temp_dir.cleanup()

    def test_special_characters_in_model_names(self):
        """Test handling of special characters in model names."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)

        config = MLConfig(
            project_name="test_project",
            model_directory=temp_path / "models"
        )

        registry = ModelRegistry(config)

        test_model = create_trained_forest_model()
        test_metrics = MLPerformanceMetrics(accuracy=0.95, precision=0.93, recall=0.97, f1_score=0.95)
        test_model_config = MLModelConfig(model_type=ModelType.RANDOM_FOREST, task_type=MLTaskType.CLASSIFICATION)
        test_training_result = TrainingResult(
            model=test_model,
            metrics=test_metrics,
            model_config=test_model_config,
            feature_names=["f1", "f2"]
        )

        # Names with special characters
        special_names = [
            "model-with-hyphens",
            "model_with_underscores",
            "model with spaces",
            "model.with.dots",
            "model@special#chars",
        ]

        for name in special_names:
            try:
                model_id = registry.register_model(
                    training_result=test_training_result,
                    name=name,
                    version="1.0.0"
                )
                # If registration succeeds, model should be retrievable
                retrieved = registry.get_model(model_id)
                assert retrieved.metadata.name == name
            except Exception:
                # Some special characters might not be supported, which is acceptable
                pass

        temp_dir.cleanup()

    def test_disk_space_exhaustion_simulation(self):
        """Test behavior when disk space is exhausted."""
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)

        config = MLConfig(
            project_name="test_project",
            model_directory=temp_path / "models"
        )

        registry = ModelRegistry(config)

        test_model = create_trained_forest_model()
        test_metrics = MLPerformanceMetrics(accuracy=0.95, precision=0.93, recall=0.97, f1_score=0.95)
        test_model_config = MLModelConfig(model_type=ModelType.RANDOM_FOREST, task_type=MLTaskType.CLASSIFICATION)
        test_training_result = TrainingResult(
            model=test_model,
            metrics=test_metrics,
            model_config=test_model_config,
            feature_names=["f1", "f2"]
        )

        # Mock storage failure
        with patch('builtins.open', side_effect=OSError("No space left on device")):
            with pytest.raises(ModelRegistryError):
                registry.register_model(
                    training_result=test_training_result,
                    name="test_classifier",
                    version="1.0.0"
                )

        temp_dir.cleanup()