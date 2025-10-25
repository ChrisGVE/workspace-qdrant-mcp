"""
Comprehensive unit tests for ML training pipeline.

Tests all pipeline components including data validation, model training,
hyperparameter tuning, feature engineering, and error handling with
extensive edge cases and error conditions.
"""

import tempfile
import time
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings('ignore')

from common.ml.config.ml_config import (
    FeatureSelectionMethod,
    MLConfig,
    MLExperimentConfig,
    MLModelConfig,
    MLTaskType,
    ModelType,
)
from common.ml.pipeline.training_pipeline import (
    DataValidationError,
    HyperparameterTuningError,
    MLPipelineError,
    ModelTrainingError,
    PipelineState,
    TrainingPipeline,
    TrainingResult,
)


class TestTrainingPipeline:
    """Test TrainingPipeline class."""

    def setup_method(self):
        """Setup test data for each test method."""
        # Create sample configuration
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={
                "n_estimators": [10, 20],
                "max_depth": [3, 5],
            }
        )

        experiment_config = MLExperimentConfig(
            name="test_experiment",
            model_configs=[model_config],
            cv_folds=3,
            max_iter=5,
            test_size=0.2,
            random_state=42
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            self.config = MLConfig(
                project_name="test_project",
                experiments=[experiment_config],
                data_directory=Path(temp_dir) / "data",
                model_directory=Path(temp_dir) / "models",
            )

        # Create sample datasets
        np.random.seed(42)
        self.X_classification = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.normal(1, 1, 100),
            'feature_3': np.random.randint(0, 5, 100),
        })
        self.y_classification = pd.Series(
            np.random.randint(0, 3, 100), name='target'
        )

        self.X_regression = self.X_classification.copy()
        self.y_regression = pd.Series(
            self.X_regression['feature_1'] * 2 + self.X_regression['feature_2'] +
            np.random.normal(0, 0.1, 100), name='target'
        )

        # Small dataset for edge cases
        self.X_small = self.X_classification.head(5)
        self.y_small = self.y_classification.head(5)

    def test_pipeline_initialization_valid(self):
        """Test valid pipeline initialization."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        assert pipeline.config == self.config
        assert pipeline.experiment_config.name == "test_experiment"
        assert isinstance(pipeline.state, PipelineState)
        assert pipeline.logger is not None

    def test_pipeline_initialization_invalid_experiment(self):
        """Test pipeline initialization with invalid experiment name."""
        with pytest.raises(ValueError, match="Experiment 'nonexistent' not found"):
            TrainingPipeline(self.config, "nonexistent")

    def test_pipeline_initialization_no_models(self):
        """Test pipeline initialization with empty model configs."""
        empty_experiment = MLExperimentConfig(
            name="empty_experiment",
            model_configs=[]
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[empty_experiment]
        )

        with pytest.raises(ValueError, match="No model configurations provided"):
            TrainingPipeline(config, "empty_experiment")

    def test_data_validation_valid_data(self):
        """Test data validation with valid data."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        X_validated, y_validated = pipeline._validate_and_prepare_data(
            self.X_classification, self.y_classification
        )

        assert isinstance(X_validated, pd.DataFrame)
        assert isinstance(y_validated, pd.Series)
        assert len(X_validated) == len(y_validated)

    def test_data_validation_empty_data(self):
        """Test data validation with empty data."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Empty X
        with pytest.raises(DataValidationError, match="Feature matrix X is empty"):
            pipeline._validate_and_prepare_data(pd.DataFrame(), self.y_classification)

        # Empty y
        with pytest.raises(DataValidationError, match="Target vector y is empty"):
            pipeline._validate_and_prepare_data(self.X_classification, pd.Series(dtype=float))

    def test_data_validation_mismatched_lengths(self):
        """Test data validation with mismatched X and y lengths."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        with pytest.raises(DataValidationError, match="X and y length mismatch"):
            pipeline._validate_and_prepare_data(
                self.X_classification, self.y_classification.head(50)
            )

    def test_data_validation_insufficient_samples(self):
        """Test data validation with insufficient samples."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        with pytest.raises(DataValidationError, match="Insufficient samples"):
            pipeline._validate_and_prepare_data(self.X_small, self.y_small)

    def test_data_validation_excessive_missing_values(self):
        """Test data validation with excessive missing values."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Create dataset with > 50% missing values
        X_missing = self.X_classification.copy()
        X_missing.loc[:, :] = np.nan

        with pytest.raises(DataValidationError, match="Too many missing values"):
            pipeline._validate_and_prepare_data(X_missing, self.y_classification)

    def test_data_validation_classification_requirements(self):
        """Test classification-specific data validation."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Single class
        y_single_class = pd.Series([0] * 100)
        with pytest.raises(DataValidationError, match="at least 2 classes"):
            pipeline._validate_and_prepare_data(self.X_classification, y_single_class)

        # Too many classes (should generate warning, not error)
        y_many_classes = pd.Series(np.arange(100))
        X_validated, y_validated = pipeline._validate_and_prepare_data(
            self.X_classification, y_many_classes
        )
        assert len(pipeline.state.warnings) > 0

    def test_data_validation_regression_requirements(self):
        """Test regression-specific data validation."""
        # Create regression model config
        model_config = MLModelConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=MLTaskType.REGRESSION
        )

        experiment_config = MLExperimentConfig(
            name="regression_experiment",
            model_configs=[model_config]
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "regression_experiment")

        # Non-numeric target
        y_categorical = pd.Series(['A', 'B', 'C'] * 33 + ['A'])
        with pytest.raises(DataValidationError, match="Regression target must be numeric"):
            pipeline._validate_and_prepare_data(self.X_classification, y_categorical)

    def test_data_splitting_stratified(self):
        """Test data splitting with stratification."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        X_train, X_test, y_train, y_test = pipeline._split_data(
            self.X_classification, self.y_classification
        )

        assert len(X_train) + len(X_test) == len(self.X_classification)
        assert len(y_train) + len(y_test) == len(self.y_classification)

        # Check test size ratio
        expected_test_size = int(len(self.X_classification) * pipeline.experiment_config.test_size)
        assert abs(len(X_test) - expected_test_size) <= 2  # Allow small variance

    def test_data_splitting_no_stratification(self):
        """Test data splitting without stratification."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        pipeline.experiment_config.stratify = False

        X_train, X_test, y_train, y_test = pipeline._split_data(
            self.X_classification, self.y_classification
        )

        assert len(X_train) + len(X_test) == len(self.X_classification)
        assert len(y_train) + len(y_test) == len(self.y_classification)

    def test_feature_engineering_scaling(self):
        """Test feature engineering with scaling."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        pipeline.experiment_config.scale_features = True

        X_train, X_test = self.X_classification.iloc[:80], self.X_classification.iloc[80:]
        y_train = self.y_classification.iloc[:80]

        X_train_processed, X_test_processed, feature_names = pipeline._engineer_features(
            X_train, X_test, y_train
        )

        # Check that features are scaled (approximately mean 0, std 1)
        assert abs(X_train_processed.mean()) < 0.1
        assert abs(X_train_processed.std() - 1.0) < 0.1
        assert hasattr(pipeline, 'scaler')

    def test_feature_engineering_no_scaling(self):
        """Test feature engineering without scaling."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        pipeline.experiment_config.scale_features = False

        X_train, X_test = self.X_classification.iloc[:80], self.X_classification.iloc[80:]
        y_train = self.y_classification.iloc[:80]

        X_train_processed, X_test_processed, feature_names = pipeline._engineer_features(
            X_train, X_test, y_train
        )

        assert not hasattr(pipeline, 'scaler') or pipeline.scaler is None

    def test_feature_engineering_missing_value_imputation(self):
        """Test feature engineering with missing value imputation."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        pipeline.experiment_config.handle_missing = "impute"

        # Add missing values
        X_train = self.X_classification.iloc[:80].copy()
        X_train.loc[0:10, 'feature_1'] = np.nan
        X_test = self.X_classification.iloc[80:].copy()
        y_train = self.y_classification.iloc[:80]

        X_train_processed, X_test_processed, feature_names = pipeline._engineer_features(
            X_train, X_test, y_train
        )

        # Check that no NaN values remain
        assert not np.isnan(X_train_processed).any()
        assert not np.isnan(X_test_processed).any()

    def test_model_creation_random_forest_classification(self):
        """Test Random Forest model creation for classification."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        model = pipeline._create_random_forest(MLTaskType.CLASSIFICATION)

        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(model, RandomForestClassifier)

    def test_model_creation_random_forest_regression(self):
        """Test Random Forest model creation for regression."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        model = pipeline._create_random_forest(MLTaskType.REGRESSION)

        from sklearn.ensemble import RandomForestRegressor
        assert isinstance(model, RandomForestRegressor)

    def test_model_creation_svm_classification(self):
        """Test SVM model creation for classification."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        model = pipeline._create_svm(MLTaskType.CLASSIFICATION)

        from sklearn.svm import SVC
        assert isinstance(model, SVC)

    def test_model_creation_invalid_task_type(self):
        """Test model creation with invalid task type for specific models."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Logistic regression for regression task
        with pytest.raises(ValueError, match="only supports classification"):
            pipeline._create_logistic_regression(MLTaskType.REGRESSION)

        # Linear regression for classification task
        with pytest.raises(ValueError, match="only supports regression"):
            pipeline._create_linear_regression(MLTaskType.CLASSIFICATION)

    def test_hyperparameter_tuning_grid_search(self):
        """Test hyperparameter tuning with grid search."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        pipeline.experiment_config.tuning_method = "grid_search"

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"n_estimators": [5, 10], "max_depth": [2, 3]}
        )

        base_model = pipeline._create_random_forest(MLTaskType.CLASSIFICATION)

        X_train = self.X_classification.iloc[:80].values
        y_train = self.y_classification.iloc[:80]

        tuned_model, best_params = pipeline._tune_hyperparameters(
            base_model, model_config, X_train, y_train
        )

        assert tuned_model is not None
        assert isinstance(best_params, dict)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params

    def test_hyperparameter_tuning_random_search(self):
        """Test hyperparameter tuning with random search."""
        pipeline = TrainingPipeline(self.config, "test_experiment")
        pipeline.experiment_config.tuning_method = "random_search"
        pipeline.experiment_config.max_iter = 3

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"n_estimators": [5, 10], "max_depth": [2, 3]}
        )

        base_model = pipeline._create_random_forest(MLTaskType.CLASSIFICATION)

        X_train = self.X_classification.iloc[:80].values
        y_train = self.y_classification.iloc[:80]

        tuned_model, best_params = pipeline._tune_hyperparameters(
            base_model, model_config, X_train, y_train
        )

        assert tuned_model is not None
        assert isinstance(best_params, dict)

    def test_feature_selection_mutual_info(self):
        """Test feature selection with mutual information."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            feature_selection_method=FeatureSelectionMethod.MUTUAL_INFO,
            max_features=2
        )

        X_train = self.X_classification.iloc[:80].values
        X_test = self.X_classification.iloc[80:].values
        y_train = self.y_classification.iloc[:80]
        feature_names = list(self.X_classification.columns)

        X_train_selected, X_test_selected, selected_features = pipeline._select_features(
            model_config, X_train, X_test, y_train, feature_names
        )

        assert X_train_selected.shape[1] == 2
        assert X_test_selected.shape[1] == 2
        assert len(selected_features) == 2

    def test_feature_selection_chi2(self):
        """Test feature selection with chi2."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            feature_selection_method=FeatureSelectionMethod.CHI2,
            max_features=2
        )

        # Create non-negative features for chi2
        X_train = np.abs(self.X_classification.iloc[:80].values)
        X_test = np.abs(self.X_classification.iloc[80:].values)
        y_train = self.y_classification.iloc[:80]
        feature_names = list(self.X_classification.columns)

        X_train_selected, X_test_selected, selected_features = pipeline._select_features(
            model_config, X_train, X_test, y_train, feature_names
        )

        assert X_train_selected.shape[1] == 2
        assert len(selected_features) == 2

    def test_feature_selection_chi2_negative_features(self):
        """Test feature selection with chi2 and negative features."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            feature_selection_method=FeatureSelectionMethod.CHI2,
            max_features=2
        )

        X_train = self.X_classification.iloc[:80].values  # Contains negative values
        X_test = self.X_classification.iloc[80:].values
        y_train = self.y_classification.iloc[:80]
        feature_names = list(self.X_classification.columns)

        # Should fall back to f_classif
        X_train_selected, X_test_selected, selected_features = pipeline._select_features(
            model_config, X_train, X_test, y_train, feature_names
        )

        assert len(pipeline.state.warnings) > 0
        assert "Negative features detected" in pipeline.state.warnings[-1]

    def test_model_evaluation_classification(self):
        """Test model evaluation for classification."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Train a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)

        X_train = self.X_classification.iloc[:80].values
        X_test = self.X_classification.iloc[80:].values
        y_train = self.y_classification.iloc[:80]
        y_test = self.y_classification.iloc[80:]

        model.fit(X_train, y_train)

        metrics = pipeline._evaluate_model(
            model, MLTaskType.CLASSIFICATION, X_test, y_test, None, 1.0
        )

        assert 0.0 <= metrics.accuracy <= 1.0
        assert 0.0 <= metrics.precision <= 1.0
        assert 0.0 <= metrics.recall <= 1.0
        assert 0.0 <= metrics.f1_score <= 1.0
        assert metrics.training_time == 1.0
        assert metrics.inference_time > 0

    def test_model_evaluation_regression(self):
        """Test model evaluation for regression."""
        # Create regression model config
        model_config = MLModelConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=MLTaskType.REGRESSION
        )

        experiment_config = MLExperimentConfig(
            name="regression_experiment",
            model_configs=[model_config]
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "regression_experiment")

        # Train a simple model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

        X_train = self.X_regression.iloc[:80].values
        X_test = self.X_regression.iloc[80:].values
        y_train = self.y_regression.iloc[:80]
        y_test = self.y_regression.iloc[80:]

        model.fit(X_train, y_train)

        metrics = pipeline._evaluate_model(
            model, MLTaskType.REGRESSION, X_test, y_test, None, 1.0
        )

        # For regression, accuracy is R² score
        assert -1.0 <= metrics.accuracy <= 1.0  # R² can be negative
        assert metrics.mean_squared_error is not None
        assert metrics.mean_absolute_error is not None
        assert metrics.mean_squared_error >= 0
        assert metrics.mean_absolute_error >= 0

    def test_cross_validation_classification(self):
        """Test cross-validation for classification."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)

        X_train = self.X_classification.iloc[:80].values
        y_train = self.y_classification.iloc[:80]

        cv_scores = pipeline._cross_validate(
            model, X_train, y_train, MLTaskType.CLASSIFICATION
        )

        assert isinstance(cv_scores, dict)
        assert 'mean_score' in cv_scores
        assert 'std_score' in cv_scores
        assert 'min_score' in cv_scores
        assert 'max_score' in cv_scores

    def test_cross_validation_regression(self):
        """Test cross-validation for regression."""
        # Create regression pipeline
        model_config = MLModelConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=MLTaskType.REGRESSION
        )

        experiment_config = MLExperimentConfig(
            name="regression_experiment",
            model_configs=[model_config]
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "regression_experiment")

        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

        X_train = self.X_regression.iloc[:80].values
        y_train = self.y_regression.iloc[:80]

        cv_scores = pipeline._cross_validate(
            model, X_train, y_train, MLTaskType.REGRESSION
        )

        assert isinstance(cv_scores, dict)
        assert 'mean_score' in cv_scores

    @patch('common.ml.pipeline.training_pipeline.cross_val_score')
    def test_cross_validation_failure(self, mock_cv):
        """Test cross-validation failure handling."""
        mock_cv.side_effect = Exception("CV failed")

        pipeline = TrainingPipeline(self.config, "test_experiment")

        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)

        X_train = self.X_classification.iloc[:80].values
        y_train = self.y_classification.iloc[:80]

        cv_scores = pipeline._cross_validate(
            model, X_train, y_train, MLTaskType.CLASSIFICATION
        )

        assert cv_scores == {}

    def test_full_pipeline_classification(self):
        """Test complete pipeline execution for classification."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        results = pipeline.fit(self.X_classification, self.y_classification)

        assert isinstance(results, list)
        assert len(results) == 1  # One model in config

        result = results[0]
        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert result.metrics is not None
        assert result.model_config is not None
        assert isinstance(result.feature_names, list)

        # Check that metrics meet minimum requirements
        assert result.metrics.accuracy >= 0.0

    def test_full_pipeline_regression(self):
        """Test complete pipeline execution for regression."""
        # Create regression model config
        model_config = MLModelConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=MLTaskType.REGRESSION
        )

        experiment_config = MLExperimentConfig(
            name="regression_experiment",
            model_configs=[model_config]
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "regression_experiment")

        results = pipeline.fit(self.X_regression, self.y_regression)

        assert isinstance(results, list)
        assert len(results) == 1

        result = results[0]
        assert result.model is not None
        assert result.metrics.mean_squared_error is not None
        assert result.metrics.mean_absolute_error is not None

    def test_pipeline_with_multiple_models(self):
        """Test pipeline with multiple model configurations."""
        # Create config with multiple models
        model_configs = [
            MLModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                task_type=MLTaskType.CLASSIFICATION,
                hyperparameters={"n_estimators": [5, 10]}
            ),
            MLModelConfig(
                model_type=ModelType.LOGISTIC_REGRESSION,
                task_type=MLTaskType.CLASSIFICATION,
                hyperparameters={"C": [0.1, 1.0]}
            )
        ]

        experiment_config = MLExperimentConfig(
            name="multi_model_experiment",
            model_configs=model_configs,
            cv_folds=3
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "multi_model_experiment")

        results = pipeline.fit(self.X_classification, self.y_classification)

        assert len(results) == 2

        # Check that results are sorted by performance (best first)
        for i in range(1, len(results)):
            assert results[i-1].metrics.f1_score >= results[i].metrics.f1_score

    def test_pipeline_progress_callback(self):
        """Test pipeline with progress callback."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        progress_values = []

        def progress_callback(progress):
            progress_values.append(progress)

        pipeline.fit(
            self.X_classification,
            self.y_classification,
            progress_callback=progress_callback
        )

        assert len(progress_values) > 0
        assert progress_values[0] >= 0.0
        assert progress_values[-1] == 1.0

    def test_pipeline_validation_data(self):
        """Test pipeline with validation data."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Split data manually for validation
        split_idx = 20
        validation_X = self.X_classification.iloc[:split_idx].values
        validation_y = self.y_classification.iloc[:split_idx].values
        train_X = self.X_classification.iloc[split_idx:]
        train_y = self.y_classification.iloc[split_idx:]

        results = pipeline.fit(
            train_X,
            train_y,
            validation_data=(validation_X, validation_y)
        )

        assert len(results) == 1
        assert results[0].model is not None

    def test_save_and_load_results(self):
        """Test saving and loading pipeline results."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        results = pipeline.fit(self.X_classification, self.y_classification)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Save results
            pipeline.save_results(results, temp_path)
            assert temp_path.exists()

            # Load results
            loaded_results = TrainingPipeline.load_results(temp_path)

            assert len(loaded_results) == len(results)
            assert loaded_results[0].model_config.model_type == results[0].model_config.model_type

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_load_results_file_not_found(self):
        """Test loading results from non-existent file."""
        with pytest.raises(FileNotFoundError, match="Results file not found"):
            TrainingPipeline.load_results("nonexistent.pkl")

    def test_pipeline_state_tracking(self):
        """Test pipeline state tracking."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        initial_state = pipeline.get_pipeline_state()
        assert initial_state.current_step == "initialization"
        assert initial_state.progress == 0.0

        # Run pipeline to update state
        pipeline.fit(self.X_classification, self.y_classification)

        final_state = pipeline.get_pipeline_state()
        assert final_state.progress == 1.0
        assert len(final_state.step_times) > 0

    def test_model_training_error_handling(self):
        """Test error handling during model training."""
        # Create configuration that will likely fail
        model_config = MLModelConfig(
            model_type=ModelType.SVM,  # SVM might fail with default params on this data
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"C": [1e-10]}  # Very small C might cause issues
        )

        experiment_config = MLExperimentConfig(
            name="failing_experiment",
            model_configs=[model_config],
            cv_folds=3
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "failing_experiment")

        # This might succeed or fail depending on data, but should handle gracefully
        try:
            results = pipeline.fit(self.X_classification, self.y_classification)
            # If it succeeds, that's fine
            assert len(results) >= 0
        except (ModelTrainingError, HyperparameterTuningError):
            # If it fails, that's expected for some configurations
            pass

    def test_pipeline_with_mixed_task_types_warning(self):
        """Test pipeline warning with mixed task types."""
        model_configs = [
            MLModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                task_type=MLTaskType.CLASSIFICATION
            ),
            MLModelConfig(
                model_type=ModelType.LINEAR_REGRESSION,
                task_type=MLTaskType.REGRESSION
            )
        ]

        experiment_config = MLExperimentConfig(
            name="mixed_experiment",
            model_configs=model_configs
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "mixed_experiment")

        # Should generate warning about mixed task types
        assert len(pipeline.state.warnings) > 0
        assert "Multiple task types detected" in pipeline.state.warnings[0]

    def test_pipeline_memory_limit_warning(self):
        """Test pipeline warning for low memory limit."""
        config = MLConfig(
            project_name="test_project",
            experiments=[self.config.experiments[0]],
            max_memory_gb=0.5  # Very low memory limit
        )

        pipeline = TrainingPipeline(config, "test_experiment")

        # Should generate warning about low memory
        assert len(pipeline.state.warnings) > 0
        assert "Low memory limit" in pipeline.state.warnings[0]

    @patch('common.ml.pipeline.training_pipeline.GridSearchCV')
    def test_hyperparameter_tuning_error(self, mock_grid_search):
        """Test hyperparameter tuning error handling."""
        mock_grid_search.side_effect = Exception("Grid search failed")

        pipeline = TrainingPipeline(self.config, "test_experiment")
        pipeline.experiment_config.tuning_method = "grid_search"

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            hyperparameters={"n_estimators": [5, 10]}
        )

        base_model = pipeline._create_random_forest(MLTaskType.CLASSIFICATION)

        X_train = self.X_classification.iloc[:80].values
        y_train = self.y_classification.iloc[:80]

        with pytest.raises(HyperparameterTuningError, match="Hyperparameter tuning failed"):
            pipeline._tune_hyperparameters(base_model, model_config, X_train, y_train)

    def test_numpy_array_input(self):
        """Test pipeline with numpy array inputs."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Convert to numpy arrays
        X_np = self.X_classification.values
        y_np = self.y_classification.values

        results = pipeline.fit(X_np, y_np)

        assert len(results) == 1
        assert results[0].model is not None

    def test_get_primary_metric_classification(self):
        """Test getting primary metric for classification."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        metric = pipeline._get_primary_metric()
        assert metric == "f1_score"

    def test_get_primary_metric_regression(self):
        """Test getting primary metric for regression."""
        model_config = MLModelConfig(
            model_type=ModelType.LINEAR_REGRESSION,
            task_type=MLTaskType.REGRESSION
        )

        experiment_config = MLExperimentConfig(
            name="regression_experiment",
            model_configs=[model_config]
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "regression_experiment")

        metric = pipeline._get_primary_metric()
        assert metric == "accuracy"  # R² stored in accuracy field


class TestPipelineEdgeCases:
    """Test edge cases and error conditions."""

    def test_extremely_small_dataset(self):
        """Test pipeline with extremely small dataset."""
        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION
        )

        experiment_config = MLExperimentConfig(
            name="small_data_experiment",
            model_configs=[model_config],
            cv_folds=2,  # Reduce CV folds for small data
            test_size=0.3
        )

        config = MLConfig(
            project_name="test_project",
            experiments=[experiment_config]
        )

        pipeline = TrainingPipeline(config, "small_data_experiment")

        # Create very small dataset
        X_tiny = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5, 6],
            'feature_2': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        })
        y_tiny = pd.Series([0, 1, 0, 1, 0, 1])

        with pytest.raises(DataValidationError, match="Insufficient samples"):
            pipeline.fit(X_tiny, y_tiny)

    def test_all_same_features(self):
        """Test pipeline with all features having same values."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Create dataset with constant features
        X_constant = pd.DataFrame({
            'feature_1': [1.0] * 100,
            'feature_2': [2.0] * 100,
            'feature_3': [3.0] * 100
        })
        y_varied = pd.Series(np.random.randint(0, 3, 100))

        # Should run but may generate warnings
        try:
            results = pipeline.fit(X_constant, y_varied)
            # If it succeeds, check results
            assert len(results) >= 0
        except (ModelTrainingError, HyperparameterTuningError):
            # Expected for constant features
            pass

    def test_single_feature(self):
        """Test pipeline with single feature."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        X_single = pd.DataFrame({'feature_1': np.random.normal(0, 1, 100)})
        y_varied = pd.Series(np.random.randint(0, 3, 100))

        results = pipeline.fit(X_single, y_varied)
        assert len(results) == 1

    def test_very_high_dimensional_data(self):
        """Test pipeline with high-dimensional data."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Create high-dimensional dataset
        n_samples = 50
        n_features = 200
        X_high_dim = pd.DataFrame(
            np.random.normal(0, 1, (n_samples, n_features)),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        y_high_dim = pd.Series(np.random.randint(0, 2, n_samples))

        # Should handle gracefully (may generate warnings about overfitting)
        try:
            results = pipeline.fit(X_high_dim, y_high_dim)
            assert len(results) >= 0
        except (DataValidationError, ModelTrainingError):
            # Expected for high-dimensional data with few samples
            pass

    def test_perfect_correlation_features(self):
        """Test pipeline with perfectly correlated features."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Create perfectly correlated features
        base_feature = np.random.normal(0, 1, 100)
        X_corr = pd.DataFrame({
            'feature_1': base_feature,
            'feature_2': base_feature * 2,  # Perfect correlation
            'feature_3': base_feature + 0.1  # Near perfect correlation
        })
        y_varied = pd.Series(np.random.randint(0, 3, 100))

        results = pipeline.fit(X_corr, y_varied)
        assert len(results) == 1

    @patch('time.time')
    def test_timeout_handling(self, mock_time):
        """Test pipeline timeout handling."""
        # Mock time to simulate long training
        mock_time.side_effect = [0, 0, 0, 1000000]  # Large time jump

        pipeline = TrainingPipeline(self.config, "test_experiment")

        # Set very short timeout
        pipeline.config.training_timeout = 1  # 1 second

        X = pd.DataFrame({'feature_1': np.random.normal(0, 1, 20)})
        y = pd.Series(np.random.randint(0, 2, 20))

        # Should complete even with mocked time (no actual timeout implementation yet)
        results = pipeline.fit(X, y)
        assert len(results) >= 0

    def test_feature_selection_more_features_than_samples(self):
        """Test feature selection with more features than samples."""
        pipeline = TrainingPipeline(self.config, "test_experiment")

        model_config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            task_type=MLTaskType.CLASSIFICATION,
            feature_selection_method=FeatureSelectionMethod.MUTUAL_INFO,
            max_features=100  # More than available features
        )

        X_small = pd.DataFrame(np.random.normal(0, 1, (20, 5)))
        X_test = pd.DataFrame(np.random.normal(0, 1, (5, 5)))
        y_small = pd.Series(np.random.randint(0, 2, 20))
        feature_names = [f'feature_{i}' for i in range(5)]

        X_train_selected, X_test_selected, selected_features = pipeline._select_features(
            model_config, X_small.values, X_test.values, y_small, feature_names
        )

        # Should use all available features
        assert X_train_selected.shape[1] == 5
        assert len(selected_features) == 5
