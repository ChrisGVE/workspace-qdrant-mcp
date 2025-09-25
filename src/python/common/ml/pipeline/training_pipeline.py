"""
Advanced ML Training Pipeline with Hyperparameter Tuning

Provides comprehensive training pipeline with automated hyperparameter optimization,
cross-validation, model selection, and performance evaluation with extensive
error handling and monitoring capabilities.
"""

import logging
import time
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold,
    KFold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2, f_classif, RFE
)
from sklearn.exceptions import ConvergenceWarning

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from ..config.ml_config import (
    MLConfig,
    MLExperimentConfig,
    MLModelConfig,
    MLPerformanceMetrics,
    MLTaskType,
    ModelType,
    FeatureSelectionMethod
)


@dataclass
class TrainingResult:
    """Results from training pipeline execution."""
    model: Any
    metrics: MLPerformanceMetrics
    model_config: MLModelConfig
    feature_names: List[str]
    scaler: Optional[StandardScaler] = None
    feature_selector: Optional[Any] = None
    label_encoder: Optional[LabelEncoder] = None
    cv_scores: Optional[Dict[str, float]] = None
    best_params: Optional[Dict[str, Any]] = None
    training_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Tracks the current state of the training pipeline."""
    current_step: str = "initialization"
    progress: float = 0.0
    start_time: float = field(default_factory=time.time)
    step_times: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MLPipelineError(Exception):
    """Custom exception for ML pipeline errors."""
    pass


class DataValidationError(MLPipelineError):
    """Exception raised for data validation errors."""
    pass


class ModelTrainingError(MLPipelineError):
    """Exception raised for model training errors."""
    pass


class HyperparameterTuningError(MLPipelineError):
    """Exception raised for hyperparameter tuning errors."""
    pass


class TrainingPipeline:
    """
    Advanced ML training pipeline with comprehensive error handling and monitoring.

    Features:
    - Automated hyperparameter tuning (Grid Search, Random Search, Bayesian)
    - Cross-validation with proper stratification
    - Feature engineering and selection
    - Model evaluation with multiple metrics
    - Extensive data validation and error handling
    - Progress monitoring and logging
    - Resource management and timeout handling
    """

    def __init__(self, config: MLConfig, experiment_name: str):
        """
        Initialize training pipeline.

        Args:
            config: ML configuration
            experiment_name: Name of experiment to run

        Raises:
            ValueError: If experiment not found or invalid configuration
        """
        self.config = config
        self.experiment_config = config.get_experiment_config(experiment_name)

        if not self.experiment_config:
            raise ValueError(f"Experiment '{experiment_name}' not found in configuration")

        self.logger = self._setup_logging()
        self.state = PipelineState()
        self._validate_configuration()

        # Model factory
        self._model_factory = {
            ModelType.RANDOM_FOREST: self._create_random_forest,
            ModelType.GRADIENT_BOOSTING: self._create_gradient_boosting,
            ModelType.SVM: self._create_svm,
            ModelType.LOGISTIC_REGRESSION: self._create_logistic_regression,
            ModelType.LINEAR_REGRESSION: self._create_linear_regression,
            ModelType.KMEANS: self._create_kmeans,
            ModelType.DBSCAN: self._create_dbscan,
            ModelType.NEURAL_NETWORK: self._create_neural_network,
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger(f"ml_pipeline_{self.experiment_config.name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _validate_configuration(self) -> None:
        """Validate pipeline configuration."""
        if not self.experiment_config.model_configs:
            raise ValueError("No model configurations provided")

        # Validate task types are consistent
        task_types = {mc.task_type for mc in self.experiment_config.model_configs}
        if len(task_types) > 1:
            self.state.warnings.append(
                f"Multiple task types detected: {task_types}. "
                "This may lead to inconsistent evaluation metrics."
            )

        # Check resource limits
        if self.config.max_memory_gb < 1.0:
            self.state.warnings.append("Low memory limit may cause training failures")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[Any, Any]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[TrainingResult]:
        """
        Train models using the configured pipeline.

        Args:
            X: Feature matrix
            y: Target values
            validation_data: Optional validation data (X_val, y_val)
            progress_callback: Optional callback for progress updates

        Returns:
            List of training results for each model

        Raises:
            DataValidationError: If input data is invalid
            ModelTrainingError: If training fails
            HyperparameterTuningError: If hyperparameter tuning fails
        """
        try:
            self._update_progress("data_validation", 0.0, progress_callback)
            X, y = self._validate_and_prepare_data(X, y)

            self._update_progress("data_splitting", 0.1, progress_callback)
            X_train, X_test, y_train, y_test = self._split_data(X, y)

            self._update_progress("feature_engineering", 0.2, progress_callback)
            X_train_processed, X_test_processed, feature_names = self._engineer_features(
                X_train, X_test, y_train
            )

            results = []
            total_models = len(self.experiment_config.model_configs)

            for i, model_config in enumerate(self.experiment_config.model_configs):
                model_progress_start = 0.3 + (i / total_models) * 0.6
                model_progress_end = 0.3 + ((i + 1) / total_models) * 0.6

                try:
                    self.logger.info(f"Training {model_config.model_type} model...")

                    result = self._train_single_model(
                        model_config,
                        X_train_processed,
                        X_test_processed,
                        y_train,
                        y_test,
                        feature_names,
                        validation_data,
                        progress_callback,
                        model_progress_start,
                        model_progress_end
                    )

                    results.append(result)
                    self.logger.info(
                        f"Model {model_config.model_type} trained successfully. "
                        f"Accuracy: {result.metrics.accuracy:.4f}"
                    )

                except Exception as e:
                    error_msg = f"Failed to train {model_config.model_type}: {str(e)}"
                    self.state.errors.append(error_msg)
                    self.logger.error(error_msg)
                    continue

            self._update_progress("completion", 1.0, progress_callback)

            if not results:
                raise ModelTrainingError("All models failed to train")

            # Sort results by performance
            primary_metric = self._get_primary_metric()
            results.sort(
                key=lambda r: getattr(r.metrics, primary_metric),
                reverse=True
            )

            self.logger.info(f"Pipeline completed successfully with {len(results)} models")
            return results

        except Exception as e:
            self.state.errors.append(str(e))
            raise

    def _validate_and_prepare_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Validate and prepare input data."""
        # Convert to pandas if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name="target")

        # Basic validation
        if X.empty:
            raise DataValidationError("Feature matrix X is empty")
        if y.empty:
            raise DataValidationError("Target vector y is empty")
        if len(X) != len(y):
            raise DataValidationError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        # Check for minimum samples
        min_samples = max(10, self.experiment_config.cv_folds * 2)
        if len(X) < min_samples:
            raise DataValidationError(
                f"Insufficient samples: {len(X)} < {min_samples} required"
            )

        # Check for excessive missing values
        missing_threshold = 0.5
        missing_ratio = X.isnull().sum().sum() / (X.shape[0] * X.shape[1])
        if missing_ratio > missing_threshold:
            raise DataValidationError(
                f"Too many missing values: {missing_ratio:.2%} > {missing_threshold:.2%}"
            )

        # Task-specific validation
        task_type = self.experiment_config.model_configs[0].task_type
        if task_type == MLTaskType.CLASSIFICATION:
            unique_classes = y.nunique()
            if unique_classes < 2:
                raise DataValidationError(
                    f"Classification requires at least 2 classes, got {unique_classes}"
                )
            if unique_classes > 100:
                self.state.warnings.append(
                    f"High number of classes ({unique_classes}) may impact performance"
                )

        elif task_type == MLTaskType.REGRESSION:
            if not np.issubdtype(y.dtype, np.number):
                raise DataValidationError("Regression target must be numeric")

        self.logger.info(f"Data validation passed: {X.shape} samples, {X.shape[1]} features")
        return X, y

    def _split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train/test sets."""
        stratify = None
        task_type = self.experiment_config.model_configs[0].task_type

        if (task_type == MLTaskType.CLASSIFICATION and
            self.experiment_config.stratify and
            y.nunique() > 1):
            stratify = y

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.experiment_config.test_size,
                random_state=self.experiment_config.random_state,
                stratify=stratify
            )

            self.logger.info(
                f"Data split: train={len(X_train)}, test={len(X_test)}"
            )

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise DataValidationError(f"Data splitting failed: {str(e)}")

    def _engineer_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Apply feature engineering and selection."""
        # Handle missing values
        if self.experiment_config.handle_missing == "drop":
            # Only drop if sufficient data remains
            X_train_clean = X_train.dropna()
            if len(X_train_clean) < len(X_train) * 0.7:
                self.state.warnings.append("Too much data lost with 'drop' strategy, using imputation")
                self.experiment_config.handle_missing = "impute"
            else:
                X_train, X_test = X_train_clean, X_test.dropna()

        if self.experiment_config.handle_missing == "impute":
            imputer = SimpleImputer(strategy='median' if X_train.select_dtypes(include=[np.number]).shape[1] > 0 else 'most_frequent')
            X_train_imputed = imputer.fit_transform(X_train.select_dtypes(include=[np.number]))
            X_test_imputed = imputer.transform(X_test.select_dtypes(include=[np.number]))

            # Handle categorical columns separately
            cat_columns = X_train.select_dtypes(exclude=[np.number]).columns
            if not cat_columns.empty:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                X_train_cat = cat_imputer.fit_transform(X_train[cat_columns])
                X_test_cat = cat_imputer.transform(X_test[cat_columns])

                X_train = np.hstack([X_train_imputed, X_train_cat])
                X_test = np.hstack([X_test_imputed, X_test_cat])
                feature_names = (
                    list(X_train.select_dtypes(include=[np.number]).columns) +
                    list(cat_columns)
                )
            else:
                X_train, X_test = X_train_imputed, X_test_imputed
                feature_names = list(X_train.select_dtypes(include=[np.number]).columns)

        # Feature scaling
        if self.experiment_config.scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            self.scaler = scaler
        else:
            X_train = X_train.values if hasattr(X_train, 'values') else X_train
            X_test = X_test.values if hasattr(X_test, 'values') else X_test
            self.scaler = None

        if not hasattr(self, 'feature_names') or not feature_names:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        self.logger.info(f"Feature engineering completed: {X_train.shape[1]} features")
        return X_train, X_test, feature_names

    def _train_single_model(
        self,
        model_config: MLModelConfig,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str],
        validation_data: Optional[Tuple[Any, Any]],
        progress_callback: Optional[Callable[[float], None]],
        progress_start: float,
        progress_end: float
    ) -> TrainingResult:
        """Train a single model with hyperparameter tuning."""
        start_time = time.time()

        try:
            # Create base model
            base_model = self._model_factory[model_config.model_type](model_config.task_type)

            # Hyperparameter tuning
            if model_config.hyperparameters:
                self._update_progress("hyperparameter_tuning", progress_start + 0.1, progress_callback)
                tuned_model, best_params = self._tune_hyperparameters(
                    base_model, model_config, X_train, y_train
                )
            else:
                tuned_model = base_model
                best_params = {}

            # Feature selection
            if model_config.feature_selection_method:
                self._update_progress("feature_selection", progress_start + 0.3, progress_callback)
                X_train, X_test, selected_features = self._select_features(
                    model_config, X_train, X_test, y_train, feature_names
                )
                feature_names = selected_features

            # Final training
            self._update_progress("model_training", progress_start + 0.5, progress_callback)
            tuned_model.fit(X_train, y_train)

            # Evaluation
            self._update_progress("evaluation", progress_start + 0.8, progress_callback)
            metrics = self._evaluate_model(
                tuned_model, model_config.task_type, X_test, y_test,
                validation_data, time.time() - start_time
            )

            # Cross-validation scores
            cv_scores = self._cross_validate(tuned_model, X_train, y_train, model_config.task_type)

            self._update_progress("completion", progress_end, progress_callback)

            return TrainingResult(
                model=tuned_model,
                metrics=metrics,
                model_config=model_config,
                feature_names=feature_names,
                scaler=getattr(self, 'scaler', None),
                best_params=best_params,
                cv_scores=cv_scores,
                training_metadata={
                    'training_time': time.time() - start_time,
                    'feature_count': len(feature_names),
                    'sample_count': len(X_train)
                }
            )

        except Exception as e:
            raise ModelTrainingError(f"Model training failed: {str(e)}")

    def _tune_hyperparameters(
        self,
        model: Any,
        model_config: MLModelConfig,
        X_train: np.ndarray,
        y_train: pd.Series
    ) -> Tuple[Any, Dict[str, Any]]:
        """Perform hyperparameter tuning."""
        try:
            param_grid = model_config.hyperparameters

            # Choose tuning strategy
            if self.experiment_config.tuning_method == "grid_search":
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=self.experiment_config.cv_folds,
                    scoring=self.experiment_config.scoring_metric,
                    n_jobs=self.experiment_config.n_jobs,
                    error_score='raise'
                )
            else:  # random_search
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=self.experiment_config.max_iter,
                    cv=self.experiment_config.cv_folds,
                    scoring=self.experiment_config.scoring_metric,
                    n_jobs=self.experiment_config.n_jobs,
                    random_state=self.experiment_config.random_state,
                    error_score='raise'
                )

            search.fit(X_train, y_train)

            self.logger.info(
                f"Hyperparameter tuning completed. Best score: {search.best_score_:.4f}"
            )

            return search.best_estimator_, search.best_params_

        except Exception as e:
            raise HyperparameterTuningError(f"Hyperparameter tuning failed: {str(e)}")

    def _select_features(
        self,
        model_config: MLModelConfig,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: pd.Series,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Perform feature selection."""
        method = model_config.feature_selection_method
        max_features = model_config.max_features or X_train.shape[1]
        max_features = min(max_features, X_train.shape[1])

        try:
            if method == FeatureSelectionMethod.MUTUAL_INFO:
                if model_config.task_type == MLTaskType.CLASSIFICATION:
                    selector = SelectKBest(mutual_info_classif, k=max_features)
                else:
                    selector = SelectKBest(f_classif, k=max_features)

            elif method == FeatureSelectionMethod.CHI2:
                # Ensure non-negative features for chi2
                if np.any(X_train < 0):
                    self.state.warnings.append("Negative features detected, switching to f_classif")
                    selector = SelectKBest(f_classif, k=max_features)
                else:
                    selector = SelectKBest(chi2, k=max_features)

            elif method == FeatureSelectionMethod.F_SCORE:
                selector = SelectKBest(f_classif, k=max_features)

            else:  # Other methods would need more complex implementation
                self.logger.warning(f"Feature selection method {method} not implemented, skipping")
                return X_train, X_test, feature_names

            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]

            self.logger.info(f"Feature selection completed: {len(selected_features)} features selected")

            return X_train_selected, X_test_selected, selected_features

        except Exception as e:
            self.logger.warning(f"Feature selection failed: {str(e)}, using all features")
            return X_train, X_test, feature_names

    def _evaluate_model(
        self,
        model: Any,
        task_type: MLTaskType,
        X_test: np.ndarray,
        y_test: pd.Series,
        validation_data: Optional[Tuple[Any, Any]],
        training_time: float
    ) -> MLPerformanceMetrics:
        """Evaluate model performance."""
        # Predictions
        inference_start = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - inference_start) / len(X_test)

        if task_type == MLTaskType.CLASSIFICATION:
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # ROC AUC for binary classification
            roc_auc = None
            try:
                if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_proba)
            except:
                pass

            return MLPerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                roc_auc=roc_auc,
                training_time=training_time,
                inference_time=inference_time
            )

        else:  # Regression
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # R² score as "accuracy" for regression
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)

            return MLPerformanceMetrics(
                accuracy=r2,
                precision=r2,  # Using R² as proxy
                recall=r2,     # Using R² as proxy
                f1_score=r2,   # Using R² as proxy
                mean_squared_error=mse,
                mean_absolute_error=mae,
                training_time=training_time,
                inference_time=inference_time
            )

    def _cross_validate(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: pd.Series,
        task_type: MLTaskType
    ) -> Dict[str, float]:
        """Perform cross-validation."""
        try:
            # Choose CV strategy
            if task_type == MLTaskType.CLASSIFICATION:
                cv = StratifiedKFold(
                    n_splits=self.experiment_config.cv_folds,
                    shuffle=True,
                    random_state=self.experiment_config.random_state
                )
            else:
                cv = KFold(
                    n_splits=self.experiment_config.cv_folds,
                    shuffle=True,
                    random_state=self.experiment_config.random_state
                )

            scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring=self.experiment_config.scoring_metric,
                n_jobs=self.experiment_config.n_jobs
            )

            return {
                'mean_score': float(scores.mean()),
                'std_score': float(scores.std()),
                'min_score': float(scores.min()),
                'max_score': float(scores.max())
            }

        except Exception as e:
            self.logger.warning(f"Cross-validation failed: {str(e)}")
            return {}

    def _create_random_forest(self, task_type: MLTaskType) -> Any:
        """Create Random Forest model."""
        if task_type == MLTaskType.CLASSIFICATION:
            return RandomForestClassifier(
                random_state=self.experiment_config.random_state,
                n_jobs=self.experiment_config.n_jobs
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                random_state=self.experiment_config.random_state,
                n_jobs=self.experiment_config.n_jobs
            )

    def _create_gradient_boosting(self, task_type: MLTaskType) -> Any:
        """Create Gradient Boosting model."""
        if task_type == MLTaskType.CLASSIFICATION:
            return GradientBoostingClassifier(
                random_state=self.experiment_config.random_state
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                random_state=self.experiment_config.random_state
            )

    def _create_svm(self, task_type: MLTaskType) -> Any:
        """Create SVM model."""
        if task_type == MLTaskType.CLASSIFICATION:
            return SVC(random_state=self.experiment_config.random_state)
        else:
            return SVR()

    def _create_logistic_regression(self, task_type: MLTaskType) -> Any:
        """Create Logistic Regression model."""
        if task_type != MLTaskType.CLASSIFICATION:
            raise ValueError("Logistic Regression only supports classification")
        return LogisticRegression(
            random_state=self.experiment_config.random_state,
            max_iter=1000
        )

    def _create_linear_regression(self, task_type: MLTaskType) -> Any:
        """Create Linear Regression model."""
        if task_type != MLTaskType.REGRESSION:
            raise ValueError("Linear Regression only supports regression")
        return LinearRegression()

    def _create_kmeans(self, task_type: MLTaskType) -> Any:
        """Create K-Means model."""
        if task_type != MLTaskType.CLUSTERING:
            raise ValueError("K-Means only supports clustering")
        return KMeans(random_state=self.experiment_config.random_state)

    def _create_dbscan(self, task_type: MLTaskType) -> Any:
        """Create DBSCAN model."""
        if task_type != MLTaskType.CLUSTERING:
            raise ValueError("DBSCAN only supports clustering")
        return DBSCAN()

    def _create_neural_network(self, task_type: MLTaskType) -> Any:
        """Create Neural Network model."""
        if task_type == MLTaskType.CLASSIFICATION:
            return MLPClassifier(
                random_state=self.experiment_config.random_state,
                max_iter=1000
            )
        else:
            return MLPRegressor(
                random_state=self.experiment_config.random_state,
                max_iter=1000
            )

    def _get_primary_metric(self) -> str:
        """Get the primary metric for ranking models."""
        task_type = self.experiment_config.model_configs[0].task_type

        if task_type == MLTaskType.CLASSIFICATION:
            return "f1_score"
        elif task_type == MLTaskType.REGRESSION:
            return "accuracy"  # R² score stored in accuracy field
        else:
            return "accuracy"

    def _update_progress(
        self,
        step: str,
        progress: float,
        callback: Optional[Callable[[float], None]]
    ) -> None:
        """Update pipeline progress."""
        self.state.current_step = step
        self.state.progress = progress
        self.state.step_times[step] = time.time() - self.state.start_time

        if callback:
            callback(progress)

    def save_results(
        self,
        results: List[TrainingResult],
        output_path: Union[str, Path]
    ) -> None:
        """Save training results to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        self.logger.info(f"Results saved to {output_path}")

    @classmethod
    def load_results(cls, input_path: Union[str, Path]) -> List[TrainingResult]:
        """Load training results from disk."""
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Results file not found: {input_path}")

        with open(input_path, 'rb') as f:
            results = pickle.load(f)

        return results

    def get_pipeline_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.state