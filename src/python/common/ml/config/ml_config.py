"""
ML Configuration Management System with lua-style configuration access.

Provides lua-style configuration access for ML pipelines, experiments,
and model deployment. All configuration is accessed through get_config()
functions matching the Rust pattern.
"""

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import yaml

# Import lua-style configuration functions
from ...core.config import (
    get_config_string, get_config_bool, get_config_int, get_config_float,
    get_config_list, get_config_dict
)


# =============================================================================
# LUA-STYLE ML CONFIGURATION ACCESS FUNCTIONS
# =============================================================================

# ML Model Configuration Functions
def get_ml_model_type() -> str:
    """Get ML model type using lua-style configuration access."""
    return get_config_string("ml.model.type", "random_forest")


def get_ml_task_type() -> str:
    """Get ML task type using lua-style configuration access."""
    return get_config_string("ml.task.type", "classification")


def get_ml_hyperparameters() -> Dict[str, Any]:
    """Get ML model hyperparameters using lua-style configuration access."""
    return get_config_dict("ml.model.hyperparameters", {})


def get_ml_feature_selection_method() -> Optional[str]:
    """Get ML feature selection method using lua-style configuration access."""
    method = get_config_string("ml.features.selection_method", "")
    return method if method else None


def get_ml_max_features() -> Optional[int]:
    """Get ML max features using lua-style configuration access."""
    max_features = get_config_int("ml.features.max_features", 0)
    return max_features if max_features > 0 else None


# ML Experiment Configuration Functions
def get_ml_experiment_name() -> str:
    """Get ML experiment name using lua-style configuration access."""
    return get_config_string("ml.experiment.name", "default_experiment")


def get_ml_experiment_description() -> str:
    """Get ML experiment description using lua-style configuration access."""
    return get_config_string("ml.experiment.description", "")


def get_ml_tuning_method() -> str:
    """Get ML hyperparameter tuning method using lua-style configuration access."""
    return get_config_string("ml.experiment.tuning.method", "random_search")


def get_ml_cv_folds() -> int:
    """Get ML cross-validation folds using lua-style configuration access."""
    return get_config_int("ml.experiment.tuning.cv_folds", 5)


def get_ml_scoring_metric() -> str:
    """Get ML scoring metric using lua-style configuration access."""
    return get_config_string("ml.experiment.tuning.scoring_metric", "accuracy")


def get_ml_max_iter() -> int:
    """Get ML max iterations using lua-style configuration access."""
    return get_config_int("ml.experiment.tuning.max_iter", 100)


def get_ml_n_jobs() -> int:
    """Get ML number of jobs using lua-style configuration access."""
    return get_config_int("ml.experiment.tuning.n_jobs", -1)


# Training Configuration Functions
def get_ml_test_size() -> float:
    """Get ML test size using lua-style configuration access."""
    return get_config_float("ml.training.test_size", 0.2)


def get_ml_random_state() -> int:
    """Get ML random state using lua-style configuration access."""
    return get_config_int("ml.training.random_state", 42)


def get_ml_stratify() -> bool:
    """Get ML stratify setting using lua-style configuration access."""
    return get_config_bool("ml.training.stratify", True)


# Feature Engineering Configuration Functions
def get_ml_scale_features() -> bool:
    """Get ML scale features setting using lua-style configuration access."""
    return get_config_bool("ml.features.scale_features", True)


def get_ml_handle_missing() -> str:
    """Get ML handle missing values method using lua-style configuration access."""
    return get_config_string("ml.features.handle_missing", "impute")


def get_ml_categorical_encoding() -> str:
    """Get ML categorical encoding method using lua-style configuration access."""
    return get_config_string("ml.features.categorical_encoding", "onehot")


# Model Validation Configuration Functions
def get_ml_min_accuracy() -> float:
    """Get ML minimum accuracy threshold using lua-style configuration access."""
    return get_config_float("ml.validation.min_accuracy", 0.6)


def get_ml_min_precision() -> float:
    """Get ML minimum precision threshold using lua-style configuration access."""
    return get_config_float("ml.validation.min_precision", 0.6)


def get_ml_min_recall() -> float:
    """Get ML minimum recall threshold using lua-style configuration access."""
    return get_config_float("ml.validation.min_recall", 0.6)


# Storage and Logging Configuration Functions
def get_ml_storage_enabled() -> bool:
    """Get ML storage enabled setting using lua-style configuration access."""
    return get_config_bool("ml.storage.enabled", True)


def get_ml_storage_backend() -> str:
    """Get ML storage backend using lua-style configuration access."""
    return get_config_string("ml.storage.backend", "local")


def get_ml_storage_path() -> str:
    """Get ML storage path using lua-style configuration access."""
    return get_config_string("ml.storage.path", "./ml_models")


def get_ml_logging_enabled() -> bool:
    """Get ML logging enabled setting using lua-style configuration access."""
    return get_config_bool("ml.logging.enabled", True)


def get_ml_logging_level() -> str:
    """Get ML logging level using lua-style configuration access."""
    return get_config_string("ml.logging.level", "info")


def get_ml_experiment_tracking_enabled() -> bool:
    """Get ML experiment tracking enabled setting using lua-style configuration access."""
    return get_config_bool("ml.experiment.tracking.enabled", False)


def get_ml_experiment_tracking_backend() -> str:
    """Get ML experiment tracking backend using lua-style configuration access."""
    return get_config_string("ml.experiment.tracking.backend", "mlflow")


# Model Serving Configuration Functions
def get_ml_serving_enabled() -> bool:
    """Get ML model serving enabled setting using lua-style configuration access."""
    return get_config_bool("ml.serving.enabled", False)


def get_ml_serving_host() -> str:
    """Get ML model serving host using lua-style configuration access."""
    return get_config_string("ml.serving.host", "127.0.0.1")


def get_ml_serving_port() -> int:
    """Get ML model serving port using lua-style configuration access."""
    return get_config_int("ml.serving.port", 8080)


def get_ml_serving_workers() -> int:
    """Get ML model serving workers using lua-style configuration access."""
    return get_config_int("ml.serving.workers", 1)


def get_ml_serving_timeout() -> int:
    """Get ML model serving timeout using lua-style configuration access."""
    return get_config_int("ml.serving.timeout", 30)


# Auto ML Configuration Functions
def get_ml_automl_enabled() -> bool:
    """Get ML AutoML enabled setting using lua-style configuration access."""
    return get_config_bool("ml.automl.enabled", False)


def get_ml_automl_time_limit() -> int:
    """Get ML AutoML time limit using lua-style configuration access."""
    return get_config_int("ml.automl.time_limit", 3600)


def get_ml_automl_metric() -> str:
    """Get ML AutoML optimization metric using lua-style configuration access."""
    return get_config_string("ml.automl.metric", "accuracy")


def get_ml_automl_algorithms() -> List[str]:
    """Get ML AutoML algorithms list using lua-style configuration access."""
    return get_config_list("ml.automl.algorithms", [
        "random_forest", "gradient_boosting", "logistic_regression"
    ])


# =============================================================================
# VALIDATION FUNCTIONS (NO LONGER CLASS-BASED)
# =============================================================================

def validate_ml_task_type(task_type: str) -> bool:
    """Validate ML task type."""
    valid_types = [
        "classification", "regression", "clustering",
        "recommendation", "document_similarity"
    ]
    return task_type in valid_types


def validate_ml_model_type(model_type: str) -> bool:
    """Validate ML model type."""
    valid_types = [
        "random_forest", "gradient_boosting", "svm",
        "logistic_regression", "linear_regression", "kmeans",
        "dbscan", "neural_network"
    ]
    return model_type in valid_types


def validate_ml_feature_selection_method(method: str) -> bool:
    """Validate ML feature selection method."""
    valid_methods = [
        "mutual_info", "chi2", "f_score",
        "recursive_elimination", "lasso", "tree_importance"
    ]
    return method in valid_methods


def validate_ml_tuning_method(method: str) -> bool:
    """Validate ML hyperparameter tuning method."""
    valid_methods = ["grid_search", "random_search", "bayesian"]
    return method in valid_methods


def validate_ml_handle_missing(method: str) -> bool:
    """Validate ML missing value handling method."""
    valid_methods = ["drop", "impute", "flag"]
    return method in valid_methods


def validate_ml_categorical_encoding(method: str) -> bool:
    """Validate ML categorical encoding method."""
    valid_methods = ["onehot", "label", "target"]
    return method in valid_methods


def validate_ml_storage_backend(backend: str) -> bool:
    """Validate ML storage backend."""
    valid_backends = ["local", "s3", "gcs", "azure"]
    return backend in valid_backends


def validate_ml_experiment_tracking_backend(backend: str) -> bool:
    """Validate ML experiment tracking backend."""
    valid_backends = ["mlflow", "wandb", "neptune", "tensorboard"]
    return backend in valid_backends


def validate_ml_config() -> List[str]:
    """Validate complete ML configuration and return any issues."""
    issues = []

    # Validate task type
    task_type = get_ml_task_type()
    if not validate_ml_task_type(task_type):
        issues.append(f"Invalid ML task type: {task_type}")

    # Validate model type
    model_type = get_ml_model_type()
    if not validate_ml_model_type(model_type):
        issues.append(f"Invalid ML model type: {model_type}")

    # Validate feature selection method if set
    feature_method = get_ml_feature_selection_method()
    if feature_method and not validate_ml_feature_selection_method(feature_method):
        issues.append(f"Invalid ML feature selection method: {feature_method}")

    # Validate tuning method
    tuning_method = get_ml_tuning_method()
    if not validate_ml_tuning_method(tuning_method):
        issues.append(f"Invalid ML tuning method: {tuning_method}")

    # Validate CV folds
    cv_folds = get_ml_cv_folds()
    if cv_folds < 2 or cv_folds > 20:
        issues.append("ML CV folds must be between 2 and 20")

    # Validate max iterations
    max_iter = get_ml_max_iter()
    if max_iter < 1:
        issues.append("ML max iterations must be at least 1")

    # Validate test size
    test_size = get_ml_test_size()
    if test_size <= 0.0 or test_size >= 1.0:
        issues.append("ML test size must be between 0.0 and 1.0 (exclusive)")

    # Validate missing value handling
    handle_missing = get_ml_handle_missing()
    if not validate_ml_handle_missing(handle_missing):
        issues.append(f"Invalid ML handle missing method: {handle_missing}")

    # Validate categorical encoding
    categorical_encoding = get_ml_categorical_encoding()
    if not validate_ml_categorical_encoding(categorical_encoding):
        issues.append(f"Invalid ML categorical encoding: {categorical_encoding}")

    # Validate thresholds
    min_accuracy = get_ml_min_accuracy()
    if min_accuracy < 0.0 or min_accuracy > 1.0:
        issues.append("ML minimum accuracy must be between 0.0 and 1.0")

    min_precision = get_ml_min_precision()
    if min_precision < 0.0 or min_precision > 1.0:
        issues.append("ML minimum precision must be between 0.0 and 1.0")

    min_recall = get_ml_min_recall()
    if min_recall < 0.0 or min_recall > 1.0:
        issues.append("ML minimum recall must be between 0.0 and 1.0")

    # Validate storage backend if enabled
    if get_ml_storage_enabled():
        storage_backend = get_ml_storage_backend()
        if not validate_ml_storage_backend(storage_backend):
            issues.append(f"Invalid ML storage backend: {storage_backend}")

    # Validate experiment tracking backend if enabled
    if get_ml_experiment_tracking_enabled():
        tracking_backend = get_ml_experiment_tracking_backend()
        if not validate_ml_experiment_tracking_backend(tracking_backend):
            issues.append(f"Invalid ML experiment tracking backend: {tracking_backend}")

    # Validate serving configuration if enabled
    if get_ml_serving_enabled():
        serving_port = get_ml_serving_port()
        if serving_port < 1 or serving_port > 65535:
            issues.append("ML serving port must be between 1 and 65535")

        serving_workers = get_ml_serving_workers()
        if serving_workers < 1:
            issues.append("ML serving workers must be at least 1")

        serving_timeout = get_ml_serving_timeout()
        if serving_timeout < 1:
            issues.append("ML serving timeout must be at least 1 second")

    # Validate AutoML configuration if enabled
    if get_ml_automl_enabled():
        automl_time_limit = get_ml_automl_time_limit()
        if automl_time_limit < 60:
            issues.append("ML AutoML time limit must be at least 60 seconds")

        automl_algorithms = get_ml_automl_algorithms()
        if not automl_algorithms:
            issues.append("ML AutoML algorithms list cannot be empty")

        for algorithm in automl_algorithms:
            if not validate_ml_model_type(algorithm):
                issues.append(f"Invalid ML AutoML algorithm: {algorithm}")

    return issues


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_ml_config_summary() -> Dict[str, Any]:
    """Get a summary of current ML configuration."""
    return {
        "model": {
            "type": get_ml_model_type(),
            "task_type": get_ml_task_type(),
            "hyperparameters": get_ml_hyperparameters(),
            "feature_selection": get_ml_feature_selection_method(),
            "max_features": get_ml_max_features(),
        },
        "experiment": {
            "name": get_ml_experiment_name(),
            "description": get_ml_experiment_description(),
            "tuning_method": get_ml_tuning_method(),
            "cv_folds": get_ml_cv_folds(),
            "scoring_metric": get_ml_scoring_metric(),
            "max_iter": get_ml_max_iter(),
            "n_jobs": get_ml_n_jobs(),
        },
        "training": {
            "test_size": get_ml_test_size(),
            "random_state": get_ml_random_state(),
            "stratify": get_ml_stratify(),
        },
        "features": {
            "scale_features": get_ml_scale_features(),
            "handle_missing": get_ml_handle_missing(),
            "categorical_encoding": get_ml_categorical_encoding(),
        },
        "validation": {
            "min_accuracy": get_ml_min_accuracy(),
            "min_precision": get_ml_min_precision(),
            "min_recall": get_ml_min_recall(),
        },
        "storage": {
            "enabled": get_ml_storage_enabled(),
            "backend": get_ml_storage_backend(),
            "path": get_ml_storage_path(),
        },
        "serving": {
            "enabled": get_ml_serving_enabled(),
            "host": get_ml_serving_host(),
            "port": get_ml_serving_port(),
            "workers": get_ml_serving_workers(),
            "timeout": get_ml_serving_timeout(),
        },
        "automl": {
            "enabled": get_ml_automl_enabled(),
            "time_limit": get_ml_automl_time_limit(),
            "metric": get_ml_automl_metric(),
            "algorithms": get_ml_automl_algorithms(),
        },
    }