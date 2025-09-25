"""
ML Configuration Management System

Provides comprehensive configuration management for ML pipelines,
experiments, and model deployment with validation and type safety.
"""

import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pydantic import BaseModel, Field, validator


class MLTaskType(str, Enum):
    """Supported ML task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    DOCUMENT_SIMILARITY = "document_similarity"


class ModelType(str, Enum):
    """Supported model types."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    LOGISTIC_REGRESSION = "logistic_regression"
    LINEAR_REGRESSION = "linear_regression"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    NEURAL_NETWORK = "neural_network"


class FeatureSelectionMethod(str, Enum):
    """Feature selection methods."""
    MUTUAL_INFO = "mutual_info"
    CHI2 = "chi2"
    F_SCORE = "f_score"
    RECURSIVE_ELIMINATION = "recursive_elimination"
    LASSO = "lasso"
    TREE_IMPORTANCE = "tree_importance"


class MLModelConfig(BaseModel):
    """Configuration for ML model parameters."""

    model_type: ModelType
    task_type: MLTaskType
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_selection_method: Optional[FeatureSelectionMethod] = None
    max_features: Optional[int] = None

    class Config:
        use_enum_values = True


class MLExperimentConfig(BaseModel):
    """Configuration for ML experiments and hyperparameter tuning."""

    name: str
    description: Optional[str] = None
    model_configs: List[MLModelConfig]

    # Hyperparameter tuning configuration
    tuning_method: str = Field(default="random_search", pattern="^(grid_search|random_search|bayesian)$")
    cv_folds: int = Field(default=5, ge=2, le=20)
    scoring_metric: str = "accuracy"
    max_iter: int = Field(default=100, ge=1)
    n_jobs: int = Field(default=-1, ge=-1)

    # Training configuration
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0)
    stratify: bool = True

    # Feature engineering
    scale_features: bool = True
    handle_missing: str = Field(default="impute", pattern="^(drop|impute|flag)$")
    categorical_encoding: str = Field(default="onehot", pattern="^(onehot|label|target)$")

    # Model validation
    min_accuracy: float = Field(default=0.6, ge=0.0, le=1.0)
    min_precision: float = Field(default=0.6, ge=0.0, le=1.0)
    min_recall: float = Field(default=0.6, ge=0.0, le=1.0)

    @validator('model_configs')
    def validate_model_configs(cls, v):
        if not v:
            raise ValueError("At least one model configuration is required")
        return v


class MLConfig(BaseModel):
    """Main ML configuration class."""

    # General settings
    project_name: str = Field(min_length=1)
    version: str = "1.0.0"
    data_directory: Path = Field(default_factory=lambda: Path("./data"))
    model_directory: Path = Field(default_factory=lambda: Path("./models"))
    artifacts_directory: Path = Field(default_factory=lambda: Path("./artifacts"))

    # Database configuration
    registry_url: Optional[str] = None
    tracking_uri: Optional[str] = None

    # Monitoring configuration
    enable_monitoring: bool = True
    drift_detection_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    performance_alert_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    monitoring_interval: int = Field(default=3600, ge=60)  # seconds

    # Resource limits
    max_memory_gb: float = Field(default=4.0, ge=0.5)
    max_cpu_cores: int = Field(default=4, ge=1)
    training_timeout: int = Field(default=3600, ge=60)  # seconds

    # Experiment configuration
    experiments: List[MLExperimentConfig] = Field(default_factory=list)

    # Deployment configuration
    deployment_environment: str = Field(default="development", pattern="^(development|staging|production)$")
    auto_deploy: bool = False
    deployment_approval_required: bool = True

    class Config:
        json_encoders = {Path: str}
        use_enum_values = True

    @validator('data_directory', 'model_directory', 'artifacts_directory')
    def ensure_path_exists(cls, v):
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'MLConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)

    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path objects
        config_dict = self.dict()
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def get_experiment_config(self, experiment_name: str) -> Optional[MLExperimentConfig]:
        """Get experiment configuration by name."""
        for exp in self.experiments:
            if exp.name == experiment_name:
                return exp
        return None

    def add_experiment(self, experiment: MLExperimentConfig) -> None:
        """Add new experiment configuration."""
        # Check for duplicate names
        existing_names = [exp.name for exp in self.experiments]
        if experiment.name in existing_names:
            raise ValueError(f"Experiment with name '{experiment.name}' already exists")

        self.experiments.append(experiment)

    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate the entire configuration and return any issues."""
        issues = {"errors": [], "warnings": []}

        # Check directory permissions
        for directory in [self.data_directory, self.model_directory, self.artifacts_directory]:
            if not os.access(directory, os.R_OK | os.W_OK):
                issues["errors"].append(f"Insufficient permissions for directory: {directory}")

        # Check resource limits
        if self.max_memory_gb > 32.0:
            issues["warnings"].append(f"Memory limit is very high: {self.max_memory_gb}GB")

        if self.max_cpu_cores > os.cpu_count():
            issues["warnings"].append(
                f"CPU cores limit ({self.max_cpu_cores}) exceeds system cores ({os.cpu_count()})"
            )

        # Validate experiments
        experiment_names = [exp.name for exp in self.experiments]
        if len(experiment_names) != len(set(experiment_names)):
            issues["errors"].append("Duplicate experiment names found")

        # Check deployment configuration
        if self.deployment_environment == "production" and not self.deployment_approval_required:
            issues["warnings"].append("Production deployment without approval required")

        return issues


@dataclass
class MLPerformanceMetrics:
    """Performance metrics for model evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    mean_squared_error: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    training_time: float = 0.0
    inference_time: float = 0.0

    def meets_requirements(self, config: MLExperimentConfig) -> bool:
        """Check if metrics meet minimum requirements."""
        return (
            self.accuracy >= config.min_accuracy and
            self.precision >= config.min_precision and
            self.recall >= config.min_recall
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "mean_squared_error": self.mean_squared_error,
            "mean_absolute_error": self.mean_absolute_error,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
        }


def create_default_config(project_name: str) -> MLConfig:
    """Create a default ML configuration for a project."""

    # Default classification experiment
    classification_config = MLExperimentConfig(
        name="document_classification",
        description="Document classification using vector embeddings",
        model_configs=[
            MLModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                task_type=MLTaskType.CLASSIFICATION,
                hyperparameters={
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                feature_selection_method=FeatureSelectionMethod.TREE_IMPORTANCE,
                max_features=1000
            ),
            MLModelConfig(
                model_type=ModelType.GRADIENT_BOOSTING,
                task_type=MLTaskType.CLASSIFICATION,
                hyperparameters={
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1, 0.15],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0],
                },
                feature_selection_method=FeatureSelectionMethod.TREE_IMPORTANCE,
                max_features=1000
            ),
        ],
        scoring_metric="f1_weighted",
        cv_folds=5,
        max_iter=50
    )

    return MLConfig(
        project_name=project_name,
        experiments=[classification_config],
        enable_monitoring=True,
        deployment_environment="development"
    )