"""
Machine Learning Pipeline and Model Management System

This module provides comprehensive ML capabilities including:
- Automated training pipelines with hyperparameter tuning
- Model versioning and deployment management
- Feature engineering and selection
- Performance monitoring and drift detection
"""

from .config.ml_config import MLConfig, MLExperimentConfig, MLModelConfig
from .management.deployment_manager import DeploymentManager
from .management.model_registry import ModelRegistry
from .monitoring.model_monitor import ModelMonitor
from .pipeline.training_pipeline import TrainingPipeline

__all__ = [
    "MLConfig",
    "MLExperimentConfig",
    "MLModelConfig",
    "TrainingPipeline",
    "ModelRegistry",
    "DeploymentManager",
    "ModelMonitor",
]

__version__ = "1.0.0"
