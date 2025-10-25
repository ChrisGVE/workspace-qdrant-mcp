"""
Data models for predictive analytics and machine learning.

This module defines models for predictions, forecasts, and ML model metrics
used in the predictive analytics system.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class PredictionType(str, Enum):
    """Types of predictions."""
    TIME_SERIES = "time_series"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANOMALY = "anomaly"
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"


class ModelType(str, Enum):
    """Types of ML models."""
    LINEAR_REGRESSION = "linear_regression"
    ARIMA = "arima"
    PROPHET = "prophet"
    RANDOM_FOREST = "random_forest"
    LSTM = "lstm"
    ISOLATION_FOREST = "isolation_forest"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"


class ConfidenceLevel(str, Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PredictionResult(BaseModel):
    """Base model for prediction results."""

    prediction_id: str = Field(..., description="Unique prediction identifier")
    prediction_type: PredictionType = Field(..., description="Type of prediction")
    model_type: ModelType = Field(..., description="ML model used")
    predicted_value: float | int | str | bool = Field(..., description="Predicted value")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence in prediction (0-1)")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level category")
    prediction_date: datetime = Field(default_factory=datetime.now, description="When prediction was made")
    target_date: datetime | None = Field(None, description="Date/time prediction refers to")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional prediction context")
    features_used: list[str] = Field(default_factory=list, description="Features used for prediction")


class TimeSeriesForecast(BaseModel):
    """Model for time series forecasting results."""

    forecast_id: str = Field(..., description="Unique forecast identifier")
    metric_name: str = Field(..., description="Name of the metric being forecasted")
    time_points: list[datetime] = Field(..., description="Time points for forecast")
    predicted_values: list[float] = Field(..., description="Predicted values")
    lower_bounds: list[float] = Field(..., description="Lower confidence bounds")
    upper_bounds: list[float] = Field(..., description="Upper confidence bounds")
    confidence_intervals: list[float] = Field(..., description="Confidence interval widths")
    model_type: ModelType = Field(..., description="Forecasting model used")
    forecast_horizon: int = Field(..., description="Number of time steps forecasted")
    model_accuracy: dict[str, float] = Field(default_factory=dict, description="Model accuracy metrics")
    seasonal_components: dict[str, list[float]] | None = Field(None, description="Seasonal decomposition")
    trend_component: list[float] | None = Field(None, description="Trend component")

    @validator('predicted_values', 'lower_bounds', 'upper_bounds', 'confidence_intervals')
    def validate_lists_same_length(cls, v, values):
        """Ensure all forecast arrays have same length as time_points."""
        if 'time_points' in values and len(v) != len(values['time_points']):
            raise ValueError("All forecast arrays must have same length as time_points")
        return v


class PerformancePrediction(BaseModel):
    """Model for system performance predictions."""

    component: str = Field(..., description="System component being predicted")
    metric_name: str = Field(..., description="Performance metric name")
    current_value: float = Field(..., description="Current metric value")
    predicted_value: float = Field(..., description="Predicted future value")
    predicted_change: float = Field(..., description="Predicted change from current")
    predicted_change_percent: float = Field(..., description="Predicted percentage change")
    time_horizon: timedelta = Field(..., description="Time horizon for prediction")
    risk_level: str = Field(..., description="Risk level (low, medium, high)")
    recommended_actions: list[str] = Field(default_factory=list, description="Recommended actions")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")

    @validator('predicted_change_percent')
    def calculate_percentage_change(cls, v, values):
        """Calculate percentage change if not provided."""
        if 'current_value' in values and values['current_value'] != 0:
            return (values.get('predicted_change', 0) / values['current_value']) * 100
        return v


class ResourceUsagePrediction(BaseModel):
    """Model for resource usage predictions."""

    resource_type: str = Field(..., description="Type of resource (cpu, memory, disk, network)")
    current_usage: float = Field(..., description="Current usage percentage")
    predicted_usage: float = Field(..., description="Predicted usage percentage")
    peak_usage_time: datetime | None = Field(None, description="When peak usage is expected")
    capacity_threshold: float = Field(default=85.0, description="Usage threshold for warnings")
    time_to_threshold: timedelta | None = Field(None, description="Time until threshold reached")
    growth_rate: float = Field(..., description="Usage growth rate per time unit")
    recommendations: list[str] = Field(default_factory=list, description="Resource optimization recommendations")
    cost_impact: float | None = Field(None, description="Predicted cost impact")


class AnomalyPrediction(BaseModel):
    """Model for anomaly prediction results."""

    metric_name: str = Field(..., description="Metric being monitored for anomalies")
    anomaly_probability: float = Field(..., ge=0, le=1, description="Probability of anomaly")
    anomaly_type: str = Field(..., description="Type of potential anomaly")
    predicted_time_window: dict[str, datetime] = Field(..., description="Time window for potential anomaly")
    severity_estimate: str = Field(..., description="Expected severity if anomaly occurs")
    contributing_factors: list[str] = Field(default_factory=list, description="Factors that may cause anomaly")
    prevention_strategies: list[str] = Field(default_factory=list, description="Prevention strategies")
    model_confidence: float = Field(..., ge=0, le=1, description="Model confidence in prediction")

    @validator('predicted_time_window')
    def validate_time_window(cls, v):
        """Validate time window has start and end."""
        if 'start' not in v or 'end' not in v:
            raise ValueError("predicted_time_window must have 'start' and 'end' keys")
        if v['start'] >= v['end']:
            raise ValueError("start time must be before end time")
        return v


class MLModelMetrics(BaseModel):
    """Model performance metrics for ML models."""

    model_id: str = Field(..., description="Unique model identifier")
    model_type: ModelType = Field(..., description="Type of ML model")
    training_date: datetime = Field(..., description="When model was trained")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update time")
    accuracy_metrics: dict[str, float] = Field(default_factory=dict, description="Accuracy metrics")
    performance_metrics: dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    feature_importance: dict[str, float] = Field(default_factory=dict, description="Feature importance scores")
    training_data_size: int = Field(..., description="Size of training dataset")
    validation_score: float = Field(..., description="Validation score")
    test_score: float | None = Field(None, description="Test score if available")
    cross_validation_scores: list[float] = Field(default_factory=list, description="Cross-validation scores")
    hyperparameters: dict[str, Any] = Field(default_factory=dict, description="Model hyperparameters")
    model_drift_score: float | None = Field(None, description="Model drift detection score")
    needs_retraining: bool = Field(default=False, description="Whether model needs retraining")
    prediction_count: int = Field(default=0, description="Number of predictions made")
