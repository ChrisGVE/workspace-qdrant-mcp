"""
Data models for the analytics framework.

This module defines Pydantic models for structured data used throughout
the analytics and intelligence system.
"""

from .analytics_models import (
    AnalyticsMetrics,
    AnomalyAlert,
    DocumentAnalytics,
    DocumentInsight,
    PerformanceMetric,
    SearchAnalytics,
    SearchPattern,
    SystemPerformance,
    UserBehaviorPattern,
)
from .prediction_models import (
    AnomalyPrediction,
    MLModelMetrics,
    PerformancePrediction,
    PredictionResult,
    ResourceUsagePrediction,
    TimeSeriesForecast,
)

__all__ = [
    "AnalyticsMetrics",
    "SearchPattern",
    "DocumentInsight",
    "PerformanceMetric",
    "AnomalyAlert",
    "SearchAnalytics",
    "DocumentAnalytics",
    "UserBehaviorPattern",
    "SystemPerformance",
    "PredictionResult",
    "TimeSeriesForecast",
    "PerformancePrediction",
    "ResourceUsagePrediction",
    "AnomalyPrediction",
    "MLModelMetrics"
]
