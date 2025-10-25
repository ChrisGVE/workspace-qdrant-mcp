"""
Advanced Data Analytics and Intelligence Framework for workspace-qdrant-mcp.

This module provides comprehensive analytics capabilities for document processing insights,
intelligent pattern recognition, predictive analytics, and data visualization.

Key Components:
    - AnalyticsEngine: Statistical analysis and performance metrics
    - IntelligenceFramework: Pattern recognition and ML-based insights
    - PredictiveAnalytics: Forecasting and system optimization
    - AnomalyDetector: Real-time anomaly detection and alerting
    - VisualizationFramework: Interactive dashboards and reporting

Features:
    - Advanced statistical analysis of search patterns and document metrics
    - ML-powered pattern recognition for user behavior and content trends
    - Predictive modeling for system performance and resource optimization
    - Real-time anomaly detection with configurable thresholds
    - Rich data visualizations with interactive dashboards
    - Production-ready with comprehensive error handling and edge case coverage
"""

from .core.analytics_engine import AnalyticsEngine
from .core.anomaly_detector import AnomalyDetector
from .core.intelligence_framework import IntelligenceFramework
from .core.metrics_collector import MetricsCollector
from .core.predictive_analytics import PredictiveAnalytics
from .models.analytics_models import (
    AnalyticsMetrics,
    AnomalyAlert,
    DocumentInsight,
    PerformanceMetric,
    PredictionResult,
    SearchPattern,
)
from .visualization.dashboard_generator import DashboardGenerator

__all__ = [
    "AnalyticsEngine",
    "IntelligenceFramework",
    "PredictiveAnalytics",
    "AnomalyDetector",
    "MetricsCollector",
    "DashboardGenerator",
    "AnalyticsMetrics",
    "SearchPattern",
    "DocumentInsight",
    "PerformanceMetric",
    "AnomalyAlert",
    "PredictionResult"
]

__version__ = "0.1.0"
