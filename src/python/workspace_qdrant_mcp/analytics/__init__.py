"""
Data Analytics and Intelligence Framework for workspace-qdrant-mcp.

This module provides comprehensive analytics capabilities including:
- Statistical analysis and data processing
- Pattern recognition and trend analysis
- Predictive analytics and forecasting
- Data visualization and dashboards
- Anomaly detection and alerting
- ML-based intelligence and insights
"""

from .engine.statistical_engine import StatisticalEngine
from .engine.pattern_recognition import PatternRecognition
from .engine.predictive_models import PredictiveModels
from .visualization.chart_builders import ChartBuilder
from .anomaly.detection_algorithms import AnomalyDetector, AnomalyResult
from .anomaly.alerting_system import (
    AlertingSystem, AlertRule, Alert, AlertSeverity, AlertType, AlertStatus,
    EmailNotificationChannel, WebhookNotificationChannel, SlackNotificationChannel
)
from .intelligence.ml_insights import (
    MLInsightEngine, Insight, InsightReport, InsightType,
    ConfidenceLevel, DataQualityIssue
)

__all__ = [
    # Core Analytics Engine
    'StatisticalEngine',
    'PatternRecognition',
    'PredictiveModels',

    # Data Visualization
    'ChartBuilder',

    # Anomaly Detection & Alerting
    'AnomalyDetector',
    'AnomalyResult',
    'AlertingSystem',
    'AlertRule',
    'Alert',
    'AlertSeverity',
    'AlertType',
    'AlertStatus',
    'EmailNotificationChannel',
    'WebhookNotificationChannel',
    'SlackNotificationChannel',

    # ML-based Intelligence Framework
    'MLInsightEngine',
    'Insight',
    'InsightReport',
    'InsightType',
    'ConfidenceLevel',
    'DataQualityIssue'
]