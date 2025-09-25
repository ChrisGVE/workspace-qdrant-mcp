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
from .visualization.dashboard_generator import DashboardGenerator
from .visualization.chart_builders import ChartBuilder
from .anomaly.detection_algorithms import AnomalyDetector
from .anomaly.alerting_system import AlertingSystem
from .intelligence.insights_generator import InsightsGenerator
from .intelligence.ml_models import MLModels

__all__ = [
    'StatisticalEngine',
    'PatternRecognition',
    'PredictiveModels',
    'DashboardGenerator',
    'ChartBuilder',
    'AnomalyDetector',
    'AlertingSystem',
    'InsightsGenerator',
    'MLModels'
]