"""
Intelligence Framework for ML-based insights and analysis.

This module provides machine learning-based intelligence capabilities including:
- Automated pattern discovery and insights generation
- Smart trend analysis and forecasting recommendations
- Data quality assessment with actionable suggestions
- Correlation analysis and feature importance detection
- Learning models that adapt to data patterns
- Intelligent preprocessing recommendations
"""

from .ml_insights import (
    MLInsightEngine,
    Insight,
    InsightReport,
    InsightType,
    ConfidenceLevel,
    DataQualityIssue
)
from .trend_analyzer import (
    AdvancedTrendAnalyzer,
    TrendAnalysisResult,
    TrendSegment,
    ChangePoint,
    TrendDirection,
    TrendStrength,
    TrendType,
    ChangePointType
)

__all__ = [
    # ML Insights
    'MLInsightEngine',
    'Insight',
    'InsightReport',
    'InsightType',
    'ConfidenceLevel',
    'DataQualityIssue',

    # Trend Analysis
    'AdvancedTrendAnalyzer',
    'TrendAnalysisResult',
    'TrendSegment',
    'ChangePoint',
    'TrendDirection',
    'TrendStrength',
    'TrendType',
    'ChangePointType'
]