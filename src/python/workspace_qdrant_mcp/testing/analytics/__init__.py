"""
Test Analytics Framework

Comprehensive test analytics system providing result processing, trend analysis,
quality metrics calculation, and dashboard generation with robust error handling.

Components:
- TestAnalyticsEngine: Core analytics processing
- DashboardGenerator: Interactive dashboard creation
- ChartGenerator: Visualization chart generation
"""

from .engine import (
    TestAnalyticsEngine,
    TestResult,
    TestMetrics,
    TrendAnalysis,
    QualityReport,
    MetricType,
    TrendDirection
)

from .dashboard import (
    DashboardGenerator,
    ChartGenerator,
    ChartConfig,
    DashboardSection
)

__all__ = [
    "TestAnalyticsEngine",
    "TestResult",
    "TestMetrics",
    "TrendAnalysis",
    "QualityReport",
    "MetricType",
    "TrendDirection",
    "DashboardGenerator",
    "ChartGenerator",
    "ChartConfig",
    "DashboardSection"
]

__version__ = "1.0.0"