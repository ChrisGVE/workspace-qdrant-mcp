"""Documentation analytics system for tracking usage and improving documentation quality."""

from .collector import AnalyticsCollector, EventType
from .storage import AnalyticsStorage
from .dashboard import AnalyticsDashboard
from .privacy import PrivacyManager

__all__ = [
    'AnalyticsCollector',
    'EventType',
    'AnalyticsStorage',
    'AnalyticsDashboard',
    'PrivacyManager'
]