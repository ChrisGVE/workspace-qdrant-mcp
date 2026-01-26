"""Unit tests for analytics dashboard system."""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add docs framework to path for testing (if present)
docs_framework_path = os.path.join(os.path.dirname(__file__), '../../docs/framework')
sys.path.insert(0, docs_framework_path)

try:
    from analytics.dashboard import (
        AnalyticsDashboard,
        ChartData,
        DashboardData,
        DashboardMetric,
    )
    from analytics.privacy import ConsentLevel, PrivacyManager
    from analytics.storage import AnalyticsEvent, AnalyticsStats, AnalyticsStorage
except ModuleNotFoundError:
    pytest.skip("analytics framework not available", allow_module_level=True)


class TestDashboardMetric:
    """Test DashboardMetric data class."""

    def test_basic_metric(self):
        """Test basic metric creation."""
        metric = DashboardMetric(
            name="Page Views",
            value=150,
            unit="views"
        )

        assert metric.name == "Page Views"
        assert metric.value == 150
        assert metric.unit == "views"
        assert metric.change_percent is None
        assert metric.trend is None

    def test_metric_with_trend(self):
        """Test metric with trend information."""
        metric = DashboardMetric(
            name="Sessions",
            value=45,
            unit="sessions",
            change_percent=15.5,
            trend="up"
        )

        assert metric.change_percent == 15.5
        assert metric.trend == "up"


class TestChartData:
    """Test ChartData data class."""

    def test_basic_chart(self):
        """Test basic chart data creation."""
        chart = ChartData(
            chart_type="line",
            title="Page Views Over Time",
            labels=["Day 1", "Day 2", "Day 3"],
            datasets=[{
                "label": "Views",
                "data": [10, 20, 15]
            }]
        )

        assert chart.chart_type == "line"
        assert chart.title == "Page Views Over Time"
        assert chart.labels == ["Day 1", "Day 2", "Day 3"]
        assert len(chart.datasets) == 1
        assert chart.options is None

    def test_chart_with_options(self):
        """Test chart data with options."""
        options = {"responsive": True}
        chart = ChartData(
            chart_type="bar",
            title="Test Chart",
            labels=[],
            datasets=[],
            options=options
        )

        assert chart.options == options


class TestDashboardData:
    """Test DashboardData data class."""

    def test_dashboard_data_creation(self):
        """Test dashboard data creation."""
        now = datetime.now()
        start_date = now - timedelta(days=30)

        dashboard = DashboardData(
            generated_at=now,
            date_range=(start_date, now),
            summary_metrics=[],
            charts=[],
            tables=[],
            insights=[]
        )

        assert dashboard.generated_at == now
        assert dashboard.date_range == (start_date, now)
        assert dashboard.summary_metrics == []
        assert dashboard.charts == []
        assert dashboard.tables == []
        assert dashboard.insights == []


class TestAnalyticsDashboard:
    """Test analytics dashboard functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_analytics.db"
        self.storage = AnalyticsStorage(self.db_path, retention_days=30)
        self.privacy_manager = Mock()
        self.privacy_manager.get_consent_level.return_value = ConsentLevel.ALL
        self.dashboard = AnalyticsDashboard(self.storage, self.privacy_manager)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test dashboard initialization."""
        assert self.dashboard.storage == self.storage
        assert self.dashboard.privacy_manager == self.privacy_manager

    def test_initialization_without_privacy_manager(self):
        """Test dashboard initialization without privacy manager."""
        dashboard = AnalyticsDashboard(self.storage)
        assert dashboard.privacy_manager is None

    def test_generate_dashboard_success(self):
        """Test successful dashboard generation."""
        # Add test data
        base_time = datetime.now()
        events = [
            AnalyticsEvent("page_view", base_time, "session1", "/page1", duration_ms=1000),
            AnalyticsEvent("page_view", base_time, "session2", "/page2", duration_ms=2000),
            AnalyticsEvent("search", base_time, "session1", "/page1", metadata={"query": "test"}),
            AnalyticsEvent("error", base_time, "session2", "/page2")
        ]

        for event in events:
            self.storage.store_event(event)

        # Generate dashboard
        dashboard_data = self.dashboard.generate_dashboard()

        assert dashboard_data is not None
        assert isinstance(dashboard_data, DashboardData)
        assert len(dashboard_data.summary_metrics) > 0
        assert len(dashboard_data.charts) > 0
        assert isinstance(dashboard_data.insights, list)

    def test_generate_dashboard_with_date_range(self):
        """Test dashboard generation with custom date range."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        # Add events in and out of range
        events = [
            AnalyticsEvent("page_view", start_date + timedelta(days=1), "session1", "/page1"),
            AnalyticsEvent("page_view", start_date - timedelta(days=1), "session2", "/page2"),  # Out of range
        ]

        for event in events:
            self.storage.store_event(event)

        dashboard_data = self.dashboard.generate_dashboard(start_date, end_date)

        assert dashboard_data is not None
        assert dashboard_data.date_range == (start_date, end_date)

    def test_generate_dashboard_no_stats(self):
        """Test dashboard generation when no stats available."""
        with patch.object(self.storage, 'get_stats') as mock_stats:
            mock_stats.return_value = None

            dashboard_data = self.dashboard.generate_dashboard()
            assert dashboard_data is None

    def test_generate_dashboard_exception(self):
        """Test dashboard generation with exception."""
        with patch.object(self.storage, 'get_stats') as mock_stats:
            mock_stats.side_effect = Exception("Database error")

            dashboard_data = self.dashboard.generate_dashboard()
            assert dashboard_data is None

    def test_generate_summary_metrics(self):
        """Test summary metrics generation."""
        stats = AnalyticsStats(
            total_events=100,
            unique_sessions=25,
            total_page_views=75,
            avg_session_duration_ms=150000,  # 2.5 minutes
            top_pages=[],
            top_search_queries=[],
            error_rate=5.0
        )

        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        metrics = self.dashboard._generate_summary_metrics(stats, start_date, end_date)

        assert len(metrics) >= 5

        # Check specific metrics
        page_views_metric = next((m for m in metrics if m.name == "Total Page Views"), None)
        assert page_views_metric is not None
        assert page_views_metric.value == 75
        assert page_views_metric.unit == "views"

        sessions_metric = next((m for m in metrics if m.name == "Unique Sessions"), None)
        assert sessions_metric is not None
        assert sessions_metric.value == 25

        duration_metric = next((m for m in metrics if "Duration" in m.name), None)
        assert duration_metric is not None
        assert duration_metric.unit == "minutes"

    def test_generate_charts_with_consent(self):
        """Test chart generation respecting consent levels."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        # Test with FUNCTIONAL consent
        charts = self.dashboard._generate_charts(start_date, end_date, ConsentLevel.FUNCTIONAL)
        [chart.chart_type for chart in charts]

        # Should include page views and top pages, but not search queries
        assert any("page" in chart.title.lower() for chart in charts)

        # Test with ANALYTICS consent
        charts = self.dashboard._generate_charts(start_date, end_date, ConsentLevel.ANALYTICS)

        # Should include search charts
        [chart.title.lower() for chart in charts]
        # Note: search charts might not appear if no search data exists

    def test_generate_page_views_chart(self):
        """Test page views chart generation."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("page_view", base_time.replace(hour=10), "session1", "/page1"),
            AnalyticsEvent("page_view", base_time.replace(hour=11), "session2", "/page2"),
            AnalyticsEvent("page_view", base_time.replace(hour=12), "session1", "/page1"),
        ]

        for event in events:
            self.storage.store_event(event)

        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        chart = self.dashboard._generate_page_views_chart(start_date, end_date)

        assert chart is not None
        assert chart.chart_type == "line"
        assert "Page Views Over Time" in chart.title
        assert len(chart.labels) >= 1
        assert len(chart.datasets) == 1

    def test_generate_page_views_chart_no_data(self):
        """Test page views chart generation with no data."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()

        chart = self.dashboard._generate_page_views_chart(start_date, end_date)

        # Should return None when no data available
        assert chart is None

    def test_get_daily_page_views(self):
        """Test daily page views aggregation."""
        base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)

        events = [
            AnalyticsEvent("page_view", base_time, "session1", "/page1"),
            AnalyticsEvent("page_view", base_time + timedelta(hours=1), "session2", "/page2"),
            AnalyticsEvent("page_view", base_time + timedelta(days=1), "session3", "/page3"),
        ]

        for event in events:
            self.storage.store_event(event)

        daily_views = self.dashboard._get_daily_page_views(
            base_time - timedelta(days=1),
            base_time + timedelta(days=2)
        )

        today = base_time.date()
        tomorrow = (base_time + timedelta(days=1)).date()

        assert daily_views[today] == 2  # Two views on base day
        assert daily_views[tomorrow] == 1  # One view on next day

    def test_get_daily_page_views_exception(self):
        """Test daily page views with exception."""
        with patch.object(self.storage, 'get_events') as mock_events:
            mock_events.side_effect = Exception("Database error")

            result = self.dashboard._get_daily_page_views(datetime.now(), datetime.now())
            assert result == {}

    def test_generate_top_pages_chart(self):
        """Test top pages chart generation."""
        # Mock stats with top pages
        stats = AnalyticsStats(
            total_events=100,
            unique_sessions=25,
            total_page_views=75,
            avg_session_duration_ms=150000,
            top_pages=[
                {"page": "/home", "views": 50},
                {"page": "/about", "views": 25},
                {"page": "/contact", "views": 10}
            ],
            top_search_queries=[],
            error_rate=5.0
        )

        with patch.object(self.storage, 'get_stats') as mock_stats:
            mock_stats.return_value = stats

            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()

            chart = self.dashboard._generate_top_pages_chart(start_date, end_date)

            assert chart is not None
            assert chart.chart_type == "bar"
            assert "Top Pages" in chart.title
            assert chart.labels == ["/home", "/about", "/contact"]
            assert chart.datasets[0]["data"] == [50, 25, 10]

    def test_generate_top_pages_chart_no_data(self):
        """Test top pages chart with no data."""
        with patch.object(self.storage, 'get_stats') as mock_stats:
            mock_stats.return_value = None

            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()

            chart = self.dashboard._generate_top_pages_chart(start_date, end_date)
            assert chart is None

    def test_generate_search_queries_chart(self):
        """Test search queries chart generation."""
        stats = AnalyticsStats(
            total_events=100,
            unique_sessions=25,
            total_page_views=75,
            avg_session_duration_ms=150000,
            top_pages=[],
            top_search_queries=[
                {"query": "documentation", "count": 20},
                {"query": "api", "count": 15},
                {"query": "tutorial", "count": 10}
            ],
            error_rate=5.0
        )

        with patch.object(self.storage, 'get_stats') as mock_stats:
            mock_stats.return_value = stats

            start_date = datetime.now() - timedelta(days=7)
            end_date = datetime.now()

            chart = self.dashboard._generate_search_queries_chart(start_date, end_date)

            assert chart is not None
            assert chart.chart_type == "doughnut"
            assert "Search Queries" in chart.title
            assert chart.labels == ["documentation", "api", "tutorial"]
            assert chart.datasets[0]["data"] == [20, 15, 10]

    def test_generate_error_types_chart(self):
        """Test error types chart generation."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("error", base_time, "session1", "/page1", metadata={"error_type": "JavaScript"}),
            AnalyticsEvent("error", base_time, "session2", "/page2", metadata={"error_type": "Python"}),
            AnalyticsEvent("error", base_time, "session3", "/page3", metadata={"error_type": "JavaScript"}),
        ]

        for event in events:
            self.storage.store_event(event)

        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        chart = self.dashboard._generate_error_types_chart(start_date, end_date)

        assert chart is not None
        assert chart.chart_type == "bar"
        assert "Error Types" in chart.title
        assert "JavaScript" in chart.labels
        assert "Python" in chart.labels

    def test_generate_error_types_chart_no_data(self):
        """Test error types chart with no data."""
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()

        chart = self.dashboard._generate_error_types_chart(start_date, end_date)
        assert chart is None

    def test_generate_tables(self):
        """Test table generation."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        # Add some error events
        base_time = datetime.now()
        events = [
            AnalyticsEvent("error", base_time, "session1", "/page1", metadata={"error_type": "JavaScript", "error_message": "Test error"}),
        ]

        for event in events:
            self.storage.store_event(event)

        tables = self.dashboard._generate_tables(start_date, end_date, ConsentLevel.ALL)

        assert isinstance(tables, list)
        # Should have at least errors table
        assert len(tables) >= 1

        # Check error table structure
        error_table = next((table for table in tables if "Error" in table.get("title", "")), None)
        if error_table:
            assert "columns" in error_table
            assert "rows" in error_table

    def test_generate_recent_errors_table(self):
        """Test recent errors table generation."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("error", base_time, "session1", "/page1",
                         metadata={"error_type": "JavaScript", "error_message": "ReferenceError: x is not defined"}),
            AnalyticsEvent("error", base_time - timedelta(minutes=5), "session2", "/page2",
                         metadata={"error_type": "Python", "error_message": "NameError: name 'y' is not defined"}),
        ]

        for event in events:
            self.storage.store_event(event)

        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        table = self.dashboard._generate_recent_errors_table(start_date, end_date)

        assert table is not None
        assert table["title"] == "Recent Errors"
        assert len(table["columns"]) == 4
        assert len(table["rows"]) == 2

        # Check column structure
        column_keys = [col["key"] for col in table["columns"]]
        assert "timestamp" in column_keys
        assert "page" in column_keys
        assert "error_type" in column_keys
        assert "error_message" in column_keys

    def test_generate_sessions_table(self):
        """Test sessions table generation."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("page_view", base_time, "session1", "/page1", duration_ms=1000),
            AnalyticsEvent("page_view", base_time + timedelta(minutes=1), "session1", "/page2", duration_ms=2000),
            AnalyticsEvent("page_view", base_time, "session2", "/page3", duration_ms=500),
        ]

        for event in events:
            self.storage.store_event(event)

        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        table = self.dashboard._generate_sessions_table(start_date, end_date)

        assert table is not None
        assert table["title"] == "Recent Sessions"
        assert len(table["columns"]) == 6
        assert len(table["rows"]) == 2  # Two unique sessions

        # Check that session data is aggregated correctly
        session1_row = next((row for row in table["rows"] if row["page_views"] == 2), None)
        assert session1_row is not None

    def test_generate_insights(self):
        """Test insights generation."""
        stats = AnalyticsStats(
            total_events=100,
            unique_sessions=25,
            total_page_views=75,
            avg_session_duration_ms=600000,  # 10 minutes
            top_pages=[
                {"page": "/popular", "views": 50},
                {"page": "/other", "views": 25}
            ],
            top_search_queries=[
                {"query": "test", "count": 30}
            ],
            error_rate=2.0
        )

        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        insights = self.dashboard._generate_insights(stats, start_date, end_date)

        assert isinstance(insights, list)
        assert len(insights) > 0

        # Should include insights about session duration, error rate, etc.
        insight_text = " ".join(insights).lower()
        assert "session" in insight_text or "error" in insight_text or "page" in insight_text

    def test_generate_insights_edge_cases(self):
        """Test insights generation with edge cases."""
        # Very short sessions
        stats_short = AnalyticsStats(
            total_events=100,
            unique_sessions=50,
            total_page_views=60,
            avg_session_duration_ms=60000,  # 1 minute
            top_pages=[],
            top_search_queries=[],
            error_rate=0.5
        )

        insights = self.dashboard._generate_insights(stats_short, datetime.now(), datetime.now())
        assert len(insights) > 0

        # High error rate
        stats_errors = AnalyticsStats(
            total_events=100,
            unique_sessions=25,
            total_page_views=75,
            avg_session_duration_ms=300000,
            top_pages=[],
            top_search_queries=[],
            error_rate=10.0
        )

        insights = self.dashboard._generate_insights(stats_errors, datetime.now(), datetime.now())
        assert any("error" in insight.lower() for insight in insights)

    def test_export_dashboard_data_success(self):
        """Test successful dashboard data export."""
        # Add test data
        base_time = datetime.now()
        event = AnalyticsEvent("page_view", base_time, "session1", "/page1")
        self.storage.store_event(event)

        export_path = Path(self.temp_dir) / "dashboard_export.json"
        result = self.dashboard.export_dashboard_data(export_path)

        assert result is True
        assert export_path.exists()

        # Verify exported data structure
        with open(export_path, encoding='utf-8') as f:
            data = json.load(f)

        assert "generated_at" in data
        assert "date_range" in data
        assert "summary_metrics" in data
        assert "charts" in data
        assert "tables" in data
        assert "insights" in data

    def test_export_dashboard_data_with_date_range(self):
        """Test dashboard export with date range."""
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        export_path = Path(self.temp_dir) / "dashboard_export_range.json"
        result = self.dashboard.export_dashboard_data(export_path, start_date, end_date)

        assert result is True

        with open(export_path, encoding='utf-8') as f:
            data = json.load(f)

        assert data["date_range"]["start"] == start_date.isoformat()
        assert data["date_range"]["end"] == end_date.isoformat()

    def test_export_dashboard_data_no_data(self):
        """Test dashboard export when no data can be generated."""
        with patch.object(self.dashboard, 'generate_dashboard') as mock_generate:
            mock_generate.return_value = None

            export_path = Path(self.temp_dir) / "dashboard_export_fail.json"
            result = self.dashboard.export_dashboard_data(export_path)

            assert result is False

    def test_export_dashboard_data_file_error(self):
        """Test dashboard export with file write error."""
        # Add minimal data
        event = AnalyticsEvent("page_view", datetime.now(), "session1", "/page1")
        self.storage.store_event(event)

        # Try to export to non-existent directory
        export_path = Path("/non/existent/directory/export.json")
        result = self.dashboard.export_dashboard_data(export_path)

        assert result is False

    def test_calculate_trend_success(self):
        """Test trend calculation with sufficient data."""
        base_time = datetime.now()

        # Add events for current and previous periods
        current_events = [
            AnalyticsEvent("page_view", base_time, "session1", "/page1"),
            AnalyticsEvent("page_view", base_time, "session2", "/page2"),
        ]

        previous_events = [
            AnalyticsEvent("page_view", base_time - timedelta(days=7), "session3", "/page3"),
        ]

        for event in current_events + previous_events:
            self.storage.store_event(event)

        # Test trend calculation
        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        trend = self.dashboard._calculate_trend(2, start_date, end_date, 'page_view')
        assert trend == "up"  # 2 vs 1 is an increase

    def test_calculate_trend_no_previous_data(self):
        """Test trend calculation with no previous data."""
        base_time = datetime.now()
        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        with patch.object(self.storage, 'get_stats') as mock_stats:
            mock_stats.return_value = None

            trend = self.dashboard._calculate_trend(10, start_date, end_date, 'page_view')
            assert trend is None

    def test_dashboard_with_privacy_constraints(self):
        """Test dashboard generation with privacy constraints."""
        # Set up privacy manager to return limited consent
        self.privacy_manager.get_consent_level.return_value = ConsentLevel.ESSENTIAL

        # Add test data
        event = AnalyticsEvent("error", datetime.now(), "session1", "/page1", metadata={"error_type": "test"})
        self.storage.store_event(event)

        dashboard_data = self.dashboard.generate_dashboard()

        assert dashboard_data is not None

        # With ESSENTIAL consent, should have limited charts
        [chart.title.lower() for chart in dashboard_data.charts]
        # Should include error charts (allowed at ESSENTIAL level)
        # Should NOT include page view charts (not allowed at ESSENTIAL level)

    def test_dashboard_performance_with_large_dataset(self):
        """Test dashboard performance with larger dataset."""
        base_time = datetime.now()

        # Generate larger dataset
        events = []
        for i in range(100):
            events.append(AnalyticsEvent(
                "page_view",
                base_time - timedelta(minutes=i),
                f"session{i % 20}",
                f"/page{i % 10}",
                duration_ms=1000 + i
            ))

        for event in events:
            self.storage.store_event(event)

        # Should still generate dashboard efficiently
        dashboard_data = self.dashboard.generate_dashboard()

        assert dashboard_data is not None
        assert len(dashboard_data.summary_metrics) > 0
        assert len(dashboard_data.charts) > 0

    def test_unicode_handling_in_dashboard(self):
        """Test dashboard handling of unicode data."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("search", base_time, "session1", "/search", metadata={"query": "测试 search 🔍"}),
            AnalyticsEvent("error", base_time, "session2", "/page", metadata={"error_type": "UnicodeError", "error_message": "Unicode тест error"}),
        ]

        for event in events:
            self.storage.store_event(event)

        dashboard_data = self.dashboard.generate_dashboard()

        assert dashboard_data is not None

        # Export should handle unicode correctly
        export_path = Path(self.temp_dir) / "unicode_dashboard.json"
        result = self.dashboard.export_dashboard_data(export_path)
        assert result is True

        # Verify unicode is preserved
        with open(export_path, encoding='utf-8') as f:
            content = f.read()
            assert "测试" in content or "тест" in content  # Unicode characters should be preserved
