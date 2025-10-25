"""
Comprehensive unit tests for DashboardGenerator with edge cases.

Tests chart generation, HTML rendering, error handling, and all visualization
functionality including missing data scenarios.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from src.python.workspace_qdrant_mcp.testing.analytics.dashboard import (
    ChartConfig,
    ChartGenerator,
    DashboardGenerator,
    DashboardSection,
)
from src.python.workspace_qdrant_mcp.testing.analytics.engine import (
    MetricType,
    QualityReport,
    TestAnalyticsEngine,
    TestMetrics,
    TrendAnalysis,
    TrendDirection,
)


class TestChartGenerator:
    """Tests for ChartGenerator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for chart output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def chart_generator(self, temp_dir):
        """Create chart generator with temporary directory."""
        return ChartGenerator(output_dir=temp_dir)

    @pytest.fixture
    def sample_trend(self):
        """Create sample trend analysis data."""
        base_time = datetime.now() - timedelta(days=7)
        data_points = [(base_time + timedelta(days=i), 80.0 + i * 2) for i in range(7)]

        return TrendAnalysis(
            metric_type=MetricType.PASS_RATE,
            direction=TrendDirection.IMPROVING,
            change_percentage=12.0,
            confidence=0.85,
            data_points=data_points,
            regression_slope=2.0,
            anomalies=[base_time + timedelta(days=3)]
        )

    @pytest.fixture
    def sample_metrics(self):
        """Create sample test metrics."""
        return TestMetrics(
            total_tests=100,
            passed=85,
            failed=10,
            skipped=3,
            errors=2,
            pass_rate=85.0,
            avg_execution_time=25.0
        )

    def test_chart_generator_initialization(self, temp_dir):
        """Test chart generator initialization."""
        generator = ChartGenerator(output_dir=temp_dir)
        assert generator.output_dir == temp_dir
        assert generator.output_dir.exists()

    def test_chart_generator_no_plotting_available(self, temp_dir):
        """Test chart generator when plotting libraries are unavailable."""
        with patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', False):
            generator = ChartGenerator(output_dir=temp_dir)

            # Should still initialize but warn about missing libraries
            assert generator.output_dir == temp_dir

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.plt')
    def test_generate_trend_chart_success(self, mock_plt, chart_generator, sample_trend):
        """Test successful trend chart generation."""
        config = ChartConfig(
            chart_type="line",
            title="Test Trend",
            width=800,
            height=600
        )

        # Mock matplotlib functions
        mock_plt.figure.return_value = None
        mock_plt.plot.return_value = None
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None

        chart_path = chart_generator.generate_trend_chart(sample_trend, config)

        assert chart_path is not None
        assert mock_plt.figure.called
        assert mock_plt.plot.called
        assert mock_plt.savefig.called

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', False)
    def test_generate_trend_chart_no_plotting(self, chart_generator, sample_trend):
        """Test trend chart generation when plotting is unavailable."""
        config = ChartConfig(chart_type="line", title="Test Trend")

        chart_path = chart_generator.generate_trend_chart(sample_trend, config)
        assert chart_path is None

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.plt')
    def test_generate_trend_chart_empty_data(self, mock_plt, chart_generator):
        """Test trend chart generation with empty data."""
        empty_trend = TrendAnalysis(
            metric_type=MetricType.PASS_RATE,
            direction=TrendDirection.STABLE,
            change_percentage=0.0,
            confidence=0.0,
            data_points=[]  # Empty data
        )

        config = ChartConfig(chart_type="line", title="Empty Trend")
        chart_path = chart_generator.generate_trend_chart(empty_trend, config)
        assert chart_path is None

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.plt')
    def test_generate_trend_chart_constant_values(self, mock_plt, chart_generator):
        """Test trend chart generation with constant values."""
        constant_trend = TrendAnalysis(
            metric_type=MetricType.PASS_RATE,
            direction=TrendDirection.STABLE,
            change_percentage=0.0,
            confidence=1.0,
            data_points=[(datetime.now() + timedelta(days=i), 50.0) for i in range(5)]
        )

        config = ChartConfig(chart_type="line", title="Constant Trend")

        mock_plt.figure.return_value = None
        mock_plt.axhline.return_value = None
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None

        chart_path = chart_generator.generate_trend_chart(constant_trend, config)

        assert chart_path is not None
        assert mock_plt.axhline.called  # Should use horizontal line for constant values

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.plt')
    def test_generate_trend_chart_with_anomalies(self, mock_plt, chart_generator, sample_trend):
        """Test trend chart generation with anomalies."""
        config = ChartConfig(chart_type="line", title="Trend with Anomalies")

        mock_plt.figure.return_value = None
        mock_plt.plot.return_value = None
        mock_plt.scatter.return_value = None
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None

        chart_path = chart_generator.generate_trend_chart(sample_trend, config)

        assert chart_path is not None
        assert mock_plt.scatter.called  # Should plot anomalies as scatter points

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.plt')
    def test_generate_trend_chart_error(self, mock_plt, chart_generator, sample_trend):
        """Test trend chart generation with matplotlib error."""
        config = ChartConfig(chart_type="line", title="Error Trend")

        # Mock error in matplotlib
        mock_plt.figure.side_effect = Exception("Plotting error")

        chart_path = chart_generator.generate_trend_chart(sample_trend, config)
        assert chart_path is None

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.plt')
    def test_generate_metrics_bar_chart_success(self, mock_plt, chart_generator):
        """Test successful metrics bar chart generation."""
        metrics = {"Pass Rate": 85.0, "Coverage": 90.0, "Performance": 75.0}
        config = ChartConfig(chart_type="bar", title="Test Metrics")

        mock_plt.figure.return_value = None
        mock_plt.bar.return_value = [Mock(), Mock(), Mock()]  # Mock bars
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None

        chart_path = chart_generator.generate_metrics_bar_chart(metrics, config)

        assert chart_path is not None
        assert mock_plt.bar.called

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    def test_generate_metrics_bar_chart_empty_data(self, chart_generator):
        """Test metrics bar chart generation with empty data."""
        config = ChartConfig(chart_type="bar", title="Empty Metrics")

        chart_path = chart_generator.generate_metrics_bar_chart({}, config)
        assert chart_path is None

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    def test_generate_metrics_bar_chart_none_values(self, chart_generator):
        """Test metrics bar chart generation with None values."""
        metrics = {"Metric 1": None, "Metric 2": None}
        config = ChartConfig(chart_type="bar", title="None Metrics")

        chart_path = chart_generator.generate_metrics_bar_chart(metrics, config)
        assert chart_path is None

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.plt')
    def test_generate_metrics_bar_chart_mixed_values(self, mock_plt, chart_generator):
        """Test metrics bar chart generation with mixed valid/None values."""
        metrics = {"Valid": 85.0, "Invalid": None, "Another Valid": 90.0}
        config = ChartConfig(chart_type="bar", title="Mixed Metrics")

        mock_plt.figure.return_value = None
        mock_plt.bar.return_value = [Mock(), Mock(), Mock()]
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None

        chart_path = chart_generator.generate_metrics_bar_chart(metrics, config)

        assert chart_path is not None
        # None values should be replaced with 0
        assert mock_plt.bar.called

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.plt')
    def test_generate_pass_rate_pie_chart_success(self, mock_plt, chart_generator, sample_metrics):
        """Test successful pass rate pie chart generation."""
        config = ChartConfig(chart_type="pie", title="Test Results")

        mock_plt.figure.return_value = None
        mock_plt.pie.return_value = None
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None

        chart_path = chart_generator.generate_pass_rate_pie_chart(sample_metrics, config)

        assert chart_path is not None
        assert mock_plt.pie.called

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    def test_generate_pass_rate_pie_chart_no_results(self, chart_generator):
        """Test pie chart generation with no test results."""
        empty_metrics = TestMetrics()  # All zeros
        config = ChartConfig(chart_type="pie", title="No Results")

        chart_path = chart_generator.generate_pass_rate_pie_chart(empty_metrics, config)
        assert chart_path is None

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', False)
    def test_generate_pass_rate_pie_chart_no_plotting(self, chart_generator, sample_metrics):
        """Test pie chart generation when plotting is unavailable."""
        config = ChartConfig(chart_type="pie", title="No Plotting")

        chart_path = chart_generator.generate_pass_rate_pie_chart(sample_metrics, config)
        assert chart_path is None

    def test_chart_to_base64_success(self, chart_generator, temp_dir):
        """Test successful chart to base64 conversion."""
        # Create a dummy image file
        image_path = temp_dir / "test_chart.png"
        image_path.write_bytes(b"fake_image_data")

        base64_data = chart_generator.chart_to_base64(str(image_path))

        assert base64_data is not None
        assert base64_data.startswith("data:image/png;base64,")

    def test_chart_to_base64_missing_file(self, chart_generator):
        """Test chart to base64 conversion with missing file."""
        base64_data = chart_generator.chart_to_base64("/nonexistent/file.png")
        assert base64_data is None

    def test_chart_to_base64_error(self, chart_generator, temp_dir):
        """Test chart to base64 conversion with error."""
        image_path = temp_dir / "test_chart.png"
        image_path.write_bytes(b"fake_image_data")

        with patch('builtins.open', side_effect=Exception("Read error")):
            base64_data = chart_generator.chart_to_base64(str(image_path))
            assert base64_data is None


class TestDashboardGenerator:
    """Tests for DashboardGenerator class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for dashboard output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_analytics_engine(self):
        """Create mock analytics engine."""
        engine = Mock(spec=TestAnalyticsEngine)

        # Mock methods
        engine.calculate_metrics.return_value = TestMetrics(
            total_tests=100,
            passed=85,
            failed=10,
            skipped=3,
            errors=2,
            pass_rate=85.0,
            avg_execution_time=25.0,
            coverage_percentage=90.0,
            flakiness_score=5.0
        )

        engine.generate_quality_report.return_value = QualityReport(
            overall_score=85.0,
            metrics={MetricType.PASS_RATE: 85.0},
            trends=[],
            recommendations=["Good test coverage"],
            warnings=["Consider reducing execution time"],
            critical_issues=[]
        )

        engine.analyze_trends.return_value = None  # No trends by default

        return engine

    @pytest.fixture
    def dashboard_generator(self, mock_analytics_engine, temp_dir):
        """Create dashboard generator with mock engine."""
        return DashboardGenerator(mock_analytics_engine, output_dir=temp_dir)

    def test_dashboard_generator_initialization(self, mock_analytics_engine, temp_dir):
        """Test dashboard generator initialization."""
        generator = DashboardGenerator(mock_analytics_engine, output_dir=temp_dir)

        assert generator.analytics_engine == mock_analytics_engine
        assert generator.output_dir == temp_dir
        assert generator.output_dir.exists()

    def test_generate_dashboard_success(self, dashboard_generator):
        """Test successful dashboard generation."""
        dashboard_path = dashboard_generator.generate_dashboard(
            title="Test Dashboard",
            period_days=7,
            include_charts=False  # Skip charts to avoid plotting dependencies
        )

        assert dashboard_path is not None
        dashboard_file = Path(dashboard_path)
        assert dashboard_file.exists()
        assert dashboard_file.suffix == '.html'

        # Check content
        content = dashboard_file.read_text()
        assert "Test Dashboard" in content
        assert "<!DOCTYPE html>" in content

    def test_generate_dashboard_with_charts(self, dashboard_generator):
        """Test dashboard generation with charts enabled."""
        # Mock chart generation
        with patch.object(dashboard_generator.chart_generator, 'generate_pass_rate_pie_chart') as mock_pie, \
             patch.object(dashboard_generator.chart_generator, 'generate_trend_chart') as mock_trend, \
             patch.object(dashboard_generator.chart_generator, 'chart_to_base64') as mock_base64:

            mock_pie.return_value = "/fake/chart.png"
            mock_trend.return_value = "/fake/trend.png"
            mock_base64.return_value = "data:image/png;base64,fake_data"

            dashboard_path = dashboard_generator.generate_dashboard(
                title="Chart Dashboard",
                include_charts=True
            )

            assert dashboard_path is not None
            dashboard_file = Path(dashboard_path)
            content = dashboard_file.read_text()

            # Should contain base64 image data
            assert "data:image/png;base64," in content

    def test_generate_dashboard_error(self, dashboard_generator):
        """Test dashboard generation with error."""
        # Mock analytics engine to raise error
        dashboard_generator.analytics_engine.calculate_metrics.side_effect = Exception("Analytics error")

        dashboard_path = dashboard_generator.generate_dashboard()

        # Should generate error dashboard
        assert dashboard_path is not None
        dashboard_file = Path(dashboard_path)
        assert dashboard_file.exists()

        content = dashboard_file.read_text()
        assert "error" in content.lower()

    def test_generate_dashboard_themes(self, dashboard_generator):
        """Test dashboard generation with different themes."""
        # Light theme
        light_path = dashboard_generator.generate_dashboard(theme="light", include_charts=False)
        light_content = Path(light_path).read_text()

        # Dark theme
        dark_path = dashboard_generator.generate_dashboard(theme="dark", include_charts=False)
        dark_content = Path(dark_path).read_text()

        # Should have different CSS
        assert light_content != dark_content

    def test_create_overview_section(self, dashboard_generator):
        """Test overview section creation."""
        metrics = TestMetrics(
            total_tests=100,
            pass_rate=85.0,
            avg_execution_time=25.0,
            coverage_percentage=90.0,
            period_start=datetime.now() - timedelta(days=7),
            period_end=datetime.now()
        )

        section = dashboard_generator._create_overview_section(metrics)

        assert section.title == "Test Suite Overview"
        assert section.content_type == "table"
        assert isinstance(section.content, dict)
        assert "total_tests" in section.content
        assert "pass_rate" in section.content

    def test_create_metrics_cards_section(self, dashboard_generator):
        """Test metrics cards section creation."""
        metrics = TestMetrics(
            total_tests=100,
            pass_rate=85.0,
            failed=15,
            avg_execution_time=25.0,
            coverage_percentage=90.0,
            flakiness_score=5.0
        )

        section = dashboard_generator._create_metrics_cards_section(metrics)

        assert section.title == "Key Metrics"
        assert section.content_type == "cards"
        assert isinstance(section.content, list)
        assert len(section.content) == 6  # Six metric cards

        # Check card colors based on values
        cards = {card['title']: card['color'] for card in section.content}
        assert cards['Pass Rate'] == 'yellow'  # 85% should be yellow
        assert cards['Coverage'] == 'green'    # 90% should be green

    def test_create_quality_section(self, dashboard_generator):
        """Test quality report section creation."""
        quality_report = QualityReport(
            overall_score=85.0,
            metrics={},
            trends=[],
            recommendations=["Good work"],
            warnings=["Minor issue"],
            critical_issues=[]
        )

        section = dashboard_generator._create_quality_section(quality_report)

        assert section.title == "Quality Assessment"
        assert section.content_type == "quality_report"
        assert section.content['overall_score'] == "85.0/100"

    def test_create_trends_section(self, dashboard_generator):
        """Test trends section creation."""
        # Mock trend analysis
        dashboard_generator.analytics_engine.analyze_trends.return_value = TrendAnalysis(
            metric_type=MetricType.PASS_RATE,
            direction=TrendDirection.IMPROVING,
            change_percentage=5.0,
            confidence=0.8
        )

        section = dashboard_generator._create_trends_section(7)

        assert section.title == "Trend Analysis"
        assert section.content_type == "trends"
        assert isinstance(section.content, dict)

    def test_create_trends_section_no_data(self, dashboard_generator):
        """Test trends section with no trend data."""
        # Mock no trends
        dashboard_generator.analytics_engine.analyze_trends.return_value = None

        section = dashboard_generator._create_trends_section(7)

        assert section.content_type == "trends"
        # Should handle empty trends gracefully

    def test_create_recommendations_section(self, dashboard_generator):
        """Test recommendations section creation."""
        quality_report = QualityReport(
            overall_score=70.0,
            metrics={},
            trends=[],
            recommendations=["Improve coverage", "Fix flaky tests"],
            warnings=["Slow tests detected"],
            critical_issues=["High failure rate"]
        )

        section = dashboard_generator._create_recommendations_section(quality_report)

        assert section.title == "Recommendations"
        assert section.content_type == "recommendations"
        assert len(section.content['recommendations']) == 2
        assert len(section.content['warnings']) == 1
        assert len(section.content['critical_issues']) == 1

    def test_render_section_table(self, dashboard_generator):
        """Test table section rendering."""
        section = DashboardSection(
            title="Test Table",
            content_type="table",
            content={"metric_1": "value_1", "metric_2": "value_2"}
        )

        html = dashboard_generator._render_section(section)

        assert '<section class="dashboard-section">' in html
        assert "Test Table" in html
        assert "metric_1" in html
        assert "value_1" in html

    def test_render_section_cards(self, dashboard_generator):
        """Test cards section rendering."""
        section = DashboardSection(
            title="Test Cards",
            content_type="cards",
            content=[
                {"title": "Card 1", "value": "100", "color": "green"},
                {"title": "Card 2", "value": "50", "color": "red"}
            ]
        )

        html = dashboard_generator._render_section(section)

        assert "Test Cards" in html
        assert "Card 1" in html
        assert "Card 2" in html
        assert "green" in html
        assert "red" in html

    def test_render_section_chart(self, dashboard_generator):
        """Test chart section rendering."""
        section = DashboardSection(
            title="Test Chart",
            content_type="chart",
            content={"image_data": "data:image/png;base64,fake", "alt": "Test Chart"}
        )

        html = dashboard_generator._render_section(section)

        assert "Test Chart" in html
        assert "data:image/png;base64,fake" in html
        assert "Test Chart" in html  # Alt text

    def test_render_section_chart_no_data(self, dashboard_generator):
        """Test chart section rendering with no data."""
        section = DashboardSection(
            title="No Chart",
            content_type="chart",
            content={}
        )

        html = dashboard_generator._render_section(section)

        assert "Chart not available" in html

    def test_render_section_quality_report(self, dashboard_generator):
        """Test quality report section rendering."""
        section = DashboardSection(
            title="Quality",
            content_type="quality_report",
            content={
                "overall_score": "85.0/100",
                "critical_issues": 0,
                "warnings": 2,
                "recommendations": 3
            }
        )

        html = dashboard_generator._render_section(section)

        assert "85.0/100" in html
        assert "Critical Issues: 0" in html
        assert "Warnings: 2" in html

    def test_render_section_trends(self, dashboard_generator):
        """Test trends section rendering."""
        section = DashboardSection(
            title="Trends",
            content_type="trends",
            content={
                "pass_rate": {
                    "direction": "improving",
                    "change": "+5.0%",
                    "confidence": "0.80"
                }
            }
        )

        html = dashboard_generator._render_section(section)

        assert "Pass Rate" in html
        assert "Improving" in html
        assert "+5.0%" in html

    def test_render_section_trends_empty(self, dashboard_generator):
        """Test trends section rendering with no data."""
        section = DashboardSection(
            title="No Trends",
            content_type="trends",
            content={}
        )

        html = dashboard_generator._render_section(section)

        assert "No trend data available" in html

    def test_render_section_recommendations(self, dashboard_generator):
        """Test recommendations section rendering."""
        section = DashboardSection(
            title="Recommendations",
            content_type="recommendations",
            content={
                "recommendations": ["Improve coverage"],
                "warnings": ["Slow tests"],
                "critical_issues": ["High failures"]
            }
        )

        html = dashboard_generator._render_section(section)

        assert "Critical Issues" in html
        assert "High failures" in html
        assert "Warnings" in html
        assert "Slow tests" in html
        assert "Recommendations" in html
        assert "Improve coverage" in html

    def test_render_section_recommendations_empty(self, dashboard_generator):
        """Test recommendations section rendering with no data."""
        section = DashboardSection(
            title="No Recommendations",
            content_type="recommendations",
            content={
                "recommendations": [],
                "warnings": [],
                "critical_issues": []
            }
        )

        html = dashboard_generator._render_section(section)

        assert "No recommendations available" in html

    def test_render_section_error(self, dashboard_generator):
        """Test section rendering with error."""
        section = DashboardSection(
            title="Error Section",
            content_type="invalid_type",
            content={}
        )

        # Mock rendering to raise error
        with patch.object(dashboard_generator, '_render_table', side_effect=Exception("Render error")):
            html = dashboard_generator._render_section(section)

            assert "Error rendering section" in html

    def test_get_css_styles_light_theme(self, dashboard_generator):
        """Test CSS styles generation for light theme."""
        css = dashboard_generator._get_css_styles("light")

        assert "background-color: #f5f5f5" in css
        assert "color: #333" in css

    def test_get_css_styles_dark_theme(self, dashboard_generator):
        """Test CSS styles generation for dark theme."""
        css = dashboard_generator._get_css_styles("dark")

        assert "background-color: #1a1a1a" in css
        assert "color: #e0e0e0" in css

    def test_generate_error_dashboard(self, dashboard_generator):
        """Test error dashboard generation."""
        error_path = dashboard_generator._generate_error_dashboard("Test error message")

        assert error_path is not None
        error_file = Path(error_path)
        assert error_file.exists()

        content = error_file.read_text()
        assert "Test error message" in content
        assert "Dashboard Generation Error" in content

    def test_generate_json_report_success(self, dashboard_generator):
        """Test JSON report generation."""
        dashboard_generator.analytics_engine.get_dashboard_data.return_value = {
            "metrics": {"total_tests": 100},
            "trends": {},
            "period": {"days": 7}
        }

        json_path = dashboard_generator.generate_json_report(7)

        assert json_path is not None
        json_file = Path(json_path)
        assert json_file.exists()
        assert json_file.suffix == '.json'

        # Verify JSON content
        with open(json_file) as f:
            data = json.load(f)
            assert "metrics" in data
            assert data["metrics"]["total_tests"] == 100

    def test_generate_json_report_error(self, dashboard_generator):
        """Test JSON report generation with error."""
        dashboard_generator.analytics_engine.get_dashboard_data.side_effect = Exception("Data error")

        json_path = dashboard_generator.generate_json_report(7)

        # Should return error.json path
        assert "error.json" in json_path

    def test_dashboard_section_creation_error(self, dashboard_generator):
        """Test dashboard section creation with analytics error."""
        dashboard_generator.analytics_engine.calculate_metrics.side_effect = Exception("Metrics error")

        sections = dashboard_generator._generate_dashboard_sections(7, False)

        # Should include error section
        assert len(sections) > 0
        error_sections = [s for s in sections if s.css_class == "error"]
        assert len(error_sections) > 0

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', True)
    def test_charts_sections_creation(self, dashboard_generator):
        """Test charts sections creation with plotting available."""
        with patch.object(dashboard_generator.chart_generator, 'generate_pass_rate_pie_chart') as mock_pie, \
             patch.object(dashboard_generator.chart_generator, 'chart_to_base64') as mock_base64:

            mock_pie.return_value = "/fake/pie.png"
            mock_base64.return_value = "data:image/png;base64,fake"

            sections = dashboard_generator._generate_dashboard_sections(7, True)
            chart_sections = [s for s in sections if s.content_type == "chart"]

            # Should have at least one chart section
            assert len(chart_sections) > 0

    @patch('src.python.workspace_qdrant_mcp.testing.analytics.dashboard.PLOTTING_AVAILABLE', False)
    def test_charts_sections_no_plotting(self, dashboard_generator):
        """Test charts sections creation without plotting available."""
        sections = dashboard_generator._generate_dashboard_sections(7, True)

        # Should still work but without charts
        chart_sections = [s for s in sections if s.content_type == "chart"]
        assert len(chart_sections) == 0 or all("not available" in str(s.content) for s in chart_sections)
