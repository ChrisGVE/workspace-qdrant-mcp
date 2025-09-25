"""
Test Analytics Dashboard

Interactive dashboard generator for test analytics with visualization,
reporting, and real-time monitoring capabilities including edge case
handling for missing data and rendering failures.
"""

import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.parse import quote
import base64
import io

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None
    sns = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from .engine import TestAnalyticsEngine, TestMetrics, TrendAnalysis, QualityReport, MetricType, TrendDirection

logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    chart_type: str  # line, bar, pie, heatmap
    title: str
    width: int = 800
    height: int = 600
    colors: List[str] = None
    show_legend: bool = True
    show_grid: bool = True


@dataclass
class DashboardSection:
    """Represents a dashboard section."""
    title: str
    content_type: str  # chart, table, metric, text
    content: Any
    order: int = 0
    css_class: str = ""


class ChartGenerator:
    """Generates charts for dashboard visualization."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize chart generator.

        Args:
            output_dir: Directory to save chart images
        """
        self.output_dir = output_dir or Path.cwd() / "charts"
        self.output_dir.mkdir(exist_ok=True)

        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available - charts will be disabled")

    def generate_trend_chart(self, trend: TrendAnalysis, config: ChartConfig) -> Optional[str]:
        """
        Generate trend line chart.

        Args:
            trend: Trend analysis data
            config: Chart configuration

        Returns:
            Path to generated chart image or None if failed
        """
        if not PLOTTING_AVAILABLE or not trend.data_points:
            logger.warning("Cannot generate trend chart - plotting unavailable or no data")
            return None

        try:
            plt.figure(figsize=(config.width/100, config.height/100))

            # Extract data
            dates = [point[0] for point in trend.data_points]
            values = [point[1] for point in trend.data_points]

            # Handle edge case: all values are the same
            if len(set(values)) == 1:
                plt.axhline(y=values[0], color='blue', linestyle='-', linewidth=2)
            else:
                plt.plot(dates, values, marker='o', linewidth=2, markersize=6)

            # Add trend line if available
            if trend.regression_slope is not None and len(values) > 1:
                x_numeric = list(range(len(dates)))
                trend_line = [trend.regression_slope * x + values[0] for x in x_numeric]
                plt.plot(dates, trend_line, '--', alpha=0.7, color='red', label='Trend')

            # Mark anomalies
            for anomaly_date in trend.anomalies:
                if anomaly_date in dates:
                    idx = dates.index(anomaly_date)
                    plt.scatter(anomaly_date, values[idx], color='red', s=100, zorder=5, label='Anomaly')

            plt.title(config.title, fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel(trend.metric_type.value.replace('_', ' ').title(), fontsize=12)

            if config.show_grid:
                plt.grid(True, alpha=0.3)

            if config.show_legend and (trend.regression_slope is not None or trend.anomalies):
                plt.legend()

            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save chart
            filename = f"trend_{trend.metric_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = self.output_dir / filename
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Failed to generate trend chart: {e}")
            plt.close()
            return None

    def generate_metrics_bar_chart(self, metrics: Dict[str, float], config: ChartConfig) -> Optional[str]:
        """
        Generate bar chart for metrics.

        Args:
            metrics: Dictionary of metric names and values
            config: Chart configuration

        Returns:
            Path to generated chart image or None if failed
        """
        if not PLOTTING_AVAILABLE or not metrics:
            return None

        try:
            plt.figure(figsize=(config.width/100, config.height/100))

            names = list(metrics.keys())
            values = list(metrics.values())

            # Handle edge case: empty or invalid values
            if not values or all(v is None for v in values):
                logger.warning("No valid metric values for bar chart")
                return None

            # Replace None values with 0
            values = [v if v is not None else 0 for v in values]

            bars = plt.bar(names, values, color=config.colors or 'skyblue')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

            plt.title(config.title, fontsize=14, fontweight='bold')
            plt.ylabel('Value', fontsize=12)

            if config.show_grid:
                plt.grid(True, axis='y', alpha=0.3)

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save chart
            filename = f"metrics_bar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = self.output_dir / filename
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Failed to generate metrics bar chart: {e}")
            plt.close()
            return None

    def generate_pass_rate_pie_chart(self, metrics: TestMetrics, config: ChartConfig) -> Optional[str]:
        """Generate pie chart for test results distribution."""
        if not PLOTTING_AVAILABLE:
            return None

        try:
            plt.figure(figsize=(config.width/100, config.height/100))

            # Prepare data with edge case handling
            labels = []
            sizes = []
            colors = ['green', 'red', 'yellow', 'orange']

            if metrics.passed > 0:
                labels.append(f'Passed ({metrics.passed})')
                sizes.append(metrics.passed)
            if metrics.failed > 0:
                labels.append(f'Failed ({metrics.failed})')
                sizes.append(metrics.failed)
            if metrics.skipped > 0:
                labels.append(f'Skipped ({metrics.skipped})')
                sizes.append(metrics.skipped)
            if metrics.errors > 0:
                labels.append(f'Errors ({metrics.errors})')
                sizes.append(metrics.errors)

            # Handle edge case: no test results
            if not sizes:
                logger.warning("No test results for pie chart")
                return None

            plt.pie(sizes, labels=labels, colors=colors[:len(sizes)], autopct='%1.1f%%',
                   startangle=90, textprops={'fontsize': 10})

            plt.title(config.title, fontsize=14, fontweight='bold')
            plt.axis('equal')

            # Save chart
            filename = f"pass_rate_pie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            chart_path = self.output_dir / filename
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()

            return str(chart_path)

        except Exception as e:
            logger.error(f"Failed to generate pass rate pie chart: {e}")
            plt.close()
            return None

    def chart_to_base64(self, chart_path: str) -> Optional[str]:
        """Convert chart image to base64 for inline HTML."""
        try:
            if not Path(chart_path).exists():
                return None

            with open(chart_path, 'rb') as f:
                img_data = f.read()

            b64_data = base64.b64encode(img_data).decode()
            return f"data:image/png;base64,{b64_data}"

        except Exception as e:
            logger.error(f"Failed to convert chart to base64: {e}")
            return None


class DashboardGenerator:
    """
    Generates interactive dashboards for test analytics.

    Creates HTML dashboards with charts, tables, and real-time monitoring
    capabilities with comprehensive error handling for missing data.
    """

    def __init__(self, analytics_engine: TestAnalyticsEngine, output_dir: Optional[Path] = None):
        """
        Initialize dashboard generator.

        Args:
            analytics_engine: Analytics engine instance
            output_dir: Directory for output files
        """
        self.analytics_engine = analytics_engine
        self.output_dir = output_dir or Path.cwd() / "dashboards"
        self.output_dir.mkdir(exist_ok=True)
        self.chart_generator = ChartGenerator(self.output_dir / "charts")

    def generate_dashboard(self,
                          title: str = "Test Analytics Dashboard",
                          period_days: int = 7,
                          include_charts: bool = True,
                          theme: str = "light") -> str:
        """
        Generate complete dashboard.

        Args:
            title: Dashboard title
            period_days: Number of days to analyze
            include_charts: Whether to include chart visualizations
            theme: Dashboard theme (light/dark)

        Returns:
            Path to generated HTML dashboard file
        """
        try:
            logger.info(f"Generating dashboard for {period_days} days")

            # Generate dashboard sections
            sections = self._generate_dashboard_sections(period_days, include_charts)

            # Generate HTML
            html_content = self._generate_html(title, sections, theme)

            # Save dashboard
            dashboard_filename = f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            dashboard_path = self.output_dir / dashboard_filename

            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            logger.info(f"Dashboard generated: {dashboard_path}")
            return str(dashboard_path)

        except Exception as e:
            logger.error(f"Failed to generate dashboard: {e}")
            # Generate error dashboard
            return self._generate_error_dashboard(str(e))

    def _generate_dashboard_sections(self, period_days: int, include_charts: bool) -> List[DashboardSection]:
        """Generate all dashboard sections."""
        sections = []

        try:
            # Get current metrics
            end_time = datetime.now()
            start_time = end_time - timedelta(days=period_days)
            current_metrics = self.analytics_engine.calculate_metrics(start_time, end_time)

            # Overview section
            sections.append(self._create_overview_section(current_metrics))

            # Metrics cards section
            sections.append(self._create_metrics_cards_section(current_metrics))

            # Charts section
            if include_charts and PLOTTING_AVAILABLE:
                sections.extend(self._create_charts_sections(current_metrics, period_days))

            # Quality report section
            quality_report = self.analytics_engine.generate_quality_report(period_days)
            sections.append(self._create_quality_section(quality_report))

            # Trends section
            sections.append(self._create_trends_section(period_days))

            # Recommendations section
            sections.append(self._create_recommendations_section(quality_report))

        except Exception as e:
            logger.error(f"Error generating dashboard sections: {e}")
            sections.append(DashboardSection(
                title="Error",
                content_type="text",
                content=f"Failed to generate dashboard sections: {e}",
                css_class="error"
            ))

        return sections

    def _create_overview_section(self, metrics: TestMetrics) -> DashboardSection:
        """Create overview section."""
        overview_data = {
            "period": f"{metrics.period_start.strftime('%Y-%m-%d')} to {metrics.period_end.strftime('%Y-%m-%d')}",
            "total_tests": metrics.total_tests,
            "pass_rate": f"{metrics.pass_rate:.1f}%",
            "avg_execution_time": f"{metrics.avg_execution_time:.2f}s",
            "coverage": f"{metrics.coverage_percentage:.1f}%"
        }

        return DashboardSection(
            title="Test Suite Overview",
            content_type="table",
            content=overview_data,
            order=1
        )

    def _create_metrics_cards_section(self, metrics: TestMetrics) -> DashboardSection:
        """Create metrics cards section."""
        cards = [
            {"title": "Total Tests", "value": metrics.total_tests, "color": "blue"},
            {"title": "Pass Rate", "value": f"{metrics.pass_rate:.1f}%", "color": "green" if metrics.pass_rate >= 90 else "yellow" if metrics.pass_rate >= 70 else "red"},
            {"title": "Failed Tests", "value": metrics.failed, "color": "red" if metrics.failed > 0 else "green"},
            {"title": "Average Time", "value": f"{metrics.avg_execution_time:.2f}s", "color": "green" if metrics.avg_execution_time < 10 else "yellow" if metrics.avg_execution_time < 30 else "red"},
            {"title": "Coverage", "value": f"{metrics.coverage_percentage:.1f}%", "color": "green" if metrics.coverage_percentage >= 80 else "yellow" if metrics.coverage_percentage >= 60 else "red"},
            {"title": "Flakiness", "value": f"{metrics.flakiness_score:.1f}%", "color": "green" if metrics.flakiness_score < 5 else "yellow" if metrics.flakiness_score < 15 else "red"}
        ]

        return DashboardSection(
            title="Key Metrics",
            content_type="cards",
            content=cards,
            order=2
        )

    def _create_charts_sections(self, metrics: TestMetrics, period_days: int) -> List[DashboardSection]:
        """Create chart sections."""
        sections = []

        try:
            # Pass rate pie chart
            pie_config = ChartConfig(
                chart_type="pie",
                title="Test Results Distribution",
                width=600,
                height=400
            )
            pie_chart_path = self.chart_generator.generate_pass_rate_pie_chart(metrics, pie_config)
            if pie_chart_path:
                pie_chart_b64 = self.chart_generator.chart_to_base64(pie_chart_path)
                sections.append(DashboardSection(
                    title="Test Results Distribution",
                    content_type="chart",
                    content={"image_data": pie_chart_b64, "alt": "Test results pie chart"},
                    order=3
                ))

            # Trend charts
            for metric_type in [MetricType.PASS_RATE, MetricType.EXECUTION_TIME]:
                trend = self.analytics_engine.analyze_trends(metric_type, days_back=period_days)
                if trend and trend.data_points:
                    trend_config = ChartConfig(
                        chart_type="line",
                        title=f"{metric_type.value.replace('_', ' ').title()} Trend",
                        width=800,
                        height=400
                    )
                    chart_path = self.chart_generator.generate_trend_chart(trend, trend_config)
                    if chart_path:
                        chart_b64 = self.chart_generator.chart_to_base64(chart_path)
                        sections.append(DashboardSection(
                            title=trend_config.title,
                            content_type="chart",
                            content={"image_data": chart_b64, "alt": f"{metric_type.value} trend chart"},
                            order=4
                        ))

        except Exception as e:
            logger.error(f"Failed to create chart sections: {e}")
            sections.append(DashboardSection(
                title="Charts Error",
                content_type="text",
                content=f"Failed to generate charts: {e}",
                css_class="error"
            ))

        return sections

    def _create_quality_section(self, quality_report: QualityReport) -> DashboardSection:
        """Create quality report section."""
        quality_data = {
            "overall_score": f"{quality_report.overall_score:.1f}/100",
            "critical_issues": len(quality_report.critical_issues),
            "warnings": len(quality_report.warnings),
            "recommendations": len(quality_report.recommendations)
        }

        return DashboardSection(
            title="Quality Assessment",
            content_type="quality_report",
            content=quality_data,
            order=5
        )

    def _create_trends_section(self, period_days: int) -> DashboardSection:
        """Create trends analysis section."""
        trends_data = {}

        for metric_type in [MetricType.PASS_RATE, MetricType.EXECUTION_TIME, MetricType.COVERAGE]:
            trend = self.analytics_engine.analyze_trends(metric_type, days_back=period_days)
            if trend:
                trends_data[metric_type.value] = {
                    "direction": trend.direction.value,
                    "change": f"{trend.change_percentage:+.1f}%",
                    "confidence": f"{trend.confidence:.2f}" if trend.confidence else "N/A"
                }

        return DashboardSection(
            title="Trend Analysis",
            content_type="trends",
            content=trends_data,
            order=6
        )

    def _create_recommendations_section(self, quality_report: QualityReport) -> DashboardSection:
        """Create recommendations section."""
        return DashboardSection(
            title="Recommendations",
            content_type="recommendations",
            content={
                "recommendations": quality_report.recommendations,
                "warnings": quality_report.warnings,
                "critical_issues": quality_report.critical_issues
            },
            order=7
        )

    def _generate_html(self, title: str, sections: List[DashboardSection], theme: str) -> str:
        """Generate complete HTML dashboard."""
        html_parts = []

        # HTML header
        html_parts.append(f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {self._get_css_styles(theme)}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        <main>''')

        # Generate sections
        sorted_sections = sorted(sections, key=lambda x: x.order)
        for section in sorted_sections:
            html_parts.append(self._render_section(section))

        # HTML footer
        html_parts.append('''
        </main>
        <footer>
            <p>Generated by Test Analytics Dashboard</p>
        </footer>
    </div>
    <script>
        // Auto-refresh every 5 minutes
        setTimeout(function() {
            location.reload();
        }, 300000);
    </script>
</body>
</html>''')

        return '\n'.join(html_parts)

    def _render_section(self, section: DashboardSection) -> str:
        """Render individual dashboard section."""
        try:
            section_html = f'<section class="dashboard-section {section.css_class}">'
            section_html += f'<h2>{section.title}</h2>'

            if section.content_type == "table":
                section_html += self._render_table(section.content)
            elif section.content_type == "cards":
                section_html += self._render_cards(section.content)
            elif section.content_type == "chart":
                section_html += self._render_chart(section.content)
            elif section.content_type == "quality_report":
                section_html += self._render_quality_report(section.content)
            elif section.content_type == "trends":
                section_html += self._render_trends(section.content)
            elif section.content_type == "recommendations":
                section_html += self._render_recommendations(section.content)
            elif section.content_type == "text":
                section_html += f'<p>{section.content}</p>'

            section_html += '</section>'
            return section_html

        except Exception as e:
            logger.error(f"Failed to render section {section.title}: {e}")
            return f'<section class="dashboard-section error"><h2>{section.title}</h2><p>Error rendering section: {e}</p></section>'

    def _render_table(self, data: Dict[str, Any]) -> str:
        """Render data table."""
        html = '<table class="data-table">'
        for key, value in data.items():
            html += f'<tr><td>{key.replace("_", " ").title()}</td><td>{value}</td></tr>'
        html += '</table>'
        return html

    def _render_cards(self, cards: List[Dict[str, Any]]) -> str:
        """Render metric cards."""
        html = '<div class="cards-container">'
        for card in cards:
            html += f'''
                <div class="metric-card {card.get('color', 'blue')}">
                    <h3>{card['title']}</h3>
                    <div class="metric-value">{card['value']}</div>
                </div>
            '''
        html += '</div>'
        return html

    def _render_chart(self, chart_data: Dict[str, Any]) -> str:
        """Render chart."""
        if chart_data.get('image_data'):
            return f'<img src="{chart_data["image_data"]}" alt="{chart_data.get("alt", "Chart")}" class="chart-image">'
        return '<p class="no-data">Chart not available</p>'

    def _render_quality_report(self, data: Dict[str, Any]) -> str:
        """Render quality report."""
        html = f'''
            <div class="quality-summary">
                <div class="score-circle">
                    <span class="score">{data['overall_score']}</span>
                </div>
                <div class="quality-details">
                    <p><strong>Critical Issues:</strong> {data['critical_issues']}</p>
                    <p><strong>Warnings:</strong> {data['warnings']}</p>
                    <p><strong>Recommendations:</strong> {data['recommendations']}</p>
                </div>
            </div>
        '''
        return html

    def _render_trends(self, trends: Dict[str, Dict[str, Any]]) -> str:
        """Render trends data."""
        if not trends:
            return '<p class="no-data">No trend data available</p>'

        html = '<div class="trends-grid">'
        for metric, trend_info in trends.items():
            direction_class = trend_info['direction']
            html += f'''
                <div class="trend-item {direction_class}">
                    <h4>{metric.replace('_', ' ').title()}</h4>
                    <p class="trend-direction">{trend_info['direction'].title()}</p>
                    <p class="trend-change">{trend_info['change']}</p>
                    <p class="trend-confidence">Confidence: {trend_info['confidence']}</p>
                </div>
            '''
        html += '</div>'
        return html

    def _render_recommendations(self, data: Dict[str, List[str]]) -> str:
        """Render recommendations."""
        html = ''

        if data['critical_issues']:
            html += '<div class="critical-issues"><h4>Critical Issues</h4><ul>'
            for issue in data['critical_issues']:
                html += f'<li>{issue}</li>'
            html += '</ul></div>'

        if data['warnings']:
            html += '<div class="warnings"><h4>Warnings</h4><ul>'
            for warning in data['warnings']:
                html += f'<li>{warning}</li>'
            html += '</ul></div>'

        if data['recommendations']:
            html += '<div class="recommendations"><h4>Recommendations</h4><ul>'
            for rec in data['recommendations']:
                html += f'<li>{rec}</li>'
            html += '</ul></div>'

        return html or '<p class="no-data">No recommendations available</p>'

    def _get_css_styles(self, theme: str) -> str:
        """Get CSS styles for dashboard."""
        base_styles = '''
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: 'Arial', sans-serif; line-height: 1.6; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            header { text-align: center; margin-bottom: 30px; }
            h1 { font-size: 2.5em; margin-bottom: 10px; }
            .subtitle { color: #666; font-size: 1.1em; }
            .dashboard-section { margin-bottom: 30px; padding: 20px; border-radius: 8px; }
            .dashboard-section h2 { margin-bottom: 15px; font-size: 1.8em; }
            .data-table { width: 100%; border-collapse: collapse; }
            .data-table td { padding: 10px; border-bottom: 1px solid #eee; }
            .cards-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }
            .metric-card { padding: 20px; border-radius: 8px; text-align: center; }
            .metric-card h3 { margin-bottom: 10px; }
            .metric-value { font-size: 2em; font-weight: bold; }
            .chart-image { max-width: 100%; height: auto; }
            .quality-summary { display: flex; align-items: center; gap: 20px; }
            .score-circle { width: 100px; height: 100px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 1.5em; font-weight: bold; }
            .trends-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
            .trend-item { padding: 15px; border-radius: 8px; text-align: center; }
            .trend-direction { font-size: 1.2em; font-weight: bold; margin: 10px 0; }
            .trend-change { font-size: 1.1em; }
            .no-data { color: #999; font-style: italic; }
            .error { background-color: #ffe6e6; border: 1px solid #ff9999; }
            footer { text-align: center; margin-top: 40px; color: #666; }
        '''

        if theme == "light":
            theme_styles = '''
                body { background-color: #f5f5f5; color: #333; }
                .dashboard-section { background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-card.green { background-color: #d4edda; border: 1px solid #c3e6cb; }
                .metric-card.yellow { background-color: #fff3cd; border: 1px solid #ffeaa7; }
                .metric-card.red { background-color: #f8d7da; border: 1px solid #f5c6cb; }
                .metric-card.blue { background-color: #d1ecf1; border: 1px solid #bee5eb; }
                .score-circle { background-color: #28a745; color: white; }
                .trend-item { background-color: #f8f9fa; }
                .trend-item.improving { border-left: 4px solid #28a745; }
                .trend-item.declining { border-left: 4px solid #dc3545; }
                .trend-item.stable { border-left: 4px solid #17a2b8; }
                .trend-item.volatile { border-left: 4px solid #ffc107; }
            '''
        else:  # dark theme
            theme_styles = '''
                body { background-color: #1a1a1a; color: #e0e0e0; }
                .dashboard-section { background-color: #2d2d2d; }
                .metric-card.green { background-color: #1e4d2b; border: 1px solid #28a745; }
                .metric-card.yellow { background-color: #4d4419; border: 1px solid #ffc107; }
                .metric-card.red { background-color: #4d1e1f; border: 1px solid #dc3545; }
                .metric-card.blue { background-color: #1e3a4d; border: 1px solid #17a2b8; }
                .score-circle { background-color: #28a745; color: white; }
                .trend-item { background-color: #3a3a3a; }
            '''

        return base_styles + theme_styles

    def _generate_error_dashboard(self, error_message: str) -> str:
        """Generate minimal error dashboard."""
        error_html = f'''<!DOCTYPE html>
<html><head><title>Dashboard Error</title></head>
<body>
<h1>Dashboard Generation Error</h1>
<p>Failed to generate dashboard: {error_message}</p>
<p>Please check the logs for more details.</p>
</body></html>'''

        error_path = self.output_dir / "error_dashboard.html"
        with open(error_path, 'w') as f:
            f.write(error_html)

        return str(error_path)

    def generate_json_report(self, period_days: int = 7) -> str:
        """Generate JSON report for API consumption."""
        try:
            dashboard_data = self.analytics_engine.get_dashboard_data(period_days)

            json_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            json_path = self.output_dir / json_filename

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard_data, f, indent=2, ensure_ascii=False)

            return str(json_path)

        except Exception as e:
            logger.error(f"Failed to generate JSON report: {e}")
            return str(self.output_dir / "error.json")