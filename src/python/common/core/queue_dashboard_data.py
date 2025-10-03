"""
Queue Dashboard Data Preparation

Provides comprehensive data preparation for queue monitoring dashboards, including
widget data formatting, layout configuration, and real-time metrics aggregation.
Integrates with all monitoring modules to provide a unified dashboard experience.

Features:
    - Multiple widget types (gauges, charts, tables, status cards, heatmaps)
    - Flexible layout system with grid-based positioning
    - Real-time data updates with caching
    - Data formatting for timestamps, numbers, colors
    - Sparkline data for trend indicators
    - Chart series preparation for time-series visualization
    - JSON export for dashboard configuration
    - Preset layouts (overview, performance, errors)

Widget Types:
    - STATUS_CARD: Summary metrics with trend indicators
    - LINE_CHART: Time-series visualizations
    - BAR_CHART: Categorical comparisons
    - TABLE: Detailed data listings
    - GAUGE: Progress/capacity indicators
    - HEATMAP: Activity patterns by time

Example:
    ```python
    from workspace_qdrant_mcp.core.queue_dashboard_data import (
        QueueDashboardDataProvider
    )

    # Initialize provider
    provider = QueueDashboardDataProvider()
    await provider.initialize()

    # Get complete dashboard layout
    layout = await provider.get_dashboard_layout()
    for widget in layout.widgets:
        print(f"Widget: {widget.title} ({widget.widget_type})")

    # Get specific widget data
    widget = await provider.get_widget_data("queue_size_gauge")
    print(f"Current queue size: {widget.data['value']}")

    # Get overview widgets
    overview = await provider.get_overview_widgets()
    for widget in overview:
        print(f"{widget.title}: {widget.data}")

    # Get real-time metrics for streaming
    metrics = await provider.get_realtime_metrics()
    print(f"Real-time metrics: {metrics}")

    # Export dashboard configuration
    json_config = await provider.export_dashboard_config()
    ```
"""

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .queue_statistics import QueueStatisticsCollector
from .queue_performance_metrics import QueuePerformanceCollector
from .queue_health import QueueHealthCalculator, HealthStatus
from .queue_trend_analysis import HistoricalTrendAnalyzer
from .queue_bottleneck_detector import BottleneckDetector
from .queue_backpressure import BackpressureDetector
from .error_message_manager import ErrorMessageManager


class WidgetType(str, Enum):
    """Dashboard widget type classifications."""

    GAUGE = "gauge"
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    TABLE = "table"
    STATUS_CARD = "status_card"
    HEATMAP = "heatmap"


@dataclass
class DashboardWidget:
    """
    Dashboard widget configuration and data.

    Attributes:
        widget_id: Unique identifier for the widget
        widget_type: Type of widget (gauge, chart, table, etc.)
        title: Display title for the widget
        data: Widget-specific data payload
        config: Widget-specific configuration options
        position: Grid position (x, y, width, height)
        refresh_interval: Seconds between data refreshes
    """

    widget_id: str
    widget_type: WidgetType
    title: str
    data: Dict[str, Any]
    config: Dict[str, Any] = field(default_factory=dict)
    position: Optional[Tuple[int, int, int, int]] = None  # (x, y, width, height)
    refresh_interval: int = 30  # seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "widget_id": self.widget_id,
            "widget_type": self.widget_type.value,
            "title": self.title,
            "data": self.data,
            "config": self.config,
            "position": {
                "x": self.position[0],
                "y": self.position[1],
                "width": self.position[2],
                "height": self.position[3]
            } if self.position else None,
            "refresh_interval": self.refresh_interval
        }


@dataclass
class DashboardLayout:
    """
    Complete dashboard layout configuration.

    Attributes:
        widgets: List of dashboard widgets
        refresh_interval: Global refresh interval in seconds
        layout_config: Grid configuration and responsive breakpoints
        preset_name: Name of layout preset (overview, performance, errors)
    """

    widgets: List[DashboardWidget]
    refresh_interval: int = 30  # seconds
    layout_config: Dict[str, Any] = field(default_factory=dict)
    preset_name: str = "overview"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "widgets": [w.to_dict() for w in self.widgets],
            "refresh_interval": self.refresh_interval,
            "layout_config": self.layout_config,
            "preset_name": self.preset_name
        }


class QueueDashboardDataProvider:
    """
    Dashboard data provider for queue monitoring.

    Aggregates data from all monitoring modules and prepares it for
    dashboard visualization with proper formatting, caching, and layout.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        cache_ttl_seconds: int = 30,
        enable_trend_analysis: bool = True,
        enable_bottleneck_detection: bool = True
    ):
        """
        Initialize dashboard data provider.

        Args:
            db_path: Optional custom database path
            cache_ttl_seconds: Cache time-to-live in seconds
            enable_trend_analysis: Whether to enable trend analysis
            enable_bottleneck_detection: Whether to enable bottleneck detection
        """
        self.db_path = db_path
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_trend_analysis = enable_trend_analysis
        self.enable_bottleneck_detection = enable_bottleneck_detection

        # Monitoring components
        self.stats_collector: Optional[QueueStatisticsCollector] = None
        self.performance_collector: Optional[QueuePerformanceCollector] = None
        self.health_calculator: Optional[QueueHealthCalculator] = None
        self.trend_analyzer: Optional[HistoricalTrendAnalyzer] = None
        self.bottleneck_detector: Optional[BottleneckDetector] = None
        self.backpressure_detector: Optional[BackpressureDetector] = None
        self.error_manager: Optional[ErrorMessageManager] = None

        # Widget data cache
        self._widget_cache: Dict[str, Tuple[datetime, DashboardWidget]] = {}
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the dashboard data provider."""
        if self._initialized:
            return

        # Initialize statistics collector
        self.stats_collector = QueueStatisticsCollector(db_path=self.db_path)
        await self.stats_collector.initialize()

        # Initialize performance collector
        self.performance_collector = QueuePerformanceCollector(
            db_path=self.db_path,
            enable_resource_tracking=True
        )
        await self.performance_collector.initialize()

        # Initialize backpressure detector
        self.backpressure_detector = BackpressureDetector(
            stats_collector=self.stats_collector
        )
        await self.backpressure_detector.initialize()

        # Initialize health calculator
        self.health_calculator = QueueHealthCalculator(
            stats_collector=self.stats_collector,
            backpressure_detector=self.backpressure_detector,
            performance_collector=self.performance_collector
        )
        await self.health_calculator.initialize()

        # Initialize trend analyzer if enabled
        if self.enable_trend_analysis:
            self.trend_analyzer = HistoricalTrendAnalyzer(db_path=self.db_path)
            await self.trend_analyzer.initialize()

        # Initialize bottleneck detector if enabled
        if self.enable_bottleneck_detection:
            self.bottleneck_detector = BottleneckDetector(db_path=self.db_path)
            await self.bottleneck_detector.initialize()

        # Initialize error statistics collector
        self.error_manager = ErrorMessageManager(db_path=self.db_path)
        await self.error_manager.initialize()

        self._initialized = True
        logger.info("Queue dashboard data provider initialized")

    async def close(self):
        """Close the dashboard data provider."""
        if not self._initialized:
            return

        if self.stats_collector:
            await self.stats_collector.close()

        if self.performance_collector:
            await self.performance_collector.close()

        if self.health_calculator:
            await self.health_calculator.close()

        if self.backpressure_detector:
            await self.backpressure_detector.close()

        if self.trend_analyzer:
            await self.trend_analyzer.close()

        if self.bottleneck_detector:
            await self.bottleneck_detector.close()

        if self.error_stats_collector:
            await self.error_manager.close()

        self._initialized = False
        logger.info("Queue dashboard data provider closed")

    async def get_dashboard_layout(
        self,
        preset: str = "overview"
    ) -> DashboardLayout:
        """
        Get complete dashboard layout with all widgets.

        Args:
            preset: Layout preset name (overview, performance, errors)

        Returns:
            DashboardLayout with configured widgets

        Raises:
            ValueError: If preset is not recognized
        """
        if preset == "overview":
            return await self._get_overview_layout()
        elif preset == "performance":
            return await self._get_performance_layout()
        elif preset == "errors":
            return await self._get_errors_layout()
        else:
            raise ValueError(f"Unknown dashboard preset: {preset}")

    async def _get_overview_layout(self) -> DashboardLayout:
        """Get overview dashboard layout."""
        widgets = []

        # Row 1: Status cards (4 columns)
        overview_widgets = await self.get_overview_widgets()
        for i, widget in enumerate(overview_widgets[:4]):
            widget.position = (i * 3, 0, 3, 2)
            widgets.append(widget)

        # Row 2: Health gauge + Charts
        health_widget = await self.get_health_status_widget()
        health_widget.position = (0, 2, 4, 4)
        widgets.append(health_widget)

        # Queue size chart
        queue_chart = await self._get_queue_size_chart()
        queue_chart.position = (4, 2, 8, 4)
        widgets.append(queue_chart)

        # Row 3: Performance charts
        perf_charts = await self.get_performance_charts()
        for i, widget in enumerate(perf_charts[:2]):
            widget.position = (i * 6, 6, 6, 4)
            widgets.append(widget)

        return DashboardLayout(
            widgets=widgets,
            refresh_interval=30,
            layout_config={
                "columns": 12,
                "row_height": 60,
                "breakpoints": {"lg": 1200, "md": 996, "sm": 768, "xs": 480}
            },
            preset_name="overview"
        )

    async def _get_performance_layout(self) -> DashboardLayout:
        """Get performance dashboard layout."""
        widgets = []

        # Performance charts
        perf_charts = await self.get_performance_charts()
        positions = [
            (0, 0, 6, 4),  # Processing rate
            (6, 0, 6, 4),  # Latency
            (0, 4, 6, 4),  # Throughput
            (6, 4, 6, 4),  # Resource usage
        ]

        for widget, position in zip(perf_charts, positions):
            widget.position = position
            widgets.append(widget)

        # Bottleneck table
        if self.enable_bottleneck_detection:
            bottleneck_table = await self._get_bottleneck_table()
            bottleneck_table.position = (0, 8, 12, 4)
            widgets.append(bottleneck_table)

        return DashboardLayout(
            widgets=widgets,
            refresh_interval=30,
            layout_config={
                "columns": 12,
                "row_height": 60,
                "breakpoints": {"lg": 1200, "md": 996, "sm": 768, "xs": 480}
            },
            preset_name="performance"
        )

    async def _get_errors_layout(self) -> DashboardLayout:
        """Get errors dashboard layout."""
        widgets = []

        # Error rate card
        error_card = await self._get_error_rate_card()
        error_card.position = (0, 0, 3, 2)
        widgets.append(error_card)

        # Error chart
        error_chart = await self._get_error_rate_chart()
        error_chart.position = (3, 0, 9, 4)
        widgets.append(error_chart)

        # Recent errors table
        errors_table = await self._get_recent_errors_table()
        errors_table.position = (0, 4, 12, 6)
        widgets.append(errors_table)

        return DashboardLayout(
            widgets=widgets,
            refresh_interval=30,
            layout_config={
                "columns": 12,
                "row_height": 60,
                "breakpoints": {"lg": 1200, "md": 996, "sm": 768, "xs": 480}
            },
            preset_name="errors"
        )

    async def get_widget_data(
        self,
        widget_id: str,
        use_cache: bool = True
    ) -> DashboardWidget:
        """
        Get specific widget data by ID.

        Args:
            widget_id: Unique widget identifier
            use_cache: Whether to use cached data if available

        Returns:
            DashboardWidget with current data

        Raises:
            ValueError: If widget_id is not recognized
        """
        # Check cache first
        if use_cache:
            async with self._lock:
                if widget_id in self._widget_cache:
                    timestamp, widget = self._widget_cache[widget_id]
                    age = (datetime.now(timezone.utc) - timestamp).total_seconds()
                    if age < self.cache_ttl_seconds:
                        logger.debug(f"Using cached data for widget '{widget_id}' (age: {age:.1f}s)")
                        return widget

        # Generate widget data
        widget = await self._generate_widget_data(widget_id)

        # Update cache
        async with self._lock:
            self._widget_cache[widget_id] = (datetime.now(timezone.utc), widget)

        return widget

    async def _generate_widget_data(self, widget_id: str) -> DashboardWidget:
        """Generate widget data by ID."""
        # Map widget IDs to generation methods
        generators = {
            "queue_size_card": self._get_queue_size_card,
            "processing_rate_card": self._get_processing_rate_card,
            "error_rate_card": self._get_error_rate_card,
            "success_rate_card": self._get_success_rate_gauge,
            "health_status": self.get_health_status_widget,
            "queue_size_chart": self._get_queue_size_chart,
            "processing_rate_chart": self._get_processing_rate_chart,
            "error_rate_chart": self._get_error_rate_chart,
            "latency_chart": self._get_latency_chart,
            "resource_usage_chart": self._get_resource_usage_chart,
            "slow_collections_bar": self._get_slow_collections_bar,
            "error_categories_bar": self._get_error_categories_bar,
            "recent_errors_table": self._get_recent_errors_table,
            "slow_operations_table": self._get_slow_operations_table,
            "bottleneck_table": self._get_bottleneck_table,
            "processing_heatmap": self._get_processing_heatmap,
            "error_heatmap": self._get_error_heatmap,
        }

        if widget_id not in generators:
            raise ValueError(f"Unknown widget_id: {widget_id}")

        return await generators[widget_id]()

    async def get_overview_widgets(self) -> List[DashboardWidget]:
        """
        Get overview status card widgets.

        Returns:
            List of STATUS_CARD widgets for dashboard overview
        """
        return [
            await self._get_queue_size_card(),
            await self._get_processing_rate_card(),
            await self._get_error_rate_card(),
            await self._get_success_rate_gauge()
        ]

    async def get_performance_charts(self) -> List[DashboardWidget]:
        """
        Get performance chart widgets.

        Returns:
            List of LINE_CHART widgets for performance metrics
        """
        return [
            await self._get_processing_rate_chart(),
            await self._get_latency_chart(),
            await self._get_queue_size_chart(),
            await self._get_resource_usage_chart()
        ]

    async def get_health_status_widget(self) -> DashboardWidget:
        """
        Get health status gauge widget.

        Returns:
            GAUGE widget with health score
        """
        health_status = await self.health_calculator.calculate_health()

        # Map health status to color
        color_map = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.DEGRADED: "yellow",
            HealthStatus.UNHEALTHY: "orange",
            HealthStatus.CRITICAL: "red"
        }

        return DashboardWidget(
            widget_id="health_status",
            widget_type=WidgetType.GAUGE,
            title="Queue Health",
            data={
                "value": health_status.score,
                "min": 0,
                "max": 100,
                "threshold_green": 80,
                "threshold_yellow": 60,
                "threshold_orange": 40,
                "status": health_status.overall_status.value,
                "color": color_map[health_status.overall_status],
                "recommendations": health_status.recommendations,
                "timestamp": self._format_timestamp(health_status.timestamp)
            },
            config={
                "show_percentage": True,
                "show_status_text": True,
                "animate": True
            }
        )

    async def get_realtime_metrics(self) -> Dict[str, Any]:
        """
        Get real-time metrics for streaming updates.

        Returns:
            Dictionary with current metric values for real-time display
        """
        stats = await self.stats_collector.get_current_statistics()
        health_status = await self.health_calculator.calculate_health()

        return {
            "timestamp": self._format_timestamp(datetime.now(timezone.utc)),
            "queue_size": stats.queue_size,
            "processing_rate": self._format_number(stats.processing_rate, 2),
            "error_rate": self._format_number(stats.failure_rate, 2),
            "success_rate": self._format_number(stats.success_rate, 2),
            "health_score": self._format_number(health_status.score, 1),
            "health_status": health_status.overall_status.value
        }

    async def export_dashboard_config(
        self,
        format: str = 'json',
        preset: str = "overview"
    ) -> str:
        """
        Export dashboard configuration.

        Args:
            format: Export format ('json' only for now)
            preset: Dashboard preset to export

        Returns:
            JSON string with dashboard configuration

        Raises:
            ValueError: If format is not supported
        """
        if format != 'json':
            raise ValueError(f"Unsupported format: {format}. Use 'json'")

        layout = await self.get_dashboard_layout(preset=preset)

        export_data = {
            "dashboard_version": "1.0",
            "export_timestamp": self._format_timestamp(datetime.now(timezone.utc)),
            "preset": preset,
            "layout": layout.to_dict()
        }

        return json.dumps(export_data, indent=2)

    # =========================================================================
    # STATUS_CARD Widget Generators
    # =========================================================================

    async def _get_queue_size_card(self) -> DashboardWidget:
        """Generate queue size status card."""
        stats = await self.stats_collector.get_current_statistics()

        # Get sparkline data (last 20 points)
        sparkline = await self._get_sparkline_data("queue_size", 20)

        # Calculate trend
        trend = "stable"
        if len(sparkline) >= 2:
            if sparkline[-1] > sparkline[0] * 1.1:
                trend = "up"
            elif sparkline[-1] < sparkline[0] * 0.9:
                trend = "down"

        return DashboardWidget(
            widget_id="queue_size_card",
            widget_type=WidgetType.STATUS_CARD,
            title="Queue Size",
            data={
                "value": stats.queue_size,
                "formatted_value": self._format_number(stats.queue_size, 0),
                "trend": trend,
                "sparkline": sparkline,
                "timestamp": self._format_timestamp(stats.timestamp),
                "color": self._get_queue_size_color(stats.queue_size)
            },
            config={
                "show_sparkline": True,
                "show_trend_icon": True
            }
        )

    async def _get_processing_rate_card(self) -> DashboardWidget:
        """Generate processing rate status card."""
        stats = await self.stats_collector.get_current_statistics()
        sparkline = await self._get_sparkline_data("processing_rate", 20)

        return DashboardWidget(
            widget_id="processing_rate_card",
            widget_type=WidgetType.STATUS_CARD,
            title="Processing Rate",
            data={
                "value": stats.processing_rate,
                "formatted_value": f"{self._format_number(stats.processing_rate, 1)} items/min",
                "sparkline": sparkline,
                "timestamp": self._format_timestamp(stats.timestamp),
                "color": "blue"
            },
            config={
                "show_sparkline": True
            }
        )

    async def _get_error_rate_card(self) -> DashboardWidget:
        """Generate error rate status card."""
        stats = await self.stats_collector.get_current_statistics()
        sparkline = await self._get_sparkline_data("error_rate", 20)

        return DashboardWidget(
            widget_id="error_rate_card",
            widget_type=WidgetType.STATUS_CARD,
            title="Error Rate",
            data={
                "value": stats.failure_rate,
                "formatted_value": f"{self._format_number(stats.failure_rate, 1)}%",
                "sparkline": sparkline,
                "timestamp": self._format_timestamp(stats.timestamp),
                "color": self._get_error_rate_color(stats.failure_rate)
            },
            config={
                "show_sparkline": True
            }
        )

    async def _get_success_rate_gauge(self) -> DashboardWidget:
        """Generate success rate gauge widget."""
        stats = await self.stats_collector.get_current_statistics()

        return DashboardWidget(
            widget_id="success_rate_card",
            widget_type=WidgetType.STATUS_CARD,
            title="Success Rate",
            data={
                "value": stats.success_rate,
                "formatted_value": f"{self._format_number(stats.success_rate, 1)}%",
                "timestamp": self._format_timestamp(stats.timestamp),
                "color": self._get_success_rate_color(stats.success_rate)
            }
        )

    # =========================================================================
    # LINE_CHART Widget Generators
    # =========================================================================

    async def _get_queue_size_chart(self) -> DashboardWidget:
        """Generate queue size time-series chart."""
        series_data = await self._get_time_series_data("queue_size", hours=24)

        return DashboardWidget(
            widget_id="queue_size_chart",
            widget_type=WidgetType.LINE_CHART,
            title="Queue Size (24h)",
            data={
                "series": [
                    {
                        "name": "Queue Size",
                        "data": series_data,
                        "color": "blue"
                    }
                ],
                "x_axis": {
                    "type": "datetime",
                    "label": "Time"
                },
                "y_axis": {
                    "label": "Items"
                }
            },
            config={
                "show_legend": True,
                "show_grid": True,
                "animate": True
            }
        )

    async def _get_processing_rate_chart(self) -> DashboardWidget:
        """Generate processing rate time-series chart."""
        series_data = await self._get_time_series_data("processing_rate", hours=24)

        return DashboardWidget(
            widget_id="processing_rate_chart",
            widget_type=WidgetType.LINE_CHART,
            title="Processing Rate (24h)",
            data={
                "series": [
                    {
                        "name": "Items/min",
                        "data": series_data,
                        "color": "green"
                    }
                ],
                "x_axis": {
                    "type": "datetime",
                    "label": "Time"
                },
                "y_axis": {
                    "label": "Items per Minute"
                }
            }
        )

    async def _get_error_rate_chart(self) -> DashboardWidget:
        """Generate error rate time-series chart."""
        series_data = await self._get_time_series_data("error_rate", hours=24)

        return DashboardWidget(
            widget_id="error_rate_chart",
            widget_type=WidgetType.LINE_CHART,
            title="Error Rate (24h)",
            data={
                "series": [
                    {
                        "name": "Error %",
                        "data": series_data,
                        "color": "red"
                    }
                ],
                "x_axis": {
                    "type": "datetime",
                    "label": "Time"
                },
                "y_axis": {
                    "label": "Error Rate (%)"
                }
            }
        )

    async def _get_latency_chart(self) -> DashboardWidget:
        """Generate latency time-series chart."""
        # Get latency metrics if available
        if self.performance_collector:
            latency_metrics = await self.performance_collector.get_latency_metrics(window_minutes=60)
            # Note: This is simplified - would need actual historical latency data
            series_data = [
                {"x": self._format_timestamp(datetime.now(timezone.utc)), "y": latency_metrics.avg_latency_ms}
            ]
        else:
            series_data = []

        return DashboardWidget(
            widget_id="latency_chart",
            widget_type=WidgetType.LINE_CHART,
            title="Queue Latency (1h)",
            data={
                "series": [
                    {
                        "name": "Avg Latency",
                        "data": series_data,
                        "color": "purple"
                    }
                ],
                "x_axis": {
                    "type": "datetime",
                    "label": "Time"
                },
                "y_axis": {
                    "label": "Latency (ms)"
                }
            }
        )

    async def _get_resource_usage_chart(self) -> DashboardWidget:
        """Generate resource usage time-series chart."""
        # Get current resource usage
        if self.performance_collector and self.performance_collector.enable_resource_tracking:
            resource_usage = self.performance_collector._get_resource_usage()
            cpu_value = resource_usage.get("cpu_percent", 0)
            memory_value = resource_usage.get("memory_percent", 0)
        else:
            cpu_value = 0
            memory_value = 0

        timestamp = self._format_timestamp(datetime.now(timezone.utc))

        return DashboardWidget(
            widget_id="resource_usage_chart",
            widget_type=WidgetType.LINE_CHART,
            title="Resource Usage",
            data={
                "series": [
                    {
                        "name": "CPU %",
                        "data": [{"x": timestamp, "y": cpu_value}],
                        "color": "orange"
                    },
                    {
                        "name": "Memory %",
                        "data": [{"x": timestamp, "y": memory_value}],
                        "color": "blue"
                    }
                ],
                "x_axis": {
                    "type": "datetime",
                    "label": "Time"
                },
                "y_axis": {
                    "label": "Percentage",
                    "max": 100
                }
            }
        )

    # =========================================================================
    # BAR_CHART Widget Generators
    # =========================================================================

    async def _get_slow_collections_bar(self) -> DashboardWidget:
        """Generate slow collections bar chart."""
        if not self.enable_bottleneck_detection or not self.bottleneck_detector:
            return DashboardWidget(
                widget_id="slow_collections_bar",
                widget_type=WidgetType.BAR_CHART,
                title="Slow Collections",
                data={"categories": [], "series": []},
                config={"message": "Bottleneck detection not enabled"}
            )

        slow_collections = await self.bottleneck_detector.identify_slow_collections()

        categories = [c.collection_name for c in slow_collections[:10]]
        values = [c.avg_time for c in slow_collections[:10]]

        return DashboardWidget(
            widget_id="slow_collections_bar",
            widget_type=WidgetType.BAR_CHART,
            title="Top 10 Slow Collections",
            data={
                "categories": categories,
                "series": [
                    {
                        "name": "Avg Processing Time",
                        "data": values,
                        "color": "red"
                    }
                ],
                "x_axis": {
                    "label": "Collection"
                },
                "y_axis": {
                    "label": "Time (ms)"
                }
            }
        )

    async def _get_error_categories_bar(self) -> DashboardWidget:
        """Generate error categories bar chart."""
        error_stats = await self.error_manager.get_error_stats()
        error_summary = type('obj', (), {'category_stats': error_stats.get('by_category', {})})()

        categories = []
        values = []

        for category, stats in list(error_summary.category_stats.items())[:10]:
            categories.append(category)
            values.append(stats["count"])

        return DashboardWidget(
            widget_id="error_categories_bar",
            widget_type=WidgetType.BAR_CHART,
            title="Top Error Categories",
            data={
                "categories": categories,
                "series": [
                    {
                        "name": "Error Count",
                        "data": values,
                        "color": "red"
                    }
                ],
                "x_axis": {
                    "label": "Category"
                },
                "y_axis": {
                    "label": "Count"
                }
            }
        )

    # =========================================================================
    # TABLE Widget Generators
    # =========================================================================

    async def _get_recent_errors_table(self) -> DashboardWidget:
        """Generate recent errors table."""
        recent_errors = await self.error_manager.get_errors(limit=20)
        recent_errors = [{"timestamp": e.timestamp, "severity": e.severity, "category": e.category, "message": e.message, "file_path": e.file_path} for e in recent_errors]

        rows = []
        for error in recent_errors:
            rows.append({
                "timestamp": self._format_timestamp(error["timestamp"]),
                "severity": error["severity"],
                "category": error["category"],
                "message": error["message"][:100],  # Truncate long messages
                "file_path": error["file_path"]
            })

        return DashboardWidget(
            widget_id="recent_errors_table",
            widget_type=WidgetType.TABLE,
            title="Recent Errors (Last 20)",
            data={
                "columns": [
                    {"key": "timestamp", "label": "Time", "width": "15%"},
                    {"key": "severity", "label": "Severity", "width": "10%"},
                    {"key": "category", "label": "Category", "width": "15%"},
                    {"key": "message", "label": "Message", "width": "40%"},
                    {"key": "file_path", "label": "File", "width": "20%"}
                ],
                "rows": rows
            },
            config={
                "sortable": True,
                "filterable": True,
                "row_highlighting": {
                    "critical": "red",
                    "error": "orange",
                    "warning": "yellow"
                }
            }
        )

    async def _get_slow_operations_table(self) -> DashboardWidget:
        """Generate slow operations table."""
        if not self.enable_bottleneck_detection or not self.bottleneck_detector:
            return DashboardWidget(
                widget_id="slow_operations_table",
                widget_type=WidgetType.TABLE,
                title="Slow Operations",
                data={"columns": [], "rows": []},
                config={"message": "Bottleneck detection not enabled"}
            )

        slow_ops = await self.bottleneck_detector.get_slowest_items(limit=10)

        rows = []
        for op in slow_ops:
            rows.append({
                "timestamp": self._format_timestamp(op.timestamp),
                "file_path": op.file_path,
                "operation": op.operation,
                "duration": f"{self._format_number(op.duration_ms, 0)}ms",
                "collection": op.collection_name
            })

        return DashboardWidget(
            widget_id="slow_operations_table",
            widget_type=WidgetType.TABLE,
            title="Top 10 Slow Operations",
            data={
                "columns": [
                    {"key": "timestamp", "label": "Time", "width": "15%"},
                    {"key": "file_path", "label": "File", "width": "35%"},
                    {"key": "operation", "label": "Operation", "width": "15%"},
                    {"key": "duration", "label": "Duration", "width": "15%"},
                    {"key": "collection", "label": "Collection", "width": "20%"}
                ],
                "rows": rows
            }
        )

    async def _get_bottleneck_table(self) -> DashboardWidget:
        """Generate bottleneck summary table."""
        if not self.enable_bottleneck_detection or not self.bottleneck_detector:
            return DashboardWidget(
                widget_id="bottleneck_table",
                widget_type=WidgetType.TABLE,
                title="Bottlenecks",
                data={"columns": [], "rows": []},
                config={"message": "Bottleneck detection not enabled"}
            )

        bottlenecks = await self.bottleneck_detector.identify_slow_operations()

        rows = []
        for bottleneck in bottlenecks[:10]:
            rows.append({
                "operation": bottleneck.operation_type,
                "avg_duration": f"{self._format_number(bottleneck.avg_duration, 0)}ms",
                "p95_duration": f"{self._format_number(bottleneck.p95_duration, 0)}ms",
                "count": bottleneck.count,
                "recommendation": bottleneck.recommendation[:100]
            })

        return DashboardWidget(
            widget_id="bottleneck_table",
            widget_type=WidgetType.TABLE,
            title="Bottleneck Summary",
            data={
                "columns": [
                    {"key": "operation", "label": "Operation", "width": "15%"},
                    {"key": "avg_duration", "label": "Avg Duration", "width": "15%"},
                    {"key": "p95_duration", "label": "P95 Duration", "width": "15%"},
                    {"key": "count", "label": "Count", "width": "10%"},
                    {"key": "recommendation", "label": "Recommendation", "width": "45%"}
                ],
                "rows": rows
            }
        )

    # =========================================================================
    # HEATMAP Widget Generators
    # =========================================================================

    async def _get_processing_heatmap(self) -> DashboardWidget:
        """Generate processing activity heatmap."""
        # This would require historical data aggregation by hour/day
        # Simplified implementation for now

        return DashboardWidget(
            widget_id="processing_heatmap",
            widget_type=WidgetType.HEATMAP,
            title="Processing Activity by Hour",
            data={
                "x_axis": ["00", "04", "08", "12", "16", "20"],  # Hours
                "y_axis": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],  # Days
                "data": [],  # Would be populated with actual activity data
                "color_scale": {
                    "min_color": "lightgreen",
                    "max_color": "darkgreen"
                }
            },
            config={
                "show_values": False,
                "message": "Historical activity tracking not yet implemented"
            }
        )

    async def _get_error_heatmap(self) -> DashboardWidget:
        """Generate error pattern heatmap."""
        return DashboardWidget(
            widget_id="error_heatmap",
            widget_type=WidgetType.HEATMAP,
            title="Error Patterns by Time",
            data={
                "x_axis": ["00", "04", "08", "12", "16", "20"],
                "y_axis": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                "data": [],
                "color_scale": {
                    "min_color": "lightyellow",
                    "max_color": "darkred"
                }
            },
            config={
                "show_values": False,
                "message": "Historical error tracking not yet implemented"
            }
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _get_sparkline_data(
        self,
        metric_name: str,
        points: int = 20
    ) -> List[float]:
        """
        Get sparkline data for a metric.

        Args:
            metric_name: Name of the metric
            points: Number of data points to retrieve

        Returns:
            List of metric values for sparkline
        """
        if not self.enable_trend_analysis or not self.trend_analyzer:
            return []

        try:
            # Get historical data
            historical_data = await self.trend_analyzer.get_historical_data(
                metric_name=metric_name,
                hours=1  # Last hour
            )

            if not historical_data:
                return []

            # Sample evenly to get desired number of points
            if len(historical_data) > points:
                step = len(historical_data) // points
                sampled = historical_data[::step][:points]
            else:
                sampled = historical_data

            return [dp.value for dp in sampled]

        except Exception as e:
            logger.warning(f"Failed to get sparkline data for {metric_name}: {e}")
            return []

    async def _get_time_series_data(
        self,
        metric_name: str,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get time-series data for chart.

        Args:
            metric_name: Name of the metric
            hours: Hours of history to retrieve

        Returns:
            List of {x: timestamp, y: value} points
        """
        if not self.enable_trend_analysis or not self.trend_analyzer:
            return []

        try:
            historical_data = await self.trend_analyzer.get_historical_data(
                metric_name=metric_name,
                hours=hours
            )

            return [
                {
                    "x": self._format_timestamp(dp.timestamp),
                    "y": self._format_number(dp.value, 2)
                }
                for dp in historical_data
            ]

        except Exception as e:
            logger.warning(f"Failed to get time-series data for {metric_name}: {e}")
            return []

    def _format_timestamp(self, dt: datetime) -> str:
        """
        Format timestamp in ISO 8601 format.

        Args:
            dt: Datetime to format

        Returns:
            ISO 8601 formatted string
        """
        return dt.isoformat()

    def _format_number(self, value: float, precision: int = 2) -> float:
        """
        Format number with appropriate precision.

        Args:
            value: Number to format
            precision: Decimal places

        Returns:
            Rounded number
        """
        return round(value, precision)

    def _get_queue_size_color(self, queue_size: int) -> str:
        """Get color code for queue size."""
        if queue_size < 1000:
            return "green"
        elif queue_size < 5000:
            return "yellow"
        elif queue_size < 10000:
            return "orange"
        else:
            return "red"

    def _get_error_rate_color(self, error_rate: float) -> str:
        """Get color code for error rate."""
        if error_rate < 1.0:
            return "green"
        elif error_rate < 5.0:
            return "yellow"
        elif error_rate < 10.0:
            return "orange"
        else:
            return "red"

    def _get_success_rate_color(self, success_rate: float) -> str:
        """Get color code for success rate."""
        if success_rate >= 95.0:
            return "green"
        elif success_rate >= 90.0:
            return "yellow"
        elif success_rate >= 80.0:
            return "orange"
        else:
            return "red"
