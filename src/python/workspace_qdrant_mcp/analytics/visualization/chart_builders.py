"""
Chart Builders for data visualization and interactive dashboards.

Provides comprehensive chart building capabilities including:
- Time series charts with trend lines
- Statistical distribution charts
- Performance monitoring charts
- Capacity utilization charts
- Pattern visualization charts
- Interactive dashboard components
"""

import json
import math
import statistics
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

import numpy as np


logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Enumeration for chart types."""
    LINE = "line"
    BAR = "bar"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    PIE = "pie"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    TIMESERIES = "timeseries"
    CANDLESTICK = "candlestick"
    AREA = "area"


class ChartTheme(Enum):
    """Enumeration for chart themes."""
    DEFAULT = "default"
    DARK = "dark"
    LIGHT = "light"
    MINIMAL = "minimal"
    COLORFUL = "colorful"
    PROFESSIONAL = "professional"


@dataclass
class ChartData:
    """Container for chart data."""

    x_values: List[Union[str, float, datetime]]
    y_values: List[Union[float, int]]
    labels: Optional[List[str]] = None
    colors: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    error_bars: Optional[List[Tuple[float, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'x_values': [x.isoformat() if isinstance(x, datetime) else x for x in self.x_values],
            'y_values': self.y_values,
            'labels': self.labels,
            'colors': self.colors,
            'metadata': self.metadata,
            'error_bars': self.error_bars
        }


@dataclass
class ChartConfig:
    """Configuration for chart appearance and behavior."""

    title: str
    x_label: str = ""
    y_label: str = ""
    theme: ChartTheme = ChartTheme.DEFAULT
    width: int = 800
    height: int = 600
    show_grid: bool = True
    show_legend: bool = True
    interactive: bool = True
    color_palette: Optional[List[str]] = None
    font_size: int = 12
    margin: Dict[str, int] = None

    def __post_init__(self):
        """Initialize default margins if not provided."""
        if self.margin is None:
            self.margin = {'top': 50, 'right': 50, 'bottom': 60, 'left': 80}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'title': self.title,
            'x_label': self.x_label,
            'y_label': self.y_label,
            'theme': self.theme.value,
            'width': self.width,
            'height': self.height,
            'show_grid': self.show_grid,
            'show_legend': self.show_legend,
            'interactive': self.interactive,
            'color_palette': self.color_palette,
            'font_size': self.font_size,
            'margin': self.margin
        }


@dataclass
class ChartOutput:
    """Container for chart output."""

    chart_type: ChartType
    config: ChartConfig
    data: ChartData
    html_content: str
    json_spec: Dict[str, Any]
    svg_content: Optional[str] = None
    png_base64: Optional[str] = None
    is_valid: bool = True
    error_message: Optional[str] = None
    render_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'chart_type': self.chart_type.value,
            'config': self.config.to_dict(),
            'data': self.data.to_dict(),
            'html_content': self.html_content,
            'json_spec': self.json_spec,
            'svg_content': self.svg_content,
            'png_base64': self.png_base64,
            'is_valid': self.is_valid,
            'error_message': self.error_message,
            'render_time_ms': self.render_time_ms
        }


class ChartBuilder:
    """
    Advanced chart building engine with comprehensive visualization capabilities.

    Handles large datasets, rendering failures, and provides interactive dashboard
    components with robust error handling.
    """

    def __init__(self, max_data_points: int = 10000, enable_streaming: bool = True):
        """
        Initialize chart builder.

        Args:
            max_data_points: Maximum data points for performance optimization
            enable_streaming: Enable streaming for large datasets
        """
        self.max_data_points = max_data_points
        self.enable_streaming = enable_streaming
        self._color_palettes = self._initialize_color_palettes()
        self._chart_templates = self._initialize_chart_templates()

    def create_time_series_chart(self,
                                data: List[Union[int, float]],
                                timestamps: Optional[List[datetime]] = None,
                                config: Optional[ChartConfig] = None,
                                trend_line: bool = True,
                                confidence_bands: Optional[List[Tuple[float, float]]] = None) -> ChartOutput:
        """
        Create time series chart with trend analysis.

        Args:
            data: Time series data values
            timestamps: Optional timestamps for each data point
            config: Chart configuration
            trend_line: Whether to include trend line
            confidence_bands: Optional confidence interval bands

        Returns:
            ChartOutput with time series visualization
        """
        start_time = datetime.now()

        # Default configuration
        if config is None:
            config = ChartConfig(
                title="Time Series Analysis",
                x_label="Time",
                y_label="Value"
            )

        try:
            # Validate and clean data
            valid_data, valid_timestamps = self._validate_time_series_data(data, timestamps)

            if not valid_data:
                return self._create_error_chart("No valid data points for time series chart")

            # Handle large datasets
            if len(valid_data) > self.max_data_points:
                valid_data, valid_timestamps = self._downsample_time_series(
                    valid_data, valid_timestamps, self.max_data_points
                )
                logger.info(f"Downsampled time series data to {len(valid_data)} points")

            # Create chart data
            chart_data = ChartData(
                x_values=valid_timestamps or list(range(len(valid_data))),
                y_values=valid_data,
                labels=[f"Point {i}" for i in range(len(valid_data))],
                colors=None,
                metadata={'data_points': len(valid_data)}
            )

            # Add confidence bands if provided
            if confidence_bands:
                chart_data.error_bars = confidence_bands[:len(valid_data)]

            # Generate chart specification
            json_spec = self._create_time_series_spec(chart_data, config, trend_line)

            # Generate HTML content
            html_content = self._generate_html_chart(json_spec, config)

            # Calculate render time
            render_time = (datetime.now() - start_time).total_seconds() * 1000

            return ChartOutput(
                chart_type=ChartType.TIMESERIES,
                config=config,
                data=chart_data,
                html_content=html_content,
                json_spec=json_spec,
                render_time_ms=render_time
            )

        except Exception as e:
            logger.error(f"Error creating time series chart: {e}")
            return self._create_error_chart(f"Time series chart error: {str(e)}")

    def create_statistical_chart(self,
                                data: List[Union[int, float]],
                                chart_type: ChartType = ChartType.HISTOGRAM,
                                config: Optional[ChartConfig] = None,
                                bins: Optional[int] = None,
                                show_statistics: bool = True) -> ChartOutput:
        """
        Create statistical distribution chart.

        Args:
            data: Statistical data values
            chart_type: Type of statistical chart (histogram, box plot, etc.)
            config: Chart configuration
            bins: Number of bins for histogram (auto-calculated if None)
            show_statistics: Whether to show statistical annotations

        Returns:
            ChartOutput with statistical visualization
        """
        start_time = datetime.now()

        if config is None:
            config = ChartConfig(
                title="Statistical Distribution",
                x_label="Value",
                y_label="Frequency"
            )

        try:
            # Validate and clean data
            valid_data = [float(x) for x in data if isinstance(x, (int, float)) and math.isfinite(x)]

            if not valid_data:
                return self._create_error_chart("No valid data points for statistical chart")

            if chart_type == ChartType.HISTOGRAM:
                return self._create_histogram_chart(valid_data, config, bins, show_statistics, start_time)
            elif chart_type == ChartType.SCATTER:
                return self._create_scatter_chart(valid_data, config, start_time)
            else:
                return self._create_error_chart(f"Unsupported statistical chart type: {chart_type}")

        except Exception as e:
            logger.error(f"Error creating statistical chart: {e}")
            return self._create_error_chart(f"Statistical chart error: {str(e)}")

    def create_performance_chart(self,
                               metrics_data: Dict[str, List[Union[int, float]]],
                               config: Optional[ChartConfig] = None,
                               thresholds: Optional[Dict[str, float]] = None,
                               show_trends: bool = True) -> ChartOutput:
        """
        Create performance monitoring chart.

        Args:
            metrics_data: Dictionary of metric names to value lists
            config: Chart configuration
            thresholds: Optional performance thresholds for each metric
            show_trends: Whether to show trend lines

        Returns:
            ChartOutput with performance visualization
        """
        start_time = datetime.now()

        if config is None:
            config = ChartConfig(
                title="Performance Metrics",
                x_label="Time Period",
                y_label="Performance Value"
            )

        try:
            if not metrics_data:
                return self._create_error_chart("No metrics data provided for performance chart")

            # Validate metrics data
            valid_metrics = {}
            for metric_name, values in metrics_data.items():
                valid_values = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
                if valid_values:
                    valid_metrics[metric_name] = valid_values

            if not valid_metrics:
                return self._create_error_chart("No valid metrics data for performance chart")

            # Create multi-series chart data
            json_spec = self._create_performance_spec(valid_metrics, config, thresholds, show_trends)
            html_content = self._generate_html_chart(json_spec, config)

            # Create consolidated chart data
            max_length = max(len(values) for values in valid_metrics.values())
            chart_data = ChartData(
                x_values=list(range(max_length)),
                y_values=[],  # Multi-series, stored in json_spec
                labels=list(valid_metrics.keys()),
                metadata={'metrics_count': len(valid_metrics), 'max_points': max_length}
            )

            render_time = (datetime.now() - start_time).total_seconds() * 1000

            return ChartOutput(
                chart_type=ChartType.LINE,
                config=config,
                data=chart_data,
                html_content=html_content,
                json_spec=json_spec,
                render_time_ms=render_time
            )

        except Exception as e:
            logger.error(f"Error creating performance chart: {e}")
            return self._create_error_chart(f"Performance chart error: {str(e)}")

    def create_capacity_chart(self,
                            utilization_data: List[Union[int, float]],
                            predicted_utilization: Optional[List[Union[int, float]]] = None,
                            config: Optional[ChartConfig] = None,
                            capacity_threshold: float = 0.8,
                            show_predictions: bool = True) -> ChartOutput:
        """
        Create capacity utilization chart.

        Args:
            utilization_data: Historical utilization data (0.0 to 1.0)
            predicted_utilization: Optional predicted future utilization
            config: Chart configuration
            capacity_threshold: Threshold for capacity warnings
            show_predictions: Whether to show prediction data

        Returns:
            ChartOutput with capacity visualization
        """
        start_time = datetime.now()

        if config is None:
            config = ChartConfig(
                title="Capacity Utilization",
                x_label="Time Period",
                y_label="Utilization %"
            )

        try:
            # Validate and clean utilization data
            valid_historical = [max(0.0, min(1.0, float(x))) for x in utilization_data
                              if isinstance(x, (int, float)) and math.isfinite(x)]

            if not valid_historical:
                return self._create_error_chart("No valid utilization data for capacity chart")

            # Validate predicted data if provided
            valid_predicted = []
            if predicted_utilization and show_predictions:
                valid_predicted = [max(0.0, min(1.0, float(x))) for x in predicted_utilization
                                 if isinstance(x, (int, float)) and math.isfinite(x)]

            # Convert to percentages for display
            historical_percent = [x * 100 for x in valid_historical]
            predicted_percent = [x * 100 for x in valid_predicted] if valid_predicted else []

            # Create chart specification
            json_spec = self._create_capacity_spec(
                historical_percent, predicted_percent, config, capacity_threshold * 100
            )
            html_content = self._generate_html_chart(json_spec, config)

            # Create chart data
            all_data = historical_percent + predicted_percent
            chart_data = ChartData(
                x_values=list(range(len(all_data))),
                y_values=all_data,
                labels=["Historical"] * len(historical_percent) + ["Predicted"] * len(predicted_percent),
                metadata={
                    'threshold': capacity_threshold * 100,
                    'current_utilization': valid_historical[-1] * 100 if valid_historical else 0,
                    'has_predictions': len(valid_predicted) > 0
                }
            )

            render_time = (datetime.now() - start_time).total_seconds() * 1000

            return ChartOutput(
                chart_type=ChartType.AREA,
                config=config,
                data=chart_data,
                html_content=html_content,
                json_spec=json_spec,
                render_time_ms=render_time
            )

        except Exception as e:
            logger.error(f"Error creating capacity chart: {e}")
            return self._create_error_chart(f"Capacity chart error: {str(e)}")

    def create_pattern_chart(self,
                           data: List[Union[int, float]],
                           patterns: List[Dict[str, Any]],
                           config: Optional[ChartConfig] = None,
                           highlight_patterns: bool = True) -> ChartOutput:
        """
        Create pattern visualization chart.

        Args:
            data: Time series data
            patterns: List of detected patterns with metadata
            config: Chart configuration
            highlight_patterns: Whether to highlight detected patterns

        Returns:
            ChartOutput with pattern visualization
        """
        start_time = datetime.now()

        if config is None:
            config = ChartConfig(
                title="Pattern Analysis",
                x_label="Time",
                y_label="Value"
            )

        try:
            # Validate data
            valid_data = [float(x) for x in data if isinstance(x, (int, float)) and math.isfinite(x)]

            if not valid_data:
                return self._create_error_chart("No valid data for pattern chart")

            # Create pattern visualization specification
            json_spec = self._create_pattern_spec(valid_data, patterns, config, highlight_patterns)
            html_content = self._generate_html_chart(json_spec, config)

            # Create chart data with pattern metadata
            chart_data = ChartData(
                x_values=list(range(len(valid_data))),
                y_values=valid_data,
                labels=[f"Point {i}" for i in range(len(valid_data))],
                metadata={
                    'patterns_detected': len(patterns),
                    'patterns': patterns
                }
            )

            render_time = (datetime.now() - start_time).total_seconds() * 1000

            return ChartOutput(
                chart_type=ChartType.LINE,
                config=config,
                data=chart_data,
                html_content=html_content,
                json_spec=json_spec,
                render_time_ms=render_time
            )

        except Exception as e:
            logger.error(f"Error creating pattern chart: {e}")
            return self._create_error_chart(f"Pattern chart error: {str(e)}")

    def create_gauge_chart(self,
                         current_value: float,
                         min_value: float = 0,
                         max_value: float = 100,
                         thresholds: Optional[List[Tuple[float, str]]] = None,
                         config: Optional[ChartConfig] = None) -> ChartOutput:
        """
        Create gauge/meter chart for single value display.

        Args:
            current_value: Current value to display
            min_value: Minimum value on the gauge
            max_value: Maximum value on the gauge
            thresholds: List of (value, color) threshold pairs
            config: Chart configuration

        Returns:
            ChartOutput with gauge visualization
        """
        start_time = datetime.now()

        if config is None:
            config = ChartConfig(
                title="Performance Gauge",
                width=400,
                height=300
            )

        try:
            # Validate values
            if not isinstance(current_value, (int, float)) or not math.isfinite(current_value):
                return self._create_error_chart("Invalid current value for gauge chart")

            if min_value >= max_value:
                return self._create_error_chart("Invalid range: min_value must be less than max_value")

            # Default thresholds if not provided
            if thresholds is None:
                range_size = max_value - min_value
                thresholds = [
                    (min_value + range_size * 0.7, "green"),
                    (min_value + range_size * 0.9, "yellow"),
                    (max_value, "red")
                ]

            # Create gauge specification
            json_spec = self._create_gauge_spec(current_value, min_value, max_value, thresholds, config)
            html_content = self._generate_html_chart(json_spec, config)

            # Create chart data
            chart_data = ChartData(
                x_values=[0],
                y_values=[current_value],
                metadata={
                    'min_value': min_value,
                    'max_value': max_value,
                    'thresholds': thresholds,
                    'percentage': ((current_value - min_value) / (max_value - min_value)) * 100
                }
            )

            render_time = (datetime.now() - start_time).total_seconds() * 1000

            return ChartOutput(
                chart_type=ChartType.GAUGE,
                config=config,
                data=chart_data,
                html_content=html_content,
                json_spec=json_spec,
                render_time_ms=render_time
            )

        except Exception as e:
            logger.error(f"Error creating gauge chart: {e}")
            return self._create_error_chart(f"Gauge chart error: {str(e)}")

    def _validate_time_series_data(self, data: List[Union[int, float]],
                                 timestamps: Optional[List[datetime]]) -> Tuple[List[float], List[datetime]]:
        """Validate and clean time series data."""
        valid_data = []
        valid_timestamps = []

        for i, value in enumerate(data):
            if isinstance(value, (int, float)) and math.isfinite(value):
                valid_data.append(float(value))

                if timestamps and i < len(timestamps) and isinstance(timestamps[i], datetime):
                    valid_timestamps.append(timestamps[i])
                else:
                    valid_timestamps.append(datetime.now() + timedelta(days=i))

        return valid_data, valid_timestamps

    def _downsample_time_series(self, data: List[float], timestamps: List[datetime],
                               max_points: int) -> Tuple[List[float], List[datetime]]:
        """Downsample time series data to reduce chart complexity."""
        if len(data) <= max_points:
            return data, timestamps

        # Use simple decimation - could be improved with more sophisticated algorithms
        step = len(data) // max_points
        downsampled_data = data[::step][:max_points]
        downsampled_timestamps = timestamps[::step][:max_points]

        return downsampled_data, downsampled_timestamps

    def _create_time_series_spec(self, data: ChartData, config: ChartConfig, trend_line: bool) -> Dict[str, Any]:
        """Create Vega-Lite specification for time series chart."""
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Time Series Chart",
            "width": config.width - config.margin['left'] - config.margin['right'],
            "height": config.height - config.margin['top'] - config.margin['bottom'],
            "title": config.title,
            "data": {
                "values": [
                    {
                        "time": x.isoformat() if isinstance(x, datetime) else str(x),
                        "value": y,
                        "index": i
                    }
                    for i, (x, y) in enumerate(zip(data.x_values, data.y_values))
                ]
            },
            "layer": []
        }

        # Main line layer
        line_layer = {
            "mark": {
                "type": "line",
                "point": True,
                "interpolate": "monotone",
                "strokeWidth": 2
            },
            "encoding": {
                "x": {
                    "field": "time",
                    "type": "temporal" if isinstance(data.x_values[0], datetime) else "ordinal",
                    "title": config.x_label,
                    "axis": {"grid": config.show_grid}
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "title": config.y_label,
                    "axis": {"grid": config.show_grid}
                },
                "color": {"value": "#1f77b4"}
            }
        }

        spec["layer"].append(line_layer)

        # Add trend line if requested
        if trend_line:
            trend_layer = {
                "mark": {
                    "type": "line",
                    "strokeDash": [3, 3],
                    "strokeWidth": 2
                },
                "transform": [{
                    "regression": "value",
                    "on": "index"
                }],
                "encoding": {
                    "x": {
                        "field": "time",
                        "type": "temporal" if isinstance(data.x_values[0], datetime) else "ordinal"
                    },
                    "y": {
                        "field": "value",
                        "type": "quantitative"
                    },
                    "color": {"value": "#d62728"}
                }
            }
            spec["layer"].append(trend_layer)

        # Add confidence bands if present
        if data.error_bars:
            confidence_data = []
            for i, ((lower, upper), x, y) in enumerate(zip(data.error_bars, data.x_values, data.y_values)):
                confidence_data.append({
                    "time": x.isoformat() if isinstance(x, datetime) else str(x),
                    "lower": lower,
                    "upper": upper,
                    "value": y,
                    "index": i
                })

            confidence_layer = {
                "data": {"values": confidence_data},
                "mark": {
                    "type": "area",
                    "opacity": 0.2
                },
                "encoding": {
                    "x": {
                        "field": "time",
                        "type": "temporal" if isinstance(data.x_values[0], datetime) else "ordinal"
                    },
                    "y": {
                        "field": "lower",
                        "type": "quantitative"
                    },
                    "y2": {
                        "field": "upper",
                        "type": "quantitative"
                    },
                    "color": {"value": "#1f77b4"}
                }
            }
            spec["layer"].insert(0, confidence_layer)  # Add as background

        return spec

    def _create_histogram_chart(self, data: List[float], config: ChartConfig,
                               bins: Optional[int], show_statistics: bool, start_time: datetime) -> ChartOutput:
        """Create histogram chart."""
        try:
            # Calculate optimal number of bins if not provided
            if bins is None:
                bins = min(50, max(10, int(math.sqrt(len(data)))))

            # Calculate histogram
            min_val, max_val = min(data), max(data)
            bin_width = (max_val - min_val) / bins
            bin_edges = [min_val + i * bin_width for i in range(bins + 1)]
            bin_counts = [0] * bins

            for value in data:
                bin_idx = min(bins - 1, int((value - min_val) / bin_width))
                bin_counts[bin_idx] += 1

            # Create histogram data
            hist_data = []
            for i in range(bins):
                hist_data.append({
                    "bin_start": bin_edges[i],
                    "bin_end": bin_edges[i + 1],
                    "count": bin_counts[i],
                    "bin_center": (bin_edges[i] + bin_edges[i + 1]) / 2
                })

            # Create specification
            json_spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "description": "Histogram Chart",
                "width": config.width - config.margin['left'] - config.margin['right'],
                "height": config.height - config.margin['top'] - config.margin['bottom'],
                "title": config.title,
                "data": {"values": hist_data},
                "mark": {
                    "type": "bar",
                    "binSpacing": 1,
                    "stroke": "white",
                    "strokeWidth": 1
                },
                "encoding": {
                    "x": {
                        "field": "bin_center",
                        "type": "quantitative",
                        "title": config.x_label,
                        "axis": {"grid": config.show_grid}
                    },
                    "y": {
                        "field": "count",
                        "type": "quantitative",
                        "title": config.y_label,
                        "axis": {"grid": config.show_grid}
                    },
                    "color": {"value": "#1f77b4"}
                }
            }

            # Add statistical annotations if requested
            if show_statistics:
                mean_val = statistics.mean(data)
                std_val = statistics.stdev(data) if len(data) > 1 else 0

                # Add mean line
                json_spec["layer"] = [
                    json_spec.copy(),
                    {
                        "data": {"values": [{"mean": mean_val}]},
                        "mark": {
                            "type": "rule",
                            "color": "red",
                            "strokeWidth": 2,
                            "strokeDash": [4, 4]
                        },
                        "encoding": {
                            "x": {
                                "field": "mean",
                                "type": "quantitative"
                            }
                        }
                    }
                ]

                # Remove the mark from the main spec since we're using layers
                del json_spec["mark"]
                del json_spec["encoding"]

            html_content = self._generate_html_chart(json_spec, config)

            chart_data = ChartData(
                x_values=[d["bin_center"] for d in hist_data],
                y_values=[d["count"] for d in hist_data],
                metadata={
                    'bins': bins,
                    'mean': statistics.mean(data),
                    'std': statistics.stdev(data) if len(data) > 1 else 0,
                    'min': min_val,
                    'max': max_val,
                    'total_count': len(data)
                }
            )

            render_time = (datetime.now() - start_time).total_seconds() * 1000

            return ChartOutput(
                chart_type=ChartType.HISTOGRAM,
                config=config,
                data=chart_data,
                html_content=html_content,
                json_spec=json_spec,
                render_time_ms=render_time
            )

        except Exception as e:
            logger.error(f"Error creating histogram: {e}")
            return self._create_error_chart(f"Histogram error: {str(e)}")

    def _create_scatter_chart(self, data: List[float], config: ChartConfig, start_time: datetime) -> ChartOutput:
        """Create scatter plot chart."""
        try:
            # For single series data, create index vs value scatter
            scatter_data = [{"index": i, "value": val} for i, val in enumerate(data)]

            json_spec = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "description": "Scatter Plot",
                "width": config.width - config.margin['left'] - config.margin['right'],
                "height": config.height - config.margin['top'] - config.margin['bottom'],
                "title": config.title,
                "data": {"values": scatter_data},
                "mark": {
                    "type": "circle",
                    "size": 60,
                    "opacity": 0.7
                },
                "encoding": {
                    "x": {
                        "field": "index",
                        "type": "quantitative",
                        "title": config.x_label or "Index",
                        "axis": {"grid": config.show_grid}
                    },
                    "y": {
                        "field": "value",
                        "type": "quantitative",
                        "title": config.y_label,
                        "axis": {"grid": config.show_grid}
                    },
                    "color": {"value": "#1f77b4"}
                }
            }

            html_content = self._generate_html_chart(json_spec, config)

            chart_data = ChartData(
                x_values=list(range(len(data))),
                y_values=data,
                metadata={'data_points': len(data)}
            )

            render_time = (datetime.now() - start_time).total_seconds() * 1000

            return ChartOutput(
                chart_type=ChartType.SCATTER,
                config=config,
                data=chart_data,
                html_content=html_content,
                json_spec=json_spec,
                render_time_ms=render_time
            )

        except Exception as e:
            logger.error(f"Error creating scatter chart: {e}")
            return self._create_error_chart(f"Scatter chart error: {str(e)}")

    def _create_performance_spec(self, metrics_data: Dict[str, List[float]], config: ChartConfig,
                               thresholds: Optional[Dict[str, float]], show_trends: bool) -> Dict[str, Any]:
        """Create performance chart specification."""
        # Prepare data for multiple series
        chart_data = []
        colors = self._get_color_palette(len(metrics_data))

        for i, (metric_name, values) in enumerate(metrics_data.items()):
            for j, value in enumerate(values):
                chart_data.append({
                    "time": j,
                    "value": value,
                    "metric": metric_name,
                    "color": colors[i % len(colors)]
                })

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Performance Metrics Chart",
            "width": config.width - config.margin['left'] - config.margin['right'],
            "height": config.height - config.margin['top'] - config.margin['bottom'],
            "title": config.title,
            "data": {"values": chart_data},
            "mark": {
                "type": "line",
                "point": True,
                "strokeWidth": 2
            },
            "encoding": {
                "x": {
                    "field": "time",
                    "type": "ordinal",
                    "title": config.x_label,
                    "axis": {"grid": config.show_grid}
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "title": config.y_label,
                    "axis": {"grid": config.show_grid}
                },
                "color": {
                    "field": "metric",
                    "type": "nominal",
                    "legend": {"title": "Metrics"} if config.show_legend else None
                }
            }
        }

        # Add threshold lines if provided
        if thresholds:
            layers = [spec]
            for metric_name, threshold_value in thresholds.items():
                threshold_layer = {
                    "data": {"values": [{"threshold": threshold_value, "metric": metric_name}]},
                    "mark": {
                        "type": "rule",
                        "strokeDash": [5, 5],
                        "strokeWidth": 2,
                        "opacity": 0.7
                    },
                    "encoding": {
                        "y": {
                            "field": "threshold",
                            "type": "quantitative"
                        },
                        "color": {"value": "red"}
                    }
                }
                layers.append(threshold_layer)

            spec = {"layer": layers}

        return spec

    def _create_capacity_spec(self, historical: List[float], predicted: List[float],
                            config: ChartConfig, threshold: float) -> Dict[str, Any]:
        """Create capacity utilization chart specification."""
        # Prepare data
        chart_data = []

        # Historical data
        for i, value in enumerate(historical):
            chart_data.append({
                "time": i,
                "utilization": value,
                "type": "Historical",
                "color": "#1f77b4"
            })

        # Predicted data
        for i, value in enumerate(predicted):
            chart_data.append({
                "time": len(historical) + i,
                "utilization": value,
                "type": "Predicted",
                "color": "#ff7f0e"
            })

        # Threshold line data
        max_time = len(historical) + len(predicted) - 1
        threshold_data = [
            {"time": 0, "threshold": threshold},
            {"time": max_time, "threshold": threshold}
        ]

        layers = [
            {
                "data": {"values": chart_data},
                "mark": {
                    "type": "area",
                    "opacity": 0.7,
                    "line": True
                },
                "encoding": {
                    "x": {
                        "field": "time",
                        "type": "ordinal",
                        "title": config.x_label,
                        "axis": {"grid": config.show_grid}
                    },
                    "y": {
                        "field": "utilization",
                        "type": "quantitative",
                        "title": config.y_label,
                        "scale": {"domain": [0, 100]},
                        "axis": {"grid": config.show_grid}
                    },
                    "color": {
                        "field": "type",
                        "type": "nominal",
                        "legend": {"title": "Data Type"} if config.show_legend else None
                    }
                }
            },
            {
                "data": {"values": threshold_data},
                "mark": {
                    "type": "line",
                    "strokeDash": [4, 4],
                    "strokeWidth": 3,
                    "color": "red"
                },
                "encoding": {
                    "x": {
                        "field": "time",
                        "type": "ordinal"
                    },
                    "y": {
                        "field": "threshold",
                        "type": "quantitative"
                    }
                }
            }
        ]

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Capacity Utilization Chart",
            "width": config.width - config.margin['left'] - config.margin['right'],
            "height": config.height - config.margin['top'] - config.margin['bottom'],
            "title": config.title,
            "layer": layers
        }

        return spec

    def _create_pattern_spec(self, data: List[float], patterns: List[Dict[str, Any]],
                           config: ChartConfig, highlight_patterns: bool) -> Dict[str, Any]:
        """Create pattern visualization specification."""
        # Main data
        chart_data = [{"index": i, "value": value} for i, value in enumerate(data)]

        layers = [
            {
                "data": {"values": chart_data},
                "mark": {
                    "type": "line",
                    "strokeWidth": 2,
                    "color": "#1f77b4"
                },
                "encoding": {
                    "x": {
                        "field": "index",
                        "type": "quantitative",
                        "title": config.x_label,
                        "axis": {"grid": config.show_grid}
                    },
                    "y": {
                        "field": "value",
                        "type": "quantitative",
                        "title": config.y_label,
                        "axis": {"grid": config.show_grid}
                    }
                }
            }
        ]

        # Add pattern highlights if requested
        if highlight_patterns:
            pattern_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

            for i, pattern in enumerate(patterns):
                if 'start_index' in pattern and 'end_index' in pattern:
                    start_idx = pattern.get('start_index', 0)
                    end_idx = pattern.get('end_index', len(data) - 1)

                    # Create highlight area
                    highlight_data = []
                    for j in range(start_idx, min(end_idx + 1, len(data))):
                        highlight_data.append({
                            "index": j,
                            "value": data[j],
                            "pattern_type": pattern.get('pattern_type', 'unknown')
                        })

                    if highlight_data:
                        highlight_layer = {
                            "data": {"values": highlight_data},
                            "mark": {
                                "type": "area",
                                "opacity": 0.3,
                                "color": pattern_colors[i % len(pattern_colors)]
                            },
                            "encoding": {
                                "x": {
                                    "field": "index",
                                    "type": "quantitative"
                                },
                                "y": {
                                    "field": "value",
                                    "type": "quantitative"
                                }
                            }
                        }
                        layers.append(highlight_layer)

        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Pattern Analysis Chart",
            "width": config.width - config.margin['left'] - config.margin['right'],
            "height": config.height - config.margin['top'] - config.margin['bottom'],
            "title": config.title,
            "layer": layers
        }

        return spec

    def _create_gauge_spec(self, current_value: float, min_value: float, max_value: float,
                         thresholds: List[Tuple[float, str]], config: ChartConfig) -> Dict[str, Any]:
        """Create gauge chart specification."""
        # Calculate angle for current value
        value_range = max_value - min_value
        value_ratio = (current_value - min_value) / value_range
        angle = value_ratio * 180 - 90  # -90 to 90 degrees

        # Create gauge data
        gauge_data = [
            {
                "category": "current",
                "value": current_value,
                "min": min_value,
                "max": max_value,
                "angle": angle,
                "ratio": value_ratio
            }
        ]

        # Simple gauge representation using arc marks
        spec = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Gauge Chart",
            "width": config.width,
            "height": config.height,
            "title": config.title,
            "data": {"values": gauge_data},
            "mark": {
                "type": "arc",
                "innerRadius": 50,
                "outerRadius": 100,
                "theta": {"field": "ratio", "type": "quantitative", "scale": {"range": [0, 3.14159]}},
                "color": self._get_gauge_color(current_value, thresholds)
            },
            "encoding": {
                "color": {
                    "value": self._get_gauge_color(current_value, thresholds)
                }
            }
        }

        return spec

    def _get_gauge_color(self, value: float, thresholds: List[Tuple[float, str]]) -> str:
        """Get color for gauge based on value and thresholds."""
        for threshold_value, color in thresholds:
            if value <= threshold_value:
                return color
        return thresholds[-1][1] if thresholds else "#1f77b4"

    def _generate_html_chart(self, json_spec: Dict[str, Any], config: ChartConfig) -> str:
        """Generate HTML content for chart visualization."""
        html_template = f'''
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
    <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        #vis {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <div id="vis"></div>
    <script type="text/javascript">
        var spec = {json.dumps(json_spec, indent=2)};
        vegaEmbed('#vis', spec, {{
            actions: {str(config.interactive).lower()},
            theme: '{config.theme.value}'
        }}).catch(console.error);
    </script>
</body>
</html>'''
        return html_template

    def _create_error_chart(self, error_message: str) -> ChartOutput:
        """Create error chart output."""
        error_config = ChartConfig(title="Chart Error", width=400, height=200)
        error_data = ChartData(x_values=[0], y_values=[0])

        error_html = f'''
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            margin: 0;
            background-color: #f8f8f8;
        }}
        .error-message {{
            background-color: #ffebee;
            border: 1px solid #f44336;
            border-radius: 4px;
            padding: 20px;
            color: #d32f2f;
            text-align: center;
            max-width: 400px;
        }}
    </style>
</head>
<body>
    <div class="error-message">
        <h3>Chart Generation Error</h3>
        <p>{error_message}</p>
    </div>
</body>
</html>'''

        return ChartOutput(
            chart_type=ChartType.LINE,
            config=error_config,
            data=error_data,
            html_content=error_html,
            json_spec={},
            is_valid=False,
            error_message=error_message
        )

    def _get_color_palette(self, num_colors: int) -> List[str]:
        """Get color palette for charts."""
        default_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ]

        if num_colors <= len(default_colors):
            return default_colors[:num_colors]
        else:
            # Generate additional colors if needed
            extended_colors = default_colors.copy()
            for i in range(len(default_colors), num_colors):
                # Generate colors using HSL
                hue = (i * 137.508) % 360  # Golden angle approximation
                extended_colors.append(f"hsl({hue}, 70%, 50%)")
            return extended_colors

    def _initialize_color_palettes(self) -> Dict[str, List[str]]:
        """Initialize predefined color palettes."""
        return {
            'default': ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            'pastel': ["#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5"],
            'dark': ["#1f1f1f", "#ff6b6b", "#4ecdc4", "#45b7d1", "#96ceb4"],
            'professional': ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83"]
        }

    def _initialize_chart_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize chart templates."""
        return {
            'time_series': {
                'mark': {'type': 'line', 'point': True},
                'encoding': {
                    'x': {'type': 'temporal'},
                    'y': {'type': 'quantitative'}
                }
            },
            'bar_chart': {
                'mark': {'type': 'bar'},
                'encoding': {
                    'x': {'type': 'ordinal'},
                    'y': {'type': 'quantitative'}
                }
            }
        }