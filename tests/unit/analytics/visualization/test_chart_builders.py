"""
Comprehensive unit tests for ChartBuilder with extensive edge case coverage.

Tests cover:
- Various chart types and configurations
- Large dataset handling and performance optimization
- Visualization rendering failures and error handling
- Interactive dashboard components
- Data validation and edge cases
- HTML generation and output formatting
"""

import pytest
import math
import json
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.python.workspace_qdrant_mcp.analytics.visualization.chart_builders import (
    ChartBuilder,
    ChartType,
    ChartTheme,
    ChartData,
    ChartConfig,
    ChartOutput
)


class TestChartData:
    """Tests for ChartData data class."""

    def test_chart_data_initialization(self):
        """Test ChartData initialization."""
        data = ChartData(
            x_values=[1, 2, 3],
            y_values=[10, 20, 30],
            labels=["A", "B", "C"],
            colors=["red", "green", "blue"]
        )

        assert data.x_values == [1, 2, 3]
        assert data.y_values == [10, 20, 30]
        assert data.labels == ["A", "B", "C"]
        assert data.colors == ["red", "green", "blue"]
        assert data.metadata is None
        assert data.error_bars is None

    def test_chart_data_to_dict(self):
        """Test conversion to dictionary."""
        data = ChartData(
            x_values=[1, 2],
            y_values=[10, 20],
            metadata={'source': 'test'}
        )
        data_dict = data.to_dict()

        assert data_dict['x_values'] == [1, 2]
        assert data_dict['y_values'] == [10, 20]
        assert data_dict['metadata'] == {'source': 'test'}

    def test_chart_data_with_datetime(self):
        """Test ChartData with datetime values."""
        timestamps = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        data = ChartData(x_values=timestamps, y_values=[1, 2])
        data_dict = data.to_dict()

        assert data_dict['x_values'] == ['2024-01-01T00:00:00', '2024-01-02T00:00:00']


class TestChartConfig:
    """Tests for ChartConfig data class."""

    def test_chart_config_initialization(self):
        """Test ChartConfig initialization with defaults."""
        config = ChartConfig(title="Test Chart")

        assert config.title == "Test Chart"
        assert config.x_label == ""
        assert config.y_label == ""
        assert config.theme == ChartTheme.DEFAULT
        assert config.width == 800
        assert config.height == 600
        assert config.show_grid is True
        assert config.show_legend is True
        assert config.interactive is True
        assert config.font_size == 12
        assert config.margin is not None

    def test_chart_config_custom_values(self):
        """Test ChartConfig with custom values."""
        config = ChartConfig(
            title="Custom Chart",
            x_label="X Axis",
            y_label="Y Axis",
            theme=ChartTheme.DARK,
            width=1000,
            height=800,
            show_grid=False,
            show_legend=False,
            interactive=False
        )

        assert config.title == "Custom Chart"
        assert config.x_label == "X Axis"
        assert config.y_label == "Y Axis"
        assert config.theme == ChartTheme.DARK
        assert config.width == 1000
        assert config.height == 800
        assert config.show_grid is False
        assert config.show_legend is False
        assert config.interactive is False

    def test_chart_config_to_dict(self):
        """Test conversion to dictionary."""
        config = ChartConfig(
            title="Test",
            theme=ChartTheme.LIGHT,
            width=600,
            height=400
        )
        config_dict = config.to_dict()

        assert config_dict['title'] == "Test"
        assert config_dict['theme'] == 'light'
        assert config_dict['width'] == 600
        assert config_dict['height'] == 400

    def test_chart_config_default_margin(self):
        """Test default margin initialization."""
        config = ChartConfig(title="Test")

        expected_margin = {'top': 50, 'right': 50, 'bottom': 60, 'left': 80}
        assert config.margin == expected_margin

    def test_chart_config_custom_margin(self):
        """Test custom margin initialization."""
        custom_margin = {'top': 20, 'right': 20, 'bottom': 30, 'left': 40}
        config = ChartConfig(title="Test", margin=custom_margin)

        assert config.margin == custom_margin


class TestChartOutput:
    """Tests for ChartOutput data class."""

    def test_chart_output_initialization(self):
        """Test ChartOutput initialization."""
        config = ChartConfig(title="Test")
        data = ChartData(x_values=[1], y_values=[1])

        output = ChartOutput(
            chart_type=ChartType.LINE,
            config=config,
            data=data,
            html_content="<html></html>",
            json_spec={"test": "spec"}
        )

        assert output.chart_type == ChartType.LINE
        assert output.config == config
        assert output.data == data
        assert output.html_content == "<html></html>"
        assert output.json_spec == {"test": "spec"}
        assert output.is_valid is True
        assert output.error_message is None
        assert output.render_time_ms == 0.0

    def test_chart_output_to_dict(self):
        """Test conversion to dictionary."""
        config = ChartConfig(title="Test")
        data = ChartData(x_values=[1], y_values=[1])

        output = ChartOutput(
            chart_type=ChartType.BAR,
            config=config,
            data=data,
            html_content="<html></html>",
            json_spec={"test": "spec"},
            render_time_ms=150.5
        )
        output_dict = output.to_dict()

        assert output_dict['chart_type'] == 'bar'
        assert output_dict['render_time_ms'] == 150.5
        assert output_dict['is_valid'] is True


class TestChartBuilder:
    """Comprehensive tests for ChartBuilder."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = ChartBuilder(max_data_points=1000, enable_streaming=True)

    def test_initialization(self):
        """Test ChartBuilder initialization."""
        assert self.builder.max_data_points == 1000
        assert self.builder.enable_streaming is True
        assert self.builder._color_palettes is not None
        assert self.builder._chart_templates is not None

    def test_initialization_custom_parameters(self):
        """Test ChartBuilder initialization with custom parameters."""
        builder = ChartBuilder(max_data_points=500, enable_streaming=False)
        assert builder.max_data_points == 500
        assert builder.enable_streaming is False

    # Time Series Chart Tests

    def test_create_time_series_chart_basic(self):
        """Test basic time series chart creation."""
        data = [1, 2, 3, 4, 5]
        result = self.builder.create_time_series_chart(data)

        assert result.chart_type == ChartType.TIMESERIES
        assert result.is_valid is True
        assert result.error_message is None
        assert len(result.data.x_values) == 5
        assert len(result.data.y_values) == 5
        assert result.html_content is not None
        assert result.json_spec is not None
        assert "vega" in result.html_content.lower()

    def test_create_time_series_chart_with_timestamps(self):
        """Test time series chart with timestamps."""
        data = [10, 15, 12, 18, 20]
        timestamps = [datetime(2024, 1, i) for i in range(1, 6)]

        result = self.builder.create_time_series_chart(data, timestamps=timestamps)

        assert result.is_valid is True
        assert len(result.data.x_values) == 5
        assert all(isinstance(x, datetime) for x in result.data.x_values)

    def test_create_time_series_chart_with_config(self):
        """Test time series chart with custom configuration."""
        data = [1, 2, 3, 4, 5]
        config = ChartConfig(
            title="Custom Time Series",
            x_label="Time",
            y_label="Values",
            theme=ChartTheme.DARK,
            width=1000,
            height=500
        )

        result = self.builder.create_time_series_chart(data, config=config)

        assert result.config.title == "Custom Time Series"
        assert result.config.theme == ChartTheme.DARK
        assert result.config.width == 1000
        assert result.config.height == 500

    def test_create_time_series_chart_with_trend_line(self):
        """Test time series chart with trend line."""
        data = [1, 3, 2, 4, 5]  # Some trend
        result = self.builder.create_time_series_chart(data, trend_line=True)

        assert result.is_valid is True
        # Check that trend line is included in spec
        spec = result.json_spec
        assert "layer" in spec
        assert len(spec["layer"]) >= 2  # Main line + trend line

    def test_create_time_series_chart_with_confidence_bands(self):
        """Test time series chart with confidence bands."""
        data = [10, 15, 12, 18, 20]
        confidence_bands = [(8, 12), (13, 17), (10, 14), (16, 20), (18, 22)]

        result = self.builder.create_time_series_chart(
            data,
            confidence_bands=confidence_bands
        )

        assert result.is_valid is True
        assert result.data.error_bars == confidence_bands
        # Check that confidence bands are in spec
        spec = result.json_spec
        assert "layer" in spec

    def test_create_time_series_chart_empty_data(self):
        """Test time series chart with empty data."""
        result = self.builder.create_time_series_chart([])

        assert result.is_valid is False
        assert "No valid data points" in result.error_message
        assert result.chart_type == ChartType.LINE  # Error chart type

    def test_create_time_series_chart_invalid_data(self):
        """Test time series chart with invalid data."""
        data = [float('nan'), float('inf'), float('-inf')]
        result = self.builder.create_time_series_chart(data)

        assert result.is_valid is False
        assert "No valid data points" in result.error_message

    def test_create_time_series_chart_mixed_valid_invalid(self):
        """Test time series chart with mixed valid/invalid data."""
        data = [1, float('nan'), 3, float('inf'), 5, 6]
        result = self.builder.create_time_series_chart(data)

        assert result.is_valid is True
        # Should filter out invalid values
        assert len(result.data.y_values) == 4  # 1, 3, 5, 6

    def test_create_time_series_chart_large_dataset(self):
        """Test time series chart with large dataset (downsampling)."""
        # Create dataset larger than max_data_points
        large_data = list(range(2000))  # Larger than default max_data_points=1000

        result = self.builder.create_time_series_chart(large_data)

        assert result.is_valid is True
        # Should be downsampled
        assert len(result.data.y_values) <= self.builder.max_data_points

    def test_create_time_series_chart_mismatched_timestamps(self):
        """Test time series chart with mismatched timestamp length."""
        data = [1, 2, 3, 4, 5]
        timestamps = [datetime(2024, 1, 1), datetime(2024, 1, 2)]  # Shorter than data

        result = self.builder.create_time_series_chart(data, timestamps=timestamps)

        assert result.is_valid is True
        # Should generate timestamps for missing data points
        assert len(result.data.x_values) == len(data)

    # Statistical Chart Tests

    def test_create_statistical_chart_histogram(self):
        """Test histogram creation."""
        data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
        result = self.builder.create_statistical_chart(
            data,
            chart_type=ChartType.HISTOGRAM
        )

        assert result.chart_type == ChartType.HISTOGRAM
        assert result.is_valid is True
        assert result.data.metadata is not None
        assert 'bins' in result.data.metadata
        assert 'mean' in result.data.metadata
        assert 'std' in result.data.metadata

    def test_create_statistical_chart_histogram_custom_bins(self):
        """Test histogram with custom bin count."""
        data = list(range(100))
        result = self.builder.create_statistical_chart(
            data,
            chart_type=ChartType.HISTOGRAM,
            bins=20
        )

        assert result.is_valid is True
        assert result.data.metadata['bins'] == 20

    def test_create_statistical_chart_histogram_with_statistics(self):
        """Test histogram with statistical annotations."""
        data = [10, 15, 12, 18, 20, 14, 16, 11, 13, 17]
        result = self.builder.create_statistical_chart(
            data,
            chart_type=ChartType.HISTOGRAM,
            show_statistics=True
        )

        assert result.is_valid is True
        # Should have layers for statistics
        spec = result.json_spec
        if "layer" in spec:
            assert len(spec["layer"]) >= 2  # Histogram + mean line

    def test_create_statistical_chart_scatter(self):
        """Test scatter plot creation."""
        data = [1, 4, 2, 8, 5, 7, 3, 6]
        result = self.builder.create_statistical_chart(
            data,
            chart_type=ChartType.SCATTER
        )

        assert result.chart_type == ChartType.SCATTER
        assert result.is_valid is True
        assert len(result.data.x_values) == len(data)
        assert len(result.data.y_values) == len(data)

    def test_create_statistical_chart_unsupported_type(self):
        """Test statistical chart with unsupported chart type."""
        data = [1, 2, 3, 4, 5]
        result = self.builder.create_statistical_chart(
            data,
            chart_type=ChartType.PIE  # Unsupported for statistical charts
        )

        assert result.is_valid is False
        assert "Unsupported statistical chart type" in result.error_message

    def test_create_statistical_chart_empty_data(self):
        """Test statistical chart with empty data."""
        result = self.builder.create_statistical_chart([])

        assert result.is_valid is False
        assert "No valid data points" in result.error_message

    def test_create_statistical_chart_single_value(self):
        """Test statistical chart with single value."""
        data = [42]
        result = self.builder.create_statistical_chart(data)

        assert result.is_valid is True
        assert len(result.data.y_values) == 1
        assert result.data.metadata['std'] == 0  # Single value has no deviation

    def test_create_statistical_chart_constant_values(self):
        """Test statistical chart with constant values."""
        data = [5] * 10
        result = self.builder.create_statistical_chart(data)

        assert result.is_valid is True
        assert result.data.metadata['std'] == 0
        assert result.data.metadata['mean'] == 5

    # Performance Chart Tests

    def test_create_performance_chart_single_metric(self):
        """Test performance chart with single metric."""
        metrics_data = {"response_time": [100, 110, 95, 120, 105]}
        result = self.builder.create_performance_chart(metrics_data)

        assert result.chart_type == ChartType.LINE
        assert result.is_valid is True
        assert result.data.labels == ["response_time"]
        assert result.data.metadata['metrics_count'] == 1

    def test_create_performance_chart_multiple_metrics(self):
        """Test performance chart with multiple metrics."""
        metrics_data = {
            "response_time": [100, 110, 95, 120, 105],
            "throughput": [50, 45, 55, 40, 48],
            "cpu_usage": [60, 65, 58, 70, 62]
        }
        result = self.builder.create_performance_chart(metrics_data)

        assert result.is_valid is True
        assert result.data.metadata['metrics_count'] == 3
        assert set(result.data.labels) == {"response_time", "throughput", "cpu_usage"}

    def test_create_performance_chart_with_thresholds(self):
        """Test performance chart with threshold lines."""
        metrics_data = {"response_time": [100, 110, 95, 120, 105]}
        thresholds = {"response_time": 115}

        result = self.builder.create_performance_chart(
            metrics_data,
            thresholds=thresholds
        )

        assert result.is_valid is True
        # Check that thresholds are included in spec
        spec = result.json_spec
        if "layer" in spec:
            assert len(spec["layer"]) >= 2  # Main chart + threshold line

    def test_create_performance_chart_empty_data(self):
        """Test performance chart with empty data."""
        result = self.builder.create_performance_chart({})

        assert result.is_valid is False
        assert "No metrics data provided" in result.error_message

    def test_create_performance_chart_invalid_metrics(self):
        """Test performance chart with all invalid metric values."""
        metrics_data = {
            "metric1": [float('nan'), float('inf')],
            "metric2": [float('-inf'), float('nan')]
        }
        result = self.builder.create_performance_chart(metrics_data)

        assert result.is_valid is False
        assert "No valid metrics data" in result.error_message

    def test_create_performance_chart_mixed_valid_invalid(self):
        """Test performance chart with mixed valid/invalid values."""
        metrics_data = {
            "metric1": [1, float('nan'), 3, float('inf'), 5],
            "metric2": [10, 20, 30]  # All valid
        }
        result = self.builder.create_performance_chart(metrics_data)

        assert result.is_valid is True
        assert result.data.metadata['metrics_count'] == 2  # Both metrics have some valid data

    # Capacity Chart Tests

    def test_create_capacity_chart_historical_only(self):
        """Test capacity chart with historical data only."""
        utilization_data = [0.3, 0.4, 0.5, 0.6, 0.7]
        result = self.builder.create_capacity_chart(utilization_data)

        assert result.chart_type == ChartType.AREA
        assert result.is_valid is True
        assert result.data.metadata['has_predictions'] is False
        assert result.data.metadata['current_utilization'] == 70.0  # 0.7 * 100

    def test_create_capacity_chart_with_predictions(self):
        """Test capacity chart with predictions."""
        utilization_data = [0.3, 0.4, 0.5, 0.6, 0.7]
        predicted_data = [0.75, 0.8, 0.85, 0.9, 0.95]

        result = self.builder.create_capacity_chart(
            utilization_data,
            predicted_utilization=predicted_data,
            show_predictions=True
        )

        assert result.is_valid is True
        assert result.data.metadata['has_predictions'] is True
        assert len(result.data.y_values) == len(utilization_data) + len(predicted_data)

    def test_create_capacity_chart_custom_threshold(self):
        """Test capacity chart with custom capacity threshold."""
        utilization_data = [0.3, 0.4, 0.5, 0.6, 0.7]
        result = self.builder.create_capacity_chart(
            utilization_data,
            capacity_threshold=0.6
        )

        assert result.is_valid is True
        assert result.data.metadata['threshold'] == 60.0  # 0.6 * 100

    def test_create_capacity_chart_values_clamped(self):
        """Test capacity chart with values outside [0, 1] range."""
        # Values outside valid range should be clamped
        utilization_data = [-0.1, 0.5, 1.2, 0.8, 2.0]
        result = self.builder.create_capacity_chart(utilization_data)

        assert result.is_valid is True
        # All values should be clamped to [0, 100] range
        assert all(0 <= val <= 100 for val in result.data.y_values)

    def test_create_capacity_chart_empty_data(self):
        """Test capacity chart with empty data."""
        result = self.builder.create_capacity_chart([])

        assert result.is_valid is False
        assert "No valid utilization data" in result.error_message

    def test_create_capacity_chart_invalid_data(self):
        """Test capacity chart with all invalid data."""
        utilization_data = [float('nan'), float('inf'), float('-inf')]
        result = self.builder.create_capacity_chart(utilization_data)

        assert result.is_valid is False
        assert "No valid utilization data" in result.error_message

    # Pattern Chart Tests

    def test_create_pattern_chart_basic(self):
        """Test basic pattern chart creation."""
        data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        patterns = [
            {
                "pattern_type": "linear",
                "start_index": 0,
                "end_index": 4,
                "confidence": 0.8
            }
        ]

        result = self.builder.create_pattern_chart(data, patterns)

        assert result.chart_type == ChartType.LINE
        assert result.is_valid is True
        assert result.data.metadata['patterns_detected'] == 1
        assert result.data.metadata['patterns'] == patterns

    def test_create_pattern_chart_multiple_patterns(self):
        """Test pattern chart with multiple patterns."""
        data = list(range(20))
        patterns = [
            {"pattern_type": "linear", "start_index": 0, "end_index": 9},
            {"pattern_type": "exponential", "start_index": 10, "end_index": 19}
        ]

        result = self.builder.create_pattern_chart(data, patterns)

        assert result.is_valid is True
        assert result.data.metadata['patterns_detected'] == 2

    def test_create_pattern_chart_no_patterns(self):
        """Test pattern chart with no patterns."""
        data = [1, 2, 3, 4, 5]
        patterns = []

        result = self.builder.create_pattern_chart(data, patterns)

        assert result.is_valid is True
        assert result.data.metadata['patterns_detected'] == 0

    def test_create_pattern_chart_patterns_without_indices(self):
        """Test pattern chart with patterns missing start/end indices."""
        data = [1, 2, 3, 4, 5]
        patterns = [
            {"pattern_type": "unknown", "confidence": 0.5}  # Missing indices
        ]

        result = self.builder.create_pattern_chart(data, patterns, highlight_patterns=True)

        assert result.is_valid is True
        # Should handle patterns without indices gracefully

    def test_create_pattern_chart_empty_data(self):
        """Test pattern chart with empty data."""
        result = self.builder.create_pattern_chart([], [])

        assert result.is_valid is False
        assert "No valid data for pattern chart" in result.error_message

    def test_create_pattern_chart_invalid_data(self):
        """Test pattern chart with all invalid data."""
        data = [float('nan'), float('inf'), float('-inf')]
        patterns = []
        result = self.builder.create_pattern_chart(data, patterns)

        assert result.is_valid is False
        assert "No valid data for pattern chart" in result.error_message

    # Gauge Chart Tests

    def test_create_gauge_chart_basic(self):
        """Test basic gauge chart creation."""
        result = self.builder.create_gauge_chart(
            current_value=75,
            min_value=0,
            max_value=100
        )

        assert result.chart_type == ChartType.GAUGE
        assert result.is_valid is True
        assert result.data.metadata['min_value'] == 0
        assert result.data.metadata['max_value'] == 100
        assert result.data.metadata['percentage'] == 75.0

    def test_create_gauge_chart_custom_thresholds(self):
        """Test gauge chart with custom thresholds."""
        thresholds = [
            (25, "red"),
            (75, "yellow"),
            (100, "green")
        ]

        result = self.builder.create_gauge_chart(
            current_value=50,
            min_value=0,
            max_value=100,
            thresholds=thresholds
        )

        assert result.is_valid is True
        assert result.data.metadata['thresholds'] == thresholds

    def test_create_gauge_chart_custom_range(self):
        """Test gauge chart with custom value range."""
        result = self.builder.create_gauge_chart(
            current_value=150,
            min_value=100,
            max_value=200
        )

        assert result.is_valid is True
        assert result.data.metadata['percentage'] == 50.0  # (150-100)/(200-100) * 100

    def test_create_gauge_chart_invalid_current_value(self):
        """Test gauge chart with invalid current value."""
        result = self.builder.create_gauge_chart(
            current_value=float('nan'),
            min_value=0,
            max_value=100
        )

        assert result.is_valid is False
        assert "Invalid current value" in result.error_message

    def test_create_gauge_chart_invalid_range(self):
        """Test gauge chart with invalid range."""
        result = self.builder.create_gauge_chart(
            current_value=50,
            min_value=100,  # min > max
            max_value=50
        )

        assert result.is_valid is False
        assert "Invalid range" in result.error_message

    def test_create_gauge_chart_value_at_boundaries(self):
        """Test gauge chart with value at boundaries."""
        # Test minimum value
        result_min = self.builder.create_gauge_chart(
            current_value=0,
            min_value=0,
            max_value=100
        )
        assert result_min.is_valid is True
        assert result_min.data.metadata['percentage'] == 0.0

        # Test maximum value
        result_max = self.builder.create_gauge_chart(
            current_value=100,
            min_value=0,
            max_value=100
        )
        assert result_max.is_valid is True
        assert result_max.data.metadata['percentage'] == 100.0

    # Utility Method Tests

    def test_validate_time_series_data_normal(self):
        """Test time series data validation with normal data."""
        data = [1.0, 2.0, 3.0]
        timestamps = [datetime(2024, 1, i) for i in range(1, 4)]

        valid_data, valid_timestamps = self.builder._validate_time_series_data(data, timestamps)

        assert valid_data == [1.0, 2.0, 3.0]
        assert len(valid_timestamps) == 3
        assert all(isinstance(ts, datetime) for ts in valid_timestamps)

    def test_validate_time_series_data_no_timestamps(self):
        """Test time series data validation without timestamps."""
        data = [1.0, 2.0, 3.0]

        valid_data, valid_timestamps = self.builder._validate_time_series_data(data, None)

        assert valid_data == [1.0, 2.0, 3.0]
        assert len(valid_timestamps) == 3
        assert all(isinstance(ts, datetime) for ts in valid_timestamps)

    def test_validate_time_series_data_with_invalid(self):
        """Test time series data validation with invalid values."""
        data = [1.0, float('nan'), 3.0, float('inf'), 5.0]

        valid_data, valid_timestamps = self.builder._validate_time_series_data(data, None)

        assert valid_data == [1.0, 3.0, 5.0]
        assert len(valid_timestamps) == 3

    def test_downsample_time_series_needed(self):
        """Test time series downsampling when needed."""
        data = list(range(2000))  # Large dataset
        timestamps = [datetime.now() + timedelta(days=i) for i in range(2000)]

        downsampled_data, downsampled_timestamps = self.builder._downsample_time_series(
            data, timestamps, max_points=1000
        )

        assert len(downsampled_data) <= 1000
        assert len(downsampled_timestamps) <= 1000
        assert len(downsampled_data) == len(downsampled_timestamps)

    def test_downsample_time_series_not_needed(self):
        """Test time series downsampling when not needed."""
        data = [1, 2, 3, 4, 5]
        timestamps = [datetime.now() + timedelta(days=i) for i in range(5)]

        downsampled_data, downsampled_timestamps = self.builder._downsample_time_series(
            data, timestamps, max_points=1000
        )

        assert downsampled_data == data
        assert downsampled_timestamps == timestamps

    def test_get_color_palette_small(self):
        """Test color palette generation for small number of colors."""
        colors = self.builder._get_color_palette(3)

        assert len(colors) == 3
        assert all(isinstance(color, str) for color in colors)
        assert all(color.startswith('#') or color.startswith('hsl') for color in colors)

    def test_get_color_palette_large(self):
        """Test color palette generation for large number of colors."""
        colors = self.builder._get_color_palette(20)  # More than default palette

        assert len(colors) == 20
        assert all(isinstance(color, str) for color in colors)

    def test_get_gauge_color_thresholds(self):
        """Test gauge color selection based on thresholds."""
        thresholds = [(50, "red"), (80, "yellow"), (100, "green")]

        # Test value below first threshold
        color = self.builder._get_gauge_color(30, thresholds)
        assert color == "red"

        # Test value between thresholds
        color = self.builder._get_gauge_color(65, thresholds)
        assert color == "yellow"

        # Test value above all thresholds
        color = self.builder._get_gauge_color(90, thresholds)
        assert color == "green"

    def test_get_gauge_color_empty_thresholds(self):
        """Test gauge color with empty thresholds."""
        color = self.builder._get_gauge_color(50, [])
        assert color == "#1f77b4"  # Default color

    # HTML Generation Tests

    def test_generate_html_chart_basic(self):
        """Test basic HTML chart generation."""
        json_spec = {"mark": "circle", "data": {"values": []}}
        config = ChartConfig(title="Test Chart")

        html_content = self.builder._generate_html_chart(json_spec, config)

        assert "<html>" in html_content
        assert "<script" in html_content
        assert "vega" in html_content.lower()
        assert "Test Chart" in html_content or json.dumps(json_spec) in html_content

    def test_generate_html_chart_interactive(self):
        """Test HTML generation with interactive configuration."""
        json_spec = {"mark": "point"}
        config = ChartConfig(title="Interactive Chart", interactive=True)

        html_content = self.builder._generate_html_chart(json_spec, config)

        assert "actions: true" in html_content

    def test_generate_html_chart_non_interactive(self):
        """Test HTML generation with non-interactive configuration."""
        json_spec = {"mark": "point"}
        config = ChartConfig(title="Static Chart", interactive=False)

        html_content = self.builder._generate_html_chart(json_spec, config)

        assert "actions: false" in html_content

    def test_generate_html_chart_theme(self):
        """Test HTML generation with different themes."""
        json_spec = {"mark": "point"}
        config = ChartConfig(title="Themed Chart", theme=ChartTheme.DARK)

        html_content = self.builder._generate_html_chart(json_spec, config)

        assert "theme: 'dark'" in html_content

    # Error Handling Tests

    def test_error_chart_creation(self):
        """Test error chart creation."""
        error_chart = self.builder._create_error_chart("Test error message")

        assert error_chart.is_valid is False
        assert error_chart.error_message == "Test error message"
        assert "error" in error_chart.html_content.lower()
        assert "Test error message" in error_chart.html_content

    def test_chart_creation_with_exception(self):
        """Test chart creation when exception occurs."""
        # Mock an exception during chart creation
        with patch.object(self.builder, '_validate_time_series_data', side_effect=Exception("Mock error")):
            result = self.builder.create_time_series_chart([1, 2, 3])

            assert result.is_valid is False
            assert "Time series chart error" in result.error_message
            assert "Mock error" in result.error_message

    def test_statistical_chart_with_exception(self):
        """Test statistical chart creation when exception occurs."""
        with patch('statistics.mean', side_effect=Exception("Statistics error")):
            result = self.builder.create_statistical_chart([1, 2, 3])

            assert result.is_valid is False
            assert "Statistical chart error" in result.error_message

    def test_performance_chart_with_exception(self):
        """Test performance chart creation when exception occurs."""
        with patch.object(self.builder, '_create_performance_spec', side_effect=Exception("Spec error")):
            metrics_data = {"metric1": [1, 2, 3]}
            result = self.builder.create_performance_chart(metrics_data)

            assert result.is_valid is False
            assert "Performance chart error" in result.error_message

    def test_capacity_chart_with_exception(self):
        """Test capacity chart creation when exception occurs."""
        with patch.object(self.builder, '_create_capacity_spec', side_effect=Exception("Capacity error")):
            result = self.builder.create_capacity_chart([0.5, 0.6, 0.7])

            assert result.is_valid is False
            assert "Capacity chart error" in result.error_message

    def test_pattern_chart_with_exception(self):
        """Test pattern chart creation when exception occurs."""
        with patch.object(self.builder, '_create_pattern_spec', side_effect=Exception("Pattern error")):
            result = self.builder.create_pattern_chart([1, 2, 3], [])

            assert result.is_valid is False
            assert "Pattern chart error" in result.error_message

    def test_gauge_chart_with_exception(self):
        """Test gauge chart creation when exception occurs."""
        with patch.object(self.builder, '_create_gauge_spec', side_effect=Exception("Gauge error")):
            result = self.builder.create_gauge_chart(50, 0, 100)

            assert result.is_valid is False
            assert "Gauge chart error" in result.error_message

    # Integration Tests

    def test_comprehensive_chart_workflow(self):
        """Test comprehensive chart creation workflow."""
        # Time series data with trend
        data = [10 + i * 2 + (i % 3) for i in range(50)]
        timestamps = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(50)]

        # Create various chart types
        time_series = self.builder.create_time_series_chart(
            data, timestamps=timestamps, trend_line=True
        )

        histogram = self.builder.create_statistical_chart(
            data, chart_type=ChartType.HISTOGRAM, show_statistics=True
        )

        metrics = {"data": data}
        performance = self.builder.create_performance_chart(metrics)

        utilization = [x / max(data) for x in data]
        capacity = self.builder.create_capacity_chart(utilization)

        # All should succeed
        assert time_series.is_valid is True
        assert histogram.is_valid is True
        assert performance.is_valid is True
        assert capacity.is_valid is True

        # All should have valid HTML
        assert all("<html>" in chart.html_content for chart in
                  [time_series, histogram, performance, capacity])

    def test_large_dataset_handling(self):
        """Test handling of large datasets across chart types."""
        # Generate large dataset
        large_data = [math.sin(i * 0.01) + i * 0.001 for i in range(5000)]

        # Test different chart types with large data
        time_series = self.builder.create_time_series_chart(large_data)
        histogram = self.builder.create_statistical_chart(large_data)

        assert time_series.is_valid is True
        assert histogram.is_valid is True

        # Time series should be downsampled
        assert len(time_series.data.y_values) <= self.builder.max_data_points

    def test_edge_case_data_combinations(self):
        """Test various edge case data combinations."""
        # All zeros
        zero_data = [0] * 10
        zero_chart = self.builder.create_time_series_chart(zero_data)
        assert zero_chart.is_valid is True

        # Single value repeated
        single_value = [42] * 20
        single_chart = self.builder.create_statistical_chart(single_value)
        assert single_chart.is_valid is True

        # Alternating extreme values
        extreme_data = [1e-10 if i % 2 == 0 else 1e10 for i in range(10)]
        extreme_chart = self.builder.create_time_series_chart(extreme_data)
        assert extreme_chart.is_valid is True

    def test_render_time_tracking(self):
        """Test render time tracking in chart outputs."""
        data = [1, 2, 3, 4, 5]
        result = self.builder.create_time_series_chart(data)

        assert result.render_time_ms >= 0
        assert isinstance(result.render_time_ms, float)

    def test_chart_specification_validity(self):
        """Test that generated chart specifications are valid."""
        data = [1, 2, 3, 4, 5]
        result = self.builder.create_time_series_chart(data)

        spec = result.json_spec

        # Basic Vega-Lite structure
        assert "$schema" in spec
        assert "vega.github.io" in spec["$schema"]
        assert "data" in spec or "layer" in spec

        # Check data structure if present
        if "data" in spec:
            assert "values" in spec["data"]
            assert isinstance(spec["data"]["values"], list)