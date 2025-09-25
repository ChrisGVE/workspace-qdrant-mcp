#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Data Analytics Framework

Tests all components with edge cases, error conditions, and boundary scenarios.
Ensures 90%+ test coverage with meaningful assertions.

Created: 2025-09-25T18:01:09+02:00
"""

import asyncio
import math
import pytest
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Import the analytics framework
from data_analytics_framework import (
    AdvancedAnalyticsEngine,
    IntelligenceFramework,
    PredictiveAnalyticsSystem,
    DataVisualizationFramework,
    AnomalyDetectionSystem,
    DataAnalyticsFramework,
    AnalyticsMetric,
    PatternInsight,
    PredictionResult,
    AnomalyAlert,
    AnalyticsLevel,
    AlertSeverity
)


class TestAdvancedAnalyticsEngine:
    """Test suite for AdvancedAnalyticsEngine"""

    def setup_method(self):
        """Setup test fixtures"""
        self.engine = AdvancedAnalyticsEngine()

    def test_calculate_descriptive_statistics_normal_data(self):
        """Test descriptive statistics with normal data"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = self.engine.calculate_descriptive_statistics(data, "test_metric")

        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert result["median"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0
        assert result["range"] == 4.0
        assert "std_dev" in result
        assert "variance" in result

    def test_calculate_descriptive_statistics_empty_data(self):
        """Test descriptive statistics with empty data - edge case"""
        data = []
        result = self.engine.calculate_descriptive_statistics(data, "empty_metric")

        assert "error" in result
        assert result["error"] == "empty_dataset"
        assert result["count"] == 0

    def test_calculate_descriptive_statistics_nan_values(self):
        """Test descriptive statistics with NaN values - edge case"""
        data = [1.0, float('nan'), 3.0, float('nan'), 5.0]
        result = self.engine.calculate_descriptive_statistics(data, "nan_metric")

        assert result["count"] == 3
        assert result["mean"] == 3.0
        assert result["min"] == 1.0
        assert result["max"] == 5.0

    def test_calculate_descriptive_statistics_single_value(self):
        """Test descriptive statistics with single value - edge case"""
        data = [42.0]
        result = self.engine.calculate_descriptive_statistics(data, "single_metric")

        assert result["count"] == 1
        assert result["mean"] == 42.0
        assert result["std_dev"] == 0.0
        assert result["variance"] == 0.0

    def test_calculate_descriptive_statistics_constant_values(self):
        """Test descriptive statistics with constant values - edge case"""
        data = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = self.engine.calculate_descriptive_statistics(data, "constant_metric")

        assert result["count"] == 5
        assert result["mean"] == 5.0
        assert result["std_dev"] == 0.0
        assert result["coefficient_of_variation"] == 0.0

    def test_calculate_descriptive_statistics_zero_mean(self):
        """Test descriptive statistics with zero mean - edge case"""
        data = [-2.0, -1.0, 0.0, 1.0, 2.0]
        result = self.engine.calculate_descriptive_statistics(data, "zero_mean_metric")

        assert result["count"] == 5
        assert result["mean"] == 0.0
        assert result["coefficient_of_variation"] == float('inf')

    def test_calculate_descriptive_statistics_large_numbers(self):
        """Test descriptive statistics with large numbers"""
        data = [1e10, 2e10, 3e10, 4e10, 5e10]
        result = self.engine.calculate_descriptive_statistics(data, "large_metric")

        assert result["count"] == 5
        assert result["mean"] == 3e10
        assert result["min"] == 1e10
        assert result["max"] == 5e10

    def test_skewness_calculation_edge_cases(self):
        """Test skewness calculation with edge cases"""
        # Insufficient data
        assert self.engine._calculate_skewness([1.0, 2.0]) == 0.0

        # Zero standard deviation
        assert self.engine._calculate_skewness([5.0, 5.0, 5.0, 5.0]) == 0.0

        # Normal distribution (should be close to 0)
        normal_data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        skewness = self.engine._calculate_skewness(normal_data)
        assert abs(skewness) < 1.0  # Should be relatively small

    def test_kurtosis_calculation_edge_cases(self):
        """Test kurtosis calculation with edge cases"""
        # Insufficient data
        assert self.engine._calculate_kurtosis([1.0, 2.0, 3.0]) == 0.0

        # Zero standard deviation
        assert self.engine._calculate_kurtosis([5.0, 5.0, 5.0, 5.0, 5.0]) == 0.0

        # Normal-like distribution
        normal_data = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 3.0]
        kurtosis = self.engine._calculate_kurtosis(normal_data)
        assert isinstance(kurtosis, float)

    def test_perform_correlation_analysis_normal_data(self):
        """Test correlation analysis with normal data"""
        datasets = {
            "metric1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metric2": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfect positive correlation
            "metric3": [5.0, 4.0, 3.0, 2.0, 1.0]   # Perfect negative correlation with metric1
        }
        result = self.engine.perform_correlation_analysis(datasets)

        assert "metric1" in result
        assert "metric2" in result
        assert "metric3" in result

        # Perfect positive correlation
        assert abs(result["metric1"]["metric2"] - 1.0) < 0.001

        # Perfect negative correlation
        assert abs(result["metric1"]["metric3"] - (-1.0)) < 0.001

        # Self-correlation should be 1.0
        assert result["metric1"]["metric1"] == 1.0

    def test_perform_correlation_analysis_edge_cases(self):
        """Test correlation analysis with edge cases"""
        # Insufficient datasets
        datasets = {"metric1": [1.0, 2.0, 3.0]}
        result = self.engine.perform_correlation_analysis(datasets)
        assert "error" in result

        # Different data lengths
        datasets = {
            "metric1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "metric2": [1.0, 2.0]  # Shorter dataset
        }
        result = self.engine.perform_correlation_analysis(datasets)
        assert "metric1" in result
        assert result["metric1"]["metric2"] == 1.0  # Should handle length difference

        # NaN values
        datasets = {
            "metric1": [1.0, 2.0, float('nan'), 4.0, 5.0],
            "metric2": [2.0, 4.0, 6.0, float('nan'), 10.0]
        }
        result = self.engine.perform_correlation_analysis(datasets)
        assert "metric1" in result
        assert isinstance(result["metric1"]["metric2"], float)

    def test_perform_correlation_analysis_insufficient_valid_data(self):
        """Test correlation analysis with insufficient valid data"""
        datasets = {
            "metric1": [1.0, float('nan')],
            "metric2": [float('nan'), 2.0]
        }
        result = self.engine.perform_correlation_analysis(datasets)

        assert result["metric1"]["metric2"] == 0.0

    def test_perform_correlation_analysis_constant_data(self):
        """Test correlation analysis with constant data - edge case"""
        datasets = {
            "metric1": [5.0, 5.0, 5.0, 5.0, 5.0],  # Constant data
            "metric2": [1.0, 2.0, 3.0, 4.0, 5.0]   # Variable data
        }
        result = self.engine.perform_correlation_analysis(datasets)

        # Correlation with constant data should be 0 or NaN (handled as 0)
        assert result["metric1"]["metric2"] == 0.0


class TestIntelligenceFramework:
    """Test suite for IntelligenceFramework"""

    def setup_method(self):
        """Setup test fixtures"""
        self.framework = IntelligenceFramework()

    def test_detect_patterns_empty_data(self):
        """Test pattern detection with empty data - edge case"""
        patterns = self.framework.detect_patterns([])
        assert patterns == []

    def test_detect_patterns_normal_data(self):
        """Test pattern detection with normal data"""
        data = [
            {"cpu": 45.0, "memory": 60.0, "timestamp": "2024-01-01T00:00:00"},
            {"cpu": 50.0, "memory": 65.0, "timestamp": "2024-01-01T01:00:00"},
            {"cpu": 55.0, "memory": 70.0, "timestamp": "2024-01-01T02:00:00"}
        ]
        patterns = self.framework.detect_patterns(data, ["trend", "clustering"])

        assert isinstance(patterns, list)
        for pattern in patterns:
            assert isinstance(pattern, PatternInsight)
            assert pattern.pattern_type in ["trend", "clustering"]
            assert 0 <= pattern.confidence <= 1.0

    @patch('pandas.DataFrame')
    def test_detect_patterns_dataframe_error(self, mock_df):
        """Test pattern detection with DataFrame creation error"""
        mock_df.side_effect = Exception("DataFrame creation failed")

        data = [{"value": 1}]
        patterns = self.framework.detect_patterns(data)
        assert patterns == []

    def test_analyze_pattern_type_unknown_type(self):
        """Test pattern analysis with unknown pattern type"""
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = self.framework._analyze_pattern_type(df, "unknown_pattern")
        assert result is None

    def test_detect_trend_patterns_no_numeric_columns(self):
        """Test trend detection with no numeric columns"""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = self.framework._detect_trend_patterns(df)
        assert result is None

    def test_detect_trend_patterns_insufficient_data(self):
        """Test trend detection with insufficient data"""
        df = pd.DataFrame({"value": [1.0, 2.0]})  # Less than 3 points
        result = self.framework._detect_trend_patterns(df)
        assert result is None

    def test_detect_trend_patterns_no_significant_trends(self):
        """Test trend detection with no significant trends"""
        df = pd.DataFrame({"value": [1.0, 1.001, 1.002, 1.003]})  # Very small changes
        result = self.framework._detect_trend_patterns(df)
        assert result is None

    def test_detect_trend_patterns_strong_trend(self):
        """Test trend detection with strong trend"""
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 5.0]})  # Strong increasing trend
        result = self.framework._detect_trend_patterns(df)

        assert result is not None
        assert result.pattern_type == "trend"
        assert result.confidence > 0
        assert len(result.data_points) > 0
        assert result.data_points[0]["direction"] == "increasing"

    def test_detect_seasonal_patterns_no_timestamp(self):
        """Test seasonal detection without timestamp column"""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        result = self.framework._detect_seasonal_patterns(df)
        assert result is None

    def test_detect_seasonal_patterns_insufficient_data(self):
        """Test seasonal detection with insufficient data"""
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq='H'),
            "value": [1, 2, 3, 4, 5]
        })
        result = self.framework._detect_seasonal_patterns(df)
        assert result is None  # Less than 24 data points

    def test_detect_seasonal_patterns_invalid_timestamp(self):
        """Test seasonal detection with invalid timestamp"""
        df = pd.DataFrame({
            "timestamp": ["invalid_date", "2024-01-01", "2024-01-02"],
            "value": [1, 2, 3]
        })

        with patch('pandas.to_datetime', side_effect=Exception("Invalid timestamp")):
            result = self.framework._detect_seasonal_patterns(df)
            assert result is None

    def test_detect_clustering_patterns_insufficient_data(self):
        """Test clustering with insufficient data"""
        df = pd.DataFrame({"value": [1.0, 2.0]})  # Less than 3 points
        result = self.framework._detect_clustering_patterns(df)
        assert result is None

    def test_detect_clustering_patterns_single_column(self):
        """Test clustering with single numeric column"""
        df = pd.DataFrame({"value": [1, 2, 3], "text": ["a", "b", "c"]})
        result = self.framework._detect_clustering_patterns(df)
        assert result is None  # Need at least 2 numeric columns

    def test_detect_clustering_patterns_successful(self):
        """Test successful clustering pattern detection"""
        # Create data with clear clusters
        df = pd.DataFrame({
            "x": [1, 1, 1, 10, 10, 10],
            "y": [1, 1, 1, 10, 10, 10]
        })
        result = self.framework._detect_clustering_patterns(df)

        assert result is not None
        assert result.pattern_type == "clustering"
        assert len(result.data_points) >= 2  # Should find at least 2 clusters

    def test_detect_outlier_patterns_no_numeric_data(self):
        """Test outlier detection with no numeric data"""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = self.framework._detect_outlier_patterns(df)
        assert result is None

    def test_detect_outlier_patterns_insufficient_data(self):
        """Test outlier detection with insufficient data"""
        df = pd.DataFrame({"value": [1.0, 2.0]})  # Less than 3 points
        result = self.framework._detect_outlier_patterns(df)
        assert result is None

    def test_detect_outlier_patterns_constant_data(self):
        """Test outlier detection with constant data"""
        df = pd.DataFrame({"value": [5.0, 5.0, 5.0, 5.0, 5.0]})
        result = self.framework._detect_outlier_patterns(df)
        assert result is None  # IQR = 0, no outliers possible

    def test_detect_outlier_patterns_with_outliers(self):
        """Test outlier detection with actual outliers"""
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 100]})  # 100 is an outlier
        result = self.framework._detect_outlier_patterns(df)

        assert result is not None
        assert result.pattern_type == "outliers"
        assert len(result.data_points) > 0
        assert result.data_points[0]["outlier_count"] > 0


class TestPredictiveAnalyticsSystem:
    """Test suite for PredictiveAnalyticsSystem"""

    def setup_method(self):
        """Setup test fixtures"""
        self.system = PredictiveAnalyticsSystem()

    def test_forecast_time_series_insufficient_data(self):
        """Test forecasting with insufficient data - edge case"""
        data = [(datetime.now(), 1.0), (datetime.now(), 2.0)]  # Less than 3 points
        result = self.system.forecast_time_series(data, timedelta(hours=1))
        assert result is None

    def test_forecast_time_series_nan_values(self):
        """Test forecasting with NaN values"""
        now = datetime.now()
        data = [
            (now, 1.0),
            (now + timedelta(hours=1), float('nan')),
            (now + timedelta(hours=2), 3.0),
            (now + timedelta(hours=3), float('nan')),
            (now + timedelta(hours=4), 5.0)
        ]
        result = self.system.forecast_time_series(data, timedelta(hours=1))

        assert result is not None
        assert isinstance(result.predicted_value, float)
        assert not math.isnan(result.predicted_value)
        assert result.metadata["data_points"] == 3  # Only valid points

    def test_forecast_time_series_all_nan_values(self):
        """Test forecasting with all NaN values - edge case"""
        now = datetime.now()
        data = [
            (now, float('nan')),
            (now + timedelta(hours=1), float('nan')),
            (now + timedelta(hours=2), float('nan'))
        ]
        result = self.system.forecast_time_series(data, timedelta(hours=1))
        assert result is None

    def test_forecast_time_series_successful(self):
        """Test successful time series forecasting"""
        now = datetime.now()
        data = [
            (now, 1.0),
            (now + timedelta(hours=1), 2.0),
            (now + timedelta(hours=2), 3.0),
            (now + timedelta(hours=3), 4.0),
            (now + timedelta(hours=4), 5.0)
        ]
        result = self.system.forecast_time_series(data, timedelta(hours=1))

        assert result is not None
        assert isinstance(result.predicted_value, float)
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2
        assert result.confidence_interval[0] <= result.confidence_interval[1]
        assert 0 <= result.model_accuracy <= 1.0
        assert result.prediction_horizon == timedelta(hours=1)

    def test_forecast_time_series_constant_values(self):
        """Test forecasting with constant values"""
        now = datetime.now()
        data = [
            (now, 5.0),
            (now + timedelta(hours=1), 5.0),
            (now + timedelta(hours=2), 5.0),
            (now + timedelta(hours=3), 5.0)
        ]
        result = self.system.forecast_time_series(data, timedelta(hours=1))

        assert result is not None
        assert abs(result.predicted_value - 5.0) < 0.1  # Should predict around 5.0
        assert result.model_accuracy >= 0  # Should not be negative

    def test_predict_resource_usage_empty_data(self):
        """Test resource prediction with empty data"""
        result = self.system.predict_resource_usage({})
        assert result == {}

    def test_predict_resource_usage_insufficient_data(self):
        """Test resource prediction with insufficient data"""
        historical_data = {
            "cpu": [1.0, 2.0],  # Less than 3 points
            "memory": [50.0, 60.0, 70.0]  # Sufficient points
        }
        result = self.system.predict_resource_usage(historical_data)

        assert "cpu" not in result  # Should be excluded
        assert "memory" in result  # Should be included

    def test_predict_resource_usage_nan_values(self):
        """Test resource prediction with NaN values"""
        historical_data = {
            "cpu": [1.0, float('nan'), 3.0, 4.0, 5.0],
            "memory": [float('nan'), float('nan'), float('nan')]  # All NaN
        }
        result = self.system.predict_resource_usage(historical_data)

        assert "cpu" in result  # Should handle NaN filtering
        assert "memory" not in result  # All NaN, should be excluded

    def test_predict_resource_usage_negative_values(self):
        """Test resource prediction with negative values"""
        historical_data = {
            "cpu": [1.0, -2.0, 3.0, -4.0, 5.0]  # Mixed positive/negative
        }
        result = self.system.predict_resource_usage(historical_data)

        assert "cpu" in result
        # Confidence interval lower bound should be >= 0
        assert result["cpu"].confidence_interval[0] >= 0

    def test_predict_resource_usage_successful(self):
        """Test successful resource usage prediction"""
        historical_data = {
            "cpu": [10.0, 20.0, 30.0, 40.0, 50.0],
            "memory": [100.0, 200.0, 300.0, 400.0, 500.0]
        }
        result = self.system.predict_resource_usage(historical_data)

        assert len(result) == 2
        for resource_name, prediction in result.items():
            assert isinstance(prediction.predicted_value, float)
            assert isinstance(prediction.confidence_interval, tuple)
            assert prediction.confidence_interval[0] >= 0  # No negative usage
            assert 0 <= prediction.model_accuracy <= 1.0

    def test_predict_resource_usage_constant_values(self):
        """Test resource prediction with constant values"""
        historical_data = {
            "cpu": [50.0, 50.0, 50.0, 50.0, 50.0]
        }
        result = self.system.predict_resource_usage(historical_data)

        assert "cpu" in result
        prediction = result["cpu"]
        assert abs(prediction.predicted_value - 50.0) < 0.1
        assert prediction.confidence_interval[0] <= prediction.predicted_value
        assert prediction.predicted_value <= prediction.confidence_interval[1]


class TestDataVisualizationFramework:
    """Test suite for DataVisualizationFramework"""

    def setup_method(self):
        """Setup test fixtures"""
        self.framework = DataVisualizationFramework()

    def create_sample_metrics(self):
        """Create sample metrics for testing"""
        return [
            AnalyticsMetric("cpu", 45.0, datetime.now(), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("memory", 78.5, datetime.now(), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("cpu", float('nan'), datetime.now(), {}, AnalyticsLevel.BASIC)
        ]

    def create_sample_patterns(self):
        """Create sample patterns for testing"""
        return [
            PatternInsight("trend", 0.8, "Increasing trend", [{"slope": 1.2}], ["Monitor trend"]),
            PatternInsight("outliers", 0.6, "Outliers detected", [{"count": 3}], ["Investigate"])
        ]

    def create_sample_predictions(self):
        """Create sample predictions for testing"""
        return {
            "cpu": PredictionResult(
                predicted_value=55.0,
                confidence_interval=(50.0, 60.0),
                model_accuracy=0.85,
                prediction_horizon=timedelta(hours=1),
                metadata={"method": "linear"}
            )
        }

    def test_create_analytics_dashboard_successful(self):
        """Test successful dashboard creation"""
        metrics = self.create_sample_metrics()
        patterns = self.create_sample_patterns()
        predictions = self.create_sample_predictions()

        result = self.framework.create_analytics_dashboard(metrics, patterns, predictions)

        assert "metrics_overview" in result
        assert "pattern_insights" in result
        assert "prediction_charts" in result
        assert "correlation_heatmap" in result
        assert "timestamp" in result
        assert "metadata" in result

        assert result["metadata"]["total_metrics"] == 3
        assert result["metadata"]["total_patterns"] == 2
        assert result["metadata"]["total_predictions"] == 1

    def test_create_analytics_dashboard_empty_data(self):
        """Test dashboard creation with empty data"""
        result = self.framework.create_analytics_dashboard([], [], {})

        assert "metrics_overview" in result
        assert "pattern_insights" in result
        assert "prediction_charts" in result
        assert "correlation_heatmap" in result
        assert result["metadata"]["total_metrics"] == 0

    @patch.object(DataVisualizationFramework, '_create_metrics_overview')
    def test_create_analytics_dashboard_error_handling(self, mock_overview):
        """Test dashboard creation with error handling"""
        mock_overview.side_effect = Exception("Overview creation failed")

        metrics = self.create_sample_metrics()
        result = self.framework.create_analytics_dashboard(metrics, [], {})

        assert "error" in result
        assert "timestamp" in result

    def test_create_metrics_overview_empty_metrics(self):
        """Test metrics overview with empty metrics"""
        result = self.framework._create_metrics_overview([])

        assert "error" in result
        assert result["error"] == "no_metrics"
        assert result["charts"] == []

    def test_create_metrics_overview_all_nan_values(self):
        """Test metrics overview with all NaN values"""
        metrics = [
            AnalyticsMetric("cpu", float('nan'), datetime.now(), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("cpu", float('nan'), datetime.now(), {}, AnalyticsLevel.BASIC)
        ]
        result = self.framework._create_metrics_overview(metrics)

        assert "charts" in result
        assert len(result["charts"]) == 0  # No valid charts created

    def test_create_metrics_overview_successful(self):
        """Test successful metrics overview creation"""
        metrics = [
            AnalyticsMetric("cpu", 45.0, datetime.now(), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("cpu", 50.0, datetime.now() + timedelta(minutes=1), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("memory", 70.0, datetime.now(), {}, AnalyticsLevel.BASIC)
        ]
        result = self.framework._create_metrics_overview(metrics)

        assert "charts" in result
        assert len(result["charts"]) == 2  # cpu and memory
        assert result["total_metrics"] == 3

        cpu_chart = next(c for c in result["charts"] if c["name"] == "cpu")
        assert cpu_chart["stats"]["trend"] == "increasing"
        assert len(cpu_chart["data"]["y"]) == 2

    def test_create_pattern_visualization_empty_patterns(self):
        """Test pattern visualization with empty patterns"""
        result = self.framework._create_pattern_visualization([])

        assert "error" in result
        assert result["error"] == "no_patterns"
        assert result["insights"] == []

    def test_create_pattern_visualization_successful(self):
        """Test successful pattern visualization"""
        patterns = self.create_sample_patterns()
        result = self.framework._create_pattern_visualization(patterns)

        assert "insights" in result
        assert len(result["insights"]) == 2
        assert result["total_patterns"] == 2

        trend_insight = next(i for i in result["insights"] if i["type"] == "trend")
        assert trend_insight["confidence"] == 0.8
        assert len(trend_insight["data_points"]) <= 10  # Limited data points
        assert len(trend_insight["recommendations"]) <= 5  # Limited recommendations

    def test_create_prediction_visualization_empty_predictions(self):
        """Test prediction visualization with empty predictions"""
        result = self.framework._create_prediction_visualization({})

        assert "error" in result
        assert result["error"] == "no_predictions"
        assert result["forecasts"] == []

    def test_create_prediction_visualization_successful(self):
        """Test successful prediction visualization"""
        predictions = self.create_sample_predictions()
        result = self.framework._create_prediction_visualization(predictions)

        assert "forecasts" in result
        assert len(result["forecasts"]) == 1
        assert result["total_predictions"] == 1

        forecast = result["forecasts"][0]
        assert forecast["metric_name"] == "cpu"
        assert forecast["predicted_value"] == 55.0
        assert forecast["accuracy"] == 0.85

    def test_create_correlation_heatmap_insufficient_metrics(self):
        """Test correlation heatmap with insufficient metrics"""
        metrics = [AnalyticsMetric("cpu", 45.0, datetime.now(), {}, AnalyticsLevel.BASIC)]
        result = self.framework._create_correlation_heatmap(metrics)

        assert "error" in result
        assert result["error"] == "insufficient_metrics"

    def test_create_correlation_heatmap_single_metric_type(self):
        """Test correlation heatmap with single metric type"""
        metrics = [
            AnalyticsMetric("cpu", 45.0, datetime.now(), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("cpu", 50.0, datetime.now(), {}, AnalyticsLevel.BASIC)
        ]
        result = self.framework._create_correlation_heatmap(metrics)

        assert "error" in result
        assert result["error"] == "insufficient_metric_types"

    def test_create_correlation_heatmap_successful(self):
        """Test successful correlation heatmap creation"""
        metrics = [
            AnalyticsMetric("cpu", 45.0, datetime.now(), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("cpu", 50.0, datetime.now(), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("memory", 70.0, datetime.now(), {}, AnalyticsLevel.BASIC),
            AnalyticsMetric("memory", 80.0, datetime.now(), {}, AnalyticsLevel.BASIC)
        ]
        result = self.framework._create_correlation_heatmap(metrics)

        assert "correlations" in result
        assert "cpu" in result["correlations"]
        assert "memory" in result["correlations"]
        assert result["correlations"]["cpu"]["cpu"] == 1.0  # Self-correlation
        assert result["metric_count"] == 2


class TestAnomalyDetectionSystem:
    """Test suite for AnomalyDetectionSystem"""

    def setup_method(self):
        """Setup test fixtures"""
        self.system = AnomalyDetectionSystem()

    def create_sample_metrics(self, values, metric_name="test_metric"):
        """Create sample metrics with given values"""
        metrics = []
        for i, value in enumerate(values):
            timestamp = datetime.now() + timedelta(minutes=i)
            metrics.append(AnalyticsMetric(metric_name, value, timestamp, {}, AnalyticsLevel.BASIC))
        return metrics

    def test_detect_anomalies_empty_metrics(self):
        """Test anomaly detection with empty metrics"""
        result = self.system.detect_anomalies([])
        assert result == []

    def test_detect_anomalies_insufficient_data(self):
        """Test anomaly detection with insufficient data per metric"""
        metrics = self.create_sample_metrics([1.0, 2.0, 3.0])  # Less than 5 points
        result = self.system.detect_anomalies(metrics)
        assert result == []

    def test_detect_anomalies_nan_values(self):
        """Test anomaly detection with NaN values"""
        values = [1.0, 2.0, float('nan'), 4.0, 5.0, 6.0]
        metrics = self.create_sample_metrics(values)
        result = self.system.detect_anomalies(metrics)

        # Should handle NaN values without crashing
        assert isinstance(result, list)

    def test_detect_anomalies_normal_data(self):
        """Test anomaly detection with normal data (no anomalies)"""
        values = [50.0, 51.0, 49.0, 52.0, 48.0, 50.5, 49.5]  # Normal variation
        metrics = self.create_sample_metrics(values)
        result = self.system.detect_anomalies(metrics)

        # Should detect few or no anomalies in normal data
        assert len(result) <= 2  # Allow some false positives

    def test_detect_anomalies_with_outliers(self):
        """Test anomaly detection with clear outliers"""
        values = [50.0, 51.0, 49.0, 100.0, 48.0, 50.5, 1.0]  # 100 and 1 are outliers
        metrics = self.create_sample_metrics(values)
        result = self.system.detect_anomalies(metrics, sensitivity=0.1)

        assert len(result) > 0
        # Should detect anomalies for extreme values
        outlier_scores = [alert.anomaly_score for alert in result]
        assert any(score > 2.0 for score in outlier_scores)

    def test_detect_anomalies_constant_data(self):
        """Test anomaly detection with constant data"""
        values = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
        metrics = self.create_sample_metrics(values)
        result = self.system.detect_anomalies(metrics)

        # Constant data should not produce anomalies
        assert len(result) == 0

    def test_detect_anomalies_multiple_metrics(self):
        """Test anomaly detection with multiple metrics"""
        cpu_metrics = self.create_sample_metrics([50.0, 51.0, 100.0, 49.0, 50.0, 51.0], "cpu")
        memory_metrics = self.create_sample_metrics([70.0, 71.0, 69.0, 1.0, 70.0, 71.0], "memory")
        all_metrics = cpu_metrics + memory_metrics

        result = self.system.detect_anomalies(all_metrics)

        # Should detect anomalies in both metrics
        assert len(result) > 0
        affected_metrics = set()
        for alert in result:
            affected_metrics.update(alert.affected_metrics)

        assert len(affected_metrics) >= 1  # At least one metric should have anomalies

    def test_detect_metric_anomalies_insufficient_data(self):
        """Test metric-specific anomaly detection with insufficient data"""
        metrics = self.create_sample_metrics([1.0, 2.0])  # Less than 5 points
        result = self.system._detect_metric_anomalies("test", metrics, 0.1)
        assert result == []

    def test_detect_metric_anomalies_zero_std(self):
        """Test metric-specific anomaly detection with zero standard deviation"""
        metrics = self.create_sample_metrics([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        result = self.system._detect_metric_anomalies("test", metrics, 0.1)

        # Should not detect anomalies in constant data
        assert len(result) == 0

    def test_isolation_forest_detection_insufficient_data(self):
        """Test Isolation Forest with insufficient data"""
        metrics = self.create_sample_metrics([1.0, 2.0, 3.0])  # Less than 10 points
        result = self.system._isolation_forest_detection("test", metrics)
        assert result == []

    def test_isolation_forest_detection_sufficient_data(self):
        """Test Isolation Forest with sufficient data"""
        values = [50.0] * 10 + [100.0]  # 10 normal values + 1 outlier
        metrics = []
        for i, value in enumerate(values):
            timestamp = datetime.now() + timedelta(minutes=i)
            metadata = {"source": "test", "processing_time": 0.1}
            metric = AnalyticsMetric("test", value, timestamp, metadata, AnalyticsLevel.BASIC)
            metrics.append(metric)

        result = self.system._isolation_forest_detection("test", metrics)

        # May or may not detect anomalies depending on Isolation Forest behavior
        assert isinstance(result, list)
        for alert in result:
            assert isinstance(alert, AnomalyAlert)
            assert alert.affected_metrics == ["test"]

    def test_calculate_alert_severity(self):
        """Test alert severity calculation"""
        threshold = 2.0

        # Test different severity levels
        assert self.system._calculate_alert_severity(threshold * 3.1, threshold) == AlertSeverity.CRITICAL
        assert self.system._calculate_alert_severity(threshold * 2.1, threshold) == AlertSeverity.HIGH
        assert self.system._calculate_alert_severity(threshold * 1.6, threshold) == AlertSeverity.MEDIUM
        assert self.system._calculate_alert_severity(threshold * 1.0, threshold) == AlertSeverity.LOW

    def test_get_anomaly_recommendations(self):
        """Test anomaly recommendation generation"""
        # High anomaly score
        recommendations = self.system._get_anomaly_recommendations("cpu_usage", 6.0)
        assert "Immediate investigation required" in recommendations
        assert len(recommendations) <= 5

        # Medium anomaly score
        recommendations = self.system._get_anomaly_recommendations("memory_usage", 3.5)
        assert "Monitor closely for pattern continuation" in recommendations
        assert "Check for memory leaks" in recommendations  # Metric-specific

        # Low anomaly score
        recommendations = self.system._get_anomaly_recommendations("response_time", 2.0)
        assert "Log for trend analysis" in recommendations
        assert "Check network latency" in recommendations  # Metric-specific


class TestDataAnalyticsFramework:
    """Test suite for DataAnalyticsFramework integration"""

    def setup_method(self):
        """Setup test fixtures"""
        self.framework = DataAnalyticsFramework()

    def create_test_data(self):
        """Create comprehensive test data"""
        return {
            "metrics": [
                {
                    "name": "cpu_usage",
                    "value": 45.2,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"source": "system"},
                    "level": "basic"
                },
                {
                    "name": "memory_usage",
                    "value": 78.5,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"source": "system"},
                    "level": "intermediate"
                },
                {
                    "name": "cpu_usage",
                    "value": "invalid",  # Edge case: invalid value
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {},
                    "level": "basic"
                }
            ],
            "time_series": {
                "response_time": [
                    {
                        "timestamp": (datetime.now() - timedelta(minutes=i)).isoformat(),
                        "value": 100 + i * 2
                    }
                    for i in range(10)
                ]
            },
            "pattern_data": [
                {"cpu": 45.2, "memory": 78.5, "timestamp": datetime.now().isoformat()},
                {"cpu": 47.1, "memory": 82.3, "timestamp": datetime.now().isoformat()},
                {"cpu": 44.8, "memory": 76.9, "timestamp": datetime.now().isoformat()}
            ]
        }

    @pytest.mark.asyncio
    async def test_run_comprehensive_analysis_successful(self):
        """Test successful comprehensive analysis"""
        test_data = self.create_test_data()
        result = await self.framework.run_comprehensive_analysis(test_data)

        # Verify all main sections are present
        assert "timestamp" in result
        assert "analytics" in result
        assert "patterns" in result
        assert "predictions" in result
        assert "visualizations" in result
        assert "anomalies" in result
        assert "summary" in result

        # Verify summary contains expected fields
        summary = result["summary"]
        assert "total_metrics" in summary
        assert "total_patterns" in summary
        assert "total_predictions" in summary
        assert "total_anomalies" in summary
        assert "health_score" in summary
        assert isinstance(summary["health_score"], float)
        assert 0 <= summary["health_score"] <= 100

    @pytest.mark.asyncio
    async def test_run_comprehensive_analysis_empty_data(self):
        """Test comprehensive analysis with empty data"""
        result = await self.framework.run_comprehensive_analysis({})

        assert "timestamp" in result
        assert "analytics" in result
        assert "patterns" in result
        assert result["patterns"] == []  # No patterns from empty data

    @pytest.mark.asyncio
    async def test_run_comprehensive_analysis_error_handling(self):
        """Test comprehensive analysis error handling"""
        with patch.object(self.framework, '_extract_metrics', side_effect=Exception("Extraction failed")):
            result = await self.framework.run_comprehensive_analysis({"test": "data"})

            assert "error" in result
            assert "timestamp" in result

    def test_extract_metrics_valid_data(self):
        """Test metric extraction from valid data"""
        data = {
            "metrics": [
                {
                    "name": "cpu",
                    "value": 45.0,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": {"source": "test"},
                    "level": "basic"
                }
            ]
        }
        metrics = self.framework._extract_metrics(data)

        assert len(metrics) == 1
        assert metrics[0].name == "cpu"
        assert metrics[0].value == 45.0
        assert metrics[0].level == AnalyticsLevel.BASIC

    def test_extract_metrics_dict_structure(self):
        """Test metric extraction from dict structure"""
        data = {
            "cpu_usage": 45.0,
            "memory_usage": 78.5,
            "invalid_metric": float('nan')
        }
        metrics = self.framework._extract_metrics(data)

        assert len(metrics) == 2  # NaN value should be excluded
        metric_names = [m.name for m in metrics]
        assert "cpu_usage" in metric_names
        assert "memory_usage" in metric_names
        assert "invalid_metric" not in metric_names

    def test_convert_to_metric_valid_data(self):
        """Test conversion of valid data to metric"""
        item = {
            "name": "test_metric",
            "value": 42.0,
            "timestamp": datetime.now().isoformat(),
            "metadata": {"source": "test"},
            "level": "advanced"
        }
        metric = self.framework._convert_to_metric(item)

        assert metric is not None
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.level == AnalyticsLevel.ADVANCED

    def test_convert_to_metric_edge_cases(self):
        """Test metric conversion with edge cases"""
        # String value that can be converted to float
        item = {"name": "test", "value": "42.5"}
        metric = self.framework._convert_to_metric(item)
        assert metric is not None
        assert metric.value == 42.5

        # Invalid string value
        item = {"name": "test", "value": "invalid"}
        metric = self.framework._convert_to_metric(item)
        assert metric is None

        # NaN value
        item = {"name": "test", "value": float('nan')}
        metric = self.framework._convert_to_metric(item)
        assert metric is None

        # Invalid timestamp
        item = {"name": "test", "value": 42.0, "timestamp": "invalid_timestamp"}
        metric = self.framework._convert_to_metric(item)
        assert metric is not None  # Should use current time as fallback

        # Invalid level
        item = {"name": "test", "value": 42.0, "level": "invalid_level"}
        metric = self.framework._convert_to_metric(item)
        assert metric is not None
        assert metric.level == AnalyticsLevel.BASIC  # Should use default

    def test_prepare_pattern_data(self):
        """Test pattern data preparation"""
        data = {
            "pattern_data": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            "other_data": "ignored"
        }
        pattern_data = self.framework._prepare_pattern_data(data)

        assert len(pattern_data) == 2
        assert pattern_data[0] == {"a": 1, "b": 2}

    def test_prepare_pattern_data_from_metrics(self):
        """Test pattern data preparation from metrics"""
        data = {
            "metrics": [
                {"name": "cpu", "value": 45.0},
                {"name": "memory", "value": 78.5}
            ]
        }
        pattern_data = self.framework._prepare_pattern_data(data)

        assert len(pattern_data) == 2
        assert {"name": "cpu", "value": 45.0} in pattern_data

    def test_extract_time_series_data(self):
        """Test time series data extraction"""
        data = {
            "time_series": {
                "metric1": [
                    {"timestamp": datetime.now().isoformat(), "value": 1.0},
                    {"timestamp": datetime.now().isoformat(), "value": 2.0}
                ],
                "metric2": [
                    {"timestamp": "invalid_timestamp", "value": 3.0},  # Invalid timestamp
                    {"timestamp": datetime.now().isoformat(), "value": float('nan')}  # NaN value
                ]
            }
        }
        time_series = self.framework._extract_time_series_data(data)

        assert "metric1" in time_series
        assert len(time_series["metric1"]) == 2

        # metric2 should have no valid data points
        assert "metric2" not in time_series or len(time_series["metric2"]) == 0

    def test_generate_analysis_summary(self):
        """Test analysis summary generation"""
        results = {
            "analytics": {"cpu": {"count": 10}, "memory": {"count": 15}},
            "patterns": [
                {"pattern_type": "trend", "confidence": 0.8},
                {"pattern_type": "outliers", "confidence": 0.6}
            ],
            "predictions": {"cpu": {}, "memory": {}},
            "anomalies": [
                {"severity": "critical"},
                {"severity": "high"},
                {"severity": "medium"}
            ]
        }
        summary = self.framework._generate_analysis_summary(results)

        assert summary["total_metrics"] == 2
        assert summary["total_patterns"] == 2
        assert summary["total_predictions"] == 2
        assert summary["total_anomalies"] == 3
        assert summary["health_score"] < 100  # Should be reduced due to anomalies
        assert len(summary["key_insights"]) > 0
        assert len(summary["recommendations"]) > 0

    def test_generate_analysis_summary_edge_cases(self):
        """Test summary generation with edge cases"""
        # Empty results
        results = {}
        summary = self.framework._generate_analysis_summary(results)

        assert summary["total_metrics"] == 0
        assert summary["total_patterns"] == 0
        assert summary["health_score"] == 100.0

        # Results with errors
        results = {
            "analytics": {"error": "failed"},
            "patterns": None,
            "anomalies": None
        }
        summary = self.framework._generate_analysis_summary(results)

        assert isinstance(summary, dict)
        assert "total_metrics" in summary


@pytest.mark.asyncio
async def test_main_function():
    """Test the main function execution"""
    # This tests the framework integration and main execution path
    try:
        from data_analytics_framework import main
        await main()
        assert True  # If no exception is raised, test passes
    except Exception as e:
        pytest.fail(f"Main function failed: {e}")


class TestEdgeCasesAndErrorConditions:
    """Additional edge cases and error condition tests"""

    def test_extreme_values(self):
        """Test handling of extreme numerical values"""
        engine = AdvancedAnalyticsEngine()

        # Very large numbers
        large_data = [1e100, 2e100, 3e100]
        result = engine.calculate_descriptive_statistics(large_data, "large")
        assert "error" not in result
        assert result["count"] == 3

        # Very small numbers
        small_data = [1e-100, 2e-100, 3e-100]
        result = engine.calculate_descriptive_statistics(small_data, "small")
        assert "error" not in result
        assert result["count"] == 3

        # Mixed extreme values
        mixed_data = [-1e50, 0, 1e50]
        result = engine.calculate_descriptive_statistics(mixed_data, "mixed")
        assert "error" not in result
        assert result["mean"] == 0

    def test_infinite_values(self):
        """Test handling of infinite values"""
        engine = AdvancedAnalyticsEngine()

        # Positive infinity
        inf_data = [1.0, 2.0, float('inf'), 4.0, 5.0]
        result = engine.calculate_descriptive_statistics(inf_data, "inf")
        # Should handle infinity appropriately (implementation dependent)
        assert isinstance(result, dict)

        # Negative infinity
        neg_inf_data = [1.0, 2.0, float('-inf'), 4.0, 5.0]
        result = engine.calculate_descriptive_statistics(neg_inf_data, "neg_inf")
        assert isinstance(result, dict)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in names"""
        framework = DataAnalyticsFramework()

        data = {
            "metrics": [
                {
                    "name": "метрика_cpu",  # Cyrillic characters
                    "value": 45.0,
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "name": "métrique_mémoire",  # French accents
                    "value": 78.5,
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "name": "指标_内存",  # Chinese characters
                    "value": 65.0,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }

        metrics = framework._extract_metrics(data)
        assert len(metrics) == 3
        names = [m.name for m in metrics]
        assert "метрика_cpu" in names
        assert "métrique_mémoire" in names
        assert "指标_内存" in names

    def test_memory_pressure_simulation(self):
        """Test behavior under simulated memory pressure"""
        framework = DataAnalyticsFramework()

        # Create large dataset
        large_metrics = []
        for i in range(1000):
            large_metrics.append({
                "name": f"metric_{i % 10}",  # 10 different metrics
                "value": float(i),
                "timestamp": (datetime.now() + timedelta(seconds=i)).isoformat(),
                "metadata": {"large_data": "x" * 100},  # Add some bulk
                "level": "basic"
            })

        data = {"metrics": large_metrics}
        extracted_metrics = framework._extract_metrics(data)

        # Should handle large datasets without crashing
        assert len(extracted_metrics) <= len(large_metrics)
        assert all(isinstance(m, AnalyticsMetric) for m in extracted_metrics)

    @pytest.mark.asyncio
    async def test_concurrent_analysis(self):
        """Test concurrent analysis execution"""
        framework = DataAnalyticsFramework()

        test_data = {
            "metrics": [
                {"name": "cpu", "value": 45.0, "timestamp": datetime.now().isoformat()},
                {"name": "memory", "value": 78.0, "timestamp": datetime.now().isoformat()}
            ]
        }

        # Run multiple analyses concurrently
        tasks = [
            framework.run_comprehensive_analysis(test_data)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert "timestamp" in result
            assert "summary" in result


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])