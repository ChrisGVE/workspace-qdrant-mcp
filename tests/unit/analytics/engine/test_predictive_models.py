"""
Comprehensive unit tests for PredictiveModels with extensive edge case coverage.

Tests cover:
- Time series forecasting with various methods
- Capacity prediction and scaling recommendations
- Performance prediction and optimization
- Concept drift detection
- Edge cases: insufficient data, concept drift, model failures
- Forecasting accuracy and reliability assessment
- Error handling and boundary conditions
"""

import pytest
import math
import statistics
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.python.workspace_qdrant_mcp.analytics.engine.predictive_models import (
    PredictiveModels,
    ForecastMethod,
    PredictionHorizon,
    ForecastResult,
    CapacityPrediction,
    PerformancePrediction
)


class TestForecastResult:
    """Tests for ForecastResult data class."""

    def test_forecast_result_initialization(self):
        """Test ForecastResult initialization."""
        result = ForecastResult(
            predictions=[1.0, 2.0, 3.0],
            confidence_intervals=[(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)],
            method_used=ForecastMethod.LINEAR_REGRESSION,
            forecast_horizon=3,
            accuracy_metrics={'mae': 0.5},
            model_parameters={'slope': 1.0}
        )

        assert result.predictions == [1.0, 2.0, 3.0]
        assert result.confidence_intervals == [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
        assert result.method_used == ForecastMethod.LINEAR_REGRESSION
        assert result.forecast_horizon == 3
        assert result.accuracy_metrics == {'mae': 0.5}
        assert result.model_parameters == {'slope': 1.0}
        assert result.is_reliable is True

    def test_forecast_result_to_dict(self):
        """Test conversion to dictionary."""
        result = ForecastResult(
            predictions=[1.0, 2.0],
            confidence_intervals=[(0.5, 1.5), (1.5, 2.5)],
            method_used=ForecastMethod.EXPONENTIAL_SMOOTHING,
            forecast_horizon=2,
            accuracy_metrics={'mae': 0.3},
            model_parameters={'alpha': 0.3}
        )
        result_dict = result.to_dict()

        assert result_dict['predictions'] == [1.0, 2.0]
        assert result_dict['method_used'] == 'exponential_smoothing'
        assert result_dict['forecast_horizon'] == 2
        assert result_dict['is_reliable'] is True


class TestCapacityPrediction:
    """Tests for CapacityPrediction data class."""

    def test_capacity_prediction_initialization(self):
        """Test CapacityPrediction initialization."""
        result = CapacityPrediction(
            current_utilization=0.6,
            predicted_utilization=[0.65, 0.7, 0.75],
            capacity_exhaustion_date=datetime(2024, 1, 15),
            recommended_scaling_points=[(datetime(2024, 1, 10), 0.8)],
            growth_rate=0.05,
            confidence_level=0.85,
            resource_type="memory",
            time_horizon=30,
            scaling_recommendations=["Scale memory by 50%"]
        )

        assert result.current_utilization == 0.6
        assert result.predicted_utilization == [0.65, 0.7, 0.75]
        assert result.resource_type == "memory"
        assert result.time_horizon == 30
        assert len(result.scaling_recommendations) == 1

    def test_capacity_prediction_to_dict(self):
        """Test conversion to dictionary."""
        result = CapacityPrediction(
            current_utilization=0.5,
            predicted_utilization=[0.55],
            capacity_exhaustion_date=None,
            recommended_scaling_points=[],
            growth_rate=0.01,
            confidence_level=0.9,
            resource_type="cpu",
            time_horizon=15,
            scaling_recommendations=[]
        )
        result_dict = result.to_dict()

        assert result_dict['current_utilization'] == 0.5
        assert result_dict['resource_type'] == "cpu"
        assert result_dict['capacity_exhaustion_date'] is None


class TestPerformancePrediction:
    """Tests for PerformancePrediction data class."""

    def test_performance_prediction_initialization(self):
        """Test PerformancePrediction initialization."""
        result = PerformancePrediction(
            metric_name="response_time",
            current_performance=100.0,
            predicted_performance=[105.0, 110.0],
            performance_trend="degrading",
            bottleneck_predictions=["CPU bottleneck"],
            optimization_recommendations=["Scale CPU"],
            confidence_scores=[0.8, 0.75],
            time_horizon=14,
            baseline_comparison={"current_vs_baseline": 1.1}
        )

        assert result.metric_name == "response_time"
        assert result.current_performance == 100.0
        assert result.performance_trend == "degrading"
        assert len(result.bottleneck_predictions) == 1

    def test_performance_prediction_to_dict(self):
        """Test conversion to dictionary."""
        result = PerformancePrediction(
            metric_name="throughput",
            current_performance=1000.0,
            predicted_performance=[950.0],
            performance_trend="stable",
            bottleneck_predictions=[],
            optimization_recommendations=[],
            confidence_scores=[0.9],
            time_horizon=7,
            baseline_comparison={}
        )
        result_dict = result.to_dict()

        assert result_dict['metric_name'] == "throughput"
        assert result_dict['current_performance'] == 1000.0
        assert result_dict['performance_trend'] == "stable"


class TestPredictiveModels:
    """Comprehensive tests for PredictiveModels engine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = PredictiveModels(min_training_points=5, confidence_level=0.95)

    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.min_training_points == 5
        assert self.engine.confidence_level == 0.95
        assert len(self.engine._model_cache) == 0

    def test_initialization_custom_parameters(self):
        """Test engine initialization with custom parameters."""
        engine = PredictiveModels(min_training_points=10, confidence_level=0.99)
        assert engine.min_training_points == 10
        assert engine.confidence_level == 0.99

    # Time Series Forecasting Tests

    def test_forecast_time_series_linear_regression(self):
        """Test time series forecasting with linear regression."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = self.engine.forecast_time_series(
            data,
            forecast_horizon=3,
            method=ForecastMethod.LINEAR_REGRESSION
        )

        assert result.method_used == ForecastMethod.LINEAR_REGRESSION
        assert len(result.predictions) == 3
        assert result.is_reliable is True
        assert result.error_message is None

        # Predictions should follow linear trend
        assert result.predictions[0] > 10  # Should be greater than last data point
        assert result.predictions[1] > result.predictions[0]  # Should be increasing

        # Should have confidence intervals
        assert len(result.confidence_intervals) == 3
        assert all(lower <= upper for lower, upper in result.confidence_intervals)

    def test_forecast_time_series_exponential_smoothing(self):
        """Test time series forecasting with exponential smoothing."""
        data = [10, 12, 11, 13, 12, 14, 13, 15, 14, 16]
        result = self.engine.forecast_time_series(
            data,
            forecast_horizon=5,
            method=ForecastMethod.EXPONENTIAL_SMOOTHING
        )

        assert result.method_used == ForecastMethod.EXPONENTIAL_SMOOTHING
        assert len(result.predictions) == 5
        assert result.is_reliable is True
        assert 'alpha' in result.model_parameters
        assert result.mae is not None

    def test_forecast_time_series_moving_average(self):
        """Test time series forecasting with moving average."""
        data = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        result = self.engine.forecast_time_series(
            data,
            forecast_horizon=2,
            method=ForecastMethod.MOVING_AVERAGE
        )

        assert result.method_used == ForecastMethod.MOVING_AVERAGE
        assert len(result.predictions) == 2
        assert 'window_size' in result.model_parameters
        assert result.mae is not None

    def test_forecast_time_series_seasonal_naive(self):
        """Test time series forecasting with seasonal naive method."""
        # Create seasonal data
        data = [1, 2, 3, 4] * 5  # Repeating pattern
        result = self.engine.forecast_time_series(
            data,
            forecast_horizon=8,
            method=ForecastMethod.SEASONAL_NAIVE,
            seasonal_period=4
        )

        assert result.method_used == ForecastMethod.SEASONAL_NAIVE
        assert len(result.predictions) == 8
        assert 'seasonal_period' in result.model_parameters
        assert result.seasonal_components is not None

        # Should repeat the seasonal pattern
        assert result.predictions[0] == result.predictions[4]  # Same seasonal position

    def test_forecast_time_series_trend_projection(self):
        """Test time series forecasting with trend projection."""
        data = [i * 2 + 1 for i in range(10)]  # Strong linear trend
        result = self.engine.forecast_time_series(
            data,
            forecast_horizon=4,
            method=ForecastMethod.TREND_PROJECTION
        )

        assert result.method_used == ForecastMethod.TREND_PROJECTION
        assert len(result.predictions) == 4
        assert 'damping_factor' in result.model_parameters
        assert result.trend_component is not None

        # Should have increasing uncertainty with horizon
        conf_intervals = result.confidence_intervals
        first_interval_width = conf_intervals[0][1] - conf_intervals[0][0]
        last_interval_width = conf_intervals[-1][1] - conf_intervals[-1][0]
        assert last_interval_width >= first_interval_width

    def test_forecast_time_series_auto_method_selection(self):
        """Test automatic method selection."""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Linear trend
        result = self.engine.forecast_time_series(data, forecast_horizon=3)

        # Should automatically select appropriate method
        assert result.method_used in list(ForecastMethod)
        assert result.is_reliable is True

    def test_forecast_time_series_with_timestamps(self):
        """Test forecasting with timestamp data."""
        data = [1, 2, 3, 4, 5]
        timestamps = [datetime(2024, 1, i) for i in range(1, 6)]

        result = self.engine.forecast_time_series(
            data,
            forecast_horizon=2,
            timestamps=timestamps
        )

        assert result.prediction_dates is not None
        assert len(result.prediction_dates) == 2
        assert all(isinstance(date, datetime) for date in result.prediction_dates)

    def test_forecast_time_series_empty_data(self):
        """Test forecasting with empty dataset."""
        result = self.engine.forecast_time_series([], forecast_horizon=3)

        assert result.is_reliable is False
        assert result.error_message == "Empty dataset provided"
        assert len(result.predictions) == 0

    def test_forecast_time_series_insufficient_data(self):
        """Test forecasting with insufficient data points."""
        data = [1, 2, 3]  # Less than minimum required
        result = self.engine.forecast_time_series(data, forecast_horizon=2)

        assert result.is_reliable is False
        assert "Insufficient data points" in result.error_message

    def test_forecast_time_series_zero_forecast_horizon(self):
        """Test forecasting with zero forecast horizon."""
        data = [1, 2, 3, 4, 5, 6]
        result = self.engine.forecast_time_series(data, forecast_horizon=0)

        assert result.is_reliable is False
        assert "Forecast horizon must be positive" in result.error_message

    def test_forecast_time_series_negative_forecast_horizon(self):
        """Test forecasting with negative forecast horizon."""
        data = [1, 2, 3, 4, 5, 6]
        result = self.engine.forecast_time_series(data, forecast_horizon=-1)

        assert result.is_reliable is False
        assert "Forecast horizon must be positive" in result.error_message

    def test_forecast_time_series_with_nan_values(self):
        """Test forecasting handling NaN values."""
        data = [1, 2, float('nan'), 4, 5, 6, 7, 8, float('nan'), 10]
        result = self.engine.forecast_time_series(data, forecast_horizon=2)

        assert result.is_reliable is True  # Should handle NaN by filtering
        assert len(result.predictions) == 2

    def test_forecast_time_series_with_infinite_values(self):
        """Test forecasting handling infinite values."""
        data = [1, 2, float('inf'), 4, 5, 6, float('-inf'), 8, 9, 10]
        result = self.engine.forecast_time_series(data, forecast_horizon=2)

        assert result.is_reliable is True  # Should handle inf by filtering
        assert len(result.predictions) == 2

    def test_forecast_time_series_all_invalid_values(self):
        """Test forecasting with all invalid values."""
        data = [float('nan'), float('inf'), float('-inf')] * 5
        result = self.engine.forecast_time_series(data, forecast_horizon=2)

        assert result.is_reliable is False
        assert "Insufficient valid data points" in result.error_message

    def test_forecast_time_series_constant_data(self):
        """Test forecasting with constant data."""
        data = [5] * 10
        result = self.engine.forecast_time_series(data, forecast_horizon=3)

        assert result.is_reliable is True
        # Predictions should be close to constant value
        assert all(abs(pred - 5.0) < 0.1 for pred in result.predictions)

    def test_forecast_time_series_extreme_values(self):
        """Test forecasting with extreme values."""
        data = [1e10, 2e10, 3e10, 4e10, 5e10, 6e10]
        result = self.engine.forecast_time_series(data, forecast_horizon=2)

        assert result.is_reliable is True
        assert len(result.predictions) == 2

    def test_forecast_time_series_very_small_values(self):
        """Test forecasting with very small values."""
        data = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10]
        result = self.engine.forecast_time_series(data, forecast_horizon=2)

        assert result.is_reliable is True
        assert len(result.predictions) == 2

    # Capacity Prediction Tests

    def test_predict_capacity_needs_normal_growth(self):
        """Test capacity prediction with normal growth pattern."""
        # Gradual increase in utilization
        utilization_data = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
        result = self.engine.predict_capacity_needs(
            utilization_data,
            resource_type="memory",
            time_horizon_days=20
        )

        assert result.resource_type == "memory"
        assert result.time_horizon == 20
        assert result.current_utilization == 0.75
        assert len(result.predicted_utilization) == 20
        assert result.growth_rate > 0  # Should detect positive growth
        assert len(result.scaling_recommendations) > 0
        assert result.capacity_exhaustion_date is not None  # Should predict exhaustion

    def test_predict_capacity_needs_stable_utilization(self):
        """Test capacity prediction with stable utilization."""
        utilization_data = [0.5] * 10  # Stable utilization
        result = self.engine.predict_capacity_needs(
            utilization_data,
            resource_type="cpu",
            time_horizon_days=15
        )

        assert result.resource_type == "cpu"
        assert result.current_utilization == 0.5
        assert result.growth_rate == 0.0  # No growth
        assert result.capacity_exhaustion_date is None  # No exhaustion predicted
        assert result.risk_assessment == "low"

    def test_predict_capacity_needs_high_utilization(self):
        """Test capacity prediction with high current utilization."""
        utilization_data = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
        result = self.engine.predict_capacity_needs(
            utilization_data,
            resource_type="storage",
            capacity_threshold=0.95
        )

        assert result.current_utilization == 0.99
        assert result.risk_assessment == "high"  # Should be high risk
        assert len(result.recommended_scaling_points) > 0
        assert any("Scale" in rec or "Critical" in rec for rec in result.scaling_recommendations)

    def test_predict_capacity_needs_empty_data(self):
        """Test capacity prediction with empty data."""
        result = self.engine.predict_capacity_needs([], "memory")

        assert result.error_message == "Empty utilization data provided"
        assert result.current_utilization == 0.0

    def test_predict_capacity_needs_insufficient_data(self):
        """Test capacity prediction with insufficient data."""
        utilization_data = [0.5, 0.6]  # Too few points
        result = self.engine.predict_capacity_needs(utilization_data, "cpu")

        assert "Insufficient valid utilization data" in result.error_message

    def test_predict_capacity_needs_invalid_utilization_values(self):
        """Test capacity prediction with invalid utilization values."""
        # Values outside [0, 1] range - should be clamped
        utilization_data = [-0.1, 0.5, 1.2, 0.7, -0.5, 1.5, 0.8, 0.9, 2.0, 0.6]
        result = self.engine.predict_capacity_needs(utilization_data, "memory")

        # Should clamp values to valid range
        assert all(0.0 <= util <= 1.0 for util in result.predicted_utilization)
        assert 0.0 <= result.current_utilization <= 1.0

    def test_predict_capacity_needs_custom_threshold(self):
        """Test capacity prediction with custom capacity threshold."""
        utilization_data = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        result = self.engine.predict_capacity_needs(
            utilization_data,
            "memory",
            capacity_threshold=0.9  # Lower threshold
        )

        # Should trigger warnings earlier due to lower threshold
        assert result.capacity_exhaustion_date is not None
        assert len(result.recommended_scaling_points) > 0

    # Performance Prediction Tests

    def test_predict_performance_improving_trend(self):
        """Test performance prediction with improving trend."""
        # Response time decreasing (improving)
        performance_data = [200, 190, 180, 170, 160, 150, 140, 130, 120, 110]
        result = self.engine.predict_performance(
            performance_data,
            metric_name="response_time",
            baseline_value=200
        )

        assert result.metric_name == "response_time"
        assert result.current_performance == 110
        assert result.performance_trend == "improving"
        assert len(result.predicted_performance) == 30  # Default horizon
        assert result.baseline_comparison['current_vs_baseline'] < 1.0  # Better than baseline
        assert len(result.optimization_recommendations) > 0

    def test_predict_performance_degrading_trend(self):
        """Test performance prediction with degrading trend."""
        # Response time increasing (degrading)
        performance_data = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
        result = self.engine.predict_performance(
            performance_data,
            metric_name="response_time",
            baseline_value=100
        )

        assert result.performance_trend == "degrading"
        assert len(result.bottleneck_predictions) > 0
        assert any("degradation" in pred.lower() for pred in result.bottleneck_predictions)
        assert any("degradation" in rec.lower() for rec in result.optimization_recommendations)

    def test_predict_performance_stable_trend(self):
        """Test performance prediction with stable trend."""
        performance_data = [100, 102, 98, 101, 99, 100, 103, 97, 102, 98]
        result = self.engine.predict_performance(
            performance_data,
            metric_name="throughput"
        )

        assert result.performance_trend == "stable"
        assert result.current_performance == 98

    def test_predict_performance_with_baseline(self):
        """Test performance prediction with baseline comparison."""
        performance_data = [80, 85, 90, 95, 100, 105, 110, 115, 120, 125]
        result = self.engine.predict_performance(
            performance_data,
            metric_name="throughput",
            baseline_value=100
        )

        assert result.baseline_comparison is not None
        assert 'current_vs_baseline' in result.baseline_comparison
        assert 'predicted_vs_baseline' in result.baseline_comparison
        assert result.sla_compliance_prediction is not None

    def test_predict_performance_empty_data(self):
        """Test performance prediction with empty data."""
        result = self.engine.predict_performance([], "response_time")

        assert result.error_message == "Empty performance data provided"
        assert result.current_performance == 0.0

    def test_predict_performance_insufficient_data(self):
        """Test performance prediction with insufficient data."""
        performance_data = [100, 105]  # Too few points
        result = self.engine.predict_performance(performance_data, "latency")

        assert "Insufficient valid performance data" in result.error_message

    def test_predict_performance_negative_values_filtered(self):
        """Test performance prediction filtering negative values."""
        # Include some negative values which should be filtered out
        performance_data = [100, -50, 110, 120, -20, 130, 140, 150, 160, 170]
        result = self.engine.predict_performance(performance_data, "throughput")

        # Should work with valid positive values only
        assert result.current_performance == 170
        assert len(result.predicted_performance) > 0

    def test_predict_performance_high_variability(self):
        """Test performance prediction with high variability data."""
        performance_data = [50, 200, 75, 180, 60, 190, 70, 185, 65, 195]
        result = self.engine.predict_performance(performance_data, "response_time")

        # Should identify high variability
        assert any("variability" in rec.lower() for rec in result.optimization_recommendations)
        assert len(result.risk_factors) > 0

    def test_predict_performance_bottleneck_detection(self):
        """Test bottleneck prediction for different metric types."""
        # Test response time degradation
        response_time_data = [100, 110, 120, 130, 140, 150]
        result = self.engine.predict_performance(response_time_data, "response_time")

        if result.bottleneck_predictions:
            assert any("degradation" in pred.lower() or "increase" in pred.lower()
                      for pred in result.bottleneck_predictions)

        # Test throughput degradation
        throughput_data = [1000, 950, 900, 850, 800, 750]
        result = self.engine.predict_performance(throughput_data, "throughput")

        if result.bottleneck_predictions:
            assert any("bottleneck" in pred.lower() or "decrease" in pred.lower()
                      for pred in result.bottleneck_predictions)

    # Concept Drift Detection Tests

    def test_detect_concept_drift_mean_increase(self):
        """Test concept drift detection with mean increase."""
        historical_data = [10, 11, 9, 12, 10, 11, 9, 10, 12, 11]  # Mean ~10.5
        recent_data = [15, 16, 14, 17, 15, 16, 14, 15, 17, 16]     # Mean ~15.5

        result = self.engine.detect_concept_drift(recent_data, historical_data)

        assert result['drift_detected'] is True
        assert result['drift_type'] == 'mean_increase'
        assert result['drift_magnitude'] > 0
        assert result['confidence'] > 0
        assert len(result['recommendations']) > 0

    def test_detect_concept_drift_mean_decrease(self):
        """Test concept drift detection with mean decrease."""
        historical_data = [20, 21, 19, 22, 20, 21, 19, 20, 22, 21]  # Mean ~20.5
        recent_data = [15, 16, 14, 17, 15, 16, 14, 15, 17, 16]      # Mean ~15.5

        result = self.engine.detect_concept_drift(recent_data, historical_data)

        assert result['drift_detected'] is True
        assert result['drift_type'] == 'mean_decrease'
        assert len(result['recommendations']) > 0

    def test_detect_concept_drift_variance_change(self):
        """Test concept drift detection with variance change."""
        historical_data = [10, 10.1, 9.9, 10.05, 9.95] * 2  # Low variance
        recent_data = [10, 5, 15, 8, 12, 7, 13, 6, 14, 9]    # High variance

        result = self.engine.detect_concept_drift(recent_data, historical_data)

        # May detect variance increase
        assert result['drift_type'] in ['variance_increase', 'mean_decrease', 'none']

    def test_detect_concept_drift_no_drift(self):
        """Test concept drift detection with no significant drift."""
        historical_data = [10, 11, 9, 12, 10, 11, 9, 10, 12, 11]
        recent_data = [10, 11, 9, 12, 10, 11, 9, 10, 12, 11]  # Same distribution

        result = self.engine.detect_concept_drift(recent_data, historical_data)

        assert result['drift_detected'] is False
        assert result['drift_type'] == 'none'
        assert result['drift_magnitude'] == 0.0

    def test_detect_concept_drift_empty_data(self):
        """Test concept drift detection with empty data."""
        result = self.engine.detect_concept_drift([], [1, 2, 3])
        assert "Empty data provided" in result['error_message']

        result = self.engine.detect_concept_drift([1, 2, 3], [])
        assert "Empty data provided" in result['error_message']

    def test_detect_concept_drift_insufficient_data(self):
        """Test concept drift detection with insufficient data."""
        result = self.engine.detect_concept_drift([1, 2], [3, 4])
        assert "Insufficient data" in result['error_message']

    def test_detect_concept_drift_with_invalid_values(self):
        """Test concept drift detection handling invalid values."""
        historical_data = [10, 11, float('nan'), 12, 10, float('inf'), 9, 10, 12, 11]
        recent_data = [15, 16, 14, float('-inf'), 15, 16, float('nan'), 15, 17, 16]

        result = self.engine.detect_concept_drift(recent_data, historical_data)

        # Should handle invalid values by filtering them out
        assert 'error_message' not in result or result['error_message'] is None

    def test_detect_concept_drift_custom_significance(self):
        """Test concept drift detection with custom significance level."""
        historical_data = [10] * 10
        recent_data = [11] * 10  # Small difference

        # More strict significance level
        result = self.engine.detect_concept_drift(
            recent_data, historical_data, significance_level=0.01
        )

        # May or may not detect drift depending on effect size

    # Method Selection Tests

    def test_method_selection_exponential_data(self):
        """Test automatic method selection for exponential data."""
        # Create exponential data
        data = [math.exp(i * 0.2) for i in range(10)]

        # Mock the method selection to test logic
        method = self.engine._select_best_method(data, None)

        # Should select exponential smoothing for exponential data
        assert method in [ForecastMethod.EXPONENTIAL_SMOOTHING, ForecastMethod.LINEAR_REGRESSION]

    def test_method_selection_seasonal_data(self):
        """Test automatic method selection for seasonal data."""
        data = [math.sin(i * math.pi / 6) + i * 0.1 for i in range(24)]

        method = self.engine._select_best_method(data, seasonal_period=12)

        assert method == ForecastMethod.SEASONAL_NAIVE

    def test_method_selection_linear_trend(self):
        """Test automatic method selection for linear trend."""
        data = [i * 2 + 3 for i in range(10)]  # Perfect linear trend

        method = self.engine._select_best_method(data, None)

        assert method == ForecastMethod.LINEAR_REGRESSION

    # Caching Tests

    def test_forecast_caching(self):
        """Test caching functionality for forecasts."""
        data = [1, 2, 3, 4, 5, 6]
        cache_key = "test_forecast_cache"

        # First call should calculate and cache
        result1 = self.engine.forecast_time_series(data, 2, cache_key=cache_key)

        # Second call should use cache
        result2 = self.engine.forecast_time_series(data, 2, cache_key=cache_key)

        assert result1.predictions == result2.predictions
        assert cache_key in self.engine._model_cache

    def test_caching_without_key(self):
        """Test that results are not cached without cache key."""
        data = [1, 2, 3, 4, 5, 6]
        result = self.engine.forecast_time_series(data, 2)

        assert len(self.engine._model_cache) == 0

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        data = [1, 2, 3, 4, 5, 6]
        cache_key = "test_clear"

        # Cache a result
        self.engine.forecast_time_series(data, 2, cache_key=cache_key)
        assert len(self.engine._model_cache) > 0

        # Clear cache
        self.engine.clear_cache()
        assert len(self.engine._model_cache) == 0

    # Error Handling Tests

    def test_error_handling_in_forecasting(self):
        """Test error handling in forecasting methods."""
        # Test with data that might cause numerical issues
        data = [1e-100] * 10  # Extremely small values
        result = self.engine.forecast_time_series(data, 3)

        # Should handle gracefully without crashing
        assert result.forecast_horizon == 3

    def test_error_handling_unsupported_method(self):
        """Test error handling with unsupported forecasting method."""
        data = [1, 2, 3, 4, 5, 6]

        # Mock an unsupported method
        with patch.object(ForecastMethod, 'LINEAR_REGRESSION', 'unsupported_method'):
            result = self.engine.forecast_time_series(
                data, 2, method=ForecastMethod.LINEAR_REGRESSION
            )
            # Should handle error gracefully

    def test_error_handling_in_capacity_prediction(self):
        """Test error handling in capacity prediction."""
        # Mock an exception during forecasting
        with patch.object(self.engine, 'forecast_time_series') as mock_forecast:
            mock_forecast.return_value = ForecastResult(
                predictions=[], confidence_intervals=[], method_used=ForecastMethod.LINEAR_REGRESSION,
                forecast_horizon=0, accuracy_metrics={}, model_parameters={},
                is_reliable=False, error_message="Mock error"
            )

            utilization_data = [0.5, 0.6, 0.7]
            result = self.engine.predict_capacity_needs(utilization_data, "memory")

            assert "Unable to generate reliable capacity forecast" in result.error_message

    def test_error_handling_in_performance_prediction(self):
        """Test error handling in performance prediction."""
        # Mock an exception during forecasting
        with patch.object(self.engine, 'forecast_time_series') as mock_forecast:
            mock_forecast.return_value = ForecastResult(
                predictions=[], confidence_intervals=[], method_used=ForecastMethod.LINEAR_REGRESSION,
                forecast_horizon=0, accuracy_metrics={}, model_parameters={},
                is_reliable=False, error_message="Mock error"
            )

            performance_data = [100, 110, 120]
            result = self.engine.predict_performance(performance_data, "latency")

            assert "Unable to generate reliable performance forecast" in result.error_message

    # Reliability Assessment Tests

    def test_forecast_reliability_assessment(self):
        """Test forecast reliability assessment."""
        # High quality data should have high reliability
        data = [i for i in range(50)]  # 50 data points, perfect trend
        result = self.engine.forecast_time_series(data, 5)

        assert result.reliability_score > 0.7  # Should be highly reliable

        # Low quality data should have lower reliability
        data_low_quality = [1, 2, 3]  # Few data points
        result_low = self.engine.forecast_time_series(data_low_quality, 10)  # Long forecast

        if result_low.is_reliable:
            assert result_low.reliability_score < result.reliability_score

    def test_accuracy_metrics_calculation(self):
        """Test accuracy metrics calculation."""
        # Create predictable data for testing
        data = list(range(20))  # Linear trend
        result = self.engine.forecast_time_series(
            data, 3, method=ForecastMethod.LINEAR_REGRESSION
        )

        if result.accuracy_metrics:
            assert 'mae' in result.accuracy_metrics
            assert 'mape' in result.accuracy_metrics
            assert 'rmse' in result.accuracy_metrics
            assert all(metric >= 0 for metric in result.accuracy_metrics.values())

    # Integration Tests

    def test_comprehensive_forecasting_workflow(self):
        """Test comprehensive forecasting workflow."""
        # Create complex data with trend and seasonality
        data = []
        for i in range(100):
            trend = i * 0.1
            seasonal = 5 * math.sin(i * 2 * math.pi / 12)
            noise = (i % 3) * 0.5  # Some noise
            data.append(10 + trend + seasonal + noise)

        # Test various forecasting approaches
        linear_result = self.engine.forecast_time_series(
            data, 12, method=ForecastMethod.LINEAR_REGRESSION
        )

        seasonal_result = self.engine.forecast_time_series(
            data, 12, method=ForecastMethod.SEASONAL_NAIVE, seasonal_period=12
        )

        auto_result = self.engine.forecast_time_series(data, 12)  # Auto-select method

        # All should complete successfully
        assert linear_result.is_reliable
        assert seasonal_result.is_reliable
        assert auto_result.is_reliable

    def test_end_to_end_capacity_planning(self):
        """Test end-to-end capacity planning workflow."""
        # Create realistic utilization data
        base_utilization = 0.4
        utilization_data = []

        for i in range(30):
            # Gradual growth with some variation
            growth = i * 0.01
            variation = 0.05 * math.sin(i * 0.5)
            utilization = base_utilization + growth + variation
            utilization_data.append(max(0, min(1, utilization)))

        # Predict capacity needs
        capacity_result = self.engine.predict_capacity_needs(
            utilization_data, "memory", time_horizon_days=60
        )

        # Should provide comprehensive capacity analysis
        assert capacity_result.resource_type == "memory"
        assert len(capacity_result.predicted_utilization) == 60
        assert capacity_result.growth_rate > 0  # Should detect growth
        assert len(capacity_result.scaling_recommendations) > 0
        assert capacity_result.risk_assessment in ["low", "medium", "high"]

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        # Create large dataset
        data = [math.sin(i * 0.01) + i * 0.001 for i in range(1000)]

        # Should complete without errors
        result = self.engine.forecast_time_series(data, 50)
        assert len(result.predictions) == 50

    def test_edge_case_combinations(self):
        """Test various edge case combinations."""
        # Very volatile data
        volatile_data = [10 + 50 * math.sin(i) for i in range(20)]
        result = self.engine.forecast_time_series(volatile_data, 5)
        assert len(result.predictions) == 5

        # Data with extreme outliers
        outlier_data = [10, 11, 12, 1000, 13, 14, 15, -500, 16, 17]
        result = self.engine.forecast_time_series(outlier_data, 3)
        assert len(result.predictions) == 3

        # Mixed positive and negative values
        mixed_data = [i if i % 2 == 0 else -i for i in range(1, 21)]
        result = self.engine.predict_performance(mixed_data, "mixed_metric")
        assert result.metric_name == "mixed_metric"